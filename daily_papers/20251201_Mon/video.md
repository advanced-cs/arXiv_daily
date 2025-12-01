# 计算机视觉 cs.CV

- **最新发布 242 篇**

- **更新 147 篇**

## 最新发布

#### [new 001] TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video
- **分类: cs.CV**

- **简介: 该论文提出TAPVid-360任务，旨在从窄视场视频中追踪全景中任意点的3D方向，解决现有方法无法跟踪视域外点的问题。通过360°视频生成带方向标注的窄视场数据集，构建新基准TAPVid360-10k。提出基于CoTracker v3的基线模型，实现跨视域的点追踪，推动全景、非视角依赖的视觉理解。**

- **链接: [https://arxiv.org/pdf/2511.21946v1](https://arxiv.org/pdf/2511.21946v1)**

> **作者:** Finlay G. C. Hudson; James A. D. Gardner; William A. P. Smith
>
> **摘要:** Humans excel at constructing panoramic mental models of their surroundings, maintaining object permanence and inferring scene structure beyond visible regions. In contrast, current artificial vision systems struggle with persistent, panoramic understanding, often processing scenes egocentrically on a frame-by-frame basis. This limitation is pronounced in the Track Any Point (TAP) task, where existing methods fail to track 2D points outside the field of view. To address this, we introduce TAPVid-360, a novel task that requires predicting the 3D direction to queried scene points across a video sequence, even when far outside the narrow field of view of the observed video. This task fosters learning allocentric scene representations without needing dynamic 4D ground truth scene models for training. Instead, we exploit 360 videos as a source of supervision, resampling them into narrow field-of-view perspectives while computing ground truth directions by tracking points across the full panorama using a 2D pipeline. We introduce a new dataset and benchmark, TAPVid360-10k comprising 10k perspective videos with ground truth directional point tracking. Our baseline adapts CoTracker v3 to predict per-point rotations for direction updates, outperforming existing TAP and TAPVid 3D methods.
>
---
#### [new 002] HMR3D: Hierarchical Multimodal Representation for 3D Scene Understanding with Large Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文针对3D场景理解任务，解决现有视觉语言模型（VLM）在3D数据上对齐不充分、性能受限的问题。提出HMR3D框架，通过多视角图像与文本描述（含3D坐标参考）在输入层显式对齐VLM，构建分层特征表示，实现局部到全局的场景推理，在多个3D问答基准上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2511.22961v1](https://arxiv.org/pdf/2511.22961v1)**

> **作者:** Chen Li; Eric Peh; Basura Fernando
>
> **摘要:** Recent advances in large vision-language models (VLMs) have shown significant promise for 3D scene understanding. Existing VLM-based approaches typically align 3D scene features with the VLM's embedding space. However, this implicit alignment often yields suboptimal performance due to the scarcity of 3D data and the inherent complexity of spatial relationships in 3D environments. To address these limitations, we propose a novel hierarchical multimodal representation for 3D scene reasoning that explicitly aligns with VLMs at the input space by leveraging both multi-view images and text descriptions. The text descriptions capture spatial relationships by referencing the 3D coordinates of detected objects, while the multi-view images include a top-down perspective and four directional views (forward, left, right, and backward), ensuring comprehensive scene coverage. Additionally, we introduce a hierarchical feature representation that aggregates patch-level image features into view-level and scene-level representations, enabling the model to reason over both local and global scene context. Experimental results on both situated 3D Q&A and general 3D Q&A benchmarks demonstrate the effectiveness of our approach.
>
---
#### [new 003] UniArt: Unified 3D Representation for Generating 3D Articulated Objects with Open-Set Articulation
- **分类: cs.CV**

- **简介: 该论文提出UniArt，一种基于扩散模型的统一框架，用于从单张图像端到端生成带开集关节的完整3D可动物体。针对人工构建3D可动物体成本高、难扩展的问题，提出联合编码几何、纹理、分割与运动参数的统一表征，并通过可逆关节-体素嵌入实现结构与运动协同学习，支持新关节类型泛化。**

- **链接: [https://arxiv.org/pdf/2511.21887v1](https://arxiv.org/pdf/2511.21887v1)**

> **作者:** Bu Jin; Weize Li; Songen Gu; Yupeng Zheng; Yuhang Zheng; Zhengyi Zhou; Yao Yao
>
> **摘要:** Articulated 3D objects play a vital role in realistic simulation and embodied robotics, yet manually constructing such assets remains costly and difficult to scale. In this paper, we present UniArt, a diffusion-based framework that directly synthesizes fully articulated 3D objects from a single image in an end-to-end manner. Unlike prior multi-stage techniques, UniArt establishes a unified latent representation that jointly encodes geometry, texture, part segmentation, and kinematic parameters. We introduce a reversible joint-to-voxel embedding, which spatially aligns articulation features with volumetric geometry, enabling the model to learn coherent motion behaviors alongside structural formation. Furthermore, we formulate articulation type prediction as an open-set problem, removing the need for fixed joint semantics and allowing generalization to novel joint categories and unseen object types. Experiments on the PartNet-Mobility benchmark demonstrate that UniArt achieves state-of-the-art mesh quality and articulation accuracy.
>
---
#### [new 004] Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model
- **分类: cs.CV**

- **简介: 该论文提出Hunyuan-GameCraft-2，一种指令驱动的交互式游戏世界生成模型。针对现有方法依赖固定动作、标注成本高、交互能力弱的问题，通过自然语言指令实现对游戏视频内容的灵活控制，构建了因果对齐的交互数据集与评估基准，实现了对相机、角色和环境的细粒度语义控制。**

- **链接: [https://arxiv.org/pdf/2511.23429v1](https://arxiv.org/pdf/2511.23429v1)**

> **作者:** Junshu Tang; Jiacheng Liu; Jiaqi Li; Longhuang Wu; Haoyu Yang; Penghao Zhao; Siruis Gong; Xiang Yuan; Shuai Shao; Qinglin Lu
>
> **备注:** Technical Report, Project page:https://hunyuan-gamecraft-2.github.io/
>
> **摘要:** Recent advances in generative world models have enabled remarkable progress in creating open-ended game environments, evolving from static scene synthesis toward dynamic, interactive simulation. However, current approaches remain limited by rigid action schemas and high annotation costs, restricting their ability to model diverse in-game interactions and player-driven dynamics. To address these challenges, we introduce Hunyuan-GameCraft-2, a new paradigm of instruction-driven interaction for generative game world modeling. Instead of relying on fixed keyboard inputs, our model allows users to control game video contents through natural language prompts, keyboard, or mouse signals, enabling flexible and semantically rich interaction within generated worlds. We formally defined the concept of interactive video data and developed an automated process to transform large-scale, unstructured text-video pairs into causally aligned interactive datasets. Built upon a 14B image-to-video Mixture-of-Experts(MoE) foundation model, our model incorporates a text-driven interaction injection mechanism for fine-grained control over camera motion, character behavior, and environment dynamics. We introduce an interaction-focused benchmark, InterBench, to evaluate interaction performance comprehensively. Extensive experiments demonstrate that our model generates temporally coherent and causally grounded interactive game videos that faithfully respond to diverse and free-form user instructions such as "open the door", "draw a torch", or "trigger an explosion".
>
---
#### [new 005] MrGS: Multi-modal Radiance Fields with 3D Gaussian Splatting for RGB-Thermal Novel View Synthesis
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MrGS，一种基于3D高斯点云的多模态辐射场方法，用于RGB-热红外图像的新型视图合成。针对现有方法忽略热传导与朗伯反射特性的问题，通过正交特征提取与物理定律建模，实现高保真多模态场景重建，并减少高斯点数量。**

- **链接: [https://arxiv.org/pdf/2511.22997v1](https://arxiv.org/pdf/2511.22997v1)**

> **作者:** Minseong Kweon; Janghyun Kim; Ukcheol Shin; Jinsun Park
>
> **备注:** Accepted at Thermal Infrared in Robotics (TIRO) Workshop, ICRA 2025 (Best Poster Award)
>
> **摘要:** Recent advances in Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) have achieved considerable performance in RGB scene reconstruction. However, multi-modal rendering that incorporates thermal infrared imagery remains largely underexplored. Existing approaches tend to neglect distinctive thermal characteristics, such as heat conduction and the Lambertian property. In this study, we introduce MrGS, a multi-modal radiance field based on 3DGS that simultaneously reconstructs both RGB and thermal 3D scenes. Specifically, MrGS derives RGB- and thermal-related information from a single appearance feature through orthogonal feature extraction and employs view-dependent or view-independent embedding strategies depending on the degree of Lambertian reflectance exhibited by each modality. Furthermore, we leverage two physics-based principles to effectively model thermal-domain phenomena. First, we integrate Fourier's law of heat conduction prior to alpha blending to model intensity interpolation caused by thermal conduction between neighboring Gaussians. Second, we apply the Stefan-Boltzmann law and the inverse-square law to formulate a depth-aware thermal radiation map that imposes additional geometric constraints on thermal rendering. Experimental results demonstrate that the proposed MrGS achieves high-fidelity RGB-T scene reconstruction while reducing the number of Gaussians.
>
---
#### [new 006] WalkCLIP: Multimodal Learning for Urban Walkability Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出WalkCLIP，一种多模态城市步行性预测框架。针对传统方法依赖单一数据源、无法全面反映步行环境的问题，融合卫星影像、街景图像与人口动态数据，通过视觉-语言表示学习与空间聚合，提升预测精度与空间一致性，实现更全面的步行性评估。**

- **链接: [https://arxiv.org/pdf/2511.21947v1](https://arxiv.org/pdf/2511.21947v1)**

> **作者:** Shilong Xiang; JangHyeon Lee; Min Namgung; Yao-Yi Chiang
>
> **摘要:** Urban walkability is a cornerstone of public health, sustainability, and quality of life. Traditional walkability assessments rely on surveys and field audits, which are costly and difficult to scale. Recent studies have used satellite imagery, street view imagery, or population indicators to estimate walkability, but these single-source approaches capture only one dimension of the walking environment. Satellite data describe the built environment from above, but overlook the pedestrian perspective. Street view imagery captures conditions at the ground level, but lacks broader spatial context. Population dynamics reveal patterns of human activity but not the visual form of the environment. We introduce WalkCLIP, a multimodal framework that integrates these complementary viewpoints to predict urban walkability. WalkCLIP learns walkability-aware vision-language representations from GPT-4o generated image captions, refines these representations with a spatial aggregation module that incorporates neighborhood context, and fuses the resulting features with representations from a population dynamics foundation model. Evaluated at 4,660 locations throughout Minneapolis-Saint Paul, WalkCLIP outperforms unimodal and multimodal baselines in both predictive accuracy and spatial alignment. These results show that the integration of visual and behavioral signals yields reliable predictions of the walking environment.
>
---
#### [new 007] DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对指针式仪表读数识别任务，解决现有方法在反光、遮挡、视角变化等挑战下性能脆弱的问题。提出大规模基准数据集RPM-10K和基于物理关系注入的视觉语言模型MRLM，通过显式建模指针与刻度的几何因果关系，实现精准读数。**

- **链接: [https://arxiv.org/pdf/2511.21982v1](https://arxiv.org/pdf/2511.21982v1)**

> **作者:** Futian Wang; Chaoliu Weng; Xiao Wang; Zhen Chen; Zhicheng Zhao; Jin Tang
>
> **摘要:** The precise reading recognition of pointer meters plays a key role in smart power systems, but existing approaches remain fragile due to challenges like reflections, occlusions, dynamic viewing angles, and overly between thin pointers and scale markings. Up to now, this area still lacks large-scale datasets to support the development of robust algorithms. To address these challenges, this paper first presents a new large-scale benchmark dataset for dial reading, termed RPM-10K, which contains 10730 meter images that fully reflect the aforementioned key challenges. Built upon the dataset, we propose a novel vision-language model for pointer meter reading recognition, termed MRLM, based on physical relation injection. Instead of exhaustively learning image-level correlations, MRLM explicitly encodes the geometric and causal relationships between the pointer and the scale, aligning perception with physical reasoning in the spirit of world-model perspectives. Through cross-attentional fusion and adaptive expert selection, the model learns to interpret dial configurations and generate precise numeric readings. Extensive experiments fully validated the effectiveness of our proposed framework on the newly proposed benchmark dataset. Both the dataset and source code will be released on https://github.com/Event-AHU/DialBench
>
---
#### [new 008] HarmoCLIP: Harmonizing Global and Regional Representations in Contrastive Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对对比视觉-语言模型CLIP在全局与局部语义表示间的权衡问题，提出HarmoCLIP框架。通过引入细粒度语义监督与区域-语言对齐策略，实现全局一致性与局部感知的协同优化，显著提升图像检索与边界框分类性能。**

- **链接: [https://arxiv.org/pdf/2511.22594v1](https://arxiv.org/pdf/2511.22594v1)**

> **作者:** Haoxi Zeng; Haoxuan Li; Yi Bin; Pengpeng Zeng; Xing Xu; Yang Yang; Heng Tao Shen
>
> **备注:** 13 pages, 7 figures, 6 tables
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) has demonstrated remarkable generalization ability and strong performance across a wide range of vision-language tasks. However, due to the lack of region-level supervision, CLIP exhibits limited fine-grained semantic understanding. Although several methods attempt to mitigate this issue, they unintentionally disrupt the global alignment, resulting in a persistent trade-off where improving local perception simultaneously degrades global coherence. In this paper, we propose HarmoCLIP, a novel framework designed to harmonize global and region representations within CLIP. We first identify that the absence of direct alignment between local textual and visual semantics is the fundamental cause of the trade-off. To address this, HarmoCLIP introduces an explicit fine-grained semantic supervision term that directly aligns textual segments with their corresponding visual regions, effectively bridging the image region space and the textual space. To further strengthen the representation capability at the local level, our method introduces a novel Region-Language Alignment supervision strategy that promotes fine-grained semantic learning without compromising global semantic consistency. Extensive experiments demonstrate that HarmoCLIP achieves state-of-the-art (improvement up to 69.78%) performance on the global task of retrieval and yields a substantial 3.2% improvement in Top-1 accuracy on the region task of bounding-box classification, consistently outperforming prior approaches while providing a balanced, efficient, and plug-and-play solution to the global-local trade-off in CLIP. Code is available at https://github.com/Erosist/HarmoCLIP.
>
---
#### [new 009] Benchmarking machine learning models for multi-class state recognition in double duantum dot data
- **分类: cs.CV; cond-mat.mes-hall; cs.LG**

- **简介: 该论文针对量子点器件状态识别任务，研究多类状态在双量子点电荷稳定性图中的自动识别。通过对比四种机器学习模型，在不同数据量和归一化方式下评估性能，发现CNN结合最小-最大归一化在实验数据上表现最佳，具备高精度与低参数量优势，为量子处理器自动化校准提供有效方案。**

- **链接: [https://arxiv.org/pdf/2511.22451v1](https://arxiv.org/pdf/2511.22451v1)**

> **作者:** Valeria Díaz Moreno; Ryan P Khalili; Daniel Schug; Patrick J. Walsh; Justyna P. Zwolak
>
> **备注:** 12 pages, 4 figures, 2 tables
>
> **摘要:** Semiconductor quantum dots (QDs) are a leading platform for scalable quantum processors. However, scaling to large arrays requires reliable, automated tuning strategies for devices' bootstrapping, calibration, and operation, with many tuning aspects depending on accurately identifying QD device states from charge-stability diagrams (CSDs). In this work, we present a comprehensive benchmarking study of four modern machine learning (ML) architectures for multi-class state recognition in double-QD CSDs. We evaluate their performance across different data budgets and normalization schemes using both synthetic and experimental data. We find that the more resource-intensive models -- U-Nets and visual transformers (ViTs) -- achieve the highest MSE score (defined as $1-\mathrm{MSE}$) on synthetic data (over $0.98$) but fail to generalize to experimental data. MDNs are the most computationally efficient and exhibit highly stable training, but with substantially lower peak performance. CNNs offer the most favorable trade-off on experimental CSDs, achieving strong accuracy with two orders of magnitude fewer parameters than the U-Nets and ViTs. Normalization plays a nontrivial role: min-max scaling generally yields higher MSE scores but less stable convergence, whereas z-score normalization produces more predictable training dynamics but at reduced accuracy for most models. Overall, our study shows that CNNs with min-max normalization are a practical approach for QD CSDs.
>
---
#### [new 010] PowerCLIP: Powerset Alignment for Contrastive Pre-Training
- **分类: cs.CV**

- **简介: 该论文提出PowerCLIP，针对视觉-语言对比预训练中难以捕捉跨区域组合语义的问题，通过幂集对齐实现图像区域与文本短语的细粒度对齐。为降低指数级计算复杂度，引入非线性聚合器，将复杂度从O(2^M)降至O(M)，在零样本分类与检索任务上表现优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.23170v1](https://arxiv.org/pdf/2511.23170v1)**

> **作者:** Masaki Kawamura; Nakamasa Inoue; Rintaro Yanagi; Hirokatsu Kataoka; Rio Yokota
>
> **备注:** Submitted to CVPR 2026
>
> **摘要:** Contrastive vision-language pre-training frameworks such as CLIP have demonstrated impressive zero-shot performance across a range of vision-language tasks. Recent studies have shown that aligning individual text tokens with specific image patches or regions enhances fine-grained compositional understanding. However, it remains challenging to capture compositional semantics that span multiple image regions. To address this limitation, we propose PowerCLIP, a novel contrastive pre-training framework enhanced by powerset alignment, which exhaustively optimizes region-to-phrase alignments by minimizing the loss defined between powersets of image regions and textual parse trees. Since the naive powerset construction incurs exponential computational cost due to the combinatorial explosion in the number of region subsets, we introduce efficient non-linear aggregators (NLAs) that reduce complexity from O(2^M) to O(M) with respect to the number of regions M, while approximating the exact loss value with arbitrary precision. Our extensive experiments demonstrate that PowerCLIP outperforms state-of-the-art methods in zero-shot classification and retrieval tasks, underscoring the compositionality and robustness of our approach. Our code will be made publicly available.
>
---
#### [new 011] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation
- **分类: cs.CV**

- **简介: 该论文针对单图密集足部接触估计任务，解决鞋款多样性和地面纹理单调导致的特征泛化难题。提出FECO框架，通过鞋款风格对抗训练实现风格不变特征学习，并引入基于空间上下文的地面特征提取器，有效利用地面信息，提升接触估计精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22184v1](https://arxiv.org/pdf/2511.22184v1)**

> **作者:** Daniel Sungho Jung; Kyoung Mu Lee
>
> **备注:** Project page: https://feco-release.github.io/
>
> **摘要:** Foot contact plays a critical role in human interaction with the world, and thus exploring foot contact can advance our understanding of human movement and physical interaction. Despite its importance, existing methods often approximate foot contact using a zero-velocity constraint and focus on joint-level contact, failing to capture the detailed interaction between the foot and the world. Dense estimation of foot contact is crucial for accurately modeling this interaction, yet predicting dense foot contact from a single RGB image remains largely underexplored. There are two main challenges for learning dense foot contact estimation. First, shoes exhibit highly diverse appearances, making it difficult for models to generalize across different styles. Second, ground often has a monotonous appearance, making it difficult to extract informative features. To tackle these issues, we present a FEet COntact estimation (FECO) framework that learns dense foot contact with shoe style-invariant and ground-aware learning. To overcome the challenge of shoe appearance diversity, our approach incorporates shoe style adversarial training that enforces shoe style-invariant features for contact estimation. To effectively utilize ground information, we introduce a ground feature extractor that captures ground properties based on spatial context. As a result, our proposed method achieves robust foot contact estimation regardless of shoe appearance and effectively leverages ground information. Code will be released.
>
---
#### [new 012] Adaptive Parameter Optimization for Robust Remote Photoplethysmography
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对远程光体积脉搏波图（rPPG）在不同环境下的适应性问题，提出无需训练的PRISM算法。通过在线自适应优化光照去趋势与颜色混合参数，提升信号质量。实验表明，PRISM在多个数据集上达到领先无监督性能，且实时运行，显著增强rPPG在复杂场景中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.21903v1](https://arxiv.org/pdf/2511.21903v1)**

> **作者:** Cecilia G. Morales; Fanurs Chi En Teh; Kai Li; Pushpak Agrawal; Artur Dubrawski
>
> **备注:** Accepted in Times Series for Health NeurIPs Workshop 2025
>
> **摘要:** Remote photoplethysmography (rPPG) enables contactless vital sign monitoring using standard RGB cameras. However, existing methods rely on fixed parameters optimized for particular lighting conditions and camera setups, limiting adaptability to diverse deployment environments. This paper introduces the Projection-based Robust Signal Mixing (PRISM) algorithm, a training-free method that jointly optimizes photometric detrending and color mixing through online parameter adaptation based on signal quality assessment. PRISM achieves state-of-the-art performance among unsupervised methods, with MAE of 0.77 bpm on PURE and 0.66 bpm on UBFC-rPPG, and accuracy of 97.3\% and 97.5\% respectively at a 5 bpm threshold. Statistical analysis confirms PRISM performs equivalently to leading supervised methods ($p > 0.2$), while maintaining real-time CPU performance without training. This validates that adaptive time series optimization significantly improves rPPG across diverse conditions.
>
---
#### [new 013] MG-Nav: Dual-Scale Visual Navigation via Sparse Spatial Memory
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MG-Nav，一种用于零样本视觉导航的双尺度框架。针对复杂场景中全局规划与局部控制的协同难题，构建稀疏空间记忆图（SMG）统一多视图语义与空间结构，结合图像-实例混合检索与几何适配模块，实现跨场景的精准导航与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22609v1](https://arxiv.org/pdf/2511.22609v1)**

> **作者:** Bo Wang; Jiehong Lin; Chenzhi Liu; Xinting Hu; Yifei Yu; Tianjia Liu; Zhongrui Wang; Xiaojuan Qi
>
> **备注:** 10pages, 5 figures
>
> **摘要:** We present MG-Nav (Memory-Guided Navigation), a dual-scale framework for zero-shot visual navigation that unifies global memory-guided planning with local geometry-enhanced control. At its core is the Sparse Spatial Memory Graph (SMG), a compact, region-centric memory where each node aggregates multi-view keyframe and object semantics, capturing both appearance and spatial structure while preserving viewpoint diversity. At the global level, the agent is localized on SMG and a goal-conditioned node path is planned via an image-to-instance hybrid retrieval, producing a sequence of reachable waypoints for long-horizon guidance. At the local level, a navigation foundation policy executes these waypoints in point-goal mode with obstacle-aware control, and switches to image-goal mode when navigating from the final node towards the visual target. To further enhance viewpoint alignment and goal recognition, we introduce VGGT-adapter, a lightweight geometric module built on the pre-trained VGGT model, which aligns observation and goal features in a shared 3D-aware space. MG-Nav operates global planning and local control at different frequencies, using periodic re-localization to correct errors. Experiments on HM3D Instance-Image-Goal and MP3D Image-Goal benchmarks demonstrate that MG-Nav achieves state-of-the-art zero-shot performance and remains robust under dynamic rearrangements and unseen scene conditions.
>
---
#### [new 014] DocVAL: Validated Chain-of-Thought Distillation for Grounded Document VQA
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文档视觉问答（DocVQA）任务，解决大模型效率低、小模型定位能力差的难题。提出DocVAL框架，通过验证的思维链蒸馏，结合多模块验证器与两阶段训练，提升学生模型的空间推理能力，实现高精度且无需OCR的高效推理。**

- **链接: [https://arxiv.org/pdf/2511.22521v1](https://arxiv.org/pdf/2511.22521v1)**

> **作者:** Ahmad Mohammadshirazi; Pinaki Prasad Guha Neogi; Dheeraj Kulshrestha; Rajiv Ramnath
>
> **摘要:** Document visual question answering (DocVQA) requires models to jointly reason over textual content and spatial layout, yet current systems exhibit a sharp accuracy--efficiency trade-off: large teacher models achieve strong grounding but are too expensive for deployment, while compact students suffer substantial drops in localization performance. We propose DocVAL, a validated chain-of-thought distillation framework that transfers the spatial reasoning ability of a large teacher into a deployable student VLM through three key components: (1) teacher supervision with validation-time text detection to filter and denoise training signals, (2) a multi-module validator (VAL) that enforces answer correctness and geometric consistency while producing fine-grained, pixel-level error feedback, and (3) a two-stage student training scheme that first learns from validated CoT traces and then undergoes iterative refinement driven by VAL feedback. Our student (Gemma-3 12B) achieves 91.4\% ANLS and 82.4\% mAP on DocVQA as a pure VLM requiring no text detection or OCR at inference. Extensive ablations demonstrate that validated feedback contributes 6.3 mAP gain and iterative refinement accounts for 9.7 mAP improvement. We release 95k high-quality, validator-verified CoT traces to advance spatial reasoning research in document understanding.
>
---
#### [new 015] MathSight: A Benchmark Exploring Have Vision-Language Models Really Seen in University-Level Mathematical Reasoning?
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉语言模型在高等数学推理中是否真正利用视觉信息的问题，提出MathSight基准。通过多模态变体与纯文本对比，发现视觉贡献随难度增加而减弱，且无图像输入的模型表现更优，揭示当前模型依赖语言先验而非真实视觉理解。**

- **链接: [https://arxiv.org/pdf/2511.23112v1](https://arxiv.org/pdf/2511.23112v1)**

> **作者:** Yuandong Wang; Yao Cui; Yuxin Zhao; Zhen Yang; Yangfu Zhu; Zhenzhou Shao
>
> **备注:** Comments: 32 pages, 15 figures, 9 tables, includes appendix. Project page: https://cnu-bot-group.github.io/MathSight/
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) have achieved impressive progress in multimodal mathematical reasoning. Yet, how much visual information truly contributes to reasoning remains unclear. Existing benchmarks report strong overall performance but seldom isolate the role of the image modality, leaving open whether VLMs genuinely leverage visual understanding or merely depend on linguistic priors. To address this, we present MathSight, a university-level multimodal mathematical reasoning benchmark designed to disentangle and quantify the effect of visual input. Each problem includes multiple visual variants -- original, hand-drawn, photo-captured -- and a text-only condition for controlled comparison. Experiments on state-of-the-art VLMs reveal a consistent trend: the contribution of visual information diminishes with increasing problem difficulty. Remarkably, Qwen3-VL without any image input surpasses both its multimodal variants and GPT-5, underscoring the need for benchmarks like MathSight to advance genuine vision-grounded reasoning in future models.
>
---
#### [new 016] NumeriKontrol: Adding Numeric Control to Diffusion Transformers for Instruction-based Image Editing
- **分类: cs.CV**

- **简介: 该论文针对指令式图像编辑中精度不足的问题，提出NumeriKontrol框架，通过数值控制实现对图像属性的连续、精准调节。利用数值适配器注入扩散模型，支持零样本多条件编辑，并构建高质量合成数据集，提升编辑准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.23105v1](https://arxiv.org/pdf/2511.23105v1)**

> **作者:** Zhenyu Xu; Xiaoqi Shen; Haotian Nan; Xinyu Zhang
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** Instruction-based image editing enables intuitive manipulation through natural language commands. However, text instructions alone often lack the precision required for fine-grained control over edit intensity. We introduce NumeriKontrol, a framework that allows users to precisely adjust image attributes using continuous scalar values with common units. NumeriKontrol encodes numeric editing scales via an effective Numeric Adapter and injects them into diffusion models in a plug-and-play manner. Thanks to a task-separated design, our approach supports zero-shot multi-condition editing, allowing users to specify multiple instructions in any order. To provide high-quality supervision, we synthesize precise training data from reliable sources, including high-fidelity rendering engines and DSLR cameras. Our Common Attribute Transform (CAT) dataset covers diverse attribute manipulations with accurate ground-truth scales, enabling NumeriKontrol to function as a simple yet powerful interactive editing studio. Extensive experiments show that NumeriKontrol delivers accurate, continuous, and stable scale control across a wide range of attribute editing scenarios. These contributions advance instruction-based image editing by enabling precise, scalable, and user-controllable image manipulation.
>
---
#### [new 017] OralGPT-Omni: A Versatile Dental Multimodal Large Language Model
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出OralGPT-Omni，首个面向牙科的多模态大语言模型，解决牙科领域数据少、标注难、模型可靠性差等问题。通过构建临床推理数据集TRACE-CoT和统一基准MMOral-Uni，采用四阶段训练，显著提升牙科影像理解与分析能力，推动智能牙科发展。**

- **链接: [https://arxiv.org/pdf/2511.22055v1](https://arxiv.org/pdf/2511.22055v1)**

> **作者:** Jing Hao; Yuci Liang; Lizhuo Lin; Yuxuan Fan; Wenkai Zhou; Kaixin Guo; Zanting Ye; Yanpeng Sun; Xinyu Zhang; Yanqi Yang; Qiankun Li; Hao Tang; James Kit-Hon Tsoi; Linlin Shen; Kuo Feng Hung
>
> **备注:** 47 pages, 42 figures, 13 tables
>
> **摘要:** Multimodal Large Language Models (MLLMs) have exhibited immense potential across numerous medical specialties; yet, dentistry remains underexplored, in part due to limited domain-specific data, scarce dental expert annotations, insufficient modality-specific modeling, and challenges in reliability. In this paper, we present OralGPT-Omni, the first dental-specialized MLLM designed for comprehensive and trustworthy analysis across diverse dental imaging modalities and clinical tasks. To explicitly capture dentists' diagnostic reasoning, we construct TRACE-CoT, a clinically grounded chain-of-thought dataset that mirrors dental radiologists' decision-making processes. This reasoning supervision, combined with our proposed four-stage training paradigm, substantially strengthens the model's capacity for dental image understanding and analysis. In parallel, we introduce MMOral-Uni, the first unified multimodal benchmark for dental image analysis. It comprises 2,809 open-ended question-answer pairs spanning five modalities and five tasks, offering a comprehensive evaluation suite to date for MLLMs in digital dentistry. OralGPT-Omni achieves an overall score of 51.84 on the MMOral-Uni benchmark and 45.31 on the MMOral-OPG benchmark, dramatically outperforming the scores of GPT-5. Our work promotes intelligent dentistry and paves the way for future advances in dental image analysis. All code, benchmark, and models will be made publicly available.
>
---
#### [new 018] DriveVGGT: Visual Geometry Transformer for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶场景下的4D视觉重建任务，解决传统VGGT模型因忽略驾驶特有先验导致性能不佳的问题。提出DriveVGGT框架，引入时序视频注意力与多相机一致性注意力，并扩展绝对尺度与车辆位姿预测头，有效利用传感器布局、已知内外参及固定相对位置等先验，显著提升重建精度。**

- **链接: [https://arxiv.org/pdf/2511.22264v1](https://arxiv.org/pdf/2511.22264v1)**

> **作者:** Xiaosong Jia; Yanhao Liu; Junqi You; Renqiu Xia; Yu Hong; Junchi Yan
>
> **摘要:** Feed-forward reconstruction has recently gained significant attention, with VGGT being a notable example. However, directly applying VGGT to autonomous driving (AD) systems leads to sub-optimal results due to the different priors between the two tasks. In AD systems, several important new priors need to be considered: (i) The overlap between camera views is minimal, as autonomous driving sensor setups are designed to achieve coverage at a low cost. (ii) The camera intrinsics and extrinsics are known, which introduces more constraints on the output and also enables the estimation of absolute scale. (iii) Relative positions of all cameras remain fixed though the ego vehicle is in motion. To fully integrate these priors into a feed-forward framework, we propose DriveVGGT, a scale-aware 4D reconstruction framework specifically designed for autonomous driving data. Specifically, we propose a Temporal Video Attention (TVA) module to process multi-camera videos independently, which better leverages the spatiotemporal continuity within each single-camera sequence. Then, we propose a Multi-camera Consistency Attention (MCA) module to conduct window attention with normalized relative pose embeddings, aiming to establish consistency relationships across different cameras while restricting each token to attend only to nearby frames. Finally, we extend the standard VGGT heads by adding an absolute scale head and an ego vehicle pose head. Experiments show that DriveVGGT outperforms VGGT, StreamVGGT, fastVGGT on autonomous driving dataset while extensive ablation studies verify effectiveness of the proposed designs.
>
---
#### [new 019] Analyzing Image Beyond Visual Aspect: Image Emotion Classification via Multiple-Affective Captioning
- **分类: cs.CV**

- **简介: 该论文聚焦图像情感分类任务，针对预训练视觉模型存在的“情感鸿沟”问题，提出基于多情感描述的文本化方法。通过层次化对比损失与情感属性链式推理生成情感语句，结合预训练语言模型实现纯文本情感分类，并引入语义相似性对比损失优化数据分布，有效提升分类性能。**

- **链接: [https://arxiv.org/pdf/2511.23115v1](https://arxiv.org/pdf/2511.23115v1)**

> **作者:** Zibo Zhou; Zhengjun Zhai; Huimin Chen; Wei Dai; Hansen Yang
>
> **摘要:** Image emotion classification (IEC) is a longstanding research field that has received increasing attention with the rapid progress of deep learning. Although recent advances have leveraged the knowledge encoded in pre-trained visual models, their effectiveness is constrained by the "affective gap" , limits the applicability of pre-training knowledge for IEC tasks. It has been demonstrated in psychology that language exhibits high variability, encompasses diverse and abundant information, and can effectively eliminate the "affective gap". Inspired by this, we propose a novel Affective Captioning for Image Emotion Classification (ACIEC) to classify image emotion based on pure texts, which effectively capture the affective information in the image. In our method, a hierarchical multi-level contrastive loss is designed for detecting emotional concepts from images, while an emotional attribute chain-of-thought reasoning is proposed to generate affective sentences. Then, a pre-trained language model is leveraged to synthesize emotional concepts and affective sentences to conduct IEC. Additionally, a contrastive loss based on semantic similarity sampling is designed to solve the problem of large intra-class differences and small inter-class differences in affective datasets. Moreover, we also take the images with embedded texts into consideration, which were ignored by previous studies. Extensive experiments illustrate that our method can effectively bridge the affective gap and achieve superior results on multiple benchmarks.
>
---
#### [new 020] WorldWander: Bridging Egocentric and Exocentric Worlds in Video Generation
- **分类: cs.CV**

- **简介: 该论文提出WorldWander框架，解决视频生成中第一人称（egocentric）与第三人称（exocentric）视角间无缝转换难题。通过引入上下文视角对齐与协作位置编码，结合自建EgoExo-8K数据集，实现跨视角同步与角色一致性，显著提升视频生成质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22098v1](https://arxiv.org/pdf/2511.22098v1)**

> **作者:** Quanjian Song; Yiren Song; Kelly Peng; Yuan Gao; Mike Zheng Shou
>
> **摘要:** Video diffusion models have recently achieved remarkable progress in realism and controllability. However, achieving seamless video translation across different perspectives, such as first-person (egocentric) and third-person (exocentric), remains underexplored. Bridging these perspectives is crucial for filmmaking, embodied AI, and world models. Motivated by this, we present WorldWander, an in-context learning framework tailored for translating between egocentric and exocentric worlds in video generation. Building upon advanced video diffusion transformers, WorldWander integrates (i) In-Context Perspective Alignment and (ii) Collaborative Position Encoding to efficiently model cross-view synchronization. To further support our task, we curate EgoExo-8K, a large-scale dataset containing synchronized egocentric-exocentric triplets from both synthetic and real-world scenarios. Experiments demonstrate that WorldWander achieves superior perspective synchronization, character consistency, and generalization, setting a new benchmark for egocentric-exocentric video translation.
>
---
#### [new 021] One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfe
- **分类: cs.CV**

- **简介: 该论文提出One-to-All Animation，解决非对齐参考图像的字符动画与姿态迁移问题。针对空间错位、部分可见及多分辨率挑战，设计自监督出图训练、参考特征提取与混合融合注意力机制，并引入身份鲁棒姿态控制与令牌替换策略，实现高保真、长视频一致的动画生成。**

- **链接: [https://arxiv.org/pdf/2511.22940v1](https://arxiv.org/pdf/2511.22940v1)**

> **作者:** Shijun Shi; Jing Xu; Zhihang Li; Chunli Peng; Xiaoda Yang; Lijing Lu; Kai Hu; Jiangning Zhang
>
> **备注:** Project Page:https://ssj9596.github.io/one-to-all-animation-project/
>
> **摘要:** Recent advances in diffusion models have greatly improved pose-driven character animation. However, existing methods are limited to spatially aligned reference-pose pairs with matched skeletal structures. Handling reference-pose misalignment remains unsolved. To address this, we present One-to-All Animation, a unified framework for high-fidelity character animation and image pose transfer for references with arbitrary layouts. First, to handle spatially misaligned reference, we reformulate training as a self-supervised outpainting task that transforms diverse-layout reference into a unified occluded-input format. Second, to process partially visible reference, we design a reference extractor for comprehensive identity feature extraction. Further, we integrate hybrid reference fusion attention to handle varying resolutions and dynamic sequence lengths. Finally, from the perspective of generation quality, we introduce identity-robust pose control that decouples appearance from skeletal structure to mitigate pose overfitting, and a token replace strategy for coherent long-video generation. Extensive experiments show that our method outperforms existing approaches. The code and model will be available at https://github.com/ssj9596/One-to-All-Animation.
>
---
#### [new 022] Synthetic Industrial Object Detection: GenAI vs. Feature-Based Methods
- **分类: cs.CV**

- **简介: 该论文研究工业场景下合成数据生成中的“仿真到现实”泛化问题，旨在减少真实数据标注成本。通过对比领域随机化、领域自适应与生成式AI方法，发现基于感知哈希的简单特征过滤在准确率和效率上优于复杂GenAI方法，证明了高效合成数据生成的可行性。**

- **链接: [https://arxiv.org/pdf/2511.23241v1](https://arxiv.org/pdf/2511.23241v1)**

> **作者:** Jose Moises Araya-Martinez; Adrián Sanchis Reig; Gautham Mohan; Sarvenaz Sardari; Jens Lambrecht; Jörg Krüger
>
> **摘要:** Reducing the burden of data generation and annotation remains a major challenge for the cost-effective deployment of machine learning in industrial and robotics settings. While synthetic rendering is a promising solution, bridging the sim-to-real gap often requires expert intervention. In this work, we benchmark a range of domain randomization (DR) and domain adaptation (DA) techniques, including feature-based methods, generative AI (GenAI), and classical rendering approaches, for creating contextualized synthetic data without manual annotation. Our evaluation focuses on the effectiveness and efficiency of low-level and high-level feature alignment, as well as a controlled diffusion-based DA method guided by prompts generated from real-world contexts. We validate our methods on two datasets: a proprietary industrial dataset (automotive and logistics) and a public robotics dataset. Results show that if render-based data with enough variability is available as seed, simpler feature-based methods, such as brightness-based and perceptual hashing filtering, outperform more complex GenAI-based approaches in both accuracy and resource efficiency. Perceptual hashing consistently achieves the highest performance, with mAP50 scores of 98% and 67% on the industrial and robotics datasets, respectively. Additionally, GenAI methods present significant time overhead for data generation at no apparent improvement of sim-to-real mAP values compared to simpler methods. Our findings offer actionable insights for efficiently bridging the sim-to-real gap, enabling high real-world performance from models trained exclusively on synthetic data.
>
---
#### [new 023] UMind-VL: A Generalist Ultrasound Vision-Language Model for Unified Grounded Perception and Comprehensive Interpretation
- **分类: cs.CV**

- **简介: 该论文提出UMind-VL，一个统一的超声视觉语言模型，旨在解决超声图像低层感知（如分割、定位）与高层解读（如诊断、推理）脱节的问题。通过构建大规模多模态数据集和轻量级动态卷积掩码解码器，实现多种任务的统一建模，在多项基准上达到或超越专业模型性能。**

- **链接: [https://arxiv.org/pdf/2511.22256v1](https://arxiv.org/pdf/2511.22256v1)**

> **作者:** Dengbo Chen; Ziwei Zhao; Kexin Zhang; Shishuang Zhao; Junjie Hou; Yaqian Wang; Nianxi Liao; Anlan Sun; Fei Gao; Jia Ding; Yuhang Liu; Dong Wang
>
> **摘要:** Despite significant strides in medical foundation models, the ultrasound domain lacks a comprehensive solution capable of bridging low-level Ultrasound Grounded Perception (e.g., segmentation, localization) and high-level Ultrasound Comprehensive Interpretation (e.g., diagnosis, reasoning). To bridge this gap, we propose UMind-VL, a unified foundation model designed to synergize pixel-level structural understanding with complex clinical reasoning. We first introduce UMind-DS, a large-scale multimodal dataset comprising 1.2 million ultrasound image-text pairs across 16 anatomical regions, enriching standard data with pixel-level annotations and clinician-validated rationales. Architecturally, UMind-VL incorporates a lightweight Dynamic Convolutional Mask Decoder that generates masks via dynamic kernels conditioned on LLM outputs. This design, combined with task-specific tokens, unifies segmentation, detection, geometric measurement, and diagnosis tasks within a single framework. Extensive evaluations demonstrate that UMind-VL significantly outperforms existing generalist multimodal models and achieves performance on par with, or superior to, state-of-the-art specialist models across segmentation, detection, keypoint localization, and diagnostic reasoning benchmarks, while maintaining strong generalization ability. We demonstrate the capability of UMind-VL in Figure 1.
>
---
#### [new 024] StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation
- **分类: cs.CV**

- **简介: 该论文针对生成模型中的矩形流（Rectified Flow）效率低下的问题，提出一套从理论到实现的加速方案。通过新型速度场批处理、异构时间步向量化及动态TensorRT编译，显著提升512×512图像生成速度，最高达原速度的6.11倍，超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22009v1](https://arxiv.org/pdf/2511.22009v1)**

> **作者:** Sen Fang; Hongbin Zhong; Yalin Feng; Dimitris N. Metaxas
>
> **备注:** Project Page at https://world-snapshot.github.io/StreamFlow/
>
> **摘要:** New technologies such as Rectified Flow and Flow Matching have significantly improved the performance of generative models in the past two years, especially in terms of control accuracy, generation quality, and generation efficiency. However, due to some differences in its theory, design, and existing diffusion models, the existing acceleration methods cannot be directly applied to the Rectified Flow model. In this article, we have comprehensively implemented an overall acceleration pipeline from the aspects of theory, design, and reasoning strategies. This pipeline uses new methods such as batch processing with a new velocity field, vectorization of heterogeneous time-step batch processing, and dynamic TensorRT compilation for the new methods to comprehensively accelerate related models based on flow models. Currently, the existing public methods usually achieve an acceleration of 18%, while experiments have proved that our new method can accelerate the 512*512 image generation speed to up to 611%, which is far beyond the current non-generalized acceleration methods.
>
---
#### [new 025] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对机器人图像与视频分割任务，解决机器人形态多样、结构复杂导致的分割难题。提出RobotSeg模型，通过结构增强记忆关联、自动提示生成和标签高效训练策略，实现无需人工标注的精准分割，并构建了大规模VRS数据集，显著提升分割性能。**

- **链接: [https://arxiv.org/pdf/2511.22950v1](https://arxiv.org/pdf/2511.22950v1)**

> **作者:** Haiyang Mei; Qiming Huang; Hai Ci; Mike Zheng Shou
>
> **备注:** Project page: https://github.com/showlab/RobotSeg
>
> **摘要:** Accurate robot segmentation is a fundamental capability for robotic perception. It enables precise visual servoing for VLA systems, scalable robot-centric data augmentation, accurate real-to-sim transfer, and reliable safety monitoring in dynamic human-robot environments. Despite the strong capabilities of modern segmentation models, surprisingly it remains challenging to segment robots. This is due to robot embodiment diversity, appearance ambiguity, structural complexity, and rapid shape changes. Embracing these challenges, we introduce RobotSeg, a foundation model for robot segmentation in image and video. RobotSeg is built upon the versatile SAM 2 foundation model but addresses its three limitations for robot segmentation, namely the lack of adaptation to articulated robots, reliance on manual prompts, and the need for per-frame training mask annotations, by introducing a structure-enhanced memory associator, a robot prompt generator, and a label-efficient training strategy. These innovations collectively enable a structure-aware, automatic, and label-efficient solution. We further construct the video robot segmentation (VRS) dataset comprising over 2.8k videos (138k frames) with diverse robot embodiments and environments. Extensive experiments demonstrate that RobotSeg achieves state-of-the-art performance on both images and videos, establishing a strong foundation for future advances in robot perception.
>
---
#### [new 026] Match-and-Fuse: Consistent Generation from Unstructured Image Sets
- **分类: cs.CV**

- **简介: 该论文提出Match-and-Fuse方法，解决无结构图像集的一致性可控生成问题。针对多视角、多时序图像集合中共享内容的连贯生成难题，通过图模型建模图像对间的联合生成与特征融合，实现零样本、无需训练的全局一致性生成，提升视觉质量与内容一致性。**

- **链接: [https://arxiv.org/pdf/2511.22287v1](https://arxiv.org/pdf/2511.22287v1)**

> **作者:** Kate Feingold; Omri Kaduri; Tali Dekel
>
> **备注:** Project page: https://match-and-fuse.github.io/
>
> **摘要:** We present Match-and-Fuse - a zero-shot, training-free method for consistent controlled generation of unstructured image sets - collections that share a common visual element, yet differ in viewpoint, time of capture, and surrounding content. Unlike existing methods that operate on individual images or densely sampled videos, our framework performs set-to-set generation: given a source set and user prompts, it produces a new set that preserves cross-image consistency of shared content. Our key idea is to model the task as a graph, where each node corresponds to an image and each edge triggers a joint generation of image pairs. This formulation consolidates all pairwise generations into a unified framework, enforcing their local consistency while ensuring global coherence across the entire set. This is achieved by fusing internal features across image pairs, guided by dense input correspondences, without requiring masks or manual supervision. It also allows us to leverage an emergent prior in text-to-image models that encourages coherent generation when multiple views share a single canvas. Match-and-Fuse achieves state-of-the-art consistency and visual quality, and unlocks new capabilities for content creation from image collections.
>
---
#### [new 027] Gaussians on Fire: High-Frequency Reconstruction of Flames
- **分类: cs.CV**

- **简介: 该论文属于动态火焰3D重建任务。针对火的高频率、透明及易变特性，提出基于高斯的时空表示方法，利用三视角图像与硬件同步，通过融合光流与深度先验，实现高精度动态火焰重建，解决了少视角下欠约束几何问题。**

- **链接: [https://arxiv.org/pdf/2511.22459v1](https://arxiv.org/pdf/2511.22459v1)**

> **作者:** Jakob Nazarenus; Dominik Michels; Wojtek Palubicki; Simin Kou; Fang-Lue Zhang; Soren Pirk; Reinhard Koch
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** We propose a method to reconstruct dynamic fire in 3D from a limited set of camera views with a Gaussian-based spatiotemporal representation. Capturing and reconstructing fire and its dynamics is highly challenging due to its volatile nature, transparent quality, and multitude of high-frequency features. Despite these challenges, we aim to reconstruct fire from only three views, which consequently requires solving for under-constrained geometry. We solve this by separating the static background from the dynamic fire region by combining dense multi-view stereo images with monocular depth priors. The fire is initialized as a 3D flow field, obtained by fusing per-view dense optical flow projections. To capture the high frequency features of fire, each 3D Gaussian encodes a lifetime and linear velocity to match the dense optical flow. To ensure sub-frame temporal alignment across cameras we employ a custom hardware synchronization pattern -- allowing us to reconstruct fire with affordable commodity hardware. Our quantitative and qualitative validations across numerous reconstruction experiments demonstrate robust performance for diverse and challenging real fire scenarios.
>
---
#### [new 028] INSIGHT: An Interpretable Neural Vision-Language Framework for Reasoning of Generative Artifacts
- **分类: cs.CV**

- **简介: 该论文提出INSIGHT框架，解决AI生成图像在低分辨率、压缩等极端条件下的检测难与解释性差问题。通过多尺度定位与语义对齐，实现高鲁棒性检测与可解释性分析，提升生成内容验证的透明度与可信度。**

- **链接: [https://arxiv.org/pdf/2511.22351v1](https://arxiv.org/pdf/2511.22351v1)**

> **作者:** Anshul Bagaria
>
> **备注:** 36 pages, 17 figures
>
> **摘要:** The growing realism of AI-generated images produced by recent GAN and diffusion models has intensified concerns over the reliability of visual media. Yet, despite notable progress in deepfake detection, current forensic systems degrade sharply under real-world conditions such as severe downsampling, compression, and cross-domain distribution shifts. Moreover, most detectors operate as opaque classifiers, offering little insight into why an image is flagged as synthetic, undermining trust and hindering adoption in high-stakes settings. We introduce INSIGHT (Interpretable Neural Semantic and Image-based Generative-forensic Hallucination Tracing), a unified multimodal framework for robust detection and transparent explanation of AI-generated images, even at extremely low resolutions (16x16 - 64x64). INSIGHT combines hierarchical super-resolution for amplifying subtle forensic cues without inducing misleading artifacts, Grad-CAM driven multi-scale localization to reveal spatial regions indicative of generative patterns, and CLIP-guided semantic alignment to map visual anomalies to human-interpretable descriptors. A vision-language model is then prompted using a structured ReAct + Chain-of-Thought protocol to produce consistent, fine-grained explanations, verified through a dual-stage G-Eval + LLM-as-a-judge pipeline to minimize hallucinations and ensure factuality. Across diverse domains, including animals, vehicles, and abstract synthetic scenes, INSIGHT substantially improves both detection robustness and explanation quality under extreme degradation, outperforming prior detectors and black-box VLM baselines. Our results highlight a practical path toward transparent, reliable AI-generated image forensics and establish INSIGHT as a step forward in trustworthy multimodal content verification.
>
---
#### [new 029] GeoZero: Incentivizing Reasoning from Scratch on Geospatial Scenes
- **分类: cs.CV**

- **简介: 该论文提出GeoZero框架，旨在解决遥感多模态大模型推理依赖昂贵标注和人类偏见的问题。通过构建无预设思维链的训练数据集与自答案引导的强化学习方法，实现零样本地理空间推理，显著提升模型在多样化任务中的泛化与自主推理能力。**

- **链接: [https://arxiv.org/pdf/2511.22645v1](https://arxiv.org/pdf/2511.22645v1)**

> **作者:** Di Wang; Shunyu Liu; Wentao Jiang; Fengxiang Wang; Yi Liu; Xiaolei Qin; Zhiming Luo; Chaoyang Zhou; Haonan Guo; Jing Zhang; Bo Du; Dacheng Tao; Liangpei Zhang
>
> **备注:** Code, data, and models will be publicly available at https://github.com/MiliLab/GeoZero
>
> **摘要:** Multimodal large language models (MLLMs) have undergone rapid development in advancing geospatial scene understanding. Recent studies have sought to enhance the reasoning capabilities of remote sensing MLLMs, typically through cold-start training with elaborately curated chain-of-thought (CoT) data. However, this approach not only incurs substantial annotation costs but also introduces human biases that may limit the diversity of model reasoning. To address these challenges, we propose GeoZero, a framework that enables MLLMs to perform geospatial reasoning without any predefined CoT supervision. Specifically, we construct two datasets, GeoZero-Instruct and GeoZero-Hard. GeoZero-Instruct allows the model to acquire preliminary geospatial knowledge through supervised fine-tuning, while GeoZero-Hard stimulates deep reasoning during the subsequent reinforcement learning stage. Furthermore, we introduce Answer-Anchored Group Relative Policy Optimization (A$^2$GRPO), where the reasoning process is regularized by the model's own answers, encouraging diverse yet accurate thinking. Extensive experiments on multiple remote sensing vision-language benchmarks demonstrate that GeoZero not only surpasses existing state-of-the-art methods but also fosters universal emergent reasoning capabilities across diverse geospatial tasks. Code,data,and models will be publicly available at https://github.com/MiliLab/GeoZero.
>
---
#### [new 030] Instruction Tuning of Large Language Models for Tabular Data Generation-in One Day
- **分类: cs.CV**

- **简介: 该论文聚焦于表格数据生成任务，解决现有指令微调在表格生成上研究不足、资源需求高的问题。作者构建高质量指令数据集，仅用7K样本和6小时A100训练，使开源模型（Llama3.1-8B-Instruct）在表格生成上达到GPT-4o水平。**

- **链接: [https://arxiv.org/pdf/2511.23220v1](https://arxiv.org/pdf/2511.23220v1)**

> **作者:** Milad Abdollahzadeh; Abdul Raheem; Zilong Zhao; Uzair Javaid; Kevin Yee; Nalam Venkata Abhishek; Tram Truong-Huu; Biplab Sikdar
>
> **备注:** Accepted International Conference on Machine Learning (ICML 2025), 1st Workshop on Foundation Models for Structured Data
>
> **摘要:** Tabular instruction tuning has emerged as a promising research direction for improving LLMs understanding of tabular data. However, the majority of existing works only consider question-answering and reasoning tasks over tabular data, leaving tabular data generation largely unnoticed. In this work, for the first time, we explore the efficacy of instruction tuning in improving LLMs tabular data generation capabilities. More specifically, given the high data and computation requirements of tabular instruction tuning, we aim to address the possibility of instruction tuning for tabular data generation with limited data and computational resources. To achieve this, we first create a high-quality instruction dataset for tabular data, enabling efficient LLM comprehension. We then instruction-tune an open-source LLM (Llama3.1-8B-Instruct) on the training set of this dataset to improve its tabular data generation performance. Our experimental results show that by using our high-quality dataset and instruction-tuning on only 7K instructions with an A100 GPU, for less than 6 hours, we achieve tabular data generation performance on par with the most capable commercial LLM, GPT-4o.
>
---
#### [new 031] Implementation of a Skin Lesion Detection System for Managing Children with Atopic Dermatitis Based on Ensemble Learning
- **分类: cs.CV**

- **简介: 该论文针对儿童特应性皮炎皮肤病变主观诊断易误诊的问题，提出基于集成学习的皮肤病变检测系统（ENSEL）。通过融合多个深度学习模型，提升诊断准确率与响应速度。实验基于真实用户拍摄图像，验证了系统高召回率与低于1秒的处理效率，推动数字医疗发展。**

- **链接: [https://arxiv.org/pdf/2511.23082v1](https://arxiv.org/pdf/2511.23082v1)**

> **作者:** Soobin Jeon; Sujong Kim; Dongmahn Seo
>
> **备注:** 16pages, 14 figures, 7 tables
>
> **摘要:** The amendments made to the Data 3 Act and impact of COVID-19 have fostered the growth of digital healthcare market and promoted the use of medical data in artificial intelligence in South Korea. Atopic dermatitis, a chronic inflammatory skin disease, is diagnosed via subjective evaluations without using objective diagnostic methods, thereby increasing the risk of misdiagnosis. It is also similar to psoriasis in appearance, further complicating its accurate diagnosis. Existing studies on skin diseases have used high-quality dermoscopic image datasets, but such high-quality images cannot be obtained in actual clinical settings. Moreover, existing systems must ensure accuracy and fast response times. To this end, an ensemble learning-based skin lesion detection system (ENSEL) was proposed herein. ENSEL enhanced diagnostic accuracy by integrating various deep learning models via an ensemble approach. Its performance was verified by conducting skin lesion detection experiments using images of skin lesions taken by actual users. Its accuracy and response time were measured using randomly sampled skin disease images. Results revealed that ENSEL achieved high recall in most images and less than 1s s processing speed. This study contributes to the objective diagnosis of skin lesions and promotes the advancement of digital healthcare.
>
---
#### [new 032] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文针对知识密集型视觉问答（KB-VQA）任务，解决现有模型在处理领域特定或需外部知识查询时因检索精度低、噪声多、推理能力弱导致的准确率问题。提出ReAG方法，结合粗细粒度检索与批判模型过滤，通过强化学习优化推理，提升答案准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.22715v1](https://arxiv.org/pdf/2511.22715v1)**

> **作者:** Alberto Compagnoni; Marco Morini; Sara Sarto; Federico Cocchi; Davide Caffagni; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown impressive capabilities in jointly understanding text, images, and videos, often evaluated via Visual Question Answering (VQA). However, even state-of-the-art MLLMs struggle with domain-specific or knowledge-intensive queries, where relevant information is underrepresented in pre-training data. Knowledge-based VQA (KB-VQA) addresses this by retrieving external documents to condition answer generation, but current retrieval-augmented approaches suffer from low precision, noisy passages, and limited reasoning. To address this, we propose ReAG, a novel Reasoning-Augmented Multimodal RAG approach that combines coarse- and fine-grained retrieval with a critic model that filters irrelevant passages, ensuring high-quality additional context. The model follows a multi-stage training strategy leveraging reinforcement learning to enhance reasoning over retrieved content, while supervised fine-tuning serves only as a cold start. Extensive experiments on Encyclopedic-VQA and InfoSeek demonstrate that ReAG significantly outperforms prior methods, improving answer accuracy and providing interpretable reasoning grounded in retrieved evidence. Our source code is publicly available at: https://github.com/aimagelab/ReAG.
>
---
#### [new 033] Optimizer Sensitivity In Vision Transformerbased Iris Recognition: Adamw Vs Sgd Vs Rmsprop
- **分类: cs.CV; stat.CO**

- **简介: 该论文研究Vision Transformer在虹膜识别任务中优化器选择的影响。针对深度学习模型训练中优化器对性能影响未明的问题，对比AdamW、SGD、RMSprop三种优化器，评估其对识别准确率与模型稳定性的作用，旨在提升生物特征识别系统的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22994v1](https://arxiv.org/pdf/2511.22994v1)**

> **作者:** Moh Imam Faiz; Aviv Yuniar Rahman; Rangga Pahlevi Putra
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** The security of biometric authentication is increasingly critical as digital identity systems expand. Iris recognition offers high reliability due to its distinctive and stable texture patterns. Recent progress in deep learning, especially Vision Transformers ViT, has improved visual recognition performance. Yet, the effect of optimizer choice on ViT-based biometric systems remains understudied. This work evaluates how different optimizers influence the accuracy and stability of ViT for iris recognition, providing insights to enhance the robustness of biometric identification models.
>
---
#### [new 034] JarvisEvo: Towards a Self-Evolving Photo Editing Agent with Synergistic Editor-Evaluator Optimization
- **分类: cs.CV**

- **简介: 该论文提出JarvisEvo，一种自进化图像编辑代理，旨在解决指令幻觉和奖励劫持问题。通过多模态思维链与编辑-评估协同优化，实现高质量、可信赖的图像编辑，支持精细局部与全局调整，在基准测试中显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.23002v1](https://arxiv.org/pdf/2511.23002v1)**

> **作者:** Yunlong Lin; Linqing Wang; Kunjie Lin; Zixu Lin; Kaixiong Gong; Wenbo Li; Bin Lin; Zhenxi Li; Shiyi Zhang; Yuyang Peng; Wenxun Dai; Xinghao Ding; Chunyu Wang; Qinglin Lu
>
> **备注:** 31 pages, 18 figures
>
> **摘要:** Agent-based editing models have substantially advanced interactive experiences, processing quality, and creative flexibility. However, two critical challenges persist: (1) instruction hallucination, text-only chain-of-thought (CoT) reasoning cannot fully prevent factual errors due to inherent information bottlenecks; (2) reward hacking, dynamic policy optimization against static reward models allows agents to exploit flaws in reward functions. To address these issues, we propose JarvisEvo, a unified image editing agent that emulates an expert human designer by iteratively editing, selecting appropriate tools, evaluating results, and reflecting on its own decisions to refine outcomes. JarvisEvo offers three key advantages: (1) an interleaved multimodal chain-of-thought (iMCoT) reasoning mechanism that enhances instruction following and editing quality; (2) a synergistic editor-evaluator policy optimization (SEPO) framework that enables self-improvement without external rewards, effectively mitigating reward hacking; and (3) support for both global and local fine-grained editing through seamless integration of Adobe Lightroom. On ArtEdit-Bench, JarvisEvo outperforms Nano-Banana by an average of 18.95% on preservative editing metrics, including a substantial 44.96% improvement in pixel-level content fidelity.
>
---
#### [new 035] CNN-Based Framework for Pedestrian Age and Gender Classification Using Far-View Surveillance in Mixed-Traffic Intersections
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出一种基于CNN的远距离行人年龄与性别分类框架，针对混合交通路口行人安全监测中缺乏实时人口统计信息的问题。利用全身体态特征，构建六类联合分类任务，在孟加拉国达卡三处高风险路口采集数据，采用ResNet50与轻量级CNN模型，实现高效准确分类，支持现有监控系统升级，助力智慧交通与安全干预。**

- **链接: [https://arxiv.org/pdf/2511.22873v1](https://arxiv.org/pdf/2511.22873v1)**

> **作者:** Shisir Shahriar Arif; Md. Muhtashim Shahrier; Nazmul Haque; Md Asif Raihan; Md. Hadiuzzaman
>
> **备注:** Accepted for poster presentation at the 105th Annual Meeting of the Transportation Research Board
>
> **摘要:** Pedestrian safety remains a pressing concern in congested urban intersections, particularly in low- and middle-income countries where traffic is multimodal, and infrastructure often lacks formal control. Demographic factors like age and gender significantly influence pedestrian vulnerability, yet real-time monitoring systems rarely capture this information. To address this gap, this study proposes a deep learning framework that classifies pedestrian age group and gender from far-view intersection footage using convolutional neural networks (CNNs), without relying on facial recognition or high-resolution imagery. The classification is structured as a unified six-class problem, distinguishing adult, teenager, and child pedestrians for both males and females, based on full-body visual cues. Video data was collected from three high-risk intersections in Dhaka, Bangladesh. Two CNN architectures were implemented: ResNet50, a deep convolutional neural network pretrained on ImageNet, and a custom lightweight CNN optimized for computational efficiency. Eight model variants explored combinations of pooling strategies and optimizers. ResNet50 with Max Pooling and SGD achieved the highest accuracy (86.19%), while the custom CNN performed comparably (84.15%) with fewer parameters and faster training. The model's efficient design enables real-time inference on standard surveillance feeds. For practitioners, this system provides a scalable, cost-effective tool to monitor pedestrian demographics at intersections using existing camera infrastructure. Its outputs can shape intersection design, optimize signal timing, and enable targeted safety interventions for vulnerable groups such as children or the elderly. By offering demographic insights often missing in conventional traffic data, the framework supports more inclusive, data-driven planning in mixed-traffic environments.
>
---
#### [new 036] UAV-MM3D: A Large-Scale Synthetic Benchmark for 3D Perception of Unmanned Aerial Vehicles with Multi-Modal Data
- **分类: cs.CV**

- **简介: 该论文针对低空无人机3D感知难题，提出UAV-MM3D合成数据集，涵盖多场景、多模态（RGB、IR、LiDAR、Radar、DVS）与精细标注，解决真实数据获取难、标注成本高的问题。构建LGFusionNet与轨迹预测基线，推动无人机3D检测、跟踪与轨迹预测研究。**

- **链接: [https://arxiv.org/pdf/2511.22404v1](https://arxiv.org/pdf/2511.22404v1)**

> **作者:** Longkun Zou; Jiale Wang; Rongqin Liang; Hai Wu; Ke Chen; Yaowei Wang
>
> **摘要:** Accurate perception of UAVs in complex low-altitude environments is critical for airspace security and related intelligent systems. Developing reliable solutions requires large-scale, accurately annotated, and multimodal data. However, real-world UAV data collection faces inherent constraints due to airspace regulations, privacy concerns, and environmental variability, while manual annotation of 3D poses and cross-modal correspondences is time-consuming and costly. To overcome these challenges, we introduce UAV-MM3D, a high-fidelity multimodal synthetic dataset for low-altitude UAV perception and motion understanding. It comprises 400K synchronized frames across diverse scenes (urban areas, suburbs, forests, coastal regions) and weather conditions (clear, cloudy, rainy, foggy), featuring multiple UAV models (micro, small, medium-sized) and five modalities - RGB, IR, LiDAR, Radar, and DVS (Dynamic Vision Sensor). Each frame provides 2D/3D bounding boxes, 6-DoF poses, and instance-level annotations, enabling core tasks related to UAVs such as 3D detection, pose estimation, target tracking, and short-term trajectory forecasting. We further propose LGFusionNet, a LiDAR-guided multimodal fusion baseline, and a dedicated UAV trajectory prediction baseline to facilitate benchmarking. With its controllable simulation environment, comprehensive scenario coverage, and rich annotations, UAV3D offers a public benchmark for advancing 3D perception of UAVs.
>
---
#### [new 037] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation
- **分类: cs.CV**

- **简介: 该论文针对语义分割中的域泛化问题，提出FLEX-Seg框架，利用生成数据与标签间的固有错位作为优势。通过多尺度原型、不确定性边界强调和难度感知采样，增强模型对风格变化的鲁棒性。在多个真实数据集上实现性能提升，验证了利用不完美数据的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22948v1](https://arxiv.org/pdf/2511.22948v1)**

> **作者:** Taeyeong Kim; SeungJoon Lee; Jung Uk Kim; MyeongAh Cho
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Domain generalization in semantic segmentation faces challenges from domain shifts, particularly under adverse conditions. While diffusion-based data generation methods show promise, they introduce inherent misalignment between generated images and semantic masks. This paper presents FLEX-Seg (FLexible Edge eXploitation for Segmentation), a framework that transforms this limitation into an opportunity for robust learning. FLEX-Seg comprises three key components: (1) Granular Adaptive Prototypes that captures boundary characteristics across multiple scales, (2) Uncertainty Boundary Emphasis that dynamically adjusts learning emphasis based on prediction entropy, and (3) Hardness-Aware Sampling that progressively focuses on challenging examples. By leveraging inherent misalignment rather than enforcing strict alignment, FLEX-Seg learns robust representations while capturing rich stylistic variations. Experiments across five real-world datasets demonstrate consistent improvements over state-of-the-art methods, achieving 2.44% and 2.63% mIoU gains on ACDC and Dark Zurich. Our findings validate that adaptive strategies for handling imperfect synthetic data lead to superior domain generalization. Code is available at https://github.com/VisualScienceLab-KHU/FLEX-Seg.
>
---
#### [new 038] Bharat Scene Text: A Novel Comprehensive Dataset and Benchmark for Indian Language Scene Text Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对印度多语言场景文本识别难题，提出Bharat Scene Text Dataset（BSTD），涵盖11种印度语言和英语，包含超10万词。数据集支持检测、脚本识别、单词识别及端到端识别任务。通过微调英文先进模型，揭示了印度语言场景文本识别的挑战与机遇，推动该领域研究发展。**

- **链接: [https://arxiv.org/pdf/2511.23071v1](https://arxiv.org/pdf/2511.23071v1)**

> **作者:** Anik De; Abhirama Subramanyam Penamakuri; Rajeev Yadav; Aditya Rathore; Harshiv Shah; Devesh Sharma; Sagar Agarwal; Pravin Kumar; Anand Mishra
>
> **备注:** Under Peer Review
>
> **摘要:** Reading scene text, that is, text appearing in images, has numerous application areas, including assistive technology, search, and e-commerce. Although scene text recognition in English has advanced significantly and is often considered nearly a solved problem, Indian language scene text recognition remains an open challenge. This is due to script diversity, non-standard fonts, and varying writing styles, and, more importantly, the lack of high-quality datasets and open-source models. To address these gaps, we introduce the Bharat Scene Text Dataset (BSTD) - a large-scale and comprehensive benchmark for studying Indian Language Scene Text Recognition. It comprises more than 100K words that span 11 Indian languages and English, sourced from over 6,500 scene images captured across various linguistic regions of India. The dataset is meticulously annotated and supports multiple scene text tasks, including: (i) Scene Text Detection, (ii) Script Identification, (iii) Cropped Word Recognition, and (iv) End-to-End Scene Text Recognition. We evaluated state-of-the-art models originally developed for English by adapting (fine-tuning) them for Indian languages. Our results highlight the challenges and opportunities in Indian language scene text recognition. We believe that this dataset represents a significant step toward advancing research in this domain. All our models and data are open source.
>
---
#### [new 039] Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective
- **分类: cs.CV**

- **简介: 该论文研究视觉生成中的高维潜在空间扩散问题，针对高维自编码器导致的重建与生成质量权衡，发现编码器对高频信息建模不足。提出频域热身（FreqWarm）策略，在训练初期增强高频潜在信号暴露，提升生成质量，无需修改自编码器，适用于多种架构。**

- **链接: [https://arxiv.org/pdf/2511.22249v1](https://arxiv.org/pdf/2511.22249v1)**

> **作者:** Bolin Lai; Xudong Wang; Saketh Rambhatla; James M. Rehg; Zsolt Kira; Rohit Girdhar; Ishan Misra
>
> **备注:** 11 pages, 7 figures, 4 tables
>
> **摘要:** Latent diffusion has become the default paradigm for visual generation, yet we observe a persistent reconstruction-generation trade-off as latent dimensionality increases: higher-capacity autoencoders improve reconstruction fidelity but generation quality eventually declines. We trace this gap to the different behaviors in high-frequency encoding and decoding. Through controlled perturbations in both RGB and latent domains, we analyze encoder/decoder behaviors and find that decoders depend strongly on high-frequency latent components to recover details, whereas encoders under-represent high-frequency contents, yielding insufficient exposure and underfitting in high-frequency bands for diffusion model training. To address this issue, we introduce FreqWarm, a plug-and-play frequency warm-up curriculum that increases early-stage exposure to high-frequency latent signals during diffusion or flow-matching training -- without modifying or retraining the autoencoder. Applied across several high-dimensional autoencoders, FreqWarm consistently improves generation quality: decreasing gFID by 14.11 on Wan2.2-VAE, 6.13 on LTX-VAE, and 4.42 on DC-AE-f32, while remaining architecture-agnostic and compatible with diverse backbones. Our study shows that explicitly managing frequency exposure can successfully turn high-dimensional latent spaces into more diffusible targets.
>
---
#### [new 040] ClearGCD: Mitigating Shortcut Learning For Robust Generalized Category Discovery
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对开放世界中的广义类别发现（GCD）任务，解决现有方法因捷径学习导致的原型混淆与已知类遗忘问题。提出ClearGCD框架，通过语义视图对齐与捷径抑制正则化，增强语义一致性并分离新旧类别，有效提升模型鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22892v1](https://arxiv.org/pdf/2511.22892v1)**

> **作者:** Kailin Lyu; Jianwei He; Long Xiao; Jianing Zeng; Liang Fan; Lin Shu; Jie Hao
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** In open-world scenarios, Generalized Category Discovery (GCD) requires identifying both known and novel categories within unlabeled data. However, existing methods often suffer from prototype confusion caused by shortcut learning, which undermines generalization and leads to forgetting of known classes. We propose ClearGCD, a framework designed to mitigate reliance on non-semantic cues through two complementary mechanisms. First, Semantic View Alignment (SVA) generates strong augmentations via cross-class patch replacement and enforces semantic consistency using weak augmentations. Second, Shortcut Suppression Regularization (SSR) maintains an adaptive prototype bank that aligns known classes while encouraging separation of potential novel ones. ClearGCD can be seamlessly integrated into parametric GCD approaches and consistently outperforms state-of-the-art methods across multiple benchmarks.
>
---
#### [new 041] ICM-SR: Image-Conditioned Manifold Regularization for Image Super-Resoultion
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对真实世界图像超分辨率任务，解决现有方法依赖文本条件生成先验导致的失真与模糊问题。提出图像条件流形正则化（ICM-SR），利用稀疏结构信息（颜色图与Canny边缘）构建稳定且任务对齐的正则化流形，提升超分质量，尤其改善感知效果。**

- **链接: [https://arxiv.org/pdf/2511.22048v1](https://arxiv.org/pdf/2511.22048v1)**

> **作者:** Junoh Kang; Donghun Ryu; Bohyung Han
>
> **摘要:** Real world image super-resolution (Real-ISR) often leverages the powerful generative priors of text-to-image diffusion models by regularizing the output to lie on their learned manifold. However, existing methods often overlook the importance of the regularizing manifold, typically defaulting to a text-conditioned manifold. This approach suffers from two key limitations. Conceptually, it is misaligned with the Real-ISR task, which is to generate high quality (HQ) images directly tied to the low quality (LQ) images. Practically, the teacher model often reconstructs images with color distortions and blurred edges, indicating a flawed generative prior for this task. To correct these flaws and ensure conceptual alignment, a more suitable manifold must incorporate information from the images. While the most straightforward approach is to condition directly on the raw input images, their high information densities make the regularization process numerically unstable. To resolve this, we propose image-conditioned manifold regularization (ICM), a method that regularizes the output towards a manifold conditioned on the sparse yet essential structural information: a combination of colormap and Canny edges. ICM provides a task-aligned and stable regularization signal, thereby avoiding the instability of dense-conditioning and enhancing the final super-resolution quality. Our experiments confirm that the proposed regularization significantly enhances super-resolution performance, particularly in perceptual quality, demonstrating its effectiveness for real-world applications. We will release the source code of our work for reproducibility.
>
---
#### [new 042] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration
- **分类: cs.CV**

- **简介: 该论文针对3D扩散模型推理慢的问题，提出训练-free的Fast3Dcache框架。通过动态调度缓存与稳定性判据，实现几何一致性下的加速，显著提升效率并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2511.22533v1](https://arxiv.org/pdf/2511.22533v1)**

> **作者:** Mengyu Yang; Yanming Yang; Chenyi Xu; Chenxi Song; Yufan Zuo; Tong Zhao; Ruibo Li; Chi Zhang
>
> **摘要:** Diffusion models have achieved impressive generative quality across modalities like 2D images, videos, and 3D shapes, but their inference remains computationally expensive due to the iterative denoising process. While recent caching-based methods effectively reuse redundant computations to speed up 2D and video generation, directly applying these techniques to 3D diffusion models can severely disrupt geometric consistency. In 3D synthesis, even minor numerical errors in cached latent features accumulate, causing structural artifacts and topological inconsistencies. To overcome this limitation, we propose Fast3Dcache, a training-free geometry-aware caching framework that accelerates 3D diffusion inference while preserving geometric fidelity. Our method introduces a Predictive Caching Scheduler Constraint (PCSC) to dynamically determine cache quotas according to voxel stabilization patterns and a Spatiotemporal Stability Criterion (SSC) to select stable features for reuse based on velocity magnitude and acceleration criterion. Comprehensive experiments show that Fast3Dcache accelerates inference significantly, achieving up to a 27.12% speed-up and a 54.8% reduction in FLOPs, with minimal degradation in geometric quality as measured by Chamfer Distance (2.48%) and F-Score (1.95%).
>
---
#### [new 043] Visual Generation Tuning
- **分类: cs.CV**

- **简介: 该论文提出VGT，一种在预训练视觉语言模型上进行高效视觉生成调优的新范式。针对现有模型生成能力不足的问题，通过解耦像素级VAE，构建语义对齐的VGT-AE，在图像重建与生成任务中均取得领先性能，显著提升生成速度与质量，赋予通用VLM视觉生成能力。**

- **链接: [https://arxiv.org/pdf/2511.23469v1](https://arxiv.org/pdf/2511.23469v1)**

> **作者:** Jiahao Guo; Sinan Du; Jingfeng Yao; Wenyu Liu; Bo Li; Haoxiang Cao; Kun Gai; Chun Yuan; Kai Wu; Xinggang Wang
>
> **摘要:** Large Vision Language Models (VLMs) effectively bridge the modality gap through extensive pretraining, acquiring sophisticated visual representations aligned with language. However, it remains underexplored whether these representations, optimized for multimodal understanding tasks, harbor an inherent potential for visual generation. In this paper, we propose VGT, Visual Generation Tuning, a novel paradigm designed to stimulate the underlying capabilities of visual generation within any vision language models. By performing efficient visual generation tuning on well-pretrained VLMs, we significantly mitigate the alignment costs and accelerate the convergence of autoregressive modeling in the continuous space (20x speedup). Specifically, we dismiss the entangled pixel-level VAEs designed for diffusion transformers and formulate VGT-AE through aligning the semantic encoders from pretrained VLMs with the latent representations of pixel decoders. In image reconstruction tasks, we achieve 26.67 PSNR and 0.50 rFID at a 28x compression ratio, outperforming specialized VAEs; in visual generation tasks, we achieve state-of-the-art outcomes among autoregressive models, 0.77 on GenEval and 78.73 on DPG-Bench. Furthermore, our proposed VGT showcases significant scaling promise and is versatile for endowing any VLMs trained for multimodal understanding with the capabilities of visual generation, which paves the new avenue to explore next-generation unified multimodal foundation models. Models and codes are available at https://github.com/hustvl/VGT.
>
---
#### [new 044] DisMo: Disentangled Motion Representations for Open-World Motion Transfer
- **分类: cs.CV**

- **简介: 该论文提出DisMo，一种从视频中学习解耦运动表示的新方法，解决现有生成模型难以分离运动与内容的问题。通过图像空间重建实现运动与外观解耦，支持跨类别、无对应关系的开放世界运动迁移，并可适配任意视频生成器。实验表明其在动作分类等下游任务上优于现有模型。**

- **链接: [https://arxiv.org/pdf/2511.23428v1](https://arxiv.org/pdf/2511.23428v1)**

> **作者:** Thomas Ressler-Antal; Frank Fundel; Malek Ben Alaya; Stefan Andreas Baumann; Felix Krause; Ming Gui; Björn Ommer
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Recent advances in text-to-video (T2V) and image-to-video (I2V) models, have enabled the creation of visually compelling and dynamic videos from simple textual descriptions or initial frames. However, these models often fail to provide an explicit representation of motion separate from content, limiting their applicability for content creators. To address this gap, we propose DisMo, a novel paradigm for learning abstract motion representations directly from raw video data via an image-space reconstruction objective. Our representation is generic and independent of static information such as appearance, object identity, or pose. This enables open-world motion transfer, allowing motion to be transferred across semantically unrelated entities without requiring object correspondences, even between vastly different categories. Unlike prior methods, which trade off motion fidelity and prompt adherence, are overfitting to source structure or drifting from the described action, our approach disentangles motion semantics from appearance, enabling accurate transfer and faithful conditioning. Furthermore, our motion representation can be combined with any existing video generator via lightweight adapters, allowing us to effortlessly benefit from future advancements in video models. We demonstrate the effectiveness of our method through a diverse set of motion transfer tasks. Finally, we show that the learned representations are well-suited for downstream motion understanding tasks, consistently outperforming state-of-the-art video representation models such as V-JEPA in zero-shot action classification on benchmarks including Something-Something v2 and Jester. Project page: https://compvis.github.io/DisMo
>
---
#### [new 045] TTSnap: Test-Time Scaling of Diffusion Models via Noise-Aware Pruning
- **分类: cs.CV**

- **简介: 该论文针对文本到图像扩散模型的测试时扩展问题，提出TTSnap框架。通过噪声感知奖励模型与自蒸馏、课程学习，实现对低质量候选样本的早期剪枝，避免全去噪计算，显著提升效率与性能，相较现有方法提升超16%。**

- **链接: [https://arxiv.org/pdf/2511.22242v1](https://arxiv.org/pdf/2511.22242v1)**

> **作者:** Qingtao Yu; Changlin Song; Minghao Sun; Zhengyang Yu; Vinay Kumar Verma; Soumya Roy; Sumit Negi; Hongdong Li; Dylan Campbell
>
> **摘要:** A prominent approach to test-time scaling for text-to-image diffusion models formulates the problem as a search over multiple noise seeds, selecting the one that maximizes a certain image-reward function. The effectiveness of this strategy heavily depends on the number and diversity of noise seeds explored. However, verifying each candidate is computationally expensive, because each must be fully denoised before a reward can be computed. This severely limits the number of samples that can be explored under a fixed budget. We propose test-time scaling with noise-aware pruning (TTSnap), a framework that prunes low-quality candidates without fully denoising them. The key challenge is that reward models are learned in the clean image domain, and the ranking of rewards predicted for intermediate estimates are often inconsistent with those predicted for clean images. To overcome this, we train noise-aware reward models via self-distillation to align the reward for intermediate estimates with that of the final clean images. To stabilize learning across different noise levels, we adopt a curriculum training strategy that progressively shifts the data domain from clean images to noise images. In addition, we introduce a new metric that measures reward alignment and computational budget utilization. Experiments demonstrate that our approach improves performance by over 16\% compared with existing methods, enabling more efficient and effective test-time scaling. It also provides orthogonal gains when combined with post-training techniques and local test-time optimization. Code: https://github.com/TerrysLearning/TTSnap/.
>
---
#### [new 046] Toward Automatic Safe Driving Instruction: A Large-Scale Vision Language Model Approach
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究自动驾驶中的安全驾驶指导任务，旨在通过多摄像头视频分析实现自动安全提醒。针对仅依赖道路视角的局限，提出融合驾驶员与道路双视角输入的方案，构建数据集并验证微调后视觉语言模型在生成精准安全指令上的有效性，揭示其对细微复杂事件检测的挑战。**

- **链接: [https://arxiv.org/pdf/2511.23311v1](https://arxiv.org/pdf/2511.23311v1)**

> **作者:** Haruki Sakajo; Hiroshi Takato; Hiroshi Tsutsui; Komei Soda; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted to MMLoSo 2025
>
> **摘要:** Large-scale Vision Language Models (LVLMs) exhibit advanced capabilities in tasks that require visual information, including object detection. These capabilities have promising applications in various industrial domains, such as autonomous driving. For example, LVLMs can generate safety-oriented descriptions of videos captured by road-facing cameras. However, ensuring comprehensive safety requires monitoring driver-facing views as well to detect risky events, such as the use of mobiles while driving. Thus, the ability to process synchronized inputs is necessary from both driver-facing and road-facing cameras. In this study, we develop models and investigate the capabilities of LVLMs by constructing a dataset and evaluating their performance on this dataset. Our experimental results demonstrate that while pre-trained LVLMs have limited effectiveness, fine-tuned LVLMs can generate accurate and safety-aware driving instructions. Nonetheless, several challenges remain, particularly in detecting subtle or complex events in the video. Our findings and error analysis provide valuable insights that can contribute to the improvement of LVLM-based systems in this domain.
>
---
#### [new 047] DualCamCtrl: Dual-Branch Diffusion Model for Geometry-Aware Camera-Controlled Video Generation
- **分类: cs.CV**

- **简介: 该论文提出DualCamCtrl，一种用于相机控制视频生成的双分支扩散模型。针对现有方法缺乏几何感知的问题，通过联合生成一致的RGB与深度序列，并引入语义引导的互对齐机制，增强场景理解。实验表明，该方法显著降低相机运动误差，提升生成视频的几何一致性。**

- **链接: [https://arxiv.org/pdf/2511.23127v1](https://arxiv.org/pdf/2511.23127v1)**

> **作者:** Hongfei Zhang; Kanghao Chen; Zixin Zhang; Harold Haodong Chen; Yuanhuiyi Lyu; Yuqi Zhang; Shuai Yang; Kun Zhou; Yingcong Chen
>
> **摘要:** This paper presents DualCamCtrl, a novel end-to-end diffusion model for camera-controlled video generation. Recent works have advanced this field by representing camera poses as ray-based conditions, yet they often lack sufficient scene understanding and geometric awareness. DualCamCtrl specifically targets this limitation by introducing a dual-branch framework that mutually generates camera-consistent RGB and depth sequences. To harmonize these two modalities, we further propose the Semantic Guided Mutual Alignment (SIGMA) mechanism, which performs RGB-depth fusion in a semantics-guided and mutually reinforced manner. These designs collectively enable DualCamCtrl to better disentangle appearance and geometry modeling, generating videos that more faithfully adhere to the specified camera trajectories. Additionally, we analyze and reveal the distinct influence of depth and camera poses across denoising stages and further demonstrate that early and late stages play complementary roles in forming global structure and refining local details. Extensive experiments demonstrate that DualCamCtrl achieves more consistent camera-controlled video generation, with over 40\% reduction in camera motion errors compared with prior methods. Our project page: https://soyouthinkyoucantell.github.io/dualcamctrl\-page/
>
---
#### [new 048] ViGG: Robust RGB-D Point Cloud Registration using Visual-Geometric Mutual Guidance
- **分类: cs.CV**

- **简介: 该论文针对RGB-D点云配准任务，解决现有方法仅依赖几何信息或简单融合图像特征导致鲁棒性不足的问题。提出ViGG方法，通过视觉-几何互引导机制，利用几何引导抑制模糊匹配，并以视觉先验约束搜索空间，提升配准精度与抗噪能力，在多个数据集上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22908v1](https://arxiv.org/pdf/2511.22908v1)**

> **作者:** Congjia Chen; Shen Yan; Yufu Qu
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Point cloud registration is a fundamental task in 3D vision. Most existing methods only use geometric information for registration. Recently proposed RGB-D registration methods primarily focus on feature fusion or improving feature learning, which limits their ability to exploit image information and hinders their practical applicability. In this paper, we propose ViGG, a robust RGB-D registration method using mutual guidance. First, we solve clique alignment in a visual-geometric combination form, employing a geometric guidance design to suppress ambiguous cliques. Second, to mitigate accuracy degradation caused by noise in visual matches, we propose a visual-guided geometric matching method that utilizes visual priors to determine the search space, enabling the extraction of high-quality, noise-insensitive correspondences. This mutual guidance strategy brings our method superior robustness, making it applicable for various RGB-D registration tasks. The experiments on 3DMatch, ScanNet and KITTI datasets show that our method outperforms recent state-of-the-art methods in both learning-free and learning-based settings. Code is available at https://github.com/ccjccjccj/ViGG.
>
---
#### [new 049] Alzheimer's Disease Prediction Using EffNetViTLoRA and BiLSTM with Multimodal Longitudinal MRI Data
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病早期预测任务，旨在区分轻度认知障碍（MCI）患者是否进展为AD。针对MCI转化不确定性问题，提出融合EffNetViTLoRA与BiLSTM的多模态模型，利用纵向MRI数据和非影像生物标志物，捕捉空间与时间特征，实现对48个月后认知状态的精准预测，准确率达95.05%。**

- **链接: [https://arxiv.org/pdf/2511.22774v1](https://arxiv.org/pdf/2511.22774v1)**

> **作者:** Mahdieh Behjat Khatooni; Mohsen Soryani
>
> **摘要:** Alzheimer's disease (AD) is a prevalent neurodegenerative disorder that progressively impairs memory, decision-making, and overall cognitive function. As AD is irreversible, early prediction is critical for timely intervention and management. Mild Cognitive Impairment (MCI), a transitional stage between cognitively normal (CN) aging and AD, plays a significant role in early AD diagnosis. However, predicting MCI progression remains a significant challenge, as not all individuals with MCI convert to AD. MCI subjects are categorized into stable MCI (sMCI) and progressive MCI (pMCI) based on conversion status. In this study, we propose a generalized, end-to-end deep learning model for AD prediction using MCI cases from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Our hybrid architecture integrates Convolutional Neural Networks and Vision Transformers to capture both local spatial features and global contextual dependencies from Magnetic Resonance Imaging (MRI) scans. To incorporate temporal progression, we further employ Bidirectional Long Short-Term Memory (BiLSTM) networks to process features extracted from four consecutive MRI timepoints along with some other non-image biomarkers, predicting each subject's cognitive status at month 48. Our multimodal model achieved an average progression prediction accuracy of 95.05\% between sMCI and pMCI, outperforming existing studies in AD prediction. This work demonstrates state-of-the-art performance in longitudinal AD prediction and highlights the effectiveness of combining spatial and temporal modeling for the early detection of Alzheimer's disease.
>
---
#### [new 050] DNA: Dual-branch Network with Adaptation for Open-Set Online Handwriting Generation
- **分类: cs.CV**

- **简介: 该论文针对开放集在线手写生成（OHG）任务，解决现有方法难以生成训练中未见字符的问题。提出双分支自适应网络DNA，分别学习书写风格与字符结构纹理，实现对未见字符和写作风格的泛化生成，显著提升真实感与适用性。**

- **链接: [https://arxiv.org/pdf/2511.22064v1](https://arxiv.org/pdf/2511.22064v1)**

> **作者:** Tsai-Ling Huang; Nhat-Tuong Do-Tran; Ngoc-Hoang-Lam Le; Hong-Han Shuai; Ching-Chun Huang
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Online handwriting generation (OHG) enhances handwriting recognition models by synthesizing diverse, human-like samples. However, existing OHG methods struggle to generate unseen characters, particularly in glyph-based languages like Chinese, limiting their real-world applicability. In this paper, we introduce our method for OHG, where the writer's style and the characters generated during testing are unseen during training. To tackle this challenge, we propose a Dual-branch Network with Adaptation (DNA), which comprises an adaptive style branch and an adaptive content branch. The style branch learns stroke attributes such as writing direction, spacing, placement, and flow to generate realistic handwriting. Meanwhile, the content branch is designed to generalize effectively to unseen characters by decomposing character content into structural information and texture details, extracted via local and global encoders, respectively. Extensive experiments demonstrate that our DNA model is well-suited for the unseen OHG setting, achieving state-of-the-art performance.
>
---
#### [new 051] Object-Centric Data Synthesis for Category-level Object Detection
- **分类: cs.CV**

- **简介: 该论文研究类别级物体检测任务，针对新类别缺乏标注数据的问题，提出基于物体中心数据（多视角图像或3D模型）的合成方法。通过图像处理、3D渲染和扩散模型生成复杂场景图像，提升模型在数据受限下的泛化能力，显著改善检测性能。**

- **链接: [https://arxiv.org/pdf/2511.23450v1](https://arxiv.org/pdf/2511.23450v1)**

> **作者:** Vikhyat Agarwal; Jiayi Cora Guo; Declan Hoban; Sissi Zhang; Nicholas Moran; Peter Cho; Srilakshmi Pattabiraman; Shantanu Joshi
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Deep learning approaches to object detection have achieved reliable detection of specific object classes in images. However, extending a model's detection capability to new object classes requires large amounts of annotated training data, which is costly and time-consuming to acquire, especially for long-tailed classes with insufficient representation in existing datasets. Here, we introduce the object-centric data setting, when limited data is available in the form of object-centric data (multi-view images or 3D models), and systematically evaluate the performance of four different data synthesis methods to finetune object detection models on novel object categories in this setting. The approaches are based on simple image processing techniques, 3D rendering, and image diffusion models, and use object-centric data to synthesize realistic, cluttered images with varying contextual coherence and complexity. We assess how these methods enable models to achieve category-level generalization in real-world data, and demonstrate significant performance boosts within this data-constrained experimental setting.
>
---
#### [new 052] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗视觉问答中诊断推理不准确的问题，提出MedEyes框架。通过引入离线专家引导的动态视觉聚焦机制，结合双模式探索与自适应采样策略，实现更贴近临床的渐进式诊断推理，显著提升模型在多基准上的表现。**

- **链接: [https://arxiv.org/pdf/2511.22018v1](https://arxiv.org/pdf/2511.22018v1)**

> **作者:** Chunzheng Zhu; Yangfang Lin; Shen Chen; Yijun Wang; Jianxin Lin
>
> **备注:** This paper has been accepted by AAAI 2026
>
> **摘要:** Accurate medical diagnosis often involves progressive visual focusing and iterative reasoning, characteristics commonly observed in clinical workflows. While recent vision-language models demonstrate promising chain-of-thought (CoT) reasoning capabilities via reinforcement learning with verifiable rewards (RLVR), their purely on-policy learning paradigm tends to reinforce superficially coherent but clinically inaccurate reasoning paths. We propose MedEyes, a novel reinforcement learning framework that dynamically models clinician-style diagnostic reasoning by progressively attending to and interpreting relevant medical image regions. By incorporating off-policy expert guidance, MedEyes converts expert visual search trajectories into structured external behavioral signals, guiding the model toward clinically aligned visual reasoning. We design the Gaze-guided Reasoning Navigator (GRN) to emulate the diagnostic process through a dual-mode exploration strategy, scanning for systematic abnormality localization and drilling for detailed regional analysis. To balance expert imitation and autonomous discovery, we introduce the Confidence Value Sampler (CVS), which employs nucleus sampling and adaptive termination to create diverse yet credible exploration paths. Finally, the dual-stream GRPO optimization framework decouples on-policy and off-policy learning signals, mitigating reward assimilation and entropy collapse. Experiments demonstrate that MedEyes achieves an average performance improvement of +8.5\% across multiple medical VQA benchmarks, validating MedEyes's potential in building interpretable medical AI systems.
>
---
#### [new 053] Controllable 3D Object Generation with Single Image Prompt
- **分类: cs.CV**

- **简介: 该论文聚焦于单图驱动的可控3D物体生成任务。针对现有方法依赖文本反转导致训练耗时且控制力弱的问题，提出无需文本反转的图像适配器与深度条件预热策略，提升生成控制性与3D一致性。实验与用户研究验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22194v1](https://arxiv.org/pdf/2511.22194v1)**

> **作者:** Jaeseok Lee; Jaekoo Lee
>
> **摘要:** Recently, the impressive generative capabilities of diffusion models have been demonstrated, producing images with remarkable fidelity. Particularly, existing methods for the 3D object generation tasks, which is one of the fastest-growing segments in computer vision, pre-dominantly use text-to-image diffusion models with textual inversion which train a pseudo text prompt to describe the given image. In practice, various text-to-image generative models employ textual inversion to learn concepts or styles of target object in the pseudo text prompt embedding space, thereby generating sophisticated outputs. However, textual inversion requires additional training time and lacks control ability. To tackle this issues, we propose two innovative methods: (1) using an off-the-shelf image adapter that generates 3D objects without textual inversion, offering enhanced control over conditions such as depth, pose, and text. (2) a depth conditioned warmup strategy to enhance 3D consistency. In experimental results, ours show qualitatively and quantitatively comparable performance and improved 3D consistency to the existing text-inversion-based alternatives. Furthermore, we conduct a user study to assess (i) how well results match the input image and (ii) whether 3D consistency is maintained. User study results show that our model outperforms the alternatives, validating the effectiveness of our approaches. Our code is available at GitHub repository:https://github.com/Seooooooogi/Control3D_IP/
>
---
#### [new 054] AnyTalker: Scaling Multi-Person Talking Video Generation with Interactivity Refinement
- **分类: cs.CV**

- **简介: 该论文针对多人物对话视频生成中数据成本高、交互自然性难的问题，提出AnyTalker框架。通过身份感知注意力机制实现任意扩展的驱动身份，仅用单人视频训练并少量多人片段优化交互性，显著降低数据依赖，提升生成视频的唇同步、视觉质量与自然交互性。**

- **链接: [https://arxiv.org/pdf/2511.23475v1](https://arxiv.org/pdf/2511.23475v1)**

> **作者:** Zhizhou Zhong; Yicheng Ji; Zhe Kong; Yiying Liu; Jiarui Wang; Jiasun Feng; Lupeng Liu; Xiangyi Wang; Yanjia Li; Yuqing She; Ying Qin; Huan Li; Shuiyang Mao; Wei Liu; Wenhan Luo
>
> **备注:** Homepage: https://hkust-c4g.github.io/AnyTalker-homepage
>
> **摘要:** Recently, multi-person video generation has started to gain prominence. While a few preliminary works have explored audio-driven multi-person talking video generation, they often face challenges due to the high costs of diverse multi-person data collection and the difficulty of driving multiple identities with coherent interactivity. To address these challenges, we propose AnyTalker, a multi-person generation framework that features an extensible multi-stream processing architecture. Specifically, we extend Diffusion Transformer's attention block with a novel identity-aware attention mechanism that iteratively processes identity-audio pairs, allowing arbitrary scaling of drivable identities. Besides, training multi-person generative models demands massive multi-person data. Our proposed training pipeline depends solely on single-person videos to learn multi-person speaking patterns and refines interactivity with only a few real multi-person clips. Furthermore, we contribute a targeted metric and dataset designed to evaluate the naturalness and interactivity of the generated multi-person videos. Extensive experiments demonstrate that AnyTalker achieves remarkable lip synchronization, visual quality, and natural interactivity, striking a favorable balance between data costs and identity scalability.
>
---
#### [new 055] Artwork Interpretation with Vision Language Models: A Case Study on Emotions and Emotion Symbols
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）在艺术作品情绪与情绪符号解读中的能力。针对情绪表达抽象性及符号识别难题，通过案例分析三种VLMs对四类递进问题的回答，并结合专家评估，发现模型对具体图像表现良好，但在抽象/象征性图像上表现不佳，且存在回答不一致问题。**

- **链接: [https://arxiv.org/pdf/2511.22929v1](https://arxiv.org/pdf/2511.22929v1)**

> **作者:** Sebastian Padó; Kerstin Thomas
>
> **备注:** Accepted for publication at the IJCNLP-AACL workshop on Multimodal Models for Low-Resource Contexts and Social Impact
>
> **摘要:** Emotions are a fundamental aspect of artistic expression. Due to their abstract nature, there is a broad spectrum of emotion realization in artworks. These are subject to historical change and their analysis requires expertise in art history. In this article, we investigate which aspects of emotional expression can be detected by current (2025) vision language models (VLMs). We present a case study of three VLMs (Llava-Llama and two Qwen models) in which we ask these models four sets of questions of increasing complexity about artworks (general content, emotional content, expression of emotions, and emotion symbols) and carry out a qualitative expert evaluation. We find that the VLMs recognize the content of the images surprisingly well and often also which emotions they depict and how they are expressed. The models perform best for concrete images but fail for highly abstract or highly symbolic images. Reliable recognition of symbols remains fundamentally difficult. Furthermore, the models continue to exhibit the well-known LLM weakness of providing inconsistent answers to related questions.
>
---
#### [new 056] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation
- **分类: cs.CV**

- **简介: 该论文针对单图3D生成任务，解决现有模型依赖哪些图像线索的问题。提出Cue3D框架，通过系统扰动七类视觉线索，量化其对3D输出的影响，发现几何线索（如阴影）更关键，且模型存在对轮廓的过度依赖，揭示了模型泛化与形状意义的关系。**

- **链接: [https://arxiv.org/pdf/2511.22121v1](https://arxiv.org/pdf/2511.22121v1)**

> **作者:** Xiang Li; Zirui Wang; Zixuan Huang; James M. Rehg
>
> **备注:** NeurIPS 2025 Highlight; Project page:https://ryanxli.github.io/cue3d/index.html
>
> **摘要:** Humans and traditional computer vision methods rely on a diverse set of monocular cues to infer 3D structure from a single image, such as shading, texture, silhouette, etc. While recent deep generative models have dramatically advanced single-image 3D generation, it remains unclear which image cues these methods actually exploit. We introduce Cue3D, the first comprehensive, model-agnostic framework for quantifying the influence of individual image cues in single-image 3D generation. Our unified benchmark evaluates seven state-of-the-art methods, spanning regression-based, multi-view, and native 3D generative paradigms. By systematically perturbing cues such as shading, texture, silhouette, perspective, edges, and local continuity, we measure their impact on 3D output quality. Our analysis reveals that shape meaningfulness, not texture, dictates generalization. Geometric cues, particularly shading, are crucial for 3D generation. We further identify over-reliance on provided silhouettes and diverse sensitivities to cues such as perspective and local continuity across model families. By dissecting these dependencies, Cue3D advances our understanding of how modern 3D networks leverage classical vision cues, and offers directions for developing more transparent, robust, and controllable single-image 3D generation models.
>
---
#### [new 057] Small Object Detection for Birds with Swin Transformer
- **分类: cs.CV**

- **简介: 该论文聚焦小目标检测任务，针对鸟类等小而稀疏的目标，提出基于Swin Transformer的颈部结构改进方法。通过调整移位窗口大小，增强特征提取能力，提升小物体检测性能，实验表明更小窗口（如2）有助于提高mAP。**

- **链接: [https://arxiv.org/pdf/2511.22310v1](https://arxiv.org/pdf/2511.22310v1)**

> **作者:** Da Huo; Marc A. Kastner; Tingwei Liu; Yasutomo Kawanishi; Takatsugu Hirayama; Takahiro Komamizu; Ichiro Ide
>
> **备注:** This paper is included in the proceedings of the 18th International Conference on Machine Vision Applications (MVA2023) (https://www.mva-org.jp/mva2023/challenge/index) The paper has received Runner-Up Solution Award (2nd) and Best Booster Award from Small Object Detection Challenge for Spotting Birds 2023 in MVA
>
> **摘要:** Object detection is the task of detecting objects in an image. In this task, the detection of small objects is particularly difficult. Other than the small size, it is also accompanied by difficulties due to blur, occlusion, and so on. Current small object detection methods are tailored to small and dense situations, such as pedestrians in a crowd or far objects in remote sensing scenarios. However, when the target object is small and sparse, there is a lack of objects available for training, making it more difficult to learn effective features. In this paper, we propose a specialized method for detecting a specific category of small objects; birds. Particularly, we improve the features learned by the neck; the sub-network between the backbone and the prediction head, to learn more effective features with a hierarchical design. We employ Swin Transformer to upsample the image features. Moreover, we change the shifted window size for adapting to small objects. Experiments show that the proposed Swin Transformer-based neck combined with CenterNet can lead to good performance by changing the window sizes. We further find that smaller window sizes (default 2) benefit mAPs for small object detection.
>
---
#### [new 058] AmodalGen3D: Generative Amodal 3D Object Reconstruction from Sparse Unposed Views
- **分类: cs.CV**

- **简介: 该论文提出AmodalGen3D，解决从稀疏无姿态视角中重建完整3D物体的问题。针对部分遮挡导致的几何不完整与不一致问题，模型融合2D可见性先验与多视图立体几何，通过注意力机制实现稀疏视图特征融合与未观测结构推断，实现高保真、完整的3D重建。**

- **链接: [https://arxiv.org/pdf/2511.21945v1](https://arxiv.org/pdf/2511.21945v1)**

> **作者:** Junwei Zhou; Yu-Wing Tai
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** Reconstructing 3D objects from a few unposed and partially occluded views is a common yet challenging problem in real-world scenarios, where many object surfaces are never directly observed. Traditional multi-view or inpainting-based approaches struggle under such conditions, often yielding incomplete or geometrically inconsistent reconstructions. We introduce AmodalGen3D, a generative framework for amodal 3D object reconstruction that infers complete, occlusion-free geometry and appearance from arbitrary sparse inputs. The model integrates 2D amodal completion priors with multi-view stereo geometry conditioning, supported by a View-Wise Cross Attention mechanism for sparse-view feature fusion and a Stereo-Conditioned Cross Attention module for unobserved structure inference. By jointly modeling visible and hidden regions, AmodalGen3D faithfully reconstructs 3D objects that are consistent with sparse-view constraints while plausibly hallucinating unseen parts. Experiments on both synthetic and real-world datasets demonstrate that AmodalGen3D achieves superior fidelity and completeness under occlusion-heavy sparse-view settings, addressing a pressing need for object-level 3D scene reconstruction in robotics, AR/VR, and embodied AI applications.
>
---
#### [new 059] Do You See What I Say? Generalizable Deepfake Detection based on Visual Speech Recognition
- **分类: cs.CV**

- **简介: 该论文聚焦于通用化深度伪造检测任务，旨在解决现有方法在未见生成技术下泛化能力差的问题。提出FauxNet模型，利用预训练视觉语音识别特征提取视频时序信息，实现零样本检测，并能区分不同生成技术。构建了两个新数据集，验证了模型优越性。**

- **链接: [https://arxiv.org/pdf/2511.22443v1](https://arxiv.org/pdf/2511.22443v1)**

> **作者:** Maheswar Bora; Tashvik Dhamija; Shukesh Reddy; Baptiste Chopin; Pranav Balaji; Abhijit Das; Antitza Dantcheva
>
> **摘要:** Deepfake generation has witnessed remarkable progress, contributing to highly realistic generated images, videos, and audio. While technically intriguing, such progress has raised serious concerns related to the misuse of manipulated media. To mitigate such misuse, robust and reliable deepfake detection is urgently needed. Towards this, we propose a novel network FauxNet, which is based on pre-trained Visual Speech Recognition (VSR) features. By extracting temporal VSR features from videos, we identify and segregate real videos from manipulated ones. The holy grail in this context has to do with zero-shot detection, i.e., generalizable detection, which we focus on in this work. FauxNet consistently outperforms the state-of-the-art in this setting. In addition, FauxNet is able to attribute - distinguish between generation techniques from which the videos stem. Finally, we propose new datasets, referred to as Authentica-Vox and Authentica-HDTF, comprising about 38,000 real and fake videos in total, the latter created with six recent deepfake generation techniques. We provide extensive analysis and results on the Authentica datasets and FaceForensics++, demonstrating the superiority of FauxNet. The Authentica datasets will be made publicly available.
>
---
#### [new 060] EASL: Multi-Emotion Guided Semantic Disentanglement for Expressive Sign Language Generation
- **分类: cs.CV**

- **简介: 该论文针对手语生成中情感表达缺失的问题，提出EASL框架。通过多情感引导的语义解耦，分离并融合情感与语义特征，在姿态解码中引入7类情感置信度，提升生成手语视频的情感自然性与表现力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22135v1](https://arxiv.org/pdf/2511.22135v1)**

> **作者:** Yanchao Zhao; Jihao Zhu; Yu Liu; Weizhuo Chen; Yuling Yang; Kun Peng
>
> **摘要:** Large language models have revolutionized sign language generation by automatically transforming text into high-quality sign language videos, providing accessible communication for the Deaf community. However, existing LLM-based approaches prioritize semantic accuracy while overlooking emotional expressions, resulting in outputs that lack naturalness and expressiveness. We propose EASL (Emotion-Aware Sign Language), a multi-emotion-guided generation architecture for fine-grained emotional integration. We introduce emotion-semantic disentanglement modules with progressive training to separately extract semantic and affective features. During pose decoding, the emotional representations guide semantic interaction to generate sign poses with 7-class emotion confidence scores, enabling emotional expression recognition. Experimental results demonstrate that EASL achieves pose accuracy superior to all compared baselines by integrating multi-emotion information and effectively adapts to diffusion models to generate expressive sign language videos.
>
---
#### [new 061] PointCNN++: Performant Convolution on Native Points
- **分类: cs.CV**

- **简介: 该论文针对3D点云处理中几何精度与计算效率的矛盾，提出PointCNN++。它将稀疏卷积推广至原始点坐标，通过原生点上的矩阵-向量乘法与归约（MVMR）设计高效GPU核，显著降低内存占用并提升速度，同时保持高几何精度，有效改善点云配准性能。**

- **链接: [https://arxiv.org/pdf/2511.23227v1](https://arxiv.org/pdf/2511.23227v1)**

> **作者:** Lihan Li; Haofeng Zhong; Rui Bu; Mingchao Sun; Wenzheng Chen; Baoquan Chen; Yangyan Li
>
> **摘要:** Existing convolutional learning methods for 3D point cloud data are divided into two paradigms: point-based methods that preserve geometric precision but often face performance challenges, and voxel-based methods that achieve high efficiency through quantization at the cost of geometric fidelity. This loss of precision is a critical bottleneck for tasks such as point cloud registration. We propose PointCNN++, a novel architectural design that fundamentally mitigates this precision-performance trade-off. It \textbf{generalizes sparse convolution from voxels to points}, treating voxel-based convolution as a specialized, degraded case of our more general point-based convolution. First, we introduce a point-centric convolution where the receptive field is centered on the original, high-precision point coordinates. Second, to make this high-fidelity operation performant, we design a computational strategy that operates \textbf{natively} on points. We formulate the convolution on native points as a Matrix-Vector Multiplication and Reduction (MVMR) problem, for which we develop a dedicated, highly-optimized GPU kernel. Experiments demonstrate that PointCNN++ \textbf{uses an order of magnitude less memory and is several times faster} than representative point-based methods. Furthermore, when used as a simple replacement for the voxel-based backbones it generalizes, it \textbf{significantly improves point cloud registration accuracies while proving both more memory-efficient and faster}. PointCNN++ shows that preserving geometric detail and achieving high performance are not mutually exclusive, paving the way for a new class of 3D learning with high fidelity and efficiency. Our code will be open sourced.
>
---
#### [new 062] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model
- **分类: cs.CV**

- **简介: 该论文提出SparseWorld-TC，一种轨迹条件下的3D场景占用预测模型。针对传统方法依赖VAE生成离散占用令牌导致表征能力受限，以及鸟瞰图投影带来的结构限制问题，提出基于Transformer的稀疏占用表示，直接从图像特征端到端预测多帧未来占用，有效捕捉时空依赖，显著提升nuScenes数据集上1-3秒预测性能。**

- **链接: [https://arxiv.org/pdf/2511.22039v1](https://arxiv.org/pdf/2511.22039v1)**

> **作者:** Jiayuan Du; Yiming Zhao; Zhenglong Guo; Yong Pan; Wenbo Hou; Zhihui Hao; Kun Zhan; Qijun Chen
>
> **摘要:** This paper introduces a novel architecture for trajectory-conditioned forecasting of future 3D scene occupancy. In contrast to methods that rely on variational autoencoders (VAEs) to generate discrete occupancy tokens, which inherently limit representational capacity, our approach predicts multi-frame future occupancy in an end-to-end manner directly from raw image features. Inspired by the success of attention-based transformer architectures in foundational vision and language models such as GPT and VGGT, we employ a sparse occupancy representation that bypasses the intermediate bird's eye view (BEV) projection and its explicit geometric priors. This design allows the transformer to capture spatiotemporal dependencies more effectively. By avoiding both the finite-capacity constraint of discrete tokenization and the structural limitations of BEV representations, our method achieves state-of-the-art performance on the nuScenes benchmark for 1-3 second occupancy forecasting, outperforming existing approaches by a significant margin. Furthermore, it demonstrates robust scene dynamics understanding, consistently delivering high accuracy under arbitrary future trajectory conditioning.
>
---
#### [new 063] VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出VQRAE，一种统一多模态理解、生成与重建的表示自编码器。针对现有方法在单一分词器下难以兼顾语义理解与生成的问题，创新性地采用高维向量量化码本，实现连续语义特征与离散生成令牌的统一表征。通过两阶段训练，在保持理解性能的同时支持高质量生成与重建。**

- **链接: [https://arxiv.org/pdf/2511.23386v1](https://arxiv.org/pdf/2511.23386v1)**

> **作者:** Sinan Du; Jiahao Guo; Bo Li; Shuhao Cui; Zhengzhuo Xu; Yifu Luo; Yongxian Wei; Kun Gai; Xinggang Wang; Kai Wu; Chun Yuan
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Unifying multimodal understanding, generation and reconstruction representation in a single tokenizer remains a key challenge in building unified models. Previous research predominantly attempts to address this in a dual encoder paradigm, e.g., utilizing the separate encoders for understanding and generation respectively or balancing semantic representations and low-level features with contrastive loss. In this paper, we propose VQRAE, a Vector Quantization version of Representation AutoEncoders, which pioneers the first exploration in unified representation to produce Continuous semantic features for image understanding and Discrete tokens for visual generation within a unified tokenizer. Specifically, we build upon pretrained vision foundation models with a symmetric ViT decoder and adopt a two-stage training strategy: first, it freezes the encoder and learns a high-dimensional semantic VQ codebook with pixel reconstruction objective; then jointly optimizes the encoder with self-distillation constraints. This design enables negligible semantic information for maintaining the ability of multimodal understanding, discrete tokens that are compatible for generation and fine-grained reconstruction. Besides, we identify the intriguing property in quantizing semantic encoders that rely on high-dimensional codebook in contrast to the previous common practice of low-dimensional codebook in image reconstruction. The semantic VQ codebook can achieve a 100% utilization ratio at a dimension of 1536. VQRAE presents competitive performance on several benchmarks of visual understanding, generation and reconstruction with promising scaling property in the autoregressive paradigm for its discrete merits.
>
---
#### [new 064] Some Modalities are More Equal Than Others: Decoding and Architecting Multimodal Integration in MLLMs
- **分类: cs.CV**

- **简介: 该论文研究多模态大模型（MLLMs）在模态冲突下的鲁棒性问题，提出MMA-Bench基准测试。通过黑箱与白箱可解释性分析，揭示现有模型在音视频不匹配或误导文本下表现脆弱。为此，提出模态对齐调优策略，提升模型跨模态推理的可靠性，推动更稳健的多模态融合。**

- **链接: [https://arxiv.org/pdf/2511.22826v1](https://arxiv.org/pdf/2511.22826v1)**

> **作者:** Tianle Chen; Chaitanya Chakka; Arjun Reddy Akula; Xavier Thomas; Deepti Ghadiyaram
>
> **摘要:** Despite remarkable advancements in Multimodal Large Language Models (MLLMs), a fundamental question remains: are MLLMs robust to contradicting modalities? To rigorously study this, we introduce MMA-Bench comprising videos and tasks that probe a model's reliance on specific modalities. Using black-box and white-box interpretability techniques, we provide a critical analysis of the brittleness of both open- and closed-sourced MLLMs. We show that current MLLMs struggle under misaligned audio-visual pairs and simple misleading text, thereby lacking robust multi-modal reasoning. Building on these findings, we propose a modality alignment tuning strategy to teach the model when to prioritize, leverage, or ignore specific modality cues. Through extensive experiments and analysis, we show that our alignment tuning yields demonstrably stronger multimodal grounding. This work provides both interpretability tools and a clear path toward developing MLLMs with intrinsically reliable cross-modal reasoning. Code and dataset will be publicly available.
>
---
#### [new 065] AnoRefiner: Anomaly-Aware Group-Wise Refinement for Zero-Shot Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对零样本工业异常检测（ZSAD）中异常定位粗粒度的问题，提出AnoRefiner。通过引入异常感知的细化解码器（ARD）和分组渐进测试时训练策略（PGT），利用异常得分图提升特征，实现从块级到像素级的精细异常定位，显著改善检测精度。**

- **链接: [https://arxiv.org/pdf/2511.22595v1](https://arxiv.org/pdf/2511.22595v1)**

> **作者:** Dayou Huang; Feng Xue; Xurui Li; Yu Zhou
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Zero-shot industrial anomaly detection (ZSAD) methods typically yield coarse anomaly maps as vision transformers (ViTs) extract patch-level features only. To solve this, recent solutions attempt to predict finer anomalies using features from ZSAD, but they still struggle to recover fine-grained anomalies without missed detections, mainly due to the gap between randomly synthesized training anomalies and real ones. We observe that anomaly score maps exactly provide complementary spatial cues that are largely absent from ZSAD's image features, a fact overlooked before. Inspired by this, we propose an anomaly-aware refiner (AnoRefiner) that can be plugged into most ZSAD models and improve patch-level anomaly maps to the pixel level. First, we design an anomaly refinement decoder (ARD) that progressively enhances image features using anomaly score maps, reducing the reliance on synthetic anomaly data. Second, motivated by the mass production paradigm, we propose a progressive group-wise test-time training (PGT) strategy that trains ARD in each product group for the refinement process in the next group, while staying compatible with any ZSAD method. Experiments on the MVTec AD and VisA datasets show that AnoRefiner boosts various ZSAD models by up to a 5.2\% gain in pixel-AP metrics, which can also be directly observed in many visualizations. The code will be available at https://github.com/HUST-SLOW/AnoRefiner.
>
---
#### [new 066] Bridging 3D Deep Learning and Curation for Analysis and High-Quality Segmentation in Practice
- **分类: cs.CV**

- **简介: 该论文针对3D显微图像分割中误差频发的问题，提出VessQC工具，通过不确定性图引导用户高效修正错误。解决了基础模型结果不可靠、人工校正效率低的难题，实现了人机协同的高精度分割，推动了真实场景下3D生物图像分析的应用。**

- **链接: [https://arxiv.org/pdf/2511.22236v1](https://arxiv.org/pdf/2511.22236v1)**

> **作者:** Simon Püttmann; Jonathan Jair Sànchez Contreras; Lennart Kowitz; Peter Lampen; Saumya Gupta; Davide Panzeri; Nina Hagemann; Qiaojie Xiong; Dirk M. Hermann; Cao Chen; Jianxu Chen
>
> **摘要:** Accurate 3D microscopy image segmentation is critical for quantitative bioimage analysis but even state-of-the-art foundation models yield error-prone results. Therefore, manual curation is still widely used for either preparing high-quality training data or fixing errors before analysis. We present VessQC, an open-source tool for uncertainty-guided curation of large 3D microscopy segmentations. By integrating uncertainty maps, VessQC directs user attention to regions most likely containing biologically meaningful errors. In a preliminary user study uncertainty-guided correction significantly improved error detection recall from 67% to 94.0% (p=0.007) without a significant increase in total curation time. VessQC thus enables efficient, human-in-the-loop refinement of volumetric segmentations and bridges a key gap in real-world applications between uncertainty estimation and practical human-computer interaction. The software is freely available at github.com/MMV-Lab/VessQC.
>
---
#### [new 067] GOATex: Geometry & Occlusion-Aware Texturing
- **分类: cs.CV**

- **简介: 该论文提出GOATex，一种基于扩散模型的3D网格纹理生成方法。针对现有方法无法有效处理遮挡内部区域的问题，引入射线投射的可见性层级机制，分层渐进式生成内外表面纹理，并通过软UV融合实现无缝衔接，无需微调模型，支持内外分别提示，提升纹理完整性和质量。**

- **链接: [https://arxiv.org/pdf/2511.23051v1](https://arxiv.org/pdf/2511.23051v1)**

> **作者:** Hyunjin Kim; Kunho Kim; Adam Lee; Wonkwang Lee
>
> **备注:** Accepted to NeurIPS 2025; Project Page: https://goatex3d.github.io/
>
> **摘要:** We present GOATex, a diffusion-based method for 3D mesh texturing that generates high-quality textures for both exterior and interior surfaces. While existing methods perform well on visible regions, they inherently lack mechanisms to handle occluded interiors, resulting in incomplete textures and visible seams. To address this, we introduce an occlusion-aware texturing framework based on the concept of hit levels, which quantify the relative depth of mesh faces via multi-view ray casting. This allows us to partition mesh faces into ordered visibility layers, from outermost to innermost. We then apply a two-stage visibility control strategy that progressively reveals interior regions with structural coherence, followed by texturing each layer using a pretrained diffusion model. To seamlessly merge textures obtained across layers, we propose a soft UV-space blending technique that weighs each texture's contribution based on view-dependent visibility confidence. Empirical results demonstrate that GOATex consistently outperforms existing methods, producing seamless, high-fidelity textures across both visible and occluded surfaces. Unlike prior works, GOATex operates entirely without costly fine-tuning of a pretrained diffusion model and allows separate prompting for exterior and interior mesh regions, enabling fine-grained control over layered appearances. For more qualitative results, please visit our project page: https://goatex3d.github.io/.
>
---
#### [new 068] From Compound Figures to Composite Understanding: Developing a Multi-Modal LLM from Biomedical Literature with Medical Multiple-Image Benchmarking and Validation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对医疗多模态大模型在多图像理解上的不足，提出基于生物医学文献中复合图像的五阶段指令生成框架，构建M3LLM模型，实现跨模态、时空关系的综合理解。通过自建专家验证的PMC-MI-Bench基准，验证其在多图像分析中的卓越性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22232v1](https://arxiv.org/pdf/2511.22232v1)**

> **作者:** Zhen Chen; Yihang Fu; Gabriel Madera; Mauro Giuffre; Serina Applebaum; Hyunjae Kim; Hua Xu; Qingyu Chen
>
> **摘要:** Multi-modal large language models (MLLMs) have shown promise in advancing healthcare. However, most existing models remain confined to single-image understanding, which greatly limits their applicability in clinical workflows. In practice, medical diagnosis and progression often require synthesizing information across multiple images from different modalities or time points. The development of medical MLLMs capable of such multi-image understanding has been hindered by the lack of large-scale, high-quality annotated training data. To address this limitation, we propose a novel framework that leverages license-permissive compound images in biomedical literature, as a rich yet underutilized data source for multi-image analysis. Specifically, we design a five-stage, context-aware instruction generation paradigm underpinned by a divide-and-conquer strategy. By decomposing multi-image analysis into manageable sub-tasks, this paradigm empowers MLLMs to move beyond single-panel analysis and provide a composite understanding by learning the complex spatial, temporal, and cross-modal relationships inherent in these compound figures. By parsing over 237,000 compound figures and their contextual text for instruction generation, we develop M3LLM, a medical multi-image multi-modal large language model. For benchmarking, we construct PMC-MI-Bench for composite understanding, manually validated by medical experts. Extensive experiments show that M3LLM significantly outperforms both general-purpose and specialized medical MLLMs across multi-image, single-image, text-only, and multi-choice scenarios. Notably, M3LLM exhibits strong generalization to longitudinal chest X-ray analysis using the MIMIC dataset. This work establishes a scalable and efficient paradigm for developing medical MLLMs capable of composite reasoning, bridging the gap between biomedical literature and real-world clinical applications.
>
---
#### [new 069] Image Valuation in NeRF-based 3D reconstruction
- **分类: cs.CV**

- **简介: 该论文研究NeRF-based 3D重建中图像的贡献度评估问题。针对输入图像质量不一导致重建效果差异的问题，提出基于PSNR和MSE的图像估值方法，通过移除低贡献图像验证其对重建质量的影响，实现图像价值量化与优化。**

- **链接: [https://arxiv.org/pdf/2511.23052v1](https://arxiv.org/pdf/2511.23052v1)**

> **作者:** Grigorios Aris Cheimariotis; Antonis Karakottas; Vangelis Chatzis; Angelos Kanlis; Dimitrios Zarpalas
>
> **备注:** Published In International Conference on Computer Analysis of Images and Patterns (pp. 375-385). Cham: Springer Nature Switzerland
>
> **摘要:** Data valuation and monetization are becoming increasingly important across domains such as eXtended Reality (XR) and digital media. In the context of 3D scene reconstruction from a set of images -- whether casually or professionally captured -- not all inputs contribute equally to the final output. Neural Radiance Fields (NeRFs) enable photorealistic 3D reconstruction of scenes by optimizing a volumetric radiance field given a set of images. However, in-the-wild scenes often include image captures of varying quality, occlusions, and transient objects, resulting in uneven utility across inputs. In this paper we propose a method to quantify the individual contribution of each image to NeRF-based reconstructions of in-the-wild image sets. Contribution is assessed through reconstruction quality metrics based on PSNR and MSE. We validate our approach by removing low-contributing images during training and measuring the resulting impact on reconstruction fidelity.
>
---
#### [new 070] From Points to Clouds: Learning Robust Semantic Distributions for Multi-modal Prompts
- **分类: cs.CV**

- **简介: 该论文针对多模态提示学习中静态点表示导致泛化能力差的问题，提出P2C框架，通过扩散模型思想将提示学习重构为动态去噪任务。引入双去噪机制，学习语义分布而非单一点，提升对新类别和模糊类别的泛化性能，在11个数据集上显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.22897v1](https://arxiv.org/pdf/2511.22897v1)**

> **作者:** Weiran Li; Yeqiang Liu; Yijie Wei; Mina Han; Xin Liu; Zhenbo Li
>
> **摘要:** Multimodal Prompt Learning (MPL) has emerged as a pivotal technique for adapting large-scale Visual Language Models (VLMs). However, current MPL methods are fundamentally limited by their optimization of a single, static point representation. This paradigm is inherently brittle, leads to overfitting on base classes, and generalizes poorly to novel or ambiguous categories. We challenge this point paradigm, proposing that robust generalization requires learning a semantic cloud (i.e., a distribution over the embedding space). To achieve this, we introduce Points-to-Clouds (P2C), a novel framework inspired by diffusion models that reframes prompt learning as a dynamic denoising task. At the core of P2C is a dual denoising mechanism: a Dynamic Prompt Denoising (DPD) mechanism perturbs text prompts with sophisticated, annealed noise to learn a smoother semantic landscape, while an auxiliary V-L Mapper denoising loss re-tasks the mapper as a denoising autoencoder. This forces the mapper to reconstruct clean visual prompts from noisy text inputs, ensuring robust cross-modal alignment. Extensive experiments across 11 datasets demonstrate that P2C consistently outperforms strong baselines. On the base-to-novel generalization benchmark, our method achieves a Harmonic Mean of 79.7%, representing a relative improvement of 1.4% over the baseline. The code and models are available at https://vranlee.github.io/P2C/.
>
---
#### [new 071] Autonomous labeling of surgical resection margins using a foundation model
- **分类: cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文针对病理切片中手术切缘定位难题，提出虚拟染色网络（VIN），利用冻结的预训练模型提取特征，结合轻量分类器实现无需物理染色的自动切缘标注。通过120例扁桃体组织全幻灯片图像数据训练与验证，实现了73.3%的区域准确率，可为数字病理提供标准化、可重复的切缘分析方案。**

- **链接: [https://arxiv.org/pdf/2511.22131v1](https://arxiv.org/pdf/2511.22131v1)**

> **作者:** Xilin Yang; Musa Aydin; Yuhong Lu; Sahan Yoruc Selcuk; Bijie Bai; Yijie Zhang; Andrew Birkeland; Katjana Ehrlich; Julien Bec; Laura Marcu; Nir Pillar; Aydogan Ozcan
>
> **备注:** 20 Pages, 5 Figures
>
> **摘要:** Assessing resection margins is central to pathological specimen evaluation and has profound implications for patient outcomes. Current practice employs physical inking, which is applied variably, and cautery artifacts can obscure the true margin on histological sections. We present a virtual inking network (VIN) that autonomously localizes the surgical cut surface on whole-slide images, reducing reliance on inks and standardizing margin-focused review. VIN uses a frozen foundation model as the feature extractor and a compact two-layer multilayer perceptron trained for patch-level classification of cautery-consistent features. The dataset comprised 120 hematoxylin and eosin (H&E) stained slides from 12 human tonsil tissue blocks, resulting in ~2 TB of uncompressed raw image data, where a board-certified pathologist provided boundary annotations. In blind testing with 20 slides from previously unseen blocks, VIN produced coherent margin overlays that qualitatively aligned with expert annotations across serial sections. Quantitatively, region-level accuracy was ~73.3% across the test set, with errors largely confined to limited areas that did not disrupt continuity of the whole-slide margin map. These results indicate that VIN captures cautery-related histomorphology and can provide a reproducible, ink-free margin delineation suitable for integration into routine digital pathology workflows and for downstream measurement of margin distances.
>
---
#### [new 072] DenoiseGS: Gaussian Reconstruction Model for Burst Denoising
- **分类: cs.CV**

- **简介: 该论文针对手持设备拍摄的图像序列（burst）去噪任务，解决现有方法在大运动下性能下降或计算成本高的问题。提出DenoiseGS，首次利用3D高斯点云的高效性，通过自一致性损失和对数加权频率损失，有效提升去噪质量与细节保留，并实现250倍加速。**

- **链接: [https://arxiv.org/pdf/2511.22939v1](https://arxiv.org/pdf/2511.22939v1)**

> **作者:** Yongsen Cheng; Yuanhao Cai; Yulun Zhang
>
> **摘要:** Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving \textbf{250$\times$} faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.
>
---
#### [new 073] Text Condition Embedded Regression Network for Automated Dental Abutment Design
- **分类: cs.CV**

- **简介: 该论文针对人工牙种植体基台设计耗时、依赖经验的问题，提出文本条件嵌入的自动基台设计框架TCEAD。通过引入文本引导定位模块与预训练编码器，提升模型对局部精细特征的捕捉能力，实现基于口腔扫描数据的精准基台区域定位与自动化设计，显著提升设计效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.22578v1](https://arxiv.org/pdf/2511.22578v1)**

> **作者:** Mianjie Zheng; Xinquan Yang; Xuguang Li; Xiaoling Luo; Xuefen Liu; Kun Tang; He Meng; Linlin Shen
>
> **摘要:** The abutment is an important part of artificial dental implants, whose design process is time-consuming and labor-intensive. Long-term use of inappropriate dental implant abutments may result in implant complications, including peri-implantitis. Using artificial intelligence to assist dental implant abutment design can quickly improve the efficiency of abutment design and enhance abutment adaptability. In this paper, we propose a text condition embedded abutment design framework (TCEAD), the novel automated abutment design solution available in literature. The proposed study extends the self-supervised learning framework of the mesh mask autoencoder (MeshMAE) by introducing a text-guided localization (TGL) module to facilitate abutment area localization. As the parameter determination of the abutment is heavily dependent on local fine-grained features (the width and height of the implant and the distance to the opposing tooth), we pre-train the encoder using oral scan data to improve the model's feature extraction ability. Moreover, considering that the abutment area is only a small part of the oral scan data, we designed a TGL module, which introduces the description of the abutment area through the text encoder of Contrastive Language-Image Pre-training (CLIP), enabling the network to quickly locate the abutment area. We validated the performance of TCEAD on a large abutment design dataset. Extensive experiments demonstrate that TCEAD achieves an Intersection over Union (IoU) improvement of 0.8%-12.85% over other mainstream methods, underscoring its potential in automated dental abutment design.
>
---
#### [new 074] GA2-CLIP: Generic Attribute Anchor for Efficient Prompt Tuningin Video-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视频-语言模型在提示调优中因微调导致的泛化能力下降问题，提出GA2-CLIP框架。通过引入外部预训练文本提示与无关视频集作为通用属性锚点，结合可学习映射层，缓解语义空间收缩，提升模型对未见类别的泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.22125v1](https://arxiv.org/pdf/2511.22125v1)**

> **作者:** Bin Wang; Ruotong Hu; Wenqian Wang; Wentong Li; Mingliang Gao; Runmin Cong; Wei Zhang
>
> **备注:** Technical Report
>
> **摘要:** Visual and textual soft prompt tuning can effectively improve the adaptability of Vision-Language Models (VLMs) in downstream tasks. However, fine-tuning on video tasks impairs the model's generalization ability to unseen classes. Existing methods attempt to mitigate this forgetting effect by regularizing the gap between hand-crafted prompts and soft prompts, but this also weakens the learning ability of soft prompts. To address this challenge, we propose a plug-and-play coupling prompt learning framework to optimize the generalization performance of V-L models in video tasks, with the core motivation of mitigating semantic space narrowing during fine-tuning by introducing an externally supervised prompt. Specifically, for textual prompts, we introduce pre-trained prompts from other datasets as hard prompt tokens. These are concatenated with soft prompt tokens and coupled via a learnable mapping layer. This competitive prompting approach prevents the semantic space from overfitting to supervised categories. In addition, we introduce a set of well-designed irrelevant video sets and negative prompts as generic attribute anchors to maintain the generic relevance of the attributes in the pre-trained semantic space, thus preserving the generalization ability. Experiments on video tasks demonstrate that our method significantly outperforms state-of-the-art prompt tuning approaches across generalization benchmarks, particularly on base-to-new class prediction.
>
---
#### [new 075] DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在综合推理与动作执行时出现的动作性能下降问题，提出DualVLA框架。通过双层数据剪枝与双教师自适应蒸馏，实现推理与动作的解耦优化，在保持强推理能力的同时提升动作准确性。研究还提出VLA Score评估体系，实现对多维度能力的细粒度评测。**

- **链接: [https://arxiv.org/pdf/2511.22134v1](https://arxiv.org/pdf/2511.22134v1)**

> **作者:** Zhen Fang; Zhuoyang Liu; Jiaming Liu; Hao Chen; Yu Zeng; Shiting Huang; Zehui Chen; Lin Chen; Shanghang Zhang; Feng Zhao
>
> **摘要:** To build a generalizable Vision-Language-Action (VLA) model with strong reasoning ability, a common strategy is to first train a specialist VLA on robot demonstrations to acquire reliable manipulation skills, and then incorporate mixed annotated robot data together with multimodal data to restore broader reasoning capabilities. However, we observe that the resulting reasoning VLA often suffers from degraded action performance compared to the specialist model before fine-tuning, a phenomenon we refer to as action degeneration. To address this issue, we propose DualVLA, which enhances action performance through carefully designed post-training while still preserving reasoning capability. We first introduce a dual-layer data pruning method that removes redundant embodied reasoning, preventing it from adversely influencing action learning. To further strengthen action generation, we design a dual-teacher adaptive distillation strategy that assigns different supervision signals to different data domains while maintaining reasoning ability. To fill the evaluation gap for generalist VLAs, we also propose VLA Score, which decouples VLA capability into reasoning, intention, action, and alignment dimensions for a more fine-grained assessment. Experiments show that DualVLA achieves an average success rate of 61.0 in SimplerEnv and an average score of 65.4 across eight competitive multimodal benchmarks, demonstrating a stronger balance between precise action execution and multimodal understanding. Project Website: https://costaliya.github.io/DualVLA/.
>
---
#### [new 076] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型中不同思维链（CoT）设计对可泛化视觉推理能力的影响。针对“为何特定CoT有效”这一问题，通过可控迷宫任务对比语言、接地与视觉CoT，发现简洁的必要接地步骤优于长而复杂的CoT，且最小接地CoT泛化性最佳，揭示“短即长”效应，为构建高效SFT数据提供指导。**

- **链接: [https://arxiv.org/pdf/2511.22586v1](https://arxiv.org/pdf/2511.22586v1)**

> **作者:** Yifan Du; Kun Zhou; Yingqian Min; Yue Ling; Wayne Xin Zhao; Youbin Wu
>
> **摘要:** We study how different Chain-of-Thought (CoT) designs affect the acquisition of the generalizable visual reasoning ability in vision-language models (VLMs). While CoT data, especially long or visual CoT such as "think with image", has been widely used to supervise intermediate reasoning, it remains unclear why specific CoT designs help and which ones truly support generalizable reasoning. To systematically evaluate this, we focus on a controlled maze-solving benchmark where reasoning rules are fully visual, difficulty can be tuned by grid size, and all the intermediate steps can be automatically generated. Using Qwen2.5-VL-7B under a standard SFT-then-RL pipeline, we compare three representative CoT formats: Language CoT, Grounding CoT (with spatial coordinate trajectories), and Visual CoT (with image manipulations). Our experiments reveal that visual and longer CoT mainly accelerate convergence but do not lift the final performance ceiling; concise CoT containing only essential grounding steps outperforms longer traces; and, strikingly, CoT retaining only the minimal grounding results generalizes best across different maze sizes. We further validate these insights on other vision-centric tasks. These findings highlight a "short is long" effect and provide practical guidance for constructing more generalizable SFT datasets for visual reasoning.
>
---
#### [new 077] Asking like Socrates: Socrates helps VLMs understand remote sensing images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像理解中模型伪推理问题，提出RS-EoT框架，通过苏格拉底式自问自答的多智能体系统与分阶段强化学习，引导模型迭代寻找视觉证据，克服“一瞥效应”，实现基于视觉证据的真实推理，在遥感视觉问答与定位任务上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2511.22396v1](https://arxiv.org/pdf/2511.22396v1)**

> **作者:** Run Shao; Ziyu Li; Zhaoyang Zhang; Linrui Xu; Xinran He; Hongyuan Yuan; Bolei He; Yongxing Dai; Yiming Yan; Yijun Chen; Wang Guo; Haifeng Li
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Recent multimodal reasoning models, inspired by DeepSeek-R1, have significantly advanced vision-language systems. However, in remote sensing (RS) tasks, we observe widespread pseudo reasoning: models narrate the process of reasoning rather than genuinely reason toward the correct answer based on visual evidence. We attribute this to the Glance Effect, where a single, coarse perception of large-scale RS imagery results in incomplete understanding and reasoning based on linguistic self-consistency instead of visual evidence. To address this, we propose RS-EoT (Remote Sensing Evidence-of-Thought), a language-driven, iterative visual evidence-seeking paradigm. To instill this paradigm, we propose SocraticAgent, a self-play multi-agent system that synthesizes reasoning traces via alternating cycles of reasoning and visual inspection. To enhance and generalize these patterns, we propose a two-stage progressive RL strategy: first, RL on fine-grained Grounding tasks to enhance RS-EoT capabilities, followed by RL on RS VQA to generalize to broader understanding scenarios. Experiments show RS-EoT achieves state-of-the-art performance on multiple RS VQA and grounding benchmarks. Analyses reveal clear iterative cycles of reasoning and evidence seeking, confirming RS-EoT mitigates the Glance Effect and enables genuine evidence-grounded reasoning. Our code, data, and models are available at https://geox-lab.github.io/Asking_like_Socrates
>
---
#### [new 078] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视图变化大时视觉失真与几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控制的动态场景仿真。构建MIRROR数据集，支持多样化驾驶场景评测，显著提升仿真真实性和可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.22187v1](https://arxiv.org/pdf/2511.22187v1)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
#### [new 079] Saddle-Free Guidance: Improved On-Manifold Sampling without Labels or Additional Training
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文针对无标签数据下得分生成模型的采样质量提升问题，提出无需额外训练的鞍点自由引导（SFG）方法。利用对数密度在鞍点区域的正曲率提供引导信号，显著提升图像多样性与保真度，在ImageNet-512等任务上达到先进水平。**

- **链接: [https://arxiv.org/pdf/2511.21863v1](https://arxiv.org/pdf/2511.21863v1)**

> **作者:** Eric Yeats; Darryl Hannan; Wilson Fearn; Timothy Doster; Henry Kvinge; Scott Mahan
>
> **摘要:** Score-based generative models require guidance in order to generate plausible, on-manifold samples. The most popular guidance method, Classifier-Free Guidance (CFG), is only applicable in settings with labeled data and requires training an additional unconditional score-based model. More recently, Auto-Guidance adopts a smaller, less capable version of the original model to guide generation. While each method effectively promotes the fidelity of generated data, each requires labeled data or the training of additional models, making it challenging to guide score-based models when (labeled) training data are not available or training new models is not feasible. We make the surprising discovery that the positive curvature of log density estimates in saddle regions provides strong guidance for score-based models. Motivated by this, we develop saddle-free guidance (SFG) which maintains estimates of maximal positive curvature of the log density to guide individual score-based models. SFG has the same computational cost of classifier-free guidance, does not require additional training, and works with off-the-shelf diffusion and flow matching models. Our experiments indicate that SFG achieves state-of-the-art FID and FD-DINOv2 metrics in single-model unconditional ImageNet-512 generation. When SFG is combined with Auto-Guidance, its unconditional samples achieve general state-of-the-art in FD-DINOv2 score. Our experiments with FLUX.1-dev and Stable Diffusion v3.5 indicate that SFG boosts the diversity of output images compared to CFG while maintaining excellent prompt adherence and image fidelity.
>
---
#### [new 080] Vision Bridge Transformer at Scale
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出大规模视觉桥接变压器（ViBT），用于条件生成任务。针对传统扩散模型效率低的问题，提出直接建模输入到输出轨迹的桥接模型，并通过Transformer架构与稳定方差的速度匹配目标，实现高效图像与视频翻译，支持指令式图像编辑与复杂视频转换。**

- **链接: [https://arxiv.org/pdf/2511.23199v1](https://arxiv.org/pdf/2511.23199v1)**

> **作者:** Zhenxiong Tan; Zeqing Wang; Xingyi Yang; Songhua Liu; Xinchao Wang
>
> **摘要:** We introduce Vision Bridge Transformer (ViBT), a large-scale instantiation of Brownian Bridge Models designed for conditional generation. Unlike traditional diffusion models that transform noise into data, Bridge Models directly model the trajectory between inputs and outputs, creating an efficient data-to-data translation paradigm. By scaling these models to 20B and 1.3B parameters, we demonstrate their effectiveness for image and video translation tasks. To support this scale, we adopt a Transformer architecture and propose a variance-stabilized velocity-matching objective for robust training. Together, these advances highlight the power of scaling Bridge Models for instruction-based image editing and complex video translation.
>
---
#### [new 081] Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究生成式图像修复在骨龄评估中的临床影响。针对儿科手部X光片中非解剖标记物的修复问题，使用大模型进行图像修补，并评估其对骨龄与性别预测性能的影响。结果表明，尽管修复图像视觉上自然，但显著降低模型性能并引入结构偏差，提示需谨慎验证生成技术在医疗AI中的适用性。**

- **链接: [https://arxiv.org/pdf/2511.23066v1](https://arxiv.org/pdf/2511.23066v1)**

> **作者:** Felipe Akio Matsuoka; Eduardo Moreno J. M. Farina; Augusto Sarquis Serpa; Soraya Monteiro; Rodrigo Ragazzini; Nitamar Abdala; Marcelo Straus Takahashi; Felipe Campos Kitamura
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Generative foundation models can remove visual artifacts through realistic image inpainting, but their impact on medical AI performance remains uncertain. Pediatric hand radiographs often contain non-anatomical markers, and it is unclear whether inpainting these regions preserves features needed for bone age and gender prediction. To evaluate the clinical reliability of generative model-based inpainting for artifact removal, we used the RSNA Bone Age Challenge dataset, selecting 200 original radiographs and generating 600 inpainted versions with gpt-image-1 using natural language prompts to target non-anatomical artifacts. Downstream performance was assessed with deep learning ensembles for bone age estimation and gender classification, using mean absolute error (MAE) and area under the ROC curve (AUC) as metrics, and pixel intensity distributions to detect structural alterations. Inpainting markedly degraded model performance: bone age MAE increased from 6.26 to 30.11 months, and gender classification AUC decreased from 0.955 to 0.704. Inpainted images displayed pixel-intensity shifts and inconsistencies, indicating structural modifications not corrected by simple calibration. These findings show that, although visually realistic, foundation model-based inpainting can obscure subtle but clinically relevant features and introduce latent bias even when edits are confined to non-diagnostic regions, underscoring the need for rigorous, task-specific validation before integrating such generative tools into clinical AI workflows.
>
---
#### [new 082] Video-CoM: Interactive Video Reasoning via Chain of Manipulations
- **分类: cs.CV**

- **简介: 该论文提出Video CoM，针对多模态大模型视频理解中“被动编码、静态推理”的局限，引入交互式视频推理新范式。通过链式操作（CoM）实现动态视觉探索，构建18K指令数据集并结合推理感知的强化学习优化策略，显著提升细粒度时空理解能力，在9个基准上平均性能提升3.6%。**

- **链接: [https://arxiv.org/pdf/2511.23477v1](https://arxiv.org/pdf/2511.23477v1)**

> **作者:** Hanoona Rasheed; Mohammed Zumri; Muhammad Maaz; Ming-Hsuan Yang; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** Technical Report
>
> **摘要:** Recent multimodal large language models (MLLMs) have advanced video understanding, yet most still "think about videos" ie once a video is encoded, reasoning unfolds entirely in text, treating visual input as a static context. This passive paradigm creates a semantic bottleneck: models cannot rewatch, refocus, or verify evidence, leading to shallow visual reasoning on tasks requiring fine grained spatio temporal understanding. In this work, we introduce Interactive Video Reasoning, a new paradigm that transforms video into an active cognitive workspace, enabling models to "think with videos". Our model, Video CoM, reasons through a Chain of Manipulations (CoM), performing iterative visual actions to gather and refine evidence. To support this behavior, we construct Video CoM Instruct, an 18K instruction tuning dataset curated for multi step manipulation reasoning. Beyond supervised learning, we further optimize the manipulation policy via reinforcement learning with reasoning aware Group Relative Policy Optimization (GRPO). Unlike prior work that relies solely on sparse answer rewards, our method introduces step level reasoning rewards, guiding the model toward grounded and consistent reasoning. Video CoM achieves strong results across nine video reasoning benchmarks, improving average performance by 3.6 percent over recent state of the art models, while training on only 25K SFT and 3K GRPO video samples, significantly fewer than comparable large scale models. Ablation studies demonstrate that reasoning aware rewards improve both accuracy and interpretability. Code: https://github.com/mbzuai-oryx/Video-CoM
>
---
#### [new 083] SemOD: Semantic Enabled Object Detection Network under Various Weather Conditions
- **分类: cs.CV; eess.SY**

- **简介: 该论文针对自动驾驶中摄像头在复杂天气下感知性能下降的问题，提出SemOD网络。通过引入语义信息增强图像修复与检测，在多种天气条件下提升目标检测精度，实现更鲁棒的视觉理解。**

- **链接: [https://arxiv.org/pdf/2511.22142v1](https://arxiv.org/pdf/2511.22142v1)**

> **作者:** Aiyinsi Zuo; Zhaoliang Zheng
>
> **摘要:** In the field of autonomous driving, camera-based perception models are mostly trained on clear weather data. Models that focus on addressing specific weather challenges are unable to adapt to various weather changes and primarily prioritize their weather removal characteristics. Our study introduces a semantic-enabled network for object detection in diverse weather conditions. In our analysis, semantics information can enable the model to generate plausible content for missing areas, understand object boundaries, and preserve visual coherency and realism across both filled-in and existing portions of the image, which are conducive to image transformation and object recognition. Specific in implementation, our architecture consists of a Preprocessing Unit (PPU) and a Detection Unit (DTU), where the PPU utilizes a U-shaped net enriched by semantics to refine degraded images, and the DTU integrates this semantic information for object detection using a modified YOLO network. Our method pioneers the use of semantic data for all-weather transformations, resulting in an increase between 1.47\% to 8.80\% in mAP compared to existing methods across benchmark datasets of different weather. This highlights the potency of semantics in image enhancement and object detection, offering a comprehensive approach to improving object detection performance. Code will be available at https://github.com/EnisZuo/SemOD.
>
---
#### [new 084] Convolutional Feature Noise Reduction for 2D Cardiac MR Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对2D心脏MR图像分割中卷积特征噪声被忽视的问题，提出一种名为CFF的特征滤波器。通过将特征视为高斯分布信号，设计低幅值通滤波器以降低特征噪声，并引入二值化熵公式量化噪声减少效果。实验验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.22983v1](https://arxiv.org/pdf/2511.22983v1)**

> **作者:** Hong Zheng; Nan Mu; Han Su; Lin Feng; Xiaoning Li
>
> **摘要:** Noise reduction constitutes a crucial operation within Digital Signal Processing. Regrettably, it frequently remains neglected when dealing with the processing of convolutional features in segmentation networks. This oversight could trigger the butterfly effect, impairing the subsequent outcomes within the entire feature system. To complete this void, we consider convolutional features following Gaussian distributions as feature signal matrices and then present a simple and effective feature filter in this study. The proposed filter is fundamentally a low-amplitude pass filter primarily aimed at minimizing noise in feature signal inputs and is named Convolutional Feature Filter (CFF). We conducted experiments on two established 2D segmentation networks and two public cardiac MR image datasets to validate the effectiveness of the CFF, and the experimental findings demonstrated a decrease in noise within the feature signal matrices. To enable a numerical observation and analysis of this reduction, we developed a binarization equation to calculate the information entropy of feature signals.
>
---
#### [new 085] Pathryoshka: Compressing Pathology Foundation Models via Multi-Teacher Knowledge Distillation with Nested Embeddings
- **分类: cs.CV**

- **简介: 该论文针对病理学基础模型参数量大、计算资源需求高的问题，提出Pathryoshka多教师知识蒸馏框架，通过嵌套嵌入实现模型压缩。在10个公开病理数据集上验证，模型尺寸减少86-92%且性能相当，优于现有单教师蒸馏方法，推动了高效病理模型的本地部署与普及。**

- **链接: [https://arxiv.org/pdf/2511.23204v1](https://arxiv.org/pdf/2511.23204v1)**

> **作者:** Christian Grashei; Christian Brechenmacher; Rao Muhammad Umer; Jingsong Liu; Carsten Marr; Ewa Szczurek; Peter J. Schüffler
>
> **摘要:** Pathology foundation models (FMs) have driven significant progress in computational pathology. However, these high-performing models can easily exceed a billion parameters and produce high-dimensional embeddings, thus limiting their applicability for research or clinical use when computing resources are tight. Here, we introduce Pathryoshka, a multi-teacher distillation framework inspired by RADIO distillation and Matryoshka Representation Learning to reduce pathology FM sizes while allowing for adaptable embedding dimensions. We evaluate our framework with a distilled model on ten public pathology benchmarks with varying downstream tasks. Compared to its much larger teachers, Pathryoshka reduces the model size by 86-92% at on-par performance. It outperforms state-of-the-art single-teacher distillation models of comparable size by a median margin of 7.0 in accuracy. By enabling efficient local deployment without sacrificing accuracy or representational richness, Pathryoshka democratizes access to state-of-the-art pathology FMs for the broader research and clinical community.
>
---
#### [new 086] Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models
- **分类: cs.CV**

- **简介: 该论文聚焦视频理解任务，针对多模态大模型在动态视觉内容推理中逻辑不一致、依赖语言先验而非视觉证据的问题。提出Video-R2模型，通过时序感知微调与基于时间对齐奖励的强化学习，提升推理一致性与视觉依存度，显著改善多基准测试性能。**

- **链接: [https://arxiv.org/pdf/2511.23478v1](https://arxiv.org/pdf/2511.23478v1)**

> **作者:** Muhammad Maaz; Hanoona Rasheed; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** Video-R2 Technical Report
>
> **摘要:** Reasoning over dynamic visual content remains a central challenge for multimodal large language models. Recent thinking models generate explicit reasoning traces for interpretability; however, their reasoning often appears convincing while being logically inconsistent or weakly grounded in visual evidence. We identify and formalize these issues through two diagnostic metrics: Think Answer Consistency (TAC), which measures the alignment between reasoning and answers, and Video Attention Score (VAS), which captures the extent to which reasoning depends on visual versus textual cues. Analysis across 11 video reasoning benchmarks shows that current models rely heavily on linguistic priors rather than visual content. To address this, we propose a reinforcement learning approach that enhances both temporal precision and reasoning consistency. Our approach combines timestamp aware supervised fine tuning with Group Relative Policy Optimization (GRPO) guided by a novel Temporal Alignment Reward (TAR). This dual step post training stage encourages temporally aligned and causally coherent video reasoning. The resulting model, Video R2, achieves consistently higher TAC, VAS, and accuracy across multiple benchmarks, demonstrating that improvements in temporal alignment and reasoning coherence lead to more accurate and trustworthy video understanding. Our code, dataset, and model will be open sourced.
>
---
#### [new 087] LC4-DViT: Land-cover Creation for Land-cover Classification with Deformable Vision Transformer
- **分类: cs.CV**

- **简介: 该论文针对高分辨率遥感图像中土地覆盖分类因标注稀缺、分布不均及几何畸变导致的精度瓶颈，提出LC4-DViT框架。通过文本引导生成增强数据，并结合变形感知视觉变压器，联合建模细粒度几何与全局上下文，显著提升分类性能与跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22812v1](https://arxiv.org/pdf/2511.22812v1)**

> **作者:** Kai Wang; Siyi Chen; Weicong Pang; Chenchen Zhang; Renjun Gao; Ziru Chen; Cheng Li; Dasa Gu; Rui Huang; Alexis Kai Hon Lau
>
> **备注:** This work has been submitted to the IEEE for possible publication.The project is available at https://github.com/weicongpang/LVC2-DViT.git
>
> **摘要:** Land-cover underpins ecosystem services, hydrologic regulation, disaster-risk reduction, and evidence-based land planning; timely, accurate land-cover maps are therefore critical for environmental stewardship. Remote sensing-based land-cover classification offers a scalable route to such maps but is hindered by scarce and imbalanced annotations and by geometric distortions in high-resolution scenes. We propose LC4-DViT (Land-cover Creation for Land-cover Classification with Deformable Vision Transformer), a framework that combines generative data creation with a deformation-aware Vision Transformer. A text-guided diffusion pipeline uses GPT-4o-generated scene descriptions and super-resolved exemplars to synthesize class-balanced, high-fidelity training images, while DViT couples a DCNv4 deformable convolutional backbone with a Vision Transformer encoder to jointly capture fine-scale geometry and global context. On eight classes from the Aerial Image Dataset (AID)-Beach, Bridge, Desert, Forest, Mountain, Pond, Port, and River-DViT achieves 0.9572 overall accuracy, 0.9576 macro F1-score, and 0.9510 Cohen' s Kappa, improving over a vanilla ViT baseline (0.9274 OA, 0.9300 macro F1, 0.9169 Kappa) and outperforming ResNet50, MobileNetV2, and FlashInternImage. Cross-dataset experiments on a three-class SIRI-WHU subset (Harbor, Pond, River) yield 0.9333 overall accuracy, 0.9316 macro F1, and 0.8989 Kappa, indicating good transferability. An LLM-based judge using GPT-4o to score Grad-CAM heatmaps further shows that DViT' s attention aligns best with hydrologically meaningful structures. These results suggest that description-driven generative augmentation combined with deformation-aware transformers is a promising approach for high-resolution land-cover mapping.
>
---
#### [new 088] ITS3D: Inference-Time Scaling for Text-Guided 3D Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本引导3D生成模型的生成质量提升问题，提出ITS3D框架。通过推理时优化噪声输入，结合验证器引导搜索与三项技术（高斯归一化、SVD压缩、奇异空间重置），在不增加训练成本下显著提升生成质量，实现高效稳定搜索。**

- **链接: [https://arxiv.org/pdf/2511.22456v1](https://arxiv.org/pdf/2511.22456v1)**

> **作者:** Zhenglin Zhou; Fan Ma; Xiaobo Xia; Hehe Fan; Yi Yang; Tat-Seng Chua
>
> **备注:** 25 pages, 11 figures
>
> **摘要:** We explore inference-time scaling in text-guided 3D diffusion models to enhance generative quality without additional training. To this end, we introduce ITS3D, a framework that formulates the task as an optimization problem to identify the most effective Gaussian noise input. The framework is driven by a verifier-guided search algorithm, where the search algorithm iteratively refines noise candidates based on verifier feedback. To address the inherent challenges of 3D generation, we introduce three techniques for improved stability, efficiency, and exploration capability. 1) Gaussian normalization is applied to stabilize the search process. It corrects distribution shifts when noise candidates deviate from a standard Gaussian distribution during iterative updates. 2) The high-dimensional nature of the 3D search space increases computational complexity. To mitigate this, a singular value decomposition-based compression technique is employed to reduce dimensionality while preserving effective search directions. 3) To further prevent convergence to suboptimal local minima, a singular space reset mechanism dynamically updates the search space based on diversity measures. Extensive experiments demonstrate that ITS3D enhances text-to-3D generation quality, which shows the potential of computationally efficient search methods in generative processes. The source code is available at https://github.com/ZhenglinZhou/ITS3D.
>
---
#### [new 089] UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes
- **分类: cs.CV**

- **简介: 该论文针对遥感图像中指令驱动分割任务，解决现有方法因任务碎片化和指令数据匮乏导致的泛化能力差问题。提出百万级数据集GeoSeg-1M与GeoSeg-Bench基准，构建统一框架UniGeoSeg，实现多任务协同与零样本泛化，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.23332v1](https://arxiv.org/pdf/2511.23332v1)**

> **作者:** Shuo Ni; Di Wang; He Chen; Haonan Guo; Ning Zhang; Jing Zhang
>
> **备注:** Datasets and source code were released at https://github.com/MiliLab/UniGeoSeg
>
> **摘要:** Instruction-driven segmentation in remote sensing generates masks from guidance, offering great potential for accessible and generalizable applications. However, existing methods suffer from fragmented task formulations and limited instruction data, hindering effective understanding and generalization. To address these issues, we introduce GeoSeg-1M, the first million-scale dataset for remote sensing instruction-driven segmentation, constructed via an automatic mask filtering and instruction generation pipeline that synthesizes referring, interactive, and reasoning segmentation instructions from multiple public datasets. GeoSeg-1M contains 590K images, 117 categories, and 1.1M image-mask-instruction triplets. Building upon this foundation, we further curate GeoSeg-Bench, a challenging benchmark designed to evaluate contextual understanding and reasoning capabilities across diverse instruction-driven tasks and complex geospatial scenes. Furthermore, we present UniGeoSeg, a unified framework that serves as a strong baseline, incorporating task-aware text enhancement, latent knowledge memory, and a progressive training strategy to facilitate multi-task learning. Extensive experiments demonstrate the state-of-the-art performance of UniGeoSeg across GeoSeg-Bench and diverse public benchmarks, while exhibiting strong zero-shot generalization. Datasets and source code were released at https://github.com/MiliLab/UniGeoSeg.
>
---
#### [new 090] Contrastive Heliophysical Image Pretraining for Solar Dynamics Observatory Records
- **分类: cs.CV**

- **简介: 该论文针对太阳图像分析中多仪器数据、类间区分弱和类内变化大等问题，提出SolarCHIP，一种基于对比学习的预训练视觉模型。通过多粒度对比损失，联合优化跨模态对齐、位置一致性和空间结构保持，显著提升低资源场景下的跨模态翻译与耀斑分类性能，为太阳物理提供可复用的高效特征提取器。**

- **链接: [https://arxiv.org/pdf/2511.22958v1](https://arxiv.org/pdf/2511.22958v1)**

> **作者:** Shiyu Shen; Zhe Gao; Taifeng Chai; Yang Huang; Bin Pan
>
> **摘要:** Deep learning has revolutionized solar image analysis, yet most approaches train task-specific encoders from scratch or rely on natural-image pretraining that ignores the unique characteristics of Solar Dynamics Observatory (SDO) data. We introduce SolarCHIP, a family of contrastively pretrained visual backbones tailored to multi-instrument SDO observations. SolarCHIP addresses three key challenges in solar imaging: multimodal sensing across AIA and HMI instruments, weak inter-class separability due to slow temporal evolution, and strong intra-class variability with sparse activity signals. Our pretraining framework employs a multi-granularity contrastive objective that jointly aligns (1) global class tokens across co-temporal AIA-HMI pairs to enhance temporal discrimination, (2) local patch tokens at fixed spatial indices to enforce position-consistent, modality-invariant features, and (3) intra-sample patches across different spatial locations to preserve fine-grained spatial structure. We train both CNN- and Vision Transformer-based autoencoders and demonstrate their effectiveness on two downstream tasks: cross-modal translation between HMI and AIA passbands via ControlNet, and full-disk flare classification. Experimental results show that SolarCHIP achieves state-of-the-art performance across both tasks, with particularly strong gains in low-resource settings where labeled data is limited. Ablation studies confirm that each contrastive component contributes essential discriminative capacity at different granularities. By publicly releasing pretrained weights and training code, we provide the heliophysics community with a practical, plug-and-play feature extractor that reduces computational requirements, improves label efficiency, and establishes a reusable foundation for diverse solar imaging applications.
>
---
#### [new 091] DM$^3$T: Harmonizing Modalities via Diffusion for Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对多模态目标跟踪（MOT）中可见光与热红外信息融合困难的问题，提出DM³T框架。通过扩散模型启发的迭代跨模态对齐，实现特征深度融合，并引入可插拔精炼器与分层跟踪器，提升轨迹一致性与鲁棒性。在VT-MOT上达到41.7 HOTA，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22896v1](https://arxiv.org/pdf/2511.22896v1)**

> **作者:** Weiran Li; Yeqiang Liu; Yijie Wei; Mina Han; Qiannan Guo; Zhenbo Li
>
> **摘要:** Multi-object tracking (MOT) is a fundamental task in computer vision with critical applications in autonomous driving and robotics. Multimodal MOT that integrates visible light and thermal infrared information is particularly essential for robust autonomous driving systems. However, effectively fusing these heterogeneous modalities is challenging. Simple strategies like concatenation or addition often fail to bridge the significant non-linear distribution gap between their feature representations, which can lead to modality conflicts and degrade tracking accuracy. Drawing inspiration from the connection between multimodal MOT and the iterative refinement in diffusion models, this paper proposes DM$^3$T, a novel framework that reformulates multimodal fusion as an iterative feature alignment process to generate accurate and temporally coherent object trajectories. Our approach performs iterative cross-modal harmonization through a proposed Cross-Modal Diffusion Fusion (C-MDF) module. In this process, features from both modalities provide mutual guidance, iteratively projecting them onto a shared, consistent feature manifold. This enables the learning of complementary information and achieves deeper fusion compared to conventional methods. Additionally, we introduce a plug-and-play Diffusion Refiner (DR) to enhance and refine the unified feature representation. To further improve tracking robustness, we design a Hierarchical Tracker that adaptively handles confidence estimation. DM$^3$T unifies object detection, state estimation, and data association into a comprehensive online tracking framework without complex post-processing. Extensive experiments on the VT-MOT benchmark demonstrate that our method achieves 41.7 HOTA, representing a 1.54% relative improvement over existing state-of-the-art methods. The code and models are available at https://vranlee.github.io/DM-3-T/.
>
---
#### [new 092] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出SciPostGen数据集，旨在连接科学论文与海报布局。针对论文结构与海报元素数量关联性不足的问题，提出检索增强的海报布局生成框架，能根据论文内容生成符合结构且满足约束的布局，有效提升科研成果展示效率。**

- **链接: [https://arxiv.org/pdf/2511.22490v1](https://arxiv.org/pdf/2511.22490v1)**

> **作者:** Shun Inadumi; Shohei Tanaka; Tosho Hirasawa; Atsushi Hashimoto; Koichiro Yoshino; Yoshitaka Ushiku
>
> **备注:** Dataset: https://huggingface.co/datasets/omron-sinicx/scipostgen, Code: https://github.com/omron-sinicx/scipostgen_dataset_construction
>
> **摘要:** As the number of scientific papers continues to grow, there is a demand for approaches that can effectively convey research findings, with posters serving as a key medium for presenting paper contents. Poster layouts determine how effectively research is communicated and understood, highlighting their growing importance. In particular, a gap remains in understanding how papers correspond to the layouts that present them, which calls for datasets with paired annotations at scale. To bridge this gap, we introduce SciPostGen, a large-scale dataset for understanding and generating poster layouts from scientific papers. Our analyses based on SciPostGen show that paper structures are associated with the number of layout elements in posters. Based on this insight, we explore a framework, Retrieval-Augmented Poster Layout Generation, which retrieves layouts consistent with a given paper and uses them as guidance for layout generation. We conducted experiments under two conditions: with and without layout constraints typically specified by poster creators. The results show that the retriever estimates layouts aligned with paper structures, and our framework generates layouts that also satisfy given constraints.
>
---
#### [new 093] From Pixels to Feelings: Aligning MLLMs with Human Cognitive Perception of Images
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文针对多模态大模型（MLLMs）在图像主观认知感知（如记忆性、美感、情感）上与人类不一致的问题，提出CogIP-Bench基准，评估并揭示模型差距；通过后训练提升模型与人类感知的对齐度，并验证其在图像生成中引导创造性的能力，推动更人性化的人工智能。**

- **链接: [https://arxiv.org/pdf/2511.22805v1](https://arxiv.org/pdf/2511.22805v1)**

> **作者:** Yiming Chen; Junlin Han; Tianyi Bai; Shengbang Tong; Filippos Kokkinos; Philip Torr
>
> **备注:** Project page with codes/datasets/models: https://follen-cry.github.io/MLLM-Cognition-project-page/
>
> **摘要:** While Multimodal Large Language Models (MLLMs) are adept at answering what is in an image-identifying objects and describing scenes-they often lack the ability to understand how an image feels to a human observer. This gap is most evident when considering subjective cognitive properties, such as what makes an image memorable, funny, aesthetically pleasing, or emotionally evocative. To systematically address this challenge, we introduce CogIP-Bench, a comprehensive benchmark for evaluating MLLMs on such image cognitive properties. Our evaluation reveals a significant gap: current models are poorly aligned with human perception of these nuanced properties. We then demonstrate that a post-training phase can effectively bridge this gap, significantly enhancing the model's alignment with human judgments. Furthermore, we show that this learned cognitive alignment is not merely predictive but also transferable to downstream creative tasks. By integrating our cognitively-aligned MLLM into an image generation pipeline, we can guide the synthesis process to produce images that better embody desired traits, such as being more memorable or visually appealing. Our work provides a benchmark to measure this human-like perception, a post-training pipeline to enhance it, and a demonstration that this alignment unlocks more human-centric AI.
>
---
#### [new 094] REASONEDIT: Towards Reasoning-Enhanced Image Editing Models
- **分类: cs.CV**

- **简介: 该论文针对图像编辑任务，提出ReasonEdit框架，通过引入思维与反思机制增强多模态大模型的推理能力。解决现有模型对抽象指令理解不足、编辑误差难修正的问题。工作包括设计思考-编辑-反思循环，提升指令理解与编辑精度，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22625v1](https://arxiv.org/pdf/2511.22625v1)**

> **作者:** Fukun Yin; Shiyu Liu; Yucheng Han; Zhibo Wang; Peng Xing; Rui Wang; Wei Cheng; Yingming Wang; Aojie Li; Zixin Yin; Pengtao Chen; Xiangyu Zhang; Daxin Jiang; Xianfang Zeng; Gang Yu
>
> **备注:** code: https://github.com/stepfun-ai/Step1X-Edit
>
> **摘要:** Recent advances in image editing models have shown remarkable progress. A common architectural design couples a multimodal large language model (MLLM) encoder with a diffusion decoder, as seen in systems such as Step1X-Edit and Qwen-Image-Edit, where the MLLM encodes both the reference image and the instruction but remains frozen during training. In this work, we demonstrate that unlocking the reasoning capabilities of MLLM can further push the boundaries of editing models. Specifically, we explore two reasoning mechanisms, thinking and reflection, which enhance instruction understanding and editing accuracy. Based on that, our proposed framework enables image editing in a thinking-editing-reflection loop: the thinking mechanism leverages the world knowledge of MLLM to interpret abstract instructions, while the reflection reviews editing results, automatically corrects unintended manipulations, and identifies the stopping round. Extensive experiments demonstrate that our reasoning approach achieves significant performance gains, with improvements of ImgEdit (+4.3%), GEdit (+4.7%), and Kris (+8.2%) when initializing our DiT from the Step1X-Edit (ReasonEdit-S), and also outperforms previous open-source methods on both GEdit and Kris when integrated with Qwen-Image-Edit (ReasonEdit-Q).
>
---
#### [new 095] DeepGI: Explainable Deep Learning for Gastrointestinal Image Classification
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 该论文针对胃肠疾病图像分类任务，解决真实世界中光照不均、视角变化等挑战。基于4000张多类内镜图像，对比分析VGG16、MobileNetV2等模型，最高达96.5%准确率，并引入Grad-CAM实现模型可解释性，提升临床可信度。**

- **链接: [https://arxiv.org/pdf/2511.21959v1](https://arxiv.org/pdf/2511.21959v1)**

> **作者:** Walid Houmaidi; Mohamed Hadadi; Youssef Sabiri; Yousra Chtouki
>
> **备注:** 7 pages, 4 figures, 2 tables. Accepted at DASET 2026
>
> **摘要:** This paper presents a comprehensive comparative model analysis on a novel gastrointestinal medical imaging dataset, comprised of 4,000 endoscopic images spanning four critical disease classes: Diverticulosis, Neoplasm, Peritonitis, and Ureters. Leveraging state-of-the-art deep learning techniques, the study confronts common endoscopic challenges such as variable lighting, fluctuating camera angles, and frequent imaging artifacts. The best performing models, VGG16 and MobileNetV2, each achieved a test accuracy of 96.5%, while Xception reached 94.24%, establishing robust benchmarks and baselines for automated disease classification. In addition to strong classification performance, the approach includes explainable AI via Grad-CAM visualization, enabling identification of image regions most influential to model predictions and enhancing clinical interpretability. Experimental results demonstrate the potential for robust, accurate, and interpretable medical image analysis even in complex real-world conditions. This work contributes original benchmarks, comparative insights, and visual explanations, advancing the landscape of gastrointestinal computer-aided diagnosis and underscoring the importance of diverse, clinically relevant datasets and model explainability in medical AI research.
>
---
#### [new 096] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对人中心场景的稀疏视图自由视角渲染任务，解决传统高斯点阵需密集输入视图的问题。提出Splat-SAP方法，通过两阶段学习：先自监督重建尺度感知点图，再基于双目匹配优化几何并生成高斯原语，实现稀疏输入下的稳定高质量渲染。**

- **链接: [https://arxiv.org/pdf/2511.22704v1](https://arxiv.org/pdf/2511.22704v1)**

> **作者:** Boyao Zhou; Shunyuan Zheng; Zhanfeng Liao; Zihan Ma; Hanzhang Tu; Boning Liu; Yebin Liu
>
> **备注:** Accepted by AAAI 2026. Project page: https://yaourtb.github.io/Splat-SAP
>
> **摘要:** We present Splat-SAP, a feed-forward approach to render novel views of human-centered scenes from binocular cameras with large sparsity. Gaussian Splatting has shown its promising potential in rendering tasks, but it typically necessitates per-scene optimization with dense input views. Although some recent approaches achieve feed-forward Gaussian Splatting rendering through geometry priors obtained by multi-view stereo, such approaches still require largely overlapped input views to establish the geometry prior. To bridge this gap, we leverage pixel-wise point map reconstruction to represent geometry which is robust to large sparsity for its independent view modeling. In general, we propose a two-stage learning strategy. In stage 1, we transform the point map into real space via an iterative affinity learning process, which facilitates camera control in the following. In stage 2, we project point maps of two input views onto the target view plane and refine such geometry via stereo matching. Furthermore, we anchor Gaussian primitives on this refined plane in order to render high-quality images. As a metric representation, the scale-aware point map in stage 1 is trained in a self-supervised manner without 3D supervision and stage 2 is supervised with photo-metric loss. We collect multi-view human-centered data and demonstrate that our method improves both the stability of point map reconstruction and the visual quality of free-viewpoint rendering.
>
---
#### [new 097] Emergent Extreme-View Geometry in 3D Foundation Models
- **分类: cs.CV**

- **简介: 该论文研究3D基础模型在极端非重叠视角下的几何推理能力。针对其在未训练条件下对极端视图理解不足的问题，提出轻量级对齐方案，仅微调少量偏置项，显著提升相对位姿估计性能，同时保持单图像深度与点云质量。构建新基准MegaUnScene用于评估。**

- **链接: [https://arxiv.org/pdf/2511.22686v1](https://arxiv.org/pdf/2511.22686v1)**

> **作者:** Yiwen Zhang; Joseph Tung; Ruojin Cai; David Fouhey; Hadar Averbuch-Elor
>
> **备注:** Project page is at https://ext-3dfms.github.io/
>
> **摘要:** 3D foundation models (3DFMs) have recently transformed 3D vision, enabling joint prediction of depths, poses, and point maps directly from images. Yet their ability to reason under extreme, non-overlapping views remains largely unexplored. In this work, we study their internal representations and find that 3DFMs exhibit an emergent understanding of extreme-view geometry, despite never being trained for such conditions. To further enhance these capabilities, we introduce a lightweight alignment scheme that refines their internal 3D representation by tuning only a small subset of backbone bias terms, leaving all decoder heads frozen. This targeted adaptation substantially improves relative pose estimation under extreme viewpoints without degrading per-image depth or point quality. Additionally, we contribute MegaUnScene, a new benchmark of Internet scenes unseen by existing 3DFMs, with dedicated test splits for both relative pose estimation and dense 3D reconstruction. All code and data will be released.
>
---
#### [new 098] Taming the Light: Illumination-Invariant Semantic 3DGS-SLAM
- **分类: cs.CV**

- **简介: 该论文针对光照变化导致3D SLAM与语义分割精度下降的问题，提出一种光照不变的语义3DGS-SLAM框架。通过内在外观归一化（IAN）模块主动分离反照率与光照，实现稳定的颜色表示；结合动态辐射平衡损失（DRB-Loss）在极端曝光时被动修正，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22968v1](https://arxiv.org/pdf/2511.22968v1)**

> **作者:** Shouhe Zhang; Dayong Ren; Sensen Song; Yurong Qian; Zhenhong Jia
>
> **摘要:** Extreme exposure degrades both the 3D map reconstruction and semantic segmentation accuracy, which is particularly detrimental to tightly-coupled systems. To achieve illumination invariance, we propose a novel semantic SLAM framework with two designs. First, the Intrinsic Appearance Normalization (IAN) module proactively disentangles the scene's intrinsic properties, such as albedo, from transient lighting. By learning a standardized, illumination-invariant appearance model, it assigns a stable and consistent color representation to each Gaussian primitive. Second, the Dynamic Radiance Balancing Loss (DRB-Loss) reactively handles frames with extreme exposure. It activates only when an image's exposure is poor, operating directly on the radiance field to guide targeted optimization. This prevents error accumulation from extreme lighting without compromising performance under normal conditions. The synergy between IAN's proactive invariance and DRB-Loss's reactive correction endows our system with unprecedented robustness. Evaluations on public datasets demonstrate state-of-the-art performance in camera tracking, map quality, and semantic and geometric accuracy.
>
---
#### [new 099] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云的版权保护问题，揭示现有水印方案在3DGS场景下的脆弱性。提出GSPure水印净化框架，通过分析视图依赖渲染与几何特征聚类，精准移除水印且保持场景质量，显著提升水印移除效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22262v1](https://arxiv.org/pdf/2511.22262v1)**

> **作者:** Wenkai Huang; Yijia Guo; Gaolei Li; Lei Ma; Hang Zhang; Liwen Hu; Jiazheng Wang; Jianhua Li; Tiejun Huang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful representation for 3D scenes, widely adopted due to its exceptional efficiency and high-fidelity visual quality. Given the significant value of 3DGS assets, recent works have introduced specialized watermarking schemes to ensure copyright protection and ownership verification. However, can existing 3D Gaussian watermarking approaches genuinely guarantee robust protection of the 3D assets? In this paper, for the first time, we systematically explore and validate possible vulnerabilities of 3DGS watermarking frameworks. We demonstrate that conventional watermark removal techniques designed for 2D images do not effectively generalize to the 3DGS scenario due to the specialized rendering pipeline and unique attributes of each gaussian primitives. Motivated by this insight, we propose GSPure, the first watermark purification framework specifically for 3DGS watermarking representations. By analyzing view-dependent rendering contributions and exploiting geometrically accurate feature clustering, GSPure precisely isolates and effectively removes watermark-related Gaussian primitives while preserving scene integrity. Extensive experiments demonstrate that our GSPure achieves the best watermark purification performance, reducing watermark PSNR by up to 16.34dB while minimizing degradation to original scene fidelity with less than 1dB PSNR loss. Moreover, it consistently outperforms existing methods in both effectiveness and generalization.
>
---
#### [new 100] DAONet-YOLOv8: An Occlusion-Aware Dual-Attention Network for Tea Leaf Pest and Disease Detection
- **分类: cs.CV**

- **简介: 该论文针对茶园中茶叶病虫害检测难题，提出DAONet-YOLOv8模型。针对复杂背景、光照变化和叶片遮挡导致的漏检与误检问题，设计了双注意力融合模块、遮挡感知检测头和动态卷积模块，有效提升小目标与不完整病斑的检测精度，显著优于YOLOv8n及其他主流模型。**

- **链接: [https://arxiv.org/pdf/2511.23222v1](https://arxiv.org/pdf/2511.23222v1)**

> **作者:** Yefeng Wu; Shan Wan; Ling Wu; Yecheng Zhao
>
> **摘要:** Accurate detection of tea leaf pests and diseases in real plantations remains challenging due to complex backgrounds, variable illumination, and frequent occlusions among dense branches and leaves. Existing detectors often suffer from missed detections and false positives in such scenarios. To address these issues, we propose DAONet-YOLOv8, an enhanced YOLOv8 variant with three key improvements: (1) a Dual-Attention Fusion Module (DAFM) that combines convolutional local feature extraction with self-attention based global context modeling to focus on subtle lesion regions while suppressing background noise; (2) an occlusion-aware detection head (Detect-OAHead) that learns the relationship between visible and occluded parts to compensate for missing lesion features; and (3) a C2f-DSConv module employing dynamic synthesis convolutions with multiple kernel shapes to better capture irregular lesion boundaries. Experiments on our real-world tea plantation dataset containing six pest and disease categories demonstrate that DAONet-YOLOv8 achieves 92.97% precision, 92.80% recall, 97.10% mAP@50 and 76.90% mAP@50:95, outperforming the YOLOv8n baseline by 2.34, 4.68, 1.40 and 1.80 percentage points respectively, while reducing parameters by 16.7%. Comparative experiments further confirm that DAONet-YOLOv8 achieves superior performance over mainstream detection models.
>
---
#### [new 101] McSc: Motion-Corrective Preference Alignment for Video Generation with Self-Critic Hierarchical Reasoning
- **分类: cs.CV**

- **简介: 该论文针对文本到视频生成中的偏好对齐问题，提出McSc框架。通过自批判维度推理、分层比较推理和运动校正的直接偏好优化，实现细粒度偏好建模与高动态视频生成，有效缓解低运动内容偏差，提升生成质量与人类偏好一致性。**

- **链接: [https://arxiv.org/pdf/2511.22974v1](https://arxiv.org/pdf/2511.22974v1)**

> **作者:** Qiushi Yang; Yingjie Chen; Yuan Yao; Yifang Men; Huaizhuo Liu; Miaomiao Cui
>
> **摘要:** Text-to-video (T2V) generation has achieved remarkable progress in producing high-quality videos aligned with textual prompts. However, aligning synthesized videos with nuanced human preference remains challenging due to the subjective and multifaceted nature of human judgment. Existing video preference alignment methods rely on costly human annotations or utilize proxy metrics to predict preference, which lacks the understanding of human preference logic. Moreover, they usually directly align T2V models with the overall preference distribution, ignoring potential conflict dimensions like motion dynamics and visual quality, which may bias models towards low-motion content. To address these issues, we present Motion-corrective alignment with Self-critic hierarchical Reasoning (McSc), a three-stage reinforcement learning framework for robust preference modeling and alignment. Firstly, Self-critic Dimensional Reasoning (ScDR) trains a generative reward model (RM) to decompose preferences into per-dimension assessments, using self-critic reasoning chains for reliable learning. Secondly, to achieve holistic video comparison, we introduce Hierarchical Comparative Reasoning (HCR) for structural multi-dimensional reasoning with hierarchical reward supervision. Finally, using RM-preferred videos, we propose Motion-corrective Direct Preference Optimization (McDPO) to optimize T2V models, while dynamically re-weighting alignment objective to mitigate bias towards low-motion content. Experiments show that McSc achieves superior performance in human preference alignment and generates videos with high-motion dynamic.
>
---
#### [new 102] BrepGPT: Autoregressive B-rep Generation with Voronoi Half-Patch
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出BrepGPT，解决CAD中B-rep生成因几何拓扑耦合导致的多阶段模型误差累积与效率低问题。通过Voronoi Half-Patch统一表示几何与拓扑，结合双VQ-VAE与Transformer实现单阶段自回归生成，支持多种条件生成与补全任务，显著提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.22171v1](https://arxiv.org/pdf/2511.22171v1)**

> **作者:** Pu Li; Wenhao Zhang; Weize Quan; Biao Zhang; Peter Wonka; Dong-Ming Yan
>
> **摘要:** Boundary representation (B-rep) is the de facto standard for CAD model representation in modern industrial design. The intricate coupling between geometric and topological elements in B-rep structures has forced existing generative methods to rely on cascaded multi-stage networks, resulting in error accumulation and computational inefficiency. We present BrepGPT, a single-stage autoregressive framework for B-rep generation. Our key innovation lies in the Voronoi Half-Patch (VHP) representation, which decomposes B-reps into unified local units by assigning geometry to nearest half-edges and sampling their next pointers. Unlike hierarchical representations that require multiple distinct encodings for different structural levels, our VHP representation facilitates unifying geometric attributes and topological relations in a single, coherent format. We further leverage dual VQ-VAEs to encode both vertex topology and Voronoi Half-Patches into vertex-based tokens, achieving a more compact sequential encoding. A decoder-only Transformer is then trained to autoregressively predict these tokens, which are subsequently mapped to vertex-based features and decoded into complete B-rep models. Experiments demonstrate that BrepGPT achieves state-of-the-art performance in unconditional B-rep generation. The framework also exhibits versatility in various applications, including conditional generation from category labels, point clouds, text descriptions, and images, as well as B-rep autocompletion and interpolation.
>
---
#### [new 103] Architecture Decoupling Is Not All You Need For Unified Multimodal Model
- **分类: cs.CV**

- **简介: 该论文研究统一多模态模型在图像生成与理解任务中的性能提升问题。针对模型解耦导致的交互能力下降问题，提出注意力交互对齐（AIA）损失，显式学习任务特异性跨模态交互模式，有效缓解任务冲突，提升生成与理解性能，且无需复杂结构设计。**

- **链接: [https://arxiv.org/pdf/2511.22663v1](https://arxiv.org/pdf/2511.22663v1)**

> **作者:** Dian Zheng; Manyuan Zhang; Hongyu Li; Kai Zou; Hongbo Liu; Ziyu Guo; Kaituo Feng; Yexin Liu; Ying Luo; Yan Feng; Peng Pei; Xunliang Cai; Hongsheng Li
>
> **备注:** Project page: https://zhengdian1.github.io/AIA-project/ Code: https://github.com/zhengdian1/AIA
>
> **摘要:** Unified multimodal models for image generation and understanding represent a significant step toward AGI and have attracted widespread attention from researchers. The main challenge of this task lies in the difficulty in establishing an optimal training paradigm due to inherent conflicting targets in understanding and generation tasks. To alleviate these conflicts and pursue higher performance, many researchers adopt varying degrees of model decoupling (e.g., Double image encoders, MOE/MOT architecture, or frozen MLLM). However, excessive model decoupling can lead to the loss of interleave generation ability, undermining the original intent of unified models. In this work, we aim to explore how to mitigate task conflicts without resorting to model decoupling. Firstly, we analyze why decoupling alleviates conflicts by studying the cross-modal attention behavior of models. We observe that model decoupling essentially drives models toward task-specific multimodal interaction patterns, as seen in Qwen-VL and HunyuanImage, and that the more thorough the decoupling, the more consistent the behavior becomes. Motivated by this observation, we propose Attention Interaction Alignment (AIA) loss, which explicitly learns Task-Specific multimodal interaction patterns during training. To demonstrate the generalizability of our AIA loss, we apply it to Emu3 and Janus-Pro during SFT and post-training stage respectively. Without bells and whistles, AIA not only refines cross-modal attention patterns, but also boosts both generation and understanding performance.
>
---
#### [new 104] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文提出多参考文本到图像生成基准MultiBanana，旨在解决现有数据集在多参考生成任务中覆盖不足、定义模糊的问题。通过设计多样化挑战场景，评估模型在多参考条件下的性能与缺陷，推动该领域标准化评测与技术进步。**

- **链接: [https://arxiv.org/pdf/2511.22989v1](https://arxiv.org/pdf/2511.22989v1)**

> **作者:** Yuta Oshima; Daiki Miyake; Kohsei Matsutani; Yusuke Iwasawa; Masahiro Suzuki; Yutaka Matsuo; Hiroki Furuta
>
> **备注:** Code: https://github.com/matsuolab/multibanana
>
> **摘要:** Recent text-to-image generation models have acquired the ability of multi-reference generation and editing; the ability to inherit the appearance of subjects from multiple reference images and re-render them under new contexts. However, the existing benchmark datasets often focus on the generation with single or a few reference images, which prevents us from measuring the progress on how model performance advances or pointing out their weaknesses, under different multi-reference conditions. In addition, their task definitions are still vague, typically limited to axes such as "what to edit" or "how many references are given", and therefore fail to capture the intrinsic difficulty of multi-reference settings. To address this gap, we introduce $\textbf{MultiBanana}$, which is carefully designed to assesses the edge of model capabilities by widely covering multi-reference-specific problems at scale: (1) varying the number of references, (2) domain mismatch among references (e.g., photo vs. anime), (3) scale mismatch between reference and target scenes, (4) references containing rare concepts (e.g., a red banana), and (5) multilingual textual references for rendering. Our analysis among a variety of text-to-image models reveals their superior performances, typical failure modes, and areas for improvement. MultiBanana will be released as an open benchmark to push the boundaries and establish a standardized basis for fair comparison in multi-reference image generation. Our data and code are available at https://github.com/matsuolab/multibanana .
>
---
#### [new 105] MIMM-X: Disentangling Spurious Correlations for Medical Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分析中深度学习模型因多重伪相关导致的泛化能力差问题，提出MIMM-X框架。通过最小化因果特征与伪相关之间的互信息，实现特征解耦，使模型基于真实因果关系做出预测。在多数据集、多模态实验中验证了其有效缓解快捷学习的能力。**

- **链接: [https://arxiv.org/pdf/2511.22990v1](https://arxiv.org/pdf/2511.22990v1)**

> **作者:** Louisa Fay; Hajer Reguigui; Bin Yang; Sergios Gatidis; Thomas Küstner
>
> **摘要:** Deep learning models can excel on medical tasks, yet often experience spurious correlations, known as shortcut learning, leading to poor generalization in new environments. Particularly in medical imaging, where multiple spurious correlations can coexist, misclassifications can have severe consequences. We propose MIMM-X, a framework that disentangles causal features from multiple spurious correlations by minimizing their mutual information. It enables predictions based on true underlying causal relationships rather than dataset-specific shortcuts. We evaluate MIMM-X on three datasets (UK Biobank, NAKO, CheXpert) across two imaging modalities (MRI and X-ray). Results demonstrate that MIMM-X effectively mitigates shortcut learning of multiple spurious correlations.
>
---
#### [new 106] SkeletonAgent: An Agentic Interaction Framework for Skeleton-based Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对骨架动作识别中语义信息利用不足的问题，提出SkeletonAgent框架。通过“提问者”与“选择器”双代理机制，使大语言模型（LLM）与识别模型协同交互，基于混淆类别提供精准语义引导，并反馈关节级约束，实现跨模态细粒度对齐，显著提升识别性能。**

- **链接: [https://arxiv.org/pdf/2511.22433v1](https://arxiv.org/pdf/2511.22433v1)**

> **作者:** Hongda Liu; Yunfan Liu; Changlu Wang; Yunlong Wang; Zhenan Sun
>
> **摘要:** Recent advances in skeleton-based action recognition increasingly leverage semantic priors from Large Language Models (LLMs) to enrich skeletal representations. However, the LLM is typically queried in isolation from the recognition model and receives no performance feedback. As a result, it often fails to deliver the targeted discriminative cues critical to distinguish similar actions. To overcome these limitations, we propose SkeletonAgent, a novel framework that bridges the recognition model and the LLM through two cooperative agents, i.e., Questioner and Selector. Specifically, the Questioner identifies the most frequently confused classes and supplies them to the LLM as context for more targeted guidance. Conversely, the Selector parses the LLM's response to extract precise joint-level constraints and feeds them back to the recognizer, enabling finer-grained cross-modal alignment. Comprehensive evaluations on five benchmarks, including NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton, FineGYM, and UAV-Human, demonstrate that SkeletonAgent consistently outperforms state-of-the-art benchmark methods. The code is available at https://github.com/firework8/SkeletonAgent.
>
---
#### [new 107] Captain Safari: A World Engine
- **分类: cs.CV**

- **简介: 该论文提出Captain Safari，一种基于姿态条件世界记忆的视频生成方法，旨在解决长时序、6-DoF复杂运动下3D一致性差与轨迹偏离问题。通过动态检索对齐的世界令牌，实现高精度相机路径跟随与稳定3D结构生成。构建OpenSafari数据集进行评估，实验表明其在视频质量、3D一致性和轨迹跟踪上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22815v1](https://arxiv.org/pdf/2511.22815v1)**

> **作者:** Yu-Cheng Chou; Xingrui Wang; Yitong Li; Jiahao Wang; Hanting Liu; Cihang Xie; Alan Yuille; Junfei Xiao
>
> **摘要:** World engines aim to synthesize long, 3D-consistent videos that support interactive exploration of a scene under user-controlled camera motion. However, existing systems struggle under aggressive 6-DoF trajectories and complex outdoor layouts: they lose long-range geometric coherence, deviate from the target path, or collapse into overly conservative motion. To this end, we introduce Captain Safari, a pose-conditioned world engine that generates videos by retrieving from a persistent world memory. Given a camera path, our method maintains a dynamic local memory and uses a retriever to fetch pose-aligned world tokens, which then condition video generation along the trajectory. This design enables the model to maintain stable 3D structure while accurately executing challenging camera maneuvers. To evaluate this setting, we curate OpenSafari, a new in-the-wild FPV dataset containing high-dynamic drone videos with verified camera trajectories, constructed through a multi-stage geometric and kinematic validation pipeline. Across video quality, 3D consistency, and trajectory following, Captain Safari substantially outperforms state-of-the-art camera-controlled generators. It reduces MEt3R from 0.3703 to 0.3690, improves AUC@30 from 0.181 to 0.200, and yields substantially lower FVD than all camera-controlled baselines. More importantly, in a 50-participant, 5-way human study where annotators select the best result among five anonymized models, 67.6% of preferences favor our method across all axes. Our results demonstrate that pose-conditioned world memory is a powerful mechanism for long-horizon, controllable video generation and provide OpenSafari as a challenging new benchmark for future world-engine research.
>
---
#### [new 108] GeoWorld: Unlocking the Potential of Geometry Models to Facilitate High-Fidelity 3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文针对图像到3D场景生成中几何失真与模糊的问题，提出GeoWorld框架。通过先生成连续视频帧，再利用几何模型提取全帧几何特征作为条件，结合几何对齐损失与自适应模块，提升几何结构一致性，实现高保真3D场景生成。**

- **链接: [https://arxiv.org/pdf/2511.23191v1](https://arxiv.org/pdf/2511.23191v1)**

> **作者:** Yuhao Wan; Lijuan Liu; Jingzhi Zhou; Zihan Zhou; Xuying Zhang; Dongbo Zhang; Shaohui Jiao; Qibin Hou; Ming-Ming Cheng
>
> **摘要:** Previous works leveraging video models for image-to-3D scene generation tend to suffer from geometric distortions and blurry content. In this paper, we renovate the pipeline of image-to-3D scene generation by unlocking the potential of geometry models and present our GeoWorld. Instead of exploiting geometric information obtained from a single-frame input, we propose to first generate consecutive video frames and then take advantage of the geometry model to provide full-frame geometry features, which contain richer information than single-frame depth maps or camera embeddings used in previous methods, and use these geometry features as geometrical conditions to aid the video generation model. To enhance the consistency of geometric structures, we further propose a geometry alignment loss to provide the model with real-world geometric constraints and a geometry adaptation module to ensure the effective utilization of geometry features. Extensive experiments show that our GeoWorld can generate high-fidelity 3D scenes from a single image and a given camera trajectory, outperforming prior methods both qualitatively and quantitatively. Project Page: https://peaes.github.io/GeoWorld/.
>
---
#### [new 109] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment
- **分类: cs.CV**

- **简介: 该论文针对正常化流（NFs）生成质量与语义表征弱的问题，提出基于逆向传递特征对齐的新策略，利用视觉基础模型提升表征能力，并设计无需训练的测试时优化方法评估语义知识。实验表明，该方法显著加速训练、提升生成与分类性能，刷新了ImageNet上的NFs新纪录。**

- **链接: [https://arxiv.org/pdf/2511.22345v1](https://arxiv.org/pdf/2511.22345v1)**

> **作者:** Yang Chen; Xiaowei Xu; Shuai Wang; Chenhui Zhu; Ruxue Wen; Xubin Li; Tiezheng Ge; Limin Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Normalizing Flows (NFs) are a class of generative models distinguished by a mathematically invertible architecture, where the forward pass transforms data into a latent space for density estimation, and the reverse pass generates new samples from this space. This characteristic creates an intrinsic synergy between representation learning and data generation. However, the generative quality of standard NFs is limited by poor semantic representations from log-likelihood optimization. To remedy this, we propose a novel alignment strategy that creatively leverages the invertibility of NFs: instead of regularizing the forward pass, we align the intermediate features of the generative (reverse) pass with representations from a powerful vision foundation model, demonstrating superior effectiveness over naive alignment. We also introduce a novel training-free, test-time optimization algorithm for classification, which provides a more intrinsic evaluation of the NF's embedded semantic knowledge. Comprehensive experiments demonstrate that our approach accelerates the training of NFs by over 3.3$\times$, while simultaneously delivering significant improvements in both generative quality and classification accuracy. New state-of-the-art results for NFs are established on ImageNet 64$\times$64 and 256$\times$256. Our code is available at https://github.com/MCG-NJU/FlowBack.
>
---
#### [new 110] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文提出一种可扩展的扩散Transformer模型，用于条件化生成全脑4D fMRI序列。针对高维异质性与缺乏神经科学验证的挑战，结合VQ-GAN压缩与CNN-Transformer架构，通过AdaLN-Zero和交叉注意力实现强任务条件控制，有效复现任务诱发激活、保留表征结构，性能优于U-Net基线。**

- **链接: [https://arxiv.org/pdf/2511.22870v1](https://arxiv.org/pdf/2511.22870v1)**

> **作者:** Jungwoo Seo; David Keetae Park; Shinjae Yoo; Jiook Cha
>
> **备注:** Accepted at NeurIPS 2025 Workshop: Foundation Models for the Brain and Body. 13 pages, 6 figures, 4 tables
>
> **摘要:** Generating whole-brain 4D fMRI sequences conditioned on cognitive tasks remains challenging due to the high-dimensional, heterogeneous BOLD dynamics across subjects/acquisitions and the lack of neuroscience-grounded validation. We introduce the first diffusion transformer for voxelwise 4D fMRI conditional generation, combining 3D VQ-GAN latent compression with a CNN-Transformer backbone and strong task conditioning via AdaLN-Zero and cross-attention. On HCP task fMRI, our model reproduces task-evoked activation maps, preserves the inter-task representational structure observed in real data (RSA), achieves perfect condition specificity, and aligns ROI time-courses with canonical hemodynamic responses. Performance improves predictably with scale, reaching task-evoked map correlation of 0.83 and RSA of 0.98, consistently surpassing a U-Net baseline on all metrics. By coupling latent diffusion with a scalable backbone and strong conditioning, this work establishes a practical path to conditional 4D fMRI synthesis, paving the way for future applications such as virtual experiments, cross-site harmonization, and principled augmentation for downstream neuroimaging models.
>
---
#### [new 111] DiffStyle360: Diffusion-Based 360° Head Stylization via Style Fusion Attention
- **分类: cs.CV**

- **简介: 该论文提出DiffStyle360，一种基于扩散模型的360°头像风格化方法，旨在实现多视角一致、身份保留的跨风格生成。针对现有方法依赖昂贵优化或特定训练的问题，提出风格解耦模块与风格融合注意力机制，仅需单张参考图即可生成高质量风格化头像，显著提升风格保真度与生成效率。**

- **链接: [https://arxiv.org/pdf/2511.22411v1](https://arxiv.org/pdf/2511.22411v1)**

> **作者:** Furkan Guzelant; Arda Goktogan; Tarık Kaya; Aysegul Dundar
>
> **摘要:** 3D head stylization has emerged as a key technique for reimagining realistic human heads in various artistic forms, enabling expressive character design and creative visual experiences in digital media. Despite the progress in 3D-aware generation, existing 3D head stylization methods often rely on computationally expensive optimization or domain-specific fine-tuning to adapt to new styles. To address these limitations, we propose DiffStyle360, a diffusion-based framework capable of producing multi-view consistent, identity-preserving 3D head stylizations across diverse artistic domains given a single style reference image, without requiring per-style training. Building upon the 3D-aware DiffPortrait360 architecture, our approach introduces two key components: the Style Appearance Module, which disentangles style from content, and the Style Fusion Attention mechanism, which adaptively balances structure preservation and stylization fidelity in the latent space. Furthermore, we employ a 3D GAN-generated multi-view dataset for robust fine-tuning and introduce a temperaturebased key scaling strategy to control stylization intensity during inference. Extensive experiments on FFHQ and RenderMe360 demonstrate that DiffStyle360 achieves superior style quality, outperforming state-of-the-art GAN- and diffusion-based stylization methods across challenging style domains.
>
---
#### [new 112] PAGen: Phase-guided Amplitude Generation for Domain-adaptive Object Detection
- **分类: cs.CV**

- **简介: 该论文针对无监督域适应中的目标检测任务，解决源域与目标域间图像风格差异导致性能下降的问题。提出PAGen方法，通过频域相位引导的幅度生成，轻量级地适应图像风格，训练时添加简单预处理模块，推理时移除，无额外开销。在多个基准上实现显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.22029v1](https://arxiv.org/pdf/2511.22029v1)**

> **作者:** Shuchen Du; Shuo Lei; Feiran Li; Jiacheng Li; Daisuke Iso
>
> **摘要:** Unsupervised domain adaptation (UDA) greatly facilitates the deployment of neural networks across diverse environments. However, most state-of-the-art approaches are overly complex, relying on challenging adversarial training strategies, or on elaborate architectural designs with auxiliary models for feature distillation and pseudo-label generation. In this work, we present a simple yet effective UDA method that learns to adapt image styles in the frequency domain to reduce the discrepancy between source and target domains. The proposed approach introduces only a lightweight pre-processing module during training and entirely discards it at inference time, thus incurring no additional computational overhead. We validate our method on domain-adaptive object detection (DAOD) tasks, where ground-truth annotations are easily accessible in source domains (e.g., normal-weather or synthetic conditions) but challenging to obtain in target domains (e.g., adverse weather or low-light scenes). Extensive experiments demonstrate that our method achieves substantial performance gains on multiple benchmarks, highlighting its practicality and effectiveness.
>
---
#### [new 113] Interpretable Multimodal Cancer Prototyping with Whole Slide Images and Incompletely Paired Genomics
- **分类: cs.CV**

- **简介: 该论文针对癌症精准医疗中病理图像与基因组数据不完整配对的问题，提出可解释的多模态原型框架。通过生物原型构建、多视图对齐、双分图融合与语义基因组补全，实现全切片图像与不完整基因组的有效整合，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.21937v1](https://arxiv.org/pdf/2511.21937v1)**

> **作者:** Yupei Zhang; Yating Huang; Wanming Hu; Lequan Yu; Hujun Yin; Chao Li
>
> **摘要:** Multimodal approaches that integrate histology and genomics hold strong potential for precision oncology. However, phenotypic and genotypic heterogeneity limits the quality of intra-modal representations and hinders effective inter-modal integration. Furthermore, most existing methods overlook real-world clinical scenarios where genomics may be partially missing or entirely unavailable. We propose a flexible multimodal prototyping framework to integrate whole slide images and incomplete genomics for precision oncology. Our approach has four key components: 1) Biological Prototyping using text prompting and prototype-wise weighting; 2) Multiview Alignment through sample- and distribution-wise alignments; 3) Bipartite Fusion to capture both shared and modality-specific information for multimodal fusion; and 4) Semantic Genomics Imputation to handle missing data. Extensive experiments demonstrate the consistent superiority of the proposed method compared to other state-of-the-art approaches on multiple downstream tasks. The code is available at https://github.com/helenypzhang/Interpretable-Multimodal-Prototyping.
>
---
#### [new 114] Resolving Evidence Sparsity: Agentic Context Engineering for Long-Document Understanding
- **分类: cs.CV**

- **简介: 该论文针对长文档理解中证据稀疏与冗余问题，提出SLEUTH多智能体框架。通过协同检索、关键线索提取与多模态证据浓缩，实现从粗到细的上下文优化，提升视觉语言模型在长文档上的推理性能，显著改善了跨页信息整合与判别准确率。**

- **链接: [https://arxiv.org/pdf/2511.22850v1](https://arxiv.org/pdf/2511.22850v1)**

> **作者:** Keliang Liu; Zizhi Chen; Mingcheng Li; Jingqun Tang; Dingkang Yang; Lihua Zhang
>
> **摘要:** Document understanding is a long standing practical task. Vision Language Models (VLMs) have gradually become a primary approach in this domain, demonstrating effective performance on single page tasks. However, their effectiveness diminishes when handling long documents. In such scenarios, clues are often scattered across multiple pages and modalities, and redundancy from lengthy inputs can impair the models judgment. While retrieval augmented generation mitigates this issue by filtering for question relevant content, the retrieved results still contain substantial redundancy. To address these limitations, we propose SLEUTH, a multi agent framework. Concretely, SLEUTH orchestrates a retriever and four collaborative agents in a coarse to fine process. The framework identifies key textual and visual clues within the retrieved pages, filters for salient visual evidence such as tables and charts, and analyzes the query to devise a reasoning strategy. It ultimately synthesizes a distilled, evidence dense multimodal context to generate the final prediction. SLEUTH is model agnostic and scalable. When paired with advanced VLM backbones, it consistently improves performance on multiple long document benchmarks, achieving state of the art results. Ablation studies verify each modules effectiveness and confirm the benefits of our hierarchical refinement paradigm.
>
---
#### [new 115] Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出Z-Image，一个6B参数的高效图像生成基础模型，针对现有模型参数庞大、难以在消费级硬件上部署的问题。通过单流扩散变压器架构与优化训练流程，实现快速训练与低延迟推理，支持消费级硬件，并在真实感与双语文本生成上达到顶尖水平。**

- **链接: [https://arxiv.org/pdf/2511.22699v1](https://arxiv.org/pdf/2511.22699v1)**

> **作者:** Z-Image Team; Huanqia Cai; Sihan Cao; Ruoyi Du; Peng Gao; Steven Hoi; Shijie Huang; Zhaohui Hou; Dengyang Jiang; Xin Jin; Liangchen Li; Zhen Li; Zhong-Yu Li; David Liu; Dongyang Liu; Junhan Shi; Qilong Wu; Feng Yu; Chi Zhang; Shifeng Zhang; Shilin Zhou
>
> **摘要:** The landscape of high-performance image generation models is currently dominated by proprietary systems, such as Nano Banana Pro and Seedream 4.0. Leading open-source alternatives, including Qwen-Image, Hunyuan-Image-3.0 and FLUX.2, are characterized by massive parameter counts (20B to 80B), making them impractical for inference, and fine-tuning on consumer-grade hardware. To address this gap, we propose Z-Image, an efficient 6B-parameter foundation generative model built upon a Scalable Single-Stream Diffusion Transformer (S3-DiT) architecture that challenges the "scale-at-all-costs" paradigm. By systematically optimizing the entire model lifecycle -- from a curated data infrastructure to a streamlined training curriculum -- we complete the full training workflow in just 314K H800 GPU hours (approx. $630K). Our few-step distillation scheme with reward post-training further yields Z-Image-Turbo, offering both sub-second inference latency on an enterprise-grade H800 GPU and compatibility with consumer-grade hardware (<16GB VRAM). Additionally, our omni-pre-training paradigm also enables efficient training of Z-Image-Edit, an editing model with impressive instruction-following capabilities. Both qualitative and quantitative experiments demonstrate that our model achieves performance comparable to or surpassing that of leading competitors across various dimensions. Most notably, Z-Image exhibits exceptional capabilities in photorealistic image generation and bilingual text rendering, delivering results that rival top-tier commercial models, thereby demonstrating that state-of-the-art results are achievable with significantly reduced computational overhead. We publicly release our code, weights, and online demo to foster the development of accessible, budget-friendly, yet state-of-the-art generative models.
>
---
#### [new 116] Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Percept-WAM，一种融合2D/3D感知的视觉语言模型，解决自动驾驶中空间理解弱、长尾场景鲁棒性差的问题。通过引入World-PV/BEV tokens和网格条件预测机制，统一感知与决策，提升小物体、远距离等复杂场景下的检测与规划性能。**

- **链接: [https://arxiv.org/pdf/2511.19221v1](https://arxiv.org/pdf/2511.19221v1)**

> **作者:** Jianhua Han; Meng Tian; Jiangtong Zhu; Fan He; Huixin Zhang; Sitong Guo; Dechang Zhu; Hao Tang; Pei Xu; Yuze Guo; Minzhe Niu; Haojie Zhu; Qichao Dong; Xuechao Yan; Siyuan Dong; Lu Hou; Qingqiu Huang; Xiaosong Jia; Hang Xu
>
> **摘要:** Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.
>
---
#### [new 117] GazeTrack: High-Precision Eye Tracking Based on Regularization and Spatial Computing
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 该论文针对虚拟/增强现实中的高精度眼动追踪问题，提出GazeTrack框架与数据集。通过形状正则化优化瞳孔拟合，创新坐标变换方法提升注视向量预测精度，构建低复杂度高精度眼动追踪模型，显著降低角度误差。**

- **链接: [https://arxiv.org/pdf/2511.22607v1](https://arxiv.org/pdf/2511.22607v1)**

> **作者:** Xiaoyin Yang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Eye tracking has become increasingly important in virtual and augmented reality applications; however, the current gaze accuracy falls short of meeting the requirements for spatial computing. We designed a gaze collection framework and utilized high-precision equipment to gather the first precise benchmark dataset, GazeTrack, encompassing diverse ethnicities, ages, and visual acuity conditions for pupil localization and gaze tracking. We propose a novel shape error regularization method to constrain pupil ellipse fitting and train on open-source datasets, enhancing semantic segmentation and pupil position prediction accuracy. Additionally, we invent a novel coordinate transformation method similar to paper unfolding to accurately predict gaze vectors on the GazeTrack dataset. Finally, we built a gaze vector generation model that achieves reduced gaze angle error with lower computational complexity compared to other methods.
>
---
#### [new 118] World in a Frame: Understanding Culture Mixing as a New Challenge for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型在文化混杂场景下的表现，属于跨文化理解任务。针对模型在多文化元素共现时无法保持文化身份一致的问题，构建了CultureMix VQA基准，评估10个模型并发现其对背景依赖性强、预测不一致。通过微调提升模型鲁棒性，呼吁关注文化混杂挑战以推动模型在多元现实环境中的可靠应用。**

- **链接: [https://arxiv.org/pdf/2511.22787v1](https://arxiv.org/pdf/2511.22787v1)**

> **作者:** Eunsu Kim; Junyeong Park; Na Min An; Junseong Kim; Hitesh Laxmichand Patel; Jiho Jin; Julia Kruk; Amit Agarwal; Srikant Panda; Fenal Ashokbhai Ilasariya; Hyunjung Shim; Alice Oh
>
> **摘要:** In a globalized world, cultural elements from diverse origins frequently appear together within a single visual scene. We refer to these as culture mixing scenarios, yet how Large Vision-Language Models (LVLMs) perceive them remains underexplored. We investigate culture mixing as a critical challenge for LVLMs and examine how current models behave when cultural items from multiple regions appear together. To systematically analyze these behaviors, we construct CultureMix, a food Visual Question Answering (VQA) benchmark with 23k diffusion-generated, human-verified culture mixing images across four subtasks: (1) food-only, (2) food+food, (3) food+background, and (4) food+food+background. Evaluating 10 LVLMs, we find consistent failures to preserve individual cultural identities in mixed settings. Models show strong background reliance, with accuracy dropping 14% when cultural backgrounds are added to food-only baselines, and they produce inconsistent predictions for identical foods across different contexts. To address these limitations, we explore three robustness strategies. We find supervised fine-tuning using a diverse culture mixing dataset substantially improve model consistency and reduce background sensitivity. We call for increased attention to culture mixing scenarios as a critical step toward developing LVLMs capable of operating reliably in culturally diverse real-world environments.
>
---
#### [new 119] Rethinking Cross-Generator Image Forgery Detection through DINOv3
- **分类: cs.CV**

- **简介: 该论文研究图像伪造检测任务，针对现有方法对特定生成器依赖性强、泛化能力差的问题，发现冻结的DINOv3模型具备天然的跨生成器检测能力。通过分析其依赖全局低频结构作为通用真伪线索，提出无需训练的令牌排序与轻量线性探针策略，筛选关键令牌以提升检测性能，提供高效可解释的基准方案。**

- **链接: [https://arxiv.org/pdf/2511.22471v1](https://arxiv.org/pdf/2511.22471v1)**

> **作者:** Zhenglin Huang; Jason Li; Haiquan Wen; Tianxiao Li; Xi Yang; Lu Qi; Bei Peng; Xiaowei Huang; Ming-Hsuan Yang; Guangliang Cheng
>
> **摘要:** As generative models become increasingly diverse and powerful, cross-generator detection has emerged as a new challenge. Existing detection methods often memorize artifacts of specific generative models rather than learning transferable cues, leading to substantial failures on unseen generators. Surprisingly, this work finds that frozen visual foundation models, especially DINOv3, already exhibit strong cross-generator detection capability without any fine-tuning. Through systematic studies on frequency, spatial, and token perspectives, we observe that DINOv3 tends to rely on global, low-frequency structures as weak but transferable authenticity cues instead of high-frequency, generator-specific artifacts. Motivated by this insight, we introduce a simple, training-free token-ranking strategy followed by a lightweight linear probe to select a small subset of authenticity-relevant tokens. This token subset consistently improves detection accuracy across all evaluated datasets. Our study provides empirical evidence and a feasible hypothesis for understanding why foundation models generalize across diverse generators, offering a universal, efficient, and interpretable baseline for image forgery detection.
>
---
#### [new 120] Real-Time Long Horizon Air Quality Forecasting via Group-Relative Policy Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对长时序空气质量预报任务，解决复杂地形下模型误报率高、与实际预警需求不匹配的问题。提出GRPO方法，结合分组相对奖励与课程式推理，提升预测可靠性。构建并发布高分辨率观测数据集，实现48-120小时实时预报，显著降低误报率。**

- **链接: [https://arxiv.org/pdf/2511.22169v1](https://arxiv.org/pdf/2511.22169v1)**

> **作者:** Inha Kang; Eunki Kim; Wonjeong Ryu; Jaeyo Shin; Seungjun Yu; Yoon-Hee Kang; Seongeun Jeong; Eunhye Kim; Soontae Kim; Hyunjung Shim
>
> **备注:** 10 pages
>
> **摘要:** Accurate long horizon forecasting of particulate matter (PM) concentration fields is essential for operational public health decisions. However, achieving reliable forecasts remains challenging in regions with complex terrain and strong atmospheric dynamics such as East Asia. While foundation models such as Aurora offer global generality, they often miss region-specific dynamics and rely on non-real-time inputs, limiting their practical utility for localized warning systems. To address this gap, we construct and release the real-world observations and high-resolution CMAQ-OBS dataset for East Asia, reducing regional error by 59.5% and enabling real-time 48-120 hour forecasts critical for public health alerts. However, standard point-wise objectives cannot reflect asymmetric operational costs, where false alarms deteriorate public trust while missed severe events endanger populations. This cost mismatch causes SFT models to over-predict and yield high False Alarm Rates. We introduce Group-Relative Policy Optimization (GRPO) with class-wise rewards and curriculum rollout to align predictions with operational priorities. Experimental results demonstrate that our framework significantly improves the reliability of the forecast. Compared to the SFT-only baseline, our model reduces the False Alarm Rate by 47.3% while achieving a competitive F1-score, proving its effectiveness for practical, real-world air quality forecasting systems on long lead time scenarios.
>
---
#### [new 121] SpaceMind: Camera-Guided Modality Fusion for Spatial Reasoning in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型在3D空间推理（如距离估计、尺寸比较）上的不足，提出SpaceMind模型。通过将相机参数作为主动引导模态，设计轻量级相机引导模态融合模块，增强RGB输入下的空间理解能力。实验表明，该方法在多个基准上达到新SOTA，显著提升模型的空间推理性能。**

- **链接: [https://arxiv.org/pdf/2511.23075v1](https://arxiv.org/pdf/2511.23075v1)**

> **作者:** Ruosen Zhao; Zhikang Zhang; Jialei Xu; Jiahao Chang; Dong Chen; Lingyun Li; Weijian Sun; Zizhuang Wei
>
> **摘要:** Large vision-language models (VLMs) show strong multimodal understanding but still struggle with 3D spatial reasoning, such as distance estimation, size comparison, and cross-view consistency. Existing 3D-aware methods either depend on auxiliary 3D information or enhance RGB-only VLMs with geometry encoders through shallow feature fusion. We propose SpaceMind, a multimodal large language model explicitly designed for spatial reasoning solely from RGB inputs. The model adopts a dual-encoder architecture, integrating VGGT as a spatial understanding encoder and InternViT as a 2D visual encoder. The key idea is to treat the camera representation as an active guiding modality rather than passive metadata. Specifically, SpaceMind introduces a lightweight Camera-Guided Modality Fusion module before the language model to replace shallow fusion. It applies camera-conditioned biasing to spatial tokens, assigns query-independent weights reflecting their geometric importance, and uses the camera embedding to gate the fused representation. Empirically, SpaceMind establishes new state-of-the-art results on VSI-Bench, SQA3D and SPBench, surpassing both open and proprietary systems on VSI-Bench and SPBench by large margins and achieving state-of-the-art performance on SQA3D. These results demonstrate that camera-guided modality fusion is an effective and practical inductive bias for equipping VLMs with genuinely spatially grounded intelligence. We will release code and model checkpoints to support future research.
>
---
#### [new 122] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型在实时任务指导中的不足，提出新基准与数据集Qualcomm Interactive Cooking，包含带时间戳的错误提醒。研究引入LiveMamba模型，实现对视频流的异步响应，解决实时交互式步骤引导难题，为未来AI助手机能提供首个专用评测基准与强基线。**

- **链接: [https://arxiv.org/pdf/2511.21998v1](https://arxiv.org/pdf/2511.21998v1)**

> **作者:** Apratim Bhattacharyya; Bicheng Xu; Sanjay Haresh; Reza Pourreza; Litian Liu; Sunny Panchal; Pulkit Madan; Leonid Sigal; Roland Memisevic
>
> **备注:** Accepted to NeurIPS 2025 (Project page: https://apratimbh.github.io/livecook)
>
> **摘要:** Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.
>
---
#### [new 123] Learning to Predict Aboveground Biomass from RGB Images with 3D Synthetic Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于单张地面RGB图像预测森林地上生物量（AGB）的新方法。针对传统方法在密集植被中精度不足的问题，利用3D合成数据集生成AGB密度图，通过深度学习模型预测并整合得到AGB估计，实现高效、低成本的森林监测。**

- **链接: [https://arxiv.org/pdf/2511.23249v1](https://arxiv.org/pdf/2511.23249v1)**

> **作者:** Silvia Zuffi
>
> **备注:** Presented at STAG 2025
>
> **摘要:** Forests play a critical role in global ecosystems by supporting biodiversity and mitigating climate change via carbon sequestration. Accurate aboveground biomass (AGB) estimation is essential for assessing carbon storage and wildfire fuel loads, yet traditional methods rely on labor-intensive field measurements or remote sensing approaches with significant limitations in dense vegetation. In this work, we propose a novel learning-based method for estimating AGB from a single ground-based RGB image. We frame this as a dense prediction task, introducing AGB density maps, where each pixel represents tree biomass normalized by the plot area and each tree's image area. We leverage the recently introduced synthetic 3D SPREAD dataset, which provides realistic forest scenes with per-image tree attributes (height, trunk and canopy diameter) and instance segmentation masks. Using these assets, we compute AGB via allometric equations and train a model to predict AGB density maps, integrating them to recover the AGB estimate for the captured scene. Our approach achieves a median AGB estimation error of 1.22 kg/m^2 on held-out SPREAD data and 1.94 kg/m^2 on a real-image dataset. To our knowledge, this is the first method to estimate aboveground biomass directly from a single RGB image, opening up the possibility for a scalable, interpretable, and cost-effective solution for forest monitoring, while also enabling broader participation through citizen science initiatives.
>
---
#### [new 124] Robust Image Self-Recovery against Tampering using Watermark Generation with Pixel Shuffling
- **分类: cs.CV**

- **简介: 该论文针对AIGC带来的图像真实性问题，提出ReImage框架，通过神经水印技术将图像像素打乱后嵌入自身实现自恢复。解决了现有方法恢复精度低的问题，设计了优化水印生成与图像增强模块，显著提升多种篡改场景下的恢复质量。**

- **链接: [https://arxiv.org/pdf/2511.22936v1](https://arxiv.org/pdf/2511.22936v1)**

> **作者:** Minyoung Kim; Paul Hongsuck Seo
>
> **备注:** 22 pages, 12 figures, 14 tables
>
> **摘要:** The rapid growth of Artificial Intelligence-Generated Content (AIGC) raises concerns about the authenticity of digital media. In this context, image self-recovery, reconstructing original content from its manipulated version, offers a practical solution for understanding the attacker's intent and restoring trustworthy data. However, existing methods often fail to accurately recover tampered regions, falling short of the primary goal of self-recovery. To address this challenge, we propose ReImage, a neural watermarking-based self-recovery framework that embeds a shuffled version of the target image into itself as a watermark. We design a generator that produces watermarks optimized for neural watermarking and introduce an image enhancement module to refine the recovered image. We further analyze and resolve key limitations of shuffled watermarking, enabling its effective use in self-recovery. We demonstrate that ReImage achieves state-of-the-art performance across diverse tampering scenarios, consistently producing high-quality recovered images. The code and pretrained models will be released upon publication.
>
---
#### [new 125] PAT3D: Physics-Augmented Text-to-3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出PAT3D，首个融合物理模拟的文本生成3D场景框架。针对现有方法生成场景缺乏物理合理性、存在物体穿插的问题，通过视觉语言模型理解文本，构建层次化场景树，并结合可微分刚体模拟器优化至静态平衡，确保物理真实、无交叠且语义一致，生成可直接用于仿真与机器人操作的高质量3D场景。**

- **链接: [https://arxiv.org/pdf/2511.21978v1](https://arxiv.org/pdf/2511.21978v1)**

> **作者:** Guying Lin; Kemeng Huang; Michael Liu; Ruihan Gao; Hanke Chen; Lyuhao Chen; Beijia Lu; Taku Komura; Yuan Liu; Jun-Yan Zhu; Minchen Li
>
> **备注:** 19 pages, 12 figures
>
> **摘要:** We introduce PAT3D, the first physics-augmented text-to-3D scene generation framework that integrates vision-language models with physics-based simulation to produce physically plausible, simulation-ready, and intersection-free 3D scenes. Given a text prompt, PAT3D generates 3D objects, infers their spatial relations, and organizes them into a hierarchical scene tree, which is then converted into initial conditions for simulation. A differentiable rigid-body simulator ensures realistic object interactions under gravity, driving the scene toward static equilibrium without interpenetrations. To further enhance scene quality, we introduce a simulation-in-the-loop optimization procedure that guarantees physical stability and non-intersection, while improving semantic consistency with the input prompt. Experiments demonstrate that PAT3D substantially outperforms prior approaches in physical plausibility, semantic consistency, and visual quality. Beyond high-quality generation, PAT3D uniquely enables simulation-ready 3D scenes for downstream tasks such as scene editing and robotic manipulation. Code and data will be released upon acceptance.
>
---
#### [new 126] TPCNet: Triple physical constraints for Low-light Image Enhancement
- **分类: cs.CV; physics.optics**

- **简介: 该论文针对低光照图像增强任务，解决现有Retinex模型忽略镜面反射、物理约束局限在图像空间的问题。提出基于Kubelka-Munk理论的三重物理约束（TPCs），将约束引入特征空间，构建TPCNet，有效提升图像质量且不增加参数，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22052v1](https://arxiv.org/pdf/2511.22052v1)**

> **作者:** Jing-Yi Shi; Ming-Fei Li; Ling-An Wu
>
> **摘要:** Low-light image enhancement is an essential computer vision task to improve image contrast and to decrease the effects of color bias and noise. Many existing interpretable deep-learning algorithms exploit the Retinex theory as the basis of model design. However, previous Retinex-based algorithms, that consider reflected objects as ideal Lambertian ignore specular reflection in the modeling process and construct the physical constraints in image space, limiting generalization of the model. To address this issue, we preserve the specular reflection coefficient and reformulate the original physical constraints in the imaging process based on the Kubelka-Munk theory, thereby constructing constraint relationship between illumination, reflection, and detection, the so-called triple physical constraints (TPCs)theory. Based on this theory, the physical constraints are constructed in the feature space of the model to obtain the TPC network (TPCNet). Comprehensive quantitative and qualitative benchmark and ablation experiments confirm that these constraints effectively improve the performance metrics and visual quality without introducing new parameters, and demonstrate that our TPCNet outperforms other state-of-the-art methods on 10 datasets.
>
---
#### [new 127] MammoRGB: Dual-View Mammogram Synthesis Using Denoising Diffusion Probabilistic Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决双视图乳腺钼靶图像合成问题。通过微调三通道去噪扩散模型，利用不同通道编码方式生成配对的CC和MLO视图，提升图像真实性和跨视图一致性，验证了其在数据增强中的潜力。**

- **链接: [https://arxiv.org/pdf/2511.22759v1](https://arxiv.org/pdf/2511.22759v1)**

> **作者:** Jorge Alberto Garza-Abdala; Gerardo A. Fumagal-González; Daly Avendano; Servando Cardona; Sadam Hussain; Eduardo de Avila-Armenta; Jasiel H. Toscano-Martínez; Diana S. M. Rosales Gurmendi; Alma A. Pedro-Pérez; Jose Gerardo Tamez-Pena
>
> **摘要:** Purpose: This study aims to develop and evaluate a three channel denoising diffusion probabilistic model (DDPM) for synthesizing single breast dual view mammograms and to assess the impact of channel representations on image fidelity and cross view consistency. Materials and Methods: A pretrained three channel DDPM, sourced from Hugging Face, was fine tuned on a private dataset of 11020 screening mammograms to generate paired craniocaudal (CC) and mediolateral oblique (MLO) views. Three third channel encodings of the CC and MLO views were evaluated: sum, absolute difference, and zero channel. Each model produced 500 synthetic image pairs. Quantitative assessment involved breast mask segmentation using Intersection over Union (IoU) and Dice Similarity Coefficient (DSC), with distributional comparisons against 2500 real pairs using Earth Movers Distance (EMD) and Kolmogorov Smirnov (KS) tests. Qualitative evaluation included a visual Turing test by a non expert radiologist to assess cross view consistency and artifacts. Results: Synthetic mammograms showed IoU and DSC distributions comparable to real images, with EMD and KS values (0.020 and 0.077 respectively). Models using sum or absolute difference encodings outperformed others in IoU and DSC (p < 0.001), though distributions remained broadly similar. Generated CC and MLO views maintained cross view consistency, with 6 to 8 percent of synthetic images exhibiting artifacts consistent with those in the training data. Conclusion: Three channel DDPMs can generate realistic and anatomically consistent dual view mammograms with promising applications in dataset augmentation.
>
---
#### [new 128] ARPGNet: Appearance- and Relation-aware Parallel Graph Attention Fusion Network for Facial Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对面部表情识别任务，解决传统方法忽视面部区域间关系的问题。提出ARPGNet，通过构建面部区域关系图并结合图注意力机制，与卷积网络提取的外观特征并行融合，协同学习空间-时序表征，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2511.22188v1](https://arxiv.org/pdf/2511.22188v1)**

> **作者:** Yan Li; Yong Zhao; Xiaohan Xia; Dongmei Jiang
>
> **备注:** Accepted by IEEE Transactions on Affective Computing. Submitted in August 2023; Accepted in October 2025
>
> **摘要:** The key to facial expression recognition is to learn discriminative spatial-temporal representations that embed facial expression dynamics. Previous studies predominantly rely on pre-trained Convolutional Neural Networks (CNNs) to learn facial appearance representations, overlooking the relationships between facial regions. To address this issue, this paper presents an Appearance- and Relation-aware Parallel Graph attention fusion Network (ARPGNet) to learn mutually enhanced spatial-temporal representations of appearance and relation information. Specifically, we construct a facial region relation graph and leverage the graph attention mechanism to model the relationships between facial regions. The resulting relational representation sequences, along with CNN-based appearance representation sequences, are then fed into a parallel graph attention fusion module for mutual interaction and enhancement. This module simultaneously explores the complementarity between different representation sequences and the temporal dynamics within each sequence. Experimental results on three facial expression recognition datasets demonstrate that the proposed ARPGNet outperforms or is comparable to state-of-the-art methods.
>
---
#### [new 129] GLOW: Global Illumination-Aware Inverse Rendering of Indoor Scenes Captured with Dynamic Co-Located Light & Camera
- **分类: cs.CV**

- **简介: 该论文针对室内场景逆渲染中反射率与光照混淆问题，提出GLOW框架。通过神经隐式表面与动态辐射缓存，联合优化几何与材质，有效处理共位光-相机设置下的全局光照、动态阴影及镜面伪影，显著提升反射率估计精度。**

- **链接: [https://arxiv.org/pdf/2511.22857v1](https://arxiv.org/pdf/2511.22857v1)**

> **作者:** Jiaye Wu; Saeed Hadadan; Geng Lin; Peihan Tu; Matthias Zwicker; David Jacobs; Roni Sengupta
>
> **摘要:** Inverse rendering of indoor scenes remains challenging due to the ambiguity between reflectance and lighting, exacerbated by inter-reflections among multiple objects. While natural illumination-based methods struggle to resolve this ambiguity, co-located light-camera setups offer better disentanglement as lighting can be easily calibrated via Structure-from-Motion. However, such setups introduce additional complexities like strong inter-reflections, dynamic shadows, near-field lighting, and moving specular highlights, which existing approaches fail to handle. We present GLOW, a Global Illumination-aware Inverse Rendering framework designed to address these challenges. GLOW integrates a neural implicit surface representation with a neural radiance cache to approximate global illumination, jointly optimizing geometry and reflectance through carefully designed regularization and initialization. We then introduce a dynamic radiance cache that adapts to sharp lighting discontinuities from near-field motion, and a surface-angle-weighted radiometric loss to suppress specular artifacts common in flashlight captures. Experiments show that GLOW substantially outperforms prior methods in material reflectance estimation under both natural and co-located illumination.
>
---
#### [new 130] Stacked Ensemble of Fine-Tuned CNNs for Knee Osteoarthritis Severity Grading
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对膝骨关节炎（KOA）严重程度评估中人工阅片主观性强、效率低的问题，提出基于微调CNN的堆叠集成模型。通过多模型融合实现二分类（有无KOA）与五级多分类（KL分级），显著提升诊断准确率，验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22143v1](https://arxiv.org/pdf/2511.22143v1)**

> **作者:** Adarsh Gupta; Japleen Kaur; Tanvi Doshi; Teena Sharma; Nishchal K. Verma; Shantaram Vasikarla
>
> **备注:** Accepted and Presented at IEEE UEMCON, IBM T.J. Watson Research Center, New York, USA, 2024
>
> **摘要:** Knee Osteoarthritis (KOA) is a musculoskeletal condition that can cause significant limitations and impairments in daily activities, especially among older individuals. To evaluate the severity of KOA, typically, X-ray images of the affected knee are analyzed, and a grade is assigned based on the Kellgren-Lawrence (KL) grading system, which classifies KOA severity into five levels, ranging from 0 to 4. This approach requires a high level of expertise and time and is susceptible to subjective interpretation, thereby introducing potential diagnostic inaccuracies. To address this problem a stacked ensemble model of fine-tuned Convolutional Neural Networks (CNNs) was developed for two classification tasks: a binary classifier for detecting the presence of KOA, and a multiclass classifier for precise grading across the KL spectrum. The proposed stacked ensemble model consists of a diverse set of pre-trained architectures, including MobileNetV2, You Only Look Once (YOLOv8), and DenseNet201 as base learners and Categorical Boosting (CatBoost) as the meta-learner. This proposed model had a balanced test accuracy of 73% in multiclass classification and 87.5% in binary classification, which is higher than previous works in extant literature.
>
---
#### [new 131] PathReasoning: A multimodal reasoning agent for query-based ROI navigation on whole-slide images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PathReasoning，一种用于全切片图像（WSI）中基于查询的感兴趣区域（ROI）导航的多模态推理代理。针对WSI过大导致的导航困难问题，通过迭代推理与自省，引导模型高效定位诊断相关区域，提升肿瘤微环境分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.21902v1](https://arxiv.org/pdf/2511.21902v1)**

> **作者:** Kunpeng Zhang; Hanwen Xu; Sheng Wang
>
> **摘要:** Deciphering tumor microenvironment from Whole Slide Images (WSIs) is intriguing as it is key to cancer diagnosis, prognosis and treatment response. While these gigapixel images on one hand offer a comprehensive portrait of cancer, on the other hand, the extremely large size, as much as more than 10 billion pixels, make it challenging and time-consuming to navigate to corresponding regions to support diverse clinical inspection. Inspired by pathologists who conducted navigation on WSIs with a combination of sampling, reasoning and self-reflection, we proposed "PathReasoning", a multi-modal reasoning agent that iteratively navigates across WSIs through multiple rounds of reasoning and refinements. Specifically, starting with randomly sampled candidate regions, PathReasoning reviews current selections with self-reflection, reasoning over the correspondence between visual observations and clinical questions, and concludes by proposing new regions to explore. Across rounds, PathReasoning builds a reasoning chain that gradually directs attention to diagnostically relevant areas. PathReasoning turns each whole slide into a sequence of question-guided views, allowing the model to efficiently find informative ROIs within a fixed number of steps, without the need for dense pixel-level annotations. PathReasoning can substantially outperform strong ROI-selection approaches by 6.7% and 3.1% of AUROC on subtyping and longitudinal analysis tasks. The high-quality ROIs further support accurate report generation on breast cancer, significantly outperforming the standard GPT-4o by 10% in accuracy. PathReasoning prioritizes question-specific regions and constructs interpretable reasoning chains, supporting efficient slide review, consistent diagnostic interpretations, comprehensive reporting, and evidence traceability in digital pathology.
>
---
#### [new 132] REVEAL: Reasoning-enhanced Forensic Evidence Analysis for Explainable AI-generated Image Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对AI生成图像难以辨别问题，提出REVEAL框架，通过多专家模型构建可验证的证据链，结合强化学习实现精准、可解释的检测。解决了现有方法缺乏因果解释与泛化能力差的问题，推动可解释图像取证发展。**

- **链接: [https://arxiv.org/pdf/2511.23158v1](https://arxiv.org/pdf/2511.23158v1)**

> **作者:** Huangsen Cao; Qin Mei; Zhiheng Li; Yuxi Li; Ying Zhang; Chen Li; Zhimeng Zhang; Xin Ding; Yongwei Wang; Jing Lyu; Fei Wu
>
> **摘要:** With the rapid advancement of generative models, visually realistic AI-generated images have become increasingly difficult to distinguish from authentic ones, posing severe threats to social trust and information integrity. Consequently, there is an urgent need for efficient and truly explainable image forensic methods. Recent detection paradigms have shifted towards explainable forensics. However, state-of-the-art approaches primarily rely on post-hoc rationalizations or visual discrimination, lacking a verifiable chain of evidence. This reliance on surface-level pattern matching limits the generation of causally grounded explanations and often results in poor generalization. To bridge this critical gap, we introduce \textbf{REVEAL-Bench}, the first reasoning-enhanced multimodal benchmark for AI-generated image detection that is explicitly structured around a chain-of-evidence derived from multiple lightweight expert models, then records step-by-step reasoning traces and evidential justifications. Building upon this dataset, we propose \textbf{REVEAL} (\underline{R}easoning-\underline{e}nhanced Forensic E\underline{v}id\underline{e}nce \underline{A}na\underline{l}ysis), an effective and explainable forensic framework that integrates detection with a novel expert-grounded reinforcement learning. Our reward mechanism is specially tailored to jointly optimize detection accuracy, explanation fidelity, and logical coherence grounded in explicit forensic evidence, enabling REVEAL to produce fine-grained, interpretable, and verifiable reasoning chains alongside its detection outcomes. Extensive experimental results demonstrate that REVEAL significantly enhances detection accuracy, explanation fidelity, and robust cross-model generalization, benchmarking a new state of the art for explainable image forensics.
>
---
#### [new 133] ABounD: Adversarial Boundary-Driven Few-Shot Learning for Multi-Class Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对少样本多类工业异常检测任务，解决数据稀缺导致正常与异常边界模糊的问题。提出ABounD框架，通过动态概念融合生成类别自适应提示，并利用对抗边界构造生成边界特征，以精确塑造决策边界，实现高精度异常检测。**

- **链接: [https://arxiv.org/pdf/2511.22436v1](https://arxiv.org/pdf/2511.22436v1)**

> **作者:** Runzhi Deng; Yundi Hu; Xinshuang Zhang; Zhao Wang; Xixi Liu; Wang-Zhou Dai; Caifeng Shan; Fang Zhao
>
> **摘要:** Few-shot multi-class industrial anomaly detection remains a challenging task. Vision-language models need to be both category-adaptive and sharply discriminative, yet data scarcity often blurs the boundary between normal and abnormal states. This ambiguity leads to missed subtle defects and the rejection of atypical normal samples. We propose ABounD, an Adversarial Boundary-Driven few-shot learning for multi-class anomaly detection, which is a unified learning framework that integrates semantic concept learning with decision boundary shaping. The Dynamic Concept Fusion (DCF) module produces class-adaptive prompts by fusing generalizable priors with class-specific cues, conditioned on image features. Meanwhile, Adversarial Boundary Forging (ABF) sculpts a more precise decision margin by generating boundary-level fence features via PGD-style perturbations. Training is conducted in a single stage under a Concept-Boundary Loss, where ABF provides the main supervisory signal and semantic-spatial regularizers stabilize the optimization. This synergy yields a decision boundary that closely follows normal data while preserving flexibility and robust semantic alignment. Experiments on MVTec-AD and VisA datasets demonstrate state-of-the-art performance in the task of few-shot multi-class anomaly detection.
>
---
#### [new 134] Layover or Direct Flight: Rethinking Audio-Guided Image Segmentation
- **分类: cs.CV**

- **简介: 该论文研究直接音频引导图像分割任务，旨在无需文本中间表示，实现语音到视觉的直接对齐。针对现有方法依赖语音转写带来的效率与鲁棒性问题，提出新数据集并验证直接音频-视觉对齐的可行性，结果表明其在应对语言差异时更具优势，推动更高效多模态系统发展。**

- **链接: [https://arxiv.org/pdf/2511.22025v1](https://arxiv.org/pdf/2511.22025v1)**

> **作者:** Joel Alberto Santos; Zongwei Wu; Xavier Alameda-Pineda; Radu Timofte
>
> **摘要:** Understanding human instructions is essential for enabling smooth human-robot interaction. In this work, we focus on object grounding, i.e., localizing an object of interest in a visual scene (e.g., an image) based on verbal human instructions. Despite recent progress, a dominant research trend relies on using text as an intermediate representation. These approaches typically transcribe speech to text, extract relevant object keywords, and perform grounding using models pretrained on large text-vision datasets. However, we question both the efficiency and robustness of such transcription-based pipelines. Specifically, we ask: Can we achieve direct audio-visual alignment without relying on text? To explore this possibility, we simplify the task by focusing on grounding from single-word spoken instructions. We introduce a new audio-based grounding dataset that covers a wide variety of objects and diverse human accents. We then adapt and benchmark several models from the closely audio-visual field. Our results demonstrate that direct grounding from audio is not only feasible but, in some cases, even outperforms transcription-based methods, especially in terms of robustness to linguistic variability. Our findings encourage a renewed interest in direct audio grounding and pave the way for more robust and efficient multimodal understanding systems.
>
---
#### [new 135] Cascaded Robust Rectification for Arbitrary Document Images
- **分类: cs.CV**

- **简介: 该论文针对真实场景中文档图像的任意形变问题，提出一种级联式鲁棒校正框架。通过分阶段（仿射、几何变形、内容感知）逐步纠正透视、纸张弯曲及细粒度内容失真，引入新评估指标以更准确衡量校正效果，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2511.23150v1](https://arxiv.org/pdf/2511.23150v1)**

> **作者:** Chaoyun Wang; Quanxin Huang; I-Chao Shen; Takeo Igarashi; Nanning Zheng; Caigui Jiang
>
> **摘要:** Document rectification in real-world scenarios poses significant challenges due to extreme variations in camera perspectives and physical distortions. Driven by the insight that complex transformations can be decomposed and resolved progressively, we introduce a novel multi-stage framework that progressively reverses distinct distortion types in a coarse-to-fine manner. Specifically, our framework first performs a global affine transformation to correct perspective distortions arising from the camera's viewpoint, then rectifies geometric deformations resulting from physical paper curling and folding, and finally employs a content-aware iterative process to eliminate fine-grained content distortions. To address limitations in existing evaluation protocols, we also propose two enhanced metrics: layout-aligned OCR metrics (AED/ACER) for a stable assessment that decouples geometric rectification quality from the layout analysis errors of OCR engines, and masked AD/AAD (AD-M/AAD-M) tailored for accurately evaluating geometric distortions in documents with incomplete boundaries. Extensive experiments show that our method establishes new state-of-the-art performance on multiple challenging benchmarks, yielding a substantial reduction of 14.1\%--34.7\% in the AAD metric and demonstrating superior efficacy in real-world applications. The code will be publicly available at https://github.com/chaoyunwang/ArbDR.
>
---
#### [new 136] Guiding Visual Autoregressive Models through Spectrum Weakening
- **分类: cs.CV**

- **简介: 该论文针对视觉自回归模型的生成质量与条件对齐问题，提出无须重训练的谱弱化框架。通过在频域选择性保留部分谱信息，实现可控信息减弱，并引入归一化策略保证稳定性。实验表明，该方法有效提升无条件生成质量，同时保持强条件对齐。**

- **链接: [https://arxiv.org/pdf/2511.22991v1](https://arxiv.org/pdf/2511.22991v1)**

> **作者:** Chaoyang Wang; Tianmeng Yang; Jingdong Wang; Yunhai Tong
>
> **摘要:** Classifier-free guidance (CFG) has become a widely adopted and practical approach for enhancing generation quality and improving condition alignment. Recent studies have explored guidance mechanisms for unconditional generation, yet these approaches remain fundamentally tied to assumptions specific to diffusion models. In this work, we propose a spectrum-weakening framework for visual autoregressive (AR) models. This method works without the need for re-training, specific conditions, or any architectural modifications. It achieves this by constructing a controllable weak model in the spectral domain. We theoretically show that invertible spectral transformations preserve information, while selectively retaining only a subset of spectrum introduces controlled information reduction. Based on this insight, we perform spectrum selection along the channel dimension of internal representations, which avoids the structural constraints imposed by diffusion models. We further introduce two spectrum renormalization strategies that ensures numerical stability during the weakening process. Extensive experiments were conducted on both discrete and continuous AR models, with text or class conditioning. The results demonstrate that our method enables high-quality unconditional generation while maintaining strong prompt alignment for conditional generation.
>
---
#### [new 137] Structure is Supervision: Multiview Masked Autoencoders for Radiology
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医学影像分析中缺乏标注数据的问题，提出多视角掩码自编码器（MVMAE）框架，利用放射科检查的多视图结构进行自监督学习，以获取鲁棒的疾病相关表示。进一步引入文本辅助的MVMAE-V2T，在低标签场景下提升模型性能，推动可扩展的临床基础模型发展。**

- **链接: [https://arxiv.org/pdf/2511.22294v1](https://arxiv.org/pdf/2511.22294v1)**

> **作者:** Sonia Laguna; Andrea Agostini; Alain Ryser; Samuel Ruiperez-Campillo; Irene Cannistraci; Moritz Vandenhirtz; Stephan Mandt; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building robust medical machine learning systems requires pretraining strategies that exploit the intrinsic structure present in clinical data. We introduce Multiview Masked Autoencoder (MVMAE), a self-supervised framework that leverages the natural multi-view organization of radiology studies to learn view-invariant and disease-relevant representations. MVMAE combines masked image reconstruction with cross-view alignment, transforming clinical redundancy across projections into a powerful self-supervisory signal. We further extend this approach with MVMAE-V2T, which incorporates radiology reports as an auxiliary text-based learning signal to enhance semantic grounding while preserving fully vision-based inference. Evaluated on a downstream disease classification task on three large-scale public datasets, MIMIC-CXR, CheXpert, and PadChest, MVMAE consistently outperforms supervised and vision-language baselines. Furthermore, MVMAE-V2T provides additional gains, particularly in low-label regimes where structured textual supervision is most beneficial. Together, these results establish the importance of structural and textual supervision as complementary paths toward scalable, clinically grounded medical foundation models.
>
---
#### [new 138] A Hierarchical Computer Vision Pipeline for Physiological Data Extraction from Bedside Monitors
- **分类: cs.CV**

- **简介: 该论文针对低资源医疗环境中孤立的无网络监护仪无法接入电子病历系统的问题，提出一种基于计算机视觉的分层数据提取管道。通过YOLOv11与PaddleOCR结合，并引入几何校正模块，实现对屏幕生理参数的高精度自动识别与数字化，有效解决了数据孤岛问题，为医疗信息化提供低成本、可扩展的解决方案。**

- **链接: [https://arxiv.org/pdf/2511.23355v1](https://arxiv.org/pdf/2511.23355v1)**

> **作者:** Vinh Chau; Khoa Le Dinh Van; Hon Huynh Ngoc; Binh Nguyen Thien; Hao Nguyen Thien; Vy Nguyen Quang; Phuc Vo Hong; Yen Lam Minh; Kieu Pham Tieu; Trinh Nguyen Thi Diem; Louise Thwaites; Hai Ho Bich
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** In many low-resource healthcare settings, bedside monitors remain standalone legacy devices without network connectivity, creating a persistent interoperability gap that prevents seamless integration of physiological data into electronic health record (EHR) systems. To address this challenge without requiring costly hardware replacement, we present a computer vision-based pipeline for the automated capture and digitisation of vital sign data directly from bedside monitor screens. Our method employs a hierarchical detection framework combining YOLOv11 for accurate monitor and region of interest (ROI) localisation with PaddleOCR for robust text extraction. To enhance reliability across variable camera angles and lighting conditions, a geometric rectification module standardizes the screen perspective before character recognition. We evaluated the system on a dataset of 6,498 images collected from open-source corpora and real-world intensive care units in Vietnam. The model achieved a mean Average Precision (mAP@50-95) of 99.5% for monitor detection and 91.5% for vital sign ROI localisation. The end-to-end extraction accuracy exceeded 98.9% for core physiological parameters, including heart rate, oxygen saturation SpO2, and arterial blood pressure. These results demonstrate that a lightweight, camera-based approach can reliably transform unstructured information from screen captures into structured digital data, providing a practical and scalable pathway to improve information accessibility and clinical documentation in low-resource settings.
>
---
#### [new 139] Unlocking Multilingual Reasoning Capability of LLMs and LVLMs through Representation Engineering
- **分类: cs.CV**

- **简介: 该论文针对大模型在低资源语言上推理能力弱的问题，提出无需训练的推理时方法MRRE。通过注入预计算向量，将非英语推理表示对齐至英语空间，并锚定目标语言输出分布，提升多语言推理性能与语言一致性，显著改善泰语、斯瓦希里语等低资源语言表现。**

- **链接: [https://arxiv.org/pdf/2511.23231v1](https://arxiv.org/pdf/2511.23231v1)**

> **作者:** Qiming Li; Xiaocheng Feng; Yixuan Ma; Zekai Ye; Ruihan Chen; Xiachong Feng; Bing Qin
>
> **摘要:** Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) demonstrate strong reasoning capabilities, yet their performance in English significantly outperforms that in low-resource languages, raising fairness concerns in multilingual applications. Existing approaches either rely on costly multilingual training or employ prompting with external translation tools, both of which are resource-intensive and sensitive to translation quality. To address these limitations, we propose a training-free inference-time method to enhance Multilingual Reasoning capabilities via Representation Engineering (MRRE) without using any additional training data or tools. MRRE sequentially injects two precomputed vectors at specific layers during inference processing: cross-lingual reasoning enhancement vectors, which steer non-English reasoning representations toward English space to unlock multilingual reasoning, and target-language output anchoring vectors, which restore the distribution of the target language to preserve input-output language consistency. Comprehensive experiments across six advanced LLMs and LVLMs on four reasoning benchmarks demonstrate that MRRE consistently enhances non-English reasoning by an average gain of 5.48% and up to 7.54% in low-resource languages (Thai and Swahili), while improving input-output language consistency by 3.78%.
>
---
#### [new 140] Semantic Anchoring for Robust Personalization in Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文研究文本到图像扩散模型的个性化生成任务。针对少量参考图下模型易过拟合或忽视新属性的问题，提出语义锚定机制，通过将新概念与预训练中的常见概念对齐，实现稳定适应，兼顾主体保真度与文本-图像一致性。**

- **链接: [https://arxiv.org/pdf/2511.22245v1](https://arxiv.org/pdf/2511.22245v1)**

> **作者:** Seoyun Yang; Gihoon Kim; Taesup Kim
>
> **摘要:** Text-to-image diffusion models have achieved remarkable progress in generating diverse and realistic images from textual descriptions. However, they still struggle with personalization, which requires adapting a pretrained model to depict user-specific subjects from only a few reference images. The key challenge lies in learning a new visual concept from a limited number of reference images while preserving the pretrained semantic prior that maintains text-image alignment. When the model focuses on subject fidelity, it tends to overfit the limited reference images and fails to leverage the pretrained distribution. Conversely, emphasizing prior preservation maintains semantic consistency but prevents the model from learning new personalized attributes. Building on these observations, we propose the personalization process through a semantic anchoring that guides adaptation by grounding new concepts in their corresponding distributions. We therefore reformulate personalization as the process of learning a rare concept guided by its frequent counterpart through semantic anchoring. This anchoring encourages the model to adapt new concepts in a stable and controlled manner, expanding the pretrained distribution toward personalized regions while preserving its semantic structure. As a result, the proposed method achieves stable adaptation and consistent improvements in both subject fidelity and text-image alignment compared to baseline methods. Extensive experiments and ablation studies further demonstrate the robustness and effectiveness of the proposed anchoring strategy.
>
---
#### [new 141] Breaking the Visual Shortcuts in Multimodal Knowledge-Based Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对多模态知识驱动视觉问答（MKB-VQA）中的“视觉捷径”问题，提出RETINA基准与MIMIR模型。通过引入关联实体图像消除视觉捷径，揭示现有模型依赖图像线索的缺陷，并设计多图增强检索器提升性能，推动任务向更鲁棒的方向发展。**

- **链接: [https://arxiv.org/pdf/2511.22843v1](https://arxiv.org/pdf/2511.22843v1)**

> **作者:** Dosung Lee; Sangwon Jung; Boyoung Kim; Minyoung Kim; Sungyeon Kim; Junyoung Sung; Paul Hongsuck Seo
>
> **摘要:** Existing Multimodal Knowledge-Based Visual Question Answering (MKB-VQA) benchmarks suffer from "visual shortcuts", as the query image typically matches the primary subject entity of the target document. We demonstrate that models can exploit these shortcuts, achieving comparable results using visual cues alone. To address this, we introduce Relational Entity Text-Image kNowledge Augmented (RETINA) benchmark, automatically constructed using an LLM-driven pipeline, consisting of 120k training and 2k human-curated test set. RETINA contains queries referencing secondary subjects (i.e. related entities) and pairs them with images of these related entities, removing the visual shortcut. When evaluated on RETINA existing models show significantly degraded performance, confirming their reliance on the shortcut. Furthermore, we propose Multi-Image MultImodal Retriever (MIMIR), which enriches document embeddings by augmenting images of multiple related entities, effectively handling RETINA, unlike prior work that uses only a single image per document. Our experiments validate the limitations of existing benchmarks and demonstrate the effectiveness of RETINA and MIMIR. Our project is available at: Project Page.
>
---
#### [new 142] DNA-Prior: Unsupervised Denoise Anything via Dual-Domain Prior
- **分类: cs.CV**

- **简介: 该论文针对医学图像去噪任务，解决现有方法依赖标注数据与模态特定训练的问题。提出DNA-Prior框架，通过隐式网络结构与显式频域-空间先验的双重约束，实现无需训练的通用去噪，有效抑制噪声并保留解剖结构。**

- **链接: [https://arxiv.org/pdf/2511.23124v1](https://arxiv.org/pdf/2511.23124v1)**

> **作者:** Yanqi Cheng; Chun-Wun Cheng; Jim Denholm; Thiago Lima; Javier A. Montoya-Zegarra; Richard Goodwin; Carola-Bibiane Schönlieb; Angelica I Aviles-Rivero
>
> **摘要:** Medical imaging pipelines critically rely on robust denoising to stabilise downstream tasks such as segmentation and reconstruction. However, many existing denoisers depend on large annotated datasets or supervised learning, which restricts their usability in clinical environments with heterogeneous modalities and limited ground-truth data. To address this limitation, we introduce DNA-Prior, a universal unsupervised denoising framework that reconstructs clean images directly from corrupted observations through a mathematically principled hybrid prior. DNA-Prior integrates (i) an implicit architectural prior, enforced through a deep network parameterisation, with (ii) an explicit spectral-spatial prior composed of a frequency-domain fidelity term and a spatial regularisation functional. This dual-domain formulation yields a well-structured optimisation problem that jointly preserves global frequency characteristics and local anatomical structure, without requiring any external training data or modality-specific tuning. Experiments across multiple modalities show that DNA achieves consistent noise suppression and structural preservation under diverse noise conditions.
>
---
#### [new 143] Intra-Class Probabilistic Embeddings for Uncertainty Estimation in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉-语言模型（VLM）在误分类时仍给出高置信度的问题，提出一种无需训练的后处理不确定性估计方法。通过类内特征一致性建模，构建类特定的概率嵌入，实现高效错误检测，适用于少样本场景且对分布偏移鲁棒，在多个数据集上表现优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22019v1](https://arxiv.org/pdf/2511.22019v1)**

> **作者:** Zhenxiang Lin; Maryam Haghighat; Will Browne; Dimity Miller
>
> **摘要:** Vision-language models (VLMs), such as CLIP, have gained popularity for their strong open vocabulary classification performance, but they are prone to assigning high confidence scores to misclassifications, limiting their reliability in safety-critical applications. We introduce a training-free, post-hoc uncertainty estimation method for contrastive VLMs that can be used to detect erroneous predictions. The key to our approach is to measure visual feature consistency within a class, using feature projection combined with multivariate Gaussians to create class-specific probabilistic embeddings. Our method is VLM-agnostic, requires no fine-tuning, demonstrates robustness to distribution shift, and works effectively with as few as 10 training images per class. Extensive experiments on ImageNet, Flowers102, Food101, EuroSAT and DTD show state-of-the-art error detection performance, significantly outperforming both deterministic and probabilistic VLM baselines. Code is available at https://github.com/zhenxianglin/ICPE.
>
---
#### [new 144] RoadSceneBench: A Lightweight Benchmark for Mid-Level Road Scene Understanding
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中缺乏对道路结构与上下文关系的推理能力问题，提出RoadSceneBench基准，聚焦中层道路语义理解。通过构建轻量级数据集并设计HRRP-T训练框架，提升视觉语言模型在空间一致性与时间连贯性上的结构化推理能力，推动从感知到规划的中间环节发展。**

- **链接: [https://arxiv.org/pdf/2511.22466v1](https://arxiv.org/pdf/2511.22466v1)**

> **作者:** Xiyan Liu; Han Wang; Yuhu Wang; Junjie Cai; Zhe Cao; Jianzhong Yang; Zhen Lu
>
> **摘要:** Understanding mid-level road semantics, which capture the structural and contextual cues that link low-level perception to high-level planning, is essential for reliable autonomous driving and digital map construction. However, existing benchmarks primarily target perception tasks such as detection or segmentation, overlooking the reasoning capabilities required to infer road topology and dynamic scene structure. To address this gap, we present RoadSceneBench, a lightweight yet information-rich benchmark designed to evaluate and advance visual reasoning in complex road environments. Unlike large-scale perception datasets, RoadSceneBench emphasizes relational understanding and structural consistency, encouraging models to capture the underlying logic of real-world road scenes. Furthermore, to enhance reasoning reliability, we propose Hierarchical Relational Reward Propagation with Temporal Consistency (HRRP-T), a training framework for Vision-Language Models (VLMs) in which reward signals adaptively promote spatial coherence and semantic alignment throughout the reasoning process. This paradigm enables models to move beyond static recognition toward geometry-aware and temporally consistent reasoning. Extensive experiments demonstrate that our method achieves state-of-the-art performance across diverse road configurations. RoadSceneBench thus provides a compact yet powerful foundation for studying mid-level road semantics and fostering structure-aware autonomous perception. Our dataset is available at https://github.com/XiyanLiu/RoadSceneBench.
>
---
#### [new 145] GoPrune: Accelerated Structured Pruning with $\ell_{2,p}$-Norm Optimization
- **分类: cs.CV**

- **简介: 该论文针对深度神经网络在边缘设备部署时存储与计算成本高的问题，提出一种基于ℓ₂,ₚ-范数的加速结构化剪枝方法GoPrune。通过扩展p∈[0,1)并设计高效近似算法，实现高压缩率与快速优化，显著提升剪枝效率与模型性能。**

- **链接: [https://arxiv.org/pdf/2511.22120v1](https://arxiv.org/pdf/2511.22120v1)**

> **作者:** Li Xu; Xianchao Xiu
>
> **摘要:** Convolutional neural networks (CNNs) suffer from rapidly increasing storage and computational costs as their depth grows, which severely hinders their deployment on resource-constrained edge devices. Pruning is a practical approach for network compression, among which structured pruning is the most effective for inference acceleration. Although existing work has applied the $\ell_p$-norm to pruning, it only considers unstructured pruning with $p\in (0, 1)$ and has low computational efficiency. To overcome these limitations, we propose an accelerated structured pruning method called GoPrune. Our method employs the $\ell_{2,p}$-norm for sparse network learning, where the value of $p$ is extended to $[0, 1)$. Moreover, we develop an efficient optimization algorithm based on the proximal alternating minimization (PAM), and the resulting subproblems enjoy closed-form solutions, thus improving compression efficiency. Experiments on the CIFAR datasets using ResNet and VGG models demonstrate the superior performance of the proposed method in network pruning. Our code is available at https://github.com/xianchaoxiu/GoPrune.
>
---
#### [new 146] A Perceptually Inspired Variational Framework for Color Enhancement
- **分类: cs.CV**

- **简介: 该论文针对图像颜色增强任务，提出一种受人类色觉感知启发的变分框架。为解决现有模型对显著图像特征（如对比度）难以量化的问题，构建满足感知一致性的能量泛函，设计三种核心函数，并通过梯度下降求解。提出高效算法将计算复杂度从O(N²)降至O(N log N)，显著提升效率。**

- **链接: [https://arxiv.org/pdf/2511.23329v1](https://arxiv.org/pdf/2511.23329v1)**

> **作者:** Rodrigo Palma-Amestoy; Edoardo Provenzi; Marcelo Bertalmío; Vicent Caselles
>
> **摘要:** Basic phenomenology of human color vision has been widely taken as an inspiration to devise explicit color correction algorithms. The behavior of these models in terms of significative image features (such as contrast and dispersion) can be difficult to characterize. To cope with this, we propose to use a variational formulation of color contrast enhancement that is inspired by the basic phenomenology of color perception. In particular, we devise a set of basic requirements to be fulfilled by an energy to be considered as `perceptually inspired', showing that there is an explicit class of functionals satisfying all of them. We single out three explicit functionals that we consider of basic interest, showing similarities and differences with existing models. The minima of such functionals is computed using a gradient descent approach. We also present a general methodology to reduce the computational cost of the algorithms under analysis from ${\cal O}(N^2)$ to ${\cal O}(N\log N)$, being $N$ the number of input pixels.
>
---
#### [new 147] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对低分辨率3D高斯点云重建中细节缺失问题，提出IE-SRGS框架。通过融合外部2D超分先验与内部3D高斯特征，利用多尺度模型生成一致且自适应的高分辨率图像与深度图，结合掩码引导融合策略，实现高质量3D高斯点云超分辨率重建。**

- **链接: [https://arxiv.org/pdf/2511.22233v1](https://arxiv.org/pdf/2511.22233v1)**

> **作者:** Xiang Feng; Tieshi Zhong; Shuo Chang; Weiliu Wang; Chengkai Wang; Yifei Chen; Yuhe Wang; Zhenzhong Kuang; Xuefei Yin; Yanming Zhu
>
> **备注:** AAAI 2026
>
> **摘要:** Reconstructing high-resolution (HR) 3D Gaussian Splatting (3DGS) models from low-resolution (LR) inputs remains challenging due to the lack of fine-grained textures and geometry. Existing methods typically rely on pre-trained 2D super-resolution (2DSR) models to enhance textures, but suffer from 3D Gaussian ambiguity arising from cross-view inconsistencies and domain gaps inherent in 2DSR models. We propose IE-SRGS, a novel 3DGS SR paradigm that addresses this issue by jointly leveraging the complementary strengths of external 2DSR priors and internal 3DGS features. Specifically, we use 2DSR and depth estimation models to generate HR images and depth maps as external knowledge, and employ multi-scale 3DGS models to produce cross-view consistent, domain-adaptive counterparts as internal knowledge. A mask-guided fusion strategy is introduced to integrate these two sources and synergistically exploit their complementary strengths, effectively guiding the 3D Gaussian optimization toward high-fidelity reconstruction. Extensive experiments on both synthetic and real-world benchmarks show that IE-SRGS consistently outperforms state-of-the-art methods in both quantitative accuracy and visual fidelity.
>
---
#### [new 148] MRI-Based Brain Age Estimation with Supervised Contrastive Learning of Continuous Representation
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究基于MRI的脑年龄估计任务，旨在更准确地捕捉神经形态变化的连续性。针对传统深度回归模型特征表示不足的问题，提出使用监督对比学习与RNC损失，结合Grad-RAM可视化解释。实验表明，该方法在小数据集上表现优异，且揭示了脑年龄差与神经退行性疾病严重程度的相关性，具备潜在生物标志物价值。**

- **链接: [https://arxiv.org/pdf/2511.22102v1](https://arxiv.org/pdf/2511.22102v1)**

> **作者:** Simon Joseph Clément Crête; Marta Kersten-Oertel; Yiming Xiao
>
> **摘要:** MRI-based brain age estimation models aim to assess a subject's biological brain age based on information, such as neuroanatomical features. Various factors, including neurodegenerative diseases, can accelerate brain aging and measuring this phenomena could serve as a potential biomarker for clinical applications. While deep learning (DL)-based regression has recently attracted major attention, existing approaches often fail to capture the continuous nature of neuromorphological changes, potentially resulting in sub-optimal feature representation and results. To address this, we propose to use supervised contrastive learning with the recent Rank-N-Contrast (RNC) loss to estimate brain age based on widely used T1w structural MRI for the first time and leverage Grad-RAM to visually explain regression results. Experiments show that our proposed method achieves a mean absolute error (MAE) of 4.27 years and an $R^2$ of 0.93 with a limited dataset of training samples, significantly outperforming conventional deep regression with the same ResNet backbone while performing better or comparably with the state-of-the-art methods with significantly larger training data. Furthermore, Grad-RAM revealed more nuanced features related to age regression with the RNC loss than conventional deep regression. As an exploratory study, we employed the proposed method to estimate the gap between the biological and chronological brain ages in Alzheimer's Disease and Parkinson's disease patients, and revealed the correlation between the brain age gap and disease severity, demonstrating its potential as a biomarker in neurodegenerative disorders.
>
---
#### [new 149] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉-语言模型在少样本和域泛化任务中适应能力弱的问题，提出变分多模态提示学习（VaMP）。通过可变的、不确定性感知的提示生成机制，实现基于输入内容的个性化提示调优，提升模型对实例差异与任务结构的建模能力。**

- **链接: [https://arxiv.org/pdf/2511.22664v1](https://arxiv.org/pdf/2511.22664v1)**

> **作者:** Silin Cheng; Kai Han
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Vision-language models (VLMs), such as CLIP, have shown strong generalization under zero-shot settings, yet adapting them to downstream tasks with limited supervision remains a significant challenge. Existing multi-modal prompt learning methods typically rely on fixed, shared prompts and deterministic parameters, which limits their ability to capture instance-level variation or model uncertainty across diverse tasks and domains. To tackle this issue, we propose a novel Variational Multi-Modal Prompt Learning (VaMP) framework that enables sample-specific, uncertainty-aware prompt tuning in multi-modal representation learning. VaMP generates instance-conditioned prompts by sampling from a learned posterior distribution, allowing the model to personalize its behavior based on input content. To further enhance the integration of local and global semantics, we introduce a class-aware prior derived from the instance representation and class prototype. Building upon these, we formulate prompt tuning as variational inference over latent prompt representations and train the entire framework end-to-end through reparameterized sampling. Experiments on few-shot and domain generalization benchmarks show that VaMP achieves state-of-the-art performance, highlighting the benefits of modeling both uncertainty and task structure in our method. Project page: https://visual-ai.github.io/vamp
>
---
#### [new 150] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文针对单目3D重建模型在几何细节和鲁棒性上的不足，提出Fin3R方法。通过轻量级LoRA适配器，将强监督的单目教师模型知识蒸馏到图像编码器中，仅微调编码器以提升精度，保持推理效率。验证表明，该方法显著改善了边界清晰度与结构恢复能力。**

- **链接: [https://arxiv.org/pdf/2511.22429v1](https://arxiv.org/pdf/2511.22429v1)**

> **作者:** Weining Ren; Hongjun Wang; Xiao Tan; Kai Han
>
> **备注:** NeurIPS 2025
>
> **摘要:** We present Fin3R, a simple, effective, and general fine-tuning method for feed-forward 3D reconstruction models. The family of feed-forward reconstruction model regresses pointmap of all input images to a reference frame coordinate system, along with other auxiliary outputs, in a single forward pass. However, we find that current models struggle with fine geometry and robustness due to (\textit{i}) the scarcity of high-fidelity depth and pose supervision and (\textit{ii}) the inherent geometric misalignment from multi-view pointmap regression. Fin3R jointly tackles two issues with an extra lightweight fine-tuning step. We freeze the decoder, which handles view matching, and fine-tune only the image encoder-the component dedicated to feature extraction. The encoder is enriched with fine geometric details distilled from a strong monocular teacher model on large, unlabeled datasets, using a custom, lightweight LoRA adapter. We validate our method on a wide range of models, including DUSt3R, MASt3R, CUT3R, and VGGT. The fine-tuned models consistently deliver sharper boundaries, recover complex structures, and achieve higher geometric accuracy in both single- and multi-view settings, while adding only the tiny LoRA weights, which leave test-time memory and latency virtually unchanged. Project page: \href{http://visual-ai.github.io/fin3r}{https://visual-ai.github.io/fin3r}
>
---
#### [new 151] MANTA: Physics-Informed Generalized Underwater Object Tracking
- **分类: cs.CV**

- **简介: 该论文针对水下目标跟踪任务，解决因光衰减与散射导致的外观畸变问题。提出MANTA框架，结合物理模型与对比学习，增强特征鲁棒性；设计多阶段追踪流程，提升遮挡与漂移下的重识别能力，并引入新评估指标，实现更优性能与高效运行。**

- **链接: [https://arxiv.org/pdf/2511.23405v1](https://arxiv.org/pdf/2511.23405v1)**

> **作者:** Suhas Srinath; Hemang Jamadagni; Aditya Chadrasekar; Prathosh AP
>
> **备注:** Accepted to the IEEE/CVF WACV 2026
>
> **摘要:** Underwater object tracking is challenging due to wavelength dependent attenuation and scattering, which severely distort appearance across depths and water conditions. Existing trackers trained on terrestrial data fail to generalize to these physics-driven degradations. We present MANTA, a physics-informed framework integrating representation learning with tracking design for underwater scenarios. We propose a dual-positive contrastive learning strategy coupling temporal consistency with Beer-Lambert augmentations to yield features robust to both temporal and underwater distortions. We further introduce a multi-stage pipeline augmenting motion-based tracking with a physics-informed secondary association algorithm that integrates geometric consistency and appearance similarity for re-identification under occlusion and drift. To complement standard IoU metrics, we propose Center-Scale Consistency (CSC) and Geometric Alignment Score (GAS) to assess geometric fidelity. Experiments on four underwater benchmarks (WebUOT-1M, UOT32, UTB180, UWCOT220) show that MANTA achieves state-of-the-art performance, improving Success AUC by up to 6 percent, while ensuring stable long-term generalized underwater tracking and efficient runtime.
>
---
#### [new 152] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SimScale框架，解决自动驾驶中真实数据稀缺且多样性不足的问题。通过神经渲染与反应式环境生成大规模未见驾驶状态，并设计伪专家提供动作监督。结合真实与仿真数据进行联合训练，显著提升规划模型的鲁棒性与泛化能力，且性能随仿真数据增加而持续提升。**

- **链接: [https://arxiv.org/pdf/2511.23369v1](https://arxiv.org/pdf/2511.23369v1)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** Project page: https://opendrivelab.com/SimScale
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +6.8 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Our simulation data and code would be released.
>
---
#### [new 153] NeuMatC: A General Neural Framework for Fast Parametric Matrix Operation
- **分类: cs.CV**

- **简介: 该论文提出NeuMatC框架，针对参数化矩阵运算（如求逆、SVD）中因重复计算导致的效率低下问题，利用参数维度上的低秩性与连续性，通过无监督学习建立参数到运算结果的低秩连续映射，实现快速高效计算，显著提升速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2511.22934v1](https://arxiv.org/pdf/2511.22934v1)**

> **作者:** Chuan Wang; Xi-le Zhao; Zhilong Han; Liang Li; Deyu Meng; Michael K. Ng
>
> **摘要:** Matrix operations (e.g., inversion and singular value decomposition (SVD)) are fundamental in science and engineering. In many emerging real-world applications (such as wireless communication and signal processing), these operations must be performed repeatedly over matrices with parameters varying continuously. However, conventional methods tackle each matrix operation independently, underexploring the inherent low-rankness and continuity along the parameter dimension, resulting in significantly redundant computation. To address this challenge, we propose \textbf{\textit{Neural Matrix Computation Framework} (NeuMatC)}, which elegantly tackles general parametric matrix operation tasks by leveraging the underlying low-rankness and continuity along the parameter dimension. Specifically, NeuMatC unsupervisedly learns a low-rank and continuous mapping from parameters to their corresponding matrix operation results. Once trained, NeuMatC enables efficient computations at arbitrary parameters using only a few basic operations (e.g., matrix multiplications and nonlinear activations), significantly reducing redundant computations. Experimental results on both synthetic and real-world datasets demonstrate the promising performance of NeuMatC, exemplified by over $3\times$ speedup in parametric inversion and $10\times$ speedup in parametric SVD compared to the widely used NumPy baseline in wireless communication, while maintaining acceptable accuracy.
>
---
#### [new 154] InstanceV: Instance-Level Video Generation
- **分类: cs.CV**

- **简介: 该论文提出InstanceV，解决文本到视频生成中缺乏实例级精细控制与全局语义一致性的问题。通过实例感知掩码交叉注意力、时序自适应提示增强和空间感知无条件引导，实现精准实例定位与高质量生成，并构建InstanceBench基准进行综合评估，显著提升生成效果。**

- **链接: [https://arxiv.org/pdf/2511.23146v1](https://arxiv.org/pdf/2511.23146v1)**

> **作者:** Yuheng Chen; Teng Hu; Jiangning Zhang; Zhucun Xue; Ran Yi; Lizhuang Ma
>
> **摘要:** Recent advances in text-to-video diffusion models have enabled the generation of high-quality videos conditioned on textual descriptions. However, most existing text-to-video models rely solely on textual conditions, lacking general fine-grained controllability over video generation. To address this challenge, we propose InstanceV, a video generation framework that enables i) instance-level control and ii) global semantic consistency. Specifically, with the aid of proposed Instance-aware Masked Cross-Attention mechanism, InstanceV maximizes the utilization of additional instance-level grounding information to generate correctly attributed instances at designated spatial locations. To improve overall consistency, We introduce the Shared Timestep-Adaptive Prompt Enhancement module, which connects local instances with global semantics in a parameter-efficient manner. Furthermore, we incorporate Spatially-Aware Unconditional Guidance during both training and inference to alleviate the disappearance of small instances. Finally, we propose a new benchmark, named InstanceBench, which combines general video quality metrics with instance-aware metrics for more comprehensive evaluation on instance-level video generation. Extensive experiments demonstrate that InstanceV not only achieves remarkable instance-level controllability in video generation, but also outperforms existing state-of-the-art models in both general quality and instance-aware metrics across qualitative and quantitative evaluations.
>
---
#### [new 155] IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对说话人脸生成任务，解决现有方法依赖显式光流导致全局运动建模差、身份漂移的问题。提出IMTalker框架，通过交叉注意力实现隐式运动迁移，结合身份自适应模块与轻量级运动生成器，提升运动精度、身份保真度与同步性，实现40-42 FPS高效生成。**

- **链接: [https://arxiv.org/pdf/2511.22167v1](https://arxiv.org/pdf/2511.22167v1)**

> **作者:** Bo Chen; Tao Liu; Qi Chen; Xie Chen; Zilong Zheng
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Talking face generation aims to synthesize realistic speaking portraits from a single image, yet existing methods often rely on explicit optical flow and local warping, which fail to model complex global motions and cause identity drift. We present IMTalker, a novel framework that achieves efficient and high-fidelity talking face generation through implicit motion transfer. The core idea is to replace traditional flow-based warping with a cross-attention mechanism that implicitly models motion discrepancy and identity alignment within a unified latent space, enabling robust global motion rendering. To further preserve speaker identity during cross-identity reenactment, we introduce an identity-adaptive module that projects motion latents into personalized spaces, ensuring clear disentanglement between motion and identity. In addition, a lightweight flow-matching motion generator produces vivid and controllable implicit motion vectors from audio, pose, and gaze cues. Extensive experiments demonstrate that IMTalker surpasses prior methods in motion accuracy, identity preservation, and audio-lip synchronization, achieving state-of-the-art quality with superior efficiency, operating at 40 FPS for video-driven and 42 FPS for audio-driven generation on an RTX 4090 GPU. We will release our code and pre-trained models to facilitate applications and future research.
>
---
#### [new 156] From Illusion to Intention: Visual Rationale Learning for Vision-Language Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言推理中“伪视觉思考”问题，提出视觉理性学习（ViRL），将视觉操作视为核心推理步骤而非可选工具。通过过程监督、奖励塑形与细粒度责任分配，实现端到端训练，使模型基于正确视觉理由得出答案，提升透明性与可信度。**

- **链接: [https://arxiv.org/pdf/2511.23031v1](https://arxiv.org/pdf/2511.23031v1)**

> **作者:** Changpeng Wang; Haozhe Wang; Xi Chen; Junhan Liu; Taofeng Xue; Chong Peng; Donglian Qi; Fangzhen Lin; Yunfeng Yan
>
> **备注:** 19 pages, 15 figures
>
> **摘要:** Recent advances in vision-language reasoning underscore the importance of thinking with images, where models actively ground their reasoning in visual evidence. Yet, prevailing frameworks treat visual actions as optional tools, boosting metrics but leaving reasoning ungrounded and crops ineffective. This gap gives rise to the illusion of thinking with images: models seem visually grounded but rely on context-agnostic actions that neither refine perception nor guide reasoning toward correct answers. We address this problem by reframing visual actions as core reasoning primitives rather than optional tools, which we term visual rationalization, the visual analogue of textual Chain-of-Thought. Building on this insight, we propose Visual Rationale Learning (ViRL), an end-to-end paradigm that grounds training in the visual rationale itself. ViRL integrates (1) Process Supervision with ground-truth rationales, (2) Objective Alignment via step-level reward shaping, and (3) Fine-Grained Credit Assignment to distinguish correct, redundant, and erroneous actions. By ensuring each action contributes meaningfully to the reasoning chain, ViRL enables models to "get the right answer for the right visual reason". Trained purely with end-to-end RL, ViRL achieves state-of-the-art results across benchmarks spanning perception, hallucination, and reasoning. This work establishes visual rationalization as a task-agnostic, process-grounded paradigm for building transparent, verifiable, and trustworthy vision-language models.
>
---
#### [new 157] Barcode and QR Code Object Detection: An Experimental Study on YOLOv8 Models
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升YOLOv8在条形码和二维码识别中的精度与实时性。通过在Kaggle数据集上训练并优化Nano、Small、Medium版本模型，对比分析其在精度、召回率和F1值上的表现，验证了模型规模对检测性能的积极影响。**

- **链接: [https://arxiv.org/pdf/2511.22937v1](https://arxiv.org/pdf/2511.22937v1)**

> **作者:** Kushagra Pandya; Heli Hathi; Het Buch; Ravikumar R N; Shailendrasinh Chauhan; Sushil Kumar Singh
>
> **备注:** 7 Pages, 16 figures, Presented at 2024 International Conference on Emerging Innovations and Advanced Computing (INNOCOMP) Conference
>
> **摘要:** This research work dives into an in-depth evaluation of the YOLOv8 (You Only Look Once) algorithm's efficiency in object detection, specially focusing on Barcode and QR code recognition. Utilizing the real-time detection abilities of YOLOv8, we performed a study aimed at enhancing its talent in swiftly and correctly figuring out objects. Through large training and high-quality-tuning on Kaggle datasets tailored for Barcode and QR code detection, our goal became to optimize YOLOv8's overall performance throughout numerous situations and environments. The look encompasses the assessment of YOLOv8 throughout special version iterations: Nano, Small, and Medium, with a meticulous attention on precision, recall, and F1 assessment metrics. The consequences exhibit large improvements in object detection accuracy with every subsequent model refinement. Specifically, we achieved an accuracy of 88.95% for the nano model, 97.10% for the small model, and 94.10% for the medium version, showcasing the incremental improvements finished via model scaling. Our findings highlight the big strides made through YOLOv8 in pushing the limits of computer vision, ensuring its function as a milestone within the subject of object detection. This study sheds light on how model scaling affects object recognition, increasing the concept of deep learning-based computer creative and prescient techniques.
>
---
#### [new 158] AI killed the video star. Audio-driven diffusion model for expressive talking head generation
- **分类: cs.CV**

- **简介: 该论文提出Dimitra++，一种音频驱动的逼真人脸生成框架，旨在解决唇动、表情与头部姿态同步生成问题。通过条件运动扩散变换器（cMDT）建模3D面部运动序列，以音频和参考图像为输入，实现高保真表达性说话头生成，在多个数据集上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22488v1](https://arxiv.org/pdf/2511.22488v1)**

> **作者:** Baptiste Chopin; Tashvik Dhamija; Pranav Balaji; Yaohui Wang; Antitza Dantcheva
>
> **备注:** arXiv admin note: text overlap with arXiv:2502.17198
>
> **摘要:** We propose Dimitra++, a novel framework for audio-driven talking head generation, streamlined to learn lip motion, facial expression, as well as head pose motion. Specifically, we propose a conditional Motion Diffusion Transformer (cMDT) to model facial motion sequences, employing a 3D representation. The cMDT is conditioned on two inputs: a reference facial image, which determines appearance, as well as an audio sequence, which drives the motion. Quantitative and qualitative experiments, as well as a user study on two widely employed datasets, i.e., VoxCeleb2 and CelebV-HQ, suggest that Dimitra++ is able to outperform existing approaches in generating realistic talking heads imparting lip motion, facial expression, and head pose.
>
---
#### [new 159] MoE3D: Mixture of Experts meets Multi-Modal 3D Understanding
- **分类: cs.CV**

- **简介: 该论文针对多模态3D理解任务，解决模态异质性导致的融合效率低问题。提出MoE3D框架，引入专家混合机制，通过专用专家网络和顶1门控实现高效跨模态融合，并设计渐进式预训练策略，显著提升性能，在Multi3DRefer上超越现有方法6.1 mIoU。**

- **链接: [https://arxiv.org/pdf/2511.22103v1](https://arxiv.org/pdf/2511.22103v1)**

> **作者:** Yu Li; Yuenan Hou; Yingmei Wei; Xinge Zhu; Yuexin Ma; Wenqi Shao; Yanming Guo
>
> **摘要:** Multi-modal 3D understanding is a fundamental task in computer vision. Previous multi-modal fusion methods typically employ a single, dense fusion network, struggling to handle the significant heterogeneity and complexity across modalities, leading to suboptimal performance. In this paper, we propose MoE3D, which integrates Mixture of Experts (MoE) into the multi-modal learning framework. The core is that we deploy a set of specialized "expert" networks, each adept at processing a specific modality or a mode of cross-modal interaction. Specifically, the MoE-based transformer is designed to better utilize the complementary information hidden in the visual features. Information aggregation module is put forward to further enhance the fusion performance. Top-1 gating is employed to make one expert process features with expert groups, ensuring high efficiency. We further propose a progressive pre-training strategy to better leverage the semantic and 2D prior, thus equipping the network with good initialization. Our MoE3D achieves competitive performance across four prevalent 3D understanding tasks. Notably, our MoE3D surpasses the top-performing counterpart by 6.1 mIoU on Multi3DRefer.
>
---
#### [new 160] Enhanced Graph Convolutional Network with Chebyshev Spectral Graph and Graph Attention for Autism Spectrum Disorder Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自闭症谱系障碍（ASD）的早期客观诊断难题，提出一种融合切比雪夫谱图卷积与图注意力网络的多模态分类模型。利用多源神经影像与表型数据，构建群体图结构，通过多分支特征提取与注意力机制提升分类性能，实现74.82%准确率，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22178v1](https://arxiv.org/pdf/2511.22178v1)**

> **作者:** Adnan Ferdous Ashrafi; Hasanul Kabir
>
> **备注:** 6 pages, 2 figures, 2 tables, Accepted and presented at Image and Vision Computing New Zealand (IVCNZ) 2025
>
> **摘要:** ASD is a complicated neurodevelopmental disorder marked by variation in symptom presentation and neurological underpinnings, making early and objective diagnosis extremely problematic. This paper presents a Graph Convolutional Network (GCN) model, incorporating Chebyshev Spectral Graph Convolution and Graph Attention Networks (GAT), to increase the classification accuracy of ASD utilizing multimodal neuroimaging and phenotypic data. Leveraging the ABIDE I dataset, which contains resting-state functional MRI (rs-fMRI), structural MRI (sMRI), and phenotypic variables from 870 patients, the model leverages a multi-branch architecture that processes each modality individually before merging them via concatenation. Graph structure is encoded using site-based similarity to generate a population graph, which helps in understanding relationship connections across individuals. Chebyshev polynomial filters provide localized spectral learning with lower computational complexity, whereas GAT layers increase node representations by attention-weighted aggregation of surrounding information. The proposed model is trained using stratified five-fold cross-validation with a total input dimension of 5,206 features per individual. Extensive trials demonstrate the enhanced model's superiority, achieving a test accuracy of 74.82\% and an AUC of 0.82 on the entire dataset, surpassing multiple state-of-the-art baselines, including conventional GCNs, autoencoder-based deep neural networks, and multimodal CNNs.
>
---
#### [new 161] All Centers Are at most a Few Tokens Apart: Knowledge Distillation with Domain Invariant Prompt Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对病理图像领域中因不同医疗中心设备与染色差异导致的域偏移问题，提出域不变提示调优（DIPT）方法。通过学习各域专属连续提示并融合为域不变提示，实现视觉-语言模型知识蒸馏，提升学生模型在多中心数据上的泛化能力，显著改善分类性能。**

- **链接: [https://arxiv.org/pdf/2511.22739v1](https://arxiv.org/pdf/2511.22739v1)**

> **作者:** Amir Mohammad Ezzati; Alireza Malekhosseini; Armin Khosravi; Mohammad Hossein Rohban
>
> **摘要:** Domain generalization is critical in computational pathology (CPath) due to inherent domain shifts caused by variations in staining protocols, scanner devices, and imaging settings across clinical centers. Vision-language models (VLMs), such as PLIP-a pathology-tuned CLIP-trained on image-text pairs across diverse domains, serve as strong knowledge distillation sources. However, their zero-shot performance with predefined prompts remains limited due to sensitivity to prompt variations. Moreover, unlike natural images, histopathology centers lack semantic descriptors (e.g., 'sketch'), making it difficult to define domain-specific prompts for clinical centers. This requires a data-driven approach for learning domain-specific and ultimately class-generic continuous prompts. We propose Domain Invariant Prompt Tuning (DIPT) for knowledge distillation process, a novel step that learns multiple input tokens for each domain. These tokens are trained separately for each domain and are averaged across domains, leading to domain-invariant prompts. Our student model then distills knowledge from PLIP's text encoder by leveraging the prompts learned by DIPT. This leads to alignment of visual features with domain-invariant embeddings, enhancing generalization by training on multiple domains. Our method adds a significant improvement in average F1-score to existing state-of-the-art (SOTA) knowledge distillation approaches in domain generalization with histopathology datasets. This work helps the way of deploying robust CPath models in real-world clinical problems with heterogeneous data sources.
>
---
#### [new 162] AnchorFlow: Training-Free 3D Editing via Latent Anchor-Aligned Flows
- **分类: cs.CV**

- **简介: 该论文提出AnchorFlow，一种无需训练的3D编辑方法。针对现有方法因扩散采样噪声导致隐空间锚点不一致、编辑不稳定的问题，通过全局共享隐空间锚点与对齐机制，实现语义一致且结构稳定的3D编辑，无需掩码监督。**

- **链接: [https://arxiv.org/pdf/2511.22357v1](https://arxiv.org/pdf/2511.22357v1)**

> **作者:** Zhenglin Zhou; Fan Ma; Chengzhuo Gui; Xiaobo Xia; Hehe Fan; Yi Yang; Tat-Seng Chua
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Training-free 3D editing aims to modify 3D shapes based on human instructions without model finetuning. It plays a crucial role in 3D content creation. However, existing approaches often struggle to produce strong or geometrically stable edits, largely due to inconsistent latent anchors introduced by timestep-dependent noise during diffusion sampling. To address these limitations, we introduce AnchorFlow, which is built upon the principle of latent anchor consistency. Specifically, AnchorFlow establishes a global latent anchor shared between the source and target trajectories, and enforces coherence using a relaxed anchor-alignment loss together with an anchor-aligned update rule. This design ensures that transformations remain stable and semantically faithful throughout the editing process. By stabilizing the latent reference space, AnchorFlow enables more pronounced semantic modifications. Moreover, AnchorFlow is mask-free. Without mask supervision, it effectively preserves geometric fidelity. Experiments on the Eval3DEdit benchmark show that AnchorFlow consistently delivers semantically aligned and structurally robust edits across diverse editing types. Code is at https://github.com/ZhenglinZhou/AnchorFlow.
>
---
#### [new 163] Guiding the Inner Eye: A Framework for Hierarchical and Flexible Visual Grounded Reasoning
- **分类: cs.CV**

- **简介: 该论文针对多模态视觉推理任务，解决现有模型在端到端强化学习与监督微调间稳定性与灵活性失衡的问题。提出GRiP框架，通过显著性加权IoU奖励和多启发式奖励，引导模型聚焦关键视觉信息并探索多样化合理推理路径，显著提升复杂场景下的视觉推理能力。**

- **链接: [https://arxiv.org/pdf/2511.22172v1](https://arxiv.org/pdf/2511.22172v1)**

> **作者:** Zhaoyang Wei; Wenchao Ding; Yanchao Hao; Xi Chen
>
> **备注:** 9pages
>
> **摘要:** Models capable of "thinking with images" by dynamically grounding their reasoning in visual evidence represent a major leap in multimodal AI. However, replicating and advancing this ability is non-trivial, with current methods often trapped between the instability of end-to-end reinforcement learning (RL) and the rigidity of supervised fine-tuning (SFT). This leads to models that either struggle to learn or lack the cognitive flexibility required for complex, real-world scenes. To navigate this dilemma, we introduce GRiP (Guided Reasoning and Perception), a novel two-stage training framework that cultivates robust and flexible visual grounded reasoning by explicitly guiding the model's perceptual focus and logical pathways. GRiP's core lies in its cognitive-enhanced RL stage, which features two key innovations: (1) a Salience-Weighted IoU Reward that incentivizes the model to prioritize the localization of mission-critical objects over trivial distractors, and (2) a Multi-Heuristic Reward that encourages cognitive flexibility by rewarding diverse yet logically valid reasoning pathways. Initialized from the Qwen2.5-VL-7B model, GRiP demonstrates significant performance gains across multiple challenging benchmarks. It achieves state-of-the-art results among open-source models on the highly challenging TreeBench and V* Bench, proving its effectiveness in complex visual reasoning. Our work demonstrates that moving beyond simplistic rewards and instead guiding models with cognitively-inspired signals for what to see and how to think is crucial for unlocking the next level of multimodal intelligence. The code will be made publicly available.
>
---
#### [new 164] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior
- **分类: cs.CV**

- **简介: 该论文提出Diff-ICMH，一种统一优化机器与人类视觉的图像压缩框架。针对传统方法分别优化感知或分析任务的问题，通过生成先验和语义一致性损失，实现高质量视觉呈现与语义保真。引入标签引导模块，以低开销提升生成能力，支持多智能任务，无需任务定制。**

- **链接: [https://arxiv.org/pdf/2511.22549v1](https://arxiv.org/pdf/2511.22549v1)**

> **作者:** Ruoyu Feng; Yunpeng Qi; Jinming Liu; Yixin Gao; Xin Li; Xin Jin; Zhibo Chen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Image compression methods are usually optimized isolatedly for human perception or machine analysis tasks. We reveal fundamental commonalities between these objectives: preserving accurate semantic information is paramount, as it directly dictates the integrity of critical information for intelligent tasks and aids human understanding. Concurrently, enhanced perceptual quality not only improves visual appeal but also, by ensuring realistic image distributions, benefits semantic feature extraction for machine tasks. Based on this insight, we propose Diff-ICMH, a generative image compression framework aiming for harmonizing machine and human vision in image compression. It ensures perceptual realism by leveraging generative priors and simultaneously guarantees semantic fidelity through the incorporation of Semantic Consistency loss (SC loss) during training. Additionally, we introduce the Tag Guidance Module (TGM) that leverages highly semantic image-level tags to stimulate the pre-trained diffusion model's generative capabilities, requiring minimal additional bit rates. Consequently, Diff-ICMH supports multiple intelligent tasks through a single codec and bitstream without any task-specific adaptation, while preserving high-quality visual experience for human perception. Extensive experimental results demonstrate Diff-ICMH's superiority and generalizability across diverse tasks, while maintaining visual appeal for human perception. Code is available at: https://github.com/RuoyuFeng/Diff-ICMH.
>
---
#### [new 165] See, Rank, and Filter: Important Word-Aware Clip Filtering via Scene Understanding for Moment Retrieval and Highlight Detection
- **分类: cs.CV**

- **简介: 该论文针对视频片段检索（MR）与关键帧检测（HD）任务，解决现有方法忽视查询中关键词重要性、依赖黑箱处理的问题。提出基于场景理解的细粒度剪辑过滤框架，通过多模态大模型识别关键语义词，并设计特征增强与排序过滤模块，实现更精准的语义匹配与剪辑筛选。**

- **链接: [https://arxiv.org/pdf/2511.22906v1](https://arxiv.org/pdf/2511.22906v1)**

> **作者:** YuEun Lee; Jung Uk Kim
>
> **摘要:** Video moment retrieval (MR) and highlight detection (HD) with natural language queries aim to localize relevant moments and key highlights in a video clips. However, existing methods overlook the importance of individual words, treating the entire text query and video clips as a black-box, which hinders contextual understanding. In this paper, we propose a novel approach that enables fine-grained clip filtering by identifying and prioritizing important words in the query. Our method integrates image-text scene understanding through Multimodal Large Language Models (MLLMs) and enhances the semantic understanding of video clips. We introduce a feature enhancement module (FEM) to capture important words from the query and a ranking-based filtering module (RFM) to iteratively refine video clips based on their relevance to these important words. Extensive experiments demonstrate that our approach significantly outperforms existing state-of-the-art methods, achieving superior performance in both MR and HD tasks. Our code is available at: https://github.com/VisualAIKHU/SRF.
>
---
#### [new 166] Fusion or Confusion? Assessing the impact of visible-thermal image fusion for automated wildlife detection
- **分类: cs.CV**

- **简介: 该论文研究可见光与热红外图像融合对野生动物自动检测的影响，旨在提升大蓝鹭个体及巢穴识别精度。针对多源图像配准与融合难题，采用早期和晚期融合策略，结合YOLO11n模型，在保持高召回率前提下显著提升F1分数，验证了多模态融合的有效性，但受限于视场角与配准精度。**

- **链接: [https://arxiv.org/pdf/2511.22768v1](https://arxiv.org/pdf/2511.22768v1)**

> **作者:** Camille Dionne-Pierre; Samuel Foucher; Jérôme Théau; Jérôme Lemaître; Patrick Charbonneau; Maxime Brousseau; Mathieu Varin
>
> **备注:** 19 pages, 9 figures, submitted to Remote Sensing in Ecology and Conservation
>
> **摘要:** Efficient wildlife monitoring methods are necessary for biodiversity conservation and management. The combination of remote sensing, aerial imagery and deep learning offer promising opportunities to renew or improve existing survey methods. The complementary use of visible (VIS) and thermal infrared (TIR) imagery can add information compared to a single-source image and improve results in an automated detection context. However, the alignment and fusion process can be challenging, especially since visible and thermal images usually have different fields of view (FOV) and spatial resolutions. This research presents a case study on the great blue heron (Ardea herodias) to evaluate the performances of synchronous aerial VIS and TIR imagery to automatically detect individuals and nests using a YOLO11n model. Two VIS-TIR fusion methods were tested and compared: an early fusion approach and a late fusion approach, to determine if the addition of the TIR image gives any added value compared to a VIS-only model. VIS and TIR images were automatically aligned using a deep learning model. A principal component analysis fusion method was applied to VIS-TIR image pairs to form the early fusion dataset. A classification and regression tree was used to process the late fusion dataset, based on the detection from the VIS-only and TIR-only trained models. Across all classes, both late and early fusion improved the F1 score compared to the VIS-only model. For the main class, occupied nest, the late fusion improved the F1 score from 90.2 (VIS-only) to 93.0%. This model was also able to identify false positives from both sources with 90% recall. Although fusion methods seem to give better results, this approach comes with a limiting TIR FOV and alignment constraints that eliminate data. Using an aircraft-mounted very high-resolution visible sensor could be an interesting option for operationalizing surveys.
>
---
#### [new 167] CoT4AD: A Vision-Language-Action Model with Explicit Chain-of-Thought Reasoning for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CoT4AD，一种面向自动驾驶的视觉-语言-动作模型，通过显式链式思维推理增强模型的数值与因果推理能力。针对现有模型在复杂场景中推理不足、输入输出映射简化的问题，该工作构建了感知-提问-预测-行动的链式推理框架，提升动态环境下的决策鲁棒性，在真实与仿真数据上均取得领先性能。**

- **链接: [https://arxiv.org/pdf/2511.22532v1](https://arxiv.org/pdf/2511.22532v1)**

> **作者:** Zhaohui Wang; Tengbo Yu; Hao Tang
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Vision-Language-Action (VLA) models have recently attracted growing attention in end-to-end autonomous driving for their strong reasoning capabilities and rich world knowledge. However, existing VLAs often suffer from limited numerical reasoning ability and overly simplified input-output mappings, which hinder their performance in complex driving scenarios requiring step-by-step causal reasoning. To address these challenges, we propose CoT4AD, a novel VLA framework that introduces Chain-of-Thought (CoT) reasoning for autonomous driving to enhance both numerical and causal reasoning in Vision-Language Models (VLMs). CoT4AD integrates visual observations and language instructions to perform semantic reasoning, scene understanding, and trajectory planning. During training, it explicitly models a perception-question-prediction-action CoT to align the reasoning space with the action space across multiple driving tasks. During inference, it performs implicit CoT reasoning to enable consistent numerical reasoning and robust decision-making in dynamic environments. Extensive experiments on both real-world and simulated benchmarks, including nuScenes and Bench2Drive, demonstrate that CoT4AD achieves state-of-the-art performance in both open-loop and closed-loop evaluations. Code will be released upon paper acceptance.
>
---
#### [new 168] A deep learning perspective on Rubens' attribution
- **分类: cs.CV**

- **简介: 该论文属于艺术风格识别任务，旨在解决鲁本斯及其工作室作品的真伪鉴定与作者归属问题。研究构建卷积神经网络，基于已验证画作数据集，提取微观风格特征，实现高精度分类，为艺术史学提供计算辅助方法。**

- **链接: [https://arxiv.org/pdf/2511.22667v1](https://arxiv.org/pdf/2511.22667v1)**

> **作者:** A. Afifi; A. Kalimullin; S. Korchagin; I. Kudryashov
>
> **摘要:** This study explores the use of deep learning for the authentication and attribution of paintings, focusing on the complex case of Peter Paul Rubens and his workshop. A convolutional neural network was trained on a curated dataset of verified and comparative artworks to identify micro-level stylistic features characteristic of the master s hand. The model achieved high classification accuracy and demonstrated the potential of computational analysis to complement traditional art historical expertise, offering new insights into authorship and workshop collaboration.
>
---
#### [new 169] Buffer replay enhances the robustness of multimodal learning under missing-modality
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态学习中模态缺失导致性能下降的问题，提出REplay Prompting（REP）方法。通过构建模态特异性与共享特征缓冲区，利用残差旁路缓存早期特征并重用于深层网络，结合任务感知动态初始化，增强模型对缺失模态的鲁棒性。实验表明，REP在多种基准上均显著优于现有方法，且参数开销极小。**

- **链接: [https://arxiv.org/pdf/2511.23070v1](https://arxiv.org/pdf/2511.23070v1)**

> **作者:** Hongye Zhu; Xuan Liu; Yanwen Ba; Jingye Xue; Shigeng Zhang
>
> **摘要:** Missing modalities consistently lead to significant performance degradation in multimodal models. Existing approaches either synthesize missing modalities at high computational cost or apply prompt-based fine-tuning that relies only on adjacent-layer features and overlooks long-distance contextual information, which may offer additional tolerance to errors when one or more modalities are missing. To address this, we introduce REplay Prompting (REP): (1) construct modality-wise feature buffers via a residual bypass to cache early-layer representations and replay them in deeper layers, mitigating information loss as network depth increases; (2) employ a private-shared feature decoupling strategy, where private buffers preserve modality-specific signals and shared buffers encode cross-modal semantics; and (3) design a task-aware dynamic initialization mechanism to configure these buffers differently, improving stability and generalization under diverse missing-modality conditions. Experiments on vision-language, vision-language-audio, and temporal multimodal benchmarks demonstrate that REP consistently outperforms prior methods under both single- and multi-modality missing scenarios, while introducing only negligible parameter overhead. These results establish REP as a lightweight and effective paradigm for robust multimodal learning in challenging missing-modality environments.
>
---
#### [new 170] PPBoost: Progressive Prompt Boosting for Text-Driven Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对文本驱动医学图像分割中空间精度低、泛化差的问题，提出PPBoost框架。通过将弱文本信号转化为强空间提示，实现零样本下精准的视觉提示生成，显著提升分割性能，且无需标注数据。**

- **链接: [https://arxiv.org/pdf/2511.21984v1](https://arxiv.org/pdf/2511.21984v1)**

> **作者:** Xuchen Li; Hengrui Gu; Mohan Zhang; Qin Liu; Zhen Tan; Xinyuan Zhu; Huixue Zhou; Tianlong Chen; Kaixiong Zhou
>
> **摘要:** Text-prompted foundation models for medical image segmentation offer an intuitive way to delineate anatomical structures from natural language queries, but their predictions often lack spatial precision and degrade under domain shift. In contrast, visual-prompted models achieve strong segmentation performance across diverse modalities by leveraging spatial cues of precise bounding-box (bbox) prompts to guide the segmentation of target lesions. However, it is costly and challenging to obtain the precise visual prompts in clinical practice. We propose PPBoost (Progressive Prompt-Boosting), a framework that bridges these limitations by transforming weak text-derived signals into strong, spatially grounded visual prompts, operating under a strict zero-shot regime with no image- or pixel-level segmentation labels. PPBoost first uses a vision-language model to produce initial pseudo-bboxes conditioned on the textual object descriptions and applies an uncertainty-aware criterion to filter unreliable predictions. The retained image-bboxes pairs are then leveraged to train a pseudo-labeled detector, producing the high-quality bboxes for the query images. During inference, PPBoost further refines the generated bboxes by appropriately expanding them to tightly cover the target anatomical structures. The enhanced spatially-grounding bbox prompts guide existing segmentation models to generate final dense masks, effectively amplifying weak text cues into strong spatial guidance. Across three datasets spanning diverse modalities and anatomies, PPBoost consistently improves Dice and Normalized Surface Distance over text- and visual-prompted baselines and, notably, surpasses few-shot segmentation models without using labeled data. PPBoost can generalize to multiple typical visual segmentation model backbones.
>
---
#### [new 171] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
- **分类: cs.CV**

- **简介: 该论文针对视觉自回归生成（VAR）中全上下文依赖导致的计算效率低、内存占用高的问题，提出Markov-VAR模型。通过将尺度预测建模为马尔可夫过程，引入滑动窗口压缩历史信息，实现高效且稳定的生成。实验表明，该方法显著降低FID和内存消耗，提升了实用性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.23334v1](https://arxiv.org/pdf/2511.23334v1)**

> **作者:** Yu Zhang; Jingyi Liu; Yiwei Shi; Qi Zhang; Duoqian Miao; Changwei Wang; Longbing Cao
>
> **摘要:** Visual AutoRegressive modeling (VAR) based on next-scale prediction has revitalized autoregressive visual generation. Although its full-context dependency, i.e., modeling all previous scales for next-scale prediction, facilitates more stable and comprehensive representation learning by leveraging complete information flow, the resulting computational inefficiency and substantial overhead severely hinder VAR's practicality and scalability. This motivates us to develop a new VAR model with better performance and efficiency without full-context dependency. To address this, we reformulate VAR as a non-full-context Markov process, proposing Markov-VAR. It is achieved via Markovian Scale Prediction: we treat each scale as a Markov state and introduce a sliding window that compresses certain previous scales into a compact history vector to compensate for historical information loss owing to non-full-context dependency. Integrating the history vector with the Markov state yields a representative dynamic state that evolves under a Markov process. Extensive experiments demonstrate that Markov-VAR is extremely simple yet highly effective: Compared to VAR on ImageNet, Markov-VAR reduces FID by 10.5% (256 $\times$ 256) and decreases peak memory consumption by 83.8% (1024 $\times$ 1024). We believe that Markov-VAR can serve as a foundation for future research on visual autoregressive generation and other downstream tasks.
>
---
#### [new 172] Stable-Drift: A Patient-Aware Latent Drift Replay Method for Stabilizing Representations in Continual Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对持续学习中的灾难性遗忘问题，提出基于患者感知的隐空间漂移重放方法Stable-Drift。通过量化样本在域适应后的特征表示变化（隐空间漂移），选择最具表征不稳定的患者数据进行重放，提升模型在跨医院新冠CT分类任务中的稳定性，显著减少遗忘。**

- **链接: [https://arxiv.org/pdf/2511.22615v1](https://arxiv.org/pdf/2511.22615v1)**

> **作者:** Paraskevi-Antonia Theofilou; Anuhya Thota; Stefanos Kollias; Mamatha Thota
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** When deep learning models are sequentially trained on new data, they tend to abruptly lose performance on previously learned tasks, a critical failure known as catastrophic forgetting. This challenge severely limits the deployment of AI in medical imaging, where models must continually adapt to data from new hospitals without compromising established diagnostic knowledge. To address this, we introduce a latent drift-guided replay method that identifies and replays samples with high representational instability. Specifically, our method quantifies this instability via latent drift, the change in a sample internal feature representation after naive domain adaptation. To ensure diversity and clinical relevance, we aggregate drift at the patient level, our memory buffer stores the per patient slices exhibiting the greatest multi-layer representation shift. Evaluated on a cross-hospital COVID-19 CT classification task using state-of-the-art CNN and Vision Transformer backbones, our method substantially reduces forgetting compared to naive fine-tuning and random replay. This work highlights latent drift as a practical and interpretable replay signal for advancing robust continual learning in real world medical settings.
>
---
#### [new 173] Flow Straighter and Faster: Efficient One-Step Generative Modeling via MeanFlow on Rectified Trajectories
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对流模型采样效率低的问题，提出Rectified MeanFlow框架，通过单次重流学习平均速度场实现一步生成。解决了传统方法需多轮重流或收敛慢的问题，结合截断启发式降低残余弯曲，显著提升样本质量和训练效率。**

- **链接: [https://arxiv.org/pdf/2511.23342v1](https://arxiv.org/pdf/2511.23342v1)**

> **作者:** Xinxi Zhang; Shiwei Tan; Quang Nguyen; Quan Dao; Ligong Han; Xiaoxiao He; Tunyu Zhang; Alen Mrdovic; Dimitris Metaxas
>
> **摘要:** Flow-based generative models have recently demonstrated strong performance, yet sampling typically relies on expensive numerical integration of ordinary differential equations (ODEs). Rectified Flow enables one-step sampling by learning nearly straight probability paths, but achieving such straightness requires multiple computationally intensive reflow iterations. MeanFlow achieves one-step generation by directly modeling the average velocity over time; however, when trained on highly curved flows, it suffers from slow convergence and noisy supervision. To address these limitations, we propose Rectified MeanFlow, a framework that models the mean velocity field along the rectified trajectory using only a single reflow step. This eliminates the need for perfectly straightened trajectories while enabling efficient training. Furthermore, we introduce a simple yet effective truncation heuristic that aims to reduce residual curvature and further improve performance. Extensive experiments on ImageNet at 64, 256, and 512 resolutions show that Re-MeanFlow consistently outperforms prior one-step flow distillation and Rectified Flow methods in both sample quality and training efficiency. Code is available at https://github.com/Xinxi-Zhang/Re-MeanFlow.
>
---
#### [new 174] Unexplored flaws in multiple-choice VQA evaluations
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多选题视觉问答（VQA）评估中的未被发现的提示格式偏差。针对当前MLLM评估中因提示格式微小变化导致结果不稳定的問題，通过大规模实验分析七种MLLM与五大数据集，揭示了提示格式对结果的显著影响，且现有缓解策略无效。**

- **链接: [https://arxiv.org/pdf/2511.22341v1](https://arxiv.org/pdf/2511.22341v1)**

> **作者:** Fabio Rosenthal; Sebastian Schmidt; Thorsten Graf; Thorsten Bagodonat; Stephan Günnemann; Leo Schwinn
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate strong capabilities in handling image-text inputs. A common way to assess this ability is through multiple-choice Visual Question Answering (VQA). Earlier works have already revealed that these benchmarks are sensitive to answer choice order, a limitation that can be mitigated through careful design. Yet, we highlight additional, unexplored biases in prompt formatting that question the reliability of current MLLM evaluations. Specifically, we identify three key variation factors in prompt formatting and analyze their impact through a large-scale study involving $\mathbf{\text{seven}}$ MLLMs and $\mathbf{\text{five}}$ VQA datasets, spanning $\mathbf{48}$ distinct $\mathbf{\text{prompt format variations}}$. Our findings reveal that multiple-choice VQA is highly sensitive to minor prompt format changes, even when these changes are semantically neutral. We further demonstrate that these biases persist independently of known order biases or the MLLM's confidence in the correct answer. Finally, we demonstrate that existing bias mitigation strategies fail to address these newly identified biases.
>
---
#### [new 175] Bringing Your Portrait to 3D Presence
- **分类: cs.CV**

- **简介: 该论文聚焦于从单张图像重建可动画3D人像的任务。针对特征表示受姿态/构图影响、数据稀缺及代理网格估计不可靠等问题，提出双UV表示、因子化合成数据集与鲁棒追踪器，实现头、半身及全身的高精度3D重建，仅用半身合成数据即达顶尖性能。**

- **链接: [https://arxiv.org/pdf/2511.22553v1](https://arxiv.org/pdf/2511.22553v1)**

> **作者:** Jiawei Zhang; Lei Chu; Jiahao Li; Zhenyu Zang; Chong Li; Xiao Li; Xun Cao; Hao Zhu; Yan Lu
>
> **备注:** project page: https://zjwfufu.github.io/HuaPi-Page/
>
> **摘要:** We present a unified framework for reconstructing animatable 3D human avatars from a single portrait across head, half-body, and full-body inputs. Our method tackles three bottlenecks: pose- and framing-sensitive feature representations, limited scalable data, and unreliable proxy-mesh estimation. We introduce a Dual-UV representation that maps image features to a canonical UV space via Core-UV and Shell-UV branches, eliminating pose- and framing-induced token shifts. We also build a factorized synthetic data manifold combining 2D generative diversity with geometry-consistent 3D renderings, supported by a training scheme that improves realism and identity consistency. A robust proxy-mesh tracker maintains stability under partial visibility. Together, these components enable strong in-the-wild generalization. Trained only on half-body synthetic data, our model achieves state-of-the-art head and upper-body reconstruction and competitive full-body results. Extensive experiments and analyses further validate the effectiveness of our approach.
>
---
#### [new 176] Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
- **分类: cs.CV**

- **简介: 该论文研究扩散模型蒸馏任务，挑战“分布匹配主导性能”的传统认知。揭示在文本到图像生成中，控制频率增广（CA）才是少步生成的核心驱动力，分布匹配仅起正则化稳定作用。提出解耦训练机制，优化噪声调度，显著提升性能，并被实际项目采用验证。**

- **链接: [https://arxiv.org/pdf/2511.22677v1](https://arxiv.org/pdf/2511.22677v1)**

> **作者:** Dongyang Liu; Peng Gao; David Liu; Ruoyi Du; Zhen Li; Qilong Wu; Xin Jin; Sihan Cao; Shifeng Zhang; Hongsheng Li; Steven Hoi
>
> **摘要:** Diffusion model distillation has emerged as a powerful technique for creating efficient few-step and single-step generators. Among these, Distribution Matching Distillation (DMD) and its variants stand out for their impressive performance, which is widely attributed to their core mechanism of matching the student's output distribution to that of a pre-trained teacher model. In this work, we challenge this conventional understanding. Through a rigorous decomposition of the DMD training objective, we reveal that in complex tasks like text-to-image generation, where CFG is typically required for desirable few-step performance, the primary driver of few-step distillation is not distribution matching, but a previously overlooked component we identify as CFG Augmentation (CA). We demonstrate that this term acts as the core ``engine'' of distillation, while the Distribution Matching (DM) term functions as a ``regularizer'' that ensures training stability and mitigates artifacts. We further validate this decoupling by demonstrating that while the DM term is a highly effective regularizer, it is not unique; simpler non-parametric constraints or GAN-based objectives can serve the same stabilizing function, albeit with different trade-offs. This decoupling of labor motivates a more principled analysis of the properties of both terms, leading to a more systematic and in-depth understanding. This new understanding further enables us to propose principled modifications to the distillation process, such as decoupling the noise schedules for the engine and the regularizer, leading to further performance gains. Notably, our method has been adopted by the Z-Image ( https://github.com/Tongyi-MAI/Z-Image ) project to develop a top-tier 8-step image generation model, empirically validating the generalization and robustness of our findings.
>
---
#### [new 177] Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文针对视频时序定位（VTG）任务中模型无法拒绝语义相似但不相关的“硬无关查询”问题，提出基于强化学习的拒绝感知微调方法RA-RFT。通过多目标奖励机制与自建HI-VTG数据集，提升模型对硬无关查询的识别与拒绝能力，增强语义推理与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.23151v1](https://arxiv.org/pdf/2511.23151v1)**

> **作者:** Jin-Seop Lee; SungJoon Lee; SeongJun Jung; Boyang Li; Jee-Hyong Lee
>
> **备注:** 19 pages
>
> **摘要:** Video Temporal Grounding (VTG) aims to localize a temporal segment in a video corresponding to a natural language query. However, existing VTG models assume that a relevant segment always exists, causing them to always predict a target segment even when the query is irrelevant to the video. While recent approaches attempt to handle irrelevant queries, they can only reject those that are entirely unrelated to the video and still fail to handle hard-irrelevant queries that are semantically similar but not actually relevant. To address this, we propose Refusal-Aware Reinforcement Fine-Tuning (RA-RFT) to effectively refuse hard-irrelevant queries in VTG. Our method is based on the Group Relative Policy Optimization (GRPO) framework and integrates four reward objectives-format, refuse-IoU, explain, and query correction-to improve both relevance discrimination and fine-grained semantic reasoning. In addition, to effectively support RA-RFT, we construct a Hard-Irrelevant VTG (HI-VTG) dataset, which includes hard-irrelevant queries and their refusal answers. We demonstrate the effectiveness of our method across various relevance-aware VTG scenarios, including hard-irrelevant VTG, simply-shuffled RA-VTG, and human-annotated RA-VTG settings. We also show that the proposed method is scalable by applying it to various LVLM-based VTG models. Our code is available at https://github.com/JINSUBY/RA-RFT.
>
---
#### [new 178] Simultaneous Image Quality Improvement and Artefacts Correction in Accelerated MRI
- **分类: cs.CV; cs.AI; physics.med-ph**

- **简介: 该论文针对加速MRI中图像质量下降与伪影并存的问题，提出USArt模型，同时实现欠采样数据重建与噪声、运动伪影校正。基于双子模型架构，有效提升信噪比与对比度，在5倍加速下仍保持高质量图像，解决了现有方法无法兼顾加速与伪影修正的局限。**

- **链接: [https://arxiv.org/pdf/2511.23274v1](https://arxiv.org/pdf/2511.23274v1)**

> **作者:** Georgia Kanli; Daniele Perlo; Selma Boudissa; Radovan Jirik; Olivier Keunen
>
> **摘要:** MR data are acquired in the frequency domain, known as k-space. Acquiring high-quality and high-resolution MR images can be time-consuming, posing a significant challenge when multiple sequences providing complementary contrast information are needed or when the patient is unable to remain in the scanner for an extended period of time. Reducing k-space measurements is a strategy to speed up acquisition, but often leads to reduced quality in reconstructed images. Additionally, in real-world MRI, both under-sampled and full-sampled images are prone to artefacts, and correcting these artefacts is crucial for maintaining diagnostic accuracy. Deep learning methods have been proposed to restore image quality from under-sampled data, while others focused on the correction of artefacts that result from the noise or motion. No approach has however been proposed so far that addresses both acceleration and artefacts correction, limiting the performance of these models when these degradation factors occur simultaneously. To address this gap, we present a method for recovering high-quality images from under-sampled data with simultaneously correction for noise and motion artefact called USArt (Under-Sampling and Artifact correction model). Customized for 2D brain anatomical images acquired with Cartesian sampling, USArt employs a dual sub-model approach. The results demonstrate remarkable increase of signal-to-noise ratio (SNR) and contrast in the images restored. Various under-sampling strategies and degradation levels were explored, with the gradient under-sampling strategy yielding the best outcomes. We achieved up to 5x acceleration and simultaneously artefacts correction without significant degradation, showcasing the model's robustness in real-world settings.
>
---
#### [new 179] DEAL-300K: Diffusion-based Editing Area Localization with a 300K-Scale Dataset and Frequency-Prompted Baseline
- **分类: cs.CV**

- **简介: 该论文针对扩散模型生成图像的局部篡改难以定位的问题，提出DEAL-300K数据集与基于多频提示调优的定位框架。通过自动生成编辑指令与图像，构建30万级像素级标注数据，结合冻结视觉基础模型与频率域特征捕捉，实现高精度编辑区域定位，为扩散图像篡改检测提供强基线。**

- **链接: [https://arxiv.org/pdf/2511.23377v1](https://arxiv.org/pdf/2511.23377v1)**

> **作者:** Rui Zhang; Hongxia Wang; Hangqing Liu; Yang Zhou; Qiang Zeng
>
> **备注:** 13pages,12 figures
>
> **摘要:** Diffusion-based image editing has made semantic level image manipulation easy for general users, but it also enables realistic local forgeries that are hard to localize. Existing benchmarks mainly focus on the binary detection of generated images or the localization of manually edited regions and do not reflect the properties of diffusion-based edits, which often blend smoothly into the original content. We present Diffusion-Based Image Editing Area Localization Dataset (DEAL-300K), a large scale dataset for diffusion-based image manipulation localization (DIML) with more than 300,000 annotated images. We build DEAL-300K by using a multi-modal large language model to generate editing instructions, a mask-free diffusion editor to produce manipulated images, and an active-learning change detection pipeline to obtain pixel-level annotations. On top of this dataset, we propose a localization framework that uses a frozen Visual Foundation Model (VFM) together with Multi Frequency Prompt Tuning (MFPT) to capture both semantic and frequency-domain cues of edited regions. Trained on DEAL-300K, our method reaches a pixel-level F1 score of 82.56% on our test split and 80.97% on the external CoCoGlide benchmark, providing strong baselines and a practical foundation for future DIML research.The dataset can be accessed via https://github.com/ymhzyj/DEAL-300K.
>
---
#### [new 180] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对2D高斯点阵渲染中纹理采样效率低的问题，提出FACT-GS框架。通过频率对齐的自适应采样策略，根据局部视觉复杂度动态调整纹理采样密度，实现更优细节表现与参数利用。在保持实时性能前提下，显著提升图像清晰度与结构保真度。**

- **链接: [https://arxiv.org/pdf/2511.23292v1](https://arxiv.org/pdf/2511.23292v1)**

> **作者:** Tianhao Xie; Linlian Jiang; Xinxin Zuo; Yang Wang; Tiberiu Popa
>
> **备注:** 11 pages, 6 figures, preprint
>
> **摘要:** Realistic scene appearance modeling has advanced rapidly with Gaussian Splatting, which enables real-time, high-quality rendering. Recent advances introduced per-primitive textures that incorporate spatial color variations within each Gaussian, improving their expressiveness. However, texture-based Gaussians parameterize appearance with a uniform per-Gaussian sampling grid, allocating equal sampling density regardless of local visual complexity. This leads to inefficient texture space utilization, where high-frequency regions are under-sampled and smooth regions waste capacity, causing blurred appearance and loss of fine structural detail. We introduce FACT-GS, a Frequency-Aligned Complexity-aware Texture Gaussian Splatting framework that allocates texture sampling density according to local visual frequency. Grounded in adaptive sampling theory, FACT-GS reformulates texture parameterization as a differentiable sampling-density allocation problem, replacing the uniform textures with a learnable frequency-aware allocation strategy implemented via a deformation field whose Jacobian modulates local sampling density. Built on 2D Gaussian Splatting, FACT-GS performs non-uniform sampling on fixed-resolution texture grids, preserving real-time performance while recovering sharper high-frequency details under the same parameter budget.
>
---
#### [new 181] Zero-Shot Multi-Criteria Visual Quality Inspection for Semi-Controlled Industrial Environments via Real-Time 3D Digital Twin Simulation
- **分类: cs.CV**

- **简介: 该论文针对半受控工业环境中视觉质量检测数据需求高、泛化性差的问题，提出一种零样本、姿态无关的实时3D数字孪生检测框架。通过RGB-D空间对比真实场景与数字孪生，结合已知CAD模型实现高效渲染与缺陷多标准标注，基于汽车电机案例验证，仅用简单距离度量即达63.3% IoU，推动低数据、可扩展的质量检测研究。**

- **链接: [https://arxiv.org/pdf/2511.23214v1](https://arxiv.org/pdf/2511.23214v1)**

> **作者:** Jose Moises Araya-Martinez; Gautham Mohan; Kenichi Hayakawa Bolaños; Roberto Mendieta; Sarvenaz Sardari; Jens Lambrecht; Jörg Krüger
>
> **摘要:** Early-stage visual quality inspection is vital for achieving Zero-Defect Manufacturing and minimizing production waste in modern industrial environments. However, the complexity of robust visual inspection systems and their extensive data requirements hinder widespread adoption in semi-controlled industrial settings. In this context, we propose a pose-agnostic, zero-shot quality inspection framework that compares real scenes against real-time Digital Twins (DT) in the RGB-D space. Our approach enables efficient real-time DT rendering by semantically describing industrial scenes through object detection and pose estimation of known Computer-Aided Design models. We benchmark tools for real-time, multimodal RGB-D DT creation while tracking consumption of computational resources. Additionally, we provide an extensible and hierarchical annotation strategy for multi-criteria defect detection, unifying pose labelling with logical and structural defect annotations. Based on an automotive use case featuring the quality inspection of an axial flux motor, we demonstrate the effectiveness of our framework. Our results demonstrate detection performace, achieving intersection-over-union (IoU) scores of up to 63.3% compared to ground-truth masks, even if using simple distance measurements under semi-controlled industrial conditions. Our findings lay the groundwork for future research on generalizable, low-data defect detection methods in dynamic manufacturing settings.
>
---
#### [new 182] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction
- **分类: cs.CV**

- **简介: 该论文针对空间转录组学中图像与基因表达的跨模态预测任务，解决现有方法忽视数据层次结构及模态信息不对称的问题。提出HyperST框架，通过超球面空间建模多层级图像-基因表示，实现层次化对齐，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2511.22107v1](https://arxiv.org/pdf/2511.22107v1)**

> **作者:** Chen Zhang; Yilu An; Ying Chen; Hao Li; Xitong Ling; Lihao Liu; Junjun He; Yuxiang Lin; Zihui Wang; Rongshan Yu
>
> **摘要:** Spatial Transcriptomics (ST) merges the benefits of pathology images and gene expression, linking molecular profiles with tissue structure to analyze spot-level function comprehensively. Predicting gene expression from histology images is a cost-effective alternative to expensive ST technologies. However, existing methods mainly focus on spot-level image-to-gene matching but fail to leverage the full hierarchical structure of ST data, especially on the gene expression side, leading to incomplete image-gene alignment. Moreover, a challenge arises from the inherent information asymmetry: gene expression profiles contain more molecular details that may lack salient visual correlates in histological images, demanding a sophisticated representation learning approach to bridge this modality gap. We propose HyperST, a framework for ST prediction that learns multi-level image-gene representations by modeling the data's inherent hierarchy within hyperbolic space, a natural geometric setting for such structures. First, we design a Multi-Level Representation Extractors to capture both spot-level and niche-level representations from each modality, providing context-aware information beyond individual spot-level image-gene pairs. Second, a Hierarchical Hyperbolic Alignment module is introduced to unify these representations, performing spatial alignment while hierarchically structuring image and gene embeddings. This alignment strategy enriches the image representations with molecular semantics, significantly improving cross-modal prediction. HyperST achieves state-of-the-art performance on four public datasets from different tissues, paving the way for more scalable and accurate spatial transcriptomics prediction.
>
---
#### [new 183] Creating Blank Canvas Against AI-enabled Image Forgery
- **分类: cs.CV**

- **简介: 该论文针对AIGC图像伪造带来的安全威胁，提出基于SAM的篡改检测新方法。通过引入频域感知的对抗扰动使SAM“看不见”图像，将图像变为神经模型视角下的“空白画布”。当图像被篡改时，异常区域会显现，从而实现精准定位。解决了高保真图像伪造难以检测的问题。**

- **链接: [https://arxiv.org/pdf/2511.22237v1](https://arxiv.org/pdf/2511.22237v1)**

> **作者:** Qi Song; Ziyuan Luo; Renjie Wan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** AIGC-based image editing technology has greatly simplified the realistic-level image modification, causing serious potential risks of image forgery. This paper introduces a new approach to tampering detection using the Segment Anything Model (SAM). Instead of training SAM to identify tampered areas, we propose a novel strategy. The entire image is transformed into a blank canvas from the perspective of neural models. Any modifications to this blank canvas would be noticeable to the models. To achieve this idea, we introduce adversarial perturbations to prevent SAM from ``seeing anything'', allowing it to identify forged regions when the image is tampered with. Due to SAM's powerful perceiving capabilities, naive adversarial attacks cannot completely tame SAM. To thoroughly deceive SAM and make it blind to the image, we introduce a frequency-aware optimization strategy, which further enhances the capability of tamper localization. Extensive experimental results demonstrate the effectiveness of our method.
>
---
#### [new 184] What Shape Is Optimal for Masks in Text Removal?
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究文档图像中文本移除任务，针对复杂密集文本场景下掩码形状对修复效果的影响问题，提出基于贝叶斯优化的灵活掩码建模方法，发现字符级掩码更优，非最小覆盖最优，为手动掩码提供实用指导。**

- **链接: [https://arxiv.org/pdf/2511.22499v1](https://arxiv.org/pdf/2511.22499v1)**

> **作者:** Hyakka Nakada; Marika Kubota
>
> **备注:** 12 pages, 17 figures
>
> **摘要:** The advent of generative models has dramatically improved the accuracy of image inpainting. In particular, by removing specific text from document images, reconstructing original images is extremely important for industrial applications. However, most existing methods of text removal focus on deleting simple scene text which appears in images captured by a camera in an outdoor environment. There is little research dedicated to complex and practical images with dense text. Therefore, we created benchmark data for text removal from images including a large amount of text. From the data, we found that text-removal performance becomes vulnerable against mask profile perturbation. Thus, for practical text-removal tasks, precise tuning of the mask shape is essential. This study developed a method to model highly flexible mask profiles and learn their parameters using Bayesian optimization. The resulting profiles were found to be character-wise masks. It was also found that the minimum cover of a text region is not optimal. Our research is expected to pave the way for a user-friendly guideline for manual masking.
>
---
#### [new 185] Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models
- **分类: cs.CV**

- **简介: 该论文提出WUKONG，一种无需训练的高保真纹理3D变形框架。针对传统方法依赖人工匹配与预处理、泛化性差的问题，利用流模型生成先验，通过最优传输和语义一致性机制实现平滑几何过渡与精确纹理保留，显著提升多样性与质量。**

- **链接: [https://arxiv.org/pdf/2511.22425v1](https://arxiv.org/pdf/2511.22425v1)**

> **作者:** Minghao Yin; Yukang Cao; Kai Han
>
> **摘要:** We present WUKONG, a novel training-free framework for high-fidelity textured 3D morphing that takes a pair of source and target prompts (image or text) as input. Unlike conventional methods -- which rely on manual correspondence matching and deformation trajectory estimation (limiting generalization and requiring costly preprocessing) -- WUKONG leverages the generative prior of flow-based transformers to produce high-fidelity 3D transitions with rich texture details. To ensure smooth shape transitions, we exploit the inherent continuity of flow-based generative processes and formulate morphing as an optimal transport barycenter problem. We further introduce a sequential initialization strategy to prevent abrupt geometric distortions and preserve identity coherence. For faithful texture preservation, we propose a similarity-guided semantic consistency mechanism that selectively retains high-frequency details and enables precise control over blending dynamics. This avoids common artifacts like oversmoothing while maintaining semantic fidelity. Extensive quantitative and qualitative evaluations demonstrate that WUKONG significantly outperforms state-of-the-art methods, achieving superior results across diverse geometry and texture variations.
>
---
#### [new 186] Fast Multi-view Consistent 3D Editing with Video Priors
- **分类: cs.CV**

- **简介: 该论文针对文本驱动3D编辑中多视图不一致、效率低的问题，提出基于视频生成模型先验的单次前向传播方法ViP3DE。通过条件化视频模型生成一致编辑视图，并结合运动保持噪声融合与几何感知去噪，实现高效高质的多视图一致性3D编辑。**

- **链接: [https://arxiv.org/pdf/2511.23172v1](https://arxiv.org/pdf/2511.23172v1)**

> **作者:** Liyi Chen; Ruihuang Li; Guowen Zhang; Pengfei Wang; Lei Zhang
>
> **摘要:** Text-driven 3D editing enables user-friendly 3D object or scene editing with text instructions. Due to the lack of multi-view consistency priors, existing methods typically resort to employing 2D generation or editing models to process each view individually, followed by iterative 2D-3D-2D updating. However, these methods are not only time-consuming but also prone to over-smoothed results because the different editing signals gathered from different views are averaged during the iterative process. In this paper, we propose generative Video Prior based 3D Editing (ViP3DE) to employ the temporal consistency priors from pre-trained video generation models for multi-view consistent 3D editing in a single forward pass. Our key insight is to condition the video generation model on a single edited view to generate other consistent edited views for 3D updating directly, thereby bypassing the iterative editing paradigm. Since 3D updating requires edited views to be paired with specific camera poses, we propose motion-preserved noise blending for the video model to generate edited views at predefined camera poses. In addition, we introduce geometry-aware denoising to further enhance multi-view consistency by integrating 3D geometric priors into video models. Extensive experiments demonstrate that our proposed ViP3DE can achieve high-quality 3D editing results even within a single forward pass, significantly outperforming existing methods in both editing quality and speed.
>
---
#### [new 187] CoordSpeaker: Exploiting Gesture Captioning for Coordinated Caption-Empowered Co-Speech Gesture Generation
- **分类: cs.CV**

- **简介: 该论文针对语音同步手势生成任务，解决手势数据缺乏文本语义标注导致的语义鸿沟及多模态协同控制难的问题。提出CoordSpeaker框架，通过运动-语言模型生成多粒度手势描述，并设计统一表示与分层去噪的条件扩散模型，实现语义一致、节奏同步的可控手势生成。**

- **链接: [https://arxiv.org/pdf/2511.22863v1](https://arxiv.org/pdf/2511.22863v1)**

> **作者:** Fengyi Fang; Sicheng Yang; Wenming Yang
>
> **摘要:** Co-speech gesture generation has significantly advanced human-computer interaction, yet speaker movements remain constrained due to the omission of text-driven non-spontaneous gestures (e.g., bowing while talking). Existing methods face two key challenges: 1) the semantic prior gap due to the lack of descriptive text annotations in gesture datasets, and 2) the difficulty in achieving coordinated multimodal control over gesture generation. To address these challenges, this paper introduces CoordSpeaker, a comprehensive framework that enables coordinated caption-empowered co-speech gesture synthesis. Our approach first bridges the semantic prior gap through a novel gesture captioning framework, leveraging a motion-language model to generate descriptive captions at multiple granularities. Building upon this, we propose a conditional latent diffusion model with unified cross-dataset motion representation and a hierarchically controlled denoiser to achieve highly controlled, coordinated gesture generation. CoordSpeaker pioneers the first exploration of gesture understanding and captioning to tackle the semantic gap in gesture generation while offering a novel perspective of bidirectional gesture-text mapping. Extensive experiments demonstrate that our method produces high-quality gestures that are both rhythmically synchronized with speeches and semantically coherent with arbitrary captions, achieving superior performance with higher efficiency compared to existing approaches.
>
---
#### [new 188] RemedyGS: Defend 3D Gaussian Splatting against Computation Cost Attacks
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文针对3D高斯溅射（3DGS）系统面临的计算成本攻击问题，提出首个黑盒防御框架RemedyGS。通过检测受污染图像并净化恢复，结合对抗训练提升重建质量，有效抵御白盒、黑盒及自适应攻击，保障3DGS系统的安全与可用性。**

- **链接: [https://arxiv.org/pdf/2511.22147v1](https://arxiv.org/pdf/2511.22147v1)**

> **作者:** Yanping Li; Zhening Liu; Zijian Li; Zehong Lin; Jun Zhang
>
> **摘要:** As a mainstream technique for 3D reconstruction, 3D Gaussian splatting (3DGS) has been applied in a wide range of applications and services. Recent studies have revealed critical vulnerabilities in this pipeline and introduced computation cost attacks that lead to malicious resource occupancies and even denial-of-service (DoS) conditions, thereby hindering the reliable deployment of 3DGS. In this paper, we propose the first effective and comprehensive black-box defense framework, named RemedyGS, against such computation cost attacks, safeguarding 3DGS reconstruction systems and services. Our pipeline comprises two key components: a detector to identify the attacked input images with poisoned textures and a purifier to recover the benign images from their attacked counterparts, mitigating the adverse effects of these attacks. Moreover, we incorporate adversarial training into the purifier to enforce distributional alignment between the recovered and original natural images, thereby enhancing the defense efficacy. Experimental results demonstrate that our framework effectively defends against white-box, black-box, and adaptive attacks in 3DGS systems, achieving state-of-the-art performance in both safety and utility.
>
---
#### [new 189] Leveraging Textual Compositional Reasoning for Robust Change Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像变化描述任务，解决视觉特征难以捕捉细微、结构性变化的问题。提出CORTEX框架，融合视觉与文本推理，通过文本引导增强对对象关系和组合语义的理解，实现更鲁棒的细粒度变化描述。**

- **链接: [https://arxiv.org/pdf/2511.22903v1](https://arxiv.org/pdf/2511.22903v1)**

> **作者:** Kyu Ri Park; Jiyoung Park; Seong Tae Kim; Hong Joo Lee; Jung Uk Kim
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Change captioning aims to describe changes between a pair of images. However, existing works rely on visual features alone, which often fail to capture subtle but meaningful changes because they lack the ability to represent explicitly structured information such as object relationships and compositional semantics. To alleviate this, we present CORTEX (COmpositional Reasoning-aware TEXt-guided), a novel framework that integrates complementary textual cues to enhance change understanding. In addition to capturing cues from pixel-level differences, CORTEX utilizes scene-level textual knowledge provided by Vision Language Models (VLMs) to extract richer image text signals that reveal underlying compositional reasoning. CORTEX consists of three key modules: (i) an Image-level Change Detector that identifies low-level visual differences between paired images, (ii) a Reasoning-aware Text Extraction (RTE) module that use VLMs to generate compositional reasoning descriptions implicit in visual features, and (iii) an Image-Text Dual Alignment (ITDA) module that aligns visual and textual features for fine-grained relational reasoning. This enables CORTEX to reason over visual and textual features and capture changes that are otherwise ambiguous in visual features alone.
>
---
#### [new 190] BlockVid: Block Diffusion for High-Quality and Consistent Minute-Long Video Generation
- **分类: cs.CV**

- **简介: 该论文针对分钟级视频生成任务，解决块扩散模型中的长时序误差累积与缺乏细粒度评估基准问题。提出BlockVid框架，引入语义感知稀疏KV缓存、块强制训练策略及分段噪声调度，提升生成质量与一致性。构建LV-Bench基准与新评估指标，实验表明其显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22973v1](https://arxiv.org/pdf/2511.22973v1)**

> **作者:** Zeyu Zhang; Shuning Chang; Yuanyu He; Yizeng Han; Jiasheng Tang; Fan Wang; Bohan Zhuang
>
> **摘要:** Generating minute-long videos is a critical step toward developing world models, providing a foundation for realistic extended scenes and advanced AI simulators. The emerging semi-autoregressive (block diffusion) paradigm integrates the strengths of diffusion and autoregressive models, enabling arbitrary-length video generation and improving inference efficiency through KV caching and parallel sampling. However, it yet faces two enduring challenges: (i) KV-cache-induced long-horizon error accumulation, and (ii) the lack of fine-grained long-video benchmarks and coherence-aware metrics. To overcome these limitations, we propose BlockVid, a novel block diffusion framework equipped with semantic-aware sparse KV cache, an effective training strategy called Block Forcing, and dedicated chunk-wise noise scheduling and shuffling to reduce error propagation and enhance temporal consistency. We further introduce LV-Bench, a fine-grained benchmark for minute-long videos, complete with new metrics evaluating long-range coherence. Extensive experiments on VBench and LV-Bench demonstrate that BlockVid consistently outperforms existing methods in generating high-quality, coherent minute-long videos. In particular, it achieves a 22.2% improvement on VDE Subject and a 19.4% improvement on VDE Clarity in LV-Bench over the state of the art approaches. Project website: https://ziplab.co/BlockVid. Inferix (Code): https://github.com/alibaba-damo-academy/Inferix.
>
---
#### [new 191] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation
- **分类: cs.CV**

- **简介: 该论文针对文本生成多人物图像时出现人脸重复、身份混淆和人数错误的问题，提出Ar2Can框架。通过“建筑师”模块规划布局，“艺术家”模块生成图像，并结合空间对齐与身份相似性奖励，实现精准的身份保留与人数控制。使用合成数据训练，显著提升多人类图像生成质量。**

- **链接: [https://arxiv.org/pdf/2511.22690v1](https://arxiv.org/pdf/2511.22690v1)**

> **作者:** Shubhankar Borse; Phuc Pham; Farzad Farhadzadeh; Seokeon Choi; Phong Ha Nguyen; Anh Tuan Tran; Sungrack Yun; Munawar Hayat; Fatih Porikli
>
> **摘要:** Despite recent advances in text-to-image generation, existing models consistently fail to produce reliable multi-human scenes, often duplicating faces, merging identities, or miscounting individuals. We present Ar2Can, a novel two-stage framework that disentangles spatial planning from identity rendering for multi-human generation. The Architect module predicts structured layouts, specifying where each person should appear. The Artist module then synthesizes photorealistic images, guided by a spatially-grounded face matching reward that combines Hungarian spatial alignment with ArcFace identity similarity. This approach ensures faces are rendered at correct locations and faithfully preserve reference identities. We develop two Architect variants, seamlessly integrated with our diffusion-based Artist model and optimized via Group Relative Policy Optimization (GRPO) using compositional rewards for count accuracy, image quality, and identity matching. Evaluated on the MultiHuman-Testbench, Ar2Can achieves substantial improvements in both count accuracy and identity preservation, while maintaining high perceptual quality. Notably, our method achieves these results using primarily synthetic data, without requiring real multi-human images.
>
---
#### [new 192] Beyond Real versus Fake Towards Intent-Aware Video Analysis
- **分类: cs.CV**

- **简介: 该论文提出面向意图的视频分析任务，旨在超越真假判断，识别伪造视频背后的动机。针对深度伪造视频带来的风险，构建了包含5168个视频的IntentHQ基准，涵盖23类细粒度意图，并提出多模态模型融合时空特征、音频与文本信息，实现意图识别。**

- **链接: [https://arxiv.org/pdf/2511.22455v1](https://arxiv.org/pdf/2511.22455v1)**

> **作者:** Saurabh Atreya; Nabyl Quignon; Baptiste Chopin; Abhijit Das; Antitza Dantcheva
>
> **摘要:** The rapid advancement of generative models has led to increasingly realistic deepfake videos, posing significant societal and security risks. While existing detection methods focus on distinguishing real from fake videos, such approaches fail to address a fundamental question: What is the intent behind a manipulated video? Towards addressing this question, we introduce IntentHQ: a new benchmark for human-centered intent analysis, shifting the paradigm from authenticity verification to contextual understanding of videos. IntentHQ consists of 5168 videos that have been meticulously collected and annotated with 23 fine-grained intent-categories, including "Financial fraud", "Indirect marketing", "Political propaganda", as well as "Fear mongering". We perform intent recognition with supervised and self-supervised multi-modality models that integrate spatio-temporal video features, audio processing, and text analysis to infer underlying motivations and goals behind videos. Our proposed model is streamlined to differentiate between a wide range of intent-categories.
>
---
#### [new 193] MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶端到端轨迹规划任务，提出MTR-VP方法，用视觉Transformer替代地图特征，通过图像与运动状态生成场景上下文嵌入，并结合多轨迹预测提升规划性能。实验表明，多轨迹输出优于单轨迹，而单纯融合视觉与运动特征效果有限。**

- **链接: [https://arxiv.org/pdf/2511.22181v1](https://arxiv.org/pdf/2511.22181v1)**

> **作者:** Maitrayee Keskar; Mohan Trivedi; Ross Greer
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** We present a method for trajectory planning for autonomous driving, learning image-based context embeddings that align with motion prediction frameworks and planning-based intention input. Within our method, a ViT encoder takes raw images and past kinematic state as input and is trained to produce context embeddings, drawing inspiration from those generated by the recent MTR (Motion Transformer) encoder, effectively substituting map-based features with learned visual representations. MTR provides a strong foundation for multimodal trajectory prediction by localizing agent intent and refining motion iteratively via motion query pairs; we name our approach MTR-VP (Motion Transformer for Vision-based Planning), and instead of the learnable intention queries used in the MTR decoder, we use cross attention on the intent and the context embeddings, which reflect a combination of information encoded from the driving scene and past vehicle states. We evaluate our methods on the Waymo End-to-End Driving Dataset, which requires predicting the agent's future 5-second trajectory in bird's-eye-view coordinates using prior camera images, agent pose history, and routing goals. We analyze our architecture using ablation studies, removing input images and multiple trajectory output. Our results suggest that transformer-based methods that are used to combine the visual features along with the kinetic features such as the past trajectory features are not effective at combining both modes to produce useful scene context embeddings, even when intention embeddings are augmented with foundation-model representations of scene context from CLIP and DINOv2, but that predicting a distribution over multiple futures instead of a single future trajectory boosts planning performance.
>
---
#### [new 194] 3D-Consistent Multi-View Editing by Diffusion Guidance
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对多视角图像编辑中的3D一致性问题，提出一种无需训练的扩散引导框架。通过引入一致性损失，确保不同视图中对应点的编辑变化一致，提升NeRF、高斯溅射等3D表示的编辑质量，支持密集与稀疏编辑，显著改善多视角几何与光影一致性。**

- **链接: [https://arxiv.org/pdf/2511.22228v1](https://arxiv.org/pdf/2511.22228v1)**

> **作者:** Josef Bengtson; David Nilsson; Dong In Lee; Fredrik Kahl
>
> **摘要:** Recent advancements in diffusion models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian Splat models. We propose a training-free diffusion framework that enforces multi-view consistency during the image editing process. The key assumption is that corresponding points in the unedited images should undergo similar transformations after editing. To achieve this, we introduce a consistency loss that guides the diffusion sampling toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view editing methods. We also show that this increased consistency enables high-quality Gaussian Splat editing with sharp details and strong fidelity to user-specified text prompts. Please refer to our project page for video results: https://3d-consistent-editing.github.io/
>
---
#### [new 195] Partially Shared Concept Bottleneck Models
- **分类: cs.CV**

- **简介: 该论文针对概念瓶颈模型（CBM）在视觉任务中存在视觉定位差、概念冗余及缺乏平衡准确率与紧凑性的度量问题。提出PS-CBM框架，通过多模态概念生成、部分共享概念策略和概念高效准确率（CEA）指标，提升模型可解释性与性能。在11个数据集上验证，显著提升准确率与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.22170v1](https://arxiv.org/pdf/2511.22170v1)**

> **作者:** Delong Zhao; Qiang Huang; Di Yan; Yiqun Sun; Jun Yu
>
> **备注:** 14 pages, 7 figures, 11 tables, Accepted to AAAI 2026
>
> **摘要:** Concept Bottleneck Models (CBMs) enhance interpretability by introducing a layer of human-understandable concepts between inputs and predictions. While recent methods automate concept generation using Large Language Models (LLMs) and Vision-Language Models (VLMs), they still face three fundamental challenges: poor visual grounding, concept redundancy, and the absence of principled metrics to balance predictive accuracy and concept compactness. We introduce PS-CBM, a Partially Shared CBM framework that addresses these limitations through three core components: (1) a multimodal concept generator that integrates LLM-derived semantics with exemplar-based visual cues; (2) a Partially Shared Concept Strategy that merges concepts based on activation patterns to balance specificity and compactness; and (3) Concept-Efficient Accuracy (CEA), a post-hoc metric that jointly captures both predictive accuracy and concept compactness. Extensive experiments on eleven diverse datasets show that PS-CBM consistently outperforms state-of-the-art CBMs, improving classification accuracy by 1.0%-7.4% and CEA by 2.0%-9.5%, while requiring significantly fewer concepts. These results underscore PS-CBM's effectiveness in achieving both high accuracy and strong interpretability.
>
---
#### [new 196] Robust 3DGS-based SLAM via Adaptive Kernel Smoothing
- **分类: cs.CV**

- **简介: 该论文针对3DGS-SLAM中渲染质量与跟踪精度的关系，提出通过自适应核平滑增强光栅化过程的鲁棒性。针对参数误差导致的跟踪不稳问题，设计CB-KNN方法动态调整邻近高斯分布，引入可控模糊作为正则化，提升姿态估计稳定性，同时保持重建质量。**

- **链接: [https://arxiv.org/pdf/2511.23221v1](https://arxiv.org/pdf/2511.23221v1)**

> **作者:** Shouhe Zhang; Dayong Ren; Sensen Song; Wenjie Li; Piaopiao Yu; Yurong Qian
>
> **摘要:** In this paper, we challenge the conventional notion in 3DGS-SLAM that rendering quality is the primary determinant of tracking accuracy. We argue that, compared to solely pursuing a perfect scene representation, it is more critical to enhance the robustness of the rasterization process against parameter errors to ensure stable camera pose tracking. To address this challenge, we propose a novel approach that leverages a smooth kernel strategy to enhance the robustness of 3DGS-based SLAM. Unlike conventional methods that focus solely on minimizing rendering error, our core insight is to make the rasterization process more resilient to imperfections in the 3DGS parameters. We hypothesize that by allowing each Gaussian to influence a smoother, wider distribution of pixels during rendering, we can mitigate the detrimental effects of parameter noise from outlier Gaussians. This approach intentionally introduces a controlled blur to the rendered image, which acts as a regularization term, stabilizing the subsequent pose optimization. While a complete redesign of the rasterization pipeline is an ideal solution, we propose a practical and effective alternative that is readily integrated into existing 3DGS frameworks. Our method, termed Corrective Blurry KNN (CB-KNN), adaptively modifies the RGB values and locations of the K-nearest neighboring Gaussians within a local region. This dynamic adjustment generates a smoother local rendering, reducing the impact of erroneous GS parameters on the overall image. Experimental results demonstrate that our approach, while maintaining the overall quality of the scene reconstruction (mapping), significantly improves the robustness and accuracy of camera pose tracking.
>
---
#### [new 197] db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉生成模型中稀疏注意力的推理加速问题，提出db-SP方法。针对序列并行导致的负载不均问题，设计双平衡分区与动态并行度调整机制，实现头级与块级近似完美负载均衡，显著提升推理效率。**

- **链接: [https://arxiv.org/pdf/2511.23113v1](https://arxiv.org/pdf/2511.23113v1)**

> **作者:** Siqi Chen; Ke Hong; Tianchen Zhao; Ruiqi Xie; Zhenhua Zhu; Xudong Zhang; Yu Wang
>
> **摘要:** Scaling Diffusion Transformer (DiT) inference via sequence parallelism is critical for reducing latency in visual generation, but is severely hampered by workload imbalance when applied to models employing block-wise sparse attention. The imbalance stems from the inherent variation in sparsity across attention heads and the irregular distribution of dense blocks within the sparse mask, when sequence parallelism is applied along the head dimension (as in Ulysses) or the block dimension (as in Ring Attention). In this paper, we formalize a sparse imbalance ratio to quantify the imbalance, and propose db-SP, a sparsity-aware sequence parallelism technique that tackles the challenge. db-SP contains a dual-level partitioning approach that achieves near-perfect workload balance at both the head and block levels with negligible overhead. Furthermore, to handle the evolving sparsity patterns across denoising steps and layers, db-SP dynamically determines the parallel degrees for the head and block dimensions at runtime. Experimental results demonstrate that db-SP delivers an end-to-end speedup of 1.25x and an attention-specific speedup of 1.40x over state-of-the-art sequence parallel methods on average. Code is available at https://github.com/thu-nics/db-SP.
>
---
#### [new 198] PROMPTMINER: Black-Box Prompt Stealing against Text-to-Image Generative Models via Reinforcement Learning and Fuzz Optimization
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成模型中的提示盗取问题，提出PROMPTMINER框架。它在黑盒环境下通过强化学习优化主体、模糊测试搜索风格修饰符，实现高效精准的提示恢复，显著提升相似度与鲁棒性，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22119v1](https://arxiv.org/pdf/2511.22119v1)**

> **作者:** Mingzhe Li; Renhao Zhang; Zhiyang Wen; Siqi Pan; Bruno Castro da Silva; Juan Zhai; Shiqing Ma
>
> **摘要:** Text-to-image (T2I) generative models such as Stable Diffusion and FLUX can synthesize realistic, high-quality images directly from textual prompts. The resulting image quality depends critically on well-crafted prompts that specify both subjects and stylistic modifiers, which have become valuable digital assets. However, the rising value and ubiquity of high-quality prompts expose them to security and intellectual-property risks. One key threat is the prompt stealing attack, i.e., the task of recovering the textual prompt that generated a given image. Prompt stealing enables unauthorized extraction and reuse of carefully engineered prompts, yet it can also support beneficial applications such as data attribution, model provenance analysis, and watermarking validation. Existing approaches often assume white-box gradient access, require large-scale labeled datasets for supervised training, or rely solely on captioning without explicit optimization, limiting their practicality and adaptability. To address these challenges, we propose PROMPTMINER, a black-box prompt stealing framework that decouples the task into two phases: (1) a reinforcement learning-based optimization phase to reconstruct the primary subject, and (2) a fuzzing-driven search phase to recover stylistic modifiers. Experiments across multiple datasets and diffusion backbones demonstrate that PROMPTMINER achieves superior results, with CLIP similarity up to 0.958 and textual alignment with SBERT up to 0.751, surpassing all baselines. Even when applied to in-the-wild images with unknown generators, it outperforms the strongest baseline by 7.5 percent in CLIP similarity, demonstrating better generalization. Finally, PROMPTMINER maintains strong performance under defensive perturbations, highlighting remarkable robustness. Code: https://github.com/aaFrostnova/PromptMiner
>
---
#### [new 199] Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval
- **分类: cs.CV**

- **简介: 该论文针对文本驱动的人体异常检索任务，旨在提升模型细粒度特征提取能力。提出局部-全局混合视角模块（LHP）与统一图文模型（UIT），设计迭代集成策略及基于LHP的特征选择算法，显著提升检索性能，在PAB数据集上实现SOTA效果。**

- **链接: [https://arxiv.org/pdf/2511.22470v1](https://arxiv.org/pdf/2511.22470v1)**

> **作者:** Tien-Huy Nguyen; Huu-Loc Tran; Huu-Phong Phan-Nguyen; Quang-Vinh Dinh
>
> **备注:** Accepted on World Wide Web 2025 Workshop
>
> **摘要:** Text-based person anomaly retrieval has emerged as a challenging task, with most existing approaches relying on complex deep-learning techniques. This raises a research question: How can the model be optimized to achieve greater fine-grained features? To address this, we propose a Local-Global Hybrid Perspective (LHP) module integrated with a Vision-Language Model (VLM), designed to explore the effectiveness of incorporating both fine-grained features alongside coarse-grained features. Additionally, we investigate a Unified Image-Text (UIT) model that combines multiple objective loss functions, including Image-Text Contrastive (ITC), Image-Text Matching (ITM), Masked Language Modeling (MLM), and Masked Image Modeling (MIM) loss. Beyond this, we propose a novel iterative ensemble strategy, by combining iteratively instead of using model results simultaneously like other ensemble methods. To take advantage of the superior performance of the LHP model, we introduce a novel feature selection algorithm based on its guidance, which helps improve the model's performance. Extensive experiments demonstrate the effectiveness of our method in achieving state-of-the-art (SOTA) performance on PAB dataset, compared with previous work, with a 9.70\% improvement in R@1, 1.77\% improvement in R@5, and 1.01\% improvement in R@10.
>
---
#### [new 200] Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis
- **分类: cs.CV**

- **简介: 该论文针对稀疏输入下动态场景视图合成中几何不一致的问题，提出GC-4DGS框架。通过动态一致性检查与全局-局部深度正则化，提升4D高斯溅射的几何一致性，实现高效高质量渲染，显著优于现有方法，适用于资源受限的边缘设备。**

- **链接: [https://arxiv.org/pdf/2511.23044v1](https://arxiv.org/pdf/2511.23044v1)**

> **作者:** Yiwei Li; Jiannong Cao; Penghui Ruan; Divya Saxena; Songye Zhu; Yinfeng Cao
>
> **摘要:** Gaussian Splatting has been considered as a novel way for view synthesis of dynamic scenes, which shows great potential in AIoT applications such as digital twins. However, recent dynamic Gaussian Splatting methods significantly degrade when only sparse input views are available, limiting their applicability in practice. The issue arises from the incoherent learning of 4D geometry as input views decrease. This paper presents GC-4DGS, a novel framework that infuses geometric consistency into 4D Gaussian Splatting (4DGS), offering real-time and high-quality dynamic scene rendering from sparse input views. While learning-based Multi-View Stereo (MVS) and monocular depth estimators (MDEs) provide geometry priors, directly integrating these with 4DGS yields suboptimal results due to the ill-posed nature of sparse-input 4D geometric optimization. To address these problems, we introduce a dynamic consistency checking strategy to reduce estimation uncertainties of MVS across spacetime. Furthermore, we propose a global-local depth regularization approach to distill spatiotemporal-consistent geometric information from monocular depths, thereby enhancing the coherent geometry and appearance learning within the 4D volume. Extensive experiments on the popular N3DV and Technicolor datasets validate the effectiveness of GC-4DGS in rendering quality without sacrificing efficiency. Notably, our method outperforms RF-DeRF, the latest dynamic radiance field tailored for sparse-input dynamic view synthesis, and the original 4DGS by 2.62dB and 1.58dB in PSNR, respectively, with seamless deployability on resource-constrained IoT edge devices.
>
---
#### [new 201] The Collapse of Patches
- **分类: cs.CV**

- **简介: 该论文提出“补丁坍缩”概念，揭示图像补丁间依赖关系。通过自编码器学习补丁间的软选择关系，构建补丁优先级排序。利用此顺序提升掩码图像建模效率，在自回归生成和分类任务中均取得显著效果，推动视觉建模的高效性。**

- **链接: [https://arxiv.org/pdf/2511.22281v1](https://arxiv.org/pdf/2511.22281v1)**

> **作者:** Wei Guo; Shunqi Mao; Zhuonan Liang; Heng Wang; Weidong Cai
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** Observing certain patches in an image reduces the uncertainty of others. Their realization lowers the distribution entropy of each remaining patch feature, analogous to collapsing a particle's wave function in quantum mechanics. This phenomenon can intuitively be called patch collapse. To identify which patches are most relied on during a target region's collapse, we learn an autoencoder that softly selects a subset of patches to reconstruct each target patch. Graphing these learned dependencies for each patch's PageRank score reveals the optimal patch order to realize an image. We show that respecting this order benefits various masked image modeling methods. First, autoregressive image generation can be boosted by retraining the state-of-the-art model MAR. Next, we introduce a new setup for image classification by exposing Vision Transformers only to high-rank patches in the collapse order. Seeing 22\% of such patches is sufficient to achieve high accuracy. With these experiments, we propose patch collapse as a novel image modeling perspective that promotes vision efficiency. Our project is available at https://github.com/wguo-ai/CoP .
>
---
#### [new 202] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文针对多模态大模型在视觉输入下生成结构化输出的能力，提出SO-Bench基准。解决现有缺乏系统评估框架的问题，覆盖四类视觉场景，包含超6500个JSON模式与1800对图像-模式对。通过基准测试揭示模型在结构化推理上的不足，并开展训练优化，推动多模态结构化生成发展。**

- **链接: [https://arxiv.org/pdf/2511.21750v1](https://arxiv.org/pdf/2511.21750v1)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We plan to make the benchmark available to the community.
>
---
#### [new 203] Language-guided 3D scene synthesis for fine-grained functionality understanding
- **分类: cs.CV**

- **简介: 该论文针对3D场景功能理解中真实数据稀缺问题，提出SynthFun3D方法，基于动作描述自动生成可执行任务的3D室内场景。通过结合带部件标注的家具库，自动识别并生成正确功能部件，实现低成本、大规模高质量标注数据的合成，有效支持数据密集型3D应用。**

- **链接: [https://arxiv.org/pdf/2511.23230v1](https://arxiv.org/pdf/2511.23230v1)**

> **作者:** Jaime Corsetti; Francesco Giuliari; Davide Boscaini; Pedro Hermosilla; Andrea Pilzer; Guofeng Mei; Alexandros Delitzas; Francis Engelmann; Fabio Poiesi
>
> **备注:** Technical report. 24 pages, 19 figures, 2 tables
>
> **摘要:** Functionality understanding in 3D, which aims to identify the functional element in a 3D scene to complete an action (e.g., the correct handle to "Open the second drawer of the cabinet near the bed"), is hindered by the scarcity of real-world data due to the substantial effort needed for its collection and annotation. To address this, we introduce SynthFun3D, the first method for task-based 3D scene synthesis. Given the action description, SynthFun3D generates a 3D indoor environment using a furniture asset database with part-level annotation, ensuring the action can be accomplished. It reasons about the action to automatically identify and retrieve the 3D mask of the correct functional element, enabling the inexpensive and large-scale generation of high-quality annotated data. We validate SynthFun3D through user studies, which demonstrate improved scene-prompt coherence compared to other approaches. Our quantitative results further show that the generated data can either replace real data with minor performance loss or supplement real data for improved performance, thereby providing an inexpensive and scalable solution for data-hungry 3D applications. Project page: github.com/tev-fbk/synthfun3d.
>
---
#### [new 204] Prompt-based Consistent Video Colorization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频着色中的时序闪烁与人工干预需求问题，提出基于语言和分割引导的自动着色方法。利用语言条件扩散模型结合自动生成的物体掩码与文本提示，通过光流对齐实现时序一致性，并引入修正步骤消除失真，显著提升着色精度与视觉真实感。**

- **链接: [https://arxiv.org/pdf/2511.22330v1](https://arxiv.org/pdf/2511.22330v1)**

> **作者:** Silvia Dani; Tiberio Uricchio; Lorenzo Seidenari
>
> **摘要:** Existing video colorization methods struggle with temporal flickering or demand extensive manual input. We propose a novel approach automating high-fidelity video colorization using rich semantic guidance derived from language and segmentation. We employ a language-conditioned diffusion model to colorize grayscale frames. Guidance is provided via automatically generated object masks and textual prompts; our primary automatic method uses a generic prompt, achieving state-of-the-art results without specific color input. Temporal stability is achieved by warping color information from previous frames using optical flow (RAFT); a correction step detects and fixes inconsistencies introduced by warping. Evaluations on standard benchmarks (DAVIS30, VIDEVO20) show our method achieves state-of-the-art performance in colorization accuracy (PSNR) and visual realism (Colorfulness, CDC), demonstrating the efficacy of automated prompt-based guidance for consistent video colorization.
>
---
#### [new 205] Ovis-Image Technical Report
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Ovis-Image，一个7B参数的文本到图像生成模型，专注高质量文本渲染。针对小模型在文本渲染任务中性能不足的问题，结合强多模态主干与文本导向训练策略，在单块高端GPU上实现高效部署，性能媲美更大或闭源模型，推动高精度文本渲染的实用化。**

- **链接: [https://arxiv.org/pdf/2511.22982v1](https://arxiv.org/pdf/2511.22982v1)**

> **作者:** Guo-Hua Wang; Liangfu Cao; Tianyu Cui; Minghao Fu; Xiaohao Chen; Pengxin Zhan; Jianshan Zhao; Lan Li; Bowen Fu; Jiaqi Liu; Qing-Guo Chen
>
> **备注:** Code is released at https://github.com/AIDC-AI/Ovis-Image
>
> **摘要:** We introduce $\textbf{Ovis-Image}$, a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints. Built upon our previous Ovis-U1 framework, Ovis-Image integrates a diffusion-based visual decoder with the stronger Ovis 2.5 multimodal backbone, leveraging a text-centric training pipeline that combines large-scale pre-training with carefully tailored post-training refinements. Despite its compact architecture, Ovis-Image achieves text rendering performance on par with significantly larger open models such as Qwen-Image and approaches closed-source systems like Seedream and GPT4o. Crucially, the model remains deployable on a single high-end GPU with moderate memory, narrowing the gap between frontier-level text rendering and practical deployment. Our results indicate that combining a strong multimodal backbone with a carefully designed, text-focused training recipe is sufficient to achieve reliable bilingual text rendering without resorting to oversized or proprietary models.
>
---
#### [new 206] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely $F_1$
- **分类: cs.PF; cs.AI; cs.CV; cs.LG; stat.ML**

- **简介: 该论文研究分类模型评价中精度与召回率的权衡问题，旨在寻找最优综合评分。针对广泛使用的Fβ分数不具最优排名性的问题，提出基于肯德尔相关系数的优化框架，给出β的闭式解，证明F₁并非最优，并提供六组案例验证。**

- **链接: [https://arxiv.org/pdf/2511.22442v1](https://arxiv.org/pdf/2511.22442v1)**

> **作者:** Sébastien Piérard; Adrien Deliège; Marc Van Droogenbroeck
>
> **摘要:** Ranking methods or models based on their performance is of prime importance but is tricky because performance is fundamentally multidimensional. In the case of classification, precision and recall are scores with probabilistic interpretations that are both important to consider and complementary. The rankings induced by these two scores are often in partial contradiction. In practice, therefore, it is extremely useful to establish a compromise between the two views to obtain a single, global ranking. Over the last fifty years or so,it has been proposed to take a weighted harmonic mean, known as the F-score, F-measure, or $F_β$. Generally speaking, by averaging basic scores, we obtain a score that is intermediate in terms of values. However, there is no guarantee that these scores lead to meaningful rankings and no guarantee that the rankings are good tradeoffs between these base scores. Given the ubiquity of $F_β$ scores in the literature, some clarification is in order. Concretely: (1) We establish that $F_β$-induced rankings are meaningful and define a shortest path between precision- and recall-induced rankings. (2) We frame the problem of finding a tradeoff between two scores as an optimization problem expressed with Kendall rank correlations. We show that $F_1$ and its skew-insensitive version are far from being optimal in that regard. (3) We provide theoretical tools and a closed-form expression to find the optimal value for $β$ for any distribution or set of performances, and we illustrate their use on six case studies.
>
---
#### [new 207] MARVO: Marine-Adaptive Radiance-aware Visual Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对水下视觉定位难题，提出MARVO框架，融合物理成像模型与学习机制。通过辐射度自适应特征匹配提升复杂水下环境下的特征一致性，结合惯性、压力与视觉信息构建因子图优化系统，利用强化学习优化全局轨迹，实现高精度实时位姿估计。**

- **链接: [https://arxiv.org/pdf/2511.22860v1](https://arxiv.org/pdf/2511.22860v1)**

> **作者:** Sacchin Sundar; Atman Kikani; Aaliya Alam; Sumukh Shrote; A. Nayeemulla Khan; A. Shahina
>
> **备注:** 10 pages, 5 figures, 3 tables, Submitted to CVPR2026
>
> **摘要:** Underwater visual localization remains challenging due to wavelength-dependent attenuation, poor texture, and non-Gaussian sensor noise. We introduce MARVO, a physics-aware, learning-integrated odometry framework that fuses underwater image formation modeling, differentiable matching, and reinforcement-learning optimization. At the front-end, we extend transformer-based feature matcher with a Physics Aware Radiance Adapter that compensates for color channel attenuation and contrast loss, yielding geometrically consistent feature correspondences under turbidity. These semi dense matches are combined with inertial and pressure measurements inside a factor-graph backend, where we formulate a keyframe-based visual-inertial-barometric estimator using GTSAM library. Each keyframe introduces (i) Pre-integrated IMU motion factors, (ii) MARVO-derived visual pose factors, and (iii) barometric depth priors, giving a full-state MAP estimate in real time. Lastly, we introduce a Reinforcement-Learningbased Pose-Graph Optimizer that refines global trajectories beyond local minima of classical least-squares solvers by learning optimal retraction actions on SE(2).
>
---
#### [new 208] DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DiskChunGS，一种基于分块内存管理的3D高斯SLAM系统，解决3D高斯溅射在大规模场景中因GPU显存不足导致的可扩展性问题。通过将场景分块，仅将活动区域保留在GPU内存中，其余部分存于磁盘，实现大场景全局一致重建，支持多种环境与嵌入式平台。**

- **链接: [https://arxiv.org/pdf/2511.23030v1](https://arxiv.org/pdf/2511.23030v1)**

> **作者:** Casimir Feldmann; Maximum Wilder-Smith; Vaishakh Patil; Michael Oechsle; Michael Niemeyer; Keisuke Tateno; Marco Hutter
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated impressive results for novel view synthesis with real-time rendering capabilities. However, integrating 3DGS with SLAM systems faces a fundamental scalability limitation: methods are constrained by GPU memory capacity, restricting reconstruction to small-scale environments. We present DiskChunGS, a scalable 3DGS SLAM system that overcomes this bottleneck through an out-of-core approach that partitions scenes into spatial chunks and maintains only active regions in GPU memory while storing inactive areas on disk. Our architecture integrates seamlessly with existing SLAM frameworks for pose estimation and loop closure, enabling globally consistent reconstruction at scale. We validate DiskChunGS on indoor scenes (Replica, TUM-RGBD), urban driving scenarios (KITTI), and resource-constrained Nvidia Jetson platforms. Our method uniquely completes all 11 KITTI sequences without memory failures while achieving superior visual quality, demonstrating that algorithmic innovation can overcome the memory constraints that have limited previous 3DGS SLAM methods.
>
---
#### [new 209] Geodiffussr: Generative Terrain Texturing with Elevation Fidelity
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出Geodiffussr，用于生成与数字高程图（DEM）严格一致的地形纹理。针对大规模地形生成劳动密集的问题，通过多尺度内容聚合机制，将DEM特征注入UNet，强化高程与外观的一致性，实现可控2.5D景观生成。**

- **链接: [https://arxiv.org/pdf/2511.23029v1](https://arxiv.org/pdf/2511.23029v1)**

> **作者:** Tai Inui; Alexander Matsumura; Edgar Simo-Serra
>
> **摘要:** Large-scale terrain generation remains a labor-intensive task in computer graphics. We introduce Geodiffussr, a flow-matching pipeline that synthesizes text-guided texture maps while strictly adhering to a supplied Digital Elevation Map (DEM). The core mechanism is multi-scale content aggregation (MCA): DEM features from a pretrained encoder are injected into UNet blocks at multiple resolutions to enforce global-to-local elevation consistency. Compared with a non-MCA baseline, MCA markedly improves visual fidelity and strengthens height-appearance coupling (FID $\downarrow$ 49.16%, LPIPS $\downarrow$ 32.33%, $Δ$dCor $\downarrow$ to 0.0016). To train and evaluate Geodiffussr, we assemble a globally distributed, biome- and climate-stratified corpus of triplets pairing SRTM-derived DEMs with Sentinel-2 imagery and vision-grounded natural-language captions that describe visible land cover. We position Geodiffussr as a strong baseline and step toward controllable 2.5D landscape generation for coarse-scale ideation and previz, complementary to physically based terrain and ecosystem simulators.
>
---
#### [new 210] SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对端到端自动驾驶中的不确定性盲区问题，提出一种基于摄像头的鲁棒规划框架。通过在BEV空间显式建模摄动不确定性，生成像素级语义-几何可行驶性图，并引入车道跟随正则化以增强规则合规性。方法提升了复杂场景下的安全性和可解释性，在NAVSIM基准上实现领先性能。**

- **链接: [https://arxiv.org/pdf/2511.22865v1](https://arxiv.org/pdf/2511.22865v1)**

> **作者:** Wonjeong Ryu; Seungjun Yu; Seokha Moon; Hojun Choi; Junsung Park; Jinkyu Kim; Hyunjung Shim
>
> **摘要:** End-to-End (E2E) planning has become a powerful paradigm for autonomous driving, yet current systems remain fundamentally uncertainty-blind. They assume perception outputs are fully reliable, even in ambiguous or poorly observed scenes, leaving the planner without an explicit measure of uncertainty. To address this limitation, we propose a camera-only E2E framework that estimates aleatoric uncertainty directly in BEV space and incorporates it into planning. Our method produces a dense, uncertainty-aware drivability map that captures both semantic structure and geometric layout at pixel-level resolution. To further promote safe and rule-compliant behavior, we introduce a lane-following regularization that encodes lane structure and traffic norms. This prior stabilizes trajectory planning under normal conditions while preserving the flexibility needed for maneuvers such as overtaking or lane changes. Together, these components enable robust and interpretable trajectory planning, even under challenging uncertainty conditions. Evaluated on the NAVSIM benchmark, our method achieves state-of-the-art performance, delivering substantial gains on both the challenging NAVHARD and NAVSAFE subsets. These results demonstrate that our principled aleatoric uncertainty modeling combined with driving priors significantly advances the safety and reliability of camera-only E2E autonomous driving.
>
---
#### [new 211] Content Adaptive Encoding For Interactive Game Streaming
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对交互式游戏流媒体（IGS）中的内容自适应编码（CAE）难题，提出一种基于历史编码块统计的轻量级分辨率自适应方法。利用训练好的CNN模型，仅用1ms CPU时间即可预测下一场景最优分辨率，显著提升画质（2.3 BD-VMAF），且无延迟开销，适用于低延迟、高算力受限的实时场景。**

- **链接: [https://arxiv.org/pdf/2511.22327v1](https://arxiv.org/pdf/2511.22327v1)**

> **作者:** Shakarim Soltanayev; Odysseas Zisimopoulos; Mohammad Ashraful Anam; Man Cheung Kung; Angeliki Katsenou; Yiannis Andreopoulos
>
> **备注:** 5 pages
>
> **摘要:** Video-on-demand streaming has benefitted from \textit{content-adaptive encoding} (CAE), i.e., adaptation of resolution and/or quantization parameters for each scene based on convex hull optimization. However, CAE is very challenging to develop and deploy for interactive game streaming (IGS). Commercial IGS services impose ultra-low latency encoding with no lookahead or buffering, and have extremely tight compute constraints for any CAE algorithm execution. We propose the first CAE approach for resolution adaptation in IGS based on compact encoding metadata from past frames. Specifically, we train a convolutional neural network (CNN) to infer the best resolution from the options available for the upcoming scene based on a running window of aggregated coding block statistics from the current scene. By deploying the trained CNN within a practical IGS setup based on HEVC encoding, our proposal: (i) improves over the default fixed-resolution ladder of HEVC by 2.3 Bjøntegaard Delta-VMAF points; (ii) infers using 1ms of a single CPU core per scene, thereby having no latency overhead.
>
---
#### [new 212] Machine Learning for Scientific Visualization: Ensemble Data Analysis
- **分类: cs.LG; cs.AI; cs.CV; cs.GR**

- **简介: 该论文面向科学可视化中的时空数据集分析任务，针对高维、复杂、缺失数据的分析难题，提出基于深度学习的降维、流场估计与时间插值方法。通过自编码器实现稳定低维嵌入，提出FLINT与HyperFLINT模型，实现无需领域假设的高质量流场重建与跨域泛化插值。**

- **链接: [https://arxiv.org/pdf/2511.23290v1](https://arxiv.org/pdf/2511.23290v1)**

> **作者:** Hamid Gadirov
>
> **备注:** PhD thesis, University of Groningen, 2025
>
> **摘要:** Scientific simulations and experimental measurements produce vast amounts of spatio-temporal data, yet extracting meaningful insights remains challenging due to high dimensionality, complex structures, and missing information. Traditional analysis methods often struggle with these issues, motivating the need for more robust, data-driven approaches. This dissertation explores deep learning methodologies to improve the analysis and visualization of spatio-temporal scientific ensembles, focusing on dimensionality reduction, flow estimation, and temporal interpolation. First, we address high-dimensional data representation through autoencoder-based dimensionality reduction for scientific ensembles. We evaluate the stability of projection metrics under partial labeling and introduce a Pareto-efficient selection strategy to identify optimal autoencoder variants, ensuring expressive and reliable low-dimensional embeddings. Next, we present FLINT, a deep learning model for high-quality flow estimation and temporal interpolation in both flow-supervised and flow-unsupervised settings. FLINT reconstructs missing velocity fields and generates high-fidelity temporal interpolants for scalar fields across 2D+time and 3D+time ensembles without domain-specific assumptions or extensive finetuning. To further improve adaptability and generalization, we introduce HyperFLINT, a hypernetwork-based approach that conditions on simulation parameters to estimate flow fields and interpolate scalar data. This parameter-aware adaptation yields more accurate reconstructions across diverse scientific domains, even with sparse or incomplete data. Overall, this dissertation advances deep learning techniques for scientific visualization, providing scalable, adaptable, and high-quality solutions for interpreting complex spatio-temporal ensembles.
>
---
#### [new 213] Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对自回归Transformer在长序列中误差累积的问题，提出闭环预测原则，通过迭代优化潜空间实现自洽推理。构建Equilibrium Transformer，利用能量函数引导潜变量收敛，提升长程推理与一致性。实验表明其在难题上显著优于传统模型，为语言建模提供了新范式。**

- **链接: [https://arxiv.org/pdf/2511.21882v1](https://arxiv.org/pdf/2511.21882v1)**

> **作者:** Akbar Anbar Jafari; Gholamreza Anbarjafari
>
> **备注:** 22 pages, 1 figure, 1 table
>
> **摘要:** Contemporary autoregressive transformers operate in open loop: each hidden state is computed in a single forward pass and never revised, causing errors to propagate uncorrected through the sequence. We identify this open-loop bottleneck as a fundamental architectural limitation underlying well-documented failures in long-range reasoning, factual consistency, and multi-step planning. To address this limitation, we introduce the closed-loop prediction principle, which requires that models iteratively refine latent representations until reaching a self-consistent equilibrium before committing to each token. We instantiate this principle as Equilibrium Transformers (EqT), which augment standard transformer layers with an Equilibrium Refinement Module that minimizes a learned energy function via gradient descent in latent space. The energy function enforces bidirectional prediction consistency, episodic memory coherence, and output confidence, all computed without external supervision. Theoretically, we prove that EqT performs approximate MAP inference in a latent energy-based model, establish linear convergence guarantees, and show that refinement improves predictions precisely on hard instances where one-shot inference is suboptimal. The framework unifies deep equilibrium models, diffusion language models, and test-time training as special cases. Preliminary experiments on the binary parity task demonstrate +3.28% average improvement on challenging sequences, with gains reaching +8.07% where standard transformers approach random performance, validating that the benefit of deliberation scales with task difficulty. Just as attention mechanisms resolved the sequential bottleneck of recurrent networks, we propose that closed-loop equilibrium may resolve the commitment bottleneck of open-loop autoregression, representing a foundational step toward language models.
>
---
#### [new 214] Geometrically-Constrained Agent for Spatial Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文针对视觉语言模型在空间推理中存在语义与几何脱节的问题，提出无需训练的几何约束代理（GCA）。通过将VLM分为语义分析与任务求解两阶段，以形式化约束确保推理过程符合几何精度，显著提升空间推理准确率，优于现有方法约27%。**

- **链接: [https://arxiv.org/pdf/2511.22659v1](https://arxiv.org/pdf/2511.22659v1)**

> **作者:** Zeren Chen; Xiaoya Lu; Zhijie Zheng; Pengrui Li; Lehan He; Yijin Zhou; Jing Shao; Bohan Zhuang; Lu Sheng
>
> **备注:** 27 pages, 13 figures
>
> **摘要:** Vision Language Models (VLMs) exhibit a fundamental semantic-to-geometric gap in spatial reasoning: they excel at qualitative semantic inference but their reasoning operates within a lossy semantic space, misaligned with high-fidelity geometry. Current paradigms fail to bridge this gap. Training-based methods suffer from an ``oracle paradox,'' learning flawed spatial logic from imperfect oracles. Tool-integrated methods constrain the final computation but critically leave the VLM's planning process unconstrained, resulting in geometrically flawed plans. In this work, we propose Geometrically-Constrained Agent (GCA), a training-free agentic paradigm that resolves this gap by introducing a formal task constraint. Specifically, we strategically decouples the VLM's role into two stages. First, acting as a semantic analyst, the VLM translates the user's ambiguous query into the formal, verifiable task constraint, which defines the reference frame and objective. Second, acting as a task solver, the VLM generates and executes tool calls strictly within the deterministic bounds defined by the constraint. This geometrically-constrained reasoning strategy successfully resolve the semantic-to-geometric gap, yielding a robust and verifiable reasoning pathway for spatial reasoning. Comprehensive experiments demonstrate that GCA achieves SOTA performance on multiple spatial reasoning benchmarks, surpassing existing training-based and tool-integrated methods by ~27%. Please see our homepage at https://gca-spatial-reasoning.github.io.
>
---
#### [new 215] Hard Spatial Gating for Precision-Driven Brain Metastasis Segmentation: Addressing the Over-Segmentation Paradox in Deep Attention Networks
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对脑转移瘤MRI分割任务，解决深度注意力网络中的“过分割悖论”问题。提出硬空间门控网络（SG-Net），通过严格特征筛选提升精度，显著改善边界定位与精确率，同时参数量更少，适用于资源受限场景。**

- **链接: [https://arxiv.org/pdf/2511.22606v1](https://arxiv.org/pdf/2511.22606v1)**

> **作者:** Rowzatul Zannath Prerona
>
> **摘要:** Brain metastasis segmentation in MRI remains a formidable challenge due to diminutive lesion sizes (5-15 mm) and extreme class imbalance (less than 2% tumor volume). While soft-attention CNNs are widely used, we identify a critical failure mode termed the "over-segmentation paradox," where models achieve high sensitivity (recall > 0.88) but suffer from catastrophic precision collapse (precision < 0.23) and boundary errors exceeding 150 mm. This imprecision poses significant risks for stereotactic radiosurgery planning. To address this, we introduce the Spatial Gating Network (SG-Net), a precision-first architecture employing hard spatial gating mechanisms. Unlike traditional soft attention, SG-Net enforces strict feature selection to aggressively suppress background artifacts while preserving tumor features. Validated on the Brain-Mets-Lung-MRI dataset (n=92), SG-Net achieves a Dice Similarity Coefficient of 0.5578 +/- 0.0243 (95% CI: 0.45-0.67), statistically outperforming Attention U-Net (p < 0.001) and ResU-Net (p < 0.001). Most critically, SG-Net demonstrates a threefold improvement in boundary precision, achieving a 95% Hausdorff Distance of 56.13 mm compared to 157.52 mm for Attention U-Net, while maintaining robust recall (0.79) and superior precision (0.52 vs. 0.20). Furthermore, SG-Net requires only 0.67M parameters (8.8x fewer than Attention U-Net), facilitating deployment in resource-constrained environments. These findings establish hard spatial gating as a robust solution for precision-driven lesion detection, directly enhancing radiosurgery accuracy.
>
---
#### [new 216] Distracted Robot: How Visual Clutter Undermine Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究机器人在杂乱场景中操作性能下降的问题，提出一种基于心理物理学的统一杂乱度度量方法。通过仿真与真实环境实验，评估视觉-语言-动作模型在不同杂乱程度下的表现，发现杂乱显著降低成功率（最高达34%），且各模型对杂乱的敏感性不同。研究表明杂乱度是性能退化的有效预测指标，且微调数据虽有帮助，但无法均衡缓解所有负面影响。**

- **链接: [https://arxiv.org/pdf/2511.22780v1](https://arxiv.org/pdf/2511.22780v1)**

> **作者:** Amir Rasouli; Montgomery Alban; Sajjad Pakdamansavoji; Zhiyuan Li; Zhanguang Zhang; Aaron Wu; Xuan Zhao
>
> **备注:** 12 figures, 2 tables
>
> **摘要:** In this work, we propose an evaluation protocol for examining the performance of robotic manipulation policies in cluttered scenes. Contrary to prior works, we approach evaluation from a psychophysical perspective, therefore we use a unified measure of clutter that accounts for environmental factors as well as the distractors quantity, characteristics, and arrangement. Using this measure, we systematically construct evaluation scenarios in both hyper-realistic simulation and real-world and conduct extensive experimentation on manipulation policies, in particular vision-language-action (VLA) models. Our experiments highlight the significant impact of scene clutter, lowering the performance of the policies, by as much as 34% and show that despite achieving similar average performance across the tasks, different VLA policies have unique vulnerabilities and a relatively low agreement on success scenarios. We further show that our clutter measure is an effective indicator of performance degradation and analyze the impact of distractors in terms of their quantity and occluding influence. At the end, we show that finetuning on enhanced data, although effective, does not equally remedy all negative impacts of clutter on performance.
>
---
#### [new 217] Optimizing Multimodal Language Models through Attention-based Interpretability
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对多模态语言模型（MLM）难以解释的问题，提出基于注意力的可解释性方法，通过分析注意力分数识别关键图像对象相关注意力头，并用于参数高效微调（PEFT）。研究聚焦图像描述任务，构建新数据集，验证高影响头微调能以极小参数量显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.23375v1](https://arxiv.org/pdf/2511.23375v1)**

> **作者:** Alexander Sergeev; Evgeny Kotelnikov
>
> **备注:** Accepted for ICAI-2025 conference
>
> **摘要:** Modern large language models become multimodal, analyzing various data formats like text and images. While fine-tuning is effective for adapting these multimodal language models (MLMs) to downstream tasks, full fine-tuning is computationally expensive. Parameter-Efficient Fine-Tuning (PEFT) methods address this by training only a small portion of model weights. However, MLMs are difficult to interpret, making it challenging to identify which components are most effective for training to balance efficiency and performance. We propose an attention-based interpretability method for MLMs by analyzing attention scores relative to image tokens. The core idea is to identify attention heads that focus on image key objects. We utilize this information to select optimal model components for PEFT in multimodal models. Our contributions include a method for identifying attention heads associated with image key objects, its application to PEFT for image captioning, and the creation of a new dataset containing images, key object masks, and their textual descriptions. We conducted experiments on MLMs with 2-3 billion parameters to validate the method's effectiveness. By calculating Head Impact (HI) scores we quantify an attention head's focus on key objects, indicating its significance in image understanding. Our fine-tuning experiments demonstrate that adapting layers with the highest HI scores leads to the most significant shifts in metrics compared to pre-trained, randomly selected, or lowest-HI-score layers. This indicates that fine-tuning a small percentage (around 0.01%) of parameters in these crucial layers can substantially influence image understanding capabilities.
>
---
#### [new 218] RealD$^2$iff: Bridging Real-World Gap in Robot Manipulation via Depth Diffusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中的仿真到现实（sim2real）视觉差距问题，提出RealD²iff框架。通过逆向的“纯净到噪声”深度扩散方法，模拟真实传感器噪声，构建无标注的成对数据，并实现零样本仿真实现真实世界操作，显著提升泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.22505v1](https://arxiv.org/pdf/2511.22505v1)**

> **作者:** Xiujian Liang; Jiacheng Liu; Mingyang Sun; Qichen He; Cewu Lu; Jianhua Sun
>
> **摘要:** Robot manipulation in the real world is fundamentally constrained by the visual sim2real gap, where depth observations collected in simulation fail to reflect the complex noise patterns inherent to real sensors. In this work, inspired by the denoising capability of diffusion models, we invert the conventional perspective and propose a clean-to-noisy paradigm that learns to synthesize noisy depth, thereby bridging the visual sim2real gap through purely simulation-driven robotic learning. Building on this idea, we introduce RealD$^2$iff, a hierarchical coarse-to-fine diffusion framework that decomposes depth noise into global structural distortions and fine-grained local perturbations. To enable progressive learning of these components, we further develop two complementary strategies: Frequency-Guided Supervision (FGS) for global structure modeling and Discrepancy-Guided Optimization (DGO) for localized refinement. To integrate RealD$^2$iff seamlessly into imitation learning, we construct a pipeline that spans six stages. We provide comprehensive empirical and experimental validation demonstrating the effectiveness of this paradigm. RealD$^2$iff enables two key applications: (1) generating real-world-like depth to construct clean-noisy paired datasets without manual sensor data collection. (2) Achieving zero-shot sim2real robot manipulation, substantially improving real-world performance without additional fine-tuning.
>
---
#### [new 219] FIGROTD: A Friendly-to-Handle Dataset for Image Guided Retrieval with Optional Text
- **分类: cs.IR; cs.CV**

- **简介: 该论文针对图像引导检索中图文可选任务（IGROT）缺乏高效基准的问题，提出轻量级高质量数据集FIGROTD。通过方差引导特征掩码（VaGFeM）与双损失设计，提升模型在视觉与组合查询间的平衡性能，在多个基准上取得优异效果。**

- **链接: [https://arxiv.org/pdf/2511.22247v1](https://arxiv.org/pdf/2511.22247v1)**

> **作者:** Hoang-Bao Le; Allie Tran; Binh T. Nguyen; Liting Zhou; Cathal Gurrin
>
> **备注:** Accepted at MMM 2026
>
> **摘要:** Image-Guided Retrieval with Optional Text (IGROT) unifies visual retrieval (without text) and composed retrieval (with text). Despite its relevance in applications like Google Image and Bing, progress has been limited by the lack of an accessible benchmark and methods that balance performance across subtasks. Large-scale datasets such as MagicLens are comprehensive but computationally prohibitive, while existing models often favor either visual or compositional queries. We introduce FIGROTD, a lightweight yet high-quality IGROT dataset with 16,474 training triplets and 1,262 test triplets across CIR, SBIR, and CSTBIR. To reduce redundancy, we propose the Variance Guided Feature Mask (VaGFeM), which selectively enhances discriminative dimensions based on variance statistics. We further adopt a dual-loss design (InfoNCE + Triplet) to improve compositional reasoning. Trained on FIGROTD, VaGFeM achieves competitive results on nine benchmarks, reaching 34.8 mAP@10 on CIRCO and 75.7 mAP@200 on Sketchy, outperforming stronger baselines despite fewer triplets.
>
---
#### [new 220] When Do Domain-Specific Foundation Models Justify Their Cost? A Systematic Evaluation Across Retinal Imaging Tasks
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究视网膜影像分类任务，旨在评估领域特定大模型是否值得其高昂成本。通过对比12-13种模型在4项任务上的表现，发现小型通用模型在多数情况下性能接近甚至超过大型专用模型，仅在复杂、不平衡的糖尿病视网膜病变分级中，专用模型具优势。**

- **链接: [https://arxiv.org/pdf/2511.22001v1](https://arxiv.org/pdf/2511.22001v1)**

> **作者:** David Isztl; Tahm Spitznagel; Gabor Mark Somfai; Rui Santos
>
> **摘要:** Large vision foundation models have been widely adopted for retinal disease classification without systematic evidence justifying their parameter requirements. In the present work we address two critical questions: First, are large domain-specific foundation models essential, or do compact general-purpose architectures suffice? Second, does specialized retinal pretraining justify its computational cost? To answer this, we benchmark initialization strategies across four retinal imaging classification tasks spanning Optical Coherence Tomography (OCT) and Color Fundus Photography (CFP) modalities: 8-class OCT classification, 3-class diabetic macular edema (DME), 5-class diabetic retinopathy (DR), and 3-class glaucoma (GL) detection. We evaluate 12-13 model configurations per task, including vision transformers (22.8M-86.6M parameters), Swin Transformers (27.6M-28.3M), ConvNeXt (28.6M), and the domain-specific RETFound models (303M), under identical training conditions. Our results challenge prevailing assumptions: First, we demonstrate that pretraining provides universal benefits (5.18-18.41% improvement), scaling with task difficulty. Second, compact architectures (27-29M) dominate Pareto frontiers; SwinV2-tiny achieves top-1 performance on three datasets. Third, RETFound (303M) justifies its computational cost only for challenging DR grading (accuracy of 71.15%), while ImageNet pretraining proves to be sufficient with all other tasks (DME accuracy: 99.24%, OCT accuracy: 97.96%). CFP tasks show larger pretraining accuracy gains (9.13-18.41%) than OCT (5.18%). Thus, the evidence suggests that compact general-purpose models deliver near-optimal performance for most retinal classification tasks; specialized foundation models warranted only for fine-grained discrimination under extreme class imbalance.
>
---
#### [new 221] LAYER: A Quantitative Explainable AI Framework for Decoding Tissue-Layer Drivers of Myofascial Low Back Pain
- **分类: eess.IV; cs.AI; cs.CV; q-bio.TO**

- **简介: 该论文提出LAYER框架，解决肌筋膜腰痛组织层驱动因素不明确、缺乏影像生物标志物的问题。通过分析4000+例3D超声数据，量化六层软组织对疼痛的贡献，发现非肌肉组织（如深筋膜）在疼痛预测中作用显著，挑战传统肌肉中心范式，建立可解释的定量分析方法。**

- **链接: [https://arxiv.org/pdf/2511.21767v1](https://arxiv.org/pdf/2511.21767v1)**

> **作者:** Zixue Zeng; Anthony M. Perti; Tong Yu; Grant Kokenberger; Hao-En Lu; Jing Wang; Xin Meng; Zhiyu Sheng; Maryam Satarpour; John M. Cormack; Allison C. Bean; Ryan P. Nussbaum; Emily Landis-Walkenhorst; Kang Kim; Ajay D. Wasan; Jiantao Pu
>
> **摘要:** Myofascial pain (MP) is a leading cause of chronic low back pain, yet its tissue-level drivers remain poorly defined and lack reliable image biomarkers. Existing studies focus predominantly on muscle while neglecting fascia, fat, and other soft tissues that play integral biomechanical roles. We developed an anatomically grounded explainable artificial intelligence (AI) framework, LAYER (Layer-wise Analysis for Yielding Explainable Relevance Tissue), that analyses six tissue layers in three-dimensional (3D) ultrasound and quantifies their contribution to MP prediction. By utilizing the largest multi-model 3D ultrasound cohort consisting of over 4,000 scans, LAYER reveals that non-muscle tissues contribute substantially to pain prediction. In B-mode imaging, the deep fascial membrane (DFM) showed the highest saliency (0.420), while in combined B-mode and shear-wave images, the collective saliency of non-muscle layers (0.316) nearly matches that of muscle (0.317), challenging the conventional muscle-centric paradigm in MP research and potentially affecting the therapy methods. LAYER establishes a quantitative, interpretable framework for linking layer-specific anatomy to pain physiology, uncovering new tissue targets and providing a generalizable approach for explainable analysis of soft-tissue imaging.
>
---
#### [new 222] TWEO: Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对大模型FP8训练中极端激活值溢出问题，提出非侵入式损失函数TWEO。通过揭示异常值源于权重矩阵共线性而非数据特性，以简单损失项有效抑制异常值，实现无需复杂工程的全模型FP8训练与量化，显著提升训练效率并达成SOTA量化性能。**

- **链接: [https://arxiv.org/pdf/2511.23225v1](https://arxiv.org/pdf/2511.23225v1)**

> **作者:** Guang Liang; Jie Shao; Ningyuan Tang; Xinyao Liu; Jianxin Wu
>
> **摘要:** Native FP8 support in modern hardware is essential for training large Transformers, but is severely hindered by extreme activation outliers. Existing solutions either rely on complex mixed-precision engineering or invasive architectural modifications. This paper fundamentally challenges the conventional wisdom that outliers are data-driven. We demonstrate that extreme outliers are a data-independent, mechanically-produced artifact of training, originating from specific structural properties of the weight matrices (i.e., colinearity). Based on this insight, we propose TWEO (Transformers Without Extreme Outliers), a novel, non-invasive loss function. TWEO effectively prevents extreme outliers via a very simple loss term, which reduces outliers from 10000+ to less than 20. TWEO then enables full-model FP8 pre-training with neither engineering tricks nor architectural changes for both LLM and ViT. When standard FP8 training catastrophically collapses, TWEO achieves performance comparable to the BF16 baseline while delivering a 36% increase in training throughput. Also, TWEO enables a new quantization paradigm. Hardware-friendly W8A8 per-tensor static quantization of LLMs, previously considered completely unusable due to outliers, achieves SOTA performance for the first time on TWEO-trained models.
>
---
#### [new 223] CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对多模态模型在真实场景中识别与解决视觉与文本矛盾的能力不足问题，提出CrossCheck-Bench基准。通过构建15k含合成矛盾的多层级任务，评估模型跨模态推理能力，发现现有模型在复杂推理上表现薄弱，揭示符号推理与视觉处理融合的重要性。**

- **链接: [https://arxiv.org/pdf/2511.21717v1](https://arxiv.org/pdf/2511.21717v1)**

> **作者:** Baoliang Tian; Yuxuan Si; Jilong Wang; Lingyao Li; Zhongyuan Bao; Zineng Zhou; Tao Wang; Sixu Li; Ziyao Xu; Mingze Wang; Zhouzhuo Zhang; Zhihao Wang; Yike Yun; Ke Tian; Ning Yang; Minghui Qiu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Large Language Models are primarily trained and evaluated on aligned image-text pairs, which leaves their ability to detect and resolve real-world inconsistencies largely unexplored. In open-domain applications visual and textual cues often conflict, requiring models to perform structured reasoning beyond surface-level alignment. We introduce CrossCheck-Bench, a diagnostic benchmark for evaluating contradiction detection in multimodal inputs. The benchmark adopts a hierarchical task framework covering three levels of reasoning complexity and defines seven atomic capabilities essential for resolving cross-modal inconsistencies. CrossCheck-Bench includes 15k question-answer pairs sourced from real-world artifacts with synthetically injected contradictions. The dataset is constructed through a multi-stage annotation pipeline involving more than 450 expert hours to ensure semantic validity and calibrated difficulty across perception, integration, and reasoning. We evaluate 13 state-of-the-art vision-language models and observe a consistent performance drop as tasks shift from perceptual matching to logical contradiction detection. Most models perform well on isolated entity recognition but fail when multiple clues must be synthesized for conflict reasoning. Capability-level analysis further reveals uneven skill acquisition, especially in tasks requiring multi-step inference or rule-based validation. Additional probing shows that conventional prompting strategies such as Chain-of-Thought and Set-of-Mark yield only marginal gains. By contrast, methods that interleave symbolic reasoning with grounded visual processing achieve more stable improvements. These results highlight a persistent bottleneck in multimodal reasoning and suggest new directions for building models capable of robust cross-modal verification.
>
---
#### [new 224] Visual Puns from Idioms: An Iterative LLM-T2IM-MLLM Framework
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究成语视觉双关图生成与理解任务，旨在自动创建既符合字面又体现隐喻意义的图像。提出迭代框架，协同LLM、T2IM和MLLM，通过循环优化提示词生成高质量视觉双关图，并构建了1000个图文数据集用于评估。实验表明MLLM性能主导，GPT表现最佳，Claude在提示生成上最优。**

- **链接: [https://arxiv.org/pdf/2511.22943v1](https://arxiv.org/pdf/2511.22943v1)**

> **作者:** Kelaiti Xiao; Liang Yang; Dongyu Zhang; Paerhati Tulajiang; Hongfei Lin
>
> **备注:** Submitted to ICASSP 2026 (under review)
>
> **摘要:** We study idiom-based visual puns--images that align an idiom's literal and figurative meanings--and present an iterative framework that coordinates a large language model (LLM), a text-to-image model (T2IM), and a multimodal LLM (MLLM) for automatic generation and evaluation. Given an idiom, the system iteratively (i) generates detailed visual prompts, (ii) synthesizes an image, (iii) infers the idiom from the image, and (iv) refines the prompt until recognition succeeds or a step limit is reached. Using 1,000 idioms as inputs, we synthesize a corresponding dataset of visual pun images with paired prompts, enabling benchmarking of both generation and understanding. Experiments across 10 LLMs, 10 MLLMs, and one T2IM (Qwen-Image) show that MLLM choice is the primary performance driver: GPT achieves the highest accuracies, Gemini follows, and the best open-source MLLM (Gemma) is competitive with some closed models. On the LLM side, Claude attains the strongest average performance for prompt generation.
>
---
#### [new 225] $\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型泛化能力弱、动作控制粗略的问题，提出E0框架。通过连续化离散扩散机制，实现细粒度动作生成与更强语义控制，提升跨场景、视角的鲁棒性与精度，在多个基准上达到最优性能。**

- **链接: [https://arxiv.org/pdf/2511.21542v1](https://arxiv.org/pdf/2511.21542v1)**

> **作者:** Zhihao Zhan; Jiaying Zhou; Likui Zhang; Qinhan Lv; Hao Liu; Jusheng Zhang; Weizheng Li; Ziliang Chen; Tianshui Chen; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.
>
---
#### [new 226] Digital Elevation Model Estimation from RGB Satellite Imagery using Generative Deep Learning
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于遥感图像生成任务，旨在利用免费RGB卫星影像生成数字高程模型（DEM），以解决资源受限地区缺乏高精度DEM的问题。研究构建了12K对RGB-DEM数据集，采用条件GAN模型并结合两阶段训练与SSIM筛选，提升复杂地形建模效果，实现低成本、可推广的DEM生成方法。**

- **链接: [https://arxiv.org/pdf/2511.21985v1](https://arxiv.org/pdf/2511.21985v1)**

> **作者:** Alif Ilham Madani; Riska A. Kuswati; Alex M. Lechner; Muhamad Risqi U. Saputra
>
> **备注:** 5 pages, 4 figures, accepted at IGARSS 2025 conference
>
> **摘要:** Digital Elevation Models (DEMs) are vital datasets for geospatial applications such as hydrological modeling and environmental monitoring. However, conventional methods to generate DEM, such as using LiDAR and photogrammetry, require specific types of data that are often inaccessible in resource-constrained settings. To alleviate this problem, this study proposes an approach to generate DEM from freely available RGB satellite imagery using generative deep learning, particularly based on a conditional Generative Adversarial Network (GAN). We first developed a global dataset consisting of 12K RGB-DEM pairs using Landsat satellite imagery and NASA's SRTM digital elevation data, both from the year 2000. A unique preprocessing pipeline was implemented to select high-quality, cloud-free regions and aggregate normalized RGB composites from Landsat imagery. Additionally, the model was trained in a two-stage process, where it was first trained on the complete dataset and then fine-tuned on high-quality samples filtered by Structural Similarity Index Measure (SSIM) values to improve performance on challenging terrains. The results demonstrate promising performance in mountainous regions, achieving an overall mean root-mean-square error (RMSE) of 0.4671 and a mean SSIM score of 0.2065 (scale -1 to 1), while highlighting limitations in lowland and residential areas. This study underscores the importance of meticulous preprocessing and iterative refinement in generative modeling for DEM generation, offering a cost-effective and adaptive alternative to conventional methods while emphasizing the challenge of generalization across diverse terrains worldwide.
>
---
#### [new 227] GEO-Detective: Unveiling Location Privacy Risks in Images with LLM Agents
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对社交媒体图像中的位置隐私泄露问题，提出Geo-Detective系统。通过模拟人类推理与工具使用，实现多步骤、自适应的图像地理定位。相比基线模型，在国家及更细粒度层级上显著提升定位准确率，尤其在缺乏明显地理特征图像中表现优异，并揭示了现有防御策略的不足，强调需加强隐私保护。**

- **链接: [https://arxiv.org/pdf/2511.22441v1](https://arxiv.org/pdf/2511.22441v1)**

> **作者:** Xinyu Zhang; Yixin Wu; Boyang Zhang; Chenhao Lin; Chao Shen; Michael Backes; Yang Zhang
>
> **备注:** 15 pages with 7 figures and 12 tables
>
> **摘要:** Images shared on social media often expose geographic cues. While early geolocation methods required expert effort and lacked generalization, the rise of Large Vision Language Models (LVLMs) now enables accurate geolocation even for ordinary users. However, existing approaches are not optimized for this task. To explore the full potential and associated privacy risks, we present Geo-Detective, an agent that mimics human reasoning and tool use for image geolocation inference. It follows a procedure with four steps that adaptively selects strategies based on image difficulty and is equipped with specialized tools such as visual reverse search, which emulates how humans gather external geographic clues. Experimental results show that GEO-Detective outperforms baseline large vision language models (LVLMs) overall, particularly on images lacking visible geographic features. In country level geolocation tasks, it achieves an improvement of over 11.1% compared to baseline LLMs, and even at finer grained levels, it still provides around a 5.2% performance gain. Meanwhile, when equipped with external clues, GEO-Detective becomes more likely to produce accurate predictions, reducing the "unknown" prediction rate by more than 50.6%. We further explore multiple defense strategies and find that Geo-Detective exhibits stronger robustness, highlighting the need for more effective privacy safeguards.
>
---
#### [new 228] ColonAdapter: Geometry Estimation Through Foundation Model Adaptation for Colonoscopy
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对结肠镜影像中3D几何估计难题，提出ColonAdapter框架，通过自监督微调适配几何基础模型。针对非朗伯表面、光源移动和纹理缺失问题，引入细节恢复模块与一致性损失，提升低纹理区域精度与尺度一致性，实现高精度相机位姿、深度图与点云重建。**

- **链接: [https://arxiv.org/pdf/2511.22250v1](https://arxiv.org/pdf/2511.22250v1)**

> **作者:** Zhiyi Jiang; Yifu Wang; Xuelian Cheng; Zongyuan Ge
>
> **摘要:** Estimating 3D geometry from monocular colonoscopy images is challenging due to non-Lambertian surfaces, moving light sources, and large textureless regions. While recent 3D geometric foundation models eliminate the need for multi-stage pipelines, their performance deteriorates in clinical scenes. These models are primarily trained on natural scene datasets and struggle with specularity and homogeneous textures typical in colonoscopy, leading to inaccurate geometry estimation. In this paper, we present ColonAdapter, a self-supervised fine-tuning framework that adapts geometric foundation models for colonoscopy geometry estimation. Our method leverages pretrained geometric priors while tailoring them to clinical data. To improve performance in low-texture regions and ensure scale consistency, we introduce a Detail Restoration Module (DRM) and a geometry consistency loss. Furthermore, a confidence-weighted photometric loss enhances training stability in clinical environments. Experiments on both synthetic and real datasets demonstrate that our approach achieves state-of-the-art performance in camera pose estimation, monocular depth prediction, and dense 3D point map reconstruction, without requiring ground-truth intrinsic parameters.
>
---
#### [new 229] Evaluating Strategies for Synthesizing Clinical Notes for Medical Multimodal AI
- **分类: cs.AI; cs.CV**

- **简介: 该论文研究医学多模态AI中临床文本合成策略，针对皮肤病学数据中文本信息匮乏问题，探索基于LLM的文本生成方法，通过优化提示设计与医学元数据融合，提升分类与跨模态检索性能，缓解数据稀缺挑战。**

- **链接: [https://arxiv.org/pdf/2511.21827v1](https://arxiv.org/pdf/2511.21827v1)**

> **作者:** Niccolo Marini; Zhaohui Liang; Sivaramakrishnan Rajaraman; Zhiyun Xue; Sameer Antani
>
> **摘要:** Multimodal (MM) learning is emerging as a promising paradigm in biomedical artificial intelligence (AI) applications, integrating complementary modality, which highlight different aspects of patient health. The scarcity of large heterogeneous biomedical MM data has restrained the development of robust models for medical AI applications. In the dermatology domain, for instance, skin lesion datasets typically include only images linked to minimal metadata describing the condition, thereby limiting the benefits of MM data integration for reliable and generalizable predictions. Recent advances in Large Language Models (LLMs) enable the synthesis of textual description of image findings, potentially allowing the combination of image and text representations. However, LLMs are not specifically trained for use in the medical domain, and their naive inclusion has raised concerns about the risk of hallucinations in clinically relevant contexts. This work investigates strategies for generating synthetic textual clinical notes, in terms of prompt design and medical metadata inclusion, and evaluates their impact on MM architectures toward enhancing performance in classification and cross-modal retrieval tasks. Experiments across several heterogeneous dermatology datasets demonstrate that synthetic clinical notes not only enhance classification performance, particularly under domain shift, but also unlock cross-modal retrieval capabilities, a downstream task that is not explicitly optimized during training.
>
---
#### [new 230] Structure-Preserving Unpaired Image Translation to Photometrically Calibrate JunoCam with Hubble Data
- **分类: astro-ph.IM; astro-ph.EP; cs.CV**

- **简介: 该论文针对朱诺号相机（JunoCam）缺乏绝对光度校准的问题，提出结构保持的无配对图像翻译方法（SP-I2I），利用哈勃望远镜数据作为校准参考，实现跨分辨率的图像转换，保留精细空间结构，提升大气动态研究的定量分析能力。**

- **链接: [https://arxiv.org/pdf/2511.22668v1](https://arxiv.org/pdf/2511.22668v1)**

> **作者:** Aditya Pratap Singh; Shrey Shah; Ramanakumar Sankar; Emma Dahl; Gerald Eichstädt; Georgios Georgakis; Bernadette Bucher
>
> **摘要:** Insights into Jupiter's atmospheric dynamics are vital for understanding planetary meteorology and exoplanetary gas giant atmospheres. To study these dynamics, we require high-resolution, photometrically calibrated observations. Over the last 9 years, the Juno spacecraft's optical camera, JunoCam, has generated a unique dataset with high spatial resolution, wide coverage during perijove passes, and a long baseline. However, JunoCam lacks absolute photometric calibration, hindering quantitative analysis of the Jovian atmosphere. Using observations from the Hubble Space Telescope (HST) as a proxy for a calibrated sensor, we present a novel method for performing unpaired image-to-image translation (I2I) between JunoCam and HST, focusing on addressing the resolution discrepancy between the two sensors. Our structure-preserving I2I method, SP-I2I, incorporates explicit frequency-space constraints designed to preserve high-frequency features ensuring the retention of fine, small-scale spatial structures - essential for studying Jupiter's atmosphere. We demonstrate that state-of-the-art unpaired image-to-image translation methods are inadequate to address this problem, and, importantly, we show the broader impact of our proposed solution on relevant remote sensing data for the pansharpening task.
>
---
#### [new 231] Closing the Performance Gap Between AI and Radiologists in Chest X-Ray Reporting
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对胸片报告中临床发现与导管/线路（L&T）信息报告效率低的问题，提出MAIRA-X多模态AI模型，实现纵向胸片报告自动生成。基于大规模数据训练，显著提升报告的语义质量、临床准确性和L&T报告精度，并通过用户评估验证其与放射科医生报告相当的可靠性，有效缓解工作负荷。**

- **链接: [https://arxiv.org/pdf/2511.21735v1](https://arxiv.org/pdf/2511.21735v1)**

> **作者:** Harshita Sharma; Maxwell C. Reynolds; Valentina Salvatelli; Anne-Marie G. Sykes; Kelly K. Horst; Anton Schwaighofer; Maximilian Ilse; Olesya Melnichenko; Sam Bond-Taylor; Fernando Pérez-García; Vamshi K. Mugu; Alex Chan; Ceylan Colak; Shelby A. Swartz; Motassem B. Nashawaty; Austin J. Gonzalez; Heather A. Ouellette; Selnur B. Erdal; Beth A. Schueler; Maria T. Wetscherek; Noel Codella; Mohit Jain; Shruthi Bannur; Kenza Bouzid; Daniel C. Castro; Stephanie Hyland; Panos Korfiatis; Ashish Khandelwal; Javier Alvarez-Valle
>
> **摘要:** AI-assisted report generation offers the opportunity to reduce radiologists' workload stemming from expanded screening guidelines, complex cases and workforce shortages, while maintaining diagnostic accuracy. In addition to describing pathological findings in chest X-ray reports, interpreting lines and tubes (L&T) is demanding and repetitive for radiologists, especially with high patient volumes. We introduce MAIRA-X, a clinically evaluated multimodal AI model for longitudinal chest X-ray (CXR) report generation, that encompasses both clinical findings and L&T reporting. Developed using a large-scale, multi-site, longitudinal dataset of 3.1 million studies (comprising 6 million images from 806k patients) from Mayo Clinic, MAIRA-X was evaluated on three holdout datasets and the public MIMIC-CXR dataset, where it significantly improved AI-generated reports over the state of the art on lexical quality, clinical correctness, and L&T-related elements. A novel L&T-specific metrics framework was developed to assess accuracy in reporting attributes such as type, longitudinal change and placement. A first-of-its-kind retrospective user evaluation study was conducted with nine radiologists of varying experience, who blindly reviewed 600 studies from distinct subjects. The user study found comparable rates of critical errors (3.0% for original vs. 4.6% for AI-generated reports) and a similar rate of acceptable sentences (97.8% for original vs. 97.4% for AI-generated reports), marking a significant improvement over prior user studies with larger gaps and higher error rates. Our results suggest that MAIRA-X can effectively assist radiologists, particularly in high-volume clinical settings.
>
---
#### [new 232] GACELLE: GPU-accelerated tools for model parameter estimation and image reconstruction
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文针对定量MRI（qMRI）参数估计计算耗时的问题，提出GACELLE框架。该框架基于GPU加速，实现快速参数映射与图像重建，支持不确定性量化与空间正则化，显著提升效率与精度，推动qMRI在临床研究中的应用。**

- **链接: [https://arxiv.org/pdf/2511.22094v1](https://arxiv.org/pdf/2511.22094v1)**

> **作者:** Kwok-Shing Chan; Hansol Lee; Yixin Ma; Berkin Bilgic; Susie Y. Huang; Hong-Hsi Lee; José P. Marques
>
> **摘要:** Quantitative MRI (qMRI) offers tissue-specific biomarkers that can be tracked over time or compared across populations; however, its adoption in clinical research is hindered by significant computational demands of parameter estimation. Images acquired at high spatial resolution or requiring fitting multiple parameters often require lengthy processing time, constraining their use in routine pipelines and slowing methodological innovation and clinical translation. We present GACELLE, an open source, GPU-accelerated framework for high-throughput qMRI analysis. GACELLE provides a stochastic gradient descent optimiser and a stochastic sampler in MATLAB, enabling fast parameter mapping, improved estimation robustness via spatial regularisation, and uncertainty quantification. GACELLE prioritises accessibility: users only need to provide a forward signal model, while GACELLE's backend manages computational parallelisation, automatic parameter updates, and memory-batching. The stochastic solver performs fully vectorised Markov chain Monte Carlo with identical likelihood on CPU and GPU, ensuring reproducibility across hardware. Benchmarking demonstrates up to 451-fold acceleration for the stochastic gradient descent solver and 14,380-fold acceleration for stochastic sampling compared to CPU-based estimation, without compromising accuracy. We demonstrated GACELLE's versatility on three representative qMRI models and on an image reconstruction task. Across these applications, GACELLE improves parameter precision, enhances test-retest reproducibility, and reduces noise in quantitative maps. By combining speed, usability and flexibility, GACELLE provides a generalisable optimisation framework for medical image analysis. It lowers the computational barrier for qMRI, paving the way for reproducible biomarker development, large-scale imaging studies, and clinical translation.
>
---
#### [new 233] Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人任务中因物理差异需精细调优的问题，提出“机器人引导”方法。通过少量示范识别任务特定注意力头，实现精准、高效、可解释的微调，显著提升模型在真实机器人上的适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22697v1](https://arxiv.org/pdf/2511.22697v1)**

> **作者:** Chancharik Mitra; Yusen Luo; Raj Saravanan; Dantong Niu; Anirudh Pai; Jesse Thomason; Trevor Darrell; Abrar Anwar; Deva Ramanan; Roei Herzig
>
> **摘要:** Vision-Language Action (VLAs) models promise to extend the remarkable success of vision-language models (VLMs) to robotics. Yet, unlike VLMs in the vision-language domain, VLAs for robotics require finetuning to contend with varying physical factors like robot embodiment, environment characteristics, and spatial relationships of each task. Existing fine-tuning methods lack specificity, adapting the same set of parameters regardless of a task's visual, linguistic, and physical characteristics. Inspired by functional specificity in neuroscience, we hypothesize that it is more effective to finetune sparse model representations specific to a given task. In this work, we introduce Robotic Steering, a finetuning approach grounded in mechanistic interpretability that leverages few-shot demonstrations to identify and selectively finetune task-specific attention heads aligned with the physical, visual, and linguistic requirements of robotic tasks. Through comprehensive on-robot evaluations with a Franka Emika robot arm, we demonstrate that Robotic Steering outperforms LoRA while achieving superior robustness under task variation, reduced computational cost, and enhanced interpretability for adapting VLAs to diverse robotic tasks.
>
---
#### [new 234] Designing Instance-Level Sampling Schedules via REINFORCE with James-Stein Shrinkage
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对文本生成图像中采样器的效率与对齐问题，提出基于强化学习与James-Stein收缩的实例级采样调度方法。通过学习提示和噪声条件下的动态采样路径，提升生成质量与控制能力，实现模型无关的后训练优化。**

- **链接: [https://arxiv.org/pdf/2511.22177v1](https://arxiv.org/pdf/2511.22177v1)**

> **作者:** Peiyu Yu; Suraj Kothawade; Sirui Xie; Ying Nian Wu; Hongliang Fei
>
> **备注:** 23 pages
>
> **摘要:** Most post-training methods for text-to-image samplers focus on model weights: either fine-tuning the backbone for alignment or distilling it for few-step efficiency. We take a different route: rescheduling the sampling timeline of a frozen sampler. Instead of a fixed, global schedule, we learn instance-level (prompt- and noise-conditioned) schedules through a single-pass Dirichlet policy. To ensure accurate gradient estimates in high-dimensional policy learning, we introduce a novel reward baseline based on a principled James-Stein estimator; it provably achieves lower estimation errors than commonly used variants and leads to superior performance. Our rescheduled samplers consistently improve text-image alignment including text rendering and compositional control across modern Stable Diffusion and Flux model families. Additionally, a 5-step Flux-Dev sampler with our schedules can attain generation quality comparable to deliberately distilled samplers like Flux-Schnell. We thus position our scheduling framework as an emerging model-agnostic post-training lever that unlocks additional generative potential in pretrained samplers.
>
---
#### [new 235] Adversarial Flow Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出对抗流模型（Adversarial Flow Models），融合对抗训练与流模型优势，解决生成模型训练不稳定与多步生成效率低的问题。通过学习确定性映射实现一步生成，提升稳定性与效率，在ImageNet上达到2.38最优FID，支持深度模型端到端训练。**

- **链接: [https://arxiv.org/pdf/2511.22475v1](https://arxiv.org/pdf/2511.22475v1)**

> **作者:** Shanchuan Lin; Ceyuan Yang; Zhijie Lin; Hao Chen; Haoqi Fan
>
> **摘要:** We present adversarial flow models, a class of generative models that unifies adversarial models and flow models. Our method supports native one-step or multi-step generation and is trained using the adversarial objective. Unlike traditional GANs, where the generator learns an arbitrary transport plan between the noise and the data distributions, our generator learns a deterministic noise-to-data mapping, which is the same optimal transport as in flow-matching models. This significantly stabilizes adversarial training. Also, unlike consistency-based methods, our model directly learns one-step or few-step generation without needing to learn the intermediate timesteps of the probability flow for propagation. This saves model capacity, reduces training iterations, and avoids error accumulation. Under the same 1NFE setting on ImageNet-256px, our B/2 model approaches the performance of consistency-based XL/2 models, while our XL/2 model creates a new best FID of 2.38. We additionally show the possibility of end-to-end training of 56-layer and 112-layer models through depth repetition without any intermediate supervision, and achieve FIDs of 2.08 and 1.94 using a single forward pass, surpassing their 2NFE and 4NFE counterparts.
>
---
#### [new 236] MICCAI STS 2024 Challenge: Semi-Supervised Instance-Level Tooth Segmentation in Panoramic X-ray and CBCT Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文聚焦于口腔医学图像中牙齿的实例级分割任务，针对标注数据稀缺问题，组织了MICCAI STS 2024挑战赛。基于超9万张2D/3D影像构建大规模数据集，通过半监督学习（SSL）方法显著提升分割性能，最优模型在OPG和CBCT任务上分别提升44和61个百分点。公开数据与代码，推动领域可复现研究。**

- **链接: [https://arxiv.org/pdf/2511.22911v1](https://arxiv.org/pdf/2511.22911v1)**

> **作者:** Yaqi Wang; Zhi Li; Chengyu Wu; Jun Liu; Yifan Zhang; Jiaxue Ni; Qian Luo; Jialuo Chen; Hongyuan Zhang; Jin Liu; Can Han; Kaiwen Fu; Changkai Ji; Xinxu Cai; Jing Hao; Zhihao Zheng; Shi Xu; Junqiang Chen; Qianni Zhang; Dahong Qian; Shuai Wang; Huiyu Zhou
>
> **摘要:** Orthopantomogram (OPGs) and Cone-Beam Computed Tomography (CBCT) are vital for dentistry, but creating large datasets for automated tooth segmentation is hindered by the labor-intensive process of manual instance-level annotation. This research aimed to benchmark and advance semi-supervised learning (SSL) as a solution for this data scarcity problem. We organized the 2nd Semi-supervised Teeth Segmentation (STS 2024) Challenge at MICCAI 2024. We provided a large-scale dataset comprising over 90,000 2D images and 3D axial slices, which includes 2,380 OPG images and 330 CBCT scans, all featuring detailed instance-level FDI annotations on part of the data. The challenge attracted 114 (OPG) and 106 (CBCT) registered teams. To ensure algorithmic excellence and full transparency, we rigorously evaluated the valid, open-source submissions from the top 10 (OPG) and top 5 (CBCT) teams, respectively. All successful submissions were deep learning-based SSL methods. The winning semi-supervised models demonstrated impressive performance gains over a fully-supervised nnU-Net baseline trained only on the labeled data. For the 2D OPG track, the top method improved the Instance Affinity (IA) score by over 44 percentage points. For the 3D CBCT track, the winning approach boosted the Instance Dice score by 61 percentage points. This challenge confirms the substantial benefit of SSL for complex, instance-level medical image segmentation tasks where labeled data is scarce. The most effective approaches consistently leveraged hybrid semi-supervised frameworks that combined knowledge from foundational models like SAM with multi-stage, coarse-to-fine refinement pipelines. Both the challenge dataset and the participants' submitted code have been made publicly available on GitHub (https://github.com/ricoleehduu/STS-Challenge-2024), ensuring transparency and reproducibility.
>
---
#### [new 237] UNION: A Lightweight Target Representation for Efficient Zero-Shot Image-Guided Retrieval with Optional Textual Queries
- **分类: cs.IR; cs.CV**

- **简介: 该论文提出UNION，一种轻量级目标表示方法，用于高效零样本图像引导检索（IGROT）。针对多模态查询下目标特征固定导致语义对齐差的问题，UNION融合图像嵌入与空文本提示，无需修改预训练模型。仅用5000样本即在CIR和SBIR任务上取得优异效果，显著提升检索性能。**

- **链接: [https://arxiv.org/pdf/2511.22253v1](https://arxiv.org/pdf/2511.22253v1)**

> **作者:** Hoang-Bao Le; Allie Tran; Binh T. Nguyen; Liting Zhou; Cathal Gurrin
>
> **备注:** Accepted at ICDM - MMSR Workshop 2025
>
> **摘要:** Image-Guided Retrieval with Optional Text (IGROT) is a general retrieval setting where a query consists of an anchor image, with or without accompanying text, aiming to retrieve semantically relevant target images. This formulation unifies two major tasks: Composed Image Retrieval (CIR) and Sketch-Based Image Retrieval (SBIR). In this work, we address IGROT under low-data supervision by introducing UNION, a lightweight and generalisable target representation that fuses the image embedding with a null-text prompt. Unlike traditional approaches that rely on fixed target features, UNION enhances semantic alignment with multimodal queries while requiring no architectural modifications to pretrained vision-language models. With only 5,000 training samples - from LlavaSCo for CIR and Training-Sketchy for SBIR - our method achieves competitive results across benchmarks, including CIRCO mAP@50 of 38.5 and Sketchy mAP@200 of 82.7, surpassing many heavily supervised baselines. This demonstrates the robustness and efficiency of UNION in bridging vision and language across diverse query types.
>
---
#### [new 238] Comparing SAM 2 and SAM 3 for Zero-Shot Segmentation of 3D Medical Data
- **分类: eess.IV; cs.CV**

- **简介: 该论文比较SAM 2与SAM 3在3D医学数据零样本分割中的表现，旨在评估SAM 3能否作为无需定制的SAM 2替代方案。研究在16个公共数据集上，基于纯视觉提示（点击、框选、掩码）进行对比，发现SAM 3在复杂解剖结构上初始化更优，整体更具通用性，尤其适用于稀疏交互和复杂拓扑场景。**

- **链接: [https://arxiv.org/pdf/2511.21926v1](https://arxiv.org/pdf/2511.21926v1)**

> **作者:** Satrajit Chakrabarty; Ravi Soni
>
> **摘要:** Foundation models for promptable segmentation, including SAM, SAM 2, and the recently released SAM 3, have renewed interest in zero-shot segmentation of medical imaging. Although these models perform strongly on natural images, their behavior on medical data remains insufficiently characterized. While SAM 2 is widely used for annotation in 3D medical workflows, SAM 3 introduces a new perception backbone, detector-tracker pipeline, and concept-level prompting that may alter its behavior under spatial prompts. We present the first controlled comparison of SAM 2 and SAM 3 for zero-shot segmentation of 3D medical volumes and videos under purely visual prompting, with concept mechanisms disabled. We assess whether SAM 3 can serve as an out-of-the-box replacement for SAM 2 without customization. We benchmark both models on 16 public datasets (CT, MRI, 3D and cine ultrasound, endoscopy) covering 54 anatomical structures, pathologies, and surgical instruments. Prompts are restricted to the first frame and use four modes: single-click, multi-click, bounding box, and dense mask. This design standardizes preprocessing, prompt placement, propagation rules, and metric computation to disentangle prompt interpretation from propagation. Prompt-frame analysis shows that SAM 3 provides substantially stronger initialization than SAM 2 for click prompting across most structures. In full-volume analysis, SAM 3 retains this advantage for complex, vascular, and soft-tissue anatomies, emerging as the more versatile general-purpose segmenter. While SAM 2 remains competitive for compact, rigid organs under strong spatial guidance, it frequently fails on challenging targets where SAM 3 succeeds. Overall, our results suggest that SAM 3 is the superior default choice for most medical segmentation tasks, particularly those involving sparse user interaction or complex anatomical topology.
>
---
#### [new 239] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多模态测试时自适应（MMTTA）任务，解决模态间分布偏移与语义错位的耦合问题。提出BriMPR框架，通过渐进式对齐：先用提示调优校准单模态特征分布，再通过对比学习增强跨模态信息交互，实现更优的多模态自适应。**

- **链接: [https://arxiv.org/pdf/2511.22862v1](https://arxiv.org/pdf/2511.22862v1)**

> **作者:** Jiacheng Li; Songhe Feng
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Test-time adaptation (TTA) enables online model adaptation using only unlabeled test data, aiming to bridge the gap between source and target distributions. However, in multimodal scenarios, varying degrees of distribution shift across different modalities give rise to a complex coupling effect of unimodal shallow feature shift and cross-modal high-level semantic misalignment, posing a major obstacle to extending existing TTA methods to the multimodal field. To address this challenge, we propose a novel multimodal test-time adaptation (MMTTA) framework, termed as Bridging Modalities via Progressive Re-alignment (BriMPR). BriMPR, consisting of two progressively enhanced modules, tackles the coupling effect with a divide-and-conquer strategy. Specifically, we first decompose MMTTA into multiple unimodal feature alignment sub-problems. By leveraging the strong function approximation ability of prompt tuning, we calibrate the unimodal global feature distributions to their respective source distributions, so as to achieve the initial semantic re-alignment across modalities. Subsequently, we assign the credible pseudo-labels to combinations of masked and complete modalities, and introduce inter-modal instance-wise contrastive learning to further enhance the information interaction among modalities and refine the alignment. Extensive experiments on MMTTA tasks, including both corruption-based and real-world domain shift benchmarks, demonstrate the superiority of our method. Our source code is available at [this URL](https://github.com/Luchicken/BriMPR).
>
---
#### [new 240] Obstruction reasoning for robotic grasping
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人在杂乱环境中抓取目标物体时的遮挡推理问题，提出UNOGrasp模型。通过视觉-语言多步推理，识别遮挡路径并规划清除顺序，结合监督与强化学习优化决策。构建了包含超10万条遮挡路径的UNOBench数据集，显著提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2511.23186v1](https://arxiv.org/pdf/2511.23186v1)**

> **作者:** Runyu Jiao; Matteo Bortolon; Francesco Giuliari; Alice Fasoli; Sergio Povoli; Guofeng Mei; Yiming Wang; Fabio Poiesi
>
> **摘要:** Successful robotic grasping in cluttered environments not only requires a model to visually ground a target object but also to reason about obstructions that must be cleared beforehand. While current vision-language embodied reasoning models show emergent spatial understanding, they remain limited in terms of obstruction reasoning and accessibility planning. To bridge this gap, we present UNOGrasp, a learning-based vision-language model capable of performing visually-grounded obstruction reasoning to infer the sequence of actions needed to unobstruct the path and grasp the target object. We devise a novel multi-step reasoning process based on obstruction paths originated by the target object. We anchor each reasoning step with obstruction-aware visual cues to incentivize reasoning capability. UNOGrasp combines supervised and reinforcement finetuning through verifiable reasoning rewards. Moreover, we construct UNOBench, a large-scale dataset for both training and benchmarking, based on MetaGraspNetV2, with over 100k obstruction paths annotated by humans with obstruction ratios, contact points, and natural-language instructions. Extensive experiments and real-robot evaluations show that UNOGrasp significantly improves obstruction reasoning and grasp success across both synthetic and real-world environments, outperforming generalist and proprietary alternatives. Project website: https://tev-fbk.github.io/UnoGrasp/.
>
---
#### [new 241] Physics-Informed Neural Networks for Thermophysical Property Retrieval
- **分类: cs.LG; cs.AI; cs.CE; cs.CV**

- **简介: 该论文针对建筑墙体热导率的非侵入式在线反演问题，提出基于物理信息神经网络（PINN）的迭代框架。通过结合实测与仿真温度数据，交替求解正向热传导问题并优化热导率参数，实现高精度、快速估计，有效克服环境扰动与稳态假设偏差的影响，为建筑节能评估提供可靠方法。**

- **链接: [https://arxiv.org/pdf/2511.23449v1](https://arxiv.org/pdf/2511.23449v1)**

> **作者:** Ali Waseem; Malcolm Mielle
>
> **备注:** 26 pages, 4 figures, 3 tables
>
> **摘要:** Inverse heat problems refer to the estimation of material thermophysical properties given observed or known heat diffusion behaviour. Inverse heat problems have wide-ranging uses, but a critical application lies in quantifying how building facade renovation reduces thermal transmittance, a key determinant of building energy efficiency. However, solving inverse heat problems with non-invasive data collected in situ is error-prone due to environmental variability or deviations from theoretically assumed conditions. Hence, current methods for measuring thermal conductivity are either invasive, require lengthy observation periods, or are sensitive to environmental and experimental conditions. Here, we present a PINN-based iterative framework to estimate the thermal conductivity k of a wall from a set of thermographs; our framework alternates between estimating the forward heat problem with a PINN for a fixed k, and optimizing k by comparing the thermographs and surface temperatures predicted by the PINN, repeating until the estimated k's convergence. Using both environmental data captured by a weather station and data generated from Finite-Volume-Method software simulations, we accurately predict k across different environmental conditions and data collection sampling times, given the temperature profile of the wall at dawn is close to steady state. Although violating the steady-state assumption impacts the accuracy of k's estimation, we show that our proposed framework still only exhibits a maximum MAE of 4.0851. Our work demonstrates the potential of PINN-based methods for reliable estimation of material properties in situ and under realistic conditions, without lengthy measurement campaigns. Given the lack of research on using machine learning, and more specifically on PINNs, for solving in-situ inverse problems, we expect our work to be a starting point for more research on the topic.
>
---
#### [new 242] Insight-A: Attribution-aware for Multimodal Misinformation Detection
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对AIGC时代多模态虚假信息检测任务，解决传统方法忽视信息溯源与主观性问题。提出Insight-A框架，通过跨源溯源提示（CAP）和去偏自动提示（ADP）实现伪造源追溯，结合图像描述增强跨模态一致性验证，构建层次化推理管道，提升检测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.21705v1](https://arxiv.org/pdf/2511.21705v1)**

> **作者:** Junjie Wu; Yumeng Fu; Chen Gong; Guohong Fu
>
> **摘要:** AI-generated content (AIGC) technology has emerged as a prevalent alternative to create multimodal misinformation on social media platforms, posing unprecedented threats to societal safety. However, standard prompting leverages multimodal large language models (MLLMs) to identify the emerging misinformation, which ignores the misinformation attribution. To this end, we present Insight-A, exploring attribution with MLLM insights for detecting multimodal misinformation. Insight-A makes two efforts: I) attribute misinformation to forgery sources, and II) an effective pipeline with hierarchical reasoning that detects distortions across modalities. Specifically, to attribute misinformation to forgery traces based on generation patterns, we devise cross-attribution prompting (CAP) to model the sophisticated correlations between perception and reasoning. Meanwhile, to reduce the subjectivity of human-annotated prompts, automatic attribution-debiased prompting (ADP) is used for task adaptation on MLLMs. Additionally, we design image captioning (IC) to achieve visual details for enhancing cross-modal consistency checking. Extensive experiments demonstrate the superiority of our proposal and provide a new paradigm for multimodal misinformation detection in the era of AIGC.
>
---
## 更新

#### [replaced 001] SAMChat: Introducing Chain of Thought Reasoning and GRPO to a Multimodal Small Language Model for Small Scale Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.07984v2](https://arxiv.org/pdf/2505.07984v2)**

> **作者:** Aybora Koksal; A. Aydin Alatan
>
> **备注:** Accepted to Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS) Special Issue on Foundation and Large Vision Models for Remote Sensing. Code and dataset are available at https://github.com/aybora/SAMChat
>
> **摘要:** Remarkable capabilities in understanding and generating text-image content have been demonstrated by recent advancements in multimodal large language models (MLLMs). However, their effectiveness in specialized domains-particularly those requiring resource-efficient and domain-specific adaptations-has remained limited. In this work, a lightweight multimodal language model termed SAMChat is introduced, specifically adapted to analyze remote sensing imagery in secluded areas, including challenging missile launch sites. A new dataset, SAMData, was compiled by verifying hundreds of aerial images through expert review, and subtle military installations were highlighted via detailed captions. Supervised fine-tuning on a 2B parameter open-source MLLM with chain-of-thought (CoT) reasoning annotations was performed, enabling more accurate and interpretable explanations. Additionally, Group Relative Policy Optimization (GRPO) was leveraged to enhance the model's ability to detect critical domain-specific cues-such as defensive layouts and key military structures-while minimizing false positives on civilian scenes. Through empirical evaluations, it has been shown that SAMChat significantly outperforms both larger, general-purpose multimodal models and existing remote sensing adapted approaches on open-ended captioning and classification metrics. Over 80% recall and 98% precision were achieved on the newly proposed SAMData benchmark, underscoring the potency of targeted fine-tuning and reinforcement learning in specialized real-world applications.
>
---
#### [replaced 002] Geometric Regularity in Deterministic Sampling of Diffusion-based Generative Models
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2506.10177v2](https://arxiv.org/pdf/2506.10177v2)**

> **作者:** Defang Chen; Zhenyu Zhou; Can Wang; Siwei Lyu
>
> **备注:** 57 pages. Accepted by Journal of Statistical Mechanics: Theory and Experiment (2025). The short version was published in ICML 2024. arXiv admin note: text overlap with arXiv:2405.11326
>
> **摘要:** Diffusion-based generative models employ stochastic differential equations (SDEs) and their equivalent probability flow ordinary differential equations (ODEs) to establish a smooth transformation between complex high-dimensional data distributions and tractable prior distributions. In this paper, we reveal a striking geometric regularity in the deterministic sampling dynamics of diffusion generative models: each simulated sampling trajectory along the gradient field lies within an extremely low-dimensional subspace, and all trajectories exhibit an almost identical boomerang shape, regardless of the model architecture, applied conditions, or generated content. We characterize several intriguing properties of these trajectories, particularly under closed-form solutions based on kernel-estimated data modeling. We also demonstrate a practical application of the discovered trajectory regularity by proposing a dynamic programming-based scheme to better align the sampling time schedule with the underlying trajectory structure. This simple strategy requires minimal modification to existing deterministic numerical solvers, incurs negligible computational overhead, and achieves superior image generation performance, especially in regions with only 5 - 10 function evaluations.
>
---
#### [replaced 003] CANVAS: A Benchmark for Vision-Language Models on Tool-Based User Interface Design
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CANVAS基准，用于评估视觉语言模型（VLMs）在工具调用下的用户界面（UI）设计能力。针对现有缺乏工具驱动设计评测标准的问题，构建包含598个任务的基准，涵盖设计复现与修改两类任务，通过真实设计软件操作验证模型性能，揭示模型策略与错误模式，推动VLM在设计协作中的发展。**

- **链接: [https://arxiv.org/pdf/2511.20737v2](https://arxiv.org/pdf/2511.20737v2)**

> **作者:** Daeheon Jeong; Seoyeon Byun; Kihoon Son; Dae Hyun Kim; Juho Kim
>
> **摘要:** User interface (UI) design is an iterative process in which designers progressively refine their work with design software such as Figma or Sketch. Recent advances in vision language models (VLMs) with tool invocation suggest these models can operate design software to edit a UI design through iteration. Understanding and enhancing this capacity is important, as it highlights VLMs' potential to collaborate with designers within conventional software. However, as no existing benchmark evaluates tool-based design performance, the capacity remains unknown. To address this, we introduce CANVAS, a benchmark for VLMs on tool-based user interface design. Our benchmark contains 598 tool-based design tasks paired with ground-truth references sampled from 3.3K mobile UI designs across 30 function-based categories (e.g., onboarding, messaging). In each task, a VLM updates the design step-by-step through context-based tool invocations (e.g., create a rectangle as a button background), linked to design software. Specifically, CANVAS incorporates two task types: (i) design replication evaluates the ability to reproduce a whole UI screen; (ii) design modification evaluates the ability to modify a specific part of an existing screen. Results suggest that leading models exhibit more strategic tool invocations, improving design quality. Furthermore, we identify common error patterns models exhibit, guiding future work in enhancing tool-based design capabilities.
>
---
#### [replaced 004] CephRes-MHNet: A Multi-Head Residual Network for Accurate and Robust Cephalometric Landmark Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10173v2](https://arxiv.org/pdf/2511.10173v2)**

> **作者:** Ahmed Jaheen; Islam Hassan; Mohanad Abouserie; Abdelaty Rehab; Adham Elasfar; Knzy Elmasry; Mostafa El-Dawlatly; Seif Eldawlatly
>
> **备注:** This submission was posted without authorization from all co-authors and supervising institutions. The authors are withdrawing the manuscript due to permission issues
>
> **摘要:** Accurate localization of cephalometric landmarks from 2D lateral skull X-rays is vital for orthodontic diagnosis and treatment. Manual annotation is time-consuming and error-prone, whereas automated approaches often struggle with low contrast and anatomical complexity. This paper introduces CephRes-MHNet, a multi-head residual convolutional network for robust and efficient cephalometric landmark detection. The architecture integrates residual encoding, dual-attention mechanisms, and multi-head decoders to enhance contextual reasoning and anatomical precision. Trained on the Aariz Cephalometric dataset of 1,000 radiographs, CephRes-MHNet achieved a mean radial error (MRE) of 1.23 mm and a success detection rate (SDR) @ 2.0 mm of 85.5%, outperforming all evaluated models. In particular, it exceeded the strongest baseline, the attention-driven AFPF-Net (MRE = 1.25 mm, SDR @ 2.0 mm = 84.1%), while using less than 25% of its parameters. These results demonstrate that CephRes-MHNet attains state-of-the-art accuracy through architectural efficiency, providing a practical solution for real-world orthodontic analysis.
>
---
#### [replaced 005] VinciCoder: Unifying Multimodal Code Generation via Coarse-to-fine Visual Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00391v2](https://arxiv.org/pdf/2511.00391v2)**

> **作者:** Xuanle Zhao; Deyang Jiang; Zhixiong Zeng; Lei Chen; Haibo Qiu; Jing Huang; Yufeng Zhong; Liming Zheng; Yilin Cao; Lin Ma
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Multimodal code generation has garnered significant interest within the research community. Despite the notable success of recent vision-language models (VLMs) on specialized tasks like chart-to-code generation, their reliance on single-task training regimens fosters a narrow paradigm that hinders the development of generalized \textbf{VI}sio\textbf{N} \textbf{C}ode \textbf{I}ntelligence. In this work, we introduce \textbf{VinciCoder}, a unified multimodal code generation model that addresses this limitation via a two-stage training framework. We begin by constructing a large-scale Supervised Finetuning (SFT) corpus comprising 1.6M image-code pairs for tasks involving direct code generation and visual-based code refinement. Subsequently, we introduce a Visual Reinforcement Learning (ViRL) strategy, which employs a coarse-to-fine reward mechanism to improve visual fidelity by calculating visual similarity across local and global image patches. Extensive experiments on diverse multimodal code generation benchmarks demonstrate that VinciCoder achieves state-of-the-art performance, surpassing recent open-source models. The ablation study further validates the effectiveness of our proposed coarse-to-fine ViRL strategy. The data, code and model is available at https://github.com/DocTron-hub/VinciCoder.
>
---
#### [replaced 006] Automated segmentation of pediatric neuroblastoma on multi-modal MRI: Results of the SPPIN challenge at MICCAI 2023
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.00369v2](https://arxiv.org/pdf/2505.00369v2)**

> **作者:** M. A. D. Buser; D. C. Simons; M. Fitski; M. H. W. A. Wijnen; A. S. Littooij; A. H. ter Brugge; I. N. Vos; M. H. A. Janse; M. de Boer; R. ter Maat; J. Sato; S. Kido; S. Kondo; S. Kasai; M. Wodzinski; H. Muller; J. Ye; J. He; Y. Kirchhoff; M. R. Rokkus; G. Haokai; S. Zitong; M. Fernández Patón; D. Veiga-Canuto; D. G. Ellis; M. R. Aizenberg; B. H. M. van der Velden; H. Kuijf; A. De Luca; A. F. W. van der Steeg
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Surgery plays an important role within the treatment for neuroblastoma, a common pediatric cancer. This requires careful planning, often via magnetic resonance imaging (MRI)-based anatomical 3D models. However, creating these models is often time-consuming and user dependent. We organized the Surgical Planning in Pediatric Neuroblastoma (SPPIN) challenge, to stimulate developments on this topic, and set a benchmark for fully automatic segmentation of neuroblastoma on multi-model MRI. The challenge started with a training phase, where teams received 78 sets of MRI scans from 34 patients, consisting of both diagnostic and post-chemotherapy MRI scans. The final test phase, consisting of 18 MRI sets from 9 patients, determined the ranking of the teams. Ranking was based on the Dice similarity coefficient (Dice score), the 95th percentile of the Hausdorff distance (HD95) and the volumetric similarity (VS). The SPPIN challenge was hosted at MICCAI 2023. The final leaderboard consisted of 9 teams. The highest-ranking team achieved a median Dice score 0.82, a median HD95 of 7.69 mm and a VS of 0.91, utilizing a large, pretrained network called STU-Net. A significant difference for the segmentation results between diagnostic and post-chemotherapy MRI scans was observed (Dice = 0.89 vs Dice = 0.59, P = 0.01) for the highest-ranking team. SPPIN is the first medical segmentation challenge in extracranial pediatric oncology. The highest-ranking team used a large pre-trained network, suggesting that pretraining can be of use in small, heterogenous datasets. Although the results of the highest-ranking team were high for most patients, segmentation especially in small, pre-treated tumors were insufficient. Therefore, more reliable segmentation methods are needed to create clinically applicable models to aid surgical planning in pediatric neuroblastoma.
>
---
#### [replaced 007] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01345v5](https://arxiv.org/pdf/2508.01345v5)**

> **作者:** Rongzhen Zhao; Jian Li; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Unsupervised video Object-Centric Learning (OCL) is promising as it enables object-level scene representation and understanding as we humans do. Mainstream video OCL methods adopt a recurrent architecture: An aggregator aggregates current video frame into object features, termed slots, under some queries; A transitioner transits current slots to queries for the next frame. This is an effective architecture but all existing implementations both (\textit{i1}) neglect to incorporate next frame features, the most informative source for query prediction, and (\textit{i2}) fail to learn transition dynamics, the knowledge essential for query prediction. To address these issues, we propose Random Slot-Feature pair for learning Query prediction (RandSF.Q): (\textit{t1}) We design a new transitioner to incorporate both slots and features, which provides more information for query prediction; (\textit{t2}) We train the transitioner to predict queries from slot-feature pairs randomly sampled from available recurrences, which drives it to learn transition dynamics. Experiments on scene representation demonstrate that our method surpass existing video OCL methods significantly, e.g., up to 10 points on object discovery, setting new state-of-the-art. Such superiority also benefits downstream tasks like scene understanding. Source Code, Model Checkpoints, Training Logs: https://github.com/Genera1Z/RandSF.Q
>
---
#### [replaced 008] Learning Plug-and-play Memory for Guiding Video Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19229v2](https://arxiv.org/pdf/2511.19229v2)**

> **作者:** Selena Song; Ziming Xu; Zijun Zhang; Kun Zhou; Jiaxian Guo; Lianhui Qin; Biwei Huang
>
> **摘要:** Diffusion Transformer(DiT) based video generation models have recently achieved impressive visual quality and temporal coherence, but they still frequently violate basic physical laws and commonsense dynamics, revealing a lack of explicit world knowledge. In this work, we explore how to equip them with a plug-and-play memory that injects useful world knowledge. Motivated by in-context memory in Transformer-based LLMs, we conduct empirical studies to show that DiT can be steered via interventions on its hidden states, and simple low-pass and high-pass filters in the embedding space naturally disentangle low-level appearance and high-level physical/semantic cues, enabling targeted guidance. Building on these observations, we propose a learnable memory encoder DiT-Mem, composed of stacked 3D CNNs, low-/high-pass filters, and self-attention layers. The encoder maps reference videos into a compact set of memory tokens, which are concatenated as the memory within the DiT self-attention layers. During training, we keep the diffusion backbone frozen, and only optimize the memory encoder. It yields a rather efficient training process on few training parameters (150M) and 10K data samples, and enables plug-and-play usage at inference time. Extensive experiments on state-of-the-art models demonstrate the effectiveness of our method in improving physical rule following and video fidelity. Our code and data are publicly released here: https://thrcle421.github.io/DiT-Mem-Web/.
>
---
#### [replaced 009] Neural Octahedral Field: Octahedral prior for simultaneous smoothing and sharp edge regularization
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2408.00303v2](https://arxiv.org/pdf/2408.00303v2)**

> **作者:** Ruichen Zheng; Tao Yu; Ruizhen Hu
>
> **备注:** project page: https://github.com/Ankbzpx/frame-field
>
> **摘要:** Neural implicit representation, the parameterization of a continuous distance function as a Multi-Layer Perceptron (MLP), has emerged as a promising lead in tackling surface reconstruction from unoriented point clouds. In the presence of noise, however, its lack of explicit neighborhood connectivity makes sharp edges identification particularly challenging, hence preventing the separation of smoothing and sharpening operations, as is achievable with its discrete counterparts. In this work, we propose to tackle this challenge with an auxiliary field, the \emph{octahedral field}. We observe that both smoothness and sharp features in the distance field can be equivalently described by the smoothness in octahedral space. Therefore, by aligning and smoothing an octahedral field alongside the implicit geometry, our method behaves analogously to bilateral filtering, resulting in a smooth reconstruction while preserving sharp edges. Despite being operated purely pointwise, our method outperforms various traditional and neural implicit fitting approaches across extensive experiments, and is very competitive with methods that require normals and data priors. Code and data of our work are available at: https://github.com/Ankbzpx/frame-field.
>
---
#### [replaced 010] Loomis Painter: Reconstructing the Painting Process
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17344v2](https://arxiv.org/pdf/2511.17344v2)**

> **作者:** Markus Pobitzer; Chang Liu; Chenyi Zhuang; Teng Long; Bin Ren; Nicu Sebe
>
> **摘要:** Step-by-step painting tutorials are vital for learning artistic techniques, but existing video resources (e.g., YouTube) lack interactivity and personalization. While recent generative models have advanced artistic image synthesis, they struggle to generalize across media and often show temporal or structural inconsistencies, hindering faithful reproduction of human creative workflows. To address this, we propose a unified framework for multi-media painting process generation with a semantics-driven style control mechanism that embeds multiple media into a diffusion models conditional space and uses cross-medium style augmentation. This enables consistent texture evolution and process transfer across styles. A reverse-painting training strategy further ensures smooth, human-aligned generation. We also build a large-scale dataset of real painting processes and evaluate cross-media consistency, temporal coherence, and final-image fidelity, achieving strong results on LPIPS, DINO, and CLIP metrics. Finally, our Perceptual Distance Profile (PDP) curve quantitatively models the creative sequence, i.e., composition, color blocking, and detail refinement, mirroring human artistic progression.
>
---
#### [replaced 011] Entropy Rectifying Guidance for Diffusion and Flow Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.13987v2](https://arxiv.org/pdf/2504.13987v2)**

> **作者:** Tariq Berrada Ifriqi; Adriana Romero-Soriano; Michal Drozdzal; Jakob Verbeek; Karteek Alahari
>
> **备注:** NeurIPS 2025
>
> **摘要:** Guidance techniques are commonly used in diffusion and flow models to improve image quality and input consistency for conditional generative tasks such as class-conditional and text-to-image generation. In particular, classifier-free guidance (CFG) is the most widely adopted guidance technique. It results, however, in trade-offs across quality, diversity and consistency: improving some at the expense of others. While recent work has shown that it is possible to disentangle these factors to some extent, such methods come with an overhead of requiring an additional (weaker) model, or require more forward passes per sampling step. In this paper, we propose Entropy Rectifying Guidance (ERG), a simple and effective guidance method based on inference-time changes in the attention mechanism of state-of-the-art diffusion transformer architectures, which allows for simultaneous improvements over image quality, diversity and prompt consistency. ERG is more general than CFG and similar guidance techniques, as it extends to unconditional sampling. We show that ERG results in significant improvements in various tasks, including text-to-image, class-conditional and unconditional image generation. We also show that ERG can be seamlessly combined with other recent guidance methods such as CADS and APG, further improving generation results.
>
---
#### [replaced 012] G$^2$VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出G²VLM，一种基于几何引导的视觉语言模型，旨在解决视觉语言模型在空间理解与推理上的不足。通过融合多视角图像与视频数据，实现统一的3D重建与空间推理，提升模型对三维空间的感知能力，为后续3D场景编辑等应用提供基础。**

- **链接: [https://arxiv.org/pdf/2511.21688v2](https://arxiv.org/pdf/2511.21688v2)**

> **作者:** Wenbo Hu; Jingli Lin; Yilin Long; Yunlong Ran; Lihan Jiang; Yifan Wang; Chenming Zhu; Runsen Xu; Tai Wang; Jiangmiao Pang
>
> **备注:** code are released at https://github.com/InternRobotics/G2VLM
>
> **摘要:** Vision-Language Models (VLMs) still lack robustness in spatial intelligence, demonstrating poor performance on spatial understanding and reasoning tasks. We attribute this gap to the absence of a visual geometry learning process capable of reconstructing 3D space from 2D images. We present G$^2$VLM, a geometry grounded vision-language model that bridges two fundamental aspects of spatial intelligence: spatial 3D reconstruction and spatial understanding. G$^2$VLM natively leverages learned 3D visual geometry features to directly predict 3D attributes and enhance spatial reasoning tasks via in-context learning and interleaved reasoning. Our unified design is highly scalable for spatial understanding: it trains on abundant multi-view image and video data, while simultaneously leveraging the benefits of 3D visual priors that are typically only derived from hard-to-collect annotations. Experimental results demonstrate G$^2$VLM is proficient in both tasks, achieving comparable results to state-of-the-art feed-forward 3D reconstruction models and achieving better or competitive results across spatial understanding and reasoning tasks. By unifying a semantically strong VLM with low-level 3D vision tasks, we hope G$^2$VLM can serve as a strong baseline for the community and unlock more future applications, such as 3D scene editing.
>
---
#### [replaced 013] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms
- **分类: cs.LG; cs.CV; math.NA; physics.comp-ph; stat.ML**

- **链接: [https://arxiv.org/pdf/2502.00234v2](https://arxiv.org/pdf/2502.00234v2)**

> **作者:** Yinuo Ren; Haoxuan Chen; Yuchen Zhu; Wei Guo; Yongxin Chen; Grant M. Rotskoff; Molei Tao; Lexing Ying
>
> **备注:** Accepted at NeurIPS 2025 as a Poster (https://openreview.net/forum?id=OuklL6Q3sO)
>
> **摘要:** Discrete diffusion models have emerged as a powerful generative modeling framework for discrete data with successful applications spanning from text generation to image synthesis. However, their deployment faces challenges due to the high dimensionality of the state space, necessitating the development of efficient inference algorithms. Current inference approaches mainly fall into two categories: exact simulation and approximate methods such as $τ$-leaping. While exact methods suffer from unpredictable inference time and redundant function evaluations, $τ$-leaping is limited by its first-order accuracy. In this work, we advance the latter category by tailoring the first extension of high-order numerical inference schemes to discrete diffusion models, enabling larger step sizes while reducing error. We rigorously analyze the proposed schemes and establish the second-order accuracy of the $θ$-Trapezoidal method in KL divergence. Empirical evaluations on GSM8K-level math-reasoning, GPT-2-level text, and ImageNet-level image generation tasks demonstrate that our method achieves superior sample quality compared to existing approaches under equivalent computational constraints, with consistent performance gains across models ranging from 200M to 8B. Our code is available at https://github.com/yuchen-zhu-zyc/DiscreteFastSolver.
>
---
#### [replaced 014] FIELDS: Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21245v2](https://arxiv.org/pdf/2511.21245v2)**

> **作者:** Chen Ling; Henglin Shi; Hedvig Kjellström
>
> **摘要:** Facial expressions convey the bulk of emotional information in human communication, yet existing 3D face reconstruction methods often miss subtle affective details due to reliance on 2D supervision and lack of 3D ground truth. We propose FIELDS (Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision) to address these limitations by extending self-supervised 2D image consistency cues with direct 3D expression parameter supervision and an auxiliary emotion recognition branch. Our encoder is guided by authentic expression parameters from spontaneous 4D facial scans, while an intensity-aware emotion loss encourages the 3D expression parameters to capture genuine emotion content without exaggeration. This dual-supervision strategy bridges the 2D/3D domain gap and mitigates expression-intensity bias, yielding high-fidelity 3D reconstructions that preserve subtle emotional cues. From a single image, FIELDS produces emotion-rich face models with highly realistic expressions, significantly improving in-the-wild facial expression recognition performance without sacrificing naturalness.
>
---
#### [replaced 015] EMO-X: Efficient Multi-Person Pose and Shape Estimation in One-Stage
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.08718v2](https://arxiv.org/pdf/2504.08718v2)**

> **作者:** Haohang Jian; Jinlu Zhang; Junyi Wu; Zhigang Tu
>
> **备注:** The manuscript is being revised to include new experimental results and an improved model architecture
>
> **摘要:** Expressive Human Pose and Shape Estimation (EHPS) aims to jointly estimate human pose, hand gesture, and facial expression from monocular images. Existing methods predominantly rely on Transformer-based architectures, which suffer from quadratic complexity in self-attention, leading to substantial computational overhead, especially in multi-person scenarios. Recently, Mamba has emerged as a promising alternative to Transformers due to its efficient global modeling capability. However, it remains limited in capturing fine-grained local dependencies, which are essential for precise EHPS. To address these issues, we propose EMO-X, the Efficient Multi-person One-stage model for multi-person EHPS. Specifically, we explore a Scan-based Global-Local Decoder (SGLD) that integrates global context with skeleton-aware local features to iteratively enhance human tokens. Our EMO-X leverages the superior global modeling capability of Mamba and designs a local bidirectional scan mechanism for skeleton-aware local refinement. Comprehensive experiments demonstrate that EMO-X strikes an excellent balance between efficiency and accuracy. Notably, it achieves a significant reduction in computational complexity, requiring 69.8% less inference time compared to state-of-the-art (SOTA) methods, while outperforming most of them in accuracy.
>
---
#### [replaced 016] ProxT2I: Efficient Reward-Guided Text-to-Image Generation via Proximal Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.18742v2](https://arxiv.org/pdf/2511.18742v2)**

> **作者:** Zhenghan Fang; Jian Zheng; Qiaozi Gao; Xiaofeng Gao; Jeremias Sulam
>
> **摘要:** Diffusion models have emerged as a dominant paradigm for generative modeling across a wide range of domains, including prompt-conditional generation. The vast majority of samplers, however, rely on forward discretization of the reverse diffusion process and use score functions that are learned from data. Such forward and explicit discretizations can be slow and unstable, requiring a large number of sampling steps to produce good-quality samples. In this work we develop a text-to-image (T2I) diffusion model based on backward discretizations, dubbed ProxT2I, relying on learned and conditional proximal operators instead of score functions. We further leverage recent advances in reinforcement learning and policy optimization to optimize our samplers for task-specific rewards. Additionally, we develop a new large-scale and open-source dataset comprising 15 million high-quality human images with fine-grained captions, called LAION-Face-T2I-15M, for training and evaluation. Our approach consistently enhances sampling efficiency and human-preference alignment compared to score-based baselines, and achieves results on par with existing state-of-the-art and open-source text-to-image models while requiring lower compute and smaller model size, offering a lightweight yet performant solution for human text-to-image generation.
>
---
#### [replaced 017] VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.13361v2](https://arxiv.org/pdf/2507.13361v2)**

> **作者:** Shmuel Berman; Jia Deng
>
> **摘要:** Vision-Language Models (VLMs) excel at complex visual tasks such as VQA and chart understanding, yet recent work suggests they struggle with simple perceptual tests. We present an evaluation of vision-language models' capacity for nonlocal visual reasoning: reasoning that requires chaining evidence collected from multiple, possibly distant regions of an image. We isolate three distinct forms of nonlocal vision: comparative perception, which demands holding two images in working memory and comparing them; saccadic search, which requires making discrete, evidence-driven jumps to locate successive targets; and smooth visual search, which involves following a continuous contour. Flagship models (e.g., GPT-5, Gemini 2.5 Pro, Claude Sonnet 4), even those that perform well on prior primitive-vision benchmarks, fail these tests and barely exceed random accuracy on two variants of our tasks that are trivial for humans. Our structured evaluation suite allows us to test whether VLMs can perform visual algorithms similar to those used by humans. Our findings show that despite gains in raw visual acuity, current models lack core visual reasoning capabilities.
>
---
#### [replaced 018] SplitFlux: Learning to Decouple Content and Style from a Single Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15258v2](https://arxiv.org/pdf/2511.15258v2)**

> **作者:** Yitong Yang; Yinglin Wang; Changshuo Wang; Yongjun Zhang; Ziyang Chen; Shuting He
>
> **摘要:** Disentangling image content and style is essential for customized image generation. Existing SDXL-based methods struggle to achieve high-quality results, while the recently proposed Flux model fails to achieve effective content-style separation due to its underexplored characteristics. To address these challenges, we conduct a systematic analysis of Flux and make two key observations: (1) Single Stream Blocks are essential for image generation; and (2) Early single stream blocks mainly control content, whereas later blocks govern style. Based on these insights, we propose SplitFlux, which disentangles content and style by fine-tuning the single stream blocks via LoRA, enabling the disentangled content to be re-embedded into new contexts. It includes two key components: (1) Rank-Constrained Adaptation. To preserve content identity and structure, we compress the rank and amplify the magnitude of updates within specific blocks, preventing content leakage into style blocks. (2) Visual-Gated LoRA. We split the content LoRA into two branches with different ranks, guided by image saliency. The high-rank branch preserves primary subject information, while the low-rank branch encodes residual details, mitigating content overfitting and enabling seamless re-embedding. Extensive experiments demonstrate that SplitFlux consistently outperforms state-of-the-art methods, achieving superior content preservation and stylization quality across diverse scenarios.
>
---
#### [replaced 019] PIS3R: Very Large Parallax Image Stitching via Deep 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04236v2](https://arxiv.org/pdf/2508.04236v2)**

> **作者:** Muhua Zhu; Xinhao Jin; Chengbo Wang; Yongcong Zhang; Yifei Xue; Tie Ji; Yizhen Lao
>
> **摘要:** Image stitching aim to align two images taken from different viewpoints into one seamless, wider image. However, when the 3D scene contains depth variations and the camera baseline is significant, noticeable parallax occurs-meaning the relative positions of scene elements differ substantially between views. Most existing stitching methods struggle to handle such images with large parallax effectively. To address this challenge, in this paper, we propose an image stitching solution called PIS3R that is robust to very large parallax based on the novel concept of deep 3D reconstruction. First, we apply visual geometry grounded transformer to two input images with very large parallax to obtain both intrinsic and extrinsic parameters, as well as the dense 3D scene reconstruction. Subsequently, we reproject reconstructed dense point cloud onto a designated reference view using the recovered camera parameters, achieving pixel-wise alignment and generating an initial stitched image. Finally, to further address potential artifacts such as holes or noise in the initial stitching, we propose a point-conditioned image diffusion module to obtain the refined result.Compared with existing methods, our solution is very large parallax tolerant and also provides results that fully preserve the geometric integrity of all pixels in the 3D photogrammetric context, enabling direct applicability to downstream 3D vision tasks such as SfM. Experimental results demonstrate that the proposed algorithm provides accurate stitching results for images with very large parallax, and outperforms the existing methods qualitatively and quantitatively.
>
---
#### [replaced 020] S2AFormer: Strip Self-Attention for Efficient Vision Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22195v2](https://arxiv.org/pdf/2505.22195v2)**

> **作者:** Guoan Xu; Wenfeng Huang; Wenjing Jia; Jiamao Li; Guangwei Gao; Guo-Jun Qi
>
> **备注:** Accepted by IEEE-TIP, 14 pages, 8 figures, 9 tables
>
> **摘要:** Vision Transformer (ViT) has made significant advancements in computer vision, thanks to its token mixer's sophisticated ability to capture global dependencies between all tokens. However, the quadratic growth in computational demands as the number of tokens increases limits its practical efficiency. Although recent methods have combined the strengths of convolutions and self-attention to achieve better trade-offs, the expensive pairwise token affinity and complex matrix operations inherent in self-attention remain a bottleneck. To address this challenge, we propose S2AFormer, an efficient Vision Transformer architecture featuring novel Strip Self-Attention (SSA). We design simple yet effective Hybrid Perception Blocks (HPBs) to effectively integrate the local perception capabilities of CNNs with the global context modeling of Transformer's attention mechanisms. A key innovation of SSA lies in its reduction of the spatial dimensions of $K$ and $V$, while compressing the channel dimensions of $Q$ and $K$. This design significantly reduces computational overhead while preserving accuracy, striking an optimal balance between efficiency and effectiveness. We evaluate the robustness and efficiency of S2AFormer through extensive experiments on multiple vision benchmarks, including ImageNet-1k for image classification, ADE20k for semantic segmentation, and COCO for object detection and instance segmentation. Results demonstrate that S2AFormer achieves significant accuracy gains with superior efficiency in both GPU and non-GPU environments, making it a strong candidate for efficient vision Transformers.
>
---
#### [replaced 021] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对长时序、高记忆需求的具身智能体决策任务，提出Memo框架。通过在训练中插入周期性摘要标记，实现视觉输入的高效压缩与记忆检索，解决Transformer因上下文过长导致的效率与泛化问题。实验表明，Memo在多任务场景中优于基线模型，具备更强的推理效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.19732v2](https://arxiv.org/pdf/2510.19732v2)**

> **作者:** Gunshi Gupta; Karmesh Yadav; Zsolt Kira; Yarin Gal; Rahaf Aljundi
>
> **备注:** Accepted for Spotlight Presentation at NeurIPS 2025
>
> **摘要:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints. Our code is available at: https://github.com/gunshi/memo.
>
---
#### [replaced 022] Advancing Semantic Future Prediction through Multimodal Visual Sequence Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.08303v2](https://arxiv.org/pdf/2501.08303v2)**

> **作者:** Efstathios Karypidis; Ioannis Kakogeorgiou; Spyros Gidaris; Nikos Komodakis
>
> **备注:** CVPR 2025
>
> **摘要:** Semantic future prediction is important for autonomous systems navigating dynamic environments. This paper introduces FUTURIST, a method for multimodal future semantic prediction that uses a unified and efficient visual sequence transformer architecture. Our approach incorporates a multimodal masked visual modeling objective and a novel masking mechanism designed for multimodal training. This allows the model to effectively integrate visible information from various modalities, improving prediction accuracy. Additionally, we propose a VAE-free hierarchical tokenization process, which reduces computational complexity, streamlines the training pipeline, and enables end-to-end training with high-resolution, multimodal inputs. We validate FUTURIST on the Cityscapes dataset, demonstrating state-of-the-art performance in future semantic segmentation for both short- and mid-term forecasting. Project page and code at https://futurist-cvpr2025.github.io/ .
>
---
#### [replaced 023] Terminal Velocity Matching
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.19797v2](https://arxiv.org/pdf/2511.19797v2)**

> **作者:** Linqi Zhou; Mathias Parger; Ayaan Haque; Jiaming Song
>
> **备注:** Blog post: https://lumalabs.ai/blog/engineering/tvm Code available at: https://github.com/lumalabs/tvm
>
> **摘要:** We propose Terminal Velocity Matching (TVM), a generalization of flow matching that enables high-fidelity one- and few-step generative modeling. TVM models the transition between any two diffusion timesteps and regularizes its behavior at its terminal time rather than at the initial time. We prove that TVM provides an upper bound on the $2$-Wasserstein distance between data and model distributions when the model is Lipschitz continuous. However, since Diffusion Transformers lack this property, we introduce minimal architectural changes that achieve stable, single-stage training. To make TVM efficient in practice, we develop a fused attention kernel that supports backward passes on Jacobian-Vector Products, which scale well with transformer architectures. On ImageNet-256x256, TVM achieves 3.29 FID with a single function evaluation (NFE) and 1.99 FID with 4 NFEs. It similarly achieves 4.32 1-NFE FID and 2.94 4-NFE FID on ImageNet-512x512, representing state-of-the-art performance for one/few-step models from scratch.
>
---
#### [replaced 024] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.05296v2](https://arxiv.org/pdf/2504.05296v2)**

> **作者:** Gal Fiebelman; Hadar Averbuch-Elor; Sagie Benaim
>
> **备注:** Project webpage: https://galfiebelman.github.io/let-it-snow/
>
> **摘要:** 3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, dynamic editing of such scenes remains a significant challenge. We introduce a novel framework, Physics-Guided Score Distillation, to address a fundamental conflict: physics simulation provides a strong motion prior that is insufficient for photorealism , while video-based Score Distillation Sampling (SDS) alone cannot generate coherent motion for complex, multi-particle scenarios. We resolve this through a unified optimization framework where physics simulation guides Score Distillation to jointly refine the motion prior for photorealism while simultaneously optimizing appearance. Specifically, we learn a neural dynamics model that predicts particle motion and appearance, optimized end-to-end via a combined loss integrating Video-SDS for photorealism with our physics-guidance prior. This allows for photorealistic refinements while ensuring the dynamics remain plausible. Our framework enables scene-wide dynamic weather effects, including snowfall, rainfall, fog, and sandstorms, with physically plausible motion. Experiments demonstrate our physics-guided approach significantly outperforms baselines, with ablations confirming this joint refinement is essential for generating coherent, high-fidelity dynamics.
>
---
#### [replaced 025] ForAug: Recombining Foregrounds and Backgrounds to Improve Vision Transformer Training with Bias Mitigation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.09399v3](https://arxiv.org/pdf/2503.09399v3)**

> **作者:** Tobias Christian Nauen; Brian Moser; Federico Raue; Stanislav Frolov; Andreas Dengel
>
> **备注:** v2: added DeiT, added ablation vs simple copy-paste
>
> **摘要:** Transformers, particularly Vision Transformers (ViTs), have achieved state-of-the-art performance in large-scale image classification. However, they often require large amounts of data and can exhibit biases, such as center or size bias, that limit their robustness and generalizability. This paper introduces ForAug, a novel data augmentation operation that addresses these challenges by explicitly imposing invariances into the training data, which are otherwise part of the neural network architecture. ForAug is constructed by using pretrained foundation models to separate and recombine foreground objects with different backgrounds. This recombination step enables us to take fine-grained control over object position and size, as well as background selection. We demonstrate that using ForAug significantly improves the accuracy of ViTs and other architectures by up to 4.5 percentage points (p.p.) on ImageNet, which translates to 7.3 p.p. on downstream tasks. Importantly, ForAug not only improves accuracy but also opens new ways to analyze model behavior and quantify biases. Namely, we introduce metrics for background robustness, foreground focus, center bias, and size bias and show that using ForAug during training substantially reduces these biases. In summary, ForAug provides a valuable tool for analyzing and mitigating biases, enabling the development of more robust and reliable computer vision models. Our code and dataset are publicly available at https://github.com/tobna/ForAug.
>
---
#### [replaced 026] FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.22070v3](https://arxiv.org/pdf/2410.22070v3)**

> **作者:** Qizhi Chen; Delin Qu; Junli Liu; Yiwen Tang; Haoming Song; Dong Wang; Bin Zhao; Xuelong Li
>
> **摘要:** Reconstructing controllable Gaussian splats for articulated objects from monocular video is especially challenging due to its inherently insufficient constraints. Existing methods address this by relying on dense masks and manually defined control signals, limiting their real-world applications. In this paper, we propose an annotation-free method, FreeGaussian, which mathematically disentangles camera egomotion and articulated movements via flow derivatives. By establishing a connection between 2D flows and 3D Gaussian dynamic flow, our method enables optimization and continuity of dynamic Gaussian motions from flow priors without any control signals. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state as a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Extensive experiments on articulated objects demonstrate the state-of-the-art visual performance and precise, part-aware controllability of our method. Code is available at: https://github.com/Tavish9/freegaussian.
>
---
#### [replaced 027] TRACE: Temporally Reliable Anatomically-Conditioned 3D CT Generation with Enhanced Efficiency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.00802v2](https://arxiv.org/pdf/2507.00802v2)**

> **作者:** Minye Shao; Xingyu Miao; Haoran Duan; Zeyu Wang; Jingkun Chen; Yawen Huang; Xian Wu; Jingjing Deng; Yang Long; Yefeng Zheng
>
> **备注:** Accepted to MICCAI 2025 (this version is not peer-reviewed; it is the extended version)
>
> **摘要:** 3D medical image generation is essential for data augmentation and patient privacy, calling for reliable and efficient models suited for clinical practice. However, current methods suffer from limited anatomical fidelity, restricted axial length, and substantial computational cost, placing them beyond reach for regions with limited resources and infrastructure. We introduce TRACE, a framework that generates 3D medical images with spatiotemporal alignment using a 2D multimodal-conditioned diffusion approach. TRACE models sequential 2D slices as video frame pairs, combining segmentation priors and radiology reports for anatomical alignment, incorporating optical flow to sustain temporal coherence. During inference, an overlapping-frame strategy links frame pairs into a flexible length sequence, reconstructed into a spatiotemporally and anatomically aligned 3D volume. Experimental results demonstrate that TRACE effectively balances computational efficiency with preserving anatomical fidelity and spatiotemporal consistency. Code is available at: https://github.com/VinyehShaw/TRACE.
>
---
#### [replaced 028] Reliable Multimodal Learning Via Multi-Level Adaptive DeConfusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.19674v2](https://arxiv.org/pdf/2502.19674v2)**

> **作者:** Tong Zhang; Shu Shen; C. L. Philip Chen
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Multimodal learning enhances the performance of various machine learning tasks by leveraging complementary information across different modalities. However, existing methods often learn multimodal representations that retain substantial inter-class confusion, making it difficult to achieve high-confidence predictions, particularly in real-world scenarios with low-quality or noisy data. To address this challenge, we propose Multi-Level Adaptive DeConfusion (MLAD), which eliminates inter-class confusion in multimodal data at both global and sample levels, significantly enhancing the classification reliability of multimodal models. Specifically, MLAD first learns class-wise latent distributions with global-level confusion removed via dynamic-exit modality encoders that adapt to the varying discrimination difficulty of each class and a cross-class residual reconstruction mechanism. Subsequently, MLAD further removes sample-specific confusion through sample-adaptive cross-modality rectification guided by confusion-free modality priors. These priors are constructed from low-confusion modality features, identified by evaluating feature confusion using the learned class-wise latent distributions and selecting those with low confusion via a Gaussian mixture model. Experiments demonstrate that MLAD outperforms state-of-the-art methods across multiple benchmarks and exhibits superior reliability.
>
---
#### [replaced 029] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06400v3](https://arxiv.org/pdf/2507.06400v3)**

> **作者:** Weiran Li; Yeqiang Liu; Qiannan Guo; Yijie Wei; Hwa Liang Leo; Zhenbo Li
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Multiple object tracking (MOT) technology has made significant progress in terrestrial applications, but underwater tracking scenarios remain underexplored despite their importance to marine ecology and aquaculture. In this paper, we present Multiple Fish Tracking Dataset 2025 (MFT25), a comprehensive dataset specifically designed for underwater multiple fish tracking, featuring 15 diverse video sequences with 408,578 meticulously annotated bounding boxes across 48,066 frames. Our dataset captures various underwater environments, fish species, and challenging conditions including occlusions, similar appearances, and erratic motion patterns. Additionally, we introduce Scale-aware and Unscented Tracker (SU-T), a specialized tracking framework featuring an Unscented Kalman Filter (UKF) optimized for non-linear swimming patterns of fish and a novel Fish-Intersection-over-Union (FishIoU) matching that accounts for the unique morphological characteristics of aquatic species. Extensive experiments demonstrate that our SU-T baseline achieves state-of-the-art performance on MFT25, with 34.1 HOTA and 44.6 IDF1, while revealing fundamental differences between fish tracking and terrestrial object tracking scenarios. The dataset and codes are released at https://vranlee.github.io/SU-T/.
>
---
#### [replaced 030] TEFormer: Texture-Aware and Edge-Guided Transformer for Semantic Segmentation of Urban Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06224v2](https://arxiv.org/pdf/2508.06224v2)**

> **作者:** Guoyu Zhou; Jing Zhang; Yi Yan; Hui Zhang; Li Zhuo
>
> **备注:** Accepted by IEEE GRSL
>
> **摘要:** Accurate semantic segmentation of urban remote sensing images (URSIs) is essential for urban planning and environmental monitoring. However, it remains challenging due to the subtle texture differences and similar spatial structures among geospatial objects, which cause semantic ambiguity and misclassification. Additional complexities arise from irregular object shapes, blurred boundaries, and overlapping spatial distributions of objects, resulting in diverse and intricate edge morphologies. To address these issues, we propose TEFormer, a texture-aware and edge-guided Transformer. Our model features a texture-aware module (TaM) in the encoder to capture fine-grained texture distinctions between visually similar categories, thereby enhancing semantic discrimination. The decoder incorporates an edge-guided tri-branch decoder (Eg3Head) to preserve local edges and details while maintaining multiscale context-awareness. Finally, an edge-guided feature fusion module (EgFFM) effectively integrates contextual, detail, and edge information to achieve refined semantic segmentation. Extensive evaluation demonstrates that TEFormer yields mIoU scores of 88.57% on Potsdam and 81.46% on Vaihingen, exceeding the next best methods by 0.73% and 0.22%. On the LoveDA dataset, it secures the second position with an overall mIoU of 53.55%, trailing the optimal performance by a narrow margin of 0.19%.
>
---
#### [replaced 031] Autoregressive Styled Text Image Generation, but Make it Reliable
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23240v2](https://arxiv.org/pdf/2510.23240v2)**

> **作者:** Carmine Zaccagnino; Fabio Quattrini; Vittorio Pippi; Silvia Cascianelli; Alessio Tonioni; Rita Cucchiara
>
> **备注:** Accepted at WACV2026
>
> **摘要:** Generating faithful and readable styled text images (especially for Styled Handwritten Text generation - HTG) is an open problem with several possible applications across graphic design, document understanding, and image editing. A lot of research effort in this task is dedicated to developing strategies that reproduce the stylistic characteristics of a given writer, with promising results in terms of style fidelity and generalization achieved by the recently proposed Autoregressive Transformer paradigm for HTG. However, this method requires additional inputs, lacks a proper stop mechanism, and might end up in repetition loops, generating visual artifacts. In this work, we rethink the autoregressive formulation by framing HTG as a multimodal prompt-conditioned generation task, and tackle the content controllability issues by introducing special textual input tokens for better alignment with the visual ones. Moreover, we devise a Classifier-Free-Guidance-based strategy for our autoregressive model. Through extensive experimental validation, we demonstrate that our approach, dubbed Eruku, compared to previous solutions requires fewer inputs, generalizes better to unseen styles, and follows more faithfully the textual prompt, improving content adherence.
>
---
#### [replaced 032] Reverberation: Learning the Latencies Before Forecasting Trajectories
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11164v2](https://arxiv.org/pdf/2511.11164v2)**

> **作者:** Conghao Wong; Ziqian Zou; Beihao Xia; Xinge You
>
> **摘要:** Bridging the past to the future, connecting agents both spatially and temporally, lies at the core of the trajectory prediction task. Despite great efforts, it remains challenging to explicitly learn and predict latencies, i.e., response intervals or temporal delays with which agents respond to various trajectory-changing events and adjust their future paths, whether on their own or interactively. Different agents may exhibit distinct latency preferences for noticing, processing, and reacting to a specific trajectory-changing event. The lack of consideration of such latencies may undermine the causal continuity of forecasting systems, leading to implausible or unintended trajectories. Inspired by reverberation in acoustics, we propose a new reverberation transform and the corresponding Reverberation (short for Rev) trajectory prediction model, which predicts both individual latency preferences and their stochastic variations accordingly, by using two explicit and learnable reverberation kernels, enabling latency-conditioned and controllable trajectory prediction of both non-interactive and social latencies. Experiments on multiple datasets, whether pedestrians or vehicles, demonstrate that Rev achieves competitive accuracy while revealing interpretable latency dynamics across agents and scenarios. Qualitative analyses further verify the properties of the reverberation transform, highlighting its potential as a general latency modeling approach.
>
---
#### [replaced 033] Dual-Model Weight Selection and Self-Knowledge Distillation for Medical Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.20461v2](https://arxiv.org/pdf/2508.20461v2)**

> **作者:** Ayaka Tsutsumi; Guang Li; Ren Togo; Takahiro Ogawa; Satoshi Kondo; Miki Haseyama
>
> **备注:** Published as a journal paper at Elsevier CIBM
>
> **摘要:** We propose a novel medical image classification method that integrates dual-model weight selection with self-knowledge distillation (SKD). In real-world medical settings, deploying large-scale models is often limited by computational resource constraints, which pose significant challenges for their practical implementation. Thus, developing lightweight models that achieve comparable performance to large-scale models while maintaining computational efficiency is crucial. To address this, we employ a dual-model weight selection strategy that initializes two lightweight models with weights derived from a large pretrained model, enabling effective knowledge transfer. Next, SKD is applied to these selected models, allowing the use of a broad range of initial weight configurations without imposing additional excessive computational cost, followed by fine-tuning for the target classification tasks. By combining dual-model weight selection with self-knowledge distillation, our method overcomes the limitations of conventional approaches, which often fail to retain critical information in compact models. Extensive experiments on publicly available datasets-chest X-ray images, lung computed tomography scans, and brain magnetic resonance imaging scans-demonstrate the superior performance and robustness of our approach compared to existing methods.
>
---
#### [replaced 034] Group Relative Attention Guidance for Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24657v2](https://arxiv.org/pdf/2510.24657v2)**

> **作者:** Xuanpu Zhang; Xuesong Niu; Ruidong Chen; Dan Song; Jianhao Zeng; Penghui Du; Haoxiang Cao; Kai Wu; An-an Liu
>
> **摘要:** Recently, image editing based on Diffusion-in-Transformer models has undergone rapid development. However, existing editing methods often lack effective control over the degree of editing, limiting their ability to achieve more customized results. To address this limitation, we investigate the MM-Attention mechanism within the DiT model and observe that the Query and Key tokens share a bias vector that is only layer-dependent. We interpret this bias as representing the model's inherent editing behavior, while the delta between each token and its corresponding bias encodes the content-specific editing signals. Based on this insight, we propose Group Relative Attention Guidance, a simple yet effective method that reweights the delta values of different tokens to modulate the focus of the model on the input image relative to the editing instruction, enabling continuous and fine-grained control over editing intensity without any tuning. Extensive experiments conducted on existing image editing frameworks demonstrate that GRAG can be integrated with as few as four lines of code, consistently enhancing editing quality. Moreover, compared to the commonly used Classifier-Free Guidance, GRAG achieves smoother and more precise control over the degree of editing. Our code will be released at https://github.com/little-misfit/GRAG-Image-Editing.
>
---
#### [replaced 035] From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文聚焦多模态大模型的复杂推理能力提升，针对现有模型推理不透明、泛化性差的问题，系统综述了多模态思维链（MCoT）方法。从技术演进与任务需求出发，分析其原理、训练与推理策略，总结评估体系与应用场景，并展望未来挑战与方向。**

- **链接: [https://arxiv.org/pdf/2511.12861v4](https://arxiv.org/pdf/2511.12861v4)**

> **作者:** Wenxin Zhu; Andong Chen; Yuchen Song; Kehai Chen; Conghui Zhu; Ziyan Chen; Tiejun Zhao
>
> **备注:** Survey; 7 figures, 3 tables, 44 pages
>
> **摘要:** With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions.
>
---
#### [replaced 036] Configurable Fairness: Direct Optimization of Parity Metrics via Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.10624v3](https://arxiv.org/pdf/2403.10624v3)**

> **作者:** Miao Zhang; Rumi Chunara
>
> **摘要:** Performance disparities of image recognition across demographic groups are known to exist in deep learning-based models, due to imbalanced group representations or spurious correlation between group and target labels. Previous work has addressed such challenges without relying on expensive group labels, typically by upweighting high-loss samples or balancing discovered clusters. However, these heuristic strategies lack direct connection to specific fairness metrics and cannot guarantee optimization of parity-based criteria like equal opportunity, which ensures equal chance to receive positive outcomes across groups. In this work, we propose a novel paradigm that directly optimizes parity-based fairness metrics through specifically designed training objectives, without requiring group labels. We leverage vision-language models to analyze sensitive attribute relevancy for individual samples, then formulate loss functions that mathematically connect to each target fairness metric. This enables flexible optimization of different fairness criteria based on application needs. Experiments on multiple image classification datasets show that our metric-specific approach significantly improves parity-based fairness criteria and outperforms existing methods.
>
---
#### [replaced 037] Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01730v2](https://arxiv.org/pdf/2508.01730v2)**

> **作者:** Jianbo Ma; Hui Luo; Qi Chen; Yuankai Qi; Yumei Sun; Amin Beheshti; Jianlin Zhang; Ming-Hsuan Yang
>
> **备注:** Accepted by the AAAI26 Conference Main Track
>
> **摘要:** Multi-object tracking (MOT) aims to track multiple objects while maintaining consistent identities across frames of a given video. In unmanned aerial vehicle (UAV) recorded videos, frequent viewpoint changes and complex UAV-ground relative motion dynamics pose significant challenges, which often lead to unstable affinity measurement and ambiguous association. Existing methods typically model motion and appearance cues separately, overlooking their spatio-temporal interplay and resulting in suboptimal tracking performance. In this work, we propose AMOT, which jointly exploits appearance and motion cues through two key components: an Appearance-Motion Consistency (AMC) matrix and a Motion-aware Track Continuation (MTC) module. Specifically, the AMC matrix computes bi-directional spatial consistency under the guidance of appearance features, enabling more reliable and context-aware identity association. The MTC module complements AMC by reactivating unmatched tracks through appearance-guided predictions that align with Kalman-based predictions, thereby reducing broken trajectories caused by missed detections. Extensive experiments on three UAV benchmarks, including VisDrone2019, UAVDT, and VT-MOT-UAV, demonstrate that our AMOT outperforms current state-of-the-art methods and generalizes well in a plug-and-play and training-free manner.
>
---
#### [replaced 038] PhysX-3D: Physical-Grounded 3D Asset Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.12465v4](https://arxiv.org/pdf/2507.12465v4)**

> **作者:** Ziang Cao; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** Accepted by NeurIPS 2025, Spotlight Project page: https://physx-3d.github.io/
>
> **摘要:** 3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX-3D}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
>
---
#### [replaced 039] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08423v2](https://arxiv.org/pdf/2511.08423v2)**

> **作者:** Yuncheng Guo; Junyan Ye; Chenjue Zhang; Hengrui Kang; Haohuan Fu; Conghui He; Weijia Li
>
> **备注:** 19 pages, 10 figures, 19 tables
>
> **摘要:** A truly universal AI-Generated Image (AIGI) detector must simultaneously generalize across diverse generative models and varied semantic content. Current state-of-the-art methods learn a single, entangled forgery representation, conflating content-dependent flaws with content-agnostic artifacts, and are further constrained by outdated benchmarks. To overcome these limitations, we propose OmniAID, a novel framework centered on a decoupled Mixture-of-Experts (MoE) architecture. The core of our method is a hybrid expert system designed to decouple: (1) semantic flaws across distinct content domains, and (2) content-dependent flaws from content-agnostic universal artifacts. This system employs a set of Routable Specialized Semantic Experts, each for a distinct domain (e.g., human, animal), complemented by a Fixed Universal Artifact Expert. This architecture is trained using a novel two-stage strategy: we first train the experts independently with domain-specific hard-sampling to ensure specialization, and subsequently train a lightweight gating network for effective input routing. By explicitly decoupling "what is generated" (content-specific flaws) from "how it is generated" (universal artifacts), OmniAID achieves robust generalization. To address outdated benchmarks and validate real-world applicability, we introduce Mirage, a new large-scale, contemporary dataset. Extensive experiments, using both traditional benchmarks and our Mirage dataset, demonstrate our model surpasses existing monolithic detectors, establishing a new and robust standard for AIGI authentication against modern, in-the-wild threats.
>
---
#### [replaced 040] FlashEdit: Decoupling Speed, Structure, and Semantics for Precise Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22244v4](https://arxiv.org/pdf/2509.22244v4)**

> **作者:** Junyi Wu; Zhiteng Li; Haotong Qin; Xiaohong Liu; Linghe Kong; Yulun Zhang; Xiaokang Yang
>
> **备注:** Our code will be made publicly available at https://github.com/JunyiWuCode/FlashEdit
>
> **摘要:** Text-guided image editing with diffusion models has achieved remarkable quality but suffers from prohibitive latency, hindering real-world applications. We introduce FlashEdit, a novel framework designed to enable high-fidelity, real-time image editing. Its efficiency stems from three key innovations: (1) a One-Step Inversion-and-Editing (OSIE) pipeline that bypasses costly iterative processes; (2) a Background Shield (BG-Shield) technique that guarantees background preservation by selectively modifying features only within the edit region; and (3) a Sparsified Spatial Cross-Attention (SSCA) mechanism that ensures precise, localized edits by suppressing semantic leakage to the background. Extensive experiments demonstrate that FlashEdit maintains superior background consistency and structural integrity, while performing edits in under 0.2 seconds, which is an over 150$\times$ speedup compared to prior multi-step methods. Our code will be made publicly available at https://github.com/JunyiWuCode/FlashEdit.
>
---
#### [replaced 041] Accelerating Parallel Diffusion Model Serving with Residual Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.17511v2](https://arxiv.org/pdf/2507.17511v2)**

> **作者:** Jiajun Luo; Yicheng Xiao; Jianru Xu; Yangxiu You; Rongwei Lu; Chen Tang; Jingyan Jiang; Zhi Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Diffusion models produce realistic images and videos but require substantial computational resources, necessitating multi-accelerator parallelism for real-time deployment. However, parallel inference introduces significant communication overhead from exchanging large activations between devices, limiting efficiency and scalability. We present CompactFusion, a compression framework that significantly reduces communication while preserving generation quality. Our key observation is that diffusion activations exhibit strong temporal redundancy-adjacent steps produce highly similar activations, saturating bandwidth with near-duplicate data carrying little new information. To address this inefficiency, we seek a more compact representation that encodes only the essential information. CompactFusion achieves this via Residual Compression that transmits only compressed residuals (step-wise activation differences). Based on empirical analysis and theoretical justification, we show that it effectively removes redundant data, enabling substantial data reduction while maintaining high fidelity. We also integrate lightweight error feedback to prevent error accumulation. CompactFusion establishes a new paradigm for parallel diffusion inference, delivering lower latency and significantly higher generation quality than prior methods. On 4xL20, it achieves 3.0x speedup while greatly improving fidelity. It also uniquely supports communication-heavy strategies like sequence parallelism on slow networks, achieving 6.7x speedup over prior overlap-based method. CompactFusion applies broadly across diffusion models and parallel settings, and integrates easily without requiring pipeline rework. Portable implementation demonstrated on xDiT is publicly available at https://github.com/Cobalt-27/CompactFusion
>
---
#### [replaced 042] EasyOcc: 3D Pseudo-Label Supervision for Fully Self-Supervised Semantic Occupancy Prediction Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.26087v3](https://arxiv.org/pdf/2509.26087v3)**

> **作者:** Seamie Hayes; Ganesh Sistu; Ciarán Eising
>
> **摘要:** Self-supervised models have recently achieved notable advancements, particularly in the domain of semantic occupancy prediction. These models utilize sophisticated loss computation strategies to compensate for the absence of ground-truth labels. For instance, techniques such as novel view synthesis, cross-view rendering, and depth estimation have been explored to address the issue of semantic and depth ambiguity. However, such techniques typically incur high computational costs and memory usage during the training stage, especially in the case of novel view synthesis. To mitigate these issues, we propose 3D pseudo-ground-truth labels generated by the foundation models Grounded-SAM and Metric3Dv2, and harness temporal information for label densification. Our 3D pseudo-labels can be easily integrated into existing models, which yields substantial performance improvements, with mIoU increasing by 45\%, from 9.73 to 14.09, when implemented into the OccNeRF model. This stands in contrast to earlier advancements in the field, which are often not readily transferable to other architectures. Additionally, we propose a streamlined model, EasyOcc, achieving 13.86 mIoU. This model conducts learning solely from our labels, avoiding complex rendering strategies mentioned previously. Furthermore, our method enables models to attain state-of-the-art performance when evaluated on the full scene without applying the camera mask, with EasyOcc achieving 7.71 mIoU, outperforming the previous best model by 31\%. These findings highlight the critical importance of foundation models, temporal context, and the choice of loss computation space in self-supervised learning for comprehensive scene understanding.
>
---
#### [replaced 043] Training-Free Adaptive Quantization for Variable Rate Image Coding for Machines
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05836v2](https://arxiv.org/pdf/2511.05836v2)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **摘要:** Image Coding for Machines (ICM) has become increasingly important with the rapid integration of computer vision technology into real-world applications. However, most neural network-based ICM frameworks operate at a fixed rate, thus requiring individual training for each target bitrate. This limitation may restrict their practical usage. Existing variable rate image compression approaches mitigate this issue but often rely on additional training, which increases computational costs and complicates deployment. Moreover, variable rate control has not been thoroughly explored for ICM. To address these challenges, we propose a training-free quantization strength control scheme that enables flexible bitrate adjustment. By exploiting the scale parameter predicted by the hyperprior network, the proposed method adaptively modulates quantization step sizes across both channel and spatial dimensions. This allows the model to preserve semantically important regions while coarsely quantizing less critical areas. Our architectural design further enables continuous bitrate control through a single parameter. Experimental results demonstrate the effectiveness of our proposed method, achieving up to 11.07% BD-rate savings over the non-adaptive variable rate baseline.
>
---
#### [replaced 044] SARD: Segmentation-Aware Anomaly Synthesis via Region-Constrained Diffusion with Discriminative Mask Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03143v2](https://arxiv.org/pdf/2508.03143v2)**

> **作者:** Yanshu Wang; Xichen Xu; Xiaoning Lei; Guoyang Xie
>
> **备注:** Accepted by The 2025 International Conference on Machine Intelligence and Nature-InspireD Computing (MIND)
>
> **摘要:** Synthesizing realistic and spatially precise anomalies is essential for enhancing the robustness of industrial anomaly detection systems. While recent diffusion-based methods have demonstrated strong capabilities in modeling complex defect patterns, they often struggle with spatial controllability and fail to maintain fine-grained regional fidelity. To overcome these limitations, we propose SARD (Segmentation-Aware anomaly synthesis via Region-constrained Diffusion with discriminative mask Guidance), a novel diffusion-based framework specifically designed for anomaly generation. Our approach introduces a Region-Constrained Diffusion (RCD) process that preserves the background by freezing it and selectively updating only the foreground anomaly regions during the reverse denoising phase, thereby effectively reducing background artifacts. Additionally, we incorporate a Discriminative Mask Guidance (DMG) module into the discriminator, enabling joint evaluation of both global realism and local anomaly fidelity, guided by pixel-level masks. Extensive experiments on the MVTec-AD and BTAD datasets show that SARD surpasses existing methods in segmentation accuracy and visual quality, setting a new state-of-the-art for pixel-level anomaly synthesis.
>
---
#### [replaced 045] Scaling Spatial Intelligence with Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文聚焦多模态基础模型的空间智能提升任务，针对现有模型在空间理解上的不足，构建了包含800万样本的SenseNova-SI数据集，通过系统性数据构建与训练，显著提升模型在多项空间智能基准上的表现，并探索了数据规模、泛化能力及推理机制，推动多模态模型向更强空间认知发展。**

- **链接: [https://arxiv.org/pdf/2511.13719v2](https://arxiv.org/pdf/2511.13719v2)**

> **作者:** Zhongang Cai; Ruisi Wang; Chenyang Gu; Fanyi Pu; Junxiang Xu; Yubo Wang; Wanqi Yin; Zhitao Yang; Chen Wei; Qingping Sun; Tongxi Zhou; Jiaqi Li; Hui En Pang; Oscar Qian; Yukun Wei; Zhiqian Lin; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Xiangyu Fan; Hanming Deng; Lewei Lu; Liang Pan; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/OpenSenseNova/SenseNova-SI; Models: https://huggingface.co/collections/sensenova/sensenova-si
>
> **摘要:** Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.7% on VSI-Bench, 43.3% on MMSI, 85.6% on MindCube, 54.6% on ViewSpatial, and 50.1% on SITE, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. SenseNova-SI is an ongoing project, and this report will be updated continuously. All newly trained multimodal foundation models are publicly released to facilitate further research in this direction.
>
---
#### [replaced 046] DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10999v2](https://arxiv.org/pdf/2505.10999v2)**

> **作者:** Weilai Xiang; Hongyu Yang; Di Huang; Yunhong Wang
>
> **备注:** Updated version. Code available at https://github.com/FutureXiang/ddae_plus_plus
>
> **摘要:** While diffusion models excel at image synthesis, useful representations have been shown to emerge from generative pre-training, suggesting a path towards unified generative and discriminative learning. However, suboptimal semantic flow within current architectures can hinder this potential: features encoding the richest high-level semantics are underutilized and diluted when propagating through decoding layers, impeding the formation of an explicit semantic bottleneck layer. To address this, we introduce self-conditioning, a lightweight mechanism that reshapes the model's layer-wise semantic hierarchy without external guidance. By aggregating and rerouting intermediate features to guide subsequent decoding layers, our method concentrates more high-level semantics, concurrently strengthening global generative guidance and forming more discriminative representations. This simple approach yields a dual-improvement trend across pixel-space UNet, UViT and latent-space DiT models with minimal overhead. Crucially, it creates an architectural semantic bridge that propagates discriminative improvements into generation and accommodates further techniques such as contrastive self-distillation. Experiments show that our enhanced models, especially self-conditioned DiT, are powerful dual learners that yield strong and transferable representations on image and dense classification tasks, surpassing various generative self-supervised models in linear probing while also improving or maintaining high generation quality.
>
---
#### [replaced 047] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型（MLLM）训练中计算成本高的问题，提出ReGATE方法。通过教师-学生框架与动态令牌剪枝，实现训练加速。在不改变模型结构前提下，减少41%以上令牌使用，显著提升训练效率，且性能优于标准训练。**

- **链接: [https://arxiv.org/pdf/2507.21420v2](https://arxiv.org/pdf/2507.21420v2)**

> **作者:** Chaoyu Li; Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** The computational cost of training multimodal large language models (MLLMs) grows rapidly with the number of processed tokens. Existing efficiency methods mainly target inference via token reduction or merging, offering limited benefits during training. We introduce ReGATE (Reference-Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. ReGATE adopts a teacher-student framework, in which a frozen teacher LLM provides per-token guidance losses that are fused with an exponential moving average of the student's difficulty estimates. This adaptive scoring mechanism dynamically selects informative tokens while skipping redundant ones in the forward pass, substantially reducing computation without altering the model architecture. Across three representative MLLMs, ReGATE matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 38% of the tokens. With extended training, it even surpasses the baseline across multiple multimodal benchmarks, cutting total token usage by over 41%. Code and models will be released publicly.
>
---
#### [replaced 048] DiP: Taming Diffusion Models in Pixel Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18822v2](https://arxiv.org/pdf/2511.18822v2)**

> **作者:** Zhennan Chen; Junwei Zhu; Xu Chen; Jiangning Zhang; Xiaobin Hu; Hanzhen Zhao; Chengjie Wang; Jian Yang; Ying Tai
>
> **摘要:** Diffusion models face a fundamental trade-off between generation quality and computational efficiency. Latent Diffusion Models (LDMs) offer an efficient solution but suffer from potential information loss and non-end-to-end training. In contrast, existing pixel space models bypass VAEs but are computationally prohibitive for high-resolution synthesis. To resolve this dilemma, we propose DiP, an efficient pixel space diffusion framework. DiP decouples generation into a global and a local stage: a Diffusion Transformer (DiT) backbone operates on large patches for efficient global structure construction, while a co-trained lightweight Patch Detailer Head leverages contextual features to restore fine-grained local details. This synergistic design achieves computational efficiency comparable to LDMs without relying on a VAE. DiP is accomplished with up to 10$\times$ faster inference speeds than previous method while increasing the total number of parameters by only 0.3%, and achieves an 1.79 FID score on ImageNet 256$\times$256.
>
---
#### [replaced 049] Ultralight Polarity-Split Neuromorphic SNN for Event-Stream Super-Resolution
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.03244v2](https://arxiv.org/pdf/2508.03244v2)**

> **作者:** Chuanzhi Xu; Haoxian Zhou; Langyi Chen; Yuk Ying Chung; Qiang Qu
>
> **备注:** 8 pages, 10 figures, 7 tables, accepted by AAAI2026
>
> **摘要:** Event cameras offer unparalleled advantages such as high temporal resolution, low latency, and high dynamic range. However, their limited spatial resolution poses challenges for fine-grained perception tasks. In this work, we propose an ultra-lightweight, stream-based event-to-event super-resolution method based on Spiking Neural Networks (SNNs), designed for real-time deployment on resource-constrained devices. To further reduce model size, we introduce a novel Dual-Forward Polarity-Split Event Encoding strategy that decouples positive and negative events into separate forward paths through a shared SNN. Furthermore, we propose a Learnable Spatio-temporal Polarity-aware Loss (LearnSTPLoss) that adaptively balances temporal, spatial, and polarity consistency using learnable uncertainty-based weights. Experimental results demonstrate that our method achieves competitive super-resolution performance on multiple datasets while significantly reducing model size and inference time. The lightweight design enables embedding the module into event cameras or using it as an efficient front-end preprocessing for downstream vision tasks.
>
---
#### [replaced 050] Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.02863v2](https://arxiv.org/pdf/2507.02863v2)**

> **作者:** Yuqi Wu; Wenzhao Zheng; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at: https://github.com/YkiWu/Point3R
>
> **摘要:** Dense 3D scene reconstruction from an ordered sequence or unordered image collections is a critical step when bringing research in computer vision into practical scenarios. Following the paradigm introduced by DUSt3R, which unifies an image pair densely into a shared coordinate system, subsequent methods maintain an implicit memory to achieve dense 3D reconstruction from more images. However, such implicit memory is limited in capacity and may suffer from information loss of earlier frames. We propose Point3R, an online framework targeting dense streaming 3D reconstruction. To be specific, we maintain an explicit spatial pointer memory directly associated with the 3D structure of the current scene. Each pointer in this memory is assigned a specific 3D position and aggregates scene information nearby in the global coordinate system into a changing spatial feature. Information extracted from the latest frame interacts explicitly with this pointer memory, enabling dense integration of the current observation into the global coordinate system. We design a 3D hierarchical position embedding to promote this interaction and design a simple yet effective fusion mechanism to ensure that our pointer memory is uniform and efficient. Our method achieves competitive or state-of-the-art performance on various tasks with low training costs. Code: https://github.com/YkiWu/Point3R.
>
---
#### [replaced 051] DreamO: A Unified Framework for Image Customization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.16915v5](https://arxiv.org/pdf/2504.16915v5)**

> **作者:** Chong Mou; Yanze Wu; Wenxu Wu; Zinan Guo; Pengze Zhang; Yufeng Cheng; Yiming Luo; Fei Ding; Shiwen Zhang; Xinghui Li; Mengtian Li; Mingcong Liu; Yi Zhang; Shaojin Wu; Songtao Zhao; Jian Zhang; Qian He; Xinglong Wu
>
> **摘要:** Recently, extensive research on image customization (e.g., identity, subject, style, background, etc.) demonstrates strong customization capabilities in large-scale generative models. However, most approaches are designed for specific tasks, restricting their generalizability to combine different types of condition. Developing a unified framework for image customization remains an open challenge. In this paper, we present DreamO, an image customization framework designed to support a wide range of tasks while facilitating seamless integration of multiple conditions. Specifically, DreamO utilizes a diffusion transformer (DiT) framework to uniformly process input of different types. During training, we construct a large-scale training dataset that includes various customization tasks, and we introduce a feature routing constraint to facilitate the precise querying of relevant information from reference images. Additionally, we design a placeholder strategy that associates specific placeholders with conditions at particular positions, enabling control over the placement of conditions in the generated results. Moreover, we employ a progressive training strategy consisting of three stages: an initial stage focused on simple tasks with limited data to establish baseline consistency, a full-scale training stage to comprehensively enhance the customization capabilities, and a final quality alignment stage to correct quality biases introduced by low-quality data. Extensive experiments demonstrate that the proposed DreamO can effectively perform various image customization tasks with high quality and flexibly integrate different types of control conditions.
>
---
#### [replaced 052] Occlusion Boundary and Depth: Mutual Enhancement via Multi-Task Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21231v3](https://arxiv.org/pdf/2505.21231v3)**

> **作者:** Lintao Xu; Yinghao Wang; Chaohui Wang
>
> **备注:** WACV 2026
>
> **摘要:** Occlusion Boundary Estimation (OBE) identifies boundaries arising from both inter-object occlusions and self-occlusion within individual objects. This task is closely related to Monocular Depth Estimation (MDE), which infers depth from a single image, as Occlusion Boundaries (OBs) provide critical geometric cues for resolving depth ambiguities, while depth can conversely refine occlusion reasoning. In this paper, we aim to systematically model and exploit this mutually beneficial relationship. To this end, we propose MoDOT, a novel framework for joint estimation of depth and OBs, which incorporates a new Cross-Attention Strip Module (CASM) to leverage mid-level OB features for depth prediction, and a novel OB-Depth Constraint Loss (OBDCL) to enforce geometric consistency. To facilitate this study, we contribute OB-Hypersim, a large-scale photorealistic dataset with precise depth and self-occlusion-handled OB annotations. Extensive experiments on two synthetic datasets and NYUD-v2 demonstrate that MoDOT achieves significantly better performance than single-task baselines and multi-task competitors. Furthermore, models trained solely on our synthetic data demonstrate strong generalization to real-world scenes without fine-tuning, producing depth maps with sharper boundaries and improved geometric fidelity. Collectively, these results underscore the significant benefits of jointly modeling OBs and depth. Code and resources are available at https://github.com/xul-ops/MoDOT.
>
---
#### [replaced 053] Event Stream-based Sign Language Translation: A High-Definition Benchmark Dataset and A Novel Baseline
- **分类: cs.CV; cs.AI; cs.CL; cs.NE**

- **简介: 该论文聚焦于基于事件流的手语翻译任务，针对传统视觉方法受光照、快速动作和隐私影响的问题，构建了高分辨率事件数据集Event-CSL，并提出EvSLT框架。通过事件相机采集数据，结合时空特征融合与记忆聚合模块，显著提升翻译性能，推动了无障碍AI发展。**

- **链接: [https://arxiv.org/pdf/2408.10488v2](https://arxiv.org/pdf/2408.10488v2)**

> **作者:** Shiao Wang; Xiao Wang; Duoqing Yang; Yao Rong; Fuling Wang; Jianing Li; Lin Zhu; Bo Jiang
>
> **摘要:** Sign Language Translation (SLT) is a core task in the field of AI-assisted disability. Traditional SLT methods are typically based on visible light videos, which are easily affected by factors such as lighting variations, rapid hand movements, and privacy concerns. This paper proposes the use of bio-inspired event cameras to alleviate the aforementioned issues. Specifically, we introduce a new high-definition event-based sign language dataset, termed Event-CSL, which effectively addresses the data scarcity in this research area. The dataset comprises 14,827 videos, 14,821 glosses, and 2,544 Chinese words in the text vocabulary. These samples are collected across diverse indoor and outdoor scenes, covering multiple viewpoints, lighting conditions, and camera motions. We have also benchmarked existing mainstream SLT methods on this dataset to facilitate fair comparisons in future research.Furthermore, we propose a novel event-based sign language translation framework, termed EvSLT. The framework first segments continuous video features into clips and employs a Mamba-based memory aggregation module to compress and aggregate spatial detail features at the clip level. Subsequently, these spatial features, along with temporal representations obtained from temporal convolution, are then fused by a graph-guided spatiotemporal fusion module. Extensive experiments on Event-CSL, as well as other publicly available datasets, demonstrate the superior performance of our method. The dataset and source code will be released on https://github.com/Event-AHU/OpenESL
>
---
#### [replaced 054] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18734v2](https://arxiv.org/pdf/2511.18734v2)**

> **作者:** Keyang Lu; Sifan Zhou; Hongbin Xu; Gang Xu; Zhifei Yang; Yikai Wang; Zhen Xiao; Jieyi Long; Ming Li
>
> **备注:** 22 pages, 16 figures
>
> **摘要:** Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualize the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization, ensuring spatially coherent city growth. To comprehensively evaluate our method, we construct a diverse benchmark dataset and design six multi-dimensional metrics that assess generation quality from the perspectives of semantics, geometry, texture, and layout. Extensive experiments demonstrate that Yo'City consistently outperforms existing state-of-the-art methods across all evaluation aspects.
>
---
#### [replaced 055] AlignBench: Benchmarking Fine-Grained Image-Text Alignment with Synthetic Image-Caption Pairs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20515v2](https://arxiv.org/pdf/2511.20515v2)**

> **作者:** Kuniaki Saito; Risa Shinoda; Shohei Tanaka; Tosho Hirasawa; Fumio Okura; Yoshitaka Ushiku
>
> **备注:** Project Page: https://dahlian00.github.io/AlignBench/
>
> **摘要:** Assessing image-text alignment models such as CLIP is crucial for bridging visual and linguistic representations. Yet existing benchmarks rely on rule-based perturbations or short captions, limiting their ability to measure fine-grained alignment. We introduce AlignBench, a benchmark that provides a new indicator of image-text alignment by evaluating detailed image-caption pairs generated by diverse image-to-text and text-to-image models. Each sentence is annotated for correctness, enabling direct assessment of VLMs as alignment evaluators. Benchmarking a wide range of decoder-based VLMs reveals three key findings: (i) CLIP-based models, even those tailored for compositional reasoning, remain nearly blind; (ii) detectors systematically over-score early sentences; and (iii) they show strong self-preference, favoring their own outputs and harming detection performance. Our project page will be available at https://dahlian00.github.io/AlignBench/.
>
---
#### [replaced 056] Wavefront-Constrained Passive Obscured Object Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.20991v2](https://arxiv.org/pdf/2511.20991v2)**

> **作者:** Zhiwen Zheng; Yiwei Ouyang; Zhao Huang; Tao Zhang; Xiaoshuai Zhang; Huiyu Zhou; Wenwen Tang; Shaowei Jiang; Jin Liu; Xingru Huang
>
> **摘要:** Accurately localizing and segmenting obscured objects from faint light patterns beyond the field of view is highly challenging due to multiple scattering and medium-induced perturbations. Most existing methods, based on real-valued modeling or local convolutional operations, are inadequate for capturing the underlying physics of coherent light propagation. Moreover, under low signal-to-noise conditions, these methods often converge to non-physical solutions, severely compromising the stability and reliability of the observation. To address these challenges, we propose a novel physics-driven Wavefront Propagating Compensation Network (WavePCNet) to simulate wavefront propagation and enhance the perception of obscured objects. This WavePCNet integrates the Tri-Phase Wavefront Complex-Propagation Reprojection (TriWCP) to incorporate complex amplitude transfer operators to precisely constrain coherent propagation behavior, along with a momentum memory mechanism to effectively suppress the accumulation of perturbations. Additionally, a High-frequency Cross-layer Compensation Enhancement is introduced to construct frequency-selective pathways with multi-scale receptive fields and dynamically model structural consistency across layers, further boosting the model's robustness and interpretability under complex environmental conditions. Extensive experiments conducted on four physically collected datasets demonstrate that WavePCNet consistently outperforms state-of-the-art methods across both accuracy and robustness.
>
---
#### [replaced 057] Building temporally coherent 3D maps with VGGT for memory-efficient Semantic SLAM
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16282v2](https://arxiv.org/pdf/2511.16282v2)**

> **作者:** Gergely Dinya; Péter Halász; András Lőrincz; Kristóf Karacs; Anna Gelencsér-Horváth
>
> **摘要:** We present a fast, spatio-temporal scene understanding framework based on Visual Geometry Grounded Transformer (VGGT). The proposed pipeline is designed to enable efficient, close to real-time performance, supporting applications including assistive navigation. To achieve continuous updates of the 3D scene representation, we process the image flow with a sliding window, aligning submaps, thereby overcoming VGGT's high memory demands. We exploit the VGGT tracking head to aggregate 2D semantic instance masks into 3D objects. To allow for temporal consistency and richer contextual reasoning the system stores timestamps and instance-level identities, thereby enabling the detection of changes in the environment. We evaluate the approach on well-known benchmarks and custom datasets specifically designed for assistive navigation scenarios. The results demonstrate the applicability of the framework to real-world scenarios.
>
---
#### [replaced 058] Enhancing Descriptive Image Quality Assessment with A Large-scale Multi-modal Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.18842v3](https://arxiv.org/pdf/2405.18842v3)**

> **作者:** Zhiyuan You; Jinjin Gu; Xin Cai; Zheyuan Li; Kaiwen Zhu; Chao Dong; Tianfan Xue
>
> **备注:** Accepted by TIP
>
> **摘要:** With the rapid advancement of Vision Language Models (VLMs), VLM-based Image Quality Assessment (IQA) seeks to describe image quality linguistically to align with human expression and capture the multifaceted nature of IQA tasks. However, current methods are still far from practical usage. First, prior works focus narrowly on specific sub-tasks or settings, which do not align with diverse real-world applications. Second, their performance is sub-optimal due to limitations in dataset coverage, scale, and quality. To overcome these challenges, we introduce the enhanced Depicted image Quality Assessment model (DepictQA-Wild). Our method includes a multi-functional IQA task paradigm that encompasses both assessment and comparison tasks, brief and detailed responses, full-reference and non-reference scenarios. We introduce a ground-truth-informed dataset construction approach to enhance data quality, and scale up the dataset to 495K under the brief-detail joint framework. Consequently, we construct a comprehensive, large-scale, and high-quality dataset, named DQ-495K. We also retain image resolution during training to better handle resolution-related quality issues, and estimate a confidence score that is helpful to filter out low-quality responses. Experimental results demonstrate that DepictQA-Wild significantly outperforms traditional score-based methods, prior VLM-based IQA models, and proprietary GPT-4V in distortion identification, instant rating, and reasoning tasks. Our advantages are further confirmed by real-world applications including assessing the web-downloaded images and ranking model-processed images. Codes, datasets, and model weights have been released in https://depictqa.github.io/.
>
---
#### [replaced 059] Fast Equivariant Imaging: Acceleration for Unsupervised Learning via Augmented Lagrangian and Auxiliary PnP Denoisers
- **分类: eess.IV; cs.CV; cs.LG; math.OC**

- **链接: [https://arxiv.org/pdf/2507.06764v3](https://arxiv.org/pdf/2507.06764v3)**

> **作者:** Guixian Xu; Jinglai Li; Junqi Tang
>
> **备注:** 17 pages
>
> **摘要:** In this work, we propose Fast Equivariant Imaging (FEI), a novel unsupervised learning framework to rapidly and efficiently train deep imaging networks without ground-truth data. From the perspective of reformulating the Equivariant Imaging based optimization problem via the method of Lagrange multipliers and utilizing plug-and-play denoisers, this novel unsupervised scheme shows superior efficiency and performance compared to the vanilla Equivariant Imaging paradigm. In particular, our FEI schemes achieve an order-of-magnitude (10x) acceleration over standard EI on training U-Net for X-ray CT reconstruction and image inpainting, with improved generalization performance.
>
---
#### [replaced 060] MambaX-Net: Dual-Input Mamba-Enhanced Cross-Attention Network for Longitudinal MRI Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.17529v2](https://arxiv.org/pdf/2510.17529v2)**

> **作者:** Yovin Yahathugoda; Davide Prezzi; Piyalitt Ittichaiwong; Vicky Goh; Sebastien Ourselin; Michela Antonelli
>
> **备注:** Updated the acknowledgments section to include the UKRI Open Access statement
>
> **摘要:** Active Surveillance (AS) is a treatment option for managing low and intermediate-risk prostate cancer (PCa), aiming to avoid overtreatment while monitoring disease progression through serial MRI and clinical follow-up. Accurate prostate segmentation is an important preliminary step for automating this process, enabling automated detection and diagnosis of PCa. However, existing deep-learning segmentation models are often trained on single-time-point and expertly annotated datasets, making them unsuitable for longitudinal AS analysis, where multiple time points and a scarcity of expert labels hinder their effective fine-tuning. To address these challenges, we propose MambaX-Net, a novel semi-supervised, dual-scan 3D segmentation architecture that computes the segmentation for time point t by leveraging the MRI and the corresponding segmentation mask from the previous time point. We introduce two new components: (i) a Mamba-enhanced Cross-Attention Module, which integrates the Mamba block into cross attention to efficiently capture temporal evolution and long-range spatial dependencies, and (ii) a Shape Extractor Module that encodes the previous segmentation mask into a latent anatomical representation for refined zone delination. Moreover, we introduce a semi-supervised self-training strategy that leverages pseudo-labels generated from a pre-trained nnU-Net, enabling effective learning without expert annotations. MambaX-Net was evaluated on a longitudinal AS dataset, and results showed that it significantly outperforms state-of-the-art U-Net and Transformer-based models, achieving superior prostate zone segmentation even when trained on limited and noisy data.
>
---
#### [replaced 061] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01873v2](https://arxiv.org/pdf/2508.01873v2)**

> **作者:** Siran Peng; Haoyuan Zhang; Li Gao; Tianshuo Zhang; Xiangyu Zhu; Bao Li; Weisong Zhao; Zhen Lei
>
> **摘要:** The rapid evolution of deepfake technologies demands robust and reliable face forgery detection algorithms. While determining whether an image has been manipulated remains essential, the ability to precisely localize forgery clues is also important for enhancing model explainability and building user trust. To address this dual challenge, we introduce DiffusionFF, a diffusion-based framework that simultaneously performs face forgery detection and fine-grained artifact localization. Our key idea is to establish a novel encoder-decoder architecture: a pretrained forgery detector serves as a powerful "artifact encoder", and a denoising diffusion model is repurposed as an "artifact decoder". Conditioned on multi-scale forgery-related features extracted by the encoder, the decoder progressively synthesizes a detailed artifact localization map. We then fuse this fine-grained localization map with high-level semantic features from the forgery detector, leading to substantial improvements in detection capability. Extensive experiments show that DiffusionFF achieves state-of-the-art (SOTA) performance across multiple benchmarks, underscoring its superior effectiveness and explainability.
>
---
#### [replaced 062] Source-free Video Domain Adaptation by Learning from Noisy Labels
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.18572v2](https://arxiv.org/pdf/2311.18572v2)**

> **作者:** Avijit Dasgupta; C. V. Jawahar; Karteek Alahari
>
> **备注:** Our extended ICVGIP paper is now accepted in Pattern Recognition
>
> **摘要:** Despite the progress seen in classification methods, current approaches for handling videos with distribution shifts in source and target domains remain source-dependent as they require access to the source data during the adaptation stage. In this paper, we present a self-training based source-free video domain adaptation approach to address this challenge by bridging the gap between the source and the target domains. We use the source pre-trained model to generate pseudo-labels for the target domain samples, which are inevitably noisy. Thus, we treat the problem of source-free video domain adaptation as learning from noisy labels and argue that the samples with correct pseudo-labels can help us in adaptation. To this end, we leverage the cross-entropy loss as an indicator of the correctness of the pseudo-labels and use the resulting small-loss samples from the target domain for fine-tuning the model. We further enhance the adaptation performance by implementing a teacher-student (TS) framework, in which the teacher, which is updated gradually, produces reliable pseudo-labels. Meanwhile, the student undergoes fine-tuning on the target domain videos using these generated pseudo-labels to improve its performance. Extensive experimental evaluations show that our methods, termed as CleanAdapt, CleanAdapt + TS, achieve state-of-the-art results, outperforming the existing approaches on various open datasets. Our source code is publicly available at https://avijit9.github.io/CleanAdapt.
>
---
#### [replaced 063] Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对数字孪生中3D场景重建任务，解决传统LiDAR-camera融合方法依赖复杂校准、难以表征玻璃等材料的问题。提出仅用摄像头的重建方法：基于多视角图像使用高斯溅射建模，结合视觉模型提取材质掩码，将材质标签投影至网格，并赋予物理材质属性，实现高保真传感器模拟。**

- **链接: [https://arxiv.org/pdf/2511.20348v2](https://arxiv.org/pdf/2511.20348v2)**

> **作者:** Andy Huynh; João Malheiro Silva; Holger Caesar; Tong Duy Son
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication. Revised version (v2) to correct author order
>
> **摘要:** 3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
>
---
#### [replaced 064] Hybrid Rendering for Multimodal Autonomous Driving: Merging Neural and Physics-Based Simulation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09464v2](https://arxiv.org/pdf/2503.09464v2)**

> **作者:** Máté Tóth; Péter Kovács; Réka Bencses; Zoltán Bendefy; Zoltán Hortsin; Balázs Teréki; Tamás Matuszka
>
> **摘要:** Neural reconstruction models for autonomous driving simulation have made significant strides in recent years, with dynamic models becoming increasingly prevalent. However, these models are typically limited to handling in-domain objects closely following their original trajectories. We introduce a hybrid approach that combines the strengths of neural reconstruction with physics-based rendering. This method enables the virtual placement of traditional mesh-based dynamic agents at arbitrary locations, adjustments to environmental conditions, and rendering from novel camera viewpoints. Our approach significantly enhances novel view synthesis quality -- especially for road surfaces and lane markings -- while maintaining interactive frame rates through our novel training method, NeRF2GS. This technique leverages the superior generalization capabilities of NeRF-based methods and the real-time rendering speed of 3D Gaussian Splatting (3DGS). We achieve this by training a customized NeRF model on the original images with depth regularization derived from a noisy LiDAR point cloud, then using it as a teacher model for 3DGS training. This process ensures accurate depth, surface normals, and camera appearance modeling as supervision. With our block-based training parallelization, the method can handle large-scale reconstructions (greater than or equal to 100,000 square meters) and predict segmentation masks, surface normals, and depth maps. During simulation, it supports a rasterization-based rendering backend with depth-based composition and multiple camera models for real-time camera simulation, as well as a ray-traced backend for precise LiDAR simulation.
>
---
#### [replaced 065] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16901v2](https://arxiv.org/pdf/2511.16901v2)**

> **作者:** Lu Zhu; Tiantian Geng; Yangye Chen; Teng Wang; Ping Lu; Feng Zheng
>
> **备注:** Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST
>
> **摘要:** Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.
>
---
#### [replaced 066] ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19145v3](https://arxiv.org/pdf/2511.19145v3)**

> **作者:** Dongha Lee; Jinhee Park; Minjun Kim; Junseok Kwon
>
> **备注:** 16 pages, 5 figures, under review
>
> **摘要:** We propose Activation Boundary Matching for Low-Rank Adaptation (ABM-LoRA), a principled initialization strategy that substantially accelerates the convergence of low-rank adapters. While LoRA offers high parameter efficiency, its random initialization restricts gradient updates to a mismatched tangent space, causing significant information loss and hindering early convergence. Our ABM-LoRA addresses this by aligning the adapter's activation boundaries with those of the pretrained model before downstream training, thereby maximizing the projection of full-parameter gradients into the adapter subspace. This alignment sharply reduces information loss at initialization, yields a lower starting loss, and accelerates convergence. We demonstrate ABM-LoRA's effectiveness across diverse architectures and tasks: language understanding (T5-Base on GLUE), dialogue generation (LLaMA2-7B on WizardLM), and vision recognition (ViT-B/16 on VTAB-1K). On VTAB-1K, it achieves the highest accuracy among all methods, with strong gains on structured reasoning tasks requiring geometric understanding.
>
---
#### [replaced 067] Infrared and Visible Image Fusion with Language-Driven Loss in CLIP Embedding Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2402.16267v3](https://arxiv.org/pdf/2402.16267v3)**

> **作者:** Yuhao Wang; Lingjuan Miao; Zhiqiang Zhou; Lei Zhang; Yajun Qiao
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Infrared-visible image fusion (IVIF) has attracted much attention owing to the highly-complementary properties of the two image modalities. Due to the lack of ground-truth fused images, the fusion output of current deep-learning based methods heavily depends on the loss functions defined mathematically. As it is hard to well mathematically define the fused image without ground truth, the performance of existing fusion methods is limited. In this paper, we propose to use natural language to express the objective of IVIF, which can avoid the explicit mathematical modeling of fusion output in current losses, and make full use of the advantage of language expression to improve the fusion performance. For this purpose, we present a comprehensive language-expressed fusion objective, and encode relevant texts into the multi-modal embedding space using CLIP. A language-driven fusion model is then constructed in the embedding space, by establishing the relationship among the embedded vectors representing the fusion objective and input image modalities. Finally, a language-driven loss is derived to make the actual IVIF aligned with the embedded language-driven fusion model via supervised training. Experiments show that our method can obtain much better fusion results than existing techniques. The code is available at https://github.com/wyhlaowang/LDFusion.
>
---
#### [replaced 068] CNN-LSTM Hybrid Architecture for Over-the-Air Automatic Modulation Classification Using SDR
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21040v2](https://arxiv.org/pdf/2511.21040v2)**

> **作者:** Dinanath Padhya; Krishna Acharya; Bipul Kumar Dahal; Dinesh Baniya Kshatri
>
> **备注:** 7 Pages, 11 figures, 2 Tables, Accepted in Journal (Journal of Innovations in Engineering Education)
>
> **摘要:** Automatic Modulation Classification (AMC) is a core technology for future wireless communication systems, enabling the identification of modulation schemes without prior knowledge. This capability is essential for applications in cognitive radio, spectrum monitoring, and intelligent communication networks. We propose an AMC system based on a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture, integrated with a Software Defined Radio (SDR) platform. The proposed architecture leverages CNNs for spatial feature extraction and LSTMs for capturing temporal dependencies, enabling efficient handling of complex, time-varying communication signals. The system's practical ability was demonstrated by identifying over-the-air (OTA) signals from a custom-built FM transmitter alongside other modulation schemes. The system was trained on a hybrid dataset combining the RadioML2018 dataset with a custom-generated dataset, featuring samples at Signal-to-Noise Ratios (SNRs) from 0 to 30dB. System performance was evaluated using accuracy, precision, recall, F1 score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). The optimized model achieved 93.48% accuracy, 93.53% precision, 93.48% recall, and an F1 score of 93.45%. The AUC-ROC analysis confirmed the model's discriminative power, even in noisy conditions. This paper's experimental results validate the effectiveness of the hybrid CNN-LSTM architecture for AMC, suggesting its potential application in adaptive spectrum management and advanced cognitive radio systems.
>
---
#### [replaced 069] AgriPotential: A Novel Multi-Spectral and Multi-Temporal Remote Sensing Dataset for Agricultural Potentials
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.11740v2](https://arxiv.org/pdf/2506.11740v2)**

> **作者:** Mohammad El Sakka; Caroline De Pourtales; Lotfi Chaari; Josiane Mothe
>
> **备注:** Accepted at CBMI 2025
>
> **摘要:** Remote sensing has emerged as a critical tool for large-scale Earth monitoring and land management. In this paper, we introduce AgriPotential, a novel benchmark dataset composed of Sentinel-2 satellite imagery captured over multiple months. The dataset provides pixel-level annotations of agricultural potentials for three major crop types - viticulture, market gardening, and field crops - across five ordinal classes. AgriPotential supports a broad range of machine learning tasks, including ordinal regression, multi-label classification, and spatio-temporal modeling. The data cover diverse areas in Southern France, offering rich spectral information. AgriPotential is the first public dataset designed specifically for agricultural potential prediction, aiming to improve data-driven approaches to sustainable land use planning. The dataset and the code are freely accessible at: https://zenodo.org/records/15551829
>
---
#### [replaced 070] VITA: Zero-Shot Value Functions via Test-Time Adaptation of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.10085v4](https://arxiv.org/pdf/2506.10085v4)**

> **作者:** Christos Ziakas; Alessandra Russo
>
> **摘要:** Vision-Language Models (VLMs) show promise as zero-shot goal-conditioned value functions, but their frozen pre-trained representations limit generalization and temporal reasoning. We introduce VITA, a zero-shot value function learning method that enhances both capabilities via test-time adaptation. At inference, a lightweight adaptation module is updated via a gradient step on a meta-learned self-supervised loss, such that each test-time update improves value estimation. By updating sequentially over a trajectory, VITA encodes history into its parameters, addressing the temporal reasoning limitations. To mitigate shortcut learning, we propose a dissimilarity-based sampling strategy that selects semantically diverse segments of the trajectory during training. In real-world robotic manipulation tasks, VITA generalizes from a single training environment to diverse out-of-distribution tasks, environments, and embodiments, outperforming the state-of-the-art zero-shot method using autoregressive VLMs. Furthermore, we demonstrate that VITA's zero-shot value estimates can be utilized for reward shaping in offline reinforcement learning, resulting in multi-task policies on the Meta-World benchmark that exceed the performance of those trained with the simulation's fuzzy-logic dense rewards.
>
---
#### [replaced 071] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **简介: 该论文聚焦多模态大模型的空间智能（SI）评估，提出EASI框架，统一现有与新构建的时空任务基准。通过在八项基准上超十亿令牌的实测，揭示当前顶尖模型（如GPT-5）虽强但仍远逊于人类，且非开源模型无显著优势。研究开放代码与排行榜，推动可复现、持续更新的SI评估。**

- **链接: [https://arxiv.org/pdf/2508.13142v4](https://arxiv.org/pdf/2508.13142v4)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
>
---
#### [replaced 072] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10013v2](https://arxiv.org/pdf/2511.10013v2)**

> **作者:** Shufeng Kong; Zijie Wang; Nuan Cui; Hao Tang; Yihan Meng; Yuanyuan Wei; Feifan Chen; Yingheng Wang; Zhuo Cai; Yaonan Wang; Yulong Zhang; Yuzheng Li; Zibin Zheng; Caihua Liu; Hao Liang
>
> **备注:** To appear at AAAI-26
>
> **摘要:** Automated interpretation of medical images demands robust modeling of complex visual-semantic relationships while addressing annotation scarcity, label imbalance, and clinical plausibility constraints. We introduce MIRNet (Medical Image Reasoner Network), a novel framework that integrates self-supervised pre-training with constrained graph-based reasoning. Tongue image diagnosis is a particularly challenging domain that requires fine-grained visual and semantic understanding. Our approach leverages self-supervised masked autoencoder (MAE) to learn transferable visual representations from unlabeled data; employs graph attention networks (GAT) to model label correlations through expert-defined structured graphs; enforces clinical priors via constraint-aware optimization using KL divergence and regularization losses; and mitigates imbalance using asymmetric loss (ASL) and boosting ensembles. To address annotation scarcity, we also introduce TongueAtlas-4K, a comprehensive expert-curated benchmark comprising 4,000 images annotated with 22 diagnostic labels--representing the largest public dataset in tongue analysis. Validation shows our method achieves state-of-the-art performance. While optimized for tongue diagnosis, the framework readily generalizes to broader diagnostic medical imaging tasks.
>
---
#### [replaced 073] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10555v5](https://arxiv.org/pdf/2511.10555v5)**

> **作者:** Huijie Liu; Shuhao Cui; Haoxiang Cao; Shuai Ma; Kai Wu; Guoliang Kang
>
> **备注:** Code: https://github.com/Kwai-Kolors/CoTyle Demo: https://huggingface.co/spaces/Kwai-Kolors/CoTyle Homepage: https://kwai-kolors.github.io/CoTyle/
>
> **摘要:** Innovative visual stylization is a cornerstone of artistic creation, yet generating novel and consistent visual styles remains a significant challenge. Existing generative approaches typically rely on lengthy textual prompts, reference images, or parameter-efficient fine-tuning to guide style-aware image generation, but often struggle with style consistency, limited creativity, and complex style representations. In this paper, we affirm that a style is worth one numerical code by introducing the novel task, code-to-style image generation, which produces images with novel, consistent visual styles conditioned solely on a numerical style code. To date, this field has only been primarily explored by the industry (e.g., Midjourney), with no open-source research from the academic community. To fill this gap, we propose CoTyle, the first open-source method for this task. Specifically, we first train a discrete style codebook from a collection of images to extract style embeddings. These embeddings serve as conditions for a text-to-image diffusion model (T2I-DM) to generate stylistic images. Subsequently, we train an autoregressive style generator on the discrete style embeddings to model their distribution, allowing the synthesis of novel style embeddings. During inference, a numerical style code is mapped to a unique style embedding by the style generator, and this embedding guides the T2I-DM to generate images in the corresponding style. Unlike existing methods, our method offers unparalleled simplicity and diversity, unlocking a vast space of reproducible styles from minimal input. Extensive experiments validate that CoTyle effectively turns a numerical code into a style controller, demonstrating a style is worth one code.
>
---
#### [replaced 074] RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.17045v2](https://arxiv.org/pdf/2511.17045v2)**

> **作者:** Linfeng Dong; Yuchen Yang; Hao Wu; Wei Wang; Yuenan Hou; Zhihang Zhong; Xiao Sun
>
> **备注:** Accepted to AAAI 2026 (Oral)
>
> **摘要:** We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at https://github.com/OrcustD/RacketVision
>
---
#### [replaced 075] Mavors: Multi-granularity Video Representation for Multimodal Large Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对多模态大模型中的长视频理解任务，解决现有方法在处理复杂视频时信息丢失的问题。提出Mavors框架，通过多粒度视频表示，结合高分辨率空间编码与跨片段时序建模，有效保留细粒度时空特征，提升长视频理解性能。**

- **链接: [https://arxiv.org/pdf/2504.10068v2](https://arxiv.org/pdf/2504.10068v2)**

> **作者:** Yang Shi; Jiaheng Liu; Yushuo Guan; Zhenhua Wu; Yuanxing Zhang; Zihao Wang; Weihong Lin; Jingyun Hua; Zekun Wang; Xinlong Chen; Bohan Zeng; Wentao Zhang; Fuzheng Zhang; Wenjing Yang; Di Zhang
>
> **备注:** 22 pages
>
> **摘要:** Long-context video understanding in multimodal large language models (MLLMs) faces a critical challenge: balancing computational efficiency with the retention of fine-grained spatio-temporal patterns. Existing approaches (e.g., sparse sampling, dense sampling with low resolution, and token compression) suffer from significant information loss in temporal dynamics, spatial details, or subtle interactions, particularly in videos with complex motion or varying resolutions. To address this, we propose $\mathbf{Mavors}$, a novel framework that introduces $\mathbf{M}$ulti-gr$\mathbf{a}$nularity $\mathbf{v}$ide$\mathbf{o}$ $\mathbf{r}$epre$\mathbf{s}$entation for holistic long-video modeling. Specifically, Mavors directly encodes raw video content into latent representations through two core components: 1) an Intra-chunk Vision Encoder (IVE) that preserves high-resolution spatial features via 3D convolutions and Vision Transformers, and 2) an Inter-chunk Feature Aggregator (IFA) that establishes temporal coherence across chunks using transformer-based dependency modeling with chunk-level rotary position encodings. Moreover, the framework unifies image and video understanding by treating images as single-frame videos via sub-image decomposition. Experiments across diverse benchmarks demonstrate Mavors' superiority in maintaining both spatial fidelity and temporal continuity, significantly outperforming existing methods in tasks requiring fine-grained spatio-temporal reasoning.
>
---
#### [replaced 076] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20295v3](https://arxiv.org/pdf/2509.20295v3)**

> **作者:** Xichen Xu; Yanshu Wang; Jinbao Wang; Xiaoning Lei; Guoyang Xie; Guannan Jiang; Zhichao Lu
>
> **摘要:** Industrial anomaly segmentation relies heavily on pixel-level annotations, yet real-world anomalies are often scarce, diverse, and costly to label. Segmentation-oriented industrial anomaly synthesis (SIAS) has emerged as a promising alternative; however, existing methods struggle to balance sampling efficiency and generation quality. Moreover, most approaches treat all spatial regions uniformly, overlooking the distinct statistical differences between anomaly and background areas. This uniform treatment hinders the synthesis of controllable, structure-specific anomalies tailored for segmentation tasks. In this paper, we propose FAST, a foreground-aware diffusion framework featuring two novel modules: the Anomaly-Informed Accelerated Sampling (AIAS) and the Foreground-Aware Reconstruction Module (FARM). AIAS is a training-free sampling algorithm specifically designed for segmentation-oriented industrial anomaly synthesis, which accelerates the reverse process through coarse-to-fine aggregation and enables the synthesis of state-of-the-art segmentation-oriented anomalies in as few as 10 steps. Meanwhile, FARM adaptively adjusts the anomaly-aware noise within the masked foreground regions at each sampling step, preserving localized anomaly signals throughout the denoising trajectory. Extensive experiments on multiple industrial benchmarks demonstrate that FAST consistently outperforms existing anomaly synthesis methods in downstream segmentation tasks. We release the code at: https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis.
>
---
#### [replaced 077] SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中视觉表示效率与可解释性不足的问题，提出基于对象-关系的紧凑表征。构建LIBERO+数据集，提供细粒度对象标注；设计SlotVLA框架，利用槽注意力实现对象与关系的联合建模，显著减少视觉令牌数，提升泛化能力，推动可解释的多任务机器人操作。**

- **链接: [https://arxiv.org/pdf/2511.06754v2](https://arxiv.org/pdf/2511.06754v2)**

> **作者:** Taisei Hanyu; Nhat Chung; Huy Le; Toan Nguyen; Yuki Ikebe; Anthony Gunderman; Duy Nguyen Ho Minh; Khoa Vo; Tung Kieu; Kashu Yamazaki; Chase Rainwater; Anh Nguyen; Ngan Le
>
> **备注:** under review
>
> **摘要:** Inspired by how humans reason over discrete objects and their relationships, we explore whether compact object-centric and object-relation representations can form a foundation for multitask robotic manipulation. Most existing robotic multitask models rely on dense embeddings that entangle both object and background cues, raising concerns about both efficiency and interpretability. In contrast, we study object-relation-centric representations as a pathway to more structured, efficient, and explainable visuomotor control. Our contributions are two-fold. First, we introduce LIBERO+, a fine-grained benchmark dataset designed to enable and evaluate object-relation reasoning in robotic manipulation. Unlike prior datasets, LIBERO+ provides object-centric annotations that enrich demonstrations with box- and mask-level labels as well as instance-level temporal tracking, supporting compact and interpretable visuomotor representations. Second, we propose SlotVLA, a slot-attention-based framework that captures both objects and their relations for action decoding. It uses a slot-based visual tokenizer to maintain consistent temporal object representations, a relation-centric decoder to produce task-relevant embeddings, and an LLM-driven module that translates these embeddings into executable actions. Experiments on LIBERO+ demonstrate that object-centric slot and object-relation slot representations drastically reduce the number of required visual tokens, while providing competitive generalization. Together, LIBERO+ and SlotVLA provide a compact, interpretable, and effective foundation for advancing object-relation-centric robotic manipulation.
>
---
#### [replaced 078] ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18192v2](https://arxiv.org/pdf/2511.18192v2)**

> **作者:** Ahmad Mohammadshirazi; Pinaki Prasad Guha Neogi; Dheeraj Kulshrestha; Rajiv Ramnath
>
> **摘要:** Document Visual Question Answering (VQA) requires models to not only extract accurate textual answers but also precisely localize them within document images, a capability critical for interpretability in high-stakes applications. However, existing systems achieve strong textual accuracy while producing unreliable spatial grounding, or sacrifice performance for interpretability. We present ARIAL (Agentic Reasoning for Interpretable Answer Localization), a modular framework that orchestrates specialized tools through an LLM-based planning agent to achieve both precise answer extraction and reliable spatial grounding. ARIAL decomposes Document VQA into structured subtasks: OCR-based text extraction with TrOCR, retrieval-augmented context selection using semantic search, answer generation via a fine-tuned Gemma 3-27B model, and explicit bounding-box localization through text-to-region alignment. This modular architecture produces transparent reasoning traces, enabling tool-level auditability and independent component optimization. We evaluate ARIAL on four benchmarks (DocVQA, FUNSD, CORD, and SROIE) using both textual accuracy (ANLS) and spatial precision (mAP at IoU 0.50 to 0.95). ARIAL achieves state-of-the-art results across all datasets: 88.7 ANLS and 50.1 mAP on DocVQA, 90.0 ANLS and 50.3 mAP on FUNSD, 85.5 ANLS and 60.2 mAP on CORD, and 93.1 ANLS on SROIE, surpassing the previous best method (DLaVA) by +2.8 ANLS and +3.9 mAP on DocVQA. Our work demonstrates how agentic orchestration of specialized tools can simultaneously improve performance and interpretability, providing a pathway toward trustworthy, explainable document AI systems.
>
---
#### [replaced 079] Exploring Convolutional Neural Networks for Rice Grain Classification: An Explainable AI Approach
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.05513v5](https://arxiv.org/pdf/2505.05513v5)**

> **作者:** Muhammad Junaid Asif; Hamza Khan; Rabia Tehseen; Rana Fayyaz Ahmad; Mujtaba Asad; Syed Tahir Hussain Rizvi; Shazia Saqib
>
> **摘要:** Rice is an essential staple food worldwide that is important in promoting international trade, economic growth, and nutrition. Asian countries such as China, India, Pakistan, Thailand, Vietnam, and Indonesia are notable for their significant contribution to the cultivation and utilization of rice. These nations are also known for cultivating different rice grains, including short and long grains. These sizes are further classified as basmati, jasmine, kainat saila, ipsala, arborio, etc., catering to diverse culinary preferences and cultural traditions. For both local and international trade, inspecting and maintaining the quality of rice grains to satisfy customers and preserve a country's reputation is necessary. Manual quality check and classification is quite a laborious and time-consuming process. It is also highly prone to mistakes. Therefore, an automatic solution must be proposed for the effective and efficient classification of different varieties of rice grains. This research paper presents an automatic framework based on a convolutional neural network (CNN) for classifying different varieties of rice grains. We evaluated the proposed model based on performance metrics such as accuracy, recall, precision, and F1-Score. The CNN model underwent rigorous training and validation, achieving a remarkable accuracy rate and a perfect area under each class's Receiver Operating Characteristic (ROC) curve. The confusion matrix analysis confirmed the model's effectiveness in distinguishing between the different rice varieties, indicating minimal misclassifications. Additionally, the integration of explainability techniques such as LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provided valuable insights into the model's decision-making process, revealing how specific features of the rice grains influenced classification outcomes.
>
---
#### [replaced 080] A Survey on Personalized Content Synthesis with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.05538v5](https://arxiv.org/pdf/2405.05538v5)**

> **作者:** Xulu Zhang; Xiaoyong Wei; Wentao Hu; Jinlin Wu; Jiaxin Wu; Wengyu Zhang; Zhaoxiang Zhang; Zhen Lei; Qing Li
>
> **摘要:** Recent advancements in diffusion models have significantly impacted content creation, leading to the emergence of Personalized Content Synthesis (PCS). By utilizing a small set of user-provided examples featuring the same subject, PCS aims to tailor this subject to specific user-defined prompts. Over the past two years, more than 150 methods have been introduced in this area. However, existing surveys primarily focus on text-to-image generation, with few providing up-to-date summaries on PCS. This paper provides a comprehensive survey of PCS, introducing the general frameworks of PCS research, which can be categorized into test-time fine-tuning (TTF) and pre-trained adaptation (PTA) approaches. We analyze the strengths, limitations, and key techniques of these methodologies. Additionally, we explore specialized tasks within the field, such as object, face, and style personalization, while highlighting their unique challenges and innovations. Despite the promising progress, we also discuss ongoing challenges, including overfitting and the trade-off between subject fidelity and text alignment. Through this detailed overview and analysis, we propose future directions to further the development of PCS.
>
---
#### [replaced 081] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17138v3](https://arxiv.org/pdf/2511.17138v3)**

> **作者:** Yushun Fang; Yuxiang Chen; Shibo Yin; Qiang Hu; Jiangchao Yao; Ya Zhang; Xiaoyun Zhang; Yanfeng Wang
>
> **摘要:** Recent advances in diffusion-based real-world image super-resolution (Real-ISR) have demonstrated remarkable perceptual quality, yet the balance between fidelity and controllability remains a problem: multi-step diffusion-based methods suffer from generative diversity and randomness, resulting in low fidelity, while one-step methods lose control flexibility due to fidelity-specific finetuning. In this paper, we present ODTSR, a one-step diffusion transformer based on Qwen-Image that performs Real-ISR considering fidelity and controllability simultaneously: a newly introduced visual stream receives low-quality images (LQ) with adjustable noise (Control Noise), and the original visual stream receives LQs with consistent noise (Prior Noise), forming the Noise-hybrid Visual Stream (NVS) design. ODTSR further employs Fidelity-aware Adversarial Training (FAA) to enhance controllability and achieve one-step inference. Extensive experiments demonstrate that ODTSR not only achieves state-of-the-art (SOTA) performance on generic Real-ISR, but also enables prompt controllability on challenging scenarios such as real-world scene text image super-resolution (STISR) of Chinese characters without training on specific datasets. Codes are available at https://github.com/RedMediaTech/ODTSR.
>
---
#### [replaced 082] GraspDiffusion: Synthesizing Realistic Whole-body Hand-Object Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.13911v3](https://arxiv.org/pdf/2410.13911v3)**

> **作者:** Patrick Kwon; Chen Chen; Hanbyul Joo
>
> **备注:** Paper has been accepted to WACV 2026
>
> **摘要:** Recent generative models can synthesize high-quality images, but they often fail to generate humans interacting with objects using their hands. This arises mostly from the model's misunderstanding of such interactions and the hardships of synthesizing intricate regions of the body. In this paper, we propose \textbf{GraspDiffusion}, a novel generative method that creates realistic scenes of human-object interaction. Given a 3D object, GraspDiffusion constructs whole-body poses with control over the object's location relative to the human body, which is achieved by separately leveraging the generative priors for body and hand poses, optimizing them into a joint grasping pose. This pose guides the image synthesis to correctly reflect the intended interaction, creating realistic and diverse human-object interaction scenes. We demonstrate that GraspDiffusion can successfully tackle the relatively uninvestigated problem of generating full-bodied human-object interactions while outperforming previous methods. Our project page is available at https://yj7082126.github.io/graspdiffusion/
>
---
#### [replaced 083] Qwen3-VL Technical Report
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.21631v2](https://arxiv.org/pdf/2511.21631v2)**

> **作者:** Shuai Bai; Yuxuan Cai; Ruizhe Chen; Keqin Chen; Xionghui Chen; Zesen Cheng; Lianghao Deng; Wei Ding; Chang Gao; Chunjiang Ge; Wenbin Ge; Zhifang Guo; Qidong Huang; Jie Huang; Fei Huang; Binyuan Hui; Shutong Jiang; Zhaohai Li; Mingsheng Li; Mei Li; Kaixin Li; Zicheng Lin; Junyang Lin; Xuejing Liu; Jiawei Liu; Chenglong Liu; Yang Liu; Dayiheng Liu; Shixuan Liu; Dunjie Lu; Ruilin Luo; Chenxu Lv; Rui Men; Lingchen Meng; Xuancheng Ren; Xingzhang Ren; Sibo Song; Yuchong Sun; Jun Tang; Jianhong Tu; Jianqiang Wan; Peng Wang; Pengfei Wang; Qiuyue Wang; Yuxuan Wang; Tianbao Xie; Yiheng Xu; Haiyang Xu; Jin Xu; Zhibo Yang; Mingkun Yang; Jianxin Yang; An Yang; Bowen Yu; Fei Zhang; Hang Zhang; Xi Zhang; Bo Zheng; Humen Zhong; Jingren Zhou; Fan Zhou; Jing Zhou; Yuanzhi Zhu; Ke Zhu
>
> **备注:** 42 pages
>
> **摘要:** We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse latency-quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger pure-text understanding, surpassing comparable text-only backbones in several cases; (ii) robust long-context comprehension with a native 256K-token window for both text and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-referencing across long documents and videos; and (iii) advanced multimodal reasoning across single-image, multi-image, and video tasks, demonstrating leading performance on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., MathVista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced interleaved-MRoPE for stronger spatial-temporal modeling across images and video; (ii) DeepStack integration, which effectively leverages multi-level ViT features to tighten vision-language alignment; and (iii) text-based time alignment for video, evolving from T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. Under comparable token budgets and latency constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for image-grounded reasoning, agentic decision-making, and multimodal code intelligence in real-world workflows.
>
---
#### [replaced 084] ArtiBench and ArtiBrain: Benchmarking Generalizable Vision-Language Articulated Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉-语言引导的可动物体操作中的泛化难题，提出ArtiBench基准与ArtiBrain框架。任务为跨类别、跨实例的长程多步操作。工作包括构建多环境、多层级评估基准，设计融合高层推理与自适应控制的模块化系统，并通过显式记忆传播可操作性知识，显著提升泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.20330v2](https://arxiv.org/pdf/2511.20330v2)**

> **作者:** Yuhan Wu; Tiantian Wei; Shuo Wang; ZhiChao Wang; Yanyong Zhang; Daniel Cremers; Yan Xia
>
> **摘要:** Interactive articulated manipulation requires long-horizon, multi-step interactions with appliances while maintaining physical consistency. Existing vision-language and diffusion-based policies struggle to generalize across parts, instances, and categories. We first introduce ArtiBench, a five-level benchmark covering kitchen, storage, office, and tool environments. ArtiBench enables structured evaluation from cross-part and cross-instance variation to long-horizon multi-object tasks, revealing the core generalization challenges of articulated object manipulation. Building on this benchmark, we propose ArtiBrain, a modular framework that unifies high-level reasoning with adaptive low-level control. ArtiBrain uses a VLM-based Task Reasoner (GPT-4.1) to decompose and validate subgoals, and employs a Hybrid Controller that combines geometry-aware keyframe execution with affordance-guided diffusion for precise and interpretable manipulation. An Affordance Memory Bank continually accumulates successful execution episodes and propagates part-level actionable affordances to unseen articulated parts and configurations. Extensive experiments on ArtiBench show that our ArtiBrain significantly outperforms state-of-the-art multimodal and diffusion-based methods in robustness and generalization. Code and dataset will be released upon acceptance.
>
---
#### [replaced 085] Are Large Vision Language Models Truly Grounded in Medical Images? Evidence from Italian Clinical Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19220v2](https://arxiv.org/pdf/2511.19220v2)**

> **作者:** Federico Felizzi; Olivia Riccomi; Michele Ferramola; Francesco Andrea Causio; Manuel Del Medico; Vittorio De Vita; Lorenzo De Mori; Alessandra Piscitelli; Pietro Eric Risuleo; Bianca Destro Castaniti; Antonio Cristiano; Alessia Longo; Luigi De Angelis; Mariapia Vassalli; Marcello Di Pumpo
>
> **备注:** Accepted at the Workshop on Multimodal Representation Learning for Healthcare (MMRL4H), EurIPS 2025
>
> **摘要:** Large vision language models (VLMs) have achieved impressive performance on medical visual question answering benchmarks, yet their reliance on visual information remains unclear. We investigate whether frontier VLMs demonstrate genuine visual grounding when answering Italian medical questions by testing four state-of-the-art models: Claude Sonnet 4.5, GPT-4o, GPT-5-mini, and Gemini 2.0 flash exp. Using 60 questions from the EuropeMedQA Italian dataset that explicitly require image interpretation, we substitute correct medical images with blank placeholders to test whether models truly integrate visual and textual information. Our results reveal striking variability in visual dependency: GPT-4o shows the strongest visual grounding with a 27.9pp accuracy drop (83.2% [74.6%, 91.7%] to 55.3% [44.1%, 66.6%]), while GPT-5-mini, Gemini, and Claude maintain high accuracy with modest drops of 8.5pp, 2.4pp, and 5.6pp respectively. Analysis of model-generated reasoning reveals confident explanations for fabricated visual interpretations across all models, suggesting varying degrees of reliance on textual shortcuts versus genuine visual analysis. These findings highlight critical differences in model robustness and the need for rigorous evaluation before clinical deployment.
>
---
#### [replaced 086] A Simple yet Effective Test-Time Adaptation for Zero-Shot Monocular Metric Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.14103v3](https://arxiv.org/pdf/2412.14103v3)**

> **作者:** Rémi Marsal; Alexandre Chapoutot; Philippe Xu; David Filliat
>
> **备注:** Published at IROS 2025 https://ieeexplore.ieee.org/document/11247168
>
> **摘要:** The recent development of \emph{foundation models} for monocular depth estimation such as Depth Anything paved the way to zero-shot monocular depth estimation. Since it returns an affine-invariant disparity map, the favored technique to recover the metric depth consists in fine-tuning the model. However, this stage is not straightforward, it can be costly and time-consuming because of the training and the creation of the dataset. The latter must contain images captured by the camera that will be used at test time and the corresponding ground truth. Moreover, the fine-tuning may also degrade the generalizing capacity of the original model. Instead, we propose in this paper a new method to rescale Depth Anything predictions using 3D points provided by sensors or techniques such as low-resolution LiDAR or structure-from-motion with poses given by an IMU. This approach avoids fine-tuning and preserves the generalizing power of the original depth estimation model while being robust to the noise of the sparse depth, of the camera-LiDAR calibration or of the depth model. Our experiments highlight enhancements relative to zero-shot monocular metric depth estimation methods, competitive results compared to fine-tuned approaches and a better robustness than depth completion approaches. Code available at github.com/ENSTA-U2IS-AI/depth-rescaling.
>
---
#### [replaced 087] Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21579v2](https://arxiv.org/pdf/2511.21579v2)**

> **作者:** Teng Hu; Zhentao Yu; Guozhen Zhang; Zihan Su; Zhengguang Zhou; Youliang Zhang; Yuan Zhou; Qinglin Lu; Ran Yi
>
> **摘要:** The synthesis of synchronized audio-visual content is a key challenge in generative AI, with open-source models facing challenges in robust audio-video alignment. Our analysis reveals that this issue is rooted in three fundamental challenges of the joint diffusion process: (1) Correspondence Drift, where concurrently evolving noisy latents impede stable learning of alignment; (2) inefficient global attention mechanisms that fail to capture fine-grained temporal cues; and (3) the intra-modal bias of conventional Classifier-Free Guidance (CFG), which enhances conditionality but not cross-modal synchronization. To overcome these challenges, we introduce Harmony, a novel framework that mechanistically enforces audio-visual synchronization. We first propose a Cross-Task Synergy training paradigm to mitigate drift by leveraging strong supervisory signals from audio-driven video and video-driven audio generation tasks. Then, we design a Global-Local Decoupled Interaction Module for efficient and precise temporal-style alignment. Finally, we present a novel Synchronization-Enhanced CFG (SyncCFG) that explicitly isolates and amplifies the alignment signal during inference. Extensive experiments demonstrate that Harmony establishes a new state-of-the-art, significantly outperforming existing methods in both generation fidelity and, critically, in achieving fine-grained audio-visual synchronization.
>
---
#### [replaced 088] Active Negative Loss: A Robust Framework for Learning with Noisy Labels
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02373v3](https://arxiv.org/pdf/2412.02373v3)**

> **作者:** Xichen Ye; Yifan Wu; Yiqi Wang; Xiaoqiang Li; Weizhong Zhang; Yifan Chen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Deep supervised learning has achieved remarkable success across a wide range of tasks, yet it remains susceptible to overfitting when confronted with noisy labels. To address this issue, noise-robust loss functions offer an effective solution for enhancing learning in the presence of label noise. In this work, we systematically investigate the limitation of the recently proposed Active Passive Loss (APL), which employs Mean Absolute Error (MAE) as its passive loss function. Despite the robustness brought by MAE, one of its key drawbacks is that it pays equal attention to clean and noisy samples; this feature slows down convergence and potentially makes training difficult, particularly in large-scale datasets. To overcome these challenges, we introduce a novel loss function class, termed Normalized Negative Loss Functions (NNLFs), which serve as passive loss functions within the APL framework. NNLFs effectively address the limitations of MAE by concentrating more on memorized clean samples. By replacing MAE in APL with our proposed NNLFs, we enhance APL and present a new framework called Active Negative Loss (ANL). Moreover, in non-symmetric noise scenarios, we propose an entropy-based regularization technique to mitigate the vulnerability to the label imbalance. Extensive experiments demonstrate that the new loss functions adopted by our ANL framework can achieve better or comparable performance to state-of-the-art methods across various label noise types and in image segmentation tasks. The source code is available at: https://github.com/Virusdoll/Active-Negative-Loss.
>
---
#### [replaced 089] SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20157v3](https://arxiv.org/pdf/2511.20157v3)**

> **作者:** Da Li; Jiping Jin; Xuanlong Yu; Wei Liu; Xiaodong Cun; Kai Chen; Rui Fan; Jiangang Kong; Xi Shen
>
> **备注:** Project page: https://pokerman8.github.io/SKEL-CF/
>
> **摘要:** Parametric 3D human models such as SMPL have driven significant advances in human pose and shape estimation, yet their simplified kinematics limit biomechanical realism. The recently proposed SKEL model addresses this limitation by re-rigging SMPL with an anatomically accurate skeleton. However, estimating SKEL parameters directly remains challenging due to limited training data, perspective ambiguities, and the inherent complexity of human articulation. We introduce SKEL-CF, a coarse-to-fine framework for SKEL parameter estimation. SKEL-CF employs a transformer-based encoder-decoder architecture, where the encoder predicts coarse camera and SKEL parameters, and the decoder progressively refines them in successive layers. To ensure anatomically consistent supervision, we convert the existing SMPL-based dataset 4DHuman into a SKEL-aligned version, 4DHuman-SKEL, providing high-quality training data for SKEL estimation. In addition, to mitigate depth and scale ambiguities, we explicitly incorporate camera modeling into the SKEL-CF pipeline and demonstrate its importance across diverse viewpoints. Extensive experiments validate the effectiveness of the proposed design. On the challenging MOYO dataset, SKEL-CF achieves 85.0 MPJPE / 51.4 PA-MPJPE, significantly outperforming the previous SKEL-based state-of-the-art HSMR (104.5 / 79.6). These results establish SKEL-CF as a scalable and anatomically faithful framework for human motion analysis, bridging the gap between computer vision and biomechanics. Our implementation is available on the project page: https://pokerman8.github.io/SKEL-CF/.
>
---
#### [replaced 090] Unlabeled Data Improves Fine-Grained Image Zero-shot Classification with Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.03195v2](https://arxiv.org/pdf/2506.03195v2)**

> **作者:** Yunqi Hong; Sohyun An; Andrew Bai; Neil Y. C. Lin; Cho-Jui Hsieh
>
> **摘要:** Despite Multimodal Large Language Models (MLLMs) showing promising results on general zero-shot image classification tasks, fine-grained image classification remains challenging. It demands precise attention to subtle visual details to distinguish between visually similar subcategories--details that MLLMs may easily overlook without explicit guidance. To address this, we introduce AutoSEP, an iterative self-supervised prompt learning framework designed to enhance MLLM fine-grained classification capabilities in a fully unsupervised manner. Our core idea is to leverage unlabeled data to learn a description prompt that guides MLLMs in identifying crucial discriminative features within an image, and boosts classification accuracy. We developed an automatic self-enhancing prompt learning framework called AutoSEP to iteratively improve the description prompt using unlabeled data, based on instance-level classification scoring function. AutoSEP only requires black-box access to MLLMs, eliminating the need for any training or fine-tuning. We evaluate our approach on multiple fine-grained classification datasets. It consistently outperforms other unsupervised baselines, demonstrating the effectiveness of our self-supervised optimization framework. Notably, AutoSEP on average improves 13 percent over standard zero-shot classification and 5 percent over the best-performing baselines. Code is available at: https://github.com/yq-hong/AutoSEP
>
---
#### [replaced 091] SONIC: Spectral Optimization of Noise for Inpainting with Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19985v2](https://arxiv.org/pdf/2511.19985v2)**

> **作者:** Seungyeon Baek; Erqun Dong; Shadan Namazifard; Mark J. Matthews; Kwang Moo Yi
>
> **摘要:** We propose a novel training-free method for inpainting with off-the-shelf text-to-image models. While guidance-based methods in theory allow generic models to be used for inverse problems such as inpainting, in practice, their effectiveness is limited, leading to the necessity of specialized inpainting-specific models. In this work, we argue that the missing ingredient for training-free inpainting is the optimization (guidance) of the initial seed noise. We propose to optimize the initial seed noise to approximately match the unmasked parts of the data - with as few as a few tens of optimization steps. We then apply conventional training-free inpainting methods on top of our optimized initial seed noise. Critically, we propose two core ideas to effectively implement this idea: (i) to avoid the costly unrolling required to relate the initial noise and the generated outcome, we perform linear approximation; and (ii) to stabilize the optimization, we optimize the initial seed noise in the spectral domain. We demonstrate the effectiveness of our method on various inpainting tasks, outperforming the state of the art. Project page: https://ubc-vision.github.io/sonic/
>
---
#### [replaced 092] Total Least Square Optimal Analytic Signal by Structure Tensor for N-D images
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2005.08108v2](https://arxiv.org/pdf/2005.08108v2)**

> **作者:** Josef Bigun; Fernando Alonso-Fernandez
>
> **备注:** Changed title, included new experimental results. Reorganized so that details are in Supplementary Material part, which is new
>
> **摘要:** We produce the analytic signal by using the Structure Tensor, which provides Total Least Squares optimal vectors for estimating orientation and scale locally. Together, these vectors represent N-D frequency components that determine adaptive, complex probing filters. The N-D analytic signal is obtained through scalar products of adaptive filters with image neighborhoods. It comprises orientation, scale, phase, and amplitude information of the neighborhood. The ST analytic signal $ f_A $ is continuous and isotropic, and its extension to N-D is straightforward. The phase gradient can be represented as a vector (instantaneous frequency) or as a tensor. Both are continuous and isotropic, while the tensor additionally preserves continuity of orientation and retains the same information as the vector representation. The tensor representation can also be used to detect singularities. Detection with known phase portraits has been demonstrated in 2-D with relevance to fringe pattern processing in wave physics, including optics and fingerprint measurements. To construct adaptive filters we have used Gabor filter family members as probing functions, but other function families can also be used to sample the spectrum, e.g., quadrature filters. A comparison to three baseline alternatives-in representation (Monogenic signal), enhancement (Monogenic signal combined with a spline-wavelet pyramid), and singularity detection (mindtct, a fingerprint minutia detector widely used in numerous studies)-is also reported using images with precisely known ground truths for location, orientation, singularity type (where applicable), and wave period.
>
---
#### [replaced 093] Visual-Word Tokenizer: Beyond Fixed Sets of Tokens in Vision Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.15397v4](https://arxiv.org/pdf/2411.15397v4)**

> **作者:** Leonidas Gee; Wing Yan Li; Viktoriia Sharmanska; Novi Quadrianto
>
> **摘要:** The cost of deploying vision transformers increasingly represents a barrier to wider industrial adoption. Existing compression techniques require additional end-to-end fine-tuning or incur a significant drawback to energy efficiency, making them ill-suited for online (real-time) inference, where a prediction is made on any new input as it comes in. We introduce the $\textbf{Visual-Word Tokenizer}$ (VWT), a training-free method for reducing energy costs while retaining performance. The VWT groups visual subwords (image patches) that are frequently used into visual words, while infrequent ones remain intact. To do so, $\textit{intra}$-image or $\textit{inter}$-image statistics are leveraged to identify similar visual concepts for sequence compression. Experimentally, we demonstrate a reduction in energy consumed of up to 47%. Comparative approaches of 8-bit quantization and token merging can lead to significantly increased energy costs (up to 500% or more). Our results indicate that VWTs are well-suited for efficient online inference with a marginal compromise on performance. The experimental code for our paper is also made publicly available.
>
---
#### [replaced 094] LD-ViCE: Latent Diffusion Model for Video Counterfactual Explanations
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.08422v3](https://arxiv.org/pdf/2509.08422v3)**

> **作者:** Payal Varshney; Adriano Lucieri; Christoph Balada; Sheraz Ahmed; Andreas Dengel
>
> **备注:** Under Review CVPR 2026 (44 Pages)
>
> **摘要:** Video-based AI systems are increasingly adopted in safety-critical domains such as autonomous driving and healthcare. However, interpreting their decisions remains challenging due to the inherent spatiotemporal complexity of video data and the opacity of deep learning models. Existing explanation techniques often suffer from limited temporal coherence and a lack of actionable causal insights. Current counterfactual explanation methods typically do not incorporate guidance from the target model, reducing semantic fidelity and practical utility. We introduce Latent Diffusion for Video Counterfactual Explanations (LD-ViCE), a novel framework designed to explain the behavior of video-based AI models. Compared to previous approaches, LD-ViCE reduces the computational costs of generating explanations by operating in latent space using a state-of-the-art diffusion model, while producing realistic and interpretable counterfactuals through an additional refinement step. Experiments on three diverse video datasets - EchoNet-Dynamic (cardiac ultrasound), FERV39k (facial expression), and Something-Something V2 (action recognition) with multiple target models covering both classification and regression tasks, demonstrate that LD-ViCE generalizes well and achieves state-of-the-art performance. On the EchoNet-Dynamic dataset, LD-ViCE achieves significantly higher regression accuracy than prior methods and exhibits high temporal consistency, while the refinement stage further improves perceptual quality. Qualitative analyses confirm that LD-ViCE produces semantically meaningful and temporally coherent explanations, providing actionable insights into model behavior. LD-ViCE advances the trustworthiness and interpretability of video-based AI systems through visually coherent counterfactual explanations.
>
---
#### [replaced 095] SACA: Selective Attention-Based Clustering Algorithm
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17150v2](https://arxiv.org/pdf/2508.17150v2)**

> **作者:** Meysam Shirdel Bilehsavar; Razieh Ghaedi; Samira Seyed Taheri; Xinqi Fan; Christian O'Reilly
>
> **备注:** 32 pages, 14 figures
>
> **摘要:** Clustering algorithms are fundamental tools across many fields, with density-based methods offering particular advantages in identifying arbitrarily shaped clusters and handling noise. However, their effectiveness is often limited by the requirement of critical parameter tuning by users, which typically requires significant domain expertise. This paper introduces a novel density-based clustering algorithm loosely inspired by the concept of selective attention, designed to minimize reliance on parameter tuning for most applications. The proposed method computes an adaptive threshold to exclude sparsely distributed points and outliers, constructs an initial cluster framework, and subsequently reintegrates the filtered points to refine the final results. Extensive experiments on diverse benchmark datasets demonstrate the robustness, accuracy, and ease of use of the proposed approach, establishing it as a powerful alternative to conventional density-based clustering techniques.
>
---
#### [replaced 096] Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.11561v3](https://arxiv.org/pdf/2501.11561v3)**

> **作者:** Zhiyuan You; Xin Cai; Jinjin Gu; Tianfan Xue; Chao Dong
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** With the rapid advancement of Multi-modal Large Language Models (MLLMs), MLLM-based Image Quality Assessment (IQA) methods have shown promising performance in linguistic quality description. However, current methods still fall short in accurately scoring image quality. In this work, we aim to leverage MLLMs to regress accurate quality scores. A key challenge is that the quality score is inherently continuous, typically modeled as a Gaussian distribution, whereas MLLMs generate discrete token outputs. This mismatch necessitates score discretization. Previous approaches discretize the mean score into a one-hot label, resulting in information loss and failing to capture inter-image relationships. We propose a distribution-based approach that discretizes the score distribution into a soft label. This method preserves the characteristics of the score distribution, achieving high accuracy and maintaining inter-image relationships. Moreover, to address dataset variation, where different IQA datasets exhibit various distributions, we introduce a fidelity loss based on Thurstone's model. This loss captures intra-dataset relationships, facilitating co-training across multiple IQA datasets. With these designs, we develop the distribution-based Depicted image Quality Assessment model for Score regression (DeQA-Score). Experiments across multiple benchmarks show that DeQA-Score stably outperforms baselines in score regression. Also, DeQA-Score can predict the score distribution that closely aligns with human annotations. Codes and model weights have been released in https://depictqa.github.io/deqa-score/.
>
---
#### [replaced 097] MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对单目视觉语言导航（VLN）中因缺乏全景信息导致性能受限的问题，提出MonoDream框架。通过构建统一导航表征（UNR）并引入潜在全景梦境（LPD）任务，使单目模型能预测未来全景特征，显著提升导航准确性，缩小与全景输入方法的差距。**

- **链接: [https://arxiv.org/pdf/2508.02549v4](https://arxiv.org/pdf/2508.02549v4)**

> **作者:** Shuo Wang; Yongcai Wang; Zhaoxin Fan; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Wanting Li; Xudong Cai; Yeying Jin; Deying Li
>
> **摘要:** Vision-Language Navigation (VLN) tasks often leverage panoramic RGB and depth inputs to provide rich spatial cues for action planning, but these sensors can be costly or less accessible in real-world deployments. Recent approaches based on Vision-Language Action (VLA) models achieve strong results with monocular input, yet they still lag behind methods using panoramic RGB-D information. We present MonoDream, a lightweight VLA framework that enables monocular agents to learn a Unified Navigation Representation (UNR). This shared feature representation jointly aligns navigation-relevant visual semantics (e.g., global layout, depth, and future cues) and language-grounded action intent, enabling more reliable action prediction. MonoDream further introduces Latent Panoramic Dreaming (LPD) tasks to supervise the UNR, which train the model to predict latent features of panoramic RGB and depth observations at both current and future steps based on only monocular input. Experiments on multiple VLN benchmarks show that MonoDream consistently improves monocular navigation performance and significantly narrows the gap with panoramic-based agents.
>
---
#### [replaced 098] Text2Traffic: A Text-to-Image Generation and Editing Method for Traffic Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12932v3](https://arxiv.org/pdf/2511.12932v3)**

> **作者:** Feng Lv; Haoxuan Feng; Zilu Zhang; Chunlong Xia; Yanfeng Li
>
> **摘要:** With the rapid advancement of intelligent transportation systems, text-driven image generation and editing techniques have demonstrated significant potential in providing rich, controllable visual scene data for applications such as traffic monitoring and autonomous driving. However, several challenges remain, including insufficient semantic richness of generated traffic elements, limited camera viewpoints, low visual fidelity of synthesized images, and poor alignment between textual descriptions and generated content. To address these issues, we propose a unified text-driven framework for both image generation and editing, leveraging a controllable mask mechanism to seamlessly integrate the two tasks. Furthermore, we incorporate both vehicle-side and roadside multi-view data to enhance the geometric diversity of traffic scenes. Our training strategy follows a two-stage paradigm: first, we perform conceptual learning using large-scale coarse-grained text-image data; then, we fine-tune with fine-grained descriptive data to enhance text-image alignment and detail quality. Additionally, we introduce a mask-region-weighted loss that dynamically emphasizes small yet critical regions during training, thereby substantially enhancing the generation fidelity of small-scale traffic elements. Extensive experiments demonstrate that our method achieves leading performance in text-based image generation and editing within traffic scenes.
>
---
#### [replaced 099] DiffFuSR: Super-Resolution of all Sentinel-2 Multispectral Bands using Diffusion Models
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.11764v2](https://arxiv.org/pdf/2506.11764v2)**

> **作者:** Muhammad Sarmad; Arnt-Børre Salberg; Michael Kampffmeyer
>
> **备注:** Accepted for Publication at IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING (TGRS)
>
> **摘要:** This paper presents DiffFuSR, a modular pipeline for super-resolving all 12 spectral bands of Sentinel-2 Level-2A imagery to a unified ground sampling distance (GSD) of 2.5 meters. The pipeline comprises two stages: (i) a diffusion-based super-resolution (SR) model trained on high-resolution RGB imagery from the NAIP and WorldStrat datasets, harmonized to simulate Sentinel-2 characteristics; and (ii) a learned fusion network that upscales the remaining multispectral bands using the super-resolved RGB image as a spatial prior. We introduce a robust degradation model and contrastive degradation encoder to support blind SR. Extensive evaluations of the proposed SR pipeline on the OpenSR benchmark demonstrate that the proposed method outperforms current SOTA baselines in terms of reflectance fidelity, spectral consistency, spatial alignment, and hallucination suppression. Furthermore, the fusion network significantly outperforms classical and learned pansharpening approaches, enabling accurate enhancement of Sentinel-2's 20 m and 60 m bands. This work proposes a novel modular framework Sentinel-2 SR that utilizes harmonized learning with diffusion models and fusion strategies. Our code and models can be found at https://github.com/NorskRegnesentral/DiffFuSR.
>
---
#### [replaced 100] IntrinsiX: High-Quality PBR Generation using Image Priors
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.01008v2](https://arxiv.org/pdf/2504.01008v2)**

> **作者:** Peter Kocsis; Lukas Höllein; Matthias Nießner
>
> **备注:** Project page: https://peter-kocsis.github.io/IntrinsiX/ Video: https://youtu.be/b0wVA44R93Y
>
> **摘要:** We introduce IntrinsiX, a novel method that generates high-quality intrinsic images from text description. In contrast to existing text-to-image models whose outputs contain baked-in scene lighting, our approach predicts physically-based rendering (PBR) maps. This enables the generated outputs to be used for content creation scenarios in core graphics applications that facilitate re-lighting, editing, and texture generation tasks. In order to train our generator, we exploit strong image priors, and pre-train separate models for each PBR material component (albedo, roughness, metallic, normals). We then align these models with a new cross-intrinsic attention formulation that concatenates key and value features in a consistent fashion. This allows us to exchange information between each output modality and to obtain semantically coherent PBR predictions. To ground each intrinsic component, we propose a rendering loss which provides image-space signals to constrain the model, thus facilitating sharp details also in the output BRDF properties. Our results demonstrate detailed intrinsic generation with strong generalization capabilities that outperforms existing intrinsic image decomposition methods used with generated images by a significant margin. Finally, we show a series of applications, including re-lighting, editing, and text-conditioned room-scale PBR texture generation.
>
---
#### [replaced 101] Histomorphology-Guided Prototypical Multi-Instance Learning for Breast Cancer WSI Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17983v2](https://arxiv.org/pdf/2503.17983v2)**

> **作者:** Baizhi Wang; Rui Yan; Wenxin Ma; Xu Zhang; Yuhao Wang; Xiaolong Li; Yunjie Gu; Zihang Jiang; S. Kevin Zhou
>
> **备注:** 11 pages,8 figures
>
> **摘要:** Histomorphology is crucial in cancer diagnosis. However, existing whole slide image (WSI) classification methods struggle to effectively incorporate histomorphology information, limiting their ability to capture key pathological features. Particularly when the number of instances within a bag is large and their features are complex, it becomes challenging to accurately identify instances decisive for the bag label, making these methods prone to interference from ambiguous instances. To address this limitation, we propose a novel Histomorphology-Guided Prototypical Multi-Instance Learning (HGPMIL) framework that explicitly learns histomorphology-guided prototypical representations by incorporating tumor cellularity, cellular morphology, and tissue architecture. Specifically, our approach consists of three key components: (1) estimating the importance of tumor-related histomorphology information at patch-level based on medical prior knowledge; (2) generating representative prototypes through histomorphology-prototypical clustering; and (3) enabling WSI classification through histomorphology-guided prototypical aggregation. HGPMIL adjusts the decision boundary by incorporating histomorphological importance to reduce instance label uncertainty, thereby reversely optimizing the bag-level boundary. Experimental results demonstrate its effectiveness, achieving high diagnostic accuracy for molecular subtyping, cancer subtyping and survival analysis. The code will be made available at https://github.com/Badgewho/HMDMIL.
>
---
#### [replaced 102] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06857v2](https://arxiv.org/pdf/2511.06857v2)**

> **作者:** Fanding Li; Xiangyu Li; Xianghe Su; Xingyu Qiu; Suyu Dong; Wei Wang; Kuanquan Wang; Gongning Luo; Shuo Li
>
> **备注:** 13 pages, 10 figures, extended version of AAAI-26 paper
>
> **摘要:** A simultaneous enhancement of accuracy and diversity of predictions remains a challenge in ambiguous medical image segmentation (AMIS) due to the inherent trade-offs. While truncated diffusion probabilistic models (TDPMs) hold strong potential with a paradigm optimization, existing TDPMs suffer from entangled accuracy and diversity of predictions with insufficient fidelity and plausibility. To address the aforementioned challenges, we propose Ambiguity-aware Truncated Flow Matching (ATFM), which introduces a novel inference paradigm and dedicated model components. Firstly, we propose Data-Hierarchical Inference, a redefinition of AMIS-specific inference paradigm, which enhances accuracy and diversity at data-distribution and data-sample level, respectively, for an effective disentanglement. Secondly, Gaussian Truncation Representation (GTR) is introduced to enhance both fidelity of predictions and reliability of truncation distribution, by explicitly modeling it as a Gaussian distribution at $T_{\text{trunc}}$ instead of using sampling-based approximations. Thirdly, Segmentation Flow Matching (SFM) is proposed to enhance the plausibility of diverse predictions by extending semantic-aware flow transformation in Flow Matching (FM). Comprehensive evaluations on LIDC and ISIC3 datasets demonstrate that ATFM outperforms SOTA methods and simultaneously achieves a more efficient inference. ATFM improves GED and HM-IoU by up to $12\%$ and $7.3\%$ compared to advanced methods.
>
---
#### [replaced 103] Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21977v4](https://arxiv.org/pdf/2507.21977v4)**

> **作者:** Jihao Gu; Kun Li; Fei Wang; Yanyan Wei; Zhiliang Wu; Hehe Fan; Meng Wang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Micro-Actions (MAs) are an important form of non-verbal communication in social interactions, with potential applications in human emotional analysis. However, existing methods in Micro-Action Recognition often overlook the inherent subtle changes in MAs, which limits the accuracy of distinguishing MAs with subtle changes. To address this issue, we present a novel Motion-guided Modulation Network (MMN) that implicitly captures and modulates subtle motion cues to enhance spatial-temporal representation learning. Specifically, we introduce a Motion-guided Skeletal Modulation module (MSM) to inject motion cues at the skeletal level, acting as a control signal to guide spatial representation modeling. In parallel, we design a Motion-guided Temporal Modulation module (MTM) to incorporate motion information at the frame level, facilitating the modeling of holistic motion patterns in micro-actions. Finally, we propose a motion consistency learning strategy to aggregate the motion cues from multi-scale features for micro-action classification. Experimental results on the Micro-Action 52 and iMiGUE datasets demonstrate that MMN achieves state-of-the-art performance in skeleton-based micro-action recognition, underscoring the importance of explicitly modeling subtle motion cues. The code will be available at https://github.com/momiji-bit/MMN.
>
---
#### [replaced 104] Activation Quantization of Vision Encoders Needs Prefixing Registers
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04547v3](https://arxiv.org/pdf/2510.04547v3)**

> **作者:** Seunghyeon Kim; Jinho Kim; Taesun Yeom; Wonpyo Park; Kyuyeun Kim; Jaeho Lee
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Transformer-based vision encoders -- such as CLIP -- are central to multimodal intelligence, powering applications from autonomous web agents to robotic control. Since these applications often demand real-time processing of massive visual data, reducing the inference cost of vision encoders is critical. Quantization offers a practical path, but remains challenging even at 8-bit precision due to massive-scale activations (i.e., outliers). In this work, we propose $\textit{RegCache}$, a training-free algorithm that mitigates outliers in large-scale pretrained vision encoders and serves as a plug-in module that can be applied on top of other quantization methods. The proposed RegCache introduces outlier-prone yet semantically meaningless prefix tokens to the target vision encoder, which prevents other tokens from having outliers. Notably, we observe that outliers in vision encoders behave differently from those in language models, motivating two technical innovations: middle-layer prefixing and token deletion. Experiments show that our method consistently improves the accuracy of quantized models across both text-supervised and self-supervised vision encoders.
>
---
#### [replaced 105] INQUIRE-Search: A Framework for Interactive Discovery in Large-Scale Biodiversity Databases
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15656v2](https://arxiv.org/pdf/2511.15656v2)**

> **作者:** Edward Vendrow; Julia Chae; Rupa Kurinchi-Vendhan; Isaac Eckert; Jazlynn Hall; Marta Jarzyna; Reymond Miyajima; Ruth Oliver; Laura Pollock; Lauren Schrack; Scott Yanco; Oisin Mac Aodha; Sara Beery
>
> **备注:** EV, JC, RKV contributed equally
>
> **摘要:** Large community science platforms such as iNaturalist contain hundreds of millions of biodiversity images that often capture ecological context on behaviors, interactions, phenology, and habitat. Yet most ecological workflows rely on metadata filtering or manual inspection, leaving this secondary information inaccessible at scale. We introduce INQUIRE-Search, an open-source system that enables scientists to rapidly and interactively search within an ecological image database for specific concepts using natural language, verify and export relevant observations, and utilize this discovered data for novel scientific analysis. Compared to traditional methods, INQUIRE-Search takes a fraction of the time, opening up new possibilities for scientific questions that can be explored. Through five case studies, we show the diversity of scientific applications that a tool like INQUIRE-Search can support, from seasonal variation in behavior across species to forest regrowth after wildfires. These examples demonstrate a new paradigm for interactive, efficient, and scalable scientific discovery that can begin to unlock previously inaccessible scientific value in large-scale biodiversity datasets. Finally, we emphasize using such AI-enabled discovery tools for science call for experts to reframe the priorities of the scientific process and develop novel methods for experiment design, data collection, survey effort, and uncertainty analysis.
>
---
#### [replaced 106] Investigating the Relationship between the Weighted Figure of Merit and Rosin's Measure
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05749v4](https://arxiv.org/pdf/2506.05749v4)**

> **作者:** Bimal Kumar Ray
>
> **摘要:** Many studies have been conducted to solve the problem of approximating a digital boundary by piece straight-line segments for the further processing required in computer vision applications. The authors of these studies compared their schemes to determine the best one. The initial measure used to assess the goodness of fit of a polygonal approximation was the figure of merit. Later,it was noted that this measure was not an appropriate metric for a valid reason which is why Rosin-through mathematical analysis-introduced a measure called merit. However,this measure involves an optimal scheme of polygonal approximation,so it is time-consuming to compute it to assess the goodness of fit of an approximation. This led many researchers to use a weighted figure of merit as a substitute for Rosin's measure to compare sub optimal schemes. An attempt is made in this communication to investigate whether the two measures-weighted figure of merit and Rosin's measure-are related so that one can be used instead of the other, and toward this end, theoretical analysis, experimental investigation and statistical analysis are carried out. The mathematical formulas for the weighted figure of merit and Rosin's measure are analyzed, and through proof of theorems,it is found that the two measures are theoretically independent of each other. The graphical analysis of experiments carried out using a public dataset supports the results of the theoretical analysis. The statistical analysis via Pearson's correlation coefficient and non-linear correlation measure also revealed that the two measures are uncorrelated. This analysis leads one to conclude that if a suboptimal scheme is found to be better (worse) than some other suboptimal scheme,as indicated by Rosin's measure,then the same conclusion cannot be drawn using a weighted figure of merit,so one cannot use a weighted figure of merit instead of Rosin's measure.
>
---
#### [replaced 107] Unveiling Hidden Vulnerabilities in Digital Human Generation via Adversarial Attacks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.17457v2](https://arxiv.org/pdf/2504.17457v2)**

> **作者:** Zhiying Li; Yeying Jin; Fan Shen; Zhi Liu; Weibin Chen; Pengju Zhang; Xiaomei Zhang; Boyu Chen; Michael Shen; Kejian Wu; Zhaoxin Fan; Jin Dong
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Expressive human pose and shape estimation (EHPS) is crucial for digital human generation, especially in applications like live streaming. While existing research primarily focuses on reducing estimation errors, it largely neglects robustness and security aspects, leaving these systems vulnerable to adversarial attacks. To address this significant challenge, we propose the \textbf{Tangible Attack (TBA)}, a novel framework designed to generate adversarial examples capable of effectively compromising any digital human generation model. Our approach introduces a \textbf{Dual Heterogeneous Noise Generator (DHNG)}, which leverages Variational Autoencoders (VAE) and ControlNet to produce diverse, targeted noise tailored to the original image features. Additionally, we design a custom \textbf{adversarial loss function} to optimize the noise, ensuring both high controllability and potent disruption. By iteratively refining the adversarial sample through multi-gradient signals from both the noise and the state-of-the-art EHPS model, TBA substantially improves the effectiveness of adversarial attacks. Extensive experiments demonstrate TBA's superiority, achieving a remarkable 41.0\% increase in estimation error, with an average improvement of approximately 17.0\%. These findings expose significant security vulnerabilities in current EHPS models and highlight the need for stronger defenses in digital human generation systems.
>
---
#### [replaced 108] UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.06165v3](https://arxiv.org/pdf/2509.06165v3)**

> **作者:** Huy Le; Nhat Chung; Tung Kieu; Jingkang Yang; Ngan Le
>
> **备注:** 11 pages, 7 figures. Accepted at WACV 2026
>
> **摘要:** Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design.
>
---
#### [replaced 109] DINO-Foresight: Looking into the Future with DINO
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.11673v2](https://arxiv.org/pdf/2412.11673v2)**

> **作者:** Efstathios Karypidis; Ioannis Kakogeorgiou; Spyros Gidaris; Nikos Komodakis
>
> **备注:** NeurIPS 2025
>
> **摘要:** Predicting future dynamics is crucial for applications like autonomous driving and robotics, where understanding the environment is key. Existing pixel-level methods are computationally expensive and often focus on irrelevant details. To address these challenges, we introduce DINO-Foresight, a novel framework that operates in the semantic feature space of pretrained Vision Foundation Models (VFMs). Our approach trains a masked feature transformer in a self-supervised manner to predict the evolution of VFM features over time. By forecasting these features, we can apply off-the-shelf, task-specific heads for various scene understanding tasks. In this framework, VFM features are treated as a latent space, to which different heads attach to perform specific tasks for future-frame analysis. Extensive experiments show the very strong performance, robustness and scalability of our framework. Project page and code at https://dino-foresight.github.io/ .
>
---
#### [replaced 110] Axial-UNet: A Neural Weather Model for Precipitation Nowcasting
- **分类: cs.LG; cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2504.19408v2](https://arxiv.org/pdf/2504.19408v2)**

> **作者:** Sumit Mamtani; Maitreya Sonawane
>
> **备注:** 16 pages, 3 figures. Accepted at the International Conference on Distributed Computing and Intelligent Technology (ICDCIT 2026), to appear in Springer LNCS
>
> **摘要:** Accurately predicting short-term precipitation is critical for weather-sensitive applications such as disaster management, aviation, and urban planning. Traditional numerical weather prediction can be computationally intensive at high resolution and short lead times. In this work, we propose a lightweight UNet-based encoder-decoder augmented with axial-attention blocks that attend along image rows and columns to capture long-range spatial interactions, while temporal context is provided by conditioning on multiple past radar frames. Our hybrid architecture captures both local and long-range spatio-temporal dependencies from radar image sequences, enabling fixed lead-time precipitation nowcasting with modest compute. Experimental results on a preprocessed subset of the HKO-7 radar dataset demonstrate that our model outperforms ConvLSTM, pix2pix-style cGANs, and a plain UNet in pixel-fidelity metrics, reaching PSNR 47.67 and SSIM 0.9943. We report PSNR/SSIM here; extending evaluation to meteorology-oriented skill measures (e.g., CSI/FSS) is left to future work. The approach is simple, scalable, and effective for resource-constrained, real-time forecasting scenarios.
>
---
#### [replaced 111] PoseAdapt: Sustainable Human Pose Estimation via Continual Learning Benchmarks and Toolkit
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.20469v2](https://arxiv.org/pdf/2409.20469v2)**

> **作者:** Muhammad Saif Ullah Khan; Didier Stricker
>
> **备注:** Accepted in WACV 2026 Applications Track
>
> **摘要:** Human pose estimators are typically retrained from scratch or naively fine-tuned whenever keypoint sets, sensing modalities, or deployment domains change--an inefficient, compute-intensive practice that rarely matches field constraints. We present PoseAdapt, an open-source framework and benchmark suite for continual pose model adaptation. PoseAdapt defines domain-incremental and class-incremental tracks that simulate realistic changes in density, lighting, and sensing modality, as well as skeleton growth. The toolkit supports two workflows: (i) Strategy Benchmarking, which lets researchers implement continual learning (CL) methods as plugins and evaluate them under standardized protocols; and (ii) Model Adaptation, which allows practitioners to adapt strong pretrained models to new tasks with minimal supervision. We evaluate representative regularization-based methods in single-step and sequential settings. Benchmarks enforce a fixed lightweight backbone, no access to past data, and tight per-step budgets. This isolates adaptation strategy effects, highlighting the difficulty of maintaining accuracy under strict resource limits. PoseAdapt connects modern CL techniques with practical pose estimation needs, enabling adaptable models that improve over time without repeated full retraining.
>
---
#### [replaced 112] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.02821v3](https://arxiv.org/pdf/2504.02821v3)**

> **作者:** Mateusz Pach; Shyamgopal Karthik; Quentin Bouniot; Serge Belongie; Zeynep Akata
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Sparse Autoencoders (SAEs) have recently gained attention as a means to improve the interpretability and steerability of Large Language Models (LLMs), both of which are essential for AI safety. In this work, we extend the application of SAEs to Vision-Language Models (VLMs), such as CLIP, and introduce a comprehensive framework for evaluating monosemanticity at the neuron-level in visual representations. To ensure that our evaluation aligns with human perception, we propose a benchmark derived from a large-scale user study. Our experimental results reveal that SAEs trained on VLMs significantly enhance the monosemanticity of individual neurons, with sparsity and wide latents being the most influential factors. Further, we demonstrate that applying SAE interventions on CLIP's vision encoder directly steers multimodal LLM outputs (e.g., LLaVA), without any modifications to the underlying language model. These findings emphasize the practicality and efficacy of SAEs as an unsupervised tool for enhancing both interpretability and control of VLMs. Code and benchmark data are available at https://github.com/ExplainableML/sae-for-vlm.
>
---
#### [replaced 113] IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.00979v4](https://arxiv.org/pdf/2506.00979v4)**

> **作者:** Changjiang Jiang; Wenhui Dong; Zhonghao Zhang; Chenyang Si; Fengchang Yu; Wei Peng; Xinbin Yuan; Yifei Bi; Ming Zhao; Zian Zhou; Caifeng Shan
>
> **备注:** 30 pages
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) techniques has enabled the creation of high-quality synthetic content, but it also raises significant security concerns. Current detection methods face two major limitations: (1) the lack of multidimensional explainable datasets for generated images and videos. Existing open-source datasets (e.g., WildFake, GenVideo) rely on oversimplified binary annotations, which restrict the explainability and trustworthiness of trained detectors. (2) Prior MLLM-based forgery detectors (e.g., FakeVLM) exhibit insufficiently fine-grained interpretability in their step-by-step reasoning, which hinders reliable localization and explanation. To address these challenges, we introduce Ivy-Fake, the first large-scale multimodal benchmark for explainable AIGC detection. It consists of over 106K richly annotated training samples (images and videos) and 5,000 manually verified evaluation examples, sourced from multiple generative models and real world datasets through a carefully designed pipeline to ensure both diversity and quality. Furthermore, we propose Ivy-xDetector, a reinforcement learning model based on Group Relative Policy Optimization (GRPO), capable of producing explainable reasoning chains and achieving robust performance across multiple synthetic content detection benchmarks. Extensive experiments demonstrate the superiority of our dataset and confirm the effectiveness of our approach. Notably, our method improves performance on GenImage from 86.88% to 96.32%, surpassing prior state-of-the-art methods by a clear margin.
>
---
#### [replaced 114] Boosting Reasoning in Large Multimodal Models via Activation Replay
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19972v2](https://arxiv.org/pdf/2511.19972v2)**

> **作者:** Yun Xing; Xiaobin Hu; Qingdong He; Jiangning Zhang; Shuicheng Yan; Shijian Lu; Yu-Gang Jiang
>
> **备注:** 11 figures, 10 tables
>
> **摘要:** Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach to incentivizing reasoning capability in Large Multimodal Models (LMMs), while the underlying mechanisms behind this post-training paradigm are poorly understood. We begin by exploring how input activations are affected by RLVR through the perspective of logit lens. Our systematic investigations across multiple post-trained LMMs suggest that RLVR shifts low-entropy activations unexpectedly, while high-entropy ones are less affected. We further demonstrate that such phenomena are associated with LMM reasoning by controlled experiments, suggesting a potentially beneficial role of modulating low-entropy activations. To this end, we propose Activation Replay, a novel simple yet effective training-free approach that boosts multimodal reasoning of post-trained LMMs without requiring expensive policy optimization. Our design involves manipulation of visual tokens at test time, replaying low-entropy activations from the input context of base LMMs to regulating the RLVR counterparts. Activation Replay triggers better reasoning across diverse scenarios, including mathematics, o3-like visual agents, and video reasoning. We further show that Activation Replay boosts Pass@K and mitigates narrower reasoning coverage of RLVR. Our design is compared against alternative choices, such as replaying high-entropy activations instead of low-entropy ones, or direct cross-model intervention instead of manipulating input tokens, demonstrating the superiority of our implementation. Codes will be made publicly available.
>
---
#### [replaced 115] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
- **分类: cs.RO; cs.CV**

- **简介: 论文针对机器人操作中因视觉相似物体导致的非马尔可夫决策问题，提出LIBERO-Mem基准与Embodied-SlotSSM框架。该框架通过槽位状态建模与关系编码器实现时空一致的物体记忆，提升长期依赖下的动作预测能力，解决了复杂交互中对象历史感知难题。**

- **链接: [https://arxiv.org/pdf/2511.11478v3](https://arxiv.org/pdf/2511.11478v3)**

> **作者:** Nhat Chung; Taisei Hanyu; Toan Nguyen; Huy Le; Frederick Bumgarner; Duy Minh Ho Nguyen; Khoa Vo; Kashu Yamazaki; Chase Rainwater; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.
>
---
#### [replaced 116] Look Where It Matters: Training-Free Ultra-HR Remote Sensing VQA via Adaptive Zoom Search
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20460v2](https://arxiv.org/pdf/2511.20460v2)**

> **作者:** Yunqi Zhou; Chengjie Jiang; Chun Yuan; Jing Li
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** With advances in satellite constellations, sensor technologies, and imaging pipelines, ultra-high-resolution (Ultra-HR) remote sensing imagery is becoming increasingly widespread. However, current remote sensing foundation models are ill-suited to such inputs: full-image encoding exhausts token and memory budgets, while resize-based preprocessing loses fine-grained and answer-critical details. In this context, guiding the model look where it matters before prediction becomes crucial. Therefore, we present ZoomSearch, a training-free, plug-and-play pipeline that decouples 'where to look' from 'how to answer' for Ultra-HR Remote Sensing Visual Question Answering (RS-VQA). ZoomSearch combines Adaptive Multi-Branch Zoom Search, which performs a hierarchical search over image patches to localize query-relevant regions, with Layout-Aware Patch Reassembly, which reorganizes the selected patches into a compact, layout-faithful canvas. We conduct comprehensive experiments on Ultra-HR RS-VQA benchmarks MME-RealWorld-RS and LRS-VQA, comparing against (i) strong general foundation models, (ii) remote sensing foundation models, (iii) Ultra-HR RS-VQA methods, and (iv) plug-and-play search-based VQA methods. When integrated with LLaVA-ov, ZoomSearch attains state-of-the-art accuracy across diverse tasks, improving the LLaVA-ov baseline by 26.3% on LRS-VQA and 114.8% on MME-RealWorld-RS. Meanwhile, it achieves much higher inference efficiency, outperforming prior search-based methods by 20%~44% in speed.
>
---
#### [replaced 117] Spacewalk-18: A Benchmark for Multimodal and Long-form Procedural Video Understanding in Novel Domains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.18773v4](https://arxiv.org/pdf/2311.18773v4)**

> **作者:** Zitian Tang; Rohan Myer Krishnan; Zhiqiu Yu; Chen Sun
>
> **备注:** WACV 2026
>
> **摘要:** Learning from (procedural) videos has increasingly served as a pathway for embodied agents to acquire skills from human demonstrations. To do this, video understanding models must be able to obtain structured understandings, such as the temporal segmentation of a demonstration into sequences of actions and skills, and to generalize the understandings to novel environments, tasks, and problem domains. In pursuit of this goal, we introduce Spacewalk-18, a benchmark containing two tasks: (1) step recognition and (2) video question answering, over a dataset of temporally segmented and labeled tasks in International Space Station spacewalk recordings. In tandem, the two tasks quantify a model's ability to: (1) generalize to novel domains; (2) utilize long temporal context and multimodal (e.g. visual and speech) information. Our extensive experimental analysis highlights the challenges of Spacewalk-18, but also suggests best practices for domain generalization and long-form understanding. Notably, we discover a promising adaptation via summarization technique that leads to significant performance improvement without model fine-tuning. The Spacewalk-18 benchmark is released at https://brown-palm.github.io/Spacewalk-18/.
>
---
#### [replaced 118] Fast Gradient Methods for Data-Consistent Local Super-Resolution of Medical Images
- **分类: eess.IV; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2202.10875v2](https://arxiv.org/pdf/2202.10875v2)**

> **作者:** Junqi Tang; Guixian Xu; Jinglai Li
>
> **摘要:** In this work, we propose a new paradigm of iterative model-based reconstruction algorithms for providing real-time solution for zooming-in and refining a region of interest in medical and clinical tomographic images. This algorithmic framework is tailored for a clinical need in medical imaging practice that after a reconstruction of the full tomographic image, the clinician may believe that some critical parts of the image are not clear enough, and may wish to see clearer these regions of interest. A naive approach (which is highly not recommended) would be to perform the global reconstruction of a higher resolution image, which has two major limitations: first, it is computationally inefficient, and second, the image regularization is still applied globally, which may over-smooth some local regions. Furthermore, if one wishes to fine-tune the regularization parameter for local parts, it would be computationally infeasible in practice for the case of using global reconstruction. Our new iterative approaches for such tasks are based on jointly utilizing the measurement information, efficient up-sampling/down-sampling across image spaces, and locally adjusted image prior for efficient and high-quality post-processing. The numerical results in low-dose X-ray CT image local zoom-in demonstrate the effectiveness of our approach.
>
---
#### [replaced 119] ControlEvents: Controllable Synthesis of Event Camera Datawith Foundational Prior from Image Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22864v3](https://arxiv.org/pdf/2509.22864v3)**

> **作者:** Yixuan Hu; Yuxuan Xue; Simon Klenk; Daniel Cremers; Gerard Pons-Moll
>
> **备注:** Accepted to WACV2026. Project website:https://yuxuan-xue.com/controlevents/
>
> **摘要:** In recent years, event cameras have gained significant attention due to their bio-inspired properties, such as high temporal resolution and high dynamic range. However, obtaining large-scale labeled ground-truth data for event-based vision tasks remains challenging and costly. In this paper, we present ControlEvents, a diffusion-based generative model designed to synthesize high-quality event data guided by diverse control signals such as class text labels, 2D skeletons, and 3D body poses. Our key insight is to leverage the diffusion prior from foundation models, such as Stable Diffusion, enabling high-quality event data generation with minimal fine-tuning and limited labeled data. Our method streamlines the data generation process and significantly reduces the cost of producing labeled event datasets. We demonstrate the effectiveness of our approach by synthesizing event data for visual recognition, 2D skeleton estimation, and 3D body pose estimation. Our experiments show that the synthesized labeled event data enhances model performance in all tasks. Additionally, our approach can generate events based on unseen text labels during training, illustrating the powerful text-based generation capabilities inherited from foundation models.
>
---
#### [replaced 120] STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19854v2](https://arxiv.org/pdf/2511.19854v2)**

> **作者:** Jiankuo Zhao; Xiangyu Zhu; Zidu Wang; Zhen Lei
>
> **备注:** 17 pages, 14 figures
>
> **摘要:** Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions. The code will be publicly available.
>
---
#### [replaced 121] Discovering Concept Directions from Diffusion-based Counterfactuals via Latent Clustering
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.07073v2](https://arxiv.org/pdf/2505.07073v2)**

> **作者:** Payal Varshney; Adriano Lucieri; Christoph Balada; Andreas Dengel; Sheraz Ahmed
>
> **备注:** Accepted at Pattern Recognition Letters Journal (14 Pages)
>
> **摘要:** Concept-based explanations have emerged as an effective approach within Explainable Artificial Intelligence, enabling interpretable insights by aligning model decisions with human-understandable concepts. However, existing methods rely on computationally intensive procedures and struggle to efficiently capture complex, semantic concepts. This work introduces the Concept Directions via Latent Clustering (CDLC), which extracts global, class-specific concept directions by clustering latent difference vectors derived from factual and diffusion-generated counterfactual image pairs. CDLC reduces storage requirements by ~4.6% and accelerates concept discovery by ~5.3% compared to the baseline method, while requiring no GPU for clustering, thereby enabling efficient extraction of multidimensional semantic concepts across latent dimensions. This approach is validated on a real-world skin lesion dataset, demonstrating that the extracted concept directions align with clinically recognized dermoscopic features and, in some cases, reveal dataset-specific biases or unknown biomarkers. These results highlight that CDLC is interpretable, scalable, and applicable across high-stakes domains and diverse data modalities.
>
---
#### [replaced 122] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.21379v2](https://arxiv.org/pdf/2509.21379v2)**

> **作者:** Enrico Cassano; Riccardo Renzulli; Marco Nurisso; Mirko Zaffaroni; Alan Perotti; Marco Grangetto
>
> **摘要:** Concept unlearning in diffusion models is hampered by feature splitting, where concepts are distributed across many latent features, making their removal challenging and computationally expensive. We introduce SAEmnesia, a supervised sparse autoencoder framework that overcomes this by enforcing one-to-one concept-neuron mappings. By systematically labeling concepts during training, our method achieves feature centralization, binding each concept to a single, interpretable neuron. This enables highly targeted and efficient concept erasure. SAEmnesia reduces hyperparameter search by 96.7% and achieves a 9.2% improvement over the state-of-the-art on the UnlearnCanvas benchmark. Our method also demonstrates superior scalability in sequential unlearning, improving accuracy by 28.4% when removing nine objects, establishing a new standard for precise and controllable concept erasure. Moreover, SAEmnesia mitigates the possibility of generating unwanted content under adversarial attack and effectively removes nudity when evaluated with I2P.
>
---
#### [replaced 123] Network Inversion for Uncertainty-Aware Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23448v2](https://arxiv.org/pdf/2505.23448v2)**

> **作者:** Pirzada Suhail; Rehna Afroz; Gouranga Bala; Amit Sethi
>
> **摘要:** Out-of-distribution (OOD) detection and uncertainty estimation (UE) are critical components for building safe machine learning systems, especially in real-world scenarios where unexpected inputs are inevitable. However the two problems have, until recently, separately been addressed. In this work, we propose a novel framework that combines network inversion with classifier training to simultaneously address both OOD detection and uncertainty estimation. For a standard n-class classification task, we extend the classifier to an (n+1)-class model by introducing a "garbage" class, initially populated with random gaussian noise to represent outlier inputs. After each training epoch, we use network inversion to reconstruct input images corresponding to all output classes that initially appear as noisy and incoherent and are therefore excluded to the garbage class for retraining the classifier. This cycle of training, inversion, and exclusion continues iteratively till the inverted samples begin to resemble the in-distribution data more closely, with a significant drop in the uncertainty, suggesting that the classifier has learned to carve out meaningful decision boundaries while sanitising the class manifolds by pushing OOD content into the garbage class. During inference, this training scheme enables the model to effectively detect and reject OOD samples by classifying them into the garbage class. Furthermore, the confidence scores associated with each prediction can be used to estimate uncertainty for both in-distribution and OOD inputs. Our approach is scalable, interpretable, and does not require access to external OOD datasets or post-hoc calibration techniques while providing a unified solution to the dual challenges of OOD detection and uncertainty estimation.
>
---
#### [replaced 124] SAM 2++: Tracking Anything at Any Granularity
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18822v3](https://arxiv.org/pdf/2510.18822v3)**

> **作者:** Jiaming Zhang; Cheng Liang; Yichun Yang; Chenkai Zeng; Yutao Cui; Xinwen Zhang; Xin Zhou; Kai Ma; Gangshan Wu; Limin Wang
>
> **备注:** 8 pages, with supp
>
> **摘要:** Video tracking aims at finding the specific target in subsequent frames given its initial state. Due to the varying granularity of target states across different tasks, most existing trackers are tailored to a single task and heavily rely on custom-designed modules within the individual task, which limits their generalization and leads to redundancy in both model design and parameters. To unify video tracking tasks, we present SAM 2++, a unified model towards tracking at any granularity, including masks, boxes, and points. First, to extend target granularity, we design task-specific prompts to encode various task inputs into general prompt embeddings, and a unified decoder to unify diverse task results into a unified form pre-output. Next, to satisfy memory matching, the core operation of tracking, we introduce a task-adaptive memory mechanism that unifies memory across different granularities. Finally, we introduce a customized data engine to support tracking training at any granularity, producing a large and diverse video tracking dataset with rich annotations at three granularities, termed Tracking-Any-Granularity, which represents a comprehensive resource for training and benchmarking on unified tracking. Comprehensive experiments on multiple benchmarks confirm that SAM 2++ sets a new state of the art across diverse tracking tasks at different granularities, establishing a unified and robust tracking framework.
>
---
#### [replaced 125] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法中多目标冲突与人类偏好不一致问题。提出PrismAudio框架，通过四类专用思维链模块与对应奖励函数实现多维强化学习优化，并引入Fast-GRPO与AudioCanvas提升效率与评估可靠性，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18833v3](https://arxiv.org/pdf/2511.18833v3)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
#### [replaced 126] CLIP-like Model as a Foundational Density Ratio Estimator
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.22881v2](https://arxiv.org/pdf/2506.22881v2)**

> **作者:** Fumiya Uchiyama; Rintaro Yanagi; Shohei Taniguchi; Shota Takashiro; Masahiro Suzuki; Hirokatsu Kataoka; Yusuke Iwasawa; Yutaka Matsuo
>
> **摘要:** Density ratio estimation is a core concept in statistical machine learning because it provides a unified mechanism for tasks such as importance weighting, divergence estimation, and likelihood-free inference, but its potential in vision and language models has not been fully explored. Modern vision-language encoders such as CLIP and SigLIP are trained with contrastive objectives that implicitly optimize log density ratios between joint and marginal image-text distributions, which implicitly learn similarity scores proportional to log density ratios. However, prior work has largely focused on their embedding utility, and the density-ratio structure induced by contrastive learning has not been systematically examined or exploited in multimodal applications. To address this gap, we reinterpret CLIP-style models as pretrained and general-purpose density ratio estimators and show that this perspective enables new algorithmic capabilities. We present a unified explanation of how contrastive objectives estimate density ratios and propose two practical applications: Importance Weight Learning and KL divergence estimation. Our Importance Weight Learning method requires only a single additional prompt and improves F1 scores by up to 7 points. We further show that CLIP-based density ratios support estimation of KL divergences that quantify how conditioning on an image or text alters the distribution of the other modality. Through qualitative examples and an N-gram analysis of captions, we find that these divergences capture semantic diversity and mode structure in multimodal data. Leveraging this property, we introduce a simple KL-guided data curation method that achieves performance competitive with LAION2B filtering.
>
---
#### [replaced 127] MRIQT: Physics-Aware Diffusion Model for Image Quality Transfer in Neonatal Ultra-Low-Field MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13232v2](https://arxiv.org/pdf/2511.13232v2)**

> **作者:** Malek Al Abed; Sebiha Demir; Anne Groteklaes; Elodie Germani; Shahrooz Faghihroohi; Hemmen Sabir; Shadi Albarqouni
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Portable ultra-low-field MRI (uLF-MRI, 0.064 T) offers accessible neuroimaging for neonatal care but suffers from low signal-to-noise ratio and poor diagnostic quality compared to high-field (HF) MRI. We propose MRIQT, a 3D conditional diffusion framework for image quality transfer (IQT) from uLF to HF MRI. MRIQT combines realistic K-space degradation for physics-consistent uLF simulation, v-prediction with classifier-free guidance for stable image-to-image generation, and an SNR-weighted 3D perceptual loss for anatomical fidelity. The model denoises from a noised uLF input conditioned on the same scan, leveraging volumetric attention-UNet architecture for structure-preserving translation. Trained on a neonatal cohort with diverse pathologies, MRIQT surpasses recent GAN and CNN baselines in PSNR 15.3% with 1.78% over the state of the art, while physicians rated 85% of its outputs as good quality with clear pathology present. MRIQT enables high-fidelity, diffusion-based enhancement of portable ultra-low-field (uLF) MRI for deliable neonatal brain assessment.
>
---
#### [replaced 128] Monet: Reasoning in Latent Visual Space Beyond Images and Language
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.21395v2](https://arxiv.org/pdf/2511.21395v2)**

> **作者:** Qixun Wang; Yang Shi; Yifei Wang; Yuanxing Zhang; Pengfei Wan; Kun Gai; Xianghua Ying; Yisen Wang
>
> **摘要:** "Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at https://github.com/NOVAglow646/Monet.
>
---
#### [replaced 129] ParticleGS: Learning Neural Gaussian Particle Dynamics from Videos for Prior-free Physical Motion Extrapolation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20270v2](https://arxiv.org/pdf/2505.20270v2)**

> **作者:** Jinsheng Quan; Qiaowei Miao; Yichao Xu; Zizhuo Lin; Ying Li; Wei Yang; Zhihui Li; Yawei Luo
>
> **摘要:** The ability to extrapolate dynamic 3D scenes beyond the observed timeframe is fundamental to advancing physical world understanding and predictive modeling. Existing dynamic 3D reconstruction methods have achieved high-fidelity rendering of temporal interpolation, but typically lack physical consistency in predicting the future. To overcome this issue, we propose ParticleGS, a physics-based framework that reformulates dynamic 3D scenes as physically grounded systems. ParticleGS comprises three key components: 1) an encoder that decomposes the scene into static properties and initial dynamic physical fields; 2) an evolver based on Neural Ordinary Differential Equations (Neural ODEs) that learns continuous-time dynamics for motion extrapolation; and 3) a decoder that reconstructs 3D Gaussians from evolved particle states for rendering. Through this design, ParticleGS integrates physical reasoning into dynamic 3D representations, enabling accurate and consistent prediction of the future. Experiments show that ParticleGS achieves state-of-the-art performance in extrapolation while maintaining rendering quality comparable to leading dynamic 3D reconstruction methods.
>
---
#### [replaced 130] Dream4D: Lifting Camera-Controlled I2V towards Spatiotemporally Consistent 4D Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07769v2](https://arxiv.org/pdf/2508.07769v2)**

> **作者:** Xiaoyan Liu; Kangrui Li; Yuehao Song; Jiaxin Liu
>
> **备注:** Project Page: https://wanderer7-sk.github.io/Dream4D.github.io/
>
> **摘要:** The synthesis of spatiotemporally coherent 4D content presents fundamental challenges in computer vision, requiring simultaneous modeling of high-fidelity spatial representations and physically plausible temporal dynamics. Current approaches often struggle to maintain view consistency while handling complex scene dynamics, particularly in large-scale environments with multiple interacting elements. This work introduces Dream4D, a novel framework that bridges this gap through a synergy of controllable video generation and neural 4D reconstruction. Our approach seamlessly combines a two-stage architecture: it first predicts optimal camera trajectories from a single image using few-shot learning, then generates geometrically consistent multi-view sequences via a specialized pose-conditioned diffusion process, which are finally converted into a persistent 4D representation. This framework is the first to leverage both rich temporal priors from video diffusion models and geometric awareness of the reconstruction models, which significantly facilitates 4D generation and shows higher quality (e.g., mPSNR, mSSIM) over existing methods.
>
---
#### [replaced 131] ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对视觉语言模型在长视频序列中推理能力弱的问题，提出ROVER框架。通过递归分解视频为短时子任务，实现局部精准推理并保留全局上下文。在机器人操作任务中验证，显著提升任务进度估计、帧级推理和视频问答性能，降低幻觉，线性降低时间复杂度。**

- **链接: [https://arxiv.org/pdf/2508.01943v2](https://arxiv.org/pdf/2508.01943v2)**

> **作者:** Philip Schroeder; Ondrej Biza; Thomas Weng; Hongyin Luo; James Glass
>
> **摘要:** Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: https://rover-vlm.github.io
>
---
#### [replaced 132] CzechLynx: A Dataset for Individual Identification and Pose Estimation of the Eurasian Lynx
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.04931v2](https://arxiv.org/pdf/2506.04931v2)**

> **作者:** Lukas Picek; Elisa Belotti; Michal Bojda; Ludek Bufka; Vojtech Cermak; Martin Dula; Rostislav Dvorak; Luboslav Hrdy; Miroslav Jirik; Vaclav Kocourek; Josefa Krausova; Jirı Labuda; Jakub Straka; Ludek Toman; Vlado Trulık; Martin Vana; Miroslav Kutal
>
> **摘要:** We introduce CzechLynx, the first large-scale, open-access dataset for individual identification, pose estimation, and instance segmentation of the Eurasian lynx (Lynx lynx). CzechLynx contains 39,760 camera trap images annotated with segmentation masks, identity labels, and 20-point skeletons and covers 319 unique individuals across 15 years of systematic monitoring in two geographically distinct regions: southwest Bohemia and the Western Carpathians. In addition to the real camera trap data, we provide a large complementary set of photorealistic synthetic images and a Unity-based generation pipeline with diffusion-based text-to-texture modeling, capable of producing arbitrarily large amounts of synthetic data spanning diverse environments, poses, and coat-pattern variations. To enable systematic testing across realistic ecological scenarios, we define three complementary evaluation protocols: (i) geo-aware, (ii) time-aware open-set, and (iii) time-aware closed-set, covering cross-regional and long-term monitoring settings. With the provided resources, CzechLynx offers a unique, flexible benchmark for robust evaluation of computer vision and machine learning models across realistic ecological scenarios.
>
---
#### [replaced 133] MoRe: Monocular Geometry Refinement via Graph Optimization for Cross-View Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07119v2](https://arxiv.org/pdf/2510.07119v2)**

> **作者:** Dongki Jung; Jaehoon Choi; Yonghan Lee; Sungmin Eum; Heesung Kwon; Dinesh Manocha
>
> **摘要:** Monocular 3D foundation models offer an extensible solution for perception tasks, making them attractive for broader 3D vision applications. In this paper, we propose MoRe, a training-free Monocular Geometry Refinement method designed to improve cross-view consistency and achieve scale alignment. To induce inter-frame relationships, our method employs feature matching between frames to establish correspondences. Rather than applying simple least squares optimization on these matched points, we formulate a graph-based optimization framework that performs local planar approximation using the estimated 3D points and surface normals estimated by monocular foundation models. This formulation addresses the scale ambiguity inherent in monocular geometric priors while preserving the underlying 3D structure. We further demonstrate that MoRe not only enhances 3D reconstruction but also improves novel view synthesis, particularly in sparse view rendering scenarios.
>
---
#### [replaced 134] Enhanced Partially Relevant Video Retrieval through Inter- and Intra-Sample Analysis with Coherence Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.19637v5](https://arxiv.org/pdf/2504.19637v5)**

> **作者:** Junlong Ren; Gangjian Zhang; Yu Hu; Jian Shu; Hui Xiong; Hao Wang
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) aims to retrieve the target video that is partially relevant to the text query. The primary challenge in PRVR arises from the semantic asymmetry between textual and visual modalities, as videos often contain substantial content irrelevant to the query. Existing methods coarsely align paired videos and text queries to construct the semantic space, neglecting the critical cross-modal dual nature inherent in this task: inter-sample correlation and intra-sample redundancy. To this end, we propose a novel PRVR framework to systematically exploit these two characteristics. Our framework consists of three core modules. First, the Inter Correlation Enhancement (ICE) module captures inter-sample correlation by identifying semantically similar yet unpaired text queries and video moments, combining them to form pseudo-positive pairs for more robust semantic space construction. Second, the Intra Redundancy Mining (IRM) module mitigates intra-sample redundancy by mining redundant moment features and distinguishing them from query-relevant moments, encouraging the model to learn more discriminative representations. Finally, to reinforce these modules, we introduce the Temporal Coherence Prediction (TCP) module, enhancing discrimination of fine-grained moment-level semantics by training the model to predict the original temporal order of randomly shuffled video sequences. Extensive experiments demonstrate the superiority of our method, achieving state-of-the-art results.
>
---
#### [replaced 135] I-INR: Iterative Implicit Neural Representations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.17364v3](https://arxiv.org/pdf/2504.17364v3)**

> **作者:** Ali Haider; Muhammad Salman Ali; Maryam Qamar; Tahir Khalil; Soo Ye Kim; Jihyong Oh; Enzo Tartaglione; Sung-Ho Bae
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Implicit Neural Representations (INRs) have revolutionized signal processing and computer vision by modeling signals as continuous, differentiable functions parameterized by neural networks. However, their inherent formulation as a regression problem makes them prone to regression to the mean, limiting their ability to capture fine details, retain high-frequency information, and handle noise effectively. To address these challenges, we propose Iterative Implicit Neural Representations (I-INRs) a novel plug-and-play framework that enhances signal reconstruction through an iterative refinement process. I-INRs effectively recover high-frequency details, improve robustness to noise, and achieve superior reconstruction quality. Our framework seamlessly integrates with existing INR architectures, delivering substantial performance gains across various tasks. Extensive experiments show that I-INRs outperform baseline methods, including WIRE, SIREN, and Gauss, in diverse computer vision applications such as image restoration, image denoising, and object occupancy prediction.
>
---
#### [replaced 136] SegDINO3D: 3D Instance Segmentation Empowered by Both Image-Level and Object-Level 2D Features
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16098v2](https://arxiv.org/pdf/2509.16098v2)**

> **作者:** Jinyuan Qu; Hongyang Li; Xingyu Chen; Shilong Liu; Yukai Shi; Tianhe Ren; Ruitao Jing; Lei Zhang
>
> **摘要:** In this paper, we present SegDINO3D, a novel Transformer encoder-decoder framework for 3D instance segmentation. As 3D training data is generally not as sufficient as 2D training images, SegDINO3D is designed to fully leverage 2D representation from a pre-trained 2D detection model, including both image-level and object-level features, for improving 3D representation. SegDINO3D takes both a point cloud and its associated 2D images as input. In the encoder stage, it first enriches each 3D point by retrieving 2D image features from its corresponding image views and then leverages a 3D encoder for 3D context fusion. In the decoder stage, it formulates 3D object queries as 3D anchor boxes and performs cross-attention from 3D queries to 2D object queries obtained from 2D images using the 2D detection model. These 2D object queries serve as a compact object-level representation of 2D images, effectively avoiding the challenge of keeping thousands of image feature maps in the memory while faithfully preserving the knowledge of the pre-trained 2D model. The introducing of 3D box queries also enables the model to modulate cross-attention using the predicted boxes for more precise querying. SegDINO3D achieves the state-of-the-art performance on the ScanNetV2 and ScanNet200 3D instance segmentation benchmarks. Notably, on the challenging ScanNet200 dataset, SegDINO3D significantly outperforms prior methods by +8.6 and +6.8 mAP on the validation and hidden test sets, respectively, demonstrating its superiority.
>
---
#### [replaced 137] OpenDance: Multimodal Controllable 3D Dance Generation with Large-scale Internet Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07565v2](https://arxiv.org/pdf/2506.07565v2)**

> **作者:** Jinlu Zhang; Zixi Kang; Libin Liu; Jianlong Chang; Qi Tian; Feng Gao; Yizhou Wang
>
> **摘要:** Music-driven 3D dance generation offers significant creative potential, yet practical applications demand versatile and multimodal control. As the highly dynamic and complex human motion covering various styles and genres, dance generation requires satisfying diverse conditions beyond just music (e.g., spatial trajectories, keyframe gestures, or style descriptions). However, the absence of a large-scale and richly annotated dataset severely hinders progress. In this paper, we build OpenDanceSet, an extensive human dance dataset comprising over 100 hours across 14 genres and 147 subjects. Each sample has rich annotations to facilitate robust cross-modal learning: 3D motion, paired music, 2D keypoints, trajectories, and expert-annotated text descriptions. Furthermore, we propose OpenDanceNet, a unified masked modeling framework for controllable dance generation, including a disentangled auto-encoder and a multimodal joint-prediction Transformer. OpenDanceNet supports generation conditioned on music and arbitrary combinations of text, keypoints, or trajectories. Comprehensive experiments demonstrate that our work achieves high-fidelity synthesis with strong diversity and realistic physical contacts, while also offering flexible control over spatial and stylistic conditions. Project Page: https://open-dance.github.io
>
---
#### [replaced 138] A Neurosymbolic Framework for Interpretable Cognitive Attack Detection in Augmented Reality
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.09185v3](https://arxiv.org/pdf/2508.09185v3)**

> **作者:** Rongqian Chen; Allison Andreyev; Yanming Xiu; Joshua Chilukuri; Shunav Sen; Mahdi Imani; Bin Li; Maria Gorlatova; Gang Tan; Tian Lan
>
> **摘要:** Augmented Reality (AR) enriches human perception by overlaying virtual elements onto the physical world. However, this tight coupling between virtual and real content makes AR vulnerable to cognitive attacks: manipulations that distort users' semantic understanding of the environment. Existing detection methods largely focus on visual inconsistencies at the pixel or image level, offering limited semantic reasoning or interpretability. To address these limitations, we introduce CADAR, a neuro-symbolic framework for cognitive attack detection in AR that integrates neural and symbolic reasoning. CADAR fuses multimodal vision-language representations from pre-trained models into a perception graph that captures objects, relations, and temporal contextual salience. Building on this structure, a particle-filter-based statistical reasoning module infers anomalies in semantic dynamics to reveal cognitive attacks. This combination provides both the adaptability of modern vision-language models and the interpretability of probabilistic symbolic reasoning. Preliminary experiments on an AR cognitive-attack dataset demonstrate consistent advantages over existing approaches, highlighting the potential of neuro-symbolic methods for robust and interpretable AR security.
>
---
#### [replaced 139] HiGFA: Hierarchical Guidance for Fine-grained Data Augmentation with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12547v3](https://arxiv.org/pdf/2511.12547v3)**

> **作者:** Zhiguang Lu; Qianqian Xu; Peisong Wen; Siran Dai; Qingming Huang
>
> **摘要:** Generative diffusion models show promise for data augmentation. However, applying them to fine-grained tasks presents a significant challenge: ensuring synthetic images accurately capture the subtle, category-defining features critical for high fidelity. Standard approaches, such as text-based Classifier-Free Guidance (CFG), often lack the required specificity, potentially generating misleading examples that degrade fine-grained classifier performance. To address this, we propose Hierarchically Guided Fine-grained Augmentation (HiGFA). HiGFA leverages the temporal dynamics of the diffusion sampling process. It employs strong text and transformed contour guidance with fixed strengths in the early-to-mid sampling stages to establish overall scene, style, and structure. In the final sampling stages, HiGFA activates a specialized fine-grained classifier guidance and dynamically modulates the strength of all guidance signals based on prediction confidence. This hierarchical, confidence-driven orchestration enables HiGFA to generate diverse yet faithful synthetic images by intelligently balancing global structure formation with precise detail refinement. Experiments on several FGVC datasets demonstrate the effectiveness of HiGFA.
>
---
#### [replaced 140] Lacking Data? No worries! How synthetic images can alleviate image scarcity in wildlife surveys: a case study with muskox (Ovibos moschatus)
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11882v2](https://arxiv.org/pdf/2511.11882v2)**

> **作者:** Simon Durand; Samuel Foucher; Alexandre Delplanque; Joëlle Taillon; Jérôme Théau
>
> **备注:** 34 pages, 10 figures, submitted to Remote Sensing in Ecology and Conservation
>
> **摘要:** Accurate population estimates are essential for wildlife management, providing critical insights into species abundance and distribution. Traditional survey methods, including visual aerial counts and GNSS telemetry tracking, are widely used to monitor muskox populations in Arctic regions. These approaches are resource intensive and constrained by logistical challenges. Advances in remote sensing, artificial intelligence, and high resolution aerial imagery offer promising alternatives for wildlife detection. Yet, the effectiveness of deep learning object detection models (ODMs) is often limited by small datasets, making it challenging to train robust ODMs for sparsely distributed species like muskoxen. This study investigates the integration of synthetic imagery (SI) to supplement limited training data and improve muskox detection in zero shot (ZS) and few-shot (FS) settings. We compared a baseline model trained on real imagery with 5 ZS and 5 FS models that incorporated progressively more SI in the training set. For the ZS models, where no real images were included in the training set, adding SI improved detection performance. As more SI were added, performance in precision, recall and F1 score increased, but eventually plateaued, suggesting diminishing returns when SI exceeded 100% of the baseline model training dataset. For FS models, combining real and SI led to better recall and slightly higher overall accuracy compared to using real images alone, though these improvements were not statistically significant. Our findings demonstrate the potential of SI to train accurate ODMs when data is scarce, offering important perspectives for wildlife monitoring by enabling rare or inaccessible species to be monitored and to increase monitoring frequency. This approach could be used to initiate ODMs without real data and refine it as real images are acquired over time.
>
---
#### [replaced 141] Taming generative video models for zero-shot optical flow extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.09082v2](https://arxiv.org/pdf/2507.09082v2)**

> **作者:** Seungwoo Kim; Khai Loong Aw; Klemen Kotar; Cristobal Eyzaguirre; Wanhee Lee; Yunong Liu; Jared Watrous; Stefan Stojanov; Juan Carlos Niebles; Jiajun Wu; Daniel L. K. Yamins
>
> **备注:** Project webpage: https://neuroailab.github.io/projects/kl_tracing
>
> **摘要:** Extracting optical flow from videos remains a core computer vision problem. Motivated by the recent success of large general-purpose models, we ask whether frozen self-supervised video models trained only to predict future frames can be prompted, without fine-tuning, to output flow. Prior attempts to read out depth or illumination from video generators required fine-tuning; that strategy is ill-suited for flow, where labeled data is scarce and synthetic datasets suffer from a sim-to-real gap. Inspired by the Counterfactual World Model (CWM) paradigm, which can obtain point-wise correspondences by injecting a small tracer perturbation into a next-frame predictor and tracking its propagation, we extend this idea to generative video models for zero-shot flow extraction. We explore several popular architectures and find that successful zero-shot flow extraction in this manner is aided by three model properties: (1) distributional prediction of future frames (avoiding blurry or noisy outputs); (2) factorized latents that treat each spatio-temporal patch independently; and (3) random-access decoding that can condition on any subset of future pixels. These properties are uniquely present in the recently introduced Local Random Access Sequence (LRAS) architecture. Building on LRAS, we propose KL-tracing: a novel test-time inference procedure that injects a localized perturbation into the first frame, rolls out the model one step, and computes the Kullback-Leibler divergence between perturbed and unperturbed predictive distributions. Without any flow-specific fine-tuning, our method is competitive with state-of-the-art, task-specific models on the real-world TAP-Vid DAVIS benchmark and the synthetic TAP-Vid Kubric. Our results show that counterfactual prompting of controllable generative video models is an effective alternative to supervised or photometric-loss methods for high-quality flow.
>
---
#### [replaced 142] A Sampling-Based Domain Generalization Study with Diffusion Generative Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2310.09213v3](https://arxiv.org/pdf/2310.09213v3)**

> **作者:** Ye Zhu; Yu Wu; Duo Xu; Zhiwei Deng; Yan Yan; Olga Russakovsky
>
> **备注:** NeurIPS 2025 Workshop on Frontiers in Probabilistic Inference: Learning meets Sampling. Code can be found at https://github.com/L-YeZhu/DiscoveryDiff
>
> **摘要:** In this work, we investigate the domain generalization capabilities of diffusion models in the context of synthesizing images that are distinct from the training data. Instead of fine-tuning, we tackle this challenge from a sampling-based perspective using frozen, pre-trained diffusion models. Specifically, we demonstrate that arbitrary out-of-domain (OOD) images establish Gaussian priors in the latent spaces of a given model after inversion, and that these priors are separable from those of the original training domain. This OOD latent property allows us to synthesize new images of the target unseen domain by discovering qualified OOD latent encodings in the inverted noisy spaces, without altering the pre-trained models. Our cross-model and cross-domain experiments show that the proposed sampling-based method can expand the latent space and generate unseen images without impairing the generation quality of the original domain. We also showcase a practical application of our approach using astrophysical data, highlighting the potential of this generalization paradigm in data-sparse fields such as scientific exploration.
>
---
#### [replaced 143] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18719v2](https://arxiv.org/pdf/2511.18719v2)**

> **作者:** Ziqi Ni; Yuanzhi Liang; Rui Li; Yi Zhou; Haibing Huang; Chi Zhang; Xuelong Li
>
> **摘要:** Reinforcement learning (RL) has become a powerful tool for post-training visual generative models, with Group Relative Policy Optimization (GRPO) increasingly used to align generators with human preferences. However, existing GRPO pipelines rely on a single scalar reward per sample, treating each image or video as a holistic entity and ignoring the rich spatial and temporal structure of visual content. This coarse supervision hinders the correction of localized artifacts and the modeling of fine-grained perceptual cues. We introduce Visual Preference Policy Optimization (ViPO), a GRPO variant that lifts scalar feedback into structured, pixel-level advantages. ViPO employs a Perceptual Structuring Module that uses pretrained vision backbones to construct spatially and temporally aware advantage maps, redistributing optimization pressure toward perceptually important regions while preserving the stability of standard GRPO. Across both image and video benchmarks, ViPO consistently outperforms vanilla GRPO, improving in-domain alignment with human-preference rewards and enhancing generalization on out-of-domain evaluations. The method is architecture-agnostic, lightweight, and fully compatible with existing GRPO training pipelines, providing a more expressive and informative learning signal for visual generation.
>
---
#### [replaced 144] CountSteer: Steering Attention for Object Counting in Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11253v2](https://arxiv.org/pdf/2511.11253v2)**

> **作者:** Hyemin Boo; Hyoryung Kim; Myungjin Lee; Seunghyeon Lee; Jiyoung Lee; Jang-Hwan Choi; Hyunsoo Cho
>
> **备注:** Accepted to AAAI 2026 Workshop on Shaping Responsible Synthetic Data in the Era of Foundation Models (RSD)
>
> **摘要:** Text-to-image diffusion models generate realistic and coherent images but often fail to follow numerical instructions in text, revealing a gap between language and visual representation. Interestingly, we found that these models are not entirely blind to numbers-they are implicitly aware of their own counting accuracy, as their internal signals shift in consistent ways depending on whether the output meets the specified count. This observation suggests that the model already encodes a latent notion of numerical correctness, which can be harnessed to guide generation more precisely. Building on this intuition, we introduce CountSteer, a training-free method that improves generation of specified object counts by steering the model's cross-attention hidden states during inference. In our experiments, CountSteer improved object-count accuracy by about 4% without compromising visual quality, demonstrating a simple yet effective step toward more controllable and semantically reliable text-to-image generation.
>
---
#### [replaced 145] Watch and Learn: Learning to Use Computers from Online Videos
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04673v2](https://arxiv.org/pdf/2510.04673v2)**

> **作者:** Chan Hee Song; Yiwen Song; Palash Goyal; Yu Su; Oriana Riva; Hamid Palangi; Tomas Pfister
>
> **摘要:** Computer-using agents (CUAs) must plan task workflows across diverse and evolving applications, yet progress is limited by the lack of large-scale, high-quality training data. Existing datasets are narrow, static, and costly to annotate, while synthetic data often yields oversimplified or misaligned behaviors. We present Watch & Learn (W&L), a framework that converts readily available Internet videos of human computer use into executable UI trajectories at scale. Instead of directly generating actions or relying on handcrafted heuristics, we cast trajectory annotation as an inverse dynamics problem that predicts user actions from consecutive screen states, which simplifies learning and generalizes across domains. Through a task-aware retrieval and labeling pipeline, W&L yields over 53K high-quality trajectories that enhance CUAs both as in-context exemplars and as supervised training data. On OSWorld, it consistently improves general-purpose and specialized CUAs, while on WindowsAgentArena it achieves state-of-the-art performance among 7B-scale models under the 15-step limit. These results show that web-scale human demonstration videos can serve as a practical and scalable foundation for advancing real-world CUAs.
>
---
#### [replaced 146] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.04675v2](https://arxiv.org/pdf/2511.04675v2)**

> **作者:** Jinlai Liu; Jian Han; Bin Yan; Hui Wu; Fengda Zhu; Xing Wang; Yi Jiang; Bingyue Peng; Zehuan Yuan
>
> **备注:** NeurIPS 2025 Oral
>
> **摘要:** We introduce InfinityStar, a unified spacetime autoregressive framework for high-resolution image and dynamic video synthesis. Building on the recent success of autoregressive modeling in both vision and language, our purely discrete approach jointly captures spatial and temporal dependencies within a single architecture. This unified design naturally supports a variety of generation tasks such as text-to-image, text-to-video, image-to-video, and long interactive video synthesis via straightforward temporal autoregression. Extensive experiments demonstrate that InfinityStar scores 83.74 on VBench, outperforming all autoregressive models by large margins, even surpassing some diffusion competitors like HunyuanVideo. Without extra optimizations, our model generates a 5s, 720p video approximately 10x faster than leading diffusion-based methods. To our knowledge, InfinityStar is the first discrete autoregressive video generator capable of producing industrial level 720p videos. We release all code and models to foster further research in efficient, high-quality video generation.
>
---
#### [replaced 147] Leveraging Semantic Attribute Binding for Free-Lunch Color Control in Diffusion Models
- **分类: cs.GR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.09864v2](https://arxiv.org/pdf/2503.09864v2)**

> **作者:** Héctor Laria; Alexandra Gomez-Villa; Jiang Qin; Muhammad Atif Butt; Bogdan Raducanu; Javier Vazquez-Corral; Joost van de Weijer; Kai Wang
>
> **备注:** WACV 2026. Project page: https://hecoding.github.io/colorwave-page
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled remarkable control over various attributes, yet precise color specification remains a fundamental challenge. Existing approaches, such as ColorPeel, rely on model personalization, requiring additional optimization and limiting flexibility in specifying arbitrary colors. In this work, we introduce ColorWave, a novel training-free approach that achieves exact RGB-level color control in diffusion models without fine-tuning. By systematically analyzing the cross-attention mechanisms within IP-Adapter, we uncover an implicit binding between textual color descriptors and reference image features. Leveraging this insight, our method rewires these bindings to enforce precise color attribution while preserving the generative capabilities of pretrained models. Our approach maintains generation quality and diversity, outperforming prior methods in accuracy and applicability across diverse object categories. Through extensive evaluations, we demonstrate that ColorWave establishes a new paradigm for structured, color-consistent diffusion-based image synthesis.
>
---
