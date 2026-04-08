# 计算机视觉 cs.CV

- **最新发布 134 篇**

- **更新 98 篇**

## 最新发布

#### [new 001] CLIP-Guided Data Augmentation for Night-Time Image Dehazing
- **分类: cs.CV**

- **简介: 该论文属于夜间图像去雾任务，解决夜间图像退化复杂、数据稀缺的问题。通过CLIP引导的数据增强和分阶段训练，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.05500](https://arxiv.org/pdf/2604.05500)**

> **作者:** Xining Ge; Weijun Yuan; Gengjia Chang; Xuyang Li; Shuhong Liu
>
> **摘要:** Nighttime image dehazing faces a more complex degradation pattern than its daytime counterpart, as haze scattering couples with low illumination, non-uniform lighting, and strong light interference. Under limited supervision, this complexity aggravates domain drift and training instability, since target-domain samples are scarce while naively introducing external data may weaken adaptation due to distribution mismatch. This paper presents our solution to the NTIRE 2026 Night Time Image Dehazing Challenge, built as a unified framework that integrates domain-aligned data construction, stage-wise training, and inference-time enhancement. Specifically, a pre-trained CLIP visual encoder screens candidate external samples by similarity to construct training data closer to the target domain. NAFNet is then trained in two stages, first adapting to the target domain and then expanding to broader degradation patterns. At inference time, TLC, x8 self-ensemble, and weighted snapshot fusion are combined to improve output stability. Rather than relying on complex network redesign, the proposed framework offers a practical and effective pipeline for nighttime image dehazing.
>
---
#### [new 002] Evaluation Before Generation: A Paradigm for Robust Multimodal Sentiment Analysis with Missing Modalities
- **分类: cs.CV**

- **简介: 该论文属于多模态情感分析任务，解决缺失模态导致的模型性能下降问题。提出框架通过评估缺失模态、解耦提示和动态加权，提升模型鲁棒性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.05558](https://arxiv.org/pdf/2604.05558)**

> **作者:** Rongfei Chen; Tingting Zhang; Xiaoyu Shen; Wei Zhang
>
> **备注:** 6 pages, 3 figures, conference
>
> **摘要:** The missing modality problem poses a fundamental challenge in multimodal sentiment analysis, significantly degrading model accuracy and generalization in real world scenarios. Existing approaches primarily improve robustness through prompt learning and pre trained models. However, two limitations remain. First, the necessity of generating missing modalities lacks rigorous evaluation. Second, the structural dependencies among multimodal prompts and their global coherence are insufficiently explored. To address these issues, a Prompt based Missing Modality Adaptation framework is proposed. A Missing Modality Evaluator is introduced at the input stage to dynamically assess the importance of missing modalities using pretrained models and pseudo labels, thereby avoiding low quality data imputation. Building on this, a Modality invariant Prompt Disentanglement module decomposes shared prompts into modality specific private prompts to capture intrinsic local correlations and improve representation quality. In addition, a Dynamic Prompt Weighting module computes mutual information based weights from cross attention outputs to adaptively suppress interference from missing modalities. To enhance global consistency, a Multi level Prompt Dynamic Connection module integrates shared prompts with self attention outputs through residual connections, leveraging global prompt priors to strengthen key guidance features. Extensive experiments on three public benchmarks, including CMU MOSI, CMU MOSEI, and CH SIMS, demonstrate that the proposed framework achieves state of the art performance and stable results under diverse missing modality settings. The implementation is available at this https URL
>
---
#### [new 003] Evaluation of Randomization through Style Transfer for Enhanced Domain Generalization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于域泛化任务，旨在解决合成数据训练模型在真实场景中泛化能力差的问题。通过系统研究风格迁移的三个设计因素，提出StyleMixDG增强泛化性能。**

- **链接: [https://arxiv.org/pdf/2604.05616](https://arxiv.org/pdf/2604.05616)**

> **作者:** Dustin Eisenhardt; Timothy Schaumlöffel; Alperen Kantarci; Gemma Roig
>
> **摘要:** Deep learning models for computer vision often suffer from poor generalization when deployed in real-world settings, especially when trained on synthetic data due to the well-known Sim2Real gap. Despite the growing popularity of style transfer as a data augmentation strategy for domain generalization, the literature contains unresolved contradictions regarding three key design axes: the diversity of the style pool, the role of texture complexity, and the choice of style source. We present a systematic empirical study that isolates and evaluates each of these factors for driving scene understanding, resolving inconsistencies in prior work. Our findings show that (i) expanding the style pool yields larger gains than repeated augmentation with few styles, (ii) texture complexity has no significant effect when the pool is sufficiently large, and (iii) diverse artistic styles outperform domain-aligned alternatives. Guided by these insights, we derive StyleMixDG (Style-Mixing for Domain Generalization), a lightweight, model-agnostic augmentation recipe that requires no architectural modifications or additional losses. Evaluated on the GTAV $\rightarrow$ {BDD100k, Cityscapes, Mapillary Vistas} benchmark, StyleMixDG demonstrates consistent improvements over strong baselines, confirming that the empirically identified design principles translate into practical gains. The code will be released on GitHub.
>
---
#### [new 004] Physics-Aware Video Instance Removal Benchmark
- **分类: cs.CV**

- **简介: 该论文属于视频实例移除任务，旨在解决移除目标物体后保持背景物理一致性的问题。工作包括构建PVIR基准并评估多种方法的表现。**

- **链接: [https://arxiv.org/pdf/2604.05898](https://arxiv.org/pdf/2604.05898)**

> **作者:** Zirui Li; Xinghao Chen; Lingyu Jiang; Dengzhe Hou; Fangzhou Lin; Kazunori Yamada; Xiangbo Gao; Zhengzhong Tu
>
> **摘要:** Video Instance Removal (VIR) requires removing target objects while maintaining background integrity and physical consistency, such as specular reflections and illumination interactions. Despite advancements in text-guided editing, current benchmarks primarily assess visual plausibility, often overlooking the physical causalities, such as lingering shadows, triggered by object removal. We introduce the Physics-Aware Video Instance Removal (PVIR) benchmark, featuring 95 high-quality videos annotated with instance-accurate masks and removal prompts. PVIR is partitioned into Simple and Hard subsets, the latter explicitly targeting complex physical interactions. We evaluate four representative methods, PISCO-Removal, UniVideo, DiffuEraser, and CoCoCo, using a decoupled human evaluation protocol across three dimensions to isolate semantic, visual, and spatial failures: instruction following, rendering quality, and edit exclusivity. Our results show that PISCO-Removal and UniVideo achieve state-of-the-art performance, while DiffuEraser frequently introduces blurring artifacts and CoCoCo struggles significantly with instruction following. The persistent performance drop on the Hard subset highlights the ongoing challenge of recovering complex physical side effects.
>
---
#### [new 005] SGANet: Semantic and Geometric Alignment for Multimodal Multi-view Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态多视角异常检测任务，旨在解决因视角变化和模态差异导致的特征不一致问题。提出SGANet框架，通过语义与几何对齐提升异常检测效果。**

- **链接: [https://arxiv.org/pdf/2604.05632](https://arxiv.org/pdf/2604.05632)**

> **作者:** Letian Bai; Chengyu Tao; Juan Du
>
> **摘要:** Multi-view anomaly detection aims to identify surface defects on complex objects using observations captured from multiple viewpoints. However, existing unsupervised methods often suffer from feature inconsistency arising from viewpoint variations and modality discrepancies. To address these challenges, we propose a Semantic and Geometric Alignment Network (SGANet), a unified framework for multimodal multi-view anomaly detection that effectively combines semantic and geometric alignment to learn physically coherent feature representations across viewpoints and modalities. SGANet consists of three key components. The Selective Cross-view Feature Refinement Module (SCFRM) selectively aggregates informative patch features from adjacent views to enhance cross-view feature interaction. The Semantic-Structural Patch Alignment (SSPA) enforces semantic alignment across modalities while maintaining structural consistency under viewpoint transformations. The Multi-View Geometric Alignment (MVGA) further aligns geometrically corresponding patches across viewpoints. By jointly modeling feature interaction, semantic and structural consistency, and global geometric correspondence, SGANet effectively enhances anomaly detection performance in multimodal multi-view settings. Extensive experiments on the SiM3D and Eyecandies datasets demonstrate that SGANet achieves state-of-the-art performance in both anomaly detection and localization, validating its effectiveness in realistic industrial scenarios.
>
---
#### [new 006] PoM: A Linear-Time Replacement for Attention with the Polynomial Mixer
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PoM，一种线性复杂度的token混合机制，替代自注意力。解决长序列计算成本高的问题，适用于多种任务，性能与注意力模型相当。**

- **链接: [https://arxiv.org/pdf/2604.06129](https://arxiv.org/pdf/2604.06129)**

> **作者:** David Picard; Nicolas Dufour; Lucas Degeorge; Arijit Ghosh; Davide Allegro; Tom Ravaud; Yohann Perron; Corentin Sautier; Zeynep Sonat Baltaci; Fei Meng; Syrine Kalleli; Marta López-Rauhut; Thibaut Loiseau; Ségolène Albouy; Raphael Baena; Elliot Vincent; Loic Landrieu
>
> **备注:** Accepted to CVPR Findings 2026
>
> **摘要:** This paper introduces the Polynomial Mixer (PoM), a novel token mixing mechanism with linear complexity that serves as a drop-in replacement for self-attention. PoM aggregates input tokens into a compact representation through a learned polynomial function, from which each token retrieves contextual information. We prove that PoM satisfies the contextual mapping property, ensuring that transformers equipped with PoM remain universal sequence-to-sequence approximators. We replace standard self-attention with PoM across five diverse domains: text generation, handwritten text recognition, image generation, 3D modeling, and Earth observation. PoM matches the performance of attention-based models while drastically reducing computational cost when working with long sequences. The code is available at this https URL.
>
---
#### [new 007] WRF4CIR: Weight-Regularized Fine-Tuning Network for Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，解决CIR中因数据有限导致的过拟合问题。通过引入权重正则化方法WRF4CIR，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.05583](https://arxiv.org/pdf/2604.05583)**

> **作者:** Yizhuo Xu; Chaojian Yu; Yuanjie Shao; Tongliang Liu; Qinmu Peng; Xinge You
>
> **摘要:** Composed Image Retrieval (CIR) task aims to retrieve target images based on reference images and modification texts. Current CIR methods primarily rely on fine-tuning vision-language pre-trained models. However, we find that these approaches commonly suffer from severe overfitting, posing challenges for CIR with limited triplet data. To better understand this issue, we present a systematic study of overfitting in VLP-based CIR, revealing a significant and previously overlooked generalization gap across different models and datasets. Motivated by these findings, we introduce WRF4CIR, a Weight-Regularized Fine-tuning network for CIR. Specifically, during the fine-tuning process, we apply adversarial perturbations to the model weights for regularization, where these perturbations are generated in the opposite direction of gradient descent. Intuitively, WRF4CIR increases the difficulty of fitting the training data, which helps mitigate overfitting in CIR under limited triplet supervision. Extensive experiments on benchmark datasets demonstrate that WRF4CIR significantly narrows the generalization gap and achieves substantial improvements over existing methods.
>
---
#### [new 008] Multi-Modal Landslide Detection from Sentinel-1 SAR and Sentinel-2 Optical Imagery Using Multi-Encoder Vision Transformers and Ensemble Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于滑坡检测任务，旨在提升灾害监测的准确性。通过融合Sentinel-1 SAR与Sentinel-2光学数据，利用多编码器视觉Transformer和集成学习方法，实现高效滑坡识别。**

- **链接: [https://arxiv.org/pdf/2604.05959](https://arxiv.org/pdf/2604.05959)**

> **作者:** Ioannis Nasios
>
> **摘要:** Landslides represent a major geohazard with severe impacts on human life, infrastructure, and ecosystems, underscoring the need for accurate and timely detection approaches to support disaster risk reduction. This study proposes a modular, multi-model framework that fuses Sentinel-2 optical imagery with Sentinel-1 Synthetic Aperture Radar (SAR) data, for robust landslide detection. The methodology leverages multi-encoder vision transformers, where each data modality is processed through separate lightweight pretrained encoders, achieving strong performance in landslide detection. In addition, the integration of multiple models, particularly the combination of neural networks and gradient boosting models (LightGBM and XGBoost), demonstrates the power of ensemble learning to further enhance accuracy and robustness. Derived spectral indices, such as NDVI, are integrated alongside original bands to enhance sensitivity to vegetation and surface changes. The proposed methodology achieves a state-of-the-art F1 score of 0.919 on landslide detection, addressing a patch-based classification task rather than pixel-level segmentation and operating without pre-event Sentinel-2 data, highlighting its effectiveness in a non-classical change detection setting. It also demonstrated top performance in a machine learning competition, achieving a strong balance between precision and recall and highlighting the advantages of explicitly leveraging the complementary strengths of optical and radar data. The conducted experiments and research also emphasize scalability and operational applicability, enabling flexible configurations with optical-only, SAR-only, or combined inputs, and offering a transferable framework for broader natural hazard monitoring and environmental change applications. Full training and inference code can be found in this https URL.
>
---
#### [new 009] Learn to Rank: Visual Attribution by Learning Importance Ranking
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉解释任务，解决复杂模型决策难以解释的问题。提出一种直接优化删除和插入指标的学习方法，生成更精确的像素级 attribution 图。**

- **链接: [https://arxiv.org/pdf/2604.05819](https://arxiv.org/pdf/2604.05819)**

> **作者:** David Schinagl; Christian Fruhwirth-Reisinger; Alexander Prutsch; Samuel Schulter; Horst Possegger
>
> **摘要:** Interpreting the decisions of complex computer vision models is crucial to establish trust and accountability, especially in safety-critical domains. An established approach to interpretability is generating visual attribution maps that highlight regions of the input most relevant to the model's prediction. However, existing methods face a three-way trade-off. Propagation-based approaches are efficient, but they can be biased and architecture-specific. Meanwhile, perturbation-based methods are causally grounded, yet they are expensive and for vision transformers often yield coarse, patch-level explanations. Learning-based explainers are fast but usually optimize surrogate objectives or distill from heuristic teachers. We propose a learning scheme that instead optimizes deletion and insertion metrics directly. Since these metrics depend on non-differentiable sorting and ranking, we frame them as permutation learning and replace the hard sorting with a differentiable relaxation using Gumbel-Sinkhorn. This enables end-to-end training through attribution-guided perturbations of the target model. During inference, our method produces dense, pixel-level attributions in a single forward pass with optional, few-step gradient refinement. Our experiments demonstrate consistent quantitative improvements and sharper, boundary-aligned explanations, particularly for transformer-based vision models.
>
---
#### [new 010] SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SnapFlow，解决流匹配视觉-语言-动作模型的推理延迟问题，通过单步生成替代多步去噪，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.05656](https://arxiv.org/pdf/2604.05656)**

> **作者:** Wuyang Luan; Junhui Li; Weiguang Zhao; Wenjian Zhang; Tieru Wu; Rui Ma
>
> **备注:** 10 pages, 6 figures, 9 tables
>
> **摘要:** Vision-Language-Action (VLA) models based on flow matching -- such as pi0, pi0.5, and SmolVLA -- achieve state-of-the-art generalist robotic manipulation, yet their iterative denoising, typically 10 ODE steps, introduces substantial latency: on a modern GPU, denoising alone accounts for 80% of end-to-end inference time. Naively reducing the step count is unreliable, degrading success on most tasks due to the velocity field being uncalibrated for single-step jumps. We present SnapFlow, a plug-and-play self-distillation method that compresses multi-step denoising into a single forward pass (1-NFE) for flow-matching VLAs. SnapFlow mixes standard flow-matching samples with consistency samples whose targets are two-step Euler shortcut velocities computed from the model's own marginal velocity predictions, avoiding the trajectory drift caused by conditional velocities, as we analyze theoretically. A zero-initialized target-time embedding lets the network switch between local velocity estimation and global one-step generation within a single architecture. SnapFlow requires no external teacher, no architecture changes, and trains in ~12h on a single GPU. We validate on two VLA architectures spanning a 6x parameter range, with identical hyperparameters: on pi0.5 (3B) across four LIBERO suites (40 tasks, 400 episodes), SnapFlow achieves 98.75% average success -- matching the 10-step teacher at 97.75% and slightly exceeding it -- with 9.6x denoising speedup and end-to-end latency reduced from 274ms to 83ms; on SmolVLA (500M), it reduces MSE by 8.3% with 3.56x end-to-end acceleration. An action-step sweep on long-horizon tasks reveals that SnapFlow maintains its advantage across execution horizons, achieving 93% at n_act=5 where the baseline reaches only 90%. SnapFlow is orthogonal to layer-distillation and token-pruning approaches, enabling compositional speedups.
>
---
#### [new 011] In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决单目深度估计在Gaussian Splatting中的应用问题。通过引入可靠的几何监督框架，提升深度先验的准确性与渲染质量。**

- **链接: [https://arxiv.org/pdf/2604.05715](https://arxiv.org/pdf/2604.05715)**

> **作者:** Wenhui Xiao; Ethan Goan; Rodrigo Santa Cruz; David Ahmedt-Aristizabal; Olivier Salvado; Clinton Fookes; Leo Lebrat
>
> **备注:** accepted to CVPR 3DMV Workshop
>
> **摘要:** Using accurate depth priors in 3D Gaussian Splatting helps mitigate artifacts caused by sparse training data and textureless surfaces. However, acquiring accurate depth maps requires specialized acquisition systems. Foundation monocular depth estimation models offer a cost-effective alternative, but they suffer from scale ambiguity, multi-view inconsistency, and local geometric inaccuracies, which can degrade rendering performance when applied naively. This paper addresses the challenge of reliably leveraging monocular depth priors for Gaussian Splatting (GS) rendering enhancement. To this end, we introduce a training framework integrating scale-ambiguous and noisy depth priors into geometric supervision. We highlight the importance of learning from weakly aligned depth variations. We introduce a method to isolate ill-posed geometry for selective monocular depth regularization, restricting the propagation of depth inaccuracies into well-reconstructed 3D structures. Extensive experiments across diverse datasets show consistent improvements in geometric accuracy, leading to more faithful depth estimation and higher rendering quality across different GS variants and monocular depth backbones tested.
>
---
#### [new 012] Unifying VLM-Guided Flow Matching and Spectral Anomaly Detection for Interpretable Veterinary Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于犬类气胸的可解释诊断任务，解决数据稀缺和模型可信度问题。提出结合视觉语言模型引导的流匹配与随机矩阵理论的诊断方法，提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.05482](https://arxiv.org/pdf/2604.05482)**

> **作者:** Pu Wang; Zhixuan Mao; Jialu Li; Zhuoran Zheng; Dianjie Lu; Youshan Zhang
>
> **摘要:** Automatic diagnosis of canine pneumothorax is challenged by data scarcity and the need for trustworthy models. To address this, we first introduce a public, pixel-level annotated dataset to facilitate research. We then propose a novel diagnostic paradigm that reframes the task as a synergistic process of signal localization and spectral detection. For localization, our method employs a Vision-Language Model (VLM) to guide an iterative Flow Matching process, which progressively refines segmentation masks to achieve superior boundary accuracy. For detection, the segmented mask is used to isolate features from the suspected lesion. We then apply Random Matrix Theory (RMT), a departure from traditional classifiers, to analyze these features. This approach models healthy tissue as predictable random noise and identifies pneumothorax by detecting statistically significant outlier eigenvalues that represent a non-random pathological signal. The high-fidelity localization from Flow Matching is crucial for purifying the signal, thus maximizing the sensitivity of our RMT detector. This synergy of generative segmentation and first-principles statistical analysis yields a highly accurate and interpretable diagnostic system (source code is available at: this https URL).
>
---
#### [new 013] Leveraging Image Editing Foundation Models for Data-Efficient CT Metal Artifact Reduction
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于医学图像重建任务，解决CT金属伪影问题。通过适配视觉语言扩散模型，实现数据高效的伪影去除。**

- **链接: [https://arxiv.org/pdf/2604.05934](https://arxiv.org/pdf/2604.05934)**

> **作者:** Ahmet Rasim Emirdagi; Süleyman Aslan; Mısra Yavuz; Görkay Aydemir; Yunus Bilge Kurt; Nasrin Rahimi; Burak Can Biner; M. Akın Yılmaz
>
> **备注:** Accepted to CVPRW 2026 Med-Reasoner
>
> **摘要:** Metal artifacts from high-attenuation implants severely degrade CT image quality, obscuring critical anatomical structures and posing a challenge for standard deep learning methods that require extensive paired training data. We propose a paradigm shift: reframing artifact reduction as an in-context reasoning task by adapting a general-purpose vision-language diffusion foundation model via parameter-efficient Low-Rank Adaptation (LoRA). By leveraging rich visual priors, our approach achieves effective artifact suppression with only 16 to 128 paired training examples reducing data requirements by two orders of magnitude. Crucially, we demonstrate that domain adaptation is essential for hallucination mitigation; without it, foundation models interpret streak artifacts as erroneous natural objects (e.g., waffles or petri dishes). To ground the restoration, we propose a multi-reference conditioning strategy where clean anatomical exemplars from unrelated subjects are provided alongside the corrupted input, enabling the model to exploit category-specific context to infer uncorrupted anatomy. Extensive evaluation on the AAPM CT-MAR benchmark demonstrates that our method achieves state-of-the-art performance on perceptual and radiological-feature metrics . This work establishes that foundation models, when appropriately adapted, offer a scalable alternative for interpretable, data-efficient medical image reconstruction. Code is available at this https URL.
>
---
#### [new 014] SEM-ROVER: Semantic Voxel-Guided Diffusion for Large-Scale Driving Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决大规模户外驾驶场景的几何一致性与渲染问题。提出基于语义体素的扩散模型，实现多视角一致的高质量场景生成。**

- **链接: [https://arxiv.org/pdf/2604.06113](https://arxiv.org/pdf/2604.06113)**

> **作者:** Hiba Dahmani; Nathan Piasco; Moussab Bennehar; Luis Roldão; Dzmitry Tsishkou; Laurent Caraffa; Jean-Philippe Tarel; Roland Brémond
>
> **摘要:** Scalable generation of outdoor driving scenes requires 3D representations that remain consistent across multiple viewpoints and scale to large areas. Existing solutions either rely on image or video generative models distilled to 3D space, harming the geometric coherence and restricting the rendering to training views, or are limited to small-scale 3D scene or object-centric generation. In this work, we propose a 3D generative framework based on $\Sigma$-Voxfield grid, a discrete representation where each occupied voxel stores a fixed number of colorized surface samples. To generate this representation, we train a semantic-conditioned diffusion model that operates on local voxel neighborhoods and uses 3D positional encodings to capture spatial structure. We scale to large scenes via progressive spatial outpainting over overlapping regions. Finally, we render the generated $\Sigma$-Voxfield grid with a deferred rendering module to obtain photorealistic images, enabling large-scale multiview-consistent 3D scene generation without per-scene optimization. Extensive experiments show that our approach can generate diverse large-scale urban outdoor scenes, renderable into photorealistic images with various sensor configurations and camera trajectories while maintaining moderate computation cost compared to existing approaches.
>
---
#### [new 015] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态检索增强生成任务，解决标准重排序器受视觉干扰影响的问题。提出Region-R1框架，通过区域选择提升重排序效果。**

- **链接: [https://arxiv.org/pdf/2604.05268](https://arxiv.org/pdf/2604.05268)**

> **作者:** Chan-Wei Hu; Zhengzhong Tu
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Multi-modal retrieval-augmented generation (MM-RAG) relies heavily on re-rankers to surface the most relevant evidence for image-question queries. However, standard re-rankers typically process the full query image as a global embedding, making them susceptible to visual distractors (e.g., background clutter) that skew similarity scores. We propose Region-R1, a query-side region cropping framework that formulates region selection as a decision-making problem during re-ranking, allowing the system to learn to retain the full image or focus only on a question-relevant region before scoring the retrieved candidates. Region-R1 learns a policy with a novel region-aware group relative policy optimization (r-GRPO) to dynamically crop a discriminative region. Across two challenging benchmarks, E-VQA and InfoSeek, Region-R1 delivers consistent gains, achieving state-of-the-art performances by increasing conditional Recall@1 by up to 20%. These results show the great promise of query-side adaptation as a simple but effective way to strengthen MM-RAG re-ranking.
>
---
#### [new 016] HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型中的对象幻觉检测与缓解任务，旨在解决模型生成描述中出现的不真实对象问题。工作包括提出HaloProbe框架，通过贝叶斯方法估计幻觉概率并实现非侵入式缓解。**

- **链接: [https://arxiv.org/pdf/2604.06165](https://arxiv.org/pdf/2604.06165)**

> **作者:** Reihaneh Zohrabi; Hosein Hasani; Akshita Gupta; Mahdieh Soleymani Baghshah; Anna Rohrbach; Marcus Rohrbach
>
> **摘要:** Large vision-language models can produce object hallucinations in image descriptions, highlighting the need for effective detection and mitigation strategies. Prior work commonly relies on the model's attention weights on visual tokens as a detection signal. We reveal that coarse-grained attention-based analysis is unreliable due to hidden confounders, specifically token position and object repetition in a description. This leads to Simpson's paradox: the attention trends reverse or disappear when statistics are aggregated. Based on this observation, we introduce HaloProbe, a Bayesian framework that factorizes external description statistics and internal decoding signals to estimate token-level hallucination probabilities. HaloProbe uses balanced training to isolate internal evidence and combines it with learned prior over external features to recover the true posterior. While intervention-based mitigation methods often degrade utility or fluency by modifying models' internals, we use HaloProbe as an external scoring signal for non-invasive mitigation. Our experiments show that HaloProbe-guided decoding reduces hallucinations more effectively than state-of-the-art intervention-based methods while preserving utility.
>
---
#### [new 017] Human Interaction-Aware 3D Reconstruction from a Single Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单图像3D重建任务，解决多人体交互场景下的几何失真和遮挡问题。提出HUG3D框架，结合群体与个体信息，生成高保真、物理合理的3D模型。**

- **链接: [https://arxiv.org/pdf/2604.05436](https://arxiv.org/pdf/2604.05436)**

> **作者:** Gwanghyun Kim; Junghun James Kim; Suh Yoon Jeon; Jason Park; Se Young Chun
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Reconstructing textured 3D human models from a single image is fundamental for AR/VR and digital human applications. However, existing methods mostly focus on single individuals and thus fail in multi-human scenes, where naive composition of individual reconstructions often leads to artifacts such as unrealistic overlaps, missing geometry in occluded regions, and distorted interactions. These limitations highlight the need for approaches that incorporate group-level context and interaction priors. We introduce a holistic method that explicitly models both group- and instance-level information. To mitigate perspective-induced geometric distortions, we first transform the input into a canonical orthographic space. Our primary component, Human Group-Instance Multi-View Diffusion (HUG-MVD), then generates complete multi-view normals and images by jointly modeling individuals and group context to resolve occlusions and proximity. Subsequently, the Human Group-Instance Geometric Reconstruction (HUG-GR) module optimizes the geometry by leveraging explicit, physics-based interaction priors to enforce physical plausibility and accurately model inter-human contact. Finally, the multi-view images are fused into a high-fidelity texture. Together, these components form our complete framework, HUG3D. Extensive experiments show that HUG3D significantly outperforms both single-human and existing multi-human methods, producing physically plausible, high-fidelity 3D reconstructions of interacting people from a single image. Project page: this https URL
>
---
#### [new 018] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CRFT，用于跨模态图像配准任务，解决不同模态图像间准确对齐问题。通过特征流学习和注意力机制，提升配准精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05689](https://arxiv.org/pdf/2604.05689)**

> **作者:** Xuecong Liu; Mengzhu Ding; Zixuan Sun; Zhang Li; Xichao Teng
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present Consistent-Recurrent Feature Flow Transformer (CRFT), a unified coarse-to-fine framework based on feature flow learning for robust cross-modal image registration. CRFT learns a modality-independent feature flow representation within a transformer-based architecture that jointly performs feature alignment and flow estimation. The coarse stage establishes global correspondences through multi-scale feature correlation, while the fine stage refines local details via hierarchical feature fusion and adaptive spatial reasoning. To enhance geometric adaptability, an iterative discrepancy-guided attention mechanism with a Spatial Geometric Transform (SGT) recurrently refines the flow field, progressively capturing subtle spatial inconsistencies and enforcing feature-level consistency. This design enables accurate alignment under large affine and scale variations while maintaining structural coherence across modalities. Extensive experiments on diverse cross-modal datasets demonstrate that CRFT consistently outperforms state-of-the-art registration methods in both accuracy and robustness. Beyond registration, CRFT provides a generalizable paradigm for multimodal spatial correspondence, offering broad applicability to remote sensing, autonomous navigation, and medical imaging. Code and datasets are publicly available at this https URL.
>
---
#### [new 019] A Synthetic Eye Movement Dataset for Script Reading Detection: Real Trajectory Replay on a 3D Simulator
- **分类: cs.CV**

- **简介: 该论文属于行为建模任务，旨在解决眼动数据稀缺问题。通过合成方法生成眼动视频数据，用于脚本阅读检测，提升行为分类器性能。**

- **链接: [https://arxiv.org/pdf/2604.05475](https://arxiv.org/pdf/2604.05475)**

> **作者:** Kidus Zewde; Yuchen Zhou; Dennis Ng; Neo Tiangratanakul; Tommy Duong; Ankit Raj; Yuxin Zhang; Xingyu Shen; Simiao Ren
>
> **备注:** Synthetic eye movement dataset generation via 3D eye simulator; iris trajectory replay; script reading detection; behavioral data augmentation
>
> **摘要:** Large vision-language models have achieved remarkable capabilities by training on massive internet-scale data, yet a fundamental asymmetry persists: while LLMs can leverage self-supervised pretraining on abundant text and image data, the same is not true for many behavioral modalities. Video-based behavioral data -- gestures, eye movements, social signals -- remains scarce, expensive to annotate, and privacy-sensitive. A promising alternative is simulation: replace real data collection with controlled synthetic generation to produce automatically labeled data at scale. We introduce infrastructure for this paradigm applied to eye movement, a behavioral signal with applications across vision-language modeling, virtual reality, robotics, accessibility systems, and cognitive science. We present a pipeline for generating synthetic labeled eye movement video by extracting real human iris trajectories from reference videos and replaying them on a 3D eye movement simulator via headless browser automation. Applying this to the task of script-reading detection during video interviews, we release final_dataset_v1: 144 sessions (72 reading, 72 conversation) totaling 12 hours of synthetic eye movement video at 25fps. Evaluation shows that generated trajectories preserve the temporal dynamics of the source data (KS D < 0.14 across all metrics). A matched frame-by-frame comparison reveals that the 3D simulator exhibits bounded sensitivity at reading-scale movements, attributable to the absence of coupled head movement -- a finding that informs future simulator design. The pipeline, dataset, and evaluation tools are released to support downstream behavioral classifier development at the intersection of behavioral modeling and vision-language systems.
>
---
#### [new 020] Towards Athlete Fatigue Assessment from Association Football Videos
- **分类: cs.CV**

- **简介: 该论文属于运动员疲劳评估任务，旨在利用足球比赛视频分析疲劳指标。通过处理视频中的球员轨迹，提取速度与加速度数据，验证其作为疲劳分析依据的可行性。**

- **链接: [https://arxiv.org/pdf/2604.05636](https://arxiv.org/pdf/2604.05636)**

> **作者:** Xavier Bou; Nathan Correger; Alexandre Cloots; Cédric Gavage; Silvio Giancola; Cédric Schwartz; François Delvaux; Rudi Cloots; Marc Van Droogenbroeck; Anthony Cioppa
>
> **摘要:** Fatigue monitoring is central in association football due to its links with injury risk and tactical performance. However, objective fatigue-related indicators are commonly derived from subjective self-reported metrics, biomarkers derived from laboratory tests, or, more recently, intrusive sensors such as heart monitors or GPS tracking data. This paper studies whether monocular broadcast videos can provide spatio-temporal signals of sufficient quality to support fatigue-oriented analysis. Building on state-of-the-art Game State Reconstruction methods, we extract player trajectories in pitch coordinates and propose a novel kinematics processing algorithm to obtain temporally consistent speed and acceleration estimates from reconstructed tracks. We then construct acceleration--speed (A-S) profiles from these signals and analyze their behavior as fatigue-related performance indicators. We evaluate the full pipeline on the public SoccerNet-GSR benchmark, considering both 30-second clips and a complete 45-minute half to examine short-term reliability and longer-term temporal consistency. Our results indicate that monocular GSR can recover kinematic patterns that are compatible with A-S profiling while also revealing sensitivity to trajectory noise, calibration errors, and temporal discontinuities inherent to broadcast footage. These findings support monocular broadcast video as a low-cost basis for fatigue analysis and delineate the methodological challenges for future research.
>
---
#### [new 021] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher
- **分类: cs.CV**

- **简介: 该论文属于多模态人类感知任务，解决模态缺失下的鲁棒性问题。提出PTA框架，通过净化知识源和对齐模态，提升单模态模型的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05584](https://arxiv.org/pdf/2604.05584)**

> **作者:** Pengcheng Weng; Yanyu Qian; Yangxin Xu; Fei Wang
>
> **备注:** Accepted by CVPR 2026 Workshop On Any-to-Any Multimodal Learning
>
> **摘要:** Robust multimodal human sensing must overcome the critical challenge of missing modalities. Two principal barriers are the Representation Gap between heterogeneous data and the Contamination Effect from low-quality modalities. These barriers are causally linked, as the corruption introduced by contamination fundamentally impedes the reduction of representation disparities. In this paper, we propose PTA, a novel "Purify-then-Align" framework that solves this causal dependency through a synergistic integration of meta-learning and knowledge diffusion. To purify the knowledge source, PTA first employs a meta-learning-driven weighting mechanism that dynamically learns to down-weight the influence of noisy, low-contributing modalities. Subsequently, to align different modalities, PTA introduces a diffusion-based knowledge distillation paradigm in which an information-rich clean teacher, formed from this purified consensus, refines the features of each student modality. The ultimate payoff of this "Purify-then-Align" strategy is the creation of exceptionally powerful single-modality encoders imbued with cross-modal knowledge. Comprehensive experiments on the large-scale MM-Fi and XRF55 datasets, under pronounced Representation Gap and Contamination Effect, demonstrate that PTA achieves state-of-the-art performance and significantly improves the robustness of single-modality models in diverse missing-modality scenarios.
>
---
#### [new 022] Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决现有基准与实际模型能力之间的差距。提出Video-MME-v2基准，通过多层次评估和非线性评价策略，提升模型的鲁棒性和推理能力。**

- **链接: [https://arxiv.org/pdf/2604.05015](https://arxiv.org/pdf/2604.05015)**

> **作者:** Chaoyou Fu; Haozhi Yuan; Yuhao Dong; Yi-Fan Zhang; Yunhang Shen; Xiaoxing Hu; Xueying Li; Jinsen Su; Chengwu Long; Xiaoyao Xie; Yongkang Xie; Xiawu Zheng; Xue Yang; Haoyu Cao; Yunsheng Wu; Ziwei Liu; Xing Sun; Caifeng Shan; Ran He
>
> **备注:** Homepage: this https URL
>
> **摘要:** With the rapid advancement of video understanding, existing benchmarks are becoming increasingly saturated, exposing a critical discrepancy between inflated leaderboard scores and real-world model capabilities. To address this widening gap, we introduce Video-MME-v2, a comprehensive benchmark designed to rigorously evaluate the robustness and faithfulness of video understanding. To systematically evaluate model capabilities, we design a \textbf{progressive tri-level hierarchy} that incrementally increases the complexity of video comprehension, ranging from multi-point visual information aggregation, to temporal dynamics modeling, and ultimately to complex multimodal reasoning. Besides, in contrast to conventional per-question accuracy, we propose a \textbf{group-based non-linear evaluation} strategy that enforces both consistency across related queries and coherence in multi-step reasoning. It penalizes fragmented or guess-based correctness and assigns credit only to answers supported by valid reasoning. To guarantee data quality, Video-MME-v2 is constructed through a rigorously controlled human annotation pipeline, involving 12 annotators and 50 independent reviewers. Backed by \textbf{3,300 human-hours} and up to \textbf{5 rounds} of quality assurance, Video-MME-v2 aims to serve as one of the most authoritative video benchmarks. Extensive experiments reveal a substantial gap between current best model Gemini-3-Pro and human experts, and uncover a clear hierarchical bottleneck where errors in visual information aggregation and temporal modeling propagate to limit high-level reasoning. We further find that thinking-based reasoning is highly dependent on textual cues, improving performance with subtitles but sometimes degrading it in purely visual settings. By exposing these limitations, Video-MME-v2 establishes a demanding new testbed for the development of next-generation video MLLMs.
>
---
#### [new 023] Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型安全任务，旨在解决AR中矛盾虚拟内容攻击的检测问题。提出ContrAR基准，评估VLMs在AR环境下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05510](https://arxiv.org/pdf/2604.05510)**

> **作者:** Yanming Xiu; Zhengayuan Jiang; Neil Zhenqiang Gong; Maria Gorlatova
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Augmented reality (AR) has rapidly expanded over the past decade. As AR becomes increasingly integrated into daily life, its security and reliability emerge as critical challenges. Among various threats, contradictory virtual content attacks, where malicious or inconsistent virtual elements are introduced into the user's view, pose a unique risk by misleading users, creating semantic confusion, or delivering harmful information. In this work, we systematically model such attacks and present ContrAR, a novel benchmark for evaluating the robustness of vision-language models (VLMs) against virtual content manipulation and contradiction in AR. ContrAR contains 312 real-world AR videos validated by 10 human participants. We further benchmark 11 VLMs, including both commercial and open-source models. Experimental results reveal that while current VLMs exhibit reasonable understanding of contradictory virtual content, room still remains for improvement in detecting and reasoning about adversarial content manipulations in AR environments. Moreover, balancing detection accuracy and latency remains challenging.
>
---
#### [new 024] EfficientMonoHair: Fast Strand-Level Reconstruction from Monocular Video via Multi-View Direction Fusion
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于虚拟人建模任务，旨在解决单目视频中头发细丝级几何重建的精度与效率矛盾。提出EfficientMonoHair方法，结合隐式神经网络与多视角几何融合，实现快速高保真重建。**

- **链接: [https://arxiv.org/pdf/2604.05794](https://arxiv.org/pdf/2604.05794)**

> **作者:** Da Li; Dominik Engel; Deng Luo; Ivan Viola
>
> **备注:** 10 pages, 6 figures, conference
>
> **摘要:** Strand-level hair geometry reconstruction is a fundamental problem in virtual human modeling and the digitization of hairstyles. However, existing methods still suffer from a significant trade-off between accuracy and efficiency. Implicit neural representations can capture the global hair shape but often fail to preserve fine-grained strand details, while explicit optimization-based approaches achieve high-fidelity reconstructions at the cost of heavy computation and poor scalability. To address this issue, we propose EfficientMonoHair, a fast and accurate framework that combines the implicit neural network with multi-view geometric fusion for strand-level reconstruction from monocular video. Our method introduces a fusion-patch-based multi-view optimization that reduces the number of optimization iterations for point cloud direction, as well as a novel parallel hair-growing strategy that relaxes voxel occupancy constraints, allowing large-scale strand tracing to remain stable and robust even under inaccurate or noisy orientation fields. Extensive experiments on representative real-world hairstyles demonstrate that our method can robustly reconstruct high-fidelity strand geometries with accuracy. On synthetic benchmarks, our method achieves reconstruction quality comparable to state-of-the-art methods, while improving runtime efficiency by nearly an order of magnitude.
>
---
#### [new 025] Analogical Reasoning as a Doctor: A Foundation Model for Gastrointestinal Endoscopy Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于胃肠内镜诊断任务，旨在解决AI模型在泛化性、适应性和鲁棒性方面的不足。提出RATNet模型，通过类比推理提升诊断效果。**

- **链接: [https://arxiv.org/pdf/2604.05649](https://arxiv.org/pdf/2604.05649)**

> **作者:** Peixi Peng; Housheng Xie; Yanling Wei; Guangcong Ruan; Xiaoyang Zou; Qian Cao; Yongjian Nian; Guoyan Zheng
>
> **摘要:** Gastrointestinal diseases impose a growing global health burden, and endoscopy is a primary tool for early diagnosis. However, routine endoscopic image interpretation still suffers from missed lesions and limited efficiency. Although AI-assisted diagnosis has shown promise, existing models often lack generalizability, adaptability, robustness, and scalability because of limited medical data, domain shift, and heterogeneous annotations. To address these challenges, we develop RATNet, a foundation model for gastrointestinal endoscopy imaging based on analogical reasoning. RATNet acquires and transfers knowledge from heterogeneous expert annotations across five gastrointestinal endoscopy datasets through a cyclic pre-training strategy. Its architecture consists of an encoder, a relevance-knowledge acquisition and transfer (RAT) module, a projector, and a multi-task head, and supports fine-tuning, linear probing, and zero-shot transfer. Evaluations show that RATNet outperforms existing foundation models, including GastroNet and GastroVision, across six scenarios: diagnosis of common gastrointestinal diseases, few-shot learning for rare diseases, zero-shot transfer to new medical sites, robustness under long-tailed disease distributions, adaptation to novel diseases, and privacy-preserving deployment via federated learning. Its advantage comes from an analogical reasoning mechanism that matches image-derived posterior knowledge to a learned prior knowledge base and transfers relative knowledge to guide diagnosis, improving generalization and resistance to bias. RATNet is open and cost-effective, supports automatic integration of heterogeneous annotations without manual label unification, and reduces data acquisition costs, making it a practical foundation for intelligent gastrointestinal diagnosis, especially in resource-limited settings.
>
---
#### [new 026] Automatic dental superimposition of 3D intraorals and 2D photographs for human identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于法医人类识别任务，旨在解决无生前记录时的牙齿比对问题。通过3D与2D图像自动叠加，实现客观、定量的形态比对。**

- **链接: [https://arxiv.org/pdf/2604.05877](https://arxiv.org/pdf/2604.05877)**

> **作者:** Antonio D. Villegas-Yeguas; Xavier Abreau-Freire; Guillermo R-García; Andrea Valsecchi; Teresa Pinho; Daniel Pérez-Mongiovi; Oscar Ibáñez; Oscar Cordón
>
> **备注:** 10 pages, 9 figures, 3 tables
>
> **摘要:** Dental comparison is considered a primary identification method, at the level of fingerprints and DNA profiling. One crucial but time-consuming step of this method is the morphological comparison. One of the main challenges to apply this method is the lack of ante-mortem medical records, specially on scenarios such as migrant death at the border and/or in countries where there is no universal healthcare. The availability of photos on social media where teeth are visible has led many odontologists to consider morphological comparison using them. However, state-of-the-art proposals have significant limitations, including the lack of proper modeling of perspective distortion and the absence of objective approaches that quantify morphological differences. Our proposal involves a 3D (post-mortem scan) - 2D (ante-mortem photos) approach. Using computer vision and optimization techniques, we replicate the ante-mortem image with the 3D model to perform the morphological comparison. Two automatic approaches have been developed: i) using paired landmarks and ii) using a segmentation of the teeth region to estimate camera parameters. Both are capable of obtaining very promising results over 20,164 cross comparisons from 142 samples, obtaining mean ranking values of 1.6 and 1.5, respectively. These results clearly outperform filtering capabilities of automatic dental chart comparison approaches, while providing an automatic, objective and quantitative score of the morphological correspondence, easily to interpret and analyze by visualizing superimposed images.
>
---
#### [new 027] Selective Aggregation of Attention Maps Improves Diffusion-Based Visual Interpretation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在提升模型的可解释性。通过选择性聚合相关注意力头，增强视觉解释能力，解决注意力地图信息冗余问题。**

- **链接: [https://arxiv.org/pdf/2604.05906](https://arxiv.org/pdf/2604.05906)**

> **作者:** Jungwon Park; Jungmin Ko; Dongnam Byun; Wonjong Rhee
>
> **摘要:** Numerous studies on text-to-image (T2I) generative models have utilized cross-attention maps to boost application performance and interpret model behavior. However, the distinct characteristics of attention maps from different attention heads remain relatively underexplored. In this study, we show that selectively aggregating cross-attention maps from heads most relevant to a target concept can improve visual interpretability. Compared to the diffusion-based segmentation method DAAM, our approach achieves higher mean IoU scores. We also find that the most relevant heads capture concept-specific features more accurately than the least relevant ones, and that selective aggregation helps diagnose prompt misinterpretations. These findings suggest that attention head selection offers a promising direction for improving the interpretability and controllability of T2I generation.
>
---
#### [new 028] HumANDiff: Articulated Noise Diffusion for Motion-Consistent Human Video Generation
- **分类: cs.CV**

- **简介: 该论文属于人体视频生成任务，解决运动一致性与物理真实性问题。提出HumANDiff框架，通过关节运动采样、外观-运动联合学习和几何运动一致性损失，提升生成视频质量。**

- **链接: [https://arxiv.org/pdf/2604.05961](https://arxiv.org/pdf/2604.05961)**

> **作者:** Tao Hu; Varun Jampani
>
> **备注:** Project page: this https URL
>
> **摘要:** Despite tremendous recent progress in human video generation, generative video diffusion models still struggle to capture the dynamics and physics of human motions faithfully. In this paper, we propose a new framework for human video generation, HumANDiff, which enhances the human motion control with three key designs: 1) Articulated motion-consistent noise sampling that correlates the spatiotemporal distribution of latent noise and replaces the unstructured random Gaussian noise with 3D articulated noise sampled on the dense surface manifold of a statistical human body template. It inherits body topology priors for spatially and temporally consistent noise sampling. 2) Joint appearance-motion learning that enhances the standard training objective of video diffusion models by jointly predicting pixel appearances and corresponding physical motions from the articulated noises. It enables high-fidelity human video synthesis, e.g., capturing motion-dependent clothing wrinkles. 3) Geometric motion consistency learning that enforces physical motion consistency across frames via a novel geometric motion consistency loss defined in the articulated noise space. HumANDiff enables scalable controllable human video generation by fine-tuning video diffusion models with articulated noise sampling. Consequently, our method is agnostic to diffusion model design, and requires no modifications to the model architecture. During inference, HumANDiff enables image-to-video generation within a single framework, achieving intrinsic motion control without requiring additional motion modules. Extensive experiments demonstrate that our method achieves state-of-the-art performance in rendering motion-consistent, high-fidelity humans with diverse clothing styles. Project page: this https URL
>
---
#### [new 029] Probing Intrinsic Medical Task Relationships: A Contrastive Learning Perspective
- **分类: cs.CV**

- **简介: 该论文研究医学视觉任务间的内在关系，旨在通过对比学习构建共享表示空间。任务属于医学图像分析，解决任务间相似性与差异性不明确的问题。工作包括引入TaCo框架，映射不同模态任务并分析其特性。**

- **链接: [https://arxiv.org/pdf/2604.05651](https://arxiv.org/pdf/2604.05651)**

> **作者:** Jonas Muth; Zdravko Marinov; Simon Reiß
>
> **摘要:** While much of the medical computer vision community has focused on advancing performance for specific tasks, the underlying relationships between tasks, i.e., how they relate, overlap, or differ on a representational level, remain largely unexplored. Our work explores these intrinsic relationships between medical vision tasks, specifically, we investigate 30 tasks, such as semantic tasks (e.g., segmentation and detection), image generative tasks (e.g., denoising, inpainting, or colorization), and image transformation tasks (e.g., geometric transformations). Our goal is to probe whether a data-driven representation space can capture an underlying structure of tasks across a variety of 39 datasets from wildly different medical imaging modalities, including computed tomography, magnetic resonance, electron microscopy, X-ray ultrasound and more. By revealing how tasks relate to one another, we aim to provide insights into their fundamental properties and interconnectedness. To this end, we introduce Task-Contrastive Learning (TaCo), a contrastive learning framework designed to embed tasks into a shared representation space. Through TaCo, we map these heterogeneous tasks from different modalities into a joint space and analyze their properties: identifying which tasks are distinctly represented, which blend together, and how iterative alterations to tasks are reflected in the embedding space. Our work provides a foundation for understanding the intrinsic structure of medical vision tasks, offering a deeper understanding of task similarities and their interconnected properties in embedding spaces.
>
---
#### [new 030] MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决多实例精细编辑一致性问题。提出MIRAGE框架，实现精准局部编辑并保持背景一致。**

- **链接: [https://arxiv.org/pdf/2604.05180](https://arxiv.org/pdf/2604.05180)**

> **作者:** Ziqian Liu; Stephan Alaniz
>
> **摘要:** Instruction-guided image editing has seen remarkable progress with models like FLUX.2 and Qwen-Image-Edit, yet they still struggle with complex scenarios with multiple similar instances each requiring individual edits. We observe that state-of-the-art models suffer from severe over-editing and spatial misalignment when faced with multiple identical instances and composite instructions. To this end, we introduce a comprehensive benchmark specifically designed to evaluate fine-grained consistency in multi-instance and multi-instruction settings. To address the failures of existing methods observed in our benchmark, we propose Multi-Instance Regional Alignment via Guided Editing (MIRAGE), a training-free framework that enables precise, localized editing. By leveraging a vision-language model to parse complex instructions into regional subsets, MIRAGE employs a multi-branch parallel denoising strategy. This approach injects latent representations of target regions into the global representation space while maintaining background integrity through a reference trajectory. Extensive evaluations on MIRA-Bench and RefEdit-Bench demonstrate that our framework significantly outperforms existing methods in achieving precise instance-level modifications while preserving background consistency. Our benchmark and code are available at this https URL.
>
---
#### [new 031] Integration of Object Detection and Small VLMs for Construction Safety Hazard Identification
- **分类: cs.CV**

- **简介: 该论文属于施工安全危害识别任务，旨在解决大模型计算成本高与小模型准确率低的矛盾。通过融合目标检测与小视觉语言模型，提升危害识别效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.05210](https://arxiv.org/pdf/2604.05210)**

> **作者:** Muhammad Adil; Mehmood Ahmed; Muhammad Aqib; Vicente A. Gonzalez; Gaang Lee; Qipei Mei
>
> **摘要:** Accurate and timely identification of construction hazards around workers is essential for preventing workplace accidents. While large vision-language models (VLMs) demonstrate strong contextual reasoning capabilities, their high computational requirements limit their applicability in near real-time construction hazard detection. In contrast, small vision-language models (sVLMs) with fewer than 4 billion parameters offer improved efficiency but often suffer from reduced accuracy and hallucination when analyzing complex construction scenes. To address this trade-off, this study proposes a detection-guided sVLM framework that integrates object detection with multimodal reasoning for contextual hazard identification. The framework first employs a YOLOv11n detector to localize workers and construction machinery within the scene. The detected entities are then embedded into structured prompts to guide the reasoning process of sVLMs, enabling spatially grounded hazard assessment. Within this framework, six sVLMs (Gemma-3 4B, Qwen-3-VL 2B/4B, InternVL-3 1B/2B, and SmolVLM-2B) were evaluated in zero-shot settings on a curated dataset of construction site images with hazard annotations and explanatory rationales. The proposed approach consistently improved hazard detection performance across all models. The best-performing model, Gemma-3 4B, achieved an F1-score of 50.6%, compared to 34.5% in the baseline configuration. Explanation quality also improved significantly, with BERTScore F1 increasing from 0.61 to 0.82. Despite incorporating object detection, the framework introduces minimal overhead, adding only 2.5 ms per image during inference. These results demonstrate that integrating lightweight object detection with small VLM reasoning provides an effective and efficient solution for context-aware construction safety hazard detection.
>
---
#### [new 032] R3PM-Net: Real-time, Robust, Real-world Point Matching Network
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于点云配准任务，旨在解决真实场景下点云匹配的精度与实时性问题。提出R3PM-Net网络及两个新数据集，实现高效准确的点云对齐。**

- **链接: [https://arxiv.org/pdf/2604.05060](https://arxiv.org/pdf/2604.05060)**

> **作者:** Yasaman Kashefbahrami; Erkut Akdag; Panagiotis Meletis; Evgeniya Balmashnova; Dip Goswami; Egor Bondarau
>
> **备注:** Accepted to CVPRw 2026 (Oral), Code and datasets at this https URL
>
> **摘要:** Accurate Point Cloud Registration (PCR) is an important task in 3D data processing, involving the estimation of a rigid transformation between two point clouds. While deep-learning methods have addressed key limitations of traditional non-learning approaches, such as sensitivity to noise, outliers, occlusion, and initialization, they are developed and evaluated on clean, dense, synthetic datasets (limiting their generalizability to real-world industrial scenarios). This paper introduces R3PM-Net, a lightweight, global-aware, object-level point matching network designed to bridge this gap by prioritizing both generalizability and real-time efficiency. To support this transition, two datasets, Sioux-Cranfield and Sioux-Scans, are proposed. They provide an evaluation ground for registering imperfect photogrammetric and event-camera scans to digital CAD models, and have been made publicly available. Extensive experiments demonstrate that R3PM-Net achieves competitive accuracy with unmatched speed. On ModelNet40, it reaches a perfect fitness score of $1$ and inlier RMSE of $0.029$ cm in only $0.007$s, approximately 7 times faster than the state-of-the-art method RegTR. This performance carries over to the Sioux-Cranfield dataset, maintaining a fitness of $1$ and inlier RMSE of $0.030$ cm with similarly low latency. Furthermore, on the highly challenging Sioux-Scans dataset, R3PM-Net successfully resolves edge cases in under 50 ms. These results confirm that R3PM-Net offers a robust, high-speed solution for critical industrial applications, where precision and real-time performance are indispensable. The code and datasets are available at this https URL.
>
---
#### [new 033] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis
- **分类: cs.CV**

- **简介: 该论文属于多模态情感分析任务，旨在解决VLM在情感内容分析中的不足。通过构建AICA-Bench基准，提出GAT提示方法提升情感理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2604.05900](https://arxiv.org/pdf/2604.05900)**

> **作者:** Dong She; Xianrong Yao; Liqun Chen; Jinghe Yu; Yang Gao; Zhanpeng Jin
>
> **备注:** Accepted by Findings of ACL 2026
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated strong capabilities in perception, yet holistic Affective Image Content Analysis (AICA), which integrates perception, reasoning, and generation into a unified framework, remains underexplored. To address this gap, we introduce AICA-Bench, a comprehensive benchmark with three core tasks: Emotion Understanding (EU), Emotion Reasoning (ER), and Emotion-Guided Content Generation (EGCG). We evaluate 23 VLMs and identify two major limitations: weak intensity calibration and shallow open-ended descriptions. To address these issues, we propose Grounded Affective Tree (GAT) Prompting, a training-free framework that combines visual scaffolding with hierarchical reasoning. Experiments show that GAT reduces intensity errors and improves descriptive depth, providing a strong baseline for future research on affective multimodal understanding and generation.
>
---
#### [new 034] Action Images: End-to-End Policy Learning via Multiview Video Generation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决现有方法依赖独立动作模块或非像素对齐表示的问题。提出Action Images，通过多视角视频生成实现统一策略学习。**

- **链接: [https://arxiv.org/pdf/2604.06168](https://arxiv.org/pdf/2604.06168)**

> **作者:** Haoyu Zhen; Zixian Gao; Qiao Sun; Yilin Zhao; Yuncong Yang; Yilun Du; Tsun-Hsuan Wang; Yi-Ling Qiao; Chuang Gan
>
> **备注:** Project Page: this https URL
>
> **摘要:** World action models (WAMs) have emerged as a promising direction for robot policy learning, as they can leverage powerful video backbones to model the future states. However, existing approaches often rely on separate action modules, or use action representations that are not pixel-grounded, making it difficult to fully exploit the pretrained knowledge of video models and limiting transfer across viewpoints and environments. In this work, we present Action Images, a unified world action model that formulates policy learning as multiview video generation. Instead of encoding control as low-dimensional tokens, we translate 7-DoF robot actions into interpretable action images: multi-view action videos that are grounded in 2D pixels and explicitly track robot-arm motion. This pixel-grounded action representation allows the video backbone itself to act as a zero-shot policy, without a separate policy head or action module. Beyond control, the same unified model supports video-action joint generation, action-conditioned video generation, and action labeling under a shared representation. On RLBench and real-world evaluations, our model achieves the strongest zero-shot success rates and improves video-action joint generation quality over prior video-space world models, suggesting that interpretable action images are a promising route to policy learning.
>
---
#### [new 035] ASSR-Net: Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像融合任务，旨在解决空间结构模糊和光谱失真问题。提出ASSR-Net，通过两阶段策略提升空间细节和光谱一致性。**

- **链接: [https://arxiv.org/pdf/2604.05742](https://arxiv.org/pdf/2604.05742)**

> **作者:** Qiya Song; Hongzhi Zhou; Lishan Tan; Renwei Dian; Shutao Li
>
> **摘要:** Hyperspectral image fusion aims to reconstruct high-spatial-resolution hyperspectral images (HR-HSI) by integrating complementary information from multi-source inputs. Despite recent progress, existing methods still face two critical challenges: (1) inadequate reconstruction of anisotropic spatial structures, resulting in blurred details and compromised spatial quality; and (2) spectral distortion during fusion, which hinders fine-grained spectral representation. To address these issues, we propose \textbf{ASSR-Net}: an Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion. ASSR-Net adopts a two-stage fusion strategy comprising anisotropic structure-aware spatial enhancement (ASSE) and hierarchical prior-guided spectral calibration (HPSC). In the first stage, a directional perception fusion module adaptively captures structural features along multiple orientations, effectively reconstructing anisotropic spatial patterns. In the second stage, a spectral recalibration module leverages the original low-resolution HSI as a spectral prior to explicitly correct spectral deviations in the fused results, thereby enhancing spectral fidelity. Extensive experiments on various benchmark datasets demonstrate that ASSR-Net consistently outperforms state-of-the-art methods, achieving superior spatial detail preservation and spectral consistency.
>
---
#### [new 036] SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration
- **分类: cs.CV**

- **简介: 该论文属于多视角烟雾恢复任务，解决真实场景中烟雾导致的几何与外观不一致问题。通过分离几何恢复与外观校正，实现更准确的3D重建。**

- **链接: [https://arxiv.org/pdf/2604.05301](https://arxiv.org/pdf/2604.05301)**

> **作者:** Xueming Fu; Lixia Han
>
> **备注:** Lab Report for NTIRE 2026 3DRR Track 2
>
> **摘要:** Real-world smoke simultaneously attenuates scene radiance, adds airlight, and destabilizes multi-view appearance consistency, making robust 3D reconstruction particularly difficult. We present \textbf{SmokeGS-R}, a practical pipeline developed for the NTIRE 2026 3D Restoration and Reconstruction Track 2 challenge. The key idea is to decouple geometry recovery from appearance correction: we generate physics-guided pseudo-clean supervision with a refined dark channel prior and guided filtering, train a sharp clean-only 3D Gaussian Splatting source model, and then harmonize its renderings with a donor ensemble using geometric-mean reference aggregation, LAB-space Reinhard transfer, and light Gaussian smoothing. On the official challenge testing leaderboard, the final submission achieved \mbox{PSNR $=15.217$} and \mbox{SSIM $=0.666$}. After the public release of RealX3D, we re-evaluated the same frozen result on the seven released challenge scenes without retraining and obtained \mbox{PSNR $=15.209$}, \mbox{SSIM $=0.644$}, and \mbox{LPIPS $=0.551$}, outperforming the strongest official baseline average on the same scenes by $+3.68$ dB PSNR. These results suggest that a geometry-first reconstruction strategy combined with stable post-render appearance harmonization is an effective recipe for real-world multi-view smoke restoration. The code is available at this https URL.
>
---
#### [new 037] OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control
- **分类: cs.CV**

- **简介: 该论文提出OmniCamera，解决视频生成中内容与相机运动耦合的问题，通过解耦两者实现灵活控制。工作包括构建混合数据集和双级课程协同训练策略。**

- **链接: [https://arxiv.org/pdf/2604.06010](https://arxiv.org/pdf/2604.06010)**

> **作者:** Yukun Wang; Ruihuang Li; Jiale Tao; Shiyuan Yang; Liyi Chen; Zhantao Yang; Handz; Yulan Guo; Shuai Shao; Qinglin Lu
>
> **摘要:** Video fundamentally intertwines two crucial axes: the dynamic content of a scene and the camera motion through which it is observed. However, existing generation models often entangle these factors, limiting independent control. In this work, we introduce OmniCamera, a unified framework designed to explicitly disentangle and command these two dimensions. This compositional approach enables flexible video generation by allowing arbitrary pairings of camera and content conditions, unlocking unprecedented creative control. To overcome the fundamental challenges of modality conflict and data scarcity inherent in such a system, we present two key innovations. First, we construct OmniCAM, a novel hybrid dataset combining curated real-world videos with synthetic data that provides diverse paired examples for robust multi-task learning. Second, we propose a Dual-level Curriculum Co-Training strategy that mitigates modality interference and synergistically learns from diverse data sources. This strategy operates on two levels: first, it progressively introduces control modalities by difficulties (condition-level), and second, trains for precise control on synthetic data before adapting to real data for photorealism (data-level). As a result, OmniCamera achieves state-of-the-art performance, enabling flexible control for complex camera movements while maintaining superior visual quality.
>
---
#### [new 038] UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation
- **分类: cs.CV**

- **简介: 该论文提出UAVReason基准，解决UAV场景下的多模态推理与生成问题，整合大量VQA数据和多任务学习方法提升性能。**

- **链接: [https://arxiv.org/pdf/2604.05377](https://arxiv.org/pdf/2604.05377)**

> **作者:** Jintao Sun; Hu Zhang; Donglin Di; Gangyi Ding; Zhedong Zheng
>
> **备注:** 20 pages, 12 figures, 7 tables
>
> **摘要:** Vision-Language models (VLMs) have demonstrated remarkable capability in ground-view visual understanding but often fracture when deployed on high-altitude Unmanned Aerial Vehicles (UAVs). The failure largely stems from a pronounced domain shift, characterized by tiny and densely packed objects, repetitive textures, and ambiguous top-down orientations. These factors severely disrupt semantic grounding and hinder both spatial reasoning and controllable generation. To bridge this critical gap, we introduce UAVReason, the first unified large-scale multi-modal benchmark dedicated to nadir-view UAV scenarios, derived from a high-fidelity UAV simulation platform. In contrast to existing UAV benchmarks, which are largely siloed and focus on single tasks like object detection or segmentation, UAVReason uniquely consolidates over 273K Visual Question Answering (VQA) pairs, including 23.6K single frames with detailed captions, 68.2K 2-frame temporal sequences, and 188.8K cross-modal generation samples. The benchmark probes 22 diverse reasoning types across spatial and temporal axes while simultaneously evaluating high-fidelity generation across RGB, depth, and segmentation modalities. We further establish a strong, unified baseline model via multi-task learning. Extensive experiments validate the efficacy of our unified approach across diverse metrics, such as EM/F1 for VQA, mIoU for segmentation, and CLIP Score for generation. These results indicate limitations of general-domain vision-language models and show that unified multi-task learning substantially improves UAV-native performance. All data, code, and evaluation tools will be publicly released to advance UAV multimodal research.
>
---
#### [new 039] Rethinking IRSTD: Single-Point Supervision Guided Encoder-only Framework is Enough for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决小目标定位与背景干扰问题。提出SPIRE方法，通过单点监督实现高精度定位，提升检测效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.05363](https://arxiv.org/pdf/2604.05363)**

> **作者:** Rixiang Ni; Boyang Li; Jun Chen; Yonghao Li; Feiyu Ren; Yuji Wang; Haoyang Yuan; Wujiao He; Wei An
>
> **摘要:** Infrared small target detection (IRSTD) aims to separate small targets from clutter backgrounds. Extensive research is dedicated to the pixel-level supervision-guided "encoder-decoder" segmentation paradigm. Although having achieved promising performance, they neglect the fact that small targets only occupy a few pixels and are usually accompanied with blurred boundary caused by clutter backgrounds. Based on this observation, we argue that the first principle of IRSTD should be target localization instead of separating all target region accompanied with indistinguishable background noise. In this paper, we reformulate IRSTD as a centroid regression task and propose a novel Single-Point Supervision guided Infrared Probabilistic Response Encoding method (namely, SPIRE), which is indeed challenging due to the mismatch between reduced supervision network and equivalent output. Specifically, we first design a Point-Response Prior Supervision (PRPS), which transforms single-point annotations into probabilistic response map consistent with infrared point-target response characteristics, with a High-Resolution Probabilistic Encoder (HRPE) that enables encoder-only, end-to-end regression without decoder reconstruction. By preserving high-resolution features and increasing effective supervision density, SPIRE alleviates optimization instability under sparse target distributions. Finally, extensive experiments on various IRSTD benchmarks, including SIRST-UAVB and SIRST4 demonstrate that SPIRE achieves competitive target-level detection performance with consistently low false alarm rate (Fa) and significantly reduced computational cost. Code is publicly available at: this https URL.
>
---
#### [new 040] Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition
- **分类: cs.CV**

- **简介: 该论文属于车辆识别任务，旨在统一细粒度车辆分类与车牌识别。针对现有研究条件受限、属性有限的问题，提出UFPR-VeSV数据集，并验证深度学习模型在复杂场景下的有效性。**

- **链接: [https://arxiv.org/pdf/2604.05271](https://arxiv.org/pdf/2604.05271)**

> **作者:** Gabriel E. Lima; Valfride Nascimento; Eduardo Santos; Eduil Nascimento Jr; Rayson Laroca; David Menotti
>
> **备注:** Accepted for publication in the Journal of the Brazilian Computer Society (JBCS)
>
> **摘要:** Extracting vehicle information from surveillance images is essential for intelligent transportation systems, enabling applications such as traffic monitoring and criminal investigations. While Automatic License Plate Recognition (ALPR) is widely used, Fine-Grained Vehicle Classification (FGVC) offers a complementary approach by identifying vehicles based on attributes such as color, make, model, and type. Although there have been advances in this field, existing studies often assume well-controlled conditions, explore limited attributes, and overlook FGVC integration with ALPR. To address these gaps, we introduce UFPR-VeSV, a dataset comprising 24,945 images of 16,297 unique vehicles with annotations for 13 colors, 26 makes, 136 models, and 14 types. Collected from the Military Police of Paraná (Brazil) surveillance system, the dataset captures diverse real-world conditions, including partial occlusions, nighttime infrared imaging, and varying lighting. All FGVC annotations were validated using license plate information, with text and corner annotations also being provided. A qualitative and quantitative comparison with established datasets confirmed the challenging nature of our dataset. A benchmark using five deep learning models further validated this, revealing specific challenges such as handling multicolored vehicles, infrared images, and distinguishing between vehicle models that share a common platform. Additionally, we apply two optical character recognition models to license plate recognition and explore the joint use of FGVC and ALPR. The results highlight the potential of integrating these complementary tasks for real-world applications. The UFPR-VeSV dataset is publicly available at: this https URL.
>
---
#### [new 041] Neural Network Pruning via QUBO Optimization
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于神经网络压缩任务，旨在解决传统剪枝方法效果不佳的问题。提出一种融合启发式与全局优化的混合QUBO框架，提升剪枝效果。**

- **链接: [https://arxiv.org/pdf/2604.05856](https://arxiv.org/pdf/2604.05856)**

> **作者:** Osama Orabi; Artur Zagitov; Hadi Salloum; Viktor A. Lobachev; Kasymkhan Khubiev; Yaroslav Kholodov
>
> **备注:** 13 pages, 5 figures, 4 tables
>
> **摘要:** Neural network pruning can be formulated as a combinatorial optimization problem, yet most existing approaches rely on greedy heuristics that ignore complex interactions between filters. Formal optimization methods such as Quadratic Unconstrained Binary Optimization (QUBO) provide a principled alternative but have so far underperformed due to oversimplified objective formulations based on metrics like the L1-norm. In this work, we propose a unified Hybrid QUBO framework that bridges heuristic importance estimation with global combinatorial optimization. Our formulation integrates gradient-aware sensitivity metrics - specifically first-order Taylor and second-order Fisher information - into the linear term, while utilizing data-driven activation similarity in the quadratic term. This allows the QUBO objective to jointly capture individual filter relevance and inter-filter functional redundancy. We further introduce a dynamic capacity-driven search to strictly enforce target sparsity without distorting the optimization landscape. Finally, we employ a two-stage pipeline featuring a Tensor-Train (TT) Refinement stage - a gradient-free optimizer that fine-tunes the QUBO-derived solution directly against the true evaluation metric. Experiments on the SIDD image denoising dataset demonstrate that the proposed Hybrid QUBO significantly outperforms both greedy Taylor pruning and traditional L1-based QUBO, with TT Refinement providing further consistent gains at appropriate combinatorial scales. This highlights the potential of hybrid combinatorial formulations for robust, scalable, and interpretable neural network compression.
>
---
#### [new 042] Cross-Resolution Diffusion Models via Network Pruning
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型在非训练分辨率下质量下降的问题。通过网络剪枝提升跨分辨率的一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.05524](https://arxiv.org/pdf/2604.05524)**

> **作者:** Jiaxuan Ren; Junhan Zhu; Huan Wang
>
> **备注:** Accepted by CVPR Findings 2026
>
> **摘要:** Diffusion models have demonstrated impressive image synthesis performance, yet many UNet-based models are trained at certain fixed resolutions. Their quality tends to degrade when generating images at out-of-training resolutions. We trace this issue to resolution-dependent parameter behaviors, where weights that function well at the default resolution can become adverse when spatial scales shift, weakening semantic alignment and causing structural instability in the UNet architecture. Based on this analysis, this paper introduces CR-Diff, a novel method that improves the cross-resolution visual consistency by pruning some parameters of the diffusion model. Specifically, CR-Diff has two stages. It first performs block-wise pruning to selectively eliminate adverse weights. Then, a pruned output amplification is conducted to further purify the pruned predictions. Empirically, extensive experiments suggest that CR-Diff can improve perceptual fidelity and semantic coherence across various diffusion backbones and unseen resolutions, while largely preserving the performance at default resolutions. Additionally, CR-Diff supports prompt-specific refinement, enabling quality enhancement on demand.
>
---
#### [new 043] Unsupervised Multi-agent and Single-agent Perception from Cooperative Views
- **分类: cs.CV**

- **简介: 该论文属于多智能体与单智能体感知任务，解决无监督环境下两者同时感知的问题。通过多智能体数据共享，提出UMS框架，提升3D目标检测性能。**

- **链接: [https://arxiv.org/pdf/2604.05354](https://arxiv.org/pdf/2604.05354)**

> **作者:** Haochen Yang; Baolu Li; Lei Li; Delin Ren; Jiacheng Guo; Minghai Qin; Tianyun Zhang; Hongkai Yu
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** The LiDAR-based multi-agent and single-agent perception has shown promising performance in environmental understanding for robots and automated vehicles. However, there is no existing method that simultaneously solves both multi-agent and single-agent perception in an unsupervised way. By sharing sensor data between multiple agents via communication, this paper discovers two key insights: 1) Improved point cloud density after the data sharing from cooperative views could benefit unsupervised object classification, 2) Cooperative view of multiple agents can be used as unsupervised guidance for the 3D object detection in the single view. Based on these two discovered insights, we propose an Unsupervised Multi-agent and Single-agent (UMS) perception framework that leverages multi-agent cooperation without human annotations to simultaneously solve multi-agent and single-agent perception. UMS combines a learning-based Proposal Purifying Filter to better classify the candidate proposals after multi-agent point cloud density cooperation, followed by a Progressive Proposal Stabilizing module to yield reliable pseudo labels by the easy-to-hard curriculum learning. Furthermore, we design a Cross-View Consensus Learning to use multi-agent cooperative view to guide detection in single-agent view. Experimental results on two public datasets V2V4Real and OPV2V show that our UMS method achieved significantly higher 3D detection performance than the state-of-the-art methods on both multi-agent and single-agent perception tasks in an unsupervised setting.
>
---
#### [new 044] Watch Before You Answer: Learning from Visually Grounded Post-Training
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决视频理解性能不足的问题。研究发现现有数据存在文本线索依赖，提出VidGround方法提升视频理解效果。**

- **链接: [https://arxiv.org/pdf/2604.05117](https://arxiv.org/pdf/2604.05117)**

> **作者:** Yuxuan Zhang; EunJeong Hwang; Huaisong Zhang; Penghui Du; Yiming Jia; Dongfu Jiang; Xuan He; Shenhui Zhang; Ping Nie; Peter West; Kelsey R. Allen
>
> **摘要:** It is critical for vision-language models (VLMs) to comprehensively understand visual, temporal, and textual cues. However, despite rapid progress in multimodal modeling, video understanding performance still lags behind text-based reasoning. In this work, we find that progress is even worse than previously assumed: commonly reported long video understanding benchmarks contain 40-60% of questions that can be answered using text cues alone. Furthermore, we find that these issues are also pervasive in widely used post-training datasets, potentially undercutting the ability of post-training to improve VLM video understanding performance. Guided by this observation, we introduce VidGround as a simple yet effective solution: using only the actual visually grounded questions without any linguistic biases for post-training. When used in tandem with RL-based post-training algorithms, this simple technique improves performance by up to 6.2 points relative to using the full dataset, while using only 69.1% of the original post-training data. Moreover, we show that data curation with a simple post-training algorithm outperforms several more complex post-training techniques, highlighting that data quality is a major bottleneck for improving video understanding in VLMs. These results underscore the importance of curating post-training data and evaluation benchmarks that truly require visual grounding to advance the development of more capable VLMs. Project page: this http URL.
>
---
#### [new 045] SVC 2026: the Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge
- **分类: cs.CV**

- **简介: 该论文介绍SVC 2026挑战，旨在解决微妙视觉信号的鲁棒检测与远程生理测量问题，通过跨域多模态欺骗检测和rPPG估计任务推动计算机视觉与多模态学习发展。**

- **链接: [https://arxiv.org/pdf/2604.05748](https://arxiv.org/pdf/2604.05748)**

> **作者:** Dongliang Zhu; Zhiyi Niu; Bo Zhao; Jiajian Huang; Shuo Ye; Xun Lin; Hui Ma; Taorui Wang; Jiayu Zhang; Chunmei Zhu; Junzhe Cao; Yingjie Ma; Rencheng Song; Albert Clapés; Sergio Escalera; Dan Guo; Zitong Yu
>
> **备注:** Accepted by the SVC workshop @ CVPR 2026
>
> **摘要:** Subtle visual signals, although difficult to perceive with the naked eye, contain important information that can reveal hidden patterns in visual data. These signals play a key role in many applications, including biometric security, multimedia forensics, medical diagnosis, industrial inspection, and affective computing. With the rapid development of computer vision and representation learning techniques, detecting and interpreting such subtle signals has become an emerging research direction. However, existing studies often focus on specific tasks or modalities, and models still face challenges in robustness, representation ability, and generalization when handling subtle and weak signals in real-world environments. To promote research in this area, we organize the Subtle visual Challenge, which aims to learn robust representations for subtle visual signals. The challenge includes two tasks: cross-domain multimodal deception detection and remote photoplethysmography (rPPG) estimation. We hope that this challenge will encourage the development of more robust and generalizable models for subtle visual understanding, and further advance research in computer vision and multimodal learning. A total of 22 teams submitted their final results to this workshop competition, and the corresponding baseline models have been released on the \href{this https URL}{MMDD2026 platform}\footnote{this https URL}
>
---
#### [new 046] Boxer: Robust Lifting of Open-World 2D Bounding Boxes to 3D
- **分类: cs.CV**

- **简介: 该论文属于3D物体定位任务，解决从2D检测结果中恢复3D边界框的问题。提出Boxer算法，通过Transformer网络实现2D到3D的提升，并在多个数据集上取得优于现有方法的效果。**

- **链接: [https://arxiv.org/pdf/2604.05212](https://arxiv.org/pdf/2604.05212)**

> **作者:** Daniel DeTone; Tianwei Shen; Fan Zhang; Lingni Ma; Julian Straub; Richard Newcombe; Jakob Engel
>
> **备注:** project page: this http URL
>
> **摘要:** Detecting and localizing objects in space is a fundamental computer vision problem. While much progress has been made to solve 2D object detection, 3D object localization is much less explored and far from solved, especially for open-world categories. To address this research challenge, we propose Boxer, an algorithm to estimate static 3D bounding boxes (3DBBs) from 2D open-vocabulary object detections, posed images and optional depth either represented as a sparse point cloud or dense depth. At its core is BoxerNet, a transformer-based network which lifts 2D bounding box (2DBB) proposals into 3D, followed by multi-view fusion and geometric filtering to produce globally consistent de-duplicated 3DBBs in metric world space. Boxer leverages the power of existing 2DBB detection algorithms (e.g. DETIC, OWLv2, SAM3) to localize objects in 2D. This allows the main BoxerNet model to focus on lifting to 3D rather than detecting, ultimately reducing the demand for costly annotated 3DBB training data. Extending the CuTR formulation, we incorporate an aleatoric uncertainty for robust regression, a median depth patch encoding to support sparse depth inputs, and large-scale training with over 1.2 million unique 3DBBs. BoxerNet outperforms state-of-the-art baselines in open-world 3DBB lifting, including CuTR in egocentric settings without dense depth (0.532 vs. 0.010 mAP) and on CA-1M with dense depth available (0.412 vs. 0.250 mAP).
>
---
#### [new 047] RHVI-FDD: A Hierarchical Decoupling Framework for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决低光图像中的噪声、细节丢失和色彩失真问题。提出RHVI-FDD框架，通过分层解耦方法分离亮度与色度、细节与噪声，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2604.05781](https://arxiv.org/pdf/2604.05781)**

> **作者:** Junhao Yang; Bo Yang; Hongwei Ge; Yanchun Liang; Heow Pueh Lee; Chunguo Wu
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Low-light images often suffer from severe noise, detail loss, and color distortion, which hinder downstream multimedia analysis and retrieval tasks. The degradation in low-light images is complex: luminance and chrominance are coupled, while within the chrominance, noise and details are deeply entangled, preventing existing methods from simultaneously correcting color distortion, suppressing noise, and preserving fine details. To tackle the above challenges, we propose a novel hierarchical decoupling framework (RHVI-FDD). At the macro level, we introduce the RHVI transform, which mitigates the estimation bias caused by input noise and enables robust luminance-chrominance decoupling. At the micro level, we design a Frequency-Domain Decoupling (FDD) module with three branches for further feature separation. Using the Discrete Cosine Transform, we decompose chrominance features into low, mid, and high-frequency bands that predominantly represent global tone, local details, and noise components, which are then processed by tailored expert networks in a divide-and-conquer manner and fused via an adaptive gating module for content-aware fusion. Extensive experiments on multiple low-light datasets demonstrate that our method consistently outperforms existing state-of-the-art approaches in both objective metrics and subjective visual quality.
>
---
#### [new 048] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors
- **分类: cs.CV; cs.AI**

- **简介: 论文研究图像压缩在比特翻转错误下的鲁棒性，属于图像压缩任务。针对传统编码器鲁棒性不足的问题，提出更鲁棒的Turbo-DDCM变体，提升抗噪能力。**

- **链接: [https://arxiv.org/pdf/2604.05743](https://arxiv.org/pdf/2604.05743)**

> **作者:** Amit Vaisman; Gal Pomerants; Raz Lapid
>
> **摘要:** Modern image compression methods are typically optimized for the rate--distortion--perception trade-off, whereas their robustness to bit-level corruption is rarely examined. We show that diffusion-based compressors built on the Reverse Channel Coding (RCC) paradigm are substantially more robust to bit flips than classical and learned codecs. We further introduce a more robust variant of Turbo-DDCM that significantly improves robustness while only minimally affecting the rate--distortion--perception trade-off. Our findings suggest that RCC-based compression can yield more resilient compressed representations, potentially reducing reliance on error-correcting codes in highly noisy environments.
>
---
#### [new 049] Semantic-Topological Graph Reasoning for Language-Guided Pulmonary Screening
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决临床文本指导下的肺部筛查问题。通过引入语义-拓扑图推理框架，提升分割精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.05620](https://arxiv.org/pdf/2604.05620)**

> **作者:** Chenyu Xue; Yiran Liu; Mian Zhou; Jionglong Su; Zhixiang Lu
>
> **摘要:** Medical image segmentation driven by free-text clinical instructions is a critical frontier in computer-aided diagnosis. However, existing multimodal and foundation models struggle with the semantic ambiguity of clinical reports and fail to disambiguate complex anatomical overlaps in low-contrast scans. Furthermore, fully fine-tuning these massive architectures on limited medical datasets invariably leads to severe overfitting. To address these challenges, we propose a novel Semantic-Topological Graph Reasoning (STGR) framework for language-guided pulmonary screening. Our approach elegantly synergizes the reasoning capabilities of large language models (LLaMA-3-V) with the zero-shot delineation of vision foundation models (MedSAM). Specifically, we introduce a Text-to-Vision Intent Distillation (TVID) module to extract precise diagnostic guidance. To resolve anatomical ambiguity, we formulate mask selection as a dynamic graph reasoning problem, where candidate lesions are modeled as nodes and edges capture spatial and semantic affinities. To ensure deployment feasibility, we introduce a Selective Asymmetric Fine-Tuning (SAFT) strategy that updates less than 1% of the parameters. Rigorous 5-fold cross-validation on the LIDC-IDRI and LNDb datasets demonstrates that our framework establishes a new state-of-the-art. Notably, it achieves an 81.5% Dice Similarity Coefficient (DSC) on LIDC-IDRI, outperforming leading LLM-based tools like LISA by over 5%. Crucially, our SAFT strategy acts as a powerful regularizer, yielding exceptional cross-fold stability (0.6% DSC variance) and paving the way for robust, context-aware clinical deployment.
>
---
#### [new 050] Protecting and Preserving Protest Dynamics for Responsible Analysis
- **分类: cs.CV**

- **简介: 该论文属于隐私保护任务，旨在解决抗议数据分析中的隐私风险问题。通过生成合成图像替代敏感内容，实现安全的集体行动分析。**

- **链接: [https://arxiv.org/pdf/2604.05256](https://arxiv.org/pdf/2604.05256)**

> **作者:** Cohen Archbold; Usman Hassan; Nazmus Sakib; Sen-ching Cheung; Abdullah-Al-Zubaer Imran
>
> **备注:** 21 pages, 6 figures, Submitted to ACM Journal on Responsible Computing
>
> **摘要:** Protest-related social media data are valuable for understanding collective action but inherently high-risk due to concerns surrounding surveillance, repression, and individual privacy. Contemporary AI systems can identify individuals, infer sensitive attributes, and cross-reference visual information across platforms, enabling surveillance that poses risks to protesters and bystanders. In such contexts, large foundation models trained on protest imagery risk memorizing and disclosing sensitive information, leading to cross-platform identity leakage and retroactive participant identification. Existing approaches to automated protest analysis do not provide a holistic pipeline that integrates privacy risk assessment, downstream analysis, and fairness considerations. To address this gap, we propose a responsible computing framework for analyzing collective protest dynamics while reducing risks to individual privacy. Our framework replaces sensitive protest imagery with well-labeled synthetic reproductions using conditional image synthesis, enabling analysis of collective patterns without direct exposure of identifiable individuals. We demonstrate that our approach produces realistic and diverse synthetic imagery while balancing downstream analytical utility with reductions in privacy risk. We further assess demographic fairness in the generated data, examining whether synthetic representations disproportionately affect specific subgroups. Rather than offering absolute privacy guarantees, our method adopts a pragmatic, harm-mitigating approach that enables socially sensitive analysis while acknowledging residual risks.
>
---
#### [new 051] High-Resolution Single-Shot Polarimetric Imaging Made Easy
- **分类: cs.CV**

- **简介: 该论文属于极化成像任务，旨在解决单次捕捉中分辨率低和伪影问题。通过三相机系统和深度学习方法，实现高分辨率极化图像重建。**

- **链接: [https://arxiv.org/pdf/2604.05581](https://arxiv.org/pdf/2604.05581)**

> **作者:** Shuangfan Zhou; Chu Zhou; Heng Guo; Youwei Lyu; Boxin Shi; Zhanyu Ma; Imari Sato
>
> **摘要:** Polarization-based vision has gained increasing attention for providing richer physical cues beyond RGB images. While achieving single-shot capture is highly desirable for practical applications, existing Division-of-Focal-Plane (DoFP) sensors inherently suffer from reduced spatial resolution and artifacts due to their spatial multiplexing mechanism. To overcome these limitations without sacrificing the snapshot capability, we propose EasyPolar, a multi-view polarimetric imaging framework. Our system is grounded in the physical insight that three independent intensity measurements are sufficient to fully characterize linear polarization. Guided by this, we design a triple-camera setup consisting of three synchronized RGB cameras that capture one unpolarized view and two polarized views with distinct orientations. Building upon this hardware design, we further propose a confidence-guided polarization reconstruction network to address the potential misalignment in multi-view fusion. The network performs multi-modal feature fusion under a confidence-aware physical guidance mechanism, which effectively suppresses warping-induced artifacts and enforces explicit geometric constraints on the solution space. Experimental results demonstrate that our method achieves high-quality results and benefits various downstream tasks.
>
---
#### [new 052] Cross-Stage Attention Propagation for Efficient Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，解决多尺度解码器计算冗余问题。提出CSAP框架，通过跨阶段传播注意力图降低计算成本，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.05431](https://arxiv.org/pdf/2604.05431)**

> **作者:** Beoungwoo Kang
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Recent lightweight semantic segmentation methods have made significant progress by combining compact backbones with efficient decoder heads. However, most multi-scale decoders compute attention independently at each feature scale, introducing substantial redundancy since the resulting attention distributions across scales are strongly correlated. We propose Cross-Stage Attention Propagation (CSAP), a decoder framework that computes attention at the deepest feature scale and propagates the resulting attention maps to shallower stages, bypassing query-key computation at those stages entirely. This design preserves multi-scale contextual reasoning while substantially reducing the decoder's computational cost. CSAP-Tiny achieves 42.9% mIoU on ADE20K with only 5.5 GFLOPs, 80.5% on Cityscapes with 21.5 GFLOPs, and 40.9% on COCO-Stuff 164K with 5.5 GFLOPs, surpassing SegNeXt-Tiny by +1.8% on ADE20K while requiring 16.8% fewer floating-point operations.
>
---
#### [new 053] Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强与去噪任务，解决现有方法需分阶段或额外模块的问题，提出单阶段信号衰减扩散模型（SADM），实现亮度调整与降噪同步优化。**

- **链接: [https://arxiv.org/pdf/2604.05727](https://arxiv.org/pdf/2604.05727)**

> **作者:** Ying Liu; Junchao Zhang; Caiyun Wu
>
> **摘要:** Diffusion models excel at image restoration via probabilistic modeling of forward noise addition and reverse denoising, and their ability to handle complex noise while preserving fine details makes them well-suited for Low-Light Image Enhancement (LLIE). Mainstream diffusion based LLIE methods either adopt a two-stage pipeline or an auxiliary correction network to refine U-Net outputs, which severs the intrinsic link between enhancement and denoising and leads to suboptimal performance owing to inconsistent optimization objectives. To address these issues, we propose the Signal Attenuation Diffusion Model (SADM), a novel diffusion process that integrates the signal attenuation mechanism into the diffusion pipeline, enabling simultaneous brightness adjustment and noise suppression in a single stage. Specifically, the signal attenuation coefficient simulates the inherent signal attenuation of low-light degradation in the forward noise addition process, encoding the physical priors of low-light degradation to explicitly guide reverse denoising toward the concurrent optimization of brightness recovery and noise suppression, thereby eliminating the need for extra correction modules or staged training relied on by existing methods. We validate that our design maintains consistency with Denoising Diffusion Implicit Models(DDIM) via multi-scale pyramid sampling, balancing interpretability, restoration quality, and computational efficiency.
>
---
#### [new 054] Active Measurement of Two-Point Correlations
- **分类: cs.CV**

- **简介: 该论文属于两两点相关函数（2PCF）测量任务，解决如何高效估计特定子集的2PCF问题。通过人机协作框架，结合预训练分类器和自适应采样，降低标注成本并提高估计精度。**

- **链接: [https://arxiv.org/pdf/2604.05227](https://arxiv.org/pdf/2604.05227)**

> **作者:** Max Hamilton; Daniel Sheldon; Subhransu Maji
>
> **备注:** AIStats 2026
>
> **摘要:** Two-point correlation functions (2PCF) are widely used to characterize how points cluster in space. In this work, we study the problem of measuring the 2PCF over a large set of points, restricted to a subset satisfying a property of interest. An example comes from astronomy, where scientists measure the 2PCF of star clusters, which make up only a tiny subset of possible sources within a galaxy. This task typically requires careful labeling of sources to construct catalogs, which is time-consuming. We present a human-in-the-loop framework for efficient estimation of 2PCF of target sources. By leveraging a pre-trained classifier to guide sampling, our approach adaptively selects the most informative points for human annotation. After each annotation, it produces unbiased estimates of pair counts across multiple distance bins simultaneously. Compared to simple Monte Carlo approaches, our method achieves substantially lower variance while significantly reducing annotation effort. We introduce a novel unbiased estimator, sampling strategy, and confidence interval construction that together enable scalable and statistically grounded measurement of two-point correlations in astronomy datasets.
>
---
#### [new 055] EDGE-Shield: Efficient Denoising-staGE Shield for Violative Content Filtering via Scalable Reference-Based Matching
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于内容过滤任务，旨在解决生成模型中侵权内容检测的效率与准确性问题。提出EDGE-Shield，在去噪过程中实现高效过滤。**

- **链接: [https://arxiv.org/pdf/2604.06063](https://arxiv.org/pdf/2604.06063)**

> **作者:** Takara Taniguchi; Ryohei Shimizu; Minh-Duc Vo; Kota Izumi; Shiqi Yang; Teppei Suzuki
>
> **摘要:** The advent of Text-to-Image generative models poses significant risks of copyright violation and deepfake generation. Since the rapid proliferation of new copyrighted works and private individuals constantly emerges, reference-based training-free content filters are essential for providing up-to-date protection without the constraints of a fixed knowledge cutoff. However, existing reference-based approaches often lack scalability when handling numerous references and require waiting for finishing image generation. To solve these problems, we propose EDGE-Shield, a scalable content filter during the denoising process that maintains practical latency while effectively blocking violative content. We leverage embedding-based matching for efficient reference comparison. Additionally, we introduce an \textit{$x$}-pred transformation that converts the model's noisy intermediate latent into the pseudo-estimated clean latent at the later stage, enhancing classification accuracy of violative content at earlier denoising stages. We conduct experiments of violative content filtering against two generative models including Z-Image-Turbo and Qwen-Image. EDGE-Shield significantly outperforms traditional reference-based methods in terms of latency; it achieves an approximate $79\%$ reduction in processing time for Z-Image-Turbo and approximate $50\%$ reduction for Qwen-Image, maintaining the filtering accuracy across different model architectures.
>
---
#### [new 056] A Unified Foundation Model for All-in-One Multi-Modal Remote Sensing Image Restoration and Fusion with Language Prompting
- **分类: cs.CV**

- **简介: 该论文属于遥感图像修复与融合任务，解决多模态、多退化类型图像处理问题。提出LLaRS模型，通过语言提示和优化传输机制实现统一处理。**

- **链接: [https://arxiv.org/pdf/2604.05629](https://arxiv.org/pdf/2604.05629)**

> **作者:** Yongchuan Cui; Peng Liu
>
> **摘要:** Remote sensing imagery suffers from clouds, haze, noise, resolution limits, and sensor heterogeneity. Existing restoration and fusion approaches train separate models per degradation type. In this work, we present Language-conditioned Large-scale Remote Sensing restoration model (LLaRS), the first unified foundation model for multi-modal and multi-task remote sensing low-level vision. LLaRS employs Sinkhorn-Knopp optimal transport to align heterogeneous bands into semantically matched slots, routes features through three complementary mixture-of-experts layers (convolutional experts for spatial patterns, channel-mixing experts for spectral fidelity, and attention experts with low-rank adapters for global context), and stabilizes joint training via step-level dynamic weight adjustment. To train LLaRS, we construct LLaRS1M, a million-scale multi-task dataset spanning eleven restoration and enhancement tasks, integrating real paired observations and controlled synthetic degradations with diverse natural language prompts. Experiments show LLaRS consistently outperforms seven competitive models, and parameter-efficient finetuning experiments demonstrate strong transfer capability and adaptation efficiency on unseen data. Repo: this https URL
>
---
#### [new 057] Lightweight True In-Pixel Encryption with FeFET Enabled Pixel Design for Secure Imaging
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于图像安全任务，旨在解决图像传感器数据泄露问题。通过设计一种基于FeFET的轻量级像素加密架构，实现像素级加密，提升图像传输安全性。**

- **链接: [https://arxiv.org/pdf/2604.05147](https://arxiv.org/pdf/2604.05147)**

> **作者:** Md Rahatul Islam Udoy; Diego Ferrer; Wantong Li; Kai Ni; Sumeet Kumar Gupta; Ahmedullah Aziz
>
> **摘要:** Ensuring end-to-end security in image sensors has become essential as visual data can be exposed through multiple stages of the imaging pipeline. Advanced protection requires encryption to occur before pixel values appear on any readout lines. This work introduces a secure pixel sensor (SecurePix), a compact CMOS-compatible pixel architecture that performs true in-pixel encryption using a symmetric key realized through programmable, non-volatile multidomain polarization states of a ferroelectric field-effect transistor. The pixel and array operations are designed and simulated in HSPICE, while a 45 nm CMOS process design kit is used for layout drawing. The resulting layout confirms a pixel pitch of 2.33 x 3.01 um^2. Each pixel's non-volatile programming level defines its analog transfer characteristic, enabling the photodiode voltage to be converted into an encrypted analog output within the pixel. Full-image evaluation shows that ResNet-18 recognition accuracy drops from 99.29 percent to 9.58 percent on MNIST and from 91.33 percent to 6.98 percent on CIFAR-10 after encryption, indicating strong resistance to neural-network-based inference. Lookup-table-based inverse mapping enables recovery for authorized receivers using the same symmetric key. Based on HSPICE simulation, the SecurePix achieves a per-pixel programming power-delay product of 17 uW us and a per-pixel sensing power-delay product of 1.25 uW us, demonstrating low-overhead hardware-level protection.
>
---
#### [new 058] CRISP: Rank-Guided Iterative Squeezing for Robust Medical Image Segmentation under Domain Shift
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决领域偏移带来的性能下降问题。提出CRISP框架，通过概率排名稳定性实现鲁棒分割。**

- **链接: [https://arxiv.org/pdf/2604.05409](https://arxiv.org/pdf/2604.05409)**

> **作者:** Yizhou Fang; Pujin Cheng; Yixiang Liu; Xiaoying Tang; Longxi Zhou
>
> **摘要:** Distribution shift in medical imaging remains a central bottleneck for the clinical translation of medical AI. Failure to address it can lead to severe performance degradation in unseen environments and exacerbate health inequities. Existing methods for domain adaptation are inherently limited by exhausting predefined possibilities through simulated shifts or pseudo-supervision. Such strategies struggle in the open-ended and unpredictable real world, where distribution shifts are effectively infinite. To address this challenge, we introduce an empirical law called ``Rank Stability of Positive Regions'', which states that the relative rank of predicted probabilities for positive voxels remains stable under distribution shift. Guided by this principle, we propose CRISP, a parameter-free and model-agnostic framework requiring no target-domain information. CRISP is the first framework to make segmentation based on rank rather than probabilities. CRISP simulates model behavior under distribution shift via latent feature perturbation, where voxel probability rankings exhibit two stable patterns: regions that consistently retain high probabilities (destined positives according to the principle) and those that remain low-probability (can be safely classified as negatives). Based on these patterns, we construct high-precision (HP) and high-recall (HR) priors and recursively refine them under perturbation. We then design an iterative training framework, making HP and HR progressively ``squeeze'' to the final segmentation. Extensive evaluations on multi-center cardiac MRI and CT-based lung vessel segmentation demonstrate CRISP's superior robustness, significantly outperforming state-of-the-art methods with striking HD95 reductions of up to 0.14 (7.0\% improvement), 1.90 (13.1\% improvement), and 8.39 (38.9\% improvement) pixels across multi-center, demographic, and modality shifts, respectively.
>
---
#### [new 059] ID-Selection: Importance-Diversity Based Visual Token Selection for Efficient LVLM Inference
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决token冗余与信息丢失的平衡问题。提出ID-Selection方法，通过重要性与多样性结合实现高效token选择。**

- **链接: [https://arxiv.org/pdf/2604.05601](https://arxiv.org/pdf/2604.05601)**

> **作者:** Zhaohong Huang; Wenjing Liu; Yuxin Zhang; Fei Chao; Rongrong Ji
>
> **摘要:** Recent advances have explored visual token pruning to accelerate the inference of large vision-language models (LVLMs). However, existing methods often struggle to balance token importance and diversity: importance-based methods tend to retain redundant tokens, whereas diversity-based methods may overlook informative ones. This trade-off becomes especially problematic under high reduction ratios, where preserving only a small subset of visual tokens is critical. To address this issue, we propose ID-Selection, a simple yet effective token selection strategy for efficient LVLM inference. The key idea is to couple importance estimation with diversity-aware iterative selection: each token is first assigned an importance score, after which high-scoring tokens are selected one by one while the scores of similar tokens are progressively suppressed. In this way, ID-Selection preserves informative tokens while reducing redundancy in a unified selection process. Extensive experiments across 5 LVLM backbones and 16 main benchmarks demonstrate that ID-Selection consistently achieves superior performance and efficiency, especially under extreme pruning ratios. For example, on LLaVA-1.5-7B, ID-Selection prunes 97.2% of visual tokens, retaining only 16 tokens, while reducing inference FLOPs by over 97% and preserving 91.8% of the original performance, all without additional training.
>
---
#### [new 060] Geometrical Cross-Attention and Nonvoid Voxelization for Efficient 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像分割任务，旨在提升分割精度与计算效率。提出GCNV-Net，结合几何交叉注意力和非空体素化，有效减少计算量并提高分割效果。**

- **链接: [https://arxiv.org/pdf/2604.05515](https://arxiv.org/pdf/2604.05515)**

> **作者:** Chenxin Yuan; Shoupeng Chen; Haojiang Ye; Yiming Miao; Limei Peng; Pin-Han Ho
>
> **备注:** 20 pages, 13 figures, supplementary material included, submitted to Medical Image Analysis
>
> **摘要:** Accurate segmentation of 3D medical scans is crucial for clinical diagnostics and treatment planning, yet existing methods often fail to achieve both high accuracy and computational efficiency across diverse anatomies and imaging modalities. To address these challenges, we propose GCNV-Net, a novel 3D medical segmentation framework that integrates a Tri-directional Dynamic Nonvoid Voxel Transformer (3DNVT), a Geometrical Cross-Attention module (GCA), and Nonvoid Voxelization. The 3DNVT dynamically partitions relevant voxels along the three orthogonal anatomical planes, namely the transverse, sagittal, and coronal planes, enabling effective modeling of complex 3D spatial dependencies. The GCA mechanism explicitly incorporates geometric positional information during multi-scale feature fusion, significantly enhancing fine-grained anatomical segmentation accuracy. Meanwhile, Nonvoid Voxelization processes only informative regions, greatly reducing redundant computation without compromising segmentation quality, and achieves a 56.13% reduction in FLOPs and a 68.49% reduction in inference latency compared to conventional voxelization. We evaluate GCNV-Net on multiple widely used benchmarks: BraTS2021, ACDC, MSD Prostate, MSD Pancreas, and AMOS2022. Our method achieves state-of-the-art segmentation performance across all datasets, outperforming the best existing methods by 0.65% on Dice, 0.63% on IoU, 1% on NSD, and relatively 14.5% on HD95. All results demonstrate that GCNV-Net effectively balances accuracy and efficiency, and its robustness across diverse organs, disease conditions, and imaging modalities highlights strong potential for clinical deployment.
>
---
#### [new 061] DetailVerifyBench: A Benchmark for Dense Hallucination Localization in Long Image Captions
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于图像描述中的幻觉检测任务，旨在解决长文本中细粒度幻觉定位问题。提出DetailVerifyBench基准，包含1000张跨领域图像及详细标注的幻觉数据。**

- **链接: [https://arxiv.org/pdf/2604.05623](https://arxiv.org/pdf/2604.05623)**

> **作者:** Xinran Wang; Yuxuan Zhang; Xiao Zhang; Haolong Yan; Muxi Diao; Songyu Xu; Zhonghao Yan; Hongbing Li; Kongming Liang; Zhanyu Ma
>
> **备注:** 8 pages, 5 figures. The dataset and code are available at this https URL
>
> **摘要:** Accurately detecting and localizing hallucinations is a critical task for ensuring high reliability of image captions. In the era of Multimodal Large Language Models (MLLMs), captions have evolved from brief sentences into comprehensive narratives, often spanning hundreds of words. This shift exponentially increases the challenge: models must now pinpoint specific erroneous spans or words within extensive contexts, rather than merely flag response-level inconsistencies. However, existing benchmarks lack the fine granularity and domain diversity required to evaluate this capability. To bridge this gap, we introduce DetailVerifyBench, a rigorous benchmark comprising 1,000 high-quality images across five distinct domains. With an average caption length of over 200 words and dense, token-level annotations of multiple hallucination types, it stands as the most challenging benchmark for precise hallucination localization in the field of long image captioning to date. Our benchmark is available at this https URL.
>
---
#### [new 062] ID-Sim: An Identity-Focused Similarity Metric
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ID-Sim，一种面向身份的相似性度量，用于解决视觉模型在身份识别与生成任务中评估不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.05039](https://arxiv.org/pdf/2604.05039)**

> **作者:** Julia Chae; Nicholas Kolkin; Jui-Hsien Wang; Richard Zhang; Sara Beery; Cusuh Ham
>
> **备注:** SB and CH equal advising; Project page this https URL
>
> **摘要:** Humans have remarkable selective sensitivity to identities -- easily distinguishing between highly similar identities, even across significantly different contexts such as diverse viewpoints or lighting. Vision models have struggled to match this capability, and progress toward identity-focused tasks such as personalized image generation is slowed by a lack of identity-focused evaluation metrics. To help facilitate progress, we propose ID-Sim, a feed-forward metric designed to faithfully reflect human selective sensitivity. To build ID-Sim, we curate a high-quality training set of images spanning diverse real-world domains, augmented with generative synthetic data that provides controlled, fine-grained identity and contextual variations. We evaluate our metric on a new unified evaluation benchmark for assessing consistency with human annotations across identity-focused recognition, retrieval, and generative tasks.
>
---
#### [new 063] Few-Shot Semantic Segmentation Meets SAM3
- **分类: cs.CV**

- **简介: 该论文属于少样本语义分割任务，旨在解决传统方法依赖大量训练的问题。通过直接使用SAM3模型，无需微调即可实现高效分割，提升了性能并揭示了负提示的局限性。**

- **链接: [https://arxiv.org/pdf/2604.05433](https://arxiv.org/pdf/2604.05433)**

> **作者:** Yi-Jen Tsai; Yen-Yu Lin; Chien-Yao Wang
>
> **备注:** 14 pages, 3 figures
>
> **摘要:** Few-Shot Semantic Segmentation (FSS) focuses on segmenting novel object categories from only a handful of annotated examples. Most existing approaches rely on extensive episodic training to learn transferable representations, which is both computationally demanding and sensitive to distribution shifts. In this work, we revisit FSS from the perspective of modern vision foundation models and explore the potential of Segment Anything Model 3 (SAM3) as a training-free solution. By repurposing its Promptable Concept Segmentation (PCS) capability, we adopt a simple spatial concatenation strategy that places support and query images into a shared canvas, allowing a fully frozen SAM3 to perform segmentation without any fine-tuning or architectural changes. Experiments on PASCAL-$5^i$ and COCO-$20^i$ show that this minimal design already achieves state-of-the-art performance, outperforming many heavily engineered methods. Beyond empirical gains, we uncover that negative prompts can be counterproductive in few-shot settings, where they often weaken target representations and lead to prediction collapse despite their intended role in suppressing distractors. These findings suggest that strong cross-image reasoning can emerge from simple spatial formulations, while also highlighting limitations in how current foundation models handle conflicting prompt signals. Code at: this https URL
>
---
#### [new 064] 3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建模型压缩任务，解决传统方法需依赖数据训练的问题。提出3DTurboQuant，无需训练即可实现高效量化压缩。**

- **链接: [https://arxiv.org/pdf/2604.05366](https://arxiv.org/pdf/2604.05366)**

> **作者:** Jae Joong Lee
>
> **备注:** Preprint
>
> **摘要:** Every existing method for compressing 3D Gaussian Splatting, NeRF, or transformer-based 3D reconstructors requires learning a data-dependent codebook through per-scene fine-tuning. We show this is unnecessary. The parameter vectors that dominate storage in these models, 45-dimensional spherical harmonics in 3DGS and 1024-dimensional key-value vectors in DUSt3R, fall in a dimension range where a single random rotation transforms any input into coordinates with a known Beta distribution. This makes precomputed, data-independent Lloyd-Max quantization near-optimal, within a factor of 2.7 of the information-theoretic lower bound. We develop 3D, deriving (1) a dimension-dependent criterion that predicts which parameters can be quantized and at what bit-width before running any experiment, (2) norm-separation bounds connecting quantization MSE to rendering PSNR per scene, (3) an entry-grouping strategy extending rotation-based quantization to 2-dimensional hash grid features, and (4) a composable pruning-quantization pipeline with a closed-form compression ratio. On NeRF Synthetic, 3DTurboQuant compresses 3DGS by 3.5x with 0.02dB PSNR loss and DUSt3R KV caches by 7.9x with 39.7dB pointmap fidelity. No training, no codebook learning, no calibration data. Compression takes seconds. The code will be released (this https URL)
>
---
#### [new 065] EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes","Hands" and "Minds"
- **分类: cs.CV**

- **简介: 该论文提出EchoAgent，解决 echocardiography 解释可靠性问题，通过整合视觉、操作和推理能力，实现端到端分析。**

- **链接: [https://arxiv.org/pdf/2604.05541](https://arxiv.org/pdf/2604.05541)**

> **作者:** Qin Wang; Zhiqing He; Yu Liu; Bowen Guo; Zeju Li; Miao Zhao; Wenhao Ju; Zhiling Luo; Xianhong Shu; Yi Guo; Yuanyuan Wang
>
> **备注:** Accepted by CVPR 2026 CV4Clinical, 11 pages, 6 figures
>
> **摘要:** Reliable interpretation of echocardiography (Echo) is crucial for assessing cardiac function, which demands clinicians to synchronously orchestrate multiple capabilities, including visual observation (eyes), manual measurement (hands), and expert knowledge learning and reasoning (minds). While current task-specific deep-learning approaches and multimodal large language models have demonstrated promise in assisting Echo analysis through automated segmentation or reasoning, they remain focused on restricted skills, i.e., eyes-hands or eyes-minds, thereby limiting clinical reliability and utility. To address these issues, we propose EchoAgent, an agentic system tailored for end-to-end Echo interpretation, which achieves a fully coordinated eyes-hands-minds workflow that learns, observes, operates, and reasons like a cardiac sonographer. First, we introduce an expertise-driven cognition engine where our agent can automatically assimilate credible Echo guidelines into a structured knowledge base, thus constructing an Echo-customized mind. Second, we devise a hierarchical collaboration toolkit to endow EchoAgent with eyes-hands, which can automatically parse Echo video streams, identify cardiac views, perform anatomical segmentation, and quantitative measurement. Third, we integrate the perceived multimodal evidence with the exclusive knowledge base into an orchestrated reasoning hub to conduct explainable inferences. We evaluate EchoAgent on CAMUS and MIMIC-EchoQA datasets, which cover 48 distinct echocardiographic views spanning 14 cardiac anatomical regions. Experimental results show that EchoAgent achieves optimal performance across diverse structure analyses, yielding overall accuracy of up to 80.00%. Importantly, EchoAgent empowers a single system with abilities to learn, observe, operate and reason like an echocardiologist, which holds great promise for reliable Echo interpretation.
>
---
#### [new 066] Sparse Gain Radio Map Reconstruction With Geometry Priors and Uncertainty-Guided Measurement Selection
- **分类: cs.CV**

- **简介: 该论文属于无线通信中的射频地图重建任务，旨在解决复杂城市环境中稀疏测量下的地图构建问题。通过结合几何先验和不确定性估计，提出GeoUQ-GFNet模型，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.05788](https://arxiv.org/pdf/2604.05788)**

> **作者:** Zhihan Zeng; Ning Wei; Muhammad Baqer Mollah; Kaihe Wang; Phee Lep Yeoh; Fei Xu; Yue Xiu; Zhongpei Zhang
>
> **摘要:** Radio maps are important for environment-aware wireless communication, network planning, and radio resource optimization. However, dense radio map construction remains challenging when only a limited number of measurements are available, especially in complex urban environments with strong blockages, irregular geometry, and restricted sensing accessibility. Existing methods have explored interpolation, low-rank cartography, deep completion, and channel knowledge map (CKM) construction, but many of these methods insufficiently exploit explicit geometric priors or overlook the value of predictive uncertainty for subsequent sensing. In this paper, we study sparse gain radio map reconstruction from a geometry-aware and active sensing perspective. We first construct \textbf{UrbanRT-RM}, a controllable ray-tracing benchmark with diverse urban layouts, multiple base-station deployments, and multiple sparse sampling modes. We then propose \textbf{GeoUQ-GFNet}, a lightweight network that jointly predicts a dense gain radio map and a spatial uncertainty map from sparse measurements and structured scene priors. The predicted uncertainty is further used to guide active measurement selection under limited sensing budgets. Extensive experiments show that our proposed GeoUQ-GFNet method achieves strong and consistent reconstruction performance across different scenes and transmitter placements generated using UrbanRT-RM. Moreover, uncertainty-guided querying provides more effective reconstruction improvement than non-adaptive sampling under the same additional measurement budget. These results demonstrate the effectiveness of combining geometry-aware learning, uncertainty estimation, and benchmark-driven evaluation for sparse radio map reconstruction in complex urban environments.
>
---
#### [new 067] Let Geometry GUIDE: Layer-wise Unrolling of Geometric Priors in Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在解决模型在处理现实视觉流时缺乏物理空间感知的问题。通过引入 GUIDE 框架，逐步注入几何先验信息，提升模型的空间推理能力。**

- **链接: [https://arxiv.org/pdf/2604.05695](https://arxiv.org/pdf/2604.05695)**

> **作者:** Chongyu Wang; Ting Huang; Chunyu Sun; Xinyu Ning; Di Wang; Hao Tang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in 2D visual tasks but still exhibit limited physical spatial awareness when processing real-world visual streams. Recently, feed-forward geometric foundation models, which implicitly extract geometric priors, have provided a new pathway to address this issue. However, existing geometry-aware MLLMs are predominantly constrained by the paradigm of single deep-layer extraction and input-level fusion. This flattened fusion leads to the loss of local geometric details and causes semantic mismatches in the early layers. To break this bottleneck, we propose GUIDE (Geometric Unrolling Inside MLLM Early-layers), a progressive geometric priors injection framework. GUIDE performs multi-level sampling within the geometric encoder, comprehensively capturing multi-granularity features ranging from local edges to global topologies. Subsequently, we rigorously align and fuse these multi-level geometric priors step-by-step with the early layers of the MLLM. Building upon the injection of multi-granularity geometric information, this design guides the model to progressively learn the 2D-to-3D transitional process. Furthermore, we introduce a context-aware gating that enables the model to fetch requisite spatial cues based on current semantics, thereby maximizing the utilization efficiency of spatial priors and effectively suppressing redundant geometric noise. Extensive experiments demonstrate that GUIDE significantly outperforms existing baselines on multiple complex spatial reasoning and perception tasks, establishing a novel paradigm for integrating 3D geometric priors into large models.
>
---
#### [new 068] Learning to Synergize Semantic and Geometric Priors for Limited-Data Wheat Disease Segmentation
- **分类: cs.CV**

- **简介: 该论文属于小麦病害分割任务，解决有限数据下的分割难题。通过融合语义与几何先验，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05415](https://arxiv.org/pdf/2604.05415)**

> **作者:** Shijie Wang; Zijian Wang; Yadan Luo; Scott Chapman; Xin Yu; Zi Huang
>
> **摘要:** Wheat disease segmentation is fundamental to precision agriculture but faces severe challenges from significant intra-class temporal variations across growth stages. Such substantial appearance shifts make collecting a representative dataset for training from scratch both labor-intensive and impractical. To address this, we propose SGPer, a Semantic-Geometric Prior Synergization framework that treats wheat disease segmentation under limited data as a coupled task of disease-specific semantic perception and disease boundary localization. Our core insight is that pretrained DINOv2 provides robust category-aware semantic priors to handle appearance shifts, which can be converted into coarse spatial prompts to guide SAM for the precise localization of disease boundaries. Specifically, SGPer designs disease-sensitive adapters with multiple disease-friendly filters and inserts them into both DINOv2 and SAM to align their pretrained representations with disease-specific characteristics. To operationalize this synergy, SGPer transforms DINOv2-derived features into dense, category-specific point prompts to ensure comprehensive spatial coverage of all disease regions. To subsequently eliminate prompt redundancy and ensure highly accurate mask generation, it dynamically filters these dense candidates by cross-referencing SAM's iterative mask confidence with the category-specific semantic consistency derived from DINOv2. Ultimately, SGPer distills a highly informative set of prompts to activate SAM's geometric priors, achieving precise and robust segmentation that remains strictly invariant to temporal appearance changes. Extensive evaluations demonstrate that SGPer consistently achieves state-of-the-art performance on wheat disease and organ segmentation benchmarks, especially in data-constrained scenarios.
>
---
#### [new 069] GESS: Multi-cue Guided Local Feature Learning via Geometric and Semantic Synergy
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的局部特征学习任务，旨在解决关键点检测不稳定和描述子区分度不足的问题。通过融合语义与几何信息，提出多线索引导的特征学习框架，提升检测鲁棒性和描述子性能。**

- **链接: [https://arxiv.org/pdf/2604.05359](https://arxiv.org/pdf/2604.05359)**

> **作者:** Yang Yi; Xieyuanli Chen; Jinpu Zhang; Hui Shen; Dewen Hu
>
> **摘要:** Robust local feature detection and description are foundational tasks in computer vision. Existing methods primarily rely on single appearance cues for modeling, leading to unstable keypoints and insufficient descriptor discriminability. In this paper, we propose a multi-cue guided local feature learning framework that leverages semantic and geometric cues to synergistically enhance detection robustness and descriptor discriminability. Specifically, we construct a joint semantic-normal prediction head and a depth stability prediction head atop a lightweight backbone. The former leverages a shared 3D vector field to deeply couple semantic and normal cues, thereby resolving optimization interference from heterogeneous inconsistencies. The latter quantifies the reliability of local regions from a geometric consistency perspective, providing deterministic guidance for robust keypoint selection. Based on these predictions, we introduce the Semantic-Depth Aware Keypoint (SDAK) mechanism for feature detection. By coupling semantic reliability with depth stability, SDAK reweights keypoint responses to suppress spurious features in unreliable regions. For descriptor construction, we design a Unified Triple-Cue Fusion (UTCF) module, which employs a semantic-scheduled gating mechanism to adaptively inject multi-attribute features, improving descriptor discriminability. Extensive experiments on four benchmarks validate the effectiveness of the proposed framework. The source code and pre-trained model will be available at: this https URL.
>
---
#### [new 070] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于长视频理解任务，旨在解决视频上下文受限和语义匹配不足的问题。提出VideoStir框架，通过结构化图与意图感知检索提升长视频的RAG效果。**

- **链接: [https://arxiv.org/pdf/2604.05418](https://arxiv.org/pdf/2604.05418)**

> **作者:** Honghao Fu; Miao Xu; Yiwei Wang; Dailing Zhang; Liu Jun; Yujun Cai
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query's intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query's reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame-query intent alignment. Experiments show that VideoStir is competitive with state-of-the-art baselines without relying on auxiliary information, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning. Codes and checkpoints are available at Github.
>
---
#### [new 071] Prior-guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images
- **分类: cs.CV**

- **简介: 该论文属于多模态变化检测任务，旨在解决跨模态交互不足和细粒度变化建模困难的问题。提出STSF-Net框架，融合光学与SAR图像特征，提升变化检测精度。**

- **链接: [https://arxiv.org/pdf/2604.05527](https://arxiv.org/pdf/2604.05527)**

> **作者:** Xuanguang Liu; Lei Ding; Yujie Li; Chenguang Dai; Zhenchao Zhang; Mengmeng Li; Ziyi Yang; Yifan Sun; Yongqi Sun; Hanyun Wang
>
> **摘要:** Multimodal change detection (MMCD) identifies changed areas in multimodal remote sensing (RS) data, demonstrating significant application value in land use monitoring, disaster assessment, and urban sustainable development. However, literature MMCD approaches exhibit limitations in cross-modal interaction and exploiting modality-specific characteristics. This leads to insufficient modeling of fine-grained change information, thus hindering the precise detection of semantic changes in multimodal data. To address the above problems, we propose STSF-Net, a framework designed for MMCD between optical and SAR images. STSF-Net jointly models modality-specific and spatio-temporal common features to enhance change representations. Specifically, modality-specific features are exploited to capture genuine semantic change signals, while spatio-temporal common features are embedded to suppress pseudo-changes caused by differences in imaging mechanisms. Furthermore, we introduce an optical and SAR feature fusion strategy that adaptively adjusts feature importance based on semantic priors obtained from pre-trained foundational models, enabling semantic-guided adaptive fusion of multi-modal information. In addition, we introduce the Delta-SN6 dataset, the first openly-accessible multiclass MMCD benchmark consisting of very-high-resolution (VHR) fully polarimetric SAR and optical images. Experimental results on Delta-SN6, BRIGHT, and Wuhan-Het datasets demonstrate that our method outperforms the state-of-the-art (SOTA) by 3.21%, 1.08%, and 1.32% in mIoU, respectively. The associated code and Delta-SN6 dataset will be released at: this https URL.
>
---
#### [new 072] Graph-PiT: Enhancing Structural Coherence in Part-Based Image Synthesis via Graph Priors
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于图像生成任务，旨在解决部分间结构不连贯的问题。通过引入图先验和层次图神经网络，增强部分间的语义与空间关系建模。**

- **链接: [https://arxiv.org/pdf/2604.06074](https://arxiv.org/pdf/2604.06074)**

> **作者:** Junbin Zhang; Meng Cao; Feng Tan; Yikai Lin; Yuexian Zou
>
> **备注:** 11 pages, 5 figures, Accepted by ICME 2026
>
> **摘要:** Achieving fine-grained and structurally sound controllability is a cornerstone of advanced visual generation. Existing part-based frameworks treat user-provided parts as an unordered set and therefore ignore their intrinsic spatial and semantic relationships, which often results in compositions that lack structural integrity. To bridge this gap, we propose Graph-PiT, a framework that explicitly models the structural dependencies of visual components using a graph prior. Specifically, we represent visual parts as nodes and their spatial-semantic relationships as edges. At the heart of our method is a Hierarchical Graph Neural Network (HGNN) module that performs bidirectional message passing between coarse-grained part-level super-nodes and fine-grained IP+ token sub-nodes, refining part embeddings before they enter the generative pipeline. We also introduce a graph Laplacian smoothness loss and an edge-reconstruction loss so that adjacent parts acquire compatible, relation-aware embeddings. Quantitative experiments on controlled synthetic domains (character, product, indoor layout, and jigsaw), together with qualitative transfer to real web images, show that Graph-PiT improves structural coherence over vanilla PiT while remaining compatible with the original IP-Prior pipeline. Ablation experiments confirm that explicit relational reasoning is crucial for enforcing user-specified adjacency constraints. Our approach not only enhances the plausibility of generated concepts but also offers a scalable and interpretable mechanism for complex, multi-part image synthesis. The code is available at this https URL.
>
---
#### [new 073] 3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决烟雾环境下多视角图像重建难题。通过引入视觉先验和轻量级介质分支，提升重建效果与视图一致性。**

- **链接: [https://arxiv.org/pdf/2604.05687](https://arxiv.org/pdf/2604.05687)**

> **作者:** Xinye Zheng; Fei Wang; Yiqi Nie; Kun Li; Junjie Chen; Jiaqi Zhao; Yanyan Wei; Zhiliang Wu
>
> **摘要:** Reconstructing 3D scenes from smoke-degraded multi-view images is particularly difficult because smoke introduces strong scattering effects, view-dependent appearance changes, and severe degradation of cross-view consistency. To address these issues, we propose a framework that integrates visual priors with efficient 3D scene modeling. We employ Nano-Banana-Pro to enhance smoke-degraded images and provide clearer visual observations for reconstruction and develop Smoke-GS, a medium-aware 3D Gaussian Splatting framework for smoke scene reconstruction and restoration-oriented novel view synthesis. Smoke-GS models the scene using explicit 3D Gaussians and introduces a lightweight view-dependent medium branch to capture direction-dependent appearance variations caused by smoke. Our method preserves the rendering efficiency of 3D Gaussian Splatting while improving robustness to smoke-induced degradation. Results demonstrate the effectiveness of our method for generating consistent and visually clear novel views in challenging smoke environments.
>
---
#### [new 074] Toward Aristotelian Medical Representations: Backpropagation-Free Layer-wise Analysis for Interpretable Generalized Metric Learning on MedMNIST
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决深度学习模型的不透明性问题。通过提出A-ROM框架，使用可解释的kNN和概念字典替代传统决策层，实现高效、透明的医学概念建模。**

- **链接: [https://arxiv.org/pdf/2604.06017](https://arxiv.org/pdf/2604.06017)**

> **作者:** Michael Karnes; Alper Yilmaz
>
> **摘要:** While deep learning has achieved remarkable success in medical imaging, the "black-box" nature of backpropagation-based models remains a significant barrier to clinical adoption. To bridge this gap, we propose Aristotelian Rapid Object Modeling (A-ROM), a framework built upon the Platonic Representation Hypothesis (PRH). This hypothesis posits that models trained on vast, diverse datasets converge toward a universal and objective representation of reality. By leveraging the generalizable metric space of pretrained Vision Transformers (ViTs), A-ROM enables the rapid modeling of novel medical concepts without the computational burden or opacity of further gradient-based fine-tuning. We replace traditional, opaque decision layers with a human-readable concept dictionary and a k-Nearest Neighbors (kNN) classifier to ensure the model's logic remains interpretable. Experiments on the MedMNIST v2 suite demonstrate that A-ROM delivers performance competitive with standard benchmarks while providing a simple and scalable, "few-shot" solution that meets the rigorous transparency demands of modern clinical environments.
>
---
#### [new 075] PDMP: Rethinking Balanced Multimodal Learning via Performance-Dominant Modality Prioritization
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决多模态模型性能不佳的问题。通过引入性能主导模态优先策略（PDMP），提升模型效果。**

- **链接: [https://arxiv.org/pdf/2604.05773](https://arxiv.org/pdf/2604.05773)**

> **作者:** Shicai Wei; Chunbo Luo; Qiang Zhu; Yang Luo
>
> **摘要:** Multimodal learning has attracted increasing attention due to its practicality. However, it often suffers from insufficient optimization, where the multimodal model underperforms even compared to its unimodal counterparts. Existing methods attribute this problem to the imbalanced learning between modalities and solve it by gradient modulation. This paper argues that balanced learning is not the optimal setting for multimodal learning. On the contrary, imbalanced learning driven by the performance-dominant modality that has superior unimodal performance can contribute to better multimodal performance. And the under-optimization problem is caused by insufficient learning of the performance-dominant modality. To this end, we propose the Performance-Dominant Modality Prioritization (PDMP) strategy to assist multimodal learning. Specifically, PDMP firstly mines the performance-dominant modality via the performance ranking of the independently trained unimodal model. Then PDMP introduces asymmetric coefficients to modulate the gradients of each modality, enabling the performance-dominant modality to dominate the optimization. Since PDMP only relies on the unimodal performance ranking, it is independent of the structures and fusion methods of the multimodal model and has great potential for practical scenarios. Finally, extensive experiments on various datasets validate the superiority of PDMP.
>
---
#### [new 076] Simultaneous Dual-View Mammogram Synthesis Using Denoising Diffusion Probabilistic Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决双视角乳腺X光片数据不完整的问题。通过差值引导的扩散模型同时生成CC和MLO视图，提升数据集质量和跨视图一致性。**

- **链接: [https://arxiv.org/pdf/2604.05110](https://arxiv.org/pdf/2604.05110)**

> **作者:** Jorge Alberto Garza-Abdala; Gerardo A. Fumagal-González; Eduardo de Avila-Armenta; Sadam Hussain; Jasiel H. Toscano-Martínezb; Diana S. M. Rosales Gurmendi; Alma A. Pedro-Pérez; Jose G. Tamez-Pena
>
> **备注:** Accepted and presented at SPIE Medical Imaging 2025 (Vancouver, Canada)
>
> **摘要:** Breast cancer screening relies heavily on mammography, where the craniocaudal (CC) and mediolateral oblique (MLO) views provide complementary information for diagnosis. However, many datasets lack complete paired views, limiting the development of algorithms that depend on cross-view consistency. To address this gap, we propose a three-channel denoising diffusion probabilistic model capable of simultaneously generating CC and MLO views of a single breast. In this configuration, the two mammographic views are stored in separate channels, while a third channel encodes their absolute difference to guide the model toward learning coherent anatomical relationships between projections. A pretrained DDPM from Hugging Face was fine-tuned on a private screening dataset and used to synthesize dual-view pairs. Evaluation included geometric consistency via automated breast mask segmentation and distributional comparison with real images, along with qualitative inspection of cross-view alignment. The results show that the difference-based encoding helps preserve the global breast structure across views, producing synthetic CC-MLO pairs that resemble real acquisitions. This work demonstrates the feasibility of simultaneous dual-view mammogram synthesis using a difference-guided DDPM, highlighting its potential for dataset augmentation and future cross-view-aware AI applications in breast imaging.
>
---
#### [new 077] DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出DiffHDR，解决LDR视频转HDR的问题，通过视频扩散模型实现高动态范围重建与可控重曝光。**

- **链接: [https://arxiv.org/pdf/2604.06161](https://arxiv.org/pdf/2604.06161)**

> **作者:** Zhengming Yu; Li Ma; Mingming He; Leo Isikdogan; Yuancheng Xu; Dmitriy Smirnov; Pablo Salamanca; Dao Mi; Pablo Delgado; Ning Yu; Julien Philip; Xin Li; Wenping Wang; Paul Debevec
>
> **备注:** Project page: this https URL
>
> **摘要:** Most digital videos are stored in 8-bit low dynamic range (LDR) formats, where much of the original high dynamic range (HDR) scene radiance is lost due to saturation and quantization. This loss of highlight and shadow detail precludes mapping accurate luminance to HDR displays and limits meaningful re-exposure in post-production workflows. Although techniques have been proposed to convert LDR images to HDR through dynamic range expansion, they struggle to restore realistic detail in the over- and underexposed regions. To address this, we present DiffHDR, a framework that formulates LDR-to-HDR conversion as a generative radiance inpainting task within the latent space of a video diffusion model. By operating in Log-Gamma color space, DiffHDR leverages spatio-temporal generative priors from a pretrained video diffusion model to synthesize plausible HDR radiance in over- and underexposed regions while recovering the continuous scene radiance of the quantized pixels. Our framework further enables controllable LDR-to-HDR video conversion guided by text prompts or reference images. To address the scarcity of paired HDR video data, we develop a pipeline that synthesizes high-quality HDR video training data from static HDRI maps. Extensive experiments demonstrate that DiffHDR significantly outperforms state-of-the-art approaches in radiance fidelity and temporal stability, producing realistic HDR videos with considerable latitude for re-exposure.
>
---
#### [new 078] Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在提升模型在低数据条件下的鲁棒性。通过扩展ZACH-ViT，评估其在图像退化和对抗攻击下的表现，验证其在真实场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2604.06099](https://arxiv.org/pdf/2604.06099)**

> **作者:** Athanasios Angelakis; Marta Gomez-Barrero
>
> **备注:** Accepted at CVPR 2026 Workshop (PHAROS-AIF-MIH)
>
> **摘要:** The recently introduced ZACH-ViT (Zero-token Adaptive Compact Hierarchical Vision Transformer) formalized a compact permutation-invariant Vision Transformer for medical imaging and argued that architectural alignment with spatial structure can matter more than universal benchmark dominance. Its design was motivated by the observation that positional embeddings and a dedicated class token encode fixed spatial assumptions that may be suboptimal when spatial organization is weakly informative, locally distributed, or variable across biomedical images. The foundational study established a regime-dependent clean performance profile across MedMNIST, but did not examine robustness in detail. In this work, we present the first robustness-focused extension of ZACH-ViT by evaluating its behavior under common image corruptions and adversarial perturbations in the same low-data setting. We compare ZACH-ViT with three scratch-trained compact baselines, ABMIL, Minimal-ViT, and TransMIL, on seven MedMNIST datasets using 50 samples per class, fixed hyperparameters, and five random seeds. Across the benchmark, ZACH-ViT achieves the best overall mean rank on clean data (1.57) and under common corruptions (1.57), indicating a favorable balance between baseline predictive performance and robustness to realistic image degradation. Under adversarial stress, all models deteriorate substantially; nevertheless, ZACH-ViT remains competitive, ranking first under FGSM (2.00) and second under PGD (2.29), where ABMIL performs best overall. These results extend the original ZACH-ViT narrative: the advantages of compact permutation-invariant transformers are not limited to clean evaluation, but can persist under realistic perturbation stress in low-data medical imaging, while adversarial robustness remains an open challenge for all evaluated models.
>
---
#### [new 079] BPC-Net: Annotation-Free Skin Lesion Segmentation via Boundary Probability Calibration
- **分类: cs.CV**

- **简介: 该论文属于皮肤病变分割任务，解决无标注数据下的分割问题。提出BPC-Net框架，通过边界概率校准提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.05594](https://arxiv.org/pdf/2604.05594)**

> **作者:** Yujie Yao; Yuhaohang He; Junjie Huang; Zhou Liu; Jiangzhao Li; Yan Qiao; Wen Xiao; Yunsen Liang; Xiaofan Li
>
> **摘要:** Annotation-free skin lesion segmentation is attractive for low-resource dermoscopic deployment. However, its performance remains constrained by three coupled challenges: noisy pseudo-label supervision, unstable transfer under limited target-domain data, and boundary probability under-confidence. Most existing annotation-free methods primarily focus on pseudo-label denoising. In contrast, the effect of compressed boundary probabilities on final mask quality has received less explicit attention, although it directly affects contour completeness and cannot be adequately corrected by global threshold adjustment alone. To address this issue, we propose BPC-Net, a boundary probability calibration framework for annotation-free skin lesion segmentation. The core of the framework is Gaussian Probability Smoothing (GPS), which performs localized probability-space calibration before thresholding to recover under-confident lesion boundaries without inducing indiscriminate foreground expansion. To support this calibration under noisy pseudo-supervision and cross-domain transfer, we further incorporate two auxiliary designs: a feature-decoupled decoder that separately handles context suppression, detail recovery, and boundary refinement, and an interaction-branch adaptation strategy that updates only the pseudo-label interaction branch while preserving the deployed image-only segmentation path. Under a strictly annotation-free protocol, no manual masks are used during training or target-domain adaptation, and validation labels, when available, are used only for final operating-point selection. Experiments on ISIC-2017, ISIC-2018, and PH2 show that the proposed framework achieves state-of-the-art performance among published unsupervised methods, reaching a macro-average Dice coefficient and Jaccard index of 85.80\% and 76.97\%, respectively, while approaching supervised reference performance on PH2.
>
---
#### [new 080] SonoSelect: Efficient Ultrasound Perception via Active Probe Exploration
- **分类: cs.CV**

- **简介: 该论文属于医学影像任务，旨在解决超声扫描中视图冗余问题。提出SonoSelect方法，通过主动探索减少视图数量，提升诊断效率。**

- **链接: [https://arxiv.org/pdf/2604.05933](https://arxiv.org/pdf/2604.05933)**

> **作者:** Yixin Zhang; Yunzhong Hou; Longqi Li; Zhenyue Qin; Yang Liu; Yue Yao
>
> **摘要:** Ultrasound perception typically requires multiple scan views through probe movement to reduce diagnostic ambiguity, mitigate acoustic occlusions, and improve anatomical coverage. However, not all probe views are equally informative. Exhaustively acquiring a large number of views can introduce substantial redundancy, increase scanning and processing costs. To address this, we define an active view exploration task for ultrasound and propose SonoSelect, an ultrasound-specific method that adaptively guides probe movement based on current observations. Specifically, we cast ultrasound active view exploration as a sequential decision-making problem. Each new 2D ultrasound view is fused into a 3D spatial memory of the observed anatomy, which guides the next probe position. On top of this formulation, we propose an ultrasound-specific objective that favors probe movements with greater organ coverage, lower reconstruction uncertainty, and less redundant scanning. Experiments on the ultrasound simulator show that SonoSelect achieves promising multi-view organ classification accuracy using only 2 out of N views. Furthermore, for a more difficult kidney cyst detection task, it reaches 54.56% kidney coverage and 35.13% cyst coverage, with short trajectories consistently centered on the target cyst.
>
---
#### [new 081] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于室内资产检测任务，解决多视角图像中目标物体的3D检测与分割问题。通过结合2D检测模型与3DGS场景，提升检测精度与一致性。**

- **链接: [https://arxiv.org/pdf/2604.05316](https://arxiv.org/pdf/2604.05316)**

> **作者:** Monica Tang; Avideh Zakhor
>
> **备注:** Accepted to CVPR 2026 3DMV Workshop
>
> **摘要:** We present an approach for object-level detection and segmentation of target indoor assets in 3D Gaussian Splatting (3DGS) scenes, reconstructed from 360° drone-captured imagery. We introduce a 3D object codebook that jointly leverages mask semantics and spatial information of their corresponding Gaussian primitives to guide multi-view mask association and indoor asset detection. By integrating 2D object detection and segmentation models with semantically and spatially constrained merging procedures, our method aggregates masks from multiple views into coherent 3D object instances. Experiments on two large indoor scenes demonstrate reliable multi-view mask consistency, improving F1 score by 65% over state-of-the-art baselines, and accurate object-level 3D indoor asset detection, achieving an 11% mAP gain over baseline methods.
>
---
#### [new 082] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于知识驱动的视觉问答任务，旨在解决传统方法过度依赖图像检索、忽视VLM潜力的问题。提出WikiSeeker框架，利用VLM作为查询优化和可靠信息筛选的代理，提升检索与回答效果。**

- **链接: [https://arxiv.org/pdf/2604.05818](https://arxiv.org/pdf/2604.05818)**

> **作者:** Yingjian Zhu; Xinming Wang; Kun Ding; Ying Wang; Bin Fan; Shiming Xiang
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Multi-modal Retrieval-Augmented Generation (RAG) has emerged as a highly effective paradigm for Knowledge-Based Visual Question Answering (KB-VQA). Despite recent advancements, prevailing methods still primarily depend on images as the retrieval key, and often overlook or misplace the role of Vision-Language Models (VLMs), thereby failing to leverage their potential fully. In this paper, we introduce WikiSeeker, a novel multi-modal RAG framework that bridges these gaps by proposing a multi-modal retriever and redefining the role of VLMs. Rather than serving merely as answer generators, we assign VLMs two specialized agents: a Refiner and an Inspector. The Refiner utilizes the capability of VLMs to rewrite the textual query according to the input image, significantly improving the performance of the multimodal retriever. The Inspector facilitates a decoupled generation strategy by selectively routing reliable retrieved context to another LLM for answer generation, while relying on the VLM's internal knowledge when retrieval is unreliable. Extensive experiments on EVQA, InfoSeek, and M2KR demonstrate that WikiSeeker achieves state-of-the-art performance, with substantial improvements in both retrieval accuracy and answer quality. Our code will be released on this https URL.
>
---
#### [new 083] The Character Error Vector: Decomposable errors for page-level OCR evaluation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文档理解任务，旨在解决页面级OCR评估难题。提出Character Error Vector（CEV）方法，分解错误来源，提升OCR质量评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.06160](https://arxiv.org/pdf/2604.06160)**

> **作者:** Jonathan Bourne; Mwiza Simbeye; Joseph Nockels
>
> **备注:** 6643 words, 5 figures, 15 tables
>
> **摘要:** The Character Error Rate (CER) is a key metric for evaluating the quality of Optical Character Recognition (OCR). However, this metric assumes that text has been perfectly parsed, which is often not the case. Under page-parsing errors, CER becomes undefined, limiting its use as a metric and making evaluating page-level OCR challenging, particularly when using data that do not share a labelling schema. We introduce the Character Error Vector (CEV), a bag-of-characters evaluator for OCR. The CEV can be decomposed into parsing and OCR, and interaction error components. This decomposability allows practitioners to focus on the part of the Document Understanding pipeline that will have the greatest impact on overall text extraction quality. The CEV can be implemented using a variety of methods, of which we demonstrate SpACER (Spatially Aware Character Error Rate) and a Character distribution method using the Jensen-Shannon Distance. We validate the CEV's performance against other metrics: first, the relationship with CER; then, parse quality; and finally, as a direct measure of page-level OCR quality. The validation process shows that the CEV is a valuable bridge between parsing metrics and local metrics like CER. We analyse a dataset of archival newspapers made of degraded images with complex layouts and find that state-of-the-art end-to-end models are outperformed by more traditional pipeline approaches. Whilst the CEV requires character-level positioning for optimal triage, thresholding on easily available values can predict the main error source with an F1 of 0.91. We provide the CEV as part of a Python library to support Document understanding research.
>
---
#### [new 084] LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LSRM模型，解决3D物体高保真重建与逆渲染问题。通过扩展上下文窗口和优化注意力机制，提升纹理与几何细节恢复效果。**

- **链接: [https://arxiv.org/pdf/2604.05182](https://arxiv.org/pdf/2604.05182)**

> **作者:** Zhengqin Li; Cheng Zhang; Jakob Engel; Zhao Dong
>
> **摘要:** We introduce the Large Sparse Reconstruction Model to study how scaling transformer context windows impacts feed-forward 3D reconstruction. Although recent object-centric feed-forward methods deliver robust, high-quality reconstruction, they still lag behind dense-view optimization in recovering fine-grained texture and appearance. We show that expanding the context window -- by substantially increasing the number of active object and image tokens -- remarkably narrows this gap and enables high-fidelity 3D object reconstruction and inverse rendering. To scale effectively, we adapt native sparse attention in our architecture design, unlocking its capacity for 3D reconstruction with three key contributions: (1) an efficient coarse-to-fine pipeline that focuses computation on informative regions by predicting sparse high-resolution residuals; (2) a 3D-aware spatial routing mechanism that establishes accurate 2D-3D correspondences using explicit geometric distances rather than standard attention scores; and (3) a custom block-aware sequence parallelism strategy utilizing an All-gather-KV protocol to balance dynamic, sparse workloads across GPUs. As a result, LSRM handles 20x more object tokens and >2x more image tokens than prior state-of-the-art (SOTA) methods. Extensive evaluations on standard novel-view synthesis benchmarks show substantial gains over the current SOTA, yielding 2.5 dB higher PSNR and 40% lower LPIPS. Furthermore, when extending LSRM to inverse rendering tasks, qualitative and quantitative evaluations on widely-used benchmarks demonstrate consistent improvements in texture and geometry details, achieving an LPIPS that matches or exceeds that of SOTA dense-view optimization methods. Code and model will be released on our project page.
>
---
#### [new 085] Reading Between the Pixels: An Inscriptive Jailbreak Attack on Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，解决T2I模型被用于生成含恶意文本的图像问题。提出Etch框架，通过分层攻击策略提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2604.05853](https://arxiv.org/pdf/2604.05853)**

> **作者:** Zonghao Ying; Haowen Dai; Lianyu Hu; Zonglei Jing; Quanchen Zou; Yaodong Yang; Aishan Liu; Xianglong Liu
>
> **摘要:** Modern text-to-image (T2I) models can now render legible, paragraph-length text, enabling a fundamentally new class of misuse. We identify and formalize the inscriptive jailbreak, where an adversary coerces a T2I system into generating images containing harmful textual payloads (e.g., fraudulent documents) embedded within visually benign scenes. Unlike traditional depictive jailbreaks that elicit visually objectionable imagery, inscriptive attacks weaponize the text-rendering capability itself. Because existing jailbreak techniques are designed for coarse visual manipulation, they struggle to bypass multi-stage safety filters while maintaining character-level fidelity. To expose this vulnerability, we propose Etch, a black-box attack framework that decomposes the adversarial prompt into three functionally orthogonal layers: semantic camouflage, visual-spatial anchoring, and typographic encoding. This decomposition reduces joint optimization over the full prompt space to tractable sub-problems, which are iteratively refined through a zero-order loop. In this process, a vision-language model critiques each generated image, localizes failures to specific layers, and prescribes targeted revisions. Extensive evaluations across 7 models on the 2 benchmarks demonstrate that Etch achieves an average attack success rate of 65.57% (peaking at 91.00%), significantly outperforming existing baselines. Our results reveal a critical blind spot in current T2I safety alignments and underscore the urgent need for typography-aware defense multimodal mechanisms.
>
---
#### [new 086] Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BADAS-2.0，解决自动驾驶中碰撞预测问题，通过改进数据集、模型压缩和可解释性，提升系统性能与实时性。**

- **链接: [https://arxiv.org/pdf/2604.05767](https://arxiv.org/pdf/2604.05767)**

> **作者:** Roni Goldshmidt; Hamish Scott; Lorenzo Niccolini; Hernan Matzner
>
> **摘要:** We present BADAS-2.0, the second generation of our collision anticipation system, building on BADAS-1.0 [7], which showed that fine-tuning V-JEPA2 [1] on large-scale ego-centric dashcam data outperforms both academic baselines and production ADAS systems. BADAS-2.0 advances the state of the art along three axes. (i) Long-tail benchmark and accuracy: We introduce a 10-group long-tail benchmark targeting rare and safety-critical scenarios. To construct it, BADAS-1.0 is used as an active oracle to score millions of unlabeled drives and surface high-risk candidates for annotation. Combined with Nexar's Atlas platform [13] for targeted data collection, this expands the dataset from 40k to 178,500 labeled videos (~2M clips), yielding consistent gains across all subgroups, with the largest improvements on the hardest long-tail cases. (ii) Knowledge distillation to edge: Domain-specific self-supervised pre-training on 2.25M unlabeled driving videos enables distillation into compact models, BADAS-2.0-Flash (86M) and BADAS-2.0-Flash-Lite (22M), achieving 7-12x speedup with near-parity accuracy, enabling real-time edge deployment. (iii) Explainability: BADAS-2.0 produces real-time object-centric attention heatmaps that localize the evidence behind predictions. BADAS-Reason [17] extends this with a vision-language model that consumes the last frame and heatmap to generate driver actions and structured textual reasoning. Inference code and evaluation benchmarks are publicly available.
>
---
#### [new 087] LUMOS: Universal Semi-Supervised OCT Retinal Layer Segmentation with Hierarchical Reliable Mutual Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决OCT图像标注稀缺和标签粒度不一致的问题。提出LUMOS框架，结合双解码器和多粒度学习，提升模型在不同粒度和领域下的分割性能。**

- **链接: [https://arxiv.org/pdf/2604.05388](https://arxiv.org/pdf/2604.05388)**

> **作者:** Yizhou Fang; Jian Zhong; Li Lin; Xiaoying Tang
>
> **备注:** 5 pages, 2 figures. Accepted to IEEE ISBI 2026. \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Optical Coherence Tomography (OCT) layer segmentation faces challenges due to annotation scarcity and heterogeneous label granularities across datasets. While semi-supervised learning helps alleviate label scarcity, existing methods typically assume a fixed granularity, failing to fully exploit cross-granularity supervision. This paper presents LUMOS, a semi-supervised universal OCT retinal layer segmentation framework based on a Dual-Decoder Network with a Hierarchical Prompting Strategy (DDN-HPS) and Reliable Progressive Multi-granularity Learning (RPML). DDN-HPS combines a dual-branch architecture with a multi-granularity prompting strategy to effectively suppress pseudo-label noise propagation. Meanwhile, RPML introduces region-level reliability weighing and a progressive training approach that guides the model from easier to more difficult tasks, ensuring the reliable selection of cross-granularity consistency targets, thereby achieving stable cross-granularity alignment. Experiments on six OCT datasets demonstrate that LUMOS largely outperforms existing methods and exhibits exceptional cross-domain and cross-granularity generalization capability.
>
---
#### [new 088] Physics-Aligned Spectral Mamba: Decoupling Semantics and Dynamics for Few-Shot Hyperspectral Target Detection
- **分类: cs.CV**

- **简介: 该论文属于少样本高光谱目标检测任务，旨在解决模型适应性差和跨域泛化能力弱的问题。提出SpecMamba框架，通过解耦语义与光谱适应提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.05562](https://arxiv.org/pdf/2604.05562)**

> **作者:** Luqi Gong; Qixin Xie; Yue Chen; Ziqiang Chen; Fanda Fan; Shuai Zhao; Chao Li
>
> **摘要:** Meta-learning facilitates few-shot hyperspectral target detection (HTD), but adapting deep backbones remains challenging. Full-parameter fine-tuning is inefficient and prone to overfitting, and existing methods largely ignore the frequency-domain structure and spectral band continuity of hyperspectral data, limiting spectral adaptation and cross-domain this http URL address these challenges, we propose SpecMamba, a parameter-efficient and frequency-aware framework that decouples stable semantic representation from agile spectral adaptation. Specifically, we introduce a Discrete Cosine Transform Mamba Adapter (DCTMA) on top of frozen Transformer representations. By projecting spectral features into the frequency domain via DCT and leveraging Mamba's linear-complexity state-space recursion, DCTMA explicitly captures global spectral dependencies and band continuity while avoiding the redundancy of full fine-tuning. Furthermore, to address prototype drift caused by limited sample sizes, we design a Prior-Guided Tri-Encoder (PGTE) that allows laboratory spectral priors to guide the optimization of the learnable adapter without disrupting the stable semantic feature space. Finally, a Self-Supervised Pseudo-Label Mapping (SSPLM) strategy is developed for test-time adaptation, enabling efficient decision boundary refinement through uncertainty-aware sampling and dual-path consistency constraints. Extensive experiments on multiple public datasets demonstrate that SpecMamba consistently outperforms state-of-the-art methods in detection accuracy and cross-domain generalization.
>
---
#### [new 089] FoleyDesigner: Immersive Stereo Foley Generation with Precise Spatio-Temporal Alignment for Film Clips
- **分类: cs.CV**

- **简介: 该论文提出FoleyDesigner，解决电影音效时空对齐问题，通过多智能体架构和扩散模型生成高质量立体音效，支持专业制作流程。**

- **链接: [https://arxiv.org/pdf/2604.05731](https://arxiv.org/pdf/2604.05731)**

> **作者:** Mengtian Li; Kunyan Dai; Yi Ding; Ruobing Ni; Ying Zhang; Wenwu Wang; Zhifeng Xie
>
> **摘要:** Foley art plays a pivotal role in enhancing immersive auditory experiences in film, yet manual creation of spatio-temporally aligned audio remains labor-intensive. We propose FoleyDesigner, a novel framework inspired by professional Foley workflows, integrating film clip analysis, spatio-temporally controllable Foley generation, and professional audio mixing capabilities. FoleyDesigner employs a multi-agent architecture for precise spatio-temporal analysis. It achieves spatio-temporal alignment through latent diffusion models trained on spatio-temporal cues extracted from video frames, combined with large language model (LLM)-driven hybrid mechanisms that emulate post-production practices in film industry. To address the lack of high-quality stereo audio datasets in film, we introduce FilmStereo, the first professional stereo audio dataset containing spatial metadata, precise timestamps, and semantic annotations for eight common Foley categories. For applications, the framework supports interactive user control while maintaining seamless integration with professional pipelines, including 5.1-channel Dolby Atmos systems compliant with ITU-R BS.775 standards, thereby offering extensive creative flexibility. Extensive experiments demonstrate that our method achieves superior spatio-temporal alignment compared to existing baselines, with seamless compatibility with professional film production standards. The project page is available at this https URL .
>
---
#### [new 090] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion
- **分类: cs.CV**

- **简介: 该论文属于3D语义场景补全任务，解决 voxel 分布不均和语义不平衡问题。提出VoxSAMNet，通过DSFR和Foreground Modulation策略提升性能。**

- **链接: [https://arxiv.org/pdf/2604.05780](https://arxiv.org/pdf/2604.05780)**

> **作者:** Yu Xue; Longjun Gao; Yuanqi Su; HaoAng Lu; Xiaoning Zhang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Monocular Semantic Scene Completion (SSC) aims to reconstruct complete 3D semantic scenes from a single RGB image, offering a cost-effective solution for autonomous driving and robotics. However, the inherently imbalanced nature of voxel distributions, where over 93% of voxels are empty and foreground classes are rare, poses significant challenges. Existing methods often suffer from redundant emphasis on uninformative voxels and poor generalization to long-tailed categories. To address these issues, we propose VoxSAMNet (Voxel Sparsity-Aware Modulation Network), a unified framework that explicitly models voxel sparsity and semantic imbalance. Our approach introduces: (1) a Dummy Shortcut for Feature Refinement (DSFR) module that bypasses empty voxels via a shared dummy node while refining occupied ones with deformable attention; and (2) a Foreground Modulation Strategy combining Foreground Dropout (FD) and Text-Guided Image Filter (TGIF) to alleviate overfitting and enhance class-relevant features. Extensive experiments on the public benchmarks SemanticKITTI and SSCBench-KITTI-360 demonstrate that VoxSAMNet achieves state-of-the-art performance, surpassing prior monocular and stereo baselines with mIoU scores of 18.2% and 20.2%, respectively. Our results highlight the importance of sparsity-aware and semantics-guided design for efficient and accurate 3D scene completion, offering a promising direction for future research.
>
---
#### [new 091] From Measurement to Mitigation: Quantifying and Reducing Identity Leakage in Image Representation Encoders with Linear Subspace Removal
- **分类: cs.CV**

- **简介: 该论文属于视觉隐私保护任务，解决图像编码器中的身份泄露问题。通过构建基准和提出线性子空间移除方法，实现隐私保护与实用性的平衡。**

- **链接: [https://arxiv.org/pdf/2604.05296](https://arxiv.org/pdf/2604.05296)**

> **作者:** Daniel George; Charles Yeh; Daniel Lee; Yifei Zhang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Frozen visual embeddings (e.g., CLIP, DINOv2/v3, SSCD) power retrieval and integrity systems, yet their use on face-containing data is constrained by unmeasured identity leakage and a lack of deployable mitigations. We take an attacker-aware view and contribute: (i) a benchmark of visual embeddings that reports open-set verification at low false-accept rates, a calibrated diffusion-based template inversion check, and face-context attribution with equal-area perturbations; and (ii) propose a one-shot linear projector that removes an estimated identity subspace while preserving the complementary space needed for utility, which for brevity we denote as the identity sanitization projection ISP. Across CelebA-20 and VGGFace2, we show that these encoders are robust under open-set linear probes, with CLIP exhibiting relatively higher leakage than DINOv2/v3 and SSCD, robust to template inversion, and are context-dominant. In addition, we show that ISP drives linear access to near-chance while retaining high non-biometric utility, and transfers across datasets with minor degradation. Our results establish the first attacker-calibrated facial privacy audit of non-FR encoders and demonstrate that linear subspace removal achieves strong privacy guarantees while preserving utility for visual search and retrieval.
>
---
#### [new 092] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP
- **分类: cs.CV**

- **简介: 该论文属于视觉模型解释任务，旨在解决CLIP特征可解释性问题。通过引入信息范围概念，区分局部与全局特征，提升对SAE特征的理解。**

- **链接: [https://arxiv.org/pdf/2604.05724](https://arxiv.org/pdf/2604.05724)**

> **作者:** Yusung Ro; Jaehyun Choi; Junmo Kim
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Sparse Autoencoders (SAEs) have emerged as a powerful tool for interpreting the internal representations of CLIP vision encoders, yet existing analyses largely focus on the semantic meaning of individual features. We introduce information scope as a complementary dimension of interpretability that characterizes how broadly an SAE feature aggregates visual evidence, ranging from localized, patch-specific cues to global, image-level signals. We observe that some SAE features respond consistently across spatial perturbations, while others shift unpredictably with minor input changes, indicating a fundamental distinction in their underlying scope. To quantify this, we propose the Contextual Dependency Score (CDS), which separates positionally stable local scope features from positionally variant global scope features. Our experiments show that features of different information scopes exert systematically different influences on CLIP's predictions and confidence. These findings establish information scope as a critical new axis for understanding CLIP representations and provide a deeper diagnostic view of SAE-derived features.
>
---
#### [new 093] Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于科学图形程序生成任务，解决数据质量与评估标准不足的问题，提出新框架和强化学习方法，提升TikZ代码生成效果。**

- **链接: [https://arxiv.org/pdf/2604.06079](https://arxiv.org/pdf/2604.06079)**

> **作者:** Juekai Lin; Yun Zhu; Honglin Lin; Sijing Li; Tianwei Lin; Zheng Liu; Xiaoyang Wang; Wenqiao Zhang; Lijun Wu
>
> **摘要:** Graphics Program Synthesis is pivotal for interpreting and editing visual data, effectively facilitating the reverse-engineering of static visuals into editable TikZ code. While TikZ is the de facto standard for scientific schematics due to its programmatic flexibility, its requirement for rigorous spatial precision presents a significant challenge for Multimodal Large Language Models. Progress is currently stifled by two primary gaps: (1) Data Quality Gap: existing image-TikZ corpora often lack strict executability and reliable visual alignment; (2) Evaluation Gap: a lack of benchmarks for both structural and visual fidelity. To address these, we present a closed-loop framework featuring: SciTikZ-230K, a large-scale, high-quality dataset from our Execution-Centric Data Engine covering 11 diverse scientific disciplines; SciTikZ-Bench, a multifaceted benchmark spanning from basic geometric constructs to intricate hierarchical schematics to evaluate both visual fidelity and structural logic. To further broaden the scope of visual-code optimization methodology, we introduce a novel Dual Self-Consistency Reinforcement Learning optimization paradigm, which utilizes Round-Trip Verification to penalize degenerate code and boost overall self-consistency. Empowered by these, our trained model SciTikZer-8B achieves state-of-the-art performance, consistently outperforming proprietary giants like Gemini-2.5-Pro and massive models like Qwen3-VL-235B-A22B-Instruct.
>
---
#### [new 094] Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉无监督强化学习任务，解决SR方法在高维环境中的泛化问题。提出SRCP框架，通过注意力机制和一致性策略提升表示质量和技能控制。**

- **链接: [https://arxiv.org/pdf/2604.05931](https://arxiv.org/pdf/2604.05931)**

> **作者:** Jingbo Sun; Qichao Zhang; Songjun Tu; Xing Fang; Yupeng Zheng; Haoran Li; Ke Chen; Dongbin Zhao
>
> **摘要:** Zero-shot unsupervised reinforcement learning (URL) offers a promising direction for building generalist agents capable of generalizing to unseen tasks without additional supervision. Among existing approaches, successor representations (SR) have emerged as a prominent paradigm due to their effectiveness in structured, low-dimensional settings. However, SR methods struggle to scale to high-dimensional visual environments. Through empirical analysis, we identify two key limitations of SR in visual URL: (1) SR objectives often lead to suboptimal representations that attend to dynamics-irrelevant regions, resulting in inaccurate successor measures and degraded task generalization; and (2) these flawed representations hinder SR policies from modeling multi-modal skill-conditioned action distributions and ensuring skill controllability. To address these limitations, we propose Saliency-Guided Representation with Consistency Policy Learning (SRCP), a novel framework that improves zero-shot generalization of SR methods in visual URL. SRCP decouples representation learning from successor training by introducing a saliency-guided dynamics task to capture dynamics-relevant representations, thereby improving successor measure and task generalization. Moreover, it integrates a fast-sampling consistency policy with URL-specific classifier-free guidance and tailored training objectives to improve skill-conditioned policy modeling and controllability. Extensive experiments on 16 tasks across 4 datasets from the ExORL benchmark demonstrate that SRCP achieves state-of-the-art zero-shot generalization in visual URL and is compatible with various SR methods.
>
---
#### [new 095] PanopticQuery: Unified Query-Time Reasoning for 4D Scenes
- **分类: cs.CV**

- **简介: 该论文提出PanopticQuery，解决4D动态场景中自然语言查询的语义一致性问题。通过多视角语义共识和神经场优化，实现精准的4D语义 grounding。**

- **链接: [https://arxiv.org/pdf/2604.05638](https://arxiv.org/pdf/2604.05638)**

> **作者:** Ruilin Tang; Yang Zhou; Zhong Ye; Wenxi Liu; Yan Huang; Shengfeng He
>
> **摘要:** Understanding dynamic 4D environments through natural language queries requires not only accurate scene reconstruction but also robust semantic grounding across space, time, and viewpoints. While recent methods using neural representations have advanced 4D reconstruction, they remain limited in contextual reasoning, especially for complex semantics such as interactions, temporal actions, and spatial relations. A key challenge lies in transforming noisy, view-dependent predictions into globally consistent 4D interpretations. We introduce PanopticQuery, a framework for unified query-time reasoning in 4D scenes. Our approach builds on 4D Gaussian Splatting for high-fidelity dynamic reconstruction and introduces a multi-view semantic consensus mechanism that grounds natural language queries by aggregating 2D semantic predictions across multiple views and time frames. This process filters inconsistent outputs, enforces geometric consistency, and lifts 2D semantics into structured 4D groundings via neural field optimization. To support evaluation, we present Panoptic-L4D, a new benchmark for language-based querying in dynamic scenes. Experiments demonstrate that PanopticQuery sets a new state of the art on complex language queries, effectively handling attributes, actions, spatial relationships, and multi-object interactions. A video demonstration is available in the supplementary materials.
>
---
#### [new 096] Not All Agents Matter: From Global Attention Dilution to Risk-Prioritized Game Planning
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决多智能体交互中的风险优先决策问题。提出GameAD框架，通过风险感知博弈实现更安全的路径规划。**

- **链接: [https://arxiv.org/pdf/2604.05449](https://arxiv.org/pdf/2604.05449)**

> **作者:** Kang Ding; Hongsong Wang; Jie Gui; Lei He
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** End-to-end autonomous driving resides not in the integration of perception and planning, but rather in the dynamic multi-agent game within a unified representation space. Most existing end-to-end models treat all agents equally, hindering the decoupling of real collision threats from complex backgrounds. To address this issue, We introduce the concept of Risk-Prioritized Game Planning, and propose GameAD, a novel framework that models end-to-end autonomous driving as a risk-aware game problem. The GameAD integrates Risk-Aware Topology Anchoring, Strategic Payload Adapter, Minimax Risk-Aware Sparse Attention, and Risk Consistent Equilibrium Stabilization to enable game theoretic decision making with risk prioritized interactions. We also present the Planning Risk Exposure metric, which quantifies the cumulative risk intensity of planned trajectories over a long horizon for safe autonomous driving. Extensive experiments on the nuScenes and Bench2Drive datasets show that our approach significantly outperforms state-of-the-art methods, especially in terms of trajectory safety.
>
---
#### [new 097] Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，针对CLIP模型的中心偏差问题，分析其成因并提出无需训练的缓解方法。**

- **链接: [https://arxiv.org/pdf/2604.05971](https://arxiv.org/pdf/2604.05971)**

> **作者:** Oscar Chew; Hsiao-Ying Huang; Kunal Jain; Tai-I Chen; Khoa D Doan; Kuan-Hao Huang
>
> **摘要:** Recent research has shown that contrastive vision-language models such as CLIP often lack fine-grained understanding of visual content. While a growing body of work has sought to address this limitation, we identify a distinct failure mode in the CLIP family, which we term center bias, that persists even in recent model variants. Specifically, CLIP tends to disproportionately focus on the central region of an image, overlooking important objects located near the boundaries. This limitation is fundamental as failure to recognize relevant objects makes it difficult to perform any sophisticated tasks that depend on those objects. To understand the underlying causes of the limitation, we conduct analyses from both representation and attention perspectives. Using interpretability methods, i.e., embedding decomposition and attention map analysis, we find that relevant concepts especially those associated with off-center objects vanish from the model's embedding in the final representation due to information loss during the aggregation of visual embeddings, particularly the reliance on pooling mechanisms. Finally, we show that this bias can be alleviated with training-free strategies such as visual prompting and attention redistribution by redirecting models' attention to off-center regions.
>
---
#### [new 098] A Weak-Signal-Aware Framework for Subsurface Defect Detection: Mechanisms for Enhancing Low-SCR Hyperbolic Signatures
- **分类: cs.CV**

- **简介: 该论文属于地下缺陷检测任务，解决弱信号识别难题。提出WSA-Net框架，通过信号增强和杂波抑制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.05490](https://arxiv.org/pdf/2604.05490)**

> **作者:** Wenbo Zhang; Zekun Long; Zican Liu; Yangchen Zeng; Keyi Hu
>
> **备注:** 8 pages, 7 figures, 5 tables. Accepted by International Joint Conference on Neural Networks (IJCNN)
>
> **摘要:** Subsurface defect detection via Ground Penetrating Radar is challenged by "weak signals" faint diffraction hyperbolas with low signal-to-clutter ratios, high wavefield similarity, and geometric degradation. Existing lightweight detectors prioritize efficiency over sensitivity, failing to preserve low-frequency structures or decouple heterogeneous clutter. We propose WSA-Net, a framework designed to enhance faint signatures through physical-feature reconstruction. Moving beyond simple parameter reduction, WSA-Net integrates four mechanisms: Signal preservation using partial convolutions; Clutter suppression via heterogeneous grouping attention; Geometric reconstruction to sharpen hyperbolic arcs; Context anchoring to resolve semantic ambiguities. Evaluations on the RTSTdataset show WSA-Net achieves 0.6958 mAP@0.5 and 164 FPS with only 2.412 M parameters. Results prove that signal-centric awareness in lightweight architectures effectively reduces false negatives in infrastructure inspection.
>
---
#### [new 099] Generative AI for Video Trailer Synthesis: From Extractive Heuristics to Autoregressive Creativity
- **分类: cs.CV; cs.AI; cs.HC; cs.IR; cs.MM**

- **简介: 该论文属于视频预告片生成任务，旨在解决传统提取方法的局限性，通过生成式AI实现更智能的叙事构建。工作包括技术演进分析与伦理探讨。**

- **链接: [https://arxiv.org/pdf/2604.04953](https://arxiv.org/pdf/2604.04953)**

> **作者:** Abhishek Dharmaratnakar; Srivaths Ranganathan; Debanshu Das; Anushree Sinha
>
> **备注:** 7 pages, 3 figures, accepted in WSDM 2026
>
> **摘要:** The domain of automatic video trailer generation is currently undergoing a profound paradigm shift, transitioning from heuristic-based extraction methods to deep generative synthesis. While early methodologies relied heavily on low-level feature engineering, visual saliency, and rule-based heuristics to select representative shots, recent advancements in Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), and diffusion-based video synthesis have enabled systems that not only identify key moments but also construct coherent, emotionally resonant narratives. This survey provides a comprehensive technical review of this evolution, with a specific focus on generative techniques including autoregressive Transformers, LLM-orchestrated pipelines, and text-to-video foundation models like OpenAI's Sora and Google's Veo. We analyze the architectural progression from Graph Convolutional Networks (GCNs) to Trailer Generation Transformers (TGT), evaluate the economic implications of automated content velocity on User-Generated Content (UGC) platforms, and discuss the ethical challenges posed by high-fidelity neural synthesis. By synthesizing insights from recent literature, this report establishes a new taxonomy for AI-driven trailer generation in the era of foundation models, suggesting that future promotional video systems will move beyond extractive selection toward controllable generative editing and semantic reconstruction of trailers.
>
---
#### [new 100] OrthoFuse: Training-free Riemannian Fusion of Orthogonal Style-Concept Adapters for Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于模型微调任务，解决如何无训练地融合不同任务的正交适配器问题。通过几何方法和谱恢复技术，实现风格与概念特征的有效融合。**

- **链接: [https://arxiv.org/pdf/2604.05183](https://arxiv.org/pdf/2604.05183)**

> **作者:** Ali Aliev; Kamil Garifullin; Nikolay Yudin; Vera Soboleva; Alexander Molozhavenko; Ivan Oseledets; Aibek Alanov; Maxim Rakhuba
>
> **摘要:** In a rapidly growing field of model training there is a constant practical interest in parameter-efficient fine-tuning and various techniques that use a small amount of training data to adapt the model to a narrow task. However, there is an open question: how to combine several adapters tuned for different tasks into one which is able to yield adequate results on both tasks? Specifically, merging subject and style adapters for generative models remains unresolved. In this paper we seek to show that in the case of orthogonal fine-tuning (OFT), we can use structured orthogonal parametrization and its geometric properties to get the formulas for training-free adapter merging. In particular, we derive the structure of the manifold formed by the recently proposed Group-and-Shuffle ($\mathcal{GS}$) orthogonal matrices, and obtain efficient formulas for the geodesics approximation between two points. Additionally, we propose a $\text{spectra restoration}$ transform that restores spectral properties of the merged adapter for higher-quality fusion. We conduct experiments in subject-driven generation tasks showing that our technique to merge two $\mathcal{GS}$ orthogonal matrices is capable of uniting concept and style features of different adapters. To the best of our knowledge, this is the first training-free method for merging multiplicative orthogonal adapters. Code is available via the $\href{this https URL}{link}$.
>
---
#### [new 101] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration
- **分类: cs.CV**

- **简介: 该论文属于视频问答任务，旨在解决传统方法依赖帧定位而非故事线推理的问题。提出SVAgent框架，通过多智能体协作实现基于故事线的视频理解。**

- **链接: [https://arxiv.org/pdf/2604.05079](https://arxiv.org/pdf/2604.05079)**

> **作者:** Zhongyu Yang; Zuhao Yang; Shuo Zhan; Tan Yue; Wei Pang; Yingfang Yuan
>
> **备注:** Published in CVPR2026
>
> **摘要:** Video question answering (VideoQA) is a challenging task that requires integrating spatial, temporal, and semantic information to capture the complex dynamics of video sequences. Although recent advances have introduced various approaches for video understanding, most existing methods still rely on locating relevant frames to answer questions rather than reasoning through the evolving storyline as humans do. Humans naturally interpret videos through coherent storylines, an ability that is crucial for making robust and contextually grounded predictions. To address this gap, we propose SVAgent, a storyline-guided cross-modal multi-agent framework for VideoQA. The storyline agent progressively constructs a narrative representation based on frames suggested by a refinement suggestion agent that analyzes historical failures. In addition, cross-modal decision agents independently predict answers from visual and textual modalities under the guidance of the evolving storyline. Their outputs are then evaluated by a meta-agent to align cross-modal predictions and enhance reasoning robustness and answer consistency. Experimental results demonstrate that SVAgent achieves superior performance and interpretability by emulating human-like storyline reasoning in video understanding.
>
---
#### [new 102] RCP: Representation Consistency Pruner for Mitigating Distribution Shift in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型压缩任务，旨在解决分布偏移问题。通过引入RCP框架，实现高效剪枝，减少计算量并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.04972](https://arxiv.org/pdf/2604.04972)**

> **作者:** Jianwei Zhang; Chaoning Zhang; Sihan Cao; Wang Liu; Pengcheng Zheng; Jiaxin Huang; Caiyan Qin; Yalan Ye; Wei Dong; Yang Yang
>
> **摘要:** Large Vision-Language Models (LVLMs) suffer from prohibitive inference costs due to the massive number of visual tokens processed by the language decoder. Existing pruning methods often lead to significant performance degradation because the irreversible removal of visual tokens causes a distribution shift in the hidden states that deviates from the pre-trained full-token regime. To address this, we propose Representation Consistency Pruner, which we refer to as RCP, as a novel framework that integrates cumulative visual token pruning with a delayed repair mechanism. Specifically, we introduce a cross-attention pruner that leverages the intrinsic attention of the LLM as a baseline to predict cumulative masks, ensuring consistent and monotonic token reduction across layers. To compensate for the resulting information loss, we design a delayed repair adapter denoted as DRA, which caches the essence of pruned tokens and applies FiLM-based modulation specifically to the answer generation tokens. We employ a repair loss to match the first and second-order statistics of the pruned representations with a full-token teacher. RCP is highly efficient because it trains only lightweight plug-in modules while allowing for physical token discarding at inference. Extensive experiments on LVLM benchmarks demonstrate that RCP removes up to 88.9\% of visual tokens and reduces FLOPs by up to 85.7\% with only a marginal average accuracy drop, and outperforms prior methods that avoid fine-tuning the original model on several widely used benchmarks.
>
---
#### [new 103] Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决可控生成中训练慢、效果差的问题。通过引入$x_0$-监督，加速收敛并提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.05761](https://arxiv.org/pdf/2604.05761)**

> **作者:** Amadou S. Sangare; Adrien Maglo; Mohamed Chaouch; Bertrand Luvison
>
> **摘要:** Text-to-Image (T2I) diffusion/flow models have recently achieved remarkable progress in visual fidelity and text alignment. However, they remain limited when users need to precisely control image layouts, something that natural language alone cannot reliably express. Controllable generation methods augment the initial T2I model with additional conditions that more easily describe the scene. Prior works straightforwardly train the augmented network with the same loss as the initial network. Although natural at first glance, this can lead to very long training times in some cases before convergence. In this work, we revisit the training objective of controllable diffusion models through a detailed analysis of their denoising dynamics. We show that direct supervision on the clean target image, dubbed $x_0$-supervision, or an equivalent re-weighting of the diffusion loss, yields faster convergence. Experiments on multiple control settings demonstrate that our formulation accelerates convergence by up to 2$\times$ according to our novel metric (mean Area Under the Convergence Curve - mAUCC), while also improving both visual quality and conditioning accuracy. Our code is available at this https URL
>
---
#### [new 104] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出OACIR任务，解决组合图像检索中实例一致性问题。构建了OACIRR基准，并提出AdaFocal框架，提升实例级检索精度。**

- **链接: [https://arxiv.org/pdf/2604.05393](https://arxiv.org/pdf/2604.05393)**

> **作者:** Yuxin Yang; Yinan Zhou; Yuxin Chen; Ziqi Zhang; Zongyang Ma; Chunfeng Yuan; Bing Li; Jun Gao; Weiming Hu
>
> **备注:** Accepted to CVPR 2026. Project page, dataset, and code are available at: this https URL
>
> **摘要:** Composed Image Retrieval (CIR) has demonstrated significant potential by enabling flexible multimodal queries that combine a reference image and modification text. However, CIR inherently prioritizes semantic matching, struggling to reliably retrieve a user-specified instance across contexts. In practice, emphasizing concrete instance fidelity over broad semantics is often more consequential. In this work, we propose Object-Anchored Composed Image Retrieval (OACIR), a novel fine-grained retrieval task that mandates strict instance-level consistency. To advance research on this task, we construct OACIRR (OACIR on Real-world images), the first large-scale, multi-domain benchmark comprising over 160K quadruples and four challenging candidate galleries enriched with hard-negative instance distractors. Each quadruple augments the compositional query with a bounding box that visually anchors the object in the reference image, providing a precise and flexible way to ensure instance preservation. To address the OACIR task, we propose AdaFocal, a framework featuring a Context-Aware Attention Modulator that adaptively intensifies attention within the specified instance region, dynamically balancing focus between the anchored instance and the broader compositional context. Extensive experiments demonstrate that AdaFocal substantially outperforms existing compositional retrieval models, particularly in maintaining instance-level fidelity, thereby establishing a robust baseline for this challenging task while opening new directions for more flexible, instance-aware retrieval systems.
>
---
#### [new 105] Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于多遍历场景重建任务，解决不同时间拍摄场景的外观不一致问题。通过显式分解背景外观，提升重建一致性。**

- **链接: [https://arxiv.org/pdf/2604.05908](https://arxiv.org/pdf/2604.05908)**

> **作者:** Yangyi Xiao; Siting Zhu; Baoquan Yang; Tianchen Deng; Yongbo Chen; Hesheng Wang
>
> **摘要:** Multi-traversal scene reconstruction is important for high-fidelity autonomous driving simulation and digital twin construction. This task involves integrating multiple sequences captured from the same geographical area at different times. In this context, a primary challenge is the significant appearance inconsistency across traversals caused by varying illumination and environmental conditions, despite the shared underlying geometry. This paper presents ADM-GS (Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction), a framework that applies an explicit appearance decomposition to the static background to alleviate appearance entanglement across traversals. For the static background, we decompose the appearance into traversal-invariant material, representing intrinsic material properties, and traversal-dependent illumination, capturing lighting variations. Specifically, we propose a neural light field that utilizes a frequency-separated hybrid encoding strategy. By incorporating surface normals and explicit reflection vectors, this design separately captures low-frequency diffuse illumination and high-frequency specular reflections. Quantitative evaluations on the Argoverse 2 and Waymo Open datasets demonstrate the effectiveness of ADM-GS. In multi-traversal experiments, our method achieves a +0.98 dB PSNR improvement over existing latent-based baselines while producing more consistent appearance across traversals. Code will be available at this https URL.
>
---
#### [new 106] FunRec: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos
- **分类: cs.CV**

- **简介: 该论文提出FunRec，用于从第一视角视频重建功能3D场景。解决无约束环境下3D场景重建问题，自动识别关节部件、估计参数并生成可模拟的网格。**

- **链接: [https://arxiv.org/pdf/2604.05621](https://arxiv.org/pdf/2604.05621)**

> **作者:** Alexandros Delitzas; Chenyangguang Zhang; Alexey Gavryushin; Tommaso Di Mario; Boyang Sun; Rishabh Dabral; Leonidas Guibas; Christian Theobalt; Marc Pollefeys; Francis Engelmann; Daniel Barath
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** We present FunRec, a method for reconstructing functional 3D digital twins of indoor scenes directly from egocentric RGB-D interaction videos. Unlike existing methods on articulated reconstruction, which rely on controlled setups, multi-state captures, or CAD priors, FunRec operates directly on in-the-wild human interaction sequences to recover interactable 3D scenes. It automatically discovers articulated parts, estimates their kinematic parameters, tracks their 3D motion, and reconstructs static and moving geometry in canonical space, yielding simulation-compatible meshes. Across new real and simulated benchmarks, FunRec surpasses prior work by a large margin, achieving up to +50 mIoU improvement in part segmentation, 5-10 times lower articulation and pose errors, and significantly higher reconstruction accuracy. We further demonstrate applications on URDF/USD export for simulation, hand-guided affordance mapping and robot-scene interaction.
>
---
#### [new 107] MPM: Mutual Pair Merging for Efficient Vision Transformers
- **分类: cs.CV**

- **简介: 该论文针对语义分割任务，解决Transformer模型中序列长度过长导致的效率问题。提出MPM方法，在不引入参数的情况下，通过互近邻对合并提升推理速度，同时保持分割精度。**

- **链接: [https://arxiv.org/pdf/2604.05718](https://arxiv.org/pdf/2604.05718)**

> **作者:** Simon Ravé; Pejman Rasti; David Rousseau
>
> **备注:** Accepted to CVPR 2026 (Findings)
>
> **摘要:** Decreasing sequence length is a common way to accelerate transformers, but prior token reduction work often targets classification and reports proxy metrics rather than end-to-end latency. For semantic segmentation, token reduction is further constrained by the need to reconstruct dense, pixel-aligned features, and on modern accelerators the overhead of computing merge maps can erase expected gains. We propose Mutual Pair Merging (MPM), a training-free token aggregation module that forms mutual nearest-neighbor pairs in cosine space, averages each pair, and records a merge map enabling a gather-based reconstruction before the decoder so that existing segmentation heads can be used unchanged. MPM introduces no learned parameters and no continuous compression knob (no keep-rate or threshold). The speed-accuracy trade-off is set by a discrete insertion schedule. We benchmark end-to-end latency on an NVIDIA H100 GPU (with and without FlashAttention-2) and a Raspberry Pi 5 across standard segmentation datasets. On ADE20K, MPM reduces per-image latency by up to 60% for ViT-Tiny on Raspberry Pi 5, and increases throughput by up to 20% on H100 with FlashAttention-2 while keeping the mIoU drop below 3%. These results suggest that simple, reconstruction-aware, training-free token merging can translate into practical wall-clock gains for segmentation when overhead is explicitly accounted for.
>
---
#### [new 108] VLA-InfoEntropy: A Training-Free Vision-Attention Information Entropy Approach for Vision-Language-Action Models Inference Acceleration and Success
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的推理加速任务，旨在解决高计算开销和低效率问题。通过引入信息熵方法，提升推理速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.05323](https://arxiv.org/pdf/2604.05323)**

> **作者:** Chuhang Liu; Yayun He; Zuheng Kang; Xiaoyang Qu; Jianzong Wang
>
> **备注:** Accepted to the 2026 IEEE International Conference on Multimedia and Expo (ICME 2026)
>
> **摘要:** Vision-Language-Action (VLA) models integrate visual perception, language understanding, and action decision-making for cross-modal semantic alignment, exhibiting broad application potential. However, the joint processing of high-dimensional visual features, complex linguistic inputs, and continuous action sequences incurs significant computational overhead and low inference efficiency, thereby hindering real-time deployment and reliability. To address this issue, we use image entropy to quantify the grayscale distribution characteristics of each visual token and introduce attention entropy to capture the distribution of attention scores over task-related text. Visual entropy identifies texture-rich or structurally informative regions, while attention entropy pinpoints semantically relevant tokens. Combined with timestep information, these metrics enable a dynamic transition strategy that shifts the model's focus from global visual features to attention-guided local informative regions. Thus, the resulting VLA-InfoEntropy method integrates spatial, semantic, and temporal cues to reduce redundancy while preserving critical content. Extensive experiments show that our method reduces inference parameters, accelerates inference speed, and outperforms existing approaches.
>
---
#### [new 109] MMEmb-R1: Reasoning-Enhanced Multimodal Embedding with Pair-Aware Selection and Adaptive Control
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态嵌入任务，旨在解决生成式推理在嵌入学习中应用的挑战。提出MMEmb-R1框架，通过自适应推理选择提升效果并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2604.06156](https://arxiv.org/pdf/2604.06156)**

> **作者:** Yuchi Wang; Haiyang Yu; Weikang Bian; Jiefeng Long; Xiao Liang; Chao Feng; Hongsheng Li
>
> **摘要:** MLLMs have been successfully applied to multimodal embedding tasks, yet their generative reasoning capabilities remain underutilized. Directly incorporating chain-of-thought reasoning into embedding learning introduces two fundamental challenges. First, structural misalignment between instance-level reasoning and pairwise contrastive supervision may lead to shortcut behavior, where the model merely learns the superficial format of reasoning. Second, reasoning is not universally beneficial for embedding tasks. Enforcing reasoning for all inputs may introduce unnecessary computation and latency, and can even obscure salient semantic signals for simple cases. To address these issues, we propose MMEmb-R1, an adaptive reasoning-based multimodal embedding framework. We formulate reasoning as a latent variable and introduce pair-aware reasoning selection that employs counterfactual intervention to identify reasoning paths beneficial for query-target alignment. Furthermore, we adopt reinforcement learning to selectively invoke reasoning only when necessary. Experiments on the MMEB-V2 benchmark demonstrate that our model achieves a score of 71.2 with only 4B parameters, establishing a new state-of-the-art while significantly reducing reasoning overhead and inference latency.
>
---
#### [new 110] LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决大尺度无人机场景下3DGS定位的鲁棒性问题。提出LSGS-Loc方法，结合尺度感知初始化和可靠性掩码，提升定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.05402](https://arxiv.org/pdf/2604.05402)**

> **作者:** Xiang Zhang; Tengfei Wang; Fang Xu; Xin Wang; Zongqian Zhan
>
> **备注:** This paper is under reviewed by RA-L. The copyright might be transferred upon acceptance
>
> **摘要:** Visual localization in large-scale UAV scenarios is a critical capability for autonomous systems, yet it remains challenging due to geometric complexity and environmental variations. While 3D Gaussian Splatting (3DGS) has emerged as a promising scene representation, existing 3DGS-based visual localization methods struggle with robust pose initialization and sensitivity to rendering artifacts in large-scale settings. To address these limitations, we propose LSGS-Loc, a novel visual localization pipeline tailored for large-scale 3DGS scenes. Specifically, we introduce a scale-aware pose initialization strategy that combines scene-agnostic relative pose estimation with explicit 3DGS scale constraints, enabling geometrically grounded localization without scene-specific training. Furthermore, in the pose refinement, to mitigate the impact of reconstruction artifacts such as blur and floaters, we develop a Laplacian-based reliability masking mechanism that guides photometric refinement toward high-quality regions. Extensive experiments on large-scale UAV benchmarks demonstrate that our method achieves state-of-the-art accuracy and robustness for unordered image queries, significantly outperforming existing 3DGS-based approaches. Code is available at: this https URL
>
---
#### [new 111] Mixture-of-Modality-Experts with Holistic Token Learning for Fine-Grained Multimodal Visual Analytics in Driver Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态视觉分析任务，旨在解决驾驶员行为识别中模态依赖性和细粒度动作理解问题。提出MoME框架和HTL策略，提升多模态协同与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.05947](https://arxiv.org/pdf/2604.05947)**

> **作者:** Tianyi Liu; Yiming Li; Wenqian Wang; Jiaojiao Wang; Chen Cai; Yi Wang; Kim-Hui Yap
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Robust multimodal visual analytics remains challenging when heterogeneous modalities provide complementary but input-dependent evidence for this http URL multimodal learning methods mainly rely on fixed fusion modules or predefined cross-modal interactions, which are often insufficient to adapt to changing modality reliability and to capture fine-grained action cues. To address this issue, we propose a Mixture-of-Modality-Experts (MoME) framework with a Holistic Token Learning (HTL) strategy. MoME enables adaptive collaboration among modality-specific experts, while HTL improves both intra-expert refinement and inter-expert knowledge transfer through class tokens and spatio-temporal tokens. In this way, our method forms a knowledge-centric multimodal learning framework that improves expert specialization while reducing ambiguity in multimodal this http URL validate the proposed framework on driver action recognition as a representative multimodal understanding taskThe experimental results on the public benchmark show that the proposed MoME framework and the HTL strategy jointly outperform representative single-modal and multimodal baselines. Additional ablation, validation, and visualization results further verify that the proposed HTL strategy improves subtle multimodal understanding and offers better interpretability.
>
---
#### [new 112] Coverage Optimization for Camera View Selection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，解决相机视角选择问题。通过优化覆盖范围，提升重建质量，提出COVER方法，有效选择 informative 视角。**

- **链接: [https://arxiv.org/pdf/2604.05259](https://arxiv.org/pdf/2604.05259)**

> **作者:** Timothy Chen; Adam Dai; Maximilian Adang; Grace Gao; Mac Schwager
>
> **摘要:** What makes a good viewpoint? The quality of the data used to learn 3D reconstructions is crucial for enabling efficient and accurate scene modeling. We study the active view selection problem and develop a principled analysis that yields a simple and interpretable criterion for selecting informative camera poses. Our key insight is that informative views can be obtained by minimizing a tractable approximation of the Fisher Information Gain, which reduces to favoring viewpoints that cover geometry that has been insufficiently observed by past cameras. This leads to a lightweight coverage-based view selection metric that avoids expensive transmittance estimation and is robust to noise and training dynamics. We call this metric COVER (Camera Optimization for View Exploration and Reconstruction). We integrate our method into the Nerfstudio framework and evaluate it on real datasets within fixed and embodied data acquisition scenarios. Across multiple datasets and radiance-field baselines, our method consistently improves reconstruction quality compared to state-of-the-art active view selection methods. Additional visualizations and our Nerfstudio package can be found at this https URL.
>
---
#### [new 113] Modality-Aware and Anatomical Vector-Quantized Autoencoding for Multimodal Brain MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种多模态脑MRI重建方法，解决单模态VAE无法充分利用多模态信息的问题。通过引入模态感知和解剖结构编码，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2604.05171](https://arxiv.org/pdf/2604.05171)**

> **作者:** Mingjie Li; Edward Kim; Yue Zhao; Ehsan Adeli; Kilian M. Pohl
>
> **备注:** CVPR Fingdings track
>
> **摘要:** Learning a robust Variational Autoencoder (VAE) is a fundamental step for many deep learning applications in medical image analysis, such as MRI synthesizes. Existing brain VAEs predominantly focus on single-modality data (i.e., T1-weighted MRI), overlooking the complementary diagnostic value of other modalities like T2-weighted MRIs. Here, we propose a modality-aware and anatomically grounded 3D vector-quantized VAE (VQ-VAE) for reconstructing multi-modal brain MRIs. Called NeuroQuant, it first learns a shared latent representation across modalities using factorized multi-axis attention, which can capture relationships between distant brain regions. It then employs a dual-stream 3D encoder that explicitly separates the encoding of modality-invariant anatomical structures from modality-dependent appearance. Next, the anatomical encoding is discretized using a shared codebook and combined with modality-specific appearance features via Feature-wise Linear Modulation (FiLM) during the decoding phase. This entire approach is trained using a joint 2D/3D strategy in order to account for the slice-based acquisition of 3D MRI data. Extensive experiments on two multi-modal brain MRI datasets demonstrate that NeuroQuant achieves superior reconstruction fidelity compared to existing VAEs, enabling a scalable foundation for downstream generative modeling and cross-modal brain image analysis.
>
---
#### [new 114] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型在处理不明确提示时的隐式决策问题。通过定位自注意力层，提出ICM方法实现精准干预，提升去偏效果。**

- **链接: [https://arxiv.org/pdf/2604.06052](https://arxiv.org/pdf/2604.06052)**

> **作者:** Katarzyna Zaleska; Łukasz Popek; Monika Wysoczańska; Kamil Deja
>
> **备注:** CVPR 2026
>
> **摘要:** Text-to-image diffusion models exhibit remarkable generative capabilities, yet their internal operations remain opaque, particularly when handling prompts that are not fully descriptive. In such scenarios, models must make implicit decisions to generate details not explicitly specified in the text. This work investigates the hypothesis that this decision-making process is not diffuse but is computationally localized within the model's architecture. While existing localization techniques focus on prompt-related interventions, we notice that such explicit conditioning may differ from implicit decisions. Therefore, we introduce a probing-based localization technique to identify the layers with the highest attribute separability for concepts. Our findings indicate that the resolution of ambiguous concepts is governed principally by self-attention layers, identifying them as the most effective point for intervention. Based on this discovery, we propose ICM (Implicit Choice-Modification) - a precise steering method that applies targeted interventions to a small subset of layers. Extensive experiments confirm that intervening on these specific self-attention layers yields superior debiasing performance compared to existing state-of-the-art methods, minimizing artifacts common to less precise approaches. The code is available at this https URL.
>
---
#### [new 115] Weather-Conditioned Branch Routing for Robust LiDAR-Radar 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决恶劣天气下传感器可靠性变化导致的检测鲁棒性问题。通过引入天气条件引导的分支路由机制，动态调整LiDAR与4D雷达的融合策略，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.05405](https://arxiv.org/pdf/2604.05405)**

> **作者:** Hongsheng Li; Lingfeng Zhang; Zexian Yang; Liang Li; Rong Yin; Xiaoshuai Hao; Wenbo Ding
>
> **摘要:** Robust 3D object detection in adverse weather is highly challenging due to the varying reliability of different sensors. While existing LiDAR-4D radar fusion methods improve robustness, they predominantly rely on fixed or weakly adaptive pipelines, failing to dy-namically adjust modality preferences as environmental conditions change. To bridge this gap, we reformulate multi-modal perception as a weather-conditioned branch routing problem. Instead of computing a single fused output, our framework explicitly maintains three parallel 3D feature streams: a pure LiDAR branch, a pure 4D radar branch, and a condition-gated fusion branch. Guided by a condition token extracted from visual and semantic prompts, a lightweight router dynamically predicts sample-specific weights to softly aggregate these representations. Furthermore, to prevent branch collapse, we introduce a weather-supervised learning strategy with auxiliary classification and diversity regularization to enforce distinct, condition-dependent routing behaviors. Extensive experiments on the K-Radar benchmark demonstrate that our method achieves state-of-the-art performance. Furthermore, it provides explicit and highly interpretable insights into modality preferences, transparently revealing how adaptive routing robustly shifts reliance between LiDAR and 4D radar across diverse adverse-weather scenarios. The source code with be released.
>
---
#### [new 116] Lightweight Multimodal Adaptation of Vision Language Models for Species Recognition and Habitat Context Interpretation in Drone Thermal Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于物种识别与生境分析任务，解决RGB预训练视觉语言模型在热成像中的适应问题。通过轻量级多模态适配，提升模型在热红外图像上的性能，并生成生境信息。**

- **链接: [https://arxiv.org/pdf/2604.06124](https://arxiv.org/pdf/2604.06124)**

> **作者:** Hao Chen; Fang Qiu; Fangchao Dong; Defei Yang; Eve Bohnett; Li An
>
> **摘要:** This study proposes a lightweight multimodal adaptation framework to bridge the representation gap between RGB-pretrained VLMs and thermal infrared imagery, and demonstrates its practical utility using a real drone-collected dataset. A thermal dataset was developed from drone-collected imagery and was used to fine-tune VLMs through multimodal projector alignment, enabling the transfer of information from RGB-based visual representations to thermal radiometric inputs. Three representative models, including InternVL3-8B-Instruct, Qwen2.5-VL-7B-Instruct, and Qwen3-VL-8B-Instruct, were benchmarked under both closed-set and open-set prompting conditions for species recognition and instance enumeration. Among the tested models, Qwen3-VL-8B-Instruct with open-set prompting achieved the best overall performance, with F1 scores of 0.935 for deer, 0.915 for rhino, and 0.968 for elephant, and within-1 enumeration accuracies of 0.779, 0.982, and 1.000, respectively. In addition, combining thermal imagery with simultaneously collected RGB imagery enabled the model to generate habitat-context information, including land-cover characteristics, key landscape features, and visible human disturbance. Overall, the findings demonstrate that lightweight projector-based adaptation provides an effective and practical route for transferring RGB-pretrained VLMs to thermal drone imagery, expanding their utility from object-level recognition to habitat-context interpretation in ecological monitoring.
>
---
#### [new 117] Hierarchical Mesh Transformers with Topology-Guided Pretraining for Morphometric Analysis of Brain Structures
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于神经影像分析任务，旨在解决多模态脑结构形态学特征学习难题。提出一种分层Transformer框架，实现跨拓扑的高效特征提取与整合。**

- **链接: [https://arxiv.org/pdf/2604.05215](https://arxiv.org/pdf/2604.05215)**

> **作者:** Yujian Xiong; Mohammad Farazi; Yanxi Chen; Wenhui Zhu; Xuanzhao Dong; Natasha Lepore; Yi Su; Raza Mushtaq; Stephen Foldes; Andrew Yang; Yalin Wang
>
> **摘要:** Representation learning on large-scale unstructured volumetric and surface meshes poses significant challenges in neuroimaging, especially when models must incorporate diverse vertex-level morphometric descriptors, such as cortical thickness, curvature, sulcal depth, and myelin content, which carry subtle disease-related signals. Current approaches either ignore these clinically informative features or support only a single mesh topology, restricting their use across imaging pipelines. We introduce a hierarchical transformer framework designed for heterogeneous mesh analysis that operates on spatially adaptive tree partitions constructed from simplicial complexes of arbitrary order. This design accommodates both volumetric and surface discretizations within a single architecture, enabling efficient multi-scale attention without topology-specific modifications. A feature projection module maps variable-length per-vertex clinical descriptors into the spatial hierarchy, separating geometric structure from feature dimensionality and allowing seamless integration of different neuroimaging feature sets. Self-supervised pretraining via masked reconstruction of both coordinates and morphometric channels on large unlabeled cohorts yields a transferable encoder backbone applicable to diverse downstream tasks and mesh modalities. We validate our approach on Alzheimer's disease classification and amyloid burden prediction using volumetric brain meshes from ADNI, as well as focal cortical dysplasia detection on cortical surface meshes from the MELD dataset, achieving state-of-the-art results across all benchmarks.
>
---
#### [new 118] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance
- **分类: cs.CV**

- **简介: 该论文属于3D生成任务，旨在解决无几何先验下3D高斯生成困难的问题。通过从点云生长高斯，并结合文本引导和扩散模型提升生成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2604.05721](https://arxiv.org/pdf/2604.05721)**

> **作者:** Weiqi Zhang; Junsheng Zhou; Haotian Geng; Kanle Shi; Shenkun Xu; Yi Fang; Yu-Shen Liu
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** 3D Gaussian Splatting has demonstrated superior performance in rendering efficiency and quality, yet the generation of 3D Gaussians still remains a challenge without proper geometric priors. Existing methods have explored predicting point maps as geometric references for inferring Gaussian primitives, while the unreliable estimated geometries may lead to poor generations. In this work, we introduce GaussianGrow, a novel approach that generates 3D Gaussians by learning to grow them from easily accessible 3D point clouds, naturally enforcing geometric accuracy in Gaussian generation. Specifically, we design a text-guided Gaussian growing scheme that leverages a multi-view diffusion model to synthesize consistent appearances from input point clouds for supervision. To mitigate artifacts caused by fusing neighboring views, we constrain novel views generated at non-preset camera poses identified in overlapping regions across different views. For completing the hard-to-observe regions, we propose to iteratively detect the camera pose by observing the largest un-grown regions in point clouds and inpainting them by inpainting the rendered view with a pretrained 2D diffusion model. The process continues until complete Gaussians are generated. We extensively evaluate GaussianGrow on text-guided Gaussian generation from synthetic and even real-scanned point clouds. Project Page: this https URL
>
---
#### [new 119] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决dMLLM在推理中过早生成答案和依赖视觉信息不足的问题。提出PSP和VRG方法提升推理质量和速度。**

- **链接: [https://arxiv.org/pdf/2604.05497](https://arxiv.org/pdf/2604.05497)**

> **作者:** Keuntae Kim; Mingyu Kang; Yong Suk Choi
>
> **备注:** CVPR 2026 - main
>
> **摘要:** Diffusion large language models (dLLMs) are emerging as promising alternatives to autoregressive (AR) LLMs. Recently, this paradigm has been extended to multimodal tasks, leading to the development of diffusion multimodal large language models (dMLLMs). These models are expected to retain the reasoning capabilities of LLMs while enabling faster inference through parallel generation. However, when combined with Chain-of-Thought (CoT) reasoning, dMLLMs exhibit two critical issues. First, we observe that dMLLMs often generate the final answer token at a very early timestep. This trend indicates that the model determines the answer before sufficient reasoning, leading to degraded reasoning performance. Second, during the initial timesteps, dMLLMs show minimal dependency on visual prompts, exhibiting a fundamentally different pattern of visual information utilization compared to AR vision-language models. In summary, these findings indicate that dMLLMs tend to generate premature final answers without sufficiently grounding on visual inputs. To address these limitations, we propose Position and Step Penalty (PSP) and Visual Reasoning Guidance (VRG). PSP penalizes tokens in later positions during early timesteps, delaying premature answer generation and encouraging progressive reasoning across timesteps. VRG, inspired by classifier-free guidance, amplifies visual grounding signals to enhance the model's alignment with visual evidence. Extensive experiments across various dMLLMs demonstrate that our method achieves up to 7.5% higher accuracy while delivering more than 3x speedup compared to reasoning with four times more diffusion steps.
>
---
#### [new 120] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言奖励建模任务，解决生成方法可解释性差与判别方法效率低的矛盾。提出VL-MDR框架，动态选择并加权评价维度，提升模型可解释性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.05445](https://arxiv.org/pdf/2604.05445)**

> **作者:** Qiyuan Chen; Hongsen Huang; Jiahe Chen; Qian Shao; Jintai Chen; Hongxia Xu; Renjie Hua; Chuan Ren; Jian Wu
>
> **备注:** ACL 2026 Main
>
> **摘要:** Vision-language reward modeling faces a dilemma: generative approaches are interpretable but slow, while discriminative ones are efficient but act as opaque "black boxes." To bridge this gap, we propose VL-MDR (Vision-Language Multi-Dimensional Reward), a framework that dynamically decomposes evaluation into granular, interpretable dimensions. Instead of outputting a monolithic scalar, VL-MDR employs a visual-aware gating mechanism to identify relevant dimensions and adaptively weight them (e.g., Hallucination, Reasoning) for each specific input. To support this, we curate a dataset of 321k vision-language preference pairs annotated across 21 fine-grained dimensions. Extensive experiments show that VL-MDR consistently outperforms existing open-source reward models on benchmarks like VL-RewardBench. Furthermore, we show that VL-MDR-constructed preference pairs effectively enable DPO alignment to mitigate visual hallucinations and improve reliability, providing a scalable solution for VLM alignment.
>
---
#### [new 121] CI-ICM: Channel Importance-driven Learned Image Coding for Machines
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于机器视觉压缩任务，解决传统方法在机器任务中效果不佳的问题。提出CI-ICM框架，通过通道重要性优化提升压缩效率与任务性能。**

- **链接: [https://arxiv.org/pdf/2604.05347](https://arxiv.org/pdf/2604.05347)**

> **作者:** Yun Zhang; Junle Liu; Huan Zhang; Zhaoqing Pan; Gangyi Jiang; Weisi Lin
>
> **摘要:** Traditional human vision-centric image compression methods are suboptimal for machine vision centric compression due to different visual properties and feature characteristics. To address this problem, we propose a Channel Importance-driven learned Image Coding for Machines (CI-ICM), aiming to maximize the performance of machine vision tasks at a given bitrate constraint. First, we propose a Channel Importance Generation (CIG) module to quantify channel importance in machine vision and develop a channel order loss to rank channels in descending order. Second, to properly allocate bitrate among feature channels, we propose a Feature Channel Grouping and Scaling (FCGS) module that non-uniformly groups the feature channels based on their importance and adjusts the dynamic range of each group. Based on FCGS, we further propose a Channel Importance-based Context (CI-CTX) module to allocate bits among feature groups and to preserve higher fidelity in critical channels. Third, to adapt to multiple machine tasks, we propose a Task-Specific Channel Adaptation (TSCA) module to adaptively enhance features for multiple downstream machine tasks. Experimental results on the COCO2017 dataset show that the proposed CI-ICM achieves BD-mAP@50:95 gains of 16.25$\%$ in object detection and 13.72$\%$ in instance segmentation over the established baseline codec. Ablation studies validate the effectiveness of each contribution, and computation complexity analysis reveals the practicability of the CI-ICM. This work establishes feature channel optimization for machine vision-centric compression, bridging the gap between image coding and machine perception.
>
---
#### [new 122] ICR-Drive: Instruction Counterfactual Robustness for End-to-End Language-Driven Autonomous Driving
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于语言驱动的自动驾驶任务，旨在解决指令鲁棒性问题。通过生成不同扰动指令变体，评估模型在真实场景中的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.05378](https://arxiv.org/pdf/2604.05378)**

> **作者:** Kaiser Hamid; Can Cui; Nade Liang
>
> **摘要:** Recent progress in vision-language-action (VLA) models has enabled language-conditioned driving agents to execute natural-language navigation commands in closed-loop simulation, yet standard evaluations largely assume instructions are precise and well-formed. In deployment, instructions vary in phrasing and specificity, may omit critical qualifiers, and can occasionally include misleading, authority-framed text, leaving instruction-level robustness under-measured. We introduce ICR-Drive, a diagnostic framework for instruction counterfactual robustness in end-to-end language-conditioned autonomous driving. ICR-Drive generates controlled instruction variants spanning four perturbation families: Paraphrase, Ambiguity, Noise, and Misleading, where Misleading variants conflict with the navigation goal and attempt to override intent. We replay identical CARLA routes under matched simulator configurations and seeds to isolate performance changes attributable to instruction language. Robustness is quantified using standard CARLA Leaderboard metrics and per-family performance degradation relative to the baseline instruction. Experiments on LMDrive and BEVDriver show that minor instruction changes can induce substantial performance drops and distinct failure modes, revealing a reliability gap for deploying embodied foundation models in safety-critical driving.
>
---
#### [new 123] StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出StarVLA，一个开源的视觉-语言-动作模型开发框架，旨在解决VLA研究中架构不兼容、评估不统一的问题，通过模块化设计和统一评估接口促进研究与应用。**

- **链接: [https://arxiv.org/pdf/2604.05014](https://arxiv.org/pdf/2604.05014)**

> **作者:** StarVLA Community
>
> **备注:** Open-source VLA infra, Technical Report
>
> **摘要:** Building generalist embodied agents requires integrating perception, language understanding, and action, which are core capabilities addressed by Vision-Language-Action (VLA) approaches based on multimodal foundation models, including recent advances in vision-language models and world models. Despite rapid progress, VLA methods remain fragmented across incompatible architectures, codebases, and evaluation protocols, hindering principled comparison and reproducibility. We present StarVLA, an open-source codebase for VLA research. StarVLA addresses these challenges in three aspects. First, it provides a modular backbone--action-head architecture that supports both VLM backbones (e.g., Qwen-VL) and world-model backbones (e.g., Cosmos) alongside representative action-decoding paradigms, all under a shared abstraction in which backbone and action head can each be swapped independently. Second, it provides reusable training strategies, including cross-embodiment learning and multimodal co-training, that apply consistently across supported paradigms. Third, it integrates major benchmarks, including LIBERO, SimplerEnv, RoboTwin~2.0, RoboCasa-GR1, and BEHAVIOR-1K, through a unified evaluation interface that supports both simulation and real-robot deployment. StarVLA also ships simple, fully reproducible single-benchmark training recipes that, despite minimal data engineering, already match or surpass prior methods on multiple benchmarks with both VLM and world-model backbones. To our best knowledge, StarVLA is one of the most comprehensive open-source VLA frameworks available, and we expect it to lower the barrier for reproducing existing methods and prototyping new ones. StarVLA is being actively maintained and expanded; we will update this report as the project evolves. The code and documentation are available at this https URL.
>
---
#### [new 124] Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于安全评估任务，旨在解决VLA模型对语言细微差异的脆弱性问题。通过提出DAERT框架，生成多样化攻击指令，有效暴露模型缺陷。**

- **链接: [https://arxiv.org/pdf/2604.05595](https://arxiv.org/pdf/2604.05595)**

> **作者:** Baoshun Tong; Haoran He; Ling Pan; Yang Liu; Liang Lin
>
> **摘要:** Vision-Language-Action (VLA) models have achieved remarkable success in robotic manipulation. However, their robustness to linguistic nuances remains a critical, under-explored safety concern, posing a significant safety risk to real-world deployment. Red teaming, or identifying environmental scenarios that elicit catastrophic behaviors, is an important step in ensuring the safe deployment of embodied AI agents. Reinforcement learning (RL) has emerged as a promising approach in automated red teaming that aims to uncover these vulnerabilities. However, standard RL-based adversaries often suffer from severe mode collapse due to their reward-maximizing nature, which tends to converge to a narrow set of trivial or repetitive failure patterns, failing to reveal the comprehensive landscape of meaningful risks. To bridge this gap, we propose a novel \textbf{D}iversity-\textbf{A}ware \textbf{E}mbodied \textbf{R}ed \textbf{T}eaming (\textbf{DAERT}) framework, to expose the vulnerabilities of VLAs against linguistic variations. Our design is based on evaluating a uniform policy, which is able to generate a diverse set of challenging instructions while ensuring its attack effectiveness, measured by execution failures in a physical simulator. We conduct extensive experiments across different robotic benchmarks against two state-of-the-art VLAs, including $\pi_0$ and OpenVLA. Our method consistently discovers a wider range of more effective adversarial instructions that reduce the average task success rate from 93.33\% to 5.85\%, demonstrating a scalable approach to stress-testing VLA agents and exposing critical safety blind spots before real-world deployment.
>
---
#### [new 125] CoStream: Codec-Guided Resource-Efficient System for Video Streaming Analytics
- **分类: cs.DC; cs.CV; cs.LG**

- **简介: 该论文提出CoStream系统，解决视频流分析中的高计算成本问题。通过利用视频编解码器的元数据，实现高效资源调度，提升处理速度并减少GPU使用。**

- **链接: [https://arxiv.org/pdf/2604.06036](https://arxiv.org/pdf/2604.06036)**

> **作者:** Yulin Zou; Yan Chen; Wenyan Chen; JooYoung Park; Shivaraman Nitin; Luo Tao; Francisco Romero; Dmitrii Ustiugov
>
> **备注:** 18 pages, 34 figures
>
> **摘要:** Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal inference limits scalability. Prior systems reduce inference cost by exploiting temporal and spatial redundancy in video streams, but they target either the vision transformer (ViT) or the LLM with a limited view, leaving end-to-end opportunities untapped. Moreover, existing methods incur significant overhead to identify redundancy, either through offline profiling and training or costly online computation, making them ill-suited for dynamic real-time streams. We present CoStream, a codec-guided streaming video analytics system built on a key observation that video codecs already extract the temporal and spatial structure of each stream as a byproduct of compression. CoStream treats this codec metadata as a low-cost runtime signal to unify optimization across video decoding, visual processing, and LLM prefilling, with transmission reduction as an inherent benefit of operating directly on compressed bitstreams. This drives codec-guided patch pruning before ViT encoding and selective key-value cache refresh during LLM prefilling, both of which are fully online and do not require offline training. Experiments show that CoStream achieves up to 3x throughput improvement and up to 87% GPU compute reduction over state-of-the-art baselines, while maintaining competitive accuracy with only 0-8% F1 drop.
>
---
#### [new 126] CoEnv: Driving Embodied Multi-Agent Collaboration via Compositional Environment
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CoEnv框架，解决多智能体协作中的空间协调与任务执行问题，通过组合环境实现仿真与现实的高效交互。**

- **链接: [https://arxiv.org/pdf/2604.05484](https://arxiv.org/pdf/2604.05484)**

> **作者:** Li Kang; Yutao Fan; Rui Li; Heng Zhou; Yiran Qin; Zhemeng Zhang; Songtao Huang; Xiufeng Song; Zaibin Zhang; Bruno N.Y. Chen; Zhenfei Yin; Dongzhan Zhou; Wangmeng Zuo; Lei Bai
>
> **备注:** 31 pages, 8 figures, including supplementary material. Project page: this https URL
>
> **摘要:** Multi-agent embodied systems hold promise for complex collaborative manipulation, yet face critical challenges in spatial coordination, temporal reasoning, and shared workspace awareness. Inspired by human collaboration where cognitive planning occurs separately from physical execution, we introduce the concept of compositional environment -- a synergistic integration of real-world and simulation components that enables multiple robotic agents to perceive intentions and operate within a unified decision-making space. Building on this concept, we present CoEnv, a framework that leverages simulation for safe strategy exploration while ensuring reliable real-world deployment. CoEnv operates through three stages: real-to-sim scene reconstruction that digitizes physical workspaces, VLM-driven action synthesis supporting both real-time planning with high-level interfaces and iterative planning with code-based trajectory generation, and validated sim-to-real transfer with collision detection for safe deployment. Extensive experiments on challenging multi-arm manipulation benchmarks demonstrate CoEnv's effectiveness in achieving high task success rates and execution efficiency, establishing a new paradigm for multi-agent embodied AI.
>
---
#### [new 127] Evaluation of Embedding-Based and Generative Methods for LLM-Driven Document Classification: Opportunities and Challenges
- **分类: cs.IR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于文档分类任务，旨在比较嵌入与生成模型在地质科学文档分类中的效果，分析其准确性、稳定性及计算成本。**

- **链接: [https://arxiv.org/pdf/2604.04997](https://arxiv.org/pdf/2604.04997)**

> **作者:** Rong Lu; Hao Liu; Song Hou
>
> **备注:** Accepted at the IMAGE'25 Workshop (PCW-11), Society of Exploration Geophysicists (SEG). Published version available at this https URL
>
> **摘要:** This work presents a comparative analysis of embedding-based and generative models for classifying geoscience technical documents. Using a multi-disciplinary benchmark dataset, we evaluated the trade-offs between model accuracy, stability, and computational cost. We find that generative Vision-Language Models (VLMs) like Qwen2.5-VL, enhanced with Chain-of-Thought (CoT) prompting, achieve superior zero-shot accuracy (82%) compared to state-of-the-art multimodal embedding models like QQMM (63%). We also demonstrate that while supervised fine-tuning (SFT) can improve VLM performance, it is sensitive to training data imbalance.
>
---
#### [new 128] AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于图像目标导航任务，旨在解决精确定位问题。通过几何查询方法，实现高精度相机位姿恢复，提升导航成功率与定位精度。**

- **链接: [https://arxiv.org/pdf/2604.05351](https://arxiv.org/pdf/2604.05351)**

> **作者:** Yijie Deng; Shuaihang Yuan; Yi Fang
>
> **摘要:** Image Goal Navigation (ImageNav) is evaluated by a coarse success criterion, the agent must stop within 1m of the target, which is sufficient for finding objects but falls short for downstream tasks such as grasping that require precise positioning. We introduce AnyImageNav, a training-free system that pushes ImageNav toward this more demanding setting. Our key insight is that the goal image can be treated as a geometric query: any photo of an object, a hallway, or a room corner can be registered to the agent's observations via dense pixel-level correspondences, enabling recovery of the exact 6-DoF camera pose. Our method realizes this through a semantic-to-geometric cascade: a semantic relevance signal guides exploration and acts as a proximity gate, invoking a 3D multi-view foundation model only when the current view is highly relevant to the goal image; the model then self-certifies its registration in a loop for an accurate recovered pose. Our method sets state-of-the-art navigation success rates on Gibson (93.1%) and HM3D (82.6%), and achieves pose recovery that prior methods do not provide: a position error of 0.27m and heading error of 3.41 degrees on Gibson, and 0.21m / 1.23 degrees on HM3D, a 5-10x improvement over adapted baselines.
>
---
#### [new 129] INTERACT: An AI-Driven Extended Reality Framework for Accesible Communication Featuring Real-Time Sign Language Interpretation and Emotion Recognition
- **分类: cs.CE; cs.AI; cs.CL; cs.CV; cs.ET**

- **简介: 该论文提出INTERACT框架，解决聋人及多语言用户在视频会议中的沟通障碍。通过AI与XR技术实现实时手语翻译和情感识别，提升沟通可及性。**

- **链接: [https://arxiv.org/pdf/2604.05605](https://arxiv.org/pdf/2604.05605)**

> **作者:** Nikolaos D. Tantaroudas; Andrew J. McCracken; Ilias Karachalios; Evangelos Papatheou
>
> **备注:** 20
>
> **摘要:** Video conferencing has become central to professional collaboration, yet most platforms offer limited support for deaf, hard-of-hearing, and multilingual users. The World Health Organisation estimates that over 430 million people worldwide require rehabilitation for disabling hearing loss, a figure projected to exceed 700 million by 2050. Conventional accessibility measures remain constrained by high costs, limited availability, and logistical barriers, while Extended Reality (XR) technologies open new possibilities for immersive and inclusive communication. This paper presents INTERACT (Inclusive Networking for Translation and Embodied Real-Time Augmented Communication Tool), an AI-driven XR platform that integrates real-time speech-to-text conversion, International Sign Language (ISL) rendering through 3D avatars, multilingual translation, and emotion recognition within an immersive virtual environment. Built on the CORTEX2 framework and deployed on Meta Quest 3 headsets, INTERACT combines Whisper for speech recognition, NLLB for multilingual translation, RoBERTa for emotion classification, and Google MediaPipe for gesture extraction. Pilot evaluations were conducted in two phases, first with technical experts from academia and industry, and subsequently with members of the deaf community. The trials reported 92% user satisfaction, transcription accuracy above 85%, and 90% emotion-detection precision, with a mean overall experience rating of 4.6 out of 5.0 and 90% of participants willing to take part in further testing. The results highlight strong potential for advancing accessibility across educational, cultural, and professional settings. An extended version of this work, including full pilot data and implementation details, has been published as an Open Research Europe article [Tantaroudas et al., 2026a].
>
---
#### [new 130] Final Report, Center for Computer-Integrated Computer-Integrated Surgical Systems and Technology, NSF ERC Cooperative Agreement EEC9731748, Volume 1
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于科技报告，描述了医疗机器人技术的发展与应用，旨在提升手术精度和医疗效果，通过工程研究推动医疗系统的革新。**

- **链接: [https://arxiv.org/pdf/2604.05272](https://arxiv.org/pdf/2604.05272)**

> **作者:** Russell H. Taylor; Gregory D. Hager; Ralph Etienne-Cummings. Eric Grimson; Ron Kikinis; Cameron Riviere
>
> **摘要:** In the last ten years, medical robotics has moved from the margins to the mainstream. Since the Engineering Research Center for Computer-Integrated Surgical Systems and Technology was Launched in 1998 with National Science Foundation funding, medical robots have been promoted from handling routine tasks to performing highly sophisticated interventions and related assignments. The CISST ERC has played a significant role in this transformation. And thanks to NSF support, the ERC has built the professional infrastructure that will continue our mission: bringing data and technology together in clinical systems that will dramatically change how surgery and other procedures are done. The enhancements we envision touch virtually every aspect of the delivery of care: - More accurate procedures - More consistent, predictable results from one patient to the next - Improved clinical outcomes - Greater patient safety - Reduced liability for healthcare providers - Lower costs for everyone - patients, facilities, insurers, government - Easier, faster recovery for patients - Effective new ways to treat health problems - Healthier patients, and a healthier system The basic science and engineering the ERC is developing now will yield profound benefits for all concerned about health care - from government agencies to insurers, from clinicians to patients to the general public. All will experience the healing touch of medical robotics, thanks in no small part to the work of the CISST ERC and its successors.
>
---
#### [new 131] Referring-Aware Visuomotor Policy Learning for Closed-Loop Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决视觉-运动策略在分布外误差和动态路径重规划中的鲁棒性问题。提出ReV框架，通过引入稀疏指代点实现闭环自适应调整。**

- **链接: [https://arxiv.org/pdf/2604.05544](https://arxiv.org/pdf/2604.05544)**

> **作者:** Jiahua Ma; Yiran Qin; Xin Wen; Yixiong Li; Yuyu Sun; Yulan Guo; Liang Lin; Ruimao Zhang
>
> **摘要:** This paper addresses a fundamental problem of visuomotor policy learning for robotic manipulation: how to enhance robustness in out-of-distribution execution errors or dynamically re-routing trajectories, where the model relies solely on the original expert demonstrations for training. We introduce the Referring-Aware Visuomotor Policy (ReV), a closed-loop framework that can adapt to unforeseen circumstances by instantly incorporating sparse referring points provided by a human or a high-level reasoning planner. Specifically, ReV leverages the coupled diffusion heads to preserve standard task execution patterns while seamlessly integrating sparse referring via a trajectory-steering strategy. Upon receiving a specific referring point, the global diffusion head firstly generates a sequence of globally consistent yet temporally sparse action anchors, while identifies the precise temporal position for the referring point within this sequence. Subsequently, the local diffusion head adaptively interpolates adjacent anchors based on the current temporal position for specific tasks. This closed-loop process repeats at every execution step, enabling real-time trajectory replanning in response to dynamic changes in the scene. In practice, rather than relying on elaborate annotations, ReV is trained only by applying targeted perturbations to expert demonstrations. Without any additional data or fine-tuning scheme, ReV achieve higher success rates across challenging simulated and real-world tasks.
>
---
#### [new 132] Part-Level 3D Gaussian Vehicle Generation with Joint and Hinge Axis Estimation
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于3D生成任务，旨在解决车辆模拟中缺乏部件级动态的问题。通过引入部分边缘精修模块和运动参数预测，生成可动画的3D车辆模型。**

- **链接: [https://arxiv.org/pdf/2604.05070](https://arxiv.org/pdf/2604.05070)**

> **作者:** Shiyao Qian; Yuan Ren; Dongfeng Bai; Bingbing Liu
>
> **备注:** submitted to IROS 2026
>
> **摘要:** Simulation is essential for autonomous driving, yet current frameworks often model vehicles as rigid assets and fail to capture part-level articulation. With perception algorithms increasingly leveraging dynamics such as wheel steering or door opening, realistic simulation requires animatable vehicle representations. Existing CAD-based pipelines are limited by library coverage and fixed templates, preventing faithful reconstruction of in-the-wild instances. We propose a generative framework that, from a single image or sparse multi-view input, synthesizes an animatable 3D Gaussian vehicle. Our method addresses two challenges: (i) large 3D asset generators are optimized for static quality but not articulation, leading to distortions at part boundaries when animated; and (ii) segmentation alone cannot provide the kinematic parameters required for motion. To overcome this, we introduce a part-edge refinement module that enforces exclusive Gaussian ownership and a kinematic reasoning head that predicts joint positions and hinge axes of movable parts. Together, these components enable faithful part-aware simulation, bridging the gap between static generation and animatable vehicle models.
>
---
#### [new 133] Training Without Orthogonalization, Inference With SVD: A Gradient Analysis of Rotation Representations
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究旋转估计任务，解决训练中正交化导致梯度问题，提出在训练中去除正交化、推理时用SVD的方案。通过分析SVD和Gram-Schmidt的梯度特性，验证9D参数化的有效性。**

- **链接: [https://arxiv.org/pdf/2604.05414](https://arxiv.org/pdf/2604.05414)**

> **作者:** Chris Choy
>
> **摘要:** Recent work has shown that removing orthogonalization during training and applying it only at inference improves rotation estimation in deep learning, with empirical evidence favoring 9D representations with SVD projection. However, the theoretical understanding of why SVD orthogonalization specifically harms training, and why it should be preferred over Gram-Schmidt at inference, remains incomplete. We provide a detailed gradient analysis of SVD orthogonalization specialized to $3 \times 3$ matrices and $SO(3)$ projection. Our central result derives the exact spectrum of the SVD backward pass Jacobian: it has rank $3$ (matching the dimension of $SO(3)$) with nonzero singular values $2/(s_i + s_j)$ and condition number $\kappa = (s_1 + s_2)/(s_2 + s_3)$, creating quantifiable gradient distortion that is most severe when the predicted matrix is far from $SO(3)$ (e.g., early in training when $s_3 \approx 0$). We further show that even stabilized SVD gradients introduce gradient direction error, whereas removing SVD from the training loop avoids this tradeoff entirely. We also prove that the 6D Gram-Schmidt Jacobian has an asymmetric spectrum: its parameters receive unequal gradient signal, explaining why 9D parameterization is preferable. Together, these results provide the theoretical foundation for training with direct 9D regression and applying SVD projection only at inference.
>
---
#### [new 134] BodhiPromptShield: Pre-Inference Prompt Mediation for Suppressing Privacy Propagation in LLM/VLM Agents
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私保护任务，解决LLM/VLM代理中提示信息泄露问题。提出BodhiPromptShield框架，通过敏感内容检测与延迟恢复机制抑制隐私传播。**

- **链接: [https://arxiv.org/pdf/2604.05793](https://arxiv.org/pdf/2604.05793)**

> **作者:** Bo Ma; Jinsong Wu; Weiqi Yan
>
> **摘要:** In LLM/VLM agents, prompt privacy risk propagates beyond a single model call because raw user content can flow into retrieval queries, memory writes, tool calls, and logs. Existing de-identification pipelines address document boundaries but not this cross-stage propagation. We propose BodhiPromptShield, a policy-aware framework that detects sensitive spans, routes them via typed placeholders, semantic abstraction, or secure symbolic mapping, and delays restoration to authorized boundaries. Relative to enterprise redaction, this adds explicit propagation-aware mediation and restoration timing as a security variable. Under controlled evaluation on the Controlled Prompt-Privacy Benchmark (CPPB), stage-wise propagation suppresses from 10.7\% to 7.1\% across retrieval, memory, and tool stages; PER reaches 9.3\% with 0.94 AC and 0.92 TSR, outperforming generic de-identification. These are controlled systems results on CPPB rather than formal privacy guarantees or public-benchmark transfer claims. The project repository is available at this https URL.
>
---
## 更新

#### [replaced 001] ForgeryGPT: A Multimodal LLM for Interpretable Image Forgery Detection and Localization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2410.10238](https://arxiv.org/pdf/2410.10238)**

> **作者:** Fanrui Zhang; Jiawei Liu; Jiaying Zhu; Esther Sun; Dong Li; Qiang Zhang; Zheng-Jun Zha
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs), such as GPT4o, have shown strong capabilities in visual reasoning and explanation generation. However, despite these strengths, they face significant challenges in the increasingly critical task of Image Forgery Detection and Localization (IFDL). Moreover, existing IFDL methods are typically limited to the learning of low-level semantic-agnostic clues and merely provide a single outcome judgment. To tackle these issues, we propose ForgeryGPT, a novel framework that advances the IFDL task by capturing high-order forensics knowledge correlations of forged images from diverse linguistic feature spaces, while enabling explainable generation and interactive dialogue through a newly customized Large Language Model (LLM) architecture. Specifically, ForgeryGPT enhances traditional LLMs by integrating the Mask-Aware Forgery Extractor, which enables the excavating of precise forgery mask information from input images and facilitating pixel-level understanding of tampering artifacts. The Mask-Aware Forgery Extractor consists of a Forgery Localization Expert (FL-Expert) and a Mask Encoder, where the FL-Expert is augmented with an Object-agnostic Forgery Prompt and a Vocabulary-enhanced Vision Encoder, allowing for effectively capturing of multi-scale fine-grained forgery details. To enhance its performance, we implement a three-stage training strategy, supported by our designed Mask-Text Alignment and IFDL Task-Specific Instruction Tuning datasets, which align vision-language modalities and improve forgery detection and instruction-following capabilities. Extensive experiments demonstrate the effectiveness of the proposed method.
>
---
#### [replaced 002] DSER: Spectral Epipolar Representation for Efficient Light Field Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08900](https://arxiv.org/pdf/2508.08900)**

> **作者:** Noor Islam S. Mohammad; Md Muntaqim Meherab
>
> **备注:** We have recently had author conflicts with this work; I heartily request to withdraw his paper as soon as possible
>
> **摘要:** Dense light field depth estimation remains challenging due to sparse angular sampling, occlusion boundaries, textureless regions, and the cost of exhaustive multi-view matching. We propose \emph{Deep Spectral Epipolar Representation} (DSER), a geometry-aware framework that introduces spectral regularization in the epipolar domain for dense disparity reconstruction. DSER models frequency-consistent EPI structure to constrain correspondence estimation and couples this prior with a hybrid inference pipeline that combines least squares gradient initialization, plane-sweeping cost aggregation, and multiscale EPI refinement. An occlusion-aware directed random walk further propagates reliable disparity along edge-consistent paths, improving boundary sharpness and weak-texture stability. Experiments on benchmark and real-world light field datasets show that DSER achieves a strong accuracy-efficiency trade-off, producing more structurally consistent depth maps than representative classical and hybrid baselines. These results establish spectral epipolar regularization as an effective inductive bias for scalable and noise-robust light field depth estimation.
>
---
#### [replaced 003] R3G: A Reasoning--Retrieval--Reranking Framework for Vision-Centric Answer Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00104](https://arxiv.org/pdf/2602.00104)**

> **作者:** Zhuohong Chen; Zhengxian Wu; Zirui Liao; Shenao Jiang; Hangrui Xu; Yang Chen; Chaokui Su; Xiaoyu Liu; Haoqian Wang
>
> **摘要:** Vision-centric retrieval for VQA requires retrieving images to supply missing visual cues and integrating them into the reasoning process. However, selecting the right images and integrating them effectively into the model's reasoning remains this http URL address this challenge, we propose R3G, a modular Reasoning-Retrieval-Reranking this http URL first produces a brief reasoning plan that specifies the required visual cues, then adopts a two-stage strategy, with coarse retrieval followed by fine-grained reranking, to select evidence this http URL MRAG-Bench, R3G improves accuracy across six MLLM backbones and nine sub-scenarios, achieving state-of-the-art overall performance. Ablations show that sufficiency-aware reranking and reasoning steps are complementary, helping the model both choose the right images and use them well. We release code and data at this https URL.
>
---
#### [replaced 004] MATRIX: Mask Track Alignment for Interaction-aware Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07310](https://arxiv.org/pdf/2510.07310)**

> **作者:** Siyoon Jin; Seongchan Kim; Dahyun Chung; Jaeho Lee; Hyunwook Choi; Jisu Nam; Jiyoung Kim; Seungryong Kim
>
> **备注:** Project Page is available at: this https URL, ICLR 2026
>
> **摘要:** Video DiTs have advanced video generation, yet they still struggle to model multi-instance or subject-object interactions. This raises a key question: How do these models internally represent interactions? To answer this, we curate MATRIX-11K, a video dataset with interaction-aware captions and multi-instance mask tracks. Using this dataset, we conduct a systematic analysis that formalizes two perspectives of video DiTs: semantic grounding, via video-to-text attention, which evaluates whether noun and verb tokens capture instances and their relations; and semantic propagation, via video-to-video attention, which assesses whether instance bindings persist across frames. We find both effects concentrate in a small subset of interaction-dominant layers. Motivated by this, we introduce MATRIX, a simple and effective regularization that aligns attention in specific layers of video DiTs with multi-instance mask tracks from the MATRIX-11K dataset, enhancing both grounding and propagation. We further propose InterGenEval, an evaluation protocol for interaction-aware video generation. In experiments, MATRIX improves both interaction fidelity and semantic alignment while reducing drift and hallucination. Extensive ablations validate our design choices. Codes and weights will be released.
>
---
#### [replaced 005] Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出“Thinking with Video”新范式，解决多模态推理问题。通过视频生成模型进行统一多模态理解与生成，构建基准测试并验证其性能。**

- **链接: [https://arxiv.org/pdf/2511.04570](https://arxiv.org/pdf/2511.04570)**

> **作者:** Jingqi Tong; Yurong Mou; Hangcheng Li; Mingzhe Li; Yongzhuo Yang; Ming Zhang; Qiguang Chen; Tianyi Liang; Xiaomeng Hu; Yining Zheng; Xinchi Chen; Jun Zhao; Xuanjing Huang; Xipeng Qiu
>
> **备注:** 34 pages, 17 figures
>
> **摘要:** The "Thinking with Text" and "Thinking with Images" paradigms significantly improve the reasoning abilities of large language models (LLMs) and Vision-Language Models (VLMs). However, these paradigms have inherent limitations. (1) Images capture only single moments and fail to represent dynamic processes or continuous changes, and (2) The separation of text and vision as distinct modalities, which hinders unified multimodal understanding and generation. Therefore, we propose "Thinking with Video", a new paradigm that leverages video generation models such as Sora-2 to use video frames as a unified medium for multimodal reasoning. To support this exploration, we developed the Video Thinking Benchmark (VideoThinkBench), which covers both vision-centric tasks (e.g., Eyeballing Puzzles) and text-centric tasks (e.g., GSM8K and MMMU). Our evaluation on VideoThinkBench establishes Sora-2 as a capable reasoner. On vision-centric tasks, Sora-2 is comparable to state-of-the-art (SOTA) VLMs, and even surpasses GPT-5 by 10% on eyeballing puzzles. On text-centric tasks, Sora-2 achieves 92% accuracy on MATH, and 69.2% accuracy on MMMU. Furthermore, we systematically analyze the source of these abilities. We also find that self-consistency and in-context learning can improve Sora-2's performance. In summary, our findings show that the video generation model is the potential unified multimodal understanding and generation model, positioning "Thinking with Video" as a potential unified multimodal reasoning paradigm.
>
---
#### [replaced 006] Diff4Splat: Controllable 4D Scene Generation with Latent Dynamic Reconstruction Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00503](https://arxiv.org/pdf/2511.00503)**

> **作者:** Panwang Pan; Chenguo Lin; Jingjing Zhao; Chenxin Li; Yuchen Lin; Haopeng Li; Honglei Yan; Kairun Wen; Yunlong Lin; Yixuan Yuan; Yadong Mu
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We introduce Diff4Splat, a feed-forward method that synthesizes controllable and explicit 4D scenes from a single image. Our approach unifies the generative priors of video diffusion models with geometry and motion constraints learned from large-scale 4D datasets. Given a single input image, a camera trajectory, and an optional text prompt, Diff4Splat directly predicts a deformable 3D Gaussian field that encodes appearance, geometry, and motion, all in a single forward pass, without test-time optimization or post-hoc refinement. At the core of our framework lies a video latent transformer, which augments video diffusion models to jointly capture spatio-temporal dependencies and predict time-varying 3D Gaussian primitives. Training is guided by objectives on appearance fidelity, geometric accuracy, and motion consistency, enabling Diff4Splat to synthesize high-quality 4D scenes in 30 seconds. We demonstrate the effectiveness of Diff4Splat across video generation, novel view synthesis, and geometry extraction, where it matches or surpasses optimization-based methods for dynamic scene synthesis while being significantly more efficient.
>
---
#### [replaced 007] Forgery-aware Layer Masking and Multi-Artifact Subspace Decomposition for Generalizable Deepfake Detection
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2601.01041](https://arxiv.org/pdf/2601.01041)**

> **作者:** Xiang Zhang; Wenliang Weng; Daoyong Fu; Beijing Chen; Ziqiang Li; Ziwen He; Zhangjie Fu
>
> **摘要:** Deepfake detection remains highly challenging, particularly in cross-dataset scenarios and complex real-world settings. This challenge mainly arises because artifact patterns vary substantially across different forgery methods, whereas adapting pretrained models to such artifacts often overemphasizes forgery-specific cues and disturbs semantic representations, thereby weakening generalization. Existing approaches typically rely on full-parameter fine-tuning or auxiliary supervision to improve discrimination. However, they often struggle to model diverse forgery artifacts without compromising pretrained representations. To address these limitations, we propose FMSD, a deepfake detection framework built upon Forgery-aware Layer Masking and Multi-Artifact Subspace Decomposition. Specifically, Forgery-aware Layer Masking evaluates the bias-variance characteristics of layer-wise gradients to identify forgery-sensitive layers, thereby selectively updating them while reducing unnecessary disturbance to pretrained representations. Building upon this, Multi-Artifact Subspace Decomposition further decomposes the selected layer weights via Singular Value Decomposition (SVD) into a semantic subspace and multiple learnable artifact subspaces. These subspaces are optimized to capture heterogeneous and complementary forgery artifacts, enabling effective modeling of diverse forgery patterns while preserving pretrained semantic representations. Furthermore, orthogonality and spectral consistency constraints are imposed to regularize the artifact subspaces, reducing redundancy across them while preserving the overall spectral structure of pretrained weights.
>
---
#### [replaced 008] DialectGen: Benchmarking and Improving Dialect Robustness in Multimodal Generation
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于多模态生成任务，旨在解决方言输入下模型性能下降的问题。通过构建方言基准，验证现有模型的不足，并提出一种提升方言鲁棒性的方法。**

- **链接: [https://arxiv.org/pdf/2510.14949](https://arxiv.org/pdf/2510.14949)**

> **作者:** Yu Zhou; Sohyun An; Haikang Deng; Da Yin; Clark Peng; Cho-Jui Hsieh; Kai-Wei Chang; Nanyun Peng
>
> **摘要:** Contact languages like English exhibit rich regional variations in the form of dialects, which are often used by dialect speakers interacting with generative models. However, can multimodal generative models effectively produce content given dialectal textual input? In this work, we study this question by constructing a new large-scale benchmark spanning six common English dialects. We work with dialect speakers to collect and verify over 4200 unique prompts and evaluate on 17 image and video generative models. Our automatic and human evaluation results show that current state-of-the-art multimodal generative models exhibit 32.26% to 48.17% performance degradation when a single dialect word is used in the prompt. Common mitigation methods such as fine-tuning and prompt rewriting can only improve dialect performance by small margins (< 7%), while potentially incurring significant performance degradation in Standard American English (SAE). To this end, we design a general encoder-based mitigation strategy for multimodal generative models. Our method teaches the model to recognize new dialect features while preserving SAE performance. Experiments on models such as Stable Diffusion 1.5 show that our method is able to simultaneously raise performance on five dialects to be on par with SAE (+34.4%), while incurring near zero cost to SAE performance.
>
---
#### [replaced 009] NoisyGRPO: Incentivizing Multimodal CoT Reasoning via Noise Injection and Bayesian Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21122](https://arxiv.org/pdf/2510.21122)**

> **作者:** Longtian Qiu; Shan Ning; Jiaxuan Sun; Xuming He
>
> **备注:** Accepted by Neurips 2025, Project page is available at this https URL
>
> **摘要:** Reinforcement learning (RL) has shown promise in enhancing the general Chain-of-Thought (CoT) reasoning capabilities of multimodal large language models (MLLMs). However, when applied to improve general CoT reasoning, existing RL frameworks often struggle to generalize beyond the training distribution. To address this, we propose NoisyGRPO, a systematic multimodal RL framework that introduces controllable noise into visual inputs for enhanced exploration and explicitly models the advantage estimation process via a Bayesian framework. Specifically, NoisyGRPO improves RL training by: (1) Noise-Injected Exploration Policy: Perturbing visual inputs with Gaussian noise to encourage exploration across a wider range of visual scenarios; and (2) Bayesian Advantage Estimation: Formulating advantage estimation as a principled Bayesian inference problem, where the injected noise level serves as a prior and the observed trajectory reward as the likelihood. This Bayesian modeling fuses both sources of information to compute a robust posterior estimate of trajectory advantage, effectively guiding MLLMs to prefer visually grounded trajectories over noisy ones. Experiments on standard CoT quality, general capability, and hallucination benchmarks demonstrate that NoisyGRPO substantially improves generalization and robustness, especially in RL settings with small-scale MLLMs such as Qwen2.5-VL 3B. The project page is available at this https URL.
>
---
#### [replaced 010] Move What Matters: Parameter-Efficient Domain Adaptation via Optimal Transport Flow for Collaborative Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.11565](https://arxiv.org/pdf/2602.11565)**

> **作者:** Zesheng Jia; Jin Wang; Siao Liu; Lingzhi Li; Ziyao Huang; Yunjiang Xu; Jianping Wang
>
> **摘要:** Fast domain adaptation remains a fundamental challenge for deploying multi-agent systems across diverse environments in Vehicle-to-Everything (V2X) collaborative perception. Despite the success of Parameter-Efficient Fine-Tuning (PEFT) in natural language processing and conventional vision tasks, directly applying PEFT to multi-agent settings leads to significant performance degradation and training instability. In this work, we conduct a detailed analysis and identify two key factors: (i) inter-frame redundancy in heterogeneous sensory streams, and (ii) erosion of fine-grained semantics in deep-layer representations under PEFT adaptation. To address these issues, we propose FlowAdapt, a parameter-efficient framework grounded in optimal transport theory, which minimizes information transport costs across both data distributions and network hierarchies. Specifically, we introduce a Wasserstein Greedy Sampling strategy to selectively filter redundant samples via a bounded covering radius. Furthermore, Progressive Knowledge Transfer module is designed to progressively inject compressed early-stage representations into later stages through learnable pathways, alleviating semantic degradation in late-stage adaptation. Extensive experiments on three benchmarks demonstrate that FlowAdapt achieves state-of-the-art performance with only 1% of trainable parameters, effectively bridging domain gaps with superior sample efficiency and generalization.
>
---
#### [replaced 011] Gesture-Aware Pretraining and Token Fusion for 3D Hand Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17396](https://arxiv.org/pdf/2603.17396)**

> **作者:** Rui Hong; Jana Kosecka
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Estimating 3D hand pose from monocular RGB images is fundamental for applications in AR/VR, human-computer interaction, and sign language understanding. In this work we focus on a scenario where a discrete set of gesture labels is available and show that gesture semantics can serve as a powerful inductive bias for 3D pose estimation. We present a two-stage framework: gesture-aware pretraining that learns an informative embedding space using coarse and fine gesture labels from InterHand2.6M, followed by a per-joint token Transformer guided by gesture embeddings as intermediate representations for final regression of MANO hand parameters. Training is driven by a layered objective over parameters, joints, and structural constraints. Experiments on InterHand2.6M demonstrate that gesture-aware pretraining consistently improves single-hand accuracy over the state-of-the-art EANet baseline, and that the benefit transfers across architectures without any modification.
>
---
#### [replaced 012] Automatic Image-Level Morphological Trait Annotation for Organismal Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.01619](https://arxiv.org/pdf/2604.01619)**

> **作者:** Vardaan Pahuja; Samuel Stevens; Alyson East; Sydne Record; Yu Su
>
> **备注:** ICLR 2026
>
> **摘要:** Morphological traits are physical characteristics of biological organisms that provide vital clues on how organisms interact with their environment. Yet extracting these traits remains a slow, expert-driven process, limiting their use in large-scale ecological studies. A major bottleneck is the absence of high-quality datasets linking biological images to trait-level annotations. In this work, we demonstrate that sparse autoencoders trained on foundation-model features yield monosemantic, spatially grounded neurons that consistently activate on meaningful morphological parts. Leveraging this property, we introduce a trait annotation pipeline that localizes salient regions and uses vision-language prompting to generate interpretable trait descriptions. Using this approach, we construct Bioscan-Traits, a dataset of 80K trait annotations spanning 19K insect images from BIOSCAN-5M. Human evaluation confirms the biological plausibility of the generated morphological descriptions. We assess design sensitivity through a comprehensive ablation study, systematically varying key design choices and measuring their impact on the quality of the resulting trait descriptions. By annotating traits with a modular pipeline rather than prohibitively expensive manual efforts, we offer a scalable way to inject biologically meaningful supervision into foundation models, enable large-scale morphological analyses, and bridge the gap between ecological relevance and machine-learning practicality.
>
---
#### [replaced 013] VideoTIR: Accurate Understanding for Long Videos with Efficient Tool-Integrated Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25021](https://arxiv.org/pdf/2603.25021)**

> **作者:** Zhe Gao; Shiyu Shen; Taifeng Chai; Weinong Wang; Haotian Xu; Xing W; Wenbin Li; Qi Fan; Yang Gao; Dacheng Tao
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) often suffer from hallucinations in long video understanding (LVU), primarily due to the imbalance between textual and visual tokens. Observing that MLLMs handle short visual inputs well, recent LVU works alleviate hallucinations by automatically parsing the vast visual data into manageable segments that can be effectively processed by MLLMs. SFT-based tool-calling methods can serve this purpose, but they typically require vast amounts of fine-grained, high-quality data and suffer from constrained tool-calling trajectories. We propose a novel VideoTIR that leverages Reinforcement Learning (RL) to encourage proper usage of comprehensive multi-level toolkits for efficient long video understanding. VideoTIR explores both Zero-RL and SFT cold-starting to enable MLLMs to retrieve and focus on meaningful video segments/images/regions, enhancing long video understanding both accurately and efficiently. To reduce redundant tool-calling, we propose Toolkit Action Grouped Policy Optimization (TAGPO), which enhances the efficiency of the calling process through stepwise reward assignment and reuse of failed rollouts. Additionally, we develop a sandbox-based trajectory synthesis framework to generate high-quality trajectories data. Extensive experiments on three long-video QA benchmarks demonstrate the effectiveness and efficiency of our method.
>
---
#### [replaced 014] Cattle-CLIP: A Multimodal Framework for Cattle Behaviour Recognition from Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09203](https://arxiv.org/pdf/2510.09203)**

> **作者:** Huimin Liu; Jing Gao; Daria Baran; AxelX Montout; Neill W Campbell; Andrew W Dowsey
>
> **备注:** 16 pages, 10 figures, submitted to Information Processing in Agriculture
>
> **摘要:** Robust behaviour recognition in real-world farm environments remains challenging due to several data-related limitations, including the scarcity of well-annotated livestock video datasets and the substantial domain gap between large-scale pre-training corpora and agricultural surveillance footage. To address these challenges, we propose Cattle-CLIP, a domain-adaptive vision-language framework that reformulates cattle behaviour recognition as cross-modal semantic alignment rather than purely visual classification. Instead of directly fine-tuning visual backbones, Cattle-CLIP incorporates a temporal integration module to extend image-level contrastive pre-training to video-based behaviour understanding, enabling consistent semantic alignment across time. To mitigate the distribution shift between web-scale image-text data used for the pre-trained model and real-world cattle surveillance footage, we further introduce tailored augmentation strategies and specialised behaviour prompts. Furthermore, we construct CattleBehaviours6, a curated and behaviour-consistent video dataset comprising 1905 annotated clips across six indoor behaviours to support model training and evaluation. Beyond serving as a benchmark for our proposed method, the dataset provides a standardised ethogram definition, offering a practical resource for future research in livestock behaviour analysis. Cattle-CLIP is evaluated under both fully-supervised and few-shot learning scenarios, with a particular focus on data-scarce behaviour recognition, an important yet under-explored goal in livestock monitoring. Experiments show that Cattle-CLIP achieves 96.1% overall accuracy across six behaviours in supervised settings, with near-perfect recall for feeding, drinking and standing-ruminating behaviours, and demonstrates robust generalisation with limited data in few-shot scenarios.
>
---
#### [replaced 015] SigLino: Efficient Multi-Teacher Distillation for Agglomerative Vision Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20157](https://arxiv.org/pdf/2512.20157)**

> **作者:** Sofian Chaybouti; Sanath Narayan; Yasser Dahou; Phúc H. Lê Khac; Ankit Singh; Ngoc Dung Huynh; Wamiq Reyaz Para; Hilde Kuehne; Hakim Hacid
>
> **备注:** 17 pages, 8 figures, 11 tables
>
> **摘要:** Vision foundation models trained via multi-teacher distillation offer a promising path toward unified visual representations, yet the learning dynamics and data efficiency of such approaches remain underexplored. In this paper, we systematically study multi-teacher distillation for vision foundation models and identify key factors that enable training at lower computational cost. We introduce SigLino, an efficient family of agglomerative vision foundation models that distill knowledge from SigLIP2 and DINOv3 simultaneously into Dense and Mixture-of-Experts students. We show that (1) our Asymmetric Relation-Knowledge Distillation loss preserves the geometric properties of each teacher while enabling effective knowledge transfer, (2) token-balanced batching that packs varying-resolution images into sequences with uniform token budgets stabilizes representation learning across resolutions without sacrificing performance, (3) hierarchical clustering and sampling of training data, typically reserved for self-supervised learning, substantially improves sample efficiency over random sampling for multi-teacher distillation, and (4) the resulting representations transfer effectively to early-fusion Grounding-VLMs, outperforming models trained from scratch. By combining these findings, we curate OpenLVD200M, a 200M-image corpus that demonstrates superior efficiency for multi-teacher distillation. Instantiated in a Mixture-of-Experts, our SigLino-MoE initializes an early-fusion Grounding-VLM that replaces the conventional ViT->LLM stack, demonstrating improved performance compared to a model trained from scratch. We release OpenLVD200M and five distilled checkpoints comprising MoE and dense variants.
>
---
#### [replaced 016] MoCHA: Denoising Caption Supervision for Motion-Text Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23684](https://arxiv.org/pdf/2603.23684)**

> **作者:** Nikolai Warner; Cameron Ethan Taylor; Irfan Essa; Apaar Sadhwani
>
> **摘要:** Text-motion retrieval systems learn shared embedding spaces from motion-caption pairs via contrastive objectives. However, each caption is not a deterministic label but a sample from a distribution of valid descriptions: different annotators produce different text for the same motion, mixing motion-recoverable semantics (action type, body parts, directionality) with annotator-specific style and inferred context that cannot be determined from 3D joint coordinates alone. Standard contrastive training treats each caption as the single positive target, overlooking this distributional structure and inducing within-motion embedding variance that weakens alignment. We propose MoCHA, a text canonicalization framework that reduces this variance by projecting each caption onto its motion-recoverable content prior to encoding, producing tighter positive clusters and better-separated embeddings. Canonicalization is a general principle: even deterministic rule-based methods improve cross-dataset transfer, though learned canonicalizers provide substantially larger gains. We present two learned variants: an LLM-based approach (GPT-5.2) and a distilled FlanT5 model requiring no LLM at inference time. MoCHA operates as a preprocessing step compatible with any retrieval architecture. Applied to MoPa (MotionPatches), MoCHA sets a new state of the art on both HumanML3D (H) and KIT-ML (K): the LLM variant achieves 13.9% T2M R@1 on H (+3.1pp) and 24.3% on K (+10.3pp), while the LLM-free T5 variant achieves gains of +2.5pp and +8.1pp. Canonicalization reduces within-motion text-embedding variance by 11-19% and improves cross-dataset transfer substantially, with H to K improving by 94% and K to H by 52%, demonstrating that standardizing the language space yields more transferable motion-language representations.
>
---
#### [replaced 017] MIMIC: Multimodal Inversion for Model Interpretation and Conceptualization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07833](https://arxiv.org/pdf/2508.07833)**

> **作者:** Animesh Jain; Alexandros Stergiou
>
> **备注:** Accepted at CVPRw 2026 - How Do Vision Models Work? (HOW) Workshop, Project page: this https URL
>
> **摘要:** Vision Language Models (VLMs) encode multimodal inputs over large, complex, and difficult-to-interpret architectures, which limit transparency and trust. We propose a Multimodal Inversion for Model Interpretation and Conceptualization (MIMIC) framework that inverts the internal encodings of VLMs. MIMIC uses a joint VLM-based inversion and a feature alignment objective to account for VLM's autoregressive processing. It additionally includes a triplet of regularizers for spatial alignment, natural image smoothness, and semantic realism. We evaluate MIMIC both quantitatively and qualitatively by inverting visual concepts across a range of free-form VLM outputs of varying length. Reported results include both standard visual quality metrics and semantic text-based metrics. To the best of our knowledge, this is the first model inversion approach addressing visual interpretations of VLM concepts.
>
---
#### [replaced 018] FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Moving Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14690](https://arxiv.org/pdf/2601.14690)**

> **作者:** Yian Huang; Qing Qin; Aji Mao; Xiangyu Qiu; Liang Xu; Xian Zhang; Zhenming Peng
>
> **备注:** Submitted to Journal IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Infrared small target detection (ISTD) has been a critical technology in defense and civilian applications over the past several decades, such as missile warning, maritime surveillance, and disaster monitoring. Nevertheless, moving infrared small target detection still faces considerable challenges: existing models suffer from insufficient spatio-temporal semantic correlation and are not lightweight-friendly, while algorithms with strong scene generalization capability are in great demand for real-world applications. To address these issues, we propose FeedbackSTS-Det, a sparse frames-based spatio-temporal semantic feedback network. Our approach introduces a closed-loop spatio-temporal semantic feedback strategy with paired forward and backward refinement modules that work cooperatively across the encoder and decoder to enhance information exchange between consecutive frames, effectively improving detection accuracy and reducing false alarms. Moreover, we introduce an embedded sparse semantic module (SSM), which operates by strategically grouping frames by interval, propagating semantics within each group, and reassembling the sequence to efficiently capture long-range temporal dependencies with low computational overhead. Extensive experiments on many widely adopted multi-frame infrared small target datasets demonstrate the generalization ability and scene adaptability of our proposed network. Code and models are available at: this https URL.
>
---
#### [replaced 019] I2E: From Image Pixels to Actionable Interactive Environments for Text-Guided Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03741](https://arxiv.org/pdf/2601.03741)**

> **作者:** Jinghan Yu; Junhao Xiao; Chenyu Zhu; Jiaming Li; Jia Li; HanMing Deng; Xirui Wang; Guoli Jia; Jianjun Li; Xiang Bai; Bowen Zhou; Zhiyuan Ma
>
> **摘要:** Existing text-guided image editing methods primarily rely on end-to-end pixel-level inpainting paradigm. Despite its success in simple scenarios, this paradigm still significantly struggles with compositional editing tasks that require precise local control and complex multi-object spatial reasoning. This paradigm is severely limited by 1) the implicit coupling of planning and execution, 2) the lack of object-level control granularity, and 3) the reliance on unstructured, pixel-centric modeling. To address these limitations, we propose I2E, a novel "Decompose-then-Action" paradigm that revisits image editing as an actionable interaction process within a structured environment. I2E utilizes a Decomposer to transform unstructured images into discrete, manipulable object layers and then introduces a physics-aware Vision-Language-Action Agent to parse complex instructions into a series of atomic actions via Chain-of-Thought reasoning. Further, we also construct I2E-Bench, a benchmark designed for multi-instance spatial reasoning and high-precision editing. Experimental results on I2E-Bench and multiple public benchmarks demonstrate that I2E significantly outperforms state-of-the-art methods in handling complex compositional instructions, maintaining physical plausibility, and ensuring multi-turn editing stability.
>
---
#### [replaced 020] SiLVi: Simple Interface for Labeling Video Interactions
- **分类: cs.CV; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2511.03819](https://arxiv.org/pdf/2511.03819)**

> **作者:** Ozan Kanbertay; Richard Vogg; Elif Karakoc; Peter M. Kappeler; Claudia Fichtel; Alexander S. Ecker
>
> **备注:** Documentation link updated, Linux version added
>
> **摘要:** Computer vision methods are increasingly used for the automated analysis of large volumes of video data collected through camera traps, drones, or direct observations of animals in the wild. While recent advances have focused primarily on detecting individual actions, much less work has addressed the detection and annotation of interactions -- a crucial aspect for understanding social and individualized animal behavior. Existing open-source annotation tools support either behavioral labeling without localization of individuals, or localization without the capacity to capture interactions. To bridge this gap, we present SiLVi, an open-source labeling software that integrates both functionalities. SiLVi enables researchers to annotate behaviors and interactions directly within video data, generating structured outputs suitable for training and validating computer vision models. By linking behavioral ecology with computer vision, SiLVi facilitates the development of automated approaches for fine-grained behavioral analyses. Although developed primarily in the context of animal behavior, SiLVi could be useful more broadly to annotate human interactions in other videos that require extracting dynamic scene graphs. The software, along with documentation and download instructions, is available at: this https URL.
>
---
#### [replaced 021] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.00503](https://arxiv.org/pdf/2604.00503)**

> **作者:** Weifu Fu; Jinyang Li; Bin-Bin Gao; Jialin Li; Yuhuan Lin; Hanqiu Deng; Wenbing Tao; Yong Liu; Chengjie Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Open-Set Object Detection (OSOD) enables recognition of novel categories beyond fixed classes but faces challenges in aligning text representations with complex visual concepts and the scarcity of image-text pairs for rare categories. This results in suboptimal performance in specialized domains or with complex objects. Recent visual-prompted methods partially address these issues but often involve complex multi-modal designs and multi-stage optimizations, prolonging the development cycle. Additionally, effective training strategies for data-driven OSOD models remain largely unexplored. To address these challenges, we propose PET-DINO, a universal detector supporting both text and visual prompts. Our Alignment-Friendly Visual Prompt Generation (AFVPG) module builds upon an advanced text-prompted detector, addressing the limitations of text representation guidance and reducing the development cycle. We introduce two prompt-enriched training strategies: Intra-Batch Parallel Prompting (IBP) at the iteration level and Dynamic Memory-Driven Prompting (DMD) at the overall training level. These strategies enable simultaneous modeling of multiple prompt routes, facilitating parallel alignment with diverse real-world usage scenarios. Comprehensive experiments demonstrate that PET-DINO exhibits competitive zero-shot object detection capabilities across various prompt-based detection protocols. These strengths can be attributed to inheritance-based philosophy and prompt-enriched training strategies, which play a critical role in building an effective generic object detector. Project page: this https URL.
>
---
#### [replaced 022] MotionAdapter: Video Motion Transfer via Content-Aware Attention Customization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01955](https://arxiv.org/pdf/2601.01955)**

> **作者:** Zhexin Zhang; Yangyang Xu; Yifeng Zhu; Long Chen; Yong Du; Shengfeng He; Jun Yu
>
> **摘要:** Recent advances in diffusion-based text-to-video models, particularly those built on the diffusion transformer architecture, have achieved remarkable progress in generating high-quality and temporally coherent videos. However, transferring complex motions between videos remains challenging. In this work, we present MotionAdapter, a content-aware motion transfer framework that enables robust and semantically aligned motion transfer within DiT-based video diffusion models. Our key insight is that effective motion transfer requires 1) explicit disentanglement of motion from appearance and 2) adaptive customization of motion to target content. MotionAdapter first isolates motion by analyzing cross-frame attention within 3D full-attention modules to extract attention-derived motion fields. To bridge the semantic gap between reference and target videos, we further introduce a DINO-guided motion customization module that rearranges and refines motion fields based on content correspondences. The customized motion field is then used to guide the DiT denoising process, ensuring that the synthesized video inherits the reference motion while preserving target appearance and semantics. Extensive experiments demonstrate that MotionAdapter outperforms state-of-the-art methods in both qualitative and quantitative evaluations. Moreover, MotionAdapter naturely support complex motion transfer and motion editing tasks such as zooming in/out and composition.
>
---
#### [replaced 023] ReMemNav: A Rethinking and Memory-Augmented Framework for Zero-Shot Object Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知环境中定位未见过目标的问题。提出ReMemNav框架，结合视觉语言模型与记忆机制，提升导航成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.26788](https://arxiv.org/pdf/2603.26788)**

> **作者:** Feng Wu; Wei Zuo; Wenliang Yang; Jun Xiao; Yang Liu; Xinhua Zeng
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Zero-shot object navigation requires agents to locate unseen target objects in unfamiliar environments without prior maps or task-specific training which remains a significant challenge. Although recent advancements in vision-language models(VLMs) provide promising commonsense reasoning capabilities for this task, these models still suffer from spatial hallucinations, local exploration deadlocks, and a disconnect between high-level semantic intent and low-level control. In this regard, we propose a novel hierarchical navigation framework named ReMemNav, which seamlessly integrates panoramic semantic priors and episodic memory with VLMs. We introduce the Recognize Anything Model to anchor the spatial reasoning process of the VLM. We also design an adaptive dual-modal rethinking mechanism based on an episodic semantic buffer queue. The proposed mechanism actively verifies target visibility and corrects decisions using historical memory to prevent deadlocks. For low-level action execution, ReMemNav extracts a sequence of feasible actions using depth masks, allowing the VLM to select the optimal action for mapping into actual spatial movement. Extensive evaluations on HM3D and MP3D demonstrate that ReMemNav outperforms existing training-free zero-shot baselines in both success rate and exploration efficiency. Specifically, we achieve significant absolute performance improvements, with SR and SPL increasing by 1.7% and 7.0% on HM3D v0.1, 18.2% and 11.1% on HM3D v0.2, and 8.7% and 7.9% on MP3D.
>
---
#### [replaced 024] MARS: Multi-Agent Robotic System with Multimodal Large Language Models for Assistive Intelligence
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MARS系统，解决智能助人机器人在复杂家庭环境中的风险规划与个性化问题，通过多智能体协作实现高效任务执行。**

- **链接: [https://arxiv.org/pdf/2511.01594](https://arxiv.org/pdf/2511.01594)**

> **作者:** Renjun Gao
>
> **备注:** 3 figures, 1 table
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable capabilities in cross-modal understanding and reasoning, offering new opportunities for intelligent assistive systems, yet existing systems still struggle with risk-aware planning, user personalization, and grounding language plans into executable skills in cluttered homes. We introduce MARS - a Multi-Agent Robotic System powered by MLLMs for assistive intelligence and designed for smart home robots supporting people with disabilities. The system integrates four agents: a visual perception agent for extracting semantic and spatial features from environment images, a risk assessment agent for identifying and prioritizing hazards, a planning agent for generating executable action sequences, and an evaluation agent for iterative optimization. By combining multimodal perception with hierarchical multi-agent decision-making, the framework enables adaptive, risk-aware, and personalized assistance in dynamic indoor environments. Experiments on multiple datasets demonstrate the superior overall performance of the proposed system in risk-aware planning and coordinated multi-agent execution compared with state-of-the-art multimodal models. The proposed approach also highlights the potential of collaborative AI for practical assistive scenarios and provides a generalizable methodology for deploying MLLM-enabled multi-agent systems in real-world environments.
>
---
#### [replaced 025] ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ProMQA-Assembly数据集，用于评估装配任务中的多模态问答系统。解决装配活动评估资源不足的问题，通过半自动标注和细粒度动作标签生成QA对，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.02949](https://arxiv.org/pdf/2509.02949)**

> **作者:** Kimihiro Hasegawa; Wiradee Imrattanatrai; Masaki Asada; Susan Holm; Yuran Wang; Vincent Zhou; Ken Fukuda; Teruko Mitamura
>
> **备注:** LREC 2026. Code and data: this https URL
>
> **摘要:** Assistants on assembly tasks show great potential to benefit humans ranging from helping with everyday tasks to interacting in industrial settings. However, evaluation resources in assembly activities are underexplored. To foster system development, we propose a new multimodal QA evaluation dataset on assembly activities. Our dataset, ProMQA-Assembly, consists of 646 QA pairs that require multimodal understanding of human activity videos and their instruction manuals in an online-style manner. For cost effectiveness in the data creation, we adopt a semi-automated QA annotation approach, where LLMs generate candidate QA pairs and humans verify them. We further improve QA generation by integrating fine-grained action labels to diversify question types. Additionally, we create 81 instruction task graphs for our target assembly tasks. These newly created task graphs are used in our benchmarking experiment, as well as in facilitating the human verification process. With our dataset, we benchmark models, including competitive proprietary multimodal models. We find that ProMQA-Assembly contains challenging multimodal questions, where reasoning models showcase promising results. We believe our new evaluation dataset contributes to the further development of procedural-activity assistants.
>
---
#### [replaced 026] DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决动作解码中早期错误无法修正的问题。提出DFM-VLA框架，通过离散流匹配实现动作迭代优化，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2603.26320](https://arxiv.org/pdf/2603.26320)**

> **作者:** Jiayi Chen; Wenxuan Song; Shuai Chen; Jingbo Wang; Zhijun Li; Haoang Li
>
> **摘要:** Vision--Language--Action (VLA) models that encode actions using a discrete tokenization scheme are increasingly adopted for robotic manipulation, but existing decoding paradigms remain fundamentally limited. Whether actions are decoded sequentially by autoregressive VLAs or in parallel by discrete diffusion VLAs, once a token is generated, it is typically fixed and cannot be revised in subsequent iterations, so early token errors cannot be effectively corrected later. We propose DFM-VLA, a discrete flow matching VLA for iterative refinement of action tokens. DFM-VLA~models a token-level probability velocity field that dynamically updates the full action sequence across refinement iterations. We investigate two ways to construct the velocity field: an auxiliary velocity-head formulation and an action-embedding-guided formulation. Our framework further adopts a two-stage decoding strategy with an iterative refinement stage followed by deterministic validation for stable convergence. Extensive experiments on CALVIN, LIBERO, and real-world manipulation tasks show that DFM-VLA consistently outperforms strong autoregressive, discrete diffusion, and continuous diffusion baselines in manipulation performance while retaining high inference efficiency. In particular, DFM-VLA achieves an average success length of 4.44 on CALVIN and an average success rate of 95.7\% on LIBERO, highlighting the value of action refinement via discrete flow matching for robotic manipulation. Our project is available this https URL
>
---
#### [replaced 027] Tokenizing Buildings: A Transformer for Layout Synthesis
- **分类: cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.04832](https://arxiv.org/pdf/2512.04832)**

> **作者:** Manuel Ladron de Guevara; Jinmo Rhee; Ardavan Bidgoli; Vaidas Razgaitis; Michael Bergin
>
> **备注:** 14 pages, 3 page References, 4 figures
>
> **摘要:** We introduce Small Building Model (SBM), a Transformer-based architecture for layout synthesis in Building Information Modeling (BIM) scenes. We address the question of how to tokenize buildings by unifying heterogeneous feature sets of architectural elements into sequences while preserving compositional structure. Such feature sets are represented as a sparse attribute-feature matrix that captures room properties. We then design a unified embedding module that learns joint representations of categorical and possibly correlated continuous feature groups. Lastly, we train a single Transformer backbone in two modes: an encoder-only pathway that yields high-fidelity room embeddings, and an encoder-decoder pipeline for autoregressive prediction of residential room entities, referred to as Data-Driven Entity Prediction (DDEP). Experiments across retrieval and generative layout synthesis show that SBM learns compact room embeddings that reliably cluster by type and topology, enabling strong semantic retrieval. In DDEP mode, SBM produces functionally sound layouts with fewer collisions and boundary violations, and improved navigability, outperforming general-purpose LLM/VLM baselines and recent domain-specific methods.
>
---
#### [replaced 028] A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00978](https://arxiv.org/pdf/2510.00978)**

> **作者:** Axel Barroso-Laguna; Tommaso Cavallari; Victor Adrian Prisacariu; Eric Brachmann
>
> **摘要:** Visually localizing an image, i.e., estimating its camera pose, requires building a scene representation that serves as a visual map. The representation we choose has direct consequences towards the practicability of our system. Even when starting from mapping images with known camera poses, state-of-the-art approaches still require hours of mapping time in the worst case, and several minutes in the best. This work raises the question whether we can achieve competitive accuracy much faster. We introduce FastForward, a method that creates a map representation and relocalizes a query image on-the-fly in a single feed-forward pass. At the core, we represent multiple mapping images as a collection of features anchored in 3D space. FastForward utilizes these mapping features to predict image-to-scene correspondences for the query image, enabling the estimation of its camera pose. We couple FastForward with image retrieval and achieve state-of-the-art accuracy when compared to other approaches with minimal map preparation time. Furthermore, FastForward demonstrates robust generalization to unseen domains, including challenging large-scale outdoor environments.
>
---
#### [replaced 029] MedShift: Implicit Conditional Transport for X-Ray Domain Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.21435](https://arxiv.org/pdf/2508.21435)**

> **作者:** Francisco Caetano; Christiaan Viviers; Peter H.N. De With; Fons van der Sommen
>
> **备注:** Accepted at the ICCV 2025 AIM Workshop
>
> **摘要:** Synthetic medical data offers a scalable solution for training robust models, but significant domain gaps limit its generalizability to real-world clinical settings. This paper addresses the challenge of cross-domain translation between synthetic and real X-ray images of the head, focusing on bridging discrepancies in attenuation behavior, noise characteristics, and soft tissue representation. We propose MedShift, a unified class-conditional generative model based on Flow Matching and Schrodinger Bridges, which enables high-fidelity, unpaired image translation across multiple domains. Unlike prior approaches that require domain-specific training or rely on paired data, MedShift learns a shared domain-agnostic latent space and supports seamless translation between any pair of domains seen during training. We introduce X-DigiSkull, a new dataset comprising aligned synthetic and real skull X-rays under varying radiation doses, to benchmark domain translation models. Experimental results demonstrate that, despite its smaller model size compared to diffusion-based approaches, MedShift offers strong performance and remains flexible at inference time, as it can be tuned to prioritize either perceptual fidelity or structural consistency, making it a scalable and generalizable solution for domain adaptation in medical imaging. The code and dataset are available at this https URL
>
---
#### [replaced 030] Matrix-game 2.0: An open-source real-time and streaming interactive world model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13009](https://arxiv.org/pdf/2508.13009)**

> **作者:** Xianglong He; Chunli Peng; Zexiang Liu; Boyang Wang; Yifan Zhang; Qi Cui; Fei Kang; Biao Jiang; Mengyin An; Yangyang Ren; Baixin Xu; Hao-Xiang Guo; Kaixiong Gong; Size Wu; Wei Li; Xuchen Song; Yang Liu; Yangguang Li; Yahui Zhou
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent advances in interactive video generations have demonstrated diffusion model's potential as world models by capturing complex physical dynamics and interactive behaviors. However, existing interactive world models depend on bidirectional attention and lengthy inference steps, severely limiting real-time performance. Consequently, they are hard to simulate real-world dynamics, where outcomes must update instantaneously based on historical context and current actions. To address this, we present Matrix-Game 2.0, an interactive world model generates long videos on-the-fly via few-step auto-regressive diffusion. Our framework consists of three key components: (1) A scalable data production pipeline for Unreal Engine and GTA5 environments to effectively produce massive amounts (about 1200 hours) of video data with diverse interaction annotations; (2) An action injection module that enables frame-level mouse and keyboard inputs as interactive conditions; (3) A few-step distillation based on the casual architecture for real-time and streaming video generation. Matrix Game 2.0 can generate high-quality minute-level videos across diverse scenes at an ultra-fast speed of 25 FPS. We open-source our model weights and codebase to advance research in interactive world modeling.
>
---
#### [replaced 031] OmniFysics: Towards Physical Intelligence Evolution via Omni-Modal Signal Processing and Network Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07064](https://arxiv.org/pdf/2602.07064)**

> **作者:** Minghao Han; Dingkang Yang; Yue Jiang; Yizhou Liu; Lihua Zhang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The autonomous evolution of networked AI systems relies heavily on robust environmental perception. However, physical understanding remains brittle in current models because key physical signals are visually ambiguous and sparsely represented in web-scale data. To bridge the gap between data-centric learning and knowledge-based physical rules, we present OmniFysics, a compact omni-modal network that unifies signal processing and understanding across images, audio, video, and text. To enable autonomous optimization and inject explicit physical knowledge, we construct a dynamic physical data engine. Within this engine, FysicsAny acts as an adaptive mechanism that produces physics-grounded supervision by mapping salient objects to verified physical attributes via hierarchical retrieval and physics-law-constrained signal verification. Concurrently, FysicsOmniCap distills web videos utilizing advanced audio-visual cross-modal signal processing, generating high-fidelity data pairs that emphasize dynamic physical cues. We optimize the OmniFysics network through staged multimodal alignment and evolutive instruction tuning, integrating latent-space flow matching for generation and an adaptive intent router for efficient execution. Experiments demonstrate that this evolutive optimization paradigm not only achieves competitive performance on standard multimodal benchmarks but also significantly advances physics-oriented evaluations.
>
---
#### [replaced 032] MAMMA: Markerless & Automatic Multi-Person Motion Action Capture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13040](https://arxiv.org/pdf/2506.13040)**

> **作者:** Hanz Cuevas-Velasquez; Anastasios Yiannakidis; Soyong Shin; Giorgio Becherini; Markus Höschle; Joachim Tesch; Taylor Obersat; Tsvetelina Alexiadis; Eni Halilaj; Michael J. Black
>
> **备注:** Main paper and supplementary material
>
> **摘要:** We present MAMMA, a markerless motion-capture pipeline that accurately recovers SMPL-X parameters from multi-view video of two-person interaction sequences. Traditional motion-capture systems rely on physical markers. Although they offer high accuracy, their requirements of specialized hardware, manual marker placement, and extensive post-processing make them costly and time-consuming. Recent learning-based methods attempt to overcome these limitations, but most are designed for single-person capture, rely on sparse keypoints, or struggle with occlusions and physical interactions. In this work, we introduce a method that predicts dense 2D contact-aware surface landmarks conditioned on segmentation masks, enabling person-specific correspondence estimation even under heavy occlusion. We employ a novel architecture that exploits learnable queries for each landmark. We demonstrate that our approach can handle complex person--person interaction and offers greater accuracy than existing methods. To train our network, we construct a large, synthetic multi-view dataset combining human motions from diverse sources, including extreme poses, hand motions, and close interactions. Our dataset yields high-variability synthetic sequences with rich body contact and occlusion, and includes SMPL-X ground-truth annotations with dense 2D landmarks. The result is a system capable of capturing human motion without the need for markers. Our approach offers competitive reconstruction quality compared to commercial marker-based motion-capture solutions, without the extensive manual cleanup. Finally, we address the absence of common benchmarks for dense-landmark prediction and markerless motion capture by introducing two evaluation settings built from real multi-view sequences. Our dataset is available in this https URL for research purposes.
>
---
#### [replaced 033] MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文提出MetaEmbed，解决多模态检索中嵌入表达不足与效率问题，通过可扩展的多向量嵌入实现高效且精准的检索。**

- **链接: [https://arxiv.org/pdf/2509.18095](https://arxiv.org/pdf/2509.18095)**

> **作者:** Zilin Xiao; Qi Ma; Mengting Gu; Chun-cheng Jason Chen; Xintao Chen; Vicente Ordonez; Vijai Mohan
>
> **备注:** ICLR 2026 Oral
>
> **摘要:** Universal multimodal embedding models have achieved great success in capturing semantic relevance between queries and candidates. However, current methods either condense queries and candidates into a single vector, potentially limiting the expressiveness for fine-grained information, or produce too many vectors that are prohibitive for multi-vector retrieval. In this work, we introduce MetaEmbed, a new framework for multimodal retrieval that rethinks how multimodal embeddings are constructed and interacted with at scale. During training, a fixed number of learnable Meta Tokens are appended to the input sequence. At test-time, their last-layer contextualized representations serve as compact yet expressive multi-vector embeddings. Through the proposed Matryoshka Multi-Vector Retrieval training, MetaEmbed learns to organize information by granularity across multiple vectors. As a result, we enable test-time scaling in multimodal retrieval where users can balance retrieval quality against efficiency demands by selecting the number of tokens used for indexing and retrieval interactions. Extensive evaluations on the Massive Multimodal Embedding Benchmark (MMEB) and the Visual Document Retrieval Benchmark (ViDoRe) confirm that MetaEmbed achieves state-of-the-art retrieval performance while scaling robustly to models with 32B parameters. Code is available at this https URL.
>
---
#### [replaced 034] Time-reversed Flow Matching with Worst Transport in High-dimensional Latent Space for Image Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05461](https://arxiv.org/pdf/2508.05461)**

> **作者:** Liangwei Li; Lin Liu; Hanzhe Liang; Juanxiu Liu; Jing Zhang; Ruqian Hao; Xiaohui Du; Yong Liu; Pan Li
>
> **摘要:** Likelihood-based deep generative models have been widely investigated for Image Anomaly Detection (IAD), particularly Normalizing Flows, yet their strict architectural invertibility needs often constrain scalability, particularly in large-scale data regimes. Although time-parameterized Flow Matching (FM) serves as a scalable alternative, it remains computationally challenging in IAD due to the prohibitive costs of Jacobian-trace estimation. This paper proposes time-reversed Flow Matching (rFM), which shifts the objective from exact likelihood computation to evaluating target-domain regularity through density proxy estimation. We uncover two fundamental theoretical bottlenecks in this paradigm: first, the reversed vector field exhibits a non-Lipschitz singularity at the initial temporal boundary, precipitating explosive estimation errors. Second, the concentration of measure in high-dimensional Gaussian manifolds induces structured irregularities, giving rise to a Centripetal Potential Field (CPF) that steers trajectories away from Optimal Transport (OT) paths. We identify these observations as the inherent dualities between FM and rFM. To address these issues, we introduce local Worst Transport Flow matching (WT-Flow), which amplifies the observed CPF of rFM to mitigate the initial singularity while circumventing the need for exact distribution transformations via density proxy. Experiments on five datasets demonstrate that WT-Flow achieves state-of-the-art performance among single-scale flow-based methods, and competitive performance against leading multi-scale approaches. Furthermore, the proposed framework enables superior one-step inference, achieving a per-image flow latency of only 6.7 ms. Our code is available on this https URL.
>
---
#### [replaced 035] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.18601](https://arxiv.org/pdf/2506.18601)**

> **作者:** Denis Rozumny; Jonathon Luiten; Numair Khan; Johannes Schönberger; Peter Kontschieder
>
> **备注:** Accepted at CVPR 2026 Workshop "4D World Models: Bridging Generation and Reconstruction"
>
> **摘要:** Transforming casually captured, monocular videos into fully immersive dynamic experiences is a highly ill-posed task, and comes with significant challenges, e.g., reconstructing unseen regions, and dealing with the ambiguity in monocular depth estimation. In this work we introduce BulletGen, an approach that takes advantage of generative models to correct errors and complete missing information in a Gaussian-based dynamic scene representation. This is done by aligning the output of a diffusion-based video generation model with the 4D reconstruction at a single frozen "bullet-time" step. The generated frames are then used to supervise the optimization of the 4D Gaussian model. Our method seamlessly blends generative content with both static and dynamic scene components, achieving state-of-the-art results on both novel-view synthesis, and 2D/3D tracking tasks.
>
---
#### [replaced 036] From Evidence to Verdict: An Agent-Based Forensic Framework for AI-Generated Image Detection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2511.00181](https://arxiv.org/pdf/2511.00181)**

> **作者:** Mengfei Liang; Yiting Qu; Yukun Jiang; Michael Backes; Yang Zhang
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** The rapid evolution of AI-generated images poses growing challenges to information integrity and media authenticity. Existing detection approaches face limitations in robustness, interpretability, and generalization across diverse generative models, particularly when relying on a single source of visual evidence. We introduce AIFo (Agent-based Image Forensics), a training-free framework that formulates AI-generated image detection as a multi-stage forensic analysis process through multi-agent collaboration. The framework integrates a set of forensic tools, including reverse image search, metadata extraction, pre-trained classifiers, and vision-language model analysis, and resolves insufficient or conflicting evidence through a structured multi-agent debate mechanism. An optional memory-augmented module further enables the framework to incorporate information from historical cases. We evaluate AIFo on a benchmark of 6,000 images spanning controlled laboratory settings and challenging real-world scenarios, where it achieves 97.05% accuracy and consistently outperforms traditional classifiers and strong vision-language model baselines. These findings demonstrate the effectiveness of agent-based procedural reasoning for AI-generated image detection.
>
---
#### [replaced 037] Unrolling Graph-based Douglas-Rachford Algorithm for Image Interpolation with Informed Initialization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11926](https://arxiv.org/pdf/2509.11926)**

> **作者:** Xue Zhang; Bingshuo Hu; Gene Cheung
>
> **备注:** 6 pages,ICME2026
>
> **摘要:** Conventional deep neural nets (DNNs) initialize network parameters at random and then optimize each one via stochastic gradient descent (SGD), resulting in substantial risk of poor-performing local minima. Focusing on image interpolation and leveraging a recent theorem that maps a (pseudo-)linear interpolator {\Theta} to a directed graph filter that is a solution to a corresponding MAP problem with a graph shift variation (GSV) prior, we first initialize a directed graph adjacency matrix A given a known interpolator {\Theta}, establishing a baseline performance. Then, towards further gain, we learn perturbation matrices P and P(2) from data to augment A, whose restoration effects are implemented progressively via Douglas-Rachford (DR) iterations, which we unroll into a lightweight and interpretable neural net. Experiments on different image interpolation scenarios demonstrate state-of-the-art performance, while drastically reducing network parameters and inference complexity.
>
---
#### [replaced 038] MedGemma Technical Report
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MedGemma，一个基于Gemma的医学视觉-语言基础模型，旨在解决医疗AI训练与部署中的数据多样性和隐私问题，提升医学图像与文本理解能力。**

- **链接: [https://arxiv.org/pdf/2507.05201](https://arxiv.org/pdf/2507.05201)**

> **作者:** Andrew Sellergren; Sahar Kazemzadeh; Tiam Jaroensri; Atilla Kiraly; Madeleine Traverse; Timo Kohlberger; Shawn Xu; Fayaz Jamil; Cían Hughes; Charles Lau; Justin Chen; Fereshteh Mahvar; Liron Yatziv; Tiffany Chen; Bram Sterling; Stefanie Anna Baby; Susanna Maria Baby; Jeremy Lai; Samuel Schmidgall; Lu Yang; Kejia Chen; Per Bjornsson; Shashir Reddy; Ryan Brush; Kenneth Philbrick; Mercy Asiedu; Ines Mezerreg; Howard Hu; Howard Yang; Richa Tiwari; Sunny Jansen; Preeti Singh; Yun Liu; Shekoofeh Azizi; Aishwarya Kamath; Johan Ferret; Shreya Pathak; Nino Vieillard; Ramona Merhej; Sarah Perrin; Tatiana Matejovicova; Alexandre Ramé; Morgane Riviere; Louis Rouillard; Thomas Mesnard; Geoffrey Cideron; Jean-bastien Grill; Sabela Ramos; Edouard Yvinec; Michelle Casbon; Elena Buchatskaya; Jean-Baptiste Alayrac; Dmitry Lepikhin; Vlad Feinberg; Sebastian Borgeaud; Alek Andreev; Cassidy Hardin; Robert Dadashi; Léonard Hussenot; Armand Joulin; Olivier Bachem; Yossi Matias; Katherine Chou; Avinatan Hassidim; Kavi Goel; Clement Farabet; Joelle Barral; Tris Warkentin; Jonathon Shlens; David Fleet; Victor Cotruta; Omar Sanseviero; Gus Martins; Phoebe Kirk; Anand Rao; Shravya Shetty; David F. Steiner; Can Kirmizibayrak; Rory Pilgrim; Daniel Golden; Lin Yang
>
> **备注:** Fix references
>
> **摘要:** Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at this https URL.
>
---
#### [replaced 039] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04576](https://arxiv.org/pdf/2604.04576)**

> **作者:** Inseong Choi; Siwoo Lee; Seung-Hun Nam; Soohwan Song
>
> **备注:** Accepted at CVPR 2026. Project Page: this https URL
>
> **摘要:** Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results. The project page is available at this https URL.
>
---
#### [replaced 040] DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DVGT-2模型，解决自动驾驶中几何重建与轨迹规划问题，通过在线处理实现高效精准的3D几何与路径预测。**

- **链接: [https://arxiv.org/pdf/2604.00813](https://arxiv.org/pdf/2604.00813)**

> **作者:** Sicheng Zuo; Zixun Xie; Wenzhao Zheng; Shaoqing Xu; Fang Li; Hanbing Li; Long Chen; Zhi-Xin Yang; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks.
>
---
#### [replaced 041] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13182](https://arxiv.org/pdf/2603.13182)**

> **作者:** Hiba Adil Al-kharsan; Róbert Rajkó
>
> **备注:** 30 pages, 29 figures
>
> **摘要:** Brain tumor classification from magnetic resonance imaging, which is also known as MRI, plays a sensitive role in computer-assisted diagnosis systems. In recent years, deep learning models have achieved high classification accuracy. However, their sensitivity to adversarial perturbations has become an important reliability concern in medical applications. This study suggests a robust brain tumor classification framework that combines Non-Negative Matrix Factorization (NNMF or NMF), lightweight convolutional neural networks (CNNs), and diffusion-based feature purification. Initially, MRI images are preprocessed and converted into a non-negative data matrix, from which compact and interpretable NNMF feature representations are extracted. Statistical metrics, including AUC, Cohen's d, and p-values, are used to rank and choose the most discriminative components. Then, a lightweight CNN classifier is trained directly on the selected feature groups. To improve adversarial robustness, a diffusion-based feature-space purification module is introduced. A forward noise method followed by a learned denoiser network is used before classification. System performance is estimated using both clean accuracy and robust accuracy under powerful adversarial attacks created by AutoAttack. The experimental results show that the proposed framework achieves competitive classification performance while significantly enhancing robustness against adversarial this http URL findings presuppose that combining interpretable NNMF-based representations with a lightweight deep approach and diffusion-based defense technique supplies an effective and reliable solution for medical image classification under adversarial conditions.
>
---
#### [replaced 042] The DeepSpeak Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.05366](https://arxiv.org/pdf/2408.05366)**

> **作者:** Sarah Barrington; Maty Bohacek; Hany Farid
>
> **备注:** this https URL
>
> **摘要:** Deepfakes represent a growing concern across domains such as disinformation, fraud, and non-consensual media. In particular, the rise of video conference and identity-driven attacks in high-stakes scenarios--such as impostor hiring--demands new forensic resources. Despite significant efforts to develop robust detection classifiers to distinguish the real from the fake, commonly used training datasets remain inadequate: relying on low-quality and outdated deepfake generators, consisting of content scraped from online repositories without participant consent, lacking in multimodal coverage, and rarely employing identity-matching protocols to ensure realistic fakes. To overcome these limitations, we present the DeepSpeak dataset, a diverse and multimodal dataset comprising over 100 hours of authentic and deepfake audiovisual content, specifically focused on the challenging and diverse talking heads context. We contribute: i) more than 50 hours of real, self-recorded data collected from 500 diverse and consenting participants, ii) more than 50 hours of state-of-the-art audio and visual deepfakes generated using 14 video synthesis engines and three voice cloning engines, and iii) an embedding-based, identity-matching approach to ensure the creation of convincing, high-quality identity face swaps that realistically simulate adversarial deepfake attacks. We also perform large-scale evaluations of state-of-the-art deepfake detectors and show that, without retraining, these detectors fail to generalize to this DeepSpeak dataset, highlighting the importance of a large and diverse dataset containing deepfakes from the latest generative-AI tools.
>
---
#### [replaced 043] Online In-Context Distillation for Low-Resource Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18117](https://arxiv.org/pdf/2510.18117)**

> **作者:** Zhiqi Kang; Rahaf Aljundi; Vaggelis Dorovatas; Karteek Alahari
>
> **摘要:** As the field continues its push for ever more resources, this work turns the spotlight on a critical question: how can vision-language models (VLMs) be adapted to thrive in low-resource, budget-constrained settings? While large VLMs offer strong performance, they are impractical to deploy in such settings. Small VLMs, on the other hand, are efficient but typically require costly fine-tuning to close the performance gap with larger models in the deployment domain. Inspired by the in-context learning framework, we propose an online In-Context Distillation (ICD) method, in which a small VLM collaborates with a stronger teacher model at inference time, distilling its knowledge via sparse demonstrations to efficiently bridge the gap between them. Our method is built on an in-depth analysis that identifies the scale and the choice of models for which vision-language ICL is currently feasible, and demonstrates the advantage of ICL over fine-tuning under constrained compute budgets. We enhance our method with a novel cross-modal demonstration selection strategy, teacher test-time scaling to reduce noise, and student uncertainty conditioning to dynamically populate a demonstration pool and minimize teacher queries. Our ICD method significantly boosts the performance of small models (up to 33%) using scarce teacher annotations (as low as 4%), and competes with the teacher's zero-shot performance.
>
---
#### [replaced 044] Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.06162](https://arxiv.org/pdf/2601.06162)**

> **作者:** Kaiyuan Deng; Gen Li; Yang Xiao; Bo Hui; Xiaolong Ma
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Text-to-image diffusion models have achieved remarkable progress, yet their use raises copyright and misuse concerns, prompting research into machine unlearning. However, extending multi-concept unlearning to large-scale scenarios remains difficult due to three challenges: (i) conflicting weight updates that hinder unlearning or degrade generation; (ii) imprecise mechanisms that cause collateral damage to similar content; and (iii) reliance on additional data or modules, creating scalability bottlenecks. To address these, we propose Scalable-Precise Concept Unlearning (ScaPre), a unified framework tailored for large-scale unlearning. ScaPre introduces a conflict-aware stable design, integrating spectral trace regularization and geometry alignment to stabilize optimization, suppress conflicts, and preserve global structure. Furthermore, an Informax Decoupler identifies concept-relevant parameters and adaptively reweights updates, strictly confining unlearning to the target subspace. ScaPre yields an efficient closed-form solution without requiring auxiliary data or sub-models. Comprehensive experiments on objects, styles, and explicit content demonstrate that ScaPre effectively removes target concepts while maintaining generation quality. It forgets up to $\times \mathbf{5}$ more concepts than the best baseline within acceptable quality limits, achieving state-of-the-art precision and efficiency for large-scale unlearning.
>
---
#### [replaced 045] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.00727](https://arxiv.org/pdf/2412.00727)**

> **作者:** Naman Deep Singh; Francesco Croce; Matthias Hein
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Vision-Language models like CLIP have been shown to be highly effective at linking visual perception and natural language understanding, enabling sophisticated image-text capabilities, including strong retrieval and zero-shot classification performance. Their widespread use, as well as the fact that CLIP models are trained on image-text pairs from the web, make them both a worthwhile and relatively easy target for backdoor attacks. As training foundational models, such as CLIP, from scratch is very expensive, this paper focuses on cleaning potentially poisoned models via fine-tuning. We first show that existing cleaning techniques are not effective against simple structured triggers used in Blended or BadNet backdoor attacks, exposing a critical vulnerability for potential real-world deployment of these models. Then, we introduce PAR, Perturb and Recover, a surprisingly simple yet effective mechanism to remove backdoors from CLIP models. Through extensive experiments across different encoders and types of backdoor attacks, we show that PAR achieves high backdoor removal rate while preserving good standard performance. Finally, we illustrate that our approach is effective even only with synthetic text-image pairs, i.e. without access to real training data. The code and models are available on \href{this https URL}{GitHub}.
>
---
#### [replaced 046] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.26481](https://arxiv.org/pdf/2603.26481)**

> **作者:** Weihong Pan; Xiaoyu Zhang; Zhuang Zhang; Zhichao Ye; Nan Wang; Haomin Liu; Guofeng Zhang
>
> **备注:** CVPR 2026. Project page: this https URL and code: this https URL
>
> **摘要:** High-quality 4D reconstruction enables photorealistic and immersive rendering of the dynamic real world. However, unlike static scenes that can be fully captured with a single camera, high-quality dynamic scenes typically require dense arrays of tens or even hundreds of synchronized cameras. Dependence on such costly lab setups severely limits practical scalability. To this end, we propose a sparse-camera dynamic reconstruction framework that exploits abundant yet inconsistent generative observations. Our key innovation is the Spatio-Temporal Distortion Field, which provides a unified mechanism for modeling inconsistencies in generative observations across both spatial and temporal dimensions. Building on this, we develop a complete pipeline that enables 4D reconstruction from sparse and uncalibrated camera inputs. We evaluate our method on multi-camera dynamic scene benchmarks, achieving spatio-temporally consistent high-fidelity renderings and significantly outperforming existing approaches. Project page available at this https URL
>
---
#### [replaced 047] Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04746](https://arxiv.org/pdf/2604.04746)**

> **作者:** Lei Zhang; Junjiao Tian; Zhipeng Fan; Kunpeng Li; Jialiang Wang; Weifeng Chen; Markos Georgopoulos; Felix Juefei-Xu; Yuxiang Bao; Julian McAuley; Manling Li; Zecheng He
>
> **摘要:** Humans paint images incrementally: they plan a global layout, sketch a coarse draft, inspect, and refine details, and most importantly, each step is grounded in the evolving visual states. However, can unified multimodal models trained on text-image interleaved datasets also imagine the chain of intermediate states? In this paper, we introduce process-driven image generation, a multi-step paradigm that decomposes synthesis into an interleaved reasoning trajectory of thoughts and actions. Rather than generating images in a single step, our approach unfolds across multiple iterations, each consisting of 4 stages: textual planning, visual drafting, textual reflection, and visual refinement. The textual reasoning explicitly conditions how the visual state should evolve, while the generated visual intermediate in turn constrains and grounds the next round of textual reasoning. A core challenge of process-driven generation stems from the ambiguity of intermediate states: how can models evaluate each partially-complete image? We address this through dense, step-wise supervision that maintains two complementary constraints: for the visual intermediate states, we enforce the spatial and semantic consistency; for the textual intermediate states, we preserve the prior visual knowledge while enabling the model to identify and correct prompt-violating elements. This makes the generation process explicit, interpretable, and directly supervisable. To validate proposed method, we conduct experiments under various text-to-image generation benchmarks.
>
---
#### [replaced 048] TRANSPORTER: Transferring Visual Semantics from VLM Manifolds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18359](https://arxiv.org/pdf/2511.18359)**

> **作者:** Alexandros Stergiou
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026, Project page: this https URL
>
> **摘要:** How do video understanding models acquire their answers? Although current Vision Language Models (VLMs) reason over complex scenes with diverse objects, action performances, and scene dynamics, understanding and controlling their internal processes remains an open challenge. Motivated by recent advancements in text-to-video (T2V) generative models, this paper introduces a logits-to-video (L2V) task alongside a model-independent approach, TRANSPORTER, to generate videos that capture the underlying rules behind VLMs' predictions. Given the high-visual-fidelity produced by T2V models, TRANSPORTER learns an optimal transport coupling to VLM's high-semantic embedding spaces. In turn, logit scores define embedding directions for conditional video generation. TRANSPORTER generates videos that reflect caption changes over diverse object attributes, action adverbs, and scene context. Quantitative and qualitative evaluations across VLMs demonstrate that L2V can provide a fidelity-rich, novel direction for model interpretability that has not been previously explored.
>
---
#### [replaced 049] Image Diffusion Preview with Consistency Solver
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13592](https://arxiv.org/pdf/2512.13592)**

> **作者:** Fu-Yun Wang; Hao Zhou; Liangzhe Yuan; Sanghyun Woo; Boqing Gong; Bohyung Han; Ming-Hsuan Yang; Han Zhang; Yukun Zhu; Ting Liu; Long Zhao
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The slow inference process of image diffusion models significantly degrades interactive user experiences. To address this, we introduce Diffusion Preview, a novel paradigm employing rapid, low-step sampling to generate preliminary outputs for user evaluation, deferring full-step refinement until the preview is deemed satisfactory. Existing acceleration methods, including training-free solvers and post-training distillation, struggle to deliver high-quality previews or ensure consistency between previews and final outputs. We propose ConsistencySolver derived from general linear multistep methods, a lightweight, trainable high-order solver optimized via Reinforcement Learning, that enhances preview quality and consistency. Experimental results demonstrate that ConsistencySolver significantly improves generation quality and consistency in low-step scenarios, making it ideal for efficient preview-and-refine workflows. Notably, it achieves FID scores on-par with Multistep DPM-Solver using 47% fewer steps, while outperforming distillation baselines. Furthermore, user studies indicate our approach reduces overall user interaction time by nearly 50% while maintaining generation quality. Code is available at this https URL.
>
---
#### [replaced 050] Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，解决对抗扰动导致的模型脆弱性问题。通过引入Sim-CLIP框架，提升CLIP编码器的鲁棒性并保持语义一致性。**

- **链接: [https://arxiv.org/pdf/2407.14971](https://arxiv.org/pdf/2407.14971)**

> **作者:** Md Zarif Hossain; Ahmed Imteaj
>
> **备注:** Accepted at IJCNN 2026
>
> **摘要:** Vision-Language Models (VLMs) rely heavily on pretrained vision encoders to support downstream tasks such as image captioning, visual question answering, and zero-shot classification. Despite their strong performance, these encoders remain highly vulnerable to imperceptible adversarial perturbations, which can severely degrade both robustness and semantic quality in multimodal reasoning. In this work, we introduce Sim-CLIP, an unsupervised adversarial fine-tuning framework that enhances the robustness of the CLIP vision encoder while preserving overall semantic representations. Sim-CLIP adopts a Siamese training architecture with a cosine similarity objective and a symmetric stop-gradient mechanism to enforce semantic alignment between clean and adversarial views. This design avoids large-batch contrastive learning and additional momentum encoders, enabling robust training with low computational overhead. We evaluate Sim-CLIP across multiple Vision-Language Models and tasks under both targeted and untargeted adversarial attacks. Experimental results demonstrate that Sim-CLIP consistently outperforms state-of-the-art robust CLIP variants, achieving stronger adversarial robustness while maintaining or improving semantic fidelity. These findings highlight the limitations of existing adversarial defenses and establish Sim-CLIP as an effective and scalable solution for robust vision-language representation learning.
>
---
#### [replaced 051] Large-scale Codec Avatars: The Unreasonable Effectiveness of Large-scale Avatar Pretraining
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.02320](https://arxiv.org/pdf/2604.02320)**

> **作者:** Junxuan Li; Rawal Khirodkar; Chengan He; Zhongshi Jiang; Giljoo Nam; Lingchen Yang; Jihyun Lee; Egor Zakharov; Zhaoen Su; Rinat Abdrashitov; Yuan Dong; Julieta Martinez; Kai Li; Qingyang Tan; Takaaki Shiratori; Matthew Hu; Peihong Guo; Xuhua Huang; Ariyan Zarei; Marco Pesavento; Yichen Xu; He Wen; Teng Deng; Wyatt Borsos; Anjali Thakrar; Jean-Charles Bazin; Carsten Stoll; Ginés Hidalgo; James Booth; Lucy Wang; Xiaowen Ma; Yu Rong; Sairanjith Thalanki; Chen Cao; Christian Häne; Abhishek Kar; Sofien Bouaziz; Jason Saragih; Yaser Sheikh; Shunsuke Saito
>
> **备注:** Accepted in CVPR2026. Website: this https URL
>
> **摘要:** High-quality 3D avatar modeling faces a critical trade-off between fidelity and generalization. On the one hand, multi-view studio data enables high-fidelity modeling of humans with precise control over expressions and poses, but it struggles to generalize to real-world data due to limited scale and the domain gap between the studio environment and the real world. On the other hand, recent large-scale avatar models trained on millions of in-the-wild samples show promise for generalization across a wide range of identities, yet the resulting avatars are often of low-quality due to inherent 3D ambiguities. To address this, we present Large-Scale Codec Avatars (LCA), a high-fidelity, full-body 3D avatar model that generalizes to world-scale populations in a feedforward manner, enabling efficient inference. Inspired by the success of large language models and vision foundation models, we present, for the first time, a pre/post-training paradigm for 3D avatar modeling at scale: we pretrain on 1M in-the-wild videos to learn broad priors over appearance and geometry, then post-train on high-quality curated data to enhance expressivity and fidelity. LCA generalizes across hair styles, clothing, and demographics while providing precise, fine-grained facial expressions and finger-level articulation control, with strong identity preservation. Notably, we observe emergent generalization to relightability and loose garment support to unconstrained inputs, and zero-shot robustness to stylized imagery, despite the absence of direct supervision.
>
---
#### [replaced 052] Aligned Vector Quantization for Edge-Cloud Collabrative Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.05961](https://arxiv.org/pdf/2411.05961)**

> **作者:** Xiao Liu; Lijun Zhang; Deepak Ganesan; Hui Guan
>
> **备注:** I found a big mistake in the paper that causes significant bias on the results. The residual links are not taken into consideration when computing the transmission. All results about the compressed data size and transmission latency would be affected
>
> **摘要:** Vision Language Models (VLMs) are central to Visual Question Answering (VQA) systems and are typically deployed in the cloud due to their high computational demands. However, this cloud-only approach underutilizes edge computational resources and requires significant bandwidth for transmitting raw images. In this paper, we introduce an edge-cloud collaborative VQA system, called LLaVA-AlignedVQ, which features a novel Aligned Vector Quantization algorithm (AlignedVQ) that efficiently compress intermediate features without compromising accuracy to support partitioned execution. Our experiments demonstrate that LLaVA-AlignedVQ achieves approximately 1365x compression rate of intermediate features, reducing data transmission overhead by 96.8% compared to transmitting JPEG90-compressed images to the cloud. LLaVA-AlignedVQ achieves an inference speedup of 2-15x while maintaining high accuracy, remaining within -2.23% to +1.6% of the original model's accuracy performance across eight VQA datasets, compared to the cloud-only solution.
>
---
#### [replaced 053] Image-Based Metrics in Ultrasound for Estimation of Global Speed-of-Sound
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2503.14094](https://arxiv.org/pdf/2503.14094)**

> **作者:** Roman Denkin; Orcun Goksel
>
> **摘要:** Accurate speed-of-sound (SoS) estimation is crucial for ultrasound image formation, yet conventional systems often rely on an assumed value for imaging. We propose to leverage conventional image analysis techniques and metrics as a novel and simple approach to estimate tissue SoS. We study eleven metrics in three categories for assessing image quality, image similarity and multi-frame variation, by testing them in numerical simulations and phantom experiments, as well as testing in an in vivo scenario. Among single-frame image quality metrics, conventional Focus and a proposed metric variation on Tenengrad present satisfactory accuracy (5-8\,m/s on phantoms), but only when the metrics are applied after compounding multiple frames. Differential image comparison metrics were more successful overall with errors consistently under 8\,m/s even applied on a single pair of frames. Mutual information and correlation metrics were found to be robust in processing relatively small image patches, making them suitable for focal estimation. We present an in vivo study on breast density classification based on SoS, to showcase clinical applicability. The studied metrics do not require access to raw channel data as they can operate on post-beamformed and/or B-mode data. These image-based methods offer a computationally efficient and data-accessible alternative to existing physics- and model-based approaches for SoS estimation.
>
---
#### [replaced 054] GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoPredict，解决机器人操作中3D推理不足的问题。通过引入预测性运动和几何模块，提升VLA模型在复杂空间任务中的表现。**

- **链接: [https://arxiv.org/pdf/2512.16811](https://arxiv.org/pdf/2512.16811)**

> **作者:** Jingjing Qian; Boyao Han; Chen Shi; Lei Xiao; Long Yang; Shaoshuai Shi; Li Jiang
>
> **摘要:** Vision-Language-Action (VLA) models achieve strong generalization in robotic manipulation but remain largely reactive and 2D-centric, making them unreliable in tasks that require precise 3D reasoning. We propose GeoPredict, a geometry-aware VLA framework that augments a continuous-action policy with predictive kinematic and geometric priors. GeoPredict introduces a trajectory-level module that encodes motion history and predicts multi-step 3D keypoint trajectories of robot arms, and a predictive 3D Gaussian geometry module that forecasts workspace geometry with track-guided refinement along future keypoint trajectories. These predictive modules serve exclusively as training-time supervision through depth-based rendering, while inference requires only lightweight additional query tokens without invoking any 3D decoding. Experiments on RoboCasa Human-50, LIBERO, and real-world manipulation tasks show that GeoPredict consistently outperforms strong VLA baselines, especially in geometry-intensive and spatially demanding scenarios.
>
---
#### [replaced 055] Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset
- **分类: cs.GR; cs.CV; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法语义控制不足和长序列不连贯的问题。提出LRCM框架，结合扩散模型与Mamba模块，实现多模态引导的自回归舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2601.03323](https://arxiv.org/pdf/2601.03323)**

> **作者:** Oran Duan; Yinghua Shen; Yingzhu Lv; Luyang Jie; Yaxin Liu; Qiong Wu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Advances in generative models and sequence learning have greatly promoted research in dance motion generation, yet current methods still suffer from coarse semantic control and poor coherence in long sequences. In this work, we present Listen to Rhythm, Choose Movements (LRCM), a multimodal-guided diffusion framework supporting both diverse input modalities and autoregressive dance motion generation. We explore a feature decoupling paradigm for dance datasets and generalize it to the Motorica Dance dataset, separating motion capture data, audio rhythm, and professionally annotated global and local text descriptions. Our diffusion architecture integrates an audio-latent Conformer and a text-latent Cross-Conformer, and incorporates a Motion Temporal Mamba Module (MTMM) to enable smooth, long-duration autoregressive synthesis. Experimental results indicate that LRCM delivers strong performance in both functional capability and quantitative metrics, demonstrating notable potential in multimodal input scenarios and extended sequence generation. The project page is available at this https URL.
>
---
#### [replaced 056] Towards Robust and Realistic Human Pose Estimation via WiFi Signals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.09411](https://arxiv.org/pdf/2501.09411)**

> **作者:** Yang Chen; Jingcai Guo
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Robust WiFi-based human pose estimation (HPE) is a challenging task that bridges discrete and subtle WiFi signals to human skeletons. We revisit this problem and reveal two critical yet overlooked issues: 1) cross-domain gap, i.e., due to significant discrepancies in pose distributions between source and target domains; and 2) structural fidelity gap, i.e., predicted skeletal poses manifest distorted topology, usually with misplaced joints and disproportionate bone lengths. This paper fills these gaps by reformulating the task into a novel two-phase framework dubbed DT-Pose: Domain-consistent representation learning and Topology-constrained Pose decoding. Concretely, we first propose a temporal consistency contrastive learning strategy with uniformity regularization, integrated into a self-supervised masked pretraining paradigm. This design facilitates robust learning of domain-consistent and motion-discriminative WiFi representations while mitigating potential mode collapse caused by signal sparsity. Beyond this, we introduce an effective hybrid decoding architecture that incorporates explicit skeletal topology constraints. By compensating for the inherent absence of spatial priors in WiFi semantic vectors, the decoder enables structured modeling of both adjacent and overarching joint relationships, producing more realistic pose predictions. Extensive experiments conducted on various benchmark datasets highlight the superior performance of our method in tackling these fundamental challenges in 2D/3D WiFi-based HPE tasks. The associated code is released at this https URL.
>
---
#### [replaced 057] Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04579](https://arxiv.org/pdf/2604.04579)**

> **作者:** Quoc-Huy Trinh; Mustapha Abdullahi; Bo Zhao; Debesh Jha
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2511.11177
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have enabled impressive progress in vision-language understanding, yet their high computational cost limits deployment in resource-constrained scenarios such as personal assistants, document understanding, and smart cameras. Most existing methods rely on Transformer-based cross-attention, whose quadratic complexity hinders efficiency. Moreover, small vision-language models often struggle to precisely capture fine-grained, task-relevant visual regions, leading to degraded performance on fine-grained reasoning tasks that limit their effectiveness in the real world. To address these issues, we introduce Firebolt-VL, an efficient vision-language model that replaces the Transformer-based decoder with a Liquid Foundation Model (LFM) decoder. To further enhance visual grounding, we propose a Token-Grid Correlation Module, which computes lightweight correlations between text tokens and image patches and modulates via the state-space model with FiLM conditioning. This enables the model to selectively emphasize visual regions relevant to the textual prompt while maintaining linear-time inference. Experimental results across multiple benchmarks demonstrate that Firebolt-VL achieves accurate, fine-grained understanding with significantly improved efficiency. Our model and code are available at: this https URL
>
---
#### [replaced 058] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18093](https://arxiv.org/pdf/2603.18093)**

> **作者:** Haoxiang Rao; Zhao Wang; Chenyang Si; Yan Lyu; Yuanyi Duan; Fang Zhao; Caifeng Shan
>
> **备注:** Accepted by CVPR2026. Code: this https URL
>
> **摘要:** Industrial anomaly detection (AD) is characterized by an abundance of normal images but a scarcity of anomalous ones. Although numerous few-shot anomaly synthesis methods have been proposed to augment anomalous data for downstream AD tasks, most existing approaches require time-consuming training and struggle to learn distributions that are faithful to real anomalies, thereby restricting the efficacy of AD models trained on such data. To address these limitations, we propose a training-free few-shot anomaly generation method, namely O2MAG, which leverages the self-attention in One reference anomalous image to synthesize More realistic anomalies, supporting effective downstream anomaly detection. Specifically, O2MAG manipulates three parallel diffusion processes via self-attention grafting and incorporates the anomaly mask to mitigate foreground-background query confusion, synthesizing text-guided anomalies that closely adhere to real anomalous distributions. To bridge the semantic gap between the encoded anomaly text prompts and the true anomaly semantics, Anomaly-Guided Optimization is further introduced to align the synthesis process with the target anomalous distribution, steering the generation toward realistic and text-consistent anomalies. Moreover, to mitigate faint anomaly synthesis inside anomaly masks, Dual-Attention Enhancement is adopted during generation to reinforce both self- and cross-attention on masked regions. Extensive experiments validate the effectiveness of O2MAG, demonstrating its superior performance over prior state-of-the-art methods on downstream AD tasks.
>
---
#### [replaced 059] Learning to Look Closer: A New Instance-Wise Loss for Small Cerebral Lesion Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17146](https://arxiv.org/pdf/2511.17146)**

> **作者:** Luc Bouteille; Alexander Jaus; Jens Kleesiek; Rainer Stiefelhagen; Lukas Heine
>
> **备注:** Accepted to IEEE ISBI 2026. 5 pages, 2 figures, 2 tables
>
> **摘要:** Traditional loss functions in medical image segmentation, such as Dice, often under-segment small lesions because their small relative volume contributes negligibly to the overall loss. To address this, instance-wise loss functions and metrics have been proposed to evaluate segmentation quality on a per-lesion basis. We introduce CC-DiceCE, a loss function based on the CC-Metrics framework, and compare it with the existing blob loss. Both are benchmarked against a DiceCE baseline within the nnU-Net framework, which provides a robust and standardized setup. We find that CC-DiceCE loss increases detection (recall) with minimal to no degradation in segmentation performance, though with dataset-dependent trade-offs in precision. Furthermore, our multi-dataset study shows that CC-DiceCE generally outperforms blob loss.
>
---
#### [replaced 060] Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting
- **分类: cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.10075](https://arxiv.org/pdf/2601.10075)**

> **作者:** Lebin Zhou; Jingchuan Xiao; Zhendong Wang; Jinhao Wang; Rongduo Han; Nam Ling; Cihan Ruan
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** In 1888, Vincent van Gogh wrote, "I am seeking exaggeration in the essential." This principle, amplifying structural form while suppressing photographic detail, lies at the core of Post-Impressionist art. However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection. To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression. We propose a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that operationalizes this principle in a mesh-free setting. Our method extracts directional flow fields from 2D paintings and back-propagates them into 3D space, rectifying Gaussian primitives to form flow-aligned brushstrokes that conform to scene topology without relying on explicit mesh priors. This enables expressive structural deformation driven directly by painterly motion rather than photometric constraints. Our contributions are threefold: (1) a projection-based, mesh-free flow guidance mechanism that transfers 2D artistic motion into 3D Gaussian geometry; (2) a luminance-structure decoupling strategy that isolates geometric deformation from color optimization, mitigating artifacts during aggressive structural abstraction; and (3) a VLM-as-a-Judge evaluation framework that assesses artistic authenticity through aesthetic judgment instead of conventional pixel-level metrics, explicitly addressing the subjective nature of artistic stylization.
>
---
#### [replaced 061] ENTER: Event Based Interpretable Reasoning for VideoQA
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.14194](https://arxiv.org/pdf/2501.14194)**

> **作者:** Hammad Ayyubi; Junzhang Liu; Ali Asgarov; Zaber Ibn Abdul Hakim; Najibul Haque Sarker; Zhecan Wang; Chia-Wei Tang; Hani Alomari; Md. Atabuzzaman; Xudong Lin; Naveen Reddy Dyava; Shih-Fu Chang; Chris Thomas
>
> **摘要:** In this paper, we present ENTER, an interpretable Video Question Answering (VideoQA) system based on event graphs. Event graphs convert videos into graphical representations, where video events form the nodes and event-event relationships (temporal/causal/hierarchical) form the edges. This structured representation offers many benefits: 1) Interpretable VideoQA via generated code that parses event-graph; 2) Incorporation of contextual visual information in the reasoning process (code generation) via event graphs; 3) Robust VideoQA via Hierarchical Iterative Update of the event graphs. Existing interpretable VideoQA systems are often top-down, disregarding low-level visual information in the reasoning plan generation, and are brittle. While bottom-up approaches produce responses from visual data, they lack interpretability. Experimental results on NExT-QA, IntentQA, and EgoSchema demonstrate that not only does our method outperform existing top-down approaches while obtaining competitive performance against bottom-up approaches, but more importantly, offers superior interpretability and explainability in the reasoning process.
>
---
#### [replaced 062] Prediction of Grade, Gender, and Academic Performance of Children and Teenagers from Handwriting Using the Sigma-Lognormal Model
- **分类: cs.HC; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11519](https://arxiv.org/pdf/2603.11519)**

> **作者:** Adrian Iste; Kazuki Nishizawa; Chisa Tanaka; Andrew Vargo; Anna Scius-Bertrand; Andreas Fischer; Koichi Kise
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Digital handwriting acquisition enables the capture of detailed temporal and kinematic signals reflecting the motor processes underlying writing behavior. While handwriting analysis has been extensively explored in clinical or adult populations, its potential for studying developmental and educational characteristics in children remains less investigated. In this work, we examine whether handwriting dynamics encode information related to student characteristics using a large-scale online dataset collected from Japanese students from elementary school to junior high school. We systematically compare three families of handwriting-derived features: basic statistical descriptors of kinematic signals, entropy-based measures of variability, and parameters obtained from the sigma-lognormal model. Although the dataset contains dense stroke-level recordings, features are aggregated at the student level to enable a controlled comparison between representations. These features are evaluated across three prediction tasks: grade prediction, gender classification, and academic performance classification, using Linear or Logistic Regression and Random Forest models under consistent experimental settings. The results show that handwriting dynamics contain measurable signals related to developmental stage and individual differences, especially for the grade prediction task. These findings highlight the potential of kinematic handwriting analysis and confirm that through their development, children's handwriting evolves toward a lognormal motor organization.
>
---
#### [replaced 063] Dual-Thresholded Heatmap-Guided Proposal Clustering and Negative Certainty Supervision with Enhanced Base Network for Weakly Supervised Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.08289](https://arxiv.org/pdf/2509.08289)**

> **作者:** Yuelin Guo; Haoyu He; Zhiyuan Chen; Zitong Huang; Renhao Lu; Lu Shi; Zejun Wang; Weizhe Zhang
>
> **备注:** IEEE TIP Major Revision
>
> **摘要:** Weakly supervised object detection (WSOD) has attracted significant attention in recent years, as it does not require box-level annotations. State-of-the-art methods generally adopt a multi-module network, which employs WSDDN as the multiple instance detection network module and uses multiple instance refinement modules to refine performance. However, these approaches suffer from three key limitations. First, existing methods tend to generate pseudo GT boxes that either focus only on discriminative parts, failing to capture the whole object, or cover the entire object but fail to distinguish between adjacent intra-class instances. Second, the foundational WSDDN architecture lacks a crucial background class representation for each proposal and exhibits a large semantic gap between its branches. Third, prior methods discard ignored proposals during optimization, leading to slow convergence. To address these challenges, we propose the Dual-thresholded heAtmap-guided proposal clustering and Negative Certainty supervision with Enhanced base network (DANCE) method for WSOD. Specifically, we first devise a heatmap-guided proposal selector (HGPS) algorithm, which utilizes dual thresholds on heatmaps to pre-select proposals, enabling pseudo GT boxes to both capture the full object extent and distinguish between adjacent intra-class instances. We then construct a weakly supervised basic detection network (WSBDN), which augments each proposal with a background class representation and uses heatmaps for pre-supervision to bridge the semantic gap between matrices. At last, we introduce a negative certainty supervision (NCS) loss on ignored proposals to accelerate convergence. Extensive experiments on the challenging PASCAL VOC and MS COCO datasets demonstrate the effectiveness and superiority of our method. Our code is publicly available at this https URL.
>
---
#### [replaced 064] TrajectoryMover: Generative Movement of Object Trajectories in Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29092](https://arxiv.org/pdf/2603.29092)**

> **作者:** Kiran Chhatre; Hyeonho Jeong; Yulia Gryaditskaya; Christopher E. Peters; Chun-Hao Paul Huang; Paul Guerrero
>
> **备注:** 24 pages, 8 figures. Project page: this https URL
>
> **摘要:** Generative video editing has enabled several intuitive editing operations for short video clips that would previously have been difficult to achieve, especially for non-expert editors. Existing methods focus on prescribing an object's 3D or 2D motion trajectory in a video, or on altering the appearance of an object or a scene, while preserving both the video's plausibility and identity. Yet a method to move an object's 3D motion trajectory in a video, i.e., moving an object while preserving its relative 3D motion, is currently still missing. The main challenge lies in obtaining paired video data for this scenario. Previous methods typically rely on clever data generation approaches to construct plausible paired data from unpaired videos, but this approach fails if one of the videos in a pair can not easily be constructed from the other. Instead, we introduce TrajectoryAtlas, a new data generation pipeline for large-scale synthetic paired video data and a video generator TrajectoryMover fine-tuned with this data. We show that this successfully enables generative movement of object trajectories. Project page: this https URL
>
---
#### [replaced 065] Controllable Image Generation with Composed Parallel Token Prediction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2405.06535](https://arxiv.org/pdf/2405.06535)**

> **作者:** Jamie Stirling; Noura Al-Moubayed; Chris G. Willcocks; Hubert P. H. Shum
>
> **备注:** 8 pages + references, 7 figures, accepted to CVPR Workshops 2026 (LoViF)
>
> **摘要:** Conditional discrete generative models struggle to faithfully compose multiple input conditions. To address this, we derive a theoretically-grounded formulation for composing discrete probabilistic generative processes, with masked generation (absorbing diffusion) as a special case. Our formulation enables precise specification of novel combinations and numbers of input conditions that lie outside the training data, with concept weighting enabling emphasis or negation of individual conditions. In synergy with the richly compositional learned vocabulary of VQ-VAE and VQ-GAN, our method attains a $63.4\%$ relative reduction in error rate compared to the previous state-of-the-art, averaged across 3 datasets (positional CLEVR, relational CLEVR and FFHQ), simultaneously obtaining an average absolute FID improvement of $-9.58$. Meanwhile, our method offers a $2.3\times$ to $12\times$ real-time speed-up over comparable methods, and is readily applied to an open pre-trained discrete text-to-image model for fine-grained control of text-to-image generation.
>
---
#### [replaced 066] Why CNN Features Are not Gaussian: A Statistical Anatomy of Deep Representations
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.05183](https://arxiv.org/pdf/2411.05183)**

> **作者:** David Chapman; Parniyan Farvardin
>
> **摘要:** Deep convolutional neural networks (CNNs) are commonly analyzed through geometric and linear-algebraic perspectives, yet the statistical distribution of their internal feature activations remains poorly understood. In many applications, deep features are implicitly treated as Gaussian when modeling densities. In this work, we empirically examine this assumption and show that it does not accurately describe the distribution of CNN feature activations. Through a systematic study across multiple architectures and datasets, we find that the feature activations deviate substantially from Gaussian and are better characterized by Weibull and related long-tailed distributions. We further introduce a novel Discretized Characteristic Function Copula (DCF-Copula) method to model multivariate feature dependencies. We find that tail-length increases with network depth and that upper-tail dependence emerges between feature pairs. These statistical findings are not consistent with the Central Limit Theorem, and are instead indicative of a Matthew process that progressively concentrates semantic signal within the tails. These statistical findings suggest that CNNs are excellent at noise reduction, yet poor at outlier removal tasks. We recommend the use of long-tailed upper-tail-dependent priors as opposed to Gaussian priors for accurately CNN deep feature density. Code available at this https URL
>
---
#### [replaced 067] Aleatoric Uncertainty Medical Image Segmentation Estimation via Flow Matching
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.22418](https://arxiv.org/pdf/2507.22418)**

> **作者:** Phi Van Nguyen; Ngoc Huynh Trinh; Duy Minh Lam Nguyen; Phu Loc Nguyen; Quoc Long Tran
>
> **摘要:** Quantifying aleatoric uncertainty in medical image segmentation is critical since it is a reflection of the natural variability observed among expert annotators. A conventional approach is to model the segmentation distribution using the generative model, but current methods limit the expression ability of generative models. While current diffusion-based approaches have demonstrated impressive performance in approximating the data distribution, their inherent stochastic sampling process and inability to model exact densities limit their effectiveness in accurately capturing uncertainty. In contrast, our proposed method leverages conditional flow matching, a simulation-free flow-based generative model that learns an exact density, to produce highly accurate segmentation results. By guiding the flow model on the input image and sampling multiple data points, our approach synthesizes segmentation samples whose pixel-wise variance reliably reflects the underlying data distribution. This sampling strategy captures uncertainties in regions with ambiguous boundaries, offering robust quantification that mirrors inter-annotator differences. Experimental results demonstrate that our method not only achieves competitive segmentation accuracy but also generates uncertainty maps that provide deeper insights into the reliability of the segmentation outcomes. The code for this paper is freely available at this https URL
>
---
#### [replaced 068] MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.10649](https://arxiv.org/pdf/2601.10649)**

> **作者:** Darshan Singh; Arsha Nagrani; Kawshik Manikantan; Harman Singh; Dinesh Tewari; Tobias Weyand; Cordelia Schmid; Anelia Angelova; Shachi Dave
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent advancements in video models have shown tremendous progress, particularly in long video understanding. However, current benchmarks predominantly feature western-centric data and English as the dominant language, introducing significant biases in evaluation. To address this, we introduce MINERVA-Cultural, a challenging benchmark for multicultural and multilingual video reasoning. MINERVA-Cultural comprises high-quality, entirely human-generated annotations from diverse, region-specific cultural videos across 18 global locales. Unlike prior work that relies on automatic translations, MINERVA-Cultural provides complex questions, answers, and multi-step reasoning steps, all crafted in native languages. Making progress on MINERVA-Cultural requires a deeply situated understanding of visual cultural context. Furthermore, we leverage MINERVA-Cultural's reasoning traces to construct evidence-based graphs and propose a novel iterative strategy using these graphs to identify fine-grained errors in reasoning. Our evaluations reveal that SoTA Video-LLMs struggle significantly, performing substantially below human-level accuracy, with errors primarily stemming from the visual perception of cultural elements. MINERVA-Cultural will be publicly available under this https URL\#minerva-cultural
>
---
#### [replaced 069] Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决预训练VLA模型在微调中性能提升有限、成本高的问题。通过解耦辅助任务目标，提升模型能力并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2603.25661](https://arxiv.org/pdf/2603.25661)**

> **作者:** Wenxuan Song; Jiayi Chen; Shuai Chen; Jingbo Wang; Pengxiang Ding; Han Zhao; Yikai Qin; Xinhu Zheng; Donglin Wang; Yan Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary tasks. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary task training within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver this goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies. The difference between the resulting model parameters can then be interpreted as capability vectors provided by auxiliary tasks. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Experimental results demonstrate that this approach is highly effective across diverse robot tasks. Project page: this https URL
>
---
#### [replaced 070] Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23202](https://arxiv.org/pdf/2603.23202)**

> **作者:** Anupam Pani; Yanchao Yang
>
> **摘要:** Despite advances in Vision-Language-Action (VLA) models, robotic manipulation struggles with fine-grained tasks because current models lack mechanisms for active visual attention allocation. Human gaze naturally encodes intent, planning, and execution patterns -- offering a powerful supervisory signal for guiding robot perception. We introduce a gaze-regularized training framework that aligns VLA models' internal attention with human visual patterns without architectural modifications or inference-time overhead. Our method transforms temporally aggregated gaze heatmaps into patch-level distributions and regularizes the transformer's attention through KL divergence, creating an inductive bias toward task-relevant features while preserving deployment efficiency. When integrated into existing VLA architectures, our approach yields 4-12% improvements across manipulation benchmarks. The gaze-regularized models reach equivalent performance with fewer training steps and maintain robustness under lighting variations and sensor noise. Beyond performance metrics, the learned attention patterns produce interpretable visualizations that mirror human strategies, enhancing trust in robotic systems. Moreover, our framework requires no eye-tracking equipment and applies directly to existing datasets. These results demonstrate that human perceptual priors can significantly accelerate robot learning while improving both task performance and system interpretability.
>
---
#### [replaced 071] InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11911](https://arxiv.org/pdf/2603.11911)**

> **作者:** InSpatio Team; Donghui Shen; Guofeng Zhang; Haomin Liu; Haoyu Ji; Jialin Liu; Jing Guo; Nan Wang; Siji Pan; Weihong Pan; Weijian Xie; Xiaojun Xiang; Xiaoyu Zhang; Xianbin Liu; Yifu Wang; Yipeng Chen; Zhewen Le; Zhichao Ye; Ziqiang Zhao
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** We present InSpatio-WorldFM, an open-source real-time frame model for spatial intelligence. Unlike video-based world models that rely on sequential frame generation and incur substantial latency due to window-level processing, InSpatio-WorldFM adopts a frame-based paradigm that generates each frame independently, enabling low-latency real-time spatial inference. By enforcing multi-view spatial consistency through explicit 3D anchors and implicit spatial memory, the model preserves global scene geometry while maintaining fine-grained visual details across viewpoint changes. We further introduce a progressive three-stage training pipeline that transforms a pretrained image diffusion model into a controllable frame model and finally into a real-time generator through few-step distillation. Experimental results show that InSpatio-WorldFM achieves strong multi-view consistency while supporting interactive exploration on consumer-grade GPUs, providing an efficient alternative to traditional video-based world models for real-time world simulation.
>
---
#### [replaced 072] Beyond Corner Patches: Semantics-Aware Backdoor Attack in Federated Learning
- **分类: cs.CR; cs.AI; cs.CV; cs.DC; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.29328](https://arxiv.org/pdf/2603.29328)**

> **作者:** Kavindu Herath; Joshua Zhao; Saurabh Bagchi
>
> **备注:** Accepted as a regular paper at IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2026
>
> **摘要:** Backdoor attacks on federated learning (FL) are most often evaluated with synthetic corner patches or out-of-distribution (OOD) patterns that are unlikely to arise in practice. In this paper, we revisit the backdoor threat to standard FL (a single global model) under a more realistic setting where triggers must be semantically meaningful, in-distribution, and visually plausible. We propose SABLE, a Semantics-Aware Backdoor for LEarning in federated settings, which constructs natural, content-consistent triggers (e.g., semantic attribute changes such as sunglasses) and optimizes an aggregation-aware malicious objective with feature separation and parameter regularization to keep attacker updates close to benign ones. We instantiate SABLE on CelebA hair-color classification and the German Traffic Sign Recognition Benchmark (GTSRB), poisoning only a small, interpretable subset of each malicious client's local data while otherwise following the standard FL protocol. Across heterogeneous client partitions and multiple aggregation rules (FedAvg, Trimmed Mean, MultiKrum, and FLAME), our semantics-driven triggers achieve high targeted attack success rates while preserving benign test accuracy. These results show that semantics-aligned backdoors remain a potent and practical threat in federated learning, and that robustness claims based solely on synthetic patch triggers can be overly optimistic.
>
---
#### [replaced 073] Cross-Domain Few-Shot Learning for Hyperspectral Image Classification Based on Mixup Foundation Model
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.22581](https://arxiv.org/pdf/2601.22581)**

> **作者:** Naeem Paeedeh; Mahardhika Pratama; Ary Shiddiqi; Zehong Cao; Mukesh Prasad; Wisnu Jatmiko
>
> **摘要:** Although cross-domain few-shot learning (CDFSL) for hyper-spectral image (HSI) classification has attracted significant research interest, existing works often rely on an unrealistic data augmentation procedure in the form of external noise to enlarge the sample size, thus greatly simplifying the issue of data scarcity. They involve a large number of parameters for model updates, being prone to the overfitting problem. To the best of our knowledge, none has explored the strength of the foundation model, having strong generalization power to be quickly adapted to downstream tasks. This paper proposes the MIxup FOundation MOdel (MIFOMO) for CDFSL of HSI classifications. MIFOMO is built upon the concept of a remote sensing (RS) foundation model, pre-trained across a large scale of RS problems, thus featuring generalizable features. The notion of coalescent projection (CP) is introduced to quickly adapt the foundation model to downstream tasks while freezing the backbone network. The concept of mixup domain adaptation (MDM) is proposed to address the extreme domain discrepancy problem. Last but not least, the label smoothing concept is implemented to cope with noisy pseudo-label problems. Our rigorous experiments demonstrate the advantage of MIFOMO, where it beats prior arts with up to 14% margin. The source code of MIFOMO is open-sourced at this https URL for reproducibility and convenient further study.
>
---
#### [replaced 074] Graphic-Design-Bench: A Comprehensive Benchmark for Evaluating AI on Graphic Design Tasks
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.04192](https://arxiv.org/pdf/2604.04192)**

> **作者:** Adrienne Deganutti; Elad Hirsch; Haonan Zhu; Jaejung Seol; Purvanshi Mehta
>
> **摘要:** We introduce GraphicDesignBench (GDB), the first comprehensive benchmark suite designed specifically to evaluate AI models on the full breadth of professional graphic design tasks. Unlike existing benchmarks that focus on natural-image understanding or generic text-to-image synthesis, GDB targets the unique challenges of professional design work: translating communicative intent into structured layouts, rendering typographically faithful text, manipulating layered compositions, producing valid vector graphics, and reasoning about animation. The suite comprises 50 tasks organized along five axes: layout, typography, infographics, template & design semantics and animation, each evaluated under both understanding and generation settings, and grounded in real-world design templates drawn from the LICA layered-composition dataset. We evaluate a set of frontier closed-source models using a standardized metric taxonomy covering spatial accuracy, perceptual quality, text fidelity, semantic alignment, and structural validity. Our results reveal that current models fall short on the core challenges of professional design: spatial reasoning over complex layouts, faithful vector code generation, fine-grained typographic perception, and temporal decomposition of animations remain largely unsolved. While high-level semantic understanding is within reach, the gap widens sharply as tasks demand precision, structure, and compositional awareness. GDB provides a rigorous, reproducible testbed for tracking progress toward AI systems that can function as capable design collaborators. The full evaluation framework is publicly available.
>
---
#### [replaced 075] PaCo-FR: Patch-Pixel Aligned End-to-End Codebook Learning for Facial Representation Pre-training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09691](https://arxiv.org/pdf/2508.09691)**

> **作者:** Yin Xie; Zhichao Chen; Zeyu Xiao; Yongle Zhao; Xiang An; Kaicheng Yang; Zimin Ran; Jia Guo; Ziyong Feng; Jiankang Deng
>
> **摘要:** Facial representation pre-training is crucial for tasks like facial recognition, expression analysis, and virtual reality. However, existing methods face three key challenges: (1) failing to capture distinct facial features and fine-grained semantics, (2) ignoring the spatial structure inherent to facial anatomy, and (3) inefficiently utilizing limited labeled data. To overcome these, we introduce PaCo-FR, an unsupervised framework that combines masked image modeling with patch-pixel alignment. Our approach integrates three innovative components: (1) a structured masking strategy that preserves spatial coherence by aligning with semantically meaningful facial regions, (2) a novel patch-based codebook that enhances feature discrimination with multiple candidate tokens, and (3) spatial consistency constraints that preserve geometric relationships between facial components. PaCo-FR achieves state-of-the-art performance across several facial analysis tasks with just 2 million unlabeled images for pre-training. Our method demonstrates significant improvements, particularly in scenarios with varying poses, occlusions, and lighting conditions. We believe this work advances facial representation learning and offers a scalable, efficient solution that reduces reliance on expensive annotated datasets, driving more effective facial analysis systems.
>
---
#### [replaced 076] A global dataset of continuous urban dashcam driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.01044](https://arxiv.org/pdf/2604.01044)**

> **作者:** Md Shadab Alam; Olena Bazilinska; Pavlo Bazilinskyy
>
> **摘要:** We introduce CROWD (City Road Observations With Dashcams), a manually curated dataset of ordinary, minute scale, temporally contiguous, unedited, front facing urban dashcam segments screened and segmented from publicly available YouTube videos. CROWD is designed to support cross-domain robustness and interaction analysis by prioritising routine driving and explicitly excluding crashes, crash aftermath, and other edited or incident-focused content. The release contains 51,753 segment records spanning 20,275.56 hours (42,032 videos), covering 7,103 named inhabited places in 238 countries and territories across all six inhabited continents (Africa, Asia, Europe, North America, South America and Oceania), with segment level manual labels for time of day (day or night) and vehicle type. To lower the barrier for benchmarking, we provide per-segment CSV files of machine-generated detections for all 80 MS-COCO classes produced with YOLOv11x, together with segment-local multi-object tracks (BoT-SORT); e.g. person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign, etc. CROWD is distributed as video identifiers with segment boundaries and derived annotations, enabling reproducible research without redistributing the underlying videos.
>
---
#### [replaced 077] Ultrasound-based detection and malignancy prediction of breast lesions eligible for biopsy: A multi-center clinical-scenario study using nomograms, large language models, and radiologist evaluation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.00946](https://arxiv.org/pdf/2509.00946)**

> **作者:** Ali Abbasian Ardakani; Afshin Mohammadi; Taha Yusuf Kuzan; Beyza Nur Kuzan; Hamid Khorshidi; Ashkan Ghorbani; Alisa Mohebbi; Fariborz Faeghi; Sepideh Hatamikia; U Rajendra Acharya
>
> **备注:** Academic Radiology (2026)
>
> **摘要:** To develop and externally validate integrated ultrasound nomograms combining BIRADS features and quantitative morphometric characteristics, and to compare their performance with expert radiologists and state of the art large language models in biopsy recommendation and malignancy prediction for breast lesions. In this retrospective multicenter, multinational study, 1747 women with pathologically confirmed breast lesions underwent ultrasound across three centers in Iran and Turkey. A total of 10 BIRADS and 26 morphological features were extracted from each lesion. A BIRADS, morphometric, and fused nomogram integrating both feature sets was constructed via logistic regression. Three radiologists (one senior, two general) and two ChatGPT variants independently interpreted deidentified breast lesion images. Diagnostic performance for biopsy recommendation (BIRADS 4,5) and malignancy prediction was assessed in internal and two external validation cohorts. In pooled analysis, the fused nomogram achieved the highest accuracy for biopsy recommendation (83.0%) and malignancy prediction (83.8%), outperforming the morphometric nomogram, three radiologists and both ChatGPT models. Its AUCs were 0.901 and 0.853 for the two tasks, respectively. In addition, the performance of the BIRADS nomogram was significantly higher than the morphometric nomogram, three radiologists and both ChatGPT models for biopsy recommendation and malignancy prediction. External validation confirmed the robust generalizability across different ultrasound platforms and populations. An integrated BIRADS morphometric nomogram consistently outperforms standalone models, LLMs, and radiologists in guiding biopsy decisions and predicting malignancy. These interpretable, externally validated tools have the potential to reduce unnecessary biopsies and enhance personalized decision making in breast imaging.
>
---
#### [replaced 078] ProBA: Probabilistic Bundle Adjustment with the Bhattacharyya Coefficient
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20858](https://arxiv.org/pdf/2505.20858)**

> **作者:** Jason Chui; Hector Andrade-Loarca; Daniel Cremers
>
> **备注:** 14 pages, 5 figures, 3 tables
>
> **摘要:** Classical Bundle Adjustment (BA) is fundamentally limited by its reliance on precise metric initialization and prior camera intrinsics. While modern dense matchers offer high-fidelity correspondences, traditional Structure-from-Motion (SfM) pipelines struggle to leverage them, as rigid track-building heuristics fail in the presence of their inherent noise. We present \textbf{ProBA (Probabilistic Bundle Adjustment)}, a probabilistic re-parameterization of the BA manifold that enables joint optimization of extrinsics, focal lengths, and geometry from a strict cold start. By replacing fragile point tracks with a flexible kinematic pose graph and representing landmarks as 3D Gaussians, our framework explicitly models spatial uncertainty through a unified Negative Log-Likelihood (NLL) objective. This volumetric formulation smooths the non-convex optimization landscape and naturally weights correspondences by their statistical confidence. To maintain global consistency, we optimize over a sparse view graph using an iterative, adaptive edge-weighting mechanism to prune erroneous topological links. Furthermore, we resolve mirror ambiguities inherent to prior-free SfM via a dual-hypothesis regularization strategy. Extensive evaluations show that our approach significantly expands the basin of attraction and achieves superior accuracy over both classical and learning-based baselines, providing a scalable foundation that greatly benefits SfM and SLAM robustness in unstructured environments.
>
---
#### [replaced 079] Unreal Robotics Lab: A High-Fidelity Robotics Simulator with Advanced Physics and Rendering
- **分类: cs.RO; cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出Unreal Robotics Lab，融合Unreal引擎与MuJoCo，解决机器人仿真中物理精度与渲染真实性的平衡问题，支持复杂环境测试与视觉导航研究。**

- **链接: [https://arxiv.org/pdf/2504.14135](https://arxiv.org/pdf/2504.14135)**

> **作者:** Jonathan Embley-Riches; Jianwei Liu; Simon Julier; Dimitrios Kanoulas
>
> **摘要:** High-fidelity simulation is essential for robotics research, enabling safe and efficient testing of perception, control, and navigation algorithms. However, achieving both photorealistic rendering and accurate physics modeling remains a challenge. This paper presents a novel simulation framework, the Unreal Robotics Lab (URL), that integrates the advanced rendering capabilities of the Unreal Engine with MuJoCo's high-precision physics simulation. Our approach enables realistic robotic perception while maintaining accurate physical interactions, facilitating benchmarking and dataset generation for vision-based robotics applications. The system supports complex environmental effects, such as smoke, fire, and water dynamics, which are critical to evaluating robotic performance under adverse conditions. We benchmark visual navigation and SLAM methods within our framework, demonstrating its utility for testing real-world robustness in controlled yet diverse scenarios. By bridging the gap between physics accuracy and photorealistic rendering, our framework provides a powerful tool for advancing robotics research and sim-to-real transfer. Our open-source framework is available at this https URL.
>
---
#### [replaced 080] IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.03054](https://arxiv.org/pdf/2601.03054)**

> **作者:** Yankai Jiang; Qiaoru Li; Binlu Xu; Haoran Sun; Chao Ding; Junting Dong; Yuxiang Cai; Xuhong Zhang; Jianwei Yin
>
> **摘要:** Recent research on medical MLLMs has gradually shifted its focus from image-level understanding to fine-grained, pixel-level comprehension. Although segmentation serves as the foundation for pixel-level understanding, existing approaches face two major challenges. First, they introduce implicit segmentation tokens and require simultaneous fine-tuning of both the MLLM and external pixel decoders, which increases the risk of catastrophic forgetting and limits generalization to out-of-domain scenarios. Second, most methods rely on single-pass reasoning and lack the capability to iteratively refine segmentation results, leading to suboptimal performance. To overcome these limitations, we propose a novel agentic MLLM, named IBISAgent, that reformulates segmentation as a vision-centric, multi-step decision-making process. IBISAgent enables MLLMs to generate interleaved reasoning and text-based click actions, invoke segmentation tools, and produce high-quality masks without architectural modifications. By iteratively performing multi-step visual reasoning on masked image features, IBISAgent naturally supports mask refinement and promotes the development of pixel-level visual reasoning capabilities. We further design a two-stage training framework consisting of cold-start supervised fine-tuning and agentic reinforcement learning with tailored, fine-grained rewards, enhancing the model's robustness in complex medical referring and reasoning segmentation tasks. Extensive experiments demonstrate that IBISAgent consistently outperforms both closed-source and open-source SOTA methods.
>
---
#### [replaced 081] In search of truth: Evaluating concordance of AI-based anatomy segmentation models
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15921](https://arxiv.org/pdf/2512.15921)**

> **作者:** Lena Giebeler; Deepa Krishnaswamy; David Clunie; Jakob Wasserthal; Lalith Kumar Shiyam Sundar; Andres Diaz-Pinto; Klaus H. Maier-Hein; Murong Xu; Bjoern Menze; Steve Pieper; Ron Kikinis; Andrey Fedorov
>
> **摘要:** Purpose AI-based methods for anatomy segmentation can help automate characterization of large imaging datasets. The growing number of similar in functionality models raises the challenge of evaluating them on datasets that do not contain ground truth annotations. We introduce a practical framework to assist in this task. Approach We harmonize the segmentation results into a standard, interoperable representation, which enables consistent, terminology-based labeling of the structures. We extend 3D Slicer to streamline loading and comparison of these harmonized segmentations, and demonstrate how standard representation simplifies review of the results using interactive summary plots and browser-based visualization using OHIF Viewer. To demonstrate the utility of the approach we apply it to evaluating segmentation of 31 anatomical structures (lungs, vertebrae, ribs, and heart) by six open-source models - TotalSegmentator 1.5 and 2.6, Auto3DSeg, MOOSE, MultiTalent, and CADS - for a sample of Computed Tomography (CT) scans from the publicly available National Lung Screening Trial (NLST) dataset. Results We demonstrate the utility of the framework in enabling automating loading, structure-wise inspection and comparison across models. Preliminary results ascertain practical utility of the approach in allowing quick detection and review of problematic results. The comparison shows excellent agreement segmenting some (e.g., lung) but not all structures (e.g., some models produce invalid vertebrae or rib segmentations). Conclusions The resources developed are linked from this https URL including segmentation harmonization scripts, summary plots, and visualization tools. This work assists in model evaluation in absence of ground truth, ultimately enabling informed model selection.
>
---
#### [replaced 082] ReaMIL: Reasoning- and Evidence-Aware Multiple Instance Learning for Whole-Slide Histopathology
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.10073](https://arxiv.org/pdf/2601.10073)**

> **作者:** Hyun Do Jung; Jungwon Choi; Hwiyoung Kim
>
> **备注:** Accepted at LFMBio Workshop, WACV 2026. Oral Presentation
>
> **摘要:** We introduce ReaMIL (Reasoning- and Evidence-Aware MIL), a multiple instance learning approach for whole-slide histopathology that adds a light selection head to a strong MIL backbone. The head produces soft per-tile gates and is trained with a budgeted-sufficiency objective: a hinge loss that enforces the true-class probability to be $\geq \tau$ using only the kept evidence, under a sparsity budget on the number of selected tiles. The budgeted-sufficiency objective yields small, spatially compact evidence sets without sacrificing baseline performance. Across TCGA-NSCLC (LUAD vs. LUSC), TCGA-BRCA (IDC vs. Others), and PANDA, ReaMIL matches or slightly improves baseline AUC and provides quantitative evidence-efficiency diagnostics. On NSCLC, it attains AUC 0.983 with a mean minimal sufficient K (MSK) $\approx 8.2$ tiles at $\tau = 0.90$ and AUKC $\approx 0.864$, showing that class confidence rises sharply and stabilizes once a small set of tiles is kept. The method requires no extra supervision, integrates seamlessly with standard MIL training, and naturally yields slide-level overlays. We report accuracy alongside MSK, AUKC, and contiguity for rigorous evaluation of model behavior on WSIs.
>
---
#### [replaced 083] MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19172](https://arxiv.org/pdf/2511.19172)**

> **作者:** Kehua Chen; Tianlu Mao; Xinzhu Ma; Hao Jiang; Zehao Li; Zihan Liu; Shuqin Gao; Honglong Zhao; Feng Dai; Yucheng Zhang; Zhaoqi Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
>
---
#### [replaced 084] MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs
- **分类: cs.CL; cs.AI; cs.CV; eess.IV**

- **简介: 该论文提出MedXIAOHE，一种用于医疗视觉-语言理解的多模态大模型，解决临床应用中的医学推理与诊断问题。通过持续预训练和强化学习提升模型性能与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.12705](https://arxiv.org/pdf/2602.12705)**

> **作者:** Baorong Shi; Bo Cui; Boyuan Jiang; Deli Yu; Fang Qian; Haihua Yang; Huichao Wang; Jiale Chen; Jianfei Pan; Jieqiong Cao; Jinghao Lin; Kai Wu; Lin Yang; Shengsheng Yao; Tao Chen; Xiaojun Xiao; Xiaozhong Ji; Xu Wang; Yijun He; Zhixiong Yang
>
> **备注:** XIAOHE Medical AI team. See paper for full author list. Currently, the model is exclusively available on XIAOHE AI Doctor, accessible via both the App Store and the Douyin Mini Program. Updated to improve the layout
>
> **摘要:** We present MedXIAOHE, a medical vision-language foundation model designed to advance general-purpose medical understanding and reasoning in real-world clinical applications. MedXIAOHE achieves state-of-the-art performance across diverse medical benchmarks and surpasses leading closed-source multimodal systems on multiple capabilities. To achieve this, we propose an entity-aware continual pretraining framework that organizes heterogeneous medical corpora to broaden knowledge coverage and reduce long-tail gaps (e.g., rare diseases). For medical expert-level reasoning and interaction, MedXIAOHE incorporates diverse medical reasoning patterns via reinforcement learning and tool-augmented agentic training, enabling multi-step diagnostic reasoning with verifiable decision traces. To improve reliability in real-world use, MedXIAOHE integrates user-preference rubrics, evidence-grounded reasoning, and low-hallucination long-form report generation, with improved adherence to medical instructions. We release this report to document our practical design choices, scaling insights, and evaluation framework, hoping to inspire further research.
>
---
#### [replaced 085] Less Detail, Better Answers: Degradation-Driven Prompting for VQA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04838](https://arxiv.org/pdf/2604.04838)**

> **作者:** Haoxuan Han; Weijie Wang; Zeyu Zhang; Yefei He; Bohan Zhuang
>
> **备注:** Accepted to CVPRW 2026. Project page: this https URL , Code: this https URL
>
> **摘要:** Recent advancements in Vision-Language Models (VLMs) have significantly pushed the boundaries of Visual Question Answering (VQA).However,high-resolution details can sometimes become noise that leads to hallucinations or reasoning errors. In this paper,we propose Degradation-Driven Prompting (DDP), a novel framework that improves VQA performance by strategically reducing image fidelity to force models to focus on essential structural information. We evaluate DDP across two distinct tasks. Physical attributes targets images prone to human misjudgment, where DDP employs a combination of 80p downsampling, structural visual aids (white background masks and orthometric lines), and In-Context Learning (ICL) to calibrate the model's focus. Perceptual phenomena addresses various machine-susceptible visual anomalies and illusions, including Visual Anomaly (VA), Color (CI), Motion(MI),Gestalt (GI), Geometric (GSI), and Visual Illusions (VI).For this task, DDP integrates a task-classification stage with specialized tools such as blur masks and contrast enhancement alongside downsampling. Our experimental results demonstrate that less is more: by intentionally degrading visual inputs and providing targeted structural prompts, DDP enables VLMs to bypass distracting textures and achieve superior reasoning accuracy on challenging visual benchmarks.
>
---
#### [replaced 086] FinCriticalED: A Visual Benchmark for Financial Fact-Level OCR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14998](https://arxiv.org/pdf/2511.14998)**

> **作者:** Yueru He; Xueqing Peng; Yupeng Cao; Yan Wang; Lingfei Qian; Haohang Li; Yi Han; Shuyao Wang; Ruoyu Xiang; Fan Zhang; Zhuohan Xie; Mingquan Lin; Prayag Tiwari; Jimin Huang; Guojun Xiong; Sophia Ananiadou
>
> **备注:** Xueqing Peng: Corresponding-Author
>
> **摘要:** Recent progress in multimodal large language models (MLLMs) has substantially improved document understanding, yet strong optical character recognition (OCR) performance on surface metrics does not guarantee faithful preservation of decision-critical evidence. This limitation is especially consequential in financial documents, where small visual errors can induce discrete shifts in meaning. To study this gap, we introduce FinCriticalED (Financial Critical Error Detection), a fact-centric visual benchmark for evaluating whether OCR and vision-language systems preserve financially critical evidence beyond lexical similarity. FinCriticalED contains 859 real-world financial document pages with 9,481 expert-annotated facts spanning five critical field types: numeric, temporal, monetary unit, reporting entity, and financial concept. We formulate the task as structured OCR with fact-level verification, and develop a Deterministic-Rule-Guided LLM-as-Judge protocol to assess whether model outputs preserve annotated facts in context. We benchmark 13 systems spanning OCR pipelines, specialized OCR VLMs, open-source MLLMs, and proprietary MLLMs. Results reveal a clear gap between lexical accuracy and factual reliability, with numerical values and monetary units emerging as the most vulnerable fact types, and critical errors concentrating in visually complex, mixed-layout documents with distinct failure patterns across model families. Overall, FinCriticalED provides a rigorous benchmark for trustworthy financial OCR and a practical testbed for evidence fidelity in high-stakes multimodal document understanding. Benchmark and dataset details available at this https URL
>
---
#### [replaced 087] DiffAttn: Diffusion-Based Drivers' Visual Attention Prediction with LLM-Enhanced Semantic Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.28251](https://arxiv.org/pdf/2603.28251)**

> **作者:** Weimin Liu; Qingkun Li; Jiyuan Qiu; Wenjun Wang; Joshua H. Meng
>
> **摘要:** Drivers' visual attention provides critical cues for anticipating latent hazards and directly shapes decision-making and control maneuvers, where its absence can compromise traffic safety. To emulate drivers' perception patterns and advance visual attention prediction for intelligent vehicles, we propose DiffAttn, a diffusion-based framework that formulates this task as a conditional diffusion-denoising process, enabling more accurate modeling of drivers' attention. To capture both local and global scene features, we adopt Swin Transformer as encoder and design a decoder that combines a Feature Fusion Pyramid for cross-layer interaction with dense, multi-scale conditional diffusion to jointly enhance denoising learning and model fine-grained local and global scene contexts. Additionally, a large language model (LLM) layer is incorporated to enhance top-down semantic reasoning and improve sensitivity to safety-critical cues. Extensive experiments on four public datasets demonstrate that DiffAttn achieves state-of-the-art (SoTA) performance, surpassing most video-based, top-down-feature-driven, and LLM-enhanced baselines. Our framework further supports interpretable driver-centric scene understanding and has the potential to improve in-cabin human-machine interaction, risk perception, and drivers' state measurement in intelligent vehicles.
>
---
#### [replaced 088] Vero: An Open RL Recipe for General Visual Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Vero，一个开放的视觉语言模型，解决跨任务视觉推理问题。通过扩展强化学习数据和奖励机制，提升模型在多种视觉推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.04917](https://arxiv.org/pdf/2604.04917)**

> **作者:** Gabriel Sarch; Linrong Cai; Qunzhong Wang; Haoyang Wu; Danqi Chen; Zhuang Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.6-5.3 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released.
>
---
#### [replaced 089] SatFusion: A Unified Framework for Enhancing Remote Sensing Images via Multi-Frame and Multi-Source Images Fusion
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.07905](https://arxiv.org/pdf/2510.07905)**

> **作者:** Yufei Tong; Guanjie Cheng; Peihan Wu; Feiyi Chen; Xinkui Zhao; Shuiguang Deng
>
> **摘要:** High-quality remote sensing (RS) image acquisition is fundamentally constrained by physical limitations. While Multi-Frame Super-Resolution (MFSR) and Pansharpening address this by exploiting complementary information, they are typically studied in isolation: MFSR lacks high-resolution (HR) structural priors for fine-grained texture recovery, whereas Pansharpening relies on upsampled low-resolution (LR) inputs and is sensitive to noise and misalignment. In this paper, we propose SatFusion, a novel and unified framework that seamlessly bridges multi-frame and multi-source RS image fusion. SatFusion extracts HR semantic features by aggregating complementary information from multiple LR multispectral frames via a Multi-Frame Image Fusion (MFIF) module, and integrates fine-grained structural details from an HR panchromatic image through a Multi-Source Image Fusion (MSIF) module with implicit pixel-level alignment. To further alleviate the lack of structural priors during multi-frame fusion, we introduce an advanced variant, SatFusion*, which integrates a panchromatic-guided mechanism into the MFIF stage. Through structure-aware feature embedding and transformer-based adaptive aggregation, SatFusion* enables spatially adaptive feature selection, strengthening the coupling between multi-frame and multi-source representations. Extensive experiments on four benchmark datasets validate our core insight: synergistically coupling multi-frame and multi-source priors effectively resolves the fragility of existing paradigms, delivering superior reconstruction fidelity, robustness, and generalizability.
>
---
#### [replaced 090] RenderFlow: Single-Step Neural Rendering via Flow Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.06928](https://arxiv.org/pdf/2601.06928)**

> **作者:** Shenghao Zhang; Runtao Liu; Christopher Schroers; Yang Zhang
>
> **备注:** CVPR 2026; Supplementary material included
>
> **摘要:** Conventional physically based rendering (PBR) pipelines generate photorealistic images through computationally intensive light transport simulations. Although recent deep learning approaches leverage diffusion model priors with geometry buffers (G-buffers) to produce visually compelling results without explicit scene geometry or light simulation, they remain constrained by two major limitations. First, the iterative nature of the diffusion process introduces substantial latency. Second, the inherent stochasticity of these generative models compromises physical accuracy and temporal consistency. In response to these challenges, we propose a novel, end-to-end, deterministic, single-step neural rendering framework, RenderFlow, built upon a flow matching paradigm. To further strengthen both rendering quality and generalization, we propose an efficient and effective module for sparse keyframe guidance. Our method significantly accelerates the rendering process and, by optionally incorporating sparsely rendered keyframes as guidance, enhances both the physical plausibility and overall visual quality of the output. The resulting pipeline achieves near real-time performance with photorealistic rendering quality, effectively bridging the gap between the efficiency of modern generative models and the precision of traditional physically based rendering. Furthermore, we demonstrate the versatility of our framework by introducing a lightweight, adapter-based module that efficiently repurposes the pretrained forward model for the inverse rendering task of intrinsic decomposition.
>
---
#### [replaced 091] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11680](https://arxiv.org/pdf/2603.11680)**

> **作者:** Cao Thien Tan; Phan Thi Thu Trang; Do Nghiem Duc; Ho Ngoc Anh; Hanyang Zhuang; Nguyen Duc Dung
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Hybrid CNN-Transformer architectures achieve strong results in image super-resolution, but scaling attention windows or convolution kernels significantly increases computational cost, limiting deployment on resource-constrained devices. We present UCAN, a lightweight network that unifies convolution and attention to expand the effective receptive field efficiently. UCAN combines window-based spatial attention with a Hedgehog Attention mechanism to model both local texture and long-range dependencies, and introduces a distillation-based large-kernel module to preserve high-frequency structure without heavy computation. In addition, we employ cross-layer parameter sharing to further reduce complexity. On Manga109 ($4\times$), UCAN-L achieves 31.63 dB PSNR with only 48.4G MACs, surpassing recent lightweight models. On BSDS100, UCAN attains 27.79 dB, outperforming methods with significantly larger models. Extensive experiments show that UCAN achieves a superior trade-off between accuracy, efficiency, and scalability, making it well-suited for practical high-resolution image restoration.
>
---
#### [replaced 092] STRADAViT: Towards a Foundational Model for Radio Astronomy through Self-Supervised Transfer
- **分类: astro-ph.IM; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29660](https://arxiv.org/pdf/2603.29660)**

> **作者:** Andrea DeMarco; Ian Fenech Conti; Hayley Camilleri; Ardiana Bushi; Simone Riggi
>
> **备注:** 19 pages
>
> **摘要:** Next-generation radio astronomy surveys are delivering millions of resolved sources, but robust and scalable morphology analysis remains difficult across heterogeneous telescopes and imaging pipelines. We present STRADAViT, a self-supervised Vision Transformer continued-pretraining framework for learning transferable encoders from radio astronomy imagery. The framework combines mixed-survey data curation, radio astronomy-aware training-view generation, and a ViT-MAE-initialized encoder family with optional register tokens, and supports reconstruction-only, contrastive-only, and two-stage branches. Our pretraining dataset comprises radio astronomy cutouts drawn from four complementary sources: MeerKAT, ASKAP, LOFAR/LoTSS, and SKA SDC1 simulated data. We evaluate transfer with linear probing and fine-tuning on three morphology benchmarks spanning binary and multi-class settings: MiraBest, LoTSS DR2, and Radio Galaxy Zoo. Relative to the ViT-MAE initialization used for continued pretraining, the best two-stage models improve Macro-F1 in all reported linear-probe settings and in two of three fine-tuning settings, with the largest gain on RGZ DR1. Relative to DINOv2, gains are selective: the best two-stage models achieve higher mean Macro-F1 than the strongest DINOv2 baseline on LoTSS DR2 and RGZ DR1 under linear probing, and on MiraBest and RGZ DR1 under fine-tuning. A targeted DINOv2 initialization ablation further indicates that the adaptation recipe is not specific to the ViT-MAE starting point. The ViT-MAE-based STRADAViT checkpoint is retained as the released checkpoint because it combines competitive transfer with lower token count and downstream cost than the DINOv2-based alternative.
>
---
#### [replaced 093] From Reasoning to Pixels: Benchmarking the Alignment Gap in Unified Multimodal Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态对齐任务，旨在解决统一多模态模型中跨模态表示不一致的问题。通过构建基准测试UReason，评估不同生成方式的效果，揭示当前模型在语义对齐上的不足。**

- **链接: [https://arxiv.org/pdf/2602.08336](https://arxiv.org/pdf/2602.08336)**

> **作者:** Cheng Yang; Chufan Shi; Bo Shui; Yaokang Wu; Muzi Tao; Huijuan Wang; Ivan Yee Lee; Yong Liu; Xuezhe Ma; Taylor Berg-Kirkpatrick
>
> **备注:** Project page: this https URL
>
> **摘要:** Unified multimodal models (UMMs) aim to integrate multimodal understanding and generation within a unified architecture, yet it remains unclear to what extent their representations are truly aligned across modalities. To investigate this question, we use reasoning-guided image generation as a diagnostic task, where models produce textual reasoning first and then generate images. We introduce UReason, a benchmark for evaluating cross-modal alignment in this paradigm, consisting of 2,000 manually curated instances spanning five reasoning-intensive tasks: Code, Arithmetic, Spatial, Attribute and Text. To enable controlled analysis, we develop an evaluation framework that compares direct generation, reasoning-guided generation and de-contextualized generation, which conditions only on the refined prompt extracted from reasoning. Across eight widely used UMMs, while we find that reasoning-guided generation yields improvements over direct generation, somewhat surprisingly, de-contextualized generation consistently outperforms reasoning-guided generation by a large margin. Our results suggest that the intended visual semantics in textual reasoning are not reliably reflected in the generated images. This finding indicates that, despite unified design and training, current UMMs still do not robustly align representations across modalities. Overall, UReason serves as a practical litmus test for cross-modal alignment and provides a challenging benchmark for developing next-generation, more tightly aligned UMMs.
>
---
#### [replaced 094] Attentive Dilated Convolution for Automatic Sleep Staging using Force-directed Layout
- **分类: eess.SP; cs.CV; cs.HC; cs.LG**

- **链接: [https://arxiv.org/pdf/2409.01962](https://arxiv.org/pdf/2409.01962)**

> **作者:** Md Jobayer; Md Mehedi Hasan Shawon; Tasfin Mahmud; Md. Borhan Uddin Antor; Arshad M. Chowdhury
>
> **备注:** Has been accepted for publication in IEEE Access
>
> **摘要:** Sleep stages play an important role in identifying sleep patterns and diagnosing sleep disorders. In this study, we present an automated sleep stage classifier called the Attentive Dilated Convolutional Neural Network (AttDiCNN), which uses deep learning methodologies to address challenges related to data heterogeneity, computational complexity, and reliable and automatic sleep staging. We employed a force-directed layout based on the visibility graph to capture the most significant information from the EEG signals, thereby representing the spatial-temporal features. The proposed network consists of three modules: the Localized Spatial Feature Extraction Network (LSFE), Spatio-Temporal-Temporal Long Retention Network (S2TLR), and Global Averaging Attention Network (G2A). The LSFE captures spatial information from sleep data, the S2TLR is designed to extract the most pertinent information in long-term contexts, and the G2A reduces computational overhead by aggregating information from the LSFE and S2TLR. We evaluated the performance of our model on three comprehensive and publicly accessible datasets, achieving state-of-the-art accuracies of 98.56%, 99.66%, and 99.08% for the EDFX, HMC, and NCH datasets, respectively, while maintaining a low computational complexity with 1.4 M parameters. Our proposed architecture surpasses existing methodologies in several performance metrics, thereby proving its potential as an automated tool for clinical settings.
>
---
#### [replaced 095] TeamPath: Building MultiModal Pathology Experts with Reasoning AI Copilots
- **分类: q-bio.QM; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17652](https://arxiv.org/pdf/2511.17652)**

> **作者:** Tianyu Liu; Weihao Xuan; Hao Wu; Peter Humphrey; Marcello DiStasio; Mohamed Kahila; Alfonso Garcia Tan; Heli Qi; Rui Yang; Simeng Han; Tinglin Huang; Fang Wu; Chen Liu; Qingyu Chen; Nan Liu; Irene Li; Hua Xu; Hongyu Zhao
>
> **备注:** 45 pages, 6 figures
>
> **摘要:** Advances in AI have introduced several strong models in computational pathology to usher it into the era of multi-modal diagnosis, analysis, and interpretation. However, the current pathology-specific visual language models still lack capacities in making the diagnosis with rigorous reasoning paths as well as handling divergent tasks, and thus, challenges of building AI Copilots for real scenarios still exist. Here we introduce TeamPath, an AI system powered by reinforcement learning and router-enhanced solutions based on large-scale histopathology multimodal datasets, to work as a virtual assistant for expert-level disease diagnosis, patch-level information summarization, and cross-modality generation to integrate transcriptomic information for clinical usage. We also collaborate with pathologists from Yale School of Medicine to demonstrate that TeamPath can assist them in working more efficiently by identifying and correcting expert conclusions and reasoning paths. We also discuss the human evaluation results to support the reasoning quality from TeamPath. Overall, TeamPath can flexibly choose the best settings according to the needs, and serve as an innovative and reliable system for information communication across different modalities and experts.
>
---
#### [replaced 096] PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.18336](https://arxiv.org/pdf/2601.18336)**

> **作者:** Isaac Deutsch; Nicolas Moënne-Loccoz; Gavriel State; Zan Gojcic
>
> **备注:** For more details and updates, please visit our project website: this https URL
>
> **摘要:** Multi-view 3D reconstruction methods remain highly sensitive to photometric inconsistencies arising from camera optical characteristics and variations in image signal processing (ISP). Existing mitigation strategies such as per-frame latent variables or affine color corrections lack physical grounding and generalize poorly to novel views. We propose the Physically-Plausible ISP (PPISP) correction module, which disentangles camera-intrinsic and capture-dependent effects through physically based and interpretable transformations. A dedicated PPISP controller, trained on the input views, predicts ISP parameters for novel viewpoints, analogous to auto exposure and auto white balance in real cameras. This design enables realistic and fair evaluation on novel views without access to ground-truth images. PPISP achieves state-of-the-art performance on standard benchmarks, while providing intuitive control and supporting the integration of metadata when available. The source code is available at: this https URL
>
---
#### [replaced 097] Toward Generalizable Forgery Detection and Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21210](https://arxiv.org/pdf/2503.21210)**

> **作者:** Yueying Gao; Dongliang Chang; Bingyao Yu; Haotian Qin; Muxi Diao; Lei Chen; Kongming Liang; Zhanyu Ma
>
> **备注:** Accepted to IEEE TIP
>
> **摘要:** Accurate and interpretable detection of AI-generated images is essential for mitigating risks associated with AI misuse. However, the substantial domain gap among generative models makes it challenging to develop a generalizable forgery detection model. Moreover, since every pixel in an AI-generated image is synthesized, traditional saliency-based forgery explanation methods are not well suited for this task. To address these challenges, we formulate detection and explanation as a unified Forgery Detection and Reasoning task (FDR-Task), leveraging Multi-Modal Large Language Models (MLLMs) to provide accurate detection through reliable reasoning over forgery attributes. To facilitate this task, we introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 120K images across 10 generative models, with 378K reasoning annotations on forgery attributes, enabling comprehensive evaluation of the FDR-Task. Furthermore, we propose FakeReasoning, a forgery detection and reasoning framework with three key components: 1) a dual-branch visual encoder that integrates CLIP and DINO to capture both high-level semantics and low-level artifacts; 2) a Forgery-Aware Feature Fusion Module that leverages DINO's attention maps and cross-attention mechanisms to guide MLLMs toward forgery-related clues; 3) a Classification Probability Mapper that couples language modeling and forgery detection, enhancing overall performance. Experiments across multiple generative models demonstrate that FakeReasoning not only achieves robust generalization but also outperforms state-of-the-art methods on both detection and reasoning tasks. The code is available at: this https URL.
>
---
#### [replaced 098] Why Can't I Open My Drawer? Mitigating Object-Driven Shortcuts in Zero-Shot Compositional Action Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.16211](https://arxiv.org/pdf/2601.16211)**

> **作者:** Geo Ahn; Inwoong Lee; Taeoh Kim; Minho Shim; Dongyoon Wee; Jinwoo Choi
>
> **备注:** The code is available at this https URL
>
> **摘要:** Zero-Shot Compositional Action Recognition (ZS-CAR) requires recognizing novel verb-object combinations composed of previously observed primitives. In this work, we tackle a key failure mode: models predict verbs via object-driven shortcuts (i.e., relying on the labeled object class) rather than temporal evidence. We argue that sparse compositional supervision and verb-object learning asymmetry can promote object-driven shortcut learning. Our analysis with proposed diagnostic metrics shows that existing methods overfit to training co-occurrence patterns and underuse temporal verb cues, resulting in weak generalization to unseen compositions. To address object-driven shortcuts, we propose Robust COmpositional REpresentations (RCORE) with two components. Co-occurrence Prior Regularization (CPR) adds explicit supervision for unseen compositions and regularizes the model against frequent co-occurrence priors by treating them as hard negatives. Temporal Order Regularization for Composition (TORC) enforces temporal-order sensitivity to learn temporally grounded verb representations. Across Sth-com and EK100-com, RCORE reduces shortcut diagnostics and consequently improves compositional generalization.
>
---
