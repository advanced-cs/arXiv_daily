# 计算机视觉 cs.CV

- **最新发布 145 篇**

- **更新 101 篇**

## 最新发布

#### [new 001] Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning
- **分类: cs.CV**

- **简介: 该论文属于手术动作识别任务，旨在提升手术技能评估与规划。通过构建大规模数据集和基础模型，实现跨专业手术动作的准确识别，并应用于手术规划与评估。**

- **链接: [https://arxiv.org/pdf/2603.12787](https://arxiv.org/pdf/2603.12787)**

> **作者:** Mengya Xu; Daiyun Shen; Jie Zhang; Hon Chi Yip; Yujia Gao; Cheng Chen; Dillan Imans; Yonghao Long; Yiru Ye; Yixiao Liu; Rongyun Mai; Kai Chen; Hongliang Ren; Yutong Ban; Guangsuo Wang; Francis Wong; Chi-Fai Ng; Kee Yuan Ngiam; Russell H. Taylor; Daguang Xu; Yueming Jin; Qi Dou
>
> **备注:** 34 pages, 8 figures
>
> **摘要:** Artificial intelligence, imaging, and large language models have the potential to transform surgical practice, training, and automation. Understanding and modeling of basic surgical actions (BSA), the fundamental unit of operation in any surgery, is important to drive the evolution of this field. In this paper, we present a BSA dataset comprising 10 basic actions across 6 surgical specialties with over 11,000 video clips, which is the largest to date. Based on the BSA dataset, we developed a new foundation model that conducts general-purpose recognition of basic actions. Our approach demonstrates robust cross-specialist performance in experiments validated on datasets from different procedural types and various body parts. Furthermore, we demonstrate downstream applications enabled by the BAS foundation model through surgical skill assessment in prostatectomy using domain-specific knowledge, and action planning in cholecystectomy and nephrectomy using large vision-language models. Multinational surgeons' evaluation of the language model's output of the action planning explainable texts demonstrated clinical relevance. These findings indicate that basic surgical actions can be robustly recognized across scenarios, and an accurate BSA understanding model can essentially facilitate complex applications and speed up the realization of surgical superintelligence.
>
---
#### [new 002] Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶场景生成任务，旨在解决传统方法无法独立控制场景元素的问题，提出CompoSIA实现对场景结构、物体身份和自车动作的解耦控制。**

- **链接: [https://arxiv.org/pdf/2603.12864](https://arxiv.org/pdf/2603.12864)**

> **作者:** Yifan Zhan; Zhengqing Chen; Qingjie Wang; Zhuo He; Muyao Niu; Xiaoyang Guo; Wei Yin; Weiqiang Ren; Qian Zhang; Yinqiang Zheng
>
> **摘要:** A major challenge in autonomous driving is the "long tail" of safety-critical edge cases, which often emerge from unusual combinations of common traffic elements. Synthesizing these scenarios is crucial, yet current controllable generative models provide incomplete or entangled guidance, preventing the independent manipulation of scene structure, object identity, and ego actions. We introduce CompoSIA, a compositional driving video simulator that disentangles these traffic factors, enabling fine-grained control over diverse adversarial driving scenarios. To support controllable identity replacement of scene elements, we propose a noise-level identity injection, allowing pose-agnostic identity generation across diverse element poses, all from a single reference image. Furthermore, a hierarchical dual-branch action control mechanism is introduced to improve action controllability. Such disentangled control enables adversarial scenario synthesis-systematically combining safe elements into dangerous configurations that entangled generators cannot produce. Extensive comparisons demonstrate superior controllable generation quality over state-of-the-art baselines, with a 17% improvement in FVD for identity editing and reductions of 30% and 47% in rotation and translation errors for action control. Furthermore, downstream stress-testing reveals substantial planner failures: across editing modalities, the average collision rate of 3s increases by 173%.
>
---
#### [new 003] Less Data, Faster Convergence: Goal-Driven Data Optimization for Multimodal Instruction Tuning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态指令调优任务，旨在解决数据利用效率低的问题。通过GDO框架，优化训练数据选择，减少样本数量，加快收敛并提升准确率。**

- **链接: [https://arxiv.org/pdf/2603.12478](https://arxiv.org/pdf/2603.12478)**

> **作者:** Rujie Wu; Haozhe Zhao; Hai Ci; Yizhou Wang
>
> **摘要:** Multimodal instruction tuning is often compute-inefficient because training budgets are spread across large mixed image-video pools whose utility is highly uneven. We present Goal-Driven Data Optimization (GDO), a framework that computes six sample descriptors for each candidate and constructs optimized 1$\times$ training subsets for different goals. Under a fixed one-epoch Qwen3-VL-8B-Instruct training and evaluation recipe on 8 H20 GPUs, GDO uses far fewer training samples than the Uni-10x baseline while converging faster and achieving higher accuracy. Relative to the fixed 512k-sample Uni-10x baseline, GDO reaches the Uni-10x reference after 35.4k samples on MVBench, 26.6k on VideoMME, 27.3k on MLVU, and 34.7k on LVBench, while improving Accuracy by +1.38, +1.67, +3.08, and +0.84 percentage points, respectively. The gains are largest on MVBench and MLVU, while LVBench improves more modestly, consistent with its ultra-long-video setting and the mismatch between that benchmark and the short-video/image-dominant training pool. Across MinLoss, Diverse, Temp, and Temp+, stronger temporal emphasis yields steadily better long-video understanding behavior. Overall, GDO provides a goal-driven data optimization framework that enables faster convergence with fewer training samples under a fixed training protocol. Code is available at this https URL.
>
---
#### [new 004] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于LiDAR视图合成任务，解决无姿态依赖的几何重建问题。提出SG-NLF框架，结合光谱信息与几何一致性，提升重建质量与姿态精度。**

- **链接: [https://arxiv.org/pdf/2603.12903](https://arxiv.org/pdf/2603.12903)**

> **作者:** Yinuo Jiang; Jun Cheng; Yiran Wang; Cheng Cheng
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Neural Radiance Fields (NeRF) have shown remarkable success in image novel view synthesis (NVS), inspiring extensions to LiDAR NVS. However, most methods heavily rely on accurate camera poses for scene reconstruction. The sparsity and textureless nature of LiDAR data also present distinct challenges, leading to geometric holes and discontinuous surfaces. To address these issues, we propose SG-NLF, a pose-free LiDAR NeRF framework that integrates spectral information with geometric consistency. Specifically, we design a hybrid representation based on spectral priors to reconstruct smooth geometry. For pose optimization, we construct a confidence-aware graph based on feature compatibility to achieve global alignment. In addition, an adversarial learning strategy is introduced to enforce cross-frame consistency, thereby enhancing reconstruction quality. Comprehensive experiments demonstrate the effectiveness of our framework, especially in challenging low-frequency scenarios. Compared to previous state-of-the-art methods, SG-NLF improves reconstruction quality and pose accuracy by over 35.8% and 68.8%. Our work can provide a novel perspective for LiDAR view synthesis.
>
---
#### [new 005] Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于3D重建任务，旨在解决 aggregates 形貌分析难题。通过标记法实现低成本高精度的3D重建，并对比2D与3D形态差异。**

- **链接: [https://arxiv.org/pdf/2603.12667](https://arxiv.org/pdf/2603.12667)**

> **作者:** Haohang Huang; Jiayi Luo; Issam Qamhia; Erol Tutumluer; John M. Hart; Andrew J. Stolba
>
> **摘要:** Aggregates, serving as the main skeleton in assemblies of construction materials, are important functional components in various building and transportation infrastructures. They can be used in unbound layer applications, e.g. pavement base and railroad ballast, bound applications of cement concrete and asphalt concrete, and as riprap and large-sized primary crushed rocks. Information on the size and shape or morphology of aggregates can greatly facilitate the Quality Assurance/Quality Control (QA/QC) process by providing insights of aggregate behavior during composition and packing. A full 3D characterization of aggregate particle morphology is difficult both during production in a quarry and at a construction site. Many aggregate imaging approaches have been developed to quantify the particle morphology by computer vision, including 2D image-based approaches that analyze particle silhouettes and 3D scanning-based methods that require expensive devices such as 3D laser scanners or X-Ray Computed Tomography (CT) equipment. This paper presents a flexible and cost-effective photogrammetry-based approach for the 3D reconstruction of aggregate particles. The proposed approach follows a marker-based design that enables background suppression, point cloud stitching, and scale referencing to obtain high-quality aggregate models. The accuracy of the reconstruction results was validated against ground-truth for selected aggregate samples. Comparative analyses were conducted on 2D and 3D morphological properties of the selected samples. Significant differences were found between the 2D and 3D statistics. Based on the presented approach, 3D shape information of aggregates can be obtained easily and at a low cost, thus allowing convenient aggregate inspection, data collection, and 3D morphological analysis.
>
---
#### [new 006] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
- **分类: cs.CV**

- **简介: 该论文属于弱监督目标定位任务，解决领域迁移中预测偏差问题。提出SFDA-DeP方法，通过迭代修正偏差提升模型在新领域的分类与定位性能。**

- **链接: [https://arxiv.org/pdf/2603.12468](https://arxiv.org/pdf/2603.12468)**

> **作者:** Alexis Guichemerre; Banafsheh Karimian; Soufiane Belharbi; Natacha Gillet; Nicolas Thome; Pourya Shamsolmoali; Mohammadhadi Shateri; Luke McCaffrey; Eric Granger
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Weakly Supervised Object Localization (WSOL) models enable joint classification and region-of-interest localization in histology images using only image-class supervision. When deployed in a target domain, distributions shift remains a major cause of performance degradation, especially when applied on new organs or institutions with different staining protocols and scanner characteristics. Under stronger cross-domain shifts, WSOL predictions can become biased toward dominant classes, producing highly skewed pseudo-label distributions in the target domain. Source-Free (Unsupervised) Domain Adaptation (SFDA) methods are commonly employed to address domain shift. However, because they rely on self-training, the initial bias is reinforced over training iterations, degrading both classification and localization tasks. We identify this amplification of prediction bias as a primary obstacle to the SFDA of WSOL models in histopathology. This paper introduces \sfdadep, a method inspired by machine unlearning that formulates SFDA as an iterative process of identifying and correcting prediction bias. It periodically identifies target images from over-predicted classes and selectively reduces the predictive confidence for uncertain (high entropy) images, while preserving confident predictions. This process reduces the drift of decision boundaries and bias toward dominant classes. A jointly optimized pixel-level classifier further restores discriminative localization features under distribution shift. Extensive experiments on cross-organ and -center histopathology benchmarks (glas, CAMELYON-16, CAMELYON-17) with several WSOL models show that SFDA-DeP consistently improves classification and localization over state-of-the-art SFDA baselines. {\small Code: \href{this https URL}{this http URL}}
>
---
#### [new 007] Vision Verification Enhanced Fusion of VLMs for Efficient Visual Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言推理任务，解决多模型融合效率与准确性问题。通过引入焦点误差多样性与CKA-focal指标，结合遗传算法优化模型组合，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2603.12669](https://arxiv.org/pdf/2603.12669)**

> **作者:** Selim Furkan Tekin; Yichang Xu; Gaowen Liu; Ramana Rao Kompella; Margaret L. Loper; Ling Liu
>
> **摘要:** With the growing number and diversity of Vision-Language Models (VLMs), many works explore language-based ensemble, collaboration, and routing techniques across multiple VLMs to improve multi-model reasoning. In contrast, we address the diverse model selection using both vision and language modalities. We introduce focal error diversity to capture complementary reasoning across VLMs and a CKA-based focal diversity metric (CKA-focal) to measure disagreement in their visual embeddings. On the constructed ensemble surface from a pool of candidate VLMs, we applied a Genetic Algorithm to effectively prune out those component VLMs that do not add value to the fusion performance. We identify the best combination for each task as well as fuse the outputs of each VLMs in the model pool, and show that heterogeneous models can capture epistemic uncertainty dynamically and mitigate hallucinations. Our V3Fusion approach is capable of producing dual focal-diversity fused predictions with high performance for vision-language reasoning, even when there is no majority consensus or the majority of VLMs make incorrect predictions. Extensive experiments validate V3Fusion on four popular VLM benchmarks (A-OKVQA, MMMU, MMMU-Pro, and OCR-VQA). The results show that V3Fusion outperforms the best-performing VLM on MMMU by 8.09% and MMMU-Pro by 4.87% gain in accuracy. For generative tasks, V3Fusion outperforms Intern-VL2-8b and Qwen2.5-VL-7b, the top-2 VLM performers on both A-OKVQA and OCR-VQA. Our code and datasets are available at this https URL.
>
---
#### [new 008] Visual-ERM: Reward Modeling for Visual Equivalence
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉到代码的重建任务，解决强化学习中奖励信号对齐问题。提出Visual-ERM模型，通过细粒度视觉反馈提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.13224](https://arxiv.org/pdf/2603.13224)**

> **作者:** Ziyu Liu; Shengyuan Ding; Xinyu Fang; Xuanlang Dai; Penghui Yang; Jianze Liang; Jiaqi Wang; Kai Chen; Dahua Lin; Yuhang Zang
>
> **备注:** Project: this https URL
>
> **摘要:** Vision-to-code tasks require models to reconstruct structured visual inputs, such as charts, tables, and SVGs, into executable or structured representations with high visual fidelity. While recent Large Vision Language Models (LVLMs) achieve strong results via supervised fine-tuning, reinforcement learning remains challenging due to misaligned reward signals. Existing rewards either rely on textual rules or coarse visual embedding similarity, both of which fail to capture fine-grained visual discrepancies and are vulnerable to reward hacking. We propose Visual Equivalence Reward Model (Visual-ERM), a multimodal generative reward model that provides fine-grained, interpretable, and task-agnostic feedback to evaluate vision-to-code quality directly in the rendered visual space. Integrated into RL, Visual-ERM improves Qwen3-VL-8B-Instruct by +8.4 on chart-to-code and yields consistent gains on table and SVG parsing (+2.7, +4.1 on average), and further strengthens test-time scaling via reflection and revision. We also introduce VisualCritic-RewardBench (VC-RewardBench), a benchmark for judging fine-grained image-to-image discrepancies on structured visual data, where Visual-ERM at 8B decisively outperforms Qwen3-VL-235B-Instruct and approaches leading closed-source models. Our results suggest that fine-grained visual reward supervision is both necessary and sufficient for vision-to-code RL, regardless of task specificity.
>
---
#### [new 009] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决生成模型与人类视觉偏好对齐的问题。提出OARS框架，通过在线优化提升感知质量并保持保真度，实现先进性能。**

- **链接: [https://arxiv.org/pdf/2603.12811](https://arxiv.org/pdf/2603.12811)**

> **作者:** Shijie Zhao; Xuanyu Zhang; Bin Chen; Weiqi Li; Qunliang Xing; Kexin Zhang; Yan Wang; Junlin Li; Li Zhang; Jian Zhang; Tianfan Xue
>
> **备注:** Super-Resolution, Reinforcement Learning
>
> **摘要:** Aligning generative real-world image super-resolution models with human visual preference is challenging due to the perception--fidelity trade-off and diverse, unknown degradations. Prior approaches rely on offline preference optimization and static metric aggregation, which are often non-interpretable and prone to pseudo-diversity under strong conditioning. We propose OARS, a process-aware online alignment framework built on COMPASS, a MLLM-based reward that evaluates the LR to SR transition by jointly modeling fidelity preservation and perceptual gain with an input-quality-adaptive trade-off. To train COMPASS, we curate COMPASS-20K spanning synthetic and real degradations, and introduce a three-stage perceptual annotation pipeline that yields calibrated, fine-grained training labels. Guided by COMPASS, OARS performs progressive online alignment from cold-start flow matching to full-reference and finally reference-free RL via shallow LoRA optimization for on-policy exploration. Extensive experiments and user studies demonstrate consistent perceptual improvements while maintaining fidelity, achieving state-of-the-art performance on Real-ISR benchmarks.
>
---
#### [new 010] MemRoPE: Training-Free Infinite Video Generation via Evolving Memory Tokens
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频生成中的上下文丢失问题。提出MemRoPE框架，通过记忆令牌和在线RoPE索引实现无训练的无限视频生成。**

- **链接: [https://arxiv.org/pdf/2603.12513](https://arxiv.org/pdf/2603.12513)**

> **作者:** Youngrae Kim; Qixin Hu; C.-C. Jay Kuo; Peter A. Beerel
>
> **备注:** 9 pages main, 3 pages references, 6 pages appendix. Project page: this https URL
>
> **摘要:** Autoregressive diffusion enables real-time frame streaming, yet existing sliding-window caches discard past context, causing fidelity degradation, identity drift, and motion stagnation over long horizons. Current approaches preserve a fixed set of early tokens as attention sinks, but this static anchor cannot reflect the evolving content of a growing video. We introduce MemRoPE, a training-free framework with two co-designed components. Memory Tokens continuously compress all past keys into dual long-term and short-term streams via exponential moving averages, maintaining both global identity and recent dynamics within a fixed-size cache. Online RoPE Indexing caches unrotated keys and applies positional embeddings dynamically at attention time, ensuring the aggregation is free of conflicting positional phases. These two mechanisms are mutually enabling: positional decoupling makes temporal aggregation well-defined, while aggregation makes fixed-size caching viable for unbounded generation. Extensive experiments validate that MemRoPE outperforms existing methods in temporal coherence, visual fidelity, and subject consistency across minute- to hour-scale generation.
>
---
#### [new 011] CalliMaster: Mastering Page-level Chinese Calligraphy via Layout-guided Spatial Planning
- **分类: cs.CV**

- **简介: 该论文属于页面级书法生成任务，解决字符精度与布局组合的平衡问题。提出CalliMaster框架，通过分阶段生成实现高保真书法合成与编辑。**

- **链接: [https://arxiv.org/pdf/2603.12482](https://arxiv.org/pdf/2603.12482)**

> **作者:** Tianshuo Xu; Tiantian Hong; Zhifei Chen; Fei Chao; Ying-cong Chen
>
> **摘要:** Page-level calligraphy synthesis requires balancing glyph precision with layout composition. Existing character models lack spatial context, while page-level methods often compromise brushwork detail. In this paper, we present \textbf{CalliMaster}, a unified framework for controllable generation and editing that resolves this conflict by decoupling spatial planning from content synthesis. Inspired by the human cognitive process of ``planning before writing'', we introduce a coarse-to-fine pipeline \textbf{(Text $\rightarrow$ Layout $\rightarrow$ Image)} to tackle the combinatorial complexity of page-scale synthesis. Operating within a single Multimodal Diffusion Transformer, a spatial planning stage first predicts character bounding boxes to establish the global spatial arrangement. This intermediate layout then serves as a geometric prompt for the content synthesis stage, where the same network utilizes flow-matching to render high-fidelity brushwork. Beyond achieving state-of-the-art generation quality, this disentanglement supports versatile downstream capabilities. By treating the layout as a modifiable constraint, CalliMaster enables controllable semantic re-planning: users can resize or reposition characters while the model automatically harmonizes the surrounding void space and brush momentum. Furthermore, we demonstrate the framework's extensibility to artifact restoration and forensic analysis, providing a comprehensive tool for digital cultural heritage.
>
---
#### [new 012] VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model
- **分类: cs.CV**

- **简介: 该论文提出VGGT-World，解决3D世界建模中的几何一致性问题，通过预测冻结特征的时序演化，提升深度预测效果。**

- **链接: [https://arxiv.org/pdf/2603.12655](https://arxiv.org/pdf/2603.12655)**

> **作者:** Xiangyu Sun; Shijie Wang; Fengyi Zhang; Lin Liu; Caiyan Jia; Ziying Song; Zi Huang; Yadan Luo
>
> **摘要:** World models that forecast scene evolution by generating future video frames devote the bulk of their capacity to photometric details, yet the resulting predictions often remain geometrically inconsistent. We present VGGT-World, a geometry world model that side-steps video generation entirely and instead forecasts the temporal evolution of frozen geometry-foundation-model (GFM) features. Concretely, we repurpose the latent tokens of a frozen VGGT as the world state and train a lightweight temporal flow transformer to autoregressively predict their future trajectory. Two technical challenges arise in this high-dimensional (d=1024) feature space: (i) standard velocity-prediction flow matching collapses, and (ii) autoregressive rollout suffers from compounding exposure bias. We address the first with a clean-target (z-prediction) parameterization that yields a substantially higher signal-to-noise ratio, and the second with a two-stage latent flow-forcing curriculum that progressively conditions the model on its own partially denoised rollouts. Experiments on KITTI, Cityscapes, and TartanAir demonstrate that VGGT-World significantly outperforms the strongest baselines in depth forecasting while running 3.6-5 times faster with only 0.43B trainable parameters, establishing frozen GFM features as an effective and efficient predictive state for 3D world modeling.
>
---
#### [new 013] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion
- **分类: cs.CV**

- **简介: 该论文属于Ego→Exo误差检测任务，解决跨视角、时间错位和冗余问题。提出SAVA-X框架，通过视图对齐与双向融合提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12764](https://arxiv.org/pdf/2603.12764)**

> **作者:** Xiang Li; Heqian Qiu; Lanxiao Wang; Benliu Qiu; Fanman Meng; Linfeng Xu; Hongliang Li
>
> **备注:** This article was accepted by CVPR 2026
>
> **摘要:** Error detection is crucial in industrial training, healthcare, and assembly quality control. Most existing work assumes a single-view setting and cannot handle the practical case where a third-person (exo) demonstration is used to assess a first-person (ego) imitation. We formalize Ego$\rightarrow$Exo Imitation Error Detection: given asynchronous, length-mismatched ego and exo videos, the model must localize procedural steps on the ego timeline and decide whether each is erroneous. This setting introduces cross-view domain shift, temporal misalignment, and heavy redundancy. Under a unified protocol, we adapt strong baselines from dense video captioning and temporal action detection and show that they struggle in this cross-view regime. We then propose SAVA-X, an Align-Fuse-Detect framework with (i) view-conditioned adaptive sampling, (ii) scene-adaptive view embeddings, and (iii) bidirectional cross-attention fusion. On the EgoMe benchmark, SAVA-X consistently improves AUPRC and mean tIoU over all baselines, and ablations confirm the complementary benefits of its components. Code is available at this https URL.
>
---
#### [new 014] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods
- **分类: cs.CV**

- **简介: 该论文属于风场重建任务，解决稀疏传感器下屋顶风速分布的准确恢复问题。通过比较传统与深度学习方法，优化传感器配置与训练策略，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.13077](https://arxiv.org/pdf/2603.13077)**

> **作者:** Yihang Zhou; Chao Lin; Hideki Kikumoto; Ryozo Ooka; Sibo Cheng
>
> **摘要:** Real-time rooftop wind-speed distribution is important for the safe operation of drones and urban air mobility systems, wind control systems, and rooftop utilization. However, rooftop flows show strong nonlinearity, separation, and cross-direction variability, which make flow field reconstruction from sparse sensors difficult. This study develops a learning-from-observation framework using wind-tunnel experimental data obtained by Particle Image Velocimetry (PIV) and compares Kriging interpolation with three deep learning models: UNet, Vision Transformer Autoencoder (ViTAE), and Conditional Wasserstein GAN (CWGAN). We evaluate two training strategies, single wind-direction training (SDT) and mixed wind-direction training (MDT), across sensor densities from 5 to 30, test robustness under sensor position perturbations of plus or minus 1 grid, and optimize sensor placement via Proper Orthogonal Decomposition with QR decomposition. Results show that deep learning methods can reconstruct rooftop wind fields from sparse sensor data effectively. Compared with Kriging interpolation, the deep learning models improved SSIM by up to 32.7%, FAC2 by 24.2%, and NMSE by 27.8%. Mixed wind-direction training further improved performance, with gains of up to 173.7% in SSIM, 16.7% in FAC2, and 98.3% in MG compared with single-direction training. The results also show that sensor configuration, optimization, and training strategy should be considered jointly for reliable deployment. QR-based optimization improved robustness by up to 27.8% under sensor perturbations, although with metric-dependent trade-offs. Training on experimental rather than simulated data also provides practical guidance for method selection and sensor placement in different scenarios.
>
---
#### [new 015] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UAV场景变化描述任务，解决动态航拍图像中因视角变化导致的场景差异描述问题。设计HDC-CL方法与DALT模型，构建UAV-SCC数据集，提升变化描述准确性。**

- **链接: [https://arxiv.org/pdf/2603.12832](https://arxiv.org/pdf/2603.12832)**

> **作者:** Fuhai Chen; Pengpeng Huang; Junwen Wu; Hehong Zhang; Shiping Wang; Xiaoguang Ma; Xuri Ge
>
> **备注:** 20 pages,10 figures
>
> **摘要:** This paper proposes a novel task for UAV scene understanding - UAV Scene Change Captioning (UAV-SCC) - which aims to generate natural language descriptions of semantic changes in dynamic aerial imagery captured from a movable viewpoint. Unlike traditional change captioning that mainly describes differences between image pairs captured from a fixed camera viewpoint over time, UAV scene change captioning focuses on image-pair differences resulting from both temporal and spatial scene variations dynamically captured by a moving camera. The key challenge lies in understanding viewpoint-induced scene changes from UAV image pairs that share only partially overlapping scene content due to viewpoint shifts caused by camera rotation, while effectively exploiting the relative orientation between the two images. To this end, we propose a Hierarchical Dual-Change Collaborative Learning (HDC-CL) method for UAV scene change captioning. In particular, a novel transformer, \emph{i.e.} Dynamic Adaptive Layout Transformer (DALT) is designed to adaptively model diverse spatial layouts of the image pair, where the interrelated features derived from the overlapping and non-overlapping regions are learned within the flexible and unified encoding layer. Furthermore, we propose a Hierarchical Cross-modal Orientation Consistency Calibration (HCM-OCC) method to enhance the model's sensitivity to viewpoint shift directions, enabling more accurate change captioning. To facilitate in-depth research on this task, we construct a new benchmark dataset, named UAV-SCC dataset, for UAV scene change captioning. Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance on this task. The dataset and code will be publicly released upon acceptance of this paper.
>
---
#### [new 016] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation
- **分类: cs.CV**

- **简介: 该论文属于海洋动物分割任务，解决SAM在细粒度和频率信息感知上的不足。提出HFP-SAM框架，结合频率引导适配器和点选择模块，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.12708](https://arxiv.org/pdf/2603.12708)**

> **作者:** Pingping Zhang; Tianyu Yan; Yuhao Wang; Yang Liu; Tongdan Tang; Yili Ma; Long Lv; Feng Tian; Weibing Sun; and Huchuan Lu
>
> **备注:** Accepted by TIP2026. More modifications may be performed
>
> **摘要:** Marine Animal Segmentation (MAS) aims at identifying and segmenting marine animals from complex marine environments. Most of previous deep learning-based MAS methods struggle with the long-distance modeling issue. Recently, Segment Anything Model (SAM) has gained popularity in general image segmentation. However, it lacks of perceiving fine-grained details and frequency information. To this end, we propose a novel learning framework, named Hierarchical Frequency Prompted SAM (HFP-SAM) for high-performance MAS. First, we design a Frequency Guided Adapter (FGA) to efficiently inject marine scene information into the frozen SAM backbone through frequency domain prior masks. Additionally, we introduce a Frequency-aware Point Selection (FPS) to generate highlighted regions through frequency analysis. These regions are combined with the coarse predictions of SAM to generate point prompts and integrate into SAM's decoder for fine predictions. Finally, to obtain comprehensive segmentation masks, we introduce a Full-View Mamba (FVM) to efficiently extract spatial and channel contextual information with linear computational complexity. Extensive experiments on four public datasets demonstrate the superior performance of our approach. The source code is publicly available at this https URL.
>
---
#### [new 017] SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于手术视频生成任务，旨在解决手术AI和模拟中的数据不足与真实感问题。提出SAW模型，通过轻量信号生成高质量手术视频，提升动作识别与仿真效果。**

- **链接: [https://arxiv.org/pdf/2603.13024](https://arxiv.org/pdf/2603.13024)**

> **作者:** Sampath Rapuri; Lalithkumar Seenivasan; Dominik Schneider; Roger Soberanis-Mukul; Yufan He; Hao Ding; Jiru Xu; Chenhao Yu; Chenyan Jing; Pengfei Guo; Daguang Xu; Mathias Unberath
>
> **备注:** The manuscript is under review
>
> **摘要:** A surgical world model capable of generating realistic surgical action videos with precise control over tool-tissue interactions can address fundamental challenges in surgical AI and simulation -- from data scarcity and rare event synthesis to bridging the sim-to-real gap for surgical automation. However, current video generation methods, the very core of such surgical world models, require expensive annotations or complex structured intermediates as conditioning signals at inference, limiting their scalability. Other approaches exhibit limited temporal consistency across complex laparoscopic scenes and do not possess sufficient realism. We propose Surgical Action World (SAW) -- a step toward surgical action world modeling through video diffusion conditioned on four lightweight signals: language prompts encoding tool-action context, a reference surgical scene, tissue affordance mask, and 2D tool-tip trajectories. We design a conditional video diffusion approach that reformulates video-to-video diffusion into trajectory-conditioned surgical action synthesis. The backbone diffusion model is fine-tuned on a custom-curated dataset of 12,044 laparoscopic clips with lightweight spatiotemporal conditioning signals, leveraging a depth consistency loss to enforce geometric plausibility without requiring depth at inference. SAW achieves state-of-the-art temporal consistency (CD-FVD: 199.19 vs. 546.82) and strong visual quality on held-out test data. Furthermore, we demonstrate its downstream utility for (a) surgical AI, where augmenting rare actions with SAW-generated videos improves action recognition (clipping F1-score: 20.93% to 43.14%; cutting: 0.00% to 8.33%) on real test data, and (b) surgical simulation, where rendering tool-tissue interaction videos from simulator-derived trajectory points toward a visually faithful simulation engine.
>
---
#### [new 018] Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support with Multi-Center Clinical Validation
- **分类: cs.CV**

- **简介: 该论文提出Surg-R1，解决手术决策支持中可解释性不足的问题，通过分层推理框架提升模型在多中心数据上的表现。**

- **链接: [https://arxiv.org/pdf/2603.12430](https://arxiv.org/pdf/2603.12430)**

> **作者:** Jian Jiang; Chenxi Lin; Yiming Gu; Zengyi Qin; Zhitao Zeng; Kun Yuan; Yonghao Long; Xiang Xia; Cheng Yuan; Yuqi Wang; Zijie Yue; Kunyi Yang; Yuting Zhang; Zhu Zhuo; Dian Qin; Xin Wang; NG Chi Fai; Brian Anthony; Daguang Xu; Guy Rosman; Ozanan Meireles; Zizhen Zhang; Nicolas Padoy; Hesheng Wang; Qi Dou; Yueming Jin; Yutong Ban
>
> **摘要:** Surgical scene understanding demands not only accurate predictions but also interpretable reasoning that surgeons can verify against clinical expertise. However, existing surgical vision-language models generate predictions without reasoning chains, and general-purpose reasoning models fail on compositional surgical tasks without domain-specific knowledge. We present Surg-R1, a surgical Vision-Language Model that addresses this gap through hierarchical reasoning trained via a four-stage pipeline. Our approach introduces three key contributions: (1) a three-level reasoning hierarchy decomposing surgical interpretation into perceptual grounding, relational understanding, and contextual reasoning; (2) the largest surgical chain-of-thought dataset with 320,000 reasoning pairs; and (3) a four-stage training pipeline progressing from supervised fine-tuning to group relative policy optimization and iterative self-improvement. Evaluation on SurgBench, comprising six public benchmarks and six multi-center external validation datasets from five institutions, demonstrates that Surg-R1 achieves the highest Arena Score (64.9%) on public benchmarks versus Gemini 3.0 Pro (46.1%) and GPT-5.1 (37.9%), outperforming both proprietary reasoning models and specialized surgical VLMs on the majority of tasks spanning instrument localization, triplet recognition, phase recognition, action recognition, and critical view of safety assessment, with a 15.2 percentage point improvement over the strongest surgical baseline on external validation.
>
---
#### [new 019] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback
- **分类: cs.CV**

- **简介: 该论文属于虚拟试衣图像质量评估任务，解决无参考情况下图像质量评价问题。提出VTON-IQA框架，通过人类反馈进行图像级质量评估，并构建大规模标注数据集进行模型评估。**

- **链接: [https://arxiv.org/pdf/2603.13057](https://arxiv.org/pdf/2603.13057)**

> **作者:** Yuki Hirakawa; Takashi Wada; Ryotaro Shimizu; Takuya Furusawa; Yuki Saito; Ryosuke Araki; Tianwei Chen; Fan Mo; Yoshimitsu Aoki
>
> **摘要:** Given a person image and a garment image, image-based Virtual Try-ON (VTON) synthesizes a try-on image of the person wearing the target garment. As VTON systems become increasingly important in practical applications such as fashion e-commerce, reliable evaluation of their outputs has emerged as a critical challenge. In real-world scenarios, ground-truth images of the same person wearing the target garment are typically unavailable, making reference-based evaluation impractical. Moreover, widely used distribution-level metrics such as Fréchet Inception Distance and Kernel Inception Distance measure dataset-level similarity and fail to reflect the perceptual quality of individual generated images. To address these limitations, we propose Image Quality Assessment for Virtual Try-On (VTON-IQA), a reference-free framework for human-aligned, image-level quality assessment without requiring ground-truth images. To model human perceptual judgments, we construct VTON-QBench, a large-scale human-annotated benchmark comprising 62,688 try-on images generated by 14 representative VTON models and 431,800 quality annotations collected from 13,838 qualified annotators. To the best of our knowledge, this is the largest dataset to date for human subjective evaluation in virtual try-on. Evaluating virtual try-on quality requires verifying both garment fidelity and the preservation of person-specific details. To explicitly model such interactions, we introduce an Interleaved Cross-Attention module that extends standard transformer blocks by inserting a cross-attention layer between self-attention and MLP in the latter blocks. Extensive experiments show that VTON-IQA achieves reliable human-aligned image-level quality prediction. Moreover, we conduct a comprehensive benchmark evaluation of 14 representative VTON models using VTON-IQA.
>
---
#### [new 020] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D医学图像分析任务，旨在解决标注数据稀缺问题。通过自监督和半监督学习，提升腹部创伤检测与分类效果。**

- **链接: [https://arxiv.org/pdf/2603.12514](https://arxiv.org/pdf/2603.12514)**

> **作者:** Shivam Chaudhary; Sheethal Bhat; Andreas Maier
>
> **备注:** 9 pages, 6 figures, 6 tables. The code is available at this https URL
>
> **摘要:** Accurate detection and localization of traumatic injuries in abdominal CT scans remains a critical challenge in emergency radiology, primarily due to severe scarcity of annotated medical data. This paper presents a label-efficient approach combining self-supervised pre-training with semi-supervised detection for 3D medical image analysis. We employ patch-based Masked Image Modeling (MIM) to pre-train a 3D U-Net encoder on 1,206 CT volumes without annotations, learning robust anatomical representations. The pretrained encoder enables two downstream clinical tasks: 3D injury detection using VDETR with Vertex Relative Position Encoding, and multi-label injury classification. For detection, semi-supervised learning with 2,000 unlabeled volumes and consistency regularization achieves 56.57% validation mAP@0.50 and 45.30% test mAP@0.50 with only 144 labeled training samples, representing a 115% improvement over supervised-only training. For classification, expanding to 2,244 labeled samples yields 94.07% test accuracy across seven injury categories using only a frozen encoder, demonstrating immediately transferable self-supervised features. Our results validate that self-supervised pre-training combined with semi-supervised learning effectively addresses label scarcity in medical imaging, enabling robust 3D object detection with limited annotations.
>
---
#### [new 021] SDF-Net: Structure-Aware Disentangled Feature Learning for Opticall-SAR Ship Re-identification
- **分类: cs.CV**

- **简介: 该论文属于跨模态船舶重识别任务，解决光学与SAR图像间辐射差异带来的匹配难题。提出SDF-Net，通过结构一致性约束和特征解耦提升识别性能。**

- **链接: [https://arxiv.org/pdf/2603.12588](https://arxiv.org/pdf/2603.12588)**

> **作者:** Furui Chen; Han Wang; Yuhan Sun; Jianing You; Yixuan Lv; Zhuang Zhou; Hong Tan; Shengyang Li
>
> **摘要:** Cross-modal ship re-identification (ReID) between optical and synthetic aperture radar (SAR) imagery is fundamentally challenged by the severe radiometric discrepancy between passive optical imaging and coherent active radar sensing. While existing approaches primarily rely on statistical distribution alignment or semantic matching, they often overlook a critical physical prior: ships are rigid objects whose geometric structures remain stable across sensing modalities, whereas texture appearance is highly modality-dependent. In this work, we propose SDF-Net, a Structure-Aware Disentangled Feature Learning Network that systematically incorporates geometric consistency into optical--SAR ship ReID. Built upon a ViT backbone, SDF-Net introduces a structure consistency constraint that extracts scale-invariant gradient energy statistics from intermediate layers to robustly anchor representations against radiometric variations. At the terminal stage, SDF-Net disentangles the learned representations into modality-invariant identity features and modality-specific characteristics. These decoupled cues are then integrated through a parameter-free additive residual fusion, effectively enhancing discriminative power. Extensive experiments on the HOSS-ReID dataset demonstrate that SDF-Net consistently outperforms existing state-of-the-art methods. The code and trained models are publicly available at this https URL.
>
---
#### [new 022] CM-Bench: A Comprehensive Cross-Modal Feature Matching Benchmark Bridging Visible and Infrared Images
- **分类: cs.CV**

- **简介: 该论文属于跨模态特征匹配任务，旨在解决红外与可见光图像匹配困难的问题。提出CM-Bench基准，涵盖多种算法并引入新数据集以提升匹配性能。**

- **链接: [https://arxiv.org/pdf/2603.12690](https://arxiv.org/pdf/2603.12690)**

> **作者:** Liangzheng Sun; Mengfan He; Xingyu Shao; Binbin Li; Zhiqiang Yan; Chunyu Li; Ziyang Meng; Fei Xing
>
> **摘要:** Infrared-visible (IR-VIS) feature matching plays an essential role in cross-modality visual localization, navigation and perception. Along with the rapid development of deep learning techniques, a number of representative image matching methods have been proposed. However, crossmodal feature matching is still a challenging task due to the significant appearance difference. A significant gap for cross-modal feature matching research lies in the absence of standardized benchmarks and metrics for evaluations. In this paper, we introduce a comprehensive cross-modal feature matching benchmark, CM-Bench, which encompasses 30 feature matching algorithms across diverse cross-modal datasets. Specifically, state-of-the-art traditional and deep learning-based methods are first summarized and categorized into sparse, semidense, and dense methods. These methods are evaluated by different tasks including homography estimation, relative pose estimation, and feature-matching-based geo-localization. In addition, we introduce a classification-network-based adaptive preprocessing front-end that automatically selects suitable enhancement strategies before matching. We also present a novel infrared-satellite cross-modal dataset with manually annotated ground-truth correspondences for practical geo-localization evaluation. The dataset and resource will be available at: this https URL.
>
---
#### [new 023] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建安全任务，解决资源靶向攻击导致的高斯过度增长问题。通过频域滤波和谱正则化实现防御，提升系统稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2603.12796](https://arxiv.org/pdf/2603.12796)**

> **作者:** Yang Chen; Yi Yu; Jiaming He; Yueqi Duan; Zheng Zhu; Yap-Peng Tan
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack. This attack poisons training images, excessively inducing Gaussian growth to cause resource exhaustion. Although efficiency-oriented methods such as smoothing, thresholding, and pruning have been explored, these spatial-domain strategies operate on visible structures but overlook how stealthy perturbations distort the underlying spectral behaviors of training data. As a result, poisoned inputs introduce abnormal high-frequency amplifications that mislead 3DGS into interpreting noisy patterns as detailed structures, ultimately causing unstable Gaussian overgrowth and degraded scene fidelity. To address this, we propose \textbf{Spectral Defense} in Gaussian and image fields. We first design a 3D frequency filter to selectively prune Gaussians exhibiting abnormally high frequencies. Since natural scenes also contain legitimate high-frequency structures, directly suppressing high frequencies is insufficient, and we further develop a 2D spectral regularization on renderings, distinguishing naturally isotropic frequencies while penalizing anisotropic angular energy to constrain noisy patterns. Experiments show that our defense builds robust, accurate, and secure 3DGS, suppressing overgrowth by up to $5.92\times$, reducing memory by up to $3.66\times$, and improving speed by up to $4.34\times$ under attacks.
>
---
#### [new 024] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肺部疾病分类任务，旨在解决CT影像诊断中的性别不公平问题。通过改进的注意力MIL模型和对抗性机制，提升诊断公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.12988](https://arxiv.org/pdf/2603.12988)**

> **作者:** Aditya Parikh; Aasa Feragen
>
> **摘要:** We present a fairness-aware framework for multi-class lung disease diagnosis from chest CT volumes, developed for the Fair Disease Diagnosis Challenge at the PHAROS-AIF-MIH Workshop (CVPR 2026). The challenge requires classifying CT scans into four categories -- Healthy, COVID-19, Adenocarcinoma, and Squamous Cell Carcinoma -- with performance measured as the average of per-gender macro F1 scores, explicitly penalizing gender-inequitable predictions. Our approach addresses two core difficulties: the sparse pathological signal across hundreds of slices, and a severe demographic imbalance compounded across disease class and gender. We propose an attention-based Multiple Instance Learning (MIL) model on a ConvNeXt backbone that learns to identify diagnostically relevant slices without slice-level supervision, augmented with a Gradient Reversal Layer (GRL) that adversarially suppresses gender-predictive structure in the learned scan representation. Training incorporates focal loss with label smoothing, stratified cross-validation over joint (class, gender) strata, and targeted oversampling of the most underrepresented subgroup. At inference, all five-fold checkpoints are ensembled with horizontal-flip test-time augmentation via soft logit voting and out-of-the-fold threshold optimization for robustness. Our model achieves a mean validation competition score of 0.685 (std - 0.030), with the best single fold reaching 0.759. All training and inference code is publicly available at this https URL
>
---
#### [new 025] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型记忆训练数据带来的版权和隐私问题。通过引入RAPTA和ADMCD方法，提升生成多样性并检测复制内容。**

- **链接: [https://arxiv.org/pdf/2603.13070](https://arxiv.org/pdf/2603.13070)**

> **作者:** Yunzhuo Chen; Jordan Vice; Naveed Akhtar; Nur Al Hasan Haldar; Ajmal Mian
>
> **摘要:** State-of-the-art text-to-image diffusion models can produce impressive visuals but may memorize and reproduce training images, creating copyright and privacy risks. Existing prompt perturbations applied at inference time, such as random token insertion or embedding noise, may lower copying but often harm image-prompt alignment and overall fidelity. To address this, we introduce two complementary methods. First, Region-Aware Prompt Augmentation (RAPTA) uses an object detector to find salient regions and turn them into semantically grounded prompt variants, which are randomly sampled during training to increase diversity, while maintaining semantic alignment. Second, Attention-Driven Multimodal Copy Detection (ADMCD) aggregates local patch, global semantic, and texture cues with a lightweight transformer to produce a fused representation, and applies simple thresholded decision rules to detect copying without training with large annotated datasets. Experiments show that RAPTA reduces overfitting while maintaining high synthesis quality, and that ADMCD reliably detects copying, outperforming single-modal metrics.
>
---
#### [new 026] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多目标跟踪任务，解决在线跟踪中的身份切换问题。提出FC-Track框架，通过抑制重叠导致的错误关联，提升跟踪可靠性。**

- **链接: [https://arxiv.org/pdf/2603.12758](https://arxiv.org/pdf/2603.12758)**

> **作者:** Cheng Ju; Zejing Zhao; Akio Namiki
>
> **摘要:** Reliable multi-object tracking (MOT) is essential for robotic systems operating in complex and dynamic environments. Despite recent advances in detection and association, online MOT methods remain vulnerable to identity switches caused by frequent occlusions and object overlap, where incorrect associations can propagate over time and degrade tracking reliability. We present a lightweight post-association correction framework (FC-Track) for online MOT that explicitly targets overlap-induced mismatches during inference. The proposed method suppresses unreliable appearance updates under high-overlap conditions using an Intersection over Area (IoA)-based filtering strategy, and locally corrects detection-to-tracklet mismatches through appearance similarity comparison within overlapped tracklet pairs. By preventing short-term mismatches from propagating, our framework effectively mitigates long-term identity switches without resorting to global optimization or re-identification. The framework operates online without global optimization or re-identification, making it suitable for real-time robotic applications. We achieve 81.73 MOTA, 82.81 IDF1, and 66.95 HOTA on the MOT17 test set with a running speed of 5.7 FPS, and 77.52 MOTA, 80.90 IDF1, and 65.67 HOTA on the MOT20 test set with a running speed of 0.6 FPS. Specifically, our framework FC-Track produces only 29.55% long-term identity switches, which is substantially lower than existing online trackers. Meanwhile, our framework maintains state-of-the-art performance on the MOT20 benchmark.
>
---
#### [new 027] HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks
- **分类: cs.CV**

- **简介: 该论文提出HiFICL，解决多模态任务中In-Context Learning（ICL）性能不稳定和计算成本高的问题。通过构建可学习上下文、低秩分解和端到端目标，提升ICL效果。**

- **链接: [https://arxiv.org/pdf/2603.12760](https://arxiv.org/pdf/2603.12760)**

> **作者:** Xiaoyu Li; Yuhang Liu; Zheng Luo; Xuanshuo Kang; Fangqi Lou; Xiaohua Wu; Zihan Xiong
>
> **备注:** Accepted to CVPR 2026. Code available at this https URL
>
> **摘要:** In-Context Learning (ICL) is a significant paradigm for Large Multimodal Models (LMMs), using a few in-context demonstrations (ICDs) for new task adaptation. However, its performance is sensitive to demonstration configurations and computationally expensive. Mathematically, the influence of these demonstrations can be decomposed into a dynamic mixture of the standard attention output and the context values. Current approximation methods simplify this process by learning a "shift vector". Inspired by the exact decomposition, we introduce High-Fidelity In-Context Learning (HIFICL) to more faithfully model the ICL mechanism. HIFICL consists of three key components: 1) a set of "virtual key-value pairs" to act as a learnable context, 2) a low-rank factorization for stable and regularized training, and 3) a simple end-to-end training objective. From another perspective, this mechanism constitutes a form of context-aware Parameter-Efficient Fine-Tuning (PEFT). Extensive experiments show that HiFICL consistently outperforms existing approximation methods on several multimodal benchmarks. The code is available at this https URL.
>
---
#### [new 028] From Sparse to Dense: Multi-View GRPO for Flow Models via Augmented Condition Space
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决传统方法在样本关系探索不足的问题。提出MV-GRPO，通过扩展条件空间实现多视角奖励映射，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.12648](https://arxiv.org/pdf/2603.12648)**

> **作者:** Jiazi Bu; Pengyang Ling; Yujie Zhou; Yibin Wang; Yuhang Zang; Tianyi Wei; Xiaohang Zhan; Jiaqi Wang; Tong Wu; Xingang Pan; Dahua Lin
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a powerful framework for preference alignment in text-to-image (T2I) flow models. However, we observe that the standard paradigm where evaluating a group of generated samples against a single condition suffers from insufficient exploration of inter-sample relationships, constraining both alignment efficacy and performance ceilings. To address this sparse single-view evaluation scheme, we propose Multi-View GRPO (MV-GRPO), a novel approach that enhances relationship exploration by augmenting the condition space to create a dense multi-view reward mapping. Specifically, for a group of samples generated from one prompt, MV-GRPO leverages a flexible Condition Enhancer to generate semantically adjacent yet diverse captions. These captions enable multi-view advantage re-estimation, capturing diverse semantic attributes and providing richer optimization signals. By deriving the probability distribution of the original samples conditioned on these new captions, we can incorporate them into the training process without costly sample regeneration. Extensive experiments demonstrate that MV-GRPO achieves superior alignment performance over state-of-the-art methods.
>
---
#### [new 029] VQQA: An Agentic Approach for Video Evaluation and Quality Improvement
- **分类: cs.CV; cs.AI; cs.LG; cs.MA**

- **简介: 该论文提出VQQA，解决视频生成质量评估与优化问题。通过多智能体框架，利用视觉问答生成可解释反馈，提升视频生成质量。**

- **链接: [https://arxiv.org/pdf/2603.12310](https://arxiv.org/pdf/2603.12310)**

> **作者:** Yiwen Song; Tomas Pfister; Yale Song
>
> **摘要:** Despite rapid advancements in video generation models, aligning their outputs with complex user intent remains challenging. Existing test-time optimization methods are typically either computationally expensive or require white-box access to model internals. To address this, we present VQQA (Video Quality Question Answering), a unified, multi-agent framework generalizable across diverse input modalities and video generation tasks. By dynamically generating visual questions and using the resulting Vision-Language Model (VLM) critiques as semantic gradients, VQQA replaces traditional, passive evaluation metrics with human-interpretable, actionable feedback. This enables a highly efficient, closed-loop prompt optimization process via a black-box natural language interface. Extensive experiments demonstrate that VQQA effectively isolates and resolves visual artifacts, substantially improving generation quality in just a few refinement steps. Applicable to both text-to-video (T2V) and image-to-video (I2V) tasks, our method achieves absolute improvements of +11.57% on T2V-CompBench and +8.43% on VBench2 over vanilla generation, significantly outperforming state-of-the-art stochastic search and prompt optimization techniques.
>
---
#### [new 030] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉重建任务，旨在解决EEG信号中视觉信息 fidelity 丢失问题。通过融合多模态信息和改进对齐机制，提升重建准确性。**

- **链接: [https://arxiv.org/pdf/2603.12722](https://arxiv.org/pdf/2603.12722)**

> **作者:** Kaifan Zhang; Lihuo He; Junjie Ke; Yuqi Ji; Lukun Wu; Lizi Wang; Xinbo Gao
>
> **摘要:** Visual stimuli reconstruction from EEG remains challenging due to fidelity loss and representation shift. We propose CognitionCapturerPro, an enhanced framework that integrates EEG with multi-modal priors (images, text, depth, and edges) via collaborative training. Our core contributions include an uncertainty-weighted similarity scoring mechanism to quantify modality-specific fidelity and a fusion encoder for integrating shared representations. By employing a simplified alignment module and a pre-trained diffusion model, our method significantly outperforms the original CognitionCapturer on the THINGS-EEG dataset, improving Top-1 and Top-5 retrieval accuracy by 25.9% and 10.6%, respectively. Code is available at: this https URL.
>
---
#### [new 031] Multimodal OCR: Parse Anything from Documents
- **分类: cs.CV**

- **简介: 该论文提出Multimodal OCR（MOCR），解决文档中文本与图形联合解析问题，通过统一表示提升文档重建质量。**

- **链接: [https://arxiv.org/pdf/2603.13032](https://arxiv.org/pdf/2603.13032)**

> **作者:** Handong Zheng; Yumeng Li; Kaile Zhang; Liang Xin; Guangwei Zhao; Hao Liu; Jiayu Chen; Jie Lou; Jiyu Qiu; Qi Fu; Rui Yang; Shuo Jiang; Weijian Luo; Weijie Su; Weijun Zhang; Xingyu Zhu; Yabin Li; Yiwei ma; Yu Chen; Zhaohui Yu; Guang Yang; Colin Zhang; Lei Zhang; Yuliang Liu; Xiang Bai
>
> **摘要:** We present Multimodal OCR (MOCR), a document parsing paradigm that jointly parses text and graphics into unified textual representations. Unlike conventional OCR systems that focus on text recognition and leave graphical regions as cropped pixels, our method, termed this http URL, treats visual elements such as charts, diagrams, tables, and icons as first-class parsing targets, enabling systems to parse documents while preserving semantic relationships across elements. It offers several advantages: (1) it reconstructs both text and graphics as structured outputs, enabling more faithful document reconstruction; (2) it supports end-to-end training over heterogeneous document elements, allowing models to exploit semantic relations between textual and visual components; and (3) it converts previously discarded graphics into reusable code-level supervision, unlocking multimodal supervision embedded in existing documents. To make this paradigm practical at scale, we build a comprehensive data engine from PDFs, rendered webpages, and native SVG assets, and train a compact 3B-parameter model through staged pretraining and supervised fine-tuning. We evaluate this http URL from two perspectives: document parsing and structured graphics parsing. On document parsing benchmarks, it ranks second only to Gemini 3 Pro on our OCR Arena Elo leaderboard, surpasses existing open-source document parsing systems, and sets a new state of the art of 83.9 on olmOCR Bench. On structured graphics parsing, this http URL achieves higher reconstruction quality than Gemini 3 Pro across image-to-SVG benchmarks, demonstrating strong performance on charts, UI layouts, scientific figures, and chemical diagrams. These results show a scalable path toward building large-scale image-to-code corpora for multimodal pretraining. Code and models are publicly available at this https URL.
>
---
#### [new 032] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决领域迁移下的开放词汇检测问题。针对无标注数据的领域，提出ABRA方法，实现知识迁移。**

- **链接: [https://arxiv.org/pdf/2603.12409](https://arxiv.org/pdf/2603.12409)**

> **作者:** Mattia Bernardi; Chiara Cappellino; Matteo Mosconi; Enver Sangineto; Angelo Porrello; Simone Calderara
>
> **摘要:** Although recent Open-Vocabulary Object Detection architectures, such as Grounding DINO, demonstrate strong zero-shot capabilities, their performance degrades significantly under domain shifts. Moreover, many domains of practical interest, such as nighttime or foggy scenes, lack large annotated datasets, preventing direct fine-tuning. In this paper, we introduce Aligned Basis Relocation for Adaptation(ABRA), a method that transfers class-specific detection knowledge from a labeled source domain to a target domain where no training images containing these classes are accessible. ABRA formulates this adaptation as a geometric transport problem in the weight space of a pretrained detector, aligning source and target domain experts to transport class-specific knowledge. Extensive experiments across challenging domain shifts demonstrate that ABRA successfully teleports class-level specialization under multiple adverse conditions. Our code will be made public upon acceptance.
>
---
#### [new 033] A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶规划任务，旨在解决现有模型缺乏可解释性和安全性的问题。提出一种结合神经网络与符号推理的框架，提升决策的透明性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.12421](https://arxiv.org/pdf/2603.12421)**

> **作者:** Hongyan Wei; Wael AbdAlmageed
>
> **备注:** Under review. 16 pages, 2 figures
>
> **摘要:** Existing end-to-end autonomous driving models rely heavily on purely data-driven inductive reasoning. This "black-box" nature leads to a lack of interpretability and absolute safety guarantees in complex, long-tail scenarios. To overcome this bottleneck, we propose a novel neuro-symbolic trajectory planning framework that seamlessly integrates rigorous deductive reasoning into end-to-end neural networks. Specifically, our framework utilizes a Large Language Model (LLM) to dynamically extract scene rules and employs an Answer Set Programming (ASP) solver for deterministic logical arbitration, generating safe and traceable discrete driving decisions. To bridge the gap between discrete symbols and continuous trajectories, we introduce a decision-conditioned decoding mechanism that transforms high-level logical decisions into learnable embedding vectors, simultaneously constraining the planning query and the physical initial velocity of a differentiable Kinematic Bicycle Model (KBM). By combining KBM-generated physical baseline trajectories with neural residual corrections, our approach inherently guarantees kinematic feasibility while ensuring a high degree of transparency. On the nuScenes benchmark, our method comprehensively outperforms the state-of-the-art baseline MomAD, reducing the L2 mean error to 0.57 m, decreasing the collision rate to 0.075%, and optimizing trajectory prediction consistency (TPC) to 0.47 m.
>
---
#### [new 034] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
- **分类: cs.CV**

- **简介: 该论文属于多视角人体与场景重建任务，解决多人群体在多视角视频中联合重建的问题。提出CHROMM框架，实现端到端的相机、场景点云和人体网格估计。**

- **链接: [https://arxiv.org/pdf/2603.12789](https://arxiv.org/pdf/2603.12789)**

> **作者:** Sangmin Kim; Minhyuk Hwang; Geonho Cha; Dongyoon Wee; Jaesik Park
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in 3D foundation models have led to growing interest in reconstructing humans and their surrounding environments. However, most existing approaches focus on monocular inputs, and extending them to multi-view settings requires additional overhead modules or preprocessed data. To this end, we present CHROMM, a unified framework that jointly estimates cameras, scene point clouds, and human meshes from multi-person multi-view videos without relying on external modules or preprocessing. We integrate strong geometric and human priors from Pi3X and Multi-HMR into a single trainable neural network architecture, and introduce a scale adjustment module to solve the scale discrepancy between humans and the scene. We also introduce a multi-view fusion strategy to aggregate per-view estimates into a single representation at test-time. Finally, we propose a geometry-based multi-person association method, which is more robust than appearance-based approaches. Experiments on EMDB, RICH, EgoHumans, and EgoExo4D show that CHROMM achieves competitive performance in global human motion and multi-view pose estimation while running over 8x faster than prior optimization-based multi-view approaches. Project page: this https URL.
>
---
#### [new 035] PVI: Plug-in Visual Injection for Vision-Language-Action Models
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决视觉信息与动作执行不匹配的问题。提出PVI模块，通过注入时间视频特征提升动作专家性能。**

- **链接: [https://arxiv.org/pdf/2603.12772](https://arxiv.org/pdf/2603.12772)**

> **作者:** Zezhou Zhang; Songxin Zhang; Xiao Xiong; Junjie Zhang; Zejian Xie; Jingyi Xi; Zunyao Mao; Zan Mao; Zhixin Mai; Zhuoyang Song; Jiaxing Zhang
>
> **摘要:** VLA architectures that pair a pretrained VLM with a flow-matching action expert have emerged as a strong paradigm for language-conditioned manipulation. Yet the VLM, optimized for semantic abstraction and typically conditioned on static visual observations, tends to attenuate fine-grained geometric cues and often lacks explicit temporal evidence for the action expert. Prior work mitigates this by injecting auxiliary visual features, but existing approaches either focus on static spatial representations or require substantial architectural modifications to accommodate temporal inputs, leaving temporal information underexplored. We propose Plug-in Visual Injection (PVI), a lightweight, encoder-agnostic module that attaches to a pretrained action expert and injects auxiliary visual representations via zero-initialized residual pathways, preserving pretrained behavior with only single-stage fine-tuning. Using PVI, we obtain consistent gains over the base policy and a range of competitive alternative injection strategies, and our controlled study shows that temporal video features (V-JEPA2) outperform strong static image features (DINOv2), with the largest gains on multi-phase tasks requiring state tracking and coordination. Real-robot experiments on long-horizon bimanual cloth folding further demonstrate the practicality of PVI beyond simulation.
>
---
#### [new 036] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，探讨通用视觉模型是否可替代专用模型。通过实验对比分析，发现通用模型在多数情况下表现更优，且具备良好的可解释性。**

- **链接: [https://arxiv.org/pdf/2603.13044](https://arxiv.org/pdf/2603.13044)**

> **作者:** Vanessa Borst; Samuel Kounev
>
> **备注:** Under review, MICCAI 2026
>
> **摘要:** Medical image segmentation (MIS) is a fundamental component of computer-assisted diagnosis and clinical decision support systems. Over the past decade, numerous architectures specifically tailored to medical imaging have emerged to address domain-specific challenges such as low contrast, small anatomical structures, and limited annotated data. In parallel, rapid progress in computer vision has produced highly capable general-purpose vision models (GP-VMs) originally designed for natural images. Despite their strong performance on standard vision benchmarks, their effectiveness for MIS remains insufficiently understood. In this work, we conduct a controlled empirical study to examine whether specialized medical segmentation architectures (SMAs) provide systematic advantages over modern GP-VMs for 2D MIS. We compare eleven SMAs and GP-VMs using a unified training and evaluation protocol. Experiments are performed across three heterogeneous datasets covering different imaging modalities, class structures, and data characteristics. Beyond segmentation accuracy, we analyze qualitative Grad-CAM visualizations to investigate explainability (XAI) behavior. Our results demonstrate that, for the analyzed datasets, GP-VMs out-perform the majority of specialized MIS models. Moreover, XAI analyses indicate that GP-VMs can capture clinically relevant structures without explicit domain-specific architectural design. These findings suggest that GP-VMs can represent a viable alternative to domain-specific methods, highlighting the importance of informed model selection for end-to-end MIS systems. All code and resources are available at GitHub.
>
---
#### [new 037] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending
- **分类: cs.CV**

- **简介: 该论文属于设计制造（DFM）任务，旨在解决CAD设计可制造性评估问题。针对现有数据不足和定义不一致的问题，提出BenDFM数据集及可制造性分类体系。**

- **链接: [https://arxiv.org/pdf/2603.13102](https://arxiv.org/pdf/2603.13102)**

> **作者:** Matteo Ballegeer; Dries F. Benoit
>
> **摘要:** Predicting the manufacturability of CAD designs early, in terms of both feasibility and required effort, is a key goal of Design for Manufacturing (DFM). Despite advances in deep learning for CAD and its widespread use in manufacturing process selection, learning-based approaches for predicting manufacturability within a specific process remain limited. Two key challenges limit progress: inconsistency across prior work in how manufacturability is defined and consequently in the associated learning targets, and a scarcity of suitable datasets. Existing labels vary significantly: they may reflect intrinsic design constraints or depend on specific manufacturing capabilities (such as available tools), and they range from discrete feasibility checks to continuous complexity measures. Furthermore, industrial datasets typically contain only manufacturable parts, offering little signal for infeasible cases, while existing synthetic datasets focus on simple geometries and subtractive processes. To address these gaps, we propose a taxonomy of manufacturability metrics along the axes of configuration dependence and measurement type, allowing clearer scoping of generalizability and learning objectives. Next, we introduce BenDFM, the first synthetic dataset for manufacturability assessment in sheet metal bending. BenDFM contains 20,000 parts, both manufacturable and unmanufacturable, generated with process-aware bending simulations, providing both folded and unfolded geometries and multiple manufacturability labels across the taxonomy, enabling systematic study of previously unexplored learning-based DFM challenges. We benchmark two state-of-the-art 3D learning architectures on BenDFM, showing that graph-based representations that capture relationships between part surfaces achieve better accuracy, and that predicting metrics that depend on specific manufacturing setups remains more challenging.
>
---
#### [new 038] A protocol for evaluating robustness to H&E staining variation in computational pathology models
- **分类: cs.CV**

- **简介: 该论文属于计算病理学任务，旨在解决H&E染色变化对模型性能的影响问题。提出三步评估协议，评估模型在不同染色条件下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12886](https://arxiv.org/pdf/2603.12886)**

> **作者:** Lydia A. Schönpflug; Nikki van den Berg; Sonali Andani; Nanda Horeweg; Jurriaan Barkey Wolf; Tjalling Bosse; Viktor H. Koelzer; Maxime W. Lafarge
>
> **摘要:** Sensitivity to staining variation remains a major barrier to deploying computational pathology (CPath) models as hematoxylin and eosin (H&E) staining varies across laboratories, requiring systematic assessment of how this variability affects model prediction. In this work, we developed a three-step protocol for evaluating robustness to H&E staining variation in CPath models. Step 1: Select reference staining conditions, Step 2: Characterize test set staining properties, Step 3: Apply CPath model(s) under simulated reference staining conditions. Here, we first created a new reference staining library based on the PLISM dataset. As an exemplary use case, we applied the protocol to assess the robustness properties of 306 microsatellite instability (MSI) classification models on the unseen SurGen colorectal cancer dataset (n=738), including 300 attention-based multiple instance learning models trained on the TCGA-COAD/READ datasets across three feature extractors (UNI2-h, H-Optimus-1, Virchow2), alongside six public MSI classification models. Classification performance was measured as AUC, and robustness as the min-max AUC range across four simulated staining conditions (low/high H&E intensity, low/high H&E color similarity). Across models and staining conditions, classification performance ranged from AUC 0.769-0.911 ($\Delta$ = 0.142). Robustness ranged from 0.007-0.079 ($\Delta$ = 0.072), and showed a weak inverse correlation with classification performance (Pearson r=-0.22, 95% CI [-0.34, -0.11]). Thus, we show that the proposed evaluation protocol enables robustness-informed CPath model selection and provides insight into performance shifts across H&E staining conditions, supporting the identification of operational ranges for reliable model deployment. Code is available at this https URL .
>
---
#### [new 039] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决真实手机RAW域退化建模问题。通过校准和渲染生成真实退化数据，提升SR模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12493](https://arxiv.org/pdf/2603.12493)**

> **作者:** Ali Mosleh; Faraz Ali; Fengjia Zhang; Stavros Tsogkas; Junyong Lee; Alex Levinshtein; Michael S. Brown
>
> **备注:** This paper has been accepted to The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Digital zoom on smartphones relies on learning-based super-resolution (SR) models that operate on RAW sensor images, but obtaining sensor-specific training data is challenging due to the lack of ground-truth images. Synthetic data generation via ``unprocessing'' pipelines offers a potential solution by simulating the degradations that transform high-resolution (HR) images into their low-resolution (LR) counterparts. However, these pipelines can introduce domain gaps due to incomplete or unrealistic degradation modeling. In this paper, we demonstrate that principled and carefully designed degradation modeling can enhance SR performance in real-world conditions. Instead of relying on generic priors for camera blur and noise, we model device-specific degradations through calibration and unprocess publicly available rendered images into the RAW domain of different smartphones. Using these image pairs, we train a single-image RAW-to-RGB SR model and evaluate it on real data from a held-out device. Our experiments show that accurate degradation modeling leads to noticeable improvements, with our SR model outperforming baselines trained on large pools of arbitrarily chosen degradations.
>
---
#### [new 040] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于室外场景分类任务，解决因光照和遮挡导致的识别难题。通过构建数据集并使用CNN处理LiDAR深度与反射图像，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2603.12663](https://arxiv.org/pdf/2603.12663)**

> **作者:** Kazuto Nakashima; Hojung Jung; Yuki Oto; Yumi Iwashita; Ryo Kurazume; Oscar Martinez Mozos
>
> **备注:** Published in Advanced Robotics on 31 Jul 2018
>
> **摘要:** Semantic place categorization, which is one of the essential tasks for autonomous robots and vehicles, allows them to have capabilities of self-decision and navigation in unfamiliar environments. In particular, outdoor places are more difficult targets than indoor ones due to perceptual variations, such as dynamic illuminance over twenty-four hours and occlusions by cars and pedestrians. This paper presents a novel method of categorizing outdoor places using convolutional neural networks (CNNs), which take omnidirectional depth/reflectance images obtained by 3D LiDARs as the inputs. First, we construct a large-scale outdoor place dataset named Multi-modal Panoramic 3D Outdoor (MPO) comprising two types of point clouds captured by two different LiDARs. They are labeled with six outdoor place categories: coast, forest, indoor/outdoor parking, residential area, and urban area. Second, we provide CNNs for LiDAR-based outdoor place categorization and evaluate our approach with the MPO dataset. Our results on the MPO dataset outperform traditional approaches and show the effectiveness in which we use both depth and reflectance modalities. To analyze our trained deep networks we visualize the learned features.
>
---
#### [new 041] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking
- **分类: cs.CV**

- **简介: 该论文提出TRACE，用于文档水印的结构感知字符编码方法，解决噪声干扰和跨媒体传输问题，通过扩散模型实现鲁棒且通用的水印嵌入。**

- **链接: [https://arxiv.org/pdf/2603.12873](https://arxiv.org/pdf/2603.12873)**

> **作者:** Jiale Meng; Jie Zhang; Runyi Hu; Zhe-Ming Lu; Tianwei Zhang; Yiming Li
>
> **摘要:** We propose TRACE, a structure-aware framework leveraging diffusion models for localized character encoding to embed data. Unlike existing methods that rely on edge features or pre-defined codebooks, TRACE exploits character structures that provide inherent resistance to noise interference due to their stability and unified representation across diverse characters. Our framework comprises three key components: (1) adaptive diffusion initialization that automatically identifies handle points, target points, and editing regions through specialized algorithms including movement probability estimator (MPE), target point estimation (TPE) and mask drawing model (MDM), (2) guided diffusion encoding for precise movement of selected point, and (3) masked region replacement with a specialized loss function to minimize feature alterations after the diffusion process. Comprehensive experiments demonstrate \name{}'s superior performance over state-of-the-art methods, achieving more than 5 dB improvement in PSNR and 5\% higher extraction accuracy following cross-media transmission. \name{} achieves broad generalizability across multiple languages and fonts, making it particularly suitable for practical document security applications.
>
---
#### [new 042] The COTe score: A decomposable framework for evaluating Document Layout Analysis models
- **分类: cs.CV**

- **简介: 该论文属于文档布局分析任务，解决传统评估指标不适用于印刷媒体的问题，提出COTe分数和SSU标签方法，提升模型评估的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.12718](https://arxiv.org/pdf/2603.12718)**

> **作者:** Jonathan Bourne; Mwiza Simbeye; Ishtar Govia
>
> **备注:** 6906 words, 4 Figures, 10 Tables,
>
> **摘要:** Document Layout analysis (DLA), is the process by which a page is parsed into meaningful elements, often using machine learning models. Typically, the quality of a model is judged using general object detection metrics such as IoU, F1 or mAP. However, these metrics are designed for images that are 2D projections of 3D space, not for the natively 2D imagery of printed media. This discrepancy can result in misleading or uninformative interpretation of model performance by the metrics. To encourage more robust, comparable, and nuanced DLA, we introduce: The Structural Semantic Unit (SSU) a relational labelling approach that shifts the focus from the physical to the semantic structure of the content; and the Coverage, Overlap, Trespass, and Excess (COTe) score, a decomposable metric for measuring page parsing quality. We demonstrate the value of these methods through case studies and by evaluating 5 common DLA models on 3 DLA datasets. We show that the COTe score is more informative than traditional metrics and reveals distinct failure modes across models, such as breaching semantic boundaries or repeatedly parsing the same region. In addition, the COTe score reduces the interpretation-performance gap by up to 76% relative to the F1. Notably, we find that the COTe's granularity robustness largely holds even without explicit SSU labelling, lowering the barriers to entry for using the system. Finally, we release an SSU labelled dataset and a Python library for applying COTe in DLA projects.
>
---
#### [new 043] Deployment-Oriented Session-wise Meta-Calibration for Landmark-Based Webcam Gaze Tracking
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于眼球追踪任务，旨在解决网络摄像头 gaze tracking 的校准负担和会话漂移问题。提出 EMC-Gaze 方法，通过轻量级地标方法实现高效、低误差的实时追踪。**

- **链接: [https://arxiv.org/pdf/2603.12388](https://arxiv.org/pdf/2603.12388)**

> **作者:** Chenkai Zhang
>
> **备注:** 24 pages, 7 figures. Deployment-oriented landmark-only webcam gaze tracking with browser-capable runtime
>
> **摘要:** Practical webcam gaze tracking is constrained not only by error, but also by calibration burden, robustness to head motion and session drift, runtime footprint, and browser use. We therefore target a deployment-oriented operating point rather than the image large-backbone regime. We cast landmark-based point-of-regard estimation as session-wise adaptation: a shared geometric encoder produces embeddings that can be aligned to a new session from a small calibration set. We present Equivariant Meta-Calibrated Gaze (EMC-Gaze), a lightweight landmark-only method combining an E(3)-equivariant landmark-graph encoder, local eye geometry, binocular emphasis, auxiliary 3D gaze-direction supervision, and a closed-form ridge calibrator differentiated through episodic meta-training. To reduce pose leakage, we use a two-view canonicalization consistency loss. The deployed predictor uses only facial landmarks and fits a per-session ridge head from brief calibration. In a fixation-style interactive evaluation over 33 sessions at 100 cm, EMC-Gaze achieves 5.79 +/- 1.81 deg RMSE after 9-point calibration versus 6.68 +/- 2.34 deg for Elastic Net; the gain is larger on still-head queries (2.92 +/- 0.75 deg vs. 4.45 +/- 0.30 deg). Across three subject holdouts of 10 subjects each, EMC-Gaze retains an advantage (5.66 +/- 0.19 deg vs. 6.49 +/- 0.33 deg). On MPIIFaceGaze with short per-session calibration, the eye-focused model reaches 8.82 +/- 1.21 deg at 16-shot calibration, ties Elastic Net at 1-shot, and outperforms it from 3-shot onward. The exported eye-focused encoder has 944,423 parameters, is 4.76 MB in ONNX, and supports calibrated browser prediction in 12.58/12.58/12.90 ms per sample (mean/median/p90) in Chromium 145 with ONNX Runtime Web. These results position EMC-Gaze as a calibration-friendly operating point rather than a universal state-of-the-art claim against heavier appearance-based systems.
>
---
#### [new 044] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于水下图像增强任务，旨在解决增强图像与自然图像分布差异导致的语义信息丢失问题。通过引入视觉-语言模型，生成语义引导图，提升关键对象的恢复效果。**

- **链接: [https://arxiv.org/pdf/2603.12773](https://arxiv.org/pdf/2603.12773)**

> **作者:** Guodong Fan; Shengning Zhou; Genji Yuan; Huiyu Li; Jingchun Zhou; Jinjiang Li
>
> **备注:** Accepted as an Oral presentation at AAAI 2026
>
> **摘要:** In recent years, learning-based underwater image enhancement (UIE) techniques have rapidly evolved. However, distribution shifts between high-quality enhanced outputs and natural images can hinder semantic cue extraction for downstream vision tasks, thereby limiting the adaptability of existing enhancement models. To address this challenge, this work proposes a new learning mechanism that leverages Vision-Language Models (VLMs) to empower UIE models with semantic-sensitive capabilities. To be concrete, our strategy first generates textual descriptions of key objects from a degraded image via VLMs. Subsequently, a text-image alignment model remaps these relevant descriptions back onto the image to produce a spatial semantic guidance map. This map then steers the UIE network through a dual-guidance mechanism, which combines cross-attention and an explicit alignment loss. This forces the network to focus its restorative power on semantic-sensitive regions during image reconstruction, rather than pursuing a globally uniform improvement, thereby ensuring the faithful restoration of key object features. Experiments confirm that when our strategy is applied to different UIE baselines, significantly boosts their performance on perceptual quality metrics as well as enhances their performance on detection and segmentation tasks, validating its effectiveness and adaptability.
>
---
#### [new 045] Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感视觉接地任务，旨在解决多实体推理不足的问题。提出ME-RSRG数据集和EAR框架，提升多实体推理与建模能力。**

- **链接: [https://arxiv.org/pdf/2603.12788](https://arxiv.org/pdf/2603.12788)**

> **作者:** Shuchang Lyu; Haiquan Wen; Guangliang Cheng; Meng Li; Zheng Zhou; You Zhou; Dingding Yao; Zhenwei Shi
>
> **备注:** 22 pages, 9 figures, 5 tables
>
> **摘要:** Recent advances in reasoning language models and reinforcement learning with verifiable rewards have significantly enhanced multi-step reasoning capabilities. This progress motivates the extension of reasoning paradigms to remote sensing visual grounding task. However, existing remote sensing grounding methods remain largely confined to perception-level matching and single-entity formulations, limiting the role of explicit reasoning and inter-entity modeling. To address this challenge, we introduce a new benchmark dataset for Multi-Entity Reasoning Grounding in Remote Sensing (ME-RSRG). Based on ME-RSRG, we reformulate remote sensing grounding as a multi-entity reasoning task and propose an Entity-Aware Reasoning (EAR) framework built upon visual-linguistic foundation models. EAR generates structured reasoning traces and subject-object grounding outputs. It adopts supervised fine-tuning for cold-start initialization and is further optimized via entity-aware reward-driven Group Relative Policy Optimization (GRPO). Extensive experiments on ME-RSRG demonstrate the challenges of multi-entity reasoning and verify the effectiveness of our proposed EAR framework. Our dataset, code, and models will be available at this https URL.
>
---
#### [new 046] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos
- **分类: cs.CV**

- **简介: 该论文提出VCBench，用于评估视频理解中时空状态维护能力。解决现有基准对状态维护观测不足的问题，通过计数任务分解为8个子类，提供标注数据和评估指标。**

- **链接: [https://arxiv.org/pdf/2603.12703](https://arxiv.org/pdf/2603.12703)**

> **作者:** Pengyiang Liu; Zhongyue Shi; Hongye Hao; Qi Fu; Xueting Bi; Siwei Zhang; Xiaoyang Hu; Zitian Wang; Linjiang Huang; Si Liu
>
> **摘要:** Video understanding requires models to continuously track and update world state during playback. While existing benchmarks have advanced video understanding evaluation across multiple dimensions, the observation of how models maintain world state remains insufficient. We propose VCBench, a streaming counting benchmark that repositions counting as a minimal probe for diagnosing world state maintenance capability. We decompose this capability into object counting (tracking currently visible objects vs.\ tracking cumulative unique identities) and event counting (detecting instantaneous actions vs.\ tracking complete activity cycles), forming 8 fine-grained subcategories. VCBench contains 406 videos with frame-by-frame annotations of 10,071 event occurrence moments and object state change moments, generating 1,000 streaming QA pairs with 4,576 query points along timelines. By observing state maintenance trajectories through streaming multi-point queries, we design three complementary metrics to diagnose numerical precision, trajectory consistency, and temporal awareness. Evaluation on mainstream video-language models shows that current models still exhibit significant deficiencies in spatial-temporal state maintenance, particularly struggling with tasks like periodic event counting. VCBench provides a diagnostic framework for measuring and improving state maintenance in video understanding systems.
>
---
#### [new 047] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在提升脑肿瘤分类的鲁棒性。通过结合NNMF、轻量CNN和扩散去噪技术，增强模型对对抗攻击的抵抗能力。**

- **链接: [https://arxiv.org/pdf/2603.13182](https://arxiv.org/pdf/2603.13182)**

> **作者:** Hiba Adil Al-kharsan; Róbert Rajkó
>
> **备注:** 30 pages, 29 figures
>
> **摘要:** Brain tumor classification from magnetic resonance imaging, which is also known as MRI, plays a sensitive role in computer-assisted diagnosis systems. In recent years, deep learning models have achieved high classification accuracy. However, their sensitivity to adversarial perturbations has become an important reliability concern in medical applications. This study suggests a robust brain tumor classification framework that combines Non-Negative Matrix Factorization (NNMF or NMF), lightweight convolutional neural networks (CNNs), and diffusion-based feature purification. Initially, MRI images are preprocessed and converted into a non-negative data matrix, from which compact and interpretable NNMF feature representations are extracted. Statistical metrics, including AUC, Cohen's d, and p-values, are used to rank and choose the most discriminative components. Then, a lightweight CNN classifier is trained directly on the selected feature groups. To improve adversarial robustness, a diffusion-based feature-space purification module is introduced. A forward noise method followed by a learned denoiser network is used before classification. System performance is estimated using both clean accuracy and robust accuracy under powerful adversarial attacks created by AutoAttack. The experimental results show that the proposed framework achieves competitive classification performance while significantly enhancing robustness against adversarial this http URL findings presuppose that combining interpretable NNMF-based representations with a lightweight deep approach and diffusion-based defense technique supplies an effective and reliable solution for medical image classification under adversarial conditions.
>
---
#### [new 048] FDeID-Toolbox: Face De-Identification Toolbox
- **分类: cs.CV**

- **简介: 该论文属于人脸去标识任务，旨在保护隐私同时保留有用信息。针对现有研究碎片化问题，提出FDeID-Toolbox，提供标准化工具链以实现可复现的比较与评估。**

- **链接: [https://arxiv.org/pdf/2603.13121](https://arxiv.org/pdf/2603.13121)**

> **作者:** Hui Wei; Hao Yu; Guoying Zhao
>
> **备注:** Technical Report. Codebase: this https URL
>
> **摘要:** Face de-identification (FDeID) aims to remove personally identifiable information from facial images while preserving task-relevant utility attributes such as age, gender, and expression. It is critical for privacy-preserving computer vision, yet the field suffers from fragmented implementations, inconsistent evaluation protocols, and incomparable results across studies. These challenges stem from the inherent complexity of the task: FDeID spans multiple downstream applications (e.g., age estimation, gender recognition, expression analysis) and requires evaluation across three dimensions (e.g., privacy protection, utility preservation, and visual quality), making existing codebases difficult to use and extend. To address these issues, we present FDeID-Toolbox, a comprehensive toolbox designed for reproducible FDeID research. Our toolbox features a modular architecture comprising four core components: (1) standardized data loaders for mainstream benchmark datasets, (2) unified method implementations spanning classical approaches to SOTA generative models, (3) flexible inference pipelines, and (4) systematic evaluation protocols covering privacy, utility, and quality metrics. Through experiments, we demonstrate that FDeID-Toolbox enables fair and reproducible comparison of diverse FDeID methods under consistent conditions.
>
---
#### [new 049] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自驾车场景重建任务，解决LiDAR与RGB信息利用不充分的问题，提出LR-SGS方法，融合LiDAR反射率与RGB，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2603.12647](https://arxiv.org/pdf/2603.12647)**

> **作者:** Ziyu Chen; Fan Zhu; Hui Zhu; Deyi Kong; Xinkai Kuang; Yujia Zhang; Chunmao Jiang
>
> **备注:** 8 pages, 7 figures, conference
>
> **摘要:** Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
>
---
#### [new 050] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MoKus框架，解决知识感知概念定制任务中的稀有标记性能不稳定问题，通过跨模态知识迁移实现高质量生成。**

- **链接: [https://arxiv.org/pdf/2603.12743](https://arxiv.org/pdf/2603.12743)**

> **作者:** Chenyang Zhu; Hongxiang Li; Xiu Li; Long Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Concept customization typically binds rare tokens to a target concept. Unfortunately, these approaches often suffer from unstable performance as the pretraining data seldom contains these rare tokens. Meanwhile, these rare tokens fail to convey the inherent knowledge of the target concept. Consequently, we introduce Knowledge-aware Concept Customization, a novel task aiming at binding diverse textual knowledge to target visual concepts. This task requires the model to identify the knowledge within the text prompt to perform high-fidelity customized generation. Meanwhile, the model should efficiently bind all the textual knowledge to the target concept. Therefore, we propose MoKus, a novel framework for knowledge-aware concept customization. Our framework relies on a key observation: cross-modal knowledge transfer, where modifying knowledge within the text modality naturally transfers to the visual modality during generation. Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept. (2) In textual knowledge updating, we update the answer for the knowledge queries to the anchor representation, enabling high-fidelity customized generation. To further comprehensively evaluate our proposed MoKus on the new task, we introduce the first benchmark for knowledge-aware concept customization: KnowCusBench. Extensive evaluations have demonstrated that MoKus outperforms state-of-the-art methods. Moreover, the cross-model knowledge transfer allows MoKus to be easily extended to other knowledge-aware applications like virtual concept creation and concept erasure. We also demonstrate the capability of our method to achieve improvements on world knowledge benchmarks.
>
---
#### [new 051] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在用少量样本实现通用图像修复。通过视频生成模型迁移学习，构建V-Bridge框架，提升修复效果。**

- **链接: [https://arxiv.org/pdf/2603.13089](https://arxiv.org/pdf/2603.13089)**

> **作者:** Shenghe Zheng; Junpeng Jiang; Wenbo Li
>
> **备注:** Transfer the prior knowledge of video generative models to image restoration tasks
>
> **摘要:** Large-scale video generative models are trained on vast and diverse visual data, enabling them to internalize rich structural, semantic, and dynamic priors of the visual world. While these models have demonstrated impressive generative capability, their potential as general-purpose visual learners remains largely untapped. In this work, we introduce V-Bridge, a framework that bridges this latent capacity to versatile few-shot image restoration tasks. We reinterpret image restoration not as a static regression problem, but as a progressive generative process, and leverage video models to simulate the gradual refinement from degraded inputs to high-fidelity outputs. Surprisingly, with only 1,000 multi-task training samples (less than 2% of existing restoration methods), pretrained video models can be induced to perform competitive image restoration, achieving multiple tasks with a single model, rivaling specialized architectures designed explicitly for this purpose. Our findings reveal that video generative models implicitly learn powerful and transferable restoration priors that can be activated with only extremely limited data, challenging the traditional boundary between generative modeling and low-level vision, and opening a new design paradigm for foundation models in visual tasks.
>
---
#### [new 052] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于磨损分类任务，旨在解决 abrasive flap wheels 磨损状态识别问题。通过构建数据集并使用深度学习方法进行分层分类，提高磨削过程的自动化水平。**

- **链接: [https://arxiv.org/pdf/2603.12852](https://arxiv.org/pdf/2603.12852)**

> **作者:** Falko Kähler; Maxim Wille; Ole Schmedemann; Thorsten Schüppstuhl
>
> **备注:** 14 pages, 11 figures, 8 tables
>
> **摘要:** Abrasive flap wheels are common for finishing complex free-form surfaces due to their flexibility. However, this flexibility results in complex wear patterns such as concave/convex flap profiles or flap tears, which influence the grinding result. This paper proposes a novel, vision-based hierarchical classification framework to automate the wear condition monitoring of flap wheels. Unlike monolithic classification approaches, we decompose the problem into three logical levels: (1) state detection (new vs. worn), (2) wear type identification (rectangular, concave, convex) and flap tear detection, and (3) severity assessment (partial vs. complete deformation). A custom-built dataset of real flap wheel images was generated and a transfer learning approach with EfficientNetV2 architecture was used. The results demonstrate high robustness with classification accuracies ranging from 93.8% (flap tears) to 99.3% (concave severity). Furthermore, Gradient-weighted Class Activation Mapping (Grad-CAM) is utilized to validate that the models learn physically relevant features and examine false classifications. The proposed hierarchical method provides a basis for adaptive process control and wear consideration in automated flap wheel grinding.
>
---
#### [new 053] Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating
- **分类: cs.CV**

- **简介: 该论文属于隐私保护任务，旨在解决LVLMs在处理敏感查询时的隐私泄露问题。通过神经元级梯度门控方法，提升模型对隐私相关问题的拒绝能力，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12598](https://arxiv.org/pdf/2603.12598)**

> **作者:** Xiangkui Cao; Jie Zhang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown remarkable potential across a wide array of vision-language tasks, leading to their adoption in critical domains such as finance and healthcare. However, their growing deployment also introduces significant security and privacy risks. Malicious actors could potentially exploit these models to extract sensitive information, highlighting a critical vulnerability. Recent studies show that LVLMs often fail to consistently refuse instructions designed to compromise user privacy. While existing work on privacy protection has made meaningful progress in preventing the leakage of sensitive data, they are constrained by limitations in both generalization and non-destructiveness. They often struggle to robustly handle unseen privacy-related queries and may inadvertently degrade a model's performance on standard tasks. To address these challenges, we introduce Neural Gate, a novel method for mitigating privacy risks through neuron-level model editing. Our method improves a model's privacy safeguards by increasing its rate of refusal for privacy-related questions, crucially extending this protective behavior to novel sensitive queries not encountered during the editing process. Neural Gate operates by learning a feature vector to identify neurons associated with privacy-related concepts within the model's representation of a subject. This localization then precisely guides the update of model parameters. Through comprehensive experiments on MiniGPT and LLaVA, we demonstrate that our method significantly boosts the model's privacy protection while preserving its original utility.
>
---
#### [new 054] Geometry-Guided Camera Motion Understanding in VideoLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决VideoLLMs对相机运动识别不足的问题。通过构建数据集和基准，提出轻量级框架提升模型对相机运动的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.13119](https://arxiv.org/pdf/2603.13119)**

> **作者:** Haoan Feng; Sri Harsha Musunuri; Guan-Ming Su
>
> **备注:** 10 pages, 7 figures, supplementary included
>
> **摘要:** Camera motion is a fundamental geometric signal that shapes visual perception and cinematic style, yet current video-capable vision-language models (VideoLLMs) rarely represent it explicitly and often fail on fine-grained motion primitives. We address this gap with a framework of $\textbf{benchmarking}$, $\textbf{diagnosis}$, and $\textbf{injection}$. We curate $\textbf{CameraMotionDataset}$, a large-scale synthetic dataset with explicit camera control, formulate camera motion as constraint-aware multi-label recognition, and construct a VQA benchmark--$\textbf{CameraMotionVQA}$. Across diverse off-the-shelf VideoLLMs, we observe substantial errors in recognizing camera motion primitives. Probing experiments on a Qwen2.5-VL vision encoder suggest that camera motion cues are weakly represented, especially in deeper ViT blocks, helping explain the observed failure modes. To bridge this gap without costly training or fine-tuning, we propose a lightweight, model-agnostic pipeline that extracts geometric camera cues from 3D foundation models (3DFMs), predicts constrained motion primitives with a temporal classifier, and injects them into downstream VideoLLM inference via structured prompting. Experiments demonstrate improved motion recognition and more camera-aware model responses, highlighting geometry-driven cue extraction and structured prompting as practical steps toward a camera-aware VideoLLM and VLA system. The dataset and benchmark is publicly available at this https URL.
>
---
#### [new 055] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频多模态语言模型任务，解决视频中对象空间精度与时间一致性问题。提出SPARROW模型，通过时空特征与双提示设计提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2603.12382](https://arxiv.org/pdf/2603.12382)**

> **作者:** Mohamad Alansari; Naufal Suryanto; Divya Velayudhan; Sajid Javed; Naoufel Werghi; Muzammal Naseer
>
> **备注:** Accepted at CVPR 2026; Project page: this https URL Repository: this https URL
>
> **摘要:** Multimodal large language models (MLLMs) have advanced from image-level reasoning to pixel-level grounding, but extending these capabilities to videos remains challenging as models must achieve spatial precision and temporally consistent reference tracking. Existing video MLLMs often rely on a static segmentation token ([SEG]) for frame-wise grounding, which provides semantics but lacks temporal context, causing spatial drift, identity switches, and unstable initialization when objects move or reappear. We introduce SPARROW, a pixel-grounded video MLLM that unifies spatial accuracy and temporal stability through two key components: (i) Target-Specific Tracked Features (TSF), which inject temporally aligned referent cues during training, and (ii) a dual-prompt design that decodes box ([BOX]) and segmentation ([SEG]) tokens to fuse geometric priors with semantic grounding. SPARROW is supported by a curated referential video dataset of 30,646 videos and 45,231 Q&A pairs and operates end-to-end without external detectors via a class-agnostic SAM2-based proposer. Integrated into three recent open-source video MLLMs (UniPixel, GLUS, and VideoGLaMM), SPARROW delivers consistent gains across six benchmarks, improving up to +8.9 J&F on RVOS, +5 mIoU on visual grounding, and +5.4 CLAIR on GCG. These results demonstrate that SPARROW substantially improves referential stability, spatial precision, and temporal coherence in pixel-grounded video understanding. Project page: this https URL
>
---
#### [new 056] Test-Time Attention Purification for Backdoored Large Vision Language Models
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于模型安全任务，解决LVLM在微调中遭受后门攻击的问题。提出CleanSight，通过检测并净化异常视觉注意力实现测试时防御。**

- **链接: [https://arxiv.org/pdf/2603.12989](https://arxiv.org/pdf/2603.12989)**

> **作者:** Zhifang Zhang; Bojun Yang; Shuo He; Weitong Chen; Wei Emma Zhang; Olaf Maennel; Lei Feng; Miao Xu
>
> **摘要:** Despite the strong multimodal performance, large vision-language models (LVLMs) are vulnerable during fine-tuning to backdoor attacks, where adversaries insert trigger-embedded samples into the training data to implant behaviors that can be maliciously activated at test time. Existing defenses typically rely on retraining backdoored parameters (e.g., adapters or LoRA modules) with clean data, which is computationally expensive and often degrades model performance. In this work, we provide a new mechanistic understanding of backdoor behaviors in LVLMs: the trigger does not influence prediction through low-level visual patterns, but through abnormal cross-modal attention redistribution, where trigger-bearing visual tokens steal attention away from the textual context - a phenomenon we term attention stealing. Motivated by this, we propose CleanSight, a training-free, plug-and-play defense that operates purely at test time. CleanSight (i) detects poisoned inputs based on the relative visual-text attention ratio in selected cross-modal fusion layers, and (ii) purifies the input by selectively pruning the suspicious high-attention visual tokens to neutralize the backdoor activation. Extensive experiments show that CleanSight significantly outperforms existing pixel-based purification defenses across diverse datasets and backdoor attack types, while preserving the model's utility on both clean and poisoned samples.
>
---
#### [new 057] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文提出WSGG任务，解决视频中对象交互的时空建模问题。通过引入4D数据集和三种方法，实现对可见与不可见对象的持续推理。**

- **链接: [https://arxiv.org/pdf/2603.13185](https://arxiv.org/pdf/2603.13185)**

> **作者:** Rohith Peddi; Saurabh; Shravan Shanmugam; Likhitha Pallapothula; Yu Xiang; Parag Singla; Vibhav Gogate
>
> **备注:** this https URL
>
> **摘要:** Spatio-temporal scene graphs provide a principled representation for modeling evolving object interactions, yet existing methods remain fundamentally frame-centric: they reason only about currently visible objects, discard entities upon occlusion, and operate in 2D. To address this, we first introduce ActionGenome4D, a dataset that upgrades Action Genome videos into 4D scenes via feed-forward 3D reconstruction, world-frame oriented bounding boxes for every object involved in actions, and dense relationship annotations including for objects that are temporarily unobserved due to occlusion or camera motion. Building on this data, we formalize World Scene Graph Generation (WSGG), the task of constructing a world scene graph at each timestamp that encompasses all interacting objects in the scene, both observed and unobserved. We then propose three complementary methods, each exploring a different inductive bias for reasoning about unobserved objects: PWG (Persistent World Graph), which implements object permanence via a zero-order feature buffer; MWAE (Masked World Auto-Encoder), which reframes unobserved-object reasoning as masked completion with cross-view associative retrieval; and 4DST (4D Scene Transformer), which replaces the static buffer with differentiable per-object temporal attention enriched by 3D motion and camera-pose features. We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction. WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.
>
---
#### [new 058] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的去偏任务，旨在解决模型继承社会偏见的问题。提出一种无需训练、无需标注数据的去偏方法，在保持模型性能的同时提升公平性。**

- **链接: [https://arxiv.org/pdf/2603.12998](https://arxiv.org/pdf/2603.12998)**

> **作者:** Tangzheng Lian; Guanyu Hu; Yijing Ren; Dimitrios Kollias; Oya Celiktutan
>
> **摘要:** While Vision-Language Models (VLMs) have achieved remarkable performance across diverse downstream tasks, recent studies have shown that they can inherit social biases from the training data and further propagate them into downstream applications. To address this issue, various debiasing approaches have been proposed, yet most of them aim to improve fairness without having a theoretical guarantee that the utility of the model is preserved. In this paper, we introduce a debiasing method that yields a \textbf{closed-form} solution in the cross-modal space, achieving Pareto-optimal fairness with \textbf{bounded utility losses}. Our method is \textbf{training-free}, requires \textbf{no annotated data}, and can jointly debias both visual and textual modalities across downstream tasks. Extensive experiments show that our method outperforms existing methods in debiasing VLMs across diverse fairness metrics and datasets for both group and \textbf{intersectional} fairness in downstream tasks such as zero-shot image classification, text-to-image retrieval, and text-to-image generation while preserving task performance.
>
---
#### [new 059] AccelAes: Accelerating Diffusion Transformers for Training-Free Aesthetic-Enhanced Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散Transformer推理速度慢的问题。通过引入美学感知的稀疏计算策略，提升生成效率与美学质量。**

- **链接: [https://arxiv.org/pdf/2603.12575](https://arxiv.org/pdf/2603.12575)**

> **作者:** Xuanhua Yin; Chuanzhi Xu; Haoxian Zhou; Boyu Wei; Weidong Cai
>
> **备注:** 32 pages, 13 tables, 12 figures
>
> **摘要:** Diffusion Transformers (DiTs) are a dominant backbone for high-fidelity text-to-image generation due to strong scalability and alignment at high resolutions. However, quadratic self-attention over dense spatial tokens leads to high inference latency and limits deployment. We observe that denoising is spatially non-uniform with respect to aesthetic descriptors in the prompt. Regions associated with aesthetic tokens receive concentrated cross-attention and show larger temporal variation, while low-affinity regions evolve smoothly with redundant computation. Based on this insight, we propose AccelAes, a training-free framework that accelerates DiTs through aesthetics-aware spatio-temporal reduction while improving perceptual aesthetics. AccelAes builds AesMask, a one-shot aesthetic focus mask derived from prompt semantics and cross-attention signals. When localized computation is feasible, SkipSparse reallocates computation and guidance to masked regions. We further reduce temporal redundancy using a lightweight step-level prediction cache that periodically replaces full Transformer evaluations. Experiments on representative DiT families show consistent acceleration and improved aesthetics-oriented quality. On Lumina-Next, AccelAes achieves a 2.11$\times$ speedup and improves ImageReward by +11.9% over the dense baseline. Code is available at this https URL.
>
---
#### [new 060] Naïve PAINE: Lightweight Text-to-Image Generation Improvement with Prompt Evaluation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成结果质量不稳定的问题。通过预测图像质量并选择优质噪声，提升扩散模型的生成效果。**

- **链接: [https://arxiv.org/pdf/2603.12506](https://arxiv.org/pdf/2603.12506)**

> **作者:** Joong Ho Kim; Nicholas Thai; Souhardya Saha Dip; Dong Lao; Keith G. Mills
>
> **备注:** Code available at this https URL
>
> **摘要:** Text-to-Image (T2I) generation is primarily driven by Diffusion Models (DM) which rely on random Gaussian noise. Thus, like playing the slots at a casino, a DM will produce different results given the same user-defined inputs. This imposes a gambler's burden: To perform multiple generation cycles to obtain a satisfactory result. However, even though DMs use stochastic sampling to seed generation, the distribution of generated content quality highly depends on the prompt and the generative ability of a DM with respect to it. To account for this, we propose Naïve PAINE for improving the generative quality of Diffusion Models by leveraging T2I preference benchmarks. We directly predict the numerical quality of an image from the initial noise and given prompt. Naïve PAINE then selects a handful of quality noises and forwards them to the DM for generation. Further, Naïve PAINE provides feedback on the DM generative quality given the prompt and is lightweight enough to seamlessly fit into existing DM pipelines. Experimental results demonstrate that Naïve PAINE outperforms existing approaches on several prompt corpus benchmarks.
>
---
#### [new 061] What Makes VLMs Robust? Towards Reconciling Robustness and Accuracy in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）领域，旨在解决模型在对抗攻击下的鲁棒性与准确性的权衡问题。通过分析发现鲁棒性主要存在于浅层，并提出R-Adapt框架实现平衡。**

- **链接: [https://arxiv.org/pdf/2603.12799](https://arxiv.org/pdf/2603.12799)**

> **作者:** Sen Nie; Jie Zhang; Zhongqi Wang; Zhaoyang Wei; Shiguang Shan; Xilin Chen
>
> **备注:** 28 pages
>
> **摘要:** Achieving adversarial robustness in Vision-Language Models (VLMs) inevitably compromises accuracy on clean data, presenting a long-standing and challenging trade-off. In this work, we revisit this trade-off by investigating a fundamental question: What makes VLMs robust? Through a detailed analysis of adversarially fine-tuned models, we examine how robustness mechanisms function internally and how they interact with clean accuracy. Our analysis reveals that adversarial robustness is not uniformly distributed across network depth. Instead, unexpectedly, it is primarily localized within the shallow layers, driven by a low-frequency spectral bias and input-insensitive attention patterns. Meanwhile, updates to the deep layers tend to undermine both clean accuracy and robust generalization. Motivated by these insights, we propose Adversarial Robustness Adaptation (R-Adapt), a simple yet effective framework that freezes all pre-trained weights and introduces minimal, insight-driven adaptations only in the initial layers. This design achieves an exceptional balance between adversarial robustness and clean accuracy. R-Adapt further supports training-free, model-guided, and data-driven paradigms, offering flexible pathways to seamlessly equip standard models with robustness. Extensive evaluations on 18 datasets and diverse tasks demonstrate our state-of-the-art performance under various attacks. Notably, R-Adapt generalizes efficiently to large vision-language models (e.g., LLaVA and Qwen-VL) to enhance their robustness. Our project page is available at this https URL.
>
---
#### [new 062] Revisiting Model Stitching In the Foundation Model Era
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究模型拼接任务，解决不同视觉基础模型是否可拼接的问题。通过引入特征匹配损失，提升拼接效果，并提出VFM Stitch Tree实现高效集成。**

- **链接: [https://arxiv.org/pdf/2603.12433](https://arxiv.org/pdf/2603.12433)**

> **作者:** Zheda Mai; Ke Zhang; Fu-En Wang; Zixiao Ken Wang; Albert Y. C. Chen; Lu Xia; Min Sun; Wei-Lun Chao; Cheng-Hao Kuo
>
> **备注:** Accepted by CVPR 2023
>
> **摘要:** Model stitching, connecting early layers of one model (source) to later layers of another (target) via a light stitch layer, has served as a probe of representational compatibility. Prior work finds that models trained on the same dataset remain stitchable (negligible accuracy drop) despite different initializations or objectives. We revisit stitching for Vision Foundation Models (VFMs) that vary in objectives, data, and modality mix (e.g., CLIP, DINOv2, SigLIP 2) and ask: Are heterogeneous VFMs stitchable? We introduce a systematic protocol spanning the stitch points, stitch layer families, training losses, and downstream tasks. Three findings emerge. (1) Stitch layer training matters: conventional approaches that match the intermediate features at the stitch point or optimize the task loss end-to-end struggle to retain accuracy, especially at shallow stitch points. (2) With a simple feature-matching loss at the target model's penultimate layer, heterogeneous VFMs become reliably stitchable across vision tasks. (3) For deep stitch points, the stitched model can surpass either constituent model at only a small inference overhead (for the stitch layer). Building on these findings, we further propose the VFM Stitch Tree (VST), which shares early layers across VFMs while retaining their later layers, yielding a controllable accuracy-latency trade-off for multimodal LLMs that often leverage multiple VFMs. Taken together, our study elevates stitching from a diagnostic probe to a practical recipe for integrating complementary VFM strengths and pinpointing where their representations align or diverge.
>
---
#### [new 063] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文聚焦多人体3D运动编辑任务，解决文本引导下多人互动运动生成的问题。提出InterEdit模型和相关数据集，提升编辑一致性和准确性。**

- **链接: [https://arxiv.org/pdf/2603.13082](https://arxiv.org/pdf/2603.13082)**

> **作者:** Yebin Yang; Di Wen; Lei Qi; Weitong Kong; Junwei Zheng; Ruiping Liu; Yufan Chen; Chengzhi Wu; Kailun Yang; Yuqian Fu; Danda Pani Paudel; Luc Van Gool; Kunyu Peng
>
> **备注:** The dataset and code will be released at this https URL
>
> **摘要:** Text-guided 3D motion editing has seen success in single-person scenarios, but its extension to multi-person settings is less explored due to limited paired data and the complexity of inter-person interactions. We introduce the task of multi-person 3D motion editing, where a target motion is generated from a source and a text instruction. To support this, we propose InterEdit3D, a new dataset with manual two-person motion change annotations, and a Text-guided Multi-human Motion Editing (TMME) benchmark. We present InterEdit, a synchronized classifier-free conditional diffusion model for TMME. It introduces Semantic-Aware Plan Token Alignment with learnable tokens to capture high-level interaction cues and an Interaction-Aware Frequency Token Alignment strategy using DCT and energy pooling to model periodic motion dynamics. Experiments show that InterEdit improves text-to-motion consistency and edit fidelity, achieving state-of-the-art TMME performance. The dataset and code will be released at this https URL.
>
---
#### [new 064] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感识别任务，旨在解决真实场景下情绪（愉悦度和唤醒度）的连续估计问题。通过融合面部、行为和音频多模态信息，提出两种融合策略提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.13056](https://arxiv.org/pdf/2603.13056)**

> **作者:** Elena Ryumina; Maxim Markitantov; Alexandr Axyonov; Dmitry Ryumin; Mikhail Dolgushin; Denis Dresvyanskiy; Alexey Karpov
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** Continuous emotion recognition in terms of valence and arousal under in-the-wild (ITW) conditions remains a challenging problem due to large variations in appearance, head pose, illumination, occlusions, and subject-specific patterns of affective expression. We present a multimodal method for valence-arousal estimation ITW. Our method combines three complementary modalities: face, behavior, and audio. The face modality relies on GRADA-based frame-level embeddings and Transformer-based temporal regression. We use Qwen3-VL-4B-Instruct to extract behavior-relevant information from video segments, while Mamba is used to model temporal dynamics across segments. The audio modality relies on WavLM-Large with attention-statistics pooling and includes a cross-modal filtering stage to reduce the influence of unreliable or non-speech segments. To fuse modalities, we explore two fusion strategies: a Directed Cross-Modal Mixture-of-Experts Fusion Strategy that learns interactions between modalities with adaptive weighting, and a Reliability-Aware Audio-Visual Fusion Strategy that combines visual features at the frame-level while using audio as complementary context. The results are reported on the Aff-Wild2 dataset following the 10th Affective Behavior Analysis in-the-Wild (ABAW) challenge protocol. Experiments demonstrate that the proposed multimodal fusion strategy achieves a Concordance Correlation Coefficient (CCC) of 0.658 on the Aff-Wild2 development set.
>
---
#### [new 065] SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization
- **分类: cs.CV**

- **简介: 该论文属于非刚性三维形状匹配任务，解决非等距变形和拓扑噪声下的对应问题。提出SGMatch框架，结合语义信息与流正则化，提升匹配精度。**

- **链接: [https://arxiv.org/pdf/2603.12937](https://arxiv.org/pdf/2603.12937)**

> **作者:** Tianwei Ye; Xiaoguang Mei; Yifan Xia; Fan Fan; Jun Huang; Jiayi Ma
>
> **备注:** 27 pages, 13 figures
>
> **摘要:** Establishing accurate point-to-point correspondences between non-rigid 3D shapes remains a critical challenge, particularly under non-isometric deformations and topological noise. Existing functional map pipelines suffer from ambiguities that geometric descriptors alone cannot resolve, and spatial inconsistencies inherent in the projection of truncated spectral bases to dense pointwise correspondences. In this paper, we introduce SGMatch, a learning-based framework for semantic-guided non-rigid shape matching. Specifically, we design a Semantic-Guided Local Cross-Attention module that integrates semantic features from vision foundation models into geometric descriptors while preserving local structural continuity. Furthermore, we introduce a regularization objective based on conditional flow matching, which supervises a time-varying velocity field to encourage spatial smoothness of the recovered correspondences. Experimental results on multiple benchmarks demonstrate that SGMatch achieves competitive performance across near-isometric settings and consistent improvements under non-isometric deformations and topological noise.
>
---
#### [new 066] A Prediction-as-Perception Framework for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文提出PAP框架，用于3D目标检测任务，通过整合预测与感知模块提升模型准确性与效率，解决目标跟踪精度低和计算资源消耗大的问题。**

- **链接: [https://arxiv.org/pdf/2603.12599](https://arxiv.org/pdf/2603.12599)**

> **作者:** Song Zhang; Haoyu Chen; Ruibo Wang
>
> **摘要:** Humans combine prediction and perception to observe the world. When faced with rapidly moving birds or insects, we can only perceive them clearly by predicting their next position and focusing our gaze there. Inspired by this, this paper proposes the Prediction-As-Perception (PAP) framework, integrating a prediction-perception architecture into 3D object perception tasks to enhance the model's perceptual accuracy. The PAP framework consists of two main modules: prediction and perception, primarily utilizing continuous frame information as input. Firstly, the prediction module forecasts the potential future positions of ego vehicles and surrounding traffic participants based on the perception results of the current frame. These predicted positions are then passed as queries to the perception module of the subsequent frame. The perceived results are iteratively fed back into the prediction module. We evaluated the PAP structure using the end-to-end model UniAD on the nuScenes dataset. The results demonstrate that the PAP structure improves UniAD's target tracking accuracy by 10% and increases the inference speed by 15%. This indicates that such a biomimetic design significantly enhances the efficiency and accuracy of perception models while reducing computational resource consumption.
>
---
#### [new 067] STRAP-ViT: Segregated Tokens with Randomized -- Transformations for Defense against Adversarial Patches in ViTs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉Transformer的对抗防御任务，解决对抗补丁攻击问题。提出STRAP-ViT方法，通过分割异常token并应用随机变换来提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12688](https://arxiv.org/pdf/2603.12688)**

> **作者:** Nandish Chattopadhyay; Anadi Goyal; Chandan Karfa; Anupam Chattopadhyay
>
> **备注:** Accepted for publication at IEEE/ACM Design Automation Conference (DAC) 2026
>
> **摘要:** Adversarial patches are physically realizable localized noise, which are able to hijack Vision Transformers (ViT) self-attention, pulling focus toward a small, high-contrast region and corrupting the class token to force confident misclassifications. In this paper, we claim that the tokens which correspond to the areas of the image that contain the adversarial noise, have different statistical properties when compared to the tokens which do not overlap with the adversarial perturbations. We use this insight to propose a mechanism, called STRAP-ViT, which uses Jensen-Shannon Divergence as a metric for segregating tokens that behave as anomalies in the Detection Phase, and then apply randomized composite transformations on them during the Mitigation Phase to make the adversarial noise ineffective. The minimum number of tokens to transform is a hyper-parameter for the defense mechanism and is chosen such that at least 50% of the patch is covered by the transformed tokens. STRAP-ViT fits as a non-trainable plug-and-play block within the ViT architectures, for inference purposes only, with a minimal computational cost and does not require any additional training cost/effort. STRAP-ViT has been tested on multiple pre-trained vision transformer architectures (ViT-base-16 and DinoV2) and datasets (ImageNet and CalTech-101), across multiple adversarial attacks (Adversarial Patch, LAVAN, GDPA and RP2), and found to provide excellent robust accuracies lying within a 2-3% range of the clean baselines, and outperform the state-of-the-art.
>
---
#### [new 068] MRGeo: Robust Cross-View Geo-Localization of Corrupted Images via Spatial and Channel Feature Enhancement
- **分类: cs.CV**

- **简介: 该论文属于跨视图地理定位任务，解决图像退化下的定位鲁棒性问题。提出MRGeo方法，通过空间-通道增强和几何对齐提升特征质量与一致性。**

- **链接: [https://arxiv.org/pdf/2603.12587](https://arxiv.org/pdf/2603.12587)**

> **作者:** Le Wu; Lv Bo; Songsong Ouyang; Yingying Zhu
>
> **摘要:** Cross-view geo-localization (CVGL) aims to accurately localize street-view images through retrieval of corresponding geo-tagged satellite images. While prior works have achieved nearly perfect performance on certain standard datasets, their robustness in real-world corrupted environments remains under-explored. This oversight causes severe performance degradation or failure when images are affected by corruption such as blur or weather, significantly limiting practical deployment. To address this critical gap, we introduce MRGeo, the first systematic method designed for robust CVGL under corruption. MRGeo employs a hierarchical defense strategy that enhances the intrinsic quality of features and then enforces a robust geometric prior. Its core is the Spatial-Channel Enhancement Block, which contains: (1) a Spatial Adaptive Representation Module that models global and local features in parallel and uses a dynamic gating mechanism to arbitrate their fusion based on feature reliability; and (2) a Channel Calibration Module that performs compensatory adjustments by modeling multi-granularity channel dependencies to counteract information loss. To prevent spatial misalignment under severe corruption, a Region-level Geometric Alignment Module imposes a geometric structure on the final descriptors, ensuring coarse-grained consistency. Comprehensive experiments on both robustness benchmark and standard datasets demonstrate that MRGeo not only achieves an average R@1 improvement of 2.92\% across three comprehensive robustness benchmarks (CVUSA-C-ALL, CVACT\_val-C-ALL, and CVACT\_test-C-ALL) but also establishes superior performance in cross-area evaluation, thereby demonstrating its robustness and generalization capability.
>
---
#### [new 069] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于联邦域泛化行人重识别任务，解决跨客户端分布差异导致的特征不一致问题。提出FedBPrompt，通过视觉提示机制增强特征区分性并降低通信成本。**

- **链接: [https://arxiv.org/pdf/2603.12912](https://arxiv.org/pdf/2603.12912)**

> **作者:** Xin Xu; Weilong Li; Wei Liu; Wenke Huang; Zhixi Yu; Bin Yang; Xiaoying Liao; Kui Jiang
>
> **摘要:** Federated Domain Generalization for Person Re-Identification (FedDG-ReID) learns domain-invariant representations from decentralized data. While Vision Transformer (ViT) is widely adopted, its global attention often fails to distinguish pedestrians from high similarity backgrounds or diverse viewpoints -- a challenge amplified by cross-client distribution shifts in FedDG-ReID. To address this, we propose Federated Body Distribution Aware Visual Prompt (FedBPrompt), introducing learnable visual prompts to guide Transformer attention toward pedestrian-centric regions. FedBPrompt employs a Body Distribution Aware Visual Prompts Mechanism (BAPM) comprising: Holistic Full Body Prompts to suppress cross-client background noise, and Body Part Alignment Prompts to capture fine-grained details robust to pose and viewpoint variations. To mitigate high communication costs, we design a Prompt-based Fine-Tuning Strategy (PFTS) that freezes the ViT backbone and updates only lightweight prompts, significantly reducing communication overhead while maintaining adaptability. Extensive experiments demonstrate that BAPM effectively enhances feature discrimination and cross-domain generalization, while PFTS achieves notable performance gains within only a few aggregation rounds. Moreover, both BAPM and PFTS can be easily integrated into existing ViT-based FedDG-ReID frameworks, making FedBPrompt a flexible and effective solution for federated person re-identification. The code is available at this https URL.
>
---
#### [new 070] RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization
- **分类: cs.CV**

- **简介: 该论文提出RoboStereo，解决 embodied AI 中真实交互成本高、安全风险大的问题，通过4D世界模型实现统一策略优化。**

- **链接: [https://arxiv.org/pdf/2603.12639](https://arxiv.org/pdf/2603.12639)**

> **作者:** Ruicheng Zhang; Guangyu Chen; Zunnan Xu; Zihao Liu; Zhizhou Zhong; Mingyang Zhang; Jun Zhou; Xiu Li
>
> **摘要:** Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks.
>
---
#### [new 071] DINOLight: Robust Ambient Light Normalization with Self-supervised Visual Prior Integration
- **分类: cs.CV**

- **简介: 该论文提出DINOLight框架，用于解决环境光归一化问题，通过整合DINOv2的视觉先验提升图像恢复效果。**

- **链接: [https://arxiv.org/pdf/2603.12579](https://arxiv.org/pdf/2603.12579)**

> **作者:** Youngjin Oh; Junhyeong Kwon; Nam Ik Cho
>
> **备注:** Submitted to ICPR 2026 (under review)
>
> **摘要:** This paper presents a new ambient light normalization framework, DINOLight, that integrates the self-supervised model DINOv2's image understanding capability into the restoration process as a visual prior. Ambient light normalization aims to restore images degraded by non-uniform shadows and lighting caused by multiple light sources and complex scene geometries. We observe that DINOv2 can reliably extract both semantic and geometric information from a degraded image. Based on this observation, we develop a novel framework to utilize DINOv2 features for lighting normalization. First, we propose an adaptive feature fusion module that combines features from different DINOv2 layers using a point-wise softmax mask. Next, the fused features are integrated into our proposed restoration network in both spatial and frequency domains through an auxiliary cross-attention mechanism. Experiments show that DINOLight achieves superior performance on the Ambient6K dataset, and that DINOv2 features are effective for enhancing ambient light normalization. We also apply our method to shadow-removal benchmark datasets, achieving competitive results compared to methods that use mask priors. Codes will be released upon acceptance.
>
---
#### [new 072] Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言接地任务，旨在解决负向语义理解不足的问题。通过构建D-Negation数据集和提出分组对立学习框架，提升模型对负向描述的准确识别与定位能力。**

- **链接: [https://arxiv.org/pdf/2603.12606](https://arxiv.org/pdf/2603.12606)**

> **作者:** Zesheng Yang; Xi Jiang; Bingzhang Hu; Weili Guan; Runmin Cong; Guo-Jun Qi; Feng Zheng
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Current vision-language detection and grounding models predominantly focus on prompts with positive semantics and often struggle to accurately interpret and ground complex expressions containing negative semantics. A key reason for this limitation is the lack of high-quality training data that explicitly captures discriminative negative samples and negation-aware language descriptions. To address this challenge, we introduce D-Negation, a new dataset that provides objects annotated with both positive and negative semantic descriptions. Building upon the observation that negation reasoning frequently appears in natural language, we further propose a grouped opposition-based learning framework that learns negation-aware representations from limited samples. Specifically, our method organizes opposing semantic descriptions from D-Negation into structured groups and formulates two complementary loss functions that encourage the model to reason about negation and semantic qualifiers. We integrate the proposed dataset and learning strategy into a state-of-the-art language-based grounding model. By fine-tuning fewer than 10 percent of the model parameters, our approach achieves improvements of up to 4.4 mAP and 5.7 mAP on positive and negative semantic evaluations, respectively. These results demonstrate that explicitly modeling negation semantics can substantially enhance the robustness and localization accuracy of vision-language grounding models.
>
---
#### [new 073] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models
- **分类: cs.CV**

- **简介: 该论文属于视频世界模型任务，旨在解决模型能否在无观察情况下独立演化状态的问题。通过设计STEVO-Bench基准，评估模型在控制观察条件下的演化能力。**

- **链接: [https://arxiv.org/pdf/2603.13215](https://arxiv.org/pdf/2603.13215)**

> **作者:** Ziqi Ma; Mengzhan Liufu; Georgia Gkioxari
>
> **备注:** this https URL
>
> **摘要:** Evolutions in the world, such as water pouring or ice melting, happen regardless of being observed. Video world models generate "worlds" via 2D frame observations. Can these generated "worlds" evolve regardless of observation? To probe this question, we design a benchmark to evaluate whether video world models can decouple state evolution from observation. Our benchmark, STEVO-Bench, applies observation control to evolving processes via instructions of occluder insertion, turning off the light, or specifying camera "lookaway" trajectories. By evaluating video models with and without camera control for a diverse set of naturally-occurring evolutions, we expose their limitations in decoupling state evolution from observation. STEVO-Bench proposes an evaluation protocol to automatically detect and disentangle failure modes of video world models across key aspects of natural state evolution. Analysis of STEVO-Bench results provide new insight into potential data and architecture bias of present-day video world models. Project website: this https URL. Blog: this https URL
>
---
#### [new 074] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering
- **分类: cs.CV**

- **简介: 该论文聚焦于CAD建模与逆向工程任务，旨在解决BRep特征理解不足的问题。通过构建包含1000万注释的A2Z数据集，提升几何深度学习能力。**

- **链接: [https://arxiv.org/pdf/2603.12605](https://arxiv.org/pdf/2603.12605)**

> **作者:** Pritham Kumar Jena; Bhavika Baburaj; Tushar Anand; Vedant Dutta; Vineeth Ulavala; Sk Aziz Ali
>
> **备注:** 27 pages, accepted to IEEE CVF CVPR 2026
>
> **摘要:** Reverse engineering and rapid prototyping of computer-aided design (CAD) models from 3D scans, sketches, or simple text prompts are vital in industrial product design. However, recent advances in geometric deep learning techniques lack a multi-modal understanding of parametric CAD features stored in their boundary representation (BRep). This study presents the largest compilation of 10 million multi-modal annotations and metadata for 1 million ABC CAD models, namely A2Z, to unlock an unprecedented level of BRep learning. A2Z comprises (i) high-resolution meshes with salient 3D scanning features, (ii) 3D hand-drawn sketches equipped with (iii) geometric and topological information about BRep co-edges, corners, and surfaces, and (iv) textual captions and tags describing the product in the mechanical world. Creating such carefully structured, large-scale data, which requires nearly 5 terabytes of storage to leverage unparalleled CAD learning/retrieval tasks, is very challenging. The scale, quality, and diversity of our multi-modal annotations are assessed using novel metrics, GPT-5, Gemini, and extensive human feedback mechanisms. To this end, we also merge an additional 25,000 CAD models of electronic enclosures (e.g., tablets, ports) designed by skilled professionals with our A2Z dataset. Subsequently, we train and benchmark a foundation model on a subset of 150K CAD models to detect BRep co-edges and corner vertices from 3D scans, a key downstream task in CAD reverse engineering. The annotated dataset, metrics, and checkpoints will be publicly released to support numerous research directions.
>
---
#### [new 075] Bin~Wan,G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文属于显著目标检测任务，解决遥感图像中尺度变化大、背景复杂的问题。提出G2HFNet网络，融合多尺度特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12680](https://arxiv.org/pdf/2603.12680)**

> **作者:** Bin Wan; Runmin Cong; Xiaofei Zhou; Hao Fang; Chengtao Lv; Sam Kwong
>
> **摘要:** Remote sensing images captured from aerial perspectives often exhibit significant scale variations and complex backgrounds, posing challenges for salient object detection (SOD). Existing methods typically extract multi-level features at a single scale using uniform attention mechanisms, leading to suboptimal representations and incomplete detection results. To address these issues, we propose a GeoGran-Aware Hierarchical Feature Fusion Network (G2HFNet) that fully exploits geometric and granular cues in optical remote sensing images. Specifically, G2HFNet adopts Swin Transformer as the backbone to extract multi-level features and integrates three key modules: the multi-scale detail enhancement (MDE) module to handle object scale variations and enrich fine details, the dual-branch geo-gran complementary (DGC) module to jointly capture fine-grained details and positional information in mid-level features, and the deep semantic perception (DSP) module to refine high-level positional cues via self-attention. Additionally, a local-global guidance fusion (LGF) module is introduced to replace traditional convolutions for effective multi-level feature integration. Extensive experiments demonstrate that G2HFNet achieves high-quality saliency maps and significantly improves detection performance in challenging remote sensing scenarios.
>
---
#### [new 076] Thinking in Streaming Video
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ThinkStream框架，解决实时视频流理解问题。针对传统批处理方法延迟高、成本大的缺陷，采用Watch--Think--Speak范式和RCSM技术，实现低延迟、高效推理。**

- **链接: [https://arxiv.org/pdf/2603.12938](https://arxiv.org/pdf/2603.12938)**

> **作者:** Zikang Liu; Longteng Guo; Handong Li; Ru Zhen; Xingjian He; Ruyi Ji; Xiaoming Ren; Yanhao Zhang; Haonan Lu; Jing Liu
>
> **摘要:** Real-time understanding of continuous video streams is essential for interactive assistants and multimodal agents operating in dynamic environments. However, most existing video reasoning approaches follow a batch paradigm that defers reasoning until the full video context is observed, resulting in high latency and growing computational cost that are incompatible with streaming scenarios. In this paper, we introduce ThinkStream, a framework for streaming video reasoning based on a Watch--Think--Speak paradigm that enables models to incrementally update their understanding as new video observations arrive. At each step, the model performs a short reasoning update and decides whether sufficient evidence has accumulated to produce a response. To support long-horizon streaming, we propose Reasoning-Compressed Streaming Memory (RCSM), which treats intermediate reasoning traces as compact semantic memory that replaces outdated visual tokens while preserving essential context. We further train the model using a Streaming Reinforcement Learning with Verifiable Rewards scheme that aligns incremental reasoning and response timing with the requirements of streaming interaction. Experiments on multiple streaming video benchmarks show that ThinkStream significantly outperforms existing online video models while maintaining low latency and memory usage. Code, models and data will be released at this https URL
>
---
#### [new 077] Towards Faithful Multimodal Concept Bottleneck Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态任务，旨在解决概念瓶颈模型（CBM）中概念检测与信息泄漏问题。通过联合优化检测与泄漏抑制，提升模型的可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.13163](https://arxiv.org/pdf/2603.13163)**

> **作者:** Pierre Moreau; Emeline Pineau Ferrand; Yann Choho; Benjamin Wong; Annabelle Blangero; Milan Bhan
>
> **摘要:** Concept Bottleneck Models (CBMs) are interpretable models that route predictions through a layer of human-interpretable concepts. While widely studied in vision and, more recently, in NLP, CBMs remain largely unexplored in multimodal settings. For their explanations to be faithful, CBMs must satisfy two conditions: concepts must be properly detected, and concept representations must encode only their intended semantics, without smuggling extraneous task-relevant or inter-concept information into final predictions, a phenomenon known as leakage. Existing approaches treat concept detection and leakage mitigation as separate problems, and typically improve one at the expense of predictive accuracy. In this work, we introduce f-CBM, a faithful multimodal CBM framework built on a vision-language backbone that jointly targets both aspects through two complementary strategies: a differentiable leakage loss to mitigate leakage, and a Kolmogorov-Arnold Network prediction head that provides sufficient expressiveness to improve concept detection. Experiments demonstrate that f-CBM achieves the best trade-off between task accuracy, concept detection, and leakage reduction, while applying seamlessly to both image and text or text-only datasets, making it versatile across modalities.
>
---
#### [new 078] Unleashing Video Language Models for Fine-grained HRCT Report Generation
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决HRCT报告生成中的精准性与 hallucination 问题。提出AbSteering框架，通过异常导向的推理和优化目标提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12469](https://arxiv.org/pdf/2603.12469)**

> **作者:** Yingying Fang; Huichi Zhou; KinHei Lee; Yijia Wang; Zhenxuan Zhang; Jiahao Huang; Guang Yang
>
> **备注:** MICCAI 2026
>
> **摘要:** Generating precise diagnostic reports from High-Resolution Computed Tomography (HRCT) is critical for clinical workflow, yet it remains a formidable challenge due to the high pathological diversity and spatial sparsity within 3D volumes. While Video Language Models (VideoLMs) have demonstrated remarkable spatio-temporal reasoning in general domains, their adaptability to domain-specific, high-volume medical interpretation remains underexplored. In this work, we present AbSteering, an abnormality-centric framework that steers VideoLMs toward precise HRCT report generation. Specifically, AbSteering introduces: (i) an abnormality-centric Chain-of-Thought scheme that enforces abnormality reasoning, and (ii) a Direct Preference Optimization objective that utilizes clinically confusable abnormalities as hard negatives to enhance fine-grained discrimination. Our results demonstrate that general-purpose VideoLMs possess strong transferability to high-volume medical imaging when guided by this paradigm. Notably, AbSteering outperforms state-of-the-art domain-specific CT foundation models, which are pretrained with large-scale CTs, achieving superior detection sensitivity while simultaneously mitigating hallucinations. Our data and model weights are released at this https URL
>
---
#### [new 079] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态情感识别任务，旨在解决视频中犹豫/矛盾情绪的识别问题。通过融合场景、面部、音频和文本信息，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.12848](https://arxiv.org/pdf/2603.12848)**

> **作者:** Elena Ryumina; Alexandr Axyonov; Dmitry Sysoev; Timur Abdulkadirov; Kirill Almetov; Yulia Morozova; Dmitry Ryumin
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Ambivalence/hesitancy recognition in unconstrained videos is a challenging problem due to the subtle, multimodal, and context-dependent nature of this behavioral state. In this paper, a multimodal approach for video-level ambivalence/hesitancy recognition is presented for the 10th ABAW Competition. The proposed approach integrates four complementary modalities: scene, face, audio, and text. Scene dynamics are captured with a VideoMAE-based model, facial information is encoded through emotional frame-level embeddings aggregated by statistical pooling, acoustic representations are extracted with EmotionWav2Vec2.0 and processed by a Mamba-based temporal encoder, and linguistic cues are modeled using fine-tuned transformer-based text models. The resulting unimodal embeddings are further combined using multimodal fusion models, including prototype-augmented variants. Experiments on the BAH corpus demonstrate clear gains of multimodal fusion over all unimodal baselines. The best unimodal configuration achieved an average MF1 of 70.02%, whereas the best multimodal fusion model reached 83.25%. The highest final test performance, 71.43%, was obtained by an ensemble of five prototype-augmented fusion models. The obtained results highlight the importance of complementary multimodal cues and robust fusion strategies for ambivalence/hesitancy recognition.
>
---
#### [new 080] VFM-Recon: Unlocking Cross-Domain Scene-Level Neural Reconstruction with Scale-Aligned Foundation Priors
- **分类: cs.CV**

- **简介: 该论文属于场景级神经体积重建任务，解决单目视频在域转移下的尺度一致性问题。通过引入轻量级尺度对齐和适配器，融合预训练视觉基础模型特征，提升重建性能。**

- **链接: [https://arxiv.org/pdf/2603.12657](https://arxiv.org/pdf/2603.12657)**

> **作者:** Yuhang Ming; Tingkang Xi; Xingrui Yang; Lixin Yang; Yong Peng; Cewu Lu; Wanzeng Kong
>
> **备注:** 19 pages, 5 figures, 4 tables
>
> **摘要:** Scene-level neural volumetric reconstruction from monocular videos remains challenging, especially under severe domain shifts. Although recent advances in vision foundation models (VFMs) provide transferable generalized priors learned from large-scale data, their scaleambiguous predictions are incompatible with the scale consistency required by volumetric fusion. To address this gap, we present VFMRecon, the first attempt to bridge transferable VFM priors with scaleconsistent requirements in scene-level neural reconstruction. Specifically, we first introduce a lightweight scale alignment stage that restores multiview scale coherence. We then integrate pretrained VFM features into the neural volumetric reconstruction pipeline via lightweight task-specific adapters, which are trained for reconstruction while preserving the crossdomain robustness of pretrained representations. We train our model on ScanNet train split and evaluate on both in-distribution ScanNet test split and out-of-distribution TUM RGB-D and Tanks and Temples datasets. The results demonstrate that our model achieves state-of-theart performance across all datasets domains. In particular, on the challenging outdoor Tanks and Temples dataset, our model achieves an F1 score of 70.1 in reconstructed mesh evaluation, substantially outperforming the closest competitor, VGGT, which only attains 51.8.
>
---
#### [new 081] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决机器人识别新物体的问题。通过观看人类演示自动生成数据，训练专用检测器，避免语言描述和复杂提示工程。**

- **链接: [https://arxiv.org/pdf/2603.12751](https://arxiv.org/pdf/2603.12751)**

> **作者:** James Akl; Jose Nicolas Avendano Arbelaez; James Barabas; Jennifer L. Barry; Kalie Ching; Noam Eshed; Jiahui Fu; Michel Hidalgo; Andrew Hoelscher; Tushar Kusnur; Andrew Messing; Zachary Nagler; Brian Okorn; Mauro Passerino; Tim J. Perkins; Eric Rosen; Ankit Shah; Tanmay Shankar; Scott Shaw
>
> **摘要:** How can a robot quickly identify and recognize new objects shown to it during a human demonstration? Existing closed-set object detectors frequently fail at this because the objects are out-of-distribution. While open-set detectors (e.g., VLMs) sometimes succeed, they often require expensive and tedious human-in-the-loop prompt engineering to uniquely recognize novel object instances. In this paper, we present a self-supervised system that eliminates the need for tedious language descriptions and expensive prompt engineering by training a bespoke object detector on an automatically created dataset, supervised by the human demonstration itself. In our approach, "Show, Don't Tell," we show the detector the specific objects of interest during the demonstration, rather than telling the detector about these objects via complex language descriptions. By bypassing language altogether, this paradigm enables us to quickly train bespoke detectors tailored to the relevant objects observed in human task demonstrations. We develop an integrated on-robot system to deploy our "Show, Don't Tell" paradigm of automatic dataset creation and novel object-detection on a real-world robot. Empirical results demonstrate that our pipeline significantly outperforms state-of-the-art detection and recognition methods for manipulated objects, leading to improved task completion for the robot.
>
---
#### [new 082] SortScrews: A Dataset and Baseline for Real-time Screw Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SortScrews数据集，用于实时螺丝分类任务，解决工业自动化中螺丝类型识别问题，通过标准化采集和轻量模型实现高准确率。**

- **链接: [https://arxiv.org/pdf/2603.13027](https://arxiv.org/pdf/2603.13027)**

> **作者:** Tianhao Fu; Bingxuan Yang; Juncheng Guo; Shrena Sribalan; Yucheng Chen
>
> **摘要:** Automatic identification of screw types is important for industrial automation, robotics, and inventory management. However, publicly available datasets for screw classification are scarce, particularly for controlled single-object scenarios commonly encountered in automated sorting systems. In this work, we introduce $\textbf{SortScrews}$, a dataset for casewise visual classification of screws. The dataset contains 560 RGB images at $512\times512$ resolution covering six screw types and a background class. Images are captured using a standardized acquisition setup and include mild variations in lighting and camera perspective across four capture settings. To facilitate reproducible research and dataset expansion, we also provide a reusable data collection script that allows users to easily construct similar datasets for custom hardware components using inexpensive camera setups. We establish baseline results using transfer learning with EfficientNet-B0 and ResNet-18 classifiers pretrained on ImageNet. In addition, we conduct a well-explored failure analysis. Despite the limited dataset size, these lightweight models achieve strong classification accuracy, demonstrating that controlled acquisition conditions enable effective learning even with relatively small datasets. The dataset, collection pipeline, and baseline training code are publicly available at this https URL.
>
---
#### [new 083] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于跨视角姿态估计任务，旨在解决地面与卫星视角差异大导致的定位误差问题。提出VIRD方法，通过双轴变换构建视图不变表示，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.12918](https://arxiv.org/pdf/2603.12918)**

> **作者:** Juhye Park; Wooju Lee; Dasol Hong; Changki Sung; Youngwoo Seo; Dongwan Kang; Hyun Myung
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Accurate global localization is crucial for autonomous driving and robotics, but GNSS-based approaches often degrade due to occlusion and multipath effects. As an emerging alternative, cross-view pose estimation predicts the 3-DoF camera pose corresponding to a ground-view image with respect to a geo-referenced satellite image. However, existing methods struggle to bridge the significant viewpoint gap between the ground and satellite views mainly due to limited spatial correspondences. We propose a novel cross-view pose estimation method that constructs view-invariant representations through dual-axis transformation (VIRD). VIRD first applies a polar transformation to the satellite view to establish horizontal correspondence, then uses context-enhanced positional attention on the ground and polar-transformed satellite features to resolve vertical misalignment, explicitly mitigating the viewpoint gap. A view-reconstruction loss is introduced to strengthen the view invariance further, encouraging the derived representations to reconstruct the original and cross-view images. Experiments on the KITTI and VIGOR datasets demonstrate that VIRD outperforms the state-of-the-art methods without orientation priors, reducing median position and orientation errors by 50.7% and 76.5% on KITTI, and 18.0% and 46.8% on VIGOR, respectively.
>
---
#### [new 084] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于RGB-T显著目标检测任务，解决RGB与热成像间显著区域不一致问题。提出RSONet，包含区域引导和显著性生成模块，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.12685](https://arxiv.org/pdf/2603.12685)**

> **作者:** Bin Wan; Runmin Cong; Xiaofei Zhou; Hao Fang; Chengtao Lv; Sam Kwong
>
> **摘要:** This paper focuses on the inconsistency in salient regions between RGB and thermal images. To address this issue, we propose the Region-guided Selective Optimization Network for RGB-T Salient Object Detection, which consists of the region guidance stage and saliency generation stage. In the region guidance stage, three parallel branches with same encoder-decoder structure equipped with the context interaction (CI) module and spatial-aware fusion (SF) module are designed to generate the guidance maps which are leveraged to calculate similarity scores. Then, in the saliency generation stage, the selective optimization (SO) module fuses RGB and thermal features based on the previously obtained similarity values to mitigate the impact of inconsistent distribution of salient targets between the two modalities. After that, to generate high-quality detection result, the dense detail enhancement (DDE) module which adopts the multiple dense connections and visual state space blocks is applied to low-level features for optimizing the detail information. In addition, the mutual interaction semantic (MIS) module is placed in the high-level features to dig the location cues by the mutual fusion strategy. We conduct extensive experiments on the RGB-T dataset, and the results demonstrate that the proposed RSONet achieves competitive performance against 27 state-of-the-art SOD methods.
>
---
#### [new 085] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有方法泛化能力差和计算复杂度高的问题。提出一种以解码器为中心的结构，结合Transformer与Mamba，提升多尺度上下文表示和分割性能。**

- **链接: [https://arxiv.org/pdf/2603.12547](https://arxiv.org/pdf/2603.12547)**

> **作者:** Fares Bougourzi; Fadi Dornaika; Abdenour Hadid
>
> **摘要:** Deep learning has achieved remarkable success in medical image segmentation, often reaching expert-level accuracy in delineating tumors and tissues. However, most existing approaches remain task-specific, showing strong performance on individual datasets but limited generalization across diverse imaging modalities. Moreover, many methods focus primarily on the encoder, relying on large pretrained backbones that increase computational complexity. In this paper, we propose a decoder-centric approach for generalized 2D medical image segmentation. The proposed Deco-Mamba follows a U-Net-like structure with a Transformer-CNN-Mamba design. The encoder combines a CNN block and Transformer backbone for efficient feature extraction, while the decoder integrates our novel Co-Attention Gate (CAG), Vision State Space Module (VSSM), and deformable convolutional refinement block to enhance multi-scale contextual representation. Additionally, a windowed distribution-aware KL-divergence loss is introduced for deep supervision across multiple decoding stages. Extensive experiments on diverse medical image segmentation benchmarks yield state-of-the-art performance and strong generalization capability while maintaining moderate model complexity. The source code will be released upon acceptance.
>
---
#### [new 086] NOIR: Neural Operator mapping for Implicit Representations
- **分类: cs.CV**

- **简介: 该论文提出NOIR框架，将医学影像任务转化为连续函数空间的算子学习，解决传统离散网格方法的局限性，通过隐式神经表示实现高分辨率、鲁棒的图像处理。**

- **链接: [https://arxiv.org/pdf/2603.13118](https://arxiv.org/pdf/2603.13118)**

> **作者:** Sidaty El Hadramy; Nazim Haouchine; Michael Wehrli; Philippe C. Cattin
>
> **摘要:** This paper presents NOIR, a framework that reframes core medical imaging tasks as operator learning between continuous function spaces, challenging the prevailing paradigm of discrete grid-based deep learning. Instead of operating on fixed pixel or voxel grids, NOIR embeds discrete medical signals into shared Implicit Neural Representations and learns a Neural Operator that maps between their latent modulations, enabling resolution-independent function-to-function transformations. We evaluate NOIR across multiple 2D and 3D downstream tasks, including segmentation, shape completion, image-to-image translation, and image synthesis, on several public datasets such as Shenzhen, OASIS-4, SkullBreak, fastMRI, as well as an in-house clinical dataset. It achieves competitive performance at native resolution while demonstrating strong robustness to unseen discretizations, and empirically satisfies key theoretical properties of neural operators. The project page is available here: this https URL.
>
---
#### [new 087] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属于图像水印任务，旨在解决扩散模型初始噪声水印易被伪造的问题。提出SLICE方法，通过细粒度语义绑定提升水印鲁棒性与可定位性。**

- **链接: [https://arxiv.org/pdf/2603.12749](https://arxiv.org/pdf/2603.12749)**

> **作者:** Zheng Gao; Yifan Yang; Xiaoyu Li; Xiaoyan Feng; Haoran Fan; Yang Song; Jiaojiao Jiang
>
> **摘要:** Watermarking the initial noise of diffusion models has emerged as a promising approach for image provenance, but content-independent noise patterns can be forged via inversion and regeneration attacks. Recent semantic-aware watermarking methods improve robustness by conditioning verification on image semantics. However, their reliance on a single global semantic binding makes them vulnerable to localized but globally coherent semantic edits. To address this limitation and provide a trustworthy semantic-aware watermark, we propose $\underline{\textbf{S}}$emantic $\underline{\textbf{L}}$atent $\underline{\textbf{I}}$njection via $\underline{\textbf{C}}$ompartmentalized $\underline{\textbf{E}}$mbedding ($\textbf{SLICE}$). Our framework decouples image semantics into four semantic factors (subject, environment, action, and detail) and precisely anchors them to distinct regions in the initial Gaussian noise. This fine-grained semantic binding enables advanced watermark verification where semantic tampering is detectable and localizable. We theoretically justify why SLICE enables robust and reliable tamper localization and provides statistical guarantees on false-accept rates. Experimental results demonstrate that SLICE significantly outperforms existing baselines against advanced semantic-guided regeneration attacks, substantially reducing attack success while preserving image quality and semantic fidelity. Overall, SLICE offers a practical, training-free provenance solution that is both fine-grained in diagnosis and robust to realistic adversarial manipulations.
>
---
#### [new 088] Topo-R1: Detecting Topological Anomalies via Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于拓扑异常检测任务，旨在无需标注数据的情况下识别结构中的拓扑错误。通过构建基准和提出Topo-R1框架提升模型的拓扑感知能力。**

- **链接: [https://arxiv.org/pdf/2603.13054](https://arxiv.org/pdf/2603.13054)**

> **作者:** Meilong Xu; Qingqiao Hu; Xiaoling Hu; Shahira Abousamra; Xin Yu; Weimin Lyu; Kehan Qi; Dimitris Samaras; Chao Chen
>
> **备注:** 28 pages, 6 figures
>
> **摘要:** Topological correctness is crucial for tubular structures such as blood vessels, nerve fibers, and road networks. Existing topology-preserving methods rely on domain-specific ground truth, which is costly and rarely transfers across domains. When deployed to a new domain without annotations, a key question arises: how can we detect topological anomalies without ground-truth supervision? We reframe this as topological anomaly detection, a structured visual reasoning task requiring a model to locate and classify topological errors in predicted segmentation masks. Vision-Language Models (VLMs) are natural candidates; however, we find that state-of-the-art VLMs perform nearly at random, lacking the fine-grained, topology-aware perception needed to identify sparse connectivity errors in dense structures. To bridge this gap, we develop an automated data-curation pipeline that synthesizes diverse topological anomalies with verifiable annotations across progressively difficult levels, thereby constructing the first large-scale, multi-domain benchmark for this task. We then introduce Topo-R1, a framework that endows VLMs with topology-aware perception via two-stage training: supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO). Central to our approach is a topology-aware composite reward that integrates type-aware Hungarian matching for structured error classification, spatial localization scoring, and a centerline Dice (clDice) reward that directly penalizes connectivity disruptions, thereby jointly incentivizing semantic precision and structural fidelity. Extensive experiments demonstrate that Topo-R1 establishes a new paradigm for annotation-free topological quality assessment, consistently outperforming general-purpose VLMs and supervised baselines across all evaluation protocols.
>
---
#### [new 089] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 referring image segmentation 任务，旨在解决语义定位不准确和边界模糊的问题。提出 SERA 架构，通过轻量级专家路由提升空间一致性与边界精度。**

- **链接: [https://arxiv.org/pdf/2603.12538](https://arxiv.org/pdf/2603.12538)**

> **作者:** Alaa Dalaq; Muzammil Behzad
>
> **摘要:** Referring image segmentation aims to produce a pixel-level mask for the image region described by a natural-language expression. Although pretrained vision-language models have improved semantic grounding, many existing methods still rely on uniform refinement strategies that do not fully match the diverse reasoning requirements of referring expressions. Because of this mismatch, predictions often contain fragmented regions, inaccurate boundaries, or even the wrong object, especially when pretrained backbones are frozen for computational efficiency. To address these limitations, we propose SERA, a Spatio-Semantic Expert Routing Architecture for referring image segmentation. SERA introduces lightweight, expression-aware expert refinement at two complementary stages within a vision-language framework. First, we design SERA-Adapter, which inserts an expression-conditioned adapter into selected backbone blocks to improve spatial coherence and boundary precision through expert-guided refinement and cross-modal attention. We then introduce SERA-Fusion, which strengthens intermediate visual representations by reshaping token features into spatial grids and applying geometry-preserving expert transformations before multimodal interaction. In addition, a lightweight routing mechanism adaptively weights expert contributions while remaining compatible with pretrained representations. To make this routing stable under frozen encoders, SERA uses a parameter-efficient tuning strategy that updates only normalization and bias terms, affecting less than 1% of the backbone parameters. Experiments on standard referring image segmentation benchmarks show that SERA consistently outperforms strong baselines, with especially clear gains on expressions that require accurate spatial localization and precise boundary delineation.
>
---
#### [new 090] Cheers: Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Cheers模型，解决多模态理解与生成任务中的统一问题。通过解耦视觉细节与语义表示，提升模型效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2603.12793](https://arxiv.org/pdf/2603.12793)**

> **作者:** Yichen Zhang; Da Peng; Zonghao Guo; Zijian Zhang; Xuesong Yang; Tong Sun; Shichu Sun; Yidan Zhang; Yanghao Li; Haiyan Zhao; Wang Xu; Qi Shi; Yangang Sun; Chi Chen; Shuo Wang; Yukun Yan; Xu Han; Qiang Ma; Wei Ke; Liang Wang; Zhiyuan Liu; Maosong Sun
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** A recent cutting-edge topic in multimodal modeling is to unify visual comprehension and generation within a single model. However, the two tasks demand mismatched decoding regimes and visual representations, making it non-trivial to jointly optimize within a shared feature space. In this work, we present Cheers, a unified multimodal model that decouples patch-level details from semantic representations, thereby stabilizing semantics for multimodal understanding and improving fidelity for image generation via gated detail residuals. Cheers includes three key components: (i) a unified vision tokenizer that encodes and compresses image latent states into semantic tokens for efficient LLM conditioning, (ii) an LLM-based Transformer that unifies autoregressive decoding for text generation and diffusion decoding for image generation, and (iii) a cascaded flow matching head that decodes visual semantics first and then injects semantically gated detail residuals from the vision tokenizer to refine high-frequency content. Experiments on popular benchmarks demonstrate that Cheers matches or surpasses advanced UMMs in both visual understanding and generation. Cheers also achieves 4x token compression, enabling more efficient high-resolution image encoding and generation. Notably, Cheers outperforms the Tar-1.5B on the popular benchmarks GenEval and MMBench, while requiring only 20% of the training cost, indicating effective and efficient (i.e., 4x token compression) unified multimodal modeling. We will release all code and data for future research.
>
---
#### [new 091] Stake the Points: Structure-Faithful Instance Unlearning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器去学习任务，解决数据删除与知识保留的平衡问题。提出结构忠实框架，通过语义锚点保持知识结构，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12915](https://arxiv.org/pdf/2603.12915)**

> **作者:** Kiseong Hong; JungKyoo Shin; Eunwoo Kim
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Machine unlearning (MU) addresses privacy risks in pretrained models. The main goal of MU is to remove the influence of designated data while preserving the utility of retained knowledge. Achieving this goal requires preserving semantic relations among retained instances, which existing studies often overlook. We observe that without such preservation, models suffer from progressive structural collapse, undermining both the deletion-retention balance. In this work, we propose a novel structure-faithful framework that introduces stakes, i.e., semantic anchors that serve as reference points to maintain the knowledge structure. By leveraging these anchors, our framework captures and stabilizes the semantic organization of knowledge. Specifically, we instantiate the anchors from language-driven attribute descriptions encoded by a semantic encoder (e.g., CLIP). We enforce preservation of the knowledge structure via structure-aware alignment and regularization: the former aligns the organization of retained knowledge before and after unlearning around anchors, while the latter regulates updates to structure-critical parameters. Results from image classification, retrieval, and face recognition show average gains of 32.9%, 22.5%, and 19.3% in performance, balancing the deletion-retention trade-off and enhancing generalization.
>
---
#### [new 092] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
- **分类: cs.CV**

- **简介: 该论文属于酶动力学参数预测任务，解决传统方法忽略催化阶段问题。提出ERBA模型，通过多模态建模提升预测性能。**

- **链接: [https://arxiv.org/pdf/2603.12845](https://arxiv.org/pdf/2603.12845)**

> **作者:** Fei Wang; Xinye Zheng; Kun Li; Yanyan Wei; Yuxin Liu; Ganpeng Hu; Tong Bao; Jingwen Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Predicting enzyme kinetic parameters quantifies how efficiently an enzyme catalyzes a specific substrate under defined biochemical conditions. Canonical parameters such as the turnover number ($k_\text{cat}$), Michaelis constant ($K_\text{m}$), and inhibition constant ($K_\text{i}$) depend jointly on the enzyme sequence, the substrate chemistry, and the conformational adaptation of the active site during binding. Many learning pipelines simplify this process to a static compatibility problem between the enzyme and substrate, fusing their representations through shallow operations and regressing a single value. Such formulations overlook the staged nature of catalysis, which involves both substrate recognition and conformational adaptation. In this regard, we reformulate kinetic prediction as a staged multimodal conditional modeling problem and introduce the Enzyme-Reaction Bridging Adapter (ERBA), which injects cross-modal information via fine-tuning into Protein Language Models (PLMs) while preserving their biochemical priors. ERBA performs conditioning in two stages: Molecular Recognition Cross-Attention (MRCA) first injects substrate information into the enzyme representation to capture specificity; Geometry-aware Mixture-of-Experts (G-MoE) then integrates active-site structure and routes samples to pocket-specialized experts to reflect induced fit. To maintain semantic fidelity, Enzyme-Substrate Distribution Alignment (ESDA) enforces distributional consistency within the PLM manifold in a reproducing kernel Hilbert space. Experiments across three kinetic endpoints and multiple PLM backbones, ERBA delivers consistent gains and stronger out-of-distribution performance compared with sequence-only and shallow-fusion baselines, offering a biologically grounded route to scalable kinetic prediction and a foundation for adding cofactors, mutations, and time-resolved structural cues.
>
---
#### [new 093] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ESPIRE，用于评估视觉语言模型的具身空间推理能力。旨在解决现有评估方法覆盖不足与范式落后的问题，通过模拟环境进行任务分解与生成式评价。**

- **链接: [https://arxiv.org/pdf/2603.13033](https://arxiv.org/pdf/2603.13033)**

> **作者:** Yanpeng Zhao; Wentao Ding; Hongtao Li; Baoxiong Jia; Zilong Zheng
>
> **摘要:** A recent trend in vision-language models (VLMs) has been to enhance their spatial cognition for embodied domains. Despite progress, existing evaluations have been limited both in paradigm and in coverage, hindering rapid, iterative model development. To address these limitations, we propose ESPIRE, a diagnostic benchmark for embodied spatial reasoning. ESPIRE offers a simulated world that physically grounds VLMs and evaluates them on spatial-reasoning-centric robotic tasks, thus narrowing the gap between evaluation and real-world deployment. To adapt VLMs to robotic tasks, we decompose each task into localization and execution, and frame both as generative problems, in stark contrast to predominant discriminative evaluations (e.g., via visual-question answering) that rely on distractors and discard execution. This decomposition further enables a fine-grained analysis beyond passive spatial reasoning toward reasoning to act. We systematically design ESPIRE both at the instruction level and at the environment level, ensuring broad coverage of spatial reasoning scenarios. We use ESPIRE to diagnose a range of frontier VLMs and provide in-depth analysis of their spatial reasoning behaviors.
>
---
#### [new 094] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation
- **分类: cs.CV**

- **简介: 该论文属于动态场景编辑任务，解决4D场景中运动伪影和时序不一致问题。提出Catalyst4D框架，通过锚点运动引导和颜色不确定性优化，实现高质量的时序一致性编辑。**

- **链接: [https://arxiv.org/pdf/2603.12766](https://arxiv.org/pdf/2603.12766)**

> **作者:** Shifeng Chen; Yihui Li; Jun Liao; Hongyu Yang; Di Huang
>
> **备注:** this https URL
>
> **摘要:** Recent advances in 3D scene editing using NeRF and 3DGS enable high-quality static scene editing. In contrast, dynamic scene editing remains challenging, as methods that directly extend 2D diffusion models to 4D often produce motion artifacts, temporal flickering, and inconsistent style propagation. We introduce Catalyst4D, a framework that transfers high-quality 3D edits to dynamic 4D Gaussian scenes while maintaining spatial and temporal coherence. At its core, Anchor-based Motion Guidance (AMG) builds a set of structurally stable and spatially representative anchors from both original and edited Gaussians. These anchors serve as robust region-level references, and their correspondences are established via optimal transport to enable consistent deformation propagation without cross-region interference or motion drift. Complementarily, Color Uncertainty-guided Appearance Refinement (CUAR) preserves temporal appearance consistency by estimating per-Gaussian color uncertainty and selectively refining regions prone to occlusion-induced artifacts. Extensive experiments demonstrate that Catalyst4D achieves temporally stable, high-fidelity dynamic scene editing and outperforms existing methods in both visual quality and motion coherence.
>
---
#### [new 095] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.LG; cs.NE; stat.ML**

- **简介: 该论文属于文本到图像生成的后训练任务，旨在提升图像质量和提示对齐。提出一种基于有限差分的在线强化学习方法，通过优化采样过程提高模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12893](https://arxiv.org/pdf/2603.12893)**

> **作者:** David McAllister; Miika Aittala; Tero Karras; Janne Hellsten; Angjoo Kanazawa; Timo Aila; Samuli Laine
>
> **备注:** Code available at this https URL
>
> **摘要:** Reinforcement learning (RL) has become a standard technique for post-training diffusion-based image synthesis models, as it enables learning from reward signals to explicitly improve desirable aspects such as image quality and prompt alignment. In this paper, we propose an online RL variant that reduces the variance in the model updates by sampling paired trajectories and pulling the flow velocity in the direction of the more favorable image. Unlike existing methods that treat each sampling step as a separate policy action, we consider the entire sampling process as a single action. We experiment with both high-quality vision language models and off-the-shelf quality metrics for rewards, and evaluate the outputs using a broad set of metrics. Our method converges faster and yields higher output quality and prompt alignment than previous approaches.
>
---
#### [new 096] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态大语言模型在抽象时空推理上的不足。通过构建基准测试和数据集，评估模型在整合时空信息方面的能力。**

- **链接: [https://arxiv.org/pdf/2603.13091](https://arxiv.org/pdf/2603.13091)**

> **作者:** Seunghwan Bang; Hwanjun Song
>
> **备注:** 35 pages, 8 figures, 21 tables
>
> **摘要:** The growing interest in embodied agents increases the demand for spatiotemporal video understanding, yet existing benchmarks largely emphasize extractive reasoning, where answers can be explicitly presented within spatiotemporal events. It remains unclear whether multimodal large language models can instead perform abstractive spatiotemporal reasoning, which requires integrating observations over time, combining dispersed cues, and inferring implicit spatial and contextual structure. To address this gap, we formalize abstractive spatiotemporal reasoning from videos by introducing a structured evaluation taxonomy that systematically targets its core dimensions and construct a controllable, scenario-driven synthetic egocentric video dataset tailored to evaluate abstractive spatiotemporal reasoning capabilities, spanning object-, room-, and floor-plan-level scenarios. Based on this framework, we present VAEX-BENCH, a benchmark comprising five abstractive reasoning tasks together with their extractive counterparts. Our extensive experiments compare the performance of state-of-the-art MLLMs under extractive and abstractive settings, exposing their limitations on abstractive tasks and providing a fine-grained analysis of the underlying bottlenecks. The dataset will be released soon.
>
---
#### [new 097] Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception
- **分类: cs.CV**

- **简介: 该论文属于多模态流感知任务，解决实时性与计算效率问题。提出一种基于相关性的轻量级调度框架，减少延迟并提升感知效果。**

- **链接: [https://arxiv.org/pdf/2603.13176](https://arxiv.org/pdf/2603.13176)**

> **作者:** Dingcheng Huang; Xiaotong Zhang; Kamal Youcef-Toumi
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** In modern human-robot collaboration (HRC) applications, multiple perception modules jointly extract visual, auditory, and contextual cues to achieve comprehensive scene understanding, enabling the robot to provide appropriate assistance to human agents intelligently. While executing multiple perception modules on a frame-by-frame basis enhances perception quality in offline settings, it inevitably accumulates latency, leading to a substantial decline in system performance in streaming perception scenarios. Recent work in scene understanding, termed Relevance, has established a solid foundation for developing efficient methodologies in HRC. However, modern perception pipelines still face challenges related to information redundancy and suboptimal allocation of computational resources. Drawing inspiration from the Relevance concept and the information sparsity in HRC events, we propose a novel lightweight perception scheduling framework that efficiently leverages output from previous frames to estimate and schedule necessary perception modules in real-time based on scene context. The experimental results demonstrate that the proposed perception scheduling framework effectively reduces computational latency by up to 27.52% compared to conventional parallel perception pipelines, while also achieving a 72.73% improvement in MMPose activation recall. Additionally, the framework demonstrates high keyframe accuracy, achieving rates of up to 98%. The results validate the framework's capability to enhance real-time perception efficiency without significantly compromising accuracy. The framework shows potential as a scalable and systematic solution for multimodal streaming perception systems in HRC.
>
---
#### [new 098] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
- **分类: cs.CV**

- **简介: 该论文提出coDrawAgents框架，解决文本生成图像中的复杂场景组合问题。通过多智能体协作，提升图像布局准确性与属性一致性。**

- **链接: [https://arxiv.org/pdf/2603.12829](https://arxiv.org/pdf/2603.12829)**

> **作者:** Chunhan Li; Qifeng Wu; Jia-Hui Pan; Ka-Hei Hui; Jingyu Hu; Yuming Jiang; Bin Sheng; Xihui Liu; Wenjuan Gong; Zhengzhe Liu
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Text-to-image generation has advanced rapidly, but existing models still struggle with faithfully composing multiple objects and preserving their attributes in complex scenes. We propose coDrawAgents, an interactive multi-agent dialogue framework with four specialized agents: Interpreter, Planner, Checker, and Painter that collaborate to improve compositional generation. The Interpreter adaptively decides between a direct text-to-image pathway and a layout-aware multi-agent process. In the layout-aware mode, it parses the prompt into attribute-rich object descriptors, ranks them by semantic salience, and groups objects with the same semantic priority level for joint generation. Guided by the Interpreter, the Planner adopts a divide-and-conquer strategy, incrementally proposing layouts for objects with the same semantic priority level while grounding decisions in the evolving visual context of the canvas. The Checker introduces an explicit error-correction mechanism by validating spatial consistency and attribute alignment, and refining layouts before they are rendered. Finally, the Painter synthesizes the image step by step, incorporating newly planned objects into the canvas to provide richer context for subsequent iterations. Together, these agents address three key challenges: reducing layout complexity, grounding planning in visual context, and enabling explicit error correction. Extensive experiments on benchmarks GenEval and DPG-Bench demonstrate that coDrawAgents substantially improves text-image alignment, spatial accuracy, and attribute binding compared to existing methods.
>
---
#### [new 099] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于癫痫发作预测任务，旨在通过视频数据提前预测癫痫发作。针对标注数据不足，提出跨物种迁移学习方法，利用鼠类视频数据辅助训练，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.12887](https://arxiv.org/pdf/2603.12887)**

> **作者:** Mingkai Zhai; Wei Wang; Zongsheng Li; Quanying Liu
>
> **摘要:** Epileptic seizure forecasting is a clinically important yet challenging problem in epilepsy research. Existing approaches predominantly rely on neural signals such as electroencephalography (EEG), which require specialized equipment and limit long-term deployment in real-world settings. In contrast, video data provide a non-invasive and accessible alternative, yet existing video-based studies mainly focus on post-onset seizure detection, leaving seizure forecasting largely unexplored. In this work, we formulate a novel task of video-based epileptic seizure forecasting, where short pre-ictal video segments (3-10 seconds) are used to predict whether a seizure will occur within the subsequent 5 seconds. To address the scarcity of annotated human epilepsy videos, we propose a cross-species transfer learning framework that leverages large-scale rodent video data for auxiliary pretraining. This enables the model to capture seizure-related behavioral dynamics that generalize across species. Experimental results demonstrate that our approach achieves over 70% prediction accuracy under a strictly video-only setting and outperforms existing baselines. These findings highlight the potential of cross-species learning for building non-invasive, scalable early-warning systems for epilepsy.
>
---
#### [new 100] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization
- **分类: cs.CV**

- **简介: 该论文属于跨域泛化任务，旨在解决单源域泛化中的因果差异问题。通过引入领域一致界和融合人类知识的多模态方法，提升图像分类的跨域性能。**

- **链接: [https://arxiv.org/pdf/2603.12369](https://arxiv.org/pdf/2603.12369)**

> **作者:** Ayan Banerjee; Kuntal Thakur; Sandeep Gupta
>
> **摘要:** Generalizing image classification across domains remains challenging in critical tasks such as fundus image-based diabetic retinopathy (DR) grading and resting-state fMRI seizure onset zone (SOZ) detection. When domains differ in unknown causal factors, achieving cross-domain generalization is difficult, and there is no established methodology to objectively assess such differences without direct metadata or protocol-level information from data collectors, which is typically inaccessible. We first introduce domain conformal bounds (DCB), a theoretical framework to evaluate whether domains diverge in unknown causal factors. Building on this, we propose GenEval, a multimodal Vision Language Models (VLM) approach that combines foundational models (e.g., MedGemma-4B) with human knowledge via Low-Rank Adaptation (LoRA) to bridge causal gaps and enhance single-source domain generalization (SDG). Across eight DR and two SOZ datasets, GenEval achieves superior SDG performance, with average accuracy of 69.2% (DR) and 81% (SOZ), outperforming the strongest baselines by 9.4% and 1.8%, respectively.
>
---
#### [new 101] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型在遥感图像中的适应任务，旨在解决文本语义覆盖不足和视觉特征适应性差的问题。提出AVION框架，通过知识蒸馏和提示调优提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12659](https://arxiv.org/pdf/2603.12659)**

> **作者:** Yu Hu; Jianyang Gu; Hao Liu; Yue Cao; Jozsef Hamari; Zheng Liu; Mohsen Zardadi
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Adapting vision-language models to remote sensing imagery remains challenging due to two key factors: limited semantic coverage in textual representations and insufficient adaptability of visual features. These issues are particularly significant in aerial scenes, which involve various visual appearances and fine-grained object distinctions. We propose AVION, a knowledge distillation framework tailored for remote sensing adaptation of vision-language models. The teacher module constructs semantically rich textual prototypes by collecting descriptions from a large language model and verifying validity using remote sensing image features. The student module integrates lightweight and learnable prompts into both vision and language encoders, guided by the teacher to align embeddings and their cross-modal relationships. Once trained, the student operates independently during inference. Experiments on six optical remote sensing benchmarks show that AVION improves few-shot classification and base-class accuracy without degrading generalization to novel categories. It also enhances mean recall for cross-modal retrieval, with minimal additional trainable parameters.
>
---
#### [new 102] Rethinking VLMs for Image Forgery Detection and Localization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像伪造检测与定位任务，旨在解决AI生成内容带来的伪造图像识别难题。通过重新设计VLMs的使用方式，提升检测与定位效果。**

- **链接: [https://arxiv.org/pdf/2603.12930](https://arxiv.org/pdf/2603.12930)**

> **作者:** Shaofeng Guo; Jiequan Cui; Richang Hong
>
> **备注:** 8pages
>
> **摘要:** With the rapid rise of Artificial Intelligence Generated Content (AIGC), image manipulation has become increasingly accessible, posing significant challenges for image forgery detection and localization (IFDL). In this paper, we study how to fully leverage vision-language models (VLMs) to assist the IFDL task. In particular, we observe that priors from VLMs hardly benefit the detection and localization performance and even have negative effects due to their inherent biases toward semantic plausibility rather than authenticity. Additionally, the location masks explicitly encode the forgery concepts, which can serve as extra priors for VLMs to ease their training optimization, thus enhancing the interpretability of detection and localization results. Building on these findings, we propose a new IFDL pipeline named IFDL-VLM. To demonstrate the effectiveness of our method, we conduct experiments on 9 popular benchmarks and assess the model performance under both in-domain and cross-dataset generalization settings. The experimental results show that we consistently achieve new state-of-the-art performance in detection, localization, and this http URL is available at: this https URL.
>
---
#### [new 103] TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出TerraFlow，用于地球观测的多模态、多时相表征学习。解决 Earth observation 数据的时序建模与风险预测问题，通过序列感知学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12762](https://arxiv.org/pdf/2603.12762)**

> **作者:** Nazar Puriy; Johannes Jakubik; Benedikt Blumenstiel; Konrad Schindler
>
> **摘要:** We propose TerraFlow, a novel approach to multimodal, multitemporal learning for Earth observation. TerraFlow builds on temporal training objectives that enable sequence-aware learning across space, time, and modality, while remaining robust to the variable-length inputs commonly encountered in real-world Earth observation data. Our experiments demonstrate superiority of TerraFlow over state-of-the-art foundation models for Earth observation across all temporal tasks of the GEO-Bench-2 benchmark. We additionally demonstrate that TerraFlow is able to make initial steps towards deep-learning based risk map prediction for natural disasters -- a task on which other state-of-the-art foundation models frequently collapse. TerraFlow outperforms state-of-the-art foundation models by up to 50% in F1 score and 24% in Brier score.
>
---
#### [new 104] CVGL: Causal Learning and Geometric Topology
- **分类: cs.CV**

- **简介: 该论文聚焦于跨视角地理定位任务，解决因视角差异和干扰因素导致的定位难题。提出CLGT框架，结合因果学习和几何拓扑融合，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.12551](https://arxiv.org/pdf/2603.12551)**

> **作者:** Songsong Ouyang; Yingying Zhu
>
> **摘要:** Cross-view geo-localization (CVGL) aims to estimate the geographic location of a street image by matching it with a corresponding aerial image. This is critical for autonomous navigation and mapping in complex real-world scenarios. However, the task remains challenging due to significant viewpoint differences and the influence of confounding factors. To tackle these issues, we propose the Causal Learning and Geometric Topology (CLGT) framework, which integrates two key components: a Causal Feature Extractor (CFE) that mitigates the influence of confounding factors by leveraging causal intervention to encourage the model to focus on stable, task-relevant semantics; and a Geometric Topology Fusion (GT Fusion) module that injects Bird's Eye View (BEV) road topology into street features to alleviate cross-view inconsistencies caused by extreme perspective changes. Additionally, we introduce a Data-Adaptive Pooling (DA Pooling) module to enhance the representation of semantically rich regions. Extensive experiments on CVUSA, CVACT, and their robustness-enhanced variants (CVUSA-C-ALL and CVACT-C-ALL) demonstrate that CLGT achieves state-of-the-art performance, particularly under challenging real-world corruptions. Our codes are available at this https URL.
>
---
#### [new 105] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于面部情绪识别与暴力分类任务，提出一种基于预训练模型的快速方法，解决帧级情感分析和细粒度暴力检测问题。**

- **链接: [https://arxiv.org/pdf/2603.12693](https://arxiv.org/pdf/2603.12693)**

> **作者:** Andrey V. Savchenko; Kseniia Tsypliakova
>
> **备注:** to be submitted to ABAW-10 workshop of CVPR 2026
>
> **摘要:** This article presents our results for the 10th Affective Behavior Analysis in-the-Wild (ABAW) competition. For frame-wise facial emotion understanding tasks (frame-wise facial expression recognition, valence-arousal estimation, action unit detection), we propose a fast approach based on facial embedding extraction with pre-trained EfficientNet-based emotion recognition models. If the latter model's confidence exceeds a threshold, its prediction is used. Otherwise, we feed embeddings into a simple multi-layered perceptron trained on the AffWild2 dataset. Estimated class-level scores are smoothed in a sliding window of fixed size to mitigate noise in frame-wise predictions. For the fine-grained violence detection task, we examine several pre-trained architectures for frame embeddings and their aggregation for video classification. Experimental results on four tasks from the ABAW challenge demonstrate that our approach significantly improves validation metrics over existing baselines.
>
---
#### [new 106] Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks
- **分类: cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于深度学习模型压缩任务，旨在解决结构剪枝和动态路由中的功能路径丢失问题。提出AGF方法，通过特征空间展开捕捉结构效用，提升模型压缩效果。**

- **链接: [https://arxiv.org/pdf/2603.12354](https://arxiv.org/pdf/2603.12354)**

> **作者:** Tianhao Qian; Zhuoxuan Li; Jinde Cao; Xinli Shi; Hanjie Liu; Leszek Rutkowski
>
> **备注:** 11 pages, 6 figures, 9 tables
>
> **摘要:** Efficient deep learning traditionally relies on static heuristics like weight magnitude or activation awareness (e.g., Wanda, RIA). While successful in unstructured settings, we observe a critical limitation when applying these metrics to the structural pruning of deep vision networks. These contemporary metrics suffer from a magnitude bias, failing to preserve critical functional pathways. To overcome this, we propose a decoupled kinetic paradigm inspired by Alternating Gradient Flow (AGF), utilizing an absolute feature-space Taylor expansion to accurately capture the network's structural "kinetic utility". First, we uncover a topological phase transition at extreme sparsity, where AGF successfully preserves baseline functionality and exhibits topological implicit regularization, avoiding the collapse seen in models trained from scratch. Second, transitioning to architectures without strict structural priors, we reveal a phenomenon of Sparsity Bottleneck in Vision Transformers (ViTs). Through a gradient-magnitude decoupling analysis, we discover that dynamic signals suffer from signal compression in converged models, rendering them suboptimal for real-time routing. Finally, driven by these empirical constraints, we design a hybrid routing framework that decouples AGF-guided offline structural search from online execution via zero-cost physical priors. We validate our paradigm on large-scale benchmarks: under a 75% compression stress test on ImageNet-1K, AGF effectively avoids the structural collapse where traditional metrics aggressively fall below random sampling. Furthermore, when systematically deployed for dynamic inference on ImageNet-100, our hybrid approach achieves Pareto-optimal efficiency. It reduces the usage of the heavy expert by approximately 50% (achieving an estimated overall cost of 0.92$\times$) without sacrificing the full-model accuracy.
>
---
#### [new 107] SAP: Segment Any 4K Panorama
- **分类: cs.CV**

- **简介: 该论文提出SAP模型，解决4K全景图像实例分割问题。通过将全景分割转化为固定轨迹的视角视频分割，提升模型在全景图像上的性能。**

- **链接: [https://arxiv.org/pdf/2603.12759](https://arxiv.org/pdf/2603.12759)**

> **作者:** Lutao Jiang; Zidong Cao; Weikai Chen; Xu Zheng; Yuanhuiyi Lyu; Zhenyang Li; Zeyu HU; Yingda Yin; Keyang Luo; Runze Zhang; Kai Yan; Shengju Qian; Haidi Fan; Yifan Peng; Xin Wang; Hui Xiong; Ying-Cong Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Promptable instance segmentation is widely adopted in embodied and AR systems, yet the performance of foundation models trained on perspective imagery often degrades on 360° panoramas. In this paper, we introduce Segment Any 4K Panorama (SAP), a foundation model for 4K high-resolution panoramic instance-level segmentation. We reformulate panoramic segmentation as fixed-trajectory perspective video segmentation, decomposing a panorama into overlapping perspective patches sampled along a continuous spherical traversal. This memory-aligned reformulation preserves native 4K resolution while restoring the smooth viewpoint transitions required for stable cross-view propagation. To enable large-scale supervision, we synthesize 183,440 4K-resolution panoramic images with instance segmentation labels using the InfiniGen engine. Trained under this trajectory-aligned paradigm, SAP generalizes effectively to real-world 360° images, achieving +17.2 zero-shot mIoU gain over vanilla SAM2 of different sizes on real-world 4K panorama benchmark.
>
---
#### [new 108] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于虚拟免疫组化染色任务，解决从H&E图像生成IHC图像的问题。通过引入病理基础模型指导生成过程，提升染色准确性和多标记支持能力。**

- **链接: [https://arxiv.org/pdf/2603.12716](https://arxiv.org/pdf/2603.12716)**

> **作者:** Jillur Rahman Saurav; Thuong Le Hoai Pham; Pritam Mukherjee; Paul Yi; Brent A. Orr; Jacob M. Luber
>
> **摘要:** Virtual immunohistochemistry (IHC) staining from hematoxylin and eosin (H&E) images can accelerate diagnostics by providing preliminary molecular insight directly from routine sections, reducing the need for repeat sectioning when tissue is limited. Existing methods improve realism through contrastive objectives, prototype matching, or domain alignment, yet the generator itself receives no direct guidance from pathology foundation models. We present UNIStainNet, a SPADE-UNet conditioned on dense spatial tokens from a frozen pathology foundation model (UNI), providing tissue-level semantic guidance for stain translation. A misalignment-aware loss suite preserves stain quantification accuracy, and learned stain embeddings enable a single model to serve multiple IHC markers simultaneously. On MIST, UNIStainNet achieves state-of-the-art distributional metrics on all four stains (HER2, Ki67, ER, PR) from a single unified model, where prior methods typically train separate per-stain models. On BCI, it also achieves the best distributional metrics. A tissue-type stratified failure analysis reveals that remaining errors are systematic, concentrating in non-tumor tissue. Code is available at this https URL.
>
---
#### [new 109] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CMHANet，解决点云配准问题，通过融合2D图像与3D点云信息，提升复杂场景下的配准精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12721](https://arxiv.org/pdf/2603.12721)**

> **作者:** Dongxu Zhang; Yingsen Wang; Yiding Sun; Haoran Xu; Peilin Fan; Jihua Zhu
>
> **摘要:** Robust point cloud registration is a fundamental task in 3D computer vision and geometric deep learning, essential for applications such as large-scale 3D reconstruction, augmented reality, and scene understanding. However, the performance of established learning-based methods often degrades in complex, real world scenarios characterized by incomplete data, sensor noise, and low overlap regions. To address these limitations, we propose CMHANet, a novel Cross-Modal Hybrid Attention Network. Our method integrates the fusion of rich contextual information from 2D images with the geometric detail of 3D point clouds, yielding a comprehensive and resilient feature representation. Furthermore, we introduce an innovative optimization function based on contrastive learning, which enforces geometric consistency and significantly improves the model's robustness to noise and partial observations. We evaluated CMHANet on the 3DMatch and the challenging 3DLoMatch datasets. \rev{Additionally, zero-shot evaluations on the TUM RGB-D SLAM dataset verify the model's generalization capability to unseen domains.} The experimental results demonstrate that our method achieves substantial improvements in both registration accuracy and overall robustness, outperforming current techniques. We also release our code in \href{this https URL}{this https URL}.
>
---
#### [new 110] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于实例分割任务，旨在解决货运列车故障检测中的准确性和鲁棒性问题。提出轻量级自提示框架，提升检测效果并适应边缘部署。**

- **链接: [https://arxiv.org/pdf/2603.12624](https://arxiv.org/pdf/2603.12624)**

> **作者:** Guodong Sun; Qihang Liang; Xingyu Pan; Moyun Liu; Yang Zhang
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Accurate visual fault detection in freight trains remains a critical challenge for intelligent transportation system maintenance, due to complex operational environments, structurally repetitive components, and frequent occlusions or contaminations in safety-critical regions. Conventional instance segmentation methods based on convolutional neural networks and Transformers often suffer from poor generalization and limited boundary accuracy under such conditions. To address these challenges, we propose a lightweight self-prompted instance segmentation framework tailored for freight train fault detection. Our method leverages the Segment Anything Model by introducing a self-prompt generation module that automatically produces task-specific prompts, enabling effective knowledge transfer from foundation models to domain-specific inspection tasks. In addition, we adopt a Tiny Vision Transformer backbone to reduce computational cost, making the framework suitable for real-time deployment on edge devices in railway monitoring systems. We construct a domain-specific dataset collected from real-world freight inspection stations and conduct extensive evaluations. Experimental results show that our method achieves 74.6 $AP^{\text{box}}$ and 74.2 $AP^{\text{mask}}$ on the dataset, outperforming existing state-of-the-art methods in both accuracy and robustness while maintaining low computational overhead. This work offers a deployable and efficient vision solution for automated freight train inspection, demonstrating the potential of foundation model adaptation in industrial-scale fault diagnosis scenarios. Project page: this https URL
>
---
#### [new 111] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering
- **分类: cs.CV**

- **简介: 该论文属于手势引导的视角视频问答任务，旨在解决MLLMs在理解用户指向意图上的不足。通过构建EgoPointVQA数据集和HINT方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12533](https://arxiv.org/pdf/2603.12533)**

> **作者:** Yura Choi; Roy Miles; Rolandos Alexandros Potamias; Ismail Elezi; Jiankang Deng; Stefanos Zafeiriou
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Understanding and answering questions based on a user's pointing gesture is essential for next-generation egocentric AI assistants. However, current Multimodal Large Language Models (MLLMs) struggle with such tasks due to the lack of gesture-rich data and their limited ability to infer fine-grained pointing intent from egocentric video. To address this, we introduce EgoPointVQA, a dataset and benchmark for gesture-grounded egocentric question answering, comprising 4000 synthetic and 400 real-world videos across multiple deictic reasoning tasks. Built upon it, we further propose Hand Intent Tokens (HINT), which encodes tokens derived from 3D hand keypoints using an off-the-shelf reconstruction model and interleaves them with the model input to provide explicit spatial and temporal context for interpreting pointing intent. We show that our model outperforms others in different backbones and model sizes. In particular, HINT-14B achieves 68.1% accuracy, on average over 6 tasks, surpassing the state-of-the-art, InternVL3-14B, by 6.6%. To further facilitate the open research, we will release the code, model, and dataset. Project page: this https URL
>
---
#### [new 112] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决VLM在空间推理上的不足。通过控制实验分析图像编码器和位置编码对空间理解的影响。**

- **链接: [https://arxiv.org/pdf/2603.12545](https://arxiv.org/pdf/2603.12545)**

> **作者:** Nahid Alam; Leema Krishna Murali; Siddhant Bharadwaj; Patrick Liu; Timothy Chung; Drishti Sharma; Akshata A.; Kranthi Kiran; Wesley Tam; Bala Krishna S Vegesna
>
> **备注:** Accepted as a poster at ICLR 2026 workshop ICBINB
>
> **摘要:** Vision-language models (VLMs) have advanced rapidly, yet they still struggle with basic spatial reasoning. Despite strong performance on general benchmarks, modern VLMs remain brittle at understanding 2D spatial relationships such as relative position, layout, and counting. We argue that this failure is not merely a data problem, but is closely tied to dominant design choices in current VLM pipelines: reliance on CLIP-style image encoders and the flattening of images into 1D token sequences with 1D positional encoding. We present a controlled diagnostic study within the LLaVA framework to isolate how these choices affect spatial grounding. We evaluate frontier models and LLaVA variants on a suite of spatial benchmarks, comparing CLIP-based encoders against alternatives trained with denser or generative objectives, as well as variants augmented with 2D positional encoding. Our results show consistent spatial performance gaps across models, and indicate that encoder objectives and positional structure shape spatial behavior, but do not fully resolve it.
>
---
#### [new 113] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型的时空推理任务，旨在解决模型在动态4D世界中的感知与推理问题。通过构建Dyn-Bench基准，评估并提升模型的时空理解和动态物体定位能力。**

- **链接: [https://arxiv.org/pdf/2603.12746](https://arxiv.org/pdf/2603.12746)**

> **作者:** Yuzhi Huang; Kairun Wen; Rongxin Gao; Dongxuan Liu; Yibin Lou; Jie Wu; Jing Xu; Jian Zhang; Zheng Yang; Yunlong Lin; Chenxin Li; Panwang Pan; Junbin Lu; Jingyan Jiang; Xinghao Ding; Yue Huang; Zhi Wang
>
> **摘要:** Humans inhabit a physical 4D world where geometric structure and semantic content evolve over time, constituting a dynamic 4D reality (spatial with temporal dimension). While current Multimodal Large Language Models (MLLMs) excel in static visual understanding, can they also be adept at "thinking in dynamics", i.e., perceive, track and reason about spatio-temporal dynamics in evolving scenes? To systematically assess their spatio-temporal reasoning and localized dynamics perception capabilities, we introduce Dyn-Bench, a large-scale benchmark built from diverse real-world and synthetic video datasets, enabling robust and scalable evaluation of spatio-temporal understanding. Through multi-stage filtering from massive 2D and 4D data sources, Dyn-Bench provides a high-quality collection of dynamic scenes, comprising 1k videos, 7k visual question answering (VQA) pairs, and 3k dynamic object grounding pairs. We probe general, spatial and region-level MLLMs to express how they think in dynamics both linguistically and visually, and find that existing models cannot simultaneously maintain strong performance in both spatio-temporal reasoning and dynamic object grounding, often producing inconsistent interpretations of motion and interaction. Notably, conventional prompting strategies (e.g., chain-of-thought or caption-based hints) provide limited improvement, whereas structured integration approaches, including Mask-Guided Fusion and Spatio-Temporal Textual Cognitive Map (ST-TCM), significantly enhance MLLMs' dynamics perception and spatio-temporal reasoning in the physical 4D world. Code and benchmark are available at this https URL.
>
---
#### [new 114] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于无监督跨域图像检索任务，解决伪标签不准确和语义退化问题。提出TPSNet，结合文本和相位先验提升检索性能。**

- **链接: [https://arxiv.org/pdf/2603.12711](https://arxiv.org/pdf/2603.12711)**

> **作者:** Jing Yang; Hui Xue; Shipeng Zhu; Pengfei Fang
>
> **摘要:** This paper studies unsupervised cross-domain image retrieval (UCDIR), which aims to retrieve images of the same category across different domains without relying on labeled data. Existing methods typically utilize pseudo-labels, derived from clustering algorithms, as supervisory signals for intra-domain representation learning and cross-domain feature alignment. However, these discrete pseudo-labels often fail to provide accurate and comprehensive semantic guidance. Moreover, the alignment process frequently overlooks the entanglement between domain-specific and semantic information, leading to semantic degradation in the learned representations and ultimately impairing retrieval performance. This paper addresses the limitations by proposing a Text-Phase Synergy Network with Dual Priors(TPSNet). Specifically, we first employ CLIP to generate a set of class-specific prompts per domain, termed as domain prompt, serving as a text prior that offers more precise semantic supervision. In parallel, we further introduce a phase prior, represented by domain-invariant phase features, which is integrated into the original image representations to bridge the domain distribution gaps while preserving semantic integrity. Leveraging the synergy of these dual priors, TPSNet significantly outperforms state-of-the-art methods on UCDIR benchmarks.
>
---
#### [new 115] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云配准任务，旨在解决噪声、遮挡和大尺度变换下的配准精度与鲁棒性问题。提出IGASA框架，结合多尺度特征提取与几何感知优化，提升配准效果。**

- **链接: [https://arxiv.org/pdf/2603.12719](https://arxiv.org/pdf/2603.12719)**

> **作者:** Dongxu Zhang; Jihua Zhu; Shiqi Li; Wenbiao Yan; Haoran Xu; Peilin Fan; Huimin Lu
>
> **摘要:** Point cloud registration (PCR) is a fundamental task in 3D vision and provides essential support for applications such as autonomous driving, robotics, and environmental modeling. Despite its widespread use, existing methods often fail when facing real-world challenges like heavy noise, significant occlusions, and large-scale transformations. These limitations frequently result in compromised registration accuracy and insufficient robustness in complex environments. In this paper, we propose IGASA as a novel registration framework constructed upon a Hierarchical Pyramid Architecture (HPA) designed for robust multi-scale feature extraction and fusion. The framework integrates two pivotal components consisting of the Hierarchical Cross-Layer Attention (HCLA) module and the Iterative Geometry-Aware Refinement (IGAR) module. The HCLA module utilizes skip attention mechanisms to align multi-resolution features and enhance local geometric consistency. Simultaneously, the IGAR module is designed for the fine matching phase by leveraging reliable correspondences established during coarse matching. This synergistic integration within the architecture allows IGASA to adapt effectively to diverse point cloud structures and intricate transformations. We evaluate the performance of IGASA on four widely recognized benchmark datasets including 3D(Lo)Match, KITTI, and nuScenes. Our extensive experiments consistently demonstrate that IGASA significantly surpasses state-of-the-art methods and achieves notable improvements in registration accuracy. This work provides a robust foundation for advancing point cloud registration techniques while offering valuable insights for practical 3D vision applications. The code for IGASA is available in \href{this https URL}{this https URL}.
>
---
#### [new 116] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
- **分类: cs.LG; cs.CV; cs.IT; math.DS**

- **简介: 该论文将扩散模型视为分段迭代函数系统，分析其去噪过程中的几何特性，解决模型设计与优化问题，提出基于分形几何的理论框架。**

- **链接: [https://arxiv.org/pdf/2603.13069](https://arxiv.org/pdf/2603.13069)**

> **作者:** Ann Dooms
>
> **摘要:** What is a diffusion model actually doing when it turns noise into a photograph? We show that the deterministic DDIM reverse chain operates as a Partitioned Iterated Function System (PIFS) and that this framework serves as a unified design language for denoising diffusion model schedules, architectures, and training objectives. From the PIFS structure we derive three computable geometric quantities: a per-step contraction threshold $L^*_t$, a diagonal expansion function $f_t(\lambda)$ and a global expansion threshold $\lambda^{**}$. These quantities require no model evaluation and fully characterize the denoising dynamics. They structurally explain the two-regime behavior of diffusion models: global context assembly at high noise via diffuse cross-patch attention and fine-detail synthesis at low noise via patch-by-patch suppression release in strict variance order. Self-attention emerges as the natural primitive for PIFS contraction. The Kaplan-Yorke dimension of the PIFS attractor is determined analytically through a discrete Moran equation on the Lyapunov spectrum. Through the study of the fractal geometry of the PIFS, we derive three optimal design criteria and show that four prominent empirical design choices (the cosine schedule offset, resolution-dependent logSNR shift, Min-SNR loss weighting, and Align Your Steps sampling) each arise as approximate solutions to our explicit geometric optimization problems tuning theory into practice.
>
---
#### [new 117] Variational Garrote for Sparse Inverse Problems
- **分类: stat.ML; cs.CV; cs.LG**

- **简介: 该论文研究稀疏逆问题中的正则化方法，比较L1与变分Garrote（VG）的效果，旨在提升重建性能和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.12562](https://arxiv.org/pdf/2603.12562)**

> **作者:** Kanghun Lee; Hyungjoon Soh; Junghyo Jo
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Sparse regularization plays a central role in solving inverse problems arising from incomplete or corrupted measurements. Different regularizers correspond to different prior assumptions about the structure of the unknown signal, and reconstruction performance depends on how well these priors match the intrinsic sparsity of the data. This work investigates the effect of sparsity priors in inverse problems by comparing conventional L1 regularization with the Variational Garrote (VG), a probabilistic method that approximates L0 sparsity through variational binary gating variables. A unified experimental framework is constructed across multiple reconstruction tasks including signal resampling, signal denoising, and sparse-view computed tomography. To enable consistent comparison across models with different parameterizations, regularization strength is swept across wide ranges and reconstruction behavior is analyzed through train-generalization error curves. Experiments reveal characteristic bias-variance tradeoff patterns across tasks and demonstrate that VG frequently achieves lower minimum generalization error and improved stability in strongly underdetermined regimes where accurate support recovery is critical. These results suggest that sparsity priors closer to spike-and-slab structure can provide advantages when the underlying coefficient distribution is strongly sparse. The study highlights the importance of prior-data alignment in sparse inverse problems and provides empirical insights into the behavior of variational L0-type methods across different information bottlenecks.
>
---
#### [new 118] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文属于机器人环境感知任务，旨在解决四足机器人在复杂环境中精准预测三维占用问题。提出PanoMMOcc数据集和VoxelHound框架，提升视觉与多模态信息融合的准确性。**

- **链接: [https://arxiv.org/pdf/2603.13108](https://arxiv.org/pdf/2603.13108)**

> **作者:** Guoqiang Zhao; Zhe Yang; Sheng Wu; Fei Teng; Mengfei Duan; Yuanfan Zheng; Kai Luo; Kailun Yang
>
> **备注:** The dataset and code will be publicly released at this https URL
>
> **摘要:** Panoramic imagery provides holistic 360° visual coverage for perception in quadruped robots. However, existing occupancy prediction methods are mainly designed for wheeled autonomous driving and rely heavily on RGB cues, limiting their robustness in complex environments. To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes. (2) We propose a panoramic multimodal occupancy perception framework, VoxelHound, tailored for legged mobility and spherical imaging. Specifically, we design (i) a Vertical Jitter Compensation (VJC) module to mitigate severe viewpoint perturbations caused by body pitch and roll during mobility, enabling more consistent spatial reasoning, and (ii) an effective Multimodal Information Prompt Fusion (MIPF) module that jointly leverages panoramic visual cues and auxiliary modalities to enhance volumetric occupancy prediction. (3) We establish a benchmark based on PanoMMOcc and provide detailed data analysis to enable systematic evaluation of perception methods under challenging embodied scenarios. Extensive experiments demonstrate that VoxelHound achieves state-of-the-art performance on PanoMMOcc (+4.16%} in mIoU). The dataset and code will be publicly released to facilitate future research on panoramic multimodal 3D perception for embodied robotic systems at this https URL, along with the calibration tools released at this https URL.
>
---
#### [new 119] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于学习带噪声标签数据的任务，旨在解决噪声校正方法失效的问题。通过实验和分析，揭示噪声校正失败并非仅因估计误差，而是源于更深层的机制问题。**

- **链接: [https://arxiv.org/pdf/2603.12997](https://arxiv.org/pdf/2603.12997)**

> **作者:** Chen Feng; Zhuo Zhi; Zhao Huang; Jiawei Ge; Ling Xiao; Nicu Sebe; Georgios Tzimiropoulos; Ioannis Patras
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Statistically consistent methods based on the noise transition matrix ($T$) offer a theoretically grounded solution to Learning with Noisy Labels (LNL), with guarantees of convergence to the optimal clean-data classifier. In practice, however, these methods are often outperformed by empirical approaches such as sample selection, and this gap is usually attributed to the difficulty of accurately estimating $T$. The common assumption is that, given a perfect $T$, noise-correction methods would recover their theoretical advantage. In this work, we put this longstanding hypothesis to a decisive test. We conduct experiments under idealized conditions, providing correction methods with a perfect, oracle transition matrix. Even under these ideal conditions, we observe that these methods still suffer from performance collapse during training. This compellingly demonstrates that the failure is not fundamentally a $T$-estimation problem, but stems from a more deeply rooted flaw. To explain this behaviour, we provide a unified analysis that links three levels: macroscopic convergence states, microscopic optimisation dynamics, and information-theoretic limits on what can be learned from noisy labels. Together, these results give a formal account of why ideal noise correction fails and offer concrete guidance for designing more reliable methods for learning with noisy labels.
>
---
#### [new 120] Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决长距离计划漂移和视觉预测冗余问题。提出StructVLA，通过预测物理有意义的结构帧实现可靠控制。**

- **链接: [https://arxiv.org/pdf/2603.12553](https://arxiv.org/pdf/2603.12553)**

> **作者:** Minghao Jin; Mozheng Liao; Mingfei Han; Zhihui Li; Xiaojun Chang
>
> **摘要:** Recent world-model-based Vision-Language-Action (VLA) architectures have improved robotic manipulation through predictive visual foresight. However, dense future prediction introduces visual redundancy and accumulates errors, causing long-horizon plan drift. Meanwhile, recent sparse methods typically represent visual foresight using high-level semantic subtasks or implicit latent states. These representations often lack explicit kinematic grounding, weakening the alignment between planning and low-level execution. To address this, we propose StructVLA, which reformulates a generative world model into an explicit structured planner for reliable control. Instead of dense rollouts or semantic goals, StructVLA predicts sparse, physically meaningful structured frames. Derived from intrinsic kinematic cues (e.g., gripper transitions and kinematic turning points), these frames capture spatiotemporal milestones closely aligned with task progress. We implement this approach through a two-stage training paradigm with a unified discrete token vocabulary: the world model is first trained to predict structured frames and subsequently optimized to map the structured foresight into low-level actions. This approach provides clear physical guidance and bridges visual planning and motion control. In our experiments, StructVLA achieves strong average success rates of 75.0% on SimplerEnv-WidowX and 94.8% on LIBERO. Real-world deployments further demonstrate reliable task completion and robust generalization across both basic pick-and-place and complex long-horizon tasks.
>
---
#### [new 121] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SldprtNet，一个用于语义驱动CAD生成的多模态数据集，解决3D设计中多模态数据不足的问题，通过整合3D模型、文本和图像，提升CAD建模效果。**

- **链接: [https://arxiv.org/pdf/2603.13098](https://arxiv.org/pdf/2603.13098)**

> **作者:** Ruogu Li; Sikai Li; Yao Mu; Mingyu Ding
>
> **备注:** Accept by ICRA 2026
>
> **摘要:** We introduce SldprtNet, a large-scale dataset comprising over 242,000 industrial parts, designed for semantic-driven CAD modeling, geometric deep learning, and the training and fine-tuning of multimodal models for 3D design. The dataset provides 3D models in both .step and .sldprt formats to support diverse training and testing. To enable parametric modeling and facilitate dataset scalability, we developed supporting tools, an encoder and a decoder, which support 13 types of CAD commands and enable lossless transformation between 3D models and a structured text representation. Additionally, each sample is paired with a composite image created by merging seven rendered views from different viewpoints of the 3D model, effectively reducing input token length and accelerating inference. By combining this image with the parameterized text output from the encoder, we employ the lightweight multimodal language model Qwen2.5-VL-7B to generate a natural language description of each part's appearance and functionality. To ensure accuracy, we manually verified and aligned the generated descriptions, rendered images, and 3D models. These descriptions, along with the parameterized modeling scripts, rendered images, and 3D model files, are fully aligned to construct SldprtNet. To assess its effectiveness, we fine-tuned baseline models on a dataset subset, comparing image-plus-text inputs with text-only inputs. Results confirm the necessity and value of multimodal datasets for CAD generation. It features carefully selected real-world industrial parts, supporting tools for scalable dataset expansion, diverse modalities, and ensured diversity in model complexity and geometric features, making it a comprehensive multimodal dataset built for semantic-driven CAD modeling and cross-modal learning.
>
---
#### [new 122] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
- **分类: cs.LG; cs.CV; math.NA; stat.ML**

- **简介: 该论文研究注意力机制的理论特性，探讨其非收敛的NTK动态。工作揭示了线性化注意力的收敛难题，分析其对模型性能和安全的影响。任务属于深度学习理论分析。**

- **链接: [https://arxiv.org/pdf/2603.13085](https://arxiv.org/pdf/2603.13085)**

> **作者:** Jose Marie Antonio Miñoza; Paulo Mario P. Medina; Sebastian C. Ibañez
>
> **摘要:** Understanding the theoretical foundations of attention mechanisms remains challenging due to their complex, non-linear dynamics. This work reveals a fundamental trade-off in the learning dynamics of linearized attention. Using a linearized attention mechanism with exact correspondence to a data-dependent Gram-induced kernel, both empirical and theoretical analysis through the Neural Tangent Kernel (NTK) framework shows that linearized attention does not converge to its infinite-width NTK limit, even at large widths. A spectral amplification result establishes this formally: the attention transformation cubes the Gram matrix's condition number, requiring width $m = \Omega(\kappa^6)$ for convergence, a threshold that exceeds any practical width for natural image datasets. This non-convergence is characterized through influence malleability, the capacity to dynamically alter reliance on training examples. Attention exhibits 6--9$\times$ higher malleability than ReLU networks, with dual implications: its data-dependent kernel can reduce approximation error by aligning with task structure, but this same sensitivity increases susceptibility to adversarial manipulation of training data. These findings suggest that attention's power and vulnerability share a common origin in its departure from the kernel regime.
>
---
#### [new 123] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval
- **分类: cs.IR; cs.CV; cs.LG**

- **简介: 该论文属于视觉文档检索任务，解决VLM检索器高延迟和GPU依赖问题。通过知识蒸馏，将大模型压缩为小文本编码器，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.12824](https://arxiv.org/pdf/2603.12824)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** Vision-Language Model (VLM) based retrievers have advanced visual document retrieval (VDR) to impressive quality. They require the same multi-billion parameter encoder for both document indexing and query encoding, incurring high latency and GPU dependence even for plain-text queries. We observe that this design is unnecessarily symmetric: documents are visually complex and demand strong visual understanding, whereas queries are just short text strings. NanoVDR exploits this query--document asymmetry by decoupling the two encoding paths: a frozen 2B VLM teacher indexes documents offline, while a distilled text-only student as small as 69M parameters encodes queries at inference. The key design choice is the distillation objective. Through systematic comparison of six objectives across three backbones and 22 ViDoRe benchmark datasets, we find that pointwise cosine alignment on query text consistently outperforms ranking-based and contrastive alternatives, while requiring only pre-cached teacher query embeddings and no document processing during training. Furthermore, we identify cross-lingual transfer as the primary performance bottleneck, and resolve it cheaply by augmenting training data with machine-translated queries. The resulting NanoVDR-S-Multi (DistilBERT, 69M) retains 95.1\% of teacher quality and outperforms DSE-Qwen2 (2B) on v2 and v3 with 32$\times$ fewer parameters and 50$\times$ lower CPU query latency, at a total training cost under 13 GPU-hours.
>
---
#### [new 124] Generation of maximal snake polyominoes using a deep neural network
- **分类: math.CO; cs.CV**

- **简介: 该论文属于生成任务，旨在解决大矩形中最大蛇形多米诺的生成问题。通过深度神经网络，无需显式编码约束即可生成有效蛇形结构。**

- **链接: [https://arxiv.org/pdf/2603.12400](https://arxiv.org/pdf/2603.12400)**

> **作者:** Benjamin Gauthier; Alain Goupil; Fadel Toure
>
> **备注:** 8-page extended abstract, plus 2 pages of references; 6 figures. Submitted to GASCom 2026
>
> **摘要:** Maximal snake polyominoes are difficult to study numerically in large rectangles, as computing them requires the complete enumeration of all snakes for a specific grid size, which corresponds to a brute force algorithm. This technique is thus challenging to use in larger rectangles, which hinders the study of maximal snakes. Furthermore, most enumerable snakes lie in small rectangles, making it difficult to study large-scale patterns. In this paper, we investigate the contribution of a deep neural network to the generation of maximal snake polyominoes from a data-driven training, where the maximality and adjacency constraints are not encoded explicitly, but learned. To this extent, we experiment with a denoising diffusion model, which we call Structured Pixel Space Diffusion (SPS Diffusion). We find that SPS Diffusion generalizes from small grids to larger ones, generating valid snakes up to 28x28 squares and producing maximal snake candidates on squares close to the current computational limit. The model is, however, prone to errors such as branching, cycles, or multiple components. Overall, the diffusion model is promising and shows that complex combinatorial objects can be understood by deep neural networks, which is useful in their investigation.
>
---
#### [new 125] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于持续学习任务，解决领域增量学习中的灾难性遗忘问题。通过结合提示适应与知识保留，提出Residual SODAP框架，提升模型在无任务标识情况下的性能。**

- **链接: [https://arxiv.org/pdf/2603.12816](https://arxiv.org/pdf/2603.12816)**

> **作者:** Gyutae Oh; Jungwoo Bae; Jitae Shin
>
> **备注:** 29 page, 10 figures
>
> **摘要:** Continual learning (CL) suffers from catastrophic forgetting, which is exacerbated in domain-incremental learning (DIL) where task identifiers are unavailable and storing past data is infeasible. While prompt-based CL (PCL) adapts representations with a frozen backbone, we observe that prompt-only improvements are often insufficient due to suboptimal prompt selection and classifier-level instability under domain shifts. We propose Residual SODAP, which jointly performs prompt-based representation adaptation and classifier-level knowledge preservation. Our framework combines $\alpha$-entmax sparse prompt selection with residual aggregation, data-free distillation with pseudo-feature replay, prompt-usage--based drift detection, and uncertainty-aware multi-loss balancing. Across three DIL benchmarks without task IDs or extra data storage, Residual SODAP achieves state-of-the-art AvgACC/AvgF of 0.850/0.047 (DR), 0.760/0.031 (Skin Cancer), and 0.995/0.003 (CORe50).
>
---
#### [new 126] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于人形机器人运动生成任务，解决扩散模型生成动作与物理合规性及文本指令不符的问题。通过集成全身控制器并优化模型，提升运动的物理真实性和任务准确性。**

- **链接: [https://arxiv.org/pdf/2603.13228](https://arxiv.org/pdf/2603.13228)**

> **作者:** Yangsong Zhang; Anujith Muraleedharan; Rikhat Akizhanov; Abdul Ahad Butt; Gül Varol; Pascal Fua; Fabio Pizzati; Ivan Laptev
>
> **摘要:** Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to transfer such models for character animation and real robot control by applying a Whole-Body Controller (WBC) that converts diffusion-generated motions into executable trajectories. While WBC trajectories become compliant with physics, they may expose substantial deviations from original motion. To address this issue, we here propose PhysMoDPO, a Direct Preference Optimization framework. Unlike prior work that relies on hand-crafted physics-aware heuristics such as foot-sliding penalties, we integrate WBC into our training pipeline and optimize diffusion model such that the output of WBC becomes compliant both with physics and original text instructions. To train PhysMoDPO we deploy physics-based and task-specific rewards and use them to assign preference to synthesized trajectories. Our extensive experiments on text-to-motion and spatial control tasks demonstrate consistent improvements of PhysMoDPO in both physical realism and task-related metrics on simulated robots. Moreover, we demonstrate that PhysMoDPO results in significant improvements when applied to zero-shot motion transfer in simulation and for real-world deployment on a G1 humanoid robot.
>
---
#### [new 127] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于多模态MRI翻译任务，旨在解决任意缺失模态下的结构不一致和纹理细节丢失问题。提出MSG-LDM框架，通过结构-风格解耦和多尺度特征建模提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.12581](https://arxiv.org/pdf/2603.12581)**

> **作者:** Jianqiang Lin; Zhiqiang Shen; Peng Cao; Jinzhu Yang; Osmar R. Zaiane; Xiaoli Liu
>
> **摘要:** Although diffusion models have achieved remarkable progress in multi-modal magnetic resonance imaging (MRI) translation tasks, existing methods still tend to suffer from anatomical inconsistencies or degraded texture details when handling arbitrary missing-modality scenarios. To address these issues, we propose a latent diffusion-based multi-modal MRI translation framework, termed MSG-LDM. By leveraging the available modalities, the proposed method infers complete structural information, which preserves reliable boundary details. Specifically, we introduce a style--structure disentanglement mechanism in the latent space, which explicitly separates modality-specific style features from shared structural representations, and jointly models low-frequency anatomical layouts and high-frequency boundary details in a multi-scale feature space. During the structure disentanglement stage, high-frequency structural information is explicitly incorporated to enhance feature representations, guiding the model to focus on fine-grained structural cues while learning modality-invariant low-frequency anatomical representations. Furthermore, to reduce interference from modality-specific styles and improve the stability of structure representations, we design a style consistency loss and a structure-aware loss. Extensive experiments on the BraTS2020 and WMH datasets demonstrate that the proposed method outperforms existing MRI synthesis approaches, particularly in reconstructing complete structures. The source code is publicly available at this https URL.
>
---
#### [new 128] Representation Learning for Spatiotemporal Physical Systems
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究物理系统的表征学习，旨在提升下游科学任务的性能。解决传统方法计算成本高、误差累积的问题，通过自监督学习获取物理相关的表征。**

- **链接: [https://arxiv.org/pdf/2603.13227](https://arxiv.org/pdf/2603.13227)**

> **作者:** Helen Qu; Rudy Morel; Michael McCabe; Alberto Bietti; François Lanusse; Shirley Ho; Yann LeCun
>
> **备注:** Published at ICLR 2026 Workshop on AI & PDE
>
> **摘要:** Machine learning approaches to spatiotemporal physical systems have primarily focused on next-frame prediction, with the goal of learning an accurate emulator for the system's evolution in time. However, these emulators are computationally expensive to train and are subject to performance pitfalls, such as compounding errors during autoregressive rollout. In this work, we take a different perspective and look at scientific tasks further downstream of predicting the next frame, such as estimation of a system's governing physical parameters. Accuracy on these tasks offers a uniquely quantifiable glimpse into the physical relevance of the representations of these models. We evaluate the effectiveness of general-purpose self-supervised methods in learning physics-grounded representations that are useful for downstream scientific tasks. Surprisingly, we find that not all methods designed for physical modeling outperform generic self-supervised learning methods on these tasks, and methods that learn in the latent space (e.g., joint embedding predictive architectures, or JEPAs) outperform those optimizing pixel-level prediction objectives. Code is available at this https URL.
>
---
#### [new 129] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出EPT，解决多任务中专家架构单一的问题，通过分层特征金字塔提升参数效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.12577](https://arxiv.org/pdf/2603.12577)**

> **作者:** Jia-Chen Zhang; Zhen-Wei Yan; Yu-Jie Xiong; Chun-Ming Xia
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has become a dominant paradigm for deploying LLMs in multi-task scenarios due to its extreme parameter efficiency. While Mixture-of-Experts (MoE) based LoRA variants have achieved promising results by dynamically routing tokens to different low-rank experts, they largely overlook the hierarchical nature of task complexity. Existing methods typically employ experts with uniform architectures, limiting their ability to capture diverse feature granularities required by distinct tasks--where some tasks demand high-level semantic abstraction while others require fine-grained syntactic manipulation. To bridge this gap, we propose Expert Pyramid Tuning (EPT), a novel architecture that integrates the multi-scale feature pyramid concept from computer vision into the realm of PEFT. Unlike standard LoRA, EPT decomposes task adaptation into two stages: (1) A shared meta-knowledge Subspace that encodes universal linguistic patterns in low dimensions; (2) A Pyramid Projection Mechanism that utilizes learnable up-projection operators to reconstruct high-dimensional features at varying scales. A task-aware router then dynamically selects the optimal combination of these multi-scale features. Extensive experiments across multiple multi-task benchmarks demonstrate that EPT significantly outperforms SOTA MoE-LoRA variants. Crucially, thanks to the re-parameterization capability of our design, EPT achieves this performance improvement while simultaneously reducing the number of training parameters.
>
---
#### [new 130] Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于糖尿病血糖监测任务，旨在通过眼白血管图像非侵入性估计血糖水平。研究提出ScleraGluNet模型，实现血糖分类与连续值预测。**

- **链接: [https://arxiv.org/pdf/2603.12715](https://arxiv.org/pdf/2603.12715)**

> **作者:** Muhammad Ahmed Khan; Manqiang Peng; Ding Lin; Saif Ur Rehman Khan
>
> **摘要:** Regular monitoring of glycemic status is essential for diabetes management, yet conventional blood-based testing can be burdensome for frequent assessment. The sclera contains superficial microvasculature that may exhibit diabetes related alterations and is readily visible on the ocular surface. We propose ScleraGluNet, a multiview deep-learning framework for three-class metabolic status classification (normal, controlled diabetes, and high-glucose diabetes) and continuous fasting plasma glucose (FPG) estimation from multidirectional scleral vessel images. The dataset comprised 445 participants (150/140/155) and 2,225 anterior-segment images acquired from five gaze directions per participant. After vascular enhancement, features were extracted using parallel convolutional branches, refined with Manta Ray Foraging Optimization (MRFO), and fused via transformer-based cross-view attention. Performance was evaluated using subject-wise five-fold cross-validation, with all images from each participant assigned to the same fold. ScleraGluNet achieved 93.8% overall accuracy, with one-vs-rest AUCs of 0.971,0.956, and 0.982 for normal, controlled diabetes, and high-glucose diabetes, respectively. For FPG estimation, the model achieved MAE = 6.42 mg/dL and RMSE = 7.91 mg/dL, with strong correlation to laboratory measurements (r = 0.983; R2 = 0.966). Bland Altman analysis showed a mean bias of +1.45 mg/dL with 95% limits of agreement from -8.33 to +11.23$ mg/dL. These results support multidirectional scleral vessel imaging with multiview learning as a promising noninvasive approach for glycemic assessment, warranting multicenter validation before clinical deployment.
>
---
#### [new 131] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning
- **分类: eess.IV; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文属于MRI加速重建任务，旨在解决数据有限情况下快速扫描的问题。通过大模型预训练和目标微调，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.13007](https://arxiv.org/pdf/2603.13007)**

> **作者:** Yamin Arefeen; Sidharth Kumar; Steven Warach; Hamidreza Saber; Jonathan Tamir
>
> **摘要:** Purpose: To develop a data-efficient strategy for accelerated MRI reconstruction with Diffusion Probabilistic Generative Models (DPMs) that enables faster scan times in clinical stroke MRI when only limited fully-sampled data samples are available. Methods: Our simple training strategy, inspired by the foundation model paradigm, first trains a DPM on a large, diverse collection of publicly available brain MRI data in fastMRI and then fine-tunes on a small dataset from the target application using carefully selected learning rates and fine-tuning durations. The approach is evaluated on controlled fastMRI experiments and on clinical stroke MRI data with a blinded clinical reader study. Results: DPMs pre-trained on approximately 4000 subjects with non-FLAIR contrasts and fine-tuned on FLAIR data from only 20 target subjects achieve reconstruction performance comparable to models trained with substantially more target-domain FLAIR data across multiple acceleration factors. Experiments reveal that moderate fine-tuning with a reduced learning rate yields improved performance, while insufficient or excessive fine-tuning degrades reconstruction quality. When applied to clinical stroke MRI, a blinded reader study involving two neuroradiologists indicates that images reconstructed using the proposed approach from $2 \times$ accelerated data are non-inferior to standard-of-care in terms of image quality and structural delineation. Conclusion: Large-scale pre-training combined with targeted fine-tuning enables DPM-based MRI reconstruction in data-constrained, accelerated clinical stroke MRI. The proposed approach substantially reduces the need for large application-specific datasets while maintaining clinically acceptable image quality, supporting the use of foundation-inspired diffusion models for accelerated MRI in targeted applications.
>
---
#### [new 132] Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation
- **分类: cs.AI; cs.CV; cs.IR; cs.MM**

- **简介: 该论文提出CRYSTAL基准，用于评估多模态推理的透明性，解决传统评估方法无法检测推理过程问题。通过引入新指标和奖励机制提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2603.13099](https://arxiv.org/pdf/2603.13099)**

> **作者:** Wayner Barrios; SouYoung Jin
>
> **摘要:** We introduce **CRYSTAL** (*__C__lear __R__easoning via __Y__ielded __S__teps, __T__raceability and __L__ogic*), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: *Match F1*, which scores step-level precision and recall via semantic similarity matching, and *Ordered Match F1*, which further penalizes disordered reasoning chains. References are constructed through a Delphi-inspired pipeline where four independent MLLMs generate trajectories, aggregated via semantic clustering and validated through human quality gates. Evaluation of 20 MLLMs, including commercial frontier systems not used during benchmark construction, reveals systematic failures invisible to accuracy: universal cherry-picking (precision far exceeds recall), non-monotonic scaling trade-offs, and disordered reasoning where no competitive model preserves more than 60% of matched steps in correct order. Beyond evaluation, we propose the **Causal Process Reward (CPR)**, a multiplicative reward that couples answer correctness with step-level alignment, and **CPR-Curriculum**, which progressively increases reasoning difficulty during training. CPR-Curriculum achieves +32% Match F1 via GRPO where additive reward strategies fail, improving reasoning without manual step annotation.
>
---
#### [new 133] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于神经影像分析任务，旨在提升SIENA在脑萎缩评估中的准确性与鲁棒性。通过引入深度学习模块替代传统图像处理步骤，改进了SIENA的性能。**

- **链接: [https://arxiv.org/pdf/2603.12951](https://arxiv.org/pdf/2603.12951)**

> **作者:** Riccardo Raciti; Lemuel Puglisi; Francesco Guarnera; Daniele Ravì; Sebastiano Battiato
>
> **摘要:** Percentage Brain Volume Change (PBVC) derived from Magnetic Resonance Imaging (MRI) is a widely used biomarker of brain atrophy, with SIENA among the most established methods for its estimation. However, SIENA relies on classical image processing steps, particularly skull stripping and tissue segmentation, whose failures can propagate through the pipeline and bias atrophy estimates. In this work, we examine whether targeted deep learning substitutions can improve SIENA while preserving its established and interpretable framework. To this end, we integrate SynthStrip and SynthSeg into SIENA and evaluate three pipeline variants on the ADNI and PPMI longitudinal cohorts. Performance is assessed using three complementary criteria: correlation with longitudinal clinical and structural decline, scan-order consistency, and end-to-end runtime. Replacing the skull-stripping module yields the most consistent gains: in ADNI, it substantially strengthens associations between PBVC and multiple measures of disease progression relative to the standard SIENA pipeline, while across both datasets it markedly improves robustness under scan reversal. The fully integrated pipeline achieves the strongest scan-order consistency, reducing the error by up to 99.1%. In addition, GPU-enabled variants reduce execution time by up to 46% while maintaining CPU runtimes comparable to standard SIENA. Overall, these findings show that deep learning can meaningfully strengthen established longitudinal atrophy pipelines when used to reinforce their weakest image processing steps. More broadly, this study highlights the value of modularly modernizing clinically trusted neuroimaging tools without sacrificing their interpretability. Code is publicly available at this https URL.
>
---
#### [new 134] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出GLEAM数据集和HAMM方法，用于青光眼分类任务。旨在解决多模态数据融合问题，通过整合眼底图像、OCT和视野图实现精准诊断。**

- **链接: [https://arxiv.org/pdf/2603.12800](https://arxiv.org/pdf/2603.12800)**

> **作者:** Jiao Wang; Chi Liu; Yiying Zhang; Hongchen Luo; Zhifen Guo; Ying Hu; Ke Xu; Jing Zhou; Hongyan Xu; Ruiting Zhou; Man Tang
>
> **摘要:** We propose glaucoma lesion evaluation and analysis with multimodal imaging (GLEAM), the first publicly available tri-modal glaucoma dataset comprising scanning laser ophthalmoscopy fundus images, circumpapillary OCT images, and visual field pattern deviation maps, annotated with four disease stages, enabling effective exploitation of multimodal complementary information and facilitating accurate diagnosis and treatment across disease stages. To effectively integrate cross-modal information, we propose hierarchical attentive masked modeling (HAMM) for multimodal glaucoma classification. Our framework employs hierarchical attentive encoders and light decoders to focus cross-modal representation learning on the encoder.
>
---
#### [new 135] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，探讨CNN在癌症病理图像中的偏差与可靠性问题。通过对比模型在含临床信息与无信息数据集上的表现，揭示现有评估方法的不可靠性。**

- **链接: [https://arxiv.org/pdf/2603.12445](https://arxiv.org/pdf/2603.12445)**

> **作者:** Michael Okonoda; Eder Martinez; Abhilekha Dalal; Lior Shamir
>
> **备注:** Electronics, published
>
> **摘要:** Convolutional Neural Networks have shown promising effectiveness in identifying different types of cancer from radiographs. However, the opaque nature of CNNs makes it difficult to fully understand the way they operate, limiting their assessment to empirical evaluation. Here we study the soundness of the standard practices by which CNNs are evaluated for the purpose of cancer pathology. Thirteen highly used cancer benchmark datasets were analyzed, using four common CNN architectures and different types of cancer, such as melanoma, carcinoma, colorectal cancer, and lung cancer. We compared the accuracy of each model with that of datasets made of cropped segments from the background of the original images that do not contain clinically relevant content. Because the rendered datasets contain no clinical information, the null hypothesis is that the CNNs should provide mere chance-based accuracy when classifying these datasets. The results show that the CNN models provided high accuracy when using the cropped segments, sometimes as high as 93\%, even though they lacked biomedical information. These results show that some CNN architectures are more sensitive to bias than others. The analysis shows that the common practices of machine learning evaluation might lead to unreliable results when applied to cancer pathology. These biases are very difficult to identify, and might mislead researchers as they use available benchmark datasets to test the efficacy of CNN methods.
>
---
#### [new 136] HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决环境变化下的导航鲁棒性问题。提出HaltNav框架，结合轻量拓扑地图与视觉停顿机制，提升长期导航的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.12696](https://arxiv.org/pdf/2603.12696)**

> **作者:** Pingcong Li; Zihui Yu; Bichi Zhang; Sören Schwertfeger
>
> **摘要:** Vision-and-Language Navigation (VLN) is shifting from rigid, step-by-step instruction following toward open-vocabulary, goal-oriented autonomy. Achieving this transition without exhaustive routing prompts requires agents to leverage structural priors. While prior work often assumes computationally heavy 2D/3D metric maps, we instead exploit a lightweight, text-based osmAG (OpenStreetMap Area Graph), a floorplan-level topological representation that is easy to obtain and maintain. However, global planning over a prior map alone is brittle in real-world deployments, where local connectivity can change (e.g., closed doors or crowded passages), leading to execution-time failures. To address this gap, we propose a hierarchical navigation framework HaltNav that couples the robust global planning of osmAG with the local exploration and instruction-grounding capability of VLN. Our approach features an MLLM-based brain module, which is capable of high-level task grounding and obstruction awareness. Conditioned on osmAG, the brain converts the global route into a sequence of localized execution snippets, providing the VLN executor with prior-grounded, goal-centric sub-instructions. Meanwhile, it detects local anomalies via a mechanism we term Reactive Visual Halting (RVH), which interrupts the local control loop, updates osmAG by invalidating the corresponding topology, and triggers replanning to orchestrate a viable detour. To train this halting capability efficiently, we introduce a data synthesis pipeline that leverages generative models to inject realistic obstacles into otherwise navigable scenes, substantially enriching hard negative samples. Extensive experiments demonstrate that our hierarchical framework outperforms several baseline methods without tedious language instructions, and significantly improves robustness for long-horizon vision-language navigation under environmental changes.
>
---
#### [new 137] Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于计算机视觉领域，解决Steerable CNN中核函数设计问题。通过构建满足对称性的基函数，避免计算Clebsch-Gordan系数，简化了等变卷积网络的设计。**

- **链接: [https://arxiv.org/pdf/2603.12459](https://arxiv.org/pdf/2603.12459)**

> **作者:** Alan Garbarz
>
> **备注:** 28 pages. Comments are welcome
>
> **摘要:** We present an alternative way of solving the steerable kernel constraint that appears in the design of steerable equivariant convolutional neural networks. We find explicit real and complex bases which are ready to use, for different symmetry groups and for feature maps of arbitrary tensor type. A major advantage of this method is that it bypasses the need to numerically or analytically compute Clebsch-Gordan coefficients and works directly with the representations of the input and output feature maps. The strategy is to find a basis of kernels that respect a simpler invariance condition at some point $x_0$, and then \textit{steer} it with the defining equation of steerability to move to some arbitrary point $x=g\cdot x_0$. This idea has already been mentioned in the literature before, but not advanced in depth and with some generality. Here we describe how it works with minimal technical tools to make it accessible for a general audience.
>
---
#### [new 138] Curriculum Sampling: A Two-Phase Curriculum for Efficient Training of Flow Matching
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型任务，解决Flow Matching中时间步采样策略的问题。通过提出两阶段的Curriculum Sampling，提升训练效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2603.12517](https://arxiv.org/pdf/2603.12517)**

> **作者:** Pengwei Sun
>
> **摘要:** Timestep sampling $p(t)$ is a central design choice in Flow Matching models, yet common practice increasingly favors static middle-biased distributions (e.g., Logit-Normal). We show that this choice induces a speed--quality trade-off: middle-biased sampling accelerates early convergence but yields worse asymptotic fidelity than Uniform sampling. By analyzing per-timestep training losses, we identify a U-shaped difficulty profile with persistent errors near the boundary regimes, implying that under-sampling the endpoints leaves fine details unresolved. Guided by this insight, we propose \textbf{Curriculum Sampling}, a two-phase schedule that begins with middle-biased sampling for rapid structure learning and then switches to Uniform sampling for boundary refinement. On CIFAR-10, Curriculum Sampling improves the best FID from $3.85$ (Uniform) to $3.22$ while reaching peak performance at $100$k rather than $150$k training steps. Our results highlight that timestep sampling should be treated as an evolving curriculum rather than a fixed hyperparameter.
>
---
#### [new 139] VLM4Rec: Multimodal Semantic Representation for Recommendation with Large Vision-Language Models
- **分类: cs.IR; cs.AI; cs.CV**

- **简介: 该论文属于推荐系统任务，旨在解决多模态推荐中语义对齐问题。通过大视觉语言模型生成物品语义描述，构建更优的多模态表示，提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2603.12625](https://arxiv.org/pdf/2603.12625)**

> **作者:** Ty Valencia; Burak Barlas; Varun Singhal; Ruchir Bhatia; Wei Yang
>
> **备注:** 13 pages, 4 figures, 1 table
>
> **摘要:** Multimodal recommendation is commonly framed as a feature fusion problem, where textual and visual signals are combined to better model user preference. However, the effectiveness of multimodal recommendation may depend not only on how modalities are fused, but also on whether item content is represented in a semantic space aligned with preference matching. This issue is particularly important because raw visual features often preserve appearance similarity, while user decisions are typically driven by higher-level semantic factors such as style, material, and usage context. Motivated by this observation, we propose LVLM-grounded Multimodal Semantic Representation for Recommendation (VLM4Rec), a lightweight framework that organizes multimodal item content through semantic alignment rather than direct feature fusion. VLM4Rec first uses a large vision-language model to ground each item image into an explicit natural-language description, and then encodes the grounded semantics into dense item representations for preference-oriented retrieval. Recommendation is subsequently performed through a simple profile-based semantic matching mechanism over historical item embeddings, yielding a practical offline-online decomposition. Extensive experiments on multiple multimodal recommendation datasets show that VLM4Rec consistently improves performance over raw visual features and several fusion-based alternatives, suggesting that representation quality may matter more than fusion complexity in this setting. The code is released at this https URL.
>
---
#### [new 140] Adaptive Vision-Language Model Routing for Computer Use Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于计算机视觉与自然语言处理任务，解决CUA中VLM路由效率与准确性问题。提出AVR框架，根据难度动态选择模型，提升效率并保证可靠性。**

- **链接: [https://arxiv.org/pdf/2603.12823](https://arxiv.org/pdf/2603.12823)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Andy Luo; Haichen Zhang; Huamin Chen
>
> **摘要:** Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: this https URL.
>
---
#### [new 141] Lyapunov Stable Graph Neural Flow
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图神经网络的鲁棒性研究，解决GNN对对抗扰动敏感的问题。通过引入Lyapunov稳定性理论，设计稳定机制提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12557](https://arxiv.org/pdf/2603.12557)**

> **作者:** Haoyu Chu; Xiaotong Chen; Wei Zhou; Wenjun Cui; Kai Zhao; Shikui Wei; Qiyu Kang
>
> **摘要:** Graph Neural Networks (GNNs) are highly vulnerable to adversarial perturbations in both topology and features, making the learning of robust representations a critical challenge. In this work, we bridge GNNs with control theory to introduce a novel defense framework grounded in integer- and fractional-order Lyapunov stability. Unlike conventional strategies that rely on resource-heavy adversarial training or data purification, our approach fundamentally constrains the underlying feature-update dynamics of the GNN. We propose an adaptive, learnable Lyapunov function paired with a novel projection mechanism that maps the network's state into a stable space, thereby offering theoretically provable stability guarantees. Notably, this mechanism is orthogonal to existing defenses, allowing for seamless integration with techniques like adversarial training to achieve cumulative robustness. Extensive experiments demonstrate that our Lyapunov-stable graph neural flows substantially outperform base neural flows and state-of-the-art baselines across standard benchmarks and various adversarial attack scenarios.
>
---
#### [new 142] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决扩散模型在压缩中计算量大、内存占用高的问题。通过引入DiT-IC，使用扩散Transformer实现高效且高质量的图像重建。**

- **链接: [https://arxiv.org/pdf/2603.13162](https://arxiv.org/pdf/2603.13162)**

> **作者:** Junqi Shi; Ming Lu; Xingchen Li; Anle Ke; Ruiqi Zhang; Zhan Ma
>
> **摘要:** Diffusion-based image compression has recently shown outstanding perceptual fidelity, yet its practicality is hindered by prohibitive sampling overhead and high memory usage. Most existing diffusion codecs employ U-Net architectures, where hierarchical downsampling forces diffusion to operate in shallow latent spaces (typically with only 8x spatial downscaling), resulting in excessive computation. In contrast, conventional VAE-based codecs work in much deeper latent domains (16x - 64x downscaled), motivating a key question: Can diffusion operate effectively in such compact latent spaces without compromising reconstruction quality? To address this, we introduce DiT-IC, an Aligned Diffusion Transformer for Image Compression, which replaces the U-Net with a Diffusion Transformer capable of performing diffusion in latent space entirely at 32x downscaled resolution. DiT-IC adapts a pretrained text-to-image multi-step DiT into a single-step reconstruction model through three key alignment mechanisms: (1) a variance-guided reconstruction flow that adapts denoising strength to latent uncertainty for efficient reconstruction; (2) a self-distillation alignment that enforces consistency with encoder-defined latent geometry to enable one-step diffusion; and (3) a latent-conditioned guidance that replaces text prompts with semantically aligned latent conditions, enabling text-free inference. With these designs, DiT-IC achieves state-of-the-art perceptual quality while offering up to 30x faster decoding and drastically lower memory usage than existing diffusion-based codecs. Remarkably, it can reconstruct 2048x2048 images on a 16 GB laptop GPU.
>
---
#### [new 143] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出SCOPE框架，用于联邦学习中的数据选择，解决数据分布不均和通信效率问题。通过分析潜在空间，优化核心数据集。**

- **链接: [https://arxiv.org/pdf/2603.12976](https://arxiv.org/pdf/2603.12976)**

> **作者:** Md Anwar Hossen; Nathan R. Tallent; Luanzheng Guo; Ali Jannesary
>
> **摘要:** Scientific discovery increasingly requires learning on federated datasets, fed by streams from high-resolution instruments, that have extreme class imbalance. Current ML approaches either require impractical data aggregation or fail due to class imbalance. Existing coreset selection methods rely on local heuristics, making them unaware of the global data landscape and prone to sub-optimal and non-representative pruning. To overcome these challenges, we introduce SCOPE (Semantic Coreset using Orthogonal Projection Embeddings for Federated learning), a coreset framework for federated data that filters anomalies and adaptively prunes redundant data to mitigate long-tail skew. By analyzing the latent space distribution, we score each data point using a representation score that measures the reliability of core class features, a diversity score that quantifies the novelty of orthogonal residuals, and a boundary proximity score that indicates similarity to competing classes. Unlike prior methods, SCOPE shares only scalar metrics with a federated server to construct a global consensus, ensuring communication efficiency. Guided by the global consensus, SCOPE dynamically filters local noise and discards redundant samples to counteract global feature skews. Extensive experiments demonstrate that SCOPE yields competitive global accuracy and robust convergence, all while achieving exceptional efficiency with a 128x to 512x reduction in uplink bandwidth, a 7.72x wall-clock acceleration and reduced FLOP and VRAM footprints for local coreset selection.
>
---
#### [new 144] DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于农业分类任务，解决数据不平衡和标签成本高的问题。提出DirPA方法，在少样本学习中模拟先验分布，提升模型泛化能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.12905](https://arxiv.org/pdf/2603.12905)**

> **作者:** Joana Reuss; Ekaterina Gikalo; Marco Körner
>
> **备注:** 20 pages, 9 Figures, 28 Tables
>
> **摘要:** Real-world agricultural monitoring is often hampered by severe class imbalance and high label acquisition costs, resulting in significant data scarcity. In few-shot learning (FSL) -- a framework specifically designed for data-scarce settings -- , training sets are often artificially balanced. However, this creates a disconnect from the long-tailed distributions observed in nature, leading to a distribution shift that undermines the model's ability to generalize to real-world agricultural tasks. We previously introduced Dirichlet Prior Augmentation (DirPA; Reuss et al., 2026a) to proactively mitigate the effects of such label distribution skews during model training. In this work, we extend the original study's geographical scope. Specifically, we evaluate this extended approach across multiple countries in the European Union (EU), moving beyond localized experiments to test the method's resilience across diverse agricultural environments. Our results demonstrate the effectiveness of DirPA across different geographical regions. We show that DirPA not only improves system robustness and stabilizes training under extreme long-tailed distributions, regardless of the target region, but also substantially improves individual class-specific performance by proactively simulating priors.
>
---
#### [new 145] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D模型生成任务，解决静态网格转可交互物体的问题。提出MotionAnymesh框架，通过物理约束提升模拟准确性，避免碰撞和错误结构。**

- **链接: [https://arxiv.org/pdf/2603.12936](https://arxiv.org/pdf/2603.12936)**

> **作者:** WenBo Xu; Liu Liu; Li Zhang; Dan Guo; RuoNan Liu
>
> **备注:** 5 figures
>
> **摘要:** Converting static 3D meshes into interactable articulated assets is crucial for embodied AI and robotic simulation. However, existing zero-shot pipelines struggle with complex assets due to a critical lack of physical grounding. Specifically, ungrounded Vision-Language Models (VLMs) frequently suffer from kinematic hallucinations, while unconstrained joint estimation inevitably leads to catastrophic mesh inter-penetration during physical simulation. To bridge this gap, we propose MotionAnymesh, an automated zero-shot framework that seamlessly transforms unstructured static meshes into simulation-ready digital twins. Our method features a kinematic-aware part segmentation module that grounds VLM reasoning with explicit SP4D physical priors, effectively eradicating kinematic hallucinations. Furthermore, we introduce a geometry-physics joint estimation pipeline that combines robust type-aware initialization with physics-constrained trajectory optimization to rigorously guarantee collision-free articulation. Extensive experiments demonstrate that MotionAnymesh significantly outperforms state-of-the-art baselines in both geometric precision and dynamic physical executability, providing highly reliable assets for downstream applications.
>
---
## 更新

#### [replaced 001] Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.14027](https://arxiv.org/pdf/2602.14027)**

> **作者:** Jia Li; Xiaomeng Fu; Xurui Peng; Weifeng Chen; Youwei Zheng; Tianyu Zhao; Jiexi Wang; Fangmin Chen; Xing Wang; Hayden Kwok-Hay So
>
> **备注:** 19 pages, 15 figures
>
> **摘要:** Autoregressive video diffusion models have emerged as a scalable paradigm for long video generation. However, they often suffer from severe extrapolation failure, where rapid error accumulation leads to significant temporal degradation when extending beyond training horizons. We identify that this failure primarily stems from the spectral bias of 3D positional embeddings and the lack of dynamic priors in noise sampling. To address these issues, we propose FLEX (Frequency-aware Length EXtension), a training-free inference-time framework that bridges the gap between short-term training and long-term inference. FLEX introduces Frequency-aware RoPE Modulation to adaptively interpolate under-trained low-frequency components while extrapolating high-frequency ones to preserve multi-scale temporal discriminability. This is integrated with Antiphase Noise Sampling (ANS) to inject high-frequency dynamic priors and Inference-only Attention Sink to anchor global structure. Extensive evaluations on VBench demonstrate that FLEX significantly outperforms state-of-the-art models at 6x extrapolation (30s duration) and matches the performance of long-video fine-tuned baselines at 12x scale (60s duration). As a plug-and-play augmentation, FLEX seamlessly integrates into existing inference pipelines for horizon extension. It effectively pushes the generation limits of models such as LongLive, supporting consistent and dynamic video synthesis at a 4-minute scale. Project page is available at this https URL.
>
---
#### [replaced 002] FCMBench: The First Large-scale Financial Credit Multimodal Benchmark for Real-world Applications
- **分类: cs.CV; cs.AI; cs.CE; cs.MM**

- **链接: [https://arxiv.org/pdf/2601.00150](https://arxiv.org/pdf/2601.00150)**

> **作者:** Yehui Yang; Dalu Yang; Fangxin Shang; Wenshuo Zhou; Jie Ren; Yifan Liu; Haojun Fei; Qing Yang; Yanwu Xu; Tao Chen
>
> **摘要:** FCMBench is the first large-scale and privacy-compliant multimodal benchmark for real-world financial credit applications, covering tasks and robustness challenges from domain specific workflows and constraints. The current version of FCMBench covers 26 certificate types, with 5198 privacy-compliant images and 13806 paired VQA samples. It evaluates models on Perception and Reasoning tasks under real-world Robustness interferences, including 3 foundational perception tasks, 4 credit-specific reasoning tasks demanding decision-oriented visual evidence interpretation, and 10 real-world challenges for rigorous robustness stress testing. Moreover, FCMBench offers privacy-compliant realism with minimal leakage risk through in-house scenario-aware captures of manually synthesized templates, without any publicly released images. We conduct extensive evaluations of 28 state-of-the-art vision-language models spanning 14 AI companies and research institutes. Among them, Gemini 3 Pro achieves the best F1 score as a commercial model (65.16), Kimi-K2.5 achieves the best score as an open-source baseline (60.58). The mean and the std. of all tested models is 44.8 and 10.3 respectively, indicating that FCMBench is non-trivial and provides strong resolution for separating modern vision-language model capabilities. Robustness evaluations reveal that even top-performing models experience notable performance degradation under the designed challenges. We have open-sourced this benchmark to advance AI research in the credit domain and provide a domain-specific task for real-world AI applications.
>
---
#### [replaced 003] ExCellGen: Fast, Controllable, Photorealistic 3D Scene Generation from a Single Real-World Exemplar
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.16253](https://arxiv.org/pdf/2412.16253)**

> **作者:** Clément Jambon; Changwoon Choi; Dongsu Zhang; Olga Sorkine-Hornung; Young Min Kim
>
> **摘要:** Photorealistic 3D scene generation is challenging due to the scarcity of large-scale, high-quality real-world 3D datasets and complex workflows requiring specialized expertise for manual modeling. These constraints often result in slow iteration cycles, where each modification demands substantial effort, ultimately stifling creativity. We propose a fast, exemplar-driven framework for generating 3D scenes from a single casual input, such as handheld video or drone footage. Our method first leverages 3D Gaussian Splatting (3DGS) to robustly reconstruct input scenes with a high-quality 3D appearance model. We then train a per-scene Generative Cellular Automaton (GCA) to produce a sparse volume of featurized voxels, effectively amortizing scene generation while enabling controllability. A subsequent patch-based remapping step composites the complete scene from the exemplar's initial 3D Gaussian splats, successfully recovering the appearance statistics of the input scene. The entire pipeline can be trained in less than 10 minutes for each exemplar and generates scenes in 0.5-2 seconds. Our method enables interactive creation with full user control, and we showcase complex 3D generation results from real-world exemplars within a self-contained interactive GUI.
>
---
#### [replaced 004] Uni-Parser Technical Report
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15098](https://arxiv.org/pdf/2512.15098)**

> **作者:** Xi Fang; Haoyi Tao; Shuwen Yang; Chaozheng Huang; Suyang Zhong; Haocheng Lu; Han Lyu; Junjie Wang; Xinyu Li; Linfeng Zhang; Guolin Ke
>
> **摘要:** This technical report introduces Uni-Parser, an industrial-grade document parsing engine tailored for scientific literature and patents, delivering high throughput, robust accuracy, and cost efficiency. Unlike pipeline-based document parsing methods, Uni-Parser employs a modular, loosely coupled multi-expert architecture that preserves fine-grained cross-modal alignments across text, equations, tables, figures, and chemical structures, while remaining easily extensible to emerging modalities. The system incorporates adaptive GPU load balancing, distributed inference, dynamic module orchestration, and configurable modes that support either holistic or modality-specific parsing. Optimized for large-scale cloud deployment, Uni-Parser achieves a processing rate of up to 20 PDF pages per second on 8 x NVIDIA RTX 4090D GPUs, enabling cost-efficient inference across billions of pages. This level of scalability facilitates a broad spectrum of downstream applications, ranging from literature retrieval and summarization to the extraction of chemical structures, reaction schemes, and bioactivity data, as well as the curation of large-scale corpora for training next-generation large language models and AI4Science models.
>
---
#### [replaced 005] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.07801](https://arxiv.org/pdf/2602.07801)**

> **作者:** Wenqi Liu; Yunxiao Wang; Shijie Ma; Meng Liu; Qile Su; Tianke Zhang; Haonan Fan; Changyi Liu; Kaiyu Jiang; Jiankang Chen; Kaiyu Tang; Bin Wen; Fan Yang; Tingting Gao; Han Li; Yinwei Wei; Xuemeng Song
>
> **摘要:** In long-video understanding, conventional uniform frame sampling often fails to capture key visual evidence, leading to degraded performance and increased hallucinations. To address this, recent agentic thinking-with-videos paradigms have emerged, adopting a localize-clip-answer pipeline in which the model actively identifies relevant video segments, performs dense sampling within those clips, and then produces answers. However, existing methods remain inefficient, suffer from weak localization, and adhere to rigid workflows. To solve these issues, we propose VideoTemp-o3, a unified agentic thinking-with-videos framework that jointly models video grounding and question answering. VideoTemp-o3 exhibits strong localization capability, supports on-demand clipping, and can refine inaccurate localizations. Specifically, in the supervised fine-tuning stage, we design a unified masking mechanism that encourages exploration while preventing noise. For reinforcement learning, we introduce dedicated rewards to mitigate reward hacking. Besides, from the data perspective, we develop an effective pipeline to construct high-quality long video grounded QA data, along with a corresponding benchmark for systematic evaluation across various video durations. Experimental results demonstrate that our method achieves remarkable performance on both long video understanding and grounding.
>
---
#### [replaced 006] Enhancing Novel View Synthesis via Geometry Grounded Set Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.07540](https://arxiv.org/pdf/2601.07540)**

> **作者:** Farhad G. Zanjani; Hong Cai; Amirhossein Habibian
>
> **备注:** Paper and supplementary materials
>
> **摘要:** We present SetDiff, a geometry-grounded multi-view diffusion framework that enhances novel-view renderings produced by 3D Gaussian Splatting. Our method integrates explicit 3D priors, pixel-aligned coordinate maps and pose-aware Plucker ray embeddings, into a set-based diffusion model capable of jointly processing variable numbers of reference and target views. This formulation enables robust occlusion handling, reduces hallucinations under low-signal conditions, and improves photometric fidelity in visual content restoration. A unified set mixer performs global token-level attention across all input views, supporting scalable multi-camera enhancement while maintaining computational efficiency through latent-space supervision and selective decoding. Extensive experiments on EUVS, Para-Lane, nuScenes, and DL3DV demonstrate significant gains in perceptual fidelity, structural similarity, and robustness under severe extrapolation. SetDiff establishes a state-of-the-art diffusion-based solution for realistic and reliable novel-view synthesis in autonomous driving scenarios.
>
---
#### [replaced 007] BitDance: Scaling Autoregressive Generative Models with Binary Tokens
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.14041](https://arxiv.org/pdf/2602.14041)**

> **作者:** Yuang Ai; Jiaming Han; Shaobin Zhuang; Weijia Mao; Xuefeng Hu; Ziyan Yang; Zhenheng Yang; Yali Wang; Huaibo Huang; Xiangyu Yue; Hao Chen
>
> **备注:** Code and models: this https URL
>
> **摘要:** We present BitDance, a scalable autoregressive (AR) image generator that predicts binary visual tokens instead of codebook indices. With high-entropy binary latents, BitDance lets each token represent up to $2^{256}$ states, yielding a compact yet highly expressive discrete representation. Sampling from such a huge token space is difficult with standard classification. To resolve this, BitDance uses a binary diffusion head: instead of predicting an index with softmax, it employs continuous-space diffusion to generate the binary tokens. Furthermore, we propose next-patch diffusion, a new decoding method that predicts multiple tokens in parallel with high accuracy, greatly speeding up inference. On ImageNet 256x256, BitDance achieves an FID of 1.24, the best among AR models. With next-patch diffusion, BitDance beats state-of-the-art parallel AR models that use 1.4B parameters, while using 5.4x fewer parameters (260M) and achieving 8.7x speedup. For text-to-image generation, BitDance trains on large-scale multimodal tokens and generates high-resolution, photorealistic images efficiently, showing strong performance and favorable scaling. When generating 1024x1024 images, BitDance achieves a speedup of over 30x compared to prior AR models. We release code and models to facilitate further research on AR foundation models. Code and models are available at: this https URL.
>
---
#### [replaced 008] Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.12067](https://arxiv.org/pdf/2603.12067)**

> **作者:** Simone Cammarasana
>
> **摘要:** The convolution operator is the fundamental building block of modern convolutional neural networks (CNNs), owing to its simplicity, translational equivariance, and efficient implementation. However, its structure as a fixed, linear, locally-averaging operator limits its ability to capture structured signal properties such as low-rank decompositions, adaptive basis representations, and non-uniform spatial dependencies. This paper presents a systematic taxonomy of operators that extend or replace the standard convolution in learning-based image processing pipelines. We organise the landscape of alternative operators into five families: (i) decomposition-based operators, which separate structural and noise components through singular value or tensor decompositions; (ii) adaptive weighted operators, which modulate kernel contributions as a function of spatial position or signal content; (iii) basis-adaptive operators, which optimise the analysis bases together with the network weights; (iv) integral and kernel operators, which generalise the convolution to position-dependent and non-linear kernels; and (v) attention-based operators, which relax the locality assumption entirely. For each family, we provide a formal definition, a discussion of its structural properties with respect to the convolution, and a critical analysis of the tasks for which the operator is most appropriate. We further provide a comparative analysis of all families across relevant dimensions -- linearity, locality, equivariance, computational cost, and suitability for image-to-image and image-to-label tasks -- and outline the open challenges and future directions of this research area.
>
---
#### [replaced 009] LADMIM: Logical Anomaly Detection with Masked Image Modeling in Discrete Latent Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.10234](https://arxiv.org/pdf/2410.10234)**

> **作者:** Shunsuke Sakai; Tatushito Hasegawa; Makoto Koshino
>
> **备注:** Accepted at TMLR2025. Code is available at this https URL
>
> **摘要:** Detecting anomalies such as an incorrect combination of objects or deviations in their positions is a challenging problem in unsupervised anomaly detection (AD). Since conventional AD methods mainly focus on local patterns of normal images, they struggle with detecting logical anomalies that appear in the global patterns. To effectively detect these challenging logical anomalies, we introduce Logical Anomaly Detection with Masked Image Modeling (LADMIM), a novel unsupervised AD framework that harnesses the power of masked image modeling and discrete representation learning. Our core insight is that predicting the missing region forces the model to learn the long-range dependencies between patches. Specifically, we formulate AD as a mask completion task, which predicts the distribution of discrete latents in the masked region. As a distribution of discrete latents is invariant to the low-level variance in the pixel space, the model can desirably focus on the logical dependencies in the image, which improves accuracy in the logical AD. We evaluate the AD performance on five benchmarks and show that our approach achieves compatible performance without any pre-trained segmentation models. We also conduct comprehensive experiments to reveal the key factors that influence logical AD performance.
>
---
#### [replaced 010] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.08605](https://arxiv.org/pdf/2603.08605)**

> **作者:** Hikmat Khan; Wei Chen; Muhammad Khalid Khan Niazi
>
> **摘要:** Background and objectives: Colorectal cancer histopathological grading depends on accurate segmentation of glandular structures. Current deep learning approaches rely on large scale pixel level annotations that are labor intensive and difficult to obtain in routine clinical practice. Weakly supervised semantic segmentation offers a promising alternative. However, class activation map based methods often produce incomplete pseudo masks that emphasize highly discriminative regions and fail to supervise unannotated glandular structures. We propose a weakly supervised teacher student framework that leverages sparse pathologist annotations and an Exponential Moving Average stabilized teacher network to generate refined pseudo masks. Methods: The framework integrates confidence based filtering, adaptive fusion of teacher predictions with limited ground truth, and curriculum guided refinement to progressively segment unannotated glandular regions. The method was evaluated on an institutional colorectal cancer cohort from The Ohio State University Wexner Medical Center consisting of 60 hematoxylin and eosin stained whole slide images and on public datasets including the Gland Segmentation dataset, TCGA COAD, TCGA READ, and SPIDER. Results: On the Gland Segmentation dataset the framework achieved a mean Intersection over Union of 80.10 and a mean Dice coefficient of 89.10. Cross cohort evaluation demonstrated robust generalization on TCGA COAD and TCGA READ without additional annotations, while reduced performance on SPIDER reflected domain shift. Conclusions: The proposed framework provides an annotation efficient and generalizable approach for gland segmentation in colorectal histopathology.
>
---
#### [replaced 011] FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.24084](https://arxiv.org/pdf/2602.24084)**

> **作者:** Matteo Ballegeer; Dries F. Benoit
>
> **备注:** Manuscript accepted at CVPR 2026
>
> **摘要:** Learning directly from boundary representations (B-reps) has significantly advanced 3D CAD analysis. However, state-of-the-art B-rep learning methods rely on absolute coordinates and normals to encode global context, making them highly sensitive to rotations. Our experiments reveal that models achieving over 95% accuracy on aligned benchmarks can collapse to as low as 10% under arbitrary $\mathbf{SO}(3)$ rotations. To address this, we introduce FoV-Net, the first B-rep learning framework that captures both local surface geometry and global structural context in a rotation-invariant manner. Each face is represented by a Local Reference Frame (LRF) UV-grid that encodes its local surface geometry, and by Field-of-View (FoV) grids that capture the surrounding 3D context by casting rays and recording intersections with neighboring faces. Lightweight CNNs extract per-face features, which are propagated over the B-rep graph using a graph attention network. FoV-Net achieves state-of-the-art performance on B-rep classification and segmentation benchmarks, demonstrating robustness to arbitrary rotations while also requiring less training data to achieve strong results.
>
---
#### [replaced 012] TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12823](https://arxiv.org/pdf/2601.12823)**

> **作者:** Belal Shaheen; Minh-Hieu Nguyen; Bach-Thuan Bui; Shubham; Tim Wu; Michael Fairley; Matthew David Zane; Michael Wu; James Tompkin
>
> **摘要:** Aerial remote sensing efficiently surveys large areas, but accurate direct object-level measurement remains difficult in complex natural scenes. Advancements in 3D computer vision, particularly radiance field representations such as NeRF and 3D Gaussian splatting, can improve reconstruction fidelity from posed imagery. Nevertheless, direct aerial measurement of important attributes like tree diameter at breast height (DBH) remains challenging. Trunks in aerial forest scans are distant and sparsely observed in image views; at typical operating altitudes, stems may span only a few pixels. With these constraints, conventional reconstruction methods have inaccurate breast-height trunk geometry. TreeDGS is an aerial image reconstruction method that uses 3D Gaussian splatting as a continuous scene representation for trunk measurement. After SfM--MVS initialization and Gaussian optimization, we extract a dense point set from the Gaussian field using RaDe-GS's depth-aware cumulative-opacity integration and associate each sample with a multi-view opacity reliability score. Then, we isolate trunk points and estimate DBH using opacity-weighted solid-circle fitting. Evaluated on 10 plots with field-measured DBH, TreeDGS reaches 4.79 cm RMSE (about 2.6 pixels at this GSD) and outperforms a LiDAR baseline (7.66 cm RMSE). This shows that TreeDGS can enable accurate, low-cost aerial DBH measurement .
>
---
#### [replaced 013] Towards Interactive Intelligence for Digital Humans
- **分类: cs.CV; cs.CL; cs.GR; cs.HC**

- **简介: 该论文提出交互智能，解决数字人缺乏真实互动的问题。通过Mio框架实现多模态交互与自我进化，提升数字人的智能交互能力。**

- **链接: [https://arxiv.org/pdf/2512.13674](https://arxiv.org/pdf/2512.13674)**

> **作者:** Yiyi Cai; Xuangeng Chu; Xiwei Gao; Sitong Gong; Yifei Huang; Caixin Kang; Kunhang Li; Haiyang Liu; Ruicong Liu; Yun Liu; Dianwen Ng; Zixiong Su; Erwin Wu; Yuhan Wu; Dingkun Yan; Tianyu Yan; Chang Zeng; Bo Zheng; You Zhou
>
> **摘要:** We introduce Interactive Intelligence, a novel paradigm of digital human that is capable of personality-aligned expression, adaptive interaction, and self-evolution. To realize this, we present Mio (Multimodal Interactive Omni-Avatar), an end-to-end framework composed of five specialized modules: Thinker, Talker, Face Animator, Body Animator, and Renderer. This unified architecture integrates cognitive reasoning with real-time multimodal embodiment to enable fluid, consistent interaction. Furthermore, we establish a new benchmark to rigorously evaluate the capabilities of interactive intelligence. Extensive experiments demonstrate that our framework achieves superior performance compared to state-of-the-art methods across all evaluated dimensions. Together, these contributions move digital humans beyond superficial imitation toward intelligent interaction.
>
---
#### [replaced 014] SATGround: A Spatially-Aware Approach for Visual Grounding in Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08881](https://arxiv.org/pdf/2512.08881)**

> **作者:** Aysim Toker; Andreea-Maria Oncescu; Roy Miles; Ismail Elezi; Jiankang Deng
>
> **摘要:** Vision-language models (VLMs) are emerging as powerful generalist tools for remote sensing, capable of integrating information across diverse tasks and enabling flexible, instruction-based interactions via a chat interface. In this work, we enhance VLM-based visual grounding in satellite imagery by proposing a novel structured localization mechanism. Our approach involves finetuning a pretrained VLM on a diverse set of instruction-following tasks, while interfacing a dedicated grounding module through specialized control tokens for localization. This method facilitates joint reasoning over both language and spatial information, significantly enhancing the model's ability to precisely localize objects in complex satellite scenes. We evaluate our framework on several remote sensing benchmarks, consistently improving the state-of-the-art, including a 33.2% relative improvement over previous methods on visual grounding. Our results highlight the benefits of integrating structured spatial reasoning into VLMs, paving the way for more reliable real-world satellite data analysis. Code will be released upon acceptance.
>
---
#### [replaced 015] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14099](https://arxiv.org/pdf/2512.14099)**

> **作者:** Ruishu Zhu; Zhihao Huang; Jiacheng Sun; Ping Luo; Hongyuan Zhang; Xuelong Li
>
> **摘要:** Motivated by discrete diffusion's success in language-vision modeling, we explore its potential for multi-view generation, a task dominated by continuous approaches. We introduce ViewMask-1-to-3, formulating multi-view synthesis as a discrete sequence modeling problem where each viewpoint is represented as visual tokens from MAGVIT-v2. Through masked token prediction, our approach enables progressive multi-view generation via iterative token unmasking, unifying language and vision in a shared token space. Importantly, simple random masking combined with self-attention naturally encourages cross-view consistency without specialized architectures or 3D geometric priors. Our method outperforms the baseline on the GSO and 3D-FUTURE benchmarks, ranking first on average across standard image metrics and improving IoU by 10.6% on 3D-FUTURE. This validates discrete diffusion as a promising candidate for multi-view generation.
>
---
#### [replaced 016] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11460](https://arxiv.org/pdf/2603.11460)**

> **作者:** Seung hee Choi; MinJu Jeon; Hyunwoo Oh; Jihwan Lee; Dong-Jin Kim
>
> **备注:** CVPR 2026 accepted paper (main track)
>
> **摘要:** Existing retrieval-augmented approaches for Dense Video Captioning (DVC) often fail to achieve accurate temporal segmentation aligned with true event boundaries, as they rely on heuristic strategies that overlook ground truth event boundaries. The proposed framework, \textbf{STaRC}, overcomes this limitation by supervising frame-level saliency through a highlight detection module. Note that the highlight detection module is trained on binary labels derived directly from DVC ground truth annotations without the need for additional annotation. We also propose to utilize the saliency scores as a unified temporal signal that drives retrieval via saliency-guided segmentation and informs caption generation through explicit Saliency Prompts injected into the decoder. By enforcing saliency-constrained segmentation, our method produces temporally coherent segments that align closely with actual event transitions, leading to more accurate retrieval and contextually grounded caption generation. We conduct comprehensive evaluations on the YouCook2 and ViTT benchmarks, where STaRC achieves state-of-the-art performance across most of the metrics. Our code is available at this https URL
>
---
#### [replaced 017] DiffProxy: Multi-View Human Mesh Recovery via Diffusion-Generated Dense Proxies
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02267](https://arxiv.org/pdf/2601.02267)**

> **作者:** Renke Wang; Zhenyu Zhang; Ying Tai; Jun Li; Jian Yang
>
> **备注:** Page: this https URL, Code: this https URL
>
> **摘要:** Precise human mesh recovery (HMR) from multi-view images remains challenging: end-to-end methods produce entangled errors hard to localize, while fitting-based methods rely on sparse keypoints that provide limited surface constraints. We observe that the true bottleneck lies in the quality of intermediate representations, and that dense pixel-to-surface correspondences can be effectively generated by repurposing pre-trained diffusion models with rich visual priors. We propose DiffProxy, a Stable-Diffusion-based framework trained on large-scale synthetic data with pixel-perfect annotations. A multi-conditional proxy generator predicts dense correspondences from multi-view images, providing uniform surface constraints that enable precise fitting. Hand refinement feeds enlarged hand crops alongside full-body images for fine-grained detail, while test-time scaling exploits diffusion stochasticity to estimate per-pixel uncertainty. Trained only on synthetic data, DiffProxy achieves state-of-the-art results on five diverse real-world benchmarks. Project page: this https URL
>
---
#### [replaced 018] MaDiS: Taming Masked Diffusion Language Models for Sign Language Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19577](https://arxiv.org/pdf/2601.19577)**

> **作者:** Ronglai Zuo; Rolandos Alexandros Potamias; Qi Sun; Evangelos Ververas; Jiankang Deng; Stefanos Zafeiriou
>
> **摘要:** Sign language generation (SLG) aims to translate written texts into expressive sign motions, bridging communication barriers for the Deaf and Hard-of-Hearing communities. Recent studies formulate SLG within the language modeling framework using autoregressive language models, which suffer from unidirectional context modeling and slow token-by-token inference. To address these limitations, we present MaDiS, a masked-diffusion-based language model for SLG that captures bidirectional dependencies and supports efficient parallel multi-token generation. We further introduce a tri-level cross-modal pretraining scheme that jointly learns from token-, latent-, and 3D physical-space objectives to leverage complementary, multi-level sign representations. To accelerate model convergence in the fine-tuning stage, we design a novel unmasking strategy with temporal checkpoints, which restructures generation in a coarse-to-fine manner and reduces the combinatorial complexity of unmasking orders by over $10^{41}$ times. In addition, a mixture-of-parts embedding layer is developed to effectively fuse information stored in different part-wise sign tokens through a learnable gate and well-optimized codebooks. Extensive experiments on CSL-Daily, Phoenix-2014T, and How2Sign demonstrate that MaDiS achieves superior performance across multiple metrics, including DTW error and two newly introduced metrics, SiBLEU and SiCLIP, while delivering a 40\% higher throughput. Code and models will be publicly released.
>
---
#### [replaced 019] GeoZero: Incentivizing Reasoning from Scratch on Geospatial Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22645](https://arxiv.org/pdf/2511.22645)**

> **作者:** Di Wang; Shunyu Liu; Wentao Jiang; Fengxiang Wang; Yi Liu; Xiaolei Qin; Zhiming Luo; Chaoyang Zhou; Haonan Guo; Jing Zhang; Bo Du; Dacheng Tao; Liangpei Zhang
>
> **备注:** Code, data, and models are available at this https URL
>
> **摘要:** Multimodal large language models (MLLMs) have undergone rapid development in advancing geospatial scene understanding. Recent studies have sought to enhance the reasoning capabilities of remote sensing MLLMs, typically through cold-start training with elaborately curated chain-of-thought (CoT) data. However, this approach not only incurs substantial annotation costs but also introduces human biases that may limit the diversity of model reasoning. To address these challenges, we propose GeoZero, a framework that enables MLLMs to perform geospatial reasoning without any predefined CoT supervision. Specifically, we construct two datasets, GeoZero-Instruct and GeoZero-Hard. GeoZero-Instruct allows the model to acquire preliminary geospatial knowledge through supervised fine-tuning, while GeoZero-Hard stimulates deep reasoning during the subsequent reinforcement learning stage. Furthermore, we introduce Answer-Anchored Group Relative Policy Optimization (A$^2$GRPO), where the reasoning process is regularized by the model's own answers, encouraging diverse yet accurate thinking. Extensive experiments on multiple remote sensing vision-language benchmarks demonstrate that GeoZero not only surpasses existing state-of-the-art methods but also fosters universal emergent reasoning capabilities across diverse geospatial tasks. Code, data, and models are available at this https URL.
>
---
#### [replaced 020] Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15475](https://arxiv.org/pdf/2601.15475)**

> **作者:** Yunshan Qi; Lin Zhu; Nan Bao; Yifan Zhao; Jia Li
>
> **备注:** Accepted by IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2026, Project Page: this https URL
>
> **摘要:** Novel view synthesis from low dynamic range (LDR) blurry images, which are common in the wild, struggles to recover high dynamic range (HDR) and sharp 3D representations in extreme lighting conditions. Although existing methods employ event data to address this issue, they ignore the sensor-physics mismatches between the camera output and physical world radiance, resulting in suboptimal HDR and deblurring results. To cope with this problem, we propose a unified sensor-physics grounded NeRF framework for sharp HDR novel view synthesis from single-exposure blurry LDR images and corresponding events. We employ NeRF to directly represent the actual radiance of the 3D scene in the HDR domain and model raw HDR scene rays hitting the sensor pixels as in the physical world. A pixel-wise RGB mapping field is introduced to align the above rendered pixel values with the sensor-recorded LDR pixel values of the input images. A novel event mapping field is also designed to bridge the physical scene dynamics and actual event sensor output. The two mapping fields are jointly optimized with the NeRF network, leveraging the spatial and temporal dynamic information in events to enhance the sharp HDR 3D representation learning. Experiments on the collected and public datasets demonstrate that our method can achieve state-of-the-art deblurring HDR novel view synthesis results with single-exposure blurry LDR images and corresponding events.
>
---
#### [replaced 021] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17704](https://arxiv.org/pdf/2509.17704)**

> **作者:** Bo Li; Yunkuo Lei; Tingting Bao; Hang Yan; Yaxian Wang; Weiping Fu; Lingling Zhang; Jun Liu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Multi-focus image fusion (MFIF) is a crucial technique in image processing, with a key challenge being the generation of decision maps with precise boundaries. However, traditional methods based on heuristic rules and deep learning methods with black-box mechanisms are difficult to generate high-quality decision maps. To overcome this challenge, we introduce neurodynamics-driven coupled neural P (CNP) systems, which are third-generation neural computation models inspired by spiking mechanisms, to enhance the accuracy of decision maps. Specifically, we first conduct an in-depth analysis of the model's neurodynamics to identify the constraints between the network parameters and the input signals. This solid analysis avoids abnormal continuous firing of neurons and ensures the model accurately distinguishes between focused and unfocused regions, generating high-quality decision maps for MFIF. Based on this analysis, we propose a Neurodynamics-Driven CNP Fusion model (ND-CNPFuse) tailored for the challenging MFIF task. Unlike current ideas of decision map generation, ND-CNPFuse distinguishes between focused and unfocused regions by mapping the source image into interpretable spike matrices. By comparing the number of spikes, an accurate decision map can be generated directly without any post-processing. Extensive experimental results show that ND-CNPFuse achieves new state-of-the-art performance on four classical MFIF datasets, including Lytro, MFFW, MFI-WHU, and Real-MFF. The code is available at this https URL.
>
---
#### [replaced 022] FSDAM: Few-Shot Driving Attention Modeling via Vision-Language Coupling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12708](https://arxiv.org/pdf/2511.12708)**

> **作者:** Kaiser Hamid; Can Cui; Khandakar Ashrafi Akbar; Ziran Wang; Nade Liang
>
> **摘要:** Understanding not only where drivers look but also why their attention shifts is essential for interpretable human-AI collaboration in autonomous driving. Driver attention is not purely perceptual but semantically structured. Thus, attention shifts can be learned through minimal semantic supervision rather than dense large-scale annotation. We present \textbf{FSDAM} (\textbf{F}ew-\textbf{S}hot \textbf{D}river \textbf{A}ttention \textbf{M}odeling), a framework that achieves joint spatial attention prediction and structured explanation generation using 90 annotated examples. Our key insight is to decompose attention into an explicit reasoning representation, including scene context, current focus, anticipated next focus, and causal explanation, and to learn next-focus anticipation through minimal-pair supervision. To address task conflict and large sample requirements of existing models, and to mitigate task interference under limited data, we introduce a novel dual-pathway architecture in which separate modules handle spatial prediction and caption generation. In addition, we use a training-only vision-language alignment mechanism that injects semantic priors into spatial learning without increasing inference complexity, mitigating task interference under few-shot training. Despite extreme data scarcity, FSDAM achieves competitive performance in gaze prediction, and generates coherent, context-aware structural reasoning for improved interpretability. The model further demonstrates strong zero-shot generalization across multiple driving benchmarks.
>
---
#### [replaced 023] Ref-DGS: Reflective Dual Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2603.07664](https://arxiv.org/pdf/2603.07664)**

> **作者:** Ningjing Fan; Yiqun Wang; Dongming Yan; Peter Wonka
>
> **备注:** Project page: this https URL
>
> **摘要:** Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.
>
---
#### [replaced 024] AIMC-Spec: A Benchmark Dataset for Automatic Intrapulse Modulation Classification under Variable Noise Conditions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08265](https://arxiv.org/pdf/2601.08265)**

> **作者:** Sebastian L. Cocks; Salvador Dreo; Brian Ng; Feras Dayoub
>
> **备注:** This version updates the previously released dataset by reducing storage requirements, revising the SNR calculation procedure, and restructuring the dataset format The first version of this work was published in IEEE Access DOI: https://doi.org/10.1109/ACCESS.2025.3645091
>
> **摘要:** A lack of standardized datasets has long hindered progress in automatic intrapulse modulation classification (AIMC), a critical task in radar signal analysis for electronic support systems, particularly under noisy or degraded conditions. AIMC seeks to identify the modulation type embedded within a single radar pulse from its complex in-phase and quadrature (I/Q) representation, enabling automated interpretation of intrapulse structure. This paper introduces AIMC-Spec, a comprehensive synthetic dataset for spectrogram-based image classification, encompassing 30 modulation types across 5 signal-to-noise ratio (SNR) levels. To benchmark AIMC-Spec, five representative deep learning algorithms ranging from lightweight CNNs and denoising architectures to transformer-based networks were re-implemented and evaluated under a unified input format. The results reveal significant performance variation, with frequency-modulated (FM) signals classified more reliably than phase-modulated (PM) types, particularly at low SNRs. A focused FM-only test further highlights how modulation type and network architecture influence classifier robustness. AIMC-Spec establishes a reproducible baseline and provides a foundation for future research and standardization in the AIMC domain.
>
---
#### [replaced 025] SuperQuadricOcc: Real-Time Self-Supervised Semantic Occupancy Estimation with Superquadric Volume Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17361](https://arxiv.org/pdf/2511.17361)**

> **作者:** Seamie Hayes; Alexandre Boulch; Andrei Bursuc; Reenu Mohandas; Ganesh Sistu; Tim Brophy; Ciaran Eising
>
> **摘要:** Self-supervision for semantic occupancy estimation is appealing as it removes the labour-intensive manual annotation, thus allowing one to scale to larger autonomous driving datasets. Superquadrics offer an expressive shape family very suitable for this task, yet their deployment in a self-supervised setting has been hindered by the lack of efficient rendering methods to bridge the 3D scene representation and 2D training pseudo-labels. To address this, we introduce SuperQuadricOcc, the first self-supervised occupancy model to leverage superquadrics for scene representation. To overcome the rendering limitation, we propose a real-time volume renderer that preserves the fidelity of the superquadric shape during rendering. It relies on spatial superquadric-voxel indexing, restricting each ray sample to query only nearby superquadrics, thereby greatly reducing memory usage and computational cost. Using drastically fewer primitives than previous Gaussian-based methods, SuperQuadricOcc achieves state-of-the-art performance on the Occ3D-nuScenes dataset, while running at real-time inference speeds with substantially reduced memory footprint.
>
---
#### [replaced 026] Motion Dreamer: Boundary Conditional Motion Reasoning for Physically Coherent Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.00547](https://arxiv.org/pdf/2412.00547)**

> **作者:** Tianshuo Xu; Zhifei Chen; Leyi Wu; Hao Lu; Yuying Chen; Lihui Jiang; Bingbing Liu; Yingcong Chen
>
> **备注:** The authors have decided to withdraw this article due to the following reasons identified after publication: Experimental Errors: Significant inaccuracies were discovered in the experimental results concerning segmentation and depth estimation. Authorship Disputes: In addition to the technical issues, there are unresolved disagreements regarding the author sequence and contributions
>
> **摘要:** Recent advances in video generation have shown promise for generating future scenarios, critical for planning and control in autonomous driving and embodied intelligence. However, real-world applications demand more than visually plausible predictions; they require reasoning about object motions based on explicitly defined boundary conditions, such as initial scene image and partial object motion. We term this capability Boundary Conditional Motion Reasoning. Current approaches either neglect explicit user-defined motion constraints, producing physically inconsistent motions, or conversely demand complete motion inputs, which are rarely available in practice. Here we introduce Motion Dreamer, a two-stage framework that explicitly separates motion reasoning from visual synthesis, addressing these limitations. Our approach introduces instance flow, a sparse-to-dense motion representation enabling effective integration of partial user-defined motions, and the motion inpainting strategy to robustly enable reasoning motions of other objects. Extensive experiments demonstrate that Motion Dreamer significantly outperforms existing methods, achieving superior motion plausibility and visual realism, thus bridging the gap towards practical boundary conditional motion reasoning. Our webpage is available: this https URL.
>
---
#### [replaced 027] VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06097](https://arxiv.org/pdf/2506.06097)**

> **作者:** Zikang Wang; Boyu Chen; Zhengrong Yue; Yi Wang; Yu Qiao; Limin Wang; Yali Wang
>
> **摘要:** Recent advances in video understanding have been driven by this http URL these MLLMs are good at analyzing short videos, while suffering from difficulties in understanding videos with a longer context. To address this difficulty, several agent methods have been proposed, using MLLMs as agents for retrieving extra contextual knowledge in a long video. However, most existing agents ignore the key fact that a long video is composed with multiple shots, i.e., to answer the user question from a long video, it is critical to deeply understand its relevant shots like human. Without such insight, these agents often mistakenly find redundant even noisy temporal context, restricting their capacity for long video understanding. To fill this gap, we propose VideoChat-A1, a novel long video agent paradigm. Different from the previous works, our VideoChat-A1 can deeply think with long videos, via a distinct chain-of-shot reasoning paradigm. More specifically, it can progressively select the relevant shots of user question, and look into these shots in a coarse-to-fine partition. By multi-modal reasoning along the shot chain, VideoChat-A1 can effectively mimic step-by-step human thinking process, allowing the interactive discovery of preferable temporal context for thoughtful understanding in long videos. Extensive experiments show that, VideoChat-A1 achieves the state-of-the-art performance on the mainstream long video QA benchmarks, e.g., it achieves 77.0 on VideoMME~(w/ subs) and 70.1 on EgoSchema, outperforming its strong baselines (e.g., InternVL2.5-8B and InternVideo2.5-8B), by up to 10.1\% and 6.2\%. Compared to leading closed-source GPT-4o and Gemini 1.5 Pro, VideoChat-A1 offers competitive accuracy, but only with 7\% input frames and 12\% inference time on average. The code is available on this https URL.
>
---
#### [replaced 028] Overcoming the Curvature Bottleneck in MeanFlow
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.23342](https://arxiv.org/pdf/2511.23342)**

> **作者:** Xinxi Zhang; Shiwei Tan; Quang Nguyen; Quan Dao; Ligong Han; Xiaoxiao He; Tunyu Zhang; Chengzhi Mao; Dimitris Metaxas; Vladimir Pavlovic
>
> **摘要:** MeanFlow offers a promising framework for one-step generative modeling by directly learning a mean-velocity field, bypassing expensive numerical integration. However, we find that the highly curved generative trajectories of existing models induce a noisy loss landscape, severely bottlenecking convergence and model quality. We leverage a fundamental geometric principle to overcome this: mean-velocity estimation is drastically simpler along straight paths. Building on this insight, we propose Rectified MeanFlow, a self-distillation approach that learns the mean-velocity field over a straightened velocity field, induced by rectified couplings from a pretrained model. To further promote linearity, we introduce a distance-based truncation heuristic that prunes residual high-curvature pairs. By smoothing the optimization landscape, our method achieves strong one-step generation performance. We improve the FID of baseline MeanFlow models from 30.9 to 8.6 under same training budget, and outperform the recent 2-rectified flow++ by 33.4% in FID while running 26x faster. Our work suggests that the difficulty of one-step flow generation stems partially from the rugged optimization landscapes induced by curved trajectories. Code is available at this https URL.
>
---
#### [replaced 029] TubeMLLM: A Foundation Model for Topology Knowledge Exploration in Vessel-like Anatomy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09217](https://arxiv.org/pdf/2603.09217)**

> **作者:** Yaoyu Liu; Minghui Zhang; Xin You; Hanxiao Zhang; Yun Gu
>
> **备注:** 18 pages, 12 figures
>
> **摘要:** Modeling medical vessel-like anatomy is challenging due to its intricate topology and sensitivity to dataset shifts. Consequently, task-specific models often suffer from topological inconsistencies, including artificial disconnections and spurious merges. Motivated by the promise of multimodal large language models (MLLMs) for zero-shot generalization, we propose TubeMLLM, a unified foundation model that couples structured understanding with controllable generation for medical vessel-like anatomy. By integrating topological priors through explicit natural language prompting and aligning them with visual representations in a shared-attention architecture, TubeMLLM significantly enhances topology-aware perception. Furthermore, we construct TubeMData, a pionner multimodal benchmark comprising comprehensive topology-centric tasks, and introduce an adaptive loss weighting strategy to emphasize topology-critical regions during training. Extensive experiments on fifteen diverse datasets demonstrate our superiority. Quantitatively, TubeMLLM achieves state-of-the-art out-of-distribution performance, substantially reducing global topological discrepancies on color fundus photography (decreasing the $\beta_{0}$ number error from 37.42 to 8.58 compared to baselines). Notably, TubeMLLM exhibits exceptional zero-shot cross-modality transferring ability on unseen X-ray angiography, achieving a Dice score of 67.50% while significantly reducing the $\beta_{0}$ error to 1.21. TubeMLLM also maintains robustness against degradations such as blur, noise, and low resolution. Furthermore, in topology-aware understanding tasks, the model achieves 97.38% accuracy in evaluating mask topological quality, significantly outperforming standard vision-language baselines.
>
---
#### [replaced 030] Omni-Video 2: Scaling MLLM-Conditioned Diffusion for Unified Video Generation and Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08820](https://arxiv.org/pdf/2602.08820)**

> **作者:** Hao Yang; Zhiyu Tan; Jia Gong; Luozheng Qin; Hesen Chen; Xiaomeng Yang; Yuqing Sun; Yuetan Lin; Mengping Yang; Hao Li
>
> **备注:** Technical Report, Project: this https URL
>
> **摘要:** We present Omni-Video 2, a scalable and computationally efficient model that connects pretrained multimodal large-language models (MLLMs) with video diffusion models for unified video generation and editing. Our key idea is to exploit the understanding and reasoning capabilities of MLLMs to produce explicit target captions to interpret user instructions. In this way, the rich contextual representations from the understanding model are directly used to guide the generative process, thereby improving performance on complex and compositional editing. Moreover, a lightweight adapter is developed to inject multimodal conditional tokens into pretrained text-to-video diffusion models, allowing maximum reuse of their powerful generative priors in a parameter-efficient manner. Benefiting from these designs, we scale up Omni-Video 2 to a 14B video diffusion model on meticulously curated training data with quality, supporting high quality text-to-video generation and various video editing tasks such as object removal, addition, background change, complex motion editing, \emph{etc.} We evaluate the performance of Omni-Video 2 on the FiVE benchmark for fine-grained video editing and the VBench benchmark for text-to-video generation. The results demonstrate its superior ability to follow complex compositional instructions in video editing, while also achieving competitive or superior quality in video generation tasks.
>
---
#### [replaced 031] AWPD: Frequency Shield Network for Agnostic Watermark Presence Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.06723](https://arxiv.org/pdf/2603.06723)**

> **作者:** Xiang Ao; Yiling Du; Zidan Wang; Mengru Chen; Siyang Lu
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Invisible watermarks, as an essential technology for image copyright protection, have been widely deployed with the rapid development of social media and AIGC. However, existing invisible watermark detection heavily relies on prior knowledge of specific algorithms, leading to limited detection capabilities for ``unknown watermarks'' in open environments. To this end, we propose a novel task named Agnostic Watermark Presence Detection (AWPD), which aims to identify whether an image carries a copyright mark without requiring decoding information. We construct the UniFreq-100K dataset, comprising large-scale samples across various invisible watermark embedding algorithms. Furthermore, we propose the Frequency Shield Network (FSNet). This model deploys an Adaptive Spectral Perception Module (ASPM) in the shallow layers, utilizing learnable frequency gating to dynamically amplify high-frequency watermark signals while suppressing low-frequency semantics. In the deep layers, the network introduces Dynamic Multi-Spectral Attention (DMSA) combined with tri-stream extremum pooling to deeply mine watermark energy anomalies, forcing the model to precisely focus on sensitive frequency bands. Extensive experiments demonstrate that FSNet exhibits superior zero-shot detection capabilities on the AWPD task, outperforming existing baseline models. Code and datasets will be released upon acceptance.
>
---
#### [replaced 032] Cross Pseudo Labeling For Weakly Supervised Video Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17077](https://arxiv.org/pdf/2602.17077)**

> **作者:** Dayeon Lee; Donghyeong Kim; Chaewon Park; Sungmin Woo; Sangyoun Lee
>
> **备注:** ICASSP 2026, this https URL
>
> **摘要:** Weakly supervised video anomaly detection aims to detect anomalies and identify abnormal categories with only video-level labels. We propose CPL-VAD, a dual-branch framework with cross pseudo labeling. The binary anomaly detection branch focuses on snippet-level anomaly localization, while the category classification branch leverages vision-language alignment to recognize abnormal event categories. By exchanging pseudo labels, the two branches transfer complementary strengths, combining temporal precision with semantic discrimination. Experiments on XD-Violence and UCF-Crime demonstrate that CPL-VAD achieves state-of-the-art performance in both anomaly detection and abnormal category classification.
>
---
#### [replaced 033] Let Your Image Move with Your Motion! -- Implicit Multi-Object Multi-Motion Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01000](https://arxiv.org/pdf/2603.01000)**

> **作者:** Yuze Li; Dong Gong; Xiao Cao; Junchao Yuan; Dongsheng Li; Lei Zhou; Yun Sing Koh; Cheng Yan; Xinyu Zhang
>
> **备注:** 15 pages, 11 figures, cvpr 2026, see this https URL
>
> **摘要:** Motion transfer has emerged as a promising direction for controllable video generation, yet existing methods largely focus on single-object scenarios and struggle when multiple objects require distinct motion patterns. In this work, we present FlexiMMT, the first implicit image-to-video (I2V) motion transfer framework that explicitly enables multi-object, multi-motion transfer. Given a static multi-object image and multiple reference videos, FlexiMMT independently extracts motion representations and accurately assigns them to different objects, supporting flexible recombination and arbitrary motion-to-object mappings. To address the core challenge of cross-object motion entanglement, we introduce a Motion Decoupled Mask Attention Mechanism that uses object-specific masks to constrain attention, ensuring that motion and text tokens only influence their designated regions. We further propose a Differentiated Mask Propagation Mechanism that derives object-specific masks directly from diffusion attention and progressively propagates them across frames efficiently. Extensive experiments demonstrate that FlexiMMT achieves precise, compositional, and state-of-the-art performance in I2V-based multi-object multi-motion transfer. Our project page is: this https URL
>
---
#### [replaced 034] Distilling the Past: Information-Dense and Style-Aware Replay for Lifelong Person Re-Identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01587](https://arxiv.org/pdf/2508.01587)**

> **作者:** Mingyu Wang; Wei Jiang; Haojie Liu; Zhiyong Li; Q. M. Jonathan Wu
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Lifelong person re-identification (LReID) aims to continuously adapt to new domains while mitigating catastrophic forgetting. While replay-based methods effectively alleviate forgetting, they are constrained by strict memory budgets, leading to limited sample diversity. Conversely, exemplar-free approaches bypass memory constraints entirely but struggle to preserve the fine-grained identity semantics crucial for Re-ID tasks. To resolve this fundamental dilemma, we propose an Information-Dense and Style-Aware Replay framework. Instead of storing a sparse set of raw historical images, we fuse the knowledge of sequential data into the pixel space of a compact replay buffer via multi-stage gradient matching and identity supervision. This condensation process not only maximizes the semantic representativeness of limited memory but also naturally conceals original visual details, inherently preserving data privacy. Furthermore, to combat forgetting induced by cross-domain shifts, we introduce a dual-alignment style replay strategy that adapts both current and fused replay samples, harmonizing feature representations across disparate domains. Extensive experiments on multiple LReID benchmarks demonstrate that our method significantly outperforms existing approaches, achieving improvements of +5.0% and +6.0% in Seen-Avg mAP over current state-of-the-art and traditional replay-based methods, respectively, thereby establishing an efficient and robust new baseline for lifelong learning.
>
---
#### [replaced 035] Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12078](https://arxiv.org/pdf/2603.12078)**

> **作者:** Hiran Sarkar; Liming Kuang; Yordanka Velikova; Benjamin Busam
>
> **备注:** Accepted to CVPR 2026. 13 pages, 9 figures
>
> **摘要:** Predicting scene dynamics from visual observations is challenging. Existing methods capture dynamics only within observed boundaries failing to extrapolate far beyond the training sequence. Node-RF (Neural ODE-based NeRF) overcomes this limitation by integrating Neural Ordinary Differential Equations (NODEs) with dynamic Neural Radiance Fields (NeRFs), enabling a continuous-time, spatiotemporal representation that generalizes beyond observed trajectories at constant memory cost. From visual input, Node-RF learns an implicit scene state that evolves over time via an ODE solver, propagating feature embeddings via differential calculus. A NeRF-based renderer interprets calculated embeddings to synthesize arbitrary views for long-range extrapolation. Training on multiple motion sequences with shared dynamics allows for generalization to unseen conditions. Our experiments demonstrate that Node-RF can characterize abstract system behavior without explicit model to identify critical points for future predictions.
>
---
#### [replaced 036] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13172](https://arxiv.org/pdf/2602.13172)**

> **作者:** Chong Cheng; Xianda Chen; Tao Xie; Wei Yin; Weiqiang Ren; Qian Zhang; Xiaoyang Guo; Hao Wang
>
> **备注:** CVPR2026 accepted
>
> **摘要:** Long-sequence streaming 3D reconstruction remains a significant open challenge. Existing autoregressive models often fail when processing long sequences because they anchor poses to the first frame, leading to attention decay, scale drift, and extrapolation errors. We introduce LongStream, a novel gauge-decoupled streaming visual geometry model for metric-scale scene reconstruction across thousands of frames under a strictly online, future-invisible setting. Our approach is threefold. First, we discard the first-frame anchor and predict keyframe-relative poses. This reformulates long-range extrapolation into a constant-difficulty local task. Second, we introduce orthogonal scale learning. This method fully disentangles geometry from scale estimation to suppress drift. Finally, we identify attention bias issues in Transformers, including attention-sink reliance and long-term KV-cache saturation. We propose cache-consistent training combined with periodic cache refresh. This approach suppresses attention biases and contamination over ultra-long sequences and reduces the gap between training and inference. Experiments show that LongStream achieves state-of-the-art performance, enabling stable, metric-scale reconstruction over kilometer-scale sequences at 18 FPS. Project Page: this https URL
>
---
#### [replaced 037] Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.18632](https://arxiv.org/pdf/2510.18632)**

> **作者:** Zhangquan Chen; Manyuan Zhang; Xinlei Yu; Xufang Luo; Mingze Sun; Zihao Pan; Xiang An; Yan Feng; Peng Pei; Xunliang Cai; Ruqi Huang
>
> **备注:** 25 pages, 17 figures
>
> **摘要:** Though recent advances in vision-language models (VLMs) have achieved remarkable progress across a wide range of multimodal tasks, understanding 3D spatial relationships from limited views remains a significant challenge. Previous reasoning methods typically rely on pure text (e.g., topological cognitive maps) or on 2D visual cues. However, their limited representational capacity hinders performance in specific tasks that require 3D spatial imagination. To address this limitation, we propose 3DThinker, a framework that can effectively exploits the rich geometric information embedded within images while reasoning, like humans do. Our framework is the first to enable 3D mentaling during reasoning without any 3D prior input, and it does not rely on explicitly labeled 3D data for training. Specifically, our training consists of two stages. First, we perform supervised training to align the 3D latent generated by VLM while reasoning with that of a 3D foundation model (e.g., VGGT). Then, we optimize the entire reasoning trajectory solely based on outcome signals, thereby refining the underlying 3D mentaling. Extensive experiments across multiple benchmarks show that 3DThinker consistently outperforms strong baselines and offers a new perspective toward unifying 3D representations into multimodal reasoning. Our code is available at this https URL.
>
---
#### [replaced 038] Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Epsilon-Scheduling
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23325](https://arxiv.org/pdf/2509.23325)**

> **作者:** Jonas Ngnawé; Maxime Heuillet; Sabyasachi Sahoo; Yann Pequignot; Ola Ahmad; Audrey Durand; Frédéric Precioso; Christian Gagné
>
> **备注:** 10 pages, 7 figures, 4 tables
>
> **摘要:** Fine-tuning pretrained models is a standard and effective workflow in modern machine learning. However, robust fine-tuning (RFT), which aims to simultaneously achieve adaptation to a downstream task and robustness to adversarial examples, remains challenging. Despite the abundance of non-robust pretrained models in open-source repositories, their potential for RFT is less understood. We address this knowledge gap by systematically examining RFT from such non-robust models. Our experiments reveal that fine-tuning non-robust models with a robust objective, even under small perturbations, can lead to poor performance, a phenomenon that we dub suboptimal transfer. In challenging scenarios (eg, difficult tasks, high perturbation), the resulting performance can be so low that it may be considered a transfer failure. We find that fine-tuning using a robust objective impedes task adaptation at the beginning of training and eventually prevents optimal transfer. However, we propose a novel heuristic, Epsilon-Scheduling, a schedule over perturbation strength used during training that promotes optimal transfer. Additionally, we introduce expected robustness, a metric that captures performance across a range of perturbations, providing a more comprehensive evaluation of the accuracy-robustness trade-off for diverse models at test time. Extensive experiments on a wide range of configurations (six pretrained models and five datasets) show that Epsilon-Scheduling successfully prevents suboptimal transfer and consistently improves expected robustness.
>
---
#### [replaced 039] JigsawComm: Joint Semantic Feature Encoding and Transmission for Communication-Efficient Cooperative Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17843](https://arxiv.org/pdf/2511.17843)**

> **作者:** Chenyi Wang; Zhaowei Li; Ming F. Li; Wujie Wen
>
> **摘要:** Multi-agent cooperative perception (CP) promises to overcome the inherent occlusion and range limitations of single-agent systems in autonomous driving, yet its practicality is severely constrained by limited Vehicle-to-Everything (V2X) communication bandwidth. Existing approaches attempt to improve bandwidth efficiency via compression or heuristic message selection, but neglect the semantic relevance and cross-agent redundancy of the transmitted data. In this paper, we formulate a joint semantic feature encoding and transmission problem that maximizes CP accuracy under a communication budget, and introduce JigsawComm, an end-to-end semantic-aware framework that learns to ``assemble the puzzle'' of multi-agent feature transmission. JigsawComm uses a regularized encoder to extract \emph{sparse, semantically relevant features}, and a lightweight Feature Utility Estimator (FUE) to predict each agent's per-cell contribution to the downstream perception task. The FUE-generated compact meta utility maps are exchanged among agents and used to compute an optimal transmission policy under the learned utility proxy. This policy inherently \emph{eliminates cross-agent redundancy}, bounding the feature transmission payload to $\mathcal{O}(1)$ as the number of agents grows, while the meta information overhead remains negligible. The whole pipeline is trained end-to-end through a differentiable scheduling module, informing the FUE to be aligned with the task objective. On the OPV2V and DAIR-V2X benchmarks, JigsawComm reduces total data volume by over 20--500${\times}$ while matching or exceeding the accuracy of state-of-the-art methods.
>
---
#### [replaced 040] AnatomiX, an Anatomy-Aware Grounded Multimodal Large Language Model for Chest X-Ray Interpretation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.03191](https://arxiv.org/pdf/2601.03191)**

> **作者:** Anees Ur Rehman Hashmi; Numan Saeed; Christoph Lippert
>
> **摘要:** Multimodal medical large language models have shown substantial progress in chest X-ray interpretation but continue to face challenges in spatial reasoning and anatomical understanding. Although existing grounding techniques improve overall performance, they often fail to establish a true anatomical correspondence, resulting in incorrect anatomical understanding in the medical domain. To address this gap, we introduce AnatomiX, a multitask multimodal large language model for anatomically grounded chest X-ray interpretation. Inspired by the radiological workflow, AnatomiX adopts a two stage approach: first, it identifies anatomical structures and extracts their features, and then leverages a large language model to perform diverse downstream tasks such as phrase grounding, report generation, visual question answering, and image understanding. Extensive experiments across multiple benchmarks demonstrate that AnatomiX achieves superior anatomical reasoning and delivers over 25% improvement in performance on anatomy grounding, phrase grounding, grounded diagnosis and grounded captioning tasks compared to existing approaches. Code and pretrained model are available at this http URL.
>
---
#### [replaced 041] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.06688](https://arxiv.org/pdf/2603.06688)**

> **作者:** Zhengjian Yao; Yongzhi Li; Xinyuan Gao; Quan Chen; Peng Jiang; Yanye Lu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** We present "Narrative Weaver", a novel framework that addresses a fundamental challenge in generative AI: achieving multi-modal controllable, long-range, and consistent visual content generation. While existing models excel at generating high-fidelity short-form visual content, they struggle to maintain narrative coherence and visual consistency across extended sequences - a critical limitation for real-world applications such as filmmaking and e-commerce advertising. Narrative Weaver introduces the first holistic solution that seamlessly integrates three essential capabilities: fine-grained control, automatic narrative planning, and long-range coherence. Our architecture combines a Multimodal Large Language Model (MLLM) for high-level narrative planning with a novel fine-grained control module featuring a dynamic Memory Bank that prevents visual drift. To enable practical deployment, we develop a progressive, multi-stage training strategy that efficiently leverages existing pre-trained models, achieving state-of-the-art performance even with limited training data. Recognizing the absence of suitable evaluation benchmarks, we construct and release the E-commerce Advertising Video Storyboard Dataset (EAVSD) - the first comprehensive dataset for this task, containing over 330K high-quality images with rich narrative annotations. Through extensive experiments across three distinct scenarios (controllable multi-scene generation, autonomous storytelling, and e-commerce advertising), we demonstrate our method's superiority while opening new possibilities for AI-driven content creation.
>
---
#### [replaced 042] SPRig: Self-Supervised Pose-Invariant Rigging from Mesh Sequences
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2602.12740](https://arxiv.org/pdf/2602.12740)**

> **作者:** Ruipeng Wang; Langkun Zhong; Miaowei Wang
>
> **备注:** Code: this https URL
>
> **摘要:** State-of-the-art rigging methods typically assume a predefined canonical rest pose. However, this assumption does not hold for dynamic mesh sequences such as DyMesh or DT4D, where no canonical T-pose is available. When applied independently frame-by-frame, existing methods lack pose invariance and often yield temporally inconsistent topologies. To address this limitation, we propose SPRig, a general fine-tuning framework that enforces cross-frame consistency across a sequence to learn pose-invariant rigs on top of existing models, covering both skeleton and skinning generation. For skeleton generation, we introduce novel consistency regularization in both token space and geometry space. For skinning, we improve temporal stability through an articulation-invariant consistency loss combined with consistency distillation and structural regularization. Extensive experiments show that SPRig achieves superior temporal coherence and significantly reduces artifacts in prior methods, without sacrificing and often even enhancing per-frame static generation quality. The code is available in the supplemental material and will be made publicly available upon publication.
>
---
#### [replaced 043] Understanding Dataset Distillation via Spectral Filtering
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.01212](https://arxiv.org/pdf/2503.01212)**

> **作者:** Deyu Bo; Songhua Liu; Xinchao Wang
>
> **备注:** Accepted by ICLR 2026. Code is available at this https URL
>
> **摘要:** Dataset distillation (DD) has emerged as a promising approach to compress datasets and speed up model training. However, the underlying connections among various DD methods remain largely unexplored. In this paper, we introduce UniDD, a spectral filtering framework that unifies diverse DD objectives. UniDD interprets each DD objective as a specific filter function that affects the eigenvalues of the feature-feature correlation (FFC) matrix and modulates the frequency components of the feature-label correlation (FLC) matrix. In this way, UniDD reveals that the essence of DD fundamentally lies in matching frequency-specific features. Moreover, according to the filter behaviors, we classify existing methods into low-frequency matching and high-frequency matching, encoding global texture and local details, respectively. However, existing methods rely on fixed filter functions throughout distillation, which cannot capture the low- and high-frequency information simultaneously. To address this limitation, we further propose Curriculum Frequency Matching (CFM), which gradually adjusts the filter parameter to cover both low- and high-frequency information of the FFC and FLC matrices. Extensive experiments on small-scale datasets, such as CIFAR-10/100, and large-scale datasets, including ImageNet-1K, demonstrate the superior performance of CFM over existing baselines and validate the practicality of UniDD.
>
---
#### [replaced 044] 3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for Indoor 3D Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.01647](https://arxiv.org/pdf/2410.01647)**

> **作者:** Yang Cao; Yuanliang Ju; Dan Xu
>
> **备注:** The code and models will be made publicly available upon acceptance at: \href{this https URL}{this https URL}
>
> **摘要:** Neural Radiance Fields (NeRF) have been adapted for indoor 3D Object Detection (3DOD), offering a promising approach to indoor 3DOD via view-synthesis representation. But its implicit nature limits representational capacity. Recently, 3D Gaussian Splatting (3DGS) has emerged as an explicit 3D representation that addresses the limitation. This work introduces 3DGS into indoor 3DOD for the first time, identifying two main challenges: (i) Ambiguous spatial distribution of Gaussian blobs -- 3DGS primarily relies on 2D pixel-level supervision, resulting in unclear 3D spatial distribution of Gaussian blobs and poor differentiation between objects and background, which hinders indoor 3DOD; (ii) Excessive background blobs -- 2D images typically include numerous background pixels, leading to densely reconstructed 3DGS with many noisy Gaussian blobs representing the background, negatively affecting detection. To tackle (i), we leverage the fact that 3DGS reconstruction is derived from 2D images, and propose an elegant solution by incorporating 2D Boundary Guidance to significantly enhance the spatial distribution of Gaussian blobs, resulting in clearer differentiation between objects and their background (please see fig:teaser). To address (ii), we propose a Box-Focused Sampling strategy using 2D boxes to generate object probability distribution in 3D space, allowing effective probabilistic sampling in 3D to retain more object blobs and reduce noisy background blobs. Benefiting from these innovations, 3DGS-DET significantly outperforms the state-of-the-art NeRF-based method, NeRF-Det++, achieving improvements of +6.0 on mAP@0.25 and +7.8 on mAP@0.5 for the ScanNet, and the +14.9 on mAP@0.25 for the ARKITScenes.
>
---
#### [replaced 045] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.14327](https://arxiv.org/pdf/2508.14327)**

> **作者:** Guile Wu; David Huang; Dongfeng Bai; Bingbing Liu
>
> **备注:** CVPR 2026 Findings Track
>
> **摘要:** Urban scene synthesis with video generation models has recently shown great potential for autonomous driving. Existing video generation approaches to autonomous driving primarily focus on RGB video generation and lack the ability to support multi-modal video generation. However, multi-modal data, such as depth maps and semantic maps, are crucial for holistic urban scene understanding in autonomous driving. Although it is feasible to use multiple models to generate different modalities, this increases the difficulty of model deployment and does not leverage complementary cues for multi-modal data generation. To address this problem, in this work, we propose a novel multi-modal multi-view video generation approach to autonomous driving. Specifically, we construct a unified diffusion transformer model composed of modal-shared components and modal-specific components. Then, we leverage diverse conditioning inputs to encode controllable scene structure and content cues into the multi-modal multi-view unified diffusion model. In this way, our approach is capable of generating multi-modal multi-view driving scene videos in a unified framework. Our thorough experiments on real-world autonomous driving dataset show that our approach achieves compelling video generation quality and controllability compared with state-of-the-art methods, while supporting multi-modal multi-view data generation.
>
---
#### [replaced 046] TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08096](https://arxiv.org/pdf/2603.08096)**

> **作者:** Bryce Grant; Aryeh Rothenberg; Atri Banerjee; Peng Wang
>
> **摘要:** Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically plausible but geometrically inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at this https URL.
>
---
#### [replaced 047] VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VIGS-SLAM，属于视觉惯性SLAM任务，解决复杂环境下定位与建图问题，通过融合视觉与惯性信息实现高精度实时跟踪与重建。**

- **链接: [https://arxiv.org/pdf/2512.02293](https://arxiv.org/pdf/2512.02293)**

> **作者:** Zihan Zhu; Wei Zhang; Moyang Li; Norbert Haala; Marc Pollefeys; Daniel Barath
>
> **备注:** Project page: this https URL
>
> **摘要:** We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under challenging conditions such as motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly optimizing camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on five challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: this https URL
>
---
#### [replaced 048] SvfEye: A Semantic-Visual Fusion Framework with Multi-Scale Visual Context for Multimodal Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.00171](https://arxiv.org/pdf/2603.00171)**

> **作者:** Yuxiang Shen; Hailong Huang; Zhenkun Gao; Xueheng Li; Man Zhou; Chengjun Xie; Haoxuan Che; Xuanhua He; Jie Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) often struggle to accurately perceive fine-grained visual details, especially when targets are tiny or visually subtle. This challenge can be addressed through semantic-visual information fusion, which integrates global image context with fine-grained local evidence for multi-scale visual understanding. Recently, a paradigm termed "Thinking with Images" has emerged, enabling models to acquire high-resolution visual evidence by zooming or cropping image regions and fusing these local details with global context during reasoning. Although training-based approaches demonstrate the effectiveness of this capability, they require extensive computational resources and large-scale task-specific data. Consequently, lightweight training-free methods have been proposed as a practical alternative to incorporate local visual evidence during inference. However, existing training-free approaches still suffer from two key limitations. First, they indiscriminately extract and fuse local visual regions for all inputs regardless of necessity, introducing computational redundancy and perceptual noise. Second, they exhibit drift between semantic intent and visual attention, preventing accurate localization of user-focused regions. To address these challenges, we propose SvfEye, a training-free framework for adaptive visual-semantic fusion. SvfEye follows a two-stage pipeline with a confidence-based decision module to determine whether additional local visual information is needed, and a semantic-attention fusion module to identify informative local regions. Experiments show that SvfEye achieves substantial performance gains while obtaining an approximately 4.0x inference speedup over the state-of-the-art method ZoomEye.
>
---
#### [replaced 049] MIND-V: Hierarchical World Model for Long-Horizon Robotic Manipulation with RL-based Physical Alignment
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MIND-V，解决长时程机器人操作数据生成问题。通过层次化模型合成物理合理视频，提升策略学习效果。**

- **链接: [https://arxiv.org/pdf/2512.06628](https://arxiv.org/pdf/2512.06628)**

> **作者:** Ruicheng Zhang; Mingyang Zhang; Jun Zhou; Zhangrui Guo; Zunnan Xu; Xiaofan Liu; Zhizhou Zhong; Puxin Yan; Haocheng Luo; Xiu Li
>
> **摘要:** Scalable embodied intelligence is constrained by the scarcity of diverse, long-horizon robotic manipulation data. Existing video world models in this domain are limited to synthesizing short clips of simple actions and often rely on manually defined trajectories. To this end, we introduce MIND-V, a cognitive hierarchical world model designed to synthesize physically plausible and logically coherent videos of long-horizon robotic manipulation. Inspired by cognitive science, MIND-V bridges high-level reasoning with pixel-level synthesis through three core components: a Semantic Reasoning Hub (SRH) that leverages a pre-trained vision-language model for task planning; a Behavioral Semantic Bridge (BSB) that translates abstract instructions into domain-invariant representations; and a Motor Video Generator (MVG) for conditional video rendering. MIND-V employs Staged Visual Future Rollouts, a test-time optimization strategy to enhance long-horizon robustness. To enforce adherence to physical laws, we introduce a GRPO reinforcement learning post-training phase guided by a novel Physical Foresight Coherence (PFC) reward. PFC leverages the V-JEPA2 world model as a physics referee to penalize implausible dynamics in the latent feature space. Experiments confirm MIND-V's SOTA performance in long-horizon simulation and its significant value for policy learning, introducing a scalable and fully autonomous framework for embodied data synthesis.
>
---
#### [replaced 050] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18507](https://arxiv.org/pdf/2511.18507)**

> **作者:** Kai Jiang; Siqi Huang; Xiangyu Chen; Jiawei Shao; Hongyuan Zhang; Ping Luo; Xuelong Li
>
> **备注:** 22 pages, 17 figures. This is a preprint version of a paper submitted to ICML 2026
>
> **摘要:** Multimodal large language models (MLLMs) deployed on devices must adapt to continuously changing visual scenarios such as variations in background and perspective, to effectively perform complex visual tasks. To investigate catastrophic forgetting under real-world scenario shifts, we construct a multimodal visual understanding dataset (MSVQA), covering four distinct scenarios and perspectives: high-altitude, underwater, low-altitude, and indoor environments. Furthermore, we propose UNIFIER (mUltimodal coNtInual learning with MLLMs From multi-scenarIo pERspectives), a continual learning (CL) framework designed to address visual discrepancies while learning different scenarios. Compared to existing CL methods, UNIFIER enables knowledge accumulation within the same scenario and mutual enhancement across different scenarios via Vision Representation Expansion (VRE) and Vision Consistency Constraint (VCC). Experimental results show that UNIFIER improves the last-step VQA scores by 2.70%~10.62% and the last-step F1 scores by 3.40%~7.69% compared to the state-of-the-art method, QUAD, in 20-step cross-scenario continual learning tasks. MSVQA dataset is available at this https URL.
>
---
#### [replaced 051] PCA-Enhanced Probabilistic U-Net for Effective Ambiguous Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11550](https://arxiv.org/pdf/2603.11550)**

> **作者:** Xiangyu Li; Chenglin Wang; Qiantong Shen; Fanding Li; Wei Wang; Kuanquan Wang; Yi Shen; Baochun Zhao; Gongning Luo
>
> **摘要:** Ambiguous Medical Image Segmentation (AMIS) is significant to address the challenges of inherent uncertainties from image ambiguities, noise, and subjective annotations. Existing conditional variational autoencoder (cVAE)-based methods effectively capture uncertainty but face limitations including redundancy in high-dimensional latent spaces and limited expressiveness of single posterior networks. To overcome these issues, we introduce a novel PCA-Enhanced Probabilistic U-Net (PEP U-Net). Our method effectively incorporates Principal Component Analysis (PCA) for dimensionality reduction in the posterior network to mitigate redundancy and improve computational efficiency. Additionally, we further employ an inverse PCA operation to reconstruct critical information, enhancing the latent space's representational capacity. Compared to conventional generative models, our method preserves the ability to generate diverse segmentation hypotheses while achieving a superior balance between segmentation accuracy and predictive variability, thereby advancing the performance of generative modeling in medical image segmentation.
>
---
#### [replaced 052] DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17354](https://arxiv.org/pdf/2511.17354)**

> **作者:** Xiangteng He; Shunsuke Sakai; Shivam Chandhok; Sara Beery; Kun Yuan; Nicolas Padoy; Tatsuhito Hasegawa; Leonid Sigal
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in self-supervised visual representation learning have demonstrated the effectiveness of predictive latent-space objectives for learning transferable features. In particular, Image-based Joint-Embedding Predictive Architecture (I-JEPA) learns representations by predicting latent embeddings of masked target regions from visible context. However, it predicts target regions in parallel and all at once, lacking ability to order predictions meaningfully. Inspired by human visual perception, which attends selectively and progressively from primary to secondary cues, we propose DSeq-JEPA, a Discriminative Sequential Joint-Embedding Predictive Architecture that bridges latent predictive and autoregressive self-supervised learning. Specifically, DSeq-JEPA integrates a discriminatively ordered sequential process with JEPA-style learning objective. This is achieved by (i) identifying primary discriminative regions using an attention-derived saliency map that serves as a proxy for visual importance, and (ii) predicting subsequent regions in discriminative order, inducing a curriculum-like semantic progression from primary to secondary cues in pre-training. Extensive experiments across tasks -- image classification (ImageNet), fine-grained visual categorization (iNaturalist21, CUB, Stanford Cars), detection/segmentation (MS-COCO, ADE20K), and low-level reasoning (CLEVR) -- show that DSeq-JEPA consistently learns more discriminative and generalizable representations compared to I-JEPA variants. Project page: this https URL.
>
---
#### [replaced 053] NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.17914](https://arxiv.org/pdf/2510.17914)**

> **作者:** Rikard Vinge; Isabelle Wittmann; Jannik Schneider; Michael Marszalek; Luis Gilch; Thomas Brunschwiler; Conrad M Albrecht
>
> **摘要:** We introduce NeuCo-Bench, a novel benchmark framework for evaluating (lossy) neural compression and representation learning in the context of Earth Observation (EO). Our approach builds on fixed-size embeddings that act as compact, task-agnostic representations applicable to a broad range of downstream tasks. NeuCo-Bench comprises three components: (i) an evaluation pipeline built around embeddings, (ii) a challenge mode with a hidden-task leaderboard designed to mitigate pretraining bias, and (iii) a scoring system that balances accuracy and stability. To support reproducibility, we release SSL4EO-S12-downstream, a curated multispectral, multitemporal EO dataset. We present results from a public challenge at the 2025 CVPR EARTHVISION workshop and conduct ablations with state-of-the-art foundation models. NeuCo-Bench provides a step towards community-driven, standardized evaluation of neural embeddings for EO and beyond.
>
---
#### [replaced 054] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21662](https://arxiv.org/pdf/2511.21662)**

> **作者:** Tianyi Xiong; Yi Ge; Ming Li; Zuolong Zhang; Pranav Kulkarni; Kaishen Wang; Qi He; Zeying Zhu; Chenxi Liu; Ruibo Chen; Tong Zheng; Yanshuo Chen; Xiyao Wang; Renrui Zhang; Wenhu Chen; Heng Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Large multimodal models (LMMs) are increasingly adopted as judges in multimodal evaluation systems due to their strong instruction following and consistency with human preferences. However, their ability to follow diverse, fine-grained evaluation criteria remains underexplored. We develop Multi-Crit, a benchmark for evaluating multimodal judges on their capacity to follow pluralistic criteria and produce reliable criterion-level judgments. Covering both open-ended generation and verifiable reasoning tasks, Multi-Crit is built through a rigorous data curation pipeline that gathers challenging response pairs with multi-criterion human annotations. It further introduces three novel metrics for systematically assessing pluralistic adherence, criterion-switching flexibility, and the ability to recognize criterion-level preference conflicts. Comprehensive analysis of 25 LMMs reveals that 1) proprietary models still struggle to maintain consistent adherence to pluralistic criteria--especially in open-ended evaluation; 2) open-source models lag further behind in flexibly following diverse criteria; and 3) critic fine-tuning with holistic judgment signals enhances visual grounding but fails to generalize to pluralistic criterion-level judgment. Additional analyses on reasoning fine-tuning, test-time scaling, and boundary consistency between open-source and proprietary models further probe the limits of current multimodal judges. As a pioneering study, Multi-Crit lays the foundation for building reliable and steerable multimodal AI evaluation.
>
---
#### [replaced 055] From Video to EEG: Adapting Joint Embedding Predictive Architecture to Uncover Saptiotemporal Dynamics in Brain Signal Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.03633](https://arxiv.org/pdf/2507.03633)**

> **作者:** Amirabbas Hojjati; Lu Li; Ibrahim Hameed; Anis Yazidi; Pedro G. Lind; Rabindra Khadka
>
> **摘要:** EEG signals capture brain activity with high temporal and low spatial resolution, supporting applications such as neurological diagnosis, cognitive monitoring, and brain-computer interfaces. However, effective analysis is hindered by limited labeled data, high dimensionality, and the absence of scalable models that fully capture spatiotemporal dependencies. Existing self-supervised learning (SSL) methods often focus on either spatial or temporal features, leading to suboptimal representations. To this end, we propose EEG-VJEPA, a novel adaptation of the Video Joint Embedding Predictive Architecture (V-JEPA) for EEG classification. By treating EEG as video-like sequences, EEG-VJEPA learns semantically meaningful spatiotemporal representations using joint embeddings and adaptive masking. To our knowledge, this is the first work that exploits V-JEPA for EEG classification and explores the visual concepts learned by the model. Evaluations on the publicly available Temple University Hospital (TUH) Abnormal EEG dataset show that EEG-VJEPA outperforms existing state-of-the-art models in classification accuracy. Beyond classification accuracy, EEG-VJEPA captures physiologically relevant spatial and temporal signal patterns, offering interpretable embeddings that may support human-AI collaboration in diagnostic workflows. These findings position EEG-VJEPA as a promising framework for scalable, trustworthy EEG analysis in real-world clinical settings.
>
---
#### [replaced 056] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07057](https://arxiv.org/pdf/2603.07057)**

> **作者:** Tong Shao; Yusen Fu; Guoying Sun; Jingde Kong; Zhuotao Tian; Jingyong Su
>
> **备注:** 23 pages, CVPR 2026 accepted
>
> **摘要:** Diffusion Transformers have become a dominant paradigm in visual generation, yet their low inference efficiency remains a key bottleneck hindering further advancement. Among common training-free techniques, caching offers high acceleration efficiency but often compromises fidelity, whereas pruning shows the opposite trade-off. Integrating caching with pruning achieves a balance between acceleration and generation quality. However, existing methods typically employ fixed and heuristic schemes to configure caching and pruning strategies. While they roughly follow the overall sensitivity trend of generation models to acceleration, they fail to capture fine-grained and complex variations, inevitably skipping highly sensitive computations and leading to quality degradation. Furthermore, such manually designed strategies exhibit poor generalization. To address these issues, we propose SODA, a Sensitivity-Oriented Dynamic Acceleration method that adaptively performs caching and pruning based on fine-grained sensitivity. SODA builds an offline sensitivity error modeling framework across timesteps, layers, and modules to capture the sensitivity to different acceleration operations. The cache intervals are optimized via dynamic programming with sensitivity error as the cost function, minimizing the impact of caching on model sensitivity. During pruning and cache reuse, SODA adaptively determines the pruning timing and rate to preserve computations of highly sensitive tokens, significantly enhancing generation fidelity. Extensive experiments on DiT-XL/2, PixArt-$\alpha$, and OpenSora demonstrate that SODA achieves state-of-the-art generation fidelity under controllable acceleration ratios. Our code is released publicly at: this https URL.
>
---
#### [replaced 057] Automatic Labelling for Low-Light Pedestrian Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.02513](https://arxiv.org/pdf/2507.02513)**

> **作者:** Dimitrios Bouzoulas; Eerik Alamikkotervo; Risto Ojala
>
> **摘要:** Pedestrian detection in RGB images is a key task in pedestrian safety, as the most common sensor in autonomous vehicles and advanced driver assistance systems is the RGB camera. A challenge in RGB pedestrian detection, that does not appear to have large public datasets, is low-light conditions. As a solution, in this research, we propose an automated infrared-RGB labeling pipeline. The proposed pipeline consists of 1) Infrared detection, where a fine-tuned model for infrared pedestrian detection is used 2) Label transfer process from the infrared detections to their RGB counterparts 3) Training object detection models using the generated labels for low-light RGB pedestrian detection. The research was performed using the KAIST dataset. For the evaluation, object detection models were trained on the generated autolabels and ground truth labels. When compared on a previously unseen image sequence, the results showed that the models trained on generated labels outperformed the ones trained on ground-truth labels in 6 out of 9 cases for the mAP@50 and mAP@50-95 metrics. The source code for this research is available at this https URL
>
---
#### [replaced 058] PatchCue: Enhancing Vision-Language Model Reasoning with Patch-Based Visual Cues
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05869](https://arxiv.org/pdf/2603.05869)**

> **作者:** Yukun Qi; Pei Fu; Hang Li; Yuhan Liu; Chao Jiang; Bin Qin; Zhenbo Luo; Jian Luan
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress on a wide range of challenging multimodal understanding and reasoning tasks. However, existing reasoning paradigms, such as the classical Chain-of-Thought (CoT), rely solely on textual information and often underutilize important visual cues. While prior work has incorporated pixel-level visual cues, these representations require precise spatial localization, introducing additional learning complexity. To address this, we propose PatchCue, a novel patch-based visual cue paradigm designed to significantly enhance the visual reasoning capabilities of VLMs. By partitioning images into patches and representing cues at the patch level, PatchCue aligns better with human perceptual habits and leverages the patch-tokenized input of modern VLMs. We train VLMs using a two-stage approach: cold-start supervised fine-tuning to output patch-level cues, followed by reinforcement learning with a process-supervised cue reward that guides intermediate visual reasoning steps. Extensive experiments on multiple VLMs and diverse benchmarks, including general visual question answering, complex reasoning, and document understanding, demonstrate that PatchCue consistently improves overall model performance. Our results show that patch-level cues outperform both pixel-level bounding boxes and point-based cues, providing a more effective and cognitively aligned visual reasoning paradigm.
>
---
#### [replaced 059] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16672](https://arxiv.org/pdf/2511.16672)**

> **作者:** Omkar Thawakar; Shravan Venkatraman; Ritesh Thawkar; Abdelrahman Shaker; Hisham Cholakkal; Rao Muhammad Anwer; Salman Khan; Fahad Khan
>
> **备注:** CVPR 2026 (findings)
>
> **摘要:** Recent advances in large multimodal models (LMMs) have enabled impressive reasoning and perception abilities, yet most existing training pipelines still depend on human-curated data or externally verified reward models, limiting their autonomy and scalability. In this work, we strive to improve LMM reasoning capabilities in a purely unsupervised fashion (without any annotated data or reward distillation). To this end, we propose a self-evolving framework, named EvoLMM, that instantiates two cooperative agents from a single backbone model: a Proposer, which generates diverse, image-grounded questions, and a Solver, which solves them through internal consistency, where learning proceeds through a continuous self-rewarding process. This dynamic feedback encourages both the generation of informative queries and the refinement of structured reasoning without relying on ground-truth or human judgments. When using the popular Qwen2.5-VL as the base model, our EvoLMM yields consistent gains upto $\sim$3\% on multimodal math-reasoning benchmarks, including ChartQA, MathVista, and MathVision, using only raw training images. We hope our simple yet effective approach will serve as a solid baseline easing future research in self-improving LMMs in a fully-unsupervised fashion. Our code and models are available at this https URL.
>
---
#### [replaced 060] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.04733](https://arxiv.org/pdf/2603.04733)**

> **作者:** Xingyu Wang; Tao Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-Time Adaptation (TTA) is essential for enabling deep learning models to handle real-world data distribution shifts. However, current approaches face significant limitations: backpropagation-based methods are not suitable for low-end deployment devices, due to their high computation and memory requirements, as well as their tendency to modify model weights during adaptation; while traditional backpropagation-free techniques exhibit constrained adaptation capabilities. In this work, we propose Forward-Only Zeroth-Order Optimization (FOZO), a novel and practical backpropagation-free paradigm for TTA. FOZO leverages a memory-efficient zeroth-order prompt optimization, which is led by objectives optimizing both intermediate feature statistics and prediction entropy. To ensure efficient and stable adaptation over the out-of-distribution data stream, we introduce a dynamically decaying perturbation scale during zeroth-order gradient estimation and theoretically prove its convergence under the TTA data stream assumption. Extensive continual adaptation experiments on ImageNet-C, ImageNet-R, and ImageNet-Sketch demonstrate FOZO's superior performance, achieving 59.52% Top-1 accuracy on ImageNet-C (5K, level 5) and outperforming main gradient-based methods and SOTA forward-only FOA (58.13%). Furthermore, FOZO exhibits strong generalization on quantized (INT8) models. These findings demonstrate that FOZO is a highly competitive solution for TTA deployment in resource-limited scenarios.
>
---
#### [replaced 061] Latent diffusion models for parameterization and data assimilation of facies-based geomodels
- **分类: cs.CV; cs.AI; cs.CE; cs.LG; physics.geo-ph**

- **链接: [https://arxiv.org/pdf/2406.14815](https://arxiv.org/pdf/2406.14815)**

> **作者:** Guido Di Federico; Louis J. Durlofsky
>
> **摘要:** Geological parameterization entails the representation of a geomodel using a small set of latent variables and a mapping from these variables to grid-block properties such as porosity and permeability. Parameterization is useful for data assimilation (history matching), as it maintains geological realism while reducing the number of variables to be determined. Diffusion models are a new class of generative deep-learning procedures that have been shown to outperform previous methods, such as generative adversarial networks, for image generation tasks. Diffusion models are trained to "denoise", which enables them to generate new geological realizations from input fields characterized by random noise. Latent diffusion models, which are the specific variant considered in this study, provide dimension reduction through use of a low-dimensional latent variable. The model developed in this work includes a variational autoencoder for dimension reduction and a U-net for the denoising process. Our application involves conditional 2D three-facies (channel-levee-mud) systems. The latent diffusion model is shown to provide realizations that are visually consistent with samples from geomodeling software. Quantitative metrics involving spatial and flow-response statistics are evaluated, and general agreement between the diffusion-generated models and reference realizations is observed. Stability tests are performed to assess the smoothness of the parameterization method. The latent diffusion model is then used for ensemble-based data assimilation. Two synthetic "true" models are considered. Significant uncertainty reduction, posterior P$_{10}$-P$_{90}$ forecasts that generally bracket observed data, and consistent posterior geomodels, are achieved in both cases. PLEASE CITE AS: https://doi.org/10.1016/j.cageo.2024.105755 this https URL NOT WITH THE ARXIV VERSION
>
---
#### [replaced 062] Parameterized Prompt for Incremental Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27316](https://arxiv.org/pdf/2510.27316)**

> **作者:** Zijia An; Boyu Diao; Ruiqi Liu; Libo Huang; Chuanguang Yang; Fei Wang; Zhulin An; Yongjun Xu
>
> **摘要:** Recent studies have demonstrated that incorporating trainable prompts into pretrained models enables effective incremental learning. However, the application of prompts in incremental object detection (IOD) remains underexplored. Our study reveals that existing prompt-pool-based approaches assume disjoint class sets across incremental tasks, which are unsuitable for IOD as they overlook the inherent co-occurrence phenomenon in detection. In co-occurring scenarios, unlabeled objects from previous tasks may appear in current task images, leading to confusion in prompts pool. In this paper, we hold that prompt structures should exhibit adaptive consolidation properties across tasks, with constrained updates to prevent confusion and catastrophic forgetting. Motivated by this, we introduce Parameterized Prompts for Incremental Object Detection (P$^2$IOD). Leveraging neural networks global evolution properties, P$^2$IOD employs networks as the parameterized prompts to adaptively consolidate knowledge across tasks. To constrain prompts structure updates, P$^2$IOD further engages a parameterized prompts fusion strategy. Extensive experiments on PASCAL VOC2007 and MS COCO datasets demonstrate that P$^2$IOD's effectiveness in IOD and achieves the state-of-the-art performance among existing baselines. Code is available at this https URL.
>
---
#### [replaced 063] From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10300](https://arxiv.org/pdf/2603.10300)**

> **作者:** Ke Zhang; Xiangchen Zhao; Yunjie Tian; Jiayu Zheng; Vishal M. Patel; Di Fu
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Conventional video classification models, acting as effective imitators, excel in scenarios with homogeneous data distributions. However, real-world applications often present an open-instance challenge, where intra-class variations are vast and complex, beyond existing benchmarks. While traditional video encoder models struggle to fit these diverse distributions, vision-language models (VLMs) offer superior generalization but have not fully leveraged their reasoning capabilities (intuition) for such tasks. In this paper, we bridge this gap with an intrinsic reasoning framework that evolves open-instance video classification from imitation to intuition. Our approach, namely DeepIntuit, begins with a cold-start supervised alignment to initialize reasoning capability, followed by refinement using Group Relative Policy Optimization (GRPO) to enhance reasoning coherence through reinforcement learning. Crucially, to translate this reasoning into accurate classification, DeepIntuit then introduces an intuitive calibration stage. In this stage, a classifier is trained on this intrinsic reasoning traces generated by the refined VLM, ensuring stable knowledge transfer without distribution mismatch. Extensive experiments demonstrate that for open-instance video classification, DeepIntuit benefits significantly from transcending simple feature imitation and evolving toward intrinsic reasoning. Our project is available at this https URL.
>
---
#### [replaced 064] Mobile-VTON: High-Fidelity On-Device Virtual Try-On
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00947](https://arxiv.org/pdf/2603.00947)**

> **作者:** Zhenchen Wan; Ce Chen; Runqi Lin; Jiaxin Huang; Tianxi Chen; Yanwu Xu; Tongliang Liu; Mingming Gong
>
> **备注:** The project page is available at: this https URL
>
> **摘要:** Virtual try-on (VTON) has recently achieved impressive visual fidelity, but most existing systems require uploading personal photos to cloud-based GPUs, raising privacy concerns and limiting on-device deployment. To address this, we present Mobile-VTON, a high-quality, privacy-preserving framework that enables fully offline virtual try-on on commodity mobile devices using only a single user image and a garment image. Mobile-VTON introduces a modular TeacherNet-GarmentNet-TryonNet (TGT) architecture that integrates knowledge distillation, garment-conditioned generation, and garment alignment into a unified pipeline optimized for on-device efficiency. Within this framework, we propose a Feature-Guided Adversarial (FGA) Distillation strategy that combines teacher supervision with adversarial learning to better match real-world image distributions. GarmentNet is trained with a trajectory-consistency loss to preserve garment semantics across diffusion steps, while TryonNet uses latent concatenation and lightweight cross-modal conditioning to enable robust garment-to-person alignment without large-scale pretraining. By combining these components, Mobile-VTON achieves high-fidelity generation with low computational overhead. Experiments on VITON-HD and DressCode at 1024 x 768 show that it matches or outperforms strong server-based baselines while running entirely offline. These results demonstrate that high-quality VTON is not only feasible but also practical on-device, offering a secure solution for real-world applications. Code and project page are available at this https URL.
>
---
#### [replaced 065] Referee: Reference-aware Audiovisual Deepfake Detection
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.27475](https://arxiv.org/pdf/2510.27475)**

> **作者:** Hyemin Boo; Eunsang Lee; Jiyoung Lee
>
> **备注:** In Progress
>
> **摘要:** Deepfakes generated by advanced generative models have rapidly posed serious threats, yet existing audiovisual deepfake detection approaches struggle to generalize to unseen manipulation methods. To address this, we propose a novel reference-aware audiovisual deepfake detection method, called Referee to capture fine-grained identity discrepancies. Unlike existing methods that overfit to transient spatiotemporal artifacts, Referee employs identity bottleneck and matching modules to model the relational consistency of speaker-specific cues captured by a single one-shot example as a biometric anchor. Extensive experiments on FakeAVCeleb, FaceForensics++, and KoDF demonstrate that Referee achieves state-of-the-art results on cross-dataset and cross-language evaluation protocols, including a 99.4% AUC on KoDF. These results highlight that explicitly correlating reference-based biometric priors is a key frontier for achieving generalized and reliable audiovisual forensics. The code is available at this https URL.
>
---
#### [replaced 066] Omni-Video: Democratizing Unified Video Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06119](https://arxiv.org/pdf/2507.06119)**

> **作者:** Zhiyu Tan; Hao Yang; Luozheng Qin; Jia Gong; Mengping Yang; Hao Li
>
> **备注:** Technical report, project page: this https URL
>
> **摘要:** Notable breakthroughs in unified understanding and generation modeling have led to remarkable advancements in image understanding, reasoning, production and editing, yet current foundational models predominantly focus on processing images, creating a gap in the development of unified models for video understanding and generation. This report presents Omni-Video, an efficient and effective unified framework for video understanding, generation, as well as instruction-based editing. Our key insight is to teach existing multimodal large language models (MLLMs) to produce continuous visual clues that are used as the input of diffusion decoders, which produce high-quality videos conditioned on these visual clues. To fully unlock the potential of our system for unified video modeling, we integrate several technical improvements: 1) a lightweight architectural design that respectively attaches a vision head on the top of MLLMs and a adapter before the input of diffusion decoders, the former produce visual tokens for the latter, which adapts these visual tokens to the conditional space of diffusion decoders; and 2) an efficient multi-stage training scheme that facilitates a fast connection between MLLMs and diffusion decoders with limited data and computational resources. We empirically demonstrate that our model exhibits satisfactory generalization abilities across video generation, editing and understanding tasks.
>
---
#### [replaced 067] Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决自动驾驶中分布外检测问题。通过自适应机制建模误差模式，提升检测效果与效率。**

- **链接: [https://arxiv.org/pdf/2509.13577](https://arxiv.org/pdf/2509.13577)**

> **作者:** Tongfei Guo; Lili Su
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors -- even on in-distribution samples -- exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.
>
---
#### [replaced 068] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2603.11975](https://arxiv.org/pdf/2603.11975)**

> **作者:** Jiayue Pu; Zhongxiang Sun; Zilu Zhang; Xiao Zhang; Jun Xu
>
> **摘要:** The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce HomeSafe-Bench, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose Hierarchical Dual-Brain Guard for Household Safety (HD-Guard), a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection.
>
---
#### [replaced 069] RLM: A Vision-Language Model Approach for Radar Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21105](https://arxiv.org/pdf/2511.21105)**

> **作者:** Pushkal Mishra; Kshitiz Bansal; Dinesh Bharadia
>
> **摘要:** Radar sensors provide reliable perception across adverse weather, lighting, and long-range conditions, yet existing machine learning approaches remain fragmented and task-specific, with each downstream task employing distinct architectures and training objectives. We present RadarVLM, a vision-language framework that learns unified scene-level representations through structured spatial language supervision. Leveraging the CARLA simulator with a realistic radar model, we collect over 800k radar-caption pairs across 110+ hours of simulated driving in diverse scenarios. We make two key contributions: (1) a structured caption framework encoding vehicle distributions in the radar's native coordinate system, and (2) Spatially-Grounded CLIP (SG-CLIP) objective that replaces binary matching with continuous scene similarity, enabling fine-grained spatial reasoning. We further propose localization-aware evaluation metrics that directly assess spatial accuracy beyond traditional linguistic similarity measures. Validated on generative captioning and vehicle segmentation, SG-CLIP achieves up to 50% relative F1-score improvement over vanilla CLIP and a 21% AP gain on segmentation, demonstrating that language grounding produces spatially structured representations.
>
---
#### [replaced 070] OmniForcing: Unleashing Real-time Joint Audio-Visual Generation
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出OmniForcing，解决音频-视觉生成中的实时性问题。通过蒸馏技术将双向模型转为流式自回归生成，提升推理速度并保持多模态同步与质量。**

- **链接: [https://arxiv.org/pdf/2603.11647](https://arxiv.org/pdf/2603.11647)**

> **作者:** Yaofeng Su; Yuming Li; Zeyue Xue; Jie Huang; Siming Fu; Haoran Li; Ying Li; Zezhong Qian; Haoyang Huang; Nan Duan
>
> **备注:** 14 pages
>
> **摘要:** Recent joint audio-visual diffusion models achieve remarkable generation quality but suffer from high latency due to their bidirectional attention dependencies, hindering real-time applications. We propose OmniForcing, the first framework to distill an offline, dual-stream bidirectional diffusion model into a high-fidelity streaming autoregressive generator. However, naively applying causal distillation to such dual-stream architectures triggers severe training instability, due to the extreme temporal asymmetry between modalities and the resulting token sparsity. We address the inherent information density gap by introducing an Asymmetric Block-Causal Alignment with a zero-truncation Global Prefix that prevents multi-modal synchronization drift. The gradient explosion caused by extreme audio token sparsity during the causal shift is further resolved through an Audio Sink Token mechanism equipped with an Identity RoPE constraint. Finally, a Joint Self-Forcing Distillation paradigm enables the model to dynamically self-correct cumulative cross-modal errors from exposure bias during long rollouts. Empowered by a modality-independent rolling KV-cache inference scheme, OmniForcing achieves state-of-the-art streaming generation at $\sim$25 FPS on a single GPU, maintaining multi-modal synchronization and visual quality on par with the bidirectional teacher.\textbf{Project Page:} \href{this https URL}{this https URL}
>
---
#### [replaced 071] LowDiff: Efficient Diffusion Sampling with Low-Resolution Condition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.15342](https://arxiv.org/pdf/2509.15342)**

> **作者:** Jiuyi Xu; Qing Jin; Meida Chen; Andrew Feng; Yang Sui; Yangming Shi
>
> **备注:** 16 pages, 7 figures, 12 tables
>
> **摘要:** Diffusion models have achieved remarkable success in image generation but their practical application is often hindered by the slow sampling speed. Prior efforts of improving efficiency primarily focus on compressing models or reducing the total number of denoising steps, largely neglecting the possibility to leverage multiple input resolutions in the generation process. In this work, we propose LowDiff, a novel and efficient diffusion framework based on a cascaded approach by generating increasingly higher resolution outputs. Besides, LowDiff employs a unified model to progressively refine images from low resolution to the desired resolution. With the proposed architecture design and generation techniques, we achieve comparable or even superior performance with much fewer high-resolution sampling steps. LowDiff is applicable to diffusion models in both pixel space and latent space. Extensive experiments on both conditional and unconditional generation tasks across CIFAR-10, FFHQ and ImageNet demonstrate the effectiveness and generality of our method. Results show over 50% throughput improvement across all datasets and settings while maintaining comparable or better quality. On unconditional CIFAR-10, LowDiff achieves an FID of 2.11 and IS of 9.87, while on conditional CIFAR-10, an FID of 1.94 and IS of 10.03. On FFHQ 64x64, LowDiff achieves an FID of 2.43, and on ImageNet 256x256, LowDiff built on LightningDiT-B/1 produces high-quality samples with a FID of 4.00 and an IS of 195.06, together with substantial efficiency gains.
>
---
#### [replaced 072] TIRAuxCloud: A Thermal Infrared Dataset for Day and Night Cloud Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21905](https://arxiv.org/pdf/2602.21905)**

> **作者:** Alexis Apostolakis; Vasileios Botsos; Niklas Wölki; Andrea Spichtinger; Nikolaos Ioannis Bountos; Ioannis Papoutsis; Panayiotis Tsanakas
>
> **摘要:** Clouds are a major obstacle in Earth observation, limiting the usability and reliability of critical remote sensing applications such as fire disaster response, urban heat island monitoring, and snow and ice cover mapping. Therefore, the ability to detect clouds 24/7 is of paramount importance. While visible and near-infrared bands are effective for daytime cloud detection, their dependence on solar illumination makes them unsuitable for nighttime monitoring. In contrast, thermal infrared (TIR) imagery plays a crucial role in detecting clouds at night, when sunlight is absent. Due to their generally lower temperatures, clouds emit distinct thermal signatures that are detectable in TIR bands. Despite this, accurate nighttime cloud detection remains challenging due to limited spectral information and the typically lower spatial resolution of TIR imagery. To address these challenges, we present TIRAuxCloud, a multi-modal dataset centered around thermal spectral data to facilitate cloud segmentation under both daytime and nighttime conditions. The dataset comprises a unique combination of multispectral data (TIR, optical, and near-infrared bands) from Landsat and VIIRS, aligned with auxiliary information layers. Elevation, land cover, meteorological variables, and cloud-free reference images are included to help reduce surface-cloud ambiguity and cloud formation uncertainty. To overcome the scarcity of manual cloud labels, we include a large set of samples with automated cloud masks and a smaller manually annotated subset to further evaluate and improve models. Comprehensive benchmarks are presented to establish performance baselines through supervised and transfer learning, demonstrating the dataset's value in advancing the development of innovative methods for day and night time cloud detection.
>
---
#### [replaced 073] MMGT: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23120](https://arxiv.org/pdf/2505.23120)**

> **作者:** Siyuan Wang; Jiawei Liu; Wei Wang; Yeying Jin; Jinsong Du; Zhi Han
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** Co-Speech Gesture Video Generation aims to generate vivid speech videos from audio-driven still images, which is challenging due to the diversity of body parts in terms of motion amplitude, audio relevance, and detailed features. Relying solely on audio as the control signal often fails to capture large gesture movements in videos, resulting in more noticeable artifacts and distortions. Existing approaches typically address this issue by adding extra prior inputs, but this can limit the practical application of the task. Specifically, we propose a Motion Mask-Guided Two-Stage Network (MMGT) that uses audio, along with motion masks and pose videos generated from the audio signal, to jointly generate synchronized speech gesture videos. In the first stage, the Spatial Mask-Guided Audio2Pose Generation (SMGA) Network generates high-quality pose videos and motion masks from audio, effectively capturing large movements in key regions such as the face and gestures. In the second stage, we integrate Motion Masked Hierarchical Audio Attention (MM-HAA) into the Stabilized Diffusion Video Generation model, addressing limitations in fine-grained motion generation and region-specific detail control found in traditional methods. This ensures high-quality, detailed upper-body videos with accurate textures and motion. Evaluations demonstrate improvements in video quality, lip-sync, and hand gestures. The model and code are available at this https URL.
>
---
#### [replaced 074] HoneyBee: Data Recipes for Vision-Language Reasoners
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.12225](https://arxiv.org/pdf/2510.12225)**

> **作者:** Hritik Bansal; Devendra Singh Sachan; Kai-Wei Chang; Aditya Grover; Gargi Ghosh; Wen-tau Yih; Ramakanth Pasunuru
>
> **备注:** 32 pages. Accepted to CVPR 2026 in Denver, Colorado, USA
>
> **摘要:** Recent advances in vision-language models (VLMs) have made them highly effective at reasoning tasks. However, the principles underlying the construction of performant VL reasoning training datasets remain poorly understood. In this work, we introduce several data curation approaches and study their impacts on VL reasoning capabilities by carefully controlling training and evaluation setups. We analyze the effects of context (image and question pair) sources, implement targeted data interventions, and explore scaling up images, questions, and chain-of-thought (CoT) solutions. Our findings reveal that (a) context source strategies significantly affect VLM performance, (b) interventions such as auxiliary signals from image captions and the inclusion of text-only reasoning yield substantial gains, and (c) scaling all data dimensions (e.g., unique questions per image and unique CoTs per image-question pair) consistently improves reasoning capability. Motivated by these insights, we introduce HoneyBee, a large-scale, high-quality CoT reasoning dataset with 2.5M examples consisting 350K image-question pairs. VLMs trained with HoneyBee outperform state-of-the-art models across model sizes. For instance, a HoneyBee-trained VLM with 3B parameters outperforms the SOTA model and the base model by 7.8% and 24.8%, respectively, on MathVerse. Furthermore, we propose a test-time scaling strategy that reduces decoding cost by 73% without sacrificing accuracy. Overall, this work presents improved strategies for VL reasoning dataset curation research. Data is available at this https URL.
>
---
#### [replaced 075] GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11266](https://arxiv.org/pdf/2511.11266)**

> **作者:** Fabian Schmidt; Markus Enzweiler; Abhinav Valada
>
> **摘要:** Vision-language models have recently emerged as promising planners for autonomous driving, where success hinges on topology-aware reasoning over spatial structure and dynamic interactions from multimodal input. However, existing models are typically trained without supervision that explicitly encodes these relational dependencies, limiting their ability to infer how agents and other traffic entities influence one another from raw sensor data. In this work, we bridge this gap with a novel model-agnostic method that conditions language-based driving models on structured relational context in the form of traffic scene graphs. We serialize scene graphs at various abstraction levels and formats, and incorporate them into models via structured prompt templates, enabling systematic analysis of when and how relational supervision is most beneficial and computationally efficient. Extensive evaluations on the LangAuto and Bench2Drive benchmarks show that scene graph conditioning yields large and persistent improvements. We observe a substantial performance increase in the Driving Score of our proposed approach versus competitive LMDrive, BEVDriver, and SimLingo baselines. These results indicate that diverse architectures can effectively internalize and ground relational priors through scene graph-conditioned training, even without requiring scene graph input at test-time. Code, fine-tuned models, and our scene graph dataset are publicly available at this https URL.
>
---
#### [replaced 076] Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection
- **分类: cs.CV; cs.LG; math.PR**

- **链接: [https://arxiv.org/pdf/2506.21486](https://arxiv.org/pdf/2506.21486)**

> **作者:** Tobias J. Riedlinger; Kira Maag; Hanno Gottschalk
>
> **备注:** 20 pages, 7 figures, 7 tables
>
> **摘要:** Deep neural networks have set the state-of-the-art in computer vision tasks such as bounding box detection and semantic segmentation. Object detectors and segmentation models assign confidence scores to predictions, reflecting the model's uncertainty in object detection or pixel-wise classification. However, these confidence estimates are often miscalibrated, as their architectures and loss functions are tailored to task performance rather than probabilistic foundation. Even with well calibrated predictions, object detectors fail to quantify uncertainty outside detected bounding boxes, i.e., the model does not make a probability assessment of whether an area without detected objects is truly free of obstacles. This poses a safety risk in applications such as automated driving, where uncertainty in empty areas remains unexplored. In this work, we propose an object detection model grounded in spatial statistics. Bounding box data matches realizations of a marked point process, commonly used to describe the probabilistic occurrence of spatial point events identified as bounding box centers, where marks are used to describe the spatial extension of bounding boxes and classes. Our statistical framework enables a likelihood-based training and provides well-defined confidence estimates for whether a region is drivable, i.e., free of objects. We demonstrate the effectiveness of our method through calibration assessments and evaluation of performance.
>
---
#### [replaced 077] Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.11448](https://arxiv.org/pdf/2602.11448)**

> **作者:** Nghia Nguyen; Tianjiao Ding; René Vidal
>
> **备注:** To be published in Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Interpretable-by-design models are gaining traction in computer vision because they provide faithful explanations for their predictions. In image classification, these models typically recover human-interpretable concepts from an image and use them for classification. Sparse concept recovery methods leverage the latent space of vision-language models to represent image embeddings as a sparse combination of concept embeddings. However, because such methods ignore the hierarchical structure of concepts, they can produce correct predictions with explanations that are inconsistent with the hierarchy. In this work, we propose Hierarchical Concept Embedding \& Pursuit (HCEP), a framework that induces a hierarchy of concept embeddings in the latent space and uses hierarchical sparse coding to recover the concepts present in an image. Given a hierarchy of semantic concepts, we construct a corresponding hierarchy of concept embeddings and, assuming the correct concepts for an image form a rooted path in the hierarchy, derive desirable conditions for identifying them in the embedded space. We show that hierarchical sparse coding reliably recovers hierarchical concept embeddings, whereas vanilla sparse coding fails. Our experiments on real-world datasets demonstrate that HCEP outperforms baselines in concept precision and recall while maintaining competitive classification accuracy. Moreover, when the number of samples is limited, HCEP achieves superior classification accuracy and concept recovery. These results show that incorporating hierarchical structures into sparse coding yields more reliable and interpretable image classification models.
>
---
#### [replaced 078] Weight Conditioning for Smooth Optimization of Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.03424](https://arxiv.org/pdf/2409.03424)**

> **作者:** Hemanth Saratchandran; Thomas X. Wang; Simon Lucey
>
> **备注:** ECCV 2024
>
> **摘要:** In this article, we introduce a novel normalization technique for neural network weight matrices, which we term weight conditioning. This approach aims to narrow the gap between the smallest and largest singular values of the weight matrices, resulting in better-conditioned matrices. The inspiration for this technique partially derives from numerical linear algebra, where well-conditioned matrices are known to facilitate stronger convergence results for iterative solvers. We provide a theoretical foundation demonstrating that our normalization technique smoothens the loss landscape, thereby enhancing convergence of stochastic gradient descent algorithms. Empirically, we validate our normalization across various neural network architectures, including Convolutional Neural Networks (CNNs), Vision Transformers (ViT), Neural Radiance Fields (NeRF), and 3D shape modeling. Our findings indicate that our normalization method is not only competitive but also outperforms existing weight normalization techniques from the literature.
>
---
#### [replaced 079] SegDAC: Visual Generalization in Reinforcement Learning via Dynamic Object Tokens
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出SegDAC，解决视觉强化学习中的泛化问题。通过动态对象标记实现高效、可变长度的物体级输入处理，提升视觉条件变化下的性能。**

- **链接: [https://arxiv.org/pdf/2508.09325](https://arxiv.org/pdf/2508.09325)**

> **作者:** Alexandre Brown; Glen Berseth
>
> **备注:** 12 pages
>
> **摘要:** Visual reinforcement learning policies trained on pixel observations often struggle to generalize when visual conditions change at test time. Object-centric representations are a promising alternative, but most approaches use fixed-size slot representations, require image reconstruction, or need auxiliary losses to learn object decompositions. As a result, it remains unclear how to learn RL policies directly from object-level inputs without these constraints. We propose SegDAC, a Segmentation-Driven Actor-Critic that operates on a variable-length set of object token embeddings. At each timestep, text-grounded segmentation produces object masks from which spatially aware token embeddings are extracted. A transformer-based actor-critic processes these dynamic tokens, using segment positional encoding to preserve spatial information across objects. We ablate these design choices and show that both segment positional encoding and variable-length processing are individually necessary for strong performance. We evaluate SegDAC on 8 ManiSkill3 manipulation tasks under 12 visual perturbation types across 3 difficulty levels. SegDAC improves over prior visual generalization methods by 15% on easy, 66% on medium, and 88% on the hardest settings. SegDAC matches the sample efficiency of the state-of-the-art visual RL methods while achieving improved generalization under visual changes. Project Page: this https URL
>
---
#### [replaced 080] DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynVLA模型，解决自动驾驶中的决策问题。通过引入Dynamics CoT，预测世界动态以提升决策的准确性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.11041](https://arxiv.org/pdf/2603.11041)**

> **作者:** Shuyao Shang; Bing Zhan; Yunfei Yan; Yuqi Wang; Yingyan Li; Yasong An; Xiaoman Wang; Jierui Liu; Lu Hou; Lue Fan; Zhaoxiang Zhang; Tieniu Tan
>
> **备注:** 18 pages, 10 figures. Project Page: this https URL
>
> **摘要:** We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT. Project Page: this https URL.
>
---
#### [replaced 081] The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17476](https://arxiv.org/pdf/2505.17476)**

> **作者:** Yuchen Zhang; Yaxiong Wang; Yujiao Wu; Lianwei Wu; Li Zhu; Zhedong Zheng
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** The detection and grounding of multimedia manipulation has emerged as a critical challenge in combating AI-generated disinformation. While existing methods have made progress in recent years, we identify two fundamental limitations in current approaches: (1) Underestimation of MLLM-driven deception risk: prevailing techniques primarily address rule-based text manipulations, yet fail to account for sophisticated misinformation synthesized by multimodal large language models (MLLMs) that can dynamically generate semantically coherent, contextually plausible yet deceptive narratives conditioned on manipulated images; (2) Unrealistic misalignment artifacts: currently focused scenarios rely on artificially misaligned content that lacks semantic coherence, rendering them easily detectable. To address these gaps holistically, we propose a new adversarial pipeline that leverages MLLMs to generate high-risk disinformation. Our approach begins with constructing the MLLM-Driven Synthetic Multimodal (MDSM) dataset, where images are first altered using state-of-the-art editing techniques and then paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. Building upon this foundation, we present the Artifact-aware Manipulation Diagnosis via MLLM (AMD) framework featuring two key innovations: Artifact Pre-perception Encoding strategy and Manipulation-Oriented Reasoning, to tame MLLMs for the MDSM problem. Comprehensive experiments validate our framework's superior generalization capabilities as a unified architecture for detecting MLLM-powered multimodal deceptions. In cross-domain testing on the MDSM dataset, AMD achieves the best average performance, with 88.18 ACC, 60.25 mAP, and 61.02 mIoU scores.
>
---
#### [replaced 082] Rethinking Attention: Polynomial Alternatives to Softmax in Transformers
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2410.18613](https://arxiv.org/pdf/2410.18613)**

> **作者:** Hemanth Saratchandran; Jianqiao Zheng; Yiping Ji; Wenbo Zhang; Simon Lucey
>
> **摘要:** This paper questions whether the strong performance of softmax attention in transformers stems from producing a probability distribution over inputs. Instead, we argue that softmax's effectiveness lies in its implicit regularization of the Frobenius norm of the attention matrix, which stabilizes training. Motivated by this, we explore alternative activations, specifically polynomials, that achieve a similar regularization effect. Our theoretical analysis shows that certain polynomials can serve as effective substitutes for softmax, achieving strong performance across transformer applications despite violating softmax's typical properties of positivity, normalization, and sparsity. Extensive experiments support these findings, offering a new perspective on attention mechanisms.
>
---
#### [replaced 083] FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14099](https://arxiv.org/pdf/2511.14099)**

> **作者:** Jingren Liu; Shuning Xu; Qirui Yang; Yun Wang; Xiangyu Chen; Zhong Ji
>
> **摘要:** All-in-One Image Restoration (AIO-IR) aims to develop a unified model that can handle multiple degradations under complex conditions. However, existing methods often rely on task-specific designs or latent routing strategies, making it hard to adapt to real-world scenarios with various degradations. We propose FAPE-IR, a Frequency-Aware Planning and Execution framework for image restoration. It uses a frozen Multimodal Large Language Model (MLLM) as a planner to analyze degraded images and generate concise, frequency-aware restoration plans. These plans guide a LoRA-based Mixture-of-Experts (LoRA-MoE) module within a diffusion-based executor, which dynamically selects high- or low-frequency experts, complemented by frequency features of the input image. To further improve restoration quality and reduce artifacts, we introduce adversarial training and a frequency regularization loss. By coupling semantic planning with frequency-based restoration, FAPE-IR offers a unified and interpretable solution for all-in-one image restoration. Extensive experiments show that FAPE-IR achieves state-of-the-art performance across seven restoration tasks and exhibits strong zero-shot generalization under mixed degradations.
>
---
#### [replaced 084] From Activation to Initialization: Scaling Insights for Optimizing Neural Fields
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2403.19205](https://arxiv.org/pdf/2403.19205)**

> **作者:** Hemanth Saratchandran; Sameera Ramasinghe; Simon Lucey
>
> **备注:** CVPR 2024
>
> **摘要:** In the realm of computer vision, Neural Fields have gained prominence as a contemporary tool harnessing neural networks for signal representation. Despite the remarkable progress in adapting these networks to solve a variety of problems, the field still lacks a comprehensive theoretical framework. This article aims to address this gap by delving into the intricate interplay between initialization and activation, providing a foundational basis for the robust optimization of Neural Fields. Our theoretical insights reveal a deep-seated connection among network initialization, architectural choices, and the optimization process, emphasizing the need for a holistic approach when designing cutting-edge Neural Fields.
>
---
#### [replaced 085] SDPose: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24980](https://arxiv.org/pdf/2509.24980)**

> **作者:** Shuang Liang; Jing He; Chuanmeizhi Wang; Lejun Liao; Guo Zhang; Yingcong Chen; Yuan Yuan
>
> **备注:** 22 pages, 10 figures, 8 tables
>
> **摘要:** Pre-trained diffusion models provide rich latent features across U-Net levels and are emerging as powerful vision backbones. While prior works such as Marigold and Lotus repurpose diffusion priors for dense geometric perception tasks such as depth and surface normal estimation, their potential for cross-domain human pose estimation remains largely unexplored. Through a systematic analysis of latent features from different upsampling levels of the Stable Diffusion U-Net, we identify the levels that deliver the strongest robustness and cross-domain generalization for pose estimation. Building on these findings, we propose \textbf{SDPose}, which (i) extracts U-Net features from the selected upsampling blocks, (ii) fuses them with a lightweight feature aggregation module to form a robust representation, and (iii) jointly optimizes keypoint heatmap supervision with an auxiliary latent reconstruction loss to regularize training and preserve the pre-trained generative prior. To evaluate cross-domain generalization and robustness, we construct COCO-OOD, a COCO-based benchmark with four subsets: three style-transferred splits to assess domain shift, and one corruption split (noise, weather, digital artifacts, and blur) to test robustness. With a shorter fine-tuning schedule, SDPose achieves performance comparable to Sapiens on COCO, surpasses Sapiens-1B on COCO-WholeBody, and establishes new state-of-the-art results on HumanArt and COCO-OOD.
>
---
#### [replaced 086] Trading Positional Complexity vs. Deepness in Coordinate Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2205.08987](https://arxiv.org/pdf/2205.08987)**

> **作者:** Jianqiao Zheng; Sameera Ramasinghe; Xueqian Li; Simon Lucey
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2107.02561
>
> **摘要:** It is well noted that coordinate-based MLPs benefit -- in terms of preserving high-frequency information -- through the encoding of coordinate positions as an array of Fourier features. Hitherto, the rationale for the effectiveness of these positional encodings has been mainly studied through a Fourier lens. In this paper, we strive to broaden this understanding by showing that alternative non-Fourier embedding functions can indeed be used for positional encoding. Moreover, we show that their performance is entirely determined by a trade-off between the stable rank of the embedded matrix and the distance preservation between embedded coordinates. We further establish that the now ubiquitous Fourier feature mapping of position is a special case that fulfills these conditions. Consequently, we present a more general theory to analyze positional encoding in terms of shifted basis functions. In addition, we argue that employing a more complex positional encoding -- that scales exponentially with the number of modes -- requires only a linear (rather than deep) coordinate function to achieve comparable performance. Counter-intuitively, we demonstrate that trading positional embedding complexity for network deepness is orders of magnitude faster than current state-of-the-art; despite the additional embedding complexity. To this end, we develop the necessary theoretical formulae and empirically verify that our theoretical claims hold in practice.
>
---
#### [replaced 087] GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20673](https://arxiv.org/pdf/2602.20673)**

> **作者:** Hao Zhang; Lue Fan; Qitai Wang; Wenbo Li; Zehuan Wu; Lewei Lu; Zhaoxiang Zhang; Hongsheng Li
>
> **摘要:** A free-viewpoint, editable, and high-fidelity driving simulator is crucial for training and evaluating end-to-end autonomous driving systems. In this paper, we present GA-Drive, a novel simulation framework capable of generating camera views along user-specified novel trajectories through Geometry-Appearance Decoupling and Diffusion-Based Generation. Given a set of images captured along a recorded trajectory and the corresponding scene geometry, GA-Drive synthesizes novel pseudo-views using geometry information. These pseudo-views are then transformed into photorealistic views using a trained video diffusion model. In this way, we decouple the geometry and appearance of scenes. An advantage of such decoupling is its support for appearance editing via state-of-the-art video-to-video editing techniques, while preserving the underlying geometry, enabling consistent edits across both original and novel trajectories. Extensive experiments demonstrate that GA-Drive substantially outperforms existing methods in terms of NTA-IoU, NTL-IoU, and FID scores.
>
---
#### [replaced 088] EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06684](https://arxiv.org/pdf/2512.06684)**

> **作者:** Yumeng He; Zanwei Zhou; Yekun Zheng; Chen Liang; Yunbo Wang; Xiaokang Yang
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Volume electron microscopy (vEM) enables nanoscale 3D imaging of biological structures but remains constrained by acquisition trade-offs, leading to anisotropic volumes with limited axial resolution. Existing deep learning methods seek to restore isotropy by leveraging lateral priors, yet their assumptions break down for morphologically anisotropic structures. We present EMGauss, a general framework for 3D reconstruction from planar scanned 2D slices with applications in vEM, which circumvents the inherent limitations of isotropy-based approaches. Our key innovation is to reframe slice-to-3D reconstruction as a 3D dynamic scene rendering problem based on Gaussian splatting, where the progression of axial slices is modeled as the temporal evolution of 2D Gaussian point clouds. To enhance fidelity in data-sparse regimes, we incorporate a Teacher-Student bootstrapping mechanism that uses high-confidence predictions on unobserved slices as pseudo-supervisory signals. Compared with diffusion- and GAN-based reconstruction methods, EMGauss substantially improves interpolation quality, enables continuous slice synthesis, and eliminates the need for large-scale pretraining. Beyond vEM, it potentially provides a generalizable slice-to-3D solution across diverse imaging domains.
>
---
#### [replaced 089] NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出NavForesee，解决长距离导航任务中复杂指令理解与环境预测问题。通过融合语言规划与环境预测，提升导航智能性。**

- **链接: [https://arxiv.org/pdf/2512.01550](https://arxiv.org/pdf/2512.01550)**

> **作者:** Fei Liu; Shichao Xie; Minghua Luo; Zedong Chu; Junjun Hu; Xiaolong Wu; Mu Xu
>
> **摘要:** Embodied navigation for long-horizon tasks, guided by complex natural language instructions, remains a formidable challenge in artificial intelligence. Existing agents often struggle with robust long-term planning about unseen environments, leading to high failure rates. To address these limitations, we introduce NavForesee, a novel Vision-Language Model (VLM) that unifies high-level language planning and predictive world model imagination within a single, unified framework. Our approach empowers a single VLM to concurrently perform planning and predictive foresight. Conditioned on the full instruction and historical observations, the model is trained to understand the navigation instructions by decomposing the task, tracking its progress, and formulating the subsequent sub-goal. Simultaneously, it functions as a generative world model, providing crucial foresight by predicting short-term environmental dynamics and long-term navigation milestones. The VLM's structured plan guides its targeted prediction, while the imagined future provides rich context to inform the navigation actions, creating a powerful internal feedback loop of perception-planning/prediction-action. We demonstrate through extensive experiments on the R2R-CE and RxR-CE benchmark that NavForesee achieves highly competitive performance in complex scenarios. Our work highlights the immense potential of fusing explicit language planning with implicit spatiotemporal prediction, paving the way for more intelligent and capable embodied agents.
>
---
#### [replaced 090] NI-Tex: Non-isometric Image-based Garment Texture Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18765](https://arxiv.org/pdf/2511.18765)**

> **作者:** Hui Shan; Ming Li; Haitao Yang; Kai Zheng; Sizhe Zheng; Yanwei Fu; Xiangru Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Existing industrial 3D garment meshes already cover most real-world clothing geometries, yet their texture diversity remains limited. To acquire more realistic textures, generative methods are often used to extract Physically-based Rendering (PBR) textures and materials from large collections of wild images and project them back onto garment meshes. However, most image-conditioned texture generation approaches require strict topological consistency between the input image and the input 3D mesh, or rely on accurate mesh deformation to match to the image poses, which significantly constrains the texture generation quality and flexibility. To address the challenging problem of non-isometric image-based garment texture generation, we construct 3D Garment Videos, a physically simulated, garment-centric dataset that provides consistent geometry and material supervision across diverse deformations, enabling robust cross-pose texture learning. We further employ Nano Banana for high-quality non-isometric image editing, achieving reliable cross-topology texture generation between non-isometric image-geometry pairs. Finally, we propose an iterative baking method via uncertainty-guided view selection and reweighting that fuses multi-view predictions into seamless, production-ready PBR textures. Through extensive experiments, we demonstrate that our feedforward dual-branch architecture generates versatile and spatially aligned PBR materials suitable for industry-level 3D garment design.
>
---
#### [replaced 091] RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出RobotArena ∞，用于机器人通用能力的基准测试，解决真实世界测试成本高、不安全等问题，通过模拟环境与人类反馈结合进行评估。**

- **链接: [https://arxiv.org/pdf/2510.23571](https://arxiv.org/pdf/2510.23571)**

> **作者:** Yash Jangir; Yidi Zhang; Pang-Chi Lo; Kashu Yamazaki; Chenyu Zhang; Kuan-Hsun Tu; Tsung-Wei Ke; Lei Ke; Yonatan Bisk; Katerina Fragkiadaki
>
> **备注:** Website: this https URL
>
> **摘要:** The pursuit of robot generalists, agents capable of performing diverse tasks across diverse environments, demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. We introduce RobotArena Infinity, a new benchmarking framework that overcomes these challenges by shifting vision-language-action (VLA) evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated vision-language-model-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, including textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world-trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape.
>
---
#### [replaced 092] PISE: Physics-Anchored Semantically-Enhanced Deep Computational Ghost Imaging for Robust Low-Bandwidth Machine Perception
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.12551](https://arxiv.org/pdf/2601.12551)**

> **作者:** Tong Wu
>
> **备注:** 4 pages, 4 figures, 4 tables. Refined version with updated references and formatting improvements
>
> **摘要:** We propose PISE, a physics-informed deep ghost imaging framework for low-bandwidth edge perception. By combining adjoint operator initialization with semantic guidance, PISE improves classification accuracy by 2.57% and reduces variance by 9x at 5% sampling.
>
---
#### [replaced 093] MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.23228](https://arxiv.org/pdf/2602.23228)**

> **作者:** Yizhi Li; Xiaohan Chen; Miao Jiang; Wentao Tang; Gaoang Wang
>
> **备注:** 6 pages, CSCWD 2026
>
> **摘要:** With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.
>
---
#### [replaced 094] SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.05343](https://arxiv.org/pdf/2512.05343)**

> **作者:** Elisabetta Fedele; Francis Engelmann; Ian Huang; Or Litany; Marc Pollefeys; Leonidas Guibas
>
> **备注:** Project page: this https URL
>
> **摘要:** Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are difficult to manipulate. In this work, we introduce SpaceControl, a training-free test-time method for explicit spatial control of 3D asset generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern generative models without requiring any additional training. A control parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that SpaceControl outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive interface for real-time superquadric editing and direct 3D asset generation, enabling seamless use in creative workflows. Project page: this https URL.
>
---
#### [replaced 095] AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21251](https://arxiv.org/pdf/2511.21251)**

> **作者:** Shuhan Xia; Peipei Li; Xuannan Liu; Dongsen Zhang; Xinyu Guo; Zekun Li
>
> **备注:** The experimental results in this paper have been further improved and updated; the baseline results do not match existing results, therefore the paper needs to be retracted
>
> **摘要:** The threat of Audio-Video (AV) forgery is rapidly evolving beyond human-centric deepfakes to include more diverse manipulations across complex natural scenes. However, existing benchmarks are still confined to DeepFake-based forgeries and single-granularity annotations, thus failing to capture the diversity and complexity of real-world forgery scenarios. To address this, we introduce AVFakeBench, the first comprehensive audio-video forgery detection benchmark that spans rich forgery semantics across both human subject and general subject. AVFakeBench comprises 12K carefully curated audio-video questions, covering seven forgery types and four levels of annotations. To ensure high-quality and diverse forgeries, we propose a multi-stage hybrid forgery framework that integrates proprietary models for task planning with expert generative models for precise manipulation. The benchmark establishes a multi-task evaluation framework covering binary judgment, forgery types classification, forgery detail selection, and explanatory reasoning. We evaluate 11 Audio-Video Large Language Models (AV-LMMs) and 2 prevalent detection methods on AVFakeBench, demonstrating the potential of AV-LMMs as emerging forgery detectors while revealing their notable weaknesses in fine-grained perception and reasoning.
>
---
#### [replaced 096] Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00705](https://arxiv.org/pdf/2510.00705)**

> **作者:** Sanghwan Kim; Rui Xiao; Stephan Alaniz; Yongqin Xian; Zeynep Akata
>
> **摘要:** Multimodal Large Language Models (MLLMs) often struggle with fine-grained perception, such as identifying small objects in high-resolution images or detecting key moments in long videos. Existing methods typically rely on complex, task-specific fine-tuning, which reduces generalizability and increases system complexity. In this work, we propose an effective, training-free framework that uses an MLLM's intrinsic uncertainty as proactive guidance. Our core insight is that a model's uncertainty decreases when provided with relevant visual information. We introduce a unified mechanism that scores candidate visual inputs by response uncertainty, enabling the model to autonomously focus on the most informative data. We apply this simple principle to three challenging visual tasks: Visual Search, Long Video Understanding, and Temporal Grounding, allowing off-the-shelf MLLMs to achieve performance competitive with specialized, fine-tuned systems. Our results demonstrate that leveraging intrinsic uncertainty is a powerful strategy for improving fine-grained multimodal performance.
>
---
#### [replaced 097] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23790](https://arxiv.org/pdf/2602.23790)**

> **作者:** Changyu Gu; Linwei Chen; Lin Gu; Ying Fu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** In remote sensing rotated object detection, mainstream methods suffer from two bottlenecks, directional incoherence at detector neck and task conflict at detecting head. Ulitising fourier rotation equivariance, we introduce Fourier Angle Alignment, which analyses angle information through frequency spectrum and aligns the main direction to a certain orientation. Then we propose two plug and play modules : FAAFusion and FAA Head. FAAFusion works at the detector neck, aligning the main direction of higher-level features to the lower-level features and then fusing them. FAA Head serves as a new detection head, which pre-aligns RoI features to a canonical angle and adds them to the original features before classification and regression. Experiments on DOTA-v1.0, DOTA-v1.5 and HRSC2016 show that our method can greatly improve previous work. Particularly, our method achieves new state-of-the-art results of 78.72% mAP on DOTA-v1.0 and 72.28% mAP on DOTA-v1.5 datasets with single scale training and testing, validating the efficacy of our approach in remote sensing object detection. The code is made publicly available at this https URL .
>
---
#### [replaced 098] Visual Alignment of Medical Vision-Language Models for Grounded Radiology Report Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16201](https://arxiv.org/pdf/2512.16201)**

> **作者:** Sarosij Bose; Ravi K. Rajendran; Biplob Debnath; Konstantinos Karydis; Amit K. Roy-Chowdhury; Srimat Chakradhar
>
> **摘要:** Radiology Report Generation (RRG) is a critical step toward automating healthcare workflows, facilitating accurate patient assessments, and reducing the workload of medical professionals. Despite recent progress in Large Medical Vision-Language Models (Med-VLMs), generating radiology reports that are both visually grounded and clinically accurate remains a significant challenge. Existing approaches often rely on large labeled corpora for pre-training, costly task-specific preference data, or retrieval-based knowledge. However, these strategies do not adequately mitigate hallucinations arising from poor cross-modal alignment between visual and linguistic representations. To address these limitations, we propose VALOR: Visual Alignment of Medical Vision-Language Models for GrOunded Radiology Report Generation, which tackles visual hallucinations through two complementary reasoning stages: (1) Clinically Informed Textual Reasoning guides the model with verifiable natural language and clinical metric rewards to produce semantically complete reports with precise medical terminology. (2) Self-Supervised Visual Reasoning leverages a frozen domain expert to compute image-text similarity scores between the input chest X-ray and generated candidates, converting these into rank-normalized advantages that explicitly steer the policy toward visually grounded outputs, requiring no preference pairs, retrieval databases, or additional annotations. Extensive experiments on multiple benchmarks demonstrate that VALOR substantially improves generation quality, as well as clinical accuracy which are visually grounded, achieving significant performance gains over state-of-the-art medical report generation benchmarks.
>
---
#### [replaced 099] AHAP: Reconstructing Arbitrary Humans from Arbitrary Perspectives with Geometric Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23951](https://arxiv.org/pdf/2602.23951)**

> **作者:** Xiaozhen Qiao; Wenjia Wang; Zhiyuan Zhao; Jiacheng Sun; Ping Luo; Hongyuan Zhang; Xuelong Li
>
> **摘要:** Reconstructing 3D humans from images captured at multiple perspectives typically requires pre-calibration, like using checkerboards or MVS algorithms, which limits scalability and applicability in diverse real-world scenarios. In this work, we present AHAP (Reconstructing Arbitrary Humans from Arbitrary Perspectives), a feed-forward framework for reconstructing arbitrary humans from arbitrary camera perspectives without requiring camera calibration. Our core lies in the effective fusion of multi-view geometry to assist human association, reconstruction and localization. Specifically, we use a Cross-View Identity Association module through learnable person queries and soft assignment, supervised by contrastive learning to resolve cross-view human identity association. A Human Head fuses cross-view features and scene context for SMPL prediction, guided by cross-view reprojection losses to enforce body pose consistency. Additionally, multi-view geometry eliminates the depth ambiguity inherent in monocular methods, providing more precise 3D human localization through multi-view triangulation. Experiments on EgoHumans and EgoExo4D demonstrate that AHAP achieves competitive performance on both world-space human reconstruction and camera pose estimation, while being 180$\times$ faster than optimization-based approaches.
>
---
#### [replaced 100] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09266](https://arxiv.org/pdf/2603.09266)**

> **作者:** Junhao Cai; Deyu Zeng; Junhao Pang; Lini Li; Zongze Wu; Xiaopin Zhong
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Current text-to-3D generation methods excel in natural scenes but struggle with industrial applications due to two critical limitations: domain adaptation challenges where conventional LoRA fusion causes knowledge interference across categories, and geometric reasoning deficiencies where pairwise consistency constraints fail to capture higher-order structural dependencies essential for precision manufacturing. We propose a novel framework named ForgeDreamer addressing both challenges through two key innovations. First, we introduce a Multi-Expert LoRA Ensemble mechanism that consolidates multiple category-specific LoRA models into a unified representation, achieving superior cross-category generalization while eliminating knowledge interference. Second, building on enhanced semantic understanding, we develop a Cross-View Hypergraph Geometric Enhancement approach that captures structural dependencies spanning multiple viewpoints simultaneously. These components work synergistically improved semantic understanding, enables more effective geometric reasoning, while hypergraph modeling ensures manufacturing-level consistency. Extensive experiments on a custom industrial dataset demonstrate superior semantic generalization and enhanced geometric fidelity compared to state-of-the-art approaches. Our code and data are provided in the supplementary material attached in the appendix for review purposes.
>
---
#### [replaced 101] Real-time Rendering-based Surgical Instrument Tracking via Evolutionary Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术器械跟踪任务，解决视觉条件下器械姿态与关节配置的准确估计问题。通过引入CMA-ES优化策略和批量渲染，提升跟踪精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.11404](https://arxiv.org/pdf/2603.11404)**

> **作者:** Hanyang Hu; Zekai Liang; Florian Richter; Michael C. Yip
>
> **摘要:** Accurate and efficient tracking of surgical instruments is fundamental for Robot-Assisted Minimally Invasive Surgery. Although vision-based robot pose estimation has enabled markerless calibration without tedious physical setups, reliable tool tracking for surgical robots still remains challenging due to partial visibility and specialized articulation design of surgical instruments. Previous works in the field are usually prone to unreliable feature detections under degraded visual quality and data scarcity, whereas rendering-based methods often struggle with computational costs and suboptimal convergence. In this work, we incorporate CMA-ES, an evolutionary optimization strategy, into a versatile tracking pipeline that jointly estimates surgical instrument pose and joint configurations. Using batch rendering to efficiently evaluate multiple pose candidates in parallel, the method significantly reduces inference time and improves convergence robustness. The proposed framework further generalizes to joint angle-free and bi-manual tracking settings, making it suitable for both vision feedback control and online surgery video calibration. Extensive experiments on synthetic and real-world datasets demonstrate that the proposed method significantly outperforms prior approaches in both accuracy and runtime.
>
---
