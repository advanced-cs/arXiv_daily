# 计算机视觉 cs.CV

- **最新发布 114 篇**

- **更新 95 篇**

## 最新发布

#### [new 001] Cross-Modal Scene Semantic Alignment for Image Complexity Assessment
- **分类: cs.CV**

- **简介: 该论文研究图像复杂度评估（ICA）任务，旨在解决现有方法因单一视觉特征导致语义表征不足的问题。提出CM-SSA方法，通过跨模态场景语义对齐，融合图像与文本的语义信息，提升复杂度预测与人类感知的一致性。**

- **链接: [http://arxiv.org/pdf/2510.18377v1](http://arxiv.org/pdf/2510.18377v1)**

> **作者:** Yuqing Luo; Yixiao Li; Jiang Liu; Jun Fu; Hadi Amirpour; Guanghui Yue; Baoquan Zhao; Padraig Corcoran; Hantao Liu; Wei Zhou
>
> **备注:** 14 pages,2 figures, British Machine Vision Conference
>
> **摘要:** Image complexity assessment (ICA) is a challenging task in perceptual evaluation due to the subjective nature of human perception and the inherent semantic diversity in real-world images. Existing ICA methods predominantly rely on hand-crafted or shallow convolutional neural network-based features of a single visual modality, which are insufficient to fully capture the perceived representations closely related to image complexity. Recently, cross-modal scene semantic information has been shown to play a crucial role in various computer vision tasks, particularly those involving perceptual understanding. However, the exploration of cross-modal scene semantic information in the context of ICA remains unaddressed. Therefore, in this paper, we propose a novel ICA method called Cross-Modal Scene Semantic Alignment (CM-SSA), which leverages scene semantic alignment from a cross-modal perspective to enhance ICA performance, enabling complexity predictions to be more consistent with subjective human perception. Specifically, the proposed CM-SSA consists of a complexity regression branch and a scene semantic alignment branch. The complexity regression branch estimates image complexity levels under the guidance of the scene semantic alignment branch, while the scene semantic alignment branch is used to align images with corresponding text prompts that convey rich scene semantic information by pair-wise learning. Extensive experiments on several ICA datasets demonstrate that the proposed CM-SSA significantly outperforms state-of-the-art approaches. Codes are available at https://github.com/XQ2K/First-Cross-Model-ICA.
>
---
#### [new 002] CMIS-Net: A Cascaded Multi-Scale Individual Standardization Network for Backchannel Agreement Estimation
- **分类: cs.CV**

- **简介: 该论文研究对话中反馈行为（如点头、“嗯”）的识别任务，旨在解决个体差异和多尺度特征建模问题。提出CMIS-Net网络，通过级联多尺度个体标准化提取相对行为变化，并引入数据增强缓解数据不平衡，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.17855v1](http://arxiv.org/pdf/2510.17855v1)**

> **作者:** Yuxuan Huang; Kangzhong Wang; Eugene Yujun Fu; Grace Ngai; Peter H. F. Ng
>
> **摘要:** Backchannels are subtle listener responses, such as nods, smiles, or short verbal cues like "yes" or "uh-huh," which convey understanding and agreement in conversations. These signals provide feedback to speakers, improve the smoothness of interaction, and play a crucial role in developing human-like, responsive AI systems. However, the expression of backchannel behaviors is often significantly influenced by individual differences, operating across multiple scales: from instant dynamics such as response intensity (frame-level) to temporal patterns such as frequency and rhythm preferences (sequence-level). This presents a complex pattern recognition problem that contemporary emotion recognition methods have yet to fully address. Particularly, existing individualized methods in emotion recognition often operate at a single scale, overlooking the complementary nature of multi-scale behavioral cues. To address these challenges, we propose a novel Cascaded Multi-Scale Individual Standardization Network (CMIS-Net) that extracts individual-normalized backchannel features by removing person-specific neutral baselines from observed expressions. Operating at both frame and sequence levels, this normalization allows model to focus on relative changes from each person's baseline rather than absolute expression values. Furthermore, we introduce an implicit data augmentation module to address the observed training data distributional bias, improving model generalization. Comprehensive experiments and visualizations demonstrate that CMIS-Net effectively handles individual differences and data imbalance, achieving state-of-the-art performance in backchannel agreement detection.
>
---
#### [new 003] UltraGen: High-Resolution Video Generation with Hierarchical Attention
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决高分辨率视频生成中的计算瓶颈问题。作者提出UltraGen框架，通过 hierarchical dual-branch attention 架构实现高效、端到端的原生高分辨率视频合成，支持1080P至4K分辨率，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.18775v1](http://arxiv.org/pdf/2510.18775v1)**

> **作者:** Teng Hu; Jiangning Zhang; Zihan Su; Ran Yi
>
> **摘要:** Recent advances in video generation have made it possible to produce visually compelling videos, with wide-ranging applications in content creation, entertainment, and virtual reality. However, most existing diffusion transformer based video generation models are limited to low-resolution outputs (<=720P) due to the quadratic computational complexity of the attention mechanism with respect to the output width and height. This computational bottleneck makes native high-resolution video generation (1080P/2K/4K) impractical for both training and inference. To address this challenge, we present UltraGen, a novel video generation framework that enables i) efficient and ii) end-to-end native high-resolution video synthesis. Specifically, UltraGen features a hierarchical dual-branch attention architecture based on global-local attention decomposition, which decouples full attention into a local attention branch for high-fidelity regional content and a global attention branch for overall semantic consistency. We further propose a spatially compressed global modeling strategy to efficiently learn global dependencies, and a hierarchical cross-window local attention mechanism to reduce computational costs while enhancing information flow across different local windows. Extensive experiments demonstrate that UltraGen can effectively scale pre-trained low-resolution video models to 1080P and even 4K resolution for the first time, outperforming existing state-of-the-art methods and super-resolution based two-stage pipelines in both qualitative and quantitative evaluations.
>
---
#### [new 004] EMA-SAM: Exponential Moving-average for SAM-based PTMC Segmentation
- **分类: cs.CV**

- **简介: 该论文针对甲状腺微小癌超声视频分割中时序不稳、伪影干扰的问题，提出EMA-SAM方法。其在SAM-2基础上引入轻量指数移动平均指针，提升时序一致性与鲁棒性，显著提高分割精度并保持实时性。**

- **链接: [http://arxiv.org/pdf/2510.18213v1](http://arxiv.org/pdf/2510.18213v1)**

> **作者:** Maryam Dialameh; Hossein Rajabzadeh; Jung Suk Sim; Hyock Ju Kwon
>
> **摘要:** Papillary thyroid microcarcinoma (PTMC) is increasingly managed with radio-frequency ablation (RFA), yet accurate lesion segmentation in ultrasound videos remains difficult due to low contrast, probe-induced motion, and heat-related artifacts. The recent Segment Anything Model 2 (SAM-2) generalizes well to static images, but its frame-independent design yields unstable predictions and temporal drift in interventional ultrasound. We introduce \textbf{EMA-SAM}, a lightweight extension of SAM-2 that incorporates a confidence-weighted exponential moving average pointer into the memory bank, providing a stable latent prototype of the tumour across frames. This design preserves temporal coherence through probe pressure and bubble occlusion while rapidly adapting once clear evidence reappears. On our curated PTMC-RFA dataset (124 minutes, 13 patients), EMA-SAM improves \emph{maxDice} from 0.82 (SAM-2) to 0.86 and \emph{maxIoU} from 0.72 to 0.76, while reducing false positives by 29\%. On external benchmarks, including VTUS and colonoscopy video polyp datasets, EMA-SAM achieves consistent gains of 2--5 Dice points over SAM-2. Importantly, the EMA pointer adds \textless0.1\% FLOPs, preserving real-time throughput of $\sim$30\,FPS on a single A100 GPU. These results establish EMA-SAM as a robust and efficient framework for stable tumour tracking, bridging the gap between foundation models and the stringent demands of interventional ultrasound. Codes are available here \hyperref[code {https://github.com/mdialameh/EMA-SAM}.
>
---
#### [new 005] GAN-based Content-Conditioned Generation of Handwritten Musical Symbols
- **分类: cs.CV; I.2.6; I.4.9; J.5**

- **简介: 该论文属光学乐谱识别任务，旨在缓解手写音乐符号标注数据稀缺问题。作者提出基于GAN生成逼真的手写音乐符号，并用Smashcima合成完整乐谱，显著提升合成乐谱的视觉真实感。**

- **链接: [http://arxiv.org/pdf/2510.17869v1](http://arxiv.org/pdf/2510.17869v1)**

> **作者:** Gerard Asbert; Pau Torras; Lei Kang; Alicia Fornés; Josep Lladós
>
> **备注:** 15 pages, 5 figures, Accepted at ICDAR workshop GREC 2025
>
> **摘要:** The field of Optical Music Recognition (OMR) is currently hindered by the scarcity of real annotated data, particularly when dealing with handwritten historical musical scores. In similar fields, such as Handwritten Text Recognition, it was proven that synthetic examples produced with image generation techniques could help to train better-performing recognition architectures. This study explores the generation of realistic, handwritten-looking scores by implementing a music symbol-level Generative Adversarial Network (GAN) and assembling its output into a full score using the Smashcima engraving software. We have systematically evaluated the visual fidelity of these generated samples, concluding that the generated symbols exhibit a high degree of realism, marking significant progress in synthetic score generation.
>
---
#### [new 006] From Volume Rendering to 3D Gaussian Splatting: Theory and Applications
- **分类: cs.CV; 68-01; A.1**

- **简介: 该论文综述3D高斯点阵化（3DGS）技术，属新视角合成任务。针对其内存高、光照固化和缺乏次级光线效果等问题，梳理改进方法，并展示其在重建、建模与内容生成中的应用。**

- **链接: [http://arxiv.org/pdf/2510.18101v1](http://arxiv.org/pdf/2510.18101v1)**

> **作者:** Vitor Pereira Matias; Daniel Perazzo; Vinicius Silva; Alberto Raposo; Luiz Velho; Afonso Paiva; Tiago Novello
>
> **备注:** Accepted at the Conference on Graphics, Patterns and Images (SIBGRAPI), math focused, 5 equations, 5 Figure, 5 pages of text and 1 of bibligraphy
>
> **摘要:** The problem of 3D reconstruction from posed images is undergoing a fundamental transformation, driven by continuous advances in 3D Gaussian Splatting (3DGS). By modeling scenes explicitly as collections of 3D Gaussians, 3DGS enables efficient rasterization through volumetric splatting, offering thus a seamless integration with common graphics pipelines. Despite its real-time rendering capabilities for novel view synthesis, 3DGS suffers from a high memory footprint, the tendency to bake lighting effects directly into its representation, and limited support for secondary-ray effects. This tutorial provides a concise yet comprehensive overview of the 3DGS pipeline, starting from its splatting formulation and then exploring the main efforts in addressing its limitations. Finally, we survey a range of applications that leverage 3DGS for surface reconstruction, avatar modeling, animation, and content generation-highlighting its efficient rendering and suitability for feed-forward pipelines.
>
---
#### [new 007] Auditing and Mitigating Bias in Gender Classification Algorithms: A Data-Centric Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对性别分类算法中的数据偏见问题，通过审计现有数据集发现交叉性表征不足，构建了均衡的新数据集BalancedFace。实验表明，使用该数据集可显著降低模型偏差，提升公平性，同时保持准确率，推动数据驱动的公平性研究。**

- **链接: [http://arxiv.org/pdf/2510.17873v1](http://arxiv.org/pdf/2510.17873v1)**

> **作者:** Tadesse K Bahiru; Natnael Tilahun Sinshaw; Teshager Hailemariam Moges; Dheeraj Kumar Singh
>
> **摘要:** Gender classification systems often inherit and amplify demographic imbalances in their training data. We first audit five widely used gender classification datasets, revealing that all suffer from significant intersectional underrepresentation. To measure the downstream impact of these flaws, we train identical MobileNetV2 classifiers on the two most balanced of these datasets, UTKFace and FairFace. Our fairness evaluation shows that even these models exhibit significant bias, misclassifying female faces at a higher rate than male faces and amplifying existing racial skew. To counter these data-induced biases, we construct BalancedFace, a new public dataset created by blending images from FairFace and UTKFace, supplemented with images from other collections to fill missing demographic gaps. It is engineered to equalize subgroup shares across 189 intersections of age, race, and gender using only real, unedited images. When a standard classifier is trained on BalancedFace, it reduces the maximum True Positive Rate gap across racial subgroups by over 50% and brings the average Disparate Impact score 63% closer to the ideal of 1.0 compared to the next-best dataset, all with a minimal loss of overall accuracy. These results underline the profound value of data-centric interventions and provide an openly available resource for fair gender classification research.
>
---
#### [new 008] Rebellious Student: A Complementary Learning Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对高光谱异常检测任务，提出“叛逆学生”框架，通过光谱教师网络与空间学生网络的互补学习，增强背景特征。学生网络主动偏离教师，学习未捕获的空间模式，实现无需调参、无需重训练的高效检测，在HAD100上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.18781v1](http://arxiv.org/pdf/2510.18781v1)**

> **作者:** Wenping Jin; Yuyang Tang; Li Zhu; Fei Guo
>
> **摘要:** A recent class of hyperspectral anomaly detection methods that can be trained once on background datasets and then universally deployed -- without per-scene retraining or parameter tuning -- has demonstrated remarkable efficiency and robustness. Building upon this paradigm, we focus on the integration of spectral and spatial cues and introduce a novel "Rebellious Student" framework for complementary feature learning. Unlike conventional teacher-student paradigms driven by imitation, our method intentionally trains the spatial branch to diverge from the spectral teacher, thereby learning complementary spatial patterns that the teacher fails to capture. A two-stage learning strategy is adopted: (1) a spectral enhancement network is first trained via reverse distillation to obtain robust background spectral representations; and (2) a spatial network -- the rebellious student -- is subsequently optimized using decorrelation losses that enforce feature orthogonality while maintaining reconstruction fidelity to avoid irrelevant noise. Once trained, the framework enhances both spectral and spatial background features, enabling parameter-free and training-free anomaly detection when paired with conventional detectors. Extensive experiments on the HAD100 benchmark show substantial improvements over several established baselines with minimal computational overhead, confirming the effectiveness and generality of the proposed complementary learning paradigm. Our code is publicly available at https://github.com/xjpp2016/FERS.
>
---
#### [new 009] Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究图像生成任务中的视觉tokenizer设计，旨在解决现有方法因知识蒸馏导致语义偏移的问题。作者提出VFM-VAE，直接利用视觉基础模型，结合多尺度融合与渐进重建机制，提升重建质量与训练效率，实现更快收敛和更优性能。**

- **链接: [http://arxiv.org/pdf/2510.18457v1](http://arxiv.org/pdf/2510.18457v1)**

> **作者:** Tianci Bi; Xiaoyi Zhang; Yan Lu; Nanning Zheng
>
> **备注:** Code and models available at: https://github.com/tianciB/VFM-VAE
>
> **摘要:** The performance of Latent Diffusion Models (LDMs) is critically dependent on the quality of their visual tokenizer. While recent works have explored incorporating Vision Foundation Models (VFMs) via distillation, we identify a fundamental flaw in this approach: it inevitably weakens the robustness of alignment with the original VFM, causing the aligned latents to deviate semantically under distribution shifts. In this paper, we bypass distillation by proposing a more direct approach: Vision Foundation Model Variational Autoencoder (VFM-VAE). To resolve the inherent tension between the VFM's semantic focus and the need for pixel-level fidelity, we redesign the VFM-VAE decoder with Multi-Scale Latent Fusion and Progressive Resolution Reconstruction blocks, enabling high-quality reconstruction from spatially coarse VFM features. Furthermore, we provide a comprehensive analysis of representation dynamics during diffusion training, introducing the proposed SE-CKNNA metric as a more precise tool for this diagnosis. This analysis allows us to develop a joint tokenizer-diffusion alignment strategy that dramatically accelerates convergence. Our innovations in tokenizer design and training strategy lead to superior performance and efficiency: our system reaches a gFID (w/o CFG) of 2.20 in merely 80 epochs (a 10x speedup over prior tokenizers). With continued training to 640 epochs, it further attains a gFID (w/o CFG) of 1.62, establishing direct VFM integration as a superior paradigm for LDMs.
>
---
#### [new 010] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文研究伪装物体检测（COD），旨在无需标注的情况下提升模型性能。提出RISE框架，利用全数据集构建环境与物体原型库，通过聚类后检索和多视图KNN生成高质量伪标签，实现自增强学习。**

- **链接: [http://arxiv.org/pdf/2510.18437v1](http://arxiv.org/pdf/2510.18437v1)**

> **作者:** Ji Du; Xin Wang; Fangwei Hao; Mingyang Yu; Chunyuan Chen; Jiesheng Wu; Bin Wang; Jing Xu; Ping Li
>
> **备注:** ICCV 2025
>
> **摘要:** At the core of Camouflaged Object Detection (COD) lies segmenting objects from their highly similar surroundings. Previous efforts navigate this challenge primarily through image-level modeling or annotation-based optimization. Despite advancing considerably, this commonplace practice hardly taps valuable dataset-level contextual information or relies on laborious annotations. In this paper, we propose RISE, a RetrIeval SElf-augmented paradigm that exploits the entire training dataset to generate pseudo-labels for single images, which could be used to train COD models. RISE begins by constructing prototype libraries for environments and camouflaged objects using training images (without ground truth), followed by K-Nearest Neighbor (KNN) retrieval to generate pseudo-masks for each image based on these libraries. It is important to recognize that using only training images without annotations exerts a pronounced challenge in crafting high-quality prototype libraries. In this light, we introduce a Clustering-then-Retrieval (CR) strategy, where coarse masks are first generated through clustering, facilitating subsequent histogram-based image filtering and cross-category retrieval to produce high-confidence prototypes. In the KNN retrieval stage, to alleviate the effect of artifacts in feature maps, we propose Multi-View KNN Retrieval (MVKR), which integrates retrieval results from diverse views to produce more robust and precise pseudo-masks. Extensive experiments demonstrate that RISE outperforms state-of-the-art unsupervised and prompt-based methods. Code is available at https://github.com/xiaohainku/RISE.
>
---
#### [new 011] MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation
- **分类: cs.CV**

- **简介: 该论文针对长视频生成中注意力计算开销大的问题，提出Mixture-of-Groups Attention（MoGA），通过可学习的轻量级路由机制实现高效稀疏注意力，在不依赖块状估计的情况下精准匹配关键token，支持长序列建模，实现了端到端分钟级高清视频生成。**

- **链接: [http://arxiv.org/pdf/2510.18692v1](http://arxiv.org/pdf/2510.18692v1)**

> **作者:** Weinan Jia; Yuning Lu; Mengqi Huang; Hualiang Wang; Binyuan Huang; Nan Chen; Mu Liu; Jidong Jiang; Zhendong Mao
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Long video generation with Diffusion Transformers (DiTs) is bottlenecked by the quadratic scaling of full attention with sequence length. Since attention is highly redundant, outputs are dominated by a small subset of query-key pairs. Existing sparse methods rely on blockwise coarse estimation, whose accuracy-efficiency trade-offs are constrained by block size. This paper introduces Mixture-of-Groups Attention (MoGA), an efficient sparse attention that uses a lightweight, learnable token router to precisely match tokens without blockwise estimation. Through semantic-aware routing, MoGA enables effective long-range interactions. As a kernel-free method, MoGA integrates seamlessly with modern attention stacks, including FlashAttention and sequence parallelism. Building on MoGA, we develop an efficient long video generation model that end-to-end produces minute-level, multi-shot, 480p videos at 24 fps, with a context length of approximately 580k. Comprehensive experiments on various video generation tasks validate the effectiveness of our approach.
>
---
#### [new 012] Chimera: Compositional Image Generation using Part-based Concepting
- **分类: cs.CV**

- **简介: 该论文提出Chimera模型，解决个性化图像生成中缺乏对多源图像部件组合控制的问题。通过构建部件级语义原子数据集，训练部分条件扩散先验模型，实现根据文本指令合成新对象，并提出PartEval指标评估生成质量。**

- **链接: [http://arxiv.org/pdf/2510.18083v1](http://arxiv.org/pdf/2510.18083v1)**

> **作者:** Shivam Singh; Yiming Chen; Agneet Chatterjee; Amit Raj; James Hays; Yezhou Yang; Chitra Baral
>
> **摘要:** Personalized image generative models are highly proficient at synthesizing images from text or a single image, yet they lack explicit control for composing objects from specific parts of multiple source images without user specified masks or annotations. To address this, we introduce Chimera, a personalized image generation model that generates novel objects by combining specified parts from different source images according to textual instructions. To train our model, we first construct a dataset from a taxonomy built on 464 unique (part, subject) pairs, which we term semantic atoms. From this, we generate 37k prompts and synthesize the corresponding images with a high-fidelity text-to-image model. We train a custom diffusion prior model with part-conditional guidance, which steers the image-conditioning features to enforce both semantic identity and spatial layout. We also introduce an objective metric PartEval to assess the fidelity and compositional accuracy of generation pipelines. Human evaluations and our proposed metric show that Chimera outperforms other baselines by 14% in part alignment and compositional accuracy and 21% in visual quality.
>
---
#### [new 013] Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-Net
- **分类: cs.CV**

- **简介: 该论文研究少样本图像分类任务，旨在解决灾害图像数据稀缺、类内差异大和类间相似度高的问题。提出ATTBHFA-Net，结合巴塔恰里亚系数与海林格距离，通过概率分布对比学习提升特征聚合与分类性能。**

- **链接: [http://arxiv.org/pdf/2510.18326v1](http://arxiv.org/pdf/2510.18326v1)**

> **作者:** Gao Yu Lee; Tanmoy Dam; Md Meftahul Ferdaus; Daniel Puiu Poenar; Vu Duong
>
> **备注:** Submitted to a SN journal
>
> **摘要:** The increasing frequency of natural and human-induced disasters necessitates advanced visual recognition techniques capable of analyzing critical photographic data. With progress in artificial intelligence and resilient computational systems, rapid and accurate disaster classification has become crucial for efficient rescue operations. However, visual recognition in disaster contexts faces significant challenges due to limited and diverse data from the difficulties in collecting and curating comprehensive, high-quality disaster imagery. Few-Shot Learning (FSL) provides a promising approach to data scarcity, yet current FSL research mainly relies on generic benchmark datasets lacking remote-sensing disaster imagery, limiting its practical effectiveness. Moreover, disaster images exhibit high intra-class variation and inter-class similarity, hindering the performance of conventional metric-based FSL methods. To address these issues, this paper introduces the Attention-based Bhattacharyya-Hellinger Feature Aggregation Network (ATTBHFA-Net), which linearly combines the Bhattacharyya coefficient and Hellinger distances to compare and aggregate feature probability distributions for robust prototype formation. The Bhattacharyya coefficient serves as a contrastive margin that enhances inter-class separability, while the Hellinger distance regularizes same-class alignment. This framework parallels contrastive learning but operates over probability distributions rather than embedded feature points. Furthermore, a Bhattacharyya-Hellinger distance-based contrastive loss is proposed as a distributional counterpart to cosine similarity loss, used jointly with categorical cross-entropy to significantly improve FSL performance. Experiments on four FSL benchmarks and two disaster image datasets demonstrate the superior effectiveness and generalization of ATTBHFA-Net compared to existing approaches.
>
---
#### [new 014] 3D Weakly Supervised Semantic Segmentation via Class-Aware and Geometry-Guided Pseudo-Label Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究3D弱监督语义分割，旨在减少对密集标注的依赖。针对伪标签质量低和几何先验利用不足问题，提出类别感知与几何引导的伪标签优化方法，并结合自训练扩展标签覆盖，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2510.17875v1](http://arxiv.org/pdf/2510.17875v1)**

> **作者:** Xiaoxu Xu; Xuexun Liu; Jinlong Li; Yitian Yuan; Qiudan Zhang; Lin Ma; Nicu Sebe; Xu Wang
>
> **摘要:** 3D weakly supervised semantic segmentation (3D WSSS) aims to achieve semantic segmentation by leveraging sparse or low-cost annotated data, significantly reducing reliance on dense point-wise annotations. Previous works mainly employ class activation maps or pre-trained vision-language models to address this challenge. However, the low quality of pseudo-labels and the insufficient exploitation of 3D geometric priors jointly create significant technical bottlenecks in developing high-performance 3D WSSS models. In this paper, we propose a simple yet effective 3D weakly supervised semantic segmentation method that integrates 3D geometric priors into a class-aware guidance mechanism to generate high-fidelity pseudo labels. Concretely, our designed methodology first employs Class-Aware Label Refinement module to generate more balanced and accurate pseudo labels for semantic categrories. This initial refinement stage focuses on enhancing label quality through category-specific optimization. Subsequently, the Geometry-Aware Label Refinement component is developed, which strategically integrates implicit 3D geometric constraints to effectively filter out low-confidence pseudo labels that fail to comply with geometric plausibility. Moreover, to address the challenge of extensive unlabeled regions, we propose a Label Update strategy that integrates Self-Training to propagate labels into these areas. This iterative process continuously enhances pseudo-label quality while expanding label coverage, ultimately fostering the development of high-performance 3D WSSS models. Comprehensive experimental validation reveals that our proposed methodology achieves state-of-the-art performance on both ScanNet and S3DIS benchmarks while demonstrating remarkable generalization capability in unsupervised settings, maintaining competitive accuracy through its robust design.
>
---
#### [new 015] UWBench: A Comprehensive Vision-Language Benchmark for Underwater Understanding
- **分类: cs.CV**

- **简介: 该论文提出UWBench，首个面向水下视觉-语言理解的综合基准，旨在解决水下图像因光照、浑浊等导致的理解难题。构建了含高质量图像与多模态标注的数据集，支持图像描述、视觉定位和问答任务，推动水下视觉语言模型发展。**

- **链接: [http://arxiv.org/pdf/2510.18262v1](http://arxiv.org/pdf/2510.18262v1)**

> **作者:** Da Zhang; Chenggang Rong; Bingyu Li; Feiyu Wang; Zhiyuan Zhao; Junyu Gao; Xuelong Li
>
> **备注:** We have released V1, which only reports the test results. Our work is still ongoing, and the next version will be coming soon
>
> **摘要:** Large vision-language models (VLMs) have achieved remarkable success in natural scene understanding, yet their application to underwater environments remains largely unexplored. Underwater imagery presents unique challenges including severe light attenuation, color distortion, and suspended particle scattering, while requiring specialized knowledge of marine ecosystems and organism taxonomy. To bridge this gap, we introduce UWBench, a comprehensive benchmark specifically designed for underwater vision-language understanding. UWBench comprises 15,003 high-resolution underwater images captured across diverse aquatic environments, encompassing oceans, coral reefs, and deep-sea habitats. Each image is enriched with human-verified annotations including 15,281 object referring expressions that precisely describe marine organisms and underwater structures, and 124,983 question-answer pairs covering diverse reasoning capabilities from object recognition to ecological relationship understanding. The dataset captures rich variations in visibility, lighting conditions, and water turbidity, providing a realistic testbed for model evaluation. Based on UWBench, we establish three comprehensive benchmarks: detailed image captioning for generating ecologically informed scene descriptions, visual grounding for precise localization of marine organisms, and visual question answering for multimodal reasoning about underwater environments. Extensive experiments on state-of-the-art VLMs demonstrate that underwater understanding remains challenging, with substantial room for improvement. Our benchmark provides essential resources for advancing vision-language research in underwater contexts and supporting applications in marine science, ecological monitoring, and autonomous underwater exploration. Our code and benchmark will be available.
>
---
#### [new 016] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文提出ImageGem数据集，旨在解决生成模型个性化中缺乏真实用户偏好标注的问题。通过收集大规模用户交互数据，支持偏好对齐、个性化图像检索与模型推荐，并提出编辑扩散模型的框架，推动生成模型个性化研究。**

- **链接: [http://arxiv.org/pdf/2510.18433v1](http://arxiv.org/pdf/2510.18433v1)**

> **作者:** Yuanhe Guo; Linxi Xie; Zhuoran Chen; Kangrui Yu; Ryan Po; Guandao Yang; Gordon Wetztein; Hongyi Wen
>
> **摘要:** We introduce ImageGem, a dataset for studying generative models that understand fine-grained individual preferences. We posit that a key challenge hindering the development of such a generative model is the lack of in-the-wild and fine-grained user preference annotations. Our dataset features real-world interaction data from 57K users, who collectively have built 242K customized LoRAs, written 3M text prompts, and created 5M generated images. With user preference annotations from our dataset, we were able to train better preference alignment models. In addition, leveraging individual user preference, we investigated the performance of retrieval models and a vision-language model on personalized image retrieval and generative model recommendation. Finally, we propose an end-to-end framework for editing customized diffusion models in a latent weight space to align with individual user preferences. Our results demonstrate that the ImageGem dataset enables, for the first time, a new paradigm for generative model personalization.
>
---
#### [new 017] IF-VidCap: Can Video Caption Models Follow Instructions?
- **分类: cs.CV**

- **简介: 该论文聚焦视频描述生成中的指令跟随能力，提出新基准IF-VidCap，评估模型在格式与内容上的指令遵循表现，揭示现有模型的不足，并推动可控视频描述任务发展。**

- **链接: [http://arxiv.org/pdf/2510.18726v1](http://arxiv.org/pdf/2510.18726v1)**

> **作者:** Shihao Li; Yuanxing Zhang; Jiangtao Wu; Zhide Lei; Yiwen He; Runzhe Wen; Chenxi Liao; Chengkang Jiang; An Ping; Shuo Gao; Suhan Wang; Zhaozhou Bian; Zijun Zhou; Jingyi Xie; Jiayi Zhou; Jing Wang; Yifan Yao; Weihao Xie; Yingshui Tan; Yanghai Wang; Qianqian Xie; Zhaoxiang Zhang; Jiaheng Liu
>
> **备注:** https://github.com/NJU-LINK/IF-VidCap
>
> **摘要:** Although Multimodal Large Language Models (MLLMs) have demonstrated proficiency in video captioning, practical applications require captions that follow specific user instructions rather than generating exhaustive, unconstrained descriptions. Current benchmarks, however, primarily assess descriptive comprehensiveness while largely overlooking instruction-following capabilities. To address this gap, we introduce IF-VidCap, a new benchmark for evaluating controllable video captioning, which contains 1,400 high-quality samples. Distinct from existing video captioning or general instruction-following benchmarks, IF-VidCap incorporates a systematic framework that assesses captions on two dimensions: format correctness and content correctness. Our comprehensive evaluation of over 20 prominent models reveals a nuanced landscape: despite the continued dominance of proprietary models, the performance gap is closing, with top-tier open-source solutions now achieving near-parity. Furthermore, we find that models specialized for dense captioning underperform general-purpose MLLMs on complex instructions, indicating that future work should simultaneously advance both descriptive richness and instruction-following fidelity.
>
---
#### [new 018] GBlobs: Local LiDAR Geometry for Improved Sensor Placement Generalization
- **分类: cs.CV**

- **简介: 该论文针对LiDAR传感器在不同安装位置下3D目标检测的泛化问题，提出GBlobs局部几何特征。通过避免模型依赖绝对坐标的位置偏差，提升跨配置的检测性能，增强了模型对点云分布变化的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.18539v1](http://arxiv.org/pdf/2510.18539v1)**

> **作者:** Dušan Malić; Christian Fruhwirth-Reisinger; Alexander Prutsch; Wei Lin; Samuel Schulter; Horst Possegger
>
> **备注:** 1st place at the IROS'25 RoboSense Challenge, Track #3: Cross-Sensor Placement 3D Object Detection
>
> **摘要:** This technical report outlines the top-ranking solution for RoboSense 2025: Track 3, achieving state-of-the-art performance on 3D object detection under various sensor placements. Our submission utilizes GBlobs, a local point cloud feature descriptor specifically designed to enhance model generalization across diverse LiDAR configurations. Current LiDAR-based 3D detectors often suffer from a \enquote{geometric shortcut} when trained on conventional global features (\ie, absolute Cartesian coordinates). This introduces a position bias that causes models to primarily rely on absolute object position rather than distinguishing shape and appearance characteristics. Although effective for in-domain data, this shortcut severely limits generalization when encountering different point distributions, such as those resulting from varying sensor placements. By using GBlobs as network input features, we effectively circumvent this geometric shortcut, compelling the network to learn robust, object-centric representations. This approach significantly enhances the model's ability to generalize, resulting in the exceptional performance demonstrated in this challenge.
>
---
#### [new 019] DWaste: Greener AI for Waste Sorting using Mobile and Edge Devices
- **分类: cs.CV**

- **简介: 该论文针对垃圾分类的可持续管理问题，提出DWaste平台，利用轻量级AI模型在手机和边缘设备上实现实时、低功耗的垃圾图像分类与检测，平衡准确率与资源消耗，支持离线运行，推动绿色AI在环保领域的应用。**

- **链接: [http://arxiv.org/pdf/2510.18513v1](http://arxiv.org/pdf/2510.18513v1)**

> **作者:** Suman Kunwar
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The rise of convenience packaging has led to generation of enormous waste, making efficient waste sorting crucial for sustainable waste management. To address this, we developed DWaste, a computer vision-powered platform designed for real-time waste sorting on resource-constrained smartphones and edge devices, including offline functionality. We benchmarked various image classification models (EfficientNetV2S/M, ResNet50/101, MobileNet) and object detection (YOLOv8n, YOLOv11n) using a subset of our own waste data set and annotated it using the custom tool Annotated Lab. We found a clear trade-off between accuracy and resource consumption: the best classifier, EfficientNetV2S, achieved high accuracy (~ 96%) but suffered from high latency (~ 0.22s) and elevated carbon emissions. In contrast, lightweight object detection models delivered strong performance (up to 77% mAP) with ultra-fast inference (~ 0.03s) and significantly smaller model sizes (< 7MB), making them ideal for real-time, low-power use. Model quantization further maximized efficiency, substantially reducing model size and VRAM usage by up to 75%. Our work demonstrates the successful implementation of "Greener AI" models to support real-time, sustainable waste sorting on edge devices.
>
---
#### [new 020] Bayesian Fully-Connected Tensor Network for Hyperspectral-Multispectral Image Fusion
- **分类: cs.CV**

- **简介: 该论文研究高光谱-多光谱图像融合，旨在解决现有张量分解方法破坏空间光谱结构、依赖人工调参及抗噪能力弱的问题。提出贝叶斯全连接张量网络（BFCTN），通过概率框架建模物理关联，结合变分贝叶斯与EM算法实现自动参数估计，提升融合精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.18400v1](http://arxiv.org/pdf/2510.18400v1)**

> **作者:** Linsong Shan; Zecan Yang; Laurence T. Yang; Changlong Li; Honglu Zhao; Xin Nie
>
> **摘要:** Tensor decomposition is a powerful tool for data analysis and has been extensively employed in the field of hyperspectral-multispectral image fusion (HMF). Existing tensor decomposition-based fusion methods typically rely on disruptive data vectorization/reshaping or impose rigid constraints on the arrangement of factor tensors, hindering the preservation of spatial-spectral structures and the modeling of cross-dimensional correlations. Although recent advances utilizing the Fully-Connected Tensor Network (FCTN) decomposition have partially alleviated these limitations, the process of reorganizing data into higher-order tensors still disrupts the intrinsic spatial-spectral structure. Furthermore, these methods necessitate extensive manual parameter tuning and exhibit limited robustness against noise and spatial degradation. To alleviate these issues, we propose the Bayesian FCTN (BFCTN) method. Within this probabilistic framework, a hierarchical sparse prior that characterizing the sparsity of physical elements, establishes connections between the factor tensors. This framework explicitly models the intrinsic physical coupling among spatial structures, spectral signatures, and local scene homogeneity. For model learning, we develop a parameter estimation method based on Variational Bayesian inference (VB) and the Expectation-Maximization (EM) algorithm, which significantly reduces the need for manual parameter tuning. Extensive experiments demonstrate that BFCTN not only achieves state-of-the-art fusion accuracy and strong robustness but also exhibits practical applicability in complex real-world scenarios.
>
---
#### [new 021] GPTFace: Generative Pre-training of Facial-Linguistic Transformer by Span Masking and Weakly Correlated Text-image Data
- **分类: cs.CV**

- **简介: 该论文提出GPTFace，旨在解决人脸识别模型依赖人工标注数据的问题。通过在弱关联图文数据上进行生成式预训练，结合掩码建模与图像-文本匹配任务，实现面部知识学习与可控生成，支持属性分类、表情识别及多种人脸编辑任务。**

- **链接: [http://arxiv.org/pdf/2510.18345v1](http://arxiv.org/pdf/2510.18345v1)**

> **作者:** Yudong Li; Hao Li; Xianxu Hou; Linlin Shen
>
> **备注:** This work was initially drafted in November 2022
>
> **摘要:** Compared to the prosperity of pre-training models in natural image understanding, the research on large-scale pre-training models for facial knowledge learning is still limited. Current approaches mainly rely on manually assembled and annotated face datasets for training, but labeling such datasets is labor-intensive and the trained models have limited scalability beyond the training data. To address these limitations, we present a generative pre-training model for facial knowledge learning that leverages large-scale web-built data for training. We use texts and images containing human faces crawled from the internet and conduct pre-training on self-supervised tasks, including masked image/language modeling (MILM) and image-text matching (ITM). During the generation stage, we further utilize the image-text matching loss to pull the generation distribution towards the control signal for controllable image/text generation. Experimental results demonstrate that our model achieves comparable performance to state-of-the-art pre-training models for various facial downstream tasks, such as attribution classification and expression recognition. Furthermore, our approach is also applicable to a wide range of face editing tasks, including face attribute editing, expression manipulation, mask removal, and photo inpainting.
>
---
#### [new 022] S2AP: Score-space Sharpness Minimization for Adversarial Pruning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究对抗性剪枝中的掩码选择不稳定问题，提出S2AP方法，通过在分数空间引入锐度最小化来稳定搜索过程，提升剪枝后模型的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.18381v1](http://arxiv.org/pdf/2510.18381v1)**

> **作者:** Giorgio Piras; Qi Zhao; Fabio Brau; Maura Pintor; Christian Wressnegger; Battista Biggio
>
> **摘要:** Adversarial pruning methods have emerged as a powerful tool for compressing neural networks while preserving robustness against adversarial attacks. These methods typically follow a three-step pipeline: (i) pretrain a robust model, (ii) select a binary mask for weight pruning, and (iii) finetune the pruned model. To select the binary mask, these methods minimize a robust loss by assigning an importance score to each weight, and then keep the weights with the highest scores. However, this score-space optimization can lead to sharp local minima in the robust loss landscape and, in turn, to an unstable mask selection, reducing the robustness of adversarial pruning methods. To overcome this issue, we propose a novel plug-in method for adversarial pruning, termed Score-space Sharpness-aware Adversarial Pruning (S2AP). Through our method, we introduce the concept of score-space sharpness minimization, which operates during the mask search by perturbing importance scores and minimizing the corresponding robust loss. Extensive experiments across various datasets, models, and sparsity levels demonstrate that S2AP effectively minimizes sharpness in score space, stabilizing the mask selection, and ultimately improving the robustness of adversarial pruning methods.
>
---
#### [new 023] RadDiagSeg-M: A Vision Language Model for Joint Diagnosis and Multi-Target Segmentation in Radiology
- **分类: cs.CV; cs.AI; 68; I.4.6**

- **简介: 该论文针对医学视觉语言模型难以同时生成诊断文本和分割掩码的问题，提出新数据集RadDiagSeg-D和模型RadDiagSeg-M，实现异常检测、诊断与多目标分割的联合输出，提升临床辅助诊断的实用性。**

- **链接: [http://arxiv.org/pdf/2510.18188v1](http://arxiv.org/pdf/2510.18188v1)**

> **作者:** Chengrun Li; Corentin Royer; Haozhe Luo; Bastian Wittmann; Xia Li; Ibrahim Hamamci; Sezgin Er; Anjany Sekuboyina; Bjoern Menze
>
> **摘要:** Most current medical vision language models struggle to jointly generate diagnostic text and pixel-level segmentation masks in response to complex visual questions. This represents a major limitation towards clinical application, as assistive systems that fail to provide both modalities simultaneously offer limited value to medical practitioners. To alleviate this limitation, we first introduce RadDiagSeg-D, a dataset combining abnormality detection, diagnosis, and multi-target segmentation into a unified and hierarchical task. RadDiagSeg-D covers multiple imaging modalities and is precisely designed to support the development of models that produce descriptive text and corresponding segmentation masks in tandem. Subsequently, we leverage the dataset to propose a novel vision-language model, RadDiagSeg-M, capable of joint abnormality detection, diagnosis, and flexible segmentation. RadDiagSeg-M provides highly informative and clinically useful outputs, effectively addressing the need to enrich contextual information for assistive diagnosis. Finally, we benchmark RadDiagSeg-M and showcase its strong performance across all components involved in the task of multi-target text-and-mask generation, establishing a robust and competitive baseline.
>
---
#### [new 024] Adapting Stereo Vision From Objects To 3D Lunar Surface Reconstruction with the StereoLunar Dataset
- **分类: cs.CV**

- **简介: 该论文聚焦月球表面3D重建任务，解决现有立体视觉方法在月球低纹理、复杂光照下性能差的问题。作者构建了首个面向月球的立体视觉数据集LunarStereo，并基于其对MASt3R模型进行微调，显著提升了在真实与合成月球数据上的3D重建与位姿估计效果。**

- **链接: [http://arxiv.org/pdf/2510.18172v1](http://arxiv.org/pdf/2510.18172v1)**

> **作者:** Clementine Grethen; Simone Gasparini; Geraldine Morin; Jeremy Lebreton; Lucas Marti; Manuel Sanchez-Gestido
>
> **备注:** Accepted to ICCV workshop 2025. The project page can be accessed via this https://clementinegrethen.github.io/publications/3D-Vast-ICCV2025.html URL. The source code is available at this https://github.com/clementinegrethen/StereoLunar URL
>
> **摘要:** Accurate 3D reconstruction of lunar surfaces is essential for space exploration. However, existing stereo vision reconstruction methods struggle in this context due to the Moon's lack of texture, difficult lighting variations, and atypical orbital trajectories. State-of-the-art deep learning models, trained on human-scale datasets, have rarely been tested on planetary imagery and cannot be transferred directly to lunar conditions. To address this issue, we introduce LunarStereo, the first open dataset of photorealistic stereo image pairs of the Moon, simulated using ray tracing based on high-resolution topography and reflectance models. It covers diverse altitudes, lighting conditions, and viewing angles around the lunar South Pole, offering physically grounded supervision for 3D reconstruction tasks. Based on this dataset, we adapt the MASt3R model to the lunar domain through fine-tuning on LunarStereo. We validate our approach through extensive qualitative and quantitative experiments on both synthetic and real lunar data, evaluating 3D surface reconstruction and relative pose estimation. Extensive experiments on synthetic and real lunar data validate the approach, demonstrating significant improvements over zero-shot baselines and paving the way for robust cross-scale generalization in extraterrestrial environments.
>
---
#### [new 025] ViSE: A Systematic Approach to Vision-Only Street-View Extrapolation
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶仿真中的街景视图外推任务，解决现有新视角合成方法在外推时图像失真问题。提出四阶段框架：伪LiDAR初始化、2D-SDF几何先验建模、生成先验构建伪标签、数据驱动自适应去伪影，在RealADSim-NVS榜单上排名第一。**

- **链接: [http://arxiv.org/pdf/2510.18341v1](http://arxiv.org/pdf/2510.18341v1)**

> **作者:** Kaiyuan Tan; Yingying Shen; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye
>
> **摘要:** Realistic view extrapolation is critical for closed-loop simulation in autonomous driving, yet it remains a significant challenge for current Novel View Synthesis (NVS) methods, which often produce distorted and inconsistent images beyond the original trajectory. This report presents our winning solution which ctook first place in the RealADSim Workshop NVS track at ICCV 2025. To address the core challenges of street view extrapolation, we introduce a comprehensive four-stage pipeline. First, we employ a data-driven initialization strategy to generate a robust pseudo-LiDAR point cloud, avoiding local minima. Second, we inject strong geometric priors by modeling the road surface with a novel dimension-reduced SDF termed 2D-SDF. Third, we leverage a generative prior to create pseudo ground truth for extrapolated viewpoints, providing auxilary supervision. Finally, a data-driven adaptation network removes time-specific artifacts. On the RealADSim-NVS benchmark, our method achieves a final score of 0.441, ranking first among all participants.
>
---
#### [new 026] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery
- **分类: cs.CV**

- **简介: 该论文研究广义类别发现任务，旨在对部分标注数据中的未标注图像进行分类。提出SEAL框架，利用自然层次结构，通过语义引导的对比学习和跨粒度一致性模块，提升模型在细粒度和粗粒度数据集上的性能。**

- **链接: [http://arxiv.org/pdf/2510.18740v1](http://arxiv.org/pdf/2510.18740v1)**

> **作者:** Zhenqi He; Yuanpei Liu; Kai Han
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** This paper investigates the problem of Generalized Category Discovery (GCD). Given a partially labelled dataset, GCD aims to categorize all unlabelled images, regardless of whether they belong to known or unknown classes. Existing approaches typically depend on either single-level semantics or manually designed abstract hierarchies, which limit their generalizability and scalability. To address these limitations, we introduce a SEmantic-aware hierArchical Learning framework (SEAL), guided by naturally occurring and easily accessible hierarchical structures. Within SEAL, we propose a Hierarchical Semantic-Guided Soft Contrastive Learning approach that exploits hierarchical similarity to generate informative soft negatives, addressing the limitations of conventional contrastive losses that treat all negatives equally. Furthermore, a Cross-Granularity Consistency (CGC) module is designed to align the predictions from different levels of granularity. SEAL consistently achieves state-of-the-art performance on fine-grained benchmarks, including the SSB benchmark, Oxford-Pet, and the Herbarium19 dataset, and further demonstrates generalization on coarse-grained datasets. Project page: https://visual-ai.github.io/seal/
>
---
#### [new 027] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation
- **分类: cs.CV**

- **简介: 该论文研究无见物体重建6D位姿估计任务，旨在解决模板匹配中因检索错误导致的位姿误差问题。提出RayPose方法，将模板视图与查询图像的对齐转化为光线对齐问题，结合扩散Transformer架构和几何先验，实现更准确的位姿预测。**

- **链接: [http://arxiv.org/pdf/2510.18521v1](http://arxiv.org/pdf/2510.18521v1)**

> **作者:** Junwen Huang; Shishir Reddy Vutukur; Peter KT Yu; Nassir Navab; Slobodan Ilic; Benjamin Busam
>
> **摘要:** Typical template-based object pose pipelines estimate the pose by retrieving the closest matching template and aligning it with the observed image. However, failure to retrieve the correct template often leads to inaccurate pose predictions. To address this, we reformulate template-based object pose estimation as a ray alignment problem, where the viewing directions from multiple posed template images are learned to align with a non-posed query image. Inspired by recent progress in diffusion-based camera pose estimation, we embed this formulation into a diffusion transformer architecture that aligns a query image with a set of posed templates. We reparameterize object rotation using object-centered camera rays and model object translation by extending scale-invariant translation estimation to dense translation offsets. Our model leverages geometric priors from the templates to guide accurate query pose inference. A coarse-to-fine training strategy based on narrowed template sampling improves performance without modifying the network architecture. Extensive experiments across multiple benchmark datasets show competitive results of our method compared to state-of-the-art approaches in unseen object pose estimation.
>
---
#### [new 028] FedDEAP: Adaptive Dual-Prompt Tuning for Multi-Domain Federated Learning
- **分类: cs.CV**

- **简介: 该论文研究联邦学习下的多域图像识别任务，旨在解决域偏移和标签异构导致的模型泛化性差问题。提出FedDEAP框架，通过解耦语义与域特征、设计双提示机制及对齐图文表示，提升CLIP在联邦场景下的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.18837v1](http://arxiv.org/pdf/2510.18837v1)**

> **作者:** Yubin Zheng; Pak-Hei Yeung; Jing Xia; Tianjie Ju; Peng Tang; Weidong Qiu; Jagath C. Rajapakse
>
> **备注:** Accepted at MM 2025
>
> **摘要:** Federated learning (FL) enables multiple clients to collaboratively train machine learning models without exposing local data, balancing performance and privacy. However, domain shift and label heterogeneity across clients often hinder the generalization of the aggregated global model. Recently, large-scale vision-language models like CLIP have shown strong zero-shot classification capabilities, raising the question of how to effectively fine-tune CLIP across domains in a federated setting. In this work, we propose an adaptive federated prompt tuning framework, FedDEAP, to enhance CLIP's generalization in multi-domain scenarios. Our method includes the following three key components: (1) To mitigate the loss of domain-specific information caused by label-supervised tuning, we disentangle semantic and domain-specific features in images by using semantic and domain transformation networks with unbiased mappings; (2) To preserve domain-specific knowledge during global prompt aggregation, we introduce a dual-prompt design with a global semantic prompt and a local domain prompt to balance shared and personalized information; (3) To maximize the inclusion of semantic and domain information from images in the generated text features, we align textual and visual representations under the two learned transformations to preserve semantic and domain consistency. Theoretical analysis and extensive experiments on four datasets demonstrate the effectiveness of our method in enhancing the generalization of CLIP for federated image recognition across multiple domains.
>
---
#### [new 029] Exploring a Unified Vision-Centric Contrastive Alternatives on Multi-Modal Web Documents
- **分类: cs.CV**

- **简介: 该论文聚焦多模态网页文档理解任务，旨在解决图文交错、对齐松散等问题。提出VC2L框架，将文本、图像统一渲染为像素输入视觉Transformer，通过片段级对比学习建模跨模态关系，无需OCR或文本编码，提升检索与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.18703v1](http://arxiv.org/pdf/2510.18703v1)**

> **作者:** Yiqi Lin; Alex Jinpeng Wang; Linjie Li; Zhengyuan Yang; Mike Zheng Shou
>
> **备注:** Project page: this https://linyq17.github.io/VC2L/
>
> **摘要:** Contrastive vision-language models such as CLIP have demonstrated strong performance across a wide range of multimodal tasks by learning from aligned image-text pairs. However, their ability to handle complex, real-world web documents remains limited, particularly in scenarios where text and images are interleaved, loosely aligned, or embedded in visual form. To address these challenges, we propose Vision-Centric Contrastive Learning (VC2L), a unified framework that models text, images, and their combinations using a single vision transformer. VC2L operates entirely in pixel space by rendering all inputs, whether textual, visual, or combined, as images, thus eliminating the need for OCR, text tokenization, or modality fusion strategy. To capture complex cross-modal relationships in multimodal web documents, VC2L employs a snippet-level contrastive learning objective that aligns consecutive multimodal segments, leveraging the inherent coherence of documents without requiring explicitly paired image-text data. To assess the effectiveness of this approach, we introduce three retrieval benchmarks, AnyCIR, SeqCIR, and CSR, designed to evaluate cross-modal retrieval, fine-grained sequential understanding, and generalization to unseen data, respectively. Empirical results show that VC2L achieves competitive or superior performance compared to CLIP-style models on both the proposed benchmarks and established datasets such as M-BEIR and MTEB. These findings underscore the potential of multimodal web data as a valuable training resource for contrastive learning and illustrate the scalability of a unified, vision-centric approach for multimodal representation learning. Code and models are available at: https://github.com/showlab/VC2L.
>
---
#### [new 030] An Explainable Hybrid AI Framework for Enhanced Tuberculosis and Symptom Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种可解释的混合AI框架，用于结核病及症状检测。针对缺乏专业医生和高质量数据的问题，采用教师-学生模型结合监督与自监督学习，提升胸部X光片的疾病分类和多症状识别性能，准确率达98.85%，并确保预测依据解剖学相关特征。**

- **链接: [http://arxiv.org/pdf/2510.18819v1](http://arxiv.org/pdf/2510.18819v1)**

> **作者:** Neel Patel; Alexander Wong; Ashkan Ebadi
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** Tuberculosis remains a critical global health issue, particularly in resource-limited and remote areas. Early detection is vital for treatment, yet the lack of skilled radiologists underscores the need for artificial intelligence (AI)-driven screening tools. Developing reliable AI models is challenging due to the necessity for large, high-quality datasets, which are costly to obtain. To tackle this, we propose a teacher--student framework which enhances both disease and symptom detection on chest X-rays by integrating two supervised heads and a self-supervised head. Our model achieves an accuracy of 98.85% for distinguishing between COVID-19, tuberculosis, and normal cases, and a macro-F1 score of 90.09% for multilabel symptom detection, significantly outperforming baselines. The explainability assessments also show the model bases its predictions on relevant anatomical features, demonstrating promise for deployment in clinical screening and triage settings.
>
---
#### [new 031] Accelerating Vision Transformers with Adaptive Patch Sizes
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视觉Transformer处理高分辨率图像时计算开销大的问题，提出自适应 patch 大小方法（APT），在内容简单区域用大 patch、复杂区域用小 patch，减少输入 token 数，显著加速训练与推理，保持性能，适用于已微调模型和密集视觉任务。**

- **链接: [http://arxiv.org/pdf/2510.18091v1](http://arxiv.org/pdf/2510.18091v1)**

> **作者:** Rohan Choudhury; JungEun Kim; Jinhyung Park; Eunho Yang; László A. Jeni; Kris M. Kitani
>
> **备注:** Project page at https://rccchoudhury.github.io/apt/
>
> **摘要:** Vision Transformers (ViTs) partition input images into uniformly sized patches regardless of their content, resulting in long input sequence lengths for high-resolution images. We present Adaptive Patch Transformers (APT), which addresses this by using multiple different patch sizes within the same image. APT reduces the total number of input tokens by allocating larger patch sizes in more homogeneous areas and smaller patches in more complex ones. APT achieves a drastic speedup in ViT inference and training, increasing throughput by 40% on ViT-L and 50% on ViT-H while maintaining downstream performance, and can be applied to a previously fine-tuned ViT, converging in as little as 1 epoch. It also significantly reduces training and inference time without loss of performance in high-resolution dense visual tasks, achieving up to 30\% faster training and inference in visual QA, object detection, and semantic segmentation.
>
---
#### [new 032] CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多模态数据集蒸馏任务，旨在解决现有方法因冻结文本编码器导致语义对齐不足的问题。作者提出CovMatch，通过跨协方差对齐与模态内分布正则化，实现双编码器联合优化，显著提升小规模合成数据下的视觉-语言模型性能。**

- **链接: [http://arxiv.org/pdf/2510.18583v1](http://arxiv.org/pdf/2510.18583v1)**

> **作者:** Yongmin Lee; Hye Won Chung
>
> **备注:** NeurIPS 2025
>
> **摘要:** Multimodal dataset distillation aims to synthesize a small set of image-text pairs that enables efficient training of large-scale vision-language models. While dataset distillation has shown promise in unimodal tasks, extending it to multimodal contrastive learning presents key challenges: learning cross-modal alignment and managing the high computational cost of large encoders. Prior approaches address scalability by freezing the text encoder and update only the image encoder and text projection layer. However, we find this severely limits semantic alignment and becomes a bottleneck for performance scaling. We propose CovMatch, a scalable dataset distillation framework that aligns the cross-covariance of real and synthetic features while regularizing feature distributions within each modality. Unlike prior approaches, CovMatch enables joint optimization of both encoders, leading to stronger cross-modal alignment and improved performance. Evaluated on Flickr30K and COCO, CovMatch outperforms state-of-the-art multimodal distillation methods and achieves up to 6.8% absolute gains in retrieval accuracy using only 500 synthetic pairs.
>
---
#### [new 033] Beyond Frequency: Scoring-Driven Debiasing for Object Detection via Blueprint-Prompted Image Synthesis
- **分类: cs.CV**

- **简介: 该论文针对目标检测中的数据偏见问题，提出基于生成图像的去偏框架。通过引入表征分数指导无偏布局生成，并用视觉蓝图替代文本提示以提升合成质量，结合生成对齐策略，有效提升稀有和大物体的检测性能。**

- **链接: [http://arxiv.org/pdf/2510.18229v1](http://arxiv.org/pdf/2510.18229v1)**

> **作者:** Xinhao Cai; Liulei Li; Gensheng Pei; Tao Chen; Jinshan Pan; Yazhou Yao; Wenguan Wang
>
> **摘要:** This paper presents a generation-based debiasing framework for object detection. Prior debiasing methods are often limited by the representation diversity of samples, while naive generative augmentation often preserves the biases it aims to solve. Moreover, our analysis reveals that simply generating more data for rare classes is suboptimal due to two core issues: i) instance frequency is an incomplete proxy for the true data needs of a model, and ii) current layout-to-image synthesis lacks the fidelity and control to generate high-quality, complex scenes. To overcome this, we introduce the representation score (RS) to diagnose representational gaps beyond mere frequency, guiding the creation of new, unbiased layouts. To ensure high-quality synthesis, we replace ambiguous text prompts with a precise visual blueprint and employ a generative alignment strategy, which fosters communication between the detector and generator. Our method significantly narrows the performance gap for underrepresented object groups, \eg, improving large/rare instances by 4.4/3.6 mAP over the baseline, and surpassing prior L2I synthesis models by 15.9 mAP for layout accuracy in generated images.
>
---
#### [new 034] Learning Human-Object Interaction as Groups
- **分类: cs.CV**

- **简介: 该论文研究人-物交互检测（HOI-DET），旨在建模群体中多人多物的集体交互行为。针对现有方法忽略高阶群体交互的问题，提出GroupHOI框架，通过几何 proximity 和语义相似性分组建模上下文关系，提升复杂交互检测性能。**

- **链接: [http://arxiv.org/pdf/2510.18357v1](http://arxiv.org/pdf/2510.18357v1)**

> **作者:** Jiajun Hong; Jianan Wei; Wenguan Wang
>
> **摘要:** Human-Object Interaction Detection (HOI-DET) aims to localize human-object pairs and identify their interactive relationships. To aggregate contextual cues, existing methods typically propagate information across all detected entities via self-attention mechanisms, or establish message passing between humans and objects with bipartite graphs. However, they primarily focus on pairwise relationships, overlooking that interactions in real-world scenarios often emerge from collective behaviors (multiple humans and objects engaging in joint activities). In light of this, we revisit relation modeling from a group view and propose GroupHOI, a framework that propagates contextual information in terms of geometric proximity and semantic similarity. To exploit the geometric proximity, humans and objects are grouped into distinct clusters using a learnable proximity estimator based on spatial features derived from bounding boxes. In each group, a soft correspondence is computed via self-attention to aggregate and dispatch contextual cues. To incorporate the semantic similarity, we enhance the vanilla transformer-based interaction decoder with local contextual cues from HO-pair features. Extensive experiments on HICO-DET and V-COCO benchmarks demonstrate the superiority of GroupHOI over the state-of-the-art methods. It also exhibits leading performance on the more challenging Nonverbal Interaction Detection (NVI-DET) task, which involves varied forms of higher-order interactions within groups.
>
---
#### [new 035] UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成模型的语义一致性评估问题，提出UniGenBench++，一个涵盖多场景、多语言、细粒度评价标准的统一基准，通过多模态大模型构建评估流程，并提供离线评估模型，系统评测现有T2I模型的优缺点。**

- **链接: [http://arxiv.org/pdf/2510.18701v1](http://arxiv.org/pdf/2510.18701v1)**

> **作者:** Yibin Wang; Zhimin Li; Yuhang Zang; Jiazi Bu; Yujie Zhou; Yi Xin; Junjun He; Chunyu Wang; Qinglin Lu; Cheng Jin; Jiaqi Wang
>
> **备注:** Project page: codegoat24.github.io/UniGenBench/
>
> **摘要:** Recent progress in text-to-image (T2I) generation underscores the importance of reliable benchmarks in evaluating how accurately generated images reflect the semantics of their textual prompt. However, (1) existing benchmarks lack the diversity of prompt scenarios and multilingual support, both essential for real-world applicability; (2) they offer only coarse evaluations across primary dimensions, covering a narrow range of sub-dimensions, and fall short in fine-grained sub-dimension assessment. To address these limitations, we introduce UniGenBench++, a unified semantic assessment benchmark for T2I generation. Specifically, it comprises 600 prompts organized hierarchically to ensure both coverage and efficiency: (1) spans across diverse real-world scenarios, i.e., 5 main prompt themes and 20 subthemes; (2) comprehensively probes T2I models' semantic consistency over 10 primary and 27 sub evaluation criteria, with each prompt assessing multiple testpoints. To rigorously assess model robustness to variations in language and prompt length, we provide both English and Chinese versions of each prompt in short and long forms. Leveraging the general world knowledge and fine-grained image understanding capabilities of a closed-source Multi-modal Large Language Model (MLLM), i.e., Gemini-2.5-Pro, an effective pipeline is developed for reliable benchmark construction and streamlined model assessment. Moreover, to further facilitate community use, we train a robust evaluation model that enables offline assessment of T2I model outputs. Through comprehensive benchmarking of both open- and closed-sourced T2I models, we systematically reveal their strengths and weaknesses across various aspects.
>
---
#### [new 036] Ranking-based Preference Optimization for Diffusion Models from Implicit User Feedback
- **分类: cs.CV**

- **简介: 该论文研究文本到图像扩散模型的偏好优化任务，旨在解决现有DPO方法因非线性估计和离线数据局限导致的训练偏差问题。作者提出Diffusion-DRO，将偏好学习转化为排名问题，结合离线专家数据与在线负样本，通过去噪目标直接优化生成质量，提升模型对人类偏好的对齐能力。**

- **链接: [http://arxiv.org/pdf/2510.18353v1](http://arxiv.org/pdf/2510.18353v1)**

> **作者:** Yi-Lun Wu; Bo-Kai Ruan; Chiang Tseng; Hong-Han Shuai
>
> **摘要:** Direct preference optimization (DPO) methods have shown strong potential in aligning text-to-image diffusion models with human preferences by training on paired comparisons. These methods improve training stability by avoiding the REINFORCE algorithm but still struggle with challenges such as accurately estimating image probabilities due to the non-linear nature of the sigmoid function and the limited diversity of offline datasets. In this paper, we introduce Diffusion Denoising Ranking Optimization (Diffusion-DRO), a new preference learning framework grounded in inverse reinforcement learning. Diffusion-DRO removes the dependency on a reward model by casting preference learning as a ranking problem, thereby simplifying the training objective into a denoising formulation and overcoming the non-linear estimation issues found in prior methods. Moreover, Diffusion-DRO uniquely integrates offline expert demonstrations with online policy-generated negative samples, enabling it to effectively capture human preferences while addressing the limitations of offline data. Comprehensive experiments show that Diffusion-DRO delivers improved generation quality across a range of challenging and unseen prompts, outperforming state-of-the-art baselines in both both quantitative metrics and user studies. Our source code and pre-trained models are available at https://github.com/basiclab/DiffusionDRO.
>
---
#### [new 037] The Impact of Image Resolution on Biomedical Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究图像分辨率对生物医学多模态大模型的影响，旨在解决低分辨率导致信息丢失的问题。通过分析不同分辨率下的模型表现，提出原生分辨率推理和混合分辨率训练策略，以提升性能并平衡计算成本。**

- **链接: [http://arxiv.org/pdf/2510.18304v1](http://arxiv.org/pdf/2510.18304v1)**

> **作者:** Liangyu Chen; James Burgess; Jeffrey J Nirschl; Orr Zohar; Serena Yeung-Levy
>
> **备注:** Proceedings of the 10th Machine Learning for Healthcare Conference, PMLR 298, 2025
>
> **摘要:** Imaging technologies are fundamental to biomedical research and modern medicine, requiring analysis of high-resolution images across various modalities. While multimodal large language models (MLLMs) show promise for biomedical image analysis, most are designed for low-resolution images from general-purpose datasets, risking critical information loss. We investigate how image resolution affects MLLM performance in biomedical applications and demonstrate that: (1) native-resolution training and inference significantly improve performance across multiple tasks, (2) misalignment between training and inference resolutions severely degrades performance, and (3) mixed-resolution training effectively mitigates misalignment and balances computational constraints with performance requirements. Based on these findings, we recommend prioritizing native-resolution inference and mixed-resolution datasets to optimize biomedical MLLMs for transformative impact in scientific research and clinical applications.
>
---
#### [new 038] VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦多模态AI安全评估任务，旨在解决现有方法忽视图文联合理解风险及难以区分安全边界的问题。作者提出VLSU框架，构建含8,187样本的大规模基准，通过细粒度分类和组合分析揭示模型在联合推理下的安全判断缺陷。**

- **链接: [http://arxiv.org/pdf/2510.18214v1](http://arxiv.org/pdf/2510.18214v1)**

> **作者:** Shruti Palaskar; Leon Gatys; Mona Abdelrahman; Mar Jacobo; Larry Lindsey; Rutika Moharir; Gunnar Lund; Yang Xu; Navid Shiee; Jeffrey Bigham; Charles Maalouf; Joseph Yitan Cheng
>
> **备注:** 10 pages, 5 figures, 4 tables. Under review
>
> **摘要:** Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90%-plus accuracy on clear unimodal safety signals, performance degrades substantially to 20-55% when joint image-text reasoning is required to determine the safety label. Most critically, 34% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from 62.4% to 10.4% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8% to 53.9%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision-language safety.
>
---
#### [new 039] StreamingTOM: Streaming Token Compression for Efficient Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究流式视频理解任务，解决因果性与token累积导致的效率瓶颈。提出StreamingTOM框架，通过因果时序压缩和在线量化记忆，在不训练的前提下显著降低内存与延迟，保持高精度。**

- **链接: [http://arxiv.org/pdf/2510.18269v1](http://arxiv.org/pdf/2510.18269v1)**

> **作者:** Xueyi Chen; Keda Tao; Kele Shao; Huan Wang
>
> **摘要:** Unlike offline processing, streaming video vision-language models face two fundamental constraints: causality and accumulation. Causality prevents access to future frames that offline methods exploit, while accumulation causes tokens to grow unbounded, creating efficiency bottlenecks. However, existing approaches only regulate post-LLM kv-cache, leaving costly pre-LLM prefill unchanged. We introduce StreamingTOM, a training-free, plug-and-play two-stage framework that addresses both pre-LLM and post-LLM bottlenecks with predictable latency. Causal Temporal Reduction imposes a fixed per-frame budget and selects tokens based on adjacent-frame changes and token saliency, drastically reducing per-frame prefill cost by processing only a compact subset of visual tokens per frame instead of all visual tokens. Online Quantized Memory stores tokens in 4-bit format, retrieves relevant groups on demand, and dequantizes them, keeping the active kv-cache bounded regardless of stream length. Experiments demonstrate our method achieves $15.7\times$ kv-cache compression, $1.2\times$ lower peak memory and $2\times$ faster TTFT compared to prior SOTA. StreamingTOM maintains state-of-the-art accuracy among training-free methods with an average of $63.8\%$ on offline benchmarks and $55.8\%/3.7$ on RVS. These results highlight the practical benefits of our two-stage approach for efficient streaming video understanding with bounded growth.
>
---
#### [new 040] Big Data, Tiny Targets: An Exploratory Study in Machine Learning-enhanced Detection of Microplastic from Filters
- **分类: cs.CV**

- **简介: 该论文研究基于SEM图像的微塑料检测，属目标检测任务。旨在解决微塑料因尺寸微小导致检测困难、传统方法效率低的问题。工作结合机器学习（YOLO模型）与滤膜图像分析，探索自动化检测可行性，并分析预处理优化及标注数据不足等挑战。**

- **链接: [http://arxiv.org/pdf/2510.18089v1](http://arxiv.org/pdf/2510.18089v1)**

> **作者:** Paul-Tiberiu Miclea; Martin Sboron; Hardik Vaghasiya; Hoang Thinh Nguyen; Meet Gadara; Thomas Schmid
>
> **摘要:** Microplastics (MPs) are ubiquitous pollutants with demonstrated potential to impact ecosystems and human health. Their microscopic size complicates detection, classification, and removal, especially in biological and environmental samples. While techniques like optical microscopy, Scanning Electron Microscopy (SEM), and Atomic Force Microscopy (AFM) provide a sound basis for detection, applying these approaches requires usually manual analysis and prevents efficient use in large screening studies. To this end, machine learning (ML) has emerged as a powerful tool in advancing microplastic detection. In this exploratory study, we investigate potential, limitations and future directions of advancing the detection and quantification of MP particles and fibres using a combination of SEM imaging and machine learning-based object detection. For simplicity, we focus on a filtration scenario where image backgrounds exhibit a symmetric and repetitive pattern. Our findings indicate differences in the quality of YOLO models for the given task and the relevance of optimizing preprocessing. At the same time, we identify open challenges, such as limited amounts of expert-labeled data necessary for reliable training of ML models.
>
---
#### [new 041] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework
- **分类: cs.CV**

- **简介: 该论文研究图表示学习中的图Transformer（GT）统一建模问题，提出层次化掩码框架，揭示架构与掩码的等价性，并据此设计M3Dphormer模型，通过多级掩码与双注意力机制提升性能与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.18825v1](http://arxiv.org/pdf/2510.18825v1)**

> **作者:** Yujie Xing; Xiao Wang; Bin Wu; Hai Huang; Chuan Shi
>
> **备注:** Accepted by NeurIPS 2025 (Poster)
>
> **摘要:** Graph Transformers (GTs) have emerged as a powerful paradigm for graph representation learning due to their ability to model diverse node interactions. However, existing GTs often rely on intricate architectural designs tailored to specific interactions, limiting their flexibility. To address this, we propose a unified hierarchical mask framework that reveals an underlying equivalence between model architecture and attention mask construction. This framework enables a consistent modeling paradigm by capturing diverse interactions through carefully designed attention masks. Theoretical analysis under this framework demonstrates that the probability of correct classification positively correlates with the receptive field size and label consistency, leading to a fundamental design principle: an effective attention mask should ensure both a sufficiently large receptive field and a high level of label consistency. While no single existing mask satisfies this principle across all scenarios, our analysis reveals that hierarchical masks offer complementary strengths, motivating their effective integration. Then, we introduce M3Dphormer, a Mixture-of-Experts-based Graph Transformer with Multi-Level Masking and Dual Attention Computation. M3Dphormer incorporates three theoretically grounded hierarchical masks and employs a bi-level expert routing mechanism to adaptively integrate multi-level interaction information. To ensure scalability, we further introduce a dual attention computation scheme that dynamically switches between dense and sparse modes based on local mask sparsity. Extensive experiments across multiple benchmarks demonstrate that M3Dphormer achieves state-of-the-art performance, validating the effectiveness of our unified framework and model design.
>
---
#### [new 042] Proactive Reasoning-with-Retrieval Framework for Medical Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对医学多模态大模型推理中知识局限导致的幻觉问题，提出首个多模态医学推理-检索框架Med-RwR，通过视觉与文本信息协同进行主动知识检索，并设计置信度驱动的图像重检索机制，提升诊断准确性和跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.18303v1](http://arxiv.org/pdf/2510.18303v1)**

> **作者:** Lehan Wang; Yi Qin; Honglong Yang; Xiaomeng Li
>
> **备注:** Work in progress
>
> **摘要:** Incentivizing the reasoning ability of Multimodal Large Language Models (MLLMs) is essential for medical applications to transparently analyze medical scans and provide reliable diagnosis. However, existing medical MLLMs rely solely on internal knowledge during reasoning, leading to hallucinated reasoning and factual inaccuracies when encountering cases beyond their training scope. Although recent Agentic Retrieval-Augmented Generation (RAG) methods elicit the medical model's proactive retrieval ability during reasoning, they are confined to unimodal LLMs, neglecting the crucial visual information during reasoning and retrieval. Consequently, we propose the first Multimodal Medical Reasoning-with-Retrieval framework, Med-RwR, which actively retrieves external knowledge by querying observed symptoms or domain-specific medical concepts during reasoning. Specifically, we design a two-stage reinforcement learning strategy with tailored rewards that stimulate the model to leverage both visual diagnostic findings and textual clinical information for effective retrieval. Building on this foundation, we further propose a Confidence-Driven Image Re-retrieval (CDIR) method for test-time scaling when low prediction confidence is detected. Evaluation on various public medical benchmarks demonstrates Med-RwR's significant improvements over baseline models, proving the effectiveness of enhancing reasoning capabilities with external knowledge integration. Furthermore, Med-RwR demonstrates remarkable generalizability to unfamiliar domains, evidenced by 8.8% performance gain on our proposed EchoCardiography Benchmark (ECBench), despite the scarcity of echocardiography data in the training corpus. Our data, model, and codes will be made publicly available at https://github.com/xmed-lab/Med-RwR.
>
---
#### [new 043] Investigating Demographic Bias in Brain MRI Segmentation: A Comparative Study of Deep-Learning and Non-Deep-Learning Methods
- **分类: cs.CV**

- **简介: 该论文研究脑MRI分割中的人口统计学偏见，比较深度学习与传统方法在不同种族和性别群体中的表现。使用四种模型对伏隔核进行分割，评估其公平性及体积差异，发现某些模型在种族匹配数据上表现更好，而nnU-Net具有更强的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.17999v1](http://arxiv.org/pdf/2510.17999v1)**

> **作者:** Ghazal Danaee; Marc Niethammer; Jarrett Rushmore; Sylvain Bouix
>
> **摘要:** Deep-learning-based segmentation algorithms have substantially advanced the field of medical image analysis, particularly in structural delineations in MRIs. However, an important consideration is the intrinsic bias in the data. Concerns about unfairness, such as performance disparities based on sensitive attributes like race and sex, are increasingly urgent. In this work, we evaluate the results of three different segmentation models (UNesT, nnU-Net, and CoTr) and a traditional atlas-based method (ANTs), applied to segment the left and right nucleus accumbens (NAc) in MRI images. We utilize a dataset including four demographic subgroups: black female, black male, white female, and white male. We employ manually labeled gold-standard segmentations to train and test segmentation models. This study consists of two parts: the first assesses the segmentation performance of models, while the second measures the volumes they produce to evaluate the effects of race, sex, and their interaction. Fairness is quantitatively measured using a metric designed to quantify fairness in segmentation performance. Additionally, linear mixed models analyze the impact of demographic variables on segmentation accuracy and derived volumes. Training on the same race as the test subjects leads to significantly better segmentation accuracy for some models. ANTs and UNesT show notable improvements in segmentation accuracy when trained and tested on race-matched data, unlike nnU-Net, which demonstrates robust performance independent of demographic matching. Finally, we examine sex and race effects on the volume of the NAc using segmentations from the manual rater and from our biased models. Results reveal that the sex effects observed with manual segmentation can also be observed with biased models, whereas the race effects disappear in all but one model.
>
---
#### [new 044] Latent-Info and Low-Dimensional Learning for Human Mesh Recovery and Parallel Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究3D人体网格重建任务，旨在解决现有方法在复杂场景下肢体错位、细节不足及计算成本高的问题。提出两阶段网络，通过隐含信息挖掘与低维并行优化，提升重建精度并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.18267v1](http://arxiv.org/pdf/2510.18267v1)**

> **作者:** Xiang Zhang; Suping Wu; Sheng Yang
>
> **备注:** Accepted by ICME2025
>
> **摘要:** Existing 3D human mesh recovery methods often fail to fully exploit the latent information (e.g., human motion, shape alignment), leading to issues with limb misalignment and insufficient local details in the reconstructed human mesh (especially in complex scenes). Furthermore, the performance improvement gained by modelling mesh vertices and pose node interactions using attention mechanisms comes at a high computational cost. To address these issues, we propose a two-stage network for human mesh recovery based on latent information and low dimensional learning. Specifically, the first stage of the network fully excavates global (e.g., the overall shape alignment) and local (e.g., textures, detail) information from the low and high-frequency components of image features and aggregates this information into a hybrid latent frequency domain feature. This strategy effectively extracts latent information. Subsequently, utilizing extracted hybrid latent frequency domain features collaborates to enhance 2D poses to 3D learning. In the second stage, with the assistance of hybrid latent features, we model the interaction learning between the rough 3D human mesh template and the 3D pose, optimizing the pose and shape of the human mesh. Unlike existing mesh pose interaction methods, we design a low-dimensional mesh pose interaction method through dimensionality reduction and parallel optimization that significantly reduces computational costs without sacrificing reconstruction accuracy. Extensive experimental results on large publicly available datasets indicate superiority compared to the most state-of-the-art.
>
---
#### [new 045] Efficient Few-shot Identity Preserving Attribute Editing for 3D-aware Deep Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究3D人脸属性编辑任务，旨在解决少样本下保持身份的3D一致编辑难题。作者基于3D感知生成模型，提出通过少量标注图像学习潜在空间编辑方向，实现高效、保身份的多属性编辑，并验证了编辑方向的线性与连续性。**

- **链接: [http://arxiv.org/pdf/2510.18287v1](http://arxiv.org/pdf/2510.18287v1)**

> **作者:** Vishal Vinod
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Identity preserving editing of faces is a generative task that enables modifying the illumination, adding/removing eyeglasses, face aging, editing hairstyles, modifying expression etc., while preserving the identity of the face. Recent progress in 2D generative models have enabled photorealistic editing of faces using simple techniques leveraging the compositionality in GANs. However, identity preserving editing for 3D faces with a given set of attributes is a challenging task as the generative model must reason about view consistency from multiple poses and render a realistic 3D face. Further, 3D portrait editing requires large-scale attribute labelled datasets and presents a trade-off between editability in low-resolution and inflexibility to editing in high resolution. In this work, we aim to alleviate some of the constraints in editing 3D faces by identifying latent space directions that correspond to photorealistic edits. To address this, we present a method that builds on recent advancements in 3D-aware deep generative models and 2D portrait editing techniques to perform efficient few-shot identity preserving attribute editing for 3D-aware generative models. We aim to show from experimental results that using just ten or fewer labelled images of an attribute is sufficient to estimate edit directions in the latent space that correspond to 3D-aware attribute editing. In this work, we leverage an existing face dataset with masks to obtain the synthetic images for few attribute examples required for estimating the edit directions. Further, to demonstrate the linearity of edits, we investigate one-shot stylization by performing sequential editing and use the (2D) Attribute Style Manipulation (ASM) technique to investigate a continuous style manifold for 3D consistent identity preserving face aging. Code and results are available at: https://vishal-vinod.github.io/gmpi-edit/
>
---
#### [new 046] MUSE: Model-based Uncertainty-aware Similarity Estimation for zero-shot 2D Object Detection and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MUSE，用于零样本2D物体检测与分割。针对无需训练的3D到2D跨模态匹配问题，MUSE利用多视角模板与图像候选框，结合类-块嵌入、联合相似性度量和不确定性感知先验，实现高效准确的检测与分割，在BOP 2025多个赛道排名第一。**

- **链接: [http://arxiv.org/pdf/2510.17866v1](http://arxiv.org/pdf/2510.17866v1)**

> **作者:** Sungmin Cho; Sungbum Park; Insoo Oh
>
> **备注:** 11 pages with 6 figures
>
> **摘要:** In this work, we introduce MUSE (Model-based Uncertainty-aware Similarity Estimation), a training-free framework designed for model-based zero-shot 2D object detection and segmentation. MUSE leverages 2D multi-view templates rendered from 3D unseen objects and 2D object proposals extracted from input query images. In the embedding stage, it integrates class and patch embeddings, where the patch embeddings are normalized using generalized mean pooling (GeM) to capture both global and local representations efficiently. During the matching stage, MUSE employs a joint similarity metric that combines absolute and relative similarity scores, enhancing the robustness of matching under challenging scenarios. Finally, the similarity score is refined through an uncertainty-aware object prior that adjusts for proposal reliability. Without any additional training or fine-tuning, MUSE achieves state-of-the-art performance on the BOP Challenge 2025, ranking first across the Classic Core, H3, and Industrial tracks. These results demonstrate that MUSE offers a powerful and generalizable framework for zero-shot 2D object detection and segmentation.
>
---
#### [new 047] Kaleido: Open-Sourced Multi-Subject Reference Video Generation Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决多主体参考视频生成中主体一致性差和背景解耦难的问题。作者提出Kaleido框架，构建高质量数据 pipeline 并设计R-RoPE编码机制，提升多图输入下的主体一致性和生成保真度。**

- **链接: [http://arxiv.org/pdf/2510.18573v1](http://arxiv.org/pdf/2510.18573v1)**

> **作者:** Zhenxing Zhang; Jiayan Teng; Zhuoyi Yang; Tiankun Cao; Cheng Wang; Xiaotao Gu; Jie Tang; Dan Guo; Meng Wang
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** We present Kaleido, a subject-to-video~(S2V) generation framework, which aims to synthesize subject-consistent videos conditioned on multiple reference images of target subjects. Despite recent progress in S2V generation models, existing approaches remain inadequate at maintaining multi-subject consistency and at handling background disentanglement, often resulting in lower reference fidelity and semantic drift under multi-image conditioning. These shortcomings can be attributed to several factors. Primarily, the training dataset suffers from a lack of diversity and high-quality samples, as well as cross-paired data, i.e., paired samples whose components originate from different instances. In addition, the current mechanism for integrating multiple reference images is suboptimal, potentially resulting in the confusion of multiple subjects. To overcome these limitations, we propose a dedicated data construction pipeline, incorporating low-quality sample filtering and diverse data synthesis, to produce consistency-preserving training data. Moreover, we introduce Reference Rotary Positional Encoding (R-RoPE) to process reference images, enabling stable and precise multi-image integration. Extensive experiments across numerous benchmarks demonstrate that Kaleido significantly outperforms previous methods in consistency, fidelity, and generalization, marking an advance in S2V generation.
>
---
#### [new 048] Robotic Classification of Divers' Swimming States using Visual Pose Keypoints as IMUs
- **分类: cs.CV; cs.RO**

- **简介: 该论文属水下行为识别任务，旨在解决传统IMU在水下通信受限的问题。作者提出用视觉生成3D姿态关键点作为伪IMU输入，构建分类器检测潜水员异常行为，集成于AUV实现安全监控。**

- **链接: [http://arxiv.org/pdf/2510.17863v1](http://arxiv.org/pdf/2510.17863v1)**

> **作者:** Demetrious T. Kutzke; Ying-Kun Wu; Elizabeth Terveen; Junaed Sattar
>
> **摘要:** Traditional human activity recognition uses either direct image analysis or data from wearable inertial measurement units (IMUs), but can be ineffective in challenging underwater environments. We introduce a novel hybrid approach that bridges this gap to monitor scuba diver safety. Our method leverages computer vision to generate high-fidelity motion data, effectively creating a ``pseudo-IMU'' from a stream of 3D human joint keypoints. This technique circumvents the critical problem of wireless signal attenuation in water, which plagues conventional diver-worn sensors communicating with an Autonomous Underwater Vehicle (AUV). We apply this system to the vital task of identifying anomalous scuba diver behavior that signals the onset of a medical emergency such as cardiac arrest -- a leading cause of scuba diving fatalities. By integrating our classifier onboard an AUV and conducting experiments with simulated distress scenarios, we demonstrate the utility and effectiveness of our method for advancing robotic monitoring and diver safety.
>
---
#### [new 049] ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究稀疏标注下的显微图像语义分割任务，旨在解决标注极少时的细胞结构分割难题。作者提出ε-Seg方法，结合分层变分自编码器、中心区域掩码、对比学习与高斯混合模型先验，通过MLP头直接预测标签，实现高效稀疏监督分割。**

- **链接: [http://arxiv.org/pdf/2510.18637v1](http://arxiv.org/pdf/2510.18637v1)**

> **作者:** Sheida Rahnamai Kordasiabi; Damian Dalle Nogare; Florian Jug
>
> **备注:** 10 pages main text, 17 pages total
>
> **摘要:** Semantic segmentation of electron microscopy (EM) images of biological samples remains a challenge in the life sciences. EM data captures details of biological structures, sometimes with such complexity that even human observers can find it overwhelming. We introduce {\epsilon}-Seg, a method based on hierarchical variational autoencoders (HVAEs), employing center-region masking, sparse label contrastive learning (CL), a Gaussian mixture model (GMM) prior, and clustering-free label prediction. Center-region masking and the inpainting loss encourage the model to learn robust and representative embeddings to distinguish the desired classes, even if training labels are sparse (0.05% of the total image data or less). For optimal performance, we employ CL and a GMM prior to shape the latent space of the HVAE such that encoded input patches tend to cluster wrt. the semantic classes we wish to distinguish. Finally, instead of clustering latent embeddings for semantic segmentation, we propose a MLP semantic segmentation head to directly predict class labels from latent embeddings. We show empirical results of {\epsilon}-Seg and baseline methods on 2 dense EM datasets of biological tissues and demonstrate the applicability of our method also on fluorescence microscopy data. Our results show that {\epsilon}-Seg is capable of achieving competitive sparsely-supervised segmentation results on complex biological image data, even if only limited amounts of training labels are available.
>
---
#### [new 050] Beyond Single Models: Mitigating Multimodal Hallucinations via Adaptive Token Ensemble Decoding
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型中的多模态幻觉问题，提出一种无需训练的自适应令牌集成解码方法（ATED），通过动态加权集成多个模型的预测，缓解对象幻觉，提升生成可靠性。**

- **链接: [http://arxiv.org/pdf/2510.18321v1](http://arxiv.org/pdf/2510.18321v1)**

> **作者:** Jinlin Li; Yuran Wang; Yifei Yuan; Xiao Zhou; Yingying Zhang; Xixian Yong; Yefeng Zheng; Xian Wu
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently achieved impressive results in multimodal tasks such as image captioning and visual question answering. However, they remain prone to object hallucination -- generating descriptions of nonexistent or misidentified objects. Prior work has partially mitigated this via auxiliary training objectives or external modules, but challenges remain in terms of scalability, adaptability, and model independence. To address these limitations, we propose Adaptive Token Ensemble Decoding (ATED), a training-free, token-level ensemble framework that mitigates hallucination by aggregating predictions from multiple LVLMs during inference. ATED dynamically computes uncertainty-based weights for each model, reflecting their reliability at each decoding step. It also integrates diverse decoding paths to improve contextual grounding and semantic consistency. Experiments on standard hallucination detection benchmarks demonstrate that ATED significantly outperforms state-of-the-art methods, reducing hallucination without compromising fluency or relevance. Our findings highlight the benefits of adaptive ensembling and point to a promising direction for improving LVLM robustness in high-stakes applications. The code is available at https://github.com/jinlin2021/ATED.
>
---
#### [new 051] World-in-World: World Models in a Closed-Loop World
- **分类: cs.CV**

- **简介: 该论文聚焦生成式世界模型在具身决策中的应用，旨在解决现有评估碎片化、重视觉轻实用的问题。作者提出World-in-World平台，首次实现闭环评估，强调任务成功率，揭示视觉质量不等于任务成功，可控性和数据扩展更关键。**

- **链接: [http://arxiv.org/pdf/2510.18135v1](http://arxiv.org/pdf/2510.18135v1)**

> **作者:** Jiahan Zhang; Muqing Jiang; Nanru Dai; Taiming Lu; Arda Uzunoglu; Shunchi Zhang; Yana Wei; Jiahao Wang; Vishal M. Patel; Paul Pu Liang; Daniel Khashabi; Cheng Peng; Rama Chellappa; Tianmin Shu; Alan Yuille; Yilun Du; Jieneng Chen
>
> **备注:** Code is at https://github.com/World-In-World/world-in-world
>
> **摘要:** Generative world models (WMs) can now simulate worlds with striking visual realism, which naturally raises the question of whether they can endow embodied agents with predictive perception for decision making. Progress on this question has been limited by fragmented evaluation: most existing benchmarks adopt open-loop protocols that emphasize visual quality in isolation, leaving the core issue of embodied utility unresolved, i.e., do WMs actually help agents succeed at embodied tasks? To address this gap, we introduce World-in-World, the first open platform that benchmarks WMs in a closed-loop world that mirrors real agent-environment interactions. World-in-World provides a unified online planning strategy and a standardized action API, enabling heterogeneous WMs for decision making. We curate four closed-loop environments that rigorously evaluate diverse WMs, prioritize task success as the primary metric, and move beyond the common focus on visual quality; we also present the first data scaling law for world models in embodied settings. Our study uncovers three surprises: (1) visual quality alone does not guarantee task success, controllability matters more; (2) scaling post-training with action-observation data is more effective than upgrading the pretrained video generators; and (3) allocating more inference-time compute allows WMs to substantially improve closed-loop performance.
>
---
#### [new 052] ManzaiSet: A Multimodal Dataset of Viewer Responses to Japanese Manzai Comedy
- **分类: cs.CV; cs.MM**

- **简介: 该论文构建了首个面向日本漫才喜剧的多模态观众反应数据集ManzaiSet，旨在缓解情感计算中的西方中心偏见。通过分析241人观看表演时的面部视频与音频，识别出三类观众，并发现观看顺序正向影响情绪反应，为非西方情境下的情感AI与个性化娱乐系统提供支持。**

- **链接: [http://arxiv.org/pdf/2510.18014v1](http://arxiv.org/pdf/2510.18014v1)**

> **作者:** Kazuki Kawamura; Kengo Nakai; Jun Rekimoto
>
> **备注:** ICCV 2025 Workshop on Affective & Behavior Analysis in-the-Wild (ABAW), Honolulu, HI, USA (Oct 19, 2025, HST). 11 pages, 5 figures
>
> **摘要:** We present ManzaiSet, the first large scale multimodal dataset of viewer responses to Japanese manzai comedy, capturing facial videos and audio from 241 participants watching up to 10 professional performances in randomized order (94.6 percent watched >= 8; analyses focus on n=228). This addresses the Western centric bias in affective computing. Three key findings emerge: (1) k means clustering identified three distinct viewer types: High and Stable Appreciators (72.8 percent, n=166), Low and Variable Decliners (13.2 percent, n=30), and Variable Improvers (14.0 percent, n=32), with heterogeneity of variance (Brown Forsythe p < 0.001); (2) individual level analysis revealed a positive viewing order effect (mean slope = 0.488, t(227) = 5.42, p < 0.001, permutation p < 0.001), contradicting fatigue hypotheses; (3) automated humor classification (77 instances, 131 labels) plus viewer level response modeling found no type wise differences after FDR correction. The dataset enables culturally aware emotion AI development and personalized entertainment systems tailored to non Western contexts.
>
---
#### [new 053] Pre to Post-Treatment Glioblastoma MRI Prediction using a Latent Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究治疗前到治疗后的胶质母细胞瘤MRI预测，属医学图像生成任务。旨在早期预测治疗响应，提出基于潜扩散模型的方法，结合预处理MRI与肿瘤位置，利用生存信息提升生成质量。**

- **链接: [http://arxiv.org/pdf/2510.17851v1](http://arxiv.org/pdf/2510.17851v1)**

> **作者:** Alexandre G. Leclercq; Sébastien Bougleux; Noémie N. Moreau; Alexis Desmonts; Romain Hérault; Aurélien Corroyer-Dulmont
>
> **备注:** 10 pages, 4 figures. Presented to the Deep Generative Models Workshop of MICCAI (DGM4MICCAI)
>
> **摘要:** Glioblastoma (GBM) is an aggressive primary brain tumor with a median survival of approximately 15 months. In clinical practice, the Stupp protocol serves as the standard first-line treatment. However, patients exhibit highly heterogeneous therapeutic responses which required at least two months before first visual impact can be observed, typically with MRI. Early prediction treatment response is crucial for advancing personalized medicine. Disease Progression Modeling (DPM) aims to capture the trajectory of disease evolution, while Treatment Response Prediction (TRP) focuses on assessing the impact of therapeutic interventions. Whereas most TRP approaches primarly rely on timeseries data, we consider the problem of early visual TRP as a slice-to-slice translation model generating post-treatment MRI from a pre-treatment MRI, thus reflecting the tumor evolution. To address this problem we propose a Latent Diffusion Model with a concatenation-based conditioning from the pre-treatment MRI and the tumor localization, and a classifier-free guidance to enhance generation quality using survival information, in particular post-treatment tumor evolution. Our model were trained and tested on a local dataset consisting of 140 GBM patients collected at Centre Fran\c{c}ois Baclesse. For each patient we collected pre and post T1-Gd MRI, tumor localization manually delineated in the pre-treatment MRI by medical experts, and survival information.
>
---
#### [new 054] AV-Master: Dual-Path Comprehensive Perception Makes Better Audio-Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文研究音频-视觉问答（AVQA）任务，旨在解决现有方法在时序采样和模态偏好上的僵化问题。作者提出AV-Master框架，通过动态时序聚焦、模态感知策略和双路径对比损失，提升复杂场景下的跨模态推理能力。**

- **链接: [http://arxiv.org/pdf/2510.18346v1](http://arxiv.org/pdf/2510.18346v1)**

> **作者:** Jiayu Zhang; Qilang Ye; Shuo Ye; Xun Lin; Zihan Song; Zitong Yu
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Audio-Visual Question Answering (AVQA) requires models to effectively utilize both visual and auditory modalities to answer complex and diverse questions about audio-visual scenes. However, existing methods lack sufficient flexibility and dynamic adaptability in temporal sampling and modality preference awareness, making it difficult to focus on key information based on the question. This limits their reasoning capability in complex scenarios. To address these challenges, we propose a novel framework named AV-Master. It enhances the model's ability to extract key information from complex audio-visual scenes with substantial redundant content by dynamically modeling both temporal and modality dimensions. In the temporal dimension, we introduce a dynamic adaptive focus sampling mechanism that progressively focuses on audio-visual segments most relevant to the question, effectively mitigating redundancy and segment fragmentation in traditional sampling methods. In the modality dimension, we propose a preference-aware strategy that models each modality's contribution independently, enabling selective activation of critical features. Furthermore, we introduce a dual-path contrastive loss to reinforce consistency and complementarity across temporal and modality dimensions, guiding the model to learn question-specific cross-modal collaborative representations. Experiments on four large-scale benchmarks show that AV-Master significantly outperforms existing methods, especially in complex reasoning tasks.
>
---
#### [new 055] Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对结肠镜3D重建中动态光照导致的渲染失真问题，提出ColIAGS方法。通过引入光照衰减感知的外观建模与高维嵌入几何建模，提升3D高斯点阵在动态光照下的视图合成质量与几何精度。**

- **链接: [http://arxiv.org/pdf/2510.18739v1](http://arxiv.org/pdf/2510.18739v1)**

> **作者:** Hao Wang; Ying Zhou; Haoyu Zhao; Rui Wang; Qiang Hu; Xing Zhang; Qiang Li; Zhiwei Wang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for real-time view synthesis in colonoscopy, enabling critical applications such as virtual colonoscopy and lesion tracking. However, the vanilla 3DGS assumes static illumination and that observed appearance depends solely on viewing angle, which causes incompatibility with the photometric variations in colonoscopic scenes induced by dynamic light source/camera. This mismatch forces most 3DGS methods to introduce structure-violating vaporous Gaussian blobs between the camera and tissues to compensate for illumination attenuation, ultimately degrading the quality of 3D reconstructions. Previous works only consider the illumination attenuation caused by light distance, ignoring the physical characters of light source and camera. In this paper, we propose ColIAGS, an improved 3DGS framework tailored for colonoscopy. To mimic realistic appearance under varying illumination, we introduce an Improved Appearance Modeling with two types of illumination attenuation factors, which enables Gaussians to adapt to photometric variations while preserving geometry accuracy. To ensure the geometry approximation condition of appearance modeling, we propose an Improved Geometry Modeling using high-dimensional view embedding to enhance Gaussian geometry attribute prediction. Furthermore, another cosine embedding input is leveraged to generate illumination attenuation solutions in an implicit manner. Comprehensive experimental results on standard benchmarks demonstrate that our proposed ColIAGS achieves the dual capabilities of novel view synthesis and accurate geometric reconstruction. It notably outperforms other state-of-the-art methods by achieving superior rendering fidelity while significantly reducing Depth MSE. Code will be available.
>
---
#### [new 056] DeepSeek-OCR: Contexts Optical Compression
- **分类: cs.CV**

- **简介: 该论文提出DeepSeek-OCR，致力于长文本上下文的光学压缩任务，旨在通过2D映射降低视觉令牌数量。其设计的DeepEncoder实现高压缩比下高效OCR解码，兼顾精度与效率，适用于历史文档压缩与LLM记忆机制研究，并具备大规模生成训练数据的能力。**

- **链接: [http://arxiv.org/pdf/2510.18234v1](http://arxiv.org/pdf/2510.18234v1)**

> **作者:** Haoran Wei; Yaofeng Sun; Yukun Li
>
> **摘要:** We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10x), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20x, the OCR accuracy still remains at about 60%. This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value. On OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens, and outperforms MinerU2.0 (6000+ tokens per page on average) while utilizing fewer than 800 vision tokens. In production, DeepSeek-OCR can generate training data for LLMs/VLMs at a scale of 200k+ pages per day (a single A100-40G). Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR.
>
---
#### [new 057] SAVANT: Semantic Analysis with Vision-Augmented Anomaly deTection
- **分类: cs.CV; cs.AI; cs.RO; I.2.9; I.4.8**

- **简介: 该论文针对自动驾驶中罕见语义异常场景检测难题，提出SAVANT框架，通过分层视觉-语言结构化分析，提升开源小模型的检测性能，实现高准确率、高召回率的异常检测，并缓解数据稀缺问题。**

- **链接: [http://arxiv.org/pdf/2510.18034v1](http://arxiv.org/pdf/2510.18034v1)**

> **作者:** Roberto Brusnicki; David Pop; Yuan Gao; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution scenarios with semantic anomalies. While Vision Language Models (VLMs) offer promising reasoning capabilities, naive prompting approaches yield unreliable performance and depend on expensive proprietary models, limiting practical deployment. We introduce SAVANT (Semantic Analysis with Vision-Augmented Anomaly deTection), a structured reasoning framework that achieves high accuracy and recall in detecting anomalous driving scenarios from input images through layered scene analysis and a two-phase pipeline: structured scene description extraction followed by multi-modal evaluation. Our approach transforms VLM reasoning from ad-hoc prompting to systematic analysis across four semantic layers: Street, Infrastructure, Movable Objects, and Environment. SAVANT achieves 89.6% recall and 88.0% accuracy on real-world driving scenarios, significantly outperforming unstructured baselines. More importantly, we demonstrate that our structured framework enables a fine-tuned 7B parameter open-source model (Qwen2.5VL) to achieve 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By automatically labeling over 9,640 real-world images with high accuracy, SAVANT addresses the critical data scarcity problem in anomaly detection and provides a practical path toward reliable, accessible semantic monitoring for autonomous systems.
>
---
#### [new 058] ViBED-Net: Video Based Engagement Detection Network Using Face-Aware and Scene-Aware Spatiotemporal Cues
- **分类: cs.CV; cs.LG; I.2.10; I.5.2**

- **简介: 该论文研究在线学习中的学生专注度检测，提出ViBED-Net模型。通过双流架构融合面部与场景时空特征，采用EfficientNetV2提取空间特征，LSTM或Transformer建模时序变化，在DAiSEE数据集上提升小类样本性能，显著提高检测准确率。**

- **链接: [http://arxiv.org/pdf/2510.18016v1](http://arxiv.org/pdf/2510.18016v1)**

> **作者:** Prateek Gothwal; Deeptimaan Banerjee; Ashis Kumer Biswas
>
> **备注:** 10 pages, 4 figures, 2 tables
>
> **摘要:** Engagement detection in online learning environments is vital for improving student outcomes and personalizing instruction. We present ViBED-Net (Video-Based Engagement Detection Network), a novel deep learning framework designed to assess student engagement from video data using a dual-stream architecture. ViBED-Net captures both facial expressions and full-scene context by processing facial crops and entire video frames through EfficientNetV2 for spatial feature extraction. These features are then analyzed over time using two temporal modeling strategies: Long Short-Term Memory (LSTM) networks and Transformer encoders. Our model is evaluated on the DAiSEE dataset, a large-scale benchmark for affective state recognition in e-learning. To enhance performance on underrepresented engagement classes, we apply targeted data augmentation techniques. Among the tested variants, ViBED-Net with LSTM achieves 73.43\% accuracy, outperforming existing state-of-the-art approaches. ViBED-Net demonstrates that combining face-aware and scene-aware spatiotemporal cues significantly improves engagement detection accuracy. Its modular design allows flexibility for application across education, user experience research, and content personalization. This work advances video-based affective computing by offering a scalable, high-performing solution for real-world engagement analysis. The source code for this project is available on https://github.com/prateek-gothwal/ViBED-Net .
>
---
#### [new 059] Hyperbolic Space Learning Method Leveraging Temporal Motion Priors for Human Mesh Recovery
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视频中3D人体网格恢复，旨在解决现有方法难以捕捉人体层次结构的问题。提出在双曲空间中结合时序运动先验进行特征学习与优化，并设计相应损失函数，提升重建精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.18256v1](http://arxiv.org/pdf/2510.18256v1)**

> **作者:** Xiang Zhang; Suping Wu; Weibin Qiu; Zhaocheng Jin; Sheng Yang
>
> **备注:** Accepted by ICME2025
>
> **摘要:** 3D human meshes show a natural hierarchical structure (like torso-limbs-fingers). But existing video-based 3D human mesh recovery methods usually learn mesh features in Euclidean space. It's hard to catch this hierarchical structure accurately. So wrong human meshes are reconstructed. To solve this problem, we propose a hyperbolic space learning method leveraging temporal motion prior for recovering 3D human meshes from videos. First, we design a temporal motion prior extraction module. This module extracts the temporal motion features from the input 3D pose sequences and image feature sequences respectively. Then it combines them into the temporal motion prior. In this way, it can strengthen the ability to express features in the temporal motion dimension. Since data representation in non-Euclidean space has been proved to effectively capture hierarchical relationships in real-world datasets (especially in hyperbolic space), we further design a hyperbolic space optimization learning strategy. This strategy uses the temporal motion prior information to assist learning, and uses 3D pose and pose motion information respectively in the hyperbolic space to optimize and learn the mesh features. Then, we combine the optimized results to get an accurate and smooth human mesh. Besides, to make the optimization learning process of human meshes in hyperbolic space stable and effective, we propose a hyperbolic mesh optimization loss. Extensive experimental results on large publicly available datasets indicate superiority in comparison with most state-of-the-art.
>
---
#### [new 060] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对预训练扩散模型采样步数多、效率低的问题，提出一种基于速度场自蒸馏的后训练加速方法，无需重新设计网络结构或从头训练，可在极短时间内将大模型压缩为几步高效的采样器，几乎零成本实现高性能few-step生成。**

- **链接: [http://arxiv.org/pdf/2510.17858v1](http://arxiv.org/pdf/2510.17858v1)**

> **作者:** Xu Cai; Yang Wu; Qianli Chen; Haoran Wu; Lichuan Xiang; Hongkai Wen
>
> **备注:** NeurIPS 2025
>
> **摘要:** We present an ultra-efficient post-training method for shortcutting large-scale pre-trained flow matching diffusion models into efficient few-step samplers, enabled by novel velocity field self-distillation. While shortcutting in flow matching, originally introduced by shortcut models, offers flexible trajectory-skipping capabilities, it requires a specialized step-size embedding incompatible with existing models unless retraining from scratch$\unicode{x2013}$a process nearly as costly as pretraining itself. Our key contribution is thus imparting a more aggressive shortcut mechanism to standard flow matching models (e.g., Flux), leveraging a unique distillation principle that obviates the need for step-size embedding. Working on the velocity field rather than sample space and learning rapidly from self-guided distillation in an online manner, our approach trains efficiently, e.g., producing a 3-step Flux less than one A100 day. Beyond distillation, our method can be incorporated into the pretraining stage itself, yielding models that inherently learn efficient, few-step flows without compromising quality. This capability also enables, to our knowledge, the first few-shot distillation method (e.g., 10 text-image pairs) for dozen-billion-parameter diffusion models, delivering state-of-the-art performance at almost free cost.
>
---
#### [new 061] SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文研究基于自然语言的协同驾驶系统全栈安全问题，提出SafeCoop防御框架，通过语义防火墙、语言-感知一致性检测和多源共识机制，抵御通信中的各类攻击，在CARLA仿真中显著提升安全性和恶意行为检测性能。**

- **链接: [http://arxiv.org/pdf/2510.18123v1](http://arxiv.org/pdf/2510.18123v1)**

> **作者:** Xiangbo Gao; Tzu-Hsiang Lin; Ruojing Song; Yuheng Wu; Kuan-Ru Huang; Zicheng Jin; Fangzhou Lin; Shinan Liu; Zhengzhong Tu
>
> **摘要:** Collaborative driving systems leverage vehicle-to-everything (V2X) communication across multiple agents to enhance driving safety and efficiency. Traditional V2X systems take raw sensor data, neural features, or perception results as communication media, which face persistent challenges, including high bandwidth demands, semantic loss, and interoperability issues. Recent advances investigate natural language as a promising medium, which can provide semantic richness, decision-level reasoning, and human-machine interoperability at significantly lower bandwidth. Despite great promise, this paradigm shift also introduces new vulnerabilities within language communication, including message loss, hallucinations, semantic manipulation, and adversarial attacks. In this work, we present the first systematic study of full-stack safety and security issues in natural-language-based collaborative driving. Specifically, we develop a comprehensive taxonomy of attack strategies, including connection disruption, relay/replay interference, content spoofing, and multi-connection forgery. To mitigate these risks, we introduce an agentic defense pipeline, which we call SafeCoop, that integrates a semantic firewall, language-perception consistency checks, and multi-source consensus, enabled by an agentic transformation function for cross-frame spatial alignment. We systematically evaluate SafeCoop in closed-loop CARLA simulation across 32 critical scenarios, achieving 69.15% driving score improvement under malicious attacks and up to 67.32% F1 score for malicious detection. This study provides guidance for advancing research on safe, secure, and trustworthy language-driven collaboration in transportation systems. Our project page is https://xiangbogaobarry.github.io/SafeCoop.
>
---
#### [new 062] C-SWAP: Explainability-Aware Structured Pruning for Efficient Neural Networks Compression
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究神经网络压缩中的结构化剪枝任务，旨在解决传统方法需反复重训练或一次性剪枝导致性能下降的问题。作者提出C-SWAP框架，结合可解释性与因果感知的渐进剪枝策略，在无需微调的情况下实现高效压缩与性能平衡。**

- **链接: [http://arxiv.org/pdf/2510.18636v1](http://arxiv.org/pdf/2510.18636v1)**

> **作者:** Baptiste Bauvin; Loïc Baret; Ola Ahmad
>
> **备注:** 10 pages, BMVC2025
>
> **摘要:** Neural network compression has gained increasing attention in recent years, particularly in computer vision applications, where the need for model reduction is crucial for overcoming deployment constraints. Pruning is a widely used technique that prompts sparsity in model structures, e.g. weights, neurons, and layers, reducing size and inference costs. Structured pruning is especially important as it allows for the removal of entire structures, which further accelerates inference time and reduces memory overhead. However, it can be computationally expensive, requiring iterative retraining and optimization. To overcome this problem, recent methods considered one-shot setting, which applies pruning directly at post-training. Unfortunately, they often lead to a considerable drop in performance. In this paper, we focus on this issue by proposing a novel one-shot pruning framework that relies on explainable deep learning. First, we introduce a causal-aware pruning approach that leverages cause-effect relations between model predictions and structures in a progressive pruning process. It allows us to efficiently reduce the size of the network, ensuring that the removed structures do not deter the performance of the model. Then, through experiments conducted on convolution neural network and vision transformer baselines, pre-trained on classification tasks, we demonstrate that our method consistently achieves substantial reductions in model size, with minimal impact on performance, and without the need for fine-tuning. Overall, our approach outperforms its counterparts, offering the best trade-off. Our code is available on GitHub.
>
---
#### [new 063] Image augmentation with invertible networks in interactive satellite image change detection
- **分类: cs.CV**

- **简介: 该论文针对卫星图像变化检测任务，提出基于主动学习的交互式算法。通过设计可逆网络将图像映射到线性化潜在空间进行数据增强，并反馈至模型迭代更新，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.18660v1](http://arxiv.org/pdf/2510.18660v1)**

> **作者:** Hichem Sahbi
>
> **摘要:** This paper devises a novel interactive satellite image change detection algorithm based on active learning. Our framework employs an iterative process that leverages a question-and-answer model. This model queries the oracle (user) about the labels of a small subset of images (dubbed as display), and based on the oracle's responses, change detection model is dynamically updated. The main contribution of our framework resides in a novel invertible network that allows augmenting displays, by mapping them from highly nonlinear input spaces to latent ones, where augmentation transformations become linear and more tractable. The resulting augmented data are afterwards mapped back to the input space, and used to retrain more effective change detection criteria in the subsequent iterations of active learning. Experimental results demonstrate superior performance of our proposed method compared to the related work.
>
---
#### [new 064] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos
- **分类: cs.CV**

- **简介: 该论文研究从无姿态单目交替曝光视频重建可渲染的4D高动态范围场景。提出Mono4D7GS-HDR，采用两阶段优化的高斯溅射框架，实现无需初始位姿的HDR时序一致重建，并构建新基准验证方法优越性。**

- **链接: [http://arxiv.org/pdf/2510.18489v1](http://arxiv.org/pdf/2510.18489v1)**

> **作者:** Jinfeng Liu; Lingtong Kong; Mi Zhou; Jinwen Chen; Dan Xu
>
> **备注:** Project page is available at https://liujf1226.github.io/Mono4DGS-HDR/
>
> **摘要:** We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.
>
---
#### [new 065] SSD: Spatial-Semantic Head Decoupling for Efficient Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文研究自回归图像生成中的高效推理问题，针对视觉token多导致的高内存与计算开销，提出一种KV缓存压缩方法。通过解耦注意力头为空间局部型和语义汇聚型，实现显著加速与降耗。**

- **链接: [http://arxiv.org/pdf/2510.18716v1](http://arxiv.org/pdf/2510.18716v1)**

> **作者:** Siyong Jian; Huan Wang
>
> **摘要:** Autoregressive image generation models like Janus-Pro produce high-quality images, but at the significant cost of high memory and ever-growing computational demands due to the large number of visual tokens. While KV cache compression has been extensively studied in language modeling, it still remains largely unexplored for the image generation domain. In this work, we begin by identifying a distinct and prominent attention phenomenon, which we term spatial locality and emergent semantic sink. To leverage this key insight, we introduce a novel KV cache compression framework. Specifically, we compress the KV cache for all visual tokens by adaptively decoupling attention heads into two separate types: for spatial-locality heads, our method maintains a short recent token window; for semantic-sink heads, it strategically preserves a compact set of highly-attended tokens. Our extensive experiments demonstrate that the proposed method achieves a 5$\times$ reduction in memory usage and a notable 6.6$\times$ speedup in overall throughput with only minimal visual quality loss, thereby enabling highly efficient native autoregressive image generation on resource-constrained hardware.
>
---
#### [new 066] Provenance of AI-Generated Images: A Vector Similarity and Blockchain-based Approach
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文针对AI生成图像难以溯源的问题，提出一种基于图像嵌入向量相似性和区块链的检测方法。通过分析AI与真实图像在嵌入空间的聚类差异，构建高效、鲁棒的生成图像鉴别框架。**

- **链接: [http://arxiv.org/pdf/2510.17854v1](http://arxiv.org/pdf/2510.17854v1)**

> **作者:** Jitendra Sharma; Arthur Carvalho; Suman Bhunia
>
> **摘要:** Rapid advancement in generative AI and large language models (LLMs) has enabled the generation of highly realistic and contextually relevant digital content. LLMs such as ChatGPT with DALL-E integration and Stable Diffusion techniques can produce images that are often indistinguishable from those created by humans, which poses challenges for digital content authentication. Verifying the integrity and origin of digital data to ensure it remains unaltered and genuine is crucial to maintaining trust and legality in digital media. In this paper, we propose an embedding-based AI image detection framework that utilizes image embeddings and a vector similarity to distinguish AI-generated images from real (human-created) ones. Our methodology is built on the hypothesis that AI-generated images demonstrate closer embedding proximity to other AI-generated content, while human-created images cluster similarly within their domain. To validate this hypothesis, we developed a system that processes a diverse dataset of AI and human-generated images through five benchmark embedding models. Extensive experimentation demonstrates the robustness of our approach, and our results confirm that moderate to high perturbations minimally impact the embedding signatures, with perturbed images maintaining close similarity matches to their original versions. Our solution provides a generalizable framework for AI-generated image detection that balances accuracy with computational efficiency.
>
---
#### [new 067] TreeFedDG: Alleviating Global Drift in Federated Domain Generalization for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文研究联邦域泛化下的医学图像分割，旨在缓解跨域场景中因信息聚合不平衡导致的全局漂移问题。提出TreeFedDG框架，通过树形拓扑聚合、参数差异驱动的风格混合与个性化融合策略，提升模型泛化性。**

- **链接: [http://arxiv.org/pdf/2510.18268v1](http://arxiv.org/pdf/2510.18268v1)**

> **作者:** Yucheng Song; Chenxi Li; Haokang Ding; Zhining Liao; Zhifang Liao
>
> **摘要:** In medical image segmentation tasks, Domain Generalization (DG) under the Federated Learning (FL) framework is crucial for addressing challenges related to privacy protection and data heterogeneity. However, traditional federated learning methods fail to account for the imbalance in information aggregation across clients in cross-domain scenarios, leading to the Global Drift (GD) problem and a consequent decline in model generalization performance. This motivates us to delve deeper and define a new critical issue: global drift in federated domain generalization for medical imaging (FedDG-GD). In this paper, we propose a novel tree topology framework called TreeFedDG. First, starting from the distributed characteristics of medical images, we design a hierarchical parameter aggregation method based on a tree-structured topology to suppress deviations in the global model direction. Second, we introduce a parameter difference-based style mixing method (FedStyle), which enforces mixing among clients with maximum parameter differences to enhance robustness against drift. Third, we develop a a progressive personalized fusion strategy during model distribution, ensuring a balance between knowledge transfer and personalized features. Finally, during the inference phase, we use feature similarity to guide the retrieval of the most relevant model chain from the tree structure for ensemble decision-making, thereby fully leveraging the advantages of hierarchical knowledge. We conducted extensive experiments on two publicly available datasets. The results demonstrate that our method outperforms other state-of-the-art domain generalization approaches in these challenging tasks and achieves better balance in cross-domain performance.
>
---
#### [new 068] OmniNWM: Omniscient Driving Navigation World Models
- **分类: cs.CV**

- **简介: 该论文提出OmniNWM，面向自动驾驶世界模型任务，解决现有模型在状态、动作、奖励建模上的局限。工作涵盖全景多模态生成、精确轨迹控制与基于3D占用的规则奖励，实现长时程稳定生成与闭环评估。**

- **链接: [http://arxiv.org/pdf/2510.18313v1](http://arxiv.org/pdf/2510.18313v1)**

> **作者:** Bohan Li; Zhuang Ma; Dalong Du; Baorui Peng; Zhujin Liang; Zhenqiang Liu; Chao Ma; Yueming Jin; Hao Zhao; Wenjun Zeng; Xin Jin
>
> **备注:** https://arlo0o.github.io/OmniNWM/
>
> **摘要:** Autonomous driving world models are expected to work effectively across three core dimensions: state, action, and reward. Existing models, however, are typically restricted to limited state modalities, short video sequences, imprecise action control, and a lack of reward awareness. In this paper, we introduce OmniNWM, an omniscient panoramic navigation world model that addresses all three dimensions within a unified framework. For state, OmniNWM jointly generates panoramic videos of RGB, semantics, metric depth, and 3D occupancy. A flexible forcing strategy enables high-quality long-horizon auto-regressive generation. For action, we introduce a normalized panoramic Plucker ray-map representation that encodes input trajectories into pixel-level signals, enabling highly precise and generalizable control over panoramic video generation. Regarding reward, we move beyond learning reward functions with external image-based models: instead, we leverage the generated 3D occupancy to directly define rule-based dense rewards for driving compliance and safety. Extensive experiments demonstrate that OmniNWM achieves state-of-the-art performance in video generation, control accuracy, and long-horizon stability, while providing a reliable closed-loop evaluation framework through occupancy-grounded rewards. Project page is available at https://github.com/Arlo0o/OmniNWM.
>
---
#### [new 069] Beyond the Pipeline: Analyzing Key Factors in End-to-End Deep Learning for Historical Writer Identification
- **分类: cs.CV**

- **简介: 该论文研究历史笔迹识别任务，旨在解决端到端深度学习方法在真实场景下泛化能力差的问题。作者分析了预处理、网络结构和后处理等因素，发现多数配置因特征提取弱、噪声敏感而表现不佳，但提出一种简化设计可达到顶尖性能。**

- **链接: [http://arxiv.org/pdf/2510.18671v1](http://arxiv.org/pdf/2510.18671v1)**

> **作者:** Hanif Rasyidi; Moshiur Farazi
>
> **备注:** Published in The 12th IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2025
>
> **摘要:** This paper investigates various factors that influence the performance of end-to-end deep learning approaches for historical writer identification (HWI), a task that remains challenging due to the diversity of handwriting styles, document degradation, and the limited number of labelled samples per writer. These conditions often make accurate recognition difficult, even for human experts. Traditional HWI methods typically rely on handcrafted image processing and clustering techniques, which tend to perform well on small and carefully curated datasets. In contrast, end-to-end pipelines aim to automate the process by learning features directly from document images. However, our experiments show that many of these models struggle to generalise in more realistic, document-level settings, especially under zero-shot scenarios where writers in the test set are not present in the training data. We explore different combinations of pre-processing methods, backbone architectures, and post-processing strategies, including text segmentation, patch sampling, and feature aggregation. The results suggest that most configurations perform poorly due to weak capture of low-level visual features, inconsistent patch representations, and high sensitivity to content noise. Still, we identify one end-to-end setup that achieves results comparable to the top-performing system, despite using a simpler design. These findings point to key challenges in building robust end-to-end systems and offer insight into design choices that improve performance in historical document writer identification.
>
---
#### [new 070] A Renaissance of Explicit Motion Information Mining from Transformers for Action Recognition
- **分类: cs.CV**

- **简介: 该论文针对动作识别中Transformer对运动建模不足的问题，提出显式运动信息挖掘模块（EMIM），借鉴传统光流中的代价体思想构建具有运动感知的注意力亲和矩阵，统一提升外观与运动建模能力，在多个数据集上取得更好性能。**

- **链接: [http://arxiv.org/pdf/2510.18705v1](http://arxiv.org/pdf/2510.18705v1)**

> **作者:** Peiqin Zhuang; Lei Bai; Yichao Wu; Ding Liang; Luping Zhou; Yali Wang; Wanli Ouyang
>
> **备注:** accepted by Pattern Recognition. We have been always curious to see whether our designs could be beneficial in other scenarios, such as embedding it into the DiT model or 3D-VAE for video generation. If you are interested in it, why not give it a shot?
>
> **摘要:** Recently, action recognition has been dominated by transformer-based methods, thanks to their spatiotemporal contextual aggregation capacities. However, despite the significant progress achieved on scene-related datasets, they do not perform well on motion-sensitive datasets due to the lack of elaborate motion modeling designs. Meanwhile, we observe that the widely-used cost volume in traditional action recognition is highly similar to the affinity matrix defined in self-attention, but equipped with powerful motion modeling capacities. In light of this, we propose to integrate those effective motion modeling properties into the existing transformer in a unified and neat way, with the proposal of the Explicit Motion Information Mining module (EMIM). In EMIM, we propose to construct the desirable affinity matrix in a cost volume style, where the set of key candidate tokens is sampled from the query-based neighboring area in the next frame in a sliding-window manner. Then, the constructed affinity matrix is used to aggregate contextual information for appearance modeling and is converted into motion features for motion modeling as well. We validate the motion modeling capacities of our method on four widely-used datasets, and our method performs better than existing state-of-the-art approaches, especially on motion-sensitive datasets, i.e., Something-Something V1 & V2.
>
---
#### [new 071] MAT-Agent: Adaptive Multi-Agent Training Optimization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对多标签图像分类中静态训练策略适应性差的问题，提出MAT-Agent框架，通过多智能体协同动态优化数据增强、优化器、学习率和损失函数，结合奖励机制与平滑技术，提升模型性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.17845v1](http://arxiv.org/pdf/2510.17845v1)**

> **作者:** Jusheng Zhang; Kaitong Cai; Yijia Fan; Ningyuan Liu; Keze Wang
>
> **备注:** Acceptance to NeurIPS 2025 Main Track
>
> **摘要:** Multi-label image classification demands adaptive training strategies to navigate complex, evolving visual-semantic landscapes, yet conventional methods rely on static configurations that falter in dynamic settings. We propose MAT-Agent, a novel multi-agent framework that reimagines training as a collaborative, real-time optimization process. By deploying autonomous agents to dynamically tune data augmentation, optimizers, learning rates, and loss functions, MAT-Agent leverages non-stationary multi-armed bandit algorithms to balance exploration and exploitation, guided by a composite reward harmonizing accuracy, rare-class performance, and training stability. Enhanced with dual-rate exponential moving average smoothing and mixed-precision training, it ensures robustness and efficiency. Extensive experiments across Pascal VOC, COCO, and VG-256 demonstrate MAT-Agent's superiority: it achieves an mAP of 97.4 (vs. 96.2 for PAT-T), OF1 of 92.3, and CF1 of 91.4 on Pascal VOC; an mAP of 92.8 (vs. 92.0 for HSQ-CvN), OF1 of 88.2, and CF1 of 87.1 on COCO; and an mAP of 60.9, OF1 of 70.8, and CF1 of 61.1 on VG-256. With accelerated convergence and robust cross-domain generalization, MAT-Agent offers a scalable, intelligent solution for optimizing complex visual models, paving the way for adaptive deep learning advancements.
>
---
#### [new 072] OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion
- **分类: cs.CV**

- **简介: 该论文提出OpenInsGaussian，解决3D实例分割中上下文信息不足和多视角融合不一致的问题。通过上下文感知特征提取与注意力驱动融合，在开放词汇3D高斯分割任务中实现最优性能。**

- **链接: [http://arxiv.org/pdf/2510.18253v1](http://arxiv.org/pdf/2510.18253v1)**

> **作者:** Tianyu Huang; Runnan Chen; Dongting Hu; Fengming Huang; Mingming Gong; Tongliang Liu
>
> **摘要:** Understanding 3D scenes is pivotal for autonomous driving, robotics, and augmented reality. Recent semantic Gaussian Splatting approaches leverage large-scale 2D vision models to project 2D semantic features onto 3D scenes. However, they suffer from two major limitations: (1) insufficient contextual cues for individual masks during preprocessing and (2) inconsistencies and missing details when fusing multi-view features from these 2D models. In this paper, we introduce \textbf{OpenInsGaussian}, an \textbf{Open}-vocabulary \textbf{Ins}tance \textbf{Gaussian} segmentation framework with Context-aware Cross-view Fusion. Our method consists of two modules: Context-Aware Feature Extraction, which augments each mask with rich semantic context, and Attention-Driven Feature Aggregation, which selectively fuses multi-view features to mitigate alignment errors and incompleteness. Through extensive experiments on benchmark datasets, OpenInsGaussian achieves state-of-the-art results in open-vocabulary 3D Gaussian segmentation, outperforming existing baselines by a large margin. These findings underscore the robustness and generality of our proposed approach, marking a significant step forward in 3D scene understanding and its practical deployment across diverse real-world scenarios.
>
---
#### [new 073] SAM 2++: Tracking Anything at Any Granularity
- **分类: cs.CV**

- **简介: 该论文聚焦视频目标跟踪任务，旨在解决现有方法因任务单一、模块定制导致泛化性差的问题。提出SAM 2++，通过统一模型支持任意粒度（掩码、框、点）跟踪，设计任务自适应记忆机制与多粒度数据引擎，实现跨粒度统一跟踪框架。**

- **链接: [http://arxiv.org/pdf/2510.18822v1](http://arxiv.org/pdf/2510.18822v1)**

> **作者:** Jiaming Zhang; Cheng Liang; Yichun Yang; Chenkai Zeng; Yutao Cui; Xinwen Zhang; Xin Zhou; Kai Ma; Gangshan Wu; Limin Wang
>
> **备注:** 8 pages, and 10 pages in Supplementary Material
>
> **摘要:** Video tracking aims at finding the specific target in subsequent frames given its initial state. Due to the varying granularity of target states across different tasks, most existing trackers are tailored to a single task and heavily rely on custom-designed modules within the individual task, which limits their generalization and leads to redundancy in both model design and parameters. To unify video tracking tasks, we present SAM 2++, a unified model towards tracking at any granularity, including masks, boxes, and points. First, to extend target granularity, we design task-specific prompts to encode various task inputs into general prompt embeddings, and a unified decoder to unify diverse task results into a unified form pre-output. Next, to satisfy memory matching, the core operation of tracking, we introduce a task-adaptive memory mechanism that unifies memory across different granularities. Finally, we introduce a customized data engine to support tracking training at any granularity, producing a large and diverse video tracking dataset with rich annotations at three granularities, termed Tracking-Any-Granularity, which represents a comprehensive resource for training and benchmarking on unified tracking. Comprehensive experiments on multiple benchmarks confirm that SAM 2++ sets a new state of the art across diverse tracking tasks at different granularities, establishing a unified and robust tracking framework.
>
---
#### [new 074] TriggerNet: A Novel Explainable AI Framework for Red Palm Mite Detection and Multi-Model Comparison and Heuristic-Guided Annotation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文提出TriggerNet，一种可解释AI框架，用于红棕象甲螨害检测。通过多模型比较与启发式标注，实现11种植物的病害分类（健康、黄斑、红褐变、丝网），提升检测准确性与标注效率。**

- **链接: [http://arxiv.org/pdf/2510.18038v1](http://arxiv.org/pdf/2510.18038v1)**

> **作者:** Harshini Suresha; Kavitha SH
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** The red palm mite infestation has become a serious concern, particularly in regions with extensive palm cultivation, leading to reduced productivity and economic losses. Accurate and early identification of mite-infested plants is critical for effective management. The current study focuses on evaluating and comparing the ML model for classifying the affected plants and detecting the infestation. TriggerNet is a novel interpretable AI framework that integrates Grad-CAM, RISE, FullGrad, and TCAV to generate novel visual explanations for deep learning models in plant classification and disease detection. This study applies TriggerNet to address red palm mite (Raoiella indica) infestation, a major threat to palm cultivation and agricultural productivity. A diverse set of RGB images across 11 plant species, Arecanut, Date Palm, Bird of Paradise, Coconut Palm, Ginger, Citrus Tree, Palm Oil, Orchid, Banana Palm, Avocado Tree, and Cast Iron Plant was utilized for training and evaluation. Advanced deep learning models like CNN, EfficientNet, MobileNet, ViT, ResNet50, and InceptionV3, alongside machine learning classifiers such as Random Forest, SVM, and KNN, were employed for plant classification. For disease classification, all plants were categorized into four classes: Healthy, Yellow Spots, Reddish Bronzing, and Silk Webbing. Snorkel was used to efficiently label these disease classes by leveraging heuristic rules and patterns, reducing manual annotation time and improving dataset reliability.
>
---
#### [new 075] DP$^2$O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究真实世界图像超分辨率（Real-ISR），利用预训练扩散模型生成细节，但存在感知质量不稳定问题。提出DP²O-SR框架，通过混合全参考与无参考图像质量评估构建奖励信号，优化感知偏好，提升生成质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.18851v1](http://arxiv.org/pdf/2510.18851v1)**

> **作者:** Rongyuan Wu; Lingchen Sun; Zhengqiang Zhang; Shihao Wang; Tianhe Wu; Qiaosi Yi; Shuai Li; Lei Zhang
>
> **备注:** Accept by NeurIPS 2025
>
> **摘要:** Benefiting from pre-trained text-to-image (T2I) diffusion models, real-world image super-resolution (Real-ISR) methods can synthesize rich and realistic details. However, due to the inherent stochasticity of T2I models, different noise inputs often lead to outputs with varying perceptual quality. Although this randomness is sometimes seen as a limitation, it also introduces a wider perceptual quality range, which can be exploited to improve Real-ISR performance. To this end, we introduce Direct Perceptual Preference Optimization for Real-ISR (DP$^2$O-SR), a framework that aligns generative models with perceptual preferences without requiring costly human annotations. We construct a hybrid reward signal by combining full-reference and no-reference image quality assessment (IQA) models trained on large-scale human preference datasets. This reward encourages both structural fidelity and natural appearance. To better utilize perceptual diversity, we move beyond the standard best-vs-worst selection and construct multiple preference pairs from outputs of the same model. Our analysis reveals that the optimal selection ratio depends on model capacity: smaller models benefit from broader coverage, while larger models respond better to stronger contrast in supervision. Furthermore, we propose hierarchical preference optimization, which adaptively weights training pairs based on intra-group reward gaps and inter-group diversity, enabling more efficient and stable learning. Extensive experiments across both diffusion- and flow-based T2I backbones demonstrate that DP$^2$O-SR significantly improves perceptual quality and generalizes well to real-world benchmarks.
>
---
#### [new 076] Automated Wicket-Taking Delivery Segmentation and Weakness Detection in Cricket Videos Using OCR-Guided YOLOv8 and Trajectory Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于OCR引导YOLOv8与轨迹建模的自动系统，用于板球视频中击倒瞬间检测、球体定位及轨迹分析，解决击球弱点识别问题，助力自动化比赛分析与战术决策。**

- **链接: [http://arxiv.org/pdf/2510.18405v1](http://arxiv.org/pdf/2510.18405v1)**

> **作者:** Mst Jannatun Ferdous; Masum Billah; Joy Karmoker; Mohd Ruhul Ameen; Akif Islam; Md. Omar Faruqe
>
> **备注:** 6 figures, 5 tables, submitted to the 11th IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering 2025
>
> **摘要:** This paper presents an automated system for cricket video analysis that leverages deep learning techniques to extract wicket-taking deliveries, detect cricket balls, and model ball trajectories. The system employs the YOLOv8 architecture for pitch and ball detection, combined with optical character recognition (OCR) for scorecard extraction to identify wicket-taking moments. Through comprehensive image preprocessing, including grayscale transformation, power transformation, and morphological operations, the system achieves robust text extraction from video frames. The pitch detection model achieved 99.5% mean Average Precision at 50% IoU (mAP50) with a precision of 0.999, while the ball detection model using transfer learning attained 99.18% mAP50 with 0.968 precision and 0.978 recall. The system enables trajectory modeling on detected pitches, providing data-driven insights for identifying batting weaknesses. Experimental results on multiple cricket match videos demonstrate the effectiveness of this approach for automated cricket analytics, offering significant potential for coaching and strategic decision-making.
>
---
#### [new 077] Online In-Context Distillation for Low-Resource Vision Language Models
- **分类: cs.CV**

- **简介: 该论文研究低资源视觉-语言模型的适应问题，提出在线上下文蒸馏（ICD）方法。小模型在推理时通过稀疏示例从大模型中蒸馏知识，结合跨模态示例选择与不确定性感知策略，以极少标注显著提升性能，优于微调，在计算受限下逼近教师模型零样本表现。**

- **链接: [http://arxiv.org/pdf/2510.18117v1](http://arxiv.org/pdf/2510.18117v1)**

> **作者:** Zhiqi Kang; Rahaf Aljundi; Vaggelis Dorovatas; Karteek Alahari
>
> **摘要:** As the field continues its push for ever more resources, this work turns the spotlight on a critical question: how can vision-language models (VLMs) be adapted to thrive in low-resource, budget-constrained settings? While large VLMs offer strong performance, they are impractical to deploy in such settings. Small VLMs, on the other hand, are efficient but typically require costly fine-tuning to close the performance gap with larger models in the deployment domain. Inspired by the in-context learning framework, we propose an online In-Context Distillation (ICD) method, in which a small VLM collaborates with a stronger teacher model at inference time, distilling its knowledge via sparse demonstrations to efficiently bridge the gap between them. Our method is built on an in-depth analysis that identifies the scale and the choice of models for which vision-language ICL is currently feasible, and demonstrates the advantage of ICL over fine-tuning under constrained compute budgets. We enhance our method with a novel cross-modal demonstration selection strategy, teacher test-time scaling to reduce noise, and student uncertainty conditioning to dynamically populate a demonstration pool and minimize teacher queries. Our ICD method significantly boosts the performance of small models (up to 33%) using scarce teacher annotations (as low as 4%), and competes with the teacher's zero-shot performance.
>
---
#### [new 078] ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言对齐任务，旨在解决CLIP文本编码器在长文本、多语言和细粒度语义理解上的局限。作者提出ProCLIP，通过课程学习逐步对齐基于LLM的嵌入器与CLIP图像编码器，利用知识蒸馏和对比调优实现更有效的跨模态对齐。**

- **链接: [http://arxiv.org/pdf/2510.18795v1](http://arxiv.org/pdf/2510.18795v1)**

> **作者:** Xiaoxing Hu; Kaicheng Yang; Ziyong Feng; Qi Ming; Zonghao Guo; Xiang An; Ziyong Feng; Junchi Yan; Xue Yang
>
> **备注:** 17 pages, 5 fiugres
>
> **摘要:** The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP
>
---
#### [new 079] DSI-Bench: A Benchmark for Dynamic Spatial Intelligence
- **分类: cs.CV**

- **简介: 该论文提出DSI-Bench，旨在评估模型对动态3D空间关系的理解能力。针对现有视觉语言模型在动态场景中混淆观察者与物体运动、存在语义偏见等问题，构建了包含近千视频和千余问题的基准，系统评测并揭示了模型在动态空间推理上的局限性。**

- **链接: [http://arxiv.org/pdf/2510.18873v1](http://arxiv.org/pdf/2510.18873v1)**

> **作者:** Ziang Zhang; Zehan Wang; Guanghao Zhang; Weilong Dai; Yan Xia; Ziang Yan; Minjie Hong; Zhou Zhao
>
> **摘要:** Reasoning about dynamic spatial relationships is essential, as both observers and objects often move simultaneously. Although vision-language models (VLMs) and visual expertise models excel in 2D tasks and static scenarios, their ability to fully understand dynamic 3D scenarios remains limited. We introduce Dynamic Spatial Intelligence and propose DSI-Bench, a benchmark with nearly 1,000 dynamic videos and over 1,700 manually annotated questions covering nine decoupled motion patterns of observers and objects. Spatially and temporally symmetric designs reduce biases and enable systematic evaluation of models' reasoning about self-motion and object motion. Our evaluation of 14 VLMs and expert models reveals key limitations: models often conflate observer and object motion, exhibit semantic biases, and fail to accurately infer relative relationships in dynamic scenarios. Our DSI-Bench provides valuable findings and insights about the future development of general and expertise models with dynamic spatial intelligence.
>
---
#### [new 080] Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views
- **分类: cs.CV; cs.AI; I.2.10**

- **简介: 该论文研究视觉-语言模型在有限视角下进行3D空间推理的任务，旨在解决现有方法因缺乏3D表征能力而导致的性能局限。作者提出3DThinker框架，首次实现无需3D先验输入和标注数据的3D心智模拟推理，通过两阶段训练融合几何信息，提升多模态空间理解能力。**

- **链接: [http://arxiv.org/pdf/2510.18632v1](http://arxiv.org/pdf/2510.18632v1)**

> **作者:** Zhangquan Chen; Manyuan Zhang; Xinlei Yu; Xufang Luo; Mingze Sun; Zihao Pan; Yan Feng; Peng Pei; Xunliang Cai; Ruqi Huang
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Though recent advances in vision-language models (VLMs) have achieved remarkable progress across a wide range of multimodal tasks, understanding 3D spatial relationships from limited views remains a significant challenge. Previous reasoning methods typically rely on pure text (e.g., topological cognitive maps) or on 2D visual cues. However, their limited representational capacity hinders performance in specific tasks that require 3D spatial imagination. To address this limitation, we propose 3DThinker, a framework that can effectively exploits the rich geometric information embedded within images while reasoning, like humans do. Our framework is the first to enable 3D mentaling during reasoning without any 3D prior input, and it does not rely on explicitly labeled 3D data for training. Specifically, our training consists of two stages. First, we perform supervised training to align the 3D latent generated by VLM while reasoning with that of a 3D foundation model (e.g., VGGT). Then, we optimize the entire reasoning trajectory solely based on outcome signals, thereby refining the underlying 3D mentaling. Extensive experiments across multiple benchmarks show that 3DThinker consistently outperforms strong baselines and offers a new perspective toward unifying 3D representations into multimodal reasoning. Our code will be available at https://github.com/zhangquanchen/3DThinker.
>
---
#### [new 081] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization
- **分类: cs.CV**

- **简介: 该论文针对视觉指令微调中数据训练成本高的问题，提出CoIDO框架，通过联合优化数据重要性和多样性，利用少量样本训练轻量评分器，实现高效数据选择，在仅用20%数据时达到全量数据98.2%的性能。**

- **链接: [http://arxiv.org/pdf/2510.17847v1](http://arxiv.org/pdf/2510.17847v1)**

> **作者:** Yichen Yan; Ming Zhong; Qi Zhu; Xiaoling Gu; Jinpeng Chen; Huan Li
>
> **备注:** 22 pages, 8 figures, 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Multimodal large language models (MLLMs) rely heavily on instruction tuning to align vision and language capabilities, yet the computational cost of training on large-scale datasets remains a major bottleneck. Existing data selection methods aim to mitigate this by selecting important and diverse subsets, but they often suffer from two critical drawbacks: high computational overhead from processing the entire dataset and suboptimal data selection due to separate treatment of importance and diversity. We introduce CoIDO, a novel dual-objective framework that jointly optimizes data importance and diversity to overcome these challenges. Unlike existing approaches that require costly evaluations across the whole dataset, CoIDO employs a lightweight plug-in scorer. This scorer is trained on just a small random sample of data to learn the distribution of the candidate set, drastically reducing computational demands. By leveraging a homoscedastic uncertainty-based formulation, CoIDO effectively balances importance and diversity during training, enabling efficient and scalable data selection. In our experiments, we trained the CoIDO scorer using only 20 percent of randomly sampled data. Once trained, CoIDO was applied to the entire dataset to select a 20 percent subset for instruction tuning. On the widely used LLaVA-1.5-7B model across ten downstream tasks, this selected subset achieved an impressive 98.2 percent of the performance of full-data fine-tuning, on average.
>
---
#### [new 082] VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频异常检测任务，旨在解决密集人群中的运动异常检测问题。作者提出VelocityNet框架，结合头部检测与光流提取个体速度，通过聚类和百分位评分实现可解释的实时异常检测。**

- **链接: [http://arxiv.org/pdf/2510.18187v1](http://arxiv.org/pdf/2510.18187v1)**

> **作者:** Fatima AlGhamdi; Omar Alharbi; Abdullah Aldwyish; Raied Aljadaany; Muhammad Kamran J Khan; Huda Alamri
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Detecting anomalies in crowded scenes is challenging due to severe inter-person occlusions and highly dynamic, context-dependent motion patterns. Existing approaches often struggle to adapt to varying crowd densities and lack interpretable anomaly indicators. To address these limitations, we introduce VelocityNet, a dual-pipeline framework that combines head detection and dense optical flow to extract person-specific velocities. Hierarchical clustering categorizes these velocities into semantic motion classes (halt, slow, normal, and fast), and a percentile-based anomaly scoring system measures deviations from learned normal patterns. Experiments demonstrate the effectiveness of our framework in real-time detection of diverse anomalous motion patterns within densely crowded environments.
>
---
#### [new 083] LAND: Lung and Nodule Diffusion for 3D Chest CT Synthesis with Anatomical Guidance
- **分类: cs.CV**

- **简介: 该论文提出一种基于肺部和结节扩散的3D胸部CT合成方法，属医学图像生成任务。为解决条件生成中解剖结构不准的问题，引入解剖掩码引导，实现高质量、低计算成本的CT图像生成，支持带或不带结节的多样化样本生成，用于AI训练与临床辅助。**

- **链接: [http://arxiv.org/pdf/2510.18446v1](http://arxiv.org/pdf/2510.18446v1)**

> **作者:** Anna Oliveras; Roger Marí; Rafael Redondo; Oriol Guardià; Ana Tost; Bhalaji Nagarajan; Carolina Migliorelli; Vicent Ribas; Petia Radeva
>
> **摘要:** This work introduces a new latent diffusion model to generate high-quality 3D chest CT scans conditioned on 3D anatomical masks. The method synthesizes volumetric images of size 256x256x256 at 1 mm isotropic resolution using a single mid-range GPU, significantly lowering the computational cost compared to existing approaches. The conditioning masks delineate lung and nodule regions, enabling precise control over the output anatomical features. Experimental results demonstrate that conditioning solely on nodule masks leads to anatomically incorrect outputs, highlighting the importance of incorporating global lung structure for accurate conditional synthesis. The proposed approach supports the generation of diverse CT volumes with and without lung nodules of varying attributes, providing a valuable tool for training AI models or healthcare professionals.
>
---
#### [new 084] Entropy-Enhanced Conformal Features from Ricci Flow for Robust Alzheimer's Disease Classification
- **分类: cs.CV**

- **简介: 该论文针对阿尔茨海默病分类任务，提出基于Ricci流的共形几何特征与熵增强方法，提取皮层表面形态学信息。利用ADNI数据集MRI构建 cortical 网格，计算共形因子、面积畸变和高斯曲率的熵特征，结合多种分类器实现98.62%准确率，提升AD自动诊断性能。**

- **链接: [http://arxiv.org/pdf/2510.18396v1](http://arxiv.org/pdf/2510.18396v1)**

> **作者:** F. Ahmadi; B. Bidabad; H. Nasiri
>
> **摘要:** Background and Objective: In brain imaging, geometric surface models are essential for analyzing the 3D shapes of anatomical structures. Alzheimer's disease (AD) is associated with significant cortical atrophy, making such shape analysis a valuable diagnostic tool. The objective of this study is to introduce and validate a novel local surface representation method for the automated and accurate diagnosis of AD. Methods: The study utilizes T1-weighted MRI scans from 160 participants (80 AD patients and 80 healthy controls) from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Cortical surface models were reconstructed from the MRI data using Freesurfer. Key geometric attributes were computed from the 3D meshes. Area distortion and conformal factor were derived using Ricci flow for conformal parameterization, while Gaussian curvature was calculated directly from the mesh geometry. Shannon entropy was applied to these three features to create compact and informative feature vectors. The feature vectors were used to train and evaluate a suite of classifiers (e.g. XGBoost, MLP, Logistic Regression, etc.). Results: Statistical significance of performance differences between classifiers was evaluated using paired Welch's t-test. The method proved highly effective in distinguishing AD patients from healthy controls. The Multi-Layer Perceptron (MLP) and Logistic Regression classifiers outperformed all others, achieving an accuracy and F$_1$ Score of 98.62%. Conclusions: This study confirms that the entropy of conformally-derived geometric features provides a powerful and robust metric for cortical morphometry. The high classification accuracy underscores the method's potential to enhance the study and diagnosis of Alzheimer's disease, offering a straightforward yet powerful tool for clinical research applications.
>
---
#### [new 085] Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究车辆品牌型号识别（VMMR）任务，旨在解决现有方法难以适应新车型的问题。提出结合视觉语言模型与检索增强生成（RAG）的零样本识别方法，通过图像转文本描述、检索匹配并生成推理结果，避免重训练，实现高效更新，准确率较CLIP提升近20%。**

- **链接: [http://arxiv.org/pdf/2510.18502v1](http://arxiv.org/pdf/2510.18502v1)**

> **作者:** Wei-Chia Chang; Yan-Ann Chen
>
> **备注:** Accepted by The 38th Conference of Open Innovations Association FRUCT, 2025
>
> **摘要:** Vehicle make and model recognition (VMMR) is an important task in intelligent transportation systems, but existing approaches struggle to adapt to newly released models. Contrastive Language-Image Pretraining (CLIP) provides strong visual-text alignment, yet its fixed pretrained weights limit performance without costly image-specific finetuning. We propose a pipeline that integrates vision language models (VLMs) with Retrieval-Augmented Generation (RAG) to support zero-shot recognition through text-based reasoning. A VLM converts vehicle images into descriptive attributes, which are compared against a database of textual features. Relevant entries are retrieved and combined with the description to form a prompt, and a language model (LM) infers the make and model. This design avoids large-scale retraining and enables rapid updates by adding textual descriptions of new vehicles. Experiments show that the proposed method improves recognition by nearly 20% over the CLIP baseline, demonstrating the potential of RAG-enhanced LM reasoning for scalable VMMR in smart-city applications.
>
---
#### [new 086] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究多模态大模型的细粒度区域理解任务，旨在解决现有方法忽略全局上下文和多区域交互的问题。提出Grasp Any Region（GAR）框架，结合RoI对齐特征重放，实现精准感知与多提示推理，并构建GAR-Bench评估其在复杂场景中的区域理解和跨区域推理能力。**

- **链接: [http://arxiv.org/pdf/2510.18876v1](http://arxiv.org/pdf/2510.18876v1)**

> **作者:** Haochen Wang; Yuhao Wang; Tao Zhang; Yikang Zhou; Yanwei Li; Jiacong Wang; Ye Tian; Jiahao Meng; Zilong Huang; Guangcan Mai; Anran Wang; Yunhai Tong; Zhuochen Wang; Xiangtai Li; Zhaoxiang Zhang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos.
>
---
#### [new 087] PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting
- **分类: cs.CV**

- **简介: 该论文研究室内场景的度量3D重建，利用平面几何先验，提出PLANA3R框架。其通过无姿态双视图图像，以无需3D平面标注的前馈方式实现平面分割、深度估计与位姿预测，提升跨域泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.18714v1](http://arxiv.org/pdf/2510.18714v1)**

> **作者:** Changkun Liu; Bin Tan; Zeran Ke; Shangzhan Zhang; Jiachen Liu; Ming Qian; Nan Xue; Yujun Shen; Tristan Braud
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025). The project page is available at: https://lck666666.github.io/plana3r
>
> **摘要:** This paper addresses metric 3D reconstruction of indoor scenes by exploiting their inherent geometric regularities with compact representations. Using planar 3D primitives - a well-suited representation for man-made environments - we introduce PLANA3R, a pose-free framework for metric Planar 3D Reconstruction from unposed two-view images. Our approach employs Vision Transformers to extract a set of sparse planar primitives, estimate relative camera poses, and supervise geometry learning via planar splatting, where gradients are propagated through high-resolution rendered depth and normal maps of primitives. Unlike prior feedforward methods that require 3D plane annotations during training, PLANA3R learns planar 3D structures without explicit plane supervision, enabling scalable training on large-scale stereo datasets using only depth and normal annotations. We validate PLANA3R on multiple indoor-scene datasets with metric supervision and demonstrate strong generalization to out-of-domain indoor environments across diverse tasks under metric evaluation protocols, including 3D surface reconstruction, depth estimation, and relative pose estimation. Furthermore, by formulating with planar 3D representation, our method emerges with the ability for accurate plane segmentation. The project page is available at https://lck666666.github.io/plana3r
>
---
#### [new 088] HouseTour: A Virtual Real Estate A(I)gent
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出HouseTour，旨在生成空间感知的3D相机轨迹与自然语言摘要。针对现有视觉语言模型缺乏几何推理能力的问题，结合扩散模型与3D高斯点阵渲染，实现高质量虚拟看房视频自动生成。**

- **链接: [http://arxiv.org/pdf/2510.18054v1](http://arxiv.org/pdf/2510.18054v1)**

> **作者:** Ata Çelen; Marc Pollefeys; Daniel Barath; Iro Armeni
>
> **备注:** Published on ICCV 2025
>
> **摘要:** We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.
>
---
#### [new 089] GeoDiff: Geometry-Guided Diffusion for Metric Depth Estimation
- **分类: cs.CV**

- **简介: 该论文研究单目图像的度量深度估计任务，旨在解决现有扩散模型难以准确恢复绝对尺度的问题。作者提出GeoDiff，利用立体视觉几何约束引导预训练扩散模型，在无需训练的情况下提升深度估计的尺度准确性。**

- **链接: [http://arxiv.org/pdf/2510.18291v1](http://arxiv.org/pdf/2510.18291v1)**

> **作者:** Tuan Pham; Thanh-Tung Le; Xiaohui Xie; Stephan Mandt
>
> **备注:** Accepted to ICCV Findings 2025. The first two authors contributed equally. The last two authors share co-corresponding authorship
>
> **摘要:** We introduce a novel framework for metric depth estimation that enhances pretrained diffusion-based monocular depth estimation (DB-MDE) models with stereo vision guidance. While existing DB-MDE methods excel at predicting relative depth, estimating absolute metric depth remains challenging due to scale ambiguities in single-image scenarios. To address this, we reframe depth estimation as an inverse problem, leveraging pretrained latent diffusion models (LDMs) conditioned on RGB images, combined with stereo-based geometric constraints, to learn scale and shift for accurate depth recovery. Our training-free solution seamlessly integrates into existing DB-MDE frameworks and generalizes across indoor, outdoor, and complex environments. Extensive experiments demonstrate that our approach matches or surpasses state-of-the-art methods, particularly in challenging scenarios involving translucent and specular surfaces, all without requiring retraining.
>
---
#### [new 090] BlendCLIP: Bridging Synthetic and Real Domains for Zero-Shot 3D Object Classification with Multimodal Pretraining
- **分类: cs.CV**

- **简介: 该论文研究零样本3D物体分类，旨在解决合成数据与真实LiDAR扫描间的域差距。提出BlendCLIP框架，通过多模态预训练和课程式数据混合策略，融合合成与真实数据优势，显著提升零样本分类性能。**

- **链接: [http://arxiv.org/pdf/2510.18244v1](http://arxiv.org/pdf/2510.18244v1)**

> **作者:** Ajinkya Khoche; Gergő László Nagy; Maciej Wozniak; Thomas Gustafsson; Patric Jensfelt
>
> **备注:** Under Review
>
> **摘要:** Zero-shot 3D object classification is crucial for real-world applications like autonomous driving, however it is often hindered by a significant domain gap between the synthetic data used for training and the sparse, noisy LiDAR scans encountered in the real-world. Current methods trained solely on synthetic data fail to generalize to outdoor scenes, while those trained only on real data lack the semantic diversity to recognize rare or unseen objects. We introduce BlendCLIP, a multimodal pretraining framework that bridges this synthetic-to-real gap by strategically combining the strengths of both domains. We first propose a pipeline to generate a large-scale dataset of object-level triplets -- consisting of a point cloud, image, and text description -- mined directly from real-world driving data and human annotated 3D boxes. Our core contribution is a curriculum-based data mixing strategy that first grounds the model in the semantically rich synthetic CAD data before progressively adapting it to the specific characteristics of real-world scans. Our experiments show that our approach is highly label-efficient: introducing as few as 1.5\% real-world samples per batch into training boosts zero-shot accuracy on the nuScenes benchmark by 27\%. Consequently, our final model achieves state-of-the-art performance on challenging outdoor datasets like nuScenes and TruckScenes, improving over the best prior method by 19.3\% on nuScenes, while maintaining strong generalization on diverse synthetic benchmarks. Our findings demonstrate that effective domain adaptation, not full-scale real-world annotation, is the key to unlocking robust open-vocabulary 3D perception. Our code and dataset will be released upon acceptance on https://github.com/kesu1/BlendCLIP.
>
---
#### [new 091] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文提出二值二次量化（BQQ），用于实值矩阵压缩。针对传统一阶量化表达能力有限的问题，BQQ利用二值二次表达提升逼近能力，在保持高压缩比的同时降低重构误差，并在矩阵压缩和视觉Transformer后训练量化中取得更好性能。**

- **链接: [http://arxiv.org/pdf/2510.18650v1](http://arxiv.org/pdf/2510.18650v1)**

> **作者:** Kyo Kuroki; Yasuyuki Okoshi; Thiem Van Chu; Kazushi Kawamura; Masato Motomura
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** This paper proposes a novel matrix quantization method, Binary Quadratic Quantization (BQQ). In contrast to conventional first-order quantization approaches, such as uniform quantization and binary coding quantization, that approximate real-valued matrices via linear combinations of binary bases, BQQ leverages the expressive power of binary quadratic expressions while maintaining an extremely compact data format. We validate our approach with two experiments: a matrix compression benchmark and post-training quantization (PTQ) on pretrained Vision Transformer-based models. Experimental results demonstrate that BQQ consistently achieves a superior trade-off between memory efficiency and reconstruction error than conventional methods for compressing diverse matrix data. It also delivers strong PTQ performance, even though we neither target state-of-the-art PTQ accuracy under tight memory constraints nor rely on PTQ-specific binary matrix optimization. For example, our proposed method outperforms the state-of-the-art PTQ method by up to 2.2\% and 59.1% on the ImageNet dataset under the calibration-based and data-free scenarios, respectively, with quantization equivalent to 2 bits. These findings highlight the surprising effectiveness of binary quadratic expressions for efficient matrix approximation and neural network compression.
>
---
#### [new 092] See the Text: From Tokenization to Visual Reading
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SeeTok，将文本转为图像输入多模态大模型，模拟人类视觉阅读，解决传统子词分词在低资源语言中过度分割、计算量大的问题，在减少70.5%计算量的同时提升跨语言与抗噪能力。**

- **链接: [http://arxiv.org/pdf/2510.18840v1](http://arxiv.org/pdf/2510.18840v1)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Hongyu Qu; Zechao Li; Jinhui Tang
>
> **摘要:** People see text. Humans read by recognizing words as visual objects, including their shapes, layouts, and patterns, before connecting them to meaning, which enables us to handle typos, distorted fonts, and various scripts effectively. Modern large language models (LLMs), however, rely on subword tokenization, fragmenting text into pieces from a fixed vocabulary. While effective for high-resource languages, this approach over-segments low-resource languages, yielding long, linguistically meaningless sequences and inflating computation. In this work, we challenge this entrenched paradigm and move toward a vision-centric alternative. Our method, SeeTok, renders text as images (visual-text) and leverages pretrained multimodal LLMs to interpret them, reusing strong OCR and text-vision alignment abilities learned from large-scale multimodal training. Across three different language tasks, SeeTok matches or surpasses subword tokenizers while requiring 4.43 times fewer tokens and reducing FLOPs by 70.5%, with additional gains in cross-lingual generalization, robustness to typographic noise, and linguistic hierarchy. SeeTok signals a shift from symbolic tokenization to human-like visual reading, and takes a step toward more natural and cognitively inspired language models.
>
---
#### [new 093] Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model for Microclimate Impact Prediction
- **分类: cs.CV; 68T07; I.2.6; I.5.4**

- **简介: 该论文属城市气候建模任务，旨在解决数据稀缺地区城市热岛预测不准问题。通过细调地理空间基础模型，结合绿地降温效应验证，实现高精度地表温度预测与缓解策略模拟。**

- **链接: [http://arxiv.org/pdf/2510.18773v1](http://arxiv.org/pdf/2510.18773v1)**

> **作者:** Jannis Fleckenstein; David Kreismann; Tamara Rosemary Govindasamy; Thomas Brunschwiler; Etienne Vos; Mattia Rigotti
>
> **备注:** 10 pages, 9 figures. Accepted at the NeurIPS 2025 Workshop on Tackling Climate Change with Machine Learning
>
> **摘要:** As urbanization and climate change progress, urban heat island effects are becoming more frequent and severe. To formulate effective mitigation plans, cities require detailed air temperature data, yet conventional machine learning models with limited data often produce inaccurate predictions, particularly in underserved areas. Geospatial foundation models trained on global unstructured data offer a promising alternative by demonstrating strong generalization and requiring only minimal fine-tuning. In this study, an empirical ground truth of urban heat patterns is established by quantifying cooling effects from green spaces and benchmarking them against model predictions to evaluate the model's accuracy. The foundation model is subsequently fine-tuned to predict land surface temperatures under future climate scenarios, and its practical value is demonstrated through a simulated inpainting that highlights its role for mitigation support. The results indicate that foundation models offer a powerful way for evaluating urban heat island mitigation strategies in data-scarce regions to support more climate-resilient cities.
>
---
#### [new 094] A Geometric Approach to Steerable Convolutions
- **分类: cs.CV**

- **简介: 该论文研究可 steer 卷积神经网络，旨在提升模型对旋转等变换的等变性。作者提出一种基于几何与模式匹配的直观推导方法，解释了Clebsch-Gordan分解和球谐函数的出现，并设计了基于插值核的新卷积层，增强对噪声的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.18813v1](http://arxiv.org/pdf/2510.18813v1)**

> **作者:** Soumyabrata Kundu; Risi Kondor
>
> **摘要:** In contrast to the somewhat abstract, group theoretical approach adopted by many papers, our work provides a new and more intuitive derivation of steerable convolutional neural networks in $d$ dimensions. This derivation is based on geometric arguments and fundamental principles of pattern matching. We offer an intuitive explanation for the appearance of the Clebsch--Gordan decomposition and spherical harmonic basis functions. Furthermore, we suggest a novel way to construct steerable convolution layers using interpolation kernels that improve upon existing implementation, and offer greater robustness to noisy data.
>
---
#### [new 095] InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation
- **分类: cs.CV**

- **简介: 该论文提出InsideOut，扩展3D高斯点阵以融合RGB表面与X射线内部结构。属多模态3D重建任务，解决RGB与X射线数据对齐难、配对数据少的问题，实现高保真内外一体三维表达。**

- **链接: [http://arxiv.org/pdf/2510.17864v1](http://arxiv.org/pdf/2510.17864v1)**

> **作者:** Jungmin Lee; Seonghyuk Hong; Juyong Lee; Jaeyoon Lee; Jongwon Choi
>
> **备注:** Published at ICCV 2025
>
> **摘要:** We introduce InsideOut, an extension of 3D Gaussian splatting (3DGS) that bridges the gap between high-fidelity RGB surface details and subsurface X-ray structures. The fusion of RGB and X-ray imaging is invaluable in fields such as medical diagnostics, cultural heritage restoration, and manufacturing. We collect new paired RGB and X-ray data, perform hierarchical fitting to align RGB and X-ray radiative Gaussian splats, and propose an X-ray reference loss to ensure consistent internal structures. InsideOut effectively addresses the challenges posed by disparate data representations between the two modalities and limited paired datasets. This approach significantly extends the applicability of 3DGS, enhancing visualization, simulation, and non-destructive testing capabilities across various domains.
>
---
#### [new 096] FeatureFool: Zero-Query Fooling of Video Models via Feature Map
- **分类: cs.CV**

- **简介: 该论文提出FeatureFool，针对视频模型的零查询黑盒攻击方法。利用DNN提取的特征图直接扰动清洁视频的特征空间，无需查询即可高效生成难以察觉的对抗视频，解决现有攻击查询成本高、难迁移的问题，适用于传统视频分类器与Video-LLMs。**

- **链接: [http://arxiv.org/pdf/2510.18362v1](http://arxiv.org/pdf/2510.18362v1)**

> **作者:** Duoxun Tang; Xi Xiao; Guangwu Hu; Kangkang Sun; Xiao Yang; Dongyang Chen; Qing Li; Yongjie Yin; Jiyao Wang
>
> **摘要:** The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.
>
---
#### [new 097] Descriptor: Occluded nuScenes: A Multi-Sensor Dataset for Evaluating Perception Robustness in Automated Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶感知任务，旨在解决现有数据集缺乏可控、可重复多传感器退化的问题。作者提出Occluded nuScenes数据集，扩展nuScenes，提供摄像头、雷达和LiDAR的参数化遮挡，支持对感知模型鲁棒性的系统评估。**

- **链接: [http://arxiv.org/pdf/2510.18552v1](http://arxiv.org/pdf/2510.18552v1)**

> **作者:** Sanjay Kumar; Tim Brophy; Reenu Mohandas; Eoin Martino Grua; Ganesh Sistu; Valentina Donzella; Ciaran Eising
>
> **摘要:** Robust perception in automated driving requires reliable performance under adverse conditions, where sensors may be affected by partial failures or environmental occlusions. Although existing autonomous driving datasets inherently contain sensor noise and environmental variability, very few enable controlled, parameterised, and reproducible degradations across multiple sensing modalities. This gap limits the ability to systematically evaluate how perception and fusion architectures perform under well-defined adverse conditions. To address this limitation, we introduce the Occluded nuScenes Dataset, a novel extension of the widely used nuScenes benchmark. For the camera modality, we release both the full and mini versions with four types of occlusions, two adapted from public implementations and two newly designed. For radar and LiDAR, we provide parameterised occlusion scripts that implement three types of degradations each, enabling flexible and repeatable generation of occluded data. This resource supports consistent, reproducible evaluation of perception models under partial sensor failures and environmental interference. By releasing the first multi-sensor occlusion dataset with controlled and reproducible degradations, we aim to advance research on robust sensor fusion, resilience analysis, and safety-critical perception in automated driving.
>
---
#### [new 098] ScaleNet: Scaling up Pretrained Neural Networks with Incremental Parameters
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ScaleNet，旨在高效扩展预训练视觉Transformer（ViT）。通过插入共享权重的层并引入少量可调参数，在几乎不增加参数量的情况下实现模型快速扩容，显著提升性能并减少训练成本。**

- **链接: [http://arxiv.org/pdf/2510.18431v1](http://arxiv.org/pdf/2510.18431v1)**

> **作者:** Zhiwei Hao; Jianyuan Guo; Li Shen; Kai Han; Yehui Tang; Han Hu; Yunhe Wang
>
> **摘要:** Recent advancements in vision transformers (ViTs) have demonstrated that larger models often achieve superior performance. However, training these models remains computationally intensive and costly. To address this challenge, we introduce ScaleNet, an efficient approach for scaling ViT models. Unlike conventional training from scratch, ScaleNet facilitates rapid model expansion with negligible increases in parameters, building on existing pretrained models. This offers a cost-effective solution for scaling up ViTs. Specifically, ScaleNet achieves model expansion by inserting additional layers into pretrained ViTs, utilizing layer-wise weight sharing to maintain parameters efficiency. Each added layer shares its parameter tensor with a corresponding layer from the pretrained model. To mitigate potential performance degradation due to shared weights, ScaleNet introduces a small set of adjustment parameters for each layer. These adjustment parameters are implemented through parallel adapter modules, ensuring that each instance of the shared parameter tensor remains distinct and optimized for its specific function. Experiments on the ImageNet-1K dataset demonstrate that ScaleNet enables efficient expansion of ViT models. With a 2$\times$ depth-scaled DeiT-Base model, ScaleNet achieves a 7.42% accuracy improvement over training from scratch while requiring only one-third of the training epochs, highlighting its efficiency in scaling ViTs. Beyond image classification, our method shows significant potential for application in downstream vision areas, as evidenced by the validation in object detection task.
>
---
#### [new 099] Conformal Lesion Segmentation for 3D Medical Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究3D医学图像病灶分割，旨在解决现有方法缺乏对假阴性率（FNR）统计控制的问题。作者提出Conformal Lesion Segmentation（CLS）框架，通过共形预测校准数据驱动阈值，确保测试时FNR低于预设容忍度，实现可靠的风险可控分割。**

- **链接: [http://arxiv.org/pdf/2510.17897v1](http://arxiv.org/pdf/2510.17897v1)**

> **作者:** Binyu Tan; Zhiyuan Wang; Jinhao Duan; Kaidi Xu; Heng Tao Shen; Xiaoshuang Shi; Fumin Shen
>
> **摘要:** Medical image segmentation serves as a critical component of precision medicine, enabling accurate localization and delineation of pathological regions, such as lesions. However, existing models empirically apply fixed thresholds (e.g., 0.5) to differentiate lesions from the background, offering no statistical guarantees on key metrics such as the false negative rate (FNR). This lack of principled risk control undermines their reliable deployment in high-stakes clinical applications, especially in challenging scenarios like 3D lesion segmentation (3D-LS). To address this issue, we propose a risk-constrained framework, termed Conformal Lesion Segmentation (CLS), that calibrates data-driven thresholds via conformalization to ensure the test-time FNR remains below a target tolerance $\varepsilon$ under desired risk levels. CLS begins by holding out a calibration set to analyze the threshold setting for each sample under the FNR tolerance, drawing on the idea of conformal prediction. We define an FNR-specific loss function and identify the critical threshold at which each calibration data point just satisfies the target tolerance. Given a user-specified risk level $\alpha$, we then determine the approximate $1-\alpha$ quantile of all the critical thresholds in the calibration set as the test-time confidence threshold. By conformalizing such critical thresholds, CLS generalizes the statistical regularities observed in the calibration set to new test data, providing rigorous FNR constraint while yielding more precise and reliable segmentations. We validate the statistical soundness and predictive performance of CLS on six 3D-LS datasets across five backbone models, and conclude with actionable insights for deploying risk-aware segmentation in clinical practice.
>
---
#### [new 100] From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation
- **分类: cs.LG; cs.CV; cs.GR**

- **简介: 该论文研究主体驱动图像生成，解决身份保持与提示遵循的权衡问题。提出Customized-GRPO框架，通过协同感知奖励塑造和时序动态加权，缓解强化学习中的竞争退化，提升生成质量与编辑性。**

- **链接: [http://arxiv.org/pdf/2510.18263v1](http://arxiv.org/pdf/2510.18263v1)**

> **作者:** Ziwei Huang; Ying Shu; Hao Fang; Quanyu Long; Wenya Wang; Qiushi Guo; Tiezheng Ge; Leilei Gan
>
> **摘要:** Subject-driven image generation models face a fundamental trade-off between identity preservation (fidelity) and prompt adherence (editability). While online reinforcement learning (RL), specifically GPRO, offers a promising solution, we find that a naive application of GRPO leads to competitive degradation, as the simple linear aggregation of rewards with static weights causes conflicting gradient signals and a misalignment with the temporal dynamics of the diffusion process. To overcome these limitations, we propose Customized-GRPO, a novel framework featuring two key innovations: (i) Synergy-Aware Reward Shaping (SARS), a non-linear mechanism that explicitly penalizes conflicted reward signals and amplifies synergistic ones, providing a sharper and more decisive gradient. (ii) Time-Aware Dynamic Weighting (TDW), which aligns the optimization pressure with the model's temporal dynamics by prioritizing prompt-following in the early, identity preservation in the later. Extensive experiments demonstrate that our method significantly outperforms naive GRPO baselines, successfully mitigating competitive degradation. Our model achieves a superior balance, generating images that both preserve key identity features and accurately adhere to complex textual prompts.
>
---
#### [new 101] DMTrack: Deformable State-Space Modeling for UAV Multi-Object Tracking with Kalman Fusion and Uncertainty-Aware Association
- **分类: eess.SY; cs.CV; cs.SY**

- **简介: 该论文针对无人机多目标跟踪中运动复杂、遮挡频繁等问题，提出DMTrack框架，通过可变形状态空间建模、Kalman融合与不确定性感知关联，提升轨迹预测与身份保持能力，无需外观模型即可实现高效精准跟踪。**

- **链接: [http://arxiv.org/pdf/2510.17860v1](http://arxiv.org/pdf/2510.17860v1)**

> **作者:** Zenghuang Fu; Xiaofeng Han; Mingda Jia; Jin ming Yang; Qi Zeng; Muyang Zahng; Changwei Wang; Weiliang Meng; Xiaopeng Zhang
>
> **摘要:** Multi-object tracking (MOT) from unmanned aerial vehicles (UAVs) presents unique challenges due to unpredictable object motion, frequent occlusions, and limited appearance cues inherent to aerial viewpoints. These issues are further exacerbated by abrupt UAV movements, leading to unreliable trajectory estimation and identity switches. Conventional motion models, such as Kalman filters or static sequence encoders, often fall short in capturing both linear and non-linear dynamics under such conditions. To tackle these limitations, we propose DMTrack, a deformable motion tracking framework tailored for UAV-based MOT. Our DMTrack introduces three key components: DeformMamba, a deformable state-space predictor that dynamically aggregates historical motion states for adaptive trajectory modeling; MotionGate, a lightweight gating module that fuses Kalman and Mamba predictions based on motion context and uncertainty; and an uncertainty-aware association strategy that enhances identity preservation by aligning motion trends with prediction confidence. Extensive experiments on the VisDrone-MOT and UAVDT benchmarks demonstrate that our DMTrack achieves state-of-the-art performance in identity consistency and tracking accuracy, particularly under high-speed and non-linear motion. Importantly, our method operates without appearance models and maintains competitive efficiency, highlighting its practicality for robust UAV-based tracking.
>
---
#### [new 102] Seg the HAB: Language-Guided Geospatial Algae Bloom Reasoning and Segmentation
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出ALGOS系统，属遥感图像理解任务，旨在解决有害藻华（HAB）的自动监测与严重程度评估问题。结合GeoSAM辅助标注与视觉语言模型微调，实现藻华分割与严重性推理，提升监测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.18751v1](http://arxiv.org/pdf/2510.18751v1)**

> **作者:** Patterson Hsieh; Jerry Yeh; Mao-Chi He; Wen-Han Hsieh; Elvis Hsieh
>
> **摘要:** Climate change is intensifying the occurrence of harmful algal bloom (HAB), particularly cyanobacteria, which threaten aquatic ecosystems and human health through oxygen depletion, toxin release, and disruption of marine biodiversity. Traditional monitoring approaches, such as manual water sampling, remain labor-intensive and limited in spatial and temporal coverage. Recent advances in vision-language models (VLMs) for remote sensing have shown potential for scalable AI-driven solutions, yet challenges remain in reasoning over imagery and quantifying bloom severity. In this work, we introduce ALGae Observation and Segmentation (ALGOS), a segmentation-and-reasoning system for HAB monitoring that combines remote sensing image understanding with severity estimation. Our approach integrates GeoSAM-assisted human evaluation for high-quality segmentation mask curation and fine-tunes vision language model on severity prediction using the Cyanobacteria Aggregated Manual Labels (CAML) from NASA. Experiments demonstrate that ALGOS achieves robust performance on both segmentation and severity-level estimation, paving the way toward practical and automated cyanobacterial monitoring systems.
>
---
#### [new 103] Prototyping an End-to-End Multi-Modal Tiny-CNN for Cardiovascular Sensor Patches
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究在资源受限的医疗边缘设备上实现心电图和心音图多模态信号的实时分类。提出一种端到端轻量级卷积神经网络，采用早期融合策略，在保证高精度的同时显著降低计算与内存开销，并验证其在微控制器上的低能耗可行性。**

- **链接: [http://arxiv.org/pdf/2510.18668v1](http://arxiv.org/pdf/2510.18668v1)**

> **作者:** Mustafa Fuad Rifet Ibrahim; Tunc Alkanat; Maurice Meijer; Felix Manthey; Alexander Schlaefer; Peer Stelldinger
>
> **备注:** Submitted to the IEEE Journal of Biomedical And Health Informatics
>
> **摘要:** The vast majority of cardiovascular diseases may be preventable if early signs and risk factors are detected. Cardiovascular monitoring with body-worn sensor devices like sensor patches allows for the detection of such signs while preserving the freedom and comfort of patients. However, the analysis of the sensor data must be robust, reliable, efficient, and highly accurate. Deep learning methods can automate data interpretation, reducing the workload of clinicians. In this work, we analyze the feasibility of applying deep learning models to the classification of synchronized electrocardiogram (ECG) and phonocardiogram (PCG) recordings on resource-constrained medical edge devices. We propose a convolutional neural network with early fusion of data to solve a binary classification problem. We train and validate our model on the synchronized ECG and PCG recordings from the Physionet Challenge 2016 dataset. Our approach reduces memory footprint and compute cost by three orders of magnitude compared to the state-of-the-art while maintaining competitive accuracy. We demonstrate the applicability of our proposed model on medical edge devices by analyzing energy consumption on a microcontroller and an experimental sensor device setup, confirming that on-device inference can be more energy-efficient than continuous data streaming.
>
---
#### [new 104] CUARewardBench: A Benchmark for Evaluating Reward Models on Computer-using Agent
- **分类: cs.SE; cs.CV**

- **简介: 该论文针对计算机使用代理（CUA）评估中脚本验证局限性问题，提出首个奖励模型基准CUARewardBench，构建包含多软件、多架构轨迹的评测集，分析现有模型缺陷，并提出UPE方法提升评估可靠性。**

- **链接: [http://arxiv.org/pdf/2510.18596v1](http://arxiv.org/pdf/2510.18596v1)**

> **作者:** Haojia Lin; Xiaoyu Tan; Yulei Qin; Zihan Xu; Yuchen Shi; Zongyi Li; Gang Li; Shaofei Cai; Siqi Cai; Chaoyou Fu; Ke Li; Xing Sun
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Computer-using agents (CUAs) enable task completion through natural interaction with operating systems and software interfaces. While script-based verifiers are widely adopted for evaluation, they suffer from limited scalability and inability to provide step-wise assessment. Reward models offer promising alternatives, but their effectiveness on CUA evaluation remains largely underexplored. To address this gap, we present CUARewardBench, comprising four key contributions: (1) First-ever Comprehensive CUA Reward Benchmark: We introduce the first benchmark for evaluating both outcome reward models (ORM) and process reward models (PRM) on CUA tasks, enabling systematic assessment across trajectory-level and step-level evaluation. (2) Diverse, Practical and Reliable Dataset: CUARewardBench encompasses trajectories from 10 software categories and 7 agent architectures with varying performance levels (25.9%-50.8% success rates). All trajectories are expertly annotated through carefully designed protocols, with rigorous quality control to ensure reliability and practical applicability. (3) Comprehensive Analysis and Insights: Through extensive experiments across 7 vision-language models and 3 prompt templates, we reveal critical limitations of current CUA RMs, including insufficient visual reasoning capabilities, knowledge deficiencies, and the superiority of general VLMs over specialized CUA models for reward evaluation. (4) Unanimous Prompt Ensemble (UPE): Based on the insights from our comprehensive analysis, we propose UPE, a novel ensemble method that significantly enhances reward model reliability through strict unanimous voting and strategic prompt-template configurations. UPE achieves 89.8% precision and 93.3% NPV for ORM, and 81.7% precision and 85.1% NPV for PRM, substantially outperforming single VLMs and traditional ensemble approaches.
>
---
#### [new 105] Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboBench，旨在评估多模态大语言模型作为具身智能体“大脑”的高阶认知能力。针对现有基准在任务真实性和评估维度上的不足，构建了涵盖5个维度、25项任务的综合评测体系，并基于真实机器人数据提升现实性，揭示当前模型在理解、推理与规划等方面的局限。**

- **链接: [http://arxiv.org/pdf/2510.17801v1](http://arxiv.org/pdf/2510.17801v1)**

> **作者:** Yulin Luo; Chun-Kai Fan; Menghang Dong; Jiayu Shi; Mengdi Zhao; Bo-Wen Zhang; Cheng Chi; Jiaming Liu; Gaole Dai; Rongyu Zhang; Ruichuan An; Kun Wu; Zhengping Che; Shaoxuan Xie; Guocai Yao; Zhongxia Zhao; Pengwei Wang; Guang Liu; Zhongyuan Wang; Tiejun Huang; Shanghang Zhang
>
> **摘要:** Building robots that can perceive, reason, and act in dynamic, unstructured environments remains a core challenge. Recent embodied systems often adopt a dual-system paradigm, where System 2 handles high-level reasoning while System 1 executes low-level control. In this work, we refer to System 2 as the embodied brain, emphasizing its role as the cognitive core for reasoning and decision-making in manipulation tasks. Given this role, systematic evaluation of the embodied brain is essential. Yet existing benchmarks emphasize execution success, or when targeting high-level reasoning, suffer from incomplete dimensions and limited task realism, offering only a partial picture of cognitive capability. To bridge this gap, we introduce RoboBench, a benchmark that systematically evaluates multimodal large language models (MLLMs) as embodied brains. Motivated by the critical roles across the full manipulation pipeline, RoboBench defines five dimensions-instruction comprehension, perception reasoning, generalized planning, affordance prediction, and failure analysis-spanning 14 capabilities, 25 tasks, and 6092 QA pairs. To ensure realism, we curate datasets across diverse embodiments, attribute-rich objects, and multi-view scenes, drawing from large-scale real robotic data. For planning, RoboBench introduces an evaluation framework, MLLM-as-world-simulator. It evaluate embodied feasibility by simulating whether predicted plans can achieve critical object-state changes. Experiments on 14 MLLMs reveal fundamental limitations: difficulties with implicit instruction comprehension, spatiotemporal reasoning, cross-scenario planning, fine-grained affordance understanding, and execution failure diagnosis. RoboBench provides a comprehensive scaffold to quantify high-level cognition, and guide the development of next-generation embodied MLLMs. The project page is in https://robo-bench.github.io.
>
---
#### [new 106] FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo
- **分类: cs.AI; cs.CV; cs.LG; stat.ML; 68T01; I.2.8**

- **简介: 该论文提出FST.ai 2.0，旨在解决跆拳道赛事中判罚不公、缺乏透明度的问题。通过融合姿态识别、不确定性建模与可解释性技术，构建可信赖的AI系统，支持裁判决策、运动员分析与残奥分级，提升判罚效率与公平性。**

- **链接: [http://arxiv.org/pdf/2510.18193v1](http://arxiv.org/pdf/2510.18193v1)**

> **作者:** Keivan Shariatmadar; Ahmad Osman; Ramin Ray; Usman Dildar; Kisam Kim
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Fair, transparent, and explainable decision-making remains a critical challenge in Olympic and Paralympic combat sports. This paper presents \emph{FST.ai 2.0}, an explainable AI ecosystem designed to support referees, coaches, and athletes in real time during Taekwondo competitions and training. The system integrates {pose-based action recognition} using graph convolutional networks (GCNs), {epistemic uncertainty modeling} through credal sets, and {explainability overlays} for visual decision support. A set of {interactive dashboards} enables human--AI collaboration in referee evaluation, athlete performance analysis, and Para-Taekwondo classification. Beyond automated scoring, FST.ai~2.0 incorporates modules for referee training, fairness monitoring, and policy-level analytics within the World Taekwondo ecosystem. Experimental validation on competition data demonstrates an {85\% reduction in decision review time} and {93\% referee trust} in AI-assisted decisions. The framework thus establishes a transparent and extensible pipeline for trustworthy, data-driven officiating and athlete assessment. By bridging real-time perception, explainable inference, and governance-aware design, FST.ai~2.0 represents a step toward equitable, accountable, and human-aligned AI in sports.
>
---
#### [new 107] Ensembling Pruned Attention Heads For Uncertainty-Aware Efficient Transformers
- **分类: cs.LG; cs.CV**

- **简介: 该论文聚焦模型不确定性量化任务，旨在解决深度模型在安全关键场景中计算成本高、扩展性差的问题。作者提出Hydra Ensembles，通过剪枝注意力头并融合构建高效集成模型，在保持低推理成本的同时，实现了优于Deep Ensembles的不确定性估计性能。**

- **链接: [http://arxiv.org/pdf/2510.18358v1](http://arxiv.org/pdf/2510.18358v1)**

> **作者:** Firas Gabetni; Giuseppe Curci; Andrea Pilzer; Subhankar Roy; Elisa Ricci; Gianni Franchi
>
> **摘要:** Uncertainty quantification (UQ) is essential for deploying deep neural networks in safety-critical settings. Although methods like Deep Ensembles achieve strong UQ performance, their high computational and memory costs hinder scalability to large models. We introduce Hydra Ensembles, an efficient transformer-based ensemble that prunes attention heads to create diverse members and merges them via a new multi-head attention with grouped fully-connected layers. This yields a compact model with inference speed close to a single network, matching or surpassing Deep Ensembles in UQ performance without retraining from scratch. We also provide an in-depth analysis of pruning, showing that naive approaches can harm calibration, whereas Hydra Ensembles preserves robust uncertainty. Experiments on image and text classification tasks, with various architectures, show consistent gains over Deep Ensembles. Remarkably, in zero-shot classification on ImageNet-1k, our approach surpasses state of the art methods, even without requiring additional training.
>
---
#### [new 108] Cross-Domain Multi-Person Human Activity Recognition via Near-Field Wi-Fi Sensing
- **分类: eess.SP; cs.CV**

- **简介: 该论文研究Wi-Fi近场感知下的跨域多人活动识别任务，旨在解决因个体差异和缺失活动类别导致的模型泛化难题。提出WiAnchor框架，通过预训练增强特征可分性，引入锚点匹配机制实现高效跨域自适应，在类别不全下仍达90%以上准确率。**

- **链接: [http://arxiv.org/pdf/2510.17816v1](http://arxiv.org/pdf/2510.17816v1)**

> **作者:** Xin Li; Jingzhi Hu; Yinghui He; Hongbo Wang; Jin Gan; Jun Luo
>
> **摘要:** Wi-Fi-based human activity recognition (HAR) provides substantial convenience and has emerged as a thriving research field, yet the coarse spatial resolution inherent to Wi-Fi significantly hinders its ability to distinguish multiple subjects. By exploiting the near-field domination effect, establishing a dedicated sensing link for each subject through their personal Wi-Fi device offers a promising solution for multi-person HAR under native traffic. However, due to the subject-specific characteristics and irregular patterns of near-field signals, HAR neural network models require fine-tuning (FT) for cross-domain adaptation, which becomes particularly challenging with certain categories unavailable. In this paper, we propose WiAnchor, a novel training framework for efficient cross-domain adaptation in the presence of incomplete activity categories. This framework processes Wi-Fi signals embedded with irregular time information in three steps: during pre-training, we enlarge inter-class feature margins to enhance the separability of activities; in the FT stage, we innovate an anchor matching mechanism for cross-domain adaptation, filtering subject-specific interference informed by incomplete activity categories, rather than attempting to extract complete features from them; finally, the recognition of input samples is further improved based on their feature-level similarity with anchors. We construct a comprehensive dataset to thoroughly evaluate WiAnchor, achieving over 90% cross-domain accuracy with absent activity categories.
>
---
#### [new 109] LightMem: Lightweight and Efficient Memory-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出LightMem，一种轻量高效的记忆增强生成系统，旨在解决大模型在动态环境中难以有效利用历史信息的问题。受人类记忆模型启发，其分三阶段记忆机制显著提升效率与性能，大幅降低开销。**

- **链接: [http://arxiv.org/pdf/2510.18866v1](http://arxiv.org/pdf/2510.18866v1)**

> **作者:** Jizhan Fang; Xinle Deng; Haoming Xu; Ziyan Jiang; Yuqi Tang; Ziwen Xu; Shumin Deng; Yunzhi Yao; Mengru Wang; Shuofei Qiao; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. Experiments on LongMemEval with GPT and Qwen backbones show that LightMem outperforms strong baselines in accuracy (up to 10.9% gains) while reducing token usage by up to 117x, API calls by up to 159x, and runtime by over 12x. The code is available at https://github.com/zjunlp/LightMem.
>
---
#### [new 110] DualHash: A Stochastic Primal-Dual Algorithm with Theoretical Guarantee for Deep Hashing
- **分类: math.OC; cs.CV**

- **简介: 该论文研究深度哈希任务，旨在解决二值量化优化中缺乏收敛保证的问题。作者提出DualHash算法，通过Fenchel对偶将非凸正则项转化为对偶空间，获得闭式解，并提供理论收敛保证，显著提升大规模图像检索性能。**

- **链接: [http://arxiv.org/pdf/2510.18218v1](http://arxiv.org/pdf/2510.18218v1)**

> **作者:** Luxuan Li; Xiao Wang; Chunfeng Cui
>
> **摘要:** Deep hashing converts high-dimensional feature vectors into compact binary codes, enabling efficient large-scale retrieval. A fundamental challenge in deep hashing stems from the discrete nature of quantization in generating the codes. W-type regularizations, such as $||z|-1|$, have been proven effective as they encourage variables toward binary values. However, existing methods often directly optimize these regularizations without convergence guarantees. While proximal gradient methods offer a promising solution, the coupling between W-type regularizers and neural network outputs results in composite forms that generally lack closed-form proximal solutions. In this paper, we present a stochastic primal-dual hashing algorithm, referred to as DualHash, that provides rigorous complexity bounds. Using Fenchel duality, we partially transform the nonconvex W-type regularization optimization into the dual space, which results in a proximal operator that admits closed-form solutions. We derive two algorithm instances: a momentum-accelerated version with $\mathcal{O}(\varepsilon^{-4})$ complexity and an improved $\mathcal{O}(\varepsilon^{-3})$ version using variance reduction. Experiments on three image retrieval databases demonstrate the superior performance of DualHash.
>
---
#### [new 111] Demystifying Transition Matching: When and Why It Can Beat Flow Matching
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究生成模型中的Flow Matching（FM）与Transition Matching（TM）性能差异，旨在解释TM何时及为何优于FM。通过理论分析与实验验证，表明在目标分布为多模态且模式分离良好时，TM因保留协方差结构而表现更优。**

- **链接: [http://arxiv.org/pdf/2510.17991v1](http://arxiv.org/pdf/2510.17991v1)**

> **作者:** Jaihoon Kim; Rajarshi Saha; Minhyuk Sung; Youngsuk Park
>
> **摘要:** Flow Matching (FM) underpins many state-of-the-art generative models, yet recent results indicate that Transition Matching (TM) can achieve higher quality with fewer sampling steps. This work answers the question of when and why TM outperforms FM. First, when the target is a unimodal Gaussian distribution, we prove that TM attains strictly lower KL divergence than FM for finite number of steps. The improvement arises from stochastic difference latent updates in TM, which preserve target covariance that deterministic FM underestimates. We then characterize convergence rates, showing that TM achieves faster convergence than FM under a fixed compute budget, establishing its advantage in the unimodal Gaussian setting. Second, we extend the analysis to Gaussian mixtures and identify local-unimodality regimes in which the sampling dynamics approximate the unimodal case, where TM can outperform FM. The approximation error decreases as the minimal distance between component means increases, highlighting that TM is favored when the modes are well separated. However, when the target variance approaches zero, each TM update converges to the FM update, and the performance advantage of TM diminishes. In summary, we show that TM outperforms FM when the target distribution has well-separated modes and non-negligible variances. We validate our theoretical results with controlled experiments on Gaussian distributions, and extend the comparison to real-world applications in image and video generation.
>
---
#### [new 112] A Generalizable Light Transport 3D Embedding for Global Illumination
- **分类: cs.GR; cs.CV**

- **简介: 该论文研究全局光照的快速近似，旨在解决传统方法计算昂贵、现有神经方法泛化性差的问题。提出一种可泛化的3D光传输嵌入，直接从3D场景配置预测全局光照，支持跨场景泛化与多任务适应，无需依赖光栅化或路径追踪输入。**

- **链接: [http://arxiv.org/pdf/2510.18189v1](http://arxiv.org/pdf/2510.18189v1)**

> **作者:** Bing Xu; Mukund Varma T; Cheng Wang; Tzumao Li; Lifan Wu; Bartlomiej Wronski; Ravi Ramamoorthi; Marco Salvi
>
> **摘要:** Global illumination (GI) is essential for realistic rendering but remains computationally expensive due to the complexity of simulating indirect light transport. Recent neural methods have mainly relied on per-scene optimization, sometimes extended to handle changes in camera or geometry. Efforts toward cross-scene generalization have largely stayed in 2D screen space, such as neural denoising or G-buffer based GI prediction, which often suffer from view inconsistency and limited spatial understanding. We propose a generalizable 3D light transport embedding that approximates global illumination directly from 3D scene configurations, without using rasterized or path-traced cues. Each scene is represented as a point cloud with geometric and material features. A scalable transformer models global point-to-point interactions to encode these features into neural primitives. At render time, each query point retrieves nearby primitives via nearest-neighbor search and aggregates their latent features through cross-attention to predict the desired rendering quantity. We demonstrate results on diffuse global illumination prediction across diverse indoor scenes with varying layouts, geometry, and materials. The embedding trained for irradiance estimation can be quickly adapted to new rendering tasks with limited fine-tuning. We also present preliminary results for spatial-directional radiance field estimation for glossy materials and show how the normalized field can accelerate unbiased path guiding. This approach highlights a path toward integrating learned priors into rendering pipelines without explicit ray-traced illumination cues.
>
---
#### [new 113] NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出NeuCo-Bench，旨在标准化评估地球观测中的神经压缩与表示学习。它构建固定大小的通用嵌入，通过可复用嵌入、隐藏任务排行榜和平衡精度稳定性的评分系统，解决预训练偏差问题，并发布数据集支持可重复研究。**

- **链接: [http://arxiv.org/pdf/2510.17914v1](http://arxiv.org/pdf/2510.17914v1)**

> **作者:** Rikard Vinge; Isabelle Wittmann; Jannik Schneider; Michael Marszalek; Luis Gilch; Thomas Brunschwiler; Conrad M Albrecht
>
> **摘要:** We introduce NeuCo-Bench, a novel benchmark framework for evaluating (lossy) neural compression and representation learning in the context of Earth Observation (EO). Our approach builds on fixed-size embeddings that act as compact, task-agnostic representations applicable to a broad range of downstream tasks. NeuCo-Bench comprises three core components: (i) an evaluation pipeline built around reusable embeddings, (ii) a new challenge mode with a hidden-task leaderboard designed to mitigate pretraining bias, and (iii) a scoring system that balances accuracy and stability. To support reproducibility, we release SSL4EO-S12-downstream, a curated multispectral, multitemporal EO dataset. We present initial results from a public challenge at the 2025 CVPR EARTHVISION workshop and conduct ablations with state-of-the-art foundation models. NeuCo-Bench provides a first step towards community-driven, standardized evaluation of neural embeddings for EO and beyond.
>
---
#### [new 114] Metrics and evaluations for computational and sustainable AI efficiency
- **分类: cs.PF; cs.AI; cs.CL; cs.CV**

- **简介: 该论文聚焦AI能效评估任务，旨在解决现有评价方法碎片化、缺乏统一标准的问题。作者提出一种集计算与环境指标于一体的可复现评估框架，涵盖延迟、吞吐、能耗及碳排放，支持跨硬件、软件和精度的公平比较，并发布开源代码促进可持续AI决策。**

- **链接: [http://arxiv.org/pdf/2510.17885v1](http://arxiv.org/pdf/2510.17885v1)**

> **作者:** Hongyuan Liu; Xinyang Liu; Guosheng Hu
>
> **备注:** 11 pages, 2 tables
>
> **摘要:** The rapid advancement of Artificial Intelligence (AI) has created unprecedented demands for computational power, yet methods for evaluating the performance, efficiency, and environmental impact of deployed models remain fragmented. Current approaches often fail to provide a holistic view, making it difficult to compare and optimise systems across heterogeneous hardware, software stacks, and numeric precisions. To address this gap, we propose a unified and reproducible methodology for AI model inference that integrates computational and environmental metrics under realistic serving conditions. Our framework provides a pragmatic, carbon-aware evaluation by systematically measuring latency and throughput distributions, energy consumption, and location-adjusted carbon emissions, all while maintaining matched accuracy constraints for valid comparisons. We apply this methodology to multi-precision models across diverse hardware platforms, from data-centre accelerators like the GH200 to consumer-level GPUs such as the RTX 4090, running on mainstream software stacks including PyTorch, TensorRT, and ONNX Runtime. By systematically categorising these factors, our work establishes a rigorous benchmarking framework that produces decision-ready Pareto frontiers, clarifying the trade-offs between accuracy, latency, energy, and carbon. The accompanying open-source code enables independent verification and facilitates adoption, empowering researchers and practitioners to make evidence-based decisions for sustainable AI deployment.
>
---
## 更新

#### [replaced 001] Glyph: Scaling Context Windows via Visual-Text Compression
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.17800v2](http://arxiv.org/pdf/2510.17800v2)**

> **作者:** Jiale Cheng; Yusen Liu; Xinyu Zhang; Yulin Fei; Wenyi Hong; Ruiliang Lyu; Weihan Wang; Zhe Su; Xiaotao Gu; Xiao Liu; Yushi Bai; Jie Tang; Hongning Wang; Minlie Huang
>
> **摘要:** Large language models (LLMs) increasingly rely on long-context modeling for tasks such as document understanding, code analysis, and multi-step reasoning. However, scaling context windows to the million-token level brings prohibitive computational and memory costs, limiting the practicality of long-context LLMs. In this work, we take a different perspective-visual context scaling-to tackle this challenge. Instead of extending token-based sequences, we propose Glyph, a framework that renders long texts into images and processes them with vision-language models (VLMs). This approach substantially compresses textual input while preserving semantic information, and we further design an LLM-driven genetic search to identify optimal visual rendering configurations for balancing accuracy and compression. Through extensive experiments, we demonstrate that our method achieves 3-4x token compression while maintaining accuracy comparable to leading LLMs such as Qwen3-8B on various long-context benchmarks. This compression also leads to around 4x faster prefilling and decoding, and approximately 2x faster SFT training. Furthermore, under extreme compression, a 128K-context VLM could scale to handle 1M-token-level text tasks. In addition, the rendered text data benefits real-world multimodal tasks, such as document understanding. Our code and model are released at https://github.com/thu-coai/Glyph.
>
---
#### [replaced 002] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17626v2](http://arxiv.org/pdf/2510.17626v2)**

> **作者:** Frédéric LIN; Biruk Abere Ambaw; Adrian Popescu; Hejer Ammar; Romaric Audigier; Hervé Le Borgne
>
> **备注:** To be published in NeurIPS 2025 Track on Datasets and Benchmarks
>
> **摘要:** AI systems must adapt to evolving visual environments, especially in domains where object appearances change over time. We introduce Car Models in Time (CaMiT), a fine-grained dataset capturing the temporal evolution of car models, a representative class of technological artifacts. CaMiT includes 787K labeled samples of 190 car models (2007-2023) and 5.1M unlabeled samples (2005-2023), supporting both supervised and self-supervised learning. Static pretraining on in-domain data achieves competitive performance with large-scale generalist models while being more resource-efficient, yet accuracy declines when models are tested across years. To address this, we propose a time-incremental classification setting, a realistic continual learning scenario with emerging, evolving, and disappearing classes. We evaluate two strategies: time-incremental pretraining, which updates the backbone, and time-incremental classifier learning, which updates only the final layer, both improving temporal robustness. Finally, we explore time-aware image generation that leverages temporal metadata during training, yielding more realistic outputs. CaMiT offers a rich benchmark for studying temporal adaptation in fine-grained visual recognition and generation.
>
---
#### [replaced 003] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10759v2](http://arxiv.org/pdf/2509.10759v2)**

> **作者:** Yi-Ruei Liu; You-Zhe Xie; Yu-Hsiang Hsu; I-Sheng Fang; Yu-Lun Liu; Jun-Cheng Chen
>
> **备注:** Paper accepted to NeurIPS 2025 Workshop SpaVLE. Project page: https://shigon255.github.io/4DGRT-project-page/
>
> **摘要:** Common computer vision systems typically assume ideal pinhole cameras but fail when facing real-world camera effects such as fisheye distortion and rolling shutter, mainly due to the lack of learning from training data with camera effects. Existing data generation approaches suffer from either high costs, sim-to-real gaps or fail to accurately model camera effects. To address this bottleneck, we propose 4D Gaussian Ray Tracing (4D-GRT), a novel two-stage pipeline that combines 4D Gaussian Splatting with physically-based ray tracing for camera effect simulation. Given multi-view videos, 4D-GRT first reconstructs dynamic scenes, then applies ray tracing to generate videos with controllable, physically accurate camera effects. 4D-GRT achieves the fastest rendering speed while performing better or comparable rendering quality compared to existing baselines. Additionally, we construct eight synthetic dynamic scenes in indoor environments across four camera effects as a benchmark to evaluate generated videos with camera effects.
>
---
#### [replaced 004] Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16888v2](http://arxiv.org/pdf/2510.16888v2)**

> **作者:** Zongjian Li; Zheyuan Liu; Qihui Zhang; Bin Lin; Feize Wu; Shenghai Yuan; Zhiyuan Yan; Yang Ye; Wangbo Yu; Yuwei Niu; Shaodong Wang; Xinhua Cheng; Li Yuan
>
> **摘要:** Instruction-based image editing has achieved remarkable progress; however, models solely trained via supervised fine-tuning often overfit to annotated patterns, hindering their ability to explore and generalize beyond training distributions. To this end, we introduce Edit-R1, a novel post-training framework for instruction-based image editing based on policy optimization. Specifically, we utilize Diffusion Negative-aware Finetuning (DiffusionNFT), a likelihood-free policy optimization method consistent with the flow matching forward process, thereby enabling the use of higher-order samplers and more efficient training. Another key challenge here is the absence of a universal reward model, resulting from the diverse nature of editing instructions and tasks. To bridge this gap, we employ a Multimodal Large Language Model (MLLM) as a unified, training-free reward model, leveraging its output logits to provide fine-grained feedback. Furthermore, we carefully design a low-variance group filtering mechanism to reduce MLLM scoring noise and stabilize optimization. UniWorld-V2, trained with this framework, achieves \textbf{state-of-the-art} results on the ImgEdit and GEdit-Bench benchmarks, scoring 4.49 and 7.83, respectively. Crucially, our framework is model-agnostic, delivering substantial performance gains when applied to diverse base models like Qwen-Image-Edit and FLUX-Kontext, demonstrating its wide applicability. Code and models are publicly available at https://github.com/PKU-YuanGroup/UniWorld-V2.
>
---
#### [replaced 005] GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.06806v2](http://arxiv.org/pdf/2507.06806v2)**

> **作者:** Eya Cherif; Arthur Ouaknine; Luke A. Brown; Phuong D. Dao; Kyle R. Kovach; Bing Lu; Daniel Mederer; Hannes Feilhauer; Teja Kattenborn; David Rolnick
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Plant traits such as leaf carbon content and leaf mass are essential variables in the study of biodiversity and climate change. However, conventional field sampling cannot feasibly cover trait variation at ecologically meaningful spatial scales. Machine learning represents a valuable solution for plant trait prediction across ecosystems, leveraging hyperspectral data from remote sensing. Nevertheless, trait prediction from hyperspectral data is challenged by label scarcity and substantial domain shifts (\eg across sensors, ecological distributions), requiring robust cross-domain methods. Here, we present GreenHyperSpectra, a pretraining dataset encompassing real-world cross-sensor and cross-ecosystem samples designed to benchmark trait prediction with semi- and self-supervised methods. We adopt an evaluation framework encompassing in-distribution and out-of-distribution scenarios. We successfully leverage GreenHyperSpectra to pretrain label-efficient multi-output regression models that outperform the state-of-the-art supervised baseline. Our empirical analyses demonstrate substantial improvements in learning spectral representations for trait prediction, establishing a comprehensive methodological framework to catalyze research at the intersection of representation learning and plant functional traits assessment. All code and data are available at: https://github.com/echerif18/HyspectraSSL.
>
---
#### [replaced 006] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08997v2](http://arxiv.org/pdf/2506.08997v2)**

> **作者:** Fabian Immel; Jan-Hendrik Pauls; Richard Fehler; Frank Bieder; Jonas Merkert; Christoph Stiller
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Autonomous vehicles rely on detailed and accurate environmental information to operate safely. High definition (HD) maps offer a promising solution, but their high maintenance cost poses a significant barrier to scalable deployment. This challenge is addressed by online HD map construction methods, which generate local HD maps from live sensor data. However, these methods are inherently limited by the short perception range of onboard sensors. To overcome this limitation and improve general performance, recent approaches have explored the use of standard definition (SD) maps as prior, which are significantly easier to maintain. We propose SDTagNet, the first online HD map construction method that fully utilizes the information of widely available SD maps, like OpenStreetMap, to enhance far range detection accuracy. Our approach introduces two key innovations. First, in contrast to previous work, we incorporate not only polyline SD map data with manually selected classes, but additional semantic information in the form of textual annotations. In this way, we enrich SD vector map tokens with NLP-derived features, eliminating the dependency on predefined specifications or exhaustive class taxonomies. Second, we introduce a point-level SD map encoder together with orthogonal element identifiers to uniformly integrate all types of map elements. Experiments on Argoverse 2 and nuScenes show that this boosts map perception performance by up to +5.9 mAP (+45%) w.r.t. map construction without priors and up to +3.2 mAP (+20%) w.r.t. previous approaches that already use SD map priors. Code is available at https://github.com/immel-f/SDTagNet
>
---
#### [replaced 007] Regression is all you need for medical image translation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02048v3](http://arxiv.org/pdf/2505.02048v3)**

> **作者:** Sebastian Rassmann; David Kügler; Christian Ewert; Martin Reuter
>
> **摘要:** While Generative Adversarial Nets (GANs) and Diffusion Models (DMs) have achieved impressive results in natural image synthesis, their core strengths - creativity and realism - can be detrimental in medical applications, where accuracy and fidelity are paramount. These models instead risk introducing hallucinations and replication of unwanted acquisition noise. Here, we propose YODA (You Only Denoise once - or Average), a 2.5D diffusion-based framework for medical image translation (MIT). Consistent with DM theory, we find that conventional diffusion sampling stochastically replicates noise. To mitigate this, we draw and average multiple samples, akin to physical signal averaging. As this effectively approximates the DM's expected value, we term this Expectation-Approximation (ExpA) sampling. We additionally propose regression sampling YODA, which retains the initial DM prediction and omits iterative refinement to produce noise-free images in a single step. Across five diverse multi-modal datasets - including multi-contrast brain MRI and pelvic MRI-CT - we demonstrate that regression sampling is not only substantially more efficient but also matches or exceeds image quality of full diffusion sampling even with ExpA. Our results reveal that iterative refinement solely enhances perceptual realism without benefiting information translation, which we confirm in relevant downstream tasks. YODA outperforms eight state-of-the-art DMs and GANs and challenges the presumed superiority of DMs and GANs over computationally cheap regression models for high-quality MIT. Furthermore, we show that YODA-translated images are interchangeable with, or even superior to, physical acquisitions for several medical applications.
>
---
#### [replaced 008] Class-wise Balancing Data Replay for Federated Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07712v3](http://arxiv.org/pdf/2507.07712v3)**

> **作者:** Zhuang Qi; Ying-Peng Tang; Lei Meng; Han Yu; Xiaoxiao Li; Xiangxu Meng
>
> **备注:** NeurIPS'25 Accepted, Oral
>
> **摘要:** Federated Class Incremental Learning (FCIL) aims to collaboratively process continuously increasing incoming tasks across multiple clients. Among various approaches, data replay has become a promising solution, which can alleviate forgetting by reintroducing representative samples from previous tasks. However, their performance is typically limited by class imbalance, both within the replay buffer due to limited global awareness and between replayed and newly arrived classes. To address this issue, we propose a class wise balancing data replay method for FCIL (FedCBDR), which employs a global coordination mechanism for class-level memory construction and reweights the learning objective to alleviate the aforementioned imbalances. Specifically, FedCBDR has two key components: 1) the global-perspective data replay module reconstructs global representations of prior task in a privacy-preserving manner, which then guides a class-aware and importance-sensitive sampling strategy to achieve balanced replay; 2) Subsequently, to handle class imbalance across tasks, the task aware temperature scaling module adaptively adjusts the temperature of logits at both class and instance levels based on task dynamics, which reduces the model's overconfidence in majority classes while enhancing its sensitivity to minority classes. Experimental results verified that FedCBDR achieves balanced class-wise sampling under heterogeneous data distributions and improves generalization under task imbalance between earlier and recent tasks, yielding a 2%-15% Top-1 accuracy improvement over six state-of-the-art methods.
>
---
#### [replaced 009] Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14677v2](http://arxiv.org/pdf/2505.14677v2)**

> **作者:** Jiaer Xia; Yuhang Zang; Peng Gao; Yixuan Li; Kaiyang Zhou
>
> **摘要:** Learning general-purpose reasoning capabilities has long been a challenging problem in AI. Recent research in large language models (LLMs), such as DeepSeek-R1, has shown that reinforcement learning techniques like GRPO can enable pre-trained LLMs to develop reasoning capabilities using simple question-answer pairs. In this paper, we aim to train visual language models (VLMs) to perform reasoning on image data through reinforcement learning and visual question-answer pairs, without any explicit chain-of-thought (CoT) supervision. Our findings indicate that simply applying reinforcement learning to a VLM -- by prompting the model to produce a reasoning chain before providing an answer -- can lead the model to develop shortcuts from easy questions, thereby reducing its ability to generalize across unseen data distributions. We argue that the key to mitigating shortcut learning is to encourage the model to interpret images prior to reasoning. Therefore, we train the model to adhere to a caption-reason-answer output format: initially generating a detailed caption for an image, followed by constructing an extensive reasoning chain. When trained on 273K CoT-free visual question-answer pairs and using only reinforcement learning, our model, named Visionary-R1, outperforms strong multimodal models, such as GPT-4o, Claude3.5-Sonnet, and Gemini-1.5-Pro, on multiple visual reasoning benchmarks.
>
---
#### [replaced 010] Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.15202v2](http://arxiv.org/pdf/2510.15202v2)**

> **作者:** Denis Janiak; Jakub Binkowski; Tomasz Kajdanowicz
>
> **摘要:** Out-of-distribution (OOD) detection is critical for the reliable deployment of deep learning models. hile Mahalanobis distance methods are widely used, the impact of representation geometry and normalization on their performance is not fully understood, which may limit their downstream application. To address this gap, we conducted a comprehensive empirical study across diverse image foundation models, datasets, and distance normalization schemes. First, our analysis shows that Mahalanobis-based methods aren't universally reliable. Second, we define the ideal geometry for data representations and demonstrate that spectral and intrinsic-dimensionality metrics can accurately predict a model's OOD performance. Finally, we analyze how normalization impacts OOD performance. Building upon these studies, we propose radially scaled $\ell_2$ normalization, a method that generalizes the standard $\ell_2$ normalization recently applied to Mahalanobis-based OOD detection. Our approach introduces a tunable parameter to directly control the radial geometry of the feature space, systematically contracting or expanding representations to significantly improve OOD detection performance. By bridging the gap between representation geometry, normalization, and OOD performance, our findings offer new insights into the design of more effective and reliable deep learning models.
>
---
#### [replaced 011] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14255v2](http://arxiv.org/pdf/2510.14255v2)**

> **作者:** Liao Shen; Wentao Jiang; Yiran Zhu; Jiahe Li; Tiezheng Ge; Zhiguo Cao; Bo Zheng
>
> **摘要:** Recent advances in image-to-video (I2V) generation have achieved remarkable progress in synthesizing high-quality, temporally coherent videos from static images. Among all the applications of I2V, human-centric video generation includes a large portion. However, existing I2V models encounter difficulties in maintaining identity consistency between the input human image and the generated video, especially when the person in the video exhibits significant expression changes and movements. This issue becomes critical when the human face occupies merely a small fraction of the image. Since humans are highly sensitive to identity variations, this poses a critical yet under-explored challenge in I2V generation. In this paper, we propose Identity-Preserving Reward-guided Optimization (IPRO), a novel video diffusion framework based on reinforcement learning to enhance identity preservation. Instead of introducing auxiliary modules or altering model architectures, our approach introduces a direct and effective tuning algorithm that optimizes diffusion models using a face identity scorer. To improve performance and accelerate convergence, our method backpropagates the reward signal through the last steps of the sampling chain, enabling richer gradient feedback. We also propose a novel facial scoring mechanism that treats faces in ground-truth videos as facial feature pools, providing multi-angle facial information to enhance generalization. A KL-divergence regularization is further incorporated to stabilize training and prevent overfitting to the reward signal. Extensive experiments on Wan 2.2 I2V model and our in-house I2V model demonstrate the effectiveness of our method. Our project and code are available at https://ipro-alimama.github.io/.
>
---
#### [replaced 012] gen2seg: Generative Models Enable Generalizable Instance Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15263v2](http://arxiv.org/pdf/2505.15263v2)**

> **作者:** Om Khangaonkar; Hamed Pirsiavash
>
> **备注:** Website: https://reachomk.github.io/gen2seg/
>
> **摘要:** By pretraining to synthesize coherent images from perturbed inputs, generative models inherently learn to understand object boundaries and scene compositions. How can we repurpose these generative representations for general-purpose perceptual organization? We finetune Stable Diffusion and MAE (encoder+decoder) for category-agnostic instance segmentation using our instance coloring loss exclusively on a narrow set of object types (indoor furnishings and cars). Surprisingly, our models exhibit strong zero-shot generalization, accurately segmenting objects of types and styles unseen in finetuning (and in many cases, MAE's ImageNet-1K pretraining too). Our best-performing models closely approach the heavily supervised SAM when evaluated on unseen object types and styles, and outperform it when segmenting fine structures and ambiguous boundaries. In contrast, existing promptable segmentation architectures or discriminatively pretrained models fail to generalize. This suggests that generative models learn an inherent grouping mechanism that transfers across categories and domains, even without internet-scale pretraining. Code, pretrained models, and demos are available on our website.
>
---
#### [replaced 013] Context-Aware Pseudo-Label Scoring for Zero-Shot Video Summarization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17501v2](http://arxiv.org/pdf/2510.17501v2)**

> **作者:** Yuanli Wu; Long Zhang; Yue Du; Bin Li
>
> **摘要:** With video exploding across social media, surveillance, and education, compressing long footage into concise yet faithful surrogates is crucial. Supervised methods learn frame/shot importance from dense labels and excel in-domain, but are costly and brittle across datasets; unsupervised methods avoid labels but often miss high-level semantics and narrative cues. Recent zero-shot pipelines use LLMs for training-free summarization, yet remain sensitive to handcrafted prompts and dataset-specific normalization.We propose a rubric-guided, pseudo-labeled prompting framework. A small subset of human annotations is converted into high-confidence pseudo labels and aggregated into structured, dataset-adaptive scoring rubrics for interpretable scene evaluation. At inference, boundary scenes (first/last) are scored from their own descriptions, while intermediate scenes include brief summaries of adjacent segments to assess progression and redundancy, enabling the LLM to balance local salience with global coherence without parameter tuning.Across three benchmarks, our method is consistently effective. On SumMe and TVSum it achieves F1 of 57.58 and 63.05, surpassing a zero-shot baseline (56.73, 62.21) by +0.85 and +0.84 and approaching supervised performance. On the query-focused QFVS benchmark it attains 53.79 F1, beating 53.42 by +0.37 and remaining stable across validation videos. These results show that rubric-guided pseudo labeling, coupled with contextual prompting, stabilizes LLM-based scoring and yields a general, interpretable zero-shot paradigm for both generic and query-focused video summarization.
>
---
#### [replaced 014] Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09423v2](http://arxiv.org/pdf/2508.09423v2)**

> **作者:** Badi Li; Ren-jie Lu; Yu Zhou; Jingke Meng; Wei-shi Zheng
>
> **摘要:** The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at https://github.com/Badi-Li/GOAL.
>
---
#### [replaced 015] VideoVerse: How Far is Your T2V Generator from a World Model?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.08398v2](http://arxiv.org/pdf/2510.08398v2)**

> **作者:** Zeqing Wang; Xinyu Wei; Bairui Li; Zhen Guo; Jinrui Zhang; Hongyang Wei; Keze Wang; Lei Zhang
>
> **备注:** 24 Pages, 8 Figures, 11 Tables
>
> **摘要:** The recent rapid advancement of Text-to-Video (T2V) generation technologies, which are critical to build ``world models'', makes the existing benchmarks increasingly insufficient to evaluate state-of-the-art T2V models. First, current evaluation dimensions, such as per-frame aesthetic quality and temporal consistency, are no longer able to differentiate state-of-the-art T2V models. Second, event-level temporal causality, which not only distinguishes video from other modalities but also constitutes a crucial component of world models, is severely underexplored in existing benchmarks. Third, existing benchmarks lack a systematic assessment of world knowledge, which are essential capabilities for building world models. To address these issues, we introduce VideoVerse, a comprehensive benchmark that focuses on evaluating whether a T2V model could understand complex temporal causality and world knowledge in the real world. We collect representative videos across diverse domains (e.g., natural landscapes, sports, indoor scenes, science fiction, chemical and physical experiments) and extract their event-level descriptions with inherent temporal causality, which are then rewritten into text-to-video prompts by independent annotators. For each prompt, we design a suite of binary evaluation questions from the perspective of dynamic and static properties, with a total of ten carefully defined evaluation dimensions. In total, our VideoVerse comprises 300 carefully curated prompts, involving 815 events and 793 binary evaluation questions. Consequently, a human preference aligned QA-based evaluation pipeline is developed by using modern vision-language models. Finally, we perform a systematic evaluation of state-of-the-art open-source and closed-source T2V models on VideoVerse, providing in-depth analysis on how far the current T2V generators are from world models.
>
---
#### [replaced 016] Re-ttention: Ultra Sparse Visual Generation via Attention Statistical Reshape
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22918v3](http://arxiv.org/pdf/2505.22918v3)**

> **作者:** Ruichen Chen; Keith G. Mills; Liyao Jiang; Chao Gao; Di Niu
>
> **备注:** author comment: This version was previously removed by arXiv administrators as the submitter did not have the rights to agree to the license at the time of submission. The authors have now obtained the necessary permissions, and the paper is resubmitted accordingly
>
> **摘要:** Diffusion Transformers (DiT) have become the de-facto model for generating high-quality visual content like videos and images. A huge bottleneck is the attention mechanism where complexity scales quadratically with resolution and video length. One logical way to lessen this burden is sparse attention, where only a subset of tokens or patches are included in the calculation. However, existing techniques fail to preserve visual quality at extremely high sparsity levels and might even incur non-negligible compute overheads. To address this concern, we propose Re-ttention, which implements very high sparse attention for visual generation models by leveraging the temporal redundancy of Diffusion Models to overcome the probabilistic normalization shift within the attention mechanism. Specifically, Re-ttention reshapes attention scores based on the prior softmax distribution history in order to preserve the visual quality of the full quadratic attention at very high sparsity levels. Experimental results on T2V/T2I models such as CogVideoX and the PixArt DiTs demonstrate that Re-ttention requires as few as 3.1% of the tokens during inference, outperforming contemporary methods like FastDiTAttn, Sparse VideoGen and MInference.
>
---
#### [replaced 017] DiffVLA++: Bridging Cognitive Reasoning and End-to-End Driving through Metric-Guided Alignment
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17148v2](http://arxiv.org/pdf/2510.17148v2)**

> **作者:** Yu Gao; Anqing Jiang; Yiru Wang; Heng Yuwen; Wang Shuo; Sun Hao; Wang Jijun
>
> **摘要:** Conventional end-to-end (E2E) driving models are effective at generating physically plausible trajectories, but often fail to generalize to long-tail scenarios due to the lack of essential world knowledge to understand and reason about surrounding environments. In contrast, Vision-Language-Action (VLA) models leverage world knowledge to handle challenging cases, but their limited 3D reasoning capability can lead to physically infeasible actions. In this work we introduce DiffVLA++, an enhanced autonomous driving framework that explicitly bridges cognitive reasoning and E2E planning through metric-guided alignment. First, we build a VLA module directly generating semantically grounded driving trajectories. Second, we design an E2E module with a dense trajectory vocabulary that ensures physical feasibility. Third, and most critically, we introduce a metric-guided trajectory scorer that guides and aligns the outputs of the VLA and E2E modules, thereby integrating their complementary strengths. The experiment on the ICCV 2025 Autonomous Grand Challenge leaderboard shows that DiffVLA++ achieves EPDMS of 49.12.
>
---
#### [replaced 018] Cryo-RL: automating prostate cancer cryoablation planning with reinforcement learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04886v3](http://arxiv.org/pdf/2509.04886v3)**

> **作者:** Trixia Simangan; Ahmed Nadeem Abbasi; Yipeng Hu; Shaheer U. Saeed
>
> **备注:** Accepted at MICAD (Medical Imaging and Computer-Aided Diagnosis) 2025
>
> **摘要:** Cryoablation is a minimally invasive localised treatment for prostate cancer that destroys malignant tissue during de-freezing, while sparing surrounding healthy structures. Its success depends on accurate preoperative planning of cryoprobe placements to fully cover the tumour and avoid critical anatomy. This planning is currently manual, expertise-dependent, and time-consuming, leading to variability in treatment quality and limited scalability. In this work, we introduce Cryo-RL, a reinforcement learning framework that models cryoablation planning as a Markov decision process and learns an optimal policy for cryoprobe placement. Within a simulated environment that models clinical constraints and stochastic intraoperative variability, an agent sequentially selects cryoprobe positions and ice sphere diameters. Guided by a reward function based on tumour coverage, this agent learns a cryoablation strategy that leads to optimal cryoprobe placements without the need for any manually-designed plans. Evaluated on 583 retrospective prostate cancer cases, Cryo-RL achieved over 8 percentage-point Dice improvements compared with the best automated baselines, based on geometric optimisation, and matched human expert performance while requiring substantially less planning time. These results highlight the potential of reinforcement learning to deliver clinically viable, reproducible, and efficient cryoablation plans.
>
---
#### [replaced 019] Increasing the Utility of Synthetic Images through Chamfer Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10631v2](http://arxiv.org/pdf/2508.10631v2)**

> **作者:** Nicola Dall'Asen; Xiaofeng Zhang; Reyhane Askari Hemmat; Melissa Hall; Jakob Verbeek; Adriana Romero-Soriano; Michal Drozdzal
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Conditional image generative models hold considerable promise to produce infinite amounts of synthetic training data. Yet, recent progress in generation quality has come at the expense of generation diversity, limiting the utility of these models as a source of synthetic training data. Although guidance-based approaches have been introduced to improve the utility of generated data by focusing on quality or diversity, the (implicit or explicit) utility functions oftentimes disregard the potential distribution shift between synthetic and real data. In this work, we introduce Chamfer Guidance: a training-free guidance approach which leverages a handful of real exemplar images to characterize the quality and diversity of synthetic data. We show that by leveraging the proposed Chamfer Guidance, we can boost the diversity of the generations w.r.t. a dataset of real images while maintaining or improving the generation quality on ImageNet-1k and standard geo-diversity benchmarks. Our approach achieves state-of-the-art few-shot performance with as little as 2 exemplar real images, obtaining 96.4% in terms of precision, and 86.4% in terms of distributional coverage, which increase to 97.5% and 92.7%, respectively, when using 32 real images. We showcase the benefits of the Chamfer Guidance generation by training downstream image classifiers on synthetic data, achieving accuracy boost of up to 15% for in-distribution over the baselines, and up to 16% in out-of-distribution. Furthermore, our approach does not require using the unconditional model, and thus obtains a 31% reduction in FLOPs w.r.t. classifier-free-guidance-based approaches at sampling time.
>
---
#### [replaced 020] LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2510.17305v2](http://arxiv.org/pdf/2510.17305v2)**

> **作者:** ZhaoYang Han; Qihan Lin; Hao Liang; Bowen Chen; Zhou Liu; Wentao Zhang
>
> **备注:** Submitted to ARR Rolling Review
>
> **摘要:** We introduce \textbf{LongInsightBench}, the first benchmark designed to assess models' ability to understand long videos, with a focus on human language, viewpoints, actions, and other contextual elements, while integrating \textbf{visual, audio, and text} modalities. Our benchmark excels in three key areas: \textbf{a) Long-Duration, Information-Dense Videos:} We carefully select approximately 1,000 videos from open-source datasets FineVideo based on duration limit and the information density of both visual and audio modalities, focusing on content like lectures, interviews, and vlogs, which contain rich language elements. \textbf{b) Diverse and Challenging Task Scenarios:} We have designed six challenging task scenarios, including both Intra-Event and Inter-Event Tasks. \textbf{c) Rigorous and Comprehensive Quality Assurance Pipelines:} We have developed a three-step, semi-automated data quality assurance pipeline to ensure the difficulty and validity of the synthesized questions and answer options. Based on LongInsightBench, we designed a series of experiments. Experimental results shows that Omni-modal models(OLMs) still face challenge in tasks requiring precise temporal localization (T-Loc) and long-range causal inference (CE-Caus). Extended experiments reveal the information loss and processing bias in multi-modal fusion of OLMs. Our dataset and code is available at https://anonymous.4open.science/r/LongInsightBench-910F/.
>
---
#### [replaced 021] Moving Object Detection from Moving Camera Using Focus of Expansion Likelihood and Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13628v2](http://arxiv.org/pdf/2507.13628v2)**

> **作者:** Masahiro Ogawa; Qi An; Atsushi Yamashita
>
> **备注:** 8 pages, 15 figures, RA-L submission
>
> **摘要:** Separating moving and static objects from a moving camera viewpoint is essential for 3D reconstruction, autonomous navigation, and scene understanding in robotics. Existing approaches often rely primarily on optical flow, which struggles to detect moving objects in complex, structured scenes involving camera motion. To address this limitation, we propose Focus of Expansion Likelihood and Segmentation (FoELS), a method based on the core idea of integrating both optical flow and texture information. FoELS computes the focus of expansion (FoE) from optical flow and derives an initial motion likelihood from the outliers of the FoE computation. This likelihood is then fused with a segmentation-based prior to estimate the final moving probability. The method effectively handles challenges including complex structured scenes, rotational camera motion, and parallel motion. Comprehensive evaluations on the DAVIS 2016 dataset and real-world traffic videos demonstrate its effectiveness and state-of-the-art performance.
>
---
#### [replaced 022] Learning to See and Act: Task-Aware View Planning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05186v2](http://arxiv.org/pdf/2508.05186v2)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Weixing Chen; Ziliang Chen; Mingtong Dai; Yongsen Zheng; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 14 pages, 8 figures, project page: https://hcplab-sysu.github.io/TAVP
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robotic manipulation commonly rely on static viewpoints and shared visual encoders, which limit 3D perception and cause task interference, hindering robustness and generalization. In this work, we propose Task-Aware View Planning (TAVP), a framework designed to overcome these challenges by integrating active view planning with task-specific representation learning. TAVP employs an efficient exploration policy, accelerated by a novel pseudo-environment, to actively acquire informative views. Furthermore, we introduce a Mixture-of-Experts (MoE) visual encoder to disentangle features across different tasks, boosting both representation fidelity and task generalization. By learning to see the world in a task-aware way, TAVP generates more complete and discriminative visual representations, demonstrating significantly enhanced action prediction across a wide array of manipulation challenges. Extensive experiments on RLBench tasks show that our proposed TAVP model achieves superior performance over state-of-the-art fixed-view approaches. Visual results and code are provided at: https://hcplab-sysu.github.io/TAVP.
>
---
#### [replaced 023] RODS: Robust Optimization Inspired Diffusion Sampling for Detecting and Reducing Hallucination in Generative Models
- **分类: cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2507.12201v2](http://arxiv.org/pdf/2507.12201v2)**

> **作者:** Yiqi Tian; Pengfei Jin; Mingze Yuan; Na Li; Bo Zeng; Quanzheng Li
>
> **摘要:** Diffusion models have achieved state-of-the-art performance in generative modeling, yet their sampling procedures remain vulnerable to hallucinations-often stemming from inaccuracies in score approximation. In this work, we reinterpret diffusion sampling through the lens of optimization and introduce RODS (Robust Optimization-inspired Diffusion Sampler), a novel method that detects and corrects high-risk sampling steps using geometric cues from the loss landscape. RODS enforces smoother sampling trajectories and adaptively adjusts perturbations, reducing hallucinations without retraining and at minimal additional inference cost. Experiments on AFHQv2, FFHQ, and 11k-hands demonstrate that RODS maintains comparable image quality and preserves generation diversity. More importantly, it improves both sampling fidelity and robustness, detecting over 70% of hallucinated samples and correcting more than 25%, all while avoiding the introduction of new artifacts. We release our code at https://github.com/Yiqi-Verna-Tian/RODS.
>
---
#### [replaced 024] REOrdering Patches Improves Vision Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23751v2](http://arxiv.org/pdf/2505.23751v2)**

> **作者:** Declan Kutscher; David M. Chan; Yutong Bai; Trevor Darrell; Ritwik Gupta
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Sequence models such as transformers require inputs to be represented as one-dimensional sequences. In vision, this typically involves flattening images using a fixed row-major (raster-scan) order. While full self-attention is permutation-equivariant, modern long-sequence transformers increasingly rely on architectural approximations that break this invariance and introduce sensitivity to patch ordering. We show that patch order significantly affects model performance in such settings, with simple alternatives like column-major or Hilbert curves yielding notable accuracy shifts. Motivated by this, we propose REOrder, a two-stage framework for discovering task-optimal patch orderings. First, we derive an information-theoretic prior by evaluating the compressibility of various patch sequences. Then, we learn a policy over permutations by optimizing a Plackett-Luce policy using REINFORCE. This approach enables efficient learning in a combinatorial permutation space. REOrder improves top-1 accuracy over row-major ordering on ImageNet-1K by up to 3.01% and Functional Map of the World by 13.35%.
>
---
#### [replaced 025] When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.10255v2](http://arxiv.org/pdf/2405.10255v2)**

> **作者:** Xianzheng Ma; Brandon Smart; Yash Bhalgat; Shuai Chen; Xinghui Li; Jian Ding; Jindong Gu; Dave Zhenyu Chen; Songyou Peng; Jia-Wang Bian; Philip H Torr; Marc Pollefeys; Matthias Nießner; Ian D Reid; Angel X. Chang; Iro Laina; Victor Adrian Prisacariu
>
> **备注:** 2nd version update to Jun.2025
>
> **摘要:** As large language models (LLMs) evolve, their integration with 3D spatial data (3D-LLMs) has seen rapid progress, offering unprecedented capabilities for understanding and interacting with physical spaces. This survey provides a comprehensive overview of the methodologies enabling LLMs to process, understand, and generate 3D data. Highlighting the unique advantages of LLMs, such as in-context learning, step-by-step reasoning, open-vocabulary capabilities, and extensive world knowledge, we underscore their potential to significantly advance spatial comprehension and interaction within embodied Artificial Intelligence (AI) systems. Our investigation spans various 3D data representations, from point clouds to Neural Radiance Fields (NeRFs). It examines their integration with LLMs for tasks such as 3D scene understanding, captioning, question-answering, and dialogue, as well as LLM-based agents for spatial reasoning, planning, and navigation. The paper also includes a brief review of other methods that integrate 3D and language. The meta-analysis presented in this paper reveals significant progress yet underscores the necessity for novel approaches to harness the full potential of 3D-LLMs. Hence, with this paper, we aim to chart a course for future research that explores and expands the capabilities of 3D-LLMs in understanding and interacting with the complex 3D world. To support this survey, we have established a project page where papers related to our topic are organized and listed: https://github.com/ActiveVisionLab/Awesome-LLM-3D.
>
---
#### [replaced 026] Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01480v2](http://arxiv.org/pdf/2506.01480v2)**

> **作者:** Kaihang Pan; Yang Wu; Wendong Bu; Kai Shen; Juncheng Li; Yingting Wang; Yunfei Li; Siliang Tang; Jun Xiao; Fei Wu; Hang Zhao; Yueting Zhuang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent endeavors in Multimodal Large Language Models (MLLMs) aim to unify visual comprehension and generation. However, these two capabilities remain largely independent, as if they are two separate functions encapsulated within the same model. Consequently, visual comprehension does not enhance visual generation, and the reasoning mechanisms of LLMs have not been fully integrated to revolutionize image generation. In this paper, we propose to enable the collaborative co-evolution of visual comprehension and generation, advancing image generation into an iterative introspective process. We introduce a two-stage training approach: supervised fine-tuning teaches the MLLM with the foundational ability to generate genuine CoT for visual generation, while reinforcement learning activates its full potential via an exploration-exploitation trade-off. Ultimately, we unlock the Aha moment in visual generation, advancing MLLMs from text-to-image tasks to unified image generation. Extensive experiments demonstrate that our model not only excels in text-to-image generation and image editing, but also functions as a superior image semantic evaluator with enhanced visual comprehension capabilities. Project Page: https://janus-pro-r1.github.io.
>
---
#### [replaced 027] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01345v2](http://arxiv.org/pdf/2508.01345v2)**

> **作者:** Rongzhen Zhao; Jian Li; Juho Kannala; Joni Pajarinen
>
> **摘要:** Unsupervised video Object-Centric Learning (OCL) is promising as it enables object-level scene representation and dynamics modeling as we humans do. Mainstream video OCL methods adopt a recurrent architecture: An aggregator aggregates current video frame into object features, termed slots, under some queries; A transitioner transits current slots to queries for the next frame. This is an effective architecture but all existing implementations both (\textit{i1}) neglect to incorporate next frame features, the most informative source for query prediction, and (\textit{i2}) fail to learn transition dynamics, the knowledge essential for query prediction. To address these issues, we propose Random Slot-Feature pair for learning Query prediction (RandSF.Q): (\textit{t1}) We design a new transitioner to incorporate both slots and features, which provides more information for query prediction; (\textit{t2}) We train the transitioner to predict queries from slot-feature pairs randomly sampled from available recurrences, which drives it to learn transition dynamics. Experiments on scene representation demonstrate that our method surpass existing video OCL methods significantly, e.g., up to 10 points on object discovery, setting new state-of-the-art. Such superiority also benefits downstream tasks like dynamics modeling. Our core source code and training logs are available on https://github.com/Genera1Z/RandSF.Q.
>
---
#### [replaced 028] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.09049v2](http://arxiv.org/pdf/2506.09049v2)**

> **作者:** Li Kang; Xiufeng Song; Heng Zhou; Yiran Qin; Jie Yang; Xiaohong Liu; Philip Torr; Lei Bai; Zhenfei Yin
>
> **备注:** Project page: https://faceong.github.io/VIKI-R/
>
> **摘要:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
>
---
#### [replaced 029] Global Prompt Refinement with Non-Interfering Attention Masking for One-Shot Federated Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.22700v2](http://arxiv.org/pdf/2509.22700v2)**

> **作者:** Zhuang Qi; Pan Yu; Lei Meng; Sijin Zhou; Han Yu; Xiaoxiao Li; Xiangxu Meng
>
> **备注:** NeurIPS'25 accepted
>
> **摘要:** Federated Prompt Learning (FPL) enables communication-efficient adaptation by tuning lightweight prompts on top of frozen pre-trained models. Existing FPL methods typically rely on global information, which is only available after the second training round, to facilitate collaboration among client models. Therefore, they are inherently dependent on multi-round communication to fully exhibit their strengths. Moreover, existing one-shot federated learning methods typically focus on fitting seen tasks, but lack cross-task generalization. To bridge this gap, we propose the Global Prompt Refinement with Non-Interfering Attention Masking (GPR-NIAM) method for one-shot FPL. The core idea is to design a masking mechanism that restricts excessive interaction between the original text embeddings and the learnable prompt embeddings. GPR-NIAM achieves this through the collaboration of two key modules. Firstly, the attention isolation module suppresses attention from the learnable prompt tokens to the original text tokens, and reweights the reverse attention which preserves generalization across tasks. Secondly, the cross-silo collaborative refinement module integrates decentralized visual knowledge into a unified base and calibrates the global prompt through multi-source cross-modal knowledge alignment, further mitigating the inconsistency caused by data heterogeneity. Extensive experiments conducted on ten benchmark datasets under two tasks show that GPR-NIAM outperforms eight state-of-the-art methods in both class-level and domain-level generalization.
>
---
#### [replaced 030] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.16714v2](http://arxiv.org/pdf/2510.16714v2)**

> **作者:** Xiongkun Linghu; Jiangyong Huang; Ziyu Zhu; Baoxiong Jia; Siyuan Huang
>
> **备注:** Project page: https://scenecot.github.io/
>
> **摘要:** Existing research on 3D Large Language Models (LLMs) still struggles to achieve grounded question-answering, primarily due to the under-exploration of the mech- anism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a grounded Chain-of- Thought reasoning method in 3D scenes (SCENECOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a method, we develop SCENECOT-185K, the first large-scale grounded CoT reasoning dataset, consisting of 185K high-quality instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves strong performance with high grounding-QA coherence. To the best of our knowledge, this is the first successful application of CoT reasoning to 3D scene understanding, enabling step-by-step human-like reasoning and showing potential for extension to broader 3D scene understanding scenarios.
>
---
#### [replaced 031] The Impact of Coreset Selection on Spurious Correlations and Group Robustness
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11690v2](http://arxiv.org/pdf/2507.11690v2)**

> **作者:** Amaya Dharmasiri; William Yang; Polina Kirichenko; Lydia Liu; Olga Russakovsky
>
> **备注:** 10 pages, 9 additional pages for Appendix
>
> **摘要:** Coreset selection methods have shown promise in reducing the training data size while maintaining model performance for data-efficient machine learning. However, as many datasets suffer from biases that cause models to learn spurious correlations instead of causal features, it is important to understand whether and how dataset reduction methods may perpetuate, amplify, or mitigate these biases. In this work, we conduct the first comprehensive analysis of the implications of data selection on the spurious bias levels of the selected coresets and the robustness of downstream models trained on them. We use an extensive experimental setting spanning ten different spurious correlations benchmarks, five score metrics to characterize sample importance/ difficulty, and five data selection policies across a broad range of coreset sizes. Thereby, we unravel a series of nontrivial nuances in interactions between sample difficulty and bias alignment, as well as dataset bias and resultant model robustness. For example, we find that selecting coresets using embedding-based sample characterization scores runs a comparatively lower risk of inadvertently exacerbating bias than selecting using characterizations based on learning dynamics. Most importantly, our analysis reveals that although some coreset selection methods could achieve lower bias levels by prioritizing difficult samples, they do not reliably guarantee downstream robustness.
>
---
#### [replaced 032] PICABench: How Far Are We from Physically Realistic Image Editing?
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.17681v2](http://arxiv.org/pdf/2510.17681v2)**

> **作者:** Yuandong Pu; Le Zhuo; Songhao Han; Jinbo Xing; Kaiwen Zhu; Shuo Cao; Bin Fu; Si Liu; Hongsheng Li; Yu Qiao; Wenlong Zhang; Xi Chen; Yihao Liu
>
> **摘要:** Image editing has achieved remarkable progress recently. Modern editing models could already follow complex instructions to manipulate the original content. However, beyond completing the editing instructions, the accompanying physical effects are the key to the generation realism. For example, removing an object should also remove its shadow, reflections, and interactions with nearby objects. Unfortunately, existing models and benchmarks mainly focus on instruction completion but overlook these physical effects. So, at this moment, how far are we from physically realistic image editing? To answer this, we introduce PICABench, which systematically evaluates physical realism across eight sub-dimension (spanning optics, mechanics, and state transitions) for most of the common editing operations (add, remove, attribute change, etc.). We further propose the PICAEval, a reliable evaluation protocol that uses VLM-as-a-judge with per-case, region-level human annotations and questions. Beyond benchmarking, we also explore effective solutions by learning physics from videos and construct a training dataset PICA-100K. After evaluating most of the mainstream models, we observe that physical realism remains a challenging problem with large rooms to explore. We hope that our benchmark and proposed solutions can serve as a foundation for future work moving from naive content editing toward physically consistent realism.
>
---
#### [replaced 033] 3D Audio-Visual Segmentation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02236v2](http://arxiv.org/pdf/2411.02236v2)**

> **作者:** Artem Sokolov; Swapnil Bhosale; Xiatian Zhu
>
> **备注:** Accepted at the NeurIPS 2024 Workshop on Audio Imagination; this version updates the project page link
>
> **摘要:** Recognizing the sounding objects in scenes is a longstanding objective in embodied AI, with diverse applications in robotics and AR/VR/MR. To that end, Audio-Visual Segmentation (AVS), taking as condition an audio signal to identify the masks of the target sounding objects in an input image with synchronous camera and microphone sensors, has been recently advanced. However, this paradigm is still insufficient for real-world operation, as the mapping from 2D images to 3D scenes is missing. To address this fundamental limitation, we introduce a novel research problem, 3D Audio-Visual Segmentation, extending the existing AVS to the 3D output space. This problem poses more challenges due to variations in camera extrinsics, audio scattering, occlusions, and diverse acoustics across sounding object categories. To facilitate this research, we create the very first simulation based benchmark, 3DAVS-S34-O7, providing photorealistic 3D scene environments with grounded spatial audio under single-instance and multi-instance settings, across 34 scenes and 7 object categories. This is made possible by re-purposing the Habitat simulator to generate comprehensive annotations of sounding object locations and corresponding 3D masks. Subsequently, we propose a new approach, EchoSegnet, characterized by integrating the ready-to-use knowledge from pretrained 2D audio-visual foundation models synergistically with 3D visual scene representation through spatial audio-aware mask alignment and refinement. Extensive experiments demonstrate that EchoSegnet can effectively segment sounding objects in 3D space on our new benchmark, representing a significant advancement in the field of embodied AI. Project page: https://x-up-lab.github.io/research/3d-audio-visual-segmentation/
>
---
#### [replaced 034] PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16505v2](http://arxiv.org/pdf/2510.16505v2)**

> **作者:** Lukas Selch; Yufang Hou; M. Jehanzeb Mirza; Sivan Doveh; James Glass; Rogerio Feris; Wei Lin
>
> **摘要:** Large Multimodal Models (LMMs) are increasingly applied to scientific research, yet it remains unclear whether they can reliably understand and reason over the multimodal complexity of papers. A central challenge lies in detecting and resolving inconsistencies across text, figures, tables, and equations, issues that are often subtle, domain-specific, and ultimately undermine clarity, reproducibility, and trust. Existing benchmarks overlook this issue, either isolating single modalities or relying on synthetic errors that fail to capture real-world complexity. We introduce PRISMM-Bench (Peer-Review-sourced Inconsistency Set for Multimodal Models), the first benchmark grounded in real reviewer-flagged inconsistencies in scientific papers. Through a multi-stage pipeline of review mining, LLM-assisted filtering and human verification, we curate 262 inconsistencies from 242 papers. Based on this set, we design three tasks, namely inconsistency identification, remedy and pair matching, which assess a model's capacity to detect, correct, and reason over inconsistencies across different modalities. Furthermore, to address the notorious problem of choice-only shortcuts in multiple-choice evaluation, where models exploit answer patterns without truly understanding the question, we further introduce structured JSON-based answer representations that minimize linguistic biases by reducing reliance on superficial stylistic cues. We benchmark 21 leading LMMs, including large open-weight models (GLM-4.5V 106B, InternVL3 78B) and proprietary models (Gemini 2.5 Pro, GPT-5 with high reasoning). Results reveal strikingly low performance (26.1-54.2%), underscoring the challenge of multimodal scientific reasoning and motivating progress towards trustworthy scientific assistants.
>
---
#### [replaced 035] Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13795v2](http://arxiv.org/pdf/2510.13795v2)**

> **作者:** Yi Zhang; Bolin Ni; Xin-Sheng Chen; Heng-Rui Zhang; Yongming Rao; Houwen Peng; Qinglin Lu; Han Hu; Meng-Hao Guo; Shi-Min Hu
>
> **备注:** homepage: https://open-bee.github.io/
>
> **摘要:** Fully open multimodal large language models (MLLMs) currently lag behind proprietary counterparts, primarily due to a significant gap in data quality for supervised fine-tuning (SFT). Existing open-source datasets are often plagued by widespread noise and a critical deficit in complex reasoning data, such as Chain-of-Thought (CoT), which hinders the development of advanced model capabilities. Addressing these challenges, our work makes three primary contributions. First, we introduce Honey-Data-15M, a new SFT dataset comprising approximately 15 million QA pairs, processed through multiple cleaning techniques and enhanced with a novel dual-level (short and long) CoT enrichment strategy. Second, we introduce HoneyPipe, the data curation pipeline, and its underlying framework DataStudio, providing the community with a transparent and adaptable methodology for data curation that moves beyond static dataset releases. Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B model on Honey-Data-15M. Experiments show that Bee-8B establishes a new state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is competitive with, and in some cases surpasses, recent semi-open models such as InternVL3.5-8B. Our work delivers to the community a suite of foundational resources, including: the Honey-Data-15M corpus; the full-stack suite comprising HoneyPipe and DataStudio; training recipes; an evaluation harness; and the model weights. This effort demonstrates that a principled focus on data quality is a key pathway to developing fully open MLLMs that are highly competitive with their semi-open counterparts.
>
---
#### [replaced 036] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17098v4](http://arxiv.org/pdf/2505.17098v4)**

> **作者:** Yanshu Li; Jianjiang Yang; Tian Yun; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** EMNLP2025 Main, 28 pages, 11 figures, 19 tables
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input ICL sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures ICL sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a novel and valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [replaced 037] COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification
- **分类: cs.CV; cs.AI; 68T07; I.1.2; I.4.0; I.4.10; I.4.0**

- **链接: [http://arxiv.org/pdf/2505.18315v2](http://arxiv.org/pdf/2505.18315v2)**

> **作者:** Mariano Rivera; Angello Hoyos
>
> **备注:** 15 pages, 13 figures
>
> **摘要:** We introduce CoLoRA (Convolutional Low-Rank Adaptation), a parameter-efficient fine-tuning method for convolutional neural networks (CNNs). CoLoRA extends LoRA to convolutional layers by decomposing kernel updates into lightweight depthwise and pointwise components.This design reduces the number of trainable parameters to 0.2 compared to conventional fine-tuning, preserves the original model size, and allows merging updates into the pretrained weights after each epoch, keeping inference complexity unchanged. On OCTMNISTv2, CoLoRA applied to VGG16 and ResNet50 achieves up to 1 percent accuracy and 0.013 AUC improvements over strong baselines (Vision Transformers, state-space, and Kolmogorov Arnold models) while reducing per-epoch training time by nearly 20 percent. Results indicate that CoLoRA provides a stable and effective alternative to full fine-tuning for medical image classification.
>
---
#### [replaced 038] Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09201v2](http://arxiv.org/pdf/2508.09201v2)**

> **作者:** Shuang Liang; Zhihao Xu; Jialing Tao; Hui Xue; Xiting Wang
>
> **备注:** 16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident
>
> **摘要:** Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.
>
---
#### [replaced 039] VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14460v2](http://arxiv.org/pdf/2505.14460v2)**

> **作者:** Tianhe Wu; Jian Zou; Jie Liang; Lei Zhang; Kede Ma
>
> **摘要:** DeepSeek-R1 has demonstrated remarkable effectiveness in incentivizing reasoning and generalization capabilities of large language models (LLMs) through reinforcement learning. Nevertheless, the potential of reasoning-induced computation has not been thoroughly explored in the context of image quality assessment (IQA), a task depending critically on visual reasoning. In this paper, we introduce VisualQuality-R1, a reasoning-induced no-reference IQA (NR-IQA) model, and we train it with reinforcement learning to rank, a learning algorithm tailored to the intrinsically relative nature of visual quality. Specifically, for a pair of images, we employ group relative policy optimization to generate multiple quality scores for each image. These estimates are used to compute comparative probabilities of one image having higher quality than the other under the Thurstone model. Rewards for each quality estimate are defined using continuous fidelity measures rather than discretized binary labels. Extensive experiments show that the proposed VisualQuality-R1 consistently outperforms discriminative deep learning-based NR-IQA models as well as a recent reasoning-induced quality regression method. Moreover, VisualQuality-R1 is capable of generating contextually rich, human-aligned quality descriptions, and supports multi-dataset training without requiring perceptual scale realignment. These features make VisualQuality-R1 especially well-suited for reliably measuring progress in a wide range of image processing tasks like super-resolution and image generation.
>
---
#### [replaced 040] Deep Learning in Palmprint Recognition-A Comprehensive Survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.01166v2](http://arxiv.org/pdf/2501.01166v2)**

> **作者:** Chengrui Gao; Ziyuan Yang; Wei Jia; Lu Leng; Bob Zhang; Andrew Beng Jin Teoh
>
> **备注:** Palmprint recognition, biometrics, deep learning, feature extraction, recognition tasks
>
> **摘要:** Palmprint recognition has emerged as a prominent biometric technology, widely applied in diverse scenarios. Traditional handcrafted methods for palmprint recognition often fall short in representation capability, as they heavily depend on researchers' prior knowledge. Deep learning (DL) has been introduced to address this limitation, leveraging its remarkable successes across various domains. While existing surveys focus narrowly on specific tasks within palmprint recognition-often grounded in traditional methodologies-there remains a significant gap in comprehensive research exploring DL-based approaches across all facets of palmprint recognition. This paper bridges that gap by thoroughly reviewing recent advancements in DL-powered palmprint recognition. The paper systematically examines progress across key tasks, including region-of-interest segmentation, feature extraction, and security/privacy-oriented challenges. Beyond highlighting these advancements, the paper identifies current challenges and uncovers promising opportunities for future research. By consolidating state-of-the-art progress, this review serves as a valuable resource for researchers, enabling them to stay abreast of cutting-edge technologies and drive innovation in palmprint recognition.
>
---
#### [replaced 041] Polyline Path Masked Attention for Vision Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15940v2](http://arxiv.org/pdf/2506.15940v2)**

> **作者:** Zhongchen Zhao; Chaodong Xiao; Hui Lin; Qi Xie; Lei Zhang; Deyu Meng
>
> **摘要:** Global dependency modeling and spatial position modeling are two core issues of the foundational architecture design in current deep learning frameworks. Recently, Vision Transformers (ViTs) have achieved remarkable success in computer vision, leveraging the powerful global dependency modeling capability of the self-attention mechanism. Furthermore, Mamba2 has demonstrated its significant potential in natural language processing tasks by explicitly modeling the spatial adjacency prior through the structured mask. In this paper, we propose Polyline Path Masked Attention (PPMA) that integrates the self-attention mechanism of ViTs with an enhanced structured mask of Mamba2, harnessing the complementary strengths of both architectures. Specifically, we first ameliorate the traditional structured mask of Mamba2 by introducing a 2D polyline path scanning strategy and derive its corresponding structured mask, polyline path mask, which better preserves the adjacency relationships among image tokens. Notably, we conduct a thorough theoretical analysis on the structural characteristics of the proposed polyline path mask and design an efficient algorithm for the computation of the polyline path mask. Next, we embed the polyline path mask into the self-attention mechanism of ViTs, enabling explicit modeling of spatial adjacency prior. Extensive experiments on standard benchmarks, including image classification, object detection, and segmentation, demonstrate that our model outperforms previous state-of-the-art approaches based on both state-space models and Transformers. For example, our proposed PPMA-T/S/B models achieve 48.7%/51.1%/52.3% mIoU on the ADE20K semantic segmentation task, surpassing RMT-T/S/B by 0.7%/1.3%/0.3%, respectively. Code is available at https://github.com/zhongchenzhao/PPMA.
>
---
#### [replaced 042] GeoArena: An Open Platform for Benchmarking Large Vision-language Models on WorldWide Image Geolocalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04334v3](http://arxiv.org/pdf/2509.04334v3)**

> **作者:** Pengyue Jia; Yingyi Zhang; Xiangyu Zhao; Sharon Li
>
> **摘要:** Image geolocalization aims to predict the geographic location of images captured anywhere on Earth, but its global nature presents significant challenges. Current evaluation methodologies suffer from two major limitations. First, data leakage: advanced approaches often rely on large vision-language models (LVLMs) to predict image locations, yet these models are frequently pretrained on the test datasets, compromising the accuracy of evaluating a model's actual geolocalization capability. Second, existing metrics primarily rely on exact geographic coordinates to assess predictions, which not only neglects the reasoning process but also raises privacy concerns when user-level location data is required. To address these issues, we propose GeoArena, a first open platform for evaluating LVLMs on worldwide image geolocalization tasks, offering true in-the-wild and human-centered benchmarking. GeoArena enables users to upload in-the-wild images for a more diverse evaluation corpus, and it leverages pairwise human judgments to determine which model output better aligns with human expectations. Our platform has been deployed online for two months, during which we collected over thousands voting records. Based on this data, we conduct a detailed analysis and establish a leaderboard of different LVLMs on the image geolocalization task. GeoArena has been open-sourced to support future research.
>
---
#### [replaced 043] UniVideo: Unified Understanding, Generation, and Editing for Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.08377v2](http://arxiv.org/pdf/2510.08377v2)**

> **作者:** Cong Wei; Quande Liu; Zixuan Ye; Qiulin Wang; Xintao Wang; Pengfei Wan; Kun Gai; Wenhu Chen
>
> **备注:** Project Website https://congwei1230.github.io/UniVideo/
>
> **摘要:** Unified multimodal models have shown promising results in multimodal content generation and editing but remain largely limited to the image domain. In this work, we present UniVideo, a versatile framework that extends unified modeling to the video domain. UniVideo adopts a dual-stream design, combining a Multimodal Large Language Model (MLLM) for instruction understanding with a Multimodal DiT (MMDiT) for video generation. This design enables accurate interpretation of complex multimodal instructions while preserving visual consistency. Built on this architecture, UniVideo unifies diverse video generation and editing tasks under a single multimodal instruction paradigm and is jointly trained across them. Extensive experiments demonstrate that UniVideo matches or surpasses state-of-the-art task-specific baselines in text/image-to-video generation, in-context video generation and in-context video editing. Notably, the unified design of UniVideo enables two forms of generalization. First, UniVideo supports task composition, such as combining editing with style transfer, by integrating multiple capabilities within a single instruction. Second, even without explicit training on free-form video editing, UniVideo transfers its editing capability from large-scale image editing data to this setting, handling unseen instructions such as green-screening characters or changing materials within a video. Beyond these core capabilities, UniVideo also supports visual-prompt-based video generation, where the MLLM interprets visual prompts and guides the MMDiT during synthesis. To foster future research, we will release our model and code.
>
---
#### [replaced 044] Latent Diffusion Model without Variational Autoencoder
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.15301v3](http://arxiv.org/pdf/2510.15301v3)**

> **作者:** Minglei Shi; Haolin Wang; Wenzhao Zheng; Ziyang Yuan; Xiaoshi Wu; Xintao Wang; Pengfei Wan; Jie Zhou; Jiwen Lu
>
> **摘要:** Recent progress in diffusion-based visual generation has largely relied on latent diffusion models with variational autoencoders (VAEs). While effective for high-fidelity synthesis, this VAE+diffusion paradigm suffers from limited training efficiency, slow inference, and poor transferability to broader vision tasks. These issues stem from a key limitation of VAE latent spaces: the lack of clear semantic separation and strong discriminative structure. Our analysis confirms that these properties are crucial not only for perception and understanding tasks, but also for the stable and efficient training of latent diffusion models. Motivated by this insight, we introduce SVG, a novel latent diffusion model without variational autoencoders, which leverages self-supervised representations for visual generation. SVG constructs a feature space with clear semantic discriminability by leveraging frozen DINO features, while a lightweight residual branch captures fine-grained details for high-fidelity reconstruction. Diffusion models are trained directly on this semantically structured latent space to facilitate more efficient learning. As a result, SVG enables accelerated diffusion training, supports few-step sampling, and improves generative quality. Experimental results further show that SVG preserves the semantic and discriminative capabilities of the underlying self-supervised representations, providing a principled pathway toward task-general, high-quality visual representations. Code and interpretations are available at https://howlin-wang.github.io/svg/.
>
---
#### [replaced 045] Pose-free 3D Gaussian splatting via shape-ray estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22978v3](http://arxiv.org/pdf/2505.22978v3)**

> **作者:** Youngju Na; Taeyeon Kim; Jumin Lee; Kyu Beom Han; Woo Jae Kim; Sung-eui Yoon
>
> **备注:** ICIP 2025 (Best Student Paper Award) Code available at: https://github.com/youngju-na/SHARE
>
> **摘要:** While generalizable 3D Gaussian splatting enables efficient, high-quality rendering of unseen scenes, it heavily depends on precise camera poses for accurate geometry. In real-world scenarios, obtaining accurate poses is challenging, leading to noisy pose estimates and geometric misalignments. To address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting framework that overcomes these ambiguities by joint shape and camera rays estimation. Instead of relying on explicit 3D transformations, SHARE builds a pose-aware canonical volume representation that seamlessly integrates multi-view information, reducing misalignment caused by inaccurate pose estimates. Additionally, anchor-aligned Gaussian prediction enhances scene reconstruction by refining local geometry around coarse anchors, allowing for more precise Gaussian placement. Extensive experiments on diverse real-world datasets show that our method achieves robust performance in pose-free generalizable Gaussian splatting. Code is avilable at https://github.com/youngju-na/SHARE
>
---
#### [replaced 046] Mask Image Watermarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12739v3](http://arxiv.org/pdf/2504.12739v3)**

> **作者:** Runyi Hu; Jie Zhang; Shiqian Zhao; Nils Lukas; Jiwei Li; Qing Guo; Han Qiu; Tianwei Zhang
>
> **备注:** Neural Information Processing Systems (NeurIPS) 2025
>
> **摘要:** We present MaskWM, a simple, efficient, and flexible framework for image watermarking. MaskWM has two variants: (1) MaskWM-D, which supports global watermark embedding, watermark localization, and local watermark extraction for applications such as tamper detection; (2) MaskWM-ED, which focuses on local watermark embedding and extraction, offering enhanced robustness in small regions to support fine-grined image protection. MaskWM-D builds on the classical encoder-distortion layer-decoder training paradigm. In MaskWM-D, we introduce a simple masking mechanism during the decoding stage that enables both global and local watermark extraction. During training, the decoder is guided by various types of masks applied to watermarked images before extraction, helping it learn to localize watermarks and extract them from the corresponding local areas. MaskWM-ED extends this design by incorporating the mask into the encoding stage as well, guiding the encoder to embed the watermark in designated local regions, which improves robustness under regional attacks. Extensive experiments show that MaskWM achieves state-of-the-art performance in global and local watermark extraction, watermark localization, and multi-watermark embedding. It outperforms all existing baselines, including the recent leading model WAM for local watermarking, while preserving high visual quality of the watermarked images. In addition, MaskWM is highly efficient and adaptable. It requires only 20 hours of training on a single A6000 GPU, achieving 15x computational efficiency compared to WAM. By simply adjusting the distortion layer, MaskWM can be quickly fine-tuned to meet varying robustness requirements.
>
---
#### [replaced 047] DeepDetect: Learning All-in-One Dense Keypoints
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17422v2](http://arxiv.org/pdf/2510.17422v2)**

> **作者:** Shaharyar Ahmed Khan Tareen; Filza Khan Tareen
>
> **备注:** 6 pages, 6 figures, 2 tables, 7 equations
>
> **摘要:** Keypoint detection is the foundation of many computer vision tasks, including image registration, structure-from motion, 3D reconstruction, visual odometry, and SLAM. Traditional detectors (SIFT, SURF, ORB, BRISK, etc.) and learning based methods (SuperPoint, R2D2, LF-Net, D2-Net, etc.) have shown strong performance yet suffer from key limitations: sensitivity to photometric changes, low keypoint density and repeatability, limited adaptability to challenging scenes, and lack of semantic understanding, often failing to prioritize visually important regions. We present DeepDetect, an intelligent, all-in-one, dense keypoint detector that unifies the strengths of classical detectors using deep learning. Firstly, we create ground-truth masks by fusing outputs of 7 keypoint and 2 edge detectors, extracting diverse visual cues from corners and blobs to prominent edges and textures in the images. Afterwards, a lightweight and efficient model: ESPNet, is trained using these masks as labels, enabling DeepDetect to focus semantically on images while producing highly dense keypoints, that are adaptable to diverse and visually degraded conditions. Evaluations on the Oxford Affine Covariant Regions dataset demonstrate that DeepDetect surpasses other detectors in keypoint density, repeatability, and the number of correct matches, achieving maximum values of 0.5143 (average keypoint density), 0.9582 (average repeatability), and 59,003 (correct matches).
>
---
#### [replaced 048] RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08458v2](http://arxiv.org/pdf/2501.08458v2)**

> **作者:** Juntao Jiang; Jiangning Zhang; Weixuan Liu; Muxuan Gao; Xiaobin Hu; Zhucun Xue; Yong Liu; Shuicheng Yan
>
> **摘要:** In recent years, significant advancements have been made in deep learning for medical image segmentation, particularly with convolutional neural networks (CNNs) and transformer models. However, CNNs face limitations in capturing long-range dependencies, while transformers suffer from high computational complexity. To address this, we propose RWKV-UNet, a novel model that integrates the RWKV (Receptance Weighted Key Value) structure into the U-Net architecture. This integration enhances the model's ability to capture long-range dependencies and to improve contextual understanding, which is crucial for accurate medical image segmentation. We build a strong encoder with developed Global-Local Spatial Perception (GLSP) blocks combining CNNs and RWKVs. We also propose a Cross-Channel Mix (CCM) module to improve skip connections with multi-scale feature fusion, achieving global channel information integration. Experiments on 11 benchmark datasets show that the RWKV-UNet achieves state-of-the-art performance on various types of medical image segmentation tasks. Additionally, smaller variants, RWKV-UNet-S and RWKV-UNet-T, balance accuracy and computational efficiency, making them suitable for broader clinical applications.
>
---
#### [replaced 049] Facial Expression-based Parkinson's Disease Severity Diagnosis via Feature Fusion and Adaptive Class Balancing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17373v2](http://arxiv.org/pdf/2510.17373v2)**

> **作者:** Yintao Zhou; Wei Huang; Zhengyu Li; Jing Huang; Meng Pang
>
> **备注:** 3 pages, 2 figures, accepted by MIND 2025
>
> **摘要:** Parkinson's disease (PD) severity diagnosis is crucial for early detecting potential patients and adopting tailored interventions. Diagnosing PD based on facial expression is grounded in PD patients' "masked face" symptom and gains growing interest recently for its convenience and affordability. However, current facial expression-based approaches often rely on single type of expression which can lead to misdiagnosis, and ignore the class imbalance across different PD stages which degrades the prediction performance. Moreover, most existing methods focus on binary classification (i.e., PD / non-PD) rather than diagnosing the severity of PD. To address these issues, we propose a new facial expression-based method for PD severity diagnosis which integrates multiple facial expression features through attention-based feature fusion. Moreover, we mitigate the class imbalance problem via an adaptive class balancing strategy which dynamically adjusts the contribution of training samples based on their class distribution and classification difficulty. Experimental results demonstrate the promising performance of the proposed method for PD severity diagnosis, as well as the efficacy of attention-based feature fusion and adaptive class balancing.
>
---
#### [replaced 050] CrossRay3D: Geometry and Distribution Guidance for Efficient Multimodal 3D Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.15991v2](http://arxiv.org/pdf/2510.15991v2)**

> **作者:** Huiming Yang
>
> **备注:** 13 pages
>
> **摘要:** The sparse cross-modality detector offers more advantages than its counterpart, the Bird's-Eye-View (BEV) detector, particularly in terms of adaptability for downstream tasks and computational cost savings. However, existing sparse detectors overlook the quality of token representation, leaving it with a sub-optimal foreground quality and limited performance. In this paper, we identify that the geometric structure preserved and the class distribution are the key to improving the performance of the sparse detector, and propose a Sparse Selector (SS). The core module of SS is Ray-Aware Supervision (RAS), which preserves rich geometric information during the training stage, and Class-Balanced Supervision, which adaptively reweights the salience of class semantics, ensuring that tokens associated with small objects are retained during token sampling. Thereby, outperforming other sparse multi-modal detectors in the representation of tokens. Additionally, we design Ray Positional Encoding (Ray PE) to address the distribution differences between the LiDAR modality and the image. Finally, we integrate the aforementioned module into an end-to-end sparse multi-modality detector, dubbed CrossRay3D. Experiments show that, on the challenging nuScenes benchmark, CrossRay3D achieves state-of-the-art performance with 72.4 mAP and 74.7 NDS, while running 1.84 faster than other leading methods. Moreover, CrossRay3D demonstrates strong robustness even in scenarios where LiDAR or camera data are partially or entirely missing.
>
---
#### [replaced 051] Visible Yet Unreadable: A Systematic Blind Spot of Vision Language Models Across Writing Systems
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06996v4](http://arxiv.org/pdf/2509.06996v4)**

> **作者:** Jie Zhang; Ting Xu; Gelei Deng; Runyi Hu; Han Qiu; Tianwei Zhang; Qing Guo; Ivor Tsang
>
> **备注:** Agent4Science 2025 Spotlight
>
> **摘要:** Writing is a universal cultural technology that reuses vision for symbolic communication. Humans display striking resilience: we readily recognize words even when characters are fragmented, fused, or partially occluded. This paper investigates whether advanced vision language models (VLMs) share this resilience. We construct two psychophysics inspired benchmarks across distinct writing systems, Chinese logographs and English alphabetic words, by splicing, recombining, and overlaying glyphs to yield ''visible but unreadable'' stimuli for models while remaining legible to humans. Despite strong performance on clean text, contemporary VLMs show a severe drop under these perturbations, frequently producing unrelated or incoherent outputs. The pattern suggests a structural limitation: models heavily leverage generic visual invariances but under rely on compositional priors needed for robust literacy. We release stimuli generation code, prompts, and evaluation protocols to facilitate transparent replication and follow up work. Our findings motivate architectures and training strategies that encode symbol segmentation, composition, and binding across scripts, and they delineate concrete challenges for deploying multimodal systems in education, accessibility, cultural heritage, and security.
>
---
#### [replaced 052] LLM-RG: Referential Grounding in Outdoor Scenarios using Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25528v2](http://arxiv.org/pdf/2509.25528v2)**

> **作者:** Pranav Saxena; Avigyan Bhattacharya; Ji Zhang; Wenshan Wang
>
> **备注:** Human-aware Embodied AI Workshop @ IROS 2025
>
> **摘要:** Referential grounding in outdoor driving scenes is challenging due to large scene variability, many visually similar objects, and dynamic elements that complicate resolving natural-language references (e.g., "the black car on the right"). We propose LLM-RG, a hybrid pipeline that combines off-the-shelf vision-language models for fine-grained attribute extraction with large language models for symbolic reasoning. LLM-RG processes an image and a free-form referring expression by using an LLM to extract relevant object types and attributes, detecting candidate regions, generating rich visual descriptors with a VLM, and then combining these descriptors with spatial metadata into natural-language prompts that are input to an LLM for chain-of-thought reasoning to identify the referent's bounding box. Evaluated on the Talk2Car benchmark, LLM-RG yields substantial gains over both LLM and VLM-based baselines. Additionally, our ablations show that adding 3D spatial cues further improves grounding. Our results demonstrate the complementary strengths of VLMs and LLMs, applied in a zero-shot manner, for robust outdoor referential grounding.
>
---
#### [replaced 053] DA$^2$: Depth Anything in Any Direction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26618v3](http://arxiv.org/pdf/2509.26618v3)**

> **作者:** Haodong Li; Wangguangdong Zheng; Jing He; Yuhao Liu; Xin Lin; Xin Yang; Ying-Cong Chen; Chunchao Guo
>
> **备注:** Work primarily done during an internship at Tencent Hunyuan. Project page: https://depth-any-in-any-dir.github.io/
>
> **摘要:** Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more complete visual description than perspective images. Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision. However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization. Furthermore, due to the spherical distortions inherent in panoramas, many approaches rely on perspective splitting (e.g., cubemaps), which leads to suboptimal efficiency. To address these challenges, we propose $\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in $\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and fully end-to-end panoramic depth estimator. Specifically, for scaling up panoramic data, we introduce a data curation engine for generating high-quality panoramic depth data from perspective, and create $\sim$543K panoramic RGB-depth pairs, bringing the total to $\sim$607K. To further mitigate the spherical distortions, we present SphereViT, which explicitly leverages spherical coordinates to enforce the spherical geometric consistency in panoramic image features, yielding improved performance. A comprehensive benchmark on multiple datasets clearly demonstrates DA$^{2}$'s SoTA performance, with an average 38% improvement on AbsRel over the strongest zero-shot baseline. Surprisingly, DA$^{2}$ even outperforms prior in-domain methods, highlighting its superior zero-shot generalization. Moreover, as an end-to-end solution, DA$^{2}$ exhibits much higher efficiency over fusion-based approaches. Both the code and the curated panoramic data has be released. Project page: https://depth-any-in-any-dir.github.io/.
>
---
#### [replaced 054] Leveraging AV1 motion vectors for Fast and Dense Feature Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17434v2](http://arxiv.org/pdf/2510.17434v2)**

> **作者:** Julien Zouein; Hossein Javidnia; François Pitié; Anil Kokaram
>
> **备注:** Accepted ICIR 2025, camera-ready version
>
> **摘要:** We repurpose AV1 motion vectors to produce dense sub-pixel correspondences and short tracks filtered by cosine consistency. On short videos, this compressed-domain front end runs comparably to sequential SIFT while using far less CPU, and yields denser matches with competitive pairwise geometry. As a small SfM demo on a 117-frame clip, MV matches register all images and reconstruct 0.46-0.62M points at 0.51-0.53,px reprojection error; BA time grows with match density. These results show compressed-domain correspondences are a practical, resource-efficient front end with clear paths to scaling in full pipelines.
>
---
#### [replaced 055] VisuRiddles: Fine-grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02537v3](http://arxiv.org/pdf/2506.02537v3)**

> **作者:** Hao Yan; Xingchen Liu; Hao Wang; Zhenbiao Cao; Handong Zheng; Liang Yin; Xinxing Su; Zihao Chen; Jihao Wu; Minghui Liao; Chao Weng; Wei Chen; Yuliang Liu; Xiang Bai
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Recent strides in multimodal large language models (MLLMs) have significantly advanced their performance in many reasoning tasks. However, Abstract Visual Reasoning (AVR) remains a critical challenge, primarily due to limitations in perceiving abstract graphics. To tackle this issue, we investigate the bottlenecks in current MLLMs and synthesize training data to improve their abstract visual perception. First, we propose VisuRiddles, a benchmark for AVR, featuring tasks meticulously constructed to assess models' reasoning capacities across five core dimensions and two high-level reasoning categories. Second, we introduce the Perceptual Riddle Synthesizer (PRS), an automated framework for generating riddles with fine-grained perceptual descriptions. PRS not only generates valuable training data for abstract graphics but also provides fine-grained perceptual description, crucially allowing for supervision over intermediate reasoning stages and thereby improving both training efficacy and model interpretability. Our extensive experimental results on VisuRiddles empirically validate that fine-grained visual perception is the principal bottleneck and our synthesis framework markedly enhances the performance of contemporary MLLMs on these challenging tasks. Our code and dataset will be released at https://github.com/yh-hust/VisuRiddles
>
---
#### [replaced 056] Foundations of a Developmental Design Paradigm for Integrated Continual Learning, Deliberative Behavior, and Comprehensibility
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13935v2](http://arxiv.org/pdf/2502.13935v2)**

> **作者:** Zeki Doruk Erden; Boi Faltings
>
> **备注:** Accepted for IEEE Transactions on Emerging Topics in Computational Intelligence
>
> **摘要:** Inherent limitations of contemporary machine learning systems in crucial areas -- importantly in continual learning, information reuse, comprehensibility, and integration with deliberate behavior -- are receiving increasing attention. To address these challenges, we introduce a system design, fueled by a novel learning approach conceptually grounded in principles of evolutionary developmental biology, that overcomes key limitations of current methods. Our design comprises three core components: The Modeller, a gradient-free learning mechanism inherently capable of continual learning and structural adaptation; a planner for goal-directed action over learned models; and a behavior encapsulation mechanism that can decompose complex behaviors into a hierarchical structure. We demonstrate proof-of-principle operation in a simple test environment. Additionally, we extend our modeling framework to higher-dimensional network-structured spaces, using MNIST for a shape detection task. Our framework shows promise in overcoming multiple major limitations of contemporary machine learning systems simultaneously and in an organic manner.
>
---
#### [replaced 057] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18094v2](http://arxiv.org/pdf/2509.18094v2)**

> **作者:** Ye Liu; Zongyang Ma; Junfu Pu; Zhongang Qi; Yang Wu; Ying Shan; Chang Wen Chen
>
> **备注:** NeurIPS 2025 Camera Ready. Project Page: https://polyu-chenlab.github.io/unipixel/
>
> **摘要:** Recent advances in Large Multi-modal Models (LMMs) have demonstrated their remarkable success as general-purpose multi-modal assistants, with particular focuses on holistic image- and video-language understanding. Conversely, less attention has been given to scaling fine-grained pixel-level understanding capabilities, where the models are expected to realize pixel-level alignment between visual signals and language semantics. Some previous studies have applied LMMs to related tasks such as region-level captioning and referring expression segmentation. However, these models are limited to performing either referring or segmentation tasks independently and fail to integrate these fine-grained perception capabilities into visual reasoning. To bridge this gap, we propose UniPixel, a large multi-modal model capable of flexibly comprehending visual prompt inputs and generating mask-grounded responses. Our model distinguishes itself by seamlessly integrating pixel-level perception with general visual understanding capabilities. Specifically, UniPixel processes visual prompts and generates relevant masks on demand, and performs subsequent reasoning conditioning on these intermediate pointers during inference, thereby enabling fine-grained pixel-level reasoning. The effectiveness of our approach has been verified on 10 benchmarks across a diverse set of tasks, including pixel-level referring/segmentation and object-centric understanding in images/videos. A novel PixelQA task that jointly requires referring, segmentation, and question answering is also designed to verify the flexibility of our method.
>
---
#### [replaced 058] Curriculum Learning with Synthetic Data for Enhanced Pulmonary Nodule Detection in Chest Radiographs
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.07681v2](http://arxiv.org/pdf/2510.07681v2)**

> **作者:** Pranav Sambhu; Om Guin; Madhav Sambhu; Jinho Cha
>
> **备注:** This version has been withdrawn due to authorship changes and a decision to substantially revise the manuscript with new methodology. A future version may be submitted separately
>
> **摘要:** This study evaluates whether integrating curriculum learning with diffusion-based synthetic augmentation can enhance the detection of difficult pulmonary nodules in chest radiographs, particularly those with low size, brightness, and contrast, which often challenge conventional AI models due to data imbalance and limited annotation. A Faster R-CNN with a Feature Pyramid Network (FPN) backbone was trained on a hybrid dataset comprising expert-labeled NODE21 (1,213 patients; 52.4 percent male; mean age 63.2 +/- 11.5 years), VinDr-CXR, CheXpert, and 11,206 DDPM-generated synthetic images. Difficulty scores based on size, brightness, and contrast guided curriculum learning. Performance was compared to a non-curriculum baseline using mean average precision (mAP), Dice score, and area under the curve (AUC). Statistical tests included bootstrapped confidence intervals, DeLong tests, and paired t-tests. The curriculum model achieved a mean AUC of 0.95 versus 0.89 for the baseline (p < 0.001), with improvements in sensitivity (70 percent vs. 48 percent) and accuracy (82 percent vs. 70 percent). Stratified analysis demonstrated consistent gains across all difficulty bins (Easy to Very Hard). Grad-CAM visualizations confirmed more anatomically focused attention under curriculum learning. These results suggest that curriculum-guided synthetic augmentation enhances model robustness and generalization for pulmonary nodule detection.
>
---
#### [replaced 059] HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15625v2](http://arxiv.org/pdf/2506.15625v2)**

> **作者:** Roey Ron; Guy Tevet; Haim Sawdayee; Amit H. Bermano
>
> **备注:** Project page: https://hoidini.github.io
>
> **摘要:** We present HOIDiNi, a text-driven diffusion framework for synthesizing realistic and plausible human-object interaction (HOI). HOI generation is extremely challenging since it induces strict contact accuracies alongside a diverse motion manifold. While current literature trades off between realism and physical correctness, HOIDiNi optimizes directly in the noise space of a pretrained diffusion model using Diffusion Noise Optimization (DNO), achieving both. This is made feasible thanks to our observation that the problem can be separated into two phases: an object-centric phase, primarily making discrete choices of hand-object contact locations, and a human-centric phase that refines the full-body motion to realize this blueprint. This structured approach allows for precise hand-object contact without compromising motion naturalness. Quantitative, qualitative, and subjective evaluations on the GRAB dataset alone clearly indicate HOIDiNi outperforms prior works and baselines in contact accuracy, physical validity, and overall quality. Our results demonstrate the ability to generate complex, controllable interactions, including grasping, placing, and full-body coordination, driven solely by textual prompts. https://hoidini.github.io.
>
---
#### [replaced 060] Neural 3D Object Reconstruction with Small-Scale Unmanned Aerial Vehicles
- **分类: cs.RO; cs.AR; cs.CV; cs.ET; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.12458v2](http://arxiv.org/pdf/2509.12458v2)**

> **作者:** Àlmos Veres-Vitàlyos; Genis Castillo Gomez-Raya; Filip Lemic; Daniel Johannes Bugelnig; Bernhard Rinner; Sergi Abadal; Xavier Costa-Pérez
>
> **备注:** 13 pages, 16 figures, 3 tables, 45 references
>
> **摘要:** Small Unmanned Aerial Vehicles (UAVs) exhibit immense potential for navigating indoor and hard-to-reach areas, yet their significant constraints in payload and autonomy have largely prevented their use for complex tasks like high-quality 3-Dimensional (3D) reconstruction. To overcome this challenge, we introduce a novel system architecture that enables fully autonomous, high-fidelity 3D scanning of static objects using UAVs weighing under 100 grams. Our core innovation lies in a dual-reconstruction pipeline that creates a real-time feedback loop between data capture and flight control. A near-real-time (near-RT) process uses Structure from Motion (SfM) to generate an instantaneous pointcloud of the object. The system analyzes the model quality on the fly and dynamically adapts the UAV's trajectory to intelligently capture new images of poorly covered areas. This ensures comprehensive data acquisition. For the final, detailed output, a non-real-time (non-RT) pipeline employs a Neural Radiance Fields (NeRF)-based Neural 3D Reconstruction (N3DR) approach, fusing SfM-derived camera poses with precise Ultra Wide-Band (UWB) location data to achieve superior accuracy. We implemented and validated this architecture using Crazyflie 2.1 UAVs. Our experiments, conducted in both single- and multi-UAV configurations, conclusively show that dynamic trajectory adaptation consistently improves reconstruction quality over static flight paths. This work demonstrates a scalable and autonomous solution that unlocks the potential of miniaturized UAVs for fine-grained 3D reconstruction in constrained environments, a capability previously limited to much larger platforms.
>
---
#### [replaced 061] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04897v2](http://arxiv.org/pdf/2506.04897v2)**

> **作者:** Tianxu Wang; Zhuofan Zhang; Ziyu Zhu; Yue Fan; Jing Xiong; Pengxiang Li; Xiaojian Ma; Qing Li
>
> **备注:** NeurIPS 2025 (Datasets and Benchmarks). Project page: https://anywhere-3d.github.io/
>
> **摘要:** 3D visual grounding has made notable progress in localizing objects within complex 3D scenes. However, grounding referring expressions beyond objects in 3D scenes remains unexplored. In this paper, we introduce Anywhere3D-Bench, a holistic 3D visual grounding benchmark consisting of 2,886 referring expression-3D bounding box pairs spanning four different grounding levels: human-activity areas, unoccupied space beyond objects, individual objects in the scene, and fine-grained object parts. We assess a range of state-of-the-art 3D visual grounding methods alongside large language models (LLMs) and multimodal LLMs (MLLMs) on Anywhere3D-Bench. Experimental results reveal that space-level and part-level visual grounding pose the greatest challenges: space-level tasks require a more comprehensive spatial reasoning ability, for example, modeling distances and spatial relations within 3D space, while part-level tasks demand fine-grained perception of object composition. Even the best performance model, OpenAI o4-mini, achieves only 23.00% accuracy on space-level tasks and 31.46% on part-level tasks, significantly lower than its performance on area-level and object-level tasks. These findings underscore a critical gap in current models' capacity to understand and reason about 3D scenes beyond object-level semantics.
>
---
#### [replaced 062] H3D-DGS: Exploring Heterogeneous 3D Motion Representation for Deformable 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.13036v3](http://arxiv.org/pdf/2408.13036v3)**

> **作者:** Bing He; Yunuo Chen; Guo Lu; Qi Wang; Qunshan Gu; Rong Xie; Li Song; Wenjun Zhang
>
> **摘要:** Dynamic scene reconstruction poses a persistent challenge in 3D vision. Deformable 3D Gaussian Splatting has emerged as an effective method for this task, offering real-time rendering and high visual fidelity. This approach decomposes a dynamic scene into a static representation in a canonical space and time-varying scene motion. Scene motion is defined as the collective movement of all Gaussian points, and for compactness, existing approaches commonly adopt implicit neural fields or sparse control points. However, these methods predominantly rely on gradient-based optimization for all motion information. Due to the high degree of freedom, they struggle to converge on real-world datasets exhibiting complex motion. To preserve the compactness of motion representation and address convergence challenges, this paper proposes heterogeneous 3D control points, termed \textbf{H3D control points}, whose attributes are obtained using a hybrid strategy combining optical flow back-projection and gradient-based methods. This design decouples directly observable motion components from those that are geometrically occluded. Specifically, components of 3D motion that project onto the image plane are directly acquired via optical flow back projection, while unobservable portions are refined through gradient-based optimization. Experiments on the Neu3DV and CMU-Panoptic datasets demonstrate that our method achieves superior performance over state-of-the-art deformable 3D Gaussian splatting techniques. Remarkably, our method converges within just 100 iterations and achieves a per-frame processing speed of 2 seconds on a single NVIDIA RTX 4070 GPU.
>
---
#### [replaced 063] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10483v2](http://arxiv.org/pdf/2504.10483v2)**

> **作者:** Xingjian Leng; Jaskirat Singh; Yunzhong Hou; Zhenchang Xing; Saining Xie; Liang Zheng
>
> **摘要:** In this paper we tackle a fundamental question: "Can we train latent diffusion models together with the variational auto-encoder (VAE) tokenizer in an end-to-end manner?" Traditional deep-learning wisdom dictates that end-to-end training is often preferable when possible. However, for latent diffusion transformers, it is observed that end-to-end training both VAE and diffusion-model using standard diffusion-loss is ineffective, even causing a degradation in final performance. We show that while diffusion loss is ineffective, end-to-end training can be unlocked through the representation-alignment (REPA) loss -- allowing both VAE and diffusion model to be jointly tuned during the training process. Despite its simplicity, the proposed training recipe (REPA-E) shows remarkable performance; speeding up diffusion model training by over 17x and 45x over REPA and vanilla training recipes, respectively. Interestingly, we observe that end-to-end tuning with REPA-E also improves the VAE itself; leading to improved latent space structure and downstream generation performance. In terms of final performance, our approach sets a new state-of-the-art; achieving FID of 1.12 and 1.69 with and without classifier-free guidance on ImageNet 256 x 256. Code is available at https://end2end-diffusion.github.io.
>
---
#### [replaced 064] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10103v3](http://arxiv.org/pdf/2503.10103v3)**

> **作者:** Jiawei Zhang; Ziyuan Liu; Leon Yan; Gen Li; Yuantao Gu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Diffusion-based inverse algorithms have shown remarkable performance across various inverse problems, yet their reliance on numerous denoising steps incurs high computational costs. While recent developments of fast diffusion ODE solvers offer effective acceleration for diffusion sampling without observations, their application in inverse problems remains limited due to the heterogeneous formulations of inverse algorithms and their prevalent use of approximations and heuristics, which often introduce significant errors that undermine the reliability of analytical solvers. In this work, we begin with an analysis of ODE solvers for inverse problems that reveals a linear combination structure of approximations for the inverse trajectory. Building on this insight, we propose a canonical form that unifies a broad class of diffusion-based inverse algorithms and facilitates the design of more generalizable solvers. Inspired by the linear subspace search strategy, we propose Learnable Linear Extrapolation (LLE), a lightweight approach that universally enhances the performance of any diffusion-based inverse algorithm conforming to our canonical form. LLE optimizes the combination coefficients to refine current predictions using previous estimates, alleviating the sensitivity of analytical solvers for inverse algorithms. Extensive experiments demonstrate consistent improvements of the proposed LLE method across multiple algorithms and tasks, indicating its potential for more efficient solutions and boosted performance of diffusion-based inverse algorithms with limited steps. Codes for reproducing our experiments are available at https://github.com/weigerzan/LLE_inverse_problem.
>
---
#### [replaced 065] RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.13144v2](http://arxiv.org/pdf/2502.13144v2)**

> **作者:** Hao Gao; Shaoyu Chen; Bo Jiang; Bencheng Liao; Yiang Shi; Xiaoyang Guo; Yuechuan Pu; Haoran Yin; Xiangyu Li; Xinbang Zhang; Ying Zhang; Wenyu Liu; Qian Zhang; Xinggang Wang
>
> **备注:** Code: https://github.com/hustvl/RAD
>
> **摘要:** Existing end-to-end autonomous driving (AD) algorithms typically follow the Imitation Learning (IL) paradigm, which faces challenges such as causal confusion and an open-loop gap. In this work, we propose RAD, a 3DGS-based closed-loop Reinforcement Learning (RL) framework for end-to-end Autonomous Driving. By leveraging 3DGS techniques, we construct a photorealistic digital replica of the real physical world, enabling the AD policy to extensively explore the state space and learn to handle out-of-distribution scenarios through large-scale trial and error. To enhance safety, we design specialized rewards to guide the policy in effectively responding to safety-critical events and understanding real-world causal relationships. To better align with human driving behavior, we incorporate IL into RL training as a regularization term. We introduce a closed-loop evaluation benchmark consisting of diverse, previously unseen 3DGS environments. Compared to IL-based methods, RAD achieves stronger performance in most closed-loop metrics, particularly exhibiting a 3x lower collision rate. Abundant closed-loop results are presented in the supplementary material. Code is available at https://github.com/hustvl/RAD for facilitating future research.
>
---
#### [replaced 066] Med-2E3: A 2D-Enhanced 3D Medical Multimodal Large Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12783v2](http://arxiv.org/pdf/2411.12783v2)**

> **作者:** Yiming Shi; Xun Zhu; Kaiwen Wang; Ying Hu; Chenyi Guo; Miao Li; Ji Wu
>
> **摘要:** 3D medical image analysis is essential for modern healthcare, yet traditional task-specific models are inadequate due to limited generalizability across diverse clinical scenarios. Multimodal large language models (MLLMs) offer a promising solution to these challenges. However, existing MLLMs have limitations in fully leveraging the rich, hierarchical information embedded in 3D medical images. Inspired by clinical practice, where radiologists focus on both 3D spatial structure and 2D planar content, we propose Med-2E3, a 3D medical MLLM that integrates a dual 3D-2D encoder architecture. To aggregate 2D features effectively, we design a Text-Guided Inter-Slice (TG-IS) scoring module, which scores the attention of each 2D slice based on slice contents and task instructions. To the best of our knowledge, Med-2E3 is the first MLLM to integrate both 3D and 2D features for 3D medical image analysis. Experiments on large-scale, open-source 3D medical multimodal datasets demonstrate that TG-IS exhibits task-specific attention distribution and significantly outperforms current state-of-the-art models. The code is available at: https://github.com/MSIIP/Med-2E3
>
---
#### [replaced 067] SAMPO:Scale-wise Autoregression with Motion PrOmpt for generative world models
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.15536v2](http://arxiv.org/pdf/2509.15536v2)**

> **作者:** Sen Wang; Jingyi Tian; Le Wang; Zhimin Liao; Jiayi Li; Huaiyi Dong; Kun Xia; Sanping Zhou; Wei Tang; Hua Gang
>
> **备注:** 22 pages,15 figures
>
> **摘要:** World models allow agents to simulate the consequences of actions in imagined environments for planning, control, and long-horizon decision-making. However, existing autoregressive world models struggle with visually coherent predictions due to disrupted spatial structure, inefficient decoding, and inadequate motion modeling. In response, we propose \textbf{S}cale-wise \textbf{A}utoregression with \textbf{M}otion \textbf{P}r\textbf{O}mpt (\textbf{SAMPO}), a hybrid framework that combines visual autoregressive modeling for intra-frame generation with causal modeling for next-frame generation. Specifically, SAMPO integrates temporal causal decoding with bidirectional spatial attention, which preserves spatial locality and supports parallel decoding within each scale. This design significantly enhances both temporal consistency and rollout efficiency. To further improve dynamic scene understanding, we devise an asymmetric multi-scale tokenizer that preserves spatial details in observed frames and extracts compact dynamic representations for future frames, optimizing both memory usage and model performance. Additionally, we introduce a trajectory-aware motion prompt module that injects spatiotemporal cues about object and robot trajectories, focusing attention on dynamic regions and improving temporal consistency and physical realism. Extensive experiments show that SAMPO achieves competitive performance in action-conditioned video prediction and model-based control, improving generation quality with 4.4$\times$ faster inference. We also evaluate SAMPO's zero-shot generalization and scaling behavior, demonstrating its ability to generalize to unseen tasks and benefit from larger model sizes.
>
---
#### [replaced 068] Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15277v3](http://arxiv.org/pdf/2411.15277v3)**

> **作者:** Yiyang Cai; Zhengkai Jiang; Yulong Liu; Chunyang Jiang; Wei Xue; Yike Guo; Wenhan Luo
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Facial personalization faces challenges to maintain identity fidelity without disrupting the foundation model's prompt consistency. The mainstream personalization models employ identity embedding to integrate identity information within the attention mechanisms. However, our preliminary findings reveal that identity embeddings compromise the effectiveness of other tokens in the prompt, thereby limiting high prompt consistency and attribute-level controllability. Moreover, by deactivating identity embedding, personalization models still demonstrate the underlying foundation models' ability to control facial attributes precisely. It suggests that such foundation models' knowledge can be leveraged to cure the ill-aligned prompt consistency of personalization models. Building upon these insights, we propose FreeCure, a framework that improves the prompt consistency of personalization models with their latent foundation models' knowledge. First, by setting a dual inference paradigm with/without identity embedding, we identify attributes (e.g., hair, accessories, etc.) for enhancements. Second, we introduce a novel foundation-aware self-attention module, coupled with an inversion-based process to bring well-aligned attribute information to the personalization process. Our approach is training-free, and can effectively enhance a wide array of facial attributes; and it can be seamlessly integrated into existing popular personalization models based on both Stable Diffusion and FLUX. FreeCure has consistently shown significant improvements in prompt consistency across these facial personalization models while maintaining the integrity of their original identity fidelity.
>
---
#### [replaced 069] Fourier Transform Multiple Instance Learning for Whole Slide Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.15138v2](http://arxiv.org/pdf/2510.15138v2)**

> **作者:** Anthony Bilic; Guangyu Sun; Ming Li; Md Sanzid Bin Hossain; Yu Tian; Wei Zhang; Laura Brattain; Dexter Hadley; Chen Chen
>
> **摘要:** Whole Slide Image (WSI) classification relies on Multiple Instance Learning (MIL) with spatial patch features, yet existing methods struggle to capture global dependencies due to the immense size of WSIs and the local nature of patch embeddings. This limitation hinders the modeling of coarse structures essential for robust diagnostic prediction. We propose Fourier Transform Multiple Instance Learning (FFT-MIL), a framework that augments MIL with a frequency-domain branch to provide compact global context. Low-frequency crops are extracted from WSIs via the Fast Fourier Transform and processed through a modular FFT-Block composed of convolutional layers and Min-Max normalization to mitigate the high variance of frequency data. The learned global frequency feature is fused with spatial patch features through lightweight integration strategies, enabling compatibility with diverse MIL architectures. FFT-MIL was evaluated across six state-of-the-art MIL methods on three public datasets (BRACS, LUAD, and IMP). Integration of the FFT-Block improved macro F1 scores by an average of 3.51% and AUC by 1.51%, demonstrating consistent gains across architectures and datasets. These results establish frequency-domain learning as an effective and efficient mechanism for capturing global dependencies in WSI classification, complementing spatial features and advancing the scalability and accuracy of MIL-based computational pathology.
>
---
#### [replaced 070] Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.07127v3](http://arxiv.org/pdf/2402.07127v3)**

> **作者:** Chrisantus Eze; Christopher Crick
>
> **备注:** Published at IEEE Access
>
> **摘要:** Robot learning of manipulation skills is hindered by the scarcity of diverse, unbiased datasets. While curated datasets can help, challenges remain in generalizability and real-world transfer. Meanwhile, large-scale "in-the-wild" video datasets have driven progress in computer vision through self-supervised techniques. Translating this to robotics, recent works have explored learning manipulation skills by passively watching abundant videos sourced online. Showing promising results, such video-based learning paradigms provide scalable supervision while reducing dataset bias. This survey reviews foundations such as video feature representation learning techniques, object affordance understanding, 3D hand/body modeling, and large-scale robot resources, as well as emerging techniques for acquiring robot manipulation skills from uncontrolled video demonstrations. We discuss how learning only from observing large-scale human videos can enhance generalization and sample efficiency for robotic manipulation. The survey summarizes video-based learning approaches, analyses their benefits over standard datasets, survey metrics, and benchmarks, and discusses open challenges and future directions in this nascent domain at the intersection of computer vision, natural language processing, and robot learning.
>
---
#### [replaced 071] DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
- **分类: cs.CV; I.4.9**

- **链接: [http://arxiv.org/pdf/2505.21089v2](http://arxiv.org/pdf/2505.21089v2)**

> **作者:** Junjue Wang; Weihao Xuan; Heli Qi; Zhihao Liu; Kunyi Liu; Yuhan Wu; Hongruixuan Chen; Jian Song; Junshi Xia; Zhuo Zheng; Naoto Yokoya
>
> **备注:** A multi-hazard, multi-sensor, and multi-task vision-language dataset for global-scale disaster assessment and response
>
> **摘要:** Large vision-language models (VLMs) have made great achievements in Earth vision. However, complex disaster scenes with diverse disaster types, geographic regions, and satellite sensors have posed new challenges for VLM applications. To fill this gap, we curate a remote sensing vision-language dataset (DisasterM3) for global-scale disaster assessment and response. DisasterM3 includes 26,988 bi-temporal satellite images and 123k instruction pairs across 5 continents, with three characteristics: 1) Multi-hazard: DisasterM3 involves 36 historical disaster events with significant impacts, which are categorized into 10 common natural and man-made disasters. 2)Multi-sensor: Extreme weather during disasters often hinders optical sensor imaging, making it necessary to combine Synthetic Aperture Radar (SAR) imagery for post-disaster scenes. 3) Multi-task: Based on real-world scenarios, DisasterM3 includes 9 disaster-related visual perception and reasoning tasks, harnessing the full potential of VLM's reasoning ability with progressing from disaster-bearing body recognition to structural damage assessment and object relational reasoning, culminating in the generation of long-form disaster reports. We extensively evaluated 14 generic and remote sensing VLMs on our benchmark, revealing that state-of-the-art models struggle with the disaster tasks, largely due to the lack of a disaster-specific corpus, cross-sensor gap, and damage object counting insensitivity. Focusing on these issues, we fine-tune four VLMs using our dataset and achieve stable improvements across all tasks, with robust cross-sensor and cross-disaster generalization capabilities. The code and data are available at: https://github.com/Junjue-Wang/DisasterM3.
>
---
#### [replaced 072] H3DE-Net: Efficient and Accurate 3D Landmark Detection in Medical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14221v3](http://arxiv.org/pdf/2502.14221v3)**

> **作者:** Zhen Huang; Tao Tang; Ronghao Xu; Yangbo Wei; Wenkai Yang; Suhua Wang; Xiaoxin Sun; Han Li; Qingsong Yao
>
> **摘要:** 3D landmark detection is a critical task in medical image analysis, and accurately detecting anatomical landmarks is essential for subsequent medical imaging tasks. However, mainstream deep learning methods in this field struggle to simultaneously capture fine-grained local features and model global spatial relationships, while maintaining a balance between accuracy and computational efficiency. Local feature extraction requires capturing fine-grained anatomical details, while global modeling requires understanding the spatial relationships within complex anatomical structures. The high-dimensional nature of 3D volume further exacerbates these challenges, as landmarks are sparsely distributed, leading to significant computational costs. Therefore, achieving efficient and precise 3D landmark detection remains a pressing challenge in medical image analysis. In this work, We propose a \textbf{H}ybrid \textbf{3}D \textbf{DE}tection \textbf{Net}(H3DE-Net), a novel framework that combines CNNs for local feature extraction with a lightweight attention mechanism designed to efficiently capture global dependencies in 3D volumetric data. This mechanism employs a hierarchical routing strategy to reduce computational cost while maintaining global context modeling. To our knowledge, H3DE-Net is the first 3D landmark detection model that integrates such a lightweight attention mechanism with CNNs. Additionally, integrating multi-scale feature fusion further enhances detection accuracy and robustness. Experimental results on a public CT dataset demonstrate that H3DE-Net achieves state-of-the-art(SOTA) performance, significantly improving accuracy and robustness, particularly in scenarios with missing landmarks or complex anatomical variations. We aready open-source our project, including code, data and model weights.
>
---
#### [replaced 073] View Transformation Robustness for Multi-View 3D Object Reconstruction with Reconstruction Error-Guided View Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11428v2](http://arxiv.org/pdf/2412.11428v2)**

> **作者:** Qi Zhang; Zhouhang Luo; Tao Yu; Hui Huang
>
> **备注:** AAAI 25
>
> **摘要:** View transformation robustness (VTR) is critical for deep-learning-based multi-view 3D object reconstruction models, which indicates the methods' stability under inputs with various view transformations. However, existing research seldom focused on view transformation robustness in multi-view 3D object reconstruction. One direct way to improve the models' VTR is to produce data with more view transformations and add them to model training. Recent progress on large vision models, particularly Stable Diffusion models, has provided great potential for generating 3D models or synthesizing novel view images with only a single image input. Directly deploying these models at inference consumes heavy computation resources and their robustness to view transformations is not guaranteed either. To fully utilize the power of Stable Diffusion models without extra inference computation burdens, we propose to generate novel views with Stable Diffusion models for better view transformation robustness. Instead of synthesizing random views, we propose a reconstruction error-guided view selection method, which considers the reconstruction errors' spatial distribution of the 3D predictions and chooses the views that could cover the reconstruction errors as much as possible. The methods are trained and tested on sets with large view transformations to validate the 3D reconstruction models' robustness to view transformations. Extensive experiments demonstrate that the proposed method can outperform state-of-the-art 3D reconstruction methods and other view transformation robustness comparison methods. Code is available at: https://github.com/zqyq/VTR.
>
---
#### [replaced 074] A Multimodal Deep Learning Approach for White Matter Shape Prediction in Diffusion MRI Tractography
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18400v4](http://arxiv.org/pdf/2504.18400v4)**

> **作者:** Yui Lo; Yuqian Chen; Dongnan Liu; Leo Zekelman; Jarrett Rushmore; Yogesh Rathi; Nikos Makris; Alexandra J. Golby; Fan Zhang; Weidong Cai; Lauren J. O'Donnell
>
> **备注:** Paper accepted to Human Brain Mapping. 25 pages, 3 figures, 8 tables
>
> **摘要:** Shape measures have emerged as promising descriptors of white matter tractography, offering complementary insights into anatomical variability and associations with cognitive and clinical phenotypes. However, conventional methods for computing shape measures are computationally expensive and time-consuming for large-scale datasets due to reliance on voxel-based representations. We propose Tract2Shape, a novel multimodal deep learning framework that leverages geometric (point cloud) and scalar (tabular) features to predict ten white matter tractography shape measures. To enhance model efficiency, we utilize a dimensionality reduction algorithm for the model to predict five primary shape components. The model is trained and evaluated on two independently acquired datasets, the HCP-YA dataset, and the PPMI dataset. We evaluate the performance of Tract2Shape by training and testing it on the HCP-YA dataset and comparing the results with state-of-the-art models. To further assess its robustness and generalization ability, we also test Tract2Shape on the unseen PPMI dataset. Tract2Shape outperforms SOTA deep learning models across all ten shape measures, achieving the highest average Pearson's r and the lowest nMSE on the HCP-YA dataset. The ablation study shows that both multimodal input and PCA contribute to performance gains. On the unseen testing PPMI dataset, Tract2Shape maintains a high Pearson's r and low nMSE, demonstrating strong generalizability in cross-dataset evaluation. Tract2Shape enables fast, accurate, and generalizable prediction of white matter shape measures from tractography data, supporting scalable analysis across datasets. This framework lays a promising foundation for future large-scale white matter shape analysis.
>
---
#### [replaced 075] Adapting Medical Vision Foundation Models for Volumetric Medical Image Segmentation via Active Learning and Selective Semi-supervised Fine-tuning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10784v2](http://arxiv.org/pdf/2509.10784v2)**

> **作者:** Jin Yang; Daniel S. Marcus; Aristeidis Sotiras
>
> **备注:** 17 pages, 5 figures, 8 tables
>
> **摘要:** Medical Vision Foundation Models (Med-VFMs) have superior capabilities of interpreting medical images due to the knowledge learned from self-supervised pre-training with extensive unannotated images. To improve their performance on adaptive downstream evaluations, especially segmentation, a few samples from target domains are selected randomly for fine-tuning them. However, there lacks works to explore the way of adapting Med-VFMs to achieve the optimal performance on target domains efficiently. Thus, it is highly demanded to design an efficient way of fine-tuning Med-VFMs by selecting informative samples to maximize their adaptation performance on target domains. To achieve this, we propose an Active Source-Free Domain Adaptation (ASFDA) method to efficiently adapt Med-VFMs to target domains for volumetric medical image segmentation. This ASFDA employs a novel Active Learning (AL) method to select the most informative samples from target domains for fine-tuning Med-VFMs without the access to source pre-training samples, thus maximizing their performance with the minimal selection budget. In this AL method, we design an Active Test Time Sample Query strategy to select samples from the target domains via two query metrics, including Diversified Knowledge Divergence (DKD) and Anatomical Segmentation Difficulty (ASD). DKD is designed to measure the source-target knowledge gap and intra-domain diversity. It utilizes the knowledge of pre-training to guide the querying of source-dissimilar and semantic-diverse samples from the target domains. ASD is designed to evaluate the difficulty in segmentation of anatomical structures by measuring predictive entropy from foreground regions adaptively. Additionally, our ASFDA method employs a Selective Semi-supervised Fine-tuning to improve the performance and efficiency of fine-tuning by identifying samples with high reliability from unqueried ones.
>
---
#### [replaced 076] FlySearch: Exploring how vision-language models explore
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02896v3](http://arxiv.org/pdf/2506.02896v3)**

> **作者:** Adam Pardyl; Dominik Matuszek; Mateusz Przebieracz; Marek Cygan; Bartosz Zieliński; Maciej Wołczyk
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks track
>
> **摘要:** The real world is messy and unstructured. Uncovering critical information often requires active, goal-driven exploration. It remains to be seen whether Vision-Language Models (VLMs), which recently emerged as a popular zero-shot tool in many difficult tasks, can operate effectively in such conditions. In this paper, we answer this question by introducing FlySearch, a 3D, outdoor, photorealistic environment for searching and navigating to objects in complex scenes. We define three sets of scenarios with varying difficulty and observe that state-of-the-art VLMs cannot reliably solve even the simplest exploration tasks, with the gap to human performance increasing as the tasks get harder. We identify a set of central causes, ranging from vision hallucination, through context misunderstanding, to task planning failures, and we show that some of them can be addressed by finetuning. We publicly release the benchmark, scenarios, and the underlying codebase.
>
---
#### [replaced 077] MSR-Align: Policy-Grounded Multimodal Alignment for Safety-Aware Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19257v2](http://arxiv.org/pdf/2506.19257v2)**

> **作者:** Yinan Xia; Yilei Jiang; Yingshui Tan; Xiaoyong Zhu; Xiangyu Yue; Bo Zheng
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in multimodal reasoning tasks through enhanced chain-of-thought capabilities. However, this advancement also introduces novel safety risks, as these models become increasingly vulnerable to harmful multimodal prompts that can trigger unethical or unsafe behaviors. Existing safety alignment approaches, primarily designed for unimodal language models, fall short in addressing the complex and nuanced threats posed by multimodal inputs. Moreover, current safety datasets lack the fine-grained, policy-grounded reasoning required to robustly align reasoning-capable VLMs. In this work, we introduce {MSR-Align}, a high-quality Multimodal Safety Reasoning dataset tailored to bridge this gap. MSR-Align supports fine-grained, deliberative reasoning over standardized safety policies across both vision and text modalities. Our data generation pipeline emphasizes multimodal diversity, policy-grounded reasoning, and rigorous quality filtering using strong multimodal judges. Extensive experiments demonstrate that fine-tuning VLMs on MSR-Align substantially improves robustness against both textual and vision-language jailbreak attacks, while preserving or enhancing general reasoning performance. MSR-Align provides a scalable and effective foundation for advancing the safety alignment of reasoning-capable VLMs. Our dataset is made publicly available at https://huggingface.co/datasets/Leigest/MSR-Align.
>
---
#### [replaced 078] SimCortex: Collision-free Simultaneous Cortical Surfaces Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06955v2](http://arxiv.org/pdf/2507.06955v2)**

> **作者:** Kaveh Moradkhani; R Jarrett Rushmore; Sylvain Bouix
>
> **备注:** Metadata update: added journal reference and DOI linking to the published chapter (Springer)
>
> **摘要:** Accurate cortical surface reconstruction from magnetic resonance imaging (MRI) data is crucial for reliable neuroanatomical analyses. Current methods have to contend with complex cortical geometries, strict topological requirements, and often produce surfaces with overlaps, self-intersections, and topological defects. To overcome these shortcomings, we introduce SimCortex, a deep learning framework that simultaneously reconstructs all brain surfaces (left/right white-matter and pial) from T1-weighted(T1w) MRI volumes while preserving topological properties. Our method first segments the T1w image into a nine-class tissue label map. From these segmentations, we generate subject-specific, collision-free initial surface meshes. These surfaces serve as precise initializations for subsequent multiscale diffeomorphic deformations. Employing stationary velocity fields (SVFs) integrated via scaling-and-squaring, our approach ensures smooth, topology-preserving transformations with significantly reduced surface collisions and self-intersections. Evaluations on standard datasets demonstrate that SimCortex dramatically reduces surface overlaps and self-intersections, surpassing current methods while maintaining state-of-the-art geometric accuracy.
>
---
#### [replaced 079] Interpretable Decision-Making for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18898v3](http://arxiv.org/pdf/2508.18898v3)**

> **作者:** Mona Mirzaie; Bodo Rosenhahn
>
> **备注:** Accepted to the ICCV 2025 2nd Workshop on the Challenge Of Out-of-Label Hazards in Autonomous Driving (2COOOL)
>
> **摘要:** Trustworthy AI is mandatory for the broad deployment of autonomous vehicles. Although end-to-end approaches derive control commands directly from raw data, interpreting these decisions remains challenging, especially in complex urban scenarios. This is mainly attributed to very deep neural networks with non-linear decision boundaries, making it challenging to grasp the logic behind AI-driven decisions. This paper presents a method to enhance interpretability while optimizing control commands in autonomous driving. To address this, we propose loss functions that promote the interpretability of our model by generating sparse and localized feature maps. The feature activations allow us to explain which image regions contribute to the predicted control command. We conduct comprehensive ablation studies on the feature extraction step and validate our method on the CARLA benchmarks. We also demonstrate that our approach improves interpretability, which correlates with reducing infractions, yielding a safer, high-performance driving model. Notably, our monocular, non-ensemble model surpasses the top-performing approaches from the CARLA Leaderboard by achieving lower infraction scores and the highest route completion rate, all while ensuring interpretability.
>
---
#### [replaced 080] ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09385v2](http://arxiv.org/pdf/2506.09385v2)**

> **作者:** Jialong Zuo; Yongtai Deng; Mengdan Tan; Rui Jin; Dongyue Wu; Nong Sang; Liang Pan; Changxin Gao
>
> **备注:** NeurIPS2025 Accepted Paper
>
> **摘要:** In real-word scenarios, person re-identification (ReID) expects to identify a person-of-interest via the descriptive query, regardless of whether the query is a single modality or a combination of multiple modalities. However, existing methods and datasets remain constrained to limited modalities, failing to meet this requirement. Therefore, we investigate a new challenging problem called Omni Multi-modal Person Re-identification (OM-ReID), which aims to achieve effective retrieval with varying multi-modal queries. To address dataset scarcity, we construct ORBench, the first high-quality multi-modal dataset comprising 1,000 unique identities across five modalities: RGB, infrared, color pencil, sketch, and textual description. This dataset also has significant superiority in terms of diversity, such as the painting perspectives and textual information. It could serve as an ideal platform for follow-up investigations in OM-ReID. Moreover, we propose ReID5o, a novel multi-modal learning framework for person ReID. It enables synergistic fusion and cross-modal alignment of arbitrary modality combinations in a single model, with a unified encoding and multi-expert routing mechanism proposed. Extensive experiments verify the advancement and practicality of our ORBench. A wide range of possible models have been evaluated and compared on it, and our proposed ReID5o model gives the best performance. The dataset and code will be made publicly available at https://github.com/Zplusdragon/ReID5o_ORBench.
>
---
#### [replaced 081] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02175v2](http://arxiv.org/pdf/2502.02175v2)**

> **作者:** Siyu Xu; Yunke Wang; Chenghao Xia; Dihao Zhu; Tao Huang; Chang Xu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated strong multi-modal reasoning capabilities, enabling direct action generation from visual perception and language instructions in an end-to-end manner. However, their substantial computational cost poses a challenge for real-time robotic control, where rapid decision-making is essential. This paper introduces VLA-Cache, a training-free inference acceleration method that reduces computational overhead by adaptively caching and reusing static visual tokens across frames. Exploiting the temporal continuity in robotic manipulation, VLA-Cache identifies minimally changed tokens between adjacent frames and reuses their cached key-value representations, thereby circumventing redundant computations. Additionally, to maintain action precision, VLA-Cache selectively re-computes task-relevant tokens that are environmentally sensitive, ensuring the fidelity of critical visual information. To further optimize efficiency, we introduce a layer adaptive token reusing strategy that dynamically adjusts the reuse ratio based on attention concentration across decoder layers, prioritizing critical tokens for recomputation. Extensive experiments on two simulation platforms (LIBERO and SIMPLER) and a real-world robotic system demonstrate that VLA-Cache achieves up to 1.7x speedup in CUDA latency and a 15% increase in control frequency, with negligible loss on task success rate. The code and videos can be found at our project page: https://vla-cache.github.io.
>
---
#### [replaced 082] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22983v3](http://arxiv.org/pdf/2503.22983v3)**

> **作者:** Ashesh Ashesh; Florian Jug
>
> **备注:** manuscript accepted at NeurIPS 2025
>
> **摘要:** Fluorescence microscopy, while being a key driver for progress in the life sciences, is also subject to technical limitations. To overcome them, computational multiplexing techniques have recently been proposed, which allow multiple cellular structures to be captured in a single image and later be unmixed. Existing image decomposition methods are trained on a set of superimposed input images and the respective unmixed target images. It is critical to note that the relative strength (mixing ratio) of the superimposed images for a given input is a priori unknown. However, existing methods are trained on a fixed intensity ratio of superimposed inputs, making them not cognizant of the range of relative intensities that can occur in fluorescence microscopy. In this work, we propose a novel method called scSplit that is cognizant of the severity of the above-mentioned mixing ratio. Our idea is based on InDI , a popular iterative method for image restoration, and an ideal starting point to embrace the unknown mixing ratio in any given input. We introduce (i) a suitably trained regressor network that predicts the degradation level (mixing ratio) of a given input image and (ii) a degradation-specific normalization module, enabling degradation-aware inference across all mixing ratios. We show that this method solves two relevant tasks in fluorescence microscopy, namely image splitting and bleedthrough removal, and empirically demonstrate the applicability of scSplit on 5 public datasets. The source code with pre-trained models is hosted at https://github.com/juglab/scSplit/.
>
---
#### [replaced 083] Exploring Cross-Modal Flows for Few-Shot Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14543v2](http://arxiv.org/pdf/2510.14543v2)**

> **作者:** Ziqi Jiang; Yanghao Wang; Long Chen
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Aligning features from different modalities, is one of the most fundamental challenges for cross-modal tasks. Although pre-trained vision-language models can achieve a general alignment between image and text, they often require parameter-efficient fine-tuning (PEFT) for further adjustment. Today's PEFT methods (e.g., prompt tuning, LoRA-based, or adapter-based) always selectively fine-tune a subset of parameters, which can slightly adjust either visual or textual features, and avoid overfitting. In this paper, we are the first to highlight that all existing PEFT methods perform one-step adjustment. It is insufficient for complex (or difficult) datasets, where features of different modalities are highly entangled. To this end, we propose the first model-agnostic multi-step adjustment approach by learning a cross-modal velocity field: Flow Matching Alignment (FMA). Specifically, to ensure the correspondence between categories during training, we first utilize a fixed coupling strategy. Then, we propose a noise augmentation strategy to alleviate the data scarcity issue. Finally, we design an early-stopping solver, which terminates the transformation process earlier, improving both efficiency and accuracy. Compared with one-step PEFT methods, FMA has the multi-step rectification ability to achieve more precise and robust alignment. Extensive results have demonstrated that FMA can consistently yield significant performance gains across various benchmarks and backbones, particularly on challenging datasets.
>
---
#### [replaced 084] VLLFL: A Vision-Language Model Based Lightweight Federated Learning Framework for Smart Agriculture
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.13365v2](http://arxiv.org/pdf/2504.13365v2)**

> **作者:** Long Li; Jiajia Li; Dong Chen; Lina Pu; Haibo Yao; Yanbo Huang
>
> **摘要:** In modern smart agriculture, object detection plays a crucial role by enabling automation, precision farming, and monitoring of resources. From identifying crop health and pest infestations to optimizing harvesting processes, accurate object detection enhances both productivity and sustainability. However, training object detection models often requires large-scale data collection and raises privacy concerns, particularly when sensitive agricultural data is distributed across farms. To address these challenges, we propose VLLFL, a vision-language model-based lightweight federated learning framework (VLLFL). It harnesses the generalization and context-aware detection capabilities of the vision-language model (VLM) and leverages the privacy-preserving nature of federated learning. By training a compact prompt generator to boost the performance of the VLM deployed across different farms, VLLFL preserves privacy while reducing communication overhead. Experimental results demonstrate that VLLFL achieves 14.53% improvement in the performance of VLM while reducing 99.3% communication overhead. Spanning tasks from identifying a wide variety of fruits to detecting harmful animals in agriculture, the proposed framework offers an efficient, scalable, and privacy-preserving solution specifically tailored to agricultural applications.
>
---
#### [replaced 085] Learning Collaborative Knowledge with Multimodal Representation for Polyp Re-Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05914v3](http://arxiv.org/pdf/2408.05914v3)**

> **作者:** Suncheng Xiang; Jiale Guan; Shilun Cai; Jiacheng Ruan; Dahong Qian
>
> **摘要:** Colonoscopic Polyp Re-Identification aims to match the same polyp from a large gallery with images from different views taken using different cameras, which plays an important role in the prevention and treatment of colorectal cancer in computer-aided diagnosis. However, traditional methods for object ReID directly adopting CNN models trained on the ImageNet dataset usually produce unsatisfactory retrieval performance on colonoscopic datasets due to the large domain gap. Worsely, these solutions typically learn unimodal modal representations on the basis of visual samples, which fails to explore complementary information from other different modalities. To address this challenge, we propose a novel Deep Multimodal Collaborative Learning framework named DMCL for polyp re-identification, which can effectively encourage multimodal knowledge collaboration and reinforce generalization capability in medical scenarios. On the basis of it, a dynamic multimodal feature fusion strategy is introduced to leverage the optimized visual-text representations for multimodal fusion via end-to-end training. Experiments on the standard benchmarks show the benefits of the multimodal setting over state-of-the-art unimodal ReID models, especially when combined with the collaborative multimodal fusion strategy. The code is publicly available at https://github.com/JeremyXSC/DMCL.
>
---
#### [replaced 086] WMamba: Wavelet-based Mamba for Face Forgery Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.09617v2](http://arxiv.org/pdf/2501.09617v2)**

> **作者:** Siran Peng; Tianshuo Zhang; Li Gao; Xiangyu Zhu; Haoyuan Zhang; Kai Pang; Zhen Lei
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** The rapid evolution of deepfake generation technologies necessitates the development of robust face forgery detection algorithms. Recent studies have demonstrated that wavelet analysis can enhance the generalization abilities of forgery detectors. Wavelets effectively capture key facial contours, often slender, fine-grained, and globally distributed, that may conceal subtle forgery artifacts imperceptible in the spatial domain. However, current wavelet-based approaches fail to fully exploit the distinctive properties of wavelet data, resulting in sub-optimal feature extraction and limited performance gains. To address this challenge, we introduce WMamba, a novel wavelet-based feature extractor built upon the Mamba architecture. WMamba maximizes the utility of wavelet information through two key innovations. First, we propose Dynamic Contour Convolution (DCConv), which employs specially crafted deformable kernels to adaptively model slender facial contours. Second, by leveraging the Mamba architecture, our method captures long-range spatial relationships with linear complexity. This efficiency allows for the extraction of fine-grained, globally distributed forgery artifacts from small image patches. Extensive experiments show that WMamba achieves state-of-the-art (SOTA) performance, highlighting its effectiveness in face forgery detection.
>
---
#### [replaced 087] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19858v2](http://arxiv.org/pdf/2505.19858v2)**

> **作者:** Zixiang Zhao; Haowen Bai; Bingxin Ke; Yukun Cui; Lilun Deng; Yulun Zhang; Kai Zhang; Konrad Schindler
>
> **备注:** Accepted by NeurIPS 2025 (Spotlight)
>
> **摘要:** The real world is dynamic, yet most image fusion methods process static frames independently, ignoring temporal correlations in videos and leading to flickering and temporal inconsistency. To address this, we propose Unified Video Fusion (UniVF), a novel and unified framework for video fusion that leverages multi-frame learning and optical flow-based feature warping for informative, temporally coherent video fusion. To support its development, we also introduce Video Fusion Benchmark (VF-Bench), the first comprehensive benchmark covering four video fusion tasks: multi-exposure, multi-focus, infrared-visible, and medical fusion. VF-Bench provides high-quality, well-aligned video pairs obtained through synthetic data generation and rigorous curation from existing datasets, with a unified evaluation protocol that jointly assesses the spatial quality and temporal consistency of video fusion. Extensive experiments show that UniVF achieves state-of-the-art results across all tasks on VF-Bench. Project page: https://vfbench.github.io.
>
---
#### [replaced 088] Implicit Neural Compression of Point Clouds
- **分类: cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.10433v2](http://arxiv.org/pdf/2412.10433v2)**

> **作者:** Hongning Ruan; Yulin Shao; Qianqian Yang; Liang Zhao; Zhaoyang Zhang; Dusit Niyato
>
> **摘要:** Point clouds have gained prominence across numerous applications due to their ability to accurately represent 3D objects and scenes. However, efficiently compressing unstructured, high-precision point cloud data remains a significant challenge. In this paper, we propose NeRC$^3$, a novel point cloud compression framework that leverages implicit neural representations (INRs) to encode both geometry and attributes of dense point clouds. Our approach employs two coordinate-based neural networks: one maps spatial coordinates to voxel occupancy, while the other maps occupied voxels to their attributes, thereby implicitly representing the geometry and attributes of a voxelized point cloud. The encoder quantizes and compresses network parameters alongside auxiliary information required for reconstruction, while the decoder reconstructs the original point cloud by inputting voxel coordinates into the neural networks. Furthermore, we extend our method to dynamic point cloud compression through techniques that reduce temporal redundancy, including a 4D spatio-temporal representation termed 4D-NeRC$^3$. Experimental results validate the effectiveness of our approach: For static point clouds, NeRC$^3$ outperforms octree-based G-PCC standard and existing INR-based methods. For dynamic point clouds, 4D-NeRC$^3$ achieves superior geometry compression performance compared to the latest G-PCC and V-PCC standards, while matching state-of-the-art learning-based methods. It also demonstrates competitive performance in joint geometry and attribute compression.
>
---
#### [replaced 089] Think With Videos For Agentic Long-Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10821v5](http://arxiv.org/pdf/2506.10821v5)**

> **作者:** Huaying Yuan; Zheng Liu; Junjie Zhou; Hongjin Qian; Yan Shu; Nicu Sebe; Ji-Rong Wen; Zhicheng Dou
>
> **摘要:** Long-video understanding~(LVU) is a challenging problem in computer vision. Existing methods either downsample frames for single-pass reasoning, sacrificing fine-grained details, or depend on textual reasoning over task-agnostic representations, hindering task-specific perception and exploration. In this paper, we propose VideoExplorer, a framework grounded in the principle of ``thinking with video'', which naturally intertwines planning, temporal grounding, and scalable perception into a coherent reasoning process. Rather than reasoning over a static context, VideoExplorer iteratively formulates sub-questions, locates relevant moments, and performs task-oriented, temporally scalable video understanding until reaching the final answer, enabling faithful, efficient, and interpretable reasoning. To address the lack of LVU training resources, we construct a long-video reasoning dataset using difficulty-adaptive sampling to ensure high-quality trajectories on complex tasks. Building on this dataset, we design a two-stage training pipeline: supervised trajectory initialization followed by trajectory-level preference optimization, encouraging adaptive temporal grounding and iterative information integration guided by downstream rewards. Extensive evaluations on popular long-video understanding and reasoning benchmarks demonstrate VideoExplorer's significant advantage over existing baselines, highlighting its robustness, adaptability, and efficiency. Our code is made publicly available in this repository(https://github.com/yhy-2000/VideoDeepResearch).
>
---
#### [replaced 090] ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22498v4](http://arxiv.org/pdf/2506.22498v4)**

> **作者:** Hao Liu; Yu Hu; Rakiba Rayhana; Ling Bai; Zheng Liu
>
> **摘要:** Bed-related falls remain a major source of injury in hospitals and long-term care facilities, yet many commercial alarms trigger only after a patient has already left the bed. We show that early bed-exit intent can be predicted using only one low-cost load cell mounted under a bed leg. The resulting load signals are first converted into a compact set of complementary images: an RGB line plot that preserves raw waveforms and three texture maps-recurrence plot, Markov transition field, and Gramian angular field-that expose higher-order dynamics. We introduce ViFusionTST, a dual-stream Swin Transformer that processes the line plot and texture maps in parallel and fuses them through cross-attention to learn data-driven modality weights. To provide a realistic benchmark, we collected six months of continuous data from 95 beds in a long-term-care facility. On this real-world dataset ViFusionTST reaches an accuracy of 0.885 and an F1 score of 0.794, surpassing recent 1D and 2D time-series baselines across F1, recall, accuracy, and AUPRC. The results demonstrate that image-based fusion of load-sensor signals for time series classification is a practical and effective solution for real-time, privacy-preserving fall prevention.
>
---
#### [replaced 091] NEBULA: Do We Evaluate Vision-Language-Action Agents Correctly?
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16263v2](http://arxiv.org/pdf/2510.16263v2)**

> **作者:** Jierui Peng; Yanyan Zhang; Yicheng Duan; Tuo Liang; Vipin Chaudhary; Yu Yin
>
> **备注:** Homepage: https://vulab-ai.github.io/NEBULA-Alpha/
>
> **摘要:** The evaluation of Vision-Language-Action (VLA) agents is hindered by the coarse, end-task success metric that fails to provide precise skill diagnosis or measure robustness to real-world perturbations. This challenge is exacerbated by a fragmented data landscape that impedes reproducible research and the development of generalist models. To address these limitations, we introduce NEBULA, a unified ecosystem for single-arm manipulation that enables diagnostic and reproducible evaluation. NEBULA features a novel dual-axis evaluation protocol that combines fine-grained capability tests for precise skill diagnosis with systematic stress tests that measure robustness. A standardized API and a large-scale, aggregated dataset are provided to reduce fragmentation and support cross-dataset training and fair comparison. Using NEBULA, we demonstrate that top-performing VLAs struggle with key capabilities such as spatial reasoning and dynamic adaptation, which are consistently obscured by conventional end-task success metrics. By measuring both what an agent can do and when it does so reliably, NEBULA provides a practical foundation for robust, general-purpose embodied agents.
>
---
#### [replaced 092] ITVTON: Virtual Try-On Diffusion Transformer Based on Integrated Image and Text
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16757v3](http://arxiv.org/pdf/2501.16757v3)**

> **作者:** Haifeng Ni; Ming Xu
>
> **备注:** Accepted by PRCV 2025
>
> **摘要:** Virtual try-on, which aims to seamlessly fit garments onto person images, has recently seen significant progress with diffusion-based models. However, existing methods commonly resort to duplicated backbones or additional image encoders to extract garment features, which increases computational overhead and network complexity. In this paper, we propose ITVTON, an efficient framework that leverages the Diffusion Transformer (DiT) as its single generator to improve image fidelity. By concatenating garment and person images along the width dimension and incorporating textual descriptions from both, ITVTON effectively captures garment-person interactions while preserving realism. To further reduce computational cost, we restrict training to the attention parameters within a single Diffusion Transformer (Single-DiT) block. Extensive experiments demonstrate that ITVTON surpasses baseline methods both qualitatively and quantitatively, setting a new standard for virtual try-on. Moreover, experiments on 10,257 image pairs from IGPair confirm its robustness in real-world scenarios.
>
---
#### [replaced 093] Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03197v3](http://arxiv.org/pdf/2506.03197v3)**

> **作者:** Baode Wang; Biao Wu; Weizhen Li; Meng Fang; Zuming Huang; Jun Huang; Haozhe Wang; Yanjie Liang; Ling Chen; Wei Chu; Yuan Qi
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Automated parsing of scanned documents into richly structured, machine-readable formats remains a critical bottleneck in Document AI, as traditional multi-stage pipelines suffer from error propagation and limited adaptability to diverse layouts. We introduce layoutRL, an end-to-end reinforcement learning framework that trains models to be explicitly layout-aware by optimizing a composite reward of normalized edit distance, paragraph count accuracy, and reading order preservation. Leveraging our newly released dataset, Infinity-Doc-55K, which combines 55K high-fidelity synthetic scanned document parsing data with expert-filtered real-world documents, we instantiate layoutRL in a vision-language-model-based parser called Infinity-Parser. Evaluated on English and Chinese benchmarks for OCR, table and formula extraction, and reading order detection, Infinity-Parser achieves new state-of-the-art performance in both accuracy and structural fidelity, outpacing specialist pipelines and general-purpose vision-language models. We will publicly release our code and dataset to accelerate progress in robust document understanding.
>
---
#### [replaced 094] MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.08567v3](http://arxiv.org/pdf/2510.08567v3)**

> **作者:** Tajamul Ashraf; Umair Nawaz; Abdelrahman M. Shaker; Rao Anwer; Philip Torr; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** We have come across a recent approach that has not been properly attributed at the time of submission and compared in a fair setting. Therefore, we would like to withdraw the paper to address these concerns
>
> **摘要:** Vision language models (VLMs) are increasingly deployed as controllers with access to external tools for complex reasoning and decision-making, yet their effectiveness remains limited by the scarcity of high-quality multimodal trajectories and the cost of manual annotation. We address this challenge with a vision-centric agent tuning framework that automatically synthesizes multimodal trajectories, generates step-wise preference pairs, and trains a VLM controller for robust tool-use reasoning. Our pipeline first constructs M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified trajectories, enabling imitation-based trajectory tuning. Building on this, we develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool reasoning. To achieve finer alignment, we further introduce Pref-X, a set of 11K automatically generated preference pairs, and optimize MATRIX on it via step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA, MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating scalable and effective multimodal tool use. Our data and code is avaliable at https://github.com/mbzuai-oryx/MATRIX.
>
---
#### [replaced 095] Monitoring morphometric drift in lifelong learning segmentation of the spinal cord
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01364v2](http://arxiv.org/pdf/2505.01364v2)**

> **作者:** Enamundram Naga Karthik; Sandrine Bédard; Jan Valošek; Christoph S. Aigner; Elise Bannier; Josef Bednařík; Virginie Callot; Anna Combes; Armin Curt; Gergely David; Falk Eippert; Lynn Farner; Michael G Fehlings; Patrick Freund; Tobias Granberg; Cristina Granziera; RHSCIR Network Imaging Group; Ulrike Horn; Tomáš Horák; Suzanne Humphreys; Markus Hupp; Anne Kerbrat; Nawal Kinany; Shannon Kolind; Petr Kudlička; Anna Lebret; Lisa Eunyoung Lee; Caterina Mainero; Allan R. Martin; Megan McGrath; Govind Nair; Kristin P. O'Grady; Jiwon Oh; Russell Ouellette; Nikolai Pfender; Dario Pfyffer; Pierre-François Pradat; Alexandre Prat; Emanuele Pravatà; Daniel S. Reich; Ilaria Ricchi; Naama Rotem-Kohavi; Simon Schading-Sassenhausen; Maryam Seif; Andrew Smith; Seth A Smith; Grace Sweeney; Roger Tam; Anthony Traboulsee; Constantina Andrada Treaba; Charidimos Tsagkas; Zachary Vavasour; Dimitri Van De Ville; Kenneth Arnold Weber II; Sarath Chandar; Julien Cohen-Adad
>
> **备注:** Under review (after 1st round of revision) at Imaging Neuroscience journal
>
> **摘要:** Morphometric measures derived from spinal cord segmentations can serve as diagnostic and prognostic biomarkers in neurological diseases and injuries affecting the spinal cord. While robust, automatic segmentation methods to a wide variety of contrasts and pathologies have been developed over the past few years, whether their predictions are stable as the model is updated using new datasets has not been assessed. This is particularly important for deriving normative values from healthy participants. In this study, we present a spinal cord segmentation model trained on a multisite $(n=75)$ dataset, including 9 different MRI contrasts and several spinal cord pathologies. We also introduce a lifelong learning framework to automatically monitor the morphometric drift as the model is updated using additional datasets. The framework is triggered by an automatic GitHub Actions workflow every time a new model is created, recording the morphometric values derived from the model's predictions over time. As a real-world application of the proposed framework, we employed the spinal cord segmentation model to update a recently-introduced normative database of healthy participants containing commonly used measures of spinal cord morphometry. Results showed that: (i) our model outperforms previous versions and pathology-specific models on challenging lumbar spinal cord cases, achieving an average Dice score of $0.95 \pm 0.03$; (ii) the automatic workflow for monitoring morphometric drift provides a quick feedback loop for developing future segmentation models; and (iii) the scaling factor required to update the database of morphometric measures is nearly constant among slices across the given vertebral levels, showing minimum drift between the current and previous versions of the model monitored by the framework. The code and model are open-source and accessible via Spinal Cord Toolbox v7.0.
>
---
