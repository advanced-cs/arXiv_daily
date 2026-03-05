# 计算机视觉 cs.CV

- **最新发布 135 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination
- **分类: cs.CV**

- **简介: 该论文属于少样本开放集动作识别任务，解决真实场景中开放集识别问题。提出特征残差鉴别器，提升未知动作拒绝能力，同时保持封闭集准确率。**

- **链接: [https://arxiv.org/pdf/2603.04125](https://arxiv.org/pdf/2603.04125)**

> **作者:** Stefano Berti; Giulia Pasquale; Lorenzo Natale
>
> **摘要:** Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: this https URL.
>
---
#### [new 002] Proact-VL: A Proactive VideoLLM for Real-Time AI Companions
- **分类: cs.CV**

- **简介: 该论文属于实时交互任务，旨在解决AI伴侣的低延迟响应、自主决策和内容控制问题。提出Proact-VL框架，提升实时互动性能。**

- **链接: [https://arxiv.org/pdf/2603.03447](https://arxiv.org/pdf/2603.03447)**

> **作者:** Weicai Yan; Yuhong Dai; Qi Ran; Haodong Li; Wang Lin; Hao Liao; Xing Xie; Tao Jin; Jianxun Lian
>
> **摘要:** Proactive and real-time interactive experiences are essential for human-like AI companions, yet face three key challenges: (1) achieving low-latency inference under continuous streaming inputs, (2) autonomously deciding when to respond, and (3) controlling both quality and quantity of generated content to meet real-time constraints. In this work, we instantiate AI companions through two gaming scenarios, commentator and guide, selected for their suitability for automatic evaluation. We introduce the Live Gaming Benchmark, a large-scale dataset with three representative scenarios: solo commentary, co-commentary, and user guidance, and present Proact-VL, a general framework that shapes multimodal language models into proactive, real-time interactive agents capable of human-like environment perception and interaction. Extensive experiments show Proact-VL achieves superior response latency and quality while maintaining strong video understanding capabilities, demonstrating its practicality for real-time interactive applications.
>
---
#### [new 003] LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing
- **分类: cs.CV**

- **简介: 该论文属于隐私保护图像处理任务，旨在解决高维图像数据应用LDP时的效用损失问题。通过位平面分解和优化策略，提出LDP-Slicing框架，在保证隐私的同时提升图像实用性。**

- **链接: [https://arxiv.org/pdf/2603.03711](https://arxiv.org/pdf/2603.03711)**

> **作者:** Yuanming Cao; Chengqi Li; Wenbo He
>
> **摘要:** Local Differential Privacy (LDP) is the gold standard trust model for privacy-preserving machine learning by guaranteeing privacy at the data source. However, its application to image data has long been considered impractical due to the high dimensionality of pixel space. Canonical LDP mechanisms are designed for low-dimensional data, resulting in severe utility degradation when applied to high-dimensional pixel spaces. This paper demonstrates that this utility loss is not inherent to LDP, but from its application to an inappropriate data representation. We introduce LDP-Slicing, a lightweight, training-free framework that resolves this domain mismatch. Our key insight is to decompose pixel values into a sequence of binary bit-planes. This transformation allows us to apply the LDP mechanism directly to the bit-level representation. To further strengthen privacy and preserve utility, we integrate a perceptual obfuscation module that mitigates human-perceivable leakage and an optimization-based privacy budget allocation strategy. This pipeline satisfies rigorous pixel-level $\varepsilon$-LDP while producing images that retain high utility for downstream tasks. Extensive experiments on face recognition and image classification demonstrate that LDP-Slicing outperforms existing DP/LDP baselines under comparable privacy budgets, with negligible computational overhead.
>
---
#### [new 004] EvoPrune: Early-Stage Visual Token Pruning for Efficient MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大模型任务，旨在提升MLLM的推理效率。针对视觉token数量爆炸问题，提出EvoPrune方法，在视觉编码阶段进行早期剪枝，保留关键信息，显著提升速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.03681](https://arxiv.org/pdf/2603.03681)**

> **作者:** Yuhao Chen; Bin Shan; Xin Ye; Cheng Chen
>
> **备注:** 16 pages, 4 figures, 3 tables
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown strong performance in vision-language tasks, but their inference efficiency is severely limited by the exponential growth of visual tokens in complex scenarios such as high-resolution images and videos. Existing visual token pruning methods mainly operate after visual encoding, overlooking the substantial computational cost incurred during the encoding stage. To address this issue, we propose EvoPrune, an early-stage visual token pruning method for MLLMs that performs pruning directly during visual encoding. Specifically, EvoPrune employs a layer-wise pruning strategy guided by token similarity, diversity, and attention-based importance to retain the most informative visual tokens at selected encoding layers. Extensive experiments on image and video benchmarks validate the effectiveness of EvoPrune. In particular, on the VideoMME dataset, EvoPrune achieves 2$\times$ inference speedup with less than 1% performance degradation, demonstrating its potential for latency-sensitive MLLM deployment.
>
---
#### [new 005] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks
- **分类: cs.CV**

- **简介: 该论文属于图像美学评估任务，解决细粒度美学差异难以区分的问题。构建了FGAesthetics数据集，并提出FGAesQ框架，通过相对排名学习判别性评分。**

- **链接: [https://arxiv.org/pdf/2603.03907](https://arxiv.org/pdf/2603.03907)**

> **作者:** Zhichao Yang; Jianjie Wang; Zhixianhe Zhang; Pangu Xie; Xiangfei Sheng; Pengfei Chen; Leida Li
>
> **备注:** The paper has been accepted by CVPR 2026
>
> **摘要:** Image aesthetic assessment (IAA) has extensive applications in content creation, album management, and recommendation systems, etc. In such applications, it is commonly needed to pick out the most aesthetically pleasing image from a series of images with subtle aesthetic variations, a topic we refer to as fine-grained IAA. Unfortunately, state-of-the-art IAA models are typically designed for coarse-grained evaluation, where images with notable aesthetic differences are evaluated independently on an absolute scale. These models are inherently limited in discriminating fine-grained aesthetic differences. To address the dilemma, we contribute FGAesthetics, a fine-grained IAA database with 32,217 images organized into 10,028 series, which are sourced from diverse categories including Natural, AIGC, and Cropping. Annotations are collected via pairwise comparisons within each series. We also devise Series Refinement and Rank Calibration to ensure the reliability of data and labels. Based on FGAesthetics, we further propose FGAesQ, a novel IAA framework that learns discriminative aesthetic scores from relative ranks through Difference-preserved Tokenization (DiffToken), Comparative Text-assisted Alignment (CTAlign), and Rank-aware Regression (RankReg). FGAesQ enables accurate aesthetic assessment in fine-grained scenarios while still maintains competitive performance in coarse-grained evaluation. Extensive experiments and comparisons demonstrate the superiority of the proposed method.
>
---
#### [new 006] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation
- **分类: cs.CV**

- **简介: 该论文提出DAGE，用于视频几何估计任务，解决高分辨率和长序列下几何与相机姿态估计的挑战。通过双流架构分离全局一致性和细节，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.03744](https://arxiv.org/pdf/2603.03744)**

> **作者:** Tuan Duc Ngo; Jiahui Huang; Seoung Wug Oh; Kevin Blackburn-Matzen; Evangelos Kalogerakis; Chuang Gan; Joon-Young Lee
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Estimating accurate, view-consistent geometry and camera poses from uncalibrated multi-view/video inputs remains challenging - especially at high spatial resolutions and over long sequences. We present DAGE, a dual-stream transformer whose main novelty is to disentangle global coherence from fine detail. A low-resolution stream operates on aggressively downsampled frames with alternating frame/global attention to build a view-consistent representation and estimate cameras efficiently, while a high-resolution stream processes the original images per-frame to preserve sharp boundaries and small structures. A lightweight adapter fuses these streams via cross-attention, injecting global context without disturbing the pretrained single-frame pathway. This design scales resolution and clip length independently, supports inputs up to 2K, and maintains practical inference cost. DAGE delivers sharp depth/pointmaps, strong cross-view consistency, and accurate poses, establishing new state-of-the-art results for video geometry estimation and multi-view reconstruction.
>
---
#### [new 007] Mask-Guided Attention Regulation for Anatomically Consistent Counterfactual CXR Synthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决扩散模型在胸部X光片反事实生成中的结构漂移和病灶表达不稳定问题。通过注意力调节机制提升生成的解剖一致性和病灶可控性。**

- **链接: [https://arxiv.org/pdf/2603.04130](https://arxiv.org/pdf/2603.04130)**

> **作者:** Zichun Zhang; Weizhi Nie; Honglin Guo; Yuting Su
>
> **摘要:** Counterfactual generation for chest X-rays (CXR) aims to simulate plausible pathological changes while preserving patient-specific anatomy. However, diffusion-based editing methods often suffer from structural drift, where stable anatomical semantics propagate globally through attention and distort non-target regions, and unstable pathology expression, since subtle and localized lesions induce weak and noisy conditioning signals. We present an inference-time attention regulation framework for reliable counterfactual CXR synthesis. An anatomy-aware attention regularization module gates self-attention and anatomy-token cross-attention with organ masks, confining structural interactions to anatomical ROIs and reducing unintended distortions. A pathology-guided module enhances pathology-token cross-attention within target lung regions during early denoising and performs lightweight latent corrections driven by an attention-concentration energy, enabling controllable lesion localization and extent. Extensive evaluations on CXR datasets show improved anatomical consistency and more precise, controllable pathological edits compared with standard diffusion editing, supporting localized counterfactual analysis and data augmentation for downstream tasks.
>
---
#### [new 008] InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频叙事一致性、多角色过渡和可扩展性问题，提出新框架与数据集，提升背景和角色一致性。**

- **链接: [https://arxiv.org/pdf/2603.03646](https://arxiv.org/pdf/2603.03646)**

> **作者:** Mohamed Elmoghany; Liangbing Zhao; Xiaoqian Shen; Subhojyoti Mukherjee; Yang Zhou; Gang Wu; Viet Dac Lai; Seunghyun Yoon; Ryan Rossi; Abdullah Rashwan; Puneet Mathur; Varun Manjunatha; Daksh Dangi; Chien Nguyen; Nedim Lipka; Trung Bui; Krishna Kumar Singh; Ruiyi Zhang; Xiaolei Huang; Jaemin Cho; Yu Wang; Namyong Park; Zhengzhong Tu; Hongjie Chen; Hoda Eldardiry; Nesreen Ahmed; Thien Nguyen; Dinesh Manocha; Mohamed Elhoseiny; Franck Dernoncourt
>
> **摘要:** Generating long-form storytelling videos with consistent visual narratives remains a significant challenge in video synthesis. We present a novel framework, dataset, and a model that address three critical limitations: background consistency across shots, seamless multi-subject shot-to-shot transitions, and scalability to hour-long narratives. Our approach introduces a background-consistent generation pipeline that maintains visual coherence across scenes while preserving character identity and spatial relationships. We further propose a transition-aware video synthesis module that generates smooth shot transitions for complex scenarios involving multiple subjects entering or exiting frames, going beyond the single-subject limitations of prior work. To support this, we contribute with a synthetic dataset of 10,000 multi-subject transition sequences covering underrepresented dynamic scene compositions. On VBench, InfinityStory achieves the highest Background Consistency (88.94), highest Subject Consistency (82.11), and the best overall average rank (2.80), showing improved stability, smoother transitions, and better temporal coherence.
>
---
#### [new 009] Understanding Sources of Demographic Predictability in Brain MRI via Disentangling Anatomy and Contrast
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决AI系统中因脑MRI数据导致的年龄、性别等人口统计属性预测偏差问题。通过解耦解剖结构与成像对比，分析预测信号来源，提出有效缓解策略。**

- **链接: [https://arxiv.org/pdf/2603.04113](https://arxiv.org/pdf/2603.04113)**

> **作者:** Mehmet Yigit Avci; Akshit Achara; Andrew King; Jorge Cardoso
>
> **摘要:** Demographic attributes such as age, sex, and race can be predicted from medical images, raising concerns about bias in clinical AI systems. In brain MRI, this signal may arise from anatomical variation, acquisition-dependent contrast differences, or both, yet these sources remain entangled in conventional analyses. Without disentangling them, mitigation strategies risk failing to address the underlying causes. We propose a controlled framework based on disentangled representation learning, decomposing brain MRI into anatomy-focused representations that suppress acquisition influence and contrast embeddings that capture acquisition-dependent characteristics. Training predictive models for age, sex, and race on full images, anatomical representations, and contrast-only embeddings allows us to quantify the relative contributions of structure and acquisition to the demographic signal. Across three datasets and multiple MRI sequences, we find that demographic predictability is primarily rooted in anatomical variation: anatomy-focused representations largely preserve the performance of models trained on raw images. Contrast-only embeddings retain a weaker but systematic signal that is dataset-specific and does not generalise across sites. These findings suggest that effective mitigation must explicitly account for the distinct anatomical and acquisition-dependent origins of the demographic signal, ensuring that any bias reduction generalizes robustly across domains.
>
---
#### [new 010] Efficient Point Cloud Processing with High-Dimensional Positional Encoding and Non-Local MLPs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云处理任务，旨在提升MLP模型的效率与效果。提出HPE模块和非局部MLP，优化特征提取，构建HPENets网络，在多个数据集上取得更好性能。**

- **链接: [https://arxiv.org/pdf/2603.04099](https://arxiv.org/pdf/2603.04099)**

> **作者:** Yanmei Zou; Hongshan Yu; Yaonan Wang; Zhengeng Yang; Xieyuanli Chen; Kailun Yang; Naveed Akhtar
>
> **备注:** Accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Source code is available at this https URL
>
> **摘要:** Multi-Layer Perceptron (MLP) models are the foundation of contemporary point cloud processing. However, their complex network architectures obscure the source of their strength and limit the application of these models. In this article, we develop a two-stage abstraction and refinement (ABS-REF) view for modular feature extraction in point cloud processing. This view elucidates that whereas the early models focused on ABS stages, the more recent techniques devise sophisticated REF stages to attain performance advantages. Then, we propose a High-dimensional Positional Encoding (HPE) module to explicitly utilize intrinsic positional information, extending the ``positional encoding'' concept from Transformer literature. HPE can be readily deployed in MLP-based architectures and is compatible with transformer-based methods. Within our ABS-REF view, we rethink local aggregation in MLP-based methods and propose replacing time-consuming local MLP operations, which are used to capture local relationships among neighbors. Instead, we use non-local MLPs for efficient non-local information updates, combined with the proposed HPE for effective local information representation. We leverage our modules to develop HPENets, a suite of MLP networks that follow the ABS-REF paradigm, incorporating a scalable HPE-based REF stage. Extensive experiments on seven public datasets across four different tasks show that HPENets deliver a strong balance between efficiency and effectiveness. Notably, HPENet surpasses PointNeXt, a strong MLP-based counterpart, by 1.1% mAcc, 4.0% mIoU, 1.8% mIoU, and 0.2% Cls. mIoU, with only 50.0%, 21.5%, 23.1%, 44.4% of FLOPs on ScanObjectNN, S3DIS, ScanNet, and ShapeNetPart, respectively. Source code is available at this https URL.
>
---
#### [new 011] Machine Pareidolia: Protecting Facial Image with Emotional Editing
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于面部隐私保护任务，旨在解决FR系统滥用导致的隐私泄露问题。通过情感编辑实现身份伪装，提升保护效果与适用性。**

- **链接: [https://arxiv.org/pdf/2603.03665](https://arxiv.org/pdf/2603.03665)**

> **作者:** Binh M. Le; Simon S. Woo
>
> **备注:** Proceedings of the AAAI Conference on Artificial Intelligence 40
>
> **摘要:** The proliferation of facial recognition (FR) systems has raised privacy concerns in the digital realm, as malicious uses of FR models pose a significant threat. Traditional countermeasures, such as makeup style transfer, have suffered from low transferability in black-box settings and limited applicability across various demographic groups, including males and individuals with darker skin tones. To address these challenges, we introduce a novel facial privacy protection method, dubbed \textbf{MAP}, a pioneering approach that employs human emotion modifications to disguise original identities as target identities in facial images. Our method uniquely fine-tunes a score network to learn dual objectives, target identity and human expression, which are jointly optimized through gradient projection to ensure convergence at a shared local optimum. Additionally, we enhance the perceptual quality of protected images by applying local smoothness regularization and optimizing the score matching loss within our network. Empirical experiments demonstrate that our innovative approach surpasses previous baselines, including noise-based, makeup-based, and freeform attribute methods, in both qualitative fidelity and quantitative metrics. Furthermore, MAP proves its effectiveness against an online FR API and shows advanced adaptability in uncommon photographic scenarios.
>
---
#### [new 012] Hold-One-Shot-Out (HOSO) for Validation-Free Few-Shot CLIP Adapters
- **分类: cs.CV**

- **简介: 该论文属于少样本学习任务，解决CLIP适配中混合比例选择的问题。提出HOSO方法，在无需验证集的情况下，通过单样本保留集学习混合比例，提升适配效果。**

- **链接: [https://arxiv.org/pdf/2603.04341](https://arxiv.org/pdf/2603.04341)**

> **作者:** Chris Vorster; Mayug Maniparambil; Noel E. O'Connor; Noel Murphy; Derek Molloy
>
> **摘要:** In many CLIP adaptation methods, a blending ratio hyperparameter controls the trade-off between general pretrained CLIP knowledge and the limited, dataset-specific supervision from the few-shot cases. Most few-shot CLIP adaptation techniques report results by ablation of the blending ratio on the test set or require additional validation sets to select the blending ratio per dataset, and thus are not strictly few-shot. We present a simple, validation-free method for learning the blending ratio in CLIP adaptation. Hold-One-Shot-Out (HOSO) presents a novel approach for CLIP-Adapter-style methods to compete in the newly established validation-free setting. CLIP-Adapter with HOSO (HOSO-Adapter) learns the blending ratio using a one-shot, hold-out set, while the adapter trains on the remaining few-shot support examples. Under the validation-free few-shot protocol, HOSO-Adapter outperforms the CLIP-Adapter baseline by more than 4 percentage points on average across 11 standard few-shot datasets. Interestingly, in the 8- and 16-shot settings, HOSO-Adapter outperforms CLIP-Adapter even with the optimal blending ratio selected on the test set. Ablation studies validate the use of a one-shot hold-out mechanism, decoupled training, and improvements over the naively learnt blending ratio baseline. Code is released here: this https URL
>
---
#### [new 013] All-in-One Image Restoration via Causal-Deconfounding Wavelet-Disentangled Prompt Network
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决传统方法存储成本高、依赖已知退化模式的问题。提出CWP-Net模型，通过解耦退化与语义特征提升修复效果。**

- **链接: [https://arxiv.org/pdf/2603.03839](https://arxiv.org/pdf/2603.03839)**

> **作者:** Bingnan Wang; Bin Qin; Jiangmeng Li; Fanjiang Xu; Fuchun Sun; Hui Xiong
>
> **备注:** Accepted by IEEE TIP 2026
>
> **摘要:** Image restoration represents a promising approach for addressing the inherent defects of image content distortion. Standard image restoration approaches suffer from high storage cost and the requirement towards the known degradation pattern, including type and degree, which can barely be satisfied in dynamic practical scenarios. In contrast, all-in-one image restoration (AiOIR) eliminates multiple degradations within a unified model to circumvent the aforementioned issues. However, according to our causal analysis, we disclose that two significant defects still exacerbate the effectiveness and generalization of AiOIR models: 1) the spurious correlation between non-degradation semantic features and degradation patterns; 2) the biased estimation of degradation patterns. To obtain the true causation between degraded images and restored images, we propose Causal-deconfounding Wavelet-disentangled Prompt Network (CWP-Net) to perform effective AiOIR. CWP-Net introduces two modules for decoupling, i.e., wavelet attention module of encoder and wavelet attention module of decoder. These modules explicitly disentangle the degradation and semantic features to tackle the issue of spurious correlation. To address the issue stemming from the biased estimation of degradation patterns, CWP-Net leverages a wavelet prompt block to generate the alternative variable for causal deconfounding. Extensive experiments on two all-in-one settings prove the effectiveness and superior performance of our proposed CWP-Net over the state-of-the-art AiOIR methods.
>
---
#### [new 014] A Hypertoroidal Covering for Perfect Color Equivariance
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决颜色变换下神经网络性能下降的问题。提出一种真正等变的颜色架构，通过将颜色值映射到圆上提升模型鲁棒性和准确性。**

- **链接: [https://arxiv.org/pdf/2603.04256](https://arxiv.org/pdf/2603.04256)**

> **作者:** Yulong Yang; Zhikun Xu; Yaojun Li; Christine Allen-Blanchette
>
> **摘要:** When the color distribution of input images changes at inference, the performance of conventional neural network architectures drops considerably. A few researchers have begun to incorporate prior knowledge of color geometry in neural network design. These color equivariant architectures have modeled hue variation with 2D rotations, and saturation and luminance transformations as 1D translations. While this approach improves neural network robustness to color variations in a number of contexts, we find that approximating saturation and luminance (interval valued quantities) as 1D translations introduces appreciable artifacts. In this paper, we introduce a color equivariant architecture that is truly equivariant. Instead of approximating the interval with the real line, we lift values on the interval to values on the circle (a double-cover) and build equivariant representations there. Our approach resolves the approximation artifacts of previous methods, improves interpretability and generalizability, and achieves better predictive performance than conventional and equivariant baselines on tasks such as fine-grained classification and medical imaging tasks. Going beyond the context of color, we show that our proposed lifting can also extend to geometric transformations such as scale.
>
---
#### [new 015] RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation
- **分类: cs.CV**

- **简介: 该论文属于RGBT跟踪任务，解决目标建模不适应外观变化和模态差异的问题。引入语言描述，提出RAGTrack框架，融合多模态信息提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.03617](https://arxiv.org/pdf/2603.03617)**

> **作者:** Hao Li; Yuhao Wang; Wenning Hao; Pingping Zhang; Dong Wang; Huchuan Lu
>
> **备注:** This work is accepted by CVPR2026. More modifications may be performed
>
> **摘要:** RGB-Thermal (RGBT) tracking aims to achieve robust object localization across diverse environmental conditions by fusing visible and thermal infrared modalities. However, existing RGBT trackers rely solely on initial-frame visual information for target modeling, failing to adapt to appearance variations due to the absence of language guidance. Furthermore, current methods suffer from redundant search regions and heterogeneous modality gaps, causing background distraction. To address these issues, we first introduce textual descriptions into RGBT tracking benchmarks. This is accomplished through a pipeline that leverages Multi-modal Large Language Models (MLLMs) to automatically produce texual annotations. Afterwards, we propose RAGTrack, a novel Retrieval-Augmented Generation framework for robust RGBT tracking. To this end, we introduce a Multi-modal Transformer Encoder (MTE) for unified visual-language modeling. Then, we design an Adaptive Token Fusion (ATF) to select target-relevant tokens and perform channel exchanges based on cross-modal correlations, mitigating search redundancies and modality gaps. Finally, we propose a Context-aware Reasoning Module (CRM) to maintain a dynamic knowledge base and employ a Retrieval-Augmented Generation (RAG) to enable temporal linguistic reasoning for robust target modeling. Extensive experiments on four RGBT benchmarks demonstrate that our framework achieves state-of-the-art performance across various challenging scenarios. The source code is available this https URL.
>
---
#### [new 016] Geographically-Weighted Weakly Supervised Bayesian High-Resolution Transformer for 200m Resolution Pan-Arctic Sea Ice Concentration Mapping and Uncertainty Estimation using Sentinel-1, RCM, and AMSR2 Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于北极海冰浓度制图任务，旨在解决海冰特征提取难、标签不精确、模型不确定性及数据异质性问题。通过设计改进的Transformer模型和融合多源数据提升精度与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2603.03503](https://arxiv.org/pdf/2603.03503)**

> **作者:** Mabel Heffring; Lincoln Linlin Xu
>
> **备注:** 23 pages, 20 figures
>
> **摘要:** Although high-resolution mapping of pan-Arctic sea ice with reliable corresponding uncertainty is essential for operational sea ice concentration (SIC) charting, it is a difficult task due to key challenges, such as the subtle nature of ice signature features, inexact SIC labels, model uncertainty, and data heterogeneity. This study presents a novel Bayesian High-Resolution Transformer approach for 200 meter resolution pan-Arctic SIC mapping and uncertainty quantification using Sentinel-1, RADARSAT Constellation Mission (RCM), and Advanced Microwave Scanning Radiometer 2 (AMSR2) data. First, to improve small and subtle sea ice feature (e.g., cracks/leads, ponds, and ice floes) extraction, we design a novel high-resolution Transformer model with both global and local modules that can better discern the subtle differences in sea ice patterns. Second, to address low-resolution and inexact SIC labels, we design a geographically-weighted weakly supervised loss function to supervise the model at region level instead of pixel level, and to prioritize pure open water and ice pack signatures while mitigating the impact of ambiguity in the marginal ice zone (MIZ). Third, to improve uncertainty quantification, we design a Bayesian extension of the proposed Transformer model, treating its parameters as random variables to more effectively capture uncertainties. Fourth, to address data heterogeneity, we fuse three different data types (Sentinel-1, RCM, and AMSR2) at decision-level to improve both SIC mapping and uncertainty quantification. The proposed approach is evaluated under pan-Arctic minimum-extent conditions in 2021 and 2025. Results demonstrate that the proposed model achieves 0.70 overall feature detection accuracy using Sentinel-1 data, while also preserving pan-Arctic SIC patterns (Sentinel-1 R\textsuperscript{2} = 0.90 relative to the ARTIST Sea Ice product).
>
---
#### [new 017] Motion Manipulation via Unsupervised Keypoint Positioning in Face Animation
- **分类: cs.CV**

- **简介: 该论文属于面部动画任务，旨在解决现有方法无法有效分离身份与运动信息的问题。通过自监督学习和变分自编码器，实现可控的面部表情生成与运动操控。**

- **链接: [https://arxiv.org/pdf/2603.04302](https://arxiv.org/pdf/2603.04302)**

> **作者:** Hong Li; Boyu Liu; Xuhui Liu; Baochang Zhang
>
> **备注:** 19 pages, 15 figures
>
> **摘要:** Face animation deals with controlling and generating facial features with a wide range of applications. The methods based on unsupervised keypoint positioning can produce realistic and detailed virtual portraits. However, they cannot achieve controllable face generation since the existing keypoint decomposition pipelines fail to fully decouple identity semantics and intertwined motion information (e.g., rotation, translation, and expression). To address these issues, we present a new method, Motion Manipulation via unsupervised keypoint positioning in Face Animation (MMFA). We first introduce self-supervised representation learning to encode and decode expressions in the latent feature space and decouple them from other motion information. Secondly, we propose a new way to compute keypoints aiming to achieve arbitrary motion control. Moreover, we design a variational autoencoder to map expression features to a continuous Gaussian distribution, allowing us for the first time to interpolate facial expressions in an unsupervised framework. We have conducted extensive experiments on publicly available datasets to validate the effectiveness of MMFA, which show that MMFA offers pronounced advantages over prior arts in creating realistic animation and manipulating face motion.
>
---
#### [new 018] Bridging Human Evaluation to Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光图像融合任务，旨在解决现有方法与人类视觉偏好不一致的问题。通过构建人因评价数据集和奖励模型，提升融合结果的感知质量。**

- **链接: [https://arxiv.org/pdf/2603.03871](https://arxiv.org/pdf/2603.03871)**

> **作者:** Jinyuan Liu; Xingyuan Li; Qingyun Mei; Haoyuan Xu; Zhiying Jiang; Long Ma; Risheng Liu; Xin Fan
>
> **摘要:** Infrared and visible image fusion (IVIF) integrates complementary modalities to enhance scene perception. Current methods predominantly focus on optimizing handcrafted losses and objective metrics, often resulting in fusion outcomes that do not align with human visual preferences. This challenge is further exacerbated by the ill-posed nature of IVIF, which severely limits its effectiveness in human perceptual environments such as security surveillance and driver assistance systems. To address these limitations, we propose a feedback reinforcement framework that bridges human evaluation to infrared and visible image fusion. To address the lack of human-centric evaluation metrics and data, we introduce the first large-scale human feedback dataset for IVIF, containing multidimensional subjective scores and artifact annotations, and enriched by a fine-tuned large language model with expert review. Based on this dataset, we design a domain-specific reward function and train a reward model to quantify perceptual quality. Guided by this reward, we fine-tune the fusion network through Group Relative Policy Optimization, achieving state-of-the-art performance that better aligns fused images with human aesthetics. Code is available at this https URL.
>
---
#### [new 019] MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis in Cattle Re-Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动物重识别任务，解决视角变化带来的挑战。提出MOO数据集，包含1000头牛的128个视角图像，分析视角影响并验证模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04314](https://arxiv.org/pdf/2603.04314)**

> **作者:** William Grolleau; Achraf Chaouch; Astrid Sabourin; Guillaume Lapouge; Catherine Achard
>
> **摘要:** Animal re-identification (ReID) faces critical challenges due to viewpoint variations, particularly in Aerial-Ground (AG-ReID) settings where models must match individuals across drastic elevation changes. However, existing datasets lack the precise angular annotations required to systematically analyze these geometric variations. To address this, we introduce the Multi-view Oriented Observation (MOO) dataset, a large-scale synthetic AG-ReID dataset of $1,000$ cattle individuals captured from $128$ uniformly sampled viewpoints ($128,000$ annotated images). Using this controlled dataset, we quantify the influence of elevation and identify a critical elevation threshold, above which models generalize significantly better to unseen views. Finally, we validate the transferability to real-world applications in both zero-shot and supervised settings, demonstrating performance gains across four real-world cattle datasets and confirming that synthetic geometric priors effectively bridge the domain gap. Collectively, this dataset and analysis lay the foundation for future model development in cross-view animal ReID. MOO is publicly available at this https URL.
>
---
#### [new 020] FocusGraph: Graph-Structured Frame Selection for Embodied Long Video Question Answering
- **分类: cs.CV**

- **简介: 该论文属于长视频问答任务，旨在解决多模态大模型在处理长视频时响应质量下降和推理时间过长的问题。提出FocusGraph框架，通过关键帧选择提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2603.04349](https://arxiv.org/pdf/2603.04349)**

> **作者:** Tatiana Zemskova; Solomon Andryushenko; Ilya Obrubov; Viktoriia Khoruzhaia; Ekaterina Eroshenko; Ekaterina Derevyanka; Dmitry Yudin
>
> **摘要:** The ability to understand long videos is vital for embodied intelligent agents, because their effectiveness depends on how well they can accumulate, organize, and leverage long-horizon perceptual memories. Recently, multimodal LLMs have been gaining popularity for solving the long video understanding task due to their general ability to understand natural language and to leverage world knowledge. However, as the number of frames provided to an MLLM increases, the quality of its responses tends to degrade, and inference time grows. Therefore, when using MLLMs for long video understanding, a crucial step is selecting key frames from the video to answer user queries. In this work, we develop FocusGraph, a framework for keyframe selection for question answering over long egocentric videos. It leverages a lightweight trainable Scene-Caption LLM Selector that selects query-relevant clips based on their graph-based captions, and a training-free method for selecting keyframes from these clips. Unlike existing methods, the proposed Scene-Caption LLM Selector does not rely on the original sequence of low-resolution frames; instead, it operates on a compact textual representation of the scene. We then design a training-free Patch-wise Sparse-Flow Retention (PSFR) method to select keyframes from the resulting sequence of clips, which are fed into an MLLM to produce the final answer. Together, these components enable FocusGraph to achieve state-of-the-art results on challenging egocentric long-video question answering benchmarks, including FindingDory and HourVideo, while significantly reducing inference time relative to baseline approaches.
>
---
#### [new 021] A novel network for classification of cuneiform tablet metadata
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种新型网络用于分类楔形文字泥板元数据，解决专家不足与数据量大的问题。采用受卷积启发的架构，结合局部与全局信息，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2603.03892](https://arxiv.org/pdf/2603.03892)**

> **作者:** Frederik Hagelskjær
>
> **备注:** Point cloud, deep learning, cuneiform
>
> **摘要:** In this paper, we present a network structure for classifying metadata of cuneiform tablets. The problem is of practical importance, as the size of the existing corpus far exceeds the number of experts available to analyze it. But the task is made difficult by the combination of limited annotated datasets and the high-resolution point-cloud representation of each tablet. To address this, we develop a convolution-inspired architecture that gradually down-scales the point cloud while integrating local neighbor information. The final down-scaled point cloud is then processed by computing neighbors in the feature space to include global information. Our method is compared with the state-of-the-art transformer-based network Point-BERT, and consistently obtains the best performance. Source code and datasets will be released at publication.
>
---
#### [new 022] CoRe-BT: A Multimodal Radiology-Pathology-Text Benchmark for Robust Brain Tumor Typing
- **分类: cs.CV**

- **简介: 该论文提出CoRe-BT基准，用于脑肿瘤分类任务，解决多模态数据缺失下的精准诊断问题，整合MRI、病理和文本信息进行模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.03618](https://arxiv.org/pdf/2603.03618)**

> **作者:** Juampablo E. Heras Rivera; Daniel K. Low; Xavier Xiong; Jacob J. Ruzevick; Daniel D. Child; Wen-wai Yim; Mehmet Kurt; Asma Ben Abacha
>
> **备注:** Under review, MICCAI 2026
>
> **摘要:** Accurate brain tumor typing requires integrating heterogeneous clinical evidence, including magnetic resonance imaging (MRI), histopathology, and pathology reports, which are often incomplete at the time of diagnosis. We introduce CoRe-BT, a cross-modal radiology-pathology-text benchmark for brain tumor typing, designed to study robust multimodal learning under missing modality conditions. The dataset comprises 310 patients with multi-sequence brain MRI (T1, T1c, T2, FLAIR), including 95 cases with paired H&E-stained whole-slide pathology images and pathology reports. All cases are annotated with tumor type and grade, and MRI volumes include expert-annotated tumor masks, enabling both region-aware modeling and auxiliary learning tasks. Tumors are categorized into six clinically relevant classes capturing the heterogeneity of common and rare glioma subtypes. We evaluate tumor typing under variable modality availability by comparing MRI-only models with multimodal approaches that incorporate pathology information when present. Baseline experiments demonstrate the feasibility of multimodal fusion and highlight complementary modality contributions across clinically relevant typing tasks. CoRe-BT provides a grounded testbed for advancing multimodal glioma typing and representation learning in realistic scenarios with incomplete clinical data.
>
---
#### [new 023] From Misclassifications to Outliers: Joint Reliability Assessment in Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于分类任务，解决可靠分类器构建问题，提出联合评估框架和新指标，提升对误分类和分布外输入的检测能力。**

- **链接: [https://arxiv.org/pdf/2603.03903](https://arxiv.org/pdf/2603.03903)**

> **作者:** Yang Li; Youyang Sha; Yinzhi Wang; Timothy Hospedales; Xi Shen; Shell Xu Hu; Xuanlong Yu
>
> **备注:** 15 pages, 3 figures. The source code is publicly available at this https URL
>
> **摘要:** Building reliable classifiers is a fundamental challenge for deploying machine learning in real-world applications. A reliable system should not only detect out-of-distribution (OOD) inputs but also anticipate in-distribution (ID) errors by assigning low confidence to potentially misclassified samples. Yet, most prior work treats OOD detection and failure prediction as separated problems, overlooking their closed connection. We argue that reliability requires evaluating them jointly. To this end, we propose a unified evaluation framework that integrates OOD detection and failure prediction, quantified by our new metrics DS-F1 and DS-AURC, where DS denotes double scoring functions. Experiments on the OpenOOD benchmark show that double scoring functions yield classifiers that are substantially more reliable than traditional single scoring approaches. Our analysis further reveals that OOD-based approaches provide notable gains under simple or far-OOD shifts, but only marginal benefits under more challenging near-OOD conditions. Beyond evaluation, we extend the reliable classifier SURE and introduce SURE+, a new approach that significantly improves reliability across diverse scenarios. Together, our framework, metrics, and method establish a new benchmark for trustworthy classification and offer practical guidance for deploying robust models in real-world settings. The source code is publicly available at this https URL.
>
---
#### [new 024] Vector-Quantized Soft Label Compression for Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在解决软标签存储成本高的问题。通过引入向量量化自编码器压缩软标签，实现高效存储同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03808](https://arxiv.org/pdf/2603.03808)**

> **作者:** Ali Abbasi; Ashkan Shahbazi; Hamed Pirsiavash; Soheil Kolouri
>
> **摘要:** Dataset distillation is an emerging technique for reducing the computational and storage costs of training machine learning models by synthesizing a small, informative subset of data that captures the essential characteristics of a much larger dataset. Recent methods pair synthetic samples and their augmentations with soft labels from a teacher model, enabling student models to generalize effectively despite the small size of the distilled dataset. While soft labels are critical for effective distillation, the storage and communication overhead they incur, especially when accounting for augmentations, is often overlooked. In practice, each distilled sample is associated with multiple soft labels, making them the dominant contributor to storage costs, particularly in large-class settings such as ImageNet-1K. In this paper, we present a rigorous analysis of bit requirements across dataset distillation frameworks, quantifying the storage demands of both distilled samples and their soft labels. To address the overhead, we introduce a vector-quantized autoencoder (VQAE) for compressing soft labels, achieving substantial compression while preserving the effectiveness of the distilled data. We validate our method on both vision and language distillation benchmarks. On ImageNet-1K, our proposed VQAE achieves 30--40x additional compression over RDED, LPLD, SRE2L, and CDA baselines while retaining over $90\%$ of their original performance.
>
---
#### [new 025] mHC-HSI: Clustering-Guided Hyper-Connection Mamba for Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在提升分类精度与模型可解释性。提出mHC-HSI模型，通过聚类引导的Mamba模块和物理语义分组，增强空间-光谱特征学习与模型解释性。**

- **链接: [https://arxiv.org/pdf/2603.03418](https://arxiv.org/pdf/2603.03418)**

> **作者:** Yimin Zhu; Zack Dewis; Quinn Ledingham; Saeid Taleghanidoozdoozan; Mabel Heffring; Zhengsen Xu; Motasem Alkayid; Megan Greenwood; Lincoln Linlin Xu
>
> **摘要:** Recently, DeepSeek has invented the manifold-constrained hyper-connection (mHC) approach which has demonstrated significant improvements over the traditional residual connection in deep learning models \cite{xie2026mhc}. Nevertheless, this approach has not been tailor-designed for improving hyperspectral image (HSI) classification. This paper presents a clustering-guided mHC Mamba model (mHC-HSI) for enhanced HSI classification, with the following contributions. First, to improve spatial-spectral feature learning, we design a novel clustering-guided Mamba module, based on the mHC framework, that explicitly learns both spatial and spectral information in HSI. Second, to decompose the complex and heterogeneous HSI into smaller clusters, we design a new implementation of the residual matrix in mHC, which can be treated as soft cluster membership maps, leading to improved explainability of the mHC approach. Third, to leverage the physical spectral knowledge, we divide the spectral bands into physically-meaningful groups and use them as the "parallel streams" in mHC, leading to a physically-meaningful approach with enhanced interpretability. The proposed approach is tested on benchmark datasets in comparison with the state-of-the-art methods, and the results suggest that the proposed model not only improves the accuracy but also enhances the model explainability. Code is available here: this https URL
>
---
#### [new 026] Degradation-based augmented training for robust individual animal re-identification
- **分类: cs.CV**

- **简介: 该论文属于野生动物个体重识别任务，旨在解决图像退化导致识别性能下降的问题。通过引入多样化的退化增强训练，提升模型在真实退化图像上的识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.04163](https://arxiv.org/pdf/2603.04163)**

> **作者:** Thanos Polychronou; Lukáš Adam; Viktor Penchev; Kostas Papafitsoros
>
> **摘要:** Wildlife re-identification aims to recognise individual animals by matching query images to a database of previously identified individuals, based on their fine-scale unique morphological characteristics. Current state-of-the-art models for multispecies re- identification are based on deep metric learning representing individual identities by fea- ture vectors in an embedding space, the similarity of which forms the basis for a fast automated identity retrieval. Yet very often, the discriminative information of individual wild animals gets significantly reduced due to the presence of several degradation factors in images, leading to reduced retrieval performance and limiting the downstream eco- logical studies. Here, starting by showing that the extent of this performance reduction greatly varies depending on the animal species (18 wild animal datasets), we introduce an augmented training framework for deep feature extractors, where we apply artificial but diverse degradations in images in the training set. We show that applying this augmented training only to a subset of individuals, leads to an overall increased re-identification performance, under the same type of degradations, even for individuals not seen during training. The introduction of diverse degradations during training leads to a gain of up to 8.5% Rank-1 accuracy to a dataset of real-world degraded animal images, selected using human re-ID expert annotations provided here for the first time. Our work is the first to systematically study image degradation in wildlife re-identification, while introducing all the necessary benchmarks, publicly available code and data, enabling further research on this topic.
>
---
#### [new 027] Scaling Dense Event-Stream Pretraining from Visual Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-事件流对齐任务，解决事件流表示学习中的标注依赖与语义崩溃问题。通过结构感知的蒸馏方法提升事件流的密集表示性能。**

- **链接: [https://arxiv.org/pdf/2603.03969](https://arxiv.org/pdf/2603.03969)**

> **作者:** Zhiwen Chen; Junhui Hou; Zhiyu Zhu; Jinjian Wu; Guangming Shi
>
> **摘要:** Learning versatile, fine-grained representations from irregular event streams is pivotal yet nontrivial, primarily due to the heavy annotation that hinders scalability in dataset size, semantic richness, and application scope. To mitigate this dilemma, we launch a novel self-supervised pretraining method that distills visual foundation models (VFMs) to push the boundaries of event representation at scale. Specifically, we curate an extensive synchronized image-event collection to amplify cross-modal alignment. Nevertheless, due to inherent mismatches in sparsity and granularity between image-event domains, existing distillation paradigms are prone to semantic collapse in event representations, particularly at high resolutions. To bridge this gap, we propose to extend the alignment objective to semantic structures provided off-the-shelf by VFMs, indicating a broader receptive field and stronger supervision. The key ingredient of our method is a structure-aware distillation loss that grounds higher-quality image-event correspondences for alignment, optimizing dense event representations. Extensive experiments demonstrate that our approach takes a great leap in downstream benchmarks, significantly surpassing traditional methods and existing pretraining techniques. This breakthrough manifests in enhanced generalization, superior data efficiency and elevated transferability.
>
---
#### [new 028] Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于长期水下视觉定位任务，旨在解决动态海底环境下的定位与地图构建问题。作者构建了首个多站点长期水下数据集，并提出基于足迹的真值方法，用于提升视觉位置识别的准确性。**

- **链接: [https://arxiv.org/pdf/2603.04056](https://arxiv.org/pdf/2603.04056)**

> **作者:** Martin Kvisvik Larsen; Oscar Pizarro
>
> **摘要:** Long-term visual localization has the potential to reduce cost and improve mapping quality in optical benthic monitoring with autonomous underwater vehicles (AUVs). Despite this potential, long-term visual localization in benthic environments remains understudied, primarily due to the lack of curated datasets for benchmarking. Moreover, limited georeferencing accuracy and image footprints necessitate precise geometric information for accurate ground-truthing. In this work, we address these gaps by presenting a curated dataset for long-term visual localization in benthic environments and a novel method to ground-truth visual localization results for near-nadir underwater imagery. Our dataset comprises georeferenced AUV imagery from five benthic reference sites, revisited over periods up to six years, and includes raw and color-corrected stereo imagery, camera calibrations, and sub-decimeter registered camera poses. To our knowledge, this is the first curated underwater dataset for long-term visual localization spanning multiple sites and photic-zone habitats. Our ground-truthing method estimates 3D seafloor image footprints and links camera views with overlapping footprints, ensuring that ground-truth links reflect shared visual content. Building on this dataset and ground truth, we benchmark eight state-of-the-art visual place recognition (VPR) methods and find that Recall@K is significantly lower on our dataset than on established terrestrial and underwater benchmarks. Finally, we compare our footprint-based ground truth to a traditional location-based ground truth and show that distance-threshold ground-truthing can overestimate VPR Recall@K at sites with rugged terrain and altitude variations. Together, the curated dataset, ground-truthing method, and VPR benchmark provide a stepping stone for advancing long-term visual localization in dynamic benthic environments.
>
---
#### [new 029] GeoSeg: Training-Free Reasoning-Driven Segmentation in Remote Sensing Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分割任务，解决因数据成本高和视角特殊导致的分割难题。提出GeoSeg框架，实现无需训练的推理驱动分割。**

- **链接: [https://arxiv.org/pdf/2603.03983](https://arxiv.org/pdf/2603.03983)**

> **作者:** Lifan Jiang; Yuhang Pei; oxi Wu; Yan Zhao; Tianrun Wu; Shulong Yu; Lihui Zhang; Deng Cai
>
> **摘要:** Recent advances in MLLMs are reframing segmentation from fixed-category prediction to instruction-grounded localization. While reasoning based segmentation has progressed rapidly in natural scenes, remote sensing lacks a generalizable solution due to the prohibitive cost of reasoning-oriented data and domain-specific challenges like overhead viewpoints. We present GeoSeg, a zero-shot, training-free framework that bypasses the supervision bottleneck for reasoning-driven remote sensing segmentation. GeoSeg couples MLLM reasoning with precise localization via: (i) bias-aware coordinate refinement to correct systematic grounding shifts and (ii) a dual-route prompting mechanism to fuse semantic intent with fine-grained spatial cues. We also introduce GeoSeg-Bench, a diagnostic benchmark of 810 image--query pairs with hierarchical difficulty levels. Experiments show that GeoSeg consistently outperforms all baselines, with extensive ablations confirming the effectiveness and necessity of each component.
>
---
#### [new 030] BLOCK: An Open-Source Bi-Stage MLLM Character-to-Skin Pipeline for Minecraft
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BLOCK，解决从角色概念生成Minecraft皮肤的任务。通过两阶段流程，结合多模态模型和优化算法，实现像素精确的皮肤生成。**

- **链接: [https://arxiv.org/pdf/2603.03964](https://arxiv.org/pdf/2603.03964)**

> **作者:** Hengquan Guo
>
> **摘要:** We present \textbf{BLOCK}, an open-source bi-stage character-to-skin pipeline that generates pixel-perfect Minecraft skins from arbitrary character concepts. BLOCK decomposes the problem into (i) a \textbf{3D preview synthesis stage} driven by a large multimodal model (MLLM) with a carefully designed prompt-and-reference template, producing a consistent dual-panel (front/back) oblique-view Minecraft-style preview; and (ii) a \textbf{skin decoding stage} based on a fine-tuned FLUX.2 model that translates the preview into a skin atlas image. We further propose \textbf{EvolveLoRA}, a progressive LoRA curriculum (text-to-image $\rightarrow$ image-to-image $\rightarrow$ preview-to-skin) that initializes each phase from the previous adapter to improve stability and efficiency. BLOCK is released with all prompt templates and fine-tuned weights to support reproducible character-to-skin generation.
>
---
#### [new 031] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR
- **分类: cs.CV; cs.GR; cs.HC**

- **简介: 该论文属于AR/VR中的第一视角人体运动估计任务，解决因视角限制、遮挡和数据不足带来的挑战，提出EgoPoseFormer v2模型及自动标注系统，提升精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04090](https://arxiv.org/pdf/2603.04090)**

> **作者:** Zhenyu Li; Sai Kumar Dwivedi; Filip Maric; Carlos Chacon; Nadine Bertsch; Filippo Arcadu; Tomas Hodan; Michael Ramamonjisoa; Peter Wonka; Amy Zhao; Robin Kips; Cem Keskin; Anastasia Tkach; Chenhongyi Yang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Egocentric human motion estimation is essential for AR/VR experiences, yet remains challenging due to limited body coverage from the egocentric viewpoint, frequent occlusions, and scarce labeled data. We present EgoPoseFormer v2, a method that addresses these challenges through two key contributions: (1) a transformer-based model for temporally consistent and spatially grounded body pose estimation, and (2) an auto-labeling system that enables the use of large unlabeled datasets for training. Our model is fully differentiable, introduces identity-conditioned queries, multi-view spatial refinement, causal temporal attention, and supports both keypoints and parametric body representations under a constant compute budget. The auto-labeling system scales learning to tens of millions of unlabeled frames via uncertainty-aware semi-supervised training. The system follows a teacher-student schema to generate pseudo-labels and guide training with uncertainty distillation, enabling the model to generalize to different environments. On the EgoBody3M benchmark, with a 0.8 ms latency on GPU, our model outperforms two state-of-the-art methods by 12.2% and 19.4% in accuracy, and reduces temporal jitter by 22.2% and 51.7%. Furthermore, our auto-labeling system further improves the wrist MPJPE by 13.1%.
>
---
#### [new 032] Small Object Detection in Complex Backgrounds with Multi-Scale Attention and Global Relation Modeling
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，旨在解决复杂背景下的小目标检测难题。通过引入多尺度注意力和全局关系建模，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.03788](https://arxiv.org/pdf/2603.03788)**

> **作者:** Wenguang Tao; Xiaotian Wang; Tian Yan; Yi Wang; Jie Yan
>
> **摘要:** Small object detection under complex backgrounds remains a challenging task due to severe feature degradation, weak semantic representation, and inaccurate localization caused by downsampling operations and background interference. Existing detection frameworks are mainly designed for general objects and often fail to explicitly address the unique characteristics of small objects, such as limited structural cues and strong sensitivity to localization errors. In this paper, we propose a multi-level feature enhancement and global relation modeling framework tailored for small object detection. Specifically, a Residual Haar Wavelet Downsampling module is introduced to preserve fine-grained structural details by jointly exploiting spatial-domain convolutional features and frequency-domain representations. To enhance global semantic awareness and suppress background noise, a Global Relation Modeling module is employed to capture long-range dependencies at high-level feature stages. Furthermore, a Cross-Scale Hybrid Attention module is designed to establish sparse and aligned interactions across multi-scale features, enabling effective fusion of high-resolution details and high-level semantic information with reduced computational overhead. Finally, a Center-Assisted Loss is incorporated to stabilize training and improve localization accuracy for small objects. Extensive experiments conducted on the large-scale RGBT-Tiny benchmark demonstrate that the proposed method consistently outperforms existing state-of-the-art detectors under both IoU-based and scale-adaptive evaluation metrics. These results validate the effectiveness and robustness of the proposed framework for small object detection in complex environments.
>
---
#### [new 033] CLIP-Guided Multi-Task Regression for Multi-View Plant Phenotyping
- **分类: cs.CV**

- **简介: 该论文属于多视角植物表型任务，解决多视角图像中因视角冗余和外观变化导致的预测困难。通过CLIP引导的多任务回归方法，提升植物年龄和叶数预测的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04091](https://arxiv.org/pdf/2603.04091)**

> **作者:** Simon Warmers; Muhammad Zawish; Fayaz Ali Dharejo; Steven Davy; Radu Timofte
>
> **备注:** Under review at IEEE Conference
>
> **摘要:** Modeling plant growth dynamics plays a central role in modern agricultural research. However, learning robust predictors from multi-view plant imagery remains challenging due to strong viewpoint redundancy and viewpoint-dependent appearance changes. We propose a level-aware vision language framework that jointly predicts plant age and leaf count using a single multi-task model built on CLIP embeddings. Our method aggregates rotational views into angle-invariant representations and conditions visual features on lightweight text priors encoding viewpoint level for stable prediction under incomplete or unordered inputs. On the GroMo25 benchmark, our approach reduces mean age MAE from 7.74 to 3.91 and mean leaf-count MAE from 5.52 to 3.08 compared to the GroMo baseline, corresponding to improvements of 49.5% and 44.2%, respectively. The unified formulation simplifies the pipeline by replacing the conventional dual-model setup while improving robustness to missing views. The models and code is available at: this https URL
>
---
#### [new 034] Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成评估任务，旨在解决合成环境增强图像的真实性评价问题。通过比较规则方法与生成模型，提出一种可扩展的评估框架，验证生成模型在不同天气条件下的表现。**

- **链接: [https://arxiv.org/pdf/2603.04325](https://arxiv.org/pdf/2603.04325)**

> **作者:** Damian J. Ruck; Paul Vautravers; Oliver Chalkley; Jake Thomas
>
> **摘要:** Evaluation of AI systems often requires synthetic test cases, particularly for rare or safety-critical conditions that are difficult to observe in operational data. Generative AI offers a promising approach for producing such data through controllable image editing, but its usefulness depends on whether the resulting images are sufficiently realistic to support meaningful evaluation. We present a scalable framework for assessing the realism of synthetic image-editing methods and apply it to the task of adding environmental conditions-fog, rain, snow, and nighttime-to car-mounted camera images. Using 40 clear-day images, we compare rule-based augmentation libraries with generative AI image-editing models. Realism is evaluated using two complementary automated metrics: a vision-language model (VLM) jury for perceptual realism assessment, and embedding-based distributional analysis to measure similarity to genuine adverse-condition imagery. Generative AI methods substantially outperform rule-based approaches, with the best generative method achieving approximately 3.6 times the acceptance rate of the best rule-based method. Performance varies across conditions: fog proves easiest to simulate, while nighttime transformations remain challenging. Notably, the VLM jury assigns imperfect acceptance even to real adverse-condition imagery, establishing practical ceilings against which synthetic methods can be judged. By this standard, leading generative methods match or exceed real-image performance for most conditions. These results suggest that modern generative image-editing models can enable scalable generation of realistic adverse-condition imagery for evaluation pipelines. Our framework therefore provides a practical approach for scalable realism evaluation, though validation against human studies remains an important direction for future work.
>
---
#### [new 035] UniSync: Towards Generalizable and High-Fidelity Lip Synchronization for Challenging Scenarios
- **分类: cs.CV**

- **简介: 该论文属于唇部同步任务，解决真实视频配音中唇形与音频不匹配的问题。提出UniSync框架，结合无掩码和有掩码策略，提升同步精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03882](https://arxiv.org/pdf/2603.03882)**

> **作者:** Ruidi Fan; Yang Zhou; Siyuan Wang; Tian Yu; Yutong Jiang; Xusheng Liu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Lip synchronization aims to generate realistic talking videos that match given audio, which is essential for high-quality video dubbing. However, current methods have fundamental drawbacks: mask-based approaches suffer from local color discrepancies, while mask-free methods struggle with global background texture misalignment. Furthermore, most methods struggle with diverse real-world scenarios such as stylized avatars, face occlusion, and extreme lighting conditions. In this paper, we propose UniSync, a unified framework designed for achieving high-fidelity lip synchronization in diverse scenarios. Specifically, UniSync uses a mask-free pose-anchored training strategy to keep head motion and eliminate synthesis color artifacts, while employing mask-based blending consistent inference to ensure structural precision and smooth blending. Notably, fine-tuning on compact but diverse videos empowers our model with exceptional domain adaptability, handling complex corner cases effectively. We also introduce the RealWorld-LipSync benchmark to evaluate models under real-world demands, which covers diverse application scenarios including both human faces and stylized avatars. Extensive experiments demonstrate that UniSync significantly outperforms state-of-the-art methods, advancing the field towards truly generalizable and production-ready lip synchronization.
>
---
#### [new 036] Weakly Supervised Patch Annotation for Improved Screening of Diabetic Retinopathy
- **分类: cs.CV**

- **简介: 该论文属于糖尿病视网膜病变筛查任务，解决标注不足导致早期病变漏诊的问题。提出SAFE框架，通过弱监督和对比学习实现病灶区域的系统标注，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03991](https://arxiv.org/pdf/2603.03991)**

> **作者:** Shramana Dey; Abhirup Banerjee; B. Uma Shankar; Ramachandran Rajalakshmi; Sushmita Mitra
>
> **摘要:** Diabetic Retinopathy (DR) requires timely screening to prevent irreversible vision loss. However, its early detection remains a significant challenge since often the subtle pathological manifestations (lesions) get overlooked due to insufficient annotation. Existing literature primarily focuses on image-level supervision, weakly-supervised localization, and clustering-based representation learning, which fail to systematically annotate unlabeled lesion region(s) for refining the dataset. Expert-driven lesion annotation is labor-intensive and often incomplete, limiting the performance of deep learning models. We introduce Similarity-based Annotation via Feature-space Ensemble (SAFE), a two-stage framework that unifies weak supervision, contrastive learning, and patch-wise embedding inference, to systematically expand sparse annotations in the pathology. SAFE preserves fine-grained details of the lesion(s) under partial clinical supervision. In the first stage, a dual-arm Patch Embedding Network learns semantically structured, class-discriminative embeddings from expert annotated patches. Next, an ensemble of independent embedding spaces extrapolates labels to the unannotated regions based on spatial and semantic proximity. An abstention mechanism ensures trade-off between highly reliable annotation and noisy coverage. Experimental results demonstrate reliable separation of healthy and diseased patches, achieving upto 0.9886 accuracy. The annotation generated from SAFE substantially improves downstream tasks such as DR classification, demonstrating a substantial increase in F1-score of the diseased class and a performance gain as high as 0.545 in Area Under the Precision-Recall Curve (AUPRC). Qualitative analysis, with explainability, confirms that SAFE focuses on clinically relevant lesion patterns; and is further validated by ophthalmologists.
>
---
#### [new 037] DISC: Dense Integrated Semantic Context for Large-Scale Open-Set Semantic Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DISC，解决开放集语义映射中上下文缺失和计算成本高的问题，通过单次、距离加权的特征提取实现高效实时的语义表示。**

- **链接: [https://arxiv.org/pdf/2603.03935](https://arxiv.org/pdf/2603.03935)**

> **作者:** Felix Igelbrink; Lennart Niecksch; Martin Atzmueller; Joachim Hertzberg
>
> **摘要:** Open-set semantic mapping enables language-driven robotic perception, but current instance-centric approaches are bottlenecked by context-depriving and computationally expensive crop-based feature extraction. To overcome this fundamental limitation, we introduce DISC (Dense Integrated Semantic Context), featuring a novel single-pass, distance-weighted extraction mechanism. By deriving high-fidelity CLIP embeddings directly from the vision transformer's intermediate layers, our approach eliminates the latency and domain-shift artifacts of traditional image cropping, yielding pure, mask-aligned semantic representations. To fully leverage these features in large-scale continuous mapping, DISC is built upon a fully GPU-accelerated architecture that replaces periodic offline processing with precise, on-the-fly voxel-level instance refinement. We evaluate our approach on standard benchmarks (Replica, ScanNet) and a newly generated large-scale-mapping dataset based on Habitat-Matterport 3D (HM3DSEM) to assess scalability across complex scenes in multi-story buildings. Extensive evaluations demonstrate that DISC significantly surpasses current state-of-the-art zero-shot methods in both semantic accuracy and query retrieval, providing a robust, real-time capable framework for robotic deployment. The full source code, data generation and evaluation pipelines will be made available at this https URL.
>
---
#### [new 038] Structure-aware Prompt Adaptation from Seen to Unseen for Open-Vocabulary Compositional Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于开放词汇组合零样本学习任务，解决模型在未见属性和物体上的泛化问题。提出SPA方法，通过结构一致性损失和引导适配策略，提升模型对未见概念的识别能力。**

- **链接: [https://arxiv.org/pdf/2603.03815](https://arxiv.org/pdf/2603.03815)**

> **作者:** Yihang Duan; Jiong Wang; Pengpeng Zeng; Ji Zhang; Lei Zhao; Chong Wang; Jingkuan Song; Lianli Gao
>
> **摘要:** The goal of Open-Vocabulary Compositional Zero-Shot Learning (OV-CZSL) is to recognize attribute-object compositions in the open-vocabulary setting, where compositions of both seen and unseen attributes and objects are evaluated. Recently, prompt tuning methods have demonstrated strong generalization capabilities in the closed setting, where only compositions of seen attributes and objects are evaluated, i.e., Compositional Zero-Shot Learning (CZSL). However, directly applying these methods to OV-CZSL may not be sufficient to generalize to unseen attributes, objects and their compositions, as it is limited to seen attributes and objects. Normally, when faced with unseen concepts, humans adopt analogies with seen concepts that have the similar semantics thereby inferring their meaning (e.g., "wet" and "damp", "shirt" and "jacket"). In this paper, we experimentally show that the distribution of semantically related attributes or objects tends to form consistent local structures in the embedding space. Based on the above structures, we propose Structure-aware Prompt Adaptation (SPA) method, which enables models to generalize from seen to unseen attributes and objects. Specifically, in the training stage, we design a Structure-aware Consistency Loss (SCL) that encourages the local structure's consistency of seen attributes and objects in each iteration. In the inference stage, we devise a Structure-guided Adaptation Strategy (SAS) that adaptively aligns the structures of unseen attributes and objects with those of trained seen attributes and objects with similar semantics. Notably, SPA is a plug-and-play method that can be seamlessly integrated into existing CZSL prompt tuning methods. Extensive experiments on OV-CZSL benchmarks demonstrate that SPA achieves competitive closed-set performance while significantly improving open-vocabulary results.
>
---
#### [new 039] PlaneCycle: Training-Free 2D-to-3D Lifting of Foundation Models Without Adapters
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PlaneCycle，解决2D基础模型到3D数据的转换问题。无需训练或适配器，通过空间聚合实现3D融合，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.04165](https://arxiv.org/pdf/2603.04165)**

> **作者:** Yinghong Yu; Guangyuan Li; Jiancheng Yang
>
> **备注:** Code is available at this https URL
>
> **摘要:** Large-scale 2D foundation models exhibit strong transferable representations, yet extending them to 3D volumetric data typically requires retraining, adapters, or architectural redesign. We introduce PlaneCycle, a training-free, adapter-free operator for architecture-agnostic 2D-to-3D lifting of foundation models. PlaneCycle reuses the original pretrained 2D backbone by cyclically distributing spatial aggregation across orthogonal HW, DW, and DH planes throughout network depth, enabling progressive 3D fusion while preserving pretrained inductive biases. The method introduces no additional parameters and is applicable to arbitrary 2D networks. Using pretrained DINOv3 models, we evaluate PlaneCycle on six 3D classification and three 3D segmentation benchmarks. Without any training, the lifted models exhibit intrinsic 3D fusion capability and, under linear probing, outperform slice-wise 2D baselines and strong 3D counterparts, approaching the performance of fully trained models. With full fine-tuning, PlaneCycle matches standard 3D architectures, highlighting its potential as a seamless and practical 2D-to-3D lifting operator. These results demonstrate that 3D capability can be unlocked from pretrained 2D foundation models without structural modification or retraining. Code is available at this https URL.
>
---
#### [new 040] ZipMap: Linear-Time Stateful 3D Reconstruction with Test-Time Training
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ZipMap，解决3D重建效率与质量的平衡问题。通过状态化模型和测试时训练，实现线性时间重建，提升速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.04385](https://arxiv.org/pdf/2603.04385)**

> **作者:** Haian Jin; Rundi Wu; Tianyuan Zhang; Ruiqi Gao; Jonathan T. Barron; Noah Snavely; Aleksander Holynski
>
> **备注:** Project page: this https URL
>
> **摘要:** Feed-forward transformer models have driven rapid progress in 3D vision, but state-of-the-art methods such as VGGT and $\pi^3$ have a computational cost that scales quadratically with the number of input images, making them inefficient when applied to large image collections. Sequential-reconstruction approaches reduce this cost but sacrifice reconstruction quality. We introduce ZipMap, a stateful feed-forward model that achieves linear-time, bidirectional 3D reconstruction while matching or surpassing the accuracy of quadratic-time methods. ZipMap employs test-time training layers to zip an entire image collection into a compact hidden scene state in a single forward pass, enabling reconstruction of over 700 frames in under 10 seconds on a single H100 GPU, more than $20\times$ faster than state-of-the-art methods such as VGGT. Moreover, we demonstrate the benefits of having a stateful representation in real-time scene-state querying and its extension to sequential streaming reconstruction.
>
---
#### [new 041] Modeling Cross-vision Synergy for Unified Large Vision Model
- **分类: cs.CV**

- **简介: 该论文属于视觉多模态任务，旨在解决统一大视觉模型中跨模态协同不足的问题。提出PolyV模型，在架构和训练上实现跨视觉协同，提升多模态理解性能。**

- **链接: [https://arxiv.org/pdf/2603.03564](https://arxiv.org/pdf/2603.03564)**

> **作者:** Shengqiong Wu; Lanhu Wu; Mingyang Bao; Wenhao Xu; Hanwang Zhang; Shuicheng Yan; Hao Fei; Tat-Seng Chua
>
> **备注:** 21 pages, 9 figures, 16 tables, CVPR
>
> **摘要:** Recent advances in large vision models (LVMs) have shifted from modality-specific designs toward unified architectures that jointly process images, videos, and 3D data. However, existing unified LVMs primarily pursue functional integration, while overlooking the deeper goal of cross-vision synergy: the ability to reason over complementary priors across visual modalities. To address this, we present PolyV, a unified LVM that achieves cross-vision synergy at both the architectural and training levels. Architecturally, PolyV adopts a sparse Mixture-of-Experts LVM coordinated by a dynamic modality router, allowing each expert to specialize in modality-specific priors while enabling bidirectional interaction and mutual refinement across modalities. Training-wise, a synergy-aware paradigm combines modality-specific pretraining with coarse-to-fine synergy tuning via knowledge distillation and object-/relation-level alignment. Extensive experiments on 10 benchmarks spanning image, video, and 3D understanding, including synergy-focused datasets requiring spatial or temporal priors, demonstrate that PolyV consistently outperforms existing models, achieving over 10% average improvement over its backbone. Overall, PolyV establishes a unified framework for synesthetic visual reasoning, advancing toward truly synergistic LVMs. Project page: this https URL.
>
---
#### [new 042] Enhancing Authorship Attribution with Synthetic Paintings
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于艺术作品作者鉴定任务，旨在解决真实数据不足的问题。通过合成图像增强分类模型性能，提升识别准确率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04343](https://arxiv.org/pdf/2603.04343)**

> **作者:** Clarissa Loures; Caio Hosken; Luan Oliveira; Gianlucca Zuin; Adriano Veloso
>
> **备注:** Accepted for publication at the 24th IEEE International Conference on Machine Learning and Applications (ICMLA 2025)
>
> **摘要:** Attributing authorship to paintings is a historically complex task, and one of its main challenges is the limited availability of real artworks for training computational models. This study investigates whether synthetic images, generated through DreamBooth fine-tuning of Stable Diffusion, can improve the performance of classification models in this context. We propose a hybrid approach that combines real and synthetic data to enhance model accuracy and generalization across similar artistic styles. Experimental results show that adding synthetic images leads to higher ROC-AUC and accuracy compared to using only real paintings. By integrating generative and discriminative methods, this work contributes to the development of computer vision techniques for artwork authentication in data-scarce scenarios.
>
---
#### [new 043] RIVER: A Real-Time Interaction Benchmark for Video LLMs
- **分类: cs.CV**

- **简介: 该论文属于视频语言模型的实时交互任务，解决在线视频理解中的实时性与连贯性问题。提出RIVER Bench基准和改进方法，提升模型的长期记忆与未来感知能力。**

- **链接: [https://arxiv.org/pdf/2603.03985](https://arxiv.org/pdf/2603.03985)**

> **作者:** Yansong Shi; Qingsong Zhao; Tianxiang Jiang; Xiangyu Zeng; Yi Wang; Limin Wang
>
> **摘要:** The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at this https URL.
>
---
#### [new 044] Glass Segmentation with Fusion of Learned and General Visual Features
- **分类: cs.CV**

- **简介: 该论文属于玻璃分割任务，旨在解决透明玻璃表面缺乏视觉特征导致的分割难题。通过融合通用与特定任务特征，提出新架构实现高效准确分割。**

- **链接: [https://arxiv.org/pdf/2603.03718](https://arxiv.org/pdf/2603.03718)**

> **作者:** Risto Ojala; Tristan Ellison; Mo Chen
>
> **摘要:** Glass surface segmentation from RGB images is a challenging task, since glass as a transparent material distinctly lacks visual characteristics. However, glass segmentation is critical for scene understanding and robotics, as transparent glass surfaces must be identified as solid material. This paper presents a novel architecture for glass segmentation, deploying a dual-backbone producing general visual features as well as task-specific learned visual features. General visual features are produced by a frozen DINOv3 vision foundation model, and the task-specific features are generated with a Swin model trained in a supervised manner. Resulting multi-scale feature representations are downsampled with residual Squeeze-and-Excitation Channel Reduction, and fed into a Mask2Former Decoder, producing the final segmentation masks. The architecture was evaluated on four commonly used glass segmentation datasets, achieving state-of-the-art results on several accuracy metrics. The model also has a competitive inference speed compared to the previous state-of-the-art method, and surpasses it when using a lighter DINOv3 backbone variant. The implementation source code and model weights are available at: this https URL
>
---
#### [new 045] Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10
- **分类: cs.CV**

- **简介: 该论文属于 underwater object detection 任务，旨在解决水下图像质量差导致的检测难题。通过引入增强模块、注意力机制和改进损失函数，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.03807](https://arxiv.org/pdf/2603.03807)**

> **作者:** Md. Mushibur Rahman; Umme Fawzia Rahim; Enam Ahmed Taufik
>
> **备注:** Accepted in 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)
>
> **摘要:** Underwater object detection constitutes a pivotal endeavor within the realms of marine surveillance and autonomous underwater systems; however, it presents significant challenges due to pronounced visual impairments arising from phenomena such as light absorption, scattering, and diminished contrast. In response to these formidable challenges, this manuscript introduces a streamlined yet robust framework for underwater object detection, grounded in the YOLOv10 architecture. The proposed method integrates a Multi-Stage Adaptive Enhancement module to improve image quality, a Dual-Pooling Sequential Attention (DPSA) mechanism embedded into the backbone to strengthen multi-scale feature representation, and a Focal Generalized IoU Objectness (FGIoU) loss to jointly improve localization accuracy and objectness prediction under class imbalance. Comprehensive experimental evaluations conducted on the RUOD and DUO benchmark datasets substantiate that the proposed DPSA_FGIoU_YOLOv10n attains exceptional performance, achieving mean Average Precision (mAP) scores of 88.9% and 88.0% at IoU threshold 0.5, respectively. In comparison to the baseline YOLOv10n, this represents enhancements of 6.7% for RUOD and 6.2% for DUO, all while preserving a compact model architecture comprising merely 2.8M parameters. These findings validate that the proposed framework establishes an efficacious equilibrium among accuracy, robustness, and real-time operational efficiency, making it suitable for deployment in resource-constrained underwater settings.
>
---
#### [new 046] DQE-CIR: Distinctive Query Embeddings through Learnable Attribute Weights and Target Relative Negative Sampling in Composed Image Retrieval
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对组合图像检索任务，解决因负样本选择不当导致的语义混淆和区分度不足问题，提出DQE-CIR方法，通过可学习属性权重和目标相关负采样提升检索性能。**

- **链接: [https://arxiv.org/pdf/2603.04037](https://arxiv.org/pdf/2603.04037)**

> **作者:** Geon Park; Ji-Hoon Park; Seong-Whan Lee
>
> **备注:** 33 pages
>
> **摘要:** Composed image retrieval (CIR) addresses the task of retrieving a target image by jointly interpreting a reference image and a modification text that specifies the intended change. Most existing methods are still built upon contrastive learning frameworks that treat the ground truth image as the only positive instance and all remaining images as negatives. This strategy inevitably introduces relevance suppression, where semantically related yet valid images are incorrectly pushed away, and semantic confusion, where different modification intents collapse into overlapping regions of the embedding space. As a result, the learned query representations often lack discriminativeness, particularly at fine-grained attribute modifications. To overcome these limitations, we propose distinctive query embeddings through learnable attribute weights and target relative negative sampling (DQE-CIR), a method designed to learn distinctive query embeddings by explicitly modeling target relative relevance during training. DQE-CIR incorporates learnable attribute weighting to emphasize distinctive visual features conditioned on the modification text, enabling more precise feature alignment between language and vision. Furthermore, we introduce target relative negative sampling, which constructs a target relative similarity distribution and selects informative negatives from a mid-zone region that excludes both easy negatives and ambiguous false negatives. This strategy enables more reliable retrieval for fine-grained attribute changes by improving query discriminativeness and reducing confusion caused by semantically similar but irrelevant candidates.
>
---
#### [new 047] A multi-center analysis of deep learning methods for video polyp detection and segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频结肠息肉检测与分割任务，旨在解决息肉漏检和切除不完全的问题。通过深度学习方法，结合序列数据和时间信息提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.04288](https://arxiv.org/pdf/2603.04288)**

> **作者:** Noha Ghatwary; Pedro Chavarias Solano; Mohamed Ramzy Ibrahim; Adrian Krenzer; Frank Puppe; Stefano Realdon; Renato Cannizzaro; Jiacheng Wang; Liansheng Wang; Thuy Nuong Tran; Lena Maier-Hein; Amine Yamlahi; Patrick Godau; Quan He; Qiming Wan; Mariia Kokshaikyna; Mariia Dobko; Haili Ye; Heng Li; Ragu B; Antony Raj; Hanaa Nagdy; Osama E Salem; James E. East; Dominique Lamarque; Thomas de Lange; Sharib Ali
>
> **备注:** 17 pages
>
> **摘要:** Colonic polyps are well-recognized precursors to colorectal cancer (CRC), typically detected during colonoscopy. However, the variability in appearance, location, and size of these polyps complicates their detection and removal, leading to challenges in effective surveillance, intervention, and subsequently CRC prevention. The processes of colonoscopy surveillance and polyp removal are highly reliant on the expertise of gastroenterologists and occur within the complexities of the colonic structure. As a result, there is a high rate of missed detections and incomplete removal of colonic polyps, which can adversely impact patient outcomes. Recently, automated methods that use machine learning have been developed to enhance polyps detection and segmentation, thus helping clinical processes and reducing missed rates. These advancements highlight the potential for improving diagnostic accuracy in real-time applications, which ultimately facilitates more effective patient management. Furthermore, integrating sequence data and temporal information could significantly enhance the precision of these methods by capturing the dynamic nature of polyp growth and the changes that occur over time. To rigorously investigate these challenges, data scientists and experts gastroenterologists collaborated to compile a comprehensive dataset that spans multiple centers and diverse populations. This initiative aims to underscore the critical importance of incorporating sequence data and temporal information in the development of robust automated detection and segmentation methods. This study evaluates the applicability of deep learning techniques developed in real-time clinical colonoscopy tasks using sequence data, highlighting the critical role of temporal relationships between frames in improving diagnostic precision.
>
---
#### [new 048] TumorFlow: Physics-Guided Longitudinal MRI Synthesis of Glioblastoma Growth
- **分类: cs.CV**

- **简介: 该论文属于医学影像生成任务，旨在解决 glioblastoma 生长模式难以准确评估的问题。通过结合生物物理模型与生成框架，合成真实3D脑MRI，实现肿瘤生长的可视化与数据分析。**

- **链接: [https://arxiv.org/pdf/2603.04058](https://arxiv.org/pdf/2603.04058)**

> **作者:** Valentin Biller; Niklas Bubeck; Lucas Zimmer; Ayhan Can Erdur; Sandeep Nagar; Anke Meyer-Baese; Daniel Rückert; Benedikt Wiestler; Jonas Weidner
>
> **摘要:** Glioblastoma exhibits diverse, infiltrative, and patient-specific growth patterns that are only partially visible on routine MRI, making it difficult to reliably assess true tumor extent and personalize treatment planning and follow-up. We present a biophysically-conditioned generative framework that synthesizes biologically realistic 3D brain MRI volumes from estimated, spatially continuous tumor-concentration fields. Our approach combines a generative model with tumor-infiltration maps that can be propagated through time using a biophysical growth model, enabling fine-grained control over tumor shape and growth while preserving patient anatomy. This enables us to synthesize consistent tumor growth trajectories directly in the space of real patients, providing interpretable, controllable estimation of tumor infiltration and progression beyond what is explicitly observed in imaging. We evaluate the framework on longitudinal glioblastoma cases and demonstrate that it can generate temporally coherent sequences with realistic changes in tumor appearance and surrounding tissue response. These results suggest that integrating mechanistic tumor growth priors with modern generative modeling can provide a practical tool for patient-specific progression visualization and for generating controlled synthetic data to support downstream neuro-oncology workflows. In longitudinal extrapolation, we achieve a consistent 75% Dice overlap with the biophysical model while maintaining a constant PSNR of 25 in the surrounding tissue. Our code is available at: this https URL
>
---
#### [new 049] Revisiting the Role of Foundation Models in Cell-Level Histopathological Image Analysis under Small-Patch Constraints -- Effects of Training Data Scale and Blur Perturbations on CNNs and Vision Transformers
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于细胞级病理图像分析任务，解决小尺寸图像块分类问题。研究比较了不同模型在有限数据下的表现，发现任务专用模型更有效。**

- **链接: [https://arxiv.org/pdf/2603.04081](https://arxiv.org/pdf/2603.04081)**

> **作者:** Hiroki Kagiyama; Toru Nagasaka; Yukari Adachi; Takaaki Tachibana; Ryota Ito; Mitsugu Fujita; Kimihiro Yamashita; Yoshihiro Kakeji
>
> **摘要:** Background and objective: Cell-level pathological image analysis requires working with extremely small image patches (40x40 pixels), far below standard ImageNet resolutions. It remains unclear whether modern deep learning architectures and foundation models can learn robust and scalable representations under this constraint. We systematically evaluated architectural suitability and data-scale effects for small-patch cell classification. Methods: We analyzed 303 colorectal cancer specimens with CD103/CD8 immunostaining, generating 185,432 annotated cell images. Eight task-specific architectures were trained from scratch at multiple data scales (FlagLimit: 256--16,384 samples per class), and three foundation models were evaluated via linear probing and fine-tuning after resizing inputs to 224x224 pixels. Robustness to blur was assessed using pre- and post-resize Gaussian perturbations. Results: Task-specific models improved consistently with increasing data scale, whereas foundation models saturated at moderate sample sizes. A Vision Transformer optimized for small patches (CustomViT) achieved the highest accuracy, outperforming all foundation models with substantially lower inference cost. Blur robustness was comparable across architectures, with no qualitative advantage observed for foundation models. Conclusion: For cell-level classification under extreme spatial constraints, task-specific architectures are more effective and efficient than foundation models once sufficient training data are available. Higher clean accuracy does not imply superior robustness, and large pre-trained models offer limited benefit in the small-patch regime.
>
---
#### [new 050] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于点云质量评估任务，解决NR-PCQA泛化能力差的问题。通过引入质量感知的域适应方法，提升模型在不同数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2603.03726](https://arxiv.org/pdf/2603.03726)**

> **作者:** Guohua Zhang; Jian Jin; Meiqin Liu; Chao Yao; Weisi Lin
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** No-Reference Point Cloud Quality Assessment (NR-PCQA) still struggles with generalization, primarily due to the scarcity of annotated point cloud datasets. Since the Human Visual System (HVS) drives perceptual quality assessment independently of media types, prior knowledge on quality learned from images can be repurposed for point clouds. This insight motivates adopting Unsupervised Domain Adaptation (UDA) to transfer quality-relevant priors from labeled images to unlabeled point clouds. However, existing UDA-based PCQA methods often overlook key characteristics of perceptual quality, such as sensitivity to quality ranking and quality-aware feature alignment, thereby limiting their effectiveness. To address these issues, we propose a novel Quality-aware Domain adaptation framework for PCQA, termed QD-PCQA. The framework comprises two main components: i) a Rank-weighted Conditional Alignment (RCA) strategy that aligns features under consistent quality levels and adaptively emphasizes misranked samples to reinforce perceptual quality ranking awareness; and ii) a Quality-guided Feature Augmentation (QFA) strategy, which includes quality-guided style mixup, multi-layer extension, and dual-domain augmentation modules to augment perceptual feature alignment. Extensive cross-domain experiments demonstrate that QD-PCQA significantly improves generalization in NR-PCQA tasks. The code is available at this https URL.
>
---
#### [new 051] ProFound: A moderate-sized vision foundation model for multi-task prostate imaging
- **分类: cs.CV**

- **简介: 该论文提出ProFound，一个用于前列腺多参数MRI的视觉基础模型，解决临床任务自动化难题。通过自监督预训练，提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2603.03961](https://arxiv.org/pdf/2603.03961)**

> **作者:** Yipei Wang; Yinsong Xu; Weixi Yi; Shaheer Ullah Saeed; Natasha Thorley; Alexander Ng; Yukun Zhou; Wen Yan; Dean Barratt; Shonit Punwani; Veeru Kasivisvanathan; Mark Emberton; Daniel C. Alexander; Yipeng Hu
>
> **摘要:** Many diagnostic and therapeutic clinical tasks for prostate cancer increasingly rely on multi-parametric MRI. Automating these tasks is challenging because they necessitate expert interpretations, which are difficult to scale to capitalise on modern deep learning. Although modern automated systems achieve expert-level performance in isolated tasks, their general clinical utility remains limited by the requirement of large task-specific labelled datasets. In this paper, we present ProFound, a domain-specialised vision foundation model for volumetric prostate mpMRI. ProFound is pre-trained using several variants of self-supervised approaches on a diverse, multi-institutional collection of 5,000 patients, with a total of over 22,000 unique 3D MRI volumes (over 1,800,000 2D image slices). We conducted a systematic evaluation of ProFound across a broad spectrum of $11$ downstream clinical tasks on over 3,000 independent patients, including prostate cancer detection, Gleason grading, lesion localisation, gland volume estimation, zonal and surrounding structure segmentation. Experimental results demonstrate that finetuned ProFound consistently outperforms or remains competitive with state-of-the-art specialised models and existing medical vision foundation models trained/finetuned on the same data.
>
---
#### [new 052] Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild
- **分类: cs.CV**

- **简介: 该论文属于文档解析任务，旨在解决VLMs在真实物理场景中表现不佳的问题。通过构建Real5-OmniDocBench基准，进行全尺度物理重建，分析性能下降原因。**

- **链接: [https://arxiv.org/pdf/2603.04205](https://arxiv.org/pdf/2603.04205)**

> **作者:** Changda Zhou; Ziyue Gao; Xueqing Wang; Tingquan Gao; Cheng Cui; Jing Tang; Yi Liu
>
> **摘要:** While Vision-Language Models (VLMs) achieve near-perfect scores on digital document benchmarks like OmniDocBench, their performance in the unpredictable physical world remains largely unknown due to the lack of controlled yet realistic evaluations. We introduce Real5-OmniDocBench, the first benchmark that performs a full-scale, one-to-one physical reconstruction of the entire OmniDocBench v1.5 (1,355 images) across five critical real-world scenarios: Scanning, Warping, Screen-Photography, Illumination, and Skew. Unlike prior benchmark that either lack digital correspondence or employ partial sampling, our complete ground-truth mapping enables, for the first time, rigorous factor-wise attribution of performance degradation-allowing us to pinpoint whether failures stem from geometric distortions, optical artifacts, or model limitations. Our benchmark establishes a challenging new standard for the community, demonstrating that the 'reality gap' in document parsing is far from closed, and provides a diagnostic tool to guide the development of truly resilient document intelligence.
>
---
#### [new 053] Volumetric Directional Diffusion: Anchoring Uncertainty Quantification in Anatomical Consensus for Ambiguous Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，解决模糊病变分割中的不确定性量化问题。提出VDD方法，在保持结构一致性的同时准确捕捉专家分歧，提升分割可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04024](https://arxiv.org/pdf/2603.04024)**

> **作者:** Chao Wu; Kangxian Xie; Mingchen Gao
>
> **摘要:** Equivocal 3D lesion segmentation exhibits high inter-observer variability. Conventional deterministic models ignore this aleatoric uncertainty, producing over-confident masks that obscure clinical risks. Conversely, while generative methods (e.g., standard diffusion) capture sample diversity, recovering complex topology from pure noise frequently leads to severe structural fractures and out-of-distribution anatomical hallucinations. To resolve this fidelity-diversity trade-off, we propose Volumetric Directional Diffusion (VDD). Unlike standard diffusion models that denoise isotropic Gaussian noise, VDD mathematically anchors the generative trajectory to a deterministic consensus prior. By restricting the generative search space to iteratively predict a 3D boundary residual field, VDD accurately explores the fine-grained geometric variations inherent in expert disagreements without risking topological collapse. Extensive validation on three multi-rater datasets (LIDC-IDRI, KiTS21, and ISBI 2015) demonstrates that VDD achieves state-of-the-art uncertainty quantification (significantly improving GED and CI) while remaining highly competitive in segmentation accuracy against deterministic upper bounds. Ultimately, VDD provides clinicians with anatomically coherent uncertainty maps, enabling safer decision-making and mitigating risks in downstream tasks (e.g., radiotherapy planning or surgical margin assessment).
>
---
#### [new 054] MPFlow: Multi-modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于MRI重建任务，解决零样本重建中的幻觉问题。通过引入多模态信息和自监督预训练，提升重建的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.03710](https://arxiv.org/pdf/2603.03710)**

> **作者:** Seunghoi Kim; Chen Jin; Henry F. J. Tregidgo; Matteo Figini; Daniel C. Alexander
>
> **摘要:** Zero-shot MRI reconstruction relies on generative priors, but single-modality unconditional priors produce hallucinations under severe ill-posedness. In many clinical workflows, complementary MRI acquisitions (e.g. high-quality structural scans) are routinely available, yet existing reconstruction methods lack mechanisms to leverage this additional information. We propose MPFlow, a zero-shot multi-modal reconstruction framework built on rectified flow that incorporates auxiliary MRI modalities at inference time without retraining the generative prior to improve anatomical fidelity. Cross-modal guidance is enabled by our proposed self-supervised pretraining strategy, Patch-level Multi-modal MR Image Pretraining (PAMRI), which learns shared representations across modalities. Sampling is jointly guided by data consistency and cross-modal feature alignment using pre-trained PAMRI, systematically suppressing intrinsic and extrinsic hallucinations. Extensive experiments on HCP and BraTS show that MPFlow matches diffusion baselines on image quality using only 20% of sampling steps while reducing tumor hallucinations by more than 15% (segmentation dice score). This demonstrates that cross-modal guidance enables more reliable and efficient zero-shot MRI reconstruction.
>
---
#### [new 055] SimpliHuMoN: Simplifying Human Motion Prediction
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体运动预测任务，旨在解决轨迹与姿态联合预测难题。提出一种基于Transformer的简化模型，有效捕捉时空关系，实现多种任务的统一处理。**

- **链接: [https://arxiv.org/pdf/2603.04399](https://arxiv.org/pdf/2603.04399)**

> **作者:** Aadya Agrawal; Alexander Schwing
>
> **备注:** 19 pages, 7 figures. Preprint
>
> **摘要:** Human motion prediction combines the tasks of trajectory forecasting and human pose prediction. For each of the two tasks, specialized models have been developed. Combining these models for holistic human motion prediction is non-trivial, and recent methods have struggled to compete on established benchmarks for individual tasks. To address this, we propose a simple yet effective transformer-based model for human motion prediction. The model employs a stack of self-attention modules to effectively capture both spatial dependencies within a pose and temporal relationships across a motion sequence. This simple, streamlined, end-to-end model is sufficiently versatile to handle pose-only, trajectory-only, and combined prediction tasks without task-specific modifications. We demonstrate that this approach achieves state-of-the-art results across all tasks through extensive experiments on a wide range of benchmark datasets, including Human3.6M, AMASS, ETH-UCY, and 3DPW.
>
---
#### [new 056] IntroductionDMD-augmented Unpaired Neural Schrödinger Bridge for Ultra-Low Field MRI Enhancement
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像增强任务，旨在提升64mT脑MRI的图像质量。通过无配对数据翻译框架，增强真实感并保持解剖结构。**

- **链接: [https://arxiv.org/pdf/2603.03769](https://arxiv.org/pdf/2603.03769)**

> **作者:** Youngmin Kim; Jaeyun Shin; Jeongchan Kim; Taehoon Lee; Jaemin Kim; Peter Hsu; Jelle Veraart; Jong Chul Ye
>
> **摘要:** Ultra Low Field (64 mT) brain MRI improves accessibility but suffers from reduced image quality compared to 3 T. As paired 64 mT - 3 T scans are scarce, we propose an unpaired 64 mT $\rightarrow$ 3 T translation framework that enhances realism while preserving anatomy. Our method builds upon the Unpaired Neural Schrödinge Bridge (UNSB) with multi-step refinement. To strengthen target distribution alignment, we augment the adversarial objective with DMD2-style diffusion-guided distribution matching using a frozen 3T diffusion teacher. To explicitly constrain global structure beyond patch-level correspondence, we combine PatchNCE with an Anatomical Structure Preservation (ASP) regularizer that enforces soft foreground background consistency and boundary aware constraints. Evaluated on two disjoint cohorts, the proposed framework achieves an improved realism structure trade-off, enhancing distribution level realism on unpaired benchmarks while increasing structural fidelity on the paired cohort compared to unpaired baselines.
>
---
#### [new 057] Towards Generalized Multimodal Homography Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态单应性估计任务，旨在解决模型在未见模态上性能下降的问题。通过生成合成数据和设计网络提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03956](https://arxiv.org/pdf/2603.03956)**

> **作者:** Jinkun You; Jiaxin Cheng; Jie Zhang; Yicong Zhou
>
> **摘要:** Supervised and unsupervised homography estimation methods depend on image pairs tailored to specific modalities to achieve high accuracy. However, their performance deteriorates substantially when applied to unseen modalities. To address this issue, we propose a training data synthesis method that generates unaligned image pairs with ground-truth offsets from a single input image. Our approach renders the image pairs with diverse textures and colors while preserving their structural information. These synthetic data empower the trained model to achieve greater robustness and improved generalization across various domains. Additionally, we design a network to fully leverage cross-scale information and decouple color information from feature representations, thus improving estimation accuracy. Extensive experiments show that our training data synthesis method improves generalization performance. The results also confirm the effectiveness of the proposed network.
>
---
#### [new 058] SPRINT: Semi-supervised Prototypical Representation for Few-Shot Class-Incremental Tabular Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SPRINT，解决小样本类增量学习任务，针对表格数据设计，利用半监督方法提升新类表示并保留旧知识。**

- **链接: [https://arxiv.org/pdf/2603.04321](https://arxiv.org/pdf/2603.04321)**

> **作者:** Umid Suleymanov; Murat Kantarcioglu; Kevin S Chan; Michael De Lucia; Kevin Hamlen; Latifur Khan; Sharad Mehrotra; Ananthram Swami; Bhavani Thuraisingham
>
> **备注:** Under Review
>
> **摘要:** Real-world systems must continuously adapt to novel concepts from limited data without forgetting previously acquired knowledge. While Few-Shot Class-Incremental Learning (FSCIL) is established in computer vision, its application to tabular domains remains largely unexplored. Unlike images, tabular streams (e.g., logs, sensors) offer abundant unlabeled data, a scarcity of expert annotations and negligible storage costs, features ignored by existing vision-based methods that rely on restrictive buffers. We introduce SPRINT, the first FSCIL framework tailored for tabular distributions. SPRINT introduces a mixed episodic training strategy that leverages confidence-based pseudo-labeling to enrich novel class representations and exploits low storage costs to retain base class history. Extensive evaluation across six diverse benchmarks spanning cybersecurity, healthcare, and ecological domains, demonstrates SPRINT's cross-domain robustness. It achieves a state-of-the-art average accuracy of 77.37% (5-shot), outperforming the strongest incremental baseline by 4.45%.
>
---
#### [new 059] Tracking Feral Horses in Aerial Video Using Oriented Bounding Boxes
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于目标跟踪任务，旨在解决多马群在航拍视频中因背景复杂、目标小、密度高导致的跟踪困难问题。通过引入带方向的边界框和头部方向估计方法提升跟踪精度。**

- **链接: [https://arxiv.org/pdf/2603.03604](https://arxiv.org/pdf/2603.03604)**

> **作者:** Saeko Takizawa; Tamao Maeda; Shinya Yamamoto; Hiroaki Kawashima
>
> **备注:** Author's version of the paper presented at AROB-ISBC 2026
>
> **摘要:** The social structures of group-living animals such as feral horses are diverse and remain insufficiently understood, even within a single species. To investigate group dynamics, aerial videos are often utilized to track individuals and analyze their movement trajectories, which are essential for evaluating inter-individual interactions and comparing social behaviors. Accurate individual tracking is therefore crucial. In multi-animal tracking, axis-aligned bounding boxes (bboxes) are widely used; however, for aerial top-view footage of entire groups, their performance degrades due to complex backgrounds, small target sizes, high animal density, and varying body orientations. To address this issue, we employ oriented bounding boxes (OBBs), which include rotation angles and reduce unnecessary background. Nevertheless, current OBB detectors such as YOLO-OBB restrict angles within a 180$^{\circ}$ range, making it impossible to distinguish head from tail and often causing sudden 180$^{\circ}$ flips across frames, which severely disrupts continuous tracking. To overcome this limitation, we propose a head-orientation estimation method that crops OBB-centered patches, applies three detectors (head, tail, and head-tail), and determines the final label through IoU-based majority voting. Experiments using 299 test images show that our method achieves 99.3% accuracy, outperforming individual models, demonstrating its effectiveness for robust OBB-based tracking.
>
---
#### [new 060] N-gram Injection into Transformers for Dynamic Language Model Adaptation in Handwritten Text Recognition
- **分类: cs.CV**

- **简介: 该论文属于手写文本识别任务，旨在解决语言分布变化导致的模型性能下降问题。通过引入外部n-gram，在推理时动态调整语言模型，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.03930](https://arxiv.org/pdf/2603.03930)**

> **作者:** Florent Meyer; Laurent Guichard; Denis Coquenet; Guillaume Gravier; Yann Soullard; Bertrand Coüasnon
>
> **摘要:** Transformer-based encoder-decoder networks have recently achieved impressive results in handwritten text recognition, partly thanks to their auto-regressive decoder which implicitly learns a language model. However, such networks suffer from a large performance drop when evaluated on a target corpus whose language distribution is shifted from the source text seen during training. To retain recognition accuracy despite this language shift, we propose an external n-gram injection (NGI) for dynamic adaptation of the network's language modeling at inference time. Our method allows switching to an n-gram language model estimated on a corpus close to the target distribution, therefore mitigating bias without any extra training on target image-text pairs. We opt for an early injection of the n-gram into the transformer decoder so that the network learns to fully leverage text-only data at the low additional cost of n-gram inference. Experiments on three handwritten datasets demonstrate that the proposed NGI significantly reduces the performance gap between source and target corpora.
>
---
#### [new 061] DeNuC: Decoupling Nuclei Detection and Classification in Histopathology
- **分类: cs.CV**

- **简介: 该论文针对病理图像中的细胞核检测与分类任务，解决传统模型在该任务上表现不佳的问题。提出DeNuC方法，通过解耦检测与分类流程，提升模型性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2603.04240](https://arxiv.org/pdf/2603.04240)**

> **作者:** Zijiang Yang; Chen Kuang; Dongmei Fu
>
> **备注:** 10 pages
>
> **摘要:** Pathology Foundation Models (FMs) have shown strong performance across a wide range of pathology image representation and diagnostic tasks. However, FMs do not exhibit the expected performance advantage over traditional specialized models in Nuclei Detection and Classification (NDC). In this work, we reveal that jointly optimizing nuclei detection and classification leads to severe representation degradation in FMs. Moreover, we identify that the substantial intrinsic disparity in task difficulty between nuclei detection and nuclei classification renders joint NDC optimization unnecessarily computationally burdensome for the detection stage. To address these challenges, we propose DeNuC, a simple yet effective method designed to break through existing bottlenecks by Decoupling Nuclei detection and Classification. DeNuC employs a lightweight model for accurate nuclei localization, subsequently leveraging a pathology FM to encode input images and query nucleus-specific features based on the detected coordinates for classification. Extensive experiments on three widely used benchmarks demonstrate that DeNuC effectively unlocks the representational potential of FMs for NDC and significantly outperforms state-of-the-art methods. Notably, DeNuC improves F1 scores by 4.2% and 3.6% (or higher) on the BRCAM2C and PUMA datasets, respectively, while using only 16% (or fewer) trainable parameters compared to other methods. Code is available at this https URL.
>
---
#### [new 062] PhyPrompt: RL-based Prompt Refinement for Physically Plausible Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到视频生成任务，旨在解决生成视频违反物理规律的问题。通过强化学习优化提示词，提升视频的物理合理性与语义一致性。**

- **链接: [https://arxiv.org/pdf/2603.03505](https://arxiv.org/pdf/2603.03505)**

> **作者:** Shang Wu; Chenwei Xu; Zhuofan Xia; Weijian Li; Lie Lu; Pranav Maneriker; Fan Du; Manling Li; Han Liu
>
> **摘要:** State-of-the-art text-to-video (T2V) generators frequently violate physical laws despite high visual quality. We show this stems from insufficient physical constraints in prompts rather than model limitations: manually adding physics details reliably produces physically plausible videos, but requires expertise and does not scale. We present PhyPrompt, a two-stage reinforcement learning framework that automatically refines prompts for physically realistic generation. First, we fine-tune a large language model on a physics-focused Chain-of-Thought dataset to integrate principles like object motion and force interactions while preserving user intent. Second, we apply Group Relative Policy Optimization with a dynamic reward curriculum that initially prioritizes semantic fidelity, then progressively shifts toward physical commonsense. This curriculum achieves synergistic optimization: PhyPrompt-7B reaches 40.8\% joint success on VideoPhy2 (8.6pp gain), improving physical commonsense by 11pp (55.8\% to 66.8\%) while simultaneously increasing semantic adherence by 4.4pp (43.4\% to 47.8\%). Remarkably, our curriculum exceeds single-objective training on both metrics, demonstrating compositional prompt discovery beyond conventional multi-objective trade-offs. PhyPrompt outperforms GPT-4o (+3.8\% joint) and DeepSeek-V3 (+2.2\%, 100$\times$ larger) using only 7B parameters. The approach transfers zero-shot across diverse T2V architectures (Lavie, VideoCrafter2, CogVideoX-5B) with up to 16.8\% improvement, establishing that domain-specialized reinforcement learning with compositional curricula surpasses general-purpose scaling for physics-aware generation.
>
---
#### [new 063] LeafInst - Unified Instance Segmentation Network for Fine-Grained Forestry Leaf Phenotype Analysis: A New UAV based Benchmark
- **分类: cs.CV**

- **简介: 该论文属于实例分割任务，旨在解决开放环境下树木细粒度叶片分析的问题。构建了首个林业叶片数据集，提出LeafInst框架提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.03616](https://arxiv.org/pdf/2603.03616)**

> **作者:** Taige Luo; Junru Xie; Chenyang Fan; Bingrong Liu; Ruisheng Wang; Yang Shao; Sheng Xu; Lin Cao
>
> **摘要:** Intelligent forest tree breeding has advanced plant phenotyping, yet existing research largely focuses on large-leaf agricultural crops, with limited attention to fine-grained leaf analysis of sapling trees in open-field environments. Natural scenes introduce challenges including scale variation, illumination changes, and irregular leaf morphology. To address these issues, we collected UAV RGB imagery of field-grown saplings and constructed the Poplar-leaf dataset, containing 1,202 branches and 19,876 pixel-level annotated leaf instances. To our knowledge, this is the first instance segmentation dataset specifically designed for forestry leaves in open-field conditions. We propose LeafInst, a novel segmentation framework tailored for irregular and multi-scale leaf structures. The model integrates an Asymptotic Feature Pyramid Network (AFPN) for multi-scale perception, a Dynamic Asymmetric Spatial Perception (DASP) module for irregular shape modeling, and a dual-residual Dynamic Anomalous Regression Head (DARH) with Top-down Concatenation decoder Feature Fusion (TCFU) to improve detection and segmentation performance. On Poplar-leaf, LeafInst achieves 68.4 mAP, outperforming YOLOv11 by 7.1 percent and MaskDINO by 6.5 percent. On the public PhenoBench benchmark, it reaches 52.7 box mAP, exceeding MaskDINO by 3.4 percent. Additional experiments demonstrate strong generalization and practical utility for large-scale leaf phenotyping.
>
---
#### [new 064] Any2Any: Unified Arbitrary Modality Translation for Remote Sensing
- **分类: cs.CV**

- **简介: 该论文提出Any2Any，解决多模态遥感图像翻译问题，通过共享潜在空间实现任意模态间的统一翻译，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04114](https://arxiv.org/pdf/2603.04114)**

> **作者:** Haoyang Chen; Jing Zhang; Hebaixu Wang; Shiqin Wang; Pohsun Huang; Jiayuan Li; Haonan Guo; Di Wang; Zheng Wang; Bo Du
>
> **摘要:** Multi-modal remote sensing imagery provides complementary observations of the same geographic scene, yet such observations are frequently incomplete in practice. Existing cross-modal translation methods treat each modality pair as an independent task, resulting in quadratic complexity and limited generalization to unseen modality combinations. We formulate Any-to-Any translation as inference over a shared latent representation of the scene, where different modalities correspond to partial observations of the same underlying semantics. Based on this formulation, we propose Any2Any, a unified latent diffusion framework that projects heterogeneous inputs into a geometrically aligned latent space. Such structure performs anchored latent regression with a shared backbone, decoupling modality-specific representation learning from semantic mapping. Moreover, lightweight target-specific residual adapters are used to correct systematic latent mismatches without increasing inference complexity. To support learning under sparse but connected supervision, we introduce RST-1M, the first million-scale remote sensing dataset with paired observations across five sensing modalities, providing supervision anchors for any-to-any translation. Experiments across 14 translation tasks show that Any2Any consistently outperforms pairwise translation methods and exhibits strong zero-shot generalization to unseen modality pairs. Code and models will be available at this https URL.
>
---
#### [new 065] Confidence-aware Monocular Depth Estimation for Minimally Invasive Surgery
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在提升微创手术中深度估计的准确性与可靠性。针对图像噪声和伪影问题，提出一种具有置信度感知的框架，通过置信度目标、损失函数和推理时置信度预测，提高模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03571](https://arxiv.org/pdf/2603.03571)**

> **作者:** Muhammad Asad; Emanuele Colleoni; Pritesh Mehta; Nicolas Toussaint; Ricardo Sanchez-Matilla; Maria Robu; Faisal Bashir; Rahim Mohammadi; Imanol Luengo; Danail Stoyanov
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Purpose: Monocular depth estimation (MDE) is vital for scene understanding in minimally invasive surgery (MIS). However, endoscopic video sequences are often contaminated by smoke, specular reflections, blur, and occlusions, limiting the accuracy of MDE models. In addition, current MDE models do not output depth confidence, which could be a valuable tool for improving their clinical reliability. Methods: We propose a novel confidence-aware MDE framework featuring three significant contributions: (i) Calibrated confidence targets: an ensemble of fine-tuned stereo matching models is used to capture disparity variance into pixel-wise confidence probabilities; (ii) Confidence-aware loss: Baseline MDE models are optimized with confidence-aware loss functions, utilizing pixel-wise confidence probabilities such that reliable pixels dominate training; and (iii) Inference-time confidence: a confidence estimation head is proposed with two convolution layers to predict per-pixel confidence at inference, enabling assessment of depth reliability. Results: Comprehensive experimental validation across internal and public datasets demonstrates that our framework improves depth estimation accuracy and can robustly quantify the prediction's confidence. On the internal clinical endoscopic dataset (StereoKP), we improve dense depth estimation accuracy by ~8% as compared to the baseline model. Conclusion: Our confidence-aware framework enables improved accuracy of MDE models in MIS, addressing challenges posed by noise and artifacts in pre-clinical and clinical data, and allows MDE models to provide confidence maps that may be used to improve their reliability for clinical applications.
>
---
#### [new 066] WSI-INR: Implicit Neural Representations for Lesion Segmentation in Whole-Slide Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决全切片图像中病灶分割的碎片化和分辨率敏感问题。提出WSI-INR框架，利用隐式神经表示实现连续空间建模与多分辨率一致特征提取。**

- **链接: [https://arxiv.org/pdf/2603.03749](https://arxiv.org/pdf/2603.03749)**

> **作者:** Yunheng Wu; Wenqi Huang; Liangyi Wang; Masahiro Oda; Yuichiro Hayashi; Daniel Rueckert; Kensaku Mori
>
> **备注:** 11 page, 4 figures
>
> **摘要:** Whole-slide images (WSIs) are fundamental for computational pathology, where accurate lesion segmentation is critical for clinical decision making. Existing methods partition WSIs into discrete patches, disrupting spatial continuity and treating multi-resolution views as independent samples, which leads to spatially fragmented segmentation and reduced robustness to resolution variations. To address the issues, we propose WSI-INR, a novel patch-free framework based on Implicit Neural Representations (INRs). WSI-INR models the WSI as a continuous implicit function mapping spatial coordinates directly to tissue semantics features, outputting segmentation results while preserving intrinsic spatial information across the entire slide. In the WSI-INR, we incorporate multi-resolution hash grid encoding to regard different resolution levels as varying sampling densities of the same continuous tissue, achieving a consistent feature representation across resolutions. In addition, by jointly training a shared INR decoder, WSI-INR can capture general priors across different cases. Experimental results showed that WSI-INR maintains robust segmentation performance across resolutions; at Base/4, our resolution-specific optimization improves Dice score by +26.11%, while U-Net and TransUNet decrease by 54.28% and 36.18%, respectively. Crucially, this work enables INRs to segment highly heterogeneous pathological lesions beyond structurally consistent anatomical tissues, offering a fresh perspective for pathological analysis.
>
---
#### [new 067] Beyond Accuracy: Evaluating Visual Grounding In Multimodal Medical Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态医学推理任务，旨在解决现有评估协议无法准确衡量视觉依赖性的问题。通过引入新的评估指标，分析模型在医学VQA任务中的视觉 grounding 能力。**

- **链接: [https://arxiv.org/pdf/2603.03437](https://arxiv.org/pdf/2603.03437)**

> **作者:** Anas Zafar; Leema Krishna Murali; Ashish Vashist
>
> **备注:** 12 pages, 2 figures, 2 tables, medical VQA / multimodal reasoning evaluation
>
> **摘要:** Recent work shows that text-only reinforcement learning with verifiable rewards (RLVR) can match or outperform image-text RLVR on multimodal medical VQA benchmarks, suggesting current evaluation protocols may fail to measure causal visual dependence. We introduce a counterfactual evaluation framework using real, blank, and shuffled images across four medical VQA benchmarks: PathVQA, PMC-VQA, SLAKE, and VQA-RAD. Beyond accuracy, we measure Visual Reliance Score (VRS), Image Sensitivity (IS), and introduce Hallucinated Visual Reasoning Rate (HVRR) to detect cases where models generate visual claims despite producing image-invariant answers. Our findings reveal that RLVR improves accuracy while degrading visual grounding: text-only RLVR achieves negative VRS on PathVQA (-0.09), performing better with mismatched images, while image-text RLVR reduces image sensitivity to 39.8% overall despite improving accuracy. On VQA-RAD, both variants achieve 63% accuracy through different mechanisms: text-only RLVR retains 81% performance with blank images, while image-text RLVR shows only 29% image sensitivity. Models generate visual claims in 68-74% of responses, yet 38-43% are ungrounded (HVRR). These findings demonstrate that accuracy-only rewards enable shortcut exploitation, and progress requires grounding-aware evaluation protocols and training objectives that explicitly enforce visual dependence.
>
---
#### [new 068] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于CAD生成任务，旨在解决命令序列无法支持实体选择及拓扑错误的问题。提出Pointer-CAD框架，通过指针选择几何实体，提升建模精度。**

- **链接: [https://arxiv.org/pdf/2603.04337](https://arxiv.org/pdf/2603.04337)**

> **作者:** Dacheng Qi; Chenyu Wang; Jingwei Xu; Tianzhe Chu; Zibo Zhao; Wen Liu; Wenrui Ding; Yi Ma; Shenghua Gao
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Constructing computer-aided design (CAD) models is labor-intensive but essential for engineering and manufacturing. Recent advances in Large Language Models (LLMs) have inspired the LLM-based CAD generation by representing CAD as command sequences. But these methods struggle in practical scenarios because command sequence representation does not support entity selection (e.g. faces or edges), limiting its ability to support complex editing operations such as chamfer or fillet. Further, the discretization of a continuous variable during sketch and extrude operations may result in topological errors. To address these limitations, we present Pointer-CAD, a novel LLM-based CAD generation framework that leverages a pointer-based command sequence representation to explicitly incorporate the geometric information of B-rep models into sequential modeling. In particular, Pointer-CAD decomposes CAD model generation into steps, conditioning the generation of each subsequent step on both the textual description and the B-rep generated from previous steps. Whenever an operation requires the selection of a specific geometric entity, the LLM predicts a Pointer that selects the most feature-consistent candidate from the available set. Such a selection operation also reduces the quantization error in the command sequence-based representation. To support the training of Pointer-CAD, we develop a data annotation pipeline that produces expert-level natural language descriptions and apply it to build a dataset of approximately 575K CAD models. Extensive experimental results demonstrate that Pointer-CAD effectively supports the generation of complex geometric structures and reduces segmentation error to an extremely low level, achieving a significant improvement over prior command sequence methods, thereby significantly mitigating the topological inaccuracies introduced by quantization error.
>
---
#### [new 069] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression
- **分类: cs.CV**

- **简介: 该论文属于分布式多视角图像压缩任务，旨在解决现有方法忽略视图间相关性差异的问题。提出OmniParallax注意力机制和ParaHydra框架，提升压缩效率。**

- **链接: [https://arxiv.org/pdf/2603.03615](https://arxiv.org/pdf/2603.03615)**

> **作者:** Haotian Zhang; Feiyue Long; Yixin Yu; Jian Xue; Haocheng Tang; Tongda Xu; Zhenning Shi; Yan Wang; Siwei Ma; Jiaqi Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multi-view image compression (MIC) aims to achieve high compression efficiency by exploiting inter-image correlations, playing a crucial role in 3D applications. As a subfield of MIC, distributed multi-view image compression (DMIC) offers performance comparable to MIC while eliminating the need for inter-view information at the encoder side. However, existing methods in DMIC typically treat all images equally, overlooking the varying degrees of correlation between different views during decoding, which leads to suboptimal coding performance. To address this limitation, we propose a novel $\textbf{OmniParallax Attention Mechanism}$ (OPAM), which is a general mechanism for explicitly modeling correlations and aligned features between arbitrary pairs of information sources. Building upon OPAM, we propose a Parallax Multi Information Fusion Module (PMIFM) to adaptively integrate information from different sources. PMIFM is incorporated into both the joint decoder and the entropy model to construct our end-to-end DMIC framework, $\textbf{ParaHydra}$. Extensive experiments demonstrate that $\textbf{ParaHydra}$ is $\textbf{the first DMIC method}$ to significantly surpass state-of-the-art MIC codecs, while maintaining low computational overhead. Performance gains become more pronounced as the number of input views increases. Compared with LDMIC, $\textbf{ParaHydra}$ achieves bitrate savings of $\textbf{19.72%}$ on WildTrack(3) and up to $\textbf{24.18%}$ on WildTrack(6), while significantly improving coding efficiency (as much as $\textbf{65}\times$ in decoding and $\textbf{34}\times$ in encoding).
>
---
#### [new 070] An Effective Data Augmentation Method by Asking Questions about Scene Text Images
- **分类: cs.CV**

- **简介: 该论文属于文本识别任务，旨在提升场景文本和手写文本的识别准确率。通过生成问题-答案对进行数据增强，引导模型更细致地理解文本结构，从而降低识别错误率。**

- **链接: [https://arxiv.org/pdf/2603.03580](https://arxiv.org/pdf/2603.03580)**

> **作者:** Xu Yao; Lei Kang
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Scene text recognition (STR) and handwritten text recognition (HTR) face significant challenges in accurately transcribing textual content from images into machine-readable formats. Conventional OCR models often predict transcriptions directly, which limits detailed reasoning about text structure. We propose a VQA-inspired data augmentation framework that strengthens OCR training through structured question-answering tasks. For each image-text pair, we generate natural-language questions probing character-level attributes such as presence, position, and frequency, with answers derived from ground-truth text. These auxiliary tasks encourage finer-grained reasoning, and the OCR model aligns visual features with textual queries to jointly reason over images and questions. Experiments on WordArt and Esposalles datasets show consistent improvements over baseline models, with significant reductions in both CER and WER. Our code is publicly available at this https URL.
>
---
#### [new 071] Phys4D: Fine-Grained Physics-Consistent 4D Modeling from Video Diffusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Phys4D，解决视频生成中物理一致性不足的问题。通过三阶段训练提升4D场景的物理合理性，增强时空一致性与动态稳定性。**

- **链接: [https://arxiv.org/pdf/2603.03485](https://arxiv.org/pdf/2603.03485)**

> **作者:** Haoran Lu; Shang Wu; Jianshu Zhang; Maojiang Su; Guo Ye; Chenwei Xu; Lie Lu; Pranav Maneriker; Fan Du; Manling Li; Zhaoran Wang; Han Liu
>
> **摘要:** Recent video diffusion models have achieved impressive capabilities as large-scale generative world models. However, these models often struggle with fine-grained physical consistency, exhibiting physically implausible dynamics over time. In this work, we present \textbf{Phys4D}, a pipeline for learning physics-consistent 4D world representations from video diffusion models. Phys4D adopts \textbf{a three-stage training paradigm} that progressively lifts appearance-driven video diffusion models into physics-consistent 4D world representations. We first bootstrap robust geometry and motion representations through large-scale pseudo-supervised pretraining, establishing a foundation for 4D scene modeling. We then perform physics-grounded supervised fine-tuning using simulation-generated data, enforcing temporally consistent 4D dynamics. Finally, we apply simulation-grounded reinforcement learning to correct residual physical violations that are difficult to capture through explicit supervision. To evaluate fine-grained physical consistency beyond appearance-based metrics, we introduce a set of \textbf{4D world consistency evaluation} that probe geometric coherence, motion stability, and long-horizon physical plausibility. Experimental results demonstrate that Phys4D substantially improves fine-grained spatiotemporal and physical consistency compared to appearance-driven baselines, while maintaining strong generative performance. Our project page is available at this https URL
>
---
#### [new 072] Detection and Identification of Penguins Using Appearance and Motion Features
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于目标检测与识别任务，解决penguin在复杂环境中的检测与跟踪问题。通过融合外观与运动特征，提升检测精度和身份一致性。**

- **链接: [https://arxiv.org/pdf/2603.03603](https://arxiv.org/pdf/2603.03603)**

> **作者:** Kasumi Seko; Hiroki Kinoshita; Raj Rajeshwar Malinda; Hiroaki Kawashima
>
> **备注:** Author's version of the paper presented at AROB-ISBC 2026
>
> **摘要:** In animal facilities, continuous surveillance of penguins is essential yet technically challenging due to their homogeneous visual characteristics, rapid and frequent posture changes, and substantial environmental noise such as water reflections. In this study, we propose a framework that enhances both detection and identification performance by integrating appearance and motion features. For detection, we adapted YOLO11 to process consecutive frames to overcome the lack of temporal consistency in single-frame detectors. This approach leverages motion cues to detect targets even when distinct visual features are obscured. Our evaluation shows that fine-tuning the model with two-frame inputs improves mAP@0.5 from 0.922 to 0.933, outperforming the baseline, and successfully recovers individuals that are indistinguishable in static images. For identification, we introduce a tracklet-based contrastive learning approach applied after tracking. Through qualitative visualization, we demonstrate that the method produces coherent feature embeddings, bringing samples from the same individual closer in the feature space, suggesting the potential for mitigating ID switching.
>
---
#### [new 073] DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出DeepScan，一个无需训练的框架，用于大视觉语言模型的视觉定位与推理。旨在解决复杂环境中准确理解视觉内容的问题，通过分层扫描、重聚焦和证据增强推理提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.03857](https://arxiv.org/pdf/2603.03857)**

> **作者:** Yangfu Li; Hongjian Zhan; Jiawei Chen; Yuning Gong; Qi Liu; Yue Lu
>
> **备注:** 18 pages 17 figures
>
> **摘要:** Humans can robustly localize visual evidence and provide grounded answers even in noisy environments by identifying critical cues and then relating them to the full context in a bottom-up manner. Inspired by this, we propose DeepScan, a training-free framework that combines Hierarchical Scanning, Refocusing, and Evidence-Enhanced Reasoning for visually grounded reasoning in Large Vision-Language Models (LVLMs). Unlike existing methods that pursue one-shot localization of complete evidence, Hierarchical Scanning performs local cue exploration and multi-scale evidence extraction to recover evidence in a bottom-up manner, effectively mitigating the impacts of distractive context. Refocusing then optimizes the localized evidence view through collaboration of LVLMs and visual experts. Finally, Evidence-Enhanced Reasoning aggregates multi-granular views via a hybrid evidence memory and yields accurate and interpretable answers. Experimental results demonstrate that DeepScan significantly boosts LVLMs in diverse visual tasks, especially in fine-grained visual understanding. It achieves 90.6% overall accuracy on V* when integrated with Qwen2.5-VL-7B. Moreover, DeepScan provides consistent improvements for LVLMs across various architectures and model scales without additional adaptation cost.
>
---
#### [new 074] Field imaging framework for morphological characterization of aggregates with computer vision: Algorithms and applications
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于计算机视觉任务，旨在解决建筑骨料形态表征的难题。通过开发场域成像框架，实现对骨料的2D/3D分析与形状预测。**

- **链接: [https://arxiv.org/pdf/2603.03654](https://arxiv.org/pdf/2603.03654)**

> **作者:** Haohang Huang
>
> **备注:** PhD thesis
>
> **摘要:** Construction aggregates, including sand and gravel, crushed stone and riprap, are the core building blocks of the construction industry. State-of-the-practice characterization methods mainly relies on visual inspection and manual measurement. State-of-the-art aggregate imaging methods have limitations that are only applicable to regular-sized aggregates under well-controlled conditions. This dissertation addresses these major challenges by developing a field imaging framework for the morphological characterization of aggregates as a multi-scenario solution. For individual and non-overlapping aggregates, a field imaging system was designed and the associated segmentation and volume estimation algorithms were developed. For 2D image analyses of aggregates in stockpiles, an automated 2D instance segmentation and morphological analysis approach was established. For 3D point cloud analyses of aggregate stockpiles, an integrated 3D Reconstruction-Segmentation-Completion (RSC-3D) approach was established: 3D reconstruction procedures from multi-view images, 3D stockpile instance segmentation, and 3D shape completion to predict the unseen sides. First, a 3D reconstruction procedure was developed to obtain high-fidelity 3D models of collected aggregate samples, based on which a 3D aggregate particle library was constructed. Next, two datasets were derived from the 3D particle library for 3D learning: a synthetic dataset of aggregate stockpiles with ground-truth instance labels, and a dataset of partial-complete shape pairs, developed with varying-view raycasting schemes. A state-of-the-art 3D instance segmentation network and a 3D shape completion network were trained on the datasets, respectively. The application of the integrated approach was demonstrated on real stockpiles and validated with ground-truth, showing good performance in capturing and predicting the unseen sides of aggregates.
>
---
#### [new 075] Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于扩散模型任务，解决采样过程中因刚性区域导致的误差问题。提出ERK-Guid方法，利用求解器误差作为引导信号，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.03692](https://arxiv.org/pdf/2603.03692)**

> **作者:** Inho Kong; Sojin Lee; Youngjoon Hong; Hyunwoo J. Kim
>
> **备注:** ICLR 2026
>
> **摘要:** Classifier-Free Guidance (CFG) has established the foundation for guidance mechanisms in diffusion models, showing that well-designed guidance proxies significantly improve conditional generation and sample quality. Autoguidance (AG) has extended this idea, but it relies on an auxiliary network and leaves solver-induced errors unaddressed. In stiff regions, the ODE trajectory changes sharply, where local truncation error (LTE) becomes a critical factor that deteriorates sample quality. Our key observation is that these errors align with the dominant eigenvector, motivating us to leverage the solver-induced error as a guidance signal. We propose Embedded Runge-Kutta Guidance (ERK-Guid), which exploits detected stiffness to reduce LTE and stabilize sampling. We theoretically and empirically analyze stiffness and eigenvector estimators with solver errors to motivate the design of ERK-Guid. Our experiments on both synthetic datasets and the popular benchmark dataset, ImageNet, demonstrate that ERK-Guid consistently outperforms state-of-the-art methods. Code is available at this https URL.
>
---
#### [new 076] TextBoost: Boosting Scene Text Fidelity in Ultra-low Bitrate Image Compression
- **分类: cs.CV**

- **简介: 该论文提出TextBoost方法，解决超低比特率图像压缩中保持小字体文本清晰与整体画质的矛盾。通过引入OCR辅助信息提升文本保真度。**

- **链接: [https://arxiv.org/pdf/2603.04115](https://arxiv.org/pdf/2603.04115)**

> **作者:** Bingxin Wang; Yuan Lan; Zhaoyi Sun; Yang Xiang; Jie Sun
>
> **摘要:** Ultra-low bitrate image compression faces a critical challenge: preserving small-font scene text while maintaining overall visual quality. Region-of-interest (ROI) bit allocation can prioritize text but often degrades global fidelity, leading to a trade-off between local accuracy and overall image quality. Instead of relying on ROI coding, we incorporate auxiliary textual information extracted by OCR and transmitted with negligible overhead, enabling the decoder to leverage this semantic guidance. Our method, TextBoost, operationalizes this idea through three strategic designs: (i) adaptively filtering OCR outputs and rendering them into a guidance map; (ii) integrating this guidance with decoder features in a calibrated manner via an attention-guided fusion block; and (iii) enforcing guidance-consistent reconstruction in text regions with a regularizing loss that promotes natural blending with the scene. Extensive experiments on TextOCR and ICDAR 2015 demonstrate that TextBoost yields up to 60.6% higher text-recognition F1 at comparable Peak Signal-to-Noise Ratio (PSNR) and bits per pixel (bpp), producing sharper small-font text while preserving global image quality and effectively decoupling text enhancement from global rate-distortion optimization.
>
---
#### [new 077] Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements
- **分类: cs.CV**

- **简介: 该论文属于单目6D位姿估计任务，旨在提高实时性与准确性。通过引入关键点增强的单阶段框架，提升3D几何理解，实现高效精确的位姿预测。**

- **链接: [https://arxiv.org/pdf/2603.03879](https://arxiv.org/pdf/2603.03879)**

> **作者:** Kemal Alperen Çetiner; Hazım Kemal Ekenel
>
> **备注:** Accepted to VISAPP 2026
>
> **摘要:** Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state-of-the-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo-Key-6D, a novel single stage, end-to-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end-to-end training, we directly regress rotation using a continuous 9D representation projected to SO(3) via singular value decomposition. On the LINEMOD and LINEMOD-Occluded benchmarks, YOLO-Key-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD(-S) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.
>
---
#### [new 078] One-Step Face Restoration via Shortcut-Enhanced Coupling Flow
- **分类: cs.CV**

- **简介: 该论文属于人脸修复任务，旨在解决现有方法依赖多步采样和忽略低质与高质数据关联的问题。提出SCFlowFR模型，通过数据相关耦合和快捷约束实现高效单步修复。**

- **链接: [https://arxiv.org/pdf/2603.03648](https://arxiv.org/pdf/2603.03648)**

> **作者:** Xiaohui Sun; Hanlin Wu
>
> **摘要:** Face restoration has advanced significantly with generative models like diffusion models and flow matching (FM), which learn continuous-time mappings between distributions. However, existing FM-based approaches often start from Gaussian noise, ignoring the inherent dependency between low-quality (LQ) and high-quality (HQ) data, resulting in path crossovers, curved trajectories, and multi-step sampling requirements. To address these issues, we propose Shortcut-enhanced Coupling flow for Face Restoration (SCFlowFR). First, it establishes a \textit{data-dependent coupling} that explicitly models the LQ--HQ dependency, minimizing path crossovers and promoting near-linear transport. Second, we employ conditional mean estimation to obtain a coarse prediction that refines the source anchor to tighten coupling and conditions the velocity field to stabilize large-step updates. Third, a shortcut constraint supervises average velocities over arbitrary time intervals, enabling accurate one-step inference. Experiments demonstrate that SCFlowFR achieves state-of-the-art one-step face restoration quality with inference speed comparable to traditional non-diffusion methods.
>
---
#### [new 079] Discriminative Perception via Anchored Description for Reasoning Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉推理分割任务，旨在解决模型推理过程偏离目标区域的问题。提出DPAD方法，通过描述性标题增强区分能力，提升分割效果并缩短推理链。**

- **链接: [https://arxiv.org/pdf/2603.04002](https://arxiv.org/pdf/2603.04002)**

> **作者:** Tao Yang; Qing Zhou; Yanliang Li; Qi Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Reasoning segmentation increasingly employs reinforcement learning to generate explanatory reasoning chains that guide Multimodal Large Language Models. While these geometric rewards are primarily confined to guiding the final localization, they are incapable of discriminating whether the reasoning process remains anchored on the referred region or strays into irrelevant context. Lacking this discriminative guidance, the model's reasoning often devolves into unfocused and verbose chains that ultimately fail to disambiguate and perceive the target in complex scenes. This suggests a need to complement the RL objective with Discriminative Perception, an ability to actively distinguish a target from its context. To realize this, we propose DPAD to compel the model to generate a descriptive caption of the referred object, which is then used to explicitly discriminate by contrasting the caption's semantic relevance to the referred object against the wider context. By optimizing for this discriminative capability, the model is forced to focus on the unique attributes of the target, leading to a more converged and efficient reasoning chain. The descriptive caption also serves as an interpretability rationale that aligns with the segmentation. Experiments on the benchmarks confirm the validity of our approach, delivering substantial performance gains, with the cIoU on ReasonSeg increasing by 3.09% and the reasoning chain length decreasing by approximately 42%. Code is available at this https URL
>
---
#### [new 080] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决冷启动阶段注意力分配不足的问题。通过引入VAS指标，提出AVAR框架提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.03825](https://arxiv.org/pdf/2603.03825)**

> **作者:** Ruilin Luo; Chufan Shi; Yizhen Zhang; Cheng Yang; Songtao Jiang; Tongkun Guan; Ruizhe Chen; Ruihang Chu; Peng Wang; Mingkun Yang; Yujiu Yang; Junyang Lin; Zhibo Yang
>
> **备注:** ICLR 2026 Poster
>
> **摘要:** The cold-start initialization stage plays a pivotal role in training Multimodal Large Reasoning Models (MLRMs), yet its mechanisms remain insufficiently understood. To analyze this stage, we introduce the Visual Attention Score (VAS), an attention-based metric that quantifies how much a model attends to visual tokens. We find that reasoning performance is strongly correlated with VAS (r=0.9616): models with higher VAS achieve substantially stronger multimodal reasoning. Surprisingly, multimodal cold-start fails to elevate VAS, resulting in attention distributions close to the base model, whereas text-only cold-start leads to a clear increase. We term this counter-intuitive phenomenon Lazy Attention Localization. To validate its causal role, we design training-free interventions that directly modulate attention allocation during inference, performance gains of 1$-$2% without any retraining. Building on these insights, we further propose Attention-Guided Visual Anchoring and Reflection (AVAR), a comprehensive cold-start framework that integrates visual-anchored data synthesis, attention-guided objectives, and visual-anchored reward shaping. Applied to Qwen2.5-VL-7B, AVAR achieves an average gain of 7.0% across 7 multimodal reasoning benchmarks. Ablation studies further confirm that each component of AVAR contributes step-wise to the overall gains. The code, data, and models are available at this https URL.
>
---
#### [new 081] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出NOVA3R，解决非像素对齐的3D重建任务，通过全局场景表示和扩散解码器，提升重建精度与完整性。**

- **链接: [https://arxiv.org/pdf/2603.04179](https://arxiv.org/pdf/2603.04179)**

> **作者:** Weirong Chen; Chuanxia Zheng; Ganlin Zhang; Andrea Vedaldi; Daniel Cremers
>
> **备注:** Accepted to ICLR 2026. Project Page: this https URL
>
> **摘要:** We present NOVA3R, an effective approach for non-pixel-aligned 3D reconstruction from a set of unposed images in a feed-forward manner. Unlike pixel-aligned methods that tie geometry to per-ray predictions, our formulation learns a global, view-agnostic scene representation that decouples reconstruction from pixel alignment. This addresses two key limitations in pixel-aligned 3D: (1) it recovers both visible and invisible points with a complete scene representation, and (2) it produces physically plausible geometry with fewer duplicated structures in overlapping regions. To achieve this, we introduce a scene-token mechanism that aggregates information across unposed images and a diffusion-based 3D decoder that reconstructs complete, non-pixel-aligned point clouds. Extensive experiments on both scene-level and object-level datasets demonstrate that NOVA3R outperforms state-of-the-art methods in terms of reconstruction accuracy and completeness.
>
---
#### [new 082] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
- **分类: cs.CV**

- **简介: 该论文属于图像去雨任务，旨在解决现有方法对不同雨况泛化能力差的问题。提出UniRain框架，结合RAG数据蒸馏和多目标优化，提升模型在多种雨景下的性能。**

- **链接: [https://arxiv.org/pdf/2603.03967](https://arxiv.org/pdf/2603.03967)**

> **作者:** Qianfeng Yang; Qiyuan Guan; Xiang Chen; Jiyu Jin; Guiyue Jin; Jiangxin Dong
>
> **备注:** Accepted by CVPR 2026; Project Page: this https URL
>
> **摘要:** Despite significant progress has been made in image deraining, we note that most existing methods are often developed for only specific types of rain degradation and fail to generalize across diverse real-world rainy scenes. How to effectively model different rain degradations within a universal framework is important for real-world image deraining. In this paper, we propose UniRain, an effective unified image deraining framework capable of restoring images degraded by rain streak and raindrop under both daytime and nighttime conditions. To better enhance unified model generalization, we construct an intelligent retrieval augmented generation (RAG)-based dataset distillation pipeline that selects high-quality training samples from all public deraining datasets for better mixed training. Furthermore, we incorporate a simple yet effective multi-objective reweighted optimization strategy into the asymmetric mixture-of-experts (MoE) architecture to facilitate consistent performance and improve robustness across diverse scenes. Extensive experiments show that our framework performs favorably against the state-of-the-art models on our proposed benchmarks and multiple public datasets.
>
---
#### [new 083] DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization
- **分类: cs.CV**

- **简介: 该论文属于3D牙齿生成任务，解决缺失牙齿布局与形状生成中的碰撞问题。提出DM-CFO模型，结合扩散模型与3D高斯优化，提升生成牙齿的合理性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.03602](https://arxiv.org/pdf/2603.03602)**

> **作者:** Yan Tian; Pengcheng Xue; Weiping Ding; Mahmoud Hassaballah; Karen Egiazarian; Aura Conci; Abdulkadir Sengur; Leszek Rutkowski
>
> **备注:** Received by IEEE Transactions on Visualization and Computer Graphics
>
> **摘要:** The automatic design of a 3D tooth model plays a crucial role in dental digitization. However, current approaches face challenges in compositional 3D tooth generation because both the layouts and shapes of missing teeth need to be this http URL addition, collision conflicts are often omitted in 3D Gaussian-based compositional 3D generation, where objects may intersect with each other due to the absence of explicit geometric information on the object surfaces. Motivated by graph generation through diffusion models and collision detection using 3D Gaussians, we propose an approach named DM-CFO for compositional tooth generation, where the layout of missing teeth is progressively restored during the denoising phase under both text and graph constraints. Then, the Gaussian parameters of each layout-guided tooth and the entire jaw are alternately updated using score distillation sampling (SDS). Furthermore, a regularization term based on the distances between the 3D Gaussians of neighboring teeth and the anchor tooth is introduced to penalize tooth intersections. Experimental results on three tooth-design datasets demonstrate that our approach significantly improves the multiview consistency and realism of the generated teeth compared with existing methods. Project page: this https URL.
>
---
#### [new 084] PinCLIP: Large-scale Foundational Multimodal Representation at Pinterest
- **分类: cs.CV**

- **简介: 该论文提出PinCLIP，解决推荐系统中多模态表示学习问题，通过融合视觉与文本信息提升检索和排序效果。**

- **链接: [https://arxiv.org/pdf/2603.03544](https://arxiv.org/pdf/2603.03544)**

> **作者:** Josh Beal; Eric Kim; Jinfeng Rao; Rex Wu; Dmitry Kislyuk; Charles Rosenberg
>
> **摘要:** While multi-modal Visual Language Models (VLMs) have demonstrated significant success across various domains, the integration of VLMs into recommendation and retrieval systems remains a challenge, due to issues like training objective discrepancies and serving efficiency bottlenecks. This paper introduces PinCLIP, a large-scale visual representation learning approach developed to enhance retrieval and ranking models at Pinterest by leveraging VLMs to learn image-text alignment. We propose a novel hybrid Vision Transformer architecture that utilizes a VLM backbone and a hybrid fusion mechanism to capture multi-modality content representation at varying granularities. Beyond standard image-to-text alignment objectives, we introduce a neighbor alignment objective to model the cross-fusion of multi-modal representations within the Pinterest Pin-Board graph. Offline evaluations show that PinCLIP outperforms state-of-the-art baselines, such as Qwen, by 20% in multi-modal retrieval tasks. Online A/B testing demonstrates significant business impact, including substantial engagement gains across all major surfaces in Pinterest. Notably, PinCLIP significantly addresses the "cold-start" problem, enhancing fresh content distribution with a 15% Repin increase in organic content and 8.7% higher click for new Ads.
>
---
#### [new 085] Spatial Causal Prediction in Video
- **分类: cs.CV**

- **简介: 该论文提出空间因果预测任务，解决模型在视频中推理未观测空间状态的能力不足问题。构建了SCP-Bench基准，评估并提升模型的空间因果推理能力。**

- **链接: [https://arxiv.org/pdf/2603.03944](https://arxiv.org/pdf/2603.03944)**

> **作者:** Yanguang Zhao; Jie Yang; Shengqiong Wu; Shutong Hu; Hongbo Qiu; Yu Wang; Guijia Zhang; Tan Kai Ze; Hao Fei; Chia-Wen Lin; Mong-Li Lee; Wynne Hsu
>
> **备注:** 30 pages, 21 figures, 17 tables, CVPR findings
>
> **摘要:** Spatial reasoning, the ability to understand spatial relations, causality, and dynamic evolution, is central to human intelligence and essential for real-world applications such as autonomous driving and robotics. Existing studies, however, primarily assess models on visible spatio-temporal understanding, overlooking their ability to infer unseen past or future spatial states. In this work, we introduce Spatial Causal Prediction (SCP), a new task paradigm that challenges models to reason beyond observation and predict spatial causal outcomes. We further construct SCP-Bench, a benchmark comprising 2,500 QA pairs across 1,181 videos spanning diverse viewpoints, scenes, and causal directions, to support systematic evaluation. Through comprehensive experiments on {23} state-of-the-art models, we reveal substantial gaps between human and model performance, limited temporal extrapolation, and weak causal grounding. We further analyze key factors influencing performance and propose perception-enhancement and reasoning-guided strategies toward advancing spatial causal intelligence. The project page is this https URL.
>
---
#### [new 086] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉合成任务，旨在解决DiT模型中表示学习不足的问题。通过增强不同层间的表示多样性，提出DiverseDiT框架，提升模型性能与收敛速度。**

- **链接: [https://arxiv.org/pdf/2603.04239](https://arxiv.org/pdf/2603.04239)**

> **作者:** Mengping Yang; Zhiyu Tan; Binglei Li; Xiaomeng Yang; Hesen Chen; Hao Li
>
> **备注:** To appear in CVPR 2026, GitHub Code: this https URL, Project Page: this https URL
>
> **摘要:** Recent breakthroughs in Diffusion Transformers (DiTs) have revolutionized the field of visual synthesis due to their superior scalability. To facilitate DiTs' capability of capturing meaningful internal representations, recent works such as REPA incorporate external pretrained encoders for representation alignment. However, the underlying mechanisms governing representation learning within DiTs are not well understood. To this end, we first systematically investigate the representation dynamics of DiTs. Through analyzing the evolution and influence of internal representations under various settings, we reveal that representation diversity across blocks is a crucial factor for effective learning. Based on this key insight, we propose DiverseDiT, a novel framework that explicitly promotes representation diversity. DiverseDiT incorporates long residual connections to diversify input representations across blocks and a representation diversity loss to encourage blocks to learn distinct features. Extensive experiments on ImageNet 256x256 and 512x512 demonstrate that our DiverseDiT yields consistent performance gains and convergence acceleration when applied to different backbones with various sizes, even when tested on the challenging one-step generation setting. Furthermore, we show that DiverseDiT is complementary to existing representation learning techniques, leading to further performance gains. Our work provides valuable insights into the representation learning dynamics of DiTs and offers a practical approach for enhancing their performance.
>
---
#### [new 087] Beyond Pixel Histories: World Models with Persistent 3D State
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉生成任务，旨在解决现有模型缺乏3D一致性与空间记忆的问题。提出PERSIST框架，实现持久化3D场景模拟与控制。**

- **链接: [https://arxiv.org/pdf/2603.03482](https://arxiv.org/pdf/2603.03482)**

> **作者:** Samuel Garcin; Thomas Walker; Steven McDonagh; Tim Pearce; Hakan Bilen; Tianyu He; Kaixin Wang; Jiang Bian
>
> **备注:** Currently under review
>
> **摘要:** Interactive world models continually generate video by responding to a user's actions, enabling open-ended generation capabilities. However, existing models typically lack a 3D representation of the environment, meaning 3D consistency must be implicitly learned from data, and spatial memory is restricted to limited temporal context windows. This results in an unrealistic user experience and presents significant obstacles to down-stream tasks such as training agents. To address this, we present PERSIST, a new paradigm of world model which simulates the evolution of a latent 3D scene: environment, camera, and renderer. This allows us to synthesize new frames with persistent spatial memory and consistent geometry. Both quantitative metrics and a qualitative user study show substantial improvements in spatial memory, 3D consistency, and long-horizon stability over existing methods, enabling coherent, evolving 3D worlds. We further demonstrate novel capabilities, including synthesising diverse 3D environments from a single image, as well as enabling fine-grained, geometry-aware control over generated experiences by supporting environment editing and specification directly in 3D space. Project page: this https URL
>
---
#### [new 088] ArtHOI: Articulated Human-Object Interaction Synthesis by 4D Reconstruction from Video Priors
- **分类: cs.CV**

- **简介: 该论文属于人-物交互生成任务，解决无3D/4D监督下生成物理合理交互的问题。通过4D重建从视频先验中合成关节式人-物交互，提升接触准确性和运动合理性。**

- **链接: [https://arxiv.org/pdf/2603.04338](https://arxiv.org/pdf/2603.04338)**

> **作者:** Zihao Huang; Tianqi Liu; Zhaoxi Chen; Shaocong Xu; Saining Zhang; Lixing Xiao; Zhiguo Cao; Wei Li; Hao Zhao; Ziwei Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Synthesizing physically plausible articulated human-object interactions (HOI) without 3D/4D supervision remains a fundamental challenge. While recent zero-shot approaches leverage video diffusion models to synthesize human-object interactions, they are largely confined to rigid-object manipulation and lack explicit 4D geometric reasoning. To bridge this gap, we formulate articulated HOI synthesis as a 4D reconstruction problem from monocular video priors: given only a video generated by a diffusion model, we reconstruct a full 4D articulated scene without any 3D supervision. This reconstruction-based approach treats the generated 2D video as supervision for an inverse rendering problem, recovering geometrically consistent and physically plausible 4D scenes that naturally respect contact, articulation, and temporal coherence. We introduce ArtHOI, the first zero-shot framework for articulated human-object interaction synthesis via 4D reconstruction from video priors. Our key designs are: 1) Flow-based part segmentation: leveraging optical flow as a geometric cue to disentangle dynamic from static regions in monocular video; 2) Decoupled reconstruction pipeline: joint optimization of human motion and object articulation is unstable under monocular ambiguity, so we first recover object articulation, then synthesize human motion conditioned on the reconstructed object states. ArtHOI bridges video-based generation and geometry-aware reconstruction, producing interactions that are both semantically aligned and physically grounded. Across diverse articulated scenes (e.g., opening fridges, cabinets, microwaves), ArtHOI significantly outperforms prior methods in contact accuracy, penetration reduction, and articulation fidelity, extending zero-shot interaction synthesis beyond rigid manipulation through reconstruction-informed synthesis.
>
---
#### [new 089] Dual Diffusion Models for Multi-modal Guided 3D Avatar Generation
- **分类: cs.CV**

- **简介: 该论文属于3D Avatar生成任务，旨在解决文本和图像驱动生成中的语义控制不足与效率低的问题。通过构建多模态数据集并提出双扩散模型框架，实现高效高质量的3D avatar生成。**

- **链接: [https://arxiv.org/pdf/2603.04307](https://arxiv.org/pdf/2603.04307)**

> **作者:** Hong Li; Yutang Feng; Minqi Meng; Yichen Yang; Xuhui Liu; Baochang Zhang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Generating high-fidelity 3D avatars from text or image prompts is highly sought after in virtual reality and human-computer interaction. However, existing text-driven methods often rely on iterative Score Distillation Sampling (SDS) or CLIP optimization, which struggle with fine-grained semantic control and suffer from excessively slow inference. Meanwhile, image-driven approaches are severely bottlenecked by the scarcity and high acquisition cost of high-quality 3D facial scans, limiting model generalization. To address these challenges, we first construct a novel, large-scale dataset comprising over 100,000 pairs across four modalities: fine-grained textual descriptions, in-the-wild face images, high-quality light-normalized texture UV maps, and 3D geometric shapes. Leveraging this comprehensive dataset, we propose PromptAvatar, a framework featuring dual diffusion models. Specifically, it integrates a Texture Diffusion Model (TDM) that supports flexible multi-condition guidance from text and/or image prompts, alongside a Geometry Diffusion Model (GDM) guided by text prompts. By learning the direct mapping from multi-modal prompts to 3D representations, PromptAvatar eliminates the need for time-consuming iterative optimization, successfully generating high-fidelity, shading-free 3D avatars in under 10 seconds. Extensive quantitative and qualitative experiments demonstrate that our method significantly outperforms existing state-of-the-art approaches in generation quality, fine-grained detail alignment, and computational efficiency.
>
---
#### [new 090] Underrepresented in Foundation Model Pretraining Data? A One-Shot Probe
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决缺乏标注数据时无法准确评估模型在特定领域表现的问题。通过生成反事实描述并测量模型区分能力，预测模型零样本准确率。**

- **链接: [https://arxiv.org/pdf/2603.04346](https://arxiv.org/pdf/2603.04346)**

> **作者:** Chris Vorster; Mayug Maniparambil; Noel E. O'Connor; Noel Murphy; Derek Molloy
>
> **摘要:** Large-scale Vision-Language Foundation Models (VLFMs), such as CLIP, now underpin a wide range of computer vision research and applications. VLFMs are often adapted to various domain-specific tasks. However, VLFM performance on novel, specialised, or underrepresented domains remains inconsistent. Evaluating VLFMs typically requires labelled test sets, which are often unavailable for niche domains of interest, particularly those from the Global South. We address this gap by proposing a highly data-efficient method to predict a VLFM's zero-shot accuracy on a target domain using only a single labelled image per class. Our approach uses a Large Language Model to generate plausible counterfactual descriptions of a given image. By measuring the VLFM's ability to distinguish the correct description from these hard negatives, we engineer features that capture the VLFM's discriminative power in its shared embedding space. A linear regressor trained on these similarity scores estimates the VLFM's zero-shot test accuracy across various visual domains with a Pearson-r correlation of 0.96. We demonstrate our method's performance across five diverse datasets, including standard benchmark datasets and underrepresented datasets from Africa. Our work provides a low-cost, reliable tool for probing VLFMs, enabling researchers and practitioners to make informed decisions about data annotation efforts before committing significant resources. The model training code, generated captions and counterfactuals are released here: this https URL.
>
---
#### [new 091] Cross-Modal Mapping and Dual-Branch Reconstruction for 2D-3D Multimodal Industrial Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业异常检测任务，解决2D+3D多模态数据下的异常定位问题。提出CMDR-IAD框架，通过跨模态映射和双分支重建实现高效、鲁棒的异常检测。**

- **链接: [https://arxiv.org/pdf/2603.03939](https://arxiv.org/pdf/2603.03939)**

> **作者:** Radia Daci; Vito Renò; Cosimo Patruno; Angelo Cardellicchio; Abdelmalik Taleb-Ahmed; Marco Leo; Cosimo Distante
>
> **摘要:** Multimodal industrial anomaly detection benefits from integrating RGB appearance with 3D surface geometry, yet existing \emph{unsupervised} approaches commonly rely on memory banks, teacher-student architectures, or fragile fusion schemes, limiting robustness under noisy depth, weak texture, or missing modalities. This paper introduces \textbf{CMDR-IAD}, a lightweight and modality-flexible unsupervised framework for reliable anomaly detection in 2D+3D multimodal as well as single-modality (2D-only or 3D-only) settings. \textbf{CMDR-IAD} combines bidirectional 2D$\leftrightarrow$3D cross-modal mapping to model appearance-geometry consistency with dual-branch reconstruction that independently captures normal texture and geometric structure. A two-part fusion strategy integrates these cues: a reliability-gated mapping anomaly highlights spatially consistent texture-geometry discrepancies, while a confidence-weighted reconstruction anomaly adaptively balances appearance and geometric deviations, yielding stable and precise anomaly localization even in depth-sparse or low-texture regions. On the MVTec 3D-AD benchmark, CMDR-IAD achieves state-of-the-art performance while operating without memory banks, reaching 97.3\% image-level AUROC (I-AUROC), 99.6\% pixel-level AUROC (P-AUROC), and 97.6\% AUPRO. On a real-world polyurethane cutting dataset, the 3D-only variant attains 92.6\% I-AUROC and 92.5\% P-AUROC, demonstrating strong effectiveness under practical industrial conditions. These results highlight the framework's robustness, modality flexibility, and the effectiveness of the proposed fusion strategies for industrial visual inspection. Our source code is available at this https URL
>
---
#### [new 092] TaxonRL: Reinforcement Learning with Intermediate Rewards for Interpretable Fine-Grained Visual Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出TaxonRL，用于解决细粒度视觉分类中的可解释性问题。通过强化学习和中间奖励，分解分类过程为层级预测，提升准确性和透明度。**

- **链接: [https://arxiv.org/pdf/2603.04380](https://arxiv.org/pdf/2603.04380)**

> **作者:** Maximilian von Klinski; Maximilian Schall
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Traditional vision-language models struggle with contrastive fine-grained taxonomic reasoning, particularly when distinguishing between visually similar species within the same genus or family. We introduce TaxonRL, a reinforcement learning approach using Group Relative Policy Optimization with intermediate rewards that decomposes the reasoning process into hierarchical taxonomic predictions. Our method incentivizes models to explicitly reason about species-level, genus-level, and family-level features before making final classifications. This structured approach is designed not only to boost accuracy but also to yield a transparent, verifiable decision-making process. On the challenging Birds-to-Words dataset, TaxonRL achieves 91.7\% average accuracy, exceeding human performance (77.3\%) while generating interpretable reasoning traces. We demonstrate strong cross-domain generalization, showing substantial gains in primate and marine species verification. Our results establish that enforcing structured, hierarchical reasoning provides a powerful and transferable framework for fine-grained visual discrimination.
>
---
#### [new 093] Crab$^{+}$: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于多模态场景理解任务，旨在解决音频-视觉任务异质性导致的负迁移问题。通过构建统一数据集和模型结构，提升多任务学习效果。**

- **链接: [https://arxiv.org/pdf/2603.04128](https://arxiv.org/pdf/2603.04128)**

> **作者:** Dongnuan Cai; Henghui Du; Chang Zhou; Xi Chen; Dan Guo; Hongyuan Zhang; Xuelong Li; Di Hu
>
> **摘要:** Developing Audio-Visual Large Language Models (AV-LLMs) for unified scene understanding is pivotal in multimodal intelligence. While instruction tuning enables pre-trained models with multi-task abilities, we observe that conventional multi-task unification methods often suffer from severe negative transfer, where nearly 55% of tasks degrade compared to single-task training. We attribute this phenomenon to audio-visual task heterogeneity, characterized by disparate task granularity and divergent capability demands, which lead to negative interference under joint training. To tackle this, we present Crab$^{+}$, a scalable and unified audio-visual scene understanding model that addresses task heterogeneity through explicit cooperation from both data and model perspectives. On the data side, we introduce AV-UIE v2, a comprehensive Audio-Visual Unified Instruction-tuning dataset with Explicit reasoning processes. It contains approximately 222K samples spanning 17 datasets and 7 tasks, enabling the model to capture cross-task relationships at different levels of granularity. On the model side, we design a unified interface to align heterogeneous task formulations, and propose Interaction-aware LoRA (I-LoRA), which explicitly models inter-task relationships via dynamic routing to coordinate distinct audio-visual interaction patterns, mitigating parameter interference. Extensive experiments show Crab$^{+}$ covers broader tasks than existing unified models while outperforming specialized models on various benchmarks. We successfully reverse the negative transfer trend, achieving positive transfer where multi-task learning surpasses single-task baselines in nearly 88% of tasks. These results hold across diverse AV-LLM paradigms and are validated through in-depth visualization, positioning Crab$^{+}$ as a robust step towards holistic audio-visual scene understanding.
>
---
#### [new 094] Seeing as Experts Do: A Knowledge-Augmented Agent for Open-Set Fine-Grained Visual Understanding
- **分类: cs.CV**

- **简介: 该论文属于细粒度视觉理解任务，解决开放集和上下文依赖下的识别与推理问题。提出KFRA框架，通过知识增强实现证据驱动的推理，提升模型的解释性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03762](https://arxiv.org/pdf/2603.03762)**

> **作者:** Junhan Chen; Zilu Zhou; Yujun Tong; Dongliang Chang; Yitao Luo; Zhanyu Ma
>
> **摘要:** Fine-grained visual understanding is shifting from static classification to knowledge-augmented reasoning, where models must justify as well as recognise. Existing approaches remain limited by closed-set taxonomies and single-label prediction, leading to significant degradation under open-set or context-dependent conditions. We present the Knowledge-Augmented Fine-Grained Reasoning Agent (KFRA), a unified framework that transforms fine-grained perception into evidence-driven reasoning. KFRA operates through a three-stage closed reasoning loop that emulates expert analysis. It first performs open-vocabulary detection and web-scale retrieval to generate category hypotheses. It then conducts discriminative regions localisation by aligning textual knowledge with visual evidence through a global-to-local focusing mechanism. Finally, it integrates all multimodal evidence within a large multimodal model to perform interpretable reasoning. Unlike existing agents that treat retrieval and reasoning as independent processes, KFRA establishes a retrieval-grounding coupling that converts retrieved knowledge into spatially grounded evidence for verification. This design enables factual, interpretable, and task-agnostic reasoning across diverse fine-grained scenarios. To evaluate this capability, we construct FGExpertBench, a benchmark designed to assess reasoning depth and cross-task generalisation across six knowledge dimensions. Extensive experiments demonstrate that KFRA consistently surpasses both standalone large multimodal models and current agent frameworks, achieving up to 19 percent improvement in reasoning accuracy and delivering evidence-grounded interpretability in open-set fine-grained visual understanding.
>
---
#### [new 095] Image-based Prompt Injection: Hijacking Multimodal LLMs through Visually Embedded Adversarial Instructions
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究图像引导的提示注入攻击，针对多模态大语言模型，通过嵌入对抗指令实现模型行为操控，提出有效攻击方法并验证其可行性。**

- **链接: [https://arxiv.org/pdf/2603.03637](https://arxiv.org/pdf/2603.03637)**

> **作者:** Neha Nagaraja; Lan Zhang; Zhilong Wang; Bo Zhang; Pawan Patil
>
> **备注:** 7 pages, published in 2025 3rd International Conference on Foundation and Large Language Models (FLLM), Vienna, Austria
>
> **摘要:** Multimodal Large Language Models (MLLMs) integrate vision and text to power applications, but this integration introduces new vulnerabilities. We study Image-based Prompt Injection (IPI), a black-box attack in which adversarial instructions are embedded into natural images to override model behavior. Our end-to-end IPI pipeline incorporates segmentation-based region selection, adaptive font scaling, and background-aware rendering to conceal prompts from human perception while preserving model interpretability. Using the COCO dataset and GPT-4-turbo, we evaluate 12 adversarial prompt strategies and multiple embedding configurations. The results show that IPI can reliably manipulate the output of the model, with the most effective configuration achieving up to 64\% attack success under stealth constraints. These findings highlight IPI as a practical threat in black-box settings and underscore the need for defenses against multimodal prompt injection.
>
---
#### [new 096] Slice-wise quality assessment of high b-value breast DWI via deep learning-based artifact detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像质量评估任务，旨在通过深度学习检测高b值乳腺DWI中的伪影。工作包括使用CNN模型进行二分类和多分类检测，并评估模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03941](https://arxiv.org/pdf/2603.03941)**

> **作者:** Ameya Markale; Luise Brock; Ihor Horishnyi; Dominika Skwierawska; Tri-Thien Nguyen; Hannes Schreiter; Shirin Heidarikahkesh; Lorenz A. Kapsner; Michael Uder; Sabine Ohlmeyer; Frederik B Laun; Andrzej Liebert; Sebastian Bickelhaupt
>
> **摘要:** Diffusion-weighted imaging (DWI) can support lesion detection and characterization in breast magnetic resonance imaging (MRI), however especially high b-value diffusion-weighted acquisitions can be prone to intensity artifacts that can affect diagnostic image assessment. This study aims to detect both hyper- and hypointense artifacts on high b-value diffusion-weighted images (b=1500 s/mm2) using deep learning, employing either a binary classification (artifact presence) or a multiclass classification (artifact intensity) approach on a slice-wise this http URL IRB-approved retrospective study used the single-center dataset comprising n=11806 slices from routine 3T breast MRI examinations performed between 2022 and mid-2023. Three convolutional neural network (CNN) architectures (DenseNet121, ResNet18, and SEResNet50) were trained for binary classification of hyper- and hypointense artifacts. The best performing model (DenseNet121) was applied to an independent holdout test set and was further trained separately for multiclass classification. Evaluation included area under receiver operating characteristic curve (AUROC), area under precision recall curve (AUPRC), precision, and recall, as well as analysis of predicted bounding box positions, derived from the network Grad-CAM heatmaps. DenseNet121 achieved AUROCs of 0.92 and 0.94 for hyper- and hypointense artifact detection, respectively, and weighted AUROCs of 0.85 and 0.88 for multiclass classification on single-slice high b-value diffusion-weighted images. A radiologist evaluated bounding box precision on a 1-5 Likert-like scale across 200 slices, achieving mean scores of 3.33+-1.04 for hyperintense artifacts and 2.62+-0.81 for hypointense artifacts. Hyper- and hypointense artifact detection in slice-wise breast DWI MRI dataset (b=1500 s/mm2) using CNNs particularly DenseNet121, seems promising and requires further validation.
>
---
#### [new 097] Gaussian Wardrobe: Compositional 3D Gaussian Avatars for Free-Form Virtual Try-On
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出Gaussian Wardrobe，解决3D虚拟试衣中服装与人体耦合的问题。通过分解人体与服装层，实现高保真动态 avatar 和自由服装迁移。**

- **链接: [https://arxiv.org/pdf/2603.04290](https://arxiv.org/pdf/2603.04290)**

> **作者:** Zhiyi Chen; Hsuan-I Ho; Tianjian Jiang; Jie Song; Manuel Kaufmann; Chen Guo
>
> **备注:** 3DV 2026, 16 pages, 12 figures
>
> **摘要:** We introduce Gaussian Wardrobe, a novel framework to digitalize compositional 3D neural avatars from multi-view videos. Existing methods for 3D neural avatars typically treat the human body and clothing as an inseparable entity. However, this paradigm fails to capture the dynamics of complex free-form garments and limits the reuse of clothing across different individuals. To overcome these problems, we develop a novel, compositional 3D Gaussian representation to build avatars from multiple layers of free-form garments. The core of our method is decomposing neural avatars into bodies and layers of shape-agnostic neural garments. To achieve this, our framework learns to disentangle each garment layer from multi-view videos and canonicalizes it into a shape-independent space. In experiments, our method models photorealistic avatars with high-fidelity dynamics, achieving new state-of-the-art performance on novel pose synthesis benchmarks. In addition, we demonstrate that the learned compositional garments contribute to a versatile digital wardrobe, enabling a practical virtual try-on application where clothing can be freely transferred to new subjects. Project page: this https URL
>
---
#### [new 098] InEdit-Bench: Benchmarking Intermediate Logical Pathways for Intelligent Image Editing Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出InEdit-Bench，用于评估图像编辑模型在中间逻辑路径上的推理能力。解决静态任务外的动态推理问题，通过构建基准测试和评估标准，推动更智能的生成模型发展。**

- **链接: [https://arxiv.org/pdf/2603.03657](https://arxiv.org/pdf/2603.03657)**

> **作者:** Zhiqiang Sheng; Xumeng Han; Zhiwei Zhang; Zenghui Xiong; Yifan Ding; Aoxiang Ping; Xiang Li; Tong Guo; Yao Mao
>
> **备注:** CVPR findings. Project page: this https URL
>
> **摘要:** Multimodal generative models have made significant strides in image editing, demonstrating impressive performance on a variety of static tasks. However, their proficiency typically does not extend to complex scenarios requiring dynamic reasoning, leaving them ill-equipped to model the coherent, intermediate logical pathways that constitute a multi-step evolution from an initial state to a final one. This capacity is crucial for unlocking a deeper level of procedural and causal understanding in visual manipulation. To systematically measure this critical limitation, we introduce InEdit-Bench, the first evaluation benchmark dedicated to reasoning over intermediate pathways in image editing. InEdit-Bench comprises meticulously annotated test cases covering four fundamental task categories: state transition, dynamic process, temporal sequence, and scientific simulation. Additionally, to enable fine-grained evaluation, we propose a set of assessment criteria to evaluate the logical coherence and visual naturalness of the generated pathways, as well as the model's fidelity to specified path constraints. Our comprehensive evaluation of 14 representative image editing models on InEdit-Bench reveals significant and widespread shortcomings in this domain. By providing a standardized and challenging benchmark, we aim for InEdit-Bench to catalyze research and steer development towards more dynamic, reason-aware, and intelligent multimodal generative models.
>
---
#### [new 099] When Visual Evidence is Ambiguous: Pareidolia as a Diagnostic Probe for Vision Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型诊断任务，研究在视觉证据模糊时模型如何解释人脸幻觉，分析不同模型的检测、不确定性与偏差，提出一种诊断框架以提升视觉语言系统的语义鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.03989](https://arxiv.org/pdf/2603.03989)**

> **作者:** Qianpu Chen; Derya Soydaner; Rob Saunders
>
> **摘要:** When visual evidence is ambiguous, vision models must decide whether to interpret face-like patterns as meaningful. Face pareidolia, the perception of faces in non-face objects, provides a controlled probe of this behavior. We introduce a representation-level diagnostic framework that analyzes detection, localization, uncertainty, and bias across class, difficulty, and emotion in face pareidolia images. Under a unified protocol, we evaluate six models spanning four representational regimes: vision-language models (VLMs; CLIP-B/32, CLIP-L/14, LLaVA-1.5-7B), pure vision classification (ViT), general object detection (YOLOv8), and face detection (RetinaFace). Our analysis reveals three mechanisms of interpretation under ambiguity. VLMs exhibit semantic overactivation, systematically pulling ambiguous non-human regions toward the Human concept, with LLaVA-1.5-7B producing the strongest and most confident over-calls, especially for negative emotions. ViT instead follows an uncertainty-as-abstention strategy, remaining diffuse yet largely unbiased. Detection-based models achieve low bias through conservative priors that suppress pareidolia responses even when localization is controlled. These results show that behavior under ambiguity is governed more by representational choices than score thresholds, and that uncertainty and bias are decoupled: low uncertainty can signal either safe suppression, as in detectors, or extreme over-interpretation, as in VLMs. Pareidolia therefore provides a compact diagnostic and a source of ambiguity-aware hard negatives for probing and improving the semantic robustness of vision-language systems. Code will be released upon publication.
>
---
#### [new 100] A Unified Framework for Joint Detection of Lacunes and Enlarged Perivascular Spaces
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决CSVD标志物（EPVS和lacunae）联合检测中的特征干扰与类别不平衡问题。提出一种形态解耦框架，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04243](https://arxiv.org/pdf/2603.04243)**

> **作者:** Lucas He; Krinos Li; Hanyuan Zhang; Runlong He; Silvia Ingala; Luigi Lorenzini; Marleen de Bruijne; Frederik Barkhof; Rhodri Davies; Carole Sudre
>
> **摘要:** Cerebral small vessel disease (CSVD) markers, specifically enlarged perivascular spaces (EPVS) and lacunae, present a unique challenge in medical image analysis due to their radiological mimicry. Standard segmentation networks struggle with feature interference and extreme class imbalance when handling these divergent targets simultaneously. To address these issues, we propose a morphology-decoupled framework where Zero-Initialized Gated Cross-Task Attention exploits dense EPVS context to guide sparse lacune detection. Furthermore, biological and topological consistency are enforced via a mixed-supervision strategy integrating Mutual Exclusion and Centerline Dice losses. Finally, we introduce an Anatomically-Informed Inference Calibration mechanism to dynamically suppress false positives based on tissue semantics. Extensive 5-folds cross-validation on the VALDO 2021 dataset (N=40) demonstrates state-of-the-art performance, notably surpassing task winners in lacunae detection precision (71.1%, p=0.01) and F1-score (62.6%, p=0.03). Furthermore, evaluation on the external EPAD cohort (N=1762) confirms the model's robustness for large-scale population studies. Code will be released upon acceptance.
>
---
#### [new 101] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出TAP框架，用于加速扩散模型生成过程。针对推理速度慢的问题，通过自适应选择预测器提升效率，无需额外训练，效果显著。**

- **链接: [https://arxiv.org/pdf/2603.03792](https://arxiv.org/pdf/2603.03792)**

> **作者:** Haowei Zhu; Tingxuan Huang; Xing Wang; Tianyu Zhao; Jiexi Wang; Weifeng Chen; Xurui Peng; Fangmin Chen; Junhai Yong; Bin Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Diffusion models achieve strong generative performance but remain slow at inference due to the need for repeated full-model denoising passes. We present Token-Adaptive Predictor (TAP), a training-free, probe-driven framework that adaptively selects a predictor for each token at every sampling step. TAP uses a single full evaluation of the model's first layer as a low-cost probe to compute proxy losses for a compact family of candidate predictors (instantiated primarily with Taylor expansions of varying order and horizon), then assigns each token the predictor with the smallest proxy error. This per-token "probe-then-select" strategy exploits heterogeneous temporal dynamics, requires no additional training, and is compatible with various predictor designs. TAP incurs negligible overhead while enabling large speedups with little or no perceptual quality loss. Extensive experiments across multiple diffusion architectures and generation tasks show that TAP substantially improves the accuracy-efficiency frontier compared to fixed global predictors and caching-only baselines.
>
---
#### [new 102] Balancing Fidelity, Utility, and Privacy in Synthetic Cardiac MRI Generation: A Comparative Study
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成任务，旨在解决数据稀缺与隐私保护问题。通过比较三种生成模型，在有限数据下平衡图像质量、实用性和隐私性。**

- **链接: [https://arxiv.org/pdf/2603.04340](https://arxiv.org/pdf/2603.04340)**

> **作者:** Madhura Edirisooriya; Dasuni Kawya; Ishan Kumarasinghe; Isuri Devindi; Mary M. Maleckar; Roshan Ragel; Isuru Nawinne; Vajira Thambawita
>
> **备注:** 7 pages, 4 figures, Preprint
>
> **摘要:** Deep learning in cardiac MRI (CMR) is fundamentally constrained by both data scarcity and privacy regulations. This study systematically benchmarks three generative architectures: Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion Models (LDM), and Flow Matching (FM) for synthetic CMR generation. Utilizing a two-stage pipeline where anatomical masks condition image synthesis, we evaluate generated data across three critical axes: fidelity, utility, and privacy. Our results show that diffusion-based models, particularly DDPM, provide the most effective balance between downstream segmentation utility, image fidelity, and privacy preservation under limited-data conditions, while FM demonstrates promising privacy characteristics with slightly lower task-level performance. These findings quantify the trade-offs between cross-domain generalization and patient confidentiality, establishing a framework for safe and effective synthetic data augmentation in medical imaging.
>
---
#### [new 103] Hazard-Aware Traffic Scene Graph Generation
- **分类: cs.CV**

- **简介: 该论文提出"交通场景图生成"任务，解决驾驶场景中安全相关关系识别问题。通过结合事故数据与深度信息，构建包含显著危险因素的场景图，提升对潜在风险的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.03584](https://arxiv.org/pdf/2603.03584)**

> **作者:** Yaoqi Huang; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **摘要:** Maintaining situational awareness in complex driving scenarios is challenging. It requires continuously prioritizing attention among extensive scene entities and understanding how prominent hazards might affect the ego vehicle. While existing studies excel at detecting specific semantic categories and visually salient regions, they lack the ability to assess safety-relevance. Meanwhile, the generic spatial predicates either for foreground objects only or for all scene entities modeled by existing scene graphs are inadequate for driving scenarios. To bridge this gap, we introduce a novel task, Traffic Scene Graph Generation, which captures traffic-specific relations between prominent hazards and the ego vehicle. We propose a novel framework that explicitly uses traffic accident data and depth cues to supplement visual features and semantic information for reasoning. The output traffic scene graphs provide intuitive guidelines that stress prominent hazards by color-coding their severity and notating their effect mechanism and relative location to the ego vehicle. We create relational annotations on Cityscapes dataset and evaluate our model on 10 tasks from 5 perspectives. The results in comparative experiments and ablation studies demonstrate our capacity in ego-centric reasoning for hazard-aware traffic scene understanding.
>
---
#### [new 104] Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications
- **分类: cs.CV**

- **简介: 该论文属于无人机视觉目标跟踪任务，解决复杂场景下跟踪鲁棒性和实时性问题。提出MATA架构，结合Transformer与卡尔曼滤波，并引入新的评估协议和指标。**

- **链接: [https://arxiv.org/pdf/2603.03904](https://arxiv.org/pdf/2603.03904)**

> **作者:** Augustin Borne; Pierre Notin; Christophe Hennequin; Sebastien Changey; Stephane Bazeille; Christophe Cudel; Franz Quint
>
> **摘要:** Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.
>
---
#### [new 105] PROSPECT: Unified Streaming Vision-Language Navigation via Semantic--Spatial Fusion and Latent Predictive Representation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PROSPECT，解决视觉语言导航任务中的环境动态与空间结构预测问题，通过语义-空间融合和潜在预测表示实现统一的流式导航。**

- **链接: [https://arxiv.org/pdf/2603.03739](https://arxiv.org/pdf/2603.03739)**

> **作者:** Zehua Fan; Wenqi Lyu; Wenxuan Song; Linge Zhao; Yifei Yang; Xi Wang; Junjie He; Lida Huang; Haiyan Liu; Bingchuan Sun; Guangjun Bao; Xuanyao Mao; Liang Xu; Yan Wang; Feng Gao
>
> **摘要:** Multimodal large language models (MLLMs) have advanced zero-shot end-to-end Vision-Language Navigation (VLN), yet robust navigation requires not only semantic understanding but also predictive modeling of environment dynamics and spatial structure. We propose PROSPECT, a unified streaming navigation agent that couples a streaming Vision-Language-Action (VLA) policy with latent predictive representation learning. PROSPECT uses CUT3R as a streaming 3D foundation spatial encoder to produce long-context, absolute-scale spatial features, and fuses them with SigLIP semantic features via cross-attention. During training, we introduce learnable stream query tokens that query the streaming context and predict next-step 2D and 3D latent features (rather than pixels or explicit modalities), supervised in the latent spaces of frozen SigLIP and CUT3R teachers. The predictive branch shapes internal representations without inference overhead. Experiments on VLN-CE benchmarks and real-robot deployment demonstrate state-of-the-art performance and improved long-horizon robustness under diverse lighting. We will release code for the community soon.
>
---
#### [new 106] Separators in Enhancing Autoregressive Pretraining for Vision Mamba
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型预训练任务，旨在解决Mamba在自回归预训练中受限于短序列的问题。通过引入STAR分离器，扩展输入序列长度并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.03806](https://arxiv.org/pdf/2603.03806)**

> **作者:** Hanpeng Liu; Zidan Wang; Shuoxi Zhang; Kaiyuan Gao; Kun He
>
> **摘要:** The state space model Mamba has recently emerged as a promising paradigm in computer vision, attracting significant attention due to its efficient processing of long sequence tasks. Mamba's inherent causal mechanism renders it particularly suitable for autoregressive pretraining. However, current autoregressive pretraining methods are constrained to short sequence tasks, failing to fully exploit Mamba's prowess in handling extended sequences. To address this limitation, we introduce an innovative autoregressive pretraining method for Vision Mamba that substantially extends the input sequence length. We introduce new \textbf{S}epara\textbf{T}ors for \textbf{A}uto\textbf{R}egressive pretraining to demarcate and differentiate between different images, known as \textbf{STAR}. Specifically, we insert identical separators before each image to demarcate its inception. This strategy enables us to quadruple the input sequence length of Vision Mamba while preserving the original dimensions of the dataset images. Employing this long sequence pretraining technique, our STAR-B model achieved an impressive accuracy of 83.5\% on ImageNet-1k, which is highly competitive in Vision Mamba. These results underscore the potential of our method in enhancing the performance of vision models through improved leveraging of long-range dependencies.
>
---
#### [new 107] Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于视觉数据压缩任务，解决可穿戴设备中冗余视频帧存储问题。通过眼动追踪数据，结合注视稳定性与瞳孔新颖性筛选关键帧，提升数据效率。**

- **链接: [https://arxiv.org/pdf/2603.04098](https://arxiv.org/pdf/2603.04098)**

> **作者:** Ajan Subramanian; Sumukh Bettadapura; Rohan Sathish
>
> **备注:** 14 pages, 4 figures, 3 tables, plus supplementary material
>
> **摘要:** Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.
>
---
#### [new 108] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，解决在线实时语义3D重建问题。提出EmbodiedSplat方法，实现从流图像中同时进行3D重建和语义理解，具有高泛化性和实时性。**

- **链接: [https://arxiv.org/pdf/2603.04254](https://arxiv.org/pdf/2603.04254)**

> **作者:** Seungjun Lee; Zihan Wang; Yunsong Wang; Gim Hee Lee
>
> **备注:** CVPR 2026, Project Page: this https URL
>
> **摘要:** Understanding a 3D scene immediately with its exploration is essential for embodied tasks, where an agent must construct and comprehend the 3D scene in an online and nearly real-time manner. In this study, we propose EmbodiedSplat, an online feed-forward 3DGS for open-vocabulary scene understanding that enables simultaneous online 3D reconstruction and 3D semantic understanding from the streaming images. Unlike existing open-vocabulary 3DGS methods which are typically restricted to either offline or per-scene optimization setting, our objectives are two-fold: 1) Reconstructs the semantic-embedded 3DGS of the entire scene from over 300 streaming images in an online manner. 2) Highly generalizable to novel scenes with feed-forward design and supports nearly real-time 3D semantic reconstruction when combined with real-time 2D models. To achieve these objectives, we propose an Online Sparse Coefficients Field with a CLIP Global Codebook where it binds the 2D CLIP embeddings to each 3D Gaussian while minimizing memory consumption and preserving the full semantic generalizability of CLIP. Furthermore, we generate 3D geometric-aware CLIP features by aggregating the partial point cloud of 3DGS through 3D U-Net to compensate the 3D geometric prior to 2D-oriented language embeddings. Extensive experiments on diverse indoor datasets, including ScanNet, ScanNet++, and Replica, demonstrate both the effectiveness and efficiency of our method. Check out our project page in this https URL.
>
---
#### [new 109] LISTA-Transformer Model Based on Sparse Coding and Attention Mechanism and Its Application in Fault Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于工业故障诊断任务，旨在解决传统模型在局部特征建模和全局依赖捕捉上的不足。提出LISTA-Transformer模型，结合稀疏编码与注意力机制，提升故障识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.04146](https://arxiv.org/pdf/2603.04146)**

> **作者:** Shuang Liu; Lina Zhao; Tian Wang; Huaqing Wang
>
> **备注:** 14 pages, 14 figures, conference paper
>
> **摘要:** Driven by the continuous development of models such as Multi-Layer Perceptron, Convolutional Neural Network (CNN), and Transformer, deep learning has made breakthrough progress in fields such as computer vision and natural language processing, and has been successfully applied in practical scenarios such as image classification and industrial fault diagnosis. However, existing models still have certain limitations in local feature modeling and global dependency capture. Specifically, CNN is limited by local receptive fields, while Transformer has shortcomings in effectively modeling local structures, and both face challenges of high model complexity and insufficient interpretability. In response to the above issues, we proposes the following innovative work: A sparse Transformer based on Learnable Iterative Shrinkage Threshold Algorithm (LISTA-Transformer) was designed, which deeply integrates LISTA sparse encoding with visual Transformer to construct a model architecture with adaptive local and global feature collaboration mechanism. This method utilizes continuous wavelet transform to convert vibration signals into time-frequency maps and inputs them into LISTA-Transformer for more effective feature extraction. On the CWRU dataset, the fault recognition rate of our method reached 98.5%, which is 3.3% higher than traditional methods and exhibits certain superiority over existing Transformer-based approaches.
>
---
#### [new 110] ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning in Instructional Videos
- **分类: cs.CV**

- **简介: 该论文属于程序规划任务，旨在提高智能体在复杂环境中的动作序列预测能力。提出ViterbiPlanNet，通过可微维特比层显式整合过程知识，提升样本效率和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04265](https://arxiv.org/pdf/2603.04265)**

> **作者:** Luigi Seminara; Davide Moltisanti; Antonino Furnari
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Procedural planning aims to predict a sequence of actions that transforms an initial visual state into a desired goal, a fundamental ability for intelligent agents operating in complex environments. Existing approaches typically rely on large-scale models that learn procedural structures implicitly, resulting in limited sample-efficiency and high computational cost. In this work we introduce ViterbiPlanNet, a principled framework that explicitly integrates procedural knowledge into the learning process through a Differentiable Viterbi Layer (DVL). The DVL embeds a Procedural Knowledge Graph (PKG) directly with the Viterbi decoding algorithm, replacing non-differentiable operations with smooth relaxations that enable end-to-end optimization. This design allows the model to learn through graph-based decoding. Experiments on CrossTask, COIN, and NIV demonstrate that ViterbiPlanNet achieves state-of-the-art performance with an order of magnitude fewer parameters than diffusion- and LLM-based planners. Extensive ablations show that performance gains arise from our differentiable structure-aware training rather than post-hoc refinement, resulting in improved sample efficiency and robustness to shorter unseen horizons. We also address testing inconsistencies establishing a unified testing protocol with consistent splits and evaluation metrics. With this new protocol, we run experiments multiple times and report results using bootstrapping to assess statistical significance.
>
---
#### [new 111] Helios: Real Real-Time Long Video Generation Model
- **分类: cs.CV**

- **简介: 该论文提出Helios，首个14B实时长视频生成模型，解决长视频生成中的漂移、效率及训练问题，支持多种视频生成任务。**

- **链接: [https://arxiv.org/pdf/2603.04379](https://arxiv.org/pdf/2603.04379)**

> **作者:** Shenghai Yuan; Yuanyang Yin; Zongjian Li; Xinwei Huang; Xiao Yang; Li Yuan
>
> **备注:** Page: this http URL
>
> **摘要:** We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to -- or lower than -- those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.
>
---
#### [new 112] Universal Pansharpening Foundation Model
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理任务，旨在解决 pansharpening 的通用性和鲁棒性问题。提出 FoundPS 模型，实现跨传感器和场景的高效融合。**

- **链接: [https://arxiv.org/pdf/2603.03831](https://arxiv.org/pdf/2603.03831)**

> **作者:** Hebaixu Wang; Jing Zhang; Haonan Guo; Di Wang; Jiayi Ma; Bo Du; Liangpei Zhang
>
> **摘要:** Pansharpening generates the high-resolution multi-spectral (MS) image by integrating spatial details from a texture-rich panchromatic (PAN) image and spectral attributes from a low-resolution MS image. Existing methods are predominantly satellite-specific and scene-dependent, which severely limits their generalization across heterogeneous sensors and varied scenes, thereby reducing their real-world practicality. To address these challenges, we present FoundPS, a universal pansharpening foundation model for satellite-agnostic and scene-robust fusion. Specifically, we introduce a modality-interleaved transformer that learns band-wise modal specializations to form reversible spectral affine bases, mapping arbitrary-band MS into a unified latent space via tensor multiplication. Building upon this, we construct a latent diffusion bridge model to progressively evolve latent representations, and incorporate bridge posterior sampling to couple latent diffusion with pixel-space observations, enabling stable and controllable fusion. Furthermore, we devise infinite-dimensional pixel-to-latent interaction mechanisms to comprehensively capture the cross-domain dependencies between PAN observations and MS representations, thereby facilitating complementary information fusion. In addition, to support large-scale training and evaluation, we construct a comprehensive pansharpening benchmark, termed PSBench, consisting of worldwide MS and PAN image pairs from multiple satellites across diverse scenes. Extensive experiments demonstrate that FoundPS consistently outperforms state-of-the-art methods, exhibiting superior generalization and robustness across a wide range of pansharpening tasks.
>
---
#### [new 113] LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于多视角立体视觉任务，旨在解决自动驾驶中深度估计的精度、一致性与跨域泛化问题。提出DriveMVS框架，结合LiDAR提示与时空融合，提升深度估计性能。**

- **链接: [https://arxiv.org/pdf/2603.03765](https://arxiv.org/pdf/2603.03765)**

> **作者:** Qihao Sun; Jiarun Liu; Ziqian Ni; Jianyun Xu; Tao Xie; Lijun Zhao; Ruifeng Li; Sheng Yang
>
> **摘要:** Accurate metric depth is critical for autonomous driving perception and simulation, yet current approaches struggle to achieve high metric accuracy, multi-view and temporal consistency, and cross-domain generalization. To address these challenges, we present DriveMVS, a novel multi-view stereo framework that reconciles these competing objectives through two key insights: (1) Sparse but metrically accurate LiDAR observations can serve as geometric prompts to anchor depth estimation in absolute scale, and (2) deep fusion of diverse cues is essential for resolving ambiguities and enhancing robustness, while a spatio-temporal decoder ensures consistency across frames. Built upon these principles, DriveMVS embeds the LiDAR prompt in two ways: as a hard geometric prior that anchors the cost volume, and as soft feature-wise guidance fused by a triple-cue combiner. Regarding temporal consistency, DriveMVS employs a spatio-temporal decoder that jointly leverages geometric cues from the MVS cost volume and temporal context from neighboring frames. Experiments show that DriveMVS achieves state-of-the-art performance on multiple benchmarks, excelling in metric accuracy, temporal stability, and zero-shot cross-domain transfer, demonstrating its practical value for scalable, reliable autonomous driving systems.
>
---
#### [new 114] RANGER: Sparsely-Gated Mixture-of-Experts with Adaptive Retrieval Re-ranking for Pathology Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于病理报告生成任务，解决WSI复杂性带来的生成困难。提出RANGER框架，结合稀疏门控MoE和自适应检索重排序，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.04348](https://arxiv.org/pdf/2603.04348)**

> **作者:** Yixin Chen; Ziyu Su; Hikmat Khan; Muhammad Khalid Khan Niazi
>
> **摘要:** Pathology report generation remains a relatively under-explored downstream task, primarily due to the gigapixel scale and complex morphological heterogeneity of Whole Slide Images (WSIs). Existing pathology report generation frameworks typically employ transformer architectures, relying on a homogeneous decoder architecture and static knowledge retrieval integration. Such architectures limit generative specialization and may introduce noisy external guidance during the report generation process. To address these limitations, we propose RANGER, a sparsely-gated Mixture-of-Experts (MoE) framework with adaptive retrieval re-ranking for pathology report generation. Specifically, we integrate a sparsely gated MoE into the decoder, along with noisy top-$k$ routing and load-balancing regularization, to enable dynamic expert specialization across various diagnostic patterns. Additionally, we introduce an adaptive retrieval re-ranking module that selectively refines retrieved memory from a knowledge base before integration, reducing noise and improving semantic alignment based on visual feature representations. We perform extensive experiments on the PathText-BRCA dataset and demonstrate consistent improvements over existing approaches across standard natural language generation metrics. Our full RANGER model achieves optimal performance on PathText dataset, reaching BLEU-1 to BLEU-4 scores of 0.4598, 0.3044, 0.2036, and 0.1435, respectively, with METEOR of 0.1883, and ROUGE-L of 0.3038, validating the effectiveness of dynamic expert routing and adaptive knowledge refinement for semantically grounded pathology report generation.
>
---
#### [new 115] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于360°视频生成任务，解决高分辨率生成难题。提出CubeComposer模型，实现4K 360°视频的原生生成与高质量合成。**

- **链接: [https://arxiv.org/pdf/2603.04291](https://arxiv.org/pdf/2603.04291)**

> **作者:** Lingen Li; Guangzhi Wang; Xiaoyu Li; Zhaoyang Zhang; Qi Dou; Jinwei Gu; Tianfan Xue; Ying Shan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Generating high-quality 360° panoramic videos from perspective input is one of the crucial applications for virtual reality (VR), whereby high-resolution videos are especially important for immersive experience. Existing methods are constrained by computational limitations of vanilla diffusion models, only supporting $\leq$ 1K resolution native generation and relying on suboptimal post super-resolution to increase resolution. We introduce CubeComposer, a novel spatio-temporal autoregressive diffusion model that natively generates 4K-resolution 360° videos. By decomposing videos into cubemap representations with six faces, CubeComposer autoregressively synthesizes content in a well-planned spatio-temporal order, reducing memory demands while enabling high-resolution output. Specifically, to address challenges in multi-dimensional autoregression, we propose: (1) a spatio-temporal autoregressive strategy that orchestrates 360° video generation across cube faces and time windows for coherent synthesis; (2) a cube face context management mechanism, equipped with a sparse context attention design to improve efficiency; and (3) continuity-aware techniques, including cube-aware positional encoding, padding, and blending to eliminate boundary seams. Extensive experiments on benchmark datasets demonstrate that CubeComposer outperforms state-of-the-art methods in native resolution and visual quality, supporting practical VR application scenarios. Project page: this https URL
>
---
#### [new 116] From Local Matches to Global Masks: Novel Instance Detection in Open-World Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于开放场景下的实例检测任务，解决在无约束环境中基于少量模板图像定位和分割新物体的问题。提出L2G-Det框架，通过局部匹配生成候选点并优化得到完整掩码。**

- **链接: [https://arxiv.org/pdf/2603.03577](https://arxiv.org/pdf/2603.03577)**

> **作者:** Qifan Zhang; Sai Haneesh Allu; Jikai Wang; Yangxiao Lu; Yu Xiang
>
> **摘要:** Detecting and segmenting novel object instances in open-world environments is a fundamental problem in robotic perception. Given only a small set of template images, a robot must locate and segment a specific object instance in a cluttered, previously unseen scene. Existing proposal-based approaches are highly sensitive to proposal quality and often fail under occlusion and background clutter. We propose L2G-Det, a local-to-global instance detection framework that bypasses explicit object proposals by leveraging dense patch-level matching between templates and the query image. Locally matched patches generate candidate points, which are refined through a candidate selection module to suppress false positives. The filtered points are then used to prompt an augmented Segment Anything Model (SAM) with instance-specific object tokens, enabling reliable reconstruction of complete instance masks. Experiments demonstrate improved performance over proposal-based methods in challenging open-world settings.
>
---
#### [new 117] SSR: A Generic Framework for Text-Aided Map Compression for Localization
- **分类: cs.CV**

- **简介: 该论文属于机器人定位任务，旨在解决地图存储与传输的高成本问题。通过引入文本增强的压缩框架SSR，结合文本和图像特征实现高效压缩，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.04272](https://arxiv.org/pdf/2603.04272)**

> **作者:** Mohammad Omama; Po-han Li; Harsh Goel; Minkyu Choi; Behdad Chalaki; Vaishnav Tadiparthi; Hossein Nourkhiz Mahjoub; Ehsan Moradi Pari; Sandeep P. Chinchali
>
> **摘要:** Mapping is crucial in robotics for localization and downstream decision-making. As robots are deployed in ever-broader settings, the maps they rely on continue to increase in size. However, storing these maps indefinitely (cold storage), transferring them across networks, or sending localization queries to cloud-hosted maps imposes prohibitive memory and bandwidth costs. We propose a text-enhanced compression framework that reduces both memory and bandwidth footprints while retaining high-fidelity localization. The key idea is to treat text as an alternative modality: one that can be losslessly compressed with large language models. We propose leveraging lightweight text descriptions combined with very small image feature vectors, which capture "complementary information" as a compact representation for the mapping task. Building on this, our novel technique, Similarity Space Replication (SSR), learns an adaptive image embedding in one shot that captures only the information "complementary" to the text descriptions. We validate our compression framework on multiple downstream localization tasks, including Visual Place Recognition as well as object-centric Monte Carlo localization in both indoor and outdoor settings. SSR achieves 2 times better compression than competing baselines on state-of-the-art datasets, including TokyoVal, Pittsburgh30k, Replica, and KITTI.
>
---
#### [new 118] Rethinking the Efficiency and Effectiveness of Reinforcement Learning for Radiology Report Generation
- **分类: cs.CV**

- **简介: 该论文聚焦于放射学报告生成任务，旨在提升强化学习的效率与效果。针对数据质量和临床关键标记的问题，提出DiTPO方法，优化临床准确性。**

- **链接: [https://arxiv.org/pdf/2603.04022](https://arxiv.org/pdf/2603.04022)**

> **作者:** Zilin Lu; Ruifeng Yuan; Weiwei Cao; Wanxing Chang; Zhongyu Wei; Sinuo Wang; Yong Xia; Ling Zhang; Jianpeng Zhang
>
> **摘要:** Radiologists highly desire fully automated AI for radiology report generation (R2G), yet existing approaches fall short in clinical utility. Reinforcement learning (RL) holds potential to address these shortcomings, but its adoption in this task remains underexplored. In this paper, we revisit RL in terms of data efficiency and optimization effectiveness for R2G tasks. First, we explore the impact of data quantity and quality on the performance of RL in medical contexts, revealing that data quality plays a more critical role than quantity. To this end, we propose a diagnostic diversity-based data sampling strategy that enables comparable performance with fewer samples. Second, we observe that the majority of tokens in radiology reports are template-like and diagnostically uninformative, whereas the low frequency of clinically critical tokens heightens the risk of being overlooked during optimization. To tackle this, we introduce Diagnostic Token-weighted Policy Optimization (DiTPO), which directly optimizes for clinical accuracy by using a diagnostic F1 score as the reward signal. Unlike standard RL approaches that treat all tokens equally, DiTPO explicitly models the varying importance of different tokens through rule- or gradient-based mechanisms to prioritize clinically relevant content. Extensive experiments on the MIMIC-CXR, IU-Xray, and CheXpert Plus datasets demonstrate that our framework achieves state-of-the-art (SOTA) performance while requiring substantially fewer training samples in RL. Notably, on MIMIC-CXR, our framework attains an F1 score of 0.516 using only 20% of the RL training samples.
>
---
#### [new 119] Beyond Mixtures and Products for Ensemble Aggregation: A Likelihood Perspective on Generalized Means
- **分类: stat.ML; cs.CV; cs.LG; math.ST; stat.ME**

- **简介: 该论文研究机器学习中的密度聚合问题，探讨不同聚合方法的优劣。通过似然视角分析广义均值，证明r∈[0,1]范围内的方法更可靠，为线性与几何聚合提供理论依据。**

- **链接: [https://arxiv.org/pdf/2603.04204](https://arxiv.org/pdf/2603.04204)**

> **作者:** Raphaël Razafindralambo; Rémy Sun; Frédéric Precioso; Damien Garreau; Pierre-Alexandre Mattei
>
> **摘要:** Density aggregation is a central problem in machine learning, for instance when combining predictions from a Deep Ensemble. The choice of aggregation remains an open question with two commonly proposed approaches being linear pooling (probability averaging) and geometric pooling (logit averaging). In this work, we address this question by studying the normalized generalized mean of order $r \in \mathbb{R} \cup \{-\infty,+\infty\}$ through the lens of log-likelihood, the standard evaluation criterion in machine learning. This provides a unifying aggregation formalism and shows different optimal configurations for different situations. We show that the regime $r \in [0,1]$ is the only range ensuring systematic improvements relative to individual distributions, thereby providing a principled justification for the reliability and widespread practical use of linear ($r=1$) and geometric ($r=0$) pooling. In contrast, we show that aggregation rules with $r \notin [0,1]$ may fail to provide consistent gains with explicit counterexamples. Finally, we corroborate our theoretical findings with empirical evaluations using Deep Ensembles on image and text classification benchmarks.
>
---
#### [new 120] Deep Sketch-Based 3D Modeling: A Survey
- **分类: cs.GR; cs.CV; cs.HC**

- **简介: 该论文属于3D建模任务，旨在解决草图到3D模型转换中的抽象与歧义问题。通过构建MORPHEUS框架，综述最新DS-3DM方法，提升交互灵活性与用户意图匹配度。**

- **链接: [https://arxiv.org/pdf/2603.03287](https://arxiv.org/pdf/2603.03287)**

> **作者:** Alberto Tono; Jiajun Wu; Gordon Wetzstein; Iro Armeni; Hariharan Subramonyam; James Landay; Martin Fischer
>
> **摘要:** In the past decade, advances in artificial intelligence have revolutionized sketch-based 3D modeling, leading to a new paradigm known as Deep Sketch-Based 3D Modeling (DS-3DM). DS-3DM offers data-driven methods that address the long-standing challenges of sketch abstraction and ambiguity. DS-3DM keeps humans at the center of the creative process by enhancing the flexibility, usability, faithfulness, and adaptability of sketch-based 3D modeling interfaces. This paper contributes a comprehensive survey of the latest DS-3DM within a novel design space: MORPHEUS. Built upon the Input-Model-Output (IMO) framework, MORPHEUS categorizes Models outputting Options of 3D Representations and Parts, derived from Human inputs (varying in quantity and modality), and Evaluated across diverse User-views and Styles. Throughout MORPHEUS we highlight limitations and identify opportunities for interdisciplinary research in Computer Vision, Computer Graphics, and Human-Computer Interaction, revealing a need for controllability and information-rich outputs. These opportunities align design processes more closely with user' intent, responding to the growing importance of user-centered approaches.
>
---
#### [new 121] RVN-Bench: A Benchmark for Reactive Visual Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RVN-Bench，用于室内安全视觉导航的基准测试。解决现有基准不适用于室内或忽略碰撞的问题，通过构建包含多样化环境和碰撞感知评估的基准，支持有效训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.03953](https://arxiv.org/pdf/2603.03953)**

> **作者:** Jaewon Lee; Jaeseok Heo; Gunmin Lee; Howoong Jun; Jeongwoo Oh; Songhwai Oh
>
> **摘要:** Safe visual navigation is critical for indoor mobile robots operating in cluttered environments. Existing benchmarks, however, often neglect collisions or are designed for outdoor scenarios, making them unsuitable for indoor visual navigation. To address this limitation, we introduce the reactive visual navigation benchmark (RVN-Bench), a collision-aware benchmark for indoor mobile robots. In RVN-Bench, an agent must reach sequential goal positions in previously unseen environments using only visual observations and no prior map, while avoiding collisions. Built on the Habitat 2.0 simulator and leveraging high-fidelity HM3D scenes, RVN-Bench provides large-scale, diverse indoor environments, defines a collision-aware navigation task and evaluation metrics, and offers tools for standardized training and benchmarking. RVN-Bench supports both online and offline learning by offering an environment for online reinforcement learning, a trajectory image dataset generator, and tools for producing negative trajectory image datasets that capture collision events. Experiments show that policies trained on RVN-Bench generalize effectively to unseen environments, demonstrating its value as a standardized benchmark for safe and robust visual navigation. Code and additional materials are available at: this https URL.
>
---
#### [new 122] The Influence of Iconicity in Transfer Learning for Sign Language Recognition
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于手势识别任务，研究如何通过迁移学习提升手语识别效果。通过比较不同手语对的象似性符号在迁移学习中的表现，验证了象似性对知识迁移的重要性。**

- **链接: [https://arxiv.org/pdf/2603.03316](https://arxiv.org/pdf/2603.03316)**

> **作者:** Keren Artiaga; Conor Lynch; Haithem Afli; Mohammed Hasanuzzaman
>
> **摘要:** Most sign language recognition research relies on Transfer Learning (TL) from vision-based datasets such as ImageNet. Some extend this to alternatively available language datasets, often focusing on signs with cross-linguistic similarities. This body of work examines the necessity of these likenesses on effective knowledge transfer by comparing TL performance between iconic signs of two different sign language pairs: Chinese to Arabic and Greek to Flemish. Google Mediapipe was utilised as an input feature extractor, enabling spatial information of these signs to be processed with a Multilayer Perceptron architecture and the temporal information with a Gated Recurrent Unit. Experimental results showed a 7.02% improvement for Arabic and 1.07% for Flemish when conducting iconic TL from Chinese and Greek respectively.
>
---
#### [new 123] Tuning Just Enough: Lightweight Backdoor Attacks on Multi-Encoder Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究多编码器扩散模型的后门攻击问题，提出轻量级攻击方法MELT，在仅调整少量参数情况下实现有效攻击，揭示了多编码器模型的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2603.04064](https://arxiv.org/pdf/2603.04064)**

> **作者:** Ziyuan Chen; Yujin Jeong; Tobias Braun; Anna Rohrbach
>
> **摘要:** As text-to-image diffusion models become increasingly deployed in real-world applications, concerns about backdoor attacks have gained significant attention. Prior work on text-based backdoor attacks has largely focused on diffusion models conditioned on a single lightweight text encoder. However, more recent diffusion models that incorporate multiple large-scale text encoders remain underexplored in this context. Given the substantially increased number of trainable parameters introduced by multiple text encoders, an important question is whether backdoor attacks can remain both efficient and effective in such settings. In this work, we study Stable Diffusion 3, which uses three distinct text encoders and has not yet been systematically analyzed for text-encoder-based backdoor vulnerabilities. To understand the role of text encoders in backdoor attacks, we define four categories of attack targets and identify the minimal sets of encoders required to achieve effective performance for each attack objective. Based on this, we further propose Multi-Encoder Lightweight aTtacks (MELT), which trains only low-rank adapters while keeping the pretrained text encoder weight frozen. We demonstrate that tuning fewer than 0.2% of the total encoder parameters is sufficient for successful backdoor attacks on Stable Diffusion 3, revealing previously underexplored vulnerabilities in practical attack scenarios in multi-encoder settings.
>
---
#### [new 124] Nearest-Neighbor Density Estimation for Dependency Suppression
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据去偏任务，旨在去除数据中的不利依赖关系。通过编码器方法学习与敏感变量无关的表示，结合变分自编码器和最近邻密度估计，实现更有效的独立性优化。**

- **链接: [https://arxiv.org/pdf/2603.04224](https://arxiv.org/pdf/2603.04224)**

> **作者:** Kathleen Anderson; Thomas Martinetz
>
> **摘要:** The ability to remove unwanted dependencies from data is crucial in various domains, including fairness, robust learning, and privacy protection. In this work, we propose an encoder-based approach that learns a representation independent of a sensitive variable but otherwise preserving essential data characteristics. Unlike existing methods that rely on decorrelation or adversarial learning, our approach explicitly estimates and modifies the data distribution to neutralize statistical dependencies. To achieve this, we combine a specialized variational autoencoder with a novel loss function driven by non-parametric nearest-neighbor density estimation, enabling direct optimization of independence. We evaluate our approach on multiple datasets, demonstrating that it can outperform existing unsupervised techniques and even rival supervised methods in balancing information removal and utility.
>
---
#### [new 125] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成任务，旨在降低扩散模型采样时的计算成本。通过引入Dual-Solver，优化预测类型、积分域和残差项，提升低NFE下的生成质量。**

- **链接: [https://arxiv.org/pdf/2603.03973](https://arxiv.org/pdf/2603.03973)**

> **作者:** Soochul Park; Yeon Ju Lee
>
> **备注:** Published as a conference paper at ICLR 2026. 36 pages, 18 figures
>
> **摘要:** Diffusion models achieve state-of-the-art image quality. However, sampling is costly at inference time because it requires a large number of function evaluations (NFEs). To reduce NFEs, classical ODE numerical methods have been adopted. Yet, the choice of prediction type and integration domain leads to different sampling behaviors. To address these issues, we introduce Dual-Solver, which generalizes multistep samplers through learnable parameters that continuously (i) interpolate among prediction types, (ii) select the integration domain, and (iii) adjust the residual terms. It retains the standard predictor-corrector structure while preserving second-order local accuracy. These parameters are learned via a classification-based objective using a frozen pretrained classifier (e.g., MobileNet or CLIP). For ImageNet class-conditional generation (DiT, GM-DiT) and text-to-image generation (SANA, PixArt-$\alpha$), Dual-Solver improves FID and CLIP scores in the low-NFE regime ($3 \le$ NFE $\le 9$) across backbones.
>
---
#### [new 126] Impact of Localization Errors on Label Quality for Online HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精地图构建任务，研究定位误差对标签质量的影响。通过引入不同噪声类型，分析其对模型性能的影响，并提出基于距离的评估指标。**

- **链接: [https://arxiv.org/pdf/2603.03452](https://arxiv.org/pdf/2603.03452)**

> **作者:** Alexander Blumberg; Jonas Merkert; Richard Fehler; Fabian Immel; Frank Bieder; Jan-Hendrik Pauls; Christoph Stiller
>
> **备注:** Accepted for the 36th IEEE Intelligent Vehicles Symposium (IV 2025), 8 pages
>
> **摘要:** High-definition (HD) maps are crucial for autonomous vehicles, but their creation and maintenance is very costly. This motivates the idea of online HD map construction. To provide a continuous large-scale stream of training data, existing HD maps can be used as labels for onboard sensor data from consumer vehicle fleets. However, compared to current, well curated HD map perception datasets, this fleet data suffers from localization errors, resulting in distorted map labels. We introduce three kinds of localization errors, Ramp, Gaussian, and Perlin noise, to examine their influence on generated map labels. We train a variant of MapTRv2, a state-of-the-art online HD map construction model, on the Argoverse 2 dataset with various levels of localization errors and assess the degradation of model performance. Since localization errors affect distant labels more severely, but are also less significant to driving performance, we introduce a distance-based map construction metric. Our experiments reveal that localization noise affects the model performance significantly. We demonstrate that errors in heading angle exert a more substantial influence than position errors, as angle errors result in a greater distortion of labels as distance to the vehicle increases. Furthermore, we can demonstrate that the model benefits from non-distorted ground truth (GT) data and that the performance decreases more than linearly with the increase in noisy data. Our study additionally provides a qualitative evaluation of the extent to which localization errors influence the construction of HD maps.
>
---
#### [new 127] Spectrum Shortage for Radio Sensing? Leveraging Ambient 5G Signals for Human Activity Detection
- **分类: cs.NI; cs.CV**

- **简介: 该论文属于无线电感知任务，解决频谱稀缺问题。通过利用5G信号实现人体活动检测，提出ARF系统及跨模态学习框架。**

- **链接: [https://arxiv.org/pdf/2603.03579](https://arxiv.org/pdf/2603.03579)**

> **作者:** Kunzhe Song; Maxime Zingraff; Huacheng Zeng
>
> **摘要:** Radio sensing in the sub-10 GHz spectrum offers unique advantages over traditional vision-based systems, including the ability to see through occlusions and preserve user privacy. However, the limited availability of spectrum in this range presents significant challenges for deploying largescale radio sensing applications. In this paper, we introduce Ambient Radio Sensing (ARS), a novel Integrated Sensing and Communications (ISAC) approach that addresses spectrum scarcity by repurposing over-the-air radio signals from existing wireless systems (e.g., 5G and Wi-Fi) for sensing applications, without interfering with their primary communication functions. ARS operates as a standalone device that passively receives communication signals, amplifies them to illuminate surrounding objects, and captures the reflected signals using a self-mixing RF architecture to extract baseband features. This hardware innovation enables robust Doppler and angular feature extraction from ambient OFDM signals. To support downstream applications, we propose a cross-modal learning framework focusing on human activity recognition, featuring a streamlined training process that leverages an off-the-shelf vision model to supervise radio model training. We have developed a prototype of ARS and validated its effectiveness through extensive experiments using ambient 5G signals, demonstrating accurate human skeleton estimation and body mask segmentation applications.
>
---
#### [new 128] Polyp Segmentation Using Wavelet-Based Cross-Band Integration for Enhanced Boundary Representation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决结肠息肉边界定位困难的问题。通过融合灰度与RGB信息，提升边界精度。**

- **链接: [https://arxiv.org/pdf/2603.03682](https://arxiv.org/pdf/2603.03682)**

> **作者:** Haesung Oh; Jaesung Lee
>
> **备注:** 39th Annual Conference on Neural Information Processing Systems in Europe (EurIPS 2025) Workshop, Copenhagen, Denmark, 2-7 December 2025 MedEurIPS:Medical Imagine Meets EurIPS
>
> **摘要:** Accurate polyp segmentation is essential for early colorectal cancer detection, yet achieving reliable boundary localization remains challenging due to low mucosal contrast, uneven illumination, and color similarity between polyps and surrounding tissue. Conventional methods relying solely on RGB information often struggle to delineate precise boundaries due to weak contrast and ambiguous structures between polyps and surrounding mucosa. To establish a quantitative foundation for this limitation, we analyzed polyp-background contrast in the wavelet domain, revealing that grayscale representations consistently preserve higher boundary contrast than RGB images across all frequency bands. This finding suggests that boundary cues are more distinctly represented in the grayscale domain than in the color domain. Motivated by this finding, we propose a segmentation model that integrates grayscale and RGB representations through complementary frequency-consistent interaction, enhancing boundary precision while preserving structural coherence. Extensive experiments on four benchmark datasets demonstrate that the proposed approach achieves superior boundary precision and robustness compared to conventional models.
>
---
#### [new 129] Structural Action Transformer for 3D Dexterous Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决高自由度机械手跨实体技能迁移问题。提出结构化动作Transformer，通过3D结构视角提升样本效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03960](https://arxiv.org/pdf/2603.03960)**

> **作者:** Xiaohan Lei; Min Wang; Bohong Weng; Wengang Zhou; Houqiang Li
>
> **备注:** Accepted by CVPR
>
> **摘要:** Achieving human-level dexterity in robots via imitation learning from heterogeneous datasets is hindered by the challenge of cross-embodiment skill transfer, particularly for high-DoF robotic hands. Existing methods, often relying on 2D observations and temporal-centric action representation, struggle to capture 3D spatial relations and fail to handle embodiment heterogeneity. This paper proposes the Structural Action Transformer (SAT), a new 3D dexterous manipulation policy that challenges this paradigm by introducing a structural-centric perspective. We reframe each action chunk not as a temporal sequence, but as a variable-length, unordered sequence of joint-wise trajectories. This structural formulation allows a Transformer to natively handle heterogeneous embodiments, treating the joint count as a variable sequence length. To encode structural priors and resolve ambiguity, we introduce an Embodied Joint Codebook that embeds each joint's functional role and kinematic properties. Our model learns to generate these trajectories from 3D point clouds via a continuous-time flow matching objective. We validate our approach by pre-training on large-scale heterogeneous datasets and fine-tuning on simulation and real-world dexterous manipulation tasks. Our method consistently outperforms all baselines, demonstrating superior sample efficiency and effective cross-embodiment skill transfer. This structural-centric representation offers a new path toward scaling policies for high-DoF, heterogeneous manipulators.
>
---
#### [new 130] Extending Neural Operators: Robust Handling of Functions Beyond the Training Set
- **分类: cs.LG; cs.CV; math.NA; math.OC; stat.ML**

- **简介: 该论文属于神经算子扩展任务，旨在解决模型在分布外输入函数上的泛化问题。通过核近似和RKHS理论，提升模型对函数及其导数的可靠捕捉能力。**

- **链接: [https://arxiv.org/pdf/2603.03621](https://arxiv.org/pdf/2603.03621)**

> **作者:** Blaine Quackenbush; Paul J. Atzberger
>
> **备注:** related open source software see this https URL
>
> **摘要:** We develop a rigorous framework for extending neural operators to handle out-of-distribution input functions. We leverage kernel approximation techniques and provide theory for characterizing the input-output function spaces in terms of Reproducing Kernel Hilbert Spaces (RKHSs). We provide theorems on the requirements for reliable extensions and their predicted approximation accuracy. We also establish formal relationships between specific kernel choices and their corresponding Sobolev Native Spaces. This connection further allows the extended neural operators to reliably capture not only function values but also their derivatives. Our methods are empirically validated through the solution of elliptic partial differential equations (PDEs) involving operators on manifolds having point-cloud representations and handling geometric contributions. We report results on key factors impacting the accuracy and computational performance of the extension approaches.
>
---
#### [new 131] HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决ORB-SLAM中二值化词袋词汇精度不足的问题。通过改进的分层BRB-KMeans方法，提升视觉词典的表达能力。**

- **链接: [https://arxiv.org/pdf/2603.04144](https://arxiv.org/pdf/2603.04144)**

> **作者:** Minjae Lee; Sang-Min Choi; Gun-Woo Kim; Suwon Lee
>
> **摘要:** In visual simultaneous localization and mapping (SLAM), the quality of the visual vocabulary is fundamental to the system's ability to represent environments and recognize locations. While ORB-SLAM is a widely used framework, its binary vocabulary, trained through the k-majority-based bag-of-words (BoW) approach, suffers from inherent precision loss. The inability of conventional binary clustering to represent subtle feature distributions leads to the degradation of visual words, a problem that is compounded as errors accumulate and propagate through the hierarchical tree structure. To address these structural deficiencies, this paper proposes hierarchical binary-to-real-and-back (HBRB)-BoW, a refined hierarchical binary vocabulary training algorithm. By integrating a global real-valued flow within the hierarchical clustering process, our method preserves high-fidelity descriptor information until the final binarization at the leaf nodes. Experimental results demonstrate that the proposed approach yields a more discriminative and well-structured vocabulary than traditional methods, significantly enhancing the representational integrity of the visual dictionary in complex environments. Furthermore, replacing the default ORB-SLAM vocabulary file with our HBRB-BoW file is expected to improve performance in loop closing and relocalization tasks.
>
---
#### [new 132] CRESTomics: Analyzing Carotid Plaques in the CREST-2 Trial with a New Additive Classification Model
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在通过影像学特征识别高风险颈动脉斑块。提出一种新模型，结合非线性分类与可解释性分析，提升斑块评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.04309](https://arxiv.org/pdf/2603.04309)**

> **作者:** Pranav Kulkarni; Brajesh K. Lal; Georges Jreij; Sai Vallamchetla; Langford Green; Jenifer Voeks; John Huston; Lloyd Edwards; George Howard; Bradley A. Maron; Thomas G. Brott; James F. Meschia; Florence X. Doo; Heng Huang
>
> **备注:** 4 pages, 3 figures, 1 table, accepted to ISBI 2026
>
> **摘要:** Accurate characterization of carotid plaques is critical for stroke prevention in patients with carotid stenosis. We analyze 500 plaques from CREST-2, a multi-center clinical trial, to identify radiomics-based markers from B-mode ultrasound images linked with high-risk. We propose a new kernel-based additive model, combining coherence loss with group-sparse regularization for nonlinear classification. Group-wise additive effects of each feature group are visualized using partial dependence plots. Results indicate our method accurately and interpretably assesses plaques, revealing a strong association between plaque texture and clinical risk.
>
---
#### [new 133] When and Where to Reset Matters for Long-Term Test-Time Adaptation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于长期测试时适应（TTA）任务，解决模型崩溃问题。提出自适应选择性重置方案，动态决定重置时机与位置，提升适应能力。**

- **链接: [https://arxiv.org/pdf/2603.03796](https://arxiv.org/pdf/2603.03796)**

> **作者:** Taejun Lim; Joong-Won Hwang; Kibok Lee
>
> **备注:** ICLR 2026
>
> **摘要:** When continual test-time adaptation (TTA) persists over the long term, errors accumulate in the model and further cause it to predict only a few classes for all inputs, a phenomenon known as model collapse. Recent studies have explored reset strategies that completely erase these accumulated errors. However, their periodic resets lead to suboptimal adaptation, as they occur independently of the actual risk of collapse. Moreover, their full resets cause catastrophic loss of knowledge acquired over time, even though such knowledge could be beneficial in the future. To this end, we propose (1) an Adaptive and Selective Reset (ASR) scheme that dynamically determines when and where to reset, (2) an importance-aware regularizer to recover essential knowledge lost due to reset, and (3) an on-the-fly adaptation adjustment scheme to enhance adaptability under challenging domain shifts. Extensive experiments across long-term TTA benchmarks demonstrate the effectiveness of our approach, particularly under challenging conditions. Our code is available at this https URL.
>
---
#### [new 134] Phi-4-reasoning-vision-15B Technical Report
- **分类: cs.AI; cs.CV**

- **简介: 该论文介绍Phi-4-reasoning-vision-15B模型，属于多模态推理任务，旨在提升小规模模型在视觉和语言任务中的表现，尤其强化科学数学推理和界面理解能力。通过优化架构和数据质量实现高效性能。**

- **链接: [https://arxiv.org/pdf/2603.03975](https://arxiv.org/pdf/2603.03975)**

> **作者:** Jyoti Aneja; Michael Harrison; Neel Joshi; Tyler LaBonte; John Langford; Eduardo Salinas
>
> **摘要:** We present Phi-4-reasoning-vision-15B, a compact open-weight multimodal reasoning model, and share the motivations, design choices, experiments, and learnings that informed its development. Our goal is to contribute practical insight to the research community on building smaller, efficient multimodal reasoning models and to share the result of these learnings as an open-weight model that is good at common vision and language tasks and excels at scientific and mathematical reasoning and understanding user interfaces. Our contributions include demonstrating that careful architecture choices and rigorous data curation enable smaller, open-weight multimodal models to achieve competitive performance with significantly less training and inference-time compute and tokens. The most substantial improvements come from systematic filtering, error correction, and synthetic augmentation -- reinforcing that data quality remains the primary lever for model performance. Systematic ablations show that high-resolution, dynamic-resolution encoders yield consistent improvements, as accurate perception is a prerequisite for high-quality reasoning. Finally, a hybrid mix of reasoning and non-reasoning data with explicit mode tokens allows a single model to deliver fast direct answers for simpler tasks and chain-of-thought reasoning for complex problems.
>
---
#### [new 135] Order Is Not Layout: Order-to-Space Bias in Image Generation
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究图像生成中的顺序-空间偏差问题，属于图像生成任务。针对实体顺序影响布局的偏差，提出OTSBench进行量化分析，并验证通过微调和早期干预可有效减少偏差。**

- **链接: [https://arxiv.org/pdf/2603.03714](https://arxiv.org/pdf/2603.03714)**

> **作者:** Yongkang Zhang; Zonglin Zhao; Yuechen Zhang; Fei Ding; Pei Li; Wenxuan Wang
>
> **摘要:** We study a systematic bias in modern image generation models: the mention order of entities in text spuriously determines spatial layout and entity--role binding. We term this phenomenon Order-to-Space Bias (OTS) and show that it arises in both text-to-image and image-to-image generation, often overriding grounded cues and causing incorrect layouts or swapped assignments. To quantify OTS, we introduce OTS-Bench, which isolates order effects with paired prompts differing only in entity order and evaluates models along two dimensions: homogenization and correctness. Experiments show that Order-to-Space Bias (OTS) is widespread in modern image generation models, and provide evidence that it is primarily data-driven and manifests during the early stages of layout formation. Motivated by this insight, we show that both targeted fine-tuning and early-stage intervention strategies can substantially reduce OTS, while preserving generation quality.
>
---
## 更新

#### [replaced 001] ROBUST-MIPS: A Combined Skeletal Pose and Instance Segmentation Dataset for Laparoscopic Surgical Instruments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21096](https://arxiv.org/pdf/2508.21096)**

> **作者:** Zhe Han; Charlie Budd; Gongyu Zhang; Huanyu Tian; Christos Bergeles; Tom Vercauteren
>
> **摘要:** Localisation of surgical tools constitutes a foundational building block for computer-assisted interventional technologies. Works in this field typically focus on training deep learning models to perform segmentation tasks. Performance of learning-based approaches is limited by the availability of diverse annotated data. We argue that skeletal pose annotations are a more efficient annotation approach for surgical tools, striking a balance between richness of semantic information and ease of annotation, thus allowing for accelerated growth of available annotated data. To encourage adoption of this annotation style, we present, ROBUST-MIPS, a combined tool pose and tool instance segmentation dataset derived from the existing ROBUST-MIS dataset. Our enriched dataset facilitates the joint study of these two annotation styles and allow head-to-head comparison on various downstream tasks. To demonstrate the adequacy of pose annotations for surgical tool localisation, we set up a simple benchmark using popular pose estimation methods and observe high-quality results. To ease adoption, together with the dataset, we release our benchmark models and custom tool pose annotation software.
>
---
#### [replaced 002] Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.19624](https://arxiv.org/pdf/2509.19624)**

> **作者:** Mahmoud Afifi; Ran Zhang; Michael S. Brown
>
> **摘要:** Digital cameras digitize scene light into linear raw representations, which the image signal processor (ISP) converts into display-ready outputs. While raw data preserves full sensor information--valuable for editing and vision tasks--formats such as Digital Negative (DNG) require large storage, making them impractical in constrained scenarios. In contrast, JPEG is a widely supported format, offering high compression efficiency and broad compatibility, but it is not well-suited for raw storage. This paper presents RawJPEG Adapter, a lightweight, learnable, and invertible preprocessing pipeline that adapts raw images for standard JPEG compression. Our method applies spatial and optional frequency-domain transforms, with compact parameters stored in the JPEG comment field, enabling accurate raw reconstruction. Experiments across multiple datasets show that our method achieves higher fidelity than direct JPEG storage, supports other codecs, and provides a favorable trade-off between compression ratio and reconstruction accuracy.
>
---
#### [replaced 003] Merlin: A Computed Tomography Vision-Language Foundation Model and Dataset
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2406.06512](https://arxiv.org/pdf/2406.06512)**

> **作者:** Louis Blankemeier; Ashwin Kumar; Joseph Paul Cohen; Jiaming Liu; Longchao Liu; Dave Van Veen; Syed Jamal Safdar Gardezi; Hongkun Yu; Magdalini Paschali; Zhihong Chen; Jean-Benoit Delbrouck; Eduardo Reis; Robbie Holland; Cesar Truyts; Christian Bluethgen; Yufu Wu; Long Lian; Malte Engmann Kjeldskov Jensen; Sophie Ostmeier; Maya Varma; Jeya Maria Jose Valanarasu; Zhongnan Fang; Zepeng Huo; Zaid Nabulsi; Diego Ardila; Wei-Hung Weng; Edson Amaro Junior; Neera Ahuja; Jason Fries; Nigam H. Shah; Greg Zaharchuk; Marc Willis; Adam Yala; Andrew Johnston; Robert D. Boutin; Andrew Wentland; Curtis P. Langlotz; Jason Hom; Sergios Gatidis; Akshay S. Chaudhari
>
> **备注:** Nature (2026)
>
> **摘要:** The large volume of abdominal computed tomography (CT) scans coupled with the shortage of radiologists have intensified the need for automated medical image analysis tools. Previous state-of-the-art approaches for automated analysis leverage vision-language models (VLMs) that jointly model images and radiology reports. However, current medical VLMs are generally limited to 2D images and short reports. Here to overcome these shortcomings for abdominal CT interpretation, we introduce Merlin, a 3D VLM that learns from volumetric CT scans, electronic health record data and radiology reports. This approach is enabled by a multistage pretraining framework that does not require additional manual annotations. We trained Merlin using a high-quality clinical dataset of paired CT scans (>6 million images from 15,331 CT scans), diagnosis codes (>1.8 million codes) and radiology reports (>6 million tokens). We comprehensively evaluated Merlin on 6 task types and 752 individual tasks that covered diagnostic, prognostic and quality-related tasks. The non-adapted (off-the-shelf) tasks included zero-shot classification of findings (30 findings), phenotype classification (692 phenotypes) and zero-shot cross-modal retrieval (image-to-findings and image-to-impression). The model-adapted tasks included 5-year chronic disease prediction (6 diseases), radiology report generation and 3D semantic segmentation (20 organs). We validated Merlin at scale, with internal testing on 5,137 CT scans and external testing on 44,098 CT scans from 3 independent sites and 2 public datasets. The results demonstrated high generalization across institutions and anatomies. Merlin outperformed 2D VLMs, CT foundation models and off-the-shelf radiology models. We also release our trained models, code, and dataset, available at: this https URL.
>
---
#### [replaced 004] CoShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02743](https://arxiv.org/pdf/2603.02743)**

> **作者:** Waqas Ahmed; Dean Diepeveen; Ferdous Sohel
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Realistic shadow generation is crucial for achieving seamless image compositing, yet existing methods primarily focus on single-object insertion and often fail to generalize when multiple foreground objects are composited into a background scene. In practice, however, modern compositing pipelines and real-world applications often insert multiple objects simultaneously, necessitating shadows that are jointly consistent in terms of geometry, attachment, and location. In this paper, we address the under-explored problem of multi-object shadow generation, aiming to synthesize physically plausible shadows for multiple inserted objects. Our approach exploits the multimodal capabilities of a pre-trained text-to-image diffusion model. An image pathway injects dense, multi-scale features to provide fine-grained spatial guidance, while a text-based pathway encodes per-object shadow bounding boxes as learned positional tokens and fuses them via cross-attention. An attention-alignment loss further grounds these tokens to their corresponding shadow regions. To support this task, we augment the DESOBAv2 dataset by constructing composite scenes with multiple inserted objects and automatically derive prompts combining object category and shadow positioning information. Experimental results demonstrate that our method achieves state-of-the-art performance in both single and multi-object shadow generation settings.
>
---
#### [replaced 005] Apple's Synthetic Defocus Noise Pattern: Characterization and Forensic Applications
- **分类: cs.CV; cs.CR; eess.IV**

- **链接: [https://arxiv.org/pdf/2505.07380](https://arxiv.org/pdf/2505.07380)**

> **作者:** David Vázquez-Padín; Fernando Pérez-González; Pablo Pérez-Miguélez
>
> **备注:** The last version of the paper is now published in IEEE Transactions on Information Forensics & Security, vol. 21, pp. 1096-1111, 2026
>
> **摘要:** iPhone portrait-mode images contain a distinctive pattern in out-of-focus regions simulating the bokeh effect, which we term Apple's Synthetic Defocus Noise Pattern (SDNP). If overlooked, this pattern can interfere with blind forensic analyses, especially PRNU-based camera source verification, as noted in earlier works. Since Apple's SDNP remains underexplored, we provide a detailed characterization, proposing a method for its precise estimation, modeling its dependence on scene brightness, ISO settings, and other factors. Leveraging this characterization, we explore forensic applications of the SDNP, including traceability of portrait-mode images across iPhone models and iOS versions in open-set scenarios, assessing its robustness under post-processing. Furthermore, we show that masking SDNP-affected regions in PRNU-based camera source verification significantly reduces false positives, overcoming a critical limitation in camera attribution, and improving state-of-the-art techniques.
>
---
#### [replaced 006] Improving Multi-View Reconstruction via Texture-Guided Gaussian-Mesh Joint Optimization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.03950](https://arxiv.org/pdf/2511.03950)**

> **作者:** Zhejia Cai; Puhua Jiang; Shiwei Mao; Hongkun Cao; Ruqi Huang
>
> **备注:** 10 pages, correct errors, clarify details, accepted to 3DV 2026
>
> **摘要:** Reconstructing real-world objects from multi-view images is essential for applications in 3D editing, AR/VR, and digital content creation. Existing methods typically prioritize either geometric accuracy (Multi-View Stereo) or photorealistic rendering (Novel View Synthesis), often decoupling geometry and appearance optimization, which hinders downstream editing tasks. This paper advocates an unified treatment on geometry and appearance optimization for seamless Gaussian-mesh joint optimization. More specifically, we propose a novel framework that simultaneously optimizes mesh geometry (vertex positions and faces) and vertex colors via Gaussian-guided mesh differentiable rendering, leveraging photometric consistency from input images and geometric regularization from normal and depth maps. The obtained high-quality 3D reconstruction can be further exploit in down-stream editing tasks, such as relighting and shape deformation. Our code will be released in this https URL
>
---
#### [replaced 007] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.19328](https://arxiv.org/pdf/2505.19328)**

> **作者:** Manuela González-González; Soufiane Belharbi; Muhammad Osama Zeeshan; Masoumeh Sharafi; Muhammad Haseeb Aslam; Marco Pedersoli; Alessandro Lameiras Koerich; Simon L Bacon; Eric Granger
>
> **备注:** 46 pages, 21 figures, ICLR 2026
>
> **摘要:** Ambivalence and hesitancy (A/H), closely related constructs, are the primary reasons why individuals delay, avoid, or abandon health behaviour changes. They are subtle and conflicting emotions that sets a person in a state between positive and negative orientations, or between acceptance and refusal to do something. They manifest as a discord in affect between multiple modalities or within a modality, such as facial and vocal expressions, and body language. Although experts can be trained to recognize A/H as done for in-person interactions, integrating them into digital health interventions is costly and less effective. Automatic A/H recognition is therefore critical for the personalization and cost-effectiveness of digital behaviour change interventions. However, no datasets currently exist for the design of machine learning models to recognize A/H. This paper introduces the Behavioural Ambivalence/Hesitancy (BAH) dataset collected for multimodal recognition of A/H in videos. It contains 1,427 videos with a total duration of 10.60 hours, captured from 300 participants across Canada, answering predefined questions to elicit A/H. It is intended to mirror real-world digital behaviour change interventions delivered online. BAH is annotated by three experts to provide timestamps that indicate where A/H occurs, and frame- and video-level annotations with A/H cues. Video transcripts, cropped and aligned faces, and participant metadata are also provided. Since A and H manifest similarly in practice, we provide a binary annotation indicating the presence or absence of A/H. Additionally, this paper includes benchmarking results using baseline models on BAH for frame- and video-level recognition, and different learning setups. The limited performance highlights the need for adapted multimodal and spatio-temporal models for A/H recognition. The data and code are publicly available.
>
---
#### [replaced 008] Specificity-aware reinforcement learning for fine-grained open-world classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03197](https://arxiv.org/pdf/2603.03197)**

> **作者:** Samuele Angheben; Davide Berasi; Alessandro Conti; Elisa Ricci; Yiming Wang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Classifying fine-grained visual concepts under open-world settings, i.e., without a predefined label set, demands models to be both accurate and specific. Recent reasoning Large Multimodal Models (LMMs) exhibit strong visual understanding capability but tend to produce overly generic predictions when performing fine-grained image classification. Our preliminary analysis reveals that models do possess the intrinsic fine-grained domain knowledge. However, promoting more specific predictions (specificity) without compromising correct ones (correctness) remains a non-trivial and understudied challenge. In this work, we investigate how to steer reasoning LMMs toward predictions that are both correct and specific. We propose a novel specificity-aware reinforcement learning framework, SpeciaRL, to fine-tune reasoning LMMs on fine-grained image classification under the open-world setting. SpeciaRL introduces a dynamic, verifier-based reward signal anchored to the best predictions within online rollouts, promoting specificity while respecting the model's capabilities to prevent incorrect predictions. Our out-of-domain experiments show that SpeciaRL delivers the best trade-off between correctness and specificity across extensive fine-grained benchmarks, surpassing existing methods and advancing open-world fine-grained image classification. Code and model are publicly available at this https URL.
>
---
#### [replaced 009] When Memory Becomes a Vulnerability: Towards Multi-turn Jailbreak Attacks against Text-to-Image Generation Systems
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2504.20376](https://arxiv.org/pdf/2504.20376)**

> **作者:** Shiqian Zhao; Jiayang Liu; Yiming Li; Runyi Hu; Xiaojun Jia; Wenshu Fan; Xiao Bao; Xinfeng Li; Jie Zhang; Wei Dong; Tianwei Zhang; Luu Anh Tuan
>
> **备注:** This work proposes a multi-turn jailbreak attack against real-world chat-based T2I generation systems that intergrate memory mechanism. It also constructed a simulation system, with considering three industrial-grade memory mechanisms, 7 kinds of safety filters (both input and output); It is going to appear on USENIX 2026
>
> **摘要:** Modern text-to-image (T2I) generation systems (e.g., DALL$\cdot$E 3) exploit the memory mechanism, which captures key information in multi-turn interactions for faithful generation. Despite its practicality, the security analyses of this mechanism have fallen far behind. In this paper, we reveal that it can exacerbate the risk of jailbreak attacks. Previous attacks fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or lead to the generation of non-unsafe images due to under- or over-detoxification. In contrast, we propose embedding the malice at the inception of the chat session in memory, addressing the above limitations. Specifically, we propose Inception, the first multi-turn jailbreak attack against real-world text-to-image generation systems that explicitly exploits their memory mechanisms. Inception is composed of two key modules: segmentation and recursion. We introduce Segmentation, a semantic-preserving method that generates multi-round prompts. By leveraging NLP analysis techniques, we design policies to decompose a prompt, together with its malicious intent, according to sentence structure, thereby evading safety filters. Recursion further addresses the challenge posed by unsafe sub-prompts that cannot be separated through simple segmentation. It firstly expands the sub-prompt, then invokes segmentation recursively. To facilitate multi-turn adversarial prompts crafting, we build VisionFlow, an emulation T2I system that integrates two-stage safety filters and industrial-grade memory mechanisms. The experiment results show that Inception successfully allures unsafe image generation, surpassing the SOTA by a 20.0\% margin in attack success rate. We also conduct experiments on the real-world commercial T2I generation platforms, further validating the threats of Inception in practice.
>
---
#### [replaced 010] Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于视频目标分割任务，解决动作引导下的标签噪声问题。通过引入噪声类型、构建基准并提出PMHM机制，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.16677](https://arxiv.org/pdf/2509.16677)**

> **作者:** Wenxin Li; Kunyu Peng; Di Wen; Ruiping Liu; Mengfei Duan; Kai Luo; Kailun Yang
>
> **备注:** Accepted to ICRA 2026. The established benchmark and source code will be made publicly available at this https URL
>
> **摘要:** Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. The established benchmark and source code will be made publicly available at this https URL.
>
---
#### [replaced 011] Kaleido: Open-Sourced Multi-Subject Reference Video Generation Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.18573](https://arxiv.org/pdf/2510.18573)**

> **作者:** Zhenxing Zhang; Jiayan Teng; Zhuoyi Yang; Tiankun Cao; Cheng Wang; Xiaotao Gu; Jie Tang; Dan Guo; Meng Wang
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** We present Kaleido, a subject-to-video~(S2V) generation framework, which aims to synthesize subject-consistent videos conditioned on multiple reference images of target subjects. Despite recent progress in S2V generation models, existing approaches remain inadequate at maintaining multi-subject consistency and at handling background disentanglement, often resulting in lower reference fidelity and semantic drift under multi-image conditioning. These shortcomings can be attributed to several factors. Primarily, the training dataset suffers from a lack of diversity and high-quality samples, as well as cross-paired data, i.e., paired samples whose components originate from different instances. In addition, the current mechanism for integrating multiple reference images is suboptimal, potentially resulting in the confusion of multiple subjects. To overcome these limitations, we propose a dedicated data construction pipeline, incorporating low-quality sample filtering and diverse data synthesis, to produce consistency-preserving training data. Moreover, we introduce Reference Rotary Positional Encoding (R-RoPE) to process reference images, enabling stable and precise multi-image integration. Extensive experiments across numerous benchmarks demonstrate that Kaleido significantly outperforms previous methods in consistency, fidelity, and generalization, marking an advance in S2V generation.
>
---
#### [replaced 012] Factuality Matters: When Image Generation and Editing Meet Structured Visuals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05091](https://arxiv.org/pdf/2510.05091)**

> **作者:** Le Zhuo; Songhao Han; Yuandong Pu; Boxiang Qiu; Sayak Paul; Yue Liao; Yihao Liu; Jie Shao; Xi Chen; Si Liu; Hongsheng Li
>
> **备注:** Accepted by ICLR 2026, Project page: this https URL
>
> **摘要:** While modern visual generation models excel at creating aesthetically pleasing natural images, they struggle with producing or editing structured visuals like charts, diagrams, and mathematical figures, which demand composition planning, text rendering, and multimodal reasoning for factual fidelity. To address this, we present the first comprehensive, systematic investigation of this domain, encompassing data construction, model training, and an evaluation benchmark. First, we construct a large-scale dataset of 1.3 million high-quality structured image pairs derived from executable drawing programs and augmented with chain-of-thought reasoning annotations. Building on it, we train a unified model that integrates a VLM with FLUX.1 Kontext via a lightweight connector for enhanced multimodal understanding. A three-stage training curriculum enables progressive feature alignment, knowledge infusion, and reasoning-augmented generation, further boosted by an external reasoner at inference time. Finally, we introduce StructBench, a novel benchmark for generation and editing with over 1,700 challenging instances, and an accompanying evaluation metric, StructScore, which employs a multi-round Q\&A protocol to assess fine-grained factual accuracy. Evaluations of 15 models reveal that even leading closed-source systems remain far from satisfactory. Our model attains strong editing performance, and inference-time reasoning yields consistent gains across diverse architectures. By releasing the dataset, model, and benchmark, we aim to advance unified multimodal foundations for structured visuals.
>
---
#### [replaced 013] Momentum Memory for Knowledge Distillation in Computational Pathology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21395](https://arxiv.org/pdf/2602.21395)**

> **作者:** Yongxin Guo; Hao Lu; Onur C. Koyun; Zhengjie Zhu; Muhammet Fatih Demir; Metin Nafi Gurcan
>
> **备注:** Accepted by CVPR 2026. Code: this https URL
>
> **摘要:** Multimodal learning that integrates genomics and histopathology has shown strong potential in cancer diagnosis, yet its clinical translation is hindered by the limited availability of paired histology-genomics data. Knowledge distillation (KD) offers a practical solution by transferring genomic supervision into histopathology models, enabling accurate inference using histology alone. However, existing KD methods rely on batch-local alignment, which introduces instability due to limited within-batch comparisons and ultimately degrades performance. To address these limitations, we propose Momentum Memory Knowledge Distillation (MoMKD), a cross-modal distillation framework driven by a momentum-updated memory. This memory aggregates genomic and histopathology information across batches, effectively enlarging the supervisory context available to each mini-batch. Furthermore, we decouple the gradients of the genomics and histology branches, preventing genomic signals from dominating histology feature learning during training and eliminating the modality-gap issue at inference time. Extensive experiments on the TCGA-BRCA benchmark (HER2, PR, and ODX classification tasks) and an independent in-house testing dataset demonstrate that MoMKD consistently outperforms state-of-the-art MIL and multimodal KD baselines, delivering strong performance and generalization under histology-only inference. Overall, MoMKD establishes a robust and generalizable knowledge distillation paradigm for computational pathology.
>
---
#### [replaced 014] Improved MambdaBDA Framework for Robust Building Damage Assessment Across Disaster Domains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01116](https://arxiv.org/pdf/2603.01116)**

> **作者:** Alp Eren Gençoğlu; Hazım Kemal Ekenel
>
> **备注:** Preprint. Accepted at VISAPP 2026
>
> **摘要:** Reliable post-disaster building damage assessment (BDA) from satellite imagery is hindered by severe class imbalance, background clutter, and domain shift across disaster types and geographies. In this work, we address these problems and explore ways to improve the MambaBDA, the BDA network of ChangeMamba architecture, one of the most successful BDA models. The approach enhances the MambaBDA with three modular components: (i) Focal Loss to mitigate class imbalance damage classification, (ii) lightweight Attention Gates to suppress irrelevant context, and (iii) a compact Alignment Module to spatially warp pre-event features toward post-event content before decoding. We experiment on multiple satellite imagery datasets, including xBD, Pakistan Flooding, Turkey Earthquake, and Ida Hurricane, and conduct in-domain and crossdataset tests. The proposed modular enhancements yield consistent improvements over the baseline model, with 0.8% to 5% performance gains in-domain, and up to 27% on unseen disasters. This indicates that the proposed enhancements are especially beneficial for the generalization capability of the system.
>
---
#### [replaced 015] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.25541](https://arxiv.org/pdf/2509.25541)**

> **作者:** Qinsi Wang; Bo Liu; Tianyi Zhou; Jing Shi; Yueqian Lin; Yiran Chen; Hai Helen Li; Kun Wan; Wentian Zhao
>
> **备注:** ICLR 2026
>
> **摘要:** Although reinforcement learning (RL) has emerged as a promising approach for improving vision-language models (VLMs) and multimodal large language models (MLLMs), current methods rely heavily on manually curated datasets and costly human verification, which limits scalable self-improvement in multimodal systems. To address this challenge, we propose Vision-Zero, a label-free, domain-agnostic multi-agent self-play framework for self-evolving VLMs through competitive visual games generated from arbitrary image inputs. Specifically, Vision-Zero encompasses three main attributes: (1) Strategic Self-Play Framework: Vision-Zero trains VLMs in "Who Is the Spy"-style games, where the models engage in strategic reasoning and actions across multiple roles. Through interactive gameplay, models autonomously generate their training data without human annotation. (2) Gameplay from Arbitrary Images: Unlike existing gamified frameworks, Vision-Zero can generate games from arbitrary images, thereby enhancing the model's reasoning ability across diverse domains and showing strong generalization to different tasks. We demonstrate this versatility using three distinct types of image datasets: CLEVR-based synthetic scenes, charts, and real-world images. (3) Sustainable Performance Gain: We introduce Iterative Self-Play Policy Optimization (Iterative-SPO), a novel training algorithm that alternates between Self-Play and reinforcement learning with verifiable rewards (RLVR), mitigating the performance plateau often seen in self-play-only training and achieving sustained long-term improvements. Despite using label-free data, Vision-Zero achieves state-of-the-art performance on reasoning, chart question answering, and vision-centric understanding tasks, surpassing other annotation-based methods. Models and code have been released at this https URL.
>
---
#### [replaced 016] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01603](https://arxiv.org/pdf/2508.01603)**

> **作者:** Yiheng Li; Zichang Tan; Guoqing Xu; Zhen Lei; Xu Zhou; Yang Yang
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** In AI-generated image detection, current cutting-edge methods typically adapt pre-trained foundation models through partial-parameter fine-tuning. However, these approaches often struggle to generalize to forgeries from unseen generators, as the fine-tuned models capture only limited patterns from training data and fail to reflect the evolving traits of new ones. To overcome this limitation, we propose Image-Adaptive Prompt Learning (IAPL), a novel paradigm that dynamically adjusts the prompts fed into the encoder according to each testing image, rather than fixing them after training. This design significantly enhances robustness and adaptability to diverse forged images. The dynamic prompts integrate conditional information with test-time adaptive tokens through a lightweight learnable scaling factor. The conditional information is produced by a Conditional Information Learner, which leverages CNN-based feature extractors to model both forgery-specific and general conditions. The test-time adaptive tokens are optimized during inference on a single sample by enforcing prediction consistency across multiple views, ensuring that the parameters align with the current image. For the final decision, the optimal input with the highest prediction confidence is selected. Extensive experiments show that IAPL achieves state-of-the-art performance, with mean accuracies of 95.61% and 96.7% on the widely used UniversalFakeDetect and GenImage datasets, respectively. Codes and weights will be released on this https URL.
>
---
#### [replaced 017] Skullptor: High Fidelity 3D Head Reconstruction in Seconds with Multi-View Normal Prediction
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2602.21100](https://arxiv.org/pdf/2602.21100)**

> **作者:** Noé Artru; Rukhshanda Hussain; Emeline Got; Alexandre Messier; David B. Lindell; Abdallah Dib
>
> **备注:** For our project page, see this https URL
>
> **摘要:** Reconstructing high-fidelity 3D head geometry from images is critical for a wide range of applications, yet existing methods face fundamental limitations. Traditional photogrammetry achieves exceptional detail but requires extensive camera arrays (25-200+ views), substantial computation, and manual cleanup in challenging areas like facial hair. Recent alternatives present a fundamental trade-off: foundation models enable efficient single-image reconstruction but lack fine geometric detail, while optimization-based methods achieve higher fidelity but require dense views and expensive computation. We bridge this gap with a hybrid approach that combines the strengths of both paradigms. Our method introduces a multi-view surface normal prediction model that extends monocular foundation models with cross-view attention to produce geometrically consistent normals in a feed-forward pass. We then leverage these predictions as strong geometric priors within an inverse rendering optimization framework to recover high-frequency surface details. Our approach outperforms state-of-the-art single-image and multi-view methods, achieving high-fidelity reconstruction on par with dense-view photogrammetry while reducing camera requirements and computational cost. The code and model will be released.
>
---
#### [replaced 018] GeoTop: Advancing Image Classification with Geometric-Topological Analysis
- **分类: cs.CV; cs.LG; eess.IV; math.GT**

- **链接: [https://arxiv.org/pdf/2311.16157](https://arxiv.org/pdf/2311.16157)**

> **作者:** Mariem Abaach; Ian Morilla
>
> **备注:** 37 pages, 6 figures
>
> **摘要:** A fundamental challenge in diagnostic imaging is the phenomenon of topological equivalence, where benign and malignant structures share global topology but differ in critical geometric detail, leading to diagnostic errors in both conventional and deep learning models. We introduce GeoTop, a mathematically principled framework that unifies Topological Data Analysis (TDA) and Lipschitz-Killing Curvatures (LKCs) to resolve this ambiguity. Unlike hybrid deep learning approaches, GeoTop provides intrinsic interpretability by fusing the capacity of persistent homology to identify robust topological signatures with the precision of LKCs in quantifying local geometric features such as boundary complexity and surface regularity. The framework's clinical utility is demonstrated through its application to skin lesion classification, where it achieves a consistent accuracy improvement of 3.6% and reduces false positives and negatives by 15-18% compared to conventional single-modality methods. Crucially, GeoTop directly addresses the problem of topological equivalence by incorporating geometric differentiators, providing both theoretical guarantees (via a formal lemma) and empirical validation via controlled benchmarks. Beyond its predictive performance, GeoTop offers inherent mathematical interpretability through persistence diagrams and curvature-based descriptors, computational efficiency for large datasets (processing 224x224 pixel images in less or equal 0.5 s), and demonstrated generalisability to molecular-level data. By unifying topological invariance with geometric sensitivity, GeoTop provides a principled, interpretable solution for advanced shape discrimination in diagnostic imaging.
>
---
#### [replaced 019] ProSMA-UNet: Decoder Conditioning for Proximal-Sparse Skip Feature Selection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03187](https://arxiv.org/pdf/2603.03187)**

> **作者:** Chun-Wun Cheng; Yanqi Cheng; Peiyuan Jing; Guang Yang; Javier A. Montoya-Zegarra; Carola-Bibiane Schönlieb; Angelica I. Aviles-Rivero
>
> **摘要:** Medical image segmentation commonly relies on U-shaped encoder-decoder architectures such as U-Net, where skip connections preserve fine spatial detail by injecting high-resolution encoder features into the decoder. However, these skip pathways also propagate low-level textures, background clutter, and acquisition noise, allowing irrelevant information to bypass deeper semantic filtering -- an issue that is particularly detrimental in low-contrast clinical imaging. Although attention gates have been introduced to address this limitation, they typically produce dense sigmoid masks that softly reweight features rather than explicitly removing irrelevant activations. We propose ProSMA-UNet (Proximal-Sparse Multi-Scale Attention U-Net), which reformulates skip gating as a decoder-conditioned sparse feature selection problem. ProSMA constructs a multi-scale compatibility field using lightweight depthwise dilated convolutions to capture relevance across local and contextual scales, then enforces explicit sparsity via an $\ell_1$ proximal operator with learnable per-channel thresholds, yielding a closed-form soft-thresholding gate that can remove noisy responses. To further suppress semantically irrelevant channels, ProSMA incorporates decoder-conditioned channel gating driven by global decoder context. Extensive experiments on challenging 2D and 3D benchmarks demonstrate state-of-the-art performance, with particularly large gains ($\approx20$\%) on difficult 3D segmentation tasks. Project page: this https URL
>
---
#### [replaced 020] Catch Me If You Can Describe Me: Open-Vocabulary Camouflaged Instance Segmentation with Diffusion
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于开放词汇伪装实例分割任务，旨在解决伪装目标与背景难以区分的问题。通过扩散模型学习多尺度文本视觉特征，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2312.17505](https://arxiv.org/pdf/2312.17505)**

> **作者:** Tuan-Anh Vu; Duc Thanh Nguyen; Qing Guo; Nhat Chung; Binh-Son Hua; Ivor W. Tsang; Sai-Kit Yeung
>
> **备注:** Accepted to IJCV 2026
>
> **摘要:** Text-to-image diffusion techniques have shown exceptional capabilities in producing high-quality, dense visual predictions from open-vocabulary text. This indicates a strong correlation between visual and textual domains in open concepts and that diffusion-based text-to-image models can capture rich and diverse information for computer vision tasks. However, we found that those advantages do not hold for learning of features of camouflaged individuals because of the significant blending between their visual boundaries and their surroundings. In this paper, while leveraging the benefits of diffusion-based techniques and text-image models in open-vocabulary settings, we aim to address a challenging problem in computer vision: open-vocabulary camouflaged instance segmentation (OVCIS). Specifically, we propose a method built upon state-of-the-art diffusion empowered by open-vocabulary to learn multi-scale textual-visual features for camouflaged object representation learning. Such cross-domain representations are desirable in segmenting camouflaged objects where visual cues subtly distinguish the objects from the background, and in segmenting novel object classes which are not seen in training. To enable such powerful representations, we devise complementary modules to effectively fuse cross-domain features, and to engage relevant features towards respective foreground objects. We validate and compare our method with existing ones on several benchmark datasets of camouflaged and generic open-vocabulary instance segmentation. The experimental results confirm the advances of our method over existing ones. We believe that our proposed method would open a new avenue for handling camouflages such as computer vision-based surveillance systems, wildlife monitoring, and military reconnaissance.
>
---
#### [replaced 021] Automatic Map Density Selection for Locally-Performant Visual Place Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21473](https://arxiv.org/pdf/2602.21473)**

> **作者:** Somayeh Hussaini; Tobias Fischer; Michael Milford
>
> **备注:** Under Review
>
> **摘要:** A key challenge in translating Visual Place Recognition (VPR) from the lab to long-term deployment is ensuring a priori that a system can meet user-specified performance requirements across different parts of an environment, rather than just on average globally. A critical mechanism for controlling local VPR performance is the density of the reference mapping database, yet this factor is largely neglected in existing work, where benchmark datasets with fixed, engineering-driven (sensors, storage, GPS frequency) sampling densities are typically used. In this paper, we propose a dynamic VPR mapping approach that uses pairs of reference traverses from the target environment to automatically select an appropriate map density to satisfy two user-defined requirements: (1) a target Local Recall@1 level, and (2) the proportion of the operational environment over which this requirement must be met or exceeded, which we term the Recall Achievement Rate (RAR). Our approach is based on the hypothesis that match patterns between multiple reference traverses, evaluated across different map densities, can be modelled to predict the density required to meet these performance targets on unseen deployment data. Through extensive experiments across multiple VPR methods and the Nordland and Oxford RobotCar benchmarks, we show that our system consistently achieves or exceeds the specified local recall level over at least the user-specified proportion of the environment. Comparisons with alternative baselines demonstrate that our approach reliably selects the correct operating point in map density, avoiding unnecessary over-densification. Finally, ablation studies and analysis evaluate sensitivity to reference map choice and local space definitions, and reveal that conventional global Recall@1 is a poor predictor of the often more operationally meaningful RAR metric.
>
---
#### [replaced 022] Toward Early Quality Assessment of Text-to-Image Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.02829](https://arxiv.org/pdf/2603.02829)**

> **作者:** Huanlei Guo; Hongxin Wei; Bingyi Jing
>
> **摘要:** Recent text-to-image (T2I) diffusion and flow-matching models can produce highly realistic images from natural language prompts. In practical scenarios, T2I systems are often run in a ``generate--then--select'' mode: many seeds are sampled and only a few images are kept for use. However, this pipeline is highly resource-intensive since each candidate requires tens to hundreds of denoising steps, and evaluation metrics such as CLIPScore and ImageReward are post-hoc. In this work, we address this inefficiency by introducing Probe-Select, a plug-in module that enables efficient evaluation of image quality within the generation process. We observe that certain intermediate denoiser activations, even at early timesteps, encode a stable coarse structure, object layout and spatial arrangement--that strongly correlates with final image fidelity. Probe-Select exploits this property by predicting final quality scores directly from early activations, allowing unpromising seeds to be terminated early. Across diffusion and flow-matching backbones, our experiments show that early evaluation at only 20\% of the trajectory accurately ranks candidate seeds and enables selective continuation. This strategy reduces sampling cost by over 60\% while improving the quality of the retained images, demonstrating that early structural signals can effectively guide selective generation without altering the underlying generative model. Code is available at this https URL.
>
---
#### [replaced 023] FINE: Factorizing Knowledge for Initialization of Variable-sized Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.19289](https://arxiv.org/pdf/2409.19289)**

> **作者:** Yucheng Xie; Fu Feng; Ruixiao Shi; Jianlu Shen; Jing Wang; Yong Rui; Xin Geng
>
> **摘要:** The training of diffusion models is computationally intensive, making effective pre-training essential. However, real-world deployments often demand models of variable sizes due to diverse memory and computational constraints, posing challenges when corresponding pre-trained versions are unavailable. To address this, we propose FINE, a novel pre-training method whose resulting model can flexibly factorize its knowledge into fundamental components, termed learngenes, enabling direct initialization of models of various sizes and eliminating the need for repeated pre-training. Rather than optimizing a conventional full-parameter model, FINE represents each layer's weights as the product of $U_{\star}$, $\Sigma_{\star}^{(l)}$, and $V_{\star}^\top$, where $U_{\star}$ and $V_{\star}$ serve as size-agnostic learngenes shared across layers, while $\Sigma_{\star}^{(l)}$ remains layer-specific. By jointly training these components, FINE forms a decomposable and transferable knowledge structure that allows efficient initialization through flexible recombination of learngenes, requiring only light retraining of $\Sigma_{\star}^{(l)}$ on limited data. Extensive experiments demonstrate the efficiency of FINE, achieving state-of-the-art performance in initializing variable-sized models across diverse resource-constrained deployments. Furthermore, models initialized by FINE effectively adapt to diverse tasks, showcasing the task-agnostic versatility of learngenes.
>
---
#### [replaced 024] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种无需噪声和条件的视觉到动作策略框架，解决视觉与动作对齐难题，提升推理速度与性能。**

- **链接: [https://arxiv.org/pdf/2507.13231](https://arxiv.org/pdf/2507.13231)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Conventional flow matching and diffusion-based policies sample via iterative denoising from standard noise distributions (e.g., Gaussian), and require conditioning modules to repeatedly incorporate visual information during the generative process, incurring substantial time and memory overhead. To reduce the complexity, we develop VITA, VIsion-To-Action policy, a noise-free and conditioning-free flow matching policy learning framework that directly flows from visual representations to latent actions. Since the source of the flow is visually grounded, VITA eliminates the need for visual conditioning during generation. As expected, bridging vision and action is challenging, because actions are lower-dimensional, less structured, and sparser than visual representations; moreover, flow matching requires the source and target to have the same dimensionality. To overcome this, we introduce an action autoencoder that maps raw actions into a structured latent space aligned with visual latents, trained jointly with flow matching. To further prevent latent action space collapse during end-to-end training, we propose flow latent decoding, which anchors the latent generation process by backpropagating the action reconstruction loss through the flow matching ODE (ordinary differential equation) solving steps. We evaluate VITA on 9 simulation and 5 real-world tasks from ALOHA and Robomimic. VITA achieves 1.5x-2x faster inference compared to conventional methods with conditioning modules, while outperforming or matching state-of-the-art policies. Project page: this https URL.
>
---
#### [replaced 025] Implicit U-KAN2.0: Dynamic, Efficient and Interpretable Medical Image Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.03141](https://arxiv.org/pdf/2503.03141)**

> **作者:** Chun-Wun Cheng; Yining Zhao; Yanqi Cheng; Javier A. Montoya-Zegarra; Carola-Bibiane Schönlieb; Angelica I Aviles-Rivero
>
> **备注:** Accepted in MICCAI 2025
>
> **摘要:** Image segmentation is a fundamental task in both image analysis and medical applications. State-of-the-art methods predominantly rely on encoder-decoder architectures with a U-shaped design, commonly referred to as U-Net. Recent advancements integrating transformers and MLPs improve performance but still face key limitations, such as poor interpretability, difficulty handling intrinsic noise, and constrained expressiveness due to discrete layer structures, often lacking a solid theoretical this http URL this work, we introduce Implicit U-KAN 2.0, a novel U-Net variant that adopts a two-phase encoder-decoder structure. In the SONO phase, we use a second-order neural ordinary differential equation (NODEs), called the SONO block, for a more efficient, expressive, and theoretically grounded modeling approach. In the SONO-MultiKAN phase, we integrate the second-order NODEs and MultiKAN layer as the core computational block to enhance interpretability and representation power. Our contributions are threefold. First, U-KAN 2.0 is an implicit deep neural network incorporating MultiKAN and second order NODEs, improving interpretability and performance while reducing computational costs. Second, we provide a theoretical analysis demonstrating that the approximation ability of the MultiKAN block is independent of the input dimension. Third, we conduct extensive experiments on a variety of 2D and a single 3D dataset, demonstrating that our model consistently outperforms existing segmentation networks. Project Website: this https URL
>
---
#### [replaced 026] Classification of Histopathology Slides with Persistent Homology Convolutions
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14378](https://arxiv.org/pdf/2507.14378)**

> **作者:** Shrunal Pothagoni; Benjamin Schweinhart
>
> **备注:** Reformatted citations and other minor adjustments
>
> **摘要:** Convolutional neural networks (CNNs) are a standard tool for computer vision tasks such as image classification. However, typical model architectures may result in the loss of topological information. In specific domains such as histopathology, topology is an important descriptor that can be used to distinguish between disease-indicating tissue by analyzing the shape characteristics of cells. Current literature suggests that reintroducing topological information using persistent homology can improve medical diagnostics; however, previous methods utilize global topological summaries which do not contain information about the locality of topological features. To address this gap, we present a novel method that generates local persistent homology-based data using a modified version of the convolution operator called \textit{Persistent Homology Convolutions}. This method captures information about the locality and translation equivariance of topological features. We perform a comparative study using various representations of histopathology slides and find that models trained with persistent homology convolutions outperform conventionally trained models and are less sensitive to hyperparameters. These results indicate that persistent homology convolutions extract meaningful geometric information from the histopathology slides.
>
---
#### [replaced 027] Category-Level Object Shape and Pose Estimation in Less Than a Millisecond
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于物体形状与位姿估计任务，解决如何快速准确估计物体形状和位置的问题。通过学习前端检测关键点，结合优化方法实现高效求解。**

- **链接: [https://arxiv.org/pdf/2509.18979](https://arxiv.org/pdf/2509.18979)**

> **作者:** Lorenzo Shaikewitz; Tim Nguyen; Luca Carlone
>
> **备注:** Accepted to ICRA 2026. This version contains appendices
>
> **摘要:** Object shape and pose estimation is a foundational robotics problem, supporting tasks from manipulation to scene understanding and navigation. We present a fast local solver for shape and pose estimation which requires only category-level object priors and admits an efficient certificate of global optimality. Given an RGB-D image of an object, we use a learned front-end to detect sparse, category-level semantic keypoints on the target object. We represent the target object's unknown shape using a linear active shape model and pose a maximum a posteriori optimization problem to solve for position, orientation, and shape simultaneously. Expressed in unit quaternions, this problem admits first-order optimality conditions in the form of an eigenvalue problem with eigenvector nonlinearities. Our primary contribution is to solve this problem efficiently with self-consistent field iteration, which only requires computing a 4-by-4 matrix and finding its minimum eigenvalue-vector pair at each iterate. Solving a linear system for the corresponding Lagrange multipliers gives a simple global optimality certificate. One iteration of our solver runs in about 100 microseconds, enabling fast outlier rejection. We test our method on synthetic data and a variety of real-world settings, including two public datasets and a drone tracking scenario. Code is released at this https URL.
>
---
#### [replaced 028] Human-Object Interaction via Automatically Designed VLM-Guided Motion Policy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18349](https://arxiv.org/pdf/2503.18349)**

> **作者:** Zekai Deng; Ye Shi; Kaiyang Ji; Lan Xu; Shaoli Huang; Jingya Wang
>
> **备注:** iclr camera ready
>
> **摘要:** Human-object interaction (HOI) synthesis is crucial for applications in animation, simulation, and robotics. However, existing approaches either rely on expensive motion capture data or require manual reward engineering, limiting their scalability and generalizability. In this work, we introduce the first unified physics-based HOI framework that leverages Vision-Language Models (VLMs) to enable long-horizon interactions with diverse object types, including static, dynamic, and articulated objects. We introduce VLM-Guided Relative Movement Dynamics (RMD), a fine-grained spatio-temporal bipartite representation that automatically constructs goal states and reward functions for reinforcement learning. By encoding structured relationships between human and object parts, RMD enables VLMs to generate semantically grounded, interaction-aware motion guidance without manual reward tuning. To support our methodology, we present Interplay, a novel dataset with thousands of long-horizon static and dynamic interaction plans. Extensive experiments demonstrate that our framework outperforms existing methods in synthesizing natural, human-like motions across both simple single-task and complex multi-task scenarios. For more details, please refer to our project webpage: this https URL.
>
---
#### [replaced 029] Beyond Dominant Patches: Spatial Credit Redistribution For Grounded Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.22469](https://arxiv.org/pdf/2602.22469)**

> **作者:** Niamul Hassan Samin; Md Arifur Rahman; Abdullah Ibne Hanif Arean; Juena Ahmed Noshin; Md Ashikur Rahman
>
> **摘要:** Vision-Language Models (VLMs) often hallucinate objects that are not present in the input image. We identify a contributing cause of this behavior, which we term spatial credit collapse: in early transformer layers, hidden-state activation concentrates on a small number of visual patches, suppressing surrounding contextual evidence and increasing reliance on language priors. Across seven models we observe a strong correlation between visual attention entropy and hallucination rate (r = -0.65, p < 0.001), suggesting that reduced spatial credit diversity contributes to hallucination. To address this issue we propose Spatial Credit Redistribution (SCR), a training-free inference-time method. SCR uses a lightweight two-pass procedure. A diagnostic pass identifies the top-K high-attention source patches and their spatial neighbors. A redistribution pass then scales each source by 1/lambda (~0.91) and injects a (lambda - 1) weighted copy of its hidden state into neighboring patches, restoring suppressed visual context without modifying model weights. Because the diagnostic pass is performed once per image and reused across the output sequence, the added latency is negligible (<0.5 ms per token for 100-token responses). We evaluate SCR across seven model configurations from four VLM families (Chameleon, LLaVA-1.5, Qwen-VL/Qwen2-VL, and InternVL2) on five benchmarks: POPE, CHAIR, MME, HallusionBench, and AMBER. SCR reduces POPE-Adversarial hallucination by 4.6-6.0 percentage points and CHAIR-s by 41-51 percent while preserving caption quality (CIDEr drop <=0.8). Compared with prior inference-time methods including OPERA, VCD, OA-VCD, DoLa, VLI, SID, and CRoPS, SCR achieves a better trade-off between hallucination reduction, generation quality, and latency.
>
---
#### [replaced 030] Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08269](https://arxiv.org/pdf/2511.08269)**

> **作者:** Nan Bao; Yifan Zhao; Lin Zhu; Jia Li
>
> **备注:** Accepted to NeurIPS 2025; code and datasets available at this https URL
>
> **摘要:** Semantic segmentation has achieved great success in ideal conditions. However, when facing extreme conditions (e.g., insufficient light, fierce camera motion), most existing methods suffer from significant information loss of RGB, severely damaging segmentation results. Several researches exploit the high-speed and high-dynamic event modality as a complement, but event and RGB are naturally heterogeneous, which leads to feature-level mismatch and inferior optimization of existing multi-modality methods. Different from these researches, we delve into the edge secret of both modalities for resilient fusion and propose a novel Edge-awareness Semantic Concordance framework to unify the multi-modality heterogeneous features with latent edge cues. In this framework, we first propose Edge-awareness Latent Re-coding, which obtains uncertainty indicators while realigning event-RGB features into unified semantic space guided by re-coded distribution, and transfers event-RGB distributions into re-coded features by utilizing a pre-established edge dictionary as clues. We then propose Re-coded Consolidation and Uncertainty Optimization, which utilize re-coded edge features and uncertainty indicators to solve the heterogeneous event-RGB fusion issues under extreme conditions. We establish two synthetic and one real-world event-RGB semantic segmentation datasets for extreme scenario comparisons. Experimental results show that our method outperforms the state-of-the-art by a 2.55% mIoU on our proposed DERS-XS, and possesses superior resilience under spatial occlusion. Our code and datasets are publicly available at this https URL.
>
---
#### [replaced 031] Token Adaptation via Side Graph Convolution for Efficient Fine-tuning of 3D Point Cloud Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.14142](https://arxiv.org/pdf/2502.14142)**

> **作者:** Takahiko Furuya
>
> **备注:** Accepted to the journal of Machine Vision and Applications
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) of pre-trained 3D point cloud Transformers has emerged as a promising technique for 3D point cloud analysis. While existing PEFT methods attempt to minimize the number of tunable parameters, they often suffer from high temporal and spatial computational costs during fine-tuning. This paper proposes a novel PEFT algorithm called Side Token Adaptation on a neighborhood Graph (STAG) to achieve superior temporal and spatial efficiency. STAG employs a graph convolutional side network operating in parallel with a frozen backbone Transformer to adapt tokens to downstream tasks. Through efficient graph convolution, parameter sharing, and reduced gradient computation, STAG significantly reduces both temporal and spatial costs for fine-tuning. We also present Point Cloud Classification 13 (PCC13), a new benchmark comprising diverse publicly available 3D point cloud datasets to facilitate comprehensive evaluation. Extensive experiments using multiple pre-trained models and PCC13 demonstrates the effectiveness of STAG. Specifically, STAG maintains classification accuracy comparable to existing methods while reducing tunable parameters to only 0.43M and achieving significant reductions in both computation time and memory consumption for fine-tuning. Code and benchmark will be available at: this https URL.
>
---
#### [replaced 032] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
- **分类: cs.CV; cs.MA**

- **链接: [https://arxiv.org/pdf/2511.19524](https://arxiv.org/pdf/2511.19524)**

> **作者:** Boyu Chen; Zikang Wang; Zhengrong Yue; Kainan Yan; Chenyun Yu; Yi Huang; Zijun Liu; Yafei Wen; Xiaoxin Chen; Yang Liu; Peng Li; Yali Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** By leveraging tool-augmented Multimodal Large Language Models (MLLMs), multi-agent frameworks are driving progress in video understanding. However, most of them adopt static and non-learnable tool invocation mechanisms, which limit the discovery of diverse clues essential for robust perception and reasoning regarding temporally or spatially complex videos. To address this challenge, we propose a novel Multi-agent system for video understanding, namely VideoChat-M1. Instead of using a single or fixed policy, VideoChat-M1 adopts a distinct Collaborative Policy Planning (CPP) paradigm with multiple policy agents, which comprises three key processes. (1) Policy Generation: Each agent generates its unique tool invocation policy tailored to the user's query; (2) Policy Execution: Each agent sequentially invokes relevant tools to execute its policy and explore the video content; (3) Policy Communication: During the intermediate stages of policy execution, agents interact with one another to update their respective policies. Through this collaborative framework, all agents work in tandem, dynamically refining their preferred policies based on contextual insights from peers to effectively respond to the user's query. Moreover, we equip our CPP paradigm with a concise Multi-Agent Reinforcement Learning (MARL) method. Consequently, the team of policy agents can be jointly optimized to enhance VideoChat-M1's performance, guided by both the final answer reward and intermediate collaborative process feedback. Extensive experiments demonstrate that VideoChat-M1 achieves SOTA performance across eight benchmarks spanning four tasks. Notably, on LongVideoBench, our method outperforms the SOTA model Gemini 2.5 pro by 3.6% and GPT-4o by 15.6%.
>
---
#### [replaced 033] Composition-Grounded Data Synthesis for Visual Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉推理任务，旨在解决多模态大模型在缺乏标注数据的领域中推理能力不足的问题。通过COGS框架，从少量种子问题生成大量合成数据，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2510.15040](https://arxiv.org/pdf/2510.15040)**

> **作者:** Xinyi Gu; Jiayuan Mao; Zhang-Wei Hong; Zhuoran Yu; Pengyuan Li; Dhiraj Joshi; Rogerio Feris; Zexue He
>
> **备注:** ICLR2026 camera-ready version. Project page: this https URL
>
> **摘要:** Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded data Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning factors, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages.
>
---
#### [replaced 034] Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2404.00636](https://arxiv.org/pdf/2404.00636)**

> **作者:** Taekyung Ki; Dongchan Min; Gyeongsu Chae
>
> **备注:** ECCV 2024. Project page: this https URL
>
> **摘要:** In this paper, we present Export3D, a one-shot 3D-aware portrait animation method that is able to control the facial expression and camera view of a given portrait image. To achieve this, we introduce a tri-plane generator with an effective expression conditioning method, which directly generates a tri-plane of 3D prior by transferring the expression parameter of 3DMM into the source image. The tri-plane is then decoded into the image of different view through a differentiable volume rendering. Existing portrait animation methods heavily rely on image warping to transfer the expression in the motion space, challenging on disentanglement of appearance and expression. In contrast, we propose a contrastive pre-training framework for appearance-free expression parameter, eliminating undesirable appearance swap when transferring a cross-identity expression. Extensive experiments show that our pre-training framework can learn the appearance-free expression representation hidden in 3DMM, and our model can generate 3D-aware expression controllable portrait images without appearance swap in the cross-identity manner.
>
---
#### [replaced 035] Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.21574](https://arxiv.org/pdf/2505.21574)**

> **作者:** Dang Nguyen; Jiping Li; Jinghao Zheng; Baharan Mirzasoleiman
>
> **摘要:** Synthetically augmenting training datasets with diffusion models has become an effective strategy for improving the generalization of image classifiers. However, existing approaches typically increase dataset size by 10-30x and struggle to ensure generation diversity, leading to substantial computational overhead. In this work, we introduce TADA (TArgeted Diffusion Augmentation), a principled framework that selectively augments examples that are not learned early in training using faithful synthetic images that preserve semantic features while varying noise. We show that augmenting only this targeted subset consistently outperforms augmenting the entire dataset. Through theoretical analysis on a two-layer CNN, we prove that TADA improves generalization by promoting homogeneity in feature learning speed without amplifying noise. Extensive experiments demonstrate that by augmenting only 30-40% of the training data, TADA improves generalization by up to 2.8% across diverse architectures including ResNet, ViT, ConvNeXt, and Swin Transformer on CIFAR-10/100, TinyImageNet, and ImageNet, using optimizers such as SGD and SAM. Notably, TADA combined with SGD outperforms the state-of-the-art optimizer SAM on CIFAR-100 and TinyImageNet. Furthermore, TADA shows promising improvements on object detection benchmarks, demonstrating its applicability beyond image classification. Our code is available at this https URL.
>
---
#### [replaced 036] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08417](https://arxiv.org/pdf/2511.08417)**

> **作者:** Xiyuan Wei; Chih-Jen Lin; Tianbao Yang
>
> **备注:** Accepted to 40th International Conference on Learning Representations. 32 pages, 5 figures
>
> **摘要:** Accurately estimating the normalization term (also known as the partition function) in the contrastive loss is a central challenge for training Contrastive Language-Image Pre-training (CLIP) models. Conventional methods rely on large batches for approximation, demanding substantial computational resources. To mitigate this issue, prior works introduced per-sample normalizer estimators, which are updated at each epoch in a blockwise coordinate manner to keep track of updated encoders. However, this scheme incurs optimization error that scales with the ratio of dataset size to batch size, limiting effectiveness for large datasets or small batches. To overcome this limitation, we propose NeuCLIP, a novel and elegant optimization framework based on two key ideas: (i) $\textbf{reformulating}$ the contrastive loss for each sample $\textbf{via convex analysis}$ into a minimization problem with an auxiliary variable representing its log-normalizer; and (ii) $\textbf{transforming}$ the resulting minimization over $n$ auxiliary variables (where $n$ is the dataset size) via $\textbf{variational analysis}$ into the minimization over a compact neural network that predicts the log-normalizers. We design an alternating optimization algorithm that jointly trains the CLIP model and the auxiliary network. By employing a tailored architecture and acceleration techniques for the auxiliary network, NeuCLIP achieves more accurate normalizer estimation, leading to improved performance compared with previous methods. Extensive experiments on large-scale CLIP training, spanning datasets from millions to billions of samples, demonstrate that NeuCLIP outperforms previous methods. Code is available at this https URL.
>
---
#### [replaced 037] Scaling Laws For Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.08184](https://arxiv.org/pdf/2410.08184)**

> **作者:** Zhengyang Liang; Hao He; Ceyuan Yang; Bo Dai
>
> **摘要:** Diffusion transformers (DiT) have already achieved appealing synthesis and scaling properties in content recreation, e.g., image and video generation. However, scaling laws of DiT are less explored, which usually offer precise predictions regarding optimal model size and data requirements given a specific compute budget. Therefore, experiments across a broad range of compute budgets, from 1e17 to 6e18 FLOPs are conducted to confirm the existence of scaling laws in DiT for the first time. Concretely, the loss of pretraining DiT also follows a power-law relationship with the involved compute. Based on the scaling law, we can not only determine the optimal model size and required data but also accurately predict the text-to-image generation loss given a model with 1B parameters and a compute budget of 1e21 FLOPs. Additionally, we also demonstrate that the trend of pre-training loss matches the generation performances (e.g., FID), even across various datasets, which complements the mapping from compute to synthesis quality and thus provides a predictable benchmark that assesses model performance and data quality at a reduced cost.
>
---
#### [replaced 038] Generating Fine Details of Entity Interactions
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决对象交互描述不足的问题。通过构建数据集和改进生成方法，提升图像中对象互动的细节质量。**

- **链接: [https://arxiv.org/pdf/2504.08714](https://arxiv.org/pdf/2504.08714)**

> **作者:** Xinyi Gu; Jiayuan Mao
>
> **备注:** EMNLP 2025. Project Page: this https URL
>
> **摘要:** Recent text-to-image models excel at generating high-quality object-centric images from instructions. However, images should also encapsulate rich interactions between objects, where existing models often fall short, likely due to limited training data and benchmarks for rare interactions. This paper explores a novel application of Multimodal Large Language Models (MLLMs) to benchmark and enhance the generation of interaction-rich images. We introduce \data, an interaction-focused dataset with 1000 LLM-generated fine-grained prompts for image generation covering (1) functional and action-based interactions, (2) multi-subject interactions, and (3) compositional spatial relationships. To address interaction-rich generation challenges, we propose a decomposition-augmented refinement procedure. Our approach, \model, leverages LLMs to decompose interactions into finer-grained concepts, uses an MLLM to critique generated images, and applies targeted refinements with a partial diffusion denoising process. Automatic and human evaluations show significantly improved image quality, demonstrating the potential of enhanced inference strategies.
>
---
#### [replaced 039] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文档理解任务，旨在降低视觉语言模型的计算成本。通过轻量级令牌剪枝框架，过滤非信息背景区域，提升效率同时保持准确率。**

- **链接: [https://arxiv.org/pdf/2509.06415](https://arxiv.org/pdf/2509.06415)**

> **作者:** Jaemin Son; Sujin Choi; Inyong Yun
>
> **备注:** Accepted to ICLR 2026 Workshop MM Intelligence
>
> **摘要:** Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy.
>
---
#### [replaced 040] EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.17896](https://arxiv.org/pdf/2506.17896)**

> **作者:** Junho Park; Andrew Sangwoo Ye; Taein Kwon
>
> **备注:** Accepted by ICLR 2026. Project Page: this https URL
>
> **摘要:** Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as the necessity of an initial egocentric frame and relative camera poses during inference. To overcome these challenges, we introduce EgoWorld, a novel framework that reconstructs an egocentric view from rich exocentric observations, including point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion model to produce dense, semantically coherent egocentric images. Evaluated on four datasets (i.e., H2O, TACO, Assembly101, and Ego-Exo4D), EgoWorld achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, EgoWorld exhibits robustness on in-the-wild examples, underscoring its practical applicability. Project page is available at this https URL.
>
---
#### [replaced 041] Partial Weakly-Supervised Oriented Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.02751](https://arxiv.org/pdf/2507.02751)**

> **作者:** Mingxin Liu; Peiyuan Zhang; Yuan Liu; Wei Zhang; Yue Zhou; Ning Liao; Ziyang Gong; Junwei Luo; Zhirui Wang; Yi Yu; Xue Yang
>
> **备注:** 10 pages, 5 figures, 4 tables, source code: this https URL
>
> **摘要:** The growing demand for oriented object detection (OOD) across various domains has driven significant research in this area. However, the high cost of dataset annotation remains a major concern. Current mainstream OOD algorithms can be mainly categorized into three types: (1) fully supervised methods using complete oriented bounding box (OBB) annotations, (2) semi-supervised methods using partial OBB annotations, and (3) weakly supervised methods using weak annotations such as horizontal boxes or points. However, these algorithms inevitably increase the cost of models in terms of annotation speed or annotation cost. To address this issue, we propose: (1) the first Partial Weakly-Supervised Oriented Object Detection (PWOOD) framework based on partially weak annotations (horizontal boxes or single points), which can efficiently leverage large amounts of unlabeled data, significantly outperforming weakly supervised algorithms trained with partially weak annotations, also offers a lower cost solution; (2) Orientation-and-Scale-aware Student (OS-Student) model capable of learning orientation and scale information with only a small amount of orientation-agnostic or scale-agnostic weak annotations; and (3) Class-Agnostic Pseudo-Label Filtering strategy (CPF) to reduce the model's sensitivity to static filtering thresholds. Comprehensive experiments on DOTA-v1.0/v1.5/v2.0 and DIOR datasets demonstrate that our PWOOD framework performs comparably to, or even surpasses traditional semi-supervised algorithms. Our code will be made publicly available.
>
---
#### [replaced 042] Generative Human Geometry Distribution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.01448](https://arxiv.org/pdf/2503.01448)**

> **作者:** Xiangjun Tang; Biao Zhang; Peter Wonka
>
> **摘要:** Realistic human geometry generation is an important yet challenging task, requiring both the preservation of fine clothing details and the accurate modeling of clothing-body interactions. To tackle this challenge, we build upon Geometry distributions, a recently proposed representation that can model a single human geometry with high fidelity using a flow matching model. However, extending a single-geometry distribution to a dataset is non-trivial and inefficient for large-scale learning. To address this, we propose a new geometry distribution model by two key techniques: (1) encoding distributions as 2D feature maps rather than network parameters, and (2) using SMPL models as the domain instead of Gaussian and refining the associated flow velocity field. We then design a generative framework adopting a two staged training paradigm analogous to state-of-the-art image and 3D generative models. In the first stage, we compress geometry distributions into a latent space using a diffusion flow model; the second stage trains another flow model on this latent space. We validate our approach on two key tasks: pose-conditioned random avatar generation and avatar-consistent novel pose synthesis. Experimental results demonstrate that our method outperforms existing state-of-the-art methods, achieving a 57% improvement in geometry quality.
>
---
#### [replaced 043] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07782](https://arxiv.org/pdf/2508.07782)**

> **作者:** Saihui Hou; Chenye Wang; Wenpeng Lang; Zhengxiang Lan; Yongzhen Huang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Recent advancements in gait recognition have significantly enhanced performance by treating silhouettes as either an unordered set or an ordered sequence. However, both set-based and sequence-based approaches exhibit notable limitations. Specifically, set-based methods tend to overlook short-range temporal context for individual frames, while sequence-based methods struggle to capture long-range temporal dependencies effectively. To address these challenges, we draw inspiration from human identification and propose a new perspective that conceptualizes human gait as a composition of individualized actions. Each action is represented by a series of frames, randomly selected from a continuous segment of the sequence, which we term a snippet. Fundamentally, the collection of snippets for a given sequence enables the incorporation of multi-scale temporal context, facilitating more comprehensive gait feature learning. Moreover, we introduce a non-trivial solution for snippet-based gait recognition, focusing on Snippet Sampling and Snippet Modeling as key components. Extensive experiments on four widely-used gait datasets validate the effectiveness of our proposed approach and, more importantly, highlight the potential of gait snippets. For instance, our method achieves the rank-1 accuracy of 77.5% on Gait3D and 81.7% on GREW using a 2D convolution-based backbone.
>
---
#### [replaced 044] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14266](https://arxiv.org/pdf/2512.14266)**

> **作者:** Shreedhar Govil; Didier Stricker; Jason Rambach
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Predicting driver attention is a critical problem for developing explainable autonomous driving systems and understanding driver behavior in mixed human-autonomous vehicle traffic scenarios. Although significant progress has been made through large-scale driver attention datasets and deep learning architectures, existing works are constrained by narrow frontal field-of-view and limited driving diversity. Consequently, they fail to capture the full spatial context of driving environments, especially during lane changes, turns, and interactions involving peripheral objects such as pedestrians or cyclists. In this paper, we introduce DriverGaze360, a large-scale 360$^\circ$ field of view driver attention dataset, containing $\sim$1 million gaze-labeled frames collected from 19 human drivers, enabling comprehensive omnidirectional modeling of driver gaze behavior. Moreover, our panoramic attention prediction approach, DriverGaze360-Net, jointly learns attention maps and attended objects by employing an auxiliary semantic segmentation head. This improves spatial awareness and attention prediction across wide panoramic inputs. Extensive experiments demonstrate that DriverGaze360-Net achieves state-of-the-art attention prediction performance on multiple metrics on panoramic driving images. Dataset and method available at this https URL.
>
---
#### [replaced 045] Generalized non-exponential Gaussian splatting
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02887](https://arxiv.org/pdf/2603.02887)**

> **作者:** Sébastien Speierer; Adrian Jarabo
>
> **备注:** 13 pages, 6 figures, 4 tables
>
> **摘要:** In this work we generalize 3D Gaussian splatting (3DGS) to a wider family of physically-based alpha-blending operators. 3DGS has become the standard de-facto for radiance field rendering and reconstruction, given its flexibility and efficiency. At its core, it is based on alpha-blending sorted semitransparent primitives, which in the limit converges to the classic radiative transfer function with exponential transmittance. Inspired by recent research on non-exponential radiative transfer, we generalize the image formation model of 3DGS to non-exponential regimes. Based on this generalization, we use a quadratic transmittance to define sub-linear, linear, and super-linear versions of 3DGS, which exhibit faster-than-exponential decay. We demonstrate that these new non-exponential variants achieve similar quality than the original 3DGS but significantly reduce the number of overdraws, which result on speed-ups of up to $4\times$ in complex real-world captures, on a ray-tracing-based renderer.
>
---
#### [replaced 046] Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于OOD检测与分割任务，旨在解决多模态数据中缺乏监督信号导致的过自信预测问题。提出Feature Mixing方法，提升模型区分ID与OOD数据的能力，并构建了CARLA-OOD数据集。**

- **链接: [https://arxiv.org/pdf/2505.16985](https://arxiv.org/pdf/2505.16985)**

> **作者:** Moru Liu; Hao Dong; Jessica Kelly; Olga Fink; Mario Trapp
>
> **备注:** NeurIPS 2025
>
> **摘要:** Out-of-distribution (OOD) detection and segmentation are crucial for deploying machine learning models in safety-critical applications such as autonomous driving and robot-assisted surgery. While prior research has primarily focused on unimodal image data, real-world applications are inherently multimodal, requiring the integration of multiple modalities for improved OOD detection. A key challenge is the lack of supervision signals from unknown data, leading to overconfident predictions on OOD samples. To address this challenge, we propose Feature Mixing, an extremely simple and fast method for multimodal outlier synthesis with theoretical support, which can be further optimized to help the model better distinguish between in-distribution (ID) and OOD data. Feature Mixing is modality-agnostic and applicable to various modality combinations. Additionally, we introduce CARLA-OOD, a novel multimodal dataset for OOD segmentation, featuring synthetic OOD objects across diverse scenes and weather conditions. Extensive experiments on SemanticKITTI, nuScenes, CARLA-OOD datasets, and the MultiOOD benchmark demonstrate that Feature Mixing achieves state-of-the-art performance with a $10 \times$ to $370 \times$ speedup. Our source code and dataset will be available at this https URL.
>
---
#### [replaced 047] TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出TIGeR框架，解决机器人视觉-语言模型在几何推理中的精度不足问题，通过集成外部工具实现精确计算。**

- **链接: [https://arxiv.org/pdf/2510.07181](https://arxiv.org/pdf/2510.07181)**

> **作者:** Yi Han; Enshen Zhou; Shanyu Rong; Jingkun An; Pengwei Wang; Zhongyuan Wang; Cheng Chi; Lu Sheng; Shanghang Zhang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.
>
---
#### [replaced 048] TRACE: Task-Adaptive Reasoning and Representation Learning for Universal Multimodal Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02929](https://arxiv.org/pdf/2603.02929)**

> **作者:** Xiangzhao Hao; Shijie Wang; Tianyu Yang; Tianyue Wang; Haiyun Guo; Jinqiao Wang
>
> **摘要:** Universal Multimodal Retrieval requires unified embedding models capable of interpreting diverse user intents, ranging from simple keywords to complex compositional instructions. While Multimodal Large Language Models (MLLMs) possess strong reasoning capabilities, prevailing adaptations confine them to static encoders, underutilizing their generative potential. This encoder-only paradigm struggles with complex intents that demand logical deduction rather than superficial pattern matching. To address this, we introduce TRACE (Task-adaptive Reasoning And Compressing Embeddings). TRACE unifies generative reasoning with discriminative representation learning. It first generates a structured Chain-of-Thought (CoT) to explicitly reason about the query, and subsequently compresses this reasoning trace into a compact embedding via a dedicated token. To train this framework, we construct M-BEIR-CoT, a large-scale dataset featuring a difficulty-aware routing strategy. Experiments on the M-BEIR benchmark establish TRACE as the new state-of-the-art. Crucially, TRACE demonstrates a learned implicit routing behavior. It autonomously activates reasoning for complex queries while bypassing it for simpler ones, achieving an optimal balance between retrieval accuracy and inference throughput. Furthermore, by internalizing the deductive process, TRACE exhibits remarkable zero-shot transferability to unseen domains and novel constraints.
>
---
#### [replaced 049] QDFlow: A Python package for physics simulations of quantum dot devices
- **分类: cond-mat.mes-hall; cs.CV; cs.LG; quant-ph**

- **链接: [https://arxiv.org/pdf/2509.13298](https://arxiv.org/pdf/2509.13298)**

> **作者:** Donovan L. Buterakos; Sandesh S. Kalantre; Joshua Ziegler; Jacob M. Taylor; Justyna P. Zwolak
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Recent advances in machine learning (ML) have accelerated progress in calibrating and operating quantum dot (QD) devices. However, most ML approaches rely on access to large, representative datasets designed to capture the full spectrum of data quality encountered in practice, with both high- and low-quality data for training, benchmarking, and validation, with labels capturing key features of the device state. Collating such datasets experimentally is challenging due to limited data availability, slow measurement bandwidths, and the labor-intensive nature of labeling. QDFlow is an open-source physics simulator for multi-QD arrays that generates realistic synthetic data with ground-truth labels. QDFlow combines a self-consistent Thomas-Fermi solver, a dynamic capacitance model, and flexible noise modules to simulate charge stability diagrams and ray-based data that closely resemble experimental results. With an extensive set of parameters that can be varied and customizable noise models, QDFlow supports the creation of large, diverse datasets for ML development, benchmarking, and quantum device research.}}
>
---
#### [replaced 050] UrbanAlign: Post-hoc Semantic Calibration for VLM-Human Preference Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19442](https://arxiv.org/pdf/2602.19442)**

> **作者:** Yecheng Zhang; Rong Zhao; Zhizhou Sha; Yong Li; Lei Wang; Ce Hou; Wen Ji; Hao Huang; Yunshan Wan; Jian Yu; Junhao Xia; Yuru Zhang; Chunlei Shi
>
> **备注:** 26 pages
>
> **摘要:** Aligning vision-language model (VLM) outputs with human preferences in domain-specific tasks typically requires fine-tuning or reinforcement learning, both of which demand labelled data and GPU compute. We show that for subjective perception tasks, this alignment can be achieved without any model training: VLMs are already strong concept extractors but poor decision calibrators, and the gap can be closed externally. We propose a training-free post-hoc concept-bottleneck pipeline consisting of three tightly coupled stages: concept mining, multi-agent structured scoring, and geometric calibration, unified by an end-to-end dimension optimization loop. Interpretable evaluation dimensions are mined from a handful of human annotations; an Observer-Debater-Judge chain extracts robust continuous concept scores from a frozen VLM; and locally-weighted ridge regression on a hybrid visual-semantic manifold calibrates these scores against human ratings. Applied to urban perception as UrbanAlign, the framework achieves 72.2% accuracy ($\kappa=0.45$) on Place Pulse 2.0 across six categories, outperforming the best supervised baseline by +15.1 pp and uncalibrated VLM scoring by +16.3 pp, with full dimension-level interpretability and zero model-weight modification.
>
---
#### [replaced 051] A dataset of high-resolution plantar pressures for gait analysis across varying footwear and walking speeds
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.17244](https://arxiv.org/pdf/2502.17244)**

> **作者:** Robyn Larracy; Angkoon Phinyomark; Ala Salehi; Eve MacDonald; Saeed Kazemi; Shikder Shafiul Bashar; Aaron Tabor; Erik Scheme
>
> **摘要:** Gait refers to the patterns of limb movement generated during walking, which are unique to each individual due to both physical and behavioral traits. Walking patterns have been widely studied in biometrics, biomechanics, sports, and rehabilitation. While traditional methods rely on video and motion capture, advances in plantar pressure sensing technology now offer deeper insights into gait. However, underfoot pressures during walking remain underexplored due to the lack of large, publicly accessible datasets. To address this, we introduce the UNB StepUP-P150 dataset: a footStep database for gait analysis and recognition using Underfoot Pressure, including data from 150 individuals. This dataset comprises high-resolution plantar pressure data (4 sensors per cm-squared) collected using a 1.2m by 3.6m pressure-sensing walkway. It contains over 200,000 footsteps from participants walking with various speeds (preferred, slow-to-stop, fast, and slow) and footwear conditions (barefoot, standard shoes, and two personal shoes), supporting advancements in biometric gait recognition and presenting new research opportunities in biomechanics and deep learning. UNB StepUP-P150 establishes a new benchmark for plantar pressure-based gait analysis and recognition.
>
---
#### [replaced 052] Weakly Supervised Concept Learning with Class-Level Priors for Interpretable Medical Diagnosis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.01131](https://arxiv.org/pdf/2511.01131)**

> **作者:** Md Nahiduzzaman; Steven Korevaar; Alireza Bab-Hadiashar; Ruwan Tennakoon
>
> **备注:** Accepted to IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Human-interpretable predictions are essential for deploying AI in medical imaging, yet most interpretable-by-design (IBD) frameworks require concept annotations for training data, which are costly and impractical to obtain in clinical contexts. Recent attempts to bypass annotation, such as zero-shot vision-language models or concept-generation frameworks, struggle to capture domain-specific medical features, leading to poor reliability. In this paper, we propose a novel Prior-guided Concept Predictor (PCP), a weakly supervised framework that enables concept answer prediction without explicit supervision or reliance on language models. PCP leverages class-level concept priors as weak supervision and incorporates a refinement mechanism with KL divergence and entropy regularization to align predictions with clinical reasoning. Experiments on PH2 (dermoscopy) and WBCatt (hematology) show that PCP improves concept-level F1-score by over 33% compared to zero-shot baselines, while delivering competitive classification performance on four medical datasets (PH2, WBCatt, HAM10000, and CXR4) relative to fully supervised concept bottleneck models (CBMs) and V-IP.
>
---
#### [replaced 053] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17807](https://arxiv.org/pdf/2602.17807)**

> **作者:** Narges Norouzi; Idil Esen Zulfikar; Niccolò Cavagnero; Tommie Kerssies; Bastian Leibe; Gijs Dubbelman; Daan de Geus
>
> **备注:** CVPR 2026. Code: this https URL
>
> **摘要:** Existing online video segmentation models typically combine a per-frame segmenter with complex specialized tracking modules. While effective, these modules introduce significant architectural complexity and computational overhead. Recent studies suggest that plain Vision Transformer (ViT) encoders, when scaled with sufficient capacity and large-scale pre-training, can conduct accurate image segmentation without requiring specialized modules. Motivated by this observation, we propose the Video Encoder-only Mask Transformer (VidEoMT), a simple encoder-only video segmentation model that eliminates the need for dedicated tracking modules. To enable temporal modeling in an encoder-only ViT, VidEoMT introduces a lightweight query propagation mechanism that carries information across frames by reusing queries from the previous frame. To balance this with adaptability to new content, it employs a query fusion strategy that combines the propagated queries with a set of temporally-agnostic learned queries. As a result, VidEoMT attains the benefits of a tracker without added complexity, achieving competitive accuracy while being 5x-10x faster, running at up to 160 FPS with a ViT-L backbone. Code: this https URL
>
---
#### [replaced 054] 3D Wavelet-Based Structural Priors for Controlled Diffusion in Whole-Body Low-Dose PET Denoising
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.07093](https://arxiv.org/pdf/2601.07093)**

> **作者:** Peiyuan Jing; Yue Yang; Chun-Wun Cheng; Zhenxuan Zhang; Liutao Yang; Thiago V. Lima; Klaus Strobel; Antoine Leimgruber; Angelica Aviles-Rivero; Guang Yang; Javier A. Montoya-Zegarra
>
> **备注:** 10 pages
>
> **摘要:** Low-dose Positron Emission Tomography (PET) imaging reduces patient radiation exposure but suffers from increased noise that degrades image quality and diagnostic reliability. Although diffusion models have demonstrated strong denoising capability, their stochastic nature makes it challenging to enforce anatomically consistent structures, particularly in low signal-to-noise regimes and volumetric whole-body imaging. We propose Wavelet-Conditioned ControlNet (WCC-Net), a fully 3D diffusion-based framework that introduces explicit frequency-domain structural priors via wavelet representations to guide volumetric PET denoising. By injecting wavelet-based structural guidance into a frozen pretrained diffusion backbone through a lightweight control branch, WCC-Net decouples anatomical structure from noise while preserving generative expressiveness and 3D structural continuity. Extensive experiments demonstrate that WCC-Net consistently outperforms CNN-, GAN-, and diffusion-based baselines. On the internal 1/20-dose test set, WCC-Net improves PSNR by +1.21 dB and SSIM by +0.008 over a strong diffusion baseline, while reducing structural distortion (GMSD) and intensity error (NMAE). Moreover, WCC-Net generalizes robustly to unseen dose levels (1/50 and 1/4), achieving superior quantitative performance and improved volumetric anatomical consistency.
>
---
#### [replaced 055] Intelligent Diagnosis Using Dual-Branch Attention Network for Rare Thyroid Carcinoma Recognition with Ultrasound Imaging
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02211](https://arxiv.org/pdf/2505.02211)**

> **作者:** Peiqi Li; Yincheng Gao; Renxing Li; Haojie Yang; Yunyun Liu; Boji Liu; Jiahui Ni; Ying Zhang; Yulu Wu; Xiaowei Fang; Lehang Guo; Liping Sun; Jiangang Chen
>
> **摘要:** Heterogeneous morphological features and data imbalance pose significant challenges in rare thyroid carcinoma classification using ultrasound imaging. To address this issue, we propose a novel multitask learning framework, Channel-Spatial Attention Synergy Network (CSASN), which integrates a dual-branch feature extractor - combining EfficientNet for local spatial encoding and ViT for global semantic modeling, with a cascaded channel-spatial attention refinement module. A residual multiscale classifier and dynamically weighted loss function further enhance classification stability and accuracy. Trained on a multicenter dataset comprising more than 2000 patients from four clinical institutions, our framework leverages a residual multiscale classifier and dynamically weighted loss function to enhance classification stability and accuracy. Extensive ablation studies demonstrate that each module contributes significantly to model performance, particularly in recognizing rare subtypes such as FTC and MTC carcinomas. Experimental results show that CSASN outperforms existing single-stream CNN or Transformer-based models, achieving a superior balance between precision and recall under class-imbalanced conditions. This framework provides a promising strategy for AI-assisted thyroid cancer diagnosis.
>
---
#### [replaced 056] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.08492](https://arxiv.org/pdf/2507.08492)**

> **作者:** Heng Li; Xiangping Wu; Qingcai Chen
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Document image dewarping remains a challenging task in the deep learning era. While existing methods have improved by leveraging text line awareness, they typically focus only on a single horizontal dimension. In this paper, we propose a fine-grained deformation perception model that focuses on Dual Dimensions of document horizontal-vertical-lines to improve document Dewarping called D2Dewarp. It can perceive distortion trends in different directions across document details. To combine the horizontal and vertical granularity features, an effective fusion module based on X and Y coordinate is designed to facilitate interaction and constraint between the two dimensions for feature complementarity. Due to the lack of annotated line features in current public dewarping datasets, we also propose an automatic fine-grained annotation method using public document texture images and automatic rendering engine to build a new large-scale distortion training dataset named DocDewarpHV. On three public Chinese and English benchmarks, both quantitative and qualitative results show that our method achieves better rectification results compared with the state-of-the-art methods. The code and dataset are available at this https URL.
>
---
#### [replaced 057] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20880](https://arxiv.org/pdf/2602.20880)**

> **作者:** Yongli Xiang; Ziming Hong; Zhaoqing Wang; Xiangyu Zhao; Bo Han; Tongliang Liu
>
> **备注:** CVPR 2026; Code is released at this https URL
>
> **摘要:** Text-to-Image (T2I) diffusion models have demonstrated significant advancements in generating high-quality images, while raising potential safety concerns regarding harmful content generation. Safety-guidance-based methods have been proposed to mitigate harmful outputs by steering generation away from harmful zones, where the zones are averaged across multiple harmful categories based on predefined keywords. However, these approaches fail to capture the complex interplay among different harm categories, leading to "harmful conflicts" where mitigating one type of harm may inadvertently amplify another, thus increasing overall harmful rate. To address this issue, we propose Conflict-aware Adaptive Safety Guidance (CASG), a training-free framework that dynamically identifies and applies the category-aligned safety direction during generation. CASG is composed of two components: (i) Conflict-aware Category Identification (CaCI), which identifies the harmful category most aligned with the model's evolving generative state, and (ii) Conflict-resolving Guidance Application (CrGA), which applies safety steering solely along the identified category to avoid multi-category interference. CASG can be applied to both latent-space and text-space safeguards. Experiments on T2I safety benchmarks demonstrate CASG's state-of-the-art performance, reducing the harmful rate by up to 15.4% compared to existing methods.
>
---
#### [replaced 058] Topological Alignment of Shared Vision-Language Embedding Space
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10889](https://arxiv.org/pdf/2510.10889)**

> **作者:** Junwon You; Dasol Kang; Jae-Hun Jung
>
> **备注:** 27 pages, 5 figures, 24 tables
>
> **摘要:** Contrastive Vision-Language Models (VLMs) have demonstrated strong zero-shot capabilities. However, their cross-modal alignment remains biased toward English due to limited multilingual multimodal data. Recent multilingual extensions have alleviated this gap but enforce instance-level alignment while neglecting the global geometry of the shared embedding space. We address this problem by introducing ToMCLIP (Topological Alignment for Multilingual CLIP), a topology-aware framework aligning embedding spaces with topology-preserving constraints. The proposed method applies persistent homology to define a topological alignment loss and approximates persistence diagram with theoretical error bounds using graph sparsification strategy. This work validates the proposed approach, showing enhanced structural coherence of multilingual representations, higher zero-shot accuracy on the CIFAR-100, and stronger multilingual retrieval performance on the xFlickr&CO. Beyond VLMs, the proposed approach provides a general method for incorporating topological alignment into representation learning. Code is available at this https URL.
>
---
#### [replaced 059] ITO: Images and Texts as One via Synergizing Multiple Alignment and Training-Time Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02767](https://arxiv.org/pdf/2603.02767)**

> **作者:** HanZpeng Liu; Yaqian Li; Zidan Wang; Shuoxi Zhang; Zonglin Zhao; Zihao Bo; Rinyoichi Takezoe; Kaiwen Long; Kun He
>
> **摘要:** Image-text contrastive pretraining has become a dominant paradigm for visual representation learning, yet existing methods often yield representations that remain partially organized by modality. We propose ITO, a framework addressing this limitation through two synergistic mechanisms. Multimodal multiple alignment enriches supervision by mining diverse image-text correspondences, while a lightweight training-time multimodal fusion module enforces structured cross-modal interaction. Crucially, the fusion module is discarded at inference, preserving the efficiency of standard dual-encoder architectures. Extensive experiments show that ITO consistently outperforms strong baselines across classification, retrieval, and multimodal benchmarks. Our analysis reveals that while multiple alignment drives discriminative power, training-time fusion acts as a critical structural regularizer -- eliminating the modality gap and stabilizing training dynamics to prevent the early saturation often observed in aggressive contrastive learning.
>
---
#### [replaced 060] A Geometry-Based View of Mahalanobis OOD Detection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15202](https://arxiv.org/pdf/2510.15202)**

> **作者:** Denis Janiak; Jakub Binkowski; Tomasz Kajdanowicz
>
> **摘要:** Out-of-distribution (OOD) detection is critical for reliable deployment of vision models. Mahalanobis-based detectors remain strong baselines, yet their performance varies widely across modern pretrained representations, and it is unclear which properties of a feature space cause these methods to succeed or fail. We conduct a large-scale study across diverse foundation-model backbones and Mahalanobis variants. First, we show that Mahalanobis-style OOD detection is not universally reliable: performance is highly representation-dependent and can shift substantially with pretraining data and fine-tuning regimes. Second, we link this variability to in-distribution geometry and identify a two-term ID summary that consistently tracks Mahalanobis OOD behavior across detectors: within-class spectral structure and local intrinsic dimensionality. Finally, we treat normalization as a geometric control mechanism and introduce radially scaled $\ell_2$ normalization, $\phi_\beta(z)=z/\|z\|^\beta$, which preserves directions while contracting or expanding feature radii. Varying $\beta$ changes the radii while preserving directions, so the same quadratic detector sees a different ID geometry. We choose $\beta$ from ID-only geometry signals and typically outperform fixed normalization baselines.
>
---
#### [replaced 061] MatPedia: A Universal Generative Foundation for High-Fidelity Material Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16957](https://arxiv.org/pdf/2511.16957)**

> **作者:** Di Luo; Shuhui Yang; Mingxin Yang; Jiawei Lu; Yixuan Tang; Xintong Han; Zhuo Chen; Beibei Wang; Chunchao Guo
>
> **摘要:** Physically-based rendering (PBR) materials are fundamental to photorealistic graphics, yet their creation remains labor-intensive and requires specialized expertise. While generative models have advanced material synthesis, existing methods lack a unified representation bridging natural image appearance and PBR properties, leading to fragmented task-specific pipelines and inability to leverage large-scale RGB image data. We present MatPedia, a foundation model built upon a novel joint RGB-PBR representation that compactly encodes materials into two interdependent latents: one for RGB appearance and one for the four PBR maps encoding complementary physical properties. By formulating them as a 5-frame sequence and employing video diffusion architectures, MatPedia naturally captures their correlations while transferring visual priors from RGB generation models. This joint representation enables a unified framework handling multiple material tasks--text-to-material generation, image-to-material generation, and intrinsic decomposition--within a single architecture. Trained on MatHybrid-410K, a mixed corpus combining PBR datasets with large-scale RGB images, MatPedia achieves native $1024\times1024$ synthesis that substantially surpasses existing approaches in both quality and diversity.
>
---
#### [replaced 062] Tracing 3D Anatomy in 2D Strokes: A Multi-Stage Projection Driven Approach to Cervical Spine Fracture Identification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.15235](https://arxiv.org/pdf/2601.15235)**

> **作者:** Fabi Nahian Madhurja; Rusab Sarmun; Muhammad E. H. Chowdhury; Adam Mushtak; Israa Al-Hashimi; Sohaib Bassam Zoghoul
>
> **备注:** 47 pages, 36 figures, 17 tables. Includes supplementary material. Under review at Medical Image Analysis
>
> **摘要:** Cervical spine fractures demand rapid and accurate diagnosis for effective clinical management. This study presents an automated, end-to-end pipeline for fracture detection across cervical vertebrae (C1--C7) that assesses the feasibility of fracture recognition from vertebra-level volumes of interest extracted using estimated 3D masks derived from fused orthogonal 2D segmentations. Unlike traditional 3D methods, our approach approximates 3D volumes via optimized 2D axial, sagittal, and coronal projections to reduce input dimensionality of intermediate pre-processing steps while maintaining high diagnostic performance for downstream fracture classification. First, spine regions of interest are localized from multi-view variance projections using a YOLOv8 detector, achieving a 3D mean Intersection over Union of 94.45%. Next, multi-label vertebra segmentation is performed using a DenseNet121-Unet architecture on energy-based sagittal and coronal projections, attaining a mean Dice score of 87.86%. The orthogonal 2D masks are then fused to reconstruct an estimated 3D mask for each vertebra, which is used to extract volumes of interest from the original CT. These extracted vertebra volumes are subsequently analyzed for fractures using an ensemble of 2.5D spatio-sequential CNN-Transformer models, yielding vertebra-level and patient-level F1 scores of 68.15 and 82.26, with area under the receiver operating characteristic curve scores of 91.62 and 83.04, respectively. The framework is further validated through an explainability study using saliency map visualizations and an interobserver variability analysis. Overall, the results indicate that this projection-based strategy delivers clinically relevant performance comparable to expert radiologists, while reducing the dimensionality of intermediate stages, supporting its potential for practical deployment.
>
---
#### [replaced 063] Scriboora: Rethinking Human Pose Forecasting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15565](https://arxiv.org/pdf/2511.15565)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** Human pose forecasting predicts future poses based on past observations, and has many significant applications in areas such as action recognition, autonomous driving or human-robot interaction. This paper evaluates a wide range of pose forecasting algorithms in the task of absolute pose forecasting, revealing many reproducibility issues, and provides a unified training and evaluation pipeline. After drawing a high-level analogy to the task of speech understanding, it is shown that recent speech models can be efficiently adapted to the task of pose forecasting, and improve current state-of-the-art performance. Finally, the robustness of the models is evaluated, using noisy joint coordinates obtained from a pose estimation model, to reflect a realistic type of noise, which is closer to real-world applications. For this a new dataset variation is introduced, and it is shown that estimated poses result in a substantial performance degradation, and how much of it can be recovered again by unsupervised finetuning.
>
---
#### [replaced 064] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.03101](https://arxiv.org/pdf/2603.03101)**

> **作者:** Jun Yeong Park; JunYoung Seo; Minji Kang; Yu Rang Park
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The CLIP model's outstanding generalization has driven recent success in Zero-Shot Anomaly Detection (ZSAD) for detecting anomalies in unseen categories. The core challenge in ZSAD is to specialize the model for anomaly detection tasks while preserving CLIP's powerful generalization capability. Existing approaches attempting to solve this challenge share the fundamental limitation of a patch-agnostic design that processes all patches monolithically without regard for their unique characteristics. To address this limitation, we propose MoECLIP, a Mixture-of-Experts (MoE) architecture for the ZSAD task, which achieves patch-level adaptation by dynamically routing each image patch to a specialized Low-Rank Adaptation (LoRA) expert based on its unique characteristics. Furthermore, to prevent functional redundancy among the LoRA experts, we introduce (1) Frozen Orthogonal Feature Separation (FOFS), which orthogonally separates the input feature space to force experts to focus on distinct information, and (2) a simplex equiangular tight frame (ETF) loss to regulate the expert outputs to form maximally equiangular representations. Comprehensive experimental results across 14 benchmark datasets spanning industrial and medical domains demonstrate that MoECLIP outperforms existing state-of-the-art methods. The code is available at this https URL.
>
---
#### [replaced 065] FlowCLAS: Enhancing Normalizing Flow Via Contrastive Learning For Anomaly Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.19888](https://arxiv.org/pdf/2411.19888)**

> **作者:** Chang Won Lee; Selina Leveugle; Svetlana Stolpner; Chris Langley; Paul Grouchy; Jonathan Kelly; Steven L. Waslander
>
> **备注:** WACV 2026 Camera Ready
>
> **摘要:** Anomaly segmentation is an essential capability for safety-critical robotics applications that must be aware of unexpected events. Normalizing flows (NFs), a class of generative models, are a promising approach for this task due to their ability to model the inlier data distribution efficiently. However, their performance falters in dynamic scenes, where complex, multi-modal data distributions cause them to struggle with identifying out-of-distribution samples, leaving a performance gap to leading discriminative methods. To address this limitation, we introduce FlowCLAS, a hybrid framework that enhances the traditional maximum likelihood objective of NFs with a discriminative, contrastive loss. Leveraging Outlier Exposure, this objective explicitly enforces a separation between normal and anomalous features in the latent space, retaining the probabilistic foundation of NFs while embedding the discriminative power they lack. The strength of this approach is demonstrated by FlowCLAS establishing new state-of-the-art (SOTA) performance across multiple challenging anomaly segmentation benchmarks for robotics, including Fishyscapes Lost & Found, Road Anomaly, SegmentMeIfYouCan-ObstacleTrack, and ALLO. Our experiments also show that this contrastive approach is more effective than other outlier-based training strategies for NFs, successfully bridging the performance gap to leading discriminative methods. Project page: this https URL
>
---
#### [replaced 066] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12880](https://arxiv.org/pdf/2508.12880)**

> **作者:** Chubin Chen; Jiashu Zhu; Xiaokun Feng; Nisha Huang; Chen Zhu; Meiqi Wu; Fangyuan Mao; Jiahong Wu; Xiangxiang Chu; Xiu Li
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Classifier-free Guidance (CFG) is a widely used technique in modern diffusion models for enhancing sample quality and prompt adherence. However, through an empirical analysis on Gaussian mixture modeling with a closed-form solution, we observe a discrepancy between the suboptimal results produced by CFG and the ground truth. The model's excessive reliance on these suboptimal predictions often leads to semantic incoherence and low-quality outputs. To address this issue, we first empirically demonstrate that the model's suboptimal predictions can be effectively refined using sub-networks of the model itself. Building on this insight, we propose S$^2$-Guidance, a novel method that leverages stochastic block-dropping during the forward process to construct stochastic sub-networks, effectively guiding the model away from potential low-quality predictions and toward high-quality outputs. Extensive qualitative and quantitative experiments on text-to-image and text-to-video generation tasks demonstrate that S$^2$-Guidance delivers superior performance, consistently surpassing CFG and other advanced guidance strategies. Our code will be released.
>
---
#### [replaced 067] Track Anything Behind Everything: Zero-Shot Amodal Video Object Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.19210](https://arxiv.org/pdf/2411.19210)**

> **作者:** Finlay G. C. Hudson; William A. P. Smith
>
> **摘要:** We present Track Anything Behind Everything (TABE), a novel dataset, pipeline, and evaluation framework for zero-shot amodal completion from visible masks. Unlike existing methods that require pretrained class labels, our approach uses a single query mask from the first frame where the object is visible, enabling flexible, zero-shot inference. Our dataset, TABE-51 provides highly accurate ground truth amodal segmentation masks without the need for human estimation or 3D reconstruction. Our TABE pipeline is specifically designed to handle amodal completion, even in scenarios where objects are completely occluded. We also introduce a specialised evaluation framework that isolates amodal completion performance, free from the influence of traditional visual segmentation metrics.
>
---
#### [replaced 068] Photo3D: Advancing Photorealistic 3D Generation through Structure-Aligned Detail Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08535](https://arxiv.org/pdf/2512.08535)**

> **作者:** Xinyue Liang; Zhinyuan Ma; Lingchen Sun; Yanjun Guo; Lei Zhang
>
> **摘要:** Although recent 3D-native generators have made great progress in synthesizing reliable geometry, they still fall short in achieving realistic appearances. A key obstacle lies in the lack of diverse and high-quality real-world 3D assets with rich texture details, since capturing such data is intrinsically difficult due to the diverse scales of scenes, non-rigid motions of objects, and the limited precision of 3D scanners. We introduce Photo3D, a framework for advancing photorealistic 3D generation, which is driven by the image data generated by the GPT-4o-Image model. Considering that the generated images can distort 3D structures due to their lack of multi-view consistency, we design a structure-aligned multi-view synthesis pipeline and construct a detail-enhanced multi-view dataset paired with 3D geometry. Building on it, we present a realistic detail enhancement scheme that leverages perceptual feature adaptation and semantic structure matching to enforce appearance consistency with realistic details while preserving the structural consistency with the 3D-native geometry. Our scheme is general to different 3D-native generators, and we present dedicated training strategies to facilitate the optimization of geometry-texture coupled and decoupled 3D-native generation paradigms. Experiments demonstrate that Photo3D generalizes well across diverse 3D-native generation paradigms and achieves state-of-the-art photorealistic 3D generation performance.
>
---
#### [replaced 069] A Unified Revisit of Temperature in Classification-Based Knowledge Distillation
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02430](https://arxiv.org/pdf/2603.02430)**

> **作者:** Logan Frank; Jim Davis
>
> **摘要:** A central idea of knowledge distillation is to expose relational structure embedded in the teacher's weights for the student to learn, which is often facilitated using a temperature parameter. Despite its widespread use, there remains limited understanding on how to select an appropriate temperature value, or how this value depends on other training elements such as optimizer, teacher pretraining/finetuning, etc. In practice, temperature is commonly chosen via grid search or by adopting values from prior work, which can be time-consuming or may lead to suboptimal student performance when training setups differ. In this work, we posit that temperature is closely linked to these training components and present a unified study that systematically examines such interactions. From analyzing these cross-connections, we identify and present common situations that have a pronounced impact on temperature selection, providing valuable guidance for practitioners employing knowledge distillation in their work.
>
---
#### [replaced 070] Beyond Accuracy: What Matters in Designing Well-Behaved Image Classification Models?
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.17110](https://arxiv.org/pdf/2503.17110)**

> **作者:** Robin Hesse; Doğukan Bağcı; Bernt Schiele; Simone Schaub-Meyer; Stefan Roth
>
> **备注:** Published in TMLR (01/2026) | OpenReview: this https URL | Project page: this https URL
>
> **摘要:** Deep learning has become an essential part of computer vision, with deep neural networks (DNNs) excelling in predictive performance. However, they often fall short in other critical quality dimensions, such as robustness, calibration, or fairness. While existing studies have focused on a subset of these quality dimensions, none have explored a more general form of "well-behavedness" of DNNs. With this work, we address this gap by simultaneously studying nine different quality dimensions for image classification. Through a large-scale study, we provide a bird's-eye view by analyzing 326 backbone models and how different training paradigms and model architectures affect these quality dimensions. We reveal various new insights such that (i) vision-language models exhibit high class balance on ImageNet-1k classification and strong robustness against domain changes; (ii) training models initialized with weights obtained through self-supervised learning is an effective strategy to improve most considered quality dimensions; and (iii) the training dataset size is a major driver for most of the quality dimensions. We conclude our study by introducing the QUBA score (Quality Understanding Beyond Accuracy), a novel metric that ranks models across multiple dimensions of quality, enabling tailored recommendations based on specific user needs.
>
---
#### [replaced 071] Natural Adversaries: Fuzzing Autonomous Vehicles with Realistic Roadside Object Placements
- **分类: cs.CV; cs.SE**

- **链接: [https://arxiv.org/pdf/2409.10562](https://arxiv.org/pdf/2409.10562)**

> **作者:** Yang Sun; Haoyu Wang; Christopher M. Poskitt; Jun Sun
>
> **备注:** Accepted by the 19th IEEE International Conference on Software Testing, Verification and Validation (ICST 2026)
>
> **摘要:** The emergence of Autonomous Vehicles (AVs) has spurred research into testing the resilience of their perception systems, i.e., ensuring that they are not susceptible to critical misjudgements. It is important that these systems are tested not only with respect to other vehicles on the road, but also with respect to objects placed on the roadside. Trash bins, billboards, and greenery are examples of such objects, typically positioned according to guidelines developed for the human visual system, which may not align perfectly with the needs of AVs. Existing tests, however, usually focus on adversarial objects with conspicuous shapes or patches, which are ultimately unrealistic due to their unnatural appearance and reliance on white-box knowledge. In this work, we introduce a black-box attack on AV perception systems that creates realistic adversarial scenarios (i.e., satisfying road design guidelines) by manipulating the positions of common roadside objects and without resorting to "unnatural" adversarial patches. In particular, we propose TrashFuzz, a fuzzing algorithm that finds scenarios in which the placement of these objects leads to substantial AV misperceptions -- such as mistaking a traffic light's colour -- with the overall goal of causing traffic-law violations. To ensure realism, these scenarios must satisfy several rules encoding regulatory guidelines governing the placement of objects on public streets. We implemented and evaluated these attacks on the Apollo autonomous driving system, finding that TrashFuzz induced violations of 15 out of 24 traffic laws.
>
---
#### [replaced 072] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.25845](https://arxiv.org/pdf/2509.25845)**

> **作者:** Jinho Chang; Jaemin Kim; Jong Chul Ye
>
> **备注:** Poster in ICLR 2026; 22 pages, 9 figures
>
> **摘要:** Recent advancements in diffusion and flow-matching models have demonstrated remarkable capabilities in high-fidelity image synthesis. A prominent line of research involves reward-guided guidance, which steers the generation process during inference to align with specific objectives. However, leveraging this reward-guided approach to the task of image editing, which requires preserving the semantic content of the source image while enhancing a target reward, is largely unexplored. In this work, we introduce a novel framework for training-free, reward-guided image editing. We formulate the editing process as a trajectory optimal control problem where the reverse process of a diffusion model is treated as a controllable trajectory originating from the source image, and the adjoint states are iteratively updated to steer the editing process. Through extensive experiments across distinct editing tasks, we demonstrate that our approach significantly outperforms existing inversion-based training-free guidance baselines, achieving a superior balance between reward maximization and fidelity to the source image without reward hacking.
>
---
#### [replaced 073] UniLight: A Unified Representation for Lighting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04267](https://arxiv.org/pdf/2512.04267)**

> **作者:** Zitian Zhang; Iliyan Georgiev; Michael Fischer; Yannick Hold-Geoffroy; Jean-François Lalonde; Valentin Deschaintre
>
> **备注:** Project page: this https URL
>
> **摘要:** Lighting has a strong influence on visual appearance, yet understanding and representing lighting in images remains notoriously difficult. Various lighting representations exist, such as environment maps, irradiance, spherical harmonics, or text, but they are incompatible, which limits cross-modal transfer. We thus propose UniLight, a joint latent space as lighting representation, that unifies multiple modalities within a shared embedding. Modality-specific encoders for text, images, irradiance, and environment maps are trained contrastively to align their representations, with an auxiliary spherical-harmonics prediction task reinforcing directional understanding. Our multi-modal data pipeline enables large-scale training and evaluation across three tasks: lighting-based retrieval, environment-map generation, and lighting control in diffusion-based image synthesis. Experiments show that our representation captures consistent and transferable lighting features, enabling flexible manipulation across modalities.
>
---
#### [replaced 074] Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.14610](https://arxiv.org/pdf/2509.14610)**

> **作者:** Yue Cao; Quansong He; Kaishen Wang; Jianlong Xiong; Zhang Yi; Tao He
>
> **摘要:** U-like networks have become fundamental frameworks in medical image segmentation through skip connections that bridge high-level semantics and low-level spatial details. Despite their success, conventional skip connections exhibit two key limitations: inter-feature constraints and intra-feature constraints. The inter-feature constraint refers to the static nature of feature fusion in traditional skip connections, where information is transmitted along fixed pathways regardless of feature content. The intra-feature constraint arises from the insufficient modeling of multi-scale feature interactions, thereby hindering the effective aggregation of global contextual information. To overcome these limitations, we propose a novel Dynamic Skip Connection (DSC) block that fundamentally enhances cross-layer connectivity through adaptive mechanisms. The DSC block integrates two complementary components. (1) Test-Time Training (TTT) module. This module addresses the inter-feature constraint by enabling dynamic adaptation of hidden representations during inference, facilitating content-aware feature refinement. (2) Dynamic Multi-Scale Kernel (DMSK) module. To mitigate the intra-feature constraint, this module adaptively selects kernel sizes based on global contextual cues, enhancing the network capacity for multi-scale feature integration. The DSC block is architecture-agnostic and can be seamlessly incorporated into existing U-like network structures. Extensive experiments demonstrate the plug-and-play effectiveness of the proposed DSC block across CNN-based, Transformer-based, hybrid CNN-Transformer, and Mamba-based U-like networks.
>
---
#### [replaced 075] Adaptive Dynamic Dehazing via Instruction-Driven and Task-Feedback Closed-Loop Optimization for Diverse Downstream Task Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00542](https://arxiv.org/pdf/2603.00542)**

> **作者:** Yafei Zhang; Shuaitian Song; Huafeng Li; Shujuan Wang; Yu Liu
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** In real-world vision systems,haze removal is required not only to enhance image visibility but also to meet the specific needs of diverse downstream this http URL address this challenge,we propose a novel adaptive dynamic dehazing framework that incorporates a closed-loop optimization this http URL enables feedback-driven refinement based on downstream task performance and user instruction-guided adjustment during inference,allowing the model to satisfy the specific requirements of multiple downstream tasks without this http URL,our framework integrates two complementary and innovative mechanisms: (1)a task feedback loop that dynamically modulates dehazing outputs based on performance across multiple downstream tasks,and (2) a text instruction interface that allows users to specify high-level task this http URL dual-guidance strategy enables the model to adapt its dehazing behavior after training,tailoring outputs in real time to the evolving needs of multiple this http URL experiments across various vision tasks demonstrate the strong effectiveness,robustness,and generalizability of our this http URL results establish a new paradigm for interactive,task-adaptive dehazing that actively collaborates with downstream applications.
>
---
#### [replaced 076] Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉token剪枝任务，解决静态策略导致性能不一致的问题。提出MoB方法，通过多目标平衡覆盖优化剪枝效果，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2505.10118](https://arxiv.org/pdf/2505.10118)**

> **作者:** Yangfu Li; Hongjian Zhan; Tianyi Chen; Qi Liu; Yue Lu
>
> **备注:** 31 pages,9 figures,conference
>
> **摘要:** Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging $\epsilon$-covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5$\times$ with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks.
>
---
#### [replaced 077] EvalMVX: A Unified Benchmarking for Neural 3D Reconstruction under Diverse Multiview Setups
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.24065](https://arxiv.org/pdf/2602.24065)**

> **作者:** Zaiyan Yang; Jieji Ren; Xiangyi Wang; zonglin li; Xu Cao; Heng Guo; Zhanyu Ma; Boxin Shi
>
> **摘要:** Recent advancements in neural surface reconstruction have significantly enhanced 3D reconstruction. However, current real world datasets mainly focus on benchmarking multiview stereo (MVS) based on RGB inputs. Multiview photometric stereo (MVPS) and multiview shape from polarization (MVSfP), though indispensable on high-fidelity surface reconstruction and sparse inputs, have not been quantitatively assessed together with MVS. To determine the working range of different MVX (MVS, MVSfP, and MVPS) techniques, we propose EvalMVX, a real-world dataset containing $25$ objects, each captured with a polarized camera under $20$ varying views and $17$ light conditions including OLAT and natural illumination, leading to $8,500$ images. Each object includes aligned ground-truth 3D mesh, facilitating quantitative benchmarking of MVX methods simultaneously. Based on our EvalMVX, we evaluate $13$ MVX methods published in recent years, record the best-performing methods, and identify open problems under diverse geometric details and reflectance types. We hope EvalMVX and the benchmarking results can inspire future research on multiview 3D reconstruction.
>
---
#### [replaced 078] DCENWCNet: A Deep CNN Ensemble Network for White Blood Cell Classification with LIME-Based Explainability
- **分类: cs.CV; cs.AI; q-bio.CB; stat.ML**

- **链接: [https://arxiv.org/pdf/2502.05459](https://arxiv.org/pdf/2502.05459)**

> **作者:** Sibasish Dhibar
>
> **摘要:** White blood cells (WBC) are important parts of our immune system, and they protect our body against infections by eliminating viruses, bacteria, parasites and fungi. The number of WBC types and the total number of WBCs provide important information about our health status. A traditional method, convolutional neural networks (CNN), a deep learning architecture, can classify the blood cell from a part of an object and perform object recognition. Various CNN models exhibit potential; however, their development often involves ad-hoc processes that neglect unnecessary layers, leading to issues with unbalanced datasets and insufficient data augmentation. To address these challenges, we propose a novel ensemble approach that integrates three CNN architectures, each uniquely configured with different dropout and max-pooling layer settings to enhance feature learning. This ensemble model, named DCENWCNet, effectively balances the bias-variance trade-off. When evaluated on the widely recognized Rabbin-WBC dataset, our model outperforms existing state-of-the-art networks, achieving highest mean accuracy. Additionally, it demonstrates superior performance in precision, recall, F1-score, and Area Under the ROC Curve (AUC) across all categories. To delve deeper into the interpretability of classifiers, we employ reliable post-hoc explanation techniques, including Local Interpretable Model-Agnostic Explanations (LIME). These methods approximate the behavior of a black-box model by elucidating the relationships between feature values and predictions. Interpretable results enable users to comprehend and validate the model's predictions, thereby increasing their confidence in the automated diagnosis.
>
---
#### [replaced 079] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01007](https://arxiv.org/pdf/2603.01007)**

> **作者:** Xubo Zhu; Haoyang Zhang; Fei He; Rui Wu; Yanhu Shan; Wen Yang; Huai Yu
>
> **备注:** 10 pages, 6 figures. Accepted at CVPR 2026
>
> **摘要:** 3D semantic occupancy prediction is crucial for autonomous driving perception, offering comprehensive geometric scene understanding and semantic recognition. However, existing methods struggle with geometric misalignment in view transformation due to the lack of pixel-level accurate depth estimation, and severe spatial class imbalance where semantic categories exhibit strong spatial anisotropy. To address these challenges, we propose \textbf{this http URL}, a depth- and region-guided occupancy prediction framework. Specifically, we introduce a depth-guided 2D-to-3D View Transformer (D$^2$-VFormer) that effectively leverages high-quality dense depth cues from MoGe-2 to construct reliable geometric priors, thereby enabling precise geometric alignment of voxel features. Moreover, inspired by the Mixture-of-Experts (MoE) framework, we propose a region-guided Expert Transformer (R/R$^2$-EFormer) that adaptively allocates region-specific experts to focus on different spatial regions, effectively addressing spatial semantic variations. Thus, the two components make complementary contributions: depth guidance ensures geometric alignment, while region experts enhance semantic learning. Experiments on the Occ3D--nuScenes benchmark demonstrate that \textbf{this http URL} improves the strong baseline BEVDet4D by 7.43\% mIoU and 3.09\% IoU under the full vision-only setting.
>
---
#### [replaced 080] Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.04336](https://arxiv.org/pdf/2501.04336)**

> **作者:** Zeyi Huang; Yuyang Ji; Xiaofang Wang; Nikhil Mehta; Tong Xiao; Donghyun Lee; Sigmund Vanvalkenburgh; Shengxin Zha; Bolin Lai; Yiqiu Ren; Licheng Yu; Ning Zhang; Yong Jae Lee; Miao Liu
>
> **摘要:** Long-form video understanding with Large Vision Language Models is challenged by the need to analyze temporally dispersed yet spatially concentrated key moments within limited context windows. In this work, we introduce VideoMindPalace, a new framework inspired by the "Mind Palace", which organizes critical video moments into a topologically structured semantic graph. VideoMindPalace organizes key information through (i) hand-object tracking and interaction, (ii) clustered activity zones representing specific areas of recurring activities, and (iii) environment layout mapping, allowing natural language parsing by LLMs to provide grounded insights on spatio-temporal and 3D context. In addition, we propose the Video MindPalace Benchmark (VMB), to assess human-like reasoning, including spatial localization, temporal reasoning, and layout-aware sequential understanding. Evaluated on VMB and established video QA datasets, including EgoSchema, NExT-QA, IntentQA, and the Active Memories Benchmark, VideoMindPalace demonstrates notable gains in spatio-temporal coherence and human-aligned reasoning, advancing long-form video analysis capabilities in VLMs.
>
---
#### [replaced 081] Measurement-Consistent Langevin Corrector for Stabilizing Latent Diffusion Inverse Problem Solvers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.04791](https://arxiv.org/pdf/2601.04791)**

> **作者:** Lee Hyoseok; Sohwi Lim; Eunju Cha; Tae-Hyun Oh
>
> **备注:** Under Review
>
> **摘要:** While latent diffusion models (LDMs) have emerged as powerful priors for inverse problems, existing LDM-based solvers frequently suffer from instability. In this work, we first identify the instability as a discrepancy between the solver dynamics and stable reverse diffusion dynamics learned by the diffusion model, and show that reducing this gap stabilizes the solver. Building on this, we introduce \textit{Measurement-Consistent Langevin Corrector (MCLC)}, a theoretically grounded plug-and-play stabilization module that remedies the LDM-based inverse problem solvers through measurement-consistent Langevin updates. Compared to prior approaches that rely on linear manifold assumptions, which often fail to hold in latent space, MCLC provides a principled stabilization mechanism, leading to more stable and reliable behavior in latent space.
>
---
#### [replaced 082] Adaptive Quantized Planetary Crater Detection System for Autonomous Space Exploration
- **分类: cs.LG; cs.AI; cs.CV; cs.ET; eess.SY**

- **链接: [https://arxiv.org/pdf/2508.18025](https://arxiv.org/pdf/2508.18025)**

> **作者:** Aditri Paul; Archan Paul
>
> **备注:** 10 pages, 6 figures. A research paper on a novel deep learning framework for planetary crater detection
>
> **摘要:** Autonomous planetary exploration demands real-time, high-fidelity environmental perception. Standard deep learning models, however, require far more memory and compute than space-qualified, radiation-hardened, power-optimized hardware can provide. This limitation creates a severe design bottleneck. Engineers struggle to deploy sophisticated detection architectures without overloading the strict power and memory limits of onboard computers of outer space planetary exploration platforms. In this foundational concept paper, we propose the Adaptive Quantized Planetary Crater Detection System (AQ-PCDSys) to resolve this bottleneck. We present an architectural blueprint integrating a Quantized Neural Network (QNN), refined through Quantization Aware Training (QAT), with an Adaptive Multi-Sensor Fusion (AMF) module and Multi-Scale Detection Heads. By forcing weights into low-precision integer arithmetic during the training and optimization phase, our framework strips away the floating-point overhead that typically overwhelms onboard computer's processors. The AMF module directly addresses sensor fragility. It dynamically selects and fuses Optical Imagery (OI) and Digital Elevation Models (DEMs) at the feature level to provide reliable sensor inputs during extreme cross-illuminations and sudden sensor dropouts. As a concept paper, this work establishes the technical and mathematical justifications for the architecture rather than presenting completed empirical ablation studies. We outline a rigorous Hardware-in-the-Loop (HITL) evaluation protocol for immediate future validation, paving the way for next-generation, hardware-aware space-mission software.
>
---
#### [replaced 083] FireANTs: Adaptive Riemannian Optimization for Multi-Scale Diffeomorphic Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.01249](https://arxiv.org/pdf/2404.01249)**

> **作者:** Rohit Jena; Pratik Chaudhari; James C. Gee
>
> **备注:** Accepted at Nature Communications
>
> **摘要:** The paper proposes FireANTs, a multi-scale Adaptive Riemannian Optimization algorithm for dense diffeomorphic image matching. Existing state-of-the-art methods for diffeomorphic image matching are slow due to inefficient implementations and slow convergence due to the ill-conditioned nature of the optimization problem. Deep learning methods offer fast inference but require extensive training time, substantial inference memory, and fail to generalize across long-tailed distributions or diverse image modalities, necessitating costly retraining. We address these challenges by proposing a training-free, GPU-accelerated multi-scale Adaptive Riemannian Optimization algorithm for fast and accurate dense diffeomorphic image matching. FireANTs runs about 2.5x faster than ANTs on a CPU, and upto 1200x faster on a GPU. On a single GPU, FireANTs performs competitively with deep learning methods on inference runtime while consuming upto 10x less memory. FireANTs shows remarkable robustness to a wide variety of matching problems across modalities, species, and organs without any domain-specific training or tuning. Our framework allows hyperparameter grid search studies with significantly less resources and time compared to traditional and deep learning registration algorithms alike.
>
---
#### [replaced 084] From Press to Pixels: Evolving Urdu Text Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.13943](https://arxiv.org/pdf/2505.13943)**

> **作者:** Samee Arif; Sualeha Farid
>
> **摘要:** This paper presents a comparative analysis of Large Language Models (LLMs) and traditional Optical Character Recognition (OCR) systems on Urdu newspapers, addressing challenges posed by complex multi-column layouts, low-resolution scans, and the stylistic variability of the Nastaliq script. To handle these challenges, we fine-tune YOLOv11x models for article- and column-level text block extraction and train a SwinIR-based super-resolution module that enhances image quality for downstream text recognition, improving accuracy by an average of 50%. We further introduce the Urdu Newspaper Benchmark (UNB), a manually annotated dataset for Urdu OCR comprising 829 paragraph images with a total of 9,982 sentences. Using UNB and the OpenITI corpus, we conduct a systematic comparison between traditional CNN+RNN-based OCR systems and modern LLMs, presenting detailed insertion, deletion, and substitution error analyses alongside character-level confusion patterns. We find that Gemini-2.5-Pro achieves the best performance on UNB (WER 0.133), while fine-tuning GPT-4o on just 500 in-domain samples yields a 6.13% absolute WER improvement, demonstrating the adaptability of LLMs to low-resource, morphologically complex scripts like Urdu. The UNB dataset and fine-tuned models are publicly available at this https URL.
>
---
#### [replaced 085] Fast Equivariant Imaging: Acceleration for Unsupervised Learning via Augmented Lagrangian and Auxiliary PnP Denoisers
- **分类: eess.IV; cs.CV; cs.LG; math.OC**

- **链接: [https://arxiv.org/pdf/2507.06764](https://arxiv.org/pdf/2507.06764)**

> **作者:** Guixian Xu; Jinglai Li; Junqi Tang
>
> **备注:** 31 pages
>
> **摘要:** In this work, we propose Fast Equivariant Imaging (FEI), a novel unsupervised learning framework to rapidly and efficiently train deep imaging networks without ground-truth data. From the perspective of reformulating the Equivariant Imaging based optimization problem via the method of Lagrange multipliers and utilizing plug-and-play denoisers, this novel unsupervised scheme shows superior efficiency and performance compared to the vanilla Equivariant Imaging paradigm. In particular, our FEI schemes achieve an order-of-magnitude (10x) acceleration over standard EI on training U-Net for X-ray CT reconstruction and image inpainting, with improved generalization performance. In addition, the proposed scheme enables efficient test-time adaptation of a pretrained model to individual samples to secure further performance improvements. Extensive experiments show that the proposed approach provides a noticeable efficiency and performance gain over existing unsupervised methods and model adaptation techniques.
>
---
