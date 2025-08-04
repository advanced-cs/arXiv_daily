# 计算机视觉 cs.CV

- **最新发布 108 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] MIHBench: Benchmarking and Mitigating Multi-Image Hallucinations in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出了一种针对多图像场景的多模态大语言模型（MLLMs）中的虚假生成问题，通过三个核心任务（多图对象存在/数量/身份一致性）和动态注意力机制，有效减少了 hallucination 发生，并提升了语义整合能力。**

- **链接: [http://arxiv.org/pdf/2508.00726v1](http://arxiv.org/pdf/2508.00726v1)**

> **作者:** Jiale Li; Mingrui Wu; Zixiang Jin; Hao Chen; Jiayi Ji; Xiaoshuai Sun; Liujuan Cao; Rongrong Ji
>
> **备注:** ACM MM25 has accepted this paper
>
> **摘要:** Despite growing interest in hallucination in Multimodal Large Language Models, existing studies primarily focus on single-image settings, leaving hallucination in multi-image scenarios largely unexplored. To address this gap, we conduct the first systematic study of hallucinations in multi-image MLLMs and propose MIHBench, a benchmark specifically tailored for evaluating object-related hallucinations across multiple images. MIHBench comprises three core tasks: Multi-Image Object Existence Hallucination, Multi-Image Object Count Hallucination, and Object Identity Consistency Hallucination, targeting semantic understanding across object existence, quantity reasoning, and cross-view identity consistency. Through extensive evaluation, we identify key factors associated with the occurrence of multi-image hallucinations, including: a progressive relationship between the number of image inputs and the likelihood of hallucination occurrences; a strong correlation between single-image hallucination tendencies and those observed in multi-image contexts; and the influence of same-object image ratios and the positional placement of negative samples within image sequences on the occurrence of object identity consistency hallucination. To address these challenges, we propose a Dynamic Attention Balancing mechanism that adjusts inter-image attention distributions while preserving the overall visual attention proportion. Experiments across multiple state-of-the-art MLLMs demonstrate that our method effectively reduces hallucination occurrences and enhances semantic integration and reasoning stability in multi-image scenarios.
>
---
#### [new 002] AniMer+: Unified Pose and Shape Estimation Across Mammalia and Aves via Family-Aware Transformer
- **分类: cs.CV**

- **简介: 该论文旨在解决跨哺乳动物与鸟类的统一姿态和形状估计问题，通过家族感知的视觉Transformer（ViT）结合Mixture-of-Experts（MoE）设计实现高效学习，同时开发扩散生成的合成数据集（CtrlAni3D/Aves），克服了传统方法网络容量小和多物种数据稀缺的挑战，显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2508.00298v1](http://arxiv.org/pdf/2508.00298v1)**

> **作者:** Jin Lyu; Liang An; Li Lin; Pujin Cheng; Yebin Liu; Xiaoying Tang
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2412.00837
>
> **摘要:** In the era of foundation models, achieving a unified understanding of different dynamic objects through a single network has the potential to empower stronger spatial intelligence. Moreover, accurate estimation of animal pose and shape across diverse species is essential for quantitative analysis in biological research. However, this topic remains underexplored due to the limited network capacity of previous methods and the scarcity of comprehensive multi-species datasets. To address these limitations, we introduce AniMer+, an extended version of our scalable AniMer framework. In this paper, we focus on a unified approach for reconstructing mammals (mammalia) and birds (aves). A key innovation of AniMer+ is its high-capacity, family-aware Vision Transformer (ViT) incorporating a Mixture-of-Experts (MoE) design. Its architecture partitions network layers into taxa-specific components (for mammalia and aves) and taxa-shared components, enabling efficient learning of both distinct and common anatomical features within a single model. To overcome the critical shortage of 3D training data, especially for birds, we introduce a diffusion-based conditional image generation pipeline. This pipeline produces two large-scale synthetic datasets: CtrlAni3D for quadrupeds and CtrlAVES3D for birds. To note, CtrlAVES3D is the first large-scale, 3D-annotated dataset for birds, which is crucial for resolving single-view depth ambiguities. Trained on an aggregated collection of 41.3k mammalian and 12.4k avian images (combining real and synthetic data), our method demonstrates superior performance over existing approaches across a wide range of benchmarks, including the challenging out-of-domain Animal Kingdom dataset. Ablation studies confirm the effectiveness of both our novel network architecture and the generated synthetic datasets in enhancing real-world application performance.
>
---
#### [new 003] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在解决管状结构分割（TSS）中领域转换带来的性能下降问题，提出TopoTTA框架，通过Stage 1增强拓扑表示和Stage 2优化连续性，实现31.81%的CLDice提升。**

- **链接: [http://arxiv.org/pdf/2508.00442v1](http://arxiv.org/pdf/2508.00442v1)**

> **作者:** Jiale Zhou; Wenhan Wang; Shikun Li; Xiaolei Qu; Xin Guo; Yizhong Liu; Wenzhong Tang; Xun Lin; Yefeng Zheng
>
> **摘要:** Tubular structure segmentation (TSS) is important for various applications, such as hemodynamic analysis and route navigation. Despite significant progress in TSS, domain shifts remain a major challenge, leading to performance degradation in unseen target domains. Unlike other segmentation tasks, TSS is more sensitive to domain shifts, as changes in topological structures can compromise segmentation integrity, and variations in local features distinguishing foreground from background (e.g., texture and contrast) may further disrupt topological continuity. To address these challenges, we propose Topology-enhanced Test-Time Adaptation (TopoTTA), the first test-time adaptation framework designed specifically for TSS. TopoTTA consists of two stages: Stage 1 adapts models to cross-domain topological discrepancies using the proposed Topological Meta Difference Convolutions (TopoMDCs), which enhance topological representation without altering pre-trained parameters; Stage 2 improves topological continuity by a novel Topology Hard sample Generation (TopoHG) strategy and prediction alignment on hard samples with pseudo-labels in the generated pseudo-break regions. Extensive experiments across four scenarios and ten datasets demonstrate TopoTTA's effectiveness in handling topological distribution shifts, achieving an average improvement of 31.81% in clDice. TopoTTA also serves as a plug-and-play TTA solution for CNN-based TSS models.
>
---
#### [new 004] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights
- **分类: cs.CV**

- **简介: 该论文旨在重新审视对抗性特征提取防御在对象检测中的应用，解决现有评估框架不统一的问题，通过构建新型基准和大规模数据集（94种特征向量/94000张图像）揭示数据分布对防御难度的影响，并提出适应性攻击与复杂模型的防御优势，为评估和设计对抗性防御提供理论依据。**

- **链接: [http://arxiv.org/pdf/2508.00649v1](http://arxiv.org/pdf/2508.00649v1)**

> **作者:** Junhao Zheng; Jiahao Sun; Chenhao Lin; Zhengyu Zhao; Chen Ma; Chong Zhang; Cong Wang; Qian Wang; Chao Shen
>
> **摘要:** Developing reliable defenses against patch attacks on object detectors has attracted increasing interest. However, we identify that existing defense evaluations lack a unified and comprehensive framework, resulting in inconsistent and incomplete assessments of current methods. To address this issue, we revisit 11 representative defenses and present the first patch defense benchmark, involving 2 attack goals, 13 patch attacks, 11 object detectors, and 4 diverse metrics. This leads to the large-scale adversarial patch dataset with 94 types of patches and 94,000 images. Our comprehensive analyses reveal new insights: (1) The difficulty in defending against naturalistic patches lies in the data distribution, rather than the commonly believed high frequencies. Our new dataset with diverse patch distributions can be used to improve existing defenses by 15.09% AP@0.5. (2) The average precision of the attacked object, rather than the commonly pursued patch detection accuracy, shows high consistency with defense performance. (3) Adaptive attacks can substantially bypass existing defenses, and defenses with complex/stochastic models or universal patch properties are relatively robust. We hope that our analyses will serve as guidance on properly evaluating patch attacks/defenses and advancing their design. Code and dataset are available at https://github.com/Gandolfczjh/APDE, where we will keep integrating new attacks/defenses.
>
---
#### [new 005] SAM-PTx: Text-Guided Fine-Tuning of SAM with Parameter-Efficient, Parallel-Text Adapters
- **分类: cs.CV; cs.LG**

- **简介: 本研究提出SAM-PTx，采用文本提示增强SAM的分割能力，通过参数高效并行文本适配器实现改进。**

- **链接: [http://arxiv.org/pdf/2508.00213v1](http://arxiv.org/pdf/2508.00213v1)**

> **作者:** Shayan Jalilian; Abdul Bais
>
> **摘要:** The Segment Anything Model (SAM) has demonstrated impressive generalization in prompt-based segmentation. Yet, the potential of semantic text prompts remains underexplored compared to traditional spatial prompts like points and boxes. This paper introduces SAM-PTx, a parameter-efficient approach for adapting SAM using frozen CLIP-derived text embeddings as class-level semantic guidance. Specifically, we propose a lightweight adapter design called Parallel-Text that injects text embeddings into SAM's image encoder, enabling semantics-guided segmentation while keeping most of the original architecture frozen. Our adapter modifies only the MLP-parallel branch of each transformer block, preserving the attention pathway for spatial reasoning. Through supervised experiments and ablations on the COD10K dataset as well as low-data subsets of COCO and ADE20K, we show that incorporating fixed text embeddings as input improves segmentation performance over purely spatial prompt baselines. To our knowledge, this is the first work to use text prompts for segmentation on the COD10K dataset. These results suggest that integrating semantic conditioning into SAM's architecture offers a practical and scalable path for efficient adaptation with minimal computational complexity.
>
---
#### [new 006] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs
- **分类: cs.CV**

- **简介: 该论文旨在解决LiDAR生成点云精度不足的问题，通过引入概率属性增强PPC模型，在3D对象检测中提升鲁棒性，有效应对低光照、远距离等场景挑战，证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.00169v1](http://arxiv.org/pdf/2508.00169v1)**

> **作者:** Bhavya Goyal; Felipe Gutierrez-Barragan; Wei Lin; Andreas Velten; Yin Li; Mohit Gupta
>
> **备注:** ICCV 2025
>
> **摘要:** LiDAR-based 3D sensors provide point clouds, a canonical 3D representation used in various scene understanding tasks. Modern LiDARs face key challenges in several real-world scenarios, such as long-distance or low-albedo objects, producing sparse or erroneous point clouds. These errors, which are rooted in the noisy raw LiDAR measurements, get propagated to downstream perception models, resulting in potentially severe loss of accuracy. This is because conventional 3D processing pipelines do not retain any uncertainty information from the raw measurements when constructing point clouds. We propose Probabilistic Point Clouds (PPC), a novel 3D scene representation where each point is augmented with a probability attribute that encapsulates the measurement uncertainty (or confidence) in the raw data. We further introduce inference approaches that leverage PPC for robust 3D object detection; these methods are versatile and can be used as computationally lightweight drop-in modules in 3D inference pipelines. We demonstrate, via both simulations and real captures, that PPC-based 3D inference methods outperform several baselines using LiDAR as well as camera-LiDAR fusion models, across challenging indoor and outdoor scenarios involving small, distant, and low-albedo objects, as well as strong ambient light. Our project webpage is at https://bhavyagoyal.github.io/ppc .
>
---
#### [new 007] Video Forgery Detection with Optical Flow Residuals and Spatial-Temporal Consistency
- **分类: cs.CV**

- **简介: 该论文旨在解决视频伪造检测中的细粒度时空一致性问题，提出基于RGB与光流残差的双分支框架，通过融合特征实现对伪造视频的检测。**

- **链接: [http://arxiv.org/pdf/2508.00397v1](http://arxiv.org/pdf/2508.00397v1)**

> **作者:** Xi Xue; Kunio Suzuki; Nabarun Goswami; Takuya Shintate
>
> **摘要:** The rapid advancement of diffusion-based video generation models has led to increasingly realistic synthetic content, presenting new challenges for video forgery detection. Existing methods often struggle to capture fine-grained temporal inconsistencies, particularly in AI-generated videos with high visual fidelity and coherent motion. In this work, we propose a detection framework that leverages spatial-temporal consistency by combining RGB appearance features with optical flow residuals. The model adopts a dual-branch architecture, where one branch analyzes RGB frames to detect appearance-level artifacts, while the other processes flow residuals to reveal subtle motion anomalies caused by imperfect temporal synthesis. By integrating these complementary features, the proposed method effectively detects a wide range of forged videos. Extensive experiments on text-to-video and image-to-video tasks across ten diverse generative models demonstrate the robustness and strong generalization ability of the proposed approach.
>
---
#### [new 008] Learning Personalised Human Internal Cognition from External Expressive Behaviours for Real Personality Recognition
- **分类: cs.CV**

- **简介: 该论文旨在解决传统基于外部行为的实人性格识别（RPR）方法性能不足的问题，提出通过模拟个性化内在认知从易访问的外部行为数据中学习的方法，构建了2D图神经网络进行特征编码与推理，实现了高效的人类内在认知模拟。**

- **链接: [http://arxiv.org/pdf/2508.00205v1](http://arxiv.org/pdf/2508.00205v1)**

> **作者:** Xiangyu Kong; Hengde Zhu; Haoqin Sun; Zhihao Guo; Jiayan Gu; Xinyi Ni; Wei Zhang; Shizhe Liu; Siyang Song
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Automatic real personality recognition (RPR) aims to evaluate human real personality traits from their expressive behaviours. However, most existing solutions generally act as external observers to infer observers' personality impressions based on target individuals' expressive behaviours, which significantly deviate from their real personalities and consistently lead to inferior recognition performance. Inspired by the association between real personality and human internal cognition underlying the generation of expressive behaviours, we propose a novel RPR approach that efficiently simulates personalised internal cognition from easy-accessible external short audio-visual behaviours expressed by the target individual. The simulated personalised cognition, represented as a set of network weights that enforce the personalised network to reproduce the individual-specific facial reactions, is further encoded as a novel graph containing two-dimensional node and edge feature matrices, with a novel 2D Graph Neural Network (2D-GNN) proposed for inferring real personality traits from it. To simulate real personality-related cognition, an end-to-end strategy is designed to jointly train our cognition simulation, 2D graph construction, and personality recognition modules.
>
---
#### [new 009] Honey Classification using Hyperspectral Imaging and Machine Learning
- **分类: cs.CV**

- **简介: 该论文提出了一种基于HPSI和机器学习的方法，用于自动分类蜂蜜的来源。解决了蜂蜜植物起源识别的问题，通过数据预处理、特征提取（LDA）、模型训练（SVM/KNN）实现高效分类。**

- **链接: [http://arxiv.org/pdf/2508.00361v1](http://arxiv.org/pdf/2508.00361v1)**

> **作者:** Mokhtar A. Al-Awadhi; Ratnadeep R. Deshmukh
>
> **摘要:** In this paper, we propose a machine learning-based method for automatically classifying honey botanical origins. Dataset preparation, feature extraction, and classification are the three main steps of the proposed method. We use a class transformation method in the dataset preparation phase to maximize the separability across classes. The feature extraction phase employs the Linear Discriminant Analysis (LDA) technique for extracting relevant features and reducing the number of dimensions. In the classification phase, we use Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) models to classify the extracted features of honey samples into their botanical origins. We evaluate our system using a standard honey hyperspectral imaging (HSI) dataset. Experimental findings demonstrate that the proposed system produces state-of-the-art results on this dataset, achieving the highest classification accuracy of 95.13% for hyperspectral image-based classification and 92.80% for hyperspectral instance-based classification.
>
---
#### [new 010] EPANet: Efficient Path Aggregation Network for Underwater Fish Detection
- **分类: cs.CV**

- **简介: 该论文旨在解决水中鱼检测任务中的低分辨率、背景干扰及目标与环境视觉相似性问题，提出EPANet通过路径聚合特征金字塔（EPA-FPN）和多尺度短路径瓶颈（MS-DDSP）结构，提升特征融合效率与检测精度，实现轻量化部署。**

- **链接: [http://arxiv.org/pdf/2508.00528v1](http://arxiv.org/pdf/2508.00528v1)**

> **作者:** Jinsong Yang; Zeyuan Hu; Yichen Li
>
> **摘要:** Underwater fish detection (UFD) remains a challenging task in computer vision due to low object resolution, significant background interference, and high visual similarity between targets and surroundings. Existing approaches primarily focus on local feature enhancement or incorporate complex attention mechanisms to highlight small objects, often at the cost of increased model complexity and reduced efficiency. To address these limitations, we propose an efficient path aggregation network (EPANet), which leverages complementary feature integration to achieve accurate and lightweight UFD. EPANet consists of two key components: an efficient path aggregation feature pyramid network (EPA-FPN) and a multi-scale diverse-division short path bottleneck (MS-DDSP bottleneck). The EPA-FPN introduces long-range skip connections across disparate scales to improve semantic-spatial complementarity, while cross-layer fusion paths are adopted to enhance feature integration efficiency. The MS-DDSP bottleneck extends the conventional bottleneck structure by introducing finer-grained feature division and diverse convolutional operations, thereby increasing local feature diversity and representation capacity. Extensive experiments on benchmark UFD datasets demonstrate that EPANet outperforms state-of-the-art methods in terms of detection accuracy and inference speed, while maintaining comparable or even lower parameter complexity.
>
---
#### [new 011] Can Large Pretrained Depth Estimation Models Help With Image Dehazing?
- **分类: cs.CV**

- **简介: 该论文旨在探讨如何利用预训练深度模型解决图像去雾问题，研究其泛化能力并提出融合模块以提升跨场景适配性。任务为解决图像去雾难题，工作包括分析预训练深度特征的一致性及构建RGB-D融合模块。**

- **链接: [http://arxiv.org/pdf/2508.00698v1](http://arxiv.org/pdf/2508.00698v1)**

> **作者:** Hongfei Zhang; Kun Zhou; Ruizheng Wu; Jiangbo Lu
>
> **备注:** Submitted to AAAI2026
>
> **摘要:** Image dehazing remains a challenging problem due to the spatially varying nature of haze in real-world scenes. While existing methods have demonstrated the promise of large-scale pretrained models for image dehazing, their architecture-specific designs hinder adaptability across diverse scenarios with different accuracy and efficiency requirements. In this work, we systematically investigate the generalization capability of pretrained depth representations-learned from millions of diverse images-for image dehazing. Our empirical analysis reveals that the learned deep depth features maintain remarkable consistency across varying haze levels. Building on this insight, we propose a plug-and-play RGB-D fusion module that seamlessly integrates with diverse dehazing architectures. Extensive experiments across multiple benchmarks validate both the effectiveness and broad applicability of our approach.
>
---
#### [new 012] On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI
- **分类: cs.CV; cs.CL**

- **简介: 该论文探讨多模态临床AI任务中文本偏见的潜在风险，提出Selective Modality Shifting（SMS）方法以量化模型对各模态的依赖性，评估六种VLMs在MIMIC-CXR和FairVLMed等数据集上的性能差异及文本输入的依赖关系，揭示了文本信息仍占主导地位的现象。**

- **链接: [http://arxiv.org/pdf/2508.00171v1](http://arxiv.org/pdf/2508.00171v1)**

> **作者:** David Restrepo; Ira Ktena; Maria Vakalopoulou; Stergios Christodoulidis; Enzo Ferrante
>
> **备注:** Accepted to MICCAI 2025 1st Workshop on Multimodal Large Language Models (MLLMs) in Clinical Practice
>
> **摘要:** Clinical decision-making relies on the integrated analysis of medical images and the associated clinical reports. While Vision-Language Models (VLMs) can offer a unified framework for such tasks, they can exhibit strong biases toward one modality, frequently overlooking critical visual cues in favor of textual information. In this work, we introduce Selective Modality Shifting (SMS), a perturbation-based approach to quantify a model's reliance on each modality in binary classification tasks. By systematically swapping images or text between samples with opposing labels, we expose modality-specific biases. We assess six open-source VLMs-four generalist models and two fine-tuned for medical data-on two medical imaging datasets with distinct modalities: MIMIC-CXR (chest X-ray) and FairVLMed (scanning laser ophthalmoscopy). By assessing model performance and the calibration of every model in both unperturbed and perturbed settings, we reveal a marked dependency on text input, which persists despite the presence of complementary visual information. We also perform a qualitative attention-based analysis which further confirms that image content is often overshadowed by text details. Our findings highlight the importance of designing and evaluating multimodal medical models that genuinely integrate visual and textual cues, rather than relying on single-modality signals.
>
---
#### [new 013] LesiOnTime -- Joint Temporal and Clinical Modeling for Small Breast Lesion Segmentation in Longitudinal DCE-MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LesiOnTime，旨在利用DCE-MRI的3D时空建模与BIRADS评分，在小乳腺结节分割中融合临床和时间信息，通过动态时间优先注意力模块与一致性约束损失提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.00496v1](http://arxiv.org/pdf/2508.00496v1)**

> **作者:** Mohammed Kamran; Maria Bernathova; Raoul Varga; Christian Singer; Zsuzsanna Bago-Horvath; Thomas Helbich; Georg Langs; Philipp Seeböck
>
> **摘要:** Accurate segmentation of small lesions in Breast Dynamic Contrast-Enhanced MRI (DCE-MRI) is critical for early cancer detection, especially in high-risk patients. While recent deep learning methods have advanced lesion segmentation, they primarily target large lesions and neglect valuable longitudinal and clinical information routinely used by radiologists. In real-world screening, detecting subtle or emerging lesions requires radiologists to compare across timepoints and consider previous radiology assessments, such as the BI-RADS score. We propose LesiOnTime, a novel 3D segmentation approach that mimics clinical diagnostic workflows by jointly leveraging longitudinal imaging and BIRADS scores. The key components are: (1) a Temporal Prior Attention (TPA) block that dynamically integrates information from previous and current scans; and (2) a BI-RADS Consistency Regularization (BCR) loss that enforces latent space alignment for scans with similar radiological assessments, thus embedding domain knowledge into the training process. Evaluated on a curated in-house longitudinal dataset of high-risk patients with DCE-MRI, our approach outperforms state-of-the-art single-timepoint and longitudinal baselines by 5% in terms of Dice. Ablation studies demonstrate that both TPA and BCR contribute complementary performance gains. These results highlight the importance of incorporating temporal and clinical context for reliable early lesion segmentation in real-world breast cancer screening. Our code is publicly available at https://github.com/cirmuw/LesiOnTime
>
---
#### [new 014] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于持续学习任务，旨在解决传统视觉语言模型在重复性任务中忽略语言指令的问题，通过融合视觉投影器与专家推荐策略进行适配性改进。**

- **链接: [http://arxiv.org/pdf/2508.00260v1](http://arxiv.org/pdf/2508.00260v1)**

> **作者:** Hyundong Jin; Hyung Jin Chang; Eunwoo Kim
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Continual learning enables pre-trained generative vision-language models (VLMs) to incorporate knowledge from new tasks without retraining data from previous ones. Recent methods update a visual projector to translate visual information for new tasks, connecting pre-trained vision encoders with large language models. However, such adjustments may cause the models to prioritize visual inputs over language instructions, particularly learning tasks with repetitive types of textual instructions. To address the neglect of language instructions, we propose a novel framework that grounds the translation of visual information on instructions for language models. We introduce a mixture of visual projectors, each serving as a specialized visual-to-language translation expert based on the given instruction context to adapt to new tasks. To avoid using experts for irrelevant instruction contexts, we propose an expert recommendation strategy that reuses experts for tasks similar to those previously learned. Additionally, we introduce expert pruning to alleviate interference from the use of experts that cumulatively activated in previous tasks. Extensive experiments on diverse vision-language tasks demonstrate that our method outperforms existing continual learning approaches by generating instruction-following responses.
>
---
#### [new 015] CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective
- **分类: cs.CV**

- **简介: 该论文提出了一种基于统一时空视图的协作感知方法（CoST），解决了传统多代理融合与时间同步化方法效率与准确性的不足，通过将空间与时间信息整合为统一视图实现高效特征传输和多源融合，适用于复杂场景感知。**

- **链接: [http://arxiv.org/pdf/2508.00359v1](http://arxiv.org/pdf/2508.00359v1)**

> **作者:** Zongheng Tang; Yi Liu; Yifan Sun; Yulu Gao; Jinyu Chen; Runsheng Xu; Si Liu
>
> **备注:** ICCV25 (Highlight)
>
> **摘要:** Collaborative perception shares information among different agents and helps solving problems that individual agents may face, e.g., occlusions and small sensing range. Prior methods usually separate the multi-agent fusion and multi-time fusion into two consecutive steps. In contrast, this paper proposes an efficient collaborative perception that aggregates the observations from different agents (space) and different times into a unified spatio-temporal space simultanesouly. The unified spatio-temporal space brings two benefits, i.e., efficient feature transmission and superior feature fusion. 1) Efficient feature transmission: each static object yields a single observation in the spatial temporal space, and thus only requires transmission only once (whereas prior methods re-transmit all the object features multiple times). 2) superior feature fusion: merging the multi-agent and multi-time fusion into a unified spatial-temporal aggregation enables a more holistic perspective, thereby enhancing perception performance in challenging scenarios. Consequently, our Collaborative perception with Spatio-temporal Transformer (CoST) gains improvement in both efficiency and accuracy. Notably, CoST is not tied to any specific method and is compatible with a majority of previous methods, enhancing their accuracy while reducing the transmission bandwidth.
>
---
#### [new 016] GV-VAD : Exploring Video Generation for Weakly-Supervised Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在探索基于生成式视频的弱监督视频异常检测框架（GV-VAD），解决现实世界中数据稀缺性带来的性能瓶颈，通过合成视频增强训练数据并优化损失策略，提升模型泛化能力与效率。**

- **链接: [http://arxiv.org/pdf/2508.00312v1](http://arxiv.org/pdf/2508.00312v1)**

> **作者:** Suhang Cai; Xiaohao Peng; Chong Wang; Xiaojie Cai; Jiangbo Qian
>
> **摘要:** Video anomaly detection (VAD) plays a critical role in public safety applications such as intelligent surveillance. However, the rarity, unpredictability, and high annotation cost of real-world anomalies make it difficult to scale VAD datasets, which limits the performance and generalization ability of existing models. To address this challenge, we propose a generative video-enhanced weakly-supervised video anomaly detection (GV-VAD) framework that leverages text-conditioned video generation models to produce semantically controllable and physically plausible synthetic videos. These virtual videos are used to augment training data at low cost. In addition, a synthetic sample loss scaling strategy is utilized to control the influence of generated synthetic samples for efficient training. The experiments show that the proposed framework outperforms state-of-the-art methods on UCF-Crime datasets. The code is available at https://github.com/Sumutan/GV-VAD.git.
>
---
#### [new 017] Wukong Framework for Not Safe For Work Detection in Text-to-Image systems
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文旨在解决文本到图像生成中的NSFW内容检测问题，提出基于扩散模型的Wukong框架，通过早期阶段的中间输出和预训练交叉注意力层实现高效安全检测，优化传统文本/图像过滤方案的效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.00591v1](http://arxiv.org/pdf/2508.00591v1)**

> **作者:** Mingrui Liu; Sixiao Zhang; Cheng Long
>
> **备注:** Under review
>
> **摘要:** Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency.
>
---
#### [new 018] Towards Robust Semantic Correspondence: A Benchmark and Insights
- **分类: cs.CV**

- **简介: 该论文旨在解决图像语义对应鲁棒性不足的问题，通过建立多场景基准评估其性能，发现大模型提升整体鲁棒性但细调效果下降，DINO等模型表现优异，强调任务特定增强策略的重要性。**

- **链接: [http://arxiv.org/pdf/2508.00272v1](http://arxiv.org/pdf/2508.00272v1)**

> **作者:** Wenyue Chong
>
> **摘要:** Semantic correspondence aims to identify semantically meaningful relationships between different images and is a fundamental challenge in computer vision. It forms the foundation for numerous tasks such as 3D reconstruction, object tracking, and image editing. With the progress of large-scale vision models, semantic correspondence has achieved remarkable performance in controlled and high-quality conditions. However, the robustness of semantic correspondence in challenging scenarios is much less investigated. In this work, we establish a novel benchmark for evaluating semantic correspondence in adverse conditions. The benchmark dataset comprises 14 distinct challenging scenarios that reflect commonly encountered imaging issues, including geometric distortion, image blurring, digital artifacts, and environmental occlusion. Through extensive evaluations, we provide several key insights into the robustness of semantic correspondence approaches: (1) All existing methods suffer from noticeable performance drops under adverse conditions; (2) Using large-scale vision models can enhance overall robustness, but fine-tuning on these models leads to a decline in relative robustness; (3) The DINO model outperforms the Stable Diffusion in relative robustness, and their fusion achieves better absolute robustness; Moreover, We evaluate common robustness enhancement strategies for semantic correspondence and find that general data augmentations are ineffective, highlighting the need for task-specific designs. These results are consistent across both our dataset and real-world benchmarks.
>
---
#### [new 019] Punching Bag vs. Punching Person: Motion Transferability in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在探索动作识别模型在跨情境（如"punching person"）中传递高阶运动概念的可行性，通过合成与真实数据集验证，发现多模态模型对细节动作不如粗粒度动作敏感，且空间-时间依赖性影响显著，最终提出改进方法以提升跨场景适应性。**

- **链接: [http://arxiv.org/pdf/2508.00085v1](http://arxiv.org/pdf/2508.00085v1)**

> **作者:** Raiyaan Abdullah; Jared Claypoole; Michael Cogswell; Ajay Divakaran; Yogesh Rawat
>
> **备注:** Accepted to ICCV 2025 main conference
>
> **摘要:** Action recognition models demonstrate strong generalization, but can they effectively transfer high-level motion concepts across diverse contexts, even within similar distributions? For example, can a model recognize the broad action "punching" when presented with an unseen variation such as "punching person"? To explore this, we introduce a motion transferability framework with three datasets: (1) Syn-TA, a synthetic dataset with 3D object motions; (2) Kinetics400-TA; and (3) Something-Something-v2-TA, both adapted from natural video datasets. We evaluate 13 state-of-the-art models on these benchmarks and observe a significant drop in performance when recognizing high-level actions in novel contexts. Our analysis reveals: 1) Multimodal models struggle more with fine-grained unknown actions than with coarse ones; 2) The bias-free Syn-TA proves as challenging as real-world datasets, with models showing greater performance drops in controlled settings; 3) Larger models improve transferability when spatial cues dominate but struggle with intensive temporal reasoning, while reliance on object and background cues hinders generalization. We further explore how disentangling coarse and fine motions can improve recognition in temporally challenging datasets. We believe this study establishes a crucial benchmark for assessing motion transferability in action recognition. Datasets and relevant code: https://github.com/raiyaan-abdullah/Motion-Transfer.
>
---
#### [new 020] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating
- **分类: cs.CV**

- **简介: 该论文提出了一种基于未校准衍射光栅的光谱敏感度估计方法，解决了传统参考目标依赖的问题，实现了对相机灵敏度与光栅参数的高效计算。**

- **链接: [http://arxiv.org/pdf/2508.00330v1](http://arxiv.org/pdf/2508.00330v1)**

> **作者:** Lilika Makabe; Hiroaki Santo; Fumio Okura; Michael S. Brown; Yasuyuki Matsushita
>
> **摘要:** This paper introduces a practical and accurate calibration method for camera spectral sensitivity using a diffraction grating. Accurate calibration of camera spectral sensitivity is crucial for various computer vision tasks, including color correction, illumination estimation, and material analysis. Unlike existing approaches that require specialized narrow-band filters or reference targets with known spectral reflectances, our method only requires an uncalibrated diffraction grating sheet, readily available off-the-shelf. By capturing images of the direct illumination and its diffracted pattern through the grating sheet, our method estimates both the camera spectral sensitivity and the diffraction grating parameters in a closed-form manner. Experiments on synthetic and real-world data demonstrate that our method outperforms conventional reference target-based methods, underscoring its effectiveness and practicality.
>
---
#### [new 021] AutoDebias: Automated Framework for Debiasing Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文旨在开发一个自动化框架AutoDebias，解决文本到图像（T2I）模型中因未提及的偏见而产生的性别/种族歧视问题。其工作包括：利用视觉语言模型识别偏见模式，生成可接受的替代提示，结合CLIP训练优化输出，提升模型对多重交互性偏见的适应能力，并验证其在25种场景下的有效性，准确率达91.6%，显著降低偏见输出。**

- **链接: [http://arxiv.org/pdf/2508.00445v1](http://arxiv.org/pdf/2508.00445v1)**

> **作者:** Hongyi Cai; Mohammad Mahdinur Rahman; Mingkang Dong; Jie Li; Muxin Pu; Zhili Fang; Yinan Peng; Hanjun Luo; Yang Liu
>
> **摘要:** Text-to-Image (T2I) models generate high-quality images from text prompts but often exhibit unintended social biases, such as gender or racial stereotypes, even when these attributes are not mentioned. Existing debiasing methods work well for simple or well-known cases but struggle with subtle or overlapping biases. We propose AutoDebias, a framework that automatically identifies and mitigates harmful biases in T2I models without prior knowledge of specific bias types. Specifically, AutoDebias leverages vision-language models to detect biased visual patterns and constructs fairness guides by generating inclusive alternative prompts that reflect balanced representations. These guides drive a CLIP-guided training process that promotes fairer outputs while preserving the original model's image quality and diversity. Unlike existing methods, AutoDebias effectively addresses both subtle stereotypes and multiple interacting biases. We evaluate the framework on a benchmark covering over 25 bias scenarios, including challenging cases where multiple biases occur simultaneously. AutoDebias detects harmful patterns with 91.6% accuracy and reduces biased outputs from 90% to negligible levels, while preserving the visual fidelity of the original model.
>
---
#### [new 022] $MV_{Hybrid}$: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models
- **分类: cs.CV; cs.AI; cs.CE; cs.LG**

- **简介: 该论文提出了一种混合架构$MV_{Hybrid}$，用于病理图像的时空转录组预测任务，解决传统ViT难以捕捉低频形态学特征的问题。通过结合状态空间模型（SSM）与Vision Transformer（ViT），实验验证其在基因表达预测、分类、生存预测等下游任务中优于ViT，并展示其在临床场景中的优越性。**

- **链接: [http://arxiv.org/pdf/2508.00383v1](http://arxiv.org/pdf/2508.00383v1)**

> **作者:** Won June Cho; Hongjun Yoon; Daeky Jeong; Hyeongyeol Lim; Yosep Chong
>
> **备注:** Accepted (Oral) in MICCAI 2025 COMPAYL Workshop
>
> **摘要:** Spatial transcriptomics reveals gene expression patterns within tissue context, enabling precision oncology applications such as treatment response prediction, but its high cost and technical complexity limit clinical adoption. Predicting spatial gene expression (biomarkers) from routine histopathology images offers a practical alternative, yet current vision foundation models (VFMs) in pathology based on Vision Transformer (ViT) backbones perform below clinical standards. Given that VFMs are already trained on millions of diverse whole slide images, we hypothesize that architectural innovations beyond ViTs may better capture the low-frequency, subtle morphological patterns correlating with molecular phenotypes. By demonstrating that state space models initialized with negative real eigenvalues exhibit strong low-frequency bias, we introduce $MV_{Hybrid}$, a hybrid backbone architecture combining state space models (SSMs) with ViT. We compare five other different backbone architectures for pathology VFMs, all pretrained on identical colorectal cancer datasets using the DINOv2 self-supervised learning method. We evaluate all pretrained models using both random split and leave-one-study-out (LOSO) settings of the same biomarker dataset. In LOSO evaluation, $MV_{Hybrid}$ achieves 57% higher correlation than the best-performing ViT and shows 43% smaller performance degradation compared to random split in gene expression prediction, demonstrating superior performance and robustness, respectively. Furthermore, $MV_{Hybrid}$ shows equal or better downstream performance in classification, patch retrieval, and survival prediction tasks compared to that of ViT, showing its promise as a next-generation pathology VFM backbone. Our code is publicly available at: https://github.com/deepnoid-ai/MVHybrid.
>
---
#### [new 023] Controllable Pedestrian Video Editing for Multi-View Driving Scenarios via Motion Sequence
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文旨在解决自动驾驶中行人检测模型在多视角场景下的鲁棒性不足问题，提出一种基于运动序列控制的视频剪辑框架，通过融合视频补全与人体姿态引导实现跨视角行人编辑，提升视觉真实性和时空一致性。**

- **链接: [http://arxiv.org/pdf/2508.00299v1](http://arxiv.org/pdf/2508.00299v1)**

> **作者:** Danzhen Fu; Jiagao Hu; Daiguo Zhou; Fei Wang; Zepeng Wang; Wenhua Liao
>
> **备注:** ICCV 2025 Workshop (HiGen)
>
> **摘要:** Pedestrian detection models in autonomous driving systems often lack robustness due to insufficient representation of dangerous pedestrian scenarios in training datasets. To address this limitation, we present a novel framework for controllable pedestrian video editing in multi-view driving scenarios by integrating video inpainting and human motion control techniques. Our approach begins by identifying pedestrian regions of interest across multiple camera views, expanding detection bounding boxes with a fixed ratio, and resizing and stitching these regions into a unified canvas while preserving cross-view spatial relationships. A binary mask is then applied to designate the editable area, within which pedestrian editing is guided by pose sequence control conditions. This enables flexible editing functionalities, including pedestrian insertion, replacement, and removal. Extensive experiments demonstrate that our framework achieves high-quality pedestrian editing with strong visual realism, spatiotemporal coherence, and cross-view consistency. These results establish the proposed method as a robust and versatile solution for multi-view pedestrian video generation, with broad potential for applications in data augmentation and scenario simulation in autonomous driving.
>
---
#### [new 024] Steering Guidance for Personalized Text-to-Image Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文旨在解决扩散模型个性化生成中的目标分布与知识保留之间的平衡问题，提出通过弱模型结合动态权重插值进行个性化引导的方法，有效提升生成质量并整合多种优化策略。**

- **链接: [http://arxiv.org/pdf/2508.00319v1](http://arxiv.org/pdf/2508.00319v1)**

> **作者:** Sunghyun Park; Seokeon Choi; Hyoungwoo Park; Sungrack Yun
>
> **备注:** ICCV 2025
>
> **摘要:** Personalizing text-to-image diffusion models is crucial for adapting the pre-trained models to specific target concepts, enabling diverse image generation. However, fine-tuning with few images introduces an inherent trade-off between aligning with the target distribution (e.g., subject fidelity) and preserving the broad knowledge of the original model (e.g., text editability). Existing sampling guidance methods, such as classifier-free guidance (CFG) and autoguidance (AG), fail to effectively guide the output toward well-balanced space: CFG restricts the adaptation to the target distribution, while AG compromises text alignment. To address these limitations, we propose personalization guidance, a simple yet effective method leveraging an unlearned weak model conditioned on a null text prompt. Moreover, our method dynamically controls the extent of unlearning in a weak model through weight interpolation between pre-trained and fine-tuned models during inference. Unlike existing guidance methods, which depend solely on guidance scales, our method explicitly steers the outputs toward a balanced latent space without additional computational overhead. Experimental results demonstrate that our proposed guidance can improve text alignment and target distribution fidelity, integrating seamlessly with various fine-tuning strategies.
>
---
#### [new 025] Object-Centric Cropping for Visual Few-Shot Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉少样本分类任务，旨在解决因图像模糊性导致的性能下降问题。通过引入对象局部位置信息并结合Segment Anything模型，显著提升分类精度，仅需像素级标注即可实现有效改进。**

- **链接: [http://arxiv.org/pdf/2508.00218v1](http://arxiv.org/pdf/2508.00218v1)**

> **作者:** Aymane Abdali; Bartosz Boguslawski; Lucas Drumetz; Vincent Gripon
>
> **摘要:** In the domain of Few-Shot Image Classification, operating with as little as one example per class, the presence of image ambiguities stemming from multiple objects or complex backgrounds can significantly deteriorate performance. Our research demonstrates that incorporating additional information about the local positioning of an object within its image markedly enhances classification across established benchmarks. More importantly, we show that a significant fraction of the improvement can be achieved through the use of the Segment Anything Model, requiring only a pixel of the object of interest to be pointed out, or by employing fully unsupervised foreground object extraction methods.
>
---
#### [new 026] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于对象计数与图像生成控制任务，旨在解决文本到图像生成中的对象数量估算问题。通过引入"卡方地图"目标和混合监督策略，YOLO-Count实现了不同iable架构下的精确计数和量控制。**

- **链接: [http://arxiv.org/pdf/2508.00728v1](http://arxiv.org/pdf/2508.00728v1)**

> **作者:** Guanning Zeng; Xiang Zhang; Zirui Wang; Haiyang Xu; Zeyuan Chen; Bingnan Li; Zhuowen Tu
>
> **备注:** ICCV 2025
>
> **摘要:** We propose YOLO-Count, a differentiable open-vocabulary object counting model that tackles both general counting challenges and enables precise quantity control for text-to-image (T2I) generation. A core contribution is the 'cardinality' map, a novel regression target that accounts for variations in object size and spatial distribution. Leveraging representation alignment and a hybrid strong-weak supervision scheme, YOLO-Count bridges the gap between open-vocabulary counting and T2I generation control. Its fully differentiable architecture facilitates gradient-based optimization, enabling accurate object count estimation and fine-grained guidance for generative models. Extensive experiments demonstrate that YOLO-Count achieves state-of-the-art counting accuracy while providing robust and effective quantity control for T2I systems.
>
---
#### [new 027] A Novel Modeling Framework and Data Product for Extended VIIRS-like Artificial Nighttime Light Image Reconstruction (1986-2024)
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出了基于两阶段框架的新型人工夜间光重建方法，解决了传统VIIRS数据覆盖不足与结构缺失问题，开发了中国1986-2024年的EVAL产品，显著提升了时空一致性与精度，为遥感研究提供了新资源。**

- **链接: [http://arxiv.org/pdf/2508.00590v1](http://arxiv.org/pdf/2508.00590v1)**

> **作者:** Yihe Tian; Kwan Man Cheng; Zhengbo Zhang; Tao Zhang; Suju Li; Dongmei Yan; Bing Xu
>
> **摘要:** Artificial Night-Time Light (NTL) remote sensing is a vital proxy for quantifying the intensity and spatial distribution of human activities. Although the NPP-VIIRS sensor provides high-quality NTL observations, its temporal coverage, which begins in 2012, restricts long-term time-series studies that extend to earlier periods. Despite the progress in extending VIIRS-like NTL time-series, current methods still suffer from two significant shortcomings: the underestimation of light intensity and the structural omission. To overcome these limitations, we propose a novel reconstruction framework consisting of a two-stage process: construction and refinement. The construction stage features a Hierarchical Fusion Decoder (HFD) designed to enhance the fidelity of the initial reconstruction. The refinement stage employs a Dual Feature Refiner (DFR), which leverages high-resolution impervious surface masks to guide and enhance fine-grained structural details. Based on this framework, we developed the Extended VIIRS-like Artificial Nighttime Light (EVAL) product for China, extending the standard data record backwards by 26 years to begin in 1986. Quantitative evaluation shows that EVAL significantly outperforms existing state-of-the-art products, boosting the $\text{R}^2$ from 0.68 to 0.80 while lowering the RMSE from 1.27 to 0.99. Furthermore, EVAL exhibits excellent temporal consistency and maintains a high correlation with socioeconomic parameters, confirming its reliability for long-term analysis. The resulting EVAL dataset provides a valuable new resource for the research community and is publicly available at https://doi.org/10.11888/HumanNat.tpdc.302930.
>
---
#### [new 028] Analyze-Prompt-Reason: A Collaborative Agent-Based Framework for Multi-Image Vision-Language Reasoning
- **分类: cs.CV; cs.MA; I.2; I.2.7**

- **简介: 该论文提出了一种基于协作的多图像视觉推理框架，解决跨模态推理问题，通过语言引擎生成任务适配提示和视觉推理器完成多模态任务，实现了自动化训练与通用性，评估表明LVLM可有效处理复杂任务并达到近天花板性能。**

- **链接: [http://arxiv.org/pdf/2508.00356v1](http://arxiv.org/pdf/2508.00356v1)**

> **作者:** Angelos Vlachos; Giorgos Filandrianos; Maria Lymperaiou; Nikolaos Spanos; Ilias Mitsouras; Vasileios Karampinis; Athanasios Voulodimos
>
> **摘要:** We present a Collaborative Agent-Based Framework for Multi-Image Reasoning. Our approach tackles the challenge of interleaved multimodal reasoning across diverse datasets and task formats by employing a dual-agent system: a language-based PromptEngineer, which generates context-aware, task-specific prompts, and a VisionReasoner, a large vision-language model (LVLM) responsible for final inference. The framework is fully automated, modular, and training-free, enabling generalization across classification, question answering, and free-form generation tasks involving one or multiple input images. We evaluate our method on 18 diverse datasets from the 2025 MIRAGE Challenge (Track A), covering a broad spectrum of visual reasoning tasks including document QA, visual comparison, dialogue-based understanding, and scene-level inference. Our results demonstrate that LVLMs can effectively reason over multiple images when guided by informative prompts. Notably, Claude 3.7 achieves near-ceiling performance on challenging tasks such as TQA (99.13% accuracy), DocVQA (96.87%), and MMCoQA (75.28 ROUGE-L). We also explore how design choices-such as model selection, shot count, and input length-influence the reasoning performance of different LVLMs.
>
---
#### [new 029] GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration
- **分类: cs.CV**

- **简介: 该论文提出GeoExplorer，用于主动地理定位目标（AGL），解决传统基于距离的RL方法在未知环境中的鲁棒性不足问题，通过好奇心驱动的奖励机制实现目标无关、多样化的探索。**

- **链接: [http://arxiv.org/pdf/2508.00152v1](http://arxiv.org/pdf/2508.00152v1)**

> **作者:** Li Mi; Manon Bechaz; Zeming Chen; Antoine Bosselut; Devis Tuia
>
> **备注:** ICCV 2025. Project page at https://limirs.github.io/GeoExplorer/
>
> **摘要:** Active Geo-localization (AGL) is the task of localizing a goal, represented in various modalities (e.g., aerial images, ground-level images, or text), within a predefined search area. Current methods approach AGL as a goal-reaching reinforcement learning (RL) problem with a distance-based reward. They localize the goal by implicitly learning to minimize the relative distance from it. However, when distance estimation becomes challenging or when encountering unseen targets and environments, the agent exhibits reduced robustness and generalization ability due to the less reliable exploration strategy learned during training. In this paper, we propose GeoExplorer, an AGL agent that incorporates curiosity-driven exploration through intrinsic rewards. Unlike distance-based rewards, our curiosity-driven reward is goal-agnostic, enabling robust, diverse, and contextually relevant exploration based on effective environment modeling. These capabilities have been proven through extensive experiments across four AGL benchmarks, demonstrating the effectiveness and generalization ability of GeoExplorer in diverse settings, particularly in localizing unfamiliar targets and environments.
>
---
#### [new 030] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决视觉导航中图像作为目标时的高效定位问题，构建基于增量式3D-Gaussian（3DGS）的框架，通过改进的离散空间匹配算法实现3D-aware定位，同时支持自由视图场景并可部署于移动设备。**

- **链接: [http://arxiv.org/pdf/2508.00823v1](http://arxiv.org/pdf/2508.00823v1)**

> **作者:** Wenxuan Guo; Xiuwei Xu; Hang Yin; Ziwei Wang; Jianjiang Feng; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to ICCV 2025. Project page: https://gwxuan.github.io/IGL-Nav/
>
> **摘要:** Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view image-goal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose. Project page: https://gwxuan.github.io/IGL-Nav/.
>
---
#### [new 031] Sortblock: Similarity-Aware Feature Reuse for Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于生成式AI模型加速任务，旨在解决扩散模型因高延迟而受限的问题。提出Sortblock通过动态特征复用机制和轻量预测，实现2×推理提速，优化了特征重复率与误差控制。**

- **链接: [http://arxiv.org/pdf/2508.00412v1](http://arxiv.org/pdf/2508.00412v1)**

> **作者:** Hanqi Chen; Xu Zhang; Xiaoliu Guan; Lielin Jiang; Guanzhong Wang; Zeyu Chen; Yi Liu
>
> **摘要:** Diffusion Transformers (DiTs) have demonstrated remarkable generative capabilities, particularly benefiting from Transformer architectures that enhance visual and artistic fidelity. However, their inherently sequential denoising process results in high inference latency, limiting their deployment in real-time scenarios. Existing training-free acceleration approaches typically reuse intermediate features at fixed timesteps or layers, overlooking the evolving semantic focus across denoising stages and Transformer blocks.To address this, we propose Sortblock, a training-free inference acceleration framework that dynamically caches block-wise features based on their similarity across adjacent timesteps. By ranking the evolution of residuals, Sortblock adaptively determines a recomputation ratio, selectively skipping redundant computations while preserving generation quality. Furthermore, we incorporate a lightweight linear prediction mechanism to reduce accumulated errors in skipped blocks.Extensive experiments across various tasks and DiT architectures demonstrate that Sortblock achieves over 2$\times$ inference speedup with minimal degradation in output quality, offering an effective and generalizable solution for accelerating diffusion-based generative models.
>
---
#### [new 032] Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis
- **分类: cs.CV; cs.AI; cs.CE; cs.LG**

- **简介: 该论文旨在改进海上焊接缺陷检测，解决传统NDT和神经网络模型在细节缺陷识别及可解释性方面的局限性，通过Adapt-WeldNet优化模型架构与参数，结合DDIA框架提升系统透明度与可靠性，推动自动化焊接检测系统的可信性。**

- **链接: [http://arxiv.org/pdf/2508.00381v1](http://arxiv.org/pdf/2508.00381v1)**

> **作者:** Kamal Basha S; Athira Nambiar
>
> **摘要:** Weld defect detection is crucial for ensuring the safety and reliability of piping systems in the oil and gas industry, especially in challenging marine and offshore environments. Traditional non-destructive testing (NDT) methods often fail to detect subtle or internal defects, leading to potential failures and costly downtime. Furthermore, existing neural network-based approaches for defect classification frequently rely on arbitrarily selected pretrained architectures and lack interpretability, raising safety concerns for deployment. To address these challenges, this paper introduces ``Adapt-WeldNet", an adaptive framework for welding defect detection that systematically evaluates various pre-trained architectures, transfer learning strategies, and adaptive optimizers to identify the best-performing model and hyperparameters, optimizing defect detection and providing actionable insights. Additionally, a novel Defect Detection Interpretability Analysis (DDIA) framework is proposed to enhance system transparency. DDIA employs Explainable AI (XAI) techniques, such as Grad-CAM and LIME, alongside domain-specific evaluations validated by certified ASNT NDE Level II professionals. Incorporating a Human-in-the-Loop (HITL) approach and aligning with the principles of Trustworthy AI, DDIA ensures the reliability, fairness, and accountability of the defect detection system, fostering confidence in automated decisions through expert validation. By improving both performance and interpretability, this work enhances trust, safety, and reliability in welding defect detection systems, supporting critical operations in offshore and marine environments.
>
---
#### [new 033] SparseRecon: Neural Implicit Surface Reconstruction from Sparse Views with Feature and Depth Consistencies
- **分类: cs.CV**

- **简介: 该论文提出了一种基于稀疏RGB图像的神经隐式表面重建方法，解决了从有限视角重建3D结构的问题。通过引入特征一致性损失和不确定性引导深度约束，有效缓解了视图不一致带来的鲁棒性不足和几何细节丢失问题，实现了高精度稀疏输入场景的重建。**

- **链接: [http://arxiv.org/pdf/2508.00366v1](http://arxiv.org/pdf/2508.00366v1)**

> **作者:** Liang Han; Xu Zhang; Haichuan Song; Kanle Shi; Yu-Shen Liu; Zhizhong Han
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Surface reconstruction from sparse views aims to reconstruct a 3D shape or scene from few RGB images. The latest methods are either generalization-based or overfitting-based. However, the generalization-based methods do not generalize well on views that were unseen during training, while the reconstruction quality of overfitting-based methods is still limited by the limited geometry clues. To address this issue, we propose SparseRecon, a novel neural implicit reconstruction method for sparse views with volume rendering-based feature consistency and uncertainty-guided depth constraint. Firstly, we introduce a feature consistency loss across views to constrain the neural implicit field. This design alleviates the ambiguity caused by insufficient consistency information of views and ensures completeness and smoothness in the reconstruction results. Secondly, we employ an uncertainty-guided depth constraint to back up the feature consistency loss in areas with occlusion and insignificant features, which recovers geometry details for better reconstruction quality. Experimental results demonstrate that our method outperforms the state-of-the-art methods, which can produce high-quality geometry with sparse-view input, especially in the scenarios with small overlapping views. Project page: https://hanl2010.github.io/SparseRecon/.
>
---
#### [new 034] Rethinking Backbone Design for Lightweight 3D Object Detection in LiDAR
- **分类: cs.CV**

- **简介: 该论文旨在解决LiDAR下3D对象检测的轻量化与精度平衡问题，通过引入Dense Backbone结合PillarNet优势，实现点云数据下高效、低延迟的检测，首次构建适合3D场景的密集层架构。**

- **链接: [http://arxiv.org/pdf/2508.00744v1](http://arxiv.org/pdf/2508.00744v1)**

> **作者:** Adwait Chandorkar; Hasan Tercan; Tobias Meisen
>
> **备注:** accepted at the Embedded Vision Workshop ICCV 2025
>
> **摘要:** Recent advancements in LiDAR-based 3D object detection have significantly accelerated progress toward the realization of fully autonomous driving in real-world environments. Despite achieving high detection performance, most of the approaches still rely on a VGG-based or ResNet-based backbone for feature exploration, which increases the model complexity. Lightweight backbone design is well-explored for 2D object detection, but research on 3D object detection still remains limited. In this work, we introduce Dense Backbone, a lightweight backbone that combines the benefits of high processing speed, lightweight architecture, and robust detection accuracy. We adapt multiple SoTA 3d object detectors, such as PillarNet, with our backbone and show that with our backbone, these models retain most of their detection capability at a significantly reduced computational cost. To our knowledge, this is the first dense-layer-based backbone tailored specifically for 3D object detection from point cloud data. DensePillarNet, our adaptation of PillarNet, achieves a 29% reduction in model parameters and a 28% reduction in latency with just a 2% drop in detection accuracy on the nuScenes test set. Furthermore, Dense Backbone's plug-and-play design allows straightforward integration into existing architectures, requiring no modifications to other network components.
>
---
#### [new 035] Minimum Data, Maximum Impact: 20 annotated samples for explainable lung nodule classification
- **分类: cs.CV**

- **简介: 该论文旨在解决医疗影像分析中因缺乏大规模属性标注数据导致的可解释性模型应用受限问题，通过生成式模型合成属性标注数据（仅用20个LIDC-IDRI样本训练）提升模型性能，实现目标预测精度提升。**

- **链接: [http://arxiv.org/pdf/2508.00639v1](http://arxiv.org/pdf/2508.00639v1)**

> **作者:** Luisa Gallée; Catharina Silvia Lisson; Christoph Gerhard Lisson; Daniela Drees; Felix Weig; Daniel Vogele; Meinrad Beer; Michael Götz
>
> **备注:** Accepted at iMIMIC - Interpretability of Machine Intelligence in Medical Image Computing workshop MICCAI 2025 Medical Image Computing and Computer Assisted Intervention
>
> **摘要:** Classification models that provide human-interpretable explanations enhance clinicians' trust and usability in medical image diagnosis. One research focus is the integration and prediction of pathology-related visual attributes used by radiologists alongside the diagnosis, aligning AI decision-making with clinical reasoning. Radiologists use attributes like shape and texture as established diagnostic criteria and mirroring these in AI decision-making both enhances transparency and enables explicit validation of model outputs. However, the adoption of such models is limited by the scarcity of large-scale medical image datasets annotated with these attributes. To address this challenge, we propose synthesizing attribute-annotated data using a generative model. We enhance the Diffusion Model with attribute conditioning and train it using only 20 attribute-labeled lung nodule samples from the LIDC-IDRI dataset. Incorporating its generated images into the training of an explainable model boosts performance, increasing attribute prediction accuracy by 13.4% and target prediction accuracy by 1.8% compared to training with only the small real attribute-annotated dataset. This work highlights the potential of synthetic data to overcome dataset limitations, enhancing the applicability of explainable models in medical image analysis.
>
---
#### [new 036] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition
- **分类: cs.CV**

- **简介: 该论文旨在解决传统单一模态系统在多模态整合中的性能瓶颈，提出了一种质量引导的混合专家框架（QME），通过引入伪质量损失和分数三元组损失优化评分分布，提升了全人生物特征识别的鲁棒性和准确性，有效应对模型间相似度域失配与数据质量差异等问题。**

- **链接: [http://arxiv.org/pdf/2508.00053v1](http://arxiv.org/pdf/2508.00053v1)**

> **作者:** Jie Zhu; Yiyang Su; Minchul Kim; Anil Jain; Xiaoming Liu
>
> **备注:** Accepted to ICCV 2025. 11 pages, 5 figures
>
> **摘要:** Whole-body biometric recognition is a challenging multimodal task that integrates various biometric modalities, including face, gait, and body. This integration is essential for overcoming the limitations of unimodal systems. Traditionally, whole-body recognition involves deploying different models to process multiple modalities, achieving the final outcome by score-fusion (e.g., weighted averaging of similarity matrices from each model). However, these conventional methods may overlook the variations in score distributions of individual modalities, making it challenging to improve final performance. In this work, we present \textbf{Q}uality-guided \textbf{M}ixture of score-fusion \textbf{E}xperts (QME), a novel framework designed for improving whole-body biometric recognition performance through a learnable score-fusion strategy using a Mixture of Experts (MoE). We introduce a novel pseudo-quality loss for quality estimation with a modality-specific Quality Estimator (QE), and a score triplet loss to improve the metric performance. Extensive experiments on multiple whole-body biometric datasets demonstrate the effectiveness of our proposed approach, achieving state-of-the-art results across various metrics compared to baseline methods. Our method is effective for multimodal and multi-model, addressing key challenges such as model misalignment in the similarity score domain and variability in data quality.
>
---
#### [new 037] HiPrune: Training-Free Visual Token Pruning via Hierarchical Attention in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出HiPrune，是一种训练自由且模型无关的视觉-语言模型（VLM）中的token剪枝框架，解决传统方法依赖特殊token或需任务特定训练的问题。通过层次注意力结构识别中层与深层特征，选择锚点、缓冲区和全局总结token，实现仅减少33.3% tokens和99.5%精度，同时降低FLOPs和延迟，适用于多种ViT-VLM架构。**

- **链接: [http://arxiv.org/pdf/2508.00553v1](http://arxiv.org/pdf/2508.00553v1)**

> **作者:** Jizhihui Liu; Feiyi Du; Guangdao Zhu; Niu Lian; Jun Li; Bin Chen
>
> **摘要:** Vision-Language Models (VLMs) encode images into lengthy sequences of visual tokens, leading to excessive computational overhead and limited inference efficiency. While prior efforts prune or merge tokens to address this issue, they often rely on special tokens (e.g., CLS) or require task-specific training, hindering scalability across architectures. In this paper, we propose HiPrune, a training-free and model-agnostic token Pruning framework that exploits the Hierarchical attention structure within vision encoders. We identify that middle layers attend to object-centric regions, while deep layers capture global contextual features. Based on this observation, HiPrune selects three types of informative tokens: (1) Anchor tokens with high attention in object-centric layers, (2) Buffer tokens adjacent to anchors for spatial continuity, and (3) Register tokens with strong attention in deep layers for global summarization. Our method requires no retraining and integrates seamlessly with any ViT-based VLM. Extensive experiments on LLaVA-1.5, LLaVA-NeXT, and Qwen2.5-VL demonstrate that HiPrune achieves state-of-the-art pruning performance, preserving up to 99.3% task accuracy with only 33.3% tokens, and maintaining 99.5% accuracy with just 11.1% tokens. Meanwhile, it reduces inference FLOPs and latency by up to 9$\times$, showcasing strong generalization across models and tasks. Code is available at https://github.com/Danielement321/HiPrune.
>
---
#### [new 038] World Consistency Score: A Unified Metric for Video Generation Quality
- **分类: cs.CV**

- **简介: 该论文提出World Consistency Score（WCS），旨在建立统一的视频生成质量评估指标，解决传统方法仅关注视觉或提示对齐的问题，通过整合对象保持性、关系稳定性、因果合规性及闪烁惩罚等四个子指标，并结合学习权重进行综合评分，验证其与人类评价的一致性。**

- **链接: [http://arxiv.org/pdf/2508.00144v1](http://arxiv.org/pdf/2508.00144v1)**

> **作者:** Akshat Rakheja; Aarsh Ashdhir; Aryan Bhattacharjee; Vanshika Sharma
>
> **备注:** 27 pages, 1 figure
>
> **摘要:** We introduce World Consistency Score (WCS), a novel unified evaluation metric for generative video models that emphasizes internal world consistency of the generated videos. WCS integrates four interpretable sub-components - object permanence, relation stability, causal compliance, and flicker penalty - each measuring a distinct aspect of temporal and physical coherence in a video. These submetrics are combined via a learned weighted formula to produce a single consistency score that aligns with human judgments. We detail the motivation for WCS in the context of existing video evaluation metrics, formalize each submetric and how it is computed with open-source tools (trackers, action recognizers, CLIP embeddings, optical flow), and describe how the weights of the WCS combination are trained using human preference data. We also outline an experimental validation blueprint: using benchmarks like VBench-2.0, EvalCrafter, and LOVE to test WCS's correlation with human evaluations, performing sensitivity analyses, and comparing WCS against established metrics (FVD, CLIPScore, VBench, FVMD). The proposed WCS offers a comprehensive and interpretable framework for evaluating video generation models on their ability to maintain a coherent "world" over time, addressing gaps left by prior metrics focused only on visual fidelity or prompt alignment.
>
---
#### [new 039] Exploring Fourier Prior and Event Collaboration for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文探讨了低光图像增强方法，旨在通过傅里叶优先与事件协同技术实现可见性恢复和结构细化。研究解决了传统方法缺乏模态优势的问题，设计了两阶段流程（恢复+细化）并开发了对比损失机制，实现了对负样本的模拟与结构优化，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.00308v1](http://arxiv.org/pdf/2508.00308v1)**

> **作者:** Chunyan She; Fujun Han; Chengyu Fang; Shukai Duan; Lidan Wang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** The event camera, benefiting from its high dynamic range and low latency, provides performance gain for low-light image enhancement. Unlike frame-based cameras, it records intensity changes with extremely high temporal resolution, capturing sufficient structure information. Currently, existing event-based methods feed a frame and events directly into a single model without fully exploiting modality-specific advantages, which limits their performance. Therefore, by analyzing the role of each sensing modality, the enhancement pipeline is decoupled into two stages: visibility restoration and structure refinement. In the first stage, we design a visibility restoration network with amplitude-phase entanglement by rethinking the relationship between amplitude and phase components in Fourier space. In the second stage, a fusion strategy with dynamic alignment is proposed to mitigate the spatial mismatch caused by the temporal resolution discrepancy between two sensing modalities, aiming to refine the structure information of the image enhanced by the visibility restoration network. In addition, we utilize spatial-frequency interpolation to simulate negative samples with diverse illumination, noise and artifact degradations, thereby developing a contrastive loss that encourages the model to learn discriminative representations. Experiments demonstrate that the proposed method outperforms state-of-the-art models.
>
---
#### [new 040] GECO: Geometrically Consistent Embedding with Lightspeed Inference
- **分类: cs.CV**

- **简介: 该论文旨在解决视觉基础模型在几何信息捕获上的不足，通过几何一致性和最优传输框架提升特征学习效果，实现更准确的语义区分和性能优化。**

- **链接: [http://arxiv.org/pdf/2508.00746v1](http://arxiv.org/pdf/2508.00746v1)**

> **作者:** Regine Hartwig; Dominik Muhle; Riccardo Marin; Daniel Cremers
>
> **摘要:** Recent advances in feature learning have shown that self-supervised vision foundation models can capture semantic correspondences but often lack awareness of underlying 3D geometry. GECO addresses this gap by producing geometrically coherent features that semantically distinguish parts based on geometry (e.g., left/right eyes, front/back legs). We propose a training framework based on optimal transport, enabling supervision beyond keypoints, even under occlusions and disocclusions. With a lightweight architecture, GECO runs at 30 fps, 98.2% faster than prior methods, while achieving state-of-the-art performance on PFPascal, APK, and CUB, improving PCK by 6.0%, 6.2%, and 4.1%, respectively. Finally, we show that PCK alone is insufficient to capture geometric quality and introduce new metrics and insights for more geometry-aware feature learning. Link to project page: https://reginehartwig.github.io/publications/geco/
>
---
#### [new 041] The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决复杂环境下的视觉-惯性跟踪问题，通过Monado SLAM数据集提供实测序列，弥补现有方法在高动态、低纹理等场景中的不足，推动VIO/SLAM技术发展。**

- **链接: [http://arxiv.org/pdf/2508.00088v1](http://arxiv.org/pdf/2508.00088v1)**

> **作者:** Mateo de Mayo; Daniel Cremers; Taihú Pire
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Humanoid robots and mixed reality headsets benefit from the use of head-mounted sensors for tracking. While advancements in visual-inertial odometry (VIO) and simultaneous localization and mapping (SLAM) have produced new and high-quality state-of-the-art tracking systems, we show that these are still unable to gracefully handle many of the challenging settings presented in the head-mounted use cases. Common scenarios like high-intensity motions, dynamic occlusions, long tracking sessions, low-textured areas, adverse lighting conditions, saturation of sensors, to name a few, continue to be covered poorly by existing datasets in the literature. In this way, systems may inadvertently overlook these essential real-world issues. To address this, we present the Monado SLAM dataset, a set of real sequences taken from multiple virtual reality headsets. We release the dataset under a permissive CC BY 4.0 license, to drive advancements in VIO/SLAM research and development.
>
---
#### [new 042] Zero-Shot Anomaly Detection with Dual-Branch Prompt Learning
- **分类: cs.CV**

- **简介: 该论文研究了零样本异常检测，提出了基于双分支提示学习的PILOT框架，解决了领域迁移问题，通过动态整合提示机制和伪标签更新策略实现高效检测。**

- **链接: [http://arxiv.org/pdf/2508.00777v1](http://arxiv.org/pdf/2508.00777v1)**

> **作者:** Zihan Wang; Samira Ebrahimi Kahou; Narges Armanfard
>
> **备注:** Accepted at BMVC 2025
>
> **摘要:** Zero-shot anomaly detection (ZSAD) enables identifying and localizing defects in unseen categories by relying solely on generalizable features rather than requiring any labeled examples of anomalies. However, existing ZSAD methods, whether using fixed or learned prompts, struggle under domain shifts because their training data are derived from limited training domains and fail to generalize to new distributions. In this paper, we introduce PILOT, a framework designed to overcome these challenges through two key innovations: (1) a novel dual-branch prompt learning mechanism that dynamically integrates a pool of learnable prompts with structured semantic attributes, enabling the model to adaptively weight the most relevant anomaly cues for each input image; and (2) a label-free test-time adaptation strategy that updates the learnable prompt parameters using high-confidence pseudo-labels from unlabeled test data. Extensive experiments on 13 industrial and medical benchmarks demonstrate that PILOT achieves state-of-the-art performance in both anomaly detection and localization under domain shift.
>
---
#### [new 043] PointGauss: Point Cloud-Guided Multi-Object Segmentation for Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出点云引导多目标分割框架PointGauss，解决实时3D分割中的初始化慢、多视角不一致等问题，通过点云驱动的Gaussian实例提取和GPU加速2D渲染实现高效分割，同时开发了DesktopObjects-360数据集以提升性能。**

- **链接: [http://arxiv.org/pdf/2508.00259v1](http://arxiv.org/pdf/2508.00259v1)**

> **作者:** Wentao Sun; Hanqing Xu; Quanyun Wu; Dedong Zhang; Yiping Chen; Lingfei Ma; John S. Zelek; Jonathan Li
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** We introduce PointGauss, a novel point cloud-guided framework for real-time multi-object segmentation in Gaussian Splatting representations. Unlike existing methods that suffer from prolonged initialization and limited multi-view consistency, our approach achieves efficient 3D segmentation by directly parsing Gaussian primitives through a point cloud segmentation-driven pipeline. The key innovation lies in two aspects: (1) a point cloud-based Gaussian primitive decoder that generates 3D instance masks within 1 minute, and (2) a GPU-accelerated 2D mask rendering system that ensures multi-view consistency. Extensive experiments demonstrate significant improvements over previous state-of-the-art methods, achieving performance gains of 1.89 to 31.78% in multi-view mIoU, while maintaining superior computational efficiency. To address the limitations of current benchmarks (single-object focus, inconsistent 3D evaluation, small scale, and partial coverage), we present DesktopObjects-360, a novel comprehensive dataset for 3D segmentation in radiance fields, featuring: (1) complex multi-object scenes, (2) globally consistent 2D annotations, (3) large-scale training data (over 27 thousand 2D masks), (4) full 360{\deg} coverage, and (5) 3D evaluation masks.
>
---
#### [new 044] Representation Shift: Unifying Token Compression with FlashAttention
- **分类: cs.CV**

- **简介: 该论文旨在解决Transformer模型因任务复杂化导致的token压缩与FlashAttention内存优化之间的兼容性问题，提出"Representation Shift"作为训练自由的指标，通过融合token压缩与FlashAttention，有效实现了跨模型（Transformer/CNN/状态空间）的高效token压缩与计算加速，实验验证了其在视频-文本检索和问答任务中的显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.00367v1](http://arxiv.org/pdf/2508.00367v1)**

> **作者:** Joonmyung Choi; Sanghyeok Lee; Byungoh Ko; Eunseo Kim; Jihyung Kil; Hyunwoo J. Kim
>
> **备注:** International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Transformers have demonstrated remarkable success across vision, language, and video. Yet, increasing task complexity has led to larger models and more tokens, raising the quadratic cost of self-attention and the overhead of GPU memory access. To reduce the computation cost of self-attention, prior work has proposed token compression techniques that drop redundant or less informative tokens. Meanwhile, fused attention kernels such as FlashAttention have been developed to alleviate memory overhead by avoiding attention map construction and its associated I/O to HBM. This, however, makes it incompatible with most training-free token compression methods, which rely on attention maps to determine token importance. Here, we propose Representation Shift, a training-free, model-agnostic metric that measures the degree of change in each token's representation. This seamlessly integrates token compression with FlashAttention, without attention maps or retraining. Our method further generalizes beyond Transformers to CNNs and state space models. Extensive experiments show that Representation Shift enables effective token compression compatible with FlashAttention, yielding significant speedups of up to 5.5% and 4.4% in video-text retrieval and video QA, respectively. Code is available at https://github.com/mlvlab/Representation-Shift.
>
---
#### [new 045] SU-ESRGAN: Semantic and Uncertainty-Aware ESRGAN for Super-Resolution of Satellite and Drone Imagery with Fine-Tuning for Cross Domain Evaluation
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文提出SU-ESRGAN，解决GAN SR缺乏语义一致性与像素不确定性的缺陷，集成E2E-GAN、DeepLabv3分割损失与Monte Carlo Dropout，提升卫星图像SR性能，适用于无人机等场景。**

- **链接: [http://arxiv.org/pdf/2508.00750v1](http://arxiv.org/pdf/2508.00750v1)**

> **作者:** Prerana Ramkumar
>
> **摘要:** Generative Adversarial Networks (GANs) have achieved realistic super-resolution (SR) of images however, they lack semantic consistency and per-pixel confidence, limiting their credibility in critical remote sensing applications such as disaster response, urban planning and agriculture. This paper introduces Semantic and Uncertainty-Aware ESRGAN (SU-ESRGAN), the first SR framework designed for satellite imagery to integrate the ESRGAN, segmentation loss via DeepLabv3 for class detail preservation and Monte Carlo dropout to produce pixel-wise uncertainty maps. The SU-ESRGAN produces results (PSNR, SSIM, LPIPS) comparable to the Baseline ESRGAN on aerial imagery. This novel model is valuable in satellite systems or UAVs that use wide field-of-view (FoV) cameras, trading off spatial resolution for coverage. The modular design allows integration in UAV data pipelines for on-board or post-processing SR to enhance imagery resulting due to motion blur, compression and sensor limitations. Further, the model is fine-tuned to evaluate its performance on cross domain applications. The tests are conducted on two drone based datasets which differ in altitude and imaging perspective. Performance evaluation of the fine-tuned models show a stronger adaptation to the Aerial Maritime Drone Dataset, whose imaging characteristics align with the training data, highlighting the importance of domain-aware training in SR-applications.
>
---
#### [new 046] Uncertainty-Aware Likelihood Ratio Estimation for Pixel-Wise Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文旨在解决像素级出错检测任务中的不确定性和稀有类错误问题，提出基于证据分类器的不确定性感知似然比估计方法，通过概率分布捕捉不确定性并优化模型性能，实验验证其在复杂场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00587v1](http://arxiv.org/pdf/2508.00587v1)**

> **作者:** Marc Hölle; Walter Kellermann; Vasileios Belagiannis
>
> **备注:** Accepted at ICCVW 2025, 11 pages, 4 figures
>
> **摘要:** Semantic segmentation models trained on known object classes often fail in real-world autonomous driving scenarios by confidently misclassifying unknown objects. While pixel-wise out-of-distribution detection can identify unknown objects, existing methods struggle in complex scenes where rare object classes are often confused with truly unknown objects. We introduce an uncertainty-aware likelihood ratio estimation method that addresses these limitations. Our approach uses an evidential classifier within a likelihood ratio test to distinguish between known and unknown pixel features from a semantic segmentation model, while explicitly accounting for uncertainty. Instead of producing point estimates, our method outputs probability distributions that capture uncertainty from both rare training examples and imperfect synthetic outliers. We show that by incorporating uncertainty in this way, outlier exposure can be leveraged more effectively. Evaluated on five standard benchmark datasets, our method achieves the lowest average false positive rate (2.5%) among state-of-the-art while maintaining high average precision (90.91%) and incurring only negligible computational overhead. Code is available at https://github.com/glasbruch/ULRE.
>
---
#### [new 047] Reducing the gap between general purpose data and aerial images in concentrated solar power plants
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文旨在解决通用数据集难以泛化到复杂太阳能场的问题，提出AerialCSP虚拟数据集作为替代方案，通过生成仿真实验数据降低手动标注成本，提升CSP相关视觉任务的性能。**

- **链接: [http://arxiv.org/pdf/2508.00440v1](http://arxiv.org/pdf/2508.00440v1)**

> **作者:** M. A. Pérez-Cutiño; J. Valverde; J. Capitán; J. M. Díaz-Báñez
>
> **摘要:** In the context of Concentrated Solar Power (CSP) plants, aerial images captured by drones present a unique set of challenges. Unlike urban or natural landscapes commonly found in existing datasets, solar fields contain highly reflective surfaces, and domain-specific elements that are uncommon in traditional computer vision benchmarks. As a result, machine learning models trained on generic datasets struggle to generalize to this setting without extensive retraining and large volumes of annotated data. However, collecting and labeling such data is costly and time-consuming, making it impractical for rapid deployment in industrial applications. To address this issue, we propose a novel approach: the creation of AerialCSP, a virtual dataset that simulates aerial imagery of CSP plants. By generating synthetic data that closely mimic real-world conditions, our objective is to facilitate pretraining of models before deployment, significantly reducing the need for extensive manual labeling. Our main contributions are threefold: (1) we introduce AerialCSP, a high-quality synthetic dataset for aerial inspection of CSP plants, providing annotated data for object detection and image segmentation; (2) we benchmark multiple models on AerialCSP, establishing a baseline for CSP-related vision tasks; and (3) we demonstrate that pretraining on AerialCSP significantly improves real-world fault detection, particularly for rare and small defects, reducing the need for extensive manual labeling. AerialCSP is made publicly available at https://mpcutino.github.io/aerialcsp/.
>
---
#### [new 048] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于任务为训练自由的类别净化，旨在解决因类冗余和视觉-语言混杂导致的分类激活质量差问题，提出FreeCP框架以改进OVSS性能，通过净化类别和修正冗余/模糊性提升分割精度。**

- **链接: [http://arxiv.org/pdf/2508.00557v1](http://arxiv.org/pdf/2508.00557v1)**

> **作者:** Qi Chen; Lingxiao Yang; Yun Chen; Nailong Zhao; Jianhuang Lai; Jie Shao; Xiaohua Xie
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Fine-tuning pre-trained vision-language models has emerged as a powerful approach for enhancing open-vocabulary semantic segmentation (OVSS). However, the substantial computational and resource demands associated with training on large datasets have prompted interest in training-free methods for OVSS. Existing training-free approaches primarily focus on modifying model architectures and generating prototypes to improve segmentation performance. However, they often neglect the challenges posed by class redundancy, where multiple categories are not present in the current test image, and visual-language ambiguity, where semantic similarities among categories create confusion in class activation. These issues can lead to suboptimal class activation maps and affinity-refined activation maps. Motivated by these observations, we propose FreeCP, a novel training-free class purification framework designed to address these challenges. FreeCP focuses on purifying semantic categories and rectifying errors caused by redundancy and ambiguity. The purified class representations are then leveraged to produce final segmentation predictions. We conduct extensive experiments across eight benchmarks to validate FreeCP's effectiveness. Results demonstrate that FreeCP, as a plug-and-play module, significantly boosts segmentation performance when combined with other OVSS methods.
>
---
#### [new 049] Stable at Any Speed: Speed-Driven Multi-Object Tracking with Learnable Kalman Filtering
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，解决传统基于固定坐标变换的跟踪方法在动态环境中的精度与稳定性不足问题。提出Speed-Guided Learnable Kalman Filter（SG-LKF）并结合MotionScaleNet进行不确定性建模与轨迹一致性优化，显著提升高动态场景下的跟踪效果。**

- **链接: [http://arxiv.org/pdf/2508.00358v1](http://arxiv.org/pdf/2508.00358v1)**

> **作者:** Yan Gong; Mengjun Chen; Hao Liu; Gao Yongsheng; Lei Yang; Naibang Wang; Ziying Song; Haoqun Ma
>
> **备注:** 9 pages, 7 figures, 5 tables
>
> **摘要:** Multi-object tracking (MOT) enables autonomous vehicles to continuously perceive dynamic objects, supplying essential temporal cues for prediction, behavior understanding, and safe planning. However, conventional tracking-by-detection methods typically rely on static coordinate transformations based on ego-vehicle poses, disregarding ego-vehicle speed-induced variations in observation noise and reference frame changes, which degrades tracking stability and accuracy in dynamic, high-speed scenarios. In this paper, we investigate the critical role of ego-vehicle speed in MOT and propose a Speed-Guided Learnable Kalman Filter (SG-LKF) that dynamically adapts uncertainty modeling to ego-vehicle speed, significantly improving stability and accuracy in highly dynamic scenarios. Central to SG-LKF is MotionScaleNet (MSNet), a decoupled token-mixing and channel-mixing MLP that adaptively predicts key parameters of SG-LKF. To enhance inter-frame association and trajectory continuity, we introduce a self-supervised trajectory consistency loss jointly optimized with semantic and positional constraints. Extensive experiments show that SG-LKF ranks first among all vision-based methods on KITTI 2D MOT with 79.59% HOTA, delivers strong results on KITTI 3D MOT with 82.03% HOTA, and outperforms SimpleTrack by 2.2% AMOTA on nuScenes 3D MOT.
>
---
#### [new 050] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在解决动态人-物交互补全中的理解局限性，通过融合物理先验与多区域 inpainting 技术，定义两个区域并优化噪点策略，显著提升HOI场景中的准确性与真实性。**

- **链接: [http://arxiv.org/pdf/2508.00427v1](http://arxiv.org/pdf/2508.00427v1)**

> **作者:** Seunggeun Chi; Enna Sachdeva; Pin-Hao Huang; Kwonjoon Lee
>
> **备注:** ICCV 2025 (Highlight)
>
> **摘要:** Amodal completion, which is the process of inferring the full appearance of objects despite partial occlusions, is crucial for understanding complex human-object interactions (HOI) in computer vision and robotics. Existing methods, such as those that use pre-trained diffusion models, often struggle to generate plausible completions in dynamic scenarios because they have a limited understanding of HOI. To solve this problem, we've developed a new approach that uses physical prior knowledge along with a specialized multi-regional inpainting technique designed for HOI. By incorporating physical constraints from human topology and contact information, we define two distinct regions: the primary region, where occluded object parts are most likely to be, and the secondary region, where occlusions are less probable. Our multi-regional inpainting method uses customized denoising strategies across these regions within a diffusion model. This improves the accuracy and realism of the generated completions in both their shape and visual detail. Our experimental results show that our approach significantly outperforms existing methods in HOI scenarios, moving machine perception closer to a more human-like understanding of dynamic environments. We also show that our pipeline is robust even without ground-truth contact annotations, which broadens its applicability to tasks like 3D reconstruction and novel view/pose synthesis.
>
---
#### [new 051] Leveraging Convolutional and Graph Networks for an Unsupervised Remote Sensing Labelling Tool
- **分类: cs.CV**

- **简介: 该论文提出了一种基于卷积与图神经网络的无监督遥感图像标注工具，旨在解决传统方法标注效率低、依赖预训练数据的问题。通过图像分割技术将像素划分为相似区域并利用图结构聚合信息，有效提升了标注精度和语义一致性。**

- **链接: [http://arxiv.org/pdf/2508.00506v1](http://arxiv.org/pdf/2508.00506v1)**

> **作者:** Tulsi Patel; Mark W. Jones; Thomas Redfern
>
> **备注:** Video supplement demonstrating feature-space exploration and interactive labelling is available at: https://youtu.be/GZl1ebZJgEA and is archived at https://doi.org/10.5281/zenodo.16676591
>
> **摘要:** Machine learning for remote sensing imaging relies on up-to-date and accurate labels for model training and testing. Labelling remote sensing imagery is time and cost intensive, requiring expert analysis. Previous labelling tools rely on pre-labelled data for training in order to label new unseen data. In this work, we define an unsupervised pipeline for finding and labelling geographical areas of similar context and content within Sentinel-2 satellite imagery. Our approach removes limitations of previous methods by utilising segmentation with convolutional and graph neural networks to encode a more robust feature space for image comparison. Unlike previous approaches we segment the image into homogeneous regions of pixels that are grouped based on colour and spatial similarity. Graph neural networks are used to aggregate information about the surrounding segments enabling the feature representation to encode the local neighbourhood whilst preserving its own local information. This reduces outliers in the labelling tool, allows users to label at a granular level, and allows a rotationally invariant semantic relationship at the image level to be formed within the encoding space.
>
---
#### [new 052] DBLP: Noise Bridge Consistency Distillation For Efficient And Reliable Adversarial Purification
- **分类: cs.CV**

- **简介: 该论文研究了对抗样本生成与净化的任务，提出DBLP框架解决传统方法因高计算成本而受限的问题。通过引入噪声桥一致性蒸馏和自适应语义增强，构建了高效的模型以提升鲁棒性和实时性。**

- **链接: [http://arxiv.org/pdf/2508.00552v1](http://arxiv.org/pdf/2508.00552v1)**

> **作者:** Chihan Huang; Belal Alsinglawi; Islam Al-qudah
>
> **摘要:** Recent advances in deep neural networks (DNNs) have led to remarkable success across a wide range of tasks. However, their susceptibility to adversarial perturbations remains a critical vulnerability. Existing diffusion-based adversarial purification methods often require intensive iterative denoising, severely limiting their practical deployment. In this paper, we propose Diffusion Bridge Distillation for Purification (DBLP), a novel and efficient diffusion-based framework for adversarial purification. Central to our approach is a new objective, noise bridge distillation, which constructs a principled alignment between the adversarial noise distribution and the clean data distribution within a latent consistency model (LCM). To further enhance semantic fidelity, we introduce adaptive semantic enhancement, which fuses multi-scale pyramid edge maps as conditioning input to guide the purification process. Extensive experiments across multiple datasets demonstrate that DBLP achieves state-of-the-art (SOTA) robust accuracy, superior image quality, and around 0.2s inference time, marking a significant step toward real-time adversarial purification.
>
---
#### [new 053] CLIPTime: Time-Aware Multimodal Representation Learning from Images and Text
- **分类: cs.CV; cs.LG**

- **简介: 该论文旨在解决生物生长时间动态建模问题，提出CLIPTime框架通过联合学习图像与文本嵌入并结合时间戳预测，利用合成数据集优化训练，同时引入新指标评估性能，有效捕捉生物进程并实现可解释的时间输出。**

- **链接: [http://arxiv.org/pdf/2508.00447v1](http://arxiv.org/pdf/2508.00447v1)**

> **作者:** Anju Rani; Daniel Ortiz-Arroyo; Petar Durdevic
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Understanding the temporal dynamics of biological growth is critical across diverse fields such as microbiology, agriculture, and biodegradation research. Although vision-language models like Contrastive Language Image Pretraining (CLIP) have shown strong capabilities in joint visual-textual reasoning, their effectiveness in capturing temporal progression remains limited. To address this, we propose CLIPTime, a multimodal, multitask framework designed to predict both the developmental stage and the corresponding timestamp of fungal growth from image and text inputs. Built upon the CLIP architecture, our model learns joint visual-textual embeddings and enables time-aware inference without requiring explicit temporal input during testing. To facilitate training and evaluation, we introduce a synthetic fungal growth dataset annotated with aligned timestamps and categorical stage labels. CLIPTime jointly performs classification and regression, predicting discrete growth stages alongside continuous timestamps. We also propose custom evaluation metrics, including temporal accuracy and regression error, to assess the precision of time-aware predictions. Experimental results demonstrate that CLIPTime effectively models biological progression and produces interpretable, temporally grounded outputs, highlighting the potential of vision-language models in real-world biological monitoring applications.
>
---
#### [new 054] Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了一种基于样本感知的测试时间适应框架（Sample-Aware TTA），解决传统TTA方法对出错样本性能下降的问题。通过引入重建模块和动态适应块，动态调整模型特征以应对不同样本特性，验证了其在低剂量CT去噪和T1-T2 MRI转换任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00766v1](http://arxiv.org/pdf/2508.00766v1)**

> **作者:** Irene Iele; Francesco Di Feola; Valerio Guarrasi; Paolo Soda
>
> **摘要:** Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods. Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both out-of-distribution and in-distribution samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios. The code is available at: https://github.com/cosbidev/Sample-Aware_TTA.
>
---
#### [new 055] TITAN-Guide: Taming Inference-Time AligNment for Guided Text-to-Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文探讨了通过训练自由引导优化文本-视频扩散模型的任务需求，旨在解决传统方法因高内存和控制效率不足导致的局限性，提出TITAN-Guide框架实现高效的扩散参数优化，提升T2V性能。**

- **链接: [http://arxiv.org/pdf/2508.00289v1](http://arxiv.org/pdf/2508.00289v1)**

> **作者:** Christian Simon; Masato Ishii; Akio Hayakawa; Zhi Zhong; Shusuke Takahashi; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** In the recent development of conditional diffusion models still require heavy supervised fine-tuning for performing control on a category of tasks. Training-free conditioning via guidance with off-the-shelf models is a favorable alternative to avoid further fine-tuning on the base model. However, the existing training-free guidance frameworks either have heavy memory requirements or offer sub-optimal control due to rough estimation. These shortcomings limit the applicability to control diffusion models that require intense computation, such as Text-to-Video (T2V) diffusion models. In this work, we propose Taming Inference Time Alignment for Guided Text-to-Video Diffusion Model, so-called TITAN-Guide, which overcomes memory space issues, and provides more optimal control in the guidance process compared to the counterparts. In particular, we develop an efficient method for optimizing diffusion latents without backpropagation from a discriminative guiding model. In particular, we study forward gradient descents for guided diffusion tasks with various options on directional directives. In our experiments, we demonstrate the effectiveness of our approach in efficiently managing memory during latent optimization, while previous methods fall short. Our proposed approach not only minimizes memory requirements but also significantly enhances T2V performance across a range of diffusion guidance benchmarks. Code, models, and demo are available at https://titanguide.github.io.
>
---
#### [new 056] SDMatte: Grafting Diffusion Models for Interactive Matting
- **分类: cs.CV**

- **简介: 该论文提出SDMatte，解决交互式图层中边缘细节提取不足的问题，通过扩散模型先验与可视化提示结合、多维度嵌入优化及掩码自注意力机制提升性能，验证其有效性并提供代码支持。**

- **链接: [http://arxiv.org/pdf/2508.00443v1](http://arxiv.org/pdf/2508.00443v1)**

> **作者:** Longfei Huang; Yu Liang; Hao Zhang; Jinwei Chen; Wei Dong; Lunde Chen; Wanyu Liu; Bo Li; Pengtao Jiang
>
> **备注:** Accepted at ICCV 2025, 11 pages, 4 figures
>
> **摘要:** Recent interactive matting methods have shown satisfactory performance in capturing the primary regions of objects, but they fall short in extracting fine-grained details in edge regions. Diffusion models trained on billions of image-text pairs, demonstrate exceptional capability in modeling highly complex data distributions and synthesizing realistic texture details, while exhibiting robust text-driven interaction capabilities, making them an attractive solution for interactive matting. To this end, we propose SDMatte, a diffusion-driven interactive matting model, with three key contributions. First, we exploit the powerful priors of diffusion models and transform the text-driven interaction capability into visual prompt-driven interaction capability to enable interactive matting. Second, we integrate coordinate embeddings of visual prompts and opacity embeddings of target objects into U-Net, enhancing SDMatte's sensitivity to spatial position information and opacity information. Third, we propose a masked self-attention mechanism that enables the model to focus on areas specified by visual prompts, leading to better performance. Extensive experiments on multiple datasets demonstrate the superior performance of our method, validating its effectiveness in interactive matting. Our code and model are available at https://github.com/vivoCameraResearch/SDMatte.
>
---
#### [new 057] iSafetyBench: A video-language benchmark for safety in industrial environment
- **分类: cs.CV**

- **简介: 该论文提出了一款用于工业场景的安全视频-语言评估工具（iSafetyBench），旨在解决传统视觉语言模型在复杂安全场景（常规/危险）识别上的不足。通过构建包含98+67个动作标签的基准数据集，并结合多选题进行零样本训练，评估了现有模型的性能差异。**

- **链接: [http://arxiv.org/pdf/2508.00399v1](http://arxiv.org/pdf/2508.00399v1)**

> **作者:** Raiyaan Abdullah; Yogesh Singh Rawat; Shruti Vyas
>
> **备注:** Accepted to VISION'25 - ICCV 2025 workshop
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled impressive generalization across diverse video understanding tasks under zero-shot settings. However, their capabilities in high-stakes industrial domains-where recognizing both routine operations and safety-critical anomalies is essential-remain largely underexplored. To address this gap, we introduce iSafetyBench, a new video-language benchmark specifically designed to evaluate model performance in industrial environments across both normal and hazardous scenarios. iSafetyBench comprises 1,100 video clips sourced from real-world industrial settings, annotated with open-vocabulary, multi-label action tags spanning 98 routine and 67 hazardous action categories. Each clip is paired with multiple-choice questions for both single-label and multi-label evaluation, enabling fine-grained assessment of VLMs in both standard and safety-critical contexts. We evaluate eight state-of-the-art video-language models under zero-shot conditions. Despite their strong performance on existing video benchmarks, these models struggle with iSafetyBench-particularly in recognizing hazardous activities and in multi-label scenarios. Our results reveal significant performance gaps, underscoring the need for more robust, safety-aware multimodal models for industrial applications. iSafetyBench provides a first-of-its-kind testbed to drive progress in this direction. The dataset is available at: https://github.com/raiyaan-abdullah/iSafety-Bench.
>
---
#### [new 058] HyPCV-Former: Hyperbolic Spatio-Temporal Transformer for 3D Point Cloud Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出hypcvformer，用于3D点云视频异常检测，解决传统方法在捕捉时空结构上的局限，通过超立方体空间建模事件层级与时间连续性，并创新性地采用hyperbolic multi-head self-attention机制，实现全空间直觉特征学习，取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.00473v1](http://arxiv.org/pdf/2508.00473v1)**

> **作者:** Jiaping Cao; Kangkang Zhou; Juan Du
>
> **摘要:** Video anomaly detection is a fundamental task in video surveillance, with broad applications in public safety and intelligent monitoring systems. Although previous methods leverage Euclidean representations in RGB or depth domains, such embeddings are inherently limited in capturing hierarchical event structures and spatio-temporal continuity. To address these limitations, we propose HyPCV-Former, a novel hyperbolic spatio-temporal transformer for anomaly detection in 3D point cloud videos. Our approach first extracts per-frame spatial features from point cloud sequences via point cloud extractor, and then embeds them into Lorentzian hyperbolic space, which better captures the latent hierarchical structure of events. To model temporal dynamics, we introduce a hyperbolic multi-head self-attention (HMHA) mechanism that leverages Lorentzian inner products and curvature-aware softmax to learn temporal dependencies under non-Euclidean geometry. Our method performs all feature transformations and anomaly scoring directly within full Lorentzian space rather than via tangent space approximation. Extensive experiments demonstrate that HyPCV-Former achieves state-of-the-art performance across multiple anomaly categories, with a 7\% improvement on the TIMo dataset and a 5.6\% gain on the DAD dataset compared to benchmarks. The code will be released upon paper acceptance.
>
---
#### [new 059] Cross-Dataset Semantic Segmentation Performance Analysis: Unifying NIST Point Cloud City Datasets for 3D Deep Learning
- **分类: cs.CV**

- **简介: 该论文旨在分析跨数据集的语义分割性能，解决公共安全领域点云数据的异构性与特征区分度问题，通过梯度模型与KPConv架构评估IoU指标，并提出标准化标注与多模态学习策略以提升安全特征检测能力。**

- **链接: [http://arxiv.org/pdf/2508.00822v1](http://arxiv.org/pdf/2508.00822v1)**

> **作者:** Alexander Nikitas Dimopoulos; Joseph Grasso
>
> **摘要:** This study analyzes semantic segmentation performance across heterogeneously labeled point-cloud datasets relevant to public safety applications, including pre-incident planning systems derived from lidar scans. Using NIST's Point Cloud City dataset (Enfield and Memphis collections), we investigate challenges in unifying differently labeled 3D data. Our methodology employs a graded schema with the KPConv architecture, evaluating performance through IoU metrics on safety-relevant features. Results indicate performance variability: geometrically large objects (e.g. stairs, windows) achieve higher segmentation performance, suggesting potential for navigational context, while smaller safety-critical features exhibit lower recognition rates. Performance is impacted by class imbalance and the limited geometric distinction of smaller objects in typical lidar scans, indicating limitations in detecting certain safety-relevant features using current point-cloud methods. Key identified challenges include insufficient labeled data, difficulties in unifying class labels across datasets, and the need for standardization. Potential directions include automated labeling and multi-dataset learning strategies. We conclude that reliable point-cloud semantic segmentation for public safety necessitates standardized annotation protocols and improved labeling techniques to address data heterogeneity and the detection of small, safety-critical elements.
>
---
#### [new 060] CoProU-VO: Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry
- **分类: cs.CV**

- **简介: 该论文旨在解决视觉去标定中动态物体和遮挡导致的pose估计误差问题，提出基于联合投影不确定性的CoProU-VO方法，通过视觉Transformer模型同时学习深度、不确定性及相机姿态，显著提升动态场景下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.00568v1](http://arxiv.org/pdf/2508.00568v1)**

> **作者:** Jingchao Xie; Oussema Dhaouadi; Weirong Chen; Johannes Meier; Jacques Kaiser; Daniel Cremers
>
> **备注:** Accepted for GCPR 2025. Project page: https://jchao-xie.github.io/CoProU/
>
> **摘要:** Visual Odometry (VO) is fundamental to autonomous navigation, robotics, and augmented reality, with unsupervised approaches eliminating the need for expensive ground-truth labels. However, these methods struggle when dynamic objects violate the static scene assumption, leading to erroneous pose estimations. We tackle this problem by uncertainty modeling, which is a commonly used technique that creates robust masks to filter out dynamic objects and occlusions without requiring explicit motion segmentation. Traditional uncertainty modeling considers only single-frame information, overlooking the uncertainties across consecutive frames. Our key insight is that uncertainty must be propagated and combined across temporal frames to effectively identify unreliable regions, particularly in dynamic scenes. To address this challenge, we introduce Combined Projected Uncertainty VO (CoProU-VO), a novel end-to-end approach that combines target frame uncertainty with projected reference frame uncertainty using a principled probabilistic formulation. Built upon vision transformer backbones, our model simultaneously learns depth, uncertainty estimation, and camera poses. Consequently, experiments on the KITTI and nuScenes datasets demonstrate significant improvements over previous unsupervised monocular end-to-end two-frame-based methods and exhibit strong performance in challenging highway scenes where other approaches often fail. Additionally, comprehensive ablation studies validate the effectiveness of cross-frame uncertainty propagation.
>
---
#### [new 061] Decouple before Align: Visual Disentanglement Enhances Prompt Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文探讨了视觉-文本信息不对称问题，在视觉语言模型中提出了**基于视觉分离的prompt tuning框架**，旨在解决传统PT方法因偏重上下文而缺乏对物体的关注问题。通过分离视觉与文本模态并结合拉普拉斯正则化，有效提升了任务迁移能力。**

- **链接: [http://arxiv.org/pdf/2508.00395v1](http://arxiv.org/pdf/2508.00395v1)**

> **作者:** Fei Zhang; Tianfei Zhou; Jiangchao Yao; Ya Zhang; Ivor W. Tsang; Yanfeng Wang
>
> **备注:** 16 pages, Accepted at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Prompt tuning (PT), as an emerging resource-efficient fine-tuning paradigm, has showcased remarkable effectiveness in improving the task-specific transferability of vision-language models. This paper delves into a previously overlooked information asymmetry issue in PT, where the visual modality mostly conveys more context than the object-oriented textual modality. Correspondingly, coarsely aligning these two modalities could result in the biased attention, driving the model to merely focus on the context area. To address this, we propose DAPT, an effective PT framework based on an intuitive decouple-before-align concept. First, we propose to explicitly decouple the visual modality into the foreground and background representation via exploiting coarse-and-fine visual segmenting cues, and then both of these decoupled patterns are aligned with the original foreground texts and the hand-crafted background classes, thereby symmetrically strengthening the modal alignment. To further enhance the visual concentration, we propose a visual pull-push regularization tailored for the foreground-background patterns, directing the original visual representation towards unbiased attention on the region-of-interest object. We demonstrate the power of architecture-free DAPT through few-shot learning, base-to-novel generalization, and data-efficient learning, all of which yield superior performance across prevailing benchmarks. Our code will be released at https://github.com/Ferenas/DAPT.
>
---
#### [new 062] IN2OUT: Fine-Tuning Video Inpainting Model for Video Outpainting Using Hierarchical Discriminator
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出了一种用于视频出图的任务，旨在解决扩展边界同时保持内容一致性的问题。通过引入层次化判别器和自适应损失函数，优化了视频 inpainting 模型，提升了视觉质量和全局一致性。**

- **链接: [http://arxiv.org/pdf/2508.00418v1](http://arxiv.org/pdf/2508.00418v1)**

> **作者:** Sangwoo Youn; Minji Lee; Nokap Tony Park; Yeonggyoo Jeon; Taeyoung Na
>
> **备注:** ICIP 2025. Code: https://github.com/sang-w00/IN2OUT
>
> **摘要:** Video outpainting presents a unique challenge of extending the borders while maintaining consistency with the given content. In this paper, we suggest the use of video inpainting models that excel in object flow learning and reconstruction in outpainting rather than solely generating the background as in existing methods. However, directly applying or fine-tuning inpainting models to outpainting has shown to be ineffective, often leading to blurry results. Our extensive experiments on discriminator designs reveal that a critical component missing in the outpainting fine-tuning process is a discriminator capable of effectively assessing the perceptual quality of the extended areas. To tackle this limitation, we differentiate the objectives of adversarial training into global and local goals and introduce a hierarchical discriminator that meets both objectives. Additionally, we develop a specialized outpainting loss function that leverages both local and global features of the discriminator. Fine-tuning on this adversarial loss function enhances the generator's ability to produce both visually appealing and globally coherent outpainted scenes. Our proposed method outperforms state-of-the-art methods both quantitatively and qualitatively. Supplementary materials including the demo video and the code are available in SigPort.
>
---
#### [new 063] Graph Lineages and Skeletal Graph Products
- **分类: cs.CV; cs.LG; cs.NA; math.CT; math.NA**

- **简介: 该论文旨在构建结构化图线性并开发其运算符，解决高阶数学模型构建问题，通过层次化结构实现高效类型理论和应用领域（如神经网络、多格方法）。**

- **链接: [http://arxiv.org/pdf/2508.00197v1](http://arxiv.org/pdf/2508.00197v1)**

> **作者:** Eric Mjolsness; Cory B. Scott
>
> **备注:** 42 pages. 33 Figures. Under review
>
> **摘要:** Graphs, and sequences of growing graphs, can be used to specify the architecture of mathematical models in many fields including machine learning and computational science. Here we define structured graph "lineages" (ordered by level number) that grow in a hierarchical fashion, so that: (1) the number of graph vertices and edges increases exponentially in level number; (2) bipartite graphs connect successive levels within a graph lineage and, as in multigrid methods, can constrain matrices relating successive levels; (3) using prolongation maps within a graph lineage, process-derived distance measures between graphs at successive levels can be defined; (4) a category of "graded graphs" can be defined, and using it low-cost "skeletal" variants of standard algebraic graph operations and type constructors (cross product, box product, disjoint sum, and function types) can be derived for graded graphs and hence hierarchical graph lineages; (5) these skeletal binary operators have similar but not identical algebraic and category-theoretic properties to their standard counterparts; (6) graph lineages and their skeletal product constructors can approach continuum limit objects. Additional space-efficient unary operators on graded graphs are also derived: thickening, which creates a graph lineage of multiscale graphs, and escalation to a graph lineage of search frontiers (useful as a generalization of adaptive grids and in defining "skeletal" functions). The result is an algebraic type theory for graded graphs and (hierarchical) graph lineages. The approach is expected to be well suited to defining hierarchical model architectures - "hierarchitectures" - and local sampling, search, or optimization algorithms on them. We demonstrate such application to deep neural networks (including visual and feature scale spaces) and to multigrid numerical methods.
>
---
#### [new 064] Exploring the Feasibility of Deep Learning Techniques for Accurate Gender Classification from Eye Images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文探讨了深度学习技术在性别分类中的应用，旨在解决因化妆品和伪装影响准确性的问题。通过分析眼周区域的颜色图像数据，提出了一种基于CNN的模型，并在CVBL和（Female & Male）两个数据集中验证其性能，最终证明了该模型的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00135v1](http://arxiv.org/pdf/2508.00135v1)**

> **作者:** Basna Mohammed Salih Hasan; Ramadhan J. Mstafa
>
> **备注:** 12 pages, 18 figures, 5 tables
>
> **摘要:** Gender classification has emerged as a crucial aspect in various fields, including security, human-machine interaction, surveillance, and advertising. Nonetheless, the accuracy of this classification can be influenced by factors such as cosmetics and disguise. Consequently, our study is dedicated to addressing this concern by concentrating on gender classification using color images of the periocular region. The periocular region refers to the area surrounding the eye, including the eyelids, eyebrows, and the region between them. It contains valuable visual cues that can be used to extract key features for gender classification. This paper introduces a sophisticated Convolutional Neural Network (CNN) model that utilizes color image databases to evaluate the effectiveness of the periocular region for gender classification. To validate the model's performance, we conducted tests on two eye datasets, namely CVBL and (Female and Male). The recommended architecture achieved an outstanding accuracy of 99% on the previously unused CVBL dataset while attaining a commendable accuracy of 96% with a small number of learnable parameters (7,235,089) on the (Female and Male) dataset. To ascertain the effectiveness of our proposed model for gender classification using the periocular region, we evaluated its performance through an extensive range of metrics and compared it with other state-of-the-art approaches. The results unequivocally demonstrate the efficacy of our model, thereby suggesting its potential for practical application in domains such as security and surveillance.
>
---
#### [new 065] Backdoor Attacks on Deep Learning Face Detection
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **简介: 该论文旨在研究深度学习在不寻常环境下的人脸检测漏洞，提出Object Generation Attacks和Landmark Shift Attack作为backdoor方法，解决环境变化导致的坐标预测失效问题。**

- **链接: [http://arxiv.org/pdf/2508.00620v1](http://arxiv.org/pdf/2508.00620v1)**

> **作者:** Quentin Le Roux; Yannick Teglia; Teddy Furon; Philippe Loubet-Moundi
>
> **摘要:** Face Recognition Systems that operate in unconstrained environments capture images under varying conditions,such as inconsistent lighting, or diverse face poses. These challenges require including a Face Detection module that regresses bounding boxes and landmark coordinates for proper Face Alignment. This paper shows the effectiveness of Object Generation Attacks on Face Detection, dubbed Face Generation Attacks, and demonstrates for the first time a Landmark Shift Attack that backdoors the coordinate regression task performed by face detectors. We then offer mitigations against these vulnerabilities.
>
---
#### [new 066] D3: Training-Free AI-Generated Video Detection Using Second-Order Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了一种训练无关的AI视频检测方法D3，通过第二阶动力学分析和中央差分特征，解决了传统方法对合成视频时间艺术的探索不足，验证了其在GenVideo等数据集上的优越性能。**

- **链接: [http://arxiv.org/pdf/2508.00701v1](http://arxiv.org/pdf/2508.00701v1)**

> **作者:** Chende Zheng; Ruiqi suo; Chenhao Lin; Zhengyu Zhao; Le Yang; Shuai Liu; Minghui Yang; Cong Wang; Chao Shen
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** The evolution of video generation techniques, such as Sora, has made it increasingly easy to produce high-fidelity AI-generated videos, raising public concern over the dissemination of synthetic content. However, existing detection methodologies remain limited by their insufficient exploration of temporal artifacts in synthetic videos. To bridge this gap, we establish a theoretical framework through second-order dynamical analysis under Newtonian mechanics, subsequently extending the Second-order Central Difference features tailored for temporal artifact detection. Building on this theoretical foundation, we reveal a fundamental divergence in second-order feature distributions between real and AI-generated videos. Concretely, we propose Detection by Difference of Differences (D3), a novel training-free detection method that leverages the above second-order temporal discrepancies. We validate the superiority of our D3 on 4 open-source datasets (Gen-Video, VideoPhy, EvalCrafter, VidProM), 40 subsets in total. For example, on GenVideo, D3 outperforms the previous best method by 10.39% (absolute) mean Average Precision. Additional experiments on time cost and post-processing operations demonstrate D3's exceptional computational efficiency and strong robust performance. Our code is available at https://github.com/Zig-HS/D3.
>
---
#### [new 067] Your other Left! Vision-Language Models Fail to Identify Relative Positions in Medical Images
- **分类: cs.CV**

- **简介: 该论文探讨了Vision-Language Models（VLMs）在医学图像中识别相对位置的能力不足，旨在解决临床决策中对空间关系的理解需求。通过分析现有模型性能并设计MIRP基准数据集，提出改进方案以提升该任务效果。**

- **链接: [http://arxiv.org/pdf/2508.00549v1](http://arxiv.org/pdf/2508.00549v1)**

> **作者:** Daniel Wolf; Heiko Hillenhagen; Billurvan Taskin; Alex Bäuerle; Meinrad Beer; Michael Götz; Timo Ropinski
>
> **备注:** Accepted at the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** Clinical decision-making relies heavily on understanding relative positions of anatomical structures and anomalies. Therefore, for Vision-Language Models (VLMs) to be applicable in clinical practice, the ability to accurately determine relative positions on medical images is a fundamental prerequisite. Despite its importance, this capability remains highly underexplored. To address this gap, we evaluate the ability of state-of-the-art VLMs, GPT-4o, Llama3.2, Pixtral, and JanusPro, and find that all models fail at this fundamental task. Inspired by successful approaches in computer vision, we investigate whether visual prompts, such as alphanumeric or colored markers placed on anatomical structures, can enhance performance. While these markers provide moderate improvements, results remain significantly lower on medical images compared to observations made on natural images. Our evaluations suggest that, in medical imaging, VLMs rely more on prior anatomical knowledge than on actual image content for answering relative position questions, often leading to incorrect conclusions. To facilitate further research in this area, we introduce the MIRP , Medical Imaging Relative Positioning, benchmark dataset, designed to systematically evaluate the capability to identify relative positions in medical images.
>
---
#### [new 068] Is It Really You? Exploring Biometric Verification Scenarios in Photorealistic Talking-Head Avatar Videos
- **分类: cs.CV; cs.AI; cs.CR; cs.MM**

- **简介: 该论文探讨了基于逼真头像视频的生物特征验证挑战，旨在通过GAGAvatar生成数据集和改进图注意力网络架构解决身份伪造问题，实验表明面部运动可达到80%以上AUC值。**

- **链接: [http://arxiv.org/pdf/2508.00748v1](http://arxiv.org/pdf/2508.00748v1)**

> **作者:** Laura Pedrouzo-Rodriguez; Pedro Delgado-DeRobles; Luis F. Gomez; Ruben Tolosana; Ruben Vera-Rodriguez; Aythami Morales; Julian Fierrez
>
> **备注:** Accepted at the IEEE International Joint Conference on Biometrics (IJCB 2025)
>
> **摘要:** Photorealistic talking-head avatars are becoming increasingly common in virtual meetings, gaming, and social platforms. These avatars allow for more immersive communication, but they also introduce serious security risks. One emerging threat is impersonation: an attacker can steal a user's avatar-preserving their appearance and voice-making it nearly impossible to detect its fraudulent usage by sight or sound alone. In this paper, we explore the challenge of biometric verification in such avatar-mediated scenarios. Our main question is whether an individual's facial motion patterns can serve as reliable behavioral biometrics to verify their identity when the avatar's visual appearance is a facsimile of its owner. To answer this question, we introduce a new dataset of realistic avatar videos created using a state-of-the-art one-shot avatar generation model, GAGAvatar, with genuine and impostor avatar videos. We also propose a lightweight, explainable spatio-temporal Graph Convolutional Network architecture with temporal attention pooling, that uses only facial landmarks to model dynamic facial gestures. Experimental results demonstrate that facial motion cues enable meaningful identity verification with AUC values approaching 80%. The proposed benchmark and biometric system are available for the research community in order to bring attention to the urgent need for more advanced behavioral biometric defenses in avatar-based communication systems.
>
---
#### [new 069] Privacy-Preserving Driver Drowsiness Detection with Spatial Self-Attention and Federated Learning
- **分类: cs.CV**

- **简介: 该论文提出了一种隐私保护的驱动疲劳检测框架，解决传统方法在分布式数据下的检测准确性不足与隐私泄露问题。通过融合空间自注意力（SSA）和长短期记忆（LSTM）网络提升特征提取能力，结合联邦学习技术优化模型聚合过程，同时开发了数据增强工具以应对现实场景的数据多样性。实验表明其在联邦环境中实现了89.9%的检测精度，具有潜在的应用价值。**

- **链接: [http://arxiv.org/pdf/2508.00287v1](http://arxiv.org/pdf/2508.00287v1)**

> **作者:** Tran Viet Khoa; Do Hai Son; Mohammad Abu Alsheikh; Yibeltal F Alem; Dinh Thai Hoang
>
> **摘要:** Driver drowsiness is one of the main causes of road accidents and is recognized as a leading contributor to traffic-related fatalities. However, detecting drowsiness accurately remains a challenging task, especially in real-world settings where facial data from different individuals is decentralized and highly diverse. In this paper, we propose a novel framework for drowsiness detection that is designed to work effectively with heterogeneous and decentralized data. Our approach develops a new Spatial Self-Attention (SSA) mechanism integrated with a Long Short-Term Memory (LSTM) network to better extract key facial features and improve detection performance. To support federated learning, we employ a Gradient Similarity Comparison (GSC) that selects the most relevant trained models from different operators before aggregation. This improves the accuracy and robustness of the global model while preserving user privacy. We also develop a customized tool that automatically processes video data by extracting frames, detecting and cropping faces, and applying data augmentation techniques such as rotation, flipping, brightness adjustment, and zooming. Experimental results show that our framework achieves a detection accuracy of 89.9% in the federated learning settings, outperforming existing methods under various deployment scenarios. The results demonstrate the effectiveness of our approach in handling real-world data variability and highlight its potential for deployment in intelligent transportation systems to enhance road safety through early and reliable drowsiness detection.
>
---
#### [new 070] SAMSA 2.0: Prompting Segment Anything with Spectral Angles for Hyperspectral Interactive Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出一种基于谱角提示的医学影像分割框架，解决了传统RGB-only方法无法有效融合多模态信息的问题，通过结合空间与光谱特征提升分割精度，实现了跨场景的高效泛化，达到+3.8% Dice增益。**

- **链接: [http://arxiv.org/pdf/2508.00493v1](http://arxiv.org/pdf/2508.00493v1)**

> **作者:** Alfie Roddan; Tobias Czempiel; Chi Xu; Daniel S. Elson; Stamatia Giannarou
>
> **摘要:** We present SAMSA 2.0, an interactive segmentation framework for hyperspectral medical imaging that introduces spectral angle prompting to guide the Segment Anything Model (SAM) using spectral similarity alongside spatial cues. This early fusion of spectral information enables more accurate and robust segmentation across diverse spectral datasets. Without retraining, SAMSA 2.0 achieves up to +3.8% higher Dice scores compared to RGB-only models and up to +3.1% over prior spectral fusion methods. Our approach enhances few-shot and zero-shot performance, demonstrating strong generalization in challenging low-data and noisy scenarios common in clinical imaging.
>
---
#### [new 071] DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior
- **分类: cs.CV**

- **简介: 本研究提出DPoser-X，一种基于扩散模型的3D人体姿态先验，解决复杂人体姿态建模和数据稀疏性问题，通过逆问题求解、时步缩放和掩码训练提升性能。**

- **链接: [http://arxiv.org/pdf/2508.00599v1](http://arxiv.org/pdf/2508.00599v1)**

> **作者:** Junzhe Lu; Jing Lin; Hongkun Dou; Ailing Zeng; Yue Deng; Xian Liu; Zhongang Cai; Lei Yang; Yulun Zhang; Haoqian Wang; Ziwei Liu
>
> **备注:** ICCV 2025 (oral); Code released: https://github.com/moonbow721/DPoser
>
> **摘要:** We present DPoser-X, a diffusion-based prior model for 3D whole-body human poses. Building a versatile and robust full-body human pose prior remains challenging due to the inherent complexity of articulated human poses and the scarcity of high-quality whole-body pose datasets. To address these limitations, we introduce a Diffusion model as body Pose prior (DPoser) and extend it to DPoser-X for expressive whole-body human pose modeling. Our approach unifies various pose-centric tasks as inverse problems, solving them through variational diffusion sampling. To enhance performance on downstream applications, we introduce a novel truncated timestep scheduling method specifically designed for pose data characteristics. We also propose a masked training mechanism that effectively combines whole-body and part-specific datasets, enabling our model to capture interdependencies between body parts while avoiding overfitting to specific actions. Extensive experiments demonstrate DPoser-X's robustness and versatility across multiple benchmarks for body, hand, face, and full-body pose modeling. Our model consistently outperforms state-of-the-art alternatives, establishing a new benchmark for whole-body human pose prior modeling.
>
---
#### [new 072] LAMIC: Layout-Aware Multi-Image Composition via Scalability of Multimodal Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出了一种布局感知的多图像合成框架LAMIC，解决从多个参考图生成一致图像的问题。通过引入组隔离注意力（GIA）和区域模态注意力（RMA），并结合三个评估指标，实现了零样本泛化，在ID-S、BG-S等任务上达到最佳性能。**

- **链接: [http://arxiv.org/pdf/2508.00477v1](http://arxiv.org/pdf/2508.00477v1)**

> **作者:** Yuzhuo Chen; Zehua Ma; Jianhua Wang; Kai Kang; Shunyu Yao; Weiming Zhang
>
> **备注:** 8 pages, 5 figures, 3 tables
>
> **摘要:** In controllable image synthesis, generating coherent and consistent images from multiple references with spatial layout awareness remains an open challenge. We present LAMIC, a Layout-Aware Multi-Image Composition framework that, for the first time, extends single-reference diffusion models to multi-reference scenarios in a training-free manner. Built upon the MMDiT model, LAMIC introduces two plug-and-play attention mechanisms: 1) Group Isolation Attention (GIA) to enhance entity disentanglement; and 2) Region-Modulated Attention (RMA) to enable layout-aware generation. To comprehensively evaluate model capabilities, we further introduce three metrics: 1) Inclusion Ratio (IN-R) and Fill Ratio (FI-R) for assessing layout control; and 2) Background Similarity (BG-S) for measuring background consistency. Extensive experiments show that LAMIC achieves state-of-the-art performance across most major metrics: it consistently outperforms existing multi-reference baselines in ID-S, BG-S, IN-R and AVG scores across all settings, and achieves the best DPG in complex composition tasks. These results demonstrate LAMIC's superior abilities in identity keeping, background preservation, layout control, and prompt-following, all achieved without any training or fine-tuning, showcasing strong zero-shot generalization ability. By inheriting the strengths of advanced single-reference models and enabling seamless extension to multi-image scenarios, LAMIC establishes a new training-free paradigm for controllable multi-image composition. As foundation models continue to evolve, LAMIC's performance is expected to scale accordingly. Our implementation is available at: https://github.com/Suchenl/LAMIC.
>
---
#### [new 073] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出了一种基于点云对齐和物理约束的扩散生成框架，解决刚体物体生成与物理合理性问题。通过SDF对齐引导反向扩散，引入非穿透性约束和分类感知机制，提升模型生成的物理一致性。**

- **链接: [http://arxiv.org/pdf/2508.00558v1](http://arxiv.org/pdf/2508.00558v1)**

> **作者:** Jens U. Kreber; Joerg Stueckler
>
> **备注:** Accepted for publication at the IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Articulated objects are an important type of interactable objects in everyday environments. In this paper, we propose PhysNAP, a novel diffusion model-based approach for generating articulated objects that aligns them with partial point clouds and improves their physical plausibility. The model represents part shapes by signed distance functions (SDFs). We guide the reverse diffusion process using a point cloud alignment loss computed using the predicted SDFs. Additionally, we impose non-penetration and mobility constraints based on the part SDFs for guiding the model to generate more physically plausible objects. We also make our diffusion approach category-aware to further improve point cloud alignment if category information is available. We evaluate the generative ability and constraint consistency of samples generated with PhysNAP using the PartNet-Mobility dataset. We also compare it with an unguided baseline diffusion model and demonstrate that PhysNAP can improve constraint consistency and provides a tradeoff with generative ability.
>
---
#### [new 074] PIF-Net: Ill-Posed Prior Guided Multispectral and Hyperspectral Image Fusion via Invertible Mamba and Fusion-Aware LoRA
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决因数据不一致导致的MHIF任务中谱信息与空间细节的矛盾性问题。提出PIF-Net框架，利用逆矩阵Mamba架构和融合感知LoRA模块，动态平衡信息一致性与计算效率，实现高效融合。**

- **链接: [http://arxiv.org/pdf/2508.00453v1](http://arxiv.org/pdf/2508.00453v1)**

> **作者:** Baisong Li; Xingwang Wang; Haixiao Xu
>
> **摘要:** The goal of multispectral and hyperspectral image fusion (MHIF) is to generate high-quality images that simultaneously possess rich spectral information and fine spatial details. However, due to the inherent trade-off between spectral and spatial information and the limited availability of observations, this task is fundamentally ill-posed. Previous studies have not effectively addressed the ill-posed nature caused by data misalignment. To tackle this challenge, we propose a fusion framework named PIF-Net, which explicitly incorporates ill-posed priors to effectively fuse multispectral images and hyperspectral images. To balance global spectral modeling with computational efficiency, we design a method based on an invertible Mamba architecture that maintains information consistency during feature transformation and fusion, ensuring stable gradient flow and process reversibility. Furthermore, we introduce a novel fusion module called the Fusion-Aware Low-Rank Adaptation module, which dynamically calibrates spectral and spatial features while keeping the model lightweight. Extensive experiments on multiple benchmark datasets demonstrate that PIF-Net achieves significantly better image restoration performance than current state-of-the-art methods while maintaining model efficiency.
>
---
#### [new 075] UIS-Mamba: Exploring Mamba for Underwater Instance Segmentation via Dynamic Tree Scan and Hidden State Weaken
- **分类: cs.CV**

- **简介: 该论文研究了基于Mamba的水下实例分割任务，解决了传统方法在深度水环境下的内部连续性破坏和背景干扰问题。通过动态树扫描（DTS）和隐藏状态减弱（HSW）模块，实现了对水下场景的高效建模与信息聚焦，实验表明模型在UIIS和USIS10K上达到最优性能，且参数量较低。**

- **链接: [http://arxiv.org/pdf/2508.00421v1](http://arxiv.org/pdf/2508.00421v1)**

> **作者:** Runmin Cong; Zongji Yu; Hao Fang; Haoyan Sun; Sam Kwong
>
> **备注:** ACM MM 2025
>
> **摘要:** Underwater Instance Segmentation (UIS) tasks are crucial for underwater complex scene detection. Mamba, as an emerging state space model with inherently linear complexity and global receptive fields, is highly suitable for processing image segmentation tasks with long sequence features. However, due to the particularity of underwater scenes, there are many challenges in applying Mamba to UIS. The existing fixed-patch scanning mechanism cannot maintain the internal continuity of scanned instances in the presence of severely underwater color distortion and blurred instance boundaries, and the hidden state of the complex underwater background can also inhibit the understanding of instance objects. In this work, we propose the first Mamba-based underwater instance segmentation model UIS-Mamba, and design two innovative modules, Dynamic Tree Scan (DTS) and Hidden State Weaken (HSW), to migrate Mamba to the underwater task. DTS module maintains the continuity of the internal features of the instance objects by allowing the patches to dynamically offset and scale, thereby guiding the minimum spanning tree and providing dynamic local receptive fields. HSW module suppresses the interference of complex backgrounds and effectively focuses the information flow of state propagation to the instances themselves through the Ncut-based hidden state weakening mechanism. Experimental results show that UIS-Mamba achieves state-of-the-art performance on both UIIS and USIS10K datasets, while maintaining a low number of parameters and computational complexity. Code is available at https://github.com/Maricalce/UIS-Mamba.
>
---
#### [new 076] PMR: Physical Model-Driven Multi-Stage Restoration of Turbulent Dynamic Videos
- **分类: cs.CV**

- **简介: 该论文提出物理模型驱动的多阶段视频恢复框架（PMR），旨在解决大气湍流引起的几何畸变、运动模糊等问题。通过轻量化架构与阶段联合训练，在高湍流场景中有效抑制运动尾迹并恢复细节，具有强泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.00406v1](http://arxiv.org/pdf/2508.00406v1)**

> **作者:** Tao Wu; Jingyuan Ye; Ying Fu
>
> **摘要:** Geometric distortions and blurring caused by atmospheric turbulence degrade the quality of long-range dynamic scene videos. Existing methods struggle with restoring edge details and eliminating mixed distortions, especially under conditions of strong turbulence and complex dynamics. To address these challenges, we introduce a Dynamic Efficiency Index ($DEI$), which combines turbulence intensity, optical flow, and proportions of dynamic regions to accurately quantify video dynamic intensity under varying turbulence conditions and provide a high-dynamic turbulence training dataset. Additionally, we propose a Physical Model-Driven Multi-Stage Video Restoration ($PMR$) framework that consists of three stages: \textbf{de-tilting} for geometric stabilization, \textbf{motion segmentation enhancement} for dynamic region refinement, and \textbf{de-blurring} for quality restoration. $PMR$ employs lightweight backbones and stage-wise joint training to ensure both efficiency and high restoration quality. Experimental results demonstrate that the proposed method effectively suppresses motion trailing artifacts, restores edge details and exhibits strong generalization capability, especially in real-world scenarios characterized by high-turbulence and complex dynamics. We will make the code and datasets openly available.
>
---
#### [new 077] Semantic and Temporal Integration in Latent Diffusion Space for High-Fidelity Video Super-Resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究了在扩散空间中整合语义与时间空间引导的方法，旨在解决视频超分辨率任务中的高保真对齐和时间一致性问题。通过结合高阶语义信息和时空信息，提出SeTe-VSR方法，在保持细节完整性的同时提升视觉质量，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00471v1](http://arxiv.org/pdf/2508.00471v1)**

> **作者:** Yiwen Wang; Xinning Chai; Yuhong Zhang; Zhengxue Cheng; Jun Zhao; Rong Xie; Li Song
>
> **摘要:** Recent advancements in video super-resolution (VSR) models have demonstrated impressive results in enhancing low-resolution videos. However, due to limitations in adequately controlling the generation process, achieving high fidelity alignment with the low-resolution input while maintaining temporal consistency across frames remains a significant challenge. In this work, we propose Semantic and Temporal Guided Video Super-Resolution (SeTe-VSR), a novel approach that incorporates both semantic and temporal-spatio guidance in the latent diffusion space to address these challenges. By incorporating high-level semantic information and integrating spatial and temporal information, our approach achieves a seamless balance between recovering intricate details and ensuring temporal coherence. Our method not only preserves high-reality visual content but also significantly enhances fidelity. Extensive experiments demonstrate that SeTe-VSR outperforms existing methods in terms of detail recovery and perceptual quality, highlighting its effectiveness for complex video super-resolution tasks.
>
---
#### [new 078] DocTron-Formula: Generalized Formula Recognition in Complex and Structured Scenarios
- **分类: cs.CV**

- **简介: 该论文旨在解决复杂科学文献中数学公式识别的结构性与多样化问题，提出DocTron-Formula框架，利用通用视觉语言模型构建CSFormula数据集，并通过监督微调实现跨领域高性能识别。**

- **链接: [http://arxiv.org/pdf/2508.00311v1](http://arxiv.org/pdf/2508.00311v1)**

> **作者:** Yufeng Zhong; Zhixiong Zeng; Lei Chen; Longrong Yang; Liming Zheng; Jing Huang; Siqi Yang; Lin Ma
>
> **摘要:** Optical Character Recognition (OCR) for mathematical formula is essential for the intelligent analysis of scientific literature. However, both task-specific and general vision-language models often struggle to handle the structural diversity, complexity, and real-world variability inherent in mathematical content. In this work, we present DocTron-Formula, a unified framework built upon general vision-language models, thereby eliminating the need for specialized architectures. Furthermore, we introduce CSFormula, a large-scale and challenging dataset that encompasses multidisciplinary and structurally complex formulas at the line, paragraph, and page levels. Through straightforward supervised fine-tuning, our approach achieves state-of-the-art performance across a variety of styles, scientific domains, and complex layouts. Experimental results demonstrate that our method not only surpasses specialized models in terms of accuracy and robustness, but also establishes a new paradigm for the automated understanding of complex scientific documents.
>
---
#### [new 079] Multimodal Referring Segmentation: A Survey
- **分类: cs.CV**

- **简介: 该论文研究了多模态指代分割任务，旨在通过文本或音频中的指代表达分割目标对象在图像、视频和3D场景中，解决了基于用户指令的精确物体感知问题，并提出了一种统一的元架构及GREx方法，实现了跨模态的高效性能比较。**

- **链接: [http://arxiv.org/pdf/2508.00265v1](http://arxiv.org/pdf/2508.00265v1)**

> **作者:** Henghui Ding; Song Tang; Shuting He; Chang Liu; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** Project Page: https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation
>
> **摘要:** Multimodal referring segmentation aims to segment target objects in visual scenes, such as images, videos, and 3D scenes, based on referring expressions in text or audio format. This task plays a crucial role in practical applications requiring accurate object perception based on user instructions. Over the past decade, it has gained significant attention in the multimodal community, driven by advances in convolutional neural networks, transformers, and large language models, all of which have substantially improved multimodal perception capabilities. This paper provides a comprehensive survey of multimodal referring segmentation. We begin by introducing this field's background, including problem definitions and commonly used datasets. Next, we summarize a unified meta architecture for referring segmentation and review representative methods across three primary visual scenes, including images, videos, and 3D scenes. We further discuss Generalized Referring Expression (GREx) methods to address the challenges of real-world complexity, along with related tasks and practical applications. Extensive performance comparisons on standard benchmarks are also provided. We continually track related works at https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation.
>
---
#### [new 080] Guided Depth Map Super-Resolution via Multi-Scale Fusion U-shaped Mamba Network
- **分类: cs.CV**

- **简介: 本研究提出了一种基于多尺度融合U形Mamba网络的引导深度地图超分辨率方法，解决传统CNN在长距离依赖和全局信息建模上的局限性，通过融合残差密集通道注意力和状态空间模块，有效提升图像质量。**

- **链接: [http://arxiv.org/pdf/2508.00248v1](http://arxiv.org/pdf/2508.00248v1)**

> **作者:** Chenggang Guo; Hao Xu; XianMing Wan
>
> **摘要:** Depth map super-resolution technology aims to improve the spatial resolution of low-resolution depth maps and effectively restore high-frequency detail information. Traditional convolutional neural network has limitations in dealing with long-range dependencies and are unable to fully model the global contextual information in depth maps. Although transformer can model global dependencies, its computational complexity and memory consumption are quadratic, which significantly limits its ability to process high-resolution depth maps. In this paper, we propose a multi-scale fusion U-shaped Mamba (MSF-UM) model, a novel guided depth map super-resolution framework. The core innovation of this model is to integrate Mamba's efficient state-space modeling capabilities into a multi-scale U-shaped fusion structure guided by a color image. The structure combining the residual dense channel attention block and the Mamba state space module is designed, which combines the local feature extraction capability of the convolutional layer with the modeling advantage of the state space model for long-distance dependencies. At the same time, the model adopts a multi-scale cross-modal fusion strategy to make full use of the high-frequency texture information from the color image to guide the super-resolution process of the depth map. Compared with existing mainstream methods, the proposed MSF-UM significantly reduces the number of model parameters while achieving better reconstruction accuracy. Extensive experiments on multiple publicly available datasets validate the effectiveness of the model, especially showing excellent generalization ability in the task of large-scale depth map super-resolution.
>
---
#### [new 081] Sari Sandbox: A Virtual Retail Store Environment for Embodied AI Agents
- **分类: cs.CV**

- **简介: 该论文提出Sari Sandbox作为用于训练实体AI助手的虚拟零售环境，旨在解决传统零售场景下难以模拟人机交互和购物任务的真实度问题。通过集成高精度3D商品库、VR与VLM模型及SariBench数据集，实现了对实体动作与人类行为的基准测试。**

- **链接: [http://arxiv.org/pdf/2508.00400v1](http://arxiv.org/pdf/2508.00400v1)**

> **作者:** Janika Deborah Gajo; Gerarld Paul Merales; Jerome Escarcha; Brenden Ashley Molina; Gian Nartea; Emmanuel G. Maminta; Juan Carlos Roldan; Rowel O. Atienza
>
> **备注:** 14 pages, accepted in ICCV 2025 Workshop on RetailVision
>
> **摘要:** We present Sari Sandbox, a high-fidelity, photorealistic 3D retail store simulation for benchmarking embodied agents against human performance in shopping tasks. Addressing a gap in retail-specific sim environments for embodied agent training, Sari Sandbox features over 250 interactive grocery items across three store configurations, controlled via an API. It supports both virtual reality (VR) for human interaction and a vision language model (VLM)-powered embodied agent. We also introduce SariBench, a dataset of annotated human demonstrations across varied task difficulties. Our sandbox enables embodied agents to navigate, inspect, and manipulate retail items, providing baselines against human performance. We conclude with benchmarks, performance analysis, and recommendations for enhancing realism and scalability. The source code can be accessed via https://github.com/upeee/sari-sandbox-env.
>
---
#### [new 082] Weakly Supervised Virus Capsid Detection with Image-Level Annotations in Electron Microscopy Images
- **分类: cs.CV**

- **简介: 该论文旨在解决依赖专家标注的病毒囊膜检测问题，通过图像级标注与预训练模型优化，开发了无需人工标注的弱监督检测方法，有效降低了标注成本并提升了效率。**

- **链接: [http://arxiv.org/pdf/2508.00563v1](http://arxiv.org/pdf/2508.00563v1)**

> **作者:** Hannah Kniesel; Leon Sick; Tristan Payer; Tim Bergner; Kavitha Shaga Devan; Clarissa Read; Paul Walther; Timo Ropinski
>
> **摘要:** Current state-of-the-art methods for object detection rely on annotated bounding boxes of large data sets for training. However, obtaining such annotations is expensive and can require up to hundreds of hours of manual labor. This poses a challenge, especially since such annotations can only be provided by experts, as they require knowledge about the scientific domain. To tackle this challenge, we propose a domain-specific weakly supervised object detection algorithm that only relies on image-level annotations, which are significantly easier to acquire. Our method distills the knowledge of a pre-trained model, on the task of predicting the presence or absence of a virus in an image, to obtain a set of pseudo-labels that can be used to later train a state-of-the-art object detection model. To do so, we use an optimization approach with a shrinking receptive field to extract virus particles directly without specific network architectures. Through a set of extensive studies, we show how the proposed pseudo-labels are easier to obtain, and, more importantly, are able to outperform other existing weak labeling methods, and even ground truth labels, in cases where the time to obtain the annotation is limited.
>
---
#### [new 083] Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving
- **分类: cs.CV; cs.CL; cs.IR; cs.RO; 68T45, 68P20, 68T10, 68T50, 68T07, 68T40; I.2.10; I.4.8; I.2.9; H.3.3**

- **简介: 该论文旨在构建一种基于开放语料库的自主驾驶场景检索方法，解决传统方法在长尾数据中的挑战，通过结合SMPL运动序列与文本查询实现高效检索。**

- **链接: [http://arxiv.org/pdf/2508.00589v1](http://arxiv.org/pdf/2508.00589v1)**

> **作者:** Stefan Englmeier; Max A. Büttner; Katharina Winter; Fabian B. Flohr
>
> **备注:** 9 pages, 10 figure, project page https://iv.ee.hm.edu/contextmotionclip/, submitted to IEEE Transactions on Intelligent Vehicles (T-IV), This work has been submitted to the IEEE for possible publication
>
> **摘要:** Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset.
>
---
#### [new 084] GeoMoE: Divide-and-Conquer Motion Field Modeling with Mixture-of-Experts for Two-View Geometry
- **分类: cs.CV**

- **简介: 该论文旨在解决两视图几何中运动场因视角变化和深度断层导致的多样性与异构性问题。通过Mixture-of-Experts（MoE）分割多子域并增强Bi-Path Rectifier，构建GeoMoE框架实现细粒度运动场建模，有效抑制偏差和冗余，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.00592v1](http://arxiv.org/pdf/2508.00592v1)**

> **作者:** Jiajun Le; Jiayi Ma
>
> **摘要:** Recent progress in two-view geometry increasingly emphasizes enforcing smoothness and global consistency priors when estimating motion fields between pairs of images. However, in complex real-world scenes, characterized by extreme viewpoint and scale changes as well as pronounced depth discontinuities, the motion field often exhibits diverse and heterogeneous motion patterns. Most existing methods lack targeted modeling strategies and fail to explicitly account for this variability, resulting in estimated motion fields that diverge from their true underlying structure and distribution. We observe that Mixture-of-Experts (MoE) can assign dedicated experts to motion sub-fields, enabling a divide-and-conquer strategy for heterogeneous motion patterns. Building on this insight, we re-architect motion field modeling in two-view geometry with GeoMoE, a streamlined framework. Specifically, we first devise a Probabilistic Prior-Guided Decomposition strategy that exploits inlier probability signals to perform a structure-aware decomposition of the motion field into heterogeneous sub-fields, sharply curbing outlier-induced bias. Next, we introduce an MoE-Enhanced Bi-Path Rectifier that enhances each sub-field along spatial-context and channel-semantic paths and routes it to a customized expert for targeted modeling, thereby decoupling heterogeneous motion regimes, suppressing cross-sub-field interference and representational entanglement, and yielding fine-grained motion-field rectification. With this minimalist design, GeoMoE outperforms prior state-of-the-art methods in relative pose and homography estimation and shows strong generalization. The source code and pre-trained models are available at https://github.com/JiajunLe/GeoMoE.
>
---
#### [new 085] DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DC-AE 1.5，旨在通过结构化潜空间（强化物体结构特征）和增强扩散训练策略解决高分辨率扩散模型收敛慢的问题。其创新点包括：1) 增加通道数以提升重建质量但导致收敛慢；2) 结构化潜空间与增强扩散训练共同优化模型性能。研究证明DC-AE-1.5在ImageNet上实现4x更快的生成速度并优于传统版本。**

- **链接: [http://arxiv.org/pdf/2508.00413v1](http://arxiv.org/pdf/2508.00413v1)**

> **作者:** Junyu Chen; Dongyun Zou; Wenkun He; Junsong Chen; Enze Xie; Song Han; Han Cai
>
> **备注:** ICCV 2025
>
> **摘要:** We present DC-AE 1.5, a new family of deep compression autoencoders for high-resolution diffusion models. Increasing the autoencoder's latent channel number is a highly effective approach for improving its reconstruction quality. However, it results in slow convergence for diffusion models, leading to poorer generation quality despite better reconstruction quality. This issue limits the quality upper bound of latent diffusion models and hinders the employment of autoencoders with higher spatial compression ratios. We introduce two key innovations to address this challenge: i) Structured Latent Space, a training-based approach to impose a desired channel-wise structure on the latent space with front latent channels capturing object structures and latter latent channels capturing image details; ii) Augmented Diffusion Training, an augmented diffusion training strategy with additional diffusion training objectives on object latent channels to accelerate convergence. With these techniques, DC-AE 1.5 delivers faster convergence and better diffusion scaling results than DC-AE. On ImageNet 512x512, DC-AE-1.5-f64c128 delivers better image generation quality than DC-AE-f32c32 while being 4x faster. Code: https://github.com/dc-ai-projects/DC-Gen.
>
---
#### [new 086] Fine-grained Spatiotemporal Grounding on Egocentric Videos
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在解决egocentric视频中基于文本查询的实体定位问题，通过分析spatiotemporal视频差异性（如时间、空间变化），提出EgoMask作为像素级基准模型，构建EgoMask-Train训练数据集，并验证其在egocentric任务上的优势。**

- **链接: [http://arxiv.org/pdf/2508.00518v1](http://arxiv.org/pdf/2508.00518v1)**

> **作者:** Shuo Liang; Yiwu Zhong; Zi-Yuan Hu; Yeyao Tao; Liwei Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Spatiotemporal video grounding aims to localize target entities in videos based on textual queries. While existing research has made significant progress in exocentric videos, the egocentric setting remains relatively underexplored, despite its growing importance in applications such as augmented reality and robotics. In this work, we conduct a systematic analysis of the discrepancies between egocentric and exocentric videos, revealing key challenges such as shorter object durations, sparser trajectories, smaller object sizes, and larger positional shifts. To address these challenges, we introduce EgoMask, the first pixel-level benchmark for fine-grained spatiotemporal grounding in egocentric videos. It is constructed by our proposed automatic annotation pipeline, which annotates referring expressions and object masks across short-, medium-, and long-term videos. Additionally, we create EgoMask-Train, a large-scale training dataset to facilitate model development. Experiments demonstrate that the state-of-the-art spatiotemporal grounding models perform poorly on our benchmark EgoMask, but fine-tuning on EgoMask-Train yields significant improvements, while preserving performance on exocentric datasets. Our work thus provides essential resources and insights for advancing egocentric video understanding. Our code is available at https://github.com/LaVi-Lab/EgoMask .
>
---
#### [new 087] Bidirectional Action Sequence Learning for Long-term Action Anticipation with Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究了长时动作预测任务，解决了单向编码-解码方法在捕捉语义子动作方面的局限性，提出BiAnt结合前向与后向预测，利用大语言模型实现双向动作序列学习，实验在Ego4D上验证其性能提升。**

- **链接: [http://arxiv.org/pdf/2508.00374v1](http://arxiv.org/pdf/2508.00374v1)**

> **作者:** Yuji Sato; Yasunori Ishii; Takayoshi Yamashita
>
> **备注:** Accepted to MVA2025 (Best Poster Award)
>
> **摘要:** Video-based long-term action anticipation is crucial for early risk detection in areas such as automated driving and robotics. Conventional approaches extract features from past actions using encoders and predict future events with decoders, which limits performance due to their unidirectional nature. These methods struggle to capture semantically distinct sub-actions within a scene. The proposed method, BiAnt, addresses this limitation by combining forward prediction with backward prediction using a large language model. Experimental results on Ego4D demonstrate that BiAnt improves performance in terms of edit distance compared to baseline methods.
>
---
#### [new 088] Cued-Agent: A Collaborative Multi-Agent System for Automatic Cued Speech Recognition
- **分类: cs.CV; eess.AS**

- **简介: 该论文旨在开发一个协作多智能体系统（Cued-Agent），解决传统ACSR方法在数据不足时无法有效融合手/唇运动的问题。通过整合四类子代理（手势识别、唇识别、动态提示解码和语义修正）和扩展Mandarin CS数据集，实现了在正常与听障场景下的卓越性能。**

- **链接: [http://arxiv.org/pdf/2508.00391v1](http://arxiv.org/pdf/2508.00391v1)**

> **作者:** Guanjie Huang; Danny H. K. Tsang; Shan Yang; Guangzhi Lei; Li Liu
>
> **备注:** 9 pages
>
> **摘要:** Cued Speech (CS) is a visual communication system that combines lip-reading with hand coding to facilitate communication for individuals with hearing impairments. Automatic CS Recognition (ACSR) aims to convert CS hand gestures and lip movements into text via AI-driven methods. Traditionally, the temporal asynchrony between hand and lip movements requires the design of complex modules to facilitate effective multimodal fusion. However, constrained by limited data availability, current methods demonstrate insufficient capacity for adequately training these fusion mechanisms, resulting in suboptimal performance. Recently, multi-agent systems have shown promising capabilities in handling complex tasks with limited data availability. To this end, we propose the first collaborative multi-agent system for ACSR, named Cued-Agent. It integrates four specialized sub-agents: a Multimodal Large Language Model-based Hand Recognition agent that employs keyframe screening and CS expert prompt strategies to decode hand movements, a pretrained Transformer-based Lip Recognition agent that extracts lip features from the input video, a Hand Prompt Decoding agent that dynamically integrates hand prompts with lip features during inference in a training-free manner, and a Self-Correction Phoneme-to-Word agent that enables post-process and end-to-end conversion from phoneme sequences to natural language sentences for the first time through semantic refinement. To support this study, we expand the existing Mandarin CS dataset by collecting data from eight hearing-impaired cuers, establishing a mixed dataset of fourteen subjects. Extensive experiments demonstrate that our Cued-Agent performs superbly in both normal and hearing-impaired scenarios compared with state-of-the-art methods. The implementation is available at https://github.com/DennisHgj/Cued-Agent.
>
---
#### [new 089] Video Color Grading via Look-Up Table Generation
- **分类: cs.CV**

- **简介: 该论文提出了一种基于LUT的视频色彩调整框架，解决了传统专业色工处理复杂流程的问题，通过扩散模型训练LUT实现色彩属性对齐与结构细节保留，结合文本提示提升低级特征增强效果，验证了其在视频色彩调整领域的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00548v1](http://arxiv.org/pdf/2508.00548v1)**

> **作者:** Seunghyun Shin; Dongmin Shin; Jisu Shin; Hae-Gon Jeon; Joon-Young Lee
>
> **备注:** ICCV2025
>
> **摘要:** Different from color correction and transfer, color grading involves adjusting colors for artistic or storytelling purposes in a video, which is used to establish a specific look or mood. However, due to the complexity of the process and the need for specialized editing skills, video color grading remains primarily the domain of professional colorists. In this paper, we present a reference-based video color grading framework. Our key idea is explicitly generating a look-up table (LUT) for color attribute alignment between reference scenes and input video via a diffusion model. As a training objective, we enforce that high-level features of the reference scenes like look, mood, and emotion should be similar to that of the input video. Our LUT-based approach allows for color grading without any loss of structural details in the whole video frames as well as achieving fast inference. We further build a pipeline to incorporate a user-preference via text prompts for low-level feature enhancement such as contrast and brightness, etc. Experimental results, including extensive user studies, demonstrate the effectiveness of our approach for video color grading. Codes are publicly available at https://github.com/seunghyuns98/VideoColorGrading.
>
---
#### [new 090] Diffusion-Based User-Guided Data Augmentation for Coronary Stenosis Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于扩散模型的用户引导数据增强技术，解决冠状动脉狭窄检测中受限数据和不平衡问题，通过生成真实病灶并调整严重性提升模型性能，验证在公共和小样本数据集上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.00438v1](http://arxiv.org/pdf/2508.00438v1)**

> **作者:** Sumin Seo; In Kyu Lee; Hyun-Woo Kim; Jaesik Min; Chung-Hwan Jung
>
> **备注:** Accepted at MICCAI 2025. Dataset available at https://github.com/medipixel/DiGDA
>
> **摘要:** Coronary stenosis is a major risk factor for ischemic heart events leading to increased mortality, and medical treatments for this condition require meticulous, labor-intensive analysis. Coronary angiography provides critical visual cues for assessing stenosis, supporting clinicians in making informed decisions for diagnosis and treatment. Recent advances in deep learning have shown great potential for automated localization and severity measurement of stenosis. In real-world scenarios, however, the success of these competent approaches is often hindered by challenges such as limited labeled data and class imbalance. In this study, we propose a novel data augmentation approach that uses an inpainting method based on a diffusion model to generate realistic lesions, allowing user-guided control of severity. Extensive evaluation on lesion detection and severity classification across various synthetic dataset sizes shows superior performance of our method on both a large-scale in-house dataset and a public coronary angiography dataset. Furthermore, our approach maintains high detection and classification performance even when trained with limited data, highlighting its clinical importance in improving the assessment of severity of stenosis and optimizing data utilization for more reliable decision support.
>
---
#### [new 091] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在构建一个开放世界下的无人机目标导航基准，解决传统视觉语言导航受限于细节指导的问题，通过高语义目标与模块化策略实现自主探索，对比多种方法验证其在复杂环境中的挑战性。**

- **链接: [http://arxiv.org/pdf/2508.00288v1](http://arxiv.org/pdf/2508.00288v1)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [new 092] Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri--Rao Product
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究了参数高效微调中低秩方法的局限性，提出通过Khatri-Rao产品优化权重更新以提升有效秩，验证了KRAdapter在大规模语言模型上的性能提升，解决了传统低秩方法难以处理高维矩阵的问题。**

- **链接: [http://arxiv.org/pdf/2508.00230v1](http://arxiv.org/pdf/2508.00230v1)**

> **作者:** Paul Albert; Frederic Z. Zhang; Hemanth Saratchandran; Anton van den Hengel; Ehsan Abbasnejad
>
> **备注:** To appear in ICCV 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) has become a standard approach for adapting large pre-trained models. Amongst PEFT methods, low-rank adaptation (LoRA) has achieved notable success. However, recent studies have highlighted its limitations compared against full-rank alternatives, particularly when applied to multimodal and large language models. In this work, we present a quantitative comparison amongst full-rank and low-rank PEFT methods using a synthetic matrix approximation benchmark with controlled spectral properties. Our results confirm that LoRA struggles to approximate matrices with relatively flat spectrums or high frequency components -- signs of high effective ranks. To this end, we introduce KRAdapter, a novel PEFT algorithm that leverages the Khatri-Rao product to produce weight updates, which, by construction, tends to produce matrix product with a high effective rank. We demonstrate performance gains with KRAdapter on vision-language models up to 1B parameters and on large language models up to 8B parameters, particularly on unseen common-sense reasoning tasks. In addition, KRAdapter maintains the memory and compute efficiency of LoRA, making it a practical and robust alternative to fine-tune billion-scale parameter models.
>
---
#### [new 093] CoRGI: Verified Chain-of-Thought Reasoning with Visual Grounding
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出**CoRGI**框架，解决多步推理中缺乏视觉验证导致的解释偏差问题，通过三阶段流程（生成推理链-提取证据-合成答案）实现可视化验证，提升VLM推理性能与可靠性。**

- **链接: [http://arxiv.org/pdf/2508.00378v1](http://arxiv.org/pdf/2508.00378v1)**

> **作者:** Shixin Yi; Lin Shang
>
> **备注:** Preparing for AAAI 2026, Multimodal Reasoning
>
> **摘要:** Chain-of-Thought (CoT) prompting has shown promise in improving reasoning in vision-language models (VLMs), but it often produces explanations that are linguistically fluent yet lack grounding in visual content. We observe that such hallucinations arise in part from the absence of an explicit verification mechanism during multi-step reasoning. To address this, we propose \textbf{CoRGI}(\textbf{C}hain \textbf{o}f \textbf{R}easoning with \textbf{G}rounded \textbf{I}nsights), a modular framework that introduces visual verification into the reasoning process. CoRGI follows a three-stage pipeline: it first generates a textual reasoning chain, then extracts supporting visual evidence for each reasoning step via a dedicated module (VEVM), and finally synthesizes the textual rationale with visual evidence to generate a grounded, verified answer. The framework can be integrated with existing VLMs without end-to-end retraining. We evaluate CoRGI on the VCR benchmark and find that it improves reasoning performance on two representative open-source VLM backbones, Qwen-2.5VL and LLaVA-1.6. Ablation studies confirm the contribution of each step in the verification module, and human evaluations suggest that CoRGI leads to more factual and helpful explanations. We also examine alternative designs for the visual verification step and discuss potential limitations of post-hoc verification frameworks. These findings highlight the importance of grounding intermediate reasoning steps in visual evidence to enhance the robustness of multimodal reasoning.
>
---
#### [new 094] Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文旨在系统总结医疗推理在LLM中的应用与挑战，提出训练与测试机制分类方法，并分析不同数据模态的应用及评估基准发展，为提升推理透明度和可验证性提供理论支持。**

- **链接: [http://arxiv.org/pdf/2508.00669v1](http://arxiv.org/pdf/2508.00669v1)**

> **作者:** Wenxuan Wang; Zizhan Ma; Meidan Ding; Shiyi Zheng; Shengyuan Liu; Jie Liu; Jiaming Ji; Wenting Chen; Xiang Li; Linlin Shen; Yixuan Yuan
>
> **摘要:** The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI.
>
---
#### [new 095] Weakly Supervised Intracranial Aneurysm Detection and Segmentation in MR angiography via Multi-task UNet with Vesselness Prior
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出了一种基于多任务UNet的弱监督方法，用于在MR angiography中检测并分割脑内动脉瘤，解决了因小尺寸和软对比度导致的检测困难及缺乏标注数据的问题，通过引入血管性优先级提升了分割与检测性能（Dice=0.614，95%HD=1.38mm）。**

- **链接: [http://arxiv.org/pdf/2508.00235v1](http://arxiv.org/pdf/2508.00235v1)**

> **作者:** Erin Rainville; Amirhossein Rasoulian; Hassan Rivaz; Yiming Xiao
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Intracranial aneurysms (IAs) are abnormal dilations of cerebral blood vessels that, if ruptured, can lead to life-threatening consequences. However, their small size and soft contrast in radiological scans often make it difficult to perform accurate and efficient detection and morphological analyses, which are critical in the clinical care of the disorder. Furthermore, the lack of large public datasets with voxel-wise expert annotations pose challenges for developing deep learning algorithms to address the issues. Therefore, we proposed a novel weakly supervised 3D multi-task UNet that integrates vesselness priors to jointly perform aneurysm detection and segmentation in time-of-flight MR angiography (TOF-MRA). Specifically, to robustly guide IA detection and segmentation, we employ the popular Frangi's vesselness filter to derive soft cerebrovascular priors for both network input and an attention block to conduct segmentation from the decoder and detection from an auxiliary branch. We train our model on the Lausanne dataset with coarse ground truth segmentation, and evaluate it on the test set with refined labels from the same database. To further assess our model's generalizability, we also validate it externally on the ADAM dataset. Our results demonstrate the superior performance of the proposed technique over the SOTA techniques for aneurysm segmentation (Dice = 0.614, 95%HD =1.38mm) and detection (false positive rate = 1.47, sensitivity = 92.9%).
>
---
#### [new 096] GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文旨在解决3D牙齿分割在CBCT中的挑战，提出GEPAR3D方法将实例检测与多类分割融合，利用统计形状模型作为先验并结合深度水方法进行分割，验证其在多个临床数据集上的性能提升。**

- **链接: [http://arxiv.org/pdf/2508.00155v1](http://arxiv.org/pdf/2508.00155v1)**

> **作者:** Tomasz Szczepański; Szymon Płotka; Michal K. Grzeszczyk; Arleta Adamowicz; Piotr Fudalej; Przemysław Korzeniowski; Tomasz Trzciński; Arkadiusz Sitek
>
> **备注:** Accepted for the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** Tooth segmentation in Cone-Beam Computed Tomography (CBCT) remains challenging, especially for fine structures like root apices, which is critical for assessing root resorption in orthodontics. We introduce GEPAR3D, a novel approach that unifies instance detection and multi-class segmentation into a single step tailored to improve root segmentation. Our method integrates a Statistical Shape Model of dentition as a geometric prior, capturing anatomical context and morphological consistency without enforcing restrictive adjacency constraints. We leverage a deep watershed method, modeling each tooth as a continuous 3D energy basin encoding voxel distances to boundaries. This instance-aware representation ensures accurate segmentation of narrow, complex root apices. Trained on publicly available CBCT scans from a single center, our method is evaluated on external test sets from two in-house and two public medical centers. GEPAR3D achieves the highest overall segmentation performance, averaging a Dice Similarity Coefficient (DSC) of 95.0% (+2.8% over the second-best method) and increasing recall to 95.2% (+9.5%) across all test sets. Qualitative analyses demonstrated substantial improvements in root segmentation quality, indicating significant potential for more accurate root resorption assessment and enhanced clinical decision-making in orthodontics. We provide the implementation and dataset at https://github.com/tomek1911/GEPAR3D.
>
---
#### [new 097] Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在解决传统3DGS扫描受限于空间的问题，通过双手机器人实现高精度3D数字孪生建模，利用深度学习与GANs进行数据融合，并优化训练流程以支持多视角扫描，验证了 Omni-Scan 在工业缺陷检测中的有效性（100字）。**

- **链接: [http://arxiv.org/pdf/2508.00354v1](http://arxiv.org/pdf/2508.00354v1)**

> **作者:** Tianshuang Qiu; Zehan Ma; Karim El-Refai; Hiya Shah; Chung Min Kim; Justin Kerr; Ken Goldberg
>
> **摘要:** 3D Gaussian Splats (3DGSs) are 3D object models derived from multi-view images. Such "digital twins" are useful for simulations, virtual reality, marketing, robot policy fine-tuning, and part inspection. 3D object scanning usually requires multi-camera arrays, precise laser scanners, or robot wrist-mounted cameras, which have restricted workspaces. We propose Omni-Scan, a pipeline for producing high-quality 3D Gaussian Splat models using a bi-manual robot that grasps an object with one gripper and rotates the object with respect to a stationary camera. The object is then re-grasped by a second gripper to expose surfaces that were occluded by the first gripper. We present the Omni-Scan robot pipeline using DepthAny-thing, Segment Anything, as well as RAFT optical flow models to identify and isolate objects held by a robot gripper while removing the gripper and the background. We then modify the 3DGS training pipeline to support concatenated datasets with gripper occlusion, producing an omni-directional (360 degree view) model of the object. We apply Omni-Scan to part defect inspection, finding that it can identify visual or geometric defects in 12 different industrial and household objects with an average accuracy of 83%. Interactive videos of Omni-Scan 3DGS models can be found at https://berkeleyautomation.github.io/omni-scan/
>
---
#### [new 098] CADS: A Comprehensive Anatomical Dataset and Segmentation for Whole-Body Anatomy in Computed Tomography
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在开发全人体CT分割模型，解决现有方法缺乏统一数据集与临床部署问题。通过构建22,022个CT数据集，集成多源标注并优化模型架构，验证其在辐射肿瘤领域的临床应用效果。**

- **链接: [http://arxiv.org/pdf/2507.22953v1](http://arxiv.org/pdf/2507.22953v1)**

> **作者:** Murong Xu; Tamaz Amiranashvili; Fernando Navarro; Maksym Fritsak; Ibrahim Ethem Hamamci; Suprosanna Shit; Bastian Wittmann; Sezgin Er; Sebastian M. Christ; Ezequiel de la Rosa; Julian Deseoe; Robert Graf; Hendrik Möller; Anjany Sekuboyina; Jan C. Peeken; Sven Becker; Giulia Baldini; Johannes Haubold; Felix Nensa; René Hosch; Nikhil Mirajkar; Saad Khalid; Stefan Zachow; Marc-André Weber; Georg Langs; Jakob Wasserthal; Mehmet Kemal Ozdemir; Andrey Fedorov; Ron Kikinis; Stephanie Tanadini-Lang; Jan S. Kirschke; Stephanie E. Combs; Bjoern Menze
>
> **摘要:** Accurate delineation of anatomical structures in volumetric CT scans is crucial for diagnosis and treatment planning. While AI has advanced automated segmentation, current approaches typically target individual structures, creating a fragmented landscape of incompatible models with varying performance and disparate evaluation protocols. Foundational segmentation models address these limitations by providing a holistic anatomical view through a single model. Yet, robust clinical deployment demands comprehensive training data, which is lacking in existing whole-body approaches, both in terms of data heterogeneity and, more importantly, anatomical coverage. In this work, rather than pursuing incremental optimizations in model architecture, we present CADS, an open-source framework that prioritizes the systematic integration, standardization, and labeling of heterogeneous data sources for whole-body CT segmentation. At its core is a large-scale dataset of 22,022 CT volumes with complete annotations for 167 anatomical structures, representing a significant advancement in both scale and coverage, with 18 times more scans than existing collections and 60% more distinct anatomical targets. Building on this diverse dataset, we develop the CADS-model using established architectures for accessible and automated full-body CT segmentation. Through comprehensive evaluation across 18 public datasets and an independent real-world hospital cohort, we demonstrate advantages over SoTA approaches. Notably, thorough testing of the model's performance in segmentation tasks from radiation oncology validates its direct utility for clinical interventions. By making our large-scale dataset, our segmentation models, and our clinical software tool publicly available, we aim to advance robust AI solutions in radiology and make comprehensive anatomical analysis accessible to clinicians and researchers alike.
>
---
#### [new 099] AI-Driven Collaborative Satellite Object Detection for Space Sustainability
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在解决低轨卫星密集带来的碰撞风险，提出AI驱动的协作卫星对象检测框架，通过高精度模拟数据集和距离感知策略优化性能，实现分布式部署，降低SWaP并提升空间可持续性。**

- **链接: [http://arxiv.org/pdf/2508.00755v1](http://arxiv.org/pdf/2508.00755v1)**

> **作者:** Peng Hu; Wenxuan Zhang
>
> **备注:** Submitted to the 13th Annual IEEE International Conference on Wireless for Space and Extreme Environments (WiSEE 2025)
>
> **摘要:** The growing density of satellites in low-Earth orbit (LEO) presents serious challenges to space sustainability, primarily due to the increased risk of in-orbit collisions. Traditional ground-based tracking systems are constrained by latency and coverage limitations, underscoring the need for onboard, vision-based space object detection (SOD) capabilities. In this paper, we propose a novel satellite clustering framework that enables the collaborative execution of deep learning (DL)-based SOD tasks across multiple satellites. To support this approach, we construct a high-fidelity dataset simulating imaging scenarios for clustered satellite formations. A distance-aware viewpoint selection strategy is introduced to optimize detection performance, and recent DL models are used for evaluation. Experimental results show that the clustering-based method achieves competitive detection accuracy compared to single-satellite and existing approaches, while maintaining a low size, weight, and power (SWaP) footprint. These findings underscore the potential of distributed, AI-enabled in-orbit systems to enhance space situational awareness and contribute to long-term space sustainability.
>
---
#### [new 100] Jet Image Generation in High Energy Physics Using Diffusion Models
- **分类: hep-ph; cs.AI; cs.CV; cs.LG**

- **简介: 该论文旨在开发基于扩散模型的高能物理Jet图像生成技术，首次将JetNet数据集的kinematic参数映射为二维图像，通过训练扩散模型学习空间分布并对比score与一致性模型，解决jet图像生成的高精度与稳定性问题，提升计算效率与研究价值。**

- **链接: [http://arxiv.org/pdf/2508.00250v1](http://arxiv.org/pdf/2508.00250v1)**

> **作者:** Victor D. Martinez; Vidya Manian; Sudhir Malik
>
> **备注:** The paper is under review at IEEE Transactions in Nuclear Science
>
> **摘要:** This article presents, for the first time, the application of diffusion models for generating jet images corresponding to proton-proton collision events at the Large Hadron Collider (LHC). The kinematic variables of quark, gluon, W-boson, Z-boson, and top quark jets from the JetNet simulation dataset are mapped to two-dimensional image representations. Diffusion models are trained on these images to learn the spatial distribution of jet constituents. We compare the performance of score-based diffusion models and consistency models in accurately generating class-conditional jet images. Unlike approaches based on latent distributions, our method operates directly in image space. The fidelity of the generated images is evaluated using several metrics, including the Fr\'echet Inception Distance (FID), which demonstrates that consistency models achieve higher fidelity and generation stability compared to score-based diffusion models. These advancements offer significant improvements in computational efficiency and generation accuracy, providing valuable tools for High Energy Physics (HEP) research.
>
---
#### [new 101] The Repeated-Stimulus Confound in Electroencephalography
- **分类: q-bio.NC; cs.CV; 62K99, 68T05**

- **简介: 该论文旨在揭示EEG中重复刺激干扰对神经编码模型训练与评估的影响，通过研究16项受影响文献的模型表现，分析其准确性偏差（4.46%-7.42%），并探讨其对科学结论的误导性作用。**

- **链接: [http://arxiv.org/pdf/2508.00531v1](http://arxiv.org/pdf/2508.00531v1)**

> **作者:** Jack A. Kilgallen; Barak A. Pearlmutter; Jeffrey Mark Siskind
>
> **备注:** 15 pages, 6 figures, 8 tables, in submission to IEEE
>
> **摘要:** In neural-decoding studies, recordings of participants' responses to stimuli are used to train models. In recent years, there has been an explosion of publications detailing applications of innovations from deep-learning research to neural-decoding studies. The data-hungry models used in these experiments have resulted in a demand for increasingly large datasets. Consequently, in some studies, the same stimuli are presented multiple times to each participant to increase the number of trials available for use in model training. However, when a decoding model is trained and subsequently evaluated on responses to the same stimuli, stimulus identity becomes a confounder for accuracy. We term this the repeated-stimulus confound. We identify a susceptible dataset, and 16 publications which report model performance based on evaluation procedures affected by the confound. We conducted experiments using models from the affected studies to investigate the likely extent to which results in the literature have been misreported. Our findings suggest that the decoding accuracies of these models were overestimated by between 4.46-7.42%. Our analysis also indicates that per 1% increase in accuracy under the confound, the magnitude of the overestimation increases by 0.26%. The confound not only results in optimistic estimates of decoding performance, but undermines the validity of several claims made within the affected publications. We conducted further experiments to investigate the implications of the confound in alternative contexts. We found that the same methodology used within the affected studies could also be used to justify an array of pseudoscientific claims, such as the existence of extrasensory perception.
>
---
#### [new 102] SpA2V: Harnessing Spatial Auditory Cues for Audio-driven Spatially-aware Video Generation
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: SpA2V是一个基于空间音频提示的视频生成框架，解决了音频驱动视频生成中语义信息不足的问题，通过分解生成步骤实现高效、准确的视频生成。**

- **链接: [http://arxiv.org/pdf/2508.00782v1](http://arxiv.org/pdf/2508.00782v1)**

> **作者:** Kien T. Pham; Yingqing He; Yazhou Xing; Qifeng Chen; Long Chen
>
> **备注:** The 33rd ACM Multimedia Conference (MM '25)
>
> **摘要:** Audio-driven video generation aims to synthesize realistic videos that align with input audio recordings, akin to the human ability to visualize scenes from auditory input. However, existing approaches predominantly focus on exploring semantic information, such as the classes of sounding sources present in the audio, limiting their ability to generate videos with accurate content and spatial composition. In contrast, we humans can not only naturally identify the semantic categories of sounding sources but also determine their deeply encoded spatial attributes, including locations and movement directions. This useful information can be elucidated by considering specific spatial indicators derived from the inherent physical properties of sound, such as loudness or frequency. As prior methods largely ignore this factor, we present SpA2V, the first framework explicitly exploits these spatial auditory cues from audios to generate videos with high semantic and spatial correspondence. SpA2V decomposes the generation process into two stages: 1) Audio-guided Video Planning: We meticulously adapt a state-of-the-art MLLM for a novel task of harnessing spatial and semantic cues from input audio to construct Video Scene Layouts (VSLs). This serves as an intermediate representation to bridge the gap between the audio and video modalities. 2) Layout-grounded Video Generation: We develop an efficient and effective approach to seamlessly integrate VSLs as conditional guidance into pre-trained diffusion models, enabling VSL-grounded video generation in a training-free manner. Extensive experiments demonstrate that SpA2V excels in generating realistic videos with semantic and spatial alignment to the input audios.
>
---
#### [new 103] STF: Shallow-Level Temporal Feedback to Enhance Spiking Transformers
- **分类: cs.NE; cs.CV**

- **简介: 该论文提出STF技术，解决SNN性能不足问题，通过结合TSPE和TF模块优化编码层，提升动态模式多样性及鲁棒性，验证其在静态场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00387v1](http://arxiv.org/pdf/2508.00387v1)**

> **作者:** Zeqi Zheng; Zizheng Zhu; Yingchao Yu; Yanchen Huang; Changze Lv; Junfeng Tang; Zhaofei Yu; Yaochu Jin
>
> **备注:** 32 pages, 4 figures
>
> **摘要:** Transformer-based Spiking Neural Networks (SNNs) suffer from a great performance gap compared to floating-point Artificial Neural Networks (ANNs) due to the binary nature of spike trains. Recent efforts have introduced deep-level feedback loops to transmit high-level semantic information to narrow this gap. However, these designs often span multiple deep layers, resulting in costly feature transformations, higher parameter overhead, increased energy consumption, and longer inference latency. To address this issue, we propose Shallow-level Temporal Feedback (STF), a lightweight plug-and-play module for the encoding layer, which consists of Temporal-Spatial Position Embedding (TSPE) and Temporal Feedback (TF).Extensive experiments show that STF consistently improves performance across various Transformer-based SNN backbones on static datasets, including CIFAR-10, CIFAR-100, and ImageNet-1K, under different spike timestep settings. Further analysis reveals that STF enhances the diversity of the spike patterns, which is key to performance gain. Moreover, evaluations on adversarial robustness and temporal sensitivity confirm that STF outperforms direct coding and its variants, highlighting its potential as a new spike encoding scheme for static scenarios. Our code will be released upon acceptance.
>
---
#### [new 104] Stress-Aware Resilient Neural Training
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出一种基于应力感知的抗扰动神经网络训练方法，解决了在不确定动态下的训练鲁棒性问题。通过动态调整优化策略，结合临时与永久变形机制，设计了Plastic Deformation Optimizer，注入适应性噪声以避开局部极小值并提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.00098v1](http://arxiv.org/pdf/2508.00098v1)**

> **作者:** Ashkan Shakarami; Yousef Yeganeh; Azade Farshad; Lorenzo Nicole; Stefano Ghidoni; Nassir Navab
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** This paper introduces Stress-Aware Learning, a resilient neural training paradigm in which deep neural networks dynamically adjust their optimization behavior - whether under stable training regimes or in settings with uncertain dynamics - based on the concept of Temporary (Elastic) and Permanent (Plastic) Deformation, inspired by structural fatigue in materials science. To instantiate this concept, we propose Plastic Deformation Optimizer, a stress-aware mechanism that injects adaptive noise into model parameters whenever an internal stress signal - reflecting stagnation in training loss and accuracy - indicates persistent optimization difficulty. This enables the model to escape sharp minima and converge toward flatter, more generalizable regions of the loss landscape. Experiments across six architectures, four optimizers, and seven vision benchmarks demonstrate improved robustness and generalization with minimal computational overhead. The code and 3D visuals will be available on GitHub: https://github.com/Stress-Aware-Learning/SAL.
>
---
#### [new 105] Occlusion-robust Stylization for Drawing-based 3D Animation
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出了一种针对手绘3D动画的风格鲁棒化框架（OSF），旨在解决传统方法在遮挡下导致的风格失真问题。通过结合光学流进行边缘引导，并采用单次推理方案，有效提升了输入与输出之间的风格一致性及效率。**

- **链接: [http://arxiv.org/pdf/2508.00398v1](http://arxiv.org/pdf/2508.00398v1)**

> **作者:** Sunjae Yoon; Gwanhyeong Koo; Younghwan Lee; Ji Woo Hong; Chang D. Yoo
>
> **备注:** 11 pages, 13 figures, ICCV 2025
>
> **摘要:** 3D animation aims to generate a 3D animated video from an input image and a target 3D motion sequence. Recent advances in image-to-3D models enable the creation of animations directly from user-hand drawings. Distinguished from conventional 3D animation, drawing-based 3D animation is crucial to preserve artist's unique style properties, such as rough contours and distinct stroke patterns. However, recent methods still exhibit quality deterioration in style properties, especially under occlusions caused by overlapping body parts, leading to contour flickering and stroke blurring. This occurs due to a `stylization pose gap' between training and inference in stylization networks designed to preserve drawing styles in drawing-based 3D animation systems. The stylization pose gap denotes that input target poses used to train the stylization network are always in occlusion-free poses, while target poses encountered in an inference include diverse occlusions under dynamic motions. To this end, we propose Occlusion-robust Stylization Framework (OSF) for drawing-based 3D animation. We found that while employing object's edge can be effective input prior for guiding stylization, it becomes notably inaccurate when occlusions occur at inference. Thus, our proposed OSF provides occlusion-robust edge guidance for stylization network using optical flow, ensuring a consistent stylization even under occlusions. Furthermore, OSF operates in a single run instead of the previous two-stage method, achieving 2.4x faster inference and 2.1x less memory.
>
---
#### [new 106] On-Device Diffusion Transformer Policy for Efficient Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文旨在开发高效机器人操控的扩散策略框架LightDP，解决移动端部署时因计算效率低和内存占用大导致的应用瓶颈，通过网络压缩、采样步长优化及一致性蒸馏技术实现实时动作预测。**

- **链接: [http://arxiv.org/pdf/2508.00697v1](http://arxiv.org/pdf/2508.00697v1)**

> **作者:** Yiming Wu; Huan Wang; Zhenghao Chen; Jianxin Pang; Dong Xu
>
> **备注:** ICCV 2025
>
> **摘要:** Diffusion Policies have significantly advanced robotic manipulation tasks via imitation learning, but their application on resource-constrained mobile platforms remains challenging due to computational inefficiency and extensive memory footprint. In this paper, we propose LightDP, a novel framework specifically designed to accelerate Diffusion Policies for real-time deployment on mobile devices. LightDP addresses the computational bottleneck through two core strategies: network compression of the denoising modules and reduction of the required sampling steps. We first conduct an extensive computational analysis on existing Diffusion Policy architectures, identifying the denoising network as the primary contributor to latency. To overcome performance degradation typically associated with conventional pruning methods, we introduce a unified pruning and retraining pipeline, optimizing the model's post-pruning recoverability explicitly. Furthermore, we combine pruning techniques with consistency distillation to effectively reduce sampling steps while maintaining action prediction accuracy. Experimental evaluations on the standard datasets, \ie, PushT, Robomimic, CALVIN, and LIBERO, demonstrate that LightDP achieves real-time action prediction on mobile devices with competitive performance, marking an important step toward practical deployment of diffusion-based policies in resource-limited environments. Extensive real-world experiments also show the proposed LightDP can achieve performance comparable to state-of-the-art Diffusion Policies.
>
---
#### [new 107] FMPlug: Plug-In Foundation Flow-Matching Priors for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文提出FMPlug框架，解决逆问题中的基流匹配（FM）先验优化任务，通过引入观察与目标对象的相似性及生成模型的高斯性，结合时间自适应策略和尖锐高斯约束，突破传统方法在无领域专有先验和非训练数据下的局限性，显著提升了图像超分辨率和高斯模糊修复性能。**

- **链接: [http://arxiv.org/pdf/2508.00721v1](http://arxiv.org/pdf/2508.00721v1)**

> **作者:** Yuxiang Wan; Ryan Devera; Wenjie Zhang; Ju Sun
>
> **摘要:** We present FMPlug, a novel plug-in framework that enhances foundation flow-matching (FM) priors for solving ill-posed inverse problems. Unlike traditional approaches that rely on domain-specific or untrained priors, FMPlug smartly leverages two simple but powerful insights: the similarity between observed and desired objects and the Gaussianity of generative flows. By introducing a time-adaptive warm-up strategy and sharp Gaussianity regularization, FMPlug unlocks the true potential of domain-agnostic foundation models. Our method beats state-of-the-art methods that use foundation FM priors by significant margins, on image super-resolution and Gaussian deblurring.
>
---
#### [new 108] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文提出了一种多模态音频生成框架AudioGen-Omni，解决跨模态协同生成问题，通过联合训练方法融合图表示、利用AdaLN-PAAPI机制优化注意力，并冻结文本实现跨模态条件能力提升，有效提升了音频质量、语义和同步准确性，推理时间仅1.9秒。**

- **链接: [http://arxiv.org/pdf/2508.00733v1](http://arxiv.org/pdf/2508.00733v1)**

> **作者:** Le Wang; Jun Wang; Feng Deng; Chen Zhang; Kun Gai; Di Zhang
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and songs coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both sung and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
## 更新

#### [replaced 001] Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16181v2](http://arxiv.org/pdf/2405.16181v2)**

> **作者:** Chunlin Qiu; Ang Li; Yiheng Duan; Shenyi Zhang; Yuanjie Zhang; Lingchen Zhao; Qian Wang
>
> **备注:** The original NCS method has been revised and renamed as MEF. A theoretical proof of the relationship between flatness and transferability is added
>
> **摘要:** Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.
>
---
#### [replaced 002] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.08451v2](http://arxiv.org/pdf/2406.08451v2)**

> **作者:** Quanfeng Lu; Wenqi Shao; Zitao Liu; Lingxiao Du; Fanqing Meng; Boxuan Li; Botong Chen; Siyuan Huang; Kaipeng Zhang; Ping Luo
>
> **备注:** 22 pages, 14 figures, ICCV 2025, a cross-app GUI navigation dataset
>
> **摘要:** Autonomous Graphical User Interface (GUI) navigation agents can enhance user experience in communication, entertainment, and productivity by streamlining workflows and reducing manual intervention. However, prior GUI agents often trained with datasets comprising tasks that can be completed within a single app, leading to poor performance in cross-app navigation. To address this problem, we present GUIOdyssey, a comprehensive dataset for cross-app mobile GUI navigation. GUIOdyssey comprises 8,334 episodes with an average of 15.3 steps per episode, covering 6 mobile devices, 212 distinct apps, and 1,357 app combinations. Each step is enriched with detailed semantic reasoning annotations, which aid the model in building cognitive processes and enhancing its reasoning abilities for complex cross-app tasks. Building on GUIOdyssey, we develop OdysseyAgent, an exploratory multimodal agent for long-step cross-app navigation equipped with a history resampler module that efficiently attends to historical screenshot tokens, balancing performance and inference speed. Extensive experiments conducted in both in-domain and out-of-domain scenarios validate the effectiveness of our approach. Moreover, we demonstrate that historial information involving actions, screenshots and context in our dataset can significantly enhances OdysseyAgent's performance on complex cross-app tasks.
>
---
#### [replaced 003] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10086v3](http://arxiv.org/pdf/2411.10086v3)**

> **作者:** Dengke Zhang; Fagui Liu; Quan Tang
>
> **备注:** Accepted to ICCV 2025 Oral
>
> **摘要:** Open-vocabulary semantic segmentation aims to assign semantic labels to each pixel without being constrained by a predefined set of categories. While Contrastive Language-Image Pre-training (CLIP) excels in zero-shot classification, it struggles to align image patches with category embeddings because of its incoherent patch correlations. This study reveals that inter-class correlations are the main reason for impairing CLIP's segmentation performance. Accordingly, we propose CorrCLIP, which reconstructs the scope and value of patch correlations. Specifically, CorrCLIP leverages the Segment Anything Model (SAM) to define the scope of patch interactions, reducing inter-class correlations. To mitigate the problem that SAM-generated masks may contain patches belonging to different classes, CorrCLIP incorporates self-supervised models to compute coherent similarity values, suppressing the weight of inter-class correlations. Additionally, we introduce two additional branches to strengthen patch features' spatial details and semantic representation. Finally, we update segmentation maps with SAM-generated masks to improve spatial consistency. Based on the improvement across patch correlations, feature representations, and segmentation maps, CorrCLIP achieves superior performance across eight benchmarks. Codes are available at: https://github.com/zdk258/CorrCLIP.
>
---
#### [replaced 004] Gaga: Group Any Gaussians via 3D-aware Memory Bank
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.07977v3](http://arxiv.org/pdf/2404.07977v3)**

> **作者:** Weijie Lyu; Xueting Li; Abhijit Kundu; Yi-Hsuan Tsai; Ming-Hsuan Yang
>
> **备注:** Project Page: https://weijielyu.github.io/Gaga
>
> **摘要:** We introduce Gaga, a framework that reconstructs and segments open-world 3D scenes by leveraging inconsistent 2D masks predicted by zero-shot class-agnostic segmentation models. Contrasted to prior 3D scene segmentation approaches that rely on video object tracking or contrastive learning methods, Gaga utilizes spatial information and effectively associates object masks across diverse camera poses through a novel 3D-aware memory bank. By eliminating the assumption of continuous view changes in training images, Gaga demonstrates robustness to variations in camera poses, particularly beneficial for sparsely sampled images, ensuring precise mask label consistency. Furthermore, Gaga accommodates 2D segmentation masks from diverse sources and demonstrates robust performance with different open-world zero-shot class-agnostic segmentation models, significantly enhancing its versatility. Extensive qualitative and quantitative evaluations demonstrate that Gaga performs favorably against state-of-the-art methods, emphasizing its potential for real-world applications such as 3D scene understanding and manipulation.
>
---
#### [replaced 005] DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24025v2](http://arxiv.org/pdf/2505.24025v2)**

> **作者:** Chenbin Pan; Wenbin He; Zhengzhong Tu; Liu Ren
>
> **摘要:** The recent explosive interest in the reasoning capabilities of large language models, such as DeepSeek-R1, has demonstrated remarkable success through reinforcement learning-based fine-tuning frameworks, exemplified by methods like Group Relative Policy Optimization (GRPO). However, such reasoning abilities remain underexplored and notably absent in vision foundation models, including representation models like the DINO series. In this work, we propose \textbf{DINO-R1}, the first such attempt to incentivize visual in-context reasoning capabilities of vision foundation models using reinforcement learning. Specifically, DINO-R1 introduces \textbf{Group Relative Query Optimization (GRQO)}, a novel reinforcement-style training strategy explicitly designed for query-based representation models, which computes query-level rewards based on group-normalized alignment quality. We also apply KL-regularization to stabilize the objectness distribution to reduce the training instability. This joint optimization enables dense and expressive supervision across queries while mitigating overfitting and distributional drift. Building upon Grounding-DINO, we train a series of DINO-R1 family models that integrate a visual prompt encoder and a visual-guided query selection mechanism. Extensive experiments on COCO, LVIS, and ODinW demonstrate that DINO-R1 significantly outperforms supervised fine-tuning baselines, achieving strong generalization in both open-vocabulary and closed-set visual prompting scenarios.
>
---
#### [replaced 006] Gradient Leakage Defense with Key-Lock Module for Federated Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.04095v3](http://arxiv.org/pdf/2305.04095v3)**

> **作者:** Hanchi Ren; Jingjing Deng; Xianghua Xie
>
> **备注:** The source code can be found at https://github.com/Rand2AI/FedKL
>
> **摘要:** Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is infeasible; and b) the global model's inference performance is significantly compromised. We discuss the theoretical underpinnings of why gradients can leak private information and provide theoretical proof of our method's effectiveness. We conducted extensive empirical evaluations with many models on several popular benchmarks, demonstrating the robustness of our proposed approach in both maintaining model performance and defending against gradient leakage attacks.
>
---
#### [replaced 007] ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16991v2](http://arxiv.org/pdf/2506.16991v2)**

> **作者:** Binbin Xiang; Maciej Wielgosz; Stefano Puliti; Kamil Král; Martin Krůček; Azim Missarov; Rasmus Astrup
>
> **摘要:** The segmentation of forest LiDAR 3D point clouds, including both individual tree and semantic segmentation, is fundamental for advancing forest management and ecological research. However, current approaches often struggle with the complexity and variability of natural forest environments. We present ForestFormer3D, a new unified and end-to-end framework designed for precise individual tree and semantic segmentation. ForestFormer3D incorporates ISA-guided query point selection, a score-based block merging strategy during inference, and a one-to-many association mechanism for effective training. By combining these new components, our model achieves state-of-the-art performance for individual tree segmentation on the newly introduced FOR-instanceV2 dataset, which spans diverse forest types and regions. Additionally, ForestFormer3D generalizes well to unseen test sets (Wytham woods and LAUTx), showcasing its robustness across different forest conditions and sensor modalities. The FOR-instanceV2 dataset and the ForestFormer3D code are publicly available at https://bxiang233.github.io/FF3D/.
>
---
#### [replaced 008] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2412.17812v2](http://arxiv.org/pdf/2412.17812v2)**

> **作者:** Weijie Lyu; Yi Zhou; Ming-Hsuan Yang; Zhixin Shu
>
> **备注:** ICCV 2025 Camera-Ready Version. Project Page: https://weijielyu.github.io/FaceLift
>
> **摘要:** We present FaceLift, a novel feed-forward approach for generalizable high-quality 360-degree 3D head reconstruction from a single image. Our pipeline first employs a multi-view latent diffusion model to generate consistent side and back views from a single facial input, which then feeds into a transformer-based reconstructor that produces a comprehensive 3D Gaussian splats representation. Previous methods for monocular 3D face reconstruction often lack full view coverage or view consistency due to insufficient multi-view supervision. We address this by creating a high-quality synthetic head dataset that enables consistent supervision across viewpoints. To bridge the domain gap between synthetic training data and real-world images, we propose a simple yet effective technique that ensures the view generation process maintains fidelity to the input by learning to reconstruct the input image alongside the view generation. Despite being trained exclusively on synthetic data, our method demonstrates remarkable generalization to real-world images. Through extensive qualitative and quantitative evaluations, we show that FaceLift outperforms state-of-the-art 3D face reconstruction methods on identity preservation, detail recovery, and rendering quality.
>
---
#### [replaced 009] TopoRec: Point Cloud Recognition Using Topological Data Analysis
- **分类: cs.RO; cs.CG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18725v2](http://arxiv.org/pdf/2506.18725v2)**

> **作者:** Anirban Ghosh; Iliya Kulbaka; Ian Dahlin; Ayan Dutta
>
> **摘要:** Point cloud-based object/place recognition remains a problem of interest in applications such as autonomous driving, scene reconstruction, and localization. Extracting a meaningful global descriptor from a query point cloud that can be matched with the descriptors of the database point clouds is a challenging problem. Furthermore, when the query point cloud is noisy or has been transformed (e.g., rotated), it adds to the complexity. To this end, we propose a novel methodology, named TopoRec, which utilizes Topological Data Analysis (TDA) for extracting local descriptors from a point cloud, thereby eliminating the need for resource-intensive GPU-based machine learning training. More specifically, we used the ATOL vectorization method to generate vectors for point clouds. To test the quality of the proposed TopoRec technique, we have implemented it on multiple real-world (e.g., Oxford RobotCar, NCLT) and realistic (e.g., ShapeNet) point cloud datasets for large-scale place and object recognition, respectively. Unlike existing learning-based approaches such as PointNetVLAD and PCAN, our method does not require extensive training, making it easily adaptable to new environments. Despite this, it consistently outperforms both state-of-the-art learning-based and handcrafted baselines (e.g., M2DP, ScanContext) on standard benchmark datasets, demonstrating superior accuracy and strong generalization.
>
---
#### [replaced 010] LargeMvC-Net: Anchor-based Deep Unfolding Network for Large-scale Multi-view Clustering
- **分类: cs.CV; stat.CO; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.20980v2](http://arxiv.org/pdf/2507.20980v2)**

> **作者:** Shide Du; Chunming Wu; Zihan Fang; Wendi Zhao; Yilin Wu; Changwei Wang; Shiping Wang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Deep anchor-based multi-view clustering methods enhance the scalability of neural networks by utilizing representative anchors to reduce the computational complexity of large-scale clustering. Despite their scalability advantages, existing approaches often incorporate anchor structures in a heuristic or task-agnostic manner, either through post-hoc graph construction or as auxiliary components for message passing. Such designs overlook the core structural demands of anchor-based clustering, neglecting key optimization principles. To bridge this gap, we revisit the underlying optimization problem of large-scale anchor-based multi-view clustering and unfold its iterative solution into a novel deep network architecture, termed LargeMvC-Net. The proposed model decomposes the anchor-based clustering process into three modules: RepresentModule, NoiseModule, and AnchorModule, corresponding to representation learning, noise suppression, and anchor indicator estimation. Each module is derived by unfolding a step of the original optimization procedure into a dedicated network component, providing structural clarity and optimization traceability. In addition, an unsupervised reconstruction loss aligns each view with the anchor-induced latent space, encouraging consistent clustering structures across views. Extensive experiments on several large-scale multi-view benchmarks show that LargeMvC-Net consistently outperforms state-of-the-art methods in terms of both effectiveness and scalability.
>
---
#### [replaced 011] ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04974v2](http://arxiv.org/pdf/2505.04974v2)**

> **作者:** Wanjiang Weng; Xiaofeng Tan; Hongsong Wang; Pan Zhou
>
> **备注:** We believe that there are some areas in the manuscript that require further improvement, and out of our commitment to refining this work, we have decided to withdraw our manuscript after careful deliberation and discussion
>
> **摘要:** Bilingual text-to-motion generation, which synthesizes 3D human motions from bilingual text inputs, holds immense potential for cross-linguistic applications in gaming, film, and robotics. However, this task faces critical challenges: the absence of bilingual motion-language datasets and the misalignment between text and motion distributions in diffusion models, leading to semantically inconsistent or low-quality motions. To address these challenges, we propose BiHumanML3D, a novel bilingual human motion dataset, which establishes a crucial benchmark for bilingual text-to-motion generation models. Furthermore, we propose a Bilingual Motion Diffusion model (BiMD), which leverages cross-lingual aligned representations to capture semantics, thereby achieving a unified bilingual model. Building upon this, we propose Reward-guided sampling Alignment (ReAlign) method, comprising a step-aware reward model to assess alignment quality during sampling and a reward-guided strategy that directs the diffusion process toward an optimally aligned distribution. This reward model integrates step-aware tokens and combines a text-aligned module for semantic consistency and a motion-aligned module for realism, refining noisy motions at each timestep to balance probability density and alignment. Experiments demonstrate that our approach significantly improves text-motion alignment and motion quality compared to existing state-of-the-art methods. Project page: https://wengwanjiang.github.io/ReAlign-page/.
>
---
#### [replaced 012] Core-Set Selection for Data-efficient Land Cover Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01225v2](http://arxiv.org/pdf/2505.01225v2)**

> **作者:** Keiller Nogueira; Akram Zaytar; Wanli Ma; Ribana Roscher; Ronny Hänsch; Caleb Robinson; Anthony Ortiz; Simone Nsutezo; Rahul Dodhia; Juan M. Lavista Ferres; Oktay Karakuş; Paul L. Rosin
>
> **摘要:** The increasing accessibility of remotely sensed data and the potential of such data to inform large-scale decision-making has driven the development of deep learning models for many Earth Observation tasks. Traditionally, such models must be trained on large datasets. However, the common assumption that broadly larger datasets lead to better outcomes tends to overlook the complexities of the data distribution, the potential for introducing biases and noise, and the computational resources required for processing and storing vast datasets. Therefore, effective solutions should consider both the quantity and quality of data. In this paper, we propose six novel core-set selection methods for selecting important subsets of samples from remote sensing image segmentation datasets that rely on imagery only, labels only, and a combination of each. We benchmark these approaches against a random-selection baseline on three commonly used land cover classification datasets: DFC2022, Vaihingen, and Potsdam. In each of the datasets, we demonstrate that training on a subset of samples outperforms the random baseline, and some approaches outperform training on all available data. This result shows the importance and potential of data-centric learning for the remote sensing domain. The code is available at https://github.com/keillernogueira/data-centric-rs-classification/.
>
---
#### [replaced 013] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.16719v3](http://arxiv.org/pdf/2411.16719v3)**

> **作者:** Xiaoling Hu; Xiangrui Zeng; Oula Puonti; Juan Eugenio Iglesias; Bruce Fischl; Yael Balbastre
>
> **备注:** 16 pages, 5 figures. Accepted by ICCV'25
>
> **摘要:** Domain randomization through synthesis is a powerful strategy to train networks that are unbiased with respect to the domain of the input images. Randomization allows networks to see a virtually infinite range of intensities and artifacts during training, thereby minimizing overfitting to appearance and maximizing generalization to unseen data. Although powerful, this approach relies on the accurate tuning of a large set of hyperparameters that govern the probabilistic distribution of the synthesized images. Instead of manually tuning these parameters, we introduce Learn2Synth, a novel procedure in which synthesis parameters are learned using a small set of real labeled data. Unlike methods that impose constraints to align synthetic data with real data (e.g., contrastive or adversarial techniques), which risk misaligning the image and its label map, we tune an augmentation engine such that a segmentation network trained on synthetic data has optimal accuracy when applied to real data. This approach allows the training procedure to benefit from real labeled examples, without ever using these real examples to train the segmentation network, which avoids biasing the network towards the properties of the training set. Specifically, we develop parametric and nonparametric strategies to enhance synthetic images in a way that improves the performance of the segmentation network. We demonstrate the effectiveness of this learning strategy on synthetic and real-world brain scans. Code is available at: https://github.com/HuXiaoling/Learn2Synth.
>
---
#### [replaced 014] AttnMod: Attention-Based New Art Styles
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.10028v2](http://arxiv.org/pdf/2409.10028v2)**

> **作者:** Shih-Chieh Su
>
> **摘要:** We introduce AttnMod, a training-free technique that modulates cross-attention in pre-trained diffusion models to generate novel, unpromptable art styles. The method is inspired by how a human artist might reinterpret a generated image, for example by emphasizing certain features, dispersing color, twisting silhouettes, or materializing unseen elements. AttnMod simulates this intent by altering how the text prompt conditions the image through attention during denoising. These targeted modulations enable diverse stylistic transformations without changing the prompt or retraining the model, and they expand the expressive capacity of text-to-image generation.
>
---
#### [replaced 015] Sign Spotting Disambiguation using Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03703v2](http://arxiv.org/pdf/2507.03703v2)**

> **作者:** JianHe Low; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in the international conference on Intelligent Virtual Agents (IVA Adjunct)
>
> **摘要:** Sign spotting, the task of identifying and localizing individual signs within continuous sign language video, plays a pivotal role in scaling dataset annotations and addressing the severe data scarcity issue in sign language translation. While automatic sign spotting holds great promise for enabling frame-level supervision at scale, it grapples with challenges such as vocabulary inflexibility and ambiguity inherent in continuous sign streams. Hence, we introduce a novel, training-free framework that integrates Large Language Models (LLMs) to significantly enhance sign spotting quality. Our approach extracts global spatio-temporal and hand shape features, which are then matched against a large-scale sign dictionary using dynamic time warping and cosine similarity. This dictionary-based matching inherently offers superior vocabulary flexibility without requiring model retraining. To mitigate noise and ambiguity from the matching process, an LLM performs context-aware gloss disambiguation via beam search, notably without fine-tuning. Extensive experiments on both synthetic and real-world sign language datasets demonstrate our method's superior accuracy and sentence fluency compared to traditional approaches, highlighting the potential of LLMs in advancing sign spotting.
>
---
#### [replaced 016] Enhanced Vision-Language Models for Diverse Sensor Understanding: Cost-Efficient Optimization and Benchmarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20750v2](http://arxiv.org/pdf/2412.20750v2)**

> **作者:** Sangyun Chung; Youngjoon Yu; Se Yeon Kim; Youngchae Chee; Yong Man Ro
>
> **摘要:** Large-scale Vision-Language Models (VLMs) have achieved notable progress in aligning visual inputs with text. However, their ability to deeply understand the unique physical properties of non-RGB vision sensor images remains limited. In this paper, we revisit and analyze these limitations and introduce a novel, cost-efficient paradigm that significantly advances sensor image understanding-without requiring extensive training data or any modifications to the existing VLM architectures. Specifically, we propose Sensor-Aware Attributes Fine-Tuning (SAFT) with the Diverse Negative Attributes (DNA) optimization, which leverages minimal sensor-specific data to enable robust learning of non-RGB characteristics and overcome RGB-centric biases inherent in current VLMs. In addition, we present VS-TDX-the first comprehensive, public benchmark designed to rigorously evaluate VLMs' sensor-specific understanding across diverse and realistic scenarios. Through extensive experiments on VLMs and various sensor modalities, we validate that our method consistently delivers superior performance and generalization under resource-constrained and architecture-invariant settings. Our approach provides a practical advance towards scalable deployment of VLMs in increasingly sensor-diverse real-world environments.
>
---
#### [replaced 017] From Press to Pixels: Evolving Urdu Text Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13943v2](http://arxiv.org/pdf/2505.13943v2)**

> **作者:** Samee Arif; Sualeha Farid
>
> **摘要:** This paper introduces an end-to-end pipeline for Optical Character Recognition (OCR) on Urdu newspapers, addressing challenges posed by complex multi-column layouts, low-resolution scans, and the stylistic variability of the Nastaliq script. Our system comprises four modules: (1) article segmentation, (2) image super-resolution, (3) column segmentation, and (4) text recognition. We fine-tune YOLOv11x for segmentation, achieving 0.963 precision for articles and 0.970 for columns. A SwinIR-based super-resolution model boosts LLM text recognition accuracy by 25-70%. We also introduce the Urdu Newspaper Benchmark (UNB), a manually annotated dataset for Urdu OCR. Using UNB and the OpenITI corpus, we compare traditional CNN+RNN-based OCR models with modern LLMs. Gemini-2.5-Pro achieves the best performance with a WER of 0.133. We further analyze LLM outputs via insertion, deletion, and substitution error breakdowns, as well as character-level confusion analysis. Finally, we show that fine-tuning on just 500 samples yields a 6.13% WER improvement, highlighting the adaptability of LLMs for Urdu OCR.
>
---
#### [replaced 018] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13180v2](http://arxiv.org/pdf/2412.13180v2)**

> **作者:** Mark Endo; Xiaohan Wang; Serena Yeung-Levy
>
> **备注:** ICCV 2025, project page: https://web.stanford.edu/~markendo/projects/feather
>
> **摘要:** Recent works on accelerating Vision-Language Models achieve strong performance across a variety of vision-language tasks despite highly compressing visual information. In this work, we examine the popular acceleration approach of early pruning of visual tokens inside the language model. Surprisingly, we find that while strong performance is maintained across many tasks, it exhibits drastically different behavior for a subset of vision-centric tasks such as localization. Upon further investigation, we uncover a core issue with the acceleration approach where most tokens towards the top of the image are pruned away. Yet, on many benchmarks aiming to evaluate vision-centric capabilities, strong performance persists with the flawed pruning strategy, highlighting these benchmarks' limited ability to assess fine-grained visual capabilities. Based on these findings, we propose FEATHER (Fast and Effective Acceleration wiTH Ensemble cRiteria), a straightforward approach that resolves the discovered early-layer pruning issue and further enhances the preservation of relevant tokens via multistage pruning with early uniform sampling to ensure broad image coverage. With comparable computational savings, we find that FEATHER achieves more than 5x performance improvement on the vision-centric localization benchmarks compared to the original acceleration approach.
>
---
#### [replaced 019] Towards a Unified Copernicus Foundation Model for Earth Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11849v3](http://arxiv.org/pdf/2503.11849v3)**

> **作者:** Yi Wang; Zhitong Xiong; Chenying Liu; Adam J. Stewart; Thomas Dujardin; Nikolaos Ioannis Bountos; Angelos Zavras; Franziska Gerken; Ioannis Papoutsis; Laura Leal-Taixé; Xiao Xiang Zhu
>
> **备注:** Accepted to ICCV 2025. 33 pages, 34 figures
>
> **摘要:** Advances in Earth observation (EO) foundation models have unlocked the potential of big satellite data to learn generic representations from space, benefiting a wide range of downstream applications crucial to our planet. However, most existing efforts remain limited to fixed spectral sensors, focus solely on the Earth's surface, and overlook valuable metadata beyond imagery. In this work, we take a step towards next-generation EO foundation models with three key components: 1) Copernicus-Pretrain, a massive-scale pretraining dataset that integrates 18.7M aligned images from all major Copernicus Sentinel missions, spanning from the Earth's surface to its atmosphere; 2) Copernicus-FM, a unified foundation model capable of processing any spectral or non-spectral sensor modality using extended dynamic hypernetworks and flexible metadata encoding; and 3) Copernicus-Bench, a systematic evaluation benchmark with 15 hierarchical downstream tasks ranging from preprocessing to specialized applications for each Sentinel mission. Our dataset, model, and benchmark greatly improve the scalability, versatility, and multimodal adaptability of EO foundation models, while also creating new opportunities to connect EO, weather, and climate research. Codes, datasets and models are available at https://github.com/zhu-xlab/Copernicus-FM.
>
---
#### [replaced 020] SynPAIN: A Synthetic Dataset of Pain and Non-Pain Facial Expressions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19673v2](http://arxiv.org/pdf/2507.19673v2)**

> **作者:** Babak Taati; Muhammad Muzammil; Yasamin Zarghami; Abhishek Moturu; Amirhossein Kazerouni; Hailey Reimer; Alex Mihailidis; Thomas Hadjistavropoulos
>
> **备注:** 10 pages, 4 figures, submitted to IEEE JBHI
>
> **摘要:** Accurate pain assessment in patients with limited ability to communicate, such as older adults with dementia, represents a critical healthcare challenge. Robust automated systems of pain detection may facilitate such assessments. Existing pain detection datasets, however, suffer from limited ethnic/racial diversity, privacy constraints, and underrepresentation of older adults who are the primary target population for clinical deployment. We present SynPAIN, a large-scale synthetic dataset containing 10,710 facial expression images (5,355 neutral/expressive pairs) across five ethnicities/races, two age groups (young: 20-35, old: 75+), and two genders. Using commercial generative AI tools, we created demographically balanced synthetic identities with clinically meaningful pain expressions. Our validation demonstrates that synthetic pain expressions exhibit expected pain patterns, scoring significantly higher than neutral and non-pain expressions using clinically validated pain assessment tools based on facial action unit analysis. We experimentally demonstrate SynPAIN's utility in identifying algorithmic bias in existing pain detection models. Through comprehensive bias evaluation, we reveal substantial performance disparities across demographic characteristics. These performance disparities were previously undetectable with smaller, less diverse datasets. Furthermore, we demonstrate that age-matched synthetic data augmentation improves pain detection performance on real clinical data, achieving a 7.0% improvement in average precision. SynPAIN addresses critical gaps in pain assessment research by providing the first publicly available, demographically diverse synthetic dataset specifically designed for older adult pain detection, while establishing a framework for measuring and mitigating algorithmic bias. The dataset is available at https://doi.org/10.5683/SP3/WCXMAP
>
---
#### [replaced 021] Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12781v2](http://arxiv.org/pdf/2410.12781v2)**

> **作者:** Chen Ziwen; Hao Tan; Kai Zhang; Sai Bi; Fujun Luan; Yicong Hong; Li Fuxin; Zexiang Xu
>
> **摘要:** We propose Long-LRM, a feed-forward 3D Gaussian reconstruction model for instant, high-resolution, 360{\deg} wide-coverage, scene-level reconstruction. Specifically, it takes in 32 input images at a resolution of 960x540 and produces the Gaussian reconstruction in just 1 second on a single A100 GPU. To handle the long sequence of 250K tokens brought by the large input size, Long-LRM features a mixture of the recent Mamba2 blocks and the classical transformer blocks, enhanced by a light-weight token merging module and Gaussian pruning steps that balance between quality and efficiency. We evaluate Long-LRM on the large-scale DL3DV benchmark and Tanks&Temples, demonstrating reconstruction quality comparable to the optimization-based methods while achieving an 800x speedup w.r.t. the optimization-based approaches and an input size at least 60x larger than the previous feed-forward approaches. We conduct extensive ablation studies on our model design choices for both rendering quality and computation efficiency. We also explore Long-LRM's compatibility with other Gaussian variants such as 2D GS, which enhances Long-LRM's ability in geometry reconstruction. Project page: https://arthurhero.github.io/projects/llrm
>
---
#### [replaced 022] CPCL: Cross-Modal Prototypical Contrastive Learning for Weakly Supervised Text-based Person Retrieval
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.10011v2](http://arxiv.org/pdf/2401.10011v2)**

> **作者:** Xinpeng Zhao; Yanwei Zheng; Chuanlin Lan; Xiaowei Zhang; Bowen Huang; Jibin Yang; Dongxiao Yu
>
> **备注:** 9 pages, 6 figures, under peer review
>
> **摘要:** Weakly supervised text-based person retrieval seeks to retrieve images of a target person using textual descriptions, without relying on identity annotations and is more challenging and practical. The primary challenge is the intra-class differences, encompassing intra-modal feature variations and cross-modal semantic gaps. Prior works have focused on instance-level samples and ignored prototypical features of each person which are intrinsic and invariant. Toward this, we propose a Cross-Modal Prototypical Contrastive Learning (CPCL) method. In practice, the CPCL introduces the CLIP model to weakly supervised text-based person retrieval to map visual and textual instances into a shared latent space. Subsequently, the proposed Prototypical Multi-modal Memory (PMM) module captures associations between heterogeneous modalities of image-text pairs belonging to the same person through the Hybrid Cross-modal Matching (HCM) module in a many-to-many mapping fashion. Moreover, the Outlier Pseudo Label Mining (OPLM) module further distinguishes valuable outlier samples from each modality, enhancing the creation of more reliable clusters by mining implicit relationships between image-text pairs. We conduct extensive experiments on popular benchmarks of weakly supervised text-based person retrieval, which validate the effectiveness, generalizability of CPCL.
>
---
#### [replaced 023] FakeIDet: Exploring Patches for Privacy-Preserving Fake ID Detection
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2504.07761v2](http://arxiv.org/pdf/2504.07761v2)**

> **作者:** Javier Muñoz-Haro; Ruben Tolosana; Ruben Vera-Rodriguez; Aythami Morales; Julian Fierrez
>
> **摘要:** Verifying the authenticity of identity documents (IDs) has become a critical challenge for real-life applications such as digital banking, crypto-exchanges, renting, etc. This study focuses on the topic of fake ID detection, covering several limitations in the field. In particular, there are no publicly available data from real IDs for proper research in this area, and most published studies rely on proprietary internal databases that are not available for privacy reasons. In order to advance this critical challenge of real data scarcity that makes it so difficult to advance the technology of machine learning-based fake ID detection, we introduce a new patch-based methodology that trades off privacy and performance, and propose a novel patch-wise approach for privacy-aware fake ID detection: FakeIDet. In our experiments, we explore: i) two levels of anonymization for an ID (i.e., fully- and pseudo-anonymized), and ii) different patch size configurations, varying the amount of sensitive data visible in the patch image. State-of-the-art methods, such as vision transformers and foundation models, are considered as backbones. Our results show that, on an unseen database (DLC-2021), our proposal for fake ID detection achieves 13.91% and 0% EERs at the patch and the whole ID level, showing a good generalization to other databases. In addition to the path-based methodology introduced and the new FakeIDet method based on it, another key contribution of our article is the release of the first publicly available database that contains 48,400 patches from real and fake IDs, called FakeIDet-db, together with the experimental framework.
>
---
#### [replaced 024] The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05101v3](http://arxiv.org/pdf/2412.05101v3)**

> **作者:** Ruoyu Wang; Huayang Huang; Ye Zhu; Olga Russakovsky; Yu Wu
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** In this work, we introduce NoiseQuery as a novel method for enhanced noise initialization in versatile goal-driven text-to-image (T2I) generation. Specifically, we propose to leverage an aligned Gaussian noise as implicit guidance to complement explicit user-defined inputs, such as text prompts, for better generation quality and controllability. Unlike existing noise optimization methods designed for specific models, our approach is grounded in a fundamental examination of the generic finite-step noise scheduler design in diffusion formulation, allowing better generalization across different diffusion-based architectures in a tuning-free manner. This model-agnostic nature allows us to construct a reusable noise library compatible with multiple T2I models and enhancement techniques, serving as a foundational layer for more effective generation. Extensive experiments demonstrate that NoiseQuery enables fine-grained control and yields significant performance boosts not only over high-level semantics but also over low-level visual attributes, which are typically difficult to specify through text alone, with seamless integration into current workflows with minimal computational overhead.
>
---
#### [replaced 025] Retinex-MEF: Retinex-based Glare Effects Aware Unsupervised Multi-Exposure Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07235v2](http://arxiv.org/pdf/2503.07235v2)**

> **作者:** Haowen Bai; Jiangshe Zhang; Zixiang Zhao; Lilun Deng; Yukun Cui; Shuang Xu
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Multi-exposure image fusion (MEF) synthesizes multiple, differently exposed images of the same scene into a single, well-exposed composite. Retinex theory, which separates image illumination from scene reflectance, provides a natural framework to ensure consistent scene representation and effective information fusion across varied exposure levels. However, the conventional pixel-wise multiplication of illumination and reflectance inadequately models the glare effect induced by overexposure. To address this limitation, we introduce an unsupervised and controllable method termed Retinex-MEF. Specifically, our method decomposes multi-exposure images into separate illumination components with a shared reflectance component, and effectively models the glare induced by overexposure. The shared reflectance is learned via a bidirectional loss, which enables our approach to effectively mitigate the glare effect. Furthermore, we introduce a controllable exposure fusion criterion, enabling global exposure adjustments while preserving contrast, thus overcoming the constraints of a fixed exposure level. Extensive experiments on diverse datasets, including underexposure-overexposure fusion, exposure controlled fusion, and homogeneous extreme exposure fusion, demonstrate the effective decomposition and flexible fusion capability of our model. The code is available at https://github.com/HaowenBai/Retinex-MEF
>
---
#### [replaced 026] Occlusion Boundary and Depth: Mutual Enhancement via Multi-Task Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21231v2](http://arxiv.org/pdf/2505.21231v2)**

> **作者:** Lintao Xu; Yinghao Wang; Chaohui Wang
>
> **备注:** 8 pages, 4 tables, 4 figures
>
> **摘要:** Occlusion Boundary Estimation (OBE) identifies boundaries arising from both inter-object occlusions and self-occlusion within individual objects, distinguishing them from ordinary edges and semantic contours to support more accurate scene understanding. This task is closely related to Monocular Depth Estimation (MDE), which infers depth from a single image, as Occlusion Boundaries (OBs) provide critical geometric cues for resolving depth ambiguities, while depth can conversely refine occlusion reasoning. In this paper, we propose MoDOT, a novel method that jointly estimates depth and OBs from a single image for the first time. MoDOT incorporates a new module, CASM, which combines cross-attention and multi-scale strip convolutions to leverage mid-level OB features for improved depth prediction. It also includes an occlusion-aware loss, OBDCL, which encourages more accurate boundaries in the predicted depth map. Extensive experiments demonstrate the mutual benefits of jointly estimating depth and OBs, and validate the effectiveness of MoDOT's design. Our method achieves state-of-the-art (SOTA) performance on two synthetic datasets and the widely used NYUD-v2 real-world dataset, significantly outperforming multi-task baselines. Furthermore, the cross-domain results of MoDOT on real-world depth prediction - trained solely on our synthetic dataset - yield promising results, preserving sharp OBs in the predicted depth maps and demonstrating improved geometric fidelity compared to competitors. We will release our code, pre-trained models, and dataset at [link].
>
---
#### [replaced 027] Simultaneous Motion And Noise Estimation with Event Cameras
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.04029v2](http://arxiv.org/pdf/2504.04029v2)**

> **作者:** Shintaro Shiba; Yoshimitsu Aoki; Guillermo Gallego
>
> **备注:** 13 pages, 13 figures, 6 tables, Project page https://github.com/tub-rip/ESMD
>
> **摘要:** Event cameras are emerging vision sensors whose noise is challenging to characterize. Existing denoising methods for event cameras are often designed in isolation and thus consider other tasks, such as motion estimation, separately (i.e., sequentially after denoising). However, motion is an intrinsic part of event data, since scene edges cannot be sensed without motion. We propose, to the best of our knowledge, the first method that simultaneously estimates motion in its various forms (e.g., ego-motion, optical flow) and noise. The method is flexible, as it allows replacing the one-step motion estimation of the widely-used Contrast Maximization framework with any other motion estimator, such as deep neural networks. The experiments show that the proposed method achieves state-of-the-art results on the E-MLB denoising benchmark and competitive results on the DND21 benchmark, while demonstrating effectiveness across motion estimation and intensity reconstruction tasks. Our approach advances event-data denoising theory and expands practical denoising use-cases via open-source code. Project page: https://github.com/tub-rip/ESMD
>
---
#### [replaced 028] Querying Autonomous Vehicle Point Clouds: Enhanced by 3D Object Counting with CounterNet
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.19209v2](http://arxiv.org/pdf/2507.19209v2)**

> **作者:** Xiaoyu Zhang; Zhifeng Bao; Hai Dong; Ziwei Wang; Jiajun Liu
>
> **摘要:** Autonomous vehicles generate massive volumes of point cloud data, yet only a subset is relevant for specific tasks such as collision detection, traffic analysis, or congestion monitoring. Effectively querying this data is essential to enable targeted analytics. In this work, we formalize point cloud querying by defining three core query types: RETRIEVAL, COUNT, and AGGREGATION, each aligned with distinct analytical scenarios. All these queries rely heavily on accurate object counts to produce meaningful results, making precise object counting a critical component of query execution. Prior work has focused on indexing techniques for 2D video data, assuming detection models provide accurate counting information. However, when applied to 3D point cloud data, state-of-the-art detection models often fail to generate reliable object counts, leading to substantial errors in query results. To address this limitation, we propose CounterNet, a heatmap-based network designed for accurate object counting in large-scale point cloud data. Rather than focusing on accurate object localization, CounterNet detects object presence by finding object centers to improve counting accuracy. We further enhance its performance with a feature map partitioning strategy using overlapping regions, enabling better handling of both small and large objects in complex traffic scenes. To adapt to varying frame characteristics, we introduce a per-frame dynamic model selection strategy that selects the most effective configuration for each input. Evaluations on three real-world autonomous vehicle datasets show that CounterNet improves counting accuracy by 5% to 20% across object categories, resulting in more reliable query outcomes across all supported query types.
>
---
#### [replaced 029] Navigating Distribution Shifts in Medical Image Analysis: A Survey
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05824v2](http://arxiv.org/pdf/2411.05824v2)**

> **作者:** Zixian Su; Jingwei Guo; Xi Yang; Qiufeng Wang; Frans Coenen; Kaizhu Huang
>
> **摘要:** Medical Image Analysis (MedIA) has become indispensable in modern healthcare, enhancing clinical diagnostics and personalized treatment. Despite the remarkable advancements supported by deep learning (DL) technologies, their practical deployment faces challenges due to distribution shifts, where models trained on specific datasets underperform across others from varying hospitals, regions, or patient populations. To navigate this issue, researchers have been actively developing strategies to increase the adaptability and robustness of DL models, enabling their effective use in unfamiliar and diverse environments. This paper systematically reviews approaches that apply DL techniques to MedIA systems affected by distribution shifts. Unlike traditional categorizations based on technical specifications, our approach is grounded in the real-world operational constraints faced by healthcare institutions. Specifically, we categorize the existing body of work into Joint Training, Federated Learning, Fine-tuning, and Domain Generalization, with each method tailored to distinct scenarios caused by Data Accessibility, Privacy Concerns, and Collaborative Protocols. This perspective equips researchers with a nuanced understanding of how DL can be strategically deployed to address distribution shifts in MedIA, ensuring diverse and robust medical applications. By delving deeper into these topics, we highlight potential pathways for future research that not only address existing limitations but also push the boundaries of deployable MedIA technologies.
>
---
#### [replaced 030] Aesthetics is Cheap, Show me the Text: An Empirical Evaluation of State-of-the-Art Generative Models for OCR
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15085v2](http://arxiv.org/pdf/2507.15085v2)**

> **作者:** Peirong Zhang; Haowei Xu; Jiaxin Zhang; Guitao Xu; Xuhan Zheng; Zhenhua Yang; Junle Liu; Yuyi Zhang; Lianwen Jin
>
> **摘要:** Text image is a unique and crucial information medium that integrates visual aesthetics and linguistic semantics in modern e-society. Due to their subtlety and complexity, the generation of text images represents a challenging and evolving frontier in the image generation field. The recent surge of specialized image generators (\emph{e.g.}, Flux-series) and unified generative models (\emph{e.g.}, GPT-4o), which demonstrate exceptional fidelity, raises a natural question: can they master the intricacies of text image generation and editing? Motivated by this, we assess current state-of-the-art generative models' capabilities in terms of text image generation and editing. We incorporate various typical optical character recognition (OCR) tasks into our evaluation and broaden the concept of text-based generation tasks into OCR generative tasks. We select 33 representative tasks and categorize them into five categories: document, handwritten text, scene text, artistic text, and complex \& layout-rich text. For comprehensive evaluation, we examine six models across both closed-source and open-source domains, using tailored, high-quality image inputs and prompts. Through this evaluation, we draw crucial observations and identify the weaknesses of current generative models for OCR tasks. We argue that photorealistic text image generation and editing should be internalized as foundational skills into general-domain generative models, rather than being delegated to specialized solutions, and we hope this empirical analysis can provide valuable insights for the community to achieve this goal. This evaluation is online and will be continuously updated at our GitHub repository.
>
---
#### [replaced 031] DiFuse-Net: RGB and Dual-Pixel Depth Estimation using Window Bi-directional Parallax Attention and Cross-modal Transfer Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14709v2](http://arxiv.org/pdf/2506.14709v2)**

> **作者:** Kunal Swami; Debtanu Gupta; Amrit Kumar Muduli; Chirag Jaiswal; Pankaj Kumar Bajpai
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** Depth estimation is crucial for intelligent systems, enabling applications from autonomous navigation to augmented reality. While traditional stereo and active depth sensors have limitations in cost, power, and robustness, dual-pixel (DP) technology, ubiquitous in modern cameras, offers a compelling alternative. This paper introduces DiFuse-Net, a novel modality decoupled network design for disentangled RGB and DP based depth estimation. DiFuse-Net features a window bi-directional parallax attention mechanism (WBiPAM) specifically designed to capture the subtle DP disparity cues unique to smartphone cameras with small aperture. A separate encoder extracts contextual information from the RGB image, and these features are fused to enhance depth prediction. We also propose a Cross-modal Transfer Learning (CmTL) mechanism to utilize large-scale RGB-D datasets in the literature to cope with the limitations of obtaining large-scale RGB-DP-D dataset. Our evaluation and comparison of the proposed method demonstrates its superiority over the DP and stereo-based baseline methods. Additionally, we contribute a new, high-quality, real-world RGB-DP-D training dataset, named Dual-Camera Dual-Pixel (DCDP) dataset, created using our novel symmetric stereo camera hardware setup, stereo calibration and rectification protocol, and AI stereo disparity estimation method.
>
---
#### [replaced 032] $\texttt{BATCLIP}$: Bimodal Online Test-Time Adaptation for CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02837v3](http://arxiv.org/pdf/2412.02837v3)**

> **作者:** Sarthak Kumar Maharana; Baoming Zhang; Leonid Karlinsky; Rogerio Feris; Yunhui Guo
>
> **备注:** ICCV 2025
>
> **摘要:** Although open-vocabulary classification models like Contrastive Language Image Pretraining (CLIP) have demonstrated strong zero-shot learning capabilities, their robustness to common image corruptions remains poorly understood. Through extensive experiments, we show that zero-shot CLIP lacks robustness to common image corruptions during test-time, necessitating the adaptation of CLIP to unlabeled corrupted images using test-time adaptation (TTA). However, we found that existing TTA methods have severe limitations in adapting CLIP due to their unimodal nature. To address these limitations, we propose $\texttt{BATCLIP}$, a bimodal $\textbf{online}$ TTA method designed to improve CLIP's robustness to common image corruptions. The key insight of our approach is not only to adapt the visual encoders for improving image features but also to strengthen the alignment between image and text features by promoting a stronger association between the image class prototype, computed using pseudo-labels, and the corresponding text feature. We evaluate our approach on benchmark image corruption datasets and achieve state-of-the-art results in online TTA for CLIP. Furthermore, we evaluate our proposed TTA approach on various domain generalization datasets to demonstrate its generalization capabilities. Our code is available at https://github.com/sarthaxxxxx/BATCLIP
>
---
#### [replaced 033] Detection, Pose Estimation and Segmentation for Multiple Bodies: Closing the Virtuous Circle
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01562v3](http://arxiv.org/pdf/2412.01562v3)**

> **作者:** Miroslav Purkrabek; Jiri Matas
>
> **备注:** Project Website: https://mirapurkrabek.github.io/BBox-Mask-Pose
>
> **摘要:** Human pose estimation methods work well on isolated people but struggle with multiple-bodies-in-proximity scenarios. Previous work has addressed this problem by conditioning pose estimation by detected bounding boxes or keypoints, but overlooked instance masks. We propose to iteratively enforce mutual consistency of bounding boxes, instance masks, and poses. The introduced BBox-Mask-Pose (BMP) method uses three specialized models that improve each other's output in a closed loop. All models are adapted for mutual conditioning, which improves robustness in multi-body scenes. MaskPose, a new mask-conditioned pose estimation model, is the best among top-down approaches on OCHuman. BBox-Mask-Pose pushes SOTA on OCHuman dataset in all three tasks - detection, instance segmentation, and pose estimation. It also achieves SOTA performance on COCO pose estimation. The method is especially good in scenes with large instances overlap, where it improves detection by 39% over the baseline detector. With small specialized models and faster runtime, BMP is an effective alternative to large human-centered foundational models. Code and models are available on https://MiraPurkrabek.github.io/BBox-Mask-Pose.
>
---
#### [replaced 034] H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.23523v2](http://arxiv.org/pdf/2507.23523v2)**

> **作者:** Hongzhe Bi; Lingxuan Wu; Tianwei Lin; Hengkai Tan; Zhizhong Su; Hang Su; Jun Zhu
>
> **摘要:** Imitation learning for robotic manipulation faces a fundamental challenge: the scarcity of large-scale, high-quality robot demonstration data. Recent robotic foundation models often pre-train on cross-embodiment robot datasets to increase data scale, while they face significant limitations as the diverse morphologies and action spaces across different robot embodiments make unified training challenging. In this paper, we present H-RDT (Human to Robotics Diffusion Transformer), a novel approach that leverages human manipulation data to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning. We introduce a two-stage training paradigm: (1) pre-training on large-scale egocentric human manipulation data, and (2) cross-embodiment fine-tuning on robot-specific data with modular action encoders and decoders. Built on a diffusion transformer architecture with 2B parameters, H-RDT uses flow matching to model complex action distributions. Extensive evaluations encompassing both simulation and real-world experiments, single-task and multitask scenarios, as well as few-shot learning and robustness assessments, demonstrate that H-RDT outperforms training from scratch and existing state-of-the-art methods, including Pi0 and RDT, achieving significant improvements of 13.9% and 40.5% over training from scratch in simulation and real-world experiments, respectively. The results validate our core hypothesis that human manipulation data can serve as a powerful foundation for learning bimanual robotic manipulation policies.
>
---
#### [replaced 035] DONUT: A Decoder-Only Model for Trajectory Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06854v2](http://arxiv.org/pdf/2506.06854v2)**

> **作者:** Markus Knoche; Daan de Geus; Bastian Leibe
>
> **备注:** ICCV 2025. Project page at https://vision.rwth-aachen.de/donut
>
> **摘要:** Predicting the motion of other agents in a scene is highly relevant for autonomous driving, as it allows a self-driving car to anticipate. Inspired by the success of decoder-only models for language modeling, we propose DONUT, a Decoder-Only Network for Unrolling Trajectories. Unlike existing encoder-decoder forecasting models, we encode historical trajectories and predict future trajectories with a single autoregressive model. This allows the model to make iterative predictions in a consistent manner, and ensures that the model is always provided with up-to-date information, thereby enhancing performance. Furthermore, inspired by multi-token prediction for language modeling, we introduce an 'overprediction' strategy that gives the model the auxiliary task of predicting trajectories at longer temporal horizons. This allows the model to better anticipate the future and further improves performance. Through experiments, we demonstrate that our decoder-only approach outperforms the encoder-decoder baseline, and achieves new state-of-the-art results on the Argoverse 2 single-agent motion forecasting benchmark.
>
---
#### [replaced 036] Three-dimentional reconstruction of complex, dynamic population canopy architecture for crops with a novel point cloud completion model: A case study in Brassica napus rapeseed
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18292v2](http://arxiv.org/pdf/2506.18292v2)**

> **作者:** Ziyue Guo; Xin Yang; Yutao Shen; Yang Zhu; Lixi Jiang; Haiyan Cen
>
> **摘要:** Quantitative descriptions of the complete canopy architecture are essential for accurately evaluating crop photosynthesis and yield performance to guide ideotype design. Although various sensing technologies have been developed for three-dimensional (3D) reconstruction of individual plants and canopies, they failed to obtain an accurate description of canopy architectures due to severe occlusion among complex canopy architectures. We proposed an effective method for 3D reconstruction of complex, dynamic population canopy architecture for rapeseed crops with a novel point cloud completion model. A complete point cloud generation framework was developed for automated annotation of the training dataset by distinguishing surface points from occluded points within canopies. The crop population point cloud completion network (CP-PCN) was then designed with a multi-resolution dynamic graph convolutional encoder (MRDG) and a point pyramid decoder (PPD) to predict occluded points. To further enhance feature extraction, a dynamic graph convolutional feature extractor (DGCFE) module was proposed to capture structural variations over the whole rapeseed growth period. The results demonstrated that CP-PCN achieved chamfer distance (CD) values of 3.35 cm -4.51 cm over four growth stages, outperforming the state-of-the-art transformer-based method (PoinTr). Ablation studies confirmed the effectiveness of the MRDG and DGCFE modules. Moreover, the validation experiment demonstrated that the silique efficiency index developed from CP-PCN improved the overall accuracy of rapeseed yield prediction by 11.2% compared to that of using incomplete point clouds. The CP-PCN pipeline has the potential to be extended to other crops, significantly advancing the quantitatively analysis of in-field population canopy architectures.
>
---
#### [replaced 037] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15867v3](http://arxiv.org/pdf/2411.15867v3)**

> **作者:** Teng Zhou; Xiaoyu Zhang; Yongchuan Tang
>
> **摘要:** Panoramic Image Generation (PIG) aims to create coherent images of arbitrary lengths. Most existing methods fall in the joint diffusion paradigm, but their complex and heuristic crop connection designs often limit their ability to achieve multilevel coherence. By deconstructing this challenge into its core components, we find it naturally aligns with next-token prediction, leading us to adopt an autoregressive (AR) paradigm for PIG modeling. However, existing visual AR (VAR) models are limited to fixed-size generation, lacking the capability to produce panoramic images. In this paper, we propose PanoLlama, a novel framework that achieves endless and coherent panorama generation with the autoregressive paradigm. Our approach develops a training-free strategy that utilizes token redirection to overcome the size limitations of existing VAR models, enabling next-crop prediction in both horizontal and vertical directions. This refreshes the PIG pipeline while achieving SOTA performance in coherence (47.50%), fidelity(28.16%), and aesthetics (15%). Additionally, PanoLlama supports applications other PIG methods cannot achieve, including mask-free layout control, multi-scale and multi-guidance synthesis. To facilitate standardized evaluation, we also establish a dataset with 1,000 prompts spanning 100+ themes, providing a new testing benchmark for PIG research. The code is available at https://github.com/0606zt/PanoLlama.
>
---
#### [replaced 038] OpenSeg-R: Improving Open-Vocabulary Segmentation via Step-by-Step Visual Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16974v2](http://arxiv.org/pdf/2505.16974v2)**

> **作者:** Zongyan Han; Jiale Cao; Shuo Chen; Tong Wang; Jorma Laaksonen; Rao Muhammad Anwer
>
> **摘要:** Open-Vocabulary Segmentation (OVS) has drawn increasing attention for its capacity to generalize segmentation beyond predefined categories. However, existing methods typically predict segmentation masks with simple forward inference, lacking explicit reasoning and interpretability. This makes it challenging for OVS model to distinguish similar categories in open-world settings due to the lack of contextual understanding and discriminative visual cues. To address this limitation, we propose a step-by-step visual reasoning framework for open-vocabulary segmentation, named OpenSeg-R. The proposed OpenSeg-R leverages Large Multimodal Models (LMMs) to perform hierarchical visual reasoning before segmentation. Specifically, we generate both generic and image-specific reasoning for each image, forming structured triplets that explain the visual reason for objects in a coarse-to-fine manner. Based on these reasoning steps, we can compose detailed description prompts, and feed them to the segmentor to produce more accurate segmentation masks. To the best of our knowledge, OpenSeg-R is the first framework to introduce explicit step-by-step visual reasoning into OVS. Experimental results demonstrate that OpenSeg-R significantly outperforms state-of-the-art methods on open-vocabulary semantic segmentation across five benchmark datasets. Moreover, it achieves consistent gains across all metrics on open-vocabulary panoptic segmentation. Qualitative results further highlight the effectiveness of our reasoning-guided framework in improving both segmentation precision and interpretability. Our code is publicly available at https://github.com/Hanzy1996/OpenSeg-R.
>
---
#### [replaced 039] Training-free Geometric Image Editing on Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23300v2](http://arxiv.org/pdf/2507.23300v2)**

> **作者:** Hanshen Zhu; Zhen Zhu; Kaile Zhang; Yiming Gong; Yuliang Liu; Xiang Bai
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** We tackle the task of geometric image editing, where an object within an image is repositioned, reoriented, or reshaped while preserving overall scene coherence. Previous diffusion-based editing methods often attempt to handle all relevant subtasks in a single step, proving difficult when transformations become large or structurally complex. We address this by proposing a decoupled pipeline that separates object transformation, source region inpainting, and target region refinement. Both inpainting and refinement are implemented using a training-free diffusion approach, FreeFine. In experiments on our new GeoBench benchmark, which contains both 2D and 3D editing scenarios, FreeFine outperforms state-of-the-art alternatives in image fidelity, and edit precision, especially under demanding transformations. Code and benchmark are available at: https://github.com/CIawevy/FreeFine
>
---
#### [replaced 040] Lossless Token Merging Even Without Fine-Tuning in Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15160v2](http://arxiv.org/pdf/2505.15160v2)**

> **作者:** Jaeyeon Lee; Dong-Wan Choi
>
> **备注:** ECAI 2025
>
> **摘要:** Although Vision Transformers (ViTs) have become the standard architecture in computer vision, their massive sizes lead to significant computational overhead. Token compression techniques have attracted considerable attention to address this issue, but they often suffer from severe information loss, requiring extensive additional training to achieve practical performance. In this paper, we propose Adaptive Token Merging (ATM), a novel method that ensures lossless token merging, eliminating the need for fine-tuning while maintaining competitive performance. ATM adaptively reduces tokens across layers and batches by carefully adjusting layer-specific similarity thresholds, thereby preventing the undesirable merging of less similar tokens with respect to each layer. Furthermore, ATM introduces a novel token matching technique that considers not only similarity but also merging sizes, particularly for the final layers, to minimize the information loss incurred from each merging operation. We empirically validate our method across a wide range of pretrained models, demonstrating that ATM not only outperforms all existing training-free methods but also surpasses most training-intensive approaches, even without additional training. Remarkably, training-free ATM achieves over a 30% reduction in FLOPs for the DeiT-T and DeiT-S models without any drop in their original accuracy.
>
---
#### [replaced 041] YOLOO: You Only Learn from Others Once
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.00618v2](http://arxiv.org/pdf/2409.00618v2)**

> **作者:** Lipeng Gu; Mingqiang Wei; Xuefeng Yan; Dingkun Zhu; Wei Zhao; Haoran Xie
>
> **摘要:** Multi-modal 3D multi-object tracking (MOT) typically necessitates extensive computational costs of deep neural networks (DNNs) to extract multi-modal representations. In this paper, we propose an intriguing question: May we learn from multiple modalities only during training to avoid multi-modal input in the inference phase? To answer it, we propose \textbf{YOLOO}, a novel multi-modal 3D MOT paradigm: You Only Learn from Others Once. YOLOO empowers the point cloud encoder to learn a unified tri-modal representation (UTR) from point clouds and other modalities, such as images and textual cues, all at once. Leveraging this UTR, YOLOO achieves efficient tracking solely using the point cloud encoder without compromising its performance, fundamentally obviating the need for computationally intensive DNNs. Specifically, YOLOO includes two core components: a unified tri-modal encoder (UTEnc) and a flexible geometric constraint (F-GC) module. UTEnc integrates a point cloud encoder with image and text encoders adapted from pre-trained CLIP. It seamlessly fuses point cloud information with rich visual-textual knowledge from CLIP into the point cloud encoder, yielding highly discriminative UTRs that facilitate the association between trajectories and detections. Additionally, F-GC filters out mismatched associations with similar representations but significant positional discrepancies. It further enhances the robustness of UTRs without requiring any scene-specific tuning, addressing a key limitation of customized geometric constraints (e.g., 3D IoU). Lastly, high-quality 3D trajectories are generated by a traditional data association component. By integrating these advancements into a multi-modal 3D MOT scheme, our YOLOO achieves substantial gains in both robustness and efficiency.
>
---
#### [replaced 042] CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17312v2](http://arxiv.org/pdf/2507.17312v2)**

> **作者:** Peiqi Chen; Lei Yu; Yi Wan; Yingying Pei; Xinyi Liu; Yongxiang Yao; Yingying Zhang; Lixiang Ru; Liheng Zhong; Jingdong Chen; Ming Yang; Yongjun Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Semi-dense feature matching methods have shown strong performance in challenging scenarios. However, the existing pipeline relies on a global search across the entire feature map to establish coarse matches, limiting further improvements in accuracy and efficiency. Motivated by this limitation, we propose a novel pipeline, CasP, which leverages cascaded correspondence priors for guidance. Specifically, the matching stage is decomposed into two progressive phases, bridged by a region-based selective cross-attention mechanism designed to enhance feature discriminability. In the second phase, one-to-one matches are determined by restricting the search range to the one-to-many prior areas identified in the first phase. Additionally, this pipeline benefits from incorporating high-level features, which helps reduce the computational costs of low-level feature extraction. The acceleration gains of CasP increase with higher resolution, and our lite model achieves a speedup of $\sim2.2\times$ at a resolution of 1152 compared to the most efficient method, ELoFTR. Furthermore, extensive experiments demonstrate its superiority in geometric estimation, particularly with impressive cross-domain generalization. These advantages highlight its potential for latency-sensitive and high-robustness applications, such as SLAM and UAV systems. Code is available at https://github.com/pq-chen/CasP.
>
---
#### [replaced 043] HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11454v3](http://arxiv.org/pdf/2505.11454v3)**

> **作者:** Shaina Raza; Aravind Narayanan; Vahid Reza Khazaie; Ashmal Vayani; Mukund S. Chettiar; Amandeep Singh; Mubarak Shah; Deval Pandya
>
> **摘要:** Large multimodal models (LMMs) have been widely tested on tasks like visual question answering (VQA), image captioning, and grounding, but lack rigorous evaluation for alignment with human-centered (HC) values such as fairness, ethics, and inclusivity. To address this gap, we introduce \textbf{HumaniBench}, a novel benchmark of 32,000 real-world image-question pairs and an evaluation suite. Labels are generated via an AI-assisted pipeline and validated by experts. HumaniBench assesses LMMs across seven key alignment principles: fairness, ethics, empathy, inclusivity, reasoning, robustness, and multilinguality, through diverse open-ended and closed-ended VQA tasks. Grounded in AI ethics and real-world needs, these principles provide a holistic lens for societal impact. Benchmarking results on different LMM shows that proprietary models generally lead in reasoning, fairness, and multilinguality, while open-source models excel in robustness and grounding. Most models struggle to balance accuracy with ethical and inclusive behavior. Techniques like Chain-of-Thought prompting and test-time scaling improve alignment. As the first benchmark tailored for HC alignment, HumaniBench offers a rigorous testbed to diagnose limitations, and promote responsible LMM development. All data and code are publicly available for reproducibility. Keywords: HumaniBench, vision-language models, responsible AI benchmark, AI alignment evaluation, AI ethics assessment, fairness in AI models, visual question answering (VQA) benchmark, image captioning evaluation, visual grounding tasks, trustworthy AI models, Chain-of-Thought prompting, test-time scaling, ethical AI development tools.
>
---
#### [replaced 044] Cross-Modal Dual-Causal Learning for Long-Term Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06603v2](http://arxiv.org/pdf/2507.06603v2)**

> **作者:** Xu Shaowu; Jia Xibin; Gao Junyu; Sun Qianmei; Chang Jing; Fan Chao
>
> **摘要:** Long-term action recognition (LTAR) is challenging due to extended temporal spans with complex atomic action correlations and visual confounders. Although vision-language models (VLMs) have shown promise, they often rely on statistical correlations instead of causal mechanisms. Moreover, existing causality-based methods address modal-specific biases but lack cross-modal causal modeling, limiting their utility in VLM-based LTAR. This paper proposes \textbf{C}ross-\textbf{M}odal \textbf{D}ual-\textbf{C}ausal \textbf{L}earning (CMDCL), which introduces a structural causal model to uncover causal relationships between videos and label texts. CMDCL addresses cross-modal biases in text embeddings via textual causal intervention and removes confounders inherent in the visual modality through visual causal intervention guided by the debiased text. These dual-causal interventions enable robust action representations to address LTAR challenges. Experimental results on three benchmarks including Charades, Breakfast and COIN, demonstrate the effectiveness of the proposed model. Our code is available at https://github.com/xushaowu/CMDCL.
>
---
#### [replaced 045] ShadowMamba: State-Space Model with Boundary-Region Selective Scan for Shadow Removal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03260v3](http://arxiv.org/pdf/2411.03260v3)**

> **作者:** Xiujin Zhu; Chee-Onn Chow; Joon Huang Chuah
>
> **摘要:** Image shadow removal is a common low-level vision problem. Shadows cause sudden brightness changes in some areas, which can affect the accuracy of downstream tasks. Currently, Transformer-based shadow removal methods improve computational efficiency by using a window mechanism. However, this approach reduces the effective receptive field and weakens the ability to model long-range dependencies in shadow images. Recently, Mamba has achieved significant success in computer vision by modeling long-sequence information globally with linear complexity. However, when applied to shadow removal, its original scanning mechanism overlooks the semantic continuity along shadow boundaries, and the coherence within each region. To solve this issue, we propose a new boundary-region selective scanning mechanism that scans shadow, boundary, and non-shadow regions separately, making pixels of the same type closer in the sequence. This increases semantic continuity and helps the model understand local details better. Incorporating this idea, we design the first Mamba-based lightweight shadow removal model, called ShadowMamba. It uses a hierarchical combination U-Net structure, which effectively reduces the number of parameters and computational complexity. Shallow layers rely on our boundary-region selective scanning to capture local details, while deeper layers use global cross-scanning to learn global brightness features. Extensive experiments show that ShadowMamba outperforms current state-of-the-art models on ISTD+, ISTD, and SRD datasets, and it also requires fewer parameters and less computational cost. (Code will be made available upon paper acceptance.)
>
---
#### [replaced 046] ProtoSolo: Interpretable Image Classification via Single-Prototype Activation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19808v3](http://arxiv.org/pdf/2506.19808v3)**

> **作者:** Yitao Peng; Lianghua He; Hongzhou Chen
>
> **摘要:** Although interpretable prototype networks have improved the transparency of deep learning image classification, the need for multiple prototypes in collaborative decision-making increases cognitive complexity and hinders user understanding. To solve this problem, this paper proposes a novel interpretable deep architecture for image classification, called ProtoSolo. Unlike existing prototypical networks, ProtoSolo requires activation of only a single prototype to complete the classification. This design significantly simplifies interpretation, as the explanation for each class requires displaying only the prototype with the highest similarity score and its corresponding feature map. Additionally, the traditional full-channel feature vector is replaced with a feature map for similarity comparison and prototype learning, enabling the use of richer global information within a single-prototype activation decision. A non-projection prototype learning strategy is also introduced to preserve the association between the prototype and image patch while avoiding abrupt structural changes in the network caused by projection, which can affect classification performance. Experiments on the CUB-200-2011 and Stanford Cars datasets demonstrate that ProtoSolo matches state-of-the-art interpretable methods in classification accuracy while achieving the lowest cognitive complexity. The code is available at https://github.com/pyt19/ProtoSolo.
>
---
#### [replaced 047] SkillFormer: Unified Multi-View Video Understanding for Proficiency Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08665v3](http://arxiv.org/pdf/2505.08665v3)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **摘要:** Assessing human skill levels in complex activities is a challenging problem with applications in sports, rehabilitation, and training. In this work, we present SkillFormer, a parameter-efficient architecture for unified multi-view proficiency estimation from egocentric and exocentric videos. Building on the TimeSformer backbone, SkillFormer introduces a CrossViewFusion module that fuses view-specific features using multi-head cross-attention, learnable gating, and adaptive self-calibration. We leverage Low-Rank Adaptation to fine-tune only a small subset of parameters, significantly reducing training costs. In fact, when evaluated on the EgoExo4D dataset, SkillFormer achieves state-of-the-art accuracy in multi-view settings while demonstrating remarkable computational efficiency, using 4.5x fewer parameters and requiring 3.75x fewer training epochs than prior baselines. It excels in multiple structured tasks, confirming the value of multi-view integration for fine-grained skill assessment.
>
---
#### [replaced 048] Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks
- **分类: cs.MM; cs.AI; cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.05695v3](http://arxiv.org/pdf/2502.05695v3)**

> **作者:** Zijiang Yan; Jianhua Pei; Hongda Wu; Hina Tabassum; Ping Wang
>
> **备注:** Accepted in IEEE Wireless Communications
>
> **摘要:** This paper proposes a novel Semantic Communication (SemCom) framework for real-time adaptive-bitrate video streaming by integrating Latent Diffusion Models (LDMs) within the FFmpeg techniques. This solution addresses the challenges of high bandwidth usage, storage inefficiencies, and quality of experience (QoE) degradation associated with traditional Constant Bitrate Streaming (CBS) and Adaptive Bitrate Streaming (ABS). The proposed approach leverages LDMs to compress I-frames into a latent space, offering significant storage and semantic transmission savings without sacrificing high visual quality. While retaining B-frames and P-frames as adjustment metadata to support efficient refinement of video reconstruction at the user side, the proposed framework further incorporates state-of-the-art denoising and Video Frame Interpolation (VFI) techniques. These techniques mitigate semantic ambiguity and restore temporal coherence between frames, even in noisy wireless communication environments. Experimental results demonstrate the proposed method achieves high-quality video streaming with optimized bandwidth usage, outperforming state-of-the-art solutions in terms of QoE and resource efficiency. This work opens new possibilities for scalable real-time video streaming in 5G and future post-5G networks.
>
---
#### [replaced 049] BlinkTrack: Feature Tracking over 80 FPS via Events and Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17981v2](http://arxiv.org/pdf/2409.17981v2)**

> **作者:** Yichen Shen; Yijin Li; Shuo Chen; Guanglin Li; Zhaoyang Huang; Hujun Bao; Zhaopeng Cui; Guofeng Zhang
>
> **摘要:** Event cameras, known for their high temporal resolution and ability to capture asynchronous changes, have gained significant attention for their potential in feature tracking, especially in challenging conditions. However, event cameras lack the fine-grained texture information that conventional cameras provide, leading to error accumulation in tracking. To address this, we propose a novel framework, BlinkTrack, which integrates event data with grayscale images for high-frequency feature tracking. Our method extends the traditional Kalman filter into a learning-based framework, utilizing differentiable Kalman filters in both event and image branches. This approach improves single-modality tracking and effectively solves the data association and fusion from asynchronous event and image data. We also introduce new synthetic and augmented datasets to better evaluate our model. Experimental results indicate that BlinkTrack significantly outperforms existing methods, exceeding 80 FPS with multi-modality data and 100 FPS with preprocessed event data. Codes and dataset are available at https://github.com/ColieShen/BlinkTrack.
>
---
#### [replaced 050] A Physical Model-Guided Framework for Underwater Image Enhancement and Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.04230v2](http://arxiv.org/pdf/2407.04230v2)**

> **作者:** Dazhao Du; Lingyu Si; Fanjiang Xu; Jianwei Niu; Fuchun Sun
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Due to the selective absorption and scattering of light by diverse aquatic media, underwater images usually suffer from various visual degradations. Existing underwater image enhancement (UIE) approaches that combine underwater physical imaging models with neural networks often fail to accurately estimate imaging model parameters such as depth and veiling light, resulting in poor performance in certain scenarios. To address this issue, we propose a physical model-guided framework for jointly training a Deep Degradation Model (DDM) with any advanced UIE model. DDM includes three well-designed sub-networks to accurately estimate various imaging parameters: a veiling light estimation sub-network, a factors estimation sub-network, and a depth estimation sub-network. Based on the estimated parameters and the underwater physical imaging model, we impose physical constraints on the enhancement process by modeling the relationship between underwater images and desired clean images, i.e., outputs of the UIE model. Moreover, while our framework is compatible with any UIE model, we design a simple yet effective fully convolutional UIE model, termed UIEConv. UIEConv utilizes both global and local features for image enhancement through a dual-branch structure. UIEConv trained within our framework achieves remarkable enhancement results across diverse underwater scenes. Furthermore, as a byproduct of UIE, the trained depth estimation sub-network enables accurate underwater scene depth estimation. Extensive experiments conducted in various real underwater imaging scenarios, including deep-sea environments with artificial light sources, validate the effectiveness of our framework and the UIEConv model.
>
---
#### [replaced 051] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.01064v4](http://arxiv.org/pdf/2412.01064v4)**

> **作者:** Taekyung Ki; Dongchan Min; Gyeongsu Chae
>
> **备注:** ICCV 2025. Project page: https://deepbrainai-research.github.io/float/
>
> **摘要:** With the rapid advancement of diffusion-based generative models, portrait image animation has achieved remarkable results. However, it still faces challenges in temporally consistent video generation and fast sampling due to its iterative sampling nature. This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow matching generative model. Instead of a pixel-based latent space, we take advantage of a learned orthogonal motion latent space, enabling efficient generation and editing of temporally consistent motion. To achieve this, we introduce a transformer-based vector field predictor with an effective frame-wise conditioning mechanism. Additionally, our method supports speech-driven emotion enhancement, enabling a natural incorporation of expressive motions. Extensive experiments demonstrate that our method outperforms state-of-the-art audio-driven talking portrait methods in terms of visual quality, motion fidelity, and efficiency.
>
---
#### [replaced 052] Spatial-Temporal-Spectral Unified Modeling for Remote Sensing Dense Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12280v3](http://arxiv.org/pdf/2505.12280v3)**

> **作者:** Sijie Zhao; Feng Liu; Enzhuo Zhang; Yiqing Guo; Pengfeng Xiao; Lei Bai; Xueliang Zhang; Hao Chen
>
> **备注:** 16 pages, 6 figures, Code link:https://github.com/walking-shadow/Official_TSSUN
>
> **摘要:** The proliferation of multi-source remote sensing data has propelled the development of deep learning for dense prediction, yet significant challenges in data and task unification persist. Current deep learning architectures for remote sensing are fundamentally rigid. They are engineered for fixed input-output configurations, restricting their adaptability to the heterogeneous spatial, temporal, and spectral dimensions inherent in real-world data. Furthermore, these models neglect the intrinsic correlations among semantic segmentation, binary change detection, and semantic change detection, necessitating the development of distinct models or task-specific decoders. This paradigm is also constrained to a predefined set of output semantic classes, where any change to the classes requires costly retraining. To overcome these limitations, we introduce the Spatial-Temporal-Spectral Unified Network (STSUN) for unified modeling. STSUN can adapt to input and output data with arbitrary spatial sizes, temporal lengths, and spectral bands by leveraging their metadata for a unified representation. Moreover, STSUN unifies disparate dense prediction tasks within a single architecture by conditioning the model on trainable task embeddings. Similarly, STSUN facilitates flexible prediction across multiple set of semantic categories by integrating trainable category embeddings as metadata. Extensive experiments on multiple datasets with diverse Spatial-Temporal-Spectral configurations in multiple scenarios demonstrate that a single STSUN model effectively adapts to heterogeneous inputs and outputs, unifying various dense prediction tasks and diverse semantic class predictions. The proposed approach consistently achieves state-of-the-art performance, highlighting its robustness and generalizability for complex remote sensing applications.
>
---
#### [replaced 053] MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.00043v2](http://arxiv.org/pdf/2507.00043v2)**

> **作者:** Mehmet Yigit Avci; Pedro Borges; Paul Wright; Mehmet Yigitsoy; Sebastien Ourselin; Jorge Cardoso
>
> **摘要:** Accurate interpretation of Magnetic Resonance Imaging scans in clinical systems is based on a precise understanding of image contrast. This contrast is primarily governed by acquisition parameters, such as echo time and repetition time, which are stored in the DICOM metadata. To simplify contrast identification, broad labels such as T1-weighted or T2-weighted are commonly used, but these offer only a coarse approximation of the underlying acquisition settings. In many real-world datasets, such labels are entirely missing, leaving raw acquisition parameters as the only indicators of contrast. Adding to this challenge, the available metadata is often incomplete, noisy, or inconsistent. The lack of reliable and standardized metadata complicates tasks such as image interpretation, retrieval, and integration into clinical workflows. Furthermore, robust contrast-aware representations are essential to enable more advanced clinical applications, such as achieving modality-invariant representations and data harmonization. To address these challenges, we propose MR-CLIP, a multimodal contrastive learning framework that aligns MR images with their DICOM metadata to learn contrast-aware representations, without relying on manual labels. Trained on a diverse clinical dataset that spans various scanners and protocols, MR-CLIP captures contrast variations across acquisitions and within scans, enabling anatomy-invariant representations. We demonstrate its effectiveness in cross-modal retrieval and contrast classification, highlighting its scalability and potential for further clinical applications. The code and weights are publicly available at https://github.com/myigitavci/MR-CLIP.
>
---
#### [replaced 054] GUAVA: Generalizable Upper Body 3D Gaussian Avatar
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03351v2](http://arxiv.org/pdf/2505.03351v2)**

> **作者:** Dongbin Zhang; Yunfei Liu; Lijian Lin; Ye Zhu; Yang Li; Minghan Qin; Yu Li; Haoqian Wang
>
> **备注:** Accepted to ICCV 2025, Project page: https://eastbeanzhang.github.io/GUAVA/
>
> **摘要:** Reconstructing a high-quality, animatable 3D human avatar with expressive facial and hand motions from a single image has gained significant attention due to its broad application potential. 3D human avatar reconstruction typically requires multi-view or monocular videos and training on individual IDs, which is both complex and time-consuming. Furthermore, limited by SMPLX's expressiveness, these methods often focus on body motion but struggle with facial expressions. To address these challenges, we first introduce an expressive human model (EHM) to enhance facial expression capabilities and develop an accurate tracking method. Based on this template model, we propose GUAVA, the first framework for fast animatable upper-body 3D Gaussian avatar reconstruction. We leverage inverse texture mapping and projection sampling techniques to infer Ubody (upper-body) Gaussians from a single image. The rendered images are refined through a neural refiner. Experimental results demonstrate that GUAVA significantly outperforms previous methods in rendering quality and offers significant speed improvements, with reconstruction times in the sub-second range (0.1s), and supports real-time animation and rendering.
>
---
#### [replaced 055] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17894v2](http://arxiv.org/pdf/2504.17894v2)**

> **作者:** Aniruddha Bala; Rohit Chowdhury; Rohan Jaiswal; Siddharth Roheda
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.
>
---
#### [replaced 056] HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19924v2](http://arxiv.org/pdf/2507.19924v2)**

> **作者:** Chang Liu; Yunfan Ye; Fan Zhang; Qingyang Zhou; Yuchuan Luo; Zhiping Cai
>
> **备注:** ICCV 2025. Project page: https://dejian-lc.github.io/humansam/
>
> **摘要:** Numerous synthesized videos from generative models, especially human-centric ones that simulate realistic human actions, pose significant threats to human information security and authenticity. While progress has been made in binary forgery video detection, the lack of fine-grained understanding of forgery types raises concerns regarding both reliability and interpretability, which are critical for real-world applications. To address this limitation, we propose HumanSAM, a new framework that builds upon the fundamental challenges of video generation models. Specifically, HumanSAM aims to classify human-centric forgeries into three distinct types of artifacts commonly observed in generated content: spatial, appearance, and motion anomaly. To better capture the features of geometry, semantics and spatiotemporal consistency, we propose to generate the human forgery representation by fusing two branches of video understanding and spatial depth. We also adopt a rank-based confidence enhancement strategy during the training process to learn more robust representation by introducing three prior scores. For training and evaluation, we construct the first public benchmark, the Human-centric Forgery Video (HFV) dataset, with all types of forgeries carefully annotated semi-automatically. In our experiments, HumanSAM yields promising results in comparison with state-of-the-art methods, both in binary and multi-class forgery classification.
>
---
#### [replaced 057] TerraMesh: A Planetary Mosaic of Multimodal Earth Observation Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11172v2](http://arxiv.org/pdf/2504.11172v2)**

> **作者:** Benedikt Blumenstiel; Paolo Fraccaro; Valerio Marsocci; Johannes Jakubik; Stefano Maurogiovanni; Mikolaj Czerkawski; Rocco Sedona; Gabriele Cavallaro; Thomas Brunschwiler; Juan Bernabe-Moreno; Nicolas Longépé
>
> **备注:** Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops
>
> **摘要:** Large-scale foundation models in Earth Observation can learn versatile, label-efficient representations by leveraging massive amounts of unlabeled data. However, existing public datasets are often limited in scale, geographic coverage, or sensor variety. We introduce TerraMesh, a new globally diverse, multimodal dataset combining optical, synthetic aperture radar, elevation, and land-cover modalities in an Analysis-Ready Data format. TerraMesh includes over 9~million samples with eight spatiotemporal aligned modalities, enabling large-scale pre-training. We provide detailed data processing steps, comprehensive statistics, and empirical evidence demonstrating improved model performance when pre-trained on TerraMesh. The dataset is hosted at https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh.
>
---
#### [replaced 058] Accurate Cross-modal Reconstruction of Vehicle Target from Sparse-aspect Multi-baseline SAR data
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2406.04158v5](http://arxiv.org/pdf/2406.04158v5)**

> **作者:** Da Li; Guoqiang Zhao; Chen Yao; Kaiqiang Zhu; Houjun Sun; Jiacheng Bao; Maokun Li
>
> **摘要:** Multi-aspect multi-baseline SAR 3D imaging is a critical remote sensing technique, promising in urban mapping and monitoring. However, sparse observation due to constrained flight trajectories degrade imaging quality, particularly for anisotropic small targets like vehicles and aircraft. In the past, compressive sensing (CS) was the mainstream approach for sparse 3D SAR reconstruction. More recently, deep learning (DL) has emerged as a powerful alternative, markedly boosting reconstruction quality and efficiency through strong data-driven representations capabilities and fast inference characteristics. However, existing DL methods typically train deep neural networks (DNNs) using only high-resolution radar images. This unimodal learning paradigm precludes the incorporation of complementary information from other data sources, thereby limiting potential improvements in reconstruction performance. In this paper, we introduce cross-modal learning and propose a Cross-Modal 3D-SAR Reconstruction Network (CMAR-Net) that enhances sparse 3D SAR reconstruction by fusing heterogeneous information. Leveraging cross-modal supervision from 2D optical images and error propagation guaranteed by differentiable rendering, CMAR-Net achieves efficient training and reconstructs highly sparse-aspect multi-baseline SAR image into visually structured and accurate 3D images, particularly for vehicle targets. Trained solely on simulated data, CMAR-Net exhibits strong generalization across extensive real-world evaluations on parking lot measurements containing numerous civilian vehicles, outperforming state-of-the-art CS and DL methods in structural accuracy. Our work highlights the potential of cross-modal learning for 3D SAR reconstruction and introduces a novel framework for radar imaging research.
>
---
#### [replaced 059] LLaVA-Video: Video Instruction Tuning With Synthetic Data
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02713v3](http://arxiv.org/pdf/2410.02713v3)**

> **作者:** Yuanhan Zhang; Jinming Wu; Wei Li; Bo Li; Zejun Ma; Ziwei Liu; Chunyuan Li
>
> **备注:** Project page: https://llava-vl.github.io/blog/2024-09-30-llava-video/; Accepted at TMLR
>
> **摘要:** The development of video large multimodal models (LMMs) has been hindered by the difficulty of curating large amounts of high-quality raw data from the web. To address this, we propose an alternative approach by creating a high-quality synthetic dataset specifically for video instruction-following, namely LLaVA-Video-178K. This dataset includes key tasks such as detailed captioning, open-ended question-answering (QA), and multiple-choice QA. By training on this dataset, in combination with existing visual instruction tuning data, we introduce LLaVA-Video, a new video LMM. Our experiments demonstrate that LLaVA-Video achieves strong performance across various video benchmarks, highlighting the effectiveness of our dataset. We plan to release the dataset, its generation pipeline, and the model checkpoints.
>
---
#### [replaced 060] Meta CLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22062v3](http://arxiv.org/pdf/2507.22062v3)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present Meta CLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, Meta CLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [replaced 061] FFGAF-SNN: The Forward-Forward Based Gradient Approximation Free Training Framework for Spiking Neural Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23643v2](http://arxiv.org/pdf/2507.23643v2)**

> **作者:** Changqing Xu; Ziqiang Yang; Yi Liu; Xinfang Liao; Guiqi Mo; Hao Zeng; Yintang Yang
>
> **摘要:** Spiking Neural Networks (SNNs) offer a biologically plausible framework for energy-efficient neuromorphic computing. However, it is a challenge to train SNNs due to their non-differentiability, efficiently. Existing gradient approximation approaches frequently sacrifice accuracy and face deployment limitations on edge devices due to the substantial computational requirements of backpropagation. To address these challenges, we propose a Forward-Forward (FF) based gradient approximation-free training framework for Spiking Neural Networks, which treats spiking activations as black-box modules, thereby eliminating the need for gradient approximation while significantly reducing computational complexity. Furthermore, we introduce a class-aware complexity adaptation mechanism that dynamically optimizes the loss function based on inter-class difficulty metrics, enabling efficient allocation of network resources across different categories. Experimental results demonstrate that our proposed training framework achieves test accuracies of 99.58%, 92.13%, and 75.64% on the MNIST, Fashion-MNIST, and CIFAR-10 datasets, respectively, surpassing all existing FF-based SNN approaches. Additionally, our proposed method exhibits significant advantages in terms of memory access and computational power consumption.
>
---
#### [replaced 062] Rethinking Pan-sharpening: Principled Design, Unified Training, and a Universal Loss Surpass Brute-Force Scaling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15059v2](http://arxiv.org/pdf/2507.15059v2)**

> **作者:** Ran Zhang; Xuanhua He; Li Xueheng; Ke Cao; Liu Liu; Wenbo Xu; Fang Jiabin; Yang Qize; Jie Zhang
>
> **摘要:** The field of pan-sharpening has recently seen a trend towards increasingly large and complex models, often trained on single, specific satellite datasets. This approach, however, leads to high computational overhead and poor generalization on full resolution data, a paradigm we challenge in this paper. In response to this issue, we propose PanTiny, a lightweight, single-step pan-sharpening framework designed for both efficiency and robust performance. More critically, we introduce multiple-in-one training paradigm, where a single, compact model is trained simultaneously on three distinct satellite datasets (WV2, WV3, and GF2) with different resolution and spectral information. Our experiments show that this unified training strategy not only simplifies deployment but also significantly boosts generalization on full-resolution data. Further, we introduce a universally powerful composite loss function that elevates the performance of almost all of models for pan-sharpening, pushing state-of-the-art metrics into a new era. Our PanTiny model, benefiting from these innovations, achieves a superior performance-to-efficiency balance, outperforming most larger, specialized models. Through extensive ablation studies, we validate that principled engineering in model design, training paradigms, and loss functions can surpass brute-force scaling. Our work advocates for a community-wide shift towards creating efficient, generalizable, and data-conscious models for pan-sharpening. The code is available at https://github.com/Zirconium233/PanTiny .
>
---
#### [replaced 063] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12811v2](http://arxiv.org/pdf/2504.12811v2)**

> **作者:** Michael Steiner; Thomas Köhler; Lukas Radl; Felix Windisch; Dieter Schmalstieg; Markus Steinberger
>
> **摘要:** Although 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it still faces challenges such as aliasing, projection artifacts, and view inconsistencies, primarily due to the simplification of treating splats as 2D entities. We argue that incorporating full 3D evaluation of Gaussians throughout the 3DGS pipeline can effectively address these issues while preserving rasterization efficiency. Specifically, we introduce an adaptive 3D smoothing filter to mitigate aliasing and present a stable view-space bounding method that eliminates popping artifacts when Gaussians extend beyond the view frustum. Furthermore, we promote tile-based culling to 3D with screen-space planes, accelerating rendering and reducing sorting costs for hierarchical rasterization. Our method achieves state-of-the-art quality on in-distribution evaluation sets and significantly outperforms other approaches for out-of-distribution views. Our qualitative evaluations further demonstrate the effective removal of aliasing, distortions, and popping artifacts, ensuring real-time, artifact-free rendering.
>
---
#### [replaced 064] C-DOG: Multi-View Multi-instance Feature Association Using Connected δ-Overlap Graphs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14095v2](http://arxiv.org/pdf/2507.14095v2)**

> **作者:** Yung-Hong Sun; Ting-Hung Lin; Jiangang Chen; Hongrui Jiang; Yu Hen Hu
>
> **摘要:** Multi-view multi-instance feature association constitutes a crucial step in 3D reconstruction, facilitating the consistent grouping of object instances across various camera perspectives. The presence of multiple identical objects within a scene often leads to ambiguities for appearance-based feature matching algorithms. Our work circumvents this challenge by exclusively employing geometrical constraints, specifically epipolar geometry, for feature association. We introduce C-DOG (Connected delta-Overlap Graph), an algorithm designed for robust geometrical feature association, even in the presence of noisy feature detections. In a C-DOG graph, two nodes representing 2D feature points from distinct views are connected by an edge if they correspond to the same 3D point. Each edge is weighted by its epipolar distance. Ideally, true associations yield a zero distance; however, noisy feature detections can result in non-zero values. To robustly retain edges where the epipolar distance is less than a threshold delta, we employ a Szymkiewicz--Simpson coefficient. This process leads to a delta-neighbor-overlap clustering of 2D nodes. Furthermore, unreliable nodes are pruned from these clusters using an Inter-quartile Range (IQR)-based criterion. Our extensive experiments on synthetic benchmarks demonstrate that C-DOG not only outperforms geometry-based baseline algorithms but also remains remarkably robust under demanding conditions. This includes scenes with high object density, no visual features, and restricted camera overlap, positioning C-DOG as an excellent solution for scalable 3D reconstruction in practical applications.
>
---
#### [replaced 065] GameFactory: Creating New Games with Generative Interactive Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08325v3](http://arxiv.org/pdf/2501.08325v3)**

> **作者:** Jiwen Yu; Yiran Qin; Xintao Wang; Pengfei Wan; Di Zhang; Xihui Liu
>
> **备注:** ICCV 2025 Highlight, Project Page: https://yujiwen.github.io/gamefactory
>
> **摘要:** Generative videos have the potential to revolutionize game development by autonomously creating new content. In this paper, we present GameFactory, a framework for action-controlled scene-generalizable game video generation. We first address the fundamental challenge of action controllability by introducing GF-Minecraft, an action-annotated game video dataset without human bias, and developing an action control module that enables precise control over both keyboard and mouse inputs. We further extend to support autoregressive generation for unlimited-length interactive videos. More importantly, GameFactory tackles the critical challenge of scene-generalizable action control, which most existing methods fail to address. To enable the creation of entirely new and diverse games beyond fixed styles and scenes, we leverage the open-domain generative priors from pre-trained video diffusion models. To bridge the domain gap between open-domain priors and small-scale game datasets, we propose a multi-phase training strategy with a domain adapter that decouples game style learning from action control. This decoupling ensures that action control learning is no longer bound to specific game styles, thereby achieving scene-generalizable action control. Experimental results demonstrate that GameFactory effectively generates open-domain action-controllable game videos, representing a significant step forward in AI-driven game generation.
>
---
#### [replaced 066] Model Stock: All we need is just a few fine-tuned models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.19522v2](http://arxiv.org/pdf/2403.19522v2)**

> **作者:** Dong-Hwan Jang; Sangdoo Yun; Dongyoon Han
>
> **备注:** ECCV 2024 oral presenetation; Code at https://github.com/naver-ai/model-stock
>
> **摘要:** This paper introduces an efficient fine-tuning method for large pre-trained models, offering strong in-distribution (ID) and out-of-distribution (OOD) performance. Breaking away from traditional practices that need a multitude of fine-tuned models for averaging, our approach employs significantly fewer models to achieve final weights yet yield superior accuracy. Drawing from key insights in the weight space of fine-tuned weights, we uncover a strong link between the performance and proximity to the center of weight space. Based on this, we introduce a method that approximates a center-close weight using only two fine-tuned models, applicable during or after training. Our innovative layer-wise weight averaging technique surpasses state-of-the-art model methods such as Model Soup, utilizing only two fine-tuned models. This strategy can be aptly coined Model Stock, highlighting its reliance on selecting a minimal number of models to draw a more optimized-averaged model. We demonstrate the efficacy of Model Stock with fine-tuned models based upon pre-trained CLIP architectures, achieving remarkable performance on both ID and OOD tasks on the standard benchmarks, all while barely bringing extra computational demands. Our code and pre-trained models are available at https://github.com/naver-ai/model-stock.
>
---
#### [replaced 067] YOLO-FireAD: Efficient Fire Detection via Attention-Guided Inverted Residual Learning and Dual-Pooling Feature Preservation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20884v3](http://arxiv.org/pdf/2505.20884v3)**

> **作者:** Weichao Pan; Bohan Xu; Xu Wang; Chengze Lv; Shuoyang Wang; Zhenke Duan; Zhen Tian
>
> **备注:** 2025 International Conference on Intelligent Computing (ICIC 2025)
>
> **摘要:** Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO-based models, this study propose You Only Look Once for Fire Detection with Attention-guided Inverted Residual and Dual-pooling Downscale Fusion (YOLO-FireAD) with two core innovations: (1) Attention-guided Inverted Residual Block (AIR) integrates hybrid channel-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; (2) Dual Pool Downscale Fusion Block (DPDF) preserves multi-scale fire patterns through learnable fusion of max-average pooling outputs, mitigating small-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters (1.45M, 51.8% lower than YOLOv8n) (4.6G, 43.2% lower than YOLOv8n), and mAP75 is higher than the mainstream real-time object detection models YOLOv8n, YOL-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3-5.5%. For more details, please visit our repository: https://github.com/JEFfersusu/YOLO-FireAD
>
---
#### [replaced 068] Multi-Cali Anything: Dense Feature Multi-Frame Structure-from-Motion for Large-Scale Camera Array Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00737v2](http://arxiv.org/pdf/2503.00737v2)**

> **作者:** Jinjiang You; Hewei Wang; Yijie Li; Mingxiao Huo; Long Van Tran Ha; Mingyuan Ma; Jinfeng Xu; Jiayi Zhang; Puzhen Wu; Shubham Garg; Wei Pu
>
> **备注:** Accepted to IROS 2025. Final camera-ready version. 8 pages
>
> **摘要:** Calibrating large-scale camera arrays, such as those in dome-based setups, is time-intensive and typically requires dedicated captures of known patterns. While extrinsics in such arrays are fixed due to the physical setup, intrinsics often vary across sessions due to factors like lens adjustments or temperature changes. In this paper, we propose a dense-feature-driven multi-frame calibration method that refines intrinsics directly from scene data, eliminating the necessity for additional calibration captures. Our approach enhances traditional Structure-from-Motion (SfM) pipelines by introducing an extrinsics regularization term to progressively align estimated extrinsics with ground-truth values, a dense feature reprojection term to reduce keypoint errors by minimizing reprojection loss in the feature space, and an intrinsics variance term for joint optimization across multiple frames. Experiments on the Multiface dataset show that our method achieves nearly the same precision as dedicated calibration processes, and significantly enhances intrinsics and 3D reconstruction accuracy. Fully compatible with existing SfM pipelines, our method provides an efficient and practical plug-and-play solution for large-scale camera setups. Our code is publicly available at: https://github.com/YJJfish/Multi-Cali-Anything
>
---
