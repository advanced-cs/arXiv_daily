# 计算机视觉 cs.CV

- **最新发布 84 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] NeuroMamba: Multi-Perspective Feature Interaction with Visual Mamba for Neuron Segmentation
- **分类: cs.CV**

- **简介: 论文提出NeuroMamba，用于神经元分割任务，解决边界模糊和细节丢失问题，通过多视角特征交互提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.15929v1](https://arxiv.org/pdf/2601.15929v1)**

> **作者:** Liuyun Jiang; Yizhuo Lu; Yanchao Zhang; Jiazheng Liu; Hua Han
>
> **摘要:** Neuron segmentation is the cornerstone of reconstructing comprehensive neuronal connectomes, which is essential for deciphering the functional organization of the brain. The irregular morphology and densely intertwined structures of neurons make this task particularly challenging. Prevailing CNN-based methods often fail to resolve ambiguous boundaries due to the lack of long-range context, whereas Transformer-based methods suffer from boundary imprecision caused by the loss of voxel-level details during patch partitioning. To address these limitations, we propose NeuroMamba, a multi-perspective framework that exploits the linear complexity of Mamba to enable patch-free global modeling and synergizes this with complementary local feature modeling, thereby efficiently capturing long-range dependencies while meticulously preserving fine-grained voxel details. Specifically, we design a channel-gated Boundary Discriminative Feature Extractor (BDFE) to enhance local morphological cues. Complementing this, we introduce the Spatial Continuous Feature Extractor (SCFE), which integrates a resolution-aware scanning mechanism into the Visual Mamba architecture to adaptively model global dependencies across varying data resolutions. Finally, a cross-modulation mechanism synergistically fuses these multi-perspective features. Our method demonstrates state-of-the-art performance across four public EM datasets, validating its exceptional adaptability to both anisotropic and isotropic resolutions. The source code will be made publicly available.
>
---
#### [new 002] Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于多模态理解任务，旨在解决CIR评估不足的问题。通过图像编辑构建细粒度基准EDIR，涵盖多种类别，评估模型性能并揭示现有基准的局限性。**

- **链接: [https://arxiv.org/pdf/2601.16125v1](https://arxiv.org/pdf/2601.16125v1)**

> **作者:** Tingyu Song; Yanzhao Zhang; Mingxin Li; Zhuoning Guo; Dingkun Long; Pengjun Xie; Siyue Zhang; Yilun Zhao; Shu Wu
>
> **备注:** Under review
>
> **摘要:** Composed Image Retrieval (CIR) is a pivotal and complex task in multimodal understanding. Current CIR benchmarks typically feature limited query categories and fail to capture the diverse requirements of real-world scenarios. To bridge this evaluation gap, we leverage image editing to achieve precise control over modification types and content, enabling a pipeline for synthesizing queries across a broad spectrum of categories. Using this pipeline, we construct EDIR, a novel fine-grained CIR benchmark. EDIR encompasses 5,000 high-quality queries structured across five main categories and fifteen subcategories. Our comprehensive evaluation of 13 multimodal embedding models reveals a significant capability gap; even state-of-the-art models (e.g., RzenEmbed and GME) struggle to perform consistently across all subcategories, highlighting the rigorous nature of our benchmark. Through comparative analysis, we further uncover inherent limitations in existing benchmarks, such as modality biases and insufficient categorical coverage. Furthermore, an in-domain training experiment demonstrates the feasibility of our benchmark. This experiment clarifies the task challenges by distinguishing between categories that are solvable with targeted data and those that expose intrinsic limitations of current model architectures.
>
---
#### [new 003] TinySense: Effective CSI Compression for Scalable and Accurate Wi-Fi Sensing
- **分类: cs.CV**

- **简介: 该论文属于Wi-Fi传感任务，旨在解决CSI数据压缩导致的资源消耗问题。通过VQGAN和Transformer优化压缩，提升HPE精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.15838v1](https://arxiv.org/pdf/2601.15838v1)**

> **作者:** Toan Gian; Dung T. Tran; Viet Quoc Pham; Francesco Restuccia; Van-Dinh Nguyen
>
> **备注:** 10 pages. This paper has been accepted for publication in IEEE PerCom 2026
>
> **摘要:** With the growing demand for device-free and privacy-preserving sensing solutions, Wi-Fi sensing has emerged as a promising approach for human pose estimation (HPE). However, existing methods often process vast amounts of channel state information (CSI) data directly, ultimately straining networking resources. This paper introduces TinySense, an efficient compression framework that enhances the scalability of Wi-Fi-based human sensing. Our approach is based on a new vector quantization-based generative adversarial network (VQGAN). Specifically, by leveraging a VQGAN-learned codebook, TinySense significantly reduces CSI data while maintaining the accuracy required for reliable HPE. To optimize compression, we employ the K-means algorithm to dynamically adjust compression bitrates to cluster a large-scale pre-trained codebook into smaller subsets. Furthermore, a Transformer model is incorporated to mitigate bitrate loss, enhancing robustness in unreliable networking conditions. We prototype TinySense on an experimental testbed using Jetson Nano and Raspberry Pi to measure latency and network resource use. Extensive results demonstrate that TinySense significantly outperforms state-of-the-art compression schemes, achieving up to 1.5x higher HPE accuracy score (PCK20) under the same compression rate. It also reduces latency and networking overhead, respectively, by up to 5x and 2.5x. The code repository is available online at here.
>
---
#### [new 004] Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation
- **分类: cs.CV**

- **简介: 该论文属于神经元分割任务，旨在解决标注数据不足和数据多样性低的问题。提出基于扩散模型的数据增强方法，生成结构合理的图像-标签对，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.15779v1](https://arxiv.org/pdf/2601.15779v1)**

> **作者:** Liuyun Jiang; Yanchao Zhang; Jinyue Guo; Yizhuo Lu; Ruining Zhou; Hua Han
>
> **摘要:** Neuron segmentation in electron microscopy (EM) aims to reconstruct the complete neuronal connectome; however, current deep learning-based methods are limited by their reliance on large-scale training data and extensive, time-consuming manual annotations. Traditional methods augment the training set through geometric and photometric transformations; however, the generated samples remain highly correlated with the original images and lack structural diversity. To address this limitation, we propose a diffusion-based data augmentation framework capable of generating diverse and structurally plausible image-label pairs for neuron segmentation. Specifically, the framework employs a resolution-aware conditional diffusion model with multi-scale conditioning and EM resolution priors to enable voxel-level image synthesis from 3D masks. It further incorporates a biology-guided mask remodeling module that produces augmented masks with enhanced structural realism. Together, these components effectively enrich the training set and improve segmentation performance. On the AC3 and AC4 datasets under low-annotation regimes, our method improves the ARAND metric by 32.1% and 30.7%, respectively, when combined with two different post-processing methods. Our code is available at https://github.com/HeadLiuYun/NeuroDiff.
>
---
#### [new 005] Zero-Shot Product Attribute Labeling with Vision-Language Models: A Three-Tier Evaluation Framework
- **分类: cs.CV**

- **简介: 该论文属于时尚属性标注任务，解决零样本多属性预测问题。提出三层评估框架，验证VLM在分类与适用性检测上的表现，发现其分类能力强但适用性检测弱。**

- **链接: [https://arxiv.org/pdf/2601.15711v1](https://arxiv.org/pdf/2601.15711v1)**

> **作者:** Shubham Shukla; Kunal Sonalkar
>
> **备注:** Accepted to WACV 2026 Workshop on Physical Retail AI (PRAW)
>
> **摘要:** Fine-grained attribute prediction is essential for fashion retail applications including catalog enrichment, visual search, and recommendation systems. Vision-Language Models (VLMs) offer zero-shot prediction without task-specific training, yet their systematic evaluation on multi-attribute fashion tasks remains underexplored. A key challenge is that fashion attributes are often conditional. For example, "outer fabric" is undefined when no outer garment is visible. This requires models to detect attribute applicability before attempting classification. We introduce a three-tier evaluation framework that decomposes this challenge: (1) overall task performance across all classes (including NA class: suggesting attribute is not applicable) for all attributes, (2) attribute applicability detection, and (3) fine-grained classification when attributes are determinable. Using DeepFashion-MultiModal, which explicitly defines NA (meaning attribute doesn't exist or is not visible) within attribute label spaces, we benchmark nine VLMs spanning flagship (GPT-5, Gemini 2.5 Pro), efficient (GPT-5 Mini, Gemini 2.5 Flash), and ultra-efficient tiers (GPT-5 Nano, Gemini 2.5 Flash-Lite) against classifiers trained on pretrained Fashion-CLIP embeddings on 5,000 images across 18 attributes. Our findings reveal that: (1) zero-shot VLMs achieve 64.0% macro-F1, a threefold improvement over logistic regression on pretrained Fashion-CLIP embeddings; (2) VLMs excel at fine-grained classification (Tier 3: 70.8% F1) but struggle with applicability detection (Tier 2: 34.1% NA-F1), identifying a key bottleneck; (3) efficient models achieve over 90% of flagship performance at lower cost, offering practical deployment paths. This diagnostic framework enables practitioners to pinpoint whether errors stem from visibility detection or classification, guiding targeted improvements for production systems.
>
---
#### [new 006] DeltaDorsal: Enhancing Hand Pose Estimation with Dorsal Features in Egocentric Views
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于手部姿态估计任务，旨在解决第一视角下手指遮挡导致的精度下降问题。通过引入双流delta编码器，利用背侧皮肤变形信息提升估计效果。**

- **链接: [https://arxiv.org/pdf/2601.15516v1](https://arxiv.org/pdf/2601.15516v1)**

> **作者:** William Huang; Siyou Pei; Leyi Zou; Eric J. Gonzalez; Ishan Chatterjee; Yang Zhang
>
> **备注:** 16 pages, 11 figures, Presented at ACM CHI 2026. For associated codebase, see https://github.com/hilab-open-source/deltadorsal
>
> **摘要:** The proliferation of XR devices has made egocentric hand pose estimation a vital task, yet this perspective is inherently challenged by frequent finger occlusions. To address this, we propose a novel approach that leverages the rich information in dorsal hand skin deformation, unlocked by recent advances in dense visual featurizers. We introduce a dual-stream delta encoder that learns pose by contrasting features from a dynamic hand with a baseline relaxed position. Our evaluation demonstrates that, using only cropped dorsal images, our method reduces the Mean Per Joint Angle Error (MPJAE) by 18% in self-occluded scenarios (fingers >=50% occluded) compared to state-of-the-art techniques that depend on the whole hand's geometry and large model backbones. Consequently, our method not only enhances the reliability of downstream tasks like index finger pinch and tap estimation in occluded scenarios but also unlocks new interaction paradigms, such as detecting isometric force for a surface "click" without visible movement while minimizing model size.
>
---
#### [new 007] DSFedMed: Dual-Scale Federated Medical Image Segmentation via Mutual Distillation Between Foundation and Lightweight Models
- **分类: cs.CV; cs.DC**

- **简介: 该论文提出DSFedMed，解决医疗图像分割中的联邦学习效率问题，通过基础模型与轻量模型的双向知识蒸馏，提升分割效果并降低计算和通信成本。**

- **链接: [https://arxiv.org/pdf/2601.16073v1](https://arxiv.org/pdf/2601.16073v1)**

> **作者:** Hanwen Zhang; Qiaojin Shen; Yuxi Liu; Yuesheng Zhu; Guibo Luo
>
> **摘要:** Foundation Models (FMs) have demonstrated strong generalization across diverse vision tasks. However, their deployment in federated settings is hindered by high computational demands, substantial communication overhead, and significant inference costs. We propose DSFedMed, a dual-scale federated framework that enables mutual knowledge distillation between a centralized foundation model and lightweight client models for medical image segmentation. To support knowledge distillation, a set of high-quality medical images is generated to replace real public datasets, and a learnability-guided sample selection strategy is proposed to enhance efficiency and effectiveness in dual-scale distillation. This mutual distillation enables the foundation model to transfer general knowledge to lightweight clients, while also incorporating client-specific insights to refine the foundation model. Evaluations on five medical imaging segmentation datasets show that DSFedMed achieves an average 2 percent improvement in Dice score while reducing communication costs and inference time by nearly 90 percent compared to existing federated foundation model baselines. These results demonstrate significant efficiency gains and scalability for resource-limited federated deployments.
>
---
#### [new 008] FAIR-ESI: Feature Adaptive Importance Refinement for Electrophysiological Source Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于脑疾病诊断任务，解决电生理源成像中的特征选择与优化问题，提出FAIR-ESI框架，通过多视角自适应特征精炼提升成像精度。**

- **链接: [https://arxiv.org/pdf/2601.15731v1](https://arxiv.org/pdf/2601.15731v1)**

> **作者:** Linyong Zou; Liang Zhang; Xiongfei Wang; Jia-Hong Gao; Yi Sun; Shurong Sheng; Kuntao Xiao; Wanli Yang; Pengfei Teng; Guoming Luan; Zhao Lv; Zikang Xu
>
> **摘要:** An essential technique for diagnosing brain disorders is electrophysiological source imaging (ESI). While model-based optimization and deep learning methods have achieved promising results in this field, the accurate selection and refinement of features remains a central challenge for precise ESI. This paper proposes FAIR-ESI, a novel framework that adaptively refines feature importance across different views, including FFT-based spectral feature refinement, weighted temporal feature refinement, and self-attention-based patch-wise feature refinement. Extensive experiments on two simulation datasets with diverse configurations and two real-world clinical datasets validate our framework's efficacy, highlighting its potential to advance brain disorder diagnosis and offer new insights into brain function.
>
---
#### [new 009] DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言动作模型任务，解决模型过度关注无关图像区域的问题。提出DTP框架动态去除干扰令牌，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.16065v1](https://arxiv.org/pdf/2601.16065v1)**

> **作者:** Chenyang Li; Jieyuan Liu; Bin Li; Bo Gao; Yilin Yuan; Yangfan He; Yuchen Li; Jingqun Tang
>
> **摘要:** Vision-Language Action (VLA) models have shown remarkable progress in robotic manipulation by leveraging the powerful perception abilities of Vision-Language Models (VLMs) to understand environments and directly output actions. However, by default, VLA models may overly attend to image tokens in the task-irrelevant region, which we describe as 'distracting tokens'. This behavior can disturb the model from the generation of the desired action tokens in each step, affecting the success rate of tasks. In this paper, we introduce a simple yet effective plug-and-play Distracting Token Pruning (DTP) framework, which dynamically detects and prunes these distracting image tokens. By correcting the model's visual attention patterns, we aim to improve the task success rate, as well as exploring the performance upper boundaries of the model without altering its original architecture or adding additional inputs. Experiments on the SIMPLER Benchmark (Li et al., 2024) show that our method consistently achieving relative improvements in task success rates across different types of novel VLA models, demonstrating generalizability to transformer-based VLAs. Further analysis reveals a negative correlation between the task success rate and the amount of attentions in the task-irrelevant region for all models tested, highlighting a common phenomenon of VLA models that could guide future research. We also publish our code at: https://anonymous.4open.science/r/CBD3.
>
---
#### [new 010] Assessing Situational and Spatial Awareness of VLMs with Synthetically Generated Video
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在评估模型在视频中对情境和空间关系的理解能力。研究提出一个合成基准，测试识别暴力行为、角色绑定及轨迹判断等挑战。**

- **链接: [https://arxiv.org/pdf/2601.15780v1](https://arxiv.org/pdf/2601.15780v1)**

> **作者:** Pascal Benschop; Justin Dauwels; Jan van Gemert
>
> **摘要:** Spatial reasoning in vision language models (VLMs) remains fragile when semantics hinge on subtle temporal or geometric cues. We introduce a synthetic benchmark that probes two complementary skills: situational awareness (recognizing whether an interaction is harmful or benign) and spatial awareness (tracking who does what to whom, and reasoning about relative positions and motion). Through minimal video pairs, we test three challenges: distinguishing violence from benign activity, binding assailant roles across viewpoints, and judging fine-grained trajectory alignment. While we evaluate recent VLMs in a training-free setting, the benchmark is applicable to any video classification model. Results show performance only slightly above chance across tasks. A simple aid, stable color cues, partly reduces assailant role confusions but does not resolve the underlying weakness. By releasing data and code, we aim to provide reproducible diagnostics and seed exploration of lightweight spatial priors to complement large-scale pretraining.
>
---
#### [new 011] Opening the Black Box: Preliminary Insights into Affective Modeling in Multimodal Foundation Models
- **分类: cs.CV**

- **简介: 该论文研究多模态基础模型中的情感建模问题，探讨情感如何被表示。通过分析不同架构和任务，发现情感适应主要集中在门控投影层（gate_proj），而非注意力模块。**

- **链接: [https://arxiv.org/pdf/2601.15906v1](https://arxiv.org/pdf/2601.15906v1)**

> **作者:** Zhen Zhang; Runhao Zeng; Sicheng Zhao; Xiping Hu
>
> **摘要:** Understanding where and how emotions are represented in large-scale foundation models remains an open problem, particularly in multimodal affective settings. Despite the strong empirical performance of recent affective models, the internal architectural mechanisms that support affective understanding and generation are still poorly understood. In this work, we present a systematic mechanistic study of affective modeling in multimodal foundation models. Across multiple architectures, training strategies, and affective tasks, we analyze how emotion-oriented supervision reshapes internal model parameters. Our results consistently reveal a clear and robust pattern: affective adaptation does not primarily focus on the attention module, but instead localizes to the feed-forward gating projection (\texttt{gate\_proj}). Through controlled module transfer, targeted single-module adaptation, and destructive ablation, we further demonstrate that \texttt{gate\_proj} is sufficient, efficient, and necessary for affective understanding and generation. Notably, by tuning only approximately 24.5\% of the parameters tuned by AffectGPT, our approach achieves 96.6\% of its average performance across eight affective tasks, highlighting substantial parameter efficiency. Together, these findings provide empirical evidence that affective capabilities in foundation models are structurally mediated by feed-forward gating mechanisms and identify \texttt{gate\_proj} as a central architectural locus of affective modeling.
>
---
#### [new 012] Skywork UniPic 3.0: Unified Multi-Image Composition via Sequence Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Skywork UniPic 3.0，解决多图像合成任务中的一致性与质量问题，通过序列建模和数据优化实现高效高质合成。**

- **链接: [https://arxiv.org/pdf/2601.15664v1](https://arxiv.org/pdf/2601.15664v1)**

> **作者:** Hongyang Wei; Hongbo Liu; Zidong Wang; Yi Peng; Baixin Xu; Size Wu; Xuying Zhang; Xianglong He; Zexiang Liu; Peiyu Wang; Xuchen Song; Yangguang Li; Yang Liu; Yahui Zhou
>
> **摘要:** The recent surge in popularity of Nano-Banana and Seedream 4.0 underscores the community's strong interest in multi-image composition tasks. Compared to single-image editing, multi-image composition presents significantly greater challenges in terms of consistency and quality, yet existing models have not disclosed specific methodological details for achieving high-quality fusion. Through statistical analysis, we identify Human-Object Interaction (HOI) as the most sought-after category by the community. We therefore systematically analyze and implement a state-of-the-art solution for multi-image composition with a primary focus on HOI-centric tasks. We present Skywork UniPic 3.0, a unified multimodal framework that integrates single-image editing and multi-image composition. Our model supports an arbitrary (1~6) number and resolution of input images, as well as arbitrary output resolutions (within a total pixel budget of 1024x1024). To address the challenges of multi-image composition, we design a comprehensive data collection, filtering, and synthesis pipeline, achieving strong performance with only 700K high-quality training samples. Furthermore, we introduce a novel training paradigm that formulates multi-image composition as a sequence-modeling problem, transforming conditional generation into unified sequence synthesis. To accelerate inference, we integrate trajectory mapping and distribution matching into the post-training stage, enabling the model to produce high-fidelity samples in just 8 steps and achieve a 12.5x speedup over standard synthesis sampling. Skywork UniPic 3.0 achieves state-of-the-art performance on single-image editing benchmark and surpasses both Nano-Banana and Seedream 4.0 on multi-image composition benchmark, thereby validating the effectiveness of our data pipeline and training paradigm. Code, models and dataset are publicly available.
>
---
#### [new 013] Clustering-Guided Spatial-Spectral Mamba for Hyperspectral Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于高光谱图像分类任务，旨在解决Mamba模型在定义高效自适应token序列上的挑战。通过引入聚类机制和注意力机制，提出CSSMamba框架以提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.16098v1](https://arxiv.org/pdf/2601.16098v1)**

> **作者:** Zack Dewis; Yimin Zhu; Zhengsen Xu; Mabel Heffring; Saeid Taleghanidoozdoozan; Quinn Ledingham; Lincoln Linlin Xu
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Although Mamba models greatly improve Hyperspectral Image (HSI) classification, they have critical challenges in terms defining efficient and adaptive token sequences for improve performance. This paper therefore presents CSSMamba (Clustering-guided Spatial-Spectral Mamba) framework to better address the challenges, with the following contributions. First, to achieve efficient and adaptive token sequences for improved Mamba performance, we integrate the clustering mechanism into a spatial Mamba architecture, leading to a cluster-guided spatial Mamba module (CSpaMamba) that reduces the Mamba sequence length and improves Mamba feature learning capability. Second, to improve the learning of both spatial and spectral information, we integrate the CSpaMamba module with a spectral mamba module (SpeMamba), leading to a complete clustering-guided spatial-spectral Mamba framework. Third, to further improve feature learning capability, we introduce an Attention-Driven Token Selection mechanism to optimize Mamba token sequencing. Last, to seamlessly integrate clustering into the Mamba model in a coherent manner, we design a Learnable Clustering Module that learns the cluster memberships in an adaptive manner. Experiments on the Pavia University, Indian Pines, and Liao-Ning 01 datasets demonstrate that CSSMamba achieves higher accuracy and better boundary preservation compared to state-of-the-art CNN, Transformer, and Mamba-based methods.
>
---
#### [new 014] SuperOcc: Toward Cohesive Temporal Modeling for Superquadric-based Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D占用预测任务，旨在解决现有方法在时间建模、几何表达和计算效率上的不足。提出SuperOcc框架，提升预测性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.15644v1](https://arxiv.org/pdf/2601.15644v1)**

> **作者:** Zichen Yu; Quanli Liu; Wei Wang; Liyong Zhang; Xiaoguang Zhao
>
> **摘要:** 3D occupancy prediction plays a pivotal role in the realm of autonomous driving, as it provides a comprehensive understanding of the driving environment. Most existing methods construct dense scene representations for occupancy prediction, overlooking the inherent sparsity of real-world driving scenes. Recently, 3D superquadric representation has emerged as a promising sparse alternative to dense scene representations due to the strong geometric expressiveness of superquadrics. However, existing superquadric frameworks still suffer from insufficient temporal modeling, a challenging trade-off between query sparsity and geometric expressiveness, and inefficient superquadric-to-voxel splatting. To address these issues, we propose SuperOcc, a novel framework for superquadric-based 3D occupancy prediction. SuperOcc incorporates three key designs: (1) a cohesive temporal modeling mechanism to simultaneously exploit view-centric and object-centric temporal cues; (2) a multi-superquadric decoding strategy to enhance geometric expressiveness without sacrificing query sparsity; and (3) an efficient superquadric-to-voxel splatting scheme to improve computational efficiency. Extensive experiments on the SurroundOcc and Occ3D benchmarks demonstrate that SuperOcc achieves state-of-the-art performance while maintaining superior efficiency. The code is available at https://github.com/Yzichen/SuperOcc.
>
---
#### [new 015] PMPBench: A Paired Multi-Modal Pan-Cancer Benchmark for Medical Image Synthesis
- **分类: cs.CV**

- **简介: 该论文提出PMPBench，一个跨器官的医学图像合成基准，解决对比剂使用受限问题，通过多模态数据实现非对比图像到对比图像的生成。**

- **链接: [https://arxiv.org/pdf/2601.15884v1](https://arxiv.org/pdf/2601.15884v1)**

> **作者:** Yifan Chen; Fei Yin; Hao Chen; Jia Wu; Chao Li
>
> **摘要:** Contrast medium plays a pivotal role in radiological imaging, as it amplifies lesion conspicuity and improves detection for the diagnosis of tumor-related diseases. However, depending on the patient's health condition or the medical resources available, the use of contrast medium is not always feasible. Recent work has explored AI-based image translation to synthesize contrast-enhanced images directly from non-contrast scans, aims to reduce side effects and streamlines clinical workflows. Progress in this direction has been constrained by data limitations: (1) existing public datasets focus almost exclusively on brain-related paired MR modalities; (2) other collections include partially paired data but suffer from missing modalities/timestamps and imperfect spatial alignment; (3) explicit labeling of CT vs. CTC or DCE phases is often absent; (4) substantial resources remain private. To bridge this gap, we introduce the first public, fully paired, pan-cancer medical imaging dataset spanning 11 human organs. The MR data include complete dynamic contrast-enhanced (DCE) sequences covering all three phases (DCE1-DCE3), while the CT data provide paired non-contrast and contrast-enhanced acquisitions (CTC). The dataset is curated for anatomical correspondence, enabling rigorous evaluation of 1-to-1, N-to-1, and N-to-N translation settings (e.g., predicting DCE phases from non-contrast inputs). Built upon this resource, we establish a comprehensive benchmark. We report results from representative baselines of contemporary image-to-image translation. We release the dataset and benchmark to catalyze research on safe, effective contrast synthesis, with direct relevance to multi-organ oncology imaging workflows. Our code and dataset are publicly available at https://github.com/YifanChen02/PMPBench.
>
---
#### [new 016] Keyframe-Based Feed-Forward Visual Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决传统方法与基础模型结合中的效率与性能问题。提出基于关键帧的前馈视觉里程计，利用强化学习自适应选择关键帧，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.16020v1](https://arxiv.org/pdf/2601.16020v1)**

> **作者:** Weichen Dai; Wenhan Su; Da Kong; Yuhang Ming; Wanzeng Kong
>
> **摘要:** The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network. However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately. This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information. Integrating traditional geometric heuristics into these methods is non-trivial, as their performance depends on high-dimensional latent representations rather than explicit geometric metrics. To bridge this gap, we propose a novel keyframe-based feed-forward VO. Instead of relying on hand-crafted rules, our approach employs reinforcement learning to derive an adaptive keyframe policy in a data-driven manner, aligning selection with the intrinsic characteristics of the underlying foundation model. We train our agent on TartanAir dataset and conduct extensive evaluations across several real-world datasets. Experimental results demonstrate that the proposed method achieves consistent and substantial improvements over state-of-the-art feed-forward VO methods.
>
---
#### [new 017] Sub-Region-Aware Modality Fusion and Adaptive Prompting for Multi-Modal Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像分割任务，旨在解决多模态信息融合与病理组织异质性适应问题。提出子区域感知注意力和自适应提示策略，提升脑肿瘤分割精度。**

- **链接: [https://arxiv.org/pdf/2601.15734v1](https://arxiv.org/pdf/2601.15734v1)**

> **作者:** Shadi Alijani; Fereshteh Aghaee Meibodi; Homayoun Najjaran
>
> **摘要:** The successful adaptation of foundation models to multi-modal medical imaging is a critical yet unresolved challenge. Existing models often struggle to effectively fuse information from multiple sources and adapt to the heterogeneous nature of pathological tissues. To address this, we introduce a novel framework for adapting foundation models to multi-modal medical imaging, featuring two key technical innovations: sub-region-aware modality attention and adaptive prompt engineering. The attention mechanism enables the model to learn the optimal combination of modalities for each tumor sub-region, while the adaptive prompting strategy leverages the inherent capabilities of foundation models to refine segmentation accuracy. We validate our framework on the BraTS 2020 brain tumor segmentation dataset, demonstrating that our approach significantly outperforms baseline methods, particularly in the challenging necrotic core sub-region. Our work provides a principled and effective approach to multi-modal fusion and prompting, paving the way for more accurate and robust foundation model-based solutions in medical imaging.
>
---
#### [new 018] A Lightweight Brain-Inspired Machine Learning Framework for Coronary Angiography: Hybrid Neural Representation and Robust Learning Strategies
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于冠脉造影图像分类任务，旨在解决复杂病变、类别不平衡和计算资源有限的问题。提出轻量级脑启发框架，结合混合神经表示和鲁棒学习策略，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.15865v1](https://arxiv.org/pdf/2601.15865v1)**

> **作者:** Jingsong Xia; Siqi Wang
>
> **摘要:** Background: Coronary angiography (CAG) is a cornerstone imaging modality for assessing coronary artery disease and guiding interventional treatment decisions. However, in real-world clinical settings, angiographic images are often characterized by complex lesion morphology, severe class imbalance, label uncertainty, and limited computational resources, posing substantial challenges to conventional deep learning approaches in terms of robustness and generalization.Methods: The proposed framework is built upon a pretrained convolutional neural network to construct a lightweight hybrid neural representation. A selective neural plasticity training strategy is introduced to enable efficient parameter adaptation. Furthermore, a brain-inspired attention-modulated loss function, combining Focal Loss with label smoothing, is employed to enhance sensitivity to hard samples and uncertain annotations. Class-imbalance-aware sampling and cosine annealing with warm restarts are adopted to mimic rhythmic regulation and attention allocation mechanisms observed in biological neural systems.Results: Experimental results demonstrate that the proposed lightweight brain-inspired model achieves strong and stable performance in binary coronary angiography classification, yielding competitive accuracy, recall, F1-score, and AUC metrics while maintaining high computational efficiency.Conclusion: This study validates the effectiveness of brain-inspired learning mechanisms in lightweight medical image analysis and provides a biologically plausible and deployable solution for intelligent clinical decision support under limited computational resources.
>
---
#### [new 019] Relative Classification Accuracy: A Calibrated Metric for Identity Consistency in Fine-Grained K-pop Face Generation
- **分类: cs.CV**

- **简介: 该论文研究K-pop偶像人脸生成任务，解决细粒度身份一致性评估问题。提出RCA指标，用于衡量生成模型的语义可控性，发现高视觉质量下存在语义模式崩溃现象。**

- **链接: [https://arxiv.org/pdf/2601.15560v1](https://arxiv.org/pdf/2601.15560v1)**

> **作者:** Sylvey Lin; Eranki Vasistha
>
> **摘要:** Denoising Diffusion Probabilistic Models (DDPMs) have achieved remarkable success in high-fidelity image generation. However, evaluating their semantic controllability-specifically for fine-grained, single-domain tasks-remains challenging. Standard metrics like FID and Inception Score (IS) often fail to detect identity misalignment in such specialized contexts. In this work, we investigate Class-Conditional DDPMs for K-pop idol face generation (32x32), a domain characterized by high inter-class similarity. We propose a calibrated metric, Relative Classification Accuracy (RCA), which normalizes generative performance against an oracle classifier's baseline. Our evaluation reveals a critical trade-off: while the model achieves high visual quality (FID 8.93), it suffers from severe semantic mode collapse (RCA 0.27), particularly for visually ambiguous identities. We analyze these failure modes through confusion matrices and attribute them to resolution constraints and intra-gender ambiguity. Our framework provides a rigorous standard for verifying identity consistency in conditional generative models.
>
---
#### [new 020] RadJEPA: Radiology Encoder for Chest X-Rays via Joint Embedding Predictive Architecture
- **分类: cs.CV**

- **简介: 该论文提出RadJEPA，一种无需语言监督的放射学图像编码框架。针对医学图像表示学习中缺乏配对图文数据的问题，通过自监督方式预训练，提升疾病分类、分割等任务性能。**

- **链接: [https://arxiv.org/pdf/2601.15891v1](https://arxiv.org/pdf/2601.15891v1)**

> **作者:** Anas Anwarul Haq Khan; Mariam Husain; Kshitij Jadhav
>
> **摘要:** Recent advances in medical vision language models guide the learning of visual representations; however, this form of supervision is constrained by the availability of paired image text data, raising the question of whether robust radiology encoders can be learned without relying on language supervision. In this work, we introduce RadJEPA, a self-supervised framework built on a Joint Embedding Predictive Architecture that learns without language supervision. Pre-trained solely on unlabeled chest X-ray images, the model learns to predict latent representations of masked image regions. This predictive objective differs fundamentally from both image text pre-training and DINO-style self-distillation: rather than aligning global representations across views or modalities, RadJEPA explicitly models latent-space prediction. We evaluate the learned encoder on disease classification, semantic segmentation, and report generation tasks. Across benchmarks, RadJEPA achieves performance exceeding state-of-the-art approaches, including Rad-DINO.
>
---
#### [new 021] An IoT-Based Smart Plant Monitoring and Irrigation System with Real-Time Environmental Sensing, Automated Alerts, and Cloud Analytics
- **分类: cs.CV**

- **简介: 该论文属于智能农业任务，旨在解决传统灌溉效率低、资源浪费的问题。通过物联网技术实现环境实时监测与自动灌溉，提升作物生长效率。**

- **链接: [https://arxiv.org/pdf/2601.15830v1](https://arxiv.org/pdf/2601.15830v1)**

> **作者:** Abdul Hasib; A. S. M. Ahsanul Sarkar Akib
>
> **摘要:** The increasing global demand for sustainable agriculture necessitates intelligent monitoring systems that optimize resource utilization and plant health management. Traditional farming methods rely on manual observation and periodic watering, often leading to water wastage, inconsistent plant growth, and delayed response to environmental changes. This paper presents a comprehensive IoT-based smart plant monitoring system that integrates multiple environmental sensors with automated irrigation and cloud analytics. The proposed system utilizes an ESP32 microcontroller to collect real-time data from DHT22 (temperature/humidity), HC-SR04 (water level), and soil moisture sensors, with visual feedback through an OLED display and auditory alerts via a buzzer. All sensor data is wirelessly transmitted to the ThingSpeak cloud platform for remote monitoring, historical analysis, and automated alert generation. Experimental results demonstrate the system's effectiveness in maintaining optimal soil moisture levels (with 92\% accuracy), providing real-time environmental monitoring, and reducing water consumption by approximately 40\% compared to conventional irrigation methods. The integrated web dashboard offers comprehensive visualization of plant health parameters, making it suitable for both small-scale gardening and commercial agriculture applications. With a total implementation cost of \$45.20, this system provides an affordable, scalable solution for precision agriculture and smart farming.
>
---
#### [new 022] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医学视觉-语言任务，旨在解决放射科报告生成中的视觉定位不准确和事实不一致问题。工作包括提出CURE框架，通过多任务训练提升报告质量和定位准确性。**

- **链接: [https://arxiv.org/pdf/2601.15408v1](https://arxiv.org/pdf/2601.15408v1)**

> **作者:** Pablo Messina; Andrés Villa; Juan León Alcázar; Karen Sánchez; Carlos Hinojosa; Denis Parra; Álvaro Soto; Bernard Ghanem
>
> **备注:** 31 pages, 7 figures, submitted to CVPR 2026 (under review)
>
> **摘要:** Medical vision-language models can automate the generation of radiology reports but struggle with accurate visual grounding and factual consistency. Existing models often misalign textual findings with visual evidence, leading to unreliable or weakly grounded predictions. We present CURE, an error-aware curriculum learning framework that improves grounding and report quality without any additional data. CURE fine-tunes a multimodal instructional model on phrase grounding, grounded report generation, and anatomy-grounded report generation using public datasets. The method dynamically adjusts sampling based on model performance, emphasizing harder samples to improve spatial and textual alignment. CURE improves grounding accuracy by +0.37 IoU, boosts report quality by +0.188 CXRFEScore, and reduces hallucinations by 18.6%. CURE is a data-efficient framework that enhances both grounding accuracy and report reliability. Code is available at https://github.com/PabloMessina/CURE and model weights at https://huggingface.co/pamessina/medgemma-4b-it-cure
>
---
#### [new 023] Region-aware Spatiotemporal Modeling with Collaborative Domain Generalization for Cross-Subject EEG Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于跨被试EEG情绪识别任务，旨在解决个体差异大、分布变化显著的问题。提出RSM-CoDG框架，结合区域感知时空建模和协同域泛化，提升模型泛化能力与跨被试性能。**

- **链接: [https://arxiv.org/pdf/2601.15615v1](https://arxiv.org/pdf/2601.15615v1)**

> **作者:** Weiwei Wu; Yueyang Li; Yuhu Shi; Weiming Zeng; Lang Qin; Yang Yang; Ke Zhou; Zhiguo Zhang; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Cross-subject EEG-based emotion recognition (EER) remains challenging due to strong inter-subject variability, which induces substantial distribution shifts in EEG signals, as well as the high complexity of emotion-related neural representations in both spatial organization and temporal evolution. Existing approaches typically improve spatial modeling, temporal modeling, or generalization strategies in isolation, which limits their ability to align representations across subjects while capturing multi-scale dynamics and suppressing subject-specific bias within a unified framework. To address these gaps, we propose a Region-aware Spatiotemporal Modeling framework with Collaborative Domain Generalization (RSM-CoDG) for cross-subject EEG emotion recognition. RSM-CoDG incorporates neuroscience priors derived from functional brain region partitioning to construct region-level spatial representations, thereby improving cross-subject comparability. It also employs multi-scale temporal modeling to characterize the dynamic evolution of emotion-evoked neural activity. In addition, the framework employs a collaborative domain generalization strategy, incorporating multidimensional constraints to reduce subject-specific bias in a fully unseen target subject setting, which enhances the generalization to unknown individuals. Extensive experimental results on SEED series datasets demonstrate that RSM-CoDG consistently outperforms existing competing methods, providing an effective approach for improving robustness. The source code is available at https://github.com/RyanLi-X/RSM-CoDG.
>
---
#### [new 024] Explainable Deepfake Detection with RL Enhanced Self-Blended Images
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决现有方法缺乏可解释性及标注数据不足的问题。提出基于自混合图像的自动数据生成框架与强化学习增强的检测方法。**

- **链接: [https://arxiv.org/pdf/2601.15624v1](https://arxiv.org/pdf/2601.15624v1)**

> **作者:** Ning Jiang; Dingheng Zeng; Yanhong Liu; Haiyang Yi; Shijie Yu; Minghe Weng; Haifeng Shen; Ying Li
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Most prior deepfake detection methods lack explainable outputs. With the growing interest in multimodal large language models (MLLMs), researchers have started exploring their use in interpretable deepfake detection. However, a major obstacle in applying MLLMs to this task is the scarcity of high-quality datasets with detailed forgery attribution annotations, as textual annotation is both costly and challenging - particularly for high-fidelity forged images or videos. Moreover, multiple studies have shown that reinforcement learning (RL) can substantially enhance performance in visual tasks, especially in improving cross-domain generalization. To facilitate the adoption of mainstream MLLM frameworks in deepfake detection with reduced annotation cost, and to investigate the potential of RL in this context, we propose an automated Chain-of-Thought (CoT) data generation framework based on Self-Blended Images, along with an RL-enhanced deepfake detection framework. Extensive experiments validate the effectiveness of our CoT data construction pipeline, tailored reward mechanism, and feedback-driven synthetic data generation approach. Our method achieves performance competitive with state-of-the-art (SOTA) approaches across multiple cross-dataset benchmarks. Implementation details are available at https://github.com/deon1219/rlsbi.
>
---
#### [new 025] Understanding the Transfer Limits of Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉基础模型在下游任务中的迁移限制，旨在解决预训练目标与任务需求不匹配的问题。通过分析两种模型在前列腺MRI任务中的表现，探讨任务对齐对迁移效果的影响。**

- **链接: [https://arxiv.org/pdf/2601.15888v1](https://arxiv.org/pdf/2601.15888v1)**

> **作者:** Shiqi Huang; Yipei Wang; Natasha Thorley; Alexander Ng; Shaheer Saeed; Mark Emberton; Shonit Punwani; Veeru Kasivisvanathan; Dean Barratt; Daniel Alexander; Yipeng Hu
>
> **备注:** accepted in ISBI 2026
>
> **摘要:** Foundation models leverage large-scale pretraining to capture extensive knowledge, demonstrating generalization in a wide range of language tasks. By comparison, vision foundation models (VFMs) often exhibit uneven improvements across downstream tasks, despite substantial computational investment. We postulate that this limitation arises from a mismatch between pretraining objectives and the demands of downstream vision-and-imaging tasks. Pretraining strategies like masked image reconstruction or contrastive learning shape representations for tasks such as recovery of generic visual patterns or global semantic structures, which may not align with the task-specific requirements of downstream applications including segmentation, classification, or image synthesis. To investigate this in a concrete real-world clinical area, we assess two VFMs, a reconstruction-focused MAE-based model (ProFound) and a contrastive-learning-based model (ProViCNet), on five prostate multiparametric MR imaging tasks, examining how such task alignment influences transfer performance, i.e., from pretraining to fine-tuning. Our findings indicate that better alignment between pretraining and downstream tasks, measured by simple divergence metrics such as maximum-mean-discrepancy (MMD) between the same features before and after fine-tuning, correlates with greater performance improvements and faster convergence, emphasizing the importance of designing and analyzing pretraining objectives with downstream applicability in mind.
>
---
#### [new 026] EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis
- **分类: cs.CV**

- **简介: 该论文提出EvolSplat4D，解决4D城市场景合成中的重建质量与效率问题，通过多分支统一高斯预测实现高效且一致的静态与动态场景重建。**

- **链接: [https://arxiv.org/pdf/2601.15951v1](https://arxiv.org/pdf/2601.15951v1)**

> **作者:** Sheng Miao; Sijin Li; Pan Wang; Dongfeng Bai; Bingbing Liu; Yue Wang; Andreas Geiger; Yiyi Liao
>
> **摘要:** Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality. While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization. Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments. We propose EvolSplat4D, a feed-forward framework that moves beyond existing per-pixel paradigms by unifying volume-based and pixel-based Gaussian prediction across three specialized branches. For close-range static regions, we predict consistent geometry of 3D Gaussians over multiple frames directly from a 3D feature volume, complemented by a semantically-enhanced image-based rendering module for predicting their appearance. For dynamic actors, we utilize object-centric canonical spaces and a motion-adjusted rendering module to aggregate temporal features, ensuring stable 4D reconstruction despite noisy motion priors. Far-Field scenery is handled by an efficient per-pixel Gaussian branch to ensure full-scene coverage. Experimental results on the KITTI-360, KITTI, Waymo, and PandaSet datasets show that EvolSplat4D reconstructs both static and dynamic environments with superior accuracy and consistency, outperforming both per-scene optimization and state-of-the-art feed-forward baselines.
>
---
#### [new 027] Evolving Without Ending: Unifying Multimodal Incremental Learning for Continual Panoptic Perception
- **分类: cs.CV**

- **简介: 该论文属于持续多模态学习任务，旨在解决多任务下语义混淆和灾难性遗忘问题。提出CPP模型，融合多模态与多任务学习，提升图像感知能力。**

- **链接: [https://arxiv.org/pdf/2601.15643v1](https://arxiv.org/pdf/2601.15643v1)**

> **作者:** Bo Yuan; Danpei Zhao; Wentao Li; Tian Li; Zhiguo Jiang
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2407.14242
>
> **摘要:** Continual learning (CL) is a great endeavour in developing intelligent perception AI systems. However, the pioneer research has predominantly focus on single-task CL, which restricts the potential in multi-task and multimodal scenarios. Beyond the well-known issue of catastrophic forgetting, the multi-task CL also brings semantic obfuscation across multimodal alignment, leading to severe model degradation during incremental training steps. In this paper, we extend CL to continual panoptic perception (CPP), integrating multimodal and multi-task CL to enhance comprehensive image perception through pixel-level, instance-level, and image-level joint interpretation. We formalize the CL task in multimodal scenarios and propose an end-to-end continual panoptic perception model. Concretely, CPP model features a collaborative cross-modal encoder (CCE) for multimodal embedding. We also propose a malleable knowledge inheritance module via contrastive feature distillation and instance distillation, addressing catastrophic forgetting from task-interactive boosting manner. Furthermore, we propose a cross-modal consistency constraint and develop CPP+, ensuring multimodal semantic alignment for model updating under multi-task incremental scenarios. Additionally, our proposed model incorporates an asymmetric pseudo-labeling manner, enabling model evolving without exemplar replay. Extensive experiments on multimodal datasets and diverse CL tasks demonstrate the superiority of the proposed model, particularly in fine-grained CL tasks.
>
---
#### [new 028] HVD: Human Vision-Driven Video Representation Learning for Text-Video Retrieval
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于文本-视频检索任务，旨在解决模型在稀疏文本查询下难以区分关键视觉信息的问题。提出HVD模型，通过粗到细的对齐机制提升检索效果。**

- **链接: [https://arxiv.org/pdf/2601.16155v1](https://arxiv.org/pdf/2601.16155v1)**

> **作者:** Zequn Xie; Xin Liu; Boyun Zhang; Yuxiao Lin; Sihang Cai; Tao Jin
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** The success of CLIP has driven substantial progress in text-video retrieval. However, current methods often suffer from "blind" feature interaction, where the model struggles to discern key visual information from background noise due to the sparsity of textual queries. To bridge this gap, we draw inspiration from human cognitive behavior and propose the Human Vision-Driven (HVD) model. Our framework establishes a coarse-to-fine alignment mechanism comprising two key components: the Frame Features Selection Module (FFSM) and the Patch Features Compression Module (PFCM). FFSM mimics the human macro-perception ability by selecting key frames to eliminate temporal redundancy. Subsequently, PFCM simulates micro-perception by aggregating patch features into salient visual entities through an advanced attention mechanism, enabling precise entity-level matching. Extensive experiments on five benchmarks demonstrate that HVD not only captures human-like visual focus but also achieves state-of-the-art performance.
>
---
#### [new 029] Consistency-Regularized GAN for Few-Shot SAR Target Recognition
- **分类: cs.CV**

- **简介: 该论文属于少样本SAR目标识别任务，解决数据稀缺导致的识别难题。提出Cr-GAN框架，通过一致性正则化生成高质量样本，提升自监督学习效果。**

- **链接: [https://arxiv.org/pdf/2601.15681v1](https://arxiv.org/pdf/2601.15681v1)**

> **作者:** Yikui Zhai; Shikuang Liu; Wenlve Zhou; Hongsheng Zhang; Zhiheng Zhou; Xiaolin Tian; C. L. Philip Chen
>
> **摘要:** Few-shot recognition in synthetic aperture radar (SAR) imagery remains a critical bottleneck for real-world applications due to extreme data scarcity. A promising strategy involves synthesizing a large dataset with a generative adversarial network (GAN), pre-training a model via self-supervised learning (SSL), and then fine-tuning on the few labeled samples. However, this approach faces a fundamental paradox: conventional GANs themselves require abundant data for stable training, contradicting the premise of few-shot learning. To resolve this, we propose the consistency-regularized generative adversarial network (Cr-GAN), a novel framework designed to synthesize diverse, high-fidelity samples even when trained under these severe data limitations. Cr-GAN introduces a dual-branch discriminator that decouples adversarial training from representation learning. This architecture enables a channel-wise feature interpolation strategy to create novel latent features, complemented by a dual-domain cycle consistency mechanism that ensures semantic integrity. Our Cr-GAN framework is adaptable to various GAN architectures, and its synthesized data effectively boosts multiple SSL algorithms. Extensive experiments on the MSTAR and SRSDD datasets validate our approach, with Cr-GAN achieving a highly competitive accuracy of 71.21% and 51.64%, respectively, in the 8-shot setting, significantly outperforming leading baselines, while requiring only ~5 of the parameters of state-of-the-art diffusion models. Code is available at: https://github.com/yikuizhai/Cr-GAN.
>
---
#### [new 030] Towards Realistic Remote Sensing Dataset Distillation with Discriminative Prototype-guided Diffusion
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决大数据带来的存储与数据泄露问题。通过训练扩散模型，将大规模数据浓缩为高质量小样本集，提升模型训练效率与安全性。**

- **链接: [https://arxiv.org/pdf/2601.15829v1](https://arxiv.org/pdf/2601.15829v1)**

> **作者:** Yonghao Xu; Pedram Ghamisi; Qihao Weng
>
> **摘要:** Recent years have witnessed the remarkable success of deep learning in remote sensing image interpretation, driven by the availability of large-scale benchmark datasets. However, this reliance on massive training data also brings two major challenges: (1) high storage and computational costs, and (2) the risk of data leakage, especially when sensitive categories are involved. To address these challenges, this study introduces the concept of dataset distillation into the field of remote sensing image interpretation for the first time. Specifically, we train a text-to-image diffusion model to condense a large-scale remote sensing dataset into a compact and representative distilled dataset. To improve the discriminative quality of the synthesized samples, we propose a classifier-driven guidance by injecting a classification consistency loss from a pre-trained model into the diffusion training process. Besides, considering the rich semantic complexity of remote sensing imagery, we further perform latent space clustering on training samples to select representative and diverse prototypes as visual style guidance, while using a visual language model to provide aggregated text descriptions. Experiments on three high-resolution remote sensing scene classification benchmarks show that the proposed method can distill realistic and diverse samples for downstream model training. Code and pre-trained models are available online (https://github.com/YonghaoXu/DPD).
>
---
#### [new 031] LL-GaussianImage: Efficient Image Representation for Zero-shot Low-Light Enhancement with 2D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，解决2DGS压缩图像处理效率低和二次降质问题，提出LL-GaussianImage框架实现直接压缩域增强。**

- **链接: [https://arxiv.org/pdf/2601.15772v1](https://arxiv.org/pdf/2601.15772v1)**

> **作者:** Yuhan Chen; Wenxuan Yu; Guofa Li; Yijun Xu; Ying Fang; Yicui Shi; Long Cao; Wenbo Chu; Keqiang Li
>
> **摘要:** 2D Gaussian Splatting (2DGS) is an emerging explicit scene representation method with significant potential for image compression due to high fidelity and high compression ratios. However, existing low-light enhancement algorithms operate predominantly within the pixel domain. Processing 2DGS-compressed images necessitates a cumbersome decompression-enhancement-recompression pipeline, which compromises efficiency and introduces secondary degradation. To address these limitations, we propose LL-GaussianImage, the first zero-shot unsupervised framework designed for low-light enhancement directly within the 2DGS compressed representation domain. Three primary advantages are offered by this framework. First, a semantic-guided Mixture-of-Experts enhancement framework is designed. Dynamic adaptive transformations are applied to the sparse attribute space of 2DGS using rendered images as guidance to enable compression-as-enhancement without full decompression to a pixel grid. Second, a multi-objective collaborative loss function system is established to strictly constrain smoothness and fidelity during enhancement, suppressing artifacts while improving visual quality. Third, a two-stage optimization process is utilized to achieve reconstruction-as-enhancement. The accuracy of the base representation is ensured through single-scale reconstruction and network robustness is enhanced. High-quality enhancement of low-light images is achieved while high compression ratios are maintained. The feasibility and superiority of the paradigm for direct processing within the compressed representation domain are validated through experimental results.
>
---
#### [new 032] Hybrid Vision Transformer_GAN Attribute Neutralizer for Mitigating Bias in Chest X_Ray Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像AI公平性任务，旨在解决胸部X光诊断中的性别和年龄偏见问题。通过替换传统编码器为Vision Transformer，有效降低属性泄露，同时保持诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.15490v1](https://arxiv.org/pdf/2601.15490v1)**

> **作者:** Jobeal Solomon; Ali Mohammed Mansoor Alsahag; Seyed Sahand Mohammadi Ziabari
>
> **摘要:** Bias in chest X-ray classifiers frequently stems from sex- and age-related shortcuts, leading to systematic underdiagnosis of minority subgroups. Previous pixel-space attribute neutralizers, which rely on convolutional encoders, lessen but do not fully remove this attribute leakage at clinically usable edit strengths. This study evaluates whether substituting the U-Net convolutional encoder with a Vision Transformer backbone in the Attribute-Neutral Framework can reduce demographic attribute leakage while preserving diagnostic accuracy. A data-efficient Image Transformer Small (DeiT-S) neutralizer was trained on the ChestX-ray14 dataset. Its edited images, generated across eleven edit-intensity levels, were evaluated with an independent AI judge for attribute leakage and with a convolutional neural network (ConvNet) for disease prediction. At a moderate edit level (alpha = 0.5), the Vision Transformer (ViT) neutralizer reduces patient sex-recognition area under the curve (AUC) to approximately 0.80, about 10 percentage points below the original framework's convolutional U-Net encoder, despite being trained for only half as many epochs. Meanwhile, macro receiver operating characteristic area under the curve (ROC AUC) across 15 findings stays within five percentage points of the unedited baseline, and the worst-case subgroup AUC remains near 0.70. These results indicate that global self-attention vision models can further suppress attribute leakage without sacrificing clinical utility, suggesting a practical route toward fairer chest X-ray AI.
>
---
#### [new 033] Event-VStream: Event-Driven Real-Time Understanding for Long Video Streams
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Event-VStream，解决长视频流实时理解问题，通过事件驱动机制减少冗余处理，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.15655v1](https://arxiv.org/pdf/2601.15655v1)**

> **作者:** Zhenghui Guo; Yuanbin Man; Junyuan Sheng; Bowen Lin; Ahmed Ahmed; Bo Jiang; Boyuan Zhang; Miao Yin; Sian Jin; Omprakash Gnawal; Chengming Zhang
>
> **摘要:** Real-time understanding of long video streams remains challenging for multimodal large language models (VLMs) due to redundant frame processing and rapid forgetting of past context. Existing streaming systems rely on fixed-interval decoding or cache pruning, which either produce repetitive outputs or discard crucial temporal information. We introduce Event-VStream, an event-aware framework that represents continuous video as a sequence of discrete, semantically coherent events. Our system detects meaningful state transitions by integrating motion, semantic, and predictive cues, and triggers language generation only at those boundaries. Each event embedding is consolidated into a persistent memory bank, enabling long-horizon reasoning while maintaining low latency. Across OVOBench-Realtime, and long-form Ego4D evaluations, Event-VStream achieves competitive performance. It improves over a VideoLLM-Online-8B baseline by +10.4 points on OVOBench-Realtime, achieves performance close to Flash-VStream-7B despite using only a general-purpose LLaMA-3-8B text backbone, and maintains around 70% GPT-5 win rate on 2-hour Ego4D streams.
>
---
#### [new 034] PhysicsMind: Sim and Real Mechanics Benchmarking for Physical Reasoning and Prediction in Foundational VLMs and World Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PhysicsMind基准，用于评估模型在物理推理和预测上的表现，解决现有基准不足的问题。任务包括VQA和视频生成，检验模型是否遵循物理定律。**

- **链接: [https://arxiv.org/pdf/2601.16007v1](https://arxiv.org/pdf/2601.16007v1)**

> **作者:** Chak-Wing Mak; Guanyu Zhu; Boyi Zhang; Hongji Li; Xiaowei Chi; Kevin Zhang; Yichen Wu; Yangfan He; Chun-Kai Fan; Wentao Lu; Kuangzhi Ge; Xinyu Fang; Hongyang He; Kuan Lu; Tianxiang Xu; Li Zhang; Yongxin Ni; Youhua Li; Shanghang Zhang
>
> **摘要:** Modern foundational Multimodal Large Language Models (MLLMs) and video world models have advanced significantly in mathematical, common-sense, and visual reasoning, but their grasp of the underlying physics remains underexplored. Existing benchmarks attempting to measure this matter rely on synthetic, Visual Question Answer templates or focus on perceptual video quality that is tangential to measuring how well the video abides by physical laws. To address this fragmentation, we introduce PhysicsMind, a unified benchmark with both real and simulation environments that evaluates law-consistent reasoning and generation over three canonical principles: Center of Mass, Lever Equilibrium, and Newton's First Law. PhysicsMind comprises two main tasks: i) VQA tasks, testing whether models can reason and determine physical quantities and values from images or short videos, and ii) Video Generation(VG) tasks, evaluating if predicted motion trajectories obey the same center-of-mass, torque, and inertial constraints as the ground truth. A broad range of recent models and video generation models is evaluated on PhysicsMind and found to rely on appearance heuristics while often violating basic mechanics. These gaps indicate that current scaling and training are still insufficient for robust physical understanding, underscoring PhysicsMind as a focused testbed for physics-aware multimodal models. Our data will be released upon acceptance.
>
---
#### [new 035] Beyond Visual Safety: Jailbreaking Multimodal Large Language Models for Harmful Image Generation via Semantic-Agnostic Inputs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全漏洞研究任务，旨在解决MLLMs的视觉安全问题。通过提出BVS框架，成功实现对模型的图像生成攻击，揭示其安全缺陷。**

- **链接: [https://arxiv.org/pdf/2601.15698v1](https://arxiv.org/pdf/2601.15698v1)**

> **作者:** Mingyu Yu; Lana Liu; Zhehao Zhao; Wei Wang; Sujuan Qin
>
> **摘要:** The rapid advancement of Multimodal Large Language Models (MLLMs) has introduced complex security challenges, particularly at the intersection of textual and visual safety. While existing schemes have explored the security vulnerabilities of MLLMs, the investigation into their visual safety boundaries remains insufficient. In this paper, we propose Beyond Visual Safety (BVS), a novel image-text pair jailbreaking framework specifically designed to probe the visual safety boundaries of MLLMs. BVS employs a "reconstruction-then-generation" strategy, leveraging neutralized visual splicing and inductive recomposition to decouple malicious intent from raw inputs, thereby leading MLLMs to be induced into generating harmful images. Experimental results demonstrate that BVS achieves a remarkable jailbreak success rate of 98.21\% against GPT-5 (12 January 2026 release). Our findings expose critical vulnerabilities in the visual safety alignment of current MLLMs.
>
---
#### [new 036] The Latency Wall: Benchmarking Off-the-Shelf Emotion Recognition for Real-Time Virtual Avatars
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于实时情感识别任务，旨在解决VR中情感识别的延迟与准确率矛盾问题。通过测试不同模型，发现通用模型无法满足实时性要求，需开发轻量专用架构。**

- **链接: [https://arxiv.org/pdf/2601.15914v1](https://arxiv.org/pdf/2601.15914v1)**

> **作者:** Yarin Benyamin
>
> **备注:** Technical Report benchmarking off-the-shelf CV latencies on commodity CPU hardware for therapeutic VR applications
>
> **摘要:** In the realm of Virtual Reality (VR) and Human-Computer Interaction (HCI), real-time emotion recognition shows promise for supporting individuals with Autism Spectrum Disorder (ASD) in improving social skills. This task requires a strict latency-accuracy trade-off, with motion-to-photon (MTP) latency kept below 140 ms to maintain contingency. However, most off-the-shelf Deep Learning models prioritize accuracy over the strict timing constraints of commodity hardware. As a first step toward accessible VR therapy, we benchmark State-of-the-Art (SOTA) models for Zero-Shot Facial Expression Recognition (FER) on virtual characters using the UIBVFED dataset. We evaluate Medium and Nano variants of YOLO (v8, v11, and v12) for face detection, alongside general-purpose Vision Transformers including CLIP, SigLIP, and ViT-FER.Our results on CPU-only inference demonstrate that while face detection on stylized avatars is robust (100% accuracy), a "Latency Wall" exists in the classification stage. The YOLOv11n architecture offers the optimal balance for detection (~54 ms). However, general-purpose Transformers like CLIP and SigLIP fail to achieve viable accuracy (<23%) or speed (>150 ms) for real-time loops. This study highlights the necessity for lightweight, domain-specific architectures to enable accessible, real-time AI in therapeutic settings.
>
---
#### [new 037] ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling
- **分类: cs.CV**

- **简介: 论文提出ThermoSplat，解决多模态3D重建中跨模态信息融合不足的问题，通过特征调制和几何解耦提升RGB与热红外数据的重建质量。**

- **链接: [https://arxiv.org/pdf/2601.15897v1](https://arxiv.org/pdf/2601.15897v1)**

> **作者:** Zhaoqi Su; Shihai Chen; Xinyan Lin; Liqin Huang; Zhipeng Su; Xiaoqiang Lu
>
> **摘要:** Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Cross-Modal FiLM Modulation mechanism that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable cross-modal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the thermal branch. Additionally, a hybrid rendering pipeline is employed to integrate explicit Spherical Harmonics with implicit neural decoding, ensuring both semantic consistency and high-frequency detail preservation. Extensive experiments on the RGBT-Scenes dataset demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums.
>
---
#### [new 038] Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊与HDR重建任务，旨在解决低动态范围模糊图像中恢复高动态范围三维场景的问题。通过结合事件数据和NeRF框架，提升三维重建的清晰度与动态范围。**

- **链接: [https://arxiv.org/pdf/2601.15475v1](https://arxiv.org/pdf/2601.15475v1)**

> **作者:** Yunshan Qi; Lin Zhu; Nan Bao; Yifan Zhao; Jia Li
>
> **摘要:** Novel view synthesis from low dynamic range (LDR) blurry images, which are common in the wild, struggles to recover high dynamic range (HDR) and sharp 3D representations in extreme lighting conditions. Although existing methods employ event data to address this issue, they ignore the sensor-physics mismatches between the camera output and physical world radiance, resulting in suboptimal HDR and deblurring results. To cope with this problem, we propose a unified sensor-physics grounded NeRF framework for sharp HDR novel view synthesis from single-exposure blurry LDR images and corresponding events. We employ NeRF to directly represent the actual radiance of the 3D scene in the HDR domain and model raw HDR scene rays hitting the sensor pixels as in the physical world. A pixel-wise RGB mapping field is introduced to align the above rendered pixel values with the sensor-recorded LDR pixel values of the input images. A novel event mapping field is also designed to bridge the physical scene dynamics and actual event sensor output. The two mapping fields are jointly optimized with the NeRF network, leveraging the spatial and temporal dynamic information in events to enhance the sharp HDR 3D representation learning. Experiments on the collected and public datasets demonstrate that our method can achieve state-of-the-art deblurring HDR novel view synthesis results with single-exposure blurry LDR images and corresponding events.
>
---
#### [new 039] Evaluating Multimodal Large Language Models for Heterogeneous Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于异构人脸识别任务，旨在评估多模态大语言模型在跨模态人脸识别中的表现，揭示其与传统系统的性能差距。**

- **链接: [https://arxiv.org/pdf/2601.15406v1](https://arxiv.org/pdf/2601.15406v1)**

> **作者:** Hatef Otroshi Shahreza; Anjith George; Sébastien Marcel
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated strong performance on a wide range of vision-language tasks, raising interest in their potential use for biometric applications. In this paper, we conduct a systematic evaluation of state-of-the-art MLLMs for heterogeneous face recognition (HFR), where enrollment and probe images are from different sensing modalities, including visual (VIS), near infrared (NIR), short-wave infrared (SWIR), and thermal camera. We benchmark multiple open-source MLLMs across several cross-modality scenarios, including VIS-NIR, VIS-SWIR, and VIS-THERMAL face recognition. The recognition performance of MLLMs is evaluated using biometric protocols and based on different metrics, including Acquire Rate, Equal Error Rate (EER), and True Accept Rate (TAR). Our results reveal substantial performance gaps between MLLMs and classical face recognition systems, particularly under challenging cross-spectral conditions, in spite of recent advances in MLLMs. Our findings highlight the limitations of current MLLMs for HFR and also the importance of rigorous biometric evaluation when considering their deployment in face recognition systems.
>
---
#### [new 040] CamPilot: Improving Camera Control in Video Diffusion Model with Efficient Camera Reward Feedback
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在提升视频扩散模型中的相机控制能力。针对现有方法在视频-相机对齐、计算效率和3D信息利用上的不足，提出一种高效的3D解码器和奖励机制。**

- **链接: [https://arxiv.org/pdf/2601.16214v1](https://arxiv.org/pdf/2601.16214v1)**

> **作者:** Wenhang Ge; Guibao Shen; Jiawei Feng; Luozhou Wang; Hao Lu; Xingye Tian; Xin Tao; Ying-Cong Chen
>
> **摘要:** Recent advances in camera-controlled video diffusion models have significantly improved video-camera alignment. However, the camera controllability still remains limited. In this work, we build upon Reward Feedback Learning and aim to further improve camera controllability. However, directly borrowing existing ReFL approaches faces several challenges. First, current reward models lack the capacity to assess video-camera alignment. Second, decoding latent into RGB videos for reward computation introduces substantial computational overhead. Third, 3D geometric information is typically neglected during video decoding. To address these limitations, we introduce an efficient camera-aware 3D decoder that decodes video latent into 3D representations for reward quantization. Specifically, video latent along with the camera pose are decoded into 3D Gaussians. In this process, the camera pose not only acts as input, but also serves as a projection parameter. Misalignment between the video latent and camera pose will cause geometric distortions in the 3D structure, resulting in blurry renderings. Based on this property, we explicitly optimize pixel-level consistency between the rendered novel views and ground-truth ones as reward. To accommodate the stochastic nature, we further introduce a visibility term that selectively supervises only deterministic regions derived via geometric warping. Extensive experiments conducted on RealEstate10K and WorldScore benchmarks demonstrate the effectiveness of our proposed method. Project page: \href{https://a-bigbao.github.io/CamPilot/}{CamPilot Page}.
>
---
#### [new 041] Controllable Layered Image Generation for Real-World Editing
- **分类: cs.CV**

- **简介: 该论文提出LASAGNA框架，解决图像分层编辑中可控性与一致性不足的问题，通过联合生成图像及其分层结构，实现高质量、真实感的图像编辑。**

- **链接: [https://arxiv.org/pdf/2601.15507v1](https://arxiv.org/pdf/2601.15507v1)**

> **作者:** Jinrui Yang; Qing Liu; Yijun Li; Mengwei Ren; Letian Zhang; Zhe Lin; Cihang Xie; Yuyin Zhou
>
> **摘要:** Recent image generation models have shown impressive progress, yet they often struggle to yield controllable and consistent results when users attempt to edit specific elements within an existing image. Layered representations enable flexible, user-driven content creation, but existing approaches often fail to produce layers with coherent compositing relationships, and their object layers typically lack realistic visual effects such as shadows and reflections. To overcome these limitations, we propose LASAGNA, a novel, unified framework that generates an image jointly with its composing layers--a photorealistic background and a high-quality transparent foreground with compelling visual effects. Unlike prior work, LASAGNA efficiently learns correct image composition from a wide range of conditioning inputs--text prompts, foreground, background, and location masks--offering greater controllability for real-world applications. To enable this, we introduce LASAGNA-48K, a new dataset composed of clean backgrounds and RGBA foregrounds with physically grounded visual effects. We also propose LASAGNABENCH, the first benchmark for layer editing. We demonstrate that LASAGNA excels in generating highly consistent and coherent results across multiple image layers simultaneously, enabling diverse post-editing applications that accurately preserve identity and visual effects. LASAGNA-48K and LASAGNABENCH will be publicly released to foster open research in the community. The project page is https://rayjryang.github.io/LASAGNA-Page/.
>
---
#### [new 042] SAMTok: Representing Any Mask with Two Words
- **分类: cs.CV**

- **简介: 该论文提出SAMTok，解决多模态大模型的像素级能力扩展问题。通过将任意区域掩码转换为两个词，使模型能高效学习和生成掩码，提升多种视觉任务性能。**

- **链接: [https://arxiv.org/pdf/2601.16093v1](https://arxiv.org/pdf/2601.16093v1)**

> **作者:** Yikang Zhou; Tao Zhang; Dengxian Gong; Yuanzheng Wu; Ye Tian; Haochen Wang; Haobo Yuan; Jiacong Wang; Lu Qi; Hao Fei; Anran Wang; Zhuochen Wang; Yujing Wang; Cheng Chen; Shunping Ji; Xiangtai Li
>
> **备注:** 27 pages, 11 figures
>
> **摘要:** Pixel-wise capabilities are essential for building interactive intelligent systems. However, pixel-wise multi-modal LLMs (MLLMs) remain difficult to scale due to complex region-level encoders, specialized segmentation decoders, and incompatible training objectives. To address these challenges, we present SAMTok, a discrete mask tokenizer that converts any region mask into two special tokens and reconstructs the mask using these tokens with high fidelity. By treating masks as new language tokens, SAMTok enables base MLLMs (such as the QwenVL series) to learn pixel-wise capabilities through standard next-token prediction and simple reinforcement learning, without architectural modifications and specialized loss design. SAMTok builds on SAM2 and is trained on 209M diverse masks using a mask encoder and residual vector quantizer to produce discrete, compact, and information-rich tokens. With 5M SAMTok-formatted mask understanding and generation data samples, QwenVL-SAMTok attains state-of-the-art or comparable results on region captioning, region VQA, grounded conversation, referring segmentation, scene graph parsing, and multi-round interactive segmentation. We further introduce a textual answer-matching reward that enables efficient reinforcement learning for mask generation, delivering substantial improvements on GRES and GCG benchmarks. Our results demonstrate a scalable and straightforward paradigm for equipping MLLMs with strong pixel-wise capabilities. Our code and models are available.
>
---
#### [new 043] ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion
- **分类: cs.CV**

- **简介: 该论文提出ActionMesh，解决生成动画3D网格的问题。通过引入时间维度的3D扩散模型，实现快速、高质量的动画生成，支持多种输入方式。**

- **链接: [https://arxiv.org/pdf/2601.16148v1](https://arxiv.org/pdf/2601.16148v1)**

> **作者:** Remy Sabathier; David Novotny; Niloy J. Mitra; Tom Monnier
>
> **摘要:** Generating animated 3D objects is at the heart of many applications, yet most advanced works are typically difficult to apply in practice because of their limited setup, their long runtime, or their limited quality. We introduce ActionMesh, a generative model that predicts production-ready 3D meshes "in action" in a feed-forward manner. Drawing inspiration from early video models, our key insight is to modify existing 3D diffusion models to include a temporal axis, resulting in a framework we dubbed "temporal 3D diffusion". Specifically, we first adapt the 3D diffusion stage to generate a sequence of synchronized latents representing time-varying and independent 3D shapes. Second, we design a temporal 3D autoencoder that translates a sequence of independent shapes into the corresponding deformations of a pre-defined reference shape, allowing us to build an animation. Combining these two components, ActionMesh generates animated 3D meshes from different inputs like a monocular video, a text description, or even a 3D mesh with a text prompt describing its animation. Besides, compared to previous approaches, our method is fast and produces results that are rig-free and topology consistent, hence enabling rapid iteration and seamless applications like texturing and retargeting. We evaluate our model on standard video-to-4D benchmarks (Consistent4D, Objaverse) and report state-of-the-art performances on both geometric accuracy and temporal consistency, demonstrating that our model can deliver animated 3D meshes with unprecedented speed and quality.
>
---
#### [new 044] Class Confidence Aware Reweighting for Long Tailed Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.PF**

- **简介: 该论文属于长尾分布下的分类任务，旨在解决类别不平衡问题。通过设计一种基于类别和置信度的重加权方法，提升模型在尾部类别的性能。**

- **链接: [https://arxiv.org/pdf/2601.15924v1](https://arxiv.org/pdf/2601.15924v1)**

> **作者:** Brainard Philemon Jagati; Jitendra Tembhurne; Harsh Goud; Rudra Pratap Singh; Chandrashekhar Meshram
>
> **备注:** 9 pages, 3 figures, IEEE Transaction on Neural Networks and Learning Systems (Submitted)
>
> **摘要:** Deep neural network models degrade significantly in the long-tailed data distribution, with the overall training data dominated by a small set of classes in the head, and the tail classes obtaining less training examples. Addressing the imbalance in the classes, attention in the related literature was given mainly to the adjustments carried out in the decision space in terms of either corrections performed at the logit level in order to compensate class-prior bias, with the least attention to the optimization process resulting from the adjustments introduced through the differences in the confidences among the samples. In the current study, we present the design of a class and confidence-aware re-weighting scheme for long-tailed learning. This scheme is purely based upon the loss level and has a complementary nature to the existing methods performing the adjustment of the logits. In the practical implementation stage of the proposed scheme, we use an Ω(p_t, f_c) function. This function enables the modulation of the contribution towards the training task based upon the confidence value of the prediction, as well as the relative frequency of the corresponding class. Our observations in the experiments are corroborated by significant experimental results performed on the CIFAR-100-LT, ImageNet-LT, and iNaturalist2018 datasets under various values of imbalance factors that clearly authenticate the theoretical discussions above.
>
---
#### [new 045] Enhanced LULC Segmentation via Lightweight Model Refinements on ALOS-2 SAR Data
- **分类: cs.CV**

- **简介: 该论文属于LULC语义分割任务，解决SAR数据中的边界过平滑、细小结构遗漏和稀有类别退化问题，通过轻量级改进提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.15705v1](https://arxiv.org/pdf/2601.15705v1)**

> **作者:** Ali Caglayan; Nevrez Imamoglu; Toru Kouyama
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** This work focuses on national-scale land-use/land-cover (LULC) semantic segmentation using ALOS-2 single-polarization (HH) SAR data over Japan, together with a companion binary water detection task. Building on SAR-W-MixMAE self-supervised pretraining [1], we address common SAR dense-prediction failure modes, boundary over-smoothing, missed thin/slender structures, and rare-class degradation under long-tailed labels, without increasing pipeline complexity. We introduce three lightweight refinements: (i) injecting high-resolution features into multi-scale decoding, (ii) a progressive refine-up head that alternates convolutional refinement and stepwise upsampling, and (iii) an $α$-scale factor that tempers class reweighting within a focal+dice objective. The resulting model yields consistent improvements on the Japan-wide ALOS-2 LULC benchmark, particularly for under-represented classes, and improves water detection across standard evaluation metrics.
>
---
#### [new 046] LL-GaussianMap: Zero-shot Low-Light Image Enhancement via 2D Gaussian Splatting Guided Gain Maps
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决传统方法忽略图像结构先验的问题。提出LL-GaussianMap框架，利用2DGS生成增益图，提升增强效果并减少数据依赖。**

- **链接: [https://arxiv.org/pdf/2601.15766v1](https://arxiv.org/pdf/2601.15766v1)**

> **作者:** Yuhan Chen; Ying Fang; Guofa Li; Wenxuan Yu; Yicui Shi; Jingrui Zhang; Kefei Qian; Wenbo Chu; Keqiang Li
>
> **摘要:** Significant progress has been made in low-light image enhancement with respect to visual quality. However, most existing methods primarily operate in the pixel domain or rely on implicit feature representations. As a result, the intrinsic geometric structural priors of images are often neglected. 2D Gaussian Splatting (2DGS) has emerged as a prominent explicit scene representation technique characterized by superior structural fitting capabilities and high rendering efficiency. Despite these advantages, the utilization of 2DGS in low-level vision tasks remains unexplored. To bridge this gap, LL-GaussianMap is proposed as the first unsupervised framework incorporating 2DGS into low-light image enhancement. Distinct from conventional methodologies, the enhancement task is formulated as a gain map generation process guided by 2DGS primitives. The proposed method comprises two primary stages. First, high-fidelity structural reconstruction is executed utilizing 2DGS. Then, data-driven enhancement dictionary coefficients are rendered via the rasterization mechanism of Gaussian splatting through an innovative unified enhancement module. This design effectively incorporates the structural perception capabilities of 2DGS into gain map generation, thereby preserving edges and suppressing artifacts during enhancement. Additionally, the reliance on paired data is circumvented through unsupervised learning. Experimental results demonstrate that LL-GaussianMap achieves superior enhancement performance with an extremely low storage footprint, highlighting the effectiveness of explicit Gaussian representations for image enhancement.
>
---
#### [new 047] White-Box mHC: Electromagnetic Spectrum-Aware and Interpretable Stream Interactions for Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决深度学习模型的不透明性问题。提出ES-mHC框架，通过结构化矩阵建模光谱交互，提升模型可解释性与结构透明度。**

- **链接: [https://arxiv.org/pdf/2601.15757v1](https://arxiv.org/pdf/2601.15757v1)**

> **作者:** Yimin Zhu; Lincoln Linlin Xu; Zhengsen Xu; Zack Dewis; Mabel Heffring; Saeid Taleghanidoozdoozan; Motasem Alkayid; Quinn Ledingham; Megan Greenwood
>
> **摘要:** In hyperspectral image classification (HSIC), most deep learning models rely on opaque spectral-spatial feature mixing, limiting their interpretability and hindering understanding of internal decision mechanisms. We present physical spectrum-aware white-box mHC, named ES-mHC, a hyper-connection framework that explicitly models interactions among different electromagnetic spectrum groupings (residual stream in mHC) interactions using structured, directional matrices. By separating feature representation from interaction structure, ES-mHC promotes electromagnetic spectrum grouping specialization, reduces redundancy, and exposes internal information flow that can be directly visualized and spatially analyzed. Using hyperspectral image classification as a representative testbed, we demonstrate that the learned hyper-connection matrices exhibit coherent spatial patterns and asymmetric interaction behaviors, providing mechanistic insight into the model internal dynamics. Furthermore, we find that increasing the expansion rate accelerates the emergence of structured interaction patterns. These results suggest that ES-mHC transforms HSIC from a purely black-box prediction task into a structurally transparent, partially white-box learning process.
>
---
#### [new 048] AI-Based Culvert-Sewer Inspection
- **分类: cs.CV**

- **简介: 该论文属于缺陷分割任务，旨在解决排水管道检测中数据稀缺的问题。通过数据增强、新架构FORTRESS和小样本学习提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.15366v1](https://arxiv.org/pdf/2601.15366v1)**

> **作者:** Christina Thrainer
>
> **备注:** Masters thesis, University of Technology Graz, 2025
>
> **摘要:** Culverts and sewer pipes are critical components of drainage systems, and their failure can lead to serious risks to public safety and the environment. In this thesis, we explore methods to improve automated defect segmentation in culverts and sewer pipes. Collecting and annotating data in this field is cumbersome and requires domain knowledge. Having a large dataset for structural defect detection is therefore not feasible. Our proposed methods are tested under conditions with limited annotated data to demonstrate applicability to real-world scenarios. Overall, this thesis proposes three methods to significantly enhance defect segmentation and handle data scarcity. This can be addressed either by enhancing the training data or by adjusting a models architecture. First, we evaluate preprocessing strategies, including traditional data augmentation and dynamic label injection. These techniques significantly improve segmentation performance, increasing both Intersection over Union (IoU) and F1 score. Second, we introduce FORTRESS, a novel architecture that combines depthwise separable convolutions, adaptive Kolmogorov-Arnold Networks (KAN), and multi-scale attention mechanisms. FORTRESS achieves state-of-the-art performance on the culvert sewer pipe defect dataset, while significantly reducing the number of trainable parameters, as well as its computational cost. Finally, we investigate few-shot semantic segmentation and its applicability to defect detection. Few-shot learning aims to train models with only limited data available. By employing a bidirectional prototypical network with attention mechanisms, the model achieves richer feature representations and achieves satisfactory results across evaluation metrics.
>
---
#### [new 049] Learning to Watermark in the Latent Space of Generative Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于AI图像水印任务，旨在解决传统像素空间水印计算成本高、有损的问题。工作是提出在潜在空间进行水印，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.16140v1](https://arxiv.org/pdf/2601.16140v1)**

> **作者:** Sylvestre-Alvise Rebuffi; Tuan Tran; Valeriu Lacatusu; Pierre Fernandez; Tomáš Souček; Nikola Jovanović; Tom Sander; Hady Elsahar; Alexandre Mourachko
>
> **备注:** Code and models are available at https://github.com/facebookresearch/distseal
>
> **摘要:** Existing approaches for watermarking AI-generated images often rely on post-hoc methods applied in pixel space, introducing computational overhead and potential visual artifacts. In this work, we explore latent space watermarking and introduce DistSeal, a unified approach for latent watermarking that works across both diffusion and autoregressive models. Our approach works by training post-hoc watermarking models in the latent space of generative models. We demonstrate that these latent watermarkers can be effectively distilled either into the generative model itself or into the latent decoder, enabling in-model watermarking. The resulting latent watermarks achieve competitive robustness while offering similar imperceptibility and up to 20x speedup compared to pixel-space baselines. Our experiments further reveal that distilling latent watermarkers outperforms distilling pixel-space ones, providing a solution that is both more efficient and more robust.
>
---
#### [new 050] ProGiDiff: Prompt-Guided Diffusion-Based Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出ProGiDiff，用于医学图像分割任务，解决传统方法难以接受自然语言提示、无法多类别分割的问题。通过预训练扩散模型与自定义编码器结合，实现精准分割与跨模态适应。**

- **链接: [https://arxiv.org/pdf/2601.16060v1](https://arxiv.org/pdf/2601.16060v1)**

> **作者:** Yuan Lin; Murong Xu; Marc Hölle; Chinmay Prabhakar; Andreas Maier; Vasileios Belagiannis; Bjoern Menze; Suprosanna Shit
>
> **备注:** 5 pages, 4 figures. It has been accepted by IEEE ISBI
>
> **摘要:** Widely adopted medical image segmentation methods, although efficient, are primarily deterministic and remain poorly amenable to natural language prompts. Thus, they lack the capability to estimate multiple proposals, human interaction, and cross-modality adaptation. Recently, text-to-image diffusion models have shown potential to bridge the gap. However, training them from scratch requires a large dataset-a limitation for medical image segmentation. Furthermore, they are often limited to binary segmentation and cannot be conditioned on a natural language prompt. To this end, we propose a novel framework called ProGiDiff that leverages existing image generation models for medical image segmentation purposes. Specifically, we propose a ControlNet-style conditioning mechanism with a custom encoder, suitable for image conditioning, to steer a pre-trained diffusion model to output segmentation masks. It naturally extends to a multi-class setting simply by prompting the target organ. Our experiment on organ segmentation from CT images demonstrates strong performance compared to previous methods and could greatly benefit from an expert-in-the-loop setting to leverage multiple proposals. Importantly, we demonstrate that the learned conditioning mechanism can be easily transferred through low-rank, few-shot adaptation to segment MR images.
>
---
#### [new 051] Masked Modeling for Human Motion Recovery Under Occlusions
- **分类: cs.CV**

- **简介: 该论文属于人体运动恢复任务，解决遮挡下运动重建难题。提出MoRo框架，通过掩码建模实现高效、鲁棒的运动恢复。**

- **链接: [https://arxiv.org/pdf/2601.16079v1](https://arxiv.org/pdf/2601.16079v1)**

> **作者:** Zhiyin Qian; Siwei Zhang; Bharat Lal Bhatnagar; Federica Bogo; Siyu Tang
>
> **备注:** Project page: https://mikeqzy.github.io/MoRo
>
> **摘要:** Human motion reconstruction from monocular videos is a fundamental challenge in computer vision, with broad applications in AR/VR, robotics, and digital content creation, but remains challenging under frequent occlusions in real-world settings.Existing regression-based methods are efficient but fragile to missing observations, while optimization- and diffusion-based approaches improve robustness at the cost of slow inference speed and heavy preprocessing steps. To address these limitations, we leverage recent advances in generative masked modeling and present MoRo: Masked Modeling for human motion Recovery under Occlusions. MoRo is an occlusion-robust, end-to-end generative framework that formulates motion reconstruction as a video-conditioned task, and efficiently recover human motion in a consistent global coordinate system from RGB videos. By masked modeling, MoRo naturally handles occlusions while enabling efficient, end-to-end inference. To overcome the scarcity of paired video-motion data, we design a cross-modality learning scheme that learns multi-modal priors from a set of heterogeneous datasets: (i) a trajectory-aware motion prior trained on MoCap datasets, (ii) an image-conditioned pose prior trained on image-pose datasets, capturing diverse per-frame poses, and (iii) a video-conditioned masked transformer that fuses motion and pose priors, finetuned on video-motion datasets to integrate visual cues with motion dynamics for robust inference. Extensive experiments on EgoBody and RICH demonstrate that MoRo substantially outperforms state-of-the-art methods in accuracy and motion realism under occlusions, while performing on-par in non-occluded scenarios. MoRo achieves real-time inference at 70 FPS on a single H200 GPU.
>
---
#### [new 052] PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PyraTok，用于视频理解与生成任务，解决跨模态对齐和零样本迁移问题。通过多尺度离散化和语言对齐优化，提升视频表示效果。**

- **链接: [https://arxiv.org/pdf/2601.16210v1](https://arxiv.org/pdf/2601.16210v1)**

> **作者:** Onkar Susladkar; Tushar Prakash; Adheesh Juvekar; Kiet A. Nguyen; Dong-Hwan Jang; Inderjit S Dhillon; Ismini Lourentzou
>
> **摘要:** Discrete video VAEs underpin modern text-to-video generation and video understanding systems, yet existing tokenizers typically learn visual codebooks at a single scale with limited vocabularies and shallow language supervision, leading to poor cross-modal alignment and zero-shot transfer. We introduce PyraTok, a language-aligned pyramidal tokenizer that learns semantically structured discrete latents across multiple spatiotemporal resolutions. PyraTok builds on a pretrained video VAE and a novel Language aligned Pyramidal Quantization (LaPQ) module that discretizes encoder features at several depths using a shared large binary codebook, yielding compact yet expressive video token sequences. To tightly couple visual tokens with language, PyraTok jointly optimizes multi-scale text-guided quantization and a global autoregressive objective over the token hierarchy. Across ten benchmarks, PyraTok delivers state-of-the-art (SOTA) video reconstruction, consistently improves text-to-video quality, and sets new SOTA zero-shot performance on video segmentation, temporal action localization, and video understanding, scaling robustly to up to 4K/8K resolutions.
>
---
#### [new 053] VIOLA: Towards Video In-Context Learning with Minimal Annotations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频领域的小样本学习任务，旨在解决标注数据稀缺问题。通过引入VIOLA框架，结合少量标注与大量未标注数据，提升模型在新视频领域的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.15549v1](https://arxiv.org/pdf/2601.15549v1)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **摘要:** Generalizing Multimodal Large Language Models (MLLMs) to novel video domains is essential for real-world deployment but remains challenging due to the scarcity of labeled data. While In-Context Learning (ICL) offers a training-free adaptation path, standard methods rely on large annotated pools, which are often impractical in specialized environments like industrial or surgical settings since they require the experts' annotations. To bridge this gap, we introduce VIOLA (Video In-cOntext Learning with minimal Annotation), a label-efficient framework that synergizes minimal expert supervision with abundant unlabeled data. First, to maximize the efficiency of a strict annotation budget, we propose density-uncertainty-weighted sampling. Unlike standard diversity or uncertainty strategies that risk selecting visual outliers, our method leverages density estimation to identify samples that are simultaneously diverse, representative, and informative. Second, to utilize the remaining unlabeled data without noise propagation, we construct a hybrid pool and introduce confidence-aware retrieval and confidence-aware prompting. These mechanisms explicitly model label reliability, retrieving demonstrations based on a composite score of similarity and confidence while enabling the MLLM to adaptively distinguish between verified ground truths and noisy pseudo-labels. Extensive experiments across nine diverse benchmarks using four MLLMs demonstrate that our framework significantly outperforms various baselines in low-resource settings, achieving robust adaptation with minimal annotation costs.
>
---
#### [new 054] A Mobile Application for Flower Recognition System Based on Convolutional Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于CNN的移动应用，用于识别花朵类型。解决非专业人士难以快速识别花卉的问题，采用MobileNet、DenseNet121和Xception模型进行分类，DenseNet121表现最佳。**

- **链接: [https://arxiv.org/pdf/2601.15810v1](https://arxiv.org/pdf/2601.15810v1)**

> **作者:** Mustafa Yurdakul; Enes Ayan; Fahrettin Horasan; Sakir Tasdemir
>
> **摘要:** A convolutional neural network (CNN) is a deep learning algorithm that has been specifically designed for computer vision applications. The CNNs proved successful in handling the increasing amount of data in many computer vision problems, where classical machine learning algorithms were insufficient. Flowers have many uses in our daily lives, from decorating to making medicines to detoxifying the environment. Identifying flower types requires expert knowledge. However, accessing experts at any time and in any location may not always be feasible. In this study a mobile application based on CNNs was developed to recognize different types of flowers to provide non-specialists with quick and easy access to information about flower types. The study employed three distinct CNN models, namely MobileNet, DenseNet121, and Xception, to determine the most suitable model for the mobile application. The classification performances of the models were evaluated by training them with seven different optimization algorithms. The DenseNet-121 architecture, which uses the stochastic gradient descent (SGD) optimization algorithm, was the most successful, achieving 95.84 % accuracy, 96.00% precision, recall, and F1-score. This result shows that CNNs can be used for flower classification in mobile applications.
>
---
#### [new 055] Breaking the Resolution Barrier: Arbitrary-resolution Deep Image Steganography Framework
- **分类: cs.CV**

- **简介: 该论文属于图像隐写任务，解决秘密图像与载体图像分辨率不一致导致的细节丢失和无法恢复原分辨率的问题。提出ARDIS框架，实现任意分辨率的隐写与恢复。**

- **链接: [https://arxiv.org/pdf/2601.15739v1](https://arxiv.org/pdf/2601.15739v1)**

> **作者:** Xinjue Hu; Chi Wang; Boyu Wang; Xiang Zhang; Zhenshan Tan; Zhangjie Fu
>
> **摘要:** Deep image steganography (DIS) has achieved significant results in capacity and invisibility. However, current paradigms enforce the secret image to maintain the same resolution as the cover image during hiding and revealing. This leads to two challenges: secret images with inconsistent resolutions must undergo resampling beforehand which results in detail loss during recovery, and the secret image cannot be recovered to its original resolution when the resolution value is unknown. To address these, we propose ARDIS, the first Arbitrary Resolution DIS framework, which shifts the paradigm from discrete mapping to reference-guided continuous signal reconstruction. Specifically, to minimize the detail loss caused by resolution mismatch, we first design a Frequency Decoupling Architecture in hiding stage. It disentangles the secret into a resolution-aligned global basis and a resolution-agnostic high-frequency latent to hide in a fixed-resolution cover. Second, for recovery, we propose a Latent-Guided Implicit Reconstructor to perform deterministic restoration. The recovered detail latent code modulates a continuous implicit function to accurately query and render high-frequency residuals onto the recovered global basis, ensuring faithful restoration of original details. Furthermore, to achieve blind recovery, we introduce an Implicit Resolution Coding strategy. By transforming discrete resolution values into dense feature maps and hiding them in the redundant space of the feature domain, the reconstructor can correctly decode the secret's resolution directly from the steganographic representation. Experimental results demonstrate that ARDIS significantly outperforms state-of-the-art methods in both invisibility and cross-resolution recovery fidelity.
>
---
#### [new 056] PAINT: Pathology-Aware Integrated Next-Scale Transformation for Virtual Immunohistochemistry
- **分类: cs.CV**

- **简介: 该论文提出PAINT方法，用于虚拟免疫组化合成。任务是将H&E图像转换为分子染色模式，解决结构与语义不一致问题，通过结构引导的生成模型提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.16024v1](https://arxiv.org/pdf/2601.16024v1)**

> **作者:** Rongze Ma; Mengkang Lu; Zhenyu Xiang; Yongsheng Pan; Yicheng Wu; Qingjie Zeng; Yong Xia
>
> **摘要:** Virtual immunohistochemistry (IHC) aims to computationally synthesize molecular staining patterns from routine Hematoxylin and Eosin (H\&E) images, offering a cost-effective and tissue-efficient alternative to traditional physical staining. However, this task is particularly challenging: H\&E morphology provides ambiguous cues about protein expression, and similar tissue structures may correspond to distinct molecular states. Most existing methods focus on direct appearance synthesis to implicitly achieve cross-modal generation, often resulting in semantic inconsistencies due to insufficient structural priors. In this paper, we propose Pathology-Aware Integrated Next-Scale Transformation (PAINT), a visual autoregressive framework that reformulates the synthesis process as a structure-first conditional generation task. Unlike direct image translation, PAINT enforces a causal order by resolving molecular details conditioned on a global structural layout. Central to this approach is the introduction of a Spatial Structural Start Map (3S-Map), which grounds the autoregressive initialization in observed morphology, ensuring deterministic, spatially aligned synthesis. Experiments on the IHC4BC and MIST datasets demonstrate that PAINT outperforms state-of-the-art methods in structural fidelity and clinical downstream tasks, validating the potential of structure-guided autoregressive modeling.
>
---
#### [new 057] Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders
- **分类: cs.CV**

- **简介: 本文研究文本到图像生成任务，探索使用表示自编码器（RAE）替代VAE的可行性。通过实验验证RAE在大规模数据下的优越性，证明其更稳定、高效。**

- **链接: [https://arxiv.org/pdf/2601.16208v1](https://arxiv.org/pdf/2601.16208v1)**

> **作者:** Shengbang Tong; Boyang Zheng; Ziteng Wang; Bingda Tang; Nanye Ma; Ellis Brown; Jihan Yang; Rob Fergus; Yann LeCun; Saining Xie
>
> **备注:** website: https://rae-dit.github.io/scale-rae/
>
> **摘要:** Representation Autoencoders (RAEs) have shown distinct advantages in diffusion modeling on ImageNet by training in high-dimensional semantic latent spaces. In this work, we investigate whether this framework can scale to large-scale, freeform text-to-image (T2I) generation. We first scale RAE decoders on the frozen representation encoder (SigLIP-2) beyond ImageNet by training on web, synthetic, and text-rendering data, finding that while scale improves general fidelity, targeted data composition is essential for specific domains like text. We then rigorously stress-test the RAE design choices originally proposed for ImageNet. Our analysis reveals that scaling simplifies the framework: while dimension-dependent noise scheduling remains critical, architectural complexities such as wide diffusion heads and noise-augmented decoding offer negligible benefits at scale Building on this simplified framework, we conduct a controlled comparison of RAE against the state-of-the-art FLUX VAE across diffusion transformer scales from 0.5B to 9.8B parameters. RAEs consistently outperform VAEs during pretraining across all model scales. Further, during finetuning on high-quality datasets, VAE-based models catastrophically overfit after 64 epochs, while RAE models remain stable through 256 epochs and achieve consistently better performance. Across all experiments, RAE-based diffusion models demonstrate faster convergence and better generation quality, establishing RAEs as a simpler and stronger foundation than VAEs for large-scale T2I generation. Additionally, because both visual understanding and generation can operate in a shared representation space, the multimodal model can directly reason over generated latents, opening new possibilities for unified models.
>
---
#### [new 058] Beyond Off-the-Shelf Models: A Lightweight and Accessible Machine Learning Pipeline for Ecologists Working with Image Data
- **分类: cs.CV; cs.LG**

- **简介: 论文提出一个轻量级机器学习流程，帮助生态学家处理图像数据。任务是图像分类，解决传统模型无法适应本地数据的问题。工作包括设计工具并应用于红鹿年龄和性别分类。**

- **链接: [https://arxiv.org/pdf/2601.15813v1](https://arxiv.org/pdf/2601.15813v1)**

> **作者:** Clare Chemery; Hendrik Edelhoff; Ludwig Bothmann
>
> **摘要:** We introduce a lightweight experimentation pipeline designed to lower the barrier for applying machine learning (ML) methods for classifying images in ecological research. We enable ecologists to experiment with ML models independently, thus they can move beyond off-the-shelf models and generate insights tailored to local datasets and specific classification tasks and target variables. Our tool combines a simple command-line interface for preprocessing, training, and evaluation with a graphical interface for annotation, error analysis, and model comparison. This design enables ecologists to build and iterate on compact, task-specific classifiers without requiring advanced ML expertise. As a proof of concept, we apply the pipeline to classify red deer (Cervus elaphus) by age and sex from 3392 camera trap images collected in the Veldenstein Forest, Germany. Using 4352 cropped images containing individual deer labeled by experts, we trained and evaluated multiple backbone architectures with a wide variety of parameters and data augmentation strategies. Our best-performing models achieved 90.77% accuracy for age classification and 96.15% for sex classification. These results demonstrate that reliable demographic classification is feasible even with limited data to answer narrow, well-defined ecological problems. More broadly, the framework provides ecologists with an accessible tool for developing ML models tailored to specific research questions, paving the way for broader adoption of ML in wildlife monitoring and demographic analysis.
>
---
#### [new 059] DuFal: Dual-Frequency-Aware Learning for High-Fidelity Extremely Sparse-view CBCT Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像重建任务，旨在解决稀疏视角锥形束CT重建中高频率结构丢失的问题。提出DuFal框架，通过双路径结构融合频域与空域信息，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2601.15416v1](https://arxiv.org/pdf/2601.15416v1)**

> **作者:** Cuong Tran Van; Trong-Thang Pham; Ngoc-Son Nguyen; Duy Minh Ho Nguyen; Ngan Le
>
> **备注:** Published with J2C Certification in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Sparse-view Cone-Beam Computed Tomography reconstruction from limited X-ray projections remains a challenging problem in medical imaging due to the inherent undersampling of fine-grained anatomical details, which correspond to high-frequency components. Conventional CNN-based methods often struggle to recover these fine structures, as they are typically biased toward learning low-frequency information. To address this challenge, this paper presents DuFal (Dual-Frequency-Aware Learning), a novel framework that integrates frequency-domain and spatial-domain processing via a dual-path architecture. The core innovation lies in our High-Local Factorized Fourier Neural Operator, which comprises two complementary branches: a Global High-Frequency Enhanced Fourier Neural Operator that captures global frequency patterns and a Local High-Frequency Enhanced Fourier Neural Operator that processes spatially partitioned patches to preserve spatial locality that might be lost in global frequency analysis. To improve efficiency, we design a Spectral-Channel Factorization scheme that reduces the Fourier Neural Operator parameter count. We also design a Cross-Attention Frequency Fusion module to integrate spatial and frequency features effectively. The fused features are then decoded through a Feature Decoder to produce projection representations, which are subsequently processed through an Intensity Field Decoding pipeline to reconstruct a final Computed Tomography volume. Experimental results on the LUNA16 and ToothFairy datasets demonstrate that DuFal significantly outperforms existing state-of-the-art methods in preserving high-frequency anatomical features, particularly under extremely sparse-view settings.
>
---
#### [new 060] 360Anything: Geometry-Free Lifting of Images and Videos to 360°
- **分类: cs.CV**

- **简介: 该论文提出360Anything，解决从透视图像视频生成360°全景的问题，无需几何信息，通过数据驱动方法实现。**

- **链接: [https://arxiv.org/pdf/2601.16192v1](https://arxiv.org/pdf/2601.16192v1)**

> **作者:** Ziyi Wu; Daniel Watson; Andrea Tagliasacchi; David J. Fleet; Marcus A. Brubaker; Saurabh Saxena
>
> **备注:** Project page: https://360anything.github.io/
>
> **摘要:** Lifting perspective images and videos to 360° panoramas enables immersive 3D world generation. Existing approaches often rely on explicit geometric alignment between the perspective and the equirectangular projection (ERP) space. Yet, this requires known camera metadata, obscuring the application to in-the-wild data where such calibration is typically absent or noisy. We propose 360Anything, a geometry-free framework built upon pre-trained diffusion transformers. By treating the perspective input and the panorama target simply as token sequences, 360Anything learns the perspective-to-equirectangular mapping in a purely data-driven way, eliminating the need for camera information. Our approach achieves state-of-the-art performance on both image and video perspective-to-360° generation, outperforming prior works that use ground-truth camera information. We also trace the root cause of the seam artifacts at ERP boundaries to zero-padding in the VAE encoder, and introduce Circular Latent Encoding to facilitate seamless generation. Finally, we show competitive results in zero-shot camera FoV and orientation estimation benchmarks, demonstrating 360Anything's deep geometric understanding and broader utility in computer vision tasks. Additional results are available at https://360anything.github.io/.
>
---
#### [new 061] Atlas-Assisted Segment Anything Model for Fetal Brain MRI (FeTal-SAM)
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于胎儿脑部MRI分割任务，旨在解决传统方法需重新训练及依赖图像对比的问题。通过结合图谱提示和基础模型，提出FeTal-SAM，实现灵活、准确的分割。**

- **链接: [https://arxiv.org/pdf/2601.15759v1](https://arxiv.org/pdf/2601.15759v1)**

> **作者:** Qi Zeng; Weide Liu; Bo Li; Ryne Didier; P. Ellen Grant; Davood Karimi
>
> **摘要:** This paper presents FeTal-SAM, a novel adaptation of the Segment Anything Model (SAM) tailored for fetal brain MRI segmentation. Traditional deep learning methods often require large annotated datasets for a fixed set of labels, making them inflexible when clinical or research needs change. By integrating atlas-based prompts and foundation-model principles, FeTal-SAM addresses two key limitations in fetal brain MRI segmentation: (1) the need to retrain models for varying label definitions, and (2) the lack of insight into whether segmentations are driven by genuine image contrast or by learned spatial priors. We leverage multi-atlas registration to generate spatially aligned label templates that serve as dense prompts, alongside a bounding-box prompt, for SAM's segmentation decoder. This strategy enables binary segmentation on a per-structure basis, which is subsequently fused to reconstruct the full 3D segmentation volumes. Evaluations on two datasets, the dHCP dataset and an in-house dataset demonstrate FeTal-SAM's robust performance across gestational ages. Notably, it achieves Dice scores comparable to state-of-the-art baselines which were trained for each dataset and label definition for well-contrasted structures like cortical plate and cerebellum, while maintaining the flexibility to segment any user-specified anatomy. Although slightly lower accuracy is observed for subtle, low-contrast structures (e.g., hippocampus, amygdala), our results highlight FeTal-SAM's potential to serve as a general-purpose segmentation model without exhaustive retraining. This method thus constitutes a promising step toward clinically adaptable fetal brain MRI analysis tools.
>
---
#### [new 062] VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VideoThinker，解决长视频理解问题。通过合成工具交互轨迹训练，提升模型动态推理与多步骤工具使用能力。**

- **链接: [https://arxiv.org/pdf/2601.15724v1](https://arxiv.org/pdf/2601.15724v1)**

> **作者:** Chenglin Li; Qianglong Chen; Feng Han; Yikun Wang; Xingxi Yin; Yan Gong; Ruilin Li; Yin Zhang; Jiaqi Wang
>
> **摘要:** Long-form video understanding remains a fundamental challenge for current Video Large Language Models. Most existing models rely on static reasoning over uniformly sampled frames, which weakens temporal localization and leads to substantial information loss in long videos. Agentic tools such as temporal retrieval, spatial zoom, and temporal zoom offer a natural way to overcome these limitations by enabling adaptive exploration of key moments. However, constructing agentic video understanding data requires models that already possess strong long-form video comprehension, creating a circular dependency. We address this challenge with VideoThinker, an agentic Video Large Language Model trained entirely on synthetic tool interaction trajectories. Our key idea is to convert videos into rich captions and employ a powerful agentic language model to generate multi-step tool use sequences in caption space. These trajectories are subsequently grounded back to video by replacing captions with the corresponding frames, yielding a large-scale interleaved video and tool reasoning dataset without requiring any long-form understanding from the underlying model. Training on this synthetic agentic dataset equips VideoThinker with dynamic reasoning capabilities, adaptive temporal exploration, and multi-step tool use. Remarkably, VideoThinker significantly outperforms both caption-only language model agents and strong video model baselines across long-video benchmarks, demonstrating the effectiveness of tool augmented synthetic data and adaptive retrieval and zoom reasoning for long-form video understanding.
>
---
#### [new 063] Why Can't I Open My Drawer? Mitigating Object-Driven Shortcuts in Zero-Shot Compositional Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于零样本组合动作识别任务，旨在解决物体驱动的动词捷径问题。通过引入RCORE框架，增强时序约束的动词学习，提升未见组合的识别性能。**

- **链接: [https://arxiv.org/pdf/2601.16211v1](https://arxiv.org/pdf/2601.16211v1)**

> **作者:** Geo Ahn; Inwoong Lee; Taeoh Kim; Minho Shim; Dongyoon Wee; Jinwoo Choi
>
> **备注:** The code is available at https://github.com/KHU-VLL/RCORE
>
> **摘要:** We study Compositional Video Understanding (CVU), where models must recognize verbs and objects and compose them to generalize to unseen combinations. We find that existing Zero-Shot Compositional Action Recognition (ZS-CAR) models fail primarily due to an overlooked failure mode: object-driven verb shortcuts. Through systematic analysis, we show that this behavior arises from two intertwined factors: severe sparsity and skewness of compositional supervision, and the asymmetric learning difficulty between verbs and objects. As training progresses, the existing ZS-CAR model increasingly ignores visual evidence and overfits to co-occurrence statistics. Consequently, the existing model does not gain the benefit of compositional recognition in unseen verb-object compositions. To address this, we propose RCORE, a simple and effective framework that enforces temporally grounded verb learning. RCORE introduces (i) a composition-aware augmentation that diversifies verb-object combinations without corrupting motion cues, and (ii) a temporal order regularization loss that penalizes shortcut behaviors by explicitly modeling temporal structure. Across two benchmarks, Sth-com and our newly constructed EK100-com, RCORE significantly improves unseen composition accuracy, reduces reliance on co-occurrence bias, and achieves consistently positive compositional gaps. Our findings reveal object-driven shortcuts as a critical limiting factor in ZS-CAR and demonstrate that addressing them is essential for robust compositional video understanding.
>
---
#### [new 064] DevPrompt: Deviation-Based Prompt Learning for One-Normal ShotImage Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于少样本图像异常检测任务，解决正常与异常提示区分度低及缺乏有效评分机制的问题。提出基于偏差的提示学习框架，提升检测精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.15453v1](https://arxiv.org/pdf/2601.15453v1)**

> **作者:** Morteza Poudineh; Marc Lalonde
>
> **备注:** 8 pages
>
> **摘要:** Few-normal shot anomaly detection (FNSAD) aims to detect abnormal regions in images using only a few normal training samples, making the task highly challenging due to limited supervision and the diversity of potential defects. Recent approaches leverage vision-language models such as CLIP with prompt-based learning to align image and text features. However, existing methods often exhibit weak discriminability between normal and abnormal prompts and lack principled scoring mechanisms for patch-level anomalies. We propose a deviation-guided prompt learning framework that integrates the semantic power of vision-language models with the statistical reliability of deviation-based scoring. Specifically, we replace fixed prompt prefixes with learnable context vectors shared across normal and abnormal prompts, while anomaly-specific suffix tokens enable class-aware alignment. To enhance separability, we introduce a deviation loss with Top-K Multiple Instance Learning (MIL), modeling patch-level features as Gaussian deviations from the normal distribution. This allows the network to assign higher anomaly scores to patches with statistically significant deviations, improving localization and interpretability. Experiments on the MVTecAD and VISA benchmarks demonstrate superior pixel-level detection performance compared to PromptAD and other baselines. Ablation studies further validate the effectiveness of learnable prompts, deviation-based scoring, and the Top-K MIL strategy.
>
---
#### [new 065] HyperAlign: Hypernetwork for Efficient Test-Time Alignment of Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型生成结果与人类偏好不一致的问题。提出HyperAlign框架，通过超网络动态调整生成过程，提升语义一致性和视觉吸引力。**

- **链接: [https://arxiv.org/pdf/2601.15968v1](https://arxiv.org/pdf/2601.15968v1)**

> **作者:** Xin Xie; Jiaxian Guo; Dong Gong
>
> **摘要:** Diffusion models achieve state-of-the-art performance but often fail to generate outputs that align with human preferences and intentions, resulting in images with poor aesthetic quality and semantic inconsistencies. Existing alignment methods present a difficult trade-off: fine-tuning approaches suffer from loss of diversity with reward over-optimization, while test-time scaling methods introduce significant computational overhead and tend to under-optimize. To address these limitations, we propose HyperAlign, a novel framework that trains a hypernetwork for efficient and effective test-time alignment. Instead of modifying latent states, HyperAlign dynamically generates low-rank adaptation weights to modulate the diffusion model's generation operators. This allows the denoising trajectory to be adaptively adjusted based on input latents, timesteps and prompts for reward-conditioned alignment. We introduce multiple variants of HyperAlign that differ in how frequently the hypernetwork is applied, balancing between performance and efficiency. Furthermore, we optimize the hypernetwork using a reward score objective regularized with preference data to reduce reward hacking. We evaluate HyperAlign on multiple extended generative paradigms, including Stable Diffusion and FLUX. It significantly outperforms existing fine-tuning and test-time scaling baselines in enhancing semantic consistency and visual appeal.
>
---
#### [new 066] A Multi-View Pipeline and Benchmark Dataset for 3D Hand Pose Estimation in Surgery
- **分类: cs.CV**

- **简介: 该论文属于3D手部姿态估计任务，旨在解决手术环境中手部姿态准确估计的问题。提出多视角管道和标注数据集，提升手术场景下的手部姿态估计性能。**

- **链接: [https://arxiv.org/pdf/2601.15918v1](https://arxiv.org/pdf/2601.15918v1)**

> **作者:** Valery Fischer; Alan Magdaleno; Anna-Katharina Calek; Nicola Cavalcanti; Nathan Hoffman; Christoph Germann; Joschua Wüthrich; Max Krähenmann; Mazda Farshad; Philipp Fürnstahl; Lilian Calvet
>
> **摘要:** Purpose: Accurate 3D hand pose estimation supports surgical applications such as skill assessment, robot-assisted interventions, and geometry-aware workflow analysis. However, surgical environments pose severe challenges, including intense and localized lighting, frequent occlusions by instruments or staff, and uniform hand appearance due to gloves, combined with a scarcity of annotated datasets for reliable model training. Method: We propose a robust multi-view pipeline for 3D hand pose estimation in surgical contexts that requires no domain-specific fine-tuning and relies solely on off-the-shelf pretrained models. The pipeline integrates reliable person detection, whole-body pose estimation, and state-of-the-art 2D hand keypoint prediction on tracked hand crops, followed by a constrained 3D optimization. In addition, we introduce a novel surgical benchmark dataset comprising over 68,000 frames and 3,000 manually annotated 2D hand poses with triangulated 3D ground truth, recorded in a replica operating room under varying levels of scene complexity. Results: Quantitative experiments demonstrate that our method consistently outperforms baselines, achieving a 31% reduction in 2D mean joint error and a 76% reduction in 3D mean per-joint position error. Conclusion: Our work establishes a strong baseline for 3D hand pose estimation in surgery, providing both a training-free pipeline and a comprehensive annotated dataset to facilitate future research in surgical computer vision.
>
---
#### [new 067] Performance-guided Reinforced Active Learning for Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，旨在解决传统主动学习未能直接关联模型性能的问题。提出MGRAL方法，通过mAP引导的强化学习选择最具信息量样本，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.15688v1](https://arxiv.org/pdf/2601.15688v1)**

> **作者:** Zhixuan Liang; Xingyu Zeng; Rui Zhao; Ping Luo
>
> **备注:** Accepted by ICASSP 2026. Camera-ready Version
>
> **摘要:** Active learning (AL) strategies aim to train high-performance models with minimal labeling efforts, only selecting the most informative instances for annotation. Current approaches to evaluating data informativeness predominantly focus on the data's distribution or intrinsic information content and do not directly correlate with downstream task performance, such as mean average precision (mAP) in object detection. Thus, we propose Performance-guided (i.e. mAP-guided) Reinforced Active Learning for Object Detection (MGRAL), a novel approach that leverages the concept of expected model output changes as informativeness. To address the combinatorial explosion challenge of batch sample selection and the non-differentiable correlation between model performance and selected batches, MGRAL skillfully employs a reinforcement learning-based sampling agent that optimizes selection using policy gradient with mAP improvement as reward. Moreover, to reduce the computational overhead of mAP estimation with unlabeled samples, MGRAL utilizes an unsupervised way with fast look-up tables, ensuring feasible deployment. We evaluate MGRAL's active learning performance on detection tasks over PASCAL VOC and COCO benchmarks. Our approach demonstrates the highest AL curve with convincing visualizations, establishing a new paradigm in reinforcement learning-driven active object detection.
>
---
#### [new 068] Out-of-Distribution Detection Based on Total Variation Estimation
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决模型部署中分布偏移问题。提出TV-OOD方法，通过总变分估计区分正常与异常数据。**

- **链接: [https://arxiv.org/pdf/2601.15867v1](https://arxiv.org/pdf/2601.15867v1)**

> **作者:** Dabiao Ma; Zhiba Su; Jian Yang; Haojun Fei
>
> **摘要:** This paper introduces a novel approach to securing machine learning model deployments against potential distribution shifts in practical applications, the Total Variation Out-of-Distribution (TV-OOD) detection method. Existing methods have produced satisfactory results, but TV-OOD improves upon these by leveraging the Total Variation Network Estimator to calculate each input's contribution to the overall total variation. By defining this as the total variation score, TV-OOD discriminates between in- and out-of-distribution data. The method's efficacy was tested across a range of models and datasets, consistently yielding results in image classification tasks that were either comparable or superior to those achieved by leading-edge out-of-distribution detection techniques across all evaluation metrics.
>
---
#### [new 069] synthocr-gen: A synthetic ocr dataset generator for low-resource languages- breaking the data barrier
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OCR任务，旨在解决低资源语言缺乏标注数据的问题。作者开发了SynthOCR-Gen工具，生成合成OCR数据集，支持如克什米尔语等语言的OCR训练。**

- **链接: [https://arxiv.org/pdf/2601.16113v1](https://arxiv.org/pdf/2601.16113v1)**

> **作者:** Haq Nawaz Malik; Kh Mohmad Shafi; Tanveer Ahmad Reshi
>
> **摘要:** Optical Character Recognition (OCR) for low-resource languages remains a significant challenge due to the scarcity of large-scale annotated training datasets. Languages such as Kashmiri, with approximately 7 million speakers and a complex Perso-Arabic script featuring unique diacritical marks, currently lack support in major OCR systems including Tesseract, TrOCR, and PaddleOCR. Manual dataset creation for such languages is prohibitively expensive, time-consuming, and error-prone, often requiring word by word transcription of printed or handwritten text. We present SynthOCR-Gen, an open-source synthetic OCR dataset generator specifically designed for low-resource languages. Our tool addresses the fundamental bottleneck in OCR development by transforming digital Unicode text corpora into ready-to-use training datasets. The system implements a comprehensive pipeline encompassing text segmentation (character, word, n-gram, sentence, and line levels), Unicode normalization with script purity enforcement, multi-font rendering with configurable distribution, and 25+ data augmentation techniques simulating real-world document degradations including rotation, blur, noise, and scanner artifacts. We demonstrate the efficacy of our approach by generating a 600,000-sample word-segmented Kashmiri OCR dataset, which we release publicly on HuggingFace. This work provides a practical pathway for bringing low-resource languages into the era of vision-language AI models, and the tool is openly available for researchers and practitioners working with underserved writing systems worldwide.
>
---
#### [new 070] A Machine Vision Approach to Preliminary Skin Lesion Assessments
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于皮肤病变初步评估任务，旨在提升恶性病变的早期检测。通过结合临床规则与机器学习，提出一种基于CNN的轻量级模型，有效提升诊断准确率。**

- **链接: [https://arxiv.org/pdf/2601.15539v1](https://arxiv.org/pdf/2601.15539v1)**

> **作者:** Ali Khreis; Ro'Yah Radaideh; Quinn McGill
>
> **备注:** 6 pages, 2 figures, 2 tables
>
> **摘要:** Early detection of malignant skin lesions is critical for improving patient outcomes in aggressive, metastatic skin cancers. This study evaluates a comprehensive system for preliminary skin lesion assessment that combines the clinically established ABCD rule of dermoscopy (analyzing Asymmetry, Borders, Color, and Dermoscopic Structures) with machine learning classification. Using a 1,000-image subset of the HAM10000 dataset, the system implements an automated, rule-based pipeline to compute a Total Dermoscopy Score (TDS) for each lesion. This handcrafted approach is compared against various machine learning solutions, including traditional classifiers (Logistic Regression, Random Forest, and SVM) and deep learning models. While the rule-based system provides high clinical interpretability, results indicate a performance bottleneck when reducing complex morphology to five numerical features. Experimental findings show that transfer learning with EfficientNet-B0 failed significantly due to domain shift between natural and medical images. In contrast, a custom three-layer Convolutional Neural Network (CNN) trained from scratch achieved 78.5% accuracy and 86.5% recall on median-filtered images, representing a 19-point accuracy improvement over traditional methods. The results demonstrate that direct pixel-level learning captures diagnostic patterns beyond handcrafted features and that purpose-built lightweight architectures can outperform large pretrained models for small, domain-specific medical datasets.
>
---
#### [new 071] Uncertainty-guided Generation of Dark-field Radiographs
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决暗场X射线图像稀缺的问题。通过引入不确定性指导的生成对抗网络，从常规X光片生成高质量暗场图像。**

- **链接: [https://arxiv.org/pdf/2601.15859v1](https://arxiv.org/pdf/2601.15859v1)**

> **作者:** Lina Felsner; Henriette Bast; Tina Dorosti; Florian Schaff; Franz Pfeiffer; Daniela Pfeiffer; Julia Schnabel
>
> **摘要:** X-ray dark-field radiography provides complementary diagnostic information to conventional attenuation imaging by visualizing microstructural tissue changes through small-angle scattering. However, the limited availability of such data poses challenges for developing robust deep learning models. In this work, we present the first framework for generating dark-field images directly from standard attenuation chest X-rays using an Uncertainty-Guided Progressive Generative Adversarial Network. The model incorporates both aleatoric and epistemic uncertainty to improve interpretability and reliability. Experiments demonstrate high structural fidelity of the generated images, with consistent improvement of quantitative metrics across stages. Furthermore, out-of-distribution evaluation confirms that the proposed model generalizes well. Our results indicate that uncertainty-guided generative modeling enables realistic dark-field image synthesis and provides a reliable foundation for future clinical applications.
>
---
#### [new 072] SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于渲染领域，解决3D高斯溅射与传统渲染管线集成困难的问题，通过GPU IPC实现与Unity等工具的无缝对接。**

- **链接: [https://arxiv.org/pdf/2601.15431v1](https://arxiv.org/pdf/2601.15431v1)**

> **作者:** Yinghan Xu; Théo Morales; John Dingliana
>
> **摘要:** Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities. They enable high-fidelity rendering with complex real-world lighting effects, but at the cost of high rendering time. 3D Gaussian Splatting solves this issue with a rasterisation-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality. However, current 3DGS implementations are difficult to integrate into traditional mesh-based rendering pipelines, which is a common use case for interactive applications and artistic exploration. To address this limitation, this software solution uses Nvidia's interprocess communication (IPC) APIs to easily integrate into implementations and allow the results to be viewed in external clients such as Unity, Blender, Unreal Engine, and OpenGL viewers. The code is available at https://github.com/RockyXu66/splatbus.
>
---
#### [new 073] GeMM-GAN: A Multimodal Generative Model Conditioned on Histopathology Images and Clinical Descriptions for Gene Expression Profile Generation
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于生成模型任务，旨在解决基因表达数据获取困难的问题。通过结合病理图像和临床描述，生成真实的基因表达谱，提升疾病类型预测准确率。**

- **链接: [https://arxiv.org/pdf/2601.15392v1](https://arxiv.org/pdf/2601.15392v1)**

> **作者:** Francesca Pia Panaccione; Carlo Sgaravatti; Pietro Pinoli
>
> **备注:** 12 pages, 2 figures. Published at Image Analysis and Processing - ICIAP 2025 Workshops
>
> **摘要:** Biomedical research increasingly relies on integrating diverse data modalities, including gene expression profiles, medical images, and clinical metadata. While medical images and clinical metadata are routinely collected in clinical practice, gene expression data presents unique challenges for widespread research use, mainly due to stringent privacy regulations and costly laboratory experiments. To address these limitations, we present GeMM-GAN, a novel Generative Adversarial Network conditioned on histopathology tissue slides and clinical metadata, designed to synthesize realistic gene expression profiles. GeMM-GAN combines a Transformer Encoder for image patches with a final Cross Attention mechanism between patches and text tokens, producing a conditioning vector to guide a generative model in generating biologically coherent gene expression profiles. We evaluate our approach on the TCGA dataset and demonstrate that our framework outperforms standard generative models and generates more realistic and functionally meaningful gene expression profiles, improving by more than 11\% the accuracy on downstream disease type prediction compared to current state-of-the-art generative models. Code will be available at: https://github.com/francescapia/GeMM-GAN
>
---
#### [new 074] Transfer Learning from ImageNet for MEG-Based Decoding of Imagined Speech
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于非侵入式想象语音解码任务，旨在解决信号弱、数据少的问题。通过将MEG信号转为图像输入，利用预训练视觉模型提升解码效果。**

- **链接: [https://arxiv.org/pdf/2601.15909v1](https://arxiv.org/pdf/2601.15909v1)**

> **作者:** Soufiane Jhilal; Stéphanie Martin; Anne-Lise Giraud
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** Non-invasive decoding of imagined speech remains challenging due to weak, distributed signals and limited labeled data. Our paper introduces an image-based approach that transforms magnetoencephalography (MEG) signals into time-frequency representations compatible with pretrained vision models. MEG data from 21 participants performing imagined speech tasks were projected into three spatial scalogram mixtures via a learnable sensor-space convolution, producing compact image-like inputs for ImageNet-pretrained vision architectures. These models outperformed classical and non-pretrained models, achieving up to 90.4% balanced accuracy for imagery vs. silence, 81.0% vs. silent reading, and 60.6% for vowel decoding. Cross-subject evaluation confirmed that pretrained models capture shared neural representations, and temporal analyses localized discriminative information to imagery-locked intervals. These findings show that pretrained vision models applied to image-based MEG representations can effectively capture the structure of imagined speech in non-invasive neural signals.
>
---
#### [new 075] FUGC: Benchmarking Semi-Supervised Learning Methods for Cervical Segmentation
- **分类: eess.IV; cs.CE; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决宫颈结构分割中标注数据不足的问题。通过构建FUGC基准，评估半监督学习方法的性能。**

- **链接: [https://arxiv.org/pdf/2601.15572v1](https://arxiv.org/pdf/2601.15572v1)**

> **作者:** Jieyun Bai; Yitong Tang; Zihao Zhou; Mahdi Islam; Musarrat Tabassum; Enrique Almar-Munoz; Hongyu Liu; Hui Meng; Nianjiang Lv; Bo Deng; Yu Chen; Zilun Peng; Yusong Xiao; Li Xiao; Nam-Khanh Tran; Dac-Phu Phan-Le; Hai-Dang Nguyen; Xiao Liu; Jiale Hu; Mingxu Huang; Jitao Liang; Chaolu Feng; Xuezhi Zhang; Lyuyang Tong; Bo Du; Ha-Hieu Pham; Thanh-Huy Nguyen; Min Xu; Juntao Jiang; Jiangning Zhang; Yong Liu; Md. Kamrul Hasan; Jie Gan; Zhuonan Liang; Weidong Cai; Yuxin Huang; Gongning Luo; Mohammad Yaqub; Karim Lekadir
>
> **摘要:** Accurate segmentation of cervical structures in transvaginal ultrasound (TVS) is critical for assessing the risk of spontaneous preterm birth (PTB), yet the scarcity of labeled data limits the performance of supervised learning approaches. This paper introduces the Fetal Ultrasound Grand Challenge (FUGC), the first benchmark for semi-supervised learning in cervical segmentation, hosted at ISBI 2025. FUGC provides a dataset of 890 TVS images, including 500 training images, 90 validation images, and 300 test images. Methods were evaluated using the Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), and runtime (RT), with a weighted combination of 0.4/0.4/0.2. The challenge attracted 10 teams with 82 participants submitting innovative solutions. The best-performing methods for each individual metric achieved 90.26\% mDSC, 38.88 mHD, and 32.85 ms RT, respectively. FUGC establishes a standardized benchmark for cervical segmentation, demonstrates the efficacy of semi-supervised methods with limited labeled data, and provides a foundation for AI-assisted clinical PTB risk assessment.
>
---
#### [new 076] DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DextER，解决语言驱动的灵巧抓取问题，通过具身推理生成抓取策略，提升抓取成功率和意图对齐。**

- **链接: [https://arxiv.org/pdf/2601.16046v1](https://arxiv.org/pdf/2601.16046v1)**

> **作者:** Junha Lee; Eunha Park; Minsu Cho
>
> **摘要:** Language-driven dexterous grasp generation requires the models to understand task semantics, 3D geometry, and complex hand-object interactions. While vision-language models have been applied to this problem, existing approaches directly map observations to grasp parameters without intermediate reasoning about physical interactions. We present DextER, Dexterous Grasp Generation with Embodied Reasoning, which introduces contact-based embodied reasoning for multi-finger manipulation. Our key insight is that predicting which hand links contact where on the object surface provides an embodiment-aware intermediate representation bridging task semantics with physical constraints. DextER autoregressively generates embodied contact tokens specifying which finger links contact where on the object surface, followed by grasp tokens encoding the hand configuration. On DexGYS, DextER achieves 67.14% success rate, outperforming state-of-the-art by 3.83%p with 96.4% improvement in intention alignment. We also demonstrate steerable generation through partial contact specification, providing fine-grained control over grasp synthesis.
>
---
#### [new 077] Neural Particle Automata: Learning Self-Organizing Particle Dynamics
- **分类: cs.NE; cs.CV**

- **简介: 该论文提出Neural Particle Automata（NPA），解决动态粒子系统建模问题。通过将神经元细胞自动机扩展为粒子系统，实现自组织动力学学习。**

- **链接: [https://arxiv.org/pdf/2601.16096v1](https://arxiv.org/pdf/2601.16096v1)**

> **作者:** Hyunsoo Kim; Ehsan Pajouheshgar; Sabine Süsstrunk; Wenzel Jakob; Jinah Park
>
> **备注:** 15 pages, 15 figures
>
> **摘要:** We introduce Neural Particle Automata (NPA), a Lagrangian generalization of Neural Cellular Automata (NCA) from static lattices to dynamic particle systems. Unlike classical Eulerian NCA where cells are pinned to pixels or voxels, NPA model each cell as a particle with a continuous position and internal state, both updated by a shared, learnable neural rule. This particle-based formulation yields clear individuation of cells, allows heterogeneous dynamics, and concentrates computation only on regions where activity is present. At the same time, particle systems pose challenges: neighborhoods are dynamic, and a naive implementation of local interactions scale quadratically with the number of particles. We address these challenges by replacing grid-based neighborhood perception with differentiable Smoothed Particle Hydrodynamics (SPH) operators backed by memory-efficient, CUDA-accelerated kernels, enabling scalable end-to-end training. Across tasks including morphogenesis, point-cloud classification, and particle-based texture synthesis, we show that NPA retain key NCA behaviors such as robustness and self-regeneration, while enabling new behaviors specific to particle systems. Together, these results position NPA as a compact neural model for learning self-organizing particle dynamics.
>
---
#### [new 078] Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决跨模态和细粒度目标定位问题。提出Phi-SegNet，结合相位信息提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.16064v1](https://arxiv.org/pdf/2601.16064v1)**

> **作者:** Shams Nafisa Ali; Taufiq Hasan
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Deep learning has substantially advanced medical image segmentation, yet achieving robust generalization across diverse imaging modalities and anatomical structures remains a major challenge. A key contributor to this limitation lies in how existing architectures, ranging from CNNs to Transformers and their hybrids, primarily encode spatial information while overlooking frequency-domain representations that capture rich structural and textural cues. Although few recent studies have begun exploring spectral information at the feature level, supervision-level integration of frequency cues-crucial for fine-grained object localization-remains largely untapped. To this end, we propose Phi-SegNet, a CNN-based architecture that incorporates phase-aware information at both architectural and optimization levels. The network integrates Bi-Feature Mask Former (BFMF) modules that blend neighboring encoder features to reduce semantic gaps, and Reverse Fourier Attention (RFA) blocks that refine decoder outputs using phase-regularized features. A dedicated phase-aware loss aligns these features with structural priors, forming a closed feedback loop that emphasizes boundary precision. Evaluated on five public datasets spanning X-ray, US, histopathology, MRI, and colonoscopy, Phi-SegNet consistently achieved state-of-the-art performance, with an average relative improvement of 1.54+/-1.26% in IoU and 0.98+/-0.71% in F1-score over the next best-performing model. In cross-dataset generalization scenarios involving unseen datasets from the known domain, Phi-SegNet also exhibits robust and superior performance, highlighting its adaptability and modality-agnostic design. These findings demonstrate the potential of leveraging spectral priors in both feature representation and supervision, paving the way for generalized segmentation frameworks that excel in fine-grained object localization.
>
---
#### [new 079] PF-D2M: A Pose-free Diffusion Model for Universal Dance-to-Music Generation
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于舞蹈到音乐生成任务，解决多舞者和非人类舞蹈场景下的音乐生成问题。提出PF-D2M模型，利用视频视觉特征和渐进训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2601.15872v1](https://arxiv.org/pdf/2601.15872v1)**

> **作者:** Jaekwon Im; Natalia Polouliakh; Taketo Akama
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Dance-to-music generation aims to generate music that is aligned with dance movements. Existing approaches typically rely on body motion features extracted from a single human dancer and limited dance-to-music datasets, which restrict their performance and applicability to real-world scenarios involving multiple dancers and non-human dancers. In this paper, we propose PF-D2M, a universal diffusion-based dance-to-music generation model that incorporates visual features extracted from dance videos. PF-D2M is trained with a progressive training strategy that effectively addresses data scarcity and generalization challenges. Both objective and subjective evaluations show that PF-D2M achieves state-of-the-art performance in dance-music alignment and music quality.
>
---
#### [new 080] High-Fidelity 3D Tooth Reconstruction by Fusing Intraoral Scans and CBCT Data via a Deep Implicit Representation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D牙齿重建任务，旨在融合CBCT和IOS数据，解决单一模态信息不全的问题。通过深度隐式表示实现高保真、无缝的牙齿模型生成。**

- **链接: [https://arxiv.org/pdf/2601.15358v1](https://arxiv.org/pdf/2601.15358v1)**

> **作者:** Yi Zhu; Razmig Kechichian; Raphaël Richert; Satoshi Ikehata; Sébastien Valette
>
> **备注:** Accepted to IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** High-fidelity 3D tooth models are essential for digital dentistry, but must capture both the detailed crown and the complete root. Clinical imaging modalities are limited: Cone-Beam Computed Tomography (CBCT) captures the root but has a noisy, low-resolution crown, while Intraoral Scanners (IOS) provide a high-fidelity crown but no root information. A naive fusion of these sources results in unnatural seams and artifacts. We propose a novel, fully-automated pipeline that fuses CBCT and IOS data using a deep implicit representation. Our method first segments and robustly registers the tooth instances, then creates a hybrid proxy mesh combining the IOS crown and the CBCT root. The core of our approach is to use this noisy proxy to guide a class-specific DeepSDF network. This optimization process projects the input onto a learned manifold of ideal tooth shapes, generating a seamless, watertight, and anatomically coherent model. Qualitative and quantitative evaluations show our method uniquely preserves both the high-fidelity crown from IOS and the patient-specific root morphology from CBCT, overcoming the limitations of each modality and naive stitching.
>
---
#### [new 081] Provable Robustness in Multimodal Large Language Models via Feature Space Smoothing
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型鲁棒性研究，旨在解决多模态大语言模型易受对抗攻击的问题。通过特征空间平滑和PSM模块提升模型的认证鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.16200v1](https://arxiv.org/pdf/2601.16200v1)**

> **作者:** Song Xia; Meiwen Ding; Chenqi Kong; Wenhan Yang; Xudong Jiang
>
> **备注:** Under review
>
> **摘要:** Multimodal large language models (MLLMs) exhibit strong capabilities across diverse applications, yet remain vulnerable to adversarial perturbations that distort their feature representations and induce erroneous predictions. To address this vulnerability, we propose the Feature-space Smoothing (FS) and theoretically prove that FS offers certified robustness on the feature representations of MLLMs. Specifically, FS transforms any feature encoder into a smoothed variant that is guaranteed to maintain a certified lower bound on the feature cosine similarity between clean and adversarial representations under $\ell_2$-bounded attacks. Moreover, we indicate that the value of this Feature Cosine Similarity Bound (FCSB) derived from FS can be improved by enlarging the defined Gaussian robustness score on the vanilla encoder. Building upon this, we introduce the Purifier and Smoothness Mapper (PSM), a plug-and-play module that improves the Gaussian robustness score of MLLMs and thus enhances their certified robustness under FS, without requiring any retraining on MLLMs. We demonstrate that the FS with PSM not only provides a strong theoretical robustness guarantee but also exhibits superior empirical performance compared to adversarial training. Extensive experiments across diverse MLLMs and downstream tasks indicate the effectiveness of the FS-PSM, reducing the Attack Success Rate (ASR) of various white-box attacks from nearly 90\% to about 1\%.
>
---
#### [new 082] The Paradigm Shift: A Comprehensive Survey on Large Vision Language Models for Multimodal Fake News Detection
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态虚假新闻检测任务，旨在解决传统方法在语义理解和跨模态交互上的不足。通过系统综述大视觉语言模型的应用，分析其技术挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2601.15316v1](https://arxiv.org/pdf/2601.15316v1)**

> **作者:** Wei Ai; Yilong Tan; Yuntao Shou; Tao Meng; Haowen Chen; Zhixiong He; Keqin Li
>
> **摘要:** In recent years, the rapid evolution of large vision-language models (LVLMs) has driven a paradigm shift in multimodal fake news detection (MFND), transforming it from traditional feature-engineering approaches to unified, end-to-end multimodal reasoning frameworks. Early methods primarily relied on shallow fusion techniques to capture correlations between text and images, but they struggled with high-level semantic understanding and complex cross-modal interactions. The emergence of LVLMs has fundamentally changed this landscape by enabling joint modeling of vision and language with powerful representation learning, thereby enhancing the ability to detect misinformation that leverages both textual narratives and visual content. Despite these advances, the field lacks a systematic survey that traces this transition and consolidates recent developments. To address this gap, this paper provides a comprehensive review of MFND through the lens of LVLMs. We first present a historical perspective, mapping the evolution from conventional multimodal detection pipelines to foundation model-driven paradigms. Next, we establish a structured taxonomy covering model architectures, datasets, and performance benchmarks. Furthermore, we analyze the remaining technical challenges, including interpretability, temporal reasoning, and domain generalization. Finally, we outline future research directions to guide the next stage of this paradigm shift. To the best of our knowledge, this is the first comprehensive survey to systematically document and analyze the transformative role of LVLMs in combating multimodal fake news. The summary of existing methods mentioned is in our Github: \href{https://github.com/Tan-YiLong/Overview-of-Fake-News-Detection}{https://github.com/Tan-YiLong/Overview-of-Fake-News-Detection}.
>
---
#### [new 083] Distillation-based Layer Dropping (DLD) Effective End-to-end Framework for Dynamic Speech Networks
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于语音识别任务，旨在解决动态网络中层剪枝导致的性能下降问题。提出DLD框架，结合知识蒸馏与层剪枝，提升动态语音网络性能。**

- **链接: [https://arxiv.org/pdf/2601.16117v1](https://arxiv.org/pdf/2601.16117v1)**

> **作者:** Abdul Hannan; Daniele Falavigna; Shah Nawaz; Mubashir Noman; Markus Schedl; Alessio Brutti
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Edge devices operate in constrained and varying resource settings, requiring dynamic architectures that can adapt to limitations of the available resources. To meet such demands, layer dropping ($\mathcal{LD}$) approach is typically used to transform static models into dynamic ones by skipping parts of the network along with reducing overall computational complexity. However, existing $\mathcal{LD}$ methods greatly impact the dynamic model's performance for low and high dropping cases, deteriorating the performance-computation trade-off. To this end, we propose a distillation-based layer dropping (DLD) framework that effectively combines the capabilities of knowledge distillation and $\mathcal{LD}$ in an end-to-end fashion, thereby achieving state-of-the-art performance for dynamic speech networks. Comprehensive experimentation utilizing well-known speech recognition methods, including conformer and WavLM, on three public benchmarks demonstrates the effectiveness of our framework, reducing the word error rate by $9.32\%$ and $2.25\%$ for high and no dropping cases with $33.3\%$ reduction in training time.
>
---
#### [new 084] CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出CASL框架，解决扩散模型中潜在表示与语义概念对齐的问题，通过监督学习实现更准确的语义控制和解释。**

- **链接: [https://arxiv.org/pdf/2601.15441v1](https://arxiv.org/pdf/2601.15441v1)**

> **作者:** Zhenghao He; Guangzhi Xiong; Boyang Wang; Sanchit Sinha; Aidong Zhang
>
> **摘要:** Internal activations of diffusion models encode rich semantic information, but interpreting such representations remains challenging. While Sparse Autoencoders (SAEs) have shown promise in disentangling latent representations, existing SAE-based methods for diffusion model understanding rely on unsupervised approaches that fail to align sparse features with human-understandable concepts. This limits their ability to provide reliable semantic control over generated images. We introduce CASL (Concept-Aligned Sparse Latents), a supervised framework that aligns sparse latent dimensions of diffusion models with semantic concepts. CASL first trains an SAE on frozen U-Net activations to obtain disentangled latent representations, and then learns a lightweight linear mapping that associates each concept with a small set of relevant latent dimensions. To validate the semantic meaning of these aligned directions, we propose CASL-Steer, a controlled latent intervention that shifts activations along the learned concept axis. Unlike editing methods, CASL-Steer is used solely as a causal probe to reveal how concept-aligned latents influence generated content. We further introduce the Editing Precision Ratio (EPR), a metric that jointly measures concept specificity and the preservation of unrelated attributes. Experiments show that our method achieves superior editing precision and interpretability compared to existing approaches. To the best of our knowledge, this is the first work to achieve supervised alignment between latent representations and semantic concepts in diffusion models.
>
---
## 更新

#### [replaced 001] Crafting Adversarial Inputs for Large Vision-Language Models Using Black-Box Optimization
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.01747v4](https://arxiv.org/pdf/2601.01747v4)**

> **作者:** Jiwei Guan; Haibo Jin; Haohan Wang
>
> **备注:** EACL
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have shown groundbreaking capabilities across diverse multimodal tasks. However, these models remain vulnerable to adversarial jailbreak attacks, where adversaries craft subtle perturbations to bypass safety mechanisms and trigger harmful outputs. Existing white-box attacks methods require full model accessibility, suffer from computing costs and exhibit insufficient adversarial transferability, making them impractical for real-world, black-box settings. To address these limitations, we propose a black-box jailbreak attack on LVLMs via Zeroth-Order optimization using Simultaneous Perturbation Stochastic Approximation (ZO-SPSA). ZO-SPSA provides three key advantages: (i) gradient-free approximation by input-output interactions without requiring model knowledge, (ii) model-agnostic optimization without the surrogate model and (iii) lower resource requirements with reduced GPU memory consumption. We evaluate ZO-SPSA on three LVLMs, including InstructBLIP, LLaVA and MiniGPT-4, achieving the highest jailbreak success rate of 83.0% on InstructBLIP, while maintaining imperceptible perturbations comparable to white-box methods. Moreover, adversarial examples generated from MiniGPT-4 exhibit strong transferability to other LVLMs, with ASR reaching 64.18%. These findings underscore the real-world feasibility of black-box jailbreaks and expose critical weaknesses in the safety mechanisms of current LVLMs
>
---
#### [replaced 002] Multi-event Video-Text Retrieval
- **分类: cs.CV; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2308.11551v3](https://arxiv.org/pdf/2308.11551v3)**

> **作者:** Gengyuan Zhang; Jisen Ren; Jindong Gu; Volker Tresp
>
> **备注:** [fixed typos in equations] accepted to ICCV2023 Poster; some figures are not supported when viewed online, please download the file and view locally
>
> **摘要:** Video-Text Retrieval (VTR) is a crucial multi-modal task in an era of massive video-text data on the Internet. A plethora of work characterized by using a two-stream Vision-Language model architecture that learns a joint representation of video-text pairs has become a prominent approach for the VTR task. However, these models operate under the assumption of bijective video-text correspondences and neglect a more practical scenario where video content usually encompasses multiple events, while texts like user queries or webpage metadata tend to be specific and correspond to single events. This establishes a gap between the previous training objective and real-world applications, leading to the potential performance degradation of earlier models during inference. In this study, we introduce the Multi-event Video-Text Retrieval (MeVTR) task, addressing scenarios in which each video contains multiple different events, as a niche scenario of the conventional Video-Text Retrieval Task. We present a simple model, Me-Retriever, which incorporates key event video representation and a new MeVTR loss for the MeVTR task. Comprehensive experiments show that this straightforward framework outperforms other models in the Video-to-Text and Text-to-Video tasks, effectively establishing a robust baseline for the MeVTR task. We believe this work serves as a strong foundation for future studies. Code is available at https://github.com/gengyuanmax/MeVTR.
>
---
#### [replaced 003] A Segmentation-driven Editing Method for Bolt Defect Augmentation and Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10509v3](https://arxiv.org/pdf/2508.10509v3)**

> **作者:** Yangjie Xiao; Ke Zhang; Jiacun Wang; Xin Sheng; Yurong Guo; Meijuan Chen; Zehua Ren; Zhaoye Zheng; Zhenbing Zhao
>
> **摘要:** Bolt defect detection is critical to ensure the safety of transmission lines. However, the scarcity of defect images and imbalanced data distributions significantly limit detection performance. To address this problem, we propose a segmentationdriven bolt defect editing method (SBDE) to augment the dataset. First, a bolt attribute segmentation model (Bolt-SAM) is proposed, which enhances the segmentation of complex bolt attributes through the CLAHE-FFT Adapter (CFA) and Multipart- Aware Mask Decoder (MAMD), generating high-quality masks for subsequent editing tasks. Second, a mask optimization module (MOD) is designed and integrated with the image inpainting model (LaMa) to construct the bolt defect attribute editing model (MOD-LaMa), which converts normal bolts into defective ones through attribute editing. Finally, an editing recovery augmentation (ERA) strategy is proposed to recover and put the edited defect bolts back into the original inspection scenes and expand the defect detection dataset. We constructed multiple bolt datasets and conducted extensive experiments. Experimental results demonstrate that the bolt defect images generated by SBDE significantly outperform state-of-the-art image editing models, and effectively improve the performance of bolt defect detection, which fully verifies the effectiveness and application potential of the proposed method. The code of the project is available at https://github.com/Jay-xyj/SBDE.
>
---
#### [replaced 004] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11393v2](https://arxiv.org/pdf/2601.11393v2)**

> **作者:** Haomiao Tang; Jinpeng Wang; Minyi Zhao; Guanghao Meng; Ruisheng Luo; Long Chen; Shu-Tao Xia
>
> **备注:** Accepted for publication and oral presentation at AAAI 2026
>
> **摘要:** Composed Image Retrieval (CIR) enables image search by combining a reference image with modification text. Intrinsic noise in CIR triplets incurs intrinsic uncertainty and threatens the model's robustness. Probabilistic learning approaches have shown promise in addressing such issues; however, they fall short for CIR due to their instance-level holistic modeling and homogeneous treatment of queries and targets. This paper introduces a Heterogeneous Uncertainty-Guided (HUG) paradigm to overcome these limitations. HUG utilizes a fine-grained probabilistic learning framework, where queries and targets are represented by Gaussian embeddings that capture detailed concepts and uncertainties. We customize heterogeneous uncertainty estimations for multi-modal queries and uni-modal targets. Given a query, we capture uncertainties not only regarding uni-modal content quality but also multi-modal coordination, followed by a provable dynamic weighting mechanism to derive comprehensive query uncertainty. We further design uncertainty-guided objectives, including query-target holistic contrast and fine-grained contrasts with comprehensive negative sampling strategies, which effectively enhance discriminative learning. Experiments on benchmarks demonstrate HUG's effectiveness beyond state-of-the-art baselines, with faithful analysis justifying the technical contributions.
>
---
#### [replaced 005] YOLO Meets Mixture-of-Experts: Adaptive Expert Routing for Robust Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13344v4](https://arxiv.org/pdf/2511.13344v4)**

> **作者:** Ori Meiraz; Sharon Shalev; Avishai Weizman
>
> **备注:** 1 figure, 1 table, Accepted to ICSEE 2026
>
> **摘要:** This paper presents a novel Mixture-of-Experts framework for object detection, incorporating adaptive routing among multiple YOLOv9-T experts to enable dynamic feature specialization and achieve higher mean Average Precision (mAP) and Average Recall (AR) compared to a single YOLOv9-T model.
>
---
#### [replaced 006] Skywork UniPic 2.0: Building Kontext Model with Online RL for Unified Multimodal Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.04548v2](https://arxiv.org/pdf/2509.04548v2)**

> **作者:** Hongyang Wei; Baixin Xu; Hongbo Liu; Size Wu; Jie Liu; Yi Peng; Peiyu Wang; Zexiang Liu; Jingwen He; Yidan Xietian; Chuanxin Tang; Zidong Wang; Yichen Wei; Liang Hu; Boyi Jiang; Wei Li; Ying He; Yang Liu; Xuchen Song; Yangguang Li; Yahui Zhou
>
> **摘要:** Recent advances in multimodal models have demonstrated impressive capabilities in unified image generation and editing. However, many prominent open-source models prioritize scaling model parameters over optimizing training strategies, limiting their efficiency and performance. In this work, we present UniPic2-SD3.5M-Kontext, a 2B-parameter DiT model based on SD3.5-Medium, which achieves state-of-the-art image generation and editing while extending seamlessly into a unified multimodal framework. Our approach begins with architectural modifications to SD3.5-Medium and large-scale pre-training on high-quality data, enabling joint text-to-image generation and editing capabilities. To enhance instruction following and editing consistency, we propose a novel Progressive Dual-Task Reinforcement strategy (PDTR), which effectively strengthens both tasks in a staged manner. We empirically validate that the reinforcement phases for different tasks are mutually beneficial and do not induce negative interference. After pre-training and reinforcement strategies, UniPic2-SD3.5M-Kontext demonstrates stronger image generation and editing capabilities than models with significantly larger generation parameters-including BAGEL (7B) and Flux-Kontext (12B). Furthermore, following the MetaQuery, we connect the UniPic2-SD3.5M-Kontext and Qwen2.5-VL-7B via a connector and perform joint training to launch a unified multimodal model UniPic2-Metaquery. UniPic2-Metaquery integrates understanding, generation, and editing, achieving top-tier performance across diverse tasks with a simple and scalable training paradigm. This consistently validates the effectiveness and generalizability of our proposed training paradigm, which we formalize as Skywork UniPic 2.0.
>
---
#### [replaced 007] Auditing and Mitigating Bias in Gender Classification Algorithms: A Data-Centric Approach
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.17873v2](https://arxiv.org/pdf/2510.17873v2)**

> **作者:** Tadesse K Bahiru; Natnael Tilahun Sinshaw; Teshager Hailemariam Moges; Dheeraj Kumar Singh
>
> **备注:** The manuscript contains a substantive error identified after submission
>
> **摘要:** Gender classification systems often inherit and amplify demographic imbalances in their training data. We first audit five widely used gender classification datasets, revealing that all suffer from significant intersectional underrepresentation. To measure the downstream impact of these flaws, we train identical MobileNetV2 classifiers on the two most balanced of these datasets, UTKFace and FairFace. Our fairness evaluation shows that even these models exhibit significant bias, misclassifying female faces at a higher rate than male faces and amplifying existing racial skew. To counter these data-induced biases, we construct BalancedFace, a new public dataset created by blending images from FairFace and UTKFace, supplemented with images from other collections to fill missing demographic gaps. It is engineered to equalize subgroup shares across 189 intersections of age, race, and gender using only real, unedited images. When a standard classifier is trained on BalancedFace, it reduces the maximum True Positive Rate gap across racial subgroups by over 50% and brings the average Disparate Impact score 63% closer to the ideal of 1.0 compared to the next-best dataset, all with a minimal loss of overall accuracy. These results underline the profound value of data-centric interventions and provide an openly available resource for fair gender classification research.
>
---
#### [replaced 008] Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12787v3](https://arxiv.org/pdf/2506.12787v3)**

> **作者:** Mufan Liu; Cixiao Zhang; Qi Yang; Yujie Cao; Yiling Xu; Yin Xu; Shu Sun; Mingzeng Dai; Yunfeng Guan
>
> **摘要:** Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. The project page is https://evan-sudo.github.io/swiftwrf/.
>
---
#### [replaced 009] Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.11109v2](https://arxiv.org/pdf/2601.11109v2)**

> **作者:** Shaofeng Yin; Jiaxin Ge; Zora Zhiruo Wang; Xiuyu Li; Michael J. Black; Trevor Darrell; Angjoo Kanazawa; Haiwen Feng
>
> **备注:** Project page: https://fugtemypt123.github.io/VIGA-website/
>
> **摘要:** Vision-as-inverse-graphics, the concept of reconstructing an image as an editable graphics program is a long-standing goal of computer vision. Yet even strong VLMs aren't able to achieve this in one-shot as they lack fine-grained spatial and physical grounding capability. Our key insight is that closing this gap requires interleaved multimodal reasoning through iterative execution and verification. Stemming from this, we present VIGA (Vision-as-Inverse-Graphic Agent) that starts from an empty world and reconstructs or edits scenes through a closed-loop write-run-render-compare-revise procedure. To support long-horizon reasoning, VIGA combines (i) a skill library that alternates generator and verifier roles and (ii) an evolving context memory that contains plans, code diffs, and render history. VIGA is task-agnostic as it doesn't require auxiliary modules, covering a wide range of tasks such as 3D reconstruction, multi-step scene editing, 4D physical interaction, and 2D document editing, etc. Empirically, we found VIGA substantially improves one-shot baselines on BlenderGym (35.32%) and SlideBench (117.17%). Moreover, VIGA is also model-agnostic as it doesn't require finetuning, enabling a unified protocol to evaluate heterogeneous foundation VLMs. To better support this protocol, we introduce BlenderBench, a challenging benchmark that stress-tests interleaved multimodal reasoning with graphics engine, where VIGA improves by 124.70%.
>
---
#### [replaced 010] Emergence and Evolution of Interpretable Concepts in Diffusion Models
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2504.15473v2](https://arxiv.org/pdf/2504.15473v2)**

> **作者:** Berk Tinaz; Zalan Fabian; Mahdi Soltanolkotabi
>
> **备注:** 32 pages, 32 figures, published at the 39th Conference on Neural Information Processing Systems (NeurIPS), 2025
>
> **摘要:** Diffusion models have become the go-to method for text-to-image generation, producing high-quality images from pure noise. However, the inner workings of diffusion models is still largely a mystery due to their black-box nature and complex, multi-step generation process. Mechanistic interpretability techniques, such as Sparse Autoencoders (SAEs), have been successful in understanding and steering the behavior of large language models at scale. However, the great potential of SAEs has not yet been applied toward gaining insight into the intricate generative process of diffusion models. In this work, we leverage the SAE framework to probe the inner workings of a popular text-to-image diffusion model, and uncover a variety of human-interpretable concepts in its activations. Interestingly, we find that even before the first reverse diffusion step is completed, the final composition of the scene can be predicted surprisingly well by looking at the spatial distribution of activated concepts. Moreover, going beyond correlational analysis, we design intervention techniques aimed at manipulating image composition and style, and demonstrate that (1) in early stages of diffusion image composition can be effectively controlled, (2) in the middle stages image composition is finalized, however stylistic interventions are effective, and (3) in the final stages only minor textural details are subject to change.
>
---
#### [replaced 011] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决动态环境中视觉驱动的协作问题。提出VIKI-Bench基准和VIKI-R框架，提升多机器人协作性能。**

- **链接: [https://arxiv.org/pdf/2506.09049v3](https://arxiv.org/pdf/2506.09049v3)**

> **作者:** Li Kang; Xiufeng Song; Heng Zhou; Yiran Qin; Jie Yang; Xiaohong Liu; Philip Torr; Lei Bai; Zhenfei Yin
>
> **备注:** Accepted by NeurIPS 2025 Track on Datasets and Benchmarks. Project page: https://faceong.github.io/VIKI-R/
>
> **摘要:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
>
---
#### [replaced 012] Yesnt: Are Diffusion Relighting Models Ready for Capture Stage Compositing? A Hybrid Alternative to Bridge the Gap
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2510.23494v2](https://arxiv.org/pdf/2510.23494v2)**

> **作者:** Elisabeth Jüttner; Janelle Pfeifer; Leona Krath; Stefan Korfhage; Hannah Dröge; Matthias B. Hullin; Markus Plack
>
> **摘要:** Volumetric video relighting is essential for bringing captured performances into virtual worlds, but current approaches struggle to deliver temporally stable, production-ready results. Diffusion-based intrinsic decomposition methods show promise for single frames, yet suffer from stochastic noise and instability when extended to sequences, while video diffusion models remain constrained by memory and scale. We propose a hybrid relighting framework that combines diffusion-derived material priors with temporal regularization and physically motivated rendering. Our method aggregates multiple stochastic estimates of per-frame material properties into temporally consistent shading components, using optical-flow-guided regularization. For indirect effects such as shadows and reflections, we extract a mesh proxy from Gaussian Opacity Fields and render it within a standard graphics pipeline. Experiments on real and synthetic captures show that this hybrid strategy achieves substantially more stable relighting across sequences than diffusion-only baselines, while scaling beyond the clip lengths feasible for video diffusion. These results indicate that hybrid approaches, which balance learned priors with physically grounded constraints, are a practical step toward production-ready volumetric video relighting.
>
---
#### [replaced 013] Efficient Multimodal Large Language Models: A Survey
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2405.10739v3](https://arxiv.org/pdf/2405.10739v3)**

> **作者:** Yizhang Jin; Jian Li; Yexin Liu; Tianjun Gu; Kai Wu; Zhengkai Jiang; Muyang He; Bo Zhao; Xin Tan; Zhenye Gan; Yabiao Wang; Chengjie Wang; Lizhuang Ma
>
> **备注:** Accepted by Visual Intelligence
>
> **摘要:** In the past year, Multimodal Large Language Models (MLLMs) have demonstrated remarkable performance in tasks such as visual question answering, visual understanding and reasoning. However, the extensive model size and high training and inference costs have hindered the widespread application of MLLMs in academia and industry. Thus, studying efficient and lightweight MLLMs has enormous potential, especially in edge computing scenarios. In this survey, we provide a comprehensive and systematic review of the current state of efficient MLLMs. Specifically, we summarize the timeline of representative efficient MLLMs, research state of efficient structures and strategies, and the applications. Finally, we discuss the limitations of current efficient MLLM research and promising future directions. Please refer to our GitHub repository for more details: https://github.com/lijiannuist/Efficient-Multimodal-LLMs-Survey.
>
---
#### [replaced 014] VideoPro: Adaptive Program Reasoning for Long Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17743v3](https://arxiv.org/pdf/2509.17743v3)**

> **作者:** Chenglin Li; Feng Han; Yikun Wang; Ruilin Li; Shuai Dong; Haowen Hou; Haitao Li; Qianglong Chen; Feng Tao; Jingqi Tong; Yin Zhang; Jiaqi Wang
>
> **摘要:** Large language models (LLMs) have shown promise in generating program workflows for visual tasks. However, previous approaches often rely on closed-source models, lack systematic reasoning, and struggle with long-form video question answering (videoQA). To address these challenges, we introduce the FS-VisPR framework, an adaptive visual program reasoning approach that balances fast reasoning for simple queries with slow reasoning for difficult ones. First, we design efficient visual modules (e.g., key clip retrieval and subtitle retrieval) to support long-form video tasks. Then, we construct a diverse and high-quality fast-slow reasoning dataset with a strong LLM to align open-source language models' ability to generate visual program workflows as FS-LLM. Next, we design a fast-slow reasoning framework with FS-LLM: Simple queries are directly solved by VideoLLMs, while difficult ones invoke visual program reasoning, motivated by human-like reasoning processes. During this process, low-confidence fast-thinking answers will trigger a second-stage slow-reasoning process, and a fallback mechanism to fast reasoning is activated if the program execution fails. Moreover, we improve visual programs through parameter search during both training and inference. By adjusting the parameters of the visual modules within the program, multiple variants are generated: during training, programs that yield correct answers are selected, while during inference, the program with the highest confidence result is applied. Experiments show that FS-VisPR improves both efficiency and reliability in visual program workflows. It achieves 50.4% accuracy on LVBench, surpassing GPT-4o, matching the performance of Qwen2.5VL-72B on VideoMME.
>
---
#### [replaced 015] An Efficient Quality Metric for Video Frame Interpolation Based on Motion-Field Divergence
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.01361v2](https://arxiv.org/pdf/2510.01361v2)**

> **作者:** Conall Daly; Darren Ramsook; Anil Kokaram
>
> **备注:** IEEE 17th International Conference on Quality of Multimedia Experience 2025 accepted manuscript, 7 pages
>
> **摘要:** Video frame interpolation is a fundamental tool for temporal video enhancement, but existing quality metrics struggle to evaluate the perceptual impact of interpolation artefacts effectively. Metrics like PSNR, SSIM and LPIPS ignore temporal coherence. State-of-the-art quality metrics tailored towards video frame interpolation, like FloLPIPS, have been developed but suffer from computational inefficiency that limits their practical application. We present $\text{PSNR}_{\text{DIV}}$, a novel full-reference quality metric that enhances PSNR through motion divergence weighting, a technique adapted from archival film restoration where it was developed to detect temporal inconsistencies. Our approach highlights singularities in motion fields which is then used to weight image errors. Evaluation on the BVI-VFI dataset (180 sequences across multiple frame rates, resolutions and interpolation methods) shows $\text{PSNR}_{\text{DIV}}$ achieves statistically significant improvements: +0.09 Pearson Linear Correlation Coefficient over FloLPIPS, while being 2.5$\times$ faster and using 4$\times$ less memory. Performance remains consistent across all content categories and are robust to the motion estimator used. The efficiency and accuracy of $\text{PSNR}_{\text{DIV}}$ enables fast quality evaluation and practical use as a loss function for training neural networks for video frame interpolation tasks. An implementation of our metric is available at www.github.com/conalld/psnr-div.
>
---
#### [replaced 016] Multi-View Projection for Unsupervised Domain Adaptation in 3D Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15545v3](https://arxiv.org/pdf/2505.15545v3)**

> **作者:** Andrew Caunes; Thierry Chateau; Vincent Fremont
>
> **摘要:** 3D semantic segmentation plays a pivotal role in autonomous driving and road infrastructure analysis, yet state-of-the-art 3D models are prone to severe domain shift when deployed across different datasets. In this paper, we propose an Unsupervised Domain Adaptation approach where a 3D segmentation model is trained on the target dataset using pseudo-labels generated by a novel multi-view projection framework. Our approach first aligns Lidar scans into coherent 3D scenes and renders them from multiple virtual camera poses to create large-scale synthetic 2D semantic segmentation datasets in various modalities. The generated datasets are used to train an ensemble of 2D segmentation models in point cloud view domain on each modality. During inference, the models process a large amount of views per scene; the resulting logits are back-projected to 3D with a depth-aware voting scheme to generate final point-wise labels. These labels are then used to fine-tune a 3D segmentation model in the target domain. We evaluate our approach Real-to-Real on the nuScenes and SemanticKITTI datasets. We also evaluate it Simulation-to-Real with the SynLidar dataset. Our contributions are a novel method that achieves state-of-the-art results in Real-to-Real Unsupervised Domain Adaptation, and we also demonstrate an application of our method to segment rare classes, for which target 3D annotations are not available, by only using 2D annotations for those classes and leveraging 3D annotations for other classes in a source domain.
>
---
#### [replaced 017] MetaDCSeg: Robust Medical Image Segmentation via Meta Dynamic Center Weighting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18894v2](https://arxiv.org/pdf/2511.18894v2)**

> **作者:** Chenyu Mu; Guihai Chen; Xun Yang; Erkun Yang; Cheng Deng
>
> **摘要:** Medical image segmentation is crucial for clinical applications, but it is frequently disrupted by noisy annotations and ambiguous anatomical boundaries, which lead to instability in model training. Existing methods typically rely on global noise assumptions or confidence-based sample selection, which inadequately mitigate the performance degradation caused by annotation noise, especially in challenging boundary regions. To address this issue, we propose MetaDCSeg, a robust framework that dynamically learns optimal pixel-wise weights to suppress the influence of noisy ground-truth labels while preserving reliable annotations. By explicitly modeling boundary uncertainty through a Dynamic Center Distance (DCD) mechanism, our approach utilizes weighted feature distances for foreground, background, and boundary centers, directing the model's attention toward hard-to-segment pixels near ambiguous boundaries. This strategy enables more precise handling of structural boundaries, which are often overlooked by existing methods, and significantly enhances segmentation performance. Extensive experiments across four benchmark datasets with varying noise levels demonstrate that MetaDCSeg consistently outperforms existing state-of-the-art methods.
>
---
#### [replaced 018] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.20879v4](https://arxiv.org/pdf/2506.20879v4)**

> **作者:** Shubhankar Borse; Seokeon Choi; Sunghyun Park; Jeongho Kim; Shreya Kadambi; Risheek Garrepalli; Sungrack Yun; Munawar Hayat; Fatih Porikli
>
> **备注:** Accepted at the NeurIPS 2025 D&B Track
>
> **摘要:** Generation of images containing multiple humans, performing complex actions, while preserving their facial identities, is a significant challenge. A major factor contributing to this is the lack of a dedicated benchmark. To address this, we introduce MultiHuman-Testbench, a novel benchmark for rigorously evaluating generative models for multi-human generation. The benchmark comprises 1,800 samples, including carefully curated text prompts, describing a range of simple to complex human actions. These prompts are matched with a total of 5,550 unique human face images, sampled uniformly to ensure diversity across age, ethnic background, and gender. Alongside captions, we provide human-selected pose conditioning images which accurately match the prompt. We propose a multi-faceted evaluation suite employing four key metrics to quantify face count, ID similarity, prompt alignment, and action detection. We conduct a thorough evaluation of a diverse set of models, including zero-shot approaches and training-based methods, with and without regional priors. We also propose novel techniques to incorporate image and region isolation using human segmentation and Hungarian matching, significantly improving ID similarity. Our proposed benchmark and key findings provide valuable insights and a standardized tool for advancing research in multi-human image generation. The dataset and evaluation codes will be available at https://github.com/Qualcomm-AI-research/MultiHuman-Testbench.
>
---
#### [replaced 019] BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中的泛化问题。通过引入贝叶斯分解和潜在动作查询，提升语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2601.15197v2](https://arxiv.org/pdf/2601.15197v2)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 020] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出RoT框架，将文本推理过程转化为图像，解决LLM推理过程不透明问题，提升推理效率与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.14750v2](https://arxiv.org/pdf/2601.14750v2)**

> **作者:** Yifan Wang; Shiyu Li; Peiming Li; Xiaochen Yang; Yang Tang; Zheng Wei
>
> **摘要:** Chain-of-Thought (CoT) prompting has achieved remarkable success in unlocking the reasoning capabilities of Large Language Models (LLMs). Although CoT prompting enhances reasoning, its verbosity imposes substantial computational overhead. Recent works often focus exclusively on outcome alignment and lack supervision on the intermediate reasoning process. These deficiencies obscure the analyzability of the latent reasoning chain. To address these challenges, we introduce Render-of-Thought (RoT), the first framework to reify the reasoning chain by rendering textual steps into images, making the latent rationale explicit and traceable. Specifically, we leverage the vision encoders of existing Vision Language Models (VLMs) as semantic anchors to align the vision embeddings with the textual space. This design ensures plug-and-play implementation without incurring additional pre-training overhead. Extensive experiments on mathematical and logical reasoning benchmarks demonstrate that our method achieves 3-4x token compression and substantial inference acceleration compared to explicit CoT. Furthermore, it maintains competitive performance against other methods, validating the feasibility of this paradigm. Our code is available at https://github.com/TencentBAC/RoT
>
---
#### [replaced 021] From Text to Image: Exploring GPT-4Vision's Potential in Advanced Radiological Analysis across Subspecialties
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2311.14777v2](https://arxiv.org/pdf/2311.14777v2)**

> **作者:** Felix Busch; Tianyu Han; Marcus Makowski; Daniel Truhn; Keno Bressem; Lisa Adams
>
> **摘要:** The study evaluates and compares GPT-4 and GPT-4Vision for radiological tasks, suggesting GPT-4Vision may recognize radiological features from images, thereby enhancing its diagnostic potential over text-based descriptions.
>
---
#### [replaced 022] PatchEAD: Unifying Industrial Visual Prompting Frameworks for Patch-Exclusive Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25856v2](https://arxiv.org/pdf/2509.25856v2)**

> **作者:** Po-Han Huang; Jeng-Lin Li; Po-Hsuan Huang; Ming-Ching Chang; Wei-Chao Chen
>
> **备注:** 10 pages, 5 figures. WACV 2026 (Accepted)
>
> **摘要:** Industrial anomaly detection is increasingly relying on foundation models, aiming for strong out-of-distribution generalization and rapid adaptation in real-world deployments. Notably, past studies have primarily focused on textual prompt tuning, leaving the intrinsic visual counterpart fragmented into processing steps specific to each foundation model. We aim to address this limitation by proposing a unified patch-focused framework, Patch-Exclusive Anomaly Detection (PatchEAD), enabling training-free anomaly detection that is compatible with diverse foundation models. The framework constructs visual prompting techniques, including an alignment module and foreground masking. Our experiments show superior few-shot and batch zero-shot performance compared to prior work, despite the absence of textual features. Our study further examines how backbone structure and pretrained characteristics affect patch-similarity robustness, providing actionable guidance for selecting and configuring foundation models for real-world visual inspection. These results confirm that a well-unified patch-only framework can enable quick, calibration-light deployment without the need for carefully engineered textual prompts.
>
---
#### [replaced 023] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10945v2](https://arxiv.org/pdf/2511.10945v2)**

> **作者:** Xingyue Zhao; Wenke Huang; Xingguang Wang; Haoyu Zhao; Linghao Zhuang; Anwen Jiang; Guancheng Wan; Mang Ye
>
> **备注:** Accepted at AAAI-26
>
> **摘要:** Federated learning enables multiple medical institutions to train a global model without sharing data, yet feature heterogeneity from diverse scanners or protocols remains a major challenge. Many existing works attempt to address this issue by leveraging model representations (e.g., mean feature vectors) to correct local training; however, they often face two key limitations: 1) Incomplete Contextual Representation Learning: Current approaches primarily focus on final-layer features, overlooking critical multi-level cues and thus diluting essential context for accurate segmentation. 2) Layerwise Style Bias Accumulation: Although utilizing representations can partially align global features, these methods neglect domain-specific biases within intermediate layers, allowing style discrepancies to build up and reduce model robustness. To address these challenges, we propose FedBCS to bridge feature representation gaps via domain-invariant contextual prototypes alignment. Specifically, we introduce a frequency-domain adaptive style recalibration into prototype construction that not only decouples content-style representations but also learns optimal style parameters, enabling more robust domain-invariant prototypes. Furthermore, we design a context-aware dual-level prototype alignment method that extracts domain-invariant prototypes from different layers of both encoder and decoder and fuses them with contextual information for finer-grained representation alignment. Extensive experiments on two public datasets demonstrate that our method exhibits remarkable performance.
>
---
#### [replaced 024] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.19328v3](https://arxiv.org/pdf/2505.19328v3)**

> **作者:** Manuela González-González; Soufiane Belharbi; Muhammad Osama Zeeshan; Masoumeh Sharafi; Muhammad Haseeb Aslam; Marco Pedersoli; Alessandro Lameiras Koerich; Simon L Bacon; Eric Granger
>
> **备注:** 45 pages, 21 figures, under review
>
> **摘要:** Ambivalence and hesitancy (A/H), a closely related construct, is the primary reasons why individuals delay, avoid, or abandon health behaviour changes. It is a subtle and conflicting emotion that sets a person in a state between positive and negative orientations, or between acceptance and refusal to do something. It manifests by a discord in affect between multiple modalities or within a modality, such as facial and vocal expressions, and body language. Although experts can be trained to recognize A/H as done for in-person interactions, integrating them into digital health interventions is costly and less effective. Automatic A/H recognition is therefore critical for the personalization and cost-effectiveness of digital behaviour change interventions. However, no datasets currently exists for the design of machine learning models to recognize A/H. This paper introduces the Behavioural Ambivalence/Hesitancy (BAH) dataset collected for multimodal recognition of A/H in videos. It contains 1,427 videos with a total duration of 10.60 hours captured from 300 participants across Canada answering predefined questions to elicit A/H. It is intended to mirror real-world online personalized behaviour change interventions. BAH is annotated by three experts to provide timestamps that indicate where A/H occurs, and frame- and video-level annotations with A/H cues. Video transcripts, cropped and aligned faces, and participants' meta-data are also provided. Since A and H manifest similarly in practice, we provide a binary annotation indicating the presence or absence of A/H. Additionally, this paper includes benchmarking results using baseline models on BAH for frame- and video-level recognition, zero-shot prediction, and personalization using source-free domain adaptation. The data, code, and pretrained weights are available.
>
---
#### [replaced 025] DECOR: Deep Embedding Clustering with Orientation Robustness
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.03328v2](https://arxiv.org/pdf/2510.03328v2)**

> **作者:** Fiona Victoria Stanley Jothiraj; Arunaggiri Pandian Karunanidhi; Seth A. Eichmeyer
>
> **备注:** Accepted to the KGML Bridge at AAAI 2026 (non-archival)
>
> **摘要:** In semiconductor manufacturing, early detection of wafer defects is critical for product yield optimization. However, raw wafer data from wafer quality tests are often complex, unlabeled, imbalanced and can contain multiple defects on a single wafer, making it crucial to design clustering methods that remain reliable under such imperfect data conditions. We introduce DECOR, a deep clustering with orientation robustness framework that groups complex defect patterns from wafer maps into consistent clusters. We evaluate our method on the open source MixedWM38 dataset, demonstrating its ability to discover clusters without manual tuning. DECOR explicitly accounts for orientation variations in wafer maps, ensuring that spatially similar defects are consistently clustered regardless of its rotation or alignment. Experiments indicate that our method outperforms existing clustering baseline methods, thus providing a reliable and scalable solution in automated visual inspection systems.
>
---
#### [replaced 026] The Percept-V Challenge: Can Multimodal LLMs Crack Simple Perception Problems?
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉感知任务，旨在评估多模态大语言模型在基础感知能力上的表现。研究构建了Percept-V数据集，发现模型在简单感知任务中表现不佳。**

- **链接: [https://arxiv.org/pdf/2508.21143v3](https://arxiv.org/pdf/2508.21143v3)**

> **作者:** Samrajnee Ghosh; Naman Agarwal; Hemanshu Garg; Chinmay Mittal; Mausam; Parag Singla
>
> **摘要:** Cognitive science research treats visual perception, the ability to understand and make sense of a visual input, as one of the early developmental signs of intelligence. Its TVPS-4 framework categorizes and tests human perception into seven skills such as visual discrimination, and form constancy. Do Multimodal Large Language Models (MLLMs) match up to humans in basic perception? Even though there are many benchmarks that evaluate MLLMs on advanced reasoning and knowledge skills, there is limited research that focuses evaluation on simple perception. In response, we introduce Percept-V, a dataset containing 6000 program-generated uncontaminated images divided into 30 domains, where each domain tests one or more TVPS-4 skills. Our focus is on perception, so we make our domains quite simple and the reasoning and knowledge required for solving them are minimal. Since modern-day MLLMs can solve much more complex tasks, our a-priori expectation is that they will solve these domains very easily. Contrary to our belief, our experiments show a weak performance of SoTA proprietary and open-source MLLMs compared to very high human performance on Percept-V. We find that as number of objects in the image increases, performance goes down rather fast. Our experiments also identify the perception skills that are considerably harder for all models.
>
---
#### [replaced 027] Language-guided Medical Image Segmentation with Target-informed Multi-level Contrastive Alignments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.13533v3](https://arxiv.org/pdf/2412.13533v3)**

> **作者:** Mingjian Li; Mingyuan Meng; Shuchang Ye; Michael Fulham; Lei Bi; Jinman Kim
>
> **摘要:** Medical image segmentation is a fundamental task in numerous medical engineering applications. Recently, language-guided segmentation has shown promise in medical scenarios where textual clinical reports are readily available as semantic guidance. Clinical reports contain diagnostic information provided by clinicians, which can provide auxiliary textual semantics to guide segmentation. However, existing language-guided segmentation methods neglect the inherent pattern gaps between image and text modalities, resulting in sub-optimal visual-language integration. Contrastive learning is a well-recognized approach to align image-text patterns, but it has not been optimized for bridging the pattern gaps in medical language-guided segmentation that relies primarily on medical image details to characterize the underlying disease/targets. Current contrastive alignment techniques typically align high-level global semantics without involving low-level localized target information, and thus cannot deliver fine-grained textual guidance on crucial image details. In this study, we propose a Target-informed Multi-level Contrastive Alignment framework (TMCA) to bridge image-text pattern gaps for medical language-guided segmentation. TMCA enables target-informed image-text alignments and fine-grained textual guidance by introducing: (i) a target-sensitive semantic distance module that utilizes target information for more granular image-text alignment modeling, (ii) a multi-level contrastive alignment strategy that directs fine-grained textual guidance to multi-scale image details, and (iii) a language-guided target enhancement module that reinforces attention to critical image regions based on the aligned image-text patterns. Extensive experiments on four public benchmark datasets demonstrate that TMCA enabled superior performance over state-of-the-art language-guided medical image segmentation methods.
>
---
#### [replaced 028] OccLE: Label-Efficient 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20617v4](https://arxiv.org/pdf/2505.20617v4)**

> **作者:** Naiyu Fang; Zheyuan Zhou; Fayao Liu; Xulei Yang; Jiacheng Wei; Lemiao Qiu; Hongsheng Li; Guosheng Lin
>
> **摘要:** 3D semantic occupancy prediction offers an intuitive and efficient scene understanding and has attracted significant interest in autonomous driving perception. Existing approaches either rely on full supervision, which demands costly voxel-level annotations, or on self-supervision, which provides limited guidance and yields suboptimal performance. To address these challenges, we propose OccLE, a Label-Efficient 3D Semantic Occupancy Prediction that takes images and LiDAR as inputs and maintains high performance with limited voxel annotations. Our intuition is to decouple the semantic and geometric learning tasks and then fuse the learned feature grids from both tasks for the final semantic occupancy prediction. Therefore, the semantic branch distills 2D foundation model to provide aligned pseudo labels for 2D and 3D semantic learning. The geometric branch integrates image and LiDAR inputs in cross-plane synergy based on their inherency, employing semi-supervision to enhance geometry learning. We fuse semantic-geometric feature grids through Dual Mamba and incorporate a scatter-accumulated projection to supervise unannotated prediction with aligned pseudo labels. Experiments show that OccLE achieves competitive performance with only 10\% of voxel annotations on the SemanticKITTI and Occ3D-nuScenes datasets. The code will be publicly released on https://github.com/NerdFNY/OccLE
>
---
#### [replaced 029] CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17590v3](https://arxiv.org/pdf/2505.17590v3)**

> **作者:** Florian Barthel; Wieland Morgenstern; Paul Hinzer; Anna Hilsmann; Peter Eisert
>
> **备注:** Main paper 12 pages, supplementary materials 8 pages
>
> **摘要:** Recently, 3D GANs based on 3D Gaussian splatting have been proposed for high quality synthesis of human heads. However, existing methods stabilize training and enhance rendering quality from steep viewpoints by conditioning the random latent vector on the current camera position. This compromises 3D consistency, as we observe significant identity changes when re-synthesizing the 3D head with each camera shift. Conversely, fixing the camera to a single viewpoint yields high-quality renderings for that perspective but results in poor performance for novel views. Removing view-conditioning typically destabilizes GAN training, often causing the training to collapse. In response to these challenges, we introduce CGS-GAN, a novel 3D Gaussian Splatting GAN framework that enables stable training and high-quality 3D-consistent synthesis of human heads without relying on view-conditioning. To ensure training stability, we introduce a multi-view regularization technique that enhances generator convergence with minimal computational overhead. Additionally, we adapt the conditional loss used in existing 3D Gaussian splatting GANs and propose a generator architecture designed to not only stabilize training but also facilitate efficient rendering and straightforward scaling, enabling output resolutions up to $2048^2$. To evaluate the capabilities of CGS-GAN, we curate a new dataset derived from FFHQ. This dataset enables very high resolutions, focuses on larger portions of the human head, reduces view-dependent artifacts for improved 3D consistency, and excludes images where subjects are obscured by hands or other objects. As a result, our approach achieves very high rendering quality, supported by competitive FID scores, while ensuring consistent 3D scene generation. Check our our project page here: https://fraunhoferhhi.github.io/cgs-gan/
>
---
#### [replaced 030] Real-Time Object Detection Meets DINOv3
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20787v3](https://arxiv.org/pdf/2509.20787v3)**

> **作者:** Shihua Huang; Yongjie Hou; Longfei Liu; Xuanlong Yu; Xi Shen
>
> **备注:** Source code available at https://github.com/Intellindust-AI-Lab/DEIMv2
>
> **摘要:** Benefiting from the simplicity and effectiveness of Dense O2O and MAL, DEIM has become the mainstream training framework for real-time DETRs, significantly outperforming the YOLO series. In this work, we extend it with DINOv3 features, resulting in DEIMv2. DEIMv2 spans eight model sizes from X to Atto, covering GPU, edge, and mobile deployment. For the X, L, M, and S variants, we adopt DINOv3-pretrained or distilled backbones and introduce a Spatial Tuning Adapter (STA), which efficiently converts DINOv3's single-scale output into multi-scale features and complements strong semantics with fine-grained details to enhance detection. For ultra-lightweight models (Nano, Pico, Femto, and Atto), we employ HGNetv2 with depth and width pruning to meet strict resource budgets. Together with a simplified decoder and an upgraded Dense O2O, this unified design enables DEIMv2 to achieve a superior performance-cost trade-off across diverse scenarios, establishing new state-of-the-art results. Notably, our largest model, DEIMv2-X, achieves 57.8 AP with only 50.3 million parameters, surpassing prior X-scale models that require over 60 million parameters for just 56.5 AP. On the compact side, DEIMv2-S is the first sub-10 million model (9.71 million) to exceed the 50 AP milestone on COCO, reaching 50.9 AP. Even the ultra-lightweight DEIMv2-Pico, with just 1.5 million parameters, delivers 38.5 AP, matching YOLOv10-Nano (2.3 million) with around 50 percent fewer parameters. Our code and pre-trained models are available at https://github.com/Intellindust-AI-Lab/DEIMv2
>
---
#### [replaced 031] DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.14957v2](https://arxiv.org/pdf/2509.14957v2)**

> **作者:** Zhuokang Shen; Kaisen Zhang; Bohan Jia; Heming Jia; Yuan Fang; Zhou Yu; Shaohui Lin
>
> **备注:** Under review
>
> **摘要:** With the increasing prevalence of synthetic images, evaluating image authenticity and locating forgeries accurately while maintaining human interpretability remains a challenging task. Existing detection models primarily focus on simple authenticity classification, ultimately providing only a forgery probability or binary judgment, which offers limited explanatory insights into image authenticity. Moreover, while MLLM-based detection methods can provide more interpretable results, they still lag behind expert models in terms of pure authenticity classification accuracy. To address this, we propose DF-LLaVA, a simple yet effective framework that unlocks the intrinsic discrimination potential of MLLMs. Our approach first extracts latent knowledge from MLLMs and then injects it into training via prompts. This framework allows LLaVA to achieve outstanding detection accuracy exceeding expert models while still maintaining the interpretability offered by MLLMs. Extensive experiments confirm the superiority of our DF-LLaVA, achieving both high accuracy and explainability in synthetic image detection. Code is available online at: https://github.com/Eliot-Shen/DF-LLaVA.
>
---
#### [replaced 032] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.06943v2](https://arxiv.org/pdf/2511.06943v2)**

> **作者:** Ayushi Sharma; Johanna Trost; Daniel Lusk; Johannes Dollinger; Julian Schrader; Christian Rossi; Javier Lopatin; Etienne Laliberté; Simon Haberstroh; Jana Eichel; Daniel Mederer; Jose Miguel Cerda-Paredes; Shyam S. Phartyal; Lisa-Maricia Schwarz; Anja Linstädter; Maria Conceição Caldeira; Teja Kattenborn
>
> **备注:** Preprint version of the paper accepted at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), organized by the Association for the Advancement of Artificial Intelligence
>
> **摘要:** Global plant maps of plant traits, such as leaf nitrogen or plant height, are essential for understanding ecosystem processes, including the carbon and energy cycles of the Earth system. However, existing trait maps remain limited by the high cost and sparse geographic coverage of field-based measurements. Citizen science initiatives offer a largely untapped resource to overcome these limitations, with over 50 million geotagged plant photographs worldwide capturing valuable visual information on plant morphology and physiology. In this study, we introduce PlantTraitNet, a multi-modal, multi-task uncertainty-aware deep learning framework that predictsfour key plant traits (plant height, leaf area, specific leaf area, and nitrogen content) from citizen science photos using weak supervision. By aggregating individual trait predictions across space, we generate global maps of trait distributions. We validate these maps against independent vegetation survey data (sPlotOpen) and benchmark them against leading global trait products. Our results show that PlantTraitNet consistently outperforms existing trait maps across all evaluated traits, demonstrating that citizen science imagery, when integrated with computer vision and geospatial AI, enables not only scalable but also more accurate global trait mapping. This approach offers a powerful new pathway for ecological research and Earth system modeling.
>
---
#### [replaced 033] No Mesh, No Problem: Estimating Coral Volume and Surface from Sparse Multi-View Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11164v3](https://arxiv.org/pdf/2509.11164v3)**

> **作者:** Diego Eustachio Farchione; Ramzi Idoughi; Peter Wonka
>
> **备注:** Reverted to previous version due to clarity issues
>
> **摘要:** Effective reef monitoring requires the quantification of coral growth via accurate volumetric and surface area estimates, which is a challenging task due to the complex morphology of corals. We propose a novel, lightweight, and scalable learning framework that addresses this challenge by predicting the 3D volume and surface area of coral-like objects from 2D multi-view RGB images. Our approach utilizes a pre-trained module (VGGT) to extract dense point maps from each view; these maps are merged into a unified point cloud and enriched with per-view confidence scores. The resulting cloud is fed to two parallel DGCNN decoder heads, which jointly output the volume and the surface area of the coral, as well as their corresponding confidence estimate. To enhance prediction stability and provide uncertainty estimates, we introduce a composite loss function based on Gaussian negative log-likelihood in both real and log domains. Our method achieves competitive accuracy and generalizes well to unseen morphologies. This framework paves the way for efficient and scalable coral geometry estimation directly from a sparse set of images, with potential applications in coral growth analysis and reef monitoring.
>
---
#### [replaced 034] StyMam: A Mamba-Based Generator for Artistic Style Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12954v2](https://arxiv.org/pdf/2601.12954v2)**

> **作者:** Zhou Hong; Rongsheng Hu; Yicheng Di; Xiaolong Xu; Ning Dong; Yihua Shao; Run Ling; Yun Wang; Juqin Wang; Zhanjie Zhang; Ao Ma
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Image style transfer aims to integrate the visual patterns of a specific artistic style into a content image while preserving its content structure. Existing methods mainly rely on the generative adversarial network (GAN) or stable diffusion (SD). GAN-based approaches using CNNs or Transformers struggle to jointly capture local and global dependencies, leading to artifacts and disharmonious patterns. SD-based methods reduce such issues but often fail to preserve content structures and suffer from slow inference. To address these issues, we revisit GAN and propose a mamba-based generator, termed as StyMam, to produce high-quality stylized images without introducing artifacts and disharmonious patterns. Specifically, we introduce a mamba-based generator with a residual dual-path strip scanning mechanism and a channel-reweighted spatial attention module. The former efficiently captures local texture features, while the latter models global dependencies. Finally, extensive qualitative and quantitative experiments demonstrate that the proposed method outperforms state-of-the-art algorithms in both quality and speed.
>
---
#### [replaced 035] Decoupling Multi-Contrast Super-Resolution: Self-Supervised Implicit Re-Representation for Unpaired Cross-Modal Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.05855v2](https://arxiv.org/pdf/2505.05855v2)**

> **作者:** Yinzhe Wu; Hongyu Rui; Fanwen Wang; Jiahao Huang; Zhenxuan Zhang; Haosen Zhang; Zi Wang; Guang Yang
>
> **摘要:** Multi-contrast super-resolution (MCSR) is crucial for enhancing MRI but current deep learning methods are limited. They typically require large, paired low- and high-resolution (LR/HR) training datasets, which are scarce, and are trained for fixed upsampling scales. While recent self-supervised methods remove the paired data requirement, they fail to leverage valuable population-level priors. In this work, we propose a novel, decoupled MCSR framework that resolves both limitations. We reformulate MCSR into two stages: (1) an unpaired cross-modal synthesis (uCMS) module, trained once on unpaired population data to learn a robust anatomical prior; and (2) a lightweight, patient-specific implicit re-representation (IrR) module. This IrR module is optimized in a self-supervised manner to fuse the population prior with the subject's own LR target data. This design uniquely fuses population-level knowledge with patient-specific fidelity without requiring any paired LR/HR or paired cross-modal training data. By building the IrR module on an implicit neural representation, our framework is also inherently scale-agnostic. Our method demonstrates superior quantitative performance on different datasets, with exceptional robustness at extreme scales (16x, 32x), a regime where competing methods fail. Our work presents a data-efficient, flexible, and computationally lightweight paradigm for MCSR, enabling high-fidelity, arbitrary-scale
>
---
#### [replaced 036] TeleMem: Building Long-Term and Multimodal Memory for Agentic AI
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出TeleMem，解决LLM长期交互与多模态记忆问题。通过动态提取和结构化写入提升记忆效率与准确性。属于对话系统与记忆管理任务。**

- **链接: [https://arxiv.org/pdf/2601.06037v4](https://arxiv.org/pdf/2601.06037v4)**

> **作者:** Chunliang Chen; Ming Guan; Xiao Lin; Jiaxu Li; Luxi Lin; Qiyi Wang; Xiangyu Chen; Jixiang Luo; Changzhi Sun; Dell Zhang; Xuelong Li
>
> **摘要:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal reasoning.To address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.
>
---
#### [replaced 037] Skin Lesion Phenotyping via Nested Multi-modal Contrastive Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.23709v2](https://arxiv.org/pdf/2505.23709v2)**

> **作者:** Dionysis Christopoulos; Sotiris Spanos; Eirini Baltzi; Valsamis Ntouskos; Konstantinos Karantzalos
>
> **摘要:** We introduce SLIMP (Skin Lesion Image-Metadata Pre-training) for learning rich representations of skin lesions through a novel nested contrastive learning approach that captures complex relationships between images and metadata. Melanoma detection and skin lesion classification based solely on images, pose significant challenges due to large variations in imaging conditions (lighting, color, resolution, distance, etc.) and lack of clinical and phenotypical context. Clinicians typically follow a holistic approach for assessing the risk level of the patient and for deciding which lesions may be malignant and need to be excised, by considering the patient's medical history as well as the appearance of other lesions of the patient. Inspired by this, SLIMP combines the appearance and the metadata of individual skin lesions with patient-level metadata relating to their medical record and other clinically relevant information. By fully exploiting all available data modalities throughout the learning process, the proposed pre-training strategy improves performance compared to other pre-training strategies on downstream skin lesions classification tasks highlighting the learned representations quality.
>
---
#### [replaced 038] Dynamic Exploration on Segment-Proposal Graphs for Tubular Centerline Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.18930v2](https://arxiv.org/pdf/2506.18930v2)**

> **作者:** Chong Di; Jinglin Zhang; Zhenjiang Li; Jean-Marie Mirebeau; Da Chen; Laurent D. Cohen
>
> **备注:** A real time interactive model that can accurately find centerline of a tubular structure even in complex scenarios. At this version, this work is independent to deep learning-based algorithms
>
> **摘要:** Optimal curve methods provide a fundamental framework for tubular centerline tracking. Point-wise approaches, such as minimal paths, are theoretically elegant but often suffer from shortcut and short-branch combination problems in complex scenarios. Nonlocal segment-wise methods address these issues by mapping pre-extracted centerline fragments onto a segment-proposal graph, performing optimization in this abstract space, and recovering the target tubular centerline from the resulting optimal path. In this paradigm, graph construction is critical, as it directly determines the quality of the final result. However, existing segment-wise methods construct graphs in a static manner, requiring all edges and their weights to be pre-computed, i.e. the graph must be sufficiently complete prior to search. Otherwise, the true path may be absent from the candidate space, leading to search failure. To address this limitation, we propose a dynamic exploration scheme for constructing segment-proposal graphs, where the graph is built on demand during the search for optimal paths. By formulating the problem as a Markov decision process, we apply Q-learning to compute edge weights only for visited transitions and adaptively expand the action space when connectivity is insufficient. Experimental results on retinal vessels, roads, and rivers demonstrate consistent improvements over state-of-the-art methods in both accuracy and efficiency.
>
---
#### [replaced 039] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.11213v2](https://arxiv.org/pdf/2503.11213v2)**

> **作者:** Fengchen He; Dayang Zhao; Hao Xu; Tingwei Quan; Shaoqun Zeng
>
> **摘要:** Many studies utilize dual-pixel (DP) sensor phase characteristics for various applications, such as depth estimation and deblurring. However, since the DP image features are entirely determined by the camera hardware, DP-depth paired datasets are very scarce, especially when performing depth estimation on customized cameras. To overcome this, studies simulate DP images using ideal optical system models. However, these simulations often violate real optical propagation laws, leading to poor generalization to real DP data. To address this, we investigate the domain gap between simulated and real DP data, and propose solutions using the Simulating DP images from ray tracing (Sdirt) scheme. The Sdirt generates realistic DP images via ray tracing and integrates them into the depth estimation training pipeline. Experimental results show that models trained with Sdirt-simulated images generalize better to real DP data. The code and collected datasets will be available at github.com/LinYark/Sdirt
>
---
#### [replaced 040] Is this chart lying to me? Automating the detection of misleading visualizations
- **分类: cs.CL; cs.CV; cs.GR**

- **简介: 该论文属于信息可视化检测任务，旨在解决误导性图表识别问题。作者构建了真实和合成数据集，并评估多种模型性能，以提升对误导性图表的自动检测能力。**

- **链接: [https://arxiv.org/pdf/2508.21675v2](https://arxiv.org/pdf/2508.21675v2)**

> **作者:** Jonathan Tonglet; Jan Zimny; Tinne Tuytelaars; Iryna Gurevych
>
> **备注:** Preprint under review. Code and data available at: https://github.com/UKPLab/arxiv2025-misviz
>
> **摘要:** Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also create Misviz-synth, a synthetic dataset of 57,665 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and image-axis classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code.
>
---
#### [replaced 041] From Canopy to Ground via ForestGen3D: Learning Cross-Domain Generation of 3D Forest Structure from Aerial-to-Terrestrial LiDAR
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.16346v2](https://arxiv.org/pdf/2509.16346v2)**

> **作者:** Juan Castorena; E. Louise Loudermilk; Scott Pokswinski; Rodman Linn
>
> **摘要:** The 3D structure of living and non-living components in ecosystems plays a critical role in determining ecological processes and feedbacks from both natural and human-driven disturbances. Anticipating the effects of wildfire, drought, disease, or atmospheric deposition depends on accurate characterization of 3D vegetation structure, yet widespread measurement remains prohibitively expensive and often infeasible. We present ForestGen3D, a cross-domain generative framework that preserves aerial LiDAR (ALS) observed 3D forest structure while inferring missing sub-canopy detail. ForestGen3D is based on conditional denoising diffusion probabilistic models trained on co-registered ALS and terrestrial LiDAR (TLS) data. The model generates realistic TLS-like point clouds that remain spatially consistent with ALS geometry, enabling landscape-scalable reconstruction of full vertical forest structure. We evaluate ForestGen3D at tree, plot, and landscape scales using real-world data from mixed conifer ecosystems, and show through qualitative and quantitative geometric and distributional analyses that it produces high-fidelity reconstructions closely matching TLS reference data in terms of 3D structural similarity and downstream biophysical metrics, including tree height, DBH, crown diameter, and crown volume. We further introduce and demonstrate the expected point containment (EPC) metric which serves as a practical proxy for generation quality in settings where TLS ground truth is unavailable. Our results demonstrate that ForestGen3D enhances the utility of ALS only environments by inferring ecologically plausible sub-canopy structure while faithfully preserving the landscape heterogeneity encoded in ALS observations, thereby providing a richer 3D representation for ecological analysis, structural fuel characterization and related remote sensing applications.
>
---
#### [replaced 042] SUG-Occ: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11396v3](https://arxiv.org/pdf/2601.11396v3)**

> **作者:** Hanlin Wu; Pengfei Lin; Ehsan Javanmardi; Naren Bao; Bo Qian; Hao Si; Manabu Tsukada
>
> **摘要:** As autonomous driving moves toward full scene understanding, 3D semantic occupancy prediction has emerged as a crucial perception task, offering voxel-level semantics beyond traditional detection and segmentation paradigms. However, such a refined representation for scene understanding incurs prohibitive computation and memory overhead, posing a major barrier to practical real-time deployment. To address this, we propose SUG-Occ, an explicit Semantics and Uncertainty Guided Sparse Learning Enabled 3D Occupancy Prediction Framework, which exploits the inherent sparsity of 3D scenes to reduce redundant computation while maintaining geometric and semantic completeness. Specifically, we first utilize semantic and uncertainty priors to suppress projections from free space during view transformation while employing an explicit unsigned distance encoding to enhance geometric consistency, producing a structurally consistent sparse 3D representation. Secondly, we design an cascade sparse completion module via hyper cross sparse convolution and generative upsampling to enable efficiently coarse-to-fine reasoning. Finally, we devise an object contextual representation (OCR) based mask decoder that aggregates global semantic context from sparse features and refines voxel-wise predictions via lightweight query-context interactions, avoiding expensive attention operations over volumetric features. Extensive experiments on SemanticKITTI benchmark demonstrate that the proposed approach outperforms the baselines, achieving a 7.34/% improvement in accuracy and a 57.8\% gain in efficiency.
>
---
#### [replaced 043] Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13944v2](https://arxiv.org/pdf/2511.13944v2)**

> **作者:** Noam Glazner; Noam Tsfaty; Sharon Shalev; Avishai Weizman
>
> **备注:** 1 figure, 1 table, Accepted to ICSEE 2026
>
> **摘要:** We propose a cluster-based frame selection strategy to mitigate information leakage in video-derived frames datasets. By grouping visually similar frames before splitting into training, validation, and test sets, the method produces more representative, balanced, and reliable dataset partitions.
>
---
#### [replaced 044] TUN: Detecting Significant Points in Persistence Diagrams with Deep Learning
- **分类: cs.CV; cs.LG; math.AT**

- **链接: [https://arxiv.org/pdf/2512.14274v2](https://arxiv.org/pdf/2512.14274v2)**

> **作者:** Yu Chen; Hongwei Lin
>
> **摘要:** Persistence diagrams (PDs) provide a powerful tool for understanding the topology of the underlying shape of a point cloud. However, identifying which points in PDs encode genuine signals remains challenging. This challenge directly hinders the practical adoption of topological data analysis in many applications, where automated and reliable interpretation of persistence diagrams is essential for downstream decision-making. In this paper, we study automatic significance detection for one-dimensional persistence diagrams. Specifically, we propose Topology Understanding Net (TUN), a multi-modal network that combines enhanced PD descriptors with self-attention, a PointNet-style point cloud encoder, learned fusion, and per-point classification, alongside stable preprocessing and imbalance-aware training. It provides an automated and effective solution for identifying significant points in PDs, which are critical for downstream applications. Experiments show that TUN outperforms classic methods in detecting significant points in PDs, illustrating its effectiveness in real-world applications.
>
---
#### [replaced 045] Boosting Generative Image Modeling via Joint Image-Feature Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.16064v3](https://arxiv.org/pdf/2504.16064v3)**

> **作者:** Theodoros Kouzelis; Efstathios Karypidis; Ioannis Kakogeorgiou; Spyros Gidaris; Nikos Komodakis
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Latent diffusion models (LDMs) dominate high-quality image generation, yet integrating representation learning with generative modeling remains a challenge. We introduce a novel generative image modeling framework that seamlessly bridges this gap by leveraging a diffusion model to jointly model low-level image latents (from a variational autoencoder) and high-level semantic features (from a pretrained self-supervised encoder like DINO). Our latent-semantic diffusion approach learns to generate coherent image-feature pairs from pure noise, significantly enhancing both generative quality and training efficiency, all while requiring only minimal modifications to standard Diffusion Transformer architectures. By eliminating the need for complex distillation objectives, our unified design simplifies training and unlocks a powerful new inference strategy: Representation Guidance, which leverages learned semantics to steer and refine image generation. Evaluated in both conditional and unconditional settings, our method delivers substantial improvements in image quality and training convergence speed, establishing a new direction for representation-aware generative modeling. Project page and code: https://representationdiffusion.github.io
>
---
#### [replaced 046] GutenOCR: A Grounded Vision-Language Front-End for Documents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GutenOCR，属于文档OCR任务，解决文本识别与定位问题。通过微调视觉语言模型，实现精准的文本检测、阅读和定位，提升OCR性能。**

- **链接: [https://arxiv.org/pdf/2601.14490v2](https://arxiv.org/pdf/2601.14490v2)**

> **作者:** Hunter Heidenreich; Ben Elliott; Olivia Dinica; Yosheb Getachew
>
> **摘要:** GutenOCR is a family of grounded OCR front-ends obtained by fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B. The resulting single-checkpoint vision-language models expose reading, detection, and grounding through a unified, prompt-based interface. Trained on business documents, scientific articles, and synthetic grounding data, the models support full-page and localized reading with line- and paragraph-level bounding boxes and conditional ``where is x?'' queries. We introduce a grounded OCR evaluation protocol and show that GutenOCR-7B more than doubles the composite grounded OCR score of its Qwen2.5-VL-7B backbone on 10.5K held-out business and scientific pages (0.40 to 0.82). On Fox and OmniDocBench v1.5, our approach substantially improves region- and line-level OCR as well as text-detection recall, but reveals trade-offs in page-level linearization, color-guided OCR, and formula-heavy layouts.
>
---
#### [replaced 047] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Ultrasound-Xray Policy with Limited Labels
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18457v2](https://arxiv.org/pdf/2511.18457v2)**

> **作者:** Duncan Stothers; Ben Stothers; Emily Schaeffer; Kishore Mulpuri
>
> **备注:** Accepted (with oral presentation) to the AAAI 2026 AI for Medicine and Healthcare Bridge Program Awarded Best Paper Runner-Up at the AAAI 2026 AI for Medicine and Healthcare Bridge Program
>
> **摘要:** We study an ultrasound-first, radiation-preserving policy for developmental dysplasia of the hip (DDH) that requests a radiograph only when needed. We (i) pretrain modality-specific encoders (ResNet-18) with SimSiam on a large unlabelled registry (37186 ultrasound; 19546 radiographs), (ii) freeze the backbones and fit small, measurement-faithful heads on DDH-relevant landmarks and measurements, (iii) calibrate a one-sided conformal deferral rule on ultrasound predictions that provides finite sample marginal coverage guarantees under exchangeability, using a held-out calibration set. Ultrasound heads predict Graf alpha, beta, and femoral head coverage; X-ray heads predict acetabular index (AI), center-edge (CE) angle and IHDI grade. On our held out labeled evaluation set, ultrasound measurement error is modest (e.g., alpha MAE ~= 9.7 degrees, coverage MAE ~= 14.0%), while radiographic probes achieve AI and CE MAEs of ~= 7.6 degrees and ~= 8.9 degrees, respectively. The calibrated US-only policy is explored across rule families (alpha-only; alpha OR coverage; alpha AND coverage), conformal miscoverage levels, and per-utility trade-offs using decision-curve analysis. Conservative settings yield high coverage with near-zero US-only rates; permissive settings (e.g., alpha OR coverage at larger deltas) achieve non-zero US-only throughput with expected coverage tradeoffs. The result is a simple, reproducible pipeline that turns limited labels into interpretable measurements and tunable selective imaging curves suitable for clinical handoff and future external validation.
>
---
#### [replaced 048] GO-MLVTON: Garment Occlusion-Aware Multi-Layer Virtual Try-On with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.13524v2](https://arxiv.org/pdf/2601.13524v2)**

> **作者:** Yang Yu; Yunze Deng; Yige Zhang; Yanjie Xiao; Youkun Ou; Wenhao Hu; Mingchao Li; Bin Feng; Wenyu Liu; Dandan Zheng; Jingdong Chen
>
> **备注:** 5pages, 3 figures, Accepted at ICASSP 2026
>
> **摘要:** Existing image-based virtual try-on (VTON) methods primarily focus on single-layer or multi-garment VTON, neglecting multi-layer VTON (ML-VTON), which involves dressing multiple layers of garments onto the human body with realistic deformation and layering to generate visually plausible outcomes. The main challenge lies in accurately modeling occlusion relationships between inner and outer garments to reduce interference from redundant inner garment features. To address this, we propose GO-MLVTON, the first multi-layer VTON method, introducing the Garment Occlusion Learning module to learn occlusion relationships and the StableDiffusion-based Garment Morphing & Fitting module to deform and fit garments onto the human body, producing high-quality multi-layer try-on results. Additionally, we present the MLG dataset for this task and propose a new metric named Layered Appearance Coherence Difference (LACD) for evaluation. Extensive experiments demonstrate the state-of-the-art performance of GO-MLVTON. Project page: https://upyuyang.github.io/go-mlvton/.
>
---
#### [replaced 049] SURE-Med: Systematic Uncertainty Reduction for Enhanced Reliability in Medical Report Generation
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01693v2](https://arxiv.org/pdf/2508.01693v2)**

> **作者:** Yuhang Gu; Xingyu Hu; Yuyu Fan; Xulin Yan; Longhuan Xu; Peng peng
>
> **备注:** fix some problems
>
> **摘要:** Automated medical report generation (MRG) holds great promise for reducing the heavy workload of radiologists. However, its clinical deployment is hindered by three major sources of uncertainty. First, visual uncertainty, caused by noisy or incorrect view annotations, compromises feature extraction. Second, label distribution uncertainty, stemming from long-tailed disease prevalence, biases models against rare but clinically critical conditions. Third, contextual uncertainty, introduced by unverified historical reports, often leads to factual hallucinations. These challenges collectively limit the reliability and clinical trustworthiness of MRG systems. To address these issues, we propose SURE-Med, a unified framework that systematically reduces uncertainty across three critical dimensions: visual, distributional, and contextual. To mitigate visual uncertainty, a Frontal-Aware View Repair Resampling module corrects view annotation errors and adaptively selects informative features from supplementary views. To tackle label distribution uncertainty, we introduce a Token Sensitive Learning objective that enhances the modeling of critical diagnostic sentences while reweighting underrepresented diagnostic terms, thereby improving sensitivity to infrequent conditions. To reduce contextual uncertainty, our Contextual Evidence Filter validates and selectively incorporates prior information that aligns with the current image, effectively suppressing hallucinations. Extensive experiments on the MIMIC-CXR and IU-Xray benchmarks demonstrate that SURE-Med achieves state-of-the-art performance. By holistically reducing uncertainty across multiple input modalities, SURE-Med sets a new benchmark for reliability in medical report generation and offers a robust step toward trustworthy clinical decision support.
>
---
#### [replaced 050] Scribble-Supervised Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14563v2](https://arxiv.org/pdf/2601.14563v2)**

> **作者:** Thanh-Huy Nguyen; Hoang-Loc Cao; Dat T. Chung; Mai-Anh Vu; Thanh-Minh Nguyen; Minh Le; Phat K. Huynh; Ulas Bagci
>
> **摘要:** Scribble-supervised methods have emerged to mitigate the prohibitive annotation burden in medical image segmentation. However, the inherent sparsity of these annotations introduces significant ambiguity, which results in noisy pseudo-label propagation and hinders the learning of robust anatomical boundaries. To address this challenge, we propose SDT-Net, a novel dual-teacher, single-student framework designed to maximize supervision quality from these weak signals. Our method features a Dynamic Teacher Switching (DTS) module to adaptively select the most reliable teacher. This selected teacher then guides the student via two synergistic mechanisms: high-confidence pseudo-labels, refined by a Pick Reliable Pixels (PRP) mechanism, and multi-level feature alignment, enforced by a Hierarchical Consistency (HiCo) module. Extensive experiments on the ACDC and MSCMRseg datasets demonstrate that SDT-Net achieves state-of-the-art performance, producing more accurate and anatomically plausible segmentation.
>
---
#### [replaced 051] CropCraft: Complete Structural Characterization of Crop Plants From Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.09693v2](https://arxiv.org/pdf/2411.09693v2)**

> **作者:** Albert J. Zhai; Xinlei Wang; Kaiyuan Li; Zhao Jiang; Junxiong Zhou; Sheng Wang; Zhenong Jin; Kaiyu Guan; Shenlong Wang
>
> **备注:** 3DV 2026 (Oral). Project page: https://ajzhai.github.io/CropCraft
>
> **摘要:** The ability to automatically build 3D digital twins of plants from images has countless applications in agriculture, environmental science, robotics, and other fields. However, current 3D reconstruction methods fail to recover complete shapes of plants due to heavy occlusion and complex geometries. In this work, we present a novel method for 3D modeling of agricultural crops based on optimizing a parametric model of plant morphology via inverse procedural modeling. Our method first estimates depth maps by fitting a neural radiance field and then optimizes a specialized loss to estimate morphological parameters that result in consistent depth renderings. The resulting 3D model is complete and biologically plausible. We validate our method on a dataset of real images of agricultural fields, and demonstrate that the reconstructed canopies can be used for a variety of monitoring and simulation applications.
>
---
