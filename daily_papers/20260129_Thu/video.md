# 计算机视觉 cs.CV

- **最新发布 78 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] Advancing Open-source World Models
- **分类: cs.CV**

- **简介: 该论文提出LingBot-World，一个高保真、支持长时记忆和实时交互的开源世界模型，旨在推动内容创作、游戏和机器人学习等应用。**

- **链接: [https://arxiv.org/pdf/2601.20540v1](https://arxiv.org/pdf/2601.20540v1)**

> **作者:** Robbyant Team; Zelin Gao; Qiuyu Wang; Yanhong Zeng; Jiapeng Zhu; Ka Leong Cheng; Yixuan Li; Hanlin Wang; Yinghao Xu; Shuailei Ma; Yihang Chen; Jie Liu; Yansong Cheng; Yao Yao; Jiayi Zhu; Yihao Meng; Kecheng Zheng; Qingyan Bai; Jingye Chen; Zehong Shen; Yue Yu; Xing Zhu; Yujun Shen; Hao Ouyang
>
> **备注:** Project page: https://technology.robbyant.com/lingbot-world; Code: https://github.com/robbyant/lingbot-world
>
> **摘要:** We present LingBot-World, an open-sourced world simulator stemming from video generation. Positioned as a top-tier world model, LingBot-World offers the following features. (1) It maintains high fidelity and robust dynamics in a broad spectrum of environments, including realism, scientific contexts, cartoon styles, and beyond. (2) It enables a minute-level horizon while preserving contextual consistency over time, which is also known as "long-term memory". (3) It supports real-time interactivity, achieving a latency of under 1 second when producing 16 frames per second. We provide public access to the code and model in an effort to narrow the divide between open-source and closed-source technologies. We believe our release will empower the community with practical applications across areas like content creation, gaming, and robot learning.
>
---
#### [new 002] RAW-Flow: Advancing RGB-to-RAW Image Reconstruction with Deterministic Latent Flow Matching
- **分类: cs.CV**

- **简介: 该论文属于RGB到RAW图像重建任务，旨在解决逆ISP过程中的细节不一致和颜色偏差问题。提出RAW-Flow框架，通过确定性潜在流匹配实现高保真重建。**

- **链接: [https://arxiv.org/pdf/2601.20364v1](https://arxiv.org/pdf/2601.20364v1)**

> **作者:** Zhen Liu; Diedong Feng; Hai Jiang; Liaoyuan Zeng; Hao Wang; Chaoyu Feng; Lei Lei; Bing Zeng; Shuaicheng Liu
>
> **备注:** AAAI2026 Oral
>
> **摘要:** RGB-to-RAW reconstruction, or the reverse modeling of a camera Image Signal Processing (ISP) pipeline, aims to recover high-fidelity RAW data from RGB images. Despite notable progress, existing learning-based methods typically treat this task as a direct regression objective and struggle with detail inconsistency and color deviation, due to the ill-posed nature of inverse ISP and the inherent information loss in quantized RGB images. To address these limitations, we pioneer a generative perspective by reformulating RGB-to-RAW reconstruction as a deterministic latent transport problem and introduce a novel framework named RAW-Flow, which leverages flow matching to learn a deterministic vector field in latent space, to effectively bridge the gap between RGB and RAW representations and enable accurate reconstruction of structural details and color information. To further enhance latent transport, we introduce a cross-scale context guidance module that injects hierarchical RGB features into the flow estimation process. Moreover, we design a dual-domain latent autoencoder with a feature alignment constraint to support the proposed latent transport framework, which jointly encodes RGB and RAW inputs while promoting stable training and high-fidelity reconstruction. Extensive experiments demonstrate that RAW-Flow outperforms state-of-the-art approaches both quantitatively and visually.
>
---
#### [new 003] PalmBridge: A Plug-and-Play Feature Alignment Framework for Open-Set Palmprint Verification
- **分类: cs.CV**

- **简介: 该论文提出PalmBridge，解决开放集掌纹验证中的特征分布偏移问题。通过特征空间对齐提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.20351v1](https://arxiv.org/pdf/2601.20351v1)**

> **作者:** Chenke Zhang; Ziyuan Yang; Licheng Yan; Shuyi Li; Andrew Beng Jin Teoh; Bob Zhang; Yi Zhang
>
> **摘要:** Palmprint recognition is widely used in biometric systems, yet real-world performance often degrades due to feature distribution shifts caused by heterogeneous deployment conditions. Most deep palmprint models assume a closed and stationary distribution, leading to overfitting to dataset-specific textures rather than learning domain-invariant representations. Although data augmentation is commonly used to mitigate this issue, it assumes augmented samples can approximate the target deployment distribution, an assumption that often fails under significant domain mismatch. To address this limitation, we propose PalmBridge, a plug-and-play feature-space alignment framework for open-set palmprint verification based on vector quantization. Rather than relying solely on data-level augmentation, PalmBridge learns a compact set of representative vectors directly from training features. During enrollment and verification, each feature vector is mapped to its nearest representative vector under a minimum-distance criterion, and the mapped vector is then blended with the original vector. This design suppresses nuisance variation induced by domain shifts while retaining discriminative identity cues. The representative vectors are jointly optimized with the backbone network using task supervision, a feature-consistency objective, and an orthogonality regularization term to form a stable and well-structured shared embedding space. Furthermore, we analyze feature-to-representative mappings via assignment consistency and collision rate to assess model's sensitivity to blending weights. Experiments on multiple palmprint datasets and backbone architectures show that PalmBridge consistently reduces EER in intra-dataset open-set evaluation and improves cross-dataset generalization with negligible to modest runtime overhead.
>
---
#### [new 004] FD-MAD: Frequency-Domain Residual Analysis for Face Morphing Attack Detection
- **分类: cs.CV**

- **简介: 该论文属于人脸活体检测任务，旨在解决单图像仿生攻击检测问题。通过频域残差分析和区域融合，提升跨数据集检测性能。**

- **链接: [https://arxiv.org/pdf/2601.20656v1](https://arxiv.org/pdf/2601.20656v1)**

> **作者:** Diogo J. Paulo; Hugo Proença; João C. Neves
>
> **摘要:** Face morphing attacks present a significant threat to face recognition systems used in electronic identity enrolment and border control, particularly in single-image morphing attack detection (S-MAD) scenarios where no trusted reference is available. In spite of the vast amount of research on this problem, morph detection systems struggle in cross-dataset scenarios. To address this problem, we introduce a region-aware frequency-based morph detection strategy that drastically improves over strong baseline methods in challenging cross-dataset and cross-morph settings using a lightweight approach. Having observed the separability of bona fide and morph samples in the frequency domain of different facial parts, our approach 1) introduces the concept of residual frequency domain, where the frequency of the signal is decoupled from the natural spectral decay to easily discriminate between morph and bona fide data; 2) additionally, we reason in a global and local manner by combining the evidence from different facial regions in a Markov Random Field, which infers a globally consistent decision. The proposed method, trained exclusively on the synthetic morphing attack detection development dataset (SMDD), is evaluated in challenging cross-dataset and cross-morph settings on FRLL-Morph and MAD22 sets. Our approach achieves an average equal error rate (EER) of 1.85\% on FRLL-Morph and ranks second on MAD22 with an average EER of 6.12\%, while also obtaining a good bona fide presentation classification error rate (BPCER) at a low attack presentation classification error rate (APCER) using only spectral features. These findings indicate that Fourier-domain residual modeling with structured regional fusion offers a competitive alternative to deep S-MAD architectures.
>
---
#### [new 005] Decoupling Perception and Calibration: Label-Efficient Image Quality Assessment Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像质量评估任务，旨在减少对人工标注的依赖。通过知识蒸馏，将大模型的感知能力迁移至轻量模型，实现高效校准。**

- **链接: [https://arxiv.org/pdf/2601.20689v1](https://arxiv.org/pdf/2601.20689v1)**

> **作者:** Xinyue Li; Zhichao Zhang; Zhiming Xu; Shubo Xu; Xiongkuo Min; Yitong Chen; Guangtao Zhai
>
> **摘要:** Recent multimodal large language models (MLLMs) have demonstrated strong capabilities in image quality assessment (IQA) tasks. However, adapting such large-scale models is computationally expensive and still relies on substantial Mean Opinion Score (MOS) annotations. We argue that for MLLM-based IQA, the core bottleneck lies not in the quality perception capacity of MLLMs, but in MOS scale calibration. Therefore, we propose LEAF, a Label-Efficient Image Quality Assessment Framework that distills perceptual quality priors from an MLLM teacher into a lightweight student regressor, enabling MOS calibration with minimal human supervision. Specifically, the teacher conducts dense supervision through point-wise judgments and pair-wise preferences, with an estimate of decision reliability. Guided by these signals, the student learns the teacher's quality perception patterns through joint distillation and is calibrated on a small MOS subset to align with human annotations. Experiments on both user-generated and AI-generated IQA benchmarks demonstrate that our method significantly reduces the need for human annotations while maintaining strong MOS-aligned correlations, making lightweight IQA practical under limited annotation budgets.
>
---
#### [new 006] IOTA: Corrective Knowledge-Guided Prompt Learning via Black-White Box Framework
- **分类: cs.CV**

- **简介: 该论文属于下游任务适应任务，旨在解决预训练模型在微调中依赖数据而忽视先验知识的问题。提出IOTA框架，结合数据驱动与知识驱动方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20526v1](https://arxiv.org/pdf/2601.20526v1)**

> **作者:** Shaokun Wang; Yifan Yu; Yuhang He; Weili Guan; Yihong Gong
>
> **摘要:** Recently, adapting pre-trained models to downstream tasks has attracted increasing interest. Previous Parameter-Efficient-Tuning (PET) methods regard the pre-trained model as an opaque Black Box model, relying purely on data-driven optimization and underutilizing their inherent prior knowledge. This oversight limits the models' potential for effective downstream task adaptation. To address these issues, we propose a novel black-whIte bOx prompT leArning framework (IOTA), which integrates a data-driven Black Box module with a knowledge-driven White Box module for downstream task adaptation. Specifically, the White Box module derives corrective knowledge by contrasting the wrong predictions with the right cognition. This knowledge is verbalized into interpretable human prompts and leveraged through a corrective knowledge-guided prompt selection strategy to guide the Black Box module toward more accurate predictions. By jointly leveraging knowledge- and data-driven learning signals, IOTA achieves effective downstream task adaptation. Experimental results on 12 image classification benchmarks under few-shot and easy-to-hard adaptation settings demonstrate the effectiveness of corrective knowledge and the superiority of our method over state-of-the-art methods.
>
---
#### [new 007] A Source-Free Approach for Domain Adaptation via Multiview Image Transformation and Latent Space Consistency
- **分类: cs.CV**

- **简介: 该论文属于域适应任务，旨在解决无源域数据情况下图像分类的域迁移问题。通过多视角增强和潜在空间一致性学习，直接从目标域提取不变特征，提升分类准确率。**

- **链接: [https://arxiv.org/pdf/2601.20284v1](https://arxiv.org/pdf/2601.20284v1)**

> **作者:** Debopom Sutradhar; Md. Abdur Rahman; Mohaimenul Azam Khan Raiaan; Reem E. Mohamed; Sami Azam
>
> **备注:** Manuscript under review in IEEE Transactions on Image Processing
>
> **摘要:** Domain adaptation (DA) addresses the challenge of transferring knowledge from a source domain to a target domain where image data distributions may differ. Existing DA methods often require access to source domain data, adversarial training, or complex pseudo-labeling techniques, which are computationally expensive. To address these challenges, this paper introduces a novel source-free domain adaptation method. It is the first approach to use multiview augmentation and latent space consistency techniques to learn domain-invariant features directly from the target domain. Our method eliminates the need for source-target alignment or pseudo-label refinement by learning transferable representations solely from the target domain by enforcing consistency between multiple augmented views in the latent space. Additionally, the method ensures consistency in the learned features by generating multiple augmented views of target domain data and minimizing the distance between their feature representations in the latent space. We also introduce a ConvNeXt-based encoder and design a loss function that combines classification and consistency objectives to drive effective adaptation directly from the target domain. The proposed model achieves an average classification accuracy of 90. 72\%, 84\%, and 97. 12\% in Office-31, Office-Home and Office-Caltech datasets, respectively. Further evaluations confirm that our study improves existing methods by an average classification accuracy increment of +1.23\%, +7.26\%, and +1.77\% on the respective datasets.
>
---
#### [new 008] Comparative evaluation of training strategies using partially labelled datasets for segmentation of white matter hyperintensities and stroke lesions in FLAIR MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决WMH和ISL难以区分的问题。通过使用部分标注数据，探索训练策略以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20503v1](https://arxiv.org/pdf/2601.20503v1)**

> **作者:** Jesse Phitidis; Alison Q. Smithard; William N. Whiteley; Joanna M. Wardlaw; Miguel O. Bernabeu; Maria Valdés Hernández
>
> **摘要:** White matter hyperintensities (WMH) and ischaemic stroke lesions (ISL) are imaging features associated with cerebral small vessel disease (SVD) that are visible on brain magnetic resonance imaging (MRI) scans. The development and validation of deep learning models to segment and differentiate these features is difficult because they visually confound each other in the fluid-attenuated inversion recovery (FLAIR) sequence and often appear in the same subject. We investigated six strategies for training a combined WMH and ISL segmentation model using partially labelled data. We combined privately held fully and partially labelled datasets with publicly available partially labelled datasets to yield a total of 2052 MRI volumes, with 1341 and 1152 containing ground truth annotations for WMH and ISL respectively. We found that several methods were able to effectively leverage the partially labelled data to improve model performance, with the use of pseudolabels yielding the best result.
>
---
#### [new 009] Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决高压缩下索引向量性能下降问题。提出SAP方法，通过中间层结构锚点剪枝实现高效压缩，保持检索精度。**

- **链接: [https://arxiv.org/pdf/2601.20107v1](https://arxiv.org/pdf/2601.20107v1)**

> **作者:** Zhuchenyang Liu; Ziyu Hu; Yao Zhang; Yu Xiao
>
> **备注:** 18 pages, 6 figures, 11 tables
>
> **摘要:** Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive index vector size overheads. Training-free pruning solutions (e.g., EOS-attention based methods) can reduce index vector size by approximately 60% without model adaptation, but often underperform random selection in high-compression scenarios (> 80%). Prior research (e.g., Light-ColPali) attributes this to the conclusion that visual token importance is inherently query-dependent, thereby questioning the feasibility of training-free pruning. In this work, we propose Structural Anchor Pruning (SAP), a training-free pruning method that identifies key visual patches from middle layers to achieve high performance compression. We also introduce Oracle Score Retention (OSR) protocol to evaluate how layer-wise information affects compression efficiency. Evaluations on the ViDoRe benchmark demonstrate that SAP reduces index vectors by over 90% while maintaining robust retrieval fidelity, providing a highly scalable solution for Visual RAG. Furthermore, our OSR-based analysis reveals that semantic structural anchor patches persist in the middle layers, unlike traditional pruning solutions that focus on the final layer where structural signals dissipate.
>
---
#### [new 010] Efficient Token Pruning for LLaDA-V
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型优化任务，旨在解决扩散模型计算效率低的问题。通过结构化token剪枝，减少计算量并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.20168v1](https://arxiv.org/pdf/2601.20168v1)**

> **作者:** Zhewen Wan; Tianchen Song; Chen Lin; Zhiyong Zhao; Xianpeng Lang
>
> **摘要:** Diffusion-based large multimodal models, such as LLaDA-V, have demonstrated impressive capabilities in vision-language understanding and generation. However, their bidirectional attention mechanism and diffusion-style iterative denoising paradigm introduce significant computational overhead, as visual tokens are repeatedly processed across all layers and denoising steps. In this work, we conduct an in-depth attention analysis and reveal that, unlike autoregressive decoders, LLaDA-V aggregates cross-modal information predominantly in middle-to-late layers, leading to delayed semantic alignment. Motivated by this observation, we propose a structured token pruning strategy inspired by FastV, selectively removing a proportion of visual tokens at designated layers to reduce FLOPs while preserving critical semantic information. To the best of our knowledge, this is the first work to investigate structured token pruning in diffusion-based large multimodal models. Unlike FastV, which focuses on shallow-layer pruning, our method targets the middle-to-late layers of the first denoising step to align with LLaDA-V's delayed attention aggregation to maintain output quality, and the first-step pruning strategy reduces the computation across all subsequent steps. Our framework provides an empirical basis for efficient LLaDA-V inference and highlights the potential of vision-aware pruning in diffusion-based multimodal models. Across multiple benchmarks, our best configuration reduces computational cost by up to 65% while preserving an average of 95% task performance.
>
---
#### [new 011] Youtu-Parsing: Perception, Structuring and Recognition via High-Parallelism Decoding
- **分类: cs.CV**

- **简介: 该论文提出Youtu-Parsing，用于文档解析任务，解决高效内容提取问题。通过高并行解码策略提升处理速度与精度。**

- **链接: [https://arxiv.org/pdf/2601.20430v1](https://arxiv.org/pdf/2601.20430v1)**

> **作者:** Kun Yin; Yunfei Wu; Bing Liu; Zhongpeng Cai; Xiaotian Li; Huang Chen; Xin Li; Haoyu Cao; Yinsong Liu; Deqiang Jiang; Xing Sun; Yunsheng Wu; Qianyu Li; Antai Guo; Yanzhen Liao; Yanqiu Qu; Haodong Lin; Chengxu He; Shuangyin Liu
>
> **摘要:** This paper presents Youtu-Parsing, an efficient and versatile document parsing model designed for high-performance content extraction. The architecture employs a native Vision Transformer (ViT) featuring a dynamic-resolution visual encoder to extract shared document features, coupled with a prompt-guided Youtu-LLM-2B language model for layout analysis and region-prompted decoding. Leveraging this decoupled and feature-reusable framework, we introduce a high-parallelism decoding strategy comprising two core components: token parallelism and query parallelism. The token parallelism strategy concurrently generates up to 64 candidate tokens per inference step, which are subsequently validated through a verification mechanism. This approach yields a 5--11x speedup over traditional autoregressive decoding and is particularly well-suited for highly structured scenarios, such as table recognition. To further exploit the advantages of region-prompted decoding, the query parallelism strategy enables simultaneous content prediction for multiple bounding boxes (up to five), providing an additional 2x acceleration while maintaining output quality equivalent to standard decoding. Youtu-Parsing encompasses a diverse range of document elements, including text, formulas, tables, charts, seals, and hierarchical structures. Furthermore, the model exhibits strong robustness when handling rare characters, multilingual text, and handwritten content. Extensive evaluations demonstrate that Youtu-Parsing achieves state-of-the-art (SOTA) performance on both the OmniDocBench and olmOCR-bench benchmarks. Overall, Youtu-Parsing demonstrates significant experimental value and practical utility for large-scale document intelligence applications.
>
---
#### [new 012] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决T2I模型在空间关系理解上的不足。提出SpatialGenEval基准和SpatialT2I数据集，评估并提升模型的空间智能。**

- **链接: [https://arxiv.org/pdf/2601.20354v1](https://arxiv.org/pdf/2601.20354v1)**

> **作者:** Zengbin Wang; Xuecai Hu; Yong Wang; Feng Xiong; Man Zhang; Xiangxiang Chu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Text-to-image (T2I) models have achieved remarkable success in generating high-fidelity images, but they often fail in handling complex spatial relationships, e.g., spatial perception, reasoning, or interaction. These critical aspects are largely overlooked by current benchmarks due to their short or information-sparse prompt design. In this paper, we introduce SpatialGenEval, a new benchmark designed to systematically evaluate the spatial intelligence of T2I models, covering two key aspects: (1) SpatialGenEval involves 1,230 long, information-dense prompts across 25 real-world scenes. Each prompt integrates 10 spatial sub-domains and corresponding 10 multi-choice question-answer pairs, ranging from object position and layout to occlusion and causality. Our extensive evaluation of 21 state-of-the-art models reveals that higher-order spatial reasoning remains a primary bottleneck. (2) To demonstrate that the utility of our information-dense design goes beyond simple evaluation, we also construct the SpatialT2I dataset. It contains 15,400 text-image pairs with rewritten prompts to ensure image consistency while preserving information density. Fine-tuned results on current foundation models (i.e., Stable Diffusion-XL, Uniworld-V1, OmniGen2) yield consistent performance gains (+4.2%, +5.7%, +4.4%) and more realistic effects in spatial relations, highlighting a data-centric paradigm to achieve spatial intelligence in T2I models.
>
---
#### [new 013] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors
- **分类: cs.CV**

- **简介: 该论文属于零样本异常检测任务，旨在无需领域内训练数据检测图像异常。针对现有方法性能不足的问题，提出AnomalyVFM框架，结合合成数据生成与参数高效适配，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.20524v1](https://arxiv.org/pdf/2601.20524v1)**

> **作者:** Matic Fučka; Vitjan Zavrtanik; Danijel Skočaj
>
> **摘要:** Zero-shot anomaly detection aims to detect and localise abnormal regions in the image without access to any in-domain training images. While recent approaches leverage vision-language models (VLMs), such as CLIP, to transfer high-level concept knowledge, methods based on purely vision foundation models (VFMs), like DINOv2, have lagged behind in performance. We argue that this gap stems from two practical issues: (i) limited diversity in existing auxiliary anomaly detection datasets and (ii) overly shallow VFM adaptation strategies. To address both challenges, we propose AnomalyVFM, a general and effective framework that turns any pretrained VFM into a strong zero-shot anomaly detector. Our approach combines a robust three-stage synthetic dataset generation scheme with a parameter-efficient adaptation mechanism, utilising low-rank feature adapters and a confidence-weighted pixel loss. Together, these components enable modern VFMs to substantially outperform current state-of-the-art methods. More specifically, with RADIO as a backbone, AnomalyVFM achieves an average image-level AUROC of 94.1% across 9 diverse datasets, surpassing previous methods by significant 3.3 percentage points. Project Page: https://maticfuc.github.io/anomaly_vfm/
>
---
#### [new 014] BLENDER: Blended Text Embeddings and Diffusion Residuals for Intra-Class Image Synthesis in Deep Metric Learning
- **分类: cs.CV**

- **简介: 该论文属于深度度量学习（DML）任务，旨在提升类别内图像合成的多样性。通过引入BLenDeR方法，利用扩散残差的集合操作，实现可控的属性组合生成，解决现有生成方法的局限性。**

- **链接: [https://arxiv.org/pdf/2601.20246v1](https://arxiv.org/pdf/2601.20246v1)**

> **作者:** Jan Niklas Kolf; Ozan Tezcan; Justin Theiss; Hyung Jun Kim; Wentao Bao; Bhargav Bhushanam; Khushi Gupta; Arun Kejariwal; Naser Damer; Fadi Boutros
>
> **摘要:** The rise of Deep Generative Models (DGM) has enabled the generation of high-quality synthetic data. When used to augment authentic data in Deep Metric Learning (DML), these synthetic samples enhance intra-class diversity and improve the performance of downstream DML tasks. We introduce BLenDeR, a diffusion sampling method designed to increase intra-class diversity for DML in a controllable way by leveraging set-theory inspired union and intersection operations on denoising residuals. The union operation encourages any attribute present across multiple prompts, while the intersection extracts the common direction through a principal component surrogate. These operations enable controlled synthesis of diverse attribute combinations within each class, addressing key limitations of existing generative approaches. Experiments on standard DML benchmarks demonstrate that BLenDeR consistently outperforms state-of-the-art baselines across multiple datasets and backbones. Specifically, BLenDeR achieves 3.7% increase in Recall@1 on CUB-200 and a 1.8% increase on Cars-196, compared to state-of-the-art baselines under standard experimental settings.
>
---
#### [new 015] Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective
- **分类: cs.CV**

- **简介: 该论文属于自然语言生成任务，解决dMLLMs中的重复文本问题。通过分析信息流，提出CoTA方法增强上下文注意力并减少重复。**

- **链接: [https://arxiv.org/pdf/2601.20520v1](https://arxiv.org/pdf/2601.20520v1)**

> **作者:** Qiyan Zhao; Xiaofeng Zhang; Shuochen Chang; Qianyu Chen; Xiaosong Yuan; Xuhang Chen; Luoqi Liu; Jiajun Zhang; Xu-Yao Zhang; Da-Han Wang
>
> **备注:** Accepted in ICLR 2026
>
> **摘要:** Recent diffusion-based Multimodal Large Language Models (dMLLMs) suffer from high inference latency and therefore rely on caching techniques to accelerate decoding. However, the application of cache mechanisms often introduces undesirable repetitive text generation, a phenomenon we term the \textbf{Repeat Curse}. To better investigate underlying mechanism behind this issue, we analyze repetition generation through the lens of information flow. Our work reveals three key findings: (1) context tokens aggregate semantic information as anchors and guide the final predictions; (2) as information propagates across layers, the entropy of context tokens converges in deeper layers, reflecting the model's growing prediction certainty; (3) Repetition is typically linked to disruptions in the information flow of context tokens and to the inability of their entropy to converge in deeper layers. Based on these insights, we present \textbf{CoTA}, a plug-and-play method for mitigating repetition. CoTA enhances the attention of context tokens to preserve intrinsic information flow patterns, while introducing a penalty term to the confidence score during decoding to avoid outputs driven by uncertain context tokens. With extensive experiments, CoTA demonstrates significant effectiveness in alleviating repetition and achieves consistent performance improvements on general tasks. Code is available at https://github.com/ErikZ719/CoTA
>
---
#### [new 016] MARE: Multimodal Alignment and Reinforcement for Explainable Deepfake Detection via Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在提升检测的准确性与可解释性。提出MARE方法，结合多模态对齐与强化学习，增强视觉-语言模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.20433v1](https://arxiv.org/pdf/2601.20433v1)**

> **作者:** Wenbo Xu; Wei Lu; Xiangyang Luo; Jiantao Zhou
>
> **摘要:** Deepfake detection is a widely researched topic that is crucial for combating the spread of malicious content, with existing methods mainly modeling the problem as classification or spatial localization. The rapid advancements in generative models impose new demands on Deepfake detection. In this paper, we propose multimodal alignment and reinforcement for explainable Deepfake detection via vision-language models, termed MARE, which aims to enhance the accuracy and reliability of Vision-Language Models (VLMs) in Deepfake detection and reasoning. Specifically, MARE designs comprehensive reward functions, incorporating reinforcement learning from human feedback (RLHF), to incentivize the generation of text-spatially aligned reasoning content that adheres to human preferences. Besides, MARE introduces a forgery disentanglement module to capture intrinsic forgery traces from high-level facial semantics, thereby improving its authenticity detection capability. We conduct thorough evaluations on the reasoning content generated by MARE. Both quantitative and qualitative experimental results demonstrate that MARE achieves state-of-the-art performance in terms of accuracy and reliability.
>
---
#### [new 017] OS-Marathon: Benchmarking Computer-Use Agents on Long-Horizon Repetitive Tasks
- **分类: cs.CV**

- **简介: 该论文提出OS-Marathon基准，用于评估计算机使用代理在长周期重复任务中的表现，解决缺乏有效评估标准的问题，通过少量示例训练代理完成大规模任务。**

- **链接: [https://arxiv.org/pdf/2601.20650v1](https://arxiv.org/pdf/2601.20650v1)**

> **作者:** Jing Wu; Daphne Barretto; Yiye Chen; Nicholas Gydé; Yanan Jian; Yuhang He; Vibhav Vineet
>
> **备注:** 22 Pages, Project Page: \url{https://os-marathon.github.io/}
>
> **摘要:** Long-horizon, repetitive workflows are common in professional settings, such as processing expense reports from receipts and entering student grades from exam papers. These tasks are often tedious for humans since they can extend to extreme lengths proportional to the size of the data to process. However, they are ideal for Computer-Use Agents (CUAs) due to their structured, recurring sub-workflows with logic that can be systematically learned. Identifying the absence of an evaluation benchmark as a primary bottleneck, we establish OS-Marathon, comprising 242 long-horizon, repetitive tasks across 2 domains to evaluate state-of-the-art (SOTA) agents. We then introduce a cost-effective method to construct a condensed demonstration using only few-shot examples to teach agents the underlying workflow logic, enabling them to execute similar workflows effectively on larger, unseen data collections. Extensive experiments demonstrate both the inherent challenges of these tasks and the effectiveness of our proposed method. Project website: https://os-marathon.github.io/.
>
---
#### [new 018] Hallucination Begins Where Saliency Drops
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决 hallucination 问题。通过分析注意力与梯度信号，提出框架提升输出的视觉一致性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.20279v1](https://arxiv.org/pdf/2601.20279v1)**

> **作者:** Xiaofeng Zhang; Yuanchao Zhu; Chaochen Gu; Xiaosong Yuan; Qiyan Zhao; Jiawei Cao; Feilong Tang; Sinan Fan; Yaomin Shen; Chen Shen; Hao Tang
>
> **备注:** Accepted in ICLR 2026
>
> **摘要:** Recent studies have examined attention dynamics in large vision-language models (LVLMs) to detect hallucinations. However, existing approaches remain limited in reliably distinguishing hallucinated from factually grounded outputs, as they rely solely on forward-pass attention patterns and neglect gradient-based signals that reveal how token influence propagates through the network. To bridge this gap, we introduce LVLMs-Saliency, a gradient-aware diagnostic framework that quantifies the visual grounding strength of each output token by fusing attention weights with their input gradients. Our analysis uncovers a decisive pattern: hallucinations frequently arise when preceding output tokens exhibit low saliency toward the prediction of the next token, signaling a breakdown in contextual memory retention. Leveraging this insight, we propose a dual-mechanism inference-time framework to mitigate hallucinations: (1) Saliency-Guided Rejection Sampling (SGRS), which dynamically filters candidate tokens during autoregressive decoding by rejecting those whose saliency falls below a context-adaptive threshold, thereby preventing coherence-breaking tokens from entering the output sequence; and (2) Local Coherence Reinforcement (LocoRE), a lightweight, plug-and-play module that strengthens attention from the current token to its most recent predecessors, actively counteracting the contextual forgetting behavior identified by LVLMs-Saliency. Extensive experiments across multiple LVLMs demonstrate that our method significantly reduces hallucination rates while preserving fluency and task performance, offering a robust and interpretable solution for enhancing model reliability. Code is available at: https://github.com/zhangbaijin/LVLMs-Saliency
>
---
#### [new 019] Say Cheese! Detail-Preserving Portrait Collection Generation via Natural Language Edits
- **分类: cs.CV**

- **简介: 该论文提出PCG任务，解决通过自然语言编辑生成高质量、连贯的肖像集问题。构建了CHEESE数据集，并提出SCheese框架以保持身份和细节一致性。**

- **链接: [https://arxiv.org/pdf/2601.20511v1](https://arxiv.org/pdf/2601.20511v1)**

> **作者:** Zelong Sun; Jiahui Wu; Ying Ba; Dong Jing; Zhiwu Lu
>
> **摘要:** As social media platforms proliferate, users increasingly demand intuitive ways to create diverse, high-quality portrait collections. In this work, we introduce Portrait Collection Generation (PCG), a novel task that generates coherent portrait collections by editing a reference portrait image through natural language instructions. This task poses two unique challenges to existing methods: (1) complex multi-attribute modifications such as pose, spatial layout, and camera viewpoint; and (2) high-fidelity detail preservation including identity, clothing, and accessories. To address these challenges, we propose CHEESE, the first large-scale PCG dataset containing 24K portrait collections and 573K samples with high-quality modification text annotations, constructed through an Large Vison-Language Model-based pipeline with inversion-based verification. We further propose SCheese, a framework that combines text-guided generation with hierarchical identity and detail preservation. SCheese employs adaptive feature fusion mechanism to maintain identity consistency, and ConsistencyNet to inject fine-grained features for detail consistency. Comprehensive experiments validate the effectiveness of CHEESE in advancing PCG, with SCheese achieving state-of-the-art performance.
>
---
#### [new 020] Artifact-Aware Evaluation for High-Quality Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成质量评估任务，旨在解决现有评价方法无法精细定位和分类生成视频中的伪影问题。工作包括构建大规模标注数据集GenVID和提出DVAR框架以提升伪影检测精度。**

- **链接: [https://arxiv.org/pdf/2601.20297v1](https://arxiv.org/pdf/2601.20297v1)**

> **作者:** Chen Zhu; Jiashu Zhu; Yanxun Li; Meiqi Wu; Bingze Song; Chubin Chen; Jiahong Wu; Xiangxiang Chu; Yangang Wang
>
> **摘要:** With the rapid advancement of video generation techniques, evaluating and auditing generated videos has become increasingly crucial. Existing approaches typically offer coarse video quality scores, lacking detailed localization and categorization of specific artifacts. In this work, we introduce a comprehensive evaluation protocol focusing on three key aspects affecting human perception: Appearance, Motion, and Camera. We define these axes through a taxonomy of 10 prevalent artifact categories reflecting common generative failures observed in video generation. To enable robust artifact detection and categorization, we introduce GenVID, a large-scale dataset of 80k videos generated by various state-of-the-art video generation models, each carefully annotated for the defined artifact categories. Leveraging GenVID, we develop DVAR, a Dense Video Artifact Recognition framework for fine-grained identification and classification of generative artifacts. Extensive experiments show that our approach significantly improves artifact detection accuracy and enables effective filtering of low-quality content.
>
---
#### [new 021] Compression Tells Intelligence: Visual Coding, Visual Token Technology, and the Unification
- **分类: cs.CV**

- **简介: 本文探讨视觉编码与视觉令牌技术的统一，旨在解决压缩效率与模型性能的平衡问题。通过分析两者优化机制，提出统一框架，推动下一代视觉编解码和令牌技术发展。**

- **链接: [https://arxiv.org/pdf/2601.20742v1](https://arxiv.org/pdf/2601.20742v1)**

> **作者:** Xin Jin; Jinming Liu; Yuntao Wei; Junyan Lin; Zhicheng Wang; Jianguo Huang; Xudong Yang; Yanxiao Liu; Wenjun Zeng
>
> **摘要:** "Compression Tells Intelligence", is supported by research in artificial intelligence, particularly concerning (multimodal) large language models (LLMs/MLLMs), where compression efficiency often correlates with improved model performance and capabilities. For compression, classical visual coding based on traditional information theory has developed over decades, achieving great success with numerous international industrial standards widely applied in multimedia (e.g., image/video) systems. Except that, the recent emergingvisual token technology of generative multi-modal large models also shares a similar fundamental objective like visual coding: maximizing semantic information fidelity during the representation learning while minimizing computational cost. Therefore, this paper provides a comprehensive overview of two dominant technique families first -- Visual Coding and Vision Token Technology -- then we further unify them from the aspect of optimization, discussing the essence of compression efficiency and model performance trade-off behind. Next, based on the proposed unified formulation bridging visual coding andvisual token technology, we synthesize bidirectional insights of themselves and forecast the next-gen visual codec and token techniques. Last but not least, we experimentally show a large potential of the task-oriented token developments in the more practical tasks like multimodal LLMs (MLLMs), AI-generated content (AIGC), and embodied AI, as well as shedding light on the future possibility of standardizing a general token technology like the traditional codecs (e.g., H.264/265) with high efficiency for a wide range of intelligent tasks in a unified and effective manner.
>
---
#### [new 022] Open-Vocabulary Functional 3D Human-Scene Interaction Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D人-场景交互生成任务，解决功能性和物理合理性问题。通过功能感知的接触推理和优化，生成符合任务描述的合理交互。**

- **链接: [https://arxiv.org/pdf/2601.20835v1](https://arxiv.org/pdf/2601.20835v1)**

> **作者:** Jie Liu; Yu Sun; Alpar Cseke; Yao Feng; Nicolas Heron; Michael J. Black; Yan Zhang
>
> **备注:** 18 pages
>
> **摘要:** Generating 3D humans that functionally interact with 3D scenes remains an open problem with applications in embodied AI, robotics, and interactive content creation. The key challenge involves reasoning about both the semantics of functional elements in 3D scenes and the 3D human poses required to achieve functionality-aware interaction. Unfortunately, existing methods typically lack explicit reasoning over object functionality and the corresponding human-scene contact, resulting in implausible or functionally incorrect interactions. In this work, we propose FunHSI, a training-free, functionality-driven framework that enables functionally correct human-scene interactions from open-vocabulary task prompts. Given a task prompt, FunHSI performs functionality-aware contact reasoning to identify functional scene elements, reconstruct their 3D geometry, and model high-level interactions via a contact graph. We then leverage vision-language models to synthesize a human performing the task in the image and estimate proposed 3D body and hand poses. Finally, the proposed 3D body configuration is refined via stage-wise optimization to ensure physical plausibility and functional correctness. In contrast to existing methods, FunHSI not only synthesizes more plausible general 3D interactions, such as "sitting on a sofa'', while supporting fine-grained functional human-scene interactions, e.g., "increasing the room temperature''. Extensive experiments demonstrate that FunHSI consistently generates functionally correct and physically plausible human-scene interactions across diverse indoor and outdoor scenes.
>
---
#### [new 023] TPGDiff: Hierarchical Triple-Prior Guided Diffusion for Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决单一模型处理多种退化类型困难的问题。提出TPGDiff网络，融合退化、结构和语义先验，提升修复效果。**

- **链接: [https://arxiv.org/pdf/2601.20306v1](https://arxiv.org/pdf/2601.20306v1)**

> **作者:** Yanjie Tu; Qingsen Yan; Axi Niu; Jiacong Tang
>
> **摘要:** All-in-one image restoration aims to address diverse degradation types using a single unified model. Existing methods typically rely on degradation priors to guide restoration, yet often struggle to reconstruct content in severely degraded regions. Although recent works leverage semantic information to facilitate content generation, integrating it into the shallow layers of diffusion models often disrupts spatial structures (\emph{e.g.}, blurring artifacts). To address this issue, we propose a Triple-Prior Guided Diffusion (TPGDiff) network for unified image restoration. TPGDiff incorporates degradation priors throughout the diffusion trajectory, while introducing structural priors into shallow layers and semantic priors into deep layers, enabling hierarchical and complementary prior guidance for image reconstruction. Specifically, we leverage multi-source structural cues as structural priors to capture fine-grained details and guide shallow layers representations. To complement this design, we further develop a distillation-driven semantic extractor that yields robust semantic priors, ensuring reliable high-level guidance at deep layers even under severe degradations. Furthermore, a degradation extractor is employed to learn degradation-aware priors, enabling stage-adaptive control of the diffusion process across all timesteps. Extensive experiments on both single- and multi-degradation benchmarks demonstrate that TPGDiff achieves superior performance and generalization across diverse restoration scenarios. Our project page is: https://leoyjtu.github.io/tpgdiff-project.
>
---
#### [new 024] Size Matters: Reconstructing Real-Scale 3D Models from Monocular Images for Food Portion Estimation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于3D重建任务，旨在从单目图像中恢复真实尺度的3D模型，以准确估算食物摄入量。通过学习视觉特征估计尺度，提升营养评估精度。**

- **链接: [https://arxiv.org/pdf/2601.20051v1](https://arxiv.org/pdf/2601.20051v1)**

> **作者:** Gautham Vinod; Bruce Coburn; Siddeshwar Raghavan; Jiangpeng He; Fengqing Zhu
>
> **摘要:** The rise of chronic diseases related to diet, such as obesity and diabetes, emphasizes the need for accurate monitoring of food intake. While AI-driven dietary assessment has made strides in recent years, the ill-posed nature of recovering size (portion) information from monocular images for accurate estimation of ``how much did you eat?'' is a pressing challenge. Some 3D reconstruction methods have achieved impressive geometric reconstruction but fail to recover the crucial real-world scale of the reconstructed object, limiting its usage in precision nutrition. In this paper, we bridge the gap between 3D computer vision and digital health by proposing a method that recovers a true-to-scale 3D reconstructed object from a monocular image. Our approach leverages rich visual features extracted from models trained on large-scale datasets to estimate the scale of the reconstructed object. This learned scale enables us to convert single-view 3D reconstructions into true-to-life, physically meaningful models. Extensive experiments and ablation studies on two publicly available datasets show that our method consistently outperforms existing techniques, achieving nearly a 30% reduction in mean absolute volume-estimation error, showcasing its potential to enhance the domain of precision nutrition. Code: https://gitlab.com/viper-purdue/size-matters
>
---
#### [new 025] Sparse CLIP: Co-Optimizing Interpretability and Performance in Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言表示学习任务，旨在解决CLIP模型可解释性与性能之间的矛盾。通过在训练中引入稀疏性，提升模型的可解释性同时保持高性能。**

- **链接: [https://arxiv.org/pdf/2601.20075v1](https://arxiv.org/pdf/2601.20075v1)**

> **作者:** Chuan Qin; Constantin Venhoff; Sonia Joseph; Fanyi Xiao; Stefan Scherer
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) has become a cornerstone in vision-language representation learning, powering diverse downstream tasks and serving as the default vision backbone in multimodal large language models (MLLMs). Despite its success, CLIP's dense and opaque latent representations pose significant interpretability challenges. A common assumption is that interpretability and performance are in tension: enforcing sparsity during training degrades accuracy, motivating recent post-hoc approaches such as Sparse Autoencoders (SAEs). However, these post-hoc approaches often suffer from degraded downstream performance and loss of CLIP's inherent multimodal capabilities, with most learned features remaining unimodal. We propose a simple yet effective approach that integrates sparsity directly into CLIP training, yielding representations that are both interpretable and performant. Compared to SAEs, our Sparse CLIP representations preserve strong downstream task performance, achieve superior interpretability, and retain multimodal capabilities. We show that multimodal sparse features enable straightforward semantic concept alignment and reveal training dynamics of how cross-modal knowledge emerges. Finally, as a proof of concept, we train a vision-language model on sparse CLIP representations that enables interpretable, vision-based steering capabilities. Our findings challenge conventional wisdom that interpretability requires sacrificing accuracy and demonstrate that interpretability and performance can be co-optimized, offering a promising design principle for future models.
>
---
#### [new 026] Latent Temporal Discrepancy as Motion Prior: A Loss-Weighting Strategy for Dynamic Fidelity in T2V
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决动态视频生成中运动质量下降的问题。通过引入LTD作为运动先验，优化损失权重，提升模型对复杂动态的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2601.20504v1](https://arxiv.org/pdf/2601.20504v1)**

> **作者:** Meiqi Wu; Bingze Song; Ruimin Lin; Chen Zhu; Xiaokun Feng; Jiahong Wu; Xiangxiang Chu; Kaiqi Huang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Video generation models have achieved notable progress in static scenarios, yet their performance in motion video generation remains limited, with quality degrading under drastic dynamic changes. This is due to noise disrupting temporal coherence and increasing the difficulty of learning dynamic regions. {Unfortunately, existing diffusion models rely on static loss for all scenarios, constraining their ability to capture complex dynamics.} To address this issue, we introduce Latent Temporal Discrepancy (LTD) as a motion prior to guide loss weighting. LTD measures frame-to-frame variation in the latent space, assigning larger penalties to regions with higher discrepancy while maintaining regular optimization for stable regions. This motion-aware strategy stabilizes training and enables the model to better reconstruct high-frequency dynamics. Extensive experiments on the general benchmark VBench and the motion-focused VMBench show consistent gains, with our method outperforming strong baselines by 3.31% on VBench and 3.58% on VMBench, achieving significant improvements in motion quality.
>
---
#### [new 027] MMSF: Multitask and Multimodal Supervised Framework for WSI Classification and Survival Analysis
- **分类: cs.CV**

- **简介: 该论文提出MMSF框架，用于WSI分类和生存分析，解决多模态数据融合难题，通过特征提取、标准化和融合提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20347v1](https://arxiv.org/pdf/2601.20347v1)**

> **作者:** Chengying She; Chengwei Chen; Xinran Zhang; Ben Wang; Lizhuang Liu; Chengwei Shao; Yun Bian
>
> **备注:** Submitted to "Biomedical Signal Processing and Control"
>
> **摘要:** Multimodal evidence is critical in computational pathology: gigapixel whole slide images capture tumor morphology, while patient-level clinical descriptors preserve complementary context for prognosis. Integrating such heterogeneous signals remains challenging because feature spaces exhibit distinct statistics and scales. We introduce MMSF, a multitask and multimodal supervised framework built on a linear-complexity MIL backbone that explicitly decomposes and fuses cross-modal information. MMSF comprises a graph feature extraction module embedding tissue topology at the patch level, a clinical data embedding module standardizing patient attributes, a feature fusion module aligning modality-shared and modality-specific representations, and a Mamba-based MIL encoder with multitask prediction heads. Experiments on CAMELYON16 and TCGA-NSCLC demonstrate 2.1--6.6\% accuracy and 2.2--6.9\% AUC improvements over competitive baselines, while evaluations on five TCGA survival cohorts yield 7.1--9.8\% C-index improvements compared with unimodal methods and 5.6--7.1\% over multimodal alternatives.
>
---
#### [new 028] bi-modal textual prompt learning for vision-language models in remote sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感领域的视觉-语言模型任务，旨在解决Prompt Learning在遥感数据中泛化能力不足的问题。提出BiMoRS框架，结合文本和视觉信息提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20675v1](https://arxiv.org/pdf/2601.20675v1)**

> **作者:** Pankhi Kashyap; Mainak Singha; Biplab Banerjee
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** Prompt learning (PL) has emerged as an effective strategy to adapt vision-language models (VLMs), such as CLIP, for downstream tasks under limited supervision. While PL has demonstrated strong generalization on natural image datasets, its transferability to remote sensing (RS) imagery remains underexplored. RS data present unique challenges, including multi-label scenes, high intra-class variability, and diverse spatial resolutions, that hinder the direct applicability of existing PL methods. In particular, current prompt-based approaches often struggle to identify dominant semantic cues and fail to generalize to novel classes in RS scenarios. To address these challenges, we propose BiMoRS, a lightweight bi-modal prompt learning framework tailored for RS tasks. BiMoRS employs a frozen image captioning model (e.g., BLIP-2) to extract textual semantic summaries from RS images. These captions are tokenized using a BERT tokenizer and fused with high-level visual features from the CLIP encoder. A lightweight cross-attention module then conditions a learnable query prompt on the fused textual-visual representation, yielding contextualized prompts without altering the CLIP backbone. We evaluate BiMoRS on four RS datasets across three domain generalization (DG) tasks and observe consistent performance gains, outperforming strong baselines by up to 2% on average. Codes are available at https://github.com/ipankhi/BiMoRS.
>
---
#### [new 029] FAIRT2V: Training-Free Debiasing for Text-to-Video Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到视频生成任务，旨在解决T2V模型中的性别偏见问题。通过无训练方式中和提示嵌入，减少偏差，同时保持视频质量与时间连贯性。**

- **链接: [https://arxiv.org/pdf/2601.20791v1](https://arxiv.org/pdf/2601.20791v1)**

> **作者:** Haonan Zhong; Wei Song; Tingxu Han; Maurice Pagnucco; Jingling Xue; Yang Song
>
> **摘要:** Text-to-video (T2V) diffusion models have achieved rapid progress, yet their demographic biases, particularly gender bias, remain largely unexplored. We present FairT2V, a training-free debiasing framework for text-to-video generation that mitigates encoder-induced bias without finetuning. We first analyze demographic bias in T2V models and show that it primarily originates from pretrained text encoders, which encode implicit gender associations even for neutral prompts. We quantify this effect with a gender-leaning score that correlates with bias in generated videos. Based on this insight, FairT2V mitigates demographic bias by neutralizing prompt embeddings via anchor-based spherical geodesic transformations while preserving semantics. To maintain temporal coherence, we apply debiasing only during early identity-forming steps through a dynamic denoising schedule. We further propose a video-level fairness evaluation protocol combining VideoLLM-based reasoning with human verification. Experiments on the modern T2V model Open-Sora show that FairT2V substantially reduces demographic bias across occupations with minimal impact on video quality.
>
---
#### [new 030] Physically Guided Visual Mass Estimation from a Single RGB Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单图像质量估计任务，解决从RGB图像中准确估算物体质量的问题。通过结合几何、语义和外观信息，提出物理引导的框架提升估计精度。**

- **链接: [https://arxiv.org/pdf/2601.20303v1](https://arxiv.org/pdf/2601.20303v1)**

> **作者:** Sungjae Lee; Junhan Jeong; Yeonjoo Hong; Kwang In Kim
>
> **摘要:** Estimating object mass from visual input is challenging because mass depends jointly on geometric volume and material-dependent density, neither of which is directly observable from RGB appearance. Consequently, mass prediction from pixels is ill-posed and therefore benefits from physically meaningful representations to constrain the space of plausible solutions. We propose a physically structured framework for single-image mass estimation that addresses this ambiguity by aligning visual cues with the physical factors governing mass. From a single RGB image, we recover object-centric three-dimensional geometry via monocular depth estimation to inform volume and extract coarse material semantics using a vision-language model to guide density-related reasoning. These geometry, semantic, and appearance representations are fused through an instance-adaptive gating mechanism, and two physically guided latent factors (volume- and density-related) are predicted through separate regression heads under mass-only supervision. Experiments on image2mass and ABO-500 show that the proposed method consistently outperforms state-of-the-art methods.
>
---
#### [new 031] DeepSeek-OCR 2: Visual Causal Flow
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决传统模型固定处理图像顺序的问题。提出DeepEncoder V2，通过因果推理动态重组视觉令牌，实现更符合人类视觉的2D理解。**

- **链接: [https://arxiv.org/pdf/2601.20552v1](https://arxiv.org/pdf/2601.20552v1)**

> **作者:** Haoran Wei; Yaofeng Sun; Yukun Li
>
> **摘要:** We present DeepSeek-OCR 2 to investigate the feasibility of a novel encoder-DeepEncoder V2-capable of dynamically reordering visual tokens upon image semantics. Conventional vision-language models (VLMs) invariably process visual tokens in a rigid raster-scan order (top-left to bottom-right) with fixed positional encoding when fed into LLMs. However, this contradicts human visual perception, which follows flexible yet semantically coherent scanning patterns driven by inherent logical structures. Particularly for images with complex layouts, human vision exhibits causally-informed sequential processing. Inspired by this cognitive mechanism, DeepEncoder V2 is designed to endow the encoder with causal reasoning capabilities, enabling it to intelligently reorder visual tokens prior to LLM-based content interpretation. This work explores a novel paradigm: whether 2D image understanding can be effectively achieved through two-cascaded 1D causal reasoning structures, thereby offering a new architectural approach with the potential to achieve genuine 2D reasoning. Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR-2.
>
---
#### [new 032] Efficient Autoregressive Video Diffusion with Dummy Head
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，针对自回归视频扩散模型效率问题，提出Dummy Forcing方法优化注意力机制，提升推理速度并保持质量。**

- **链接: [https://arxiv.org/pdf/2601.20499v1](https://arxiv.org/pdf/2601.20499v1)**

> **作者:** Hang Guo; Zhaoyang Jia; Jiahao Li; Bin Li; Yuanhao Cai; Jiangshan Wang; Yawei Li; Yan Lu
>
> **备注:** Technical Report
>
> **摘要:** The autoregressive video diffusion model has recently gained considerable research interest due to its causal modeling and iterative denoising. In this work, we identify that the multi-head self-attention in these models under-utilizes historical frames: approximately 25% heads attend almost exclusively to the current frame, and discarding their KV caches incurs only minor performance degradation. Building upon this, we propose Dummy Forcing, a simple yet effective method to control context accessibility across different heads. Specifically, the proposed heterogeneous memory allocation reduces head-wise context redundancy, accompanied by dynamic head programming to adaptively classify head types. Moreover, we develop a context packing technique to achieve more aggressive cache compression. Without additional training, our Dummy Forcing delivers up to 2.0x speedup over the baseline, supporting video generation at 24.3 FPS with less than 0.5% quality drop. Project page is available at https://csguoh.github.io/project/DummyForcing/.
>
---
#### [new 033] DiSa: Saliency-Aware Foreground-Background Disentangled Framework for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义分割任务，解决VLM在分割中的前景偏差和边界模糊问题。提出DiSa框架，分离建模前景与背景特征，并引入层级细化模块提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.20064v1](https://arxiv.org/pdf/2601.20064v1)**

> **作者:** Zhen Yao; Xin Li; Taotao Jing; Shuai Zhang; Mooi Choo Chuah
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Open-vocabulary semantic segmentation aims to assign labels to every pixel in an image based on text labels. Existing approaches typically utilize vision-language models (VLMs), such as CLIP, for dense prediction. However, VLMs, pre-trained on image-text pairs, are biased toward salient, object-centric regions and exhibit two critical limitations when adapted to segmentation: (i) Foreground Bias, which tends to ignore background regions, and (ii) Limited Spatial Localization, resulting in blurred object boundaries. To address these limitations, we introduce DiSa, a novel saliency-aware foreground-background disentangled framework. By explicitly incorporating saliency cues in our designed Saliency-aware Disentanglement Module (SDM), DiSa separately models foreground and background ensemble features in a divide-and-conquer manner. Additionally, we propose a Hierarchical Refinement Module (HRM) that leverages pixel-wise spatial contexts and enables channel-wise feature refinement through multi-level updates. Extensive experiments on six benchmarks demonstrate that DiSa consistently outperforms state-of-the-art methods.
>
---
#### [new 034] DiffVC-RT: Towards Practical Real-Time Diffusion-based Perceptual Neural Video Compression
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，解决扩散模型在实时性、信息损失和时间一致性上的问题。提出DiffVC-RT框架，优化架构、增强一致性并设计并行解码管道，实现高效实时压缩。**

- **链接: [https://arxiv.org/pdf/2601.20564v1](https://arxiv.org/pdf/2601.20564v1)**

> **作者:** Wenzhuo Ma; Zhenzhong Chen
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** The practical deployment of diffusion-based Neural Video Compression (NVC) faces critical challenges, including severe information loss, prohibitive inference latency, and poor temporal consistency. To bridge this gap, we propose DiffVC-RT, the first framework designed to achieve real-time diffusion-based perceptual NVC. First, we introduce an Efficient and Informative Model Architecture. Through strategic module replacements and pruning, this architecture significantly reduces computational complexity while mitigating structural information loss. Second, to address generative flickering artifacts, we propose Explicit and Implicit Consistency Modeling. We enhance temporal consistency by explicitly incorporating a zero-cost Online Temporal Shift Module within the U-Net, complemented by hybrid implicit consistency constraints. Finally, we present an Asynchronous and Parallel Decoding Pipeline incorporating Mixed Half Precision, which enables asynchronous latent decoding and parallel frame reconstruction via a Batch-dimension Temporal Shift design. Experiments show that DiffVC-RT achieves 80.1% bitrate savings in terms of LPIPS over VTM-17.0 on HEVC dataset with real-time encoding and decoding speeds of 206 / 30 fps for 720p videos on an NVIDIA H800 GPU, marking a significant milestone in diffusion-based video compression.
>
---
#### [new 035] Reversible Efficient Diffusion for Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于图像融合任务，旨在解决扩散模型在融合过程中细节丢失的问题。提出RED模型，在保持生成能力的同时提升融合效果与效率。**

- **链接: [https://arxiv.org/pdf/2601.20260v1](https://arxiv.org/pdf/2601.20260v1)**

> **作者:** Xingxin Xu; Bing Cao; DongDong Li; Qinghua Hu; Pengfei Zhu
>
> **摘要:** Multi-modal image fusion aims to consolidate complementary information from diverse source images into a unified representation. The fused image is expected to preserve fine details and maintain high visual fidelity. While diffusion models have demonstrated impressive generative capabilities in image generation, they often suffer from detail loss when applied to image fusion tasks. This issue arises from the accumulation of noise errors inherent in the Markov process, leading to inconsistency and degradation in the fused results. However, incorporating explicit supervision into end-to-end training of diffusion-based image fusion introduces challenges related to computational efficiency. To address these limitations, we propose the Reversible Efficient Diffusion (RED) model - an explicitly supervised training framework that inherits the powerful generative capability of diffusion models while avoiding the distribution estimation.
>
---
#### [new 036] Bridging the Applicator Gap with Data-Doping:Dual-Domain Learning for Precise Bladder Segmentation in CT-Guided Brachytherapy
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决CT引导的阴道近距离治疗中膀胱分割因数据分布差异导致的性能下降问题。通过融合不同分布的数据提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.20302v1](https://arxiv.org/pdf/2601.20302v1)**

> **作者:** Suresh Das; Siladittya Manna; Sayantari Ghosh
>
> **摘要:** Performance degradation due to covariate shift remains a major challenge for deep learning models in medical image segmentation. An open question is whether samples from a shifted distribution can effectively support learning when combined with limited target domain data. We investigate this problem in the context of bladder segmentation in CT guided gynecological brachytherapy, a critical task for accurate dose optimization and organ at risk sparing. While CT scans without brachytherapy applicators (no applicator: NA) are widely available, scans with applicators inserted (with applicator: WA) are scarce and exhibit substantial anatomical deformation and imaging artifacts, making automated segmentation particularly difficult. We propose a dual domain learning strategy that integrates NA and WA CT data to improve robustness and generalizability under covariate shift. Using a curated assorted dataset, we show that NA data alone fail to capture the anatomical and artifact related characteristics of WA images. However, introducing a modest proportion of WA data into a predominantly NA training set leads to significant performance improvements. Through systematic experiments across axial, coronal, and sagittal planes using multiple deep learning architectures, we demonstrate that doping only 10 to 30 percent WA data achieves segmentation performance comparable to models trained exclusively on WA data. The proposed approach attains Dice similarity coefficients of up to 0.94 and Intersection over Union scores of up to 0.92, indicating effective domain adaptation and improved clinical reliability. This study highlights the value of integrating anatomically similar but distribution shifted datasets to overcome data scarcity and enhance deep learning based segmentation for brachytherapy treatment planning.
>
---
#### [new 037] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting
- **分类: cs.CV**

- **简介: 该论文提出CPiRi，解决多变量时间序列预测中通道依赖与独立的矛盾问题。通过通道排列不变框架，提升模型适应性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.20318v1](https://arxiv.org/pdf/2601.20318v1)**

> **作者:** Jiyuan Xu; Wenyu Zhang; Xin Jing; Shuai Chen; Shuai Zhang; Jiahao Nie
>
> **备注:** 22 pages, ICLR 2026
>
> **摘要:** Current methods for multivariate time series forecasting can be classified into channel-dependent and channel-independent models. Channel-dependent models learn cross-channel features but often overfit the channel ordering, which hampers adaptation when channels are added or reordered. Channel-independent models treat each channel in isolation to increase flexibility, yet this neglects inter-channel dependencies and limits performance. To address these limitations, we propose \textbf{CPiRi}, a \textbf{channel permutation invariant (CPI)} framework that infers cross-channel structure from data rather than memorizing a fixed ordering, enabling deployment in settings with structural and distributional co-drift without retraining. CPiRi couples \textbf{spatio-temporal decoupling architecture} with \textbf{permutation-invariant regularization training strategy}: a frozen pretrained temporal encoder extracts high-quality temporal features, a lightweight spatial module learns content-driven inter-channel relations, while a channel shuffling strategy enforces CPI during training. We further \textbf{ground CPiRi in theory} by analyzing permutation equivariance in multivariate time series forecasting. Experiments on multiple benchmarks show state-of-the-art results. CPiRi remains stable when channel orders are shuffled and exhibits strong \textbf{inductive generalization} to unseen channels even when trained on \textbf{only half} of the channels, while maintaining \textbf{practical efficiency} on large-scale datasets. The source code is released at https://github.com/JasonStraka/CPiRi.
>
---
#### [new 038] OSDEnhancer: Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于视频超分辨率任务，解决真实场景下时空视频超分辨率问题。提出OSDEnhancer框架，通过单步扩散和混合专家策略提升时空一致性与细节质量。**

- **链接: [https://arxiv.org/pdf/2601.20308v1](https://arxiv.org/pdf/2601.20308v1)**

> **作者:** Shuoyan Wei; Feng Li; Chen Zhou; Runmin Cong; Yao Zhao; Huihui Bai
>
> **备注:** 17 pages, 10 figures. Code will be released upon publication
>
> **摘要:** Diffusion models (DMs) have demonstrated exceptional success in video super-resolution (VSR), showcasing a powerful capacity for generating fine-grained details. However, their potential for space-time video super-resolution (STVSR), which necessitates not only recovering realistic visual content from low-resolution to high-resolution but also improving the frame rate with coherent temporal dynamics, remains largely underexplored. Moreover, existing STVSR methods predominantly address spatiotemporal upsampling under simplified degradation assumptions, which often struggle in real-world scenarios with complex unknown degradations. Such a high demand for reconstruction fidelity and temporal consistency makes the development of a robust STVSR framework particularly non-trivial. To address these challenges, we propose OSDEnhancer, a novel framework that, to the best of our knowledge, represents the first method to achieve real-world STVSR through an efficient one-step diffusion process. OSDEnhancer initializes essential spatiotemporal structures through a linear pre-interpolation strategy and pivots on training temporal refinement and spatial enhancement mixture of experts (TR-SE MoE), which allows distinct expert pathways to progressively learn robust, specialized representations for temporal coherence and spatial detail, further collaboratively reinforcing each other during inference. A bidirectional deformable variational autoencoder (VAE) decoder is further introduced to perform recurrent spatiotemporal aggregation and propagation, enhancing cross-frame reconstruction fidelity. Experiments demonstrate that the proposed method achieves state-of-the-art performance while maintaining superior generalization capability in real-world scenarios.
>
---
#### [new 039] ProSkill: Segment-Level Skill Assessment in Procedural Videos
- **分类: cs.CV**

- **简介: 该论文属于技能评估任务，旨在解决程序性视频中缺乏大规模标注数据的问题。提出ProSkill数据集，支持动作级技能评估，并设计了高效的标注方法。**

- **链接: [https://arxiv.org/pdf/2601.20661v1](https://arxiv.org/pdf/2601.20661v1)**

> **作者:** Michele Mazzamuto; Daniele Di Mauro; Gianpiero Francesca; Giovanni Maria Farinella; Antonino Furnari
>
> **备注:** Accepted at The IEEE/CVF Winter Conference on Applications of Computer Vision 2026
>
> **摘要:** Skill assessment in procedural videos is crucial for the objective evaluation of human performance in settings such as manufacturing and procedural daily tasks. Current research on skill assessment has predominantly focused on sports and lacks large-scale datasets for complex procedural activities. Existing studies typically involve only a limited number of actions, focus on either pairwise assessments (e.g., A is better than B) or on binary labels (e.g., good execution vs needs improvement). In response to these shortcomings, we introduce ProSkill, the first benchmark dataset for action-level skill assessment in procedural tasks. ProSkill provides absolute skill assessment annotations, along with pairwise ones. This is enabled by a novel and scalable annotation protocol that allows for the creation of an absolute skill assessment ranking starting from pairwise assessments. This protocol leverages a Swiss Tournament scheme for efficient pairwise comparisons, which are then aggregated into consistent, continuous global scores using an ELO-based rating system. We use our dataset to benchmark the main state-of-the-art skill assessment algorithms, including both ranking-based and pairwise paradigms. The suboptimal results achieved by the current state-of-the-art highlight the challenges and thus the value of ProSkill in the context of skill assessment for procedural videos. All data and code are available at https://fpv-iplab.github.io/ProSkill/
>
---
#### [new 040] FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在提升视图外推的渲染质量。提出FreeFix方法，在不微调扩散模型的前提下，通过2D-3D协同优化和置信度掩码，提高多帧一致性与精度。**

- **链接: [https://arxiv.org/pdf/2601.20857v1](https://arxiv.org/pdf/2601.20857v1)**

> **作者:** Hongyu Zhou; Zisen Shao; Sheng Miao; Pan Wang; Dongfeng Bai; Bingbing Liu; Yiyi Liao
>
> **备注:** Our project page is at https://xdimlab.github.io/freefix
>
> **摘要:** Neural Radiance Fields and 3D Gaussian Splatting have advanced novel view synthesis, yet still rely on dense inputs and often degrade at extrapolated views. Recent approaches leverage generative models, such as diffusion models, to provide additional supervision, but face a trade-off between generalization and fidelity: fine-tuning diffusion models for artifact removal improves fidelity but risks overfitting, while fine-tuning-free methods preserve generalization but often yield lower fidelity. We introduce FreeFix, a fine-tuning-free approach that pushes the boundary of this trade-off by enhancing extrapolated rendering with pretrained image diffusion models. We present an interleaved 2D-3D refinement strategy, showing that image diffusion models can be leveraged for consistent refinement without relying on costly video diffusion models. Furthermore, we take a closer look at the guidance signal for 2D refinement and propose a per-pixel confidence mask to identify uncertain regions for targeted improvement. Experiments across multiple datasets show that FreeFix improves multi-frame consistency and achieves performance comparable to or surpassing fine-tuning-based methods, while retaining strong generalization ability.
>
---
#### [new 041] Semi-Supervised Masked Autoencoders: Unlocking Vision Transformer Potential with Limited Data
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉Transformer的半监督学习任务，解决标签数据少的问题。通过SSMAE框架，结合伪标签与自编码器，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20072v1](https://arxiv.org/pdf/2601.20072v1)**

> **作者:** Atik Faysal; Mohammad Rostami; Reihaneh Gh. Roshan; Nikhil Muralidhar; Huaxia Wang
>
> **摘要:** We address the challenge of training Vision Transformers (ViTs) when labeled data is scarce but unlabeled data is abundant. We propose Semi-Supervised Masked Autoencoder (SSMAE), a framework that jointly optimizes masked image reconstruction and classification using both unlabeled and labeled samples with dynamically selected pseudo-labels. SSMAE introduces a validation-driven gating mechanism that activates pseudo-labeling only after the model achieves reliable, high-confidence predictions that are consistent across both weakly and strongly augmented views of the same image, reducing confirmation bias. On CIFAR-10 and CIFAR-100, SSMAE consistently outperforms supervised ViT and fine-tuned MAE, with the largest gains in low-label regimes (+9.24% over ViT on CIFAR-10 with 10% labels). Our results demonstrate that when pseudo-labels are introduced is as important as how they are generated for data-efficient transformer training. Codes are available at https://github.com/atik666/ssmae.
>
---
#### [new 042] Let's Roll a BiFTA: Bi-refinement for Fine-grained Text-visual Alignment in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言对齐任务，旨在提升预训练模型的零样本性能。针对文本与图像中冗余信息影响对齐效果的问题，提出BiFTA方法，通过视图和描述精炼去除冗余，提高对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.20419v1](https://arxiv.org/pdf/2601.20419v1)**

> **作者:** Yuhao Sun; Chengyi Cai; Jiacheng Zhang; Zesheng Ye; Xingliang Yuan; Feng Liu
>
> **备注:** 25 pages
>
> **摘要:** Recent research has shown that aligning fine-grained text descriptions with localized image patches can significantly improve the zero-shot performance of pre-trained vision-language models (e.g., CLIP). However, we find that both fine-grained text descriptions and localized image patches often contain redundant information, making text-visual alignment less effective. In this paper, we tackle this issue from two perspectives: \emph{View Refinement} and \emph{Description refinement}, termed as \textit{\textbf{Bi}-refinement for \textbf{F}ine-grained \textbf{T}ext-visual \textbf{A}lignment} (BiFTA). \emph{View refinement} removes redundant image patches with high \emph{Intersection over Union} (IoU) ratios, resulting in more distinctive visual samples. \emph{Description refinement} removes redundant text descriptions with high pairwise cosine similarity, ensuring greater diversity in the remaining descriptions. BiFTA achieves superior zero-shot performance on 6 benchmark datasets for both ViT-based and ResNet-based CLIP, justifying the necessity to remove redundant information in visual-text alignment.
>
---
#### [new 043] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决流匹配模型对齐中的稀疏奖励问题。提出DenseGRPO框架，通过密集奖励评估每一步贡献，并优化探索空间。**

- **链接: [https://arxiv.org/pdf/2601.20218v1](https://arxiv.org/pdf/2601.20218v1)**

> **作者:** Haoyou Deng; Keyu Yan; Chaojie Mao; Xiang Wang; Yu Liu; Changxin Gao; Nong Sang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Recent GRPO-based approaches built on flow matching models have shown remarkable improvements in human preference alignment for text-to-image generation. Nevertheless, they still suffer from the sparse reward problem: the terminal reward of the entire denoising trajectory is applied to all intermediate steps, resulting in a mismatch between the global feedback signals and the exact fine-grained contributions at intermediate denoising steps. To address this issue, we introduce \textbf{DenseGRPO}, a novel framework that aligns human preference with dense rewards, which evaluates the fine-grained contribution of each denoising step. Specifically, our approach includes two key components: (1) we propose to predict the step-wise reward gain as dense reward of each denoising step, which applies a reward model on the intermediate clean images via an ODE-based approach. This manner ensures an alignment between feedback signals and the contributions of individual steps, facilitating effective training; and (2) based on the estimated dense rewards, a mismatch drawback between the uniform exploration setting and the time-varying noise intensity in existing GRPO-based methods is revealed, leading to an inappropriate exploration space. Thus, we propose a reward-aware scheme to calibrate the exploration space by adaptively adjusting a timestep-specific stochasticity injection in the SDE sampler, ensuring a suitable exploration space at all timesteps. Extensive experiments on multiple standard benchmarks demonstrate the effectiveness of the proposed DenseGRPO and highlight the critical role of the valid dense rewards in flow matching model alignment.
>
---
#### [new 044] Test-Time Adaptation for Anomaly Segmentation via Topology-Aware Optimal Transport Chaining
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于异常分割任务，旨在解决分布偏移下的模型适应问题。提出TopoOT框架，结合拓扑分析与最优传输，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.20333v1](https://arxiv.org/pdf/2601.20333v1)**

> **作者:** Ali Zia; Usman Ali; Umer Ramzan; Abdul Rehman; Abdelwahed Khamis; Wei Xiang
>
> **摘要:** Deep topological data analysis (TDA) offers a principled framework for capturing structural invariants such as connectivity and cycles that persist across scales, making it a natural fit for anomaly segmentation (AS). Unlike thresholdbased binarisation, which produces brittle masks under distribution shift, TDA allows anomalies to be characterised as disruptions to global structure rather than local fluctuations. We introduce TopoOT, a topology-aware optimal transport (OT) framework that integrates multi-filtration persistence diagrams (PDs) with test-time adaptation (TTA). Our key innovation is Optimal Transport Chaining, which sequentially aligns PDs across thresholds and filtrations, yielding geodesic stability scores that identify features consistently preserved across scales. These stabilityaware pseudo-labels supervise a lightweight head trained online with OT-consistency and contrastive objectives, ensuring robust adaptation under domain shift. Across standard 2D and 3D anomaly detection benchmarks, TopoOT achieves state-of-the-art performance, outperforming the most competitive methods by up to +24.1% mean F1 on 2D datasets and +10.2% on 3D AS benchmarks.
>
---
#### [new 045] Exploiting the Final Component of Generator Architectures for AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有检测器对未知生成器泛化能力差的问题。通过利用生成器的最终组件污染真实图像，训练检测器提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.20461v1](https://arxiv.org/pdf/2601.20461v1)**

> **作者:** Yanzhu Liu; Xiao Liu; Yuexuan Wang; Mondal Soumik
>
> **摘要:** With the rapid proliferation of powerful image generators, accurate detection of AI-generated images has become essential for maintaining a trustworthy online environment. However, existing deepfake detectors often generalize poorly to images produced by unseen generators. Notably, despite being trained under vastly different paradigms, such as diffusion or autoregressive modeling, many modern image generators share common final architectural components that serve as the last stage for converting intermediate representations into images. Motivated by this insight, we propose to "contaminate" real images using the generator's final component and train a detector to distinguish them from the original real images. We further introduce a taxonomy based on generators' final components and categorize 21 widely used generators accordingly, enabling a comprehensive investigation of our method's generalization capability. Using only 100 samples from each of three representative categories, our detector-fine-tuned on the DINOv3 backbone-achieves an average accuracy of 98.83% across 22 testing sets from unseen generators.
>
---
#### [new 046] TeleStyle: Content-Preserving Style Transfer in Images and Videos
- **分类: cs.CV**

- **简介: 该论文提出TeleStyle，解决图像和视频的风格迁移任务，旨在保持内容不变的同时实现风格转换。通过优化模型和训练策略，提升风格相似性与内容一致性。**

- **链接: [https://arxiv.org/pdf/2601.20175v1](https://arxiv.org/pdf/2601.20175v1)**

> **作者:** Shiwen Zhang; Xiaoyan Yang; Bojia Zi; Haibin Huang; Chi Zhang; Xuelong Li
>
> **摘要:** Content-preserving style transfer, generating stylized outputs based on content and style references, remains a significant challenge for Diffusion Transformers (DiTs) due to the inherent entanglement of content and style features in their internal representations. In this technical report, we present TeleStyle, a lightweight yet effective model for both image and video stylization. Built upon Qwen-Image-Edit, TeleStyle leverages the base model's robust capabilities in content preservation and style customization. To facilitate effective training, we curated a high-quality dataset of distinct specific styles and further synthesized triplets using thousands of diverse, in-the-wild style categories. We introduce a Curriculum Continual Learning framework to train TeleStyle on this hybrid dataset of clean (curated) and noisy (synthetic) triplets. This approach enables the model to generalize to unseen styles without compromising precise content fidelity. Additionally, we introduce a video-to-video stylization module to enhance temporal consistency and visual quality. TeleStyle achieves state-of-the-art performance across three core evaluation metrics: style similarity, content consistency, and aesthetic quality. Code and pre-trained models are available at https://github.com/Tele-AI/TeleStyle
>
---
#### [new 047] GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态讽刺检测任务，旨在解决跨模态语义不一致识别难题。提出GDCNet框架，利用生成的客观描述作为语义锚点，融合视觉与文本信息提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.20618v1](https://arxiv.org/pdf/2601.20618v1)**

> **作者:** Shuguang Zhang; Junhong Lian; Guoxin Yu; Baoxun Xu; Xiang Ao
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** Multimodal sarcasm detection (MSD) aims to identify sarcasm within image-text pairs by modeling semantic incongruities across modalities. Existing methods often exploit cross-modal embedding misalignment to detect inconsistency but struggle when visual and textual content are loosely related or semantically indirect. While recent approaches leverage large language models (LLMs) to generate sarcastic cues, the inherent diversity and subjectivity of these generations often introduce noise. To address these limitations, we propose the Generative Discrepancy Comparison Network (GDCNet). This framework captures cross-modal conflicts by utilizing descriptive, factually grounded image captions generated by Multimodal LLMs (MLLMs) as stable semantic anchors. Specifically, GDCNet computes semantic and sentiment discrepancies between the generated objective description and the original text, alongside measuring visual-textual fidelity. These discrepancy features are then fused with visual and textual representations via a gated module to adaptively balance modality contributions. Extensive experiments on MSD benchmarks demonstrate GDCNet's superior accuracy and robustness, establishing a new state-of-the-art on the MMSD2.0 benchmark.
>
---
#### [new 048] NucFuseRank: Dataset Fusion and Performance Ranking for Nuclei Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于 nuclei 实例分割任务，旨在解决数据集不统一导致的模型评估偏差问题。通过融合数据集并提出统一测试与训练集，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.20104v1](https://arxiv.org/pdf/2601.20104v1)**

> **作者:** Nima Torbati; Anastasia Meshcheryakova; Ramona Woitek; Sepideh Hatamikia; Diana Mechtcheriakova; Amirreza Mahbod
>
> **备注:** 31 pages
>
> **摘要:** Nuclei instance segmentation in hematoxylin and eosin (H&E)-stained images plays an important role in automated histological image analysis, with various applications in downstream tasks. While several machine learning and deep learning approaches have been proposed for nuclei instance segmentation, most research in this field focuses on developing new segmentation algorithms and benchmarking them on a limited number of arbitrarily selected public datasets. In this work, rather than focusing on model development, we focused on the datasets used for this task. Based on an extensive literature review, we identified manually annotated, publicly available datasets of H&E-stained images for nuclei instance segmentation and standardized them into a unified input and annotation format. Using two state-of-the-art segmentation models, one based on convolutional neural networks (CNNs) and one based on a hybrid CNN and vision transformer architecture, we systematically evaluated and ranked these datasets based on their nuclei instance segmentation performance. Furthermore, we proposed a unified test set (NucFuse-test) for fair cross-dataset evaluation and a unified training set (NucFuse-train) for improved segmentation performance by merging images from multiple datasets. By evaluating and ranking the datasets, performing comprehensive analyses, generating fused datasets, conducting external validation, and making our implementation publicly available, we provided a new benchmark for training, testing, and evaluating nuclei instance segmentation models on H&E-stained histological images.
>
---
#### [new 049] LEMON: How Well Do MLLMs Perform Temporal Multimodal Understanding on Instructional Videos?
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LEMON基准，用于评估MLLM在教学视频中的时空多模态理解能力，解决其在长时序、知识密集型内容上的性能不足问题。**

- **链接: [https://arxiv.org/pdf/2601.20705v1](https://arxiv.org/pdf/2601.20705v1)**

> **作者:** Zhuang Yu; Lei Shen; Jing Zhao; Shiliang Sun
>
> **摘要:** Recent multimodal large language models (MLLMs) have shown remarkable progress across vision, audio, and language tasks, yet their performance on long-form, knowledge-intensive, and temporally structured educational content remains largely unexplored. To bridge this gap, we introduce LEMON, a Lecture-based Evaluation benchmark for MultimOdal uNderstanding, focusing on STEM lecture videos that require long-horizon reasoning and cross-modal integration. LEMON comprises 2,277 video segments spanning 5 disciplines and 29 courses, with an average duration of 196.1 seconds, yielding 4,181 high-quality QA pairs, including 3,413 multiple-choice and 768 open-ended questions. Distinct from existing video benchmarks, LEMON features: (1) semantic richness and disciplinary density, (2) tightly coupled video-audio-text modalities, (3) explicit temporal and pedagogical structure, and (4) contextually linked multi-turn questioning. It further encompasses six major tasks and twelve subtasks, covering the full cognitive spectrum from perception to reasoning and then to generation. Comprehensive experiments reveal substantial performance gaps across tasks, highlighting that even state-of-the-art MLLMs like GPT-4o struggle with temporal reasoning and instructional prediction. We expect LEMON to serve as an extensible and challenging benchmark for advancing multimodal perception, reasoning, and generation in long-form instructional contents.
>
---
#### [new 050] CURVE: Learning Causality-Inspired Invariant Representations for Robust Scene Understanding via Uncertainty-Guided Regularization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于场景理解任务，旨在解决场景图过拟合于虚假相关的问题。通过引入因果启发的框架CURVE，结合不确定性建模与正则化，学习稳定、稀疏的拓扑结构。**

- **链接: [https://arxiv.org/pdf/2601.20355v1](https://arxiv.org/pdf/2601.20355v1)**

> **作者:** Yue Liang; Jiatong Du; Ziyi Yang; Yanjun Huang; Hong Chen
>
> **摘要:** Scene graphs provide structured abstractions for scene understanding, yet they often overfit to spurious correlations, severely hindering out-of-distribution generalization. To address this limitation, we propose CURVE, a causality-inspired framework that integrates variational uncertainty modeling with uncertainty-guided structural regularization to suppress high-variance, environment-specific relations. Specifically, we apply prototype-conditioned debiasing to disentangle invariant interaction dynamics from environment-dependent variations, promoting a sparse and domain-stable topology. Empirically, we evaluate CURVE in zero-shot transfer and low-data sim-to-real adaptation, verifying its ability to learn domain-stable sparse topologies and provide reliable uncertainty estimates to support risk prediction under distribution shifts.
>
---
#### [new 051] Person Re-ID in 2025: Supervised, Self-Supervised, and Language-Aligned. What Works?
- **分类: cs.CV; cs.AI**

- **简介: 本文研究2025年Person Re-ID任务，比较监督、自监督和语言对齐模型的跨域泛化能力，分析模型优缺点。**

- **链接: [https://arxiv.org/pdf/2601.20598v1](https://arxiv.org/pdf/2601.20598v1)**

> **作者:** Lakshman Balasubramanian
>
> **摘要:** Person Re-Identification (ReID) remains a challenging problem in computer vision. This work reviews various training paradigm and evaluates the robustness of state-of-the-art ReID models in cross-domain applications and examines the role of foundation models in improving generalization through richer, more transferable visual representations. We compare three training paradigms, supervised, self-supervised, and language-aligned models. Through the study the aim is to answer the following questions: Can supervised models generalize in cross-domain scenarios? How does foundation models like SigLIP2 perform for the ReID tasks? What are the weaknesses of current supervised and foundational models for ReID? We have conducted the analysis across 11 models and 9 datasets. Our results show a clear split: supervised models dominate their training domain but crumble on cross-domain data. Language-aligned models, however, show surprising robustness cross-domain for ReID tasks, even though they are not explicitly trained to do so. Code and data available at: https://github.com/moiiai-tech/object-reid-benchmark.
>
---
#### [new 052] Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的感知与轨迹预测任务，解决模块化系统信息受限和误差放大问题。提出Li-ViP3D++框架，通过查询空间融合相机与LiDAR数据，提升检测与预测性能。**

- **链接: [https://arxiv.org/pdf/2601.20720v1](https://arxiv.org/pdf/2601.20720v1)**

> **作者:** Matej Halinkovic; Nina Masarykova; Alexey Vinel; Marek Galinski
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** End-to-end perception and trajectory prediction from raw sensor data is one of the key capabilities for autonomous driving. Modular pipelines restrict information flow and can amplify upstream errors. Recent query-based, fully differentiable perception-and-prediction (PnP) models mitigate these issues, yet the complementarity of cameras and LiDAR in the query-space has not been sufficiently explored. Models often rely on fusion schemes that introduce heuristic alignment and discrete selection steps which prevent full utilization of available information and can introduce unwanted bias. We propose Li-ViP3D++, a query-based multimodal PnP framework that introduces Query-Gated Deformable Fusion (QGDF) to integrate multi-view RGB and LiDAR in query space. QGDF (i) aggregates image evidence via masked attention across cameras and feature levels, (ii) extracts LiDAR context through fully differentiable BEV sampling with learned per-query offsets, and (iii) applies query-conditioned gating to adaptively weight visual and geometric cues per agent. The resulting architecture jointly optimizes detection, tracking, and multi-hypothesis trajectory forecasting in a single end-to-end model. On nuScenes, Li-ViP3D++ improves end-to-end behavior and detection quality, achieving higher EPA (0.335) and mAP (0.502) while substantially reducing false positives (FP ratio 0.147), and it is faster than the prior Li-ViP3D variant (139.82 ms vs. 145.91 ms). These results indicate that query-space, fully differentiable camera-LiDAR fusion can increase robustness of end-to-end PnP without sacrificing deployability.
>
---
#### [new 053] Feature Projection Learning for Better Vision-Language Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言推理任务，旨在解决CLIP模型在下游任务中适应效率低的问题。提出FPL方法，通过特征投影提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.20224v1](https://arxiv.org/pdf/2601.20224v1)**

> **作者:** Yi Zhang; Weicheng Lin; Liang-Jie Zhang
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Vision-Language Pre-Trained models, notably CLIP, that utilize contrastive learning have proven highly adept at extracting generalizable visual features. To inherit the well-learned knowledge of VLP models for downstream tasks, several approaches aim to adapt them efficiently with limited supervision. However, these methods either suffer from limited performance, excessive learnable parameters, or extended training times, all of which hinder their effectiveness in adapting the CLIP model to downstream tasks. In this work, we propose a simple yet efficient and effective method called \textit{\textbf{F}eature \textbf{P}rojection \textbf{L}earning(FPL)} to address these problems. Specifically, we develop a projection model that projects class prototype features into the query image feature space and reconstructs the query image feature map. The negative average squared reconstruction error is used as the class score. In this way, we transform the classification problem into a feature projection problem. The final output of this method is a combination of the prediction from the projection model and the original pre-trained CLIP. Comprehensive empirical evaluations confirm that FPL delivers superior accuracy, surpassing the current state-of-the-art methods by a substantial margin.
>
---
#### [new 054] StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval
- **分类: cs.CV**

- **简介: 该论文属于持续文本到视频检索任务，解决模型在持续学习中出现的灾难性遗忘问题。提出StructAlign方法，通过跨模态对齐和关系保持缓解特征漂移。**

- **链接: [https://arxiv.org/pdf/2601.20597v1](https://arxiv.org/pdf/2601.20597v1)**

> **作者:** Shaokun Wang; Weili Guan; Jizhou Han; Jianlong Wu; Yupeng Hu; Liqiang Nie
>
> **摘要:** Continual Text-to-Video Retrieval (CTVR) is a challenging multimodal continual learning setting, where models must incrementally learn new semantic categories while maintaining accurate text-video alignment for previously learned ones, thus making it particularly prone to catastrophic forgetting. A key challenge in CTVR is feature drift, which manifests in two forms: intra-modal feature drift caused by continual learning within each modality, and non-cooperative feature drift across modalities that leads to modality misalignment. To mitigate these issues, we propose StructAlign, a structured cross-modal alignment method for CTVR. First, StructAlign introduces a simplex Equiangular Tight Frame (ETF) geometry as a unified geometric prior to mitigate modality misalignment. Building upon this geometric prior, we design a cross-modal ETF alignment loss that aligns text and video features with category-level ETF prototypes, encouraging the learned representations to form an approximate simplex ETF geometry. In addition, to suppress intra-modal feature drift, we design a Cross-modal Relation Preserving loss, which leverages complementary modalities to preserve cross-modal similarity relations, providing stable relational supervision for feature updates. By jointly addressing non-cooperative feature drift across modalities and intra-modal feature drift, StructAlign effectively alleviates catastrophic forgetting in CTVR. Extensive experiments on benchmark datasets demonstrate that our method consistently outperforms state-of-the-art continual retrieval approaches.
>
---
#### [new 055] CLEAR-Mamba:Towards Accurate, Adaptive and Trustworthy Multi-Sequence Ophthalmic Angiography Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决多模态眼底血管造影分类中的泛化性和可靠性问题。提出CLEAR-Mamba框架，引入自适应模块和可靠性预测机制，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.20601v1](https://arxiv.org/pdf/2601.20601v1)**

> **作者:** Zhuonan Wang; Wenjie Yan; Wenqiao Zhang; Xiaohui Song; Jian Ma; Ke Yao; Yibo Yu; Beng Chin Ooi
>
> **备注:** 10 pages,7 figures
>
> **摘要:** Medical image classification is a core task in computer-aided diagnosis (CAD), playing a pivotal role in early disease detection, treatment planning, and patient prognosis assessment. In ophthalmic practice, fluorescein fundus angiography (FFA) and indocyanine green angiography (ICGA) provide hemodynamic and lesion-structural information that conventional fundus photography cannot capture. However, due to the single-modality nature, subtle lesion patterns, and significant inter-device variability, existing methods still face limitations in generalization and high-confidence prediction. To address these challenges, we propose CLEAR-Mamba, an enhanced framework built upon MedMamba with optimizations in both architecture and training strategy. Architecturally, we introduce HaC, a hypernetwork-based adaptive conditioning layer that dynamically generates parameters according to input feature distributions, thereby improving cross-domain adaptability. From a training perspective, we develop RaP, a reliability-aware prediction scheme built upon evidential uncertainty learning, which encourages the model to emphasize low-confidence samples and improves overall stability and reliability. We further construct a large-scale ophthalmic angiography dataset covering both FFA and ICGA modalities, comprising multiple retinal disease categories for model training and evaluation. Experimental results demonstrate that CLEAR-Mamba consistently outperforms multiple baseline models, including the original MedMamba, across various metrics-showing particular advantages in multi-disease classification and reliability-aware prediction. This study provides an effective solution that balances generalizability and reliability for modality-specific medical image classification tasks.
>
---
#### [new 056] A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于道路表面分类任务，旨在解决现有方法在复杂环境下的泛化能力不足问题。通过引入多模态框架和新数据集ROAD，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.20847v1](https://arxiv.org/pdf/2601.20847v1)**

> **作者:** Willams de Lima Costa; Thifany Ketuli Silva de Souza; Jonas Ferreira Silva; Carlos Gabriel Bezerra Pereira; Bruno Reis Vila Nova; Leonardo Silvino Brito; Rafael Raider Leoni; Juliano Silva; Valter Ferreira; Sibele Miguel Soares Neto; Samantha Uehara; Daniel Giacomo; João Marcelo Teixeira; Veronica Teichrieb; Cristiano Coelho de Araújo
>
> **摘要:** Road surface classification (RSC) is a key enabler for environment-aware predictive maintenance systems. However, existing RSC techniques often fail to generalize beyond narrow operational conditions due to limited sensing modalities and datasets that lack environmental diversity. This work addresses these limitations by introducing a multimodal framework that fuses images and inertial measurements using a lightweight bidirectional cross-attention module followed by an adaptive gating layer that adjusts modality contributions under domain shifts. Given the limitations of current benchmarks, especially regarding lack of variability, we introduce ROAD, a new dataset composed of three complementary subsets: (i) real-world multimodal recordings with RGB-IMU streams synchronized using a gold-standard industry datalogger, captured across diverse lighting, weather, and surface conditions; (ii) a large vision-only subset designed to assess robustness under adverse illumination and heterogeneous capture setups; and (iii) a synthetic subset generated to study out-of-distribution generalization in scenarios difficult to obtain in practice. Experiments show that our method achieves a +1.4 pp improvement over the previous state-of-the-art on the PVS benchmark and an +11.6 pp improvement on our multimodal ROAD subset, with consistently higher F1-scores on minority classes. The framework also demonstrates stable performance across challenging visual conditions, including nighttime, heavy rain, and mixed-surface transitions. These findings indicate that combining affordable camera and IMU sensors with multimodal attention mechanisms provides a scalable, robust foundation for road surface understanding, particularly relevant for regions where environmental variability and cost constraints limit the adoption of high-end sensing suites.
>
---
#### [new 057] HINT: Hierarchical Interaction Modeling for Autoregressive Multi-Human Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于多人体运动生成任务，解决复杂交互下文本驱动的长序列生成问题。提出HINT框架，通过分层交互建模和滑动窗口策略，实现高效、连贯的多人体运动生成。**

- **链接: [https://arxiv.org/pdf/2601.20383v1](https://arxiv.org/pdf/2601.20383v1)**

> **作者:** Mengge Liu; Yan Di; Gu Wang; Yun Qu; Dekai Zhu; Yanyan Li; Xiangyang Ji
>
> **摘要:** Text-driven multi-human motion generation with complex interactions remains a challenging problem. Despite progress in performance, existing offline methods that generate fixed-length motions with a fixed number of agents, are inherently limited in handling long or variable text, and varying agent counts. These limitations naturally encourage autoregressive formulations, which predict future motions step by step conditioned on all past trajectories and current text guidance. In this work, we introduce HINT, the first autoregressive framework for multi-human motion generation with Hierarchical INTeraction modeling in diffusion. First, HINT leverages a disentangled motion representation within a canonicalized latent space, decoupling local motion semantics from inter-person interactions. This design facilitates direct adaptation to varying numbers of human participants without requiring additional refinement. Second, HINT adopts a sliding-window strategy for efficient online generation, and aggregates local within-window and global cross-window conditions to capture past human history, inter-person dependencies, and align with text guidance. This strategy not only enables fine-grained interaction modeling within each window but also preserves long-horizon coherence across all the long sequence. Extensive experiments on public benchmarks demonstrate that HINT matches the performance of strong offline models and surpasses autoregressive baselines. Notably, on InterHuman, HINT achieves an FID of 3.100, significantly improving over the previous state-of-the-art score of 5.154.
>
---
#### [new 058] Automated Marine Biofouling Assessment: Benchmarking Computer Vision and Multimodal LLMs on the Level of Fouling Scale
- **分类: cs.CV**

- **简介: 该论文属于生物污损分类任务，旨在解决传统人工检测效率低的问题。通过计算机视觉和多模态大模型进行自动化评估，提升准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.20196v1](https://arxiv.org/pdf/2601.20196v1)**

> **作者:** Brayden Hamilton; Tim Cashmore; Peter Driscoll; Trevor Gee; Henry Williams
>
> **备注:** Australasian Conference on Robotics and Automation, ACRA2025 13 Pages, 8 Figures
>
> **摘要:** Marine biofouling on vessel hulls poses major ecological, economic, and biosecurity risks. Traditional survey methods rely on diver inspections, which are hazardous and limited in scalability. This work investigates automated classification of biofouling severity on the Level of Fouling (LoF) scale using both custom computer vision models and large multimodal language models (LLMs). Convolutional neural networks, transformer-based segmentation, and zero-shot LLMs were evaluated on an expert-labelled dataset from the New Zealand Ministry for Primary Industries. Computer vision models showed high accuracy at extreme LoF categories but struggled with intermediate levels due to dataset imbalance and image framing. LLMs, guided by structured prompts and retrieval, achieved competitive performance without training and provided interpretable outputs. The results demonstrate complementary strengths across approaches and suggest that hybrid methods integrating segmentation coverage with LLM reasoning offer a promising pathway toward scalable and interpretable biofouling assessment.
>
---
#### [new 059] RepSFNet : A Single Fusion Network with Structural Reparameterization for Crowd Counting
- **分类: cs.CV**

- **简介: 该论文属于人群计数任务，解决密度变化、遮挡和计算成本高的问题。提出RepSFNet，通过结构重参数化和特征融合实现高效准确计数。**

- **链接: [https://arxiv.org/pdf/2601.20369v1](https://arxiv.org/pdf/2601.20369v1)**

> **作者:** Mas Nurul Achmadiah; Chi-Chia Sun; Wen-Kai Kuo; Jun-Wei Hsieh
>
> **备注:** 6 pages. Published in Proceedings of the IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS) 2025
>
> **摘要:** Crowd counting remains challenging in variable-density scenes due to scale variations, occlusions, and the high computational cost of existing models. To address these issues, we propose RepSFNet (Reparameterized Single Fusion Network), a lightweight architecture designed for accurate and real-time crowd estimation. RepSFNet leverages a RepLK-ViT backbone with large reparameterized kernels for efficient multi-scale feature extraction. It further integrates a Feature Fusion module combining Atrous Spatial Pyramid Pooling (ASPP) and Context-Aware Network (CAN) to achieve robust, density-adaptive context modeling. A Concatenate Fusion module is employed to preserve spatial resolution and generate high-quality density maps. By avoiding attention mechanisms and multi-branch designs, RepSFNet significantly reduces parameters and computational complexity. The training objective combines Mean Squared Error and Optimal Transport loss to improve both count accuracy and spatial distribution alignment. Experiments conducted on ShanghaiTech, NWPU, and UCF-QNRF datasets demonstrate that RepSFNet achieves competitive accuracy while reducing inference latency by up to 34 percent compared to recent state-of-the-art methods, making it suitable for real-time and low-power edge computing applications.
>
---
#### [new 060] Structure-constrained Language-informed Diffusion Model for Unpaired Low-dose Computed Tomography Angiography Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于低剂量CT血管造影重建任务，旨在减少碘对比剂用量同时保持诊断效果。通过引入结构约束和语义监督的扩散模型，提升图像重建精度。**

- **链接: [https://arxiv.org/pdf/2601.20304v1](https://arxiv.org/pdf/2601.20304v1)**

> **作者:** Genyuan Zhang; Zihao Wang; Zhifan Gao; Lei Xu; Zhen Zhou; Haijun Yu; Jianjia Zhang; Xiujian Liu; Weiwei Zhang; Shaoyu Wang; Huazhu Fu; Fenglin Liu; Weiwen Wu
>
> **摘要:** The application of iodinated contrast media (ICM) improves the sensitivity and specificity of computed tomography (CT) for a wide range of clinical indications. However, overdose of ICM can cause problems such as kidney damage and life-threatening allergic reactions. Deep learning methods can generate CT images of normal-dose ICM from low-dose ICM, reducing the required dose while maintaining diagnostic power. However, existing methods are difficult to realize accurate enhancement with incompletely paired images, mainly because of the limited ability of the model to recognize specific structures. To overcome this limitation, we propose a Structure-constrained Language-informed Diffusion Model (SLDM), a unified medical generation model that integrates structural synergy and spatial intelligence. First, the structural prior information of the image is effectively extracted to constrain the model inference process, thus ensuring structural consistency in the enhancement process. Subsequently, semantic supervision strategy with spatial intelligence is introduced, which integrates the functions of visual perception and spatial reasoning, thus prompting the model to achieve accurate enhancement. Finally, the subtraction angiography enhancement module is applied, which serves to improve the contrast of the ICM agent region to suitable interval for observation. Qualitative analysis of visual comparison and quantitative results of several metrics demonstrate the effectiveness of our method in angiographic reconstruction for low-dose contrast medium CT angiography.
>
---
#### [new 061] GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决Gaussian深度监督不准确的问题。提出多视角几何约束和渐进式单目深度校准，提升表面重建精度。**

- **链接: [https://arxiv.org/pdf/2601.20331v1](https://arxiv.org/pdf/2601.20331v1)**

> **作者:** Mai Su; Qihan Yu; Zhongtao Wang; Yilong Li; Chengwei Pan; Yisong Chen; Guoping Wang
>
> **摘要:** 3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging. Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors. However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision. To address these limitations, we introduce a Gaussian visibility-aware multi-view geometric consistency constraint that aggregates the visibility of shared Gaussian primitives across views, enabling more accurate and stable geometric supervision. In addition, we propose a progressive quadtree-calibrated Monocular depth constraint that performs block-wise affine calibration from coarse to fine spatial scales, mitigating the scale ambiguity of depth priors while preserving fine-grained surface details. Extensive experiments on DTU and TNT datasets demonstrate consistent improvements in geometric accuracy over prior Gaussian-based and implicit surface reconstruction methods. Codes are available at an anonymous repository: https://github.com/GVGScode/GVGS.
>
---
#### [new 062] Visual Prompt-Agnostic Evolution
- **分类: cs.CV**

- **简介: 该论文属于视觉任务，解决VPT训练不稳定问题。提出PAE方法，通过建模提示动态提升性能，加速收敛并提高准确率。**

- **链接: [https://arxiv.org/pdf/2601.20232v1](https://arxiv.org/pdf/2601.20232v1)**

> **作者:** Junze Wang; Lei Fan; Dezheng Zhang; Weipeng Jing; Donglin Di; Yang Song; Sidong Liu; Cong Cong
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Visual Prompt Tuning (VPT) adapts a frozen Vision Transformer (ViT) to downstream tasks by inserting a small number of learnable prompt tokens into the token sequence at each layer. However, we observe that existing VPT variants often suffer from unstable training dynamics, characterized by gradient oscillations. A layer-wise analysis reveals that shallow-layer prompts tend to stagnate early, while deeper-layer prompts exhibit high-variance oscillations, leading to cross-layer mismatch. These issues slow convergence and degrade final performance. To address these challenges, we propose Prompt-Agnostic Evolution ($\mathtt{PAE}$), which strengthens vision prompt tuning by explicitly modeling prompt dynamics. From a frequency-domain perspective, we initialize prompts in a task-aware direction by uncovering and propagating frequency shortcut patterns that the backbone inherently exploits for recognition. To ensure coherent evolution across layers, we employ a shared Koopman operator that imposes a global linear transformation instead of uncoordinated, layer-specific updates. Finally, inspired by Lyapunov stability theory, we introduce a regularizer that constrains error amplification during evolution. Extensive experiments show that $\mathtt{PAE}$ accelerates convergence with an average $1.41\times$ speedup and improves accuracy by 1--3% on 25 datasets across multiple downstream tasks. Beyond performance, $\mathtt{PAE}$ is prompt-agnostic and lightweight, and it integrates seamlessly with diverse VPT variants without backbone modification or inference-time changes.
>
---
#### [new 063] Towards Compact and Robust DNNs via Compression-aware Sharpness Minimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度学习模型压缩任务，旨在解决模型紧凑性与鲁棒性之间的矛盾。通过引入C-SAM框架，在训练中扰动剪枝掩码，提升模型在剪枝后的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.20301v1](https://arxiv.org/pdf/2601.20301v1)**

> **作者:** Jialuo He; Huangxun Chen
>
> **摘要:** Sharpness-Aware Minimization (SAM) has recently emerged as an effective technique for improving DNN robustness to input variations. However, its interplay with the compactness requirements of on-device DNN deployments remains less explored. Simply pruning a SAM-trained model can undermine robustness, since flatness in the continuous parameter space does not necessarily translate to robustness under the discrete structural changes induced by pruning. Conversely, applying SAM after pruning may be fundamentally constrained by architectural limitations imposed by an early, robustness-agnostic pruning pattern. To address this gap, we propose Compression-aware ShArpness Minimization (C-SAM), a framework that shifts sharpness-aware learning from parameter perturbations to mask perturbations. By explicitly perturbing pruning masks during training, C-SAM promotes a flatter loss landscape with respect to model structure, enabling the discovery of pruning patterns that simultaneously optimize model compactness and robustness to input variations. Extensive experiments on CelebA-HQ, Flowers-102, and CIFAR-10-C across ResNet-18, GoogLeNet, and MobileNet-V2 show that C-SAM consistently achieves higher certified robustness than strong baselines, with improvements of up to 42%, while maintaining task accuracy comparable to the corresponding unpruned models.
>
---
#### [new 064] Quartet of Diffusions: Structure-Aware Point Cloud Generation through Part and Symmetry Guidance
- **分类: cs.CV**

- **简介: 该论文属于3D点云生成任务，解决如何同时建模部件和对称性的生成问题。通过四个协同扩散模型，实现结构一致、可控制的点云生成。**

- **链接: [https://arxiv.org/pdf/2601.20425v1](https://arxiv.org/pdf/2601.20425v1)**

> **作者:** Chenliang Zhou; Fangcheng Zhong; Weihao Xia; Albert Miao; Canberk Baykal; Cengiz Oztireli
>
> **摘要:** We introduce the Quartet of Diffusions, a structure-aware point cloud generation framework that explicitly models part composition and symmetry. Unlike prior methods that treat shape generation as a holistic process or only support part composition, our approach leverages four coordinated diffusion models to learn distributions of global shape latents, symmetries, semantic parts, and their spatial assembly. This structured pipeline ensures guaranteed symmetry, coherent part placement, and diverse, high-quality outputs. By disentangling the generative process into interpretable components, our method supports fine-grained control over shape attributes, enabling targeted manipulation of individual parts while preserving global consistency. A central global latent further reinforces structural coherence across assembled parts. Our experiments show that the Quartet achieves state-of-the-art performance. To our best knowledge, this is the first 3D point cloud generation framework that fully integrates and enforces both symmetry and part priors throughout the generative process.
>
---
#### [new 065] Dual-Modality IoT Framework for Integrated Access Control and Environmental Safety Monitoring with Real-Time Cloud Analytics
- **分类: cs.CV**

- **简介: 该论文属于物联网集成任务，解决传统安全与环境监控系统分离的问题，通过双模态框架实现门禁与环境监测的整合，提升效率与响应速度。**

- **链接: [https://arxiv.org/pdf/2601.20366v1](https://arxiv.org/pdf/2601.20366v1)**

> **作者:** Abdul Hasib; A. S. M. Ahsanul Sarkar Akib; Nihal Das Ankur; Anish Giri
>
> **摘要:** The integration of physical security systems with environmental safety monitoring represents a critical advancement in smart infrastructure management. Traditional approaches maintain these systems as independent silos, creating operational inefficiencies, delayed emergency responses, and increased management complexity. This paper presents a comprehensive dual-modality Internet of Things framework that seamlessly integrates RFID-based access control with multi-sensor environmental safety monitoring through a unified cloud architecture. The system comprises two coordinated subsystems: Subsystem 1 implements RFID authentication with servo-actuated gate control and real-time Google Sheets logging, while Subsystem 2 provides comprehensive safety monitoring incorporating flame detection, water flow measurement, LCD status display, and personnel identification. Both subsystems utilize ESP32 microcontrollers for edge processing and wireless connectivity. Experimental evaluation over 45 days demonstrates exceptional performance metrics: 99.2\% RFID authentication accuracy with 0.82-second average response time, 98.5\% flame detection reliability within 5-meter range, and 99.8\% cloud data logging success rate. The system maintains operational integrity during network disruptions through intelligent local caching mechanisms and achieves total implementation cost of 5,400 BDT (approximately \$48), representing an 82\% reduction compared to commercial integrated solutions. This research establishes a practical framework for synergistic security-safety integration, demonstrating that professional-grade performance can be achieved through careful architectural design and component optimization while maintaining exceptional cost-effectiveness and accessibility for diverse application scenarios.
>
---
#### [new 066] Primitive-Driven Acceleration of Hyperdimensional Computing for Real-Time Image Classification
- **分类: cs.AR; cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决HDC在实时应用中的计算效率问题。通过设计高效编码算法和FPGA加速器，提升HDC的推理速度与性能。**

- **链接: [https://arxiv.org/pdf/2601.20061v1](https://arxiv.org/pdf/2601.20061v1)**

> **作者:** Dhruv Parikh; Jebacyril Arockiaraj; Viktor Prasanna
>
> **摘要:** Hyperdimensional Computing (HDC) represents data using extremely high-dimensional, low-precision vectors, termed hypervectors (HVs), and performs learning and inference through lightweight, noise-tolerant operations. However, the high dimensionality, sparsity, and repeated data movement involved in HDC make these computations difficult to accelerate efficiently on conventional processors. As a result, executing core HDC operations: binding, permutation, bundling, and similarity search: on CPUs or GPUs often leads to suboptimal utilization, memory bottlenecks, and limits on real-time performance. In this paper, our contributions are two-fold. First, we develop an image-encoding algorithm that, similar in spirit to convolutional neural networks, maps local image patches to hypervectors enriched with spatial information. These patch-level hypervectors are then merged into a global representation using the fundamental HDC operations, enabling spatially sensitive and robust image encoding. This encoder achieves 95.67% accuracy on MNIST and 85.14% on Fashion-MNIST, outperforming prior HDC-based image encoders. Second, we design an end-to-end accelerator that implements these compute operations on an FPGA through a pipelined architecture that exploits parallelism both across the hypervector dimensionality and across the set of image patches. Our Alveo U280 implementation delivers 0.09ms inference latency, achieving up to 1300x and 60x speedup over state-of-the-art CPU and GPU baselines, respectively.
>
---
#### [new 067] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型中的记忆问题。通过分析对数概率的各向异性，提出一种新的检测方法，提升检测效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.20642v1](https://arxiv.org/pdf/2601.20642v1)**

> **作者:** Rohan Asthana; Vasileios Belagiannis
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Diffusion-based image generative models produce high-fidelity images through iterative denoising but remain vulnerable to memorization, where they unintentionally reproduce exact copies or parts of training images. Recent memorization detection methods are primarily based on the norm of score difference as indicators of memorization. We prove that such norm-based metrics are mainly effective under the assumption of isotropic log-probability distributions, which generally holds at high or medium noise levels. In contrast, analyzing the anisotropic regime reveals that memorized samples exhibit strong angular alignment between the guidance vector and unconditional scores in the low-noise setting. Through these insights, we develop a memorization detection metric by integrating isotropic norm and anisotropic alignment. Our detection metric can be computed directly on pure noise inputs via two conditional and unconditional forward passes, eliminating the need for costly denoising steps. Detection experiments on Stable Diffusion v1.4 and v2 show that our metric outperforms existing denoising-free detection methods while being at least approximately 5x faster than the previous best approach. Finally, we demonstrate the effectiveness of our approach by utilizing a mitigation strategy that adapts memorized prompts based on our developed metric.
>
---
#### [new 068] SegRap2025: A Benchmark of Gross Tumor Volume and Lymph Node Clinical Target Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分割任务，旨在提升鼻咽癌放疗靶区分割模型的泛化与鲁棒性。通过构建多中心、多模态数据集，评估模型在不同中心和成像模式下的表现。**

- **链接: [https://arxiv.org/pdf/2601.20575v1](https://arxiv.org/pdf/2601.20575v1)**

> **作者:** Jia Fu; Litingyu Wang; He Li; Zihao Luo; Huamin Wang; Chenyuan Bian; Zijun Gao; Chunbin Gu; Xin Weng; Jianghao Wu; Yicheng Wu; Jin Ye; Linhao Li; Yiwen Ye; Yong Xia; Elias Tappeiner; Fei He; Abdul qayyum; Moona Mazher; Steven A Niederer; Junqiang Chen; Chuanyi Huang; Lisheng Wang; Zhaohu Xing; Hongqiu Wang; Lei Zhu; Shichuan Zhang; Shaoting Zhang; Wenjun Liao; Guotai Wang
>
> **摘要:** Accurate delineation of Gross Tumor Volume (GTV), Lymph Node Clinical Target Volume (LN CTV), and Organ-at-Risk (OAR) from Computed Tomography (CT) scans is essential for precise radiotherapy planning in Nasopharyngeal Carcinoma (NPC). Building upon SegRap2023, which focused on OAR and GTV segmentation using single-center paired non-contrast CT (ncCT) and contrast-enhanced CT (ceCT) scans, the SegRap2025 challenge aims to enhance the generalizability and robustness of segmentation models across imaging centers and modalities. SegRap2025 comprises two tasks: Task01 addresses GTV segmentation using paired CT from the SegRap2023 dataset, with an additional external testing set to evaluate cross-center generalization, and Task02 focuses on LN CTV segmentation using multi-center training data and an unseen external testing set, where each case contains paired CT scans or a single modality, emphasizing both cross-center and cross-modality robustness. This paper presents the challenge setup and provides a comprehensive analysis of the solutions submitted by ten participating teams. For GTV segmentation task, the top-performing models achieved average Dice Similarity Coefficient (DSC) of 74.61% and 56.79% on the internal and external testing cohorts, respectively. For LN CTV segmentation task, the highest average DSC values reached 60.24%, 60.50%, and 57.23% on paired CT, ceCT-only, and ncCT-only subsets, respectively. SegRap2025 establishes a large-scale multi-center, multi-modality benchmark for evaluating the generalization and robustness in radiotherapy target segmentation, providing valuable insights toward clinically applicable automated radiotherapy planning systems. The benchmark is available at: https://hilab-git.github.io/SegRap2025_Challenge.
>
---
#### [new 069] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出MeanCache，用于加速流匹配推理。针对高加速比下轨迹偏差和误差累积问题，通过平均速度和JVP缓存提升稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2601.19961v1](https://arxiv.org/pdf/2601.19961v1)**

> **作者:** Huanlin Gao; Ping Chen; Fuyuan Shi; Ruijia Wu; Li YanTao; Qiang Hui; Yuren You; Ting Lu; Chao Tan; Shaoan Zhao; Zhaoxiang Liu; Fang Zhao; Kai Wang; Shiguo Lian
>
> **摘要:** We present MeanCache, a training-free caching framework for efficient Flow Matching inference. Existing caching methods reduce redundant computation but typically rely on instantaneous velocity information (e.g., feature caching), which often leads to severe trajectory deviations and error accumulation under high acceleration ratios. MeanCache introduces an average-velocity perspective: by leveraging cached Jacobian--vector products (JVP) to construct interval average velocities from instantaneous velocities, it effectively mitigates local error accumulation. To further improve cache timing and JVP reuse stability, we develop a trajectory-stability scheduling strategy as a practical tool, employing a Peak-Suppressed Shortest Path under budget constraints to determine the schedule. Experiments on FLUX.1, Qwen-Image, and HunyuanVideo demonstrate that MeanCache achieves 4.12X and 4.56X and 3.59X acceleration, respectively, while consistently outperforming state-of-the-art caching baselines in generation quality. We believe this simple yet effective approach provides a new perspective for Flow Matching inference and will inspire further exploration of stability-driven acceleration in commercial-scale generative models.
>
---
#### [new 070] TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决变形物体在复杂纹理下的功能区域识别问题。提出TRACER框架，通过语义分解与边界优化提升操作精度与连续性。**

- **链接: [https://arxiv.org/pdf/2601.20208v1](https://arxiv.org/pdf/2601.20208v1)**

> **作者:** Wanjun Jia; Kang Li; Fan Yang; Mengfei Duan; Wenrui Chen; Yiming Jiang; Hui Zhang; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** The source code and dataset will be made publicly available at https://github.com/Dikay1/TRACER
>
> **摘要:** The central challenge in robotic manipulation of deformable objects lies in aligning high-level semantic instructions with physical interaction points under complex appearance and texture variations. Due to near-infinite degrees of freedom, complex dynamics, and heterogeneous patterns, existing vision-based affordance prediction methods often suffer from boundary overflow and fragmented functional regions. To address these issues, we propose TRACER, a Texture-Robust Affordance Chain-of-thought with dEformable-object Refinement framework, which establishes a cross-hierarchical mapping from hierarchical semantic reasoning to appearance-robust and physically consistent functional region refinement. Specifically, a Tree-structured Affordance Chain-of-Thought (TA-CoT) is formulated to decompose high-level task intentions into hierarchical sub-task semantics, providing consistent guidance across various execution stages. To ensure spatial integrity, a Spatial-Constrained Boundary Refinement (SCBR) mechanism is introduced to suppress prediction spillover, guiding the perceptual response to converge toward authentic interaction manifolds. Furthermore, an Interactive Convergence Refinement Flow (ICRF) is developed to aggregate discrete pixels corrupted by appearance noise, significantly enhancing the spatial continuity and physical plausibility of the identified functional regions. Extensive experiments conducted on the Fine-AGDDO15 dataset and a real-world robotic platform demonstrate that TRACER significantly improves affordance grounding precision across diverse textures and patterns inherent to deformable objects. More importantly, it enhances the success rate of long-horizon tasks, effectively bridging the gap between high-level semantic reasoning and low-level physical execution. The source code and dataset will be made publicly available at https://github.com/Dikay1/TRACER.
>
---
#### [new 071] C3Box: A CLIP-based Class-Incremental Learning Toolbox
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出C3Box，一个基于CLIP的类增量学习工具箱，解决传统方法在动态数据流中遗忘旧知识的问题，整合多种CIL方法，提供统一框架以促进研究与应用。**

- **链接: [https://arxiv.org/pdf/2601.20852v1](https://arxiv.org/pdf/2601.20852v1)**

> **作者:** Hao Sun; Da-Wei Zhou
>
> **备注:** The code is available at https://github.com/LAMDA-CL/C3Box
>
> **摘要:** Traditional machine learning systems are typically designed for static data distributions, which suffer from catastrophic forgetting when learning from evolving data streams. Class-Incremental Learning (CIL) addresses this challenge by enabling learning systems to continuously learn new classes while preserving prior knowledge. With the rise of pre-trained models (PTMs) such as CLIP, leveraging their strong generalization and semantic alignment capabilities has become a promising direction in CIL. However, existing CLIP-based CIL methods are often scattered across disparate codebases, rely on inconsistent configurations, hindering fair comparisons, reproducibility, and practical adoption. Therefore, we propose C3Box (CLIP-based Class-inCremental learning toolBOX), a modular and comprehensive Python toolbox. C3Box integrates representative traditional CIL methods, ViT-based CIL methods, and state-of-the-art CLIP-based CIL methods into a unified CLIP-based framework. By inheriting the streamlined design of PyCIL, C3Box provides a JSON-based configuration and standardized execution pipeline. This design enables reproducible experimentation with low engineering overhead and makes C3Box a reliable benchmark platform for continual learning research. Designed to be user-friendly, C3Box relies only on widely used open-source libraries and supports major operating systems. The code is available at https://github.com/LAMDA-CL/C3Box.
>
---
#### [new 072] UnlearnShield: Shielding Forgotten Privacy against Unlearning Inversion
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私保护任务，旨在解决机器学习中数据遗忘后的隐私泄露问题。提出UnlearnShield防御机制，通过扰动和约束降低逆向攻击风险，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20325v1](https://arxiv.org/pdf/2601.20325v1)**

> **作者:** Lulu Xue; Shengshan Hu; Wei Lu; Ziqi Zhou; Yufei Song; Jianhong Cheng; Minghui Li; Yanjun Zhang; Leo Yu Zhang
>
> **备注:** This work has been accepted by ICASSP 2026
>
> **摘要:** Machine unlearning is an emerging technique that aims to remove the influence of specific data from trained models, thereby enhancing privacy protection. However, recent research has uncovered critical privacy vulnerabilities, showing that adversaries can exploit unlearning inversion to reconstruct data that was intended to be erased. Despite the severity of this threat, dedicated defenses remain lacking. To address this gap, we propose UnlearnShield, the first defense specifically tailored to counter unlearning inversion. UnlearnShield introduces directional perturbations in the cosine representation space and regulates them through a constraint module to jointly preserve model accuracy and forgetting efficacy, thereby reducing inversion risk while maintaining utility. Experiments demonstrate that it achieves a good trade-off among privacy protection, accuracy, and forgetting.
>
---
#### [new 073] NCSAM Noise-Compensated Sharpness-Aware Minimization for Noisy Label Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于噪声标签学习任务，旨在解决标签噪声对模型性能的影响。通过理论分析与方法创新，提出NCSAM算法提升模型的泛化能力和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.19947v1](https://arxiv.org/pdf/2601.19947v1)**

> **作者:** Jiayu Xu; Junbiao Pang
>
> **摘要:** Learning from Noisy Labels (LNL) presents a fundamental challenge in deep learning, as real-world datasets often contain erroneous or corrupted annotations, \textit{e.g.}, data crawled from Web. Current research focuses on sophisticated label correction mechanisms. In contrast, this paper adopts a novel perspective by establishing a theoretical analysis the relationship between flatness of the loss landscape and the presence of label noise. In this paper, we theoretically demonstrate that carefully simulated label noise synergistically enhances both the generalization performance and robustness of label noises. Consequently, we propose Noise-Compensated Sharpness-aware Minimization (NCSAM) to leverage the perturbation of Sharpness-Aware Minimization (SAM) to remedy the damage of label noises. Our analysis reveals that the testing accuracy exhibits a similar behavior that has been observed on the noise-clear dataset. Extensive experimental results on multiple benchmark datasets demonstrate the consistent superiority of the proposed method over existing state-of-the-art approaches on diverse tasks.
>
---
#### [new 074] StreamFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs
- **分类: cs.DC; cs.CV**

- **简介: 该论文属于图像生成任务，针对扩散Transformer在GPU上分布式推理的效率问题，提出StreamFusion框架，通过优化通信模式和计算重叠提升性能。**

- **链接: [https://arxiv.org/pdf/2601.20273v1](https://arxiv.org/pdf/2601.20273v1)**

> **作者:** Jiacheng Yang; Jun Wu; Yaoyao Ding; Zhiying Xu; Yida Wang; Gennady Pekhimenko
>
> **摘要:** Diffusion Transformers (DiTs) have gained increasing adoption in high-quality image and video generation. As demand for higher-resolution images and longer videos increases, single-GPU inference becomes inefficient due to increased latency and large activation sizes. Current frameworks employ sequence parallelism (SP) techniques such as Ulysses Attention and Ring Attention to scale inference. However, these implementations have three primary limitations: (1) suboptimal communication patterns for network topologies on modern GPU machines, (2) latency bottlenecks from all-to-all operations in inter-machine communication, and (3) GPU sender-receiver synchronization and computation overheads from using two-sided communication libraries. To address these issues, we present StreamFusion, a topology-aware efficient DiT serving engine. StreamFusion incorporates three key innovations: (1) a topology-aware sequence parallelism technique that accounts for inter- and intra-machine bandwidth differences, (2) Torus Attention, a novel SP technique enabling overlapping of inter-machine all-to-all operations with computation, and (3) a one-sided communication implementation that minimizes GPU sender-receiver synchronization and computation overheads. Our experiments demonstrate that StreamFusion outperforms the state-of-the-art approach by an average of $1.35\times$ (up to $1.77\times$).
>
---
#### [new 075] oculomix: Hierarchical Sampling for Retinal-Based Systemic Disease Prediction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于医疗预测任务，旨在解决 retinal-based 系统疾病预测中的患者特征丢失问题。提出 Oculomix 方法，通过层次采样保留患者特异性信息，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.19939v1](https://arxiv.org/pdf/2601.19939v1)**

> **作者:** Hyunmin Kim; Yukun Zhou; Rahul A. Jonas; Lie Ju; Sunjin Hwang; Pearse A. Keane; Siegfried K. Wagner
>
> **备注:** Accepted to ISBI 2026
>
> **摘要:** Oculomics - the concept of predicting systemic diseases, such as cardiovascular disease and dementia, through retinal imaging - has advanced rapidly due to the data efficiency of transformer-based foundation models like RETFound. Image-level mixed sample data augmentations, such as CutMix and MixUp, are frequently used for training transformers, yet these techniques perturb patient-specific attributes, such as medical comorbidity and clinical factors, since they only account for images and labels. To address this limitation, we propose a hierarchical sampling strategy, Oculomix, for mixed sample augmentations. Our method is based on two clinical priors. First (exam level), images acquired from the same patient at the same time point share the same attributes. Second (patient level), images acquired from the same patient at different time points have a soft temporal trend, as morbidity generally increases over time. Guided by these priors, our method constrains the mixing space to the patient and exam levels to better preserve patient-specific characteristics and leverages their hierarchical relationships. The proposed method is validated using ViT models on a five-year prediction of major adverse cardiovascular events (MACE) in a large ethnically diverse population (Alzeye). We show that Oculomix consistently outperforms image-level CutMix and MixUp by up to 3% in AUROC, demonstrating the necessity and value of the proposed method in oculomics.
>
---
#### [new 076] Continual GUI Agents
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出Continual GUI Agents任务，解决GUI代理在动态环境中的持续学习问题。通过GUI-AiF框架，利用新奖励机制提升代理对变化界面的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.20732v1](https://arxiv.org/pdf/2601.20732v1)**

> **作者:** Ziwei Liu; Borui Kang; Hangjie Yuan; Zixiang Zhao; Wei Li; Yifan Zhu; Tao Feng
>
> **摘要:** As digital environments (data distribution) are in flux, with new GUI data arriving over time-introducing new domains or resolutions-agents trained on static environments deteriorate in performance. In this work, we introduce Continual GUI Agents, a new task that requires GUI agents to perform continual learning under shifted domains and resolutions. We find existing methods fail to maintain stable grounding as GUI distributions shift over time, due to the diversity of UI interaction points and regions in fluxing scenarios. To address this, we introduce GUI-Anchoring in Flux (GUI-AiF), a new reinforcement fine-tuning framework that stabilizes continual learning through two novel rewards: Anchoring Point Reward in Flux (APR-iF) and Anchoring Region Reward in Flux (ARR-iF). These rewards guide the agents to align with shifting interaction points and regions, mitigating the tendency of existing reward strategies to over-adapt to static grounding cues (e.g., fixed coordinates or element scales). Extensive experiments show GUI-AiF surpasses state-of-the-art baselines. Our work establishes the first continual learning framework for GUI agents, revealing the untapped potential of reinforcement fine-tuning for continual GUI Agents.
>
---
#### [new 077] GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering
- **分类: cs.GR; cs.AR; cs.CV**

- **简介: 该论文属于3D渲染任务，旨在解决Gaussian ray tracing效率低的问题。通过优化加速结构和硬件支持，提升渲染性能。**

- **链接: [https://arxiv.org/pdf/2601.20429v1](https://arxiv.org/pdf/2601.20429v1)**

> **作者:** Junseo Lee; Sangyun Jeon; Jungi Lee; Junyong Park; Jaewoong Sim
>
> **备注:** To appear at the 32nd International Symposium on High-Performance Computer Architecture (HPCA 2026)
>
> **摘要:** 3D Gaussian Splatting has gained widespread adoption across diverse applications due to its exceptional rendering performance and visual quality. While most existing methods rely on rasterization to render Gaussians, recent research has started investigating ray tracing approaches to overcome the fundamental limitations inherent in rasterization. However, current Gaussian ray tracing methods suffer from inefficiencies such as bloated acceleration structures and redundant node traversals, which greatly degrade ray tracing performance. In this work, we present GRTX, a set of software and hardware optimizations that enable efficient ray tracing for 3D Gaussian-based rendering. First, we introduce a novel approach for constructing streamlined acceleration structures for Gaussian primitives. Our key insight is that anisotropic Gaussians can be treated as unit spheres through ray space transformations, which substantially reduces BVH size and traversal overhead. Second, we propose dedicated hardware support for traversal checkpointing within ray tracing units. This eliminates redundant node visits during multi-round tracing by resuming traversal from checkpointed nodes rather than restarting from the root node in each subsequent round. Our evaluation shows that GRTX significantly improves ray tracing performance compared to the baseline ray tracing method with a negligible hardware cost.
>
---
#### [new 078] SemBind: Binding Diffusion Watermarks to Semantics Against Black-Box Forgery Attacks
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文属于图像水印任务，解决黑盒伪造攻击问题。提出SemBind框架，通过语义掩码绑定潜在信号，增强水印抗伪造能力，同时保持图像质量。**

- **链接: [https://arxiv.org/pdf/2601.20310v1](https://arxiv.org/pdf/2601.20310v1)**

> **作者:** Xin Zhang; Zijin Yang; Kejiang Chen; Linfeng Ma; Weiming Zhang; Nenghai Yu
>
> **摘要:** Latent-based watermarks, integrated into the generation process of latent diffusion models (LDMs), simplify detection and attribution of generated images. However, recent black-box forgery attacks, where an attacker needs at least one watermarked image and black-box access to the provider's model, can embed the provider's watermark into images not produced by the provider, posing outsized risk to provenance and trust. We propose SemBind, the first defense framework for latent-based watermarks that resists black-box forgery by binding latent signals to image semantics via a learned semantic masker. Trained with contrastive learning, the masker yields near-invariant codes for the same prompt and near-orthogonal codes across prompts; these codes are reshaped and permuted to modulate the target latent before any standard latent-based watermark. SemBind is generally compatible with existing latent-based watermarking schemes and keeps image quality essentially unchanged, while a simple mask-ratio parameter offers a tunable trade-off between anti-forgery strength and robustness. Across four mainstream latent-based watermark methods, our SemBind-enabled anti-forgery variants markedly reduce false acceptance under black-box forgery while providing a controllable robustness-security balance.
>
---
## 更新

#### [replaced 001] FLOL: Fast Baselines for Real-World Low-Light Enhancement
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于低光图像增强任务，旨在解决真实场景下图像增强的效率与鲁棒性问题。提出轻量级网络FLOL，在频域和空域结合处理，实现快速且高质量的增强效果。**

- **链接: [https://arxiv.org/pdf/2501.09718v2](https://arxiv.org/pdf/2501.09718v2)**

> **作者:** Juan C. Benito; Daniel Feijoo; Alvaro Garcia; Marcos V. Conde
>
> **备注:** Journal Preprint
>
> **摘要:** Low-Light Image Enhancement (LLIE) is a key task in computational photography and imaging. The problem of enhancing images captured during night or in dark environments has been well-studied in the computer vision literature. However, current deep learning-based solutions struggle with efficiency and robustness for real-world scenarios (e.g., scenes with noise, saturated pixels). We propose a lightweight neural network that combines image processing in the frequency and spatial domains. Our baseline method, FLOL, is one of the fastest models for this task, achieving results comparable to the state-of-the-art on popular real-world benchmarks such as LOLv2, LSRW, MIT-5K and UHD-LL. Moreover, we are able to process 1080p images in real-time under 12ms. Code and models at https://github.com/cidautai/FLOL
>
---
#### [replaced 002] Tri-Reader: An Open-Access, Multi-Stage AI Pipeline for First-Pass Lung Nodule Annotation in Screening CT
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.19380v2](https://arxiv.org/pdf/2601.19380v2)**

> **作者:** Fakrul Islam Tushar; Joseph Y. Lo
>
> **备注:** 1 figure , 2 tables, 20 page supplement
>
> **摘要:** Using multiple open-access models trained on public datasets, we developed Tri-Reader, a comprehensive, freely available pipeline that integrates lung segmentation, nodule detection, and malignancy classification into a unified tri-stage workflow. The pipeline is designed to prioritize sensitivity while reducing the candidate burden for annotators. To ensure accuracy and generalizability across diverse practices, we evaluated Tri-Reader on multiple internal and external datasets as compared with expert annotations and dataset-provided reference standards.
>
---
#### [replaced 003] Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决现有方法依赖人工标注和推理延迟的问题。提出IVT-LR，在潜在空间中融合视觉与文本信息，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.12603v2](https://arxiv.org/pdf/2510.12603v2)**

> **作者:** Chao Chen; Zhixin Ma; Yongqi Li; Yupeng Hu; Yinwei Wei; Wenjie Li; Liqiang Nie
>
> **摘要:** Multimodal reasoning aims to enhance the capabilities of MLLMs by incorporating intermediate reasoning steps before reaching the final answer. It has evolved from text-only reasoning to the integration of visual information, enabling the thought process to be conveyed through both images and text. Despite its effectiveness, current multimodal reasoning methods depend on explicit reasoning steps that require labor-intensive vision-text annotations and inherently introduce significant inference latency. To address these issues, we introduce multimodal latent reasoning with the advantages of multimodal representation, reduced annotation, and inference efficiency. To facilitate it, we propose Interleaved Vision-Text Latent Reasoning (IVT-LR), which injects both visual and textual information in the reasoning process within the latent space. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: latent text (the hidden states from the previous step) and latent vision (a set of selected image embeddings). We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps. Experiments on M$^3$CoT and ScienceQA demonstrate that our IVT-LR method achieves an average performance increase of 5.45\% in accuracy, while simultaneously achieving a speed increase of over 5 times compared to existing approaches.
>
---
#### [replaced 004] Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.22258v4](https://arxiv.org/pdf/2509.22258v4)**

> **作者:** Miao Jing; Mengting Jia; Junling Lin; Zhongxia Shen; Huan Gao; Mingkun Xu; Shangyang Li
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have achieved remarkable performance on standard medical benchmarks, yet their true clinical reasoning ability remains unclear. Existing datasets predominantly emphasize classification accuracy, creating an evaluation illusion in which models appear proficient while still failing at high-stakes diagnostic reasoning. We introduce Neural-MedBench, a compact yet reasoning-intensive benchmark specifically designed to probe the limits of multimodal clinical reasoning in neurology. Neural-MedBench integrates multi-sequence MRI scans, structured electronic health records, and clinical notes, and encompasses three core task families: differential diagnosis, lesion recognition, and rationale generation. To ensure reliable evaluation, we develop a hybrid scoring pipeline that combines LLM-based graders, clinician validation, and semantic similarity metrics. Through systematic evaluation of state-of-the-art VLMs, including GPT-4o, Claude-4, and MedGemma, we observe a sharp performance drop compared to conventional datasets. Error analysis shows that reasoning failures, rather than perceptual errors, dominate model shortcomings. Our findings highlight the necessity of a Two-Axis Evaluation Framework: breadth-oriented large datasets for statistical generalization, and depth-oriented, compact benchmarks such as Neural-MedBench for reasoning fidelity. We release Neural-MedBench at https://neuromedbench.github.io/ as an open and extensible diagnostic testbed, which guides the expansion of future benchmarks and enables rigorous yet cost-effective assessment of clinically trustworthy AI.
>
---
#### [replaced 005] Learning Stochastic Bridges for Video Object Removal via Video-to-Video Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12066v2](https://arxiv.org/pdf/2601.12066v2)**

> **作者:** Zijie Lou; Xiangwei Feng; Jiaxin Wang; Jiangtao Yao; Fei Che; Tianbao Liu; Chengjing Wu; Xiaochao Qu; Luoqi Liu; Ting Liu
>
> **摘要:** Existing video object removal methods predominantly rely on diffusion models following a noise-to-data paradigm, where generation starts from uninformative Gaussian noise. This approach discards the rich structural and contextual priors present in the original input video. Consequently, such methods often lack sufficient guidance, leading to incomplete object erasure or the synthesis of implausible content that conflicts with the scene's physical logic. In this paper, we reformulate video object removal as a video-to-video translation task via a stochastic bridge model. Unlike noise-initialized methods, our framework establishes a direct stochastic path from the source video (with objects) to the target video (objects removed). This bridge formulation effectively leverages the input video as a strong structural prior, guiding the model to perform precise removal while ensuring that the filled regions are logically consistent with the surrounding environment. To address the trade-off where strong bridge priors hinder the removal of large objects, we propose a novel adaptive mask modulation strategy. This mechanism dynamically modulates input embeddings based on mask characteristics, balancing background fidelity with generative flexibility. Extensive experiments demonstrate that our approach significantly outperforms existing methods in both visual quality and temporal consistency. The project page is https://bridgeremoval.github.io/.
>
---
#### [replaced 006] A Roadmap for Greater Public Use of Privacy-Sensitive Government Data: Workshop Report
- **分类: cs.CR; cs.CV; cs.CY; cs.LG**

- **链接: [https://arxiv.org/pdf/2208.01636v2](https://arxiv.org/pdf/2208.01636v2)**

> **作者:** Chris Clifton; Bradley Malin; Anna Oganian; Ramesh Raskar; Vivek Sharma
>
> **备注:** 23 pages. Web: https://may2021privacy.github.io/
>
> **摘要:** Government agencies collect and manage a wide range of ever-growing datasets. While such data has the potential to support research and evidence-based policy making, there are concerns that the dissemination of such data could infringe upon the privacy of the individuals (or organizations) from whom such data was collected. To appraise the current state of data sharing, as well as learn about opportunities for stimulating such sharing at a faster pace, a virtual workshop was held on May 21st and 26th, 2021, sponsored by the National Science Foundation (NSF) and National Institute of Standards and Technologies (NIST), and the White House Office of Science and Technology Policy (OSTP), where a multinational collection of researchers and practitioners were brought together to discuss their experiences and learn about recently developed technologies for managing privacy while sharing data. The workshop specifically focused on challenges and successes in government data sharing at various levels. The first day focused on successful examples of new technology applied to sharing of public data, including formal privacy techniques, synthetic data, and cryptographic approaches. Day two emphasized brainstorming sessions on some of the challenges and directions to address them.
>
---
#### [replaced 007] GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10631v2](https://arxiv.org/pdf/2510.10631v2)**

> **作者:** Zhaolin Hu; Kun Li; Hehe Fan; Yi Yang
>
> **备注:** Accepted by WWW 2026. Research Tracks - Graph Algorithms and Modeling for the Web
>
> **摘要:** Linear attention mechanisms have emerged as efficient alternatives to full self-attention in Graph Transformers, offering linear time complexity. However, existing linear attention models often suffer from a significant drop in expressiveness due to low-rank projection structures and overly uniform attention distributions. We theoretically prove that these properties reduce the class separability of node representations, limiting the model's classification ability. To address this, we propose a novel hybrid framework that enhances both the rank and focus of attention. Specifically, we enhance linear attention by attaching a gated local graph network branch to the value matrix, thereby increasing the rank of the resulting attention map. Furthermore, to alleviate the excessive smoothing effect inherent in linear attention, we introduce a learnable log-power function into the attention scores to reduce entropy and sharpen focus. We theoretically show that this function decreases entropy in the attention distribution, enhancing the separability of learned embeddings. Extensive experiments on both homophilic and heterophilic graph benchmarks demonstrate that our method achieves competitive performance while preserving the scalability of linear attention.
>
---
#### [replaced 008] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05150v2](https://arxiv.org/pdf/2512.05150v2)**

> **作者:** Zhenglin Cheng; Peng Sun; Jianguo Li; Tao Lin
>
> **备注:** arxiv v1, accepted to ICLR 2026
>
> **摘要:** Recent advances in large multi-modal generative models have demonstrated impressive capabilities in multi-modal generation, including image and video generation. These models are typically built upon multi-step frameworks like diffusion and flow matching, which inherently limits their inference efficiency (requiring 40-100 Number of Function Evaluations (NFEs)). While various few-step methods aim to accelerate the inference, existing solutions have clear limitations. Prominent distillation-based methods, such as progressive and consistency distillation, either require an iterative distillation procedure or show significant degradation at very few steps (< 4-NFE). Meanwhile, integrating adversarial training into distillation (e.g., DMD/DMD2 and SANA-Sprint) to enhance performance introduces training instability, added complexity, and high GPU memory overhead due to the auxiliary trained models. To this end, we propose TwinFlow, a simple yet effective framework for training 1-step generative models that bypasses the need of fixed pretrained teacher models and avoids standard adversarial networks during training, making it ideal for building large-scale, efficient models. On text-to-image tasks, our method achieves a GenEval score of 0.83 in 1-NFE, outperforming strong baselines like SANA-Sprint (a GAN loss-based framework) and RCGM (a consistency-based framework). Notably, we demonstrate the scalability of TwinFlow by full-parameter training on Qwen-Image-20B and transform it into an efficient few-step generator. With just 1-NFE, our approach matches the performance of the original 100-NFE model on both the GenEval and DPG-Bench benchmarks, reducing computational cost by $100\times$ with minor quality degradation. Project page is available at https://zhenglin-cheng.com/twinflow.
>
---
#### [replaced 009] Visual Instruction Pretraining for Domain-Specific Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17562v3](https://arxiv.org/pdf/2509.17562v3)**

> **作者:** Yuxuan Li; Yicheng Zhang; Wenhao Tang; Yimian Dai; Ming-Ming Cheng; Xiang Li; Jian Yang
>
> **摘要:** Modern computer vision is converging on a closed loop in which perception, reasoning and generation mutually reinforce each other. However, this loop remains incomplete: the top-down influence of high-level reasoning on the foundational learning of low-level perceptual features is not yet underexplored. This paper addresses this gap by proposing a new paradigm for pretraining foundation models in downstream domains. We introduce Visual insTruction Pretraining (ViTP), a novel approach that directly leverages reasoning to enhance perception. ViTP embeds a Vision Transformer (ViT) backbone within a Vision-Language Model and pretrains it end-to-end using a rich corpus of visual instruction data curated from target downstream domains. ViTP is powered by our proposed Visual Robustness Learning (VRL), which compels the ViT to learn robust and domain-relevant features from a sparse set of visual tokens. Extensive experiments on 16 challenging remote sensing and medical imaging benchmarks demonstrate that ViTP establishes new state-of-the-art performance across a diverse range of downstream tasks. The code is available at https://github.com/zcablii/ViTP.
>
---
#### [replaced 010] Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19489v2](https://arxiv.org/pdf/2601.19489v2)**

> **作者:** Ziyu Zhang; Tianle Liu; Diantao Tu; Shuhan Shen
>
> **备注:** First Rank of SIGGRAPH Asia 2025 3DGS Challenge. Code available at https://github.com/will-zzy/siggraph_asia
>
> **摘要:** We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution. In the first round, we use reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS and Speedy-splat, load-balanced tiling, an anchor-based Neural-Gaussian representation enabling rapid convergence with fewer learnable parameters, initialization from monocular depth and partially from feed-forward 3DGS models, and a global pose refinement module for noisy SLAM trajectories. In the final round, the accurate COLMAP poses change the optimization landscape; we disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead, introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS, and introduce a depth estimator to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition.
>
---
#### [replaced 011] RxnBench: A Multimodal Benchmark for Evaluating Large Language Models on Chemical Reaction Understanding from Scientific Literature
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.23565v5](https://arxiv.org/pdf/2512.23565v5)**

> **作者:** Hanzheng Li; Xi Fang; Yixuan Li; Chaozheng Huang; Junjie Wang; Xi Wang; Hongzhe Bai; Bojun Hao; Shenyu Lin; Huiqi Liang; Linfeng Zhang; Guolin Ke
>
> **摘要:** The integration of Multimodal Large Language Models (MLLMs) into chemistry promises to revolutionize scientific discovery, yet their ability to comprehend the dense, graphical language of reactions within authentic literature remains underexplored. Here, we introduce RxnBench, a multi-tiered benchmark designed to rigorously evaluate MLLMs on chemical reaction understanding from scientific PDFs. RxnBench comprises two tasks: Single-Figure QA (SF-QA), which tests fine-grained visual perception and mechanistic reasoning using 1,525 questions derived from 305 curated reaction schemes, and Full-Document QA (FD-QA), which challenges models to synthesize information from 108 articles, requiring cross-modal integration of text, schemes, and tables. Our evaluation of MLLMs reveals a critical capability gap: while models excel at extracting explicit text, they struggle with deep chemical logic and precise structural recognition. Notably, models with inference-time reasoning significantly outperform standard architectures, yet none achieve 50\% accuracy on FD-QA. These findings underscore the urgent need for domain-specific visual encoders and stronger reasoning engines to advance autonomous AI chemists.
>
---
#### [replaced 012] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出DaMO，解决视频语言模型中时间推理不足的问题。通过多模态融合和高效训练策略，提升时间对齐与理解能力。**

- **链接: [https://arxiv.org/pdf/2506.11558v4](https://arxiv.org/pdf/2506.11558v4)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen; An-Zi Yen
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [replaced 013] ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型对隐含颜色概念理解不足的问题。通过构建基准数据集，评估模型在概率颜色分布下的表现，揭示现有模型在抽象语义上的局限性。**

- **链接: [https://arxiv.org/pdf/2601.16836v2](https://arxiv.org/pdf/2601.16836v2)**

> **作者:** Chenxi Ruan; Yu Xiao; Yihan Hou; Guosheng Hu; Wei Zeng
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** While text-to-image (T2I) models have advanced considerably, their capability to associate colors with implicit concepts remains underexplored. To address the gap, we introduce ColorConceptBench, a new human-annotated benchmark to systematically evaluate color-concept associations through the lens of probabilistic color distributions. ColorConceptBench moves beyond explicit color names or codes by probing how models translate 1,281 implicit color concepts using a foundation of 6,369 human annotations. Our evaluation of seven leading T2I models reveals that current models lack sensitivity to abstract semantics, and crucially, this limitation appears resistant to standard interventions (e.g., scaling and guidance). This demonstrates that achieving human-like color semantics requires more than larger models, but demands a fundamental shift in how models learn and represent implicit meaning.
>
---
#### [replaced 014] RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.17045v3](https://arxiv.org/pdf/2511.17045v3)**

> **作者:** Linfeng Dong; Yuchen Yang; Hao Wu; Wei Wang; Yuenan Hou; Zhihang Zhong; Xiao Sun
>
> **备注:** Accepted to AAAI 2026 (Oral)
>
> **摘要:** We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at https://github.com/OrcustD/RacketVision
>
---
#### [replaced 015] QVGen: Pushing the Limit of Quantized Video Generative Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.11497v5](https://arxiv.org/pdf/2505.11497v5)**

> **作者:** Yushi Huang; Ruihao Gong; Jing Liu; Yifu Ding; Chengtao Lv; Haotong Qin; Jun Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Video diffusion models (DMs) have enabled high-quality video synthesis. Yet, their substantial computational and memory demands pose serious challenges to real-world deployment, even on high-end GPUs. As a commonly adopted solution, quantization has proven notable success in reducing cost for image DMs, while its direct application to video DMs remains ineffective. In this paper, we present QVGen, a novel quantization-aware training (QAT) framework tailored for high-performance and inference-efficient video DMs under extremely low-bit quantization (e.g., 4-bit or below). We begin with a theoretical analysis demonstrating that reducing the gradient norm is essential to facilitate convergence for QAT. To this end, we introduce auxiliary modules ($Φ$) to mitigate large quantization errors, leading to significantly enhanced convergence. To eliminate the inference overhead of $Φ$, we propose a rank-decay strategy that progressively eliminates $Φ$. Specifically, we repeatedly employ singular value decomposition (SVD) and a proposed rank-based regularization $\mathbfγ$ to identify and decay low-contributing components. This strategy retains performance while zeroing out additional inference overhead. Extensive experiments across $4$ state-of-the-art (SOTA) video DMs, with parameter sizes ranging from $1.3\text{B}\sim14\text{B}$, show that QVGen is the first to reach full-precision comparable quality under 4-bit settings. Moreover, it significantly outperforms existing methods. For instance, our 3-bit CogVideoX-2B achieves improvements of $+25.28$ in Dynamic Degree and $+8.43$ in Scene Consistency on VBench. Code and models are available at https://github.com/ModelTC/QVGen.
>
---
#### [replaced 016] Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文聚焦于多模态大语言模型的推理能力提升，解决其复杂推理激活困难的问题，通过优化冷启动、解决梯度停滞及分阶段训练，提出ReVisual-R1模型，取得新成果。**

- **链接: [https://arxiv.org/pdf/2506.04207v2](https://arxiv.org/pdf/2506.04207v2)**

> **作者:** Shuang Chen; Yue Guo; Zhaochen Su; Yafu Li; Yulun Wu; Jiacheng Chen; Jiayu Chen; Weijie Wang; Xiaoye Qu; Yu Cheng
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025.
>
---
#### [replaced 017] AdaSCALE: Adaptive Scaling for OOD Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08023v3](https://arxiv.org/pdf/2503.08023v3)**

> **作者:** Sudarshan Regmi
>
> **备注:** https://github.com/sudarshanregmi/AdaSCALE/
>
> **摘要:** The ability of the deep learning model to recognize when a sample falls outside its learned distribution is critical for safe and reliable deployment. Recent state-of-the-art out-of-distribution (OOD) detection methods leverage activation shaping to improve the separation between in-distribution (ID) and OOD inputs. These approaches resort to sample-specific scaling but apply a static percentile threshold across all samples regardless of their nature, resulting in suboptimal ID-OOD separability. In this work, we propose \textbf{AdaSCALE}, an adaptive scaling procedure that dynamically adjusts the percentile threshold based on a sample's estimated OOD likelihood. This estimation leverages our key observation: OOD samples exhibit significantly more pronounced activation shifts at high-magnitude activations under minor perturbation compared to ID samples. AdaSCALE enables stronger scaling for likely ID samples and weaker scaling for likely OOD samples, yielding highly separable energy scores. Our approach achieves state-of-the-art OOD detection performance, outperforming the latest rival OptFS by 14.94% in near-OOD and 21.67% in far-OOD datasets in average FPR@95 metric on the ImageNet-1k benchmark across eight diverse architectures. The code is available at: https://github.com/sudarshanregmi/AdaSCALE/
>
---
#### [replaced 018] XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a Novel Benchmark
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2504.10258v3](https://arxiv.org/pdf/2504.10258v3)**

> **作者:** Shuai Liu; Youmeng Li; Jizeng Wei
>
> **摘要:** Document Reading Order Recovery is a fundamental task in document image understanding, playing a pivotal role in enhancing Retrieval-Augmented Generation (RAG) and serving as a critical preprocessing step for large language models (LLMs). Existing methods often struggle with complex layouts(e.g., multi-column newspapers), high-overhead interactions between cross-modal elements (visual regions and textual semantics), and a lack of robust evaluation benchmarks. We introduce XY-Cut++, an advanced layout ordering method that integrates pre-mask processing, multi-granularity segmentation, and cross-modal matching to address these challenges. Our method significantly enhances layout ordering accuracy compared to traditional XY-Cut techniques. Specifically, XY-Cut++ achieves state-of-the-art performance (98.8 BLEU overall) while maintaining simplicity and efficiency. It outperforms existing baselines by up to 24\% and demonstrates consistent accuracy across simple and complex layouts on the newly introduced DocBench-100 dataset. This advancement establishes a reliable foundation for document structure recovery, setting a new standard for layout ordering tasks and facilitating more effective RAG and LLM preprocessing.
>
---
#### [replaced 019] Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.16512v5](https://arxiv.org/pdf/2505.16512v5)**

> **作者:** Jiaxin Liu; Jia Wang; Saihui Hou; Min Ren; Huijia Wu; Long Ma; Renwang Pei; Zhaofeng He
>
> **摘要:** In recent years, the explosive advancement of deepfake technology has posed a critical and escalating threat to public security: diffusion-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency via multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the new large-scale multimodal digital human forgery dataset based on diffusion models. Leveraging five of the latest digital human generation methods and a voice cloning method, we systematically construct a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies demonstrate that the misrecognition rate by participants for DigiFakeAV reaches as high as 68%. Moreover, the substantial performance degradation of existing detection models on our dataset further highlights its challenges. To address this problem, we propose DigiShield, an effective detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves state-of-the-art (SOTA) performance on the DigiFakeAV and shows strong generalization on other datasets.
>
---
#### [replaced 020] BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.14318v5](https://arxiv.org/pdf/2506.14318v5)**

> **作者:** Amirreza Fateh; Yasin Rezvani; Sara Moayedi; Sadjad Rezvani; Fatemeh Fateh; Mansoor Fateh; Vahid Abolghasemi
>
> **摘要:** Accurate segmentation and classification of brain tumors from Magnetic Resonance Imaging (MRI) remain key challenges in medical image analysis, primarily due to the lack of high-quality, balanced, and diverse datasets with expert annotations. In this work, we address this gap by introducing BRISC, a dataset designed for brain tumor segmentation and classification tasks, featuring high-resolution segmentation masks. The dataset comprises 6,000 contrast-enhanced T1-weighted MRI scans, which were collated from multiple public datasets that lacked segmentation labels. Our primary contribution is the subsequent expert annotation of these images, performed by certified radiologists and physicians. It includes three major tumor types, namely glioma, meningioma, and pituitary, as well as non-tumorous cases. Each sample includes high-resolution labels and is categorized across axial, sagittal, and coronal imaging planes to facilitate robust model development and cross-view generalization. To demonstrate the utility of the dataset, we provide benchmark results for both tasks using standard deep learning models. The BRISC dataset is made publicly available. datasetlink: https://www.kaggle.com/datasets/briscdataset/brisc2025/
>
---
#### [replaced 021] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16324v2](https://arxiv.org/pdf/2505.16324v2)**

> **作者:** Cheng Cheng; Lin Song; Di An; Yicheng Xiao; Xuchong Zhang; Hongbin Sun; Ying Shan
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Autoregressive (AR) image generators offer a language-model-friendly approach to image generation by predicting discrete image tokens in a causal sequence. However, unlike diffusion models, AR models lack a mechanism to refine previous predictions, limiting their generation quality. In this paper, we introduce TensorAR, a new AR paradigm that reformulates image generation from next-token prediction to next-tensor prediction. By generating overlapping windows of image patches (tensors) in a sliding fashion, TensorAR enables iterative refinement of previously generated content. To prevent information leakage during training, we propose a discrete tensor noising scheme, which perturbs input tokens via codebook-indexed noise. TensorAR is implemented as a plug-and-play module compatible with existing AR models. Extensive experiments on LlamaGEN, Open-MAGVIT2, and RAR demonstrate that TensorAR significantly improves the generation performance of autoregressive models.
>
---
#### [replaced 022] A Multi-Stage Deep Learning Framework with PKCP-MixUp Augmentation for Pediatric Liver Tumor Diagnosis Using Multi-Phase Contrast-Enhanced CT
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.19478v3](https://arxiv.org/pdf/2511.19478v3)**

> **作者:** Wanqi Wang; Chun Yang; Jianbo Shao; Yaokai Zhang; Xuehua Peng; Jin Sun; Chao Xiong; Long Lu; Lianting Hu
>
> **摘要:** Pediatric liver tumors are one of the most common solid tumors in pediatrics, with differentiation of benign or malignant status and pathological classification critical for clinical treatment. While pathological examination is the gold standard, the invasive biopsy has notable limitations: the highly vascular pediatric liver and fragile tumor tissue raise complication risks such as bleeding; additionally, young children with poor compliance require anesthesia for biopsy, increasing medical costs or psychological trauma. Although many efforts have been made to utilize AI in clinical settings, most researchers have overlooked its importance in pediatric liver tumors. To establish a non-invasive examination procedure, we developed a multi-stage deep learning (DL) framework for automated pediatric liver tumor diagnosis using multi-phase contrast-enhanced CT. Two retrospective and prospective cohorts were enrolled. We established a novel PKCP-MixUp data augmentation method to address data scarcity and class imbalance. We also trained a tumor detection model to extract ROIs, and then set a two-stage diagnosis pipeline with three backbones with ROI-masked images. Our tumor detection model has achieved high performance (mAP=0.871), and the first stage classification model between benign and malignant tumors reached an excellent performance (AUC=0.989). Final diagnosis models also exhibited robustness, including benign subtype classification (AUC=0.915) and malignant subtype classification (AUC=0.979). We also conducted multi-level comparative analyses, such as ablation studies on data and training pipelines, as well as Shapley-Value and CAM interpretability analyses. This framework fills the pediatric-specific DL diagnostic gap, provides actionable insights for CT phase selection and model design, and paves the way for precise, accessible pediatric liver tumor diagnosis.
>
---
#### [replaced 023] ReactionMamba: Generating Short & Long Human Reaction Sequences
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00208v2](https://arxiv.org/pdf/2512.00208v2)**

> **作者:** Hajra Anwar Beg; Baptiste Chopin; Hao Tang; Mohamed Daoudi
>
> **摘要:** We present ReactionMamba, a novel framework for generating long 3D human reaction motions. Reaction-Mamba integrates a motion VAE for efficient motion encoding with Mamba-based state-space models to decode temporally consistent reactions. This design enables ReactionMamba to generate both short sequences of simple motions and long sequences of complex motions, such as dance and martial arts. We evaluate ReactionMamba on three datasets--NTU120-AS, Lindy Hop, and InterX--and demonstrate competitive performance in terms of realism, diversity, and long-sequence generation compared to previous methods, including InterFormer, ReMoS, and Ready-to-React, while achieving substantial improvements in inference speed.
>
---
#### [replaced 024] Diffusion in SPAD Signals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.07599v2](https://arxiv.org/pdf/2601.07599v2)**

> **作者:** Lior Dvir; Nadav Torem; Yoav Y. Schechner
>
> **摘要:** We derive the likelihood of a raw signal in a single photon avalanche diode (SPAD), given a fixed photon flux. The raw signal comprises timing of detection events, which are nonlinearly related to the flux. Moreover, they are naturally stochastic. We then derive a score function of the signal. This is a key for solving inverse problems based on SPAD signals. We focus on deriving solutions involving a diffusion model, to express image priors. We demonstrate the effect of low or high photon counts, and the consequence of exploiting timing of detection events.
>
---
#### [replaced 025] AGFS-Tractometry: A Novel Atlas-Guided Fine-Scale Tractometry Approach for Enhanced Along-Tract Group Statistical Comparison Using Diffusion MRI Tractography
- **分类: q-bio.QM; cs.CV; cs.LG; eess.IV; stat.ME**

- **链接: [https://arxiv.org/pdf/2507.10601v2](https://arxiv.org/pdf/2507.10601v2)**

> **作者:** Ruixi Zheng; Wei Zhang; Yijie Li; Xi Zhu; Zhou Lan; Jarrett Rushmore; Yogesh Rathi; Nikos Makris; Lauren J. O'Donnell; Fan Zhang
>
> **备注:** 31 pages and 7 figures
>
> **摘要:** Diffusion MRI (dMRI) tractography is currently the only method for in vivo mapping of the brain's white matter (WM) connections. Tractometry is an advanced tractography analysis technique for along-tract profiling to investigate the morphology and microstructural properties along the fiber tracts. Tractometry has become an essential tool for studying local along-tract differences between different populations (e.g., health vs disease). In this study, we propose a novel atlas-guided fine-scale tractometry method, namely AGFS-Tractometry, that leverages tract spatial information and permutation testing to enhance the along-tract statistical analysis between populations. There are two major contributions in AGFS-Tractometry. First, we create a novel atlas-guided tract profiling template that enables consistent, fine-scale, along-tract parcellation of subject-specific fiber tracts. Second, we propose a novel nonparametric permutation testing group comparison method to enable simultaneous analysis across all along-tract parcels while correcting for multiple comparisons. We perform experimental evaluations on synthetic datasets with known group differences and in vivo real data. We compare AGFS-Tractometry with two state-of-the-art tractometry methods, including Automated Fiber-tract Quantification (AFQ) and BUndle ANalytics (BUAN). Our results show that the proposed AGFS-Tractometry obtains enhanced sensitivity and specificity in detecting local WM differences. In the real data analysis experiments, AGFS-Tractometry can identify more regions with significant differences, which are anatomically consistent with the existing literature. Overall, these demonstrate the ability of AGFS-Tractometry to detect subtle or spatially localized WM group-level differences. The created tract profiling template and related code are available at: https://github.com/ZhengRuixi/AGFS-Tractometry.git.
>
---
#### [replaced 026] PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.17354v2](https://arxiv.org/pdf/2601.17354v2)**

> **作者:** Wenzhi Guo; Guangchi Fang; Shu Yang; Bing Wang
>
> **摘要:** Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
>
---
#### [replaced 027] From Specialist to Generalist: Unlocking SAM's Learning Potential on Unlabeled Medical Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.17934v2](https://arxiv.org/pdf/2601.17934v2)**

> **作者:** Vi Vu; Thanh-Huy Nguyen; Tien-Thinh Nguyen; Ba-Thinh Lam; Hoang-Thien Nguyen; Tianyang Wang; Xingjian Li; Min Xu
>
> **备注:** Accepted to ISBI 2026
>
> **摘要:** Foundation models like the Segment Anything Model (SAM) show strong generalization, yet adapting them to medical images remains difficult due to domain shift, scarce labels, and the inability of Parameter-Efficient Fine-Tuning (PEFT) to exploit unlabeled data. While conventional models like U-Net excel in semi-supervised medical learning, their potential to assist a PEFT SAM has been largely overlooked. We introduce SC-SAM, a specialist-generalist framework where U-Net provides point-based prompts and pseudo-labels to guide SAM's adaptation, while SAM serves as a powerful generalist supervisor to regularize U-Net. This reciprocal guidance forms a bidirectional co-training loop that allows both models to effectively exploit the unlabeled data. Across prostate MRI and polyp segmentation benchmarks, our method achieves state-of-the-art results, outperforming other existing semi-supervised SAM variants and even medical foundation models like MedSAM, highlighting the value of specialist-generalist cooperation for label-efficient medical image segmentation. Our code is available at https://github.com/vnlvi2k3/SC-SAM.
>
---
#### [replaced 028] GeoDiff3D: Self-Supervised 3D Scene Generation with Geometry-Constrained 2D Diffusion Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19785v2](https://arxiv.org/pdf/2601.19785v2)**

> **作者:** Haozhi Zhu; Miaomiao Zhao; Dingyao Liu; Runze Tian; Yan Zhang; Jie Guo; Fenggen Yu
>
> **摘要:** 3D scene generation is a core technology for gaming, film/VFX, and VR/AR. Growing demand for rapid iteration, high-fidelity detail, and accessible content creation has further increased interest in this area. Existing methods broadly follow two paradigms - indirect 2D-to-3D reconstruction and direct 3D generation - but both are limited by weak structural modeling and heavy reliance on large-scale ground-truth supervision, often producing structural artifacts, geometric inconsistencies, and degraded high-frequency details in complex scenes. We propose GeoDiff3D, an efficient self-supervised framework that uses coarse geometry as a structural anchor and a geometry-constrained 2D diffusion model to provide texture-rich reference images. Importantly, GeoDiff3D does not require strict multi-view consistency of the diffusion-generated references and remains robust to the resulting noisy, inconsistent guidance. We further introduce voxel-aligned 3D feature aggregation and dual self-supervision to maintain scene coherence and fine details while substantially reducing dependence on labeled data. GeoDiff3D also trains with low computational cost and enables fast, high-quality 3D scene generation. Extensive experiments on challenging scenes show improved generalization and generation quality over existing baselines, offering a practical solution for accessible and efficient 3D scene construction.
>
---
#### [replaced 029] Energy Efficient Exact and Approximate Systolic Array Architecture for Matrix Multiplication
- **分类: cs.AR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.00778v2](https://arxiv.org/pdf/2509.00778v2)**

> **作者:** Pragun Jaswal; L. Hemanth Krishna; B. Srinivasu
>
> **备注:** 39th International Conference on VLSI Design (VLSID), 2026
>
> **摘要:** Deep Neural Networks (DNNs) require highly efficient matrix multiplication engines for complex computations. This paper presents a systolic array architecture incorporating novel exact and approximate processing elements (PEs), designed using energy-efficient positive partial product and negative partial product cells, termed as PPC and NPPC, respectively. The proposed 8-bit exact and approximate PE designs are employed in a 8x8 systolic array, which achieves a energy savings of 22% and 32%, respectively, compared to the existing design. To demonstrate their effectiveness, the proposed PEs are integrated into a systolic array (SA) for Discrete Cosine Transform (DCT) computation, achieving high output quality with a PSNR of 38.21,dB. Furthermore, in an edge detection application using convolution, the approximate PE achieves a PSNR of 30.45,dB. These results highlight the potential of the proposed design to deliver significant energy efficiency while maintaining competitive output quality, making it well-suited for error-resilient image and vision processing applications.
>
---
#### [replaced 030] Progressive $\mathcal{J}$-Invariant Self-supervised Learning for Low-Dose CT Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14180v2](https://arxiv.org/pdf/2601.14180v2)**

> **作者:** Yichao Liu; Zongru Shao; Yueyang Teng; Junwen Guo
>
> **摘要:** Self-supervised learning has been increasingly investigated for low-dose computed tomography (LDCT) image denoising, as it alleviates the dependence on paired normal-dose CT (NDCT) data, which are often difficult to collect. However, many existing self-supervised blind-spot denoising methods suffer from training inefficiencies and suboptimal performance due to restricted receptive fields. To mitigate this issue, we propose a novel Progressive $\mathcal{J}$-invariant Learning that maximizes the use of $\mathcal{J}$-invariant to enhance LDCT denoising performance. We introduce a step-wise blind-spot denoising mechanism that enforces conditional independence in a progressive manner, enabling more fine-grained learning for denoising. Furthermore, we explicitly inject a combination of controlled Gaussian and Poisson noise during training to regularize the denoising process and mitigate overfitting. Extensive experiments on the Mayo LDCT dataset demonstrate that the proposed method consistently outperforms existing self-supervised approaches and achieves performance comparable to, or better than, several representative supervised denoising methods.
>
---
#### [replaced 031] DAUNet: A Lightweight UNet Variant with Deformable Convolutions and Parameter-Free Attention for Medical Image Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.07051v2](https://arxiv.org/pdf/2512.07051v2)**

> **作者:** Adnan Munir; Muhammad Shahid Jabbar; Shujaat Khan
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Medical image segmentation plays a pivotal role in automated diagnostic and treatment planning systems. In this work, we present DAUNet, a novel lightweight UNet variant that integrates Deformable V2 Convolutions and Parameter-Free Attention (SimAM) to improve spatial adaptability and context-aware feature fusion without increasing model complexity. DAUNet's bottleneck employs dynamic deformable kernels to handle geometric variations, while the decoder and skip pathways are enhanced using SimAM attention modules for saliency-aware refinement. Extensive evaluations on two challenging datasets, FH-PS-AoP (fetal head and pubic symphysis ultrasound) and FUMPE (CT-based pulmonary embolism detection), demonstrate that DAUNet outperforms state-of-the-art models in Dice score, HD95, and ASD, while maintaining superior parameter efficiency. Ablation studies highlight the individual contributions of deformable convolutions and SimAM attention. DAUNet's robustness to missing context and low-contrast regions establishes its suitability for deployment in real-time and resource-constrained clinical environments.
>
---
#### [replaced 032] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OmniEVA，解决 embodied intelligence 中的3D适应性和物理约束问题，通过任务自适应3D接地和具身感知推理，提升任务规划能力。**

- **链接: [https://arxiv.org/pdf/2509.09332v3](https://arxiv.org/pdf/2509.09332v3)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [replaced 033] WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction
- **分类: cs.CV; eess.IV; eess.SP**

- **链接: [https://arxiv.org/pdf/2509.19073v3](https://arxiv.org/pdf/2509.19073v3)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the low-resolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.
>
---
#### [replaced 034] Atomic Depth Estimation From Noisy Electron Microscopy Data Via Deep Learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.17046v3](https://arxiv.org/pdf/2601.17046v3)**

> **作者:** Matan Leibovich; Mai Tan; Ramon Manzorro; Adria Marcos-Morales; Sreyas Mohan; Peter A. Crozier; Carlos Fernandez-Granda
>
> **摘要:** We present a novel approach for extracting 3D atomic-level information from transmission electron microscopy (TEM) images affected by significant noise. The approach is based on formulating depth estimation as a semantic segmentation problem. We address the resulting segmentation problem by training a deep convolutional neural network to generate pixel-wise depth segmentation maps using simulated data corrupted by synthetic noise. The proposed method was applied to estimate the depth of atomic columns in CeO2 nanoparticles from simulated images and real-world TEM data. Our experiments show that the resulting depth estimates are accurate, calibrated and robust to noise.
>
---
#### [replaced 035] Modality-Balanced Collaborative Distillation for Multi-Modal Domain Generalization
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.20258v3](https://arxiv.org/pdf/2511.20258v3)**

> **作者:** Xiaohan Wang; Zhangtao Cheng; Ting Zhong; Leiting Chen; Fan Zhou
>
> **摘要:** Weight Averaging (WA) has emerged as a powerful technique for enhancing generalization by promoting convergence to a flat loss landscape, which correlates with stronger out-of-distribution performance. However, applying WA directly to multi-modal domain generalization (MMDG) is challenging: differences in optimization speed across modalities lead WA to overfit to faster-converging ones in early stages, suppressing the contribution of slower yet complementary modalities, thereby hindering effective modality fusion and skewing the loss surface toward sharper, less generalizable minima. To address this issue, we propose MBCD, a unified collaborative distillation framework that retains WA's flatness-inducing advantages while overcoming its shortcomings in multi-modal contexts. MBCD begins with adaptive modality dropout in the student model to curb early-stage bias toward dominant modalities. A gradient consistency constraint then aligns learning signals between uni-modal branches and the fused representation, encouraging coordinated and smoother optimization. Finally, a WA-based teacher conducts cross-modal distillation by transferring fused knowledge to each uni-modal branch, which strengthens cross-modal interactions and steer convergence toward flatter solutions. Extensive experiments on MMDG benchmarks show that MBCD consistently outperforms existing methods, achieving superior accuracy and robustness across diverse unseen domains.
>
---
#### [replaced 036] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20707v2](https://arxiv.org/pdf/2510.20707v2)**

> **作者:** Xuyang Liu; Xiyan Gui; Yuchao Zhang; Linfeng Zhang
>
> **备注:** Accepted by ICLR 2026. Our code is available at https://github.com/xuyang-liu16/MixKV
>
> **摘要:** Recent large vision-language models (LVLMs) demonstrate remarkable capabilities in processing extended multi-modal sequences, yet the resulting key-value (KV) cache expansion creates a critical memory bottleneck that fundamentally limits deployment scalability. While existing KV cache compression methods focus on retaining high-importance KV pairs to minimize storage, they often overlook the modality-specific semantic redundancy patterns that emerge distinctively in multi-modal KV caches. In this work, we first analyze how, beyond simple importance, the KV cache in LVLMs exhibits varying levels of redundancy across attention heads. We show that relying solely on importance can only cover a subset of the full KV cache information distribution, leading to potential loss of semantic coverage. To address this, we propose MixKV, a novel method that mixes importance with diversity for optimized KV cache compression in LVLMs. MixKV adapts to head-wise semantic redundancy, selectively balancing diversity and importance when compressing KV pairs. Extensive experiments demonstrate that MixKV consistently enhances existing methods across multiple LVLMs. Under extreme compression (budget=64), MixKV improves baseline methods by an average of 5.1% across five multi-modal understanding benchmarks and achieves remarkable gains of 8.0% and 9.0% for SnapKV and AdaKV on GUI grounding tasks, all while maintaining comparable inference efficiency. Furthermore, MixKV extends seamlessly to LLMs with comparable performance gains. Our code is available at https://github.com/xuyang-liu16/MixKV.
>
---
#### [replaced 037] Improving Fine-Grained Control via Aggregation of Multiple Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.01262v5](https://arxiv.org/pdf/2410.01262v5)**

> **作者:** Conghan Yue; Zhengwei Peng; Shiyan Du; Zhi Ji; Chuangjian Cai; Le Wan; Dongyu Zhang
>
> **摘要:** While many diffusion models perform well when controlling particular aspects such as style, character, and interaction, they struggle with fine-grained control due to dataset limitations and intricate model architecture design. This paper introduces a novel training-free algorithm for fine-grained generation, called Aggregation of Multiple Diffusion Models (AMDM). The algorithm integrates features in the latent data space from multiple diffusion models within the same ecosystem into a specified model, thereby activating particular features and enabling fine-grained control. Experimental results demonstrate that AMDM significantly improves fine-grained control without training, validating its effectiveness. Additionally, it reveals that diffusion models initially focus on features such as position, attributes, and style, with later stages improving generation quality and consistency. AMDM offers a new perspective for tackling the challenges of fine-grained conditional generation in diffusion models. Specifically, it allows us to fully utilize existing or develop new conditional diffusion models that control specific aspects, and then aggregate them using the AMDM algorithm. This eliminates the need for constructing complex datasets, designing intricate model architectures, and incurring high training costs. Code is available at: https://github.com/Hammour-steak/AMDM.
>
---
#### [replaced 038] LogogramNLP: Comparing Visual and Textual Representations of Ancient Logographic Writing Systems for NLP
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决古代 logographic 语言数据难以转化为文本的问题。通过引入 LogogramNLP 基准，对比视觉与文本表示效果，探索直接处理图像的可行性。**

- **链接: [https://arxiv.org/pdf/2408.04628v2](https://arxiv.org/pdf/2408.04628v2)**

> **作者:** Danlu Chen; Freda Shi; Aditi Agarwal; Jacobo Myerston; Taylor Berg-Kirkpatrick
>
> **备注:** correct wrong refs, typos
>
> **摘要:** Standard natural language processing (NLP) pipelines operate on symbolic representations of language, which typically consist of sequences of discrete tokens. However, creating an analogous representation for ancient logographic writing systems is an extremely labor intensive process that requires expert knowledge. At present, a large portion of logographic data persists in a purely visual form due to the absence of transcription -- this issue poses a bottleneck for researchers seeking to apply NLP toolkits to study ancient logographic languages: most of the relevant data are images of writing. This paper investigates whether direct processing of visual representations of language offers a potential solution. We introduce LogogramNLP, the first benchmark enabling NLP analysis of ancient logographic languages, featuring both transcribed and visual datasets for four writing systems along with annotations for tasks like classification, translation, and parsing. Our experiments compare systems that employ recent visual and text encoding strategies as backbones. The results demonstrate that visual representations outperform textual representations for some investigated tasks, suggesting that visual processing pipelines may unlock a large amount of cultural heritage data of logographic languages for NLP-based analyses.
>
---
#### [replaced 039] Feature-Space Adversarial Robustness Certification for Multimodal Large Language Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16200v2](https://arxiv.org/pdf/2601.16200v2)**

> **作者:** Song Xia; Meiwen Ding; Chenqi Kong; Wenhan Yang; Xudong Jiang
>
> **备注:** Under review
>
> **摘要:** Multimodal large language models (MLLMs) exhibit strong capabilities across diverse applications, yet remain vulnerable to adversarial perturbations that distort their feature representations and induce erroneous predictions. To address this vulnerability, we propose Feature-space Smoothing (FS), a general framework that provides certified robustness guarantees at the feature representation level of MLLMs. We theoretically prove that FS converts a given feature extractor into a smoothed variant that is guaranteed a certified lower bound on the cosine similarity between clean and adversarial features under $\ell_2$-bounded perturbations. Moreover, we establish that the value of this Feature Cosine Similarity Bound (FCSB) is determined by the intrinsic Gaussian robustness score of the given encoder. Building on this insight, we introduce the Gaussian Smoothness Booster (GSB), a plug-and-play module that enhances the Gaussian robustness score of pretrained MLLMs, thereby strengthening the robustness guaranteed by FS, without requiring additional MLLM retraining. Extensive experiments demonstrate that applying the FS to various MLLMs yields strong certified feature-space robustness and consistently leads to robust task-oriented performance across diverse applications.
>
---
#### [replaced 040] Dense-SfM: Structure from Motion with Dense Consistent Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.14277v3](https://arxiv.org/pdf/2501.14277v3)**

> **作者:** JongMin Lee; Sungjoo Yoo
>
> **摘要:** We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods. Project page: https://icetea-cv.github.io/densesfm/.
>
---
#### [replaced 041] Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow
- **分类: cs.MA; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21789v3](https://arxiv.org/pdf/2509.21789v3)**

> **作者:** Xinlei Yu; Chengming Xu; Guibin Zhang; Yongbo He; Zhangquan Chen; Zhucun Xue; Jiangning Zhang; Yue Liao; Xiaobin Hu; Yu-Gang Jiang; Shuicheng Yan
>
> **摘要:** Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, plug-and-play mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code is publicly available at: https://github.com/YU-deep/ViF.git.
>
---
#### [replaced 042] NavFormer: IGRF Forecasting in Moving Coordinate Frames
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18800v2](https://arxiv.org/pdf/2601.18800v2)**

> **作者:** Yoontae Hwang; Dongwoo Lee; Minseok Choi; Heechan Park; Yong Sup Ihn; Daham Kim; Deok-Young Lee
>
> **摘要:** Triad magnetometer components change with sensor attitude even when the IGRF total intensity target stays invariant. NavFormer forecasts this invariant target with rotation invariant scalar features and a Canonical SPD module that stabilizes the spectrum of window level second moments of the triads without sign discontinuities. The module builds a canonical frame from a Gram matrix per window and applies state dependent spectral scaling in the original coordinates. Experiments across five flights show lower error than strong baselines in standard training, few shot training, and zero shot transfer. The code is available at: https://anonymous.4open.science/r/NavFormer-Robust-IGRF-Forecasting-for-Autonomous-Navigators-0765
>
---
#### [replaced 043] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction
- **分类: cs.CV; cs.CR; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.18988v2](https://arxiv.org/pdf/2507.18988v2)**

> **作者:** Chao Wang; Zijin Yang; Yaofei Wang; Weiming Zhang; Kejiang Chen
>
> **备注:** 7 pages. Accepted by AAAI 2026 Oral
>
> **摘要:** The rapid advancement of image-generation technologies has made it possible for anyone to create photorealistic images using generative models, raising significant security concerns. To mitigate malicious use, tracing the origin of such images is essential. Reconstruction-based attribution methods offer a promising solution, but they often suffer from reduced accuracy and high computational costs when applied to state-of-the-art (SOTA) models. To address these challenges, we propose AEDR (AutoEncoder Double-Reconstruction), a novel training-free attribution method designed for generative models with continuous autoencoders. Unlike existing reconstruction-based approaches that rely on the value of a single reconstruction loss, AEDR performs two consecutive reconstructions using the model's autoencoder, and adopts the ratio of these two reconstruction losses as the attribution signal. This signal is further calibrated using the image homogeneity metric to improve accuracy, which inherently cancels out absolute biases caused by image complexity, with autoencoder-based reconstruction ensuring superior computational efficiency. Experiments on eight top latent diffusion models show that AEDR achieves 25.5% higher attribution accuracy than existing reconstruction-based methods, while requiring only 1% of the computational time.
>
---
#### [replaced 044] SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05301v2](https://arxiv.org/pdf/2506.05301v2)**

> **作者:** Jianyi Wang; Shanchuan Lin; Zhijie Lin; Yuxi Ren; Meng Wei; Zongsheng Yue; Shangchen Zhou; Hao Chen; Yang Zhao; Ceyuan Yang; Xuefeng Xiao; Chen Change Loy; Lu Jiang
>
> **备注:** Camera Ready of ICLR2026. Project page: https://iceclear.github.io/projects/seedvr2/
>
> **摘要:** Recent advances in diffusion-based video restoration (VR) demonstrate significant improvement in visual quality, yet yield a prohibitive computational cost during inference. While several distillation-based approaches have exhibited the potential of one-step image restoration, extending existing approaches to VR remains challenging and underexplored, particularly when dealing with high-resolution video in real-world settings. In this work, we propose a one-step diffusion-based VR model, termed as SeedVR2, which performs adversarial VR training against real data. To handle the challenging high-resolution VR within a single step, we introduce several enhancements to both model architecture and training procedures. Specifically, an adaptive window attention mechanism is proposed, where the window size is dynamically adjusted to fit the output resolutions, avoiding window inconsistency observed under high-resolution VR using window attention with a predefined window size. To stabilize and improve the adversarial post-training towards VR, we further verify the effectiveness of a series of losses, including a proposed feature matching loss without significantly sacrificing training efficiency. Extensive experiments show that SeedVR2 can achieve comparable or even better performance compared with existing VR approaches in a single step.
>
---
#### [replaced 045] DA-Occ: Direction-Aware 2D Convolution for Efficient and Geometry-Preserving 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.23599v3](https://arxiv.org/pdf/2507.23599v3)**

> **作者:** Yuchen Zhou; Yan Luo; Xiaogang Wang; Xingjian Gu; Mingzhou Lu
>
> **摘要:** Efficient and high-accuracy 3D occupancy prediction is crucial for ensuring the performance of autonomous driving (AD) systems. However, many existing methods involve trade-offs between accuracy and efficiency. Some achieve high precision but with slow inference speed, while others adopt purely bird's-eye-view (BEV)-based 2D representations to accelerate processing, inevitably sacrificing vertical cues and compromising geometric integrity. To overcome these limitations, we propose a pure 2D framework that achieves efficient 3D occupancy prediction while preserving geometric integrity. Unlike conventional Lift-Splat-Shoot (LSS) methods that rely solely on depth scores to lift 2D features into 3D space, our approach additionally introduces a height-score projection to encode vertical geometric structure. We further employ direction-aware convolution to extract geometric features along both vertical and horizontal orientations, effectively balancing accuracy and computational efficiency. On the Occ3D-nuScenes, the proposed method achieves an mIoU of 39.3\% and an inference speed of 27.7 FPS, effectively balancing accuracy and efficiency. In simulations on edge devices, the inference speed reaches 14.8 FPS, further demonstrating the method's applicability for real-time deployment in resource-constrained environments.
>
---
#### [replaced 046] Diagnosing Vision Language Models' Perception by Leveraging Human Methods for Color Vision Deficiencies
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态AI研究，旨在解决视觉语言模型对色觉差异的感知问题。通过Ishihara测试评估模型，发现其无法准确模拟色觉缺陷者的视觉体验。**

- **链接: [https://arxiv.org/pdf/2505.17461v2](https://arxiv.org/pdf/2505.17461v2)**

> **作者:** Kazuki Hayashi; Shintaro Ozaki; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted to appear in the main conference of EACL 2026
>
> **摘要:** Large-scale Vision-Language Models (LVLMs) are being deployed in real-world settings that require visual inference. As capabilities improve, applications in navigation, education, and accessibility are becoming practical. These settings require accommodation of perceptual variation rather than assuming a uniform visual experience. Color perception illustrates this requirement: it is central to visual understanding yet varies across individuals due to Color Vision Deficiencies, an aspect largely ignored in multimodal AI. In this work, we examine whether LVLMs can account for variation in color perception using the Ishihara Test. We evaluate model behavior through generation, confidence, and internal representation, using Ishihara plates as controlled stimuli that expose perceptual differences. Although models possess factual knowledge about color vision deficiencies and can describe the test, they fail to reproduce the perceptual outcomes experienced by affected individuals and instead default to normative color perception. These results indicate that current systems lack mechanisms for representing alternative perceptual experiences, raising concerns for accessibility and inclusive deployment in multimodal settings.
>
---
#### [replaced 047] PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18714v2](https://arxiv.org/pdf/2510.18714v2)**

> **作者:** Changkun Liu; Bin Tan; Zeran Ke; Shangzhan Zhang; Jiachen Liu; Ming Qian; Nan Xue; Yujun Shen; Tristan Braud
>
> **备注:** Camera-ready version of a paper in 39th Conference on Neural Information Processing Systems (NeurIPS 2025). The project page is available at: https://lck666666.github.io/plana3r
>
> **摘要:** This paper addresses metric 3D reconstruction of indoor scenes by exploiting their inherent geometric regularities with compact representations. Using planar 3D primitives - a well-suited representation for man-made environments - we introduce PLANA3R, a pose-free framework for metric Planar 3D Reconstruction from unposed two-view images. Our approach employs Vision Transformers to extract a set of sparse planar primitives, estimate relative camera poses, and supervise geometry learning via planar splatting, where gradients are propagated through high-resolution rendered depth and normal maps of primitives. Unlike prior feedforward methods that require 3D plane annotations during training, PLANA3R learns planar 3D structures without explicit plane supervision, enabling scalable training on large-scale stereo datasets using only depth and normal annotations. We validate PLANA3R on multiple indoor-scene datasets with metric supervision and demonstrate strong generalization to out-of-domain indoor environments across diverse tasks under metric evaluation protocols, including 3D surface reconstruction, depth estimation, and relative pose estimation. Furthermore, by formulating with planar 3D representation, our method emerges with the ability for accurate plane segmentation. The project page is available at https://lck666666.github.io/plana3r
>
---
#### [replaced 048] TIPO: Text to Image with Text Presampling for Prompt Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.08127v4](https://arxiv.org/pdf/2411.08127v4)**

> **作者:** Shih-Ying Yeh; Sang-Hyun Park; Yi Li; Giyeong Oh; Xuehai Wang; Min Song; Youngjae Yu
>
> **备注:** 50 pages, 28 figures
>
> **摘要:** TIPO (Text-to-Image Prompt Optimization) introduces an efficient approach for automatic prompt refinement in text-to-image (T2I) generation. Starting from simple user prompts, TIPO leverages a lightweight pre-trained model to expand these prompts into richer and more detailed versions. Conceptually, TIPO samples refined prompts from a targeted sub-distribution within the broader semantic space, preserving the original intent while significantly improving visual quality, coherence, and detail. Unlike resource-intensive methods based on large language models (LLMs) or reinforcement learning (RL), TIPO offers strong computational efficiency and scalability, opening new possibilities for effective automated prompt engineering in T2I tasks. Extensive experiments across multiple domains demonstrate that TIPO achieves stronger text alignment, reduced visual artifacts, and consistently higher human preference rates, while maintaining competitive aesthetic quality. These results highlight the effectiveness of distribution-aligned prompt engineering and point toward broader opportunities for scalable, automated refinement in text-to-image generation.
>
---
#### [replaced 049] Predictor-Free and Hardware-Aware Federated Neural Architecture Search via Pareto-Guided Supernet Training
- **分类: cs.LG; cs.CV; cs.DC**

- **链接: [https://arxiv.org/pdf/2601.15127v2](https://arxiv.org/pdf/2601.15127v2)**

> **作者:** Bostan Khan; Masoud Daneshtalab
>
> **备注:** This paper significantly extends the preliminary work accepted at ESANN 2026. Source Code: https://github.com/bostankhan6/DeepFedNAS
>
> **摘要:** Federated Neural Architecture Search (FedNAS) aims to automate model design for privacy-preserving Federated Learning (FL) but currently faces two critical bottlenecks: unguided supernet training that yields suboptimal models, and costly multi-hour pipelines for post-training subnet discovery. We introduce DeepFedNAS, a novel, two-phase framework underpinned by a multi-objective fitness function that synthesizes mathematical network design with architectural heuristics. Enabled by a re-engineered supernet, DeepFedNAS introduces Federated Pareto Optimal Supernet Training, which leverages a pre-computed Pareto-optimal cache of high-fitness architectures as an intelligent curriculum to optimize shared supernet weights. Subsequently, its Predictor-Free Search Method eliminates the need for costly accuracy surrogates by utilizing this fitness function as a direct, zero-cost proxy for accuracy, enabling on-demand subnet discovery in mere seconds. DeepFedNAS achieves state-of-the-art accuracy (e.g., up to 1.21% absolute improvement on CIFAR-100), superior parameter and communication efficiency, and a substantial ~61x speedup in total post-training search pipeline time. By reducing the pipeline from over 20 hours to approximately 20 minutes (including initial cache generation) and enabling 20-second individual subnet searches, DeepFedNAS makes hardware-aware FL deployments instantaneous and practical. The complete source code and experimental scripts are available at: https://github.com/bostankhan6/DeepFedNAS
>
---
#### [replaced 050] WaterFlow: Explicit Physics-Prior Rectified Flow for Underwater Saliency Mask Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12605v2](https://arxiv.org/pdf/2510.12605v2)**

> **作者:** Runting Li; Shijie Lian; Hua Li; Yutong Li; Wenhui Wu; Sam Kwong
>
> **摘要:** Underwater Salient Object Detection (USOD) faces significant challenges, including underwater image quality degradation and domain gaps. Existing methods tend to ignore the physical principles of underwater imaging or simply treat degradation phenomena in underwater images as interference factors that must be eliminated, failing to fully exploit the valuable information they contain. We propose WaterFlow, a rectified flow-based framework for underwater salient object detection that innovatively incorporates underwater physical imaging information as explicit priors directly into the network training process and introduces temporal dimension modeling, significantly enhancing the model's capability for salient object identification. On the USOD10K dataset, WaterFlow achieves a 0.072 gain in S_m, demonstrating the effectiveness and superiority of our method. https://github.com/Theo-polis/WaterFlow.
>
---
#### [replaced 051] Splat Feature Solver
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12216v2](https://arxiv.org/pdf/2508.12216v2)**

> **作者:** Butian Xiong; Rong Liu; Kenneth Xu; Meida Chen; Andrew Feng
>
> **备注:** ICLR 2026 Accepted
>
> **摘要:** Feature lifting has emerged as a crucial component in 3D scene understanding, enabling the attachment of rich image feature descriptors (e.g., DINO, CLIP) onto splat-based 3D representations. The core challenge lies in optimally assigning rich general attributes to 3D primitives while addressing the inconsistency issues from multi-view images. We present a unified, kernel- and feature-agnostic formulation of the feature lifting problem as a sparse linear inverse problem, which can be solved efficiently in closed form. Our approach admits a provable upper bound on the global optimal error under convex losses for delivering high quality lifted features. To address inconsistencies and noise in multi-view observations, we introduce two complementary regularization strategies to stabilize the solution and enhance semantic fidelity. Tikhonov Guidance enforces numerical stability through soft diagonal dominance, while Post-Lifting Aggregation filters noisy inputs via feature clustering. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on open-vocabulary 3D segmentation benchmarks, outperforming training-based, grouping-based, and heuristic-forward baselines while producing lifted features in minutes. Our \textbf{code} is available in the \href{https://github.com/saliteta/splat-distiller/tree/main}{\textcolor{blue}{GitHub}}. We provide additional \href{https://splat-distiller.pages.dev/}{\textcolor{blue}{website}} for more visualization, as well as the \href{https://www.youtube.com/watch?v=CH-G5hbvArM}{\textcolor{blue}{video}}.
>
---
#### [replaced 052] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.01256v3](https://arxiv.org/pdf/2412.01256v3)**

> **作者:** Bikang Pan; Qun Li; Xiaoying Tang; Wei Huang; Zhen Fang; Feng Liu; Jingya Wang; Jingyi Yu; Ye Shi
>
> **摘要:** The emergence of vision-language foundation models, such as CLIP, has revolutionized image-text representation, enabling a broad range of applications via prompt learning. Despite its promise, real-world datasets often contain noisy labels that can degrade prompt learning performance. In this paper, we demonstrate that using mean absolute error (MAE) loss in prompt learning, named PromptMAE, significantly enhances robustness against noisy labels while maintaining high accuracy. Though MAE is straightforward and recognized for its robustness, it is rarely used in noisy-label learning due to its slow convergence and poor performance outside prompt learning scenarios. To elucidate the robustness of PromptMAE, we leverage feature learning theory to show that MAE can suppress the influence of noisy samples, thereby improving the signal-to-noise ratio and enhancing overall robustness. Additionally, we introduce PromptOT, a prompt-based optimal transport data purification method to enhance the robustness further. PromptOT employs text features in vision-language models as prototypes to construct an optimal transportation matrix. This matrix effectively partitions datasets into clean and noisy subsets, allowing for the application of cross-entropy loss to the clean subset and MAE loss to the noisy subset. Our Noise-Label Prompt Learning method, named NLPrompt, offers a simple and efficient approach that leverages the expressive representations and precise alignment capabilities of vision-language models for robust prompt learning. We validate NLPrompt through extensive experiments across various noise settings, demonstrating significant performance improvements.
>
---
#### [replaced 053] Spatiotemporal Semantic V2X Framework for Cooperative Collision Prediction
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.17216v2](https://arxiv.org/pdf/2601.17216v2)**

> **作者:** Murat Arda Onsu; Poonam Lohan; Burak Kantarci; Aisha Syed; Matthew Andrews; Sean Kennedy
>
> **备注:** 6 pages 5 figures, accepted to IEEE ICC 2026
>
> **摘要:** Intelligent Transportation Systems (ITS) demand real-time collision prediction to ensure road safety and reduce accident severity. Conventional approaches rely on transmitting raw video or high-dimensional sensory data from roadside units (RSUs) to vehicles, which is impractical under vehicular communication bandwidth and latency constraints. In this work, we propose a semantic V2X framework in which RSU-mounted cameras generate spatiotemporal semantic embeddings of future frames using the Video Joint Embedding Predictive Architecture (V-JEPA). To evaluate the system, we construct a digital twin of an urban traffic environment enabling the generation of d verse traffic scenarios with both safe and collision events. These embeddings of the future frame, extracted from V-JEPA, capture task-relevant traffic dynamics and are transmitted via V2X links to vehicles, where a lightweight attentive probe and classifier decode them to predict imminent collisions. By transmitting only semantic embeddings instead of raw frames, the proposed system significantly reduces communication overhead while maintaining predictive accuracy. Experimental results demonstrate that the framework with an appropriate processing method achieves a 10% F1-score improvement for collision prediction while reducing transmission requirements by four orders of magnitude compared to raw video. This validates the potential of semantic V2X communication to enable cooperative, real-time collision prediction in ITS.
>
---
#### [replaced 054] DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.08005v5](https://arxiv.org/pdf/2502.08005v5)**

> **作者:** Wenlin Chen; Mingtian Zhang; Jiajun He; Zijing Ou; José Miguel Hernández-Lobato; Bernhard Schölkopf; David Barber
>
> **备注:** 22 pages, 8 figures, 5 tables, 2 algorithms
>
> **摘要:** Score-based distillation methods (e.g., variational score distillation) train one-step diffusion models by first pre-training a teacher score model and then distilling it into a one-step student model. However, the gradient estimator in the distillation stage usually suffers from two sources of bias: (1) biased teacher supervision due to score estimation error incurred during pre-training, and (2) the student model's score estimation error during distillation. These biases can degrade the quality of the resulting one-step diffusion model. To address this, we propose DiffRatio, a new framework for training one-step diffusion models: instead of estimating the teacher and student scores independently and then taking their difference, we directly estimate the score difference as the gradient of a learned log density ratio between the student and data distributions across diffusion time steps. This approach greatly simplifies the training pipeline, significantly reduces gradient estimation bias, and improves one-step generation quality. Additionally, it also reduces auxiliary network size by using a lightweight density-ratio network instead of two full score networks, which improves computational and memory efficiency. DiffRatio achieves competitive one-step generation results on CIFAR-10 and ImageNet (64x64 and 512x512), outperforming most teacher-supervised distillation methods. Moreover, the learned density ratio naturally serves as a verifier, enabling a principled inference-time parallel scaling scheme that further improves the generation quality without external rewards or additional sequential computation.
>
---
#### [replaced 055] MSPCaps: A Multi-Scale Patchify Capsule Network with Cross-Agreement Routing for Visual Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.16922v2](https://arxiv.org/pdf/2508.16922v2)**

> **作者:** Yudong Hu; Yueju Han; Rui Sun; Jinke Ren
>
> **备注:** 9 pages, 4 figures; Code is available at https://github.com/abdn-hyd/MSPCaps
>
> **摘要:** Capsule Network (CapsNet) has demonstrated significant potential in visual recognition by capturing spatial relationships and part-whole hierarchies for learning equivariant feature representations. However, existing CapsNet and variants often rely on a single high-level feature map, overlooking the rich complementary information from multi-scale features. Furthermore, conventional feature fusion strategies (e.g., addition and concatenation) struggle to reconcile multi-scale feature discrepancies, leading to suboptimal classification performance. To address these limitations, we propose the Multi-Scale Patchify Capsule Network (MSPCaps), a novel architecture that integrates multi-scale feature learning and efficient capsule routing. Specifically, MSPCaps consists of three key components: a Multi-Scale ResNet Backbone (MSRB), a Patchify Capsule Layer (PatchifyCaps), and Cross-Agreement Routing (CAR) blocks. First, the MSRB extracts diverse multi-scale feature representations from input images, preserving both fine-grained details and global contextual information. Second, the PatchifyCaps partitions these multi-scale features into primary capsules using a uniform patch size, equipping the model with the ability to learn from diverse receptive fields. Finally, the CAR block adaptively routes the multi-scale capsules by identifying cross-scale prediction pairs with maximum agreement. Unlike the simple concatenation of multiple self-routing blocks, CAR ensures that only the most coherent capsules contribute to the final voting. Our proposed MSPCaps achieves remarkable scalability and superior robustness, consistently surpassing multiple baseline methods in terms of classification accuracy, with configurations ranging from a highly efficient Tiny model (344.3K parameters) to a powerful Large model (10.9M parameters), highlighting its potential in advancing feature representation learning.
>
---
#### [replaced 056] Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues
- **分类: cs.MM; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2601.19750v2](https://arxiv.org/pdf/2601.19750v2)**

> **作者:** Junchen Fu; Wenhao Deng; Kaiwen Zheng; Ioannis Arapakis; Yu Ye; Yongxin Ni; Joemon M. Jose; Xuri Ge
>
> **摘要:** Missing-modality information on e-commerce platforms, such as absent product images or textual descriptions, often arises from annotation errors or incomplete metadata, impairing both product presentation and downstream applications such as recommendation systems. Motivated by the multimodal generative capabilities of recent Multimodal Large Language Models (MLLMs), this work investigates a fundamental yet underexplored question: can MLLMs generate missing modalities for products in e-commerce scenarios? We propose the Missing Modality Product Completion Benchmark (MMPCBench), which consists of two sub-benchmarks: a Content Quality Completion Benchmark and a Recommendation Benchmark. We further evaluate six state-of-the-art MLLMs from the Qwen2.5-VL and Gemma-3 model families across nine real-world e-commerce categories, focusing on image-to-text and text-to-image completion tasks. Experimental results show that while MLLMs can capture high-level semantics, they struggle with fine-grained word-level and pixel- or patch-level alignment. In addition, performance varies substantially across product categories and model scales, and we observe no trivial correlation between model size and performance, in contrast to trends commonly reported in mainstream benchmarks. We also explore Group Relative Policy Optimization (GRPO) to better align MLLMs with this task. GRPO improves image-to-text completion but does not yield gains for text-to-image completion. Overall, these findings expose the limitations of current MLLMs in real-world cross-modal generation and represent an early step toward more effective missing-modality product completion.
>
---
#### [replaced 057] SiNGER: A Clearer Voice Distills Vision Transformers Further
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.20986v3](https://arxiv.org/pdf/2509.20986v3)**

> **作者:** Geunhyeok Yu; Sunjae Jeong; Yoonyoung Choi; Jaeseung Kim; Hyoseok Hwang
>
> **备注:** Main paper: 12 pages (including 3 pages of references), 6 figures, 6 tables. Appendix: 9 pages, 7 figures. ICLR 2026 accepted
>
> **摘要:** Vision Transformers are widely adopted as the backbone of vision foundation models, but they are known to produce high-norm artifacts that degrade representation quality. When knowledge distillation transfers these features to students, high-norm artifacts dominate the objective, so students overfit to artifacts and underweight informative signals, diminishing the gains from larger models. Prior work attempted to remove artifacts but encountered an inherent trade-off between artifact suppression and preserving informative signals from teachers. To address this, we introduce Singular Nullspace-Guided Energy Reallocation (SiNGER), a novel distillation framework that suppresses artifacts while preserving informative signals. The key idea is principled teacher feature refinement: during refinement, we leverage the nullspace-guided perturbation to preserve information while suppressing artifacts. Then, the refined teacher's features are distilled to a student. We implement this perturbation efficiently with a LoRA-based adapter that requires minimal structural modification. Extensive experiments show that \oursname consistently improves student models, achieving state-of-the-art performance in multiple downstream tasks and producing clearer and more interpretable representations.
>
---
#### [replaced 058] Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于移动服务机器人领域，探讨如何将基础模型应用于具身AI，解决指令理解、多模态感知、不确定性处理和实时部署等问题。**

- **链接: [https://arxiv.org/pdf/2505.20503v2](https://arxiv.org/pdf/2505.20503v2)**

> **作者:** Matthew Lisondra; Beno Benhabib; Goldie Nejat
>
> **备注:** v2: Expanded systematic review; resubmitted to Robotics
>
> **摘要:** Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models, have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interaction, mobile service robots can achieve more flexible understanding, adaptive behavior, and robust task execution in dynamic real-world environments. Despite this progress, embodied AI for mobile service robots continues to face fundamental challenges related to the translation of natural language instructions into executable robot actions, multimodal perception in human-centered environments, uncertainty estimation for safe decision-making, and computational constraints for real-time onboard deployment. In this paper, we present the first systematic review focused specifically on the integration of foundation models in mobile service robotics. We analyze how recent advances in foundation models address these core challenges through language-conditioned control, multimodal sensor fusion, uncertainty-aware reasoning, and efficient model scaling. We further examine real-world applications in domestic assistance, healthcare, and service automation, highlighting how foundation models enable context-aware, socially responsive, and generalizable robot behaviors. Beyond technical considerations, we discuss ethical, societal, and human-interaction implications associated with deploying foundation model-enabled service robots in human environments. Finally, we outline future research directions emphasizing reliability and lifelong adaptation, privacy-aware and resource-constrained deployment, and governance and human-in-the-loop frameworks required for safe, scalable, and trustworthy mobile service robotics.
>
---
#### [replaced 059] CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.00753v2](https://arxiv.org/pdf/2504.00753v2)**

> **作者:** Elyar Esmaeilzadeh; Ehsan Garaaghaji; Farzad Hallaji Azad; Doruk Oner
>
> **摘要:** Promoting the connectivity of curvilinear structures, such as neuronal processes in biomedical scans and blood vessels in CT images, remains a key challenge in semantic segmentation. Traditional pixel-wise loss functions, including cross-entropy and Dice losses, often fail to capture high-level topological connectivity, resulting in topological mistakes in graphs obtained from prediction maps. In this paper, we propose CAPE (Connectivity-Aware Path Enforcement), a novel loss function designed to enforce connectivity in graphs obtained from segmentation maps by optimizing a graph connectivity metric. CAPE uses the graph representation of the ground truth to select node pairs and determine their corresponding paths within the predicted segmentation through a shortest-path algorithm. Using this, we penalize both disconnections and false positive connections, effectively promoting the model to preserve topological correctness. Experiments on 2D and 3D datasets, including neuron and blood vessel tracing demonstrate that CAPE significantly improves topology-aware metrics and outperforms state-of-the-art methods.
>
---
#### [replaced 060] Neuro Symbolic Knowledge Reasoning for Procedural Video Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.14957v5](https://arxiv.org/pdf/2503.14957v5)**

> **作者:** Basura Fernando; Thanh-Son Nguyen; Hong Yang; Tzeh Yuan Neoh; Hao Zhang; Ee Yeo Keat
>
> **备注:** This paper is under review at IEEE Transactions on Neural Networks and Learning Systems. Personal use is permitted, but republication/redistribution requires IEEE permission
>
> **摘要:** In this work we present Knowledge Module Learning (KML) to understand and reason over procedural tasks that requires models to learn structured and compositional procedural knowledge. KML is a neurosymbolic framework that learns relation categories within a knowledge graph as neural knowledge modules and composes them into executable reasoning programs generated by large language models (LLMs). Each module encodes a specific procedural relation capturing how each entity type such as tools are related to steps, purpose of each tool, and steps of each task. Given a question conditioned on a task shown in a video, then KML performs multistep reasoning with transparent, traceable intermediate states. Our theoretical analysis demonstrated two desired properties of KML. KML satisfy strong optimal conditions for modelling KG relations as neural mappings, providing strong foundations for generalizable procedural reasoning. It also shows a bound on the expected error when it performs multistep reasoning. To evaluate this model, we construct a large procedural knowledge graph (PKG) consisting of diverse instructional domains by integrating the COIN instructional video dataset, and COIN ontology, commonsense relations from ConceptNet, and structured extractions from LLMs, followed by expert verification. We then generate question and answer pairs by applying graph traversal templates over the PKG, constructing the PKR-QA benchmark for procedural knowledge reasoning. Experiments show that KML improves structured reasoning performance while providing interpretable step-by-step traces, outperforming LLM-only and black-box neural baselines. Code is publicly available at https://github.com/LUNAProject22/KML.
>
---
#### [replaced 061] The SAGES Critical View of Safety Challenge: A Global Benchmark for AI-Assisted Surgical Quality Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17100v2](https://arxiv.org/pdf/2509.17100v2)**

> **作者:** Deepak Alapatt; Jennifer Eckhoff; Zhiliang Lyu; Yutong Ban; Jean-Paul Mazellier; Sarah Choksi; Kunyi Yang; Po-Hsing Chiang; Noemi Zorzetti; Samuele Cannas; Daniel Neimark; Omri Bar; Amine Yamlahi; Jakob Hennighausen; Xiaohan Wang; Rui Li; Long Liang; Yuxian Wang; Saurabh Koju; Binod Bhattarai; Tim Jaspers; Zhehua Mao; Anjana Wijekoon; Jun Ma; Yinan Xu; Zhilong Weng; Ammar M. Okran; Hatem A. Rashwan; Boyang Shen; Kaixiang Yang; Yutao Zhang; Hao Wang; 2024 CVS Challenge Consortium; Quanzheng Li; Filippo Filicori; Xiang Li; Pietro Mascagni; Daniel A. Hashimoto; Guy Rosman; Ozanan Meireles; Nicolas Padoy
>
> **备注:** 21 pages, 10 figures
>
> **摘要:** Advances in artificial intelligence (AI) for surgical quality assessment promise to democratize access to expertise, with applications in training, guidance, and accreditation. This study presents the SAGES Critical View of Safety (CVS) Challenge, the first AI competition organized by a surgical society, using the CVS in laparoscopic cholecystectomy, a universally recommended yet inconsistently performed safety step, as an exemplar of surgical quality assessment. A global collaboration across 54 institutions in 24 countries engaged hundreds of clinicians and engineers to curate 1,000 videos annotated by 20 surgical experts according to a consensus-validated protocol. The challenge addressed key barriers to real-world deployment in surgery, including achieving high performance, capturing uncertainty in subjective assessment, and ensuring robustness to clinical variability. To enable this scale of effort, we developed EndoGlacier, a framework for managing large, heterogeneous surgical video and multi-annotator workflows. Thirteen international teams participated, achieving up to a 17% relative gain in assessment performance, over 80% reduction in calibration error, and a 17% relative improvement in robustness over the state-of-the-art. Analysis of results highlighted methodological trends linked to model performance, providing guidance for future research toward robust, clinically deployable AI for surgical quality assessment.
>
---
#### [replaced 062] CLIP-Guided Unsupervised Semantic-Aware Exposure Correction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.19129v2](https://arxiv.org/pdf/2601.19129v2)**

> **作者:** Puzhen Wu; Han Weng; Quan Zheng; Yi Zhan; Hewei Wang; Yiming Li; Jiahui Han; Rui Xu
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Improper exposure often leads to severe loss of details, color distortion, and reduced contrast. Exposure correction still faces two critical challenges: (1) the ignorance of object-wise regional semantic information causes the color shift artifacts; (2) real-world exposure images generally have no ground-truth labels, and its labeling entails massive manual editing. To tackle the challenges, we propose a new unsupervised semantic-aware exposure correction network. It contains an adaptive semantic-aware fusion module, which effectively fuses the semantic information extracted from a pre-trained Fast Segment Anything Model into a shared image feature space. Then the fused features are used by our multi-scale residual spatial mamba group to restore the details and adjust the exposure. To avoid manual editing, we propose a pseudo-ground truth generator guided by CLIP, which is fine-tuned to automatically identify exposure situations and instruct the tailored corrections. Also, we leverage the rich priors from the FastSAM and CLIP to develop a semantic-prompt consistency loss to enforce semantic consistency and image-prompt alignment for unsupervised training. Comprehensive experimental results illustrate the effectiveness of our method in correcting real-world exposure images and outperforms state-of-the-art unsupervised methods both numerically and visually.
>
---
#### [replaced 063] FaithSCAN: Model-Driven Single-Pass Hallucination Detection for Faithful Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.00269v2](https://arxiv.org/pdf/2601.00269v2)**

> **作者:** Chaodong Tong; Qi Zhang; Chen Li; Lei Jiang; Yanbing Liu
>
> **备注:** 21 pages, 13 figures, 8 tables
>
> **摘要:** Faithfulness hallucinations in VQA occur when vision-language models produce fluent yet visually ungrounded answers, severely undermining their reliability in safety-critical applications. Existing detection methods mainly fall into two categories: external verification approaches relying on auxiliary models or knowledge bases, and uncertainty-driven approaches using repeated sampling or uncertainty estimates. The former suffer from high computational overhead and are limited by external resource quality, while the latter capture only limited facets of model uncertainty and fail to sufficiently explore the rich internal signals associated with the diverse failure modes. Both paradigms thus have inherent limitations in efficiency, robustness, and detection performance. To address these challenges, we propose FaithSCAN: a lightweight network that detects hallucinations by exploiting rich internal signals of VLMs, including token-level decoding uncertainty, intermediate visual representations, and cross-modal alignment features. These signals are fused via branch-wise evidence encoding and uncertainty-aware attention. We also extend the LLM-as-a-Judge paradigm to VQA hallucination and propose a low-cost strategy to automatically generate model-dependent supervision signals, enabling supervised training without costly human labels while maintaining high detection accuracy. Experiments on multiple VQA benchmarks show that FaithSCAN significantly outperforms existing methods in both effectiveness and efficiency. In-depth analysis shows hallucinations arise from systematic internal state variations in visual perception, cross-modal reasoning, and language decoding. Different internal signals provide complementary diagnostic cues, and hallucination patterns vary across VLM architectures, offering new insights into the underlying causes of multimodal hallucinations.
>
---
#### [replaced 064] Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03553v2](https://arxiv.org/pdf/2512.03553v2)**

> **作者:** Wei Chee Yew; Hailun Xu; Sanjay Saha; Xiaotian Fan; Hiok Hian Ong; David Yuchen Wang; Kanchan Sarkar; Zhenheng Yang; Danhui Guan
>
> **备注:** To be published at KDD 2026 (ADS track)
>
> **摘要:** Content moderation remains a critical yet challenging task for large-scale user-generated video platforms, especially in livestreaming environments where moderation must be timely, multimodal, and robust to evolving forms of unwanted content. We present a hybrid moderation framework deployed at production scale that combines supervised classification for known violations with reference-based similarity matching for novel or subtle cases. This hybrid design enables robust detection of both explicit violations and novel edge cases that evade traditional classifiers. Multimodal inputs (text, audio, visual) are processed through both pipelines, with a multimodal large language model (MLLM) distilling knowledge into each to boost accuracy while keeping inference lightweight. In production, the classification pipeline achieves 67% recall at 80% precision, and the similarity pipeline achieves 76% recall at 80% precision. Large-scale A/B tests show a 6-8% reduction in user views of unwanted livestreams}. These results demonstrate a scalable and adaptable approach to multimodal content governance, capable of addressing both explicit violations and emerging adversarial behaviors.
>
---
#### [replaced 065] TBC: A Target-Background Contrast Metric for Low-Altitude Infrared and Visible Image Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15211v3](https://arxiv.org/pdf/2512.15211v3)**

> **作者:** Yufeng Xie; Cong Wang
>
> **备注:** In the subsequent research, we discovered that the research methods employed in the article were logically unsound and had flaws, making it impossible to draw reliable conclusions. Therefore, we believe it is necessary to retract this article for correction
>
> **摘要:** Infrared and visible image fusion (IVIF) is a pivotal technology in low-altitude Unmanned Aerial Vehicle (UAV) reconnaissance missions, enabling robust target detection and tracking by integrating thermal saliency with environmental textures. However, traditional no-reference metrics (Statistics-based metrics and Gradient-based metrics) fail in complex low-light environments, termed the ``Noise Trap''. This paper mathematically prove that these metrics are positively correlated with high-frequency sensor noise, paradoxically assigning higher scores to degraded images and misguiding algorithm optimization. To address this, we propose the Target-Background Contrast (TBC) metric. Inspired by Weber's Law, TBC focuses on the relative contrast of salient targets rather than global statistics. Unlike traditional metrics, TBC penalizes background noise and rewards target visibility. Extensive experiments on the DroneVehicle dataset demonstrate the superiority of TBC. Results show that TBC exhibits high ``Semantic Discriminability'' in distinguishing thermal targets from background clutter. Furthermore, TBC achieves remarkable computational efficiency, making it a reliable and real-time standard for intelligent UAV systems.
>
---
#### [replaced 066] X-SAM: From Segment Anything to Any Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.04655v2](https://arxiv.org/pdf/2508.04655v2)**

> **作者:** Hao Wang; Limeng Qiao; Zequn Jie; Zhijian Huang; Chengjian Feng; Qingfang Zheng; Lin Ma; Xiangyuan Lan; Xiaodan Liang
>
> **备注:** AAAI2026
>
> **摘要:** Large Language Models (LLMs) demonstrate strong capabilities in broad knowledge representation, yet they are inherently deficient in pixel-level perceptual understanding. Although the Segment Anything Model (SAM) represents a significant advancement in visual-prompt-driven image segmentation, it exhibits notable limitations in multi-mask prediction and category-specific segmentation tasks, and it cannot integrate all segmentation tasks within a unified model architecture. To address these limitations, we present X-SAM, a streamlined Multimodal Large Language Model (MLLM) framework that extends the segmentation paradigm from \textit{segment anything} to \textit{any segmentation}. Specifically, we introduce a novel unified framework that enables more advanced pixel-level perceptual comprehension for MLLMs. Furthermore, we propose a new segmentation task, termed Visual GrounDed (VGD) segmentation, which segments all instance objects with interactive visual prompts and empowers MLLMs with visual grounded, pixel-wise interpretative capabilities. To enable effective training on diverse data sources, we present a unified training strategy that supports co-training across multiple datasets. Experimental results demonstrate that X-SAM achieves state-of-the-art performance on a wide range of image segmentation benchmarks, highlighting its efficiency for multimodal, pixel-level visual understanding. Code is available at https://github.com/wanghao9610/X-SAM.
>
---
#### [replaced 067] Weakly supervised framework for wildlife detection and counting in challenging Arctic environments: a case study on caribou (Rangifer tarandus)
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18891v2](https://arxiv.org/pdf/2601.18891v2)**

> **作者:** Ghazaleh Serati; Samuel Foucher; Jerome Theau
>
> **备注:** 30 pages, 8 figures, submitted to Frontiers in Ecology and Evolution
>
> **摘要:** Caribou across the Arctic has declined in recent decades, motivating scalable and accurate monitoring approaches to guide evidence-based conservation actions and policy decisions. Manual interpretation from this imagery is labor-intensive and error-prone, underscoring the need for automatic and reliable detection across varying scenes. Yet, such automatic detection is challenging due to severe background heterogeneity, dominant empty terrain (class imbalance), small or occluded targets, and wide variation in density and scale. To make the detection model (HerdNet) more robust to these challenges, a weakly supervised patch-level pretraining based on a detection network's architecture is proposed. The detection dataset includes five caribou herds distributed across Alaska. By learning from empty vs. non-empty labels in this dataset, the approach produces early weakly supervised knowledge for enhanced detection compared to HerdNet, which is initialized from generic weights. Accordingly, the patch-based pretrain network attained high accuracy on multi-herd imagery (2017) and on an independent year's (2019) test sets (F1: 93.7%/92.6%, respectively), enabling reliable mapping of regions containing animals to facilitate manual counting on large aerial imagery. Transferred to detection, initialization from weakly supervised pretraining yielded consistent gains over ImageNet weights on both positive patches (F1: 92.6%/93.5% vs. 89.3%/88.6%), and full-image counting (F1: 95.5%/93.3% vs. 91.5%/90.4%). Remaining limitations are false positives from animal-like background clutter and false negatives related to low animal density occlusions. Overall, pretraining on coarse labels prior to detection makes it possible to rely on weakly-supervised pretrained weights even when labeled data are limited, achieving results comparable to generic-weight initialization.
>
---
#### [replaced 068] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于Talking Head Generation任务，解决扩散模型推理慢和非自回归范式限制的问题。提出REST框架，通过紧凑视频潜在空间、ID-Context Cache和ASD策略实现实时端到端生成。**

- **链接: [https://arxiv.org/pdf/2512.11229v2](https://arxiv.org/pdf/2512.11229v2)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation (THG). However, slow inference speeds and prevalent non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, a pioneering diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through a spatiotemporal variational autoencoder with a high compression ratio. Additionally, to enable semi-autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles into key-value caching for maintaining identity consistency and temporal coherence during long-term streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) strategy is proposed to mitigate error accumulation and enhance temporal consistency in streaming generation, leveraging a non-streaming teacher with an asynchronous noise schedule to supervise the streaming student. REST bridges the gap between autoregressive and diffusion-based approaches, achieving a breakthrough in efficiency for applications requiring real-time THG. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [replaced 069] JAFAR: Jack up Any Feature at Any Resolution
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.11136v3](https://arxiv.org/pdf/2506.11136v3)**

> **作者:** Paul Couairon; Loick Chambon; Louis Serrano; Jean-Emmanuel Haugeard; Matthieu Cord; Nicolas Thome
>
> **备注:** Code available at https://github.com/PaulCouairon/JAFAR
>
> **摘要:** Foundation Vision Encoders have become essential for a wide range of dense vision tasks. However, their low-resolution spatial feature outputs necessitate feature upsampling to produce the high-resolution modalities required for downstream tasks. In this work, we introduce JAFAR, a lightweight and flexible feature upsampler that enhances the spatial resolution of visual features from any Foundation Vision Encoder to an arbitrary target resolution. JAFAR employs an attention-based module designed to promote semantic alignment between high-resolution queries, derived from low-level image features, and semantically enriched low-resolution keys, using Spatial Feature Transform (SFT) modulation. Notably, despite the absence of high-resolution supervision, we demonstrate that learning at low upsampling ratios and resolutions generalizes remarkably well to significantly higher output scales. Extensive experiments show that JAFAR effectively recovers fine-grained spatial details and consistently outperforms existing feature upsampling methods across a diverse set of downstream tasks. Project page at https://jafar-upsampler.github.io
>
---
#### [replaced 070] AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出AdaReasoner，解决多模态大模型中的视觉推理问题，通过动态工具调度实现高效任务处理。**

- **链接: [https://arxiv.org/pdf/2601.18631v2](https://arxiv.org/pdf/2601.18631v2)**

> **作者:** Mingyang Song; Haoyu Sun; Jiawei Gu; Linjie Li; Luxin Xu; Ranjay Krishna; Yu Cheng
>
> **备注:** 28 pages, 10 figures and 13 tables
>
> **摘要:** When humans face problems beyond their immediate capabilities, they rely on tools, providing a promising paradigm for improving visual reasoning in multimodal large language models (MLLMs). Effective reasoning, therefore, hinges on knowing which tools to use, when to invoke them, and how to compose them over multiple steps, even when faced with new tools or new tasks. We introduce \textbf{AdaReasoner}, a family of multimodal models that learn tool use as a general reasoning skill rather than as tool-specific or explicitly supervised behavior. AdaReasoner is enabled by (i) a scalable data curation pipeline exposing models to long-horizon, multi-step tool interactions; (ii) Tool-GRPO, a reinforcement learning algorithm that optimizes tool selection and sequencing based on end-task success; and (iii) an adaptive learning mechanism that dynamically regulates tool usage. Together, these components allow models to infer tool utility from task context and intermediate outcomes, enabling coordination of multiple tools and generalization to unseen tools. Empirically, AdaReasoner exhibits strong tool-adaptive and generalization behaviors: it autonomously adopts beneficial tools, suppresses irrelevant ones, and adjusts tool usage frequency based on task demands, despite never being explicitly trained to do so. These capabilities translate into state-of-the-art performance across challenging benchmarks, improving the 7B base model by +24.9\% on average and surpassing strong proprietary systems such as GPT-5 on multiple tasks, including VSP and Jigsaw.
>
---
#### [replaced 071] Range Image-Based Implicit Neural Compression for LiDAR Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.17229v2](https://arxiv.org/pdf/2504.17229v2)**

> **作者:** Akihiro Kuwabara; Sorachi Kato; Toshiaki Koike-Akino; Takuya Fujihashi
>
> **备注:** Accepted for publication in IEEE Access
>
> **摘要:** This paper presents a novel scheme to efficiently compress Light Detection and Ranging~(LiDAR) point clouds, enabling high-precision 3D scene archives, and such archives pave the way for a detailed understanding of the corresponding 3D scenes. We focus on 2D range images~(RIs) as a lightweight format for representing 3D LiDAR observations. Although conventional image compression techniques can be adapted to improve compression efficiency for RIs, their practical performance is expected to be limited due to differences in bit precision and the distinct pixel value distribution characteristics between natural images and RIs. We propose a novel implicit neural representation~(INR)--based RI compression method that effectively handles floating-point valued pixels. The proposed method divides RIs into depth and mask images and compresses them using patch-wise and pixel-wise INR architectures with model pruning and quantization, respectively. Experiments on the KITTI dataset show that the proposed method outperforms existing image, point cloud, RI, and INR-based compression methods in terms of 3D reconstruction and detection quality at low bitrates and decoding latency.
>
---
#### [replaced 072] Dynamic Novel View Synthesis in High Dynamic Range
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21853v3](https://arxiv.org/pdf/2509.21853v3)**

> **作者:** Kaixuan Zhang; Zhipeng Xiong; Minxian Li; Mingwu Ren; Jiankang Deng; Xiatian Zhu
>
> **备注:** It has been accepted by ICLR 2026
>
> **摘要:** High Dynamic Range Novel View Synthesis (HDR NVS) seeks to learn an HDR 3D model from Low Dynamic Range (LDR) training images captured under conventional imaging conditions. Current methods primarily focus on static scenes, implicitly assuming all scene elements remain stationary and non-living. However, real-world scenarios frequently feature dynamic elements, such as moving objects, varying lighting conditions, and other temporal events, thereby presenting a significantly more challenging scenario. To address this gap, we propose a more realistic problem named HDR Dynamic Novel View Synthesis (HDR DNVS), where the additional dimension ``Dynamic'' emphasizes the necessity of jointly modeling temporal radiance variations alongside sophisticated 3D translation between LDR and HDR. To tackle this complex, intertwined challenge, we introduce HDR-4DGS, a Gaussian Splatting-based architecture featured with an innovative dynamic tone-mapping module that explicitly connects HDR and LDR domains, maintaining temporal radiance coherence by dynamically adapting tone-mapping functions according to the evolving radiance distributions across the temporal dimension. As a result, HDR-4DGS achieves both temporal radiance consistency and spatially accurate color translation, enabling photorealistic HDR renderings from arbitrary viewpoints and time instances. Extensive experiments demonstrate that HDR-4DGS surpasses existing state-of-the-art methods in both quantitative performance and visual fidelity. Source code is available at https://github.com/prinasi/HDR-4DGS.
>
---
#### [replaced 073] UDEEP: Edge-based Computer Vision for In-Situ Underwater Crayfish and Plastic Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2401.06157v2](https://arxiv.org/pdf/2401.06157v2)**

> **作者:** Dennis Monari; Farhad Fassihi Tash; Jordan J. Bird; Ahmad Lotfi; Isibor Kennedy Ihianle; Salisu Wada Yahaya; Isibor Kennedy Ihianle; Md Mahmudul Hasan; Pedro Sousa; Pedro Machado
>
> **摘要:** Invasive signal crayfish have a detrimental impact on ecosystems. They spread the fungal-type crayfish plague disease (Aphanomyces astaci) that is lethal to the native white clawed crayfish, the only native crayfish species in Britain. Invasive signal crayfish extensively burrow, causing habitat destruction, erosion of river banks and adverse changes in water quality, while also competing with native species for resources leading to declines in native populations. Moreover, pollution exacerbates the vulnerability of White-clawed crayfish, with their populations declining by over 90%. To safeguard aquatic ecosystems, it is imperative to address the challenges posed by invasive species and pollution in aquatic ecosystem's. This article introduces the Cognitive Edge Device (CED) computing platform for the detection of crayfish and plastic. It also presents two publicly available underwater datasets, annotated with sequences of crayfish and aquatic plastic debris. Four You Only Look Once (YOLO) variants were trained and evaluated for crayfish and plastic object detection. YOLOv5s achieved the highest detection accuracy, with an mAP@0.5 of 0.90, and achieved the best precision
>
---
#### [replaced 074] EgoLife: Towards Egocentric Life Assistant
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.03803v2](https://arxiv.org/pdf/2503.03803v2)**

> **作者:** Jingkang Yang; Shuai Liu; Hongming Guo; Yuhao Dong; Xiamengwei Zhang; Sicheng Zhang; Pengyun Wang; Zitang Zhou; Binzhu Xie; Ziyue Wang; Bei Ouyang; Zhengyu Lin; Marco Cominelli; Zhongang Cai; Yuanhan Zhang; Peiyuan Zhang; Fangzhou Hong; Joerg Widmer; Francesco Gringoli; Lei Yang; Bo Li; Ziwei Liu
>
> **备注:** Accepted to CVPR 2025. Project Page: https://egolife-ai.github.io/. Code: https://github.com/EvolvingLMMs-Lab/EgoLife
>
> **摘要:** We introduce EgoLife, a project to develop an egocentric life assistant that accompanies and enhances personal efficiency through AI-powered wearable glasses. To lay the foundation for this assistant, we conducted a comprehensive data collection study where six participants lived together for one week, continuously recording their daily activities - including discussions, shopping, cooking, socializing, and entertainment - using AI glasses for multimodal egocentric video capture, along with synchronized third-person-view video references. This effort resulted in the EgoLife Dataset, a comprehensive 300-hour egocentric, interpersonal, multiview, and multimodal daily life dataset with intensive annotation. Leveraging this dataset, we introduce EgoLifeQA, a suite of long-context, life-oriented question-answering tasks designed to provide meaningful assistance in daily life by addressing practical questions such as recalling past relevant events, monitoring health habits, and offering personalized recommendations. To address the key technical challenges of (1) developing robust visual-audio models for egocentric data, (2) enabling identity recognition, and (3) facilitating long-context question answering over extensive temporal information, we introduce EgoButler, an integrated system comprising EgoGPT and EgoRAG. EgoGPT is an omni-modal model trained on egocentric datasets, achieving state-of-the-art performance on egocentric video understanding. EgoRAG is a retrieval-based component that supports answering ultra-long-context questions. Our experimental studies verify their working mechanisms and reveal critical factors and bottlenecks, guiding future improvements. By releasing our datasets, models, and benchmarks, we aim to stimulate further research in egocentric AI assistants.
>
---
#### [replaced 075] BlindSight: Harnessing Sparsity for Efficient Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.09071v2](https://arxiv.org/pdf/2507.09071v2)**

> **作者:** Tharun Adithya Srikrishnan; Deval Shah; Timothy Hein; Ahmed Hasssan; Stephen Youn; Steven K. Reinhardt
>
> **摘要:** Large vision-language models (VLMs) enable joint processing of text and images. However, incorporating vision data significantly increases the prompt length, resulting in a longer time to first token (TTFT). This bottleneck can be alleviated by leveraging the inherent sparsity in the attention computation. Analyzing these attention patterns in VLMs when processing a series of images, we observe the absence of inter-image attention in a substantial portion of layers. Based on this, we propose BlindSight: an approach to optimize multi-image VLM inference using an input-template-aware attention sparsity mask with no runtime overhead. We utilize a dataset to derive a prompt-agnostic categorization for attention heads: Dense, Sink, Intra-Image, and Intra-Image+Sink. We develop a Triton-based GPU kernel to leverage this sparsity. BlindSight achieves a 1.8-3.2x speedup in the attention computation (prompt length 36K-300K). BlindSight generalizes across VLMs (Qwen2-VL, Qwen2.5-VL, Gemma 3), with only a 0.78% absolute accuracy degradation on average on multi-image comprehension benchmarks. Finally, we advocate for the design of efficient VLMs that combine BlindSight-inspired sparse and dense layers.
>
---
#### [replaced 076] SAM-Aug: Leveraging SAM Priors for Few-Shot Parcel Segmentation in Satellite Time Series
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09110v2](https://arxiv.org/pdf/2601.09110v2)**

> **作者:** Kai Hu; Yaozu Feng; Vladimir Lysenko; Ya Guo; Huayi Wu
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Few-shot semantic segmentation of time-series remote sensing images remains a critical challenge, particularly in regions where labeled data is scarce or costly to obtain. While state-of-the-art models perform well under full supervision, their performance degrades significantly under limited labeling, limiting their real-world applicability. In this work, we propose SAM-Aug, a new annotation-efficient framework that leverages the geometry-aware segmentation capability of the Segment Anything Model (SAM) to improve few-shot land cover mapping. Our approach constructs cloud-free composite images from temporal sequences and applies SAM in a fully unsupervised manner to generate geometry-aware mask priors. These priors are then integrated into training through a proposed loss function called RegionSmoothLoss, which enforces prediction consistency within each SAM-derived region across temporal frames, effectively regularizing the model to respect semantically coherent structures. Extensive experiments on the PASTIS-R benchmark under a 5 percent labeled setting demonstrate the effectiveness and robustness of SAM-Aug. Averaged over three random seeds (42, 2025, 4090), our method achieves a mean test mIoU of 36.21 percent, outperforming the state-of-the-art baseline by +2.33 percentage points, a relative improvement of 6.89 percent. Notably, on the most favorable split (seed=42), SAM-Aug reaches a test mIoU of 40.28 percent, representing an 11.2 percent relative gain with no additional labeled data. The consistent improvement across all seeds confirms the generalization power of leveraging foundation model priors under annotation scarcity. Our results highlight that vision models like SAM can serve as useful regularizers in few-shot remote sensing learning, offering a scalable and plug-and-play solution for land cover monitoring without requiring manual annotations or model fine-tuning.
>
---
#### [replaced 077] PromptVFX: Text-Driven Fields for Open-World 3D Gaussian Animation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01091v2](https://arxiv.org/pdf/2506.01091v2)**

> **作者:** Mert Kiray; Paul Uhlenbruck; Nassir Navab; Benjamin Busam
>
> **摘要:** Visual effects (VFX) are key to immersion in modern films, games, and AR/VR. Creating 3D effects requires specialized expertise and training in 3D animation software and can be time consuming. Generative solutions typically rely on computationally intense methods such as diffusion models which can be slow at 4D inference. We reformulate 3D animation as a field prediction task and introduce a text-driven framework that infers a time-varying 4D flow field acting on 3D Gaussians. By leveraging large language models (LLMs) and vision-language models (VLMs) for function generation, our approach interprets arbitrary prompts (e.g., "make the vase glow orange, then explode") and instantly updates color, opacity, and positions of 3D Gaussians in real time. This design avoids overheads such as mesh extraction, manual or physics-based simulations and allows both novice and expert users to animate volumetric scenes with minimal effort on a consumer device even in a web browser. Experimental results show that simple textual instructions suffice to generate compelling time-varying VFX, reducing the manual effort typically required for rigging or advanced modeling. We thus present a fast and accessible pathway to language-driven 3D content creation that can pave the way to democratize VFX further. Code available at https://obsphera.github.io/promptvfx/.
>
---
#### [replaced 078] All-in-One Video Restoration under Smoothly Evolving Unknown Weather Degradations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00533v2](https://arxiv.org/pdf/2601.00533v2)**

> **作者:** Wenrui Li; Hongtao Chen; Yao Xiao; Wangmeng Zuo; Jiantao Zhou; Yonghong Tian; Xiaopeng Fan
>
> **摘要:** All-in-one image restoration aims to recover clean images from diverse unknown degradations using a single model. But extending this task to videos faces unique challenges. Existing approaches primarily focus on frame-wise degradation variation, overlooking the temporal continuity that naturally exists in real-world degradation processes. In practice, degradation types and intensities evolve smoothly over time, and multiple degradations may coexist or transition gradually. In this paper, we introduce the Smoothly Evolving Unknown Degradations (SEUD) scenario, where both the active degradation set and degradation intensity change continuously over time. To support this scenario, we design a flexible synthesis pipeline that generates temporally coherent videos with single, compound, and evolving degradations. To address the challenges in the SEUD scenario, we propose an all-in-One Recurrent Conditional and Adaptive prompting Network (ORCANet). First, a Coarse Intensity Estimation Dehazing (CIED) module estimates haze intensity using physical priors and provides coarse dehazed features as initialization. Second, a Flow Prompt Generation (FPG) module extracts degradation features. FPG generates both static prompts that capture segment-level degradation types and dynamic prompts that adapt to frame-level intensity variations. Furthermore, a label-aware supervision mechanism improves the discriminability of static prompt representations under different degradations. Extensive experiments show that ORCANet achieves superior restoration quality, temporal consistency, and robustness over image and video-based baselines. Code is available at https://github.com/Friskknight/ORCANet-SEUD.
>
---
#### [replaced 079] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.12742v2](https://arxiv.org/pdf/2505.12742v2)**

> **作者:** Jinhua Zhang; Wei Long; Minghao Han; Weiyi You; Shuhang Gu
>
> **备注:** Accepted to ICLR 2026. Project page: https://nuanbaobao.github.io/MVAR
>
> **摘要:** Essential to visual generation is efficient modeling of visual data priors. Conventional next-token prediction methods define the process as learning the conditional probability distribution of successive tokens. Recently, next-scale prediction methods redefine the process to learn the distribution over multi-scale representations, significantly reducing generation latency. However, these methods condition each scale on all previous scales and require each token to consider all preceding tokens, exhibiting scale and spatial redundancy. To better model the distribution by mitigating redundancy, we propose Markovian Visual AutoRegressive modeling (MVAR), a novel autoregressive framework that introduces scale and spatial Markov assumptions to reduce the complexity of conditional probability modeling. Specifically, we introduce a scale-Markov trajectory that only takes as input the features of adjacent preceding scale for next-scale prediction, enabling the adoption of a parallel training strategy that significantly reduces GPU memory consumption. Furthermore, we propose spatial-Markov attention, which restricts the attention of each token to a localized neighborhood of size k at corresponding positions on adjacent scales, rather than attending to every token across these scales, for the pursuit of reduced modeling complexity. Building on these improvements, we reduce the computational complexity of attention calculation from O(N^2) to O(Nk), enabling training with just eight NVIDIA RTX 4090 GPUs and eliminating the need for KV cache during inference. Extensive experiments on ImageNet demonstrate that MVAR achieves comparable or superior performance with both small model trained from scratch and large fine-tuned models, while reducing the average GPU memory footprint by 3.0x.
>
---
#### [replaced 080] Neural Cellular Automata: From Cells to Pixels
- **分类: cs.CV; cs.GR; cs.LG; cs.MA; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.22899v2](https://arxiv.org/pdf/2506.22899v2)**

> **作者:** Ehsan Pajouheshgar; Yitao Xu; Ali Abbasi; Alexander Mordvintsev; Wenzel Jakob; Sabine Süsstrunk
>
> **备注:** 9 pages, 14 figures, +7 pages of Appendix
>
> **摘要:** Neural Cellular Automata (NCAs) are bio-inspired dynamical systems in which identical cells iteratively apply a learned local update rule to self-organize into complex patterns, exhibiting regeneration, robustness, and spontaneous dynamics. Despite their success in texture synthesis and morphogenesis, NCAs remain largely confined to low-resolution outputs. This limitation stems from (1) training time and memory requirements that grow quadratically with grid size, (2) the strictly local propagation of information that impedes long-range cell communication, and (3) the heavy compute demands of real-time inference at high resolution. In this work, we overcome this limitation by pairing an NCA that evolves on a coarse grid with a lightweight implicit decoder that maps cell states and local coordinates to appearance attributes, enabling the same model to render outputs at arbitrary resolution. Moreover, because both the decoder and NCA updates are local, inference remains highly parallelizable. To supervise high-resolution outputs efficiently, we introduce task-specific losses for morphogenesis (growth from a seed) and texture synthesis with minimal additional memory and computation overhead. Our experiments across 2D/3D grids and mesh domains demonstrate that our hybrid models produce high-resolution outputs in real-time, and preserve the characteristic self-organizing behavior of NCAs.
>
---
#### [replaced 081] GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18543v2](https://arxiv.org/pdf/2601.18543v2)**

> **作者:** Kaixun Jiang; Yuzheng Wang; Junjie Zhou; Pandeng Li; Zhihang Liu; Chen-Wei Xie; Zhaoyu Chen; Yun Zheng; Wenqiang Zhang
>
> **摘要:** We introduce GenAgent, unifying visual understanding and generation through an agentic multimodal model. Unlike unified models that face expensive training costs and understanding-generation trade-offs, GenAgent decouples these capabilities through an agentic framework: understanding is handled by the multimodal model itself, while generation is achieved by treating image generation models as invokable tools. Crucially, unlike existing modular systems constrained by static pipelines, this design enables autonomous multi-turn interactions where the agent generates multimodal chains-of-thought encompassing reasoning, tool invocation, judgment, and reflection to iteratively refine outputs. We employ a two-stage training strategy: first, cold-start with supervised fine-tuning on high-quality tool invocation and reflection data to bootstrap agent behaviors; second, end-to-end agentic reinforcement learning combining pointwise rewards (final image quality) and pairwise rewards (reflection accuracy), with trajectory resampling for enhanced multi-turn exploration. GenAgent significantly boosts base generator(FLUX.1-dev) performance on GenEval++ (+23.6\%) and WISE (+14\%). Beyond performance gains, our framework demonstrates three key properties: 1) cross-tool generalization to generators with varying capabilities, 2) test-time scaling with consistent improvements across interaction rounds, and 3) task-adaptive reasoning that automatically adjusts to different tasks. Our code will be available at \href{https://github.com/deep-kaixun/GenAgent}{this url}.
>
---
#### [replaced 082] X-LRM: X-ray Large Reconstruction Model for Extremely Sparse-View Computed Tomography Recovery in One Second
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06382v2](https://arxiv.org/pdf/2503.06382v2)**

> **作者:** Guofeng Zhang; Ruyi Zha; Hao He; Yixun Liang; Alan Yuille; Hongdong Li; Yuanhao Cai
>
> **备注:** 3DV 2026; A large reconstruction model and the largest dataset (16K samples) for sparse-view CT recovery
>
> **摘要:** Sparse-view 3D CT reconstruction aims to recover volumetric structures from a limited number of 2D X-ray projections. Existing feedforward methods are constrained by the scarcity of large-scale training datasets and the absence of direct and consistent 3D representations. In this paper, we propose an X-ray Large Reconstruction Model (X-LRM) for extremely sparse-view ($<$10 views) CT reconstruction. X-LRM consists of two key components: X-former and X-triplane. X-former can handle an arbitrary number of input views using an MLP-based image tokenizer and a Transformer-based encoder. The output tokens are then upsampled into our X-triplane representation, which models the 3D radiodensity as an implicit neural field. To support the training of X-LRM, we introduce Torso-16K, a large-scale dataset comprising over 16K volume-projection pairs of various torso organs. Extensive experiments demonstrate that X-LRM outperforms the state-of-the-art method by 1.5 dB and achieves 27$\times$ faster speed with better flexibility. Furthermore, the evaluation of lung segmentation tasks also suggests the practical value of our approach. Our code and dataset will be released at https://github.com/Richard-Guofeng-Zhang/X-LRM
>
---
#### [replaced 083] Random forest-based out-of-distribution detection for robust lung cancer segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.19112v4](https://arxiv.org/pdf/2508.19112v4)**

> **作者:** Aneesh Rangnekar; Harini Veeraraghavan
>
> **备注:** Accepted at SPIE Medical Imaging 2026
>
> **摘要:** Accurate detection and segmentation of cancerous lesions from computed tomography (CT) scans is essential for automated treatment planning and cancer treatment response assessment. Transformer-based models with self-supervised pretraining can produce reliably accurate segmentation from in-distribution (ID) data but degrade when applied to out-of-distribution (OOD) datasets. We address this challenge with RF-Deep, a random forest classifier that utilizes deep features from a pretrained transformer encoder of the segmentation model to detect OOD scans and enhance segmentation reliability. The segmentation model comprises a Swin Transformer encoder, pretrained with masked image modeling (SimMIM) on 10,432 unlabeled 3D CT scans covering cancerous and non-cancerous conditions, with a convolution decoder, trained to segment lung cancers in 317 3D scans. Independent testing was performed on 603 3D CT public datasets that included one ID dataset and four OOD datasets comprising chest CTs with pulmonary embolism (PE) and COVID-19, and abdominal CTs with kidney cancers and healthy volunteers. RF-Deep detected OOD cases with a FPR95 of 18.26%, 27.66%, and less than 0.1% on PE, COVID-19, and abdominal CTs, consistently outperforming established OOD approaches. The RF-Deep classifier provides a simple and effective approach to enhance reliability of cancer segmentation in ID and OOD scenarios.
>
---
#### [replaced 084] Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19506v2](https://arxiv.org/pdf/2601.19506v2)**

> **作者:** Zhengjian Yao; Jiakui Hu; Kaiwen Li; Hangzhou He; Xinliang Zhang; Shuang Zeng; Lei Zhu; Yanye Lu
>
> **摘要:** Blind face restoration remains a persistent challenge due to the inherent ill-posedness of reconstructing holistic structures from severely constrained observations. Current generative approaches, while capable of synthesizing realistic textures, often suffer from information asymmetry -- the intrinsic disparity between the information-sparse low quality inputs and the information-dense high quality outputs. This imbalance leads to a one-to-many mapping, where insufficient constraints result in stochastic uncertainty and hallucinatory artifacts. To bridge this gap, we present \textbf{Pref-Restore}, a hierarchical framework that integrates discrete semantic logic with continuous texture generation to achieve deterministic, preference-aligned restoration. Our methodology fundamentally addresses this information disparity through two complementary strategies: (1) Augmenting Input Density: We employ an auto-regressive integrator to reformulate textual instructions into dense latent queries, injecting high-level semantic stability to constrain the degraded signals; (2) Pruning Output Distribution: We pioneer the integration of on-policy reinforcement learning directly into the diffusion restoration loop. By transforming human preferences into differentiable constraints, we explicitly penalize stochastic deviations, thereby sharpening the posterior distribution toward the desired high-fidelity outcomes. Extensive experiments demonstrate that Pref-Restore achieves state-of-the-art performance across synthetic and real-world benchmarks. Furthermore, empirical analysis confirms that our preference-aligned strategy significantly reduces solution entropy, establishing a robust pathway toward reliable and deterministic blind restoration.
>
---
#### [replaced 085] Semantic Depth Matters: Explaining Errors of Deep Vision Networks through Perceived Class Similarities
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.09956v2](https://arxiv.org/pdf/2504.09956v2)**

> **作者:** Katarzyna Filus; Michał Romaszewski; Mateusz Żarski
>
> **摘要:** Understanding deep neural network (DNN) behavior requires more than evaluating classification accuracy alone; analyzing errors and their predictability is equally crucial. Current evaluation methodologies lack transparency, particularly in explaining the underlying causes of network misclassifications. To address this, we introduce a novel framework that investigates the relationship between the semantic hierarchy depth perceived by a network and its real-data misclassification patterns. Central to our framework is the Similarity Depth (SD) metric, which quantifies the semantic hierarchy depth perceived by a network along with a method of evaluation of how closely the network's errors align with its internally perceived similarity structure. We also propose a graph-based visualization of model semantic relationships and misperceptions. A key advantage of our approach is that leveraging class templates -- representations derived from classifier layer weights -- is applicable to already trained networks without requiring additional data or experiments. Our approach reveals that deep vision networks encode specific semantic hierarchies and that high semantic depth improves the compliance between perceived class similarities and actual errors.
>
---
