# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Joint Learning Global-Local Speaker Classification to Enhance End-to-End Speaker Diarization and Recognition
- **分类: cs.SD**

- **简介: 该论文属于说话人二值化与识别任务，解决LALMs在说话人区分上的不足。提出GLSC-SDR框架，结合全局与局部分类策略，提升区分能力。**

- **链接: [https://arxiv.org/pdf/2603.25377](https://arxiv.org/pdf/2603.25377)**

> **作者:** Yuhang Dai; Haopeng Lin; Jiale Qian; Ruiqi Yan; Hao Meng; Hanke Xie; Hanlin Wen; Shunshun Yin; Ming Tao; Xie Chen; Lei Xie; Xinsheng Wang
>
> **备注:** 5 pages, 2 figures, 2 tables
>
> **摘要:** Large Audio-Language Models (LALMs) have demonstrated remarkable performance in end-to-end speaker diarization and recognition. However, their speaker discriminability remains limited due to the scarcity of large-scale conversational data and the absence of explicit speaker representation optimization. To address this, we propose GLSC-SDR, a paradigm that jointly trains speaker classification with diarization and recognition. We further introduce a Global-Local Speaker Classification strategy, which uses clustered speakers as global labels and re-encoded intra-cluster speakers as local labels. This hierarchical design enhances fine-grained speaker discrimination while preserving semantic transcription accuracy. Experiments on AliMeeting, AISHELL-4, and AMI-SDM demonstrate that GLSC-SDR achieves competitive or superior performance compared to simulation-based and multi-encoder approaches, without relying on large-scale real conversational data.
>
---
#### [new 002] CoDeTT: A Context-Aware Decision Benchmark for Turn-Taking Evaluation
- **分类: cs.SD**

- **简介: 该论文属于对话系统中的话轮转换任务，旨在解决现有评估方法碎片化的问题。提出CoDeTT基准，通过结构化决策问题和多场景数据集进行系统评估。**

- **链接: [https://arxiv.org/pdf/2603.25434](https://arxiv.org/pdf/2603.25434)**

> **作者:** Huan Shen; Yingao Wang; Shangkun Huang; Wei Zou; Yunzhang Chen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Turn-taking modeling is fundamental to spoken dialogue systems, yet its evaluation remains fragmented and often limited to binary boundary detection under narrow interaction settings. Such protocols hinder systematic comparison and obscure model weaknesses across conversational conditions. We present CoDeTT, a context-aware decision benchmark for turn-taking evaluation. CoDeTT formulates turn-taking as a structured decision problem and constructs a multi-scenario dataset with fine-grained decision categories and controlled context variations. Under a unified evaluation protocol, we assess representative existing models and observe substantial performance disparities across decision types and interaction scenarios. CoDeTT provides a standardized benchmark for systematic and context-aware evaluation of turn-taking systems. The benchmark dataset and evaluation toolkit are available at this https URL.
>
---
#### [new 003] Unified Diffusion Refinement for Multi-Channel Speech Enhancement and Separation
- **分类: eess.AS**

- **简介: 该论文提出Uni-ArrayDPS，用于多通道语音增强与分离任务。针对现有方法生成不自然语音的问题，利用扩散先验进行优化，无需训练即可提升效果。**

- **链接: [https://arxiv.org/pdf/2603.24810](https://arxiv.org/pdf/2603.24810)**

> **作者:** Zhongweiyang Xu; Ashutosh Pandey; Juan Azcarreta; Zhaoheng Ni; Sanjeel Parekh; Buye Xu; Romit Roy Choudhury
>
> **备注:** Paper in submission
>
> **摘要:** We propose Uni-ArrayDPS, a novel diffusion-based refinement framework for unified multi-channel speech enhancement and separation. Existing methods for multi-channel speech enhancement/separation are mostly discriminative and are highly effective at producing high-SNR outputs. However, they can still generate unnatural speech with non-linear distortions caused by the neural network and regression-based objectives. To address this issue, we propose Uni-ArrayDPS, which refines the outputs of any strong discriminative model using a speech diffusion prior. Uni-ArrayDPS is generative, array-agnostic, and training-free, and supports both enhancement and separation. Given a discriminative model's enhanced/separated speech, we use it, together with the noisy mixtures, to estimate the noise spatial covariance matrix (SCM). We then use this SCM to compute the likelihood required for diffusion posterior sampling of the clean speech source(s). Uni-ArrayDPS requires only a pre-trained clean-speech diffusion model as a prior and does not require additional training or fine-tuning, allowing it to generalize directly across tasks (enhancement/separation), microphone array geometries, and discriminative model backbones. Extensive experiments show that Uni-ArrayDPS consistently improves a wide range of discriminative models for both enhancement and separation tasks. We also report strong results on a real-world dataset. Audio demos are provided at \href{this https URL}{this https URL}.
>
---
#### [new 004] CLAR: CIF-Localized Alignment for Retrieval-Augmented Speech LLM-Based Contextual ASR
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，解决Speech LLM在命名实体和长尾词上的识别问题。提出CLAR模型，通过双编码器和CIF技术实现精准热词定位，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.25460](https://arxiv.org/pdf/2603.25460)**

> **作者:** Shangkun Huang; Huan Shen; Wei Zou; Yunzhang Chen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech LLM-based ASR often struggles with named entities and long-tail words due to strong internal language-model priors. Retrieval-augmented biasing can help, but its effectiveness depends on accurate hotword localization in full-utterance speech under weak supervision. We propose CLAR, a dual-encoder speech-text retriever that uses Continuous Integrate-and-Fire (CIF) to learn monotonic token-level alignments without timestamps. With length-aware localized matching, CLAR anchors short-entity acoustic cues and reduces representation dilution and attention drift. The retriever is trained with a multi-granularity objective combining global and local segment-level contrastive losses and a CIF quantity constraint. At inference, top-ranked hotwords are injected as contextual prompts for the Speech LLM, improving recognition without shallow fusion. Experiments show that CLAR significantly improves hotword retrieval and reduces both CER and B-WER against strong contextual ASR baselines.
>
---
#### [new 005] AdaLTM: Adaptive Layer-wise Task Vector Merging for Categorical Speech Emotion Recognition with ASR Knowledge Integration
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，解决ASR与SER融合中的性能瓶颈和优化冲突问题。通过引入自适应层间任务向量合并方法，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.25041](https://arxiv.org/pdf/2603.25041)**

> **作者:** Chia-Yu Lee; Huang-Cheng Chou; Tzu-Quan Lin; Yuanchao Li; Ya-Tse Wu; Shrikanth Narayanan; Chi-Chun Lee
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Integrating Automatic Speech Recognition (ASR) into Speech Emotion Recognition (SER) enhances modeling by providing linguistic context. However, conventional feature fusion faces performance bottlenecks, and multi-task learning often suffers from optimization conflicts. While task vectors and model merging have addressed such conflicts in NLP and CV, their potential in speech tasks remains largely unexplored. In this work, we propose an Adaptive Layer-wise Task Vector Merging (AdaLTM) framework based on WavLM-Large. Instead of joint optimization, we extract task vectors from in-domain ASR and SER models fine-tuned on emotion datasets. These vectors are integrated into a frozen base model using layer-wise learnable coefficients. This strategy enables depth-aware balancing of linguistic and paralinguistic knowledge across transformer layers without gradient interference. Experiments on the MSP-Podcast demonstrate that the proposed approach effectively mitigates conflicts between ASR and SER.
>
---
#### [new 006] X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音大模型任务，旨在解决语音LLM性能低于文本模型的问题。通过X-OPD框架，利用文本教师模型指导语音学生模型，提升其能力对齐。**

- **链接: [https://arxiv.org/pdf/2603.24596](https://arxiv.org/pdf/2603.24596)**

> **作者:** Di Cao; Dongjie Fu; Hai Yu; Siqi Zheng; Xu Tan; Tao Jin
>
> **备注:** 5 pages
>
> **摘要:** While the shift from cascaded dialogue systems to end-to-end (E2E) speech Large Language Models (LLMs) improves latency and paralinguistic modeling, E2E models often exhibit a significant performance degradation compared to their text-based counterparts. The standard Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training methods fail to close this gap. To address this, we propose X-OPD, a novel Cross-Modal On-Policy Distillation framework designed to systematically align the capabilities of Speech LLMs to their text-based counterparts. X-OPD enables the Speech LLM to explore its own distribution via on-policy rollouts, where a text-based teacher model evaluates these trajectories and provides token-level feedback, effectively distilling teacher's capabilities into student's multi-modal representations. Extensive experiments across multiple benchmarks demonstrate that X-OPD significantly narrows the gap in complex tasks while preserving the model's inherent capabilities.
>
---
#### [new 007] When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文研究自动抑郁检测任务，指出半结构化访谈中面试者提示导致模型偏差。工作包括分析数据集，发现模型依赖固定提示而非患者语言，提出限制使用患者话语以提高可靠性。**

- **链接: [https://arxiv.org/pdf/2603.24651](https://arxiv.org/pdf/2603.24651)**

> **作者:** Hasindri Watawana; Sergio Burdisso; Diego A. Moreno-Galván; Fernando Sánchez-Vega; A. Pastor López-Monroy; Petr Motlicek; Esaú Villatoro-Tello
>
> **备注:** Accepted to LREC 2026 Conference
>
> **摘要:** Automatic depression detection from doctor-patient conversations has gained momentum thanks to the availability of public corpora and advances in language modeling. However, interpretability remains limited: strong performance is often reported without revealing what drives predictions. We analyze three datasets: ANDROIDS, DAIC-WOZ, E-DAIC and identify a systematic bias from interviewer prompts in semi-structured interviews. Models trained on interviewer turns exploit fixed prompts and positions to distinguish depressed from control subjects, often achieving high classification scores without using participant language. Restricting models to participant utterances distributes decision evidence more broadly and reflects genuine linguistic cues. While semi-structured protocols ensure consistency, including interviewer prompts inflates performance by leveraging script artifacts. Our results highlight a cross-dataset, architecture-agnostic bias and emphasize the need for analyses that localize decision evidence by time and speaker to ensure models learn from participants' language.
>
---
#### [new 008] AVControl: Efficient Framework for Training Audio-Visual Controls
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出AVControl框架，解决多模态音频-视频生成控制问题。通过LoRA方法实现高效、可扩展的模型训练，支持多种控制方式。**

- **链接: [https://arxiv.org/pdf/2603.24793](https://arxiv.org/pdf/2603.24793)**

> **作者:** Matan Ben-Yosef; Tavi Halperin; Naomi Ken Korem; Mohammad Salama; Harel Cain; Asaf Joseph; Anthony Chen; Urska Jelercic; Ofir Bibi
>
> **备注:** Project page: this https URL
>
> **摘要:** Controlling video and audio generation requires diverse modalities, from depth and pose to camera trajectories and audio transformations, yet existing approaches either train a single monolithic model for a fixed set of controls or introduce costly architectural changes for each new modality. We introduce AVControl, a lightweight, extendable framework built on LTX-2, a joint audio-visual foundation model, where each control modality is trained as a separate LoRA on a parallel canvas that provides the reference signal as additional tokens in the attention layers, requiring no architectural changes beyond the LoRA adapters themselves. We show that simply extending image-based in-context methods to video fails for structural control, and that our parallel canvas approach resolves this. On the VACE Benchmark, we outperform all evaluated baselines on depth- and pose-guided generation, inpainting, and outpainting, and show competitive results on camera control and audio-visual benchmarks. Our framework supports a diverse set of independently trained modalities: spatially-aligned controls such as depth, pose, and edges, camera trajectory with intrinsics, sparse motion control, video editing, and, to our knowledge, the first modular audio-visual controls for a joint generation model. Our method is both compute- and data-efficient: each modality requires only a small dataset and converges within a few hundred to a few thousand training steps, a fraction of the budget of monolithic alternatives. We publicly release our code and trained LoRA checkpoints.
>
---
#### [new 009] SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于多模态深度伪造检测任务，旨在解决合成数据依赖导致的泛化能力不足问题。通过自监督学习，利用视觉伪篡改和音画不同步特征提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.25140](https://arxiv.org/pdf/2603.25140)**

> **作者:** Sahibzada Adil Shahzad; Ammarah Hashmi; Junichi Yamagishi; Yusuke Yasuda; Yu Tsao; Chia-Wen Lin; Yan-Tsung Peng; Hsin-Min Wang
>
> **摘要:** Multimodal deepfakes can exhibit subtle visual artifacts and cross-modal inconsistencies, which remain challenging to detect, especially when detectors are trained primarily on curated synthetic forgeries. Such synthetic dependence can introduce dataset and generator bias, limiting scalability and robustness to unseen manipulations. We propose SAVe, a self-supervised audio-visual deepfake detection framework that learns entirely on authentic videos. SAVe generates on-the-fly, identity-preserving, region-aware self-blended pseudo-manipulations to emulate tampering artifacts, enabling the model to learn complementary visual cues across multiple facial granularities. To capture cross-modal evidence, SAVe also models lip-speech synchronization via an audio-visual alignment component that detects temporal misalignment patterns characteristic of audio-visual forgeries. Experiments on FakeAVCeleb and AV-LipSync-TIMIT demonstrate competitive in-domain performance and strong cross-dataset generalization, highlighting self-supervised learning as a scalable paradigm for multimodal deepfake detection.
>
---
#### [new 010] Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于跨语言语音识别任务，旨在解决帕金森病患者构音障碍检测中数据不足的问题。通过语言迁移方法调整语音表示，提升跨语言检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22225](https://arxiv.org/pdf/2603.22225)**

> **作者:** Abner Hernandez; Eunjung Yeo; Kwanghee Choi; Chin-Jou Li; Zhengjun Yue; Rohan Kumar Das; Jan Rusz; Mathew Magimai Doss; Juan Rafael Orozco-Arroyave; Tomás Arias-Vergara; Andreas Maier; Elmar Nöth; David R. Mortensen; David Harwath; Paula Andrea Perez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure.
>
---
## 更新

#### [replaced 001] ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于语音安全检测任务，旨在解决语音欺骗、深度伪造和对抗攻击的检测问题。通过分析53个团队的提交结果，评估不同技术在对抗环境下的表现。**

- **链接: [https://arxiv.org/pdf/2601.03944](https://arxiv.org/pdf/2601.03944)**

> **作者:** Xin Wang; Héctor Delgado; Nicholas Evans; Xuechen Liu; Tomi Kinnunen; Hemlata Tak; Kong Aik Lee; Ivan Kukanov; Md Sahidullah; Massimiliano Todisco; Junichi Yamagishi
>
> **备注:** This work has been submitted to the IEEE TASLP for possible publication
>
> **摘要:** ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake detection solutions. A significant change from previous challenge editions is a new crowdsourced database collected from a substantially greater number of speakers under diverse recording conditions, and a mix of cutting-edge and legacy generative speech technology. With the new database described elsewhere, we provide in this paper an overview of the ASVspoof 5 challenge results for the submissions of 53 participating teams. While many solutions perform well, performance degrades under adversarial attacks and the application of neural encoding/compression schemes. Together with a review of post-challenge results, we also report a study of calibration in addition to other principal challenges and outline a road-map for the future of ASVspoof.
>
---
#### [replaced 002] MiDashengLM: Efficient Audio Understanding with General Audio Captions
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出MiDashengLM，一个用于高效音频理解的开放音频-语言模型，解决传统模型依赖封闭数据的问题。通过通用音频描述训练，融合语音、声音和音乐信息，提升音频场景的全面理解。**

- **链接: [https://arxiv.org/pdf/2508.03983](https://arxiv.org/pdf/2508.03983)**

> **作者:** Heinrich Dinkel; Gang Li; Jizhong Liu; Jian Luan; Yadong Niu; Xingwei Sun; Tianzi Wang; Qiyang Xiao; Junbo Zhang; Jiahao Zhou
>
> **备注:** Added ACAVCaps reference (ICASSP 2026)
>
> **摘要:** Current approaches for large audio language models (LALMs) often rely on closed data sources or proprietary models, limiting their generalization and accessibility. This paper introduces MiDashengLM, a novel open audio-language model designed for efficient and comprehensive audio understanding through the use of general audio captions using our novel ACAVCaps training dataset. MiDashengLM exclusively relies on publicly available pretraining and supervised fine-tuning (SFT) datasets, ensuring full transparency and reproducibility. At its core, MiDashengLM integrates Dasheng, an open-source audio encoder, specifically engineered to process diverse auditory information effectively. Unlike previous works primarily focused on Automatic Speech Recognition (ASR) based audio-text alignment, our strategy centers on general audio captions, fusing speech, sound and music information into one textual representation, enabling a holistic textual representation of complex audio scenes. Lastly, MiDashengLM provides an up to 4x speedup in terms of time-to-first-token (TTFT) and up to 20x higher throughput than comparable models. Checkpoints are available online at this https URL and this https URL.
>
---
#### [replaced 003] U-DREAM: Unsupervised Dereverberation guided by a Reverberation Model
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于语音去混响任务，解决缺乏配对数据的问题。通过无监督学习，利用混响模型引导深度网络，提升去混响效果。**

- **链接: [https://arxiv.org/pdf/2507.14237](https://arxiv.org/pdf/2507.14237)**

> **作者:** Louis Bahrman; Marius Rodrigues; Mathieu Fontaine; Gaël Richard
>
> **摘要:** This paper explores the outcome of training state-of-the-art dereverberation models with supervision settings ranging from weakly-supervised to virtually unsupervised, relying solely on reverberant signals and an acoustic model for training. Most of the existing deep learning approaches typically require paired dry and reverberant data, which are difficult to obtain in practice. We develop instead a sequential learning strategy motivated by a maximum-likelihood formulation of the dereverberation problem, wherein acoustic parameters and dry signals are estimated from reverberant inputs using deep neural networks, guided by a reverberation matching loss. Our most data-efficient variant requires only 100 reverberation-parameter-labeled samples to outperform an unsupervised baseline, demonstrating the effectiveness and practicality of the proposed method in low-resource scenarios.
>
---
#### [replaced 004] A Lightweight Two-Branch Architecture for Multi-instrument Transcription via Note-Level Contrastive Clustering
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于多乐器音高转录任务，解决模型泛化能力差、计算量大等问题。提出轻量双分支架构，通过音符级对比聚类实现动态分离与联合转录。**

- **链接: [https://arxiv.org/pdf/2509.12712](https://arxiv.org/pdf/2509.12712)**

> **作者:** Ruigang Li; Yongxu Zhu
>
> **摘要:** Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments, rigid source-count constraints, and high computational demands that hinder deployment on low-resource devices. We address these limitations with a lightweight model that extends a timbre-agnostic transcription backbone with a dedicated timbre encoder and performs deep clustering at the note level, enabling joint transcription and dynamic separation of arbitrary instruments given a specified number of instrument classes. Practical optimizations including spectral normalization, dilated convolutions, and contrastive clustering further improve efficiency and robustness. Despite its small size and fast inference, the model achieves competitive performance with heavier baselines in terms of transcription accuracy and separation quality, and shows promising generalization ability, making it highly suitable for real-world deployment in practical and resource-constrained settings.
>
---
#### [replaced 005] DashengTokenizer: One layer is enough for unified audio understanding and generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DashengTokenizer，用于统一的音频理解和生成任务。解决传统方法依赖固定语义特征的问题，通过注入声学信息提升性能。实验表明其在多项任务中表现优异，且无需VAE架构即可实现音频合成。**

- **链接: [https://arxiv.org/pdf/2602.23765](https://arxiv.org/pdf/2602.23765)**

> **作者:** Heinrich Dinkel; Xingwei Sun; Gang Li; Jiahao Mei; Yadong Niu; Jizhong Liu; Xiyang Li; Yifan Liao; Jiahao Zhou; Junbo Zhang; Jian Luan
>
> **备注:** Added ACAVCaps reference
>
> **摘要:** This paper introduces DashengTokenizer, a continuous audio tokenizer engineered for joint use in both understanding and generation tasks. Unlike conventional approaches, which train acoustic tokenizers and subsequently integrate frozen semantic knowledge, our method inverts this paradigm: we leverage frozen semantic features and inject acoustic information. In linear evaluation across 22 diverse tasks, our method outperforms previous audio codec and audio encoder baselines by a significant margin while maintaining competitive audio reconstruction quality. Notably, we demonstrate that this acoustic injection improves performance for tasks such as speech emotion recognition, music understanding, and acoustic scene classification. We further evaluate the tokenizer's generative performance on text-to-audio (TTA), text-to-music (TTM), and speech enhancement (SE). Our approach surpasses standard variational autoencoder (VAE)-based methods on TTA and TTM tasks, while its effectiveness on SE underscores its capabilities as a general-purpose audio encoder. Finally, our results challenge the prevailing assumption that VAE-based architectures are a prerequisite for audio synthesis. Checkpoints are available at this https URL.
>
---
#### [replaced 006] Acoustic Imaging for Low-SNR UAV Detection: Dense Beamformed Energy Maps and U-Net SELD
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于声源定位任务，解决低信噪比下无人机检测问题。通过U-Net模型对波束成形能量图进行语义分割，实现高精度方向估计。**

- **链接: [https://arxiv.org/pdf/2508.00307](https://arxiv.org/pdf/2508.00307)**

> **作者:** Belman Jahir Rodriguez; Sergio F. Chevtchenko; Marcelo Herrera Martinez; Yeshwant Bethy; Saeed Afshar
>
> **摘要:** We introduce a U-net model for 360° acoustic source localization formulated as a spherical semantic segmentation task. Rather than regressing discrete direction-of-arrival (DoA) angles, our model segments beamformed audio maps (azimuth and elevation) into regions of active sound presence. Using delay-and-sum (DAS) beamforming on a custom 24-microphone array, we generate signals aligned with drone GPS telemetry to create binary supervision masks. A modified U-Net, trained on frequency-domain representations of these maps, learns to identify spatially distributed source regions while addressing class imbalance via the Tversky loss. Because the network operates on beamformed energy maps, the approach is inherently array-independent and can adapt to different microphone configurations without retraining from scratch. The segmentation outputs are post-processed by computing centroids over activated regions, enabling robust DoA estimates. Our dataset includes real-world open-field recordings of a DJI Air 3 drone, synchronized with 360° video and flight logs across multiple dates and locations. Experimental results show that U-net generalizes across environments, providing improved angular precision, offering a new paradigm for dense spatial audio understanding beyond traditional Sound Source Localization (SSL).
>
---
#### [replaced 007] Enhancing Efficiency and Performance in Deepfake Audio Detection through Neuron-level Dropin & Neuroplasticity Mechanisms
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于深度伪造音频检测任务，旨在提升检测效率与性能。针对现有模型参数扩展受限的问题，提出dropin和神经可塑性机制，动态调整神经元数量，实现更高效的模型优化。**

- **链接: [https://arxiv.org/pdf/2603.24343](https://arxiv.org/pdf/2603.24343)**

> **作者:** Yupei Li; Shuaijie Shao; Manuel Milling; Björn Schuller
>
> **备注:** Accepted at IJCNN 2026
>
> **摘要:** Current audio deepfake detection has achieved remarkable performance using diverse deep learning architectures such as ResNet, and has seen further improvements with the introduction of large models (LMs) like Wav2Vec. The success of large language models (LLMs) further demonstrates the benefits of scaling model parameters, but also highlights one bottleneck where performance gains are constrained by parameter counts. Simply stacking additional layers, as done in current LLMs, is computationally expensive and requires full retraining. Furthermore, existing low-rank adaptation methods are primarily applied to attention-based architectures, which limits their scope. Inspired by the neuronal plasticity observed in mammalian brains, we propose novel algorithms, dropin and further plasticity, that dynamically adjust the number of neurons in certain layers to flexibly modulate model parameters. We evaluate these algorithms on multiple architectures, including ResNet, Gated Recurrent Neural Networks, and Wav2Vec. Experimental results using the widely recognised ASVSpoof2019 LA, PA, and FakeorReal dataset demonstrate consistent improvements in computational efficiency with the dropin approach and a maximum of around 39% and 66% relative reduction in Equal Error Rate with the dropin and plasticity approach among these dataset, respectively. The code and supplementary material are available at Github link.
>
---
#### [replaced 008] Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation
- **分类: cs.SD; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于自动和弦识别任务，解决标注数据稀缺问题。通过伪标签和知识蒸馏方法，提升模型性能，显著改善罕见和弦识别效果。**

- **链接: [https://arxiv.org/pdf/2602.19778](https://arxiv.org/pdf/2602.19778)**

> **作者:** Nghia Phan; Rong Jin; Gang Liu; Xiao Dong
>
> **备注:** 9 pages, 6 figures, 3 tables
>
> **摘要:** Automatic Chord Recognition (ACR) is constrained by the scarcity of aligned chord labels, as well-aligned annotations are costly to acquire. At the same time, open-weight pre-trained models are currently more accessible than their proprietary training data. In this work, we present a two-stage training pipeline that leverages pre-trained models together with unlabeled audio. The proposed method decouples training into two stages. In the first stage, we use a pre-trained BTC model as a teacher to generate pseudo-labels for over 1,000 hours of diverse unlabeled audio and train a student model solely on these pseudo-labels. In the second stage, the student is continually trained on ground-truth labels as they become available. To prevent catastrophic forgetting of the representations learned in the first stage, we apply selective knowledge distillation (KD) from the teacher as a regularizer. In our experiments, two models (BTC, 2E1D) were used as students. In stage 1, using only pseudo-labels, the BTC student achieves over 98% of the teacher's performance, while the 2E1D model achieves about 96% across seven standard mir_eval metrics. After a single training run for both students in stage 2, the resulting BTC student model surpasses the traditional supervised learning baseline by 2.5% and the original pre-trained teacher model by 1.55% on average across all metrics. The resulting 2E1D student model improves over the traditional supervised learning baseline by 2.67% on average and achieves almost the same performance as the teacher. Both cases show large gains on rare chord qualities.
>
---
