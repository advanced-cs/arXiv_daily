# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Unified Multi-task Learning for Voice-Based Detection of Diverse Clinical Conditions
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出MARVEL框架，通过统一多任务学习从语音中检测多种临床条件（神经/呼吸/发声障碍），解决传统方法单一任务、无法利用语音多维信息的问题，采用双分支结构共享声学骨干，使用衍生特征提升多任务检测性能。**

- **链接: [http://arxiv.org/pdf/2508.20717v1](http://arxiv.org/pdf/2508.20717v1)**

> **作者:** Ran Piao; Yuan Lu; Hareld Kemps; Tong Xia; Aaqib Saeed
>
> **摘要:** Voice-based health assessment offers unprecedented opportunities for scalable, non-invasive disease screening, yet existing approaches typically focus on single conditions and fail to leverage the rich, multi-faceted information embedded in speech. We present MARVEL (Multi-task Acoustic Representations for Voice-based Health Analysis), a privacy-conscious multitask learning framework that simultaneously detects nine distinct neurological, respiratory, and voice disorders using only derived acoustic features, eliminating the need for raw audio transmission. Our dual-branch architecture employs specialized encoders with task-specific heads sharing a common acoustic backbone, enabling effective cross-condition knowledge transfer. Evaluated on the large-scale Bridge2AI-Voice v2.0 dataset, MARVEL achieves an overall AUROC of 0.78, with exceptional performance on neurological disorders (AUROC = 0.89), particularly for Alzheimer's disease/mild cognitive impairment (AUROC = 0.97). Our framework consistently outperforms single-modal baselines by 5-19% and surpasses state-of-the-art self-supervised models on 7 of 9 tasks, while correlation analysis reveals that the learned representations exhibit meaningful similarities with established acoustic features, indicating that the model's internal representations are consistent with clinically recognized acoustic patterns. By demonstrating that a single unified model can effectively screen for diverse conditions, this work establishes a foundation for deployable voice-based diagnostics in resource-constrained and remote healthcare settings.
>
---
#### [new 002] OLMoASR: Open Models and Data for Training Robust Speech Recognition Models
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文聚焦于语音识别任务，旨在训练鲁棒的零样本模型。通过构建大规模数据集OLMoASR-Pool，清洗生成高质量数据集OLMoASR-Mix，并训练多尺度模型，其性能与Whisper相当，推动了鲁棒语音处理研究。**

- **链接: [http://arxiv.org/pdf/2508.20869v1](http://arxiv.org/pdf/2508.20869v1)**

> **作者:** Huong Ngo; Matt Deitke; Martijn Bartelds; Sarah Pratt; Josh Gardner; Matt Jordan; Ludwig Schmidt
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Improvements in training data scale and quality have led to significant advances, yet its influence in speech recognition remains underexplored. In this paper, we present a large-scale dataset, OLMoASR-Pool, and series of models, OLMoASR, to study and develop robust zero-shot speech recognition models. Beginning from OLMoASR-Pool, a collection of 3M hours of English audio and 17M transcripts, we design text heuristic filters to remove low-quality or mistranscribed data. Our curation pipeline produces a new dataset containing 1M hours of high-quality audio-transcript pairs, which we call OLMoASR-Mix. We use OLMoASR-Mix to train the OLMoASR-Mix suite of models, ranging from 39M (tiny.en) to 1.5B (large.en) parameters. Across all model scales, OLMoASR achieves comparable average performance to OpenAI's Whisper on short and long-form speech recognition benchmarks. Notably, OLMoASR-medium.en attains a 12.8\% and 11.0\% word error rate (WER) that is on par with Whisper's largest English-only model Whisper-medium.en's 12.4\% and 10.5\% WER for short and long-form recognition respectively (at equivalent parameter count). OLMoASR-Pool, OLMoASR models, and filtering, training and evaluation code will be made publicly available to further research on robust speech processing.
>
---
#### [new 003] WoW-Bench: Evaluating Fine-Grained Acoustic Perception in Audio-Language Models via Marine Mammal Vocalizations
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出WoW-Bench，通过海洋哺乳动物叫声评估音频语言模型的细粒度声学感知（如音高、时长）及认知能力，揭示其在低级听觉任务上远逊于人类，需强化听觉基础。**

- **链接: [http://arxiv.org/pdf/2508.20976v1](http://arxiv.org/pdf/2508.20976v1)**

> **作者:** Jaeyeon Kim; Heeseung Yun; Sang Hoon Woo; Chao-Han Huck Yang; Gunhee Kim
>
> **备注:** Preprint. Project page: https://jaeyeonkim99.github.io/wow_bench/
>
> **摘要:** Large audio language models (LALMs) extend language understanding into the auditory domain, yet their ability to perform low-level listening, such as pitch and duration detection, remains underexplored. However, low-level listening is critical for real-world, out-of-distribution tasks where models must reason about unfamiliar sounds based on fine-grained acoustic cues. To address this gap, we introduce the World-of-Whale benchmark (WoW-Bench) to evaluate low-level auditory perception and cognition using marine mammal vocalizations. WoW-bench is composed of a Perception benchmark for categorizing novel sounds and a Cognition benchmark, inspired by Bloom's taxonomy, to assess the abilities to remember, understand, apply, and analyze sound events. For the Cognition benchmark, we additionally introduce distractor questions to evaluate whether models are truly solving problems through listening rather than relying on other heuristics. Experiments with state-of-the-art LALMs show performance far below human levels, indicating a need for stronger auditory grounding in LALMs.
>
---
#### [new 004] Learning Robust Spatial Representations from Binaural Audio through Feature Distillation
- **分类: cs.SD; cs.LG; eess.AS; 68T10; I.2.6**

- **简介: 该论文针对双耳音频的DOA估计任务，通过特征蒸馏预训练鲁棒空间表示，无需标签，微调后在噪声和混响环境中优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.20914v1](http://arxiv.org/pdf/2508.20914v1)**

> **作者:** Holger Severin Bovbjerg; Jan Østergaard; Jesper Jensen; Shinji Watanabe; Zheng-Hua Tan
>
> **备注:** To appear in Proc. WASPAA 2025, October 12-15, 2025, Tahoe, US. Copyright (c) 2025 IEEE. 5 pages, 2 figures, 2 tables
>
> **摘要:** Recently, deep representation learning has shown strong performance in multiple audio tasks. However, its use for learning spatial representations from multichannel audio is underexplored. We investigate the use of a pretraining stage based on feature distillation to learn a robust spatial representation of binaural speech without the need for data labels. In this framework, spatial features are computed from clean binaural speech samples to form prediction labels. These clean features are then predicted from corresponding augmented speech using a neural network. After pretraining, we throw away the spatial feature predictor and use the learned encoder weights to initialize a DoA estimation model which we fine-tune for DoA estimation. Our experiments demonstrate that the pretrained models show improved performance in noisy and reverberant environments after fine-tuning for direction-of-arrival estimation, when compared to fully supervised models and classic signal processing methods.
>
---
#### [new 005] Flowing Straighter with Conditional Flow Matching for Accurate Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文针对语音增强任务，提出条件流匹配以生成更直的概率路径，提升增强效果，并通过单步推理优化模型。**

- **链接: [http://arxiv.org/pdf/2508.20584v1](http://arxiv.org/pdf/2508.20584v1)**

> **作者:** Mattias Cross; Anton Ragni
>
> **备注:** preprint, accepted
>
> **摘要:** Current flow-based generative speech enhancement methods learn curved probability paths which model a mapping between clean and noisy speech. Despite impressive performance, the implications of curved probability paths are unknown. Methods such as Schrodinger bridges focus on curved paths, where time-dependent gradients and variance do not promote straight paths. Findings in machine learning research suggest that straight paths, such as conditional flow matching, are easier to train and offer better generalisation. In this paper we quantify the effect of path straightness on speech enhancement quality. We report experiments with the Schrodinger bridge, where we show that certain configurations lead to straighter paths. Conversely, we propose independent conditional flow-matching for speech enhancement, which models straight paths between noisy and clean speech. We demonstrate empirically that a time-independent variance has a greater effect on sample quality than the gradient. Although conditional flow matching improves several speech quality metrics, it requires multiple inference steps. We rectify this with a one-step solution by inferring the trained flow-based model as if it was directly predictive. Our work suggests that straighter time-independent probability paths improve generative speech enhancement over curved time-dependent paths.
>
---
#### [new 006] Amadeus: Autoregressive Model with Bidirectional Attribute Modelling for Symbolic Music
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出Amadeus框架，用于符号音乐生成，解决属性间严格时序依赖的假设问题，采用双层架构与两个增强模块，显著提升生成性能并实现属性控制。**

- **链接: [http://arxiv.org/pdf/2508.20665v1](http://arxiv.org/pdf/2508.20665v1)**

> **作者:** Hongju Su; Ke Li; Lan Yang; Honggang Zhang; Yi-Zhe Song
>
> **备注:** Under review
>
> **摘要:** Existing state-of-the-art symbolic music generation models predominantly adopt autoregressive or hierarchical autoregressive architectures, modelling symbolic music as a sequence of attribute tokens with unidirectional temporal dependencies, under the assumption of a fixed, strict dependency structure among these attributes. However, we observe that using different attributes as the initial token in these models leads to comparable performance. This suggests that the attributes of a musical note are, in essence, a concurrent and unordered set, rather than a temporally dependent sequence. Based on this insight, we introduce Amadeus, a novel symbolic music generation framework. Amadeus adopts a two-level architecture: an autoregressive model for note sequences and a bidirectional discrete diffusion model for attributes. To enhance performance, we propose Music Latent Space Discriminability Enhancement Strategy(MLSDES), incorporating contrastive learning constraints that amplify discriminability of intermediate music representations. The Conditional Information Enhancement Module (CIEM) simultaneously strengthens note latent vector representation via attention mechanisms, enabling more precise note decoding. We conduct extensive experiments on unconditional and text-conditioned generation tasks. Amadeus significantly outperforms SOTA models across multiple metrics while achieving at least 4$\times$ speed-up. Furthermore, we demonstrate training-free, fine-grained note attribute control feasibility using our model. To explore the upper performance bound of the Amadeus architecture, we compile the largest open-source symbolic music dataset to date, AMD (Amadeus MIDI Dataset), supporting both pre-training and fine-tuning.
>
---
#### [new 007] SincQDR-VAD: A Noise-Robust Voice Activity Detection Framework Leveraging Learnable Filters and Ranking-Aware Optimization
- **分类: cs.SD**

- **简介: 该论文针对语音活动检测在噪声环境中的鲁棒性不足问题，提出结合可学习滤波器和排序损失的框架，提升检测性能并减少参数量。**

- **链接: [http://arxiv.org/pdf/2508.20885v1](http://arxiv.org/pdf/2508.20885v1)**

> **作者:** Chien-Chun Wang; En-Lun Yu; Jeih-Weih Hung; Shih-Chieh Huang; Berlin Chen
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Voice activity detection (VAD) is essential for speech-driven applications, but remains far from perfect in noisy and resource-limited environments. Existing methods often lack robustness to noise, and their frame-wise classification losses are only loosely coupled with the evaluation metric of VAD. To address these challenges, we propose SincQDR-VAD, a compact and robust framework that combines a Sinc-extractor front-end with a novel quadratic disparity ranking loss. The Sinc-extractor uses learnable bandpass filters to capture noise-resistant spectral features, while the ranking loss optimizes the pairwise score order between speech and non-speech frames to improve the area under the receiver operating characteristic curve (AUROC). A series of experiments conducted on representative benchmark datasets show that our framework considerably improves both AUROC and F2-Score, while using only 69% of the parameters compared to prior arts, confirming its efficiency and practical viability.
>
---
#### [new 008] Speech Emotion Recognition via Entropy-Aware Score Selection
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出一种多模态语音情感识别方法，通过熵感知分数选择融合语音与文本预测，解决单一模态置信度问题，提升IEMOCAP和MSP-IMPROV数据集性能。**

- **链接: [http://arxiv.org/pdf/2508.20796v1](http://arxiv.org/pdf/2508.20796v1)**

> **作者:** ChenYi Chua; JunKai Wong; Chengxin Chen; Xiaoxiao Miao
>
> **备注:** The paper has been accepted by APCIPA ASC 2025
>
> **摘要:** In this paper, we propose a multimodal framework for speech emotion recognition that leverages entropy-aware score selection to combine speech and textual predictions. The proposed method integrates a primary pipeline that consists of an acoustic model based on wav2vec2.0 and a secondary pipeline that consists of a sentiment analysis model using RoBERTa-XLM, with transcriptions generated via Whisper-large-v3. We propose a late score fusion approach based on entropy and varentropy thresholds to overcome the confidence constraints of primary pipeline predictions. A sentiment mapping strategy translates three sentiment categories into four target emotion classes, enabling coherent integration of multimodal predictions. The results on the IEMOCAP and MSP-IMPROV datasets show that the proposed method offers a practical and reliable enhancement over traditional single-modality systems.
>
---
#### [new 009] MoTAS: MoE-Guided Feature Selection from TTS-Augmented Speech for Enhanced Multimodal Alzheimer's Early Screening
- **分类: cs.SD; cs.MM**

- **简介: 论文针对阿尔茨海默病早期筛查中的数据不足和特征选择问题，提出MoTAS框架，通过TTS增强数据和MoE机制优化多模态特征选择，提升筛查准确率至85.71%。**

- **链接: [http://arxiv.org/pdf/2508.20513v1](http://arxiv.org/pdf/2508.20513v1)**

> **作者:** Yongqi Shao; Binxin Mei; Cong Tan; Hong Huo; Tao Fang
>
> **摘要:** Early screening for Alzheimer's Disease (AD) through speech presents a promising non-invasive approach. However, challenges such as limited data and the lack of fine-grained, adaptive feature selection often hinder performance. To address these issues, we propose MoTAS, a robust framework designed to enhance AD screening efficiency. MoTAS leverages Text-to-Speech (TTS) augmentation to increase data volume and employs a Mixture of Experts (MoE) mechanism to improve multimodal feature selection, jointly enhancing model generalization. The process begins with automatic speech recognition (ASR) to obtain accurate transcriptions. TTS is then used to synthesize speech that enriches the dataset. After extracting acoustic and text embeddings, the MoE mechanism dynamically selects the most informative features, optimizing feature fusion for improved classification. Evaluated on the ADReSSo dataset, MoTAS achieves a leading accuracy of 85.71\%, outperforming existing baselines. Ablation studies further validate the individual contributions of TTS augmentation and MoE in boosting classification performance. These findings highlight the practical value of MoTAS in real-world AD screening scenarios, particularly in data-limited settings.
>
---
#### [new 010] Unifying Diarization, Separation, and ASR with Multi-Speaker Encoder
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出统一模型处理语音语义分割、分离和多说话人ASR，通过共享编码器与残差加权编码解决任务间依赖问题，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.20474v1](http://arxiv.org/pdf/2508.20474v1)**

> **作者:** Muhammad Shakeel; Yui Sudo; Yifan Peng; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** This paper presents a unified multi-speaker encoder (UME), a novel architecture that jointly learns representations for speaker diarization (SD), speech separation (SS), and multi-speaker automatic speech recognition (ASR) tasks using a shared speech foundational encoder. We leverage the hidden representations from multiple layers of UME as a residual weighted-sum encoding (RWSE) to effectively use information from different semantic levels, contributing to bottom-up alignment between tasks. This joint training approach captures the inherent interdependencies among the tasks, enhancing overall performance on overlapping speech data. Our evaluations demonstrate that UME substantially improves over the single-task baselines dedicated to SD, SS, and multi-speaker ASR on LibriMix evaluation sets. Notably, for SD, UME outperforms the previous studies, achieving diarization error rates of 1.37% and 2.29% on Libri2Mix and Libri3Mix evaluation sets, respectively.
>
---
#### [new 011] Live Vocal Extraction from K-pop Performances
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 论文提出通过源分离、交叉相关和幅度缩放技术，解决K-pop现场表演中自动分离人声与预录vocals/伴奏的问题，定义了live vocal separation任务。**

- **链接: [http://arxiv.org/pdf/2508.20273v1](http://arxiv.org/pdf/2508.20273v1)**

> **作者:** Yujin Kim; Richa Namballa; Magdalena Fuentes
>
> **备注:** 2 pages + references, 1 figure, Extended Abstracts for the Late-Breaking Demo Session of the 26th International Society for Music Information Retrieval Conference
>
> **摘要:** K-pop's global success is fueled by its dynamic performances and vibrant fan engagement. Inspired by K-pop fan culture, we propose a methodology for automatically extracting live vocals from performances. We use a combination of source separation, cross-correlation, and amplitude scaling to automatically remove pre-recorded vocals and instrumentals from a live performance. Our preliminary work introduces the task of live vocal separation and provides a foundation for future research in this topic.
>
---
#### [new 012] Scaling Fabric-Based Piezoresistive Sensor Arrays for Whole-Body Tactile Sensing
- **分类: cs.RO; eess.SP**

- **简介: 论文提出一种全身体触觉传感系统，解决布线复杂、数据通量和可靠性问题，通过织物传感器、定制电子和新型SPI拓扑实现高精度实时反馈，成功应用于机器人抓取任务。**

- **链接: [http://arxiv.org/pdf/2508.20959v1](http://arxiv.org/pdf/2508.20959v1)**

> **作者:** Curtis C. Johnson; Daniel Webb; David Hill; Marc D. Killpack
>
> **备注:** In submission to IEEE Sensors
>
> **摘要:** Scaling tactile sensing for robust whole-body manipulation is a significant challenge, often limited by wiring complexity, data throughput, and system reliability. This paper presents a complete architecture designed to overcome these barriers. Our approach pairs open-source, fabric-based sensors with custom readout electronics that reduce signal crosstalk to less than 3.3% through hardware-based mitigation. Critically, we introduce a novel, daisy-chained SPI bus topology that avoids the practical limitations of common wireless protocols and the prohibitive wiring complexity of USB hub-based systems. This architecture streams synchronized data from over 8,000 taxels across 1 square meter of sensing area at update rates exceeding 50 FPS, confirming its suitability for real-time control. We validate the system's efficacy in a whole-body grasping task where, without feedback, the robot's open-loop trajectory results in an uncontrolled application of force that slowly crushes a deformable cardboard box. With real-time tactile feedback, the robot transforms this motion into a gentle, stable grasp, successfully manipulating the object without causing structural damage. This work provides a robust and well-characterized platform to enable future research in advanced whole-body control and physical human-robot interaction.
>
---
#### [new 013] Exploring Machine Learning and Language Models for Multimodal Depression Detection
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文针对多模态人格感知抑郁症检测任务，通过比较XGBoost、Transformer和LLM在音频、视频、文本数据上的表现，分析其捕捉抑郁信号的能力，提出有效多模态表示策略。**

- **链接: [http://arxiv.org/pdf/2508.20805v1](http://arxiv.org/pdf/2508.20805v1)**

> **作者:** Javier Si Zhao Hong; Timothy Zoe Delaya; Sherwyn Chan Yin Kit; Pai Chet Ng; Xiaoxiao Miao
>
> **备注:** This paper has been accepted by APCIPA ASC 2025
>
> **摘要:** This paper presents our approach to the first Multimodal Personality-Aware Depression Detection Challenge, focusing on multimodal depression detection using machine learning and deep learning models. We explore and compare the performance of XGBoost, transformer-based architectures, and large language models (LLMs) on audio, video, and text features. Our results highlight the strengths and limitations of each type of model in capturing depression-related signals across modalities, offering insights into effective multimodal representation strategies for mental health prediction.
>
---
#### [new 014] CodecBench: A Comprehensive Benchmark for Acoustic and Semantic Evaluation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出CodecBench，一个综合评估音频编解码器声学和语义性能的基准数据集，旨在解决现有评估方法在复杂场景下的局限性，促进编解码器技术的发展。**

- **链接: [http://arxiv.org/pdf/2508.20660v1](http://arxiv.org/pdf/2508.20660v1)**

> **作者:** Ruifan Deng; Yitian Gong; Qinghui Gao; Luozhijie Jin; Qinyuan Cheng; Zhaoye Fei; Shimin Li; Xipeng Qiu
>
> **摘要:** With the rise of multimodal large language models (LLMs), audio codec plays an increasingly vital role in encoding audio into discrete tokens, enabling integration of audio into text-based LLMs. Current audio codec captures two types of information: acoustic and semantic. As audio codec is applied to diverse scenarios in speech language model , it needs to model increasingly complex information and adapt to varied contexts, such as scenarios with multiple speakers, background noise, or richer paralinguistic information. However, existing codec's own evaluation has been limited by simplistic metrics and scenarios, and existing benchmarks for audio codec are not designed for complex application scenarios, which limits the assessment performance on complex datasets for acoustic and semantic capabilities. We introduce CodecBench, a comprehensive evaluation dataset to assess audio codec performance from both acoustic and semantic perspectives across four data domains. Through this benchmark, we aim to identify current limitations, highlight future research directions, and foster advances in the development of audio codec. The codes are available at https://github.com/RayYuki/CodecBench.
>
---
#### [new 015] Automatic Inspection Based on Switch Sounds of Electric Point Machines
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文旨在通过分析电动转辙机切换声音实现自动化检测，解决传统视觉检查成本高、效率低的问题。2019年在NS型设备中安装传感器，利用声音信息实时监测道岔切换误差，实现故障早期预警，减少人工巡检需求。**

- **链接: [http://arxiv.org/pdf/2508.20870v1](http://arxiv.org/pdf/2508.20870v1)**

> **作者:** Ayano Shibata; Toshiki Gunji; Mitsuaki Tsuda; Takashi Endo; Kota Dohi; Tomoya Nishida; Satoko Nomoto
>
> **备注:** Accepted at ASPECT 2025
>
> **摘要:** Since 2018, East Japan Railway Company and Hitachi, Ltd. have been working to replace human inspections with IoT-based monitoring. The purpose is Labor-saving required for equipment inspections and provide appropriate preventive maintenance. As an alternative to visual inspection, it has been difficult to substitute electrical characteristic monitoring, and the introduction of new high-performance sensors has been costly. In 2019, we implemented cameras and microphones in an ``NS'' electric point machines to reduce downtime from equipment failures, allowing for remote monitoring of lock-piece conditions. This method for detecting turnout switching errors based on sound information was proposed, and the expected test results were obtained. The proposed method will make it possible to detect equipment failures in real time, thereby reducing the need for visual inspections. This paper presents the results of our technical studies aimed at automating the inspection of electronic point machines using sound, specifically focusing on ``switch sound'' beginning in 2019.
>
---
#### [new 016] Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels
- **分类: cs.CV; eess.SP**

- **简介: 论文针对自动调制识别（AMR）中依赖大量标注数据的问题，提出融合监督、自监督与重建目标的ViT框架，通过轻量化解码器和I/Q结构重建提升特征学习，仅用15-20%标签数据在RML2018.01A上达ResNet水平，实现高效、通用的AMR解决方案。**

- **链接: [http://arxiv.org/pdf/2508.20193v1](http://arxiv.org/pdf/2508.20193v1)**

> **作者:** Hossein Ahmadi; Banafsheh Saffari
>
> **摘要:** Automatic modulation recognition (AMR) is critical for cognitive radio, spectrum monitoring, and secure wireless communication. However, existing solutions often rely on large labeled datasets or multi-stage training pipelines, which limit scalability and generalization in practice. We propose a unified Vision Transformer (ViT) framework that integrates supervised, self-supervised, and reconstruction objectives. The model combines a ViT encoder, a lightweight convolutional decoder, and a linear classifier; the reconstruction branch maps augmented signals back to their originals, anchoring the encoder to fine-grained I/Q structure. This strategy promotes robust, discriminative feature learning during pretraining, while partial label supervision in fine-tuning enables effective classification with limited labels. On the RML2018.01A dataset, our approach outperforms supervised CNN and ViT baselines in low-label regimes, approaches ResNet-level accuracy with only 15-20% labeled data, and maintains strong performance across varying SNR levels. Overall, the framework provides a simple, generalizable, and label-efficient solution for AMR.
>
---
## 更新

#### [replaced 001] Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.17336v2](http://arxiv.org/pdf/2508.17336v2)**

> **作者:** Yunsik Kim; Yoonyoung Chung
>
> **摘要:** Body-conduction microphone signals (BMS) bypass airborne sound, providing strong noise resistance. However, a complementary modality is required to compensate for the inherent loss of high-frequency information. In this study, we propose a novel multi-modal framework that combines BMS and acoustic microphone signals (AMS) to achieve both noise suppression and high-frequency reconstruction. Unlike conventional multi-modal approaches that simply merge features, our method employs two specialized networks: a mapping-based model to enhance BMS and a masking-based model to denoise AMS. These networks are integrated through a dynamic fusion mechanism that adapts to local noise conditions, ensuring the optimal use of each modality's strengths. We performed evaluations on the TAPS dataset, augmented with DNS-2023 noise clips, using objective speech quality metrics. The results clearly demonstrate that our approach outperforms single-modal solutions in a wide range of noisy environments.
>
---
#### [replaced 002] Parallel GPT: Harmonizing the Independence and Interdependence of Acoustic and Semantic Information for Zero-Shot Text-to-Speech
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.04141v2](http://arxiv.org/pdf/2508.04141v2)**

> **作者:** Jingyuan Xing; Zhipeng Li; Jialong Mai; Xiaofen Xing; Xiangmin Xu
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
>
> **摘要:** Advances in speech representation and large language models have enhanced zero-shot text-to-speech (TTS) performance. However, existing zero-shot TTS models face challenges in capturing the complex correlations between acoustic and semantic features, resulting in a lack of expressiveness and similarity. The primary reason lies in the complex relationship between semantic and acoustic features, which manifests independent and interdependent aspects.This paper introduces a TTS framework that combines both autoregressive (AR) and non-autoregressive (NAR) modules to harmonize the independence and interdependence of acoustic and semantic information. The AR model leverages the proposed Parallel Tokenizer to synthesize the top semantic and acoustic tokens simultaneously. In contrast, considering the interdependence, the Coupled NAR model predicts detailed tokens based on the general AR model's output. Parallel GPT, built on this architecture, is designed to improve zero-shot text-to-speech synthesis through its parallel structure. Experiments on English and Chinese datasets demonstrate that the proposed model significantly outperforms the quality and efficiency of the synthesis of existing zero-shot TTS models. Speech demos are available at https://t1235-ch.github.io/pgpt/.
>
---
#### [replaced 003] Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation
- **分类: cs.CL; cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.16188v2](http://arxiv.org/pdf/2508.16188v2)**

> **作者:** Weiting Tan; Jiachen Lian; Hirofumi Inaguma; Paden Tomasello; Philipp Koehn; Xutai Ma
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** We present an Audio-Visual Language Model (AVLM) for expressive speech generation by integrating full-face visual cues into a pre-trained expressive speech model. We explore multiple visual encoders and multimodal fusion strategies during pre-training to identify the most effective integration approach. Subsequent fine-tuning on emotion recognition and expressive dialogue tasks yields substantial gains over speech-only baselines (e.g., +5 F1 in emotion recognition). AVLM highlights the value of expressive visual information in guiding speech generation and offers a foundation for end-to-end multimodal conversational systems.
>
---
#### [replaced 004] Noro: Noise-Robust One-shot Voice Conversion with Hidden Speaker Representation Learning
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.19770v2](http://arxiv.org/pdf/2411.19770v2)**

> **作者:** Haorui He; Yuchen Song; Yuancheng Wang; Haoyang Li; Xueyao Zhang; Li Wang; Gongping Huang; Eng Siong Chng; Zhizheng Wu
>
> **备注:** Accepted by APSIPA ASC 2025
>
> **摘要:** The effectiveness of one-shot voice conversion (VC) decreases in real-world scenarios where reference speeches, which are often sourced from the internet, contain various disturbances like background noise. To address this issue, we introduce Noro, a noise-robust one-shot VC system. Noro features innovative components tailored for VC using noisy reference speeches, including a dual-branch reference encoding module and a noise-agnostic contrastive speaker loss. Experimental results demonstrate that Noro outperforms our baseline system in both clean and noisy scenarios, highlighting its efficacy for real-world applications. Additionally, we investigate the hidden speaker representation capabilities of our baseline system by repurposing its reference encoder as a speaker encoder. The results show that it is competitive with several advanced self-supervised learning models for speaker representation under the SUPERB settings, highlighting the potential for advancing speaker representation learning through one-shot VC tasks.
>
---
#### [replaced 005] OLKAVS: An Open Large-Scale Korean Audio-Visual Speech Dataset
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2301.06375v2](http://arxiv.org/pdf/2301.06375v2)**

> **作者:** Jeongkyun Park; Jung-Wook Hwang; Kwanghee Choi; Seung-Hyun Lee; Jun Hwan Ahn; Rae-Hong Park; Hyung-Min Park
>
> **备注:** Accepted to ICASSP 2024
>
> **摘要:** Inspired by humans comprehending speech in a multi-modal manner, various audio-visual datasets have been constructed. However, most existing datasets focus on English, induce dependencies with various prediction models during dataset preparation, and have only a small number of multi-view videos. To mitigate the limitations, we recently developed the Open Large-scale Korean Audio-Visual Speech (OLKAVS) dataset, which is the largest among publicly available audio-visual speech datasets. The dataset contains 1,150 hours of transcribed audio from 1,107 Korean speakers in a studio setup with nine different viewpoints and various noise situations. We also provide the pre-trained baseline models for two tasks, audio-visual speech recognition and lip reading. We conducted experiments based on the models to verify the effectiveness of multi-modal and multi-view training over uni-modal and frontal-view-only training. We expect the OLKAVS dataset to facilitate multi-modal research in broader areas such as Korean speech recognition, speaker recognition, pronunciation level classification, and mouth motion analysis.
>
---
#### [replaced 006] Computational Extraction of Intonation and Tuning Systems from Multiple Microtonal Monophonic Vocal Recordings with Diverse Modes
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11956v2](http://arxiv.org/pdf/2503.11956v2)**

> **作者:** Sepideh Shafiei; Shapour Hakam
>
> **摘要:** This paper presents a computational methodology for analyzing intonation and deriving tuning systems in microtonal oral traditions, utilizing pitch histograms, Dynamic Time Warping (DTW), and optimization techniques, with a case study on a complete repertoire performed by a master of Iranian Classical Vocal Music (145 pieces). Pitch frequencies are extracted directly from vocal performances, and while alignment with MIDI notes is not a standard practice in our approach, we incorporate it where available, using DTW to refine interval analysis. By modeling intonation variations across multiple recordings, we derive structured tuning frameworks that capture both the flexibility of performance and the underlying systematic tendencies. Optimization techniques are applied to align intervals across the oral tradition repertoire, capturing the specific tunings and modal structures involved. Our methodology highlights the potential of computational techniques in advancing musicological and ethnomusicological research, offering a data-driven approach to defining tuning systems in microtonal vocal traditions.
>
---
#### [replaced 007] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.10082v3](http://arxiv.org/pdf/2507.10082v3)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [replaced 008] Advancing Speech Quality Assessment Through Scientific Challenges and Open-source Activities
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.00317v2](http://arxiv.org/pdf/2508.00317v2)**

> **作者:** Wen-Chin Huang
>
> **备注:** APSIPA ASC 2025 perspective paper
>
> **摘要:** Speech quality assessment (SQA) refers to the evaluation of speech quality, and developing an accurate automatic SQA method that reflects human perception has become increasingly important, in order to keep up with the generative AI boom. In recent years, SQA has progressed to a point that researchers started to faithfully use automatic SQA in research papers as a rigorous measurement of goodness for speech generation systems. We believe that the scientific challenges and open-source activities of late have stimulated the growth in this field. In this paper, we review recent challenges as well as open-source implementations and toolkits for SQA, and highlight the importance of maintaining such activities to facilitate the development of not only SQA itself but also generative AI for speech.
>
---
