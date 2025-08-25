# 音频 cs.SD;  eess.SP

- **最新发布 12 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] QvTAD: Differential Relative Attribute Learning for Voice Timbre Attribute Detection
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.15931v1](http://arxiv.org/pdf/2508.15931v1)**

> **作者:** Zhiyu Wu; Jingyi Fang; Yufei Tang; Yuanzhong Zheng; Yaoxuan Wang; Haojun Fei
>
> **备注:** Accepted by National Conference on Man-Machine Speech Communication, NCMMSC'2025
>
> **摘要:** Voice Timbre Attribute Detection (vTAD) plays a pivotal role in fine-grained timbre modeling for speech generation tasks. However, it remains challenging due to the inherently subjective nature of timbre descriptors and the severe label imbalance in existing datasets. In this work, we present QvTAD, a novel pairwise comparison framework based on differential attention, designed to enhance the modeling of perceptual timbre attributes. To address the label imbalance in the VCTK-RVA dataset, we introduce a graph-based data augmentation strategy that constructs a Directed Acyclic Graph and employs Disjoint-Set Union techniques to automatically mine unobserved utterance pairs with valid attribute comparisons. Our framework leverages speaker embeddings from a pretrained FACodec, and incorporates a Relative Timbre Shift-Aware Differential Attention module. This module explicitly models attribute-specific contrasts between paired utterances via differential denoising and contrast amplification mechanisms. Experimental results on the VCTK-RVA benchmark demonstrate that QvTAD achieves substantial improvements across multiple timbre descriptors, with particularly notable gains in cross-speaker generalization scenarios.
>
---
#### [new 002] Vevo2: Bridging Controllable Speech and Singing Voice Generation via Unified Prosody Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2框架，统一建模可控语音与歌唱生成任务。针对标注歌唱数据稀缺和控制灵活性不足问题，设计双音频分词器与两级建模结构，实现文本、韵律、风格与音色的解耦控制，提升跨模态迁移能力与合成质量。**

- **链接: [http://arxiv.org/pdf/2508.16332v1](http://arxiv.org/pdf/2508.16332v1)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** We will release code and model checkpoints at https://github.com/open-mmlab/Amphion
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a low-frame-rate (12.5 Hz) content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during pre-training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the AR model's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at https://versasinger.github.io/.
>
---
#### [new 003] Head-Related Transfer Function Individualization Using Anthropometric Features and Spatially Independent Latent Representation
- **分类: cs.SD; eess.AS**

- **简介: 论文提出一种基于体型参数的头相关传输函数（HRTF）个性化方法，旨在解决因测量成本高导致数据稀缺的问题。通过条件自编码器提取频域潜变量，融合多数据集并降低模型复杂度，提升估计精度。**

- **链接: [http://arxiv.org/pdf/2508.16176v1](http://arxiv.org/pdf/2508.16176v1)**

> **作者:** Ryan Niu; Shoichi Koyama; Tomohiko Nakamura
>
> **备注:** Accepted to IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** A method for head-related transfer function (HRTF) individualization from the subject's anthropometric parameters is proposed. Due to the high cost of measurement, the number of subjects included in many HRTF datasets is limited, and the number of those that include anthropometric parameters is even smaller. Therefore, HRTF individualization based on deep neural networks (DNNs) is a challenging task. We propose a HRTF individualization method using the latent representation of HRTF magnitude obtained through an autoencoder conditioned on sound source positions, which makes it possible to combine multiple HRTF datasets with different measured source positions, and makes the network training tractable by reducing the number of parameters to be estimated from anthropometric parameters. Experimental evaluation shows that high estimation accuracy is achieved by the proposed method, compared to current DNN-based methods.
>
---
#### [new 004] Beyond Transcription: Mechanistic Interpretability in ASR
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究自动语音识别（ASR）中的可解释性问题，旨在揭示模型内部如何处理声学与语义信息。作者应用logit lens、线性探测和激活修补等方法，发现编码器-解码器交互导致重复幻觉及深层声学表示中的语义偏差，提升了ASR的透明度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15882v1](http://arxiv.org/pdf/2508.15882v1)**

> **作者:** Neta Glazer; Yael Segal-Feldman; Hilit Segev; Aviv Shamsian; Asaf Buchnick; Gill Hetz; Ethan Fetaya; Joseph Keshet; Aviv Navon
>
> **摘要:** Interpretability methods have recently gained significant attention, particularly in the context of large language models, enabling insights into linguistic representations, error detection, and model behaviors such as hallucinations and repetitions. However, these techniques remain underexplored in automatic speech recognition (ASR), despite their potential to advance both the performance and interpretability of ASR systems. In this work, we adapt and systematically apply established interpretability methods such as logit lens, linear probing, and activation patching, to examine how acoustic and semantic information evolves across layers in ASR systems. Our experiments reveal previously unknown internal dynamics, including specific encoder-decoder interactions responsible for repetition hallucinations and semantic biases encoded deep within acoustic representations. These insights demonstrate the benefits of extending and applying interpretability techniques to speech recognition, opening promising directions for future research on improving model transparency and robustness.
>
---
#### [new 005] Continuous Determination of Respiratory Rate in Hospitalized Patients using Machine Learning Applied to Electrocardiogram Telemetry
- **分类: eess.SP; cs.CY; cs.LG**

- **简介: 论文提出用神经网络从心电图信号中连续准确估算呼吸频率，解决人工测量不准确、普通病房缺乏自动监测的问题。通过多数据集验证，误差低于1.78次/分，证明其在早期预警系统中的潜力。**

- **链接: [http://arxiv.org/pdf/2508.15947v1](http://arxiv.org/pdf/2508.15947v1)**

> **作者:** Thomas Kite; Brian Ayers; Nicholas Houstis; Asishana A. Osho; Thoralf M. Sundt; Aaron D Aguirre
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Respiration rate (RR) is an important vital sign for clinical monitoring of hospitalized patients, with changes in RR being strongly tied to changes in clinical status leading to adverse events. Human labels for RR, based on counting breaths, are known to be inaccurate and time consuming for medical staff. Automated monitoring of RR is in place for some patients, typically those in intensive care units (ICUs), but is absent for the majority of inpatients on standard medical wards who are still at risk for clinical deterioration. This work trains a neural network (NN) to label RR from electrocardiogram (ECG) telemetry waveforms, which like many biosignals, carry multiple signs of respiratory variation. The NN shows high accuracy on multiple validation sets (internal and external, same and different sources of RR labels), with mean absolute errors less than 1.78 breaths per minute (bpm) in the worst case. The clinical utility of such a technology is exemplified by performing a retrospective analysis of two patient cohorts that suffered adverse events including respiratory failure, showing that continuous RR monitoring could reveal dynamics that strongly tracked with intubation events. This work exemplifies the method of combining pre-existing telemetry monitoring systems and artificial intelligence (AI) to provide accurate, automated and scalable patient monitoring, all of which builds towards an AI-based hospital-wide early warning system (EWS).
>
---
#### [new 006] Robust Small Methane Plume Segmentation in Satellite Imagery
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于遥感图像中的小甲烷泄漏检测任务，旨在提升对小型甲烷排放源的识别精度。作者提出基于U-Net与ResNet34编码器的深度学习模型，结合双光谱增强技术，实现400平方米小 plume 检测，F1-score达78.39%，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.16282v1](http://arxiv.org/pdf/2508.16282v1)**

> **作者:** Khai Duc Minh Tran; Hoa Van Nguyen; Aimuni Binti Muhammad Rawi; Hareeshrao Athinarayanarao; Ba-Ngu Vo
>
> **备注:** 6 pages, 3 figures. This paper is submitted to the International Conference on Control, Automation and Information Sciences (ICCAIS) 2025, Jeju, Korea
>
> **摘要:** This paper tackles the challenging problem of detecting methane plumes, a potent greenhouse gas, using Sentinel-2 imagery. This contributes to the mitigation of rapid climate change. We propose a novel deep learning solution based on U-Net with a ResNet34 encoder, integrating dual spectral enhancement techniques (Varon ratio and Sanchez regression) to optimise input features for heightened sensitivity. A key achievement is the ability to detect small plumes down to 400 m2 (i.e., for a single pixel at 20 m resolution), surpassing traditional methods limited to larger plumes. Experiments show our approach achieves a 78.39% F1-score on the validation set, demonstrating superior performance in sensitivity and precision over existing remote sensing techniques for automated methane monitoring, especially for small plumes.
>
---
#### [new 007] A XAI-based Framework for Frequency Subband Characterization of Cough Spectrograms in Chronic Respiratory Disease
- **分类: cs.LG; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于医学信号分析任务，旨在通过XAI增强的频带分解方法，识别慢性呼吸疾病（尤其是COPD）咳嗽声谱中的可解释特征。工作包括训练CNN模型、生成遮挡图定位关键频段，并提取各子带特征以区分疾病类型与病程。**

- **链接: [http://arxiv.org/pdf/2508.16237v1](http://arxiv.org/pdf/2508.16237v1)**

> **作者:** Patricia Amado-Caballero; Luis M. San-José-Revuelta; Xinheng Wang; José Ramón Garmendia-Leiza; Carlos Alberola-López; Pablo Casaseca-de-la-Higuera
>
> **摘要:** This paper presents an explainable artificial intelligence (XAI)-based framework for the spectral analysis of cough sounds associated with chronic respiratory diseases, with a particular focus on Chronic Obstructive Pulmonary Disease (COPD). A Convolutional Neural Network (CNN) is trained on time-frequency representations of cough signals, and occlusion maps are used to identify diagnostically relevant regions within the spectrograms. These highlighted areas are subsequently decomposed into five frequency subbands, enabling targeted spectral feature extraction and analysis. The results reveal that spectral patterns differ across subbands and disease groups, uncovering complementary and compensatory trends across the frequency spectrum. Noteworthy, the approach distinguishes COPD from other respiratory conditions, and chronic from non-chronic patient groups, based on interpretable spectral markers. These findings provide insight into the underlying pathophysiological characteristics of cough acoustics and demonstrate the value of frequency-resolved, XAI-enhanced analysis for biomedical signal interpretation and translational respiratory disease diagnostics.
>
---
#### [new 008] TinyML Towards Industry 4.0: Resource-Efficient Process Monitoring of a Milling Machine
- **分类: cs.LG; cs.CV; cs.ET; cs.SY; eess.SP; eess.SY; I.2.1; I.5.4; C.5.3; C.3**

- **简介: 论文提出基于TinyML的铣床工艺监控方案，解决工业4.0中资源受限设备的实时质量监测问题。构建MillingVibes数据集，设计8-bit量化CNN模型，在ARM Cortex M4F上实现100%准确率、15.4ms推理时间，验证了其可行性。**

- **链接: [http://arxiv.org/pdf/2508.16553v1](http://arxiv.org/pdf/2508.16553v1)**

> **作者:** Tim Langer; Matthias Widra; Volkhard Beyer
>
> **备注:** 10 pages, 5 figures, 1 table
>
> **摘要:** In the context of industry 4.0, long-serving industrial machines can be retrofitted with process monitoring capabilities for future use in a smart factory. One possible approach is the deployment of wireless monitoring systems, which can benefit substantially from the TinyML paradigm. This work presents a complete TinyML flow from dataset generation, to machine learning model development, up to implementation and evaluation of a full preprocessing and classification pipeline on a microcontroller. After a short review on TinyML in industrial process monitoring, the creation of the novel MillingVibes dataset is described. The feasibility of a TinyML system for structure-integrated process quality monitoring could be shown by the development of an 8-bit-quantized convolutional neural network (CNN) model with 12.59kiB parameter storage. A test accuracy of 100.0% could be reached at 15.4ms inference time and 1.462mJ per quantized CNN inference on an ARM Cortex M4F microcontroller, serving as a reference for future TinyML process monitoring solutions.
>
---
#### [new 009] Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation
- **分类: cs.CL; cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 论文提出Audio-Visual Language Model（AVLM），通过融合面部视觉信息提升表达性语音生成效果，解决仅依赖语音导致的情感表达不足问题。工作包括探索视觉编码器与融合策略，并在情感识别和对话任务上实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.16188v1](http://arxiv.org/pdf/2508.16188v1)**

> **作者:** Weiting Tan; Jiachen Lian; Hirofumi Inaguma; Paden Tomasello; Philipp Koehn; Xutai Ma
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** We present an Audio-Visual Language Model (AVLM) for expressive speech generation by integrating full-face visual cues into a pre-trained expressive speech model. We explore multiple visual encoders and multimodal fusion strategies during pre-training to identify the most effective integration approach. Subsequent fine-tuning on emotion recognition and expressive dialogue tasks yields substantial gains over speech-only baselines (e.g., +5 F1 in emotion recognition). AVLM highlights the value of expressive visual information in guiding speech generation and offers a foundation for end-to-end multimodal conversational systems.
>
---
#### [new 010] MGSC: A Multi-granularity Consistency Framework for Robust End-to-end Asr
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; I.2.7**

- **简介: 该论文针对端到端语音识别（ASR）在噪声环境下易产生严重语义错误的问题，提出多粒度一致性框架MGSC。通过同时约束句子级语义和词元级对齐的一致性，显著提升模型鲁棒性，降低字符错误率。**

- **链接: [http://arxiv.org/pdf/2508.15853v1](http://arxiv.org/pdf/2508.15853v1)**

> **作者:** Xuwen Yang
>
> **备注:** 12 pages, 5figures
>
> **摘要:** End-to-end ASR models, despite their success on benchmarks, often pro-duce catastrophic semantic errors in noisy environments. We attribute this fragility to the prevailing 'direct mapping' objective, which solely penalizes final output errors while leaving the model's internal computational pro-cess unconstrained. To address this, we introduce the Multi-Granularity Soft Consistency (MGSC) framework, a model-agnostic, plug-and-play module that enforces internal self-consistency by simultaneously regulariz-ing macro-level sentence semantics and micro-level token alignment. Cru-cially, our work is the first to uncover a powerful synergy between these two consistency granularities: their joint optimization yields robustness gains that significantly surpass the sum of their individual contributions. On a public dataset, MGSC reduces the average Character Error Rate by a relative 8.7% across diverse noise conditions, primarily by preventing se-vere meaning-altering mistakes. Our work demonstrates that enforcing in-ternal consistency is a crucial step towards building more robust and trust-worthy AI.
>
---
#### [new 011] Terrain Classification for the Spot Quadrupedal Mobile Robot Using Only Proprioceptive Sensing
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于机器人地形分类任务，旨在解决四足机器人在复杂地形中易陷入或打滑的问题。作者利用波士顿动力Spot机器人的本体感觉信号，通过降维和分类技术，实现了三种地形的高精度识别（约97%），为路径规划提供 traversability 信息。**

- **链接: [http://arxiv.org/pdf/2508.16504v1](http://arxiv.org/pdf/2508.16504v1)**

> **作者:** Sophie Villemure; Jefferson Silveira; Joshua A. Marshall
>
> **摘要:** Quadrupedal mobile robots can traverse a wider range of terrain types than their wheeled counterparts but do not perform the same on all terrain types. These robots are prone to undesirable behaviours like sinking and slipping on challenging terrains. To combat this issue, we propose a terrain classifier that provides information on terrain type that can be used in robotic systems to create a traversability map to plan safer paths for the robot to navigate. The work presented here is a terrain classifier developed for a Boston Dynamics Spot robot. Spot provides over 100 measured proprioceptive signals describing the motions of the robot and its four legs (e.g., foot penetration, forces, joint angles, etc.). The developed terrain classifier combines dimensionality reduction techniques to extract relevant information from the signals and then applies a classification technique to differentiate terrain based on traversability. In representative field testing, the resulting terrain classifier was able to identify three different terrain types with an accuracy of approximately 97%
>
---
#### [new 012] Audio2Face-3D: Audio-driven Realistic Facial Animation For Digital Avatars
- **分类: cs.GR; cs.HC; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Audio2Face-3D系统，解决音频驱动数字人面部动画生成问题。通过构建数据集、网络架构和重定向方法，实现实时、逼真的面部动画，助力游戏开发与虚拟角色创作。**

- **链接: [http://arxiv.org/pdf/2508.16401v1](http://arxiv.org/pdf/2508.16401v1)**

> **作者:** NVIDIA; :; Chaeyeon Chung; Ilya Fedorov; Michael Huang; Aleksey Karmanov; Dmitry Korobchenko; Roger Ribera; Yeongho Seol
>
> **摘要:** Audio-driven facial animation presents an effective solution for animating digital avatars. In this paper, we detail the technical aspects of NVIDIA Audio2Face-3D, including data acquisition, network architecture, retargeting methodology, evaluation metrics, and use cases. Audio2Face-3D system enables real-time interaction between human users and interactive avatars, facilitating facial animation authoring for game characters. To assist digital avatar creators and game developers in generating realistic facial animations, we have open-sourced Audio2Face-3D networks, SDK, training framework, and example dataset.
>
---
## 更新

#### [replaced 001] Privacy in Speech Technology
- **分类: eess.AS; cs.CR; cs.SD**

- **链接: [http://arxiv.org/pdf/2305.05227v3](http://arxiv.org/pdf/2305.05227v3)**

> **作者:** Tom Bäckström
>
> **摘要:** Speech technology for communication, accessing information, and services has rapidly improved in quality. It is convenient and appealing because speech is the primary mode of communication for humans. Such technology, however, also presents proven threats to privacy. Speech is a tool for communication and it will thus inherently contain private information. Importantly, it however also contains a wealth of side information, such as information related to health, emotions, affiliations, and relationships, all of which are private. Exposing such private information can lead to serious threats such as price gouging, harassment, extortion, and stalking. This paper is a tutorial on privacy issues related to speech technology, modeling their threats, approaches for protecting users' privacy, measuring the performance of privacy-protecting methods, perception of privacy as well as societal and legal consequences. In addition to a tutorial overview, it also presents lines for further development where improvements are most urgently needed.
>
---
#### [replaced 002] Improving Speech Enhancement with Multi-Metric Supervision from Learned Quality Assessment
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12260v2](http://arxiv.org/pdf/2506.12260v2)**

> **作者:** Wei Wang; Wangyou Zhang; Chenda Li; Jiatong Shi; Shinji Watanabe; Yanmin Qian
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Speech quality assessment (SQA) aims to predict the perceived quality of speech signals under a wide range of distortions. It is inherently connected to speech enhancement (SE), which seeks to improve speech quality by removing unwanted signal components. While SQA models are widely used to evaluate SE performance, their potential to guide SE training remains underexplored. In this work, we investigate a training framework that leverages a SQA model, trained to predict multiple evaluation metrics from a public SE leaderboard, as a supervisory signal for SE. This approach addresses a key limitation of conventional SE objectives, such as SI-SNR, which often fail to align with perceptual quality and generalize poorly across evaluation metrics. Moreover, it enables training on real-world data where clean references are unavailable. Experiments on both simulated and real-world test sets show that SQA-guided training consistently improves performance across a range of quality metrics. Code and checkpoints are available at https://github.com/urgent-challenge/urgent2026_challenge_track2
>
---
#### [replaced 003] Enhancing Code-switched Text-to-Speech Synthesis Capability in Large Language Models with only Monolingual Corpora
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.10969v2](http://arxiv.org/pdf/2409.10969v2)**

> **作者:** Jing Xu; Daxin Tan; Jiaqi Wang; Xiao Chen
>
> **备注:** Accepted to ASRU2025
>
> **摘要:** While Large Language Models (LLMs) have shown potential in speech generation and recognition, their applications are mainly confined to monolingual scenarios, with limited explorations in code-switched (CS) contexts. In this paper, we propose a Code-Switched Large Language Model (CS-LLM) to enhance the code-switched text-to-speech synthesis (CS TTS) capability in LLMs with only monolingual corpora. Specifically, we begin by enhancing the multilingual speech processing ability of LLMs through multilingual speech recognition and synthesis tasks. Then, we develop an effective code-switched (CS) data construction strategy that splits and concatenates words from different monolingual speech corpora to equip LLMs with improved CS TTS ability. Experiments show that our approach outperforms baselines in CS TTS in terms of naturalness, speaker consistency and similarity even with limited data. Additionally, the constructed CS data further improves multilingual speech synthesis and recognition.
>
---
#### [replaced 004] Revealing the Role of Audio Channels in ASR Performance Degradation
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08967v2](http://arxiv.org/pdf/2508.08967v2)**

> **作者:** Kuan-Tang Huang; Li-Wei Chen; Hung-Shin Lee; Berlin Chen; Hsin-Min Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences.
>
---
#### [replaced 005] Sentiment Reasoning for Healthcare
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.21054v5](http://arxiv.org/pdf/2407.21054v5)**

> **作者:** Khai-Nguyen Nguyen; Khai Le-Duc; Bach Phan Tat; Duy Le; Long Vo-Dang; Truong-Son Hy
>
> **备注:** ACL 2025 Industry Track (Oral)
>
> **摘要:** Transparency in AI healthcare decision-making is crucial. By incorporating rationales to explain reason for each predicted label, users could understand Large Language Models (LLMs)'s reasoning to make better decision. In this work, we introduce a new task - Sentiment Reasoning - for both speech and text modalities, and our proposed multimodal multitask framework and the world's largest multimodal sentiment analysis dataset. Sentiment Reasoning is an auxiliary task in sentiment analysis where the model predicts both the sentiment label and generates the rationale behind it based on the input transcript. Our study conducted on both human transcripts and Automatic Speech Recognition (ASR) transcripts shows that Sentiment Reasoning helps improve model transparency by providing rationale for model prediction with quality semantically comparable to humans while also improving model's classification performance (+2% increase in both accuracy and macro-F1) via rationale-augmented fine-tuning. Also, no significant difference in the semantic quality of generated rationales between human and ASR transcripts. All code, data (five languages - Vietnamese, English, Chinese, German, and French) and models are published online: https://github.com/leduckhai/Sentiment-Reasoning
>
---
