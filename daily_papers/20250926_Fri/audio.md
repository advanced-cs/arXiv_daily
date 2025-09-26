# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] SingVERSE: A Diverse, Real-World Benchmark for Singing Voice Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出了SingVERSE，首个真实场景下的歌声增强基准数据集，旨在解决缺乏真实评估数据的问题。通过该基准，全面评估了先进模型，并揭示了感知质量与可懂度的权衡关系，为未来研究提供了基础支持和关键见解。**

- **链接: [http://arxiv.org/pdf/2509.20969v1](http://arxiv.org/pdf/2509.20969v1)**

> **作者:** Shaohan Jiang; Junan Zhang; Yunjia Zhang; Jing Yang; Fan Fan; Zhizheng Wu
>
> **备注:** Demopage: https://singverse.github.io, Dataset: https://huggingface.co/datasets/amphion/SingVERSE
>
> **摘要:** This paper presents a benchmark for singing voice enhancement. The development of singing voice enhancement is limited by the lack of realistic evaluation data. To address this gap, this paper introduces SingVERSE, the first real-world benchmark for singing voice enhancement, covering diverse acoustic scenarios and providing paired, studio-quality clean references. Leveraging SingVERSE, we conduct a comprehensive evaluation of state-of-the-art models and uncover a consistent trade-off between perceptual quality and intelligibility. Finally, we show that training on in-domain singing data substantially improves enhancement performance without degrading speech capabilities, establishing a simple yet effective path forward. This work offers the community a foundational benchmark together with critical insights to guide future advances in this underexplored domain. Demopage: https://singverse.github.io
>
---
#### [new 002] i-LAVA: Insights on Low Latency Voice-2-Voice Architecture for Agents
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究低延迟语音到语音（V-2-V）系统，旨在优化实时对话应用的处理速度与交互质量。重点分析ASR、TTS和对话管理组件，发现TTS对实时性影响最大，并通过减少RVQ迭代和代码本数量实现性能优化。**

- **链接: [http://arxiv.org/pdf/2509.20971v1](http://arxiv.org/pdf/2509.20971v1)**

> **作者:** Anupam Purwar; Aditya Choudhary
>
> **备注:** This paper analyzes a low-latency, end-to-end voice-to-voice (V-2-V) architecture, identifying that the Text-to-Speech (TTS) component has the highest impact on real-time performance. By reducing the number of Residual Vector Quantization (RVQ) iterations in the TTS model, latency can be effectively halved, creating a direct trade-off between conversational speed and audio quality
>
> **摘要:** We experiment with a low-latency, end-to-end voice-to-voice communication model to optimize it for real-time conversational applications. By analyzing components essential to voice to voice (V-2-V) system viz. automatic speech recognition (ASR), text-to-speech (TTS), and dialog management, our work analyzes how to reduce processing time while maintaining high-quality interactions to identify the levers for optimizing V-2-V system. Our work identifies that TTS component which generates life-like voice, full of emotions including natural pauses and exclamations has highest impact on Real time factor (RTF). The experimented V-2-V architecture utilizes CSM1b has the capability to understand tone as well as context of conversation by ingesting both audio and text of prior exchanges to generate contextually accurate speech. We explored optimization of Residual Vector Quantization (RVQ) iterations by the TTS decoder which come at a cost of decrease in the quality of voice generated. Our experimental evaluations also demonstrate that for V-2-V implementations based on CSM most important optimizations can be brought by reducing the number of RVQ Iterations along with the codebooks used in Mimi.
>
---
#### [new 003] SupCLAP: Controlling Optimization Trajectory Drift in Audio-Text Contrastive Learning with Support Vector Regularization
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对音频-文本对比学习中的优化轨迹漂移问题，提出支持向量正则化（SVR）方法，通过引入辅助支持向量控制负样本的垂直推力，提升训练稳定性与模型性能。**

- **链接: [http://arxiv.org/pdf/2509.21033v1](http://arxiv.org/pdf/2509.21033v1)**

> **作者:** Jiehui Luo; Yuguo Yin; Yuxin Xie; Jinghan Ru; Xianwei Zhuang; Minghua He; Aofan Liu; Zihan Xiong; Dongchao Yang
>
> **摘要:** Contrastive language-audio pretraining, which aims to unify multimodal representations in a shared embedding space, serves as a cornerstone for building a wide range of applications, from cross-modal retrieval to cutting-edge multimodal large language models. However, we find that the perpendicular component of the pushing force from negative samples in contrastive learning is a double-edged sword: it contains rich supplementary information from negative samples, yet its unconstrained nature causes optimization trajectory drift and training instability. To address this, we propose Support Vector Regularization (SVR), a method that introduces an auxiliary support vector to control this perpendicular component, aiming to harness its rich information while mitigating the associated trajectory drift. The efficacy of SVR is critically governed by its semantic radius, for which we explore two unsupervised modeling strategies: direct parameterization and an adaptive radius predictor module enhanced with constraints to improve its predicting accuracy. Extensive experimental results demonstrate that our method surpasses widely used baselines like InfoNCE and SigLIP loss across classification, monolingual retrieval, and multilingual retrieval on standard audio-text datasets. Both the theoretical analysis and the experimental results on optimizing trajectory drift validate the correctness and effectiveness of our SVR method.
>
---
#### [new 004] Addressing Gradient Misalignment in Data-Augmented Training for Robust Speech Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对语音深度伪造检测任务中数据增强训练存在的梯度错位问题，提出双路径数据增强（DPDA）框架，通过对齐原始与增强输入的梯度方向，减少优化冲突，提升模型收敛速度和检测性能。**

- **链接: [http://arxiv.org/pdf/2509.20682v1](http://arxiv.org/pdf/2509.20682v1)**

> **作者:** Duc-Tuan Truong; Tianchi Liu; Junjie Li; Ruijie Tao; Kong Aik Lee; Eng Siong Chng
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** In speech deepfake detection (SDD), data augmentation (DA) is commonly used to improve model generalization across varied speech conditions and spoofing attacks. However, during training, the backpropagated gradients from original and augmented inputs may misalign, which can result in conflicting parameter updates. These conflicts could hinder convergence and push the model toward suboptimal solutions, thereby reducing the benefits of DA. To investigate and address this issue, we design a dual-path data-augmented (DPDA) training framework with gradient alignment for SDD. In our framework, each training utterance is processed through two input paths: one using the original speech and the other with its augmented version. This design allows us to compare and align their backpropagated gradient directions to reduce optimization conflicts. Our analysis shows that approximately 25% of training iterations exhibit gradient conflicts between the original inputs and their augmented counterparts when using RawBoost augmentation. By resolving these conflicts with gradient alignment, our method accelerates convergence by reducing the number of training epochs and achieves up to an 18.69% relative reduction in Equal Error Rate on the In-the-Wild dataset compared to the baseline.
>
---
#### [new 005] UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出UniSS，一种统一的表达型语音到语音翻译框架，旨在解决语音数据稀缺、多阶段处理复杂及翻译能力迁移受限的问题。通过设计语音语义与风格建模，并结合跨模态推理方法，实现高质量、保留语音特征的翻译。**

- **链接: [http://arxiv.org/pdf/2509.21144v1](http://arxiv.org/pdf/2509.21144v1)**

> **作者:** Sitong Cheng; Weizhen Bian; Xinsheng Wang; Ruibin Yuan; Jianyi Chen; Shunshun Yin; Yike Guo; Wei Xue
>
> **摘要:** The ultimate goal of expressive speech-to-speech translation (S2ST) is to accurately translate spoken content while preserving the speaker identity and emotional style. However, progress in this field is largely hindered by three key challenges: the scarcity of paired speech data that retains expressive styles, the complexity of multi-stage processing pipelines, and the limited transfer of translation capabilities from large language models (LLMs). In this work, we address these challenges by introducing UniSS, a novel single-stage framework for expressive S2ST. Our approach features carefully designed speech semantic and style modeling, enabling seamless integration with existing text-based LLM frameworks to develop a unified text-speech language model. To transfer translation capabilities from text to speech, we propose a cross-modal chain-of-thought prompting process that progressively aligns audio semantics with text and ensures style preservation in the decoded results. Furthermore, we construct and release a large-scale, high-quality expressive S2ST dataset, UniST, comprising 44.8k hours of data. Experimental results show that UniSS significantly outperforms previous methods in translation fidelity and speech quality while preserving voice, emotion, and duration consistency. Our work establishes a simpler and more effective paradigm for building the next generation of expressive S2ST systems. Audio samples are available at https://cmots.github.io/uniss-demo.
>
---
#### [new 006] QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出QAMO，用于语音深度伪造检测。针对传统单中心一类学习忽略语音质量的问题，QAMO引入多质量感知中心，更好建模真实语音的类内变化，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.20679v1](http://arxiv.org/pdf/2509.20679v1)**

> **作者:** Duc-Tuan Truong; Tianchi Liu; Ruijie Tao; Junjie Li; Kong Aik Lee; Eng Siong Chng
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Recent work shows that one-class learning can detect unseen deepfake attacks by modeling a compact distribution of bona fide speech around a single centroid. However, the single-centroid assumption can oversimplify the bona fide speech representation and overlook useful cues, such as speech quality, which reflects the naturalness of the speech. Speech quality can be easily obtained using existing speech quality assessment models that estimate it through Mean Opinion Score. In this paper, we propose QAMO: Quality-Aware Multi-Centroid One-Class Learning for speech deepfake detection. QAMO extends conventional one-class learning by introducing multiple quality-aware centroids. In QAMO, each centroid is optimized to represent a distinct speech quality subspaces, enabling better modeling of intra-class variability in bona fide speech. In addition, QAMO supports a multi-centroid ensemble scoring strategy, which improves decision thresholding and reduces the need for quality labels during inference. With two centroids to represent high- and low-quality speech, our proposed QAMO achieves an equal error rate of 5.09% in In-the-Wild dataset, outperforming previous one-class and quality-aware systems.
>
---
#### [new 007] AIBA: Attention-based Instrument Band Alignment for Text-to-Audio Diffusion
- **分类: cs.SD**

- **简介: 该论文提出AIBA，一种无需训练的轻量级方法，用于分析文本到音频扩散模型在时频平面上的关注区域。通过交叉注意力机制，将关注概率映射到梅尔网格，并用可解释指标评估与乐器频段的对齐程度，应用于Slakh2100数据集。**

- **链接: [http://arxiv.org/pdf/2509.20891v1](http://arxiv.org/pdf/2509.20891v1)**

> **作者:** Junyoung Koh; Soo Yong Kim; Gyu Hyeong Choi; Yongwon Choi
>
> **备注:** NeurIPS 2025 AI for Music Workshop
>
> **摘要:** We present AIBA (Attention-In-Band Alignment), a lightweight, training-free pipeline to quantify where text-to-audio diffusion models attend on the time-frequency (T-F) plane. AIBA (i) hooks cross-attention at inference to record attention probabilities without modifying weights; (ii) projects them to fixed-size mel grids that are directly comparable to audio energy; and (iii) scores agreement with instrument-band ground truth via interpretable metrics (T-F IoU/AP, frequency-profile correlation, and a pointing game). On Slakh2100 with an AudioLDM2 backbone, AIBA reveals consistent instrument-dependent trends (e.g., bass favoring low bands) and achieves high precision with moderate recall.
>
---
#### [new 008] Investigating Modality Contribution in Audio LLMs for Music
- **分类: cs.LG; cs.SD**

- **简介: 该论文研究音频大语言模型（Audio LLMs）在音乐任务中对音频和文本模态的依赖程度。针对模型是否真正“听”音频的问题，作者采用MM-SHAP方法量化各模态贡献，评估发现高准确率模型更依赖文本，但音频仍有一定作用。这是首次将MM-SHAP应用于音频LLMs，为可解释性AI提供基础。**

- **链接: [http://arxiv.org/pdf/2509.20641v1](http://arxiv.org/pdf/2509.20641v1)**

> **作者:** Giovana Morais; Magdalena Fuentes
>
> **摘要:** Audio Large Language Models (Audio LLMs) enable human-like conversation about music, yet it is unclear if they are truly listening to the audio or just using textual reasoning, as recent benchmarks suggest. This paper investigates this issue by quantifying the contribution of each modality to a model's output. We adapt the MM-SHAP framework, a performance-agnostic score based on Shapley values that quantifies the relative contribution of each modality to a model's prediction. We evaluate two models on the MuChoMusic benchmark and find that the model with higher accuracy relies more on text to answer questions, but further inspection shows that even if the overall audio contribution is low, models can successfully localize key sound events, suggesting that audio is not entirely ignored. Our study is the first application of MM-SHAP to Audio LLMs and we hope it will serve as a foundational step for future research in explainable AI and audio.
>
---
#### [new 009] Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出Phoenix-VAD，一种基于大语言模型的流式语义端点检测方法，用于全双工语音交互。旨在解决现有对话系统缺乏可靠语义端点检测模块的问题，通过滑动窗口训练策略实现独立优化的预测模块，提升交互的灵活性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.20410v1](http://arxiv.org/pdf/2509.20410v1)**

> **作者:** Weijie Wu; Wenhao Guan; Kaidi Wang; Peijie Chen; Zhuanling Zha; Junbo Li; Jun Fang; Lin Li; Qingyang Hong
>
> **摘要:** Spoken dialogue models have significantly advanced intelligent human\textendash computer interaction, yet they lack a plug\textendash and\textendash play full\textendash duplex prediction module for semantic endpoint detection, hindering seamless audio interactions. In this paper, we introduce Phoenix\textendashVAD, an LLM\textendash based model that enables streaming semantic endpoint detection. Specifically, Phoenix\textendash VAD leverages the semantic comprehension capability of the LLM and a sliding window training strategy to achieve reliable semantic endpoint detection while supporting streaming inference. Experiments on both semantically complete and incomplete speech scenarios indicate that Phoenix\textendash VAD achieves excellent and competitive performance. Furthermore, this design enables the full\textendash duplex prediction module to be optimized independently of the dialogue model, providing more reliable and flexible support for next\textendash generation human\textendash computer interaction.
>
---
#### [new 010] MI-Fuse: Label Fusion for Unsupervised Domain Adaptation with Closed-Source Large-Audio Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对语音情感识别（SER）任务，研究在源域数据不可用、仅能通过API访问大模型的情况下，如何利用目标域无标签数据提升学生模型性能。提出MI-Fuse方法，融合大模型与辅助教师模型的预测，实现跨域自适应，实验表明效果优于基线和原始大模型。**

- **链接: [http://arxiv.org/pdf/2509.20706v1](http://arxiv.org/pdf/2509.20706v1)**

> **作者:** Hsiao-Ying Huang; Yi-Cheng Lin; Hung-yi Lee
>
> **备注:** 5 pages, 2 figures, 2 tables
>
> **摘要:** Large audio-language models (LALMs) show strong zero-shot ability on speech tasks, suggesting promise for speech emotion recognition (SER). However, SER in real-world deployments often fails under domain mismatch, where source data are unavailable and powerful LALMs are accessible only through an API. We ask: given only unlabeled target-domain audio and an API-only LALM, can a student model be adapted to outperform the LALM in the target domain? To this end, we propose MI-Fuse, a denoised label fusion framework that supplements the LALM with a source-domain trained SER classifier as an auxiliary teacher. The framework draws multiple stochastic predictions from both teachers, weights their mean distributions by mutual-information-based uncertainty, and stabilizes training with an exponential moving average teacher. Experiments across three public emotion datasets and six cross-domain transfers show consistent gains, with the student surpassing the LALM and outperforming the strongest baseline by 3.9%. This approach strengthens emotion-aware speech systems without sharing source data, enabling realistic adaptation.
>
---
#### [new 011] Building Tailored Speech Recognizers for Japanese Speaking Assessment
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究针对日语口语评估的定制语音识别任务，旨在解决音素标注（含重音标记）数据稀缺问题。提出了多任务学习和估计器融合方法，有效降低了音节标注错误率。**

- **链接: [http://arxiv.org/pdf/2509.20655v1](http://arxiv.org/pdf/2509.20655v1)**

> **作者:** Yotaro Kubo; Richard Sproat; Chihiro Taguchi; Llion Jones
>
> **摘要:** This paper presents methods for building speech recognizers tailored for Japanese speaking assessment tasks. Specifically, we build a speech recognizer that outputs phonemic labels with accent markers. Although Japanese is resource-rich, there is only a small amount of data for training models to produce accurate phonemic transcriptions that include accent marks. We propose two methods to mitigate data sparsity. First, a multitask training scheme introduces auxiliary loss functions to estimate orthographic text labels and pitch patterns of the input signal, so that utterances with only orthographic annotations can be leveraged in training. The second fuses two estimators, one over phonetic alphabet strings, and the other over text token sequences. To combine these estimates we develop an algorithm based on the finite-state transducer framework. Our results indicate that the use of multitask learning and fusion is effective for building an accurate phonemic recognizer. We show that this approach is advantageous compared to the use of generic multilingual recognizers. The relative advantages of the proposed methods were also compared. Our proposed methods reduced the average of mora-label error rates from 12.3% to 7.1% over the CSJ core evaluation sets.
>
---
#### [new 012] AuthGlass: Enhancing Voice Authentication on Smart Glasses via Air-Bone Acoustic Features
- **分类: cs.HC; cs.SD**

- **简介: 该论文提出AuthGlass，一种针对智能眼镜的语音认证方法，通过结合空气传导和骨传导声学特征，提升认证准确性和防攻击能力。研究构建了含14个空气麦克风和2个骨传导传感器的原型，并在42名参与者中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.20799v1](http://arxiv.org/pdf/2509.20799v1)**

> **作者:** Weiye Xu; Zhang Jiang; Siqi Zheng; Xiyuxing Zhang; Yankai Zhao; Changhao Zhang; Jian Liu; Weiqiang Wang; Yuntao Wang
>
> **备注:** 24 pages, 12 figures, submitted to CHI'26
>
> **摘要:** With the rapid advancement of smart glasses, voice interaction has become widely deployed due to its naturalness and convenience. However, its practicality is often undermined by the vulnerability to spoofing attacks and interference from surrounding sounds, making seamless voice authentication crucial for smart glasses usage. To address this challenge, we propose AuthGlass, a voice authentication approach that leverages both air- and bone-conducted speech features to enhance accuracy and liveness detection. Aiming to gain comprehensive knowledge on speech-related acoustic and vibration features, we built a smart glasses prototype with redundant synchronized microphones: 14 air-conductive microphones and 2 bone-conductive units. In a study with 42 participants, we validated that combining sound-field and vibration features significantly improves authentication robustness and attack resistance. Furthermore, experiments demonstrated that AuthGlass maintains competitive accuracy even under various practical scenarios, highlighting its applicability and scalability for real-world deployment.
>
---
#### [new 013] Data-Efficient ASR Personalization for Non-Normative Speech Using an Uncertainty-Based Phoneme Difficulty Score for Guided Sampling
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文针对非规范性语音的ASR个性化问题，提出一种基于不确定性的音素难度评分方法，通过蒙特卡洛Dropout估计模型困难音素并进行针对性过采样，显著提升了ASR准确性。**

- **链接: [http://arxiv.org/pdf/2509.20396v1](http://arxiv.org/pdf/2509.20396v1)**

> **作者:** Niclas Pokel; Pehuén Moure; Roman Boehringer; Yingqiang Gao
>
> **摘要:** Automatic speech recognition (ASR) systems struggle with non-normative speech from individuals with impairments caused by conditions like cerebral palsy or structural anomalies. The high acoustic variability and scarcity of training data severely degrade model performance. This work introduces a data-efficient personalization method that quantifies phoneme-level uncertainty to guide fine-tuning. We leverage Monte Carlo Dropout to estimate which phonemes a model finds most difficult and use these estimates for a targeted oversampling strategy. We validate our method on English and German datasets. Crucially, we demonstrate that our model-derived uncertainty strongly correlates with phonemes identified as challenging in an expert clinical logopedic report, marking, to our knowledge, the first work to successfully align model uncertainty with expert assessment of speech difficulty. Our results show that this clinically-validated, uncertainty-guided sampling significantly improves ASR accuracy, delivering a practical framework for personalized and inclusive ASR.
>
---
#### [new 014] Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究语音增强系统的对抗攻击问题，指出现代模型易受语义篡改攻击，并验证了具有随机采样器的扩散模型具备天然鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.21087v1](http://arxiv.org/pdf/2509.21087v1)**

> **作者:** Rostislav Makarov; Lea Schönherr; Timo Gerkmann
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.
>
---
#### [new 015] Objective Evaluation of Prosody and Intelligibility in Speech Synthesis via Conditional Prediction of Discrete Tokens
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文针对语音合成的客观评价问题，提出TTScore框架。通过条件预测离散语音令牌，分别评估语音的可懂度和韵律，实现无参考、细粒度的评价，提升与人类感知的相关性。**

- **链接: [http://arxiv.org/pdf/2509.20485v1](http://arxiv.org/pdf/2509.20485v1)**

> **作者:** Ismail Rasim Ulgen; Zongyang Du; Junchen Lu; Philipp Koehn; Berrak Sisman
>
> **备注:** Under review for IEEE OJSP
>
> **摘要:** Objective evaluation of synthesized speech is critical for advancing speech generation systems, yet existing metrics for intelligibility and prosody remain limited in scope and weakly correlated with human perception. Word Error Rate (WER) provides only a coarse text-based measure of intelligibility, while F0-RMSE and related pitch-based metrics offer a narrow, reference-dependent view of prosody. To address these limitations, we propose TTScore, a targeted and reference-free evaluation framework based on conditional prediction of discrete speech tokens. TTScore employs two sequence-to-sequence predictors conditioned on input text: TTScore-int, which measures intelligibility through content tokens, and TTScore-pro, which evaluates prosody through prosody tokens. For each synthesized utterance, the predictors compute the likelihood of the corresponding token sequences, yielding interpretable scores that capture alignment with intended linguistic content and prosodic structure. Experiments on the SOMOS, VoiceMOS, and TTSArena benchmarks demonstrate that TTScore-int and TTScore-pro provide reliable, aspect-specific evaluation and achieve stronger correlations with human judgments of overall quality than existing intelligibility and prosody-focused metrics.
>
---
#### [new 016] SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SPADE框架，用于高效的大语言模型语音合成（LLM-TTS）。针对模型参数多、延迟高的问题，通过结构化剪枝和自适应知识蒸馏，减少Transformer层数，降低显存占用并提升生成速度，同时保持语音质量。**

- **链接: [http://arxiv.org/pdf/2509.20802v1](http://arxiv.org/pdf/2509.20802v1)**

> **作者:** Tan Dat Nguyen; Jaehun Kim; Ji-Hoon Kim; Shukjae Choi; Youshin Lim; Joon Son Chung
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
>
---
#### [new 017] Why Speech Deepfake Detectors Won't Generalize: The Limits of Detection in an Open World
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文研究语音深度伪造检测任务，指出检测器在开放环境中存在“覆盖债务”问题，导致性能下降。通过分析跨测试框架结果，发现新合成方法和对话场景最难防御，强调应将检测作为多层防御的辅助手段。**

- **链接: [http://arxiv.org/pdf/2509.20405v1](http://arxiv.org/pdf/2509.20405v1)**

> **作者:** Visar Berisha; Prad Kadambi; Isabella Lenz
>
> **摘要:** Speech deepfake detectors are often evaluated on clean, benchmark-style conditions, but deployment occurs in an open world of shifting devices, sampling rates, codecs, environments, and attack families. This creates a ``coverage debt" for AI-based detectors: every new condition multiplies with existing ones, producing data blind spots that grow faster than data can be collected. Because attackers can target these uncovered regions, worst-case performance (not average benchmark scores) determines security. To demonstrate the impact of the coverage debt problem, we analyze results from a recent cross-testing framework. Grouping performance by bona fide domain and spoof release year, two patterns emerge: newer synthesizers erase the legacy artifacts detectors rely on, and conversational speech domains (teleconferencing, interviews, social media) are consistently the hardest to secure. These findings show that detection alone should not be relied upon for high-stakes decisions. Detectors should be treated as auxiliary signals within layered defenses that include provenance, personhood credentials, and policy safeguards.
>
---
## 更新

#### [replaced 001] Application of Audio Fingerprinting Techniques for Real-Time Scalable Speech Retrieval and Speech Clusterization
- **分类: cs.IR; cs.SD; eess.AS; H.3**

- **链接: [http://arxiv.org/pdf/2410.21876v2](http://arxiv.org/pdf/2410.21876v2)**

> **作者:** Kemal Altwlkany; Sead Delalić; Adis Alihodžić; Elmedin Selmanović; Damir Hasić
>
> **备注:** Proceedings of the International Convention MIPRO
>
> **摘要:** Audio fingerprinting techniques have seen great advances in recent years, enabling accurate and fast audio retrieval even in conditions when the queried audio sample has been highly deteriorated or recorded in noisy conditions. Expectedly, most of the existing work is centered around music, with popular music identification services such as Apple's Shazam or Google's Now Playing designed for individual audio recognition on mobile devices. However, the spectral content of speech differs from that of music, necessitating modifications to current audio fingerprinting approaches. This paper offers fresh insights into adapting existing techniques to address the specialized challenge of speech retrieval in telecommunications and cloud communications platforms. The focus is on achieving rapid and accurate audio retrieval in batch processing instead of facilitating single requests, typically on a centralized server. Moreover, the paper demonstrates how this approach can be utilized to support audio clustering based on speech transcripts without undergoing actual speech-to-text conversion. This optimization enables significantly faster processing without the need for GPU computing, a requirement for real-time operation that is typically associated with state-of-the-art speech-to-text tools.
>
---
#### [replaced 002] LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.15299v3](http://arxiv.org/pdf/2412.15299v3)**

> **作者:** Sangmin Lee; Woo-Jin Chung; Hong-Goo Kang
>
> **备注:** Accepted to AAAI 2025 (Oral Presentation)
>
> **摘要:** Building a universal multilingual automatic speech recognition (ASR) model that performs equitably across languages has long been a challenge due to its inherent difficulties. To address this task we introduce a Language-Agnostic Multilingual ASR pipeline through orthography Unification and language-specific Transliteration (LAMA-UT). LAMA-UT operates without any language-specific modules while matching the performance of state-of-the-art models trained on a minimal amount of data. Our pipeline consists of two key steps. First, we utilize a universal transcription generator to unify orthographic features into Romanized form and capture common phonetic characteristics across diverse languages. Second, we utilize a universal converter to transform these universal transcriptions into language-specific ones. In experiments, we demonstrate the effectiveness of our proposed method leveraging universal transcriptions for massively multilingual ASR. Our pipeline achieves a relative error reduction rate of 45% when compared to Whisper and performs comparably to MMS, despite being trained on only 0.1% of Whisper's training data. Furthermore, our pipeline does not rely on any language-specific modules. However, it performs on par with zero-shot ASR approaches which utilize additional language-specific lexicons and language models. We expect this framework to serve as a cornerstone for flexible multilingual ASR systems that are generalizable even to unseen languages.
>
---
#### [replaced 003] MAGE: A Coarse-to-Fine Speech Enhancer with Masked Generative Model
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.19881v2](http://arxiv.org/pdf/2509.19881v2)**

> **作者:** The Hieu Pham; Tan Dat Nguyen; Phuong Thanh Tran; Joon Son Chung; Duc Dung Nguyen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech enhancement remains challenging due to the trade-off between efficiency and perceptual quality. In this paper, we introduce MAGE, a Masked Audio Generative Enhancer that advances generative speech enhancement through a compact and robust design. Unlike prior masked generative models with random masking, MAGE employs a scarcity-aware coarse-to-fine masking strategy that prioritizes frequent tokens in early steps and rare tokens in later refinements, improving efficiency and generalization. We also propose a lightweight corrector module that further stabilizes inference by detecting low-confidence predictions and re-masking them for refinement. Built on BigCodec and finetuned from Qwen2.5-0.5B, MAGE is reduced to 200M parameters through selective layer retention. Experiments on DNS Challenge and noisy LibriSpeech show that MAGE achieves state-of-the-art perceptual quality and significantly reduces word error rate for downstream recognition, outperforming larger baselines. Audio examples are available at https://hieugiaosu.github.io/MAGE/.
>
---
#### [replaced 004] Mixture-of-Experts Framework for Field-of-View Enhanced Signal-Dependent Binauralization of Moving Talkers
- **分类: cs.SD; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.13548v3](http://arxiv.org/pdf/2509.13548v3)**

> **作者:** Manan Mittal; Thomas Deppisch; Joseph Forrer; Chris Le Sueur; Zamir Ben-Hur; David Lou Alon; Daniel D. E. Wong
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** We propose a novel mixture of experts framework for field-of-view enhancement in binaural signal matching. Our approach enables dynamic spatial audio rendering that adapts to continuous talker motion, allowing users to emphasize or suppress sounds from selected directions while preserving natural binaural cues. Unlike traditional methods that rely on explicit direction-of-arrival estimation or operate in the Ambisonics domain, our signal-dependent framework combines multiple binaural filters in an online manner using implicit localization. This allows for real-time tracking and enhancement of moving sound sources, supporting applications such as speech focus, noise reduction, and world-locked audio in augmented and virtual reality. The method is agnostic to array geometry offering a flexible solution for spatial audio capture and personalized playback in next-generation consumer audio devices.
>
---
#### [replaced 005] Speech Language Models for Under-Represented Languages: Insights from Wolof
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15362v2](http://arxiv.org/pdf/2509.15362v2)**

> **作者:** Yaya Sy; Dioula Doucouré; Christophe Cerisara; Irina Illina
>
> **摘要:** We present our journey in training a speech language model for Wolof, an underrepresented language spoken in West Africa, and share key insights. We first emphasize the importance of collecting large-scale, spontaneous, high-quality unsupervised speech data, and show that continued pretraining HuBERT on this dataset outperforms both the base model and African-centric models on ASR. We then integrate this speech encoder into a Wolof LLM to train the first Speech LLM for this language, extending its capabilities to tasks such as speech translation. Furthermore, we explore training the Speech LLM to perform multi-step Chain-of-Thought before transcribing or translating. Our results show that the Speech LLM not only improves speech recognition but also performs well in speech translation. The models and the code will be openly shared.
>
---
#### [replaced 006] MNV-17: A High-Quality Performative Mandarin Dataset for Nonverbal Vocalization Recognition in Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.18196v2](http://arxiv.org/pdf/2509.18196v2)**

> **作者:** Jialong Mai; Jinxin Ji; Xiaofen Xing; Chen Yang; Weidong Chen; Jingyuan Xing; Xiangmin Xu
>
> **备注:** Official dataset available at: https://github.com/yongaifadian1/MNV-17. Submitted to ICASSP 2026
>
> **摘要:** Mainstream Automatic Speech Recognition (ASR) systems excel at transcribing lexical content, but largely fail to recognize nonverbal vocalizations (NVs) embedded in speech, such as sighs, laughs, and coughs. This capability is important for a comprehensive understanding of human communication, as NVs convey crucial emotional and intentional cues. Progress in NV-aware ASR has been hindered by the lack of high-quality, well-annotated datasets. To address this gap, we introduce MNV-17, a 7.55-hour performative Mandarin speech dataset. Unlike most existing corpora that rely on model-based detection, MNV-17's performative nature ensures high-fidelity, clearly articulated NV instances. To the best of our knowledge, MNV-17 provides the most extensive set of nonverbal vocalization categories, comprising 17 distinct and well-balanced classes of common NVs. We benchmarked MNV-17 on four mainstream ASR architectures, evaluating their joint performance on semantic transcription and NV classification. The dataset and the pretrained model checkpoints will be made publicly available to facilitate future research in expressive ASR.
>
---
#### [replaced 007] Neural Audio Codecs for Prompt-Driven Universal Sound Separation
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11717v3](http://arxiv.org/pdf/2509.11717v3)**

> **作者:** Adhiraj Banerjee; Vipul Arora
>
> **备注:** main content- 10 pages, total - 23 pages, 1 figure, pre-print, under review
>
> **摘要:** Text-guided sound separation supports flexible audio editing across media and assistive applications, but existing models like AudioSep are too compute-heavy for edge deployment. Neural audio codec (NAC) models such as CodecFormer and SDCodec are compute-efficient but limited to fixed-class separation. We introduce CodecSep, the first NAC-based model for on-device universal, text-driven separation. CodecSep combines DAC compression with a Transformer masker modulated by CLAP-derived FiLM parameters. Across six open-domain benchmarks under matched training/prompt protocols, \textbf{CodecSep} surpasses \textbf{AudioSep} in separation fidelity (SI-SDR) while remaining competitive in perceptual quality (ViSQOL) and matching or exceeding fixed-stem baselines (TDANet, CodecFormer, SDCodec). In code-stream deployments, it needs just 1.35~GMACs end-to-end -- approximately $54\times$ less compute ($25\times$ architecture-only) than spectrogram-domain separators like AudioSep -- while remaining fully bitstream-compatible.
>
---
#### [replaced 008] Interpretable Embeddings of Speech Enhance and Explain Brain Encoding Performance of Audio Models
- **分类: q-bio.NC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16080v2](http://arxiv.org/pdf/2507.16080v2)**

> **作者:** Riki Shimizu; Richard J. Antonello; Chandan Singh; Nima Mesgarani
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Speech foundation models (SFMs) are increasingly hailed as powerful computational models of human speech perception. However, since their representations are inherently black-box, it remains unclear what drives their alignment with brain responses. To remedy this, we built linear encoding models from six interpretable feature families: mel-spectrogram, Gabor filter bank features, speech presence, phonetic, syntactic, and semantic features, and contextualized embeddings from three state-of-the-art SFMs (Whisper, HuBERT, WavLM), quantifying electrocorticography (ECoG) response variance shared between feature classes. Variance-partitioning analyses revealed several key insights: First, the SFMs' alignment with the brain can be mostly explained by their ability to learn and encode simple interpretable speech features. Second, SFMs exhibit a systematic trade-off between encoding of brain-relevant low-level and high-level features across layers. Finally, our results show that SFMs learn brain-relevant semantics which cannot be explained by lower-level speech features, with this capacity increasing with model size and context length. Together, our findings suggest a principled approach to build more interpretable, accurate, and efficient encoding models of the brain by augmenting SFM embeddings with interpretable features.
>
---
#### [replaced 009] Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute
- **分类: cs.LG; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.15882v2](http://arxiv.org/pdf/2506.15882v2)**

> **作者:** Sheng Liu; Tianlang Chen; Pan Lu; Haotian Ye; Yizheng Chen; Lei Xing; James Zou
>
> **备注:** 18 pages, 5 figures, Project website: https://shengliu66.github.io/fractreason/
>
> **摘要:** Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent steering vector associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models.
>
---
#### [replaced 010] Facilitating Personalized TTS for Dysarthric Speakers Using Knowledge Anchoring and Curriculum Learning
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2508.10412v2](http://arxiv.org/pdf/2508.10412v2)**

> **作者:** Yejin Jeon; Solee Im; Youngjae Kim; Gary Geunbae Lee
>
> **备注:** Interspeech 2025
>
> **摘要:** Dysarthric speakers experience substantial communication challenges due to impaired motor control of the speech apparatus, which leads to reduced speech intelligibility. This creates significant obstacles in dataset curation since actual recording of long, articulate sentences for the objective of training personalized TTS models becomes infeasible. Thus, the limited availability of audio data, in addition to the articulation errors that are present within the audio, complicates personalized speech synthesis for target dysarthric speaker adaptation. To address this, we frame the issue as a domain transfer task and introduce a knowledge anchoring framework that leverages a teacher-student model, enhanced by curriculum learning through audio augmentation. Experimental results show that the proposed zero-shot multi-speaker TTS model effectively generates synthetic speech with markedly reduced articulation errors and high speaker fidelity, while maintaining prosodic naturalness.
>
---
#### [replaced 011] SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for South-East Asian
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.19865v2](http://arxiv.org/pdf/2509.19865v2)**

> **作者:** Jinyang Wu; Nana Hou; Zihan Pan; Qiquan Zhang; Sailor Hardik Bhupendra; Soumik Mondal
>
> **备注:** 5 pages, 1 figure, 3 tables
>
> **摘要:** The rapid growth of the digital economy in South-East Asia (SEA) has amplified the risks of audio deepfakes, yet current datasets cover SEA languages only sparsely, leaving models poorly equipped to handle this critical region. This omission is critical: detection models trained on high-resource languages collapse when applied to SEA, due to mismatches in synthesis quality, language-specific characteristics, and data scarcity. To close this gap, we present SEA-Spoof, the first large-scale Audio Deepfake Detection (ADD) dataset especially for SEA languages. SEA-Spoof spans 300+ hours of paired real and spoof speech across Tamil, Hindi, Thai, Indonesian, Malay, and Vietnamese. Spoof samples are generated from a diverse mix of state-of-the-art open-source and commercial systems, capturing wide variability in style and fidelity. Benchmarking state-of-the-art detection models reveals severe cross-lingual degradation, but fine-tuning on SEA-Spoof dramatically restores performance across languages and synthesis sources. These results highlight the urgent need for SEA-focused research and establish SEA-Spoof as a foundation for developing robust, cross-lingual, and fraud-resilient detection systems.
>
---
#### [replaced 012] Scaling Rich Style-Prompted Text-to-Speech Datasets
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.04713v2](http://arxiv.org/pdf/2503.04713v2)**

> **作者:** Anuj Diwan; Zhisheng Zheng; David Harwath; Eunsol Choi
>
> **备注:** EMNLP 2025
>
> **摘要:** We introduce Paralinguistic Speech Captions (ParaSpeechCaps), a large-scale dataset that annotates speech utterances with rich style captions. While rich abstract tags (e.g. guttural, nasal, pained) have been explored in small-scale human-annotated datasets, existing large-scale datasets only cover basic tags (e.g. low-pitched, slow, loud). We combine off-the-shelf text and speech embedders, classifiers and an audio language model to automatically scale rich tag annotations for the first time. ParaSpeechCaps covers a total of 59 style tags, including both speaker-level intrinsic tags and utterance-level situational tags. It consists of 342 hours of human-labelled data (PSC-Base) and 2427 hours of automatically annotated data (PSC-Scaled). We finetune Parler-TTS, an open-source style-prompted TTS model, on ParaSpeechCaps, and achieve improved style consistency (+7.9% Consistency MOS) and speech quality (+15.5% Naturalness MOS) over the best performing baseline that combines existing rich style tag datasets. We ablate several of our dataset design choices to lay the foundation for future work in this space. Our dataset, models and code are released at https://github.com/ajd12342/paraspeechcaps .
>
---
#### [replaced 013] On the Language and Gender Biases in PSTN, VoIP and Neural Audio Codecs
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.02545v2](http://arxiv.org/pdf/2506.02545v2)**

> **作者:** Kemal Altwlkany; Amar Kuric; Emanuel Lacic
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** In recent years, there has been a growing focus on fairness and inclusivity within speech technology, particularly in areas such as automatic speech recognition and speech sentiment analysis. When audio is transcoded prior to processing, as is the case in streaming or real-time applications, any inherent bias in the coding mechanism may result in disparities. This not only affects user experience but can also have broader societal implications by perpetuating stereotypes and exclusion. Thus, it is important that audio coding mechanisms are unbiased. In this work, we contribute towards the scarce research with respect to language and gender biases of audio codecs. By analyzing the speech quality of over 2 million multilingual audio files after transcoding through a representative subset of codecs (PSTN, VoIP and neural), our results indicate that PSTN codecs are strongly biased in terms of gender and that neural codecs introduce language biases.
>
---
