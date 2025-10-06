# 音频 cs.SD;  eess.SP

- **最新发布 19 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] WavInWav: Time-domain Speech Hiding via Invertible Neural Network
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于语音隐写任务，旨在解决隐藏语音信息时恢复质量不佳的问题。通过引入可逆神经网络和时间频域损失，提升隐写音频的可逆性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.02915v1](http://arxiv.org/pdf/2510.02915v1)**

> **作者:** Wei Fan; Kejiang Chen; Xiangkun Wang; Weiming Zhang; Nenghai Yu
>
> **备注:** 13 pages, 5 figures, project page: https://cyberrrange.github.io/project/wavinwav
>
> **摘要:** Data hiding is essential for secure communication across digital media, and recent advances in Deep Neural Networks (DNNs) provide enhanced methods for embedding secret information effectively. However, previous audio hiding methods often result in unsatisfactory quality when recovering secret audio, due to their inherent limitations in the modeling of time-frequency relationships. In this paper, we explore these limitations and introduce a new DNN-based approach. We use a flow-based invertible neural network to establish a direct link between stego audio, cover audio, and secret audio, enhancing the reversibility of embedding and extracting messages. To address common issues from time-frequency transformations that degrade secret audio quality during recovery, we implement a time-frequency loss on the time-domain signal. This approach not only retains the benefits of time-frequency constraints but also enhances the reversibility of message recovery, which is vital for practical applications. We also add an encryption technique to protect the hidden data from unauthorized access. Experimental results on the VCTK and LibriSpeech datasets demonstrate that our method outperforms previous approaches in terms of subjective and objective metrics and exhibits robustness to various types of noise, suggesting its utility in targeted secure communication scenarios.
>
---
#### [new 002] Latent Multi-view Learning for Robust Environmental Sound Representations
- **分类: cs.SD**

- **简介: 该论文属于环境声音表示学习任务，旨在解决SSL方法融合不足的问题。提出多视角框架，结合对比学习与生成模型，提升声音源和设备分类性能。**

- **链接: [http://arxiv.org/pdf/2510.02500v1](http://arxiv.org/pdf/2510.02500v1)**

> **作者:** Sivan Sing; Julia Wilkins; Magdalena Fuentes; Juan Pablo Bello
>
> **备注:** Accepted to DCASE 2025 Workshop. 4+1 pages, 2 figures, 2 tables
>
> **摘要:** Self-supervised learning (SSL) approaches, such as contrastive and generative methods, have advanced environmental sound representation learning using unlabeled data. However, how these approaches can complement each other within a unified framework remains relatively underexplored. In this work, we propose a multi-view learning framework that integrates contrastive principles into a generative pipeline to capture sound source and device information. Our method encodes compressed audio latents into view-specific and view-common subspaces, guided by two self-supervised objectives: contrastive learning for targeted information flow between subspaces, and reconstruction for overall information preservation. We evaluate our method on an urban sound sensor network dataset for sound source and sensor classification, demonstrating improved downstream performance over traditional SSL techniques. Additionally, we investigate the model's potential to disentangle environmental sound attributes within the structured latent space under varied training configurations.
>
---
#### [new 003] Linear RNNs for autoregressive generation of long music samples
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频生成任务，旨在解决长序列自回归生成难题。通过改进线性RNN结构，提升模型在长时间音频上的表现。**

- **链接: [http://arxiv.org/pdf/2510.02401v1](http://arxiv.org/pdf/2510.02401v1)**

> **作者:** Konrad Szewczyk; Daniel Gallo Fernández; James Townsend
>
> **摘要:** Directly learning to generate audio waveforms in an autoregressive manner is a challenging task, due to the length of the raw sequences and the existence of important structure on many different timescales. Traditional approaches based on recurrent neural networks, as well as causal convolutions and self-attention, have only had limited success on this task. However, recent work has shown that deep state space models, also referred to as linear RNNs, can be highly efficient in this context. In this work, we push the boundaries of linear RNNs applied to raw audio modeling, investigating the effects of different architectural choices and using context-parallelism to enable training on sequences up to one minute (1M tokens) in length. We present a model, HarmonicRNN, which attains state of the art log-likelihoods and perceptual metrics on small-scale datasets.
>
---
#### [new 004] SALSA-V: Shortcut-Augmented Long-form Synchronized Audio from Videos
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出SALSA-V，解决视频转高质量长音频的问题。通过掩码扩散和快捷损失，实现高同步、高保真音频生成，适用于专业音效合成。**

- **链接: [http://arxiv.org/pdf/2510.02916v1](http://arxiv.org/pdf/2510.02916v1)**

> **作者:** Amir Dellali; Luca A. Lanzendörfer; Florian Grötschla; Roger Wattenhofer
>
> **摘要:** We propose SALSA-V, a multimodal video-to-audio generation model capable of synthesizing highly synchronized, high-fidelity long-form audio from silent video content. Our approach introduces a masked diffusion objective, enabling audio-conditioned generation and the seamless synthesis of audio sequences of unconstrained length. Additionally, by integrating a shortcut loss into our training process, we achieve rapid generation of high-quality audio samples in as few as eight sampling steps, paving the way for near-real-time applications without requiring dedicated fine-tuning or retraining. We demonstrate that SALSA-V significantly outperforms existing state-of-the-art methods in both audiovisual alignment and synchronization with video content in quantitative evaluation and a human listening study. Furthermore, our use of random masking during training enables our model to match spectral characteristics of reference audio samples, broadening its applicability to professional audio synthesis tasks such as Foley generation and sound design.
>
---
#### [new 005] Forensic Similarity for Speech Deepfakes
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在判断两个音频片段是否具有相同的伪造痕迹。通过构建深度学习系统实现这一目标。**

- **链接: [http://arxiv.org/pdf/2510.02864v1](http://arxiv.org/pdf/2510.02864v1)**

> **作者:** Viola Negroni; Davide Salvi; Daniele Ugo Leonzio; Paolo Bestagini; Stefano Tubaro
>
> **备注:** Submitted @ IEEE OJSP
>
> **摘要:** In this paper, we introduce a digital audio forensics approach called Forensic Similarity for Speech Deepfakes, which determines whether two audio segments contain the same forensic traces or not. Our work is inspired by prior work in the image domain on forensic similarity, which proved strong generalization capabilities against unknown forensic traces, without requiring prior knowledge of them at training time. To achieve this in the audio setting, we propose a two-part deep-learning system composed of a feature extractor based on a speech deepfake detector backbone and a shallow neural network, referred to as the similarity network. This system maps pairs of audio segments to a score indicating whether they contain the same or different forensic traces. We evaluate the system on the emerging task of source verification, highlighting its ability to identify whether two samples originate from the same generative model. Additionally, we assess its applicability to splicing detection as a complementary use case. Experiments show that the method generalizes to a wide range of forensic traces, including previously unseen ones, illustrating its flexibility and practical value in digital audio forensics.
>
---
#### [new 006] TART: A Comprehensive Tool for Technique-Aware Audio-to-Tab Guitar Transcription
- **分类: cs.SD**

- **简介: 该论文属于音乐转录任务，解决吉他音频转谱的难题。针对现有系统无法准确识别技巧和弦位问题，提出四阶段端到端框架，生成精确的吉他谱。**

- **链接: [http://arxiv.org/pdf/2510.02597v1](http://arxiv.org/pdf/2510.02597v1)**

> **作者:** Akshaj Gupta; Andrea Guzman; Anagha Badriprasad; Hwi Joo Park; Upasana Puranik; Robin Netzorg; Jiachen Lian; Gopala Krishna Anumanchipalli
>
> **摘要:** Automatic Music Transcription (AMT) has advanced significantly for the piano, but transcription for the guitar remains limited due to several key challenges. Existing systems fail to detect and annotate expressive techniques (e.g., slides, bends, percussive hits) and incorrectly map notes to the wrong string and fret combination in the generated tablature. Furthermore, prior models are typically trained on small, isolated datasets, limiting their generalizability to real-world guitar recordings. To overcome these limitations, we propose a four-stage end-to-end pipeline that produces detailed guitar tablature directly from audio. Our system consists of (1) Audio-to-MIDI pitch conversion through a piano transcription model adapted to guitar datasets; (2) MLP-based expressive technique classification; (3) Transformer-based string and fret assignment; and (4) LSTM-based tablature generation. To the best of our knowledge, this framework is the first to generate detailed tablature with accurate fingerings and expressive labels from guitar audio.
>
---
#### [new 007] Flamed-TTS: Flow Matching Attention-Free Models for Efficient Generating and Dynamic Pacing Zero-shot Text-to-Speech
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于零样本文本到语音合成任务，旨在解决生成不稳定、计算成本高和时间多样性不足的问题。提出Flamed-TTS框架，结合离散与连续表示，提升合成质量与效率。**

- **链接: [http://arxiv.org/pdf/2510.02848v1](http://arxiv.org/pdf/2510.02848v1)**

> **作者:** Hieu-Nghia Huynh-Nguyen; Huynh Nguyen Dang; Ngoc-Son Nguyen; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) has recently advanced significantly, enabling models to synthesize speech from text using short, limited-context prompts. These prompts serve as voice exemplars, allowing the model to mimic speaker identity, prosody, and other traits without extensive speaker-specific data. Although recent approaches incorporating language models, diffusion, and flow matching have proven their effectiveness in zero-shot TTS, they still encounter challenges such as unreliable synthesis caused by token repetition or unexpected content transfer, along with slow inference and substantial computational overhead. Moreover, temporal diversity-crucial for enhancing the naturalness of synthesized speech-remains largely underexplored. To address these challenges, we propose Flamed-TTS, a novel zero-shot TTS framework that emphasizes low computational cost, low latency, and high speech fidelity alongside rich temporal diversity. To achieve this, we reformulate the flow matching training paradigm and incorporate both discrete and continuous representations corresponding to different attributes of speech. Experimental results demonstrate that Flamed-TTS surpasses state-of-the-art models in terms of intelligibility, naturalness, speaker similarity, acoustic characteristics preservation, and dynamic pace. Notably, Flamed-TTS achieves the best WER of 4% compared to the leading zero-shot TTS baselines, while maintaining low latency in inference and high fidelity in generated speech. Code and audio samples are available at our demo page https://flamed-tts.github.io.
>
---
#### [new 008] Accelerated Convolutive Transfer Function-Based Multichannel NMF Using Iterative Source Steering
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音分离任务，解决CTF-MNMF计算复杂度高的问题，提出融合ISS的高效方法，在保持性能的同时降低计算成本。**

- **链接: [http://arxiv.org/pdf/2510.02382v1](http://arxiv.org/pdf/2510.02382v1)**

> **作者:** Xuemai Xie; Xianrui Wang; Liyuan Zhang; Yichen Yang; Shoji Makino
>
> **摘要:** Among numerous blind source separation (BSS) methods, convolutive transfer function-based multichannel non-negative matrix factorization (CTF-MNMF) has demonstrated strong performance in highly reverberant environments by modeling multi-frame correlations of delayed source signals. However, its practical deployment is hindered by the high computational cost associated with the iterative projection (IP) update rule, which requires matrix inversion for each source. To address this issue, we propose an efficient variant of CTF-MNMF that integrates iterative source steering (ISS), a matrix inversion-free update rule for separation filters. Experimental results show that the proposed method achieves comparable or superior separation performance to the original CTF-MNMF, while significantly reducing the computational complexity.
>
---
#### [new 009] AudioToolAgent: An Agentic Framework for Audio-Language Models
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型任务，解决LALMs缺乏多步骤推理和工具调用的问题，提出AudioToolAgent框架提升音频问答性能。**

- **链接: [http://arxiv.org/pdf/2510.02995v1](http://arxiv.org/pdf/2510.02995v1)**

> **作者:** Gijs Wijngaard; Elia Formisano; Michel Dumontier
>
> **摘要:** Large Audio-Language Models (LALMs) perform well on audio understanding tasks but lack multi-step reasoning and tool-calling found in recent Large Language Models (LLMs). This paper presents AudioToolAgent, a framework that coordinates audio-language models as tools via a central LLM agent that accesses tool adapters for audio question answering and speech-to-text. The agent selects tools, asks follow-up questions, and compares outputs for verification. Experiments with MMAU, MMAR, and MMAU-Pro show state-of-the-art accuracy: up to 74.10% on MMAU, 68.80% on MMAR, and 57.96% on MMAU-Pro. Monte Carlo sampling for shapley values across 374 configurations identifies effective agent-tool combinations. The modular design allows integration of new tools and eliminates the use of data and training costs. Code and reproduction materials are available at: github.com/GLJS/AudioToolAgent
>
---
#### [new 010] Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于文本到音视频生成任务，旨在解决文本条件下的模态干扰和跨模态交互问题。提出HVGC框架和BridgeDiT模型，实现音视频同步与对齐。**

- **链接: [http://arxiv.org/pdf/2510.03117v1](http://arxiv.org/pdf/2510.03117v1)**

> **作者:** Kaisi Guan; Xihua Wang; Zhengfeng Lai; Xin Cheng; Peng Zhang; XiaoJiang Liu; Ruihua Song; Meng Cao
>
> **摘要:** This study focuses on a challenging yet promising task, Text-to-Sounding-Video (T2SV) generation, which aims to generate a video with synchronized audio from text conditions, meanwhile ensuring both modalities are aligned with text. Despite progress in joint audio-video training, two critical challenges still remain unaddressed: (1) a single, shared text caption where the text for video is equal to the text for audio often creates modal interference, confusing the pretrained backbones, and (2) the optimal mechanism for cross-modal feature interaction remains unclear. To address these challenges, we first propose the Hierarchical Visual-Grounded Captioning (HVGC) framework that generates pairs of disentangled captions, a video caption, and an audio caption, eliminating interference at the conditioning stage. Based on HVGC, we further introduce BridgeDiT, a novel dual-tower diffusion transformer, which employs a Dual CrossAttention (DCA) mechanism that acts as a robust ``bridge" to enable a symmetric, bidirectional exchange of information, achieving both semantic and temporal synchronization. Extensive experiments on three benchmark datasets, supported by human evaluations, demonstrate that our method achieves state-of-the-art results on most metrics. Comprehensive ablation studies further validate the effectiveness of our contributions, offering key insights for the future T2SV task. All the codes and checkpoints will be publicly released.
>
---
#### [new 011] WEE-Therapy: A Mixture of Weak Encoders Framework for Psychological Counseling Dialogue Analysis
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于心理辅导对话分析任务，旨在解决现有模型难以捕捉情绪和专业技巧的问题。提出WEE-Therapy框架，结合多种弱编码器提升性能。**

- **链接: [http://arxiv.org/pdf/2510.02320v1](http://arxiv.org/pdf/2510.02320v1)**

> **作者:** Yongqi Kang; Yong Zhao
>
> **备注:** 5 pages
>
> **摘要:** The advancement of computational psychology requires AI tools capable of deeply understanding counseling dialogues. Existing audio language models (AudioLLMs) often rely on single speech encoders pre-trained on general data, struggling to capture domain-specific features like complex emotions and professional techniques. To address this, we propose WEE-Therapy, a multi-task AudioLLM incorporating a Weak Encoder Ensemble (WEE) mechanism. This supplements a powerful base encoder with a pool of lightweight, specialized encoders. A novel dual-routing strategy combines stable, data-independent domain knowledge with dynamic, data-dependent expert selection. Evaluated on emotion recognition, technique classification, risk detection, and summarization, WEE-Therapy achieves significant performance gains across all tasks with minimal parameter overhead, demonstrating strong potential for AI-assisted clinical analysis.
>
---
#### [new 012] Multi-Source Position and Direction-of-Arrival Estimation Based on Euclidean Distance Matrices
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于多声源定位与波达方向估计任务，解决传统SRP方法计算复杂的问题，通过欧几里得距离矩阵降低计算成本并提高精度。**

- **链接: [http://arxiv.org/pdf/2510.02556v1](http://arxiv.org/pdf/2510.02556v1)**

> **作者:** Klaus Brümann; Simon Doclo
>
> **备注:** 13 pages, 6 figures, submitted to IEEE Transactions on Audio, Speech and Language Processing (awaiting review)
>
> **摘要:** A popular method to estimate the positions or directions-of-arrival (DOAs) of multiple sound sources using an array of microphones is based on steered-response power (SRP) beamforming. For a three-dimensional scenario, SRP-based methods need to jointly optimize three continuous variables for position estimation or two continuous variables for DOA estimation, which can be computationally expensive. In this paper, we propose novel methods for multi-source position and DOA estimation by exploiting properties of Euclidean distance matrices (EDMs) and their respective Gram matrices. In the proposed multi-source position estimation method only a single continuous variable, representing the distance between each source and a reference microphone, needs to be optimized. For each source, the optimal continuous distance variable and set of candidate time-difference of arrival (TDOA) estimates are determined by minimizing a cost function that is defined using the eigenvalues of the Gram matrix. The estimated relative source positions are then mapped to estimated absolute source positions by solving an orthogonal Procrustes problem for each source. The proposed multi-source DOA estimation method entirely eliminates the need for continuous variable optimization by defining a relative coordinate system per source such that one of its coordinate axes is aligned with the respective source DOA. The optimal set of candidate TDOA estimates is determined by minimizing a cost function that is defined using the eigenvalues of a rank-reduced Gram matrix. The computational cost of the proposed EDM-based methods is significantly reduced compared to the SRP-based methods. Experimental results for different source and microphone configurations show that the proposed EDM-based method consistently outperforms the SRP-based method in terms of two-source position and DOA estimation accuracy.
>
---
#### [new 013] Listening or Reading? Evaluating Speech Awareness in Chain-of-Thought Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究语音到文本翻译任务，旨在解决错误传播和语音信息利用不足的问题。通过分析CoT方法，发现其主要依赖文本而非语音，提出训练改进以增强语音利用。**

- **链接: [http://arxiv.org/pdf/2510.03115v1](http://arxiv.org/pdf/2510.03115v1)**

> **作者:** Jacobo Romero-Díaz; Gerard I. Gállego; Oriol Pareras; Federico Costa; Javier Hernando; Cristina España-Bonet
>
> **摘要:** Speech-to-Text Translation (S2TT) systems built from Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) modules face two major limitations: error propagation and the inability to exploit prosodic or other acoustic cues. Chain-of-Thought (CoT) prompting has recently been introduced, with the expectation that jointly accessing speech and transcription will overcome these issues. Analyzing CoT through attribution methods, robustness evaluations with corrupted transcripts, and prosody-awareness, we find that it largely mirrors cascaded behavior, relying mainly on transcripts while barely leveraging speech. Simple training interventions, such as adding Direct S2TT data or noisy transcript injection, enhance robustness and increase speech attribution. These findings challenge the assumed advantages of CoT and highlight the need for architectures that explicitly integrate acoustic information into translation.
>
---
#### [new 014] A UAV-Based VNIR Hyperspectral Benchmark Dataset for Landmine and UXO Detection
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于地雷与未爆弹检测任务，旨在提供高精度的UAV VNIR hyperspectral数据集，解决开放数据不足的问题，并通过多传感器融合提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.02700v1](http://arxiv.org/pdf/2510.02700v1)**

> **作者:** Sagar Lekhak; Emmett J. Ientilucci; Jasper Baur; Susmita Ghosh
>
> **备注:** This work has been accepted and will be presented at the Indian Geoscience and Remote Sensing Symposium (InGARSS) 2025 in India and will appear in the IEEE InGARSS 2025 Proceedings
>
> **摘要:** This paper introduces a novel benchmark dataset of Visible and Near-Infrared (VNIR) hyperspectral imagery acquired via an unmanned aerial vehicle (UAV) platform for landmine and unexploded ordnance (UXO) detection research. The dataset was collected over a controlled test field seeded with 143 realistic surrogate landmine and UXO targets, including surface, partially buried, and fully buried configurations. Data acquisition was performed using a Headwall Nano-Hyperspec sensor mounted on a multi-sensor drone platform, flown at an altitude of approximately 20.6 m, capturing 270 contiguous spectral bands spanning 398-1002 nm. Radiometric calibration, orthorectification, and mosaicking were performed followed by reflectance retrieval using a two-point Empirical Line Method (ELM), with reference spectra acquired using an SVC spectroradiometer. Cross-validation against six reference objects yielded RMSE values below 1.0 and SAM values between 1 and 6 degrees in the 400-900 nm range, demonstrating high spectral fidelity. The dataset is released alongside raw radiance cubes, GCP/AeroPoint data, and reference spectra to support reproducible research. This contribution fills a critical gap in open-access UAV-based hyperspectral data for landmine detection and offers a multi-sensor benchmark when combined with previously published drone-based electromagnetic induction (EMI) data from the same test field.
>
---
#### [new 015] When Voice Matters: Evidence of Gender Disparity in Positional Bias of SpeechLLMs
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音语言模型公平性研究，解决MCQA基准在语音领域中的性别与位置偏见问题。通过实验分析温度、提示设计及语音性别对偏见的影响，发现女性语音偏差更显著，提出需改进评估方法。**

- **链接: [http://arxiv.org/pdf/2510.02398v1](http://arxiv.org/pdf/2510.02398v1)**

> **作者:** Shree Harsha Bokkahalli Satish; Gustav Eje Henter; Éva Székely
>
> **备注:** 16 pages, 5 figures, To Appear in SPECOM 2025
>
> **摘要:** The rapid development of SpeechLLM-based conversational AI systems has created a need for robustly benchmarking these efforts, including aspects of fairness and bias. At present, such benchmarks typically rely on multiple choice question answering (MCQA). In this paper, we present the first token-level probabilistic evaluation and response-based study of several issues affecting the use of MCQA in SpeechLLM benchmarking: 1) we examine how model temperature and prompt design affect gender and positional bias on an MCQA gender-bias benchmark; 2) we examine how these biases are affected by the gender of the input voice; and 3) we study to what extent observed trends carry over to a second gender-bias benchmark. Our results show that concerns about positional bias from the text domain are equally valid in the speech domain. We also find the effect to be stronger for female voices than for male voices. To our knowledge, this is the first study to isolate positional bias effects in SpeechLLM-based gender-bias benchmarks. We conclude that current MCQA benchmarks do not account for speech-based bias and alternative strategies are needed to ensure fairness towards all users.
>
---
#### [new 016] CVSM: Contrastive Vocal Similarity Modeling
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出CVSM方法，用于音乐和人声相似性建模，解决音频表示学习问题。通过对比学习提升人声与音乐混合的相似性，有效提升下游任务表现。**

- **链接: [http://arxiv.org/pdf/2510.03025v1](http://arxiv.org/pdf/2510.03025v1)**

> **作者:** Christos Garoufis; Athanasia Zlatintsi; Petros Maragos
>
> **备注:** 13 pages, 3 tables, 8 figures. Submitted article at IEEE Trans. on Audio, Speech and Language Proc. (pre-print version)
>
> **摘要:** The availability of large, unlabeled datasets across various domains has contributed to the development of a plethora of methods that learn representations for multiple target (downstream) tasks through self-supervised pre-training. In this work, we introduce CVSM (Contrastive Vocal Similarity Modeling), a contrastive self-supervised procedure for music signal representation learning in the audio domain that can be utilized for musical and vocal similarity modeling. Our method operates under a contrastive framework, maximizing the similarity between vocal excerpts and musical mixtures containing the same vocals; we devise both a label-informed protocol, leveraging artist identity information to sample the contrastive pairs, and a label-agnostic scheme, involving artificial mixture creation from randomly sampled vocal and accompaniment excerpts, which are paired with vocals from the same audio segment. We evaluate our proposed method in measuring vocal similarity both objectively, through linear probing on a suite of appropriate downstream tasks, and subjectively, via conducting a user study consisting of pairwise comparisons between different models in a recommendation-by-query setting. Our results indicate that the representations learned through CVSM are effective in musical and vocal similarity modeling, outperforming numerous baselines across both isolated vocals and complete musical mixtures. Moreover, while the availability of artist identity labels during pre-training leads to overall more consistent performance both in the evaluated downstream tasks and the user study, a label-agnostic CVSM variant incorporating hybrid pre-training with real and artificial mixtures achieves comparable performance to the label-informed one in artist identification and perceived vocal similarity.
>
---
#### [new 017] Learning a distance measure from the information-estimation geometry of data
- **分类: eess.IV; cs.CV; cs.IT; eess.SP; math.IT; stat.ML**

- **简介: 该论文提出一种基于信息估计几何的距离度量方法，用于图像质量评估，解决如何有效衡量信号间差异的问题。**

- **链接: [http://arxiv.org/pdf/2510.02514v1](http://arxiv.org/pdf/2510.02514v1)**

> **作者:** Guy Ohayon; Pierre-Etienne H. Fiquet; Florentin Guth; Jona Ballé; Eero P. Simoncelli
>
> **备注:** Code available at https://github.com/ohayonguy/information-estimation-metric
>
> **摘要:** We introduce the Information-Estimation Metric (IEM), a novel form of distance function derived from an underlying continuous probability density over a domain of signals. The IEM is rooted in a fundamental relationship between information theory and estimation theory, which links the log-probability of a signal with the errors of an optimal denoiser, applied to noisy observations of the signal. In particular, the IEM between a pair of signals is obtained by comparing their denoising error vectors over a range of noise amplitudes. Geometrically, this amounts to comparing the score vector fields of the blurred density around the signals over a range of blur levels. We prove that the IEM is a valid global metric and derive a closed-form expression for its local second-order approximation, which yields a Riemannian metric. For Gaussian-distributed signals, the IEM coincides with the Mahalanobis distance. But for more complex distributions, it adapts, both locally and globally, to the geometry of the distribution. In practice, the IEM can be computed using a learned denoiser (analogous to generative diffusion models) and solving a one-dimensional integral. To demonstrate the value of our framework, we learn an IEM on the ImageNet database. Experiments show that this IEM is competitive with or outperforms state-of-the-art supervised image quality metrics in predicting human perceptual judgments.
>
---
#### [new 018] STSM-FiLM: A FiLM-Conditioned Neural Architecture for Time-Scale Modification of Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音时长修改任务，旨在解决传统方法在极端情况下的失真问题。提出STSM-FILM模型，利用FiLM机制实现更灵活的时长调整。**

- **链接: [http://arxiv.org/pdf/2510.02672v1](http://arxiv.org/pdf/2510.02672v1)**

> **作者:** Dyah A. M. G. Wisnu; Ryandhimas E. Zezario; Stefano Rini; Fo-Rui Li; Yan-Tsung Peng; Hsin-Min Wang; Yu Tsao
>
> **摘要:** Time-Scale Modification (TSM) of speech aims to alter the playback rate of audio without changing its pitch. While classical methods like Waveform Similarity-based Overlap-Add (WSOLA) provide strong baselines, they often introduce artifacts under non-stationary or extreme stretching conditions. We propose STSM-FILM - a fully neural architecture that incorporates Feature-Wise Linear Modulation (FiLM) to condition the model on a continuous speed factor. By supervising the network using WSOLA-generated outputs, STSM-FILM learns to mimic alignment and synthesis behaviors while benefiting from representations learned through deep learning. We explore four encoder--decoder variants: STFT-HiFiGAN, WavLM-HiFiGAN, Whisper-HiFiGAN, and EnCodec, and demonstrate that STSM-FILM is capable of producing perceptually consistent outputs across a wide range of time-scaling factors. Overall, our results demonstrate the potential of FiLM-based conditioning to improve the generalization and flexibility of neural TSM models.
>
---
#### [new 019] Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，旨在比较CoT与直接提示方法的效果。通过增加数据量，发现直接方法在大规模数据下表现更优。**

- **链接: [http://arxiv.org/pdf/2510.03093v1](http://arxiv.org/pdf/2510.03093v1)**

> **作者:** Oriol Pareras; Gerard I. Gállego; Federico Costa; Cristina España-Bonet; Javier Hernando
>
> **摘要:** Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
>
---
## 更新

#### [replaced 001] JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17568v2](http://arxiv.org/pdf/2505.17568v2)**

> **作者:** Zifan Peng; Yule Liu; Zhen Sun; Mingchen Li; Zeren Luo; Jingyi Zheng; Wenhan Dong; Xinlei He; Xuechao Wang; Yingjie Xue; Shengmin Xu; Xinyi Huang
>
> **摘要:** Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, a comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 11,316 text samples and 245,355 audio samples with over 1,000 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and architecture. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.
>
---
#### [replaced 002] YOLO-Based Defect Detection for Metal Sheets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP; 68T45, 68T07; I.2.10; I.4.7; I.5.4**

- **链接: [http://arxiv.org/pdf/2509.25659v2](http://arxiv.org/pdf/2509.25659v2)**

> **作者:** Po-Heng Chou; Chun-Chi Wang; Wei-Lung Mao
>
> **备注:** 5 pages, 8 figures, 2 tables, and published in IEEE IST 2024
>
> **摘要:** In this paper, we propose a YOLO-based deep learning (DL) model for automatic defect detection to solve the time-consuming and labor-intensive tasks in industrial manufacturing. In our experiments, the images of metal sheets are used as the dataset for training the YOLO model to detect the defects on the surfaces and in the holes of metal sheets. However, the lack of metal sheet images significantly degrades the performance of detection accuracy. To address this issue, the ConSinGAN is used to generate a considerable amount of data. Four versions of the YOLO model (i.e., YOLOv3, v4, v7, and v9) are combined with the ConSinGAN for data augmentation. The proposed YOLOv9 model with ConSinGAN outperforms the other YOLO models with an accuracy of 91.3%, and a detection time of 146 ms. The proposed YOLOv9 model is integrated into manufacturing hardware and a supervisory control and data acquisition (SCADA) system to establish a practical automated optical inspection (AOI) system. Additionally, the proposed automated defect detection is easily applied to other components in industrial manufacturing.
>
---
#### [replaced 003] A Speech Enhancement Method Using Fast Fourier Transform and Convolutional Autoencoder
- **分类: cs.SD; eess.AS; 68T07, 68T10, 68T20, 35R25, 35R30**

- **链接: [http://arxiv.org/pdf/2501.01650v2](http://arxiv.org/pdf/2501.01650v2)**

> **作者:** Pu-Yun Kow; Pu-Zhao Kow
>
> **备注:** The paper has been reorganized, and its title has been revised
>
> **摘要:** This paper addresses the reconstruction of audio signals from degraded measurements. We propose a lightweight model that combines the discrete Fourier transform with a Convolutional Autoencoder (FFT-ConvAE), which enabled our team to achieve second place in the Helsinki Speech Challenge 2024. Our results, together with those of other teams, demonstrate the potential of neural-network-free approaches for effective speech signal reconstruction.
>
---
#### [replaced 004] PAGURI: a user experience study of creative interaction with text-to-music models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.04333v4](http://arxiv.org/pdf/2407.04333v4)**

> **作者:** Francesca Ronchini; Luca Comanducci; Gabriele Perego; Fabio Antonacci
>
> **摘要:** In recent years, text-to-music models have been the biggest breakthrough in automatic music generation. While they are unquestionably a showcase of technological progress, it is not clear yet how they can be realistically integrated into the artistic practice of musicians and music practitioners. This paper aims to address this question via Prompt Audio Generation User Research Investigation (PAGURI), a user experience study where we leverage recent text-to-music developments to study how musicians and practitioners interact with these systems, evaluating their satisfaction levels. We developed an online tool through which users can generate music samples and/or apply recently proposed personalization techniques based on fine-tuning to allow the text-to-music model to generate sounds closer to their needs and preferences. Using semi-structured interviews, we analyzed different aspects related to how participants interacted with the proposed tool to understand the current effectiveness and limitations of text-to-music models in enhancing users' creativity. Our research centers on user experiences to uncover insights that can guide the future development of TTM models and their role in AI-driven music creation. Additionally, they offered insightful perspectives on potential system improvements and their integration into their music practices. The results obtained through the study reveal the pros and cons of the use of TTMs for creative endeavors. Participants recognized the system's creative potential and appreciated the usefulness of its personalization features. However, they also identified several challenges that must be addressed before TTMs are ready for real-world music creation, particularly issues of prompt ambiguity, limited controllability, and integration into existing workflows.
>
---
#### [replaced 005] AudioStory: Generating Long-Form Narrative Audio with Large Language Models
- **分类: cs.CV; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.20088v2](http://arxiv.org/pdf/2508.20088v2)**

> **作者:** Yuxin Guo; Teng Wang; Yuying Ge; Shijie Ma; Yixiao Ge; Wei Zou; Ying Shan
>
> **摘要:** Recent advances in text-to-audio (TTA) generation excel at synthesizing short audio clips but struggle with long-form narrative audio, which requires temporal coherence and compositional reasoning. To address this gap, we propose AudioStory, a unified framework that integrates large language models (LLMs) with TTA systems to generate structured, long-form audio narratives. AudioStory possesses strong instruction-following reasoning generation capabilities. It employs LLMs to decompose complex narrative queries into temporally ordered sub-tasks with contextual cues, enabling coherent scene transitions and emotional tone consistency. AudioStory has two appealing features: (1) Decoupled bridging mechanism: AudioStory disentangles LLM-diffuser collaboration into two specialized components, i.e., a bridging query for intra-event semantic alignment and a residual query for cross-event coherence preservation. (2) End-to-end training: By unifying instruction comprehension and audio generation within a single end-to-end framework, AudioStory eliminates the need for modular training pipelines while enhancing synergy between components. Furthermore, we establish a benchmark AudioStory-10K, encompassing diverse domains such as animated soundscapes and natural sound narratives. Extensive experiments show the superiority of AudioStory on both single-audio generation and narrative audio generation, surpassing prior TTA baselines in both instruction-following ability and audio fidelity. Our code is available at https://github.com/TencentARC/AudioStory
>
---
#### [replaced 006] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02038v3](http://arxiv.org/pdf/2411.02038v3)**

> **作者:** Yongxin Zhu; Bocheng Li; Yifei Xin; Zhihua Xia; Linli Xu
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** Vector Quantization (VQ) is essential for discretizing continuous representations in unsupervised learning but suffers from representation collapse, causing low codebook utilization and limiting scalability. Existing solutions often rely on complex optimizations or reduce latent dimensionality, which compromises model capacity and fails to fully solve the problem. We identify the root cause as disjoint codebook optimization, where only a few code vectors are updated via gradient descent. To fix this, we propose \textbf{Sim}ple\textbf{VQ}, which reparameterizes code vectors through a learnable linear transformation layer over a latent basis, optimizing the \textit{entire linear space} rather than nearest \textit{individual code vectors}. Although the multiplication of two linear matrices is equivalent to applying a single linear layer, this simple approach effectively prevents collapse. Extensive experiments on image and audio tasks demonstrate that SimVQ improves codebook usage, is easy to implement, and generalizes well across modalities and architectures. The code is available at https://github.com/youngsheen/SimVQ.
>
---
#### [replaced 007] SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.01812v2](http://arxiv.org/pdf/2510.01812v2)**

> **作者:** Yuxun Tang; Lan Liu; Wenhao Feng; Yiwen Zhao; Jionghao Han; Yifeng Yu; Jiatong Shi; Qin Jin
>
> **备注:** 4 pages, 5 figures;
>
> **摘要:** Singing voice generation progresses rapidly, yet evaluating singing quality remains a critical challenge. Human subjective assessment, typically in the form of listening tests, is costly and time consuming, while existing objective metrics capture only limited perceptual aspects. In this work, we introduce SingMOS-Pro, a dataset for automatic singing quality assessment. Building on our preview version SingMOS, which provides only overall ratings, SingMOS-Pro expands annotations of the additional part to include lyrics, melody, and overall quality, offering broader coverage and greater diversity. The dataset contains 7,981 singing clips generated by 41 models across 12 datasets, spanning from early systems to recent advances. Each clip receives at least five ratings from professional annotators, ensuring reliability and consistency. Furthermore, we explore how to effectively utilize MOS data annotated under different standards and benchmark several widely used evaluation methods from related tasks on SingMOS-Pro, establishing strong baselines and practical references for future research. The dataset can be accessed at https://huggingface.co/datasets/TangRain/SingMOS-Pro.
>
---
