# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] MSP-Conversation: A Corpus for Naturalistic, Time-Continuous Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MSP-Conversation语料库，用于自然对话中的连续情绪识别。解决传统数据集不足和标注方法局限的问题，通过大规模、细粒度标注数据推动动态情绪识别研究。**

- **链接: [https://arxiv.org/pdf/2603.22536](https://arxiv.org/pdf/2603.22536)**

> **作者:** Luz Martinez-Lucas; Pravin Mote; Abinay Reddy Naini; Mohammed Abdelwahab; Carlos Busso
>
> **摘要:** Affective computing aims to understand and model human emotions for computational systems. Within this field, speech emotion recognition (SER) focuses on predicting emotions conveyed through speech. While early SER systems relied on limited datasets and traditional machine learning models, recent deep learning approaches demand largescale, naturalistic emotional corpora. To address this need, we introduce the MSP-Conversation corpus: a dataset of more than 70 hours of conversational audio with time-continuous emotional annotations and detailed speaker diarizations. The time-continuous annotations capture the dynamic and contextdependent nature of emotional expression. The annotations in the corpus include fine-grained temporal traces of valence, arousal, and dominance. The audio data is sourced from publicly available podcasts and overlaps with a subset of the isolated speaking turns in the MSP-Podcast corpus to facilitate direct comparisons between annotation methods (i.e., in-context versus out-of-context annotations). The paper outlines the development of the corpus, annotation methodology, analyses of the annotations, and baseline SER experiments, establishing the MSP-Conversation corpus as a valuable resource for advancing research in dynamic SER in naturalistic settings.
>
---
#### [new 002] Modelling Emotions is an Elusive Pursuit in Affective Computing
- **分类: eess.AS**

- **简介: 论文属于情感计算领域，探讨情绪建模的挑战。指出分类情绪标签限制了应用效果，提出需采用连续维度定义以提高准确性与实用性。**

- **链接: [https://arxiv.org/pdf/2603.23017](https://arxiv.org/pdf/2603.23017)**

> **作者:** Anders Rolighed Larsen; Sneha Das; Line Clemmensen
>
> **摘要:** Affective computing - combining sensor technology, machine learning, and psychology - have been studied for over three decades and is employed in AI-powered technologies to enhance emotional awareness in AI systems, and detect symptoms of mental health disorders such as anxiety and depression. However, the uncertainty in such systems remains high, and the application areas are limited by categorical definitions of emotions and emotional concepts. This paper argues that categorical emotion labels obscure emotional nuance in affective computing, and therefore continuous dimensional definitions are needed to advance the field, increase application usefulness, and lower uncertainties.
>
---
#### [new 003] The Interspeech 2026 Audio Encoder Capability Challenge for Large Audio Language Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频编码器评估任务，旨在解决LALMs依赖高质量音频表示的问题。通过构建XARES-LLM框架，对音频编码器进行多任务测试，推动通用音频表征的发展。**

- **链接: [https://arxiv.org/pdf/2603.22728](https://arxiv.org/pdf/2603.22728)**

> **作者:** Heinrich Dinkel; Jiahao Zhou; Guanbo Wang; Yadong Niu; Junbo Zhang; Yufeng Hao; Ying Liu; Ke Li; Wenwu Wang; Zhiyong Wu; Jian Luan
>
> **备注:** Interspeech 2026 Challenge
>
> **摘要:** This paper presents the Interspeech 2026 Audio Encoder Capability Challenge, a benchmark specifically designed to evaluate and advance the performance of pre-trained audio encoders as front-end modules for Large Audio Language Models (LALMs). While LALMs have shown remarkable understanding of complex acoustic scenes, their performance depends on the semantic richness of the underlying audio encoder representations. This challenge addresses the integration gap by providing a unified generative evaluation framework, XARES-LLM, which assesses submitted encoders across a diverse suite of downstream classification and generation tasks. By decoupling encoder development from LLM fine-tuning, the challenge establishes a standardized protocol for general-purpose audio representations that can effectively be used for the next generation of multimodal language models.
>
---
#### [new 004] MSR-HuBERT: Self-supervised Pre-training for Adaptation to Multiple Sampling Rates
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音处理任务，解决多采样率数据适配问题。通过改进HuBERT模型，设计多采样率自适应预训练方法，提升语音识别与重建效果。**

- **链接: [https://arxiv.org/pdf/2603.23048](https://arxiv.org/pdf/2603.23048)**

> **作者:** Zikang Huang; Meng Ge; Tianrui Wang; Xuanchen Li; Xiaobao Wang; Longbiao Wang; Jianwu Dang
>
> **摘要:** Self-supervised learning (SSL) has advanced speech processing. However, existing speech SSL methods typically assume a single sampling rate and struggle with mixed-rate data due to temporal resolution mismatch. To address this limitation, we propose MSRHuBERT, a multi-sampling-rate adaptive pre-training method. Building on HuBERT, we replace its single-rate downsampling CNN with a multi-sampling-rate adaptive downsampling CNN that maps raw waveforms from different sampling rates to a shared temporal resolution without resampling. This design enables unified mixed-rate pre-training and fine-tuning. In experiments spanning 16 to 48 kHz, MSRHuBERT outperforms HuBERT on speech recognition and full-band speech reconstruction, preserving high-frequency detail while modeling low-frequency semantic structure. Moreover, MSRHuBERT retains HuBERT's mask-prediction objective and Transformer encoder, so existing analyses and improvements that were developed for HuBERT can apply directly.
>
---
#### [new 005] Prompt Amplification and Zero-Shot Late Fusion in Audio-Language Models for Speech Emotion Recognition
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于语音情感识别任务，旨在提升零样本情感识别效果。通过融合音频语言模型与专业模型，提出ZS-Fuse方法，并引入提示增强技术，实现性能提升。**

- **链接: [https://arxiv.org/pdf/2603.23057](https://arxiv.org/pdf/2603.23057)**

> **作者:** Saurabh Kataria; Xiao Hu
>
> **摘要:** Audio-Language Models (ALMs) are making strides in understanding speech and non-speech audio. However, domain-specialist Foundation Models (FMs) remain the best for closed-ended speech processing tasks such as Speech Emotion Recognition (SER). Using ALMs for Zero-shot SER is a popular choice, but their potential to work with specialists to achieve state-of-the-art (SOTA) performance remains unexplored. We propose ZS-Fuse, a late-fusion method that combines zero-shot emotion estimates from a dual-encoder ALM with specialist FMs. To handle ambiguity in emotions and sensitivity to prompt choice, 1) we use a simple prompt ensemble and 2) suggest a novel technique called prompt amplification, which repeats audio and text queries to discover stronger zero-shot capabilities. We demonstrate the efficacy of our technique by evaluating ZS-Fuse with three dual-encoder ALMs and two FMs, and report improvements over SOTA baselines, such as WavLM-Large, on three speech emotion recognition datasets.
>
---
#### [new 006] Velocity Potential Neural Field for Efficient Ambisonics Impulse Response Modeling
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于空间音频建模任务，旨在解决FOA信号的物理一致性问题。通过引入速度势函数，使预测信号自动满足物理方程，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.22589](https://arxiv.org/pdf/2603.22589)**

> **作者:** Yoshiki Masuyama; Francois G. Germain; Gordon Wichern; Chiori Hori; Jonathan Le Roux
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** First-order Ambisonics (FOA) is a standard spatial audio format based on spherical harmonic decomposition. Its zeroth- and first-order components capture the sound pressure and particle velocity, respectively. Recently, physics-informed neural networks have been applied to the spatial interpolation of FOA signals, regularizing the network outputs based on soft penalty terms derived from physical principles, e.g., the linearized momentum equation. In this paper, we reformulate the task so that the predicted FOA signal automatically satisfies the linearized momentum equation. Our network approximates a scalar function called velocity potential, rather than the FOA signal itself. Then, the FOA signal can be readily recovered through the partial derivatives of the velocity potential with respect to the network inputs (i.e., time and microphone position) according to physics of sound propagation. By deriving the four channels of FOA from the single-channel velocity potential, the reconstructed signal follows the physical principle at any time and position by construction. Experimental results on room impulse response reconstruction confirm the effectiveness of the proposed framework.
>
---
#### [new 007] ST-GDance++: A Scalable Spatial-Temporal Diffusion for Long-Duration Group Choreography
- **分类: cs.LG; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于群体舞蹈生成任务，解决多舞者协调与长序列生成效率问题。提出ST-GDance++框架，通过解耦时空依赖提升生成效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.22316](https://arxiv.org/pdf/2603.22316)**

> **作者:** Jing Xu; Weiqiang Wang; Cunjian Chen; Jun Liu; Qiuhong Ke
>
> **摘要:** Group dance generation from music requires synchronizing multiple dancers while maintaining spatial coordination, making it highly relevant to applications such as film production, gaming, and animation. Recent group dance generation models have achieved promising generation quality, but they remain difficult to deploy in interactive scenarios due to bidirectional attention dependencies. As the number of dancers and the sequence length increase, the attention computation required for aligning music conditions with motion sequences grows quadratically, leading to reduced efficiency and increased risk of motion collisions. Effectively modeling dense spatial-temporal interactions is therefore essential, yet existing methods often struggle to capture such complexity, resulting in limited scalability and unstable multi-dancer coordination. To address these challenges, we propose ST-GDance++, a scalable framework that decouples spatial and temporal dependencies to enable efficient and collision-aware group choreography generation. For spatial modeling, we introduce lightweight distance-aware graph convolutions to capture inter-dancer relationships while reducing computational overhead. For temporal modeling, we design a diffusion noise scheduling strategy together with an efficient temporal-aligned attention mask, enabling stream-based generation for long motion sequences and improving scalability in long-duration scenarios. Experiments on the AIOZ-GDance dataset show that ST-GDance++ achieves competitive generation quality with significantly reduced latency compared to existing methods.
>
---
#### [new 008] MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation
- **分类: cs.AI; cs.SD**

- **简介: 该论文提出MUQ-EVAL，一个用于AI音乐生成质量评估的开源逐样本指标，解决传统方法无法准确评估单个音乐片段的问题。通过训练轻量模型，实现高人类相关性评分。**

- **链接: [https://arxiv.org/pdf/2603.22677](https://arxiv.org/pdf/2603.22677)**

> **作者:** Di Zhu; Zixuan Li
>
> **备注:** 10 Pages, 6 figures
>
> **摘要:** Distributional metrics such as Fréchet Audio Distance cannot score individual music clips and correlate poorly with human judgments, while the only per-sample learned metric achieving high human correlation is closed-source. We introduce MUQ-EVAL, an open-source per-sample quality metric for AIgenerated music built by training lightweight prediction heads on frozen MuQ-310M features using MusicEval, a dataset of generated clips from 31 text-to-music systems with expert quality ratings. Our simplest model, frozen features with attention pooling and a two-layer MLP, achieves system-level SRCC = 0.957 and utterance-level SRCC = 0.838 with human mean opinion scores. A systematic ablation over training objectives and adaptation strategies shows that no addition meaningfully improves the frozen baseline, indicating that frozen MuQ representations already capture quality-relevant information. Encoder choice is the dominant design factor, outweighing all architectural and training decisions. LoRA-adapted models trained on as few as 150 clips already achieve usable correlation, enabling personalized quality evaluators from individual listener annotations. A controlled degradation analysis reveals selective sensitivity to signal-level artifacts but insensitivity to musical-structural distortions. Our metric, MUQ-EVAL, is fully open-source, outperforms existing open per-sample metrics, and runs in real time on a single consumer GPU. Code, model weights, and evaluation scripts are available at this https URL.
>
---
#### [new 009] Who Spoke What When? Evaluating Spoken Language Models for Conversational ASR with Semantic and Overlap-Aware Metrics
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于对话语音识别任务，旨在解决多说话人场景下的语音识别问题。通过对比LLM与模块化系统，提出新评估指标，分析不同场景下的性能差异。**

- **链接: [https://arxiv.org/pdf/2603.22709](https://arxiv.org/pdf/2603.22709)**

> **作者:** Naohiro Tawara; Samuele Cornell; Alexander Polok; Marc Delcroix; Lukáš Burget; Shinji Watanabe
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Conversational automatic speech recognition remains challenging due to overlapping speech, far-field noise, and varying speaker counts. While recent LLM-based systems perform well on single-speaker benchmarks, their robustness in multi-speaker settings is unclear. We systematically compare LLM-based and modular pipeline approaches along four axes: overlap robustness, semantic fidelity, speaker count, and single- versus multi-channel input. To capture meaning-altering errors that conventional metrics miss, we introduce tcpSemER, which extends tcpWER by replacing Levenshtein distance with embedding-based semantic similarity. We further decompose tcpWER into overlapping and non-overlapping components for finer-grained analysis. Experiments across three datasets show that LLM-based systems are competitive in two-speaker settings but degrade as speaker count and overlap increase, whereas modular pipelines remain more robust.
>
---
#### [new 010] Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks
- **分类: cs.LG; cs.CR; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升ASR系统对对抗攻击的鲁棒性。通过在推理时随机调整模型精度，增强抗攻击能力，并提出一种检测策略。**

- **链接: [https://arxiv.org/pdf/2603.22590](https://arxiv.org/pdf/2603.22590)**

> **作者:** Matías Pizarro; Raghavan Narasimhan; Asja Fischer
>
> **摘要:** With the increasing deployment of automated and agentic systems, ensuring the adversarial robustness of automatic speech recognition (ASR) models has become critical. We observe that changing the precision of an ASR model during inference reduces the likelihood of adversarial attacks succeeding. We take advantage of this fact to make the models more robust by simple random sampling of the precision during prediction. Moreover, the insight can be turned into an adversarial example detection strategy by comparing outputs resulting from different precisions and leveraging a simple Gaussian classifier. An experimental analysis demonstrates a significant increase in robustness and competitive detection performance for various ASR models and attack types.
>
---
## 更新

#### [replaced 001] Voice Privacy from an Attribute-based Perspective
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音隐私研究任务，旨在解决语音匿名化中的属性泄露问题。通过分析真实属性、原始语音推断属性和保护后语音的属性差异，评估隐私风险。**

- **链接: [https://arxiv.org/pdf/2603.20301](https://arxiv.org/pdf/2603.20301)**

> **作者:** Mehtab Ur Rahman; Martha Larson; Cristian Tejedor-Garcia
>
> **备注:** Submitted to InterSpeech 2026. Author name corrected
>
> **摘要:** Voice privacy approaches that preserve the anonymity of speakers modify speech in an attempt to break the link with the true identity of the speaker. Current benchmarks measure speaker protection based on signal-to-signal comparisons. In this paper, we introduce an attribute-based perspective, where we measure privacy protection in terms of comparisons between sets of speaker attributes. First, we analyze privacy impact by calculating speaker uniqueness for ground truth attributes, attributes inferred on the original speech, and attributes inferred on speech protected with standard anonymization. Next, we examine a threat scenario involving only a single utterance per speaker and calculate attack error rates. Overall, we observe that inferred attributes still present a risk despite attribute inference errors. Our research points to the importance of considering both attribute-related threats and protection mechanisms in future voice privacy research.
>
---
#### [replaced 002] DreamAudio: Customized Text-to-Audio Generation with Diffusion Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出DreamAudio，解决定制化文本到音频生成任务中的细粒度声学控制问题，通过引入新框架实现个性化音频事件生成。**

- **链接: [https://arxiv.org/pdf/2509.06027](https://arxiv.org/pdf/2509.06027)**

> **作者:** Yi Yuan; Xubo Liu; Haohe Liu; Xiyuan Kang; Zhuo Chen; Yuxuan Wang; Mark D. Plumbley; Wenwu Wang
>
> **备注:** Accepted by IEEE/ACM Transactions on Audio, Speech, and Language Processing. Demos are available at this https URL
>
> **摘要:** With the development of large-scale diffusion-based and language-modeling-based generative models, impressive progress has been achieved in text-to-audio generation. Despite producing high-quality outputs, existing text-to-audio models mainly aim to generate semantically aligned sound and fall short of controlling fine-grained acoustic characteristics of specific sounds. As a result, users who need specific sound content may find it difficult to generate the desired audio clips. In this paper, we present DreamAudio for customized text-to-audio generation (CTTA). Specifically, we introduce a new framework that is designed to enable the model to identify auditory information from user-provided reference concepts for audio generation. Given a few reference audio samples containing personalized audio events, our system can generate new audio samples that include these specific events. In addition, two types of datasets are developed for training and testing the proposed systems. The experiments show that DreamAudio generates audio samples that are highly consistent with the customized audio features and aligned well with the input text prompts. Furthermore, DreamAudio offers comparable performance in general text-to-audio tasks. We also provide a human-involved dataset containing audio events from real-world CTTA cases as the benchmark for customized generation tasks.
>
---
#### [replaced 003] U3-xi: Pushing the Boundaries of Speaker Recognition by Incorporating Uncertainty
- **分类: cs.SD**

- **简介: 该论文属于说话人识别任务，旨在解决帧级信息对最终说话人表示贡献不均的问题。通过估计帧级不确定性并赋予自适应权重，提出U3-xi框架提升模型可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.15719](https://arxiv.org/pdf/2601.15719)**

> **作者:** Junjie Li; Kong Aik Lee
>
> **摘要:** An utterance-level speaker embedding is typically obtained by aggregating a sequence of frame-level representations. However, in real-world scenarios, individual frames encode not only speaker-relevant information but also various nuisance factors. As a result, different frames contribute unequally to the final utterance-level speaker representation for Automatic Speaker Verification systems. To address this issue, we propose to estimate the inherent uncertainty of each frame and assign adaptive weights accordingly, where frames with higher uncertainty receive lower attention. Based on this idea, we present U3-xi, a comprehensive framework designed to produce more reliable and interpretable uncertainty estimates for speaker embeddings. Specifically, we introduce several strategies for uncertainty supervision. First, we propose speaker-level uncertainty supervision via a Stochastic Variance Loss, where the distance between an utterance embedding and its corresponding speaker centroid serves as a pseudo ground truth for uncertainty learning. Second, we incorporate global-level uncertainty supervision by injecting the predicted uncertainty into the sof tmax scale during training. This adaptive scaling mechanism adjusts the sharpness of the decision boundary according to sample difficulty, providing global guidance. Third, we redesign the uncertainty estimation module by integrating a Transformer encoder with multi-view self-attention, enabling the model to capture rich local and long-range temporal dependencies. Comprehensive experiments demonstrate that U3-xi is model-agnostic and can be seamlessly applied to various speaker encoders. In particular, when applied to ECAPA-TDNN, it achieves 21.1% and 15.57% relative improvements on the VoxCeleb1 test sets in terms of EER and minDCF, respectively.
>
---
#### [replaced 004] Towards Inclusive Communication: A Unified Framework for Generating Spoken Language from Sign, Lip, and Audio
- **分类: cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于多模态语言生成任务，旨在解决聋哑人群沟通障碍问题。通过统一框架处理手语、唇读和音频，提升语音文本生成效果。**

- **链接: [https://arxiv.org/pdf/2508.20476](https://arxiv.org/pdf/2508.20476)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Sungjune Park; Junil Won; Yong Man Ro
>
> **备注:** Updated the professional title of the corresponding author. Added an Acknowledgement section
>
> **摘要:** Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such audio-centric systems inherently exclude individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we propose the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and Audio-Visual Speech Recognition. Furthermore, our analysis reveals a key linguistic insight: explicitly modeling lip movements as a distinct modality significantly improves SLT performance by capturing critical non-manual cues.
>
---
#### [replaced 005] Structural and Statistical Audio Texture Knowledge Distillation for Acoustic Classification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于环境声音分类任务，旨在解决知识蒸馏中忽略低级音频纹理特征的问题，提出SSATKD框架融合结构与统计纹理特征，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2501.01921](https://arxiv.org/pdf/2501.01921)**

> **作者:** Jarin Ritu; Amirmohammad Mohammadi; Davelle Carreiro; Alexandra Van Dine; Joshua Peeples
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** While knowledge distillation has shown success in various audio tasks, its application to environmental sound classification often overlooks essential low-level audio texture features needed to capture local patterns in complex acoustic environments. To address this gap, the Structural and Statistical Audio Texture Knowledge Distillation (SSATKD) framework is proposed, which combines high-level contextual information with low-level structural and statistical audio textures extracted from intermediate layers. To evaluate its generalizability across diverse acoustic domains, SSATKD is tested on four datasets within the environmental sound classification domain, including two passive sonar datasets (DeepShip and Vessel Type Underwater Acoustic Data (VTUAD)) and two general environmental sound datasets (Environmental Sound Classification 50 (ESC-50) and Tampere University of Technology (TUT) Acoustic Scenes). Two teacher adaptation strategies are explored: classifier-head-only adaptation and full fine-tuning. The framework is further evaluated using various convolutional and transformer-based teacher models. Experimental results demonstrate consistent accuracy improvements across all datasets and settings, confirming the effectiveness and robustness of SSATKD in real-world sound classification tasks.
>
---
#### [replaced 006] WiRD-Gest: Gesture Recognition In The Real World Using Range-Doppler Wi-Fi Sensing on COTS Hardware
- **分类: eess.AS**

- **简介: 该论文属于手势识别任务，旨在解决Wi-Fi传感在实际环境中的敏感性和设备放置问题。通过单个COTS设备实现基于范围-多普勒信息的准确手势识别。**

- **链接: [https://arxiv.org/pdf/2603.22131](https://arxiv.org/pdf/2603.22131)**

> **作者:** Jessica Sanson; Rahul C. Shah; Yazhou Zhu; Rafael Rosales; Valerio Frascolla
>
> **摘要:** Wi-Fi sensing has emerged as a promising technique for gesture recognition, yet its practical deployment is hindered by environmental sensitivity and device placement challenges. To overcome these limitations we propose Wi-Fi Range and Doppler (WiRD)-Gest, a novel system that performs gesture recognition using a single, unmodified Wi-Fi transceiver on a commercial off-the-shelf (COTS) laptop. The system leverages an monostatic full duplex sensing pipeline capable of extracting Range-Doppler (RD) information. Utilizing this, we present the first benchmark of deep learning models for gesture recognition based on monostatic sensing. The key innovation lies in how monostatic sensing and spatial (range) information fundamentally transforms accuracy, robustness and generalization compared to prior approaches. We demonstrate excellent performance in crowded, unseen public spaces with dynamic interference and additional moving targets even when trained on data from controlled environments only. These are scenarios where prior Wi-Fi sensing approaches often fail, however, our system suffers minor degradation. The WiRD-Gest benchmark and dataset will also be released as open source.
>
---
#### [replaced 007] Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频理解任务，旨在解决现有基准未充分评估音频作用的问题。通过引入语音编码器，验证音频在跨模态任务中的重要性。**

- **链接: [https://arxiv.org/pdf/2509.17901](https://arxiv.org/pdf/2509.17901)**

> **作者:** Geewook Kim; Minjoon Seo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech and audio encoders developed over years of community effort are routinely excluded from video understanding pipelines -- not because they fail, but because benchmarks never required listening. We audit 10 video benchmarks and find items largely solvable from visual cues alone: a single-frame probe answers ~76% of AVQA without audio, suggesting poor measurement of audio-visual reasoning. Building on LLaVA-OneVision, we attach a speech/audio encoder and compare five compressor architectures under 25x token reduction (25 Hz to 1 Hz). Across 10 benchmarks -- with and without filtering -- audio yields clear gains on tasks requiring speech comprehension or cross-modal grounding, while vision-centric suites remain largely unaffected. Our results show that speech encoders play a larger role in video understanding than current benchmarks suggest. We will fully open-source our work at this https URL.
>
---
#### [replaced 008] When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究音频与文本冲突时语言模型的模态选择问题，属于多模态融合任务。通过构建数据集和指标，分析模型对音频的依赖程度，揭示文本主导现象及其影响因素。**

- **链接: [https://arxiv.org/pdf/2602.11488](https://arxiv.org/pdf/2602.11488)**

> **作者:** Jayadev Billa
>
> **备注:** 13 pages, 18 tables, 4 figures, benchmark and code at this https URL
>
> **摘要:** When audio and text conflict, speech-enabled language models follow text far more often than they do when arbitrating between two conflicting text sources, even under explicit instructions to trust the audio. We introduce ALME (Audio-LLM Modality Evaluation), a dataset of 57,602 controlled audio-text conflict stimuli across eight languages, together with Text Dominance Ratio (TDR), which measures how often a model follows conflicting text when instructed to follow audio. Gemini 2.0 Flash and GPT-4o show TDR 10--26$\times$ higher than a baseline that replaces audio with its transcript under otherwise identical conditions (Gemini 2.0 Flash: 16.6% vs. 1.6%; GPT-4o: 23.2% vs. 0.9%). These results suggest that text dominance reflects not only information content, but also an asymmetry in arbitration accessibility, i.e., how easily the model can use competing representations at decision time. Framing the transcript as deliberately corrupted reduces TDR by 80%, whereas forcing explicit transcription increases it by 14%. A fine-tuning ablation further suggests that arbitration behavior depends more on LLM reasoning than on the audio input path alone. Across four audio-LLMs, we observe the same qualitative pattern with substantial cross-model and cross-linguistic variation.
>
---
#### [replaced 009] ASK: Adaptive Self-improving Knowledge Framework for Audio Text Retrieval
- **分类: eess.AS; cs.IR; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于音频-文本检索任务，旨在解决梯度局部性瓶颈和表征漂移不匹配问题。提出ASK框架，通过多粒度知识注入和动态优化提升检索性能。**

- **链接: [https://arxiv.org/pdf/2512.19703](https://arxiv.org/pdf/2512.19703)**

> **作者:** Siyuan Fu; Xuchen Guo; Mingjun Liu; Hongxiang Li; Boyin Tan; Gongxi Zhu; Xianwei Zhuang; Jinghan Ru; Yuxin Xie; Yuguo Yin
>
> **摘要:** The dominant paradigm for Audio-Text Retrieval (ATR) relies on dual-encoder architectures optimized via mini-batch contrastive learning. However, restricting optimization to local in-batch samples creates a fundamental limitation we term the Gradient Locality Bottleneck (GLB), which prevents the resolution of acoustic ambiguities and hinders the learning of rare long-tail concepts. While external knowledge injection can break this bottleneck, it often triggers a problem called Representation-Drift Mismatch (RDM), where a static knowledge base becomes misaligned with evolving encoders, degrading guidance into noise. To address these intertwined challenges, we propose the Adaptive Self-improving Knowledge (ASK) framework. ASK breaks the GLB via multi-grained knowledge injection and mitigates RDM through a dynamic refinement strategy that synchronizes the knowledge base with the model. Additionally, an adaptive reliability weighting scheme is employed to filter retrieval noise based on cross-modal consistency. Extensive experiments across multiple benchmarks demonstrate that ASK consistently achieves new state-of-the-art performance across various backbones.
>
---
#### [replaced 010] Selective Classifier-free Guidance for Zero-shot Text-to-speech
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于零样本文本到语音任务，旨在平衡语音保真度与文本一致性。通过改进CFG策略，提升说话人相似度并减少文本偏差。**

- **链接: [https://arxiv.org/pdf/2509.19668](https://arxiv.org/pdf/2509.19668)**

> **作者:** John Zheng; Farhad Maleki
>
> **备注:** 5 pages, 7 figures, 1 table. Revision 1: removed ICASSP copyright notice
>
> **摘要:** In zero-shot text-to-speech, achieving a balance between fidelity to the target speaker and adherence to text content remains a challenge. While classifier-free guidance (CFG) strategies have shown promising results in image generation, their application to speech synthesis are underexplored. Separating the conditions used for CFG enables trade-offs between different desired characteristics in speech synthesis. In this paper, we evaluate the adaptability of CFG strategies originally developed for image generation to speech synthesis and extend separated-condition CFG approaches for this domain. Our results show that CFG strategies effective in image generation generally fail to improve speech synthesis. We also find that we can improve speaker similarity while limiting degradation of text adherence by applying standard CFG during early timesteps and switching to selective CFG only in later timesteps. Surprisingly, we observe that the effectiveness of a selective CFG strategy is highly text-representation dependent, as differences between the two languages of English and Mandarin can lead to different results even with the same model.
>
---
#### [replaced 011] Investigating self-supervised representations for audio-visual deepfake detection
- **分类: cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于音频-视频深度伪造检测任务，旨在探索自监督表示的有效性。研究评估了不同模态和领域的自监督特征，发现其具有互补性且能捕捉深度伪造相关信息，但真实场景下仍面临挑战。**

- **链接: [https://arxiv.org/pdf/2511.17181](https://arxiv.org/pdf/2511.17181)**

> **作者:** Dragos-Alexandru Boldisor; Stefan Smeu; Dan Oneata; Elisabeta Oneata
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Self-supervised representations excel at many vision and speech tasks, but their potential for audio-visual deepfake detection remains underexplored. Unlike prior work that uses these features in isolation or buried within complex architectures, we systematically evaluate them across modalities (audio, video, multimodal) and domains (lip movements, generic visual content). We assess three key dimensions: detection effectiveness, interpretability of encoded information, and cross-modal complementarity. We find that most self-supervised features capture deepfake-relevant information, and that this information is complementary. Moreover, models primarily attend to semantically meaningful regions rather than spurious artifacts (such as the leading silence). Among the investigated features, audio-informed representations generalize best and achieve state-of-the-art results. However, generalization to realistic in-the-wild data remains challenging. Our analysis indicates this gap stems from intrinsic dataset difficulty rather than from features latching onto superficial patterns. Project webpage: this https URL.
>
---
#### [replaced 012] Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨语言语音识别任务，旨在解决帕金森病失语检测中数据不足的问题。通过语言迁移方法对自监督语音表示进行调整，提升跨语言检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22225](https://arxiv.org/pdf/2603.22225)**

> **作者:** Abner Hernandez; Eunjung Yeo; Kwanghee Choi; Chin-Jou Li; Zhengjun Yue; Rohan Kumar Das; Jan Rusz; Mathew Magimai Doss; Juan Rafael Orozco-Arroyave; Tomás Arias-Vergara; Andreas Maier; Elmar Nöth; David R. Mortensen; David Harwath; Paula Andrea Perez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure.
>
---
