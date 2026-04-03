# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Reverberation-Robust Localization of Speakers Using Distinct Speech Onsets and Multi-channel Cross-Correlations
- **分类: eess.AS**

- **简介: 该论文属于语音源定位任务，旨在解决强混响环境下说话人定位的问题。通过提出两种算法，利用时频分解和多通道互相关抑制混响，实现静止与移动说话人的可靠定位。**

- **链接: [https://arxiv.org/pdf/2604.01524](https://arxiv.org/pdf/2604.01524)**

> **作者:** Shoufeng Lin
>
> **摘要:** Many speaker localization methods can be found in the literature. However, speaker localization under strong reverberation still remains a major challenge in the real-world applications. This paper proposes two algorithms for localizing speakers using microphone array recordings of reverberated sounds. To separate concurrent speakers, the first algorithm decomposes microphone signals spectrotemporally into subbands via an auditory filterbank. To suppress reverberation, we propose a novel speech onset detection approach derived from the speech signal and impulse response models, and further propose to formulate the multi-channel cross-correlation coefficient (MCCC) of encoded speech onsets in each subband. The subband results are combined to estimate the directions-of-arrival (DOAs) of speakers. The second algorithm extends the generalized cross-correlation - phase transform (GCC-PHAT) method by using redundant information of multiple microphones to address the reverberation problem. The proposed methods have been evaluated under adverse conditions using not only simulated signals (reverberation time $T_{60}$ of up to $1$s) but also recordings in a real reverberant room ($T_{60} \approx 0.65$s). Comparing with some state-of-the-art localization methods, experimental results confirm that the proposed methods can reliably locate static and moving speakers, in presence of reverberation.
>
---
#### [new 002] Woosh: A Sound Effects Foundation Model
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文介绍了一个名为Woosh的声音效果基础模型，用于生成和处理音效。任务是提升音效生成性能，解决现有模型不足。工作包括设计音频编码器、文本-音频对齐模型及生成模型，并进行性能评估。**

- **链接: [https://arxiv.org/pdf/2604.01929](https://arxiv.org/pdf/2604.01929)**

> **作者:** Gaëtan Hadjeres; Marc Ferras; Khaled Koutini; Benno Weck; Alexandre Bittar; Thomas Hummel; Zineb Lahrici; Hakim Missoum; Joan Serrà; Yuki Mitsufuji
>
> **摘要:** The audio research community depends on open generative models as foundational tools for building novel approaches and establishing baselines. In this report, we present Woosh, Sony AI's publicly released sound effect foundation model, detailing its architecture, training process, and an evaluation against other popular open models. Being optimized for sound effects, we provide (1) a high-quality audio encoder/decoder model and (2) a text-audio alignment model for conditioning, together with (3) text-to-audio and (4) video-to-audio generative models. Distilled text-to-audio and video-to-audio models are also included in the release, allowing for low-resource operation and fast inference. Our evaluation on both public and private data shows competitive or better performance for each module when compared to existing open alternatives like StableAudio-Open and TangoFlux. Inference code and model weights are available at this https URL. Demo samples can be found at this https URL.
>
---
#### [new 003] Robust Pitch Estimation and Tracking for Speakers Based on Subband Encoding and the Generalized Labeled Multi-Bernoulli Filter
- **分类: eess.AS**

- **简介: 该论文属于语音处理中的音高估计与跟踪任务，旨在提高噪声和混响环境下的音高估计准确性。通过子带编码和GLMB滤波器实现更鲁棒的音高跟踪。**

- **链接: [https://arxiv.org/pdf/2604.01541](https://arxiv.org/pdf/2604.01541)**

> **作者:** Shoufeng Lin
>
> **摘要:** This paper proposes a new pitch estimator and a novel pitch tracker for speakers. We first decompose the sound signal into subbands using an auditory filterbank, assuming time-frequency sparsity of human speech. Instead of directly selecting the number of subbands according to experience, we propose a novel frequency coverage metric to derive the number of subbands and the center frequencies of the filterbank. The subband signals are then encoded inspired by the computational auditory scene analysis (CASA) approach, and the normalized autocorrelations are calculated for pitch estimation. To suppress spurious errors and track the speaker identity, the temporal continuity constraint is exploited and a Generalized Labeled Multi-Bernoulli (GLMB) filter is adapted for pitch tracking, where we use a novel pitch state transition model based on the Ornstein-Uhlenbeck process, and the measurement driven birth model for adaptive new births of pitch targets. Experimental evaluations with various additive noises demonstrate that the proposed methods have achieved better accuracy compared with several state-of-the-art pitch estimation methods in most studied scenarios. Tests using real recordings in a reverberant room also show that the proposed method is robust against reverberation.
>
---
#### [new 004] T5Gemma-TTS Technical Report
- **分类: eess.AS**

- **简介: 该论文提出T5Gemma-TTS，解决语音合成中的文本条件控制与持续性问题，通过编码器-解码器结构和PM-RoPE提升语音质量与长度控制。**

- **链接: [https://arxiv.org/pdf/2604.01760](https://arxiv.org/pdf/2604.01760)**

> **作者:** Chihiro Arata; Kiyoshi Kurihara
>
> **摘要:** Autoregressive neural codec language models have shown strong zero-shot voice cloning ability, but decoder-only architectures treat input text as a prefix that competes with the growing audio sequence for positional capacity, weakening text conditioning over long utterances. We present T5Gemma-TTS, an encoder-decoder codec language model that maintains persistent text conditioning by routing bidirectional text representations through cross-attention at every decoder layer. Built on the T5Gemma pretrained encoder-decoder backbone (2B encoder + 2B decoder; 4B parameters), it inherits rich linguistic knowledge without phoneme conversion and processes text directly at the subword level. To improve duration control, we introduce Progress-Monitoring Rotary Position Embedding (PM-RoPE) in all 26 cross-attention layers, injecting normalized progress signals that help the decoder track target speech length. Trained on 170,000 hours of multilingual speech in English, Chinese, and Japanese, T5Gemma-TTS achieves a statistically significant speaker-similarity gain on Japanese over XTTSv2 (0.677 vs. 0.622; non-overlapping 95% confidence intervals) and the highest numerical Korean speaker similarity (0.747) despite Korean not being included in training, although this margin over XTTSv2 (0.741) is not statistically conclusive. It also attains the lowest numerical Japanese character error rate among five baselines (0.126), though this ranking should be interpreted cautiously because of partial confidence-interval overlap with Kokoro. English results on LibriSpeech should be viewed as an upper-bound estimate because LibriHeavy is a superset of LibriSpeech. Using the same checkpoint, disabling PM-RoPE at inference causes near-complete synthesis failure: CER degrades from 0.129 to 0.982 and duration accuracy drops from 79% to 46%. Code and weights are available at this https URL.
>
---
#### [new 005] Evolutionary Multi-Objective Fusion of Deepfake Speech Detectors
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; cs.NE**

- **简介: 该论文属于深度伪造语音检测任务，旨在解决传统融合方法复杂度高、效果提升有限的问题。通过进化多目标优化，实现检测精度与系统复杂度的平衡。**

- **链接: [https://arxiv.org/pdf/2604.01330](https://arxiv.org/pdf/2604.01330)**

> **作者:** Vojtěch Staněk; Martin Perešíni; Lukáš Sekanina; Anton Firc; Kamil Malinka
>
> **备注:** Accepted to WCCI CEC 2026
>
> **摘要:** While deepfake speech detectors built on large self-supervised learning (SSL) models achieve high accuracy, employing standard ensemble fusion to further enhance robustness often results in oversized systems with diminishing returns. To address this, we propose an evolutionary multi-objective score fusion framework that jointly minimizes detection error and system complexity. We explore two encodings optimized by NSGA-II: binary-coded detector selection for score averaging and a real-valued scheme that optimizes detector weights for a weighted sum. Experiments on the ASVspoof 5 dataset with 36 SSL-based detectors show that the obtained Pareto fronts outperform simple averaging and logistic regression baselines. The real-valued variant achieves 2.37% EER (0.0684 minDCF) and identifies configurations that match state-of-the-art performance while significantly reducing system complexity, requiring only half the parameters. Our method also provides a diverse set of trade-off solutions, enabling deployment choices that balance accuracy and computational cost.
>
---
#### [new 006] Validating Computational Markers of Depressive Behavior: Cross-Linguistic Speech-Based Depression Detection with Neurophysiological Validation
- **分类: eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在验证跨语言的语音特征及神经生理基础。通过扩展CDMA框架，融合不同情感状态的语音，并结合EEG数据验证模型有效性。**

- **链接: [https://arxiv.org/pdf/2604.01533](https://arxiv.org/pdf/2604.01533)**

> **作者:** Fuxiang Tao; Dongwei Li; Shuning Tang; Xuri Ge; Wei Ma; Anna Esposito; Alessandro Vinciarelli
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Speech-based depression detection has shown promise as an objective diagnostic tool, yet the cross-linguistic robustness of acoustic markers and their neurobiological underpinnings remain underexplored. This study extends Cross-Data Multilevel Attention (CDMA) framework, initially validated on Italian, to investigate these dimensions using a Chinese Mandarin dataset with Electroencephalography (EEG) recordings. We systematically fuse read speech with spontaneous speech across different emotional valences (positive, neutral, negative) to investigate whether emotional arousal is a more critical factor than valence polarity in enhancing detection performance in speech. Additionally, we establish the first neurophysiological validation for a speech-based depression model by correlating its predictions with neural oscillatory patterns during emotional face processing. Our results demonstrate strong cross-linguistic generalizability of the CDMA framework, achieving state-of-the-art performance (F1-score up to 89.6%) on the Chinese dataset, which is comparable to the previous Italian validation. Critically, emotionally valenced speech (both positive and negative) significantly outperformed neutral speech. This comparable performance between positive and negative tasks supports the emotional arousal hypothesis. Most importantly, EEG analysis revealed significant correlations between the model's speech-derived depression estimates and neural oscillatory patterns (theta and alpha bands), demonstrating alignment with established neural markers of emotional dysregulation in depression. This alignment, combined with the model's cross-linguistic robustness, not only supports that the CDMA framework's approach is a universally applicable and neurobiologically validated strategy but also establishes a novel paradigm for the neurophysiological validation of computational mental health models.
>
---
#### [new 007] GAP-URGENet: A Generative-Predictive Fusion Framework for Universal Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出GAP-URGENet，用于通用语音增强任务，融合生成与预测分支，提升语音质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.01832](https://arxiv.org/pdf/2604.01832)**

> **作者:** Xiaobin Rong; Yushi Wang; Zheng Wang; Jing Lu
>
> **备注:** Awarded 1st place in the URGENT 2026 Challenge (objective phase), accepted by ICASSP 2026
>
> **摘要:** We introduce GAP-URGENet, a generative-predictive fusion framework developed for Track 1 of the ICASSP 2026 URGENT Challenge. The system integrates a generative branch, which performs full-stack speech restoration in a self-supervised representation domain and reconstructs the waveform via a neural vocoder, along with a predictive branch that performs spectrogram-domain enhancement, providing complementary cues. Outputs from both branches are fused by a post-processing module, which also performs bandwidth extension to generate the enhanced waveform at 48 kHz, later downsampled to the original sampling rate. This generative-predictive fusion improves robustness and perceptual quality, achieving top performance in the blind-test phase and ranking 1st in the objective evaluation. Audio examples are available at this https URL.
>
---
#### [new 008] PhiNet: Speaker Verification with Phonetic Interpretability
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出PhiNet，解决自动说话人验证（ASV）缺乏透明性的问题。通过引入语音学解释，增强决策的可解释性，提升验证结果的可信度与可分析性。**

- **链接: [https://arxiv.org/pdf/2604.01590](https://arxiv.org/pdf/2604.01590)**

> **作者:** Yi Ma; Shuai Wang; Tianchi Liu; Haizhou Li
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing. Codes: this https URL
>
> **摘要:** Despite remarkable progress, automatic speaker verification (ASV) systems typically lack the transparency required for high-accountability applications. Motivated by how human experts perform forensic speaker comparison (FSC), we propose a speaker verification network with phonetic interpretability, PhiNet, designed to enhance both local and global interpretability by leveraging phonetic evidence in decision-making. For users, PhiNet provides detailed phonetic-level comparisons that enable manual inspection of speaker-specific features and facilitate a more critical evaluation of verification outcomes. For developers, it offers explicit reasoning behind verification decisions, simplifying error tracing and informing hyperparameter selection. In our experiments, we demonstrate PhiNet's interpretability with practical examples, including its application in analyzing the impact of different hyperparameters. We conduct both qualitative and quantitative evaluations of the proposed interpretability methods and assess speaker verification performance across multiple benchmark datasets, including VoxCeleb, SITW, and LibriSpeech. Results show that PhiNet achieves performance comparable to traditional black-box ASV models while offering meaningful, interpretable explanations for its decisions, bridging the gap between ASV and forensic analysis.
>
---
#### [new 009] FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
- **分类: cs.SD**

- **简介: 该论文属于语音对话系统中的端点检测任务，旨在解决实时全双工通信中低延迟和鲁棒的说话人切换问题。提出FastTurn框架，结合声学与语义线索，提升决策准确性与响应速度。**

- **链接: [https://arxiv.org/pdf/2604.01897](https://arxiv.org/pdf/2604.01897)**

> **作者:** Chengyou Wang; Hongfei Xue; Chunjiang He; Jingbin Hu; Shuiyuan Wang; Bo Wu; Yuyu Ji; Jimeng Zheng; Ruofei Chen; Zhou Zhu; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
>
---
#### [new 010] Acoustic and perceptual differences between standard and accented Chinese speech and their voice clones
- **分类: cs.SD; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于语音处理任务，研究标准与方言普通话及其语音克隆的声学和感知差异。旨在解决语音克隆中口音保留与身份匹配的问题，通过计算与感知实验发现口音影响克隆相似度和可懂度。**

- **链接: [https://arxiv.org/pdf/2604.01562](https://arxiv.org/pdf/2604.01562)**

> **作者:** Tianle Yang; Chengzhe Sun; Phil Rose; Siwei Lyu
>
> **摘要:** Voice cloning is often evaluated in terms of overall quality, but less is known about accent preservation and its perceptual consequences. We compare standard and heavily accented Mandarin speech and their voice clones using a combined computational and perceptual design. Embedding-based analyses show no reliable accented-standard difference in original-clone distances across systems. In the perception study, clones are rated as more similar to their originals for standard than for accented speakers, and intelligibility increases from original to clone, with a larger gain for accented speech. These results show that accent variation can shape perceived identity match and intelligibility in voice cloning even when it is not reflected in an off-the-shelf speaker-embedding distance, and they motivate evaluating speaker identity preservation and accent preservation as separable dimensions.
>
---
#### [new 011] Combining Masked Language Modeling and Cross-Modal Contrastive Learning for Prosody-Aware TTS
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升语音的韵律建模。通过结合掩码语言模型和跨模态对比学习，优化语音生成质量，但发现过度优化韵律可能影响音素区分。**

- **链接: [https://arxiv.org/pdf/2604.01247](https://arxiv.org/pdf/2604.01247)**

> **作者:** Kirill Borodin; Vasiliy Kudryavtsev; Maxim Maslov; Nikita Vasiliev; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** This paper has been submitted to Interspeech 2026 for review
>
> **摘要:** We investigate multi-stage pretraining for prosody modeling in diffusion-based TTS. A speaker-conditioned dual-stream encoder is trained with masked language modeling followed by SigLIP-style cross-modal contrastive learning using mixed-phoneme batches, with an additional same-phoneme refinement stage studied separately. We evaluate intrinsic text-audio retrieval and downstream synthesis in Grad-TTS and a latent diffusion TTS system. The two-stage curriculum (MLM + mixed-phoneme contrastive learning) achieves the best overall synthesis quality in terms of intelligibility, speaker similarity, and perceptual measures. Although same-phoneme refinement improves prosodic retrieval, it reduces phoneme discrimination and degrades synthesis. These findings indicate that improvements in embedding-space metrics do not necessarily translate to better generative performance and highlight the need to balance phoneme discrimination and prosodic sensitivity in TTS pretraining.
>
---
#### [new 012] Prosodic ABX: A Language-Agnostic Method for Measuring Prosodic Contrast in Speech Representations
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音表示研究任务，旨在测量语音模型对韵律对比的敏感性。提出Prosodic ABX方法，无需标签即可评估韵律对比，验证了模型在不同语言中的表现。**

- **链接: [https://arxiv.org/pdf/2604.02102](https://arxiv.org/pdf/2604.02102)**

> **作者:** Haitong Sun; Stephen McIntosh; Kwanghee Choi; Eunjung Yeo; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Submitted to Interspeech 2026; 6 pages, 4 figures
>
> **摘要:** Speech representations from self-supervised speech models (S3Ms) are known to be sensitive to phonemic contrasts, but their sensitivity to prosodic contrasts has not been directly measured. The ABX discrimination task has been used to measure phonemic contrast in S3M representations via minimal pairs. We introduce prosodic ABX, an extension of this framework to evaluate prosodic contrast with only a handful of examples and no explicit labels. Also, we build and release a dataset of English and Japanese minimal pairs and use it along with a Mandarin dataset to evaluate contrast in English stress, Japanese pitch accent, and Mandarin tone. Finally, we show that model and layer rankings are often preserved across several experimental conditions, making it practical for low-resource settings.
>
---
#### [new 013] Tracking the emergence of linguistic structure in self-supervised models learning from speech
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究自监督语音模型在训练过程中语言结构的形成时间与模式，旨在揭示不同语言层次的结构如何随训练进展而发展。**

- **链接: [https://arxiv.org/pdf/2604.02043](https://arxiv.org/pdf/2604.02043)**

> **作者:** Marianne de Heer Kloots; Martijn Bentum; Hosein Mohebbi; Charlotte Pouw; Gaofei Shen; Willem Zuidema
>
> **摘要:** Self-supervised speech models learn effective representations of spoken language, which have been shown to reflect various aspects of linguistic structure. But when does such structure emerge in model training? We study the encoding of a wide range of linguistic structures, across layers and intermediate checkpoints of six Wav2Vec2 and HuBERT models trained on spoken Dutch. We find that different levels of linguistic structure show notably distinct layerwise patterns as well as learning trajectories, which can partially be explained by differences in their degree of abstraction from the acoustic signal and the timescale at which information from the input is integrated. Moreover, we find that the level at which pre-training objectives are defined strongly affects both the layerwise organization and the learning trajectories of linguistic structures, with greater parallelism induced by higher-order prediction tasks (i.e. iteratively refined pseudo-labels).
>
---
## 更新

#### [replaced 001] J-CHAT: Japanese Large-scale Spoken Dialogue Corpus for Spoken Dialogue Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文介绍J-CHAT，一个大规模日语口语对话语料库，用于解决口语对话系统数据不足的问题。通过构建高质量语料，提升对话模型性能。**

- **链接: [https://arxiv.org/pdf/2407.15828](https://arxiv.org/pdf/2407.15828)**

> **作者:** Wataru Nakata; Kentaro Seki; Hitomi Yanaka; Yuki Saito; Shinnosuke Takamichi; Hiroshi Saruwatari
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Spoken dialogue is essential for human-AI interactions, providing expressive capabilities beyond text. Developing effective spoken dialogue systems (SDSs) requires large-scale, high-quality, and diverse spoken dialogue corpora. However, existing datasets are often limited in size, spontaneity, or linguistic coherence. To address these limitations, we introduce J-CHAT, a 76,000-hour open-source Japanese spoken dialogue corpus. Constructed using an automated, language-independent methodology, J-CHAT ensures acoustic cleanliness, diversity, and natural spontaneity. The corpus is built from YouTube and podcast data, with extensive filtering and denoising to enhance quality. Experimental results with generative spoken dialogue language models trained on J-CHAT demonstrate its effectiveness for SDS development. By providing a robust foundation for training advanced dialogue models, we anticipate that J-CHAT will drive progress in human-AI dialogue research and applications.
>
---
#### [replaced 002] OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出OmniVoice，解决多语言零样本文本到语音合成问题。采用创新的扩散语言模型架构，直接将文本映射到音频标记，提升效率与覆盖语言数量。**

- **链接: [https://arxiv.org/pdf/2604.00688](https://arxiv.org/pdf/2604.00688)**

> **作者:** Han Zhu; Lingxuan Ye; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhifeng Han; Weiji Zhuang; Long Lin; Daniel Povey
>
> **摘要:** We present OmniVoice, a massive multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL.
>
---
#### [replaced 003] PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion
- **分类: cs.SD**

- **简介: 该论文提出PFluxTTS，属于文本到语音合成任务，解决流匹配TTS的稳定性、跨语言语音克隆和音频质量问题，通过双解码器设计和改进声码器提升性能。**

- **链接: [https://arxiv.org/pdf/2602.04160](https://arxiv.org/pdf/2602.04160)**

> **作者:** Vikentii Pankov; Artem Gribul; Oktai Tatanov; Vladislav Proskurov; Yuliya Korotkova; Darima Mylzenova; Dmitrii Vypirailenko
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** We present PFluxTTS, a hybrid text-to-speech system addressing three gaps in flow-matching TTS: the stability-naturalness trade-off, weak cross-lingual voice cloning, and limited audio quality from low-rate mel features. Our contributions are: (1) a dual-decoder design combining duration-guided and alignment-free models through inference-time vector-field fusion; (2) robust cloning using a sequence of speech-prompt embeddings in a FLUX-based decoder, preserving speaker traits across languages without prompt transcripts; and (3) a modified PeriodWave vocoder with super-resolution to 48 kHz. On cross-lingual in-the-wild data, PFluxTTS clearly outperforms F5-TTS, FishSpeech, and SparkTTS, matches ChatterBox in naturalness (MOS 4.11) while achieving 23% lower WER (6.9% vs. 9.0%), and surpasses ElevenLabs in speaker similarity (+0.32 SMOS). The system remains robust in challenging scenarios where most open-source models fail, while requiring only short reference audio and no extra training. Audio demos are available at this https URL
>
---
#### [replaced 004] RIFT: Entropy-Optimised Fractional Wavelet Constellations for Ideal Time-Frequency Estimation
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于时间-频率分析任务，旨在解决非平稳信号的高分辨率时频表示问题。提出RIFT方法，通过优化分数小波变换集，抑制交叉项，提升时频精度。**

- **链接: [https://arxiv.org/pdf/2501.15764](https://arxiv.org/pdf/2501.15764)**

> **作者:** James M. Cozens; Simon J. Godsill
>
> **备注:** Minor revision; no substantive changes
>
> **摘要:** We introduce a new method for estimating the Ideal Time-Frequency Representation (ITFR) of complex nonstationary signals. The Reconstructive Ideal Fractional Transform (RIFT) computes a constellation of Continuous Fractional Wavelet Transforms (CFWTs) aligned to different local time-frequency curvatures. This constellation is combined into a single optimised time-frequency energy representation via a localised entropy-based sparsity measure, designed to resolve auto-terms and attenuate cross-terms. Finally, a positivity-constrained Lucy-Richardson deconvolution with total-variation regularisation is applied to estimate the ITFR, achieving auto-term resolution comparable to that of the Wigner-Ville Distribution (WVD), yielding the high-resolution RIFT representation. The required Cohen's class convolutional kernels are fully derived in the paper for the chosen CFWT constellations. Additionally, the optimisation yields an Instantaneous Phase Direction (IPD) field, which allows the localised curvature in speech or music extracts to be visualised and utilised within a Kalman tracking scheme, enabling the extraction of signal component trajectories and the construction of the Spline-RIFT variant. Evaluation on synthetic and real-world signals demonstrates the algorithm's ability to effectively suppress cross-terms and achieve superior time-frequency precision relative to competing methods. This advance holds significant potential for a wide range of applications requiring high-resolution cross-term-free time-frequency analysis.
>
---
#### [replaced 005] Inter-Speaker Relative Cues for Two-Stage Text-Guided Target Speech Extraction
- **分类: eess.AS**

- **简介: 该论文属于语音提取任务，解决文本引导下目标说话人分离问题。提出两阶段框架，利用相对线索提升分类与提取性能。**

- **链接: [https://arxiv.org/pdf/2603.01316](https://arxiv.org/pdf/2603.01316)**

> **作者:** Wang Dai; Archontis Politis; Tuomas Virtanen
>
> **备注:** Submitted to IEEE TASLP
>
> **摘要:** This paper investigates the use of relative cues for text-based target speech extraction (TSE). We first provide a theoretical justification for relative cues from the perspectives of human perception and label quantization, showing that relative cues preserve fine-grained distinctions that are often lost in absolute categorical representations for continuous-valued attributes. Building on this analysis, we propose a two-stage TSE framework in which a speech separation model first generates candidate sources, followed by a text-guided classifier that selects the target speaker based on embedding similarity. Within this framework, we train two separate classification models to evaluate the advantages of relative cues over independent cues in case of continuous-valued attributes, considering both classification accuracy and TSE performance. Experimental results demonstrate that (i) relative cues achieve higher overall classification accuracy and improved TSE performance compared with independent cues; (ii) the proposed two-stage framework substantially outperforms single-stage text-conditioned extraction methods on both signal-level and objective perceptual metrics; and (iii) several relative cues, including language, loudness, distance, temporal order, speaking duration, random cues, and all cues, can even surpass the performance of an enrollment-audio-based TSE system. Further analysis reveals notable differences in discriminative power across cue types, providing insights into the effectiveness of different relative cues for TSE.
>
---
