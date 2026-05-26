# 音频 cs.SD;  eess.AS

- **最新发布 26 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] cSTMM: A Unified Complex Spherical Student's $t$ Mixture Model for Directional Statistics in Mask-Based Blind Speech Separation
- **分类: eess.AS**

- **简介: 该论文属于语音分离任务，解决盲源分离中方向统计建模问题。提出cSTMM模型，统一多种混合模型，提升分离性能。**

- **链接: [https://arxiv.org/pdf/2605.25512](https://arxiv.org/pdf/2605.25512)**

> **作者:** Nobutaka Ito
>
> **摘要:** Mask-based blind speech separation (BSS) estimates source-wise time-frequency (TF) masks by clustering multichannel observations using spatial information. The directional statistical approach clusters normalized multichannel observations on the complex unit sphere, without explicitly extracting phase and level difference features based on the plane-wave or spherical-wave assumptions. However, prior studies have mostly compared a small number of separately defined directional statistical mixture models, whereas a broader distribution family would enable a more systematic study of how density profiles affect separation performance. We propose the complex spherical Student's t mixture model (cSTMM), a directional mixture model that connects the complex angular central Gaussian mixture model (cACGMM), complex Bingham mixture model (cBMM), and complex Watson mixture model (cWMM) through the degrees-of-freedom parameter $\nu$. We also derive a generalized minorization-maximization (MM) based procedure for parameter estimation. A no-restart evaluation on noise-free LibriSpeech mixtures reverberated with measured room impulse responses shows that a single development-selected value $\nu^\ast=1$ achieved higher test-set mean signal-to-distortion ratio improvements (SDRi) than the cACGMM-equivalent setting $\nu=M$ in all acoustic conditions, with an average condition-wise gain of 0.25dB. The experiments also numerically verify that the proposed formulation numerically recovers the cACGMM, cBMM, and cWMM cases.
>
---
#### [new 002] Score-Agnostic Structure Analysis in Large-Scale Performance Datasets
- **分类: cs.SD**

- **简介: 该论文属于音乐信息检索任务，旨在解决大尺度转录数据中结构差异问题。通过序列对齐与聚类，实现无乐谱的结构分析与分组。**

- **链接: [https://arxiv.org/pdf/2605.25951](https://arxiv.org/pdf/2605.25951)**

> **作者:** Patricia Hu; Silvan Peter; Gerhard Widmer
>
> **备注:** published at the Music Encoding Conference (MEC) 2026
>
> **摘要:** In recent years, thanks to advances in automatic music transcription (AMT), several large-scale datasets of automatically transcribed piano solo music have been released. While these datasets undoubtedly offer extensive material for performance studies, they vary substantially in quality. In the case of classical music, performances often differ not only in expressive aspects such as tempo, but also in their structural interpretation of the score (including repeat patterns and edition-specific variants). To meaningfully use large-scale transcribed datasets for performance research, transcriptions of the same piece must be grouped according to their underlying structural realisation to support valid comparison. We address this by applying sequence-to-sequence alignment followed by hierarchical clustering: we create pairwise alignments for all pairs of transcriptions of a given piece, and use the alignment cost and (dis)similarity of performed sequence lengths to resolve structural mismatches as features for grouping. We propose this approach as a first step towards automatically evaluating large-scale transcribed datasets that lack ground-truth score and/or audio, shifting the evaluation criterion from truth-based accuracy to musical coherence and plausibility. We demonstrate our score-agnostic approach on around 1,500 transcriptions of 88 compositions from a recently published large-scale transcribed piano performance dataset.
>
---
#### [new 003] WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders With Residual Denoising and Sub-Modeling for GAN and Diffusion Models
- **分类: eess.AS**

- **简介: 该论文提出WaveNeXt 2，解决神经声码器在GAN和扩散模型中性能不足的问题，通过残差去噪和子模型优化，提升多说话人场景下的速度与质量。**

- **链接: [https://arxiv.org/pdf/2605.25506](https://arxiv.org/pdf/2605.25506)**

> **作者:** Wangzixi Zhou; Takuma Okamoto; Yamato Ohtani; Sakriani Sakti; Hisashi Kawai
>
> **备注:** ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
> **摘要:** Most neural vocoders are limited to one type: either GAN or diffusion-based. While state-of-the-art models like Vocos and WaveNeXt use powerful ConvNeXt-based generators, they have only been used in GAN frameworks and have limited performance in multi-speaker settings. Moreover, diffusion models, despite training faster than GANs, have slow CPU inference. In this paper, we introduce WaveNeXt 2, a unified ConvNeXt-based framework compatible with both GAN and diffusion vocoders. Its core innovation is residual denoising and sub-modeling, where each sub-model progressively refines the waveform. Experimental results in the multi-speaker dataset demonstrate the effectiveness of our approach: (1) GAN-WaveNeXt 2 is much faster than HiFi-GAN and WaveFit, and (2) Diff-WaveNeXt 2 also delivers much faster inference and competitive synthesis quality compared with FastDiff with 4 steps. The Diff-WaveNeXt 2 is very training-efficient, training in only 32 hours, making it ideal for resource-constrained applications.
>
---
#### [new 004] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations
- **分类: eess.AS**

- **简介: 该论文属于零样本文本到语音任务，旨在解决风格与音色分离控制的问题。通过引入双参考语音的框架FC-TTS，实现风格与音色的独立精确控制。**

- **链接: [https://arxiv.org/pdf/2605.24618](https://arxiv.org/pdf/2605.24618)**

> **作者:** Yoonhyung Lee; Hyunsin Park; Jinhwan Park; Jinkyu Lee
>
> **备注:** Accepted to ACL 2026 (Main Conference). 20 pages, 8 figures, 7 tables. Demo page: this https URL
>
> **摘要:** Recent advances in zero-shot text-to-speech (TTS) have enabled accurate imitation of reference speech in terms of both speaking style and speaker timbre. However, achieving disentangled control over these aspects from separate references remains a challenging task. Several studies have proposed disentangled speech representations that decompose speech into interpretable attributes (e.g., timbre, prosody, and content), providing a promising foundation for TTS with attribute control from separate references. Yet, how to effectively integrate such representations into TTS systems to achieve independent and precise control remains underexplored. In this paper, we present FC-TTS, a zero-shot TTS framework that enables disentangled control of style and timbre by conditioning on two distinct reference utterances. Unlike existing systems that inherit limitations from those pre-trained disentangled representations, FC-TTS introduces key design strategies, including architectural choices, training framework, and auxiliary training objectives, which improve the reliability of attribute separation and dual-reference control. Experiments show that FC-TTS achieves high-fidelity synthesis and competitive zero-shot naturalness, while uniquely supporting consistent and independent manipulation of style and timbre. Audio samples are available at this https URL
>
---
#### [new 005] Continual Speaker Identity Unlearning with Minimal Interference
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成中的说话人身份消除任务，解决连续移除说话人特征时导致之前移除的说话人信息被恢复的问题。提出CORTIS框架，实现持续无干扰的说话人身份遗忘。**

- **链接: [https://arxiv.org/pdf/2605.25962](https://arxiv.org/pdf/2605.25962)**

> **作者:** Jinju Kim; Yunsung Kang; Gyeong-Moon Park; Jong Hwan Ko
>
> **备注:** preprint
>
> **摘要:** Machine unlearning removes designated concepts or knowledge from pre-trained models. Recent work has extended this paradigm to speaker identity unlearning in zero-shot text-to-speech (ZS-TTS), the task of selectively erasing a model's ability to replicate a speaker's voice. Existing methods, however, quietly assume all unlearning requests arrive at once; an unrealistic assumption, since privacy-motivated removals arrive sequentially over time. We show this assumption breaks state-of-the-art methods: unlearning each new speaker fully revives previously unlearned speakers, reintroducing the very privacy risk unlearning was meant to eliminate. We present Cumulative ORThogonal Identity Suppression (CORTIS), the first framework for continual speaker identity unlearning in ZS-TTS that requires no access to previously-unlearned speaker data. CORTIS combines Fisher-information-based parameter masking, which localizes updates to speaker-relevant weights, with orthogonal projection against subspaces spanned by prior unlearning updates. With VoiceBox, CORTIS unlearns each requested speaker while keeping previously unlearned speakers forgotten across long request sequences, substantially outperforming sequential application of prior methods. The demo is available at this https URL .
>
---
#### [new 006] Subspace Track-before-Detect for Passive Multi-Target Tracking with Unknown Emitted Signals
- **分类: eess.AS**

- **简介: 该论文属于被动多目标跟踪任务，解决未知信号干扰下的跟踪问题。提出子空间TBD方法，无需建模信号，通过子空间对齐提高跟踪精度。**

- **链接: [https://arxiv.org/pdf/2605.25498](https://arxiv.org/pdf/2605.25498)**

> **作者:** Nobutaka Ito; Yoshiaki Bando
>
> **摘要:** Passive multi-target tracking (MTT) aims to infer the kinematic states of multiple targets from noisy sensor data in which contributions from unknown target-emitted signals are superposed. Track-before-detect (TBD) methods improve robustness to noise by operating directly on raw sensor data without relying on a preceding detection stage. However, many existing TBD methods assume that each target's contribution to the sensor data is determined solely by its kinematic state. This assumption limits their applicability to passive MTT, where each target's contribution depends on both its kinematic state and the unknown emitted signal. We propose subspace TBD, a passive multi-target TBD method based on a likelihood derived from the complex Bingham distribution that does not require explicit modeling or estimation of the unknown emitted signals. In a particle filter (PF) framework, each multi-target hypothesis is mapped to a low-dimensional subspace spanned by the steering vectors corresponding to the hypothesized target states. The likelihood is then used to evaluate the alignment of the normalized multichannel sensor data with this subspace. Preliminary experiments with simulated acoustic measurements and a given target activity pattern show that the proposed method can track two moving targets emitting unknown signals at a signal-to-noise ratio (SNR) of -10dB, whereas a conventional TBD baseline yields substantially larger tracking errors.
>
---
#### [new 007] Rethinking Continual Learning for Speech and Audio: A Representation-Centric Taxonomy and Open Problems
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音与音频领域的持续学习任务，旨在解决基础模型在非平稳环境中的表征保持与演化问题。提出以表征为中心的分类框架，识别现有方法与模型行为的不匹配，指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2605.24863](https://arxiv.org/pdf/2605.24863)**

> **作者:** Yang Xiao; Siyi Wang; Eun-Jung Holden; Ting Dang
>
> **备注:** 4 pages, 1 figure, working in process
>
> **摘要:** Speech and audio systems operate in inherently non-stationary environments, yet continual learning (CL) research in this domain, especially in the foundation model era, remains fragmented that fail to account for the coupled, geometry-sensitive nature of acoustic representations. Modern speech foundation models operate over highly entangled, continuous representations that jointly encode linguistic, speaker, and paralinguistic factors within a shared latent space. CL is therefore fundamentally about preserving and evolving shared representation structure rather than retaining isolated task knowledge. In this work, we revisit CL for speech from a representation-centered perspective, and introduce a new taxonomy that organizes CL according to how underlying representation geometry evolves under non-stationary acoustic conditions. We further identify key mismatches between current CL assumptions and speech foundation model behavior, and finally outline a set of open challenges and future research directions.
>
---
#### [new 008] PiAnnotate: A Web Annotation Tool for Piano Fingering, with a Diagnostic Probe
- **分类: cs.SD**

- **简介: 该论文提出PiAnnotate工具，用于标注钢琴演奏的指法。解决指法标注困难的问题，整合多源数据并保留标注过程，便于分析与学习。**

- **链接: [https://arxiv.org/pdf/2605.23982](https://arxiv.org/pdf/2605.23982)**

> **作者:** Joonhyung Bae; Kirak Kim; Hyeyoon Cho; Sein Lee; Yoon-Seok Choi; Hyeon Hur; Gyubin Lee; Akira Maezawa; Jonghwa Park; Jaebum Park; Juhan Nam
>
> **摘要:** Piano fingering shapes how a passage can be played, yet it is difficult to label after a performance. An annotator must decide which finger produced each note while reconciling the score, timing, video, and hand motion. We present PiAnnotate, a web-based pipeline for adding expert fingering annotations to the FurElise performance dataset. The tool brings together a piano-roll view, performance video, and a 3D MANO hand mesh so that reviewers can inspect each assignment in musical and physical context. Rather than storing only the final answer, PiAnnotate keeps paired rule-based and human-edited fingering tracks. These paired tracks make the annotation history auditable by showing where a geometric rule was sufficient, where experts intervened, and how labels changed across review passes. As a final diagnostic, we train a small Transformer probe on the paired tracks. The probe improves on the rule baseline on held-out pieces while remaining conservative about changing labels that were already correct, suggesting that the edited labels contain learnable structure rather than only isolated fixes.
>
---
#### [new 009] Toward Natural Emotional Text-To-Speech System with Fine-Grained Non-Verbal Expression Control
- **分类: eess.AS**

- **简介: 该论文属于情感文本到语音合成任务，旨在解决非语言表达控制不足的问题。通过构建标注数据集和新标注方案，提升情感表达的准确性与自然度。**

- **链接: [https://arxiv.org/pdf/2605.25504](https://arxiv.org/pdf/2605.25504)**

> **作者:** Wangzixi Zhou; Bagus Tris Atmaja; Sakriani Sakti
>
> **备注:** 2025 28th Conference of the Oriental COCOSDA International Committee for the Co-ordination and Standardisation of Speech Databases and Assessment Techniques (O-COCOSDA)
>
> **摘要:** While current emotional Text-to-Speech (TTS) models have successfully controlled verbal prosody, they often ignore non-verbal vocalizations (NVs), which are essential for authentic human emotion. Although some non-verbal datasets have recently emerged, they often lack high-quality, fine-grained annotations, which restricts a model's ability to precisely control NV generation. To address this limitation, we propose a novel approach for fine-grained non-verbal expression synthesis. We curate and reprocess female NV utterances from the EARS corpus, develop a new annotation scheme using tags to encode NV types, frequencies, and durations, and build an emotional TTS benchmark to demonstrate its effectiveness. Our evaluation shows that while our NV approach leads to minor trade-offs in perceived naturalness, it significantly improves expressiveness (eMOS 4.20) and emotional recognition accuracy (78.8%). Emotion-specific analysis further reveals that NV cues are highly effective for high-arousal emotions like happy (82.5%) and fear (82.7%), and almost perfectly convey sadness (98.3%).
>
---
#### [new 010] Zero-Shot Parkinson's Disease Detection from Speech: Comparing Large Audio and Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于帕金森病检测任务，研究不同音频输入方式对零样本检测效果的影响。比较了手工特征与原始波形输入，在多语言数据集上验证性能差异。**

- **链接: [https://arxiv.org/pdf/2605.24806](https://arxiv.org/pdf/2605.24806)**

> **作者:** Muhammad Ashad Kabir; Sirajam Munira
>
> **备注:** 6 pages
>
> **摘要:** Large audio and language models have recently demonstrated zero-shot reasoning capabilities across various domains. However, it remains unclear how the form of audio input, whether handcrafted acoustic features extracted from speech or the raw audio waveform itself, affects performance for Parkinson's disease (PD) detection across different languages. In this study, we systematically compare two input modalities for zero-shot PD detection: (i) handcrafted acoustic features extracted from speech recordings analyzed by a general-purpose LLM, and (ii) direct waveform input analyzed by audio-capable models. Experiments on PD speech datasets in four languages show that performance varies across input modalities, speech tasks, and languages. Handcrafted acoustic features provide more stable performance in a low-resource language (e.g., Bengali), whereas audio input yields dataset-dependent gains. These findings highlight the impact of input modality on zero-shot PD detection from speech.
>
---
#### [new 011] CosyEdit2: Speech-Editing-Oriented Reinforcement Learning Unlocks Better Zero-Shot TTS
- **分类: cs.SD**

- **简介: 该论文属于语音编辑与零样本TTS任务，解决数据不足和优化信号粗糙的问题。提出CosyEdit2模型，通过两阶段训练提升编辑性能和零样本TTS能力。**

- **链接: [https://arxiv.org/pdf/2605.25930](https://arxiv.org/pdf/2605.25930)**

> **作者:** Junyang Chen; Yuhang Jia; Hui Wang; Jiaming Zhou; Yongchang Gan; Yong Qin
>
> **摘要:** Speech editing and zero-shot Text-to-Speech (TTS) share a similar generative foundation conditioned on speech prompts, yet speech editing demands far stricter local acoustic consistency with surrounding unedited content. While prior work has shown that Supervised Fine-Tuning (SFT) enables TTS models to acquire functional editing capability, this approach remains fundamentally bottlenecked by imperfect paired editing data and coarse-grained optimization signals. To address these limitations, we propose CosyEdit2, a speech editing model built on a two-stage post-training framework that progresses from supervised editing initialization to editing-oriented Group Relative Policy Optimization (GRPO) over target-speech-free data. Extensive experiments demonstrate that CosyEdit2 not only substantially advances speech editing performance, but also unlocks better zero-shot TTS capability, revealing a deeper mutual relationship between the two tasks. Audio samples are available at this https URL.
>
---
#### [new 012] Decoding Stimulus Reconstruction-Based Auditory Attention Robustly in Unbalanced EEG Datasets
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于脑信号解码任务，旨在解决不平衡EEG数据集对刺激重构型听觉注意力解码性能的高估问题，提出LOPEO验证方法以提高评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.25605](https://arxiv.org/pdf/2605.25605)**

> **作者:** Yuanming Zhang; Yayun Liang; Zhibin Lin; Jing Lu
>
> **摘要:** In the past decade, numerous studies have applied deep neural networks (DNNs) to decode auditory attention (AAD) from Electroencephalogram (EEG) signals via stimulus reconstruction. However, the influence of dataset balance on the decoding performance of stimulus reconstruction-based AAD remains unexplored. In this study, three publicly available EEG-AAD datasets - KUL, DTU, and NJU cEEGrid - are used to construct both balanced and unbalanced experimental conditions. We hypothesize and demonstrate that stimulus reconstruction-based DNN decoders tend to produce overestimated decoding performance on unbalanced datasets. To address this issue, we propose a leave-one-paired-envelope-out (LOPEO) cross-validation protocol. Experimental results confirm that LOPEO effectively prevents inflated decoding accuracy on unbalanced datasets. While balanced datasets are generally preferred in experimental design, LOPEO provides a principled evaluation framework for unbalanced datasets that have already been published, filling an important gap in the field.
>
---
#### [new 013] Rubato: Transcribing Piano Music with Timestamps
- **分类: cs.SD; cs.CL; cs.MM**

- **简介: 该论文属于音乐转录任务，解决将音频转换为带时间戳的乐谱问题。提出Rubato模型和InterMo表示，提升转录准确性。**

- **链接: [https://arxiv.org/pdf/2605.24291](https://arxiv.org/pdf/2605.24291)**

> **作者:** Nazif Can Tamer; Victoria Ebert; Guang Yang; Noah A. Smith
>
> **备注:** 18 pages, 7 figures, 5 tables
>
> **摘要:** We consider the conversion of musical recordings into human-readable sheet music annotated with timestamps. Such output lets a listener clearly visualize rubato (temporally expressive playing), a learner diagnose ensemble precision and timing choices against the written music, and a musicology scholar compare performance styles across recordings of the same work. We introduce (1) a prompt-conditioned encoder-decoder model, named Rubato, trained to output (2) a new textual representation for polyphonic music, named InterMo, which we designed for compatibility with sequence-to-sequence training. Our experiments demonstrate that Rubato produces timestamped piano sheet music from audio with higher notational accuracy than the best existing approaches, which are based on cascades. We find that even if the cascade is given ground-truth MIDI instead of audio, Rubato performs better, suggesting that the ceiling of existing approaches is primarily representational, not acoustic. Further, because Rubato is trained on several related tasks (with prompts), it competes with or outperforms the best single-task systems on related but simpler tasks like MIDI note grounding and beat/downbeat detection. A demo is available at this https URL .
>
---
#### [new 014] A Multimodal Framework for Dementia Detection via Linguistic and Acoustic Representation Learning
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于 dementia 检测任务，旨在通过语音和文本的多模态分析早期识别阿尔茨海默病。工作包括提出一种联合学习框架，融合语音与文本特征，并引入互信息最大化机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.25540](https://arxiv.org/pdf/2605.25540)**

> **作者:** Loukas Ilias; Dimitris Askounis
>
> **摘要:** Alzheimer's disease (AD) is a progressive neurodegenerative disorder and the leading cause of dementia, affecting memory, reasoning, communication, and daily functioning. Early diagnosis is particularly important, as timely intervention may help slow cognitive decline and improve patient care. Recent studies have demonstrated that spontaneous speech contains valuable linguistic and acoustic biomarkers associated with dementia. However, existing approaches often rely on independently trained modality-specific models, feature concatenation strategies, ensemble methods, or attention-based fusion mechanisms that do not explicitly maximize the dependency between speech and transcript representations. In this work, we propose a multimodal deep learning framework for automatic dementia detection that jointly exploits speech and transcript information in an end-to-end trainable manner. Specifically, speech recordings are divided into 10-second segments and passed through a pre-trained HuBERT model to extract contextualized acoustic representations. To better capture informative temporal speech characteristics, attentive statistics pooling is employed to aggregate frame-level acoustic embeddings. For the textual modality, transcripts are encoded using a pre-trained BERT model, where the [CLS] token representation is used as the linguistic embedding. The acoustic and textual representations are subsequently combined using an attention-based Audio-Text Fusion (AT-Fusion) mechanism. In addition, we introduce a MINE objective to maximize the mutual information between modalities and improve multimodal representation alignment. The fused multimodal representation is finally used for dementia classification. Experiments conducted on the publicly available ADReSS Challenge and PROCESS-2 dataset demonstrate the effectiveness and robustness of the proposed approach for speech-based dementia assessment.
>
---
#### [new 015] Music Transcription with (Almost) No Supervision
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐转录任务，解决标注数据稀缺问题。通过使用少量配对数据和大量未配对音频与乐谱，提升转录效果，验证了未配对数据的有效性。**

- **链接: [https://arxiv.org/pdf/2605.24193](https://arxiv.org/pdf/2605.24193)**

> **作者:** Saebyeol Shin; Chao Wan; Zhenzhen Liu; Justin Lovelace; Daniel C. Lin; Kilian Q. Weinberger; John Thickstun
>
> **摘要:** Competitive music transcription models require large amounts of paired audio-score data, which is scarce due to collection costs, alignment difficulty, and copyright restrictions. Meanwhile, vast quantities of unpaired audio recordings and symbolic scores are freely available but have gone unused. We adopt a cycle-consistent translation framework in which a small amount of paired data acts as a minimal anchor, unlocking the full potential of the unpaired pool. We find that: unpaired data yields surprisingly large gains, especially under limited supervision; unpaired audio contributes more than unpaired scores; incorporating unlabeled audio from a new instrument during training improves transcription for that instrument without any paired supervision. Together, these results suggest that scaling unpaired data offers a practical path toward high-quality transcription for instruments where labeled data remains scarce.
>
---
#### [new 016] Ultra-Low-Bitrate Mel-Spectrogram-based Neural Speech Coding with Flow-Matching-based Refinement and Vocoding-driven Reconstruction
- **分类: eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决超低比特率下语音自然度和说话人识别的问题。提出FMelCodec框架，通过编码-精炼-重建三阶段提升语音质量与相似度。**

- **链接: [https://arxiv.org/pdf/2605.25669](https://arxiv.org/pdf/2605.25669)**

> **作者:** Hui-Peng Du; Yang Ai; Xiao-Hang Jiang; Yuan Tian; Zhen-Hua Ling
>
> **备注:** Published at IEEE/ACM Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Ultra-low-bitrate speech coding is pivotal for bandwidth-constrained communication and deep compression, yet maintaining naturalness and speaker identity at such extreme bit budgets remains challenging due to pronounced information loss and quantization instability. To this end, we propose FMelCodec, an ultra-low-bitrate neural speech codec in the mel-spectrogram domain, cast as a three-stage coding-refinement-reconstruction (CRR) framework that can operate at as low as 250 bps. In the CRR framework, the front-end mel-spectrogram coding stage employs a highly aggressive 640x compression/decompression encoder-decoder structure with a single 1024-entry VQ codebook, coupled with an online clustering strategy that reassigns underused codewords to prevent codebook collapse and preserve codebook diversity. The subsequent conditional flow matching (CFM)-based mel-spectrogram refinement stage leverages a lightweight velocity-field estimator and CFM-based solver to refine the codec-degraded mel-spectrogram produced by the preceding decoder, and adopts a self-consistency training scheme that supports fewer iterative inference steps for the purpose of reducing computational overhead. Finally, the vocoding-driven waveform reconstruction stage employs a HiFi-GAN vocoder to faithfully reconstruct waveform from the refined mel-spectrogram. Experiments conducted on two datasets spanning two sampling rates show that, under ultra-low-bitrate constraints of 250 bps for 16 kHz and 750 bps for 48 kHz, both objective and subjective evaluations consistently demonstrate that FMelCodec achieves higher speech reconstruction quality and speaker similarity, while incurring lower computational and model complexity.
>
---
#### [new 017] Time Segmented Beamforming via Dynamic Programming: Theory and Implementation
- **分类: eess.SP; cs.SD; eess.AS; eess.SY; math.OC**

- **简介: 该论文属于信号处理任务，旨在解决动态环境中干扰源变化导致的波束成形性能下降问题。通过引入时间分段的无失真响应波束成形方法，动态调整协方差矩阵估计窗口，提升对时变干扰的跟踪能力。**

- **链接: [https://arxiv.org/pdf/2605.24825](https://arxiv.org/pdf/2605.24825)**

> **作者:** Manan Mittal; Ryan M. Corey; Diego Cuji; John R. Buck; Andrew C. Singer
>
> **备注:** 16 pages, 17 figures, Beamforming New Approach Regret Bounds
>
> **摘要:** In dynamic acoustic environments with time-varying interferers, effective beamforming requires identifying stationary regions over time. The Capon beamformer, a whitened matched filter constrained to maintain unity gain in the desired direction, theoretically relies on the instantaneous ensemble covariance matrix. Practical implementations rely on the batch Capon (or Sample Matrix Inversion), which estimates the sample covariance matrix (SCM) by averaging over a block of snapshots. This practical approach implicitly assumes that the data within the batch window is stationary and can be coherently combined. In non-stationary settings, a batch approach that averages over fixed or excessively long windows fails, as moving interferers smear the SCM and degrade the beamformer's nulling capabilities. To address this, this paper introduces a temporally segmented distortionless response beamformer. Inspired by the segmented least squares method, which fits piecewise polynomials to data while penalizing excessive segmentation to prevent overfitting, the framework extends practical Capon beamforming by incorporating data-driven temporal segmentation. This formulation minimizes output power while dynamically adapting the SCM estimation windows to local stationarity, offering a principled approach to tracking time-varying interferers.
>
---
#### [new 018] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于心理健康评估任务，旨在通过语音特征分析支持临床决策。研究提取语音的声学和语言特征，结合机器学习方法，探索其与抑郁、焦虑等症状的关系。**

- **链接: [https://arxiv.org/pdf/2605.24678](https://arxiv.org/pdf/2605.24678)**

> **作者:** Vassilis Lyberatos; Edmund G. Dervakos; Eleni Adamidi; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted to CLPsych 2026, part of ACL 2026
>
> **摘要:** Speech and language technologies offer valuable opportunities for supporting mental health assessment through objective and interpretable cues. We present a systematic feature-based analysis framework leveraging perceptually grounded acoustic and linguistic characteristics, including prosody, vocal quality, semantic coherence, syntactic structure, and sarcasm. Using statistical analysis and interpretable machine learning (XGBoost with SHAP and LIME), we examine associations between speech features and validated symptom measures of depression, anxiety, and ADHD. Evaluated on both controlled benchmark datasets (StressID, DAIC-WOZ, Androids, EATD) and a real-world clinical dataset, the framework reveals stable and consistent relationships between symptom severity and vocal irregularities (e.g., shimmer, jitter), lexical-syntactic patterns, and affective tone. An ablation study conducted across all datasets further identifies the most informative feature groups. This work explores a transparent and clinically interpretable approach to speech-based mental health analysis.
>
---
#### [new 019] Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决英语-汉语代码切换转写问题。通过DPO方法训练模型，提升其正确保留混合语言的能力，减少转写错误。**

- **链接: [https://arxiv.org/pdf/2605.23975](https://arxiv.org/pdf/2605.23975)**

> **作者:** Trung Nguyen Quang; Cheng Yi Lewis Won; Minh Duc Pham; Yingxu He; Shuo Sun; Ai Ti Aw
>
> **摘要:** Audio large language models (Audio LLMs) exhibit systematic failures in transcribing code-switching speech despite strong multilingual capabilities. Focusing on English-Mandarin, we identify three failure modes: language omission, translation-instead-of-transcription, and hallucination. We apply Direct Preference Optimization (DPO) to align models, constructing preference pairs in which chosen responses preserve mixed-language content while rejected responses mimic failure patterns. Training three Audio LLMs on 100K pairs (570 hours), we observe consistent behavioral shifts: models learn to preserve language composition rather than translating when prompted for transcription. This alignment yields MER reductions up to 89.6% (in-distribution) and 20.0% (out-of-distribution). Our findings suggest DPO can effectively elicit correct code-switching transcription behavior from multilingual Audio LLMs.
>
---
#### [new 020] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于音频水印任务，解决合成音频的溯源问题。提出一种无需梯度、鲁棒的水印方法，通过优化词汇提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.25967](https://arxiv.org/pdf/2605.25967)**

> **作者:** Georgios Milis; Yubin Qin; Yihan Wu; Heng Huang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** As policy catches up with the capabilities of generative AI, watermarking is central to content provenance efforts. Inference-time watermarks for autoregressive models are unfit for continuous modalities due to discretization inconsistencies. Existing methods overcome this by finetuning the modality tokenizers, nullifying the watermark's training-free advantage. In this work, motivated by the vocabulary redundancy of discretization, we propose an elegant solution for powerful and robust watermarking of synthetic audio. We theoretically analyze the impact of token errors on watermark detection, and effectively mitigate them using a reduced vocabulary obtained via community detection. Thorough experiments showcase that our gradient-free method can boost detectability by several orders of magnitude, while also achieving built-in robustness to audio modifications. Broadly, we discover a new state-of-the-art for token-level watermarks in multimedia, which simply arises from the nature of discrete representation learning.
>
---
#### [new 021] Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决误差传播问题。通过因果感知的错误诊断与交互澄清，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.25404](https://arxiv.org/pdf/2605.25404)**

> **作者:** Yizhou Peng; Ziyang Ma; Changsong Liu; Yi-Wen Chao; Xie Chen; Eng Siong Chng
>
> **摘要:** Cascaded Automatic Speech Recognition -- Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that exploit deep ASR latent representations to disentangle token-level errors into perception, comprehension, and deletion failures. This fine-grained diagnostic intelligence empowers the LLM to orchestrate targeted, multi-turn clarification strategies, effectively transforming ambiguous signals into seamless user interactions. Experimental results validate the precision of our approach, which more than doubles the recall on domain-shift errors (57.96% vs. 23.66%) compared to baselines. Crucially, this diagnostic precision yields up to a 30% reduction in WER and a 17% improvement on the downstream task across diverse accents, distortions, and domains.
>
---
#### [new 022] EchoDistill:Alignment Noisy-to-Clean Self-Distillation for Robust Audio LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于音频大模型鲁棒性提升任务，解决噪声环境下语义漂移问题。提出EchoDistill框架，通过自蒸馏增强模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.23954](https://arxiv.org/pdf/2605.23954)**

> **作者:** Liang Lin; Chunxi Luo; Kaiwen Luo; Jie Zhang; Jin Wang; Yuanhe Zhang; Cai Yuchen; Qiankun Li; Gongli Xi; Zhenhong Zhou; Kun Wang; Junhao Dong
>
> **摘要:** Audio Large Language Models (ALLMs) are highly vulnerable to real-world noise, which often induces severe semantic drift and hallucinations. Existing robustness methods primarily rely on waveform-level acoustic enhancement, answer-level supervision, or the internal suppression of noise representations. To address these issues, we propose echodistill, an alignment-based noisy-to-clean self-distillation framework. Echodistill leverages a frozen clean-audio teacher to provide semantic references for an inference-time noisy-audio student. Specifically, the student samples candidate responses under noisy conditions to expose its test-time behavior. These trajectories are then optimized via group-relative policy optimization (GRPO), where the token-level consistency with the teacher acts as a reward bonus. By aligning the noisy student's candidate responses with clean semantic evidence, and applying audio-aware reward shaping, our method encourages reasoning trajectories that are both correct and genuinely acoustically grounded. Echodistill significantly improves the semantic reliability and task performance of Audio LLMs under complex noise, without introducing any additional inference costs. Extensive experiments show that: (I) Compared with the strongest baseline, echodistill achieves average improvements of 4.18\%$\uparrow$ in GSR under strong noise. (II) Ablation results on Qwen-Omni further show that echodistill improves over the GRPO-only variant by 3.02\%$\uparrow$ in Acc, 3.89\%$\uparrow$ in Noisy, and 4.53\%$\uparrow$ in GSR on average. Our codes are available at this https URL.
>
---
#### [new 023] Thaka at KSAA-2026 Task 2: Regularized Fine-Tuning for Arabic Speech Diacritization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语语音加注任务，旨在从语音和无注音文本中生成完整注音文本。工作包括使用CATT-Whisper模型并结合正则化技术提升性能。**

- **链接: [https://arxiv.org/pdf/2605.25928](https://arxiv.org/pdf/2605.25928)**

> **作者:** Meshal Alamr; Hassan Alqaeri; Abdullah Aldahlawi
>
> **备注:** 4 pages, 1 figure. Published in Proceedings of OSACT7 (LREC 2026). Winning system for KSAA-2026 Task 2 on Arabic Speech Diacritization
>
> **摘要:** We describe the winning system for Task 2 of the KSAA-2026 Shared Task on Arabic Speech Dictation with Automatic Diacritization. The task requires producing fully diacritized Arabic text from speech audio and undiacritized transcripts, with only 2,327 training samples available and no external data permitted. Our system fine-tunes CATT-Whisper, a character-level multimodal model combining a pretrained CATT text encoder with a frozen Whisper speech encoder. The key to our approach is training regularization: R-Drop consistency regularization, Optuna-optimized hyperparameters with high weight decay, and Focal Loss. At inference, we average 200 stochastic forward passes across four model checkpoints using Monte Carlo Dropout at the softmax probability level. The system achieves 23.26% WER on the primary leaderboard metric (with case endings, including no-diacritic positions), placing 1st among all participants.
>
---
#### [new 024] Raon-Speech Technical Report
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出Raon-Speech和Raon-SpeechChat，解决语音理解和生成任务，通过多阶段训练提升模型性能并支持实时对话。**

- **链接: [https://arxiv.org/pdf/2605.23912](https://arxiv.org/pdf/2605.23912)**

> **作者:** Beomsoo Kim; Changho Choi; Dohyun Kim; Dongki Lee; Ethan Ewer; Eunchong Kim; Gyeongman Kim; Haechan Kim; Hyeonghwan Kim; Inkyu Park; Jihun Yun; Jihwan Moon; Jiyun Kim; Joonghyun Bae; Junhyuck Kim; Minkyu Kim; Sehun Lee; Seungjun Chung; Sungwoo Cho; Dongmin Park; Dongwon Kim; Hara Kang; Jonghyun Lee; Keon Lee; Kangwook Lee; Jaewoong Cho
>
> **摘要:** We present Raon-Speech, a top-performing 9B-parameter speech language model (SpeechLM) for English and Korean speech understanding, answering, and generation, and Raon-SpeechChat, a high-performing full-duplex extension for natural real-time conversation. Raon-Speech successfully transforms a pre-trained LLM into a SpeechLM that both understands and generates speech while preserving strong text capabilities. It trains on 1.38M hours of highly curated English and Korean speech and text datasets with the following training stages: (1) speech modules alignment, (2) end-to-end SpeechLM pre-training with knowledge distillation, and (3) multi-task preference optimization-based post-training. Across 42 English and Korean speech and text benchmarks, Raon-Speech establishes the strongest overall profile on speech-centric tasks in our comparison against eight similarly sized recent audio foundation models, including Qwen2.5-Omni and Fun-Audio-Chat, while preserving strong text question answering performance. Building upon it, Raon-SpeechChat enables natural full-duplex conversation by continual training on 119K hours of time-aligned real and synthetic dialogue data. It proceeds through three complementary training stages: (1) causal encoder adaptation, (2) full-duplex pre-training, (3) full-duplex fine-tuning for voice and role-control. On multiple full-duplex benchmarks, Raon-SpeechChat shows its clearest strengths on the turn-taking and interruption-sensitive behaviors covered by FDB v1.0, and remains competitive across the broader full-duplex evaluation suite. We open-source all model checkpoints, the training and inference pipeline, and an interactive demo.
>
---
#### [new 025] AVBench: Human-Aligned and Automated Evaluation Benchmark for Audio-Video Generative Models
- **分类: cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出AVBench，用于评估音视频生成模型。针对现有评估方法不精准的问题，设计了细粒度指标和专用评估器，实现自动化、可靠的人类对齐评价。**

- **链接: [https://arxiv.org/pdf/2605.24652](https://arxiv.org/pdf/2605.24652)**

> **作者:** Jialiang Yang; Bin Xia; Ruihang Chu; Dingdong Wang; Wanke Xia; Zhun Mou; Tianyang Zhong; Yiting Zhao; Wenming Yang
>
> **摘要:** Rapid advances in audio-video (AV) generation have enabled high-fidelity synthesis with synchronized sound, particularly for human-related scenarios involving speech and interactions. Yet evaluation for AV generation remains at an early stage, with only a few coarse-grained benchmarks for human-related scenarios and relying on limited preset evaluations with generic multimodal LLMs, leading to inaccurate assessments of model capabilities. To address these issues, we introduce AVBench, a fully automated benchmark tailored for human-centric AV generation. AVBench is built on two key designs for comprehensive and accurate evaluation: (i) Human-centric and fine-grained metrics. AVBench integrates ten evaluation dimensions designed for human-centered real-world scenarios, covering visual quality, audio quality, and multi-level consistency across modalities. These practical metrics capture human-related details that existing benchmarks often overlook. (ii) Specialized evaluators via preference learning. To address the lack of specialized training data, we construct large-scale supervision by transforming real-world videos into diverse training pairs with controlled perturbations. After fine-tuning on this high-quality dataset, the evaluators learn to reliably detect subtle cross-modal inconsistencies. Crucially, instead of producing discrete textual judgment, AVBench derives continuous evaluation scores from the model's prediction confidence on binary decisions. This probabilistic scoring mechanism enables a more reliable assessment than traditional VQA-style evaluation and aligns closely with human judgment. Taken together, AVBench offers automated evaluation for AV generation, demonstrates strong potential for data filtering, and serves as a differentiable reward signal for Reinforcement Learning from Human Feedback (RLHF).
>
---
#### [new 026] A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在评估临床访谈数据集的基准性能。通过多角度验证模型可靠性，发现现有评估存在偏差，模型在不同数据上的表现差异显著。**

- **链接: [https://arxiv.org/pdf/2605.23977](https://arxiv.org/pdf/2605.23977)**

> **作者:** Takehiro Ishikawa; Jon Duke
>
> **摘要:** This paper audits benchmark evaluation in clinical-interview depression detection through four complementary probes across DAIC/E-DAIC, CMDC, ANDROIDS, MODMA, and PDCH. First, we re-evaluate E-DAIC under strict subject-disjoint leave-one-subject-out cross-validation. A lightweight hybrid text-plus-LLM-score model reaches macro-F1 = 0.723 - the highest reported under this protocol, to our knowledge - providing a conservative out-of-fold reference point that does not depend on the privileged official holdout. Second, we test whether the E-DAIC official split supports fine-grained leaderboard rankings by sweeping 96 model configurations across modality bundles, pooling strategies, and learners. Development-side cross-validation and official-test rankings align only moderately: the best cross-validation configuration ranks twentieth on the official test, the official-test winner ranks forty-first by cross-validation, top-3 overlap is zero, and the apparent winner is rank-1 in only 32.3% of subject bootstraps. Third, we externally validate strong public CMDC and ANDROIDS baselines that achieve near-ceiling in-domain performance. Zero-shot transfer to external corpora is substantially weaker. Finally, we stress-test E-DAIC text and audio models using paired symptom-dense versus symptom-light interview slices defined by an SRDS-based annotator. Text scores rise sharply on symptom-dense slices, whereas audio scores remain nearly flat; the text-minus-audio gap is positive across all five seeds.
>
---
## 更新

#### [replaced 001] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出JAEGER框架，解决2D感知与3D环境不匹配的问题，通过融合RGB-D和多通道音频实现3D空间定位与推理。**

- **链接: [https://arxiv.org/pdf/2602.18527](https://arxiv.org/pdf/2602.18527)**

> **作者:** Zhan Liu; Changli Tang; Yuxin Wang; Zhiyuan Zhu; Youjun Chen; Yiwen Shao; Tianzi Wang; Lei Ke; Zengrui Jin; Chao Zhang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Current audio-visual large language models (AV-LLMs) are predominantly restricted to 2D perception, relying on RGB video and monaural audio. This design choice introduces a fundamental dimensionality mismatch that precludes reliable source localization and spatial reasoning in complex 3D environments. We address this limitation by presenting JAEGER, a framework that extends AV-LLMs to 3D space, to enable joint spatial grounding and reasoning through the integration of RGB-D observations and multi-channel first-order ambisonics. A core contribution of our work is the neural intensity vector (Neural IV), a learned spatial audio representation that encodes robust directional cues to enhance direction-of-arrival estimation, even in adverse acoustic scenarios with overlapping sources. To facilitate large-scale training and systematic evaluation, we propose SpatialSceneQA, a benchmark of 61k instruction-tuning samples curated from simulated physical environments. Extensive experiments demonstrate that our approach consistently surpasses 2D-centric baselines across diverse spatial perception and reasoning tasks, underscoring the necessity of explicit 3D modelling for advancing AI in physical environments. Our source code, pre-trained model checkpoints, and datasets are available at this https URL.
>
---
#### [replaced 002] RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于语音克隆任务，旨在评估模型在各种挑战下的鲁棒性。工作包括构建RVCBench数据集，测试不同场景下的模型表现，并揭示其潜在缺陷。**

- **链接: [https://arxiv.org/pdf/2602.00443](https://arxiv.org/pdf/2602.00443)**

> **作者:** Ruinan Jin; Xinting Liao; Hanlin Yu; Deval Pandya; Xiaoxiao Li
>
> **备注:** 65 pages, 10 figures
>
> **摘要:** Modern voice cloning, also known as zero-shot text-to-speech (TTS), can synthesize speech that closely matches a target speaker from only seconds of reference audio, enabling applications such as personalized speech interfaces and dubbing. In practice, these systems often face noisy reference audio, imperfect text prompts, multilingual and long-form generation, post-processing, and adversarial perturbations, all of which can weaken robustness. Despite rapid progress in codec-token language models and diffusion-based TTS, robustness under realistic deployment shifts remains underexplored. This paper introduces RVCBench, a comprehensive dataset and benchmark for evaluating robustness in voice cloning. RVCBench provides task-aligned tests covering controlled text-audio pairing, multilingual and long-form scenarios, expressive prompts, post-processing conditions, and passive or proactive audio perturbations. Across 18 robustness evaluations, 225 speakers, and 14,370 utterances, RVCBench supports unified evaluation of input sensitivity, generation stability, output resilience, perturbation robustness, speaker similarity, and deepfake detectability. We evaluate 18 representative open-source voice cloning models and reveal systematic vulnerabilities in content consistency, speaker similarity, long-form stability, post-processing resilience, adversarial robustness, and detector-facing separability. We release the code and dataset to support reproducible evaluation and future research on robust voice cloning, speech synthesis, and audio generation. Code: this https URL. Dataset: this https URL.
>
---
#### [replaced 003] Voice of India: A Large-Scale Benchmark for Real-World Speech Recognition in India
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决现有数据集存在的过拟合和拼写偏差问题。构建了包含15种印度语言的大规模真实语音数据集Voice of India，并分析了不同因素对ASR性能的影响。**

- **链接: [https://arxiv.org/pdf/2604.19151](https://arxiv.org/pdf/2604.19151)**

> **作者:** Kaushal Bhogale; Manas Dhir; Amritansh Walecha; Manmeet Kaur; Vanshika Chhabra; Aaditya Pareek; Hanuman Sidh; Mahima Manik; Sagar Jain; Bhaskar Singh; Utkarsh Singh; Tahir Javed; Shobhit Banga; Mitesh M. Khapra
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Existing Indic ASR benchmarks often use scripted, clean speech and leaderboard driven evaluation that encourages dataset specific overfitting. In addition, strict single reference WER penalizes natural spelling variation in Indian languages, including non standardized spellings of code-mixed English origin words. To address these limitations, we introduce Voice of India, a closed source benchmark built from unscripted telephonic conversations covering 15 major Indian languages across 139 regional clusters. The dataset contains 306230 utterances, totaling 536 hours of speech from 36691 speakers with transcripts accounting for spelling variations. We also analyze performance geographically at the district level, revealing disparities. Finally, we provide detailed analysis across factors such as audio quality, speaking rate, gender, and device type, highlighting where current ASR systems struggle and offering insights for improving real world Indic ASR systems.
>
---
#### [replaced 004] Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音编辑检测任务，旨在解决现有数据多样性不足和删除类编辑检测困难的问题。通过构建大规模数据集并引入生成式框架，实现检测与定位的统一。**

- **链接: [https://arxiv.org/pdf/2601.21463](https://arxiv.org/pdf/2601.21463)**

> **作者:** Jun Xue; Yi Chai; Yanzhen Ren; Jinshen He; Zhiqiang Tang; Zhuolin Yi; Yihuan Huang; Yuankun Xie; Yujie Chen
>
> **摘要:** Existing speech editing detection (SED) datasets are predominantly constructed using manual splicing or limited editing operations, resulting in restricted diversity and poor coverage of realistic editing scenarios. Meanwhile, current SED methods rely heavily on frame-level supervision to detect observable acoustic anomalies, which fundamentally limits their ability to handle deletion-type edits, where the manipulated content is entirely absent from the signal. To address these challenges, we present a unified framework that bridges speech editing detection and content localization through a generative formulation based on Audio Large Language Models (Audio LLMs). We first introduce AiEdit, this https URL, a large-scale bilingual dataset (approximately 140 hours) that covers addition, deletion, and modification operations using state-of-the-art end-to-end speech editing systems, providing a more realistic benchmark for modern threats. Building upon this, we reformulate SED as a structured text generation task, enabling joint reasoning over edit type identification, and content localization. To enhance the grounding of generative models in acoustic evidence, we propose a prior-enhanced prompting strategy that injects word-level probabilistic cues derived from a frame-level detector. Furthermore, we introduce an acoustic consistency-aware loss that explicitly enforces the separation between normal and anomalous acoustic representations in the latent space. Experimental results demonstrate that the proposed approach consistently outperforms existing methods across both detection and localization tasks.
>
---
#### [replaced 005] Go witheFlow: Real-time Emotion Driven Audio Effects Modulation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文介绍witheFlow系统，用于实时音乐表演中根据生物信号和音频特征自动调节音效。属于音乐与情感计算任务，解决机器缺乏情感表达的问题。**

- **链接: [https://arxiv.org/pdf/2510.02171](https://arxiv.org/pdf/2510.02171)**

> **作者:** Edmund Dervakos; Spyridon Kantarelis; Vassilis Lyberatos; Jason Liartis; Giorgos Stamou
>
> **备注:** Accepted at NeurIPS Creative AI Track 2025: Humanity
>
> **摘要:** Music performance is a distinctly human activity, intrinsically linked to the performer's ability to convey, evoke, or express emotion. Machines cannot perform music in the human sense; they can produce, reproduce, execute, or synthesize music, but they lack the capacity for affective or emotional experience. As such, music performance is an ideal candidate through which to explore aspects of collaboration between humans and machines. In this paper, we introduce the witheFlow system, designed to enhance real-time music performance by automatically modulating audio effects based on features extracted from both biosignals and the audio itself. The system, currently in a proof-of-concept phase, is designed to be lightweight, able to run locally on a laptop, and is open-source given the availability of a compatible Digital Audio Workstation and sensors.
>
---
#### [replaced 006] Focus Then Listen: Exploring Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文属于音频理解任务，旨在提升大音频语言模型在噪声环境下的鲁棒性。提出FTL音频增强器，通过分离语音与非语音信号并融合优化，提高模型性能，无需额外微调。**

- **链接: [https://arxiv.org/pdf/2603.04862](https://arxiv.org/pdf/2603.04862)**

> **作者:** Han Yin; Yang Xiao; Younghoo Kwon; Ting Dang; Jung-Woo Choi
>
> **摘要:** Large audio language models (LALMs) are a class of foundation models for audio understanding. Existing LALMs tend to degrade significantly in real-world noisy acoustic conditions where speech and non-speech sounds interfere. While noise-aware fine-tuning can improve robustness, it requires task-specific noisy data and expensive retraining, limiting scalability. To address this issue, we propose Focus-Then-Listen (FTL), a plug-and-play audio enhancer that improves LALMs' noise robustness. Specifically, FTL first separates the input waveform into speech and non-speech, and a modality router is applied to predict the target audio modality (e.g., speech) based on the user's instruction. Finally, a modality-aware fusion block generates a task-adaptive enhanced signal for improved downstream perception and reasoning. Experiments across multiple LALMs and tasks show that FTL improves performance across different noise levels without fine-tuning on LALMs.
>
---
#### [replaced 007] Decoding Speech Envelopes from Electroencephalogram with a Contrastive Pearson Correlation Coefficient Loss
- **分类: eess.AS**

- **简介: 该论文属于语音注意力解码任务，旨在提升多说话人环境下的听觉注意识别。通过引入对比PCC损失函数，优化模型区分关注与非关注语音的能力。**

- **链接: [https://arxiv.org/pdf/2601.20542](https://arxiv.org/pdf/2601.20542)**

> **作者:** Yayun Liang; Yuanming Zhang; Fei Chen; Jing Lu; Zhibin Lin
>
> **摘要:** Recent advances in reconstructing speech envelopes from Electroencephalogram (EEG) signals have enabled continuous auditory attention decoding (AAD) in multi-speaker environments. Most Deep Neural Network (DNN)-based envelope reconstruction models are trained to maximize the Pearson correlation coefficients (PCC) between the attended envelope and the reconstructed envelope (attended PCC). While the difference between the attended PCC and the unattended PCC plays an essential role in auditory attention decoding, existing methods often focus on maximizing the attended PCC. We therefore propose a contrastive PCC loss which represents the difference between the attended PCC and the unattended PCC. The proposed approach is evaluated on three public EEG AAD datasets using four DNN architectures. Across many settings, the proposed objective improves envelope separability and AAD accuracy, while also revealing dataset- and architecture-dependent failure cases.
>
---
#### [replaced 008] Diffusion-based Frameworks for Unsupervised Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决无监督环境下的语音清晰度提升问题。提出一种基于扩散模型的框架，显式建模语音和噪声，提升增强效果。**

- **链接: [https://arxiv.org/pdf/2601.09931](https://arxiv.org/pdf/2601.09931)**

> **作者:** Jean-Eudes Ayilo; Mostafa Sadeghi; Romain Serizel; Xavier Alameda-Pineda
>
> **摘要:** This paper addresses unsupervised diffusion-based single-channel speech enhancement (SE). Prior work in this direction combines a score-based diffusion model trained on clean speech with a Gaussian noise model whose covariance is structured by non-negative matrix factorization (NMF). This combination is used within an iterative expectation-maximization (EM) scheme, in which a diffusion-based posterior-sampling E-step estimates the clean speech. We first revisit this framework and propose to explicitly model both speech and acoustic noise as latent variables, jointly sampling them in the E-step instead of sampling speech alone as in previous approaches. We then introduce a new semi-supervised SE framework that replaces the NMF noise prior with a diffusion-based noise model, learned jointly with the speech prior in a single conditional score model. Within this framework, we derive two variants: one that implicitly accounts for noise and one that explicitly treats noise as a latent variable. Experiments on WSJ0-QUT and VoiceBank-DEMAND show that explicit noise modeling systematically improves SE performance for both NMF-based and diffusion-based noise priors. Under matched conditions, the diffusion-based noise model attains the best overall quality and intelligibility among unsupervised methods, while under mismatched conditions the proposed NMF-based explicit-noise framework is more robust and suffers less degradation than several supervised baselines. Code, demo, and supplementary materials are publicly available.
>
---
#### [replaced 009] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding
- **分类: eess.AS**

- **简介: 该论文提出PlanRAG-Audio，解决长音频理解任务中的挑战，通过规划与检索增强生成，提升推理准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.20414](https://arxiv.org/pdf/2605.20414)**

> **作者:** Masao Someki; Chien-yu Huang; Siddhant Arora; Samuele Cornell; Markus Müller; Nathan Susanj; Rupak V Swaminathan; Grant P Strimel; Jing Liu; Shinji Watanabe
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Long-form audio understanding poses significant challenges for large audio language models (LALMs) due to the extreme length of audio sequences and the need to reason over heterogeneous acoustic cues distributed over time, such as speech content, speaker identity, emotion, and sound events. To address these challenges, we propose \textbf{PlanRAG-Audio}, a planning-based retrieval-augmented generation framework for scalable long-form audio understanding. Rather than having audio LALMs process entire recordings directly, PlanRAG-Audio explicitly plans which modalities and temporal spans are required for a given query, and retrieves only query-relevant information from a structured text and audio database. This retrieval planning enables effective reasoning over complex, cross-domain audio queries while substantially reducing the input length passed to the large language models. Experiments across a wide range of speech/audio retrieval demonstrate that PlanRAG-Audio improves reasoning accuracy and stabilizes performance as audio duration increases by decoupling inference cost from raw audio length.
>
---
#### [replaced 010] On the Distillation Loss Functions of Speech VAE for Unified Reconstruction, Understanding, and Generation
- **分类: cs.SD**

- **简介: 该论文研究语音VAE的蒸馏损失函数，旨在提升语音重建、理解与生成的统一性能。针对现有对齐方法的不足，探索不同对齐策略并优化损失设计。**

- **链接: [https://arxiv.org/pdf/2604.12383](https://arxiv.org/pdf/2604.12383)**

> **作者:** Changhao Cheng; Wei Wang; Wangyou Zhang; Dongya Jia; Jian Wu; Zhuo Chen; Yanmin Qian
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Continuous speech representations based on Variational Autoencoders (VAEs) have emerged as a promising alternative to traditional spectrogram or discrete token based features for speech generation and reconstruction. Recent research has tried to enrich the structural information in VAE latent representations by aligning with self-supervised learning (SSL) features, aiming for better generation performance. However, it remains unclear whether the widely-used alignment approach based on time-axis distillation is optimal when considering more tasks. To address this problem, this paper systematically explores different alignment approaches and analyzes their impact on the performances over three axes: reconstruction, understanding, and generation. We investigate various design choices in the distillation loss. Extensive experiments show that the joint-marginal alignment approach with adaptive weighting can achieve the best overall performance while allowing for a controllable balance.
>
---
#### [replaced 011] Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于音频语言模型安全研究，解决如何高效进行越狱攻击的问题。通过分析梯度结构，提出稀疏优化方法TAGO，仅保留高梯度区域，提升攻击效率。**

- **链接: [https://arxiv.org/pdf/2605.04700](https://arxiv.org/pdf/2605.04700)**

> **作者:** Zheng Fang; Xiaosen Wang; Shenyi Zhang; Shaokang Wang; Zhijin Ge
>
> **备注:** To appear in the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Jailbreak attacks on audio language models (ALMs) optimize audio perturbations to elicit unsafe generations, and they typically update the entire waveform densely throughout optimization. In this work, we investigate the necessity of such dense optimization by analyzing the structure of token-aligned gradients in ALMs. We find that gradient energy is highly non-uniform across audio tokens, indicating that only a small subset of token-aligned audio regions dominates the optimization signal. Motivated by this observation, we propose Token-Aware Gradient Optimization (TAGO), which enables sparse jailbreak optimization by retaining only waveform gradients aligned with audio tokens that have high gradient energy, while masking the remaining gradients at each iteration. Across three ALMs, TAGO outperforms baselines, and substantial sparsification preserves strong attack success rates (e.g. on Qwen3-Omni, $\mathrm{ASR}_{l}$ remains at 86% with a token retention ratio of 0.25, compared to 87% with full token retention). These results demonstrate that dense waveform updates are largely redundant, and we advocate that future audio jailbreak and safety alignment research should further leverage this heterogeneous token-level gradient structure.
>
---
#### [replaced 012] RADAR Challenge 2026: Robust Audio Deepfake Recognition under Media Transformations
- **分类: eess.AS**

- **简介: 该论文介绍RADAR Challenge 2026，聚焦于多语言和媒体变换下的音频深度伪造检测任务，旨在提升音频真伪识别的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.09568](https://arxiv.org/pdf/2605.09568)**

> **作者:** Hieu-Thi Luong; Xuechen Liu; Ivan Kukanov; Zheng Xin Chai; Kong Aik Lee
>
> **备注:** Submitted to APSIPA 2026
>
> **摘要:** RADAR Challenge 2026 is an APSIPA Grand Challenge on Robust Audio Deepfake Recognition under Media Transformations, designed to simulate realistic media conditions in real-world audio distribution pipelines, including compression, resampling, noise, and reverberation. It consists of two phases: an English development phase with labeled data for analysis and paper writing, and a multilingual evaluation phase containing more than 100,000 utterances in English, Singapore English, Mandarin Chinese, Taiwanese Mandarin, Japanese, and Vietnamese. Systems are evaluated using equal error rate (EER) for binary real/fake classification. This paper describes the challenge task, the construction of the data set, the evaluation protocol, and the overall results. During the challenge, 33 teams submitted to the development phase and 22 teams submitted to the final evaluation phase. The reported results highlight the remaining challenges of robust audio deepfake detection under multilingual and media-transformed conditions.
>
---
#### [replaced 013] KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于实时语音对话AI任务，旨在解决实时模型知识不足与延迟系统知识丰富但延迟高的问题。提出KAME架构，结合实时语音模型与后端语言模型，提升响应准确性同时保持低延迟。**

- **链接: [https://arxiv.org/pdf/2510.02327](https://arxiv.org/pdf/2510.02327)**

> **作者:** So Kuroki; Yotaro Kubo; Takuya Akiba; Yujin Tang
>
> **备注:** Published at IEEE ICASSP 2026
>
> **摘要:** Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline.
>
---
#### [replaced 014] CounterFlow: A Two-Phase Inference-Time Sampling for Counterfactual Video Foley Generation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视频音频生成任务，解决视频与文本不一致时生成矛盾声音的问题。提出ConterFlow方法，在推理阶段分两步优化音频生成效果。**

- **链接: [https://arxiv.org/pdf/2605.18916](https://arxiv.org/pdf/2605.18916)**

> **作者:** Gyubin Lee; Junwon Lee; Juhan Nam
>
> **备注:** accepted to CVPR 2026 Workshop on Sight and Sound
>
> **摘要:** We investigate Counterfactual Video Foley Generation, which aims to adopt a sound-source identity that contradicts the visual evidence while remaining temporally synchronized to a silent video. Existing Video&Text-to-Audio (VT2A) models struggle with this, often remaining anchored to the visually implied sound source when video and text contents disagree. We present ConterFlow, an inference-time dual-phase sampling scheme for pretrained flow-matching VT2A models. Phase 1 builds a video-derived temporal structure while suppressing the visually implied source; Phase 2 drops video conditioning to focus entirely on shaping audio timbre toward the target prompt. ConterFlow substantially improves counterfactual Video Foley generation compared to naive negative prompting and state-of-the-art baselines. To evaluate replacement quality, we propose a metric leveraging a text-audio co-embedding space to measure both target-prompt evidence and residual visually implied source leakage. Video demonstrations and code are available at this https URL
>
---
#### [replaced 015] Position: Towards Responsible Evaluation for Text-to-Speech
- **分类: eess.AS**

- **简介: 该论文属于文本到语音技术评估任务，旨在解决现有评估方法不足的问题，提出负责任评估框架，涵盖能力反映、标准比较和伦理安全三个层面。**

- **链接: [https://arxiv.org/pdf/2510.06927](https://arxiv.org/pdf/2510.06927)**

> **作者:** Yifan Yang; Hui Wang; Bing Han; Shujie Liu; Jinyu Li; Yong Qin; Xie Chen
>
> **备注:** Accepted in ICML 2026
>
> **摘要:** Recent advances in text-to-speech (TTS) technology have enabled systems to generate speech that is often indistinguishable from human speech, bringing benefits to accessibility, content creation, and human-computer interaction. However, current evaluation practices are increasingly inadequate for capturing the full range of capabilities, limitations, and societal impacts of modern TTS systems. This position paper introduces the concept of Responsible Evaluation and argues that it is essential and urgent for the next phase of TTS development, structured through three progressive levels: (1) ensuring the faithful and accurate reflection of a model's true capabilities and limitations, with more robust, discriminative, and comprehensive objective and subjective scoring methodologies; (2) enabling comparability, standardization, and transferability through standardized benchmarks, transparent reporting, and transferable evaluation metrics; and (3) assessing governance, fairness, and security concerns around data provenance, disparities, misuse, spoofing, and traceability. Through this concept, we critically examine current evaluation practices, identify systemic shortcomings, and propose actionable recommendations. We hope this concept will not only foster more reliable TTS technology but also guide its development toward ethically sound and societally beneficial applications.
>
---
