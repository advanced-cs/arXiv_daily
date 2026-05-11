# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] TARNet: A Temporal-Aware Multi-Scale Architecture for Closed-Set Speaker Identification
- **分类: cs.SD**

- **简介: 该论文提出TARNet，用于闭集说话人识别任务，解决多时间尺度语音特征建模问题，通过多阶段编码器和注意力统计池实现有效特征融合。**

- **链接: [https://arxiv.org/pdf/2605.07735](https://arxiv.org/pdf/2605.07735)**

> **作者:** Yassin Terraf; Youssef Iraqi
>
> **备注:** Accepted at IEEE International Conference on Multimedia and Expo (ICME) 2026. Code available at: this https URL
>
> **摘要:** Closed-Set speaker identification aims to assign a speech utterance to one of a predefined set of enrolled speakers and requires robust modeling of speaker-specific characteristics across multiple temporal scales. While recent deep learning approaches have achieved strong performance, many existing architectures provide limited mechanisms for modeling temporal dependencies across different time scales, which can restrict the effective use of complementary short-, mid-, and long-term speaker characteristics. In this paper, we propose TARNet, a lightweight Temporal-Aware Representation Network for closed-set speaker identification. TARNet explicitly models temporal information at multiple time scales using a multi-stage temporal encoder with stage-specific dilation configurations. The resulting multi-scale representations are fused and aggregated via an Attentive Statistics Pooling (ASP) module to produce a discriminative utterance-level speaker embedding. Experiments on the VoxCeleb1 and LibriSpeech datasets show that TARNet outperforms state-of-the-art methods while maintaining competitive computational complexity, making it suitable for practical speaker identification systems. The code is publicly available at this https URL.
>
---
#### [new 002] Dependence on Early and Late Reverberation of Single-Channel Speaker Distance Estimation
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文研究单通道说话人距离估计任务，探讨模型如何依赖房间冲激响应的不同部分。通过分解RIR并测试不同校准场景，发现时间校准对精度至关重要。**

- **链接: [https://arxiv.org/pdf/2605.07694](https://arxiv.org/pdf/2605.07694)**

> **作者:** Michael Neri; Archontis Politis; Tuomas Virtanen
>
> **备注:** Submitted to IWAENC 2026
>
> **摘要:** Single-channel speaker distance estimation has recently achieved centimeter-level accuracy in simulated environments, yet it remains unclear which components of the room impulse response (RIR) the model exploits and how performance depends on the recording conditions. In this work, we decompose simulated RIRs into four variants (full, direct-only, no-late, and no-early) using the mixing time estimated from the echo density function as the boundary between early reflections and late reverberation. We define four calibration scenarios, from fully calibrated (synchronised capture, known source level) to fully uncalibrated (arbitrary onset, unknown level), and evaluate all combinations on a matched dataset. Results show that without time calibration, mean absolute error (MAE) increases to $1.29$ m and the model extracts reverberation-based cues, with early reflections emerging as the most informative component. Further analysis against DRR, $C_{50}$, and $T_{60}$ confirms that estimation accuracy improves with stronger early energy and degrades in highly reverberant environments. When time calibration is available, the model achieves a MAE of $0.14$ m by extracting the propagation delay alone, regardless of the RIR content.
>
---
#### [new 003] Evaluating voice anonymisation using similarity rank disclosure
- **分类: eess.AS**

- **简介: 该论文属于语音匿名化评估任务，旨在解决现有评估方法不足的问题。通过引入相似性排名泄露（SRD）指标，提供更全面的隐私风险分析。**

- **链接: [https://arxiv.org/pdf/2605.07291](https://arxiv.org/pdf/2605.07291)**

> **作者:** Shilpa Chandra; Matteo Pettenò; Nicholas Evans; Michele Panariello; Massimiliano Todisco; Tom Bäckström; Dorothea Kolossa; Rainer Martin; Themos Stafylakis; Nicolas Gengembre
>
> **摘要:** The evaluation of voice anonymisation remains challenging. Current practice relies on automatic speaker verification metrics such as the equal error rate (EER). Performance estimates dependent on the classifier and operating point provide an incomplete or even misleading characterisation of privacy risk. We investigate the use of similarity rank disclosure (SRD), an information-theoretic metric, which operates on feature representations rather than classifier decisions, providing a threshold-independent assessment of privacy and analysis of both average and worst-case disclosure. We report its application to speaker embeddings, fundamental frequency, and phone embeddings using 2024 VoicePrivacy Challenge systems. The SRD reveals privacy leaks and system-specific weaknesses missed by EER-based evaluation. Findings highlight the merit of representation-level metrics and demonstrate the potential of SRD as a flexible and interpretable tool for the evaluation of voice anonymisation.
>
---
#### [new 004] BeeVe: Unsupervised Acoustic State Discovery in Honey Bee Buzzing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于无监督声学状态发现任务，旨在从蜜蜂嗡嗡声中提取结构。通过VQ-VAE学习离散代码本，无需标注即可识别蜂群状态。**

- **链接: [https://arxiv.org/pdf/2605.07903](https://arxiv.org/pdf/2605.07903)**

> **作者:** Hamze Hammami; Nidhal Abdulaziz
>
> **摘要:** Discovering structure in biological signals without supervision is a fundamental problem in computational intelligence, yet existing bioacoustic methods assume vocal production models or predefined semantic units, leaving non-vocal species poorly served. This work introduces BeeVe, an unsupervised framework for acoustic state discovery in collective honey bee buzzing. BeeVe uses the self-supervised Patchout Spectrogram Transformer (PaSST) as a frozen feature extractor, then trains a Vector-Quantized Variational Autoencoder (VQ-VAE) without labels on those embeddings, learning a finite discrete codebook of acoustic tokens directly from unlabelled hive audio. No labels, pretext tasks, or contrastive objectives are used at any stage. Post-hoc evaluation against known queen status reveals that the learned tokens separate queenright and queenless conditions with Jensen-Shannon Divergence values between 0.609 and 0.688, and that the queenless condition further decomposes into three internally coherent sub-states stable across experiments with different codebook sizes and random seeds. Token transition analysis confirms non-random sequential structure (p << 0.001) across all experiments. Generalisation to unseen recordings preserves both token overlap (Jaccard = 0.947) and global manifold topology. These results demonstrate that unsupervised discrete codebook learning can recover repeatable acoustic structure from a non-vocal biological signal without annotation, opening a path toward non-invasive acoustic hive health monitoring.
>
---
#### [new 005] Do Joint Audio-Video Generation Models Understand Physics?
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于多模态生成任务，旨在评估联合音视频生成模型是否理解物理常识。通过构建基准测试，发现模型在物理一致性上仍有显著不足。**

- **链接: [https://arxiv.org/pdf/2605.07061](https://arxiv.org/pdf/2605.07061)**

> **作者:** Zijun Cui; Xiulong Liu; Hao Fang; Mingwei Xu; Jiageng Liu; Zexin Xu; Weiguo Pian; Shijian Deng; Feiyu Du; Chenming Ge; Yapeng Tian
>
> **备注:** Preprint. Full abstract appears in the PDF
>
> **摘要:** Joint audio-video generation models are rapidly approaching professional production quality, raising a central question: do they understand audio-visual physics, or merely generate plausible sounds and frames that violate real-world consistency? We introduce AV-Phys Bench, a benchmark for evaluating physical commonsense in joint audio-video generation. AV-Phys Bench tests models across three scene categories: Steady State, Event Transition, and Environment Transition. It covers physics-grounded subcategories drawn from real-world scenes, plus Anti-AV-Physics prompts that deliberately request physically inconsistent audio-video behavior. Each generation is evaluated along five dimensions: visual semantic adherence, audio semantic adherence, visual physical commonsense, audio physical commonsense, and cross-modal physical commonsense. Across three proprietary and four open-source models, we find that Seedance 2.0 performs best overall, but all models remain far from robust physical understanding. Performance drops sharply on event-driven and environment-driven transitions, and even strong proprietary systems collapse on Anti-AV-Physics prompts. We further introduce AV-Phys Agent, a ReAct-style evaluator that combines a multimodal language model with deterministic acoustic measurement tools, producing rankings that closely align with human ratings. Our results identify cross-modal physical consistency and transition-driven scene dynamics as key open challenges for joint audio-video generation.
>
---
#### [new 006] A Decomposed Retrieval-Edit-Rerank Framework for Chord Generation
- **分类: cs.SD; cs.MM; eess.SP**

- **简介: 该论文属于音乐生成任务，旨在解决 chord generation 中风格多样性与音乐理论可行性之间的平衡问题。提出一种分解的 Retrieval-Edit-Rerank 框架，分阶段处理生成过程，提升可控制性和输出质量。**

- **链接: [https://arxiv.org/pdf/2605.07489](https://arxiv.org/pdf/2605.07489)**

> **作者:** Qiqi He; Dichucheng Li; Xiaoheng Sun; Anqi Huang
>
> **备注:** Accepted by the 2026 ACM International Conference on Multimedia Retrieval (ICMR 2026)
>
> **摘要:** Chord generation is an inherently constrained creative task that requires balancing stylistic diversity with music-theoretic feasibility. Existing approaches typically entangle candidate generation and constraint enforcement within a single model, making the diversity-feasibility trade-off difficult to control and interpret. In this work, we approach chord generation from a system-level perspective, introducing a Retrieval-Edit-Rerank (RER) framework that decomposes the task into three explicit stages: i) retrieval, which defines a stylistically plausible candidate space; ii) editing, which enforces music-theoretic feasibility through minimal modifications; and iii) reranking, which resolves soft preferences among feasible candidates. This separation provides a controllable pipeline, where each component addresses a distinct aspect of the generation process, thereby enhancing both the interpretability and adjustability of the output chords. Through objective metrics and subjective evaluation, our decomposed system outperforms all end-to-end chord generation baselines in balancing chord diversity and music-theoretic feasibility. Ablation studies further confirm the complementary roles of each stage in creative exploration and constraint satisfaction.
>
---
#### [new 007] An audio-to-analysis pipeline with certified transcription for information-theoretic profiling of the piano repertoire
- **分类: cs.SD; eess.AS; stat.AP**

- **简介: 该论文提出一种音频分析管道，用于生成作曲家级信息理论特征，解决音乐风格分析问题。通过认证的转录和统计方法，分析和声分布与风格相似性。**

- **链接: [https://arxiv.org/pdf/2605.06685](https://arxiv.org/pdf/2605.06685)**

> **作者:** Fred Jalbert-Desforges
>
> **备注:** 25 pages, 4 figures, 25 references
>
> **摘要:** We present an audio-to-analysis pipeline that produces composer-level information-theoretic profiles : reflecting compositional vocabulary as it emerges from aggregated performances : from raw recordings, built on a transcription layer whose accuracy we certify on a standard benchmark (F1 = 0.9791 on the MAESTRO v3.0.0 test set). Applied to 1,238 pieces and 15 MAESTRO composers with at least ten attributed pieces, spanning the Baroque through the early twentieth century, the pipeline derives empirical distributions over harmonic scale degrees and analyzes them through Shannon entropy, asymmetric Kullback-Leibler divergence, and Zipfian rank-frequency modeling. The resulting profiles (i) order composers along an interpretable axis of harmonic predictability, with a narrow entropy range (3.33-3.86 bits) that reveals the marginal-level similarity of tonal vocabularies; (ii) recover known stylistic lineages (Haydn-Beethoven, Liszt-Rachmaninoff, Schubert-Schumann) through the smallest KL divergences in the corpus, with Mendelssohn emerging as a stable outlier within this corpus; and (iii) separate contemporary neoclassical artists (Richter, Frahm, Glass, Arnalds, Jóhannsson) from historical composers on the quality of Zipfian fit to the transition distribution, with mean $R^2 = 0.78$ for neoclassical versus 0.46 for historical (N $\geq$ 10 pieces each). This gap is larger than the spread within either group and is consistent with a minimalist compositional tendency: a compact transition vocabulary used with sharper frequency-rank regularity than historical composers. All estimates are reported with Laplace-smoothed bootstrap 95% confidence intervals.
>
---
#### [new 008] MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes
- **分类: cs.CL; cs.AI; cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出MIST数据集，解决智能家庭中多模态语音交互工具调用的问题，旨在提升语音助手对物理世界约束的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.06897](https://arxiv.org/pdf/2605.06897)**

> **作者:** Maximillian Chen; Xuanming Zhang; Michael Peng; Zhou Yu; Alexandros Papangelis; Yohan Jo
>
> **备注:** Project Page: this https URL
>
> **摘要:** The rise of Internet of Things (IoT) devices in the physical world necessitates voice-based interfaces capable of handling complex user experiences. While modern Large Language Models (LLMs) already demonstrate strong tool-usage capabilities, modeling real-world IoT devices presents a difficult, understudied challenge which combines modeling spatiotemporal constraints with speech inputs, dynamic state tracking, and mixed-initiative interaction patterns. We introduce MIST (the Multimodal Interactive Speech-based Tool-calling Dataset), a synthetic multi-turn, voice-driven code generation task that operates over IoT devices. We find that there is a significant gap between open- and closed-weight multimodal LLMs on MIST, and that even frontier closed-weight LLMs have substantial headroom. We release MIST and an extensible data generation framework to build related datasets in order to facilitate research on mixed-initiative voice assistants which reason about physical world constraints.
>
---
#### [new 009] Zero-Shot Imagined Speech Decoding via Imagined-to-Listened MEG Mapping
- **分类: cs.LG; eess.AS**

- **简介: 该论文属于脑机接口任务，旨在解决想象言语解码难题。通过建立想象与聆听的MEG映射，利用聆听数据提升解码效果，验证了想象言语可被有效解码。**

- **链接: [https://arxiv.org/pdf/2605.08075](https://arxiv.org/pdf/2605.08075)**

> **作者:** Maryam Maghsoudi; Shihab Shamma
>
> **摘要:** Decoding imagined speech from non-invasive brain recordings is challenging because imagined datasets are scarce and difficult to align temporally across subjects and sessions In this work, we propose a new approach to the decoding of imagined speech that leverages the richer and more reliably labeled recordings during listening to speech. We collected paired listened and imagined MEG recordings to rhythmic melodic and spoken stimuli from trained musicians. Using trained musicians helped improve temporal alignment across conditions. We then developed a three-stage decoding pipeline that revealed consistent and meaningful relationships between neural activity evoked by imagining and listening to the same stimuli. First, we trained six linear and neural models to map imagined MEG responses to listened responses. We evaluated these models against a null baseline from unseen subjects to validate that the predicted-listening responses preserve stimulus-specific information. In the second stage, we trained a contrastive word decoder exclusively on the listened MEG responses, and evaluated it using four embedding strategies including semantic, acoustic, and phonetic representations. In the third stage, we process the imagined MEG responses from held-out subjects through the mapping pipeline to compute the corresponding listening responses that are then decoded by the listened decoder. Using rank-based analysis, we show that the imagined words are decodable significantly above chance. We shall report here the results of a proof-of-concept implementation to decode imagined speech, where all evaluations are performed on held-out subjects. We also demonstrate that performance improves with training data size, suggesting that this approach is scalable and can directly be made applicable to realistic brain-computer interface scenarios.
>
---
#### [new 010] Asymmetric Phase Coding Audio Watermarking
- **分类: cs.CR; eess.AS**

- **简介: 该论文属于音频水印任务，旨在解决深度伪造音频的认证问题。提出一种无需训练的对称相位编码方法，实现可靠、抗攻击的音频溯源。**

- **链接: [https://arxiv.org/pdf/2605.07241](https://arxiv.org/pdf/2605.07241)**

> **作者:** Guang Yang; Amir Ghasemian; Ninareh Mehrabi; Homa Hosseinmardi
>
> **备注:** 13 pages, 12 figures, 3 tables
>
> **摘要:** The proliferation of deepfake audio challenges voice-based authentication systems; passive forensic detectors are sensitive to evolving generative models and to real-world channel distortions. We propose Asymmetric Phase Coding (APC), a training-free cryptographic signing layer for audio, designed as a compact and auditable provenance primitive that can stand alone or be stacked with learned watermarks. APC combines Ed25519 digital signatures (EdDSA, FIPS 186-5; 64-byte signatures) with Reed-Solomon error correction, pseudo-random STFT phase-bin selection, and a redundant quantization-index-modulation (QIM) code on log-magnitude differences of adjacent bin pairs, yielding a compact, non-repudiable, blind-extractable watermark. We evaluate APC on 1,000 LibriSpeech test-clean clips (10 s each, 44.1 kHz) under eight attack configurations -- identity, 10% end-cropping, 20% end-cropping, 8 kHz low-pass, 16 kHz round-trip resampling, FLAC re-encoding, MP3 at 128 kbps, and OGG-Vorbis at 128 kbps -- and achieve cryptographic verification rates between 97.5% and 98.3% on every condition at mean PESQ=3.02 and tens-of-milliseconds CPU latency. We explicitly compare APC against recent neural baselines (AudioSeal, WavMark, SilentCipher), detail the threat model (forgery resistance vs. erasure), characterize the dataset, define all metrics, quantify an adaptive white-box erasure attack, and release code, keys, and metadata for reproducibility.
>
---
## 更新

#### [replaced 001] Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音特征分析任务，研究自监督学习模型如何编码说话人信息。通过PCA分析发现不同特征分布在不同维度，且可独立操控。**

- **链接: [https://arxiv.org/pdf/2603.03096](https://arxiv.org/pdf/2603.03096)**

> **作者:** Kyle Janse van Rensburg; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 pages, 7 figures, submitted to IEEE Signal Processing Letters
>
> **摘要:** How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. For a range of SSL models, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. We then use synthesis analyses to show that the dimensions for most characteristics are isolated from each other's influence. We further show that characteristics can be changed by manipulating the corresponding dimensions.
>
---
#### [replaced 002] Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决语音与文本模型间的模态差距问题。通过改进输入端的语音处理，提出TextPro-SLM模型，提升语音理解能力。**

- **链接: [https://arxiv.org/pdf/2605.05927](https://arxiv.org/pdf/2605.05927)**

> **作者:** Wenqian Cui; Xiao-Hui Li; Daxin Tan; Qiyong Zheng; Irwin King
>
> **备注:** Work in progress
>
> **摘要:** Speech large language models (SLMs) are typically built from text large language model (TLM) checkpoints, yet they still suffer from a substantial modality gap. Prior work has mainly attempted to reduce this gap from the output side by making speech generation more text-like, but the gap remains. We argue that the key remaining bottleneck lies on the input side. We propose TextPro-SLM, an SLM that makes spoken input more closely resemble that of a prosody-aware text LLM. TextPro-SLM combines WhisperPro, a unified speech encoder that produces synchronized text tokens and prosody embeddings, with an LLM backbone trained to preserve the semantic capabilities of the original TLM while learning paralinguistic understanding. Experiments show that TextPro-SLM achieves the lowest modality gap among leading SLMs at both 3B and 7B scales, while also delivering strong overall performance on paralinguistic understanding tasks. These gains are achieved with only roughly 1,000 hours of LLM training audio, suggesting that reducing the modality gap from the input side is both effective and data-efficient.
>
---
#### [replaced 003] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音指令跟随任务，旨在解决现有评估基准忽略韵律等非语言信息的问题。提出S2S-Arena基准，通过语音原生评估提升模型表达能力。**

- **链接: [https://arxiv.org/pdf/2503.05085](https://arxiv.org/pdf/2503.05085)**

> **作者:** Feng Jiang; Zhiyu Lin; Yiyang Liu; Liumeng Xue; Fan Bu; Yuhao Du; Xiangying Chen; Benyou Wang; Haizhou Li
>
> **备注:** Accepted by ACL 2026 main
>
> **摘要:** Recent advances in large language models (LLMs) have fundamentally reshaped speech-to-speech (S2S) systems, enabling increasingly natural spoken interaction. However, existing benchmarks still rely heavily on text-based evaluation and largely ignore paralinguistic cues such as prosody, emotion, and speaker traits, which are central to expressive and human-like communication. We introduce S2S-Arena, a speech-native benchmark for evaluating instruction-following S2S models with explicit assessment of both semantic understanding and paralinguistic expression. S2S-Arena features a four-level interaction protocol that systematically probes models under increasing paralinguistic complexity, a two-stage data construction pipeline that produces 1,243 speech samples spanning 100+ real-world tasks, and an arena-style evaluation framework that enables reference-free, pairwise comparison directly in the speech modality. Benchmarking 10 state-of-the-art S2S systems over 1,000+ comparisons reveals substantial performance gaps (especially under complex paralinguistic demands) between current academic and industrial systems. Our analysis further identifies key design factors governing expressive instruction following, providing actionable insights for building more natural, robust, and human-aligned speech agents.
>
---
#### [replaced 004] Optimising MFCC parameters for the automatic detection of respiratory diseases
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于呼吸疾病自动检测任务，研究MFCC参数对诊断的影响，通过调整系数数量、帧长和帧移，提升分类准确率。**

- **链接: [https://arxiv.org/pdf/2408.07522](https://arxiv.org/pdf/2408.07522)**

> **作者:** Yuyang Yan; Sami O. Simons; Loes van Bemmel; Lauren Reinders; Frits M.E. Franssen; Visara Urovi
>
> **摘要:** Voice signals originating from the respiratory tract are utilized as valuable acoustic biomarkers for the diagnosis and assessment of respiratory diseases. Among the employed acoustic features, Mel Frequency Cepstral Coefficients (MFCC) is widely used for automatic analysis, with MFCC extraction commonly relying on default parameters. However, no comprehensive study has systematically investigated the impact of MFCC extraction parameters on respiratory disease diagnosis. In this study, we address this gap by examining the effects of key parameters, namely the number of coefficients, frame length, and hop length between frames, on respiratory condition examination. Our investigation uses four datasets: the Cambridge COVID-19 Sound database, the Coswara dataset, the Saarbrucken Voice Disorders (SVD) database, and a TACTICAS dataset. The Support Vector Machine (SVM) is employed as the classifier, given its widespread adoption and efficacy. Our findings indicate that the accuracy of MFCC decreases as hop length increases, and the optimal number of coefficients is observed to be approximately 30. The performance of MFCC varies with frame length across the datasets: for the COVID-19 datasets (Cambridge COVID-19 Sound database and Coswara dataset), performance declines with longer frame lengths, while for the SVD dataset, performance improves with increasing frame length (from 50 ms to 500 ms). Furthermore, we investigate the optimized combination of these parameters and observe substantial enhancements in accuracy. Compared to the worst combination, the SVM model achieves an accuracy of 81.1%, 80.6%, and 71.7%, with improvements of 19.6%, 16.10%, and 14.90% for the Cambridge COVID-19 Sound database, the Coswara dataset, and the SVD dataset respectively.
>
---
#### [replaced 005] AsymTalker: Identity-Consistent Long-Term Talking Head Generation via Asymmetric Distillation
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于 Talking Head 生成任务，旨在解决长视频生成中的身份不一致问题。提出 AsymTalker 方法，通过时间参考编码和非对称知识蒸馏，实现高保真、身份一致的长时视频生成。**

- **链接: [https://arxiv.org/pdf/2605.02948](https://arxiv.org/pdf/2605.02948)**

> **作者:** Yuxin Lu; Qian Qiao; Jiayang Sun; Guibo Zhu; Min Cao
>
> **摘要:** Diffusion-based talking head generation has achieved remarkable visual quality, yet scaling it to long-term videos remains challenging. The widely adopted chunk-wise paradigm introduces two fundamental failures: (1) temporal-spatial misalignment between static identity references and dynamic audio streams, and (2) cascading identity drift propagated through self-generated continuity references across chunks. To address both issues, we propose AsymTalker, a novel diffusion-based talking head generation method comprising Temporal Reference Encoding (TRE) and Asymmetric Knowledge Distillation (AKD). First, TRE mitigates temporal-spatial misalignment by transforming the static identity image into a temporally coherent latent representation through encoding of a temporally replicated pseudo-video, without introducing additional parameters. Second, AKD resolves the inherent conditioning dilemma in chunk-wise training: using ground-truth references causes train-inference mismatch, while self-generated references entangle supervision with identity drift. Our asymmetric design circumvents this by anchoring the teacher model with ground-truth continuity references to provide drift-free, chunk-level supervision, thereby avoiding the teacher bottleneck. Meanwhile, the student model learns under inference-aligned conditions, conditioned only on self-generated references, and is trained via distribution matching to preserve identity over long horizons. Extensive experiments show AsymTalker achieves state-of-the-art results on HDTF and VFHQ. It guarantees high-fidelity, identity-consistent synthesis over 600-second videos and reaches a real-time inference speed of 66 FPS.
>
---
#### [replaced 006] Multi-Axis Speech Similarity via Factor-Partitioned Embeddings
- **分类: eess.AS; cs.IR**

- **简介: 该论文属于语音相似性任务，解决传统嵌入方法混淆多属性的问题。通过因子分区嵌入框架，将语音映射到包含不同属性子空间的向量，提升属性条件下的检索效果。**

- **链接: [https://arxiv.org/pdf/2605.02804](https://arxiv.org/pdf/2605.02804)**

> **作者:** Jim O'Regan; Jens Edlund
>
> **备注:** 7 pages, accepted at Odyssey 2026
>
> **摘要:** Speech encodes multiple simultaneous attributes -- linguistic content, speaker identity, dialect, gender --that conventional single-vector embeddings conflate. We present a factor-partitioned embedding framework that maps each utterance into a single vector whose subspaces correspond to distinct axes of variation. A shared acoustic encoder feeds per-axis linear projection heads, each trained via distillation from a specialist teacher or a contrastive objective over shared-label pairs. The resulting embeddings support attribute-conditioned retrieval: similarity is computed as a signed weighted sum over per-axis cosine scores, allowing retrieval that jointly considers what was said and how -- or explicitly suppresses one attribute to surface another. We evaluate on cross-corpus retrieval over corpora sharing the Harvard sentence prompts, demonstrating that signed axis weighting can suppress same-speaker bias and surface semantically matched utterances across recording conditions. Code is available at: this https URL
>
---
