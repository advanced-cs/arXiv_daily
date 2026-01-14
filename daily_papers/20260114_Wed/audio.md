# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Decoding Order Matters in Autoregressive Speech Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，研究解码顺序对生成质量的影响。通过掩码扩散框架，探索不同解码顺序，发现自适应解码优于固定顺序，且低比特量化仍可生成高质量语音。**

- **链接: [https://arxiv.org/pdf/2601.08450v1](https://arxiv.org/pdf/2601.08450v1)**

> **作者:** Minghui Zhao; Anton Ragni
>
> **摘要:** Autoregressive speech synthesis often adopts a left-to-right order, yet generation order is a modelling choice. We investigate decoding order through masked diffusion framework, which progressively unmasks positions and allows arbitrary decoding orders during training and inference. By interpolating between identity and random permutations, we show that randomness in decoding order affects speech quality. We further compare fixed strategies, such as \texttt{l2r} and \texttt{r2l} with adaptive ones, such as Top-$K$, finding that fixed-order decoding, including the dominating left-to-right approach, is suboptimal, while adaptive decoding yields better performance. Finally, since masked diffusion requires discrete inputs, we quantise acoustic representations and find that even 1-bit quantisation can support reasonably high-quality speech.
>
---
#### [new 002] Robust CAPTCHA Using Audio Illusions in the Era of Large Language Models: from Evaluation to Advances
- **分类: cs.SD; cs.CY; eess.AS**

- **简介: 该论文属于安全任务，旨在解决音频CAPTCHA在大语言模型和语音识别攻击下的脆弱性问题。提出AI-CAPTCHA框架和IllusionAudio方法，有效提升音频CAPTCHA的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.08516v1](https://arxiv.org/pdf/2601.08516v1)**

> **作者:** Ziqi Ding; Yunfeng Wan; Wei Song; Yi Liu; Gelei Deng; Nan Sun; Huadong Mo; Jingling Xue; Shidong Pan; Yuekang Li
>
> **摘要:** CAPTCHAs are widely used by websites to block bots and spam by presenting challenges that are easy for humans but difficult for automated programs to solve. To improve accessibility, audio CAPTCHAs are designed to complement visual ones. However, the robustness of audio CAPTCHAs against advanced Large Audio Language Models (LALMs) and Automatic Speech Recognition (ASR) models remains unclear. In this paper, we introduce AI-CAPTCHA, a unified framework that offers (i) an evaluation framework, ACEval, which includes advanced LALM- and ASR-based solvers, and (ii) a novel audio CAPTCHA approach, IllusionAudio, leveraging audio illusions. Through extensive evaluations of seven widely deployed audio CAPTCHAs, we show that most existing methods can be solved with high success rates by advanced LALMs and ASR models, exposing critical security weaknesses. To address these vulnerabilities, we design a new audio CAPTCHA approach, IllusionAudio, which exploits perceptual illusion cues rooted in human auditory mechanisms. Extensive experiments demonstrate that our method defeats all tested LALM- and ASR-based attacks while achieving a 100% human pass rate, significantly outperforming existing audio CAPTCHA methods.
>
---
#### [new 003] Quantitative Analysis of Proxy Tasks for Anomalous Sound Detection
- **分类: eess.AS**

- **简介: 该论文属于异常声音检测任务，旨在探讨代理任务与检测性能的关系。通过实验分析五种代理任务，发现强代理性能未必提升检测效果，强调任务难度与目标对齐的重要性。**

- **链接: [https://arxiv.org/pdf/2601.08480v1](https://arxiv.org/pdf/2601.08480v1)**

> **作者:** Seunghyeon Shin; Seokjin Lee
>
> **备注:** 13 pages, 5 figures, Submitted to IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** Anomalous sound detection (ASD) typically involves self-supervised proxy tasks to learn feature representations from normal sound data, owing to the scarcity of anomalous samples. In ASD research, proxy tasks such as AutoEncoders operate under the explicit assumption that models trained on normal data will increase the reconstruction errors related to anomalies. A natural extension suggests that improved proxy task performance should improve ASD capability; however, this relationship has received little systematic attention. This study addresses this research gap by quantitatively analyzing the relationship between proxy task metrics and ASD performance across five configurations, namely, AutoEncoders, classification, source separation, contrastive learning, and pre-trained models. We evaluate the learned representations using linear probe (linear separability) and Mahalanobis distance (distributional compactness). Our experiments reveal that strong proxy performance does not necessarily improve anomalous sound detection performance. Specifically, classification tasks experience performance saturation owing to insufficient task difficulty, whereas contrastive learning fails to learn meaningful features owing to limited data diversity. Notably, source separation is the only task demonstrating a strong positive correlation, such that improved separation consistently improves anomaly detection. Based on these findings, we highlight the critical importance of task difficulty and objective alignment. Finally, we propose a three-stage alignment verification protocol to guide the design of highly effective proxy tasks for ASD systems.
>
---
#### [new 004] Weakly Supervised Tabla Stroke Transcription via TI-SDRM: A Rhythm-Aware Lattice Rescoring Framework
- **分类: eess.AS**

- **简介: 该论文属于音乐信息检索任务，解决弱监督下的塔布拉鼓击序列转录问题。通过结合声学模型与节奏重评分框架，提升转录准确性。**

- **链接: [https://arxiv.org/pdf/2601.08537v1](https://arxiv.org/pdf/2601.08537v1)**

> **作者:** Rahul Bapusaheb Kodag; Vipul Arora
>
> **摘要:** Tabla Stroke Transcription (TST) is central to the analysis of rhythmic structure in Hindustani classical music, yet remains challenging due to complex rhythmic organization and the scarcity of strongly annotated data. Existing approaches largely rely on fully supervised learning with onset-level annotations, which are costly and impractical at scale. This work addresses TST in a weakly supervised setting, using only symbolic stroke sequences without temporal alignment. We propose a framework that combines a CTC-based acoustic model with sequence-level rhythmic rescoring. The acoustic model produces a decoding lattice, which is refined using a \textbf{$T\bar{a}la$}-Independent Static--Dynamic Rhythmic Model (TI-SDRM) that integrates long-term rhythmic structure with short-term adaptive dynamics through an adaptive interpolation mechanism. We curate a new real-world tabla solo dataset and a complementary synthetic dataset, establishing the first benchmark for weakly supervised TST in Hindustani classical music. Experiments demonstrate consistent and substantial reductions in stroke error rate over acoustic-only decoding, confirming the importance of explicit rhythmic structure for accurate transcription.
>
---
#### [new 005] Tuberculosis Screening from Cough Audio: Baseline Models, Clinical Variables, and Uncertainty Quantification
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于 tuberculosis 检测任务，旨在通过咳嗽音频和临床数据进行自动筛查。提出标准化框架，解决评估不一致问题，构建基线模型并量化不确定性。**

- **链接: [https://arxiv.org/pdf/2601.07969v1](https://arxiv.org/pdf/2601.07969v1)**

> **作者:** George P. Kafentzis; Efstratios Selisios
>
> **摘要:** In this paper, we propose a standardized framework for automatic tuberculosis (TB) detection from cough audio and routinely collected clinical data using machine learning. While TB screening from audio has attracted growing interest, progress is difficult to measure because existing studies vary substantially in datasets, cohort definitions, feature representations, model families, validation protocols, and reported metrics. Consequently, reported gains are often not directly comparable, and it remains unclear whether improvements stem from modeling advances or from differences in data and evaluation. We address this gap by establishing a strong, well-documented baseline for TB prediction using cough recordings and accompanying clinical metadata from a recently compiled dataset from several countries. Our pipeline is reproducible end-to-end, covering feature extraction, multimodal fusion, cougher-independent evaluation, and uncertainty quantification, and it reports a consistent suite of clinically relevant metrics to enable fair comparison. We further quantify performance for cough audio-only and fused (audio + clinical metadata) models, and release the full experimental protocol to facilitate benchmarking. This baseline is intended to serve as a common reference point and to reduce methodological variance that currently holds back progress in the field.
>
---
#### [new 006] LJ-Spoof: A Generatively Varied Corpus for Audio Anti-Spoofing and Synthesis Source Tracing
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出LJ-Spoof数据集，用于音频防欺骗和合成源追踪任务。针对现有数据集不足，通过系统化变化生成参数，提升防欺骗与溯源能力。**

- **链接: [https://arxiv.org/pdf/2601.07958v1](https://arxiv.org/pdf/2601.07958v1)**

> **作者:** Surya Subramani; Hashim Ali; Hafiz Malik
>
> **摘要:** Speaker-specific anti-spoofing and synthesis-source tracing are central challenges in audio anti-spoofing. Progress has been hampered by the lack of datasets that systematically vary model architectures, synthesis pipelines, and generative parameters. To address this gap, we introduce LJ-Spoof, a speaker-specific, generatively diverse corpus that systematically varies prosody, vocoders, generative hyperparameters, bona fide prompt sources, training regimes, and neural post-processing. The corpus spans one speakers-including studio-quality recordings-30 TTS families, 500 generatively variant subsets, 10 bona fide neural-processing variants, and more than 3 million utterances. This variation-dense design enables robust speaker-conditioned anti-spoofing and fine-grained synthesis-source tracing. We further position this dataset as both a practical reference training resource and a benchmark evaluation suite for anti-spoofing and source tracing.
>
---
#### [new 007] VoxCog: Towards End-to-End Multilingual Cognitive Impairment Classification through Dialectal Knowledge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于认知障碍分类任务，旨在通过语音识别方言来检测阿尔茨海默病和轻度认知障碍。工作包括提出端到端框架VoxCog，利用预训练方言模型提升诊断准确率。**

- **链接: [https://arxiv.org/pdf/2601.07999v1](https://arxiv.org/pdf/2601.07999v1)**

> **作者:** Tiantian Feng; Anfeng Xu; Jinkook Lee; Shrikanth Narayanan
>
> **摘要:** In this work, we present a novel perspective on cognitive impairment classification from speech by integrating speech foundation models that explicitly recognize speech dialects. Our motivation is based on the observation that individuals with Alzheimer's Disease (AD) or mild cognitive impairment (MCI) often produce measurable speech characteristics, such as slower articulation rate and lengthened sounds, in a manner similar to dialectal phonetic variations seen in speech. Building on this idea, we introduce VoxCog, an end-to-end framework that uses pre-trained dialect models to detect AD or MCI without relying on additional modalities such as text or images. Through experiments on multiple multilingual datasets for AD and MCI detection, we demonstrate that model initialization with a dialect classifier on top of speech foundation models consistently improves the predictive performance of AD or MCI. Our trained models yield similar or often better performance compared to previous approaches that ensembled several computational methods using different signal modalities. Particularly, our end-to-end speech-based model achieves 87.5% and 85.9% accuracy on the ADReSS 2020 challenge and ADReSSo 2021 challenge test sets, outperforming existing solutions that use multimodal ensemble-based computation or LLMs.
>
---
#### [new 008] Decodable but not structured: linear probing enables Underwater Acoustic Target Recognition with pretrained audio embeddings
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于水下声学目标识别任务，旨在解决标记数据不足的问题。通过迁移学习和线性探测，利用预训练音频模型实现高效自动识别。**

- **链接: [https://arxiv.org/pdf/2601.08358v1](https://arxiv.org/pdf/2601.08358v1)**

> **作者:** Hilde I. Hummel; Sandjai Bhulai; Rob D. van der Mei; Burooj Ghani
>
> **摘要:** Increasing levels of anthropogenic noise from ships contribute significantly to underwater sound pollution, posing risks to marine ecosystems. This makes monitoring crucial to understand and quantify the impact of the ship radiated noise. Passive Acoustic Monitoring (PAM) systems are widely deployed for this purpose, generating years of underwater recordings across diverse soundscapes. Manual analysis of such large-scale data is impractical, motivating the need for automated approaches based on machine learning. Recent advances in automatic Underwater Acoustic Target Recognition (UATR) have largely relied on supervised learning, which is constrained by the scarcity of labeled data. Transfer Learning (TL) offers a promising alternative to mitigate this limitation. In this work, we conduct the first empirical comparative study of transfer learning for UATR, evaluating multiple pretrained audio models originating from diverse audio domains. The pretrained model weights are frozen, and the resulting embeddings are analyzed through classification, clustering, and similarity-based evaluations. The analysis shows that the geometrical structure of the embedding space is largely dominated by recording-specific characteristics. However, a simple linear probe can effectively suppress this recording-specific information and isolate ship-type features from these embeddings. As a result, linear probing enables effective automatic UATR using pretrained audio models at low computational cost, significantly reducing the need for a large amounts of high-quality labeled ship recordings.
>
---
#### [new 009] Elastic overtones: an equal temperament 12 tone music system with "perfect" fifths
- **分类: physics.soc-ph; cs.SD; eess.AS; physics.pop-ph**

- **简介: 论文提出一种新的12音律系统，解决传统十二平均律无法同时满足完美五度和八度的问题。通过调整谐波结构，实现和谐音程与调性灵活性的结合。**

- **链接: [https://arxiv.org/pdf/2601.08074v1](https://arxiv.org/pdf/2601.08074v1)**

> **作者:** X. Hernandez; Luis Nasser; Pablo Garcia-Valenzuela
>
> **备注:** 14 pages, 4 figures, 6 audio files
>
> **摘要:** The impossibility of a transposable 12 semitone tuning of the octave arises from the mathematical fact that $2 \times 2^{7/12} \neq 3$ i.e., the second harmonic of the fifth can not exactly match the third harmonic of the fundamental. This in turn, stems from the whole number harmonic structure of western music, and the subsequent fundamental character of the octave interval as multiples of 2 in frequency, a property inherited by our music system from the physics of instruments with vibrating elements being to a good approximation one dimensional. In the current era of electronic music, one can relax the above assumptions to construct an analogous music system where all the structural properties of the standard music system are preserved, but where harmonics are not whole number multiples of the fundamental frequency, and the octave is no longer a factor of 2 in frequency. This now allows to construct a transposable 12 semitone music system where the second harmonic of the fifth exactly matches the third harmonic of the fundamental. The enhanced harmonic qualities of this system recover to a good approximation the musical qualities of Just Intonation, whilst retaining by construction all the versatility and modulating ability of 12TET.
>
---
#### [new 010] FusID: Modality-Fused Semantic IDs for Generative Music Recommendation
- **分类: cs.IR; cs.SD; eess.AS**

- **简介: 该论文属于生成式音乐推荐任务，解决多模态冗余和交互不足问题。提出FusID框架，通过多模态融合、表示学习和量化提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2601.08764v1](https://arxiv.org/pdf/2601.08764v1)**

> **作者:** Haven Kim; Yupeng Hou; Julian McAuley
>
> **摘要:** Generative recommendation systems have achieved significant advances by leveraging semantic IDs to represent items. However, existing approaches that tokenize each modality independently face two critical limitations: (1) redundancy across modalities that reduces efficiency, and (2) failure to capture inter-modal interactions that limits item representation. We introduce FusID, a modality-fused semantic ID framework that addresses these limitations through three key components: (i) multimodal fusion that learns unified representations by jointly encoding information across modalities, (ii) representation learning that brings frequently co-occurring item embeddings closer while maintaining distinctiveness and preventing feature redundancy, and (iii) product quantization that converts the fused continuous embeddings into multiple discrete tokens to mitigate ID conflict. Evaluated on a multimodal next-song recommendation (i.e., playlist continuation) benchmark, FusID achieves zero ID conflicts, ensuring that each token sequence maps to exactly one song, mitigates codebook underutilization, and outperforms baselines in terms of MRR and Recall@k (k = 1, 5, 10, 20).
>
---
## 更新

#### [replaced 001] Apollo: Unified Multi-Task Audio-Video Joint Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文提出Apollo模型，解决音频视频联合生成中的同步性、对齐和泛化问题。通过改进架构、训练策略和数据集构建，实现高质量、多任务的音视频生成。**

- **链接: [https://arxiv.org/pdf/2601.04151v2](https://arxiv.org/pdf/2601.04151v2)**

> **作者:** Jun Wang; Chunyu Qiang; Yuxin Guo; Yiran Wang; Xijuan Zeng; Feng Deng
>
> **摘要:** Audio-video joint generation has progressed rapidly, yet substantial challenges still remain. Non-commercial approaches still suffer audio-visual asynchrony, poor lip-speech alignment, and unimodal degradation, which can be stemmed from weak audio-visual correspondence modeling, limited generalization, and scarce high-quality dense-caption data. To address these issues, we introduce Apollo and delve into three axes--model architecture, training strategy, and data curation. Architecturally, we adopt a single-tower design with unified DiT blocks and an Omni-Full Attention mechanism, achieving tight audio-visual alignment and strong scalability. Training-wise, we adopt a progressive multitask regime--random modality masking to joint optimization across tasks, and a multistage curriculum, yielding robust representations, strengthening A-V aligned world knowledge, and preventing unimodal collapse. For datasets, we present the first large-scale audio-video dataset with dense captions, and introduce a novel automated data-construction pipeline which annotates and filters millions of diverse, high-quality, strictly aligned audio-video-caption triplets. Building on this, Apollo scales to large datasets, delivering high-fidelity, semantically and temporally aligned, instruction-following generation in both joint and unimodal settings while generalizing robustly to out-of-distribution scenarios. Across tasks, it substantially outperforms prior methods by a large margin and achieves performance comparable to Veo 3, offering a unified, scalable path toward next-generation audio-video synthesis.
>
---
#### [replaced 002] HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决韩英混语识别难题。提出HiKE框架，包含真实数据和分层标注，用于评估和提升模型的混语识别能力。**

- **链接: [https://arxiv.org/pdf/2509.24613v4](https://arxiv.org/pdf/2509.24613v4)**

> **作者:** Gio Paik; Yongbeom Kim; Soungmin Lee; Sangmin Ahn; Chanwoo Kim
>
> **备注:** EACL Findings 2026
>
> **摘要:** Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible non-synthetic evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that although most multilingual ASR models initially exhibit inadequate CS-ASR performance, this capability can be enabled through fine-tuning with synthetic CS data. HiKE is available at https://github.com/ThetaOne-AI/HiKE.
>
---
#### [replaced 003] DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，旨在提升单通道语音质量。提出DPDFNet模型，通过双路径RNN增强时间与频带建模，优化性能并实现边缘设备部署。**

- **链接: [https://arxiv.org/pdf/2512.16420v2](https://arxiv.org/pdf/2512.16420v2)**

> **作者:** Daniel Rika; Nino Sapir; Ido Gus
>
> **摘要:** We present DPDFNet, a causal single-channel speech enhancement model that extends DeepFilterNet2 architecture with dual-path blocks in the encoder, strengthening long-range temporal and cross-band modeling while preserving the original enhancement framework. In addition, we demonstrate that adding a loss component to mitigate over-attenuation in the enhanced speech, combined with a fine-tuning phase tailored for "always-on" applications, leads to substantial improvements in overall model performance. To compare our proposed architecture with a variety of causal open-source models, we created a new evaluation set comprising long, low-SNR recordings in 12 languages across everyday noise scenarios, better reflecting real-world conditions than commonly used benchmarks. On this evaluation set, DPDFNet delivers superior performance to other causal open-source models, including some that are substantially larger and more computationally demanding. We also propose an holistic metric named PRISM, a composite, scale-normalized aggregate of intrusive and non-intrusive metrics, which demonstrates clear scalability with the number of dual-path blocks. We further demonstrate on-device feasibility by deploying DPDFNet on Ceva-NeuPro-Nano edge NPUs. Results indicate that DPDFNet-4, our second-largest model, achieves real-time performance on NPN32 and runs even faster on NPN64, confirming that state-of-the-art quality can be sustained within strict embedded power and latency constraints.
>
---
#### [replaced 004] A dataset and model for auditory scene recognition for hearing devices: AHEAD-DS and OpenYAMNet
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于听觉场景识别任务，旨在解决听力设备数据集不足和模型部署难题。通过构建AHEAD-DS数据集和提出OpenYAMNet模型，提升场景识别性能并实现实时应用。**

- **链接: [https://arxiv.org/pdf/2508.10360v5](https://arxiv.org/pdf/2508.10360v5)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard Lyon
>
> **摘要:** Scene recognition is important for hearing devices, however; this is challenging, in part because of the limitations of existing datasets. Datasets often lack public accessibility, completeness, or audiologically relevant labels, hindering systematic comparison of machine learning models. Deploying such models on resource-constrained edge devices presents another challenge.The proposed solution is two-fold, a repack and refinement of several open source datasets to create AHEAD-DS, a dataset designed for auditory scene recognition for hearing devices, and introduce OpenYAMNet, a sound recognition model. AHEAD-DS aims to provide a standardised, publicly available dataset with consistent labels relevant to hearing aids, facilitating model comparison. OpenYAMNet is designed for deployment on edge devices like smartphones connected to hearing devices, such as hearing aids and wireless earphones with hearing aid functionality, serving as a baseline model for sound-based scene recognition. OpenYAMNet achieved a mean average precision of 0.86 and accuracy of 0.93 on the testing set of AHEAD-DS across fourteen categories relevant to auditory scene recognition. Real-time sound-based scene recognition capabilities were demonstrated on edge devices by deploying OpenYAMNet to an Android smartphone. Even with a 2018 Google Pixel 3, a phone with modest specifications, the model processes audio with approximately 50ms of latency to load the model, and an approximate linear increase of 30ms per 1 second of audio. The project website with links to code, data, and models. https://github.com/Australian-Future-Hearing-Initiative
>
---
#### [replaced 005] Continuous Audio Language Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出连续音频语言模型（CALM），解决音频生成中质量与计算成本的矛盾，通过避免有损压缩提升音质和效率。任务为高质量音频生成。**

- **链接: [https://arxiv.org/pdf/2509.06926v3](https://arxiv.org/pdf/2509.06926v3)**

> **作者:** Simon Rouard; Manu Orsini; Axel Roebel; Neil Zeghidour; Alexandre Défossez
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Audio Language Models (ALM) have emerged as the dominant paradigm for speech and music generation by representing audio as sequences of discrete tokens. Yet, unlike text tokens, which are invertible, audio tokens are extracted from lossy codecs with a limited bitrate. As a consequence, increasing audio quality requires generating more tokens, which imposes a trade-off between fidelity and computational cost. We address this issue by studying Continuous Audio Language Models (CALM). These models instantiate a large Transformer backbone that produces a contextual embedding at every timestep. This sequential information then conditions an MLP that generates the next continuous frame of an audio VAE through consistency modeling. By avoiding lossy compression, CALM achieves higher quality at lower computational cost than their discrete counterpart. Experiments on speech and music demonstrate improved efficiency and fidelity over state-of-the-art discrete audio language models, facilitating lightweight, high-quality audio generation. Samples are available at hf.co/spaces/kyutai/calm-samples. Finally, we release Pocket TTS, an open-source 100M-parameter text-to-speech model that can run faster than real time on a laptop CPU: github.com/kyutai-labs/pocket-tts.
>
---
#### [replaced 006] A Scalable Pipeline for Enabling Non-Verbal Speech Generation and Understanding
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决非言语发声（NVs）在语音系统中被忽视的问题。提出一种可扩展的自动标注框架，生成真实多样的NVs数据集，提升情感交流能力。**

- **链接: [https://arxiv.org/pdf/2508.05385v2](https://arxiv.org/pdf/2508.05385v2)**

> **作者:** Runchuan Ye; Yixuan Zhou; Renjie Yu; Zijian Lin; Kehan Li; Xiang Li; Xin Liu; Guoyang Zeng; Zhiyong Wu
>
> **摘要:** Non-verbal Vocalizations (NVs), such as laughter and sighs, are vital for conveying emotion and intention in human speech, yet most existing speech systems neglect them, which severely compromises communicative richness and emotional intelligence. Existing methods for NVs acquisition are either costly and unscalable (relying on manual annotation/recording) or unnatural (relying on rule-based synthesis). To address these limitations, we propose a highly scalable automatic annotation framework to label non-verbal phenomena from natural speech, which is low-cost, easily extendable, and inherently diverse and natural. This framework leverages a unified detection model to accurately identify NVs in natural speech and integrates them with transcripts via temporal-semantic alignment method. Using this framework, we created and released \textbf{NonVerbalSpeech-38K}, a diverse, real-world dataset featuring 38,718 samples across 10 NV categories collected from in-the-wild media. Experimental results demonstrate that our dataset provides superior controllability for NVs generation and achieves comparable performance for NVs understanding.
>
---
