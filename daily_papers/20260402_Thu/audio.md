# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Sona: Real-Time Multi-Target Sound Attenuation for Noise Sensitivity
- **分类: cs.SD; cs.HC**

- **简介: 该论文提出Sona系统，解决噪声敏感人群在嘈杂环境中难以区分和抑制干扰声音的问题。通过实时多目标声音衰减技术，保留有用音频，提升听觉舒适度与环境意识。**

- **链接: [https://arxiv.org/pdf/2604.00447](https://arxiv.org/pdf/2604.00447)**

> **作者:** Jeremy Zhengqi Huang; Emani Hicks; Sidharth; Gillian R. Hayes; Dhruv Jain
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** For people with noise sensitivity, everyday soundscapes can be overwhelming. Existing tools such as active noise cancellation reduce discomfort by suppressing the entire acoustic environment, often at the cost of awareness of surrounding people and events. We present Sona, an interactive mobile system for real-time soundscape mediation that selectively attenuates bothersome sounds while preserving desired audio. Sona is built on a target-conditioned neural pipeline that supports simultaneous attenuation of multiple overlapping sound sources, overcoming the single-target limitation of prior systems. It runs in real time on-device and supports user-extensible sound classes through in-situ audio examples, without retraining. Sona is informed by a formative study with 68 noise-sensitive individuals. Through technical benchmarking and an in-situ study with 10 participants, we show that Sona achieves low-latency, multi-target attenuation suitable for live listening, and enables meaningful reductions in bothersome sounds while maintaining awareness of surroundings. These results point toward a new class of personal AI systems that support comfort and social participation by mediating real-world acoustic environments.
>
---
#### [new 002] FineLAP: Taming Heterogeneous Supervision for Fine-grained Language-Audio Pretraining
- **分类: cs.SD**

- **简介: 该论文提出FineLAP，解决音频-文本预训练中多粒度监督不一致的问题，通过双流损失和聚类采样提升片段级与帧级对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.01155](https://arxiv.org/pdf/2604.01155)**

> **作者:** Xiquan Li; Xuenan Xu; Ziyang Ma; Wenxi Chen; Haolin He; Qiuqiang Kong; Xie Chen
>
> **摘要:** Contrastively pretrained audio-language models (e.g., CLAP) excel at clip-level understanding but struggle with frame-level tasks. Existing extensions fail to exploit the varying granularity of real-world audio-text data, where massive clip-level textual descriptions coexist with limited frame-level annotations. This paper proposes Fine-grained Language-Audio Pretraining (FineLAP), a novel training paradigm that advances both clip- and frame-level alignment in CLAP with heterogeneous data. FineLAP introduces a dual-stream sigmoid loss with a cluster-based sampling strategy to jointly learn from clip- and frame-level supervision. To capture both global semantics and local details, FineLAP uses a decoupled audio projector on top of a self-supervised encoder. To alleviate the scarcity of temporally annotated data, we present FineLAP-100k, a large-scale synthetic SED dataset constructed through a scalable curation pipeline. Extensive experiments demonstrate that FineLAP achieves SOTA performance across multiple audio understanding tasks, including retrieval, classification, sound event detection, and text-to-audio grounding. Ablation studies further show that coarse- and fine-grained alignment are mutually beneficial, providing insights for building better audio-language models (ALMs).
>
---
#### [new 003] Description and Discussion on DCASE 2026 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes
- **分类: eess.AS**

- **简介: 该论文介绍DCASE 2026 Task 4任务，旨在解决复杂声场中声音事件的联合检测与分离问题，提升沉浸式通信基础。**

- **链接: [https://arxiv.org/pdf/2604.00776](https://arxiv.org/pdf/2604.00776)**

> **作者:** Masahiro Yasuda; Binh Thien Nguyen; Noboru Harada; Romain Serizel; Mayank Mishra; Marc Delcroix; Carlos Hernandez-Olivan; Shoko Araki; Daiki Takeuchi; Tomohiro Nakatani; Nobutaka Ono
>
> **摘要:** This paper presents an overview of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2026 Challenge Task 4, Spatial Semantic Segmentation of Sound Scenes (S5). The S5 task focuses on the joint detection and separation of sound events in complex spatial audio mixtures, contributing to the foundation of immersive communication. First introduced in DCASE 2025, the S5 task continues in DCASE 2026 Task 4 with key changes to better reflect real-world conditions, including allowing mixtures to contain multiple sources of the same class and to contain no target sources. In this paper, we describe task setting, along with the corresponding updates to the evaluation metrics and dataset. The experimental results of the submitted systems are also reported and analyzed. The official access point for data and code is this https URL.
>
---
#### [new 004] Diff-VS: Efficient Audio-Aware Diffusion U-Net for Vocals Separation
- **分类: eess.AS**

- **简介: 该论文属于语音分离任务，旨在解决现有生成方法在客观指标上表现不佳的问题。提出基于EDM框架的音频感知扩散U-Net模型，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2604.01120](https://arxiv.org/pdf/2604.01120)**

> **作者:** Yun-Ning; Hung; Richard Vogl; Filip Korzeniowski; Igor Pereira
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** While diffusion models are best known for their performance in generative tasks, they have also been successfully applied to many other tasks, including audio source separation. However, current generative approaches to music source separation often underperform on standard objective metrics. In this paper, we address this issue by introducing a novel generative vocal separation model based on the Elucidated Diffusion Model (EDM) framework. Our model processes complex short-time Fourier transform spectrograms and employs an improved U-Net architecture based on music-informed design choices. Our approach matches discriminative baselines on objective metrics and achieves perceptual quality comparable to state-of-the-art systems, as assessed by proxy subjective metrics. We hope these results encourage broader exploration of generative methods for music source separation
>
---
#### [new 005] VisG AV-HuBERT: Viseme-Guided AV-HuBERT
- **分类: eess.AS**

- **简介: 该论文属于音频-视觉语音识别任务，旨在提升噪声环境下模型性能。通过引入辅助的可视音素分类任务，增强编码器对视觉信息的依赖，改善语音单元区分能力。**

- **链接: [https://arxiv.org/pdf/2604.00982](https://arxiv.org/pdf/2604.00982)**

> **作者:** Aristeidis Papadopoulos; Rishabh Jain; Naomi Harte
>
> **备注:** Includes Supplementary Material. Accepted for Publication at International Conference on Pattern Recognition 2026 - ICPR 2026. Code is available at this https URL
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) systems nowadays integrate Large Language Model (LLM) decoders with transformer-based encoders, achieving state-of-the-art results. However, the relative contributions of improved language modelling versus enhanced audiovisual encoding remain unclear. We propose Viseme-Guided AV-HuBERT (VisG AV-HuBERT), a multi-task fine-tuning framework that incorporates auxiliary viseme classification to strengthen the model's reliance on visual articulatory features. By extending AV-HuBERT with a lightweight viseme prediction sub-network, this method explicitly guides the encoder to preserve visual speech information. Evaluated on LRS3, VisG AV-HuBERT achieves comparable or improved performance over the baseline AV-HuBERT, with notable gains under heavy noise conditions. WER reduces from 13.59% to 6.60% (51.4% relative improvement) at -10 dB Signal-to-Noise Ratio (SNR) for Speech noise. Deeper analysis reveals substantial reductions in substitution errors across noise types, demonstrating improved speech unit discrimination. Evaluation on LRS2 confirms generalization capability. Our results demonstrate that explicit viseme modelling enhances encoder representations, and provides a foundation for enhancing noise-robust AVSR through encoder-level improvements.
>
---
#### [new 006] MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于文本到语音合成任务，旨在通过纯状态空间模型替代注意力机制，提升效率与稳定性。工作包括设计新型编码器与解码器，优化条件生成流程。**

- **链接: [https://arxiv.org/pdf/2604.00292](https://arxiv.org/pdf/2604.00292)**

> **作者:** Sahil Kumar; Namrataben Patel; Honggang Wang; Youshan Zhang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** MambaVoiceCloning (MVC) asks whether the conditioning path of diffusion-based TTS can be made fully SSM-only at inference, removing all attention and explicit RNN-style recurrence layers across text, rhythm, and prosody, while preserving or improving quality under controlled conditions. MVC combines a gated bidirectional Mamba text encoder, a Temporal Bi-Mamba supervised by a lightweight alignment teacher discarded after training, and an Expressive Mamba with AdaLN modulation, yielding linear-time O(T) conditioning with bounded activation memory and practical finite look-ahead streaming. Unlike prior Mamba-TTS systems that remain hybrid at inference, MVC removes attention-based duration and style modules under a fixed StyleTTS2 mel-diffusion-vocoder backbone. Trained on LJSpeech/LibriTTS and evaluated on VCTK, CSS10 (ES/DE/FR), and long-form Gutenberg passages, MVC achieves modest but statistically reliable gains over StyleTTS2, VITS, and Mamba-attention hybrids in MOS/CMOS, F0 RMSE, MCD, and WER, while reducing encoder parameters to 21M and improving throughput by 1.6x. Diffusion remains the dominant latency source, but SSM-only conditioning improves memory footprint, stability, and deployability.
>
---
#### [new 007] TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文属于音频伪造检测任务，解决部分音频深度伪造的检测问题。提出TRACE方法，通过分析语音基础模型的嵌入轨迹动态，实现无需训练的检测。**

- **链接: [https://arxiv.org/pdf/2604.01083](https://arxiv.org/pdf/2604.01083)**

> **作者:** Awais Khan; Muhammad Umar Farooq; Kutub Uddin; Khalid Malik
>
> **摘要:** Partial audio deepfakes, where synthesized segments are spliced into genuine recordings, are particularly deceptive because most of the audio remains authentic. Existing detectors are supervised: they require frame-level annotations, overfit to specific synthesis pipelines, and must be retrained as new generative models emerge. We argue that this supervision is unnecessary. We hypothesize that speech foundation models implicitly encode a forensic signal: genuine speech forms smooth, slowly varying embedding trajectories, while splice boundaries introduce abrupt disruptions in frame-level transitions. Building on this, we propose TRACE (Training-free Representation-based Audio Countermeasure via Embedding dynamics), a training-free framework that detects partial audio deepfakes by analyzing the first-order dynamics of frozen speech foundation model representations without any training, labeled data, or architectural modification. We evaluate TRACE on four benchmarks that span two languages using six speech foundation models. In PartialSpoof, TRACE achieves 8.08% EER, competitive with fine-tuned supervised baselines. In LlamaPartialSpoof, the most challenging benchmark featuring LLM-driven commercial synthesis, TRACE surpasses a supervised baseline outright (24.12% vs. 24.49% EER) without any target-domain data. These results show that temporal dynamics in speech foundation models provide an effective, generalize signal for training-free audio forensics.
>
---
#### [new 008] Vocal Prognostic Digital Biomarkers in Monitoring Chronic Heart Failure: A Longitudinal Observational Study
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于医疗监测任务，旨在通过语音特征预测慢性心衰患者健康恶化。研究采集患者语音数据，分析其与健康状态的关系，以实现早期预警。**

- **链接: [https://arxiv.org/pdf/2604.00308](https://arxiv.org/pdf/2604.00308)**

> **作者:** Fan Wu; Matthias P. Nägele; Daryush D. Mehta; Elgar Fleisch; Frank Ruschitzka; Andreas J. Flammer; Filipe Barata
>
> **摘要:** Objective: This study aimed to evaluate which voice features can predict health deterioration in patients with chronic HF. Background: Heart failure (HF) is a chronic condition with progressive deterioration and acute decompensations, often requiring hospitalization and imposing substantial healthcare and economic burdens. Current standard-of-care (SoC) home monitoring, such as weight tracking, lacks predictive accuracy and requires high patient engagement. Voice is a promising non-invasive biomarker, though prior studies have mainly focused on acute HF stages. Methods: In a 2-month longitudinal study, 32 patients with HF collected daily voice recordings and SoC measures of weight and blood pressure at home, with biweekly questionnaires for health status. Acoustic analysis generated detailed vowel and speech features. Time-series features were extracted from aggregated lookback windows (e.g., 7 days) to predict next-day health status. Explainable machine learning with nested cross-validation identified top vocal biomarkers, and a case study illustrated model application. Results: A total of 21,863 recordings were analyzed. Acoustic vowel features showed strong correlations with health status. Time-series voice features within the lookback window outperformed corresponding standard care measures, achieving peak sensitivity and specificity of 0.826 and 0.782 versus 0.783 and 0.567 for SoC metrics. Key prognostic voice features identifying deterioration included delayed energy shift, low energy variability, and higher shimmer variability in vowels, along with reduced speaking and articulation rate, lower phonation ratio, decreased voice quality, and increased formant variability in speech. Conclusion: Voice-based monitoring offers a non-invasive approach to detect early health changes in chronic HF, supporting proactive and personalized care.
>
---
#### [new 009] Semantic Audio-Visual Navigation in Continuous Environments
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉导航任务，解决连续环境中目标失声导致的导航问题。提出MAGNet模型，结合多模态信息与历史上下文，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2603.19660](https://arxiv.org/pdf/2603.19660)**

> **作者:** Yichen Zeng; Hebaixu Wang; Meng Liu; Yu Zhou; Chen Gao; Kehan Chen; Gongping Huang
>
> **备注:** This paper has been accepted to CVPR 2026
>
> **摘要:** Audio-visual navigation enables embodied agents to navigate toward sound-emitting targets by leveraging both auditory and visual cues. However, most existing approaches rely on precomputed room impulse responses (RIRs) for binaural audio rendering, restricting agents to discrete grid positions and leading to spatially discontinuous observations. To establish a more realistic setting, we introduce Semantic Audio-Visual Navigation in Continuous Environments (SAVN-CE), where agents can move freely in 3D spaces and perceive temporally and spatially coherent audio-visual streams. In this setting, targets may intermittently become silent or stop emitting sound entirely, causing agents to lose goal information. To tackle this challenge, we propose MAGNet, a multimodal transformer-based model that jointly encodes spatial and semantic goal representations and integrates historical context with self-motion cues to enable memory-augmented goal reasoning. Comprehensive experiments demonstrate that MAGNet significantly outperforms state-of-the-art methods, achieving up to a 12.1\% absolute improvement in success rate. These results also highlight its robustness to short-duration sounds and long-distance navigation scenarios. The code is available at this https URL.
>
---
#### [new 010] OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出OmniVoice，解决多语言零样本文本转语音任务。通过创新架构直接将文本映射到声学标记，提升效率与清晰度。**

- **链接: [https://arxiv.org/pdf/2604.00688](https://arxiv.org/pdf/2604.00688)**

> **作者:** Han Zhu; Lingxuan Ye; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhifeng Han; Weiji Zhuang; Long Lin; Daniel Povey
>
> **摘要:** We present OmniVoice, a massive multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL.
>
---
#### [new 011] An Empirical Recipe for Universal Phone Recognition
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决多语言和低资源环境下语音识别性能不佳的问题。通过大规模数据训练，提出新方法 PhoneticXEUS，提升多语言及口音英语的识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.29042](https://arxiv.org/pdf/2603.29042)**

> **作者:** Shikhar Bharadwaj; Chin-Jou Li; Kwanghee Choi; Eunjung Yeo; William Chen; Shinji Watanabe; David R. Mortensen
>
> **备注:** Submitted to Interspeech 2026. Code: this https URL
>
> **摘要:** Phone recognition (PR) is a key enabler of multilingual and low-resource speech processing tasks, yet robust performance remains elusive. Highly performant English-focused models do not generalize across languages, while multilingual models underutilize pretrained representations. It also remains unclear how data scale, architecture, and training objective contribute to multilingual PR. We present PhoneticXEUS -- trained on large-scale multilingual data and achieving state-of-the-art performance on both multilingual (17.7% PFER) and accented English speech (10.6% PFER). Through controlled ablations with evaluations across 100+ languages under a unified scheme, we empirically establish our training recipe and quantify the impact of SSL representations, data scale, and loss objectives. In addition, we analyze error patterns across language families, accented speech, and articulatory features. All data and code are released openly.
>
---
## 更新

#### [replaced 001] How Open is Open TTS? A Practical Evaluation of Open Source TTS Tools
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在评估开源TTS工具在不同资源条件下的适用性。研究分析了四个框架的安装、数据准备和性能，揭示了在低资源环境中的挑战。**

- **链接: [https://arxiv.org/pdf/2603.24116](https://arxiv.org/pdf/2603.24116)**

> **作者:** Teodora Răgman; Adrian Bogdan Stânea; Horia Cucu; Adriana Stan
>
> **备注:** Published in IEEE Access this https URL
>
> **摘要:** Open-source text-to-speech (TTS) frameworks have emerged as highly adaptable platforms for developing speech synthesis systems across a wide range of languages. However, their applicability is not uniform -- particularly when the target language is under-resourced or when computational resources are constrained. In this study, we systematically assess the feasibility of building novel TTS models using four widely adopted open-source architectures: FastPitch, VITS, Grad-TTS, and Matcha-TTS. Our evaluation spans multiple dimensions, including qualitative aspects such as ease of installation, dataset preparation, and hardware requirements, as well as quantitative assessments of synthesis quality for Romanian. We employ both objective metrics and subjective listening tests to evaluate intelligibility, speaker similarity, and naturalness of the generated speech. The results reveal significant challenges in tool chain setup, data preprocessing, and computational efficiency, which can hinder adoption in low-resource contexts. By grounding the analysis in reproducible protocols and accessible evaluation criteria, this work aims to inform best practices and promote more inclusive, language-diverse TTS development. All information needed to reproduce this study (i.e. code and data) are available in our git repository: this https URL
>
---
#### [replaced 002] Robust Residual Finite Scalar Quantization for Neural Compression
- **分类: eess.IV; cs.CV; eess.AS**

- **简介: 该论文属于神经压缩任务，解决多阶段FSQ中残差幅度衰减问题，提出RFSQ方法，通过自适应缩放和层归一化提升性能。**

- **链接: [https://arxiv.org/pdf/2508.15860](https://arxiv.org/pdf/2508.15860)**

> **作者:** Xiaoxu Zhu; Xiaojie Yu; Guangchao Yao; Yiming Ren; Baoxiang Li
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Finite Scalar Quantization (FSQ) offers simplified training but suffers from residual magnitude decay in multi-stage settings, where subsequent stages receive exponentially weaker signals. We propose Robust Residual Finite Scalar Quantization (RFSQ), addressing this fundamental limitation through two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our experiments across audio and image modalities demonstrate RFSQ's effectiveness and generalizability. In audio reconstruction at 24 bits/frame, RFSQ-LayerNorm achieves 3.646 DNSMOS, a 3.6% improvement over state-of-the-art RVQ (3.518). On ImageNet, RFSQ achieves 0.102 L1 loss and 0.100 perceptual loss, with LayerNorm providing 9.7% L1 improvement and 17.4% perceptual improvement over unconditioned variants. The LayerNorm strategy consistently outperforms alternatives by maintaining normalized input statistics across stages, effectively preventing exponential magnitude decay that limits naive residual approaches. RFSQ combines FSQ's simplicity with multi-stage quantization's representational power, establishing a new standard for neural compression across diverse modalities.
>
---
#### [replaced 003] Enhancing Infant Crying Detection with Gradient Boosting for Improved Emotional and Mental Health Diagnostics
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于婴儿啼哭检测任务，旨在提升情感与心理健康诊断的准确性。通过融合音频特征与梯度提升方法，提高了啼哭识别性能。**

- **链接: [https://arxiv.org/pdf/2410.09236](https://arxiv.org/pdf/2410.09236)**

> **作者:** Kyunghun Lee; Lauren M. Henry; Eleanor Hansen; Elizabeth Tandilashvili; Lauren S. Wakschlag; Elizabeth Norton; Daniel S. Pine; Melissa A. Brotman; Francisco Pereira
>
> **摘要:** Infant crying can serve as a crucial indicator of various physiological and emotional states. This paper introduces a comprehensive approach detecting infant cries within audio data. We integrate Wav2Vec with traditional audio features and employ Gradient Boosting Machines for cry classification. We validate our approach on a real world dataset, demonstrating significant performance improvements over existing methods.
>
---
#### [replaced 004] CoDeTT: A Context-Aware Decision Benchmark for Turn-Taking Evaluation
- **分类: cs.SD**

- **简介: 该论文属于对话系统中的话轮转换任务，旨在解决评估碎片化问题。提出CoDeTT基准，构建多场景数据集，进行系统性评估。**

- **链接: [https://arxiv.org/pdf/2603.25434](https://arxiv.org/pdf/2603.25434)**

> **作者:** Huan Shen; Yingao Wang; Shangkun Huang; Wei Zou; Yunzhang Chen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Turn-taking modeling is fundamental to spoken dialogue systems, yet its evaluation remains fragmented and often limited to binary boundary detection under narrow interaction settings. Such protocols hinder systematic comparison and obscure model weaknesses across conversational conditions. We present CoDeTT, a context-aware decision benchmark for turn-taking evaluation. CoDeTT formulates turn-taking as a structured decision problem and constructs a multi-scenario dataset with fine-grained decision categories and controlled context variations. Under a unified evaluation protocol, we assess representative existing models and observe substantial performance disparities across decision types and interaction scenarios. CoDeTT provides a standardized benchmark for systematic and context-aware evaluation of turn-taking systems. The benchmark dataset and evaluation toolkit are available at this https URL.
>
---
#### [replaced 005] DuoTok: Source-Aware Dual-Track Tokenization for Multi-Track Music Language Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出DuoTok，解决多轨音乐语言建模中的音频分词问题，通过双轨结构实现高保真与强预测性的平衡。**

- **链接: [https://arxiv.org/pdf/2511.20224](https://arxiv.org/pdf/2511.20224)**

> **作者:** Rui Lin; Zhiyue Wu; Jiahe Le; Kangdi Wang; Weixiong Chen; Junyu Dai; Tao Jiang
>
> **备注:** 17 pages, 5 figures, 8 tables. Project page: this https URL
>
> **摘要:** Audio tokenization bridges continuous waveforms and multi-track music language models. In dual-track modeling, tokens should preserve three properties at once: high-fidelity reconstruction, strong predictability under a language model, and cross-track correspondence. We introduce DuoTok, a source-aware dual-track tokenizer that addresses this trade-off through staged disentanglement. DuoTok first pretrains a semantic encoder, then regularizes it with multi-task supervision, freezes the encoder, and applies hard dual-codebook routing while keeping auxiliary objectives on quantized codes. A diffusion decoder reconstructs high-frequency details, allowing tokens to focus on structured information for sequence modeling. On standard benchmarks, DuoTok achieves a favorable predictability-fidelity trade-off, reaching the lowest cnBPT while maintaining competitive reconstruction at 0.75 kbps. Under a held-constant dual-track language modeling protocol, enBPT also improves, indicating gains beyond codebook size effects. Controlled diagnostics show larger predictability costs under cross-track corruption and larger gains from longer context, suggesting that models trained on DuoTok tokens use cross-track structure and non-local history.
>
---
#### [replaced 006] Measuring Prosody Diversity in Zero-Shot TTS: A New Metric, Benchmark, and Exploration
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决零样本TTS中韵律多样性评估问题。提出新指标DS-WED和数据集ProsodyEval，以更准确衡量韵律变化。**

- **链接: [https://arxiv.org/pdf/2509.19928](https://arxiv.org/pdf/2509.19928)**

> **作者:** Yifan Yang; Bing Han; Hui Wang; Long Zhou; Wei Wang; Mingyu Cui; Xu Tan; Xie Chen
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** Prosody diversity is essential for achieving naturalness and expressiveness in zero-shot text-to-speech (TTS). However, frequently used acoustic metrics capture only partial views of prosodic variation and correlate poorly with human perception, leaving the problem of reliably quantifying prosody diversity underexplored. To bridge this gap, we introduce ProsodyEval, a prosody diversity assessment dataset that provides Prosody Mean Opinion Score (PMOS) alongside conventional acoustic metrics. ProsodyEval comprises 1000 speech samples derived from 7 mainstream TTS systems, with 2000 human ratings. Building on this, we propose the Discretized Speech Weighted Edit Distance (DS-WED), a new objective diversity metric that quantifies prosodic variation via weighted edit distance over semantic tokens. Experiments on ProsodyEval show that DS-WED achieves substantially higher correlation with human judgments than existing acoustic metrics, while remaining highly robust in speech tokenization from HuBERT and WavLM. Leveraging DS-WED, we benchmark state-of-the-art open-source TTS systems on LibriSpeech test-clean and Seed-TTS test-en, and further explorations uncover several factors that influence prosody diversity, including generative modeling paradigms, duration control, and reinforcement learning. Moreover, we find that current large audio language models (LALMs) remain limited in capturing prosodic variations. Audio samples are available at this https URL.
>
---
#### [replaced 007] Speaker Disentanglement of Speech Pre-trained Model Based on Interpretability
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音模型去耦任务，旨在解决内容与说话人信息纠缠的问题。通过可解释性分析，提出方法有效去除嵌入中的说话人信息，同时保持识别准确率。**

- **链接: [https://arxiv.org/pdf/2507.17851](https://arxiv.org/pdf/2507.17851)**

> **作者:** Xiaoxu Zhu; Junhua Li; Aaron J. Li; Guangchao Yao; Xiaojie Yu
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Self-supervised speech models learn representations that capture both content and speaker information. Yet this entanglement creates problems: content tasks suffer from speaker bias, and privacy concerns arise when speaker identity leaks through supposedly anonymized representations. We present two contributions to address these challenges. First, we develop InterpTRQE-SptME (Timbre Residual Quantitative Evaluation Benchmark of Speech pre-training Models Encoding via Interpretability), a benchmark that directly measures residual speaker information in content embeddings using SHAP-based interpretability analysis. Unlike existing indirect metrics, our approach quantifies the exact proportion of speaker information remaining after disentanglement. Second, we propose InterpTF-SptME, which uses these interpretability insights to filter speaker information from embeddings. Testing on VCTK with seven models including HuBERT, WavLM, and ContentVec, we find that SHAP Noise filtering reduces speaker residuals from 18.05% to nearly zero while maintaining recognition accuracy (CTC loss increase under 1%). The method is model-agnostic and requires no retraining.
>
---
#### [replaced 008] Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音生物识别任务，解决性别相关的性能差异问题。通过Fair-Gate框架，减少性别偏差，提升公平性与识别效果。**

- **链接: [https://arxiv.org/pdf/2603.11360](https://arxiv.org/pdf/2603.11360)**

> **作者:** Yangyang Qu; Massimiliano Todisco; Chiara Galdi; Nicholas Evans
>
> **摘要:** Voice biometric systems can exhibit sex-related performance gaps even when overall verification accuracy is strong. We attribute these gaps to two practical mechanisms: (i) demographic shortcut learning, where speaker classification training exploits spurious correlations between sex and speaker identity, and (ii) feature entanglement, where sex-linked acoustic variation overlaps with identity cues and cannot be removed without degrading speaker discrimination. We propose Fair-Gate, a fairness-aware and interpretable risk-gating framework that addresses both mechanisms in a single pipeline. Fair-Gate applies risk extrapolation to reduce variation in speaker-classification risk across proxy sex groups, and introduces a local complementary gate that routes intermediate features into an identity branch and a sex branch. The gate provides interpretability by producing an explicit routing mask that can be inspected to understand which features are allocated to identity versus sex-related pathways. Experiments on VoxCeleb1 show that Fair-Gate improves the utility--fairness trade-off, yielding more sex-fair ASV performance under challenging evaluation conditions.
>
---
#### [replaced 009] MATHDance: Mamba-Transformer Architecture with Uniform Tokenization for High-Quality 3D Dance Generation
- **分类: cs.SD; cs.GR; cs.MM; eess.AS**

- **简介: 该论文属于音乐到舞蹈生成任务，旨在解决舞蹈动作与音乐不一致的问题。提出MatchDance框架，结合Mamba-Transformer架构和量化技术，提升生成舞蹈的质量与一致性。**

- **链接: [https://arxiv.org/pdf/2505.14222](https://arxiv.org/pdf/2505.14222)**

> **作者:** Kaixing Yang; Xulong Tang; Ziqiao Peng; Yuxuan Hu; Xiangyue Zhang; Puwei Wang; Hongyan Liu; Jun He; Zhaoxin Fan
>
> **摘要:** Music-to-dance generation represents a challenging yet pivotal task at the intersection of choreography, virtual reality, and creative content generation. Despite its significance, existing methods face substantial limitation in achieving choreographic consistency. To address the challenge, we propose MatchDance, a novel framework for music-to-dance generation that constructs a latent representation to enhance choreographic consistency. MatchDance employs a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) with kinematic-dynamic constraints and reconstructs them with high fidelity, and (2) a Hybrid Music-to-Dance Generation Stage(HMDGS), which uses a Mamba-Transformer hybrid architecture to map music into the latent representation, followed by the KDQS decoder to generate 3D dance motions. Additionally, a music-dance retrieval framework and comprehensive metrics are introduced for evaluation. Extensive experiments on the FineDance dataset demonstrate state-of-the-art performance.
>
---
