# 音频 cs.SD;  eess.AS

- **最新发布 22 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Distilling LLM Semantic Priors into Encoder-Only Multi-Talker ASR with Talker-Count Routing
- **分类: cs.SD**

- **简介: 该论文属于多说话人自动语音识别任务，旨在解决LLM在多说话人场景下计算成本高、稳定性差的问题。提出一种编码器-only框架，通过知识蒸馏将LLM语义先验融入模型，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.10587](https://arxiv.org/pdf/2603.10587)**

> **作者:** Hao Shi; Yusuke Fujita; Roman Koshkin; Mengjie Zhao; Yuan Gao; Lianbo Liu; Yui Sudo
>
> **摘要:** Large language models (LLMs) provide strong semantic priors that can improve multi-talker automatic speech recognition (MT-ASR), but using an LLM as an autoregressive decoder is computationally expensive and remains fragile under heavy overlap. In this paper, we propose an encoder-only MT-ASR framework that adapts an LLM to multi-talker conditioning and distills its semantic guidance into the encoder during training, while retaining fast CTC-style decoding at inference. Our model employs a post-encoder separator with serialized CTC to produce talker-ordered transcripts, and leverages an adapted LLM-based SOT objective as a multi-talker-aware teacher signal to explicitly regularize mixed-speech representations. To further support variable numbers of talkers, we introduce a Talker-Count Head that predicts the talker count and dynamically selects the appropriate decoding branch. Experiments on LibriMix show that the proposed encoder-only model achieves comparable performance to LLM-based systems in the two-talker condition, while delivering significant improvements in the three-talker condition with significant small RTF.
>
---
#### [new 002] When Fine-Tuning Fails and when it Generalises: Role of Data Diversity and Mixed Training in LLM-based TTS
- **分类: cs.SD; cs.AI; cs.ET**

- **简介: 该论文属于语音合成任务，解决LLM在语音克隆中的适应性问题。通过LoRA微调提升语音一致性、音质和说话人相似度，强调数据多样性的重要性。**

- **链接: [https://arxiv.org/pdf/2603.10904](https://arxiv.org/pdf/2603.10904)**

> **作者:** Anupam Purwar; Aditya Choudhary
>
> **备注:** We finetune the Qwen 0.5B backbone in an LLM TTS with LoRA to raise MOS speaker similarity and SNR. It works best with diverse training audio with uniform data it can amplify noise so tune decoding and use GGUF quantization for low latency stable quality
>
> **摘要:** Large language models are increasingly adopted as semantic backbones for neural text-to-speech systems. However, frozen LLM representations are insufficient for modeling speaker specific acoustic and perceptual characteristics. Our experiments involving fine tuning of the Language Model backbone of TTS show promise in improving the voice consistency and Signal to Noise ratio SNR in voice cloning task. Across multiple speakers LoRA finetuning consistently outperforms the non-finetuned base Qwen-0.5B model across three complementary dimensions of speech quality. First, perceptual quality improves significantly with DNS-MOS gains of up to 0.42 points for speakers whose training data exhibits sufficient acoustic variability. Second, speaker fidelity improves for all evaluated speakers with consistent increases in voice similarity indicating that LoRA effectively adapts speaker identity representations without degrading linguistic modeling. Third, signal level quality improves in most cases with signal to noise ratio increasing by as much as 34 percent. Crucially these improvements are strongly governed by the characteristics of the training data. Speakers with high variability in acoustic energy and perceptual quality achieve simultaneous gains in DNS-MOS voice similarity and SNR. Overall this work establishes that LoRA finetuning is not merely a parameter efficient optimization technique but an effective mechanism for better speaker level adaptation in compact LLM-based TTS systems. When supported by sufficiently diverse training data LoRA adapted Qwen-0.5B consistently surpasses its frozen base model in perceptual quality speaker similarity with low latency using GGUF model hosted in quantized form.
>
---
#### [new 003] FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System
- **分类: eess.AS; cs.SD**

- **简介: 本文提出FireRedASR2S，一个集成ASR、VAD、LID和Punc的工业级语音识别系统，解决多语言、多方言及语音转写问题。**

- **链接: [https://arxiv.org/pdf/2603.10420](https://arxiv.org/pdf/2603.10420)**

> **作者:** Kaituo Xu; Yan Jia; Kai Huang; Junjie Chen; Wenpeng Li; Kun Liu; Feng-Long Xie; Xu Tang; Yao Hu
>
> **摘要:** We present FireRedASR2S, a state-of-the-art industrial-grade all-in-one automatic speech recognition (ASR) system. It integrates four modules in a unified pipeline: ASR, Voice Activity Detection (VAD), Spoken Language Identification (LID), and Punctuation Prediction (Punc). All modules achieve SOTA performance on the evaluated benchmarks: FireRedASR2: An ASR module with two variants, FireRedASR2-LLM (8B+ parameters) and FireRedASR2-AED (1B+ parameters), supporting speech and singing transcription for Mandarin, Chinese dialects and accents, English, and code-switching. Compared to FireRedASR, FireRedASR2 delivers improved recognition accuracy and broader dialect and accent coverage. FireRedASR2-LLM achieves 2.89% average CER on 4 public Mandarin benchmarks and 11.55% on 19 public Chinese dialects and accents benchmarks, outperforming competitive baselines including Doubao-ASR, Qwen3-ASR, and Fun-ASR. FireRedVAD: An ultra-lightweight module (0.6M parameters) based on the Deep Feedforward Sequential Memory Network (DFSMN), supporting streaming VAD, non-streaming VAD, and multi-label VAD (mVAD). On the FLEURS-VAD-102 benchmark, it achieves 97.57% frame-level F1 and 99.60% AUC-ROC, outperforming Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD. FireRedLID: An Encoder-Decoder LID module supporting 100+ languages and 20+ Chinese dialects and accents. On FLEURS (82 languages), it achieves 97.18% utterance-level accuracy, outperforming Whisper and SpeechBrain. FireRedPunc: A BERT-style punctuation prediction module for Chinese and English. On multi-domain benchmarks, it achieves 78.90% average F1, outperforming FunASR-Punc (62.77%). To advance research in speech processing, we release model weights and code at this https URL.
>
---
#### [new 004] AlphaFlowTSE: One-Step Generative Target Speaker Extraction via Conditional AlphaFlow
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于目标说话人提取任务，旨在从多人混音中提取目标语音。针对多步采样延迟高和单步方法依赖不可靠时间坐标的问题，提出AlphaFlowTSE模型，提升语音相似性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10701](https://arxiv.org/pdf/2603.10701)**

> **作者:** Duojia Li; Shuhan Zhang; Zihan Qian; Wenxuan Wu; Shuai Wang; Qingyang Hong; Lin Li; Haizhou Li
>
> **备注:** Submitted to Interspeech 2026 for review
>
> **摘要:** In target speaker extraction (TSE), we aim to recover target speech from a multi-talker mixture using a short enrollment utterance as reference. Recent studies on diffusion and flow-matching generators have improved target-speech fidelity. However, multi-step sampling increases latency, and one-step solutions often rely on a mixture-dependent time coordinate that can be unreliable for real-world conversations. We present AlphaFlowTSE, a one-step conditional generative model trained with a Jacobian-vector product (JVP)-free AlphaFlow objective. AlphaFlowTSE learns mean-velocity transport along a mixture-to-target trajectory starting from the observed mixture, eliminating auxiliary mixing-ratio prediction, and stabilizes training by combining flow matching with an interval-consistency teacher-student target. Experiments on Libri2Mix and REAL-T confirm that AlphaFlowTSE improves target-speaker similarity and real-mixture generalization for downstream automatic speech recognition (ASR).
>
---
#### [new 005] Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于说话人验证任务，旨在解决语音感知大模型是否编码说话人身份的问题。通过提出评分协议并引入轻量增强方法，提升模型的说话人识别能力。**

- **链接: [https://arxiv.org/pdf/2603.10827](https://arxiv.org/pdf/2603.10827)**

> **作者:** Thomas Thebaud; Yuzhe Wang; Laureano Moro-Velazquez; Jesus Villalba-Lopez; Najim Dehak
>
> **备注:** 3 Tables, 1 Figure, Under review
>
> **摘要:** Speech-aware large language models (LLMs) can accept speech inputs, yet their training objectives largely emphasize linguistic content or specific fields such as emotions or the speaker's gender, leaving it unclear whether they encode speaker identity. First, we propose a model-agnostic scoring protocol that produces continuous verification scores for both API-only and open-weight models, using confidence scores or log-likelihood ratios from the Yes/No token probabilities. Using this protocol, we benchmark recent speech-aware LLMs and observe weak speaker discrimination (EERs above 20% on VoxCeleb1). Second, we introduce a lightweight augmentation that equips an LLM with ASV capability by injecting frozen ECAPA-TDNN speaker embeddings through a learned projection and training only LoRA adapters. On TinyLLaMA-1.1B, the resulting ECAPA-LLM achieves 1.03% EER on VoxCeleb1-E, approaching a dedicated speaker verification system while preserving a natural-language interface.
>
---
#### [new 006] OSUM-Pangu: An Open-Source Multidimension Speech Understanding Foundation Model Built upon OpenPangu on Ascend NPUs
- **分类: cs.SD**

- **简介: 该论文提出OSUM-Pangu，一个基于Ascend NPU的开源语音理解模型，解决非CUDA平台部署问题，实现多模态语音理解。**

- **链接: [https://arxiv.org/pdf/2603.10862](https://arxiv.org/pdf/2603.10862)**

> **作者:** Yujie Liao; Xuelong Geng; Hongfei Xue; Shuiyuan Wang; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advancements in Speech Large Language Models have significantly enhanced multi-dimensional speech understanding. However, the majority of high-performance frameworks are predominantly optimized for GPU centric ecosystems and proprietary backbones, creating a significant gap for deployment on non-CUDA computing infrastructures. In this paper, we present OSUM-Pangu, a fully open-source speech understanding foundation model developed on a completely non-CUDA software and hardware stack. By integrating an audio encoder with the openPangu-7B LLM backbone, we successfully implement the entire training and inference pipeline on the Ascend NPU platform. To facilitate efficient task alignment under non-CUDA resource constraints, we adopt a practical training process that sequentially bridges speech perception and user intent recognition. Experimental results demonstrate that OSUM-Pangu achieves task accuracy comparable to mainstream GPU-based models while maintaining robust natural language interaction capabilities. Our work provides a reproducible, non-CUDA baseline for the open-source speech community, promoting the independent evolution of multimodal intelligence.
>
---
#### [new 007] G-STAR: End-to-End Global Speaker-Tracking Attributed Recognition
- **分类: eess.AS; cs.AI; cs.HC; cs.MM; cs.SD**

- **简介: 该论文提出G-STAR系统，解决长时多说话人语音的时空标注问题，通过结合时间感知跟踪模块与语音大模型，实现一致的说话人身份识别与时间标注。**

- **链接: [https://arxiv.org/pdf/2603.10468](https://arxiv.org/pdf/2603.10468)**

> **作者:** Jing Peng; Ziyi Chen; Haoyu Li; Yucheng Wang; Duo Ma; Mengtian Li; Yunfan Du; Dezhu Xu; Kai Yu; Shuai Wang
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** We study timestamped speaker-attributed ASR for long-form, multi-party speech with overlap, where chunk-wise inference must preserve meeting-level speaker identity consistency while producing time-stamped, speaker-labeled transcripts. Previous Speech-LLM systems tend to prioritize either local diarization or global labeling, but often lack the ability to capture fine-grained temporal boundaries or robust cross-chunk identity linking. We propose G-STAR, an end-to-end system that couples a time-aware speaker-tracking module with a Speech-LLM transcription backbone. The tracker provides structured speaker cues with temporal grounding, and the LLM generates attributed text conditioned on these cues. G-STAR supports both component-wise optimization and joint end-to-end training, enabling flexible learning under heterogeneous supervision and domain shift. Experiments analyze cue fusion, local versus long-context trade-offs and hierarchical objectives.
>
---
#### [new 008] Speech Codec Probing from Semantic and Phonetic Perspectives
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音编码任务，旨在解决语音表示与文本语义不匹配的问题。通过分析语音分词器的语义和语音内容，发现其主要捕捉语音特征而非语义结构，为下一代语音编码方法提供指导。**

- **链接: [https://arxiv.org/pdf/2603.10371](https://arxiv.org/pdf/2603.10371)**

> **作者:** Xuan Shi; Chang Zeng; Tiantian Feng; Shih-Heng Wang; Jianbo Ma; Shrikanth Narayanan
>
> **摘要:** Speech tokenizers are essential for connecting speech to large language models (LLMs) in multimodal systems. These tokenizers are expected to preserve both semantic and acoustic information for downstream understanding and generation. However, emerging evidence suggests that what is termed "semantic" in speech representations does not align with text-derived semantics: a mismatch that can degrade multimodal LLM performance. In this paper, we systematically analyze the information encoded by several widely used speech tokenizers, disentangling their semantic and phonetic content through word-level probing tasks, layerwise representation analysis, and cross-modal alignment metrics such as CKA. Our results show that current tokenizers primarily capture phonetic rather than lexical-semantic structure, and we derive practical implications for the design of next-generation speech tokenization methods.
>
---
#### [new 009] Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有方法泛化能力差和缺乏可解释性的问题。提出HIR-SDD框架，结合大音频语言模型与人类推理，提升检测效果并提供合理解释。**

- **链接: [https://arxiv.org/pdf/2603.10725](https://arxiv.org/pdf/2603.10725)**

> **作者:** Artem Dvirniak; Evgeny Kushnir; Dmitrii Tarasov; Artem Iudin; Oleg Kiriukhin; Mikhail Pautov; Dmitrii Korzh; Oleg Y. Rogov
>
> **摘要:** The modern generative audio models can be used by an adversary in an unlawful manner, specifically, to impersonate other people to gain access to private information. To mitigate this issue, speech deepfake detection (SDD) methods started to evolve. Unfortunately, current SDD methods generally suffer from the lack of generalization to new audio domains and generators. More than that, they lack interpretability, especially human-like reasoning that would naturally explain the attribution of a given audio to the bona fide or spoof class and provide human-perceptible cues. In this paper, we propose HIR-SDD, a novel SDD framework that combines the strengths of Large Audio Language Models (LALMs) with the chain-of-thought reasoning derived from the novel proposed human-annotated dataset. Experimental evaluation demonstrates both the effectiveness of the proposed method and its ability to provide reasonable justifications for predictions.
>
---
#### [new 010] Geo-ATBench: A Benchmark for Geospatial Audio Tagging with Geospatial Semantic Context
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出Geo-AT任务，解决多标签音频事件识别中的歧义问题。通过引入地理语义上下文，提升音频分类性能，并构建了Geo-ATBench基准与GeoFusion-AT框架。**

- **链接: [https://arxiv.org/pdf/2603.10623](https://arxiv.org/pdf/2603.10623)**

> **作者:** Yuanbo Hou; Yanru Wu; Qiaoqiao Ren; Shengchen Li; Stephen Roberts; Dick Botteldooren
>
> **摘要:** Environmental sound understanding in computational auditory scene analysis (CASA) is often formulated as an audio-only recognition problem. This formulation leaves a persistent drawback in multi-label audio tagging (AT): acoustic similarity can make certain events difficult to separate from waveforms alone. In such cases, disambiguating cues often lie outside the waveform. Geospatial semantic context (GSC), derived from geographic information system data, e.g., points of interest (POI), provides location-tied environmental priors that can help reduce this ambiguity. A systematic study of this direction is enabled through the proposed geospatial audio tagging (Geo-AT) task, which conditions multi-label sound event tagging on GSC alongside audio. To benchmark Geo-AT, Geo-ATBench is introduced as a polyphonic audio benchmark with geographical annotations, containing 10.71 hours of audio across 28 event categories; each clip is paired with a GSC representation from 11 semantic context categories. GeoFusion-AT is proposed as a unified geo-audio fusion framework that evaluates feature-, representation-, and decision-level fusion on representative audio backbones, with audio- and GSC-only baselines. Results show that incorporating GSC improves AT performance, especially on acoustically confounded labels, indicating geospatial semantics provide effective priors beyond audio alone. A crowdsourced listening study with 10 participants on 579 samples shows that there is no significant difference in performance between models on Geo-ATBench labels and aggregated human labels, supporting Geo-ATBench as a human-aligned benchmark. The Geo-AT task, benchmark Geo-ATBench, and reproducible geo-audio fusion framework GeoFusion-AT provide a foundation for studying AT with geospatial semantic context within the CASA community. Dataset, code, models are on homepage (this https URL).
>
---
#### [new 011] Calibration-Reasoning Framework for Descriptive Speech Quality Assessment
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音质量评估任务，旨在提升描述性质量评估的准确性。通过校准和强化学习方法，增强模型对音频缺陷的识别与定位能力。**

- **链接: [https://arxiv.org/pdf/2603.10175](https://arxiv.org/pdf/2603.10175)**

> **作者:** Elizaveta Kostenok; Mathieu Salzmann; Milos Cernak
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Explainable speech quality assessment requires moving beyond Mean Opinion Scores (MOS) to analyze underlying perceptual dimensions. To address this, we introduce a novel post-training method that tailors the foundational Audio Large Language Model for multidimensional reasoning, detection and classification of audio artifacts. First, a calibration stage aligns the model to predict predefined perceptual dimensions. Second, a reinforcement learning stage leverages Group Relative Policy Optimization (GRPO) with dimension-specific rewards to heavily enhance accuracy of descriptions and temporal localization of quality issues. With this approach we reach state-of-the-art results of 0.71 mean PCC score on the multidimensional QualiSpeech benchmark and 13% improvement in MOS prediction driven by RL-based reasoning. Furthermore, our fine-grained GRPO rewards substantially advance the model's ability to pinpoint and classify audio artifacts in time.
>
---
#### [new 012] nlm: Real-Time Non-linear Modal Synthesis in Max
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决实时非线性模态合成的问题。通过开发Max外挂模块nlm，实现对弦、膜、板的物理参数交互控制与多通道输出。**

- **链接: [https://arxiv.org/pdf/2603.10240](https://arxiv.org/pdf/2603.10240)**

> **作者:** Rodrigo Diaz; Rodrigo Constanzo; Mark Sandler
>
> **备注:** accepted to PdMaxCon25~ (this https URL)
>
> **摘要:** We present \texttt{nlm}, a set of Max externals that enable efficient real-time non-linear modal synthesis for strings, membranes, and plates. The externals, implemented in C++, offer interactive control of physical parameters, allow the loading of custom modal data, and provide multichannel output. By integrating interactive physical-modelling capabilities into a familiar environment, \texttt{nlm} lowers the barrier for composers, performers, and sound designers to explore the expressive potential of non-linear modal synthesis. The externals are available as open-source software at this https URL.
>
---
#### [new 013] ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA
- **分类: cs.SD; cs.CV; cs.GR**

- **简介: 该论文提出ID-LoRA，解决音视频个性化生成任务，通过联合生成外观和声音，提升语音相似度与风格控制。**

- **链接: [https://arxiv.org/pdf/2603.10256](https://arxiv.org/pdf/2603.10256)**

> **作者:** Aviad Dahan; Moran Yanuka; Noa Kraicer; Lior Wolf; Raja Giryes
>
> **摘要:** Existing video personalization methods preserve visual likeness but treat video and audio separately. Without access to the visual scene, audio models cannot synchronize sounds with on-screen actions; and because classical voice-cloning models condition only on a reference recording, a text prompt cannot redirect speaking style or acoustic environment. We propose ID-LoRA (Identity-Driven In-Context LoRA), which jointly generates a subject's appearance and voice in a single model, letting a text prompt, a reference image, and a short audio clip govern both modalities together. ID-LoRA adapts the LTX-2 joint audio-video diffusion backbone via parameter-efficient In-Context LoRA and, to our knowledge, is the first method to personalize visual appearance and voice in a single generative pass. Two challenges arise. Reference and generation tokens share the same positional-encoding space, making them hard to distinguish; we address this with negative temporal positions, placing reference tokens in a disjoint RoPE region while preserving their internal temporal structure. Speaker characteristics also tend to be diluted during denoising; we introduce identity guidance, a classifier-free guidance variant that amplifies speaker-specific features by contrasting predictions with and without the reference signal. In human preference studies, ID-LoRA is preferred over Kling 2.6 Pro by 73% of annotators for voice similarity and 65% for speaking style. On cross-environment settings, speaker similarity improves by 24% over Kling, with the gap widening as conditions diverge. A preliminary user study further suggests that joint generation provides a useful inductive bias for physically grounded sound synthesis. ID-LoRA achieves these results with only ~3K training pairs on a single GPU. Code, models, and data will be released.
>
---
#### [new 014] Probabilistic Verification of Voice Anti-Spoofing Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音防欺骗任务，解决现有模型缺乏鲁棒性保证的问题。提出PV-VASM框架，验证语音反欺骗模型的鲁棒性，有效应对未知合成技术。**

- **链接: [https://arxiv.org/pdf/2603.10713](https://arxiv.org/pdf/2603.10713)**

> **作者:** Evgeny Kushnir; Alexandr Kozodaev; Dmitrii Korzh; Mikhail Pautov; Oleg Kiriukhin; Oleg Y. Rogov
>
> **摘要:** Recent advances in generative models have amplified the risk of malicious misuse of speech synthesis technologies, enabling adversaries to impersonate target speakers and access sensitive resources. Although speech deepfake detection has progressed rapidly, most existing countermeasures lack formal robustness guarantees or fail to generalize to unseen generation techniques. We propose PV-VASM, a probabilistic framework for verifying the robustness of voice anti-spoofing models (VASMs). PV-VASM estimates the probability of misclassification under text-to-speech (TTS), voice cloning (VC), and parametric signal transformations. The approach is model-agnostic and enables robustness verification against unseen speech synthesis techniques and input perturbations. We derive a theoretical upper bound on the error probability and validate the method across diverse experimental settings, demonstrating its effectiveness as a practical robustness verification tool.
>
---
#### [new 015] MoXaRt: Audio-Visual Object-Guided Sound Interaction for XR
- **分类: cs.SD; cs.CV; cs.HC**

- **简介: 该论文提出MoXaRt系统，解决XR中复杂声景干扰问题，通过音视频协同分离声音源，提升语音可懂度与用户体验。**

- **链接: [https://arxiv.org/pdf/2603.10465](https://arxiv.org/pdf/2603.10465)**

> **作者:** Tianyu Xu; Sieun Kim; Qianhui Zheng; Ruoyu Xu; Tejasvi Ravi; Anuva Kulkarni; Katrina Passarella-Ward; Junyi Zhu; Adarsh Kowdle
>
> **摘要:** In Extended Reality (XR), complex acoustic environments often overwhelm users, compromising both scene awareness and social engagement due to entangled sound sources. We introduce MoXaRt, a real-time XR system that uses audio-visual cues to separate these sources and enable fine-grained sound interaction. MoXaRt's core is a cascaded architecture that performs coarse, audio-only separation in parallel with visual detection of sources (e.g., faces, instruments). These visual anchors then guide refinement networks to isolate individual sources, separating complex mixes of up to 5 concurrent sources (e.g., 2 voices + 3 instruments) with ~2 second processing latency. We validate MoXaRt through a technical evaluation on a new dataset of 30 one-minute recordings featuring concurrent speech and music, and a 22-participant user study. Empirical results indicate that our system significantly enhances speech intelligibility, yielding a 36.2% (p < 0.01) increase in listening comprehension within adversarial acoustic environments while substantially reducing cognitive load (p < 0.001), thereby paving the way for more perceptive and socially adept XR experiences.
>
---
#### [new 016] MOS-Bias: From Hidden Gender Bias to Gender-Aware Speech Quality Assessment
- **分类: eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决MOS中的性别偏差问题。通过分析发现男性听众评分偏高，提出性别感知模型提升评估公平性。**

- **链接: [https://arxiv.org/pdf/2603.10723](https://arxiv.org/pdf/2603.10723)**

> **作者:** Wenze Ren; Yi-Cheng Lin; Wen-Chin Huang; Erica Cooper; Ryandhimas E. Zezario; Hsin-Min Wang; Hung-yi Lee; Yu Tsao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The Mean Opinion Score (MOS) serves as the standard metric for speech quality assessment, yet biases in human annotations remain underexplored. We conduct the first systematic analysis of gender bias in MOS, revealing that male listeners consistently assign higher scores than female listeners--a gap that is most pronounced in low-quality speech and gradually diminishes as quality improves. This quality-dependent structure proves difficult to eliminate through simple calibration. We further demonstrate that automated MOS models trained on aggregated labels exhibit predictions skewed toward male standards of perception. To address this, we propose a gender-aware model that learns gender-specific scoring patterns through abstracting binary group embeddings, thereby improving overall and gender-specific prediction accuracy. This study establishes that gender bias in MOS constitutes a systematic, learnable pattern demanding attention in equitable speech evaluation.
>
---
#### [new 017] VoxCare: Studying Natural Communication Behaviors of Hospital Caregivers through Wearable Sensing of Egocentric Audio
- **分类: cs.SD**

- **简介: 该论文提出VoxCare系统，通过可穿戴设备捕捉医护人员的自然沟通行为，解决临床环境中沟通活动难以测量的问题。任务属于医疗行为分析，旨在通过音频传感与模型分析，理解沟通模式与工作负荷的关系。**

- **链接: [https://arxiv.org/pdf/2603.10888](https://arxiv.org/pdf/2603.10888)**

> **作者:** Tiantian Feng; Kleanthis Avramidis; Anfeng Xu; Deqi Wang; Brandon M Booth; Shrikanth Narayanan
>
> **摘要:** Healthcare professionals work in complex, high-stakes environments where effective communication is critical for care delivery, team coordination, and individual well-being. However, communication activity in everyday clinical settings remains challenging to measure and largely unexplored in human behavioral research. We present VoxCare, a scalable egocentric wearable audio sensing and computing system that captures natural communication behaviors of hospital professionals in real-world settings without storing raw audio. VoxCare performs real-time, on-device acoustic feature extraction and applies a speech foundation model-guided teacher-student framework to identify foreground speech activity. From these features, VoxCare derives interpretable behavioral measures of communication frequency, duration, and vocal arousal. Our analyses reveal how, when, and how often clinicians communicate across different shifts and working units, and suggest that communication activity reflects underlying workload and stress. By enabling continuous assessment of communication patterns in everyday contexts, this study provides data-driven approaches to understand the behaviors of healthcare providers and ultimately improve healthcare delivery.
>
---
#### [new 018] Training-Free Multi-Step Inference for Target Speaker Extraction
- **分类: cs.SD**

- **简介: 该论文属于目标说话人提取任务，旨在从混合语音中提取特定说话人的语音。提出一种无需训练的多步推理方法，通过迭代优化提升提取效果。**

- **链接: [https://arxiv.org/pdf/2603.10921](https://arxiv.org/pdf/2603.10921)**

> **作者:** Zhenghai You; Ying Shi; Lantian Li; Dong Wang
>
> **摘要:** Target speaker extraction (TSE) aims to recover a target speaker's speech from a mixture using a reference utterance as a cue. Most TSE systems adopt conditional auto-encoder architectures with one-step inference. Inspired by test-time scaling, we propose a training-free multi-step inference method that enables iterative refinement with a frozen pretrained model. At each step, new candidates are generated by interpolating the original mixture and the previous estimate, and the best candidate is selected for further refinement until convergence. Experiments show that, when ground-truth target speech is available, optimizing an intrusive metric (SI-SDRi) yields consistent gains across multiple evaluation metrics. Without ground truth, optimizing non-intrusive metrics (UTMOS or SpkSim) improves the corresponding metric but may hurt others. We therefore introduce joint metric optimization to balance these objectives, enabling controllable extraction preferences for practical deployment.
>
---
#### [new 019] AMB-DSGDN: Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network for Multimodal Emotion Recognition
- **分类: cs.MM; cs.AI; cs.SD**

- **简介: 该论文属于多模态情感识别任务，旨在解决情感依赖建模和模态平衡问题。提出AMB-DSGDN网络，通过动态图结构和自适应机制提升情感表示效果。**

- **链接: [https://arxiv.org/pdf/2603.10043](https://arxiv.org/pdf/2603.10043)**

> **作者:** Yunsheng Wang; Yuntao Shou; Yilong Tan; Wei Ai; Tao Meng; Keqin Li
>
> **备注:** 18 pages
>
> **摘要:** Multimodal dialogue emotion recognition captures emotional cues by fusing text, visual, and audio modalities. However, existing approaches still suffer from notable limitations in modeling emotional dependencies and learning multimodal representations. On the one hand, they are unable to effectively filter out redundant or noisy signals within multimodal features, which hinders the accurate capture of the dynamic evolution of emotional states across and within speakers. On the other hand, during multimodal feature learning, dominant modalities tend to overwhelm the fusion process, thereby suppressing the complementary contributions of non-dominant modalities such as speech and vision, ultimately constraining the overall recognition performance. To address these challenges, we propose an Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network (AMB-DSGDN). Concretely, we first construct modality-specific subgraphs for text, speech, and vision, where each modality contains intra-speaker and inter-speaker graphs to capture both self-continuity and cross-speaker emotional dependencies. On top of these subgraphs, we introduce a differential graph attention mechanism, which computes the discrepancy between two sets of attention maps. By explicitly contrasting these attention distributions, the mechanism cancels out shared noise patterns while retaining modality-specific and context-relevant signals, thereby yielding purer and more discriminative emotional representations. In addition, we design an adaptive modality balancing mechanism, which estimates a dropout probability for each modality according to its relative contribution in emotion modeling.
>
---
#### [new 020] NasoVoce: A Nose-Mounted Low-Audibility Speech Interface for Always-Available Speech Interaction
- **分类: cs.HC; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出NasoVoce，一种鼻部佩戴的低可听语音接口，解决无声或耳语语音交互的难题。通过融合麦克风和振动传感器信号，提升语音识别的准确性和抗干扰能力。属于语音交互任务。**

- **链接: [https://arxiv.org/pdf/2603.10324](https://arxiv.org/pdf/2603.10324)**

> **作者:** Jun Rekimoto; Yu Nishimura; Bojian Yang
>
> **备注:** ACM CHI 2026 paper
>
> **摘要:** Silent and whispered speech offer promise for always-available voice interaction with AI, yet existing methods struggle to balance vocabulary size, wearability, silence, and noise robustness. We present NasoVoce, a nose-bridge-mounted interface that integrates a microphone and a vibration sensor. Positioned at the nasal pads of smart glasses, it unobtrusively captures both acoustic and vibration signals. The nasal bridge, close to the mouth, allows access to bone- and skin-conducted speech and enables reliable capture of low-volume utterances such as whispered speech. While the microphone captures high-quality audio, it is highly sensitive to environmental noise. Conversely, the vibration sensor is robust to noise but yields lower signal quality. By fusing these complementary inputs, NasoVoce generates high-quality speech robust against interference. Evaluation with Whisper Large-v2, PESQ, STOI, and MUSHRA ratings confirms improved recognition and quality. NasoVoce demonstrates the feasibility of a practical interface for always-available, continuous, and discreet AI voice conversations.
>
---
#### [new 021] PRoADS: Provably Secure and Robust Audio Diffusion Steganography with latent optimization and backward Euler Inversion
- **分类: cs.CR; cs.MM; cs.SD**

- **简介: 该论文属于音频隐写任务，旨在解决扩散模型隐写中重建误差导致的高误码率问题。通过引入潜在优化和反向欧拉逆过程，提升隐写安全性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10314](https://arxiv.org/pdf/2603.10314)**

> **作者:** YongPeng Yan; Yanan Li; Qiyang Xiao; Yanzhen Ren
>
> **备注:** This paper has been accepted for presentation at the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** This paper proposes PRoADS, a provably secure and robust audio steganographic framework based on audio diffusion models. As a generative steganography scheme, PRoADS embeds secret messages into the initial noise of diffusion models via orthogonal matrix projection. To address the reconstruction errors in diffusion inversion that cause high bit error rates (BER), we introduce Latent Optimization and Backward Euler Inversion to minimize the latent reconstruction and diffusion inversion errors. Comprehensive experiments demonstrate that our scheme sustains a remarkably low BER of 0.15\% under 64 kbps MP3 compression, significantly outperforming existing methods and exhibiting strong robustness.
>
---
#### [new 022] V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于视频到音乐生成任务，解决视频与音乐时间对齐问题。通过分析视频和音乐的时序结构，无需配对数据即可生成同步音乐。**

- **链接: [https://arxiv.org/pdf/2603.11042](https://arxiv.org/pdf/2603.11042)**

> **作者:** Yan-Bo Lin; Jonah Casebeer; Long Mai; Aniruddha Mahapatra; Gedas Bertasius; Nicholas J. Bryan
>
> **备注:** Project page: this https URL
>
> **摘要:** Generating music that temporally aligns with video events is challenging for existing text-to-music models, which lack fine-grained temporal control. We introduce V2M-Zero, a zero-pair video-to-music generation approach that outputs time-aligned music for video. Our method is motivated by a key observation: temporal synchronization requires matching when and how much change occurs, not what changes. While musical and visual events differ semantically, they exhibit shared temporal structure that can be captured independently within each modality. We capture this structure through event curves computed from intra-modal similarity using pretrained music and video encoders. By measuring temporal change within each modality independently, these curves provide comparable representations across modalities. This enables a simple training strategy: fine-tune a text-to-music model on music-event curves, then substitute video-event curves at inference without cross-modal training or paired data. Across OES-Pub, MovieGenBench-Music, and AIST++, V2M-Zero achieves substantial gains over paired-data baselines: 5-21% higher audio quality, 13-15% better semantic alignment, 21-52% improved temporal synchronization, and 28% higher beat alignment on dance videos. We find similar results via a large crowd-source subjective listening test. Overall, our results validate that temporal alignment through within-modality features, rather than paired cross-modal supervision, is effective for video-to-music generation. Results are available at this https URL
>
---
## 更新

#### [replaced 001] Trade-offs between structural richness and communication efficiency in music network representations
- **分类: physics.soc-ph; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文研究音乐网络表示中结构丰富性与通信效率的权衡问题。通过分析不同特征编码对网络结构和不确定性的影响，探讨其对听众预期的合理性。**

- **链接: [https://arxiv.org/pdf/2509.14053](https://arxiv.org/pdf/2509.14053)**

> **作者:** Lluc Bono Rosselló; Robert Jankowski; Hugues Bersini; Marián Boguñá; M. Ángeles Serrano
>
> **摘要:** Music is a structured and perceptually rich sequence of sounds in time, whose perception is shaped by the interplay of expectation and uncertainty about what comes next. Yet the uncertainty we infer from music depends on how the musical piece is encoded as an event sequence. In this work, we use network representations, in which event types are nodes and observed transitions are directed edges, to compare how different feature encodings shape the transition structure we recover and what this implies for both the descriptive uncertainty expectation under imperfect memory and noise. We systematically analyse eight encodings of piano music, from single-feature vocabularies to richer multi-feature combinations. These representational choices reorganize the state space and fundamentally reshape network topology, shifting how uncertainty is distributed across transitions. To connect these descriptive differences to perception, we adopt a perceptual-constraint model that captures imperfect access to transition statistics. Overall, compressed single-feature representations yield dense transition structures with higher entropy rates, corresponding to higher average uncertainty per step, yet low model error, indicating that the constrained estimate stays close to the corpus transitions. In contrast, richer multi-feature representations preserve finer distinctions but expand the state space, sharpen transition profiles, lower entropy rates, and increase model error. Finally, across representations, uncertainty concentrates in diffusion-central nodes while model error remains low there, suggesting an informational landscape in which predictable flow coexists with localized surprise. Overall, our results show that feature choice shapes not only the networks we reconstruct, but also whether their resulting uncertainty is a plausible proxy for the expectations listeners can realistically learn and use.
>
---
#### [replaced 002] Fish Audio S2 Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文介绍Fish Audio S2，一个开源文本到语音系统，解决多说话人、多轮生成及指令跟随问题。通过多阶段训练和数据管道提升性能，并提供可部署的推理引擎。**

- **链接: [https://arxiv.org/pdf/2603.08823](https://arxiv.org/pdf/2603.08823)**

> **作者:** Shijia Liao; Yuxuan Wang; Songting Liu; Yifan Cheng; Ruoyi Zhang; Tianyu Li; Shidong Li; Yisheng Zheng; Xingwei Liu; Qingzheng Wang; Zhizhuo Zhou; Jiahua Liu; Xin Chen; Dawei Han
>
> **摘要:** We introduce Fish Audio S2, an open-sourced text-to-speech system featuring multi-speaker, multi-turn generation, and, most importantly, instruction-following control via natural-language descriptions. To scale training, we develop a multi-stage training recipe together with a staged data pipeline covering video captioning and speech captioning, voice-quality assessment, and reward modeling. To push the frontier of open-source TTS, we release our model weights, fine-tuning code, and an SGLang-based inference engine. The inference engine is production-ready for streaming, achieving an RTF of 0.195 and a time-to-first-audio below 100 this http URL code and weights are available on GitHub (this https URL) and Hugging Face (this https URL). We highly encourage readers to visit this https URL to try custom voices.
>
---
#### [replaced 003] Evaluation of Audio Compression Codecs
- **分类: cs.SD**

- **简介: 该论文属于音频编码评估任务，旨在解决如何选择合适的音频压缩编码器问题。通过分析压缩效率和听觉质量，比较不同编码器性能。**

- **链接: [https://arxiv.org/pdf/2511.11527](https://arxiv.org/pdf/2511.11527)**

> **作者:** Thien T. Duong; Jan P. Springer
>
> **摘要:** Perceptual quality of audio is the combination of aural accuracy and listener-perceived sound fidelity. It is how humans respond to the accuracy, intelligibility, and fidelity of aural media. Today this fidelity is also heavily influenced by the use of audio compression codecs for storing aural media in digital form. We argue that, when choosing an audio compression codec, users should not only look at compression efficiency but also consider the sonic perceptual quality properties of available audio compression codecs. We evaluate several commonly used audio compression codecs in terms of compression performance as well as their sonic perceptual quality via codec performance measurements, visualizations, and PEAQ scores. We demonstrate how perceptual quality is affected by digital audio compression techniques, providing insights for users in the process of choosing a digital audio compression scheme.
>
---
#### [replaced 004] Robust Audio-Visual Target Speaker Extraction with Emotion-Aware Multiple Enrollment Fusion
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频-视觉目标说话人提取任务，旨在解决多模态信号缺失下的鲁棒性问题。通过融合互补的帧级面部和唇部特征，提升模型在极端缺失情况下的性能。**

- **链接: [https://arxiv.org/pdf/2509.12583](https://arxiv.org/pdf/2509.12583)**

> **作者:** Zhan Jin; Bang Zeng; Peijun Yang; Jiarong Du; Wei Ju; Yao Tian; Juan Liu; Ming Li
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Audio-Visual Target Speaker Extraction (AVTSE) is crucial for cocktail party scenarios. Leveraging multiple cues --such as utterance-level speaker embeddings or steady face images, and frame-level lip motion or facial expression features --can significantly improve performance. However, real-world applications often suffer from intermittent signal loss, especially for frame-level cues. This paper systematically investigates the robustness of multi-enrollment fusion under varying degrees of modality missing. Results show that while full multimodal fusion excels under ideal conditions, its performance degrades sharply when encountering unseen modalities missing during the testing. Crucially, training with a high missing rate dramatically enhances robustness, maintaining stable performance even under severe test-time modality missing. We demonstrate that fusing the complementary one frame of face image with frame-level lip features achieves both strong performance and robustness for the AVTSE task. The model and codes are shared.
>
---
#### [replaced 005] Multi-View Based Audio Visual Target Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文属于音频-视觉目标说话人提取任务，旨在解决非正面视角下语音分离效果差的问题。提出多视角张量融合框架，通过多视角学习提升单视角性能和系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07696](https://arxiv.org/pdf/2603.07696)**

> **作者:** Peijun Yang; Zhan Jin; Juan Liu; Ming Li
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Audio-Visual Target Speaker Extraction (AVTSE) aims to separate a target speaker's voice from a mixed audio signal using the corresponding visual cues. While most existing AVTSE methods rely exclusively on frontal-view videos, this limitation restricts their robustness in real-world scenarios where non-frontal views are prevalent. Such visual perspectives often contain complementary articulatory information that could enhance speech extraction. In this work, we propose Multi-View Tensor Fusion (MVTF), a novel framework that transforms multi-view learning into single-view performance gains. During the training stage, we leverage synchronized multi-perspective lip videos to learn cross-view correlations through MVTF, where pairwise outer products explicitly model multiplicative interactions between different views of input lip embeddings. At the inference stage, the system supports both single-view and multi-view inputs. Experimental results show that in the single-view inputs, our framework leverages multi-view knowledge to achieve significant performance gains, while in the multi-view mode, it further improves overall performance and enhances the robustness. Our demo, code and data are available at this https URL
>
---
#### [replaced 006] Computational modeling of early language learning from acoustic speech and audiovisual input without linguistic priors
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语言习得研究任务，旨在探讨如何通过计算模型理解婴儿从语音和视听输入中学习语言的过程，解决无语言先验条件下的语言习得问题。工作包括回顾自监督和视觉引导的模型进展。**

- **链接: [https://arxiv.org/pdf/2603.08359](https://arxiv.org/pdf/2603.08359)**

> **作者:** Okko Räsänen
>
> **摘要:** Learning to understand speech appears almost effortless for typically developing infants, yet from an information-processing perspective, acquiring a language from acoustic speech is an enormous challenge. This chapter reviews recent developments in using computational models to understand early language acquisition from speech and audiovisual input. The focus is on self-supervised and visually grounded models of perceptual learning. We show how these models are becoming increasingly powerful in learning various aspects of speech without strong linguistic priors, and how many features of early language development can be explained through a shared set of learning principles-principles broadly compatible with multiple theories of language acquisition and human cognition. We also discuss how modern learning simulations are gradually becoming more realistic, both in terms of input data and in linking model behavior to empirical findings on infant language development.
>
---
#### [replaced 007] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频-视觉语音分离任务，旨在提升分离效率。针对现有方法参数多、计算成本高的问题，提出Dolphin模型，结合轻量编码器与多尺度注意力机制，实现高效准确的语音分离。**

- **链接: [https://arxiv.org/pdf/2509.23610](https://arxiv.org/pdf/2509.23610)**

> **作者:** Kai Li; Kejun Gao; Xiaolin Hu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Audio-visual speech separation (AVSS) methods leverage visual cues to extract target speech and have demonstrated strong separation quality in noisy acoustic environments. However, these methods usually involve a large number of parameters and require high computational cost, which is unacceptable in many applications where speech separation serves as only a preprocessing step for further speech processing. To address this issue, we propose an efficient AVSS method, named Dolphin. For visual feature extraction, we develop DP-LipCoder, a dual-path lightweight video encoder that transforms lip-motion into discrete audio-aligned semantic tokens. For audio separation, we construct a lightweight encoder-decoder separator, in which each layer incorporates a global-local attention (GLA) block to efficiently capture multi-scale dependencies. Experiments on three benchmark datasets showed that Dolphin not only surpassed the current state-of-the-art (SOTA) model in separation quality but also achieved remarkable improvements in efficiency: over 50% fewer parameters, more than 2.4x reduction in MACs, and over 6x faster GPU inference speed. These results indicate that Dolphin offers a practical and deployable solution for high-performance AVSS in real-world scenarios. Our code and demo page are publicly available at this http URL.
>
---
#### [replaced 008] Are Deep Speech Denoising Models Robust to Adversarial Noise?
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音降噪任务，研究深度降噪模型对对抗噪声的鲁棒性。工作表明，添加隐蔽对抗噪声可使模型输出不可理解，揭示安全风险。**

- **链接: [https://arxiv.org/pdf/2503.11627](https://arxiv.org/pdf/2503.11627)**

> **作者:** Will Schwarzer; Neel Chaudhari; Philip S. Thomas; Andrea Fanelli; Xiaoyu Liu
>
> **备注:** 22 pages, 14 figures. Related conference version accepted to ICLR 2026: see this https URL
>
> **摘要:** Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. However, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of psychoacoustically hidden adversarial noise, even in low-background-noise and simulated over-the-air settings. For three of the models, a small transcription study with audio and multimedia experts confirms unintelligibility of the attacked audio; simultaneously, an ABX study shows that the adversarial noise is generally imperceptible, with some variance between participants and samples. While we also establish several negative results around targeted attacks and model transfer, our results nevertheless highlight the need for practical countermeasures before open-source DNS systems can be used in safety-critical applications.
>
---
#### [replaced 009] Modeling strategies for speech enhancement in the latent space of a neural audio codec
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，研究神经音频编解码器的潜在空间表示在语音增强中的应用，比较连续与离散表示的效果，提出不同模型结构并验证其性能。**

- **链接: [https://arxiv.org/pdf/2510.26299](https://arxiv.org/pdf/2510.26299)**

> **作者:** Sofiene Kammoun; Xavier Alameda-Pineda; Simon Leglaive
>
> **摘要:** Neural audio codecs (NACs) provide compact latent speech representations in the form of sequences of continuous vectors or discrete tokens. In this work, we investigate how these two types of speech representations compare when used as training targets for supervised speech enhancement. We consider both autoregressive and non-autoregressive speech enhancement models based on the Conformer architecture, as well as a simple baseline where the NAC encoder is simply fine-tuned for speech enhancement. Our experiments reveal three key findings: predicting continuous latent representations consistently outperforms discrete token prediction; autoregressive models achieve higher quality but at the expense of intelligibility and efficiency, making non-autoregressive models more attractive in practice; and adding encoder fine-tuning yields the strongest enhancement metrics overall, though at the cost of degraded codec reconstruction. The code and audio samples are available online.
>
---
#### [replaced 010] HyWA: Hypernetwork Weight Adapting Personalized Voice Activity Detection
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于语音活动检测任务，解决个性化语音活动检测问题。提出HyWA方法，通过超网络生成特定权重，提升检测精度并简化部署。**

- **链接: [https://arxiv.org/pdf/2510.12947](https://arxiv.org/pdf/2510.12947)**

> **作者:** Mahsa Ghazvini Nejad; Hamed Jafarzadeh Asl; Amin Edraki; Mohammadreza Sadeghi; Masoud Asgharian; Yuanhao Yu; Vahid Partovi Nia
>
> **备注:** Mahsa Ghazvini Nejad and Hamed Jafarzadeh Asl contributed equally to this work. Submitted to Interspeech 2026
>
> **摘要:** Personalized Voice Activity Detection (PVAD) systems activate only in response to a specific target speaker. Speaker-conditioning methods are employed to inject information about the target speaker into a VAD pipeline, to achieve personalization. Existing speaker-conditioning methods typically modify the inputs or activations of a VAD model. We propose an alternative perspective to speaker conditioning. Our approach, HyWA, employs a hypernetwork to generate personalized weights for a few selected layers of a standard VAD model. We evaluate HyWA against multiple baseline speaker-conditioning techniques using a fixed backbone VAD. Our comparison shows consistent improvements in PVAD performance. This new approach improves the current speaker-conditioning techniques in two ways: i) increases the mean average precision, ii) facilitates deployment by reusing the same VAD architecture.
>
---
