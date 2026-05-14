# 音频 cs.SD;  eess.AS

- **最新发布 6 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Text2Score: Generating Sheet Music From Textual Prompts
- **分类: cs.SD**

- **简介: 该论文提出Text2Score，解决从文本生成乐谱的任务。针对数据稀缺和自动化标注不可靠的问题，设计两阶段框架，生成符合结构约束的ABC乐谱，并通过专家评估验证效果。**

- **链接: [https://arxiv.org/pdf/2605.13431](https://arxiv.org/pdf/2605.13431)**

> **作者:** Keshav Bhandari; Sungkyun Chang; Abhinaba Roy; Francesca Ronchini; Emmanouil Benetos; Dorien Herremans; Simon Colton
>
> **备注:** 8 pages including references, 1 figure
>
> **摘要:** Developing text-driven symbolic music generation models remains challenging due to the scarcity of aligned text-music datasets and the unreliability of automated captioning pipelines. While most efforts have focused on MIDI, sheet music representations are largely underexplored in text-driven generation. We present Text2Score, a two-stage framework comprising a planning stage and an execution stage for generating sheet music from natural language prompts. By deriving supervision signals directly from symbolic XML data, we propose an alternative training paradigm that bypasses noisy or scarce text-music pairs. In the planning stage, an LLM orchestrator translates a natural language prompt into a structured measure-wise plan defining musical attributes such as instruments, key, time signatures, harmony, etc. This plan is then consumed by a generative model in the execution stage to produce interleaved ABC notation conditioned on the plan's structural constraints. To assess output quality, we introduce an evaluation framework covering playability, readability, instrument utilization, structural complexity, and prompt adherence, validated by expert musicians. Text2Score consistently outperforms both a pure LLM-based agentic framework and three end-to-end baselines across objective and subjective dimensions. We open-source the dataset, code, evaluation set and LLM prompts used in this work; a demo is available on our project page (this https URL).
>
---
#### [new 002] NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出NAACA，解决音频中显著事件检测问题。通过神经启发的振荡工作记忆，提升注意力效率，增强对罕见事件的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.13651](https://arxiv.org/pdf/2605.13651)**

> **作者:** Zhongju Yuan; Geraint Wiggins; Dick Botteldooren
>
> **备注:** Accepted as a regular paper by ICML 2026
>
> **摘要:** Audio provides critical situational cues, yet current Audio Language Models (ALMs) face an attention bottleneck in long-form recordings where dominant background patterns can dilute rare, salient events. We introduce NAACA, a training-free NeuroAuditory Attentive Cognitive Architecture that reframes attention allocation as an auditory salience filtering problem. At its core is OWM, a neuro-inspired Oscillatory Working Memory that maintains stable attractor-like states and triggers higher-cognition ALM processing only when adaptive energy fluctuations signal perceptual salience, triggering higher-level reasoning. On XD-Violence, NAACA improves AudioQwen's average precision (AP) from 53.50% to 70.60% while reducing unnecessary ALM invocations. Furthermore, qualitative case studies on the Urban Soundscapes of the World (USoW) dataset show that OWM captures novel events and subcategory shifts while remaining robust to transient pauses and ambient urban noise.
>
---
#### [new 003] Seconds-Aligned PCA-DAC Latent Diffusion for Symbolic-to-Audio Drum Rendering
- **分类: cs.SD**

- **简介: 该论文属于符号到音频鼓声生成任务，解决如何在保持时间与动态信息的同时合成真实波形的问题。提出Sec2Drum-DAC模型，利用主成分分析和扩散模型进行高效渲染。**

- **链接: [https://arxiv.org/pdf/2605.13404](https://arxiv.org/pdf/2605.13404)**

> **作者:** Konstantinos Soiledis; Maximos Kaliakatsos Papakostas; Dimos Makris; Konstantinos Tsamis
>
> **摘要:** Symbolic-control drum generation requires preserving explicit event timing and dynamics while synthesizing acoustically plausible waveforms. We present Sec2Drum-DAC, a conditional latent-diffusion model for symbolic-to-audio drum rendering. The model conditions on event features sampled in physical time at codec-frame locations and predicts standardized principal-component coordinates of frozen DAC summed-codebook embeddings rather than waveform samples. In the evaluated DAC configuration, 72 principal components capture the observed training-frame summed-latent subspace under the stated SVD threshold, yielding a compact continuous denoising target with a deterministic reconstruction path to the 1024-dimensional DAC latent space before waveform decoding. Across 1,733 held-out four-beat windows, PCA diffusion improves paired spectral and transient metrics over deterministic PCA regression and a symbolic rendering baseline, while direct regression remains stronger on phase-sensitive waveform L1. Auxiliary RVQ cross-entropy improves short-step diffusion on mel error, onset-flux cosine, and waveform L1, with the most favorable trade-offs occurring at 6-25 denoising steps depending on the metric.
>
---
#### [new 004] EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出EVA-Bench，用于评估语音助手的性能。解决语音助手在真实对话生成和质量测量方面的评估难题。工作包括构建模拟对话系统和引入两个综合评价指标。**

- **链接: [https://arxiv.org/pdf/2605.13841](https://arxiv.org/pdf/2605.13841)**

> **作者:** Tara Bogavelli; Gabrielle Gauthier Melançon; Katrina Stankiewicz; Oluwanifemi Bamgbose; Fanny Riols; Hoang H. Nguyen; Raghav Mehndiratta; Lindsay Devon Brin; Joseph Marinier; Hari Subramani; Anil Madamala; Sridhar Krishna Nemala; Srinivas Sunkara
>
> **备注:** Work in progress
>
> **摘要:** Voice agents, artificial intelligence systems that conduct spoken conversations to complete tasks, are increasingly deployed across enterprise applications. However, no existing benchmark jointly addresses two core evaluation challenges: generating realistic simulated conversations, and measuring quality across the full scope of voice-specific failure modes. We present EVA-Bench, an end-to-end evaluation framework that addresses both. On the simulation side, EVA-Bench orchestrates bot-to-bot audio conversations over dynamic multi-turn dialogues, with automatic simulation validation that detects user simulator error and appropriately regenerates conversations before scoring. On the measurement side, EVA-Bench introduces two composite metrics: EVA-A (Accuracy), capturing task completion, faithfulness, and audio-level speech fidelity; and EVA-X (Experience), capturing conversation progression, spoken conciseness, and turn-taking timing. Both metrics apply to different agent architectures, enabling direct cross-architecture comparison. EVA-Bench includes 213 scenarios across three enterprise domains, a controlled perturbation suite for accent and noise robustness, and pass@1, pass@k, pass^k measurements that distinguish peak from reliable capability. Across 12 systems spanning all three architectures, we find: (1) no system simultaneously exceeds 0.5 on both EVA-A pass@1 and EVA-X pass@1; (2) peak and reliable performance diverge substantially (median pass@k - pass^k gap of 0.44 on EVA-A); and (3) accent and noise perturbations expose substantial robustness gaps, with effects varying across architectures, systems, and metrics (mean up to 0.314). We release the full framework, evaluation suite, and benchmark data under an open-source license.
>
---
#### [new 005] Bypassing Direct Reconstruction: Speech Detection from MEG via Large-Scale Audio Retrieval
- **分类: cs.SD**

- **简介: 该论文属于语音检测任务，旨在从MEG信号中识别语音。通过检索外部音频库并直接生成语音/静音序列，解决了直接重建语音的难题。**

- **链接: [https://arxiv.org/pdf/2605.13099](https://arxiv.org/pdf/2605.13099)**

> **作者:** Boda Xiao; Bo Wang; Heping Cheng
>
> **备注:** ranked first at LibriBrain Competition 2025 this https URL
>
> **摘要:** Decoding speech from non-invasive brain signals is challenging. For the LibriBrain 2025 Speech Detection task, we propose a novel two-step framework that bypasses direct reconstruction. First, a contrastive learning model retrieves the matching speech segment for the given test MEG from a large-scale audio library (LibriVox). Second, a speech detection model generates the binary silence/speech sequence directly from this retrieved audio. With this approach, our team Sherlock Holmes achieved first place in the extended track (F1-score: 0.962), demonstrating that leveraging external audio databases is a highly effective strategy.
>
---
#### [new 006] BioSEN: A Bio-acoustic Signal Enhancement Network for Animal Vocalizations
- **分类: cs.SD; cs.LG; q-bio.NC**

- **简介: 该论文属于生物声学信号增强任务，解决动物叫声在噪声中难以识别的问题。提出BioSEN模型，通过多尺度注意力、谐波增强和能量自适应门控提升信号质量。**

- **链接: [https://arxiv.org/pdf/2605.12534](https://arxiv.org/pdf/2605.12534)**

> **作者:** Tianyu Song; Ton Viet Ta; Ngamta Thamwattana; Hisako Nomura; Linh Thi Hoai Nguyen
>
> **摘要:** Most work in audio enhancement targets human speech, while bioacoustics is less studied due to noisy recordings and the distinct traits of animal sounds. To fill this gap, we adapt speech enhancement methods and build BioSEN, a model made for bioacoustic signals. BioSEN has three modules: a multi-scale dual-axis attention unit for time-frequency feature extraction, a bio-harmonic multi-scale enhancement unit for capturing harmonic structures, and an energy-adaptive gating connection unit that uses frequency weights to keep vocalizations from being removed as noise. Tests on three bioacoustic datasets show that BioSEN matches or exceeds state-of-the-art speech enhancement models while using far less computation. These results show BioSEN's strength for bioacoustic audio enhancement and its promise for biodiversity monitoring and conservation.
>
---
## 更新

#### [replaced 001] Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究自动语音识别（ASR）和语音翻译（ST）任务，旨在评估最小贝叶斯风险解码（MBR）在这些任务中的效果。实验表明，MBR在多数情况下优于传统束搜索方法。**

- **链接: [https://arxiv.org/pdf/2510.19471](https://arxiv.org/pdf/2510.19471)**

> **作者:** Yuu Jinnai
>
> **摘要:** Recent work has shown that sample-based Minimum Bayes Risk (MBR) decoding outperforms beam search in text-to-text generation tasks, such as machine translation, text summarization, and image captioning. On the other hand, beam search is the current practice for speech-to-text tasks such as automatic speech recognition (ASR) and Speech Translation (ST). Given that MBR decoding is effective in text-to-text generation tasks, it is reasonable to expect it to also be effective for speech-to-text tasks. In this paper, we evaluate MBR decoding for ASR and ST tasks on English and Japanese using Whisper and its derivative models. We observe that the accuracy of MBR decoding outperforms that of beam search in most of the experimental settings we have evaluated. The results show that MBR decoding is a promising method for offline ASR and ST tasks that require high accuracy. The code is available at this https URL
>
---
#### [replaced 002] Unifying Diarization, Separation, and ASR with Multi-Speaker Encoder
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出一种统一的多说话人编码器（UME），用于联合解决语音分离、说话人辨识和多说话人ASR任务，提升重叠语音处理效果。**

- **链接: [https://arxiv.org/pdf/2508.20474](https://arxiv.org/pdf/2508.20474)**

> **作者:** Muhammad Shakeel; Yui Sudo; Yifan Peng; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** This paper presents a unified multi-speaker encoder (UME), a novel architecture that jointly learns representations for speaker diarization (SD), speech separation (SS), and multi-speaker automatic speech recognition (ASR) tasks using a shared speech foundational encoder. We leverage the hidden representations from multiple layers of UME as a residual weighted-sum encoding (RWSE) to effectively use information from different semantic levels, contributing to bottom-up alignment between tasks. This joint training approach captures the inherent interdependencies among the tasks, enhancing overall performance on overlapping speech data. Our evaluations demonstrate that UME substantially improves over the single-task baselines dedicated to SD, SS, and multi-speaker ASR on LibriMix evaluation sets. Notably, for SD, UME outperforms the previous studies, achieving diarization error rates of 1.37% and 2.29% on Libri2Mix and Libri3Mix evaluation sets, respectively.
>
---
#### [replaced 003] Aliasing-Free Neural Audio Synthesis
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于音频合成任务，旨在解决神经音频合成中的混叠问题。通过引入可微抗混叠技术，提出Pupu-Vocoder和Pupu-Codec模型，提升音乐和人声合成质量。**

- **链接: [https://arxiv.org/pdf/2512.20211](https://arxiv.org/pdf/2512.20211)**

> **作者:** Yicheng Gu; Junan Zhang; Chaoren Wang; Jerry Li; Zhizheng Wu; Lauri Juvela
>
> **备注:** Accepted by TASLP
>
> **摘要:** In neural audio synthesis, neural vocoders and codecs are models that reconstruct waveforms from acoustic and latent representations, which are essential to the resulting audio quality. While current models are capable of generating perceptually natural speech, they still struggle with high-fidelity music and singing voice synthesis, as severe aliasing artifacts are introduced by non-linear activation functions and upsampling layers in existing architectures. Although various anti-aliasing techniques have been proposed in digital signal processing, their integration into neural vocoders and codecs remains under-explored. This paper incorporates differentiable anti-aliasing techniques into the activation and upsampling modules to bridge this gap, and thus presents Pupu-Vocoder and Pupu-Codec. We build a test signal benchmark to evaluate the anti-aliased modules, and validate our proposed models on speech, singing voice, music, and audio. Experimental results show that Pupu-Vocoder and Pupu-Codec outperform existing systems on singing voice, music, and audio, while achieving comparable performance on speech. Demos, codes, and checkpoints are available at this http URL.
>
---
#### [replaced 004] Repurposing Image Diffusion Models for Training-Free Music Style Transfer on Mel-spectrograms
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐风格迁移任务，解决零样本方法难以捕捉音频细节的问题。通过复用预训练图像扩散模型，在梅尔频谱图上实现无训练风格迁移。**

- **链接: [https://arxiv.org/pdf/2411.15913](https://arxiv.org/pdf/2411.15913)**

> **作者:** Heehwan Wang; Joonwoo Kwon; Sooyoung Kim; Jungwoo Seo; Shinjae Yoo; Yuewei Lin; Jiook Cha
>
> **备注:** Accepted by ICIP 2026
>
> **摘要:** Music style transfer blends source structure with reference style to enable personalized music creation. However, existing zero-shot methods often struggle to capture fine-grained audio nuances, relying on coarse text descriptions or requiring expensive task-specific training. We propose Stylus, a training-free framework that repurposes pretrained image diffusion models for music style transfer in the Mel-spectrogram domain. By treating audio as structured time-frequency images, Stylus manipulates self-attention by injecting style keys and values while preserving source structural queries. To ensure high fidelity, we introduce a phase-preserving reconstruction strategy to mitigate spectrogram inversion artifacts, alongside a classifier-free-guidance-inspired control for adjustable stylization. Extensive evaluations including 2,925 human ratings demonstrate that Stylus outperforms state-of-the-art baselines, achieving 34.1% higher content preservation and 25.7% better perceptual quality. Our work validates that generic image priors can be effectively leveraged for the training-free transformation of structured Mel-spectrograms. Code and materials are available at this https URL.
>
---
#### [replaced 005] DeePen: Penetration Testing for Audio Deepfake Detection
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在评估检测模型的鲁棒性。通过提出DeePen方法，使用信号处理攻击测试模型弱点，发现现有系统易被简单操作欺骗。**

- **链接: [https://arxiv.org/pdf/2502.20427](https://arxiv.org/pdf/2502.20427)**

> **作者:** Nicolas Müller; Piotr Kawa; Adriana Stan; Thien-Phuc Doan; Souhwan Jung; Wei Herng Choong; Philip Sperl; Konstantin Böttinger
>
> **摘要:** Deepfakes - manipulated or forged audio and video media - pose significant security risks to individuals, organizations, and society at large. To address these challenges, machine learning-based classifiers are commonly employed to detect deepfake content. In this paper, we assess the robustness of such classifiers through a systematic penetration testing methodology, which we introduce as DeePen. Our approach operates without prior knowledge of or access to the target deepfake detection models. Instead, it leverages a set of carefully selected signal processing modifications - referred to as attacks - to evaluate model vulnerabilities. Using DeePen, we analyze both real-world production systems and publicly available academic model checkpoints, demonstrating that all tested systems exhibit weaknesses and can be reliably deceived by simple manipulations such as time-stretching or echo addition. Furthermore, our findings reveal that while some attacks can be mitigated by retraining detection systems with knowledge of the specific attack, others remain persistently effective.
>
---
#### [replaced 006] How Much Does Machine Identity Matter in Anomalous Sound Detection at Test Time?
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于异常声音检测任务，旨在解决测试时机器身份未知带来的性能下降问题。通过修改评估协议，合并多机器数据进行联合评估，揭示方法在真实场景下的鲁棒性差异。**

- **链接: [https://arxiv.org/pdf/2602.16253](https://arxiv.org/pdf/2602.16253)**

> **作者:** Kevin Wilkinghoff; Keisuke Imoto; Zheng-Hua Tan
>
> **摘要:** Anomalous sound detection (ASD) benchmarks typically assume that the identity of the monitored machine is known at test time and that recordings are evaluated in a machine-wise manner. However, in realistic monitoring scenarios with multiple known machines operating concurrently, test recordings may not be reliably attributable to a specific machine, and requiring machine identity imposes deployment constraints such as dedicated sensors per machine. To reveal performance degradations and method-specific differences in robustness that are hidden under standard machine-wise evaluation, we consider a minimal modification of the ASD evaluation protocol in which test recordings from multiple machines are merged and evaluated jointly without access to machine identity at inference time. Training data and evaluation metrics remain unchanged, and machine identity labels are used only for post hoc evaluation. Experiments with representative ASD methods show that relaxing this assumption reveals performance degradations and method-specific differences in robustness that are hidden under standard machine-wise evaluation, and that these degradations are strongly related to implicit machine identification accuracy.
>
---
#### [replaced 007] LMU-Based Sequential Learning and Posterior Ensemble Fusion for Cross-Domain Infant Cry Classification
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于跨域婴儿啼哭分类任务，解决信号短、标注少和领域差异大的问题。提出融合多特征的CNN-LMU模型与后验集成方法，提升分类性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.02245](https://arxiv.org/pdf/2603.02245)**

> **作者:** Niloofar Jazaeri; Hilmi R. Dajani; Marco Janeczek; Martin Bouchard
>
> **备注:** 7 pages, to appear in Proc. Int. Conf. IEEE Engineering in Medicine and Biology Society (EMBC 2026), Toronto, Canada, July 26-30 2026
>
> **摘要:** Decoding infant cry causes remains challenging for healthcare monitoring due to short nonstationary signals, limited annotations, and strong domain shifts across infants and datasets. We propose a compact acoustic framework that fuses mel-frequency cepstral coefficients (MFCCs), short-time Fourier transform (STFT) features, and fundamental-frequency (F0) contours within a multi-branch convolutional neural network (CNN) encoder, and models temporal dynamics using an enhanced Legendre Memory Unit (LMU). Compared to LSTMs, the LMU backbone provides stable sequence modeling with substantially fewer recurrent parameters, supporting efficient deployment. To improve cross-dataset generalization, we introduce calibrated posterior ensemble fusion with entropy-gated weighting to preserve domain-specific expertise while mitigating dataset bias. Experiments on Baby2020 and Baby Crying demonstrate improved macro-F1 under cross-domain evaluation, along with leakage aware splits and real-time feasibility for on-device monitoring.
>
---
#### [replaced 008] TW-Sound580K: A Regional Audio-Text Dataset with Verification-Guided Curation for Localized Audio-Language Modeling
- **分类: cs.SD**

- **简介: 该论文属于音频-文本建模任务，旨在解决方言语音建模数据不足的问题。通过构建TW-Sound580K数据集并采用动态仲裁策略，提升模型在本地化语音上的性能。**

- **链接: [https://arxiv.org/pdf/2603.05094](https://arxiv.org/pdf/2603.05094)**

> **作者:** Hao-Hui Xie; Ho-Lam Chung; Yi-Cheng Lin; Ke-Han Lu; Wenze Ren; Xie Chen; Hung-yi Lee
>
> **摘要:** Large Audio-Language Models (LALMs) typically struggle with localized dialectal prosody due to the scarcity of specialized corpora. We present TW-Sound580K, a Taiwanese audio-text instruction dataset developed through a Verify-Generate-Critique (VGC) protocol. This pipeline leverages Dual-ASR validation to filter 522K raw clips, subsequently expanding them into 580,000 high-fidelity instruction pairs using a teacher model. The dataset's utility is demonstrated through Tai-LALM, which fine-tunes a DeSTA 2.5-Audio-initialized backbone and incorporates a dynamic Dual-ASR Arbitration strategy to optimize transcription selection during inference. On the TAU Benchmark, Tai-LALM reaches 49.1% accuracy, marking a 6.5% absolute improvement over the zero-shot baseline (42.6% with ASR text conditioning). This confirms that integrating regional corpora with rigorous curation and dynamic arbitration significantly enhances LALM performance on localized speech.
>
---
#### [replaced 009] Adapting a Text-to-Audio Model for Room Impulse Response Generation
- **分类: eess.AS**

- **简介: 该论文属于语音信号处理任务，旨在解决RIR数据稀缺问题。通过适配文本到音频模型生成RIR，并利用视觉语言模型构建数据集。**

- **链接: [https://arxiv.org/pdf/2603.09708](https://arxiv.org/pdf/2603.09708)**

> **作者:** Kirak Kim; Sungyoung Kim
>
> **备注:** 4 pages, 1 figure, submitted to IWAENC 2026
>
> **摘要:** Room Impulse Responses (RIRs) enable realistic acoustic simulation, with applications ranging from multimedia production to speech data augmentation. However, acquiring high-quality real-world RIRs is labor-intensive, and data scarcity remains a challenge for data-driven RIR generation approaches. In this paper, we propose a novel approach to RIR generation by adapting a pre-trained text-to-audio model, demonstrating for the first time that large-scale generative audio priors can be effectively leveraged for the task. To address the lack of text-RIR paired data, we utilize a labeling pipeline leveraging vision-language models to extract acoustic descriptions from existing image-RIR datasets. We introduce an in-context learning strategy to accommodate free-form user prompts during inference. Evaluations including subjective listening test demonstrate that our model generates plausible RIRs. Audio examples are available on our demo website.
>
---
#### [replaced 010] TiCo: Time-Controllable Spoken Dialogue Model
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出TiCo模型，解决语音对话系统中响应时长控制问题。通过引入时间标记，实现对响应时长的精确控制，提升交互质量。**

- **链接: [https://arxiv.org/pdf/2603.22267](https://arxiv.org/pdf/2603.22267)**

> **作者:** Kai-Wei Chang; Wei-Chih Chen; En-Pei Hu; Hung-yi Lee; James Glass
>
> **摘要:** We introduce TiCo, a time-controllable spoken dialogue model (SDM) that follows time-constrained instructions (e.g., "Please generate a response lasting about 15 seconds") and generates spoken responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions. To systematically evaluate this, we introduce TiCo-Bench, the first benchmark for time-controllable instruction following in SDMs, on which existing open-source and commercial models frequently fail to satisfy explicit time constraints. TiCo addresses this limitation by enabling an SDM to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is post-trained efficiently without question-answer paired data, relying on self-generation and reinforcement learning with verifiable reward. Experimental results show that TiCo reduces duration error by 2.7x over its backbone and 1.6x over the strongest baseline, while preserving response quality.
>
---
#### [replaced 011] CALM: Joint Contextual Acoustic-Linguistic Modeling for Personalization of Multi-Speaker ASR
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出CALM框架，解决多说话人ASR中的个性化问题，通过联合声学与语言建模提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.22792](https://arxiv.org/pdf/2601.22792)**

> **作者:** Muhammad Shakeel; Yosuke Fukumoto; Chikara Maeda; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** We present CALM, a joint Contextual Acoustic-Linguistic Modeling framework for multi-speaker automatic speech recognition (ASR). In personalized AI scenarios, the joint availability of acoustic and linguistic cues naturally motivates the integration of target-speaker conditioning with contextual biasing in overlapping conversations. CALM implements this integration in an end-to-end framework through speaker embedding-driven target-speaker extraction and dynamic vocabulary-based contextual biasing. We evaluate CALM on simulated English (LibriSpeechMix) and Japanese (Corpus of Spontaneous Japanese mixtures, CSJMix). On two-speaker mixtures, CALM reduces biased word error rate (B-WER) from 12.7 to 4.7 on LibriSpeech2Mix and biased character error rate (B-CER) from 16.6 to 8.4 on CSJMix2 (eval3), demonstrating the effectiveness of joint acoustic-linguistic modeling across languages. We additionally report results on the AMI corpus (IHM-mix condition) to validate performance on standardized speech mixtures.
>
---
