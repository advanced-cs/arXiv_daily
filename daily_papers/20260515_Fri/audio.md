# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] SpeakerLLM: A Speaker-Specialized Audio-LLM for Speaker Understanding and Verification Reasoning
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文提出SpeakerLLM，解决语音识别中的说话人理解和验证问题。通过整合语音特征与语言推理，提升说话人身份确认的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.15044](https://arxiv.org/pdf/2605.15044)**

> **作者:** KiHyun Nam; Jungwoo Heo; Siu Bae; Ha-Jin Yu; Joon Son Chung
>
> **摘要:** As audio-first agents become increasingly common in physical AI, conversational robots, and screenless wearables, audio large language models (audio-LLMs) must integrate speaker-specific understanding to support user authorization, personalization, and context-aware interaction. This requires modeling who is speaking, how the voice sounds, and how recording conditions affect speaker cues. Conventional speaker verification systems provide strong scalar scores but little linguistic evidence, while current audio-LLMs and speaker-aware language models have limited ability to organize speaker information beyond binary labels or descriptive profiles. We present SpeakerLLM, a speaker-specialized audio-LLM framework that unifies single-utterance speaker profiling, recording-condition understanding, utterance-pair speaker comparison, and evidence-organized verification reasoning within a natural-language interface. We construct verification-reasoning targets and a decision-composition policy that separate profile-level evidence from the final same-or-different decision and organize recording condition, profile evidence, and the decision into a structured trace. At its core, SpeakerLLM uses a hierarchical speaker tokenizer designed to capture multiple granularities of speaker evidence. Utterance-level speaker embeddings summarize identity and profile-level cues, whereas frame-level speaker features preserve fine-grained acoustic descriptors. Experiments show that SpeakerLLM-Base improves speaker-profile and recording-condition understanding over general audio-LLMs, while SpeakerLLM-VR preserves strong generated-verdict accuracy and produces decision traces grounded in the supervised verification reasoning schema. We will release the metadata-enriched supervision dataset and target-construction code for reproducibility.
>
---
#### [new 002] Masked Autoencoders with Limited Data: Does It Work? A Fine-Grained Bioacoustics Case Study
- **分类: cs.SD; cs.CV; cs.LG**

- **简介: 该论文属于生物声学分类任务，旨在解决有限标注数据下的模型训练问题。通过研究掩码自编码器在iNatSounds数据集上的预训练效果，分析数据规模、领域特异性等因素的影响。**

- **链接: [https://arxiv.org/pdf/2605.14031](https://arxiv.org/pdf/2605.14031)**

> **作者:** Wuao Liu; Mustafa Chasmai; Subhransu Maji; Grant Van Horn
>
> **备注:** Workshop on Fine-Grained Visual Categorization (FGVC) at CVPR 2026. 8 pages, 6 figures
>
> **摘要:** Bioacoustic recognition requires fine-grained acoustic understanding to distinguish similar-sounding species. However, many large-scale data repositories such as iNaturalist are weakly annotated, often with only a single positive species label per recording, making supervised learning particularly challenging. Inspired by advances in computer vision, recent approaches have shifted toward self-supervised learning to capture the underlying structure of audio without relying on exhaustive annotations. In particular, masked autoencoders (MAE) have shown strong transferability on massive audio corpora, yet their effectiveness in more modest bioacoustic settings remains underexplored. In this work, we conduct a systematic study of MAE pretraining for species classification on iNatSounds, analyzing the impacts of pretraining data scale, domain specificity, data curation, and transfer strategies. Consistent with prior work, we find that models pretrained on diverse general audio data achieve the best transfer performance on iNatSounds. Contrary to observations from large-scale audio benchmarks, we find that (1) additional masked reconstruction pretraining on domain-specific data provides limited benefits and may even degrade performance relative to off-the-shelf models, and (2) selective data filtering offers a negligible advantage when the overall data scale is limited. Our results indicate that, in moderate-sized fine-grained bioacoustic settings, pretraining scale dominates objective design. These findings further clarify when MAE-based pretraining is effective and provide practical guidance for model selection under limited supervision.
>
---
#### [new 003] FSD50K-Solo: Automated Curation of Single-Source Sound Events
- **分类: eess.AS**

- **简介: 该论文属于音频数据集构建任务，旨在解决多源声音样本影响数据质量的问题。通过生成模型和分类器自动筛选单源声音，构建高质量数据集FSD50K-Solo。**

- **链接: [https://arxiv.org/pdf/2605.13931](https://arxiv.org/pdf/2605.13931)**

> **作者:** Ningyuan Yang; Sile Yin; Li-Chia Yang; Bryce Irvin; Xiao Quan; Marko Stamenovic; Shuo Zhang
>
> **备注:** Accepted to EUSIPCO 2026. 5 pages, 3 figures
>
> **摘要:** High-quality training datasets are essential for the performance of neural networks. However, the audio domain still lacks a large-scale, strongly-labeled, and single-source sound event dataset. The FSD50K dataset, despite being relatively large and open, contains a considerable fraction of multi-source samples where background interference or overlapping events could limit the usefulness of the data. To address this challenge, we introduce a data curation framework designed for large-scale open audio corpora. Our approach leverages a generative diffusion model to synthesize clean single-class events to construct controlled noisy mixtures for supervision. We subsequently employ a pre-trained audio encoder coupled with a discriminative classifier to automatically identify and filter out multi-source samples. Experiments show that our framework achieves strong performance on a human expert-curated test set. Finally, we release FSD50K-Solo, a model-curated subset of FSD50K containing single-source audio samples identified by our method. Beyond FSD50K, our method establishes a scalable paradigm for curating open source audio corpora.
>
---
#### [new 004] Refining Pseudo-Audio Prompts with Speech-Text Alignment for Text-Only Domain Adaptation in LLM-Based ASR
- **分类: cs.SD**

- **简介: 该论文属于文本域自适应任务，旨在解决语音识别模型在数据稀缺下的适应问题。通过建模语音与文本对齐，生成高质量伪音频提示，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.14340](https://arxiv.org/pdf/2605.14340)**

> **作者:** Ryo Magoshi; Takashi Maekaku; Yusuke Shinohara
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** LLM-based automatic speech recognition models demonstrate strong performance by connecting audio encoders and LLMs. However, data scarcity of paired speech and transcription often hinders their adaptation to new domains, making text-only domain adaptation crucial. Existing methods typically rely on either fine-tuning the LLM alone or employing pseudo-audio prompts. The former neglects essential acoustic context, while the latter either suffers from limited scalability in data-scarce conditions, or yields inexpressive prompts by leveraging only textual features, ignoring audio modality. To address this, we propose an enhanced framework that explicitly models speech-text alignment. Our method efficiently generates highly expressive pseudo-audio prompts that bridges the modality gap, enabling effective target-domain adaptation. Experiments demonstrate that our approach outperforms existing text-only methods, improving both overall error rates and out-of-vocabulary coverage.
>
---
#### [new 005] Text-Dependent Speaker Verification (TdSV) Challenge 2024: Team Naive System Report
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于文本依赖说话人验证任务，旨在提升验证准确率。通过优化神经网络和数据增强，提出高效模型并实现多模型集成，以解决资源受限下的验证问题。**

- **链接: [https://arxiv.org/pdf/2605.14896](https://arxiv.org/pdf/2605.14896)**

> **作者:** Amir Mohammad Rostami; Pourya Jafarzadeh
>
> **摘要:** This paper presents a system for the 2024 Text-Dependent Speaker Verification (TdSV) Challenge. The system achieved a Minimum Detection Cost Function (MinDCF) of 0.0461 and an Equal Error Rate (EER) of 1.3\%. Our approach focused on adapting existing state-of-the-art neural networks, ResNet-TDNN and NeXt-TDNN, originally trained on the VoxCeleb dataset. This strategy was chosen because of the limited challenge duration and the available resources at the time. In addition, we designed a lightweight and resource-efficient model, EfficientNet-A0, trained specifically on the challenge dataset to improve adaptation and strengthen the ensemble approach. Our system combines advanced neural architectures, extensive data augmentation, and optimised hyperparameters. These components helped achieve strong performance in text-dependent speaker verification. The results also demonstrate the effectiveness of multi-model ensemble learning for both speaker and phrase verification.
>
---
#### [new 006] Physics-Based iOCT Sonification for Real-time Interaction Awareness in Subretinal Injection
- **分类: cs.SD; cs.HC; eess.IV**

- **简介: 该论文属于医学影像与手术辅助任务，旨在解决 subretinal injection 中实时感知问题。通过 iOCT 数据生成声音反馈，提升医生对针尖位置和视网膜变形的感知能力。**

- **链接: [https://arxiv.org/pdf/2605.14500](https://arxiv.org/pdf/2605.14500)**

> **作者:** Luis D. Reyes Vargas; Veronica Ruozzi; Andrea K. M. Ross; Shervin Dehghani; Michael Sommersperger; Koorosh Faridpooya; Mohammad Ali Nasseri; Merle Fairhurst; Nassir Navab; Sasan Matinfar
>
> **摘要:** Subretinal injection is a delicate vitreoretinal procedure requiring precise needle placement within the subretinal space while avoiding perforation of the retinal pigment epithelium (RPE), a layer directly beneath the target with extremely limited regenerative capacity. To enhance depth perception during cannula advancement, intraoperative optical coherence tomography (iOCT) offers high-resolution cross-sectional visualization of needle-tissue interaction; however, interpreting these images requires sustained visual attention alongside the en face microscope view, thereby increasing cognitive load during critical phases and placing additional demands on the surgeon's proprioceptive control. In this paper, we propose a structured, real-time sonification framework designed for extensible mapping of iOCT-derived anatomical features into perceptual auditory feedback. The method employs a physics-inspired acoustic model driven by segmented retinal layers from a stream of iOCT B-scans, with needle motion and injection-induced retinal layer displacements serving as excitation inputs to the sound model, enabling perception of tool position and retinal deformation. In a controlled user study (n=34), the proposed sonification achieved high retinal layer identification accuracy and robust detection of retinal deformation-related events, significantly outperforming a state-of-the-art baseline in overall event identification (83.4% vs. 60.6%, p < 0.001), with gains driven primarily by enhanced detection of injection-induced retinal deformation. Evaluation by experts (n=4) confirmed the clinical relevance and potential intraoperative applicability of the method. These results establish structured iOCT sonification as a viable complementary modality for real-time surgical guidance in subretinal injection.
>
---
#### [new 007] Persian MusicGen: A Large-Scale Dataset and Culturally-Aware Generative Model for Persian Music
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音乐生成任务，旨在解决西方音乐模型在 Persian 音乐生成中的适应性问题。通过构建大规模 Persian 音乐数据集并微调 MusicGen 模型，提升其对 Persian 音乐风格的生成能力。**

- **链接: [https://arxiv.org/pdf/2605.14765](https://arxiv.org/pdf/2605.14765)**

> **作者:** Mohammad Hossein Sameti; Diba Hadi Esfangereh; Sepehr Harfi Moridani; Leili Javidpour; Mahdieh Soleymani Baghshah
>
> **备注:** 9 pages, 2 figures, 3 tables
>
> **摘要:** Persian music, with its unique tonalities, modal systems (Dastgah), and rhythmic structures, presents significant challenges for music generation models trained primarily on Western music. We address this gap by curating the first large-scale dataset of Persian songs, comprising over 900 hours high-quality audio samples across diverse sub-genres, including pop, traditional, and contemporary styles. This dataset captures the rich melodic and cultural diversity of Persian music and serves as the foundation for fine-tuning MusicGen, a state-of-the-art generative music model. We adapt MusicGen to this domain and evaluate its performance by utilizing subjective and objective metrics. To assess the semantic alignment between generated music and intended style tags, we report the proportion of relevant tags accurately reflected in the generated outputs. Our results demonstrate that the fine-tuned model produces compositions that more align with Persian stylistic conventions. This work introduces a new resource for generative music research and illustrates the adaptability of music generation models to underrepresented cultural and linguistic contexts.
>
---
#### [new 008] Break-the-Beat! Controllable MIDI-to-Drum Audio Synthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于MIDI到鼓音频的合成任务，解决传统方法控制不足的问题。提出“Break-the-Beat!”模型，通过参考音频生成高质量、节奏准确的鼓音。**

- **链接: [https://arxiv.org/pdf/2605.14555](https://arxiv.org/pdf/2605.14555)**

> **作者:** Shuyang Cui; Zhi Zhong; Qiyu Wu; Zachary Novack; Woosung Choi; Keisuke Toyama; Kin Wai Cheuk; Junghyun Koo; Yukara Ikemiya; Christian Simon; Chihiro Nagashima; Shusuke Takahashi
>
> **摘要:** Current methods for creating drum loop audio in digital music production, such as using one-shot samples or resampling, often demand non-trivial efforts of creators. While recent generative models achieve high fidelity and adhere to text, they lack the specific control needed for such a task. Existing symbolic-to-audio research often focuses on single, tonal instruments, leaving the challenge of polyphonic, percussive drum synthesis unaddressed. We address this gap by introducing ``Break-the-Beat!,'' a model capable of rendering a drum MIDI with the timbre of a reference audio. It is built by fine-tuning a pre-trained text-to-audio model with our proposed content encoder and a effective hybrid conditioning mechanism. To enable this, we construct a new dataset of paired target-reference drum audio from existing drum audio datasets. Experiments demonstrate that our model generates high-quality drum audio that follows high-resolution drum MIDI, achieving strong performance across metrics of audio quality, rhythmic alignment, and beat continuity. This offer producers a new, controllable tool for creative production. Demo page: this https URL
>
---
#### [new 009] A Benchmark for Early-stage Parkinson's Disease Detection from Speech
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于早期帕金森病检测任务，旨在解决不同研究间结果难以比较的问题。提出首个语音基准，涵盖多种任务和评估维度，促进方法公平比较与临床应用。**

- **链接: [https://arxiv.org/pdf/2605.14066](https://arxiv.org/pdf/2605.14066)**

> **作者:** Terry Yi Zhong; Cristian Tejedor-Garcia; Khiet P. Truong; Janna Maas; Louis ten Bosch; Bastiaan R. Bloem
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Early-stage Parkinson's disease (EarlyPD) detection from speech is clinically meaningful yet underexplored, and published results are hard to compare because studies differ in datasets, languages, tasks, evaluation protocols, and EarlyPD definitions. To address this issue, we propose the first benchmark for speech-based EarlyPD detection, with a speaker-independent split designed for fair and replicable cross-method evaluation on researcher-accessible datasets. The benchmark covers three common speech tasks and evaluates methods under different training-resource settings. We also present multi-dimensional evaluation breakdowns by dataset, aggregation level, gender, and disease stage to support fine-grained comparisons and clinical adoption. Our results provide a replicable reference and actionable insights, encouraging the adoption of this publicly available benchmark to advance robust and clinically meaningful EarlyPD detection from speech.
>
---
#### [new 010] IsoNet: Spatially-aware audio-visual target speech extraction in complex acoustic environments
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出IsoNet，解决紧凑设备中复杂环境下的语音提取问题。通过结合音频视觉信息与空间特征，提升语音质量，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2605.14736](https://arxiv.org/pdf/2605.14736)**

> **作者:** Dinanath Pathya; Sajen Maharjan; Binita Adhikari; Ishwor Raj Pokharel
>
> **备注:** 8 pages
>
> **摘要:** Target speech extraction remains difficult for compact devices because monaural neural models lack spatial evidence and classical beamformers lose resolving power when the microphone aperture is only a few centimetres. We present IsoNet, a user-selectable audio-visual target speech extraction system for a compact 4-microphone array. IsoNet combines complex multi-channel STFT features, GCC-PHAT spatial cues, face-conditioned visual embeddings, and auxiliary direction-of-arrival supervision inside a U-Net mask estimation network. Three curriculum variants were trained on 25,000 simulated VoxCeleb mixtures with progressively difficult SNR regimes. On a hard test set spanning -1 to 10 dB SNR, IsoNet-CL1 achieves 9.31 dB SI-SDR, a 4.85 dB improvement over the mixture, with PESQ 2.13 and STOI 0.84. Oracle delay-and-sum and MVDR beamformers degrade the same mixtures by 4.82 dB and 6.08 dB SI-SDRi, respectively, showing that the proposed learned multimodal conditioning solves a regime where conventional spatial filtering is ineffective. Ablation studies show consistent gains from visual conditioning, GCC-PHAT features, and extended delay-bin encoding. The results establish a compact-array, face-selectable speech extraction baseline under controlled simulation and identify the remaining barriers to real deployment, especially phase reconstruction, multi-interferer mixtures, and simulation-to-real transfer.
>
---
#### [new 011] PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出PROCESS-2，一个用于早期认知障碍检测的语音基准数据集，解决缺乏真实场景下临床验证数据的问题。通过收集不同认知状态参与者的语音样本，支持自动评估研究。**

- **链接: [https://arxiv.org/pdf/2605.14888](https://arxiv.org/pdf/2605.14888)**

> **作者:** Madhurananda Pahar; Caitlin H. Illingworth; Bahman Mirheidari; Hend Elghazaly; Fritz Peters; Sophie Young; Wing-Zin Leung; Labhpreet Kaur; Daniel Blackburn; Heidi Christensen
>
> **摘要:** Speech-based analysis offers a scalable and non-invasive approach for detecting cognitive decline, yet progress has been constrained by the limited availability of clinically validated datasets collected under realistic conditions. We introduce PROCESS-2, a large-scale speech dataset designed to support research on automatic assessment of cognitive impairment from spontaneous and task-oriented speech. The dataset comprises recordings from 200 healthy controls, 150 mild cognitive impairment, and 50 dementia diagnoses collected using the CognoMemory digital assessment platform. Each participant completed a single assessment session, including picture description and verbal fluency tasks, accompanied by manually verified transcripts and participant-level metadata. PROCESS-2 contains approximately 21 hours of speech audio with predefined train/test partitions. Comprehensive technical validation evaluated demographic balance, clinical consistency, recording stability, embedding-space structure, and reproducible baseline modelling performance, demonstrating clinically meaningful group separation and stable performance across modelling approaches while preserving real-world conversational variability. PROCESS-2 is released under controlled access via Hugging Face to enable responsible reuse while protecting participant privacy, providing a reproducible benchmark resource for speech-based cognitive assessment research.
>
---
#### [new 012] Case Studies and Reflections on Agentic Software Engineering for Rapid Development of Digital Music Instruments
- **分类: cs.SE; cs.SD**

- **简介: 该论文属于软件工程任务，旨在解决数字音乐工具开发中的可互操作性和入门门槛问题。通过案例研究，使用ASE技术开发音频软件，提升开发效率和软件寿命。**

- **链接: [https://arxiv.org/pdf/2605.14016](https://arxiv.org/pdf/2605.14016)**

> **作者:** Matthew John Yee-King
>
> **摘要:** The article explores the use of agentic software engineering (ASE) in the development of innovative audio software. It begins with a review of background work that lays out the challenges of longevity, interoperability and barriers to entry in digital music instrument creation, explaining recent developments in ASE and highlighting the possibility that ASE can lower barriers to entry and facilitate creation of interoperable software with greater longevity. Following that, we present case studies wherein we used ASE technology in three distinct ways to develop audio software in the C++ language with the JUCE framework. In case study 1, we re-implement Laurie Spiegel's `Music Mouse' software as a native plugin. In case study 2, we translate Pachet's `Continuator' system from Python into a native plugin. In case study 3, we develop a new 3D user interface for an existing `tracker' sequencer using OpenGL. We describe the experiences of the human developer in the case studies via autoethnographic discussion of the prompt logs and snapshots of the software as it was developed. We identify effective practice for ASE use in this domain and suggest future steps for the work involving evaluation of the method with non-programmer musicians.
>
---
#### [new 013] UMo: Unified Sparse Motion Modeling for Real-Time Co-Speech Avatars
- **分类: cs.GR; cs.CV; cs.SD**

- **简介: 该论文属于语音驱动动画生成任务，旨在解决实时共语面部与手势动画的高质量生成问题。提出UMo架构，实现文本、音频与动作的统一建模，提升实时性能与动画质量。**

- **链接: [https://arxiv.org/pdf/2605.14731](https://arxiv.org/pdf/2605.14731)**

> **作者:** Xiaoyu Zhan; Xinyu Fu; Chenghao Yang; Xiaohong Zhang; Dongjie Fu; Pengcheng Fang; Tengjiao Sun; Xiaohao Cai; Hansung Kim; Yuanqi Li; Jie Guo; Yanwen Guo
>
> **摘要:** Speech-driven gestures and facial animations are fundamental to expressive digital avatars in games, virtual production, and interactive media. However, existing methods are either limited to a single modality for audio motion alignment, failing to fully utilize the potential of massive human motion data, or are constrained by the representation ability and throughput of multimodal models, which makes it difficult to achieve high-quality motion generation or real-time performance. We present UMo, a unified sparse motion modeling architecture for real-time co-speech avatars, which processes text, audio, and motion tokens within a unified formulation. Leveraging a spatially sparse Mixture-of-Experts framework and a temporally sparse, keyframe-centric design, UMo efficiently performs real-time dense reconstruction, enabling temporally coherent and high-fidelity animation generation for both facial expressions and gestures. Furthermore, we implement a multi-stage training strategy with targeted audio augmentation to enhance acoustic diversity and semantic consistency. Consequently, UMo preserves fine-grained speech-motion alignment even under strict latency constraints. Extensive quantitative and qualitative evaluations show that UMo achieves better output quality under low latency and real-time performance constraints, offering a practical solution for high-fidelity real-time co-speech avatars.
>
---
#### [new 014] A Calculus-Based Framework for Determining Vocabulary Size in End-to-End ASR
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决端到端ASR中词汇量选择的问题。通过数学方法优化词汇量参数，提升ASR性能。**

- **链接: [https://arxiv.org/pdf/2605.14427](https://arxiv.org/pdf/2605.14427)**

> **作者:** Sunil Kumar Kopparapu
>
> **备注:** 8 pages, is an extension of the paper S. K. Kopparapu and A. Panda, A cost minimization approach to fix the vocabulary size in a tokenizer for an end-to-end ASR system, in Proceedings of the 2024 International Conference on Pattern Recognition, Kolkata, India, 2024
>
> **摘要:** In hybrid automatic speech recognition (ASR) systems, the vocabulary size is unambiguous, typically determined by the number of phones, bi-phones, or tri-phones present in the language. In contrast, end-to-end ASR systems derive their vocabulary, often referred to as tokens from the text corpus used for training. The choice and, more importantly, the size of this vocabulary is a critical hyper-parameter in training end-to-end ASR systems. Tokenization algorithms such as Byte Pair Encoding (BPE), WordPiece, and Unigram Language Model (ULM) use the vocabulary size as an input hyper-parameter to generate the sub-words employed during ASR training. Popular toolkits like ESPNet provide a fixed vocabulary size in their training recipes, but there is little documentation or discussion in the literature regarding how these values are determined. Recent work [1] has formalized an approach to identify the vocabulary size best suited for end-to-end ASR, introducing a cost function framework that treats the tokenization process as a black box. In this paper, we build upon that foundation by curve fitting the training data and using the principle of first and second derivative tests in calculus to formally estimate the vocabulary size hyper-parameter. We demonstrate the utility and usefulness of our approach by applying it on a standard Librispeech corpus and show that the optimal choice of vocabulary size hyper-parameter improves the performance of the ASR. The main contribution of this paper in formalizing an approach to identify the vocabulary size best suited for training an end-to-end ASR system.
>
---
#### [new 015] Streaming Speech-to-Text Translation with a SpeechLLM
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语音到文本的实时翻译任务，旨在解决传统系统延迟高、无法实时处理的问题。提出一种基于大语言模型的架构，使系统能根据音频内容动态生成译文，实现低延迟的流式翻译。**

- **链接: [https://arxiv.org/pdf/2605.14766](https://arxiv.org/pdf/2605.14766)**

> **作者:** Titouan Parcollet; Shucong Zhang; Xianrui Zheng; Rogier C. van Dalen
>
> **备注:** 9 pages of main text; 24 pages in total
>
> **摘要:** Normally, a system that translates speech into text consists of separate modules for speech recognition and text-to-text translation. Combining those tasks into a SpeechLLM promises to exploit paralinguistic information in the speech and to reduce cascaded errors. But existing SpeechLLM systems are slow since they do not work in a real streaming fashion: they wait for a complete utterance of audio before outputting a translation, or output tokens at fixed intervals, which is not suitable for real applications. This work proposes an LLM-based architecture for real streaming speech-to-text translation. The LLM learns not just to emit output tokens, but also to decide whether it has seen enough audio to do so. The system is trained using automatic alignments of the input speech and the output text. In experiments on different language pairs, the system achieves a translation quality close to the non-streaming baseline, but with a latency of only 1-2 seconds.
>
---
#### [new 016] AudioMosaic: Contrastive Masked Audio Representation Learning
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文提出AudioMosaic，一种基于对比学习的音频编码器，用于解决音频表示学习问题。通过结构化时频掩码构建正样本对，提升模型在不同数据集和环境中的迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.14231](https://arxiv.org/pdf/2605.14231)**

> **作者:** Hanxun Huang; Qizhou Wang; Xingjun Ma; Cihang Xie; Christopher Leckie; Sarah Erfani
>
> **备注:** ICML2026
>
> **摘要:** Audio self-supervised learning (SSL) aims to learn general-purpose representations from large-scale unlabeled audio data. While recent advances have been driven mainly by generative reconstruction objectives, contrastive approaches remain less explored, partly due to the difficulty of designing effective audio augmentations and the large batch sizes required for contrastive pre-training. We introduce \textbf{AudioMosaic}, a contrastive learning-based audio encoder for general audio understanding. During pre-training, AudioMosaic constructs positive pairs by applying structured time-frequency masking to spectrogram patches, which reduces memory usage and enables efficient large-batch training. Compared with generative approaches, the AudioMosaic encoder learns more discriminative utterance-level representations that demonstrate strong transferability across datasets, domains, and acoustic conditions. Extensive experiments show that AudioMosaic achieves state-of-the-art performance on several standard audio benchmarks under both linear probing and fine-tuning. We further show that integrating the pretrained AudioMosaic encoder into audio-language models improves performance on audio-language tasks. The code is publicly available in our \href{this https URL}{GitHub repository}.
>
---
## 更新

#### [replaced 001] BioSEN: A Bio-acoustic Signal Enhancement Network for Animal Vocalizations
- **分类: cs.SD; cs.LG; q-bio.NC**

- **简介: 该论文属于生物声学信号增强任务，旨在解决动物叫声提取困难的问题。通过构建BioSEN模型，提升生物音频质量，助力生态保护。**

- **链接: [https://arxiv.org/pdf/2605.12534](https://arxiv.org/pdf/2605.12534)**

> **作者:** Tianyu Song; Ton Viet Ta; Ngamta Thamwattana; Hisako Nomura; Linh Thi Hoai Nguyen
>
> **摘要:** Most work in audio enhancement targets human speech, while bioacoustics is less studied due to noisy recordings and the distinct traits of animal sounds. To fill this gap, we adapt speech enhancement methods and build BioSEN, a model made for bioacoustic signals. BioSEN has three modules: a multi-scale dual-axis attention unit for time-frequency feature extraction, a bio-harmonic multi-scale enhancement unit for capturing harmonic structures, and an energy-adaptive gating connection unit that uses frequency weights to keep vocalizations from being removed as noise. Tests on three bioacoustic datasets show that BioSEN matches or exceeds state-of-the-art speech enhancement models while using far less computation. These results show BioSEN's strength for bioacoustic audio enhancement and its promise for biodiversity monitoring and conservation.
>
---
#### [replaced 002] AVEX: What Matters for Animal Vocalization Encoding
- **分类: cs.SD; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于生物声学领域，旨在解决有限标注数据下通用声学编码器的训练问题。通过大规模实验，提出一种基于自监督预训练和混合数据微调的编码器，提升多种任务性能。**

- **链接: [https://arxiv.org/pdf/2508.11845](https://arxiv.org/pdf/2508.11845)**

> **作者:** Marius Miron; David Robinson; Milad Alizadeh; Ellen Gilsenan-McMahon; Gagan Narula; Emmanuel Chemla; Maddie Cusimano; Felix Effenberger; Masato Hagiwara; Benjamin Hoffman; Sara Keen; Diane Kim; Jane Lawton; Jen-Yu Liu; Aza Raskin; Olivier Pietquin; Matthieu Geist
>
> **备注:** In The Fourteenth International Conference on Learning Representations 2026
>
> **摘要:** Bioacoustics, the study of sounds produced by living organisms, plays a vital role in conservation, biodiversity monitoring, and behavioral studies. Many tasks in this field, such as species, individual, and behavior classification and detection, are well-suited to machine learning. However, they often suffer from limited annotated data, highlighting the need for a general-purpose bioacoustic encoder capable of extracting useful representations for diverse downstream tasks. Such encoders have been proposed before, but are often limited in scope due to a focus on a narrow range of species (typically birds), and a reliance on a single model architecture or training paradigm. Moreover, they are usually evaluated on a small set of tasks and datasets. In this work, we present a large-scale empirical study that covers aspects of bioacoustics that are relevant to research but have previously been scarcely considered: training data diversity and scale, model architectures and training recipes, and the breadth of evaluation tasks and datasets. We obtain encoders that are state-of-the-art on the existing and proposed benchmarks. We also identify what matters for training these encoders, such that this work can be extended when more data are available or better architectures are proposed. Specifically, across 26 datasets with tasks including species classification, detection, individual ID, and vocal repertoire discovery, we find self-supervised pre-training followed by supervised post-training on a mixed bioacoustics + general-audio corpus yields the strongest in- and out-of-distribution performance. We show the importance of data diversity in both stages. To support ongoing research and application, we will release the model checkpoints.
>
---
#### [replaced 003] AaSP: Aliasing-aware Self-Supervised Pre-Training for Audio Spectrogram Transformers
- **分类: cs.SD; cs.LG; stat.ML**

- **简介: 该论文提出AaSP框架，解决音频谱图Transformer中的混叠问题。通过改进补丁表示和掩码建模，提升模型在低频和高频信息上的稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2512.03637](https://arxiv.org/pdf/2512.03637)**

> **作者:** Kohei Yamamoto; Kosuke Okusa
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech and Language Processing (TALSP). Copyright IEEE
>
> **摘要:** Transformer-based audio self-supervised learning (SSL) models commonly use spectrograms, vision-style Transformers, and masked modeling objectives. However, convolutional patchification with temporal downsampling lowers the effective Nyquist frequency and introduces aliasing, while naïve low-pass filtering may remove task-relevant high-frequency cues. We present AaSP, an aliasing-aware self-supervised pre-training framework for audio spectrogram transformers. AaSP combines an aliasing-aware patch representation, teacher-student masked modeling, a cross-attention predictor, and multi-mask contrastive regularization to learn representations that integrate features from alias-prone modulation bands while remaining stable across masked views. Its patch-embedding module, Aliasing-aware Patch Embedding (AaPE), augments standard patch tokens with features from alias-prone modulation bands using a band-limited complex sinusoidal kernel with a two-sided exponential window. The kernel's frequency and decay parameters are estimated from the input, enabling adaptive subband analysis whose outputs are fused with standard patch tokens. We pre-train on AudioSet and evaluate the learned representations by fine-tuning and linear evaluation on acoustic/environmental, speech, and music recognition benchmarks. Under fine-tuning, the full AaSP framework achieves state-of-the-art results on AS-20K, ESC-50, and NSynth among compared self-supervised baselines, while remaining competitive elsewhere. Linear evaluation shows a similar trend, including gains on US8K and NSynth. Overall, AaSP learns representations that are more stable under aliasing-sensitive temporal perturbations and competitive for downstream transfer.
>
---
#### [replaced 004] The Spheres Dataset: Multitrack Orchestral Recordings for Music Source Separation and Information Retrieval
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文介绍The Spheres数据集，用于古典音乐源分离和信息检索任务，解决复杂管弦乐场景下的声音分离问题，通过多轨录音和声学分析提供基准测试支持。**

- **链接: [https://arxiv.org/pdf/2511.21247](https://arxiv.org/pdf/2511.21247)**

> **作者:** Jaime Garcia-Martinez; David Diaz-Guerra; John Anderson; Ricardo Falcon-Perez; Pablo Cabañas-Molero; Tuomas Virtanen; Julio J. Carabias-Orti; Pedro Vera-Candeas
>
> **摘要:** This paper introduces The Spheres dataset, multitrack orchestral recordings designed to advance machine learning research in music source separation and related MIR tasks within the classical music domain. The dataset is composed of over one hour recordings of musical pieces performed by the Colibrì Ensemble at The Spheres recording studio, capturing two canonical works - Tchaikovsky's Romeo and Juliet and Mozart's Symphony No. 40 - along with chromatic scales and solo excerpts for each instrument. The recording setup employed 23 microphones, including close spot, main, and ambient microphones, enabling the creation of realistic stereo mixes with controlled bleeding and providing isolated stems for supervised training of source separation models. In addition, room impulse responses were estimated for each instrument position, offering valuable acoustic characterization of the recording space. We present the dataset structure, acoustic analysis, and baseline evaluations using X-UMX based models for orchestral family separation and microphone debleeding. Results highlight both the potential and the challenges of source separation in complex orchestral scenarios, underscoring the dataset's value for benchmarking and for exploring new approaches to separation, localization, dereverberation, and immersive rendering of classical music.
>
---
#### [replaced 005] V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于视频到音乐生成任务，解决视频与音乐时间对齐问题。提出V2M-ZERO方法，在无需视频-音乐配对数据的情况下，通过时序结构对齐实现高质量音乐生成。**

- **链接: [https://arxiv.org/pdf/2603.11042](https://arxiv.org/pdf/2603.11042)**

> **作者:** Yan-Bo Lin; Jonah Casebeer; Long Mai; Aniruddha Mahapatra; Gedas Bertasius; Nicholas J. Bryan
>
> **备注:** Project page: this https URL
>
> **摘要:** Generating music that temporally aligns with video events is challenging for existing text-to-music models, which lack fine-grained temporal control. We introduce V2M-ZERO, a video-to-music generation approach that generates time-aligned music with disentangled time synchronization and semantic control (e.g., genre, mood) from video while requiring zero video-music pairs at training time. Our method is motivated by a key observation: temporal synchronization requires matching when and how much change occurs, not what changes. While musical and visual events differ semantically, they exhibit shared temporal structure that can be captured independently within each modality. We capture this structure through event curves computed from intra-modal similarity using pretrained music and video encoders. By measuring temporal change within each modality independently, these curves provide comparable representations across modalities. This enables a simple training strategy: fine-tune a text-to-music model on music-event curves, then substitute video-event curves at inference without cross-modal training or paired data. Across OES-Pub, MovieGenBench-Music, and AIST++, V2M-ZERO achieves state-of-the-art performance without any paired music-video data, surpassing the strongest prior baselines per metric with 5-9% higher audio quality, 13-15% better semantic alignment, 21-52% improved temporal synchronization, and 28% higher beat alignment on dance videos. We find similar results via a large crowd-source subjective listening test. Our results validate that temporal alignment through within-modality features is not only effective for video-to-music generation but also leads to better performance than paired cross-modal supervision. Furthermore, our approach enables independent controls for timing and music style (e.g., genre, mood) for more controllable generation.
>
---
#### [replaced 006] Asymmetric Encoder-Decoder Based on Time-Frequency Correlation for Speech Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，解决复杂声学环境中多人说话、噪声和混响同时存在的分离难题。提出SR-CorrNet框架，通过时间-频率域的分离与重建策略提升分离效果。**

- **链接: [https://arxiv.org/pdf/2603.29097](https://arxiv.org/pdf/2603.29097)**

> **作者:** Ui-Hyeop Shin; Hyung-Min Park
>
> **备注:** Submitted to IEEE Transactions on Audio, Speech, and Language Processing (TASLPRO) Code: this https URL
>
> **摘要:** Speech separation in realistic acoustic environments remains challenging because overlapping speakers, background noise, and reverberation must be resolved simultaneously. Although recent time-frequency (TF) domain models have shown strong performance, most still rely on late-split architectures, where speaker disentanglement is deferred to the final stage, creating an information bottleneck and weakening discriminability under adverse conditions. To address this issue, we propose SR-CorrNet, an asymmetric encoder-decoder framework that introduces the separation-reconstruction (SepRe) strategy into a TF dual-path backbone. The encoder performs coarse separation from mixture observations, while the weight-shared decoder progressively reconstructs speaker-discriminative features with cross-speaker interaction, enabling stage-wise refinement. To complement this architecture, we formulate speech separation as a structured correlation-to-filter problem: spatio-spectro-temporal correlations computed from the observations are used as input features, and the corresponding deep filters are estimated to recover target signals. We further incorporate an attractor-based dynamic split module to adapt the number of output streams to the actual speaker configuration. Experimental results on WSJ0-{2,3,4,5}Mix, WHAMR!, and LibriCSS demonstrate consistent improvements across anechoic, noisy-reverberant, and real-recorded conditions in both single- and multi-channel settings, highlighting the effectiveness of TF-domain SepRe with correlation-based filter estimation for speech separation.
>
---
#### [replaced 007] Instantaneous Spectra Analysis of Pulse Series -- Application to Lung Sounds with Abnormalities
- **分类: physics.soc-ph; cs.SD**

- **简介: 该论文属于信号分析任务，旨在解决传统傅里叶分析时间-频率分辨率受限的问题。通过引入LXC替代PBC，实现脉冲序列的瞬时谱分析，并应用于异常肺音研究。**

- **链接: [https://arxiv.org/pdf/2602.03680](https://arxiv.org/pdf/2602.03680)**

> **作者:** Fumihiko Ishiyama
>
> **备注:** 10 pages, 7 figures. To appear Proc. IEEE CSPA 2026
>
> **摘要:** The origin of the "theoretical limit of time-frequency resolution of Fourier analysis" is from its numerical implementation, especially from an assumption of "Periodic Boundary Condition (PBC)," which was introduced a century ago. We previously proposed to replace this condition with "Linear eXtrapolation Condition (LXC)," which does not require periodicity. This feature makes instantaneous spectra analysis of pulse series available, which replaces the short time Fourier transform (STFT). We applied the instantaneous spectra analysis to two lung sounds with abnormalities (crackles and wheezing) and to a normal lung sound, as a demonstration. Among them, crackles contains a random pulse series. The spectrum of each pulse is available, and the spectrogram of pulse series is available with assembling each spectrum. As a result, the time-frequency structure of given pulse series is visualized.
>
---
