# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] MIDI-Informed Singing Accompaniment Generation in a Compositional Song Pipeline
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于歌曲生成任务，旨在解决传统模型数据与计算需求高、编辑性差的问题。提出MIDI-SAG方法，通过MIDI信息提升伴奏与演唱的对齐效果，并支持间断演唱场景。**

- **链接: [https://arxiv.org/pdf/2602.22029v1](https://arxiv.org/pdf/2602.22029v1)**

> **作者:** Fang-Duo Tsai; Yi-An Lai; Fei-Yueh Chen; Hsueh-Wei Fu; Li Chai; Wei-Jaw Lee; Hao-Chung Cheng; Yi-Hsuan Yang
>
> **摘要:** Song generation aims to produce full songs with vocals and accompaniment from lyrics and text descriptions, yet end-to-end models remain data- and compute-intensive and provide limited editability. We advocate a compositional alternative that decomposes the task into melody composition, singing voice synthesis, and singing accompaniment generation. Central to our approach is MIDI-informed singing accompaniment generation (MIDI-SAG), which conditions accompaniment on the symbolic vocal-melody MIDI to improve rhythmic and harmonic alignment between singing and instrumentation. Moreover, beyond conventional SAG settings that assume continuously sung vocals, compositional song generation features intermittent vocals; we address this by combining explicit rhythmic/harmonic controls with audio continuation to keep the backing track consistent across vocal and non-vocal regions. With lightweight newly trained components requiring only 2.5k hours of audio on a single RTX 3090, our pipeline approaches the perceptual quality of recent open-source end-to-end baselines in several metrics. We provide audio demos and will open-source our model at https://composerflow.github.io/web/.
>
---
#### [new 002] UniWhisper: Efficient Continual Multi-task Training for Robust Universal Audio Representation
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出UniWhisper，解决音频表示学习中多任务性能不均衡的问题。通过统一指令格式进行持续多任务训练，提升通用音频编码器的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21772v1](https://arxiv.org/pdf/2602.21772v1)**

> **作者:** Yuxuan Chen; Peize He; Haoyuan Xu; Junzi Zhang
>
> **摘要:** A universal audio representation should capture fine-grained speech cues and high-level semantics for environmental sounds and music in a single encoder. Existing encoders often excel in one domain but degrade in others. We propose UniWhisper, an efficient continual multi-task training framework that casts heterogeneous audio tasks into a unified instruction and answer format. This enables standard next-token training without task-specific heads and losses. We train it on 38k hours of public audio and assess the encoder using shallow MLP probes and k-nearest neighbors (kNN) on 20 tasks spanning speech, environmental sound, and music. UniWhisper reaches normalized weighted averages of 0.81 with MLP probes and 0.61 with kNN, compared to 0.64 and 0.46 for Whisper, while retaining strong speech performance.
>
---
#### [new 003] iMiGUE-Speech: A Spontaneous Speech Dataset for Affective Analysis
- **分类: eess.AS; cs.CL**

- **简介: 论文介绍iMiGUE-Speech数据集，用于情感分析任务，解决自发情绪研究问题。该数据集包含语音、转录文本及对齐信息，支持语音情感识别和基于文本的 sentiment 分析。**

- **链接: [https://arxiv.org/pdf/2602.21464v1](https://arxiv.org/pdf/2602.21464v1)**

> **作者:** Sofoklis Kakouros; Fang Kang; Haoyu Chen
>
> **备注:** Accepted to Speech Prosody 2026
>
> **摘要:** This work presents iMiGUE-Speech, an extension of the iMiGUE dataset that provides a spontaneous affective corpus for studying emotional and affective states. The new release focuses on speech and enriches the original dataset with additional metadata, including speech transcripts, speaker-role separation between interviewer and interviewee, and word-level forced alignments. Unlike existing emotional speech datasets that rely on acted or laboratory-elicited emotions, iMiGUE-Speech captures spontaneous affect arising naturally from real match outcomes. To demonstrate the utility of the dataset and establish initial benchmarks, we introduce two evaluation tasks for comparative assessment: speech emotion recognition and transcript-based sentiment analysis. These tasks leverage state-of-the-art pre-trained representations to assess the dataset's ability to capture spontaneous affective states from both acoustic and linguistic modalities. iMiGUE-Speech can also be synchronously paired with micro-gesture annotations from the original iMiGUE dataset, forming a uniquely multimodal resource for studying speech-gesture affective dynamics. The extended dataset is available at https://github.com/CV-AC/imigue-speech.
>
---
#### [new 004] TG-ASR: Translation-Guided Learning with Parallel Gated Cross Attention for Low-Resource Automatic Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 论文提出TG-ASR框架，解决低资源语言（如台湾闽南语）语音识别难题。通过翻译引导和跨语言注意力机制，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2602.22039v1](https://arxiv.org/pdf/2602.22039v1)**

> **作者:** Cheng-Yeh Yang; Chien-Chun Wang; Li-Wei Chen; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Low-resource automatic speech recognition (ASR) continues to pose significant challenges, primarily due to the limited availability of transcribed data for numerous languages. While a wealth of spoken content is accessible in television dramas and online videos, Taiwanese Hokkien exemplifies this issue, with transcriptions often being scarce and the majority of available subtitles provided only in Mandarin. To address this deficiency, we introduce TG-ASR for Taiwanese Hokkien drama speech recognition, a translation-guided ASR framework that utilizes multilingual translation embeddings to enhance recognition performance in low-resource environments. The framework is centered around the parallel gated cross-attention (PGCA) mechanism, which adaptively integrates embeddings from various auxiliary languages into the ASR decoder. This mechanism facilitates robust cross-linguistic semantic guidance while ensuring stable optimization and minimizing interference between languages. To support ongoing research initiatives, we present YT-THDC, a 30-hour corpus of Taiwanese Hokkien drama speech with aligned Mandarin subtitles and manually verified Taiwanese Hokkien transcriptions. Comprehensive experiments and analyses identify the auxiliary languages that most effectively enhance ASR performance, achieving a 14.77% relative reduction in character error rate and demonstrating the efficacy of translation-guided learning for underrepresented languages in practical applications.
>
---
#### [new 005] A Knowledge-Driven Approach to Music Segmentation, Music Source Separation and Cinematic Audio Source Separation
- **分类: eess.AS; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属于音频处理任务，解决音乐分割与声源分离问题。通过结合知识（如乐谱）和模型（如隐马尔可夫模型），实现无需预分割数据的自主学习，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2602.21476v1](https://arxiv.org/pdf/2602.21476v1)**

> **作者:** Chun-wei Ho; Sabato Marco Siniscalchi; Kai Li; Chin-Hui Lee
>
> **摘要:** We propose a knowledge-driven, model-based approach to segmenting audio into single-category and mixed-category chunks with applications to source separation. "Knowledge" here denotes information associated with the data, such as music scores. "Model" here refers to tool that can be used for audio segmentation and recognition, such as hidden Markov models. In contrast to conventional learning that often relies on annotated data with given segment categories and their corresponding boundaries to guide the learning process, the proposed framework does not depend on any pre-segmented training data and learns directly from the input audio and its related knowledge sources to build all necessary models autonomously. Evaluation on simulation data shows that score-guided learning achieves very good music segmentation and separation results. Tested on movie track data for cinematic audio source separation also shows that utilizing sound category knowledge achieves better separation results than those obtained with data-driven techniques without using such information.
>
---
#### [new 006] EmoOmni: Bridging Emotional Understanding and Expression in Omni-Modal LLMs
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多模态情感对话任务，旨在解决Omni-LLMs情感理解与表达不足的问题。提出EmoOmni框架及E-CoT机制，提升情感推理与表达准确性。**

- **链接: [https://arxiv.org/pdf/2602.21900v1](https://arxiv.org/pdf/2602.21900v1)**

> **作者:** Wenjie Tian; Zhixian Zhao; Jingbin Hu; Huakang Chen; Haohe Liu; Binshen Mu; Lei Xie
>
> **摘要:** The evolution of Omni-Modal Large Language Models~(Omni-LLMs) has revolutionized human--computer interaction, enabling unified audio-visual perception and speech response. However, existing Omni-LLMs struggle with complex real-world scenarios, often leading to superficial understanding and contextually mismatched emotional responses. This issue is further intensified by Omni-LLM's Thinker-Talker architectures, which are implicitly connected through hidden states, leading to the loss of emotional details. In this work, we present EmoOmni, a unified framework for accurate understanding and expression in multimodal emotional dialogue. At its core, we introduce the emotional Chain-of-Thought~(E-CoT), which enforces a reasoning from fine-grained multimodal perception to textual response. Moreover, we explicitly treat E-CoT as high-level emotional instructions that guide the talker, enabling accurate emotional expression. Complementing the model, we construct EmoOmniPipe to obtain the real-world annotated dialogue data and establish a benchmark, EmoOmniEval, to facilitate systematic assessment of multimodal emotional dialogue task. Experiments show that EmoOmni-7B achieves comparable performance with Qwen3Omni-30B-A3B-Thinking under the same talker.
>
---
#### [new 007] Robust Long-Form Bangla Speech Processing: Automatic Speech Recognition and Speaker Diarization
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于 Bengali 语音处理任务，解决长文本语音识别和说话人分割问题。通过模型微调、语音分离和分段优化，提升了识别与分割效果。**

- **链接: [https://arxiv.org/pdf/2602.21741v1](https://arxiv.org/pdf/2602.21741v1)**

> **作者:** MD. Sagor Chowdhury; Adiba Fairooz Chowdhury
>
> **备注:** 6 pages, 5 figures, 3 tables; system paper submitted to DL Sprint 4.0 (Kaggle)
>
> **摘要:** We describe our end-to-end system for Bengali long-form speech recognition (ASR) and speaker diarization submitted to the DL Sprint 4.0 competition on Kaggle. Bengali presents substantial challenges for both tasks: a large phoneme inventory, significant dialectal variation, frequent code-mixing with English, and a relative scarcity of large-scale labelled corpora. For ASR we achieve a best private Word Error Rate (WER) of 0.37738 and public WER of 0.36137, combining a BengaliAI fine-tuned Whisper medium model with Demucs source separation for vocal isolation, silence-boundary chunking, and carefully tuned generation hyperparameters. For speaker diarization we reach a best private Diarization Error Rate (DER) of 0.27671 and public DER of 0.20936 by replacing the default segmentation model inside the pyannote.audio pipeline with a Bengali-fine-tuned variant, pairing it with wespeaker-voxceleb-resnet34-LM embeddings and centroid-based agglomerative clustering. Our experiments demonstrate that domain-specific fine-tuning of the segmentation component, vocal source separation, and natural silence-aware chunking are the three most impactful design choices for low-resource Bengali speech processing.
>
---
## 更新

#### [replaced 001] MDM-ASR: Bridging Accuracy and Efficiency in ASR with Diffusion-Based Non-Autoregressive Decoding
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决NAR模型性能不足与AR模型效率低的问题。提出基于扩散模型的NAR框架，提升准确率同时保持并行解码效率。**

- **链接: [https://arxiv.org/pdf/2602.18952v2](https://arxiv.org/pdf/2602.18952v2)**

> **作者:** Hao Yen; Pin-Jui Ku; Ante Jukić; Sabato Marco Siniscalchi
>
> **备注:** 10 pages, submitted to Interspeech 2026 Long Paper track
>
> **摘要:** In sequence-to-sequence Transformer ASR, autoregressive (AR) models achieve strong accuracy but suffer from slow decoding, while non-autoregressive (NAR) models enable parallel decoding at the cost of degraded performance. We propose a principled NAR ASR framework based on Masked Diffusion Models to reduce this gap. A pre-trained speech encoder is coupled with a Transformer diffusion decoder conditioned on acoustic features and partially masked transcripts for parallel token prediction. To mitigate the training-inference mismatch, we introduce Iterative Self-Correction Training that exposes the model to its own intermediate predictions. We also design a Position-Biased Entropy-Bounded Confidence-based sampler with positional bias to further boost results. Experiments across multiple benchmarks demonstrate consistent gains over prior NAR models and competitive performance with strong AR baselines, while retaining parallel decoding efficiency.
>
---
#### [replaced 002] Continuous Telemonitoring of Heart Failure using Personalised Speech Dynamics
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于心力衰竭远程监测任务，解决个体语音特征差异导致的模型准确性问题，提出LIPT框架和PSE模型以提升监测效果。**

- **链接: [https://arxiv.org/pdf/2602.19674v2](https://arxiv.org/pdf/2602.19674v2)**

> **作者:** Yue Pan; Xingyao Wang; Hanyue Zhang; Liwei Liu; Changxin Li; Gang Yang; Rong Sheng; Yili Xia; Ming Chu
>
> **摘要:** Remote monitoring of heart failure (HF) via speech signals provides a non-invasive and cost-effective solution for long-term patient management. However, substantial inter-individual heterogeneity in vocal characteristics often limits the accuracy of traditional cross-sectional classification models. To address this, we propose a Longitudinal Intra-Patient Tracking (LIPT) scheme designed to capture the trajectory of relative symptomatic changes within individuals. Central to this framework is a Personalised Sequential Encoder (PSE), which transforms longitudinal speech recordings into context-aware latent representations. By incorporating historical data at each timestamp, the PSE facilitates a holistic assessment of the clinical trajectory rather than modelling discrete visits independently. Experimental results from a cohort of 225 patients demonstrate that the LIPT paradigm significantly outperforms the classic cross-sectional approaches, achieving a recognition accuracy of 99.7% for clinical status transitions. The model's high sensitivity was further corroborated by additional follow-up data, confirming its efficacy in predicting HF deterioration and its potential to secure patient safety in remote, home-based settings. Furthermore, this work addresses the gap in existing literature by providing a comprehensive analysis of different speech task designs and acoustic features. Taken together, the superior performance of the LIPT framework and PSE architecture validates their readiness for integration into long-term telemonitoring systems, offering a scalable solution for remote heart failure management.
>
---
#### [replaced 003] The Affective Bridge: Preserving Speech Representations while Enhancing Deepfake Detection vian emotional Constraints
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升检测效果同时保留语义信息。通过引入情感作为约束，优化语音表示，提高检测准确性。**

- **链接: [https://arxiv.org/pdf/2512.11241v2](https://arxiv.org/pdf/2512.11241v2)**

> **作者:** Yupei Li; Chenyang Lyu; Longyue Wang; Weihua Luo; Kaifu Zhang; Björn W. Schuller
>
> **备注:** Submitted to interspeech 2026 for review
>
> **摘要:** Speech deepfake detection (DFD) has benefited from diverse acoustic and semantic speech representations, many of which encode valuable speech information and are costly to train. Existing approaches typically enhance DFD by tuning the representations or applying post-hoc classification on frozen features, limiting control over improving discriminative DF cues without distorting original semantics. We find that emotion is encoded across diverse speech features and correlates with DFD. Therefore, we introduce a unified, feature-agnostic, and non-destructive training framework that uses emotion as a bridging constraint to guide speech features toward DFD, treating emotion recognition as a representation alignment objective rather than an auxiliary task, while preserving the original semantic information. Experiments on FakeOrReal and IntheWild show accuracy improvements of up to 6\% and 2\%, respectively, with corresponding reductions in equal error rate. Code is in the supplementary material.
>
---
#### [replaced 004] Metric Analysis for Spatial Semantic Segmentation of Sound Scenes
- **分类: cs.SD**

- **简介: 该论文针对声场空间语义分割（S5）任务，解决分离与分类指标不兼容的问题，提出CASA-SDR新度量，更准确评估系统性能。**

- **链接: [https://arxiv.org/pdf/2511.07075v2](https://arxiv.org/pdf/2511.07075v2)**

> **作者:** Mayank Mishra; Paul Magron; Romain Serizel
>
> **备注:** 5 pages; content+bibliography
>
> **摘要:** Spatial semantic segmentation of sound scenes (S5) consists of jointly performing audio source separation and sound event classification from a multichannel audio mixture. Evaluating S5 systems with separation and classification metrics individually makes system comparison difficult, whereas existing joint metrics, such as the class-aware signal-to-distortion ratio (CA-SDR), can conflate separation and labeling errors. In particular, CA-SDR relies on predicted class labels for source matching, which may obscure label swaps or misclassifications when the underlying source estimates remain perceptually correct. In this work, we introduce the class and source-aware signal-to-distortion ratio (CASA-SDR), a new metric that performs permutation-invariant source matching before computing classification errors, thereby shifting from a classification-focused approach to a separation-focused approach. We first analyze CA-SDR in controlled scenarios with oracle separation and synthetic classification errors, as well as under controlled cross-contamination between sources, and compare its behavior to that of the classical SDR and CASA-SDR. We also study the impact of classification errors on the metrics by introducing error-based and source-based aggregation strategies. Finally, we compare CA-SDR and CASA-SDR on systems submitted to Task 4 of the DCASE 2025 challenge, highlighting the cases where CA-SDR over-penalizes label swaps or poorly separated sources, while CASA-SDR provides a more interpretable separation-centric assessment of S5 performance.
>
---
#### [replaced 005] Aligning Audio Captions with Human Preferences
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音频字幕任务，旨在解决传统方法依赖昂贵标注数据且不符人类偏好的问题。通过RLHF和CLAP模型，提升字幕质量与人类偏好一致。**

- **链接: [https://arxiv.org/pdf/2509.14659v2](https://arxiv.org/pdf/2509.14659v2)**

> **作者:** Kartik Hegde; Rehana Mahfuz; Yinyi Guo; Erik Visser
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** Current audio captioning relies on supervised learning with paired audio-caption data, which is costly to curate and may not reflect human preferences in real-world scenarios. To address this, we propose a preference-aligned audio captioning framework based on Reinforcement Learning from Human Feedback (RLHF). To capture nuanced preferences, we train a Contrastive Language-Audio Pretraining (CLAP) based reward model using human-labeled pairwise preference data. This reward model is integrated into an RL framework to fine-tune any baseline captioning system without ground-truth annotations. Extensive human evaluations across multiple datasets show that our method produces captions preferred over baseline models, particularly when baselines fail to provide correct and natural captions. Furthermore, our framework achieves performance comparable to supervised approaches with ground-truth data, demonstrating effective alignment with human preferences and scalability in real-world use.
>
---
#### [replaced 006] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决低帧率下语义信息丢失的问题。提出FlexiCodec，通过动态帧率和双流结构提升语义保留与音频重建质量。**

- **链接: [https://arxiv.org/pdf/2510.00981v3](https://arxiv.org/pdf/2510.00981v3)**

> **作者:** Jiaqi Li; Yao Qian; Yuxuan Hu; Leying Zhang; Xiaofei Wang; Heng Lu; Manthan Thakker; Jinyu Li; Sheng Zhao; Zhizheng Wu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that a major challenge for very low frame rate tokens is missing semantic information. This paper introduces FlexiCodec to address this limitation. FlexiCodec improves semantic preservation with a dynamic frame rate approach and introduces a novel architecture featuring an ASR feature-assisted dual stream encoding and Transformer bottlenecks. With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. A dynamic frame rate also allows FlexiCodec to support inference-time controllable frame rates between 3Hz and 12.5Hz. Experiments on 6.25Hz, 8.3Hz and 12.5Hz average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS. Demos are available at: https://flexicodec.github.io. Code is available at: https://github.com/amphionteam/flexicodec.
>
---
#### [replaced 007] Discrete Optimal Transport and Voice Conversion
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音转换任务，解决如何更有效地对齐说话人特征的问题。提出kDOT框架，利用离散最优传输构建传输映射，提升语音转换效果及安全性。**

- **链接: [https://arxiv.org/pdf/2505.04382v4](https://arxiv.org/pdf/2505.04382v4)**

> **作者:** Anton Selitskiy; Maitreya Kocharekar
>
> **备注:** 5 pages, 1 figure, 7 table
>
> **摘要:** We propose kDOT, a discrete optimal transport (OT) framework for voice conversion (VC) operating in a pretrained speech embedding space. In contrast to the averaging strategies used in kNN-VC and SinkVC, and the independence assumption adopted in MKL, our method employs the barycentric projection of the discrete OT plan to construct a transport map between source and target speaker embedding distributions. We conduct a comprehensive ablation study over the number of transported embeddings and systematically analyze the impact of source and target utterance duration. Experiments on LibriSpeech demonstrate that OT with barycentric projection consistently improves distribution alignment and often outperforms averaging-based approaches in terms of WER, MOS, and FAD. Furthermore, we show that applying discrete OT as a post-processing step can transform spoofed speech into samples that are misclassified as bona fide by a state-of-the-art spoofing detector. This demonstrates the strong domain adaptation capability of OT in embedding space, while also revealing important security implications for spoof detection systems.
>
---
#### [replaced 008] OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出OmniCustom，解决同步音视频定制任务，通过联合生成模型同时保持视频身份和音频音色，实现零样本音视频生成。**

- **链接: [https://arxiv.org/pdf/2602.12304v2](https://arxiv.org/pdf/2602.12304v2)**

> **作者:** Maomao Li; Zhen Li; Kaipeng Zhang; Guosheng Yin; Zhifeng Li; Dong Xu
>
> **备注:** code: https://github.com/OmniCustom-project/OmniCustom
>
> **摘要:** Existing mainstream video customization methods focus on generating identity-consistent videos based on given reference images and textual prompts. Benefiting from the rapid advancement of joint audio-video generation, this paper proposes a more compelling new task: sync audio-video customization, which aims to synchronously customize both video identity and audio timbre. Specifically, given a reference image $I^{r}$ and a reference audio $A^{r}$, this novel task requires generating videos that maintain the identity of the reference image while imitating the timbre of the reference audio, with spoken content freely specifiable through user-provided textual prompts. To this end, we propose OmniCustom, a powerful DiT-based audio-video customization framework that can synthesize a video following reference image identity, audio timbre, and text prompts all at once in a zero-shot manner. Our framework is built on three key contributions. First, identity and audio timbre control are achieved through separate reference identity and audio LoRA modules that operate through self-attention layers within the base audio-video generation model. Second, we introduce a contrastive learning objective alongside the standard flow matching objective. It uses predicted flows conditioned on reference inputs as positive examples and those without reference conditions as negative examples, thereby enhancing the model ability to preserve identity and timbre. Third, we train OmniCustom on our constructed large-scale, high-quality audio-visual human dataset. Extensive experiments demonstrate that OmniCustom outperforms existing methods in generating audio-video content with consistent identity and timbre fidelity. Project page: https://omnicustom-project.github.io/page/.
>
---
#### [replaced 009] Sonic4D: Spatial Audio Generation for Immersive 4D Scene Exploration
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出Sonic4D，解决4D场景中缺乏空间音频的问题，通过三阶段方法生成与视觉一致的沉浸式空间音频。**

- **链接: [https://arxiv.org/pdf/2506.15759v2](https://arxiv.org/pdf/2506.15759v2)**

> **作者:** Siyi Xie; Hanxin Zhu; Tianyu He; Xin Li; Zhibo Chen
>
> **备注:** 17 pages, 7 figures. Project page: https://x-drunker.github.io/Sonic4D-project-page/
>
> **摘要:** Recent advancements in 4D generation have demonstrated its remarkable capability in synthesizing photorealistic renderings of dynamic 3D scenes. However, despite achieving impressive visual performance, almost all existing methods overlook the generation of spatial audio aligned with the corresponding 4D scenes, posing a significant limitation to truly immersive audiovisual experiences. To mitigate this issue, we propose Sonic4D, a novel framework that enables spatial audio generation for immersive exploration of 4D scenes. Specifically, our method is composed of three stages: 1) To capture both the dynamic visual content and raw auditory information from a monocular video, we first employ pre-trained expert models to generate the 4D scene and its corresponding monaural audio. 2) Subsequently, to transform the monaural audio into spatial audio, we localize and track the sound sources within the 4D scene, where their 3D spatial coordinates at different timestamps are estimated via a pixel-level visual grounding strategy. 3) Based on the estimated sound source locations, we further synthesize plausible spatial audio that varies across different viewpoints and timestamps using physics-based simulation. Extensive experiments have demonstrated that our proposed method generates realistic spatial audio consistent with the synthesized 4D scene in a training-free manner, significantly enhancing the immersive experience for users. Generated audio and video examples are available at https://x-drunker.github.io/Sonic4D-project-page.
>
---
