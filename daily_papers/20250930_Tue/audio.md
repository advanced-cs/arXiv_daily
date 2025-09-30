# 音频 cs.SD;  eess.SP

- **最新发布 50 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于多标签音频分类任务，解决全局池化导致的信息瓶颈问题。提出二值原型探针方法，提升嵌入质量评估效果。**

- **链接: [http://arxiv.org/pdf/2509.24901v1](http://arxiv.org/pdf/2509.24901v1)**

> **作者:** Lukas Rauch; René Heinrich; Houtan Ghaffari; Lukas Miklautz; Ilyass Moummad; Bernhard Sick; Christoph Scholz
>
> **备注:** Currently under review @ICLR2026
>
> **摘要:** Although probing frozen models has become a standard evaluation paradigm, self-supervised learning in audio defaults to fine-tuning. A key reason is that global pooling creates an information bottleneck causing linear probes to misrepresent the embedding quality: The $\texttt{cls}$-token discards crucial token information about dispersed, localized events in multi-label audio. This weakness is rooted in the mismatch between the pretraining objective (operating globally) and the downstream task (localized events). Across a comprehensive benchmark of 13 datasets and 6 spectrogram-based encoders, we first investigate the global pooling bottleneck. We then introduce binarized prototypical probes: a lightweight and simple pooling method that learns prototypes to perform class-wise information aggregation. Despite its simplicity, our method notably outperforms linear and attentive probing. Our work establishes probing as a competitive and efficient paradigm for evaluating audio SSL models, challenging the reliance on costly fine-tuning.
>
---
#### [new 002] Emotional Styles Hide in Deep Speaker Embeddings: Disentangle Deep Speaker Embeddings for Speaker Clustering
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于说话人聚类任务，旨在解决情感表达影响说话人嵌入效果的问题。提出DTG-VAE方法，在VAE框架中分离情感因素，提升聚类性能。**

- **链接: [http://arxiv.org/pdf/2509.23358v1](http://arxiv.org/pdf/2509.23358v1)**

> **作者:** Chaohao Lin; Xu Zheng; Kaida Wu; Peihao Xiang; Ou Bai
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Speaker clustering is the task of identifying the unique speakers in a set of audio recordings (each belonging to exactly one speaker) without knowing who and how many speakers are present in the entire data, which is essential for speaker diarization processes. Recently, off-the-shelf deep speaker embedding models have been leveraged to capture speaker characteristics. However, speeches containing emotional expressions pose significant challenges, often affecting the accuracy of speaker embeddings and leading to a decline in speaker clustering performance. To tackle this problem, we propose DTG-VAE, a novel disentanglement method that enhances clustering within a Variational Autoencoder (VAE) framework. This study reveals a direct link between emotional states and the effectiveness of deep speaker embeddings. As demonstrated in our experiments, DTG-VAE extracts more robust speaker embeddings and significantly enhances speaker clustering performance.
>
---
#### [new 003] An Agent-Based Framework for Automated Higher-Voice Harmony Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决算法作曲中的和声生成问题。通过多智能体系统协作，实现从乐谱解析到音频合成的自动化和声创作。**

- **链接: [http://arxiv.org/pdf/2509.24463v1](http://arxiv.org/pdf/2509.24463v1)**

> **作者:** Nia D'Souza Ganapathy; Arul Selvamani Shaja
>
> **摘要:** The generation of musically coherent and aesthetically pleasing harmony remains a significant challenge in the field of algorithmic composition. This paper introduces an innovative Agentic AI-enabled Higher Harmony Music Generator, a multi-agent system designed to create harmony in a collaborative and modular fashion. Our framework comprises four specialized agents: a Music-Ingestion Agent for parsing and standardizing input musical scores; a Chord-Knowledge Agent, powered by a Chord-Former (Transformer model), to interpret and provide the constituent notes of complex chord symbols; a Harmony-Generation Agent, which utilizes a Harmony-GPT and a Rhythm-Net (RNN) to compose a melodically and rhythmically complementary harmony line; and an Audio-Production Agent that employs a GAN-based Symbolic-to-Audio Synthesizer to render the final symbolic output into high-fidelity audio. By delegating specific tasks to specialized agents, our system effectively mimics the collaborative process of human musicians. This modular, agent-based approach allows for robust data processing, deep theoretical understanding, creative composition, and realistic audio synthesis, culminating in a system capable of generating sophisticated and contextually appropriate higher-voice harmonies for given melodies.
>
---
#### [new 004] The Shape of Surprise: Structured Uncertainty and Co-Creativity in AI Music Tools
- **分类: cs.SD**

- **简介: 该论文属于AI音乐生成领域，探讨如何通过结构化不确定性实现创意协作。解决随机性与连贯性之间的矛盾，分析六种系统的设计模式。**

- **链接: [http://arxiv.org/pdf/2509.25028v1](http://arxiv.org/pdf/2509.25028v1)**

> **作者:** Eric Browne
>
> **备注:** 12 Pages, 2 figures, 1 table, The AI Music Creativity Conference (AIMC), 2025
>
> **摘要:** Randomness plays a pivotal yet paradoxical role in computational music creativity: it can spark novelty, but unchecked chance risks incoherence. This paper presents a thematic review of contemporary AI music systems, examining how designers incorporate randomness and uncertainty into creative practice. I draw on the concept of structured uncertainty to analyse how stochastic processes are constrained within musical and interactive frameworks. Through a comparative analysis of six systems - Musika (Pasini and Schl\"uter, 2022), MIDI-DDSP (Wu et al., 2021), Melody RNN (Magenta Project), RAVE (Caillon and Esling, 2021), Wekinator (Fiebrink and Cook, 2010), and Somax 2 (Borg, 2019) - we identify recurring design patterns that support musical coherence, user control, and co-creativity. To my knowledge, this is the first thematic review examining randomness in AI music through structured uncertainty, offering practical insights for designers and artists aiming to support expressive, collaborative, or improvisational interactions.
>
---
#### [new 005] Enhanced Automatic Drum Transcription via Drum Stem Source Separation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于自动鼓乐转录任务，旨在提升转录的 realism。通过结合源分离工具，扩展鼓类至七种，并估计MIDI速度值，以提高转录质量。**

- **链接: [http://arxiv.org/pdf/2509.24853v1](http://arxiv.org/pdf/2509.24853v1)**

> **作者:** Xavier Riley; Simon Dixon
>
> **摘要:** Automatic Drum Transcription (ADT) remains a challenging task in MIR but recent advances allow accurate transcription of drum kits with up 5 classes - kick, snare, hi-hats, toms and cymbals - via the ADTOF package. In addition, several drum kit \emph{stem} separation models in the open source community support separation for more than 6 stem classes, including distinct crash and ride cymbals. In this work we explore the benefits of combining these tools to improve the realism of drum transcriptions. We describe a simple post-processing step which expands the transcription output from five to seven classes and furthermore, we are able to estimate MIDI velocity values based on the separated stems. Our solution achieves strong performance when assessed against a baseline of 8-class drum transcription and produces realistic MIDI transcriptions suitable for MIR or music production tasks.
>
---
#### [new 006] Disentangling Score Content and Performance Style for Joint Piano Rendering and Transcription
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于音乐信息检索领域，解决EPR与APT任务的协同建模问题。通过分离音符内容与风格表示，提出统一框架实现高效渲染与转录。**

- **链接: [http://arxiv.org/pdf/2509.23878v1](http://arxiv.org/pdf/2509.23878v1)**

> **作者:** Wei Zeng; Junchuan Zhao; Ye Wang
>
> **备注:** 30 pages, 13 figures
>
> **摘要:** Expressive performance rendering (EPR) and automatic piano transcription (APT) are fundamental yet inverse tasks in music information retrieval: EPR generates expressive performances from symbolic scores, while APT recovers scores from performances. Despite their dual nature, prior work has addressed them independently. In this paper we propose a unified framework that jointly models EPR and APT by disentangling note-level score content and global performance style representations from both paired and unpaired data. Our framework is built on a transformer-based sequence-to-sequence architecture and is trained using only sequence-aligned data, without requiring fine-grained note-level alignment. To automate the rendering process while ensuring stylistic compatibility with the score, we introduce an independent diffusion-based performance style recommendation module that generates style embeddings directly from score content. This modular component supports both style transfer and flexible rendering across a range of expressive styles. Experimental results from both objective and subjective evaluations demonstrate that our framework achieves competitive performance on EPR and APT tasks, while enabling effective content-style disentanglement, reliable style transfer, and stylistically appropriate rendering. Demos are available at https://jointpianist.github.io/epr-apt/
>
---
#### [new 007] DiaMoE-TTS: A Unified IPA-Based Dialect TTS Framework with Mixture-of-Experts and Parameter-Efficient Zero-Shot Adaptation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，解决方言TTS数据少、发音不一致的问题。提出DiaMoE-TTS框架，结合IPA标准化和专家混合模型，实现高效零样本迁移。**

- **链接: [http://arxiv.org/pdf/2509.22727v1](http://arxiv.org/pdf/2509.22727v1)**

> **作者:** Ziqi Chen; Gongyu Chen; Yihua Wang; Chaofan Ding; Zihao chen; Wei-Qiang Zhang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Dialect speech embodies rich cultural and linguistic diversity, yet building text-to-speech (TTS) systems for dialects remains challenging due to scarce data, inconsistent orthographies, and complex phonetic variation. To address these issues, we present DiaMoE-TTS, a unified IPA-based framework that standardizes phonetic representations and resolves grapheme-to-phoneme ambiguities. Built upon the F5-TTS architecture, the system introduces a dialect-aware Mixture-of-Experts (MoE) to model phonological differences and employs parameter-efficient adaptation with Low-Rank Adaptors (LoRA) and Conditioning Adapters for rapid transfer to new dialects. Unlike approaches dependent on large-scale or proprietary resources, DiaMoE-TTS enables scalable, open-data-driven synthesis. Experiments demonstrate natural and expressive speech generation, achieving zero-shot performance on unseen dialects and specialized domains such as Peking Opera with only a few hours of data.
>
---
#### [new 008] VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，解决传统TTS中离散令牌与连续信号的权衡问题。提出无需分词器的VoxCPM模型，通过层次化语义-声学建模实现高质量语音生成。**

- **链接: [http://arxiv.org/pdf/2509.24650v1](http://arxiv.org/pdf/2509.24650v1)**

> **作者:** Yixuan Zhou; Guoyang Zeng; Xin Liu; Xiang Li; Renjie Yu; Ziyang Wang; Runchuan Ye; Weiyue Sun; Jiancheng Gui; Kehan Li; Zhiyong Wu; Zhiyuan Liu
>
> **备注:** Technical Report
>
> **摘要:** Generative models for speech synthesis face a fundamental trade-off: discrete tokens ensure stability but sacrifice expressivity, while continuous signals retain acoustic richness but suffer from error accumulation due to task entanglement. This challenge has driven the field towards multi-stage pipelines that rely on pre-trained speech tokenizers, but these create a semantic-acoustic divide, limiting holistic and expressive speech generation. We resolve these dilemma through hierarchical semantic-acoustic modeling with semi-discrete residual representations and present a novel tokenizer-free TTS model VoxCPM. Our framework introduces a differentiable quantization bottleneck that induces natural specialization: a Text-Semantic Language Model (TSLM) generates semantic-prosodic plans, while a Residual Acoustic Model (RALM) recovers fine-grained acoustic details. This hierarchical semantic-acoustic representation guides a local diffusion-based decoder to generate high-fidelity speech latents. Critically, the entire architecture is trained end-to-end under a simple diffusion objective, eliminating dependency on external speech tokenizers. Trained on a massive 1.8 million hours of bilingual corpus, our VoxCPM-0.5B model achieves state-of-the-art zero-shot TTS performance among open-source systems, demonstrating that our approach delivers expressive and stable synthesis. Besides, VoxCPM shows the capability to comprehend text to infer and generate appropriate prosody and style, delivering speech with context-aware expressiveness and natural flow. To facilitate community-driven research and development, VoxCPM is publicly accessible under Apache 2.0.
>
---
#### [new 009] From Sound to Setting: AI-Based Equalizer Parameter Prediction for Piano Tone Replication
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频处理任务，旨在通过AI预测钢琴音色的均衡器参数，解决自动音色匹配问题。工作包括构建数据集并评估回归与神经网络模型。**

- **链接: [http://arxiv.org/pdf/2509.24404v1](http://arxiv.org/pdf/2509.24404v1)**

> **作者:** Song-Ze Yu
>
> **备注:** Undergraduate project technical preprint. 4 pages, 6 figures. Code & data: https://github.com/vaclisinc/Vaclis_Tone_Replication Primary: cs.SD, Secondary: cs.LG
>
> **摘要:** This project presents an AI-based system for tone replication in music production, focusing on predicting EQ parameter settings directly from audio features. Unlike traditional audio-to-audio methods, our approach outputs interpretable parameter values (e.g., EQ band gains) that musicians can further adjust in their workflow. Using a dataset of piano recordings with systematically varied EQ settings, we evaluate both regression and neural network models. The neural network achieves a mean squared error of 0.0216 on multi-band tasks. The system enables practical, flexible, and automated tone matching for music producers and lays the foundation for extensions to more complex audio effects.
>
---
#### [new 010] An Efficient Transfer Learning Method Based on Adapter with Local Attributes for Speech Emotion Recognition
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，解决数据不足和跨场景适应性差的问题。提出带有局部属性的适配器，提升迁移学习效率。**

- **链接: [http://arxiv.org/pdf/2509.23795v1](http://arxiv.org/pdf/2509.23795v1)**

> **作者:** Haoyu Song; Ian McLoughlin; Qing Gu; Nan Jiang; Yan Song
>
> **摘要:** Existing speech emotion recognition (SER) methods commonly suffer from the lack of high-quality large-scale corpus, partly due to the complex, psychological nature of emotion which makes accurate labeling difficult and time consuming. Recently, transfer learning based methods that exploit the encoders pretrained on large-scale speech corpus (e.g., Wav2Vec2.0 and HuBERT) have shown strong potential for downstream SER tasks. However, task-specific fine-tuning remains necessary for various conversational scenarios of different topics, speakers and languages to achieve satisfactory performance. It generally requires costly encoder retraining for individual SER tasks. To address this issue, we propose to train an adapter with local attributes for efficient transfer learning. Specifically, a weighted average pooling-Transformer (WAP-Transformer) is proposed as a lightweight backbone to enrich the frame-level representation. An adapter with teacher-student branches is exploited for task-agnostic transfer learning, where the student branch is jointly optimized via mask prediction and self-distillation objectives, and the teacher branch is obtained online from the student via exponential moving average (EMA). Meanwhile, local attributes are learned from the teacher branch via unsupervised clustering, which aims to act as a universal model that provides additional semantic-rich supervisions. A statistical attentive pooling (SAP) module is proposed to obtain utterance representation for fine-tuning. To evaluate the effectiveness of the proposed adapter with local attributes, extensive experiments have been conducted on IEMOCAP. Superior performance has been reported, compared to the previous state-of-the-art methods in similar settings.
>
---
#### [new 011] UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities
- **分类: cs.SD**

- **简介: 该论文属于音频生成任务，旨在统一处理时间对齐与非时间对齐任务。提出UniFlow-Audio框架，通过流匹配和双融合机制实现高效多模态生成。**

- **链接: [http://arxiv.org/pdf/2509.24391v1](http://arxiv.org/pdf/2509.24391v1)**

> **作者:** Xuenan Xu; Jiahao Mei; Zihao Zheng; Ye Tao; Zeyu Xie; Yaoyun Zhang; Haohe Liu; Yuning Wu; Ming Yan; Wen Wu; Chao Zhang; Mengyue Wu
>
> **备注:** Project page: https://wsntxxn.github.io/uniflow_audio
>
> **摘要:** Audio generation, including speech, music and sound effects, has advanced rapidly in recent years. These tasks can be divided into two categories: time-aligned (TA) tasks, where each input unit corresponds to a specific segment of the output audio (e.g., phonemes aligned with frames in speech synthesis); and non-time-aligned (NTA) tasks, where such alignment is not available. Since modeling paradigms for the two types are typically different, research on different audio generation tasks has traditionally followed separate trajectories. However, audio is not inherently divided into such categories, making a unified model a natural and necessary goal for general audio generation. Previous unified audio generation works have adopted autoregressive architectures, while unified non-autoregressive approaches remain largely unexplored. In this work, we propose UniFlow-Audio, a universal audio generation framework based on flow matching. We propose a dual-fusion mechanism that temporally aligns audio latents with TA features and integrates NTA features via cross-attention in each model block. Task-balanced data sampling is employed to maintain strong performance across both TA and NTA tasks. UniFlow-Audio supports omni-modalities, including text, audio, and video. By leveraging the advantage of multi-task learning and the generative modeling capabilities of flow matching, UniFlow-Audio achieves strong results across 7 tasks using fewer than 8K hours of public training data and under 1B trainable parameters. Even the small variant with only ~200M trainable parameters shows competitive performance, highlighting UniFlow-Audio as a potential non-auto-regressive foundation model for audio generation. Code and models will be available at https://wsntxxn.github.io/uniflow_audio.
>
---
#### [new 012] Introducing Multimodal Paradigm for Learning Sleep Staging PSG via General-Purpose Model
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于睡眠分期任务，旨在解决传统方法依赖专业数据和缺乏直观性的问题。通过将PSG信号转为图像并微调多模态模型，提升分期准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.22810v1](http://arxiv.org/pdf/2509.22810v1)**

> **作者:** Jianheng Zhou; Chenyu Liu; Jinan Zhou; Yi Ding; Yang Liu; Haoran Luo; Ziyu Jia; Xinliang Zhou
>
> **摘要:** Sleep staging is essential for diagnosing sleep disorders and assessing neurological health. Existing automatic methods typically extract features from complex polysomnography (PSG) signals and train domain-specific models, which often lack intuitiveness and require large, specialized datasets. To overcome these limitations, we introduce a new paradigm for sleep staging that leverages large multimodal general-purpose models to emulate clinical diagnostic practices. Specifically, we convert raw one-dimensional PSG time-series into intuitive two-dimensional waveform images and then fine-tune a multimodal large model to learn from these representations. Experiments on three public datasets (ISRUC, MASS, SHHS) demonstrate that our approach enables general-purpose models, without prior exposure to sleep data, to acquire robust staging capabilities. Moreover, explanation analysis reveals our model learned to mimic the visual diagnostic workflow of human experts for sleep staging by PSG images. The proposed method consistently outperforms state-of-the-art baselines in accuracy and robustness, highlighting its efficiency and practical value for medical applications. The code for the signal-to-image pipeline and the PSG image dataset will be released.
>
---
#### [new 013] VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐信息检索任务，旨在解决自动音乐转录中忽略演奏技巧的问题。提出VioPTT模型，同时转录音高和小提琴演奏技巧，并发布合成数据集MOSA-VPT。**

- **链接: [http://arxiv.org/pdf/2509.23759v1](http://arxiv.org/pdf/2509.23759v1)**

> **作者:** Ting-Kang Wang; Yueh-Po Peng; Li Su; Vincent K. M. Cheung
>
> **摘要:** While automatic music transcription is well-established in music information retrieval, most models are limited to transcribing pitch and timing information from audio, and thus omit crucial expressive and instrument-specific nuances. One example is playing technique on the violin, which affords its distinct palette of timbres for maximal emotional impact. Here, we propose \textbf{VioPTT} (Violin Playing Technique-aware Transcription), a lightweight, end-to-end model that directly transcribes violin playing technique in addition to pitch onset and offset. Furthermore, we release \textbf{MOSA-VPT}, a novel, high-quality synthetic violin playing technique dataset to circumvent the need for manually labeled annotations. Leveraging this dataset, our model demonstrated strong generalization to real-world note-level violin technique recordings in addition to achieving state-of-the-art transcription performance. To our knowledge, VioPTT is the first to jointly combine violin transcription and playing technique prediction within a unified framework.
>
---
#### [new 014] ABC-Eval: Benchmarking Large Language Models on Symbolic Music Understanding and Instruction Following
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到符号音乐理解任务，旨在评估大语言模型在ABC记谱法上的理解和指令遵循能力。研究提出ABC-Eval基准，测试模型在多种音乐任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.23350v1](http://arxiv.org/pdf/2509.23350v1)**

> **作者:** Jiahao Zhao; Yunjia Li; Wei Li; Kazuyoshi Yoshii
>
> **摘要:** As large language models continue to develop, the feasibility and significance of text-based symbolic music tasks have become increasingly prominent. While symbolic music has been widely used in generation tasks, LLM capabilities in understanding and reasoning about symbolic music remain largely underexplored. To address this gap, we propose ABC-Eval, the first open-source benchmark dedicated to the understanding and instruction-following capabilities in text-based ABC notation scores. It comprises 1,086 test samples spanning 10 sub-tasks, covering scenarios from basic musical syntax comprehension to complex sequence-level reasoning. Such a diverse scope poses substantial challenges to models' ability to handle symbolic music tasks. We evaluated seven state-of-the-art LLMs on ABC-Eval, and the results reveal notable limitations in existing models' symbolic music processing capabilities. Furthermore, the consistent performance of individual baselines across different sub-tasks supports the reliability of our benchmark.
>
---
#### [new 015] Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在检测音乐嵌入模型中的文化偏见。通过CAVs分析非音乐属性对流派表示的影响，提出后处理去偏策略，以减轻模型中的不公平表现。**

- **链接: [http://arxiv.org/pdf/2509.24482v1](http://arxiv.org/pdf/2509.24482v1)**

> **作者:** Roman B. Gebhardt; Arne Kuhle; Eylül Bektur
>
> **备注:** ISMIR 2025
>
> **摘要:** Music representation models are widely used for tasks such as tagging, retrieval, and music understanding. Yet, their potential to encode cultural bias remains underexplored. In this paper, we apply Concept Activation Vectors (CAVs) to investigate whether non-musical singer attributes - such as gender and language - influence genre representations in unintended ways. We analyze four state-of-the-art models (MERT, Whisper, MuQ, MuQ-MuLan) using the STraDa dataset, carefully balancing training sets to control for genre confounds. Our results reveal significant model-specific biases, aligning with disparities reported in MIR and music sociology. Furthermore, we propose a post-hoc debiasing strategy using concept vector manipulation, demonstrating its effectiveness in mitigating these biases. These findings highlight the need for bias-aware model design and show that conceptualized interpretability methods offer practical tools for diagnosing and mitigating representational bias in MIR.
>
---
#### [new 016] GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决吉他数据集稀缺与标注不足的问题。作者构建了GOAT数据集，包含高质量吉他音频与tablature标注，并提出数据增强策略，以促进吉他相关MIR研究。**

- **链接: [http://arxiv.org/pdf/2509.22655v1](http://arxiv.org/pdf/2509.22655v1)**

> **作者:** Jackson Loth; Pedro Sarmento; Saurjya Sarkar; Zixun Guo; Mathieu Barthet; Mark Sandler
>
> **备注:** To be published in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2025
>
> **摘要:** In recent years, the guitar has received increased attention from the music information retrieval (MIR) community driven by the challenges posed by its diverse playing techniques and sonic characteristics. Mainly fueled by deep learning approaches, progress has been limited by the scarcity and limited annotations of datasets. To address this, we present the Guitar On Audio and Tablatures (GOAT) dataset, comprising 5.9 hours of unique high-quality direct input audio recordings of electric guitars from a variety of different guitars and players. We also present an effective data augmentation strategy using guitar amplifiers which delivers near-unlimited tonal variety, of which we provide a starting 29.5 hours of audio. Each recording is annotated using guitar tablatures, a guitar-specific symbolic format supporting string and fret numbers, as well as numerous playing techniques. For this we utilise both the Guitar Pro format, a software for tablature playback and editing, and a text-like token encoding. Furthermore, we present competitive results using GOAT for MIDI transcription and preliminary results for a novel approach to automatic guitar tablature transcription. We hope that GOAT opens up the possibilities to train novel models on a wide variety of guitar-related MIR tasks, from synthesis to transcription to playing technique detection.
>
---
#### [new 017] MeanFlowSE: One-Step Generative Speech Enhancement via MeanFlow
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升噪声环境中语音的清晰度和可懂度。提出MeanFlowSE框架，通过单步生成实现高效实时处理，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.23299v1](http://arxiv.org/pdf/2509.23299v1)**

> **作者:** Yike Zhu; Boyi Kang; Ziqian Wang; Xingchen Li; Zihan Zhang; Wenjie Li; Longshuai Xiao; Wei Xue; Lei Xie
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech enhancement (SE) recovers clean speech from noisy signals and is vital for applications such as telecommunications and automatic speech recognition (ASR). While generative approaches achieve strong perceptual quality, they often rely on multi-step sampling (diffusion/flow-matching) or large language models, limiting real-time deployment. To mitigate these constraints, we present MeanFlowSE, a one-step generative SE framework. It adopts MeanFlow to predict an average-velocity field for one-step latent refinement and conditions the model on self-supervised learning (SSL) representations rather than VAE latents. This design accelerates inference and provides robust acoustic-semantic guidance during training. In the Interspeech 2020 DNS Challenge blind test set and simulated test set, MeanFlowSE attains state-of-the-art (SOTA) level perceptual quality and competitive intelligibility while significantly lowering both real-time factor (RTF) and model size compared with recent generative competitors, making it suitable for practical use. The code will be released upon publication at https://github.com/Hello3orld/MeanFlowSE.
>
---
#### [new 018] Text-Independent Speaker Identification Using Audio Looping With Margin Based Loss Functions
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升文本无关说话人识别的准确性。通过改进VGG16模型并使用ArcFace和CosFace损失函数，对比传统Softmax方法，验证了新方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.22838v1](http://arxiv.org/pdf/2509.22838v1)**

> **作者:** Elliot Q C Garcia; Nicéias Silva Vilela; Kátia Pires Nascimento do Sacramento; Tiago A. E. Ferreira
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Speaker identification has become a crucial component in various applications, including security systems, virtual assistants, and personalized user experiences. In this paper, we investigate the effectiveness of CosFace Loss and ArcFace Loss for text-independent speaker identification using a Convolutional Neural Network architecture based on the VGG16 model, modified to accommodate mel spectrogram inputs of variable sizes generated from the Voxceleb1 dataset. Our approach involves implementing both loss functions to analyze their effects on model accuracy and robustness, where the Softmax loss function was employed as a comparative baseline. Additionally, we examine how the sizes of mel spectrograms and their varying time lengths influence model performance. The experimental results demonstrate superior identification accuracy compared to traditional Softmax loss methods. Furthermore, we discuss the implications of these findings for future research.
>
---
#### [new 019] Prompt-aware classifier free guidance for diffusion models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于图像生成任务，解决扩散模型中固定引导尺度适应性差的问题，通过引入提示感知框架动态选择最佳引导尺度，提升生成质量与对齐度。**

- **链接: [http://arxiv.org/pdf/2509.22728v1](http://arxiv.org/pdf/2509.22728v1)**

> **作者:** Xuanhao Zhang; Chang Li
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Diffusion models have achieved remarkable progress in image and audio generation, largely due to Classifier-Free Guidance. However, the choice of guidance scale remains underexplored: a fixed scale often fails to generalize across prompts of varying complexity, leading to oversaturation or weak alignment. We address this gap by introducing a prompt-aware framework that predicts scale-dependent quality and selects the optimal guidance at inference. Specifically, we construct a large synthetic dataset by generating samples under multiple scales and scoring them with reliable evaluation metrics. A lightweight predictor, conditioned on semantic embeddings and linguistic complexity, estimates multi-metric quality curves and determines the best scale via a utility function with regularization. Experiments on MSCOCO~2014 and AudioCaps show consistent improvements over vanilla CFG, enhancing fidelity, alignment, and perceptual preference. This work demonstrates that prompt-aware scale selection provides an effective, training-free enhancement for pretrained diffusion backbones.
>
---
#### [new 020] AudioMoG: Guiding Audio Generation with Mixture-of-Guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于跨模态音频生成任务，旨在解决单一引导方法限制生成质量与多样性的问题。提出AudioMoG框架，融合多种引导原则，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2509.23727v1](http://arxiv.org/pdf/2509.23727v1)**

> **作者:** Junyou Wang; Zehua Chen; Binjie Yuan; Kaiwen Zheng; Chang Li; Yuxuan Jiang; Jun Zhu
>
> **摘要:** Guidance methods have demonstrated significant improvements in cross-modal audio generation, including text-to-audio (T2A) and video-to-audio (V2A) generation. The popularly adopted method, classifier-free guidance (CFG), steers generation by emphasizing condition alignment, enhancing fidelity but often at the cost of diversity. Recently, autoguidance (AG) has been explored for audio generation, encouraging the sampling to faithfully reconstruct the target distribution and showing increased diversity. Despite these advances, they usually rely on a single guiding principle, e.g., condition alignment in CFG or score accuracy in AG, leaving the full potential of guidance for audio generation untapped. In this work, we explore enriching the composition of the guidance method and present a mixture-of-guidance framework, AudioMoG. Within the design space, AudioMoG can exploit the complementary advantages of distinctive guiding principles by fulfilling their cumulative benefits. With a reduced form, AudioMoG can consider parallel complements or recover a single guiding principle, without sacrificing generality. We experimentally show that, given the same inference speed, AudioMoG approach consistently outperforms single guidance in T2A generation across sampling steps, concurrently showing advantages in V2A, text-to-music, and image generation. These results highlight a "free lunch" in current cross-modal audio generation systems: higher quality can be achieved through mixed guiding principles at the sampling stage without sacrificing inference efficiency. Demo samples are available at: https://audio-mog.github.io.
>
---
#### [new 021] YOLO-based Bearing Fault Diagnosis With Continuous Wavelet Transform
- **分类: eess.SP; cs.AI; cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于轴承故障诊断任务，旨在提高故障识别准确率。通过结合CWT与YOLO模型，将振动信号转换为时频图并进行分类，实现高精度故障检测与定位。**

- **链接: [http://arxiv.org/pdf/2509.03070v2](http://arxiv.org/pdf/2509.03070v2)**

> **作者:** Po-Heng Chou; Wei-Lung Mao; Ru-Ping Lin
>
> **备注:** 5 pages, 2 figures, 2 tables, submitted to IEEE Sensors Letters
>
> **摘要:** This letter proposes a YOLO-based framework for spatial bearing fault diagnosis using time-frequency spectrograms derived from continuous wavelet transform (CWT). One-dimensional vibration signals are first transformed into time-frequency spectrograms using Morlet wavelets to capture transient fault signatures. These spectrograms are then processed by YOLOv9, v10, and v11 models to classify fault types. Evaluated on three benchmark datasets, including Case Western Reserve University (CWRU), Paderborn University (PU), and Intelligent Maintenance System (IMS), the proposed CWT-YOLO pipeline achieves significantly higher accuracy and generalizability than the baseline MCNN-LSTM model. Notably, YOLOv11 reaches mAP scores of 99.4% (CWRU), 97.8% (PU), and 99.5% (IMS). In addition, its region-aware detection mechanism enables direct visualization of fault locations in spectrograms, offering a practical solution for condition monitoring in rotating machinery.
>
---
#### [new 022] Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决跨方法、说话人和环境的分布偏移问题。提出IB-CAAN模型，通过对抗对齐和信息瓶颈提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23618v1](http://arxiv.org/pdf/2509.23618v1)**

> **作者:** Pu Huang; Shouguang Wang; Siya Yao; Mengchu Zhou
>
> **摘要:** Neural speech synthesis techniques have enabled highly realistic speech deepfakes, posing major security risks. Speech deepfake detection is challenging due to distribution shifts across spoofing methods and variability in speakers, channels, and recording conditions. We explore learning shared discriminative features as a path to robust detection and propose Information Bottleneck enhanced Confidence-Aware Adversarial Network (IB-CAAN). Confidence-guided adversarial alignment adaptively suppresses attack-specific artifacts without erasing discriminative cues, while the information bottleneck removes nuisance variability to preserve transferable features. Experiments on ASVspoof 2019/2021, ASVspoof 5, and In-the-Wild demonstrate that IB-CAAN consistently outperforms baseline and achieves state-of-the-art performance on many benchmarks.
>
---
#### [new 023] Discovering "Words" in Music: Unsupervised Learning of Compositional Sparse Code for Symbolic Music
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音乐模式识别任务，旨在解决音乐中语义模糊的问题。通过无监督学习提取“音乐词”，构建稀疏编码模型，用于音乐结构分析与生成。**

- **链接: [http://arxiv.org/pdf/2509.24603v1](http://arxiv.org/pdf/2509.24603v1)**

> **作者:** Tianle Wang; Sirui Zhang; Xinyi Tong; Peiyang Yu; Jishang Chen; Liangke Zhao; Xinpu Gao; Yves Zhu; Tiezheng Ge; Bo Zheng; Duo Xu; Yang Liu; Xin Jin; Feng Yu; Songchun Zhu
>
> **摘要:** This paper presents an unsupervised machine learning algorithm that identifies recurring patterns -- referred to as ``music-words'' -- from symbolic music data. These patterns are fundamental to musical structure and reflect the cognitive processes involved in composition. However, extracting these patterns remains challenging because of the inherent semantic ambiguity in musical interpretation. We formulate the task of music-word discovery as a statistical optimization problem and propose a two-stage Expectation-Maximization (EM)-based learning framework: 1. Developing a music-word dictionary; 2. Reconstructing the music data. When evaluated against human expert annotations, the algorithm achieved an Intersection over Union (IoU) score of 0.61. Our findings indicate that minimizing code length effectively addresses semantic ambiguity, suggesting that human optimization of encoding systems shapes musical semantics. This approach enables computers to extract ``basic building blocks'' from music data, facilitating structural analysis and sparse encoding. The method has two primary applications. First, in AI music, it supports downstream tasks such as music generation, classification, style transfer, and improvisation. Second, in musicology, it provides a tool for analyzing compositional patterns and offers insights into the principle of minimal encoding across diverse musical styles and composers.
>
---
#### [new 024] WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频表示学习任务，解决通用音频建模问题。提出WavJEPA模型，通过语义学习提升音频基础模型性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23238v1](http://arxiv.org/pdf/2509.23238v1)**

> **作者:** Goksenin Yuksel; Pierre Guetschel; Michael Tangermann; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.
>
---
#### [new 025] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频-视觉语音分离任务，旨在提升分离质量同时降低计算成本。提出Dolphin方法，通过轻量编码器和多尺度注意力机制实现高效分离。**

- **链接: [http://arxiv.org/pdf/2509.23610v1](http://arxiv.org/pdf/2509.23610v1)**

> **作者:** Kai Li; Kejun Gao; Xiaolin Hu
>
> **备注:** Technical Report
>
> **摘要:** Audio-visual speech separation (AVSS) methods leverage visual cues to extract target speech and have demonstrated strong separation quality in noisy acoustic environments. However, these methods usually involve a large number of parameters and require high computational cost, which is unacceptable in many applications where speech separation serves as only a preprocessing step for further speech processing. To address this issue, we propose an efficient AVSS method, named Dolphin. For visual feature extraction, we develop DP-LipCoder, a dual-path lightweight video encoder that transforms lip-motion into discrete audio-aligned semantic tokens. For audio separation, we construct a lightweight encoder-decoder separator, in which each layer incorporates a global-local attention (GLA) block to efficiently capture multi-scale dependencies. Experiments on three benchmark datasets showed that Dolphin not only surpassed the current state-of-the-art (SOTA) model in separation quality but also achieved remarkable improvements in efficiency: over 50% fewer parameters, more than 2.4x reduction in MACs, and over 6x faster GPU inference speed. These results indicate that Dolphin offers a practical and deployable solution for high-performance AVSS in real-world scenarios. Our code and demo page are publicly available at http://cslikai.cn/Dolphin/.
>
---
#### [new 026] When Audio Generators Become Good Listeners: Generative Features for Understanding Tasks
- **分类: cs.SD; 68Txx; I.2**

- **简介: 该论文属于音频理解任务，旨在解决传统特征丢失细节的问题。通过融合生成特征与判别特征，提升音频分类、标注和描述等任务性能。**

- **链接: [http://arxiv.org/pdf/2509.24635v1](http://arxiv.org/pdf/2509.24635v1)**

> **作者:** Zeyu Xie; Chenxing Li; Xuenan Xu; Mengyue Wu; Wenfu Wang; Ruibo Fu; Meng Yu; Dong Yu; Yuexian Zou
>
> **摘要:** This work pioneers the utilization of generative features in enhancing audio understanding. Unlike conventional discriminative features that directly optimize posterior and thus emphasize semantic abstraction while losing fine grained details, audio generation models inherently encode both spatiotemporal perception (capturing local acoustic texture across time and frequency) and semantic prior (knowing what to generate). It motivates us to explore the bridge of these complementary strengths. We provide a systematic investigation of their differences and complementary relationships, and ultimately propose an effective fusion strategy. Experiments across multiple tasks, including sound event classification, tagging, and particularly the fine grained task of audio captioning, demonstrate consistent performance gains. Beyond empirical improvements, this work more importantly introduces a new perspective on audio representation learning, highlighting that generative discriminative complementarity can provide both detailed perception and semantic awareness for audio understanding.
>
---
#### [new 027] Efficient Speech Watermarking for Speech Synthesis via Progressive Knowledge Distillation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于语音水印任务，解决语音合成中的隐私与安全问题。提出PKDMark方法，通过知识蒸馏提升水印的效率和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.19812v1](http://arxiv.org/pdf/2509.19812v1)**

> **作者:** Yang Cui; Peter Pan; Lei He; Sheng Zhao
>
> **备注:** 6 pages of main text, 1 page of references, 2 figures, 2 tables, accepted at ASRU 2025
>
> **摘要:** With the rapid advancement of speech generative models, unauthorized voice cloning poses significant privacy and security risks. Speech watermarking offers a viable solution for tracing sources and preventing misuse. Current watermarking technologies fall mainly into two categories: DSP-based methods and deep learning-based methods. DSP-based methods are efficient but vulnerable to attacks, whereas deep learning-based methods offer robust protection at the expense of significantly higher computational cost. To improve the computational efficiency and enhance the robustness, we propose PKDMark, a lightweight deep learning-based speech watermarking method that leverages progressive knowledge distillation (PKD). Our approach proceeds in two stages: (1) training a high-performance teacher model using an invertible neural network-based architecture, and (2) transferring the teacher's capabilities to a compact student model through progressive knowledge distillation. This process reduces computational costs by 93.6% while maintaining high level of robust performance and imperceptibility. Experimental results demonstrate that our distilled model achieves an average detection F1 score of 99.6% with a PESQ of 4.30 in advanced distortions, enabling efficient speech watermarking for real-time speech synthesis applications.
>
---
#### [new 028] AudioRole: An Audio Dataset for Character Role-Playing in Large Language Models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于音频角色扮演任务，旨在解决语音与角色一致性问题。提出AudioRole数据集及评估框架，提升模型角色扮演能力。**

- **链接: [http://arxiv.org/pdf/2509.23435v1](http://arxiv.org/pdf/2509.23435v1)**

> **作者:** Wenyu Li; Xiaoqi Jiao; Yi Chang; Guangyan Zhang; Yiwen Guo
>
> **摘要:** The creation of high-quality multimodal datasets remains fundamental for advancing role-playing capabilities in large language models (LLMs). While existing works predominantly focus on text-based persona simulation, Audio Role-Playing (ARP) presents unique challenges due to the need for synchronized alignment of semantic content and vocal characteristics. To address this gap, we propose AudioRole, a meticulously curated dataset from 13 TV series spanning 1K+ hours with 1M+ character-grounded dialogues, providing synchronized audio-text pairs annotated with speaker identities and contextual metadata. In addition, to demonstrate the effectiveness of the dataset, we introduced ARP-Eval, a dual-aspect evaluation framework that assesses both response quality and role fidelity. Empirical validation showing GLM-4-Voice trained on AudioRole (which we called ARP-Model) achieve an average Acoustic Personalization score of 0.31, significantly outperforming the original GLM-4-voice and the more powerful model MiniCPM-O-2.6, which specifically supports role-playing in one-shot scenarios. The ARP-Model also achieves a Content Personalization score of 0.36, surpassing the untrained original model by about 38% and maintaining the same level as MiniCPM-O-2.6. AudioRole features dialogues from over 115 main characters, 6 trained ARP-Models that role-play different characters, and evaluation protocols. Together, they provide an essential resource for advancing audio-grounded role-playing research.
>
---
#### [new 029] Sparse Autoencoders Make Audio Foundation Models more Explainable
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决音频预训练模型表示不透明的问题。通过使用稀疏自编码器分析隐藏表示，提升对语音属性的解耦能力。**

- **链接: [http://arxiv.org/pdf/2509.24793v1](http://arxiv.org/pdf/2509.24793v1)**

> **作者:** Théo Mariotte; Martin Lebourdais; Antonio Almudévar; Marie Tahon; Alfonso Ortega; Nicolas Dugué
>
> **备注:** 5 pages, 5 figures, 1 table, submitted to ICASSP 2026
>
> **摘要:** Audio pretrained models are widely employed to solve various tasks in speech processing, sound event detection, or music information retrieval. However, the representations learned by these models are unclear, and their analysis mainly restricts to linear probing of the hidden representations. In this work, we explore the use of Sparse Autoencoders (SAEs) to analyze the hidden representations of pretrained models, focusing on a case study in singing technique classification. We first demonstrate that SAEs retain both information about the original representations and class labels, enabling their internal structure to provide insights into self-supervised learning systems. Furthermore, we show that SAEs enhance the disentanglement of vocal attributes, establishing them as an effective tool for identifying the underlying factors encoded in the representations.
>
---
#### [new 030] MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech
- **分类: cs.SD; cs.AI; cs.CL; cs.CV; cs.MM**

- **简介: 该论文属于多模态语音生成任务，旨在解决长时序、个性化语音生成问题。通过统一架构和高效训练策略，实现自然、连贯的语音输出。**

- **链接: [http://arxiv.org/pdf/2509.25131v1](http://arxiv.org/pdf/2509.25131v1)**

> **作者:** Chengyao Wang; Zhisheng Zhong; Bohao Peng; Senqiao Yang; Yuqi Liu; Haokun Gui; Bin Xia; Jingyao Li; Bei Yu; Jiaya Jia
>
> **备注:** Code is available at https://github.com/dvlab-research/MGM-Omni
>
> **摘要:** We present MGM-Omni, a unified Omni LLM for omni-modal understanding and expressive, long-horizon speech generation. Unlike cascaded pipelines that isolate speech synthesis, MGM-Omni adopts a "brain-mouth" design with a dual-track, token-based architecture that cleanly decouples multimodal reasoning from real-time speech generation. This design enables efficient cross-modal interaction and low-latency, streaming speech generation. For understanding, a unified training strategy coupled with a dual audio encoder design enables long-form audio perception across diverse acoustic conditions. For generation, a chunk-based parallel decoding scheme narrows the text speech token-rate gap, accelerating inference and supporting streaming zero-shot voice cloning with stable timbre over extended durations. Compared to concurrent work, MGM-Omni achieves these capabilities with markedly data-efficient training. Extensive experiments demonstrate that MGM-Omni outperforms existing open source models in preserving timbre identity across extended sequences, producing natural and context-aware speech, and achieving superior long-form audio and omnimodal understanding. MGM-Omni establishes an efficient, end-to-end paradigm for omnimodal understanding and controllable, personalised long-horizon speech generation.
>
---
#### [new 031] XGC-AVis: Towards Audio-Visual Content Understanding with a Multi-Agent Collaborative System
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于多模态内容理解任务，旨在解决音频视频同步与关键片段检索问题。提出XGC-AVis框架和XGC-AVQuiz基准，提升模型在真实与AI生成场景下的理解能力。**

- **链接: [http://arxiv.org/pdf/2509.23251v1](http://arxiv.org/pdf/2509.23251v1)**

> **作者:** Yuqin Cao; Xiongkuo Min; Yixuan Gao; Wei Sun; Zicheng Zhang; Jinliang Han; Guangtao Zhai
>
> **摘要:** In this paper, we propose XGC-AVis, a multi-agent framework that enhances the audio-video temporal alignment capabilities of multimodal large models (MLLMs) and improves the efficiency of retrieving key video segments through 4 stages: perception, planning, execution, and reflection. We further introduce XGC-AVQuiz, the first benchmark aimed at comprehensively assessing MLLMs' understanding capabilities in both real-world and AI-generated scenarios. XGC-AVQuiz consists of 2,685 question-answer pairs across 20 tasks, with two key innovations: 1) AIGC Scenario Expansion: The benchmark includes 2,232 videos, comprising 1,102 professionally generated content (PGC), 753 user-generated content (UGC), and 377 AI-generated content (AIGC). These videos cover 10 major domains and 53 fine-grained categories. 2) Quality Perception Dimension: Beyond conventional tasks such as recognition, localization, and reasoning, we introduce a novel quality perception dimension. This requires MLLMs to integrate low-level sensory capabilities with high-level semantic understanding to assess audio-visual quality, synchronization, and coherence. Experimental results on XGC-AVQuiz demonstrate that current MLLMs struggle with quality perception and temporal alignment tasks. XGC-AVis improves these capabilities without requiring additional training, as validated on two benchmarks.
>
---
#### [new 032] Predictability and Statistical Memory in Classical Sonatas and Quartets
- **分类: physics.soc-ph; cs.SD**

- **简介: 该论文属于音乐分析任务，旨在研究古典奏鸣曲和四重奏的统计依赖性。通过高阶马尔可夫模型等方法，比较不同作曲家的音乐预测性与结构差异。**

- **链接: [http://arxiv.org/pdf/2509.24172v1](http://arxiv.org/pdf/2509.24172v1)**

> **作者:** Linus Chen-Plotkin; Suman S. Kulkarni; Dani S. Bassett
>
> **备注:** 34 pages, 7 figures
>
> **摘要:** Statistical models and information theory have provided a useful set of tools for studying music from a quantitative perspective. These approaches have been employed to generate compositions, analyze structural patterns, and model cognitive processes that underlie musical perception. A common framework used in such studies is a Markov chain model, which models the probability of a musical event -- such as a note, chord, or rhythm -- based on a sequence of preceding events. While many studies focus on first-order models, relatively few have used more complex models to systematically compare across composers and compositional forms. In this study, we examine statistical dependencies in classical sonatas and quartets using higher-order Markov chains fit to sequences of top notes. Our data set of 605 MIDI files comprises piano sonatas and string quartets by Mozart, Haydn, Beethoven, and Schubert, from which we analyze sequences of top notes. We probe statistical dependencies using three distinct methods: Markov chain fits, time-delayed mutual information, and mixture transition distribution analysis. We find that, in general, the statistical dependencies in Mozart's music notably differ from that of the other three composers. Markov chain models of higher order provide significantly better fits than low-order models for Beethoven, Haydn, and Schubert, but not for Mozart. At the same time, we observe nuances across compositional forms and composers: for example, in the string quartets, certain metrics yield comparable results for Mozart and Beethoven. Broadly, our study extends the analysis of statistical dependencies in music, and highlights systematic distinctions in the predictability of sonatas and quartets from different classical composers. These findings motivate future work comparing across composers for other musical forms, or in other eras, cultures, or musical traditions.
>
---
#### [new 033] Learning What To Hear: Boosting Sound-Source Association For Robust Audiovisual Instance Segmentation
- **分类: eess.AS; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于音频视觉实例分割任务，解决视觉偏差问题。通过音频中心查询生成和显式计数监督，提升分割准确性。**

- **链接: [http://arxiv.org/pdf/2509.22740v1](http://arxiv.org/pdf/2509.22740v1)**

> **作者:** Jinbae Seo; Hyeongjun Kwon; Kwonyoung Kim; Jiyoung Lee; Kwanghoon Sohn
>
> **摘要:** Audiovisual instance segmentation (AVIS) requires accurately localizing and tracking sounding objects throughout video sequences. Existing methods suffer from visual bias stemming from two fundamental issues: uniform additive fusion prevents queries from specializing to different sound sources, while visual-only training objectives allow queries to converge to arbitrary salient objects. We propose Audio-Centric Query Generation using cross-attention, enabling each query to selectively attend to distinct sound sources and carry sound-specific priors into visual decoding. Additionally, we introduce Sound-Aware Ordinal Counting (SAOC) loss that explicitly supervises sounding object numbers through ordinal regression with monotonic consistency constraints, preventing visual-only convergence during training. Experiments on AVISeg benchmark demonstrate consistent improvements: +1.64 mAP, +0.6 HOTA, and +2.06 FSLA, validating that query specialization and explicit counting supervision are crucial for accurate audiovisual instance segmentation.
>
---
#### [new 034] VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.SD**

- **简介: 该论文属于视频条件下的声音和语音生成任务，旨在统一视频到声音（V2S）和视觉文本到语音（VisualTTS）任务。通过联合学习框架VSSFlow解决两者分离处理的问题。**

- **链接: [http://arxiv.org/pdf/2509.24773v1](http://arxiv.org/pdf/2509.24773v1)**

> **作者:** Xin Cheng; Yuyue Wang; Xihua Wang; Yihan Wu; Kaisi Guan; Yijing Chen; Peng Zhang; Xiaojiang Liu; Meng Cao; Ruihua Song
>
> **备注:** Paper Under Review
>
> **摘要:** Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models.
>
---
#### [new 035] S$^3$F-Net: A Multi-Modal Approach to Medical Image Classification via Spatial-Spectral Summarizer Fusion Network
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于医学图像分类任务，旨在解决单一空间特征学习不足的问题。提出S$^3$F-Net，融合空间与频谱信息，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2509.23442v1](http://arxiv.org/pdf/2509.23442v1)**

> **作者:** Md. Saiful Bari Siddiqui; Mohammed Imamul Hassan Bhuiyan
>
> **备注:** Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI). This preprint includes few additional details not present in the journal submission
>
> **摘要:** Convolutional Neural Networks have become a cornerstone of medical image analysis due to their proficiency in learning hierarchical spatial features. However, this focus on a single domain is inefficient at capturing global, holistic patterns and fails to explicitly model an image's frequency-domain characteristics. To address these challenges, we propose the Spatial-Spectral Summarizer Fusion Network (S$^3$F-Net), a dual-branch framework that learns from both spatial and spectral representations simultaneously. The S$^3$F-Net performs a fusion of a deep spatial CNN with our proposed shallow spectral encoder, SpectraNet. SpectraNet features the proposed SpectralFilter layer, which leverages the Convolution Theorem by applying a bank of learnable filters directly to an image's full Fourier spectrum via a computation-efficient element-wise multiplication. This allows the SpectralFilter layer to attain a global receptive field instantaneously, with its output being distilled by a lightweight summarizer network. We evaluate S$^3$F-Net across four medical imaging datasets spanning different modalities to validate its efficacy and generalizability. Our framework consistently and significantly outperforms its strong spatial-only baseline in all cases, with accuracy improvements of up to 5.13%. With a powerful Bilinear Fusion, S$^3$F-Net achieves a SOTA competitive accuracy of 98.76% on the BRISC2025 dataset. Concatenation Fusion performs better on the texture-dominant Chest X-Ray Pneumonia dataset, achieving 93.11% accuracy, surpassing many top-performing, much deeper models. Our explainability analysis also reveals that the S$^3$F-Net learns to dynamically adjust its reliance on each branch based on the input pathology. These results verify that our dual-domain approach is a powerful and generalizable paradigm for medical image analysis.
>
---
#### [new 036] Training-Free Multimodal Guidance for Video to Audio Generation
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于视频到音频生成任务，解决现有方法依赖大量数据或无法保持多模态一致的问题。提出无需训练的多模态引导机制，提升音频与视频、文本的对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.24550v1](http://arxiv.org/pdf/2509.24550v1)**

> **作者:** Eleonora Grassucci; Giuliano Galadini; Giordano Cicchetti; Aurelio Uncini; Fabio Antonacci; Danilo Comminiello
>
> **摘要:** Video-to-audio (V2A) generation aims to synthesize realistic and semantically aligned audio from silent videos, with potential applications in video editing, Foley sound design, and assistive multimedia. Although the excellent results, existing approaches either require costly joint training on large-scale paired datasets or rely on pairwise similarities that may fail to capture global multimodal coherence. In this work, we propose a novel training-free multimodal guidance mechanism for V2A diffusion that leverages the volume spanned by the modality embeddings to enforce unified alignment across video, audio, and text. The proposed multimodal diffusion guidance (MDG) provides a lightweight, plug-and-play control signal that can be applied on top of any pretrained audio diffusion model without retraining. Experiments on VGGSound and AudioCaps demonstrate that our MDG consistently improves perceptual quality and multimodal alignment compared to baselines, proving the effectiveness of a joint multimodal guidance for V2A.
>
---
#### [new 037] Index-MSR: A high-efficiency multimodal fusion framework for speech recognition
- **分类: eess.AS; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于语音识别任务，解决领域术语和短语识别问题。提出Index-MSR框架，融合视频文本信息提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.22744v1](http://arxiv.org/pdf/2509.22744v1)**

> **作者:** Jinming Chen; Lu Wang; Zheshu Song; Wei Deng
>
> **备注:** Submit to icassp 2026
>
> **摘要:** Driven by large scale datasets and LLM based architectures, automatic speech recognition (ASR) systems have achieved remarkable improvements in accuracy. However, challenges persist for domain-specific terminology, and short utterances lacking semantic coherence, where recognition performance often degrades significantly. In this work, we present Index-MSR, an efficient multimodal speech recognition framework. At its core is a novel Multimodal Fusion Decoder (MFD), which effectively incorporates text-related information from videos (e.g., subtitles and presentation slides) into the speech recognition. This cross-modal integration not only enhances overall ASR accuracy but also yields substantial reductions in substitution errors. Extensive evaluations on both an in-house subtitle dataset and a public AVSR dataset demonstrate that Index-MSR achieves sota accuracy, with substitution errors reduced by 20,50%. These results demonstrate that our approach efficiently exploits text-related cues from video to improve speech recognition accuracy, showing strong potential in applications requiring strict audio text synchronization, such as audio translation.
>
---
#### [new 038] Unsupervised Speech Enhancement using Data-defined Priors
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音增强任务，解决无监督训练数据不足的问题。提出双分支编码器-解码器结构，利用未配对数据定义先验，提升增强效果。**

- **链接: [http://arxiv.org/pdf/2509.22942v1](http://arxiv.org/pdf/2509.22942v1)**

> **作者:** Dominik Klement; Matthew Maciejewski; Sanjeev Khudanpur; Jan Černocký; Lukáš Burget
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** The majority of deep learning-based speech enhancement methods require paired clean-noisy speech data. Collecting such data at scale in real-world conditions is infeasible, which has led the community to rely on synthetically generated noisy speech. However, this introduces a gap between the training and testing phases. In this work, we propose a novel dual-branch encoder-decoder architecture for unsupervised speech enhancement that separates the input into clean speech and residual noise. Adversarial training is employed to impose priors on each branch, defined by unpaired datasets of clean speech and, optionally, noise. Experimental results show that our method achieves performance comparable to leading unsupervised speech enhancement approaches. Furthermore, we demonstrate the critical impact of clean speech data selection on enhancement performance. In particular, our findings reveal that performance may appear overly optimistic when in-domain clean speech data are used for prior definition -- a practice adopted in previous unsupervised speech enhancement studies.
>
---
#### [new 039] LORT: Locally Refined Convolution and Taylor Transformer for Monaural Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在提升语音质量同时降低模型复杂度。提出LORT架构，结合局部细化卷积和改进的Transformer模块，实现高效准确的语音增强。**

- **链接: [http://arxiv.org/pdf/2509.23832v1](http://arxiv.org/pdf/2509.23832v1)**

> **作者:** Junyu Wang; Zizhen Lin; Tianrui Wang; Meng Ge; Longbiao Wang; Jianwu Dang
>
> **备注:** Speech Communication
>
> **摘要:** Achieving superior enhancement performance while maintaining a low parameter count and computational complexity remains a challenge in the field of speech enhancement. In this paper, we introduce LORT, a novel architecture that integrates spatial-channel enhanced Taylor Transformer and locally refined convolution for efficient and robust speech enhancement. We propose a Taylor multi-head self-attention (T-MSA) module enhanced with spatial-channel enhancement attention (SCEA), designed to facilitate inter-channel information exchange and alleviate the spatial attention limitations inherent in Taylor-based Transformers. To complement global modeling, we further present a locally refined convolution (LRC) block that integrates convolutional feed-forward layers, time-frequency dense local convolutions, and gated units to capture fine-grained local details. Built upon a U-Net-like encoder-decoder structure with only 16 output channels in the encoder, LORT processes noisy inputs through multi-resolution T-MSA modules using alternating downsampling and upsampling operations. The enhanced magnitude and phase spectra are decoded independently and optimized through a composite loss function that jointly considers magnitude, complex, phase, discriminator, and consistency objectives. Experimental results on the VCTK+DEMAND and DNS Challenge datasets demonstrate that LORT achieves competitive or superior performance to state-of-the-art (SOTA) models with only 0.96M parameters, highlighting its effectiveness for real-world speech enhancement applications with limited computational resources.
>
---
#### [new 040] HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，解决韩英混合语言识别问题。提出HiKE框架，提供高质量数据和分层标注，提升模型对代码切换的识别能力。**

- **链接: [http://arxiv.org/pdf/2509.24613v1](http://arxiv.org/pdf/2509.24613v1)**

> **作者:** Gio Paik; Yongbeom Kim; Soungmin Lee; Sangmin Ahn; Chanwoo Kim
>
> **备注:** 5 pages, 2 figures, Submitted to ICASSP2026
>
> **摘要:** Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that while most multilingual ASR models initially struggle with CS-ASR, this capability can be enabled through fine-tuning with CS data. HiKE will be available at https://github.com/ThetaOne-AI/HiKE.
>
---
#### [new 041] PerformSinger: Multimodal Singing Voice Synthesis Leveraging Synchronized Lip Cues from Singing Performance Videos
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文属于语音合成任务，解决传统模型依赖精细音素时长的问题，通过引入视频唇部信息实现无需时长的高质量歌声合成。**

- **链接: [http://arxiv.org/pdf/2509.22718v1](http://arxiv.org/pdf/2509.22718v1)**

> **作者:** Ke Gu; Zhicong Wu; Peng Bai; Sitong Qiao; Zhiqi Jiang; Junchen Lu; Xiaodong Shi; Xinyuan Qian
>
> **摘要:** Existing singing voice synthesis (SVS) models largely rely on fine-grained, phoneme-level durations, which limits their practical application. These methods overlook the complementary role of visual information in duration prediction.To address these issues, we propose PerformSinger, a pioneering multimodal SVS framework, which incorporates lip cues from video as a visual modality, enabling high-quality "duration-free" singing voice synthesis. PerformSinger comprises parallel multi-branch multimodal encoders, a feature fusion module, a duration and variational prediction network, a mel-spectrogram decoder and a vocoder. The fusion module, composed of adapter and fusion blocks, employs a progressive fusion strategy within an aligned semantic space to produce high-quality multimodal feature representations, thereby enabling accurate duration prediction and high-fidelity audio synthesis. To facilitate the research, we design, collect and annotate a novel SVS dataset involving synchronized video streams and precise phoneme-level manual annotations. Extensive experiments demonstrate the state-of-the-art performance of our proposal in both subjective and objective evaluations. The code and dataset will be publicly available.
>
---
#### [new 042] Safe Task Space Synchronization with Time-Delayed Information
- **分类: cs.RO; cs.SY; eess.SP; eess.SY**

- **简介: 该论文属于人机协作任务，解决机器人与人类轨迹同步问题。通过设计自适应控制器，利用延迟信息实现安全同步。**

- **链接: [http://arxiv.org/pdf/2509.22976v1](http://arxiv.org/pdf/2509.22976v1)**

> **作者:** Rounak Bhattacharya; Vrithik R. Guthikonda; Ashwin P. Dani
>
> **摘要:** In this paper, an adaptive controller is designed for the synchronization of the trajectory of a robot with unknown kinematics and dynamics to that of the current human trajectory in the task space using the delayed human trajectory information. The communication time delay may be a result of various factors that arise in human-robot collaboration tasks, such as sensor processing or fusion to estimate trajectory/intent, network delays, or computational limitations. The developed adaptive controller uses Barrier Lyapunov Function (BLF) to constrain the Cartesian coordinates of the robot to ensure safety, an ICL-based adaptive law to account for the unknown kinematics, and a gradient-based adaptive law to estimate unknown dynamics. Barrier Lyapunov-Krasovskii (LK) functionals are used for the stability analysis to show that the synchronization and parameter estimation errors remain semi-globally uniformly ultimately bounded (SGUUB). The simulation results based on a human-robot synchronization scenario with time delay are provided to demonstrate the effectiveness of the designed synchronization controller with safety constraints.
>
---
#### [new 043] AI-Assisted Music Production: A User Study on Text-to-Music Models
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于音乐生成任务，探讨文本到音乐模型在音乐制作中的应用。研究解决TTM如何影响创作流程的问题，通过用户实验和访谈分析其效果与挑战。**

- **链接: [http://arxiv.org/pdf/2509.23364v1](http://arxiv.org/pdf/2509.23364v1)**

> **作者:** Francesca Ronchini; Luca Comanducci; Simone Marcucci; Fabio Antonacci
>
> **备注:** Accepted at 17th International Symposium on Computer Music Multidisciplinary Research (CMMR 25)
>
> **摘要:** Text-to-music models have revolutionized the creative landscape, offering new possibilities for music creation. Yet their integration into musicians workflows remains underexplored. This paper presents a case study on how TTM models impact music production, based on a user study of their effect on producers creative workflows. Participants produce tracks using a custom tool combining TTM and source separation models. Semi-structured interviews and thematic analysis reveal key challenges, opportunities, and ethical considerations. The findings offer insights into the transformative potential of TTMs in music production, as well as challenges in their real-world integration.
>
---
#### [new 044] End-to-end Topographic Auditory Models Replicate Signatures of Human Auditory Cortex
- **分类: q-bio.NC; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于 auditory modeling 任务，旨在解决现有模型缺乏生物拓扑结构的问题。通过引入拓扑约束，构建了具有类脑拓扑结构的 TopoAudio 模型。**

- **链接: [http://arxiv.org/pdf/2509.24039v1](http://arxiv.org/pdf/2509.24039v1)**

> **作者:** Haider Al-Tahan; Mayukh Deb; Jenelle Feather; N. Apurva Ratan Murty
>
> **摘要:** The human auditory cortex is topographically organized. Neurons with similar response properties are spatially clustered, forming smooth maps for acoustic features such as frequency in early auditory areas, and modular regions selective for music and speech in higher-order cortex. Yet, evaluations for current computational models of auditory perception do not measure whether such topographic structure is present in a candidate model. Here, we show that cortical topography is not present in the previous best-performing models at predicting human auditory fMRI responses. To encourage the emergence of topographic organization, we adapt a cortical wiring-constraint loss originally designed for visual perception. The new class of topographic auditory models, TopoAudio, are trained to classify speech, and environmental sounds from cochleagram inputs, with an added constraint that nearby units on a 2D cortical sheet develop similar tuning. Despite these additional constraints, TopoAudio achieves high accuracy on benchmark tasks comparable to the unconstrained non-topographic baseline models. Further, TopoAudio predicts the fMRI responses in the brain as well as standard models, but unlike standard models, TopoAudio develops smooth, topographic maps for tonotopy and amplitude modulation (common properties of early auditory representation, as well as clustered response modules for music and speech (higher-order selectivity observed in the human auditory cortex). TopoAudio is the first end-to-end biologically grounded auditory model to exhibit emergent topography, and our results emphasize that a wiring-length constraint can serve as a general-purpose regularization tool to achieve biologically aligned representations.
>
---
#### [new 045] AudioFuse: Unified Spectral-Temporal Learning via a Hybrid ViT-1D CNN Architecture for Robust Phonocardiogram Classification
- **分类: eess.AS; cs.AI; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于心音图分类任务，解决如何有效融合频谱与时间信息的问题。提出AudioFuse架构，结合ViT和1D CNN，提升分类性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.23454v1](http://arxiv.org/pdf/2509.23454v1)**

> **作者:** Md. Saiful Bari Siddiqui; Utsab Saha
>
> **备注:** Submitted to ICASSP 2026. This preprint includes some additional details beyond the conference submission
>
> **摘要:** Biomedical audio signals, such as phonocardiograms (PCG), are inherently rhythmic and contain diagnostic information in both their spectral (tonal) and temporal domains. Standard 2D spectrograms provide rich spectral features but compromise the phase information and temporal precision of the 1D waveform. We propose AudioFuse, an architecture that simultaneously learns from both complementary representations to classify PCGs. To mitigate the overfitting risk common in fusion models, we integrate a custom, wide-and-shallow Vision Transformer (ViT) for spectrograms with a shallow 1D CNN for raw waveforms. On the PhysioNet 2016 dataset, AudioFuse achieves a state-of-the-art competitive ROC-AUC of 0.8608 when trained from scratch, outperforming its spectrogram (0.8066) and waveform (0.8223) baselines. Moreover, it demonstrates superior robustness to domain shift on the challenging PASCAL dataset, maintaining an ROC-AUC of 0.7181 while the spectrogram baseline collapses (0.4873). Fusing complementary representations thus provides a strong inductive bias, enabling the creation of efficient, generalizable classifiers without requiring large-scale pre-training.
>
---
#### [new 046] Unsupervised Single-Channel Speech Separation with a Diffusion Prior under Speaker-Embedding Guidance
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，解决无监督条件下语音源分离问题。通过引入说话人嵌入引导的扩散模型，提升分离结果的说话人一致性与质量。**

- **链接: [http://arxiv.org/pdf/2509.24395v1](http://arxiv.org/pdf/2509.24395v1)**

> **作者:** Runwu Shi; Kai Li; Chang Li; Jiang Wang; Sihan Tan; Kazuhiro Nakadai
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP 2026
>
> **摘要:** Speech separation is a fundamental task in audio processing, typically addressed with fully supervised systems trained on paired mixtures. While effective, such systems typically rely on synthetic data pipelines, which may not reflect real-world conditions. Instead, we revisit the source-model paradigm, training a diffusion generative model solely on anechoic speech and formulating separation as a diffusion inverse problem. However, unconditional diffusion models lack speaker-level conditioning, they can capture local acoustic structure but produce temporally inconsistent speaker identities in separated sources. To address this limitation, we propose Speaker-Embedding guidance that, during the reverse diffusion process, maintains speaker coherence within each separated track while driving embeddings of different speakers further apart. In addition, we propose a new separation-oriented solver tailored for speech separation, and both strategies effectively enhance performance on the challenging task of unsupervised source-model-based speech separation, as confirmed by extensive experimental results. Audio samples and code are available at https://runwushi.github.io/UnSepDiff_demo.
>
---
#### [new 047] SynthCloner: Synthesizer Preset Conversion via Factorized Codec with ADSR Envelope Control
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音色转换任务，解决合成器预设转换中控制不足的问题。提出SynthCloner模型，分离音频为ADSR、音色和内容三属性，实现独立控制。**

- **链接: [http://arxiv.org/pdf/2509.24286v1](http://arxiv.org/pdf/2509.24286v1)**

> **作者:** Jeng-Yue Liu; Ting-Chao Hsu; Yen-Tung Yeh; Li Su; Yi-Hsuan Yang
>
> **备注:** Submitted to ICASSP26
>
> **摘要:** Electronic synthesizer sounds are controlled by presets, parameters settings that yield complex timbral characteristics and ADSR envelopes, making preset conversion particularly challenging. Recent approaches to timbre transfer often rely on spectral objectives or implicit style matching, offering limited control over envelope shaping. Moreover, public synthesizer datasets rarely provide diverse coverage of timbres and ADSR envelopes. To address these gaps, we present SynthCloner, a factorized codec model that disentangles audio into three attributes: ADSR envelope, timbre, and content. This separation enables expressive synthesizer preset conversion with independent control over these three attributes. Additionally, we introduce SynthCAT, a new synthesizer dataset with a task-specific rendering pipeline covering 250 timbres, 120 ADSR envelopes, and 100 MIDI sequences. Experiments show that SynthCloner outperforms baselines on both objective and subjective metrics, while enabling independent attribute control. The code, model checkpoint, and audio examples are available at https://buffett0323.github.io/synthcloner/.
>
---
#### [new 048] AISHELL6-whisper: A Chinese Mandarin Audio-visual Whisper Speech Dataset with Speech Recognition Baselines
- **分类: eess.AS; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决中文耳语识别数据不足的问题。提出AISHELL6-Whisper数据集及基于Whisper-Flamingo的音视频识别模型，提升耳语识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.23833v1](http://arxiv.org/pdf/2509.23833v1)**

> **作者:** Cancan Li; Fei Su; Juan Liu; Hui Bu; Yulong Wan; Hongbin Suo; Ming Li
>
> **摘要:** Whisper speech recognition is crucial not only for ensuring privacy in sensitive communications but also for providing a critical communication bridge for patients under vocal restraint and enabling discrete interaction in noise-sensitive environments. The development of Chinese mandarin audio-visual whisper speech recognition is hindered by the lack of large-scale datasets. We present AISHELL6-Whisper, a large-scale open-source audio-visual whisper speech dataset, featuring 30 hours each of whisper speech and parallel normal speech, with synchronized frontal facial videos. Moreover, we propose an audio-visual speech recognition (AVSR) baseline based on the Whisper-Flamingo framework, which integrates a parallel training strategy to align embeddings across speech types, and employs a projection layer to adapt to whisper speech's spectral properties. The model achieves a Character Error Rate (CER) of 4.13% for whisper speech and 1.11% for normal speech in the test set of our dataset, and establishes new state-of-the-art results on the wTIMIT benchmark. The dataset and the AVSR baseline codes are open-sourced at https://zutm.github.io/AISHELL6-Whisper.
>
---
#### [new 049] BFA: Real-time Multilingual Text-to-speech Forced Alignment
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音文本对齐任务，解决多语言实时对齐问题。提出BFA系统，结合CUPE和CTC解码器，提升对齐速度与精度。**

- **链接: [http://arxiv.org/pdf/2509.23147v1](http://arxiv.org/pdf/2509.23147v1)**

> **作者:** Abdul Rehman; Jingyao Cai; Jian-Jun Zhang; Xiaosong Yang
>
> **备注:** Under review
>
> **摘要:** We present Bournemouth Forced Aligner (BFA), a system that combines a Contextless Universal Phoneme Encoder (CUPE) with a connectionist temporal classification (CTC)based decoder. BFA introduces explicit modelling of inter-phoneme gaps and silences and hierarchical decoding strategies, enabling fine-grained boundary prediction. Evaluations on TIMIT and Buckeye corpora show that BFA achieves competitive recall relative to Montreal Forced Aligner at relaxed tolerance levels, while predicting both onset and offset boundaries for richer temporal structure. BFA processes speech up to 240x faster than MFA, enabling faster than real-time alignment. This combination of speed and silence-aware alignment opens opportunities for interactive speech applications previously constrained by slow aligners.
>
---
#### [new 050] Word-Level Emotional Expression Control in Zero-Shot Text-to-Speech Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决零样本场景下单词级情感控制问题。通过自训练框架WeSCon实现情感与语速的精准控制。**

- **链接: [http://arxiv.org/pdf/2509.24629v1](http://arxiv.org/pdf/2509.24629v1)**

> **作者:** Tianrui Wang; Haoyu Wang; Meng Ge; Cheng Gong; Chunyu Qiang; Ziyang Ma; Zikang Huang; Guanrou Yang; Xiaobao Wang; Eng Siong Chng; Xie Chen; Longbiao Wang; Jianwu Dang
>
> **摘要:** While emotional text-to-speech (TTS) has made significant progress, most existing research remains limited to utterance-level emotional expression and fails to support word-level control. Achieving word-level expressive control poses fundamental challenges, primarily due to the complexity of modeling multi-emotion transitions and the scarcity of annotated datasets that capture intra-sentence emotional and prosodic variation. In this paper, we propose WeSCon, the first self-training framework that enables word-level control of both emotion and speaking rate in a pretrained zero-shot TTS model, without relying on datasets containing intra-sentence emotion or speed transitions. Our method introduces a transition-smoothing strategy and a dynamic speed control mechanism to guide the pretrained TTS model in performing word-level expressive synthesis through a multi-round inference process. To further simplify the inference, we incorporate a dynamic emotional attention bias mechanism and fine-tune the model via self-training, thereby activating its ability for word-level expressive control in an end-to-end manner. Experimental results show that WeSCon effectively overcomes data scarcity, achieving state-of-the-art performance in word-level emotional expression control while preserving the strong zero-shot synthesis capabilities of the original TTS model.
>
---
## 更新

#### [replaced 001] SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions
- **分类: cs.SD; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17091v2](http://arxiv.org/pdf/2509.17091v2)**

> **作者:** Massa Baali; Sarthak Bisht; Francisco Teixeira; Kateryna Shapovalenko; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification (SV) models are increasingly integrated into security, personalization, and access control systems, yet their robustness to many real-world challenges remains inadequately benchmarked. These include a variety of natural and maliciously created conditions causing signal degradations or mismatches between enrollment and test data, impacting performance. Existing benchmarks evaluate only subsets of these conditions, missing others entirely. We introduce SVeritas, a comprehensive Speaker Verification tasks benchmark suite, assessing SV systems under stressors like recording duration, spontaneity, content, noise, microphone distance, reverberation, channel mismatches, audio bandwidth, codecs, speaker age, and susceptibility to spoofing and adversarial attacks. While several benchmarks do exist that each cover some of these issues, SVeritas is the first comprehensive evaluation that not only includes all of these, but also several other entirely new, but nonetheless important, real-life conditions that have not previously been benchmarked. We use SVeritas to evaluate several state-of-the-art SV models and observe that while some architectures maintain stability under common distortions, they suffer substantial performance degradation in scenarios involving cross-language trials, age mismatches, and codec-induced compression. Extending our analysis across demographic subgroups, we further identify disparities in robustness across age groups, gender, and linguistic backgrounds. By standardizing evaluation under realistic and synthetic stress conditions, SVeritas enables precise diagnosis of model weaknesses and establishes a foundation for advancing equitable and reliable speaker verification systems.
>
---
#### [replaced 002] Versatile Symbolic Music-for-Music Modeling via Function Alignment
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15548v2](http://arxiv.org/pdf/2506.15548v2)**

> **作者:** Junyan Jiang; Daniel Chin; Liwei Lin; Xuanjie Liu; Gus Xia
>
> **摘要:** Many music AI models learn a map between music content and human-defined labels. However, many annotations, such as chords, can be naturally expressed within the music modality itself, e.g., as sequences of symbolic notes. This observation enables both understanding tasks (e.g., chord recognition) and conditional generation tasks (e.g., chord-conditioned melody generation) to be unified under a music-for-music sequence modeling paradigm. In this work, we propose parameter-efficient solutions for a variety of symbolic music-for-music tasks. The high-level idea is that (1) we utilize a pretrained Language Model (LM) for both the reference and the target sequence and (2) we link these two LMs via a lightweight adapter. Experiments show that our method achieves superior performance among different tasks such as chord recognition, melody generation, and drum track generation. All demos, code and model weights are publicly available.
>
---
#### [replaced 003] Evaluating the Effectiveness of Transformer Layers in Wav2Vec 2.0, XLS-R, and Whisper for Speaker Identification Tasks
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.00230v2](http://arxiv.org/pdf/2509.00230v2)**

> **作者:** Linus Stuhlmann; Michael Alexander Saxer
>
> **备注:** This was a conducted student project at our univerity, we don't think this fulfills the requirements for a publication on arxiv
>
> **摘要:** This study evaluates the performance of three advanced speech encoder models, Wav2Vec 2.0, XLS-R, and Whisper, in speaker identification tasks. By fine-tuning these models and analyzing their layer-wise representations using SVCCA, k-means clustering, and t-SNE visualizations, we found that Wav2Vec 2.0 and XLS-R capture speaker-specific features effectively in their early layers, with fine-tuning improving stability and performance. Whisper showed better performance in deeper layers. Additionally, we determined the optimal number of transformer layers for each model when fine-tuned for speaker identification tasks.
>
---
#### [replaced 004] DM-Codec: Distilling Multimodal Representations for Speech Tokenization
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.15017v2](http://arxiv.org/pdf/2410.15017v2)**

> **作者:** Md Mubtasim Ahasan; Md Fahim; Tasnim Mohiuddin; A K M Mahbubur Rahman; Aman Chadha; Tariq Iqbal; M Ashraful Amin; Md Mofijul Islam; Amin Ahsan Ali
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Recent advancements in speech-language models have yielded significant improvements in speech tokenization and synthesis. However, effectively mapping the complex, multidimensional attributes of speech into discrete tokens remains challenging. This process demands acoustic, semantic, and contextual information for precise speech representations. Existing speech representations generally fall into two categories: acoustic tokens from audio codecs and semantic tokens from speech self-supervised learning models. Although recent efforts have unified acoustic and semantic tokens for improved performance, they overlook the crucial role of contextual representation in comprehensive speech modeling. Our empirical investigations reveal that the absence of contextual representations results in elevated Word Error Rate (WER) and Word Information Lost (WIL) scores in speech transcriptions. To address these limitations, we propose two novel distillation approaches: (1) a language model (LM)-guided distillation method that incorporates contextual information, and (2) a combined LM and self-supervised speech model (SM)-guided distillation technique that effectively distills multimodal representations (acoustic, semantic, and contextual) into a comprehensive speech tokenizer, termed DM-Codec. The DM-Codec architecture adopts a streamlined encoder-decoder framework with a Residual Vector Quantizer (RVQ) and incorporates the LM and SM during the training process. Experiments show DM-Codec significantly outperforms state-of-the-art speech tokenization models, reducing WER by up to 13.46%, WIL by 9.82%, and improving speech quality by 5.84% and intelligibility by 1.85% on the LibriSpeech benchmark dataset. Code, samples, and checkpoints are available at https://github.com/mubtasimahasan/DM-Codec.
>
---
#### [replaced 005] Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.06820v2](http://arxiv.org/pdf/2506.06820v2)**

> **作者:** Wenyu Zhang; Yingxu He; Geyu Lin; Zhuohan Liu; Shuo Sun; Bin Wang; Xunlong Zou; Jeremy H. M. Wong; Qiongqiong Wang; Hardik B. Sailor; Nancy F. Chen; Ai Ti Aw
>
> **摘要:** Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses. Experiments on two out-of-domain datasets demonstrate the generalization capabilities of the resulting model.
>
---
#### [replaced 006] TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation
- **分类: cs.IR; cs.AI; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09685v2](http://arxiv.org/pdf/2509.09685v2)**

> **作者:** Keunwoo Choi; Seungheon Doh; Juhan Nam
>
> **摘要:** We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In the proposed pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at https://talkpl.ai/talkplaydata2.html.
>
---
#### [replaced 007] Discrete Audio Tokens: More Than a Survey!
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10274v3](http://arxiv.org/pdf/2506.10274v3)**

> **作者:** Pooneh Mousavi; Gallil Maimon; Adel Moumen; Darius Petermann; Jiatong Shi; Haibin Wu; Haici Yang; Anastasia Kuznetsova; Artem Ploujnikov; Ricard Marxer; Bhuvana Ramabhadran; Benjamin Elizalde; Loren Lugosch; Jinyu Li; Cem Subakan; Phil Woodland; Minje Kim; Hung-yi Lee; Shinji Watanabe; Yossi Adi; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream tasks. They provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: https://poonehmousavi.github.io/dates-website/.
>
---
#### [replaced 008] GCDance: Genre-Controlled Music-Driven 3D Full Body Dance Generation
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.18309v3](http://arxiv.org/pdf/2502.18309v3)**

> **作者:** Xinran Liu; Xu Dong; Shenbin Qian; Diptesh Kanojia; Wenwu Wang; Zhenhua Feng
>
> **摘要:** Music-driven dance generation is a challenging task as it requires strict adherence to genre-specific choreography while ensuring physically realistic and precisely synchronized dance sequences with the music's beats and rhythm. Although significant progress has been made in music-conditioned dance generation, most existing methods struggle to convey specific stylistic attributes in generated dance. To bridge this gap, we propose a diffusion-based framework for genre-specific 3D full-body dance generation, conditioned on both music and descriptive text. To effectively incorporate genre information, we develop a text-based control mechanism that maps input prompts, either explicit genre labels or free-form descriptive text, into genre-specific control signals, enabling precise and controllable text-guided generation of genre-consistent dance motions. Furthermore, to enhance the alignment between music and textual conditions, we leverage the features of a music foundation model, facilitating coherent and semantically aligned dance synthesis. Last, to balance the objectives of extracting text-genre information and maintaining high-quality generation results, we propose a novel multi-task optimization strategy. This effectively balances competing factors such as physical realism, spatial accuracy, and text classification, significantly improving the overall quality of the generated sequences. Extensive experimental results obtained on the FineDance and AIST++ datasets demonstrate the superiority of GCDance over the existing state-of-the-art approaches.
>
---
#### [replaced 009] Xi+: Uncertainty Supervision for Robust Speaker Embedding
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05993v3](http://arxiv.org/pdf/2509.05993v3)**

> **作者:** Junjie Li; Kong Aik Lee; Duc-Tuan Truong; Tianchi Liu; Man-Wai Mak
>
> **摘要:** There are various factors that can influence the performance of speaker recognition systems, such as emotion, language and other speaker-related or context-related variations. Since individual speech frames do not contribute equally to the utterance-level representation, it is essential to estimate the importance or reliability of each frame. The xi-vector model addresses this by assigning different weights to frames based on uncertainty estimation. However, its uncertainty estimation model is implicitly trained through classification loss alone and does not consider the temporal relationships between frames, which may lead to suboptimal supervision. In this paper, we propose an improved architecture, xi+. Compared to xi-vector, xi+ incorporates a temporal attention module to capture frame-level uncertainty in a context-aware manner. In addition, we introduce a novel loss function, Stochastic Variance Loss, which explicitly supervises the learning of uncertainty. Results demonstrate consistent performance improvements of about 10\% on the VoxCeleb1-O set and 11\% on the NIST SRE 2024 evaluation set.
>
---
#### [replaced 010] IML-Spikeformer: Input-aware Multi-Level Spiking Transformer for Speech Processing
- **分类: cs.MM; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07396v2](http://arxiv.org/pdf/2507.07396v2)**

> **作者:** Zeyang Song; Shimin Zhang; Yuhong Chou; Jibin Wu; Haizhou Li
>
> **备注:** Accepted by TNNLS
>
> **摘要:** Spiking Neural Networks (SNNs), inspired by biological neural mechanisms, represent a promising neuromorphic computing paradigm that offers energy-efficient alternatives to traditional Artificial Neural Networks (ANNs). Despite proven effectiveness, SNN architectures have struggled to achieve competitive performance on large-scale speech processing tasks. Two key challenges hinder progress: (1) the high computational overhead during training caused by multi-timestep spike firing, and (2) the absence of large-scale SNN architectures tailored to speech processing tasks. To overcome the issues, we introduce Input-aware Multi-Level Spikeformer, i.e. IML-Spikeformer, a spiking Transformer architecture specifically designed for large-scale speech processing. Central to our design is the Input-aware Multi-Level Spike (IMLS) mechanism, which simulates multi-timestep spike firing within a single timestep using an adaptive, input-aware thresholding scheme. IML-Spikeformer further integrates a Re-parameterized Spiking Self-Attention (RepSSA) module with a Hierarchical Decay Mask (HDM), forming the HD-RepSSA module. This module enhances the precision of attention maps and enables modeling of multi-scale temporal dependencies in speech signals. Experiments demonstrate that IML-Spikeformer achieves word error rates of 6.0\% on AiShell-1 and 3.4\% on Librispeech-960, comparable to conventional ANN transformers while reducing theoretical inference energy consumption by 4.64$\times$ and 4.32$\times$ respectively. IML-Spikeformer marks an advance of scalable SNN architectures for large-scale speech processing in both task performance and energy efficiency. Our source code and model checkpoints are publicly available at github.com/Pooookeman/IML-Spikeformer.
>
---
#### [replaced 011] ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signals
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.14689v3](http://arxiv.org/pdf/2508.14689v3)**

> **作者:** Yucong Zhang; Juan Liu; Ming Li
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Pre-trained foundation models have demonstrated remarkable success in audio, vision and language, yet their potential for general machine signal modeling with arbitrary sampling rates-covering acoustic, vibration, and other industrial sensor data-remains under-explored. In this work, we propose a novel foundation model ECHO that integrates an advanced band-split architecture with frequency positional embeddings, enabling spectral localization across arbitrary sampling configurations. Moreover, the model incorporates sliding patches to support inputs of variable length without padding or cropping, producing a concise embedding that retains both temporal and spectral fidelity and naturally extends to streaming scenarios. We evaluate our method on various kinds of machine signal datasets, including previous DCASE task 2 challenges (2020-2025), and widely-used industrial signal corpora. Experimental results demonstrate consistent state-of-the-art performance in machine signal anomaly detection and fault classification, confirming the effectiveness and generalization capability of the proposed model. We open-sourced ECHO on https://github.com/yucongzh/ECHO.
>
---
#### [replaced 012] M6(GPT)3: Generating Multitrack Modifiable Multi-Minute MIDI Music from Text using Genetic algorithms, Probabilistic methods and GPT Models in any Progression and Time Signature
- **分类: cs.SD; cs.HC; eess.AS; H.5.5; I.2.7; I.2.8; G.3**

- **链接: [http://arxiv.org/pdf/2409.12638v3](http://arxiv.org/pdf/2409.12638v3)**

> **作者:** Jakub Poćwiardowski; Mateusz Modrzejewski; Marek S. Tatara
>
> **备注:** Published in 2025 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)
>
> **摘要:** This work introduces the M6(GPT)3 composer system, capable of generating complete, multi-minute musical compositions with complex structures in any time signature, in the MIDI domain from input descriptions in natural language. The system utilizes an autoregressive transformer language model to map natural language prompts to composition parameters in JSON format. The defined structure includes time signature, scales, chord progressions, and valence-arousal values, from which accompaniment, melody, bass, motif, and percussion tracks are created. We propose a genetic algorithm for the generation of melodic elements. The algorithm incorporates mutations with musical significance and a fitness function based on normal distribution and predefined musical feature values. The values adaptively evolve, influenced by emotional parameters and distinct playing styles. The system for generating percussion in any time signature utilises probabilistic methods, including Markov chains. Through both human and objective evaluations, we demonstrate that our music generation approach outperforms baselines on specific, musically meaningful metrics, offering a viable alternative to purely neural network-based systems.
>
---
#### [replaced 013] Attentive Dilated Convolution for Automatic Sleep Staging using Force-directed Layout
- **分类: eess.SP; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.01962v2](http://arxiv.org/pdf/2409.01962v2)**

> **作者:** Md Jobayer; Md Mehedi Hasan Shawon; Tasfin Mahmud; Md. Borhan Uddin Antor; Arshad M. Chowdhury
>
> **备注:** 19-pages main paper and 3-pages supplementary material
>
> **摘要:** Sleep stages play an important role in identifying sleep patterns and diagnosing sleep disorders. In this study, we present an automated sleep stage classifier called the Attentive Dilated Convolutional Neural Network (AttDiCNN), which uses deep learning methodologies to address challenges related to data heterogeneity, computational complexity, and reliable and automatic sleep staging. We employed a force-directed layout based on the visibility graph to capture the most significant information from the EEG signals, thereby representing the spatial-temporal features. The proposed network consists of three modules: the Localized Spatial Feature Extraction Network (LSFE), Spatio-Temporal-Temporal Long Retention Network (S2TLR), and Global Averaging Attention Network (G2A). The LSFE captures spatial information from sleep data, the S2TLR is designed to extract the most pertinent information in long-term contexts, and the G2A reduces computational overhead by aggregating information from the LSFE and S2TLR. We evaluated the performance of our model on three comprehensive and publicly accessible datasets, achieving state-of-the-art accuracies of 98.56%, 99.66%, and 99.08% for the EDFX, HMC, and NCH datasets, respectively, while maintaining a low computational complexity with 1.4 M parameters. Our proposed architecture surpasses existing methodologies in several performance metrics, thereby proving its potential as an automated tool for clinical settings.
>
---
#### [replaced 014] Generative Video Semantic Communication via Multimodal Semantic Fusion with Large Model
- **分类: eess.SP; cs.CV; cs.IT; eess.IV; math.IT**

- **链接: [http://arxiv.org/pdf/2502.13838v2](http://arxiv.org/pdf/2502.13838v2)**

> **作者:** Hang Yin; Li Qiao; Yu Ma; Shuo Sun; Kan Li; Zhen Gao; Dusit Niyato
>
> **备注:** IEEE Transactions on Vehicular Technology
>
> **摘要:** Despite significant advancements in traditional syntactic communications based on Shannon's theory, these methods struggle to meet the requirements of 6G immersive communications, especially under challenging transmission conditions. With the development of generative artificial intelligence (GenAI), progress has been made in reconstructing videos using high-level semantic information. In this paper, we propose a scalable generative video semantic communication framework that extracts and transmits semantic information to achieve high-quality video reconstruction. Specifically, at the transmitter, description and other condition signals (e.g., first frame, sketches, etc.) are extracted from the source video, functioning as text and structural semantics, respectively. At the receiver, the diffusion-based GenAI large models are utilized to fuse the semantics of the multiple modalities for reconstructing the video. Simulation results demonstrate that, at an ultra-low channel bandwidth ratio (CBR), our scheme effectively captures semantic information to reconstruct videos aligned with human perception under different signal-to-noise ratios. Notably, the proposed ``First Frame+Desc." scheme consistently achieves CLIP score exceeding 0.92 at CBR = 0.0057 for SNR > 0 dB. This demonstrates its robust performance even under low SNR conditions.
>
---
#### [replaced 015] FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.11425v2](http://arxiv.org/pdf/2509.11425v2)**

> **作者:** Md Mubtasim Ahasan; Rafat Hasan Khan; Tasnim Mohiuddin; Aman Chadha; Tariq Iqbal; M Ashraful Amin; Amin Ahsan Ali; Md Mofijul Islam; A K M Mahbubur Rahman
>
> **摘要:** Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at https://github.com/mubtasimahasan/FuseCodec.
>
---
#### [replaced 016] HDA-SELD: Hierarchical Cross-Modal Distillation with Multi-Level Data Augmentation for Low-Resource Audio-Visual Sound Event Localization and Detection
- **分类: cs.SD; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.12334v2](http://arxiv.org/pdf/2508.12334v2)**

> **作者:** Qing Wang; Ya Jiang; Hang Chen; Sabato Marco Siniscalchi; Jun Du; Jianqing Gao
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** This work presents HDA-SELD, a unified framework that combines hierarchical cross-modal distillation (HCMD) and multi-level data augmentation to address low-resource audio-visual (AV) sound event localization and detection (SELD). An audio-only SELD model acts as the teacher, transferring knowledge to an AV student model through both output responses and intermediate feature representations. To enhance learning, data augmentation is applied by mixing features randomly selected from multiple network layers and associated loss functions tailored to the SELD task. Extensive experiments on the DCASE 2023 and 2024 Challenge SELD datasets show that the proposed method significantly improves AV SELD performance, yielding relative gains of 21%-38% in the overall metric over the baselines. Notably, our proposed HDA-SELD achieves results comparable to or better than teacher models trained on much larger datasets, surpassing state-of-the-art methods on both DCASE 2023 and 2024 Challenge SELD tasks.
>
---
#### [replaced 017] i-LAVA: Insights on Low Latency Voice-2-Voice Architecture for Agents
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20971v2](http://arxiv.org/pdf/2509.20971v2)**

> **作者:** Anupam Purwar; Aditya Choudhary
>
> **备注:** This paper analyzes a low-latency, end-to-end voice-to-voice (V-2-V) architecture, identifying that the Text-to-Speech (TTS) component has the highest impact on real-time performance. By reducing the number of Residual Vector Quantization (RVQ) iterations in the TTS model, latency can be effectively halved. Its accepted at AIML Systems 2025
>
> **摘要:** We experiment with a low-latency, end-to-end voice-to-voice communication model to optimize it for real-time conversational applications. By analyzing components essential to voice to voice (V-2-V) system viz. automatic speech recognition (ASR), text-to-speech (TTS), and dialog management, our work analyzes how to reduce processing time while maintaining high-quality interactions to identify the levers for optimizing V-2-V system. Our work identifies that TTS component which generates life-like voice, full of emotions including natural pauses and exclamations has highest impact on Real time factor (RTF). The experimented V-2-V architecture utilizes CSM1b has the capability to understand tone as well as context of conversation by ingesting both audio and text of prior exchanges to generate contextually accurate speech. We explored optimization of Residual Vector Quantization (RVQ) iterations by the TTS decoder which come at a cost of decrease in the quality of voice generated. Our experimental evaluations also demonstrate that for V-2-V implementations based on CSM most important optimizations can be brought by reducing the number of RVQ Iterations along with the codebooks used in Mimi.
>
---
#### [replaced 018] EnvSDD: Benchmarking Environmental Sound Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19203v2](http://arxiv.org/pdf/2505.19203v2)**

> **作者:** Han Yin; Yang Xiao; Rohan Kumar Das; Jisheng Bai; Haohe Liu; Wenwu Wang; Mark D Plumbley
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Audio generation systems now create very realistic soundscapes that can enhance media production, but also pose potential risks. Several studies have examined deepfakes in speech or singing voice. However, environmental sounds have different characteristics, which may make methods for detecting speech and singing deepfakes less effective for real-world sounds. In addition, existing datasets for environmental sound deepfake detection are limited in scale and audio types. To address this gap, we introduce EnvSDD, the first large-scale curated dataset designed for this task, consisting of 45.25 hours of real and 316.74 hours of fake audio. The test set includes diverse conditions to evaluate the generalizability, such as unseen generation models and unseen datasets. We also propose an audio deepfake detection system, based on a pre-trained audio foundation model. Results on EnvSDD show that our proposed system outperforms the state-of-the-art systems from speech and singing domains.
>
---
#### [replaced 019] Sidon: Fast and Robust Open-Source Multilingual Speech Restoration for Large-scale Dataset Cleansing
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.17052v2](http://arxiv.org/pdf/2509.17052v2)**

> **作者:** Wataru Nakata; Yuki Saito; Yota Ueda; Hiroshi Saruwatari
>
> **备注:** 5 pages, 1 figures
>
> **摘要:** Large-scale text-to-speech (TTS) systems are limited by the scarcity of clean, multilingual recordings. We introduce Sidon, a fast, open-source speech restoration model that converts noisy in-the-wild speech into studio-quality speech and scales to dozens of languages. Sidon consists of two models: w2v-BERT 2.0 finetuned feature predictor to cleanse features from noisy speech and vocoder trained to synthesize restored speech from the cleansed features. Sidon achieves restoration performance comparable to Miipher: Google's internal speech restoration model with the aim of dataset cleansing for speech synthesis. Sidon is also computationally efficient, running up to 500 times faster than real time on a single GPU. We further show that training a TTS model using a Sidon-cleansed automatic speech recognition corpus improves the quality of synthetic speech in a zero-shot setting. Code and model are released to facilitate reproducible dataset cleansing for the research community.
>
---
#### [replaced 020] Collection: UAV-Based RSS Measurements from the AFAR Challenge in Digital Twin and Real-World Environments
- **分类: eess.SP; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.06823v2](http://arxiv.org/pdf/2505.06823v2)**

> **作者:** Saad Masrur; Ozgur Ozdemir; Anil Gurses; Ismail Guvenc; Mihail L. Sichitiu; Rudra Dutta; Magreth Mushi; homas Zajkowski; Cole Dickerson; Gautham Reddy; Sergio Vargas Villar; Chau-Wai Wong; Baisakhi Chatterjee; Sonali Chaudhari; Zhizhen Li; Yuchen Liu; Paul Kudyba; Haijian Sun; Jaya Sravani Mandapaka; Kamesh Namuduri; Weijie Wang; Fraida Fund
>
> **备注:** Accepted at IEEE Data Descriptions
>
> **摘要:** This paper presents a comprehensive real-world and Digital Twin (DT) dataset collected as part of the AERPAW Find A Rover (AFAR) Challenge, organized by the NSF Aerial Experimentation and Research Platform for Advanced Wireless (AERPAW) testbed and hosted at the Lake Wheeler Field in Raleigh, North Carolina. The AFAR Challenge was a competition involving five finalist university teams, focused on promoting innovation in unmanned aerial vehicle (UAV)-assisted radio frequency (RF) source localization. Participating teams were tasked with designing UAV flight trajectories and localization algorithms to detect the position of a hidden unmanned ground vehicle (UGV), also referred to as a rover, emitting probe signals generated by GNU Radio. The competition was structured to evaluate solutions in a DT environment first, followed by deployment and testing in the AERPAW outdoor wireless testbed. For each team, the UGV was placed at three different positions, resulting in a total of 29 datasets, 15 collected in a DT simulation environment and 14 in a physical outdoor testbed. Each dataset contains time-synchronized measurements of received signal strength (RSS), received signal quality (RSQ), GPS coordinates, UAV velocity, and UAV orientation (roll, pitch, and yaw). Data is organized into structured folders by team, environment (DT and real-world), and UGV location. The dataset supports research in UAV-assisted RF source localization, air-to-ground (A2G) wireless propagation modeling, trajectory optimization, signal prediction, autonomous navigation, and DT validation. With 300k time-synchronized samples from the real-world experiments, the AFAR dataset enables effective training/testing of deep learning (DL) models and supports robust, real-world UAV-based wireless communication and sensing research.
>
---
#### [replaced 021] GRAM: Spatial general-purpose audio representation models for real-world applications
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00934v2](http://arxiv.org/pdf/2506.00934v2)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Although audio foundations models have seen great progress on a wide variety of tasks, their application in real-world acoustic environments with reverberation and noise has been less successful. Moreover, as audio foundation models are typically trained on dry, single-channel audio clips, the inherent spatial nature of real-world sound scenes is overlooked and tasks involving sound localization ruled out. To address these limitations, we propose GRAM: a General-purpose Real-world Audio Model utilizing a multi-channel masked auto-encoder approach to efficiently learn spatial audio representations from high-quality simulated real-world scenes. To evaluate the performance of GRAM and other audio foundation models in real-world sound scenes, we release Nat-HEAR: A naturalistic version of the HEAR benchmark suite comprising a simulated real-world version, as well as two new sound localization tasks. We show that the performance of GRAM surpasses all state-of-the-art self-supervised audio foundation models and speech models on both HEAR and Nat-HEAR, while using only a fraction of the training data. GRAM also showcases state-of-the-art localization performance, surpassing even supervised sound localization approaches, and can be flexibly applied either to a two-channel, binaural sound format or a four-channel, Ambisonics format. Validating GRAM's performance on real-world sound recordings demonstrates robust transfer to real-world scenes. Taken together, GRAM presents a significant advancement towards robust, spatial audio foundation models for real-world applications.
>
---
