# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] On the Difficulty of Token-Level Modeling of Dysfluency and Fluency Shaping Artifacts
- **分类: eess.AS; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对语音识别中口吃与流畅性修饰的建模难题，提出轻量级适配方法，将非流利现象作为特殊标记纳入转录。通过多步微调与语言自适应预训练，在英德语数据上提升识别效果，揭示现有系统对非英语数据的偏差问题。**

- **链接: [https://arxiv.org/pdf/2512.02027v1](https://arxiv.org/pdf/2512.02027v1)**

> **作者:** Kashaf Gulzar; Dominik Wagner; Sebastian P. Bayerl; Florian Hönig; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** 6 pages, 1 figure. Accepted to ASRU 2025. This is the arXiv preprint of the accepted paper
>
> **摘要:** Automatic transcription of stuttered speech remains a challenge, even for modern end-to-end (E2E) automatic speech recognition (ASR) frameworks. Dysfluencies and fluency-shaping artifacts are often overlooked, resulting in non-verbatim transcriptions with limited clinical and research value. We propose a parameter-efficient adaptation method to decode dysfluencies and fluency modifications as special tokens within transcriptions, evaluated on simulated (LibriStutter, English) and natural (KSoF, German) stuttered speech datasets. To mitigate ASR performance disparities and bias towards English, we introduce a multi-step fine-tuning strategy with language-adaptive pretraining. Tokenization analysis further highlights the tokenizer's English-centric bias, which poses challenges for improving performance on German data. Our findings demonstrate the effectiveness of lightweight adaptation techniques for dysfluency-aware ASR while exposing key limitations in multilingual E2E systems.
>
---
#### [new 002] Perceptual evaluation of Acoustic Level of Detail in Virtual Acoustic Environments
- **分类: eess.AS**

- **简介: 该论文研究虚拟声学环境中声学细节水平（ALOD）对听觉感知的影响，旨在确定实现真实感所需的最低细节。通过改变早期反射数量与几何细节，对比不同ALOD下脉冲、贝斯和语音的感知差异，评估真实性、语音可懂度与外化效果。结果表明，大幅降低ALOD仍可保持良好感知质量，关键在于合理模拟混响尾部。**

- **链接: [https://arxiv.org/pdf/2512.02891v1](https://arxiv.org/pdf/2512.02891v1)**

> **作者:** Stefan Fichna; Steven van de Par; Bernhard U. Seeber; Stephan D. Ewert
>
> **备注:** This work has been submitted to Acoustics for possible publication. Template provided by MDPI
>
> **摘要:** Virtual acoustic environments enable the creation and simulation of realistic and eco-logically valid daily-life situations vital for hearing research and audiology. Reverberant indoor environments are particularly important. For real-time applications, room acous-tics simulation requires simplifications, however, the necessary acoustic level of detail (ALOD) remains unclear in order to capture all perceptually relevant effects. This study examines the impact of varying ALOD in simulations of three real environments: a living room with a coupled kitchen, a pub, and an underground station. ALOD was varied by generating different numbers of image sources for early reflections, or by excluding geo-metrical room details specific for each environment. Simulations were perceptually eval-uated using headphones in comparison to binaural room impulse responses measured with a dummy head in the corresponding real environments, or by using loudspeakers. The study assessed the perceived overall difference for a pulse stimulus, a played electric bass and a speech token. Additionally, plausibility, speech intelligibility, and externaliza-tion were evaluated. Results indicate that a strong reduction in ALOD is feasible while maintaining similar plausibility, speech intelligibility, and externalization as with dummy head recordings. The number and accuracy of early reflections appear less relevant, pro-vided diffuse late reverberation is appropriately represented.
>
---
#### [new 003] Generative Multi-modal Feedback for Singing Voice Synthesis Evaluation
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对歌唱语音合成（SVS）评估中单一评分无法全面反映音准、表现力等多维度问题，提出一种生成式多模态反馈框架。通过音频-语言模型生成文本与音频批评，结合人类反应与合成数据进行微调，实现可解释、高质量的评估，有效支持生成模型优化。**

- **链接: [https://arxiv.org/pdf/2512.02523v1](https://arxiv.org/pdf/2512.02523v1)**

> **作者:** Xueyan Li; Yuxin Wang; Mengjie Jiang; Qingzi Zhu; Jiang Zhang; Zoey Kim; Yazhe Niu
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Singing voice synthesis (SVS) has advanced significantly, enabling models to generate vocals with accurate pitch and consistent style. As these capabilities improve, the need for reliable evaluation and optimization becomes increasingly critical. However, current methods like reward systems often rely on single numerical scores, struggle to capture various dimensions such as phrasing or expressiveness, and require costly annotations, limiting interpretability and generalization. To address these issues, we propose a generative feedback (i.e., reward model) framework that provides multi-dimensional language and audio feedback for SVS assessment. Our approach leverages an audio-language model to generate text and audio critiques-covering aspects such as melody, content, and auditory quality. The model is fine-tuned on a hybrid dataset combining human music reactions and synthetic critiques from a MLLMs, enhancing diversity and linguistic richness. Quantitative experiments validate the effectiveness of the proposed dataset and training strategy, demonstrating that the framework produces musically accurate and interpretable evaluations suitable for guiding generative model improvement. The code is at [https://github.com/opendilab/VocalCritic](https://github.com/opendilab/VocalCritic)
>
---
#### [new 004] VibOmni: Towards Scalable Bone-conduction Speech Enhancement on Earables
- **分类: cs.SD**

- **简介: 该论文针对耳戴设备在噪声环境下语音质量差的问题，提出VibOmni系统，利用IMU捕捉骨传导振动，融合音频与振动信号进行端到端语音增强。通过创新的数据增强方法生成合成振动数据，并设计多模态SNR估计算法实现自适应优化，显著提升语音清晰度与识别率，适用于资源受限的移动设备。**

- **链接: [https://arxiv.org/pdf/2512.02515v1](https://arxiv.org/pdf/2512.02515v1)**

> **作者:** Lixing He; Yunqi Guo; Haozheng Hou; Zhenyu Yan
>
> **备注:** Submitted to TMC
>
> **摘要:** Earables, such as True Wireless Stereo earphones and VR/AR headsets, are increasingly popular, yet their compact design poses challenges for robust voice-related applications like telecommunication and voice assistant interactions in noisy environments. Existing speech enhancement systems, reliant solely on omnidirectional microphones, struggle with ambient noise like competing speakers. To address these issues, we propose VibOmni, a lightweight, end-to-end multi-modal speech enhancement system for earables that leverages bone-conducted vibrations captured by widely available Inertial Measurement Units (IMUs). VibOmni integrates a two-branch encoder-decoder deep neural network to fuse audio and vibration features. To overcome the scarcity of paired audio-vibration datasets, we introduce a novel data augmentation technique that models Bone Conduction Functions (BCFs) from limited recordings, enabling synthetic vibration data generation with only 4.5% spectrogram similarity error. Additionally, a multi-modal SNR estimator facilitates continual learning and adaptive inference, optimizing performance in dynamic, noisy settings without on-device back-propagation. Evaluated on real-world datasets from 32 volunteers with different devices, VibOmni achieves up to 21% improvement in Perceptual Evaluation of Speech Quality (PESQ), 26% in Signal-to-Noise Ratio (SNR), and about 40% WER reduction with much less latency on mobile devices. A user study with 35 participants showed 87% preferred VibOmni over baselines, demonstrating its effectiveness for deployment in diverse acoustic environments.
>
---
#### [new 005] Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文针对音乐表现力生成任务，解决小规模标注数据限制模型扩展的问题。提出Pianist Transformer，通过统一MIDI表示、高效异构架构、100亿词自监督预训练，实现大规模数据与模型的可扩展性，显著提升演奏表现的自然度与质量，达到人类主观评价水平。**

- **链接: [https://arxiv.org/pdf/2512.02652v1](https://arxiv.org/pdf/2512.02652v1)**

> **作者:** Hong-Jie You; Jie-Jing Shao; Xiao-Wen Yang; Lin-Han Jia; Lan-Zhe Guo; Yu-Feng Li
>
> **摘要:** Existing methods for expressive music performance rendering rely on supervised learning over small labeled datasets, which limits scaling of both data volume and model size, despite the availability of vast unlabeled music, as in vision and language. To address this gap, we introduce Pianist Transformer, with four key contributions: 1) a unified Musical Instrument Digital Interface (MIDI) data representation for learning the shared principles of musical structure and expression without explicit annotation; 2) an efficient asymmetric architecture, enabling longer contexts and faster inference without sacrificing rendering quality; 3) a self-supervised pre-training pipeline with 10B tokens and 135M-parameter model, unlocking data and model scaling advantages for expressive performance rendering; 4) a state-of-the-art performance model, which achieves strong objective metrics and human-level subjective ratings. Overall, Pianist Transformer establishes a scalable path toward human-like performance synthesis in the music domain.
>
---
#### [new 006] Story2MIDI: Emotionally Aligned Music Generation from Text
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Story2MIDI，一个基于Transformer的序列到序列模型，旨在从文本生成情感一致的音乐。针对文本与音乐情感对齐问题，构建了包含文本-音乐配对的情感数据集，通过客观指标与人类听觉实验验证，模型能有效捕捉并生成符合文本情绪的多样化音乐。**

- **链接: [https://arxiv.org/pdf/2512.02192v1](https://arxiv.org/pdf/2512.02192v1)**

> **作者:** Mohammad Shokri; Alexandra C. Salem; Gabriel Levine; Johanna Devaney; Sarah Ita Levitan
>
> **备注:** 8 pages (6 pages of main text + 2 pages of references and appendices), 4 figures, 1 table. Presented at IEEE Big Data 2025 3rd Workshop on AI Music Generation (AIMG 2025)
>
> **摘要:** In this paper, we introduce Story2MIDI, a sequence-to-sequence Transformer-based model for generating emotion-aligned music from a given piece of text. To develop this model, we construct the Story2MIDI dataset by merging existing datasets for sentiment analysis from text and emotion classification in music. The resulting dataset contains pairs of text blurbs and music pieces that evoke the same emotions in the reader or listener. Despite the small scale of our dataset and limited computational resources, our results indicate that our model effectively learns emotion-relevant features in music and incorporates them into its generation process, producing samples with diverse emotional responses. We evaluate the generated outputs using objective musical metrics and a human listening study, confirming the model's ability to capture intended emotional cues.
>
---
#### [new 007] SAND Challenge: Four Approaches for Dysartria Severity Classification
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文针对神经退行性疾病语音分析中的构音障碍严重程度分类任务，比较了四种模型：ViT-OF、1D-CNN、BiLSTM-OF与层级XGBoost。基于同一语音数据集，评估其性能，结果显示特征工程的XGBoost表现最佳，而深度学习模型亦具竞争力并提供互补见解。**

- **链接: [https://arxiv.org/pdf/2512.02669v1](https://arxiv.org/pdf/2512.02669v1)**

> **作者:** Gauri Deshpande; Harish Battula; Ashish Panda; Sunil Kumar Kopparapu
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** This paper presents a unified study of four distinct modeling approaches for classifying dysarthria severity in the Speech Analysis for Neurodegenerative Diseases (SAND) challenge. All models tackle the same five class classification task using a common dataset of speech recordings. We investigate: (1) a ViT-OF method leveraging a Vision Transformer on spectrogram images, (2) a 1D-CNN approach using eight 1-D CNN's with majority-vote fusion, (3) a BiLSTM-OF approach using nine BiLSTM models with majority vote fusion, and (4) a Hierarchical XGBoost ensemble that combines glottal and formant features through a two stage learning framework. Each method is described, and their performances on a validation set of 53 speakers are compared. Results show that while the feature-engineered XGBoost ensemble achieves the highest macro-F1 (0.86), the deep learning models (ViT, CNN, BiLSTM) attain competitive F1-scores (0.70) and offer complementary insights into the problem.
>
---
#### [new 008] Exploring Definitions of Quality and Diversity in Sonic Measurement Spaces
- **分类: cs.SD; cs.NE**

- **简介: 该论文研究音效生成中的质量多样性（QD）算法，旨在解决传统方法依赖人工声学特征导致探索偏差的问题。通过引入无监督降维（PCA与自编码器），自动构建并动态重构声学行为空间，提升音效多样性与探索效率，实现无需人工干预的自动化声学发现。**

- **链接: [https://arxiv.org/pdf/2512.02783v1](https://arxiv.org/pdf/2512.02783v1)**

> **作者:** Björn Þór Jónsson; Çağrı Erdem; Stefano Fasciani; Kyrre Glette
>
> **摘要:** Digital sound synthesis presents the opportunity to explore vast parameter spaces containing millions of configurations. Quality diversity (QD) evolutionary algorithms offer a promising approach to harness this potential, yet their success hinges on appropriate sonic feature representations. Existing QD methods predominantly employ handcrafted descriptors or supervised classifiers, potentially introducing unintended exploration biases and constraining discovery to familiar sonic regions. This work investigates unsupervised dimensionality reduction methods for automatically defining and dynamically reconfiguring sonic behaviour spaces during QD search. We apply Principal Component Analysis (PCA) and autoencoders to project high-dimensional audio features onto structured grids for MAP-Elites, implementing dynamic reconfiguration through model retraining at regular intervals. Comparison across two experimental scenarios shows that automatic approaches achieve significantly greater diversity than handcrafted behaviour spaces while avoiding expert-imposed biases. Dynamic behaviour-space reconfiguration maintains evolutionary pressure and prevents stagnation, with PCA proving most effective among the dimensionality reduction techniques. These results contribute to automated sonic discovery systems capable of exploring vast parameter spaces without manual intervention or supervised training constraints.
>
---
#### [new 009] Continual Learning for Singing Voice Separation with Human in the Loop Adaptation
- **分类: cs.SD**

- **简介: 该论文针对歌唱语音分离任务，提出一种人机协同的持续学习框架。针对现有模型在真实场景中因音乐风格和乐器差异导致性能下降的问题，通过用户标记误检区域，实现模型在线微调，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2512.02432v1](https://arxiv.org/pdf/2512.02432v1)**

> **作者:** Ankur Gupta; Anshul Rai; Archit Bansal; Vipul Arora
>
> **备注:** Proceedings of the 26th International Symposium on Frontiers of Research in Speech and Music, 2021
>
> **摘要:** Deep learning-based works for singing voice separation have performed exceptionally well in the recent past. However, most of these works do not focus on allowing users to interact with the model to improve performance. This can be crucial when deploying the model in real-world scenarios where music tracks can vary from the original training data in both genre and instruments. In this paper, we present a deep learning-based interactive continual learning framework for singing voice separation that allows users to fine-tune the vocal separation model to conform it to new target songs. We use a U-Net-based base model architecture that produces a mask for separating vocals from the spectrogram, followed by a human-in-the-loop task where the user provides feedback by marking a few false positives, i.e., regions in the extracted vocals that should have been silence. We propose two continual learning algorithms. Experiments substantiate the improvement in singing voice separation performance by the proposed algorithms over the base model in intra-dataset and inter-dataset settings.
>
---
#### [new 010] Towards Language-Independent Face-Voice Association with Multimodal Foundation Models
- **分类: eess.AS; cs.SD; eess.IV**

- **简介: 该论文针对跨模态验证任务，解决多语言环境下语音与人脸关联的泛化问题。提出基于ImageBind-LoRA的foundation model方法，利用阿拉伯语数据微调，实现对未见语言（英语、德语）的强跨语言泛化，获FAME2026挑战赛第二名。**

- **链接: [https://arxiv.org/pdf/2512.02759v1](https://arxiv.org/pdf/2512.02759v1)**

> **作者:** Aref Farhadipour; Teodora Vukovic; Volker Dellwo
>
> **备注:** This paper presents the system description of the UZH-CL team for the FAME2026 Challenge at ICASSP 2026. Our model achieved second place in the final ranking
>
> **摘要:** This paper describes the UZH-CL system submitted to the FAME2026 Challenge. The challenge focuses on cross-modal verification under unique multilingual conditions, specifically unseen and unheard languages. Our approach investigates two distinct architectures, consisting of a baseline dual-encoder system trained from scratch using contrastive and orthogonal projection losses, and a foundation model approach leveraging ImageBind with LoRA. To address the data scarcity and language constraints of the challenge, we curated an external Arabic dataset from VoxBlink. Our best-performing system, ImageBind-LoRA, demonstrates remarkable cross-lingual generalization: despite being fine-tuned exclusively on Arabic audio, it achieved an EER of 24.73% on the evaluation set (English and German), securing 2nd place in the competition.
>
---
#### [new 011] Dialect Identification Using Resource-Efficient Fine-Tuning Approaches
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究方言识别任务，针对语音模型微调计算成本高、内存占用大的问题，探索适用于预训练语音模型的内存高效微调（MEFT）方法。以Whisper模型在KeSpeech数据集上识别六种普通话子方言为例，实验表明MEFT可降低73.25%显存使用，提速2.1倍，同时保持与传统方法相当的准确率。**

- **链接: [https://arxiv.org/pdf/2512.02074v1](https://arxiv.org/pdf/2512.02074v1)**

> **作者:** Zirui Lin; Haris Gulzar; Monnika Roslianna Busto; Akiko Masaki; Takeharu Eda; Kazuhiro Nakadai
>
> **备注:** Published in APSIPA ASC 2025
>
> **摘要:** Dialect Identification (DI) is a task to recognize different dialects within the same language from a speech signal. DI can help to improve the downstream speech related tasks even when speakers have a strong dialect. However, fine-tuning a speech model for tasks like DI is expensive in terms of computation cost and memory requirement. Recent studies have explored fine-tuning pre-trained speech models for tasks like DI using Parameter-Efficient Fine-Tuning (PEFT) methods, which offer parameter efficiency but limited improvement in memory efficiency and training speed. To address these challenges, we explore Memory-Efficient Fine-Tuning (MEFT) methods, originally proposed for language processing, and apply them to the general-purpose pre-trained speech model. We then comprehensively analyze the GPU memory usage and fine-tuning speed based on various MEFT methods. As a case study, we fine-tune the Whisper model to identify six Mandarin subdialects from the KeSpeech dataset, reducing GPU memory usage by up to 73.25% and accelerating training speed by a factor of 2.1, while maintaining accuracy comparable to vanilla fine-tuning and PEFT methods.
>
---
#### [new 012] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出文本条件选择性视频转音频（V2A）任务，旨在从多物体视频中分离出用户指定的声音。针对现有方法无法精准定位声源的问题，提出SelVA模型，通过文本提示引导视频编码器提取相关特征，并利用补充令牌增强跨注意力，实现语义与时间上的精准对齐。**

- **链接: [https://arxiv.org/pdf/2512.02650v1](https://arxiv.org/pdf/2512.02650v1)**

> **作者:** Junwon Lee; Juhan Nam; Jiyoung Lee
>
> **摘要:** This work introduces a new task, text-conditioned selective video-to-audio (V2A) generation, which produces only the user-intended sound from a multi-object video. This capability is especially crucial in multimedia production, where audio tracks are handled individually for each sound source for precise editing, mixing, and creative control. However, current approaches generate single source-mixed sounds at once, largely because visual features are entangled, and region cues or prompts often fail to specify the source. We propose SelVA, a novel text-conditioned V2A model that treats the text prompt as an explicit selector of target source and modulates video encoder to distinctly extract prompt-relevant video features. The proposed supplementary tokens promote cross-attention by suppressing text-irrelevant activations with efficient parameter tuning, yielding robust semantic and temporal grounding. SelVA further employs a self-augmentation scheme to overcome the lack of mono audio track supervision. We evaluate SelVA on VGG-MONOAUDIO, a curated benchmark of clean single-source videos for such a task. Extensive experiments and ablations consistently verify its effectiveness across audio quality, semantic alignment, and temporal synchronization. Code and demo are available at https://jnwnlee.github.io/selva-demo/.
>
---
#### [new 013] Spoken Conversational Agents with Large Language Models
- **分类: cs.CL; cs.MA; cs.NE; cs.SD; eess.AS**

- **简介: 该论文研究语音对话系统，聚焦将文本大模型适配至语音场景。解决从分步式到端到端系统的演进难题，开展跨模态对齐与联合训练，对比不同架构设计，提出可复现基准，推动语音助手在隐私、安全与评估方面的进展。**

- **链接: [https://arxiv.org/pdf/2512.02593v1](https://arxiv.org/pdf/2512.02593v1)**

> **作者:** Chao-Han Huck Yang; Andreas Stolcke; Larry Heck
>
> **备注:** Accepted to EMNLP 2025 Tutorial
>
> **摘要:** Spoken conversational agents are converging toward voice-native LLMs. This tutorial distills the path from cascaded ASR/NLU to end-to-end, retrieval-and vision-grounded systems. We frame adaptation of text LLMs to audio, cross-modal alignment, and joint speech-text training; review datasets, metrics, and robustness across accents and compare design choices (cascaded vs. E2E, post-ASR correction, streaming). We link industrial assistants to current open-domain and task-oriented agents, highlight reproducible baselines, and outline open problems in privacy, safety, and evaluation. Attendees leave with practical recipes and a clear systems-level roadmap.
>
---
#### [new 014] WhAM: Towards A Translative Model of Sperm Whale Vocalization
- **分类: cs.LG; cs.SD**

- **简介: 该论文提出WhAM，首个基于Transformer的精子鲸鸣叫生成模型，通过微调预训练模型，从任意音频提示生成高保真合成鸣叫。旨在解决自然动物语音生成难题，提升对鲸类交流的理解。模型在音律、社会单元等分类任务中表现优异，验证了其表征能力。**

- **链接: [https://arxiv.org/pdf/2512.02206v1](https://arxiv.org/pdf/2512.02206v1)**

> **作者:** Orr Paradise; Pranav Muralikrishnan; Liangyuan Chen; Hugo Flores García; Bryan Pardo; Roee Diamant; David F. Gruber; Shane Gero; Shafi Goldwasser
>
> **备注:** NeurIPS 2025
>
> **摘要:** Sperm whales communicate in short sequences of clicks known as codas. We present WhAM (Whale Acoustics Model), the first transformer-based model capable of generating synthetic sperm whale codas from any audio prompt. WhAM is built by finetuning VampNet, a masked acoustic token model pretrained on musical audio, using 10k coda recordings collected over the past two decades. Through iterative masked token prediction, WhAM generates high-fidelity synthetic codas that preserve key acoustic features of the source recordings. We evaluate WhAM's synthetic codas using Fréchet Audio Distance and through perceptual studies with expert marine biologists. On downstream classification tasks including rhythm, social unit, and vowel classification, WhAM's learned representations achieve strong performance, despite being trained for generation rather than classification. Our code is available at https://github.com/Project-CETI/wham
>
---
## 更新

#### [replaced 001] IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
- **分类: eess.AS**

- **简介: 该论文针对语音匿名化中的伪说话人生成问题，提出基于身份索引到向量映射的IDMap框架。解决了现有方法在伪说话人唯一性不足及大规模生成时计算成本高的问题，通过前馈结构实现高效、稳定的伪说话人生成，显著提升语音隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2511.06246v2](https://arxiv.org/pdf/2511.06246v2)**

> **作者:** Zeyan Liu; Liping Chen; Kong Aik Lee; Zhenhua Ling
>
> **备注:** The current version lacks critical validation components, which limits the comprehensiveness and reliability of the conclusions. It thus cannot fully meet the academic standards for publication
>
> **摘要:** Facilitated by the speech generation framework that disentangles speech into content, speaker, and prosody, voice anonymization is accomplished by substituting the original speaker embedding vector with that of a pseudo-speaker. In this framework, the pseudo-speaker generation forms a fundamental challenge. Current pseudo-speaker generation methods demonstrate limitations in the uniqueness of pseudo-speakers, consequently restricting their effectiveness in voice privacy protection. Besides, existing model-based methods suffer from heavy computation costs. Especially, in the large-scale scenario where a huge number of pseudo-speakers are generated, the limitations of uniqueness and computational inefficiency become more significant. To this end, this paper proposes a framework for pseudo-speaker generation, which establishes a mapping from speaker identity index to speaker vector in the feedforward architecture, termed IDMap. Specifically, the framework is specified into two models: IDMap-MLP and IDMap-Diff. Experiments were conducted on both small- and large-scale evaluation datasets. Small-scale evaluations on the LibriSpeech dataset validated the effectiveness of the proposed IDMap framework in enhancing the uniqueness of pseudo-speakers, thereby improving voice privacy protection, while at a reduced computational cost. Large-scale evaluations on the MLS and Common Voice datasets further justified the superiority of the IDMap framework regarding the stability of the voice privacy protection capability as the number of pseudo-speakers increased. Audio samples and open-source code can be found in https://github.com/VoicePrivacy/IDMap.
>
---
#### [replaced 002] IHearYou: Linking Acoustic Features to DSM-5 Depressive Behavior Indicators
- **分类: cs.SD**

- **简介: 该论文提出IHearYou系统，旨在通过分析家庭环境中被动采集的语音声学特征，自动识别抑郁症症状。针对现有诊断依赖主观评估、缺乏客观指标的问题，研究构建了基于DSM-5的声学特征-行为指标关联框架，实现本地化、可解释的抑郁状态监测，验证了其在真实数据上的可行性和一致性。**

- **链接: [https://arxiv.org/pdf/2511.14801v2](https://arxiv.org/pdf/2511.14801v2)**

> **作者:** Jonas Länzlinger; Katharina Müller; Burkhard Stiller; Bruno Rodrigues
>
> **摘要:** Depression affects over millions people worldwide, yet diagnosis still relies on subjective self-reports and interviews that may not capture authentic behavior. We present IHearYou, an approach to automated depression detection focused on speech acoustics. Using passive sensing in household environments, IHearYou extracts voice features and links them to DSM-5 (Diagnostic and Statistical Manual of Mental Disorders) indicators through a structured Linkage Framework instantiated for Major Depressive Disorder. The system runs locally to preserve privacy and includes a persistence schema and dashboard, presenting real-time throughput on a commodity laptop. To ensure reproducibility, we define a configuration-driven protocol with False Discovery Rate (FDR) correction and gender-stratified testing. Applied to the DAIC-WOZ dataset, this protocol reveals directionally consistent feature-indicator associations, while a TESS-based audio streaming experiment validates end-to-end feasibility. Our results show how passive voice sensing can be turned into explainable DSM-5 indicator scores, bridging the gap between black-box detection and clinically interpretable, on-device analysis.
>
---
#### [replaced 003] Text-Queried Audio Source Separation via Hierarchical Modeling
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文针对自然语言查询下的音频源分离任务，解决跨模态对齐与数据依赖问题。提出分层框架HSM-TSS，通过全局-局部语义分离与结构保持重建，实现高效、语义一致的音频分离，支持灵活声音操作。**

- **链接: [https://arxiv.org/pdf/2505.21025v2](https://arxiv.org/pdf/2505.21025v2)**

> **作者:** Xinlei Yin; Xiulian Peng; Xue Jiang; Zhiwei Xiong; Yan Lu
>
> **备注:** Accepted by TASLP
>
> **摘要:** Target audio source separation with natural language queries presents a promising paradigm for extracting arbitrary audio events through arbitrary text descriptions. Existing methods mainly face two challenges, the difficulty in jointly modeling acoustic-textual alignment and semantic-aware separation within a blindly-learned single-stage architecture, and the reliance on large-scale accurately-labeled training data to compensate for inefficient cross-modal learning and separation. To address these challenges, we propose a hierarchical decomposition framework, HSM-TSS, that decouples the task into global-local semantic-guided feature separation and structure-preserving acoustic reconstruction. Our approach introduces a dual-stage mechanism for semantic separation, operating on distinct global and local semantic feature spaces. We first perform global-semantic separation through a global semantic feature space aligned with text queries. A Q-Audio architecture is employed to align audio and text modalities, serving as pretrained global-semantic encoders. Conditioned on the predicted global feature, we then perform the second-stage local-semantic separation on AudioMAE features that preserve time-frequency structures, followed by acoustic reconstruction. We also propose an instruction processing pipeline to parse arbitrary text queries into structured operations, extraction or removal, coupled with audio descriptions, enabling flexible sound manipulation. Our method achieves state-of-the-art separation performance with data-efficient training while maintaining superior semantic consistency with queries in complex auditory scenes.
>
---
