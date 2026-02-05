# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion
- **分类: cs.SD**

- **简介: 该论文提出PFluxTTS，解决流匹配TTS中的稳定性与自然度平衡、跨语言语音克隆和音频质量不足问题，通过双解码器设计和改进声码器提升性能。**

- **链接: [https://arxiv.org/pdf/2602.04160v1](https://arxiv.org/pdf/2602.04160v1)**

> **作者:** Vikentii Pankov; Artem Gribul; Oktai Tatanov; Vladislav Proskurov; Yuliya Korotkova; Darima Mylzenova; Dmitrii Vypirailenko
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** We present PFluxTTS, a hybrid text-to-speech system addressing three gaps in flow-matching TTS: the stability-naturalness trade-off, weak cross-lingual voice cloning, and limited audio quality from low-rate mel features. Our contributions are: (1) a dual-decoder design combining duration-guided and alignment-free models through inference-time vector-field fusion; (2) robust cloning using a sequence of speech-prompt embeddings in a FLUX-based decoder, preserving speaker traits across languages without prompt transcripts; and (3) a modified PeriodWave vocoder with super-resolution to 48 kHz. On cross-lingual in-the-wild data, PFluxTTS clearly outperforms F5-TTS, FishSpeech, and SparkTTS, matches ChatterBox in naturalness (MOS 4.11) while achieving 23% lower WER (6.9% vs. 9.0%), and surpasses ElevenLabs in speaker similarity (+0.32 SMOS). The system remains robust in challenging scenarios where most open-source models fail, while requiring only short reference audio and no extra training. Audio demos are available at https://braskai.github.io/pfluxtts/
>
---
#### [new 002] Frontend Token Enhancement for Token-Based Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决噪声环境下语义令牌性能下降的问题。通过前端增强模型提升噪声语音中的干净令牌，实验表明波形到令牌的增强方法效果最佳。**

- **链接: [https://arxiv.org/pdf/2602.04217v1](https://arxiv.org/pdf/2602.04217v1)**

> **作者:** Takanori Ashihara; Shota Horiguchi; Kohei Matsuura; Tsubasa Ochiai; Marc Delcroix
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Discretized representations of speech signals are efficient alternatives to continuous features for various speech applications, including automatic speech recognition (ASR) and speech language models. However, these representations, such as semantic or phonetic tokens derived from clustering outputs of self-supervised learning (SSL) speech models, are susceptible to environmental noise, which can degrade backend task performance. In this work, we introduce a frontend system that estimates clean speech tokens from noisy speech and evaluate it on an ASR backend using semantic tokens. We consider four types of enhancement models based on their input/output domains: wave-to-wave, token-to-token, continuous SSL features-to-token, and wave-to-token. These models are trained independently of ASR backends. Experiments on the CHiME-4 dataset demonstrate that wave-to-token enhancement achieves the best performance among the frontends. Moreover, it mostly outperforms the ASR system based on continuous SSL features.
>
---
#### [new 003] Benchmarking Automatic Speech Recognition for Indian Languages in Agricultural Contexts
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决印度农业领域多语言ASR系统的性能评估问题。通过构建基准框架，分析不同语言和模型的表现，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.03868v1](https://arxiv.org/pdf/2602.03868v1)**

> **作者:** Chandrashekar M S; Vineet Singh; Lakshmi Pedapudi
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The digitization of agricultural advisory services in India requires robust Automatic Speech Recognition (ASR) systems capable of accurately transcribing domain-specific terminology in multiple Indian languages. This paper presents a benchmarking framework for evaluating ASR performance in agricultural contexts across Hindi, Telugu, and Odia languages. We introduce evaluation metrics including Agriculture Weighted Word Error Rate (AWWER) and domain-specific utility scoring to complement traditional metrics. Our evaluation of 10,934 audio recordings, each transcribed by up to 10 ASR models, reveals performance variations across languages and models, with Hindi achieving the best overall performance (WER: 16.2%) while Odia presents the greatest challenges (best WER: 35.1%, achieved only with speaker diarization). We characterize audio quality challenges inherent to real-world agricultural field recordings and demonstrate that speaker diarization with best-speaker selection can substantially reduce WER for multi-speaker recordings (upto 66% depending on the proportion of multi-speaker audio). We identify recurring error patterns in agricultural terminology and provide practical recommendations for improving ASR systems in low-resource agricultural domains. The study establishes baseline benchmarks for future agricultural ASR development.
>
---
#### [new 004] Fine-Grained Frame Modeling in Multi-head Self-Attention for Speech Deepfake Detection
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决如何准确识别语音中的细微伪造线索。提出FGFM方法，通过选择关键帧并进行跨层优化，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.04702v1](https://arxiv.org/pdf/2602.04702v1)**

> **作者:** Tuan Dat Phuong; Duc-Tuan Truong; Long-Vu Hoang; Trang Nguyen Thi Thu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Transformer-based models have shown strong performance in speech deepfake detection, largely due to the effectiveness of the multi-head self-attention (MHSA) mechanism. MHSA provides frame-level attention scores, which are particularly valuable because deepfake artifacts often occur in small, localized regions along the temporal dimension of speech. This makes fine-grained frame modeling essential for accurately detecting subtle spoofing cues. In this work, we propose fine-grained frame modeling (FGFM) for MHSA-based speech deepfake detection, where the most informative frames are first selected through a multi-head voting (MHV) module. These selected frames are then refined via a cross-layer refinement (CLR) module to enhance the model's ability to learn subtle spoofing cues. Experimental results demonstrate that our method outperforms the baseline model and achieves Equal Error Rate (EER) of 0.90%, 1.88%, and 6.64% on the LA21, DF21, and ITW datasets, respectively. These consistent improvements across multiple benchmarks highlight the effectiveness of our fine-grained modeling for robust speech deepfake detection.
>
---
#### [new 005] Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection
- **分类: eess.AS; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频亮点检测任务，旨在解决音频信息利用不足的问题。提出双路径音频编码器，分别捕捉语义和动态特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.03891v1](https://arxiv.org/pdf/2602.03891v1)**

> **作者:** Seohyun Joo; Yoori Oh
>
> **备注:** 5 pages, 2 figures, to appear in ICASSP 2026
>
> **摘要:** Audio-visual video highlight detection aims to automatically identify the most salient moments in videos by leveraging both visual and auditory cues. However, existing models often underutilize the audio modality, focusing on high-level semantic features while failing to fully leverage the rich, dynamic characteristics of sound. To address this limitation, we propose a novel framework, Dual-Pathway Audio Encoders for Video Highlight Detection (DAViHD). The dual-pathway audio encoder is composed of a semantic pathway for content understanding and a dynamic pathway that captures spectro-temporal dynamics. The semantic pathway extracts high-level information by identifying the content within the audio, such as speech, music, or specific sound events. The dynamic pathway employs a frequency-adaptive mechanism as time evolves to jointly model these dynamics, enabling it to identify transient acoustic events via salient spectral bands and rapid energy changes. We integrate the novel audio encoder into a full audio-visual framework and achieve new state-of-the-art performance on the large-scale Mr.HiSum benchmark. Our results demonstrate that a sophisticated, dual-faceted audio representation is key to advancing the field of highlight detection.
>
---
#### [new 006] BASS: Benchmarking Audio LMs for Musical Structure and Semantic Reasoning
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出BASS，用于评估音频语言模型在音乐结构和语义推理方面的能力，解决音乐理解任务中的多模态分析问题。**

- **链接: [https://arxiv.org/pdf/2602.04085v1](https://arxiv.org/pdf/2602.04085v1)**

> **作者:** Min Jang; Orevaoghene Ahia; Nazif Tamer; Sachin Kumar; Yulia Tsvetkov; Noah A. Smith
>
> **摘要:** Music understanding is a complex task that often requires reasoning over both structural and semantic elements of audio. We introduce BASS, designed to evaluate music understanding and reasoning in audio language models across four broad categories: structural segmentation, lyric transcription, musicological analysis, and artist collaboration. BASS comprises 2658 questions spanning 12 tasks, 1993 unique songs and covering over 138 hours of music from a wide range of genres and tracks, crafted to assess musicological knowledge and reasoning in real-world scenarios. We evaluate 14 open-source and frontier multimodal LMs, finding that even state-of-the-art models struggle on higher-level reasoning tasks such as structural segmentation and artist collaboration, while performing best on lyric transcription. Our analysis reveals that current models leverage linguistic priors effectively but remain limited in reasoning over musical structure, vocal, and musicological attributes. BASS provides an evaluation framework with widespread applications in music recommendation and search and has the potential to guide the development of audio LMs.
>
---
#### [new 007] Audio ControlNet for Fine-Grained Audio Generation and Editing
- **分类: cs.SD; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于文本到音频生成任务，旨在解决音频属性控制不足的问题。通过提出T2A-Adapter模型，实现对音量、音高和声音事件的精确控制，并扩展至音频编辑应用。**

- **链接: [https://arxiv.org/pdf/2602.04680v1](https://arxiv.org/pdf/2602.04680v1)**

> **作者:** Haina Zhu; Yao Xiao; Xiquan Li; Ziyang Ma; Jianwei Yu; Bowen Zhang; Mingqi Yang; Xie Chen
>
> **摘要:** We study the fine-grained text-to-audio (T2A) generation task. While recent models can synthesize high-quality audio from text descriptions, they often lack precise control over attributes such as loudness, pitch, and sound events. Unlike prior approaches that retrain models for specific control types, we propose to train ControlNet models on top of pre-trained T2A backbones to achieve controllable generation over loudness, pitch, and event roll. We introduce two designs, T2A-ControlNet and T2A-Adapter, and show that the T2A-Adapter model offers a more efficient structure with strong control ability. With only 38M additional parameters, T2A-Adapter achieves state-of-the-art performance on the AudioSet-Strong in both event-level and segment-level F1 scores. We further extend this framework to audio editing, proposing T2A-Editor for removing and inserting audio events at time locations specified by instructions. Models, code, dataset pipelines, and benchmarks will be released to support future research on controllable audio generation and editing.
>
---
#### [new 008] Speaker-Aware Simulation Improves Conversational Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决对话语音识别中数据不足的问题。通过构建仿真的多说话人对话数据，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2602.04776v1](https://arxiv.org/pdf/2602.04776v1)**

> **作者:** Máté Gedeon; Péter Mihajlik
>
> **摘要:** Automatic speech recognition (ASR) for conversational speech remains challenging due to the limited availability of large-scale, well-annotated multi-speaker dialogue data and the complex temporal dynamics of natural interactions. Speaker-aware simulated conversations (SASC) offer an effective data augmentation strategy by transforming single-speaker recordings into realistic multi-speaker dialogues. However, prior work has primarily focused on English data, leaving questions about the applicability to lower-resource languages. In this paper, we adapt and implement the SASC framework for Hungarian conversational ASR. We further propose C-SASC, an extended variant that incorporates pause modeling conditioned on utterance duration, enabling a more faithful representation of local temporal dependencies observed in human conversation while retaining the simplicity and efficiency of the original approach. We generate synthetic Hungarian dialogues from the BEA-Large corpus and combine them with real conversational data for ASR training. Both SASC and C-SASC are evaluated extensively under a wide range of simulation configurations, using conversational statistics derived from CallHome, BEA-Dialogue, and GRASS corpora. Experimental results show that speaker-aware conversational simulation consistently improves recognition performance over naive concatenation-based augmentation. While the additional duration conditioning in C-SASC yields modest but systematic gains--most notably in character-level error rates--its effectiveness depends on the match between source conversational statistics and the target domain. Overall, our findings confirm the robustness of speaker-aware conversational simulation for Hungarian ASR and highlight the benefits and limitations of increasingly detailed temporal modeling in synthetic dialogue generation.
>
---
#### [new 009] UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization
- **分类: cs.SD**

- **简介: 该论文提出UniAudio 2.0，解决音频语言模型的表示和泛化问题。通过分层音频编码器和统一生成架构，提升理解与生成效果，并在多种任务中展示出色性能。**

- **链接: [https://arxiv.org/pdf/2602.04683v1](https://arxiv.org/pdf/2602.04683v1)**

> **作者:** Dongchao Yang; Yuanyuan Wang; Dading Chong; Songxiang Liu; Xixin Wu; Helen Meng
>
> **摘要:** We study two foundational problems in audio language models: (1) how to design an audio tokenizer that can serve as an intermediate representation for both understanding and generation; and (2) how to build an audio foundation model that generalizes in few-shot and zero-shot settings, analogous to large language models. To this end, we make the following two contributions. First, we propose ReasoningCodec, a discrete audio codec that factorizes audio into (i) reasoning tokens, which encode text-aligned, high-level analysis and planning representations for audio understanding and hierarchical generation, and (ii) reconstruction tokens, which encode semantic-rich acoustic cues for high-fidelity waveform reconstruction. This design achieves understanding performance comparable to strong continuous representations while improving generation quality and reconstruction fidelity over prior discrete tokenizers. Second, we introduce a unified autoregressive architecture for text and audio, together with multi-stage training and multi-task data construction. Using this framework, we train UniAudio 2.0 on 100B text tokens and 60B audio tokens. Across a wide range of speech, sound, and music tasks, UniAudio 2.0 performs competitively on in-domain evaluations and demonstrates strong few-shot and zero-shot generalization to unseen tasks. Demo, code, and checkpoints will be available at \href{https://dongchaoyang.top/UniAudio2Demo/}{https://dongchaoyang.top/UniAudio2Demo/}.
>
---
#### [new 010] HoliAntiSpoof: Audio LLM for Holistic Speech Anti-Spoofing
- **分类: cs.SD**

- **简介: 该论文属于语音防欺骗任务，旨在解决传统方法忽视语义影响的问题。提出HoliAntiSpoof框架，通过文本生成实现多属性联合分析。**

- **链接: [https://arxiv.org/pdf/2602.04535v1](https://arxiv.org/pdf/2602.04535v1)**

> **作者:** Xuenan Xu; Yiming Ren; Liwei Liu; Wen Wu; Baoxiang Li; Chaochao Lu; Shuai Wang; Chao Zhang
>
> **摘要:** Recent advances in speech synthesis and editing have made speech spoofing increasingly challenging. However, most existing methods treat spoofing as binary classification, overlooking that diverse spoofing techniques manipulate multiple, coupled speech attributes and their semantic effects. In this paper, we introduce HoliAntiSpoof, the first audio large language model (ALLM) framework for holistic speech anti-spoofing analysis. HoliAntiSpoof reformulates spoofing analysis as a unified text generation task, enabling joint reasoning over spoofing methods, affected speech attributes, and their semantic impacts. To support semantic-level analysis, we introduce DailyTalkEdit, a new anti-spoofing benchmark that simulates realistic conversational manipulations and provides annotations of semantic influence. Extensive experiments demonstrate that HoliAntiSpoof outperforms conventional baselines across multiple settings, while preliminary results show that in-context learning further improves out-of-domain generalization. These findings indicate that ALLMs not only enhance speech spoofing detection performance but also enable interpretable analysis of spoofing behaviors and their semantic effects, pointing towards more trustworthy and explainable speech security. Data and code are publicly available.
>
---
#### [new 011] LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation in Multi-Turn Spoken Dialogues
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音安全评估任务，旨在解决多轮对话中有害内容检测问题。通过构建基准数据集，评估大音频语言模型的判别能力，分析不同模态与架构的优劣。**

- **链接: [https://arxiv.org/pdf/2602.04796v1](https://arxiv.org/pdf/2602.04796v1)**

> **作者:** Amir Ivry; Shinji Watanabe
>
> **摘要:** Spoken dialogues with and between voice agents are becoming increasingly common, yet assessing them for their socially harmful content such as violence, harassment, and hate remains text-centric and fails to account for audio-specific cues and transcription errors. We present LALM-as-a-Judge, the first controlled benchmark and systematic study of large audio-language models (LALMs) as safety judges for multi-turn spoken dialogues. We generate 24,000 unsafe and synthetic spoken dialogues in English that consist of 3-10 turns, by having a single dialogue turn including content with one of 8 harmful categories (e.g., violence) and on one of 5 grades, from very mild to severe. On 160 dialogues, 5 human raters confirmed reliable unsafe detection and a meaningful severity scale. We benchmark three open-source LALMs: Qwen2-Audio, Audio Flamingo 3, and MERaLiON as zero-shot judges that output a scalar safety score in [0,1] across audio-only, transcription-only, or multimodal inputs, along with a transcription-only LLaMA baseline. We measure the judges' sensitivity to detecting unsafe content, the specificity in ordering severity levels, and the stability of the score in dialogue turns. Results reveal architecture- and modality-dependent trade-offs: the most sensitive judge is also the least stable across turns, while stable configurations sacrifice detection of mild harmful content. Transcription quality is a key bottleneck: Whisper-Large may significantly reduce sensitivity for transcription-only modes, while largely preserving severity ordering. Audio becomes crucial when paralinguistic cues or transcription fidelity are category-critical. We summarize all findings and provide actionable guidance for practitioners.
>
---
#### [new 012] Universal Robust Speech Adaptation for Cross-Domain Speech Recognition and Enhancement
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别与增强任务，旨在解决域迁移导致的性能下降问题。提出URSA-GAN框架，通过双编码器和动态扰动提升模型在噪声和信道失配下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04307v1](https://arxiv.org/pdf/2602.04307v1)**

> **作者:** Chien-Chun Wang; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech and Language Processing (IEEE TASLP)
>
> **摘要:** Pre-trained models for automatic speech recognition (ASR) and speech enhancement (SE) have exhibited remarkable capabilities under matched noise and channel conditions. However, these models often suffer from severe performance degradation when confronted with domain shifts, particularly in the presence of unseen noise and channel distortions. In view of this, we in this paper present URSA-GAN, a unified and domain-aware generative framework specifically designed to mitigate mismatches in both noise and channel conditions. URSA-GAN leverages a dual-embedding architecture that consists of a noise encoder and a channel encoder, each pre-trained with limited in-domain data to capture domain-relevant representations. These embeddings condition a GAN-based speech generator, facilitating the synthesis of speech that is acoustically aligned with the target domain while preserving phonetic content. To enhance generalization further, we propose dynamic stochastic perturbation, a novel regularization technique that introduces controlled variability into the embeddings during generation, promoting robustness to unseen domains. Empirical results demonstrate that URSA-GAN effectively reduces character error rates in ASR and improves perceptual metrics in SE across diverse noisy and mismatched channel scenarios. Notably, evaluations on compound test conditions with both channel and noise degradations confirm the generalization ability of URSA-GAN, yielding relative improvements of 16.16% in ASR performance and 15.58% in SE metrics.
>
---
#### [new 013] Decoding Ambiguous Emotions with Test-Time Scaling in Audio-Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于情感识别任务，旨在解决语音中模糊情绪的识别问题。通过引入测试时缩放技术，评估不同模型在模糊情绪数据上的表现，探索模型能力与情绪模糊性的关系。**

- **链接: [https://arxiv.org/pdf/2602.03873v1](https://arxiv.org/pdf/2602.03873v1)**

> **作者:** Hong Jia; Weibin Li; Jingyao Wu; Xiaofeng Yu; Yan Gao; Jintao Cheng; Xiaoyu Tang; Feng Xia; Ting Dang
>
> **摘要:** Emotion recognition from human speech is a critical enabler for socially aware conversational AI. However, while most prior work frames emotion recognition as a categorical classification problem, real-world affective states are often ambiguous, overlapping, and context-dependent, posing significant challenges for both annotation and automatic modeling. Recent large-scale audio language models (ALMs) offer new opportunities for nuanced affective reasoning without explicit emotion supervision, but their capacity to handle ambiguous emotions remains underexplored. At the same time, advances in inference-time techniques such as test-time scaling (TTS) have shown promise for improving generalization and adaptability in hard NLP tasks, but their relevance to affective computing is still largely unknown. In this work, we introduce the first benchmark for ambiguous emotion recognition in speech with ALMs under test-time scaling. Our evaluation systematically compares eight state-of-the-art ALMs and five TTS strategies across three prominent speech emotion datasets. We further provide an in-depth analysis of the interaction between model capacity, TTS, and affective ambiguity, offering new insights into the computational and representational challenges of ambiguous emotion understanding. Our benchmark establishes a foundation for developing more robust, context-aware, and emotionally intelligent speech-based AI systems, and highlights key future directions for bridging the gap between model assumptions and the complexity of real-world human emotion.
>
---
#### [new 014] DementiaBank-Emotion: A Multi-Rater Emotion Annotation Corpus for Alzheimer's Disease Speech (Version 1.0)
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于情感标注任务，旨在研究阿尔茨海默病患者情绪表达。通过多标注者标注语料，分析其情绪特征与健康对照组的差异。**

- **链接: [https://arxiv.org/pdf/2602.04247v1](https://arxiv.org/pdf/2602.04247v1)**

> **作者:** Cheonkam Jeong; Jessica Liao; Audrey Lu; Yutong Song; Christopher Rashidian; Donna Krogh; Erik Krogh; Mahkameh Rasouli; Jung-Ah Lee; Nikil Dutt; Lisa M Gibbs; David Sultzer; Julie Rousseau; Jocelyn Ludlow; Margaret Galvez; Alexander Nuth; Chet Khay; Sabine Brunswicker; Adeline Nyamathi
>
> **备注:** Accepted at HeaLING Workshop @ EACL 2026. 9 pages, 3 figures, 8 tables
>
> **摘要:** We present DementiaBank-Emotion, the first multi-rater emotion annotation corpus for Alzheimer's disease (AD) speech. Annotating 1,492 utterances from 108 speakers for Ekman's six basic emotions and neutral, we find that AD patients express significantly more non-neutral emotions (16.9%) than healthy controls (5.7%; p < .001). Exploratory acoustic analysis suggests a possible dissociation: control speakers showed substantial F0 modulation for sadness (Delta = -3.45 semitones from baseline), whereas AD speakers showed minimal change (Delta = +0.11 semitones; interaction p = .023), though this finding is based on limited samples (sadness: n=5 control, n=15 AD) and requires replication. Within AD speech, loudness differentiates emotion categories, indicating partially preserved emotion-prosody mappings. We release the corpus, annotation guidelines, and calibration workshop materials to support research on emotion recognition in clinical populations.
>
---
#### [new 015] Audit After Segmentation: Reference-Free Mask Quality Assessment for Language-Referred Audio-Visual Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出MQA-RefAVS任务，解决无参考的音频-视觉分割质量评估问题。构建基准数据集并设计MQ-Auditor模型，实现分割掩码的定量与定性评估。**

- **链接: [https://arxiv.org/pdf/2602.03892v1](https://arxiv.org/pdf/2602.03892v1)**

> **作者:** Jinxing Zhou; Yanghao Zhou; Yaoting Wang; Zongyan Han; Jiaqi Ma; Henghui Ding; Rao Muhammad Anwer; Hisham Cholakkal
>
> **摘要:** Language-referred audio-visual segmentation (Ref-AVS) aims to segment target objects described by natural language by jointly reasoning over video, audio, and text. Beyond generating segmentation masks, providing rich and interpretable diagnoses of mask quality remains largely underexplored. In this work, we introduce Mask Quality Assessment in the Ref-AVS context (MQA-RefAVS), a new task that evaluates the quality of candidate segmentation masks without relying on ground-truth annotations as references at inference time. Given audio-visual-language inputs and each provided segmentation mask, the task requires estimating its IoU with the unobserved ground truth, identifying the corresponding error type, and recommending an actionable quality-control decision. To support this task, we construct MQ-RAVSBench, a benchmark featuring diverse and representative mask error modes that span both geometric and semantic issues. We further propose MQ-Auditor, a multimodal large language model (MLLM)-based auditor that explicitly reasons over multimodal cues and mask information to produce quantitative and qualitative mask quality assessments. Extensive experiments demonstrate that MQ-Auditor outperforms strong open-source and commercial MLLMs and can be integrated with existing Ref-AVS systems to detect segmentation failures and support downstream segmentation improvement. Data and codes will be released at https://github.com/jasongief/MQA-RefAVS.
>
---
## 更新

#### [replaced 001] VidTune: Creating Video Soundtracks with Generative Music and Contextual Thumbnails
- **分类: cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 论文提出VidTune系统，解决视频配乐匹配视频情绪和叙事的问题。通过生成音乐并结合上下文缩略图，帮助创作者高效选择和调整配乐。**

- **链接: [https://arxiv.org/pdf/2601.12180v2](https://arxiv.org/pdf/2601.12180v2)**

> **作者:** Mina Huh; C. Ailie Fraser; Dingzeyu Li; Mira Dontcheva; Bryan Wang
>
> **备注:** Accepted to CHI 2026
>
> **摘要:** Music shapes the tone of videos, yet creators often struggle to find soundtracks that match their video's mood and narrative. Recent text-to-music models let creators generate music from text prompts, but our formative study (N=8) shows creators struggle to construct diverse prompts, quickly review and compare tracks, and understand their impact on the video. We present VidTune, a system that supports soundtrack creation by generating diverse music options from a creator's prompt and producing contextual thumbnails for rapid review. VidTune extracts representative video subjects to ground thumbnails in context, maps each track's valence and energy onto visual cues like color and brightness, and depicts prominent genres and instruments. Creators can refine tracks through natural language edits, which VidTune expands into new generations. In a controlled user study (N=12) and an exploratory case study (N=6), participants found VidTune helpful for efficiently reviewing and comparing music options and described the process as playful and enriching.
>
---
#### [replaced 002] Noise-Conditioned Mixture-of-Experts Framework for Robust Speaker Verification
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于语音识别任务，解决噪声环境下说话人验证问题。提出一种噪声条件的专家混合框架，通过分解特征空间提升模型鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.18533v2](https://arxiv.org/pdf/2510.18533v2)**

> **作者:** Bin Gu; Haitao Zhao; Jibo Wei
>
> **摘要:** Robust speaker verification under noisy conditions remains an open challenge. Conventional deep learning methods learn a robust unified speaker representation space against diverse background noise and achieve significant improvement. In contrast, this paper presents a noise-conditioned mixture-ofexperts framework that decomposes the feature space into specialized noise-aware subspaces for speaker verification. Specifically, we propose a noise-conditioned expert routing mechanism, a universal model based expert specialization strategy, and an SNR-decaying curriculum learning protocol, collectively improving model robustness and generalization under diverse noise conditions. The proposed method can automatically route inputs to expert networks based on noise information derived from the inputs, where each expert targets distinct noise characteristics while preserving speaker identity information. Comprehensive experiments demonstrate consistent superiority over baselines, confirming that explicit noise-dependent feature modeling significantly enhances robustness without sacrificing verification accuracy.
>
---
#### [replaced 003] MOSA: Mixtures of Simple Adapters Outperform Monolithic Approaches in LLM-based Multilingual ASR
- **分类: eess.AS**

- **简介: 该论文属于多语言语音识别任务，解决数据稀缺与语言间参数干扰问题。提出MOSA架构，通过简单适配器混合提升性能与参数效率。**

- **链接: [https://arxiv.org/pdf/2508.18998v3](https://arxiv.org/pdf/2508.18998v3)**

> **作者:** Junjie Li; Jing Peng; Yangui Fang; Shuai Wang; Kai Yu
>
> **备注:** 5 pages, 3 figures, accepted to ICASSP 2026
>
> **摘要:** LLM-based ASR overcomes multilingual data scarcity by projecting speech representations into the LLM space to leverage its robust semantic and reasoning capabilities. However, while previous approaches typically enhance performance by scaling data or model parameters, a single projector often struggles to effectively align representations across different languages. In this work, we propose an MoE-based projector named MOSA (Mixture of Simple Adapters). By aggregating multiple simple adapters, this architecture enables different experts to specialize in learning either language-shared or language-specific knowledge. This approach not only mitigates parameter interference between languages but also facilitates positive transfer from high-resource to low-resource languages, effectively alleviating data scarcity issues. Experimental results demonstrate that MOSA-Base achieves a 15.4% relative reduction in average WER compared to the Ideal-LLM Base, consistently outperforming it across all languages. Notably, MOSA achieves a 13.3% WER reduction over the Ideal-LLM Base while utilizing only 60% of its parameters. These findings highlight MOSA's superior parameter efficiency and robustness against data imbalance, suggesting that a mixture of simple adapters is more suitable for multilingual LLM-based ASR than complex single-adapter designs.
>
---
#### [replaced 004] Efficient Solutions for Mitigating Initialization Bias in Unsupervised Self-Adaptive Auditory Attention Decoding
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于脑机接口中的听觉注意力解码任务，旨在解决无监督自适应解码中的初始化偏差问题。提出三种高效算法，在保持性能的同时降低计算成本。**

- **链接: [https://arxiv.org/pdf/2509.14764v2](https://arxiv.org/pdf/2509.14764v2)**

> **作者:** Yuanyuan Yao; Simon Geirnaert; Tinne Tuytelaars; Alexander Bertrand
>
> **摘要:** Decoding the attended speaker in a multi-speaker environment from electroencephalography (EEG) has attracted growing interest in recent years, with neuro-steered hearing devices as a driver application. Current approaches typically rely on ground-truth labels of the attended speaker during training, necessitating calibration sessions for each user and each EEG set-up to achieve optimal performance. While unsupervised self-adaptive auditory attention decoding (AAD) for stimulus reconstruction has been developed to eliminate the need for labeled data, it suffers from an initialization bias that can compromise performance. Although an unbiased variant has been proposed to address this limitation, it introduces substantial computational complexity that scales with data size. This paper presents three computationally efficient alternatives that achieve comparable performance, but with a significantly lower and constant computational cost. The code for the proposed algorithms is available at https://github.com/YYao-42/Unsupervised_AAD.
>
---
#### [replaced 005] EMO-TTA: Improving Test-Time Adaptation of Audio-Language Models for Speech Emotion Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音情感识别任务，解决测试时分布变化导致模型性能下降的问题。提出Emo-TTA框架，通过统计适应提升模型在域外场景的准确性。**

- **链接: [https://arxiv.org/pdf/2509.25495v2](https://arxiv.org/pdf/2509.25495v2)**

> **作者:** Jiacheng Shi; Hongfei Du; Y. Alicia Hong; Ye Gao
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speech emotion recognition (SER) with audio-language models (ALMs) remains vulnerable to distribution shifts at test time, leading to performance degradation in out-of-domain scenarios. Test-time adaptation (TTA) provides a promising solution but often relies on gradient-based updates or prompt tuning, limiting flexibility and practicality. We propose Emo-TTA, a lightweight, training-free adaptation framework that incrementally updates class-conditional statistics via an Expectation-Maximization procedure for explicit test-time distribution estimation, using ALM predictions as priors. Emo-TTA operates on individual test samples without modifying model weights. Experiments on six out-of-domain SER benchmarks show consistent accuracy improvements over prior TTA baselines, demonstrating the effectiveness of statistical adaptation in aligning model predictions with evolving test distributions.
>
---
#### [replaced 006] A framework for diffuseness evaluation using a tight-frame microphone array configuration
- **分类: eess.AS**

- **简介: 该论文属于声场分析任务，旨在解决不同麦克风阵列下声场方向与扩散度的评估问题。通过提出紧框架阵列，实现一致的扩散度评估与方向估计。**

- **链接: [https://arxiv.org/pdf/2510.22183v3](https://arxiv.org/pdf/2510.22183v3)**

> **作者:** Akira Omoto
>
> **备注:** 16 pages including 16 files: This version has been substantially revised in response to reviewers' comments, with clarified theoretical assumptions and extended comparative evaluations
>
> **摘要:** This work presents a unified framework for estimating both sound-field direction and diffuseness using practical microphone arrays with different spatial configurations. Building on covariance-based diffuseness models, we formulate a velocity-only covariance approach that enables consistent diffuseness evaluation across heterogeneous array geometries without requiring mode whitening or spherical-harmonic decomposition. Three array types -- an A-format array, a rigid-sphere array, and a newly proposed tight-frame array -- are modeled and compared through both simulations and measurement-based experiments. The results show that the tight-frame configuration achieves near-isotropic directional sampling and reproduces diffuseness characteristics comparable to those of higher-order spherical arrays, while maintaining a compact physical structure. We further examine the accuracy of direction-of-arrival estimation based on acoustic intensity within the same framework. These findings connect theoretical diffuseness analysis with implementable array designs and support the development of robust, broadband methods for spatial-sound-field characterization.
>
---
#### [replaced 007] EDNet: A Versatile Speech Enhancement Framework with Gating Mamba Mechanism and Phase Shift-Invariant Training
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出EDNet框架，用于语音增强任务，解决噪声、混响等多类型畸变问题。通过门控Mamba模块和相位不变训练策略，实现掩码与映射的自适应结合。**

- **链接: [https://arxiv.org/pdf/2506.16231v2](https://arxiv.org/pdf/2506.16231v2)**

> **作者:** Doyeop Kwak; Youngjoon Jang; Seongyu Kim; Joon Son Chung
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing. Copyright IEEE. The final version will appear in IEEE Xplore
>
> **摘要:** Speech signals in real-world environments are frequently affected by various distortions such as additive noise, reverberation, and bandwidth limitation, which may appear individually or in combination. Traditional speech enhancement methods typically rely on either masking, which focuses on suppressing non-speech components while preserving observable structure, or mapping, which seeks to recover clean speech through direct transformation of the input. Each approach offers strengths in specific scenarios but may be less effective outside its target conditions. We propose the Erase and Draw Network (EDNet), a versatile speech enhancement framework designed to handle a broad range of distortion types without prior assumptions about task or input characteristics. EDNet consists of two main components: (1) the Gating Mamba (GM) module, which adaptively combines masking and mapping through a learnable gating mechanism that selects between suppression (Erase) and reconstruction (Draw) based on local signal features, and (2) Phase Shift-Invariant Training (PSIT), a shift tolerant supervision strategy that improves phase estimation by enabling dynamic alignment during training while remaining compatible with standard loss functions. Experimental results on denoising, dereverberation, bandwidth extension, and multi distortion enhancement tasks show that EDNet consistently achieves strong performance across conditions, demonstrating its architectural flexibility and adaptability to diverse task settings.
>
---
#### [replaced 008] Adapting Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，解决重叠语音转录问题。提出一种结合说话人建模与序列输出训练的模型，通过联合解码提升识别效果。**

- **链接: [https://arxiv.org/pdf/2510.03723v2](https://arxiv.org/pdf/2510.03723v2)**

> **作者:** Martin Kocour; Martin Karafiat; Alexander Polok; Dominik Klement; Lukáš Burget; Jan Černocký
>
> **摘要:** We propose a speaker-attributed (SA) Whisper-based model for multi-talker speech recognition that combines target-speaker modeling with serialized output training (SOT). Our approach leverages a Diarization-Conditioned Whisper (DiCoW) encoder to extract target-speaker embeddings, which are concatenated into a single representation and passed to a shared decoder. This enables the model to transcribe overlapping speech as a serialized output stream with speaker tags and timestamps. In contrast to target-speaker ASR systems such as DiCoW, which decode each speaker separately, our approach performs joint decoding, allowing the decoder to condition on the context of all speakers simultaneously. Experiments show that the model outperforms existing SOT-based approaches and surpasses DiCoW on multi-talker mixtures (e.g., LibriMix).
>
---
#### [replaced 009] Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决固定帧率导致的序列过长问题。提出DyCAST，通过动态帧率和字符对齐实现更高效的语音令牌化。**

- **链接: [https://arxiv.org/pdf/2601.23174v2](https://arxiv.org/pdf/2601.23174v2)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Neural audio codecs are at the core of modern conversational speech technologies, converting continuous speech into sequences of discrete tokens that can be processed by LLMs. However, existing codecs typically operate at fixed frame rates, allocating tokens uniformly in time and producing unnecessarily long sequences. In this work, we introduce DyCAST, a Dynamic Character-Aligned Speech Tokenizer that enables variable-frame-rate tokenization through soft character-level alignment and explicit duration modeling. DyCAST learns to associate tokens with character-level linguistic units during training and supports alignment-free inference with direct control over token durations at decoding time. To improve speech resynthesis quality at low frame rates, we further introduce a retrieval-augmented decoding mechanism that enhances reconstruction fidelity without increasing bitrate. Experiments show that DyCAST achieves competitive speech resynthesis quality and downstream performance while using significantly fewer tokens than fixed-frame-rate codecs. Code and checkpoints will be released publicly at https://github.com/lucadellalib/dycast.
>
---
#### [replaced 010] WAXAL: A Large-Scale Multilingual African Language Speech Corpus
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文介绍WAXAL，一个针对21种非洲语言的大规模语音数据集，旨在解决低资源语言在语音技术中的不足，促进包容性技术发展。**

- **链接: [https://arxiv.org/pdf/2602.02734v2](https://arxiv.org/pdf/2602.02734v2)**

> **作者:** Abdoulaye Diack; Perry Nelson; Kwaku Agbesi; Angela Nakalembe; MohamedElfatih MohamedKhair; Vusumuzi Dube; Tavonga Siyavora; Subhashini Venugopalan; Jason Hickey; Uche Okonkwo; Abhishek Bapna; Isaac Wiafe; Raynard Dodzi Helegah; Elikem Doe Atsakpo; Charles Nutrokpor; Fiifi Baffoe Payin Winful; Kafui Kwashie Solaga; Jamal-Deen Abdulai; Akon Obu Ekpezu; Audace Niyonkuru; Samuel Rutunda; Boris Ishimwe; Michael Melese; Engineer Bainomugisha; Joyce Nakatumba-Nabende; Andrew Katumba; Claire Babirye; Jonathan Mukiibi; Vincent Kimani; Samuel Kibacia; James Maina; Fridah Emmah; Ahmed Ibrahim Shekarau; Ibrahim Shehu Adamu; Yusuf Abdullahi; Howard Lakougna; Bob MacDonald; Hadar Shemtov; Aisha Walcott-Bryant; Moustapha Cisse; Avinatan Hassidim; Jeff Dean; Yossi Matias
>
> **备注:** Initial dataset release
>
> **摘要:** The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 21 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with over 180 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at https://huggingface.co/datasets/google/WaxalNLP under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages.
>
---
#### [replaced 011] PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升可穿戴设备的语音质量。针对单通道环境下区分目标语音与干扰语音的难题，提出PAS-SE方法，结合个性化和辅助传感器信息，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2509.20875v2](https://arxiv.org/pdf/2509.20875v2)**

> **作者:** Mattes Ohlenbusch; Mikolaj Kegler; Marko Stamenovic
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speech enhancement for voice pickup in hearables aims to improve the user's voice by suppressing noise and interfering talkers, while maintaining own-voice quality. For single-channel methods, it is particularly challenging to distinguish the target from interfering talkers without additional context. In this paper, we compare two strategies to resolve this ambiguity: personalized speech enhancement (PSE), which uses enrollment utterances to represent the target, and auxiliary-sensor speech enhancement (AS-SE), which uses in-ear microphones as additional input. We evaluate the strategies on two public datasets, employing different auxiliary sensor arrays, to investigate their cross-dataset generalization. We propose training-time augmentations to facilitate cross-dataset generalization of AS-SE systems. We also show that combining PSE and AS-SE (PAS-SE) provides complementary performance benefits, especially when enrollment speech is recorded with the in-ear microphone. We further demonstrate that PAS-SE personalized with noisy in-ear enrollments maintains performance benefits over the AS-SE system.
>
---
#### [replaced 012] GRAM: Spatial general-purpose audio representations for real-world environments
- **分类: cs.SD**

- **简介: 该论文提出GRAM模型，解决真实环境中音频表示学习问题，通过多通道自编码器提升空间音频建模能力。任务为构建鲁棒的实时音频基础模型。**

- **链接: [https://arxiv.org/pdf/2602.03307v2](https://arxiv.org/pdf/2602.03307v2)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** I have accidentally uploaded a revised version of my old paper. I meant to revise arXiv:2506.00934 rather than upload a new version
>
> **摘要:** Audio foundation models learn general-purpose audio representations that facilitate a wide range of downstream tasks. While the performance of these models has greatly increased for conventional single-channel, dry audio clips, their success in real-world acoustic environments with reverberation and noise is limited. Furthermore, most audio foundation models ignore the spatial dimension of real-world acoustic environments, ruling out tasks involving sound localization. To address these limitations, we propose GRAM: a general-purpose real-world audio model that employs a multi-channel masked autoencoder to efficiently learn spatial audio representations. We evaluated GRAM and other audio foundation models in a standardized manner on high-quality simulations of naturalistic, spatial acoustic environments as well as recordings of real-world environments and release these two complementary benchmark task suites: NatHEAR and RealSELD. Our results demonstrate that GRAM outperforms all state-of-the-art self-supervised audio foundation models on NatHEAR and the clean, single-channel version HEAR, while using only a fraction of the training data. GRAM also shows state-of-the-art localization performance in simulated environments and generalizes efficiently to real-world recordings in RealSELD. Taken together, GRAM presents a significant advance toward robust spatial audio foundation models for real-world environments.
>
---
#### [replaced 013] Content Anonymization for Privacy in Long-form Audio
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音隐私任务，解决长音频中说话人身份泄露问题。通过内容重写方法消除风格特征，同时保持语义，提升隐私安全性。**

- **链接: [https://arxiv.org/pdf/2510.12780v2](https://arxiv.org/pdf/2510.12780v2)**

> **作者:** Cristina Aggazzotti; Ashi Garg; Zexin Cai; Nicholas Andrews
>
> **备注:** Accepted to ICASSP 2026; v2: added more related work, used a more speech-adapted content-attack model, added a github link to code/prompts
>
> **摘要:** Voice anonymization techniques have been found to successfully obscure a speaker's acoustic identity in short, isolated utterances in benchmarks such as the VoicePrivacy Challenge. In practice, however, utterances seldom occur in isolation: long-form audio is commonplace in domains such as interviews, phone calls, and meetings. In these cases, many utterances from the same speaker are available, which pose a significantly greater privacy risk: given multiple utterances from the same speaker, an attacker could exploit an individual's vocabulary, syntax, and turns of phrase to re-identify them, even when their voice is completely disguised. To address this risk, we propose a new approach that performs a contextual rewriting of the transcripts in an ASR-TTS pipeline to eliminate speaker-specific style while preserving meaning. We present results in a long-form telephone conversation setting demonstrating the effectiveness of a content-based attack on voice-anonymized speech. Then we show how the proposed content-based anonymization methods can mitigate this risk while preserving speech utility. Overall, we find that paraphrasing is an effective defense against content-based attacks and recommend that stakeholders adopt this step to ensure anonymity in long-form audio.
>
---
#### [replaced 014] Scaling Spoken Language Models with Syllabic Speech Tokenization
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言建模任务，旨在解决长序列处理效率低的问题。通过引入音节级分词，提升模型效率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2509.26634v2](https://arxiv.org/pdf/2509.26634v2)**

> **作者:** Nicholas Lee; Cheol Jun Cho; Alan W Black; Gopala K. Anumanchipalli
>
> **备注:** ICASSP 2026
>
> **摘要:** Spoken language models (SLMs) typically discretize speech into high-frame-rate tokens extracted from SSL speech models. As the most successful LMs are based on the Transformer architecture, processing these long token streams with self-attention is expensive, as attention scales quadratically with sequence length. A recent SSL work introduces acoustic tokenization of speech at the syllable level, which is more interpretable and potentially more scalable with significant compression in token lengths (4-5 Hz). Yet, their value for spoken language modeling is not yet fully explored. We present the first systematic study of syllabic tokenization for spoken language modeling, evaluating models on a suite of SLU benchmarks while varying training data scale. Syllabic tokens can match or surpass the previous high-frame rate tokens while significantly cutting training and inference costs, achieving more than a 2x reduction in training time and a 5x reduction in FLOPs. Our findings highlight syllable-level language modeling as a promising path to efficient long-context spoken language models.
>
---
#### [replaced 015] Conditional Flow Matching for Visually-Guided Acoustic Highlighting
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于视觉引导的音频增强任务，解决音视频焦点不一致问题。通过生成模型和条件流匹配框架，实现更准确的音频重混。**

- **链接: [https://arxiv.org/pdf/2602.03762v2](https://arxiv.org/pdf/2602.03762v2)**

> **作者:** Hugo Malard; Gael Le Lan; Daniel Wong; David Lou Alon; Yi-Chiao Wu; Sanjeel Parekh
>
> **摘要:** Visually-guided acoustic highlighting seeks to rebalance audio in alignment with the accompanying video, creating a coherent audio-visual experience. While visual saliency and enhancement have been widely studied, acoustic highlighting remains underexplored, often leading to misalignment between visual and auditory focus. Existing approaches use discriminative models, which struggle with the inherent ambiguity in audio remixing, where no natural one-to-one mapping exists between poorly-balanced and well-balanced audio mixes. To address this limitation, we reframe this task as a generative problem and introduce a Conditional Flow Matching (CFM) framework. A key challenge in iterative flow-based generation is that early prediction errors -- in selecting the correct source to enhance -- compound over steps and push trajectories off-manifold. To address this, we introduce a rollout loss that penalizes drift at the final step, encouraging self-correcting trajectories and stabilizing long-range flow integration. We further propose a conditioning module that fuses audio and visual cues before vector field regression, enabling explicit cross-modal source selection. Extensive quantitative and qualitative evaluations show that our method consistently surpasses the previous state-of-the-art discriminative approach, establishing that visually-guided audio remixing is best addressed through generative modeling.
>
---
#### [replaced 016] The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era
- **分类: cs.SD; cs.CL; cs.HC; eess.AS**

- **简介: 该论文属于对话系统任务，旨在提升语音对话系统的拟人化能力。解决情感理解和实时交互问题，通过构建数据集和两个评估赛道进行基准测试。**

- **链接: [https://arxiv.org/pdf/2601.05564v2](https://arxiv.org/pdf/2601.05564v2)**

> **作者:** Zhixian Zhao; Shuiyuan Wang; Guojian Li; Hongfei Xue; Chengyou Wang; Shuai Wang; Longshuai Xiao; Zihan Zhang; Hui Bu; Xin Xu; Xinsheng Wang; Hexin Liu; Eng Siong Chng; Hung-yi Lee; Lei Xie
>
> **备注:** Official summary paper for the ICASSP 2026 HumDial Challenge
>
> **摘要:** Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.
>
---
#### [replaced 017] The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频场景分析任务，解决现有模型忽略空间信息的问题。通过引入空间理解框架和四项创新，提升模型的立体音频感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.02954v2](https://arxiv.org/pdf/2601.02954v2)**

> **作者:** Yuhuan You; Lai Wei; Xihong Wu; Tianshu Qu
>
> **摘要:** Existing large audio-language models perceive the world as "mono"-a single stream of audio that ignores the critical spatial dimension ("where") required for universal audio scene analysis (ASA). To bridge this gap, we first introduce a hierarchical framework for audio scene analysis. Guided by this framework, we introduce a system that enables large audio-language models (LALMs) to understand and reason about the complex acoustic world. Our system endows LALMs with universal spatial understanding through four key innovations: (1) A scalable simulation pipeline that synthesizes high-quality First-Order-Ambisonics(FOA) data; (2) A unified model framework that integrates universal spatial encoding with a dense hybrid projection mechanism to bridge the modality gap; (3) A progressive training curriculum that evolves from representation alignment to reinforcement learning-based reasoning; and (4) A comprehensive benchmark for audio scene analysis (ASA) designed to rigorously evaluate atomic perception, relational integration, and cognitive reasoning capabilities, on which our model demonstrates comparatively strong capability for spatial understanding. Our work provides a clear pathway for leveraging the powerful reasoning abilities of LALMs towards holistic ASA, advancing from "mono" semantic recognition to spatial intelligence.
>
---
#### [replaced 018] GRAM: Spatial general-purpose audio representation models for real-world applications
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出GRAM模型，解决真实环境中音频表示与声源定位问题。通过多通道自编码器提升空间音频建模，实验表明其性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2506.00934v5](https://arxiv.org/pdf/2506.00934v5)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Revise with RealSELD
>
> **摘要:** Audio foundation models learn general-purpose audio representations that facilitate a wide range of downstream tasks. While the performance of these models has greatly increased for conventional single-channel, dry audio clips, their success in real-world acoustic environments with reverberation and noise is limited. Furthermore, most audio foundation models ignore the spatial dimension of real-world acoustic environments, ruling out tasks involving sound localization. To address these limitations, we propose GRAM: a general-purpose real-world audio model that employs a multi-channel masked autoencoder to efficiently learn spatial audio representations. We evaluated GRAM and other audio foundation models in a standardized manner on high-quality simulations of naturalistic, spatial acoustic environments as well as recordings of real-world environments and release these two complementary benchmark task suites: NatHEAR and RealSELD. Our results demonstrate that GRAM outperforms all state-of-the-art self-supervised audio foundation models on NatHEAR and the clean, single-channel version HEAR, while using only a fraction of the training data. GRAM also shows state-of-the-art localization performance in simulated environments and generalizes efficiently to real-world recordings in RealSELD. Taken together, GRAM presents a significant advance toward robust spatial audio foundation models for real-world environments.
>
---
#### [replaced 019] When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **简介: 该论文属于音频-语言模型安全任务，旨在解决音频输入被用于攻击AI系统的问题。提出WhisperInject框架，通过良性音频注入有害内容，实现对多模态模型的操控。**

- **链接: [https://arxiv.org/pdf/2508.03365v3](https://arxiv.org/pdf/2508.03365v3)**

> **作者:** Hiskias Dingeto; Taeyoun Kwon; Dasol Choi; Bodam Kim; DongGeon Lee; Haon Park; JaeHoon Lee; Jongho Shin
>
> **摘要:** As large language models (LLMs) become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that manipulates state-of-the-art audio language models to generate harmful content. Our method embeds harmful payloads as subtle perturbations into audio inputs that remain intelligible to human listeners. The first stage uses a novel reward-based white-box optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to jailbreak the target model and elicit harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use gradient-based optimization to embed subtle perturbations into benign audio carriers, such as weather queries or greeting messages. Our method achieves average attack success rates of 60-78% across two benchmarks and five multimodal LLMs, validated by multiple evaluation frameworks. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating multimodal AI systems.
>
---
#### [replaced 020] ConceptCaps: a Distilled Concept Dataset for Interpretability in Music Models
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 论文提出ConceptCaps数据集，解决音乐模型可解释性问题。该数据集包含21k音乐-文字-标签三元组，用于训练和评估概念可解释方法。**

- **链接: [https://arxiv.org/pdf/2601.14157v3](https://arxiv.org/pdf/2601.14157v3)**

> **作者:** Bruno Sienkiewicz; Łukasz Neumann; Mateusz Modrzejewski
>
> **摘要:** Concept-based interpretability methods like TCAV require clean, well-separated positive and negative examples for each concept. Existing music datasets lack this structure: tags are sparse, noisy, or ill-defined. We introduce ConceptCaps, a dataset of 21k music-caption-tags triplets with explicit labels from a 200-attribute taxonomy. Our pipeline separates semantic modeling from text generation: a VAE learns plausible attribute co-occurrence patterns, a fine-tuned LLM converts attribute lists into professional descriptions, and MusicGen synthesizes corresponding audio. This separation improves coherence and controllability over end-to-end approaches. We validate the dataset through audio-text alignment (CLAP), linguistic quality metrics (BERTScore, MAUVE), and TCAV analysis confirming that concept probes recover musically meaningful patterns. Dataset and code are available online.
>
---
#### [replaced 021] SAVGBench: Benchmarking Spatially Aligned Audio-Video Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于空间对齐音视频生成任务，旨在解决音视频空间不对齐问题。构建了相关数据集和评估指标，对比了两种基线方法。**

- **链接: [https://arxiv.org/pdf/2412.13462v2](https://arxiv.org/pdf/2412.13462v2)**

> **作者:** Kazuki Shimada; Christian Simon; Takashi Shibuya; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** 5 pages, 2 figures, accepted for publication in IEEE ICASSP 2026
>
> **摘要:** This work addresses the lack of multimodal generative models capable of producing high-quality videos with spatially aligned audio. While recent advancements in generative models have been successful in video generation, they often overlook the spatial alignment between audio and visuals, which is essential for immersive experiences. To tackle this problem, we establish a new research direction in benchmarking the Spatially Aligned Audio-Video Generation (SAVG) task. We introduce a spatially aligned audio-visual dataset, whose audio and video data are curated based on whether sound events are onscreen or not. We also propose a new alignment metric that aims to evaluate the spatial alignment between audio and video. Then, using the dataset and metric, we benchmark two types of baseline methods: one is based on a joint audio-video generation model, and the other is a two-stage method that combines a video generation model and a video-to-audio generation model. Our experimental results demonstrate that gaps exist between the baseline methods and the ground truth in terms of video and audio quality, as well as spatial alignment between the two modalities.
>
---
