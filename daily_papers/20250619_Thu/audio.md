# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Probabilistic Trajectory GOSPA: A Metric for Uncertainty-Aware Multi-Object Tracking Performance Evaluation
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于多目标跟踪任务，旨在解决轨迹估计中不确定性评估的问题。提出了一种新的概率轨迹GOSPA度量，用于更准确地评价跟踪算法性能。**

- **链接: [http://arxiv.org/pdf/2506.15148v1](http://arxiv.org/pdf/2506.15148v1)**

> **作者:** Yuxuan Xia; Ángel F. García-Fernández; Johan Karlsson; Yu Ge; Lennart Svensson; Ting Yuan
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** This paper presents a generalization of the trajectory general optimal sub-pattern assignment (GOSPA) metric for evaluating multi-object tracking algorithms that provide trajectory estimates with track-level uncertainties. This metric builds on the recently introduced probabilistic GOSPA metric to account for both the existence and state estimation uncertainties of individual object states. Similar to trajectory GOSPA (TGOSPA), it can be formulated as a multidimensional assignment problem, and its linear programming relaxation--also a valid metric--is computable in polynomial time. Additionally, this metric retains the interpretability of TGOSPA, and we show that its decomposition yields intuitive costs terms associated to expected localization error and existence probability mismatch error for properly detected objects, expected missed and false detection error, and track switch error. The effectiveness of the proposed metric is demonstrated through a simulation study.
>
---
#### [new 002] A Comparative Evaluation of Deep Learning Models for Speech Enhancement in Real-World Noisy Environments
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决噪声环境下语音质量与可懂度问题。通过对比三种深度学习模型，评估其在噪声抑制、语音质量和说话人特征保留方面的性能。**

- **链接: [http://arxiv.org/pdf/2506.15000v1](http://arxiv.org/pdf/2506.15000v1)**

> **作者:** Md Jahangir Alam Khondkar; Ajan Ahmed; Masudul Haider Imtiaz; Stephanie Schuckers
>
> **摘要:** Speech enhancement, particularly denoising, is vital in improving the intelligibility and quality of speech signals for real-world applications, especially in noisy environments. While prior research has introduced various deep learning models for this purpose, many struggle to balance noise suppression, perceptual quality, and speaker-specific feature preservation, leaving a critical research gap in their comparative performance evaluation. This study benchmarks three state-of-the-art models Wave-U-Net, CMGAN, and U-Net, on diverse datasets such as SpEAR, VPQAD, and Clarkson datasets. These models were chosen due to their relevance in the literature and code accessibility. The evaluation reveals that U-Net achieves high noise suppression with SNR improvements of +71.96% on SpEAR, +64.83% on VPQAD, and +364.2% on the Clarkson dataset. CMGAN outperforms in perceptual quality, attaining the highest PESQ scores of 4.04 on SpEAR and 1.46 on VPQAD, making it well-suited for applications prioritizing natural and intelligible speech. Wave-U-Net balances these attributes with improvements in speaker-specific feature retention, evidenced by VeriSpeak score gains of +10.84% on SpEAR and +27.38% on VPQAD. This research indicates how advanced methods can optimize trade-offs between noise suppression, perceptual quality, and speaker recognition. The findings may contribute to advancing voice biometrics, forensic audio analysis, telecommunication, and speaker verification in challenging acoustic conditions.
>
---
#### [new 003] An accurate and revised version of optical character recognition-based speech synthesis using LabVIEW
- **分类: cs.SD; cs.CL; cs.CV; eess.AS; 14J60; I.2.7; I.4; I.5; I.7.5**

- **简介: 该论文属于语音合成任务，旨在解决视障人士获取书籍困难的问题。通过OCR技术与LabVIEW实现准确的语音转换系统。**

- **链接: [http://arxiv.org/pdf/2506.15029v1](http://arxiv.org/pdf/2506.15029v1)**

> **作者:** Prateek Mehta; Anasuya Patil
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Knowledge extraction through sound is a distinctive property. Visually impaired individuals often rely solely on Braille books and audio recordings provided by NGOs. Due to limitations in these approaches, blind individuals often cannot access books of their choice. Speech is a more effective mode of communication than text for blind and visually impaired persons, as they can easily respond to sounds. This paper presents the development of an accurate, reliable, cost-effective, and user-friendly optical character recognition (OCR)-based speech synthesis system. The OCR-based system has been implemented using Laboratory Virtual Instrument Engineering Workbench (LabVIEW).
>
---
#### [new 004] Diff-TONE: Timestep Optimization for iNstrument Editing in Text-to-Music Diffusion Models
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于文本到音乐生成任务，旨在解决乐器编辑中保持内容不变的同时更换音色的问题。通过选择合适的时间步进行优化实现目标。**

- **链接: [http://arxiv.org/pdf/2506.15530v1](http://arxiv.org/pdf/2506.15530v1)**

> **作者:** Teysir Baoueb; Xiaoyu Bie; Xi Wang; Gaël Richard
>
> **摘要:** Breakthroughs in text-to-music generation models are transforming the creative landscape, equipping musicians with innovative tools for composition and experimentation like never before. However, controlling the generation process to achieve a specific desired outcome remains a significant challenge. Even a minor change in the text prompt, combined with the same random seed, can drastically alter the generated piece. In this paper, we explore the application of existing text-to-music diffusion models for instrument editing. Specifically, for an existing audio track, we aim to leverage a pretrained text-to-music diffusion model to edit the instrument while preserving the underlying content. Based on the insight that the model first focuses on the overall structure or content of the audio, then adds instrument information, and finally refines the quality, we show that selecting a well-chosen intermediate timestep, identified through an instrument classifier, yields a balance between preserving the original piece's content and achieving the desired timbre. Our method does not require additional training of the text-to-music diffusion model, nor does it compromise the generation process's speed.
>
---
#### [new 005] SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS; 68T10 (Primary), 68T50 (Secondary); H.5.5; H.5.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.15154v1](http://arxiv.org/pdf/2506.15154v1)**

> **作者:** Anuradha Chopra; Abhinaba Roy; Dorien Herremans
>
> **备注:** 14 pages, 2 figures, Accepted to AIMC 2025
>
> **摘要:** Detailed captions that accurately reflect the characteristics of a music piece can enrich music databases and drive forward research in music AI. This paper introduces a multi-task music captioning model, SonicVerse, that integrates caption generation with auxiliary music feature detection tasks such as key detection, vocals detection, and more, so as to directly capture both low-level acoustic details as well as high-level musical attributes. The key contribution is a projection-based architecture that transforms audio input into language tokens, while simultaneously detecting music features through dedicated auxiliary heads. The outputs of these heads are also projected into language tokens, to enhance the captioning input. This framework not only produces rich, descriptive captions for short music fragments but also directly enables the generation of detailed time-informed descriptions for longer music pieces, by chaining the outputs using a large-language model. To train the model, we extended the MusicBench dataset by annotating it with music features using MIRFLEX, a modular music feature extractor, resulting in paired audio, captions and music feature data. Experimental results show that incorporating features in this way improves the quality and detail of the generated captions.
>
---
#### [new 006] pycnet-audio: A Python package to support bioacoustics data processing
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文介绍了一个用于生物声学数据处理的Python包pycnet-audio，旨在解决大规模音频数据自动检测与分析的问题。**

- **链接: [http://arxiv.org/pdf/2506.14864v1](http://arxiv.org/pdf/2506.14864v1)**

> **作者:** Zachary J. Ruff; Damon B. Lesmeister
>
> **摘要:** Passive acoustic monitoring is an emerging approach in wildlife research that leverages recent improvements in purpose-made automated recording units (ARUs). The general approach is to deploy ARUs in the field to record on a programmed schedule for extended periods (weeks or months), after which the audio data are retrieved. These data must then be processed, typically either by measuring or analyzing characteristics of the audio itself (e.g. calculating acoustic indices), or by searching for some signal of interest within the recordings, e.g. vocalizations or other sounds produced by some target species, anthropogenic or environmental noise, etc. In the latter case, some method is required to locate the signal(s) of interest within the audio. While very small datasets can simply be searched manually, even modest projects can produce audio datasets on the order of 105 hours of recordings, making manual review impractical and necessitating some form of automated detection. pycnet-audio (Ruff 2024) is intended to provide a practical processing workflow for acoustic data, built around the PNW-Cnet model, which was initially developed by the U.S. Forest Service to support population monitoring of northern spotted owls (Strix occidentalis caurina) and other forest owls (Lesmeister and Jenkins 2022; Ruff et al. 2020). PNW-Cnet has been expanded to detect vocalizations of ca. 80 forest wildlife species and numerous forms of anthropogenic and environmental noise (Ruff et al. 2021, 2023).
>
---
#### [new 007] Versatile Symbolic Music-for-Music Modeling via Function Alignment
- **分类: cs.SD**

- **简介: 该论文属于音乐AI领域，解决音乐内容与标签映射问题。通过预训练语言模型和轻量适配器，实现多种符号音乐任务的高效建模与生成。**

- **链接: [http://arxiv.org/pdf/2506.15548v1](http://arxiv.org/pdf/2506.15548v1)**

> **作者:** Junyan Jiang; Daniel Chin; Liwei Lin; Xuanjie Liu; Gus Xia
>
> **摘要:** Many music AI models learn a map between music content and human-defined labels. However, many annotations, such as chords, can be naturally expressed within the music modality itself, e.g., as sequences of symbolic notes. This observation enables both understanding tasks (e.g., chord recognition) and conditional generation tasks (e.g., chord-conditioned melody generation) to be unified under a music-for-music sequence modeling paradigm. In this work, we propose parameter-efficient solutions for a variety of symbolic music-for-music tasks. The high-level idea is that (1) we utilize a pretrained Language Model (LM) for both the reference and the target sequence and (2) we link these two LMs via a lightweight adapter. Experiments show that our method achieves superior performance among different tasks such as chord recognition, melody generation, and drum track generation. All demos, code and model weights are publicly available.
>
---
#### [new 008] TTSOps: A Closed-Loop Corpus Optimization Framework for Training Multi-Speaker TTS Models from Dark Data
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，解决从噪声数据中训练多说话人TTS模型的问题。提出TTSOps框架，实现数据自动收集、动态清洗和闭环优化。**

- **链接: [http://arxiv.org/pdf/2506.15614v1](http://arxiv.org/pdf/2506.15614v1)**

> **作者:** Kentaro Seki; Shinnosuke Takamichi; Takaaki Saeki; Hiroshi Saruwatari
>
> **摘要:** This paper presents TTSOps, a fully automated closed-loop framework for constructing multi-speaker text-to-speech (TTS) systems from noisy, uncurated web-scale speech data, often referred to as ``dark data,'' such as online videos. Conventional TTS training pipelines require well-curated corpora with high acoustic quality and accurate text-speech alignment, which severely limits scalability, speaker diversity, and real-world applicability. While recent studies have proposed acoustic-quality-based data selection techniques, they often overlook two critical aspects: (1) the inherent robustness of modern TTS models to noise, and (2) the potential contribution of perceptually low-quality yet informative samples. To address these issues, TTSOps introduces a data-centric training pipeline that integrates three core components: (1) automated data collection from dark data sources, (2) utterance-level dynamic selection of data cleansing methods based on training data quality, and (3) evaluation-in-the-loop data selection using automatically predicted mean opinion scores (MOS) to estimate each utterance's impact on model performance. Furthermore, TTSOps jointly optimizes the corpus and the TTS model in a closed-loop framework by dynamically adapting both data selection and data cleansing processes to the characteristics of the target TTS model. Extensive experiments on Japanese YouTube data demonstrate that TTSOps outperforms conventional acoustic-quality-based baselines in both the naturalness and speaker diversity of synthesized speech.
>
---
#### [new 009] Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于自动歌词转录任务，旨在解决音乐中人声与伴奏干扰导致的识别难题。通过音乐源分离技术提升Whisper模型的转录效果，提出短长文本处理方法，取得最佳开源结果。**

- **链接: [http://arxiv.org/pdf/2506.15514v1](http://arxiv.org/pdf/2506.15514v1)**

> **作者:** Jaza Syed; Ivan Meresman Higgs; Ondřej Cífka; Mark Sandler
>
> **备注:** Accepted at 2025 ICME Workshop AI for Music
>
> **摘要:** Automatic lyrics transcription (ALT) remains a challenging task in the field of music information retrieval, despite great advances in automatic speech recognition (ASR) brought about by transformer-based architectures in recent years. One of the major challenges in ALT is the high amplitude of interfering audio signals relative to conventional ASR due to musical accompaniment. Recent advances in music source separation have enabled automatic extraction of high-quality separated vocals, which could potentially improve ALT performance. However, the effect of source separation has not been systematically investigated in order to establish best practices for its use. This work examines the impact of source separation on ALT using Whisper, a state-of-the-art open source ASR model. We evaluate Whisper's performance on original audio, separated vocals, and vocal stems across short-form and long-form transcription tasks. For short-form, we suggest a concatenation method that results in a consistent reduction in Word Error Rate (WER). For long-form, we propose an algorithm using source separation as a vocal activity detector to derive segment boundaries, which results in a consistent reduction in WER relative to Whisper's native long-form algorithm. Our approach achieves state-of-the-art results for an open source system on the Jam-ALT long-form ALT benchmark, without any training or fine-tuning. We also publish MUSDB-ALT, the first dataset of long-form lyric transcripts following the Jam-ALT guidelines for which vocal stems are publicly available.
>
---
#### [new 010] PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决实时语音交互中大语言模型的延迟问题。通过输入时间预测生成，减少音频输出延迟。**

- **链接: [http://arxiv.org/pdf/2506.15556v1](http://arxiv.org/pdf/2506.15556v1)**

> **作者:** Shufan Li; Aditya Grover
>
> **备注:** 16 pages,4 figures
>
> **摘要:** Large Language Models (LLMs) are widely used in real-time voice chat applications, typically in combination with text-to-speech (TTS) systems to generate audio responses. However, their large size often leads to noticeable latency between the end of user input and the start of audio output, resulting in suboptimal user experiences. This latency is particularly evident when LLMs are deployed as single-user voice assistants on consumer-grade hardware with limited computing capacity. We discovered that this latency is primarily dominated by the time it takes for the LLMs to generate the first sentence, which is required as input by the TTS systems that synthesize audio responses on a sentence-by-sentence basis. To address this bottleneck, we propose Predictive Generation (PredGen), a novel framework that mitigates-or even eliminates-this delay through speculative decoding at input time. PredGen generates candidate responses while the user is still speaking, enabling the system to begin TTS processing with minimal delay. Simulated experiments on the Lmsys and MT-Bench datasets show that the proposed method can effectively reduce the latency by around 2x across a wide range of use cases, while incurring only minimal additional computation cost at input time-computation that would otherwise go unused.
>
---
#### [new 011] video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **简介: 该论文属于视频描述生成任务，旨在提升视频字幕的准确性和完整性。通过改进的DPO方法和LoRA技术，优化模型性能，实现更高质量的视频描述生成。**

- **链接: [http://arxiv.org/pdf/2506.15220v1](http://arxiv.org/pdf/2506.15220v1)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [new 012] I Know You're Listening: Adaptive Voice for HRI
- **分类: cs.RO; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于人机交互任务，旨在提升语言教学机器人的语音效果。针对语音表达不足和L2学习者理解困难的问题，提出轻量表达语音、环境自适应策略及增强清晰度的TTS系统。**

- **链接: [http://arxiv.org/pdf/2506.15107v1](http://arxiv.org/pdf/2506.15107v1)**

> **作者:** Paige Tuttösí
>
> **备注:** PhD Thesis Simon Fraser University https://summit.sfu.ca/item/39353 Read the Room: Adapting a Robot's Voice to Ambient and Social Contexts IROS 23 Mmm whatcha say? Uncovering distal and proximal context effects in first and second-language word perception using psychophysical reverse correlation INTERSPEECH 24 Emojivoice: Towards long-term controllable expressivity in robot speech RO-MAN 25
>
> **摘要:** While the use of social robots for language teaching has been explored, there remains limited work on a task-specific synthesized voices for language teaching robots. Given that language is a verbal task, this gap may have severe consequences for the effectiveness of robots for language teaching tasks. We address this lack of L2 teaching robot voices through three contributions: 1. We address the need for a lightweight and expressive robot voice. Using a fine-tuned version of Matcha-TTS, we use emoji prompting to create an expressive voice that shows a range of expressivity over time. The voice can run in real time with limited compute resources. Through case studies, we found this voice more expressive, socially appropriate, and suitable for long periods of expressive speech, such as storytelling. 2. We explore how to adapt a robot's voice to physical and social ambient environments to deploy our voices in various locations. We found that increasing pitch and pitch rate in noisy and high-energy environments makes the robot's voice appear more appropriate and makes it seem more aware of its current environment. 3. We create an English TTS system with improved clarity for L2 listeners using known linguistic properties of vowels that are difficult for these listeners. We used a data-driven, perception-based approach to understand how L2 speakers use duration cues to interpret challenging words with minimal tense (long) and lax (short) vowels in English. We found that the duration of vowels strongly influences the perception for L2 listeners and created an "L2 clarity mode" for Matcha-TTS that applies a lengthening to tense vowels while leaving lax vowels unchanged. Our clarity mode was found to be more respectful, intelligible, and encouraging than base Matcha-TTS while reducing transcription errors in these challenging tense/lax minimal pairs.
>
---
#### [new 013] Factorized RVQ-GAN For Disentangled Speech Tokenization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出HAC，一种统一的语音编解码器，解决语音表示学习问题。通过分层结构分离语音的音素和词义信息，提升语音生成与理解效果。**

- **链接: [http://arxiv.org/pdf/2506.15456v1](http://arxiv.org/pdf/2506.15456v1)**

> **作者:** Sameer Khurana; Dominik Klement; Antoine Laurent; Dominik Bobos; Juraj Novosad; Peter Gazdik; Ellen Zhang; Zili Huang; Amir Hussein; Ricard Marxer; Yoshiki Masuyama; Ryo Aihara; Chiori Hori; Francois G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We propose Hierarchical Audio Codec (HAC), a unified neural speech codec that factorizes its bottleneck into three linguistic levels-acoustic, phonetic, and lexical-within a single model. HAC leverages two knowledge distillation objectives: one from a pre-trained speech encoder (HuBERT) for phoneme-level structure, and another from a text-based encoder (LaBSE) for lexical cues. Experiments on English and multilingual data show that HAC's factorized bottleneck yields disentangled token sets: one aligns with phonemes, while another captures word-level semantics. Quantitative evaluations confirm that HAC tokens preserve naturalness and provide interpretable linguistic information, outperforming single-level baselines in both disentanglement and reconstruction quality. These findings underscore HAC's potential as a unified discrete speech representation, bridging acoustic detail and lexical meaning for downstream speech generation and understanding tasks.
>
---
#### [new 014] Beyond Universality: Cultural Diversity in Music and Its Implications for Sound Design and Sonification
- **分类: physics.soc-ph; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.14877v1](http://arxiv.org/pdf/2506.14877v1)**

> **作者:** Rubén García-Benito
>
> **备注:** 12 pages, 1 figure. Long paper accepted for publication at the Audio Mostly & ICAD Joint Conference (AM.ICAD 2025). To appear in the ACM International Conference Proceedings Series (ICPS)
>
> **摘要:** The Audio Mostly (AM) conference has long been a platform for exploring the intersection of sound, technology, and culture. Despite growing interest in sonic cultures, discussions on the role of cultural diversity in sound design and sonification remain limited. This paper investigates the implicit biases and gaps within the discourse on music and sound aesthetics, challenging the notion of music as a 'universal language'. Through a historical and cross-cultural analysis of musicology and ethnomusicology, the profound influence of cultural context on auditory perception and aesthetic appraisal is highlighted. By drawing parallels between historical music practices and contemporary sound design, the paper advocates for a more inclusive approach that recognizes the diversity of sonic traditions. Using music as a case study, we underscore broader implications for sound design and sonification, emphasizing the need to integrate cultural perspectives into auditory design practices. A reevaluation of existing frameworks in sound design and sonification is proposed, emphasizing the necessity of culturally informed practices that resonate with global audiences. Ultimately, embracing cultural diversity in sound design is suggested to lead to richer, more meaningful auditory experiences and to foster greater inclusivity within the field.
>
---
## 更新

#### [replaced 001] Video-Guided Text-to-Music Generation Using Public Domain Movie Collections
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12573v2](http://arxiv.org/pdf/2506.12573v2)**

> **作者:** Haven Kim; Zachary Novack; Weihan Xu; Julian McAuley; Hao-Wen Dong
>
> **备注:** ISMIR 2025 regular paper. Dataset, code, and demo available at https://havenpersona.github.io/ossl-v1
>
> **摘要:** Despite recent advancements in music generation systems, their application in film production remains limited, as they struggle to capture the nuances of real-world filmmaking, where filmmakers consider multiple factors-such as visual content, dialogue, and emotional tone-when selecting or composing music for a scene. This limitation primarily stems from the absence of comprehensive datasets that integrate these elements. To address this gap, we introduce Open Screen Sound Library (OSSL), a dataset consisting of movie clips from public domain films, totaling approximately 36.5 hours, paired with high-quality soundtracks and human-annotated mood information. To demonstrate the effectiveness of our dataset in improving the performance of pre-trained models on film music generation tasks, we introduce a new video adapter that enhances an autoregressive transformer-based text-to-music model by adding video-based conditioning. Our experimental results demonstrate that our proposed approach effectively enhances MusicGen-Medium in terms of both objective measures of distributional and paired fidelity, and subjective compatibility in mood and genre. The dataset and code are available at https://havenpersona.github.io/ossl-v1.
>
---
#### [replaced 002] MERGE -- A Bimodal Audio-Lyrics Dataset for Static Music Emotion Recognition
- **分类: cs.SD; cs.IR; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.06060v3](http://arxiv.org/pdf/2407.06060v3)**

> **作者:** Pedro Lima Louro; Hugo Redinho; Ricardo Santos; Ricardo Malheiro; Renato Panda; Rui Pedro Paiva
>
> **备注:** 18 pages, 2 figures, 8 tables, submitted to IEEE Transactions on Affective Computing
>
> **摘要:** The Music Emotion Recognition (MER) field has seen steady developments in recent years, with contributions from feature engineering, machine learning, and deep learning. The landscape has also shifted from audio-centric systems to bimodal ensembles that combine audio and lyrics. However, a lack of public, sizable and quality-controlled bimodal databases has hampered the development and improvement of bimodal audio-lyrics systems. This article proposes three new audio, lyrics, and bimodal MER research datasets, collectively referred to as MERGE, which were created using a semi-automatic approach. To comprehensively assess the proposed datasets and establish a baseline for benchmarking, we conducted several experiments for each modality, using feature engineering, machine learning, and deep learning methodologies. Additionally, we propose and validate fixed train-validation-test splits. The obtained results confirm the viability of the proposed datasets, achieving the best overall result of 81.74\% F1-score for bimodal classification.
>
---
#### [replaced 003] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13300v3](http://arxiv.org/pdf/2506.13300v3)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [replaced 004] A Bird Song Detector for improving bird identification through Deep Learning: a case study from Doñana
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.NE; I.5.4; I.2.6; I.4.8**

- **链接: [http://arxiv.org/pdf/2503.15576v2](http://arxiv.org/pdf/2503.15576v2)**

> **作者:** Alba Márquez-Rodríguez; Miguel Ángel Mohedano-Munoz; Manuel J. Marín-Jiménez; Eduardo Santamaría-García; Giulia Bastianelli; Pedro Jordano; Irene Mendoza
>
> **备注:** 23 pages, 14 images, for associated dataset see https://huggingface.co/datasets/GrunCrow/BIRDeep_AudioAnnotations , for associated code see https://github.com/GrunCrow/BIRDeep_BirdSongDetector_NeuralNetworks and https://github.com/GrunCrow/Bird-Song-Detector
>
> **摘要:** Passive Acoustic Monitoring is a key tool for biodiversity conservation, but the large volumes of unsupervised audio it generates present major challenges for extracting meaningful information. Deep Learning offers promising solutions. BirdNET, a widely used bird identification model, has shown success in many study systems but is limited at local scale due to biases in its training data, which focus on specific locations and target sounds rather than entire soundscapes. A key challenge in bird species identification is that many recordings either lack target species or contain overlapping vocalizations, complicating automatic identification. To address these problems, we developed a multi-stage pipeline for automatic bird vocalization identification in Do\~nana National Park (SW Spain), a wetland of high conservation concern. We deployed AudioMoth recorders in three main habitats across nine locations and manually annotated 461 minutes of audio, resulting in 3749 labeled segments spanning 34 classes. We first applied a Bird Song Detector to isolate bird vocalizations using spectrogram-based image processing. Then, species were classified using custom models trained at the local scale. Applying the Bird Song Detector before classification improved species identification, as all models performed better when analyzing only the segments where birds were detected. Specifically, the combination of detector and fine-tuned BirdNET outperformed the baseline without detection. This approach demonstrates the effectiveness of integrating a Bird Song Detector with local classification models. These findings highlight the need to adapt general-purpose tools to specific ecological challenges. Automatically detecting bird species helps track the health of this threatened ecosystem, given birds sensitivity to environmental change, and supports conservation planning to reduce biodiversity loss.
>
---
#### [replaced 005] Synthesizing Composite Hierarchical Structure from Symbolic Music Corpora
- **分类: cs.AI; cs.LO; cs.SD; G.1.6; I.2.4; J.5; G.2.2**

- **链接: [http://arxiv.org/pdf/2502.15849v3](http://arxiv.org/pdf/2502.15849v3)**

> **作者:** Ilana Shapiro; Ruanqianqian Huang; Zachary Novack; Cheng-i Wang; Hao-Wen Dong; Taylor Berg-Kirkpatrick; Shlomo Dubnov; Sorin Lerner
>
> **备注:** In Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI '25), Montreal, Canada, August 2025
>
> **摘要:** Western music is an innately hierarchical system of interacting levels of structure, from fine-grained melody to high-level form. In order to analyze music compositions holistically and at multiple granularities, we propose a unified, hierarchical meta-representation of musical structure called the structural temporal graph (STG). For a single piece, the STG is a data structure that defines a hierarchy of progressively finer structural musical features and the temporal relationships between them. We use the STG to enable a novel approach for deriving a representative structural summary of a music corpus, which we formalize as a dually NP-hard combinatorial optimization problem extending the Generalized Median Graph problem. Our approach first applies simulated annealing to develop a measure of structural distance between two music pieces rooted in graph isomorphism. Our approach then combines the formal guarantees of SMT solvers with nested simulated annealing over structural distances to produce a structurally sound, representative centroid STG for an entire corpus of STGs from individual pieces. To evaluate our approach, we conduct experiments verifying that structural distance accurately differentiates between music pieces, and that derived centroids accurately structurally characterize their corpora.
>
---
