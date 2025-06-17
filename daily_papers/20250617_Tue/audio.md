# 音频 cs.SD;  eess.SP

- **最新发布 40 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于实时音频处理任务，旨在解决神经网络推理在实时音频应用中的延迟问题。提出ANIRA库，支持多种框架，并通过线程池优化性能。**

- **链接: [http://arxiv.org/pdf/2506.12665v1](http://arxiv.org/pdf/2506.12665v1)**

> **作者:** Valentin Ackva; Fares Schulz
>
> **备注:** 8 pages, accepted to the Proceedings of the 5th IEEE International Symposium on the Internet of Sounds (2024) - repository: github.com/anira-project/anira
>
> **摘要:** Numerous tools for neural network inference are currently available, yet many do not meet the requirements of real-time audio applications. In response, we introduce anira, an efficient cross-platform library. To ensure compatibility with a broad range of neural network architectures and frameworks, anira supports ONNX Runtime, LibTorch, and TensorFlow Lite as backends. Each inference engine exhibits real-time violations, which anira mitigates by decoupling the inference from the audio callback to a static thread pool. The library incorporates built-in latency management and extensive benchmarking capabilities, both crucial to ensure a continuous signal flow. Three different neural network architectures for audio effect emulation are then subjected to benchmarking across various configurations. Statistical modeling is employed to identify the influence of various factors on performance. The findings indicate that for stateless models, ONNX Runtime exhibits the lowest runtimes. For stateful models, LibTorch demonstrates the fastest performance. Our results also indicate that for certain model-engine combinations, the initial inferences take longer, particularly when these inferences exhibit a higher incidence of real-time violations.
>
---
#### [new 002] Style-based Composer Identification and Attribution of Symbolic Music Scores: a Systematic Survey
- **分类: cs.SD; cs.AI; cs.CV; cs.DL; eess.AS**

- **简介: 该论文属于音乐风格识别与作者归属任务，旨在解决音乐作品作者鉴定的可靠性问题。通过系统综述58篇文献，分析方法与挑战，提出提升研究质量的建议。**

- **链接: [http://arxiv.org/pdf/2506.12440v1](http://arxiv.org/pdf/2506.12440v1)**

> **作者:** Federico Simonetta
>
> **备注:** Accepted at the TISMIR
>
> **摘要:** This paper presents the first comprehensive systematic review of literature on style-based composer identification and authorship attribution in symbolic music scores. Addressing the critical need for improved reliability and reproducibility in this field, the review rigorously analyzes 58 peer-reviewed papers published across various historical periods, with the search adapted to evolving terminology. The analysis critically assesses prevailing repertoires, computational approaches, and evaluation methodologies, highlighting significant challenges. It reveals that a substantial portion of existing research suffers from inadequate validation protocols and an over-reliance on simple accuracy metrics for often imbalanced datasets, which can undermine the credibility of attribution claims. The crucial role of robust metrics like Balanced Accuracy and rigorous cross-validation in ensuring trustworthy results is emphasized. The survey also details diverse feature representations and the evolution of machine learning models employed. Notable real-world authorship attribution cases, such as those involving works attributed to Bach, Josquin Desprez, and Lennon-McCartney, are specifically discussed, illustrating the opportunities and pitfalls of applying computational techniques to resolve disputed musical provenance. Based on these insights, a set of actionable guidelines for future research are proposed. These recommendations are designed to significantly enhance the reliability, reproducibility, and musicological validity of composer identification and authorship attribution studies, fostering more robust and interpretable computational stylistic analysis.
>
---
#### [new 003] GSDNet: Revisiting Incomplete Multimodal-Diffusion from Graph Spectrum Perspective for Conversation Emotion Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于对话情感识别任务，解决模态缺失问题。通过图谱扩散网络GSDNet恢复缺失模态数据，提升情感识别性能。**

- **链接: [http://arxiv.org/pdf/2506.12325v1](http://arxiv.org/pdf/2506.12325v1)**

> **作者:** Yuntao Shou; Jun Yao; Tao Meng; Wei Ai; Cen Chen; Keqin Li
>
> **摘要:** Multimodal emotion recognition in conversations (MERC) aims to infer the speaker's emotional state by analyzing utterance information from multiple sources (i.e., video, audio, and text). Compared with unimodality, a more robust utterance representation can be obtained by fusing complementary semantic information from different modalities. However, the modality missing problem severely limits the performance of MERC in practical scenarios. Recent work has achieved impressive performance on modality completion using graph neural networks and diffusion models, respectively. This inspires us to combine these two dimensions through the graph diffusion model to obtain more powerful modal recovery capabilities. Unfortunately, existing graph diffusion models may destroy the connectivity and local structure of the graph by directly adding Gaussian noise to the adjacency matrix, resulting in the generated graph data being unable to retain the semantic and topological information of the original graph. To this end, we propose a novel Graph Spectral Diffusion Network (GSDNet), which maps Gaussian noise to the graph spectral space of missing modalities and recovers the missing data according to its original distribution. Compared with previous graph diffusion methods, GSDNet only affects the eigenvalues of the adjacency matrix instead of destroying the adjacency matrix directly, which can maintain the global topological information and important spectral features during the diffusion process. Extensive experiments have demonstrated that GSDNet achieves state-of-the-art emotion recognition performance in various modality loss scenarios.
>
---
#### [new 004] SONIC: Sound Optimization for Noise In Crowds
- **分类: cs.SD; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决嘈杂环境中语音清晰度问题。提出SONIC系统，采用自适应滤波技术实现低功耗实时降噪。**

- **链接: [http://arxiv.org/pdf/2506.13272v1](http://arxiv.org/pdf/2506.13272v1)**

> **作者:** Pranav M N; Gandham Sai Santhosh; Tejas Joshi; S Sriniketh Desikan; Eswar Gupta
>
> **摘要:** This paper presents SONIC, an embedded real-time noise suppression system implemented on the ARM Cortex-M7-based STM32H753ZI microcontroller. Using adaptive filtering (LMS), the system improves speech intelligibility in noisy environments. SONIC focuses on a novel approach to noise suppression in audio signals, specifically addressing the limitations of traditional Active Noise Cancellation (ANC) systems. The paper explores various signal processing algorithms in a micro-controller point of view, highlighting various performance factors and which were considered optimal in our embedded system. Additionally we also discussed the system architecture, explaining how the MCU's efficiency was harnessed, along with an in-depth overview of how the audio signals were translated within the processor. The results demonstrate improved speech clarity and practical real-time performance, showing low-power DSP as an alternative to complex AI denoising methods.
>
---
#### [new 005] StreamMel: Real-Time Zero-shot Text-to-Speech via Interleaved Continuous Autoregressive Modeling
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决实时生成与高保真度的问题。提出StreamMel框架，实现单阶段连续频谱建模，提升实时性与语音质量。**

- **链接: [http://arxiv.org/pdf/2506.12570v1](http://arxiv.org/pdf/2506.12570v1)**

> **作者:** Hui Wang; Yifan Yang; Shujie Liu; Jinyu Li; Lingwei Meng; Yanqing Liu; Jiaming Zhou; Haoqin Sun; Yan Lu; Yong Qin
>
> **摘要:** Recent advances in zero-shot text-to-speech (TTS) synthesis have achieved high-quality speech generation for unseen speakers, but most systems remain unsuitable for real-time applications because of their offline design. Current streaming TTS paradigms often rely on multi-stage pipelines and discrete representations, leading to increased computational cost and suboptimal system performance. In this work, we propose StreamMel, a pioneering single-stage streaming TTS framework that models continuous mel-spectrograms. By interleaving text tokens with acoustic frames, StreamMel enables low-latency, autoregressive synthesis while preserving high speaker similarity and naturalness. Experiments on LibriSpeech demonstrate that StreamMel outperforms existing streaming TTS baselines in both quality and latency. It even achieves performance comparable to offline systems while supporting efficient real-time generation, showcasing broad prospects for integration with real-time speech large language models. Audio samples are available at: https://aka.ms/StreamMel.
>
---
#### [new 006] Video-Guided Text-to-Music Generation Using Public Domain Movie Collections
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于视频引导的文本到音乐生成任务，旨在解决电影配乐中多因素融合不足的问题。通过构建OSSL数据集并引入视频适配器提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12573v1](http://arxiv.org/pdf/2506.12573v1)**

> **作者:** Haven Kim; Zachary Novack; Weihan Xu; Julian McAuley; Hao-Wen Dong
>
> **备注:** ISMIR 2025 regular paper. Dataset and code available at https://havenpersona.github.io/ossl-v1
>
> **摘要:** Despite recent advancements in music generation systems, their application in film production remains limited, as they struggle to capture the nuances of real-world filmmaking, where filmmakers consider multiple factors-such as visual content, dialogue, and emotional tone-when selecting or composing music for a scene. This limitation primarily stems from the absence of comprehensive datasets that integrate these elements. To address this gap, we introduce Open Screen Sound Library (OSSL), a dataset consisting of movie clips from public domain films, totaling approximately 36.5 hours, paired with high-quality soundtracks and human-annotated mood information. To demonstrate the effectiveness of our dataset in improving the performance of pre-trained models on film music generation tasks, we introduce a new video adapter that enhances an autoregressive transformer-based text-to-music model by adding video-based conditioning. Our experimental results demonstrate that our proposed approach effectively enhances MusicGen-Medium in terms of both objective measures of distributional and paired fidelity, and subjective compatibility in mood and genre. The dataset and code are available at https://havenpersona.github.io/ossl-v1.
>
---
#### [new 007] Personalizable Long-Context Symbolic Music Infilling with MIDI-RWKV
- **分类: cs.SD; cs.LG; cs.MM; eess.AS; I.2.1; I.2.6; H.5.5; J.5**

- **简介: 该论文属于音乐生成任务，旨在解决计算机辅助创作中缺乏交互性的问题。提出MIDI-RWKV模型，实现个性化、多轨、长上下文的音乐补全。**

- **链接: [http://arxiv.org/pdf/2506.13001v1](http://arxiv.org/pdf/2506.13001v1)**

> **作者:** Christian Zhou-Zheng; Philippe Pasquier
>
> **摘要:** Existing work in automatic music generation has primarily focused on end-to-end systems that produce complete compositions or continuations. However, because musical composition is typically an iterative process, such systems make it difficult to engage in the back-and-forth between human and machine that is essential to computer-assisted creativity. In this study, we address the task of personalizable, multi-track, long-context, and controllable symbolic music infilling to enhance the process of computer-assisted composition. We present MIDI-RWKV, a novel model based on the RWKV-7 linear architecture, to enable efficient and coherent musical cocreation on edge devices. We also demonstrate that MIDI-RWKV admits an effective method of finetuning its initial state for personalization in the very-low-sample regime. We evaluate MIDI-RWKV and its state tuning on several quantitative and qualitative metrics, and release model weights and code at https://github.com/christianazinn/MIDI-RWKV.
>
---
#### [new 008] SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频自监督学习任务，旨在解决SSL模型在复杂多音音频中的泛化能力不足问题。通过引入SSLAM方法，提升模型在多音场景下的性能。**

- **链接: [http://arxiv.org/pdf/2506.12222v1](http://arxiv.org/pdf/2506.12222v1)**

> **作者:** Tony Alex; Sara Ahmed; Armin Mustafa; Muhammad Awais; Philip JB Jackson
>
> **备注:** Accepted at ICLR 2025. Code and pre-trained models are available at \url{https://github.com/ta012/SSLAM}
>
> **摘要:** Self-supervised pre-trained audio networks have seen widespread adoption in real-world systems, particularly in multi-modal large language models. These networks are often employed in a frozen state, under the assumption that the SSL pre-training has sufficiently equipped them to handle real-world audio. However, a critical question remains: how well do these models actually perform in real-world conditions, where audio is typically polyphonic and complex, involving multiple overlapping sound sources? Current audio SSL methods are often benchmarked on datasets predominantly featuring monophonic audio, such as environmental sounds, and speech. As a result, the ability of SSL models to generalize to polyphonic audio, a common characteristic in natural scenarios, remains underexplored. This limitation raises concerns about the practical robustness of SSL models in more realistic audio settings. To address this gap, we introduce Self-Supervised Learning from Audio Mixtures (SSLAM), a novel direction in audio SSL research, designed to improve, designed to improve the model's ability to learn from polyphonic data while maintaining strong performance on monophonic data. We thoroughly evaluate SSLAM on standard audio SSL benchmark datasets which are predominantly monophonic and conduct a comprehensive comparative analysis against SOTA methods using a range of high-quality, publicly available polyphonic datasets. SSLAM not only improves model performance on polyphonic audio, but also maintains or exceeds performance on standard audio SSL benchmarks. Notably, it achieves up to a 3.9\% improvement on the AudioSet-2M (AS-2M), reaching a mean average precision (mAP) of 50.2. For polyphonic datasets, SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1\% (mAP).
>
---
#### [new 009] TuneGenie: Reasoning-based LLM agents for preferential music generation
- **分类: cs.SD; cs.MA; eess.AS; I.2.6**

- **简介: 该论文属于音乐生成任务，旨在通过LLM分析用户偏好并生成有效提示，用于音乐创作。**

- **链接: [http://arxiv.org/pdf/2506.12083v1](http://arxiv.org/pdf/2506.12083v1)**

> **作者:** Amitesh Pandey; Jafarbek Arifdjanov; Ansh Tiwari
>
> **备注:** 15 pages
>
> **摘要:** Recently, Large language models (LLMs) have shown great promise across a diversity of tasks, ranging from generating images to reasoning spatially. Considering their remarkable (and growing) textual reasoning capabilities, we investigate LLMs' potency in conducting analyses of an individual's preferences in music (based on playlist metadata, personal write-ups, etc.) and producing effective prompts (based on these analyses) to be passed to Suno AI (a generative AI tool for music production). Our proposition of a novel LLM-based textual representation to music model (which we call TuneGenie) and the various methods we develop to evaluate & benchmark similar models add to the increasing (and increasingly controversial) corpus of research on the use of AI in generating art.
>
---
#### [new 010] I$^2$S-TFCKD: Intra-Inter Set Knowledge Distillation with Time-Frequency Calibration for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决低复杂度模型性能不足的问题。提出I²S-TFCKD框架，通过时间-频率校准的知识蒸馏提升模型效果。**

- **链接: [http://arxiv.org/pdf/2506.13127v1](http://arxiv.org/pdf/2506.13127v1)**

> **作者:** Jiaming Cheng; Ruiyu Liang; Chao Xu; Ye Ni; Wei Zhou; Björn W. Schuller; Xiaoshuai Hao
>
> **备注:** submitted to IEEE Transactions on Neural Networks and Learning Systems
>
> **摘要:** In recent years, complexity compression of neural network (NN)-based speech enhancement (SE) models has gradually attracted the attention of researchers, especially in scenarios with limited hardware resources or strict latency requirements. The main difficulties and challenges lie in achieving a balance between complexity and performance according to the characteristics of the task. In this paper, we propose an intra-inter set knowledge distillation (KD) framework with time-frequency calibration (I$^2$S-TFCKD) for SE. Different from previous distillation strategies for SE, the proposed framework fully utilizes the time-frequency differential information of speech while promoting global knowledge flow. Firstly, we propose a multi-layer interactive distillation based on dual-stream time-frequency cross-calibration, which calculates the teacher-student similarity calibration weights in the time and frequency domains respectively and performs cross-weighting, thus enabling refined allocation of distillation contributions across different layers according to speech characteristics. Secondly, we construct a collaborative distillation paradigm for intra-set and inter-set correlations. Within a correlated set, multi-layer teacher-student features are pairwise matched for calibrated distillation. Subsequently, we generate representative features from each correlated set through residual fusion to form the fused feature set that enables inter-set knowledge interaction. The proposed distillation strategy is applied to the dual-path dilated convolutional recurrent network (DPDCRN) that ranked first in the SE track of the L3DAS23 challenge. Objective evaluations demonstrate that the proposed KD strategy consistently and effectively improves the performance of the low-complexity student model and outperforms other distillation schemes.
>
---
#### [new 011] Improving Speech Enhancement with Multi-Metric Supervision from Learned Quality Assessment
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决传统目标与感知质量不一致的问题。通过引入多指标监督的语音质量评估模型，提升语音增强效果。**

- **链接: [http://arxiv.org/pdf/2506.12260v1](http://arxiv.org/pdf/2506.12260v1)**

> **作者:** Wei Wang; Wangyou Zhang; Chenda Li; Jiatong Shi; Shinji Watanabe; Yanmin Qian
>
> **备注:** Submitted to ASRU 2025
>
> **摘要:** Speech quality assessment (SQA) aims to predict the perceived quality of speech signals under a wide range of distortions. It is inherently connected to speech enhancement (SE), which seeks to improve speech quality by removing unwanted signal components. While SQA models are widely used to evaluate SE performance, their potential to guide SE training remains underexplored. In this work, we investigate a training framework that leverages a SQA model, trained to predict multiple evaluation metrics from a public SE leaderboard, as a supervisory signal for SE. This approach addresses a key limitation of conventional SE objectives, such as SI-SNR, which often fail to align with perceptual quality and generalize poorly across evaluation metrics. Moreover, it enables training on real-world data where clean references are unavailable. Experiments on both simulated and real-world test sets show that SQA-guided training consistently improves performance across a range of quality metrics.
>
---
#### [new 012] ViSAGe: Video-to-Spatial Audio Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于视频到空间音频生成任务，旨在从无声视频直接生成第一阶全向声场。工作包括构建数据集、提出新评估指标，并设计端到端框架ViSAGe。**

- **链接: [http://arxiv.org/pdf/2506.12199v1](http://arxiv.org/pdf/2506.12199v1)**

> **作者:** Jaeyeon Kim; Heeseung Yun; Gunhee Kim
>
> **备注:** ICLR 2025. Project page: https://jaeyeonkim99.github.io/visage/
>
> **摘要:** Spatial audio is essential for enhancing the immersiveness of audio-visual experiences, yet its production typically demands complex recording systems and specialized expertise. In this work, we address a novel problem of generating first-order ambisonics, a widely used spatial audio format, directly from silent videos. To support this task, we introduce YT-Ambigen, a dataset comprising 102K 5-second YouTube video clips paired with corresponding first-order ambisonics. We also propose new evaluation metrics to assess the spatial aspect of generated audio based on audio energy maps and saliency metrics. Furthermore, we present Video-to-Spatial Audio Generation (ViSAGe), an end-to-end framework that generates first-order ambisonics from silent video frames by leveraging CLIP visual features, autoregressive neural audio codec modeling with both directional and visual guidance. Experimental results demonstrate that ViSAGe produces plausible and coherent first-order ambisonics, outperforming two-stage approaches consisting of video-to-audio generation and audio spatialization. Qualitative examples further illustrate that ViSAGe generates temporally aligned high-quality spatial audio that adapts to viewpoint changes.
>
---
#### [new 013] Methods for pitch analysis in contemporary popular music: multiple pitches from harmonic tones in Vitalic's music
- **分类: cs.SD; 00A65; J.5**

- **简介: 该论文属于音乐信号处理任务，研究当代流行音乐中单个谐波音色产生的多个感知音高现象。通过实验分析音高感知与信号特征的关系，探讨其在音乐创作中的应用。**

- **链接: [http://arxiv.org/pdf/2506.12405v1](http://arxiv.org/pdf/2506.12405v1)**

> **作者:** Emmanuel Deruty; David Meredith; Maarten Grachten; Pascal Arbez-Nicolas; Andreas Hasselholt Jørgensen; Oliver Søndermølle Hansen; Magnus Stensli; Christian Nørkær Petersen
>
> **备注:** Pending review, Journal of the Audio Engineering Society
>
> **摘要:** Aims. This study suggests that the use of multiple perceived pitches arising from a single harmonic complex tone is an active and intentional feature of contemporary popular music. The phenomenon is illustrated through examples drawn from the work of electronic artist Vitalic and others. Methods. Two listening tests were conducted: (1) evaluation of the number of simultaneous pitches perceived from single harmonic tones, and (2) manual pitch transcription of sequences of harmonic tones. Relationships between signal characteristics and pitch perception were then analyzed. Results. The synthetic harmonic tones found in the musical sequences under study were observed to transmit more perceived pitches than their acoustic counterparts, with significant variation across listeners. Multiple ambiguous pitches were associated with tone properties such as prominent upper partials and particular autocorrelation profiles. Conclusions. Harmonic tones in a context of contemporary popular music can, in general, convey several ambiguous pitches. The set of perceived pitches depends on both the listener and the listening conditions.
>
---
#### [new 014] Persistent Homology of Music Network with Three Different Distances
- **分类: cs.SD; cs.CG; eess.AS**

- **简介: 该论文属于音乐数据分析任务，旨在研究不同距离定义对音乐图持久同调的影响，通过比较三种距离方法验证其在拓扑结构中的差异。**

- **链接: [http://arxiv.org/pdf/2506.13595v1](http://arxiv.org/pdf/2506.13595v1)**

> **作者:** Eunwoo Heo; Byeongchan Choi; Myung ock Kim; Mai Lan Tran; Jae-Hun Jung
>
> **摘要:** Persistent homology has been widely used to discover hidden topological structures in data across various applications, including music data. To apply persistent homology, a distance or metric must be defined between points in a point cloud or between nodes in a graph network. These definitions are not unique and depend on the specific objectives of a given problem. In other words, selecting different metric definitions allows for multiple topological inferences. In this work, we focus on applying persistent homology to music graph with predefined weights. We examine three distinct distance definitions based on edge-wise pathways and demonstrate how these definitions affect persistent barcodes, persistence diagrams, and birth/death edges. We found that there exist inclusion relations in one-dimensional persistent homology reflected on persistence barcode and diagram among these three distance definitions. We verified these findings using real music data.
>
---
#### [new 015] SC-SOT: Conditioning the Decoder on Diarized Speaker Information for End-to-End Overlapped Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于端到端多说话人语音识别任务，旨在解决重叠语音识别问题。通过引入说话人信息增强解码器，提升识别效果。**

- **链接: [http://arxiv.org/pdf/2506.12672v1](http://arxiv.org/pdf/2506.12672v1)**

> **作者:** Yuta Hirano; Sakriani Sakti
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We propose Speaker-Conditioned Serialized Output Training (SC-SOT), an enhanced SOT-based training for E2E multi-talker ASR. We first probe how SOT handles overlapped speech, and we found the decoder performs implicit speaker separation. We hypothesize this implicit separation is often insufficient due to ambiguous acoustic cues in overlapping regions. To address this, SC-SOT explicitly conditions the decoder on speaker information, providing detailed information about "who spoke when". Specifically, we enhance the decoder by incorporating: (1) speaker embeddings, which allow the model to focus on the acoustic characteristics of the target speaker, and (2) speaker activity information, which guides the model to suppress non-target speakers. The speaker embeddings are derived from a jointly trained E2E speaker diarization model, mitigating the need for speaker enrollment. Experimental results demonstrate the effectiveness of our conditioning approach on overlapped speech.
>
---
#### [new 016] Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决Whisper模型不支持流式识别的问题。通过两阶段解码结构和混合分词方法，提升其流式识别能力。**

- **链接: [http://arxiv.org/pdf/2506.12154v1](http://arxiv.org/pdf/2506.12154v1)**

> **作者:** Haoran Zhou; Xingchen Song; Brendan Fahy; Qiaochu Song; Binbin Zhang; Zhendong Peng; Anshul Wadhawan; Denglin Jiang; Apurv Verma; Vinay Ramesh; Srivas Prasad; Michele M. Franceschini
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** OpenAI Whisper is a family of robust Automatic Speech Recognition (ASR) models trained on 680,000 hours of audio. However, its encoder-decoder architecture, trained with a sequence-to-sequence objective, lacks native support for streaming ASR. In this paper, we fine-tune Whisper for streaming ASR using the WeNet toolkit by adopting a Unified Two-pass (U2) structure. We introduce an additional Connectionist Temporal Classification (CTC) decoder trained with causal attention masks to generate streaming partial transcripts, while the original Whisper decoder reranks these partial outputs. Our experiments on LibriSpeech and an earnings call dataset demonstrate that, with adequate fine-tuning data, Whisper can be adapted into a capable streaming ASR model. We also introduce a hybrid tokenizer approach, which uses a smaller token space for the CTC decoder while retaining Whisper's original token space for the attention decoder, resulting in improved data efficiency and generalization.
>
---
#### [new 017] Phonikud: Hebrew Grapheme-to-Phoneme Conversion for Real-Time Text-to-Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于 Hebrew G2P 任务，解决实时 TTS 中因拼写复杂导致的语音转换不准确问题。提出 Phonikud 系统，采用轻量适配器提升准确性，并发布 ILSpeech 数据集。**

- **链接: [http://arxiv.org/pdf/2506.12311v1](http://arxiv.org/pdf/2506.12311v1)**

> **作者:** Yakov Kolani; Maxim Melichov; Cobi Calev; Morris Alper
>
> **备注:** Project page: https://phonikud.github.io
>
> **摘要:** Real-time text-to-speech (TTS) for Modern Hebrew is challenging due to the language's orthographic complexity. Existing solutions ignore crucial phonetic features such as stress that remain underspecified even when vowel marks are added. To address these limitations, we introduce Phonikud, a lightweight, open-source Hebrew grapheme-to-phoneme (G2P) system that outputs fully-specified IPA transcriptions. Our approach adapts an existing diacritization model with lightweight adaptors, incurring negligible additional latency. We also contribute the ILSpeech dataset of transcribed Hebrew speech with IPA annotations, serving as a benchmark for Hebrew G2P and as training data for TTS systems. Our results demonstrate that Phonikud G2P conversion more accurately predicts phonemes from Hebrew text compared to prior methods, and that this enables training of effective real-time Hebrew TTS models with superior speed-accuracy trade-offs. We release our code, data, and models at https://phonikud.github.io.
>
---
#### [new 018] Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音与文本对齐任务，旨在解决失语症语音与目标文本的准确对齐问题。提出Neural LCS方法，提升对齐精度和分割效果。**

- **链接: [http://arxiv.org/pdf/2506.12073v1](http://arxiv.org/pdf/2506.12073v1)**

> **作者:** Zongli Ye; Jiachen Lian; Xuanru Zhou; Jinming Zhang; Haodong Li; Shuhe Li; Chenxu Guo; Anaisha Das; Peter Park; Zoe Ezzes; Jet Vonk; Brittany Morin; Rian Bogley; Lisa Wauters; Zachary Miller; Maria Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** Accepted for Interspeech2025
>
> **摘要:** Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment.
>
---
#### [new 019] Exploring Audio Cues for Enhanced Test-Time Video Model Adaptation
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于视频分类任务，旨在提升测试时自适应（TTA）性能。通过引入音频信息生成伪标签，增强模型适应能力。**

- **链接: [http://arxiv.org/pdf/2506.12481v1](http://arxiv.org/pdf/2506.12481v1)**

> **作者:** Runhao Zeng; Qi Deng; Ronghao Zhang; Shuaicheng Niu; Jian Chen; Xiping Hu; Victor C. M. Leung
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Test-time adaptation (TTA) aims to boost the generalization capability of a trained model by conducting self-/unsupervised learning during the testing phase. While most existing TTA methods for video primarily utilize visual supervisory signals, they often overlook the potential contribution of inherent audio data. To address this gap, we propose a novel approach that incorporates audio information into video TTA. Our method capitalizes on the rich semantic content of audio to generate audio-assisted pseudo-labels, a new concept in the context of video TTA. Specifically, we propose an audio-to-video label mapping method by first employing pre-trained audio models to classify audio signals extracted from videos and then mapping the audio-based predictions to video label spaces through large language models, thereby establishing a connection between the audio categories and video labels. To effectively leverage the generated pseudo-labels, we present a flexible adaptation cycle that determines the optimal number of adaptation iterations for each sample, based on changes in loss and consistency across different views. This enables a customized adaptation process for each sample. Experimental results on two widely used datasets (UCF101-C and Kinetics-Sounds-C), as well as on two newly constructed audio-video TTA datasets (AVE-C and AVMIT-C) with various corruption types, demonstrate the superiority of our approach. Our method consistently improves adaptation performance across different video classification models and represents a significant step forward in integrating audio information into video TTA. Code: https://github.com/keikeiqi/Audio-Assisted-TTA.
>
---
#### [new 020] Magnetoencephalography (MEG) Based Non-Invasive Chinese Speech Decoding
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于非侵入式中文语音解码任务，旨在解决中文语音BCI研究不足的问题，提出多模态辅助解码算法并构建相关数据集。**

- **链接: [http://arxiv.org/pdf/2506.12817v1](http://arxiv.org/pdf/2506.12817v1)**

> **作者:** Zhihong Jia; Hongbin Wang; Yuanzhong Shen; Feng Hu; Jiayu An; Kai Shu; Dongrui Wu
>
> **摘要:** As an emerging paradigm of brain-computer interfaces (BCIs), speech BCI has the potential to directly reflect auditory perception and thoughts, offering a promising communication alternative for patients with aphasia. Chinese is one of the most widely spoken languages in the world, whereas there is very limited research on speech BCIs for Chinese language. This paper reports a text-magnetoencephalography (MEG) dataset for non-invasive Chinese speech BCIs. It also proposes a multi-modality assisted speech decoding (MASD) algorithm to capture both text and acoustic information embedded in brain signals during speech activities. Experiment results demonstrated the effectiveness of both our text-MEG dataset and our proposed MASD algorithm. To our knowledge, this is the first study on modality-assisted decoding for non-invasive speech BCIs.
>
---
#### [new 021] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **简介: 该论文属于人工智能与通信技术融合任务，旨在解决大模型资源消耗高和通信需求大的问题，提出AI Flow框架实现高效智能服务。**

- **链接: [http://arxiv.org/pdf/2506.12479v1](http://arxiv.org/pdf/2506.12479v1)**

> **作者:** Hongjun An; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [new 022] SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音频-语言模型的逻辑推理任务，旨在解决音频模态下推理能力不足的问题。通过构建ALR数据集并提出SoundMind算法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12935v1](http://arxiv.org/pdf/2506.12935v1)**

> **作者:** Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Peijun Qing; Soroush Vosoughi; Jiang Gui
>
> **摘要:** While large language models have shown reasoning capabilities, their application to the audio modality, particularly in large audio-language models (ALMs), remains significantly underdeveloped. Addressing this gap requires a systematic approach, involving a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this study, we present a comprehensive solution: we introduce the Audio Logical Reasoning (ALR) dataset, consisting of 6,446 text-audio annotated samples specifically designed for complex reasoning tasks. Building on this resource, we propose SoundMind, a rule-based reinforcement learning (RL) algorithm tailored to endow ALMs with deep bimodal reasoning abilities. By training Qwen2.5-Omni-7B on the ALR dataset using SoundMind, our approach achieves state-of-the-art performance in audio logical reasoning. This work highlights the impact of combining high-quality, reasoning-focused datasets with specialized RL techniques, advancing the frontier of auditory intelligence in language models. Our code and the proposed dataset are available at https://github.com/xid32/SoundMind.
>
---
#### [new 023] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别与说话人分割任务，解决ASR和SD-ASR问题。提出多阶段训练方法提升模型推理与自纠错能力，取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2506.13300v1](http://arxiv.org/pdf/2506.13300v1)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [new 024] ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决零样本TTS模型推理速度慢的问题，提出ZipVoice模型，通过流匹配和压缩设计实现高质量且快速的语音生成。**

- **链接: [http://arxiv.org/pdf/2506.13053v1](http://arxiv.org/pdf/2506.13053v1)**

> **作者:** Han Zhu; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhaoqing Li; Weiji Zhuang; Long Lin; Daniel Povey
>
> **摘要:** Existing large-scale zero-shot text-to-speech (TTS) models deliver high speech quality but suffer from slow inference speeds due to massive parameters. To address this issue, this paper introduces ZipVoice, a high-quality flow-matching-based zero-shot TTS model with a compact model size and fast inference speed. Key designs include: 1) a Zipformer-based flow-matching decoder to maintain adequate modeling capabilities under constrained size; 2) Average upsampling-based initial speech-text alignment and Zipformer-based text encoder to improve speech intelligibility; 3) A flow distillation method to reduce sampling steps and eliminate the inference overhead associated with classifier-free guidance. Experiments on 100k hours multilingual datasets show that ZipVoice matches state-of-the-art models in speech quality, while being 3 times smaller and up to 30 times faster than a DiT-based flow-matching baseline. Codes, model checkpoints and demo samples are publicly available.
>
---
#### [new 025] CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出CMI-Bench，用于评估音频文本大模型在音乐信息检索任务中的表现，解决现有基准不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.12285v1](http://arxiv.org/pdf/2506.12285v1)**

> **作者:** Yinghao Ma; Siyou Li; Juntao Yu; Emmanouil Benetos; Akira Maezawa
>
> **备注:** Accepted by ISMIR 2025
>
> **摘要:** Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multi-choice evaluations that fail to reflect the complexity of real-world music analysis. We reinterpret a broad range of traditional MIR annotations as instruction-following formats and introduce CMI-Bench, a comprehensive music instruction following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. These include genre classification, emotion regression, emotion tagging, instrument classification, pitch estimation, key detection, lyrics transcription, melody extraction, vocal technique recognition, instrument performance technique detection, music tagging, music captioning, and (down)beat tracking: reflecting core challenges in MIR research. Unlike previous benchmarks, CMI-Bench adopts standardized evaluation metrics consistent with previous state-of-the-art MIR models, ensuring direct comparability with supervised approaches. We provide an evaluation toolkit supporting all open-source audio-textual LLMs, including LTU, Qwen-audio, SALMONN, MusiLingo, etc. Experiment results reveal significant performance gaps between LLMs and supervised models, along with their culture, chronological and gender bias, highlighting the potential and limitations of current models in addressing MIR tasks. CMI-Bench establishes a unified foundation for evaluating music instruction following, driving progress in music-aware LLMs.
>
---
#### [new 026] Boundary-Informed Sound Field Reconstruction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声场重建任务，解决在边界信息不全时的高精度声场恢复问题。通过引入边界先验信息，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2506.13279v1](http://arxiv.org/pdf/2506.13279v1)**

> **作者:** David Sundström; Filip Elvander; Andreas Jakobsson
>
> **备注:** Accepted for publication at EUSIPCO 2025
>
> **摘要:** We consider the problem of reconstructing the sound field in a room using prior information of the boundary geometry, represented as a point cloud. In general, when no boundary information is available, an accurate sound field reconstruction over a large spatial region and at high frequencies requires numerous microphone measurements. On the other hand, if all geometrical and acoustical aspects of the boundaries are known, the sound field could, in theory, be simulated without any measurements. In this work, we address the intermediate case, where only partial or uncertain boundary information is available. This setting is similar to one studied in virtual reality applications, where the goal is to create a perceptually convincing audio experience. In this work, we focus on spatial sound control applications, which in contrast require an accurate sound field reconstruction. Therefore, we formulate the problem within a linear Bayesian framework, incorporating a boundary-informed prior derived from impedance boundary conditions. The formulation allows for joint optimization of the unknown hyperparameters, including the noise and signal variances and the impedance boundary conditions. Using numerical experiments, we show that incorporating the boundary-informed prior significantly enhances the reconstruction, notably even when only a few hundreds of boundary points are available or when the boundary positions are calibrated with an uncertainty up to 1 dm.
>
---
#### [new 027] Frequency Dynamic Convolutions for Sound Event Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声事件检测任务，旨在解决传统卷积模型在处理频率依赖特性时的不足，提出多种频率自适应卷积方法以提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.12785v1](http://arxiv.org/pdf/2506.12785v1)**

> **作者:** Hyeonuk Nam
>
> **备注:** Ph. D. Dissertation in English(KAIST)
>
> **摘要:** Recent research in deep learning-based Sound Event Detection (SED) has primarily focused on Convolutional Recurrent Neural Networks (CRNNs) and Transformer models. However, conventional 2D convolution-based models assume shift invariance along both the temporal and frequency axes, leadin to inconsistencies when dealing with frequency-dependent characteristics of acoustic signals. To address this issue, this study proposes Frequency Dynamic Convolution (FDY conv), which dynamically adjusts convolutional kernels based on the frequency composition of the input signal to enhance SED performance. FDY conv constructs an optimal frequency response by adaptively weighting multiple basis kernels based on frequency-specific attention weights. Experimental results show that applying FDY conv to CRNNs improves performance on the DESED dataset by 7.56% compared to the baseline CRNN. However, FDY conv has limitations in that it combines basis kernels of the same shape across all frequencies, restricting its ability to capture diverse frequency-specific characteristics. Additionally, the $3\times3$ basis kernel size is insufficient to capture a broader frequency range. To overcome these limitations, this study introduces an extended family of FDY conv models. Dilated FDY conv (DFD conv) applies convolutional kernels with various dilation rates to expand the receptive field along the frequency axis and enhance frequency-specific feature representation. Experimental results show that DFD conv improves performance by 9.27% over the baseline. Partial FDY conv (PFD conv) addresses the high computational cost of FDY conv, which results from performing all convolution operations with dynamic kernels. Since FDY conv may introduce unnecessary adaptivity for quasi-stationary sound events, PFD conv integrates standard 2D convolutions with frequency-adaptive kernels to reduce computational complexity while maintaining performance. Experimental results demonstrate that PFD conv improves performance by 7.80% over the baseline while reducing the number of parameters by 54.4% compared to FDY conv. Multi-Dilated FDY conv (MDFD conv) extends DFD conv by addressing its structural limitation of applying the same dilation across all frequencies. By utilizing multiple convolutional kernels with different dilation rates, MDFD conv effectively captures diverse frequency-dependent patterns. Experimental results indicate that MDFD conv achieves the highest performance, improving the baseline CRNN performance by 10.98%. Furthermore, standard FDY conv employs Temporal Average Pooling, which assigns equal weight to all frames along the time axis, limiting its ability to effectively capture transient events. To overcome this, this study proposes TAP-FDY conv (TFD conv), which integrates Temporal Attention Pooling (TA) that focuses on salient features, Velocity Attention Pooling (VA) that emphasizes transient characteristics, and Average Pooling (AP) that captures stationary properties. TAP-FDY conv achieves the same performance as MDFD conv but reduces the number of parameters by approximately 30.01% (12.703M vs. 18.157M), achieving equivalent accuracy with lower computational complexity. Class-wise performance analysis reveals that FDY conv improves detection of non-stationary events, DFD conv is particularly effective for events with broad spectral features, and PFD conv enhances the detection of quasi-stationary events. Additionally, TFD conv (TFD-CRNN) demonstrates strong performance in detecting transient events. In the case studies, PFD conv effectively captures stable signal patterns in tank powertrain fault recognition, DFD conv recognizes wide harmonic spectral patterns on speed-varying motor fault recognition, while TFD conv outperforms other models in detecting transient signals in offshore arc detection. These results suggest that frequency-adaptive convolutions and their extended variants provide a robust alternative to conventional 2D convolutions in deep learning-based audio processing.
>
---
#### [new 028] Evaluating Logit-Based GOP Scores for Mispronunciation Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音识别中的发音评估任务，旨在解决传统GOP评分不足的问题。通过比较基于logit和概率的GOP方法，提升误读检测效果。**

- **链接: [http://arxiv.org/pdf/2506.12067v1](http://arxiv.org/pdf/2506.12067v1)**

> **作者:** Aditya Kamlesh Parikh; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** Accepted to Interspeech 2025. This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** Pronunciation assessment relies on goodness of pronunciation (GOP) scores, traditionally derived from softmax-based posterior probabilities. However, posterior probabilities may suffer from overconfidence and poor phoneme separation, limiting their effectiveness. This study compares logit-based GOP scores with probability-based GOP scores for mispronunciation detection. We conducted our experiment on two L2 English speech datasets spoken by Dutch and Mandarin speakers, assessing classification performance and correlation with human ratings. Logit-based methods outperform probability-based GOP in classification, but their effectiveness depends on dataset characteristics. The maximum logit GOP shows the strongest alignment with human perception, while a combination of different GOP scores balances probability and logit features. The findings suggest that hybrid GOP methods incorporating uncertainty modeling and phoneme-specific weighting improve pronunciation assessment.
>
---
#### [new 029] Mitigating Non-Target Speaker Bias in Guided Speaker Embedding
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于说话人识别任务，解决多说话人场景下非目标说话人干扰导致的嵌入质量下降问题，通过改进统计模块提升验证与聚类性能。**

- **链接: [http://arxiv.org/pdf/2506.12500v1](http://arxiv.org/pdf/2506.12500v1)**

> **作者:** Shota Horiguchi; Takanori Ashihara; Marc Delcroix; Atsushi Ando; Naohiro Tawara
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Obtaining high-quality speaker embeddings in multi-speaker conditions is crucial for many applications. A recently proposed guided speaker embedding framework, which utilizes speech activities of target and non-target speakers as clues, drastically improved embeddings under severe overlap with small degradation in low-overlap cases. However, since extreme overlaps are rare in natural conversations, this degradation cannot be overlooked. This paper first reveals that the degradation is caused by the global-statistics-based modules, widely used in speaker embedding extractors, being overly sensitive to intervals containing only non-target speakers. As a countermeasure, we propose an extension of such modules that exploit the target speaker activity clues, to compute statistics from intervals where the target is active. The proposed method improves speaker verification performance in both low and high overlap ratios, and diarization performance on multiple datasets.
>
---
#### [new 030] Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
- **分类: cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于多模态交互任务，旨在解决模态对齐效率问题。提出Stream-Omni模型，通过不同方式实现视觉、语音与文本的高效对齐。**

- **链接: [http://arxiv.org/pdf/2506.13642v1](http://arxiv.org/pdf/2506.13642v1)**

> **作者:** Shaolei Zhang; Shoutao Guo; Qingkai Fang; Yan Zhou; Yang Feng
>
> **备注:** Code: https://github.com/ictnlp/Stream-Omni , Model: https://huggingface.co/ICTNLP/stream-omni-8b
>
> **摘要:** The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.
>
---
#### [new 031] Parkinson's Disease Freezing of Gait (FoG) Symptom Detection Using Machine Learning from Wearable Sensor Data
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医疗健康领域，旨在检测帕金森病患者的步态冻结（FoG）症状。通过融合Transformer和Bi-LSTM模型，利用可穿戴传感器数据进行实时识别。**

- **链接: [http://arxiv.org/pdf/2506.12561v1](http://arxiv.org/pdf/2506.12561v1)**

> **作者:** Mahmudul Hasan
>
> **摘要:** Freezing of gait (FoG) is a special symptom found in patients with Parkinson's disease (PD). Patients who have FoG abruptly lose the capacity to walk as they normally would. Accelerometers worn by patients can record movement data during these episodes, and machine learning algorithms can be useful to categorize this information. Thus, the combination may be able to identify FoG in real time. In order to identify FoG events in accelerometer data, we introduce the Transformer Encoder-Bi-LSTM fusion model in this paper. The model's capability to differentiate between FoG episodes and normal movement was used to evaluate its performance, and on the Kaggle Parkinson's Freezing of Gait dataset, the proposed Transformer Encoder-Bi-LSTM fusion model produced 92.6% accuracy, 80.9% F1 score, and 52.06% in terms of mean average precision. The findings highlight how Deep Learning-based approaches may progress the field of FoG identification and help PD patients receive better treatments and management plans.
>
---
#### [new 032] SpeechRefiner: Towards Perceptual Quality Refinement for Front-End Algorithms
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决前端算法处理后残留噪声或引入伪影的问题。提出SpeechRefiner，利用CFM提升语音感知质量。**

- **链接: [http://arxiv.org/pdf/2506.13709v1](http://arxiv.org/pdf/2506.13709v1)**

> **作者:** Sirui Li; Shuai Wang; Zhijun Liu; Zhongjie Jiang; Yannan Wang; Haizhou Li
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Speech pre-processing techniques such as denoising, de-reverberation, and separation, are commonly employed as front-ends for various downstream speech processing tasks. However, these methods can sometimes be inadequate, resulting in residual noise or the introduction of new artifacts. Such deficiencies are typically not captured by metrics like SI-SNR but are noticeable to human listeners. To address this, we introduce SpeechRefiner, a post-processing tool that utilizes Conditional Flow Matching (CFM) to improve the perceptual quality of speech. In this study, we benchmark SpeechRefiner against recent task-specific refinement methods and evaluate its performance within our internal processing pipeline, which integrates multiple front-end algorithms. Experiments show that SpeechRefiner exhibits strong generalization across diverse impairment sources, significantly enhancing speech perceptual quality. Audio demos can be found at https://speechrefiner.github.io/SpeechRefiner/.
>
---
#### [new 033] Autonomous 3D Moving Target Encirclement and Interception with Range measurement
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于无人机自主拦截任务，旨在解决非合作目标的3D围捕与拦截问题。通过距离测量和运动控制实现目标包围与中和。**

- **链接: [http://arxiv.org/pdf/2506.13106v1](http://arxiv.org/pdf/2506.13106v1)**

> **作者:** Fen Liu; Shenghai Yuan; Thien-Minh Nguyen; Rong Su
>
> **备注:** Paper has been accepted into IROS 2025
>
> **摘要:** Commercial UAVs are an emerging security threat as they are capable of carrying hazardous payloads or disrupting air traffic. To counter UAVs, we introduce an autonomous 3D target encirclement and interception strategy. Unlike traditional ground-guided systems, this strategy employs autonomous drones to track and engage non-cooperative hostile UAVs, which is effective in non-line-of-sight conditions, GPS denial, and radar jamming, where conventional detection and neutralization from ground guidance fail. Using two noisy real-time distances measured by drones, guardian drones estimate the relative position from their own to the target using observation and velocity compensation methods, based on anti-synchronization (AS) and an X$-$Y circular motion combined with vertical jitter. An encirclement control mechanism is proposed to enable UAVs to adaptively transition from encircling and protecting a target to encircling and monitoring a hostile target. Upon breaching a warning threshold, the UAVs may even employ a suicide attack to neutralize the hostile target. We validate this strategy through real-world UAV experiments and simulated analysis in MATLAB, demonstrating its effectiveness in detecting, encircling, and intercepting hostile drones. More details: https://youtu.be/5eHW56lPVto.
>
---
#### [new 034] Instance-Specific Test-Time Training for Speech Editing in the Wild
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音编辑任务，解决真实场景下语音编辑性能下降的问题。通过实例特定的测试时训练方法，提升编辑效果和语音流畅度。**

- **链接: [http://arxiv.org/pdf/2506.13295v1](http://arxiv.org/pdf/2506.13295v1)**

> **作者:** Taewoo Kim; Uijong Lee; Hayoung Park; Choongsang Cho; Nam In Park; Young Han Lee
>
> **备注:** Submitted to IEEE Signal Processing Letters
>
> **摘要:** Speech editing systems aim to naturally modify speech content while preserving acoustic consistency and speaker identity. However, previous studies often struggle to adapt to unseen and diverse acoustic conditions, resulting in degraded editing performance in real-world scenarios. To address this, we propose an instance-specific test-time training method for speech editing in the wild. Our approach employs direct supervision from ground-truth acoustic features in unedited regions, and indirect supervision in edited regions via auxiliary losses based on duration constraints and phoneme prediction. This strategy mitigates the bandwidth discontinuity problem in speech editing, ensuring smooth acoustic transitions between unedited and edited regions. Additionally, it enables precise control over speech rate by adapting the model to target durations via mask length adjustment during test-time training. Experiments on in-the-wild benchmark datasets demonstrate that our method outperforms existing speech editing systems in both objective and subjective evaluations.
>
---
#### [new 035] Towards Neural Audio Codec Source Parsing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决源编码器归属问题。提出NACSP框架，通过结构化回归预测NAC参数，提升检测精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.12627v1](http://arxiv.org/pdf/2506.12627v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Arun Balaji Buduru; Rajesh Sharma
>
> **摘要:** A new class of audio deepfakes-codecfakes (CFs)-has recently caught attention, synthesized by Audio Language Models that leverage neural audio codecs (NACs) in the backend. In response, the community has introduced dedicated benchmarks and tailored detection strategies. As the field advances, efforts have moved beyond binary detection toward source attribution, including open-set attribution, which aims to identify the NAC responsible for generation and flag novel, unseen ones during inference. This shift toward source attribution improves forensic interpretability and accountability. However, open-set attribution remains fundamentally limited: while it can detect that a NAC is unfamiliar, it cannot characterize or identify individual unseen codecs. It treats such inputs as generic ``unknowns'', lacking insight into their internal configuration. This leads to major shortcomings: limited generalization to new NACs and inability to resolve fine-grained variations within NAC families. To address these gaps, we propose Neural Audio Codec Source Parsing (NACSP) - a paradigm shift that reframes source attribution for CFs as structured regression over generative NAC parameters such as quantizers, bandwidth, and sampling rate. We formulate NACSP as a multi-task regression task for predicting these NAC parameters and establish the first comprehensive benchmark using various state-of-the-art speech pre-trained models (PTMs). To this end, we propose HYDRA, a novel framework that leverages hyperbolic geometry to disentangle complex latent properties from PTM representations. By employing task-specific attention over multiple curvature-aware hyperbolic subspaces, HYDRA enables superior multi-task generalization. Our extensive experiments show HYDRA achieves top results on benchmark CFs datasets compared to baselines operating in Euclidean space.
>
---
#### [new 036] Using Neurogram Similarity Index Measure (NSIM) to Model Hearing Loss and Cochlear Neural Degeneration
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于听力研究任务，旨在解决听力损失与耳蜗神经退化（CND）的量化问题。通过NSIM方法评估听觉神经反应，以准确反映听力表现和CND缺陷。**

- **链接: [http://arxiv.org/pdf/2506.12705v1](http://arxiv.org/pdf/2506.12705v1)**

> **作者:** Ahsan J. Cheema; Sunil Puria
>
> **备注:** Accepted for presentation at INTERSPEECH 2025
>
> **摘要:** Trouble hearing in noisy situations remains a common complaint for both individuals with hearing loss and individuals with normal hearing. This is hypothesized to arise due to condition called: cochlear neural degeneration (CND) which can also result in significant variabilities in hearing aids outcomes. This paper uses computational models of auditory periphery to simulate various hearing tasks. We present an objective method to quantify hearing loss and CND by comparing auditory nerve fiber responses using a Neurogram Similarity Index Measure (NSIM). Specifically study 1, shows that NSIM can be used to map performance of individuals with hearing loss on phoneme recognition task with reasonable accuracy. In the study 2, we show that NSIM is a sensitive measure that can also be used to capture the deficits resulting from CND and can be a candidate for noninvasive biomarker of auditory synaptopathy.
>
---
#### [new 037] CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在提升复杂场景下的识别效果。通过整合预训练模型与大语言模型，提出统一框架解决重叠语音和罕见词识别问题。**

- **链接: [http://arxiv.org/pdf/2506.12059v1](http://arxiv.org/pdf/2506.12059v1)**

> **作者:** Jiajun He; Naoki Sawada; Koichi Miyazaki; Tomoki Toda
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios.
>
---
#### [new 038] Stereo sound event localization and detection based on PSELDnet pretraining and BiMamba sequence modeling
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声事件定位与检测任务，旨在解决Transformer模型计算复杂度高的问题。通过引入BiMamba模块和非对称卷积，提升性能并降低复杂度。**

- **链接: [http://arxiv.org/pdf/2506.13455v1](http://arxiv.org/pdf/2506.13455v1)**

> **作者:** Wenmiao Gao; Yang Xiao
>
> **备注:** Technical report for DCASE 2025 Challenge Task 3
>
> **摘要:** Pre-training methods have achieved significant performance improvements in sound event localization and detection (SELD) tasks, but existing Transformer-based models suffer from high computational complexity. In this work, we propose a stereo sound event localization and detection system based on pre-trained PSELDnet and bidirectional Mamba sequence modeling. We replace the Conformer module with a BiMamba module and introduce asymmetric convolutions to more effectively model the spatiotemporal relationships between time and frequency dimensions. Experimental results demonstrate that the proposed method achieves significantly better performance than the baseline and the original PSELDnet with Conformer decoder architecture on the DCASE2025 Task 3 development dataset, while also reducing computational complexity. These findings highlight the effectiveness of the BiMamba architecture in addressing the challenges of the SELD task.
>
---
#### [new 039] Do Music Preferences Reflect Cultural Values? A Cross-National Analysis Using Music Embedding and World Values Survey
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨文化分析任务，旨在探究音乐偏好是否反映文化价值观。通过分析全球音乐数据与文化调查，发现音乐集群与文化区域显著相关。**

- **链接: [http://arxiv.org/pdf/2506.13199v1](http://arxiv.org/pdf/2506.13199v1)**

> **作者:** Yongjae Kim; Seongchan Park
>
> **摘要:** This study explores the extent to which national music preferences reflect underlying cultural values. We collected long-term popular music data from YouTube Music Charts across 62 countries, encompassing both Western and non-Western regions, and extracted audio embeddings using the CLAP model. To complement these quantitative representations, we generated semantic captions for each track using LP-MusicCaps and GPT-based summarization. Countries were clustered based on contrastive embeddings that highlight deviations from global musical norms. The resulting clusters were projected into a two-dimensional space via t-SNE for visualization and evaluated against cultural zones defined by the World Values Survey (WVS). Statistical analyses, including MANOVA and chi-squared tests, confirmed that music-based clusters exhibit significant alignment with established cultural groupings. Furthermore, residual analysis revealed consistent patterns of overrepresentation, suggesting non-random associations between specific clusters and cultural zones. These findings indicate that national-level music preferences encode meaningful cultural signals and can serve as a proxy for understanding global cultural boundaries.
>
---
#### [new 040] Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音大模型任务，旨在提升语音识别与语言建模性能。通过融合Whisper与不同LLM，优化编码器、投影器和解码器，降低WER/CER。**

- **链接: [http://arxiv.org/pdf/2506.13596v1](http://arxiv.org/pdf/2506.13596v1)**

> **作者:** Tuan Nguyen; Long-Vu Hoang; Huy-Dat Tran
>
> **备注:** Technical report for Interspeech 2025 MLC-SLM Challenge
>
> **摘要:** This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model.
>
---
## 更新

#### [replaced 001] Is Smaller Always Faster? Tradeoffs in Compressing Self-Supervised Speech Transformers
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2211.09949v3](http://arxiv.org/pdf/2211.09949v3)**

> **作者:** Tzu-Quan Lin; Tsung-Huan Yang; Chun-Yao Chang; Kuang-Ming Chen; Tzu-hsun Feng; Hung-yi Lee; Hao Tang
>
> **摘要:** Transformer-based self-supervised models have achieved remarkable success in speech processing, but their large size and high inference cost present significant challenges for real-world deployment. While numerous compression techniques have been proposed, inconsistent evaluation metrics make it difficult to compare their practical effectiveness. In this work, we conduct a comprehensive study of four common compression methods, including weight pruning, head pruning, low-rank approximation, and knowledge distillation on self-supervised speech Transformers. We evaluate each method under three key metrics: parameter count, multiply-accumulate operations, and real-time factor. Results show that each method offers distinct advantages. In addition, we contextualize recent compression techniques, comparing DistilHuBERT, FitHuBERT, LightHuBERT, ARMHuBERT, and STaRHuBERT under the same framework, offering practical guidance on compression for deployment.
>
---
#### [replaced 002] Heart Rate Classification in ECG Signals Using Machine Learning and Deep Learning
- **分类: eess.SP; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06349v2](http://arxiv.org/pdf/2506.06349v2)**

> **作者:** Thien Nhan Vo
>
> **摘要:** This study addresses the classification of heartbeats from ECG signals through two distinct approaches: traditional machine learning utilizing hand-crafted features and deep learning via transformed images of ECG beats. The dataset underwent preprocessing steps, including downsampling, filtering, and normalization, to ensure consistency and relevance for subsequent analysis. In the first approach, features such as heart rate variability (HRV), mean, variance, and RR intervals were extracted to train various classifiers, including SVM, Random Forest, AdaBoost, LSTM, Bi-directional LSTM, and LightGBM. The second approach involved transforming ECG signals into images using Gramian Angular Field (GAF), Markov Transition Field (MTF), and Recurrence Plots (RP), with these images subsequently classified using CNN architectures like VGG and Inception. Experimental results demonstrate that the LightGBM model achieved the highest performance, with an accuracy of 99% and an F1 score of 0.94, outperforming the image-based CNN approach (F1 score of 0.85). Models such as SVM and AdaBoost yielded significantly lower scores, indicating limited suitability for this task. The findings underscore the superior ability of hand-crafted features to capture temporal and morphological variations in ECG signals compared to image-based representations of individual beats. Future investigations may benefit from incorporating multi-lead ECG signals and temporal dependencies across successive beats to enhance classification accuracy further.
>
---
#### [replaced 003] A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.11130v2](http://arxiv.org/pdf/2506.11130v2)**

> **作者:** Cheng-Kang Chou; Chan-Jan Hsu; Ho-Lam Chung; Liang-Hsuan Tseng; Hsi-Chun Cheng; Yu-Kuan Fu; Kuan Po Huang; Hung-Yi Lee
>
> **摘要:** We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings.
>
---
#### [replaced 004] Performance Modeling for Correlation-based Neural Decoding of Auditory Attention to Speech
- **分类: eess.SP; cs.SD; eess.AS; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2503.09349v2](http://arxiv.org/pdf/2503.09349v2)**

> **作者:** Simon Geirnaert; Jonas Vanthornhout; Tom Francart; Alexander Bertrand
>
> **摘要:** Correlation-based auditory attention decoding (AAD) algorithms exploit neural tracking mechanisms to determine listener attention among competing speech sources via, e.g., electroencephalography signals. The correlation coefficients between the decoded neural responses and encoded speech stimuli of the different speakers then serve as AAD decision variables. A critical trade-off exists between the temporal resolution (the decision window length used to compute these correlations) and the AAD accuracy. This trade-off is typically characterized by evaluating AAD accuracy across multiple window lengths, leading to the performance curve. We propose a novel method to model this trade-off curve using labeled correlations from only a single decision window length. Our approach models the (un)attended correlations with a normal distribution after applying the Fisher transformation, enabling accurate AAD accuracy prediction across different window lengths. We validate the method on two distinct AAD implementations: a linear decoder and the non-linear VLAAI deep neural network, evaluated on separate datasets. Results show consistently low modeling errors of approximately 2 percent points, with 94% of true accuracies falling within estimated 95%-confidence intervals. The proposed method enables efficient performance curve modeling without extensive multi-window length evaluation, facilitating practical applications in, e.g., performance tracking in neuro-steered hearing devices to continuously adapt the system parameters over time.
>
---
#### [replaced 005] Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.02084v3](http://arxiv.org/pdf/2410.02084v3)**

> **作者:** Weihan Xu; Julian McAuley; Taylor Berg-Kirkpatrick; Shlomo Dubnov; Hao-Wen Dong
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Recent years have seen many audio-domain text-to-music generation models that rely on large amounts of text-audio pairs for training. However, symbolic-domain controllable music generation has lagged behind partly due to the lack of a large-scale symbolic music dataset with extensive metadata and captions. In this work, we present MetaScore, a new dataset consisting of 963K musical scores paired with rich metadata, including free-form user-annotated tags, collected from an online music forum. To approach text-to-music generation, We employ a pretrained large language model (LLM) to generate pseudo-natural language captions for music from its metadata tags. With the LLM-enhanced MetaScore, we train a text-conditioned music generation model that learns to generate symbolic music from the pseudo captions, allowing control of instruments, genre, composer, complexity and other free-form music descriptors. In addition, we train a tag-conditioned system that supports a predefined set of tags available in MetaScore. Our experimental results show that both the proposed text-to-music and tags-to-music models outperform a baseline text-to-music model in a listening test. While a concurrent work Text2MIDI also supports free-form text input, our models achieve comparable performance. Moreover, the text-to-music system offers a more natural interface than the tags-to-music model, as it allows users to provide free-form natural language prompts.
>
---
#### [replaced 006] Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21138v2](http://arxiv.org/pdf/2505.21138v2)**

> **作者:** Tianyi Xu; Hongjie Chen; Wang Qing; Lv Hang; Jian Kang; Li Jie; Zhennan Lin; Yongxiang Li; Xie Lei
>
> **摘要:** Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre-training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
>
---
#### [replaced 007] LeVo: High-Quality Song Generation with Multi-Preference Alignment
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07520v2](http://arxiv.org/pdf/2506.07520v2)**

> **作者:** Shun Lei; Yaoxun Xu; Zhiwei Lin; Huaicheng Zhang; Wei Tan; Hangting Chen; Jianwei Yu; Yixuan Zhang; Chenyu Yang; Haina Zhu; Shuai Wang; Zhiyong Wu; Dong Yu
>
> **摘要:** Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at https://levo-demo.github.io/. Code is released at https://github.com/tencent-ailab/songgeneration.
>
---
#### [replaced 008] Directional Source Separation for Robust Speech Recognition on Smart Glasses
- **分类: cs.SD; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.10993v2](http://arxiv.org/pdf/2309.10993v2)**

> **作者:** Tiantian Feng; Ju Lin; Yiteng Huang; Weipeng He; Kaustubh Kalgaonkar; Niko Moritz; Li Wan; Xin Lei; Ming Sun; Frank Seide
>
> **备注:** Published in ICASSP 2025, Hyderabad, India, 2025
>
> **摘要:** Modern smart glasses leverage advanced audio sensing and machine learning technologies to offer real-time transcribing and captioning services, considerably enriching human experiences in daily communications. However, such systems frequently encounter challenges related to environmental noises, resulting in degradation to speech recognition and speaker change detection. To improve voice quality, this work investigates directional source separation using the multi-microphone array. We first explore multiple beamformers to assist source separation modeling by strengthening the directional properties of speech signals. In addition to relying on predetermined beamformers, we investigate neural beamforming in multi-channel source separation, demonstrating that automatic learning directional characteristics effectively improves separation quality. We further compare the ASR performance leveraging separated outputs to noisy inputs. Our results show that directional source separation benefits ASR for the wearer but not for the conversation partner. Lastly, we perform the joint training of the directional source separation and ASR model, achieving the best overall ASR performance.
>
---
#### [replaced 009] Incorporating Linguistic Constraints from External Knowledge Source for Audio-Visual Target Speech Extraction
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.09792v2](http://arxiv.org/pdf/2506.09792v2)**

> **作者:** Wenxuan Wu; Shuai Wang; Xixin Wu; Helen Meng; Haizhou Li
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Audio-visual target speaker extraction (AV-TSE) models primarily rely on target visual cues to isolate the target speaker's voice from others. We know that humans leverage linguistic knowledge, such as syntax and semantics, to support speech perception. Inspired by this, we explore the potential of pre-trained speech-language models (PSLMs) and pre-trained language models (PLMs) as auxiliary knowledge sources for AV-TSE. In this study, we propose incorporating the linguistic constraints from PSLMs or PLMs for the AV-TSE model as additional supervision signals. Without introducing any extra computational cost during inference, the proposed approach consistently improves speech quality and intelligibility. Furthermore, we evaluate our method in multi-language settings and visual cue-impaired scenarios and show robust performance gains.
>
---
#### [replaced 010] Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.00861v2](http://arxiv.org/pdf/2506.00861v2)**

> **作者:** Parismita Gogoi; Vishwanath Pratap Singh; Seema Khadirnaikar; Soma Siddhartha; Sishir Kalita; Jagabandhu Mishra; Md Sahidullah; Priyankoo Sarmah; S. R. M. Prasanna
>
> **备注:** Accepted in Interspeech, All codes are available in GitHub repo https://github.com/seemark11/DhiNirnayaAMFM
>
> **摘要:** This study explores the potential of Rhythm Formant Analysis (RFA) to capture long-term temporal modulations in dementia speech. Specifically, we introduce RFA-derived rhythm spectrograms as novel features for dementia classification and regression tasks. We propose two methodologies: (1) handcrafted features derived from rhythm spectrograms, and (2) a data-driven fusion approach, integrating proposed RFA-derived rhythm spectrograms with vision transformer (ViT) for acoustic representations along with BERT-based linguistic embeddings. We compare these with existing features. Notably, our handcrafted features outperform eGeMAPs with a relative improvement of $14.2\%$ in classification accuracy and comparable performance in the regression task. The fusion approach also shows improvement, with RFA spectrograms surpassing Mel spectrograms in classification by around a relative improvement of $13.1\%$ and a comparable regression score with the baselines.
>
---
#### [replaced 011] S2ST-Omni: An Efficient and Scalable Multilingual Speech-to-Speech Translation Framework via Seamlessly Speech-Text Alignment and Streaming Speech Decoder
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.11160v2](http://arxiv.org/pdf/2506.11160v2)**

> **作者:** Yu Pan; Yuguang Yang; Yanni Hu; Jianhao Ye; Xiang Zhang; Hongbin Zhou; Lei Ma; Jianjun Zhao
>
> **备注:** Working in progress
>
> **摘要:** Multilingual speech-to-speech translation (S2ST) aims to directly convert spoken utterances from multiple source languages into fluent and intelligible speech in a target language. Despite recent progress, several critical challenges persist: 1) achieving high-quality and low-latency S2ST remains a significant obstacle; 2) most existing S2ST methods rely heavily on large-scale parallel speech corpora, which are difficult and resource-intensive to obtain. To tackle these challenges, we introduce S2ST-Omni, a novel, efficient, and scalable framework tailored for multilingual speech-to-speech translation. To enable high-quality S2TT while mitigating reliance on large-scale parallel speech corpora, we leverage powerful pretrained models: Whisper for robust audio understanding and Qwen 3.0 for advanced text comprehension. A lightweight speech adapter is introduced to bridge the modality gap between speech and text representations, facilitating effective utilization of pretrained multimodal knowledge. To ensure both translation accuracy and real-time responsiveness, we adopt a streaming speech decoder in the TTS stage, which generates the target speech in an autoregressive manner. Extensive experiments conducted on the CVSS benchmark demonstrate that S2ST-Omni consistently surpasses several state-of-the-art S2ST baselines in translation quality, highlighting its effectiveness and superiority.
>
---
#### [replaced 012] Melody predominates over harmony in the evolution of musical scales across 96 countries
- **分类: cs.SD; eess.AS; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2408.12633v2](http://arxiv.org/pdf/2408.12633v2)**

> **作者:** John M McBride; Elizabeth Phillips; Patrick E Savage; Steven Brown; Tsvi Tlusty
>
> **摘要:** The standard theory of musical scales since antiquity has been based on harmony, rather than melody. While recent analyses provide mixed support for a role of melody as well as harmony, we lack a comparative analysis based on cross-cultural data. We address this longstanding problem through a rigorous computational comparison of the main theories using 1,314 scales from 96 countries. There is near-universal support for melodic theories, which predict step-sizes of 1-3 semitones. Harmony accounts for the prevalence of certain simple-integer-ratio intervals, particularly for music-theoretic scales from Eurasian societies, which may explain their dominance amongst Western scholars. However, harmony is a poor predictor of scales measured from ethnographic recordings, particularly outside of Eurasia. Overall, we show that the historical emphasis on harmony is misguided and that melody is the primary determinant of the world's musical scales.
>
---
#### [replaced 013] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.20290v3](http://arxiv.org/pdf/2503.20290v3)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Yu Tsao; Junichi Yamagishi; Yuxuan Wang; Chao Zhang
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** This paper explores a novel perspective to speech quality assessment by leveraging natural language descriptions, offering richer, more nuanced insights than traditional numerical scoring methods. Natural language feedback provides instructive recommendations and detailed evaluations, yet existing datasets lack the comprehensive annotations needed for this approach. To bridge this gap, we introduce QualiSpeech, a comprehensive low-level speech quality assessment dataset encompassing 11 key aspects and detailed natural language comments that include reasoning and contextual insights. Additionally, we propose the QualiSpeech Benchmark to evaluate the low-level speech understanding capabilities of auditory large language models (LLMs). Experimental results demonstrate that finetuned auditory LLMs can reliably generate detailed descriptions of noise and distortion, effectively identifying their types and temporal characteristics. The results further highlight the potential for incorporating reasoning to enhance the accuracy and reliability of quality assessments. The dataset will be released at https://huggingface.co/datasets/tsinghua-ee/QualiSpeech.
>
---
#### [replaced 014] SMILE: Speech Meta In-Context Learning for Low-Resource Language Automatic Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.10429v2](http://arxiv.org/pdf/2409.10429v2)**

> **作者:** Ming-Hao Hsu; Hung-yi Lee
>
> **摘要:** Automatic Speech Recognition (ASR) models demonstrate outstanding performance on high-resource languages but face significant challenges when applied to low-resource languages due to limited training data and insufficient cross-lingual generalization. Existing adaptation strategies, such as shallow fusion, data augmentation, and direct fine-tuning, either rely on external resources, suffer computational inefficiencies, or fail in test-time adaptation scenarios. To address these limitations, we introduce Speech Meta In-Context LEarning (SMILE), an innovative framework that combines meta-learning with speech in-context learning (SICL). SMILE leverages meta-training from high-resource languages to enable robust, few-shot generalization to low-resource languages without explicit fine-tuning on the target domain. Extensive experiments on the ML-SUPERB benchmark show that SMILE consistently outperforms baseline methods, significantly reducing character and word error rates in training-free few-shot multilingual ASR tasks.
>
---
