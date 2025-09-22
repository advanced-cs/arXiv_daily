# 音频 cs.SD;  eess.SP

- **最新发布 45 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] SightSound-R1: Cross-Modal Reasoning Distillation from Vision to Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出SightSound-R1框架，旨在通过跨模态知识蒸馏将视觉语言模型的推理能力迁移至音频语言模型，以提升其在复杂声音场景中的理解与推理表现。**

- **链接: [http://arxiv.org/pdf/2509.15661v1](http://arxiv.org/pdf/2509.15661v1)**

> **作者:** Qiaolin Wang; Xilin Jiang; Linyang He; Junkai Wu; Nima Mesgarani
>
> **摘要:** While large audio-language models (LALMs) have demonstrated state-of-the-art audio understanding, their reasoning capability in complex soundscapes still falls behind large vision-language models (LVLMs). Compared to the visual domain, one bottleneck is the lack of large-scale chain-of-thought audio data to teach LALM stepwise reasoning. To circumvent this data and modality gap, we present SightSound-R1, a cross-modal distillation framework that transfers advanced reasoning from a stronger LVLM teacher to a weaker LALM student on the same audio-visual question answering (AVQA) dataset. SightSound-R1 consists of three core steps: (i) test-time scaling to generate audio-focused chains of thought (CoT) from an LVLM teacher, (ii) audio-grounded validation to filter hallucinations, and (iii) a distillation pipeline with supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for the LALM student. Results show that SightSound-R1 improves LALM reasoning performance both in the in-domain AVQA test set as well as in unseen auditory scenes and questions, outperforming both pretrained and label-only distilled baselines. Thus, we conclude that vision reasoning can be effectively transferred to audio models and scaled with abundant audio-visual data.
>
---
#### [new 002] CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频反伪造任务，提出CompSpoof数据集和分离增强的联合学习框架，解决现有方法无法检测组件级伪造的问题。通过分离语音与环境声并分别检测，提升反伪造精度。**

- **链接: [http://arxiv.org/pdf/2509.15804v1](http://arxiv.org/pdf/2509.15804v1)**

> **作者:** Xueping Zhang; Liwei Jin; Yechen Wang; Linxi Li; Ming Li
>
> **摘要:** Component-level audio Spoofing (Comp-Spoof) targets a new form of audio manipulation where only specific components of a signal, such as speech or environmental sound, are forged or substituted while other components remain genuine. Existing anti-spoofing datasets and methods treat an utterance or a segment as entirely bona fide or entirely spoofed, and thus cannot accurately detect component-level spoofing. To address this, we construct a new dataset, CompSpoof, covering multiple combinations of bona fide and spoofed speech and environmental sound. We further propose a separation-enhanced joint learning framework that separates audio components apart and applies anti-spoofing models to each one. Joint learning is employed, preserving information relevant for detection. Extensive experiments demonstrate that our method outperforms the baseline, highlighting the necessity of separate components and the importance of detecting spoofing for each component separately. Datasets and code are available at: https://github.com/XuepingZhang/CompSpoof.
>
---
#### [new 003] The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍了2025年歌唱语音转换挑战赛，任务包括歌手身份和演唱风格转换。为评估系统性能，构建了新数据库、设计了两项任务、开放了基线模型，并进行了大规模主观和客观测试。结果显示，身份转换效果较好，但风格建模仍具挑战性。**

- **链接: [http://arxiv.org/pdf/2509.15629v1](http://arxiv.org/pdf/2509.15629v1)**

> **作者:** Lester Phillip Violeta; Xueyao Zhang; Jiatong Shi; Yusuke Yasuda; Wen-Chin Huang; Zhizheng Wu; Tomoki Toda
>
> **摘要:** We present the findings of the latest iteration of the Singing Voice Conversion Challenge, a scientific event aiming to compare and understand different voice conversion systems in a controlled environment. Compared to previous iterations which solely focused on converting the singer identity, this year we also focused on converting the singing style of the singer. To create a controlled environment and thorough evaluations, we developed a new challenge database, introduced two tasks, open-sourced baselines, and conducted large-scale crowd-sourced listening tests and objective evaluations. The challenge was ran for two months and in total we evaluated 26 different systems. The results of the large-scale crowd-sourced listening test showed that top systems had comparable singer identity scores to ground truth samples. However, modeling the singing style and consequently achieving high naturalness still remains a challenge in this task, primarily due to the difficulty in modeling dynamic information in breathy, glissando, and vibrato singing styles.
>
---
#### [new 004] Reverse Engineering of Music Mixing Graphs with Differentiable Processors and Iterative Pruning
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于音乐混音逆向工程任务，旨在解决如何从最终混音中还原干信号处理与组合方式的问题。作者提出使用可微处理器和迭代剪枝方法构建并优化混音图结构，通过批量处理和干湿参数加速搜索，实现高效、高质量的混音逆向分析。**

- **链接: [http://arxiv.org/pdf/2509.15948v1](http://arxiv.org/pdf/2509.15948v1)**

> **作者:** Sungho Lee; Marco Martínez-Ramírez; Wei-Hsiang Liao; Stefan Uhlich; Giorgio Fabbro; Kyogu Lee; Yuki Mitsufuji
>
> **备注:** JAES, extension of arxiv.org/abs/2408.03204 and arxiv.org/abs/2406.01049
>
> **摘要:** Reverse engineering of music mixes aims to uncover how dry source signals are processed and combined to produce a final mix. We extend the prior works to reflect the compositional nature of mixing and search for a graph of audio processors. First, we construct a mixing console, applying all available processors to every track and subgroup. With differentiable processor implementations, we optimize their parameters with gradient descent. Then, we repeat the process of removing negligible processors and fine-tuning the remaining ones. This way, the quality of the full mixing console can be preserved while removing approximately two-thirds of the processors. The proposed method can be used not only to analyze individual music mixes but also to collect large-scale graph data that can be used for downstream tasks, e.g., automatic mixing. Especially for the latter purpose, efficient implementation of the search is crucial. To this end, we present an efficient batch-processing method that computes multiple processors in parallel. We also exploit the "dry/wet" parameter of the processors to accelerate the search. Extensive quantitative and qualitative analyses are conducted to evaluate the proposed method's performance, behavior, and computational cost.
>
---
#### [new 005] Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出COSE，一种面向语音增强的一阶流匹配框架。针对传统扩散模型计算成本高、依赖多步生成的问题，通过引入速度组合恒等式，高效计算平均速度场，实现更快的采样与训练效率，同时保持语音质量。**

- **链接: [http://arxiv.org/pdf/2509.15952v1](http://arxiv.org/pdf/2509.15952v1)**

> **作者:** Gang Yang; Yue Lei; Wenxin Tai; Jin Wu; Jia Chen; Ting Zhong; Fan Zhou
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP 2026
>
> **摘要:** Diffusion and flow matching (FM) models have achieved remarkable progress in speech enhancement (SE), yet their dependence on multi-step generation is computationally expensive and vulnerable to discretization errors. Recent advances in one-step generative modeling, particularly MeanFlow, provide a promising alternative by reformulating dynamics through average velocity fields. In this work, we present COSE, a one-step FM framework tailored for SE. To address the high training overhead of Jacobian-vector product (JVP) computations in MeanFlow, we introduce a velocity composition identity to compute average velocity efficiently, eliminating expensive computation while preserving theoretical consistency and achieving competitive enhancement quality. Extensive experiments on standard benchmarks show that COSE delivers up to 5x faster sampling and reduces training cost by 40%, all without compromising speech quality. Code is available at https://github.com/ICDM-UESTC/COSE.
>
---
#### [new 006] A Novel Semantic Compression Approach for Ultra-low Bandwidth Voice Communication
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一种新型语义压缩方法，用于极低带宽语音通信。针对传统音频编码器在低比特率下质量下降的问题，利用生成式语音模型提取高层语义特征，在2-4倍更低比特率下实现语音任务性能不降甚至提升。**

- **链接: [http://arxiv.org/pdf/2509.15462v1](http://arxiv.org/pdf/2509.15462v1)**

> **作者:** Ryan Collette; Ross Greenwood; Serena Nicoll
>
> **备注:** 5 pages, 2 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** While existing speech audio codecs designed for compression exploit limited forms of temporal redundancy and allow for multi-scale representations, they tend to represent all features of audio in the same way. In contrast, generative voice models designed for text-to-speech and voice transfer tasks have recently proved effective at factorizing audio signals into high-level semantic representations of fundamentally distinct features. In this paper, we leverage such representations in a novel semantic communications approach to achieve lower bitrates without sacrificing perceptual quality or suitability for specific downstream tasks. Our technique matches or outperforms existing audio codecs on transcription, sentiment analysis, and speaker verification when encoding at 2-4x lower bitrate -- notably surpassing Encodec in perceptual quality and speaker verification while using up to 4x less bitrate.
>
---
#### [new 007] TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出TISDiSS，一种用于区分性声源分离的可扩展框架。针对分离性能提升依赖大模型导致成本高的问题，设计了训练与推理阶段均可扩展的结构，实现灵活的速度-性能权衡。实验表明其在减少参数量下达到最优性能。**

- **链接: [http://arxiv.org/pdf/2509.15666v1](http://arxiv.org/pdf/2509.15666v1)**

> **作者:** Yongsheng Feng; Yuetonghui Xu; Jiehui Luo; Hongjia Liu; Xiaobing Li; Feng Yu; Wei Li
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates early-split multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation.
>
---
#### [new 008] Differentiable Acoustic Radiance Transfer
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出DART，一种可微分的声辐射传输方法，用于优化材料属性并预测新声源-接收器设置的能量响应，解决稀疏测量下的声学建模问题。**

- **链接: [http://arxiv.org/pdf/2509.15946v1](http://arxiv.org/pdf/2509.15946v1)**

> **作者:** Sungho Lee; Matteo Scerbo; Seungu Han; Min Jun Choi; Kyogu Lee; Enzo De Sena
>
> **摘要:** Geometric acoustics is an efficient approach to room acoustics modeling, governed by the canonical time-dependent rendering equation. Acoustic radiance transfer (ART) solves the equation through discretization, modeling the time- and direction-dependent energy exchange between surface patches given with flexible material properties. We introduce DART, a differentiable and efficient implementation of ART that enables gradient-based optimization of material properties. We evaluate DART on a simpler variant of the acoustic field learning task, which aims to predict the energy responses of novel source-receiver settings. Experimental results show that DART exhibits favorable properties, e.g., better generalization under a sparse measurement scenario, compared to existing signal processing and neural network baselines, while remaining a simple, fully interpretable system.
>
---
#### [new 009] Direct Simultaneous Translation Activation for Large Audio-Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究实时语音到文本翻译（Simul-S2TT）任务，旨在无需修改模型结构的情况下激活大音视频语言模型的实时翻译能力。提出SimulSA方法，通过随机截断语音和构建部分对齐数据增强，有效弥合预训练与推理间的分布差异，仅需1%的同步数据即可显著提升性能。**

- **链接: [http://arxiv.org/pdf/2509.15692v1](http://arxiv.org/pdf/2509.15692v1)**

> **作者:** Pei Zhang; Yiming Wang; Jialong Tang; Baosong Yang; Rui Wang; Derek F. Wong; Fei Huang
>
> **摘要:** Simultaneous speech-to-text translation (Simul-S2TT) aims to translate speech into target text in real time, outputting translations while receiving source speech input, rather than waiting for the entire utterance to be spoken. Simul-S2TT research often modifies model architectures to implement read-write strategies. However, with the rise of large audio-language models (LALMs), a key challenge is how to directly activate Simul-S2TT capabilities in base models without additional architectural changes. In this paper, we introduce {\bf Simul}taneous {\bf S}elf-{\bf A}ugmentation ({\bf SimulSA}), a strategy that utilizes LALMs' inherent capabilities to obtain simultaneous data by randomly truncating speech and constructing partially aligned translation. By incorporating them into offline SFT data, SimulSA effectively bridges the distribution gap between offline translation during pretraining and simultaneous translation during inference. Experimental results demonstrate that augmenting only about {\bf 1\%} of the simultaneous data, compared to the full offline SFT data, can significantly activate LALMs' Simul-S2TT capabilities without modifications to model architecture or decoding strategy.
>
---
#### [new 010] EmoQ: Speech Emotion Recognition via Speech-Aware Q-Former and Large Language Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出EmoQ，一种基于MLLM的语音情感识别（SER）框架。针对多模态对齐困难及推理错误问题，设计了EmoQ-Former和软提示注入策略，并采用多目标情感学习，实现IEMOCAP和MELD数据集上的SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.15775v1](http://arxiv.org/pdf/2509.15775v1)**

> **作者:** Yiqing Yang; Man-Wai Mak
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** The performance of speech emotion recognition (SER) is limited by the insufficient emotion information in unimodal systems and the feature alignment difficulties in multimodal systems. Recently, multimodal large language models (MLLMs) have made progress in SER. However, MLLMs still suffer from hallucination and misclassification problems in complex emotion reasoning. To address these problems, we propose an MLLM-based framework called EmoQ, which generates query embeddings that fuse multimodal information through an EmoQ-Former and uses multi-objective affective learning (MAL) to achieve co-optimization. The framework also provides a soft-prompt injection strategy to inject multimodal representations into the LLM. This end-to-end architecture achieves state-of-the-art performance on the IEMOCAP and MELD datasets, providing a new multimodal fusion paradigm for SER.
>
---
#### [new 011] Thinking in cocktail party: Chain-of-Thought and reinforcement learning for target speaker automatic speech recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对鸡尾酒会场景下的目标说话人语音识别（TS-ASR）任务，提出结合思维链（CoT）和强化学习（RL）的新型框架。通过构建CoT数据集并分阶段训练模型，显著提升了TS-ASR性能，达到当前最优水平。**

- **链接: [http://arxiv.org/pdf/2509.15612v1](http://arxiv.org/pdf/2509.15612v1)**

> **作者:** Yiru Zhang; Hang Su; Lichun Fan; Zhenbo Luo; Jian Luan
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Target Speaker Automatic Speech Recognition (TS-ASR) aims to transcribe the speech of a specified target speaker from multi-speaker mixtures in cocktail party scenarios. Recent advancement of Large Audio-Language Models (LALMs) has already brought some new insights to TS-ASR. However, significant room for optimization remains for the TS-ASR task within the LALMs architecture. While Chain of Thoughts (CoT) and Reinforcement Learning (RL) have proven effective in certain speech tasks, TS-ASR, which requires the model to deeply comprehend speech signals, differentiate various speakers, and handle overlapping utterances is particularly well-suited to a reasoning-guided approach. Therefore, we propose a novel framework that incorporates CoT and RL training into TS-ASR for performance improvement. A novel CoT dataset of TS-ASR is constructed, and the TS-ASR model is first trained on regular data and then fine-tuned on CoT data. Finally, the model is further trained with RL using selected data to enhance generalized reasoning capabilities. Experiment results demonstrate a significant improvement of TS-ASR performance with CoT and RL training, establishing a state-of-the-art performance compared with previous works of TS-ASR on comparable datasets.
>
---
#### [new 012] Mamba-2 audio captioning: design space exploration and analysis
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究音频描述生成任务，基于Mamba-2模型探索不同设计参数（如LLM规模、LoRA秩、连接器设计）对性能的影响，并分析参数数量、音频编码策略等因素的作用。**

- **链接: [http://arxiv.org/pdf/2509.15680v1](http://arxiv.org/pdf/2509.15680v1)**

> **作者:** Taehan Lee; Jaehan Jung; Hyukjun Lee
>
> **备注:** Submitted to the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026). Under review
>
> **摘要:** We present an audio captioning model built on the Mamba-2 large language model backbone, which is a state-of-the-art (SOTA) state-space model (SSM). We systematically explore the design space: LLM sizes, LoRA ranks, and connector designs leveraging Mamba-2's linear-time complexity with respect to sequence length. Across benchmarks, our models achieve strong captioning performance compared with larger language models trained on the same dataset, despite using fewer parameters. For the first time, we conduct an in-depth analysis of how the number of LLM parameters, audio encoder fine-tuning strategies, audio feature diversity, and different feature reduction or expansion techniques affect performance.
>
---
#### [new 013] EMO-RL: Emotion-Rule-Based Reinforcement Learning Enhanced Audio-Language Model for Generalized Speech Emotion Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出EMO-RL，一种基于情绪规则的强化学习框架，用于提升语音情感识别（SER）任务中大音频-语言模型的情感推理能力。针对情感边界模糊和小模型推理能力弱的问题，引入ESWR和ESR机制，实验表明其在MELD和IEMOCAP数据集上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.15654v1](http://arxiv.org/pdf/2509.15654v1)**

> **作者:** Pengcheng Li; Botao Zhao; Zuheng Kang; Junqing Peng; Xiaoyang Qu; Yayun He; Jianzong Wang
>
> **备注:** Accpeted by the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Although Large Audio-Language Models (LALMs) have exhibited outstanding performance in auditory understanding, their performance in affective computing scenarios, particularly in emotion recognition, reasoning, and subtle sentiment differentiation, remains suboptimal. Recent advances in Reinforcement Learning (RL) have shown promise in improving LALMs' reasoning abilities. However, two critical challenges hinder the direct application of RL techniques to Speech Emotion Recognition (SER) tasks: (1) convergence instability caused by ambiguous emotional boundaries and (2) limited reasoning ability when using relatively small models (e.g., 7B-parameter architectures). To overcome these limitations, we introduce EMO-RL, a novel framework incorporating reinforcement learning with two key innovations: Emotion Similarity-Weighted Reward (ESWR) and Explicit Structured Reasoning (ESR). Built upon pretrained LALMs, our method employs group-relative policy optimization with emotion constraints. Comprehensive experiments demonstrate that our EMO-RL training strategies can significantly enhance the emotional reasoning capabilities of LALMs, attaining state-of-the-art results on both the MELD and IEMOCAP datasets, and cross-dataset experiments prove the strong superiority of generalization.
>
---
#### [new 014] The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出TRIA，一种基于掩码语言模型的音频生成方法，用于将节奏性声音提示转换为高保真鼓录音。任务是鼓音轨生成，旨在解决从简单节奏提示高效生成完整鼓录音的问题。模型使用少于10小时数据训练，实现了零样本跨音色生成。**

- **链接: [http://arxiv.org/pdf/2509.15625v1](http://arxiv.org/pdf/2509.15625v1)**

> **作者:** Patrick O'Reilly; Julia Barnett; Hugo Flores García; Annie Chu; Nathan Pruyne; Prem Seetharaman; Bryan Pardo
>
> **备注:** ISMIR 2025
>
> **摘要:** Musicians and nonmusicians alike use rhythmic sound gestures, such as tapping and beatboxing, to express drum patterns. While these gestures effectively communicate musical ideas, realizing these ideas as fully-produced drum recordings can be time-consuming, potentially disrupting many creative workflows. To bridge this gap, we present TRIA (The Rhythm In Anything), a masked transformer model for mapping rhythmic sound gestures to high-fidelity drum recordings. Given an audio prompt of the desired rhythmic pattern and a second prompt to represent drumkit timbre, TRIA produces audio of a drumkit playing the desired rhythm (with appropriate elaborations) in the desired timbre. Subjective and objective evaluations show that a TRIA model trained on less than 10 hours of publicly-available drum data can generate high-quality, faithful realizations of sound gestures across a wide range of timbres in a zero-shot manner.
>
---
#### [new 015] Blind Source Separation of Radar Signals in Time Domain Using Deep Learning
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文属于雷达信号盲源分离任务，旨在解决同方向、同频段多信号难以分离的问题。作者借鉴音频分离方法，利用深度学习模型在时域中实现单通道接收下的未知信号分离。**

- **链接: [http://arxiv.org/pdf/2509.15603v1](http://arxiv.org/pdf/2509.15603v1)**

> **作者:** Sven Hinderer
>
> **摘要:** Identification and further analysis of radar emitters in a contested environment requires detection and separation of incoming signals. If they arrive from the same direction and at similar frequencies, deinterleaving them remains challenging. A solution to overcome this limitation becomes increasingly important with the advancement of emitter capabilities. We propose treating the problem as blind source separation in time domain and apply supervisedly trained neural networks to extract the underlying signals from the received mixture. This allows us to handle highly overlapping and also continuous wave (CW) signals from both radar and communication emitters. We make use of advancements in the field of audio source separation and extend a current state-of-the-art model with the objective of deinterleaving arbitrary radio frequency (RF) signals. Results show, that our approach is capable of separating two unknown waveforms in a given frequency band with a single channel receiver.
>
---
#### [new 016] FocalCodec-Stream: Streaming Low-Bitrate Speech Coding via Causal Distillation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出FocalCodec-Stream，一种流式低码率语音编码器，旨在解决实时应用中非流式编解码器的限制。通过因果蒸馏和轻量架构优化，在0.55-0.80 kbps下实现高质量语音压缩与语义保留。**

- **链接: [http://arxiv.org/pdf/2509.16195v1](http://arxiv.org/pdf/2509.16195v1)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Neural audio codecs are a fundamental component of modern generative audio pipelines. Although recent codecs achieve strong low-bitrate reconstruction and provide powerful representations for downstream tasks, most are non-streamable, limiting their use in real-time applications. We present FocalCodec-Stream, a hybrid codec based on focal modulation that compresses speech into a single binary codebook at 0.55 - 0.80 kbps with a theoretical latency of 80 ms. Our approach combines multi-stage causal distillation of WavLM with targeted architectural improvements, including a lightweight refiner module that enhances quality under latency constraints. Experiments show that FocalCodec-Stream outperforms existing streamable codecs at comparable bitrates, while preserving both semantic and acoustic information. The result is a favorable trade-off between reconstruction quality, downstream task performance, latency, and efficiency. Code and checkpoints will be released at https://github.com/lucadellalib/focalcodec.
>
---
#### [new 017] Beyond Video-to-SFX: Video to Audio Synthesis with Environmentally Aware Speech
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出BVS方法，用于视频到音频合成任务，旨在解决现有方法难以生成与视频同步且可理解的环境语音的问题。通过两阶段模型（V2AS和VS2A），实现视频引导的语义与声学特征生成，提升语音与背景音效的沉浸感与同步性。**

- **链接: [http://arxiv.org/pdf/2509.15492v1](http://arxiv.org/pdf/2509.15492v1)**

> **作者:** Xinlei Niu; Jianbo Ma; Dylan Harper-Harris; Xiangyu Zhang; Charles Patrick Martin; Jing Zhang
>
> **摘要:** The generation of realistic, context-aware audio is important in real-world applications such as video game development. While existing video-to-audio (V2A) methods mainly focus on Foley sound generation, they struggle to produce intelligible speech. Meanwhile, current environmental speech synthesis approaches remain text-driven and fail to temporally align with dynamic video content. In this paper, we propose Beyond Video-to-SFX (BVS), a method to generate synchronized audio with environmentally aware intelligible speech for given videos. We introduce a two-stage modeling method: (1) stage one is a video-guided audio semantic (V2AS) model to predict unified audio semantic tokens conditioned on phonetic cues; (2) stage two is a video-conditioned semantic-to-acoustic (VS2A) model that refines semantic tokens into detailed acoustic tokens. Experiments demonstrate the effectiveness of BVS in scenarios such as video-to-context-aware speech synthesis and immersive audio background conversion, with ablation studies further validating our design. Our demonstration is available at~\href{https://xinleiniu.github.io/BVS-demo/}{BVS-Demo}.
>
---
#### [new 018] Contrastive Learning with Spectrum Information Augmentation in Abnormal Sound Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对异常声音检测任务，旨在解决无监督学习中模型难以学习正常数据分布的问题。提出了一种基于对比学习的高频频谱信息增强方法，使模型更关注低频的正常声音特征，并在DCASE 2020和2022数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.15570v1](http://arxiv.org/pdf/2509.15570v1)**

> **作者:** Xinxin Meng; Jiangtao Guo; Yunxiang Zhang; Shun Huang
>
> **备注:** Accepted CVIPPR 2024 April Xiamen China
>
> **摘要:** The outlier exposure method is an effective approach to address the unsupervised anomaly sound detection problem. The key focus of this method is how to make the model learn the distribution space of normal data. Based on biological perception and data analysis, it is found that anomalous audio and noise often have higher frequencies. Therefore, we propose a data augmentation method for high-frequency information in contrastive learning. This enables the model to pay more attention to the low-frequency information of the audio, which represents the normal operational mode of the machine. We evaluated the proposed method on the DCASE 2020 Task 2. The results showed that our method outperformed other contrastive learning methods used on this dataset. We also evaluated the generalizability of our method on the DCASE 2022 Task 2 dataset.
>
---
#### [new 019] DISPATCH: Distilling Selective Patches for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对语音增强任务中知识蒸馏效果有限的问题，提出DISPatch框架，通过选择性地对教师模型表现优于学生的频谱区域进行蒸馏，并引入MSSP方法处理频谱异质性，有效提升了轻量模型的性能。**

- **链接: [http://arxiv.org/pdf/2509.15922v1](http://arxiv.org/pdf/2509.15922v1)**

> **作者:** Dohwan Kim; Jung-Woo Choi
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** In speech enhancement, knowledge distillation (KD) compresses models by transferring a high-capacity teacher's knowledge to a compact student. However, conventional KD methods train the student to mimic the teacher's output entirely, which forces the student to imitate the regions where the teacher performs poorly and to apply distillation to the regions where the student already performs well, which yields only marginal gains. We propose Distilling Selective Patches (DISPatch), a KD framework for speech enhancement that applies the distillation loss to spectrogram patches where the teacher outperforms the student, as determined by a Knowledge Gap Score. This approach guides optimization toward areas with the most significant potential for student improvement while minimizing the influence of regions where the teacher may provide unreliable instruction. Furthermore, we introduce Multi-Scale Selective Patches (MSSP), a frequency-dependent method that uses different patch sizes across low- and high-frequency bands to account for spectral heterogeneity. We incorporate DISPatch into conventional KD methods and observe consistent gains in compact students. Moreover, integrating DISPatch and MSSP into a state-of-the-art frequency-dependent KD method considerably improves performance across all metrics.
>
---
#### [new 020] LibriTTS-VI: A Public Corpus and Novel Methods for Efficient Voice Impression Control
- **分类: cs.SD; eess.AS**

- **简介: 该论文聚焦文本到语音中的**语音印象控制任务**，旨在解决**印象泄露**和**缺乏公开标注数据集**的问题。提出了两种新方法以减少泄露，并构建了首个公开的语音印象数据集 LibriTTS-VI，提升了语音印象的可控性与生成质量。**

- **链接: [http://arxiv.org/pdf/2509.15626v1](http://arxiv.org/pdf/2509.15626v1)**

> **作者:** Junki Ohmura; Yuki Ito; Emiru Tsunoo; Toshiyuki Sekiya; Toshiyuki Kumakura
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Fine-grained control over voice impressions (e.g., making a voice brighter or calmer) is a key frontier for creating more controllable text-to-speech. However, this nascent field faces two key challenges. The first is the problem of impression leakage, where the synthesized voice is undesirably influenced by the speaker's reference audio, rather than the separately specified target impression, and the second is the lack of a public, annotated corpus. To mitigate impression leakage, we propose two methods: 1) a training strategy that separately uses an utterance for speaker identity and another utterance of the same speaker for target impression, and 2) a novel reference-free model that generates a speaker embedding solely from the target impression, achieving the benefits of improved robustness against the leakage and the convenience of reference-free generation. Objective and subjective evaluations demonstrate a significant improvement in controllability. Our best method reduced the mean squared error of 11-dimensional voice impression vectors from 0.61 to 0.41 objectively and from 1.15 to 0.92 subjectively, while maintaining high fidelity. To foster reproducible research, we introduce LibriTTS-VI, the first public voice impression dataset released with clear annotation standards, built upon the LibriTTS-R corpus.
>
---
#### [new 021] Fed-PISA: Federated Voice Cloning via Personalized Identity-Style Adaptation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对联邦学习在语音克隆中的高通信成本和个性化不足问题，提出Fed-PISA方法。通过引入低秩适配机制（LoRA）和协作过滤式聚合策略，在保证隐私的同时提升语音风格表达与个性化效果。**

- **链接: [http://arxiv.org/pdf/2509.16010v1](http://arxiv.org/pdf/2509.16010v1)**

> **作者:** Qi Wang; Shituo Ma; Guoxin Yu; Hanyang Peng; Yue Yu
>
> **摘要:** Voice cloning for Text-to-Speech (TTS) aims to generate expressive and personalized speech from text using limited data from a target speaker. Federated Learning (FL) offers a collaborative and privacy-preserving framework for this task, but existing approaches suffer from high communication costs and tend to suppress stylistic heterogeneity, resulting in insufficient personalization. To address these issues, we propose Fed-PISA, which stands for Federated Personalized Identity-Style Adaptation. To minimize communication costs, Fed-PISA introduces a disentangled Low-Rank Adaptation (LoRA) mechanism: the speaker's timbre is retained locally through a private ID-LoRA, while only a lightweight style-LoRA is transmitted to the server, thereby minimizing parameter exchange. To harness heterogeneity, our aggregation method, inspired by collaborative filtering, is introduced to create custom models for each client by learning from stylistically similar peers. Experiments show that Fed-PISA improves style expressivity, naturalness, and speaker similarity, outperforming standard federated baselines with minimal communication costs.
>
---
#### [new 022] Impact of Phonetics on Speaker Identity in Adversarial Voice Attack
- **分类: cs.SD; cs.AI; cs.CR; eess.AS; I.2.0; I.2.7; I.5.4; K.6.5**

- **简介: 该论文研究对抗性语音攻击中语音特征对说话人身份的影响，属于语音安全任务。论文分析对抗扰动在音素层面的混淆机制，揭示其导致语音识别和说话人验证错误的原因，并通过实验验证音素感知防御的必要性。**

- **链接: [http://arxiv.org/pdf/2509.15437v1](http://arxiv.org/pdf/2509.15437v1)**

> **作者:** Daniyal Kabir Dar; Qiben Yan; Li Xiao; Arun Ross
>
> **备注:** Additional figures for extended visualization: https://daniyalkabir.github.io/icassp-2025-results/
>
> **摘要:** Adversarial perturbations in speech pose a serious threat to automatic speech recognition (ASR) and speaker verification by introducing subtle waveform modifications that remain imperceptible to humans but can significantly alter system outputs. While targeted attacks on end-to-end ASR models have been widely studied, the phonetic basis of these perturbations and their effect on speaker identity remain underexplored. In this work, we analyze adversarial audio at the phonetic level and show that perturbations exploit systematic confusions such as vowel centralization and consonant substitutions. These distortions not only mislead transcription but also degrade phonetic cues critical for speaker verification, leading to identity drift. Using DeepSpeech as our ASR target, we generate targeted adversarial examples and evaluate their impact on speaker embeddings across genuine and impostor samples. Results across 16 phonetically diverse target phrases demonstrate that adversarial audio induces both transcription errors and identity drift, highlighting the need for phonetic-aware defenses to ensure the robustness of ASR and speaker recognition systems.
>
---
#### [new 023] SONAR: Self-Distilled Continual Pre-training for Domain Adaptive Audio Representation
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SONAR，一种用于音频表征的持续预训练框架，旨在解决在新领域适应时避免灾难性遗忘的问题。通过联合采样、正则化和动态扩展码本，实现高效且稳健的跨领域音频表征学习。**

- **链接: [http://arxiv.org/pdf/2509.15703v1](http://arxiv.org/pdf/2509.15703v1)**

> **作者:** Yizhou Zhang; Yuan Gao; Wangjin Zhou; Zicheng Yuan; Keisuke Imoto; Tatsuya Kawahara
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Self-supervised learning (SSL) on large-scale datasets like AudioSet has become the dominant paradigm for audio representation learning. While the continuous influx of new, unlabeled audio presents an opportunity to enrich these static representations, a naive approach is to retrain the model from scratch using all available data. However, this method is computationally prohibitive and discards the valuable knowledge embedded in the previously trained model weights. To address this inefficiency, we propose SONAR (Self-distilled cONtinual pre-training for domain adaptive Audio Representation), a continual pre-training framework built upon BEATs. SONAR effectively adapts to new domains while mitigating catastrophic forgetting by tackling three key challenges: implementing a joint sampling strategy for new and prior data, applying regularization to balance specificity and generality, and dynamically expanding the tokenizer codebook for novel acoustic patterns. Experiments across four distinct domains demonstrate that our method achieves both high adaptability and robust resistance to forgetting.
>
---
#### [new 024] From Independence to Interaction: Speaker-Aware Simulation of Multi-Speaker Conversational Timing
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究多说话人对话时序模拟任务，旨在解决传统方法忽略说话人特性和交互动态的问题。提出一种说话人感知的方法，通过个性化时序建模、马尔可夫链控制对话轮次及统一间隙分布，提升对话模拟的时序一致性和真实性。**

- **链接: [http://arxiv.org/pdf/2509.15808v1](http://arxiv.org/pdf/2509.15808v1)**

> **作者:** Máté Gedeon; Péter Mihajlik
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We present a speaker-aware approach for simulating multi-speaker conversations that captures temporal consistency and realistic turn-taking dynamics. Prior work typically models aggregate conversational statistics under an independence assumption across speakers and turns. In contrast, our method uses speaker-specific deviation distributions enforcing intra-speaker temporal consistency, while a Markov chain governs turn-taking and a fixed room impulse response preserves spatial realism. We also unify pauses and overlaps into a single gap distribution, modeled with kernel density estimation for smooth continuity. Evaluation on Switchboard using intrinsic metrics - global gap statistics, correlations between consecutive gaps, copula-based higher-order dependencies, turn-taking entropy, and gap survival functions - shows that speaker-aware simulation better aligns with real conversational patterns than the baseline method, capturing fine-grained temporal dependencies and realistic speaker alternation, while revealing open challenges in modeling long-range conversational structure.
>
---
#### [new 025] Exploring Fine-Tuning of Large Audio Language Models for Spoken Language Understanding under Limited Speech data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究了在有限语音数据下，对大型音语模型（LALMs）进行微调以提升语音语言理解（SLU）效果的问题。工作包括对比文本微调、混合训练和课程学习等方法，验证了少量语音数据与文本结合可显著提升性能，并探索了跨语言适应的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15389v1](http://arxiv.org/pdf/2509.15389v1)**

> **作者:** Youngwon Choi; Jaeyoon Jung; Hyeonyu Kim; Huu-Kim Nguyen; Hwayeon Kim
>
> **备注:** 4 pages (excluding references), 2 figures, submitted to ICASSP 2026
>
> **摘要:** Large Audio Language Models (LALMs) have emerged as powerful tools for speech-related tasks but remain underexplored for fine-tuning, especially with limited speech data. To bridge this gap, we systematically examine how different fine-tuning schemes including text-only, direct mixing, and curriculum learning affect spoken language understanding (SLU), focusing on scenarios where text-label pairs are abundant while paired speech-label data are limited. Results show that LALMs already achieve competitive performance with text-only fine-tuning, highlighting their strong generalization ability. Adding even small amounts of speech data (2-5%) yields substantial further gains, with curriculum learning particularly effective under scarce data. In cross-lingual SLU, combining source-language speech data with target-language text and minimal target-language speech data enables effective adaptation. Overall, this study provides practical insights into the LALM fine-tuning under realistic data constraints.
>
---
#### [new 026] Emotion-Aware Speech Generation with Character-Specific Voices for Comics
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出一个端到端系统，用于从漫画生成角色专属且情感感知的语音。通过图像处理和语言模型分析对话与情感，并用定制语音合成模型生成语音，旨在实现漫画自动配音，提升阅读沉浸感。**

- **链接: [http://arxiv.org/pdf/2509.15253v1](http://arxiv.org/pdf/2509.15253v1)**

> **作者:** Zhiwen Qian; Jinhua Liang; Huan Zhang
>
> **摘要:** This paper presents an end-to-end pipeline for generating character-specific, emotion-aware speech from comics. The proposed system takes full comic volumes as input and produces speech aligned with each character's dialogue and emotional state. An image processing module performs character detection, text recognition, and emotion intensity recognition. A large language model performs dialogue attribution and emotion analysis by integrating visual information with the evolving plot context. Speech is synthesized through a text-to-speech model with distinct voice profiles tailored to each character and emotion. This work enables automated voiceover generation for comics, offering a step toward interactive and immersive comic reading experience.
>
---
#### [new 027] De-crackling Virtual Analog Controls with Asymptotically Stable Recurrent Neural Networks
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究虚拟模拟建模中的递归神经网络，旨在解决控制信号变化导致的音频伪影问题。提出渐近稳定RNN结构，消除输入为零和时变条件下的音频瑕疵，并为其他音频模型提供参考方案。**

- **链接: [http://arxiv.org/pdf/2509.15622v1](http://arxiv.org/pdf/2509.15622v1)**

> **作者:** Valtteri Kallinen; Lauri Juvela
>
> **摘要:** Recurrent neural networks are used in virtual analog modeling applications to digitally replicate the sound of analog hardware audio processors. The controls of hardware devices can be used as a conditioning input to these networks. A common method for introducing control conditioning to these models is the direct static concatenation of controls with input audio samples, which we show produces audio artifacts under time-varied conditioning. Here we derive constraints for asymptotically stable variants of commonly used recurrent neural networks and demonstrate that asymptotical stability in recurrent neural networks can eliminate audio artifacts from the model output under zero input and time-varied conditioning. Furthermore, our results suggest a possible general solution to mitigate conditioning-induced artifacts in other audio neural network architectures, such as convolutional and state-space models.
>
---
#### [new 028] VOX-KRIKRI: Unifying Speech and Language through Continuous Fusion
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出VOX-KRIKRI，一种统一语音与语言的连续融合框架。通过跨模态注意力机制，将Whisper的解码器状态与LLM融合，构建语音驱动的LLM。解决了多模态对齐问题，实现了希腊语ASR的SOTA性能，提升约20%。**

- **链接: [http://arxiv.org/pdf/2509.15667v1](http://arxiv.org/pdf/2509.15667v1)**

> **作者:** Dimitrios Damianos; Leon Voukoutis; Georgios Paraskevopoulos; Vassilis Katsouros
>
> **摘要:** We present a multimodal fusion framework that bridges pre-trained decoder-based large language models (LLM) and acoustic encoder-decoder architectures such as Whisper, with the aim of building speech-enabled LLMs. Instead of directly using audio embeddings, we explore an intermediate audio-conditioned text space as a more effective mechanism for alignment. Our method operates fully in continuous text representation spaces, fusing Whisper's hidden decoder states with those of an LLM through cross-modal attention, and supports both offline and streaming modes. We introduce \textit{VoxKrikri}, the first Greek speech LLM, and show through analysis that our approach effectively aligns representations across modalities. These results highlight continuous space fusion as a promising path for multilingual and low-resource speech LLMs, while achieving state-of-the-art results for Automatic Speech Recognition in Greek, providing an average $\sim20\%$ relative improvement across benchmarks.
>
---
#### [new 029] Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出Chunk SSL方法，针对语音预训练中的低延迟需求，解决流式与离线场景的统一建模问题。通过分块自监督学习、高分辨率标量量化和掩码预测损失，提升语音到文本任务的性能。**

- **链接: [http://arxiv.org/pdf/2509.15579v1](http://arxiv.org/pdf/2509.15579v1)**

> **作者:** Yun Tang; Cindy Tseng
>
> **摘要:** Low latency speech human-machine communication is becoming increasingly necessary as speech technology advances quickly in the last decade. One of the primary factors behind the advancement of speech technology is self-supervised learning. Most self-supervised learning algorithms are designed with full utterance assumption and compromises have to made if partial utterances are presented, which are common in the streaming applications. In this work, we propose a chunk based self-supervised learning (Chunk SSL) algorithm as an unified solution for both streaming and offline speech pre-training. Chunk SSL is optimized with the masked prediction loss and an acoustic encoder is encouraged to restore indices of those masked speech frames with help from unmasked frames in the same chunk and preceding chunks. A copy and append data augmentation approach is proposed to conduct efficient chunk based pre-training. Chunk SSL utilizes a finite scalar quantization (FSQ) module to discretize input speech features and our study shows a high resolution FSQ codebook, i.e., a codebook with vocabulary size up to a few millions, is beneficial to transfer knowledge from the pre-training task to the downstream tasks. A group masked prediction loss is employed during pre-training to alleviate the high memory and computation cost introduced by the large codebook. The proposed approach is examined in two speech to text tasks, i.e., speech recognition and speech translation. Experimental results on the \textsc{Librispeech} and \textsc{Must-C} datasets show that the proposed method could achieve very competitive results for speech to text tasks at both streaming and offline modes.
>
---
#### [new 030] Bench-RNR: Dataset for Benchmarking Repetitive and Non-repetitive Scanning LiDAR for Infrastructure-based Vehicle Localization
- **分类: cs.RO; eess.SP**

- **简介: 该论文针对基础设施车辆定位任务，研究重复与非重复扫描LiDAR的性能差异。为解决非重复扫描LiDAR在该领域应用不足的问题，作者构建了一个包含5445帧点云数据的公开数据集Bench-RNR，并建立基准实验进行对比分析。**

- **链接: [http://arxiv.org/pdf/2509.15583v1](http://arxiv.org/pdf/2509.15583v1)**

> **作者:** Runxin Zhao; Chunxiang Wang; Hanyang Zhuang; Ming Yang
>
> **摘要:** Vehicle localization using roadside LiDARs can provide centimeter-level accuracy for cloud-controlled vehicles while simultaneously serving multiple vehicles, enhanc-ing safety and efficiency. While most existing studies rely on repetitive scanning LiDARs, non-repetitive scanning LiDAR offers advantages such as eliminating blind zones and being more cost-effective. However, its application in roadside perception and localization remains limited. To address this, we present a dataset for infrastructure-based vehicle localization, with data collected from both repetitive and non-repetitive scanning LiDARs, in order to benchmark the performance of different LiDAR scanning patterns. The dataset contains 5,445 frames of point clouds across eight vehicle trajectory sequences, with diverse trajectory types. Our experiments establish base-lines for infrastructure-based vehicle localization and compare the performance of these methods using both non-repetitive and repetitive scanning LiDARs. This work offers valuable insights for selecting the most suitable LiDAR scanning pattern for infrastruc-ture-based vehicle localization. Our dataset is a signifi-cant contribution to the scientific community, supporting advancements in infrastructure-based perception and vehicle localization. The dataset and source code are publicly available at: https://github.com/sjtu-cyberc3/BenchRNR.
>
---
#### [new 031] Jamendo-QA: A Large-Scale Music Question Answering Dataset
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出了Jamendo-QA，一个大规模音乐问答数据集，旨在解决音乐理解领域缺乏专用QA数据的问题。利用Jamendo平台的音乐和Qwen-Omni模型自动标注，构建了对齐音频的问答对与描述，支持监督训练与零样本评估，推动音乐理解和多模态研究。**

- **链接: [http://arxiv.org/pdf/2509.15662v1](http://arxiv.org/pdf/2509.15662v1)**

> **作者:** Junyoung Koh; Soo Yong Kim; Yongwon Choi; Gyu Hyeong Choi
>
> **备注:** 4 pages, 8 figures. Submitted to ICASSP 2026
>
> **摘要:** We introduce Jamendo-QA, a large-scale dataset for Music Question Answering (Music-QA). The dataset is built on freely licensed tracks from the Jamendo platform and is automatically annotated using the Qwen-Omni model. Jamendo-QA provides question-answer pairs and captions aligned with music audio, enabling both supervised training and zero-shot evaluation. Our resource aims to fill the gap of music-specific QA datasets and foster further research in music understanding, retrieval, and generative applications. In addition to its scale, Jamendo-QA covers a diverse range of genres, instruments, and metadata attributes, allowing robust model benchmarking across varied musical contexts. We also provide detailed dataset statistics and highlight potential biases such as genre and gender imbalance to guide fair evaluation. We position Jamendo-QA as a scalable and publicly available benchmark that can facilitate future research in music understanding, multimodal modeling, and fair evaluation of music-oriented QA systems.
>
---
#### [new 032] Breathing and Semantic Pause Detection and Exertion-Level Classification in Post-Exercise Speech
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音与生理信号分析任务，旨在检测运动后语音中的语义停顿、呼吸停顿及联合停顿，并分类运动强度。基于同步音频与呼吸数据集，系统标注并对比多种模型与特征方法，在检测与分类任务上取得优于前人的结果。**

- **链接: [http://arxiv.org/pdf/2509.15473v1](http://arxiv.org/pdf/2509.15473v1)**

> **作者:** Yuyu Wang; Wuyue Xia; Huaxiu Yao; Jingping Nie
>
> **备注:** 6 pages, 3rd ACM International Workshop on Intelligent Acoustic Systems and Applications (IASA 25)
>
> **摘要:** Post-exercise speech contains rich physiological and linguistic cues, often marked by semantic pauses, breathing pauses, and combined breathing-semantic pauses. Detecting these events enables assessment of recovery rate, lung function, and exertion-related abnormalities. However, existing works on identifying and distinguishing different types of pauses in this context are limited. In this work, building on a recently released dataset with synchronized audio and respiration signals, we provide systematic annotations of pause types. Using these annotations, we systematically conduct exploratory breathing and semantic pause detection and exertion-level classification across deep learning models (GRU, 1D CNN-LSTM, AlexNet, VGG16), acoustic features (MFCC, MFB), and layer-stratified Wav2Vec2 representations. We evaluate three setups-single feature, feature fusion, and a two-stage detection-classification cascade-under both classification and regression formulations. Results show per-type detection accuracy up to 89$\%$ for semantic, 55$\%$ for breathing, 86$\%$ for combined pauses, and 73$\%$overall, while exertion-level classification achieves 90.5$\%$ accuracy, outperformin prior work.
>
---
#### [new 033] Indoor Positioning Based on Active Radar Sensing and Passive Reflectors: Reflector Placement Optimization
- **分类: cs.RO; eess.SP**

- **简介: 该论文研究基于雷达感知的室内定位系统，针对低成本高精度定位问题，提出结合被动反射器与FMCW雷达，并采用多目标粒子群算法优化反射器布局。**

- **链接: [http://arxiv.org/pdf/2509.15613v1](http://arxiv.org/pdf/2509.15613v1)**

> **作者:** Sven Hinderer; Pascal Schlachter; Zhibin Yu; Xiaofeng Wu; Bin Yang
>
> **摘要:** We extend our work on a novel indoor positioning system (IPS) for autonomous mobile robots (AMRs) based on radar sensing of local, passive radar reflectors. Through the combination of simple reflectors and a single-channel frequency modulated continuous wave (FMCW) radar, high positioning accuracy at low system cost can be achieved. Further, a multi-objective (MO) particle swarm optimization (PSO) algorithm is presented that optimizes the 2D placement of radar reflectors in complex room settings.
>
---
#### [new 034] Speech Language Models for Under-Represented Languages: Insights from Wolof
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦于欠代表语言Wolof的语音语言模型构建。针对数据稀缺问题，研究团队收集了大规模高质量语音数据，并基于此继续预训练HuBERT模型，提升了语音识别（ASR）效果。进一步整合为首个Wolof语音大模型，拓展至语音翻译等任务，并探索了多步推理能力。**

- **链接: [http://arxiv.org/pdf/2509.15362v1](http://arxiv.org/pdf/2509.15362v1)**

> **作者:** Yaya Sy; Dioula Doucouré; Christophe Cerisara; Irina Illina
>
> **摘要:** We present our journey in training a speech language model for Wolof, an underrepresented language spoken in West Africa, and share key insights. We first emphasize the importance of collecting large-scale, spontaneous, high-quality speech data, and show that continued pretraining HuBERT on this dataset outperforms both the base model and African-centric models on ASR. We then integrate this speech encoder into a Wolof LLM to train the first Speech LLM for this language, extending its capabilities to tasks such as speech translation. Furthermore, we explore training the Speech LLM to perform multi-step Chain-of-Thought before transcribing or translating. Our results show that the Speech LLM not only improves speech recognition but also performs well in speech translation. The models and the code will be openly shared.
>
---
#### [new 035] Pre-training Autoencoder for Acoustic Event Classification via Blinky
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究声学事件分类任务，针对Blinky系统中光传输带宽低、抗噪差的问题，提出基于预训练自编码器的声光转换方法。通过在编码器中注入噪声增强鲁棒性，并优化模型以适配边缘设备，实验表明其性能优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.15261v1](http://arxiv.org/pdf/2509.15261v1)**

> **作者:** Xiaoyang Liu; Yuma Kinoshita
>
> **备注:** Accepted to APSIPA ASC 2025. 6 pages, 1 figures
>
> **摘要:** In the acoustic event classification (AEC) framework that employs Blinkies, audio signals are converted into LED light emissions and subsequently captured by a single video camera. However, the 30 fps optical transmission channel conveys only about 0.2% of the normal audio bandwidth and is highly susceptible to noise. We propose a novel sound-to-light conversion method that leverages the encoder of a pre-trained autoencoder (AE) to distill compact, discriminative features from the recorded audio. To pre-train the AE, we adopt a noise-robust learning strategy in which artificial noise is injected into the encoder's latent representations during training, thereby enhancing the model's robustness against channel noise. The encoder architecture is specifically designed for the memory footprint of contemporary edge devices such as the Raspberry Pi 4. In a simulation experiment on the ESC-50 dataset under a stringent 15 Hz bandwidth constraint, the proposed method achieved higher macro-F1 scores than conventional sound-to-light conversion approaches.
>
---
#### [new 036] AFT: An Exemplar-Free Class Incremental Learning Method for Environmental Sound Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对环境声音分类任务中的增量学习问题，提出了一种无需存储旧数据的AFT方法。通过特征变换对齐新旧类别的时序特征，缓解模型在学习新类别时对旧类别的遗忘问题，在两个数据集上均取得性能提升。**

- **链接: [http://arxiv.org/pdf/2509.15523v1](http://arxiv.org/pdf/2509.15523v1)**

> **作者:** Xinyi Chen; Xi Chen; Zhenyu Weng; Yang Xiao
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** As sounds carry rich information, environmental sound classification (ESC) is crucial for numerous applications such as rare wild animals detection. However, our world constantly changes, asking ESC models to adapt to new sounds periodically. The major challenge here is catastrophic forgetting, where models lose the ability to recognize old sounds when learning new ones. Many methods address this using replay-based continual learning. This could be impractical in scenarios such as data privacy concerns. Exemplar-free methods are commonly used but can distort old features, leading to worse performance. To overcome such limitations, we propose an Acoustic Feature Transformation (AFT) technique that aligns the temporal features of old classes to the new space, including a selectively compressed feature space. AFT mitigates the forgetting of old knowledge without retaining past data. We conducted experiments on two datasets, showing consistent improvements over baseline models with accuracy gains of 3.7\% to 3.9\%.
>
---
#### [new 037] Fine-Tuning Large Multimodal Models for Automatic Pronunciation Assessment
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究了在自动发音评估任务中微调大型多模态模型的效果，旨在提升细粒度评估能力。通过在公开和私有数据集上实验，发现微调显著优于零样本设置，尤其在词和句子层面表现良好，但音素级评估仍有挑战，并指出Spearman相关系数更适合作序数一致性评估。**

- **链接: [http://arxiv.org/pdf/2509.15701v1](http://arxiv.org/pdf/2509.15701v1)**

> **作者:** Ke Wang; Wenning Wei; Yan Deng; Lei He; Sheng Zhao
>
> **备注:** submitted to ICASSP2026
>
> **摘要:** Automatic Pronunciation Assessment (APA) is critical for Computer-Assisted Language Learning (CALL), requiring evaluation across multiple granularities and aspects. Large Multimodal Models (LMMs) present new opportunities for APA, but their effectiveness in fine-grained assessment remains uncertain. This work investigates fine-tuning LMMs for APA using the Speechocean762 dataset and a private corpus. Fine-tuning significantly outperforms zero-shot settings and achieves competitive results on single-granularity tasks compared to public and commercial systems. The model performs well at word and sentence levels, while phoneme-level assessment remains challenging. We also observe that the Pearson Correlation Coefficient (PCC) reaches 0.9, whereas Spearman's rank Correlation Coefficient (SCC) remains around 0.6, suggesting that SCC better reflects ordinal consistency. These findings highlight both the promise and limitations of LMMs for APA and point to future work on fine-grained modeling and rank-aware evaluation.
>
---
#### [new 038] MAGENTA: Magnitude and Geometry-ENhanced Training Approach for Robust Long-Tailed Sound Event Localization and Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对长尾数据下的声事件定位与检测（SELD）任务，提出MAGENTA方法，通过几何分解回归误差，增强对罕见事件的建模，提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.15599v1](http://arxiv.org/pdf/2509.15599v1)**

> **作者:** Jun-Wei Yeow; Ee-Leng Tan; Santi Peksi; Woon-Seng Gan
>
> **备注:** This work has been submitted to IEEE ICASSP 2026 for possible publication
>
> **摘要:** Deep learning-based Sound Event Localization and Detection (SELD) systems degrade significantly on real-world, long-tailed datasets. Standard regression losses bias learning toward frequent classes, causing rare events to be systematically under-recognized. To address this challenge, we introduce MAGENTA (Magnitude And Geometry-ENhanced Training Approach), a unified loss function that counteracts this bias within a physically interpretable vector space. MAGENTA geometrically decomposes the regression error into radial and angular components, enabling targeted, rarity-aware penalties and strengthened directional modeling. Empirically, MAGENTA substantially improves SELD performance on imbalanced real-world data, providing a principled foundation for a new class of geometry-aware SELD objectives. Code is available at: https://github.com/itsjunwei/MAGENTA_ICASSP
>
---
#### [new 039] A Steered Response Power Method for Sound Source Localization With Generic Acoustic Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对声源定位任务，提出一种通用的SRP方法，解决传统方法依赖简化声学假设的问题。通过引入通用声学模型和优化波束成形准则，提升了复杂环境下的定位精度，实验证明在噪声条件下误差降低超60%。**

- **链接: [http://arxiv.org/pdf/2509.15702v1](http://arxiv.org/pdf/2509.15702v1)**

> **作者:** Kaspar Müller; Markus Buck; Simon Doclo; Jan Østergaard; Tobias Wolff
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** The steered response power (SRP) method is one of the most popular approaches for acoustic source localization with microphone arrays. It is often based on simplifying acoustic assumptions, such as an omnidirectional sound source in the far field of the microphone array(s), free field propagation, and spatially uncorrelated noise. In reality, however, there are many acoustic scenarios where such assumptions are violated. This paper proposes a generalization of the conventional SRP method that allows to apply generic acoustic models for localization with arbitrary microphone constellations. These models may consider, for instance, level differences in distributed microphones, the directivity of sources and receivers, or acoustic shadowing effects. Moreover, also measured acoustic transfer functions may be applied as acoustic model. We show that the delay-and-sum beamforming of the conventional SRP is not optimal for localization with generic acoustic models. To this end, we propose a generalized SRP beamforming criterion that considers generic acoustic models and spatially correlated noise, and derive an optimal SRP beamformer. Furthermore, we propose and analyze appropriate frequency weightings. Unlike the conventional SRP, the proposed method can jointly exploit observed level and time differences between the microphone signals to infer the source location. Realistic simulations of three different microphone setups with speech under various noise conditions indicate that the proposed method can significantly reduce the mean localization error compared to the conventional SRP and, in particular, a reduction of more than 60% can be archived in noisy conditions.
>
---
#### [new 040] BiRQ: Bi-Level Self-Labeling Random Quantization for Self-Supervised Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出BiRQ，一种用于自监督语音识别的双层随机量化框架。旨在解决伪标签生成中效率与性能的平衡问题，通过模型自身生成增强标签，结合高效量化与迭代优化，提升表示学习效果，验证于多个语音数据集。**

- **链接: [http://arxiv.org/pdf/2509.15430v1](http://arxiv.org/pdf/2509.15430v1)**

> **作者:** Liuyuan Jiang; Xiaodong Cui; Brian Kingsbury; Tianyi Chen; Lisha Chen
>
> **备注:** 5 pages including reference
>
> **摘要:** Speech is a rich signal, and labeled audio-text pairs are costly, making self-supervised learning essential for scalable representation learning. A core challenge in speech SSL is generating pseudo-labels that are both informative and efficient: strong labels, such as those used in HuBERT, improve downstream performance but rely on external encoders and multi-stage pipelines, while efficient methods like BEST-RQ achieve simplicity at the cost of weaker labels. We propose BiRQ, a bilevel SSL framework that combines the efficiency of BEST-RQ with the refinement benefits of HuBERT-style label enhancement. The key idea is to reuse part of the model itself as a pseudo-label generator: intermediate representations are discretized by a random-projection quantizer to produce enhanced labels, while anchoring labels derived directly from the raw input stabilize training and prevent collapse. Training is formulated as an efficient first-order bilevel optimization problem, solved end-to-end with differentiable Gumbel-softmax selection. This design eliminates the need for external label encoders, reduces memory cost, and enables iterative label refinement in an end-to-end fashion. BiRQ consistently improves over BEST-RQ while maintaining low complexity and computational efficiency. We validate our method on various datasets, including 960-hour LibriSpeech, 150-hour AMI meetings and 5,000-hour YODAS, demonstrating consistent gains over BEST-RQ.
>
---
#### [new 041] Rec-RIR: Monaural Blind Room Impulse Response Identification via DNN-based Reverberant Speech Reconstruction in STFT Domain
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出Rec-RIR，用于单通道盲环境脉冲响应（RIR）识别。通过在STFT域中利用DNN重建混响语音，估计CTF滤波器，并转换为时域RIR，实现了SOTA性能的RIR识别与声学参数估计。**

- **链接: [http://arxiv.org/pdf/2509.15628v1](http://arxiv.org/pdf/2509.15628v1)**

> **作者:** Pengyu Wang; Xiaofei Li
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Room impulse response (RIR) characterizes the complete propagation process of sound in an enclosed space. This paper presents Rec-RIR for monaural blind RIR identification. Rec-RIR is developed based on the convolutive transfer function (CTF) approximation, which models reverberation effect within narrow-band filter banks in the short-time Fourier transform (STFT) domain. Specifically, we propose a deep neural network (DNN) with cross-band and narrow-band blocks to estimate the CTF filter. The DNN is trained through reconstructing the noise-free reverberant speech spectra. This objective enables stable and straightforward supervised training. Subsequently, a pseudo intrusive measurement process is employed to convert the CTF filter estimate into time-domain RIR by simulating a common intrusive RIR measurement procedure. Experimental results demonstrate that Rec-RIR achieves state-of-the-art (SOTA) performance in both RIR identification and acoustic parameter estimation. Open-source codes are available online at https://github.com/Audio-WestlakeU/Rec-RIR.
>
---
#### [new 042] VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文提出VoXtream，一种低延迟的流式文本到语音（TTS）系统。任务是实现实时语音合成，解决初始延迟高的问题。采用全自回归结构和动态对齐机制，使系统在GPU上初始延迟仅为102 ms，且在质量上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.15969v1](http://arxiv.org/pdf/2509.15969v1)**

> **作者:** Nikita Torgashov; Gustav Eje Henter; Gabriel Skantze
>
> **备注:** 5 pages, 1 figure, submitted to IEEE ICASSP 2026
>
> **摘要:** We present VoXtream, a fully autoregressive, zero-shot streaming text-to-speech (TTS) system for real-time use that begins speaking from the first word. VoXtream directly maps incoming phonemes to audio tokens using a monotonic alignment scheme and a dynamic look-ahead that does not delay onset. Built around an incremental phoneme transformer, a temporal transformer predicting semantic and duration tokens, and a depth transformer producing acoustic tokens, VoXtream achieves, to our knowledge, the lowest initial delay among publicly available streaming TTS: 102 ms on GPU. Despite being trained on a mid-scale 9k-hour corpus, it matches or surpasses larger baselines on several metrics, while delivering competitive quality in both output- and full-streaming settings. Demo and code are available at https://herimor.github.io/voxtream.
>
---
#### [new 043] EmoHeal: An End-to-End System for Personalized Therapeutic Music Retrieval from Fine-grained Emotions
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文提出EmoHeal系统，用于个性化治疗性音乐推荐。针对现有工具忽视细腻情绪的问题，通过情感识别、知识图谱和音频检索技术，实现基于27种细粒度情绪的精准音乐疗愈，验证了情绪感知与疗效的相关性。**

- **链接: [http://arxiv.org/pdf/2509.15986v1](http://arxiv.org/pdf/2509.15986v1)**

> **作者:** Xinchen Wan; Jinhua Liang; Huan Zhang
>
> **备注:** 5 pages, 5 figures. Submitted to the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)
>
> **摘要:** Existing digital mental wellness tools often overlook the nuanced emotional states underlying everyday challenges. For example, pre-sleep anxiety affects more than 1.5 billion people worldwide, yet current approaches remain largely static and "one-size-fits-all", failing to adapt to individual needs. In this work, we present EmoHeal, an end-to-end system that delivers personalized, three-stage supportive narratives. EmoHeal detects 27 fine-grained emotions from user text with a fine-tuned XLM-RoBERTa model, mapping them to musical parameters via a knowledge graph grounded in music therapy principles (GEMS, iso-principle). EmoHeal retrieves audiovisual content using the CLAMP3 model to guide users from their current state toward a calmer one ("match-guide-target"). A within-subjects study (N=40) demonstrated significant supportive effects, with participants reporting substantial mood improvement (M=4.12, p<0.001) and high perceived emotion recognition accuracy (M=4.05, p<0.001). A strong correlation between perceived accuracy and therapeutic outcome (r=0.72, p<0.001) validates our fine-grained approach. These findings establish the viability of theory-driven, emotion-aware digital wellness tools and provides a scalable AI blueprint for operationalizing music therapy principles.
>
---
#### [new 044] State-of-the-Art Dysarthric Speech Recognition with MetaICL for on-the-fly Personalization
- **分类: eess.AS; cs.SD**

- **简介: 该论文聚焦于失语症语音识别任务，旨在解决个性化模型训练和存储的挑战。提出了一种基于元学习和上下文学习的混合方法，实现零样本和少样本的即时个性化，显著提升了识别准确率，并验证了示例选择对效率的重要性。**

- **链接: [http://arxiv.org/pdf/2509.15516v1](http://arxiv.org/pdf/2509.15516v1)**

> **作者:** Dhruuv Agarwal; Harry Zhang; Yang Yu; Quan Wang
>
> **摘要:** Personalizing Automatic Speech Recognition (ASR) for dysarthric speech is crucial but challenging due to training and storing of individual user adapters. We propose a hybrid meta-training method for a single model, excelling in zero-shot and few-shot on-the-fly personalization via in-context learning (ICL). Measuring Word Error Rate (WER) on state-of-the-art subsets, the model achieves 13.9% WER on Euphonia which surpasses speaker-independent baselines (17.5% WER) and rivals user-specific personalized models. On SAP Test 1, its 5.3% WER significantly bests the 8% from even personalized adapters. We also demonstrate the importance of example curation, where an oracle text-similarity method shows 5 curated examples can achieve performance similar to 19 randomly selected ones, highlighting a key area for future efficiency gains. Finally, we conduct data ablations to measure the data efficiency of this approach. This work presents a practical, scalable, and personalized solution.
>
---
#### [new 045] Interpretable Modeling of Articulatory Temporal Dynamics from real-time MRI for Phoneme Recognition
- **分类: eess.IV; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在利用实时MRI数据提升音素识别的准确性与可解释性。研究比较了多种特征表示方法，并结合多特征建模，发现舌部和唇部运动对识别贡献显著，最终实现0.34的最低音素错误率。**

- **链接: [http://arxiv.org/pdf/2509.15689v1](http://arxiv.org/pdf/2509.15689v1)**

> **作者:** Jay Park; Hong Nguyen; Sean Foley; Jihwan Lee; Yoonjeong Lee; Dani Byrd; Shrikanth Narayanan
>
> **摘要:** Real-time Magnetic Resonance Imaging (rtMRI) visualizes vocal tract action, offering a comprehensive window into speech articulation. However, its signals are high dimensional and noisy, hindering interpretation. We investigate compact representations of spatiotemporal articulatory dynamics for phoneme recognition from midsagittal vocal tract rtMRI videos. We compare three feature types: (1) raw video, (2) optical flow, and (3) six linguistically-relevant regions of interest (ROIs) for articulator movements. We evaluate models trained independently on each representation, as well as multi-feature combinations. Results show that multi-feature models consistently outperform single-feature baselines, with the lowest phoneme error rate (PER) of 0.34 obtained by combining ROI and raw video. Temporal fidelity experiments demonstrate a reliance on fine-grained articulatory dynamics, while ROI ablation studies reveal strong contributions from tongue and lips. Our findings highlight how rtMRI-derived features provide accuracy and interpretability, and establish strategies for leveraging articulatory data in speech processing.
>
---
## 更新

#### [replaced 001] LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models Using in-the-wild Data
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04586v2](http://arxiv.org/pdf/2506.04586v2)**

> **作者:** Wen Ding; Fan Qian
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Although state-of-the-art Speech Foundation Models can produce high-quality text pseudo-labels, applying Semi-Supervised Learning (SSL) for in-the-wild real-world data remains challenging due to its richer and more complex acoustics compared to curated datasets. To address the challenges, we introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that uses Large Language Models (LLMs) to correct pseudo-labels generated on in-the-wild data. In the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and further improved by a data filtering strategy. Across Mandarin ASR and Spanish-to-English AST evaluations, LESS delivers consistent gains, with an absolute Word Error Rate reduction of 3.8% on WenetSpeech, and BLEU score increase of 0.8 and 0.7, achieving 34.0 on Callhome and 64.7 on Fisher testsets respectively. These results highlight LESS's effectiveness across diverse languages, tasks, and domains. We have released the recipe as open source to facilitate further research in this area.
>
---
#### [replaced 002] Sound-Based Spin Estimation in Table Tennis: Dataset and Real-Time Classification Pipeline
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.11760v2](http://arxiv.org/pdf/2409.11760v2)**

> **作者:** Thomas Gossard; Julian Schmalzl; Andreas Ziegler; Andreas Zell
>
> **备注:** Accepted to IEEE Star 2025
>
> **摘要:** Sound can complement vision in ball sports by providing subtle cues about contact dynamics. In table tennis, the brief, high-frequency sounds produced during racket-ball impacts carry information about the racket type, the surface contacted, and whether spin was applied. We address three key problems in this domain: (1) precise bounce detection with millisecond-level temporal accuracy, (2) classification of bounce surface (e.g., racket, table, floor), and (3) spin detection from audio alone. To this end, we propose a real-time-capable pipeline that combines energy-based peak detection with convolutional neural networks trained on a novel dataset of 3,396 bounce samples recorded across 10 racket configurations. The system achieves accurate and low-latency detection of bounces, and reliably classifies both the surface of contact and whether spin was applied. This audio-based approach opens up new possibilities for spin estimation in robotic systems and for real-time feedback in coaching tools. We publicly release both the dataset and code to support further research.
>
---
#### [replaced 003] MeanFlowSE: one-step generative speech enhancement via conditional mean flow
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.14858v2](http://arxiv.org/pdf/2509.14858v2)**

> **作者:** Duojia Li; Shenghui Lu; Hongchen Pan; Zongyi Zhan; Qingyang Hong; Lin Li
>
> **摘要:** Multistep inference is a bottleneck for real-time generative speech enhancement because flow- and diffusion-based systems learn an instantaneous velocity field and therefore rely on iterative ordinary differential equation (ODE) solvers. We introduce MeanFlowSE, a conditional generative model that learns the average velocity over finite intervals along a trajectory. Using a Jacobian-vector product (JVP) to instantiate the MeanFlow identity, we derive a local training objective that directly supervises finite-interval displacement while remaining consistent with the instantaneous-field constraint on the diagonal. At inference, MeanFlowSE performs single-step generation via a backward-in-time displacement, removing the need for multistep solvers; an optional few-step variant offers additional refinement. On VoiceBank-DEMAND, the single-step model achieves strong intelligibility, fidelity, and perceptual quality with substantially lower computational cost than multistep baselines. The method requires no knowledge distillation or external teachers, providing an efficient, high-fidelity framework for real-time generative speech enhancement. The proposed method is open-sourced at https://github.com/liduojia1/MeanFlowSE.
>
---
#### [replaced 004] Enhancing Speech Large Language Models with Prompt-Aware Mixture of Audio Encoders
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.15178v3](http://arxiv.org/pdf/2502.15178v3)**

> **作者:** Weiqiao Shan; Yuang Li; Yuhao Zhang; Yingfeng Luo; Chen Xu; Xiaofeng Zhao; Long Meng; Yunfei Lu; Min Zhang; Hao Yang; Tong Xiao; Jingbo Zhu
>
> **备注:** 16 pages,5 figures, 13 tables, to be published in EMNLP 2025 main conference
>
> **摘要:** Connecting audio encoders with large language models (LLMs) allows the LLM to perform various audio understanding tasks, such as automatic speech recognition (ASR) and audio captioning (AC). Most research focuses on training an adapter layer to generate a unified audio feature for the LLM. However, different tasks may require distinct features that emphasize either semantic or acoustic aspects, making task-specific audio features more desirable. In this paper, we propose Prompt-aware Mixture (PaM) to enhance the Speech LLM that uses multiple audio encoders. Our approach involves using different experts to extract different features based on the prompt that indicates different tasks. Experiments demonstrate that with PaM, only one Speech LLM surpasses the best performances achieved by all single-encoder Speech LLMs on ASR, Speaker Number Verification, and AC tasks. PaM also outperforms other feature fusion baselines, such as concatenation and averaging. Our code would be available at: https://github.com/shanweiqiao/PaM
>
---
#### [replaced 005] SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning
- **分类: eess.SP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19668v4](http://arxiv.org/pdf/2502.19668v4)**

> **作者:** Mingsheng Cai; Jiuming Jiang; Wenhao Huang; Che Liu; Rossella Arcucci
>
> **备注:** Findings of The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) is critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME is pre-trained using structured diagnostic labels derived from ECG report entities through a one-time offline extraction with Large Language Models (LLMs), which help denoise, standardize cardiac concepts, and improve clinical representation learning. By fusing ECG signals with textual cardiac queries instead of fixed labels, SuPreME enables zero-shot classification of unseen conditions without further fine-tuning. We evaluate SuPreME on six downstream datasets covering 106 cardiac conditions, achieving superior zero-shot AUC performance of $77.20\%$, surpassing state-of-the-art eSSLs by $4.98\%$. Results demonstrate SuPreME's effectiveness in leveraging structured, clinically relevant knowledge for high-quality ECG representations.
>
---
#### [replaced 006] From Hype to Insight: Rethinking Large Language Model Integration in Visual Speech Recognition
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.14880v2](http://arxiv.org/pdf/2509.14880v2)**

> **作者:** Rishabh Jain; Naomi Harte
>
> **备注:** The authors have decided to withdraw the paper for further development
>
> **摘要:** Advances in self-supervised encoders have improved Visual Speech Recognition (VSR). Recent approaches integrating these encoders with LLM decoders improves transcription accuracy; however, it remains unclear whether these gains stem from visual understanding or stronger language modeling. In this work, we systematically evaluate LLM decoders by freezing or selectively updating the visual encoder, scaling decoder size, comparing adaptation strategies and architectures, and varying training data across LRS2, LRS3, and their combination. Evaluation on LRS2, LRS3, and WildVSR shows that scaling and adaptation yield limited improvements, while combining datasets enhances generalization. Semantic analysis reveals that gains arise primarily from lexical rather than semantic processing. Our Llama-2-13B model trained on the combined set achieves 24.7\% WER on LRS3 and 47.0\% on WildVSR, establishing SOTA among models trained without additional supervision. Our findings indicate LLM decoders refine contextual reasoning rather than visual features, emphasizing the need for stronger visual encoders to drive meaningful progress.
>
---
#### [replaced 007] DiTSE: High-Fidelity Generative Speech Enhancement via Latent Diffusion Transformers
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.09381v2](http://arxiv.org/pdf/2504.09381v2)**

> **作者:** Heitor R. Guimarães; Jiaqi Su; Rithesh Kumar; Tiago H. Falk; Zeyu Jin
>
> **备注:** Manuscript under review
>
> **摘要:** Real-world speech recordings suffer from degradations such as background noise and reverberation. Speech enhancement aims to mitigate these issues by generating clean high-fidelity signals. While recent generative approaches for speech enhancement have shown promising results, they still face two major challenges: (1) content hallucination, where plausible phonemes generated differ from the original utterance; and (2) inconsistency, failing to preserve speaker's identity and paralinguistic features from the input speech. In this work, we introduce DiTSE (Diffusion Transformer for Speech Enhancement), which addresses quality issues of degraded speech in full bandwidth. Our approach employs a latent diffusion transformer model together with robust conditioning features, effectively addressing these challenges while remaining computationally efficient. Experimental results from both subjective and objective evaluations demonstrate that DiTSE achieves state-of-the-art audio quality that, for the first time, matches real studio-quality audio from the DAPS dataset. Furthermore, DiTSE significantly improves the preservation of speaker identity and content fidelity, reducing hallucinations across datasets compared to state-of-the-art enhancers. Audio samples are available at: http://hguimaraes.me/DiTSE
>
---
#### [replaced 008] Generating Moving 3D Soundscapes with Latent Diffusion Models
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07318v2](http://arxiv.org/pdf/2507.07318v2)**

> **作者:** Christian Templin; Yanda Zhu; Hao Wang
>
> **摘要:** Spatial audio has become central to immersive applications such as VR/AR, cinema, and music. Existing generative audio models are largely limited to mono or stereo formats and cannot capture the full 3D localization cues available in first-order Ambisonics (FOA). Recent FOA models extend text-to-audio generation but remain restricted to static sources. In this work, we introduce SonicMotion, the first end-to-end latent diffusion framework capable of generating FOA audio with explicit control over moving sound sources. SonicMotion is implemented in two variations: 1) a descriptive model conditioned on natural language prompts, and 2) a parametric model conditioned on both text and spatial trajectory parameters for higher precision. To support training and evaluation, we construct a new dataset of over one million simulated FOA caption pairs that include both static and dynamic sources with annotated azimuth, elevation, and motion attributes. Experiments show that SonicMotion achieves state-of-the-art semantic alignment and perceptual quality comparable to leading text-to-audio systems, while uniquely attaining low spatial localization error.
>
---
#### [replaced 009] Perceptually Transparent Binaural Auralization of Simulated Sound Fields
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2412.05015v2](http://arxiv.org/pdf/2412.05015v2)**

> **作者:** Jens Ahrens
>
> **摘要:** Contrary to geometric acoustics-based simulations where the spatial information is available in a tangible form, it is not straightforward to auralize wave-based simulations. A variety of methods have been proposed that compute the ear signals of a virtual listener with known head-related transfer functions from sampling either the sound pressure or the particle velocity (or both) of the simulated sound field. This article summarizes the most common binaural auralization methods with and without intermediate ambisonic representation of volumetrically sampled sound pressure or sound pressure and particle velocity sampled on spherical or cubical surfaces and presents a perceptual validation thereof. A triangular test ($N=19$) confirmed that all evaluated grids resulted in a perceptually transparent auralization for the three tested sound incidence angles under reverberant conditions. Under anechoic conditions, only the high-density spherical and cubical surface grids lead to transparent auralization. All tested methods are available open source in the Chalmers Auralization Toolbox that accompanies this article.
>
---
#### [replaced 010] Tiny is not small enough: High-quality, low-resource facial animation models through hybrid knowledge distillation
- **分类: cs.GR; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.18352v2](http://arxiv.org/pdf/2507.18352v2)**

> **作者:** Zhen Han; Mattias Teye; Derek Yadgaroff; Judith Bütepage
>
> **备注:** Accepted to ACM TOG 2025 (SIGGRAPH journal track); Project page: https://electronicarts.github.io/tiny-voice2face/
>
> **摘要:** The training of high-quality, robust machine learning models for speech-driven 3D facial animation requires a large, diverse dataset of high-quality audio-animation pairs. To overcome the lack of such a dataset, recent work has introduced large pre-trained speech encoders that are robust to variations in the input audio and, therefore, enable the facial animation model to generalize across speakers, audio quality, and languages. However, the resulting facial animation models are prohibitively large and lend themselves only to offline inference on a dedicated machine. In this work, we explore on-device, real-time facial animation models in the context of game development. We overcome the lack of large datasets by using hybrid knowledge distillation with pseudo-labeling. Given a large audio dataset, we employ a high-performing teacher model to train very small student models. In contrast to the pre-trained speech encoders, our student models only consist of convolutional and fully-connected layers, removing the need for attention context or recurrent updates. In our experiments, we demonstrate that we can reduce the memory footprint to up to 3.4 MB and required future audio context to up to 81 ms while maintaining high-quality animations. This paves the way for on-device inference, an important step towards realistic, model-driven digital characters.
>
---
#### [replaced 011] FOVAL: Calibration-Free and Subject-Invariant Fixation Depth Estimation Across Diverse Eye-Tracking Datasets
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2408.03591v2](http://arxiv.org/pdf/2408.03591v2)**

> **作者:** Benedikt W. Hosp
>
> **摘要:** Accurate fixation depth estimation is essential for applications in extended reality (XR), robotics, and human-computer interaction. However, current methods heavily depend on user-specific calibration, which limits their scalability and usability. We introduce FOVAL, a robust calibration-free approach that combines spatiotemporal sequence modelling via Long Short-Term Memory (LSTM) networks with subject-invariant feature engineering and normalisation. Compared to Transformers, Temporal Convolutional Networks (TCNs), and CNNs, FOVAL achieves superior performance, particularly in scenarios with limited and noisy gaze data. Evaluations across three benchmark datasets using Leave-One-Out Cross-Validation (LOOCV) and cross-dataset validation show a mean absolute error (MAE) of 9.1 cm and strong generalisation without calibration. We further analyse inter-subject variability and domain shifts, providing insight into model robustness and adaptation. FOVAL's scalability and accuracy make it highly suitable for real-world deployment.
>
---
#### [replaced 012] Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.14049v2](http://arxiv.org/pdf/2509.14049v2)**

> **作者:** Jordi Grau-Haro; Ruben Ribes-Serrano; Javier Naranjo-Alcazar; Marta Garcia-Ballesteros; Pedro Zuccarello
>
> **备注:** Accepted at Computing Conference 2026, London, UK
>
> **摘要:** Convolutional Neural Networks (CNNs) have demonstrated exceptional performance in audio tagging tasks. However, deploying these models on resource-constrained devices like the Raspberry Pi poses challenges related to computational efficiency and thermal management. In this paper, a comprehensive evaluation of multiple convolutional neural network (CNN) architectures for audio tagging on the Raspberry Pi is conducted, encompassing all 1D and 2D models from the Pretrained Audio Neural Networks (PANNs) framework, a ConvNeXt-based model adapted for audio classification, as well as MobileNetV3 architectures. In addition, two PANNs-derived networks, CNN9 and CNN13, recently proposed, are also evaluated. To enhance deployment efficiency and portability across diverse hardware platforms, all models are converted to the Open Neural Network Exchange (ONNX) format. Unlike previous works that focus on a single model, our analysis encompasses a broader range of architectures and involves continuous 24-hour inference sessions to assess performance stability. Our experiments reveal that, with appropriate model selection and optimization, it is possible to maintain consistent inference latency and manage thermal behavior effectively over extended periods. These findings provide valuable insights for deploying audio tagging models in real-world edge computing scenarios.
>
---
#### [replaced 013] Improving Anomalous Sound Detection with Attribute-aware Representation from Domain-adaptive Pre-training
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.12845v2](http://arxiv.org/pdf/2509.12845v2)**

> **作者:** Xin Fang; Guirui Zhong; Qing Wang; Fan Chu; Lei Wang; Mengui Qian; Mingqi Cai; Jiangzhao Wu; Jianqing Gao; Jun Du
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Anomalous Sound Detection (ASD) is often formulated as a machine attribute classification task, a strategy necessitated by the common scenario where only normal data is available for training. However, the exhaustive collection of machine attribute labels is laborious and impractical. To address the challenge of missing attribute labels, this paper proposes an agglomerative hierarchical clustering method for the assignment of pseudo-attribute labels using representations derived from a domain-adaptive pre-trained model, which are expected to capture machine attribute characteristics. We then apply model adaptation to this pre-trained model through supervised fine-tuning for machine attribute classification, resulting in a new state-of-the-art performance. Evaluation on the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge dataset demonstrates that our proposed approach yields significant performance gains, ultimately outperforming our previous top-ranking system in the challenge.
>
---
#### [replaced 014] Evaluation of the Pronunciation of Tajweed Rules Based on DNN as a Step Towards Interactive Recitation Learning
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.23470v3](http://arxiv.org/pdf/2503.23470v3)**

> **作者:** Dim Shaiakhmetov; Gulnaz Gimaletdinova; Kadyrmamat Momunov; Selcuk Cankurt
>
> **摘要:** Proper recitation of the Quran, adhering to the rules of Tajweed, is crucial for preventing mistakes during recitation and requires significant effort to master. Traditional methods of teaching these rules are limited by the availability of qualified instructors and time constraints. Automatic evaluation of recitation can address these challenges by providing prompt feedback and supporting independent practice. This study focuses on developing a deep learning model to classify three Tajweed rules - separate stretching (Al Mad), tight noon (Ghunnah), and hide (Ikhfaa) - using the publicly available QDAT dataset, which contains over 1,500 audio recordings. The input data consisted of audio recordings from this dataset, transformed into normalized mel-spectrograms. For classification, the EfficientNet-B0 architecture was used, enhanced with a Squeeze-and-Excitation attention mechanism. The developed model achieved accuracy rates of 95.35%, 99.34%, and 97.01% for the respective rules. An analysis of the learning curves confirmed the model's robustness and absence of overfitting. The proposed approach demonstrates high efficiency and paves the way for developing interactive educational systems for Tajweed study.
>
---
