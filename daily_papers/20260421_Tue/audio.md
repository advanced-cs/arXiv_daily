# 音频 cs.SD;  eess.AS

- **最新发布 30 篇**

- **更新 30 篇**

## 最新发布

#### [new 001] A novel LSTM music generator based on the fractional time-frequency feature extraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于音乐生成任务，旨在利用AI生成高质量音乐。通过结合分数阶傅里叶变换和LSTM网络，提取音乐特征并生成新音乐。**

- **链接: [https://arxiv.org/pdf/2604.17823](https://arxiv.org/pdf/2604.17823)**

> **作者:** Li Ya; Chen Wei; Li Xiulai; Yu Lei; Deng Xinyi; Chen Chaofan
>
> **备注:** This work was supported by Hainan Provincial Natural Science Foundation of China (Grant No. 723QN238)
>
> **摘要:** In this paper, we propose a novel approach for generating music based on an artificial intelligence (AI) system. We analyze the features of music and use them to fit and predict the music. The fractional Fourier transform (FrFT) and the long short-term memory (LSTM) network are the foundations of our method. The FrFT method is used to extract the spectral features of a music piece, where the music signal is expressed on the time and frequency domains. The LSTM network is used to generate new music based on the extracted features, where we predict the music according to the hidden layer features and real-time inputs using GiantMIDI-Piano dataset. The results of our experiments show that our proposed system is capable of generating high-quality music that is comparable to human-generated music.
>
---
#### [new 002] Coexisting Tempo Traditions in Beethoven's Piano and Cello Sonatas: A K-means Clustering Analysis of Recorded Performances, 1930-2012
- **分类: cs.SD**

- **简介: 该论文属于音乐表演分析任务，旨在探讨贝多芬钢琴与大提琴奏鸣曲录音中的节奏传统。通过k-means聚类分析，发现多种独立的节奏传统共存，而非单一演变。**

- **链接: [https://arxiv.org/pdf/2604.16658](https://arxiv.org/pdf/2604.16658)**

> **作者:** Ignasi Sole
>
> **摘要:** Empirical studies of recorded performance have conventionally modelled tempo change as a unidirectional historical process, fitting linear regression lines to tempo data plotted against recording year. This paper argues that such approaches impose a false narrative of uniform stylistic evolution on what is, in fact, a plurality of coexisting interpretive traditions. Applying k-means clustering (k=3) to bar-level BPM data from over one hundred recordings of Beethoven's five piano and cello sonatas (Op. 5 Nos. 1 and 2; Op. 69; Op. 102 Nos. 1 and 2) spanning 1930-2012, this study reveals that every movement supports at least two, and usually three, discrete tempo traditions (slow, mid-range, and fast), whose internal regression slopes are negligible (R-squared <= 0.25 in all but one case), demonstrating that each tradition is independently stable across eight decades. The mid-range cluster dominates in all movements, typically comprising 55-70% of recordings. A slow cluster is absent from fast-character movements (Op. 5 Rondos, Op. 69 Scherzo), reflecting a shared rhetorical consensus about their character. The single case of significant intra-cluster drift (Op. 102 No. 1 Allegro con brio, R-squared=0.246, p=0.013) indicates a moderate mid-range deceleration of approximately 3.2 BPM across the study period. No correlation is found between cluster membership and performers' generational, national, or pedagogical backgrounds, suggesting that tempo tradition reflects individual interpretive choice rather than collective cultural inheritance. The paper proposes an ecological model of stylistic change - coexisting traditions shifting in relative prevalence rather than a single tradition evolving - and argues that this reframing has broad implications for how empirical performance studies interpret corpus-level tempo data.
>
---
#### [new 003] Neural Encoding Detection is Not All You Need for Synthetic Speech Detection
- **分类: eess.AS**

- **简介: 该论文属于合成语音检测任务，旨在探讨神经编码检测的局限性，提出未来研究方向，解决过度依赖单一方法的风险。**

- **链接: [https://arxiv.org/pdf/2604.16700](https://arxiv.org/pdf/2604.16700)**

> **作者:** Luca Cuccovillo; Xin Wang; Milica Gerhardt; Patrick Aichroth
>
> **备注:** To appear in the proceedings of the IEEE International Workshop on Biometrics and Forensics (IWBF), Sophia Antipolis (France), 2026. Supplementary material available online at: this https URL
>
> **摘要:** This paper reviews the current state and emerging trends in synthetic speech detection. It outlines the main data-driven approaches, discusses the advantages and drawbacks of focusing future research solely on neural encoding detection, and offers recommendations for promising research directions. Unlike works that introduce new detection methods or datasets, this paper aims to guide future state-of-the-art research in the field and to highlight the risk of overcommitting to approaches that may not stand the test of time.
>
---
#### [new 004] Aligning Language Models for Lyric-to-Melody Generation with Rule-Based Musical Constraints
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于歌词到旋律生成任务，解决模型生成音乐不合规问题。通过定义音乐规则约束，构建偏好数据集，并使用DPO和KTO优化模型，提升生成旋律的音乐性与连贯性。**

- **链接: [https://arxiv.org/pdf/2604.18489](https://arxiv.org/pdf/2604.18489)**

> **作者:** Hao Meng; Siyuan Zheng; Shuran Zhou; Qiangqiang Wang; Yang Song
>
> **备注:** Accepted by IEEE ICASSP 2026
>
> **摘要:** Large Language Models (LLMs) show promise in lyric-to-melody generation, but models trained with Supervised Fine-Tuning (SFT) often produce musically implausible melodies with issues like poor rhythm and unsuitable vocal ranges, a phenomenon we term "constraint violation". To address this, we propose a novel alignment framework that instills musical knowledge without human annotation. We define rule-based musical constraints to automatically generate a preference dataset from an SFT model's outputs. The model is then aligned through a sequential process, first using Direct Preference Optimization (DPO) on paired preference data, followed by Kahneman-Tversky Optimization (KTO) on unpaired negative samples. Experimental results demonstrate that our aligned model substantially reduces rule violations and outperforms strong baselines in both objective and subjective evaluations, generating melodies with substantially improved musicality and coherence. An interactive demo with audio comparisons is available at this https URL.
>
---
#### [new 005] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare
- **分类: eess.AS**

- **简介: 该论文提出HCFD任务，旨在检测病理语音中的编解码器伪造音频。通过构建病理感知数据集，验证现有模型不足，并提出PHOENIX-Mamba框架提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.17642](https://arxiv.org/pdf/2604.17642)**

> **作者:** Mohd Mujtaba Akhtar; Girish; Muskaan Singh
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** In this study, we present Healthcare Codec-Fake Detection (HCFD), a new task for detecting codec-fakes under pathological speech conditions. We intentionally focus on codec based synthetic speech in this work, since neural codec decoding forms a core building block in modern speech generation pipelines. First, we release Healthcare CodecFake, the first pathology-aware dataset containing paired real and NAC-synthesized speech across multipl clinical conditions and codec families. Our evaluations show that SOTA codec-fake detectors trained primarily on healthy speech perform poorly on Healthcare CodecFake, highlighting the need for HCFD-specific models. Second, we demonstrate that PaSST outperforms existing speech-based models for HCFD, benefiting from its patch-based spectro-temporal representation. Finally, we propose PHOENIX-Mamba, a geometry-aware framework that models codec-fakes as multiple self-discovered modes in hyperbolic space and achieves the strongest performance on HCFD across clinical conditions and codecs. Experiments on HCFK show that PHOENIX-Mamba (PaSST) achieves the best overall performance, reaching 97.04 Acc on E-Dep, 96.73 on E-Alz, and 96.57 on E-Dys, while maintaining strong results on Chinese with 94.41 (Dep), 94.40 (Alz), and 93.20 (Dys). This geometry-aware formulation enables self-discovered clustering of heterogeneous codec-fake modes in hyperbolic space, facilitating robust discrimination under pathological speech variability. PHOENIX-Mamba achieves topmost performance on the HCFD task across clinical conditions and codecs.
>
---
#### [new 006] iPhoneme: Brain-to-Text Communication for ALS Using ConformerXL Decoding
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于脑机接口任务，旨在解决ALS患者语音恢复问题。通过改进的解码模型和交互设计，提升脑电到文本的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.16441](https://arxiv.org/pdf/2604.16441)**

> **作者:** Yoonmin Cha; Dawit Chun; Sung Park
>
> **摘要:** Brain-computer interfaces (BCIs) for speech restoration hold transformative potential for the approximately 173,000--232,500 individuals worldwide with ALS-related dysarthria. Despite recent progress, high-performance speech BCIs have been demonstrated in only 22--31 patients globally, largely due to limitations in neural decoding accuracy and practical input interfaces. We present iPhoneme, a brain-to-text communication system that jointly addresses these challenges through integrated modeling and interaction design. The system combines a deep learning phoneme decoder based on a modified Conformer architecture (ConformerXL, 192.9M parameters) with a gaze-assisted phoneme input interface that mitigates the Midas touch problem in eye-tracking systems. The acoustic model incorporates a temporal prenet with multi-scale dilated convolutions and bidirectional GRU for neural jitter correction, temporal subsampling for CTC stability, and Pre-RMSNorm stabilization across 12 encoder blocks, trained with AdamW and cosine scheduling. On the interaction side, iPhoneme introduces a chorded gaze-plus-silent-speech paradigm that replaces dwell-time selection, enabling more efficient input. We evaluate the system on the T15 dataset (45 sessions, 8,071 trials) of 256-channel intracranial EEG from speech motor cortex regions. A 6-gram phoneme language model trained on 3.1M sequences, combined with WFST beam search (beam=128), achieves 92.14% phoneme accuracy (7.86% PER) and 73.39% word accuracy (26.61% WER), approximately 3% above prior state-of-the-art. The system operates on CPU with 180 ms latency, demonstrating real-time, high-accuracy brain-to-text communication for ALS.
>
---
#### [new 007] NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决LLM在ASR中的效率、鲁棒性和可定制性问题。通过优化训练策略和引入生产级优化，提升模型性能与实用性。**

- **链接: [https://arxiv.org/pdf/2604.18105](https://arxiv.org/pdf/2604.18105)**

> **作者:** Yuan Xie; Jiaqi Song; Guang Qiu; Xianliang Wang; Kai Qiao; Junfeng Yuan; Shengqing Liu; Yi Zhang; Bowen Chen; Ming Lei; Jie Gao; Jie Wu
>
> **摘要:** Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a mainstream paradigm in recent years. Although existing LLM-based ASR models demonstrate impressive performance on public benchmarks, their training remains predominantly data-driven, leaving key practical challenges insufficiently addressed -- particularly limited downward scalability in resource-constrained deployments and hallucinations under acoustically challenging conditions. To address these issues, we present NIM4-ASR, a production-oriented LLM-based ASR framework optimized for both efficiency and robustness. Grounded in a principled delineation of functional roles between the encoder and the LLM, we redesign the multi-stage training paradigm to align each module with its intended capability boundary. Specifically, we reformulate the pre-training architecture and objective to mitigate the modality gap and improve parameter efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic fidelity and constrain representation drift; and design an ASR-specialized reinforcement learning stage to further enhance recognition quality and robustness. We additionally incorporate a suite of production-oriented optimizations, including robustness under noisy and silent conditions, real-time streaming inference, and hotword customization via retrieval-augmented generation (RAG). Experiments show that NIM4-ASR achieves state-of-the-art performance on multiple public benchmarks with merely 2.3B parameters, while substantially outperforming larger-scale competitors on internal benchmarks -- particularly in entity-intensive real-world scenarios. NIM4-ASR further supports million-scale hotword customization via RAG with sub-millisecond retrieval latency, enabling efficient adaptation to emerging entities and personalized user requirements.
>
---
#### [new 008] Towards Building Speech Large Language Models for Multitask Understanding in Low-Resource Languages
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对低资源语言泰语的多任务语音大语言模型构建问题，提出XLSR-Thai、U-Align和Thai-SUP，解决数据不足与模型效果下降难题。**

- **链接: [https://arxiv.org/pdf/2509.14804](https://arxiv.org/pdf/2509.14804)**

> **作者:** Mingchen Shao; Bingshen Mu; Chengyou Wang; Hai Li; Ying Yan; Zhonghua Fu; Lei Xie
>
> **摘要:** Speech large language models (SLLMs) built on speech encoders, adapters, and LLMs demonstrate remarkable multitask understanding performance in high-resource languages such as English and Chinese. However, their effectiveness substantially degrades in low-resource languages such as Thai. This limitation arises from three factors: (1) existing commonly used speech encoders, like the Whisper family, underperform in low-resource languages and lack support for broader spoken language understanding tasks; (2) the ASR-based alignment paradigm requires training the entire SLLM, leading to high computational cost; (3) paired speech-text data in low-resource languages is scarce. To overcome these challenges in the low-resource language Thai, we introduce XLSR-Thai, the first self-supervised learning (SSL) speech encoder for Thai. It is obtained by continuously training the standard SSL XLSR model on 36,000 hours of Thai speech data. Furthermore, we propose U-Align, a speech-text alignment method that is more resource-efficient and multitask-effective than typical ASR-based alignment. Finally, we present Thai-SUP, a pipeline for generating Thai spoken language understanding data from high-resource languages, yielding the first Thai spoken language understanding dataset of over 1,000 hours. Multiple experiments demonstrate the effectiveness of our methods in building a Thai multitask-understanding SLLM. We open-source XLSR-Thai and Thai-SUP to facilitate future research.
>
---
#### [new 009] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频-文本检索任务，旨在解决传统基准与实际搜索行为不匹配的问题。提出OEA模型，引入用户意图查询和硬负样本评估方法，提升检索鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.18360](https://arxiv.org/pdf/2604.18360)**

> **作者:** HaeJun Yoo; Yongseop Shin; Insung Lee; Myoung-Wan Koo; Du-Seong Chang
>
> **备注:** Accepted at ACL 2026 Main Conference. Camera-ready version
>
> **摘要:** Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
>
---
#### [new 010] Latent Fourier Transform
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出LatentFT，用于生成音乐模型的频率域控制。任务是提升音乐生成的可解释性和可控性，通过分离不同时间尺度的音乐模式，实现更精确的条件生成与混合。**

- **链接: [https://arxiv.org/pdf/2604.17986](https://arxiv.org/pdf/2604.17986)**

> **作者:** Mason Wang; Cheng-Zhi Anna Huang
>
> **备注:** ICLR 2026 Oral
>
> **摘要:** We introduce the Latent Fourier Transform (LatentFT), a framework that provides novel frequency-domain controls for generative music models. LatentFT combines a diffusion autoencoder with a latent-space Fourier transform to separate musical patterns by timescale. By masking latents in the frequency domain during training, our method yields representations that can be manipulated coherently at inference. This allows us to generate musical variations and blends from reference examples while preserving characteristics at desired timescales, which are specified as frequencies in the latent space. LatentFT parallels the role of the equalizer in music production: while traditional equalizers operates on audible frequencies to shape timbre, LatentFT operates on latent-space frequencies to shape musical structure. Experiments and listening tests show that LatentFT improves condition adherence and quality compared to baselines. We also present a technique for hearing frequencies in the latent space in isolation, and show different musical attributes reside in different regions of the latent spectrum. Our results show how frequency-domain control in latent space provides an intuitive, continuous frequency axis for conditioning and blending, advancing us toward more interpretable and interactive generative music models.
>
---
#### [new 011] Video-Robin: Autoregressive Diffusion Planning for Intent-Grounded Video-to-Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视频到音乐生成任务，旨在解决现有模型在语义控制和音频质量上的不足。提出Video-Robin，结合自回归规划与扩散合成，提升生成音乐的语义对齐和质量。**

- **链接: [https://arxiv.org/pdf/2604.17656](https://arxiv.org/pdf/2604.17656)**

> **作者:** Vaibhavi Lokegaonkar; Aryan Vijay Bhosale; Vishnu Raj; Gouthaman KV; Ramani Duraiswami; Lie Lu; Sreyan Ghosh; Dinesh Manocha
>
> **摘要:** Video-to-music (V2M) is the fundamental task of creating background music for an input video. Recent V2M models achieve audiovisual alignment by typically relying on visual conditioning alone and provide limited semantic and stylistic controllability to the end user. In this paper, we present Video-Robin, a novel text-conditioned video-to-music generation model that enables fast, high-quality, semantically aligned music generation for video content. To balance musical fidelity and semantic understanding, Video-Robin integrates autoregressive planning with diffusion-based synthesis. Specifically, an autoregressive module models global structure by semantically aligning visual and textual inputs to produce high-level music latents. These latents are subsequently refined into coherent, high-fidelity music using local Diffusion Transformers. By factoring semantically driven planning into diffusion-based synthesis, Video-Robin enables fine-grained creator control without sacrificing audio realism. Our proposed model outperforms baselines that solely accept video input and additional feature conditioned baselines on both in-distribution and out-of-distribution benchmarks with a 2.21x speed in inference compared to SOTA. We will open-source everything upon paper acceptance.
>
---
#### [new 012] ICLAD: In-Context Learning with Comparison-Guidance for Audio Deepfake Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有系统对真实场景下深度伪造音频泛化能力差的问题。提出ICLAD框架，结合对比引导的上下文学习，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.16749](https://arxiv.org/pdf/2604.16749)**

> **作者:** Benjamin Chou; Yi Zhu; Surya Koppisetti
>
> **备注:** To appear at ACL Findings 2026
>
> **摘要:** Audio deepfakes pose a significant security threat, yet current state-of-the-art (SOTA) detection systems do not generalize well to realistic in-the-wild deepfakes. We introduce a novel \textbf{I}n-\textbf{C}ontext \textbf{L}earning paradigm with comparison-guidance for \textbf{A}udio \textbf{D}eepfake detection (\textbf{ICLAD}). The framework enables the use of audio language models (ALMs) for training-free generalization to unseen deepfakes and provides textual rationales on the detection outcome. At the core of ICLAD is a pairwise comparative reasoning strategy that guides the ALM to discover and filter hallucinations and deepfake-irrelevant acoustic attributes. The ALM works alongside a specialized deepfake detector, whereby a routing mechanism feeds out-of-distribution samples to the ALM. On in-the-wild datasets, ICLAD improves macro F1 over the specialized detector, with up to $2\times$ relative improvement. Further analysis demonstrates the flexibility of ICLAD and its potential for deployment on recent open-source ALMs.
>
---
#### [new 013] Audio-DeepThinker: Progressive Reasoning-Aware Reinforcement Learning for High-Quality Chain-of-Thought Emergence in Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频语言模型的推理任务，旨在解决音频理解中缺乏明确推理过程的问题。通过引入混合奖励和渐进式训练，提升模型生成高质量推理链的能力。**

- **链接: [https://arxiv.org/pdf/2604.18187](https://arxiv.org/pdf/2604.18187)**

> **作者:** Xiang He; Chenxing Li; Jinting Wang; Yan Rong; Tianxin Xie; Wenfu Wang; Li Liu; Dong Yu
>
> **摘要:** Large Audio-Language Models (LALMs) have made significant progress in audio understanding, yet they primarily operate as perception-and-answer systems without explicit reasoning processes. Existing methods for enhancing audio reasoning rely either on supervised chain-of-thought (CoT) fine-tuning, which is limited by training data quality, or on reinforcement learning (RL) with coarse rewards that do not directly evaluate reasoning quality. As a result, the generated reasoning chains often appear well-structured yet lack specific acoustic grounding. We propose Audio-DeepThinker, a framework built on two core ideas. First, we introduce a hybrid reasoning similarity reward that directly supervises the quality of generated reasoning chains by combining an LLM evaluator assessing logical path alignment, key step coverage, and analytical depth with an embedding similarity component enforcing semantic alignment with reference reasoning chains. Second, we propose a progressive two-stage curriculum that enables high-quality CoT reasoning to emerge through pure RL exploration, without any supervised reasoning fine-tuning, from an instruction-tuned model that possesses no prior chain-of-thought capability. Stage 1 trains on foundational audio QA with the hybrid reward to foster basic reasoning patterns, while Stage 2 shifts to acoustically challenging boundary cases with an LLM-only reward for greater reasoning diversity. Audio-DeepThinker achieves state-of-the-art results on MMAR (74.0%), MMAU-test-mini (78.5%), and MMSU (77.26%), winning 1st Place in the Interspeech 2026 Audio Reasoning Challenge (Single Model Track). Interpretability analyses further reveal that RL training primarily reshapes upper-layer MoE gating mechanisms and that reasoning tokens crystallize progressively in the upper transformer layers, offering mechanistic insights into how audio reasoning emerges through exploration.
>
---
#### [new 014] MINT-Bench: A Comprehensive Multilingual Benchmark for Instruction-Following Text-to-Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MINT-Bench，一个用于指令跟随文本转语音的多语言基准。解决现有评估不足的问题，通过构建多维度数据和评估体系，推动可控、多语言TTS研究。**

- **链接: [https://arxiv.org/pdf/2604.17958](https://arxiv.org/pdf/2604.17958)**

> **作者:** Huakang Chen; Jingbin Hu; Liumeng Xue; Qirui Zhan; Wenhao Li; Guobin Ma; Hanke Xie; Dake Guo; Linhan Ma; Yuepeng Jiang; Bengu Wu; Pengyuan Xie; Chuan Xie; Qiang Zhang; Lei Xie
>
> **摘要:** Instruction-following text-to-speech (TTS) has emerged as an important capability for controllable and expressive speech generation, yet its evaluation remains underdeveloped due to limited benchmark coverage, weak diagnostic granularity, and insufficient multilingual support. We present \textbf{MINT-Bench}, a comprehensive multilingual benchmark for instruction-following TTS. MINT-Bench is built upon a hierarchical multi-axis taxonomy, a scalable multi-stage data construction pipeline, and a hierarchical hybrid evaluation protocol that jointly assesses content consistency, instruction following, and perceptual quality. Experiments across ten languages show that current systems remain far from solved: frontier commercial systems lead overall, while leading open-source models become highly competitive and can even outperform commercial counterparts in localized settings such as Chinese. The benchmark further reveals that harder compositional and paralinguistic controls remain major bottlenecks for current systems. We release MINT-Bench together with the data construction and evaluation toolkit to support future research on controllable, multilingual, and diagnostically grounded TTS evaluation. The leaderboard and demo are available at this https URL
>
---
#### [new 015] Incremental learning for audio classification with Hebbian Deep Neural Networks
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于音频分类任务，旨在解决持续学习中的知识遗忘问题。通过引入Hebbian学习和核可塑性方法，提升模型在增量学习中的稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2604.18270](https://arxiv.org/pdf/2604.18270)**

> **作者:** Riccardo Casciotti; Francesco De Santis; Alberto Antonietti; Annamaria Mesaros
>
> **备注:** ICASSP 2026
>
> **摘要:** The ability of humans for lifelong learning is an inspiration for deep learning methods and in particular for continual learning. In this work, we apply Hebbian learning, a biologically inspired learning process, to sound classification. We propose a kernel plasticity approach that selectively modulates network kernels during incremental learning, acting on selected kernels to learn new information and on others to retain previous knowledge. Using the ESC-50 dataset, the proposed method achieves 76.3% overall accuracy over five incremental steps, outperforming a baseline without kernel plasticity (68.7%) and demonstrating significantly greater stability across tasks.
>
---
#### [new 016] Prosody as Supervision: Bridging the Non-Verbal--Verbal for Multilingual Speech Emotion Recognition
- **分类: eess.AS**

- **简介: 该论文属于多语言语音情感识别任务，解决低资源环境下情感识别效果差的问题。通过非言语到言语的迁移学习，提出NOVA-ARC框架提升性能。**

- **链接: [https://arxiv.org/pdf/2604.17647](https://arxiv.org/pdf/2604.17647)**

> **作者:** Girish; Mohd Mujtaba Akhtar; Muskaan Singh
>
> **备注:** Accepted to ACL 2026 (main)
>
> **摘要:** In this work, we introduce a paralinguistic supervision paradigm for low-resource multilingual speech emotion recognition (LRM-SER) that leverages non-verbal vocalizations to exploit prosody-centric emotion cues. Unlike conventional SER systems that rely heavily on labeled verbal speech and suffer from poor cross-lingual transfer, our approach reformulates LRM-SER as non-verbal-to-verbal transfer, where supervision from a labeled non-verbal source domain is adapted to unlabeled verbal speech across multiple target languages. To this end, we propose NOVA ARC, a geometry-aware framework that models affective structure in the Poincaré ball, discretizes paralinguistic patterns via a hyperbolic vector-quantized prosody codebook, and captures emotion intensity through a hyperbolic emotion lens. For unsupervised adaptation, NOVA-ARC performs optimal transport based prototype alignment between source emotion prototypes and target utterances, inducing soft supervision for unlabeled speech while being stabilized through consistency regularization. Experiments show that NOVA-ARC delivers the strongest performance under both non-verbal-to-verbal adaptation and the complementary verbal-to-verbal transfer setting, consistently outperforming Euclidean counterparts and strong SSL baselines. To the best of our knowledge, this work is the first to move beyond verbal-speech-centric supervision by introducing a non-verbal-to-verbal transfer paradigm for SER.
>
---
#### [new 017] Anonymization, Not Elimination: Utility-Preserved Speech Anonymization
- **分类: eess.AS**

- **简介: 该论文属于语音匿名化任务，旨在保护隐私同时保持数据效用。解决现有方法降低语音数据质量的问题，提出两阶段框架，提升隐私保护并维持ASR、TTS等任务性能。**

- **链接: [https://arxiv.org/pdf/2604.17000](https://arxiv.org/pdf/2604.17000)**

> **作者:** Yunchong Xiao; Yuxiang Zhao; Ziyang Ma; Shuai Wang; Kai Yu; Jiachun Liao; Xie Chen
>
> **摘要:** The growing reliance on large-scale speech data has made privacy protection a critical concern. However, existing anonymization approaches often degrade data utility, for example by disrupting acoustic continuity or reducing vocal diversity, which compromises the value of speech data for downstream tasks such as Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Speech Emotion Recognition (SER). Current evaluation practices are also limited, as they mainly rely on direct testing of anonymized speech with pretrained models, providing only a partial view of utility. To address these issues, we propose a novel two-stage framework that protects both linguistic content and acoustic identity while maintaining usability. For content privacy, we employ a generative speech editing model to seamlessly replace personally identifiable information (PII), and for voice privacy, we introduce F3-VA, a flow-matching-based anonymization framework with a three-stage design that produces diverse and distinct anonymized speakers. To enable a more comprehensive assessment, we evaluate privacy using both acoustic- and content-based speaker verification metrics, and assess utility by training ASR, TTS, and SER models from scratch. Experimental results show that our framework achieves stronger privacy protection with minimal utility degradation compared to baselines from the VoicePrivacy Challenge, while the proposed evaluation protocol provides a more realistic reflection of the utility of anonymized speech under privacy protection.
>
---
#### [new 018] LLM-Codec: Neural Audio Codec Meets Language Model Objectives
- **分类: cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决神经音频编解码器与语言模型不匹配的问题。通过引入未来令牌预测和语义对齐机制，提升令牌可预测性并降低困惑度。**

- **链接: [https://arxiv.org/pdf/2604.17852](https://arxiv.org/pdf/2604.17852)**

> **作者:** Ho-Lam Chung; Yiming Chen; Hung-yi Lee
>
> **备注:** ACL2026 Finding
>
> **摘要:** Neural audio codecs are widely used as tokenizers for spoken language models, but they are optimized for waveform reconstruction rather than autoregressive prediction. This mismatch injects acoustically driven uncertainty into the discrete token space and increases language-model perplexity. We propose \ours, which augments codec training with language-model-facing objectives while keeping both codec and LLM architectures unchanged. \ours introduces (i) future token prediction with Medusa-style multi-step heads to encourage multi-step predictability, and (ii) semantic alignment that matches audio and text representations via a memory-bank contrastive loss. A differentiable Gumbel bridge enables end-to-end gradients from these objectives to the codec encoder. On SALMon speech coherence, token LMs trained on \ours reach 61.6% accuracy (+12.1 points over AUV) while reducing perplexity 35. On Codec-SUPERB-tiny, \ours improves speech Mel distance by 5.0% over AUV while simultaneously achieving the learnability gains, demonstrating that reconstruction fidelity and token predictability can be improved together.
>
---
#### [new 019] Deep Hierarchical Knowledge Loss for Fault Intensity Diagnosis
- **分类: eess.AS; cs.AI; cs.CV; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于故障强度诊断任务，解决类别间依赖关系被忽视的问题。提出深度分层知识损失框架，提升细微故障识别效果。**

- **链接: [https://arxiv.org/pdf/2604.16459](https://arxiv.org/pdf/2604.16459)**

> **作者:** Yu Sha; Shuiping Gou; Bo Liu; Haofan Lu; Ningtao Liu; Jiahui Fu; Horst Stoecker; Domagoj Vnucec; Nadine Wetzstein; Andreas Widl; Kai Zhou
>
> **备注:** The paper has been accepted by Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026)
>
> **摘要:** Fault intensity diagnosis (FID) plays a pivotal role in intelligent manufacturing while neglecting dependencies among target classes hinders its practical deployment. This paper introduces a novel and general framework with deep hierarchical knowledge loss (DHK) to achieve hierarchical consistent representation and prediction. We develop a novel hierarchical tree loss to enable a holistic mapping of same-attribute classes, leveraging tree-based positive and negative hierarchical knowledge constraints. We further design a focal hierarchical tree loss to enhance its extensibility and devise two adaptive weighting schemes based on tree height. In addition, we propose a group tree triplet loss with hierarchical dynamic margin by incorporating hierarchical group concepts and tree distance to model boundary structural knowledge across classes. The joint two losses significantly improve the recognition of subtle faults. Extensive experiments are performed on four real-world datasets from various industrial domains (three cavitation datasets from SAMSON AG and one publicly available dataset) for FID, all showing superior results and outperforming recent state-of-the-art FID methods.
>
---
#### [new 020] VIBE: Voice-Induced open-ended Bias Evaluation for Large Audio-Language Models via Real-World Speech
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音-语言模型公平性评估任务，旨在解决LALMs生成偏见问题。通过真实语音数据进行开放式任务评估，发现性别线索比口音更易引发分布偏移。**

- **链接: [https://arxiv.org/pdf/2604.17248](https://arxiv.org/pdf/2604.17248)**

> **作者:** Yi-Cheng Lin; Yusuke Hirota; Sung-Feng Huang; Hung-yi Lee
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Large Audio-Language Models (LALMs) are increasingly integrated into daily applications, yet their generative biases remain underexplored. Existing speech fairness benchmarks rely on synthetic speech and Multiple-Choice Questions (MCQs), both offering a fragmented view of fairness. We propose VIBE, a framework that evaluates generative bias through open-ended tasks such as personalized recommendations, using real-world human recordings. Unlike MCQs, our method allows stereotypical associations to manifest organically without predefined options, making it easily extensible to new tasks. Evaluating 11 state-of-the-art LALMs reveals systematic biases in realistic scenarios. We find that gender cues often trigger larger distributional shifts than accent cues, indicating that current LALMs reproduce social stereotypes.
>
---
#### [new 021] SAND: The Challenge on Speech Analysis for Neurodegenerative Disease Assessment
- **分类: eess.AS; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于ALS早期诊断任务，旨在通过语音分析解决神经退行性疾病评估问题。研究构建了标注数据集并发起SAND挑战，推动AI模型开发与评估。**

- **链接: [https://arxiv.org/pdf/2604.16445](https://arxiv.org/pdf/2604.16445)**

> **作者:** Giovanna Sannino; Ivanoe De Falco; Nadia Brancati; Laura Verde; Maria Frucci; Daniel Riccio; Vincenzo Bevilacqua; Antonio Di Marino; Lucia Aruta; Valentina Virginia Iuzzolino; Gianmaria Senerchia; Myriam Spisto; Raffaele Dubbioso
>
> **摘要:** Recent advances in Artificial Intelligence (AI) and the exploration of noninvasive, objective biomarkers, such as speech signals, have encouraged the development of algorithms to support the early diagnosis of neurodegenerative diseases, including Amyotrophic Lateral Sclerosis (ALS). Voice changes in subjects suffering from ALS typically manifest as progressive dysarthria, which is a prominent neurodegenerative symptom because it affects patients as the disease progresses. Since voice signals are complex data, the development and use of advanced AI techniques are fundamental to extracting distinctive patterns from them. Validating AI algorithms for ALS diagnosis and monitoring using voice signals is challenging, particularly due to the lack of annotated reference datasets. In this work, we present the outcome of a collaboration between a multidisciplinary team of clinicians and Machine Learning experts to create both a clinically annotated validation dataset and the "Speech Analysis for Neurodegenerative Diseases" (SAND) challenge based on it. Specifically, by analyzing voice disorders, the SAND challenge provides an opportunity to develop, test, and evaluate AI models for the automatic early identification and prediction of ALS disease progression.
>
---
#### [new 022] A state-space representation of the boundary integral equation for room acoustic modelling
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出一种基于边界积分方程的房间声学建模新框架，解决传统模型表示不足的问题。通过状态空间方法，构建了边界积分算子状态空间模型（BIOSS），实现多种等效声学表示。**

- **链接: [https://arxiv.org/pdf/2604.16970](https://arxiv.org/pdf/2604.16970)**

> **作者:** Randall Ali; Thomas Dietzen; Matteo Scerbo; Enzo De Sena; Toon van Waterschoot
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** We introduce a new framework for room acoustics modelling based on a state-space model of the boundary integral equation representing the sound field in a room. Whereas state-space models of linear time-invariant systems are traditionally constructed by means of a state vector and a 4-tuple of system matrices, the state-space representation introduced in this work consists of a state function representing the pressure distribution at the room boundary, and a 4-tuple of integral operators. We refer to this representation as a boundary integral operator state-space (BIOSS) model and provide a physical interpretation for each of the integral operators. As many mathematical operations on vectors and matrices translate to functions and operators, the BIOSS representation can be manipulated to obtain two transfer function representations, having either a feedback or a parallel feedforward structure. Consequently, various equivalent representations for room acoustics are obtained in the BIOSS framework, in the time or frequency domain, and in continuous or discrete space. We discuss two future directions for how the proposed framework can be fertile for research on room acoustics modelling. Firstly, we identify equivalences between the BIOSS framework and various existing room acoustics models (boundary element models, delay networks, geometric models), which may be used to establish relations between existing models and to develop novel room acoustics models. Secondly, we postulate on how concepts from state-space theory, such as observability, controllability, and state realization, can be used for developing new inference and control methods for room acoustics.
>
---
#### [new 023] AVRT: Audio-Visual Reasoning Transfer through Single-Modality Teachers
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于多模态推理任务，旨在解决多模态数据推理能力不足的问题。通过单模态教师模型生成高质量推理轨迹，并融合训练多模态模型，提升其在音频-视觉任务上的表现。**

- **链接: [https://arxiv.org/pdf/2604.16617](https://arxiv.org/pdf/2604.16617)**

> **作者:** Edson Araujo; Saurabhchand Bhati; M. Jehanzeb Mirza; Brian Kingsbury; Samuel Thomas; Rogerio Feris; James R. Glass; Hilde Kuehne
>
> **摘要:** Recent advances in reasoning models have shown remarkable progress in text-based domains, but transferring those capabilities to multimodal settings, e.g., to allow reasoning over audio-visual data, still remains a challenge, in part because of the limited availability of high-quality reasoning data in targeted multimodal combinations. To address this problem, we introduce AVRT, a novel framework that generates high-quality audio-visual reasoning traces from single-modality teacher models. We generate independent vision- and audio-reasoning traces via models specialized to reason over their respective modalities and merge the resulting traces with an LLM merger model. The resulting multimodal traces are used in a supervised fine-tuning (SFT) cold start to adapt the target model to audio-visual reasoning traces first, before training it in a second reinforcement learning stage on larger-scale data. Evaluated on seven audio-visual and audio benchmarks, our 3B and 7B parameter models achieve state-of-the-art results among models of comparable size including OmniBench and DailyOmni for audio-visual and MMAR for audio-only reasoning, showing that cross-modal training also transfers to single-modality tasks and establishing a new training pipeline for multimodal reasoning models.
>
---
#### [new 024] TeMuDance: Contrastive Alignment-Based Textual Control for Music-Driven Dance Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于音乐驱动舞蹈生成任务，旨在解决文本语义控制不足的问题。通过引入运动中心对齐框架和轻量文本控制分支，实现无需标注数据的文本控制生成。**

- **链接: [https://arxiv.org/pdf/2604.17005](https://arxiv.org/pdf/2604.17005)**

> **作者:** Xinran Liu; Diptesh Kanojia; Wenwu Wang; Zhenhua Feng
>
> **摘要:** Existing music-driven dance generation approaches have achieved strong realism and effective audio-motion alignment. However, they generally lack semantic controllability, making it difficult to guide specific movements through natural language descriptions. This limitation primarily stems from the absence of large-scale datasets that jointly align music, text, and motion for supervised learning of text-conditioned control. To address this challenge, we propose TeMuDance, a framework that enables text-based control for music-conditioned dance generation without requiring any manually annotated music-text-motion triplet dataset. TeMuDance introduces a motion-centred bridging paradigm that leverages motion as a shared semantic anchor to align disjoint music-dance and text-motion datasets within a unified embedding space, enabling cross-modal retrieval of missing modalities for end-to-end training. A lightweight text control branch is then trained on top of a frozen music-to-dance diffusion backbone, preserving rhythmic fidelity while enabling fine-grained semantic guidance. To further suppress noise inherent in the retrieved supervision, we design a dual-stream fine-tuning strategy with confidence-based filtering. We also propose a novel task-aligned metric that quantifies whether textual prompts induce the intended kinematic attributes under music conditioning. Extensive experiments demonstrate that TeMuDance achieves competitive dance quality while substantially improving text-conditioned control over existing methods.
>
---
#### [new 025] MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音翻译任务，解决非语言发声丢失问题。提出MoVE模型，通过混合专家架构提升情感表达的准确性与自然度。**

- **链接: [https://arxiv.org/pdf/2604.17435](https://arxiv.org/pdf/2604.17435)**

> **作者:** Szu-Chi Chen; I-Ning Tsai; Yi-Cheng Lin; Sung-Feng Huang; Hung-yi Lee
>
> **备注:** Submitted to Interspeech. Audio Demo and Dataset: this https URL
>
> **摘要:** Recent Speech-to-Speech Translation (S2ST) systems achieve strong semantic accuracy yet consistently strip away non-verbal vocalizations (NVs), such as laughter and crying that convey pragmatic intent, which severely limits real-world utility. We address this via three contributions. First, we propose a synthesis pipeline for building scalable expressive datasets to overcome the data scarcity limitation. Second, we propose MoVE, a Mixture-of-LoRA-Experts architecture with expressive-specialized adapters and a soft-weighting router that blends experts for capturing hybrid expressive states. Third, we show pretrained AudioLLMs enable striking data efficiency: 30 minutes of curated data is enough for strong performance. On English-Chinese S2ST, while comparing with strong baselines, MoVE reproduces target NVs in 76% of cases and achieves the highest human-rated naturalness and emotional fidelity among all compared systems, where existing S2ST systems preserve at most 14% of NVs.
>
---
#### [new 026] EchoChain: A Full-Duplex Benchmark for State-Update Reasoning Under Interruptions
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出EchoChain，用于评估语音助手在中断后的状态更新推理能力。针对实时语音交互中的状态更新问题，设计基准测试，发现并分析常见失败模式。**

- **链接: [https://arxiv.org/pdf/2604.16456](https://arxiv.org/pdf/2604.16456)**

> **作者:** Smit Nautambhai Modi; Gandharv Mahajan; Marc Wetter; Randall Welles
>
> **摘要:** Real-time voice assistants must revise task state when users interrupt mid-response, but existing spoken-dialog benchmarks largely evaluate turn-based interaction and miss this failure mode. We introduce EchoChain, a controlled benchmark for evaluating full-duplex state-update reasoning under mid-speech interruptions. EchoChain identifies three recurring failure patterns in post-interruption continuations: contextual inertia, interruption amnesia, and objective displacement. The benchmark generates scenario-driven conversations and injects interruptions at a standardized point relative to assistant speech onset, enabling controlled cross-model comparison. In a paired half-duplex control, total failures drop by 40.2% relative to interrupted runs, indicating that many errors are driven by state-update reasoning under interruption rather than task difficulty alone. Across evaluated real-time voice models, no system exceeds a 50% pass rate, showing substantial room for improvement in mid-generation state revision. EchoChain provides a reproducible benchmark for diagnosing state-update reasoning failures in full-duplex voice interaction.
>
---
#### [new 027] Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于音频大模型安全研究任务，解决良性微调导致安全对齐失效的问题。通过分析音频与文本在嵌入空间中的差异，揭示了音频模型的安全风险，并提出两种防御方法。**

- **链接: [https://arxiv.org/pdf/2604.16659](https://arxiv.org/pdf/2604.16659)**

> **作者:** Jaechul Roh; Amir Houmansadr
>
> **摘要:** Prior work shows that fine-tuning aligned models on benign data degrades safety in text and vision modalities, and that proximity to harmful content in representation space predicts which samples cause the most damage. However, existing analyses operate within a single, undifferentiated embedding space -- leaving open whether distinct input properties drive the vulnerability differently. Audio introduces a structurally richer problem: a benign sample can neighbor harmful content not only through what is said but through how it sounds, even when its words are entirely innocuous. We present the first systematic study of benign fine-tuning safety in Audio LLMs, evaluating three state-of-the-art models with a proximity-based filtering framework that selects benign audio by embedding-space distance to harmful content. By decomposing proximity into semantic, acoustic, and mixed axes using external reference encoders alongside each model's own internal encoder, we show that benign fine-tuning elevates Jailbreak Success Rate (JSR) from single digits to as high as 87.12%. Crucially, the dominant vulnerability axis and the relative risk of audio versus text fine-tuning are both architecture-conditioned -- determined by how each model's encoder and projector transform audio into the LLM's input space. We propose two defenses: filtering training data to maximize distance from harmful embeddings, and a textual system prompt at inference, both reducing JSR to near-zero without architectural modification. Our mechanistic analysis on two architectures reveals that fine-tuning selectively suppresses the late-layer refusal circuit while the frozen encoder preserves representations, and that even the suppression pattern is architecture-conditioned, mirroring the behavioral asymmetries across modalities. Safety degradation from benign fine-tuning is a qualitatively distinct risk in Audio LLMs.
>
---
#### [new 028] FLiP: Towards understanding and interpreting multimodal multilingual sentence embeddings
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出FLiP模型，用于理解多模态多语言句向量空间。解决如何从预训练句向量中恢复词汇内容的问题，通过实验验证其有效性并分析编码器的偏见。**

- **链接: [https://arxiv.org/pdf/2604.18109](https://arxiv.org/pdf/2604.18109)**

> **作者:** Santosh Kesiraju; Bolaji Yusuf; Šimon Sedláček; Oldřich Plchot; Petr Schwarz
>
> **备注:** Under review
>
> **摘要:** This paper presents factorized linear projection (FLiP) models for understanding pretrained sentence embedding spaces. We train FLiP models to recover the lexical content from multilingual (LaBSE), multimodal (SONAR) and API-based (Gemini) sentence embedding spaces in several high- and mid-resource languages. We show that FLiP can recall more than 75% of lexical content from the embeddings, significantly outperforming existing non-factorized baselines. Using this as a diagnostic tool, we uncover the modality and language biases across the selected sentence encoders and provide practitioners with intrinsic insights about the encoders without relying on conventional downstream evaluation tasks. Our implementation is public this https URL.
>
---
#### [new 029] A High-Accuracy Optical Music Recognition Method Based on Bottleneck Residual Convolutions
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于光学音乐识别任务，旨在将音乐乐谱图像转换为符号表示。通过结合残差瓶颈卷积和双向GRU序列建模，提升识别精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.16446](https://arxiv.org/pdf/2604.16446)**

> **作者:** Junwen Ma; Huhu Xue; Xingyuan Zhao; and Weicheng Fu
>
> **备注:** 2 figs, and 13 tables
>
> **摘要:** Optical Music Recognition (OMR) aims to convert printed or handwritten music score images into editable symbolic representations. This paper presents an end-to-end OMR framework that combines residual bottleneck convolutions with bidirectional gated recurrent unit (BiGRU)-based sequence modeling. A convolutional neural network with ResNet-v2-style residual bottleneck blocks and multi-scale dilated convolutions is used to extract features that encode both fine-grained symbol details and global staff-line structures. The extracted feature sequences are then fed into a BiGRU network to model temporal dependencies among musical symbols. The model is trained using the Connectionist Temporal Classification loss, enabling end-to-end prediction without explicit alignment annotations. Experimental results on the Camera-PrIMuS and PrIMuS datasets demonstrate the effectiveness of the proposed framework. On Camera-PrIMuS, the proposed method achieves a sequence error rate (SeER) of $7.52\%$ and a symbol error rate (SyER) of $0.45\%$, with pitch, type, and note accuracies of $99.33\%$, $99.60\%$, and $99.28\%$, respectively. The average training time is 1.74~s per epoch, demonstrating high computational efficiency while maintaining strong recognition performance. On PrIMuS, the method achieves a SeER of $8.11\%$ and a SyER of $0.49\%$, with pitch, type, and note accuracies of $99.27\%$, $99.58\%$, and $99.21\%$, respectively. A fine-grained error analysis further confirms the effectiveness of the proposed model.
>
---
#### [new 030] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音助手鲁棒性研究任务，旨在解决第三方干扰识别问题。通过构建数据集和评估框架，提升模型对语音中断的准确判断能力。**

- **链接: [https://arxiv.org/pdf/2604.17358](https://arxiv.org/pdf/2604.17358)**

> **作者:** Dongwook Lee; Eunwoo Song; Che Hyun Lee; Heeseung Kim; Sungroh Yoon
>
> **备注:** ACL 2026 main conference
>
> **摘要:** While recent Spoken Language Models (SLMs) have been actively deployed in real-world scenarios, they lack the capability to discern Third-Party Interruptions (TPI) from the primary user's ongoing flow, leaving them vulnerable to contextual failures. To bridge this gap, we introduce TPI-Train, a dataset of 88K instances designed with speaker-aware hard negatives to enforce acoustic cue prioritization for interruption handling, and TPI-Bench, a comprehensive evaluation framework designed to rigorously measure the interruption-handling strategy and precise speaker discrimination in deceptive contexts. Experiments demonstrate that our dataset design mitigates semantic shortcut learning-a critical pitfall where models exploit semantic context while neglecting acoustic signals essential for discerning speaker changes. We believe our work establishes a foundational resource for overcoming text-dominated unimodal reliance in SLMs, paving the way for more robust multi-party spoken interaction. The code for the framework is publicly available at this https URL
>
---
## 更新

#### [replaced 001] Generalizable Prompt Tuning for Audio-Language Models via Semantic Expansion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-语言模型任务，解决提示调优在ALMs中的泛化能力问题。提出SEPT框架，通过语义扩展增强嵌入空间结构，提升泛化性能。**

- **链接: [https://arxiv.org/pdf/2601.20867](https://arxiv.org/pdf/2601.20867)**

> **作者:** Jaehyuk Jang; Wonjun Lee; Kangwook Ko; Changick Kim
>
> **备注:** ACL 2026 findings
>
> **摘要:** Prompt tuning has achieved remarkable progress in vision-language models (VLMs) and is recently being adopted for audio-language models (ALMs). However, its generalization ability in ALMs remains largely underexplored. We observe that conventional prompt tuning for ALMs also suffers from the Base-New Tradeoff, and we identify that this issue stems from the disrupted semantic structure of the embedding space. To address this issue, we propose Semantically Expanded Prompt Tuning (SEPT)-a plug-and-play framework that explicitly regularizes the prompt embedding space by incorporating semantic neighbors generated by large language models. SEPT introduces a novel semantic expansion loss with margin constraints that promote intra-class compactness and inter-class separability, thereby enhancing the semantic structure of the prompt embedding space. For comprehensive evaluation, we establish the first benchmark setup for prompt generalization in ALMs, covering both base-to-new generalization and cross-dataset transferability. Extensive experiments demonstrate that SEPT consistently improves generalization performance across multiple prompt tuning baselines, while maintaining computational cost during inference.
>
---
#### [replaced 002] VoxSafeBench: Not Just What Is Said, but Who, How, and Where
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音语言模型的安全评估任务，旨在解决语音场景下的安全、公平与隐私问题。工作包括构建VoxSafeBench基准，分层评估语音相关风险。**

- **链接: [https://arxiv.org/pdf/2604.14548](https://arxiv.org/pdf/2604.14548)**

> **作者:** Yuxiang Wang; Hongyu Liu; Yijiang Xu; Qinke Ni; Li Wang; Wan Lin; Kunyu Feng; Dekun Chen; Xu Tan; Lei Wang; Jie Shi; Zhizheng Wu
>
> **摘要:** As speech language models (SLMs) transition from personal devices into shared, multi-user environments, their responses must account for far more than the words alone. Who is speaking, how they sound, and where the conversation takes place can each turn an otherwise benign request into one that is unsafe, unfair, or privacy-violating. Existing benchmarks, however, largely focus on basic audio comprehension, study individual risks in isolation, or conflate content that is inherently harmful with content that only becomes problematic due to its acoustic context. We introduce VoxSafeBench, among the first benchmarks to jointly evaluate social alignment in SLMs across three dimensions: safety, fairness, and privacy. VoxSafeBench adopts a Two-Tier design: Tier1 evaluates content-centric risks using matched text and audio inputs, while Tier2 targets audio-conditioned risks in which the transcript is benign but the appropriate response hinges on the speaker, paralinguistic cues, or the surrounding environment. To validate Tier2, we include intermediate perception probes and confirm that frontier SLMs can successfully detect these acoustic cues yet still fail to act on them appropriately. Across 22 tasks with bilingual coverage, we find that safeguards appearing robust on text often degrade in speech: safety awareness drops for speaker- and scene-conditioned risks, fairness erodes when demographic differences are conveyed vocally, and privacy protections falter when contextual cues arrive acoustically. Together, these results expose a pervasive speech grounding gap: current SLMs frequently recognize the relevant social norm in text but fail to apply it when the decisive cue must be grounded in speech. Code and data are publicly available at: this https URL
>
---
#### [replaced 003] Reverberation-based Features for Sound Event Localization and Detection with Distance Estimation
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于声事件定位与检测（SELD）任务，旨在解决3D空间中声源距离估计问题。提出基于混响的特征，提升距离估计性能。**

- **链接: [https://arxiv.org/pdf/2504.08644](https://arxiv.org/pdf/2504.08644)**

> **作者:** Davide Berghi; Philip J. B. Jackson
>
> **摘要:** Sound event localization and detection (SELD) involves predicting active sound event classes over time while estimating their positions. The localization subtask in SELD is usually treated as a direction of arrival estimation problem, ignoring source distance. Only recently, SELD was extended to 3D by incorporating distance estimation, enabling the prediction of sound event positions in 3D space (3D SELD). However, existing methods lack input features specifically designed for distance estimation. We address this gap by introducing two novel reverberation-based feature formats: one using the direct-to-reverberant ratio (DRR) and another leveraging signal autocorrelation to capture early reflections. We extensively evaluate and benchmark these features on the STARSS23 dataset, combining them with established SELD features for sound event detection (SED) and direction-of-arrival estimation (DOAE), and testing across different network architectures. Our proposed features, applicable to both FOA and MIC formats, achieve state-of-the-art distance estimation, enhancing overall 3D SELD performance.
>
---
#### [replaced 004] SonicRadiation: A Hybrid Numerical Solution for Sound Radiation without Ghost Cells
- **分类: cs.SD; cs.GR; math.NA**

- **简介: 该论文属于声学仿真任务，解决复杂边界下声辐射模拟的误差问题。提出SonicRadiation方法，结合TDBEM与FDTD，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2508.08775](https://arxiv.org/pdf/2508.08775)**

> **作者:** Xutong Jin; Fei Zhu; Guoping Wang; Sheng Li
>
> **备注:** 11 pages
>
> **摘要:** Interactive synthesis of physical sound effects is crucial in digital media production. Sound radiation simulation, a key component of physically based sound synthesis, has posed challenges in the context of complex object boundaries. Previous methods, such as ghost cell-based finite-difference time-domain (FDTD) wave solver, have struggled to address these challenges, leading to large errors and failures in complex boundaries because of the limitation of ghost cells. We present SonicRadiation, a hybrid numerical solution capable of handling complex and dynamic object boundaries in sound radiation simulation without relying on ghost cells. We derive a consistent formulation to connect the physical quantities on grid cells in FDTD with the boundary elements in the time-domain boundary element method (TDBEM). Hereby, we propose a boundary grid synchronization strategy to seamlessly integrate TDBEM with FDTD while maintaining high numerical accuracy. Our method holds both advantages from the accuracy of TDBEM for the near-field and the efficiency of FDTD for the far-field. Experimental results demonstrate the superiority of our method in sound radiation simulation over previous approaches in terms of accuracy and efficiency, particularly in complex scenes, further validating its effectiveness.
>
---
#### [replaced 005] End-to-end Listen, Look, Speak and Act
- **分类: cs.AI; cs.CL; cs.CV; cs.RO; eess.AS**

- **简介: 该论文提出ELLSA模型，解决多模态交互任务，实现语音、视觉、文本和动作的同步感知与生成，支持自然的人机互动。**

- **链接: [https://arxiv.org/pdf/2510.16756](https://arxiv.org/pdf/2510.16756)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Chao Zhang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released at this https URL.
>
---
#### [replaced 006] ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于AI生成音乐检测任务，旨在解决AI音乐识别问题。提出ArtifactNet框架，通过分析音频编码器留下的物理残差进行检测。**

- **链接: [https://arxiv.org/pdf/2604.16254](https://arxiv.org/pdf/2604.16254)**

> **作者:** Heewon Oh
>
> **备注:** v2: Added SONICS 3-way (n=23,288), OOD taxonomy, benchmark coverage table, baseline reproduction appendix; toned-down claims; reframed discussion as asymmetric defender advantage. 8 pages, 6 figs, 12 tables
>
> **摘要:** We present ArtifactNet, a lightweight framework that detects AI-generated music by reframing the problem as forensic physics -- extracting and analyzing the physical artifacts that neural audio codecs inevitably imprint on generated audio. A bounded-mask UNet (ArtifactUNet, 3.6M parameters) extracts codec residuals from magnitude spectrograms, which are then decomposed via HPSS into 7-channel forensic features for classification by a compact CNN (0.4M parameters; 4.0M total). We introduce ArtifactBench, a multi-generator evaluation benchmark comprising 6,183 tracks (4,383 AI from 22 generators and 1,800 real from 6 diverse sources). Each track is tagged with bench_origin for fair zero-shot evaluation. On the unseen test partition (n=2,263), ArtifactNet achieves F1 = 0.9829 with FPR = 1.49%, compared to CLAM (F1 = 0.7576, FPR = 69.26%) and SpecTTTra (F1 = 0.7713, FPR = 19.43%) evaluated under identical conditions with published checkpoints. Codec-aware training (4-way WAV/MP3/AAC/Opus augmentation) further reduces cross-codec probability drift by 83% (Delta = 0.95 -> 0.16), resolving the primary codec-invariance failure mode. These results establish forensic physics -- direct extraction of codec-level artifacts -- as a more generalizable and parameter-efficient paradigm for AI music detection than representation learning, using 49x fewer parameters than CLAM and 4.8x fewer than SpecTTTra.
>
---
#### [replaced 007] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
- **分类: eess.AS**

- **简介: 该论文属于语音-文本预训练任务，旨在解决细粒度说话风格建模难题。通过构建FCaps数据集和提出CLSP模型，实现多粒度对齐的语音-文本表示学习。**

- **链接: [https://arxiv.org/pdf/2601.03065](https://arxiv.org/pdf/2601.03065)**

> **作者:** Yifan Yang; Bing Han; Hui Wang; Wei Wang; Ziyang Ma; Long Zhou; Zengrui Jin; Guanrou Yang; Tianrui Wang; Xu Tan; Xie Chen
>
> **备注:** Accepted in ACL 2026 Main
>
> **摘要:** Modeling fine-grained speaking styles remains challenging for language-speech representation pre-training, as existing speech-text models are typically trained with coarse captions or task-specific supervision, and scalable fine-grained style annotations are unavailable. We present FCaps, a large-scale dataset with fine-grained free-text style descriptions, encompassing 47k hours of speech and 19M fine-grained captions annotated via a novel end-to-end pipeline that directly grounds detailed captions in audio, thereby avoiding the error propagation caused by LLM-based rewriting in existing cascaded pipelines. Evaluations using LLM-as-a-judge demonstrate that our annotations surpass existing cascaded annotations in terms of correctness, coverage, and naturalness. Building on FCaps, we propose CLSP, a contrastive language-speech pre-trained model that integrates global and fine-grained supervision, enabling unified representations across multiple granularities. Extensive experiments demonstrate that CLSP learns fine-grained and multi-granular speech-text representations that perform reliably across global and fine-grained speech-text retrieval, zero-shot paralinguistic classification, and speech style similarity scoring, with strong alignment to human judgments. Code and dataset are publicly available at this https URL.
>
---
#### [replaced 008] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于医疗语音分析任务，旨在从语音中检测疾病。针对标签稀缺和标注主观的问题，提出一种多层级半监督学习框架，有效利用未标记数据。**

- **链接: [https://arxiv.org/pdf/2601.04744](https://arxiv.org/pdf/2601.04744)**

> **作者:** Xingyuan Li; Mengyue Wu
>
> **备注:** Accepted for publication as a Findings paper at the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Detecting medical conditions from speech acoustics is fundamentally a weakly-supervised learning problem: a single, often noisy, session-level label must be linked to nuanced patterns within a long, complex audio recording. This task is further hampered by severe data scarcity and the subjective nature of clinical annotations. While semi-supervised learning (SSL) offers a viable path to leverage unlabeled data, existing audio methods often fail to address the core challenge that pathological traits are not uniformly expressed in a patient's speech. We propose a novel, audio-only SSL framework that explicitly models this hierarchy by jointly learning from frame-level, segment-level, and session-level representations within unsegmented clinical dialogues. Our end-to-end approach dynamically aggregates these multi-granularity features and generates high-quality pseudo-labels to efficiently utilize unlabeled data. Extensive experiments show the framework is model-agnostic, robust across languages and conditions, and highly data-efficient-achieving, for instance, 90% of fully-supervised performance using only 11 labeled samples. This work provides a principled approach to learning from weak, far-end supervision in medical speech analysis. The code is available at this https URL.
>
---
#### [replaced 009] ClariCodec: Optimising Neural Speech Codes for 200bps Communication using Reinforcement Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决超低比特率下语音可懂性下降的问题。通过强化学习优化量化策略，提升语音识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.14654](https://arxiv.org/pdf/2604.14654)**

> **作者:** Junyi Wang; Chi Zhang; Jing Qian; Haifeng Luo; Hao Wang; Zengrui Jin; Chao Zhang
>
> **备注:** Withdrawn by the authors due to incomplete bitrate accounting in the ILN-based pipeline. The side information introduced by ILN was not fully included in the effective bitrate, making the reported 200 bps results and related comparisons unreliable. The withdrawal does not concern the paper's core RL-based methodological idea. A corrected version may follow
>
> **摘要:** In bandwidth-constrained communication such as satellite and underwater channels, speech must often be transmitted at ultra-low bitrates where intelligibility is the primary objective. At such extreme compression levels, codecs trained with acoustic reconstruction losses tend to allocate bits to perceptual detail, leading to substantial degradation in word error rate (WER). This paper proposes ClariCodec, a neural speech codec operating at 200 bit per second (bps) that reformulates quantisation as a stochastic policy, enabling reinforcement learning (RL)-based optimisation of intelligibility. Specifically, the encoder is fine-tuned using WER-driven rewards while the acoustic reconstruction pipeline remains frozen. Even without RL, ClariCodec achieves 3.68% WER on the LibriSpeech test-clean set at 200 bps, already competitive with codecs operating at higher bitrates. Further RL fine-tuning reduces WER to 3.20% on test-clean and 8.93% on test-other, corresponding to a 13% relative reduction while preserving perceptual quality.
>
---
#### [replaced 010] emg2speech: Synthesizing speech from electromyography using self-supervised speech models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在将肌电信号直接转换为语音。通过自监督语音模型，实现无需显式建模的EMG到语音生成。**

- **链接: [https://arxiv.org/pdf/2510.23969](https://arxiv.org/pdf/2510.23969)**

> **作者:** Harshavardhana T. Gowda; Daniel C. Comstock; Lee M. Miller
>
> **摘要:** We present a neuromuscular speech interface that translates electromyographic (EMG) signals recorded from orofacial muscles during speech articulation directly into audio. We find that self-supervised speech (S3) representations are strongly linearly related to the electrical power of muscle activity: a simple linear mapping predicts EMG power from S3 representations with a correlation of r = 0.85. In addition, EMG power vectors associated with distinct articulatory gestures form structured, separable clusters. Together, these observations suggest that S3 models implicitly encode articulatory mechanisms, as reflected in EMG activity. Leveraging this structure, we map EMG signals into the S3 representation space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory modeling or vocoder training. We demonstrate this system with a participant with amyotrophic lateral sclerosis (ALS), converting orofacial EMG recorded while she silently articulated speech into audio.
>
---
#### [replaced 011] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决零样本TTS中风格控制不足的问题。通过引入DCFG和LoRA技术，实现连续且相对的风格控制，提升合成语音的多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.03632](https://arxiv.org/pdf/2601.03632)**

> **作者:** Haitao Li; Chunxiang Jin; Chenglin Li; Wenhao Guan; Zhengxing Huang; Xie Chen
>
> **备注:** ACL 2026
>
> **摘要:** Zero-shot text-to-speech models can clone a speaker's timbre from a short reference audio, but they also strongly inherit the speaking style present in the reference. As a result, synthesizing speech with a desired style often requires carefully selecting reference audio, which is impractical when only limited or mismatched references are available. While recent controllable TTS methods attempt to address this issue, they typically rely on absolute style targets and discrete textual prompts, and therefore do not support continuous and reference-relative style control. We propose ReStyle-TTS, a framework that enables continuous and reference-relative style control in zero-shot TTS. Our key insight is that effective style control requires first reducing the model's implicit dependence on reference style before introducing explicit control mechanisms. To this end, we introduce Decoupled Classifier-Free Guidance (DCFG), which independently controls text and reference guidance, reducing reliance on reference style while preserving text fidelity. On top of this, we apply style-specific LoRAs together with Orthogonal LoRA Fusion to enable continuous and disentangled multi-attribute control, and introduce a Timbre Consistency Optimization module to mitigate timbre drift caused by weakened reference guidance. Experiments show that ReStyle-TTS enables user-friendly, continuous, and relative control over pitch, energy, and multiple emotions while maintaining intelligibility and speaker timbre, and performs robustly in challenging mismatched reference-target style scenarios.
>
---
#### [replaced 012] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于自动语音识别任务，解决域转移下伪标签偏差问题。通过模型参数差异构建修正向量，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2510.08047](https://arxiv.org/pdf/2510.08047)**

> **作者:** Yi-Cheng Lin; Yu-Hsuan Li Liang; Hsuan Su; Tzu-Quan Lin; Shang-Tse Chen; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Robust ASR under domain shift is crucial because real-world systems encounter unseen accents and domains with limited labeled data. Although pseudo-labeling offers a practical workaround, it often introduces systematic, accent-specific errors that filtering fails to fix. We ask: How can we correct these recurring biases without target ground truth? We propose a simple parameter-space correction: in a source domain containing both real and pseudo-labeled data, two ASR models are fine-tuned from the same initialization, one on ground-truth labels and the other on pseudo-labels, and their weight difference forms a correction vector that captures pseudo-label biases. When applied to a pseudo-labeled target model, this vector enhances recognition, achieving up to a 35% relative Word Error Rate (WER) reduction on AfriSpeech-200 across ten African accents with the Whisper tiny model.
>
---
#### [replaced 013] Audio-Visual Speech Enhancement: Architectural Design and Deployment Strategies
- **分类: cs.SD; eess.SP**

- **简介: 该论文属于音频-视觉语音增强任务，解决实时交互多媒体服务中的网络延迟与计算延迟问题。设计并部署了基于云边协同的AVSE系统，分析不同网络条件下的性能表现。**

- **链接: [https://arxiv.org/pdf/2508.08468](https://arxiv.org/pdf/2508.08468)**

> **作者:** Anis Hamadouche; Haifeng Luo; Mathini Sellathurai; Amir Hussain; Tharm Ratnarajah
>
> **摘要:** Real-time audio-visual speech enhancement (AVSE) is a key enabler for immersive and interactive multimedia services, yet its performance is tightly constrained by network latency, uplink capacity, and computational delay. This paper presents the design, deployment, and evaluation of a complete cloud-edge-assisted AVSE system operating over a public 5G edge network. The system integrates CNN-based acoustic enhancement and OpenCV-based facial feature extraction with an LSTM fusion network to preserve temporal coherence, and is deployed on a Vodafone-compatible AWS Wavelength edge cloud. Through extensive stress testing, we analyze end-to-end performance under varying network load and adaptive multimedia profiles. Results show that compute placement at the network edge is critical for meeting real-time coherence constraints, and that uplink capacity is often the dominant bottleneck for interactive AVSE services. Only 5G and wired Ethernet consistently satisfied the required communication delay bound for uncompressed audio-video chunks, while aggressive compression reduced payload sizes by up to 80% with negligible perceptual degradation, enabling robust operation under constrained conditions. We further demonstrate a fundamental trade-off between processing latency and enhancement quality, where reduced model complexity lowers delay but degrades reconstruction performance in low-SNR scenarios. Our findings indicate that public 5G edge environments can sustain real-time, interactive AVSE workloads when network and compute resources are carefully orchestrated, although performance margins remain tighter than in dedicated infrastructures. The architectural insights derived from this study provide practical guidelines for the design of delay-sensitive multimedia and perceptual enhancement services on emerging 5G edge-cloud platforms.
>
---
#### [replaced 014] FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于视频到音频生成任务，旨在解决多事件场景下时间控制不足的问题。提出FoleyDirector框架，通过结构化脚本实现精准时间引导，提升可控性与音质。**

- **链接: [https://arxiv.org/pdf/2603.19857](https://arxiv.org/pdf/2603.19857)**

> **作者:** You Li; Dewei Zhou; Fan Ma; Fu Li; Dongliang He; Yi Yang
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026, 18 pages
>
> **摘要:** Recent Video-to-Audio (V2A) methods have achieved remarkable progress, enabling the synthesis of realistic, high-quality audio. However, they struggle with fine-grained temporal control in multi-event scenarios or when visual cues are insufficient, such as small regions, off-screen sounds, or occluded or partially visible objects. In this paper, we propose FoleyDirector, a framework that, for the first time, enables precise temporal guidance in DiT-based V2A generation while preserving the base model's audio quality and allowing seamless switching between V2A generation and temporally controlled synthesis. FoleyDirector introduces Structured Temporal Scripts (STS), a set of captions corresponding to short temporal segments, to provide richer temporal information. These features are integrated via the Script-Guided Temporal Fusion Module, which employs Temporal Script Attention to fuse STS features coherently. To handle complex multi-event scenarios, we further propose Bi-Frame Sound Synthesis, enabling parallel in-frame and out-of-frame audio generation and improving controllability. To support training and evaluation, we construct the DirectorSound dataset and introduce VGGSoundDirector and DirectorBench. Experiments demonstrate that FoleyDirector substantially enhances temporal controllability while maintaining high audio fidelity, empowering users to act as Foley directors and advancing V2A toward more expressive and controllable generation.
>
---
#### [replaced 015] StereoFoley: Object-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 论文提出StereoFoley，解决视频到立体音频生成任务中的对象感知问题。通过合成数据和模型优化，实现语义、时间同步和空间准确的立体声生成。**

- **链接: [https://arxiv.org/pdf/2509.18272](https://arxiv.org/pdf/2509.18272)**

> **作者:** Tornike Karchkhadze; Kuan-Lin Chen; Mojtaba Heydari; Robert Henzel; Alessandro Toso; Mehrez Souden; Joshua Atkins
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** We present StereoFoley, a video-to-audio generation framework that produces semantically aligned, temporally synchronized, and spatially accurate stereo sound at 48 kHz. While recent generative video-to-audio models achieve strong semantic and temporal fidelity, they largely remain limited to mono or fail to deliver object-aware stereo imaging, constrained by the lack of professionally mixed, spatially accurate video-to-audio datasets. First, we develop a base model that generates stereo audio from video, achieving performance on par with state-of-the-art V2A models in both semantic accuracy and synchronization. Next, to overcome dataset limitations, we introduce a synthetic data generation pipeline that combines video analysis, object tracking, and audio synthesis with dynamic panning and distance-based loudness controls, enabling spatially accurate object-aware sound. Finally, we fine-tune the base model on this synthetic dataset, yielding clear object-audio correspondence. Since no established metrics exist, we introduce a stereo object-awareness metric and report it alongside a human listening study; the two evaluations exhibit consistent trends. This work establishes the first end-to-end framework for stereo object-aware video-to-audio generation, addressing a critical gap in the field.
>
---
#### [replaced 016] State Space Models for Bioacoustics: A Comparative Evaluation with Transformers
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于生物声学任务，旨在解决音频分类与检测问题。通过提出BioMamba模型，对比Transformer架构，验证其在计算效率上的优势。**

- **链接: [https://arxiv.org/pdf/2512.03563](https://arxiv.org/pdf/2512.03563)**

> **作者:** Chengyu Tang; Sanjeev Baskiyar
>
> **摘要:** In this study, we evaluate the efficacy of the Mamba architecture bioacoustics by introducing BioMamba, a Mamba-based audio representation model for wildlife sounds. We pre-train a BioMamba using self-supervised learning on a large audio corpus and evaluate it on the BEANS benchmark across diverse classification and detection tasks. Compared to the state-of-the-art Transformer-based model (AVES), BioMamba achieves comparable performance while significantly reducing VRAM consumption. Our results demonstrate Mamba's potential as a computationally efficient alternative for real-world environmental monitoring.
>
---
#### [replaced 017] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决生成音频在时间控制和语音清晰度上的不足。通过多任务学习和渐进式扩散模型，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2510.08878](https://arxiv.org/pdf/2510.08878)**

> **作者:** Yuxuan Jiang; Zehua Chen; Zeqian Ju; Yusheng Dai; Weibei Dou; Jun Zhu
>
> **备注:** Accepted at ACL 2026 Main
>
> **摘要:** Text-to-audio (TTA) generation with fine-grained control signals, e.g., precise timing control or intelligible speech content, has been explored in recent works. However, constrained by data scarcity, their generation performance at scale is still compromised. In this study, we recast controllable TTA generation as a multi-task learning problem and introduce a progressive diffusion modeling approach, ControlAudio. Our method adeptly fits distributions conditioned on more fine-grained information, including text, timing, and phoneme features, through a step-by-step strategy. First, we propose a data construction method spanning both annotation and simulation, augmenting condition information in the sequence of text, timing, and phoneme. Second, at the model training stage, we pretrain a diffusion transformer (DiT) on large-scale text-audio pairs, achieving scalable TTA generation, and then incrementally integrate the timing and phoneme features with unified semantic representations, expanding controllability. Finally, at the inference stage, we propose progressively guided generation, which sequentially emphasizes more fine-grained information, aligning inherently with the coarse-to-fine sampling nature of DiT. Extensive experiments show that ControlAudio achieves state-of-the-art performance in terms of temporal accuracy and speech clarity, significantly outperforming existing methods on both objective and subjective evaluations. Demo samples are available at: this https URL.
>
---
#### [replaced 018] Multi-Source Position and Direction-of-Arrival Estimation Based on Euclidean Distance Matrices
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于多声源定位与波达方向估计任务，旨在解决传统方法计算复杂的问题。通过利用欧几里得距离矩阵特性，提出新方法提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2510.02556](https://arxiv.org/pdf/2510.02556)**

> **作者:** Klaus Brümann; Simon Doclo
>
> **备注:** 13 pages, 4 figures, submitted to IEEE Transactions on Audio, Speech and Language Processing (awaiting review)
>
> **摘要:** A popular method to estimate the positions or directions-of-arrival (DOAs) of multiple sound sources using an array of microphones is based on steered-response power (SRP) beamforming. For a three-dimensional scenario, SRP-based methods require joint optimization of three continuous variables for position estimation or two continuous variables for DOA estimation, which can be computationally expensive when high localization accuracy is desired. In this paper, we propose novel methods for multi-source position and DOA estimation by exploiting properties of Euclidean distance matrices (EDMs) and their respective Gram matrices. All methods require estimated time-differences of arrival (TDOAs) between the microphones. In the proposed multi-source position estimation method, only a single continuous variable per source, representing the distance to a reference microphone, needs to be optimized. For each source, the optimal distance variable and set of candidate TDOA estimates are determined by minimizing a cost function defined using the eigenvalues of the Gram matrix. The estimated relative source positions are then mapped to absolute source positions by solving an orthogonal Procrustes problem. The proposed multi-source DOA estimation method eliminates the need for continuous variable optimization. The optimal set of candidate TDOA estimates is determined by minimizing a cost function defined using the eigenvalues of a rank-reduced Gram matrix. For two sources in a noisy and reverberant environment, experimental results for different source and microphone configurations with six microphones show that the proposed EDM-based method consistently outperforms the SRP-based method in terms of position and DOA estimation accuracy and run time.
>
---
#### [replaced 019] Leveraging Large Language Models for Sarcastic Speech Annotation in Sarcasm Detection
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于讽刺检测任务，旨在解决语音中讽刺识别数据不足的问题。通过大语言模型生成讽刺标注数据，并构建大规模数据集PodSarc，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2506.00955](https://arxiv.org/pdf/2506.00955)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **备注:** Interspeech 2025; Project page: this https URL
>
> **摘要:** Sarcasm fundamentally alters meaning through tone and context, yet detecting it in speech remains a challenge due to data scarcity. In addition, existing detection systems often rely on multimodal data, limiting their applicability in contexts where only speech is available. To address this, we propose an annotation pipeline that leverages large language models (LLMs) to generate a sarcasm dataset. Using a publicly available sarcasm-focused podcast, we employ GPT-4o and LLaMA 3 for initial sarcasm annotations, followed by human verification to resolve disagreements. We validate this approach by comparing annotation quality and detection performance on a publicly available sarcasm dataset using a collaborative gating architecture. Finally, we introduce PodSarc, a large-scale sarcastic speech dataset created through this pipeline. The detection model achieves a 73.63% F1 score, demonstrating the dataset's potential as a benchmark for sarcasm detection research.
>
---
#### [replaced 020] RSA-Bench: Benchmarking Audio Large Models in Real-World Acoustic Scenarios
- **分类: cs.SD**

- **简介: 该论文属于音频模型鲁棒性评估任务，旨在解决真实声学场景下模型性能下降的问题。通过构建多场景噪声数据集，测试模型在不同干扰下的表现，揭示模型在高阶推理中的弱点。**

- **链接: [https://arxiv.org/pdf/2601.10384](https://arxiv.org/pdf/2601.10384)**

> **作者:** Yibo Zhang; Liang Lin; Kaiwen Luo; Shilinlu Yan; Jin Wang; Yaoqi Guo; Yitian Chen; Yalan Qin; Zhenhong Zhou; Kun Wang; Li Sun
>
> **摘要:** While Audio Large Models (ALMs) have achieved remarkable proficiency, their robustness remains brittle in real-world deployment. Existing evaluations largely rely on synthetic Gaussian noise or simplistic single-source interference, failing to capture the intricate, multi-layered acoustic dynamics -- or ``Acoustic Ecology'' -- that characterize authentic physical environments. To bridge this ecological gap, we introduce \textbf{RSA-Bench}, a comprehensive robustness benchmark designed to stress-test ALLMs through high-fidelity auditory scene simulations. Unlike traditional methods, we construct evaluation samples by naturally superimposing diverse environmental soundscapes -- spanning \textit{Pasture}, \textit{Extreme Weather}, \textit{Classroom}, and \textit{Outdoors} -- onto clean speech signals across a spectrum of interference intensities. By evaluating models on six core tasks ranging from fundamental perception to complex reasoning, our study unveils three macro-level insights: \textbf{(I) The Perception-Cognition Gap:} Models maintain relative resilience in low-level recognition but suffer a \textbf{functional collapse} in high-order reasoning tasks under stress; \textbf{(II) Scenario Sensitivity:} ``Vocal-like'' interference (e.g., background laughter) proves significantly more destructive than mechanical noise, challenging the model's auditory attention mechanisms; and \textbf{(III) The Denoising Paradox:} Standard speech enhancement often exacerbates performance degradation, as ALLMs prove highly sensitive to the semantic distortions introduced by denoising artifacts.
>
---
#### [replaced 021] Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于多模态情感分析任务，解决缺失模态场景下的情感预测问题。提出知识迁移网络和跨模态注意力机制，以重建缺失音频特征并提升情感识别效果。**

- **链接: [https://arxiv.org/pdf/2401.10747](https://arxiv.org/pdf/2401.10747)**

> **作者:** Weide Liu; Huijing Zhan
>
> **摘要:** Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most existing research assume that all modalities are available during both training and testing, which makes their algorithms susceptible to the missing-modality scenarios. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio features. Moreover, we develop a cross-modality attention mechanism to maximize the information extracted from the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baseline methods and achieve comparable results to the previous methods with complete multi-modality supervision.
>
---
#### [replaced 022] Mechanisms of Multimodal Synchronization: Insights from Decoder-Based Video-Text-to-Speech Synthesis
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文研究视频-文本到语音合成任务，解决多模态同步问题。通过统一解码器模型，分析模态信息互补、位置编码策略及顺序对同步性能的影响，提出时间同步度量方法。**

- **链接: [https://arxiv.org/pdf/2411.17690](https://arxiv.org/pdf/2411.17690)**

> **作者:** Akshita Gupta; Tatiana Likhomanenko; Karren Dai Yang; Richard He Bai; Zakaria Aldeneh; Navdeep Jaitly
>
> **备注:** 30 pages, Decoder-only model, Speech Synthesis
>
> **摘要:** Unified decoder-only transformers have shown promise for multimodal generation, yet the mechanisms by which they synchronize modalities with heterogeneous sampling rates remain underexplored. We investigate these mechanisms through video-text-to-speech (VTTS) synthesis-a controlled task requiring fine-grained temporal alignment between sparse text, video, and continuous speech. Using a unified decoder-only transformer, dubbed Visatronic, trained on VoxCeleb2, we study: (i) how modalities contribute complementary information, (ii) how positional encoding strategies enable synchronization across heterogeneous rates, (iii) how modality ordering shapes the trade-off between in-domain performance and cross-domain transfer, (iv) how phoneme-level synchronization metrics provide diagnostic insight into per-phoneme timing errors. Our findings reveal that both "global sequential indexing'' (unique position IDs across modalities) and "co-temporal ordered indexing'' (identical IDs for temporally corresponding tokens) achieve strong synchronization performance, with co-temporal ordered indexing providing a simple mechanism without explicit timestamp metadata. Both text and video contribute complementary signals: text ensures intelligibility while video provides temporal cues and emotional expressiveness. Modality ordering reveals a consistent trade-off: video-first ordering achieves stronger in-domain performance while text-first ordering generalizes more robustly to unseen domains. Our findings also reveal, that diverse large-scale training enables transferable synchronization strategies. To enable fine-grained analysis, we also introduce TimeSync, a phoneme-level metric that reveals temporal misalignments overlooked by frame-level metrics. These insights establish VTTS as a valuable testbed for understanding temporal synchronization in unified multimodal decoders.
>
---
#### [replaced 023] Closing the Modality Reasoning Gap for Speech Large Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音大语言模型任务，旨在解决语音输入推理性能弱于文本的问题。提出TARS框架，通过强化学习对齐语音与文本轨迹，提升跨模态推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05543](https://arxiv.org/pdf/2601.05543)**

> **作者:** Chaoren Wang; Heng Lu; Xueyao Zhang; Shujie Liu; Yan Lu; Jinyu Li; Zhizheng Wu
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Although Speech Large Language Models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs.
>
---
#### [replaced 024] Non-invasive electromyographic speech neuroprosthesis: a geometric perspective
- **分类: eess.AS**

- **简介: 该论文属于语音神经假体任务，旨在恢复丧失言语能力患者的通信能力。通过非侵入式EMG信号直接转换为文本，解决传统方法依赖音频的问题。**

- **链接: [https://arxiv.org/pdf/2502.05762](https://arxiv.org/pdf/2502.05762)**

> **作者:** Harshavardhana T. Gowda; Lee M. Miller
>
> **摘要:** We present a neuromuscular speech interface that translates silently voiced articulations directly into text. We record surface electromyographic (EMG) signals from multiple articulatory sites on the face and neck as participants silently articulate speech, enabling direct EMG-to-text translation. Such an interface has the potential to restore communication for individuals who have lost the ability to produce intelligible speech due to laryngectomy, neuromuscular disease, stroke, or trauma-induced damage (e.g., radiotherapy toxicity) to the speech articulators. Prior work has largely focused on mapping EMG collected during audible articulation to time-aligned audio targets or transferring these targets to silent EMG recordings, which inherently requires audio and limits applicability to patients who can no longer speak. In contrast, we propose an efficient representation of high-dimensional EMG signals and demonstrate direct sequence-to-sequence EMG-to-text conversion at the phonemic level without relying on time-aligned audio.
>
---
#### [replaced 025] Audio-Cogito: Towards Deep Audio Reasoning in Large Audio Language Models
- **分类: eess.AS**

- **简介: 该论文属于音频推理任务，旨在解决大音频语言模型中推理能力不足的问题。通过构建高质量数据集并采用自蒸馏策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12527](https://arxiv.org/pdf/2604.12527)**

> **作者:** Longhao Li; Hongjie Chen; Zehan Li; Qihan Hu; Jian Kang; Jie Li; Lei Xie; Yongxiang Li
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advances in reasoning models have driven significant progress in text and multimodal domains, yet audio reasoning remains relatively limited. Only a few Large Audio Language Models (LALMs) incorporate explicit Chain-of-Thought (CoT) reasoning, and their capabilities are often inconsistent and insufficient for complex tasks. To bridge this gap, we introduce Audio-Cogito, a fully open-source solution for deep audio reasoning. We develop Cogito-pipe for high-quality audio reasoning data curation, producing 545k reasoning samples that will be released after review. Based on this dataset, we adopt a self-distillation strategy for model fine-tuning. Experiments on the MMAR benchmark, the only audio benchmark evaluating the CoT process, show that our model achieves the best performance among open-source models and matches or surpasses certain closed-source models in specific metrics. Our approach also ranks among the top-tier systems in the Interspeech 2026 Audio Reasoning Challenge.
>
---
#### [replaced 026] From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音代理评估任务，旨在解决现有基准未充分覆盖主动干预的问题。提出ProVoice-Bench框架，包含四项新任务，以评估语音代理的主动性。**

- **链接: [https://arxiv.org/pdf/2604.15037](https://arxiv.org/pdf/2604.15037)**

> **作者:** Ke Xu; Yuhao Wang; Yu Wang
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents.
>
---
#### [replaced 027] Musical Score Understanding Benchmark: Evaluating Large Language Models' Comprehension of Complete Musical Scores
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出MSU-Bench基准，用于评估大语言模型对完整乐谱的理解能力。任务属于多模态理解，解决模型在音高、节奏等音乐要素上的推理不足问题。工作包括构建基准数据集并测试多种模型表现。**

- **链接: [https://arxiv.org/pdf/2511.20697](https://arxiv.org/pdf/2511.20697)**

> **作者:** Congren Dai; Yue Yang; Krinos Li; Huichi Zhou; Shijie Liang; Bo Zhang; Enyang Liu; Ge Jin; Hongran An; Haosen Zhang; Peiyuan Jing; Kinhei Lee; Z henxuan Zhang; Xiaobing Li; Maosong Sun
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Understanding complete musical scores entails integrated reasoning over pitch, rhythm, harmony, and large-scale structure, yet the ability of Large Language Models and Vision--Language Models to interpret full musical notation remains insufficiently examined. We introduce Musical Score Understanding Benchmark (MSU-Bench), a human-curated benchmark for score-level musical understanding across textual (ABC notation) and visual (PDF) modalities. MSU-Bench contains 1,800 generative question-answer pairs from works by Bach, Beethoven, Chopin, Debussy, and others, organised into four levels of increasing difficulty, ranging from onset information to texture and form. Evaluations of more than fifteen state-of-the-art models, in both zero-shot and fine-tuned settings, reveal pronounced modality gaps, unstable level-wise performance, and challenges in maintaining multilevel correctness. Fine-tuning substantially improves results across modalities while preserving general knowledge, positioning MSU-Bench as a robust foundation for future research in multimodal reasoning. The benchmark and code are available at this https URL.
>
---
#### [replaced 028] Modelling Emotions is an Elusive Pursuit in Affective Computing
- **分类: eess.AS**

- **简介: 论文属于情感计算领域，探讨情绪建模的挑战。指出分类情绪标签限制了应用，提出需采用连续维度定义以提升准确性与实用性。**

- **链接: [https://arxiv.org/pdf/2603.23017](https://arxiv.org/pdf/2603.23017)**

> **作者:** Anders Rolighed Larsen; Sneha Das; Nicole Nadine Lønfeldt; Paula Petcu; Line Clemmensen
>
> **摘要:** Affective computing - combining sensor technology, machine learning, and psychology - have been studied for over three decades and is employed in AI-powered technologies to enhance emotional awareness in AI systems, and detect symptoms of mental health disorders such as anxiety and depression. However, the uncertainty in such systems remains high, and the application areas are limited by categorical definitions of emotions and emotional concepts. This paper argues that categorical emotion labels obscure emotional nuance in affective computing, and therefore continuous dimensional definitions are needed to advance the field, increase application usefulness, and lower uncertainties.
>
---
#### [replaced 029] MimicLM: Zero-Shot Voice Imitation through Autoregressive Modeling of Pseudo-Parallel Speech Corpora
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出MimicLM，解决语音模仿任务中的数据稀缺问题，通过合成语音训练并保留真实录音作为目标，提升语音模仿质量。**

- **链接: [https://arxiv.org/pdf/2604.11552](https://arxiv.org/pdf/2604.11552)**

> **作者:** Tao Feng; Yuxiang Wang; Yuancheng Wang; Xueyao Zhang; Dekun Chen; Chaoren Wang; Xun Guan; Zhizheng Wu
>
> **摘要:** Voice imitation aims to transform source speech to match a reference speaker's timbre and speaking style while preserving linguistic content. A straightforward approach is to train on triplets of (source, reference, target), where source and target share the same content but target matches the reference's voice characteristics, yet such data is extremely scarce. Existing approaches either employ carefully designed disentanglement architectures to bypass this data scarcity or leverage external systems to synthesize pseudo-parallel training data. However, the former requires intricate model design, and the latter faces a quality ceiling when synthetic speech is used as training targets. To address these limitations, we propose MimicLM, which takes a novel approach by using synthetic speech as training sources while retaining real recordings as targets. This design enables the model to learn directly from real speech distributions, breaking the synthetic quality ceiling. Building on this data construction approach, we incorporate interleaved text-audio modeling to guide the generation of content-accurate speech and apply post-training with preference alignment to mitigate the inherent distributional mismatch when training on synthetic data. Experiments demonstrate that MimicLM achieves superior voice imitation quality with a simple yet effective architecture, significantly outperforming existing methods in naturalness while maintaining competitive similarity scores across speaker identity, accent, and emotion dimensions.
>
---
#### [replaced 030] FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统中的说话人切换检测任务，旨在解决实时全双工通信中的低延迟与鲁棒性问题。提出FastTurn框架，结合声学与语义线索，提升检测准确性与响应速度。**

- **链接: [https://arxiv.org/pdf/2604.01897](https://arxiv.org/pdf/2604.01897)**

> **作者:** Chengyou Wang; Hongfei Xue; Chunjiang He; Jingbin Hu; Shuiyuan Wang; Bo Wu; Yuyu Ji; Jimeng Zheng; Ruofei Chen; Zhou Zhu; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
>
---
