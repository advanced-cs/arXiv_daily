# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] IntMeanFlow: Few-step Speech Generation with Integral Velocity Distillation
- **分类: cs.SD**

- **简介: 该论文属于文本到语音合成任务，旨在解决生成速度慢和内存开销大的问题。论文提出IntMeanFlow框架，通过积分速度蒸馏减少迭代步骤和GPU内存使用，并引入O3S算法优化采样步数，实现高质量快速语音生成。**

- **链接: [http://arxiv.org/pdf/2510.07979v1](http://arxiv.org/pdf/2510.07979v1)**

> **作者:** Wei Wang; Rong Cao; Yi Guo; Zhengyang Chen; Kuan Chen; Yuanyuan Huo
>
> **摘要:** Flow-based generative models have greatly improved text-to-speech (TTS) synthesis quality, but inference speed remains limited by the iterative sampling process and multiple function evaluations (NFE). The recent MeanFlow model accelerates generation by modeling average velocity instead of instantaneous velocity. However, its direct application to TTS encounters challenges, including GPU memory overhead from Jacobian-vector products (JVP) and training instability due to self-bootstrap processes. To address these issues, we introduce IntMeanFlow, a framework for few-step speech generation with integral velocity distillation. By approximating average velocity with the teacher's instantaneous velocity over a temporal interval, IntMeanFlow eliminates the need for JVPs and self-bootstrap, improving stability and reducing GPU memory usage. We also propose the Optimal Step Sampling Search (O3S) algorithm, which identifies the model-specific optimal sampling steps, improving speech synthesis without additional inference overhead. Experiments show that IntMeanFlow achieves 1-NFE inference for token-to-spectrogram and 3-NFE for text-to-spectrogram tasks while maintaining high-quality synthesis. Demo samples are available at https://vvwangvv.github.io/intmeanflow.
>
---
#### [new 002] Personality-Enhanced Multimodal Depression Detection in the Elderly
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于多模态抑郁检测任务，旨在解决老年人抑郁检测中忽略个性特征的问题。作者提出一种融合个性特征的多模态模型，结合音频与视频特征，并设计交互模块捕捉个性与多模态特征的关系，显著提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2510.08004v1](http://arxiv.org/pdf/2510.08004v1)**

> **作者:** Honghong Wang; Jing Deng; Rong Zheng
>
> **备注:** 6 pages,2 figures,accepted by ACM Multimedia Asia 2025
>
> **摘要:** This paper presents our solution to the Multimodal Personality-aware Depression Detection (MPDD) challenge at ACM MM 2025. We propose a multimodal depression detection model in the Elderly that incorporates personality characteristics. We introduce a multi-feature fusion approach based on a co-attention mechanism to effectively integrate LLDs, MFCCs, and Wav2Vec features in the audio modality. For the video modality, we combine representations extracted from OpenFace, ResNet, and DenseNet to construct a comprehensive visual feature set. Recognizing the critical role of personality in depression detection, we design an interaction module that captures the relationships between personality traits and multimodal features. Experimental results from the MPDD Elderly Depression Detection track demonstrate that our method significantly enhances performance, providing valuable insights for future research in multimodal depression detection among elderly populations.
>
---
#### [new 003] Attribution-by-design: Ensuring Inference-Time Provenance in Generative Music Systems
- **分类: cs.SD; cs.AI; cs.HC**

- **简介: 该论文属于音乐版权与AI伦理任务，旨在解决AI生成音乐中创作者权益保障不足的问题。论文提出“归因设计”框架，区分训练与推理阶段的归因方式，强调推理时归因以实现透明、可验证的版税分配，确保艺术家在AI创作中的权益。**

- **链接: [http://arxiv.org/pdf/2510.08062v1](http://arxiv.org/pdf/2510.08062v1)**

> **作者:** Fabio Morreale; Wiebke Hutiri; Joan Serrà; Alice Xiang; Yuki Mitsufuji
>
> **摘要:** The rise of AI-generated music is diluting royalty pools and revealing structural flaws in existing remuneration frameworks, challenging the well-established artist compensation systems in the music industry. Existing compensation solutions, such as piecemeal licensing agreements, lack scalability and technical rigour, while current data attribution mechanisms provide only uncertain estimates and are rarely implemented in practice. This paper introduces a framework for a generative music infrastructure centred on direct attribution, transparent royalty distribution, and granular control for artists and rights' holders. We distinguish ontologically between the training set and the inference set, which allows us to propose two complementary forms of attribution: training-time attribution and inference-time attribution. We here favour inference-time attribution, as it enables direct, verifiable compensation whenever an artist's catalogue is used to condition a generated output. Besides, users benefit from the ability to condition generations on specific songs and receive transparent information about attribution and permitted usage. Our approach offers an ethical and practical solution to the pressing need for robust compensation mechanisms in the era of AI-generated music, ensuring that provenance and fairness are embedded at the core of generative systems.
>
---
#### [new 004] Leveraging Whisper Embeddings for Audio-based Lyrics Matching
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频歌词匹配任务，旨在解决现有方法缺乏可复现性与统一基准的问题。作者提出了WEALY，利用Whisper解码器嵌入实现歌词匹配，建立了可复现的基线，并探索了多模态扩展。实验表明其性能与现有最优方法相当，同时提供了消融实验与分析，为未来研究提供了可靠基准。**

- **链接: [http://arxiv.org/pdf/2510.08176v1](http://arxiv.org/pdf/2510.08176v1)**

> **作者:** Eleonora Mancini; Joan Serrà; Paolo Torroni; Yuki Mitsufuji
>
> **摘要:** Audio-based lyrics matching can be an appealing alternative to other content-based retrieval approaches, but existing methods often suffer from limited reproducibility and inconsistent baselines. In this work, we introduce WEALY, a fully reproducible pipeline that leverages Whisper decoder embeddings for lyrics matching tasks. WEALY establishes robust and transparent baselines, while also exploring multimodal extensions that integrate textual and acoustic features. Through extensive experiments on standard datasets, we demonstrate that WEALY achieves a performance comparable to state-of-the-art methods that lack reproducibility. In addition, we provide ablation studies and analyses on language robustness, loss functions, and embedding strategies. This work contributes a reliable benchmark for future research, and underscores the potential of speech technologies for music information retrieval tasks.
>
---
#### [new 005] ACMID: Automatic Curation of Musical Instrument Dataset for 7-Stem Music Source Separation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在解决训练数据不足与标签不准确问题。作者通过网络爬取大量数据，利用预训练音频编码器构建分类器，自动清洗并整理出高质量的7类乐器数据集ACMID-Cleaned。实验表明，该数据集提升了模型性能，验证了清洗方法的有效性与数据集的价值。**

- **链接: [http://arxiv.org/pdf/2510.07840v1](http://arxiv.org/pdf/2510.07840v1)**

> **作者:** Ji Yu; Yang shuo; Xu Yuetonghui; Liu Mengmei; Ji Qiang; Han Zerui
>
> **摘要:** Most current music source separation (MSS) methods rely on supervised learning, limited by training data quantity and quality. Though web-crawling can bring abundant data, platform-level track labeling often causes metadata mismatches, impeding accurate "audio-label" pair acquisition. To address this, we present ACMID: a dataset for MSS generated through web crawling of extensive raw data, followed by automatic cleaning via an instrument classifier built on a pre-trained audio encoder that filters and aggregates clean segments of target instruments from the crawled tracks, resulting in the refined ACMID-Cleaned dataset. Leveraging abundant data, we expand the conventional classification from 4-stem (Vocal/Bass/Drums/Others) to 7-stem (Piano/Drums/Bass/Acoustic Guitar/Electric Guitar/Strings/Wind-Brass), enabling high granularity MSS systems. Experiments on SOTA MSS model demonstrates two key results: (i) MSS model trained with ACMID-Cleaned achieved a 2.39dB improvement in SDR performance compared to that with ACMID-Uncleaned, demostrating the effectiveness of our data cleaning procedure; (ii) incorporating ACMID-Cleaned to training enhances MSS model's average performance by 1.16dB, confirming the value of our dataset. Our data crawling code, cleaning model code and weights are available at: https://github.com/scottishfold0621/ACMID.
>
---
#### [new 006] Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于视频到音频生成任务，旨在解决模型生成无视觉来源声音的“插入幻觉”问题。作者提出了评估框架与新指标，并设计了无需训练的后处理方法PFC，有效降低幻觉发生率与持续时间，提升生成可靠性。**

- **链接: [http://arxiv.org/pdf/2510.08078v1](http://arxiv.org/pdf/2510.08078v1)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **摘要:** Video-to-Audio generation has made remarkable strides in automatically synthesizing sound for video. However, existing evaluation metrics, which focus on semantic and temporal alignment, overlook a critical failure mode: models often generate acoustic events, particularly speech and music, that have no corresponding visual source. We term this phenomenon Insertion Hallucination and identify it as a systemic risk driven by dataset biases, such as the prevalence of off-screen sounds, that remains completely undetected by current metrics. To address this challenge, we first develop a systematic evaluation framework that employs a majority-voting ensemble of multiple audio event detectors. We also introduce two novel metrics to quantify the prevalence and severity of this issue: IH@vid (the fraction of videos with hallucinations) and IH@dur (the fraction of hallucinated duration). Building on this, we propose Posterior Feature Correction, a novel training-free inference-time method that mitigates IH. PFC operates in a two-pass process: it first generates an initial audio output to detect hallucinated segments, and then regenerates the audio after masking the corresponding video features at those timestamps. Experiments on several mainstream V2A benchmarks first reveal that state-of-the-art models suffer from severe IH. In contrast, our PFC method reduces both the prevalence and duration of hallucinations by over 50\% on average, without degrading, and in some cases even improving, conventional metrics for audio quality and temporal synchronization. Our work is the first to formally define, systematically measure, and effectively mitigate Insertion Hallucination, paving the way for more reliable and faithful V2A models.
>
---
#### [new 007] INFER : Learning Implicit Neural Frequency Response Fields for Confined Car Cabin
- **分类: cs.SD**

- **简介: 该论文属于神经声学建模任务，旨在解决车内声学建模不准确的问题。现有方法依赖手动调参、硬件测试，难以应对频率选择性和动态变化。论文提出INFER，通过频域神经网络联合建模声源与接收位置，引入频响场学习、感知监督和物理约束，提升了车内声场建模精度。**

- **链接: [http://arxiv.org/pdf/2510.07442v1](http://arxiv.org/pdf/2510.07442v1)**

> **作者:** Harshvardhan C. Takawale; Nirupam Roy; Phil Brown
>
> **摘要:** Accurate modeling of spatial acoustics is critical for immersive and intelligible audio in confined, resonant environments such as car cabins. Current tuning methods are manual, hardware-intensive, and static, failing to account for frequency selective behaviors and dynamic changes like passenger presence or seat adjustments. To address this issue, we propose INFER: Implicit Neural Frequency Response fields, a frequency-domain neural framework that is jointly conditioned on source and receiver positions, orientations to directly learn complex-valued frequency response fields inside confined, resonant environments like car cabins. We introduce three key innovations over current neural acoustic modeling methods: (1) novel end-to-end frequency-domain forward model that directly learns the frequency response field and frequency-specific attenuation in 3D space; (2) perceptual and hardware-aware spectral supervision that emphasizes critical auditory frequency bands and deemphasizes unstable crossover regions; and (3) a physics-based Kramers-Kronig consistency constraint that regularizes frequency-dependent attenuation and delay. We evaluate our method over real-world data collected in multiple car cabins. Our approach significantly outperforms time- and hybrid-domain baselines on both simulated and real-world automotive datasets, cutting average magnitude and phase reconstruction errors by over 39% and 51%, respectively. INFER sets a new state-of-the-art for neural acoustic modeling in automotive spaces
>
---
#### [new 008] IsoSignVid2Aud: Sign Language Video to Audio Conversion without Text Intermediaries
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于跨模态翻译任务，旨在解决手语视频到语音的直接转换问题。现有方法依赖文本中介，存在延迟和级联错误。为此，作者提出IsoSignVid2Aud框架，采用I3D特征提取、特征转换网络和音频生成流程，并引入NMS算法实现非连续手语序列的时序检测，提升了实时性和准确性。**

- **链接: [http://arxiv.org/pdf/2510.07837v1](http://arxiv.org/pdf/2510.07837v1)**

> **作者:** Harsh Kavediya; Vighnesh Nayak; Bheeshm Sharma; Balamurugan Palaniappan
>
> **备注:** Accepted in AIML-Systems-2025
>
> **摘要:** Sign language to spoken language audio translation is important to connect the hearing- and speech-challenged humans with others. We consider sign language videos with isolated sign sequences rather than continuous grammatical signing. Such videos are useful in educational applications and sign prompt interfaces. Towards this, we propose IsoSignVid2Aud, a novel end-to-end framework that translates sign language videos with a sequence of possibly non-grammatic continuous signs to speech without requiring intermediate text representation, providing immediate communication benefits while avoiding the latency and cascading errors inherent in multi-stage translation systems. Our approach combines an I3D-based feature extraction module with a specialized feature transformation network and an audio generation pipeline, utilizing a novel Non-Maximal Suppression (NMS) algorithm for the temporal detection of signs in non-grammatic continuous sequences. Experimental results demonstrate competitive performance on ASL-Citizen-1500 and WLASL-100 datasets with Top-1 accuracies of 72.01\% and 78.67\%, respectively, and audio quality metrics (PESQ: 2.67, STOI: 0.73) indicating intelligible speech output. Code is available at: https://github.com/BheeshmSharma/IsoSignVid2Aud_AIMLsystems-2025.
>
---
#### [new 009] AV-EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Omni-modal LLMS with Audio-visual Cues
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于多模态情感推理任务，旨在解决当前大语言模型在结合音频和视觉线索进行情感推理方面缺乏系统评估的问题。作者构建了AV-EMO-Reasoning基准，通过合成和真实场景的多模态数据，评估模型在不同情感指标下的表现，推动更自然的人机交互。**

- **链接: [http://arxiv.org/pdf/2510.07355v1](http://arxiv.org/pdf/2510.07355v1)**

> **作者:** Krish Patel; Dingkun Zhou; Ajay Kankipati; Akshaj Gupta; Zeyi Austin Li; Mohul Shukla; Vibhor Narang; Sara Kofman; Zongli Ye; Grace Wang; Xiaoyu Shi; Tingle Li; Guan-Ting Lin; Kan Jen Cheng; Huang-Cheng Chou; Jiachen Lian; Gopala Anumanchipalli
>
> **摘要:** Emotions conveyed through voice and face shape engagement and context in human-AI interaction. Despite rapid progress in omni-modal large language models (LLMs), the holistic evaluation of emotional reasoning with audiovisual cues remains limited. To address this gap, we introduce AV-EMO-Reasoning, a benchmark designed to systematically assess emotional coherence in LLMs. The framework leverages a curated, single- and multi-turn synthetic audiovisual corpus with a real-world set and is assessed under continuous, categorical, and perceptual metrics. Experiments with leading LLMs show that visual cues reliably improve emotional coherence over audio-only baselines. Moreover, LLMs can leverage audio-visual cues to generate more emotion-aware speech. Models exhibit complementary strengths across metric families, indicating that automatic scores capture facets distinct from perceptual judgments. By releasing a systematic evaluation benchmark, AV-EMO-Reasoning offers a reproducible standard for evaluating emotion-aware dialogue and advances toward more natural, adaptive human-AI interaction.
>
---
#### [new 010] MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion via Mean Flows
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音转换任务，旨在解决零样本、流式语音转换中音色迁移与语音质量间的平衡问题。提出MeanVC方法，结合扩散模型与自回归策略，通过平均流回归实现高效推理，减少参数量，提升转换质量与效率。**

- **链接: [http://arxiv.org/pdf/2510.08392v1](http://arxiv.org/pdf/2510.08392v1)**

> **作者:** Guobin Ma; Jixun Yao; Ziqian Ning; Yuepeng Jiang; Lingxin Xiong; Lei Xie; Pengcheng Zhu
>
> **摘要:** Zero-shot voice conversion (VC) aims to transfer timbre from a source speaker to any unseen target speaker while preserving linguistic content. Growing application scenarios demand models with streaming inference capabilities. This has created a pressing need for models that are simultaneously fast, lightweight, and high-fidelity. However, existing streaming methods typically rely on either autoregressive (AR) or non-autoregressive (NAR) frameworks, which either require large parameter sizes to achieve strong performance or struggle to generalize to unseen speakers. In this study, we propose MeanVC, a lightweight and streaming zero-shot VC approach. MeanVC introduces a diffusion transformer with a chunk-wise autoregressive denoising strategy, combining the strengths of both AR and NAR paradigms for efficient streaming processing. By introducing mean flows, MeanVC regresses the average velocity field during training, enabling zero-shot VC with superior speech quality and speaker similarity in a single sampling step by directly mapping from the start to the endpoint of the flow trajectory. Additionally, we incorporate diffusion adversarial post-training to mitigate over-smoothing and further enhance speech quality. Experimental results demonstrate that MeanVC significantly outperforms existing zero-shot streaming VC systems, achieving superior conversion quality with higher efficiency and significantly fewer parameters. Audio demos and code are publicly available at https://aslp-lab.github.io/MeanVC.
>
---
#### [new 011] Audio-Visual Separation with Hierarchical Fusion and Representation Alignment
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于自监督音视频分离任务，旨在解决如何有效融合多模态信息以提升音频分离效果的问题。作者提出了一种层次化融合策略，结合中层与后期融合，并引入表示对齐方法，使音频特征与预训练模型对齐，从而缩小模态差异，提升分离效果。**

- **链接: [http://arxiv.org/pdf/2510.07326v1](http://arxiv.org/pdf/2510.07326v1)**

> **作者:** Han Hu; Dongheng Lin; Qiming Huang; Yuqi Hou; Hyung Jin Chang; Jianbo Jiao
>
> **摘要:** Self-supervised audio-visual source separation leverages natural correlations between audio and vision modalities to separate mixed audio signals. In this work, we first systematically analyse the performance of existing multimodal fusion methods for audio-visual separation task, demonstrating that the performance of different fusion strategies is closely linked to the characteristics of the sound: middle fusion is better suited for handling short, transient sounds, while late fusion is more effective for capturing sustained and harmonically rich sounds. We thus propose a hierarchical fusion strategy that effectively integrates both fusion stages. In addition, training can be made easier by incorporating high-quality external audio representations, rather than relying solely on the audio branch to learn them independently. To explore this, we propose a representation alignment approach that aligns the latent features of the audio encoder with embeddings extracted from pre-trained audio models. Extensive experiments on MUSIC, MUSIC-21 and VGGSound datasets demonstrate that our approach achieves state-of-the-art results, surpassing existing methods under the self-supervised setting. We further analyse the impact of representation alignment on audio features, showing that it reduces modality gap between the audio and visual modalities.
>
---
## 更新

#### [replaced 001] Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.16622v2](http://arxiv.org/pdf/2509.16622v2)**

> **作者:** Mengqi Wang; Zhan Liu; Zengrui Jin; Guangzhi Sun; Chao Zhang; Philip C. Woodland
>
> **摘要:** Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements.
>
---
#### [replaced 002] Evaluating Sound Similarity Metrics for Differentiable, Iterative Sound-Matching
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.22628v2](http://arxiv.org/pdf/2506.22628v2)**

> **作者:** Amir Salimi; Abram Hindle; Osmar R. Zaiane
>
> **摘要:** Manual sound design with a synthesizer is inherently iterative: an artist compares the synthesized output to a mental target, adjusts parameters, and repeats until satisfied. Iterative sound-matching automates this workflow by continually programming a synthesizer under the guidance of a loss function (or similarity measure) toward a target sound. Prior comparisons of loss functions have typically favored one metric over another, but only within narrow settings: limited synthesis methods, few loss types, often without blind listening tests. This leaves open the question of whether a universally optimal loss exists, or the choice of loss remains a creative decision conditioned on the synthesis method and the sound designer's preference. We propose differentiable iterative sound-matching as the natural extension of the available literature, since it combines the manual approach to sound design with modern advances in machine learning. To analyze the variability of loss function performance across synthesizers, we implemented a mix of four novel and established differentiable loss functions, and paired them with differentiable subtractive, additive, and AM synthesizers. For each of the sixteen synthesizer--loss combinations, we ran 300 randomized sound-matching trials. Performance was measured using parameter differences, spectrogram-distance metrics, and manually assigned listening scores. We observed a moderate level of consistency among the three performance measures. Our post-hoc analysis shows that the loss function performance is highly dependent on the synthesizer. These findings underscore the value of expanding the scope of sound-matching experiments and developing new similarity metrics tailored to specific synthesis techniques rather than pursuing one-size-fits-all solutions.
>
---
#### [replaced 003] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.06961v2](http://arxiv.org/pdf/2510.06961v2)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Nithin Koluguri; Piotr Żelasko; Somshubra Majumdar; Adel Moumen; Sanchit Gandhi
>
> **备注:** Submitted to ICASSP 2026; Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard ; Code: https://github.com/huggingface/open_asr_leaderboard
>
> **摘要:** Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including dedicated multilingual and long-form tracks. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation.
>
---
#### [replaced 004] BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response
- **分类: cs.CV; cs.AI; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.06019v4](http://arxiv.org/pdf/2501.06019v4)**

> **作者:** Hongruixuan Chen; Jian Song; Olivier Dietrich; Clifford Broni-Bediako; Weihao Xuan; Junjue Wang; Xinlei Shao; Yimin Wei; Junshi Xia; Cuiling Lan; Konrad Schindler; Naoto Yokoya
>
> **摘要:** Disaster events occur around the world and cause significant damage to human life and property. Earth observation (EO) data enables rapid and comprehensive building damage assessment (BDA), an essential capability in the aftermath of a disaster to reduce human casualties and to inform disaster relief efforts. Recent research focuses on the development of AI models to achieve accurate mapping of unseen disaster events, mostly using optical EO data. However, solutions based on optical data are limited to clear skies and daylight hours, preventing a prompt response to disasters. Integrating multimodal (MM) EO data, particularly the combination of optical and SAR imagery, makes it possible to provide all-weather, day-and-night disaster responses. Despite this potential, the development of robust multimodal AI models has been constrained by the lack of suitable benchmark datasets. In this paper, we present a BDA dataset using veRy-hIGH-resoluTion optical and SAR imagery (BRIGHT) to support AI-based all-weather disaster response. To the best of our knowledge, BRIGHT is the first open-access, globally distributed, event-diverse MM dataset specifically curated to support AI-based disaster response. It covers five types of natural disasters and two types of man-made disasters across 14 regions worldwide, with a particular focus on developing countries where external assistance is most needed. The optical and SAR imagery in BRIGHT, with a spatial resolution between 0.3-1 meters, provides detailed representations of individual buildings, making it ideal for precise BDA. In our experiments, we have tested seven advanced AI models trained with our BRIGHT to validate the transferability and robustness. The dataset and code are available at https://github.com/ChenHongruixuan/BRIGHT. BRIGHT also serves as the official dataset for the 2025 IEEE GRSS Data Fusion Contest.
>
---
#### [replaced 005] TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation
- **分类: cs.IR; cs.AI; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09685v4](http://arxiv.org/pdf/2509.09685v4)**

> **作者:** Keunwoo Choi; Seungheon Doh; Juhan Nam
>
> **备注:** 2025-10-08: updating the stat table with the latest numbers. updated the abstract per the latest license terms
>
> **摘要:** We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In the proposed pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are released at https://talkpl.ai/talkplaydata2.
>
---
#### [replaced 006] SeamlessEdit: Background Noise Aware Zero-Shot Speech Editing with in-Context Enhancement
- **分类: eess.AS; cs.SD; 68T45; I.2.7; H.5.5**

- **链接: [http://arxiv.org/pdf/2505.14066v2](http://arxiv.org/pdf/2505.14066v2)**

> **作者:** Kuan-Yu Chen; Jeng-Lin Li; Jian-Jiun Ding
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** With the fast development of zero-shot text-to-speech technologies, it is possible to generate high-quality speech signals that are indistinguishable from the real ones. Speech editing, including speech insertion and replacement, appeals to researchers due to its potential applications. However, existing studies only considered clean speech scenarios. In real-world applications, the existence of environmental noise could significantly degrade the quality of generation. In this study, we propose a noise-resilient speech editing framework, SeamlessEdit, for noisy speech editing. SeamlessEdit adopts a frequency-band-aware noise suppression module and an in-content refinement strategy. It can well address the scenario where the frequency bands of voice and background noise are not separated. The proposed SeamlessEdit framework outperforms state-of-the-art approaches in multiple quantitative and qualitative evaluations.
>
---
#### [replaced 007] Multi-Target Backdoor Attacks Against Speaker Recognition
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08559v3](http://arxiv.org/pdf/2508.08559v3)**

> **作者:** Alexandrine Fortier; Sonal Joshi; Thomas Thebaud; Jesús Villalba; Najim Dehak; Patrick Cardinal
>
> **备注:** Accepted to IEEE Automatic Speech Recognition and Understanding Workshop 2025
>
> **摘要:** In this work, we propose a multi-target backdoor attack against speaker identification using position-independent clicking sounds as triggers. Unlike previous single-target approaches, our method targets up to 50 speakers simultaneously, achieving success rates of up to 95.04%. To simulate more realistic attack conditions, we vary the signal-to-noise ratio between speech and trigger, demonstrating a trade-off between stealth and effectiveness. We further extend the attack to the speaker verification task by selecting the most similar training speaker - based on cosine similarity - as a proxy target. The attack is most effective when target and enrolled speaker pairs are highly similar, reaching success rates of up to 90% in such cases.
>
---
#### [replaced 008] Provable Speech Attributes Conversion via Latent Independence
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.05191v2](http://arxiv.org/pdf/2510.05191v2)**

> **作者:** Jonathan Svirsky; Ofir Lindenbaum; Uri Shaham
>
> **摘要:** While signal conversion and disentangled representation learning have shown promise for manipulating data attributes across domains such as audio, image, and multimodal generation, existing approaches, especially for speech style conversion, are largely empirical and lack rigorous theoretical foundations to guarantee reliable and interpretable control. In this work, we propose a general framework for speech attribute conversion, accompanied by theoretical analysis and guarantees under reasonable assumptions. Our framework builds on a non-probabilistic autoencoder architecture with an independence constraint between the predicted latent variable and the target controllable variable. This design ensures a consistent signal transformation, conditioned on an observed style variable, while preserving the original content and modifying the desired attribute. We further demonstrate the versatility of our method by evaluating it on speech styles, including speaker identity and emotion. Quantitative evaluations confirm the effectiveness and generality of the proposed approach.
>
---
#### [replaced 009] I$^2$RF-TFCKD: Intra-Inter Representation Fusion with Time-Frequency Calibration Knowledge Distillation for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13127v2](http://arxiv.org/pdf/2506.13127v2)**

> **作者:** Jiaming Cheng; Ruiyu Liang; Ye Ni; Chao Xu; Jing Li; Wei Zhou; Rui Liu; Björn W. Schuller; Xiaoshuai Hao
>
> **备注:** submitted to Information Fusion
>
> **摘要:** In this paper, we propose an intra-inter representation fusion knowledge distillation (KD) framework with time-frequency calibration (I$^2$RF-TFCKD) for SE, which achieves distillation through the fusion of multi-layer teacher-student feature flows. Different from previous distillation strategies for SE, the proposed framework fully utilizes the time-frequency differential information of speech while promoting global knowledge flow. Firstly, we construct a collaborative distillation paradigm for intra-set and inter-set correlations. Within a correlated set, multi-layer teacher-student features are pairwise matched for calibrated distillation. Subsequently, we generate representative features from each correlated set through residual fusion to form the fused feature set that enables inter-set knowledge interaction. Secondly, we propose a multi-layer interactive distillation based on dual-stream time-frequency cross-calibration, which calculates the teacher-student similarity calibration weights in the time and frequency domains respectively and performs cross-weighting, thus enabling refined allocation of distillation contributions across different layers according to speech characteristics. The proposed distillation strategy is applied to the dual-path dilated convolutional recurrent network (DPDCRN) that ranked first in the SE track of the L3DAS23 challenge. To evaluate the effectiveness of I$^2$RF-TFCKD, we conduct experiments on both single-channel and multi-channel SE datasets. Objective evaluations demonstrate that the proposed KD strategy consistently and effectively improves the performance of the low-complexity student model and outperforms other distillation schemes.
>
---
#### [replaced 010] STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution
- **分类: cs.SD; cs.AI; cs.CR; eess.AS; 68T45, 68T10, 94A08; I.2.7; I.5.4; K.4.1**

- **链接: [http://arxiv.org/pdf/2505.19644v3](http://arxiv.org/pdf/2505.19644v3)**

> **作者:** Anton Firc; Manasi Chhibber; Jagabandhu Mishra; Vishwanath Pratap Singh; Tomi Kinnunen; Kamil Malinka
>
> **备注:** Published at Interspeech 2025 conference
>
> **摘要:** A key research area in deepfake speech detection is source tracing - determining the origin of synthesised utterances. The approaches may involve identifying the acoustic model (AM), vocoder model (VM), or other generation-specific parameters. However, progress is limited by the lack of a dedicated, systematically curated dataset. To address this, we introduce STOPA, a systematically varied and metadata-rich dataset for deepfake speech source tracing, covering 8 AMs, 6 VMs, and diverse parameter settings across 700k samples from 13 distinct synthesisers. Unlike existing datasets, which often feature limited variation or sparse metadata, STOPA provides a systematically controlled framework covering a broader range of generative factors, such as the choice of the vocoder model, acoustic model, or pretrained weights, ensuring higher attribution reliability. This control improves attribution accuracy, aiding forensic analysis, deepfake detection, and generative model transparency.
>
---
