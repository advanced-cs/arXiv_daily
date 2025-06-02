# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Introducing voice timbre attribute detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出音色属性检测任务（vTAD），旨在通过感知属性量化语音音色差异。研究基于说话人嵌入框架，对比ECAPA-TDNN和FACodec两种编码器在VCTK-RVA数据集的表现：ECAPA在已知说话人场景效果更佳，FACodec在未知说话人中泛化能力更强，同时公开了数据集与代码。**

- **链接: [http://arxiv.org/pdf/2505.09661v1](http://arxiv.org/pdf/2505.09661v1)**

> **作者:** Jinghao He; Zhengyan Sheng; Liping Chen; Kong Aik Lee; Zhen-Hua Ling
>
> **摘要:** This paper focuses on explaining the timbre conveyed by speech signals and introduces a task termed voice timbre attribute detection (vTAD). In this task, voice timbre is explained with a set of sensory attributes describing its human perception. A pair of speech utterances is processed, and their intensity is compared in a designated timbre descriptor. Moreover, a framework is proposed, which is built upon the speaker embeddings extracted from the speech utterances. The investigation is conducted on the VCTK-RVA dataset. Experimental examinations on the ECAPA-TDNN and FACodec speaker encoders demonstrated that: 1) the ECAPA-TDNN speaker encoder was more capable in the seen scenario, where the testing speakers were included in the training set; 2) the FACodec speaker encoder was superior in the unseen scenario, where the testing speakers were not part of the training, indicating enhanced generalization capability. The VCTK-RVA dataset and open-source code are available on the website https://github.com/vTAD2025-Challenge/vTAD.
>
---
#### [new 002] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对文本生成音频（T2A）任务中复杂多事件音频与人类偏好匹配不足的问题，提出基于AI反馈的优化方法。通过设计事件存在性、序列对齐和音质评分机制构建细粒度评价体系，并创建含41k标注的音频数据集T2A-FeedBack及评测基准T2A-EpicBench。实验表明，偏好调优显著提升了模型在简单与复杂场景下的生成效果。**

- **链接: [http://arxiv.org/pdf/2505.10561v1](http://arxiv.org/pdf/2505.10561v1)**

> **作者:** Zehan Wang; Ke Lei; Chen Zhu; Jiawei Huang; Sashuai Zhou; Luping Liu; Xize Cheng; Shengpeng Ji; Zhenhui Ye; Tao Jin; Zhou Zhao
>
> **备注:** ACL 2025
>
> **摘要:** Text-to-audio (T2A) generation has achieved remarkable progress in generating a variety of audio outputs from language prompts. However, current state-of-the-art T2A models still struggle to satisfy human preferences for prompt-following and acoustic quality when generating complex multi-event audio. To improve the performance of the model in these high-level applications, we propose to enhance the basic capabilities of the model with AI feedback learning. First, we introduce fine-grained AI audio scoring pipelines to: 1) verify whether each event in the text prompt is present in the audio (Event Occurrence Score), 2) detect deviations in event sequences from the language description (Event Sequence Score), and 3) assess the overall acoustic and harmonic quality of the generated audio (Acoustic&Harmonic Quality). We evaluate these three automatic scoring pipelines and find that they correlate significantly better with human preferences than other evaluation metrics. This highlights their value as both feedback signals and evaluation metrics. Utilizing our robust scoring pipelines, we construct a large audio preference dataset, T2A-FeedBack, which contains 41k prompts and 249k audios, each accompanied by detailed scores. Moreover, we introduce T2A-EpicBench, a benchmark that focuses on long captions, multi-events, and story-telling scenarios, aiming to evaluate the advanced capabilities of T2A models. Finally, we demonstrate how T2A-FeedBack can enhance current state-of-the-art audio model. With simple preference tuning, the audio generation model exhibits significant improvements in both simple (AudioCaps test set) and complex (T2A-EpicBench) scenarios.
>
---
#### [new 003] Learning Nonlinear Dynamics in Physical Modelling Synthesis using Neural Ordinary Differential Equations
- **分类: cs.SD; cs.LG; eess.AS; physics.comp-ph**

- **简介: 该论文研究物理建模中的非线性动力学学习，属于动态系统建模任务。针对分布式音乐系统（如弦乐）的非线性振动难以精确建模的问题，提出结合模态分解与神经常微分方程的方法：利用解析解处理线性振动，用神经网络捕捉非线性效应，保持物理参数可解释性。通过弦振动合成数据验证了模型有效性。**

- **链接: [http://arxiv.org/pdf/2505.10511v1](http://arxiv.org/pdf/2505.10511v1)**

> **作者:** Victor Zheleznov; Stefan Bilbao; Alec Wright; Simon King
>
> **备注:** Accepted for publication in Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25), Ancona, Italy, September 2025
>
> **摘要:** Modal synthesis methods are a long-standing approach for modelling distributed musical systems. In some cases extensions are possible in order to handle geometric nonlinearities. One such case is the high-amplitude vibration of a string, where geometric nonlinear effects lead to perceptually important effects including pitch glides and a dependence of brightness on striking amplitude. A modal decomposition leads to a coupled nonlinear system of ordinary differential equations. Recent work in applied machine learning approaches (in particular neural ordinary differential equations) has been used to model lumped dynamic systems such as electronic circuits automatically from data. In this work, we examine how modal decomposition can be combined with neural ordinary differential equations for modelling distributed musical systems. The proposed model leverages the analytical solution for linear vibration of system's modes and employs a neural network to account for nonlinear dynamic behaviour. Physical parameters of a system remain easily accessible after the training without the need for a parameter encoder in the network architecture. As an initial proof of concept, we generate synthetic data for a nonlinear transverse string and show that the model can be trained to reproduce the nonlinear dynamics of the system. Sound examples are presented.
>
---
#### [new 004] SpecWav-Attack: Leveraging Spectrogram Resizing and Wav2Vec 2.0 for Attacking Anonymized Speech
- **分类: cs.SD; cs.AI; eess.AS; I.2.0**

- **简介: 该论文属于对抗攻击任务，针对语音匿名化系统的漏洞，提出SpecWav-Attack模型。通过Wav2Vec2提取特征，结合频谱图缩放和增量训练提升说话人识别能力，在LibriSpeech数据集上超越传统攻击方法，揭示现有匿名技术的脆弱性，推动防御机制改进。**

- **链接: [http://arxiv.org/pdf/2505.09616v1](http://arxiv.org/pdf/2505.09616v1)**

> **作者:** Yuqi Li; Yuanzhong Zheng; Zhongtian Guo; Yaoxuan Wang; Jianjun Yin; Haojun Fei
>
> **备注:** 2 pages,3 figures,1 chart
>
> **摘要:** This paper presents SpecWav-Attack, an adversarial model for detecting speakers in anonymized speech. It leverages Wav2Vec2 for feature extraction and incorporates spectrogram resizing and incremental training for improved performance. Evaluated on librispeech-dev and librispeech-test, SpecWav-Attack outperforms conventional attacks, revealing vulnerabilities in anonymized speech systems and emphasizing the need for stronger defenses, benchmarked against the ICASSP 2025 Attacker Challenge.
>
---
#### [new 005] LAV: Audio-Driven Dynamic Visual Generation with Neural Compression and StyleGAN2
- **分类: cs.SD; cs.AI; cs.GR; cs.MM; eess.AS**

- **简介: 该论文提出LAV系统，属于音频驱动视觉生成任务，旨在解决传统方法依赖显式特征映射导致的语义缺失问题。通过融合EnCodec的音频压缩与StyleGAN2的生成能力，将音频嵌入直接映射到视觉风格空间，实现语义连贯的动态视听转换。**

- **链接: [http://arxiv.org/pdf/2505.10101v1](http://arxiv.org/pdf/2505.10101v1)**

> **作者:** Jongmin Jung; Dasaem Jeong
>
> **备注:** Paper accepted at ISEA 2025, The 30th International Symposium on Electronic/Emerging Art, Seoul, Republic of Korea, 23 - 29 May 2025
>
> **摘要:** This paper introduces LAV (Latent Audio-Visual), a system that integrates EnCodec's neural audio compression with StyleGAN2's generative capabilities to produce visually dynamic outputs driven by pre-recorded audio. Unlike previous works that rely on explicit feature mappings, LAV uses EnCodec embeddings as latent representations, directly transformed into StyleGAN2's style latent space via randomly initialized linear mapping. This approach preserves semantic richness in the transformation, enabling nuanced and semantically coherent audio-visual translations. The framework demonstrates the potential of using pretrained audio compression models for artistic and computational applications.
>
---
#### [new 006] Detecting Musical Deepfakes
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频分类任务，旨在检测AI生成的音乐（深度伪造）。为解决TTM平台带来的音乐伪造问题，研究者通过时频变换增强数据鲁棒性，利用梅尔谱图训练卷积神经网络进行真伪分类，并探讨了相关伦理影响，强调检测技术对保护音乐产业的重要性。**

- **链接: [http://arxiv.org/pdf/2505.09633v1](http://arxiv.org/pdf/2505.09633v1)**

> **作者:** Nick Sunday
>
> **备注:** Submitted as part of coursework at UT Austin. Accompanying code available at: https://github.com/nicksunday/deepfake-music-detector
>
> **摘要:** The proliferation of Text-to-Music (TTM) platforms has democratized music creation, enabling users to effortlessly generate high-quality compositions. However, this innovation also presents new challenges to musicians and the broader music industry. This study investigates the detection of AI-generated songs using the FakeMusicCaps dataset by classifying audio as either deepfake or human. To simulate real-world adversarial conditions, tempo stretching and pitch shifting were applied to the dataset. Mel spectrograms were generated from the modified audio, then used to train and evaluate a convolutional neural network. In addition to presenting technical results, this work explores the ethical and societal implications of TTM platforms, arguing that carefully designed detection systems are essential to both protecting artists and unlocking the positive potential of generative AI in music.
>
---
#### [new 007] Theoretical Model of Acoustic Power Transfer Through Solids
- **分类: cs.SD; eess.AS; physics.app-ph**

- **简介: 该论文属于声能传输的理论建模任务，旨在解决该技术基础理论不足的问题。研究提出通过固体介质传输机械波的声能理论模型，分析其工作机制，为人工耳蜗、声纳及无线充电等应用提供理论支撑，推动这一新兴技术的基础研究发展。**

- **链接: [http://arxiv.org/pdf/2505.09784v1](http://arxiv.org/pdf/2505.09784v1)**

> **作者:** Ippokratis Kochliaridis; Michail E. Kiziroglou
>
> **备注:** 8th International Workoshop on Microsystems, International Hellenic University
>
> **摘要:** Acoustic Power Transfer is a relatively new technology. It is a modern type of a wireless interface, where data signals and supply voltages are transmitted, with the use of mechanical waves, through a medium. The simplest application of such systems is the measurement of frequency response for audio speakers. It consists of a variable signal generator, a measuring amplifier which drives an acoustic source and the loudspeaker driver. The receiver contains a microphone circuit with a level recorder. Acoustic Power Transfer could have many applications, such as: Cochlear Implants, Sonar Systems and Wireless Charging. However, it is a new technology, thus it needs further investigation.
>
---
#### [new 008] ListenNet: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究听觉注意力检测（AAD），旨在通过脑电信号识别多说话人场景中的注意目标。针对现有EEG方法忽视时空依赖、泛化能力弱的问题，提出轻量级网络ListenNet，通过时空编码、多尺度时序增强和跨嵌套注意力模块强化特征提取，在三个数据集上实现最优性能且参数减少约7倍。**

- **链接: [http://arxiv.org/pdf/2505.10348v1](http://arxiv.org/pdf/2505.10348v1)**

> **作者:** Cunhang Fan; Xiaoke Yang; Hongyu Zhang; Ying Chen; Lu Li; Jian Zhou; Zhao Lv
>
> **摘要:** Auditory attention detection (AAD) aims to identify the direction of the attended speaker in multi-speaker environments from brain signals, such as Electroencephalography (EEG) signals. However, existing EEG-based AAD methods overlook the spatio-temporal dependencies of EEG signals, limiting their decoding and generalization abilities. To address these issues, this paper proposes a Lightweight Spatio-Temporal Enhancement Nested Network (ListenNet) for AAD. The ListenNet has three key components: Spatio-temporal Dependency Encoder (STDE), Multi-scale Temporal Enhancement (MSTE), and Cross-Nested Attention (CNA). The STDE reconstructs dependencies between consecutive time windows across channels, improving the robustness of dynamic pattern extraction. The MSTE captures temporal features at multiple scales to represent both fine-grained and long-range temporal patterns. In addition, the CNA integrates hierarchical features more effectively through novel dynamic attention mechanisms to capture deep spatio-temporal correlations. Experimental results on three public datasets demonstrate the superiority of ListenNet over state-of-the-art methods in both subject-dependent and challenging subject-independent settings, while reducing the trainable parameter count by approximately 7 times. Code is available at:https://github.com/fchest/ListenNet.
>
---
#### [new 009] Quantized Approximate Signal Processing (QASP): Towards Homomorphic Encryption for audio
- **分类: eess.AS; cs.CR; cs.SD**

- **简介: 该论文属于隐私保护的音频信号处理任务，旨在解决全同态加密（FHE）在音频时频变换中的效率与安全问题。作者提出量化近似信号处理方法，构建安全计算流程实现四种时频特征加密处理，开发低复杂度近似算法，并通过实验验证其降低计算开销和错误率的有效性。**

- **链接: [http://arxiv.org/pdf/2505.10500v1](http://arxiv.org/pdf/2505.10500v1)**

> **作者:** Tu Duyen Nguyen; Adrien Lesage; Clotilde Cantini; Rachid Riad
>
> **备注:** 34 pages, 5 figures
>
> **摘要:** Audio and speech data are increasingly used in machine learning applications such as speech recognition, speaker identification, and mental health monitoring. However, the passive collection of this data by audio listening devices raises significant privacy concerns. Fully homomorphic encryption (FHE) offers a promising solution by enabling computations on encrypted data and preserving user privacy. Despite its potential, prior attempts to apply FHE to audio processing have faced challenges, particularly in securely computing time frequency representations, a critical step in many audio tasks. Here, we addressed this gap by introducing a fully secure pipeline that computes, with FHE and quantized neural network operations, four fundamental time-frequency representations: Short-Time Fourier Transform (STFT), Mel filterbanks, Mel-frequency cepstral coefficients (MFCCs), and gammatone filters. Our methods also support the private computation of audio descriptors and convolutional neural network (CNN) classifiers. Besides, we proposed approximate STFT algorithms that lighten computation and bit use for statistical and machine learning analyses. We ran experiments on the VocalSet and OxVoc datasets demonstrating the fully private computation of our approach. We showed significant performance improvements with STFT approximation in private statistical analysis of audio markers, and for vocal exercise classification with CNNs. Our results reveal that our approximations substantially reduce error rates compared to conventional STFT implementations in FHE. We also demonstrated a fully private classification based on the raw audio for gender and vocal exercise classification. Finally, we provided a practical heuristic for parameter selection, making quantized approximate signal processing accessible to researchers and practitioners aiming to protect sensitive audio data.
>
---
#### [new 010] Who Said What WSW 2.0? Enhanced Automated Analysis of Preschool Classroom Speech
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音识别与教育分析交叉任务，旨在提升学前课堂语音交互分析的准确性和规模。通过整合wav2vec2说话人分类与Whisper语音转录技术，WSW2.0实现了儿童/教师语音区分（F1=.845）及转录（词错率0.12-0.24），验证其与人工标注的高相关性（ICC=0.64-0.98），并在1592小时数据中证实扩展性，为教育干预提供量化支持。**

- **链接: [http://arxiv.org/pdf/2505.09972v1](http://arxiv.org/pdf/2505.09972v1)**

> **作者:** Anchen Sun; Tiantian Feng; Gabriela Gutierrez; Juan J Londono; Anfeng Xu; Batya Elbaum; Shrikanth Narayanan; Lynn K Perry; Daniel S Messinger
>
> **备注:** 8 pages, 2 figures, 5 tables
>
> **摘要:** This paper introduces an automated framework WSW2.0 for analyzing vocal interactions in preschool classrooms, enhancing both accuracy and scalability through the integration of wav2vec2-based speaker classification and Whisper (large-v2 and large-v3) speech transcription. A total of 235 minutes of audio recordings (160 minutes from 12 children and 75 minutes from 5 teachers), were used to compare system outputs to expert human annotations. WSW2.0 achieves a weighted F1 score of .845, accuracy of .846, and an error-corrected kappa of .672 for speaker classification (child vs. teacher). Transcription quality is moderate to high with word error rates of .119 for teachers and .238 for children. WSW2.0 exhibits relatively high absolute agreement intraclass correlations (ICC) with expert transcriptions for a range of classroom language features. These include teacher and child mean utterance length, lexical diversity, question asking, and responses to questions and other utterances, which show absolute agreement intraclass correlations between .64 and .98. To establish scalability, we apply the framework to an extensive dataset spanning two years and over 1,592 hours of classroom audio recordings, demonstrating the framework's robustness for broad real-world applications. These findings highlight the potential of deep learning and natural language processing techniques to revolutionize educational research by providing accurate measures of key features of preschool classroom speech, ultimately guiding more effective intervention strategies and supporting early childhood language development.
>
---
#### [new 011] Spatially Selective Active Noise Control for Open-fitting Hearables with Acausal Optimization
- **分类: eess.AS; cs.SD; cs.SY; eess.SP; eess.SY**

- **简介: 该论文属于主动噪声控制领域，针对开放式耳机在保留特定方向目标声源时抑制噪声的难题。提出基于非因果相对脉冲响应的优化方法，通过仿真验证其在无混响环境中相较传统因果方法能更有效降低语音失真、提升降噪效果与信噪比，优化了空间选择性声学控制性能。**

- **链接: [http://arxiv.org/pdf/2505.10372v1](http://arxiv.org/pdf/2505.10372v1)**

> **作者:** Tong Xiao; Simon Doclo
>
> **备注:** Forum Acusticum/Euronoise 2025
>
> **摘要:** Recent advances in active noise control have enabled the development of hearables with spatial selectivity, which actively suppress undesired noise while preserving desired sound from specific directions. In this work, we propose an improved approach to spatially selective active noise control that incorporates acausal relative impulse responses into the optimization process, resulting in significantly improved performance over the causal design. We evaluate the system through simulations using a pair of open-fitting hearables with spatially localized speech and noise sources in an anechoic environment. Performance is evaluated in terms of speech distortion, noise reduction, and signal-to-noise ratio improvement across different delays and degrees of acausality. Results show that the proposed acausal optimization consistently outperforms the causal approach across all metrics and scenarios, as acausal filters more effectively characterize the response of the desired source.
>
---
## 更新

#### [replaced 001] ImprovNet -- Generating Controllable Musical Improvisations with Iterative Corruption Refinement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.04522v3](http://arxiv.org/pdf/2502.04522v3)**

> **作者:** Keshav Bhandari; Sungkyun Chang; Tongyu Lu; Fareza R. Enus; Louis B. Bradshaw; Dorien Herremans; Simon Colton
>
> **备注:** 10 pages, 6 figures, IJCNN 2025 conference
>
> **摘要:** Despite deep learning's remarkable advances in style transfer across various domains, generating controllable performance-level musical style transfer for complete symbolically represented musical works remains a challenging area of research. Much of this is owed to limited datasets, especially for genres such as jazz, and the lack of unified models that can handle multiple music generation tasks. This paper presents ImprovNet, a transformer-based architecture that generates expressive and controllable musical improvisations through a self-supervised corruption-refinement training strategy. The improvisational style transfer is aimed at making meaningful modifications to one or more musical elements - melody, harmony or rhythm of the original composition with respect to the target genre. ImprovNet unifies multiple capabilities within a single model: it can perform cross-genre and intra-genre improvisations, harmonize melodies with genre-specific styles, and execute short prompt continuation and infilling tasks. The model's iterative generation framework allows users to control the degree of style transfer and structural similarity to the original composition. Objective and subjective evaluations demonstrate ImprovNet's effectiveness in generating musically coherent improvisations while maintaining structural relationships with the original pieces. The model outperforms Anticipatory Music Transformer in short continuation and infilling tasks and successfully achieves recognizable genre conversion, with 79\% of participants correctly identifying jazz-style improvisations of classical pieces. Our code and demo page can be found at https://github.com/keshavbhandari/improvnet.
>
---
#### [replaced 002] Self-supervised Learning for Acoustic Few-Shot Classification
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.09647v2](http://arxiv.org/pdf/2409.09647v2)**

> **作者:** Jingyong Liang; Bernd Meyer; Isaac Ning Lee; Thanh-Toan Do
>
> **摘要:** Labelled data are limited and self-supervised learning is one of the most important approaches for reducing labelling requirements. While it has been extensively explored in the image domain, it has so far not received the same amount of attention in the acoustic domain. Yet, reducing labelling is a key requirement for many acoustic applications. Specifically in bioacoustic, there are rarely sufficient labels for fully supervised learning available. This has led to the widespread use of acoustic recognisers that have been pre-trained on unrelated data for bioacoustic tasks. We posit that training on the actual task data and combining self-supervised pre-training with few-shot classification is a superior approach that has the ability to deliver high accuracy even when only a few labels are available. To this end, we introduce and evaluate a new architecture that combines CNN-based preprocessing with feature extraction based on state space models (SSMs). This combination is motivated by the fact that CNN-based networks alone struggle to capture temporal information effectively, which is crucial for classifying acoustic signals. SSMs, specifically S4 and Mamba, on the other hand, have been shown to have an excellent ability to capture long-range dependencies in sequence data. We pre-train this architecture using contrastive learning on the actual task data and subsequent fine-tuning with an extremely small amount of labelled data. We evaluate the performance of this proposed architecture for ($n$-shot, $n$-class) classification on standard benchmarks as well as real-world data. Our evaluation shows that it outperforms state-of-the-art architectures on the few-shot classification problem.
>
---
#### [replaced 003] Acoustic Disturbance Sensing Level Detection for ASD Diagnosis and Intelligibility Enhancement
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2401.11832v3](http://arxiv.org/pdf/2401.11832v3)**

> **作者:** Marcelo Pillonetto; Anderson Queiroz; Rosângela Coelho
>
> **备注:** 4 pages, 3 figures, 2 tables
>
> **摘要:** The acoustic sensitivity of Autism Spectrum Disorder (ASD) individuals highly impacts their intelligibility in noisy urban environments. In this Letter, the disturbance sensing level is examined with perceptual listening tests that demonstrate the impact of their append High Internal Noise (HIN) profile on intelligibility. This particular sensing level is then proposed as additional aid to ASD diagnosis. In this Letter, a novel intelligibility enhancement scheme is also introduced for ASD particular circumstances. For this proposal, harmonic features estimated from speech signal frames are considered as center frequencies of auditory filterbanks. A gain factor is further applied to the output of the filtered samples. The experimental results demonstrate that the proposal improved the acoustic intelligibility of ASD and Neurotypicals (NT) people considering four acoustic noises at different signal-to-noise ratios.
>
---
#### [replaced 004] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.14074v3](http://arxiv.org/pdf/2409.14074v3)**

> **作者:** Khai Le-Duc; Phuc Phan; Tan-Hanh Pham; Bach Phan Tat; Minh-Huong Ngo; Chris Ngo; Thanh Nguyen-Tang; Truong-Son Hy
>
> **备注:** ACL 2025, 38 pages
>
> **摘要:** Multilingual automatic speech recognition (ASR) in the medical domain serves as a foundational task for various downstream applications such as speech translation, spoken language understanding, and voice-activated assistants. This technology improves patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we introduce MultiMed, the first multilingual medical ASR dataset, along with the first collection of small-to-large end-to-end medical ASR models, spanning five languages: Vietnamese, English, German, French, and Mandarin Chinese. To our best knowledge, MultiMed stands as the world's largest medical ASR dataset across all major benchmarks: total duration, number of recording conditions, number of accents, and number of speaking roles. Furthermore, we present the first multilinguality study for medical ASR, which includes reproducible empirical baselines, a monolinguality-multilinguality analysis, Attention Encoder Decoder (AED) vs Hybrid comparative study and a linguistic analysis. We present practical ASR end-to-end training schemes optimized for a fixed number of trainable parameters that are common in industry settings. All code, data, and models are available online: https://github.com/leduckhai/MultiMed/tree/master/MultiMed.
>
---
#### [replaced 005] CoGenAV: Versatile Audio-Visual Representation Learning via Contrastive-Generative Synchronization
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03186v2](http://arxiv.org/pdf/2505.03186v2)**

> **作者:** Detao Bai; Zhiheng Ma; Xihan Wei; Liefeng Bo
>
> **摘要:** The inherent synchronization between a speaker's lip movements, voice, and the underlying linguistic content offers a rich source of information for improving speech processing tasks, especially in challenging conditions where traditional audio-only systems falter. We introduce CoGenAV, a powerful and data-efficient model designed to learn versatile audio-visual representations applicable across a wide range of speech and audio-visual tasks. CoGenAV is trained by optimizing a dual objective derived from natural audio-visual synchrony, contrastive feature alignment and generative text prediction, using only 223 hours of labeled data from the LRS2 dataset. This contrastive-generative synchronization strategy effectively captures fundamental cross-modal correlations. We showcase the effectiveness and versatility of the learned CoGenAV representations on multiple benchmarks. When utilized for Audio-Visual Speech Recognition (AVSR) on LRS2, these representations contribute to achieving a state-of-the-art Word Error Rate (WER) of 1.27. They also enable strong performance in Visual Speech Recognition (VSR) with a WER of 20.5 on LRS2, and significantly improve performance in noisy environments by over 70%. Furthermore, CoGenAV representations benefit speech reconstruction tasks, boosting performance in Speech Enhancement and Separation, and achieve competitive results in audio-visual synchronization tasks like Active Speaker Detection (ASD). Our model will be open-sourced to facilitate further development and collaboration within both academia and industry.
>
---
#### [replaced 006] uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.01257v5](http://arxiv.org/pdf/2407.01257v5)**

> **作者:** Abdul Waheed; Karima Kadaoui; Bhiksha Raj; Muhammad Abdul-Mageed
>
> **备注:** Accepted to NAACL'25 main conference
>
> **摘要:** Recent work on distilling Whisper's knowledge into small models using pseudo-labels shows promising performance while reducing the size by up to 50%. This results in small, efficient, and dedicated models. However, a critical step of distillation using pseudo-labels involves filtering high-quality predictions and using only those during training. This step requires ground truth labels to compare with and filter low-quality examples, making the process dependent on human labels. Additionally, the distillation process requires a large amount of data thereby limiting its applicability in low-resource settings. To address this, we propose a distillation framework that does not require any labeled data. Through experimentation, we show that our best-distilled models outperform the teacher model by 5-7 WER points and are on par with or outperform similar supervised data filtering setups. When scaling the data, our models significantly outperform all zero-shot and supervised models. Our models are also 25-50% more compute- and memory-efficient while maintaining performance equal to or better than that of the teacher model. For more details about our models, dataset, and other resources, please visit our GitHub page: https://github.com/UBC-NLP/uDistilWhisper.
>
---
#### [replaced 007] An unsupervised method for MRI recovery: Deep image prior with structured sparsity
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.01482v2](http://arxiv.org/pdf/2501.01482v2)**

> **作者:** Muhammad Ahmad Sultan; Chong Chen; Yingmin Liu; Katarzyna Gil; Karolina Zareba; Rizwan Ahmad
>
> **备注:** Magn Reson Mater Phy (2025)
>
> **摘要:** Objective: To propose and validate an unsupervised MRI reconstruction method that does not require fully sampled k-space data. Materials and Methods: The proposed method, deep image prior with structured sparsity (DISCUS), extends the deep image prior (DIP) by introducing group sparsity to frame-specific code vectors, enabling the discovery of a low-dimensional manifold for capturing temporal variations. \discus was validated using four studies: (I) simulation of a dynamic Shepp-Logan phantom to demonstrate its manifold discovery capabilities, (II) comparison with compressed sensing and DIP-based methods using simulated single-shot late gadolinium enhancement (LGE) image series from six distinct digital cardiac phantoms in terms of normalized mean square error (NMSE) and structural similarity index measure (SSIM), (III) evaluation on retrospectively undersampled single-shot LGE data from eight patients, and (IV) evaluation on prospectively undersampled single-shot LGE data from eight patients, assessed via blind scoring from two expert readers. Results: DISCUS outperformed competing methods, demonstrating superior reconstruction quality in terms of NMSE and SSIM (Studies I--III) and expert reader scoring (Study IV). Discussion: An unsupervised image reconstruction method is presented and validated on simulated and measured data. These developments can benefit applications where acquiring fully sampled data is challenging.
>
---
#### [replaced 008] In-Materia Speech Recognition
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.10434v2](http://arxiv.org/pdf/2410.10434v2)**

> **作者:** Mohamadreza Zolfagharinejad; Julian Büchel; Lorenzo Cassola; Sachin Kinge; Ghazi Sarwat Syed; Abu Sebastian; Wilfred G. van der Wiel
>
> **摘要:** With the rise of decentralized computing, as in the Internet of Things, autonomous driving, and personalized healthcare, it is increasingly important to process time-dependent signals at the edge efficiently: right at the place where the temporal data are collected, avoiding time-consuming, insecure, and costly communication with a centralized computing facility (or cloud). However, modern-day processors often cannot meet the restrained power and time budgets of edge systems because of intrinsic limitations imposed by their architecture (von Neumann bottleneck) or domain conversions (analogue-to-digital and time-to-frequency). Here, we propose an edge temporal-signal processor based on two in-materia computing systems for both feature extraction and classification, reaching a software-level accuracy of 96.2% for the TI-46-Word speech-recognition task. First, a nonlinear, room-temperature dopant-network-processing-unit (DNPU) layer realizes analogue, time-domain feature extraction from the raw audio signals, similar to the human cochlea. Second, an analogue in-memory computing (AIMC) chip, consisting of memristive crossbar arrays, implements a compact neural network trained on the extracted features for classification. With the DNPU feature extraction consuming 100s nW and AIMC-based classification having the potential for less than 10 fJ per multiply-accumulate operation, our findings offer a promising avenue for advancing the compactness, efficiency, and performance of heterogeneous smart edge processors through in-materia computing hardware.
>
---
