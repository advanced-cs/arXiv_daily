# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] All-in-One ASR: Unifying Encoder-Decoder Models of CTC, Attention, and Transducer in Dual-Mode ASR
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决多模式ASR系统独立建模带来的高成本问题。提出All-in-One ASR统一框架，通过多模式联合器集成CTC、注意力和Transducer模型，支持离线与流式识别，减小模型体积并提升性能。**

- **链接: [https://arxiv.org/pdf/2512.11543v1](https://arxiv.org/pdf/2512.11543v1)**

> **作者:** Takafumi Moriya; Masato Mimura; Tomohiro Tanaka; Hiroshi Sato; Ryo Masumura; Atsunori Ogawa
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** This paper proposes a unified framework, All-in-One ASR, that allows a single model to support multiple automatic speech recognition (ASR) paradigms, including connectionist temporal classification (CTC), attention-based encoder-decoder (AED), and Transducer, in both offline and streaming modes. While each ASR architecture offers distinct advantages and trade-offs depending on the application, maintaining separate models for each scenario incurs substantial development and deployment costs. To address this issue, we introduce a multi-mode joiner that enables seamless integration of various ASR modes within a single unified model. Experiments show that All-in-One ASR significantly reduces the total model footprint while matching or even surpassing the recognition performance of individually optimized ASR models. Furthermore, joint decoding leverages the complementary strengths of different ASR modes, yielding additional improvements in recognition accuracy.
>
---
#### [new 002] The Affective Bridge: Unifying Feature Representations for Speech Deepfake Detection
- **分类: cs.SD**

- **简介: 该论文研究语音深伪检测任务，旨在解决现有方法特征不统一且难以解释的问题。作者提出以情感为桥梁的统一特征表示框架，通过情感关联提升检测性能与可解释性，在多个数据集上显著提升准确率并降低错误率。**

- **链接: [https://arxiv.org/pdf/2512.11241v1](https://arxiv.org/pdf/2512.11241v1)**

> **作者:** Yupei Li; Chenyang Lyu; Longyue Wang; Weihua Luo; Kaifu Zhang; Björn W. Schuller
>
> **摘要:** Speech deepfake detection has been widely explored using low-level acoustic descriptors. However, each study tends to select different feature sets, making it difficult to establish a unified representation for the task. Moreover, such features are not intuitive for humans to perceive, as the distinction between bona fide and synthesized speech becomes increasingly subtle with the advancement of deepfake generation techniques. Emotion, on the other hand, remains a unique human attribute that current deepfake generator struggles to fully replicate, reflecting the gap toward true artificial general intelligence. Interestingly, many existing acoustic and semantic features have implicit correlations with emotion. For instance, speech features recognized by automatic speech recognition systems often varies naturally with emotional expression. Based on this insight, we propose a novel training framework that leverages emotion as a bridge between conventional deepfake features and emotion-oriented representations. Experiments on the widely used FakeOrReal and In-the-Wild datasets demonstrate consistent and substantial improvements in accuracy, up to approximately 6% and 2% increases, respectively, and in equal error rate (EER), showing reductions of up to about 4% and 1%, respectively, while achieving comparable results on ASVspoof2019. This approach provides a unified training strategy for all features and interpretable feature direction for deepfake detection while improving model performance through emotion-informed learning.
>
---
#### [new 003] PhraseVAE and PhraseLDM: Latent Diffusion for Full-Song Multitrack Symbolic Music Generation
- **分类: cs.SD**

- **简介: 该论文研究全曲多轨符号音乐生成，旨在解决现有模型因基于音符级建模导致的长序列、上下文受限和结构连贯性差问题。提出PhraseVAE和PhraseLDM，首次采用短语级潜在扩散框架，实现高效、高质量完整歌曲生成。**

- **链接: [https://arxiv.org/pdf/2512.11348v1](https://arxiv.org/pdf/2512.11348v1)**

> **作者:** Longshen Ou; Ye Wang
>
> **摘要:** This technical report presents a new paradigm for full-song symbolic music generation. Existing symbolic models operate on note-attribute tokens and suffer from extremely long sequences, limited context length, and weak support for long-range structure. We address these issues by introducing PhraseVAE and PhraseLDM, the first latent diffusion framework designed for full-song multitrack symbolic music. PhraseVAE compresses variable-length polyphonic note sequences into compact 64-dimensional phrase-level representations with high reconstruction fidelity, allowing efficient training and a well-structured latent space. Built on this latent space, PhraseLDM generates an entire multi-track song in a single pass without any autoregressive components. The system eliminates bar-wise sequential modeling, supports up to 128 bars of music (8 minutes in 64 bpm), and produces complete songs with coherent local texture, idiomatic instrument patterns, and clear global structure. With only 45M parameters, our framework generates a full song within seconds while maintaining competitive musical quality and generation diversity. Together, these results show that phrase-level latent diffusion provides an effective and scalable solution to long-sequence modeling in symbolic music generation. We hope this work encourages future symbolic music research to move beyond note-attribute tokens and to consider phrase-level units as a more effective and musically meaningful modeling target.
>
---
#### [new 004] The TCG CREST -- RKMVERI Submission for the NCIIPC Startup India AI Grand Challenge
- **分类: cs.SD**

- **简介: 该论文针对语言无关的说话人识别与分离、转录及翻译任务，提出一个多语言语音处理系统。重点改进了低资源下的语音活动检测与说话人嵌入模型，并采用多核共识谱聚类提升说话人分离效果，集成ASR与翻译模块，增强实际应用性能。**

- **链接: [https://arxiv.org/pdf/2512.11009v1](https://arxiv.org/pdf/2512.11009v1)**

> **作者:** Nikhil Raghav; Arnab Banerjee; Janojit Chakraborty; Avisek Gupta; Swami Punyeshwarananda; Md Sahidullah
>
> **备注:** 6 pages, 3 tables, 3 figures, report submission for the NCIIPC Startup India AI Grand Challenge, Problem Statement 06
>
> **摘要:** In this report, we summarize the integrated multilingual audio processing pipeline developed by our team for the inaugural NCIIPC Startup India AI GRAND CHALLENGE, addressing Problem Statement 06: Language-Agnostic Speaker Identification and Diarisation, and subsequent Transcription and Translation System. Our primary focus was on advancing speaker diarization, a critical component for multilingual and code-mixed scenarios. The main intent of this work was to study the real-world applicability of our in-house speaker diarization (SD) systems. To this end, we investigated a robust voice activity detection (VAD) technique and fine-tuned speaker embedding models for improved speaker identification in low-resource settings. We leveraged our own recently proposed multi-kernel consensus spectral clustering framework, which substantially improved the diarization performance across all recordings in the training corpus provided by the organizers. Complementary modules for speaker and language identification, automatic speech recognition (ASR), and neural machine translation were integrated in the pipeline. Post-processing refinements further improved system robustness.
>
---
#### [new 005] Mitigation of multi-path propagation artefacts in acoustic targets with cepstral adaptive filtering
- **分类: cs.SD; cs.CE**

- **简介: 该论文属被动声学感知任务，旨在解决多径传播导致的信号干扰问题。提出基于倒谱域自适应带阻滤波的方法，分离目标信号与反射成分，提升信噪比和分类性能，适用于运动声源的去混响与分类。**

- **链接: [https://arxiv.org/pdf/2512.11165v1](https://arxiv.org/pdf/2512.11165v1)**

> **作者:** Lucas C. F. Domingos; Russell S. A. Brinkworth; Paulo E. Santos; Karl Sammut
>
> **摘要:** Passive acoustic sensing is a cost-effective solution for monitoring moving targets such as vessels and aircraft, but its performance is hindered by complex propagation effects like multi-path reflections and motion-induced artefacts. Existing filtering techniques do not properly incorporate the characteristics of the environment or account for variability in medium properties, limiting their effectiveness in separating source and reflection components. This paper proposes a method for separating target signals from their reflections in a spectrogram. Temporal filtering is applied to cepstral coefficients using an adaptive band-stop filter, which dynamically adjusts its bandwidth based on the relative intensity of the quefrency components. The method improved the signal-to-noise ratio (SNR), log-spectral distance (LSD), and Itakura-Saito (IS) distance across velocities ranging from 10 to 100 metres per second in aircraft noise with simulated motion. It also enhanced the performance of ship-type classification in underwater tasks by 2.28 and 2.62 Matthews Correlation Coefficient percentage points for the DeepShip and VTUAD v2 datasets, respectively. These results demonstrate the potential of the proposed pipeline to improve acoustic target classification and time-delay estimation in multi-path environments, with future work aimed at amplitude preservation and multi-sensor applications.
>
---
#### [new 006] Graph Embedding with Mel-spectrograms for Underwater Acoustic Target Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究水下声学目标识别任务，旨在解决传统深度学习模型忽略信号非欧特性的局限。作者提出UATR-GTransformer模型，通过Mel谱图分块与图Transformer结合，利用图神经网络挖掘频域局部关系，提升了特征表示能力与模型可解释性。**

- **链接: [https://arxiv.org/pdf/2512.11545v1](https://arxiv.org/pdf/2512.11545v1)**

> **作者:** Sheng Feng; Shuqing Ma; Xiaoqian Zhu
>
> **摘要:** Underwater acoustic target recognition (UATR) is extremely challenging due to the complexity of ship-radiated noise and the variability of ocean environments. Although deep learning (DL) approaches have achieved promising results, most existing models implicitly assume that underwater acoustic data lie in a Euclidean space. This assumption, however, is unsuitable for the inherently complex topology of underwater acoustic signals, which exhibit non-stationary, non-Gaussian, and nonlinear characteristics. To overcome this limitation, this paper proposes the UATR-GTransformer, a non-Euclidean DL model that integrates Transformer architectures with graph neural networks (GNNs). The model comprises three key components: a Mel patchify block, a GTransformer block, and a classification head. The Mel patchify block partitions the Mel-spectrogram into overlapping patches, while the GTransformer block employs a Transformer Encoder to capture mutual information between split patches to generate Mel-graph embeddings. Subsequently, a GNN enhances these embeddings by modeling local neighborhood relationships, and a feed-forward network (FFN) further performs feature transformation. Experiments results based on two widely used benchmark datasets demonstrate that the UATR-GTransformer achieves performance competitive with state-of-the-art methods. In addition, interpretability analysis reveals that the proposed model effectively extracts rich frequency-domain information, highlighting its potential for applications in ocean engineering.
>
---
#### [new 007] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文研究语音驱动的实时流式说话人头生成任务，旨在解决扩散模型推理慢、难以实时生成的问题。提出REST框架，通过紧凑潜在空间、ID-Context缓存机制和异步流式蒸馏策略，实现快速、连贯的端到端生成。**

- **链接: [https://arxiv.org/pdf/2512.11229v1](https://arxiv.org/pdf/2512.11229v1)**

> **作者:** Haotian Wang; Yuzhe Weng; Xinyi Yu; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 10pages, 4 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation. However, the slow inference speeds and non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, the first diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through high spatiotemporal VAE compression. Additionally, to enable autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles to key-value caching for maintaining temporal consistency and identity coherence during long-time streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) training strategy is proposed to mitigate error accumulation in autoregressive generation and enhance temporal consistency, which leverages a non-streaming teacher with an asynchronous noise schedule to supervise the training of the streaming student model. REST bridges the gap between autoregressive and diffusion-based approaches, demonstrating substantial value for applications requiring real-time talking head generation. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [new 008] Processing through encoding: Quantum circuit approaches for point-wise multiplication and convolution
- **分类: quant-ph; cs.ET; cs.SD; eess.SP**

- **简介: 该论文研究量子信号处理，提出通过编码实现复函数逐点乘法与卷积的量子电路方法。利用辅助量子比特编码函数，结合量子傅里叶变换实现乘法与卷积，集成至quantumaudio工具包并验证实验效果。**

- **链接: [https://arxiv.org/pdf/2512.11457v1](https://arxiv.org/pdf/2512.11457v1)**

> **作者:** Andreas Papageorgiou; Paulo Vitor Itaborai; Kostas Blekos; Karl Jansen
>
> **备注:** Presented at ISQCMC '25: 3rd International Symposium on Quantum Computing and Musical Creativity
>
> **摘要:** This paper introduces quantum circuit methodologies for pointwise multiplication and convolution of complex functions, conceptualized as "processing through encoding". Leveraging known techniques, we describe an approach where multiple complex functions are encoded onto auxiliary qubits. Applying the proposed scheme for two functions $f$ and $g$, their pointwise product $f(x)g(x)$ is shown to naturally form as the coefficients of part of the resulting quantum state. Adhering to the convolution theorem, we then demonstrate how the convolution $f*g$ can be constructed. Similarly to related work, this involves the encoding of the Fourier coefficients $\mathcal{F}[f]$ and $\mathcal{F}[g]$, which facilitates their pointwise multiplication, followed by the inverse Quantum Fourier Transform. We discuss the simulation of these techniques, their integration into an extended \verb|quantumaudio| package for audio signal processing, and present initial experimental validations. This work offers a promising avenue for quantum signal processing, with potential applications in areas such as quantum-enhanced audio manipulation and synthesis.
>
---
#### [new 009] Robust Detection of Underwater Target Against Non-Uniform Noise With Optical Fiber DAS Array
- **分类: eess.SP; eess.AS**

- **简介: 该论文研究水下目标方位检测，旨在解决非均匀噪声干扰问题。提出基于光纤DAS阵列与广义稀疏协方差拟合的新方法，采用螺旋增敏光缆提升灵敏度，通过仿真和实验验证了方法在复杂噪声下的高精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.11231v1](https://arxiv.org/pdf/2512.11231v1)**

> **作者:** Siyuan Cang; Cong Liu; Xueli Sheng; Xiaoming Cui; Chao Li; Changxin Fa; Jiantong Chen; Chaoran Yang; Huayong Yang
>
> **备注:** 17 pages, 29 figures. The IEEE Transactions on Instrumentation and Measurement has accepted this research for publication, and it is currently accessible in its early access version
>
> **摘要:** The detection of underwater targets is severely affected by the non-uniform spatial characteristics of marine environmental noise. Additionally, the presence of both natural and anthropogenic acoustic sources, including shipping traffic, marine life, and geological activity, further complicates the underwater acoustic landscape. Addressing these challenges requires advanced underwater sensors and robust signal processing techniques. In this paper, we present a novel approach that leverages an optical fiber distributed acoustic sensing (DAS) system combined with a broadband generalized sparse covariance-fitting framework for underwater target direction sensing, particularly focusing on robustness against non-uniform noise. The DAS system incorporates a newly developed spiral-sensitized optical cable, which significantly improves sensitivity compared to conventional submarine cables. This innovative design enables the system to capture acoustic signals with greater precision. Notably, the sensitivity of the spiral-wound sensitized cable is around -145.69 dB re: 1 rad / (uPa*m), as measured inside the standing-wave tube. Employing simulations, we assess the performance of the algorithm across diverse noise levels and target configurations, consistently revealing higher accuracy and reduced background noise compared to conventional beamforming techniques and other sparse techniques. In a controlled pool experiment, the correlation coefficient between waveforms acquired by the DAS system and a standard hydrophone reached 0.973, indicating high fidelity in signal capture.
>
---
#### [new 010] ASR Under the Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文评估印度多语言临床场景中语音识别（ASR）系统的偏见问题，属语音技术公平性研究。针对患者与医生、不同性别及语言群体，测试多种ASR模型在真实临床对话中的表现，揭示跨语言、角色与性别的系统性性能差距，推动医疗ASR的包容性发展。**

- **链接: [https://arxiv.org/pdf/2512.10967v1](https://arxiv.org/pdf/2512.10967v1)**

> **作者:** Subham Kumar; Prakrithi Shivaprakash; Abhishek Manoharan; Astut Kurariya; Diptadhi Mukherjee; Lekhansh Shukla; Animesh Mukherjee; Prabhat Chand; Pratima Murthy
>
> **摘要:** Automatic Speech Recognition (ASR) is increasingly used to document clinical encounters, yet its reliability in multilingual and demographically diverse Indian healthcare contexts remains largely unknown. In this study, we conduct the first systematic audit of ASR performance on real world clinical interview data spanning Kannada, Hindi, and Indian English, comparing leading models including Indic Whisper, Whisper, Sarvam, Google speech to text, Gemma3n, Omnilingual, Vaani, and Gemini. We evaluate transcription accuracy across languages, speakers, and demographic subgroups, with a particular focus on error patterns affecting patients vs. clinicians and gender based or intersectional disparities. Our results reveal substantial variability across models and languages, with some systems performing competitively on Indian English but failing on code mixed or vernacular speech. We also uncover systematic performance gaps tied to speaker role and gender, raising concerns about equitable deployment in clinical settings. By providing a comprehensive multilingual benchmark and fairness analysis, our work highlights the need for culturally and demographically inclusive ASR development for healthcare ecosystem in India.
>
---
#### [new 011] Benchmarking Automatic Speech Recognition Models for African Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决非洲语言因资源稀缺导致的模型选择与优化问题。作者系统评测了四种主流ASR模型在13种非洲语言上的表现，分析不同数据规模和解码策略下的性能差异，揭示模型特性与资源条件的相互作用，为低资源语言提供实用设计指导。**

- **链接: [https://arxiv.org/pdf/2512.10968v1](https://arxiv.org/pdf/2512.10968v1)**

> **作者:** Alvin Nahabwe; Sulaiman Kagumire; Denis Musinguzi; Bruno Beijuka; Jonah Mubuuke Kyagaba; Peter Nabende; Andrew Katumba; Joyce Nakatumba-Nabende
>
> **备注:** 19 pages, 8 figures, Deep Learning Indiba, Proceedings of Machine Learning Research
>
> **摘要:** Automatic speech recognition (ASR) for African languages remains constrained by limited labeled data and the lack of systematic guidance on model selection, data scaling, and decoding strategies. Large pre-trained systems such as Whisper, XLS-R, MMS, and W2v-BERT have expanded access to ASR technology, but their comparative behavior in African low-resource contexts has not been studied in a unified and systematic way. In this work, we benchmark four state-of-the-art ASR models across 13 African languages, fine-tuning them on progressively larger subsets of transcribed data ranging from 1 to 400 hours. Beyond reporting error rates, we provide new insights into why models behave differently under varying conditions. We show that MMS and W2v-BERT are more data efficient in very low-resource regimes, XLS-R scales more effectively as additional data becomes available, and Whisper demonstrates advantages in mid-resource conditions. We also analyze where external language model decoding yields improvements and identify cases where it plateaus or introduces additional errors, depending on the alignment between acoustic and text resources. By highlighting the interaction between pre-training coverage, model architecture, dataset domain, and resource availability, this study offers practical and insights into the design of ASR systems for underrepresented languages.
>
---
## 更新

#### [replaced 001] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models
- **分类: cs.SD**

- **简介: 该论文聚焦音频-语言模型的时序理解任务，旨在解决现有模型在时间定位和长音频理解上的局限。作者提出TimeAudio方法，通过时间标记、绝对时间编码和分段令牌合并模块，提升模型对时间敏感任务的性能，并构建新数据集与评测指标进行验证。**

- **链接: [https://arxiv.org/pdf/2511.11039v2](https://arxiv.org/pdf/2511.11039v2)**

> **作者:** Hualei Wang; Yiming Li; Shuo Ma; Hong Liu; Xiangdong Wang
>
> **备注:** Accepted by The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Recent Large Audio-Language Models (LALMs) exhibit impressive capabilities in understanding audio content for conversational QA tasks. However, these models struggle to accurately understand timestamps for temporal localization (e.g., Temporal Audio Grounding) and are restricted to short audio perception, leading to constrained capabilities on fine-grained tasks. We identify three key aspects that limit their temporal localization and long audio understanding: (i) timestamp representation, (ii) architecture, and (iii) data. To address this, we introduce TimeAudio, a novel method that empowers LALMs to connect their understanding of audio content with precise temporal perception. Specifically, we incorporate unique temporal markers to improve time-sensitive reasoning and apply an absolute time-aware encoding that explicitly grounds the acoustic features with absolute time information. Moreover, to achieve end-to-end long audio understanding, we introduce a segment-level token merging module to substantially reduce audio token redundancy and enhance the efficiency of information extraction. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing audio datasets into a new dataset focused on temporal tasks and establish a series of metrics to evaluate the fine-grained performance. Evaluations show strong performance across a variety of fine-grained tasks, such as dense captioning, temporal grounding, and timeline speech summarization, demonstrating TimeAudio's robust temporal localization and reasoning capabilities.
>
---
#### [replaced 002] Video Echoed in Music: Semantic, Temporal, and Rhythmic Alignment for Video-to-Music Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文研究视频到音乐生成任务，旨在解决现有方法在语义、时序和节奏对齐上的不足。提出VeM模型，通过分层视频解析、跨模态注意力和节拍对齐机制，实现音乐与视频的精准同步，并构建新数据集与评估指标验证效果。**

- **链接: [https://arxiv.org/pdf/2511.09585v4](https://arxiv.org/pdf/2511.09585v4)**

> **作者:** Xinyi Tong; Yiran Zhu; Jishang Chen; Chunru Zhan; Tianle Wang; Sirui Zhang; Nian Liu; Tiezheng Ge; Duo Xu; Xin Jin; Feng Yu; Song-Chun Zhu
>
> **摘要:** Video-to-Music generation seeks to generate musically appropriate background music that enhances audiovisual immersion for videos. However, current approaches suffer from two critical limitations: 1) incomplete representation of video details, leading to weak alignment, and 2) inadequate temporal and rhythmic correspondence, particularly in achieving precise beat synchronization. To address the challenges, we propose Video Echoed in Music (VeM), a latent music diffusion that generates high-quality soundtracks with semantic, temporal, and rhythmic alignment for input videos. To capture video details comprehensively, VeM employs a hierarchical video parsing that acts as a music conductor, orchestrating multi-level information across modalities. Modality-specific encoders, coupled with a storyboard-guided cross-attention mechanism (SG-CAtt), integrate semantic cues while maintaining temporal coherence through position and duration encoding. For rhythmic precision, the frame-level transition-beat aligner and adapter (TB-As) dynamically synchronize visual scene transitions with music beats. We further contribute a novel video-music paired dataset sourced from e-commerce advertisements and video-sharing platforms, which imposes stricter transition-beat synchronization requirements. Meanwhile, we introduce novel metrics tailored to the task. Experimental results demonstrate superiority, particularly in semantic relevance and rhythmic precision.
>
---
#### [replaced 003] Joint Learning of Wording and Formatting for Singable Melody-to-Lyric Generation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究旋律到歌词生成任务，旨在提升生成歌词的可唱性。通过联合学习用词与格式（如行数、音节数），引入音乐学启发的辅助目标，增强模型对韵律与结构的建模，显著提升格式准确率与人工评价得分。**

- **链接: [https://arxiv.org/pdf/2307.02146v3](https://arxiv.org/pdf/2307.02146v3)**

> **作者:** Longshen Ou; Xichu Ma; Ye Wang
>
> **备注:** An extension of our previous work arXiv:2305.16816 [cs.CL]
>
> **摘要:** Despite progress in melody-to-lyric generation, a substantial singability gap remains between machine-generated lyrics and those written by human lyricists. In this work, we aim to narrow this gap by jointly learning both wording and formatting for melody-to-lyric generation. After general-domain pretraining, our model acquires length awareness through an self-supervised stage trained on a large text-only lyric corpus. During supervised melody-to-lyric training, we introduce multiple auxiliary supervision objective informed by musicological findings on melody--lyric relationships, encouraging the model to capture fine-grained prosodic and structural patterns. Compared with naïve fine-tuning, our approach improves adherence to line-count and syllable-count requirements by 3.8% and 21.4% absolute, respectively, without degrading text quality. In human evaluation, it achieves 42.2% and 74.2% relative gains in overall quality over two task-specific baselines, underscoring the importance of formatting-aware training for generating singable lyrics.
>
---
#### [replaced 004] Diffusion-based Surrogate Model for Time-varying Underwater Acoustic Channels
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对水下声学信道建模难题，提出一种基于扩散模型的生成式代理模型StableUASim。它通过预训练学习信道随机动态，实现数据高效、可推广的时变信道模拟，支持条件生成与快速环境适配，用于通信系统设计与机器学习应用。**

- **链接: [https://arxiv.org/pdf/2511.18078v2](https://arxiv.org/pdf/2511.18078v2)**

> **作者:** Kexin Li; Mandar Chitre
>
> **备注:** Updated references with DOIs
>
> **摘要:** Accurate modeling of time-varying underwater acoustic channels is essential for the design, evaluation, and deployment of reliable underwater communication systems. Conventional physics models require detailed environmental knowledge, while stochastic replay methods are constrained by the limited diversity of measured channels and often fail to generalize to unseen scenarios, reducing their practical applicability. To address these challenges, we propose StableUASim, a pre-trained conditional latent diffusion surrogate model that captures the stochastic dynamics of underwater acoustic communication channels. Leveraging generative modeling, StableUASim produces diverse and statistically realistic channel realizations, while supporting conditional generation from specific measurement samples. Pre-training enables rapid adaptation to new environments using minimal additional data, and the autoencoder latent representation facilitates efficient channel analysis and compression. Experimental results demonstrate that StableUASim accurately reproduces key channel characteristics and communication performance, providing a scalable, data-efficient, and physically consistent surrogate model for both system design and machine learning-driven underwater applications.
>
---
#### [replaced 005] Recent Advances in Discrete Speech Tokens: A Review
- **分类: eess.AS; cs.AI; cs.MM; cs.SD; eess.SP**

- **简介: 该论文属综述任务，旨在梳理离散语音标记的最新进展。它分类并比较了声学与语义标记方法，分析其优劣，探讨现存挑战，提出未来方向，推动语音与大语言模型融合。**

- **链接: [https://arxiv.org/pdf/2502.06490v4](https://arxiv.org/pdf/2502.06490v4)**

> **作者:** Yiwei Guo; Zhihan Li; Hankun Wang; Bohan Li; Chongtian Shao; Hanglei Zhang; Chenpeng Du; Xie Chen; Shujie Liu; Kai Yu
>
> **备注:** 26 pages, 8 figures, 3 tables. Accepted to IEEE TPAMI
>
> **摘要:** The rapid advancement of speech generation technologies in the era of large language models (LLMs) has established discrete speech tokens as a foundational paradigm for speech representation. These tokens, characterized by their discrete, compact, and concise nature, are not only advantageous for efficient transmission and storage, but also inherently compatible with the language modeling framework, enabling seamless integration of speech into text-dominated LLM architectures. Current research categorizes discrete speech tokens into two principal classes: acoustic tokens and semantic tokens, each of which has evolved into a rich research domain characterized by unique design philosophies and methodological approaches. This survey systematically synthesizes the existing taxonomy and recent innovations in discrete speech tokenization, conducts a critical examination of the strengths and limitations of each paradigm, and presents systematic experimental comparisons across token types. Furthermore, we identify persistent challenges in the field and propose potential research directions, aiming to offer actionable insights to inspire future advancements in the development and application of discrete speech tokens.
>
---
#### [replaced 006] End-to-end transfer learning for speaker-independent cross-language and cross-corpus speech emotion recognition
- **分类: eess.AS**

- **简介: 该论文研究跨语言、跨语料库的说话人无关语音情感识别任务，旨在解决数据分布差异导致的性能下降问题。提出基于wav2vec 2.0和Deep-WCCN层的端到端迁移学习模型，有效降低语言、说话人等变异影响，提升小样本目标语言下的识别准确率。**

- **链接: [https://arxiv.org/pdf/2311.13678v3](https://arxiv.org/pdf/2311.13678v3)**

> **作者:** Duowei Tang; Peter Kuppens; Lucca Geurts; Toon van Waterschoot
>
> **备注:** 27 pages, 6 figures, 4 tables
>
> **摘要:** Data-driven models achieve successful results in Speech Emotion Recognition (SER). However, these models, which are often based on general acoustic features or end-to-end approaches, show poor performance when the testing set has a different language than the training set or when these sets are taken from different datasets. To alleviate these problems, this paper presents an end-to-end Deep Neural Network (DNN) model based on transfer learning for cross-language and cross-corpus SER. We use the wav2vec 2.0 pre-trained model to transform audio time-domain waveforms from different languages, different speakers and different recording conditions into a feature space shared by multiple languages, thereby reducing the language variabilities in the speech embeddings. Next, we propose a new Deep-Within-Class Covariance Normalisation (Deep-WCCN) layer that can be inserted into the DNN model and aims to reduce other variabilities including speaker variability, channel variability and so on. The entire model is fine-tuned in an end-to-end manner on a combined loss and is validated on datasets from three languages (i.e. English, German, Chinese). Experimental results show that our proposed method outperforms the baseline model that is based on common acoustic feature sets for SER in the within-language setting and the cross-language setting. In addition, we also experimentally validate the effectiveness of Deep-WCCN, which can further improve the model performance. Next, we show that the proposed transfer learning method has good data efficiency when merging target language data into the fine-tuning process. The model speaker-independent SER performance increases with up to 15.6% when only 160s of target language data is used. Finally, our proposed model shows significantly better performance than other state-of-the-art models in cross-language SER.
>
---
