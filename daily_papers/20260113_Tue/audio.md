# 音频 cs.SD;  eess.AS

- **最新发布 19 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Stereo Audio Rendering for Personal Sound Zones Using a Binaural Spatially Adaptive Neural Network (BSANN)
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频渲染任务，旨在解决多听众独立立体声播放问题。通过BSANN方法实现耳级控制和主动串扰消除，提升空间音频精度与隔离度。**

- **链接: [https://arxiv.org/pdf/2601.06621v1](https://arxiv.org/pdf/2601.06621v1)**

> **作者:** Hao Jiang; Edgar Choueiri
>
> **备注:** Submitted to IEEE Transactions on Audio, Speech, and Language Processing (TASLP)
>
> **摘要:** A binaural rendering framework for personal sound zones (PSZs) is proposed to enable multiple head-tracked listeners to receive fully independent stereo audio programs. Current PSZ systems typically rely on monophonic rendering and therefore cannot control the left and right ears separately, which limits the quality and accuracy of spatial imaging. The proposed method employs a Binaural Spatially Adaptive Neural Network (BSANN) to generate ear-optimized loudspeaker filters that reconstruct the desired acoustic field at each ear of multiple listeners. The framework integrates anechoically measured loudspeaker frequency responses, analytically modeled transducer directivity, and rigid-sphere head-related transfer functions (HRTFs) to enhance acoustic accuracy and spatial rendering fidelity. An explicit active crosstalk cancellation (XTC) stage further improves three-dimensional spatial perception. Experiments show significant gains in measured objective performance metrics, including inter-zone isolation (IZI), inter-program isolation (IPI), and crosstalk cancellation (XTC), with log-frequency-weighted values of 10.23/10.03 dB (IZI), 11.11/9.16 dB (IPI), and 10.55/11.13 dB (XTC), respectively, over 100-20,000 Hz. The combined use of ear-wise control, accurate acoustic modeling, and integrated active XTC produces a unified rendering method that delivers greater isolation performance, increased robustness to room asymmetry, and more faithful spatial reproduction in real acoustic environments.
>
---
#### [new 002] Directional Selective Fixed-Filter Active Noise Control Based on a Convolutional Neural Network in Reverberant Environments
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于主动降噪任务，旨在解决混响环境中方向性噪声控制问题。通过引入卷积神经网络，实现基于声源方向的固定滤波降噪，提升降噪效果与响应速度。**

- **链接: [https://arxiv.org/pdf/2601.06981v1](https://arxiv.org/pdf/2601.06981v1)**

> **作者:** Boxiang Wang; Zhengding Luo; Haowen Li; Dongyuan Shi; Junwei Ji; Ziyi Yang; Woon-Seng Gan
>
> **摘要:** Selective fixed-filter active noise control (SFANC) is a novel approach capable of mitigating noise with varying frequency characteristics. It offers faster response and greater computational efficiency compared to traditional adaptive algorithms. However, spatial factors, particularly the influence of the noise source location, are often overlooked. Some existing studies have explored the impact of the direction-of-arrival (DoA) of the noise source on ANC performance, but they are mostly limited to free-field conditions and do not consider the more complex indoor reverberant environments. To address this gap, this paper proposes a learning-based directional SFANC method that incorporates the DoA of the noise source in reverberant environments. In this framework, multiple reference signals are processed by a convolutional neural network (CNN) to estimate the azimuth and elevation angles of the noise source, as well as to identify the most appropriate control filter for effective noise cancellation. Compared to traditional adaptive algorithms, the proposed approach achieves superior noise reduction with shorter response times, even in the presence of reverberations.
>
---
#### [new 003] An Intelligent AI glasses System with Multi-Agent Architecture for Real-Time Voice Processing and Task Execution
- **分类: cs.SD; cs.AI; cs.HC**

- **简介: 该论文提出一种智能AI眼镜系统，采用多智能体架构实现实时语音处理与任务执行，解决跨平台语音交互与任务协调问题。**

- **链接: [https://arxiv.org/pdf/2601.06235v1](https://arxiv.org/pdf/2601.06235v1)**

> **作者:** Sheng-Kai Chen; Jyh-Horng Wu; Ching-Yao Lin; Yen-Ting Lin
>
> **备注:** Published in NCS 2025 (Paper No. N0180)
>
> **摘要:** This paper presents an AI glasses system that integrates real-time voice processing, artificial intelligence(AI) agents, and cross-network streaming capabilities. The system employs dual-agent architecture where Agent 01 handles Automatic Speech Recognition (ASR) and Agent 02 manages AI processing through local Large Language Models (LLMs), Model Context Protocol (MCP) tools, and Retrieval-Augmented Generation (RAG). The system supports real-time RTSP streaming for voice and video data transmission, eye tracking data collection, and remote task execution through RabbitMQ messaging. Implementation demonstrates successful voice command processing with multilingual support and cross-platform task execution capabilities.
>
---
#### [new 004] Dereverberation Filter by Deconvolution with Frequency Bin Specific Faded Impulse Response
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决非理想录音中的混响问题。通过频段特定的衰减方法优化冲激响应，实现更清晰的直接路径信号重建。**

- **链接: [https://arxiv.org/pdf/2601.06662v1](https://arxiv.org/pdf/2601.06662v1)**

> **作者:** Stefan Ciba
>
> **备注:** 8 pages, 3 figures, github repository with code and audio
>
> **摘要:** This work introduces a robust single-channel inverse filter for dereverberation of non-ideal recordings, validated on real audio. The developed method focuses on the calculation and modification of a discrete impulse response in order to filter the characteristics from a known digital single channel recording setup and room characteristics such as early reflections and reverberations. The aim is a dryer and clearer signal reconstruction, which ideally would be the direct-path signal. The time domain impulse response is calculated from the cepstral domain and faded by means of frequency bin specific exponential decay in the spectrum. The decay rates are obtained by using the blind estimates of reverberation time ratio between recorded output and test signals for each frequency bin. The modified impulse response does filter a recorded audio-signal by deconvolution. The blind estimation is well known and stands out for its robustness to noise and non-idealities. Estimation of a direct path signal is key to many applications.
>
---
#### [new 005] SEE: Signal Embedding Energy for Quantifying Noise Interference in Large Audio Language Models
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频语言模型鲁棒性研究，解决噪声影响量化问题。提出SEE方法，通过模型内部表示分析噪声影响，提升模型在真实场景下的稳定性。**

- **链接: [https://arxiv.org/pdf/2601.07331v1](https://arxiv.org/pdf/2601.07331v1)**

> **作者:** Yuanhe Zhang; Jiayu Tian; Yibo Zhang; Shilinlu Yan; Liang Lin; Zhenhong Zhou; Li Sun; Sen Su
>
> **摘要:** Large Audio Language Models (LALMs) have been widely applied in real-time scenarios, such as in-car assistants and online meeting comprehension. In practice, audio inputs are often corrupted by device and environmental noise, leading to performance degradation. However, existing LALM studies on noise lack quantitative analysis and rely mainly on intuition and empirical observation, thus failing to understand practical robustness. To address this issue, we introduce Signal Embedding Energy (SEE), a method for quantifying the impact of noise intensity on LALM inputs, enabling the differentiation of LALM robustness in real-world deployments. SEE introduces a perspective based on structured activation subspaces derived from the model's internal representations, which more accurately captures its perception of noise than raw audio features. Across experiments, SEE exhibits a strong correlation with LALM performance, achieving a correlation of 0.98. Surprisingly, traditional audio denoising methods are only marginally effective for LALMs, and, in some cases, even increase SEE and impair performance. This suggests a mismatch between speech-centric denoising objectives and the noise sensitivity of modern LALMs. Therefore, we propose a mitigation strategy derived from SEE to denoise LALM inputs, outperforming existing denoising methods. This paper introduces a novel metric for noise quantification in LALMs, providing guidance for robustness improvements in real-world deployments.
>
---
#### [new 006] DIVINE: Coordinating Multimodal Disentangled Representations for Oro-Facial Neurological Disorder Assessment
- **分类: eess.AS**

- **简介: 该论文属于多模态情感分析任务，旨在提升神经面部疾病评估的准确性与可解释性。通过分离共享与模态特定表示，结合自适应融合与多任务学习，提出DIVINE框架，实现音频与视频信息的联合分析。**

- **链接: [https://arxiv.org/pdf/2601.07014v1](https://arxiv.org/pdf/2601.07014v1)**

> **作者:** Mohd Mujtaba Akhtar; Girish; Muskaan Singh
>
> **备注:** Accepted to EACL 2026
>
> **摘要:** In this study, we present a multimodal framework for predicting neuro-facial disorders by capturing both vocal and facial cues. We hypothesize that explicitly disentangling shared and modality-specific representations within multimodal foundation model embeddings can enhance clinical interpretability and generalization. To validate this hypothesis, we propose DIVINE a fully disentangled multimodal framework that operates on representations extracted from state-of-the-art (SOTA) audio and video foundation models, incorporating hierarchical variational bottlenecks, sparse gated fusion, and learnable symptom tokens. DIVINE operates in a multitask learning setup to jointly predict diagnostic categories (Healthy Control,ALS, Stroke) and severity levels (Mild, Moderate, Severe). The model is trained using synchronized audio and video inputs and evaluated on the Toronto NeuroFace dataset under full (audio-video) as well as single-modality (audio- only and video-only) test conditions. Our proposed approach, DIVINE achieves SOTA result, with the DeepSeek-VL2 and TRILLsson combination reaching 98.26% accuracy and 97.51% F1-score. Under modality-constrained scenarios, the framework performs well, showing strong generalization when tested with video-only or audio-only inputs. It consistently yields superior performance compared to unimodal models and baseline fusion techniques. To the best of our knowledge, DIVINE is the first framework that combines cross-modal disentanglement, adaptive fusion, and multitask learning to comprehensively assess neurological disorders using synchronized speech and facial video.
>
---
#### [new 007] Auditory Filter Behavior and Updated Estimated Constants
- **分类: eess.AS; cs.SD; eess.SP; eess.SY; q-bio.TO**

- **简介: 该论文属于信号处理任务，旨在改进听觉滤波器常数的估计。通过新方法分析滤波器特性，解决传统值过时的问题，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2601.06094v1](https://arxiv.org/pdf/2601.06094v1)**

> **作者:** Samiya A Alkhairy
>
> **备注:** 19 pages, 36 equations, 10 figures, 2 tables, submitted
>
> **摘要:** Filters from the Gammatone family are often used to model auditory signal processing, but the filter constant values used to mimic human hearing are largely set to values based on historical psychoacoustic data collected several decades ago. Here, we move away from this long-standing convention, and estimate filter constants using a range of more recent reported filter characteristics (such as quality factors and ratios between quality factors and peak group delay) within a characteristics-based framework that clarifies how filter behavior is related to the underlying constants. Using a sharp-filter approximation that captures shared peak-region behavior across certain classes of filters, we analyze the range of behaviors accessible when the full degrees of freedom of the filter are utilized rather than fixing the filter order or exponent to historically prescribed values. Filter behavior is characterized using magnitude-based and phase-based characteristics and their ratios, which reveal which characteristics are informative for constraining filter constants and which are only weakly constraining. We show that these insights and estimation methods extend to multiple realizable filter classes from the Gammatone family and apply them, together with recent physiological and psychoacoustic observations, to derive constraints on and estimates for filter constants for human auditory filters. More broadly, this framework supports the design of auditory filters with arbitrary characteristic-level specifications and enables systematic assessment of how variations in filter characteristics influence auditory models, perceptual findings, and technologies that rely on auditory filterbanks.
>
---
#### [new 008] ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 该论文属于语音防伪任务，旨在解决组件级深度伪造音频检测问题。提出CompSpoofV2数据集和联合学习框架，开展ESDD2挑战赛，提升真实环境下的检测能力。**

- **链接: [https://arxiv.org/pdf/2601.07303v1](https://arxiv.org/pdf/2601.07303v1)**

> **作者:** Xueping Zhang; Han Yin; Yang Xiao; Lin Zhang; Ting Dang
>
> **摘要:** Audio recorded in real-world environments often contains a mixture of foreground speech and background environmental sounds. With rapid advances in text-to-speech, voice conversion, and other generation models, either component can now be modified independently. Such component-level manipulations are harder to detect, as the remaining unaltered component can mislead the systems designed for whole deepfake audio, and they often sound more natural to human listeners. To address this gap, we have proposed CompSpoofV2 dataset and a separation-enhanced joint learning framework. CompSpoofV2 is a large-scale curated dataset designed for component-level audio anti-spoofing, which contains over 250k audio samples, with a total duration of approximately 283 hours. Based on the CompSpoofV2 and the separation-enhanced joint learning framework, we launch the Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2), focusing on component-level spoofing, where both speech and environmental sounds may be manipulated or synthesized, creating a more challenging and realistic detection scenario. The challenge will be held in conjunction with the IEEE International Conference on Multimedia and Expo 2026 (ICME 2026).
>
---
#### [new 009] FOCAL: A Novel Benchmarking Technique for Multi-modal Agents
- **分类: cs.SD**

- **简介: 该论文提出FOCAL框架，用于评估多模态代理的端到端推理和错误传播。解决多模态代理测试中的误差分析问题，引入新指标提升对话有效性评估。**

- **链接: [https://arxiv.org/pdf/2601.07367v1](https://arxiv.org/pdf/2601.07367v1)**

> **作者:** Aditya Choudhary; Anupam Purwar
>
> **备注:** We present a framework for evaluation of Multi-modal Agents consisting of Voice-to-voice model components viz. Text to Speech (TTS), Retrieval Augmented Generation (RAG) and Speech-to-text (STT)
>
> **摘要:** With the recent advancements in reasoning capa- bilities, tool calling using MCP servers and Audio Language Models (ALMs), development and integration of multi-modal agents (with voice and text support) has come to the industry forefront. Cascading pipelines for voice agents still play a central role in the industry owing to their superior reasoning capabilities facilitated by LLMs. Although, cascading pipelines often present error propagation through the pipeline. We propose a framework, FOCAL to benchmark end-to-end reasoning, component-wise error propagation and error analysis for automated as well as human-assisted testing of multi-modal agents (voice to voice + text input). We also share two novel metrics viz. Reasoning and Semantic scores to evaluate efficacy of the agent in having meaningful conversations in voice mode.
>
---
#### [new 010] Bridging Attribution and Open-Set Detection using Graph-Augmented Instance Learning in Synthetic Speech
- **分类: eess.AS**

- **简介: 该论文属于合成语音溯源与开放集检测任务，旨在解决合成语音来源识别及未知生成器检测问题。提出SIGNAL框架，结合图神经网络和KNN实现有效分析与检测。**

- **链接: [https://arxiv.org/pdf/2601.07064v1](https://arxiv.org/pdf/2601.07064v1)**

> **作者:** Mohd Mujtaba Akhtar; Girish; Farhan Sheth; Muskaan Singh
>
> **备注:** Accepted to EACL 2026
>
> **摘要:** We propose a unified framework for not only attributing synthetic speech to its source but also for detecting speech generated by synthesizers that were not encountered during training. This requires methods that move beyond simple detection to support both detailed forensic analysis and open-set generalization. To address this, we introduce SIGNAL, a hybrid framework that combines speech foundation models (SFMs) with graph-based modeling and open-set-aware inference. Our framework integrates Graph Neural Networks (GNNs) and a k-Nearest Neighbor (KNN) classifier, allowing it to capture meaningful relationships between utterances and recognize speech that doesn`t belong to any known generator. It constructs a query-conditioned graph over generator class prototypes, enabling the GNN to reason over relationships among candidate generators, while the KNN branch supports open-set detection via confidence-based thresholding. We evaluate SIGNAL using the DiffSSD dataset, which offers a diverse mix of real speech and synthetic audio from both open-source and commercial diffusion-based TTS systems. To further assess generalization, we also test on the SingFake benchmark. Our results show that SIGNAL consistently improves performance across both tasks, with Mamba-based embeddings delivering especially strong results. To the best of our knowledge, this is the first study to unify graph-based learning and open-set detection for tracing synthetic speech back to its origin.
>
---
#### [new 011] Lightweight Resolution-Aware Audio Deepfake Detection via Cross-Scale Attention and Consistency Learning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决复杂环境下检测音频伪造的问题。通过跨尺度注意力和一致性学习，构建轻量级框架，提升检测性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.06560v1](https://arxiv.org/pdf/2601.06560v1)**

> **作者:** K. A. Shahriar
>
> **摘要:** Audio deepfake detection has become increasingly challenging due to rapid advances in speech synthesis and voice conversion technologies, particularly under channel distortions, replay attacks, and real-world recording conditions. This paper proposes a resolution-aware audio deepfake detection framework that explicitly models and aligns multi-resolution spectral representations through cross-scale attention and consistency learning. Unlike conventional single-resolution or implicit feature-fusion approaches, the proposed method enforces agreement across complementary time--frequency scales. The proposed framework is evaluated on three representative benchmarks: ASVspoof 2019 (LA and PA), the Fake-or-Real (FoR) dataset, and the In-the-Wild Audio Deepfake dataset under a speaker-disjoint protocol. The method achieves near-perfect performance on ASVspoof LA (EER 0.16%), strong robustness on ASVspoof PA (EER 5.09%), FoR rerecorded audio (EER 4.54%), and in-the-wild deepfakes (AUC 0.98, EER 4.81%), significantly outperforming single-resolution and non-attention baselines under challenging conditions. The proposed model remains lightweight and efficient, requiring only 159k parameters and less than 1~GFLOP per inference, making it suitable for practical deployment. Comprehensive ablation studies confirm the critical contributions of cross-scale attention and consistency learning, while gradient-based interpretability analysis reveals that the model learns resolution-consistent and semantically meaningful spectral cues across diverse spoofing conditions. These results demonstrate that explicit cross-resolution modeling provides a principled, robust, and scalable foundation for next-generation audio deepfake detection systems.
>
---
#### [new 012] The ICASSP 2026 Automatic Song Aesthetics Evaluation Challenge
- **分类: eess.AS; cs.SD**

- **简介: 该论文介绍ICASSP 2026自动歌曲美学评估挑战，旨在预测AI生成歌曲的主观美学评分。任务包括整体音乐性评分和五个细粒度评分，推动人机审美对齐的评估方法。**

- **链接: [https://arxiv.org/pdf/2601.07237v1](https://arxiv.org/pdf/2601.07237v1)**

> **作者:** Guobin Ma; Yuxuan Xia; Jixun Yao; Huixin Xue; Hexin Liu; Shuai Wang; Hao Liu; Lei Xie
>
> **备注:** Official summary paper for the ICASSP 2026 ASAE Challenge
>
> **摘要:** This paper summarizes the ICASSP 2026 Automatic Song Aesthetics Evaluation (ASAE) Challenge, which focuses on predicting the subjective aesthetic scores of AI-generated songs. The challenge consists of two tracks: Track 1 targets the prediction of the overall musicality score, while Track 2 focuses on predicting five fine-grained aesthetic scores. The challenge attracted strong interest from the research community and received numerous submissions from both academia and industry. Top-performing systems significantly surpassed the official baseline, demonstrating substantial progress in aligning objective metrics with human aesthetic preferences. The outcomes establish a standardized benchmark and advance human-aligned evaluation methodologies for modern music generation systems.
>
---
#### [new 013] Representing Sounds as Neural Amplitude Fields: A Benchmark of Coordinate-MLPs and A Fourier Kolmogorov-Arnold Framework
- **分类: cs.SD**

- **简介: 该论文属于音频信号表示任务，解决Coordinate-MLPs在音频中应用的挑战，提出Fourier-ASR框架提升音频表示效果。**

- **链接: [https://arxiv.org/pdf/2601.06406v1](https://arxiv.org/pdf/2601.06406v1)**

> **作者:** Linfei Li; Lin Zhang; Zhong Wang; Fengyi Zhang; Zelin Li; Ying Shen
>
> **备注:** Accepted by AAAI 2025. Code: https://github.com/lif314/Fourier-ASR
>
> **摘要:** Although Coordinate-MLP-based implicit neural representations have excelled in representing radiance fields, 3D shapes, and images, their application to audio signals remains underexplored. To fill this gap, we investigate existing implicit neural representations, from which we extract 3 types of positional encoding and 16 commonly used activation functions. Through combinatorial design, we establish the first benchmark for Coordinate-MLPs in audio signal representations. Our benchmark reveals that Coordinate-MLPs require complex hyperparameter tuning and frequency-dependent initialization, limiting their robustness. To address these issues, we propose Fourier-ASR, a novel framework based on the Fourier series theorem and the Kolmogorov-Arnold representation theorem. Fourier-ASR introduces Fourier Kolmogorov-Arnold Networks (Fourier-KAN), which leverage periodicity and strong nonlinearity to represent audio signals, eliminating the need for additional positional encoding. Furthermore, a Frequency-adaptive Learning Strategy (FaLS) is proposed to enhance the convergence of Fourier-KAN by capturing high-frequency components and preventing overfitting of low-frequency signals. Extensive experiments conducted on natural speech and music datasets reveal that: (1) well-designed positional encoding and activation functions in Coordinate-MLPs can effectively improve audio representation quality; and (2) Fourier-ASR can robustly represent complex audio signals without extensive hyperparameter tuning. Looking ahead, the continuity and infinite resolution of implicit audio representations make our research highly promising for tasks such as audio compression, synthesis, and generation. The source code will be released publicly to ensure reproducibility. The code is available at https://github.com/lif314/Fourier-ASR.
>
---
#### [new 014] Directional reflection modeling via wavenumber-domain reflection coefficient for 3D acoustic field simulation
- **分类: eess.AS**

- **简介: 该论文属于声场模拟任务，解决方向依赖反射建模问题。通过波数域反射系数构建声导纳模型，实现更精确的三维声反射与散射模拟。**

- **链接: [https://arxiv.org/pdf/2601.07481v1](https://arxiv.org/pdf/2601.07481v1)**

> **作者:** Satoshi Hoshika; Takahiro Iwami; Akira Omoto
>
> **备注:** Submitted to Proceedings of Meetings on Acoustics (PoMA)
>
> **摘要:** This study proposes a framework for incorporating wavenumber-domain acoustic reflection coefficients into sound field analysis to characterize direction-dependent material reflection and scattering phenomena. The reflection coefficient is defined as the amplitude ratio between incident and reflected waves for each propagation direction and is estimated from spatial Fourier transforms of the incident and reflected sound fields. The resulting wavenumber-domain reflection coefficients are converted into an acoustic admittance representation that is directly compatible with numerical methods such as the Boundary Element Method (BEM), enabling simulation of reflections beyond simple specular components. Unlike conventional extended reaction models, the proposed approach avoids explicit modeling of the material interior. This significantly reduces computational cost while allowing direct use of measured data, empirical models, or user-defined directional reflection characteristics. The validity of the proposed formulation was previously demonstrated by the authors through two-dimensional sound field simulations, in which accurate reproduction of direction-dependent reflection behavior was confirmed. In the present work, the framework is extended to three-dimensional analysis, demonstrating its applicability to more realistic and complex acoustic environments. The proposed approach provides a practical and flexible tool for simulating direction-dependent acoustic reflections and scattering, with potential applications in architectural acoustics, material characterization, and noise control.
>
---
#### [new 015] TagSpeech: End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出TagSpeech，解决多说话人语音识别与说话人分离任务，通过联合建模实现“谁在何时说了什么”的端到端系统。**

- **链接: [https://arxiv.org/pdf/2601.06896v1](https://arxiv.org/pdf/2601.06896v1)**

> **作者:** Mingyue Huo; Yiwen Shao; Yuheng Zhang
>
> **摘要:** We present TagSpeech, a unified LLM-based framework that utilizes Temporal Anchor Grounding for joint multi-speaker ASR and diarization. The framework is built on two key designs: (1) decoupled semantic and speaker streams fine-tuned via Serialized Output Training (SOT) to learn turn-taking dynamics; and (2) an interleaved time anchor mechanism that not only supports fine-grained timestamp prediction but also acts as a synchronization signal between semantic understanding and speaker tracking. Compared to previous works that primarily focus on speaker-attributed ASR or implicit diarization, TagSpeech addresses the challenge of fine-grained speaker-content alignment and explicitly models "who spoke what and when" in an end-to-end manner. Experiments on AMI and AliMeeting benchmarks demonstrate that our method achieves consistent improvements in Diarization Error Rate (DER) over strong end-to-end baselines, including Qwen-Omni and Gemini, particularly in handling complex speech overlaps. Moreover, TagSpeech employs a parameter-efficient training paradigm in which the LLM backbone is frozen and only lightweight projectors are trained, resulting in strong performance with low computational cost.
>
---
#### [new 016] FastSLM: Hierarchical Frame Q-Former for Effective Speech Modality Adaptation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出FastSLM，解决语音模态适应问题，通过Hierarchical Frame Q-Former和三阶段训练策略，实现高效语音理解与推理。**

- **链接: [https://arxiv.org/pdf/2601.06199v1](https://arxiv.org/pdf/2601.06199v1)**

> **作者:** Junseok Lee; Sangyong Lee; Chang-Jae Chun
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated human-expert-level capabilities, driving significant interest in their potential for achieving artificial general intelligence (AGI). In particular, there is growing momentum in adapting LLMs to various modalities, including vision, video, and speech, through the development of multimodal LLMs (MLLMs). However, existing speech-language model (SLM) research has largely overlooked cost-effective adaptation strategies for leveraging LLMs in the speech domain. In this paper, we propose FastSLM, a lightweight yet efficient SLM designed for effective understanding and reasoning over long-form speech. To address the challenge of aligning high-frame-rate speech features with LLMs, we introduce the Hierarchical Frame Querying Transformer (HFQ-Former), which compresses frame-level speech features while capturing both local and global context. Furthermore, we present a novel three-stage training strategy that enhances generalization across a wide range of speech-related tasks. Experimental results demonstrate that FastSLM achieves competitive performance compared to existing state-of-the-art models, despite operating with significantly lower FLOPs and parameter counts, while representing speech with only 1.67 tokens per second. The source code and model checkpoints are available at https://huggingface.co/okestro-ai-lab/FastSLM.
>
---
#### [new 017] MoEScore: Mixture-of-Experts-Based Text-Audio Relevance Score Prediction for Text-to-Audio System Evaluation
- **分类: cs.SD**

- **简介: 该论文属于文本-音频生成系统评估任务，旨在解决TTA系统语义一致性评估难题。提出基于MoE和SeqCoAttn的客观评价模型，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.06829v1](https://arxiv.org/pdf/2601.06829v1)**

> **作者:** Bochao Sun; Yang Xiao; Han Yin
>
> **摘要:** Recent advances in generative models have enabled modern Text-to-Audio (TTA) systems to synthesize audio with high perceptual quality. However, TTA systems often struggle to maintain semantic consistency with the input text, leading to mismatches in sound events, temporal tructures, or contextual relationships. Evaluating semantic fidelity in TTA remains a significant challenge. Traditional methods primarily rely on subjective human listening tests, which is time-consuming. To solve this, we propose an objective evaluator based on a Mixture of Experts (MoE) architecture with Sequential Cross-Attention (SeqCoAttn). Our model achieves the first rank in the XACLE Challenge, with an SRCC of 0.6402 (an improvement of 30.6% over the challenge baseline) on the test dataset. Code is available at: https://github.com/S-Orion/MOESCORE.
>
---
#### [new 018] AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出AZeroS，解决将大语言模型扩展到语音领域的问题。通过自生成无指令微调（SIFT），无需任务特定数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.06086v1](https://arxiv.org/pdf/2601.06086v1)**

> **作者:** Yiwen Shao; Wei Liu; Jiahong Li; Tianzi Wang; Kun Wei; Meng Yu; Dong Yu
>
> **备注:** Technical Report
>
> **摘要:** Extending large language models (LLMs) to the speech domain has recently gained significant attention. A typical approach connects a pretrained LLM with an audio encoder through a projection module and trains the resulting model on large-scale, task-specific instruction-tuning datasets. However, curating such instruction-tuning data for specific requirements is time-consuming, and models trained in this manner often generalize poorly to unseen tasks. In this work, we first formulate that the strongest generalization of a speech-LLM is achieved when it is trained with Self-Generated Instruction-Free Tuning (SIFT), in which supervision signals are generated by a frozen LLM using textual representations of speech as input. Our proposed SIFT paradigm eliminates the need for collecting task-specific question-answer pairs and yields the theoretically best generalization to unseen tasks. Building upon this paradigm, we introduce AZeroS (Auden Zero-instruction-tuned Speech-LLM), which is trained on speech-text pairs derived from publicly available corpora, including approximately 25,000 hours of speech with ASR transcripts and 3,000 hours of speech with paralinguistic labels. Built upon Qwen2.5-7B-Instruct, the model updates only two lightweight projection modules (23.8 million parameters each), while keeping both the LLM and audio encoders frozen. Despite the minimal training cost and modest data scale, AZeroS achieves state-of-the-art performance on both semantic and paralinguistic benchmarks, including VoiceBench, AIR-Bench Foundation (Speech), and AIR-Bench Chat (Speech).
>
---
#### [new 019] Variational decomposition autoencoding improves disentanglement of latent representations
- **分类: cs.LG; cs.AI; eess.AS; eess.SP; stat.ML**

- **简介: 该论文属于表示学习任务，旨在解决复杂信号中潜在表征的解耦问题。提出VDA框架，通过信号分解增强VAE，提升表征的可解释性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.06844v1](https://arxiv.org/pdf/2601.06844v1)**

> **作者:** Ioannis Ziogas; Aamna Al Shehhi; Ahsan H. Khandoker; Leontios J. Hadjileontiadis
>
> **备注:** Supplementary information file at: https://drive.google.com/drive/folders/1sZl2AcCtRK-1oav7XZSaxlu0Cq0-3MMs?usp=sharing
>
> **摘要:** Understanding the structure of complex, nonstationary, high-dimensional time-evolving signals is a central challenge in scientific data analysis. In many domains, such as speech and biomedical signal processing, the ability to learn disentangled and interpretable representations is critical for uncovering latent generative mechanisms. Traditional approaches to unsupervised representation learning, including variational autoencoders (VAEs), often struggle to capture the temporal and spectral diversity inherent in such data. Here we introduce variational decomposition autoencoding (VDA), a framework that extends VAEs by incorporating a strong structural bias toward signal decomposition. VDA is instantiated through variational decomposition autoencoders (DecVAEs), i.e., encoder-only neural networks that combine a signal decomposition model, a contrastive self-supervised task, and variational prior approximation to learn multiple latent subspaces aligned with time-frequency characteristics. We demonstrate the effectiveness of DecVAEs on simulated data and three publicly available scientific datasets, spanning speech recognition, dysarthria severity evaluation, and emotional speech classification. Our results demonstrate that DecVAEs surpass state-of-the-art VAE-based methods in terms of disentanglement quality, generalization across tasks, and the interpretability of latent encodings. These findings suggest that decomposition-aware architectures can serve as robust tools for extracting structured representations from dynamic signals, with potential applications in clinical diagnostics, human-computer interaction, and adaptive neurotechnologies.
>
---
## 更新

#### [replaced 001] Confidence-Based Self-Training for EMG-to-Speech: Leveraging Synthetic EMG for Robust Modeling
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于EMG到语音的重建任务，解决数据稀缺问题。通过自训练方法和合成数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.11862v2](https://arxiv.org/pdf/2506.11862v2)**

> **作者:** Xiaodan Chen; Xiaoxue Gao; Mathias Quoy; Alexandre Pitti; Nancy F. Chen
>
> **备注:** Version 2: Updated with acceptance notice for the IEEE Automatic Speech Recognition and Understanding (ASRU) Workshop 2025. Minor revisions from the review process incorporated. This is the camera-ready manuscript version
>
> **摘要:** Voiced Electromyography (EMG)-to-Speech (V-ETS) models reconstruct speech from muscle activity signals, facilitating applications such as neurolaryngologic diagnostics. Despite its potential, the advancement of V-ETS is hindered by a scarcity of paired EMG-speech data. To address this, we propose a novel Confidence-based Multi-Speaker Self-training (CoM2S) approach, along with a newly curated Libri-EMG dataset. This approach leverages synthetic EMG data generated by a pre-trained model, followed by a proposed filtering mechanism based on phoneme-level confidence to enhance the ETS model through the proposed self-training techniques. Experiments demonstrate our method improves phoneme accuracy, reduces phonological confusion, and lowers word error rate, confirming the effectiveness of our CoM2S approach for V-ETS. In support of future research, we will release the codes and the proposed Libri-EMG dataset-an open-access, time-aligned, multi-speaker voiced EMG and speech recordings.
>
---
#### [replaced 002] Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音频安全领域，旨在解决LALMs的越狱攻击问题。构建了Jailbreak-AudioBench工具集、数据集和基准，评估并分析音频越狱威胁。**

- **链接: [https://arxiv.org/pdf/2501.13772v4](https://arxiv.org/pdf/2501.13772v4)**

> **作者:** Hao Cheng; Erjia Xiao; Jing Shao; Yichi Wang; Le Yang; Chao Shen; Philip Torr; Jindong Gu; Renjing Xu
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant safety problems, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also various editing techniques for injecting audio hidden semantics. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs and establish the most comprehensive Jailbreak benchmark to date for audio modality. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.
>
---
#### [replaced 003] Memory-Efficient Training for Text-Dependent SV with Independent Pre-trained Models
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于文本依赖说话人验证任务，旨在解决传统方法计算成本高和模型适应性差的问题。通过独立使用预训练模型并进行领域适配，实现了高效且准确的验证系统。**

- **链接: [https://arxiv.org/pdf/2411.10828v2](https://arxiv.org/pdf/2411.10828v2)**

> **作者:** Seyed Ali Farokh; Hossein Zeinali
>
> **备注:** Accepted at ROCLING 2025
>
> **摘要:** This paper presents our submission to the Iranian division of the Text-Dependent Speaker Verification Challenge (TdSV) 2024. Conventional TdSV approaches typically jointly model speaker and linguistic features, requiring unsegmented inputs during training and incurring high computational costs. Additionally, these methods often fine-tune large-scale pre-trained speaker embedding models on the target domain dataset, which may compromise the pre-trained models' original ability to capture speaker-specific characteristics. To overcome these limitations, we employ a TdSV system that utilizes two pre-trained models independently and demonstrate that, by leveraging pre-trained models with targeted domain adaptation, competitive results can be achieved while avoiding the substantial computational costs associated with joint fine-tuning on unsegmented inputs in conventional approaches. Our best system reached a MinDCF of 0.0358 on the evaluation subset and secured first place in the challenge.
>
---
#### [replaced 004] MMMOS: Multi-domain Multi-axis Audio Quality Assessment
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于音频质量评估任务，解决现有模型无法多维度评估不同音频类型的问题。提出MMMOS系统，从四个维度评估音频质量。**

- **链接: [https://arxiv.org/pdf/2507.04094v2](https://arxiv.org/pdf/2507.04094v2)**

> **作者:** Yi-Cheng Lin; Jia-Hung Chen; Hung-yi Lee
>
> **备注:** 4 pages including 1 page of reference. Accepted by ASRU Audio MOS 2025 Challenge
>
> **摘要:** Accurate audio quality estimation is essential for developing and evaluating audio generation, retrieval, and enhancement systems. Existing non-intrusive assessment models predict a single Mean Opinion Score (MOS) for speech, merging diverse perceptual factors and failing to generalize beyond speech. We propose MMMOS, a no-reference, multi-domain audio quality assessment system that estimates four orthogonal axes: Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness across speech, music, and environmental sounds. MMMOS fuses frame-level embeddings from three pretrained encoders (WavLM, MuQ, and M2D) and evaluates three aggregation strategies with four loss functions. By ensembling the top eight models, MMMOS shows a 20-30% reduction in mean squared error and a 4-5% increase in Kendall's τ versus baseline, gains first place in six of eight Production Complexity metrics, and ranks among the top three on 17 of 32 challenge metrics.
>
---
#### [replaced 005] Speak the Art: A Direct Speech to Image Generation Framework
- **分类: eess.AS; cs.AI; cs.MM**

- **简介: 该论文属于语音到图像生成任务，旨在解决现有方法在语义表示和生成稳定性上的不足。提出STA框架，结合语音编码与VQ-Diffusion网络，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2601.00827v2](https://arxiv.org/pdf/2601.00827v2)**

> **作者:** Mariam Saeed; Manar Amr; Farida Adel; Nada Hassan; Nour Walid; Eman Mohamed; Mohamed Hussein; Marwan Torki
>
> **摘要:** Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called Speak the Art (STA) which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin.
>
---
#### [replaced 006] A Comprehensive Study on the Effectiveness of ASR Representations for Noise-Robust Speech Emotion Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于噪声鲁棒语音情感识别任务，旨在解决真实环境中非平稳噪声对情感识别的影响。通过引入ASR模型作为特征提取器，提升情感识别性能。**

- **链接: [https://arxiv.org/pdf/2311.07093v4](https://arxiv.org/pdf/2311.07093v4)**

> **作者:** Xiaohan Shi; Jiajun He; Xingfeng Li; Tomoki Toda
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech, and Language Processing
>
> **摘要:** This paper proposes an efficient attempt to noisy speech emotion recognition (NSER). Conventional NSER approaches have proven effective in mitigating the impact of artificial noise sources, such as white Gaussian noise, but are limited to non-stationary noises in real-world environments due to their complexity and uncertainty. To overcome this limitation, we introduce a new method for NSER by adopting the automatic speech recognition (ASR) model as a noise-robust feature extractor to eliminate non-vocal information in noisy speech. We first obtain intermediate layer information from the ASR model as a feature representation for emotional speech and then apply this representation for the downstream NSER task. Our experimental results show that 1) the proposed method achieves better NSER performance compared with the conventional noise reduction method, 2) outperforms self-supervised learning approaches, and 3) even outperforms text-based approaches using ASR transcription or the ground truth transcription of noisy speech.
>
---
#### [replaced 007] SIGNL: A Label-Efficient Audio Deepfake Detection System via Spectral-Temporal Graph Non-Contrastive Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决标注数据不足的问题。提出SIGNL系统，通过频时图非对比学习，有效提取音频特征，实现高效检测。**

- **链接: [https://arxiv.org/pdf/2501.04942v2](https://arxiv.org/pdf/2501.04942v2)**

> **作者:** Falih Gozi Febrinanto; Kristen Moore; Chandra Thapa; Jiangang Ma; Vidya Saikrishna
>
> **摘要:** Audio deepfake detection is increasingly important as synthetic speech becomes more realistic and accessible. Recent methods, including those using graph neural networks (GNNs) to model frequency and temporal dependencies, show strong potential but need large amounts of labeled data, which limits their practical use. Label-efficient alternatives like graph-based non-contrastive learning offer a potential solution, as they can learn useful representations from unlabeled data without using negative samples. However, current graph non-contrastive approaches are built for single-view graph representations and cannot be directly used for audio, which has unique spectral and temporal structures. Bridging this gap requires dual-view graph modeling suited to audio signals. In this work, we introduce SIGNL (Spectral-temporal vIsion Graph Non-contrastive Learning), a label-efficient expert system for detecting audio deepfakes. SIGNL operates on the visual representation of audio, such as spectrograms or other time-frequency encodings, transforming them into spectral and temporal graphs for structured feature extraction. It then employs graph convolutional encoders to learn complementary frequency-time features, effectively capturing the unique characteristics of audio. These encoders are pre-trained using a non-contrastive self-supervised learning strategy on augmented graph pairs, enabling effective representation learning without labeled data. The resulting encoders are then fine-tuned on minimal labelled data for downstream deepfake detection. SIGNL achieves strong performance on multiple audio deepfake detection benchmarks, including 7.88% EER on ASVspoof 2021 DF and 3.95% EER on ASVspoof 5 using only 5% labeled data. It also generalizes well to unseen conditions, reaching 10.16% EER on the In-The-Wild dataset when trained on CFAD.
>
---
#### [replaced 008] Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于可控长歌词生成任务，旨在解决学术研究不可复现的问题。工作包括发布开源系统Muse及合成数据集，实现细粒度风格控制的歌曲生成。**

- **链接: [https://arxiv.org/pdf/2601.03973v3](https://arxiv.org/pdf/2601.03973v3)**

> **作者:** Changhao Jiang; Jiahao Chen; Zhenghao Xiang; Zhixiong Yang; Hanchen Wang; Jiabao Zhuang; Xinmeng Che; Jiajun Sun; Hui Li; Yifei Cao; Shihan Dou; Ming Zhang; Junjie Ye; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at https://github.com/yuhui1038/Muse.
>
---
#### [replaced 009] TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出TELEVAL，一个面向中文口语交互场景的动态评估基准，解决现有评估标准与真实对话不匹配的问题。工作包括构建内容准确性和互动恰当性两个核心评估维度。**

- **链接: [https://arxiv.org/pdf/2507.18061v3](https://arxiv.org/pdf/2507.18061v3)**

> **作者:** Zehan Li; Hongjie Chen; Qing Wang; Yuxin Zhang; Jing Zhou; Hang Lv; Mengjie Du; Yaodong Song; Jie Lian; Jian Kang; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** Spoken language models (SLMs) have advanced rapidly in recent years, accompanied by a growing number of evaluation benchmarks. However, most existing benchmarks emphasize task completion and capability scaling, while remaining poorly aligned with how users interact with SLMs in real-world spoken conversations. Effective spoken interaction requires not only accurate understanding of user intent and content, but also the ability to respond with appropriate interactional strategies. In this paper, we present TELEVAL, a dynamic, user-centered benchmark for evaluating SLMs in realistic Chinese spoken interaction scenarios. TELEVAL consolidates evaluation into two core aspects. Reliable Content Fulfillment assesses whether models can comprehend spoken inputs and produce semantically correct responses. Interactional Appropriateness evaluates whether models act as socially capable interlocutors, requiring them not only to generate human-like, colloquial responses, but also to implicitly incorporate paralinguistic cues for natural interaction. Experiments reveal that, despite strong performance on semantic and knowledge-oriented tasks, current SLMs still struggle to produce natural and interactionally appropriate responses, highlighting the need for more interaction-faithful evaluation.
>
---
#### [replaced 010] Accelerated Interactive Auralization of Highly Reverberant Spaces using Graphics Hardware
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频渲染任务，旨在解决高混响空间实时声学模拟的计算瓶颈。通过GPU加速实现低延迟的多通道声场合成与反馈抑制。**

- **链接: [https://arxiv.org/pdf/2509.04390v2](https://arxiv.org/pdf/2509.04390v2)**

> **作者:** Hannes Rosseel; Toon van Waterschoot
>
> **备注:** 9 pages, 6 figures, submitted to Journal of the Audio Engineering Society
>
> **摘要:** Interactive acoustic auralization allows users to explore virtual acoustic environments in real-time, enabling the acoustic recreation of concert hall or Historical Worship Spaces (HWS) that are either no longer accessible, acoustically altered, or impractical to visit. Interactive acoustic synthesis requires real-time convolution of input signals with a set of synthesis filters that model the space-time acoustic response of the space. The acoustics in concert halls and HWS are both characterized by a long reverberation time, resulting in synthesis filters containing many filter taps. As a result, the convolution process can be computationally demanding, introducing significant latency that limits the real-time interactivity of the auralization system. In this paper, the implementation of a real-time multichannel loudspeaker-based auralization system is presented. This system is capable of synthesizing the acoustics of highly reverberant spaces in real-time using GPU-acceleration. A comparison between traditional CPU-based convolution and GPU-accelerated convolution is presented, showing that the latter can achieve real-time performance with significantly lower latency. Additionally, the system integrates acoustic synthesis with acoustic feedback cancellation on the GPU, creating a unified loudspeaker-based auralization framework that minimizes processing latency.
>
---
#### [replaced 011] From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于音频-语言对齐任务，旨在解决ALLM的遗忘问题和数据资源消耗大问题。通过合成数据生成框架BALSa，提升模型区分有无声音的能力及多音频理解能力。**

- **链接: [https://arxiv.org/pdf/2505.20166v3](https://arxiv.org/pdf/2505.20166v3)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. This adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where crucial textual capabilities like instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making it resource-intensive. To address these issues, previous works have leveraged the backbone LLMs to synthesize general-purpose, caption-style alignment data. In this paper, we propose a data generation framework that produces contrastive-like training data, designed to enhance ALLMs' ability to differentiate between present and absent sounds. We further extend our approach to multi-audio scenarios, enabling the model to either explain differences between audio inputs or produce unified captions that describe all inputs, thereby enhancing audio-language alignment. We refer to the entire ALLM training framework as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance on audio understanding and reasoning benchmarks, as well as instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to developing ALLMs.
>
---
#### [replaced 012] MMEDIT: A Unified Framework for Multi-Type Audio Editing via Audio Language Model
- **分类: cs.SD**

- **简介: 该论文提出MMEdit，解决文本引导的音频编辑任务，针对现有方法在数据稀缺和编辑操作有限等问题，设计统一框架和数据合成方法，提升编辑精度与指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2512.20339v2](https://arxiv.org/pdf/2512.20339v2)**

> **作者:** Ye Tao; Wen Wu; Chao Zhang; Mengyue Wu; Shuai Wang; Xuenan Xu
>
> **备注:** Under review
>
> **摘要:** Text-guided audio editing aims to modify specific acoustic events while strictly preserving non-target content. Despite recent progress, existing approaches remain fundamentally limited. Training-free methods often suffer from signal degradation caused by diffusion inversion, while training-based methods, although achieving higher generation quality, are severely constrained by the scarcity of high-quality paired data and task formulations that cover only a narrow subset of editing operations. In addition, standard architectures typically decouple text and audio processing, limiting the ability to align instructions with specific acoustic contexts. To address these challenges, we propose MMEdit, an audio-language-model-driven framework for unified audio editing. We systematically extend task definitions to cover a comprehensive range of editing operations, including addition, replacement, removal, reordering, and attribute modification. Furthermore, we design a scalable data synthesis pipeline to construct large-scale paired datasets with fine-grained event-level annotations. To capture complex editing semantics, we integrate a Qwen2-Audio encoder with an MMDiT-based generator, enabling precise cross-modal alignment and localized editing. Experimental results demonstrate that our method achieves superior editing localization accuracy, robust instruction following, and high fidelity in non-edited regions.
>
---
#### [replaced 013] Omni2Sound: Towards Unified Video-Text-to-Audio Generation
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文属于视频-文本到音频生成任务，解决数据稀缺和多任务竞争问题，提出SoundAtlas数据集和Omni2Sound模型，实现统一高效生成。**

- **链接: [https://arxiv.org/pdf/2601.02731v2](https://arxiv.org/pdf/2601.02731v2)**

> **作者:** Yusheng Dai; Zehua Chen; Yuxuan Jiang; Baolong Gao; Qiuhong Ke; Jun Zhu; Jianfei Cai
>
> **摘要:** Training a unified model integrating video-to-audio (V2A), text-to-audio (T2A), and joint video-text-to-audio (VT2A) generation offers significant application flexibility, yet faces two unexplored foundational challenges: (1) the scarcity of high-quality audio captions with tight A-V-T alignment, leading to severe semantic conflict between multimodal conditions, and (2) cross-task and intra-task competition, manifesting as an adverse V2A-T2A performance trade-off and modality bias in the VT2A task. First, to address data scarcity, we introduce SoundAtlas, a large-scale dataset (470k pairs) that significantly outperforms existing benchmarks and even human experts in quality. Powered by a novel agentic pipeline, it integrates Vision-to-Language Compression to mitigate visual bias of MLLMs, a Junior-Senior Agent Handoff for a 5 times cost reduction, and rigorous Post-hoc Filtering to ensure fidelity. Consequently, SoundAtlas delivers semantically rich and temporally detailed captions with tight V-A-T alignment. Second, we propose Omni2Sound, a unified VT2A diffusion model supporting flexible input modalities. To resolve the inherent cross-task and intra-task competition, we design a three-stage multi-task progressive training schedule that converts cross-task competition into joint optimization and mitigates modality bias in the VT2A task, maintaining both audio-visual alignment and off-screen audio generation faithfulness. Finally, we construct VGGSound-Omni, a comprehensive benchmark for unified evaluation, including challenging off-screen tracks. With a standard DiT backbone, Omni2Sound achieves unified SOTA performance across all three tasks within a single model, demonstrating strong generalization across benchmarks with heterogeneous input conditions. The project page is at https://swapforward.github.io/Omni2Sound.
>
---
#### [replaced 014] A dataset and model for auditory scene recognition for hearing devices: AHEAD-DS and OpenYAMNet
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于听觉场景识别任务，旨在解决现有数据集不足和模型部署困难的问题。通过构建AHEAD-DS数据集和提出OpenYAMNet模型，提升助听设备的场景识别能力。**

- **链接: [https://arxiv.org/pdf/2508.10360v4](https://arxiv.org/pdf/2508.10360v4)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard Lyon
>
> **摘要:** Scene recognition is important for hearing devices, however; this is challenging, in part because of the limitations of existing datasets. Datasets often lack public accessibility, completeness, or audiologically relevant labels, hindering systematic comparison of machine learning models. Deploying such models on resource-constrained edge devices presents another challenge.The proposed solution is two-fold, a repack and refinement of several open source datasets to create AHEAD-DS, a dataset designed for auditory scene recognition for hearing devices, and introduce OpenYAMNet, a sound recognition model. AHEAD-DS aims to provide a standardised, publicly available dataset with consistent labels relevant to hearing aids, facilitating model comparison. OpenYAMNet is designed for deployment on edge devices like smartphones connected to hearing devices, such as hearing aids and wireless earphones with hearing aid functionality, serving as a baseline model for sound-based scene recognition. OpenYAMNet achieved a mean average precision of 0.86 and accuracy of 0.93 on the testing set of AHEAD-DS across fourteen categories relevant to auditory scene recognition. Real-time sound-based scene recognition capabilities were demonstrated on edge devices by deploying OpenYAMNet to an Android smartphone. Even with a 2018 Google Pixel 3, a phone with modest specifications, the model processes audio with approximately 50ms of latency to load the model, and an approximate linear increase of 30ms per 1 second of audio. The project website with links to code, data, and models. \href{https://github.com/Australian-Future-Hearing-Initiative}{https://github.com/Australian-Future-Hearing-Initiative}
>
---
#### [replaced 015] Word-Level Emotional Expression Control in Zero-Shot Text-to-Speech Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决零样本场景下单词级情感控制问题。通过自训练框架WeSCon，实现无需句内情感标注数据的单词级情感与语速控制。**

- **链接: [https://arxiv.org/pdf/2509.24629v2](https://arxiv.org/pdf/2509.24629v2)**

> **作者:** Tianrui Wang; Haoyu Wang; Meng Ge; Cheng Gong; Chunyu Qiang; Ziyang Ma; Zikang Huang; Guanrou Yang; Xiaobao Wang; Eng Siong Chng; Xie Chen; Longbiao Wang; Jianwu Dang
>
> **摘要:** While emotional text-to-speech (TTS) has made significant progress, most existing research remains limited to utterance-level emotional expression and fails to support word-level control. Achieving word-level expressive control poses fundamental challenges, primarily due to the complexity of modeling multi-emotion transitions and the scarcity of annotated datasets that capture intra-sentence emotional and prosodic variation. In this paper, we propose WeSCon, the first self-training framework that enables word-level control of both emotion and speaking rate in a pretrained zero-shot TTS model, without relying on datasets containing intra-sentence emotion or speed transitions. Our method introduces a transition-smoothing strategy and a dynamic speed control mechanism to guide the pretrained TTS model in performing word-level expressive synthesis through a multi-round inference process. To further simplify the inference, we incorporate a dynamic emotional attention bias mechanism and fine-tune the model via self-training, thereby activating its ability for word-level expressive control in an end-to-end manner. Experimental results show that WeSCon effectively overcomes data scarcity, achieving state-of-the-art performance in word-level emotional expression control while preserving the strong zero-shot synthesis capabilities of the original TTS model.
>
---
