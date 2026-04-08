# 音频 cs.SD;  eess.AS

- **最新发布 12 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Time-Domain Voice Identity Morphing (TD-VIM): A Signal-Level Approach to Morphing Attacks on Speaker Verification Systems
- **分类: cs.SD**

- **简介: 该论文属于语音生物识别安全研究，旨在解决语音身份伪造问题。提出TD-VIM方法，在信号层融合两个身份的语音特征，生成易被欺骗的语音样本，评估其对语音验证系统的威胁。**

- **链接: [https://arxiv.org/pdf/2604.05683](https://arxiv.org/pdf/2604.05683)**

> **作者:** Aravinda Reddy PN; Raghavendra Ramachandra; K.Sreenivasa Rao; Pabitra Mitra; Kunal Singh
>
> **摘要:** In biometric systems, it is a common practice to associate each sample or template with a specific individual. Nevertheless, recent studies have demonstrated the feasibility of generating "morphed" biometric samples capable of matching multiple identities. These morph attacks have been recognized as potential security risks for biometric systems. However, most research on morph attacks has focused on biometric modalities that operate within the image domain, such as the face, fingerprints, and iris. In this work, we introduce Time-domain Voice Identity Morphing (TD-VIM), a novel approach for voice-based biometric morphing. This method enables the blending of voice characteristics from two distinct identities at the signal level, creating morphed samples that present a high vulnerability for speaker verification systems. Leveraging the Multilingual Audio-Visual Smartphone database, our study created four distinct morphed signals based on morphing factors and evaluated their effectiveness using a comprehensive vulnerability analysis. To assess the security impact of TD-VIM, we benchmarked our approach using the Generalized Morphing Attack Potential (G-MAP) metric, measuring attack success across two deep-learning-based Speaker Verification Systems (SVS) and one commercial system, Verispeak. Our findings indicate that the morphed voice samples achieved a high attack success rate, with G-MAP values reaching 99.40% on iPhone-11 and 99.74% on Samsung S8 in text-dependent scenarios, at a false match rate of 0.1%.
>
---
#### [new 002] Generalizable Audio-Visual Navigation via Binaural Difference Attention and Action Transition Prediction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频视觉导航任务，旨在提升模型在未见环境中的泛化能力。提出BDATP框架，通过建模双耳差异和预测动作转移，增强空间定位并减少过拟合。**

- **链接: [https://arxiv.org/pdf/2604.05007](https://arxiv.org/pdf/2604.05007)**

> **作者:** Jia Li; Yinfeng Yu
>
> **备注:** Main paper (6 pages). Accepted for publication by the International Joint Conference on Neural Networks (IJCNN 2026)
>
> **摘要:** In Audio-Visual Navigation (AVN), agents must locate sound sources in unseen 3D environments using visual and auditory cues. However, existing methods often struggle with generalization in unseen scenarios, as they tend to overfit to semantic sound features and specific training environments. To address these challenges, we propose the \textbf{Binaural Difference Attention with Action Transition Prediction (BDATP)} framework, which jointly optimizes perception and policy. Specifically, the \textbf{Binaural Difference Attention (BDA)} module explicitly models interaural differences to enhance spatial orientation, reducing reliance on semantic categories. Simultaneously, the \textbf{Action Transition Prediction (ATP)} task introduces an auxiliary action prediction objective as a regularization term, mitigating environment-specific overfitting. Extensive experiments on the Replica and Matterport3D datasets demonstrate that BDATP can be seamlessly integrated into various mainstream baselines, yielding consistent and significant performance gains. Notably, our framework achieves state-of-the-art Success Rates across most settings, with a remarkable absolute improvement of up to 21.6 percentage points in Replica dataset for unheard sounds. These results underscore BDATP's superior generalization capability and its robustness across diverse navigation architectures.
>
---
#### [new 003] Generating Synthetic Doctor-Patient Conversations for Long-form Audio Summarization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于长文本音频摘要任务，旨在解决长上下文音频推理数据不足和评估困难的问题。通过生成合成对话数据，构建训练与评估环境，并验证模型效果。**

- **链接: [https://arxiv.org/pdf/2604.06138](https://arxiv.org/pdf/2604.06138)**

> **作者:** Yanis Labrak; David Grünert; Séverin Baroudi; Jiyun Chun; Pawel Cyrta; Sergio Burdisso; Ahmed Hassoon; David Liu; Adam Rothschild; Reed Van Deusen; Petr Motlicek; Andrew Perrault; Ricard Marxer; Thomas Schaaf
>
> **备注:** Submitted for review at Interspeech 2026
>
> **摘要:** Long-context audio reasoning is underserved in both training data and evaluation. Existing benchmarks target short-context tasks, and the open-ended generation tasks most relevant to long-context reasoning pose well-known challenges for automatic evaluation. We propose a synthetic data generation pipeline designed to serve both as a training resource and as a controlled evaluation environment, and instantiate it for first-visit doctor-patient conversations with SOAP note generation as the task. The pipeline has three stages, persona-driven dialogue generation, multi-speaker audio synthesis with overlap/pause modeling, room acoustics, and sound events, and LLM-based reference SOAP note production, built entirely on open-weight models. We release 8,800 synthetic conversations with 1.3k hours of corresponding audio and reference notes. Evaluating current open-weight systems, we find that cascaded approaches still substantially outperform end-to-end models.
>
---
#### [new 004] Controllable Singing Style Conversion with Boundary-Aware Information Bottleneck
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于歌唱风格转换任务，解决风格泄露、动态渲染和数据不足问题，提出边界感知瓶颈、帧级技术矩阵和高频补全策略，提升转换自然度与控制性。**

- **链接: [https://arxiv.org/pdf/2604.05526](https://arxiv.org/pdf/2604.05526)**

> **作者:** Zhetao Hu; Yiquan Zhou; Wenyu Wang; Zhiyu Wu; Xin Gao; Jihua Zhu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** This paper presents the submission of the S4 team to the Singing Voice Conversion Challenge 2025 (SVCC2025)-a novel singing style conversion system that advances fine-grained style conversion and control within in-domain settings. To address the critical challenges of style leakage, dynamic rendering, and high-fidelity generation with limited data, we introduce three key innovations: a boundary-aware Whisper bottleneck that pools phoneme-span representations to suppress residual source style while preserving linguistic content; an explicit frame-level technique matrix, enhanced by targeted F0 processing during inference, for stable and distinct dynamic style rendering; and a perceptually motivated high-frequency band completion strategy that leverages an auxiliary standard 48kHz SVC model to augment the high-frequency spectrum, thereby overcoming data scarcity without overfitting. In the official SVCC2025 subjective evaluation, our system achieves the best naturalness performance among all submissions while maintaining competitive results in speaker similarity and technique control, despite using significantly less extra singing data than other top-performing systems. Audio samples are available online.
>
---
#### [new 005] YMIR: A new Benchmark Dataset and Model for Arabic Yemeni Music Genre Classification Using Convolutional Neural Networks
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐流派分类任务，旨在解决西方音乐占主导的现状下，阿拉伯也门传统音乐分类不足的问题。研究构建了YMIR数据集并提出YMCM模型，通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.05011](https://arxiv.org/pdf/2604.05011)**

> **作者:** Moeen AL-Makhlafi; Abdulrahman A. AlKannad; Eiad Almekhlafi; Nawaf Q. Othman Ahmed Mohammed; Saher Qaid
>
> **摘要:** Automatic music genre classification is a major task in music information retrieval; however, most current benchmarks and models have been developed primarily for Western music, leaving culturally specific traditions underrepresented. In this paper, we introduce the Yemeni Music Information Retrieval (YMIR) dataset, which contains 1,475 carefully selected audio clips covering five traditional Yemeni genres: Sanaani, Hadhrami, Lahji, Tihami, and Adeni. The dataset was labeled by five Yemeni music experts following a clear and structured protocol, resulting in strong inter-annotator agreement (Fleiss kappa = 0.85). We also propose the Yemeni Music Classification Model (YMCM), a convolutional neural network (CNN)-based system designed to classify music genres from time-frequency features. Using a consistent preprocessing pipeline, we perform a systematic comparison across six experimental groups and five different architectures, resulting in a total of 30 experiments. Specifically, we evaluate several feature representations, including Mel-spectrograms, Chroma, FilterBank, and MFCCs with 13, 20, and 40 coefficients, and benchmark YMCM against standard models (AlexNet, VGG16, MobileNet, and a baseline CNN) under the same experimental conditions. The experimental findings reveal that YMCM is the most effective, achieving the highest accuracy of 98.8% with Mel-spectrogram features. The results also provide practical insights into the relationship between feature representation and model capacity. The findings establish YMIR as a useful benchmark and YMCM as a strong baseline for classifying Yemeni music genres.
>
---
#### [new 006] Multimodal Deep Learning Method for Real-Time Spatial Room Impulse Response Computing
- **分类: eess.AS**

- **简介: 该论文属于音频处理任务，旨在实时生成空间房间冲激响应（SRIR），解决传统方法计算复杂的问题。通过融合场景信息与波形数据，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.05545](https://arxiv.org/pdf/2604.05545)**

> **作者:** Zhiyu Li; Xinwen Yue; Shenghui Zhao; Jing Wang
>
> **备注:** This work was accepted by ICASSP 2026
>
> **摘要:** We propose a multimodal deep learning model for VR auralization that generates spatial room impulse responses (SRIRs) in real time to reconstruct scene-specific auditory perception. Employing SRIRs as the output reduces computational complexity and facilitates integration with personalized head-related transfer functions. The model takes two modalities as input: scene information and waveforms, where the waveform corresponds to the low-order reflections (LoR). LoR can be efficiently computed using geometrical acoustics (GA) but remains difficult for deep learning models to predict accurately. Scene geometry, acoustic properties, source coordinates, and listener coordinates are first used to compute LoR in real time via GA, and both LoR and these features are subsequently provided as inputs to the model. A new dataset was constructed, consisting of multiple scenes and their corresponding SRIRs. The dataset exhibits greater diversity. Experimental results demonstrate the superior performance of the proposed model.
>
---
#### [new 007] Active noise cancellation on open-ear smart glasses
- **分类: eess.AS; cs.HC; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于噪声抑制任务，旨在解决开放耳智能眼镜在嘈杂环境中降噪的问题。通过集成麦克风和微型扬声器，实现实时主动降噪。**

- **链接: [https://arxiv.org/pdf/2604.05519](https://arxiv.org/pdf/2604.05519)**

> **作者:** Kuang Yuan; Freddy Yifei Liu; Tong Xiao; Yiwen Song; Chengyi Shen; Saksham Bhutani; Justin Chan; Swarun Kumar
>
> **摘要:** Smart glasses are becoming an increasingly prevalent wearable platform, with audio as a key interaction modality. However, hearing in noisy environments remains challenging because smart glasses are equipped with open-ear speakers that do not seal the ear canal. Furthermore, the open-ear design is incompatible with conventional active noise cancellation (ANC) techniques, which rely on an error microphone inside or at the entrance of the ear canal to measure the residual sound heard after cancellation. Here we present the first real-time ANC system for open-ear smart glasses that suppresses environmental noise using only microphones and miniaturized open-ear speakers embedded in the glasses frame. Our low-latency computational pipeline estimates the noise at the ear from an array of eight microphones distributed around the glasses frame and generates an anti-noise signal in real-time to cancel environmental noise. We develop a custom glasses prototype and evaluate it in a user study across 8 environments under mobility in the 100--1000 Hz frequency range, where environmental noise is concentrated. We achieve a mean noise reduction of 9.6 dB without any calibration, and 11.2 dB with a brief user-specific calibration.
>
---
#### [new 008] Exploring Speech Foundation Models for Speaker Diarization Across Lifespan
- **分类: eess.AS**

- **简介: 该论文属于语音处理任务，研究跨年龄的说话人辨识问题。针对模型在不同年龄段数据上的泛化能力不足，提出多年龄联合训练和领域适应方法，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05201](https://arxiv.org/pdf/2604.05201)**

> **作者:** Anfeng Xu; Tiantian Feng; Shrikanth Narayanan
>
> **备注:** Under review for Interspeech 2026
>
> **摘要:** Speech foundation models have shown strong transferability across a wide range of speech applications. However, their robustness to age-related domain shift in speaker diarization remains underexplored. In this work, we present a cross-lifespan evaluation within a unified end-to-end neural diarization framework (EEND-VC), covering speech samples from conversations involving children, adults, and older adults. We compare models under zero-shot cross-age inference, joint multi-age training, and domain-specific adaptation. Results show substantial performance degradation when models trained on adult-specific speech are applied to child and older-adult conversational data. Moreover, joint multi-age training across different age groups improves robustness without reducing diarization performance in canonical adult conversations, while targeted age group adaptation yields further gains in diarization performance, particularly when using the Whisper encoder.
>
---
#### [new 009] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于符号音乐生成任务，旨在解决长序列生成中的结构不连贯问题。提出ACG框架，通过锚点特征减少误差积累，提升音乐质量与结构完整性。**

- **链接: [https://arxiv.org/pdf/2604.05343](https://arxiv.org/pdf/2604.05343)**

> **作者:** Boyu Cao; Lekai Qian; Dehan Li; Haoyu Gu; Mingda Xu; Qi Liu
>
> **备注:** Accepted at ACL 2026 Findings
>
> **摘要:** Generating long sequences with structural coherence remains a fundamental challenge for autoregressive models across sequential generation tasks. In symbolic music generation, this challenge is particularly pronounced, as existing methods are constrained by the inherent severe error accumulation problem of autoregressive models, leading to poor performance in music quality and structural integrity. In this paper, we propose the Anchored Cyclic Generation (ACG) paradigm, which relies on anchor features from already identified music to guide subsequent generation during the autoregressive process, effectively mitigating error accumulation in autoregressive methods. Based on the ACG paradigm, we further propose the Hierarchical Anchored Cyclic Generation (Hi-ACG) framework, which employs a systematic global-to-local generation strategy and is highly compatible with our specifically designed piano token, an efficient musical representation. The experimental results demonstrate that compared to traditional autoregressive models, the ACG paradigm achieves reduces cosine distance by an average of 34.7% between predicted feature vectors and ground-truth semantic vectors. In long-sequence symbolic music generation tasks, the Hi-ACG framework significantly outperforms existing mainstream methods in both subjective and objective evaluations. Furthermore, the framework exhibits excellent task generalization capabilities, achieving superior performance in related tasks such as music completion.
>
---
#### [new 010] Brain-to-Speech: Prosody Feature Engineering and Transformer-Based Reconstruction
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文属于脑到语音合成任务，旨在从脑电数据中重建自然语音。通过提取韵律特征并使用Transformer模型提升语音质量。**

- **链接: [https://arxiv.org/pdf/2604.05751](https://arxiv.org/pdf/2604.05751)**

> **作者:** Mohammed Salah Al-Radhi; Géza Németh; Andon Tchechmedjiev; Binbin Xu
>
> **备注:** OpenAccess chapter: https://doi.org/10.1007/978-3-032-10561-5_16. In: Curry, E., et al. Artificial Intelligence, Data and Robotics (2026)
>
> **摘要:** This chapter presents a novel approach to brain-to-speech (BTS) synthesis from intracranial electroencephalography (iEEG) data, emphasizing prosody-aware feature engineering and advanced transformer-based models for high-fidelity speech reconstruction. Driven by the increasing interest in decoding speech directly from brain activity, this work integrates neuroscience, artificial intelligence, and signal processing to generate accurate and natural speech. We introduce a novel pipeline for extracting key prosodic features directly from complex brain iEEG signals, including intonation, pitch, and rhythm. To effectively utilize these crucial features for natural-sounding speech, we employ advanced deep learning models. Furthermore, this chapter introduces a novel transformer encoder architecture specifically designed for brain-to-speech tasks. Unlike conventional models, our architecture integrates the extracted prosodic features to significantly enhance speech reconstruction, resulting in generated speech with improved intelligibility and expressiveness. A detailed evaluation demonstrates superior performance over established baseline methods, such as traditional Griffin-Lim and CNN-based reconstruction, across both quantitative and perceptual metrics. By demonstrating these advancements in feature extraction and transformer-based learning, this chapter contributes to the growing field of AI-driven neuroprosthetics, paving the way for assistive technologies that restore communication for individuals with speech impairments. Finally, we discuss promising future research directions, including the integration of diffusion models and real-time inference systems.
>
---
#### [new 011] StrADiff: A Structured Source-Wise Adaptive Diffusion Framework for Linear and Nonlinear Blind Source Separation
- **分类: stat.ML; cs.LG; cs.SD**

- **简介: 该论文提出StrADiff框架，用于解决线性和非线性盲源分离问题。通过为每个源分配自适应扩散机制，实现源级建模与联合优化。**

- **链接: [https://arxiv.org/pdf/2604.04973](https://arxiv.org/pdf/2604.04973)**

> **作者:** Yuan-Hao Wei
>
> **摘要:** This paper presents a Structured Source-Wise Adaptive Diffusion Framework for linear and nonlinear blind source separation. The framework interprets each latent dimension as a source component and assigns to it an individual adaptive diffusion mechanism, thereby establishing source-wise latent modeling rather than relying on a single shared latent prior. The resulting formulation learns source recovery and the mixing/reconstruction process jointly within a unified end-to-end objective, allowing model parameters and latent sources to adapt simultaneously during training. This yields a common framework for both linear and nonlinear blind source separation. In the present instantiation, each source is further equipped with its own adaptive Gaussian process (GP) prior to impose source-wise temporal structure on the latent trajectories, while the overall framework is not restricted to Gaussian process priors and can in principle accommodate other structured source priors. The proposed model thus provides a general structured diffusion-based route to unsupervised source recovery, with potential relevance beyond blind source separation to interpretable latent modeling, source-wise disentanglement, and potentially identifiable nonlinear latent-variable learning under appropriate structural conditions.
>
---
#### [new 012] GLANCE: A Global-Local Coordination Multi-Agent Framework for Music-Grounded Non-Linear Video Editing
- **分类: cs.MA; cs.MM; cs.SD**

- **简介: 该论文提出GLANCE框架，解决音乐引导的非线性视频编辑任务，通过全局-局部协作机制提升视频合成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2604.05076](https://arxiv.org/pdf/2604.05076)**

> **作者:** Zihao Lin; Haibo Wang; Zhiyang Xu; Siyao Dai; Huanjie Dong; Xiaohan Wang; Yolo Y. Tang; Yixin Wang; Qifan Wang; Lifu Huang
>
> **备注:** 14 pages, 4 figures, under review
>
> **摘要:** Music-grounded mashup video creation is a challenging form of video non-linear editing, where a system must compose a coherent timeline from large collections of source videos while aligning with music rhythm, user intent, story completeness, and long-range structural constraints. Existing approaches typically rely on fixed pipelines or simplified retrieval-and-concatenation paradigms, limiting their ability to adapt to diverse prompts and heterogeneous source materials. In this paper, we present GLANCE, a global-local coordination multi-agent framework for music-grounded nonlinear video editing. GLANCE adopts a bi-loop architecture for better editing practice: an outer loop performs long-horizon planning and task-graph construction, and an inner loop adopts the "Observe-Think-Act-Verify" flow for segment-wise editing tasks and their refinements. To address the cross-segment and global conflict emerging after subtimelines composition, we introduce a dedicated global-local coordination mechanism with both preventive and corrective components, which includes a novelly designed context controller, conflict region decomposition module, and a bottom-up dynamic negotiation mechanism. To support rigorous evaluation, we construct MVEBench, a new benchmark that factorizes editing difficulty along task type, prompt specificity, and music length, and propose an agent-as-a-judge evaluation framework for scalable multi-dimensional assessment. Experimental results show that GLANCE consistently outperforms prior research baselines and open-source product baselines under the same backbone models. With GPT-4o-mini as the backbone, GLANCE improves over the strongest baseline by 33.2% and 15.6% on two task settings, respectively. Human evaluation further confirms the quality of the generated videos and validates the effectiveness of the proposed evaluation framework.
>
---
## 更新

#### [replaced 001] ML-ARIS: Multilayer Underwater Acoustic Reconfigurable Intelligent Surface with High-Resolution Reflection Control
- **分类: eess.AS; cs.SD; eess.SP; eess.SY**

- **简介: 该论文属于 underwater communication 任务，旨在解决信号反射控制问题。通过设计多层可重构智能表面 ML-ARIS，实现高精度反射波控制，提升通信效果。**

- **链接: [https://arxiv.org/pdf/2501.18355](https://arxiv.org/pdf/2501.18355)**

> **作者:** Lina Pu; Yu Luo; Aijun Song
>
> **备注:** 16 pages, 19 figures
>
> **摘要:** This article introduces a multilayered acoustic reconfigurable intelligent surface (ML-ARIS) architecture designed for the next generation of underwater communications. ML-ARIS incorporates multiple layers of piezoelectric material in each acoustic reflector, with the load impedance of each layer independently adjustable via a control circuit. This design increases the flexibility in generating reflected signals with desired amplitudes and orthogonal phases, enabling passive synthetic reflection using a single acoustic reflector. Such a feature enables precise beam steering, enhancing sound levels in targeted directions while minimizing interference in surrounding environments. Extensive simulations and tank experiments were conducted to verify the feasibility of ML-ARIS. The experimental results indicate that implementing synthetic reflection with a multilayer structure is indeed practical in real-world scenarios, making it possible to use a single reflection unit to generate reflected waves with high-resolution amplitudes and phases.
>
---
#### [replaced 002] FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统中的说话人切换检测任务，旨在解决实时全双工通信中的低延迟与鲁棒性问题。提出FastTurn框架，结合声学与语义线索，提升决策准确性和响应速度。**

- **链接: [https://arxiv.org/pdf/2604.01897](https://arxiv.org/pdf/2604.01897)**

> **作者:** Chengyou Wang; Hongfei Xue; Chunjiang He; Jingbin Hu; Shuiyuan Wang; Bo Wu; Yuyu Ji; Jimeng Zheng; Ruofei Chen; Zhou Zhu; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
>
---
#### [replaced 003] Rewriting TTS Inference Economics: Lightning V2 on Tenstorrent Achieves 4x Lower Cost Than NVIDIA L40S
- **分类: eess.AS; cs.DC; cs.SD**

- **简介: 该论文属于语音合成任务，解决TTS模型在低精度计算中的质量下降问题。通过硬件软件协同优化，实现高效低成本的TTS推理。**

- **链接: [https://arxiv.org/pdf/2604.03279](https://arxiv.org/pdf/2604.03279)**

> **作者:** Ranjith M. S.; Akshat Mandloi; Sudarshan Kamath
>
> **摘要:** Text-to-Speech (TTS) models are significantly more numerically fragile than Large Language Models (LLMs) due to their continuous waveform generation and perceptual sensitivity to small numerical perturbations. While aggressive precision reduction techniques such as BlockFloat8 (BFP8) and low-fidelity (LoFi) compute have been widely adopted in language models, applying similar strategies to TTS systems often results in audible artifacts, phase instability, and spectral distortion. In this work, we present Lightning V2, a production-grade TTS model co-optimized for Tenstorrent hardware. Through precision-aware architectural design and hardware-software co-optimization, we achieve over 95% LoFi computational fidelity and more than 80% BlockFloat8 deployment without measurable degradation in audio quality. Leveraging Tenstorrent's Network-on-Chip (NoC), distributed SRAM, and deterministic execution model, we reduce memory movement and redundant weight fetches, enabling efficient low-precision inference. Compared to an NVIDIA L40S baseline, Lightning V2 achieves approximately 4x lower on-prem accelerator cost at equivalent throughput, while maintaining production audio fidelity. Our results demonstrate that precision co-design, combined with hardware-aware optimization, can fundamentally reshape the economics of real-time speech inference.
>
---
#### [replaced 004] PhyAVBench: A Challenging Audio Physics-Sensitivity Benchmark for Physically Grounded Text-to-Audio-Video Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到音视频生成任务，旨在解决现有模型缺乏物理合理性的问题。提出PhyAVBench基准，通过音频物理敏感性测试评估模型生成的音视频是否符合物理规律。**

- **链接: [https://arxiv.org/pdf/2512.23994](https://arxiv.org/pdf/2512.23994)**

> **作者:** Tianxin Xie; Wentao Lei; Kai Jiang; Guanjie Huang; Pengfei Zhang; Chunhui Zhang; Fengji Ma; Haoyu He; Han Zhang; Jiangshan He; Jinting Wang; Linghan Fang; Lufei Gao; Orkesh Ablet; Peihua Zhang; Ruolin Hu; Shengyu Li; Weilin Lin; Xiaoyang Feng; Xinyue Yang; Yan Rong; Yanyun Wang; Zihang Shao; Zelin Zhao; Chenxing Li; Shan Yang; Wenfu Wang; Meng Yu; Dong Yu; Li Liu
>
> **备注:** 6 major physical dimensions, 41 fine-grained test points, 337 groups of variable-controlled test samples, 11,605 newly recorded videos
>
> **摘要:** Text-to-audio-video (T2AV) generation is central to applications such as filmmaking and world modeling. However, current models often fail to produce physically plausible sounds. Previous benchmarks primarily focus on audio-video temporal synchronization, while largely overlooking explicit evaluation of audio-physics grounding, thereby limiting the study of physically plausible audio-visual generation. To address this issue, we present PhyAVBench, the first benchmark that systematically evaluates the audio-physics grounding capabilities of T2AV, image-to-audio-video (I2AV), and video-to-audio (V2A) models. PhyAVBench offers PhyAV-Sound-11K, a new dataset of 25.5 hours of 11,605 audible videos collected from 184 participants to ensure diversity and avoid data leakage. It contains 337 paired-prompt groups with controlled physical variations that drive sound differences, each grounded with an average of 17 videos and spanning 6 audio-physics dimensions and 41 fine-grained test points. Each prompt pair is annotated with the physical factors underlying their acoustic differences. Importantly, PhyAVBench leverages paired text prompts to evaluate this capability. We term this evaluation paradigm the Audio-Physics Sensitivity Test (APST) and introduce a novel metric, the Contrastive Physical Response Score (CPRS), which quantifies the acoustic consistency between generated videos and their real-world counterparts. We conduct a comprehensive evaluation of 17 state-of-the-art models. Our results reveal that even leading commercial models struggle with fundamental audio-physical phenomena, exposing a critical gap beyond audio-visual synchronization and pointing to future research directions. We hope PhyAVBench will serve as a foundation for advancing physically grounded audio-visual generation. Prompts, ground-truth, and generated video samples are available at this https URL.
>
---
#### [replaced 005] On The Landscape of Spoken Language Models: A Comprehensive Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型研究，旨在梳理SLM发展脉络，分析其架构、训练与评估方法，解决领域术语不统一及研究分散的问题。**

- **链接: [https://arxiv.org/pdf/2504.08528](https://arxiv.org/pdf/2504.08528)**

> **作者:** Siddhant Arora; Kai-Wei Chang; Chung-Ming Chien; Yifan Peng; Haibin Wu; Yossi Adi; Emmanuel Dupoux; Hung-Yi Lee; Karen Livescu; Shinji Watanabe
>
> **备注:** Published in Transactions on Machine Learning Research
>
> **摘要:** The field of spoken language processing is undergoing a shift from training custom-built, task-specific models toward using and optimizing spoken language models (SLMs) which act as universal speech processing systems. This trend is similar to the progression toward universal language models that has taken place in the field of (text) natural language processing. SLMs include both "pure" language models of speech -- models of the distribution of tokenized speech sequences -- and models that combine speech encoders with text language models, often including both spoken and written input or output. Work in this area is very diverse, with a range of terminology and evaluation settings. This paper aims to contribute an improved understanding of SLMs via a unifying literature survey of recent work in the context of the evolution of the field. Our survey categorizes the work in this area by model architecture, training, and evaluation choices, and describes some key challenges and directions for future work.
>
---
#### [replaced 006] Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset
- **分类: cs.GR; cs.CV; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法语义控制不足和长序列不连贯的问题。提出LRCM框架，结合扩散模型与Mamba模块，实现多模态引导的自回归舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2601.03323](https://arxiv.org/pdf/2601.03323)**

> **作者:** Oran Duan; Yinghua Shen; Yingzhu Lv; Luyang Jie; Yaxin Liu; Qiong Wu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Advances in generative models and sequence learning have greatly promoted research in dance motion generation, yet current methods still suffer from coarse semantic control and poor coherence in long sequences. In this work, we present Listen to Rhythm, Choose Movements (LRCM), a multimodal-guided diffusion framework supporting both diverse input modalities and autoregressive dance motion generation. We explore a feature decoupling paradigm for dance datasets and generalize it to the Motorica Dance dataset, separating motion capture data, audio rhythm, and professionally annotated global and local text descriptions. Our diffusion architecture integrates an audio-latent Conformer and a text-latent Cross-Conformer, and incorporates a Motion Temporal Mamba Module (MTMM) to enable smooth, long-duration autoregressive synthesis. Experimental results indicate that LRCM delivers strong performance in both functional capability and quantitative metrics, demonstrating notable potential in multimodal input scenarios and extended sequence generation. The project page is available at this https URL.
>
---
#### [replaced 007] StressTest: Can YOUR Speech LM Handle the Stress?
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决句子重音对语义影响评估不足的问题。通过构建StressTest基准和Stress-17k数据集，提升模型对重音推理的能力。**

- **链接: [https://arxiv.org/pdf/2505.22765](https://arxiv.org/pdf/2505.22765)**

> **作者:** Iddo Yosha; Gallil Maimon; Yossi Adi
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Sentence stress refers to emphasis on words within a spoken utterance to highlight or contrast an idea. It is often used to imply an underlying intention not explicitly stated. Recent speech-aware language models (SLMs) have enabled direct audio processing, allowing models to access the full richness of speech to perform audio reasoning tasks such as spoken question answering. Despite the crucial role of sentence stress in shaping meaning and intent, it remains largely overlooked in evaluation and development of SLMs. We address this gap by introducing StressTest, a benchmark designed to evaluate models' ability to distinguish between meanings of speech based on the stress pattern. We evaluate leading SLMs, and find that despite their overall capabilities, they perform poorly on such tasks. Hence, we propose a novel data generation pipeline, and create Stress-17k, a training set that simulates change of meaning implied by stress variation. Results suggest, that our finetuned model, StresSLM, generalizes well to real recordings and notably outperforms existing SLMs on sentence stress reasoning and detection. Models, code, data, samples - this http URL.
>
---
#### [replaced 008] Where Do Backdoors Live? A Component-Level Analysis of Backdoor Propagation in Speech Language Models
- **分类: cs.CL; cs.CR; cs.SD**

- **简介: 该论文研究语音语言模型中的后门传播问题，分析组件级影响及多任务嵌入中的后门编码，旨在揭示模型的脆弱性。**

- **链接: [https://arxiv.org/pdf/2510.01157](https://arxiv.org/pdf/2510.01157)**

> **作者:** Alexandrine Fortier; Thomas Thebaud; Jesús Villalba; Najim Dehak; Patrick Cardinal; Peter West
>
> **摘要:** Speech language models (SLMs) are systems of systems: independent components that unite to achieve a common goal. Despite their heterogeneous nature, SLMs are often studied end-to-end; how information flows through the pipeline remains obscure. We investigate this question through the lens of backdoor attacks. We first establish that backdoors can propagate through the SLM, leaving all tasks highly vulnerable. From this, we design a component analysis to reveal the role each component takes in backdoor learning. We find that backdoor persistence or erasure is highly dependent on the targeted component. Beyond propagation, we examine how backdoors are encoded in shared multitask embeddings, showing that poisoned samples are not directly separable from benign ones, challenging a common separability assumption used in filtering defenses. Our findings emphasize the need to treat multimodal pipelines as intricate systems with unique vulnerabilities, not solely extensions of unimodal ones.
>
---
