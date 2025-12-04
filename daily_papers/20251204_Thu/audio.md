# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] AaPE: Aliasing-aware Patch Embedding for Self-Supervised Audio Representation Learning
- **分类: cs.SD; cs.LG; stat.ML**

- **简介: 该论文针对自监督音频表示学习中因频谱图处理导致的混叠问题，提出Aliasing-aware Patch Embedding（AaPE）方法。通过自适应带限正弦核捕捉高频信息，缓解混叠并保留关键声学特征，在多个音频任务上实现最优或竞争力表现。**

- **链接: [https://arxiv.org/pdf/2512.03637v1](https://arxiv.org/pdf/2512.03637v1)**

> **作者:** Kohei Yamamoto; Kosuke Okusa
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Transformer-based audio SSL (self-supervised learning) models often treat spectrograms as images, applying convolutional patchification with heavy temporal downsampling. This lowers the effective Nyquist frequency and introduces aliasing, while naïve low-pass filtering removes task-relevant high-frequency cues. In this study, we present Aliasing-aware Patch Embedding (AaPE), a drop-in patch stem that mitigates aliasing while preserving high-frequency information. AaPE augments standard patch tokens with features produced by a band-limited complex sinusoidal kernel using a two-sided exponential window that dynamically targets alias-prone bands. Frequency and decay parameters of the kernel are estimated from the input, enabling parallel, adaptive subband analysis whose outputs are fused with the standard patch tokens. AaPE integrates seamlessly into the masked teacher-student self-supervised learning. In addition, we combine a multi-mask strategy with a contrastive objective to enforce consistency across diverse mask patterns, stabilizing training. Pre-training on AudioSet followed by fine-tuning evaluation across diverse downstream benchmarks, which spanned categories, such as environmental sounds and other common audio domains. This approach yields state-of-the-art performance on a subset of tasks and competitive results across the remainder. Complementary linear probing evaluation mirrors this pattern, yielding clear gains on several benchmarks and strong performance elsewhere. The collective analysis of these results indicates that AaPE serves to mitigate the effects of aliasing without discarding of informative high-frequency content.
>
---
#### [new 002] State Space Models for Bioacoustics: A comparative Evaluation with Transformers
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究生物声学中的音频分析任务，旨在解决Transformer模型内存消耗大的问题。作者预训练基于Mamba的音频大模型BioMamba，并在BEANS基准上评估其在分类与检测任务中的表现，结果表明其性能接近SOTA模型AVES，但显著降低显存占用。**

- **链接: [https://arxiv.org/pdf/2512.03563v1](https://arxiv.org/pdf/2512.03563v1)**

> **作者:** Chengyu Tang; Sanjeev Baskiyar
>
> **摘要:** In this study, we evaluate the efficacy of the Mamba model in the field of bioacoustics. We first pretrain a Mamba-based audio large language model (LLM) on a large corpus of audio data using self-supervised learning. We fine-tune and evaluate BioMamba on the BEANS benchmark, a collection of diverse bioacoustic tasks including classification and detection, and compare its performance and efficiency with multiple baseline models, including AVES, a state-of-the-art Transformer-based model. The results show that BioMamba achieves comparable performance with AVES while consumption significantly less VRAM, demonstrating its potential in this domain.
>
---
#### [new 003] A Universal Harmonic Discriminator for High-quality GAN-based Vocoder
- **分类: eess.AS**

- **简介: 该论文针对生成对抗网络（GAN）语音合成器中的判别器性能问题，提出一种通用谐波判别器。为解决传统STFT频谱固定分辨率导致的低频谐波表征不足问题，设计可学习的三角形带通滤波器组与半谐波模块，实现动态频率分辨率与谐波跟踪，显著提升语音与歌唱音质。**

- **链接: [https://arxiv.org/pdf/2512.03486v1](https://arxiv.org/pdf/2512.03486v1)**

> **作者:** Nan Xu; Zhaolong Huang; Xiao Zeng
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** With the emergence of GAN-based vocoders, the discriminator, as a crucial component, has been developed recently. In our work, we focus on improving the time-frequency based discriminator. Particularly, Short-Time Fourier Transform (STFT) representation is usually used as input of time-frequency based discriminator. However, the STFT spectrogram has the same frequency resolution at different frequency bins, which results in an inferior performance, especially for singing voices. Motivated by this, we propose a universal harmonic discriminator for dynamic frequency resolution modeling and harmonic tracking. Specifically, we design a harmonic filter with learnable triangular band-pass filter banks, where each frequency bin has a flexible bandwidth. Additionally, we add a half-harmonic to capture fine-grained harmonic relationships at low-frequency band. Experiments on speech and singing datasets validate the effectiveness of the proposed discriminator on both subjective and objective metrics.
>
---
#### [new 004] Comparing Unsupervised and Supervised Semantic Speech Tokens: A Case Study of Child ASR
- **分类: eess.AS**

- **简介: 该论文研究儿童语音识别（child ASR）任务，对比监督与非监督语义语音标记的性能。针对低资源场景下标记效率与识别准确率问题，利用预训练语音基础模型，比较了基于K-means的无监督方法与基于ASR损失的监督量化方法。结果表明，监督方法不仅优于无监督方法，且在极低比特率下仍表现优异，甚至超越连续表示。**

- **链接: [https://arxiv.org/pdf/2512.03301v1](https://arxiv.org/pdf/2512.03301v1)**

> **作者:** Mohan Shi; Natarajan Balaji Shankar; Kaiyuan Zhang; Zilai Wang; Abeer Alwan
>
> **备注:** ASRU-AI4CSL
>
> **摘要:** Discrete speech tokens have gained attention for their storage efficiency and integration with Large Language Models (LLMs). They are commonly categorized into acoustic and semantic tokens, with the latter being more advantageous for Automatic Speech Recognition (ASR). Traditionally, unsupervised K-means clustering has been used to extract semantic speech tokens from Speech Foundation Models (SFMs). Recently, supervised methods, such as finite scalar quantization (FSQ) trained with ASR loss, have emerged for speech generation. Both approaches leverage pre-trained SFMs, benefiting low-resource tasks such as child ASR. This paper systematically compares supervised and unsupervised semantic speech tokens for child ASR. Results show that supervised methods not only outperform unsupervised ones but even unexpectedly surpass continuous representations, and they perform well even in ultra-low bitrate settings. These findings highlight the advantages of supervised semantic tokens and offer insights for improving discrete speech tokenization.
>
---
#### [new 005] Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning
- **分类: cs.AI; cs.SD**

- **简介: 该论文针对多模态模型推理僵化问题，提出Omni-AutoThink框架，通过自适应监督微调与强化学习，动态调整推理深度。构建跨模态基准评估，显著提升模型在复杂任务中的自适应推理能力。**

- **链接: [https://arxiv.org/pdf/2512.03783v1](https://arxiv.org/pdf/2512.03783v1)**

> **作者:** Dongchao Yang; Songxiang Liu; Disong Wang; Yuanyuan Wang; Guanglu Wan; Helen Meng
>
> **摘要:** Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors, either overthinking simple problems or failing to reason when necessary. To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model's reasoning depth according to task difficulty. Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback. We further construct a comprehensive adaptive reasoning benchmark that spans text-only, text-audio, text-visual, and text-audio-visual modalities, providing both training and evaluation splits for multimodal reasoning assessment. Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines. All benchmark data and code will be publicly released.
>
---
#### [new 006] Head, posture, and full-body gestures in interactive communication
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究听觉障碍环境下全身动作对交互沟通的影响，旨在揭示身体各部位（手、头、躯干、腿）在噪声中的运动规律。通过虚拟声学环境下的双人对话实验，采用新标注系统分析动作频率与质量，发现噪声增加时手势复杂度上升，头部动作模式改变，但语音-手势同步性未显著变化，揭示了多模态适应机制。**

- **链接: [https://arxiv.org/pdf/2512.03636v1](https://arxiv.org/pdf/2512.03636v1)**

> **作者:** Ľuboš Hládek; Bernhard U. Seeber
>
> **备注:** 7 figures, 10 tables, 30 pages
>
> **摘要:** When face-to-face communication becomes effortful due to background noise or interfering talkers, the role of visual cues becomes increasingly important for communication success. While previous research has selectively examined head or hand movements, here we explore movements of the whole body in acoustically adverse conditions. We hypothesized that increasing background noise in conversations would lead to increased gesture frequency in hand, head, trunk, and leg movements typical of conversation. Increased use of hand movements should support the speaker's role, while increased head and trunk movements may help the listener. We conducted a free dyadic conversation experiment with normal-hearing participants (n=8) in a virtual acoustic environment. Conversational movements were described with a newly developed labeling system for typical conversational actions, and the frequency of individual types was analyzed. In addition, we analyzed gesture quality by assessing hand-speech synchrony, with the hypothesis that higher levels of background noise would lead to a loss of synchrony according to an interactive coupling model. Higher noise levels led to increased hand-gesture complexity during speaking and listening, more pronounced up-down head movements, and contrary to expectations, head movements during listening generally decreased relative to speaking. Synchrony and peak velocity were unaffected by noise, while gesture quality scaled only modestly. The results support previous findings regarding gesturing frequency, but we found only limited evidence for changes in speech-gesture synchrony. This work reveals communication patterns of the whole body and illustrates multimodal adaptation to communication demands.
>
---
#### [new 007] A Convolutional Framework for Mapping Imagined Auditory MEG into Listened Brain Responses
- **分类: eess.SP; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于神经解码任务，旨在将想象中的听觉信息映射为类似真实聆听的脑响应。针对想象语音解码中时间不确定性与数据稀缺问题，研究构建了基于MEG的音乐与诗歌想象数据集，提出带个体校准层的卷积神经网络模型，实现跨被试稳定映射，显著提升预测准确性，为脑机接口中想象言语与音乐应用奠定基础。**

- **链接: [https://arxiv.org/pdf/2512.03458v1](https://arxiv.org/pdf/2512.03458v1)**

> **作者:** Maryam Maghsoudi; Mohsen Rezaeizadeh; Shihab Shamma
>
> **摘要:** Decoding imagined speech engages complex neural processes that are difficult to interpret due to uncertainty in timing and the limited availability of imagined-response datasets. In this study, we present a Magnetoencephalography (MEG) dataset collected from trained musicians as they imagined and listened to musical and poetic stimuli. We show that both imagined and perceived brain responses contain consistent, condition-specific information. Using a sliding-window ridge regression model, we first mapped imagined responses to listened responses at the single-subject level, but found limited generalization across subjects. At the group level, we developed an encoder-decoder convolutional neural network with a subject-specific calibration layer that produced stable and generalizable mappings. The CNN consistently outperformed the null model, yielding significantly higher correlations between predicted and true listened responses for nearly all held-out subjects. Our findings demonstrate that imagined neural activity can be transformed into perception-like responses, providing a foundation for future brain-computer interface applications involving imagined speech and music.
>
---
## 更新

#### [replaced 001] CoHear: Conversation Enhancement via Multi-Earphone Collaboration
- **分类: cs.SD**

- **简介: 该论文针对嘈杂环境中“鸡尾酒会聋”问题，提出ClearSphere系统，通过多耳机协作实现对话级语音增强。解决多说话人重叠、背景噪声干扰下的清晰语音提取难题。工作包括设计对话驱动的网络协议与高效语音提取模型，实现在无基础设施下实时协同，显著提升语音质量与用户体验。**

- **链接: [https://arxiv.org/pdf/2505.21004v3](https://arxiv.org/pdf/2505.21004v3)**

> **作者:** Lixing He; Yunqi Guo; Zhenyu Yan; Guoliang Xing
>
> **备注:** Submitted to IMWUT
>
> **摘要:** In crowded places such as conferences, background noise, overlapping voices, and lively interactions make it difficult to have clear conversations. This situation often worsens the phenomenon known as "cocktail party deafness." We present ClearSphere, the collaborative system that enhances speech at the conversation level with multi-earphones. Real-time conversation enhancement requires a holistic modeling of all the members in the conversation, and an effective way to extract the speech from the mixture. ClearSphere bridges the acoustic sensor system and state-of-the-art deep learning for target speech extraction by making two key contributions: 1) a conversation-driven network protocol, and 2) a robust target conversation extraction model. Our networking protocol enables mobile, infrastructure-free coordination among earphone devices. Our conversation extraction model can leverage the relay audio in a bandwidth-efficient way. ClearSphere is evaluated in both real-world experiments and simulations. Results show that our conversation network obtains more than 90\% accuracy in group formation, improves the speech quality by up to 8.8 dB over state-of-the-art baselines, and demonstrates real-time performance on a mobile device. In a user study with 20 participants, ClearSphere has a much higher score than baseline with good usability.
>
---
#### [replaced 002] First Deep Learning Approach to Hammering Acoustics for Stem Stability Assessment in Total Hip Arthroplasty
- **分类: eess.AS**

- **简介: 该论文提出首个基于深度学习的髋关节置换术中锤击声分析方法，用于评估股骨柄初始稳定性。针对手术中声学信号受个体差异影响大、传统方法受限的问题，采用TimeMIL模型结合对数梅尔谱图与伪标签技术，实现高精度稳定性分类，验证了音频事件分类在术中评估中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.18725v2](https://arxiv.org/pdf/2511.18725v2)**

> **作者:** Dongqi Zhu; Zhuwen Xu; Youyuan Chen; Minghao Jin; Wan Zheng; Yi Zhou; Huiwu Li; Yongyun Chang; Feng Hong; Zanjing Zhai
>
> **备注:** The manuscript, including both the title and the main text, contains issues with clarity and precision in its overall presentation, necessitating a complete withdrawal for revision
>
> **摘要:** Audio event classification has recently emerged as a promising approach in medical applications. In total hip arthroplasty (THA), intra-operative hammering acoustics provide critical cues for assessing the initial stability of the femoral stem, yet variability due to femoral morphology, implant size, and surgical technique constrains conventional assessment methods. We propose the first deep learning framework for this task, employing a TimeMIL model trained on Log-Mel Spectrogram features and enhanced with pseudo-labeling. On intra-operative recordings, the method achieved 91.17 % +/- 2.79 % accuracy, demonstrating reliable estimation of stem stability. Comparative experiments further show that reducing the diversity of femoral stem brands improves model performance, although limited dataset size remains a bottleneck. These results establish deep learning-based audio event classification as a feasible approach for intra-operative stability assessment in THA.
>
---
#### [replaced 003] Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对多标签音频分类中自监督学习模型评估依赖昂贵微调的问题，指出全局池化导致的特征信息瓶颈使线性探针无法准确反映嵌入质量。提出二值原型探针，通过学习类别原型实现类内信息聚合，显著优于传统方法，验证了探针法在音频SSL评估中的有效性与高效性。**

- **链接: [https://arxiv.org/pdf/2509.24901v3](https://arxiv.org/pdf/2509.24901v3)**

> **作者:** Lukas Rauch; René Heinrich; Houtan Ghaffari; Lukas Miklautz; Ilyass Moummad; Bernhard Sick; Christoph Scholz
>
> **备注:** Currently under review
>
> **摘要:** Although probing frozen models has become a standard evaluation paradigm, self-supervised learning in audio defaults to fine-tuning when pursuing state-of-the-art on AudioSet. A key reason is that global pooling creates an information bottleneck causing linear probes to misrepresent the embedding quality: The $\texttt{cls}$-token discards crucial token information about dispersed, localized events in audio. This weakness is rooted in the mismatch between the pretraining objective (globally) and the downstream task (localized). Across a comprehensive benchmark of 13 datasets and 6 spectrogram-based encoders, we investigate the global pooling bottleneck. We introduce binarized prototypical probes: a lightweight and simple pooling method that learns prototypes to perform class-wise information aggregation. Despite its simplicity, our method notably outperforms linear and attentive probing. Our work establishes probing as a competitive and efficient paradigm for evaluating audio SSL models, challenging the reliance on costly fine-tuning.
>
---
#### [replaced 004] Probabilistic Fusion and Calibration of Neural Speaker Diarization Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对端到端神经说话人聚类（EEND）系统中概率输出未被充分利用的问题，提出一种基于概率的融合与校准框架。通过连续概率输出实现更优的模型融合与校准，显著提升性能（最高19%相对DER降低），并证明联合校准和“先融合后校准”策略更优，为下游应用提供可靠置信度。**

- **链接: [https://arxiv.org/pdf/2511.22696v3](https://arxiv.org/pdf/2511.22696v3)**

> **作者:** Juan Ignacio Alvarez-Trejos; Sergio A. Balanya; Daniel Ramos; Alicia Lozano-Diez
>
> **摘要:** End-to-End Neural Diarization (EEND) systems produce frame-level probabilistic speaker activity estimates, yet since evaluation focuses primarily on Diarization Error Rate (DER), the reliability and calibration of these confidence scores have been largely neglected. When fusing multiple diarization systems, DOVER-Lap remains the only established approach, operating at the segment level with hard decisions. We propose working with continuous probability outputs, which enables more sophisticated fusion and calibration techniques that can leverage model uncertainty and complementary strengths across different architectures. This paper presents the first comprehensive framework for calibrating and fusing EEND models at the probability level. We investigate two output formulations (multilabel and powerset representations) and their impact on calibration and fusion effectiveness. Through extensive experiments on the CallHome two-speaker benchmark, we demonstrate that proper calibration provides substantial improvements even for individual models (up to 19% relative DER reduction), in some cases mitigating the absence of domain adaptation. We reveal that joint calibration in powerset space consistently outperforms independent per-speaker calibration, that fusion substantially improves over individual models, and that the Fuse-then-Calibrate ordering generally outperforms both calibrating before fusion and uncalibrated fusion while requiring calibration of only a single combined model. Our best configuration outperforms DOVER-Lap in terms of DER while providing reliable confidence estimates essential for downstream applications. This work proposes best practices for probability-level fusion of EEND systems and demonstrates the advantages of leveraging soft outputs over hard decisions.
>
---
#### [replaced 005] ERF-BA-TFD+: A Multimodal Model for Audio-Visual Deepfake Detection
- **分类: cs.AI; cs.SD**

- **简介: 该论文针对音视频多模态深度伪造检测任务，提出ERF-BA-TFD+模型。通过增强感受野和跨模态融合，有效捕捉长程依赖关系，提升对细微伪造痕迹的识别能力。在DDL-AV数据集上实现领先性能，获竞赛第一名。**

- **链接: [https://arxiv.org/pdf/2508.17282v2](https://arxiv.org/pdf/2508.17282v2)**

> **作者:** Xin Zhang; Jiaming Chu; Jian Zhao; Yuchu Jiang; Xu Yang; Lei Jin; Chi Zhang; Xuelong Li
>
> **备注:** The paper is withdrawn after discovering a flaw in the theoretical derivation presented in Section Method. The incorrect step leads to conclusions that are not supported by the corrected derivation. We plan to reconstruct the argument and will release an updated version once the issue is fully resolved
>
> **摘要:** Deepfake detection is a critical task in identifying manipulated multimedia content. In real-world scenarios, deepfake content can manifest across multiple modalities, including audio and video. To address this challenge, we present ERF-BA-TFD+, a novel multimodal deepfake detection model that combines enhanced receptive field (ERF) and audio-visual fusion. Our model processes both audio and video features simultaneously, leveraging their complementary information to improve detection accuracy and robustness. The key innovation of ERF-BA-TFD+ lies in its ability to model long-range dependencies within the audio-visual input, allowing it to better capture subtle discrepancies between real and fake content. In our experiments, we evaluate ERF-BA-TFD+ on the DDL-AV dataset, which consists of both segmented and full-length video clips. Unlike previous benchmarks, which focused primarily on isolated segments, the DDL-AV dataset allows us to assess the model's performance in a more comprehensive and realistic setting. Our method achieves state-of-the-art results on this dataset, outperforming existing techniques in terms of both accuracy and processing speed. The ERF-BA-TFD+ model demonstrated its effectiveness in the "Workshop on Deepfake Detection, Localization, and Interpretability," Track 2: Audio-Visual Detection and Localization (DDL-AV), and won first place in this competition.
>
---
#### [replaced 006] IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
- **分类: eess.AS**

- **简介: 该论文针对语音匿名化中伪说话人唯一性差、计算成本高的问题，提出IDMap框架，通过建立说话人身份索引到向量的前馈映射，实现高效且独特的伪说话人生成。实验表明其在小规模和大规模场景下均提升了语音隐私保护效果与计算效率。**

- **链接: [https://arxiv.org/pdf/2511.06246v3](https://arxiv.org/pdf/2511.06246v3)**

> **作者:** Zeyan Liu; Liping Chen; Kong Aik Lee; Zhenhua Ling
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Facilitated by the speech generation framework that disentangles speech into content, speaker, and prosody, voice anonymization is accomplished by substituting the original speaker embedding vector with that of a pseudo-speaker. In this framework, the pseudo-speaker generation forms a fundamental challenge. Current pseudo-speaker generation methods demonstrate limitations in the uniqueness of pseudo-speakers, consequently restricting their effectiveness in voice privacy protection. Besides, existing model-based methods suffer from heavy computation costs. Especially, in the large-scale scenario where a huge number of pseudo-speakers are generated, the limitations of uniqueness and computational inefficiency become more significant. To this end, this paper proposes a framework for pseudo-speaker generation, which establishes a mapping from speaker identity index to speaker vector in the feedforward architecture, termed IDMap. Specifically, the framework is specified into two models: IDMap-MLP and IDMap-Diff. Experiments were conducted on both small- and large-scale evaluation datasets. Small-scale evaluations on the LibriSpeech dataset validated the effectiveness of the proposed IDMap framework in enhancing the uniqueness of pseudo-speakers, thereby improving voice privacy protection, while at a reduced computational cost. Large-scale evaluations on the MLS and Common Voice datasets further justified the superiority of the IDMap framework regarding the stability of the voice privacy protection capability as the number of pseudo-speakers increased. Audio samples and open-source code can be found in https://github.com/VoicePrivacy/IDMap.
>
---
#### [replaced 007] Direction-of-Arrival and Noise Covariance Matrix joint estimation for beamforming
- **分类: eess.AS; math.OC**

- **简介: 该论文针对波束成形中的方向到达（DoA）与噪声协方差矩阵（NCM）联合估计问题，提出一种简化计算的准线性方法，结合全频段DoA估计，提升混响环境下的鲁棒性。实验表明，该方法在中高角度场景下优于传统MUSIC算法，具有更优的信号增强、噪声抑制与干扰消除能力。**

- **链接: [https://arxiv.org/pdf/2511.10639v2](https://arxiv.org/pdf/2511.10639v2)**

> **作者:** Vitor Gelsleichter Probst Curtarelli; Stephan Paul; Anderson Wedderhoff Spengler
>
> **摘要:** We propose a joint estimation method for the Direction-of-Arrival (DoA) and the Noise Covariance Matrix (NCM) tailored for beamforming applications. Building upon an existing NCM framework, our approach simplifies the estimation procedure by deriving an quasi-linear solution, instead of the traditional exhaustive search. Additionally, we introduce a novel DoA estimation technique that operates across all frequency bins, improving robustness in reverberant environments. Simulation results demonstrate that our method outperforms classical techniques, such as MUSIC, in mid- to high-angle scenarios, achieving lower angular errors and superior signal enhancement through beamforming. The proposed framework was also fared against other techniques for signal enhancement, having better noise rejection and interference canceling capabilities. These improvements are validated using both theoretical and empirical performance metrics.
>
---
#### [replaced 008] Musical consonance: a review of theory and evidence on perception and preference of auditory roughness in humans and other animals
- **分类: physics.soc-ph; cs.SD; eess.AS**

- **简介: 该论文综述音乐协和性的理论与证据，聚焦听觉粗糙度的感知与偏好。针对人类及其他动物的协和性起源问题，分析粗糙度、谐波性与文化学习三假说，指出当前研究在定义、测量及模型构建上的缺陷，强调未来需发展更简洁、广覆盖的理论模型。**

- **链接: [https://arxiv.org/pdf/2510.14159v2](https://arxiv.org/pdf/2510.14159v2)**

> **作者:** John M. McBride
>
> **摘要:** The origins of consonance in human music has long been contested, and today there are three primary hypotheses: aversion to roughness, preference for harmonicity, and learned preferences from cultural exposure. While the evidence is currently insufficient to disentangle the contributions of these hypotheses, I propose several reasons why roughness is an especially promising area for future study. The aim of this review is to summarize and critically evaluate roughness theory and models, experimental data, to highlight areas that deserve further research. I identify 2 key areas: There are fundamental issues with the definition and interpretation of results due to tautology in the definition of roughness, and the lack of independence in empirical measurements. Despite extensive model development, there are many duplications and models have issues with data quality and overfitting. Future theory development should aim for model simplicity, and extra assumptions, features and parameters should be evaluated systematically. Model evaluation should aim to maximise the breadth of stimuli that are predicted.
>
---
#### [replaced 009] STCTS: Generative Semantic Compression for Ultra-Low Bitrate Speech via Explicit Text-Prosody-Timbre Decomposition
- **分类: cs.SD; cs.MM**

- **简介: 该论文针对超低比特率语音传输难题，提出STCTS框架，通过显式分解语音为语义、韵律和音色三部分，分别采用上下文编码、稀疏韵律传输与声纹嵌入压缩，实现80 bps下的自然语音通信，显著降低比特率并保持高质量，支持隐私保护与边缘部署。**

- **链接: [https://arxiv.org/pdf/2512.00451v2](https://arxiv.org/pdf/2512.00451v2)**

> **作者:** Siyu Wang; Haitao Li; Donglai Zhu
>
> **备注:** The complete source code and online speech reconstruction demo is publicly available at https://github.com/dywsy21/STCTS
>
> **摘要:** Voice communication in bandwidth-constrained environments--maritime, satellite, and tactical networks--remains prohibitively expensive. Traditional codecs struggle below 1 kbps, while existing semantic approaches (STT-TTS) sacrifice prosody and speaker identity. We present STCTS, a generative semantic compression framework enabling natural voice communication at 80 bps. STCTS explicitly decomposes speech into linguistic content, prosodic expression, and speaker timbre, applying tailored compression: context-aware text encoding (70 bps), sparse prosody transmission via TTS interpolation (<14 bps at 0.1-1 Hz), and amortized speaker embedding. Evaluations on LibriSpeech demonstrate a 75x bitrate reduction versus Opus (6 kbps) and 12x versus EnCodec (1 kbps), while maintaining perceptual quality (NISQA MOS > 4.26), graceful degradation under packet loss and noise resilience. We also discover a bimodal quality distribution with prosody sampling rate: sparse and dense updates both achieve high quality, while mid-range rates degrade due to perceptual discontinuities--guiding optimal configuration design. Beyond efficiency, our modular architecture supports privacy-preserving encryption, human-interpretable transmission, and flexible deployment on edge devices, offering a robust solution for ultra-low bandwidth scenarios.
>
---
