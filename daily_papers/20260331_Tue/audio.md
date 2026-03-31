# 音频 cs.SD;  eess.AS

- **最新发布 31 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Diachronic Modeling of Tonal Coherence on the Tonnetz Across Classical and Popular Repertoires
- **分类: cs.SD**

- **简介: 该论文属于音乐分析任务，旨在探讨不同音乐传统如何实现调性连贯性。通过构建两个新指标，分析了古典与流行音乐的调性特征差异。**

- **链接: [https://arxiv.org/pdf/2603.27035](https://arxiv.org/pdf/2603.27035)**

> **作者:** Weilun Xu; Edward Hall; Martin Rohrmeier
>
> **摘要:** How do different musical traditions achieve tonal coherence? Most computational measures to date have analysed tonal coherence in terms of a single dimension, whereas a multi-dimensional analyses have not been sufficiently explored. We propose a new model drawing on the concept of the Tonnetz -- we define two partially independent measures: \emph{tonal focus}, the concentration of pitch content near a tonal center; and \emph{tonal connection}, the degree to which pitch content reflects structured intervallic pathways back to that center. Analyzing over 2,800 pieces from Western classical and popular traditions, we find that these traditions occupy overlapping yet distinguishable regions of the two-dimensional space. Popular music shows higher tonal focus, while classical music exhibits higher tonal connection. Our complementary measures ground the differences between different tonal styles in quantitative evidence, and offer interpretable dimensions for computational music analysis and controllable generation.
>
---
#### [new 002] SHroom: A Python Framework for Ambisonics Room Acoustics Simulation and Binaural Rendering
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SHroom，一个用于混音和双耳渲染的Python框架，解决房间声学模拟问题，通过球面谐波实现高效实时处理。**

- **链接: [https://arxiv.org/pdf/2603.27342](https://arxiv.org/pdf/2603.27342)**

> **作者:** Yhonatan Gayer
>
> **摘要:** Spherical Harmonics ROOM), an open-source Python library for room acoustics simulation using Ambisonics, available at this https URL and installable via \texttt{pip install pyshroom}. \textbf{shroom} projects image-source contributions onto a Spherical Harmonics (SH) basis, yielding a composable pipeline for binaural decoding, spherical array simulation, and real-time head rotation. Benchmarked against \texttt{pyroomacoustics} with an $N=30$ reference, \textbf{shroom} with Magnitude Least Squares (MagLS) achieves perceptual transparency (2.02~dB Log Spectral Distance (LSD) at $N=5$, within the 1--2~dB Just Noticeable Difference (JND)) while its fixed-once decode amortises over multiple sources ($K=1$-to-$8$: slowdown narrows from $7\times$ to $3.1\times$). For dynamic head rotation, \textbf{shroom} applies a Wigner-D multiply at $<1$~ms/frame, making it the only architecturally viable real-time choice.
>
---
#### [new 003] A Probabilistic Generative Model for Spectral Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决听力辅助设备在非平稳环境中的适应性问题。提出一种概率生成模型，实现自适应和个性化处理。**

- **链接: [https://arxiv.org/pdf/2603.28436](https://arxiv.org/pdf/2603.28436)**

> **作者:** Marco Hidalgo-Araya; Raphaël Trésor; Bart Van Erp; Wouter W.L. Nuijten; Thijs Van De Laar; Bert De Vries
>
> **备注:** Submitted to the IEEE Open Journal of Signal Processing
>
> **摘要:** Speech enhancement in hearing aids remains a difficult task in nonstationary acoustic environments, mainly because current signal processing algorithms rely on fixed, manually tuned parameters that cannot adapt in situ to different users or listening contexts. This paper introduces a unified modular framework that formulates signal processing, learning, and personalization as Bayesian inference with explicit uncertainty tracking. The proposed framework replaces ad hoc algorithm design with a single probabilistic generative model that continuously adapts to changing acoustic conditions and user preferences. It extends spectral subtraction with principled mechanisms for in-situ personalization and adaptation to acoustic context. The system is implemented as an interconnected probabilistic state-space model, and inference is performed via variational message passing in the \texttt{this http URL} probabilistic programming environment, enabling real-time Bayesian processing under hearing-aid constraints. Proof-of-concept experiments on the \emph{VoiceBank+DEMAND} corpus show competitive speech quality and noise reduction with 85 effective parameters. The framework provides an interpretable, data-efficient foundation for uncertainty-aware, adaptive hearing-aid processing and points toward devices that learn continuously through probabilistic inference.
>
---
#### [new 004] On the Usefulness of Diffusion-Based Room Impulse Response Interpolation to Microphone Array Processing
- **分类: cs.SD**

- **简介: 论文属于空间音频处理任务，解决房间脉冲响应估计问题。通过扩散插值方法提升麦克风阵列处理性能，并验证其在真实场景中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28209](https://arxiv.org/pdf/2603.28209)**

> **作者:** Sagi Della Torre; Mirco Pezzoli; Fabio Antonacci; Sharon Gannot
>
> **摘要:** Room Impulse Responses estimation is a fundamental problem in spatial audio processing and speech enhancement. In this paper, we build upon our previously introduced diffusion-based inpainting framework for Room Impulse Response interpolation and demonstrate its applicability to enhancing the performance of practical multi-microphone array processing tasks. Furthermore, we validate the robustness of this method in interpolating real-world Room Impulse Responses.
>
---
#### [new 005] Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于音乐转录任务，旨在提升多乐器音频的自动转录。通过2025年AMT挑战赛，评估并改进了模型在多声部和音色变化上的表现。**

- **链接: [https://arxiv.org/pdf/2603.27528](https://arxiv.org/pdf/2603.27528)**

> **作者:** Ojas Chaturvedi; Kayshav Bhardwaj; Tanay Gondil; Benjamin Shiue-Hal Chou; Kristen Yeon-Ji Yun; Yung-Hsiang Lu; Yujia Yan; Sungkyun Chang
>
> **备注:** 7 pages, 3 figures. Accepted to the AI for Music Workshop at NeurIPS 2025
>
> **摘要:** This paper presents the results of the 2025 Automatic Music Transcription (AMT) Challenge, an online competition to benchmark progress in multi-instrument transcription. Eight teams submitted valid solutions; two outperformed the baseline MT3 model. The results highlight both advances in transcription accuracy and the remaining difficulties in handling polyphony and timbre variation. We conclude with directions for future challenges: broader genre coverage and stronger emphasis on instrument detection.
>
---
#### [new 006] Audio Language Model for Deepfake Detection Grounded in Acoustic Chain-of-Thought
- **分类: cs.SD**

- **简介: 该论文属于深度伪造语音检测任务，旨在解决现有系统无法提供可解释推理的问题。通过引入结构化声学特征和链式思维推理，提升检测准确性和解释性。**

- **链接: [https://arxiv.org/pdf/2603.28021](https://arxiv.org/pdf/2603.28021)**

> **作者:** Runkun Chen; Yixiong Fang; Pengyu Chang; Yuante Li; Massa Baali; Bhiksha Ramakrishnan
>
> **摘要:** Deepfake speech detection systems are often limited to binary classification tasks and struggle to generate interpretable reasoning or provide context-rich explanations for their decisions. These models primarily extract latent embeddings for authenticity detection but fail to leverage structured acoustic evidence such as prosodic, spectral, and physiological attributes in a meaningful manner. This paper introduces CoLMbo-DF, a Feature-Guided Audio Language Model that addresses these limitations by integrating robust deepfake detection with explicit acoustic chain-of-thought reasoning. By injecting structured textual representations of low-level acoustic features directly into the model prompt, our approach grounds the model's reasoning in interpretable evidence and improves detection accuracy. To support this framework, we introduce a novel dataset of audio pairs paired with chain-of-thought annotations. Experiments show that our method, trained on a lightweight open-source language model, significantly outperforms existing audio language model baselines despite its smaller scale, marking a significant advancement in explainable deepfake speech detection.
>
---
#### [new 007] MOSS-VoiceGenerator: Create Realistic Voices with Natural Language Descriptions
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音生成任务，旨在通过自然语言描述生成真实语音。解决现有模型语音过于人工、缺乏真实感的问题，通过大规模影视数据训练提升语音自然度。**

- **链接: [https://arxiv.org/pdf/2603.28086](https://arxiv.org/pdf/2603.28086)**

> **作者:** Kexin Huang; Liwei Fan; Botian Jiang; Yaozhou Jiang; Qian Tu; Jie Zhu; Yuqian Zhang; Yiwei Zhao; Chenchen Yang; Zhaoye Fei; Shimin Li; Xiaogui Yang; Qinyuan Cheng; Xipeng Qiu
>
> **摘要:** Voice design from natural language aims to generate speaker timbres directly from free-form textual descriptions, allowing users to create voices tailored to specific roles, personalities, and emotions. Such controllable voice creation benefits a wide range of downstream applications-including storytelling, game dubbing, role-play agents, and conversational assistants, making it a significant task for modern Text-to-Speech models. However, existing models are largely trained on carefully recorded studio data, which produces speech that is clean and well-articulated, yet lacks the lived-in qualities of real human voices. To address these limitations, we present MOSS-VoiceGenerator, an open-source instruction-driven voice generation model that creates new timbres directly from natural language prompts. Motivated by the hypothesis that exposure to real-world acoustic variation produces more perceptually natural voices, we train on large-scale expressive speech data sourced from cinematic content. Subjective preference studies demonstrate its superiority in overall performance, instruction-following, and naturalness compared to other voice design models.
>
---
#### [new 008] BiFormer3D: Grid-Free Time-Domain Reconstruction of Head-Related Impulse Responses with a Spatially Encoded Transformer
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于声场重建任务，解决从稀疏测量中恢复个体化HRIR的问题。提出BiFormer3D模型，在时域实现无网格的HRIR重构。**

- **链接: [https://arxiv.org/pdf/2603.27998](https://arxiv.org/pdf/2603.27998)**

> **作者:** Shaoheng Xu; Chunyi Sun; Jihui Zhang; Amy Bastine; Prasanga N. Samarasinghe; Thushara D. Abhayapala; Hongdong Li
>
> **备注:** The paper was submitted for review to Interspeech 2026
>
> **摘要:** Individualized head-related impulse responses (HRIRs) enable binaural rendering, but dense per-listener measurements are costly. We address HRIR spatial up-sampling from sparse per-listener measurements: given a few measured HRIRs for a listener, predict HRIRs at unmeasured target directions. Prior learning methods often work in the frequency domain, rely on minimum-phase assumptions or separate timing models, and use a fixed direction grid, which can degrade temporal fidelity and spatial continuity. We propose BiFormer3D, a time-domain, grid-free binaural Transformer for reconstructing HRIRs at arbitrary directions from sparse inputs. It uses sinusoidal spatial features, a Conv1D refinement module, and auxiliary interaural time difference (ITD) and interaural level difference (ILD) heads. On SONICOM, it improves normalized mean squared error (NMSE), cosine distance, and ITD/ILD errors over prior methods; ablations validate modules and show minimum-phase pre-processing is unnecessary.
>
---
#### [new 009] EvA: An Evidence-First Audio Understanding Paradigm for LALMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频理解任务，解决LALMs在复杂声学场景中的证据瓶颈问题。提出EvA架构，通过双路径融合提升音频证据保留，增强推理效果。**

- **链接: [https://arxiv.org/pdf/2603.27667](https://arxiv.org/pdf/2603.27667)**

> **作者:** Xinyuan Xie; Shunian Chen; Zhiheng Liu; Yuhao Zhang; Zhiqiang Lv; Liyin Liang; Benyou Wang
>
> **摘要:** Large Audio Language Models (LALMs) still struggle in complex acoustic scenes because they often fail to preserve task-relevant acoustic evidence before reasoning begins. We call this failure the evidence bottleneck: state-of-the-art systems show larger deficits in evidence extraction than in downstream reasoning, suggesting that the main limitation lies in upstream perception rather than reasoning policy. To address this problem, we propose EvA (Evidence-First Audio), a dual-path architecture that combines Whisper and CED-Base through non-compressive, time-aligned fusion. EvA first aggregates intermediate CED layers to preserve multi-scale acoustic cues, then aligns the aggregated CED features to the Whisper timeline and adds the two streams without changing sequence length. We also build EvA-Perception, a large-scale open-source training set with about 54K event-ordered captions (150 h) and about 500K QA pairs. Under a unified zero-shot protocol, EvA achieves the best open-source Perception scores on MMAU, MMAR, and MMSU, and improves over Kimi-Audio-7B on all reported metrics, with the largest gains on perception-heavy splits. These results support the evidence-first hypothesis: stronger audio understanding depends on preserving acoustic evidence before reasoning.
>
---
#### [new 010] Acoustic-to-articulatory Inversion of the Complete Vocal Tract from RT-MRI with Various Audio Embeddings and Dataset Sizes
- **分类: eess.AS**

- **简介: 该论文属于语音到发音体逆问题，旨在通过RT-MRI数据实现完整声道的逆向建模。工作包括使用MRI轮廓和音频嵌入，构建Bi-LSTM模型，并评估不同音频特征和数据量的影响。**

- **链接: [https://arxiv.org/pdf/2603.28723](https://arxiv.org/pdf/2603.28723)**

> **作者:** Sofiane Azzouz; Pierre-André Vuissoz; Yves Laprie
>
> **摘要:** Articulatory-to-acoustic inversion strongly depends on the type of data used. While most previous studies rely on EMA, which is limited by the number of sensors and restricted to accessible articulators, we propose an approach aiming at a complete inversion of the vocal tract, from the glottis to the lips. To this end, we used approximately 3.5 hours of RT-MRI data from a single speaker. The innovation of our approach lies in the use of articulator contours automatically extracted from MRI images, rather than relying on the raw images themselves. By focusing on these contours, the model prioritizes the essential geometric dynamics of the vocal tract while discarding redundant pixel-level information. These contours, alongside denoised audio, were then processed using a Bi-LSTM architecture. Two experiments were conducted: (1) the analysis of the impact of the audio embedding, for which three types of embeddings were evaluated as input to the model (MFCCs, LCCs, and HuBERT), and (2) the study of the influence of the dataset size, which we varied from 10 minutes to 3.5 hours. Evaluation was performed on the test data using RMSE, median error, as well as Tract Variables, to which we added an additional measurement: the larynx height. The average RMSE obtained is 1.48\,mm, compared with the pixel size (1.62\,mm). These results confirm the feasibility of a complete vocal-tract inversion using RT-MRI data.
>
---
#### [new 011] Membership Inference Attacks against Large Audio Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于隐私安全任务，解决LALM的成员推断攻击问题。通过分析音频数据分布，提出可靠评估方法，揭示模型跨模态记忆机制。**

- **链接: [https://arxiv.org/pdf/2603.28378](https://arxiv.org/pdf/2603.28378)**

> **作者:** Jia-Kai Dong; Yu-Xiang Lin; Hung-Yi Lee
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** We present the first systematic Membership Inference Attack (MIA) evaluation of Large Audio Language Models (LALMs). As audio encodes non-semantic information, it induces severe train and test distribution shifts and can lead to spurious MIA performance. Using a multi-modal blind baseline based on textual, spectral, and prosodic features, we demonstrate that common speech datasets exhibit near-perfect train/test separability (AUC approximately 1.0) even without model inference, and the standard MIA scores strongly correlate with these blind acoustic artifacts (correlation greater than 0.7). Using this blind baseline, we identify that distribution-matched datasets enable reliable MIA evaluation without distribution shift confounds. We benchmark multiple MIA methods and conduct modality disentanglement experiments on these datasets. The results reveal that LALM memorization is cross-modal, arising only from binding a speaker's vocal identity with its text. These findings establish a principled standard for auditing LALMs beyond spurious correlations.
>
---
#### [new 012] A General Model for Deepfake Speech Detection: Diverse Bonafide Resources or Diverse AI-Based Generators
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于深度伪造语音检测任务，旨在解决模型泛化能力不足的问题。通过分析真实资源与生成器的影响，提出平衡数据集并验证其对模型性能的关键作用。**

- **链接: [https://arxiv.org/pdf/2603.27557](https://arxiv.org/pdf/2603.27557)**

> **作者:** Lam Pham; Khoi Vu; Dat Tran; David Fischinger; Simon Freitter; Marcel Hasenbalg; Davide Antonutti; Alexander Schindler; Martin Boyer; Ian McLoughlin
>
> **摘要:** In this paper, we analyze two main factors of Bonafide Resource (BR) or AI-based Generator (AG) which affect the performance and the generality of a Deepfake Speech Detection (DSD) model. To this end, we first propose a deep-learning based model, referred to as the baseline. Then, we conducted experiments on the baseline by which we indicate how Bonafide Resource (BR) and AI-based Generator (AG) factors affect the threshold score used to detect fake or bonafide input audio in the inference process. Given the experimental results, a dataset, which re-uses public Deepfake Speech Detection (DSD) datasets and shows a balance between Bonafide Resource (BR) or AI-based Generator (AG), is proposed. We then train various deep-learning based models on the proposed dataset and conduct cross-dataset evaluation on different benchmark datasets. The cross-dataset evaluation results prove that the balance of Bonafide Resources (BR) and AI-based Generators (AG) is the key factor to train and achieve a general Deepfake Speech Detection (DSD) model.
>
---
#### [new 013] Can pre-trained Deep Learning models predict groove ratings?
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决通过音频预测节奏感（groove）的问题。研究比较了深度学习模型与传统特征在预测groove上的效果，并分析了不同乐器的贡献。**

- **链接: [https://arxiv.org/pdf/2603.27237](https://arxiv.org/pdf/2603.27237)**

> **作者:** Axel Marmoret; Nicolas Farrugia; Jan Alexander Stupacher
>
> **备注:** Submitted to the SMC 2026 conference. 3 figures and 2 tables
>
> **摘要:** This study explores the extent to which deep learning models can predict groove and its related perceptual dimensions directly from audio signals. We critically examine the effectiveness of seven state-of-the-art deep learning models in predicting groove ratings and responses to groove-related queries through the extraction of audio embeddings. Additionally, we compare these predictions with traditional handcrafted audio features. To better understand the underlying mechanics, we extend this methodology to analyze predictions based on source-separated instruments, thereby isolating the contributions of individual musical elements. Our analysis reveals a clear separation of groove characteristics driven by the underlying musical style of the tracks (funk, pop, and rock). These findings indicate that deep audio representations can successfully encode complex, style-dependent groove components that traditional features often miss. Ultimately, this work highlights the capacity of advanced deep learning models to capture the multifaceted concept of groove, demonstrating the strong potential of representation learning to advance predictive Music Information Retrieval methodologies.
>
---
#### [new 014] Rhythmic segment analysis: Conceptualizing, visualizing, and measuring rhythmic data
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一种分析节奏数据的框架，通过区间段概念统一可视化方法并改进测量指标，解决节奏规律性分析问题。**

- **链接: [https://arxiv.org/pdf/2603.26988](https://arxiv.org/pdf/2603.26988)**

> **作者:** Bas Cornelissen
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** This paper develops a framework for conceptualizing, visualizing, and measuring regularities in rhythmic data. I propose to think about rhythmic data in terms of interval segments: fixed-length groups of consecutive intervals, which can be decomposed into a duration and a pattern (the ratios between the intervals). This simple conceptual framework unifies three rhythmic visualization methods and yields a fourth: the pattern-duration plot. When paired with a cluster transition network, it intuitively reveals regularities in both synthetic and real-world rhythmic data. Moreover, the framework generalizes two common measures of rhythmic structure: rhythm ratios and the normalized pairwise variability index (nPVI). In particular, nPVI can be reconstructed as the average distance from isochrony, and I propose a more general measure of anisochrony to replace it. Finally, the novel concept of quantality may shed light on wider debates regarding small-integer-ratio rhythms.
>
---
#### [new 015] Unsupervised Evaluation of Deep Audio Embeddings for Music Structure Analysis
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于音乐结构分析任务，旨在解决监督学习依赖标注数据的问题。通过无监督方法评估深度音频嵌入，比较不同分割算法效果，提出更严谨的评估标准。**

- **链接: [https://arxiv.org/pdf/2603.27218](https://arxiv.org/pdf/2603.27218)**

> **作者:** Axel Marmoret
>
> **备注:** Submitted to the SMC 2026 conference. 2 figures and 2 tables in the main document, 7 figures in Appendix
>
> **摘要:** Music Structure Analysis (MSA) aims to uncover the high-level organization of musical pieces. State-of-the-art methods are often based on supervised deep learning, but these methods are bottlenecked by the need for heavily annotated data and inherent structural ambiguities. In this paper, we propose an unsupervised evaluation of nine open-source, generic pre-trained deep audio models, on MSA. For each model, we extract barwise embeddings and segment them using three unsupervised segmentation algorithms (Foote's checkerboard kernels, spectral clustering, and Correlation Block-Matching (CBM)), focusing exclusively on boundary retrieval. Our results demonstrate that modern, generic deep embeddings generally outperform traditional spectrogram-based baselines, but not systematically. Furthermore, our unsupervised boundary estimation methodology generally yields stronger performance than recent linear probing baselines. Among the evaluated techniques, the CBM algorithm consistently emerges as the most effective downstream segmentation method. Finally, we highlight the artificial inflation of standard evaluation metrics and advocate for the systematic adoption of ``trimming'', or even ``double trimming'' annotations to establish more rigorous MSA evaluation standards.
>
---
#### [new 016] HASS: Hierarchical Simulation of Logopenic Aphasic Speech for Scalable PPA Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于PPA检测任务，旨在解决数据稀缺问题。通过构建HASS框架模拟logopenic variant PPA的多层级语言缺陷，生成更真实的训练数据。**

- **链接: [https://arxiv.org/pdf/2603.26795](https://arxiv.org/pdf/2603.26795)**

> **作者:** Harrison Li; Kevin Wang; Cheol Jun Cho; Jiachen Lian; Rabab Rangwala; Chenxu Guo; Emma Yang; Lynn Kurteff; Zoe Ezzes; Willa Keegan-Rodewald; Jet Vonk; Siddarth Ramkrishnan; Giada Antonicelli; Zachary Miller; Marilu Gorno Tempini; Gopala Anumanchipalli
>
> **摘要:** Building a diagnosis model for primary progressive aphasia (PPA) has been challenging due to the data scarcity. Collecting clinical data at scale is limited by the high vulnerability of clinical population and the high cost of expert labeling. To circumvent this, previous studies simulate dysfluent speech to generate training data. However, those approaches are not comprehensive enough to simulate PPA as holistic, multi-level phenotypes, instead relying on isolated dysfluencies. To address this, we propose a novel, clinically grounded simulation framework, Hierarchical Aphasic Speech Simulation (HASS). HASS aims to simulate behaviors of logopenic variant of PPA (lvPPA) with varying degrees of severity. To this end, semantic, phonological, and temporal deficits of lvPPA are systematically identified by clinical experts, and simulated. We demonstrate that our framework enables more accurate and generalizable detection models.
>
---
#### [new 017] Constructing Composite Features for Interpretable Music-Tagging
- **分类: cs.SD; cs.LG; cs.MM**

- **简介: 该论文属于音乐标签任务，旨在解决深度学习特征融合缺乏可解释性的问题。通过遗传编程自动构建可解释的复合特征，提升 tagging 性能。**

- **链接: [https://arxiv.org/pdf/2603.28644](https://arxiv.org/pdf/2603.28644)**

> **作者:** Chenhao Xue; Weitao Hu; Joyraj Chakraborty; Zhijin Guo; Kang Li; Tianyu Shi; Martin Reed; Nikolaos Thomos
>
> **备注:** 5 pages, 8 figures, accepted at ICASSP 2026
>
> **摘要:** Combining multiple audio features can improve the performance of music tagging, but common deep learning-based feature fusion methods often lack interpretability. To address this problem, we propose a Genetic Programming (GP) pipeline that automatically evolves composite features by mathematically combining base music features, thereby capturing synergistic interactions while preserving interpretability. This approach provides representational benefits similar to deep feature fusion without sacrificing interpretability. Experiments on the MTG-Jamendo and GTZAN datasets demonstrate consistent improvements compared to state-of-the-art systems across base feature sets at different abstraction levels. It should be noted that most of the performance gains are noticed within the first few hundred GP evaluations, indicating that effective feature combinations can be identified under modest search budgets. The top evolved expressions include linear, nonlinear, and conditional forms, with various low-complexity solutions at top performance aligned with parsimony pressure to prefer simpler expressions. Analyzing these composite features further reveals which interactions and transformations tend to be beneficial for tagging, offering insights that remain opaque in black-box deep models.
>
---
#### [new 018] Dual-branch Graph Domain Adaptation for Cross-scenario Multi-modal Emotion Recognition
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于多模态情感识别任务，解决跨场景下的模型泛化问题。提出DGDA框架，通过双分支图网络和领域对抗学习，提升模型在不同场景下的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.26840](https://arxiv.org/pdf/2603.26840)**

> **作者:** Yuntao Shou; Jun Zhou; Tao Meng; Wei Ai; Keqin Li
>
> **备注:** 29 pages
>
> **摘要:** Multimodal Emotion Recognition in Conversations (MERC) aims to predict speakers' emotional states in multi-turn dialogues through text, audio, and visual cues. In real-world settings, conversation scenarios differ significantly in speakers, topics, styles, and noise levels. Existing MERC methods generally neglect these cross-scenario variations, limiting their ability to transfer models trained on a source domain to unseen target domains. To address this issue, we propose a Dual-branch Graph Domain Adaptation framework (DGDA) for multimodal emotion recognition under cross-scenario conditions. We first construct an emotion interaction graph to characterize complex emotional dependencies among utterances. A dual-branch encoder, consisting of a hypergraph neural network (HGNN) and a path neural network (PathNN), is then designed to explicitly model multivariate relationships and implicitly capture global dependencies. To enable out-of-domain generalization, a domain adversarial discriminator is introduced to learn invariant representations across domains. Furthermore, a regularization loss is incorporated to suppress the negative influence of noisy labels. To the best of our knowledge, DGDA is the first MERC framework that jointly addresses domain shift and label noise. Theoretical analysis provides tighter generalization bounds, and extensive experiments on IEMOCAP and MELD demonstrate that DGDA consistently outperforms strong baselines and better adapts to cross-scenario conversations. Our code is available at this https URL.
>
---
#### [new 019] Investigation on the Robustness of Acoustic Foundation Models on Post Exercise Speech
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，研究运动后语音的鲁棒性问题。通过对比不同模型在运动后语音上的表现，探讨模型适应性和语言流畅性影响。**

- **链接: [https://arxiv.org/pdf/2603.27508](https://arxiv.org/pdf/2603.27508)**

> **作者:** Xiangyuan Xue; Yuyu Wang; Ruijie Yao; Xiaoyue Ni; Xiaofan Jiang; Jingping Nie
>
> **摘要:** Automatic speech recognition (ASR) has been extensively studied on neutral and stationary speech, yet its robustness under post-exercise physiological shift remains underexplored. Compared with resting speech, post-exercise speech often contains micro-breaths, non-semantic pauses, unstable phonation, and repetitions caused by reduced breath support, making transcription more difficult. In this work, we benchmark acoustic foundation models on post-exercise speech under a unified evaluation protocol. We compare sequence-to-sequence models (Whisper and FunASR/Paraformer) and self-supervised encoders with CTC decoding (Wav2Vec2, HuBERT, and WavLM), under both off-the-shelf inference and post-exercise in-domain fine-tuning. Across the Static/Post-All benchmark, most models degrade on post-exercise speech, while FunASR shows the strongest baseline robustness at 14.57% WER and 8.21% CER on Post-All. Fine-tuning substantially improves several CTC-based models, whereas Whisper shows unstable adaptation. As an exploratory case study, we further stratify results by fluent and non-fluent speakers; although the non-fluent subset is small, it is consistently more challenging than the fluent subset. Overall, our findings show that post-exercise ASR robustness is strongly model-dependent, that in-domain adaptation can be highly effective but not uniformly stable, and that future post-exercise ASR studies should explicitly separate fluency-related effects from exercise-induced speech variation.
>
---
#### [new 020] AFSS: Artifact-Focused Self-Synthesis for Mitigating Bias in Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决检测器在未见数据上泛化能力差的偏差问题。通过生成伪假样本并强制同说话人约束，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.26856](https://arxiv.org/pdf/2603.26856)**

> **作者:** Hai-Son Nguyen-Le; Hung-Cuong Nguyen-Thanh; Nhien-An Le-Khac; Dinh-Thuc Nguyen; Hong-Hanh Nguyen-Le
>
> **备注:** Accepted at International Joint Conference on Neural Networks 2026
>
> **摘要:** The rapid advancement of generative models has enabled highly realistic audio deepfakes, yet current detectors suffer from a critical bias problem, leading to poor generalization across unseen datasets. This paper proposes Artifact-Focused Self-Synthesis (AFSS), a method designed to mitigate this bias by generating pseudo-fake samples from real audio via two mechanisms: self-conversion and self-reconstruction. The core insight of AFSS lies in enforcing same-speaker constraints, ensuring that real and pseudo-fake samples share identical speaker identity and semantic content. This forces the detector to focus exclusively on generation artifacts rather than irrelevant confounding factors. Furthermore, we introduce a learnable reweighting loss to dynamically emphasize synthetic samples during training. Extensive experiments across 7 datasets demonstrate that AFSS achieves state-of-the-art performance with an average EER of 5.45\%, including a significant reduction to 1.23\% on WaveFake and 2.70\% on In-the-Wild, all while eliminating the dependency on pre-collected fake datasets. Our code is publicly available at this https URL.
>
---
#### [new 021] Two-Stage Acoustic Adaptation with Gated Cross-Attention Adapters for LLM-Based Multi-Talker Speech Recognition
- **分类: cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在提升LLM在复杂场景下的性能。通过引入声学特征增强解码过程，提出两阶段适配框架，提高识别鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27205](https://arxiv.org/pdf/2603.27205)**

> **作者:** Hao Shi; Yuan Gao; Xugang Lu; Tatsuya Kawahara
>
> **摘要:** Large Language Models (LLMs) are strong decoders for Serialized Output Training (SOT) in two-talker Automatic Speech Recognition (ASR), yet their performance degrades substantially in challenging conditions such as three-talker mixtures. A key limitation is that current systems inject acoustic evidence only through a projected prefix, which can be lossy and imperfectly aligned with the LLM input space, providing insufficient fine-grained grounding during decoding. Addressing this limitation is crucial for robust multi-talker ASR, especially in three-talker mixtures. This paper improves LLM-based multi-talker ASR by explicitly injecting talker-aware acoustic evidence into the decoder. We first revisit Connectionist Temporal Classification (CTC)-derived prefix prompting and compare three variants with increasing acoustic content. The CTC information is obtained using the serialized CTC proposed in our previous works. While acoustic-enriched prompts outperform the SOT-only baseline, prefix-only conditioning remains inadequate for three-talker mixtures. We therefore propose a lightweight gated residual cross-attention adapter and design a two-stage acoustic adaptation framework based on low-rank updates (LoRA). In Stage 1, we insert gated cross-attention adapters after the self-attention sub-layer to stably inject acoustic embeddings as external memory. In Stage 2, we refine both the cross-attention adapters and the pretrained LLM's self-attention projections using parameter-efficient LoRA, improving robustness for large backbones under limited data; the learned updates are merged into the base weights for inference. Experiments on Libri2Mix/Libri3Mix under clean and noisy conditions show consistent gains, with particularly large improvements in three-talker settings.
>
---
#### [new 022] PHONOS: PHOnetic Neutralization for Online Streaming Applications
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于语音匿名化任务，旨在解决非母语口音影响匿名性的问题。通过生成本土化语音样本，训练实时模块以中和口音，提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2603.27001](https://arxiv.org/pdf/2603.27001)**

> **作者:** Waris Quamer; Mu-Ruei Tseng; Ghady Nasrallah; Ricardo Gutierrez-Osuna
>
> **备注:** The paper is submitted to Interspeech 2026 and currently under review
>
> **摘要:** Speaker anonymization (SA) systems modify timbre while leaving regional or non-native accents intact, which is problematic because accents can narrow the anonymity set. To address this issue, we present PHONOS, a streaming module for real-time SA that neutralizes non-native accent to sound native-like. Our approach pre-generates golden speaker utterances that preserve source timbre and rhythm but replace foreign segmentals with native ones using silence-aware DTW alignment and zero-shot voice conversion. These utterances supervise a causal accent translator that maps non-native content tokens to native equivalents with at most 40ms look-ahead, trained using joint cross-entropy and CTC losses. Our evaluations show an 81% reduction in non-native accent confidence, with listening-test ratings consistent with this shift, and reduced speaker linkability as accent-neutralized utterances move away from the original speaker in embedding space while having latency under 241 ms on single GPU.
>
---
#### [new 023] Multilingual Stutter Event Detection for English, German, and Mandarin Speech
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决多语言口吃检测问题。通过多语种数据训练模型，捕捉口吃的共性特征，提升检测的泛化能力与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.26939](https://arxiv.org/pdf/2603.26939)**

> **作者:** Felix Haas; Sebastian P. Bayerl
>
> **摘要:** This paper presents a multi-label stuttering detection system trained on multi-corpus, multilingual data in English, German, and this http URL leveraging annotated stuttering data from three languages and four corpora, the model captures language-independent characteristics of stuttering, enabling robust detection across linguistic contexts. Experimental results demonstrate that multilingual training achieves performance comparable to and, in some cases, even exceeds that of previous systems. These findings suggest that stuttering exhibits cross-linguistic consistency, which supports the development of language-agnostic detection systems. Our work demonstrates the feasibility and advantages of using multilingual data to improve generalizability and reliability in automated stuttering detection.
>
---
#### [new 024] ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出ParaSpeechCLAP，一个双编码器模型，用于语音与文本风格描述的联合嵌入，解决跨模态风格对齐问题。通过专门化和统一模型提升风格分类与检索性能。**

- **链接: [https://arxiv.org/pdf/2603.28737](https://arxiv.org/pdf/2603.28737)**

> **作者:** Anuj Diwan; Eunsol Choi; David Harwath
>
> **备注:** Under review
>
> **摘要:** We introduce ParaSpeechCLAP, a dual-encoder contrastive model that maps speech and text style captions into a common embedding space, supporting a wide range of intrinsic (speaker-level) and situational (utterance-level) descriptors (such as pitch, texture and emotion) far beyond the narrow set handled by existing models. We train specialized ParaSpeechCLAP-Intrinsic and ParaSpeechCLAP-Situational models alongside a unified ParaSpeechCLAP-Combined model, finding that specialization yields stronger performance on individual style dimensions while the unified model excels on compositional evaluation. We further show that ParaSpeechCLAP-Intrinsic benefits from an additional classification loss and class-balanced training. We demonstrate our models' performance on style caption retrieval, speech attribute classification and as an inference-time reward model that improves style-prompted TTS without additional training. ParaSpeechCLAP outperforms baselines on most metrics across all three applications. Our models and code are released at this https URL .
>
---
#### [new 025] VAANI: Capturing the language landscape for an inclusive digital India
- **分类: eess.AS**

- **简介: 该论文介绍VAANI项目，旨在构建涵盖印度165个地区的多模态数据集，解决语言多样性不足的问题，通过收集语音和图像数据，促进印度的言语包容性研究。**

- **链接: [https://arxiv.org/pdf/2603.28714](https://arxiv.org/pdf/2603.28714)**

> **作者:** Sujith Pulikodan; Abhayjeet Singh; Agneedh Basu; Lokesh Rady; Nihar Desai; Pavan Kumar J; Prajjwal Srivastav; Pranav D Bhat; Raghu Dharmaraju; Ritika Gupta; Sathvik Udupa; Saurabh Kumar; Sumit Sharma; Vaibhav Vishwakarma; Visruth Sanka; Dinesh Tewari; Harsh Dhand; Amrita Kamat; Sukhwinder Singh; Shikhar Vashishth; Partha Talukdar; Raj Acharya; Prasanta Kumar Ghosh
>
> **摘要:** Project VAANI is an initiative to create an India-representative multi-modal dataset that comprehensively maps India's linguistic diversity, starting with 165 districts across the country in its first two phases. Speech data is collected through a carefully structured process that uses image-based prompts to encourage spontaneous responses. Images are captured through a separate process that encompasses a broad range of topics, gathered from both within and across districts. The collected data undergoes a rigorous multi-stage quality evaluation, including both automated and manual checks to ensure highest possible standards in audio quality and transcription accuracy. Following this thorough validation, we have open-sourced around 289K images, approximately 31,270 hours of audio recordings, and around 2,067 hours of transcribed speech, encompassing 112 languages from 165 districts from 31 States and Union territories. Notably, significant of these languages are being represented for the first time in a dataset of this scale, making the VAANI project a groundbreaking effort in preserving and promoting linguistic inclusivity. This data can be instrumental in building inclusive speech models for India, and in advancing research and development across speech, image, and multimodal applications.
>
---
#### [new 026] Algo Pärt: An Algorithmic Reconstruction of Arvo Pärt's Summa
- **分类: cs.SD**

- **简介: 该论文属于音乐分析任务，旨在解析阿沃·帕特的《Summa》是否为算法化作品。通过构建算法重建乐谱，验证其高度算法性。**

- **链接: [https://arxiv.org/pdf/2603.26989](https://arxiv.org/pdf/2603.26989)**

> **作者:** Bas Cornelissen
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Arvo Pärt is one of the most popular contemporary composers, known for his highly original tintinnabuli style. Works in this style are typically composed according to precise procedures and have even been described as algorithmic compositions. To understand how algorithmic Pärt's music exactly is, this paper presents an analysis by synthesis: it proposes an algorithm that almost completely reconstructs the score of Summa, his "most strictly constructed and most encrypted work," according to Pärt himself in 1994. The piece is analyzed and then formalized using so-called tintinnabuli processes. An implementation of the resulting algorithm generates a musical score matching Summa in over 93% of the notes. Due to interdependencies between the voices, only half of the mistakes (3.5%) need to be corrected to reproduce the original score faithfully. This study shows that Summa is a largely algorithmic composition and offers new perspectives on the music of Arvo Pärt.
>
---
#### [new 027] Can Hierarchical Cross-Modal Fusion Predict Human Perception of AI Dubbed Content?
- **分类: eess.AS**

- **简介: 该论文属于AI配音质量评估任务，旨在解决人工评分成本高、难以大规模应用的问题。通过多模态融合与轻量微调，实现对AI配音内容的自动感知评价。**

- **链接: [https://arxiv.org/pdf/2603.28717](https://arxiv.org/pdf/2603.28717)**

> **作者:** Ashwini Dasare; Nirmesh Shah; Ashishkumar Gudmalwar; Pankaj Wasnik
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Evaluating AI generated dubbed content is inherently multi-dimensional, shaped by synchronization, intelligibility, speaker consistency, emotional alignment, and semantic context. Human Mean Opinion Scores (MOS) remain the gold standard but are costly and impractical at scale. We present a hierarchical multimodal architecture for perceptually meaningful dubbing evaluation, integrating complementary cues from audio, video, and text. The model captures fine-grained features such as speaker identity, prosody, and content from audio, facial expressions and scene-level cues from video and semantic context from text, which are progressively fused through intra and inter-modal layers. Lightweight LoRA adapters enable parameter-efficient fine-tuning across modalities. To overcome limited subjective labels, we derive proxy MOS by aggregating objective metrics with weights optimized via active learning. The proposed architecture was trained on 12k Hindi-English bidirectional dubbed clips, followed by fine-tuning with human MOS. Our approach achieves strong perceptual alignment (PCC > 0.75), providing a scalable solution for automatic evaluation of AI-dubbed content.
>
---
#### [new 028] On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究ASR任务中Whisper编码器的剪枝与LoRA微调效果，旨在提升模型效率并保持性能。通过实验验证剪枝2层仅导致2-4% WER上升，结合LoRA可进一步优化。**

- **链接: [https://arxiv.org/pdf/2603.27981](https://arxiv.org/pdf/2603.27981)**

> **作者:** Ganesh Pavan Kartikeya Bharadwaj Kolluri; Michael Kampouridis; Ravi Shekhar
>
> **备注:** Accepted at SPEAKABLE Workshop, LREC 2026
>
> **摘要:** Automatic speech recognition (ASR) has advanced rapidly in recent years, driven by large-scale pretrained models and end-to-end architectures such as SLAM-ASR. A key component of SLAM-ASR systems is the Whisper speech encoder, which provides robust acoustic representations. While model pruning has been explored for the full Whisper encoder-decoder architecture, its impact within the SLAM-ASR setting remains under-investigated. In this work, we analyze the effects of layer pruning in the Whisper encoder when used as the acoustic backbone of SLAM-ASR. We further examine the extent to which LoRA-based fine-tuning can recover performance degradation caused by pruning. Experiments conducted across three Whisper variants (Small, Medium, Large-v2), three languages representing distinct resource levels (Danish, Dutch, English), and over 200 training runs demonstrate that pruning two encoder layers causes only 2-4% WER degradation, and that combining this pruning with LoRA adaptation consistently outperforms the unpruned baseline while reducing total parameters by 7-14%. Moreover, our error analysis reveals that LoRA primarily compensates through the language model's linguistic priors, reducing total word errors by 11-21% for Dutch and English, with substitutions and deletions showing the largest reductions. However, for low-resource Danish, the reduction is smaller (4-7%), and LoRA introduces increased insertion errors, indicating that compensation effectiveness depends on the LLM's pre-existing language proficiency and available training data.
>
---
#### [new 029] TokenDance: Token-to-Token Music-to-Dance Generation with Bidirectional Mamba
- **分类: cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于音乐到舞蹈生成任务，旨在解决现有模型泛化能力差、生成舞蹈单一的问题。通过双模态分词和双向Mamba架构，提升生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.27314](https://arxiv.org/pdf/2603.27314)**

> **作者:** Ziyue Yang; Kaixing Yang; Xulong Tang
>
> **备注:** CVPR2026 Workshop on HuMoGen
>
> **摘要:** Music-to-dance generation has broad applications in virtual reality, dance education, and digital character animation. However, the limited coverage of existing 3D dance datasets confines current models to a narrow subset of music styles and choreographic patterns, resulting in poor generalization to real-world music. Consequently, generated dances often become overly simplistic and repetitive, substantially degrading expressiveness and realism. To tackle this problem, we present TokenDance, a two-stage music-to-dance generation framework that explicitly addresses this limitation through dual-modality tokenization and efficient token-level generation. In the first stage, we discretize both dance and music using Finite Scalar Quantization, where dance motions are factorized into upper and lower-body components with kinematic-dynamic constraints, and music is decomposed into semantic and acoustic features with dedicated codebooks to capture choreography-specific structures. In the second stage, we introduce a Local-Global-Local token-to-token generator built on a Bidirectional Mamba backbone, enabling coherent motion synthesis, strong music-dance alignment, and efficient non-autoregressive inference. Extensive experiments demonstrate that TokenDance achieves overall state-of-the-art (SOTA) performance in both generation quality and inference speed, highlighting its effectiveness and practical value for real-world music-to-dance applications.
>
---
#### [new 030] HumMusQA: A Human-written Music Understanding QA Benchmark Dataset
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs评估标准不足的问题。构建了人工编写的问题数据集，用于更准确地测试模型的音乐理解能力。**

- **链接: [https://arxiv.org/pdf/2603.27877](https://arxiv.org/pdf/2603.27877)**

> **作者:** Benno Weck; Pablo Puentes; Andrea Poltronieri; Satyajeet Prabhu; Dmitry Bogdanov
>
> **备注:** Dataset available at this https URL
>
> **摘要:** The evaluation of music understanding in Large Audio-Language Models (LALMs) requires a rigorously defined benchmark that truly tests whether models can perceive and interpret music, a standard that current data methodologies frequently fail to meet. This paper introduces a meticulously structured approach to music evaluation, proposing a new dataset of 320 hand-written questions curated and validated by experts with musical training, arguing that such focused, manual curation is superior for probing complex audio comprehension. To demonstrate the use of the dataset, we benchmark six state-of-the-art LALMs and additionally test their robustness to uni-modal shortcuts.
>
---
#### [new 031] SonoWorld: From One Image to a 3D Audio-Visual Scene
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出Image2AVScene任务，旨在从单张图像生成3D音视频场景。工作包括生成全景图、构建3D场景、放置声音锚点并渲染空间音频，实现音视频同步的沉浸式体验。**

- **链接: [https://arxiv.org/pdf/2603.28757](https://arxiv.org/pdf/2603.28757)**

> **作者:** Derong Jin; Xiyi Chen; Ming C. Lin; Ruohan Gao
>
> **备注:** Accepted by CVPR 2026, project page: this https URL
>
> **摘要:** Tremendous progress in visual scene generation now turns a single image into an explorable 3D world, yet immersion remains incomplete without sound. We introduce Image2AVScene, the task of generating a 3D audio-visual scene from a single image, and present SonoWorld, the first framework to tackle this challenge. From one image, our pipeline outpaints a 360° panorama, lifts it into a navigable 3D scene, places language-guided sound anchors, and renders ambisonics for point, areal, and ambient sources, yielding spatial audio aligned with scene geometry and semantics. Quantitative evaluations on a newly curated real-world dataset and a controlled user study confirm the effectiveness of our approach. Beyond free-viewpoint audio-visual rendering, we also demonstrate applications to one-shot acoustic learning and audio-visual spatial source separation. Project website: this https URL
>
---
## 更新

#### [replaced 001] POTSA: A Cross-Lingual Speech Alignment Framework for Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，旨在解决多语言翻译中的语义偏差问题。提出POTSA框架，通过跨语言对齐和最优传输技术提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.09232](https://arxiv.org/pdf/2511.09232)**

> **作者:** Xuanchen Li; Chenrui Cui; Tianrui Wang; Meng Ge; Zikang Huang; Jin Li; Yizhou Peng; Yuheng Lu; Nyima Tashi; Longbiao Wang; Jianwu Dang
>
> **摘要:** Speech Large Language Models have achieved breakthroughs in multilingual speech-to-text translation. However, existing approaches often overlook semantic commonalities across source languages, leading to biased translation performance. In this work, we propose POTSA (Parallel Optimal Transport for Speech Alignment), a new framework based on cross-lingual parallel speech pairs and Optimal Transport, designed to bridge high- and low-resource translation gaps. First, we introduce a Bias Compensation module to coarsely align initial speech representations. Second, we impose token-level OT constraints on a Q-Former using parallel pairs to establish fine-grained representation consistency. Then, we apply a layer scheduling strategy to focus OT constraints on semantically beneficial layers. Experiments on FLEURS show our method achieves SOTA performance, with +1.29 BLEU over five common languages and +2.93 BLEU on zero-shot languages, using only 10 hours of parallel speech per language.
>
---
#### [replaced 002] Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation
- **分类: cs.SD; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于自动和弦识别任务，解决标注数据稀缺问题。通过伪标签和知识蒸馏方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.19778](https://arxiv.org/pdf/2602.19778)**

> **作者:** Nghia Phan; Rong Jin; Gang Liu; Xiao Dong
>
> **备注:** 8 pages, 6 figures, 3 tables
>
> **摘要:** Automatic Chord Recognition (ACR) is constrained by the scarcity of aligned chord labels, as well-aligned annotations are costly to acquire. At the same time, open-weight pre-trained models are currently more accessible than their proprietary training data. In this work, we present a two-stage training pipeline that leverages pre-trained models together with unlabeled audio. The proposed method decouples training into two stages. In the first stage, we use a pre-trained BTC model as a teacher to generate pseudo-labels for over 1,000 hours of diverse unlabeled audio and train a student model solely on these pseudo-labels. In the second stage, the student is continually trained on ground-truth labels as they become available. To prevent catastrophic forgetting of the representations learned in the first stage, we apply selective knowledge distillation (KD) from the teacher as a regularizer. In our experiments, two models (BTC, 2E1D) were used as students. In stage 1, using only pseudo-labels, the BTC student achieves over 99% of the teacher's performance, while the 2E1D model achieves about 97% across seven standard mir_eval metrics. After a single training run for both students in stage 2, the resulting BTC student model surpasses the traditional supervised learning baseline by 2.5% and the original pre-trained teacher model by 1.1-3.2% across all metrics. The resulting 2E1D student model improves over the traditional supervised learning baseline by 2.67% on average and achieves almost the same performance as the teacher. Both cases show large gains on rare chord qualities.
>
---
#### [replaced 003] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决多语言和长文本评估的可复现性问题。构建了Open ASR Leaderboard平台，对比多种系统，标准化评估指标，促进透明化研究。**

- **链接: [https://arxiv.org/pdf/2510.06961](https://arxiv.org/pdf/2510.06961)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Nithin Rao Koluguri; Piotr Żelasko; Somshubra Majumdar; Adel Moumen; Sanchit Gandhi
>
> **备注:** Leaderboard: this https URL ; Code: this https URL
>
> **摘要:** We present the Open ASR Leaderboard, a reproducible benchmarking platform with community contributions from academia and industry. It compares 86 open-source and proprietary systems across 12 datasets, with English short- and long-form and multilingual short-form tracks. We standardize word error rate (WER) and inverse real-time factor (RTFx) evaluation for consistent accuracy-efficiency comparisons across model architectures and toolkits (e.g., ESPNet, NeMo, SpeechBrain, Transformers). We observe that Conformer-based encoders paired with transformer-based decoders achieve the best average WER, while connectionist temporal classification (CTC) and token-and-duration transducer (TDT) decoders offer superior RTFx, making them better suited for long-form and batched processing. All code and dataset loaders are open-sourced to support transparent, extensible evaluation. We present our evaluation methodology to facilitate community-driven benchmarking in ASR and other tasks.
>
---
#### [replaced 004] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出UniLS，解决音频驱动的虚拟人对话任务中监听动作不自然的问题。通过两阶段训练，实现端到端的说话与倾听表达生成。**

- **链接: [https://arxiv.org/pdf/2512.09327](https://arxiv.org/pdf/2512.09327)**

> **作者:** Xuangeng Chu; Ruicong Liu; Yifei Huang; Yun Liu; Yichen Peng; Bo Zheng
>
> **备注:** CVPR 2026, code is available at this https URL, more demos are available at this https URL
>
> **摘要:** Generating lifelike conversational avatars requires modeling not just isolated speakers, but the dynamic, reciprocal interaction of speaking and listening. However, modeling the listener is exceptionally challenging: direct audio-driven training fails, producing stiff, static listening motions. This failure stems from a fundamental imbalance: the speaker's motion is strongly driven by speech audio, while the listener's motion primarily follows an internal motion prior and is only loosely guided by external speech. This challenge has led most methods to focus on speak-only generation. The only prior attempt at joint generation relies on extra speaker's motion to produce the listener. This design is not end-to-end, thereby hindering the real-time applicability. To address this limitation, we present UniLS, the first end-to-end framework for generating unified speak-listen expressions, driven by only dual-track audio. Our method introduces a novel two-stage training paradigm. Stage 1 first learns the internal motion prior by training an audio-free autoregressive generator, capturing the spontaneous dynamics of natural facial motion. Stage 2 then introduces the dual-track audio, fine-tuning the generator to modulate the learned motion prior based on external speech cues. Extensive evaluations show UniLS achieves state-of-the-art speaking accuracy. More importantly, it delivers up to 44.1\% improvement in listening metrics, generating significantly more diverse and natural listening expressions. This effectively mitigates the stiffness problem and provides a practical, high-fidelity audio-driven solution for interactive digital humans. Code and demos are available at this https URL.
>
---
#### [replaced 005] PAVAS: Physics-Aware Video-to-Audio Synthesis
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频到音频生成任务，旨在解决现有模型忽略物理因素的问题。提出PAVAS方法，结合物理推理生成更真实的音频。**

- **链接: [https://arxiv.org/pdf/2512.08282](https://arxiv.org/pdf/2512.08282)**

> **作者:** Oh Hyun-Bin; Yuhta Takida; Toshimitsu Uesaka; Tae-Hyun Oh; Yuki Mitsufuji
>
> **摘要:** Recent advances in Video-to-Audio (V2A) generation have achieved impressive perceptual quality and temporal synchronization, yet most models remain appearance-driven, capturing visual-acoustic correlations without considering the physical factors that shape real-world sounds. We present Physics-Aware Video-to-Audio Synthesis (PAVAS), a method that incorporates physical reasoning into a latent diffusion-based V2A generation through the Physics-Driven Audio Adapter (Phy-Adapter). The adapter receives object-level physical parameters estimated by the Physical Parameter Estimator (PPE), which uses a Vision-Language Model (VLM) to infer the moving-object mass and a segmentation-based dynamic 3D reconstruction module to recover its motion trajectory for velocity computation. These physical cues enable the model to synthesize sounds that reflect underlying physical factors. To assess physical realism, we curate VGG-Impact, a benchmark focusing on object-object interactions, and introduce Audio-Physics Correlation Coefficient (APCC), an evaluation metric that measures consistency between physical and auditory attributes. Comprehensive experiments show that PAVAS produces physically plausible and perceptually coherent audio, outperforming existing V2A models in both quantitative and qualitative evaluations. Visit this https URL for demo videos.
>
---
#### [replaced 006] Sommelier: Scalable Open Multi-turn Audio Pre-processing for Full-duplex Speech Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决全双工语音语言模型的数据瓶颈问题。针对多说话人对话数据稀缺及处理难题，提出一个可扩展的开源预处理管道。**

- **链接: [https://arxiv.org/pdf/2603.25750](https://arxiv.org/pdf/2603.25750)**

> **作者:** Kyudan Jung; Jihwan Kim; Soyoon Kim; Jeonghoon Kim; Jaegul Choo; Cheonbok Park
>
> **备注:** 34 pages, 7 figures, 11 tables
>
> **摘要:** As the paradigm of AI shifts from text-based LLMs to Speech Language Models (SLMs), there is a growing demand for full-duplex systems capable of real-time, natural human-computer interaction. However, the development of such models is constrained by the scarcity of high-quality, multi-speaker conversational data, as existing large-scale resources are predominantly single-speaker or limited in volume. Addressing the complex dynamics of natural dialogue, such as overlapping and back-channeling remains a challenge, with standard processing pipelines suffering from diarization errors and ASR hallucinations. To bridge this gap, we present a robust and scalable open-source data processing pipeline designed for full-duplex model.
>
---
#### [replaced 007] Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset
- **分类: cs.GR; cs.CV; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法语义控制不足和长序列不连贯的问题。提出LRCM框架，结合扩散模型与Mamba模块，实现多模态引导的自回归舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2601.03323](https://arxiv.org/pdf/2601.03323)**

> **作者:** Oran Duan; Yinghua Shen; Yingzhu Lv; Luyang Jie; Yaxin Liu; Qiong Wu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Advances in generative models and sequence learning have greatly promoted research in dance motion generation, yet current methods still suffer from coarse semantic control and poor coherence in long sequences. In this work, we present Listen to Rhythm, Choose Movements (LRCM), a multimodal-guided diffusion framework supporting both diverse input modalities and autoregressive dance motion generation. We explore a feature decoupling paradigm for dance datasets and generalize it to the Motorica Dance dataset, separating motion capture data, audio rhythm, and professionally annotated global and local text descriptions. Our diffusion architecture integrates an audio-latent Conformer and a text-latent Cross-Conformer, and incorporates a Motion Temporal Mamba Module (MTMM) to enable smooth, long-duration autoregressive synthesis. Experimental results indicate that LRCM delivers strong performance in both functional capability and quantitative metrics, demonstrating notable potential in multimodal input scenarios and extended sequence generation. We will release the full codebase, dataset, and pretrained models publicly upon acceptance.
>
---
#### [replaced 008] Foundation Models for Bioacoustics -- a Comparative Review
- **分类: cs.SD; cs.LG; eess.AS; q-bio.QM**

- **简介: 该论文属于生物声学领域，研究预训练模型在分类任务中的迁移能力，通过实验比较不同模型性能，为模型选择提供指导。**

- **链接: [https://arxiv.org/pdf/2508.01277](https://arxiv.org/pdf/2508.01277)**

> **作者:** Raphael Schwinger; Paria Vali Zadeh; Lukas Rauch; Mats Kurz; Tom Hauschild; Sam Lapp; Sven Tomforde
>
> **备注:** Preprint
>
> **摘要:** Automated bioacoustic analysis is essential for biodiversity monitoring and conservation, requiring advanced deep learning models that can adapt to diverse bioacoustic tasks. This article presents a comprehensive review of large-scale pretrained bioacoustic foundation models and systematically investigates their transferability across multiple bioacoustic classification tasks. We overview bioacoustic representation learning by analysing pretraining data sources and benchmarks. On this basis, we review bioacoustic foundation models, dissecting the models' training data, preprocessing, augmentations, architecture, and training paradigm. Additionally, we conduct an extensive empirical study of selected models on the BEANS and BirdSet benchmarks, evaluating generalisability under linear and attentive probing. Our experimental analysis reveals that Perch~2.0 achieves the highest BirdSet score (restricted evaluation) and the strongest linear probing result on BEANS, building on diverse multi-taxa supervised pretraining; that BirdMAE is the best model among probing-based strategies on BirdSet and second on BEANS after BEATs$_{NLM}$, the encoder of NatureLM-audio; that attentive probing is beneficial to extract the full performance of transformer-based models; and that general-purpose audio models trained with self-supervised learning on AudioSet outperform many specialised bird sound models on BEANS when evaluated with attentive probing. These findings provide valuable guidance for practitioners selecting appropriate models to adapt them to new bioacoustic classification tasks via probing.
>
---
#### [replaced 009] Acoustic Overspecification in Electronic Dance Music Taxonomy
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于音乐分类任务，旨在解决EDM分类中商业标签是否反映真实声学差异的问题。通过无监督方法发现EDM的自然声学结构，表明现有分类过于细分。**

- **链接: [https://arxiv.org/pdf/2509.11474](https://arxiv.org/pdf/2509.11474)**

> **作者:** Weilun Xu; Tianhao Dai; Oscar Goudet; Xiaoxuan Wang
>
> **摘要:** Electronic Dance Music (EDM) classification typically relies on industry-defined taxonomies, with current supervised approaches naturally assuming the validity of prescribed subgenre labels. However, whether these commercial distinctions reflect genuine acoustic differences remains largely unexplored. In this paper, we propose an unsupervised approach to discover the natural acoustic structure of EDM independent of commercial labels. To address the historical lack of EDM-specific feature design in MIR, we systematically construct a tailored, interpretable acoustic feature space capturing the genre's defining production techniques, spectral textures, and layered rhythmic patterns. To ensure our findings reflect inherent acoustic structure rather than feature engineering artifacts, we validate our clustering against state-of-the-art pre-trained audio embeddings (MERT and CLAP). Across both our bespoke feature space and the pre-trained embeddings, clustering consistently identifies 20 or fewer natural acoustic families -- suggesting current commercial EDM taxonomy is acoustically overspecified by nearly one-half.
>
---
#### [replaced 010] Nwāchā Munā: A Devanagari Speech Corpus and Proximal Transfer Benchmark for Nepal Bhasha ASR
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决尼泊尔语资源匮乏问题。构建了首个尼泊尔语语音语料库，并通过邻近语言迁移提升ASR性能。**

- **链接: [https://arxiv.org/pdf/2603.07554](https://arxiv.org/pdf/2603.07554)**

> **作者:** Rishikesh Kumar Sharma; Safal Narshing Shrestha; Jenny Poudel; Rupak Tiwari; Arju Shrestha; Rupak Raj Ghimire; Bal Krishna Bal
>
> **备注:** Accepted in CHiPSAL@LREC 2026
>
> **摘要:** Nepal Bhasha (Newari), an endangered language of the Kathmandu Valley, remains digitally marginalized due to the severe scarcity of annotated speech resources. In this work, we introduce Nwāchā Munā, a newly curated 5.39-hour manually transcribed Devanagari speech corpus for Nepal Bhasha, and establish the first benchmark using script-preserving acoustic modeling. We investigate whether proximal cross-lingual transfer from a geographically and linguistically adjacent language (Nepali) can rival large-scale multilingual pretraining in an ultra-low-resource Automatic Speech Recognition (ASR) setting. Fine-tuning a Nepali Conformer model reduces the Character Error Rate (CER) from a 52.54% zero-shot baseline to 17.59% with data augmentation, effectively matching the performance of the multilingual Whisper-Small model despite utilizing significantly fewer parameters. Our findings demonstrate that proximal transfer from Nepali language serves as a computationally efficient alternative to massive multilingual models. We openly release the dataset and benchmarks to digitally enable the Newari community and foster further research in Nepal Bhasha.
>
---
#### [replaced 011] DiffAU: Diffusion-Based Ambisonics Upscaling
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频信号处理任务，旨在解决Ambisonics上采样问题。通过引入扩散模型，提出DiffAU方法，从一阶Ambisonics生成三阶Ambisonics，提升空间音效的分辨率与真实感。**

- **链接: [https://arxiv.org/pdf/2510.00180](https://arxiv.org/pdf/2510.00180)**

> **作者:** Amit Milstein; Nir Shlezinger; Boaz Rafaely
>
> **摘要:** Spatial audio enhances immersion by reproducing 3D sound fields, with Ambisonics offering a scalable format for this purpose. While first-order Ambisonics (FOA) notably facilitates hardware-efficient acquisition and storage of sound fields as compared to high-order Ambisonics (HOA), its low spatial resolution limits realism, highlighting the need for Ambisonics upscaling (AU) as an approach for increasing the order of Ambisonics signals. In this work we propose DiffAU, a cascaded AU method that leverages recent developments in diffusion models combined with novel adaptation to spatial audio to generate 3rd order Ambisonics from FOA. By learning data distributions, DiffAU provides a principled approach that rapidly and reliably reproduces HOA in various settings. Experiments in anechoic conditions with multiple speakers, show strong objective and perceptual performance.
>
---
#### [replaced 012] X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音大模型对齐任务，旨在解决语音LLM性能低于文本模型的问题。通过X-OPD框架，利用文本教师模型指导语音学生模型，提升其能力。**

- **链接: [https://arxiv.org/pdf/2603.24596](https://arxiv.org/pdf/2603.24596)**

> **作者:** Di Cao; Dongjie Fu; Hai Yu; Siqi Zheng; Xu Tan; Tao Jin
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While the shift from cascaded dialogue systems to end-to-end (E2E) speech Large Language Models (LLMs) improves latency and paralinguistic modeling, E2E models often exhibit a significant performance degradation compared to their text-based counterparts. The standard Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training methods fail to close this gap. To address this, we propose X-OPD, a novel Cross-Modal On-Policy Distillation framework designed to systematically align the capabilities of Speech LLMs to their text-based counterparts. X-OPD enables the Speech LLM to explore its own distribution via on-policy rollouts, where a text-based teacher model evaluates these trajectories and provides token-level feedback, effectively distilling teacher's capabilities into student's multi-modal representations. Extensive experiments across multiple benchmarks demonstrate that X-OPD significantly narrows the gap in complex tasks while preserving the model's inherent capabilities.
>
---
#### [replaced 013] Joint Optimization of Speaker and Spoof Detectors for Spoofing-Robust Automatic Speaker Verification
- **分类: eess.AS**

- **简介: 该论文属于语音安全任务，旨在提升对抗环境下的说话人验证鲁棒性。通过联合优化说话人和欺骗检测模块，改进后端融合策略，显著提升了系统性能。**

- **链接: [https://arxiv.org/pdf/2510.01818](https://arxiv.org/pdf/2510.01818)**

> **作者:** Oğuzhan Kurnaz; Jagabandhu Mishra; Tomi H. Kinnunen; Cemal Hanilçi
>
> **备注:** submitted to IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** Spoofing-robust speaker verification (SASV) combines the tasks of speaker and spoof detection to authenticate speakers under adversarial settings. Many SASV systems rely on fusion of speaker and spoof cues at embedding, score or decision levels, based on independently trained subsystems. In this study, we respect similar modularity of the two subsystems, by integrating their outputs using trainable back-end classifiers. In particular, we explore various approaches for directly optimizing the back-end for the recently-proposed SASV performance metric (a-DCF) as a training objective. Our experiments on the ASVspoof 5 dataset demonstrate two important findings: (i) nonlinear score fusion consistently improves a-DCF over linear fusion, and (ii) the combination of weighted cosine scoring for speaker detection with SSL-AASIST for spoof detection achieves state-of-the-art performance, reducing min a-DCF to 0.196 and SPF-EER to 7.6%. These contributions highlight the importance of modular design, calibrated integration, and task-aligned optimization for advancing robust and interpretable SASV systems.
>
---
