# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] TripleC Learning and Lightweight Speech Enhancement for Multi-Condition Target Speech Extraction
- **分类: eess.AS**

- **简介: 该论文研究多条件目标语音提取任务，旨在解决复杂真实场景下模型泛化能力不足的问题。提出TripleC学习策略与并行通用训练方法，结合轻量级增强模块，提升模型在未见条件下的鲁棒性与通用性。**

- **链接: [https://arxiv.org/pdf/2512.04945v1](https://arxiv.org/pdf/2512.04945v1)**

> **作者:** Ziling Huang
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** In our recent work, we proposed Lightweight Speech Enhancement Guided Target Speech Extraction (LGTSE) and demonstrated its effectiveness in multi-speaker-plus-noise scenarios. However, real-world applications often involve more diverse and complex conditions, such as one-speaker-plus-noise or two-speaker-without-noise. To address this challenge, we extend LGTSE with a Cross-Condition Consistency learning strategy, termed TripleC Learning. This strategy is first validated under multi-speaker-plus-noise condition and then evaluated for its generalization across diverse scenarios. Moreover, building upon the lightweight front-end denoiser in LGTSE, which can flexibly process both noisy and clean mixtures and shows strong generalization to unseen conditions, we integrate TripleC learning with a proposed parallel universal training scheme that organizes batches containing multiple scenarios for the same target speaker. By enforcing consistent extraction across different conditions, easier cases can assist harder ones, thereby fully exploiting diverse training data and fostering a robust universal model. Experimental results on the Libri2Mix three-condition tasks demonstrate that the proposed LGTSE with TripleC learning achieves superior performance over condition-specific models, highlighting its strong potential for universal deployment in real-world speech applications.
>
---
#### [new 002] M3-TTS: Multi-modal DiT Alignment & Mel-latent for Zero-shot High-fidelity Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文研究非自回归文本到语音合成任务，旨在解决现有方法在自然度和效率上的局限。提出M3-TTS，采用多模态DiT实现跨模态对齐与细节建模，结合mel-vae加速训练，实现高保真零样本语音合成。**

- **链接: [https://arxiv.org/pdf/2512.04720v1](https://arxiv.org/pdf/2512.04720v1)**

> **作者:** Xiaopeng Wang; Chunyu Qiang; Ruibo Fu; Zhengqi Wen; Xuefei Liu; Yukun Liu; Yuzhe Liang; Kang Yin; Yuankun Xie; Heng Xie; Chenxing Li; Chen Zhang; Changsheng Li
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Non-autoregressive (NAR) text-to-speech synthesis relies on length alignment between text sequences and audio representations, constraining naturalness and expressiveness. Existing methods depend on duration modeling or pseudo-alignment strategies that severely limit naturalness and computational efficiency. We propose M3-TTS, a concise and efficient NAR TTS paradigm based on multi-modal diffusion transformer (MM-DiT) architecture. M3-TTS employs joint diffusion transformer layers for cross-modal alignment, achieving stable monotonic alignment between variable-length text-speech sequences without pseudo-alignment requirements. Single diffusion transformer layers further enhance acoustic detail modeling. The framework integrates a mel-vae codec that provides 3* training acceleration. Experimental results on Seed-TTS and AISHELL-3 benchmarks demonstrate that M3-TTS achieves state-of-the-art NAR performance with the lowest word error rates (1.36\% English, 1.31\% Chinese) while maintaining competitive naturalness scores. Code and demos will be available at https://wwwwxp.github.io/M3-TTS.
>
---
#### [new 003] Large Speech Model Enabled Semantic Communication
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究语音语义通信任务，旨在解决现有系统泛化性差、抗干扰能力弱的问题。提出LargeSC系统，结合大模型与自适应控制，实现高效压缩、鲁棒传输与低延迟重建，支持极低带宽下的高质量语音通信。**

- **链接: [https://arxiv.org/pdf/2512.04711v1](https://arxiv.org/pdf/2512.04711v1)**

> **作者:** Yun Tian; Zhijin Qin; Guocheng Lv; Ye Jin; Kaibin Huang; Zhu Han
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Existing speech semantic communication systems mainly based on Joint Source-Channel Coding (JSCC) architectures have demonstrated impressive performance, but their effectiveness remains limited by model structures specifically designed for particular tasks and datasets. Recent advances indicate that generative large models pre-trained on massive datasets, can achieve outstanding performance arexhibit exceptional performance across diverse downstream tasks with minimal fine-tuning. To exploit the rich semantic knowledge embedded in large models and enable adaptive transmission over lossy channels, we propose a Large Speech Model enabled Semantic Communication (LargeSC) system. Simultaneously achieving adaptive compression and robust transmission over lossy channels remains challenging, requiring trade-offs among compression efficiency, speech quality, and latency. In this work, we employ the Mimi as a speech codec, converting speech into discrete tokens compatible with existing network architectures. We propose an adaptive controller module that enables adaptive transmission and in-band Unequal Error Protection (UEP), dynamically adjusting to both speech content and packet loss probability under bandwidth constraints. Additionally, we employ Low-Rank Adaptation (LoRA) to finetune the Moshi foundation model for generative recovery of lost speech tokens. Simulation results show that the proposed system supports bandwidths ranging from 550 bps to 2.06 kbps, outperforms conventional baselines in speech quality under high packet loss rates and achieves an end-to-end latency of approximately 460 ms, thereby demonstrating its potential for real-time deployment.
>
---
#### [new 004] Shared Multi-modal Embedding Space for Face-Voice Association
- **分类: cs.SD; cs.CV**

- **简介: 该论文针对多语言环境下人脸-语音关联任务，提出一种共享多模态嵌入空间方法。通过单模态特征提取与年龄性别辅助信息融合，结合自适应角边距损失，在FAME 2026挑战赛中取得第一。**

- **链接: [https://arxiv.org/pdf/2512.04814v1](https://arxiv.org/pdf/2512.04814v1)**

> **作者:** Christopher Simic; Korbinian Riedhammer; Tobias Bocklet
>
> **备注:** Ranked 1st in Fame 2026 Challenge, ICASSP
>
> **摘要:** The FAME 2026 challenge comprises two demanding tasks: training face-voice associations combined with a multilingual setting that includes testing on languages on which the model was not trained. Our approach consists of separate uni-modal processing pipelines with general face and voice feature extraction, complemented by additional age-gender feature extraction to support prediction. The resulting single-modal features are projected into a shared embedding space and trained with an Adaptive Angular Margin (AAM) loss. Our approach achieved first place in the FAME 2026 challenge, with an average Equal-Error Rate (EER) of 23.99%.
>
---
#### [new 005] YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing with Annotation-free Melody Guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究零样本歌声合成与编辑任务，旨在摆脱对音素级对齐和人工旋律标注的依赖。作者提出YingMusic-Singer框架，利用DiT架构与无标注旋律引导，结合教师模型指导的旋律提取、隐式对齐机制及强化学习优化发音与旋律质量，实现高质量、可扩展的歌声合成。**

- **链接: [https://arxiv.org/pdf/2512.04779v1](https://arxiv.org/pdf/2512.04779v1)**

> **作者:** Junjie Zheng; Chunbo Hao; Guobin Ma; Xiaoyu Zhang; Gongyu Chen; Chaofan Ding; Zihao Chen; Lei Xie
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Singing Voice Synthesis (SVS) remains constrained in practical deployment due to its strong dependence on accurate phoneme-level alignment and manually annotated melody contours, requirements that are resource-intensive and hinder scalability. To overcome these limitations, we propose a melody-driven SVS framework capable of synthesizing arbitrary lyrics following any reference melody, without relying on phoneme-level alignment. Our method builds on a Diffusion Transformer (DiT) architecture, enhanced with a dedicated melody extraction module that derives melody representations directly from reference audio. To ensure robust melody encoding, we employ a teacher model to guide the optimization of the melody extractor, alongside an implicit alignment mechanism that enforces similarity distribution constraints for improved melodic stability and coherence. Additionally, we refine duration modeling using weakly annotated song data and introduce a Flow-GRPO reinforcement learning strategy with a multi-objective reward function to jointly enhance pronunciation clarity and melodic fidelity. Experiments show that our model achieves superior performance over existing approaches in both objective measures and subjective listening tests, especially in zero-shot and lyric adaptation settings, while maintaining high audio quality without manual annotation. This work offers a practical and scalable solution for advancing data-efficient singing voice synthesis. To support reproducibility, we release our inference code and model checkpoints.
>
---
#### [new 006] Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and Frame-Level Attention
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究语音情感识别任务，旨在解决情感复杂性和标注数据不足的问题。提出多损失学习框架，结合能量自适应mixup和帧级注意力模块，提升特征区分性与模型性能，在四个基准数据集上达到最优效果。**

- **链接: [https://arxiv.org/pdf/2512.04551v1](https://arxiv.org/pdf/2512.04551v1)**

> **作者:** Cong Wang; Yizhong Geng; Yuhua Wen; Qifei Li; Yingming Gao; Ruimin Wang; Chunfeng Wang; Hao Li; Ya Li; Wei Chen
>
> **备注:** Submitted to ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Speech emotion recognition (SER) is an important technology in human-computer interaction. However, achieving high performance is challenging due to emotional complexity and scarce annotated data. To tackle these challenges, we propose a multi-loss learning (MLL) framework integrating an energy-adaptive mixup (EAM) method and a frame-level attention module (FLAM). The EAM method leverages SNR-based augmentation to generate diverse speech samples capturing subtle emotional variations. FLAM enhances frame-level feature extraction for multi-frame emotional cues. Our MLL strategy combines Kullback-Leibler divergence, focal, center, and supervised contrastive loss to optimize learning, address class imbalance, and improve feature separability. We evaluate our method on four widely used SER datasets: IEMOCAP, MSP-IMPROV, RAVDESS, and SAVEE. The results demonstrate our method achieves state-of-the-art performance, suggesting its effectiveness and robustness.
>
---
#### [new 007] HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages
- **分类: eess.AS**

- **简介: 该论文研究自动发音评估（APA）任务，旨在解决自由说话场景下发音质量评估难题。提出HiPPO模型，从语音中多层次评估二语发音水平，并引入对比序数正则化和课程学习提升效果，在Speechocean762数据集上验证了方法优越性。**

- **链接: [https://arxiv.org/pdf/2512.04964v1](https://arxiv.org/pdf/2512.04964v1)**

> **作者:** Bi-Cheng Yan; Hsin-Wei Wang; Fu-An Chao; Tien-Hong Lo; Yung-Chang Hsu; Berlin Chen
>
> **备注:** Accepted and to appear in AACL-IJCNLP2025
>
> **摘要:** Automatic pronunciation assessment (APA) seeks to quantify a second language (L2) learner's pronunciation proficiency in a target language by offering timely and fine-grained diagnostic feedback. Most existing efforts on APA have predominantly concentrated on highly constrained reading-aloud tasks (where learners are prompted to read a reference text aloud); however, assessing pronunciation quality in unscripted speech (or free-speaking scenarios) remains relatively underexplored. In light of this, we first propose HiPPO, a hierarchical pronunciation assessment model tailored for spoken languages, which evaluates an L2 learner's oral proficiency at multiple linguistic levels based solely on the speech uttered by the learner. To improve the overall accuracy of assessment, a contrastive ordinal regularizer and a curriculum learning strategy are introduced for model training. The former aims to generate score-discriminative features by exploiting the ordinal nature of regression targets, while the latter gradually ramps up the training complexity to facilitate the assessment task that takes unscripted speech as input. Experiments conducted on the Speechocean762 benchmark dataset validates the feasibility and superiority of our method in relation to several cutting-edge baselines.
>
---
#### [new 008] Contract-Driven QoE Auditing for Speech and Singing Services: From MOS Regression to Service Graphs
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究语音与歌唱服务的体验质量（QoE）评估，提出合同驱动的审计框架，用可解释的体验合同替代传统MOS标量，提升评估稳定性与可比性。基于真实数据验证其在跨视图一致性与学习效率上的优势。**

- **链接: [https://arxiv.org/pdf/2512.04827v1](https://arxiv.org/pdf/2512.04827v1)**

> **作者:** Wenzhang Du
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Subjective mean opinion scores (MOS) remain the de-facto target for non-intrusive speech and singing quality assessment. However, MOS is a scalar that collapses heterogeneous user expectations, ignores service-level objectives, and is difficult to compare across deployment graphs. We propose a contract-driven QoE auditing framework: each service graph G is evaluated under a set of human-interpretable experience contracts C, yielding a contract-level satisfaction vector Q(G, C). We show that (i) classical MOS regression is a special case with a degenerate contract set, (ii) contract-driven quality is more stable than MOS under graph view transformations (e.g., pooling by system vs. by system type), and (iii) the effective sample complexity of learning contracts is governed by contract semantics rather than merely the dimensionality of C. We instantiate the framework on URGENT2024 MOS (6.9k speech utterances with raw rating vectors) and SingMOS v1 (7,981 singing clips; 80 systems). On URGENT, we train a contract-aware neural auditor on self-supervised WavLM embeddings; on SingMOS, we perform contract-driven graph auditing using released rating vectors and metadata without decoding audio. Empirically, our auditor matches strong MOS predictors in MOS accuracy while providing calibrated contract probabilities; on SingMOS, Q(G, C) exhibits substantially smaller cross-view drift than raw MOS and graph-only baselines; on URGENT, difficulty curves reveal that mis-specified "simple" contracts can be harder to learn than richer but better aligned contract sets.
>
---
#### [new 009] YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究零样本歌唱声音转换（SVC），旨在解决真实歌曲中和声干扰、音高误差等问题。提出YingMusic-SVC框架，引入歌唱专用归纳偏置与Flow-GRPO强化学习，提升音色保真度与自然性，增强实际应用鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.04793v1](https://arxiv.org/pdf/2512.04793v1)**

> **作者:** Gongyu Chen; Xiaoyu Zhang; Zhenqiang Weng; Junjie Zheng; Da Shen; Chaofan Ding; Wei-Qiang Zhang; Zihao Chen
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Singing voice conversion (SVC) aims to render the target singer's timbre while preserving melody and lyrics. However, existing zero-shot SVC systems remain fragile in real songs due to harmony interference, F0 errors, and the lack of inductive biases for singing. We propose YingMusic-SVC, a robust zero-shot framework that unifies continuous pre-training, robust supervised fine-tuning, and Flow-GRPO reinforcement learning. Our model introduces a singing-trained RVC timbre shifter for timbre-content disentanglement, an F0-aware timbre adaptor for dynamic vocal expression, and an energy-balanced rectified flow matching loss to enhance high-frequency fidelity. Experiments on a graded multi-track benchmark show that YingMusic-SVC achieves consistent improvements over strong open-source baselines in timbre similarity, intelligibility, and perceptual naturalness, especially under accompanied and harmony-contaminated conditions, demonstrating its effectiveness for real-world SVC deployment.
>
---
#### [new 010] Towards predicting binaural audio quality in listeners with normal and impaired hearing
- **分类: eess.AS**

- **简介: 该论文旨在预测正常与听力受损者对双耳音频的质量感知。针对现有模型未考虑听力损失影响的问题，作者扩展了eMoBi-Q模型，引入基于生理机制的非线性听觉滤波器组，以整合响度感知，提升对听力障碍者的音频质量预测能力。**

- **链接: [https://arxiv.org/pdf/2512.04792v1](https://arxiv.org/pdf/2512.04792v1)**

> **作者:** Thomas Biberger; Stephan D. Ewert
>
> **备注:** accepted for publication in Forum Acusticum
>
> **摘要:** Eurich et al. (2024) recently introduced the computationally efficient monaural and binaural audio quality model (eMoBi-Q). This model integrates both monaural and binaural auditory features and has been validated across six audio datasets encompassing quality ratings for music and speech, processed via algorithms commonly employed in modern hearing devices (e.g., acoustic transparency, feedback cancellation, and binaural beamforming) or presented via loudspeakers. In the current study, we expand eMoBi-Q to account for perceptual effects of sensorineural hearing loss (HL) on audio quality. For this, the model was extended by a nonlinear auditory filterbank. Given that altered loudness perception is a prevalent issue among listeners with hearing impairment, our goal is to incorporate loudness as a sub-dimension for predicting audio quality in both normal-hearing and hearing-impaired populations. While predicting loudness itself is important in the context of loudness-based hearing aid fitting, loudness as audio quality sub-measure may be helpful for the selection of reliable auditory features in hearing impaired listeners. The parameters of the filterbank and subsequent processing stages were informed by the physiologically-based (binaural) loudness model proposed by Pieper et al. (2018). This study presents and discusses the initial implementation of the extended binaural quality model.
>
---
#### [new 011] RRPO: Robust Reward Policy Optimization for LLM-based Emotional TTS
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究基于大语言模型的情感语音合成，针对可微强化学习中奖励模型易被策略模型利用导致音质下降的问题，提出RRPO框架，通过混合正则化构建更鲁棒的奖励模型，有效抑制奖励欺骗，提升情感表现力与自然度。**

- **链接: [https://arxiv.org/pdf/2512.04552v1](https://arxiv.org/pdf/2512.04552v1)**

> **作者:** Cong Wang; Changfeng Gao; Yang Xiang; Zhihao Du; Keyu An; Han Zhao; Qian Chen; Xiangang Li; Yingming Gao; Ya Li
>
> **备注:** Submitted to ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Differentiable reinforcement learning (RL) frameworks like DiffRO offer a powerful approach for controllable text-to-speech (TTS), but are vulnerable to reward hacking, particularly for nuanced tasks like emotion control. The policy model can exploit a vanilla Reward Model (RM) by generating acoustic artifacts to achieve spurious rewards, but at the cost of degrading perceptual quality. To address this, we propose Robust Reward Policy Optimization (RRPO), a novel framework that employs a hybrid regularization scheme. This scheme develops a robust RM whose reward signal is more reliably aligned with human perception, compelling the policy to abandon detrimental shortcuts and instead learn the complex features of genuine emotions. Our ablation study confirms the enhanced robustness of our RM, as evidenced by its strong cross-lingual generalization. The subjective evaluation demonstrates that this robust RM effectively mitigates reward hacking, leading to significant improvements in both emotional expressiveness and naturalness over all baselines. Demo page: https://lrwinr.github.io/RRPO-CosyVoice.
>
---
#### [new 012] Standard audiogram classification from loudness scaling data using unsupervised, supervised, and explainable machine learning techniques
- **分类: cs.SD; physics.med-ph**

- **简介: 该论文研究利用无需校准的响度感知数据，通过机器学习分类方法预测标准Bisgaard听力图类型，旨在解决远程听力评估中的校准难题。工作涵盖无监督、有监督和可解释模型的比较，验证了其在资源受限场景的应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.04616v1](https://arxiv.org/pdf/2512.04616v1)**

> **作者:** Chen Xu; Lena Schell-Majoor; Birger Kollmeier
>
> **摘要:** To address the calibration and procedural challenges inherent in remote audiogram assessment for rehabilitative audiology, this study investigated whether calibration-independent adaptive categorical loudness scaling (ACALOS) data can be used to approximate individual audiograms by classifying listeners into standard Bisgaard audiogram types using machine learning. Three classes of machine learning approaches - unsupervised, supervised, and explainable - were evaluated. Principal component analysis (PCA) was performed to extract the first two principal components, which together explained more than 50 percent of the variance. Seven supervised multi-class classifiers were trained and compared, alongside unsupervised and explainable methods. Model development and evaluation used a large auditory reference database containing ACALOS data (N = 847). The PCA factor map showed substantial overlap between listeners, indicating that cleanly separating participants into six Bisgaard classes based solely on their loudness patterns is challenging. Nevertheless, the models demonstrated reasonable classification performance, with logistic regression achieving the highest accuracy among supervised approaches. These findings demonstrate that machine learning models can predict standard Bisgaard audiogram types, within certain limits, from calibration-independent loudness perception data, supporting potential applications in remote or resource-limited settings without requiring a traditional audiogram.
>
---
#### [new 013] Language Models as Semantic Teachers: Post-Training Alignment for Medical Audio Understanding
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦医疗音频理解任务，旨在解决预训练音频模型缺乏临床语义理解的问题。作者提出AcuLa框架，通过与医学语言模型对齐，赋予音频编码器临床语义理解能力，显著提升多类心肺诊断任务性能。**

- **链接: [https://arxiv.org/pdf/2512.04847v1](https://arxiv.org/pdf/2512.04847v1)**

> **作者:** Tsai-Ning Wang; Lin-Lin Chen; Neil Zeghidour; Aaqib Saeed
>
> **摘要:** Pre-trained audio models excel at detecting acoustic patterns in auscultation sounds but often fail to grasp their clinical significance, limiting their use and performance in diagnostic tasks. To bridge this gap, we introduce AcuLa (Audio-Clinical Understanding via Language Alignment), a lightweight post-training framework that instills semantic understanding into any audio encoder by aligning it with a medical language model, which acts as a "semantic teacher." To enable alignment at scale, we construct a large-scale dataset by leveraging off-the-shelf large language models to translate the rich, structured metadata accompanying existing audio recordings into coherent clinical reports. Our alignment strategy combines a representation-level contrastive objective with a self-supervised modeling, ensuring that the model learns clinical semantics while preserving fine-grained temporal cues. AcuLa achieves state-of-the-art results across 18 diverse cardio-respiratory tasks from 10 different datasets, improving the mean AUROC on classification benchmarks from 0.68 to 0.79 and, on the most challenging COVID-19 cough detection task, boosting the AUROC from 0.55 to 0.89. Our work demonstrates that this audio-language alignment transforms purely acoustic models into clinically-aware diagnostic tools, establishing a novel paradigm for enhancing physiological understanding in audio-based health monitoring.
>
---
## 更新

#### [replaced 001] Head, posture, and full-body gestures in dyadic conversations
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究双人对话中头部、姿势和全身动作在噪声环境下的变化。旨在探究视觉线索如何辅助交流，通过虚拟实验分析不同噪声水平下说话与倾听时的身体动作频率及手-语同步性，揭示多模态适应机制。**

- **链接: [https://arxiv.org/pdf/2512.03636v2](https://arxiv.org/pdf/2512.03636v2)**

> **作者:** Ľuboš Hládek; Bernhard U. Seeber
>
> **备注:** 7 figures, 10 tables, 29 pages
>
> **摘要:** When face-to-face communication becomes effortful due to background noise and interfering talkers, the role of visual cues becomes increasingly important for communication success. While previous research has selectively investigated head or hand movements, here we explore the combination of movements of head, hand and the whole body in acoustically adverse conditions. We hypothesize that with increasing background noise level, the frequency of typical conversational movements of hand, head, trunk, and legs increases to support the speakers role while the listeners support their role by increased use of confirmative head gestures and head and trunk movements to increase the signal-to-noise ratio. We conducted a dyadic conversation experiment in which (n=8) normal hearing participants stood freely in an audiovisual virtual environment. The conversational movements were described by a newly developed labeling system for typical conversational movements, and the frequency of individual types was analyzed. Increased levels of background noise led to increased hand-gesture complexity and modulation of head movements without a clear pattern. People leaned forward slightly more and used less head movements during listening than during speaking. Additional analysis of hand-speech synchrony with hypothesized loss of synchrony due to the background noise showed a modest decrease of synchrony in terms of increased standard deviation at moderate sound levels. The results support previous findings in terms of the gesturing frequency, and we found a limited support for the changes in speech-gesture synchrony. The work reveals communication patterns of the whole body and exemplifies interactive communication in context of multimodal adaptation to communication needs.
>
---
#### [replaced 002] A Lightweight Architecture for Multi-instrument Transcription with Practical Optimizations
- **分类: cs.SD; cs.IR**

- **简介: 该论文研究多乐器自动记谱任务，旨在解决现有模型泛化性差、计算量大、依赖预设乐器数等问题。作者提出轻量模型，结合音色无关主干、音色编码器与深层聚类，实现高效、动态的乐器分离与记谱。**

- **链接: [https://arxiv.org/pdf/2509.12712v2](https://arxiv.org/pdf/2509.12712v2)**

> **作者:** Ruigang Li; Yongxu Zhu
>
> **摘要:** Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments, rigid source-count constraints, and high computational demands that hinder deployment on low-resource devices. We address these limitations with a lightweight model that extends a timbre-agnostic transcription backbone with a dedicated timbre encoder and performs deep clustering at the note level, enabling joint transcription and dynamic separation of arbitrary instruments. Practical optimizations including spectral normalization, dilated convolutions, and contrastive clustering further improve efficiency and robustness. Despite its small size and fast inference, the model achieves competitive performance with heavier baselines in transcription accuracy and separation quality, and shows promising generalization ability, making it highly suitable for real-world deployment in practical and resource-constrained settings.
>
---
#### [replaced 003] Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning
- **分类: cs.AI; cs.SD**

- **简介: 该论文提出Omni-AutoThink框架，解决现有Omni模型在多模态任务中推理行为僵化的问题。通过自适应监督微调和强化学习，使模型能根据任务难度动态调整推理深度，提升多模态自适应推理能力，并构建了涵盖多种模态的评测基准。**

- **链接: [https://arxiv.org/pdf/2512.03783v2](https://arxiv.org/pdf/2512.03783v2)**

> **作者:** Dongchao Yang; Songxiang Liu; Disong Wang; Yuanyuan Wang; Guanglu Wan; Helen Meng
>
> **摘要:** Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors, either overthinking simple problems or failing to reason when necessary. To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model's reasoning depth according to task difficulty. Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback. We further construct a comprehensive adaptive reasoning benchmark that spans text-only, text-audio, text-visual, and text-audio-visual modalities, providing both training and evaluation splits for multimodal reasoning assessment. Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines. All benchmark data and code will be publicly released.
>
---
#### [replaced 004] MelTok: 2D Tokenization for Single-Codebook Audio Compression
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究单码本音频压缩任务，旨在解决现有1D tokenizer在保持细粒度声学细节上的局限。提出MelTok，一种二维tokenizer，将音频转为紧凑的mel-spectrogram表示，并设计对应vocoder实现高质量重建，提升单码本压缩性能。**

- **链接: [https://arxiv.org/pdf/2510.01903v3](https://arxiv.org/pdf/2510.01903v3)**

> **作者:** Jingyi Li; Zhiyuan Zhao; Zhisheng Zhang; Yunfei Liu; Lijian Lin; Ye Zhu; Jiahao Wu; Qiuqiang Kong; Yu Li
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Large Audio Language Models (LALMs) have emerged with strong performance across diverse audio understanding tasks and can be further enhanced by neural audio codecs. Transitioning from multi-layer residual vector quantizers to a single-layer quantizer has been shown to facilitate more efficient downstream language models decoding. However, the ability of a single codebook to capture fine-grained acoustic details remains limited, as the frequency-variant nature of 1D tokenizers leads to redundancy. To address this issue, we propose MelTok, a two-dimensional (2D) tokenizer that effectively compresses acoustic details of 44.1 KHz audio into a single codebook. The tokenizer encodes audio into a more compact representation than one-dimensional tokenizers. Furthermore, to recover audio from mel-spectrogram tokens, we propose a token-based vocoder. Both objective and subjective evaluations demonstrate that MelTok achieves quality comparable to multi-codebook codecs and outperforms existing state-of-the-art neural codecs with a single codebook on high-fidelity audio reconstruction. By preserving acoustic details, MelTok offers a strong representation for downstream understanding tasks.
>
---
