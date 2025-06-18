# 音频 cs.SD;  eess.SP

- **最新发布 22 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Refining music sample identification with a self-supervised graph neural network
- **分类: cs.SD; cs.AI; cs.IR; H.5.5; I.2.6**

- **简介: 该论文属于音乐样本识别任务，旨在解决音频检索中因音乐处理导致的样本识别困难问题。提出一种轻量级图神经网络模型，提升识别准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.14684v1](http://arxiv.org/pdf/2506.14684v1)**

> **作者:** Aditya Bhattacharjee; Ivan Meresman Higgs; Mark Sandler; Emmanouil Benetos
>
> **备注:** Accepted at International Conference for Music Information Retrieval (ISMIR) 2025
>
> **摘要:** Automatic sample identification (ASID), the detection and identification of portions of audio recordings that have been reused in new musical works, is an essential but challenging task in the field of audio query-based retrieval. While a related task, audio fingerprinting, has made significant progress in accurately retrieving musical content under "real world" (noisy, reverberant) conditions, ASID systems struggle to identify samples that have undergone musical modifications. Thus, a system robust to common music production transformations such as time-stretching, pitch-shifting, effects processing, and underlying or overlaying music is an important open challenge. In this work, we propose a lightweight and scalable encoding architecture employing a Graph Neural Network within a contrastive learning framework. Our model uses only 9% of the trainable parameters compared to the current state-of-the-art system while achieving comparable performance, reaching a mean average precision (mAP) of 44.2%. To enhance retrieval quality, we introduce a two-stage approach consisting of an initial coarse similarity search for candidate selection, followed by a cross-attention classifier that rejects irrelevant matches and refines the ranking of retrieved candidates - an essential capability absent in prior models. In addition, because queries in real-world applications are often short in duration, we benchmark our system for short queries using new fine-grained annotations for the Sample100 dataset, which we publish as part of this work.
>
---
#### [new 002] Evolving music theory for emerging musical languages
- **分类: cs.SD; 00A65; J.5**

- **简介: 该论文属于音乐理论研究，探讨电子音乐中音高概念的演变。它提出音高是感知构造而非客观属性，分析了音高多义性和可变性，挑战传统理论。**

- **链接: [http://arxiv.org/pdf/2506.14504v1](http://arxiv.org/pdf/2506.14504v1)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2025, Innovation in Music Conference. 20-22 June, 2025, Bath Spa University, Bath, UK
>
> **摘要:** This chapter reconsiders the concept of pitch in contemporary popular music (CPM), particularly in electronic contexts where traditional assumptions may fail. Drawing on phenomenological and inductive methods, it argues that pitch is not an ontologically objective property but a perceptual construct shaped by listeners and conditions. Analyses of quasi-harmonic tones reveal that a single tone can convey multiple pitches, giving rise to tonal fission. The perception of pitch may also be multistable, varying for the same listener over time. In this framework, the tuning system may emerge from a tone's internal structure. A parallel with the coastline paradox supports a model of pitch grounded in perceptual variability, challenging inherited theoretical norms.
>
---
#### [new 003] Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于非侵入式物体分类任务，旨在通过声波散射实现头发类型与湿度的识别，利用AI进行声音分类并验证多种方法。**

- **链接: [http://arxiv.org/pdf/2506.14148v1](http://arxiv.org/pdf/2506.14148v1)**

> **作者:** Long-Vu Hoang; Tuan Nguyen; Tran Huy Dat
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries.
>
---
#### [new 004] Adaptive Accompaniment with ReaLchords
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决在线即兴伴奏问题。提出ReaLchords模型，通过预训练和强化学习实现与用户旋律的协同创作。**

- **链接: [http://arxiv.org/pdf/2506.14723v1](http://arxiv.org/pdf/2506.14723v1)**

> **作者:** Yusong Wu; Tim Cooijmans; Kyle Kastner; Adam Roberts; Ian Simon; Alexander Scarlatos; Chris Donahue; Cassie Tarakajian; Shayegan Omidshafiei; Aaron Courville; Pablo Samuel Castro; Natasha Jaques; Cheng-Zhi Anna Huang
>
> **备注:** Accepted by ICML 2024
>
> **摘要:** Jamming requires coordination, anticipation, and collaborative creativity between musicians. Current generative models of music produce expressive output but are not able to generate in an \emph{online} manner, meaning simultaneously with other musicians (human or otherwise). We propose ReaLchords, an online generative model for improvising chord accompaniment to user melody. We start with an online model pretrained by maximum likelihood, and use reinforcement learning to finetune the model for online use. The finetuning objective leverages both a novel reward model that provides feedback on both harmonic and temporal coherency between melody and chord, and a divergence term that implements a novel type of distillation from a teacher model that can see the future melody. Through quantitative experiments and listening tests, we demonstrate that the resulting model adapts well to unfamiliar input and produce fitting accompaniment. ReaLchords opens the door to live jamming, as well as simultaneous co-creation in other modalities.
>
---
#### [new 005] Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于音乐信息检索任务，解决吉他MIDI转谱表的问题。提出Fretting-Transformer模型，处理弦位歧义与可演奏性，提升转录准确性。**

- **链接: [http://arxiv.org/pdf/2506.14223v1](http://arxiv.org/pdf/2506.14223v1)**

> **作者:** Anna Hamberger; Sebastian Murgul; Jochen Schmidt; Michael Heizmann
>
> **备注:** Accepted to the 50th International Computer Music Conference (ICMC), 2025
>
> **摘要:** Music transcription plays a pivotal role in Music Information Retrieval (MIR), particularly for stringed instruments like the guitar, where symbolic music notations such as MIDI lack crucial playability information. This contribution introduces the Fretting-Transformer, an encoderdecoder model that utilizes a T5 transformer architecture to automate the transcription of MIDI sequences into guitar tablature. By framing the task as a symbolic translation problem, the model addresses key challenges, including string-fret ambiguity and physical playability. The proposed system leverages diverse datasets, including DadaGP, GuitarToday, and Leduc, with novel data pre-processing and tokenization strategies. We have developed metrics for tablature accuracy and playability to quantitatively evaluate the performance. The experimental results demonstrate that the Fretting-Transformer surpasses baseline methods like A* and commercial applications like Guitar Pro. The integration of context-sensitive processing and tuning/capo conditioning further enhances the model's performance, laying a robust foundation for future developments in automated guitar transcription.
>
---
#### [new 006] Unifying Streaming and Non-streaming Zipformer-based ASR
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在统一流式与非流式模型。通过引入动态右文信息，提升模型性能并灵活控制延迟与准确率的平衡。**

- **链接: [http://arxiv.org/pdf/2506.14434v1](http://arxiv.org/pdf/2506.14434v1)**

> **作者:** Bidisha Sharma; Karthik Pandia Durai; Shankar Venkatesan; Jeena J Prakash; Shashi Kumar; Malolan Chetlur; Andreas Stolcke
>
> **备注:** Accepted in ACL2025 Industry track
>
> **摘要:** There has been increasing interest in unifying streaming and non-streaming automatic speech recognition (ASR) models to reduce development, training, and deployment costs. We present a unified framework that trains a single end-to-end ASR model for both streaming and non-streaming applications, leveraging future context information. We propose to use dynamic right-context through the chunked attention masking in the training of zipformer-based ASR models. We demonstrate that using right-context is more effective in zipformer models compared to other conformer models due to its multi-scale nature. We analyze the effect of varying the number of right-context frames on accuracy and latency of the streaming ASR models. We use Librispeech and large in-house conversational datasets to train different versions of streaming and non-streaming models and evaluate them in a production grade server-client setup across diverse testsets of different domains. The proposed strategy reduces word error by relative 7.9\% with a small degradation in user-perceived latency. By adding more right-context frames, we are able to achieve streaming performance close to that of non-streaming models. Our approach also allows flexible control of the latency-accuracy tradeoff according to customers requirements.
>
---
#### [new 007] Investigation of Zero-shot Text-to-Speech Models for Enhancing Short-Utterance Speaker Verification
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决短语句说话人验证准确率低的问题。通过引入零样本文本转语音系统进行数据增强，提升验证效果。**

- **链接: [http://arxiv.org/pdf/2506.14226v1](http://arxiv.org/pdf/2506.14226v1)**

> **作者:** Yiyang Zhao; Shuai Wang; Guangzhi Sun; Zehua Chen; Chao Zhang; Mingxing Xu; Thomas Fang Zheng
>
> **摘要:** Short-utterance speaker verification presents significant challenges due to the limited information in brief speech segments, which can undermine accuracy and reliability. Recently, zero-shot text-to-speech (ZS-TTS) systems have made considerable progress in preserving speaker identity. In this study, we explore, for the first time, the use of ZS-TTS systems for test-time data augmentation for speaker verification. We evaluate three state-of-the-art pre-trained ZS-TTS systems, NatureSpeech 3, CosyVoice, and MaskGCT, on the VoxCeleb 1 dataset. Our experimental results show that combining real and synthetic speech samples leads to 10%-16% relative equal error rate (EER) reductions across all durations, with particularly notable improvements for short utterances, all without retraining any existing systems. However, our analysis reveals that longer synthetic speech does not yield the same benefits as longer real speech in reducing EERs. These findings highlight the potential and challenges of using ZS-TTS for test-time speaker verification, offering insights for future research.
>
---
#### [new 008] Exploring Speaker Diarization with Mixture of Experts
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音处理中的说话人日志任务，旨在提升复杂环境下的说话人识别性能。通过引入记忆感知嵌入和混合专家模型，增强了系统的鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.14750v1](http://arxiv.org/pdf/2506.14750v1)**

> **作者:** Gaobin Yang; Maokui He; Shutong Niu; Ruoyu Wang; Hang Chen; Jun Du
>
> **摘要:** In this paper, we propose a novel neural speaker diarization system using memory-aware multi-speaker embedding with sequence-to-sequence architecture (NSD-MS2S), which integrates a memory-aware multi-speaker embedding module with a sequence-to-sequence architecture. The system leverages a memory module to enhance speaker embeddings and employs a Seq2Seq framework to efficiently map acoustic features to speaker labels. Additionally, we explore the application of mixture of experts in speaker diarization, and introduce a Shared and Soft Mixture of Experts (SS-MoE) module to further mitigate model bias and enhance performance. Incorporating SS-MoE leads to the extended model NSD-MS2S-SSMoE. Experiments on multiple complex acoustic datasets, including CHiME-6, DiPCo, Mixer 6 and DIHARD-III evaluation sets, demonstrate meaningful improvements in robustness and generalization. The proposed methods achieve state-of-the-art results, showcasing their effectiveness in challenging real-world scenarios.
>
---
#### [new 009] An Open Research Dataset of the 1932 Cairo Congress of Arab Music
- **分类: cs.SD; cs.DL; eess.AS**

- **简介: 该论文介绍了一个开放数据集ORD-CC32，用于研究1932年开罗阿拉伯音乐大会的录音，旨在支持计算民族音乐学和音乐信息检索等任务。**

- **链接: [http://arxiv.org/pdf/2506.14503v1](http://arxiv.org/pdf/2506.14503v1)**

> **作者:** Baris Bozkurt
>
> **备注:** 14 pages, 4 figures, 4 tables
>
> **摘要:** This paper introduces ORD-CC32 , an open research dataset derived from the 1932 Cairo Congress of Arab Music recordings, a historically significant collection representing diverse Arab musical traditions. The dataset includes structured metadata, melodic and rhythmic mode tags (maqam and iqa), manually labeled tonic information, and acoustic features extracted using state-of-the-art pitch detection methods. These resources support computational studies of tuning, temperament, and regional variations in Arab music. A case study using pitch histograms demonstrates the potential for data-driven analysis of microtonal differences across regions. By making this dataset openly available, we aim to enable interdisciplinary research in computational ethnomusicology, music information retrieval (MIR), cultural studies, and digital heritage preservation. ORD-CC32 is shared on Zenodo with tools for feature extraction and metadata retrieval.
>
---
#### [new 010] The Perception of Phase Intercept Distortion and its Application in Data Augmentation
- **分类: eess.SP; cs.LG; eess.AS**

- **简介: 该论文研究相位截距失真对人类感知的影响及其在数据增强中的应用，旨在提升音频机器学习性能。**

- **链接: [http://arxiv.org/pdf/2506.14571v1](http://arxiv.org/pdf/2506.14571v1)**

> **作者:** Venkatakrishnan Vaidyanathapuram Krishnan; Nathaniel Condit-Schultz
>
> **备注:** Submitted to the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** Phase distortion refers to the alteration of the phase relationships between frequencies in a signal, which can be perceptible. In this paper, we discuss a special case of phase distortion known as phase-intercept distortion, which is created by a frequency-independent phase shift. We hypothesize that, though this form of distortion changes a signal's waveform significantly, the distortion is imperceptible. Human-subject experiment results are reported which are consistent with this hypothesis. Furthermore, we discuss how the imperceptibility of phase-intercept distortion can be useful for machine learning, specifically for data augmentation. We conducted multiple experiments using phase-intercept distortion as a novel approach to data augmentation, and obtained improved results for audio machine learning tasks.
>
---
#### [new 011] Making deep neural networks work for medical audio: representation, compression and domain adaptation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于医疗音频分析任务，旨在提升深度学习在医学音频中的应用。解决数据少、模型复杂和跨域适应等问题，提出迁移学习、模型压缩和领域自适应方法，并发布公开数据集。**

- **链接: [http://arxiv.org/pdf/2506.13970v1](http://arxiv.org/pdf/2506.13970v1)**

> **作者:** Charles C Onu
>
> **备注:** PhD Thesis
>
> **摘要:** This thesis addresses the technical challenges of applying machine learning to understand and interpret medical audio signals. The sounds of our lungs, heart, and voice convey vital information about our health. Yet, in contemporary medicine, these sounds are primarily analyzed through auditory interpretation by experts using devices like stethoscopes. Automated analysis offers the potential to standardize the processing of medical sounds, enable screening in low-resource settings where physicians are scarce, and detect subtle patterns that may elude human perception, thereby facilitating early diagnosis and treatment. Focusing on the analysis of infant cry sounds to predict medical conditions, this thesis contributes on four key fronts. First, in low-data settings, we demonstrate that large databases of adult speech can be harnessed through neural transfer learning to develop more accurate and robust models for infant cry analysis. Second, in cost-effective modeling, we introduce an end-to-end model compression approach for recurrent networks using tensor decomposition. Our method requires no post-hoc processing, achieves compression rates of several hundred-fold, and delivers accurate, portable models suitable for resource-constrained devices. Third, we propose novel domain adaptation techniques tailored for audio models and adapt existing methods from computer vision. These approaches address dataset bias and enhance generalization across domains while maintaining strong performance on the original data. Finally, to advance research in this domain, we release a unique, open-source dataset of infant cry sounds, developed in collaboration with clinicians worldwide. This work lays the foundation for recognizing the infant cry as a vital sign and highlights the transformative potential of AI-driven audio monitoring in shaping the future of accessible and affordable healthcare.
>
---
#### [new 012] Set theoretic solution for the tuning problem
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐调音任务，旨在解决如何量化音乐协和性的难题。通过集合论定义了和谐度和亲和力指标，构建动态调音系统。**

- **链接: [http://arxiv.org/pdf/2506.13969v1](http://arxiv.org/pdf/2506.13969v1)**

> **作者:** Vsevolod Vladimirovich Deriushkin
>
> **摘要:** In this paper I want to suggest a new solution to the problem of musical tuning. On one hand, I see it as a generalization of Just Intonation (JI) to inharmonic timbers, on another, as a unification of spectral interference and harmonicity contributions to consonance within a single framework. The main achievement of the work is the ability to mathematically quantify the phenomenon of musical consonance using set theory. That quantification is done by defining two measures of consonance: affinity and harmonicity. These measures naturally generate sets of intervals that can be used as dynamic tuning systems. The paper is aimed at a broad audience of people who may not be skilled in music and tuning theory or mathematics. Thus, I attempt to give as much details and explanations as I can, while keeping the number of pages as low as possible.
>
---
#### [new 013] Manipulated Regions Localization For Partially Deepfake Audio: A Survey
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决部分深度伪造音频中篡改区域定位问题，综述现有方法、挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2506.14396v1](http://arxiv.org/pdf/2506.14396v1)**

> **作者:** Jiayi He; Jiangyan Yi; Jianhua Tao; Siding Zeng; Hao Gu
>
> **摘要:** With the development of audio deepfake techniques, attacks with partially deepfake audio are beginning to rise. Compared to fully deepfake, it is much harder to be identified by the detector due to the partially cryptic manipulation, resulting in higher security risks. Although some studies have been launched, there is no comprehensive review to systematically introduce the current situations and development trends for addressing this issue. Thus, in this survey, we are the first to outline a systematic introduction for partially deepfake audio manipulated region localization tasks, including the fundamentals, branches of existing methods, current limitations and potential trends, providing a revealing insight into this scope.
>
---
#### [new 014] SLEEPING-DISCO 9M: A large-scale pre-training dataset for generative music modeling
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出Sleeping-DISCO 9M数据集，用于生成式音乐建模任务。旨在解决现有数据集缺乏真实流行音乐的问题，通过收集真实歌曲和知名艺术家作品构建高质量数据集。**

- **链接: [http://arxiv.org/pdf/2506.14293v1](http://arxiv.org/pdf/2506.14293v1)**

> **作者:** Tawsif Ahmed; Andrej Radonjic; Gollam Rabby
>
> **摘要:** We present Sleeping-DISCO 9M, a large-scale pre-training dataset for music and song. To the best of our knowledge, there are no open-source high-quality dataset representing popular and well-known songs for generative music modeling tasks such as text-music, music-captioning, singing-voice synthesis, melody reconstruction and cross-model retrieval. Past contributions focused on isolated and constrained factors whose core perspective was to create synthetic or re-recorded music corpus (e.g. GTSinger, M4Singer) and arbitrarily large-scale audio datasets (e.g. DISCO-10M and LAIONDISCO-12M) had been another focus for the community. Unfortunately, adoption of these datasets has been below substantial in the generative music community as these datasets fail to reflect real-world music and its flavour. Our dataset changes this narrative and provides a dataset that is constructed using actual popular music and world-renowned artists.
>
---
#### [new 015] Pushing the Performance of Synthetic Speech Detection with Kolmogorov-Arnold Networks and Self-Supervised Learning Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成检测任务，旨在提升对抗语音欺骗攻击的性能。通过引入Kolmogorov-Arnold网络改进SSL模型，显著提高了检测效果。**

- **链接: [http://arxiv.org/pdf/2506.14153v1](http://arxiv.org/pdf/2506.14153v1)**

> **作者:** Tuan Dat Phuong; Long-Vu Hoang; Huy Dat Tran
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Recent advancements in speech synthesis technologies have led to increasingly advanced spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer model, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a novel architecture based on the Kolmogorov-Arnold representation theorem. Our results on ASVspoof2021 demonstrate that integrating KAN into the SSL-based models can improve the performance by 60.55% relatively on LA and DF sets, further achieving 0.70% EER on the 21LA set. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection.
>
---
#### [new 016] A Survey on World Models Grounded in Acoustic Physical Information
- **分类: cs.SD; cs.AI; cs.RO; eess.AS; physics.app-ph; 68T07, 35L05, 78A45; I.2.6; H.5.5; I.2.9**

- **简介: 该论文属于感知与建模任务，旨在利用声学物理信息构建高保真环境模型，解决动态事件预测与因果推理问题，提出相关方法并探讨其应用与挑战。**

- **链接: [http://arxiv.org/pdf/2506.13833v1](http://arxiv.org/pdf/2506.13833v1)**

> **作者:** Xiaoliang Chen; Le Chang; Xin Yu; Yunhe Huang; Xianling Tu
>
> **备注:** 28 pages,11 equations
>
> **摘要:** This survey provides a comprehensive overview of the emerging field of world models grounded in the foundation of acoustic physical information. It examines the theoretical underpinnings, essential methodological frameworks, and recent technological advancements in leveraging acoustic signals for high-fidelity environmental perception, causal physical reasoning, and predictive simulation of dynamic events. The survey explains how acoustic signals, as direct carriers of mechanical wave energy from physical events, encode rich, latent information about material properties, internal geometric structures, and complex interaction dynamics. Specifically, this survey establishes the theoretical foundation by explaining how fundamental physical laws govern the encoding of physical information within acoustic signals. It then reviews the core methodological pillars, including Physics-Informed Neural Networks (PINNs), generative models, and self-supervised multimodal learning frameworks. Furthermore, the survey details the significant applications of acoustic world models in robotics, autonomous driving, healthcare, and finance. Finally, it systematically outlines the important technical and ethical challenges while proposing a concrete roadmap for future research directions toward robust, causal, uncertainty-aware, and responsible acoustic intelligence. These elements collectively point to a research pathway towards embodied active acoustic intelligence, empowering AI systems to construct an internal "intuitive physics" engine through sound.
>
---
#### [new 017] A Comparative Study on Proactive and Passive Detection of Deepfake Speech
- **分类: cs.SD**

- **简介: 该论文属于深度伪造语音检测任务，旨在比较主动水印与被动检测方法。通过统一框架评估两者性能及鲁棒性，以解决不同场景下的最佳方案选择问题。**

- **链接: [http://arxiv.org/pdf/2506.14398v1](http://arxiv.org/pdf/2506.14398v1)**

> **作者:** Chia-Hua Wu; Wanying Ge; Xin Wang; Junichi Yamagishi; Yu Tsao; Hsin-Min Wang
>
> **摘要:** Solutions for defending against deepfake speech fall into two categories: proactive watermarking models and passive conventional deepfake detectors. While both address common threats, their differences in training, optimization, and evaluation prevent a unified protocol for joint evaluation and selecting the best solutions for different cases. This work proposes a framework to evaluate both model types in deepfake speech detection. To ensure fair comparison and minimize discrepancies, all models were trained and tested on common datasets, with performance evaluated using a shared metric. We also analyze their robustness against various adversarial attacks, showing that different models exhibit distinct vulnerabilities to different speech attribute distortions. Our training and evaluation code is available at Github.
>
---
#### [new 018] Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition for Online and Offline Scenarios
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在提升在线和离线场景下的实时性和准确性。通过改进SOT框架、引入CSS和双模型结构解决延迟与精度平衡问题。**

- **链接: [http://arxiv.org/pdf/2506.14204v1](http://arxiv.org/pdf/2506.14204v1)**

> **作者:** Aswin Shanmugam Subramanian; Amit Das; Naoyuki Kanda; Jinyu Li; Xiaofei Wang; Yifan Gong
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We extend the frameworks of Serialized Output Training (SOT) to address practical needs of both streaming and offline automatic speech recognition (ASR) applications. Our approach focuses on balancing latency and accuracy, catering to real-time captioning and summarization requirements. We propose several key improvements: (1) Leveraging Continuous Speech Separation (CSS) single-channel front-end with end-to-end (E2E) systems for highly overlapping scenarios, challenging the conventional wisdom of E2E versus cascaded setups. The CSS framework improves the accuracy of the ASR system by separating overlapped speech from multiple speakers. (2) Implementing dual models -- Conformer Transducer for streaming and Sequence-to-Sequence for offline -- or alternatively, a two-pass model based on cascaded encoders. (3) Exploring segment-based SOT (segSOT) which is better suited for offline scenarios while also enhancing readability of multi-talker transcriptions.
>
---
#### [new 019] Uncertainty-Driven Radar-Inertial Fusion for Instantaneous 3D Ego-Velocity Estimation
- **分类: cs.RO; cs.AI; eess.SP**

- **简介: 该论文属于自主导航中的运动估计任务，解决传统雷达测速精度不足的问题，通过融合雷达与惯性数据，利用神经网络估计速度及不确定性，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2506.14294v1](http://arxiv.org/pdf/2506.14294v1)**

> **作者:** Prashant Kumar Rai; Elham Kowsari; Nataliya Strokina; Reza Ghabcheloo
>
> **备注:** This paper has been accepted for presentation at the 28th International Conference on Information Fusion (Fusion 2025)
>
> **摘要:** We present a method for estimating ego-velocity in autonomous navigation by integrating high-resolution imaging radar with an inertial measurement unit. The proposed approach addresses the limitations of traditional radar-based ego-motion estimation techniques by employing a neural network to process complex-valued raw radar data and estimate instantaneous linear ego-velocity along with its associated uncertainty. This uncertainty-aware velocity estimate is then integrated with inertial measurement unit data using an Extended Kalman Filter. The filter leverages the network-predicted uncertainty to refine the inertial sensor's noise and bias parameters, improving the overall robustness and accuracy of the ego-motion estimation. We evaluated the proposed method on the publicly available ColoRadar dataset. Our approach achieves significantly lower error compared to the closest publicly available method and also outperforms both instantaneous and scan matching-based techniques.
>
---
#### [new 020] A Variational Framework for Improving Naturalness in Generative Spoken Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，旨在提升生成语音的自然度。针对现有方法依赖人工特征的问题，提出一种端到端变分框架，自动编码连续语音属性以增强语义标记。**

- **链接: [http://arxiv.org/pdf/2506.14767v1](http://arxiv.org/pdf/2506.14767v1)**

> **作者:** Li-Wei Chen; Takuya Higuchi; Zakaria Aldeneh; Ahmed Hussen Abdelaziz; Alexander Rudnicky
>
> **备注:** International Conference on Machine Learning (ICML) 2025
>
> **摘要:** The success of large language models in text processing has inspired their adaptation to speech modeling. However, since speech is continuous and complex, it is often discretized for autoregressive modeling. Speech tokens derived from self-supervised models (known as semantic tokens) typically focus on the linguistic aspects of speech but neglect prosodic information. As a result, models trained on these tokens can generate speech with reduced naturalness. Existing approaches try to fix this by adding pitch features to the semantic tokens. However, pitch alone cannot fully represent the range of paralinguistic attributes, and selecting the right features requires careful hand-engineering. To overcome this, we propose an end-to-end variational approach that automatically learns to encode these continuous speech attributes to enhance the semantic tokens. Our approach eliminates the need for manual extraction and selection of paralinguistic features. Moreover, it produces preferred speech continuations according to human raters. Code, samples and models are available at https://github.com/b04901014/vae-gslm.
>
---
#### [new 021] AsyncSwitch: Asynchronous Text-Speech Adaptation for Code-Switched ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决代码转换ASR中的语言歧义和数据不足问题。通过AsyncSwitch框架，利用文本数据预训练并微调模型，提升识别效果。**

- **链接: [http://arxiv.org/pdf/2506.14190v1](http://arxiv.org/pdf/2506.14190v1)**

> **作者:** Tuan Nguyen; Huy-Dat Tran
>
> **备注:** This work has been submitted to the IEEE for possible publication. This paper is a preprint version submitted to the 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU 2025)
>
> **摘要:** Developing code-switched ASR systems is challenging due to language ambiguity and limited exposure to multilingual, code-switched data, while collecting such speech is costly. Prior work generates synthetic audio from text, but these methods are computationally intensive and hard to scale. We introduce AsyncSwitch, a novel asynchronous adaptation framework that leverages large-scale, text-rich web data to pre-expose ASR models to diverse code-switched domains before fine-tuning on paired speech-text corpora. Our three-stage process (1) trains decoder self-attention and feedforward layers on code-switched text, (2) aligns decoder and encoder via cross-attention using limited speech-text data, and (3) fully fine-tunes the entire model. Experiments with Whisper on Malay-English code-switching demonstrate a 9.02% relative WER reduction, while improving monolingual performance in Singlish, Malay, and other English variants.
>
---
#### [new 022] Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决代码切换数据稀缺问题。通过生成合成数据提升ASR性能，验证了在无真实数据情况下的可行性。**

- **链接: [http://arxiv.org/pdf/2506.14177v1](http://arxiv.org/pdf/2506.14177v1)**

> **作者:** Tuan Nguyen; Huy-Dat Tran
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Code-switching (CS), common in multilingual settings, presents challenges for ASR due to scarce and costly transcribed data caused by linguistic complexity. This study investigates building CS-ASR using synthetic CS data. We propose a phrase-level mixing method to generate synthetic CS data that mimics natural patterns. Utilizing monolingual augmented with synthetic phrase-mixed CS data to fine-tune large pretrained ASR models (Whisper, MMS, SeamlessM4T). This paper focuses on three under-resourced Southeast Asian language pairs: Malay-English (BM-EN), Mandarin-Malay (ZH-BM), and Tamil-English (TA-EN), establishing a new comprehensive benchmark for CS-ASR to evaluate the performance of leading ASR models. Experimental results show that the proposed training strategy enhances ASR performance on monolingual and CS tests, with BM-EN showing highest gains, then TA-EN and ZH-BM. This finding offers a cost-effective approach for CS-ASR development, benefiting research and industry.
>
---
## 更新

#### [replaced 001] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13300v2](http://arxiv.org/pdf/2506.13300v2)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [replaced 002] ArrayDPS: Unsupervised Blind Speech Separation with a Diffusion Prior
- **分类: eess.AS; cs.LG; cs.MM; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.05657v3](http://arxiv.org/pdf/2505.05657v3)**

> **作者:** Zhongweiyang Xu; Xulin Fan; Zhong-Qiu Wang; Xilin Jiang; Romit Roy Choudhury
>
> **备注:** Paper Accepted at ICML2025 Demo: https://arraydps.github.io/ArrayDPSDemo/ Code: https://github.com/ArrayDPS/ArrayDPS
>
> **摘要:** Blind Speech Separation (BSS) aims to separate multiple speech sources from audio mixtures recorded by a microphone array. The problem is challenging because it is a blind inverse problem, i.e., the microphone array geometry, the room impulse response (RIR), and the speech sources, are all unknown. We propose ArrayDPS to solve the BSS problem in an unsupervised, array-agnostic, and generative manner. The core idea builds on diffusion posterior sampling (DPS), but unlike DPS where the likelihood is tractable, ArrayDPS must approximate the likelihood by formulating a separate optimization problem. The solution to the optimization approximates room acoustics and the relative transfer functions between microphones. These approximations, along with the diffusion priors, iterate through the ArrayDPS sampling process and ultimately yield separated voice sources. We only need a simple single-speaker speech diffusion model as a prior along with the mixtures recorded at the microphones; no microphone array information is necessary. Evaluation results show that ArrayDPS outperforms all baseline unsupervised methods while being comparable to supervised methods in terms of SDR. Audio demos are provided at: https://arraydps.github.io/ArrayDPSDemo/.
>
---
#### [replaced 003] Generative Deep Learning and Signal Processing for Data Augmentation of Cardiac Auscultation Signals: Improving Model Robustness Using Synthetic Audio
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2410.10125v2](http://arxiv.org/pdf/2410.10125v2)**

> **作者:** Leigh Abbott; Milan Marocchi; Matthew Fynn; Yue Rong; Sven Nordholm
>
> **备注:** 21 pages, 8 figures, 10 tables
>
> **摘要:** Accurately interpreting cardiac auscultation signals plays a crucial role in diagnosing and managing cardiovascular diseases. However, the paucity of labelled data inhibits classification models' training. Researchers have turned to generative deep learning techniques combined with signal processing to augment the existing data and improve cardiac auscultation classification models to overcome this challenge. However, the primary focus of prior studies has been on model performance as opposed to model robustness. Robustness, in this case, is defined as both the in-distribution and out-of-distribution performance by measures such as Matthew's correlation coefficient. This work shows that more robust abnormal heart sound classifiers can be trained using an augmented dataset. The augmentations consist of traditional audio approaches and the creation of synthetic audio conditionally generated using the WaveGrad and DiffWave diffusion models. It is found that both the in-distribution and out-of-distribution performance can be improved over various datasets when training a convolutional neural network-based classification model with this augmented dataset. With the performance increase encompassing not only accuracy but also balanced accuracy and Matthew's correlation coefficient, an augmented dataset significantly contributes to resolving issues of imbalanced datasets. This, in turn, helps provide a more general and robust classifier.
>
---
#### [replaced 004] Controllable Dance Generation with Style-Guided Motion Diffusion
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.07871v2](http://arxiv.org/pdf/2406.07871v2)**

> **作者:** Hongsong Wang; Ying Zhu; Yang Zhang; Junbo Wang; Xin Geng; Liang Wang
>
> **摘要:** Dance plays an important role as an artistic form and expression in human culture, yet the creation of dance remains a challenging task. Most dance generation methods primarily rely solely on music, seldom taking into consideration intrinsic attributes such as music style or genre. In this work, we introduce Flexible Dance Generation with Style Description Prompts (DGSDP), a diffusion-based framework suitable for diversified tasks of dance generation by fully leveraging the semantics of music style. The core component of this framework is Music-Conditioned Style-Aware Diffusion (MCSAD), which comprises a Transformer-based network and a music Style Modulation module. The MCSAD seemly integrates music conditions and style description prompts into the dance generation framework, ensuring that generated dances are consistent with the music content and style. To facilitate flexible dance generation and accommodate different tasks, a spatial-temporal masking strategy is effectively applied in the backward diffusion process. The proposed framework successfully generates realistic dance sequences that are accurately aligned with music for a variety of tasks such as long-term generation, dance in-betweening, dance inpainting, and etc. We hope that this work has the potential to inspire dance generation and creation, with promising applications in entertainment, art, and education. Code is available on Github: https://github.com/mucunzhuzhu/DGSDP.
>
---
#### [replaced 005] Target Speaker Extraction through Comparing Noisy Positive and Negative Audio Enrollments
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.16611v2](http://arxiv.org/pdf/2502.16611v2)**

> **作者:** Shitong Xu; Yiyuan Yang; Niki Trigoni; Andrew Markham
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Target speaker extraction focuses on isolating a specific speaker's voice from an audio mixture containing multiple speakers. To provide information about the target speaker's identity, prior works have utilized clean audio samples as conditioning inputs. However, such clean audio examples are not always readily available. For instance, obtaining a clean recording of a stranger's voice at a cocktail party without leaving the noisy environment is generally infeasible. Limited prior research has explored extracting the target speaker's characteristics from noisy enrollments, which may contain overlapping speech from interfering speakers. In this work, we explore a novel enrollment strategy that encodes target speaker information from the noisy enrollment by comparing segments where the target speaker is talking (Positive Enrollments) with segments where the target speaker is silent (Negative Enrollments). Experiments show the effectiveness of our model architecture, which achieves over 2.1 dB higher SI-SNRi compared to prior works in extracting the monaural speech from the mixture of two speakers. Additionally, the proposed two-stage training strategy accelerates convergence, reducing the number of optimization steps required to reach 3 dB SNR by 60\%. Overall, our method achieves state-of-the-art performance in the monaural target speaker extraction conditioned on noisy enrollments.
>
---
#### [replaced 006] Multi-Source Music Generation with Latent Diffusion
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.06190v4](http://arxiv.org/pdf/2409.06190v4)**

> **作者:** Zhongweiyang Xu; Debottam Dutta; Yu-Lin Wei; Romit Roy Choudhury
>
> **摘要:** Most music generation models directly generate a single music mixture. To allow for more flexible and controllable generation, the Multi-Source Diffusion Model (MSDM) has been proposed to model music as a mixture of multiple instrumental sources (e.g. piano, drums, bass, and guitar). Its goal is to use one single diffusion model to generate mutually-coherent music sources, that are then mixed to form the music. Despite its capabilities, MSDM is unable to generate music with rich melodies and often generates empty sounds. Its waveform diffusion approach also introduces significant Gaussian noise artifacts that compromise audio quality. In response, we introduce a Multi-Source Latent Diffusion Model (MSLDM) that employs Variational Autoencoders (VAEs) to encode each instrumental source into a distinct latent representation. By training a VAE on all music sources, we efficiently capture each source's unique characteristics in a "source latent." The source latents are concatenated and our diffusion model learns this joint latent space. This approach significantly enhances the total and partial generation of music by leveraging the VAE's latent compression and noise-robustness. The compressed source latent also facilitates more efficient generation. Subjective listening tests and Frechet Audio Distance (FAD) scores confirm that our model outperforms MSDM, showcasing its practical and enhanced applicability in music generation systems. We also emphasize that modeling sources is more effective than direct music mixture modeling. Codes and models are available at https://github.com/XZWY/MSLDM. Demos are available at https://xzwy.github.io/MSLDMDemo/.
>
---
#### [replaced 007] Quality-aware Masked Diffusion Transformer for Enhanced Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.15863v4](http://arxiv.org/pdf/2405.15863v4)**

> **作者:** Chang Li; Ruoyu Wang; Lijuan Liu; Jun Du; Yixuan Sun; Zilu Guo; Zhenrong Zhang; Yuan Jiang; Jianqing Gao; Feng Ma
>
> **备注:** IJCAI
>
> **摘要:** Text-to-music (TTM) generation, which converts textual descriptions into audio, opens up innovative avenues for multimedia creation. Achieving high quality and diversity in this process demands extensive, high-quality data, which are often scarce in available datasets. Most open-source datasets frequently suffer from issues like low-quality waveforms and low text-audio consistency, hindering the advancement of music generation models. To address these challenges, we propose a novel quality-aware training paradigm for generating high-quality, high-musicality music from large-scale, quality-imbalanced datasets. Additionally, by leveraging unique properties in the latent space of musical signals, we adapt and implement a masked diffusion transformer (MDT) model for the TTM task, showcasing its capacity for quality control and enhanced musicality. Furthermore, we introduce a three-stage caption refinement approach to address low-quality captions' issue. Experiments show state-of-the-art (SOTA) performance on benchmark datasets including MusicCaps and the Song-Describer Dataset with both objective and subjective metrics. Demo audio samples are available at https://qa-mdt.github.io/, code and pretrained checkpoints are open-sourced at https://github.com/ivcylc/OpenMusic.
>
---
#### [replaced 008] Discrete Audio Tokens: More Than a Survey!
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10274v2](http://arxiv.org/pdf/2506.10274v2)**

> **作者:** Pooneh Mousavi; Gallil Maimon; Adel Moumen; Darius Petermann; Jiatong Shi; Haibin Wu; Haici Yang; Anastasia Kuznetsova; Artem Ploujnikov; Ricard Marxer; Bhuvana Ramabhadran; Benjamin Elizalde; Loren Lugosch; Jinyu Li; Cem Subakan; Phil Woodland; Minje Kim; Hung-yi Lee; Shinji Watanabe; Yossi Adi; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream tasks. They provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: https://poonehmousavi.github.io/dates-website/.
>
---
