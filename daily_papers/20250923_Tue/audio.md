# 音频 cs.SD;  eess.SP

- **最新发布 57 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] RISE: Adaptive music playback for Realtime Intensity Synchronization with Exercise
- **分类: cs.SD**

- **简介: 该论文提出RISE系统，旨在解决运动时音乐节奏与锻炼强度不匹配的问题。通过动态调整音乐段落，使高能量部分与高强度训练同步，提升锻炼动力和效果。**

- **链接: [http://arxiv.org/pdf/2509.17112v1](http://arxiv.org/pdf/2509.17112v1)**

> **作者:** Alexander Wang; Chris Donahue; Dhruv Jain
>
> **备注:** ISMIR 2025
>
> **摘要:** We propose a system to adapt a user's music to their exercise by aligning high-energy music segments with intense intervals of the workout. Listening to music during exercise can boost motivation and performance. However, the structure of the music may be different from the user's natural phases of rest and work, causing users to rest longer than needed while waiting for a motivational section, or lose motivation mid-work if the section ends too soon. To address this, our system, called RISE, automatically estimates the intense segments in music and uses component-based music rearrangement techniques to dynamically extend and shorten different segments of the user's song to fit the ongoing exercise routine. Our system takes as input the rest and work durations to guide adaptation. Currently, this is determined either via a pre-defined plan or manual input during the workout. We evaluated RISE with 12 participants and compared our system to a non-adaptive music baseline while exercising in our lab. Participants found our rearrangements keeps intensity estimation accurate, and many recalled moments when intensity alignment helped them push through their workout.
>
---
#### [new 002] AISTAT lab system for DCASE2025 Task6: Language-based audio retrieval
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对DCASE2025任务6中的语言驱动音频检索问题，提出双编码器架构，结合对比学习、知识蒸馏和大模型数据增强技术，并引入聚类辅助微调，有效提升了检索性能。**

- **链接: [http://arxiv.org/pdf/2509.16649v1](http://arxiv.org/pdf/2509.16649v1)**

> **作者:** Hyun Jun Kim; Hyeong Yong Choi; Changwon Lim
>
> **备注:** 5 pages, 1 figure, DCASE2025 Task2 technical report
>
> **摘要:** This report presents the AISTAT team's submission to the language-based audio retrieval task in DCASE 2025 Task 6. Our proposed system employs dual encoder architecture, where audio and text modalities are encoded separately, and their representations are aligned using contrastive learning. Drawing inspiration from methodologies of the previous year's challenge, we implemented a distillation approach and leveraged large language models (LLMs) for effective data augmentation techniques, including back-translation and LLM mix. Additionally, we incorporated clustering to introduce an auxiliary classification task for further finetuning. Our best single system achieved a mAP@16 of 46.62, while an ensemble of four systems reached a mAP@16 of 48.83 on the Clotho development test split.
>
---
#### [new 003] Drum-to-Vocal Percussion Sound Conversion and Its Evaluation Methodology
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出鼓声到人声打击乐（VP）的音色转换任务，旨在解决传统语音合成方法无法生成VP独特音色的问题。作者基于节奏与音色对应关系构建转换模型，并提出评估标准，使用RAVE模型进行实验验证。**

- **链接: [http://arxiv.org/pdf/2509.16862v1](http://arxiv.org/pdf/2509.16862v1)**

> **作者:** Rinka Nobukawa; Makito Kitamura; Tomohiko Nakamura; Shinnosuke Takamichi; Hiroshi Saruwatari
>
> **备注:** 6 pages, 5 figures, accepted for 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** This paper defines the novel task of drum-to-vocal percussion (VP) sound conversion. VP imitates percussion instruments through human vocalization and is frequently employed in contemporary a cappella music. It exhibits acoustic properties distinct from speech and singing (e.g., aperiodicity, noisy transients, and the absence of linguistic structure), making conventional speech or singing synthesis methods unsuitable. We thus formulate VP synthesis as a timbre transfer problem from drum sounds, leveraging their rhythmic and timbral correspondence. To support this formulation, we define three requirements for successful conversion: rhythmic fidelity, timbral consistency, and naturalness as VP. We also propose corresponding subjective evaluation criteria. We implement two baseline conversion methods using a neural audio synthesizer, the real-time audio variational autoencoder (RAVE), with and without vector quantization (VQ). Subjective experiments show that both methods produce plausible VP outputs, with the VQ-based RAVE model yielding more consistent conversion.
>
---
#### [new 004] Brainprint-Modulated Target Speaker Extraction
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出BM-TSE，用于神经引导的目标说话人提取（TSE）任务。针对EEG信号非平稳性和个体差异问题，设计了自适应编码与个性化调制机制，提升助听设备的个性化语音分离效果。**

- **链接: [http://arxiv.org/pdf/2509.17883v1](http://arxiv.org/pdf/2509.17883v1)**

> **作者:** Qiushi Han; Yuan Liao; Youhao Si; Liya Huang
>
> **备注:** 5 pages, 2 figures, conference
>
> **摘要:** Achieving robust and personalized performance in neuro-steered Target Speaker Extraction (TSE) remains a significant challenge for next-generation hearing aids. This is primarily due to two factors: the inherent non-stationarity of EEG signals across sessions, and the high inter-subject variability that limits the efficacy of generalized models. To address these issues, we propose Brainprint-Modulated Target Speaker Extraction (BM-TSE), a novel framework for personalized and high-fidelity extraction. BM-TSE first employs a spatio-temporal EEG encoder with an Adaptive Spectral Gain (ASG) module to extract stable features resilient to non-stationarity. The core of our framework is a personalized modulation mechanism, where a unified brainmap embedding is learned under the joint supervision of subject identification (SID) and auditory attention decoding (AAD) tasks. This learned brainmap, encoding both static user traits and dynamic attentional states, actively refines the audio separation process, dynamically tailoring the output to each user. Evaluations on the public KUL and Cocktail Party datasets demonstrate that BM-TSE achieves state-of-the-art performance, significantly outperforming existing methods. Our code is publicly accessible at: https://github.com/rosshan-orz/BM-TSE.
>
---
#### [new 005] Barwise Section Boundary Detection in Symbolic Music Using Convolutional Neural Networks
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐结构分析任务，旨在解决符号音乐中段落边界检测的问题。作者构建了一个人工标注的MIDI数据集，并提出基于卷积神经网络的方法，利用合成泛音编码进行分类，实现了优于音频方法的性能。**

- **链接: [http://arxiv.org/pdf/2509.16566v1](http://arxiv.org/pdf/2509.16566v1)**

> **作者:** Omar Eldeeb; Martin Malandro
>
> **摘要:** Current methods for Music Structure Analysis (MSA) focus primarily on audio data. While symbolic music can be synthesized into audio and analyzed using existing MSA techniques, such an approach does not exploit symbolic music's rich explicit representation of pitch, timing, and instrumentation. A key subproblem of MSA is section boundary detection-determining whether a given point in time marks the transition between musical sections. In this paper, we study automatic section boundary detection for symbolic music. First, we introduce a human-annotated MIDI dataset for section boundary detection, consisting of metadata from 6134 MIDI files that we manually curated from the Lakh MIDI dataset. Second, we train a deep learning model to classify the presence of section boundaries within a fixed-length musical window. Our data representation involves a novel encoding scheme based on synthesized overtones to encode arbitrary MIDI instrumentations into 3-channel piano rolls. Our model achieves an F1 score of 0.77, improving over the analogous audio-based supervised learning approach and the unsupervised block-matching segmentation (CBM) audio approach by 0.22 and 0.31, respectively. We release our dataset, code, and models.
>
---
#### [new 006] MBCodec:Thorough disentangle for high-fidelity audio compression
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出MBCodec，一种基于RVQ的多码本音频编解码器，用于高保真语音压缩。针对语义与声学信息难以解耦的问题，设计分层表征和自监督语义标记化，实现170倍压缩比（2.2 kbps）并保持高质量语音重建。**

- **链接: [http://arxiv.org/pdf/2509.17006v1](http://arxiv.org/pdf/2509.17006v1)**

> **作者:** Ruonan Zhang; Xiaoyang Hao; Yichen Han; Junjie Cao; Yue Liu; Kai Zhang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** High-fidelity neural audio codecs in Text-to-speech (TTS) aim to compress speech signals into discrete representations for faithful reconstruction. However, prior approaches faced challenges in effectively disentangling acoustic and semantic information within tokens, leading to a lack of fine-grained details in synthesized speech. In this study, we propose MBCodec, a novel multi-codebook audio codec based on Residual Vector Quantization (RVQ) that learns a hierarchically structured representation. MBCodec leverages self-supervised semantic tokenization and audio subband features from the raw signals to construct a functionally-disentangled latent space. In order to encourage comprehensive learning across various layers of the codec embedding space, we introduce adaptive dropout depths to differentially train codebooks across layers, and employ a multi-channel pseudo-quadrature mirror filter (PQMF) during training. By thoroughly decoupling semantic and acoustic features, our method not only achieves near-lossless speech reconstruction but also enables a remarkable 170x compression of 24 kHz audio, resulting in a low bit rate of just 2.2 kbps. Experimental evaluations confirm its consistent and substantial outperformance of baselines across all evaluations.
>
---
#### [new 007] SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出SVeritas，一个全面评估说话人验证系统在多种真实与合成压力条件下的基准测试套件。任务是提升说话人验证的鲁棒性，解决现有基准覆盖不全的问题。工作包括构建涵盖噪声、年龄、语言等多维度的测试集，并分析模型性能差异与弱点。**

- **链接: [http://arxiv.org/pdf/2509.17091v1](http://arxiv.org/pdf/2509.17091v1)**

> **作者:** Massa Baali; Sarthak Bisht; Francisco Teixeira; Kateryna Shapovalenko; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification (SV) models are increasingly integrated into security, personalization, and access control systems, yet their robustness to many real-world challenges remains inadequately benchmarked. These include a variety of natural and maliciously created conditions causing signal degradations or mismatches between enrollment and test data, impacting performance. Existing benchmarks evaluate only subsets of these conditions, missing others entirely. We introduce SVeritas, a comprehensive Speaker Verification tasks benchmark suite, assessing SV systems under stressors like recording duration, spontaneity, content, noise, microphone distance, reverberation, channel mismatches, audio bandwidth, codecs, speaker age, and susceptibility to spoofing and adversarial attacks. While several benchmarks do exist that each cover some of these issues, SVeritas is the first comprehensive evaluation that not only includes all of these, but also several other entirely new, but nonetheless important, real-life conditions that have not previously been benchmarked. We use SVeritas to evaluate several state-of-the-art SV models and observe that while some architectures maintain stability under common distortions, they suffer substantial performance degradation in scenarios involving cross-language trials, age mismatches, and codec-induced compression. Extending our analysis across demographic subgroups, we further identify disparities in robustness across age groups, gender, and linguistic backgrounds. By standardizing evaluation under realistic and synthetic stress conditions, SVeritas enables precise diagnosis of model weaknesses and establishes a foundation for advancing equitable and reliable speaker verification systems.
>
---
#### [new 008] PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control
- **分类: cs.SD; cs.AI; eess.IV**

- **简介: 该论文提出PGSTalker，一种基于3D高斯点绘制的实时音频驱动虚拟人头生成方法。针对NeRF方法渲染效率低、视听同步差的问题，设计像素感知密度控制策略和轻量多模态融合模块，提升渲染质量与唇音同步精度。**

- **链接: [http://arxiv.org/pdf/2509.16922v1](http://arxiv.org/pdf/2509.16922v1)**

> **作者:** Tianheng Zhu; Yinfeng Yu; Liejun Wang; Fuchun Sun; Wendong Zheng
>
> **备注:** Main paper (15 pages). Accepted for publication by ICONIP( International Conference on Neural Information Processing) 2025
>
> **摘要:** Audio-driven talking head generation is crucial for applications in virtual reality, digital avatars, and film production. While NeRF-based methods enable high-fidelity reconstruction, they suffer from low rendering efficiency and suboptimal audio-visual synchronization. This work presents PGSTalker, a real-time audio-driven talking head synthesis framework based on 3D Gaussian Splatting (3DGS). To improve rendering performance, we propose a pixel-aware density control strategy that adaptively allocates point density, enhancing detail in dynamic facial regions while reducing redundancy elsewhere. Additionally, we introduce a lightweight Multimodal Gated Fusion Module to effectively fuse audio and spatial features, thereby improving the accuracy of Gaussian deformation prediction. Extensive experiments on public datasets demonstrate that PGSTalker outperforms existing NeRF- and 3DGS-based approaches in rendering quality, lip-sync precision, and inference speed. Our method exhibits strong generalization capabilities and practical potential for real-world deployment.
>
---
#### [new 009] Convolutional Neural Network Optimization for Beehive Classification Using Bioacoustic Signals
- **分类: cs.SD; cs.OH**

- **简介: 该论文属于分类任务，旨在通过蜂巢生物声学信号识别不同蜂巢状态。研究采用卷积神经网络，并对比多种时频图像表示方法，发现Cochleagram表现最佳。同时通过剪枝、量化等优化策略，显著减小模型体积并加速推理，提升实时应用可行性。**

- **链接: [http://arxiv.org/pdf/2509.17800v1](http://arxiv.org/pdf/2509.17800v1)**

> **作者:** Harshit; Rahul Jana; Ritesh Kumar
>
> **摘要:** The behavior of honeybees is an important ecological phenomenon not only in terms of honey and beeswax production but also due to the proliferation of flora and fauna around it. The best way to study this significant phenomenon is by non-invasive monitoring of beehives using the sounds produced by various body movements that give out audio signals which can be exploited for various predictions related to the objectives mentioned above. This study investigates the application of Convolutional Neural Networks to classify and monitor different hive states with the help of joint time and frequency image representations such as Spectrogram, Mel-Spectrogram, Smoothed-Spectrogram, and Cochleagram. Our findings indicate that the Cochleagram outperformed all the other representations, achieving an accuracy of 98.31% on unseen data. Furthermore, we employed various strategies including pruning, quantization, and knowledge distillation to optimize the network and prevent any potential issues with model size. With these optimizations, the network size was lowered by 91.8% and the inference time was accelerated by 66%, increasing its suitability for real-time applications. Thus our study emphasizes the significance of using optimization approaches to minimize model size, avoid deployment problems, and expedite inference for real-time application as well as the selection of an appropriate time-frequency representation for optimal performance.
>
---
#### [new 010] Etude: Piano Cover Generation with a Three-Stage Approach -- Extract, strucTUralize, and DEcode
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出Etude，用于钢琴改编生成任务。针对现有模型难以保持原曲结构一致的问题，设计了包含提取、结构化和解码的三阶段方法，通过节奏信息预提取和改进的REMI标记方法，提升了生成音乐的结构一致性与质量。**

- **链接: [http://arxiv.org/pdf/2509.16522v1](http://arxiv.org/pdf/2509.16522v1)**

> **作者:** Tse-Yang Che; Yuh-Jzer Joung
>
> **摘要:** Piano cover generation aims to automatically transform a pop song into a piano arrangement. While numerous deep learning approaches have been proposed, existing models often fail to maintain structural consistency with the original song, likely due to the absence of beat-aware mechanisms or the difficulty of modeling complex rhythmic patterns. Rhythmic information is crucial, as it defines structural similarity (e.g., tempo, BPM) and directly impacts the overall quality of the generated music. In this paper, we introduce Etude, a three-stage architecture consisting of Extract, strucTUralize, and DEcode stages. By pre-extracting rhythmic information and applying a novel, simplified REMI-based tokenization, our model produces covers that preserve proper song structure, enhance fluency and musical dynamics, and support highly controllable generation through style injection. Subjective evaluations with human listeners show that Etude substantially outperforms prior models, achieving a quality level comparable to that of human composers.
>
---
#### [new 011] Bridging the gap between training and inference in LM-based TTS models
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对LM-based TTS任务中训练与推理不一致的问题，提出了一种提示引导的混合训练方案，结合教师强制与自由运行，并引入EOS预测机制，有效缩小训练-推理差距，提升长文本语音合成质量。**

- **链接: [http://arxiv.org/pdf/2509.17021v1](http://arxiv.org/pdf/2509.17021v1)**

> **作者:** Ruonan Zhang; Lingzhou Mu; Xixin Wu; Kai Zhang
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Recent advancements in text-to-speech (TTS) have shown that language model (LM) based systems offer competitive performance compared to traditional approaches. However, in training, TTS models use ground-truth (GT) tokens as prefixes to predict the next token, while in inference these tokens are not available, a gap between training and inference that is often neglected. In this study, we propose a prompt-guided hybrid training scheme to mitigate exposure bias in popular LM-based TTS systems. Our core idea is to adopt a hybrid training paradigm that combines teacher forcing with free running, thereby introducing self-generated tokens into the training process. This makes the training mode more consistent with inference, reducing the training-inference gap. In addition, we incorporate an EOS prediction mechanism during training to detect incorrect sequence termination and adaptively control the free running process. Experimental results provide a comprehensive evaluation of the impact of exposure bias on LM-based TTS, and demonstrate that our method effectively narrows the training-inference gap, thereby improving the quality of synthesized long-form speech.
>
---
#### [new 012] Attention-based Mixture of Experts for Robust Speech Deepfake Detection
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决AI生成语音难以区分的问题。提出基于注意力机制的专家混合模型，结合多个先进检测器，通过动态加权提升检测性能，在SAFE挑战赛中排名第一。**

- **链接: [http://arxiv.org/pdf/2509.17585v1](http://arxiv.org/pdf/2509.17585v1)**

> **作者:** Viola Negroni; Davide Salvi; Alessandro Ilic Mezza; Paolo Bestagini; Stefano Tubaro
>
> **备注:** Accepted @ IEEE WIFS 2025
>
> **摘要:** AI-generated speech is becoming increasingly used in everyday life, powering virtual assistants, accessibility tools, and other applications. However, it is also being exploited for malicious purposes such as impersonation, misinformation, and biometric spoofing. As speech deepfakes become nearly indistinguishable from real human speech, the need for robust detection methods and effective countermeasures has become critically urgent. In this paper, we present the ISPL's submission to the SAFE challenge at IH&MMSec 2025, where our system ranked first across all tasks. Our solution introduces a novel approach to audio deepfake detection based on a Mixture of Experts architecture. The proposed system leverages multiple state-of-the-art detectors, combining their outputs through an attention-based gating network that dynamically weights each expert based on the input speech signal. In this design, each expert develops a specialized understanding of the shared training data by learning to capture different complementary aspects of the same input through inductive biases. Experimental results indicate that our method outperforms existing approaches across multiple datasets. We further evaluate and analyze the performance of our system in the SAFE challenge.
>
---
#### [new 013] AudioGenie-Reasoner: A Training-Free Multi-Agent Framework for Coarse-to-Fine Audio Deep Reasoning
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出AudioGenie-Reasoner（AGR），一种无需训练的多智能体框架，用于音频深度推理任务。旨在解决音频感知与推理能力之间的差距问题，通过将音频转化为文本并迭代优化证据链，实现从粗到细的推理过程，达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.16971v1](http://arxiv.org/pdf/2509.16971v1)**

> **作者:** Yan Rong; Chenxing Li; Dong Yu; Li Liu
>
> **摘要:** Audio deep reasoning is a challenging task that requires expert-level perception, multi-step logical inference, and the integration of contextual knowledge. However, existing models suffer from a gap between audio perception and reasoning abilities due to the lack of training data with explicit reasoning chains and the absence of mechanisms for active exploration and iterative refinement. To address these challenges, we propose AudioGenie-Reasoner (AGR), the first unified training-free multi-agent system that coordinates perception and reasoning over an evolving chain of textual evidence. Our key idea is a paradigm shift that transforms audio deep reasoning into complex text understanding task from a new perspective, thereby unlocking the full potential of large language models. Specifically, the design of AGR mimics the human coarse-to-fine cognitive process. It first transforms the input audio into a coarse text-based document. Then, we design a novel proactive iterative document refinement loop, featuring tool-augmented routes and specialized agents, to continuously search for missing information and augment the evidence chain in a coarse-to-fine manner until sufficient question-related information is gathered for making final predictions. Experimental results show that AGR achieves state-of-the-art (SOTA) performance over existing open-source audio deep reasoning models across various benchmarks. The code will be made publicly available.
>
---
#### [new 014] Virtual Consistency for Audio Editing
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出一种基于虚拟一致性的音频编辑方法，旨在解决文本驱动音频编辑中依赖缓慢反转过程的问题。通过改进扩散模型的采样过程，无需微调或修改模型结构，实现了高效且高质量的音频编辑。**

- **链接: [http://arxiv.org/pdf/2509.17219v1](http://arxiv.org/pdf/2509.17219v1)**

> **作者:** Matthieu Cervera; Francesco Paissan; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Free-form, text-based audio editing remains a persistent challenge, despite progress in inversion-based neural methods. Current approaches rely on slow inversion procedures, limiting their practicality. We present a virtual-consistency based audio editing system that bypasses inversion by adapting the sampling process of diffusion models. Our pipeline is model-agnostic, requiring no fine-tuning or architectural changes, and achieves substantial speed-ups over recent neural editing baselines. Crucially, it achieves this efficiency without compromising quality, as demonstrated by quantitative benchmarks and a user study involving 16 participants.
>
---
#### [new 015] Difficulty-Aware Score Generation for Piano Sight-Reading
- **分类: cs.SD**

- **简介: 该论文属于符号音乐生成任务，旨在解决钢琴视奏练习材料难度控制的问题。提出一种辅助优化目标，增强模型对难度的感知和生成能力，提升练习材料的个性化与教育价值。**

- **链接: [http://arxiv.org/pdf/2509.16913v1](http://arxiv.org/pdf/2509.16913v1)**

> **作者:** Pedro Ramoneda; Masahiro Suzuki; Akira Maezawa; Xavier Serra
>
> **摘要:** Adapting learning materials to the level of skill of a student is important in education. In the context of music training, one essential ability is sight-reading -- playing unfamiliar scores at first sight -- which benefits from progressive and level-appropriate practice. However, creating exercises at the appropriate level of difficulty demands significant time and effort. We address this challenge as a controlled symbolic music generation task that aims to produce piano scores with a desired difficulty level. Controlling symbolic generation through conditioning is commonly done using control tokens, but these do not always have a clear impact on global properties, such as difficulty. To improve conditioning, we introduce an auxiliary optimization target for difficulty prediction that helps prevent conditioning collapse -- a common issue in which models ignore control signals in the absence of explicit supervision. This auxiliary objective helps the model to learn internal representations aligned with the target difficulty, enabling more precise and adaptive score generation. Evaluation with automatic metrics and expert judgments shows better control of difficulty and potential educational value. Our approach represents a step toward personalized music education through the generation of difficulty-aware practice material.
>
---
#### [new 016] Idiosyncratic Versus Normative Modeling of Atypical Speech Recognition: Dysarthric Case Studies
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究了非典型语音（如构音障碍）的自动语音识别（ASR）问题。任务是提升ASR在特殊人群上的表现，对比了四种建模策略，发现结合常规与个性化模式的方法效果更优，且所需数据更少。**

- **链接: [http://arxiv.org/pdf/2509.16718v1](http://arxiv.org/pdf/2509.16718v1)**

> **作者:** Vishnu Raja; Adithya V Ganesan; Anand Syamkumar; Ritwik Banerjee; H Andrew Schwartz
>
> **备注:** Will appear in EMNLP 2025 Main Proceedings
>
> **摘要:** State-of-the-art automatic speech recognition (ASR) models like Whisper, perform poorly on atypical speech, such as that produced by individuals with dysarthria. Past works for atypical speech have mostly investigated fully personalized (or idiosyncratic) models, but modeling strategies that can both generalize and handle idiosyncracy could be more effective for capturing atypical speech. To investigate this, we compare four strategies: (a) $\textit{normative}$ models trained on typical speech (no personalization), (b) $\textit{idiosyncratic}$ models completely personalized to individuals, (c) $\textit{dysarthric-normative}$ models trained on other dysarthric speakers, and (d) $\textit{dysarthric-idiosyncratic}$ models which combine strategies by first modeling normative patterns before adapting to individual speech. In this case study, we find the dysarthric-idiosyncratic model performs better than idiosyncratic approach while requiring less than half as much personalized data (36.43 WER with 128 train size vs 36.99 with 256). Further, we found that tuning the speech encoder alone (as opposed to the LM decoder) yielded the best results reducing word error rate from 71% to 32% on average. Our findings highlight the value of leveraging both normative (cross-speaker) and idiosyncratic (speaker-specific) patterns to improve ASR for underrepresented speech populations.
>
---
#### [new 017] Cross-Attention with Confidence Weighting for Multi-Channel Audio Alignment
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文针对多通道音频对齐任务，旨在解决非线性时钟漂移和缺乏不确定性量化的问题。提出结合交叉注意力机制与置信度加权评分的方法，提升同步精度，并在BioDCASE 2025 Task 1中取得最佳表现。**

- **链接: [http://arxiv.org/pdf/2509.16926v1](http://arxiv.org/pdf/2509.16926v1)**

> **作者:** Ragib Amin Nihal; Benjamin Yen; Takeshi Ashizawa; Kazuhiro Nakadai
>
> **备注:** Accepted on Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2025)
>
> **摘要:** Multi-channel audio alignment is a key requirement in bioacoustic monitoring, spatial audio systems, and acoustic localization. However, existing methods often struggle to address nonlinear clock drift and lack mechanisms for quantifying uncertainty. Traditional methods like Cross-correlation and Dynamic Time Warping assume simple drift patterns and provide no reliability measures. Meanwhile, recent deep learning models typically treat alignment as a binary classification task, overlooking inter-channel dependencies and uncertainty estimation. We introduce a method that combines cross-attention mechanisms with confidence-weighted scoring to improve multi-channel audio synchronization. We extend BEATs encoders with cross-attention layers to model temporal relationships between channels. We also develop a confidence-weighted scoring function that uses the full prediction distribution instead of binary thresholding. Our method achieved first place in the BioDCASE 2025 Task 1 challenge with 0.30 MSE average across test datasets, compared to 0.58 for the deep learning baseline. On individual datasets, we achieved 0.14 MSE on ARU data (77% reduction) and 0.45 MSE on zebra finch data (18% reduction). The framework supports probabilistic temporal alignment, moving beyond point estimates. While validated in a bioacoustic context, the approach is applicable to a broader range of multi-channel audio tasks where alignment confidence is critical. Code available on: https://github.com/Ragib-Amin-Nihal/BEATsCA
>
---
#### [new 018] Fusing Spectral Correlation Density Imaging with Deep Learning for Intelligent Fault Diagnosis in Rotating Machinery
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于故障诊断任务，旨在解决旋转机械轴承故障的早期检测问题。通过将振动信号转化为SCD图像，并结合三种CNN模型进行分类，实现了高精度的多工况故障识别，验证了方法在不同环境下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16580v1](http://arxiv.org/pdf/2509.16580v1)**

> **作者:** Dilshara Herath; Chinthaka Abeyrathne; Chamindu Adithya; Chathura Seneviratne
>
> **摘要:** Bearing fault diagnosis in rotating machinery is critical for ensuring operational reliability, therefore early fault detection is essential to avoid catastrophic failures and expensive emergency repairs. Traditional methods like Fast Fourier Transform (FFT) often fail to capture the complex, non-stationary nature of vibration signals. This study leverages the cyclostationary properties of vibration data through Spectral Correlation Density (SCD) images to enhance fault detection and apply deep learning for classification. Using a publicly available dataset with bearing faults seeded in two distinct housings (A and B) under varying load conditions (0 Nm, 2 Nm, 4 Nm), we processed vibration signals into 2D SCD images to reveal fault-specific periodicities, such as broadband spectra (2000--8000 Hz) for larger faults. Three convolutional neural network (CNN) models, Custom CNN, ResNet152V2, and EfficientNetB0, were developed to classify seven bearing conditions. The custom CNN achieved the highest accuracies of 96.58\% and 94.95\% on Housing A and B, respectively, followed by ResNet152V2 at 96.49\% and 95.35\%, and EfficientNetB0 at 94.16\% and 91.65\%, respectively. The models' high accuracies across different housings demonstrate a robust solution suitable for cost-effective condition monitoring deployable near sensing platforms, contributing to applied machine learning for edge intelligence and showcasing effective signal processing strategies for handling complex, potentially large-scale vibration data.
>
---
#### [new 019] Sidon: Fast and Robust Open-Source Multilingual Speech Restoration for Large-scale Dataset Cleansing
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Sidon，一种快速开源的多语言语音修复模型，用于大规模数据集清洗。针对真实场景中噪声语音的问题，Sidon通过特征预测和声码器合成高质量语音，性能接近Google内部模型且计算高效，显著提升TTS系统表现。**

- **链接: [http://arxiv.org/pdf/2509.17052v1](http://arxiv.org/pdf/2509.17052v1)**

> **作者:** Wataru Nakata; Yuki Saito; Yota Ueda; Hiroshi Saruwatari
>
> **备注:** 5 pages, 1 figures
>
> **摘要:** Large-scale text-to-speech (TTS) systems are limited by the scarcity of clean, multilingual recordings. We introduce Sidon, a fast, open-source speech restoration model that converts noisy in-the-wild speech into studio-quality speech and scales to dozens of languages. Sidon consists of two models: w2v-BERT 2.0 finetuned feature predictor to cleanse features from noisy speech and vocoder trained to synthesize restored speech from the cleansed features. Sidon achieves restoration performance comparable to Miipher: Google's internal speech restoration model with the aim of dataset cleansing for speech synthesis. Sidon is also computationally efficient, running up to 3,390 times faster than real time on a single GPU. We further show that training a TTS model using a Sidon-cleansed automatic speech recognition corpus improves the quality of synthetic speech in a zero-shot setting. Code and model are released to facilitate reproducible dataset cleansing for the research community.
>
---
#### [new 020] STAR: Speech-to-Audio Generation via Representation Learning
- **分类: cs.SD; eess.AS; 68Txx; I.2**

- **简介: 该论文提出STAR，首个端到端语音到音频生成框架。旨在解决级联系统效率低和误差传播问题。通过表示学习提取语音语义，并采用桥接网络和两阶段训练策略，实现高效音频生成。**

- **链接: [http://arxiv.org/pdf/2509.17164v1](http://arxiv.org/pdf/2509.17164v1)**

> **作者:** Zeyu Xie; Xuenan Xu; Yixuan Li; Mengyue Wu; Yuexian Zou
>
> **摘要:** This work presents STAR, the first end-to-end speech-to-audio generation framework, designed to enhance efficiency and address error propagation inherent in cascaded systems. Unlike prior approaches relying on text or vision, STAR leverages speech as it constitutes a natural modality for interaction. As an initial step to validate the feasibility of the system, we demonstrate through representation learning experiments that spoken sound event semantics can be effectively extracted from raw speech, capturing both auditory events and scene cues. Leveraging the semantic representations, STAR incorporates a bridge network for representation mapping and a two-stage training strategy to achieve end-to-end synthesis. With a 76.9% reduction in speech processing latency, STAR demonstrates superior generation performance over the cascaded systems. Overall, STAR establishes speech as a direct interaction signal for audio generation, thereby bridging representation learning and multimodal synthesis. Generated samples are available at https://zeyuxie29.github.io/STAR.
>
---
#### [new 021] Leveraging Multiple Speech Enhancers for Non-Intrusive Intelligibility Prediction for Hearing-Impaired Listeners
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究非侵入式语音可懂度预测任务，旨在解决听力障碍者在无参考信号情况下评估助听器性能的问题。提出利用多个语音增强器构建增强信号路径，并引入2-clips数据增强策略，提升跨数据集泛化能力，优于现有基线方法。**

- **链接: [http://arxiv.org/pdf/2509.16979v1](http://arxiv.org/pdf/2509.16979v1)**

> **作者:** Boxuan Cao; Linkai Li; Hanlin Yu; Changgeng Mo; Haoshuai Zhou; Shan Xiang Wang
>
> **摘要:** Speech intelligibility evaluation for hearing-impaired (HI) listeners is essential for assessing hearing aid performance, traditionally relying on listening tests or intrusive methods like HASPI. However, these methods require clean reference signals, which are often unavailable in real-world conditions, creating a gap between lab-based and real-world assessments. To address this, we propose a non-intrusive intelligibility prediction framework that leverages speech enhancers to provide a parallel enhanced-signal pathway, enabling robust predictions without reference signals. We evaluate three state-of-the-art enhancers and demonstrate that prediction performance depends on the choice of enhancer, with ensembles of strong enhancers yielding the best results. To improve cross-dataset generalization, we introduce a 2-clips augmentation strategy that enhances listener-specific variability, boosting robustness on unseen datasets. Our approach consistently outperforms the non-intrusive baseline, CPC2 Champion across multiple datasets, highlighting the potential of enhancer-guided non-intrusive intelligibility prediction for real-world applications.
>
---
#### [new 022] FakeSound2: A Benchmark for Explainable and Generalizable Deepfake Sound Detection
- **分类: cs.SD; eess.AS; 68Txx; I.2**

- **简介: 该论文提出FakeSound2，用于提升深度伪造音频检测的可解释性与泛化能力。针对现有方法在定位、溯源和泛化上的不足，构建包含6种篡改类型和12个来源的基准，推动更鲁棒、可信的音频认证研究。**

- **链接: [http://arxiv.org/pdf/2509.17162v1](http://arxiv.org/pdf/2509.17162v1)**

> **作者:** Zeyu Xie; Yaoyun Zhang; Xuenan Xu; Yongkang Yin; Chenxing Li; Mengyue Wu; Yuexian Zou
>
> **摘要:** The rapid development of generative audio raises ethical and security concerns stemming from forged data, making deepfake sound detection an important safeguard against the malicious use of such technologies. Although prior studies have explored this task, existing methods largely focus on binary classification and fall short in explaining how manipulations occur, tracing where the sources originated, or generalizing to unseen sources-thereby limiting the explainability and reliability of detection. To address these limitations, we present FakeSound2, a benchmark designed to advance deepfake sound detection beyond binary accuracy. FakeSound2 evaluates models across three dimensions: localization, traceability, and generalization, covering 6 manipulation types and 12 diverse sources. Experimental results show that although current systems achieve high classification accuracy, they struggle to recognize forged pattern distributions and provide reliable explanations. By highlighting these gaps, FakeSound2 establishes a comprehensive benchmark that reveals key challenges and aims to foster robust, explainable, and generalizable approaches for trustworthy audio authentication.
>
---
#### [new 023] Interpretable Audio Editing Evaluation via Chain-of-Thought Difference-Commonality Reasoning with Multimodal LLMs
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频编辑评估任务，提出首个基于多模态大语言模型的自然语言自动化评估框架。通过引入细粒度微调任务和链式推理提示，提升了模型对多音频的理解与可解释性评估能力，实验表明其效果优于基线方法且贴近人类判断。**

- **链接: [http://arxiv.org/pdf/2509.16975v1](http://arxiv.org/pdf/2509.16975v1)**

> **作者:** Yuhang Jia; Xu Zhang; Yang Chen; Hui Wang; Enzhi Wang; Yong Qin
>
> **摘要:** Automatic mean opinion score (MOS) prediction provides a more perceptual alternative to objective metrics, offering deeper insights into the evaluated models. With the rapid progress of multimodal large language models (MLLMs), their enhanced perceptual and reasoning abilities enable more comprehensive and interpretable audio quality assessment. In this work, we tackle the challenging task of audio editing evaluation and propose the first natural language-based automated evaluation framework built on MLLMs. Our approach introduces two fine-tuning tasks to boost multi-audio understanding, combined with Chain-of-Thought prompting, and lightweight instruction tuning, to enhance step-by-step reasoning. Experiment demonstrate that our framework delivers accurate, interpretable, and text-based editing evaluation, closely aligning with human judgments and objective metrics while substantially improving over baselines. The code and demo are available at https://github.com/NKU-HLT/Eval_Reasoning.
>
---
#### [new 024] On the de-duplication of the Lakh MIDI dataset
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决Lakh MIDI数据集中的重复数据问题。通过评估多种方法（包括规则方法、模型和对比学习），提出三种去重版本，最多过滤38,134个样本，提升数据质量。**

- **链接: [http://arxiv.org/pdf/2509.16662v1](http://arxiv.org/pdf/2509.16662v1)**

> **作者:** Eunjin Choi; Hyerin Kim; Jiwoo Ryu; Juhan Nam; Dasaem Jeong
>
> **备注:** The paper has been accepted for publication at ISMIR 2025
>
> **摘要:** A large-scale dataset is essential for training a well-generalized deep-learning model. Most such datasets are collected via scraping from various internet sources, inevitably introducing duplicated data. In the symbolic music domain, these duplicates often come from multiple user arrangements and metadata changes after simple editing. However, despite critical issues such as unreliable training evaluation from data leakage during random splitting, dataset duplication has not been extensively addressed in the MIR community. This study investigates the dataset duplication issues regarding Lakh MIDI Dataset (LMD), one of the largest publicly available sources in the symbolic music domain. To find and evaluate the best retrieval method for duplicated data, we employed the Clean MIDI subset of the LMD as a benchmark test set, in which different versions of the same songs are grouped together. We first evaluated rule-based approaches and previous symbolic music retrieval models for de-duplication and also investigated with a contrastive learning-based BERT model with various augmentations to find duplicate files. As a result, we propose three different versions of the filtered list of LMD, which filters out at least 38,134 samples in the most conservative settings among 178,561 files.
>
---
#### [new 025] MRADNET: a Compact Radar Object Detector with MetaFormer
- **分类: eess.SP; cs.CV**

- **简介: 该论文提出mRadNet，一种用于雷达目标检测的轻量级模型。针对车载嵌入式系统对模型紧凑性和效率的需求，采用U-net结构与MetaFormer模块，结合分离卷积和注意力机制，提升检测性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.16223v1](http://arxiv.org/pdf/2509.16223v1)**

> **作者:** Huaiyu Chen; Fahed Hassanat; Robert Laganiere; Martin Bouchard
>
> **备注:** 5 pages, 2 figures, submitted to IEEE Icassp 2026
>
> **摘要:** Frequency-modulated continuous wave radars have gained increasing popularity in the automotive industry. Its robustness against adverse weather conditions makes it a suitable choice for radar object detection in advanced driver assistance systems. These real-time embedded systems have requirements for the compactness and efficiency of the model, which have been largely overlooked in previous work. In this work, we propose mRadNet, a novel radar object detection model with compactness in mind. mRadNet employs a U-net style architecture with MetaFormer blocks, in which separable convolution and attention token mixers are used to capture both local and global features effectively. More efficient token embedding and merging strategies are introduced to further facilitate the lightweight design of the model. The performance of mRadNet is validated on the CRUW dataset, improving state-of-the-art performance.
>
---
#### [new 026] Speech-to-See: End-to-End Speech-Driven Open-Set Object Detection
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文研究语音驱动的开放集目标检测任务，旨在通过语音直接定位和识别图像中的物体。针对数据稀缺和依赖文本中介的问题，提出Speech-to-See方法，采用端到端预训练与微调框架，引入语义聚合模块和高效MoLE架构，提升跨模态适应能力与泛化性能。**

- **链接: [http://arxiv.org/pdf/2509.16670v1](http://arxiv.org/pdf/2509.16670v1)**

> **作者:** Wenhuan Lu; Xinyue Song; Wenjun Ke; Zhizhi Yu; Wenhao Yang; Jianguo Wei
>
> **摘要:** Audio grounding, or speech-driven open-set object detection, aims to localize and identify objects directly from speech, enabling generalization beyond predefined categories. This task is crucial for applications like human-robot interaction where textual input is impractical. However, progress in this domain faces a fundamental bottleneck from the scarcity of large-scale, paired audio-image data, and is further constrained by previous methods that rely on indirect, text-mediated pipelines. In this paper, we introduce Speech-to-See (Speech2See), an end-to-end approach built on a pre-training and fine-tuning paradigm. Specifically, in the pre-training stage, we design a Query-Guided Semantic Aggregation module that employs learnable queries to condense redundant speech embeddings into compact semantic representations. During fine-tuning, we incorporate a parameter-efficient Mixture-of-LoRA-Experts (MoLE) architecture to achieve deeper and more nuanced cross-modal adaptation. Extensive experiments show that Speech2See achieves robust and adaptable performance across multiple benchmarks, demonstrating its strong generalization ability and broad applicability.
>
---
#### [new 027] Audio Super-Resolution with Latent Bridge Models
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究音频超分辨率任务，旨在提升低分辨率音频的采样质量。提出基于潜在桥接模型（LBM）的新方法，通过压缩音频到潜在空间并设计频率感知模型，实现高质量的任何采样率到48kHz/192kHz的音频上采样，达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.17609v1](http://arxiv.org/pdf/2509.17609v1)**

> **作者:** Chang Li; Zehua Chen; Liyuan Wang; Jun Zhu
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Audio super-resolution (SR), i.e., upsampling the low-resolution (LR) waveform to the high-resolution (HR) version, has recently been explored with diffusion and bridge models, while previous methods often suffer from sub-optimal upsampling quality due to their uninformative generation prior. Towards high-quality audio super-resolution, we present a new system with latent bridge models (LBMs), where we compress the audio waveform into a continuous latent space and design an LBM to enable a latent-to-latent generation process that naturally matches the LR-toHR upsampling process, thereby fully exploiting the instructive prior information contained in the LR waveform. To further enhance the training results despite the limited availability of HR samples, we introduce frequency-aware LBMs, where the prior and target frequency are taken as model input, enabling LBMs to explicitly learn an any-to-any upsampling process at the training stage. Furthermore, we design cascaded LBMs and present two prior augmentation strategies, where we make the first attempt to unlock the audio upsampling beyond 48 kHz and empower a seamless cascaded SR process, providing higher flexibility for audio post-production. Comprehensive experimental results evaluated on the VCTK, ESC-50, Song-Describer benchmark datasets and two internal testsets demonstrate that we achieve state-of-the-art objective and perceptual quality for any-to-48kHz SR across speech, audio, and music signals, as well as setting the first record for any-to-192kHz audio SR. Demo at https://AudioLBM.github.io/.
>
---
#### [new 028] Reference-aware SFM layers for intrusive intelligibility prediction
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究语音可懂度预测任务，旨在提升基于参考信号的预测模型性能。作者结合参考信号与多层语音基础模型（SFM）表示，提出新的方法，在CPC3数据集上取得最优效果。**

- **链接: [http://arxiv.org/pdf/2509.17270v1](http://arxiv.org/pdf/2509.17270v1)**

> **作者:** Hanlin Yu; Haoshuai Zhou; Boxuan Cao; Changgeng Mo; Linkai Li; Shan X. Wang
>
> **备注:** Preprint; submitted to ICASSP 2026. 5 pages. CPC3 system: Dev RMSE 22.36, Eval RMSE 24.98 (ranked 1st)
>
> **摘要:** Intrusive speech-intelligibility predictors that exploit explicit reference signals are now widespread, yet they have not consistently surpassed non-intrusive systems. We argue that a primary cause is the limited exploitation of speech foundation models (SFMs). This work revisits intrusive prediction by combining reference conditioning with multi-layer SFM representations. Our final system achieves RMSE 22.36 on the development set and 24.98 on the evaluation set, ranking 1st on CPC3. These findings provide practical guidance for constructing SFM-based intrusive intelligibility predictors.
>
---
#### [new 029] LenslessMic: Audio Encryption and Authentication via Lensless Computational Imaging
- **分类: cs.CR; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出LenslessMic，一种基于无透镜相机的音频加密与认证方法。通过光学硬件实现物理层安全，解决传统音频加密依赖软件的问题，提供高安全性与音质保障，并通过低成本原型验证效果。**

- **链接: [http://arxiv.org/pdf/2509.16418v1](http://arxiv.org/pdf/2509.16418v1)**

> **作者:** Petr Grinberg; Eric Bezzam; Paolo Prandoni; Martin Vetterli
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** With society's increasing reliance on digital data sharing, the protection of sensitive information has become critical. Encryption serves as one of the privacy-preserving methods; however, its realization in the audio domain predominantly relies on signal processing or software methods embedded into hardware. In this paper, we introduce LenslessMic, a hybrid optical hardware-based encryption method that utilizes a lensless camera as a physical layer of security applicable to multiple types of audio. We show that LenslessMic enables (1) robust authentication of audio recordings and (2) encryption strength that can rival the search space of 256-bit digital standards, while maintaining high-quality signals and minimal loss of content information. The approach is validated with a low-cost Raspberry Pi prototype and is open-sourced together with datasets to facilitate research in the area.
>
---
#### [new 030] HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO; cs.AI; cs.ET; cs.SY; eess.SP; eess.SY**

- **简介: 该论文提出HuMam，一种基于端到端深度强化学习的人形机器人运动控制框架。针对训练不稳定、特征融合效率低和能耗高的问题，采用Mamba编码器进行状态融合，并设计六项奖励函数优化控制策略，提升了学习效率与稳定性，降低了能耗。**

- **链接: [http://arxiv.org/pdf/2509.18046v1](http://arxiv.org/pdf/2509.18046v1)**

> **作者:** Yinuo Wang; Yuanyang Qi; Jinzhao Zhou; Gavin Tao
>
> **备注:** 10 pages
>
> **摘要:** End-to-end reinforcement learning (RL) for humanoid locomotion is appealing for its compact perception-action mapping, yet practical policies often suffer from training instability, inefficient feature fusion, and high actuation cost. We present HuMam, a state-centric end-to-end RL framework that employs a single-layer Mamba encoder to fuse robot-centric states with oriented footstep targets and a continuous phase clock. The policy outputs joint position targets tracked by a low-level PD loop and is optimized with PPO. A concise six-term reward balances contact quality, swing smoothness, foot placement, posture, and body stability while implicitly promoting energy saving. On the JVRC-1 humanoid in mc-mujoco, HuMam consistently improves learning efficiency, training stability, and overall task performance over a strong feedforward baseline, while reducing power consumption and torque peaks. To our knowledge, this is the first end-to-end humanoid RL controller that adopts Mamba as the fusion backbone, demonstrating tangible gains in efficiency, stability, and control economy.
>
---
#### [new 031] Harmonic Summation-Based Robust Pitch Estimation in Noisy and Reverberant Environments
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对噪声和混响环境下语音基频估计的挑战，提出了一种基于谐波求和的鲁棒方法。通过改进NAMDF计算、引入概率状态和Viterbi连续性约束，有效降低了基频误差，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16480v1](http://arxiv.org/pdf/2509.16480v1)**

> **作者:** Anup Singh; Kris Demuynck
>
> **摘要:** Accurate pitch estimation is essential for numerous speech processing applications, yet it remains challenging in high-distortion environments. This paper proposes a robust pitch estimation method that delivers robust pitch estimates in challenging noise environments. Our approach computes the Normalized Average Magnitude Difference Function (NAMDF), transforms it into a likelihood function, and generates probabilistic pitch states for frames at each sample shift. To enhance noise robustness, we aggregate likelihood values across integer multiples of the pitch period and neighboring frames. Furthermore, we introduce a simple yet effective continuity constraint in the Viterbi algorithm to refine pitch selection among multiple candidates. Experimental results show that our method consistently achieves lower Gross Pitch Error (GPE) and Voicing Decision Error (VDE) across various SNR levels, outperforming existing methods in both noisy and reverberant conditions.
>
---
#### [new 032] Comparator Loss: An Ordinal Contrastive Loss to Derive a Severity Score for Speech-based Health Monitoring
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出一种用于语音健康监测的序数对比损失（Comparator Loss），旨在解决神经退行性疾病严重程度量化问题。通过训练模型生成疾病严重程度评分，可反映患者病情进展，并与临床指标相关联，提升小数据集的利用效率。**

- **链接: [http://arxiv.org/pdf/2509.17661v1](http://arxiv.org/pdf/2509.17661v1)**

> **作者:** Jacob J Webber; Oliver Watts; Lovisa Wihlborg; David Wheatley; Johnny Tam; Christine Weaver; Suvankar Pal; Siddharthan Chandran; Cassia Valentini-Botinhao
>
> **备注:** Submitted to ICASSP 2026. This work is supported by NEURii, a collaborative partnership involving the University of Edinburgh, Gates Ventures, Eisai, LifeArc and Health Data Research UK (HDR UK)
>
> **摘要:** Monitoring the progression of neurodegenerative disease has important applications in the planning of treatment and the evaluation of future medications. Whereas much of the state-of-the-art in health monitoring from speech has been focused on classifying patients versus healthy controls, or predicting real-world health metrics, we propose here a novel measure of disease progression: the severity score. This score is derived from a model trained to minimize what we call the comparator loss. The comparator loss ensures scores follow an ordering relation, which can be based on diagnosis, clinically annotated scores, or simply the chronological order of the recordings. In addition to giving a more detailed picture than a simple discrete classification, the proposed comparator loss-based system has the potential to incorporate information from disparate health metrics, which is critical for making full use of small health-related datasets. We evaluated our proposed models based on their ability to affirmatively track the progression of patients with motor neuron disease (MND), the correlation of their output with clinical annotations such as ALSFRS-R, as well as their ability to distinguish between subjects with MND and healthy controls.
>
---
#### [new 033] Advancing Speech Understanding in Speech-Aware Language Models with GRPO
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出基于GRPO的方法，用于训练面向开放格式语音理解任务（如语音问答和语音翻译）的语音感知大语言模型（SALLMs），利用BLEU作为奖励信号优化模型，超越了标准SFT方法。**

- **链接: [http://arxiv.org/pdf/2509.16990v1](http://arxiv.org/pdf/2509.16990v1)**

> **作者:** Avishai Elmakies; Hagai Aronowitz; Nimrod Shabtay; Eli Schwartz; Ron Hoory; Avihu Dekel
>
> **摘要:** In this paper, we introduce a Group Relative Policy Optimization (GRPO)-based method for training Speech-Aware Large Language Models (SALLMs) on open-format speech understanding tasks, such as Spoken Question Answering and Automatic Speech Translation. SALLMs have proven highly effective for speech understanding tasks. GRPO has recently gained traction for its efficiency in training LLMs, and prior work has explored its application to SALLMs, primarily in multiple-choice tasks. Building on this, we focus on open-format tasks that better reflect the generative abilities of the models. Our approach leverages GRPO with BLEU as the reward signal to optimize SALLMs, and we demonstrate empirically that it surpasses standard SFT across several key metrics. Finally, we explore the potential of incorporating off-policy samples within GRPO for these tasks, highlighting avenues for further improvement and further research.
>
---
#### [new 034] Reverse Attention for Lightweight Speech Enhancement on Edge Devices
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出一种轻量级语音增强模型，用于边缘设备的实时处理。基于U-Net架构，引入软注意力门控机制，在保证性能的同时提升效率。实验表明，其在语音质量和可懂度指标上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.16705v1](http://arxiv.org/pdf/2509.16705v1)**

> **作者:** Shuubham Ojha; Felix Gervits; Carol Espy-Wilson
>
> **摘要:** This paper introduces a lightweight deep learning model for real-time speech enhancement, designed to operate efficiently on resource-constrained devices. The proposed model leverages a compact architecture that facilitates rapid inference without compromising performance. Key contributions include infusing soft attention-based attention gates in the U-Net architecture which is known to perform well for segmentation tasks and is optimized for GPUs. Experimental evaluations demonstrate that the model achieves competitive speech quality and intelligibility metrics, such as PESQ and Word Error Rates (WER), improving the performance of similarly sized baseline models. We are able to achieve a 6.24% WER improvement and a 0.64 PESQ score improvement over un-enhanced waveforms.
>
---
#### [new 035] An Octave-based Multi-Resolution CQT Architecture for Diffusion-based Audio Generation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MR-CQTdiff，一种基于多分辨率CQT的扩散音频生成架构。针对低频时间分辨率低的问题，设计了可逆且灵活的CQT框架，实现更高质量的音频生成，并在FAD指标上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2509.16603v1](http://arxiv.org/pdf/2509.16603v1)**

> **作者:** Maurício do V. M. da Costa; Eloi Moliner
>
> **备注:** accepted at IEEE International Symposium on the Internet of Sounds
>
> **摘要:** This paper introduces MR-CQTdiff, a novel neural-network architecture for diffusion-based audio generation that leverages a multi-resolution Constant-$Q$ Transform (C$Q$T). The proposed architecture employs an efficient, invertible CQT framework that adjusts the time-frequency resolution on an octave-by-octave basis. This design addresses the issue of low temporal resolution at lower frequencies, enabling more flexible and expressive audio generation. We conduct an evaluation using the Fr\'echet Audio Distance (FAD) metric across various architectures and two datasets. Experimental results demonstrate that MR-CQTdiff achieves state-of-the-art audio quality, outperforming competing architectures.
>
---
#### [new 036] WenetSpeech-Chuan: A Large-Scale Sichuanese Corpus with Rich Annotation for Dialectal Speech Processing
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出WenetSpeech-Chuan，一个1万小时的带丰富标注的四川话语料库，旨在解决方言语音数据稀缺问题。通过Chuan-Pipeline框架构建，并发布ASR和TTS基准测试集，推动方言语音处理研究与公平性。**

- **链接: [http://arxiv.org/pdf/2509.18004v1](http://arxiv.org/pdf/2509.18004v1)**

> **作者:** Yuhang Dai; Ziyu Zhang; Shuai Wang; Longhao Li; Zhao Guo; Tianlun Zuo; Shuiyuan Wang; Hongfei Xue; Chengyou Wang; Qing Wang; Xin Xu; Hui Bu; Jie Li; Jian Kang; Binbin Zhang; Lei Xie
>
> **备注:** 4 pages, 5 figures, 4 tables
>
> **摘要:** The scarcity of large-scale, open-source data for dialects severely hinders progress in speech technology, a challenge particularly acute for the widely spoken Sichuanese dialects of Chinese. To address this critical gap, we introduce WenetSpeech-Chuan, a 10,000-hour, richly annotated corpus constructed using our novel Chuan-Pipeline, a complete data processing framework for dialectal speech. To facilitate rigorous evaluation and demonstrate the corpus's effectiveness, we also release high-quality ASR and TTS benchmarks, WenetSpeech-Chuan-Eval, with manually verified transcriptions. Experiments show that models trained on WenetSpeech-Chuan achieve state-of-the-art performance among open-source systems and demonstrate results comparable to commercial services. As the largest open-source corpus for Sichuanese dialects, WenetSpeech-Chuan not only lowers the barrier to research in dialectal speech processing but also plays a crucial role in promoting AI equity and mitigating bias in speech technologies. The corpus, benchmarks, models, and receipts are publicly available on our project page.
>
---
#### [new 037] FUN-SSL: Full-band Layer Followed by U-Net with Narrow-band Layers for Multiple Moving Sound Source Localization
- **分类: eess.AS; eess.SP**

- **简介: 该论文针对多移动声源定位任务，提出FUN-SSL模型。通过引入U-Net结构进行窄带多分辨率处理，替代原有计算复杂的LSTM模块，降低了计算量并提升了性能。**

- **链接: [http://arxiv.org/pdf/2509.17490v1](http://arxiv.org/pdf/2509.17490v1)**

> **作者:** Yuseon Choi; Hyeonseung Kim; Jewoo Jun; Jong Won Shin
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Dual-path processing along the temporal and spectral dimensions has shown to be effective in various speech processing applications. While the sound source localization (SSL) models utilizing dual-path processing such as the FN-SSL and IPDnet demonstrated impressive performances in localizing multiple moving sources, they require significant amount of computation. In this paper, we propose an architecture for SSL which introduces a U-Net to perform narrow-band processing in multiple resolutions to reduce computational complexity. The proposed model replaces the full-narrow network block in the IPDnet consisting of one full-band LSTM layer along the spectral dimension followed by one narrow-band LSTM layer along the temporal dimension with the FUN block composed of one Full-band layer followed by a U-net with Narrow-band layers in multiple scales. On top of the skip connections within each U-Net, we also introduce the skip connections between FUN blocks to enrich information. Experimental results showed that the proposed FUN-SSL outperformed previously proposed approaches with computational complexity much lower than that of the IPDnet.
>
---
#### [new 038] RADE for Land Mobile Radio: A Neural Codec for Transmission of Speech over Baseband FM Radio Channels
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出RADE，一种基于神经网络的语音编解码器，用于在陆地移动无线电（LMR）系统中通过基带FM信道传输高质量语音。旨在提升BBFM架构下的语音质量，解决传统FM在衰落环境下的性能问题。**

- **链接: [http://arxiv.org/pdf/2509.17286v1](http://arxiv.org/pdf/2509.17286v1)**

> **作者:** David Rowe; Tibor Bece
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** In the 1990s Land Mobile Radio (LMR) systems evolved from analog frequency modulation (FM) to standardised digital systems. Both digital and analog FM systems now co-exist in various services and exhibit similar speech quality. The architecture of many digital radios retains the analog FM modulator and demodulator from legacy analog radios, but driven by a multi-level digital pulse train rather than an analog voice signal. We denote this architecture baseband FM (BBFM). In this paper we describe a modern machine learning approach that uses an autoencoder to send high quality, 8 kHz bandwidth speech over the BBFM channel. The speech quality is shown to be superior to analog FM over simulated LMR channels in the presence of fading, and a demonstration of the system running over commodity UHF radios is presented.
>
---
#### [new 039] VAInpaint: Zero-Shot Video-Audio inpainting framework with LLMs-driven Module
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出VAInpaint，一种基于LLM的视频-音频补全框架，旨在解决从视频中精准移除对象及其对应音频的问题。通过分割模型生成掩码，并结合视频补全与文本驱动的音频分离模型实现零样本音视频修复。**

- **链接: [http://arxiv.org/pdf/2509.17022v1](http://arxiv.org/pdf/2509.17022v1)**

> **作者:** Kam Man Wu; Zeyue Tian; Liya Ji; Qifeng Chen
>
> **摘要:** Video and audio inpainting for mixed audio-visual content has become a crucial task in multimedia editing recently. However, precisely removing an object and its corresponding audio from a video without affecting the rest of the scene remains a significant challenge. To address this, we propose VAInpaint, a novel pipeline that first utilizes a segmentation model to generate masks and guide a video inpainting model in removing objects. At the same time, an LLM then analyzes the scene globally, while a region-specific model provides localized descriptions. Both the overall and regional descriptions will be inputted into an LLM, which will refine the content and turn it into text queries for our text-driven audio separation model. Our audio separation model is fine-tuned on a customized dataset comprising segmented MUSIC instrument images and VGGSound backgrounds to enhance its generalization performance. Experiments show that our method achieves performance comparable to current benchmarks in both audio and video inpainting.
>
---
#### [new 040] TF-CorrNet: Leveraging Spatial Correlation for Continuous Speech Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出TF-CorrNet，用于连续语音分离任务。针对多通道语音分离中空间信息利用不足的问题，通过引入麦克风间的相关性（PHAT-beta）和双路径时空建模策略，提升语音分离效果。实验表明其在LibriCSS数据集上性能优越且计算成本低。**

- **链接: [http://arxiv.org/pdf/2509.16481v1](http://arxiv.org/pdf/2509.16481v1)**

> **作者:** Ui-Hyeop Shin; Bon Hyeok Ku; Hyung-Min Park
>
> **备注:** Accepted in SPL
>
> **摘要:** In general, multi-channel source separation has utilized inter-microphone phase differences (IPDs) concatenated with magnitude information in time-frequency domain, or real and imaginary components stacked along the channel axis. However, the spatial information of a sound source is fundamentally contained in the differences between microphones, specifically in the correlation between them, while the power of each microphone also provides valuable information about the source spectrum, which is why the magnitude is also included. Therefore, we propose a network that directly leverages a correlation input with phase transform (PHAT)-beta to estimate the separation filter. In addition, the proposed TF-CorrNet processes the features alternately across time and frequency axes as a dual-path strategy in terms of spatial information. Furthermore, we add a spectral module to model source-related direct time-frequency patterns for improved speech separation. Experimental results demonstrate that the proposed TF-CorrNet effectively separates the speech sounds, showing high performance with a low computational cost in the LibriCSS dataset.
>
---
#### [new 041] Automotive Sound Quality for EVs: Psychoacoustic Metrics with Reproducible AI/ML Baselines
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究电动汽车（EV）的声音质量，结合标准化心理声学指标与轻量级AI/ML模型，提供可复现的基线方法，解决EV声音特征建模问题，工作包括实现心理声学指标和构建可复用的机器学习模型。**

- **链接: [http://arxiv.org/pdf/2509.16901v1](http://arxiv.org/pdf/2509.16901v1)**

> **作者:** Mandip Goswami
>
> **摘要:** We present an open, reproducible reference for automotive sound quality that connects standardized psychoacoustic metrics with lightweight AI/ML baselines, with a specific focus on electric vehicles (EVs). We implement loudness (ISO 532-1/2), tonality (DIN 45681), and modulation-based descriptors (roughness, fluctuation strength), and document assumptions and parameterizations for reliable reuse. For modeling, we provide simple, fully reproducible baselines (logistic regression, random forest, SVM) on synthetic EV-like cases using fixed splits and seeds, reporting accuracy and rank correlations as examples of end-to-end workflows rather than a comparative benchmark. Program-level normalization is reported in LUFS via ITU-R BS.1770, while psychoacoustic analysis uses ISO-532 loudness (sones). All figures and tables are regenerated by scripts with pinned environments; code and minimal audio stimuli are released under permissive licenses to support teaching, replication, and extension to EV-specific noise phenomena (e.g., inverter whine, reduced masking).
>
---
#### [new 042] Sound field estimation with moving microphones using kernel ridge regression
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出一种基于核岭回归（KRR）的移动麦克风声场估计方法，结合离散傅里叶变换和赫格洛茨波函数建模，并引入正则化先验知识。为降低计算成本，还提出了使用随机傅里叶特征的近似方法。**

- **链接: [http://arxiv.org/pdf/2509.16358v1](http://arxiv.org/pdf/2509.16358v1)**

> **作者:** Jesper Brunnström; Martin Bo Møller; Jan Østergaard; Shoichi Koyama; Toon van Waterschoot; Marc Moonen
>
> **摘要:** Sound field estimation with moving microphones can increase flexibility, decrease measurement time, and reduce equipment constraints compared to using stationary microphones. In this paper a sound field estimation method based on kernel ridge regression (KRR) is proposed for moving microphones. The proposed KRR method is constructed using a discrete time continuous space sound field model based on the discrete Fourier transform and the Herglotz wave function. The proposed method allows for the inclusion of prior knowledge as a regularization penalty, similar to kernel-based methods with stationary microphones, which is novel for moving microphones. Using a directional weighting for the proposed method, the sound field estimates are improved, which is demonstrated on both simulated and real data. Due to the high computational cost of sound field estimation with moving microphones, an approximate KRR method is proposed, using random Fourier features (RFF) to approximate the kernel. The RFF method is shown to decrease computational cost while obtaining less accurate estimates compared to KRR, providing a trade-off between cost and performance.
>
---
#### [new 043] AuditoryBench++: Can Language Models Understand Auditory Knowledge without Hearing?
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出AuditoryBench++，一个用于评估语言模型在文本环境下理解听觉知识的基准，旨在解决模型缺乏听觉常识推理能力的问题，并引入AIR-CoT方法提升推理效果。**

- **链接: [http://arxiv.org/pdf/2509.17641v1](http://arxiv.org/pdf/2509.17641v1)**

> **作者:** Hyunjong Ok; Suho Yoo; Hyeonjun Kim; Jaeho Lee
>
> **备注:** Preprint
>
> **摘要:** Even without directly hearing sounds, humans can effortlessly reason about auditory properties, such as pitch, loudness, or sound-source associations, drawing on auditory commonsense. In contrast, language models often lack this capability, limiting their effectiveness in multimodal interactions. As an initial step to address this gap, we present AuditoryBench++, a comprehensive benchmark for evaluating auditory knowledge and reasoning in text-only settings. The benchmark encompasses tasks that range from basic auditory comparisons to contextually grounded reasoning, enabling fine-grained analysis of how models process and integrate auditory concepts. In addition, we introduce AIR-CoT, a novel auditory imagination reasoning method that generates and integrates auditory information during inference through span detection with special tokens and knowledge injection. Extensive experiments with recent LLMs and Multimodal LLMs demonstrate that AIR-CoT generally outperforms both the off-the-shelf models and those augmented with auditory knowledge. The project page is available at https://auditorybenchpp.github.io.
>
---
#### [new 044] SongPrep: A Preprocessing Framework and End-to-end Model for Full-song Structure Parsing and Lyrics Transcription
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出SongPrep和SongPrepE2E，用于自动处理歌曲数据的结构解析与歌词转录任务。旨在解决歌曲生成模型训练数据准备耗时费力的问题，通过端到端模型实现高效、精准的数据预处理，提升生成歌曲质量。**

- **链接: [http://arxiv.org/pdf/2509.17404v1](http://arxiv.org/pdf/2509.17404v1)**

> **作者:** Wei Tan; Shun Lei; Huaicheng Zhang; Guangzheng Li; Yixuan Zhang; Hangting Chen; Jianwei Yu; Rongzhi Gu; Dong Yu
>
> **摘要:** Artificial Intelligence Generated Content (AIGC) is currently a popular research area. Among its various branches, song generation has attracted growing interest. Despite the abundance of available songs, effective data preparation remains a significant challenge. Converting these songs into training-ready datasets typically requires extensive manual labeling, which is both time consuming and costly. To address this issue, we propose SongPrep, an automated preprocessing pipeline designed specifically for song data. This framework streamlines key processes such as source separation, structure analysis, and lyric recognition, producing structured data that can be directly used to train song generation models. Furthermore, we introduce SongPrepE2E, an end-to-end structured lyrics recognition model based on pretrained language models. Without the need for additional source separation, SongPrepE2E is able to analyze the structure and lyrics of entire songs and provide precise timestamps. By leveraging context from the whole song alongside pretrained semantic knowledge, SongPrepE2E achieves low Diarization Error Rate (DER) and Word Error Rate (WER) on the proposed SSLD-200 dataset. Downstream tasks demonstrate that training song generation models with the data output by SongPrepE2E enables the generated songs to closely resemble those produced by humans.
>
---
#### [new 045] Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声音景观情感识别（SER）任务，旨在解决特征选择问题。作者结合图学习与信息准则，提出一种基于图拓扑推断的特征选择框架，并引入广义肘部检测器确定稀疏度，揭示了特征与情感输出的关系，挑战了传统假设。**

- **链接: [http://arxiv.org/pdf/2509.16760v1](http://arxiv.org/pdf/2509.16760v1)**

> **作者:** Samuel Rey; Luca Martino; Roberto San Millan; Eduardo Morgado
>
> **摘要:** Research on soundscapes has shifted the focus of environmental acoustics from noise levels to the perception of sounds, incorporating contextual factors. Soundscape emotion recognition (SER) models perception using a set of features, with arousal and valence commonly regarded as sufficient descriptors of affect. In this work, we blend \emph{graph learning} techniques with a novel \emph{information criterion} to develop a feature selection framework for SER. Specifically, we estimate a sparse graph representation of feature relations using linear structural equation models (SEM) tailored to the widely used Emo-Soundscapes dataset. The resulting graph captures the relations between input features and the two emotional outputs. To determine the appropriate level of sparsity, we propose a novel \emph{generalized elbow detector}, which provides both a point estimate and an uncertainty interval. We conduct an extensive evaluation of our methods, including visualizations of the inferred relations. While several of our findings align with previous studies, the graph representation also reveals a strong connection between arousal and valence, challenging common SER assumptions.
>
---
#### [new 046] Similarity-Guided Diffusion for Long-Gap Music Inpainting
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究音乐修复任务，旨在解决长时隙音频缺失的重建问题。提出SimDPS方法，结合扩散模型与相似性检索，提升多秒级缺失片段的可听性与连贯性。**

- **链接: [http://arxiv.org/pdf/2509.16342v1](http://arxiv.org/pdf/2509.16342v1)**

> **作者:** Sean Turland; Eloi Moliner; Vesa Välimäki
>
> **备注:** 5 pages, 2 figures. Submitted to IEEE ICASSP 2026. Audio examples and supplementary material are available at: https://s-turland.github.io/SimDPS/
>
> **摘要:** Music inpainting aims to reconstruct missing segments of a corrupted recording. While diffusion-based generative models improve reconstruction for medium-length gaps, they often struggle to preserve musical plausibility over multi-second gaps. We introduce Similarity-Guided Diffusion Posterior Sampling (SimDPS), a hybrid method that combines diffusion-based inference with similarity search. Candidate segments are first retrieved from a corpus based on contextual similarity, then incorporated into a modified likelihood that guides the diffusion process toward contextually consistent reconstructions. Subjective evaluation on piano music inpainting with 2-s gaps shows that the proposed SimDPS method enhances perceptual plausibility compared to unguided diffusion and frequently outperforms similarity search alone when moderately similar candidates are available. These results demonstrate the potential of a hybrid similarity approach for diffusion-based audio enhancement with long gaps.
>
---
#### [new 047] DeepASA: An Object-Oriented One-for-All Network for Auditory Scene Analysis
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出DeepASA，一种面向听觉场景分析的统一模型，可同时完成声源分离、去混响、事件检测等多任务。通过面向对象处理和时序一致性匹配机制，解决了传统方法中的参数关联模糊问题，在复杂空间音频场景中表现出色。**

- **链接: [http://arxiv.org/pdf/2509.17247v1](http://arxiv.org/pdf/2509.17247v1)**

> **作者:** Dongheon Lee; Younghoo Kwon; Jung-Woo Choi
>
> **备注:** 26 pages, 13 figures, 8 tables, accepted in NeurIPS 2025
>
> **摘要:** We propose DeepASA, a one-for-all model for auditory scene analysis that performs multi-input multi-output (MIMO) source separation, dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a unified framework. DeepASA is designed for complex auditory scenes where multiple, often similar, sound sources overlap in time and move dynamically in space. To achieve robust and consistent inference across tasks, we introduce an object-oriented processing (OOP) strategy. This approach encapsulates diverse auditory features into object-centric representations and refines them through a chain-of-inference (CoI) mechanism. The pipeline comprises a dynamic temporal kernel-based feature extractor, a transformer-based aggregator, and an object separator that yields per-object features. These features feed into multiple task-specific decoders. Our object-centric representations naturally resolve the parameter association ambiguity inherent in traditional track-wise processing. However, early-stage object separation can lead to failure in downstream ASA tasks. To address this, we implement temporal coherence matching (TCM) within the chain-of-inference, enabling multi-task fusion and iterative refinement of object features using estimated auditory parameters. We evaluate DeepASA on representative spatial audio benchmark datasets, including ASA2, MC-FUSS, and STARSS23. Experimental results show that our model achieves state-of-the-art performance across all evaluated tasks, demonstrating its effectiveness in both source separation and auditory parameter estimation under diverse spatial auditory scenes.
>
---
#### [new 048] BeepBank-500: A Synthetic Earcon Mini-Corpus for UI Sound Research and Psychoacoustics Research
- **分类: eess.AS; cs.SD**

- **简介: 论文提出BeepBank-500，一个合成耳音数据集，用于人机交互与音频机器学习研究。通过参数化生成波形、频率、调制等特征，并提供元数据和基线任务，解决实验素材不足问题，支持耳音分类、音色分析等任务。**

- **链接: [http://arxiv.org/pdf/2509.17277v1](http://arxiv.org/pdf/2509.17277v1)**

> **作者:** Mandip Goswami
>
> **备注:** Data note; 6 to 8 pages; 1 to 2 figures; dataset: CC0-1.0; code: MIT
>
> **摘要:** We introduce BeepBank-500, a compact, fully synthetic earcon/alert dataset (300-500 clips) designed for rapid, rights-clean experimentation in human-computer interaction and audio machine learning. Each clip is generated from a parametric recipe controlling waveform family (sine, square, triangle, FM), fundamental frequency, duration, amplitude envelope, amplitude modulation (AM), and lightweight Schroeder-style reverberation. We use three reverberation settings: dry, and two synthetic rooms denoted 'rir small' ('small') and 'rir medium' ('medium') throughout the paper and in the metadata. We release mono 48 kHz WAV audio (16-bit), a rich metadata table (signal/spectral features), and tiny reproducible baselines for (i) waveform-family classification and (ii) f0 regression on single tones. The corpus targets tasks such as earcon classification, timbre analyses, and onset detection, with clearly stated licensing and limitations. Audio is dedicated to the public domain via CC0-1.0; code is under MIT. Data DOI: https://doi.org/10.5281/zenodo.17172015. Code: https://github.com/mandip42/earcons-mini-500.
>
---
#### [new 049] Audio-Guided Dynamic Modality Fusion with Stereo-Aware Attention for Audio-Visual Navigation
- **分类: cs.AI; cs.SD**

- **简介: 该论文针对音频-视觉导航任务，旨在解决复杂环境中声源定位性能下降的问题。提出SAM模块利用立体声音空间信息，AGDF模块动态融合视听特征，提升导航鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2509.16924v1](http://arxiv.org/pdf/2509.16924v1)**

> **作者:** Jia Li; Yinfeng Yu; Liejun Wang; Fuchun Sun; Wendong Zheng
>
> **备注:** Main paper (14 pages). Accepted for publication by ICONIP( International Conference on Neural Information Processing) 2025
>
> **摘要:** In audio-visual navigation (AVN) tasks, an embodied agent must autonomously localize a sound source in unknown and complex 3D environments based on audio-visual signals. Existing methods often rely on static modality fusion strategies and neglect the spatial cues embedded in stereo audio, leading to performance degradation in cluttered or occluded scenes. To address these issues, we propose an end-to-end reinforcement learning-based AVN framework with two key innovations: (1) a \textbf{S}tereo-Aware \textbf{A}ttention \textbf{M}odule (\textbf{SAM}), which learns and exploits the spatial disparity between left and right audio channels to enhance directional sound perception; and (2) an \textbf{A}udio-\textbf{G}uided \textbf{D}ynamic \textbf{F}usion Module (\textbf{AGDF}), which dynamically adjusts the fusion ratio between visual and auditory features based on audio cues, thereby improving robustness to environmental changes. Extensive experiments are conducted on two realistic 3D scene datasets, Replica and Matterport3D, demonstrating that our method significantly outperforms existing approaches in terms of navigation success rate and path efficiency. Notably, our model achieves over 40\% improvement under audio-only conditions compared to the best-performing baselines. These results highlight the importance of explicitly modeling spatial cues from stereo channels and performing deep multi-modal fusion for robust and efficient audio-visual navigation.
>
---
#### [new 050] A Scalable and Interoperable Platform for Transforming Building Information with Brick Ontology
- **分类: cs.CY; eess.SP**

- **简介: 该论文提出一个基于Brick本体的平台，用于建筑信息的转换与管理。任务是解决建筑自动化中的可扩展性和互操作性问题。工作包括构建半自动化的平台，利用图结构和Brick本体实现数据转换与数字孪生应用，提升建筑信息管理效率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.16259v1](http://arxiv.org/pdf/2509.16259v1)**

> **作者:** Rozita Teymourzadeh; Yuya Nakazawa
>
> **摘要:** In the digital twin and building information era, many building automation companies searched for scalable methods to extract and analyze different building data, including Internet of Things (IoT) sensors, actuators, layout sections, zones, etc. The necessity for engineers to continuously manage the entire process for each new building creates scalability challenges. Furthermore, because construction information is sensitive, transferring data on vendor platforms via the cloud creates problems. This paper introduces a platform designed to address some of the common challenges in building automation. This is a smart platform designed for the transformation of building information into Brick ontology (Brick 2020) and graph formats. This technology makes it easy to retrieve historical data and converts the building point list into a Brick schema model for use in digital twin applications. The overarching goal of the proposed platform development is semi-automate the process while offering adaptability to various building configurations. This platform uses Brick schema and graph data structure techniques to minimize complexity, offering a semi-automated approach through its use of a tree-based graph structure. Moreover, the integration of Brick ontology creates a common language for interoperability and improves building information management. The seamless and offline integration of historical data within the developed platform minimizes data security risks when handling building information.
>
---
#### [new 051] Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文研究了基于扩散的大型语言模型（LLaDA）在语音识别（ASR）中的应用，旨在提升识别准确率。通过将LLaDA作为Whisper-LLaMA的外部处理模块，并探索不同策略，显著降低了词错误率（WER），验证了音频条件嵌入的重要性。**

- **链接: [http://arxiv.org/pdf/2509.16622v1](http://arxiv.org/pdf/2509.16622v1)**

> **作者:** Mengqi Wang; Zhan Liu; Zengrui Jin; Guangzhi Sun; Chao Zhang; Philip C. Woodland
>
> **摘要:** Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements.
>
---
#### [new 052] Does Audio Matter for Modern Video-LLMs and Their Benchmarks?
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究视频大模型中音频的作用，指出当前多数评估忽略音频。作者分析音频对视频理解的影响，提出轻量压缩方法，并构建了含音频的评测集，推动真实场景下的音视频联合建模。**

- **链接: [http://arxiv.org/pdf/2509.17901v1](http://arxiv.org/pdf/2509.17901v1)**

> **作者:** Geewook Kim; Minjoon Seo
>
> **备注:** 5 pages, 2 figures, under review. Project page: https://github.com/naver-ai/LLaVA-AV-SSM
>
> **摘要:** Modern multimodal large language models often claim "video understanding," yet most evaluations use muted videos or simply discard audio. We ask a direct question: how much does audio actually matter for contemporary Video-LLMs and the benchmarks that certify them? We audit widely used suites and observe that many items are even solvable from a single frame, rendering audio largely redundant. Building on LLaVA-OneVision architecture, we attach a speech/audio encoder (e.g., Whisper) and analyze when audio helps, while addressing audio token explosion with a lightweight Mamba-based state-space token compressor. We find that audio yields minimal gains on recent video benchmarks but is decisive on curated, audio-sensitive subsets. To enable faithful evaluation, we release AVQA-Hard and Music-AVQA-Hard, our model, and code. Our findings surface a growing gap between current academic practice and real-world expectations, and provide practical tools for scalable audio-visual Video-LLMs. We will fully open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.
>
---
#### [new 053] Cross-Attention is Half Explanation in Speech-to-Text Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音到文本（S2T）任务中交叉注意力的解释能力。针对其是否能反映输入相关性的假设，作者通过与特征归因方法对比，发现交叉注意力仅能解释约50%的输入重要性，揭示了其作为解释工具的局限性。**

- **链接: [http://arxiv.org/pdf/2509.18010v1](http://arxiv.org/pdf/2509.18010v1)**

> **作者:** Sara Papi; Dennis Fucci; Marco Gaido; Matteo Negri; Luisa Bentivogli
>
> **摘要:** Cross-attention is a core mechanism in encoder-decoder architectures, widespread in many fields, including speech-to-text (S2T) processing. Its scores have been repurposed for various downstream applications--such as timestamp estimation and audio-text alignment--under the assumption that they reflect the dependencies between input speech representation and the generated text. While the explanatory nature of attention mechanisms has been widely debated in the broader NLP literature, this assumption remains largely unexplored within the speech domain. To address this gap, we assess the explanatory power of cross-attention in S2T models by comparing its scores to input saliency maps derived from feature attribution. Our analysis spans monolingual and multilingual, single-task and multi-task models at multiple scales, and shows that attention scores moderately to strongly align with saliency-based explanations, particularly when aggregated across heads and layers. However, it also shows that cross-attention captures only about 50% of the input relevance and, in the best case, only partially reflects how the decoder attends to the encoder's representations--accounting for just 52-75% of the saliency. These findings uncover fundamental limitations in interpreting cross-attention as an explanatory proxy, suggesting that it offers an informative yet incomplete view of the factors driving predictions in S2T models.
>
---
#### [new 054] The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦于语言模型在言语病理学中的应用，旨在解决临床资源不足的问题。研究构建了首个综合基准测试，涵盖5类核心任务，并通过领域数据微调提升模型性能30%以上，揭示了现有模型的潜力与局限性。**

- **链接: [http://arxiv.org/pdf/2509.16765v1](http://arxiv.org/pdf/2509.16765v1)**

> **作者:** Fagun Patel; Duc Q. Nguyen; Sang T. Truong; Jody Vaynshtok; Sanmi Koyejo; Nick Haber
>
> **备注:** EMNLP 2025 Oral Presentation
>
> **摘要:** According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 30% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.
>
---
#### [new 055] RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds
- **分类: cs.RO; eess.SP**

- **简介: 该论文提出RadarSFD，一种基于预训练先验的单帧扩散模型，用于从毫米波雷达点云重建高密度LiDAR-like点云。解决了小尺寸机器人系统中多帧依赖的问题，实现了无运动、无SAR的高精度点云感知。**

- **链接: [http://arxiv.org/pdf/2509.18068v1](http://arxiv.org/pdf/2509.18068v1)**

> **作者:** Bin Zhao; Nakul Garg
>
> **摘要:** Millimeter-wave radar provides perception robust to fog, smoke, dust, and low light, making it attractive for size, weight, and power constrained robotic platforms. Current radar imaging methods, however, rely on synthetic aperture or multi-frame aggregation to improve resolution, which is impractical for small aerial, inspection, or wearable systems. We present RadarSFD, a conditional latent diffusion framework that reconstructs dense LiDAR-like point clouds from a single radar frame without motion or SAR. Our approach transfers geometric priors from a pretrained monocular depth estimator into the diffusion backbone, anchors them to radar inputs via channel-wise latent concatenation, and regularizes outputs with a dual-space objective combining latent and pixel-space losses. On the RadarHD benchmark, RadarSFD achieves 35 cm Chamfer Distance and 28 cm Modified Hausdorff Distance, improving over the single-frame RadarHD baseline (56 cm, 45 cm) and remaining competitive with multi-frame methods using 5-41 frames. Qualitative results show recovery of fine walls and narrow gaps, and experiments across new environments confirm strong generalization. Ablation studies highlight the importance of pretrained initialization, radar BEV conditioning, and the dual-space loss. Together, these results establish the first practical single-frame, no-SAR mmWave radar pipeline for dense point cloud perception in compact robotic systems.
>
---
#### [new 056] DroFiT: A Lightweight Band-fused Frequency Attention Toward Real-time UAV Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出DroFiT，一种轻量级无人机语音增强网络，用于在强自噪声环境下提升单麦克风语音质量。通过融合频域Transformer与混合编解码器及TCN后端，实现高效实时处理，降低计算与内存需求。**

- **链接: [http://arxiv.org/pdf/2509.16945v1](http://arxiv.org/pdf/2509.16945v1)**

> **作者:** Jeongmin Lee; Chanhong Jeon; Hyungjoo Seo; Taewook Kang
>
> **摘要:** This paper proposes DroFiT (Drone Frequency lightweight Transformer for speech enhancement, a single microphone speech enhancement network for severe drone self-noise. DroFit integrates a frequency-wise Transformer with a full/sub-band hybrid encoder-decoder and a TCN back-end for memory-efficient streaming. A learnable skip-and-gate fusion with a combined spectral-temporal loss further refines reconstruction. The model is trained on VoiceBank-DEMAND mixed with recorded drone noise (-5 to -25 dB SNR) and evaluate using standard speech enhancement metrics and computational efficiency. Experimental results show that DroFiT achieves competitive enhancement performance while significantly reducing computational and memory demands, paving the way for real-time processing on resource-constrained UAV platforms. Audio demo samples are available on our demo page.
>
---
#### [new 057] Investigating Polyglot Speech Foundation Models for Learning Collective Emotion from Crowds
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究多语言语音基础模型在人群情绪识别任务中的应用，旨在解决嘈杂复杂环境下短时音频情绪识别的问题。通过对比实验验证多语言模型在多种音频长度下的优越性能，为该任务建立新基准。**

- **链接: [http://arxiv.org/pdf/2509.16329v1](http://arxiv.org/pdf/2509.16329v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Panchal Nayak; Priyabrata Mallick; Swarup Ranjan Behera; Parabattina Bhagath; Pailla Balakrishna Reddy; Arun Balaji Buduru
>
> **备注:** Accepted to APSIPA-ASC 2025
>
> **摘要:** This paper investigates the polyglot (multilingual) speech foundation models (SFMs) for Crowd Emotion Recognition (CER). We hypothesize that polyglot SFMs, pre-trained on diverse languages, accents, and speech patterns, are particularly adept at navigating the noisy and complex acoustic environments characteristic of crowd settings, thereby offering a significant advantage for CER. To substantiate this, we perform a comprehensive analysis, comparing polyglot, monolingual, and speaker recognition SFMs through extensive experiments on a benchmark CER dataset across varying audio durations (1 sec, 500 ms, and 250 ms). The results consistently demonstrate the superiority of polyglot SFMs, outperforming their counterparts across all audio lengths and excelling even with extremely short-duration inputs. These findings pave the way for adaptation of SFMs in setting up new benchmarks for CER.
>
---
## 更新

#### [replaced 001] EMO-RL: Emotion-Rule-Based Reinforcement Learning Enhanced Audio-Language Model for Generalized Speech Emotion Recognition
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15654v2](http://arxiv.org/pdf/2509.15654v2)**

> **作者:** Pengcheng Li; Botao Zhao; Zuheng Kang; Junqing Peng; Xiaoyang Qu; Yayun He; Jianzong Wang
>
> **备注:** Accepted by the Findings of 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP Findings 2025)
>
> **摘要:** Although Large Audio-Language Models (LALMs) have exhibited outstanding performance in auditory understanding, their performance in affective computing scenarios, particularly in emotion recognition, reasoning, and subtle sentiment differentiation, remains suboptimal. Recent advances in Reinforcement Learning (RL) have shown promise in improving LALMs' reasoning abilities. However, two critical challenges hinder the direct application of RL techniques to Speech Emotion Recognition (SER) tasks: (1) convergence instability caused by ambiguous emotional boundaries and (2) limited reasoning ability when using relatively small models (e.g., 7B-parameter architectures). To overcome these limitations, we introduce EMO-RL, a novel framework incorporating reinforcement learning with two key innovations: Emotion Similarity-Weighted Reward (ESWR) and Explicit Structured Reasoning (ESR). Built upon pretrained LALMs, our method employs group-relative policy optimization with emotion constraints. Comprehensive experiments demonstrate that our EMO-RL training strategies can significantly enhance the emotional reasoning capabilities of LALMs, attaining state-of-the-art results on both the MELD and IEMOCAP datasets, and cross-dataset experiments prove the strong superiority of generalization.
>
---
#### [replaced 002] Speech Recognition on TV Series with Video-guided Post-ASR Correction
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07323v2](http://arxiv.org/pdf/2506.07323v2)**

> **作者:** Haoyuan Yang; Yue Zhang; Liqiang Jing; John H. L. Hansen
>
> **摘要:** Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where multiple speakers, overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing approaches fail to explicitly leverage the rich temporal and contextual information available in the video. To address this limitation, we propose a Video-Guided Post-ASR Correction (VPC) framework that uses a Video-Large Multimodal Model (VLMM) to capture video context and refine ASR outputs. Evaluations on a TV-series benchmark show that our method consistently improves transcription accuracy in complex multimedia environments.
>
---
#### [replaced 003] TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15666v2](http://arxiv.org/pdf/2509.15666v2)**

> **作者:** Yongsheng Feng; Yuetonghui Xu; Jiehui Luo; Hongjia Liu; Xiaobing Li; Feng Yu; Wei Li
>
> **备注:** Submitted to ICASSP 2026.(C) 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work
>
> **摘要:** Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates early-split multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation. Code is available at https://github.com/WingSingFung/TISDiSS.
>
---
#### [replaced 004] Extract and Diffuse: Latent Integration for Improved Diffusion-based Speech and Vocal Enhancement
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.09642v2](http://arxiv.org/pdf/2409.09642v2)**

> **作者:** Yudong Yang; Zhan Liu; Wenyi Yu; Guangzhi Sun; Qiuqiang Kong; Chao Zhang
>
> **备注:** Accepted by NCMMSC 2025
>
> **摘要:** Diffusion-based generative models have recently achieved remarkable results in speech and vocal enhancement due to their ability to model complex speech data distributions. While these models generalize well to unseen acoustic environments, they may not achieve the same level of fidelity as the discriminative models specifically trained to enhance particular acoustic conditions. In this paper, we propose Ex-Diff, a novel score-based diffusion model that integrates the latent representations produced by a discriminative model to improve speech and vocal enhancement, which combines the strengths of both generative and discriminative models. Experimental results on the widely used MUSDB dataset show relative improvements of 3.7% in SI-SDR and 10.0% in SI-SIR compared to the baseline diffusion model for speech and vocal enhancement tasks, respectively. Additionally, case studies are provided to further illustrate and analyze the complementary nature of generative and discriminative models in this context.
>
---
#### [replaced 005] VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09716v2](http://arxiv.org/pdf/2509.09716v2)**

> **作者:** Jun Zhan; Mingyang Han; Yuxuan Xie; Chen Wang; Dong Zhang; Kexin Huang; Haoxiang Shi; DongXiao Wang; Tengtao Song; Qinyuan Cheng; Shimin Li; Jun Song; Xipeng Qiu; Bo Zheng
>
> **摘要:** Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{https://junzhan2000.github.io/VStyle.github.io/}{project's homepage}.
>
---
#### [replaced 006] KNN-MMD: Cross Domain Wireless Sensing via Local Distribution Alignment
- **分类: cs.CV; cs.AI; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.04783v4](http://arxiv.org/pdf/2412.04783v4)**

> **作者:** Zijian Zhao; Zhijie Cai; Tingwei Chen; Xiaoyang Li; Hang Li; Qimei Chen; Guangxu Zhu
>
> **摘要:** Wireless sensing has recently found widespread applications in diverse environments, including homes, offices, and public spaces. By analyzing patterns in channel state information (CSI), it is possible to infer human actions for tasks such as person identification, gesture recognition, and fall detection. However, CSI is highly sensitive to environmental changes, where even minor alterations can significantly distort the CSI patterns. This sensitivity often leads to performance degradation or outright failure when applying wireless sensing models trained in one environment to another. To address this challenge, Domain Alignment (DAL) has been widely adopted for cross-domain classification tasks, as it focuses on aligning the global distributions of the source and target domains in feature space. Despite its popularity, DAL often neglects inter-category relationships, which can lead to misalignment between categories across domains, even when global alignment is achieved. To overcome these limitations, we propose K-Nearest Neighbors Maximum Mean Discrepancy (KNN-MMD), a novel few-shot method for cross-domain wireless sensing. Our approach begins by constructing a help set using KNN from the target domain, enabling local alignment between the source and target domains within each category using MMD. Additionally, we address a key instability issue commonly observed in cross-domain methods, where model performance fluctuates sharply between epochs. Further, most existing methods struggle to determine an optimal stopping point during training due to the absence of labeled data from the target domain. Our method resolves this by excluding the support set from the target domain during training and employing it as a validation set to determine the stopping criterion.The dataset and code are publicly available at https://github.com/RS2002/KNN-MMD .
>
---
#### [replaced 007] PerceiverS: A Multi-Scale Perceiver with Effective Segmentation for Long-Term Expressive Symbolic Music Generation
- **分类: cs.AI; cs.MM; cs.SD; eess.AS; I.2.7; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.08307v3](http://arxiv.org/pdf/2411.08307v3)**

> **作者:** Yungang Yi; Weihua Li; Matthew Kuo; Quan Bai
>
> **摘要:** AI-based music generation has made significant progress in recent years. However, generating symbolic music that is both long-structured and expressive remains a significant challenge. In this paper, we propose PerceiverS (Segmentation and Scale), a novel architecture designed to address this issue by leveraging both Effective Segmentation and Multi-Scale attention mechanisms. Our approach enhances symbolic music generation by simultaneously learning long-term structural dependencies and short-term expressive details. By combining cross-attention and self-attention in a Multi-Scale setting, PerceiverS captures long-range musical structure while preserving performance nuances. The proposed model has been evaluated using the Maestro dataset and has demonstrated improvements in generating coherent and diverse music, characterized by both structural consistency and expressive variation. The project demos and the generated music samples can be accessed through the link: https://perceivers.github.io.
>
---
#### [replaced 008] PoolingVQ: A VQVAE Variant for Reducing Audio Redundancy and Boosting Multi-Modal Fusion in Music Emotion Analysis
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.11976v2](http://arxiv.org/pdf/2509.11976v2)**

> **作者:** Dinghao Zou; Yicheng Gong; Xiaokang Li; Xin Cao; Sunbowen Lee
>
> **摘要:** Multimodal music emotion analysis leverages audio and MIDI modalities to enhance performance. While mainstream approaches focus on complex feature extraction networks, we posit that shortening the length of audio sequence features to mitigate redundancy, especially in contrast to MIDI's compact representation, may effectively boost task performance. To achieve this, we developed PoolingVQ by combining Vector Quantized Variational Autoencoder (VQVAE) with spatial pooling, which directly compresses audio feature sequences through local aggregation to reduce redundancy, then devised a two-stage co-attention approach to fuse audio and MIDI information. Experimental results on the public datasets EMOPIA and VGMIDI demonstrate that our multimodal framework achieves state-of-the-art overall performance, with PoolingVQ yielding some improvement.
>
---
#### [replaced 009] Audio Contrastive-based Fine-tuning: Decoupling Representation Learning and Classification
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.11895v4](http://arxiv.org/pdf/2309.11895v4)**

> **作者:** Yang Wang; Qibin Liang; Chenghao Xiao; Yizhi Li; Noura Al Moubayed; Chenghua Lin
>
> **备注:** This paper has been submitted to ICASSP 2026 and is currently under review
>
> **摘要:** Standard fine-tuning of pre-trained audio models couples representation learning with classifier training, which can obscure the true quality of the learned representations. In this work, we advocate for a disentangled two-stage framework that separates representation refinement from downstream evaluation. First, we employ a "contrastive-tuning" stage to explicitly improve the geometric structure of the model's embedding space. Subsequently, we introduce a dual-probe evaluation protocol to assess the quality of these refined representations from a geometric perspective. This protocol uses a linear probe to measure global linear separability and a k-Nearest Neighbours probe to investigate the local structure of class clusters. Our experiments on a diverse set of audio classification tasks show that our framework provides a better foundation for classification, leading to improved accuracy. Our newly proposed dual-probing framework acts as a powerful analytical lens, demonstrating why contrastive learning is more effective by revealing a superior embedding space. It significantly outperforms vanilla fine-tuning, particularly on single-label datasets with a large number of classes, and also surpasses strong baselines on multi-label tasks using a Jaccard-weighted loss. Our findings demonstrate that decoupling representation refinement from classifier training is a broadly effective strategy for unlocking the full potential of pre-trained audio models. Our code will be publicly available.
>
---
#### [replaced 010] Handling Domain Shifts for Anomalous Sound Detection: A Review of DCASE-Related Work
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.10435v3](http://arxiv.org/pdf/2503.10435v3)**

> **作者:** Kevin Wilkinghoff; Takuya Fujimura; Keisuke Imoto; Jonathan Le Roux; Zheng-Hua Tan; Tomoki Toda
>
> **摘要:** When detecting anomalous sounds in complex environments, one of the main difficulties is that trained models must be sensitive to subtle differences in monitored target signals, while many practical applications also require them to be insensitive to changes in acoustic domains. Examples of such domain shifts include changing the type of microphone or the location of acoustic sensors, which can have a much stronger impact on the acoustic signal than subtle anomalies themselves. Moreover, users typically aim to train a model only on source domain data, which they may have a relatively large collection of, and they hope that such a trained model will be able to generalize well to an unseen target domain by providing only a minimal number of samples to characterize the acoustic signals in that domain. In this work, we review and discuss recent publications focusing on this domain generalization problem for anomalous sound detection in the context of the DCASE challenges on acoustic machine condition monitoring.
>
---
#### [replaced 011] Survey on the Evaluation of Generative Models in Music
- **分类: cs.SD; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05104v4](http://arxiv.org/pdf/2506.05104v4)**

> **作者:** Alexander Lerch; Claire Arthur; Nick Bryan-Kinns; Corey Ford; Qianyi Sun; Ashvala Vinay
>
> **备注:** Accepted paper submitted to ACM CSUR on 12-Sep-2025, original manuscript submitted on 26-Jun-2024
>
> **摘要:** Research on generative systems in music has seen considerable attention and growth in recent years. A variety of attempts have been made to systematically evaluate such systems. We present an interdisciplinary review of the common evaluation targets, methodologies, and metrics for the evaluation of both system output and model use, covering subjective and objective approaches, qualitative and quantitative approaches, as well as empirical and computational methods. We examine the benefits and limitations of these approaches from a musicological, an engineering, and an HCI perspective.
>
---
#### [replaced 012] Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.02318v2](http://arxiv.org/pdf/2503.02318v2)**

> **作者:** Zhifei Xie; Mingbao Lin; Zihang Liu; Pengcheng Wu; Shuicheng Yan; Chunyan Miao
>
> **备注:** Technical report, in process
>
> **摘要:** Recent advancements in multimodal reasoning have largely overlooked the audio modality. We introduce Audio-Reasoner, a large-scale audio language model for deep reasoning in audio tasks. We meticulously curated a large-scale and diverse multi-task audio dataset with simple annotations. Then, we leverage closed-source models to conduct secondary labeling, QA generation, along with structured COT process. These datasets together form a high-quality reasoning dataset with 1.2 million reasoning-rich samples, which we name CoTA. Following inference scaling principles, we train Audio-Reasoner on CoTA, enabling it to achieve great logical capabilities in audio reasoning. Experiments show state-of-the-art performance across key benchmarks, including MMAU-mini (+25.42%), AIR-Bench chat/foundation(+14.57%/+10.13%), and MELD (+8.01%). Our findings stress the core of structured CoT training in advancing audio reasoning.
>
---
#### [replaced 013] From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.01659v2](http://arxiv.org/pdf/2508.01659v2)**

> **作者:** Yuhang Jia; Xu Zhang; Yujie Guo; Yang Chen; Shiwan Zhao
>
> **摘要:** Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
>
---
#### [replaced 014] Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15952v2](http://arxiv.org/pdf/2509.15952v2)**

> **作者:** Gang Yang; Yue Lei; Wenxin Tai; Jin Wu; Jia Chen; Ting Zhong; Fan Zhou
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP 2026
>
> **摘要:** Diffusion and flow matching (FM) models have achieved remarkable progress in speech enhancement (SE), yet their dependence on multi-step generation is computationally expensive and vulnerable to discretization errors. Recent advances in one-step generative modeling, particularly MeanFlow, provide a promising alternative by reformulating dynamics through average velocity fields. In this work, we present COSE, a one-step FM framework tailored for SE. To address the high training overhead of Jacobian-vector product (JVP) computations in MeanFlow, we introduce a velocity composition identity to compute average velocity efficiently, eliminating expensive computation while preserving theoretical consistency and achieving competitive enhancement quality. Extensive experiments on standard benchmarks show that COSE delivers up to 5x faster sampling and reduces training cost by 40%, all without compromising speech quality. Code is available at https://github.com/ICDM-UESTC/COSE.
>
---
#### [replaced 015] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05983v3](http://arxiv.org/pdf/2509.05983v3)**

> **作者:** Minh N. H. Nguyen; Anh Nguyen Tran; Dung Truong Dinh; Nam Van Vo
>
> **备注:** Update new version
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 19.9% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios
>
---
#### [replaced 016] SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12935v2](http://arxiv.org/pdf/2506.12935v2)**

> **作者:** Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Peijun Qing; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to EMNLP 2025 Main Conference (Oral Presentation)
>
> **摘要:** While large language models have demonstrated impressive reasoning abilities, their extension to the audio modality, particularly within large audio-language models (LALMs), remains underexplored. Addressing this gap requires a systematic approach that involves a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this work, we present a comprehensive solution for audio logical reasoning (ALR) tasks: we introduce SoundMind, a dataset of 6,446 audio-text annotated samples specifically curated to support complex reasoning. Building on this resource, we propose SoundMind-RL, a rule-based reinforcement learning (RL) algorithm designed to equip audio-language models with robust audio-text reasoning capabilities. By fine-tuning Qwen2.5-Omni-7B on the proposed SoundMind dataset using SoundMind-RL, we achieve strong and consistent improvements over state-of-the-art baselines on the SoundMind benchmark. This work highlights the benefit of combining high-quality, reasoning-focused datasets with specialized RL techniques, and contributes to advancing auditory intelligence in language models. The code and dataset introduced in this work are publicly available at https://github.com/xid32/SoundMind.
>
---
#### [replaced 017] An Effective Strategy for Modeling Score Ordinality and Non-uniform Intervals in Automated Speaking Assessment
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.03372v2](http://arxiv.org/pdf/2509.03372v2)**

> **作者:** Tien-Hong Lo; Szu-Yu Chen; Yao-Ting Sung; Berlin Chen
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** A recent line of research on automated speaking assessment (ASA) has benefited from self-supervised learning (SSL) representations, which capture rich acoustic and linguistic patterns in non-native speech without underlying assumptions of feature curation. However, speech-based SSL models capture acoustic-related traits but overlook linguistic content, while text-based SSL models rely on ASR output and fail to encode prosodic nuances. Moreover, most prior arts treat proficiency levels as nominal classes, ignoring their ordinal structure and non-uniform intervals between proficiency labels. To address these limitations, we propose an effective ASA approach combining SSL with handcrafted indicator features via a novel modeling paradigm. We further introduce a multi-margin ordinal loss that jointly models both the score ordinality and non-uniform intervals of proficiency labels. Extensive experiments on the TEEMI corpus show that our method consistently outperforms strong baselines and generalizes well to unseen prompts.
>
---
#### [replaced 018] GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.13093v3](http://arxiv.org/pdf/2509.13093v3)**

> **作者:** Yujie Guo; Jiaming Zhou; Yuhang Jia; Shiwan Zhao; Yong Qin
>
> **摘要:** End-to-end multi-talker automatic speech recognition (MTASR) faces significant challenges in accurately transcribing overlapping speech, especially under high-overlap conditions. To address these challenges, we proposed Global-Local Aware Dynamic (GLAD) Mixture-of-Experts, which dynamically fuse speaker-aware global information and fine-grained local features to guide expert selection. This mechanism enables speaker-specific routing by leveraging both global context and local acoustic cues. Experiments on LibriSpeechMix show that GLAD outperforms existing MTASR approaches, particularly in challenging multi-talker scenarios. To our best knowledge, this is the first work to apply Mixture-of-Experts (MoE) to end-to-end MTASR with a global-local fusion strategy. Our code and train dataset can be found at https://github.com/NKU-HLT/GLAD.
>
---
#### [replaced 019] Neural Audio Codecs for Prompt-Driven Universal Source Separation
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11717v2](http://arxiv.org/pdf/2509.11717v2)**

> **作者:** Adhiraj Banerjee; Vipul Arora
>
> **备注:** main content- 10 pages, total - 23 pages, 1 figure, pre-print, under review
>
> **摘要:** Text-guided source separation supports flexible audio editing across media and assistive applications, but existing models like AudioSep are too compute-heavy for edge deployment. Neural audio codec (NAC) models such as CodecFormer and SDCodec are compute-efficient but limited to fixed-class separation. We introduce CodecSep, the first NAC-based model for on-device universal, text-driven separation. CodecSep combines DAC compression with a Transformer masker modulated by CLAP-derived FiLM parameters. Across six open-domain benchmarks under matched training/prompt protocols, \textbf{CodecSep} surpasses \textbf{AudioSep} in separation fidelity (SI-SDR) while remaining competitive in perceptual quality (ViSQOL) and matching or exceeding fixed-stem baselines (TDANet, CodecFormer, SDCodec). In code-stream deployments, it needs just 1.35~GMACs end-to-end -- approximately $54\times$ less compute ($25\times$ architecture-only) than spectrogram-domain separators like AudioSep -- while remaining fully bitstream-compatible.
>
---
#### [replaced 020] CAARMA: Class Augmentation with Adversarial Mixup Regularization
- **分类: cs.SD; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16718v3](http://arxiv.org/pdf/2503.16718v3)**

> **作者:** Massa Baali; Xiang Li; Hao Chen; Syed Abdul Hannan; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. The code is available at: https://github.com/massabaali7/CAARMA/
>
---
#### [replaced 021] Exploring How Audio Effects Alter Emotion with Foundation Models
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15151v2](http://arxiv.org/pdf/2509.15151v2)**

> **作者:** Stelios Katsis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **备注:** https://github.com/stelioskt/audioFX
>
> **摘要:** Audio effects (FX) such as reverberation, distortion, modulation, and dynamic range processing play a pivotal role in shaping emotional responses during music listening. While prior studies have examined links between low-level audio features and affective perception, the systematic impact of audio FX on emotion remains underexplored. This work investigates how foundation models - large-scale neural architectures pretrained on multimodal data - can be leveraged to analyze these effects. Such models encode rich associations between musical structure, timbre, and affective meaning, offering a powerful framework for probing the emotional consequences of sound design techniques. By applying various probing methods to embeddings from deep learning models, we examine the complex, nonlinear relationships between audio FX and estimated emotion, uncovering patterns tied to specific effects and evaluating the robustness of foundation audio models. Our findings aim to advance understanding of the perceptual impact of audio production practices, with implications for music cognition, performance, and affective computing.
>
---
#### [replaced 022] Cross-Lingual F5-TTS: Towards Language-Agnostic Voice Cloning and Speech Synthesis
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.14579v2](http://arxiv.org/pdf/2509.14579v2)**

> **作者:** Qingyu Liu; Yushen Chen; Zhikang Niu; Chunhui Wang; Yunting Yang; Bowen Zhang; Jian Zhao; Pengcheng Zhu; Kai Yu; Xie Chen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models have shown high-quality speech synthesis. However, most current flow-matching-based TTS models still rely on reference transcripts corresponding to the audio prompt for synthesis. This dependency prevents cross-lingual voice cloning when audio prompt transcripts are unavailable, particularly for unseen languages. The key challenges for flow-matching-based TTS models to remove audio prompt transcripts are identifying word boundaries during training and determining appropriate duration during inference. In this paper, we introduce Cross-Lingual F5-TTS, a framework that enables cross-lingual voice cloning without audio prompt transcripts. Our method preprocesses audio prompts by forced alignment to obtain word boundaries, enabling direct synthesis from audio prompts while excluding transcripts during training. To address the duration modeling challenge, we train speaking rate predictors at different linguistic granularities to derive duration from speaker pace. Experiments show that our approach matches the performance of F5-TTS while enabling cross-lingual voice cloning.
>
---
