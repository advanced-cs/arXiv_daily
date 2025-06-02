# 音频 cs.SD;  eess.SP

- **最新发布 36 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] The Computation of Generalized Embeddings for Underwater Acoustic Target Recognition using Contrastive Learning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究水下声学目标识别的无监督学习任务，旨在解决标注数据稀缺问题。通过对比学习框架，使用Conformer编码器和VICReg损失在低质量未标注数据上训练，生成通用声学嵌入，成功迁移至船只类型和海洋生物叫声分类任务，验证了无监督方法在声学分析的潜力。**

- **链接: [http://arxiv.org/pdf/2505.12904v1](http://arxiv.org/pdf/2505.12904v1)**

> **作者:** Hilde I. Hummel; Arwin Gansekoele; Sandjai Bhulai; Rob van der Mei
>
> **摘要:** The increasing level of sound pollution in marine environments poses an increased threat to ocean health, making it crucial to monitor underwater noise. By monitoring this noise, the sources responsible for this pollution can be mapped. Monitoring is performed by passively listening to these sounds. This generates a large amount of data records, capturing a mix of sound sources such as ship activities and marine mammal vocalizations. Although machine learning offers a promising solution for automatic sound classification, current state-of-the-art methods implement supervised learning. This requires a large amount of high-quality labeled data that is not publicly available. In contrast, a massive amount of lower-quality unlabeled data is publicly available, offering the opportunity to explore unsupervised learning techniques. This research explores this possibility by implementing an unsupervised Contrastive Learning approach. Here, a Conformer-based encoder is optimized by the so-called Variance-Invariance-Covariance Regularization loss function on these lower-quality unlabeled data and the translation to the labeled data is made. Through classification tasks involving recognizing ship types and marine mammal vocalizations, our method demonstrates to produce robust and generalized embeddings. This shows to potential of unsupervised methods for various automatic underwater acoustic analysis tasks.
>
---
#### [new 002] Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于对话语音合成任务，旨在解决现有模型因情感感知不足和离散编码冗余导致的共情表达问题。提出Chain-Talker三阶段框架：情感理解提取对话情感特征，语义理解生成紧凑编码，共情渲染合成语音，并开发LLM驱动的CSS-EmCap工具自动生成情感标注，实验验证其合成效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.12597v1](http://arxiv.org/pdf/2505.12597v1)**

> **作者:** Yifan Hu; Rui Liu; Yi Ren; Xiang Yin; Haizhou Li
>
> **备注:** 16 pages, 5 figures, 5 tables. Accepted by ACL 2025 (Findings)
>
> **摘要:** Conversational Speech Synthesis (CSS) aims to align synthesized speech with the emotional and stylistic context of user-agent interactions to achieve empathy. Current generative CSS models face interpretability limitations due to insufficient emotional perception and redundant discrete speech coding. To address the above issues, we present Chain-Talker, a three-stage framework mimicking human cognition: Emotion Understanding derives context-aware emotion descriptors from dialogue history; Semantic Understanding generates compact semantic codes via serialized prediction; and Empathetic Rendering synthesizes expressive speech by integrating both components. To support emotion modeling, we develop CSS-EmCap, an LLM-driven automated pipeline for generating precise conversational speech emotion captions. Experiments on three benchmark datasets demonstrate that Chain-Talker produces more expressive and empathetic speech than existing methods, with CSS-EmCap contributing to reliable emotion modeling. The code and demos are available at: https://github.com/AI-S2-Lab/Chain-Talker.
>
---
#### [new 003] Unified Cross-modal Translation of Score Images, Symbolic Music, and Performance Audio
- **分类: cs.SD; cs.AI; cs.CV; eess.AS**

- **简介: 该论文提出统一跨模态音乐翻译模型，解决传统方法需为各任务单独训练的问题。通过构建1300小时配对数据集及统一标记化框架，将乐谱图像、符号音乐和音频转化为序列，使用单一Transformer处理多任务。实验表明模型在光学识谱错误率（降至13.67%）等任务上超越单任务基线，并首次实现乐谱图像生成音频。**

- **链接: [http://arxiv.org/pdf/2505.12863v1](http://arxiv.org/pdf/2505.12863v1)**

> **作者:** Jongmin Jung; Dongmin Kim; Sihun Lee; Seola Cho; Hyungjoon Soh; Irmak Bukey; Chris Donahue; Dasaem Jeong
>
> **备注:** Submitted to IEEE Transactions on Audio, Speech and Language Processing (TASLPRO)
>
> **摘要:** Music exists in various modalities, such as score images, symbolic scores, MIDI, and audio. Translations between each modality are established as core tasks of music information retrieval, such as automatic music transcription (audio-to-MIDI) and optical music recognition (score image to symbolic score). However, most past work on multimodal translation trains specialized models on individual translation tasks. In this paper, we propose a unified approach, where we train a general-purpose model on many translation tasks simultaneously. Two key factors make this unified approach viable: a new large-scale dataset and the tokenization of each modality. Firstly, we propose a new dataset that consists of more than 1,300 hours of paired audio-score image data collected from YouTube videos, which is an order of magnitude larger than any existing music modal translation datasets. Secondly, our unified tokenization framework discretizes score images, audio, MIDI, and MusicXML into a sequence of tokens, enabling a single encoder-decoder Transformer to tackle multiple cross-modal translation as one coherent sequence-to-sequence task. Experimental results confirm that our unified multitask model improves upon single-task baselines in several key areas, notably reducing the symbol error rate for optical music recognition from 24.58% to a state-of-the-art 13.67%, while similarly substantial improvements are observed across the other translation tasks. Notably, our approach achieves the first successful score-image-conditioned audio generation, marking a significant breakthrough in cross-modal music generation.
>
---
#### [new 004] MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文提出MMAR基准测试，用于评估音频-语言模型在跨语音、音乐及混合场景中的深度推理能力。针对现有基准领域单一、缺乏复杂推理任务的问题，构建了包含四层推理结构（信号、感知、语义、文化）的1000项数据集，标注思维链解释，并通过多模型测试揭示当前模型在深层理解和专业知识上的不足。**

- **链接: [http://arxiv.org/pdf/2505.13032v1](http://arxiv.org/pdf/2505.13032v1)**

> **作者:** Ziyang Ma; Yinghao Ma; Yanqiao Zhu; Chen Yang; Yi-Wen Chao; Ruiyang Xu; Wenxi Chen; Yuanzhe Chen; Zhuo Chen; Jian Cong; Kai Li; Keliang Li; Siyou Li; Xinfeng Li; Xiquan Li; Zheng Lian; Yuzhe Liang; Minghao Liu; Zhikang Niu; Tianrui Wang; Yuping Wang; Yuxuan Wang; Yihao Wu; Guanrou Yang; Jianwei Yu; Ruibin Yuan; Zhisheng Zheng; Ziya Zhou; Haina Zhu; Wei Xue; Emmanouil Benetos; Kai Yu; Eng-Siong Chng; Xie Chen
>
> **备注:** Open-source at https://github.com/ddlBoJack/MMAR
>
> **摘要:** We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. Unlike existing benchmarks that are limited to specific domains of sound, music, or speech, MMAR extends them to a broad spectrum of real-world audio scenarios, including mixed-modality combinations of sound, music, and speech. Each question in MMAR is hierarchically categorized across four reasoning layers: Signal, Perception, Semantic, and Cultural, with additional sub-categories within each layer to reflect task diversity and complexity. To further foster research in this area, we annotate every question with a Chain-of-Thought (CoT) rationale to promote future advancements in audio reasoning. Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. We evaluate MMAR using a broad set of models, including Large Audio-Language Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models (OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with audio caption inputs. The performance of these models on MMAR highlights the benchmark's challenging nature, and our analysis further reveals critical limitations of understanding and reasoning capabilities among current models. We hope MMAR will serve as a catalyst for future advances in this important but little-explored area.
>
---
#### [new 005] Text2midi-InferAlign: Improving Symbolic Music Generation with Inference-Time Alignment
- **分类: cs.SD; cs.AI; cs.MM; eess.AS; 68T07; I.2.1**

- **简介: 该论文研究符号音乐生成任务，解决生成音乐与文本描述不一致的问题。提出Text2midi-InferAlign方法，在推理阶段通过节奏对齐奖励（文本-音频一致性）与和声一致性评分优化生成结果，无需额外训练即可提升现有自回归模型的生成质量与结构合理性。**

- **链接: [http://arxiv.org/pdf/2505.12669v1](http://arxiv.org/pdf/2505.12669v1)**

> **作者:** Abhinaba Roy; Geeta Puri; Dorien Herremans
>
> **备注:** 7 pages, 1 figure, 5 tables
>
> **摘要:** We present Text2midi-InferAlign, a novel technique for improving symbolic music generation at inference time. Our method leverages text-to-audio alignment and music structural alignment rewards during inference to encourage the generated music to be consistent with the input caption. Specifically, we introduce two objectives scores: a text-audio consistency score that measures rhythmic alignment between the generated music and the original text caption, and a harmonic consistency score that penalizes generated music containing notes inconsistent with the key. By optimizing these alignment-based objectives during the generation process, our model produces symbolic music that is more closely tied to the input captions, thereby improving the overall quality and coherence of the generated compositions. Our approach can extend any existing autoregressive model without requiring further training or fine-tuning. We evaluate our work on top of Text2midi - an existing text-to-midi generation model, demonstrating significant improvements in both objective and subjective evaluation metrics.
>
---
#### [new 006] MultiActor-Audiobook: Zero-Shot Audiobook Generation with Faces and Voices of Multiple Speakers
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于零样本语音合成任务，旨在解决传统有声书生成需手动配置韵律、语调单一及训练成本高的问题。提出MultiActor-Audiobook方法，结合多模态说话人特征生成（MSP）和大模型脚本指令生成（LSI），无需训练即可生成情感丰富、韵律一致的多说话人有声书，实验验证其优于商业系统。**

- **链接: [http://arxiv.org/pdf/2505.13082v1](http://arxiv.org/pdf/2505.13082v1)**

> **作者:** Kyeongman Park; Seongho Joo; Kyomin Jung
>
> **摘要:** We introduce MultiActor-Audiobook, a zero-shot approach for generating audiobooks that automatically produces consistent, expressive, and speaker-appropriate prosody, including intonation and emotion. Previous audiobook systems have several limitations: they require users to manually configure the speaker's prosody, read each sentence with a monotonic tone compared to voice actors, or rely on costly training. However, our MultiActor-Audiobook addresses these issues by introducing two novel processes: (1) MSP (**Multimodal Speaker Persona Generation**) and (2) LSI (**LLM-based Script Instruction Generation**). With these two processes, MultiActor-Audiobook can generate more emotionally expressive audiobooks with a consistent speaker prosody without additional training. We compare our system with commercial products, through human and MLLM evaluations, achieving competitive results. Furthermore, we demonstrate the effectiveness of MSP and LSI through ablation studies.
>
---
#### [new 007] Distilling a speech and music encoder with task arithmetic
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多模态音频表示学习任务，旨在解决语音和音乐自监督模型分离导致的通用音频理解受限问题。提出通过任务算术分解蒸馏语音/音乐编码器的特征向量，线性组合形成统一模型，在保持灵活域权重调节的同时降低训练成本，实验显示其性能优于集成蒸馏方法。**

- **链接: [http://arxiv.org/pdf/2505.13270v1](http://arxiv.org/pdf/2505.13270v1)**

> **作者:** Fabian Ritter-Gutierrez; Yi-Cheng Lin; Jui-Chiang Wei; Jeremy H. M Wong; Eng Siong Chng; Nancy F. Chen; Hung-yi Lee
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Despite the progress in self-supervised learning (SSL) for speech and music, existing models treat these domains separately, limiting their capacity for unified audio understanding. A unified model is desirable for applications that require general representations, e.g. audio large language models. Nonetheless, directly training a general model for speech and music is computationally expensive. Knowledge Distillation of teacher ensembles may be a natural solution, but we posit that decoupling the distillation of the speech and music SSL models allows for more flexibility. Thus, we propose to learn distilled task vectors and then linearly interpolate them to form a unified speech+music model. This strategy enables flexible domain emphasis through adjustable weights and is also simpler to train. Experiments on speech and music benchmarks demonstrate that our method yields superior overall performance compared to ensemble distillation.
>
---
#### [new 008] Codec-Based Deepfake Source Tracing via Neural Audio Codec Taxonomy
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造溯源任务，旨在解决现有反伪造研究忽视追踪生成模型的问题。针对基于神经音频编解码器的深度伪造语音（CodecFake），作者提出通过编解码器分类法溯源生成系统，实验验证了可行性并揭示了技术挑战。**

- **链接: [http://arxiv.org/pdf/2505.12994v1](http://arxiv.org/pdf/2505.12994v1)**

> **作者:** Xuanjun Chen; I-Ming Lin; Lin Zhang; Jiawei Du; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Recent advances in neural audio codec-based speech generation (CoSG) models have produced remarkably realistic audio deepfakes. We refer to deepfake speech generated by CoSG systems as codec-based deepfake, or CodecFake. Although existing anti-spoofing research on CodecFake predominantly focuses on verifying the authenticity of audio samples, almost no attention was given to tracing the CoSG used in generating these deepfakes. In CodecFake generation, processes such as speech-to-unit encoding, discrete unit modeling, and unit-to-speech decoding are fundamentally based on neural audio codecs. Motivated by this, we introduce source tracing for CodecFake via neural audio codec taxonomy, which dissects neural audio codecs to trace CoSG. Our experimental results on the CodecFake+ dataset provide promising initial evidence for the feasibility of CodecFake source tracing while also highlighting several challenges that warrant further investigation.
>
---
#### [new 009] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决传统方法忽略多属性建模、计算成本高的问题。提出OZSpeech框架，首次将最优传输条件流匹配与单步采样结合，通过分解语音成分的标记化表示精准建模属性，并利用学习先验简化生成步骤，提升语音克隆的准确性、自然度和效率。**

- **链接: [http://arxiv.org/pdf/2505.12800v1](http://arxiv.org/pdf/2505.12800v1)**

> **作者:** Hieu-Nghia Huynh-Nguyen; Ngoc Son Nguyen; Huynh Nguyen Dang; Thieu Vo; Truong-Son Hy; Van Nguyen
>
> **摘要:** Text-to-speech (TTS) systems have seen significant advancements in recent years, driven by improvements in deep learning and neural network architectures. Viewing the output speech as a data distribution, previous approaches often employ traditional speech representations, such as waveforms or spectrograms, within the Flow Matching framework. However, these methods have limitations, including overlooking various speech attributes and incurring high computational costs due to additional constraints introduced during training. To address these challenges, we introduce OZSpeech, the first TTS method to explore optimal transport conditional flow matching with one-step sampling and a learned prior as the condition, effectively disregarding preceding states and reducing the number of sampling steps. Our approach operates on disentangled, factorized components of speech in token format, enabling accurate modeling of each speech attribute, which enhances the TTS system's ability to precisely clone the prompt speech. Experimental results show that our method achieves promising performance over existing methods in content accuracy, naturalness, prosody generation, and speaker style preservation. Audio samples are available at our demo page https://ozspeech.github.io/OZSpeech_Web/.
>
---
#### [new 010] SepPrune: Structured Pruning for Efficient Deep Speech Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音分离模型压缩任务，旨在解决现有模型计算效率低的问题。提出SepPrune框架，通过分析计算瓶颈、可微分通道剪枝和快速微调，在保持性能的同时显著降低计算成本，收敛速度提升36倍。**

- **链接: [http://arxiv.org/pdf/2505.12079v1](http://arxiv.org/pdf/2505.12079v1)**

> **作者:** Yuqi Li; Kai Li; Xin Yin; Zhifei Yang; Junhao Dong; Zeyu Dong; Chuanguang Yang; Yingli Tian; Yao Lu
>
> **摘要:** Although deep learning has substantially advanced speech separation in recent years, most existing studies continue to prioritize separation quality while overlooking computational efficiency, an essential factor for low-latency speech processing in real-time applications. In this paper, we propose SepPrune, the first structured pruning framework specifically designed to compress deep speech separation models and reduce their computational cost. SepPrune begins by analyzing the computational structure of a given model to identify layers with the highest computational burden. It then introduces a differentiable masking strategy to enable gradient-driven channel selection. Based on the learned masks, SepPrune prunes redundant channels and fine-tunes the remaining parameters to recover performance. Extensive experiments demonstrate that this learnable pruning paradigm yields substantial advantages for channel pruning in speech separation models, outperforming existing methods. Notably, a model pruned with SepPrune can recover 85% of the performance of a pre-trained model (trained over hundreds of epochs) with only one epoch of fine-tuning, and achieves convergence 36$\times$ faster than training from scratch. Code is available at https://github.com/itsnotacie/SepPrune.
>
---
#### [new 011] ASR-FAIRBENCH: Measuring and Benchmarking Equity Across Speech Recognition Systems
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别公平性评估任务，旨在解决ASR系统在不同人口群体中的性能差异问题。研究者构建了ASR-FAIRBENCH评测框架，结合公平评分（通过混合效应模型计算）与词错误率，提出FAAS综合指标，利用Fair-Speech数据集揭示主流模型的群体性能差距，为开发包容性技术提供基准。**

- **链接: [http://arxiv.org/pdf/2505.11572v1](http://arxiv.org/pdf/2505.11572v1)**

> **作者:** Anand Rai; Satyam Rahangdale; Utkarsh Anand; Animesh Mukherjee
>
> **备注:** Paper accepted at INTERSPEECH 2025
>
> **摘要:** Automatic Speech Recognition (ASR) systems have become ubiquitous in everyday applications, yet significant disparities in performance across diverse demographic groups persist. In this work, we introduce the ASR-FAIRBENCH leaderboard which is designed to assess both the accuracy and equity of ASR models in real-time. Leveraging the Meta's Fair-Speech dataset, which captures diverse demographic characteristics, we employ a mixed-effects Poisson regression model to derive an overall fairness score. This score is integrated with traditional metrics like Word Error Rate (WER) to compute the Fairness Adjusted ASR Score (FAAS), providing a comprehensive evaluation framework. Our approach reveals significant performance disparities in SOTA ASR models across demographic groups and offers a benchmark to drive the development of more inclusive ASR technologies.
>
---
#### [new 012] DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation
- **分类: cs.SD; eess.AS**

- **简介: 该研究属于语音生成领域，针对神经音频编解码器在帧率与音质间的权衡问题，提出DualCodec双流编码模型。通过融合自监督语义表征与波形特征，在低帧率下增强语义信息并保持高音质，提升生成效率。实验证明其优于主流编解码系统。**

- **链接: [http://arxiv.org/pdf/2505.13000v1](http://arxiv.org/pdf/2505.13000v1)**

> **作者:** Jiaqi Li; Xiaolong Lin; Zhekai Li; Shixi Huang; Yuancheng Wang; Chaoren Wang; Zhenpeng Zhan; Zhizheng Wu
>
> **备注:** Accepted to Interspeech 2025. Github: https://github.com/jiaqili3/dualcodec
>
> **摘要:** Neural audio codecs form the foundational building blocks for language model (LM)-based speech generation. Typically, there is a trade-off between frame rate and audio quality. This study introduces a low-frame-rate, semantically enhanced codec model. Existing approaches distill semantically rich self-supervised (SSL) representations into the first-layer codec tokens. This work proposes DualCodec, a dual-stream encoding approach that integrates SSL and waveform representations within an end-to-end codec framework. In this setting, DualCodec enhances the semantic information in the first-layer codec and enables the codec system to maintain high audio quality while operating at a low frame rate. Note that a low-frame-rate codec improves the efficiency of speech generation. Experimental results on audio codec and speech generation tasks confirm the effectiveness of the proposed DualCodec compared to state-of-the-art codec systems, such as Mimi Codec, SpeechTokenizer, DAC, and Encodec. Demos and codes are available at: https://dualcodec.github.io
>
---
#### [new 013] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于语音安全领域，旨在防御基于扩散模型的未经授权语音克隆。针对现有防御方法不兼容扩散机制的问题，提出VoiceCloak框架：通过对抗扰动混淆说话人身份（扭曲表征、干扰注意力），并降低语音质量（评分放大、噪声破坏语义），阻断克隆过程。实验验证其防御效果显著。**

- **链接: [http://arxiv.org/pdf/2505.12332v1](http://arxiv.org/pdf/2505.12332v1)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.
>
---
#### [new 014] SounDiT: Geo-Contextual Soundscape-to-Landscape Generation
- **分类: cs.SD; cs.AI; cs.GR; cs.HC; eess.AS**

- **简介: 该论文研究地理上下文声景到景观生成任务（GeoS2L），解决现有音频生成图像方法忽略地理环境导致不真实的问题。提出SounDiT模型，结合地理知识构建多模态数据集，并设计地理一致性评估框架，提升生成图像的地理合理性与视觉质量。**

- **链接: [http://arxiv.org/pdf/2505.12734v1](http://arxiv.org/pdf/2505.12734v1)**

> **作者:** Junbo Wang; Haofeng Tan; Bowen Liao; Albert Jiang; Teng Fei; Qixing Huang; Zhengzhong Tu; Shan Ye; Yuhao Kang
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** We present a novel and practically significant problem-Geo-Contextual Soundscape-to-Landscape (GeoS2L) generation-which aims to synthesize geographically realistic landscape images from environmental soundscapes. Prior audio-to-image generation methods typically rely on general-purpose datasets and overlook geographic and environmental contexts, resulting in unrealistic images that are misaligned with real-world environmental settings. To address this limitation, we introduce a novel geo-contextual computational framework that explicitly integrates geographic knowledge into multimodal generative modeling. We construct two large-scale geo-contextual multimodal datasets, SoundingSVI and SonicUrban, pairing diverse soundscapes with real-world landscape images. We propose SounDiT, a novel Diffusion Transformer (DiT)-based model that incorporates geo-contextual scene conditioning to synthesize geographically coherent landscape images. Furthermore, we propose a practically-informed geo-contextual evaluation framework, the Place Similarity Score (PSS), across element-, scene-, and human perception-levels to measure consistency between input soundscapes and generated landscape images. Extensive experiments demonstrate that SounDiT outperforms existing baselines in both visual fidelity and geographic settings. Our work not only establishes foundational benchmarks for GeoS2L generation but also highlights the importance of incorporating geographic domain knowledge in advancing multimodal generative models, opening new directions at the intersection of generative AI, geography, urban planning, and environmental sciences.
>
---
#### [new 015] Time-Frequency-Based Attention Cache Memory Model for Real-Time Speech Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对实时语音分离任务中因果模型因历史信息保留不足导致的性能差距问题，提出时频注意力缓存记忆模型（TFACM）。通过LSTM捕获频域特征，结合缓存存储历史信息，并利用因果注意力细化模块增强时间特征，在保持低复杂度前提下达到与主流因果模型相当的分离性能。**

- **链接: [http://arxiv.org/pdf/2505.13094v1](http://arxiv.org/pdf/2505.13094v1)**

> **作者:** Guo Chen; Kai Li; Runxuan Yang; Xiaolin Hu
>
> **摘要:** Existing causal speech separation models often underperform compared to non-causal models due to difficulties in retaining historical information. To address this, we propose the Time-Frequency Attention Cache Memory (TFACM) model, which effectively captures spatio-temporal relationships through an attention mechanism and cache memory (CM) for historical information storage. In TFACM, an LSTM layer captures frequency-relative positions, while causal modeling is applied to the time dimension using local and global representations. The CM module stores past information, and the causal attention refinement (CAR) module further enhances time-based feature representations for finer granularity. Experimental results showed that TFACM achieveed comparable performance to the SOTA TF-GridNet-Causal model, with significantly lower complexity and fewer trainable parameters. For more details, visit the project page: https://cslikai.cn/TFACM/.
>
---
#### [new 016] Personalized Fine-Tuning with Controllable Synthetic Speech from LLM-Generated Transcripts for Dysarthric Speech Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升构音障碍患者的语音识别准确率。通过参数高效微调、合成语音数据生成（Parler-TTS模拟病理语音）及个性化优化（x-vector、AdaLoRA适配器），结合wav2vec 2.0音频表征，显著降低词错误率（WER），较传统方法提升达23%。**

- **链接: [http://arxiv.org/pdf/2505.12991v1](http://arxiv.org/pdf/2505.12991v1)**

> **作者:** Dominik Wagner; Ilja Baumann; Natalie Engert; Seanie Lee; Elmar Nöth; Korbinian Riedhammer; Tobias Bocklet
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** In this work, we present our submission to the Speech Accessibility Project challenge for dysarthric speech recognition. We integrate parameter-efficient fine-tuning with latent audio representations to improve an encoder-decoder ASR system. Synthetic training data is generated by fine-tuning Parler-TTS to mimic dysarthric speech, using LLM-generated prompts for corpus-consistent target transcripts. Personalization with x-vectors consistently reduces word error rates (WERs) over non-personalized fine-tuning. AdaLoRA adapters outperform full fine-tuning and standard low-rank adaptation, achieving relative WER reductions of ~23% and ~22%, respectively. Further improvements (~5% WER reduction) come from incorporating wav2vec 2.0-based audio representations. Training with synthetic dysarthric speech yields up to ~7% relative WER improvement over personalized fine-tuning alone.
>
---
#### [new 017] Acoustic Field Reconstruction in Tubes via Physics-Informed Neural Networks
- **分类: eess.AS; cs.SD; eess.SP; physics.app-ph**

- **简介: 该论文属于逆问题求解任务，旨在解决管道声场重建中辐射模型未知、噪声干扰及数据有限的问题。通过构建物理信息神经网络(PINNs)框架，结合提出的PINN-FTM方法和传统优化法，实现了仅基于辐射端压力数据的声场重构与模型系数预测。结果表明PINN-FTM在噪声耐受性和预测可靠性上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.12557v1](http://arxiv.org/pdf/2505.12557v1)**

> **作者:** Xinmeng Luan; Kazuya Yokota; Gary Scavone
>
> **备注:** 8 pages, 5 figures, conference
>
> **摘要:** This study investigates the application of Physics-Informed Neural Networks (PINNs) to inverse problems in acoustic tube analysis, focusing on reconstructing acoustic fields from noisy and limited observation data. Specifically, we address scenarios where the radiation model is unknown, and pressure data is only available at the tube's radiation end. A PINNs framework is proposed to reconstruct the acoustic field, along with the PINN Fine-Tuning Method (PINN-FTM) and a traditional optimization method (TOM) for predicting radiation model coefficients. The results demonstrate that PINNs can effectively reconstruct the tube's acoustic field under noisy conditions, even with unknown radiation parameters. PINN-FTM outperforms TOM by delivering balanced and reliable predictions and exhibiting robust noise-tolerance capabilities.
>
---
#### [new 018] Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于声学识别的计算优化任务，旨在解决连续小波变换（CWT）在深度学习特征提取中计算成本高的问题。通过优化小波核长度和输出尺度图的跳跃步长，降低计算复杂度，实验证明该方法在保持模型性能的同时显著减少运算开销。**

- **链接: [http://arxiv.org/pdf/2505.13017v1](http://arxiv.org/pdf/2505.13017v1)**

> **作者:** Dang Thoai Phan; Tuan Anh Huynh; Van Tuan Pham; Cao Minh Tran; Van Thuan Mai; Ngoc Quy Tran
>
> **摘要:** The Continuous Wavelet Transform (CWT) is an effective tool for feature extraction in acoustic recognition using Convolutional Neural Networks (CNNs), particularly when applied to non-stationary audio. However, its high computational cost poses a significant challenge, often leading researchers to prefer alternative methods such as the Short-Time Fourier Transform (STFT). To address this issue, this paper proposes a method to reduce the computational complexity of CWT by optimizing the length of the wavelet kernel and the hop size of the output scalogram. Experimental results demonstrate that the proposed approach significantly reduces computational cost while maintaining the robust performance of the trained model in acoustic recognition tasks.
>
---
#### [new 019] Improving Open-Set Semantic Segmentation in 3D Point Clouds by Conditional Channel Capacity Maximization: Preliminary Results
- **分类: cs.CV; eess.SP**

- **简介: 该论文研究开放集3D点云语义分割（O3S），解决现有闭集模型无法识别训练外类别的问题。提出条件通道容量最大化（3CM）正则化方法，通过最大化特征与预测的条件互信息，增强编码器保留类别相关特征的能力，从而提升未知类别的检测与分割效果。实验验证了方法的有效性，并探讨动态开放世界适应的未来方向。**

- **链接: [http://arxiv.org/pdf/2505.11521v1](http://arxiv.org/pdf/2505.11521v1)**

> **作者:** Wang Fang; Shirin Rahimi; Olivia Bennett; Sophie Carter; Mitra Hassani; Xu Lan; Omid Javadi; Lucas Mitchell
>
> **摘要:** Point-cloud semantic segmentation underpins a wide range of critical applications. Although recent deep architectures and large-scale datasets have driven impressive closed-set performance, these models struggle to recognize or properly segment objects outside their training classes. This gap has sparked interest in Open-Set Semantic Segmentation (O3S), where models must both correctly label known categories and detect novel, unseen classes. In this paper, we propose a plug and play framework for O3S. By modeling the segmentation pipeline as a conditional Markov chain, we derive a novel regularizer term dubbed Conditional Channel Capacity Maximization (3CM), that maximizes the mutual information between features and predictions conditioned on each class. When incorporated into standard loss functions, 3CM encourages the encoder to retain richer, label-dependent features, thereby enhancing the network's ability to distinguish and segment previously unseen categories. Experimental results demonstrate effectiveness of proposed method on detecting unseen objects. We further outline future directions for dynamic open-world adaptation and efficient information-theoretic estimation.
>
---
#### [new 020] BINAQUAL: A Full-Reference Objective Localization Similarity Metric for Binaural Audio
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频质量评估任务，旨在解决双耳音频空间定位失真的客观评价问题。提出BINAQUAL全参考指标，通过改进AMBIQUAL算法适配双耳音频，验证其对声源位置变化、音频劣化等场景的敏感性，证明其与主观测试结果高度相关，为沉浸式音频处理提供可靠的空间保真度评估工具。**

- **链接: [http://arxiv.org/pdf/2505.11915v1](http://arxiv.org/pdf/2505.11915v1)**

> **作者:** Davoud Shariat Panah; Dan Barry; Alessandro Ragano; Jan Skoglund; Andrew Hines
>
> **备注:** Submitted to the Journal of Audio Engineering Society (JAES)
>
> **摘要:** Spatial audio enhances immersion in applications such as virtual reality, augmented reality, gaming, and cinema by creating a three-dimensional auditory experience. Ensuring the spatial fidelity of binaural audio is crucial, given that processes such as compression, encoding, or transmission can alter localization cues. While subjective listening tests like MUSHRA remain the gold standard for evaluating spatial localization quality, they are costly and time-consuming. This paper introduces BINAQUAL, a full-reference objective metric designed to assess localization similarity in binaural audio recordings. BINAQUAL adapts the AMBIQUAL metric, originally developed for localization quality assessment in ambisonics audio format to the binaural domain. We evaluate BINAQUAL across five key research questions, examining its sensitivity to variations in sound source locations, angle interpolations, surround speaker layouts, audio degradations, and content diversity. Results demonstrate that BINAQUAL effectively differentiates between subtle spatial variations and correlates strongly with subjective listening tests, making it a reliable metric for binaural localization quality assessment. The proposed metric provides a robust benchmark for ensuring spatial accuracy in binaural audio processing, paving the way for improved objective evaluations in immersive audio applications.
>
---
#### [new 021] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言建模任务，旨在解决现有模型依赖离散化分层结构导致的误差和复杂性问题。作者提出SLED方法，通过连续潜在空间编码语音波形并采用能量距离目标进行自回归建模，避免了量化误差，简化了架构，同时保持信息完整性和推理效率。**

- **链接: [http://arxiv.org/pdf/2505.13181v1](http://arxiv.org/pdf/2505.13181v1)**

> **作者:** Zhengrui Ma; Yang Feng; Chenze Shao; Fandong Meng; Jie Zhou; Min Zhang
>
> **备注:** Demos and code are available at https://github.com/ictnlp/SLED-TTS
>
> **摘要:** We introduce SLED, an alternative approach to speech language modeling by encoding speech waveforms into sequences of continuous latent representations and modeling them autoregressively using an energy distance objective. The energy distance offers an analytical measure of the distributional gap by contrasting simulated and target samples, enabling efficient training to capture the underlying continuous autoregressive distribution. By bypassing reliance on residual vector quantization, SLED avoids discretization errors and eliminates the need for the complicated hierarchical architectures common in existing speech language models. It simplifies the overall modeling pipeline while preserving the richness of speech information and maintaining inference efficiency. Empirical results demonstrate that SLED achieves strong performance in both zero-shot and streaming speech synthesis, showing its potential for broader applications in general-purpose speech language models.
>
---
#### [new 022] Exploring the Potential of SSL Models for Sound Event Detection
- **分类: eess.AS; cs.AI; cs.SD; I.5.4; I.2.10; H.5.5**

- **简介: 该论文研究自监督学习（SSL）模型在声音事件检测（SED）中的协同应用，解决模型选择与融合策略不明确的问题。通过评估多类SSL模型（如BEATs、WavLM），提出三种融合框架及自适应后处理方法（nSEBBs），实验表明双模态融合和边界优化可提升检测性能，为SED系统设计提供理论指导。**

- **链接: [http://arxiv.org/pdf/2505.11889v1](http://arxiv.org/pdf/2505.11889v1)**

> **作者:** Hanfang Cui; Longfei Song; Li Li; Dongxing Xu; Yanhua Long
>
> **备注:** 27 pages, 5 figures, submitted to the Journal of King Saud University - Computer and Information Sciences (under review)
>
> **摘要:** Self-supervised learning (SSL) models offer powerful representations for sound event detection (SED), yet their synergistic potential remains underexplored. This study systematically evaluates state-of-the-art SSL models to guide optimal model selection and integration for SED. We propose a framework that combines heterogeneous SSL representations (e.g., BEATs, HuBERT, WavLM) through three fusion strategies: individual SSL embedding integration, dual-modal fusion, and full aggregation. Experiments on the DCASE 2023 Task 4 Challenge reveal that dual-modal fusion (e.g., CRNN+BEATs+WavLM) achieves complementary performance gains, while CRNN+BEATs alone delivers the best results among individual SSL models. We further introduce normalized sound event bounding boxes (nSEBBs), an adaptive post-processing method that dynamically adjusts event boundary predictions, improving PSDS1 by up to 4% for standalone SSL models. These findings highlight the compatibility and complementarity of SSL architectures, providing guidance for task-specific fusion and robust SED system design.
>
---
#### [new 023] Unified Architecture and Unsupervised Speech Disentanglement for Speaker Embedding-Free Enrollment in Personalized Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究语音增强（SE）和个性化语音增强（PSE）的统一模型，解决传统PSE对注册语音敏感及模型分离部署的问题。提出USEF-PNet统一SE/PSE架构，以及DSEF-PNet通过无监督语音解缠分离说话人身份与干扰因素（如情绪），提升鲁棒性。实验验证模型在性能与部署效率上的优势。**

- **链接: [http://arxiv.org/pdf/2505.12288v1](http://arxiv.org/pdf/2505.12288v1)**

> **作者:** Ziling Huang; Haixin Guan; Yanhua Long
>
> **备注:** Submitted to the IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
>
> **摘要:** Conventional speech enhancement (SE) aims to improve speech perception and intelligibility by suppressing noise without requiring enrollment speech as reference, whereas personalized SE (PSE) addresses the cocktail party problem by extracting a target speaker's speech using enrollment speech. While these two tasks tackle different yet complementary challenges in speech signal processing, they often share similar model architectures, with PSE incorporating an additional branch to process enrollment speech. This suggests developing a unified model capable of efficiently handling both SE and PSE tasks, thereby simplifying deployment while maintaining high performance. However, PSE performance is sensitive to variations in enrollment speech, like emotional tone, which limits robustness in real-world applications. To address these challenges, we propose two novel models, USEF-PNet and DSEF-PNet, both extending our previous SEF-PNet framework. USEF-PNet introduces a unified architecture for processing enrollment speech, integrating SE and PSE into a single framework to enhance performance and streamline deployment. Meanwhile, DSEF-PNet incorporates an unsupervised speech disentanglement approach by pairing a mixture speech with two different enrollment utterances and enforcing consistency in the extracted target speech. This strategy effectively isolates high-quality speaker identity information from enrollment speech, reducing interference from factors such as emotion and content, thereby improving PSE robustness. Additionally, we explore a long-short enrollment pairing (LSEP) strategy to examine the impact of enrollment speech duration during both training and evaluation. Extensive experiments on the Libri2Mix and VoiceBank DEMAND demonstrate that our proposed USEF-PNet, DSEF-PNet all achieve substantial performance improvements, with random enrollment duration performing slightly better.
>
---
#### [new 024] BenSParX: A Robust Explainable Machine Learning Framework for Parkinson's Disease Detection from Bengali Conversational Speech
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于医学AI任务，旨在通过孟加拉语对话语音检测帕金森病。针对现有研究缺乏非英语数据集、声学特征单一及模型可解释性不足的问题，作者构建首个孟加拉PD语音数据集BenSParX，开发了融合多特征优化和SHAP可解释分析的机器学习框架，实现95.77%准确率，并验证跨语言有效性。**

- **链接: [http://arxiv.org/pdf/2505.12192v1](http://arxiv.org/pdf/2505.12192v1)**

> **作者:** Riad Hossain; Muhammad Ashad Kabir; Arat Ibne Golam Mowla; Animesh Chandra Roy; Ranjit Kumar Ghosh
>
> **备注:** 46 pages, 16 figures
>
> **摘要:** Parkinson's disease (PD) poses a growing global health challenge, with Bangladesh experiencing a notable rise in PD-related mortality. Early detection of PD remains particularly challenging in resource-constrained settings, where voice-based analysis has emerged as a promising non-invasive and cost-effective alternative. However, existing studies predominantly focus on English or other major languages; notably, no voice dataset for PD exists for Bengali - posing a significant barrier to culturally inclusive and accessible healthcare solutions. Moreover, most prior studies employed only a narrow set of acoustic features, with limited or no hyperparameter tuning and feature selection strategies, and little attention to model explainability. This restricts the development of a robust and generalizable machine learning model. To address this gap, we present BenSparX, the first Bengali conversational speech dataset for PD detection, along with a robust and explainable machine learning framework tailored for early diagnosis. The proposed framework incorporates diverse acoustic feature categories, systematic feature selection methods, and state-of-the-art machine learning algorithms with extensive hyperparameter optimization. Furthermore, to enhance interpretability and trust in model predictions, the framework incorporates SHAP (SHapley Additive exPlanations) analysis to quantify the contribution of individual acoustic features toward PD detection. Our framework achieves state-of-the-art performance, yielding an accuracy of 95.77%, F1 score of 95.57%, and AUC-ROC of 0.982. We further externally validated our approach by applying the framework to existing PD datasets in other languages, where it consistently outperforms state-of-the-art approaches. To facilitate further research and reproducibility, the dataset has been made publicly available at https://github.com/Riad071/BenSParX.
>
---
#### [new 025] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究小设备关键词检测的类增量学习任务，解决传统方法因依赖旧数据导致的隐私与资源消耗问题。提出AnalyticKWS方法，通过闭式解析解更新模型，无需存储历史数据或反向传播，单次适应新关键词，降低计算开销并缓解灾难性遗忘。**

- **链接: [http://arxiv.org/pdf/2505.11817v1](http://arxiv.org/pdf/2505.11817v1)**

> **作者:** Yang Xiao; Tianyi Peng; Rohan Kumar Das; Yuchen Hu; Huiping Zhuang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Keyword spotting (KWS) offers a vital mechanism to identify spoken commands in voice-enabled systems, where user demands often shift, requiring models to learn new keywords continually over time. However, a major problem is catastrophic forgetting, where models lose their ability to recognize earlier keywords. Although several continual learning methods have proven their usefulness for reducing forgetting, most existing approaches depend on storing and revisiting old data to combat catastrophic forgetting. Though effective, these methods face two practical challenges: 1) privacy risks from keeping user data and 2) large memory and time consumption that limit deployment on small devices. To address these issues, we propose an exemplar-free Analytic Continual Learning (AnalyticKWS) method that updates model parameters without revisiting earlier data. Inspired by efficient learning principles, AnalyticKWS computes a closed-form analytical solution for model updates and requires only a single epoch of adaptation for incoming keywords. AnalyticKWS demands fewer computational resources by avoiding gradient-based updates and does not store old data. By eliminating the need for back-propagation during incremental learning, the model remains lightweight and efficient. As a result, AnalyticKWS meets the challenges mentioned earlier and suits resource-limited settings well. Extensive experiments on various datasets and settings show that AnalyticKWS consistently outperforms existing continual learning methods.
>
---
#### [new 026] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，旨在提升基于流匹配的TTS模型的自然度与推理效率。针对传统方法计算成本高、生成质量不足的问题，提出浅层流匹配机制（SFM），通过粗粒度中间状态构建和单段分段流策略优化训练路径，并采用中间状态启动推理以减少计算量。实验证明SFM在提升语音自然度的同时显著加速合成速度。**

- **链接: [http://arxiv.org/pdf/2505.12226v1](http://arxiv.org/pdf/2505.12226v1)**

> **作者:** Dong Yang; Yiyi Cai; Yuki Saito; Lixu Wang; Hiroshi Saruwatari
>
> **摘要:** We propose a shallow flow matching (SFM) mechanism to enhance flow matching (FM)-based text-to-speech (TTS) models within a coarse-to-fine generation paradigm. SFM constructs intermediate states along the FM paths using coarse output representations. During training, we introduce an orthogonal projection method to adaptively determine the temporal position of these states, and apply a principled construction strategy based on a single-segment piecewise flow. The SFM inference starts from the intermediate state rather than pure noise and focuses computation on the latter stages of the FM paths. We integrate SFM into multiple TTS models with a lightweight SFM head. Experiments show that SFM consistently improves the naturalness of synthesized speech in both objective and subjective evaluations, while significantly reducing inference when using adaptive-step ODE solvers. Demo and codes are available at https://ydqmkkx.github.io/SFMDemo/.
>
---
#### [new 027] Automatic Speech Recognition for African Low-Resource Languages: Challenges and Future Directions
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于低资源语言的自动语音识别（ASR）开发任务，旨在解决非洲语言因数据稀缺、计算资源不足等挑战导致的ASR技术发展滞后问题。研究分析了技术障碍，提出社区协作数据采集、轻量化模型等策略，并通过案例验证了定制化方案的可行性，推动包容性ASR系统建设。**

- **链接: [http://arxiv.org/pdf/2505.11690v1](http://arxiv.org/pdf/2505.11690v1)**

> **作者:** Sukairaj Hafiz Imam; Babangida Sani; Dawit Ketema Gete; Bedru Yimam Ahamed; Ibrahim Said Ahmad; Idris Abdulmumin; Seid Muhie Yimam; Muhammad Yahuza Bello; Shamsuddeen Hassan Muhammad
>
> **摘要:** Automatic Speech Recognition (ASR) technologies have transformed human-computer interaction; however, low-resource languages in Africa remain significantly underrepresented in both research and practical applications. This study investigates the major challenges hindering the development of ASR systems for these languages, which include data scarcity, linguistic complexity, limited computational resources, acoustic variability, and ethical concerns surrounding bias and privacy. The primary goal is to critically analyze these barriers and identify practical, inclusive strategies to advance ASR technologies within the African context. Recent advances and case studies emphasize promising strategies such as community-driven data collection, self-supervised and multilingual learning, lightweight model architectures, and techniques that prioritize privacy. Evidence from pilot projects involving various African languages showcases the feasibility and impact of customized solutions, which encompass morpheme-based modeling and domain-specific ASR applications in sectors like healthcare and education. The findings highlight the importance of interdisciplinary collaboration and sustained investment to tackle the distinct linguistic and infrastructural challenges faced by the continent. This study offers a progressive roadmap for creating ethical, efficient, and inclusive ASR systems that not only safeguard linguistic diversity but also improve digital accessibility and promote socioeconomic participation for speakers of African languages.
>
---
#### [new 028] Event-based Star Tracking under Spacecraft Jitter: the e-STURT Dataset
- **分类: cs.CV; eess.SP**

- **简介: 该论文针对航天器抖动影响光学任务精度的问题，构建首个事件相机抖动星跟踪数据集e-STURT，通过压电执行器模拟高频抖动并采集真实数据，包含200组序列，同时提出基于事件流的抖动估计算法，为空间传感任务提供算法开发基础。**

- **链接: [http://arxiv.org/pdf/2505.12588v1](http://arxiv.org/pdf/2505.12588v1)**

> **作者:** Samya Bagchi; Peter Anastasiou; Matthew Tetlow; Tat-Jun Chin; Yasir Latif
>
> **摘要:** Jitter degrades a spacecraft's fine-pointing ability required for optical communication, earth observation, and space domain awareness. Development of jitter estimation and compensation algorithms requires high-fidelity sensor observations representative of on-board jitter. In this work, we present the Event-based Star Tracking Under Jitter (e-STURT) dataset -- the first event camera based dataset of star observations under controlled jitter conditions. Specialized hardware employed for the dataset emulates an event-camera undergoing on-board jitter. While the event camera provides asynchronous, high temporal resolution star observations, systematic and repeatable jitter is introduced using a micrometer accurate piezoelectric actuator. Various jitter sources are simulated using distinct frequency bands and utilizing both axes of motion. Ground-truth jitter is captured in hardware from the piezoelectric actuator. The resulting dataset consists of 200 sequences and is made publicly available. This work highlights the dataset generation process, technical challenges and the resulting limitations. To serve as a baseline, we propose a high-frequency jitter estimation algorithm that operates directly on the event stream. The e-STURT dataset will enable the development of jitter aware algorithms for mission critical event-based space sensing applications.
>
---
#### [new 029] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多模态推理评估任务，旨在解决大型音频-语言模型（LALMs）多跳推理能力未被系统评估的问题。作者提出SAKURA基准测试，发现LALMs难以整合语音/音频表征进行多步推理，揭示了多模态推理的核心挑战，为后续研究提供资源。**

- **链接: [http://arxiv.org/pdf/2505.13237v1](http://arxiv.org/pdf/2505.13237v1)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [new 030] WaLRUS: Wavelets for Long-range Representation Using SSMs
- **分类: eess.IV; cs.LG; cs.SY; eess.AS; eess.SP; eess.SY**

- **简介: 该论文属于序列建模任务，旨在解决状态空间模型（SSMs）依赖特定基函数的局限性。基于SaFARi框架，提出WaLRUS方法，利用Daubechies小波构建冗余基函数，扩展SSM多样性，增强长程依赖建模能力。**

- **链接: [http://arxiv.org/pdf/2505.12161v1](http://arxiv.org/pdf/2505.12161v1)**

> **作者:** Hossein Babaei; Mel White; Sina Alemohammad; Richard G. Baraniuk
>
> **备注:** 15 pages, 8 figures. Submitted to Neurips 2025
>
> **摘要:** State-Space Models (SSMs) have proven to be powerful tools for modeling long-range dependencies in sequential data. While the recent method known as HiPPO has demonstrated strong performance, and formed the basis for machine learning models S4 and Mamba, it remains limited by its reliance on closed-form solutions for a few specific, well-behaved bases. The SaFARi framework generalized this approach, enabling the construction of SSMs from arbitrary frames, including non-orthogonal and redundant ones, thus allowing an infinite diversity of possible "species" within the SSM family. In this paper, we introduce WaLRUS (Wavelets for Long-range Representation Using SSMs), a new implementation of SaFARi built from Daubechies wavelets.
>
---
#### [new 031] Learning to Highlight Audio by Watching Movies
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出视听引导的音频高亮任务，解决视频制作中音频与视觉显著性不匹配的问题。通过基于Transformer的多模态框架，利用电影数据生成模拟低质音频的混合数据集，训练模型以视频指导优化音频，提升视听和谐度。方法在定量和主观评估中优于基线。**

- **链接: [http://arxiv.org/pdf/2505.12154v1](http://arxiv.org/pdf/2505.12154v1)**

> **作者:** Chao Huang; Ruohan Gao; J. M. F. Tsang; Jan Kurcius; Cagdas Bilen; Chenliang Xu; Anurag Kumar; Sanjeel Parekh
>
> **备注:** CVPR 2025. Project page: https://wikichao.github.io/VisAH/
>
> **摘要:** Recent years have seen a significant increase in video content creation and consumption. Crafting engaging content requires the careful curation of both visual and audio elements. While visual cue curation, through techniques like optimal viewpoint selection or post-editing, has been central to media production, its natural counterpart, audio, has not undergone equivalent advancements. This often results in a disconnect between visual and acoustic saliency. To bridge this gap, we introduce a novel task: visually-guided acoustic highlighting, which aims to transform audio to deliver appropriate highlighting effects guided by the accompanying video, ultimately creating a more harmonious audio-visual experience. We propose a flexible, transformer-based multimodal framework to solve this task. To train our model, we also introduce a new dataset -- the muddy mix dataset, leveraging the meticulous audio and video crafting found in movies, which provides a form of free supervision. We develop a pseudo-data generation process to simulate poorly mixed audio, mimicking real-world scenarios through a three-step process -- separation, adjustment, and remixing. Our approach consistently outperforms several baselines in both quantitative and subjective evaluation. We also systematically study the impact of different types of contextual guidance and difficulty levels of the dataset. Our project page is here: https://wikichao.github.io/VisAH/.
>
---
#### [new 032] Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于多模态模型评估任务，旨在解决大型音频语言模型（LALMs）在时间推理能力上的评测不足问题。通过构建TREA数据集对开源LALMs进行基准测试，发现其性能低于人类水平，并提出衡量输入扰动的置信度指标，揭示模型准确性与稳定性无强关联，强调高风险场景需综合评估。**

- **链接: [http://arxiv.org/pdf/2505.13115v1](http://arxiv.org/pdf/2505.13115v1)**

> **作者:** Debarpan Bhattacharya; Apoorva Kulkarni; Sriram Ganapathy
>
> **备注:** Accepted in INTERSPEECH, 2025, Rotterdam, The Netherlands
>
> **摘要:** The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA). We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications.
>
---
#### [new 033] RoVo: Robust Voice Protection Against Unauthorized Speech Synthesis with Embedding-Level Perturbations
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音安全防御任务，旨在防止未经授权的AI语音合成滥用。针对现有音频对抗扰动易被语音增强消除的问题，提出RoVo方法：通过在高维嵌入向量中注入扰动生成受保护语音，有效抵御合成攻击并抵抗二次增强攻击。实验显示防御成功率提升超70%，用户验证其语音自然性。**

- **链接: [http://arxiv.org/pdf/2505.12686v1](http://arxiv.org/pdf/2505.12686v1)**

> **作者:** Seungmin Kim; Sohee Park; Donghyun Kim; Jisu Lee; Daeseon Choi
>
> **摘要:** With the advancement of AI-based speech synthesis technologies such as Deep Voice, there is an increasing risk of voice spoofing attacks, including voice phishing and fake news, through unauthorized use of others' voices. Existing defenses that inject adversarial perturbations directly into audio signals have limited effectiveness, as these perturbations can easily be neutralized by speech enhancement methods. To overcome this limitation, we propose RoVo (Robust Voice), a novel proactive defense technique that injects adversarial perturbations into high-dimensional embedding vectors of audio signals, reconstructing them into protected speech. This approach effectively defends against speech synthesis attacks and also provides strong resistance to speech enhancement models, which represent a secondary attack threat. In extensive experiments, RoVo increased the Defense Success Rate (DSR) by over 70% compared to unprotected speech, across four state-of-the-art speech synthesis models. Specifically, RoVo achieved a DSR of 99.5% on a commercial speaker-verification API, effectively neutralizing speech synthesis attack. Moreover, RoVo's perturbations remained robust even under strong speech enhancement conditions, outperforming traditional methods. A user study confirmed that RoVo preserves both naturalness and usability of protected speech, highlighting its effectiveness in complex and evolving threat scenarios.
>
---
#### [new 034] Suicide Risk Assessment Using Multimodal Speech Features: A Study on the SW1 Challenge Dataset
- **分类: cs.CL; cs.SD; eess.AS; I.2.7; I.5.1**

- **简介: 该论文属于多模态分类任务，旨在通过语音数据评估青少年自杀风险。研究整合自动转录（WhisperX）、语言（RoBERTa）与音频（WavLM）嵌入及手工声学特征，测试三种特征融合策略。加权注意力结合混合正则化在开发集达69%准确率，但测试集性能差距揭示模型泛化难题，需优化嵌入表达与融合机制提升可靠性。**

- **链接: [http://arxiv.org/pdf/2505.13069v1](http://arxiv.org/pdf/2505.13069v1)**

> **作者:** Ambre Marie; Ilias Maoudj; Guillaume Dardenne; Gwenolé Quellec
>
> **备注:** Submitted to the SpeechWellness Challenge at Interspeech 2025; 5 pages, 2 figures, 2 tables
>
> **摘要:** The 1st SpeechWellness Challenge conveys the need for speech-based suicide risk assessment in adolescents. This study investigates a multimodal approach for this challenge, integrating automatic transcription with WhisperX, linguistic embeddings from Chinese RoBERTa, and audio embeddings from WavLM. Additionally, handcrafted acoustic features -- including MFCCs, spectral contrast, and pitch-related statistics -- were incorporated. We explored three fusion strategies: early concatenation, modality-specific processing, and weighted attention with mixup regularization. Results show that weighted attention provided the best generalization, achieving 69% accuracy on the development set, though a performance gap between development and test sets highlights generalization challenges. Our findings, strictly tied to the MINI-KID framework, emphasize the importance of refining embedding representations and fusion mechanisms to enhance classification reliability.
>
---
#### [new 035] Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP**

- **简介: 该论文研究参数高效微调任务，解决现有方法（如LoRA）在低参数量下性能不足的问题。提出WaveFT方法，通过小波变换在残差矩阵的频域学习稀疏更新，实现参数量的精细控制。相比直接权重域稀疏方法SHiRA，WaveFT在Stable Diffusion XL个性化图像生成任务中显著提升生成质量，尤其在极低参数量时效果突出。**

- **链接: [http://arxiv.org/pdf/2505.12532v1](http://arxiv.org/pdf/2505.12532v1)**

> **作者:** Ahmet Bilican; M. Akın Yılmaz; A. Murat Tekalp; R. Gökberk Cinbiş
>
> **摘要:** Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum -- ideal for extreme parameter-efficient scenarios. In order to demonstrate the effect of the wavelet transform, we compare WaveFT with a special case, called SHiRA, that entails applying sparse updates directly in the weight domain. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity.
>
---
#### [new 036] Hearing from Silence: Reasoning Audio Descriptions from Silent Videos via Vision-Language Model
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文研究多模态大模型从无声视频推理音频描述的能力（SVAD任务），解决现有方法无法获取音频描述的问题。提出构建CoT-AudioCaps数据集和思维链微调策略，增强模型跨模态推理能力，实验证明其有效提升SVAD及后续视频到音频任务的性能。**

- **链接: [http://arxiv.org/pdf/2505.13062v1](http://arxiv.org/pdf/2505.13062v1)**

> **作者:** Yong Ren; Chenxing Li; Le Xu; Hao Gu; Duzhen Zhang; Yujie Chen; Manjie Xu; Ruibo Fu; Shan Yang; Dong Yu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Humans can intuitively infer sounds from silent videos, but whether multimodal large language models can perform modal-mismatch reasoning without accessing target modalities remains relatively unexplored. Current text-assisted-video-to-audio (VT2A) methods excel in video foley tasks but struggle to acquire audio descriptions during inference. We introduce the task of Reasoning Audio Descriptions from Silent Videos (SVAD) to address this challenge and investigate vision-language models' (VLMs) capabilities on this task. To further enhance the VLMs' reasoning capacity for the SVAD task, we construct a CoT-AudioCaps dataset and propose a Chain-of-Thought-based supervised fine-tuning strategy. Experiments on SVAD and subsequent VT2A tasks demonstrate our method's effectiveness in two key aspects: significantly improving VLMs' modal-mismatch reasoning for SVAD and effectively addressing the challenge of acquiring audio descriptions during VT2A inference.
>
---
## 更新

#### [replaced 001] SSR: Alignment-Aware Modality Connector for Speech Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.00168v2](http://arxiv.org/pdf/2410.00168v2)**

> **作者:** Weiting Tan; Hirofumi Inaguma; Ning Dong; Paden Tomasello; Xutai Ma
>
> **备注:** IWSLT 2025
>
> **摘要:** Fusing speech into pre-trained language model (SpeechLM) usually suffers from inefficient encoding of long-form speech and catastrophic forgetting of pre-trained text modality. We propose SSR-Connector (Segmented Speech Representation Connector) for better modality fusion. Leveraging speech-text alignments, our approach segments and compresses speech features to match the granularity of text embeddings. Additionally, we introduce a two-stage training pipeline that includes the distillation and fine-tuning phases to mitigate catastrophic forgetting. SSR-Connector outperforms existing mechanism for speech-text modality fusion, consistently achieving better speech understanding (e.g., +10 accuracy on StoryCloze and +20 on Speech-MMLU) while preserving pre-trained text ability.
>
---
#### [replaced 002] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.10362v3](http://arxiv.org/pdf/2502.10362v3)**

> **作者:** Shangda Wu; Zhancheng Guo; Ruibin Yuan; Junyan Jiang; Seungheon Doh; Gus Xia; Juhan Nam; Xiaobing Li; Feng Yu; Maosong Sun
>
> **备注:** 20 pages, 8 figures, 12 tables, accepted by ACL 2025
>
> **摘要:** CLaMP 3 is a unified framework developed to address challenges of cross-modal and cross-lingual generalization in music information retrieval. Using contrastive learning, it aligns all major music modalities--including sheet music, performance signals, and audio recordings--with multilingual text in a shared representation space, enabling retrieval across unaligned modalities with text as a bridge. It features a multilingual text encoder adaptable to unseen languages, exhibiting strong cross-lingual generalization. Leveraging retrieval-augmented generation, we curated M4-RAG, a web-scale dataset consisting of 2.31 million music-text pairs. This dataset is enriched with detailed metadata that represents a wide array of global musical traditions. To advance future research, we release WikiMT-X, a benchmark comprising 1,000 triplets of sheet music, audio, and richly varied text descriptions. Experiments show that CLaMP 3 achieves state-of-the-art performance on multiple MIR tasks, significantly surpassing previous strong baselines and demonstrating excellent generalization in multimodal and multilingual music contexts.
>
---
#### [replaced 003] EMelodyGen: Emotion-Conditioned Melody Generation in ABC Notation with the Musical Feature Template
- **分类: cs.IR; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.13259v3](http://arxiv.org/pdf/2309.13259v3)**

> **作者:** Monan Zhou; Xiaobing Li; Feng Yu; Wei Li
>
> **备注:** 6 pages, 4 figures, accepted by ICMEW2025
>
> **摘要:** The EMelodyGen system focuses on emotional melody generation in ABC notation controlled by the musical feature template. Owing to the scarcity of well-structured and emotionally labeled sheet music, we designed a template for controlling emotional melody generation by statistical correlations between musical features and emotion labels derived from small-scale emotional symbolic music datasets and music psychology conclusions. We then automatically annotated a large, well-structured sheet music collection with rough emotional labels by the template, converted them into ABC notation, and reduced label imbalance by data augmentation, resulting in a dataset named Rough4Q. Our system backbone pre-trained on Rough4Q can achieve up to 99% music21 parsing rate and melodies generated by our template can lead to a 91% alignment on emotional expressions in blind listening tests. Ablation studies further validated the effectiveness of the feature controls in the template. Available code and demos are at https://github.com/monetjoe/EMelodyGen.
>
---
#### [replaced 004] BAT: Learning to Reason about Spatial Sounds with Large Language Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2402.01591v3](http://arxiv.org/pdf/2402.01591v3)**

> **作者:** Zhisheng Zheng; Puyuan Peng; Ziyang Ma; Xie Chen; Eunsol Choi; David Harwath
>
> **备注:** Accepted to ICML 2024. Our demo, dataset, code and model weights are available at: https://zhishengzheng.com/bat
>
> **摘要:** Spatial sound reasoning is a fundamental human skill, enabling us to navigate and interpret our surroundings based on sound. In this paper we present BAT, which combines the spatial sound perception ability of a binaural acoustic scene analysis model with the natural language reasoning capabilities of a large language model (LLM) to replicate this innate ability. To address the lack of existing datasets of in-the-wild spatial sounds, we synthesized a binaural audio dataset using AudioSet and SoundSpaces 2.0. Next, we developed SpatialSoundQA, a spatial sound-based question-answering dataset, offering a range of QA tasks that train BAT in various aspects of spatial sound perception and reasoning. The acoustic front end encoder of BAT is a novel spatial audio encoder named Spatial Audio Spectrogram Transformer, or Spatial-AST, which by itself achieves strong performance across sound event detection, spatial localization, and distance estimation. By integrating Spatial-AST with LLaMA-2 7B model, BAT transcends standard Sound Event Localization and Detection (SELD) tasks, enabling the model to reason about the relationships between the sounds in its environment. Our experiments demonstrate BAT's superior performance on both spatial sound perception and reasoning, showcasing the immense potential of LLMs in navigating and interpreting complex spatial audio environments.
>
---
#### [replaced 005] Audio xLSTMs: Learning Self-Supervised Audio Representations with xLSTMs
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.16568v3](http://arxiv.org/pdf/2408.16568v3)**

> **作者:** Sarthak Yadav; Sergios Theodoridis; Zheng-Hua Tan
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** While the transformer has emerged as the eminent neural architecture, several independent lines of research have emerged to address its limitations. Recurrent neural approaches have observed a lot of renewed interest, including the extended long short-term memory (xLSTM) architecture, which reinvigorates the original LSTM. However, while xLSTMs have shown competitive performance compared to the transformer, their viability for learning self-supervised general-purpose audio representations has not been evaluated. This work proposes Audio xLSTM (AxLSTM), an approach for learning audio representations from masked spectrogram patches in a self-supervised setting. Pretrained on the AudioSet dataset, the proposed AxLSTM models outperform comparable self-supervised audio spectrogram transformer (SSAST) baselines by up to 25% in relative performance across a set of ten diverse downstream tasks while having up to 45% fewer parameters.
>
---
#### [replaced 006] USpeech: Ultrasound-Enhanced Speech with Minimal Human Effort via Cross-Modal Synthesis
- **分类: cs.SD; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.22076v2](http://arxiv.org/pdf/2410.22076v2)**

> **作者:** Luca Jiang-Tao Yu; Running Zhao; Sijie Ji; Edith C. H. Ngai; Chenshu Wu
>
> **备注:** Accepted by Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (ACM IMWUT/UbiComp 2025)
>
> **摘要:** Speech enhancement is crucial for ubiquitous human-computer interaction. Recently, ultrasound-based acoustic sensing has emerged as an attractive choice for speech enhancement because of its superior ubiquity and performance. However, due to inevitable interference from unexpected and unintended sources during audio-ultrasound data acquisition, existing solutions rely heavily on human effort for data collection and processing. This leads to significant data scarcity that limits the full potential of ultrasound-based speech enhancement. To address this, we propose USpeech, a cross-modal ultrasound synthesis framework for speech enhancement with minimal human effort. At its core is a two-stage framework that establishes the correspondence between visual and ultrasonic modalities by leveraging audio as a bridge. This approach overcomes challenges from the lack of paired video-ultrasound datasets and the inherent heterogeneity between video and ultrasound data. Our framework incorporates contrastive video-audio pre-training to project modalities into a shared semantic space and employs an audio-ultrasound encoder-decoder for ultrasound synthesis. We then present a speech enhancement network that enhances speech in the time-frequency domain and recovers the clean speech waveform via a neural vocoder. Comprehensive experiments show USpeech achieves remarkable performance using synthetic ultrasound data comparable to physical data, outperforming state-of-the-art ultrasound-based speech enhancement baselines. USpeech is open-sourced at https://github.com/aiot-lab/USpeech/.
>
---
#### [replaced 007] Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.03612v2](http://arxiv.org/pdf/2501.03612v2)**

> **作者:** Bang Zeng; Ming Li
>
> **备注:** Accepted by Computer Speech and Language (CSL)
>
> **摘要:** Determining 'who spoke what and when' remains challenging in real-world applications. In typical scenarios, Speaker Diarization (SD) is employed to address the problem of 'who spoke when,' while Target Speaker Extraction (TSE) or Target Speaker Automatic Speech Recognition (TSASR) techniques are utilized to resolve the issue of 'who spoke what.' Although some works have achieved promising results by combining SD and TSE systems, inconsistencies remain between SD and TSE regarding both output inconsistency and scenario mismatch. To address these limitations, we propose a Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection (USEF-TP) model that jointly performs TSE and Personal Voice Activity Detection (PVAD). USEF-TP leverages frame-level features obtained through a cross-attention mechanism as speaker-related features instead of using speaker embeddings as in traditional approaches. Additionally, a multi-task learning algorithm with a scenario-aware differentiated loss function is applied to ensure robust performance across various levels of speaker overlap. The experimental results show that our proposed USEF-TP model achieves superior performance in TSE and PVAD tasks on the LibriMix and SparseLibriMix datasets. The results on the CALLHOME dataset demonstrate the competitive performance of our model on real recordings.
>
---
#### [replaced 008] M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis
- **分类: cs.GR; cs.AI; cs.CV; cs.SD; eess.AS; I.3.6**

- **链接: [http://arxiv.org/pdf/2505.08293v2](http://arxiv.org/pdf/2505.08293v2)**

> **作者:** Zhizhuo Yin; Yuk Hang Tsui; Pan Hui
>
> **备注:** 9 Pages, 4 figures
>
> **摘要:** Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures.
>
---
#### [replaced 009] USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.02615v2](http://arxiv.org/pdf/2409.02615v2)**

> **作者:** Bang Zeng; Ming Li
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Target speaker extraction aims to separate the voice of a specific speaker from mixed speech. Traditionally, this process has relied on extracting a speaker embedding from a reference speech, in which a speaker recognition model is required. However, identifying an appropriate speaker recognition model can be challenging, and using the target speaker embedding as reference information may not be optimal for target speaker extraction tasks. This paper introduces a Universal Speaker Embedding-Free Target Speaker Extraction (USEF-TSE) framework that operates without relying on speaker embeddings. USEF-TSE utilizes a multi-head cross-attention mechanism as a frame-level target speaker feature extractor. This innovative approach allows mainstream speaker extraction solutions to bypass the dependency on speaker recognition models and better leverage the information available in the enrollment speech, including speaker characteristics and contextual details. Additionally, USEF-TSE can seamlessly integrate with other time-domain or time-frequency domain speech separation models to achieve effective speaker extraction. Experimental results show that our proposed method achieves state-of-the-art (SOTA) performance in terms of Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) on the WSJ0-2mix, WHAM!, and WHAMR! datasets, which are standard benchmarks for monaural anechoic, noisy and noisy-reverberant two-speaker speech separation and speaker extraction. The results on the LibriMix and the blind test set of the ICASSP 2023 DNS Challenge demonstrate that the model performs well on more diverse and out-of-domain data. For access to the source code, please visit: https://github.com/ZBang/USEF-TSE.
>
---
#### [replaced 010] Streaming Sequence Transduction through Dynamic Compression
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.01172v2](http://arxiv.org/pdf/2402.01172v2)**

> **作者:** Weiting Tan; Yunmo Chen; Tongfei Chen; Guanghui Qin; Haoran Xu; Heidi C. Zhang; Benjamin Van Durme; Philipp Koehn
>
> **备注:** IWSLT 2025
>
> **摘要:** We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
>
---
#### [replaced 011] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.14971v2](http://arxiv.org/pdf/2410.14971v2)**

> **作者:** Jilong Li; Zhenxi Song; Jiaqi Wang; Meishan Zhang; Honghai Liu; Min Zhang; Zhiguo Zhang
>
> **摘要:** Current EEG/MEG-to-text decoding systems suffer from three key limitations: (1) reliance on teacher-forcing methods, which compromises robustness during inference, (2) sensitivity to session-specific noise, hindering generalization across subjects, and (3) misalignment between brain signals and linguistic representations due to pre-trained language model over-dominance. To overcome these challenges, we propose BrainECHO (Brain signal decoding via vEctor-quantized speCtrogram reconstruction for WHisper-enhanced text generatiOn), a multi-stage framework that employs decoupled representation learning to achieve state-of-the-art performance on both EEG and MEG datasets. Specifically, BrainECHO consists of three stages: (1) Discrete autoencoding, which transforms continuous Mel spectrograms into a finite set of high-quality discrete representations for subsequent stages. (2) Frozen alignment, where brain signal embeddings are mapped to corresponding Mel spectrogram embeddings in a frozen latent space, effectively filtering session-specific noise through vector-quantized reconstruction, yielding a 3.65% improvement in BLEU-4 score. (3) Constrained decoding fine-tuning, which leverages the pre-trained Whisper model for audio-to-text translation, balancing signal adaptation with knowledge preservation, and achieving 74%-89% decoding BLEU scores without excessive reliance on teacher forcing. BrainECHO demonstrates robustness across sentence, session, and subject-independent conditions, passing Gaussian noise tests and showcasing its potential for enhancing language-based brain-computer interfaces.
>
---
#### [replaced 012] ArrayDPS: Unsupervised Blind Speech Separation with a Diffusion Prior
- **分类: eess.AS; cs.LG; cs.MM; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.05657v2](http://arxiv.org/pdf/2505.05657v2)**

> **作者:** Zhongweiyang Xu; Xulin Fan; Zhong-Qiu Wang; Xilin Jiang; Romit Roy Choudhury
>
> **备注:** Paper Accepted at ICML2025 Demo: https://arraydps.github.io/ArrayDPSDemo/ Code: https://github.com/ArrayDPS/ArrayDPS
>
> **摘要:** Blind Speech Separation (BSS) aims to separate multiple speech sources from audio mixtures recorded by a microphone array. The problem is challenging because it is a blind inverse problem, i.e., the microphone array geometry, the room impulse response (RIR), and the speech sources, are all unknown. We propose ArrayDPS to solve the BSS problem in an unsupervised, array-agnostic, and generative manner. The core idea builds on diffusion posterior sampling (DPS), but unlike DPS where the likelihood is tractable, ArrayDPS must approximate the likelihood by formulating a separate optimization problem. The solution to the optimization approximates room acoustics and the relative transfer functions between microphones. These approximations, along with the diffusion priors, iterate through the ArrayDPS sampling process and ultimately yield separated voice sources. We only need a simple single-speaker speech diffusion model as a prior along with the mixtures recorded at the microphones; no microphone array information is necessary. Evaluation results show that ArrayDPS outperforms all baseline unsupervised methods while being comparable to supervised methods in terms of SDR. Audio demos are provided at: https://arraydps.github.io/ArrayDPSDemo/.
>
---
#### [replaced 013] An interpretable speech foundation model for depression detection by revealing prediction-relevant acoustic features from long speech
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.03138v3](http://arxiv.org/pdf/2406.03138v3)**

> **作者:** Qingkun Deng; Saturnino Luz; Sofia de la Fuente Garcia
>
> **备注:** 5 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2309.13476
>
> **摘要:** Speech-based depression detection tools could aid early screening. Here, we propose an interpretable speech foundation model approach to enhance the clinical applicability of such tools. We introduce a speech-level Audio Spectrogram Transformer (AST) to detect depression using long-duration speech instead of short segments, along with a novel interpretation method that reveals prediction-relevant acoustic features for clinician interpretation. Our experiments show the proposed model outperforms a segment-level AST, highlighting the impact of segment-level labelling noise and the advantage of leveraging longer speech duration for more reliable depression detection. Through interpretation, we observe our model identifies reduced loudness and F0 as relevant depression signals, aligning with documented clinical findings. This interpretability supports a responsible AI approach for speech-based depression detection, rendering such tools more clinically applicable.
>
---
