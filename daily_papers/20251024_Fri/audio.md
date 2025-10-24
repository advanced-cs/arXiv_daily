# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] R2-SVC: Towards Real-World Robust and Expressive Zero-shot Singing Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对真实场景下歌唱语音转换（SVC）中的噪声鲁棒性与表现力不足问题，提出R2-SVC框架。通过模拟噪声与基频扰动增强鲁棒性，融合多源声乐数据优化音色与风格表征，并引入神经声源-滤波器模型提升音质自然度与可控性，显著提升复杂环境下的转换性能。**

- **链接: [http://arxiv.org/pdf/2510.20677v1](http://arxiv.org/pdf/2510.20677v1)**

> **作者:** Junjie Zheng; Gongyu Chen; Chaofan Ding; Zihao Chen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** In real-world singing voice conversion (SVC) applications, environmental noise and the demand for expressive output pose significant challenges. Conventional methods, however, are typically designed without accounting for real deployment scenarios, as both training and inference usually rely on clean data. This mismatch hinders practical use, given the inevitable presence of diverse noise sources and artifacts from music separation. To tackle these issues, we propose R2-SVC, a robust and expressive SVC framework. First, we introduce simulation-based robustness enhancement through random fundamental frequency ($F_0$) perturbations and music separation artifact simulations (e.g., reverberation, echo), substantially improving performance under noisy conditions. Second, we enrich speaker representation using domain-specific singing data: alongside clean vocals, we incorporate DNSMOS-filtered separated vocals and public singing corpora, enabling the model to preserve speaker timbre while capturing singing style nuances. Third, we integrate the Neural Source-Filter (NSF) model to explicitly represent harmonic and noise components, enhancing the naturalness and controllability of converted singing. R2-SVC achieves state-of-the-art results on multiple SVC benchmarks under both clean and noisy conditions.
>
---
#### [new 002] Speaking Clearly: A Simplified Whisper-Based Codec for Low-Bitrate Speech Coding
- **分类: cs.SD**

- **简介: 该论文提出一种基于Whisper的低比特率语音编码方法SimWhisper-Codec，属于语音编码任务。针对传统编码中声学保真与语义保留的矛盾，提出从语义模型出发的简化架构，利用冻结的简化Whisper编码器，在无需外部监督下实现高效语义与声学双重保真，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.20504v1](http://arxiv.org/pdf/2510.20504v1)**

> **作者:** Xin Zhang; Lin Li; Xiangni Lu; Jianquan Liu; Kong Aik Lee
>
> **备注:** 5 pages, 3 figures, 2 tables
>
> **摘要:** Speech codecs serve as bridges between continuous speech signals and large language models, yet face an inherent conflict between acoustic fidelity and semantic preservation. To mitigate this conflict, prevailing methods augment acoustic codecs with complex semantic supervision. We explore the opposite direction: a semantic-first approach that starts from a semantically-capable model and adapts it for high-fidelity acoustic reconstruction. Through empirical analysis, we discover that targeted architectural simplification can unlock the acoustic modeling potential of Whisper, a text-aligned Automatic Speech Recognition (ASR) model. Based on this finding, we propose SimWhisper-Codec, a novel codec that balances the semantic and acoustic preservation by leveraging a frozen, simplified Whisper encoder without requiring external supervision. Experimental results demonstrate that SimWhisper-Codec achieves superior performance in both semantic preservation and acoustic quality compared to semantically-supervised codecs such as Mimi Codec and SpeechTokenizer at similar bitrates, validating the effectiveness of our semantic-first approach. Code is available at https://github.com/ZhangXinWhut/SimWhisper-Codec.
>
---
#### [new 003] Neural Directional Filtering with Configurable Directivity Pattern at Inference
- **分类: eess.AS**

- **简介: 该论文提出一种可配置方向性模式的神经方向滤波方法（UNDF），旨在实现推理时用户自定义方向性模式的音频空间滤波。通过引入特征逐项线性调制（FiLM）架构，使模型能泛化至未见模式，并支持不规则形状逼近与方向调整，显著优于传统方法。**

- **链接: [http://arxiv.org/pdf/2510.20253v1](http://arxiv.org/pdf/2510.20253v1)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Emanuël A. P. Habets
>
> **摘要:** Spatial filtering with a desired directivity pattern is advantageous for many audio applications. In this work, we propose neural directional filtering with user-defined directivity patterns (UNDF), which enables spatial filtering based on directivity patterns that users can define during inference. To achieve this, we propose a DNN architecture that integrates feature-wise linear modulation (FiLM), allowing user-defined patterns to serve as conditioning inputs. Through analysis, we demonstrate that the FiLM-based architecture enables the UNDF to generalize to unseen user-defined patterns during interference with higher directivities, scaling variations, and different steering directions. Furthermore, we progressively refine training strategies to enhance pattern approximation and enable UNDF to approximate irregular shapes. Lastly, experimental comparisons show that UNDF outperforms conventional methods.
>
---
#### [new 004] Vox-Evaluator: Enhancing Stability and Fidelity for Zero-shot TTS with A Multi-Level Evaluator
- **分类: cs.SD**

- **简介: 该论文针对零样本文本到语音合成（TTS）中的稳定性与保真度问题，提出Vox-Evaluator多级评估器。通过识别错误语音段、自动掩码并重生成，提升语音质量与鲁棒性，并用于偏好对齐。构建了带细粒度标注的合成数据集，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.20210v1](http://arxiv.org/pdf/2510.20210v1)**

> **作者:** Hualei Wang; Na Li; Chuke Wang; Shu Wu; Zhifeng Li; Dong Yu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent advances in zero-shot text-to-speech (TTS), driven by language models, diffusion models and masked generation, have achieved impressive naturalness in speech synthesis. Nevertheless, stability and fidelity remain key challenges, manifesting as mispronunciations, audible noise, and quality degradation. To address these issues, we introduce Vox-Evaluator, a multi-level evaluator designed to guide the correction of erroneous speech segments and preference alignment for TTS systems. It is capable of identifying the temporal boundaries of erroneous segments and providing a holistic quality assessment of the generated speech. Specifically, to refine erroneous segments and enhance the robustness of the zero-shot TTS model, we propose to automatically identify acoustic errors with the evaluator, mask the erroneous segments, and finally regenerate speech conditioning on the correct portions. In addition, the fine-gained information obtained from Vox-Evaluator can guide the preference alignment for TTS model, thereby reducing the bad cases in speech synthesis. Due to the lack of suitable training datasets for the Vox-Evaluator, we also constructed a synthesized text-speech dataset annotated with fine-grained pronunciation errors or audio quality issues. The experimental results demonstrate the effectiveness of the proposed Vox-Evaluator in enhancing the stability and fidelity of TTS systems through the speech correction mechanism and preference optimization. The demos are shown.
>
---
#### [new 005] Controllable Embedding Transformation for Mood-Guided Music Retrieval
- **分类: cs.SD**

- **简介: 该论文针对音乐推荐中的可控检索任务，解决现有嵌入表示难以独立调控单一属性（如仅改变情绪）的问题。提出基于情绪引导的嵌入变换框架，通过采样代理目标与轻量翻译模型，实现情绪迁移同时保留风格、乐器等特征，显著优于无训练基线。**

- **链接: [http://arxiv.org/pdf/2510.20759v1](http://arxiv.org/pdf/2510.20759v1)**

> **作者:** Julia Wilkins; Jaehun Kim; Matthew E. P. Davies; Juan Pablo Bello; Matthew C. McCallum
>
> **备注:** Preprint; under review
>
> **摘要:** Music representations are the backbone of modern recommendation systems, powering playlist generation, similarity search, and personalized discovery. Yet most embeddings offer little control for adjusting a single musical attribute, e.g., changing only the mood of a track while preserving its genre or instrumentation. In this work, we address the problem of controllable music retrieval through embedding-based transformation, where the objective is to retrieve songs that remain similar to a seed track but are modified along one chosen dimension. We propose a novel framework for mood-guided music embedding transformation, which learns a mapping from a seed audio embedding to a target embedding guided by mood labels, while preserving other musical attributes. Because mood cannot be directly altered in the seed audio, we introduce a sampling mechanism that retrieves proxy targets to balance diversity with similarity to the seed. We train a lightweight translation model using this sampling strategy and introduce a novel joint objective that encourages transformation and information preservation. Extensive experiments on two datasets show strong mood transformation performance while retaining genre and instrumentation far better than training-free baselines, establishing controllable embedding transformation as a promising paradigm for personalized music retrieval.
>
---
#### [new 006] UniSE: A Unified Framework for Decoder-only Autoregressive LM-based Speech Enhancement
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出UniSE，一种基于解码器的自回归语言模型框架，用于统一处理语音增强中的语音恢复、目标说话人提取和语音分离任务。通过将输入语音特征作为条件，生成目标语音的离散标记，实现多任务兼容，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.20441v1](http://arxiv.org/pdf/2510.20441v1)**

> **作者:** Haoyin Yan; Chengwei Liu; Shaofei Xue; Xiaotao Liang; Zheng Xue
>
> **备注:** 5 pages, submitted to ICASSP 2026
>
> **摘要:** The development of neural audio codecs (NACs) has largely promoted applications of language models (LMs) to speech processing and understanding. However, there lacks the verification on the effectiveness of autoregressive (AR) LMbased models in unifying different sub-tasks of speech enhancement (SE). In this work, we propose UniSE, a unified decoder-only LM-based framework to handle different SE tasks including speech restoration, target speaker extraction and speech separation. It takes input speech features as conditions and generates discrete tokens of the target speech using AR modeling, which facilitates a compatibility between distinct learning patterns of multiple tasks. Experiments on several benchmarks indicate the proposed UniSE can achieve competitive performance compared to discriminative and generative baselines, showing the capacity of LMs in unifying SE tasks. The demo page is available here: https://github.com/hyyan2k/UniSE.
>
---
#### [new 007] Resounding Acoustic Fields with Reciprocity
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文提出“resounding”任务，旨在从稀疏测量点估计任意声源位置的房间冲激响应。利用声学互易性，通过交换声源与听者位置生成密集虚拟数据，提出Versa方法，结合自监督学习解决增益不一致问题，显著提升音场建模精度与沉浸式听觉体验。**

- **链接: [http://arxiv.org/pdf/2510.20602v1](http://arxiv.org/pdf/2510.20602v1)**

> **作者:** Zitong Lan; Yiduo Hao; Mingmin Zhao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Achieving immersive auditory experiences in virtual environments requires flexible sound modeling that supports dynamic source positions. In this paper, we introduce a task called resounding, which aims to estimate room impulse responses at arbitrary emitter location from a sparse set of measured emitter positions, analogous to the relighting problem in vision. We leverage the reciprocity property and introduce Versa, a physics-inspired approach to facilitating acoustic field learning. Our method creates physically valid samples with dense virtual emitter positions by exchanging emitter and listener poses. We also identify challenges in deploying reciprocity due to emitter/listener gain patterns and propose a self-supervised learning approach to address them. Results show that Versa substantially improve the performance of acoustic field learning on both simulated and real-world datasets across different metrics. Perceptual user studies show that Versa can greatly improve the immersive spatial sound experience. Code, dataset and demo videos are available on the project website: https://waves.seas.upenn.edu/projects/versa.
>
---
#### [new 008] Decoding the Ear: A Framework for Objectifying Expressiveness from Human Preference Through Efficient Alignment
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文针对语音合成中表达性不足的问题，提出DeEAR框架，通过情感、韵律、自然性三维度将人类偏好转化为客观评分，实现高效精准评估。仅用500样本即达高相关性（SRCC=0.86），支持模型对比与数据优化，显著提升合成语音表达性。**

- **链接: [http://arxiv.org/pdf/2510.20513v1](http://arxiv.org/pdf/2510.20513v1)**

> **作者:** Zhiyu Lin; Jingwen Yang; Jiale Zhao; Meng Liu; Sunzhu Li; Benyou Wang
>
> **备注:** Submitted to ICASSP 2026. Demos and codes are available at https://github.com/FreedomIntelligence/ExpressiveSpeech
>
> **摘要:** Recent speech-to-speech (S2S) models generate intelligible speech but still lack natural expressiveness, largely due to the absence of a reliable evaluation metric. Existing approaches, such as subjective MOS ratings, low-level acoustic features, and emotion recognition are costly, limited, or incomplete. To address this, we present DeEAR (Decoding the Expressive Preference of eAR), a framework that converts human preference for speech expressiveness into an objective score. Grounded in phonetics and psychology, DeEAR evaluates speech across three dimensions: Emotion, Prosody, and Spontaneity, achieving strong alignment with human perception (Spearman's Rank Correlation Coefficient, SRCC = 0.86) using fewer than 500 annotated samples. Beyond reliable scoring, DeEAR enables fair benchmarking and targeted data curation. It not only distinguishes expressiveness gaps across S2S models but also selects 14K expressive utterances to form ExpressiveSpeech, which improves the expressive score (from 2.0 to 23.4 on a 100-point scale) of S2S models. Demos and codes are available at https://github.com/FreedomIntelligence/ExpressiveSpeech
>
---
#### [new 009] Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy
- **分类: eess.SP; cs.IT; eess.AS; math.IT; stat.ME; stat.ML**

- **简介: 该论文针对高维时间序列复杂性排序问题，提出基于预测误差协方差的条件微分熵上界方法。利用信息论边界与哈达玛不等式，构建可计算的复杂性代理指标，通过合成数据实验验证其有效性，实现了无需已知分布的高效复杂性排序。**

- **链接: [http://arxiv.org/pdf/2510.20551v1](http://arxiv.org/pdf/2510.20551v1)**

> **作者:** Jacob Ayers; Richard Hahnloser; Julia Ulrich; Lothar Sebastian Krapp; Remo Nitschke; Sabine Stoll; Balthasar Bickel; Reinhard Furrer
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Conditional differential entropy provides an intuitive measure for relatively ranking time-series complexity by quantifying uncertainty in future observations given past context. However, its direct computation for high-dimensional processes from unknown distributions is often intractable. This paper builds on the information theoretic prediction error bounds established by Fang et al. \cite{fang2019generic}, which demonstrate that the conditional differential entropy \textbf{$h(X_k \mid X_{k-1},...,X_{k-m})$} is upper bounded by a function of the determinant of the covariance matrix of next-step prediction errors for any next step prediction model. We add to this theoretical framework by further increasing this bound by leveraging Hadamard's inequality and the positive semi-definite property of covariance matrices. To see if these bounds can be used to rank the complexity of time series, we conducted two synthetic experiments: (1) controlled linear autoregressive processes with additive Gaussian noise, where we compare ordinary least squares prediction error entropy proxies to the true entropies of various additive noises, and (2) a complexity ranking task of bio-inspired synthetic audio data with unknown entropy, where neural network prediction errors are used to recover the known complexity ordering. This framework provides a computationally tractable method for time-series complexity ranking using prediction errors from next-step prediction models, that maintains a theoretical foundation in information theory.
>
---
#### [new 010] SpeechAgent: An End-to-End Mobile Infrastructure for Speech Impairment Assistance
- **分类: eess.SY; cs.SD; cs.SY**

- **简介: 该论文提出SpeechAgent，一种面向语音障碍者的端到端移动辅助系统。针对现有技术难以实现实时、个性化语音通信的问题，融合大语言模型与先进语音处理，实现低延迟、高精度的语音识别与合成，支持多种障碍类型，具备实际部署可行性。**

- **链接: [http://arxiv.org/pdf/2510.20113v1](http://arxiv.org/pdf/2510.20113v1)**

> **作者:** Haowei Lou; Chengkai Huang; Hye-young Paik; Yongquan Hu; Aaron Quigley; Wen Hu; Lina Yao
>
> **摘要:** Speech is essential for human communication, yet millions of people face impairments such as dysarthria, stuttering, and aphasia conditions that often lead to social isolation and reduced participation. Despite recent progress in automatic speech recognition (ASR) and text-to-speech (TTS) technologies, accessible web and mobile infrastructures for users with impaired speech remain limited, hindering the practical adoption of these advances in daily communication. To bridge this gap, we present SpeechAgent, a mobile SpeechAgent designed to facilitate people with speech impairments in everyday communication. The system integrates large language model (LLM)- driven reasoning with advanced speech processing modules, providing adaptive support tailored to diverse impairment types. To ensure real-world practicality, we develop a structured deployment pipeline that enables real-time speech processing on mobile and edge devices, achieving imperceptible latency while maintaining high accuracy and speech quality. Evaluation on real-world impaired speech datasets and edge-device latency profiling confirms that SpeechAgent delivers both effective and user-friendly performance, demonstrating its feasibility for personalized, day-to-day assistive communication.
>
---
#### [new 011] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era
- **分类: cs.IR; cs.HC; cs.MA; cs.SD**

- **简介: 该论文针对生成式AI在音乐创作中导致的版权归属不清、收益分配不公问题，提出基于块级检索与代理协作的音乐AI代理架构。通过将创作过程分解为可追溯的“区块”并嵌入溯源层，实现细粒度版权追踪与实时结算，构建公平的后流媒体时代音乐生态。**

- **链接: [http://arxiv.org/pdf/2510.20276v1](http://arxiv.org/pdf/2510.20276v1)**

> **作者:** Wonil Kim; Hyeongseok Wi; Seungsoon Park; Taejun Kim; Sangeun Keum; Keunhyoung Kim; Taewan Kim; Jongmin Jung; Taehyoung Kim; Gaetan Guerrero; Mael Le Goff; Julie Po; Dongjoo Moon; Juhan Nam; Jongpil Lee
>
> **备注:** Accepted to the NeurIPS 2025 AI4Music Workshop
>
> **摘要:** Generative AI is reshaping music creation, but its rapid growth exposes structural gaps in attribution, rights management, and economic models. Unlike past media shifts, from live performance to recordings, downloads, and streaming, AI transforms the entire lifecycle of music, collapsing boundaries between creation, distribution, and monetization. However, existing streaming systems, with opaque and concentrated royalty flows, are ill-equipped to handle the scale and complexity of AI-driven production. We propose a content-based Music AI Agent architecture that embeds attribution directly into the creative workflow through block-level retrieval and agentic orchestration. Designed for iterative, session-based interaction, the system organizes music into granular components (Blocks) stored in BlockDB; each use triggers an Attribution Layer event for transparent provenance and real-time settlement. This framework reframes AI from a generative tool into infrastructure for a Fair AI Media Platform. By enabling fine-grained attribution, equitable compensation, and participatory engagement, it points toward a post-streaming paradigm where music functions not as a static catalog but as a collaborative and adaptive ecosystem.
>
---
## 更新

#### [replaced 001] LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.15299v4](http://arxiv.org/pdf/2412.15299v4)**

> **作者:** Sangmin Lee; Woo-Jin Chung; Hong-Goo Kang
>
> **备注:** Accepted to AAAI 2025 (Oral Presentation)
>
> **摘要:** Building a universal multilingual automatic speech recognition (ASR) model that performs equitably across languages has long been a challenge due to its inherent difficulties. To address this task we introduce a Language-Agnostic Multilingual ASR pipeline through orthography Unification and language-specific Transliteration (LAMA-UT). LAMA-UT operates without any language-specific modules while matching the performance of state-of-the-art models trained on a minimal amount of data. Our pipeline consists of two key steps. First, we utilize a universal transcription generator to unify orthographic features into Romanized form and capture common phonetic characteristics across diverse languages. Second, we utilize a universal converter to transform these universal transcriptions into language-specific ones. In experiments, we demonstrate the effectiveness of our proposed method leveraging universal transcriptions for massively multilingual ASR. Our pipeline achieves a relative error reduction rate of 45% when compared to Whisper and performs comparably to MMS, despite being trained on only 0.1% of Whisper's training data. Furthermore, our pipeline does not rely on any language-specific modules. However, it performs on par with zero-shot ASR approaches which utilize additional language-specific lexicons and language models. We expect this framework to serve as a cornerstone for flexible multilingual ASR systems that are generalizable even to unseen languages.
>
---
#### [replaced 002] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19634v2](http://arxiv.org/pdf/2507.19634v2)**

> **作者:** Sara Papi; Maike Züfle; Marco Gaido; Beatrice Savoldi; Danni Liu; Ioannis Douros; Luisa Bentivogli; Jan Niehues
>
> **备注:** Data available at https://huggingface.co/datasets/FBK-MT/MCIF | Evaluation and baselines available at https://github.com/hlt-mt/mcif
>
> **摘要:** Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations -- hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities -- speech, vision, and text -- and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development.
>
---
#### [replaced 003] LeVo: High-Quality Song Generation with Multi-Preference Alignment
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07520v3](http://arxiv.org/pdf/2506.07520v3)**

> **作者:** Shun Lei; Yaoxun Xu; Zhiwei Lin; Huaicheng Zhang; Wei Tan; Hangting Chen; Jianwei Yu; Yixuan Zhang; Chenyu Yang; Haina Zhu; Shuai Wang; Zhiyong Wu; Dong Yu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in audio quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, a language model based framework consisting of LeLM and Music Codec. LeLM is capable of parallel modeling of two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve better vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following ability, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and post-training. Experimental results demonstrate that LeVo significantly outperforms existing open-source methods in both objective and subjective metrics, while performing competitively with industry systems. Ablation studies further justify the effectiveness of our designs. Audio examples and source code are available at https://levo-demo.github.io and https://github.com/tencent-ailab/songgeneration.
>
---
#### [replaced 004] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.12226v2](http://arxiv.org/pdf/2505.12226v2)**

> **作者:** Dong Yang; Yiyi Cai; Yuki Saito; Lixu Wang; Hiroshi Saruwatari
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We propose Shallow Flow Matching (SFM), a novel mechanism that enhances flow matching (FM)-based text-to-speech (TTS) models within a coarse-to-fine generation paradigm. Unlike conventional FM modules, which use the coarse representations from the weak generator as conditions, SFM constructs intermediate states along the FM paths from these representations. During training, we introduce an orthogonal projection method to adaptively determine the temporal position of these states, and apply a principled construction strategy based on a single-segment piecewise flow. The SFM inference starts from the intermediate state rather than pure noise, thereby focusing computation on the latter stages of the FM paths. We integrate SFM into multiple TTS models with a lightweight SFM head. Experiments demonstrate that SFM yields consistent gains in speech naturalness across both objective and subjective evaluations, and significantly accelerates inference when using adaptive-step ODE solvers. Demo and codes are available at https://ydqmkkx.github.io/SFMDemo/.
>
---
#### [replaced 005] MLMA: Towards Multilingual ASR With Mamba-based Architectures
- **分类: cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.18684v2](http://arxiv.org/pdf/2510.18684v2)**

> **作者:** Mohamed Nabih Ali; Daniele Falavigna; Alessio Brutti
>
> **备注:** The paper is under review at ICASSP 2026
>
> **摘要:** Multilingual automatic speech recognition (ASR) remains a challenging task, especially when balancing performance across high- and low-resource languages. Recent advances in sequence modeling suggest that architectures beyond Transformers may offer better scalability and efficiency. In this work, we introduce MLMA (Multilingual Language Modeling with Mamba for ASR), a new approach that leverages the Mamba architecture -- an efficient state-space model optimized for long-context sequence processing -- for multilingual ASR. Using Mamba, MLMA implicitly incorporates language-aware conditioning and shared representations to support robust recognition across diverse languages. Experiments on standard multilingual benchmarks show that MLMA achieves competitive performance compared to Transformer-based architectures. These results highlight Mamba's potential as a strong backbone for scalable, efficient, and accurate multilingual speech recognition.
>
---
