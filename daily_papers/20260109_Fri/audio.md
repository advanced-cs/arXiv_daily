# 音频 cs.SD;  eess.AS

- **最新发布 18 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Latent-Level Enhancement with Flow Matching for Robust Automatic Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决噪声环境下识别性能下降的问题。通过在推理阶段对编码器潜在表示进行优化，提出FM-Refiner模块提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.04459v1](https://arxiv.org/pdf/2601.04459v1)**

> **作者:** Da-Hee Yang; Joon-Hyuk Chang
>
> **备注:** Accepted for publication in IEEE Signal Processing Letters
>
> **摘要:** Noise-robust automatic speech recognition (ASR) has been commonly addressed by applying speech enhancement (SE) at the waveform level before recognition. However, speech-level enhancement does not always translate into consistent recognition improvements due to residual distortions and mismatches with the latent space of the ASR encoder. In this letter, we introduce a complementary strategy termed latent-level enhancement, where distorted representations are refined during ASR inference. Specifically, we propose a plug-and-play Flow Matching Refinement module (FM-Refiner) that operates on the output latents of a pretrained CTC-based ASR encoder. Trained to map imperfect latents-either directly from noisy inputs or from enhanced-but-imperfect speech-toward their clean counterparts, the FM-Refiner is applied only at inference, without fine-tuning ASR parameters. Experiments show that FM-Refiner consistently reduces word error rate, both when directly applied to noisy inputs and when combined with conventional SE front-ends. These results demonstrate that latent-level refinement via flow matching provides a lightweight and effective complement to existing SE approaches for robust ASR.
>
---
#### [new 002] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于医疗语音分析任务，解决弱监督下疾病检测问题。通过多层级数据建模，提升未标注数据利用效率，实现高效准确的病理检测。**

- **链接: [https://arxiv.org/pdf/2601.04744v1](https://arxiv.org/pdf/2601.04744v1)**

> **作者:** Xingyuan Li; Mengyue Wu
>
> **摘要:** Detecting medical conditions from speech acoustics is fundamentally a weakly-supervised learning problem: a single, often noisy, session-level label must be linked to nuanced patterns within a long, complex audio recording. This task is further hampered by severe data scarcity and the subjective nature of clinical annotations. While semi-supervised learning (SSL) offers a viable path to leverage unlabeled data, existing audio methods often fail to address the core challenge that pathological traits are not uniformly expressed in a patient's speech. We propose a novel, audio-only SSL framework that explicitly models this hierarchy by jointly learning from frame-level, segment-level, and session-level representations within unsegmented clinical dialogues. Our end-to-end approach dynamically aggregates these multi-granularity features and generates high-quality pseudo-labels to efficiently utilize unlabeled data. Extensive experiments show the framework is model-agnostic, robust across languages and conditions, and highly data-efficient-achieving, for instance, 90\% of fully-supervised performance using only 11 labeled samples. This work provides a principled approach to learning from weak, far-end supervision in medical speech analysis.
>
---
#### [new 003] SmoothSync: Dual-Stream Diffusion Transformers for Jitter-Robust Beat-Synchronized Gesture Generation from Quantized Audio
- **分类: cs.SD; cs.AI; cs.RO; eess.AS**

- **简介: 该论文属于语音同步手势生成任务，旨在解决节奏不一致、动作抖动等问题。提出SmoothSync框架，通过双流扩散Transformer和量化音频实现更稳定、多样的手势生成。**

- **链接: [https://arxiv.org/pdf/2601.04236v1](https://arxiv.org/pdf/2601.04236v1)**

> **作者:** Yujiao Jiang; Qingmin Liao; Zongqing Lu
>
> **摘要:** Co-speech gesture generation is a critical area of research aimed at synthesizing speech-synchronized human-like gestures. Existing methods often suffer from issues such as rhythmic inconsistency, motion jitter, foot sliding and limited multi-sampling diversity. In this paper, we present SmoothSync, a novel framework that leverages quantized audio tokens in a novel dual-stream Diffusion Transformer (DiT) architecture to synthesis holistic gestures and enhance sampling variation. Specifically, we (1) fuse audio-motion features via complementary transformer streams to achieve superior synchronization, (2) introduce a jitter-suppression loss to improve temporal smoothness, (3) implement probabilistic audio quantization to generate distinct gesture sequences from identical inputs. To reliably evaluate beat synchronization under jitter, we introduce Smooth-BC, a robust variant of the beat consistency metric less sensitive to motion noise. Comprehensive experiments on the BEAT2 and SHOW datasets demonstrate SmoothSync's superiority, outperforming state-of-the-art methods by -30.6% FGD, 10.3% Smooth-BC, and 8.4% Diversity on BEAT2, while reducing jitter and foot sliding by -62.9% and -17.1% respectively. The code will be released to facilitate future research.
>
---
#### [new 004] Predictive Controlled Music
- **分类: cs.SD; eess.AS; eess.SY**

- **简介: 该论文提出预测控制音乐（PCM），将模型预测控制与音乐生成结合，解决音乐算法创作问题。通过神经网络评估和优化生成乐谱，实现反馈控制的音乐生成。**

- **链接: [https://arxiv.org/pdf/2601.04221v1](https://arxiv.org/pdf/2601.04221v1)**

> **作者:** Midhun T. Augustine
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** This paper presents a new approach to algorithmic composition, called predictive controlled music (PCM), which combines model predictive control (MPC) with music generation. PCM uses dynamic models to predict and optimize the music generation process, where musical notes are computed in a manner similar to an MPC problem by optimizing a performance measure. A feedforward neural network-based assessment function is used to evaluate the generated musical score, which serves as the objective function of the PCM optimization problem. Furthermore, a recurrent neural network model is employed to capture the relationships among the variables in the musical notes, and this model is then used to define the constraints in the PCM. Similar to MPC, the proposed PCM computes musical notes in a receding-horizon manner, leading to feedback controlled prediction. Numerical examples are presented to illustrate the PCM generation method.
>
---
#### [new 005] Summary of The Inaugural Music Source Restoration Challenge
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文介绍音乐源恢复任务，旨在从混音和退化音频中恢复原始乐器音轨。通过挑战赛评估不同系统性能，分析恢复难度差异。**

- **链接: [https://arxiv.org/pdf/2601.04343v1](https://arxiv.org/pdf/2601.04343v1)**

> **作者:** Yongyi Zang; Jiarui Hai; Wanying Ge; Qiuqiang Kong; Zheqi Dai; Helin Wang; Yuki Mitsufuji; Mark D. Plumbley
>
> **摘要:** Music Source Restoration (MSR) aims to recover original, unprocessed instrument stems from professionally mixed and degraded audio, requiring the reversal of both production effects and real-world degradations. We present the inaugural MSR Challenge, which features objective evaluation on studio-produced mixtures using Multi-Mel-SNR, Zimtohrli, and FAD-CLAP, alongside subjective evaluation on real-world degraded recordings. Five teams participated in the challenge. The winning system achieved 4.46 dB Multi-Mel-SNR and 3.47 MOS-Overall, corresponding to relative improvements of 91% and 18% over the second-place system, respectively. Per-stem analysis reveals substantial variation in restoration difficulty across instruments, with bass averaging 4.59 dB across all teams, while percussion averages only 0.29 dB. The dataset, evaluation protocols, and baselines are available at https://msrchallenge.com/.
>
---
#### [new 006] When Tone and Words Disagree: Towards Robust Speech Emotion Recognition under Acoustic-Semantic Conflict
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，解决声学与语义冲突下的情感识别问题。提出FAS框架，通过分离声学与语义路径提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04564v1](https://arxiv.org/pdf/2601.04564v1)**

> **作者:** Dawei Huang; Yongjie Lv; Ruijie Xiong; Chunxiang Jin; Xiaojiang Peng
>
> **摘要:** Speech Emotion Recognition (SER) systems often assume congruence between vocal emotion and lexical semantics. However, in real-world interactions, acoustic-semantic conflict is common yet overlooked, where the emotion conveyed by tone contradicts the literal meaning of spoken words. We show that state-of-the-art SER models, including ASR-based, self-supervised learning (SSL) approaches and Audio Language Models (ALMs), suffer performance degradation under such conflicts due to semantic bias or entangled acoustic-semantic representations. To address this, we propose the Fusion Acoustic-Semantic (FAS) framework, which explicitly disentangles acoustic and semantic pathways and bridges them through a lightweight, query-based attention module. To enable systematic evaluation, we introduce the Conflict in Acoustic-Semantic Emotion (CASE), the first dataset dominated by clear and interpretable acoustic-semantic conflicts in varied scenarios. Extensive experiments demonstrate that FAS consistently outperforms existing methods in both in-domain and zero-shot settings. Notably, on the CASE benchmark, conventional SER models fail dramatically, while FAS sets a new SOTA with 59.38% accuracy. Our code and datasets is available at https://github.com/24DavidHuang/FAS.
>
---
#### [new 007] LAMB: LLM-based Audio Captioning with Modality Gap Bridging via Cauchy-Schwarz Divergence
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频描述任务，旨在解决音频与文本嵌入空间对齐问题。提出LAMB框架，通过跨模态对齐和双流适配器提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.04658v1](https://arxiv.org/pdf/2601.04658v1)**

> **作者:** Hyeongkeun Lee; Jongmin Choi; KiHyun Nam; Joon Son Chung
>
> **备注:** 5 pages, 2 figures;
>
> **摘要:** Automated Audio Captioning aims to describe the semantic content of input audio. Recent works have employed large language models (LLMs) as a text decoder to leverage their reasoning capabilities. However, prior approaches that project audio features into the LLM embedding space without considering cross-modal alignment fail to fully utilize these capabilities. To address this, we propose LAMB, an LLM-based audio captioning framework that bridges the modality gap between audio embeddings and the LLM text embedding space. LAMB incorporates a Cross-Modal Aligner that minimizes Cauchy-Schwarz divergence while maximizing mutual information, yielding tighter alignment between audio and text at both global and token levels. We further design a Two-Stream Adapter that extracts semantically enriched audio embeddings, thereby delivering richer information to the Cross-Modal Aligner. Finally, leveraging the aligned audio embeddings, a proposed Token Guide directly computes scores within the LLM text embedding space to steer the output logits of generated captions. Experimental results confirm that our framework strengthens the reasoning capabilities of the LLM decoder, achieving state-of-the-art performance on AudioCaps.
>
---
#### [new 008] FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions
- **分类: cs.SD**

- **简介: 该论文提出FlexiVoice，属于文本到语音合成任务，解决零样本语音克隆中的风格控制问题。通过自然语言指令和语音参考实现灵活风格控制。**

- **链接: [https://arxiv.org/pdf/2601.04656v1](https://arxiv.org/pdf/2601.04656v1)**

> **作者:** Dekun Chen; Xueyao Zhang; Yuancheng Wang; Kenan Dai; Li Ma; Zhizheng Wu
>
> **摘要:** This study proposes FlexiVoice, a text-to-speech (TTS) synthesis system capable of flexible style control with zero-shot voice cloning. The speaking style is controlled by a natural-language instruction and the voice timbre is provided by a speech reference in zero-shot manner. FlexiVoice is built with an LLM core, which takes text as input, and also takes an optional natural language instruction and an optional speech reference to control style and timbre, respectively. FlexiVoice is equipped with a novel Progressive Post-Training (PPT) scheme that progressively unlocks accurate and flexible controllability. In particular, it first employs Direct Preference Optimization (DPO) to enable FlexiVoice to accurately follow both natural language instruction and speech reference simultaneously. It then uses a multi-objective Group Relative Policy Optimization (GRPO) to disentangle style instruction, reference timbre, and textual content. Finally, it adapts instruction GRPO for more advanced instruction following. Experimental results show that FlexiVoice surpasses competing baselines and demonstrates strong capability in decoupling control factors. Human evaluations further confirm its naturalness, controllability, and robustness. Audio samples are available at https://flexi-voice.github.io.
>
---
#### [new 009] Defense Against Synthetic Speech: Real-Time Detection of RVC Voice Conversion Attacks
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音安全任务，旨在检测RVC生成的合成语音。通过分析音频特征，实现对AI生成语音的实时识别，提升通信安全性。**

- **链接: [https://arxiv.org/pdf/2601.04227v1](https://arxiv.org/pdf/2601.04227v1)**

> **作者:** Prajwal Chinchmalatpure; Suyash Chinchmalatpure; Siddharth Chavan
>
> **摘要:** Generative audio technologies now enable highly realistic voice cloning and real-time voice conversion, increasing the risk of impersonation, fraud, and misinformation in communication channels such as phone and video calls. This study investigates real-time detection of AI-generated speech produced using Retrieval-based Voice Conversion (RVC), evaluated on the DEEP-VOICE dataset, which includes authentic and voice-converted speech samples from multiple well-known speakers. To simulate realistic conditions, deepfake generation is applied to isolated vocal components, followed by the reintroduction of background ambiance to suppress trivial artifacts and emphasize conversion-specific cues. We frame detection as a streaming classification task by dividing audio into one-second segments, extracting time-frequency and cepstral features, and training supervised machine learning models to classify each segment as real or voice-converted. The proposed system enables low-latency inference, supporting both segment-level decisions and call-level aggregation. Experimental results show that short-window acoustic features can reliably capture discriminative patterns associated with RVC speech, even in noisy backgrounds. These findings demonstrate the feasibility of practical, real-time deepfake speech detection and underscore the importance of evaluating under realistic audio mixing conditions for robust deployment.
>
---
#### [new 010] From Imitation to Innovation: The Divergent Paths of Techno in Germany and the USA
- **分类: cs.SD; eess.AS**

- **简介: 论文分析德美两国早期浩室与科技音乐的音频特征，探讨其发展差异。属于音乐数据分析任务，旨在验证音乐风格演变的文献记载，并解释为何科技音乐在德国流行而在美国边缘化。**

- **链接: [https://arxiv.org/pdf/2601.04222v1](https://arxiv.org/pdf/2601.04222v1)**

> **作者:** Tim Ziemer; Simon Linke
>
> **摘要:** Many documentaries on early house and techno music exist. Here, protagonists from the scenes describe key elements and events that affected the evolution of the music. In the research community, there is consensus that such descriptions have to be examined critically. Yet, there have not been attempts to validate such statements on the basis of audio analyses. In this study, over 9,000 early house and techno tracks from Germany and the United States of America are analyzed using recording studio features, machine learning and inferential statistics. Three observations can be made: 1.) German and US house/techno music are distinct, 2.) US styles are much more alike, and 3.) scarcely evolved over time compared to German house/techno regarding the recording studio features. These findings are in agreement with documented statements and thus provide an audio-based perspective on why techno became a mass phenomenon in Germany but remained a fringe phenomenon in the USA. Observations like these can help the music industry estimate whether new trends will experience a breakthrough or disappear.
>
---
#### [new 011] ChronosAudio: A Comprehensive Long-Audio Benchmark for Evaluating Audio-Large Language Models
- **分类: cs.SD**

- **简介: 该论文提出ChronosAudio基准，解决长音频理解问题，评估音频大语言模型。通过多任务测试，揭示模型在长上下文中的性能下降问题。**

- **链接: [https://arxiv.org/pdf/2601.04876v1](https://arxiv.org/pdf/2601.04876v1)**

> **作者:** Kaiwen Luo; Liang Lin; Yibo Zhang; Moayad Aloqaily; Dexian Wang; Zhenhong Zhou; Junwei Zhang; Kun Wang; Li Sun; Qingsong Wen
>
> **摘要:** Although Audio Large Language Models (ALLMs) have witnessed substantial advancements, their long audio understanding capabilities remain unexplored. A plethora of benchmarks have been proposed for general audio tasks, they predominantly focus on short-form clips, leaving without a consensus on evaluating ALLMs over extended durations. This paper proposes ChronosAudio, the first multi-task benchmark tailored for long-audio understanding in ALLMs. It encompasses six major task categories and comprises 36,000 test instances totaling over 200 hours audio, stratified into short, middle, and long-form categories to comprehensively evaluate length generalization. Extensive experiments on 16 state-of-the-art models using ChronosAudio yield three critical findings: 1.Precipitous Long-Context Collapse: ALLMs exhibit a severe inability to sustain performance, with the transition from short to long contexts triggering a staggering performance degradation of over 90% in specific tasks. 2.Structural Attention Dilution: Performance degradation stems from a fundamental failure in maintaining temporal locality; attention mechanisms suffer from significant diffusion in later sequences. 3.Restorative Ceiling of Mitigation: Current strategies only offer 50% recovery. These findings reveal significant challenges in long-audio, underscoring the urgent need for approaches to achieve robust, document-level audio reasoning.
>
---
#### [new 012] LLMs-Integrated Automatic Hate Speech Recognition Using Controllable Text Generation Models
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于仇恨言论识别任务，旨在提升自动语音识别中的有害内容过滤效果。通过整合LLM与ASR模型，生成并筛选仇恨内容样本，实现更有效的文本屏蔽。**

- **链接: [https://arxiv.org/pdf/2601.04654v1](https://arxiv.org/pdf/2601.04654v1)**

> **作者:** Ryutaro Oshima; Yuya Hosoda; Youji Iiguni
>
> **备注:** In Proceedings of the 17th Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC 2025)
>
> **摘要:** This paper proposes an automatic speech recognition (ASR) model for hate speech using large language models (LLMs). The proposed method integrates the encoder of the ASR model with the decoder of the LLMs, enabling simultaneous transcription and censorship tasks to prevent the exposure of harmful content. Instruction tuning of the LLM to mask hate-related words with specific tokens requires an annotated hate speech dataset, which is limited. We generate text samples using an LLM with the Chain-of-Thought (CoT) prompting technique guided by cultural context and examples and then convert them into speech samples using a text-to-speech (TTS) system. However, some of them contain non-hate speech samples with hate-related words, which degrades the censorship performance. This paper filters the samples which text classification models correctly label as hate content. By adjusting the threshold for the number of correct answer models, we can control the level of hate in the generated dataset, allowing us to train the LLMs through curriculum learning in a gradual manner. Experimental results show that the proposed method achieves a masking accuracy of 58.6\% for hate-related words, surpassing previous baselines. We also confirm that the curriculum training contributes to the efficiency of both transcription and censorship tasks.
>
---
#### [new 013] Gradient-based Optimisation of Modulation Effects
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音频效果建模任务，旨在解决模拟调制效果（如混响、合唱、相位器）的计算效率与精度问题。通过基于梯度优化的数字信号处理框架，实现低延迟、高质量的声效模拟。**

- **链接: [https://arxiv.org/pdf/2601.04867v1](https://arxiv.org/pdf/2601.04867v1)**

> **作者:** Alistair Carson; Alec Wright; Stefan Bilbao
>
> **备注:** Submitted to J. Audio Eng. Soc. Dec. 2025
>
> **摘要:** Modulation effects such as phasers, flangers and chorus effects are heavily used in conjunction with the electric guitar. Machine learning based emulation of analog modulation units has been investigated in recent years, but most methods have either been limited to one class of effect or suffer from a high computational cost or latency compared to canonical digital implementations. Here, we build on previous work and present a framework for modelling flanger, chorus and phaser effects based on differentiable digital signal processing. The model is trained in the time-frequency domain, but at inference operates in the time-domain, requiring zero latency. We investigate the challenges associated with gradient-based optimisation of such effects, and show that low-frequency weighting of loss functions avoids convergence to local minima when learning delay times. We show that when trained against analog effects units, sound output from the model is in some cases perceptually indistinguishable from the reference, but challenges still remain for effects with long delay times and feedback.
>
---
#### [new 014] LEMAS: Large A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出LEMAS-Dataset，一个包含15万小时多语言语音的大型标注数据集，用于解决多语言语音合成与编辑问题。通过构建高效数据处理流程和训练两种模型，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.04233v1](https://arxiv.org/pdf/2601.04233v1)**

> **作者:** Zhiyuan Zhao; Lijian Lin; Ye Zhu; Kai Xie; Yunfei Liu; Yu Li
>
> **备注:** Demo page: https://lemas-project.github.io/LEMAS-Project
>
> **摘要:** We present the LEMAS-Dataset, which, to our knowledge, is currently the largest open-source multilingual speech corpus with word-level timestamps. Covering over 150,000 hours across 10 major languages, LEMAS-Dataset is constructed via a efficient data processing pipeline that ensures high-quality data and annotations. To validate the effectiveness of LEMAS-Dataset across diverse generative paradigms, we train two benchmark models with distinct architectures and task specializations on this dataset. LEMAS-TTS, built upon a non-autoregressive flow-matching framework, leverages the dataset's massive scale and linguistic diversity to achieve robust zero-shot multilingual synthesis. Our proposed accent-adversarial training and CTC loss mitigate cross-lingual accent issues, enhancing synthesis stability. Complementarily, LEMAS-Edit employs an autoregressive decoder-only architecture that formulates speech editing as a masked token infilling task. By exploiting precise word-level alignments to construct training masks and adopting adaptive decoding strategies, it achieves seamless, smooth-boundary speech editing with natural transitions. Experimental results demonstrate that models trained on LEMAS-Dataset deliver high-quality synthesis and editing performance, confirming the dataset's quality. We envision that this richly timestamp-annotated, fine-grained multilingual corpus will drive future advances in prompt-based speech generation systems.
>
---
#### [new 015] Leveraging Prediction Entropy for Automatic Prompt Weighting in Zero-Shot Audio-Language Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于零样本音频-语言分类任务，旨在解决文本提示对模型性能影响大的问题。通过熵引导的提示加权方法提升预测置信度，无需额外标注数据。**

- **链接: [https://arxiv.org/pdf/2601.05011v1](https://arxiv.org/pdf/2601.05011v1)**

> **作者:** Karim El Khoury; Maxime Zanella; Tiffanie Godelaine; Christophe De Vleeschouwer; Benoit Macq
>
> **摘要:** Audio-language models have recently demonstrated strong zero-shot capabilities by leveraging natural-language supervision to classify audio events without labeled training data. Yet, their performance is highly sensitive to the wording of text prompts, with small variations leading to large fluctuations in accuracy. Prior work has mitigated this issue through prompt learning or prompt ensembling. However, these strategies either require annotated data or fail to account for the fact that some prompts may negatively impact performance. In this work, we present an entropy-guided prompt weighting approach that aims to find a robust combination of prompt contributions to maximize prediction confidence. To this end, we formulate a tailored objective function that minimizes prediction entropy to yield new prompt weights, utilizing low-entropy as a proxy for high confidence. Our approach can be applied to individual samples or a batch of audio samples, requiring no additional labels and incurring negligible computational overhead. Experiments on five audio classification datasets covering environmental, urban, and vocal sounds, demonstrate consistent gains compared to classical prompt ensembling methods in a zero-shot setting, with accuracy improvements 5-times larger across the whole benchmark.
>
---
#### [new 016] A Unified Spoken Language Model with Injected Emotional-Attribution Thinking for Human-like Interaction
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于情感智能对话系统任务，旨在提升模型的情感理解与回应能力。通过引入IEAT方法，使模型内化情感推理，解决情感表达不自然的问题。**

- **链接: [https://arxiv.org/pdf/2601.04960v1](https://arxiv.org/pdf/2601.04960v1)**

> **作者:** Qing Wang; Zehan Li; Yaodong Song; Hongjie Chen; Jian Kang; Jie Lian; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** This paper presents a unified spoken language model for emotional intelligence, enhanced by a novel data construction strategy termed Injected Emotional-Attribution Thinking (IEAT). IEAT incorporates user emotional states and their underlying causes into the model's internal reasoning process, enabling emotion-aware reasoning to be internalized rather than treated as explicit supervision. The model is trained with a two-stage progressive strategy. The first stage performs speech-text alignment and emotional attribute modeling via self-distillation, while the second stage conducts end-to-end cross-modal joint optimization to ensure consistency between textual and spoken emotional expressions. Experiments on the Human-like Spoken Dialogue Systems Challenge (HumDial) Emotional Intelligence benchmark demonstrate that the proposed approach achieves top-ranked performance across emotional trajectory modeling, emotional reasoning, and empathetic response generation under both LLM-based and human evaluations.
>
---
#### [new 017] WESR: Scaling and Evaluating Word-level Event-Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出WESR，解决语音中非语言事件的精确定位问题。构建了新的分类体系和评估集，建立强基线模型，提升事件检测精度。**

- **链接: [https://arxiv.org/pdf/2601.04508v1](https://arxiv.org/pdf/2601.04508v1)**

> **作者:** Chenchen Yang; Kexin Huang; Liwei Fan; Qian Tu; Botian Jiang; Dong Zhang; Linqi Yin; Shimin Li; Zhaoye Fei; Qinyuan Cheng; Xipeng Qiu
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Speech conveys not only linguistic information but also rich non-verbal vocal events such as laughing and crying. While semantic transcription is well-studied, the precise localization of non-verbal events remains a critical yet under-explored challenge. Current methods suffer from insufficient task definitions with limited category coverage and ambiguous temporal granularity. They also lack standardized evaluation frameworks, hindering the development of downstream applications. To bridge this gap, we first develop a refined taxonomy of 21 vocal events, with a new categorization into discrete (standalone) versus continuous (mixed with speech) types. Based on the refined taxonomy, we introduce WESR-Bench, an expert-annotated evaluation set (900+ utterances) with a novel position-aware protocol that disentangles ASR errors from event detection, enabling precise localization measurement for both discrete and continuous events. We also build a strong baseline by constructing a 1,700+ hour corpus, and train specialized models, surpassing both open-source audio-language models and commercial APIs while preserving ASR quality. We anticipate that WESR will serve as a foundational resource for future research in modeling rich, real-world auditory scenes.
>
---
#### [new 018] Density Matrix RNN (DM-RNN): A Quantum Information Theoretic Framework for Modeling Musical Context and Polyphony
- **分类: cs.LG; cs.SD; math-ph**

- **简介: 该论文提出DM-RNN，用于音乐上下文与复调建模，解决传统RNN信息瓶颈问题，通过密度矩阵和量子理论提升对音乐不确定性的建模能力。**

- **链接: [https://arxiv.org/pdf/2601.04592v1](https://arxiv.org/pdf/2601.04592v1)**

> **作者:** Joonwon Seo; Mariana Montiel
>
> **备注:** Submitted to the 10th International Conference on Mathematics and Computation in Music (MCM 2026)
>
> **摘要:** Classical Recurrent Neural Networks (RNNs) summarize musical context into a deterministic hidden state vector, imposing an information bottleneck that fails to capture the inherent ambiguity in music. We propose the Density Matrix RNN (DM-RNN), a novel theoretical architecture utilizing the Density Matrix. This allows the model to maintain a statistical ensemble of musical interpretations (a mixed state), capturing both classical probabilities and quantum coherences. We rigorously define the temporal dynamics using Quantum Channels (CPTP maps). Crucially, we detail a parameterization strategy based on the Choi-Jamiolkowski isomorphism, ensuring the learned dynamics remain physically valid (CPTP) by construction. We introduce an analytical framework using Von Neumann Entropy to quantify musical uncertainty and Quantum Mutual Information (QMI) to measure entanglement between voices. The DM-RNN provides a mathematically rigorous framework for modeling complex, ambiguous musical structures.
>
---
## 更新

#### [replaced 001] MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转写任务，解决会议录音中说话人识别与时间戳问题。提出MOSS Transcribe Diarize模型，实现端到端的说话人归属与时间标注。**

- **链接: [https://arxiv.org/pdf/2601.01554v3](https://arxiv.org/pdf/2601.01554v3)**

> **作者:** MOSI. AI; :; Donghua Yu; Zhengyuan Lin; Chen Yang; Yiyang Zhang; Hanfu Chen; Jingqi Chen; Ke Chen; Liwei Fan; Yi Jiang; Jie Zhu; Muchen Li; Wenxuan Wang; Yang Wang; Zhe Xu; Yitian Gong; Yuqian Zhang; Wenbo Zhang; Zhaoye Fei; Songlin Wang; Zhiyu Wu; Qinyuan Cheng; Shimin Li; Xipeng Qiu
>
> **摘要:** Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks.
>
---
#### [replaced 002] MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于多模态生成任务，解决音频视频同步与零样本语音克隆问题。提出MM-Sonate框架，实现精准语音控制与高质量语音克隆。**

- **链接: [https://arxiv.org/pdf/2601.01568v2](https://arxiv.org/pdf/2601.01568v2)**

> **作者:** Chunyu Qiang; Jun Wang; Xiaopeng Wang; Kang Yin; Yuxin Guo
>
> **摘要:** Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems.
>
---
#### [replaced 003] VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音对话系统任务，旨在解决现有基准不足的问题。构建了VCB Bench，一个基于真实中文语音的评估基准，从指令遵循、知识理解和鲁棒性三个方面评估大模型。**

- **链接: [https://arxiv.org/pdf/2510.11098v3](https://arxiv.org/pdf/2510.11098v3)**

> **作者:** Jiliang Hu; Wenfu Wang; Zuchao Li; Chenxing Li; Yiyang Zhao; Hanzhao Li; Liqiang Zhang; Meng Yu; Dong Yu
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.
>
---
#### [replaced 004] Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于可控长歌曲生成任务，旨在解决学术研究不可复现的问题。工作包括发布开源系统Muse及合成数据集，实现细粒度风格控制的歌曲生成。**

- **链接: [https://arxiv.org/pdf/2601.03973v2](https://arxiv.org/pdf/2601.03973v2)**

> **作者:** Changhao Jiang; Jiahao Chen; Zhenghao Xiang; Zhixiong Yang; Hanchen Wang; Jiabao Zhuang; Xinmeng Che; Jiajun Sun; Hui Li; Yifei Cao; Shihan Dou; Ming Zhang; Junjie Ye; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at https://github.com/yuhui1038/Muse.
>
---
#### [replaced 005] IndexTTS 2.5 Technical Report
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决零样本多语言情感语音生成问题，通过优化模型结构和引入多语言策略提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.03888v2](https://arxiv.org/pdf/2601.03888v2)**

> **作者:** Yunpei Li; Xun Zhou; Jinchao Wang; Lu Wang; Yong Wu; Siyi Zhou; Yiquan Zhou; Jingchen Shu
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** In prior work, we introduced IndexTTS 2, a zero-shot neural text-to-speech foundation model comprising two core components: a transformer-based Text-to-Semantic (T2S) module and a non-autoregressive Semantic-to-Mel (S2M) module, which together enable faithful emotion replication and establish the first autoregressive duration-controllable generative paradigm. Building upon this, we present IndexTTS 2.5, which significantly enhances multilingual coverage, inference speed, and overall synthesis quality through four key improvements: 1) Semantic Codec Compression: we reduce the semantic codec frame rate from 50 Hz to 25 Hz, halving sequence length and substantially lowering both training and inference costs; 2) Architectural Upgrade: we replace the U-DiT-based backbone of the S2M module with a more efficient Zipformer-based modeling architecture, achieving notable parameter reduction and faster mel-spectrogram generation; 3) Multilingual Extension: We propose three explicit cross-lingual modeling strategies, boundary-aware alignment, token-level concatenation, and instruction-guided generation, establishing practical design principles for zero-shot multilingual emotional TTS that supports Chinese, English, Japanese, and Spanish, and enables robust emotion transfer even without target-language emotional training data; 4) Reinforcement Learning Optimization: we apply GRPO in post-training of the T2S module, improving pronunciation accuracy and natrualness. Experiments show that IndexTTS 2.5 not only supports broader language coverage but also replicates emotional prosody in unseen languages under the same zero-shot setting. IndexTTS 2.5 achieves a 2.28 times improvement in RTF while maintaining comparable WER and speaker similarity to IndexTTS 2.
>
---
#### [replaced 006] TellWhisper: Tell Whisper Who Speaks When
- **分类: eess.AS**

- **简介: 该论文属于多说话人语音识别任务，解决快速切换和重叠语音下的说话人与时间建模问题。提出TellWhisper框架，联合建模说话人身份与时间信息，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.03712v2](https://arxiv.org/pdf/2601.03712v2)**

> **作者:** Yifan Hu; Peiji Yang; Zhisheng Wang; Yicheng Zhong; Rui Liu
>
> **备注:** 14 pages, 6 figures, 8 tables
>
> **摘要:** Multi-speaker automatic speech recognition (MASR) aims to predict ''who spoke when and what'' from multi-speaker speech, a key technology for multi-party dialogue understanding. However, most existing approaches decouple temporal modeling and speaker modeling when addressing ''when'' and ''who'': some inject speaker cues before encoding (e.g., speaker masking), which can cause irreversible information loss; others fuse identity by mixing speaker posteriors after encoding, which may entangle acoustic content with speaker identity. This separation is brittle under rapid turn-taking and overlapping speech, often leading to degraded performance. To address these limitations, we propose TellWhisper, a unified framework that jointly models speaker identity and temporal within the speech encoder. Specifically, we design TS-RoPE, a time-speaker rotary positional encoding: time coordinates are derived from frame indices, while speaker coordinates are derived from speaker activity and pause cues. By applying region-specific rotation angles, the model explicitly captures per-speaker continuity, speaker-turn transitions, and state dynamics, enabling the attention mechanism to simultaneously attend to ''when'' and ''who''. Moreover, to estimate frame-level speaker activity, we develop Hyper-SD, which casts speaker classification in hyperbolic space to enhance inter-class separation and refine speaker-activity estimates. Extensive experiments demonstrate the effectiveness of the proposed approach.
>
---
#### [replaced 007] MoE Adapter for Large Audio Language Models: Sparsity, Disentanglement, and Gradient-Conflict-Free
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多模态语言模型任务，解决音频信息异质性导致的梯度冲突问题。提出MoE-Adapter架构，通过稀疏专家机制实现特征解耦与细粒度学习。**

- **链接: [https://arxiv.org/pdf/2601.02967v2](https://arxiv.org/pdf/2601.02967v2)**

> **作者:** Yishu Lei; Shuwei He; Jing Hu; Dan Zhang; Xianlong Luo; Danxiang Zhu; Shikun Feng; Rui Liu; Jingzhou He; Yu Sun; Hua Wu; Haifeng Wang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Extending the input modality of Large Language Models~(LLMs) to the audio domain is essential for achieving comprehensive multimodal perception. However, it is well-known that acoustic information is intrinsically \textit{heterogeneous}, entangling attributes such as speech, music, and environmental context. Existing research is limited to a dense, parameter-shared adapter to model these diverse patterns, which induces \textit{gradient conflict} during optimization, as parameter updates required for distinct attributes contradict each other. To address this limitation, we introduce the \textit{\textbf{MoE-Adapter}}, a sparse Mixture-of-Experts~(MoE) architecture designed to decouple acoustic information. Specifically, it employs a dynamic gating mechanism that routes audio tokens to specialized experts capturing complementary feature subspaces while retaining shared experts for global context, thereby mitigating gradient conflicts and enabling fine-grained feature learning. Comprehensive experiments show that the MoE-Adapter achieves superior performance on both audio semantic and paralinguistic tasks, consistently outperforming dense linear baselines with comparable computational costs. Furthermore, we will release the related code and models to facilitate future research.
>
---
