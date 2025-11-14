# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究多模态LLM的语音-音频复合攻击问题，提出SACRED-Bench基准评估安全漏洞，并设计SALMONN-Guard模型联合分析语音、音频与文本，将攻击成功率从66%降至20%，推动音频感知安全防御。**

- **链接: [https://arxiv.org/pdf/2511.10222v1](https://arxiv.org/pdf/2511.10222v1)**

> **作者:** Yudong Yang; Xuezhen Zhang; Zhifeng Han; Siyin Wang; Jimin Zhuang; Zengrui Jin; Jing Shao; Guangzhi Sun; Chao Zhang
>
> **摘要:** Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
>
---
#### [new 002] Video Echoed in Music: Semantic, Temporal, and Rhythmic Alignment for Video-to-Music Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出VeM模型，用于视频到音乐生成任务，解决语义、时序与节奏对齐不足的问题。通过分层视频解析、故事板引导注意力和帧级节拍对齐机制，实现高精度音乐生成，并构建新数据集与评估指标。**

- **链接: [https://arxiv.org/pdf/2511.09585v1](https://arxiv.org/pdf/2511.09585v1)**

> **作者:** Xinyi Tong; Yiran Zh; Jishang Chen; Chunru Zhan; Tianle Wang; Sirui Zhang; Nian Liu; Tiezheng Ge; Duo Xu; Xin Jin; Feng Yu; Song-Chun Zhu
>
> **摘要:** Video-to-Music generation seeks to generate musically appropriate background music that enhances audiovisual immersion for videos. However, current approaches suffer from two critical limitations: 1) incomplete representation of video details, leading to weak alignment, and 2) inadequate temporal and rhythmic correspondence, particularly in achieving precise beat synchronization. To address the challenges, we propose Video Echoed in Music (VeM), a latent music diffusion that generates high-quality soundtracks with semantic, temporal, and rhythmic alignment for input videos. To capture video details comprehensively, VeM employs a hierarchical video parsing that acts as a music conductor, orchestrating multi-level information across modalities. Modality-specific encoders, coupled with a storyboard-guided cross-attention mechanism (SG-CAtt), integrate semantic cues while maintaining temporal coherence through position and duration encoding. For rhythmic precision, the frame-level transition-beat aligner and adapter (TB-As) dynamically synchronize visual scene transitions with music beats. We further contribute a novel video-music paired dataset sourced from e-commerce advertisements and video-sharing platforms, which imposes stricter transition-beat synchronization requirements. Meanwhile, we introduce novel metrics tailored to the task. Experimental results demonstrate superiority, particularly in semantic relevance and rhythmic precision.
>
---
#### [new 003] A Study of Binaural Deep Beamforming With Interpretable Beampatterns Guided by Time-Varying RTF
- **分类: eess.AS**

- **简介: 该论文研究基于时变RTF引导的深度双耳波束形成，用于动态声环境中的语音增强。通过最小化SI-SDR损失，利用RTF引导网络生成可解释、时空一致的波束图，提升目标声源追踪能力，并输出双耳信号以保留空间线索。**

- **链接: [https://arxiv.org/pdf/2511.10168v1](https://arxiv.org/pdf/2511.10168v1)**

> **作者:** Ilai Zaidel; Sharon Gannot
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** In this work, a deep beamforming framework for speech enhancement in dynamic acoustic environments is studied. The time-varying beamformer weights are estimated from the noisy multichannel signals by minimizing an SI-SDR loss. The estimation is guided by the continuously tracked relative transfer functions (RTFs) of the moving target speaker. The spatial behavior of the network is evaluated through both narrowband and wideband beampatterns under three settings: (i) oracle guidance using true RTFs, (ii) estimated RTFs obtained by a subspace tracking method, and (iii) without the RTF guidance. Results show that RTF-guided models produce smoother, spatially consistent beampatterns that accurately track the target's direction of arrival. In contrast, the model fails to maintain a clear spatial focus when guidance is absent. Using the estimated RTFs as guidance closely matches the oracle RTF behavior, confirming the effectiveness of the tracking scheme. The model also outputs a binaural signal to preserve the speaker's spatial cues, which promotes hearing aid and hearables applications.
>
---
#### [new 004] Time-Layer Adaptive Alignment for Speaker Similarity in Flow-Matching Based Zero-Shot TTS
- **分类: eess.AS**

- **简介: 该论文面向零样本语音合成任务，解决流匹配框架中说话人信息分布不均导致的相似性不足问题，提出时间-层自适应对齐损失（TLA-SA），提升说话人一致性，显著改善跨模型架构的说话人相似度。**

- **链接: [https://arxiv.org/pdf/2511.09995v1](https://arxiv.org/pdf/2511.09995v1)**

> **作者:** Haoyu Li; Mingyang Han; Yu Xi; Dongxiao Wang; Hankun Wang; Haoxiang Shi; Boyu Li; Jun Song; Bo Zheng; Shuai Wang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Flow-Matching (FM)-based zero-shot text-to-speech (TTS) systems exhibit high-quality speech synthesis and robust generalization capabilities. However, the speaker representation ability of such systems remains underexplored, primarily due to the lack of explicit speaker-specific supervision in the FM framework. To this end, we conduct an empirical analysis of speaker information distribution and reveal its non-uniform allocation across time steps and network layers, underscoring the need for adaptive speaker alignment. Accordingly, we propose Time-Layer Adaptive Speaker Alignment (TLA-SA), a loss that enhances speaker consistency by jointly leveraging temporal and hierarchical variations in speaker information. Experimental results show that TLA-SA significantly improves speaker similarity compared to baseline systems on both research- and industrial-scale datasets and generalizes effectively across diverse model architectures, including decoder-only language models (LM) and FM-based TTS systems free of LM.
>
---
#### [new 005] WaveRoll: JavaScript Library for Comparative MIDI Piano-Roll Visualization
- **分类: cs.SD**

- **简介: WaveRoll是一个用于比较多个MIDI钢琴卷帘的JavaScript库，解决AMT模型输出难以直观对比的问题，支持同步播放与时间对齐的可视化，辅助模型评估与误差分析。**

- **链接: [https://arxiv.org/pdf/2511.09562v1](https://arxiv.org/pdf/2511.09562v1)**

> **作者:** Hannah Park; Dasaem Jeong
>
> **备注:** Late-breaking/demo (LBD) at ISMIR 2025. https://ismir2025program.ismir.net/lbd_459.html
>
> **摘要:** WaveRoll is an interactive JavaScript library that enables comparative visualization and synchronized playback of multiple MIDI piano rolls on a browser. It addresses a specific evaluation need in Automatic Music Transcription (AMT), contrasting multiple MIDI outputs produced from the same input. The library displays multiple MIDI tracks on a single, time-aligned grid with synchronized audio, allowing users to compare pitch and timing, identify missed or extra notes, and observe onset and offset differences, as well as section-level patterns. We expect that such comparisons would assist in model evaluation and error analysis, and help readers to understand the model behavior better. The open-source library is available at https://github.com/crescent-stdio/wave-roll
>
---
#### [new 006] Direction-of-Arrival and Noise Covariance Matrix joint estimation for beamforming
- **分类: eess.AS; math.OC**

- **简介: 该论文针对波束形成中的DoA与噪声协方差矩阵联合估计问题，提出一种准线性解法，替代传统穷举搜索，并引入全频带DoA估计提升抗混响能力，显著降低角度误差并增强降噪与干扰抑制性能。**

- **链接: [https://arxiv.org/pdf/2511.10639v1](https://arxiv.org/pdf/2511.10639v1)**

> **作者:** Vitor Gelsleichter Probst Curtarelli
>
> **摘要:** We propose a joint estimation method for the Direction-of-Arrival (DoA) and the Noise Covariance Matrix (NCM) tailored for beamforming applications. Building upon an existing NCM framework, our approach simplifies the estimation procedure by deriving an quasi-linear solution, instead of the traditional exhaustive search. Additionally, we introduce a novel DoA estimation technique that operates across all frequency bins, improving robustness in reverberant environments. Simulation results demonstrate that our method outperforms classical techniques, such as MUSIC, in mid- to high-angle scenarios, achieving lower angular errors and superior signal enhancement through beamforming. The proposed framework was also fared against other techniques for signal enhancement, having better noise rejection and interference canceling capabilities. These improvements are validated using both theoretical and empirical performance metrics.
>
---
#### [new 007] Music Flamingo: Scaling Music Understanding in Audio Language Models
- **分类: eess.AS; cs.CL**

- **简介: 论文提出Music Flamingo，面向音乐理解的音频语言模型，解决音乐数据稀缺与模型浅层理解问题。通过构建MF-Skills数据集与MF-Think推理数据，结合强化学习，实现对音乐结构、文化背景等深层特征的多维理解与推理。**

- **链接: [https://arxiv.org/pdf/2511.10289v1](https://arxiv.org/pdf/2511.10289v1)**

> **作者:** Sreyan Ghosh; Arushi Goel; Lasha Koroshinadze; Sang-gil Lee; Zhifeng Kong; Joao Felipe Santos; Ramani Duraiswami; Dinesh Manocha; Wei Ping; Mohammad Shoeybi; Bryan Catanzaro
>
> **备注:** Project Page: https://research.nvidia.com/labs/adlr/MF/
>
> **摘要:** We introduce Music Flamingo, a novel large audio-language model designed to advance music (including song) understanding in foundational audio models. While audio-language research has progressed rapidly, music remains challenging due to its dynamic, layered, and information-dense nature. Progress has been further limited by the difficulty of scaling open audio understanding models, primarily because of the scarcity of high-quality music data and annotations. As a result, prior models are restricted to producing short, high-level captions, answering only surface-level questions, and showing limited generalization across diverse musical cultures. To address these challenges, we curate MF-Skills, a large-scale dataset labeled through a multi-stage pipeline that yields rich captions and question-answer pairs covering harmony, structure, timbre, lyrics, and cultural context. We fine-tune an enhanced Audio Flamingo 3 backbone on MF-Skills and further strengthen multiple skills relevant to music understanding. To improve the model's reasoning abilities, we introduce a post-training recipe: we first cold-start with MF-Think, a novel chain-of-thought dataset grounded in music theory, followed by GRPO-based reinforcement learning with custom rewards. Music Flamingo achieves state-of-the-art results across 10+ benchmarks for music understanding and reasoning, establishing itself as a generalist and musically intelligent audio-language model. Beyond strong empirical results, Music Flamingo sets a new standard for advanced music understanding by demonstrating how models can move from surface-level recognition toward layered, human-like perception of songs. We believe this work provides both a benchmark and a foundation for the community to build the next generation of models that engage with music as meaningfully as humans do.
>
---
#### [new 008] FabasedVC: Enhancing Voice Conversion with Text Modality Fusion and Phoneme-Level SSL Features
- **分类: cs.SD**

- **简介: 该论文面向语音转换任务，旨在提升目标说话人音色、韵律与语义完整性。提出FabasedVC，融合文本特征、音素级SSL特征与持续时间预测器，基于VITS架构实现更自然、高相似度的语音转换。**

- **链接: [https://arxiv.org/pdf/2511.10112v1](https://arxiv.org/pdf/2511.10112v1)**

> **作者:** Wenyu Wang; Zhetao Hu; Yiquan Zhou; Jiacheng Xu; Zhiyu Wu; Chen Li; Shihao Li
>
> **备注:** Accepted by ACMMM-Asia 2025
>
> **摘要:** In voice conversion (VC), it is crucial to preserve complete semantic information while accurately modeling the target speaker's timbre and prosody. This paper proposes FabasedVC to achieve VC with enhanced similarity in timbre, prosody, and duration to the target speaker, as well as improved content integrity. It is an end-to-end VITS-based VC system that integrates relevant textual modality information, phoneme-level self-supervised learning (SSL) features, and a duration predictor. Specifically, we employ a text feature encoder to encode attributes such as text, phonemes, tones and BERT features. We then process the frame-level SSL features into phoneme-level features using two methods: average pooling and attention mechanism based on each phoneme's duration. Moreover, a duration predictor is incorporated to better align the speech rate and prosody of the target speaker. Experimental results demonstrate that our method outperforms competing systems in terms of naturalness, similarity, and content integrity.
>
---
#### [new 009] Proceedings of The third international workshop on eXplainable AI for the Arts (XAIxArts)
- **分类: cs.AI; cs.HC; cs.MM; cs.SD**

- **简介: 该论文是会议论文集，记录了第三届XAIxArts研讨会，汇聚HCI、AI与数字艺术研究者，探讨可解释AI（XAI）在艺术创作与体验中的作用，旨在促进跨学科对话与应用。**

- **链接: [https://arxiv.org/pdf/2511.10482v1](https://arxiv.org/pdf/2511.10482v1)**

> **作者:** Corey Ford; Elizabeth Wilson; Shuoyang Zheng; Gabriel Vigliensoni; Jeba Rezwana; Lanxi Xiao; Michael Clemens; Makayla Lewis; Drew Hemment; Alan Chamberlain; Helen Kennedy; Nick Bryan-Kinns
>
> **备注:** Proceedings of The second international workshop on eXplainable AI for the Arts (XAIxArts)
>
> **摘要:** This third international workshop on explainable AI for the Arts (XAIxArts) brought together a community of researchers in HCI, Interaction Design, AI, explainable AI (XAI), and digital arts to explore the role of XAI for the Arts. Workshop held at the 17th ACM Conference on Creativity and Cognition (C&C 2025), online.
>
---
#### [new 010] HI-TransPA: Hearing Impairments Translation Personal Assistant
- **分类: cs.CL; cs.MM; cs.SD**

- **简介: HI-TransPA是一款面向听障人士的多模态个人助手，通过融合语音与唇动信息，实现端到端的语音-文本翻译与对话。论文构建了高质量数据集与预处理管道，提出课程学习与SigLIP+3D-Resampler架构，提升噪声环境下模型鲁棒性，首次将Omni-Model应用于助听通信。**

- **链接: [https://arxiv.org/pdf/2511.09915v1](https://arxiv.org/pdf/2511.09915v1)**

> **作者:** Zhiming Ma; Shiyu Gan; Junhao Zhao; Xianming Li; Qingyun Pan; Peidong Wang; Mingjun Pan; Yuhao Mo; Jiajie Cheng; Chengxin Chen; Zhonglun Cao; Chonghan Liu; Shi Cheng
>
> **摘要:** To provide a unified and flexible solution for daily communication among hearing-impaired individuals, we introduce the Omni-Model paradigm into assistive technology and present HI-TransPA, an instruction-driven audio-visual personal assistant. The model fuses indistinct speech with high-frame-rate lip dynamics, enabling both translation and dialogue within a single multimodal framework. To tackle the challenges of noisy and heterogeneous raw data and the limited adaptability of existing Omni-Models to hearing-impaired speech, we construct a comprehensive preprocessing and curation pipeline that detects facial landmarks, isolates and stabilizes the lip region, and quantitatively assesses multimodal sample quality. These quality scores guide a curriculum learning strategy that first trains on clean, high-confidence samples and progressively incorporates harder cases to strengthen model robustness. We further adopt a SigLIP encoder combined with a Unified 3D-Resampler to efficiently encode high-frame-rate lip motion. Experiments on our purpose-built HI-Dialogue dataset show that HI-TransPA achieves state-of-the-art performance in both literal accuracy and semantic fidelity. This work establishes a foundation for applying Omni-Models to assistive communication technology, providing an end-to-end modeling framework and essential processing tools for future research.
>
---
#### [new 011] Investigation of Feature Selection and Pooling Methods for Environmental Sound Classification
- **分类: eess.SP; cs.SD**

- **简介: 该论文面向环境声音分类任务，旨在提升轻量级CNN的效率与精度。通过对比SSRP系列池化方法与PCA，验证了SSRP-T在ESC-50数据集上显著优于基线模型，证明稀疏池化可高效平衡精度与计算成本。**

- **链接: [https://arxiv.org/pdf/2511.09802v1](https://arxiv.org/pdf/2511.09802v1)**

> **作者:** Parinaz Binandeh Dehaghani; Danilo Pena; A. Pedro Aguiar
>
> **备注:** 6 pages, 7 figures (including subfigures)
>
> **摘要:** This paper explores the impact of dimensionality reduction and pooling methods for Environmental Sound Classification (ESC) using lightweight CNNs. We evaluate Sparse Salient Region Pooling (SSRP) and its variants, SSRP-Basic (SSRP-B) and SSRP-Top-K (SSRP-T), under various hyperparameter settings and compare them with Principal Component Analysis (PCA). Experiments on the ESC-50 dataset demonstrate that SSRP-T achieves up to 80.69 % accuracy, significantly outperforming both the baseline CNN (66.75 %) and the PCA-reduced model (37.60 %). Our findings confirm that a well-tuned sparse pooling strategy provides a robust, efficient, and high-performing solution for ESC tasks, particularly in resource-constrained scenarios where balancing accuracy and computational cost is crucial.
>
---
#### [new 012] Audio-VLA: Adding Contact Audio Perception to Vision-Language-Action Model for Robotic Manipulation
- **分类: cs.RO; cs.SD**

- **简介: 论文提出Audio-VLA，将接触音频引入视觉-语言-动作模型，解决视觉仅靠无法感知操作动态过程的问题。通过多模态融合与音频增强仿真，提升机器人对操作过程的感知能力，并提出TCR指标评估动态过程表现。**

- **链接: [https://arxiv.org/pdf/2511.09958v1](https://arxiv.org/pdf/2511.09958v1)**

> **作者:** Xiangyi Wei; Haotian Zhang; Xinyi Cao; Siyu Xie; Weifeng Ge; Yang Li; Changbo Wang
>
> **摘要:** The Vision-Language-Action models (VLA) have achieved significant advances in robotic manipulation recently. However, vision-only VLA models create fundamental limitations, particularly in perceiving interactive and manipulation dynamic processes. This paper proposes Audio-VLA, a multimodal manipulation policy that leverages contact audio to perceive contact events and dynamic process feedback. Audio-VLA overcomes the vision-only constraints of VLA models. Additionally, this paper introduces the Task Completion Rate (TCR) metric to systematically evaluate dynamic operational processes. Audio-VLA employs pre-trained DINOv2 and SigLIP as visual encoders, AudioCLIP as the audio encoder, and Llama2 as the large language model backbone. We apply LoRA fine-tuning to these pre-trained modules to achieve robust cross-modal understanding of both visual and acoustic inputs. A multimodal projection layer aligns features from different modalities into the same feature space. Moreover RLBench and LIBERO simulation environments are enhanced by adding collision-based audio generation to provide realistic sound feedback during object interactions. Since current robotic manipulation evaluations focus on final outcomes rather than providing systematic assessment of dynamic operational processes, the proposed TCR metric measures how well robots perceive dynamic processes during manipulation, creating a more comprehensive evaluation metric. Extensive experiments on LIBERO, RLBench, and two real-world tasks demonstrate Audio-VLA's superior performance over vision-only comparative methods, while the TCR metric effectively quantifies dynamic process perception capabilities.
>
---
#### [new 013] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 论文提出MTR-DuplexBench基准，用于评估全双工语音语言模型在多轮对话中的表现，解决现有评测忽视多轮交互、指令遵循与安全性的不足，通过分段对话实现细粒度多维度评估。**

- **链接: [https://arxiv.org/pdf/2511.10262v1](https://arxiv.org/pdf/2511.10262v1)**

> **作者:** He Zhang; Wenqian Cui; Haoning Xu; Xiaohui Li; Lei Zhu; Shaohua Ma; Irwin King
>
> **备注:** Work in progress
>
> **摘要:** Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions and conversational features, neglecting the complexities of multi-round communication and critical capabilities such as instruction following and safety. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark that segments continuous full-duplex dialogues into discrete turns, enabling comprehensive, turn-by-turn evaluation of FD-SLMs across dialogue quality, conversational dynamics, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our proposed benchmark. The benchmark and code will be available in the future.
>
---
#### [new 014] VocalNet-M2: Advancing Low-Latency Spoken Language Modeling via Integrated Multi-Codebook Tokenization and Multi-Token Prediction
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 论文提出VocalNet-M2，面向低延迟语音语言建模任务，解决传统模型因自回归生成和流匹配导致的高延迟问题，通过多码本分词与多令牌预测策略，显著降低首块延迟至350ms，提升实时交互效率。**

- **链接: [https://arxiv.org/pdf/2511.10232v1](https://arxiv.org/pdf/2511.10232v1)**

> **作者:** Yuhao Wang; Ziyang Cheng; Heyang Liu; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** Current end-to-end spoken language models (SLMs) have made notable progress, yet they still encounter considerable response latency. This delay primarily arises from the autoregressive generation of speech tokens and the reliance on complex flow-matching models for speech synthesis. To overcome this, we introduce VocalNet-M2, a novel low-latency SLM that integrates a multi-codebook tokenizer and a multi-token prediction (MTP) strategy. Our model directly generates multi-codebook speech tokens, thus eliminating the need for a latency-inducing flow-matching model. Furthermore, our MTP strategy enhances generation efficiency and improves overall performance. Extensive experiments demonstrate that VocalNet-M2 achieves a substantial reduction in first chunk latency (from approximately 725ms to 350ms) while maintaining competitive performance across mainstream SLMs. This work also provides a comprehensive comparison of single-codebook and multi-codebook strategies, offering valuable insights for developing efficient and high-performance SLMs for real-time interactive applications.
>
---
#### [new 015] Rebellion: Noise-Robust Reasoning Training for Audio Reasoning Models
- **分类: cs.AI; cs.SD**

- **简介: 该论文面向音频推理模型（ARMs）的安全性问题，提出Rebellion方法，通过对抗表示漂移的鲁棒训练，有效防御新型音频越狱攻击，提升安全与性能的平衡，优于标准训练方法。**

- **链接: [https://arxiv.org/pdf/2511.09682v1](https://arxiv.org/pdf/2511.09682v1)**

> **作者:** Tiansheng Huang; Virat Shejwalkar; Oscar Chang; Milad Nasr; Ling Liu
>
> **摘要:** Instilling reasoning capabilities in large models (LMs) using reasoning training (RT) significantly improves LMs' performances. Thus Audio Reasoning Models (ARMs), i.e., audio LMs that can reason, are becoming increasingly popular. However, no work has studied the safety of ARMs against jailbreak attacks that aim to elicit harmful responses from target models. To this end, first, we show that standard RT with appropriate safety reasoning data can protect ARMs from vanilla audio jailbreaks, but cannot protect them against our proposed simple yet effective jailbreaks. We show that this is because of the significant representation drift between vanilla and advanced jailbreaks which forces the target ARMs to emit harmful responses. Based on this observation, we propose Rebellion, a robust RT that trains ARMs to be robust to the worst-case representation drift. All our results are on Qwen2-Audio; they demonstrate that Rebellion: 1) can protect against advanced audio jailbreaks without compromising performance on benign tasks, and 2) significantly improves accuracy-safety trade-off over standard RT method.
>
---
## 更新

#### [replaced 001] MiDashengLM: Efficient Audio Understanding with General Audio Captions
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2508.03983v2](https://arxiv.org/pdf/2508.03983v2)**

> **作者:** Heinrich Dinkel; Gang Li; Jizhong Liu; Jian Luan; Yadong Niu; Xingwei Sun; Tianzi Wang; Qiyang Xiao; Junbo Zhang; Jiahao Zhou
>
> **摘要:** Current approaches for large audio language models (LALMs) often rely on closed data sources or proprietary models, limiting their generalization and accessibility. This paper introduces MiDashengLM, a novel open audio-language model designed for efficient and comprehensive audio understanding through the use of general audio captions using our novel ACAVCaps training dataset. MiDashengLM exclusively relies on publicly available pretraining and supervised fine-tuning (SFT) datasets, ensuring full transparency and reproducibility. At its core, MiDashengLM integrates Dasheng, an open-source audio encoder, specifically engineered to process diverse auditory information effectively. Unlike previous works primarily focused on Automatic Speech Recognition (ASR) based audio-text alignment, our strategy centers on general audio captions, fusing speech, sound and music information into one textual representation, enabling a holistic textual representation of complex audio scenes. Lastly, MiDashengLM provides an up to 4x speedup in terms of time-to-first-token (TTFT) and up to 20x higher throughput than comparable models. Checkpoints are available online at https://huggingface.co/mispeech/midashenglm-7b and https://github.com/xiaomi-research/dasheng-lm.
>
---
#### [replaced 002] SpikCommander: A High-performance Spiking Transformer with Multi-view Learning for Efficient Speech Command Recognition
- **分类: cs.SD; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.07883v2](https://arxiv.org/pdf/2511.07883v2)**

> **作者:** Jiaqi Wang; Liutao Yu; Xiongri Shen; Sihang Guo; Chenlin Zhou; Leilei Zhao; Yi Zhong; Zhiguo Zhang; Zhengyu Ma
>
> **备注:** Accepted by The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Spiking neural networks (SNNs) offer a promising path toward energy-efficient speech command recognition (SCR) by leveraging their event-driven processing paradigm. However, existing SNN-based SCR methods often struggle to capture rich temporal dependencies and contextual information from speech due to limited temporal modeling and binary spike-based representations. To address these challenges, we first introduce the multi-view spiking temporal-aware self-attention (MSTASA) module, which combines effective spiking temporal-aware attention with a multi-view learning framework to model complementary temporal dependencies in speech commands. Building on MSTASA, we further propose SpikCommander, a fully spike-driven transformer architecture that integrates MSTASA with a spiking contextual refinement channel MLP (SCR-MLP) to jointly enhance temporal context modeling and channel-wise feature integration. We evaluate our method on three benchmark datasets: the Spiking Heidelberg Dataset (SHD), the Spiking Speech Commands (SSC), and the Google Speech Commands V2 (GSC). Extensive experiments demonstrate that SpikCommander consistently outperforms state-of-the-art (SOTA) SNN approaches with fewer parameters under comparable time steps, highlighting its effectiveness and efficiency for robust speech command recognition.
>
---
#### [replaced 003] A Critical Review of the Need for Knowledge-Centric Evaluation of Quranic Recitation
- **分类: cs.CL; cs.AI; cs.SD**

- **链接: [https://arxiv.org/pdf/2510.12858v2](https://arxiv.org/pdf/2510.12858v2)**

> **作者:** Mohammed Hilal Al-Kharusi; Khizar Hayat; Khalil Bader Al Ruqeishi; Haroon Rashid Lone
>
> **备注:** 32 pages
>
> **摘要:** The art and science of Quranic recitation (Tajweed), a discipline governed by meticulous phonetic, rhythmic, and theological principles, confronts substantial educational challenges in today's digital age. Although modern technology offers unparalleled opportunities for learning, existing automated systems for evaluating recitation have struggled to gain broad acceptance or demonstrate educational effectiveness. This literature review examines this crucial disparity, offering a thorough analysis of scholarly research, digital platforms, and commercial tools developed over the past twenty years. Our analysis uncovers a fundamental flaw in current approaches that adapt Automatic Speech Recognition (ASR) systems, which emphasize word identification over qualitative acoustic evaluation. These systems suffer from limitations such as reliance on biased datasets, demographic disparities, and an inability to deliver meaningful feedback for improvement. Challenging these data-centric methodologies, we advocate for a paradigm shift toward a knowledge-based computational framework. By leveraging the unchanging nature of the Quranic text and the well-defined rules of Tajweed, we propose that an effective evaluation system should be built upon rule-based acoustic modeling centered on canonical pronunciation principles and articulation points (Makhraj), rather than depending on statistical patterns derived from flawed or biased data. The review concludes that the future of automated Quranic recitation assessment lies in hybrid systems that combine linguistic expertise with advanced audio processing. Such an approach paves the way for developing reliable, fair, and pedagogically effective tools that can authentically assist learners across the globe.
>
---
#### [replaced 004] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2509.04685v3](https://arxiv.org/pdf/2509.04685v3)**

> **作者:** Rui-Chen Zheng; Wenrui Liu; Hui-Peng Du; Qinglin Zhang; Chong Deng; Qian Chen; Wen Wang; Yang Ai; Zhen-Hua Ling
>
> **备注:** Accepted to AAAI 2026. Project page: https://zhengrachel.github.io/VARSTok
>
> **摘要:** Existing speech tokenizers typically assign a fixed number of tokens per second, regardless of the varying information density or temporal fluctuations in the speech signal. This uniform token allocation mismatches the intrinsic structure of speech, where information is distributed unevenly over time. To address this, we propose VARSTok, a VAriable-frame-Rate Speech Tokenizer that adapts token allocation based on local feature similarity. VARSTok introduces two key innovations: (1) a temporal-aware density peak clustering algorithm that adaptively segments speech into variable-length units, and (2) a novel implicit duration coding scheme that embeds both content and temporal span into a single token index, eliminating the need for auxiliary duration predictors. Extensive experiments show that VARSTok significantly outperforms strong fixed-rate baselines. Notably, it achieves superior reconstruction naturalness while using up to 23% fewer tokens than a 40 Hz fixed-frame-rate baseline. VARSTok further yields lower word error rates and improved naturalness in zero-shot text-to-speech synthesis. To the best of our knowledge, this is the first work to demonstrate that a fully dynamic, variable-frame-rate acoustic speech tokenizer can be seamlessly integrated into downstream speech language models.
>
---
#### [replaced 005] Neural Directional Filtering Using a Compact Microphone Array
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2511.07185v2](https://arxiv.org/pdf/2511.07185v2)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Mhd Modar Halimeh; Oliver Thiergart; Emanuël A. P. Habets
>
> **摘要:** Beamforming with desired directivity patterns using compact microphone arrays is essential in many audio applications. Directivity patterns achievable using traditional beamformers depend on the number of microphones and the array aperture. Generally, their effectiveness degrades for compact arrays. To overcome these limitations, we propose a neural directional filtering (NDF) approach that leverages deep neural networks to enable sound capture with a predefined directivity pattern. The NDF computes a single-channel complex mask from the microphone array signals, which is then applied to a reference microphone to produce an output that approximates a virtual directional microphone with the desired directivity pattern. We introduce training strategies and propose data-dependent metrics to evaluate the directivity pattern and directivity factor. We show that the proposed method: i) achieves a frequency-invariant directivity pattern even above the spatial aliasing frequency, ii) can approximate diverse and higher-order patterns, iii) can steer the pattern in different directions, and iv) generalizes to unseen conditions. Lastly, experimental comparisons demonstrate superior performance over conventional beamforming and parametric approaches.
>
---
#### [replaced 006] DOTA-ME-CS: Daily Oriented Text Audio-Mandarin English-Code Switching Dataset
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2501.12122v2](https://arxiv.org/pdf/2501.12122v2)**

> **作者:** Yupei Li; Zifan Wei; Heng Yu; Jiahao Xue; Huichi Zhou; Björn W. Schuller
>
> **摘要:** Code-switching, the alternation between two or more languages within communication, poses great challenges for Automatic Speech Recognition (ASR) systems. Existing models and datasets are limited in their ability to effectively handle these challenges. To address this gap and foster progress in code-switching ASR research, we introduce the DOTA-ME-CS: Daily oriented text audio Mandarin-English code-switching dataset, which consists of 18.54 hours of audio data, including 9,300 recordings from 34 participants. To enhance the dataset's diversity, we apply artificial intelligence (AI) techniques such as AI timbre synthesis, speed variation, and noise addition, thereby increasing the complexity and scalability of the task. The dataset is carefully curated to ensure both diversity and quality, providing a robust resource for researchers addressing the intricacies of bilingual speech recognition with detailed data analysis. We further demonstrate the dataset's potential in future research. The DOTA-ME-CS dataset, along with accompanying code, will be made publicly available.
>
---
#### [replaced 007] Backdoor Attacks Against Speech Language Models
- **分类: cs.CL; cs.CR; cs.SD**

- **链接: [https://arxiv.org/pdf/2510.01157v2](https://arxiv.org/pdf/2510.01157v2)**

> **作者:** Alexandrine Fortier; Thomas Thebaud; Jesús Villalba; Najim Dehak; Patrick Cardinal
>
> **摘要:** Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.
>
---
#### [replaced 008] A Phase Synthesizer for Decorrelation to Improve Acoustic Feedback Cancellation
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2510.12377v2](https://arxiv.org/pdf/2510.12377v2)**

> **作者:** Klaus Linhard; Philipp Bulling
>
> **摘要:** Undesired acoustic feedback is a known issue in communication systems, such as speech in-car communication, public address systems, or hearing aids. Without additional precautions, there is a high risk that the adaptive filter - intended to cancel the feedback path - also suppresses parts of the desired signal. One solution is to decorrelate the loudspeaker and microphone signals. In this work, we combine the two decorrelation approaches frequency shifting and phase modulation in a unified framework: a so-called \textit{phase synthesizer}, implemented in a discrete Fourier transform (DFT) filter bank. Furthermore, we extend the phase modulation technique using variable delay lines, as known from vibrato and chorus effects. We demonstrate the benefits of the proposed phase synthesizer using an example from speech in-car communication, employing an adaptive frequency-domain Kalman filter. Improvements in system stability, speech quality measured by perceptual evaluation of speech quality (PESQ) are presented.
>
---
#### [replaced 009] Unmasking Deepfakes: Leveraging Augmentations and Features Variability for Deepfake Speech Detection
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2501.05545v2](https://arxiv.org/pdf/2501.05545v2)**

> **作者:** Inbal Rimon; Oren Gal; Haim Permuter
>
> **摘要:** Deepfake speech detection presents a growing challenge as generative audio technologies continue to advance. We propose a hybrid training framework that advances detection performance through novel augmentation strategies. First, we introduce a dual-stage masking approach that operates both at the spectrogram level (MaskedSpec) and within the latent feature space (MaskedFeature), providing complementary regularization that improves tolerance to localized distortions and enhances generalization learning. Second, we introduce compression-aware strategy during self-supervised to increase variability in low-resource scenarios while preserving the integrity of learned representations, thereby improving the suitability of pretrained features for deepfake detection. The framework integrates a learnable self-supervised feature extractor with a ResNet classification head in a unified training pipeline, enabling joint adaptation of acoustic representations and discriminative patterns. On the ASVspoof5 Challenge (Track~1), the system achieves state-of-the-art results with an Equal Error Rate (EER) of 4.08% under closed conditions, further reduced to 2.71% through fusion of models with diverse pretrained feature extractors. when trained on ASVspoof2019, our system obtaining leading performance on the ASVspoof2019 evaluation set (0.18% EER) and the ASVspoof2021 DF task (2.92% EER).
>
---
#### [replaced 010] Disentangling the effects of peripheral hearing loss and higher-level processes on speech intelligibility in older adults
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2510.25235v2](https://arxiv.org/pdf/2510.25235v2)**

> **作者:** Toshio Irino; Ayako Yamamoto; Fuki Miyazaki
>
> **备注:** This manuscript was submitted to Trends in Hearing on November 13, 2025, after editorial revision
>
> **摘要:** This paper introduces a novel approach to disentangle the effects of peripheral hearing loss (HL) and higher-level processes on speech intelligibility (SI). We conducted an SI experiment with 15 young normal-hearing (YNH) listeners using stimuli processed by the WHIS simulator to emulate the hearing loss profile of a specific older adult (OA) from a previous study involving 14 OA participants. Speech-in-noise materials were presented either with ideal ratio mask (IRM) enhancement or in an unprocessed form. Results showed that the target OA achieved higher SI scores than the average YNH listener, suggesting that the OA's higher-level processes may perform more effectively than those of younger listeners. To examine the characteristics of the remaining OAs, we employed the GESI objective intelligibility measure to predict SI performance. GESI provided reasonably accurate predictions for both YNH and OA listeners. Using parameters estimated from the YNH experiment, we predicted SI scores for the 14 OA participants. The results revealed substantial variability: several OAs achieved higher SI scores than the average YNH listener, while one OA scored lower. These differences likely reflect individual variations in the efficiency of higher-level processing. Overall, these findings demonstrate that WHIS and GESI enable contrastive experiments between YNH and OA listeners, independent of hearing level, and offer a framework for investigating the role of higher-level processes in older adults on an individual basis.
>
---
#### [replaced 011] Can Current Detectors Catch Face-to-Voice Deepfake Attacks?
- **分类: cs.CR; cs.LG; cs.MM; cs.SD**

- **链接: [https://arxiv.org/pdf/2510.21004v2](https://arxiv.org/pdf/2510.21004v2)**

> **作者:** Nguyen Linh Bao Nguyen; Alsharif Abuadbba; Kristen Moore; Tingmin Wu
>
> **备注:** 8 pages, Accepted at Workshop on AI for Cyber Threat Intelligence, co-located with ACSAC 2025
>
> **摘要:** The rapid advancement of generative models has enabled the creation of increasingly stealthy synthetic voices, commonly referred to as audio deepfakes. A recent technique, FOICE [USENIX'24], demonstrates a particularly alarming capability: generating a victim's voice from a single facial image, without requiring any voice sample. By exploiting correlations between facial and vocal features, FOICE produces synthetic voices realistic enough to bypass industry-standard authentication systems, including WeChat Voiceprint and Microsoft Azure. This raises serious security concerns, as facial images are far easier for adversaries to obtain than voice samples, dramatically lowering the barrier to large-scale attacks. In this work, we investigate two core research questions: (RQ1) can state-of-the-art audio deepfake detectors reliably detect FOICE-generated speech under clean and noisy conditions, and (RQ2) whether fine-tuning these detectors on FOICE data improves detection without overfitting, thereby preserving robustness to unseen voice generators such as SpeechT5. Our study makes three contributions. First, we present the first systematic evaluation of FOICE detection, showing that leading detectors consistently fail under both standard and noisy conditions. Second, we introduce targeted fine-tuning strategies that capture FOICE-specific artifacts, yielding significant accuracy improvements. Third, we assess generalization after fine-tuning, revealing trade-offs between specialization to FOICE and robustness to unseen synthesis pipelines. These findings expose fundamental weaknesses in today's defenses and motivate new architectures and training protocols for next-generation audio deepfake detection.
>
---
