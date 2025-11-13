# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
- **分类: cs.SD; cs.CL**

- **简介: 论文提出CLSR，一种端到端对比语言-语音检索器，用于长音频口语问答。通过将声学特征先转为类文本表示，提升跨模态对齐效果，显著优于现有语音检索方法，为长音频SQA提供新方案。**

- **链接: []()**

> **作者:** Jiliang Hu; Zuchao Li; Baoyuan Qi; Liu Guoming; Ping Wang
>
> **备注:** 12 pages, 7 figures, accepted by AAAI 2026
>
> **摘要:** Significant progress has been made in spoken question answering (SQA) in recent years. However, many existing methods, including large audio language models, struggle with processing long audio. Follow the success of retrieval augmented generation, a speech-related retriever shows promising in help preprocessing long-form speech. But the performance of existing speech-related retrievers is lacking. To address this challenge, we propose CLSR, an end-to-end contrastive language-speech retriever that efficiently extracts question-relevant segments from long audio recordings for downstream SQA task. Unlike conventional speech-text contrastive models, CLSR incorporates an intermediate step that converts acoustic features into text-like representations prior to alignment, thereby more effectively bridging the gap between modalities. Experimental results across four cross-modal retrieval datasets demonstrate that CLSR surpasses both end-to-end speech related retrievers and pipeline approaches combining speech recognition with text retrieval, providing a robust foundation for advancing practical long-form SQA applications.
>
---
#### [new 002] Chord-conditioned Melody and Bass Generation
- **分类: cs.SD**

- **简介: 该论文研究和比较五种基于Transformer的和弦条件化旋律与贝斯生成方法，旨在提升生成音乐的和弦音使用与风格一致性，发现贝斯优先模型效果最佳。**

- **链接: []()**

> **作者:** Alexandra C Salem; Mohammad Shokri; Johanna Devaney
>
> **备注:** To appear at NeurIPS 2025 Workshop on AI for Music (AI4Music)
>
> **摘要:** We evaluate five Transformer-based strategies for chord-conditioned melody and bass generation using a set of music theory-motivated metrics capturing pitch content, pitch interval size, and chord tone usage. The evaluated models include (1) no chord conditioning, (2) independent line chord-conditioned generation, (3) bass-first chord-conditioned generation, (4) melody-first chord-conditioned generation, and (5) chord-conditioned co-generation. We show that chord-conditioning improves the replication of stylistic pitch content and chord tone usage characteristics, particularly for the bass-first model.
>
---
#### [new 003] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-aware Speech-to-Speech Interaction
- **分类: eess.AS; eess.SP**

- **简介: 论文提出ParaS2S，面向口语对话中的语用特征（如情绪、语调）建模，解决现有S2S模型忽视风格响应的问题。构建ParaS2SBench评估基准，结合RL框架GRPO，实现波形级风格与内容协同优化，显著超越监督微调方法。**

- **链接: []()**

> **作者:** Shu-wen Yang; Ming Tu; Andy T. Liu; Xinghua Qu; Hung-yi Lee; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **摘要:** Speech-to-Speech (S2S) models have shown promising dialogue capabilities, but their ability to handle paralinguistic cues--such as emotion, tone, and speaker attributes--and to respond appropriately in both content and style remains underexplored. Progress is further hindered by the scarcity of high-quality and expressive demonstrations. To address this, we introduce a novel reinforcement learning (RL) framework for paralinguistic-aware S2S, ParaS2S, which evaluates and optimizes both content and speaking style directly at the waveform level. We first construct ParaS2SBench, a benchmark comprehensively evaluates S2S models' output for content and style appropriateness from diverse and challenging input queries. It scores the fitness of input-output pairs and aligns well with human judgments, serving as an automatic judge for model outputs. With this scalable scoring feedback, we enable the model to explore and learn from diverse unlabeled speech via Group Relative Policy Optimization (GRPO). Experiments show that existing S2S models fail to respond appropriately to paralinguistic attributes, performing no better than pipeline-based baselines. Our RL approach achieves a 11% relative improvement in response content and style's appropriateness on ParaS2SBench over supervised fine-tuning (SFT), surpassing all prior models while requiring substantially fewer warm-up annotations than pure SFT.
>
---
#### [new 004] Sound impact of simple viscoelastic damping changes due to aging and the role of the double bentside on soundboard tension in a 1755 Dulcken harpsichord
- **分类: cs.SD; nlin.AO**

- **简介: 该论文研究1755年Dulcken大键琴因木材老化导致阻尼变化对音色亮度的影响，通过FDTD模拟声板响应，发现高频弦亮度反而降低；并用FEM验证双弯边结构对声板张力无显著影响，揭示其设计另有原因。**

- **链接: []()**

> **作者:** Rolf Bader; Niko Plath; Patrick Kontopidis
>
> **摘要:** The sound perception of wood aging is investigated on a Dulcken harpsichord of 1755 from the Museum of Applied Arts in Hamburg, Germany using a Finite-Difference Time Domain (FDTD) model of the harpsichords soundboard. The soundboard thickness was measured on the instrument at 497 positions during strings being deattached and used in the model. Impulse responses were taken on the instrument to estimate the present internal damping by calculating the T60 decay time and used as a model input. By varying the internal damping from this measured damping as a logarithmic decrement, impulse responses were simulated at 52 string positions on both, the 8' and 4' bridge. To estimate the changed sound brightness due to changed internal damping, spectral centroids were calculated from the simulated impulse responses. A dependency of brightness change due to aging on string position was found, where the lower strings have higher brightness, as expected, while the higher strings have decreased brightness. This counterintuitive finding is caused by the frequency-dependent filter effect of changed damping. Future studies need to incorporate viscoelasticity to differentiate this effect further. Furthermore, the attachment of the 8' string to the outer instead of the inner wall, a characteristic feature of Dulcken harpsichords, is investigated using a 3D Finite-Element Method (FEM) model simulation of the whole instrument. No considerable changes on the soundboard tension were found compared to an attachment of the 8' strings to the inner wall, pointing to another reason for this special construction.
>
---
#### [new 005] Non-verbal Perception of Room Acoustics using Multi Dimensional Scaling Metho
- **分类: cs.SD; nlin.AO**

- **简介: 该论文研究非语言环境下人们对厅堂声学的感知，采用多维尺度（MDS）分析音乐与双耳脉冲响应卷积后的主观体验，识别出5个关键感知维度，解决传统主观评价依赖记忆或双极量表的问题。**

- **链接: []()**

> **作者:** Leonie Böhlke; Tim Ziemer; Rolf Bader
>
> **摘要:** Subjective room acoustics impressions play an important role for the performance and reception of music in concert venues and auralizations. Therefore, room acoustics since the 20th century dealt with the relationship between objective, acoustic parameters and subjective impressions of room acoustics. One common approach is to correlate acoustic measures with experts' subjective ratings of rooms as recalled from their long-term memory, and explain them using acoustical measures. Another approach is to let listeners rate auralized room acoustics on bipolar scales and find objective correlates. In this study, we present an alternative approach to characterizing the subjective impressions of room acoustics. We concolve music with binaural room impulse response measurements and utilize Multi Dimensional Scaling (MDS) to identify the perceptual dimensions of room acoustics. Results show that the perception of room acoustics has $5$ dimensions that can be explained by the (psycho-)acoustical measures echo density, fractal correlation dimension, roughness, loudness, and early decay time.
>
---
#### [new 006] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出Diff-V2M，用于视频到音乐生成任务，解决节奏建模缺失与视觉特征融合困难问题。通过层次化扩散模型，结合显式节奏预测与多特征条件融合，实现音画时序对齐与情感一致生成。**

- **链接: []()**

> **作者:** Shulei Ji; Zihao Wang; Jiaxing Yu; Xiangyuan Yang; Shuyu Li; Songruoyao Wu; Kejun Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Video-to-music (V2M) generation aims to create music that aligns with visual content. However, two main challenges persist in existing methods: (1) the lack of explicit rhythm modeling hinders audiovisual temporal alignments; (2) effectively integrating various visual features to condition music generation remains non-trivial. To address these issues, we propose Diff-V2M, a general V2M framework based on a hierarchical conditional diffusion model, comprising two core components: visual feature extraction and conditional music generation. For rhythm modeling, we begin by evaluating several rhythmic representations, including low-resolution mel-spectrograms, tempograms, and onset detection functions (ODF), and devise a rhythmic predictor to infer them directly from videos. To ensure contextual and affective coherence, we also extract semantic and emotional features. All features are incorporated into the generator via a hierarchical cross-attention mechanism, where emotional features shape the affective tone via the first layer, while semantic and rhythmic features are fused in the second cross-attention layer. To enhance feature integration, we introduce timestep-aware fusion strategies, including feature-wise linear modulation (FiLM) and weighted fusion, allowing the model to adaptively balance semantic and rhythmic cues throughout the diffusion process. Extensive experiments identify low-resolution ODF as a more effective signal for modeling musical rhythm and demonstrate that Diff-V2M outperforms existing models on both in-domain and out-of-domain datasets, achieving state-of-the-art performance in terms of objective metrics and subjective comparisons. Demo and code are available at https://Tayjsl97.github.io/Diff-V2M-Demo/.
>
---
#### [new 007] Towards Effective and Efficient Non-autoregressive decoders for Conformer and LLM-based ASR using Block-based Attention Mask
- **分类: eess.AS**

- **简介: 该论文面向语音识别（ASR）任务，提出一种基于块注意力掩码的非自回归解码器（AMD），在保持精度前提下提升Conformer和LLM模型的推理速度，实现并行解码与动态概率融合，显著加速解码过程。**

- **链接: []()**

> **作者:** Tianzi Wang; Xurong Xie; Zengrui Jin; Mengzhe Geng; Jiajun Deng; Zhaoqing Li; Shoukang Hu; Shujie Hu; Guinan Li; Mingyu Cui; Helen Meng; Xunying Liu
>
> **备注:** Accepted by regular paper in the IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Automatic speech recognition (ASR) systems often rely on autoregressive (AR) Transformer decoder architectures, which limit efficient inference parallelization due to their sequential nature. To this end, non-autoregressive (NAR) approaches aim primarily to achieve significant decoding speedup while the maintaining recognition accuracy that is comparable to AR baselines. This paper proposes a novel NAR block-based attention mask decoder (AMD) that effectively improves decoding efficiency while maintaining ASR accuracy, and also offering flexibility in balancing the performance-efficiency trade-off on both Conformer and large language model (LLM)-based ASR systems. The proposed AMD performs parallel inference within contiguous blocks of output labels while maintaining monotonic left-to-right prediction between blocks. A one-pass beam search algorithm is designed to dynamically fuse Connectionist Temporal Classification (CTC), AR decoder, and AMD probabilities. Experiments are conducted on normal speech LS960 and DBank elderly speech across: a) The Conformer encoder-decoder ASR system with filterbank input features; b) its integration with WavLM features; and c) further advancement by integrating an LLM-based decoder. On the LS960 task, the proposed AMD empowered tripartite decoder achieves decoding speedup ratios of up to 1.44x, 1.55x, and 2.31x under the three model configurations over the CTC + AR baselines, without statistically significant WER increases. When operating with real-time factors (RTFs) comparable to the baselines, the tripartite decoder produces statistically significant WER reductions of 0.19%, 0.62% and 0.13% absolute (4.3%, 16.3%, and 3.8% relative). Similar improvements are also obtained on the DBank task.
>
---
#### [new 008] Robust Multi-modal Task-oriented Communications with Redundancy-aware Representations
- **分类: eess.IV; cs.MM; cs.SD**

- **简介: 该论文面向多模态任务型通信，解决噪声信道下模态冗余与语义可靠性矛盾问题，提出双阶段VIB框架，通过模态内压缩与互信息最小化抑制冗余，提升低信噪比下的情感识别准确率与鲁棒性。**

- **链接: []()**

> **作者:** Jingwen Fu; Ming Xiao; Zhonghao Lyu; Mikael Skoglund; Celimuge Wu
>
> **摘要:** Semantic communications for multi-modal data can transmit task-relevant information efficiently over noisy and bandwidth-limited channels. However, a key challenge is to simultaneously compress inter-modal redundancy and improve semantic reliability under channel distortion. To address the challenge, we propose a robust and efficient multi-modal task-oriented communication framework that integrates a two-stage variational information bottleneck (VIB) with mutual information (MI) redundancy minimization. In the first stage, we apply uni-modal VIB to compress each modality separately, i.e., text, audio, and video, while preserving task-specific features. To enhance efficiency, an MI minimization module with adversarial training is then used to suppress cross-modal dependencies and to promote complementarity rather than redundancy. In the second stage, a multi-modal VIB is further used to compress the fused representation and to enhance robustness against channel distortion. Experimental results on multi-modal emotion recognition tasks demonstrate that the proposed framework significantly outperforms existing baselines in accuracy and reliability, particularly under low signal-to-noise ratio regimes. Our work provides a principled framework that jointly optimizes modality-specific compression, inter-modal redundancy, and communication reliability.
>
---
#### [new 009] Spatial Audio Rendering for Real-Time Speech Translation in Virtual Meetings
- **分类: cs.HC; cs.SD**

- **简介: 该论文研究空间音频渲染对实时语音翻译在虚拟会议中理解力、认知负荷与用户体验的影响。通过对比四种音频条件，发现空间音频使理解率翻倍，提升清晰度与参与感，推动包容性跨语言通信系统设计。**

- **链接: []()**

> **作者:** Margarita Geleta; Hong Sodoma; Hannes Gamper
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Language barriers in virtual meetings remain a persistent challenge to global collaboration. Real-time translation offers promise, yet current integrations often neglect perceptual cues. This study investigates how spatial audio rendering of translated speech influences comprehension, cognitive load, and user experience in multilingual meetings. We conducted a within-subjects experiment with 8 bilingual confederates and 47 participants simulating global team meetings with English translations of Greek, Kannada, Mandarin Chinese, and Ukrainian - languages selected for their diversity in grammar, script, and resource availability. Participants experienced four audio conditions: spatial audio with and without background reverberation, and two non-spatial configurations (diotic, monaural). We measured listener comprehension accuracy, workload ratings, satisfaction scores, and qualitative feedback. Spatially-rendered translations doubled comprehension compared to non-spatial audio. Participants reported greater clarity and engagement when spatial cues and voice timbre differentiation were present. We discuss design implications for integrating real-time translation into meeting platforms, advancing inclusive, cross-language communication in telepresence systems.
>
---
#### [new 010] POTSA: A Cross-Lingual Speech Alignment Framework for Low Resource Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: POTSA提出一种跨语言语音对齐框架，用于低资源语音到文本翻译任务，解决源语言语义共性被忽视导致的偏差问题。通过最优传输与偏差补偿模块，实现跨语言表征对齐，在仅10小时数据下显著提升翻译性能。**

- **链接: []()**

> **作者:** Xuanchen Li; Chenrui Cui; Tianrui Wang; Meng Ge; Zikang Huang; Jin Li; Yizhou Peng; Longbiao Wang; Jianwu Dang; Nyima Tashi
>
> **备注:** 5 pages, 3 figures, submitted to ICASSP 2026
>
> **摘要:** Speech Large Language Models (SpeechLLMs) have achieved breakthroughs in multilingual speech-to-text translation (S2TT). However, existing approaches often overlook semantic commonalities across source languages, leading to biased translation performance. In this work, we propose \textbf{POTSA} (Parallel Optimal Transport for Speech Alignment), a new framework based on cross-lingual parallel speech pairs and Optimal Transport (OT), designed to bridge high- and low-resource translation gaps. First, we introduce a Bias Compensation module to coarsely align initial speech representations across languages. Second, we impose token-level OT constraints on a Q-Former using parallel speech pairs to establish fine-grained consistency of representations. Then, we apply a layer scheduling strategy to focus OT constraints on the most semantically beneficial layers. Experiments on the FLEURS dataset show that our method achieves SOTA performance, with +0.93 BLEU on average over five common languages and +5.05 BLEU on zero-shot languages, using only 10 hours of parallel speech per source language.
>
---
## 更新

#### [replaced 001] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
- **分类: cs.SD**

- **链接: []()**

> **作者:** Bingshen Mu; Hexin Liu; Hongfei Xue; Kun Wei; Lei Xie
>
> **备注:** AAAI 2026
>
> **摘要:** Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
>
---
#### [replaced 002] MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages
- **分类: cs.SD; cs.AI**

- **链接: []()**

> **作者:** Hardik B. Sailor; Aw Ai Ti; Chen Fang Yih Nancy; Chiu Ying Lay; Ding Yang; He Yingxu; Jiang Ridong; Li Jingtao; Liao Jingyi; Liu Zhuohan; Lu Yanfeng; Ma Yi; Manas Gupta; Muhammad Huzaifah Bin Md Shahrin; Nabilah Binte Md Johan; Nattadaporn Lertcheva; Pan Chunlei; Pham Minh Duc; Siti Maryam Binte Ahmad Subaidi; Siti Umairah Binte Mohammad Salleh; Sun Shuo; Tarun Kumar Vangani; Wang Qiongqiong; Won Cheng Yi Lewis; Wong Heng Meng Jeremy; Wu Jinyang; Zhang Huayun; Zhang Longyin; Zou Xunlong
>
> **备注:** https://huggingface.co/MERaLiON/MERaLiON-SER-v1
>
> **摘要:** We present MERaLiON-SER, a robust speech emotion recognition model designed for English and Southeast Asian languages. The model is trained using a hybrid objective combining weighted categorical cross-entropy and Concordance Correlation Coefficient (CCC) losses for joint discrete and dimensional emotion modelling. This dual approach enables the model to capture both the distinct categories of emotion (like happy or angry) and the fine-grained, such as arousal (intensity), valence (positivity/negativity), and dominance (sense of control), leading to a more comprehensive and robust representation of human affect. Extensive evaluations across multilingual Singaporean languages (English, Chinese, Malay, and Tamil ) and other public benchmarks show that MERaLiON-SER consistently surpasses both open-source speech encoders and large Audio-LLMs. These results underscore the importance of specialised speech-only models for accurate paralinguistic understanding and cross-lingual generalisation. Furthermore, the proposed framework provides a foundation for integrating emotion-aware perception into future agentic audio systems, enabling more empathetic and contextually adaptive multimodal reasoning.
>
---
#### [replaced 003] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: []()**

> **作者:** Bingsong Bai; Yizhong Geng; Fengping Wang; Cong Wang; Puyuan Guo; Yingming Gao; Ya Li
>
> **备注:** Accepted by AAAI 2026 main technical track
>
> **摘要:** Zero-shot singing voice conversion (SVC) transforms a source singer's timbre to an unseen target speaker's voice while preserving melodic content without fine-tuning. Existing methods model speaker timbre and vocal content separately, losing essential acoustic information that degrades output quality while requiring significant computational resources. To overcome these limitations, we propose HQ-SVC, an efficient framework for high-quality zero-shot SVC. HQ-SVC first extracts jointly content and speaker features using a decoupled codec. It then enhances fidelity through pitch and volume modeling, preserving critical acoustic information typically lost in separate modeling approaches, and progressively refines outputs via differentiable signal processing and diffusion techniques. Evaluations confirm HQ-SVC significantly outperforms state-of-the-art zero-shot SVC methods in conversion quality and efficiency. Beyond voice conversion, HQ-SVC achieves superior voice naturalness compared to specialized audio super-resolution methods while natively supporting voice super-resolution tasks.
>
---
#### [replaced 004] SteerMusic: Enhanced Musical Consistency for Zero-shot Text-Guided and Personalized Music Editing
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: []()**

> **作者:** Xinlei Niu; Kin Wai Cheuk; Jing Zhang; Naoki Murata; Chieh-Hsin Lai; Michele Mancusi; Woosung Choi; Giorgio Fabbro; Wei-Hsiang Liao; Charles Patrick Martin; Yuki Mitsufuji
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Music editing is an important step in music production, which has broad applications, including game development and film production. Most existing zero-shot text-guided editing methods rely on pretrained diffusion models by involving forward-backward diffusion processes. However, these methods often struggle to preserve the musical content. Additionally, text instructions alone usually fail to accurately describe the desired music. In this paper, we propose two music editing methods that improve the consistency between the original and edited music by leveraging score distillation. The first method, SteerMusic, is a coarse-grained zero-shot editing approach using delta denoising score. The second method, SteerMusic+, enables fine-grained personalized music editing by manipulating a concept token that represents a user-defined musical style. SteerMusic+ allows for the editing of music into user-defined musical styles that cannot be achieved by the text instructions alone. Experimental results show that our methods outperform existing approaches in preserving both music content consistency and editing fidelity. User studies further validate that our methods achieve superior music editing quality.
>
---
#### [replaced 005] Two-stage Audio-Visual Target Speaker Extraction System for Real-Time Processing On Edge Device
- **分类: cs.SD; eess.AS**

- **链接: []()**

> **作者:** Zixuan Li; Xueliang Zhang; Lei Miao; Zhipeng Yan; Ying Sun; Chong Zhu
>
> **摘要:** Audio-Visual Target Speaker Extraction (AVTSE) aims to isolate a target speaker's voice in a multi-speaker environment with visual cues as auxiliary. Most of the existing AVTSE methods encode visual and audio features simultaneously, resulting in extremely high computational complexity and making it impractical for real-time processing on edge devices. To tackle this issue, we proposed a two-stage ultra-compact AVTSE system. Specifically, in the first stage, a compact network is employed for voice activity detection (VAD) using visual information. In the second stage, the VAD results are combined with audio inputs to isolate the target speaker's voice. Experiments show that the proposed system effectively suppresses background noise and interfering voices while spending little computational resources.
>
---
#### [replaced 006] SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: []()**

> **作者:** Yang Chen; Hui Wang; Shiyao Wang; Junyang Chen; Jiabei He; Jiaming Zhou; Xi Yang; Yequan Wang; Yonghua Lin; Yong Qin
>
> **摘要:** While voice technologies increasingly serve aging populations, current systems exhibit significant performance gaps due to inadequate training data capturing elderly-specific vocal characteristics like presbyphonia and dialectal variations. The limited data available on super-aged individuals in existing elderly speech datasets, coupled with overly simple recording styles and annotation dimensions, exacerbates this issue. To address the critical scarcity of speech data from individuals aged 75 and above, we introduce SeniorTalk, a carefully annotated Chinese spoken dialogue dataset. This dataset contains 55.53 hours of speech from 101 natural conversations involving 202 participants, ensuring a strategic balance across gender, region, and age. Through detailed annotation across multiple dimensions, it can support a wide range of speech tasks. We perform extensive experiments on speaker verification, speaker diarization, speech recognition, and speech editing tasks, offering crucial insights for the development of speech technologies targeting this age group.
>
---
