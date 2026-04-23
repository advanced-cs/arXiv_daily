# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] ONOTE: Benchmarking Omnimodal Notation Processing for Expert-level Music Intelligence
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于音乐智能领域，旨在解决多模态乐谱处理的基准评估问题。针对现有研究碎片化和度量不准确的问题，提出ONOTE基准，以提升模型对音乐逻辑的理解能力。**

- **链接: [https://arxiv.org/pdf/2604.20719](https://arxiv.org/pdf/2604.20719)**

> **作者:** Menghe Ma; Siqing Wei; Yuecheng Xing; Yaheng Wang; Fanhong Meng; Peijun Han; Luu Anh Tuan; Haoran Luo
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Omnimodal Notation Processing (ONP) represents a unique frontier for omnimodal AI due to the rigorous, multi-dimensional alignment required across auditory, visual, and symbolic domains. Current research remains fragmented, focusing on isolated transcription tasks that fail to bridge the gap between superficial pattern recognition and the underlying musical logic. This landscape is further complicated by severe notation biases toward Western staff and the inherent unreliability of "LLM-as-a-judge" metrics, which often mask structural reasoning failures with systemic hallucinations. To establish a more rigorous standard, we introduce ONOTE, a multi-format benchmark that utilizes a deterministic pipeline--grounded in canonical pitch projection--to eliminate subjective scoring biases across diverse notation systems. Our evaluation of leading omnimodal models exposes a fundamental disconnect between perceptual accuracy and music-theoretic comprehension, providing a necessary framework for diagnosing reasoning vulnerabilities in complex, rule-constrained domains.
>
---
#### [new 002] ATIR: Towards Audio-Text Interleaved Contextual Retrieval
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出ATIR任务，解决音频与文本交错的上下文检索问题。构建基准数据集，改进多模态模型，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.20267](https://arxiv.org/pdf/2604.20267)**

> **作者:** Tong Zhao; Chenghao Zhang; Yutao Zhu; Zhicheng Dou
>
> **摘要:** Audio carries richer information than text, including emotion, speaker traits, and environmental context, while also enabling lower-latency processing compared to speech-to-text pipelines. However, recent multimodal information retrieval research has predominantly focused on images, largely overlooking audio, especially in the setting of interleaved audio-text contextual retrieval. In this work, we introduce the Audio-Text Interleaved contextual Retrieval (ATIR) task, where queries can alternate between audio and text modalities. We construct an ATIR benchmark by integrating several Automatic Speech Recognition (ASR), QA, and retrieval datasets, ultimately unifying four types of contextual retrieval tasks. This benchmark substantially addresses the limitations of existing audio retrieval datasets in semantic retrieval. To study this task, we evaluate several off-the-shelf retrievers and train our ATIR model based on a Multimodal Large Language Model (MLLM). We further introduce a novel token compression mechanism that is orthogonal to existing compression methods, thereby alleviating the issue of excessive audio tokens in MLLM-based ATIR models. Experimental results demonstrate that our ATIR model achieves substantial improvements over strong baselines.
>
---
#### [new 003] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages
- **分类: eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决Indic语言中神经音频编码器合成语音的检测问题。构建了ICF数据集，并提出SATYAM模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.19949](https://arxiv.org/pdf/2604.19949)**

> **作者:** Girish; Mohd Mujtaba Akhtar; Orchid Chetia Phukan; Arun Balaji Buduru
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** The rapid advancement of Audio Large Language Models (ALMs), driven by Neural Audio Codecs (NACs), has led to the emergence of highly realistic speech deepfakes, commonly referred to as CodecFakes (CFs). Consequently, CF detection has attracted increasing attention from the research community. However, existing studies predominantly focus on English or Chinese, leaving the vulnerability of Indic languages largely unexplored. To bridge this gap, we introduce Indic-CodecFake (ICF) dataset, the first large-scale benchmark comprising real and NAC-synthesized speech across multiple Indic languages, diverse speaker profiles, and multiple NAC types. We use IndicSUPERB as the real speech corpus for generation of ICF dataset. Our experiments demonstrate that state-of-the-art (SOTA) CF detectors trained on English-centric datasets fail to generalize to ICF, underscoring the challenges posed by phonetic diversity and prosodic variability in Indic speech. Further, we present systematic evaluation of SOTA ALMs in a zero-shot setting on ICF dataset. We evaluate these ALMs as they have shown effectiveness for different speech tasks. However, our findings reveal that current ALMs exhibit consistently poor performance. To address this, we propose SATYAM, a novel hyperbolic ALM tailored for CF detection in Indic languages. SATYAM integrates semantic representations from Whisper and prosodic representations from TRILLsson using through Bhattacharya distance in hyperbolic space and subsequently performs the same alignment procedure between the fused speech representation and an input conditioning prompt. This dual-stage fusion framework enables SATYAM to effectively model hierarchical relationships both within speech (semantic-prosodic) and across modalities (speech-text). Extensive evaluations show that SATYAM consistently outperforms competitive end-to-end and ALM-based baselines on the ICF benchmark.
>
---
#### [new 004] Enhancing Speaker Verification with Whispered Speech via Post-Processing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决 whispered speech 对说话人验证系统性能的影响。通过改进模型结构，提升系统在 whispered speech 下的鲁棒性，取得显著性能提升。**

- **链接: [https://arxiv.org/pdf/2604.20229](https://arxiv.org/pdf/2604.20229)**

> **作者:** Magdalena Gołębiowska; Piotr Syga
>
> **摘要:** Speaker verification is a task of confirming an individual's identity through the analysis of their voice. Whispered speech differs from phonated speech in acoustic characteristics, which degrades the performance of speaker verification systems in real-life scenarios, including avoiding fully phonated speech to protect privacy, disrupt others, or when the lack of full vocalization is dictated by a disease. In this paper we propose a model with a training recipe to obtain more robust representations against whispered speech hindrances. The proposed system employs an encoder--decoder structure built atop a fine-tuned speaker verification backbone, optimized jointly using cosine similarity--based classification and triplet loss. We gain relative improvement of 22.26\% compared to the baseline (baseline 6.77\% vs ours 5.27\%) in normal vs whispered speech trials, achieving AUC of 98.16\%. In tests comparing whispered to whispered, our model attains an EER of 1.88\% with AUC equal to 99.73\%, which represents a 15\% relative enhancement over the prior leading ReDimNet-B2. We also offer a summary of the most popular and state-of-the-art speaker verification models in terms of their performance with whispered speech. Additionally, we evaluate how these models perform under noisy audios, obtaining that generally the same relative level of noise degrades the performance of speaker verification more significantly on whispered speech than on normal speech.
>
---
#### [new 005] Embedding-Based Intrusive Evaluation Metrics for Musical Source Separation Using MERT Representations
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音乐源分离任务，旨在解决传统评估指标与主观质量评分相关性低的问题，提出基于MERT嵌入的侵入式评估方法，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.20270](https://arxiv.org/pdf/2604.20270)**

> **作者:** Paul A. Bereuter; Alois Sontacchi
>
> **备注:** Presented at DAGA 2026 (Annual German Conference on Acoustics)
>
> **摘要:** Evaluation of musical source separation (MSS) has traditionally relied on Blind Source Separation Evaluation (BSS-Eval) metrics. However, recent work suggests that BSS-Eval metrics exhibit low correlation between metrics and perceptual audio quality ratings from a listening test, which is considered the gold standard evaluation method. As an alternative approach in singing voice separation, embedding-based intrusive metrics that leverage latent representations from large self-supervised audio models such as Music undERstanding with large-scale self-supervised Training (MERT) embeddings have been introduced. In this work, we analyze the correlation of perceptual audio quality ratings with two intrusive embedding-based metrics: a mean squared error (MSE) and an intrusive variant of the Fréchet Audio Distance (FAD) calculated on MERT embeddings. Experiments on two independent datasets show that these metrics correlate more strongly with perceptual audio quality ratings than traditional BSS-Eval metrics across all analyzed stem and model types.
>
---
#### [new 006] Enhancing ASR Performance in the Medical Domain for Dravidian Languages
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升低资源达罗毗荼语在医疗领域的ASR性能。针对数据不足和形态复杂问题，提出一种结合真实与合成数据的置信度训练框架，有效降低词错误率。**

- **链接: [https://arxiv.org/pdf/2604.19797](https://arxiv.org/pdf/2604.19797)**

> **作者:** Sri Charan Devarakonda; Ravi Sastry Kolluru; Manjula Sri Rayudu; Rashmi Kapoor; Madhu G; Anil Kumar Vuppala
>
> **摘要:** Automatic Speech Recognition (ASR) for low-resource Dravidian languages like Telugu and Kannada faces significant challenges in specialized medical domains due to limited annotated data and morphological complexity. This work proposes a novel confidence-aware training framework that integrates real and synthetic speech data through a hybrid confidence mechanism combining static perceptual and acoustic similarity metrics with dynamic model entropy. Unlike direct fine-tuning approaches, the proposed methodology employs both fixed-weight and learnable-weight confidence aggregation strategies to guide sample weighting during training, enabling effective utilization of heterogeneous data sources. The framework is evaluated on Telugu and Kannada medical datasets containing both real recordings and TTS-generated synthetic speech. A 5-gram KenLM language model is applied for post-decoding correction. Results show that the hybrid confidence-aware approach with learnable weights substantially reduces recognition errors: Telugu Word Error Rate (WER) decreases from 24.3% to 15.8% (8.5% absolute improvement), while Kannada WER drops from 31.7% to 25.4% (6.3% absolute improvement), both significantly outperforming standard fine-tuning baselines. These findings confirm that combining adaptive confidence-aware training with statistical language modeling delivers superior performance for domain-specific ASR in morphologically complex Dravidian languages.
>
---
#### [new 007] From Image to Music Language: A Two-Stage Structure Decoding Approach for Complex Polyphonic OMR
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于光学音乐识别任务，解决复杂钢琴乐谱的结构解码问题。通过拓扑识别与概率搜索方法，将符号和事件解码为结构化乐谱。**

- **链接: [https://arxiv.org/pdf/2604.20522](https://arxiv.org/pdf/2604.20522)**

> **作者:** Nan Xu; Shiheng Li; Shengchao Hou
>
> **备注:** 49 pages, 16 figures, 16 tables
>
> **摘要:** We propose a new approach for the second stage of a practical two-stage Optical Music Recognition (OMR) pipeline. Given symbol and event candidates from the visual pipeline, we decode them into an editable, verifiable, and exportable score structure. We focus on complex polyphonic staff notation, especially piano scores, where voice separation and intra-measure timing are the main bottlenecks. Our approach formulates second-stage decoding as a structure decoding problem and uses topology recognition with probability-guided search (BeadSolver) as its core method. We also describe a data strategy that combines procedural generation with recognition-feedback annotations. The result is a practical decoding component for real OMR systems and a path to accumulate structured score data for future end-to-end, multimodal, and RL-style methods.
>
---
#### [new 008] Before the Mic: Physical-Layer Voiceprint Anonymization with Acoustic Metamaterials
- **分类: cs.SD**

- **简介: 该论文属于语音安全任务，旨在解决语音识别信息泄露问题。通过声学超材料实现物理层语音匿名化，有效阻止攻击者获取清晰语音特征。**

- **链接: [https://arxiv.org/pdf/2604.20116](https://arxiv.org/pdf/2604.20116)**

> **作者:** Zhiyuan Ning; Zhanyong Tang; Xiaojiang Chen; Zheng Wang
>
> **摘要:** Voiceprints are widely used for authentication; however, they are easily captured in public settings and cannot be revoked once leaked. Existing anonymization systems operate inside recording devices, which makes them ineffective when microphones or software are untrusted, as in conference rooms, lecture halls, and interviews. We present EchoMask, the first practical physical-layer system for real-time voiceprint anonymization using acoustic metamaterials. By modifying sound waves before they reach the microphone, EchoMask prevents attackers from capturing clean voiceprints through compromised devices. Our design combines three key innovations: frequency-selective interference to disrupt voiceprint features while preserving speech intelligibility, an acoustic-field model to ensure stability under speaker movement, and reconfigurable structures that create time-varying interference to prevent learning or canceling a fixed acoustic pattern. EchoMask is low-cost, power-free, and 3D-printable, requiring no machine learning, software support, or microphone modification. Experiments conducted across eight microphones in diverse environments demonstrate that EchoMask increases the Miss-match Rate, i.e., the fraction of failed voiceprint matching attempts, to over 90%, while maintaining high speech intelligibility.
>
---
#### [new 009] Explainable Speech Emotion Recognition: Weighted Attribute Fairness to Model Demographic Contributions to Social Bias
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决模型中的社会偏见问题。通过引入加权属性公平性方法，分析不同人口属性对偏见的贡献，评估并揭示了模型中的性别偏见。**

- **链接: [https://arxiv.org/pdf/2604.19763](https://arxiv.org/pdf/2604.19763)**

> **作者:** Tomisin Ogunnubi; Yupei Li; Björn Schuller
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Speech Emotion Recognition (SER) systems have growing applications in sensitive domains such as mental health and education, where biased predictions can cause harm. Traditional fairness metrics, such as Equalised Odds and Demographic Parity, often overlook the joint dependency between demographic attributes and model predictions. We propose a fairness modelling approach for SER that explicitly captures allocative bias by learning the joint relationship between demographic attributes and model error. We validate our fairness metric on synthetic data, then apply it to evaluate HuBERT and WavLM models finetuned on the CREMA-D dataset. Our results indicate that the proposed fairness model captures more mutual information between protected attributes and biases and quantifies the absolute contribution of individual attributes to bias in SSL-based SER models. Additionally, our analysis reveals indications of gender bias in both HuBERT and WavLM.
>
---
#### [new 010] Utterance-Level Methods for Identifying Reliable ASR-Output for Child Speech
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别可靠性评估任务，旨在解决儿童语音ASR输出不可靠的问题。通过提出两种新的话语级选择方法，提高可靠转录的识别精度。**

- **链接: [https://arxiv.org/pdf/2604.19801](https://arxiv.org/pdf/2604.19801)**

> **作者:** Gus Lathouwers; Lingyun Gao; Catia Cucchiarini; Helmer Strik
>
> **备注:** Submitted for Interspeech 2026, currently under review
>
> **摘要:** Automatic Speech Recognition (ASR) is increasingly used in applications involving child speech, such as language learning and literacy acquisition. However, the effectiveness of such applications is limited by high ASR error rates. The negative effects can be mitigated by identifying in advance which ASR-outputs are reliable. This work aims to develop two novel approaches for selecting reliable ASR-output at the utterance level, one for selecting reliable read speech and one for dialogue speech material. Evaluations were done on an English and a Dutch dataset, each with a baseline and finetuned model. The results show that utterance-level selection methods for identifying reliably transcribed speech recordings have high precision for the best strategy (P > 97.4) for both read speech and dialogue material, for both languages. Using the current optimal strategy allows 21.0% to 55.9% of dialogue/read speech datasets to be automatically selected with low (UER of < 2.6) error rates.
>
---
#### [new 011] SpeechParaling-Bench: A Comprehensive Benchmark for Paralinguistic-Aware Speech Generation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音生成任务，旨在解决LALMs在paralinguistic特征建模上的不足。提出SpeechParaling-Bench基准，扩展特征覆盖并设计评估方法，揭示现有模型的局限性。**

- **链接: [https://arxiv.org/pdf/2604.20842](https://arxiv.org/pdf/2604.20842)**

> **作者:** Ruohan Liu; Shukang Yin; Tao Wang; Dong Zhang; Weiji Zhuang; Shuhuai Ren; Ran He; Caifeng Shan; Chaoyou Fu
>
> **备注:** Project page: this https URL
>
> **摘要:** Paralinguistic cues are essential for natural human-computer interaction, yet their evaluation in Large Audio-Language Models (LALMs) remains limited by coarse feature coverage and the inherent subjectivity of assessment. To address these challenges, we introduce SpeechParaling-Bench, a comprehensive benchmark for paralinguistic-aware speech generation. It expands existing coverage from fewer than 50 to over 100 fine-grained features, supported by more than 1,000 English-Chinese parallel speech queries, and is organized into three progressively challenging tasks: fine-grained control, intra-utterance variation, and context-aware adaptation. To enable reliable evaluation, we further develop a pairwise comparison pipeline, in which candidate responses are evaluated against a fixed baseline by an LALM-based judge. By framing evaluation as relative preference rather than absolute scoring, this approach mitigates subjectivity and yields more stable and scalable assessments without costly human annotation. Extensive experiments reveal substantial limitations in current LALMs. Even leading proprietary models struggle with comprehensive static control and dynamic modulation of paralinguistic features, while failure to correctly interpret paralinguistic cues accounts for 43.3% of errors in situational dialogue. These findings underscore the need for more robust paralinguistic modeling toward human-aligned voice assistants.
>
---
#### [new 012] KoALa-Bench: Evaluating Large Audio Language Models on Korean Speech Understanding and Faithfulness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出KoALa-Bench，用于评估大音频语言模型在韩语语音理解和忠实度上的表现。旨在解决非英语语言基准不足的问题，涵盖六项任务，包含韩语特定内容。**

- **链接: [https://arxiv.org/pdf/2604.19782](https://arxiv.org/pdf/2604.19782)**

> **作者:** Jinyoung Kim; Hyeongsoo Lim; Eunseo Seo; Minho Jang; Keunwoo Choi; Seungyoun Shin; Ji Won Yoon
>
> **备注:** Under Review
>
> **摘要:** Recent advances in large audio language models (LALMs) have enabled multilingual speech understanding. However, benchmarks for evaluating LALMs remain scarce for non-English languages, with Korean being one such underexplored case. In this paper, we introduce KoALa-Bench, a comprehensive benchmark for evaluating Korean speech understanding and speech faithfulness of LALMs. In particular, KoALa-Bench comprises six tasks. Four tasks evaluate fundamental speech understanding capabilities, including automatic speech recognition, speech translation, speech question answering, and speech instruction following, while the remaining two tasks evaluate speech faithfulness, motivated by our observation that several LALMs often fail to fully leverage the speech modality. Furthermore, to reflect Korea-specific knowledge, our benchmark incorporates listening questions from the Korean college scholastic ability test as well as content covering Korean cultural domains. We conduct extensive experiments across six models, including both white-box and black-box ones. Our benchmark, evaluation code, and leaderboard are publicly available at this https URL.
>
---
#### [new 013] Tonnetz Theory, Classical Harmony, and the Combinatorial Geometry of Abstract Musical Resources
- **分类: math.CO; eess.AS; math.AG**

- **简介: 该论文将音乐理论与组合几何结合，研究不同音乐系统中的音程关系，通过构建特定配置模型解决音乐结构分析问题。**

- **链接: [https://arxiv.org/pdf/2604.19960](https://arxiv.org/pdf/2604.19960)**

> **作者:** Jeffrey R. Boland; Lane P. Hughston
>
> **备注:** 26 pp, 18 figs. Our earlier submission 2505.08752v4 (55 pp) has now been split into two independent articles. The first of these appears as 2505.08752v6 (37 pp, 19 figs) with title "Configurations, Tessellations and Tone Networks". The second is the present submission, with title "Tonnetz Theory, Classical Harmony, and the Combinatorial Geometry of Abstract Musical Resources". arXiv admin note: text overlap with arXiv:2505.08752
>
> **摘要:** In a previous submission, we established a fundamental relation between tone networks and configurations. It was shown that the Eulerian tonnetz can be represented by a $\{12_3\}$ of Daublebsky von Sterneck type D222. We also constructed a tonnetz for Tristan-genus chords (dominant sevenths and half-diminished sevenths) and we showed that this tonnetz can be represented by a $\{12_3\}$ of type D228. In both of these constructions the associated Levi graphs play an important role. Here we look at the tonnetze associated with some other musical systems, thereby offering several concrete examples of an abstract view of music as combinatorial geometry. First, we look at the tonal harmonies typical of the classical period. In the case of diatonic triads, we show the existence of a bipartite graph of type $\{7_3\}$ and girth four that represents the well-known relations between the seven diatonic degrees and their pitch classes. In the case of diatonic seventh chords, we obtain a Fano configuration $\{7_3\}$ which gives a complete characterization of the voice-leading relations that hold between such chords. Next, we construct a tonnetz for pentatonic music based on the Desargues configuration $\{10_3\}$ and we construct a tonnetz for the 12-tone system based on the Cremona-Richmond configuration $\{15_3\}$. Both can be used as a resource for musical compositions. Finally, we show that the relation between the chromatic pitch class set and the major triad set is also represented by a D222. The minor triads are in one-to-one correspondence with the members of a certain class of hexacycles in the Levi graph of this configuration. In this way, the characteristic duality between major and minor triads in the tonnetz can be broken.
>
---
## 更新

#### [replaced 001] Constraint Optimized Multichannel Mixer-limiter Design
- **分类: cs.SD; eess.AS; eess.SP; math.OC**

- **简介: 该论文属于音频处理任务，解决多通道混音与限制器设计问题。通过耦合设计降低失真，并优化计算效率。**

- **链接: [https://arxiv.org/pdf/2507.06769](https://arxiv.org/pdf/2507.06769)**

> **作者:** Yuancheng Luo; Dmitriy Yamkovoy; Guillermo Garcia
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Multichannel audio mixer and limiter designs are conventionally decoupled for content reproduction over loudspeaker arrays due to high computational complexity and run-time costs. We propose a coupled mixer-limiter-envelope design formulated as an efficient linear-constrained quadratic program that minimizes a distortion objective over multichannel gain variables subject to sample mixture constraints. Novel methods for asymmetric constant overlap-add window optimization, objective function approximation, variable and constraint reduction are presented. Experiments demonstrate distortion reduction of the coupled design, and computational trade-offs required for efficient real-time processing.
>
---
#### [replaced 002] Interpreting Multi-Branch Anti-Spoofing Architectures: Correlating Internal Strategy with Empirical Performance
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究音频反欺骗任务，解决多分支网络内部决策机制不透明的问题，通过分析分支协作与竞争模式，揭示模型性能与结构的关系。**

- **链接: [https://arxiv.org/pdf/2602.17711](https://arxiv.org/pdf/2602.17711)**

> **作者:** Ivan Viakhirev; Kirill Borodin; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** Published at MDPI Mathematics (see at this https URL)
>
> **摘要:** Multi-branch deep neural networks like AASIST3 achieve state-of-the-art comparable performance in audio anti-spoofing, yet their internal decision dynamics remain opaque compared to traditional input-level saliency methods. While existing interpretability efforts largely focus on visualizing input artifacts, the way individual architectural branches cooperate or compete under different spoofing attacks is not well characterized. This paper develops a framework for interpreting AASIST3 at the component level. Intermediate activations from fourteen branches and global attention modules are modeled with covariance operators whose leading eigenvalues form low-dimensional spectral signatures. These signatures train a CatBoost meta-classifier to generate TreeSHAP-based branch attributions, which we convert into normalized contribution shares and confidence scores (Cb) to quantify the model's operational strategy. By analyzing 13 spoofing attacks from the ASVspoof 2019 benchmark, we identify four operational archetypes-ranging from Effective Specialization (e.g., A09, Equal Error Rate (EER) 0.04%, C=1.56) to Ineffective Consensus (e.g., A08, EER 3.14%, C=0.33). Crucially, our analysis exposes a Flawed Specialization mode where the model places high confidence in an incorrect branch, leading to severe performance degradation for attacks A17 and A18 (EER 14.26% and 28.63%, respectively). These quantitative findings link internal architectural strategy directly to empirical reliability, highlighting specific structural dependencies that standard performance metrics overlook.
>
---
#### [replaced 003] Throat and acoustic paired speech dataset for deep learning-based speech enhancement
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决高噪声环境下语音清晰度不足的问题。通过构建TAPS数据集并采用深度学习方法提升喉部麦克风的语音质量。**

- **链接: [https://arxiv.org/pdf/2502.11478](https://arxiv.org/pdf/2502.11478)**

> **作者:** Yunsik Kim; Yonghun Song; Yoonyoung Chung
>
> **摘要:** In high-noise environments such as factories, subways, and busy streets, capturing clear speech is challenging. Throat microphones can offer a solution because of their inherent noise-suppression capabilities; however, the passage of sound waves through skin and tissue attenuates high-frequency information, reducing speech clarity. Recent deep learning approaches have shown promise in enhancing throat microphone recordings, but further progress is constrained by the lack of a standard dataset. Here, we introduce the Throat and Acoustic Paired Speech (TAPS) dataset, a collection of paired utterances recorded from 60 native Korean speakers using throat and acoustic microphones. Furthermore, an optimal alignment approach was developed and applied to address the inherent signal mismatch between the two microphones. We tested three baseline deep learning models on the TAPS dataset and found mapping-based approaches to be superior for improving speech quality and restoring content. These findings demonstrate the TAPS dataset's utility for speech enhancement tasks and support its potential as a standard resource for advancing research in throat microphone-based applications.
>
---
#### [replaced 004] X-VC: Zero-shot Streaming Voice Conversion in Codec Space
- **分类: eess.AS; cs.AI**

- **简介: 该论文提出X-VC，解决零样本语音转换中的高保真与低延迟问题，通过编码器空间的一步转换实现高效语音转换。**

- **链接: [https://arxiv.org/pdf/2604.12456](https://arxiv.org/pdf/2604.12456)**

> **作者:** Qixi Zheng; Yuxiang Zhao; Tianrui Wang; Wenxi Chen; Kele Xu; Yikang Li; Qinyuan Chen; Xipeng Qiu; Kai Yu; Xie Chen
>
> **摘要:** Zero-shot voice conversion (VC) aims to convert a source utterance into the voice of an unseen target speaker while preserving its linguistic content. Although recent systems have improved conversion quality, building zero-shot VC systems for interactive scenarios remains challenging because high-fidelity speaker transfer and low-latency streaming inference are difficult to achieve simultaneously. In this work, we present X-VC, a zero-shot streaming VC system that performs one-step conversion in the latent space of a pretrained neural codec. X-VC uses a dual-conditioning acoustic converter that jointly models source codec latents and frame-level acoustic conditions derived from target reference speech, while injecting utterance-level target speaker information through adaptive normalization. To reduce the mismatch between training and inference, we train the model with generated paired data and a role-assignment strategy that combines standard, reconstruction, and reversed modes. For streaming inference, we further adopt a chunkwise inference scheme with overlap smoothing that is aligned with the segment-based training paradigm of the codec. Experiments on Seed-TTS-Eval show that X-VC achieves the best streaming WER in both English and Chinese, strong speaker similarity in same-language and cross-lingual settings, and substantially lower offline real-time factor than the compared baselines. These results suggest that codec-space one-step conversion is a practical approach for building high-quality low-latency zero-shot VC systems. Our audio samples, code and checkpoints are released at this https URL.
>
---
#### [replaced 005] When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音 spoof 检测任务，旨在解决跨语言检测鲁棒性问题。通过构建多语言数据集，评估不同模型在66种语言中的检测效果，揭示语言对检测性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.02364](https://arxiv.org/pdf/2603.02364)**

> **作者:** Kirill Borodin; Vasiliy Kudryavtsev; Maxim Maslov; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** This paper has been submitted to Interspeech 2026 for review
>
> **摘要:** We introduce LRLspoof, a large-scale multilingual synthetic-speech corpus for cross-lingual spoof detection, comprising 2,732 hours of audio generated with 24 open-source TTS systems across 66 languages, including 45 low-resource languages under our operational definition. To evaluate robustness without requiring target-domain bonafide speech, we benchmark 11 publicly available countermeasures using threshold transfer: for each model we calibrate an EER operating point on pooled external benchmarks and apply the resulting threshold, reporting spoof rejection rate (SRR). Results show model-dependent cross-lingual disparity, with spoof rejection varying markedly across languages even under controlled conditions, highlighting language as an independent source of domain shift in spoof detection. The dataset is publicly available at \href{this https URL}{\textbf{\underline{\textit{HuggingFace}}}} and \href{this https URL}{\textbf{\underline{\textit{ModelScope}}}}
>
---
