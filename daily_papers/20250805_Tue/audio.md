# 音频 cs.SD;  eess.SP

- **最新发布 35 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] Non-Verbal Vocalisations and their Challenges: Emotion, Privacy, Sparseness, and Real Life
- **分类: cs.SD; 68T10 (Primary) 68T45 (Secondary); I.2.7**

- **简介: 该论文探讨非语言声音（NVVs）的挑战及其解决路径，提出通过语料库和AI模型克服隐私与数据稀疏性问题的工作方法，旨在优化NVVs在现实场景中的建模效果。**

- **链接: [http://arxiv.org/pdf/2508.01960v1](http://arxiv.org/pdf/2508.01960v1)**

> **作者:** Anton Batliner; Shahin Amiriparian; Björn W. Schuller
>
> **摘要:** Non-Verbal Vocalisations (NVVs) are short `non-word' utterances without proper linguistic (semantic) meaning but conveying connotations -- be this emotions/affects or other paralinguistic information. We start this contribution with a historic sketch: how they were addressed in psychology and linguistics in the last two centuries, how they were neglected later on, and how they came to the fore with the advent of emotion research. We then give an overview of types of NVVs (formal aspects) and functions of NVVs, exemplified with the typical NVV \textit{ah}. Interesting as they are, NVVs come, however, with a bunch of challenges that should be accounted for: Privacy and general ethical considerations prevent them of being recorded in real-life (private) scenarios to a sufficient extent. Isolated, prompted (acted) exemplars do not necessarily model NVVs in context; yet, this is the preferred strategy so far when modelling NVVs, especially in AI. To overcome these problems, we argue in favour of corpus-based approaches. This guarantees a more realistic modelling; however, we are still faced with privacy and sparse data problems.
>
---
#### [new 002] Foundation Models for Bioacoustics -- a Comparative Review
- **分类: cs.SD; cs.LG; eess.AS; q-bio.QM**

- **简介: 该论文探讨了基于深度学习的生物声学建模任务，旨在解决适应多变生物声学分类需求的问题。研究综述了大型预训练模型的设计决策与性能评估，并通过BEANS/BirdSet基准测试验证其有效性，发现BirdMAE在BirdSet上表现最优，同时对比了多种模型在不同探测策略下的优势。**

- **链接: [http://arxiv.org/pdf/2508.01277v1](http://arxiv.org/pdf/2508.01277v1)**

> **作者:** Raphael Schwinger; Paria Vali Zadeh; Lukas Rauch; Mats Kurz; Tom Hauschild; Sam Lapp; Sven Tomforde
>
> **备注:** Preprint
>
> **摘要:** Automated bioacoustic analysis is essential for biodiversity monitoring and conservation, requiring advanced deep learning models that can adapt to diverse bioacoustic tasks. This article presents a comprehensive review of large-scale pretrained bioacoustic foundation models and systematically investigates their transferability across multiple bioacoustic classification tasks. We overview bioacoustic representation learning including major pretraining data sources and benchmarks. On this basis, we review bioacoustic foundation models by thoroughly analysing design decisions such as model architecture, pretraining scheme, and training paradigm. Additionally, we evaluate selected foundation models on classification tasks from the BEANS and BirdSet benchmarks, comparing the generalisability of learned representations under both linear and attentive probing strategies. Our comprehensive experimental analysis reveals that BirdMAE, trained on large-scale bird song data with a self-supervised objective, achieves the best performance on the BirdSet benchmark. On BEANS, BEATs$_{NLM}$, the extracted encoder of the NatureLM-audio large audio model, is slightly better. Both transformer-based models require attentive probing to extract the full performance of their representations. ConvNext$_{BS}$ and Perch models trained with supervision on large-scale bird song data remain competitive for passive acoustic monitoring classification tasks of BirdSet in linear probing settings. Training a new linear classifier has clear advantages over evaluating these models without further training. While on BEANS, the baseline model BEATs trained with self-supervision on AudioSet outperforms bird-specific models when evaluated with attentive probing. These findings provide valuable guidance for practitioners selecting appropriate models to adapt them to new bioacoustic classification tasks via probing.
>
---
#### [new 003] Voxlect: A Speech Foundation Model Benchmark for Modeling Dialects and Regional Languages Around the Globe
- **分类: cs.SD; cs.CL**

- **简介: 该论文旨在构建跨语言方言分类模型 Voxlect，解决多语言方言识别与下游应用问题，通过公开语料集评估模型鲁棒性并展示其在语音识别和生成中的应用，开源提供研究工具。**

- **链接: [http://arxiv.org/pdf/2508.01691v1](http://arxiv.org/pdf/2508.01691v1)**

> **作者:** Tiantian Feng; Kevin Huang; Anfeng Xu; Xuan Shi; Thanathai Lertpetchpun; Jihwan Lee; Yoonjeong Lee; Dani Byrd; Shrikanth Narayanan
>
> **摘要:** We present Voxlect, a novel benchmark for modeling dialects and regional languages worldwide using speech foundation models. Specifically, we report comprehensive benchmark evaluations on dialects and regional language varieties in English, Arabic, Mandarin and Cantonese, Tibetan, Indic languages, Thai, Spanish, French, German, Brazilian Portuguese, and Italian. Our study used over 2 million training utterances from 30 publicly available speech corpora that are provided with dialectal information. We evaluate the performance of several widely used speech foundation models in classifying speech dialects. We assess the robustness of the dialectal models under noisy conditions and present an error analysis that highlights modeling results aligned with geographic continuity. In addition to benchmarking dialect classification, we demonstrate several downstream applications enabled by Voxlect. Specifically, we show that Voxlect can be applied to augment existing speech recognition datasets with dialect information, enabling a more detailed analysis of ASR performance across dialectal variations. Voxlect is also used as a tool to evaluate the performance of speech generation systems. Voxlect is publicly available with the license of the RAIL family at: https://github.com/tiantiaf0627/voxlect.
>
---
#### [new 004] Inference-time Scaling for Diffusion-based Audio Super-resolution
- **分类: cs.SD; cs.AI**

- **简介: 该论文探讨了扩散模型在音频超分辨率中的改进方法，旨在通过推理时间缩放探索多解路径，开发任务特定验证器并结合随机/零次搜索算法，解决传统方法因采样随机性导致的质量限制问题，验证了其在语音、音乐等领域的有效性和性能提升。**

- **链接: [http://arxiv.org/pdf/2508.02391v1](http://arxiv.org/pdf/2508.02391v1)**

> **作者:** Yizhu Jin; Zhen Ye; Zeyue Tian; Haohe Liu; Qiuqiang Kong; Yike Guo; Wei Xue
>
> **摘要:** Diffusion models have demonstrated remarkable success in generative tasks, including audio super-resolution (SR). In many applications like movie post-production and album mastering, substantial computational budgets are available for achieving superior audio quality. However, while existing diffusion approaches typically increase sampling steps to improve quality, the performance remains fundamentally limited by the stochastic nature of the sampling process, leading to high-variance and quality-limited outputs. Here, rather than simply increasing the number of sampling steps, we propose a different paradigm through inference-time scaling for SR, which explores multiple solution trajectories during the sampling process. Different task-specific verifiers are developed, and two search algorithms, including the random search and zero-order search for SR, are introduced. By actively guiding the exploration of the high-dimensional solution space through verifier-algorithm combinations, we enable more robust and higher-quality outputs. Through extensive validation across diverse audio domains (speech, music, sound effects) and frequency ranges, we demonstrate consistent performance gains, achieving improvements of up to 9.70% in aesthetics, 5.88% in speaker similarity, 15.20% in word error rate, and 46.98% in spectral distance for speech SR from 4kHz to 24kHz, showcasing the effectiveness of our approach. Audio samples are available at: https://racerk.github.io/tt-scale-audiosr/.
>
---
#### [new 005] WhiSQA: Non-Intrusive Speech Quality Prediction Using Whisper Encoder Features
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该研究旨在开发非侵入式语音质量预测模型，解决SE系统性能评估问题。通过构建大规模音频-质量数据集并利用ASR模型提取特征，提出新型SQ预测方法，显著优于现有方法，实现更高精度与领域适应性。**

- **链接: [http://arxiv.org/pdf/2508.02210v1](http://arxiv.org/pdf/2508.02210v1)**

> **作者:** George Close; Kris Hong; Thomas Hain; Stefan Goetze
>
> **备注:** Accepted at SPECOM 2025
>
> **摘要:** There has been significant research effort developing neural-network-based predictors of SQ in recent years. While a primary objective has been to develop non-intrusive, i.e.~reference-free, metrics to assess the performance of SE systems, recent work has also investigated the direct inference of neural SQ predictors within the loss function of downstream speech tasks. To aid in the training of SQ predictors, several large datasets of audio with corresponding human labels of quality have been created. Recent work in this area has shown that speech representations derived from large unsupervised or semi-supervised foundational speech models are useful input feature representations for neural SQ prediction. In this work, a novel and robust SQ predictor is proposed based on feature representations extracted from an ASR model, found to be a powerful input feature for the SQ prediction task. The proposed system achieves higher correlation with human MOS ratings than recent approaches on all NISQA test sets and shows significantly better domain adaption compared to the commonly used DNSMOS metric.
>
---
#### [new 006] Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation
- **分类: cs.SD; cs.AI**

- **简介: 该论文旨在解决音乐AIGC中长旋律生成的挑战，通过引入Bar-level符号分数和高效处理机制，实现了人类可编辑的高效生成，突破传统直接从音频学习音乐理论的局限性，显著提升性能与实用性。**

- **链接: [http://arxiv.org/pdf/2508.01394v1](http://arxiv.org/pdf/2508.01394v1)**

> **作者:** Tongxi Wang; Yang Yu; Qing Wang; Junlang Qian
>
> **摘要:** Song generation is regarded as the most challenging problem in music AIGC; nonetheless, existing approaches have yet to fully overcome four persistent limitations: controllability, generalizability, perceptual quality, and duration. We argue that these shortcomings stem primarily from the prevailing paradigm of attempting to learn music theory directly from raw audio, a task that remains prohibitively difficult for current models. To address this, we present Bar-level AI Composing Helper (BACH), the first model explicitly designed for song generation through human-editable symbolic scores. BACH introduces a tokenization strategy and a symbolic generative procedure tailored to hierarchical song structure. Consequently, it achieves substantial gains in the efficiency, duration, and perceptual quality of song generation. Experiments demonstrate that BACH, with a small model size, establishes a new SOTA among all publicly reported song generation systems, even surpassing commercial solutions such as Suno. Human evaluations further confirm its superiority across multiple subjective metrics.
>
---
#### [new 007] ShrutiSense: Microtonal Modeling and Correction in Indian Classical Music
- **分类: cs.SD; cs.AI**

- **简介: 该论文旨在解决印度古典音乐中微分音系统与文化特定raga语法的建模与纠正问题，通过互补模型（FST+GC-SHMM）实现修正和补充缺失值，验证了其在5种ragas上的91.3%分类准确率及-50cents噪声下的稳健性，从而保留印度古典音乐的文化表达真实性。**

- **链接: [http://arxiv.org/pdf/2508.01498v1](http://arxiv.org/pdf/2508.01498v1)**

> **作者:** Rajarshi Ghosh; Jayanth Athipatla
>
> **摘要:** Indian classical music relies on a sophisticated microtonal system of 22 shrutis (pitch intervals), which provides expressive nuance beyond the 12-tone equal temperament system. Existing symbolic music processing tools fail to account for these microtonal distinctions and culturally specific raga grammars that govern melodic movement. We present ShrutiSense, a comprehensive symbolic pitch processing system designed for Indian classical music, addressing two critical tasks: (1) correcting westernized or corrupted pitch sequences, and (2) completing melodic sequences with missing values. Our approach employs complementary models for different tasks: a Shruti-aware finite-state transducer (FST) that performs contextual corrections within the 22-shruti framework and a grammar-constrained Shruti hidden Markov model (GC-SHMM) that incorporates raga-specific transition rules for contextual completions. Comprehensive evaluation on simulated data across five ragas demonstrates that ShrutiSense (FST model) achieves 91.3% shruti classification accuracy for correction tasks, with example sequences showing 86.7-90.0% accuracy at corruption levels of 0.2 to 0.4. The system exhibits robust performance under pitch noise up to +/-50 cents, maintaining consistent accuracy across ragas (90.7-91.8%), thus preserving the cultural authenticity of Indian classical music expression.
>
---
#### [new 008] Detecting COPD Through Speech Analysis: A Dataset of Danish Speech and Machine Learning Approach
- **分类: cs.SD; cs.HC; cs.LG**

- **简介: 该论文旨在验证语音作为COPD生物标志物的可行性，通过丹麦语数据集训练机器学习模型（openSMILE+X-Vector嵌入），发现67%准确率，探索其在远程筛查中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2508.02354v1](http://arxiv.org/pdf/2508.02354v1)**

> **作者:** Cuno Sankey-Olsen; Rasmus Hvass Olesen; Tobias Oliver Eberhard; Andreas Triantafyllopoulos; Björn Schuller; Ilhan Aslan
>
> **摘要:** Chronic Obstructive Pulmonary Disease (COPD) is a serious and debilitating disease affecting millions around the world. Its early detection using non-invasive means could enable preventive interventions that improve quality of life and patient outcomes, with speech recently shown to be a valuable biomarker. Yet, its validity across different linguistic groups remains to be seen. To that end, audio data were collected from 96 Danish participants conducting three speech tasks (reading, coughing, sustained vowels). Half of the participants were diagnosed with different levels of COPD and the other half formed a healthy control group. Subsequently, we investigated different baseline models using openSMILE features and learnt x-vector embeddings. We obtained a best accuracy of 67% using openSMILE features and logistic regression. Our findings support the potential of speech-based analysis as a non-invasive, remote, and scalable screening tool as part of future COPD healthcare solutions.
>
---
#### [new 009] Translation-Equivariant Self-Supervised Learning for Pitch Estimation with Optimal Transport
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出一种基于最优运输的目标函数，用于改进一维对称系统的自监督频率估计任务，解决了传统方法在稳定性与简单性方面的不足。**

- **链接: [http://arxiv.org/pdf/2508.01493v1](http://arxiv.org/pdf/2508.01493v1)**

> **作者:** Bernardo Torres; Alain Riou; Gaël Richard; Geoffroy Peeters
>
> **备注:** Extended Abstracts for the Late-Breaking Demo Session of the 26th International Society for Music Information Retrieval Conference
>
> **摘要:** In this paper, we propose an Optimal Transport objective for learning one-dimensional translation-equivariant systems and demonstrate its applicability to single pitch estimation. Our method provides a theoretically grounded, more numerically stable, and simpler alternative for training state-of-the-art self-supervised pitch estimators.
>
---
#### [new 010] Towards Reliable Audio Deepfake Attribution and Model Recognition: A Multi-Level Autoencoder-Based Framework
- **分类: cs.SD; cs.CV**

- **简介: 该论文提出了一种基于多层级自编码器的音频深度伪造归因与模型识别框架（LAVA），解决音频深度伪造检测与溯源问题。通过注意力增强的特征提取和两个分类器实现技术与模型识别，结合信心阈值提升鲁棒性，并验证了在多个公开数据集上的优异性能。**

- **链接: [http://arxiv.org/pdf/2508.02521v1](http://arxiv.org/pdf/2508.02521v1)**

> **作者:** Andrea Di Pierno; Luca Guarnera; Dario Allegra; Sebastiano Battiato
>
> **摘要:** The proliferation of audio deepfakes poses a growing threat to trust in digital communications. While detection methods have advanced, attributing audio deepfakes to their source models remains an underexplored yet crucial challenge. In this paper we introduce LAVA (Layered Architecture for Voice Attribution), a hierarchical framework for audio deepfake detection and model recognition that leverages attention-enhanced latent representations extracted by a convolutional autoencoder trained solely on fake audio. Two specialized classifiers operate on these features: Audio Deepfake Attribution (ADA), which identifies the generation technology, and Audio Deepfake Model Recognition (ADMR), which recognize the specific generative model instance. To improve robustness under open-set conditions, we incorporate confidence-based rejection thresholds. Experiments on ASVspoof2021, FakeOrReal, and CodecFake show strong performance: the ADA classifier achieves F1-scores over 95% across all datasets, and the ADMR module reaches 96.31% macro F1 across six classes. Additional tests on unseen attacks from ASVpoof2019 LA and error propagation analysis confirm LAVA's robustness and reliability. The framework advances the field by introducing a supervised approach to deepfake attribution and model recognition under open-set conditions, validated on public benchmarks and accompanied by publicly released models and code. Models and code are available at https://www.github.com/adipiz99/lava-framework.
>
---
#### [new 011] From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs
- **分类: cs.SD**

- **简介: 该论文旨在解决音频-文本跨模态理解中的差异描述与AC风格之间的语义偏差问题。提出Audio Commonality Captioning (ACC)方法，通过强调音频共通性语义而非细节差异，有效提升了多模态大语言模型在VSC、SER等下游任务中的泛化能力，验证了其在增强跨模态理解中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.01659v1](http://arxiv.org/pdf/2508.01659v1)**

> **作者:** Yuhang Jia; Xu Zhang; Yong Qin
>
> **摘要:** Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of multimodal large language models (MLLMs). To further strengthen this alignment, recent works have proposed Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained audio discrimination. However, despite its effectiveness in enabling difference-telling and detailed discrimination, ADC introduces a notable semantic gap between the input audios-often rich in diverse sound events-and the relatively brief, difference-focused output captions. This deviation from AC-style descriptions leads to a mismatch with the pretraining objective, resulting in catastrophic forgetting during finetuning. To mitigate this issue, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that encourages the model to capture the shared semantics across audio clips rather than emphasizing their detailed differences. Experimental results demonstrate that ACC not only effectively enhances audio-text understanding on primary captioning benchmarks but also better preserves general capabilities across diverse speech and music-related downstream tasks, such as vocal sound classification (VSC), speech emotion recognition (SER), musical instrument classification (MIC), and music genre classification (MGC), compared to ADC. These findings validate that ACC contributes to more robust cross-modal understanding and achieves a better balance between generalization and task-specific performance in the context of MLLMs.
>
---
#### [new 012] Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers
- **分类: cs.SD; cs.CL**

- **简介: 该论文旨在揭示音频大语言模型（ALLMs）的安全性问题，提出HIN框架并开发AudioSafe基准测试，评估其对环境噪声等音频特征的攻击成功率及敏感度差异，发现现有模型存在显著漏洞。**

- **链接: [http://arxiv.org/pdf/2508.02175v1](http://arxiv.org/pdf/2508.02175v1)**

> **作者:** Liang Lin; Miao Yu; Kaiwen Luo; Yibo Zhang; Lilan Peng; Dexian Wang; Xuehai Tang; Yuanhe Zhang; Xikang Yang; Zhenhong Zhou; Kun Wang; Yang Liu
>
> **摘要:** As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.
>
---
#### [new 013] Unsupervised Multi-channel Speech Dereverberation via Diffusion
- **分类: cs.SD; eess.AS**

- **简介: 该论文旨在解决多通道单源盲去混响问题，通过无监督扩散模型与后验采样联合训练，估计麦克风通道的房间响应（RIR），并结合前向卷积预测实现高效建模，提升去混响性能。**

- **链接: [http://arxiv.org/pdf/2508.02071v1](http://arxiv.org/pdf/2508.02071v1)**

> **作者:** Yulun Wu; Zhongweiyang Xu; Jianchong Chen; Zhong-Qiu Wang; Romit Roy Choudhury
>
> **摘要:** We consider the problem of multi-channel single-speaker blind dereverberation, where multi-channel mixtures are used to recover the clean anechoic speech. To solve this problem, we propose USD-DPS, {U}nsupervised {S}peech {D}ereverberation via {D}iffusion {P}osterior {S}ampling. USD-DPS uses an unconditional clean speech diffusion model as a strong prior to solve the problem by posterior sampling. At each diffusion sampling step, we estimate all microphone channels' room impulse responses (RIRs), which are further used to enforce a multi-channel mixture consistency constraint for diffusion guidance. For multi-channel RIR estimation, we estimate reference-channel RIR by optimizing RIR parameters of a sub-band RIR signal model, with the Adam optimizer. We estimate non-reference channels' RIRs analytically using forward convolutive prediction (FCP). We found that this combination provides a good balance between sampling efficiency and RIR prior modeling, which shows superior performance among unsupervised dereverberation approaches. An audio demo page is provided in https://usddps.github.io/USDDPS_demo/.
>
---
#### [new 014] Charting 15 years of progress in deep learning for speech emotion recognition: A replication study
- **分类: cs.SD**

- **简介: 该论文旨在量化15年语音情感识别（SER）技术的发展进展，评估深层神经网络与早期方法的差异，并探讨未来研究方向。通过对比音频/文本模型及其引入Transformer后的表现，揭示进步瓶颈及感知偏差机制。**

- **链接: [http://arxiv.org/pdf/2508.02448v1](http://arxiv.org/pdf/2508.02448v1)**

> **作者:** Andreas Triantafyllopoulos; Anton Batliner; Björn W. Schuller
>
> **备注:** Code: https://github.com/CHI-TUM/ser-progress-replication Submitted for review
>
> **摘要:** Speech emotion recognition (SER) has long benefited from the adoption of deep learning methodologies. Deeper models -- with more layers and more trainable parameters -- are generally perceived as being `better' by the SER community. This raises the question -- \emph{how much better} are modern-era deep neural networks compared to their earlier iterations? Beyond that, the more important question of how to move forward remains as poignant as ever. SER is far from a solved problem; therefore, identifying the most prominent avenues of future research is of paramount importance. In the present contribution, we attempt a quantification of progress in the 15 years of research beginning with the introduction of the landmark 2009 INTERSPEECH Emotion Challenge. We conduct a large scale investigation of model architectures, spanning both audio-based models that rely on speech inputs and text-baed models that rely solely on transcriptions. Our results point towards diminishing returns and a plateau after the recent introduction of transformer architectures. Moreover, we demonstrate how perceptions of progress are conditioned on the particular selection of models that are compared. Our findings have important repercussions about the state-of-the-art in SER research and the paths forward
>
---
#### [new 015] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
- **分类: cs.SD**

- **简介: 该论文提出了一种多模态检索与选择方法（MARS），通过增强基于LLM的ASR在处理多模态历史上下文中实现更优识别效果。解决了传统单一历史上下文对语义理解不足的问题，提升了模型对自然语言连续对话场景的感知能力。**

- **链接: [http://arxiv.org/pdf/2508.01166v1](http://arxiv.org/pdf/2508.01166v1)**

> **作者:** Bingshen Mu; Hexin Liu; Hongfei Xue; Kun Wei; Lei Xie
>
> **摘要:** Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
>
---
#### [new 016] Toward a reliable PWM-based light-emitting diode visual stimulus for improved SSVEP response with minimal visual fatigue
- **分类: eess.SP; cs.CV; cs.SE**

- **简介: 该论文旨在通过极端PWM duty-cycles提升SSVEP响应并减少视觉疲劳，解决传统方法因高频率刺激导致的疲劳与低精度问题。研究通过测试验证了85%的peak响应对所有频率均有效，为SSVEP的实用化提供了新方案。**

- **链接: [http://arxiv.org/pdf/2508.02359v1](http://arxiv.org/pdf/2508.02359v1)**

> **作者:** Surej Mouli; Ramaswamy Palaniappan
>
> **摘要:** Steady state visual evoked response (SSVEP) is widely used in visual-based diagnosis and applications such as brain computer interfacing due to its high information transfer rate and the capability to activate commands through simple gaze control. However, one major impediment in using flashing visual stimulus to obtain SSVEP is eye fatigue that prevents continued long term use preventing practical deployment. This combined with the difficulty in establishing precise pulse-width modulation (PWM) that results in poorer accuracy warrants the development of appropriate approach to solve these issues. Various studies have suggested the usage of high frequencies of visual stimulus to reduce the visual fatigue for the user but this results in poor response performance. Here, the authors study the use of extremely high duty-cycles in the stimulus in the hope of solving these constraints. Electroencephalogram data was recorded with PWM duty-cycles of 50 to 95% generated by a precise custom-made light-emitting diode hardware and tested ten subjects responded that increasing duty-cycles had less visual strain for all the frequency values and the SSVEP exhibited a subject-independent peak response for duty-cycle of 85%. This could pave the way for increased usage of SSVEP for practical applications.
>
---
#### [new 017] Generalizable Audio Deepfake Detection via Hierarchical Structure Learning and Feature Whitening in Poincaré sphere
- **分类: cs.SD**

- **简介: 该论文旨在解决音频深度伪造检测中的通用性不足问题，通过构建基于Poincaré球面的分层结构学习与特征白化框架，提升域不变性并优化攻击识别效果。**

- **链接: [http://arxiv.org/pdf/2508.01897v1](http://arxiv.org/pdf/2508.01897v1)**

> **作者:** Mingru Yang; Yanmei Gu; Qianhua He; Yanxiong Li; Peirong Zhang; Yongqiang Chen; Zhiming Wang; Huijia Zhu; Jian Liu; Weiqiang Wang
>
> **备注:** Accepted for publication on Interspeech 2025
>
> **摘要:** Audio deepfake detection (ADD) faces critical generalization challenges due to diverse real-world spoofing attacks and domain variations. However, existing methods primarily rely on Euclidean distances, failing to adequately capture the intrinsic hierarchical structures associated with attack categories and domain factors. To address these issues, we design a novel framework Poin-HierNet to construct domain-invariant hierarchical representations in the Poincar\'e sphere. Poin-HierNet includes three key components: 1) Poincar\'e Prototype Learning (PPL) with several data prototypes aligning sample features and capturing multilevel hierarchies beyond human labels; 2) Hierarchical Structure Learning (HSL) leverages top prototypes to establish a tree-like hierarchical structure from data prototypes; and 3) Poincar\'e Feature Whitening (PFW) enhances domain invariance by applying feature whitening to suppress domain-sensitive features. We evaluate our approach on four datasets: ASVspoof 2019 LA, ASVspoof 2021 LA, ASVspoof 2021 DF, and In-The-Wild. Experimental results demonstrate that Poin-HierNet exceeds state-of-the-art methods in Equal Error Rate.
>
---
#### [new 018] Advancing the Foundation Model for Music Understanding
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出了一种统一基础模型MuFun，用于多任务音乐理解，解决传统单一任务模型难以应对复杂任务的问题，通过联合处理乐器与歌词并利用大规模数据训练，开发了MuCUE基准评估，展示了其在多任务性能上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.01178v1](http://arxiv.org/pdf/2508.01178v1)**

> **作者:** Yi Jiang; Wei Wang; Xianwen Guo; Huiyun Liu; Hanrui Wang; Youri Xu; Haoqi Gu; Zhongqian Xie; Chuanjiang Luo
>
> **摘要:** The field of Music Information Retrieval (MIR) is fragmented, with specialized models excelling at isolated tasks. In this work, we challenge this paradigm by introducing a unified foundation model named MuFun for holistic music understanding. Our model features a novel architecture that jointly processes instrumental and lyrical content, and is trained on a large-scale dataset covering diverse tasks such as genre classification, music tagging, and question answering. To facilitate robust evaluation, we also propose a new benchmark for multi-faceted music understanding called MuCUE (Music Comprehensive Understanding Evaluation). Experiments show our model significantly outperforms existing audio large language models across the MuCUE tasks, demonstrating its state-of-the-art effectiveness and generalization ability.
>
---
#### [new 019] StutterCut: Uncertainty-Guided Normalised Cut for Dysfluency Segmentation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文旨在解决语音失语检测与分割问题，提出StutterCut半监督框架，通过图模型将语音窗口嵌入表示为节点，并利用不确定性控制连接权重，同时扩展FluencyBank数据集以提升真实性和准确性。**

- **链接: [http://arxiv.org/pdf/2508.02255v1](http://arxiv.org/pdf/2508.02255v1)**

> **作者:** Suhita Ghosh; Melanie Jouaiti; Jan-Ole Perschewski; Sebastian Stober
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** Detecting and segmenting dysfluencies is crucial for effective speech therapy and real-time feedback. However, most methods only classify dysfluencies at the utterance level. We introduce StutterCut, a semi-supervised framework that formulates dysfluency segmentation as a graph partitioning problem, where speech embeddings from overlapping windows are represented as graph nodes. We refine the connections between nodes using a pseudo-oracle classifier trained on weak (utterance-level) labels, with its influence controlled by an uncertainty measure from Monte Carlo dropout. Additionally, we extend the weakly labelled FluencyBank dataset by incorporating frame-level dysfluency boundaries for four dysfluency types. This provides a more realistic benchmark compared to synthetic datasets. Experiments on real and synthetic datasets show that StutterCut outperforms existing methods, achieving higher F1 scores and more precise stuttering onset detection.
>
---
#### [new 020] Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder
- **分类: cs.SD; eess.AS**

- **简介: 本论文旨在通过扩展带宽提升合成声波频谱真实性，结合神经网络与Vocos技术解决高频率频谱差异问题，并优化vocoder以增强音频真实感。**

- **链接: [http://arxiv.org/pdf/2508.01796v1](http://arxiv.org/pdf/2508.01796v1)**

> **作者:** Runxuan Yang; Kai Li; Guo Chen; Xiaolin Hu
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** This paper addresses the challenge of enhancing the realism of vocoder-generated singing voice audio by mitigating the distinguishable disparities between synthetic and real-life recordings, particularly in high-frequency spectrogram components. Our proposed approach combines two innovations: an explicit linear spectrogram estimation step using denoising diffusion process with DiT-based neural network architecture optimized for time-frequency data, and a redesigned vocoder based on Vocos specialized in handling large linear spectrograms with increased frequency bins. This integrated method can produce audio with high-fidelity spectrograms that are challenging for both human listeners and machine classifiers to differentiate from authentic recordings. Objective and subjective evaluations demonstrate that our streamlined approach maintains high audio quality while achieving this realism. This work presents a substantial advancement in overcoming the limitations of current vocoding techniques, particularly in the context of adversarial attacks on fake spectrogram detection.
>
---
#### [new 021] Localizing Audio-Visual Deepfakes via Hierarchical Boundary Modeling
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文旨在解决音频-视觉深度伪造的时序定位问题，通过构建层次化边界模型（HBMNet）实现多模态特征融合与多尺度线索整合，分别通过特征编码、粗粒度生成和细粒度概率优化，提升定位精度和召回能力，验证了模块间的互补性及模型的泛化潜力。**

- **链接: [http://arxiv.org/pdf/2508.02000v1](http://arxiv.org/pdf/2508.02000v1)**

> **作者:** Xuanjun Chen; Shih-Peng Cheng; Jiawei Du; Lin Zhang; Xiaoxiao Miao; Chung-Che Wang; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Work in progress
>
> **摘要:** Audio-visual temporal deepfake localization under the content-driven partial manipulation remains a highly challenging task. In this scenario, the deepfake regions are usually only spanning a few frames, with the majority of the rest remaining identical to the original. To tackle this, we propose a Hierarchical Boundary Modeling Network (HBMNet), which includes three modules: an Audio-Visual Feature Encoder that extracts discriminative frame-level representations, a Coarse Proposal Generator that predicts candidate boundary regions, and a Fine-grained Probabilities Generator that refines these proposals using bidirectional boundary-content probabilities. From the modality perspective, we enhance audio-visual learning through dedicated encoding and fusion, reinforced by frame-level supervision to boost discriminability. From the temporal perspective, HBMNet integrates multi-scale cues and bidirectional boundary-content relationships. Experiments show that encoding and fusion primarily improve precision, while frame-level supervision boosts recall. Each module (audio-visual fusion, temporal scales, bi-directionality) contributes complementary benefits, collectively enhancing localization performance. HBMNet outperforms BA-TFD and UMMAFormer and shows improved potential scalability with more training data.
>
---
#### [new 022] GeHirNet: A Gender-Aware Hierarchical Model for Voice Pathology Classification
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出GeHirNet（性别感知层次模型）用于语音病理学分类，解决性别偏见与数据稀疏性问题，通过两阶段架构（性别特征识别+条件分类）提升性能（97.63% ACC/95.25% MCC），有效缓解性别偏差并优化模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.01172v1](http://arxiv.org/pdf/2508.01172v1)**

> **作者:** Fan Wu; Kaicheng Zhao; Elgar Fleisch; Filipe Barata
>
> **摘要:** AI-based voice analysis shows promise for disease diagnostics, but existing classifiers often fail to accurately identify specific pathologies because of gender-related acoustic variations and the scarcity of data for rare diseases. We propose a novel two-stage framework that first identifies gender-specific pathological patterns using ResNet-50 on Mel spectrograms, then performs gender-conditioned disease classification. We address class imbalance through multi-scale resampling and time warping augmentation. Evaluated on a merged dataset from four public repositories, our two-stage architecture with time warping achieves state-of-the-art performance (97.63\% accuracy, 95.25\% MCC), with a 5\% MCC improvement over single-stage baseline. This work advances voice pathology classification while reducing gender bias through hierarchical modeling of vocal characteristics.
>
---
#### [new 023] Automatic Melody Reduction via Shortest Path Finding
- **分类: cs.SD**

- **简介: 该论文研究了自动旋律还原任务，通过图论中的最短路径算法解决旋律压缩问题，提出了一种基于计算音乐理论的新型方法，有效提升了旋律还原质量并应用于音乐生成任务。**

- **链接: [http://arxiv.org/pdf/2508.01571v1](http://arxiv.org/pdf/2508.01571v1)**

> **作者:** Ziyu Wang; Yuxuan Wu; Roger B. Dannenberg; Gus Xia
>
> **备注:** Accepted paper at ISMIR 2025. https://ismir2025.ismir.net/accepted-papers
>
> **摘要:** Melody reduction, as an abstract representation of musical compositions, serves not only as a tool for music analysis but also as an intermediate representation for structured music generation. Prior computational theories, such as the Generative Theory of Tonal Music, provide insightful interpretations of music, but they are not fully automatic and usually limited to the classical genre. In this paper, we propose a novel and conceptually simple computational method for melody reduction using a graph-based representation inspired by principles from computational music theories, where the reduction process is formulated as finding the shortest path. We evaluate our algorithm on pop, folk, and classical genres, and experimental results show that the algorithm produces melody reductions that are more faithful to the original melody and more musically coherent than other common melody downsampling methods. As a downstream task, we use melody reductions to generate symbolic music variations. Experiments show that our method achieves higher quality than state-of-the-art style transfer methods.
>
---
#### [new 024] PESTO: Real-Time Pitch Estimation with Self-supervised Transposition-equivariant Objective
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出一种自监督实时pitch估计模型PESTO，解决如何高效且准确地从视频帧中提取单个音调问题。通过设计equivariant结构和类目标函数，实现了无需标注数据的轻量化训练，在MIR-1K等音乐和语音数据集上表现优异，适用于实时应用。**

- **链接: [http://arxiv.org/pdf/2508.01488v1](http://arxiv.org/pdf/2508.01488v1)**

> **作者:** Alain Riou; Bernardo Torres; Ben Hayes; Stefan Lattner; Gaëtan Hadjeres; Gaël Richard; Geoffroy Peeters
>
> **备注:** Accepted to the Transactions of the International Society for Music Information Retrieval
>
> **摘要:** In this paper, we introduce PESTO, a self-supervised learning approach for single-pitch estimation using a Siamese architecture. Our model processes individual frames of a Variable-$Q$ Transform (VQT) and predicts pitch distributions. The neural network is designed to be equivariant to translations, notably thanks to a Toeplitz fully-connected layer. In addition, we construct pitch-shifted pairs by translating and cropping the VQT frames and train our model with a novel class-based transposition-equivariant objective, eliminating the need for annotated data. Thanks to this architecture and training objective, our model achieves remarkable performances while being very lightweight ($130$k parameters). Evaluations on music and speech datasets (MIR-1K, MDB-stem-synth, and PTDB) demonstrate that PESTO not only outperforms self-supervised baselines but also competes with supervised methods, exhibiting superior cross-dataset generalization. Finally, we enhance PESTO's practical utility by developing a streamable VQT implementation using cached convolutions. Combined with our model's low latency (less than 10 ms) and minimal parameter count, this makes PESTO particularly suitable for real-time applications.
>
---
#### [new 025] Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning
- **分类: cs.AI; cs.MM; cs.SD; 68; I.2.10**

- **简介: 该论文旨在解决多模态情感推理中的情感冲突问题，通过构建CA-MER基准并提出MoSEAR框架（减少模态偏倚、提升一致性），实现了在不同模态间平衡整合的效果，达到多模态推理的优化目标。**

- **链接: [http://arxiv.org/pdf/2508.01181v1](http://arxiv.org/pdf/2508.01181v1)**

> **作者:** Zhiyuan Han; Beier Zhu; Yanlong Xu; Peipei Song; Xun Yang
>
> **备注:** ACM Multimedia 2025
>
> **摘要:** Despite their strong performance in multimodal emotion reasoning, existing Multimodal Large Language Models (MLLMs) often overlook the scenarios involving emotion conflicts, where emotional cues from different modalities are inconsistent. To fill this gap, we first introduce CA-MER, a new benchmark designed to examine MLLMs under realistic emotion conflicts. It consists of three subsets: video-aligned, audio-aligned, and consistent, where only one or all modalities reflect the true emotion. However, evaluations on our CA-MER reveal that current state-of-the-art emotion MLLMs systematically over-rely on audio signal during emotion conflicts, neglecting critical cues from visual modality. To mitigate this bias, we propose MoSEAR, a parameter-efficient framework that promotes balanced modality integration. MoSEAR consists of two modules: (1)MoSE, modality-specific experts with a regularized gating mechanism that reduces modality bias in the fine-tuning heads; and (2)AR, an attention reallocation mechanism that rebalances modality contributions in frozen backbones during inference. Our framework offers two key advantages: it mitigates emotion conflicts and improves performance on consistent samples-without incurring a trade-off between audio and visual modalities. Experiments on multiple benchmarks-including MER2023, EMER, DFEW, and our CA-MER-demonstrate that MoSEAR achieves state-of-the-art performance, particularly under modality conflict conditions.
>
---
#### [new 026] Perception of dynamic multi-speaker auditory scenes under different modes of attention
- **分类: q-bio.NC; eess.AS; eess.SP**

- **简介: 该论文研究了动态多语言听觉场景中不同注意力模式（特征/对象/全局）的感知与神经机制，探讨了其交互关系及任务差异，通过cocktail party实验发现对象注意力更有效，揭示了底物相关性对注意力分配的影响，发现EEG数据中的源-采样差异，验证了任务导向下的认知加工机制。**

- **链接: [http://arxiv.org/pdf/2508.02620v1](http://arxiv.org/pdf/2508.02620v1)**

> **作者:** Stephanie Graceffo; David F Little; Emine Merve Kaya; Mounya Elhilali
>
> **摘要:** Attention is not monolithic; rather, it operates in multiple forms to facilitate efficient cognitive processing. In the auditory domain, attention enables the prioritization of relevant sounds in an auditory scene and can be either attracted by elements in the scene in a bottom-up fashion or directed towards features, objects, or the entire scene in a top-down fashion. How these modes of attention interact and whether their neural underpinnings are distinct remains unclear. In this work, we investigate the perceptual and neural correlates of different attentional modes in a controlled "cocktail party" paradigm, where listeners listen to the same stimuli and attend to either a spatial location (feature-based), a speaker (object-based), or the entire scene (global or free-listening) while detecting deviations in pitch of a voice in the scene. Our findings indicate that object-based attention is more perceptually effective than feature-based or global attention. Furthermore, object-based and spatial-based attention engage distinct neural mechanisms and are differentially modulated by bottom-up salience. Notably, while bottom-up salience aids in the initial segregation of auditory objects, it plays a reduced role in object tracking once attention has been voluntarily allocated. In addition, decoding the stimulus envelope from the EEG data revealed a source-sampling scheme in the global attention mode that is not present in the object or spatial modes. Overall, the study shows that the perception of the same acoustic scene differs according to the listening task, guided by an interaction between top-down and bottom-up processes.
>
---
#### [new 027] EgoTrigger: Toward Audio-Driven Image Capture for Human Memory Enhancement in All-Day Energy-Efficient Smart Glasses
- **分类: cs.CV; cs.ET; cs.HC; cs.LG; cs.SD**

- **简介: 该论文旨在开发基于音频驱动的图像捕捉技术，解决智能眼镜在连续感知与能量效率之间的平衡问题。EgoTrigger通过音频信号（如手部动作）激活相机，降低能耗并提升记忆增强功能，采用YAMNet轻量化模型和自定义分类头进行数据处理，验证其在QA-Ego4D和HME-QA数据集上的性能表现。**

- **链接: [http://arxiv.org/pdf/2508.01915v1](http://arxiv.org/pdf/2508.01915v1)**

> **作者:** Akshay Paruchuri; Sinan Hersek; Lavisha Aggarwal; Qiao Yang; Xin Liu; Achin Kulshrestha; Andrea Colaco; Henry Fuchs; Ishan Chatterjee
>
> **备注:** 15 pages, 6 figres, 6 tables. Accepted to ISMAR 2025 as a TVCG journal paper
>
> **摘要:** All-day smart glasses are likely to emerge as platforms capable of continuous contextual sensing, uniquely positioning them for unprecedented assistance in our daily lives. Integrating the multi-modal AI agents required for human memory enhancement while performing continuous sensing, however, presents a major energy efficiency challenge for all-day usage. Achieving this balance requires intelligent, context-aware sensor management. Our approach, EgoTrigger, leverages audio cues from the microphone to selectively activate power-intensive cameras, enabling efficient sensing while preserving substantial utility for human memory enhancement. EgoTrigger uses a lightweight audio model (YAMNet) and a custom classification head to trigger image capture from hand-object interaction (HOI) audio cues, such as the sound of a drawer opening or a medication bottle being opened. In addition to evaluating on the QA-Ego4D dataset, we introduce and evaluate on the Human Memory Enhancement Question-Answer (HME-QA) dataset. Our dataset contains 340 human-annotated first-person QA pairs from full-length Ego4D videos that were curated to ensure that they contained audio, focusing on HOI moments critical for contextual understanding and memory. Our results show EgoTrigger can use 54% fewer frames on average, significantly saving energy in both power-hungry sensing components (e.g., cameras) and downstream operations (e.g., wireless transmission), while achieving comparable performance on datasets for an episodic memory task. We believe this context-aware triggering strategy represents a promising direction for enabling energy-efficient, functional smart glasses capable of all-day use -- supporting applications like helping users recall where they placed their keys or information about their routine activities (e.g., taking medications).
>
---
#### [new 028] Sonify Anything: Towards Context-Aware Sonic Interactions in AR
- **分类: cs.HC; cs.CV; cs.SD; H.5.5; H.5.2; H.5.1; I.3.5**

- **简介: 该论文旨在解决AR中缺乏物理性虚拟物体导致的声学不自然问题，提出通过计算机视觉识别材料属性并结合物理建模合成的实时声音框架，以增强现实环境的真实感与交互体验。**

- **链接: [http://arxiv.org/pdf/2508.01789v1](http://arxiv.org/pdf/2508.01789v1)**

> **作者:** Laura Schütz; Sasan Matinfar; Ulrich Eck; Daniel Roth; Nassir Navab
>
> **摘要:** In Augmented Reality (AR), virtual objects interact with real objects. However, the lack of physicality of virtual objects leads to the absence of natural sonic interactions. When virtual and real objects collide, either no sound or a generic sound is played. Both lead to an incongruent multisensory experience, reducing interaction and object realism. Unlike in Virtual Reality (VR) and games, where predefined scenes and interactions allow for the playback of pre-recorded sound samples, AR requires real-time sound synthesis that dynamically adapts to novel contexts and objects to provide audiovisual congruence during interaction. To enhance real-virtual object interactions in AR, we propose a framework for context-aware sounds using methods from computer vision to recognize and segment the materials of real objects. The material's physical properties and the impact dynamics of the interaction are used to generate material-based sounds in real-time using physical modelling synthesis. In a user study with 24 participants, we compared our congruent material-based sounds to a generic sound effect, mirroring the current standard of non-context-aware sounds in AR applications. The results showed that material-based sounds led to significantly more realistic sonic interactions. Material-based sounds also enabled participants to distinguish visually similar materials with significantly greater accuracy and confidence. These findings show that context-aware, material-based sonic interactions in AR foster a stronger sense of realism and enhance our perception of real-world surroundings.
>
---
#### [new 029] DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 该论文旨在解决多模态情感识别中的模态异质性与一致性问题，提出DRKF方法通过分层结构模块（ORL-KF-ED）融合知识与优化表示，实现高精度情感分类。**

- **链接: [http://arxiv.org/pdf/2508.01644v1](http://arxiv.org/pdf/2508.01644v1)**

> **作者:** Peiyuan Jiang; Yao Liu; Qiao Liu; Zongshun Zhang; Jiaye Yang; Lu Liu; Daibing Yao
>
> **备注:** Published in ACM Multimedia 2025. 10 pages, 4 figures
>
> **摘要:** Multimodal emotion recognition (MER) aims to identify emotional states by integrating and analyzing information from multiple modalities. However, inherent modality heterogeneity and inconsistencies in emotional cues remain key challenges that hinder performance. To address these issues, we propose a Decoupled Representations with Knowledge Fusion (DRKF) method for MER. DRKF consists of two main modules: an Optimized Representation Learning (ORL) Module and a Knowledge Fusion (KF) Module. ORL employs a contrastive mutual information estimation method with progressive modality augmentation to decouple task-relevant shared representations and modality-specific features while mitigating modality heterogeneity. KF includes a lightweight self-attention-based Fusion Encoder (FE) that identifies the dominant modality and integrates emotional information from other modalities to enhance the fused representation. To handle potential errors from incorrect dominant modality selection under emotionally inconsistent conditions, we introduce an Emotion Discrimination Submodule (ED), which enforces the fused representation to retain discriminative cues of emotional inconsistency. This ensures that even if the FE selects an inappropriate dominant modality, the Emotion Classification Submodule (EC) can still make accurate predictions by leveraging preserved inconsistency information. Experiments show that DRKF achieves state-of-the-art (SOTA) performance on IEMOCAP, MELD, and M3ED. The source code is publicly available at https://github.com/PANPANKK/DRKF.
>
---
#### [new 030] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出了一种集成语音克隆与情绪控制的多模态语音合成系统，旨在解决传统方法难以实现高表达性、自然性及跨语境情感控制的问题。通过引入分批对比学习和情感嵌入机制，构建了CSEMOTIONS数据集，并验证了其在语音清晰度和情感丰富度方面的显著提升。**

- **链接: [http://arxiv.org/pdf/2508.02038v1](http://arxiv.org/pdf/2508.02038v1)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis.
>
---
#### [new 031] RestAware: Non-Invasive Sleep Monitoring Using FMCW Radar and AI-Generated Summaries
- **分类: cs.HC; cs.CY; eess.SP**

- **简介: 该论文提出了一种基于24GHz FMCW radar 的非侵入式睡眠监测系统，旨在解决传统设备易受遮挡、隐私风险等问题。通过KNN算法实现8种睡眠姿势的92%分类准确率，并结合指令调优的AI模型生成个性化睡眠总结，提供低成本、隐私安全的实时部署方案。**

- **链接: [http://arxiv.org/pdf/2508.00848v1](http://arxiv.org/pdf/2508.00848v1)**

> **作者:** Agniva Banerjee; Bhanu Partap Paregi; Haroon R. Lone
>
> **摘要:** Monitoring sleep posture and behavior is critical for diagnosing sleep disorders and improving overall sleep quality. However, traditional approaches, such as wearable devices, cameras, and pressure sensors, often compromise user comfort, fail under obstructions like blankets, and raise privacy concerns. To overcome these limitations, we present RestAware, a non-invasive, contactless sleep monitoring system based on a 24GHz frequency-modulated continuous wave (FMCW) radar. Our system is evaluated on 25 participants across eight common sleep postures, achieving 92% classification accuracy and an F1-score of 0.91 using a K-Nearest Neighbors (KNN) classifier. In addition, we integrate instruction-tuned large language models (Mistral, Llama, and Falcon) to generate personalized, human-readable sleep summaries from radar-derived posture data. This low-cost ($ 35), privacy-preserving solution offers a practical alternative for real-time deployment in smart homes and clinical environments.
>
---
#### [new 032] Test-Time Training for Speech Enhancement
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文介绍了一种基于测试时间训练（TTT）的语音增强方法，解决了未知噪声和领域转换带来的挑战。通过Y型结构结合主任务与自监督辅助任务，动态优化噪声增强和掩码频谱预测等策略，有效提升了语音质量并验证了其在不同数据集上的性能优势。**

- **链接: [http://arxiv.org/pdf/2508.01847v1](http://arxiv.org/pdf/2508.01847v1)**

> **作者:** Avishkar Behera; Riya Ann Easow; Venkatesh Parvathala; K. Sri Rama Murty
>
> **备注:** Accepted to Interspeech 2025. 5 pages, 2 figures
>
> **摘要:** This paper introduces a novel application of Test-Time Training (TTT) for Speech Enhancement, addressing the challenges posed by unpredictable noise conditions and domain shifts. This method combines a main speech enhancement task with a self-supervised auxiliary task in a Y-shaped architecture. The model dynamically adapts to new domains during inference time by optimizing the proposed self-supervised tasks like noise-augmented signal reconstruction or masked spectrogram prediction, bypassing the need for labeled data. We further introduce various TTT strategies offering a trade-off between adaptation and efficiency. Evaluations across synthetic and real-world datasets show consistent improvements across speech quality metrics, outperforming the baseline model. This work highlights the effectiveness of TTT in speech enhancement, providing insights for future research in adaptive and robust speech processing.
>
---
#### [new 033] Accessibility and Social Inclusivity: A Literature Review of Music Technology for Blind and Low Vision People
- **分类: cs.HC; cs.CY; cs.SD; eess.AS**

- **简介: 该论文为盲人/低视力人群音乐技术设计提供系统性文献综述，聚焦其空间感知、信息获取、非语言沟通与记忆等需求，提出4项设计原则并建议增加实证研究和协作测试，以推动从"技术可访问"到"社会包容"的转变。**

- **链接: [http://arxiv.org/pdf/2508.00929v1](http://arxiv.org/pdf/2508.00929v1)**

> **作者:** Shumeng Zhang; Raul Masu; Mela Bettega; Mingming Fan
>
> **备注:** Accepted by ASSETS'25 - The 27th International ACM SIGACCESS Conference on Computers and Accessibility
>
> **摘要:** This paper presents a systematic literature review of music technology tailored for blind and low vision (BLV) individuals. Music activities can be particularly beneficial for BLV people. However, a systematic approach to organizing knowledge on designing accessible technology for BLV people has yet to be attempted. We categorize the existing studies based on the type of technology and the extent of BLV people's involvement in the research. We identify six main categories of BLV people-oriented music technology and highlight four key trends in design goals. Based on these categories, we propose four general insights focusing on (1) spatial awareness, (2) access to information, (3) (non-verbal) communication, and (4) memory. The identified trends suggest that more empirical studies involving BLV people in real-world scenarios are needed to ensure that technological advancements can enhance musical experiences and social inclusion. This research proposes collaborative music technology and inclusive real-world testing with the target group as two key areas missing in current research. They serve as a foundational step in shifting the focus from ``accessible technology'' to ``inclusive technology'' for BLV individuals within the broader field of accessibility research.
>
---
#### [new 034] CAK: Emergent Audio Effects from Minimal Deep Learning
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 本论文提出CAK，利用3x3卷积核生成音频效果，通过条件感知卷积和对抗训练发现音频变换，解决了传统方法无法从少量数据中提取效果的问题。**

- **链接: [http://arxiv.org/pdf/2508.02643v1](http://arxiv.org/pdf/2508.02643v1)**

> **作者:** Austin Rockman
>
> **备注:** 8 pages, 3 figures, code and other resources at https://github.com/gloame-ai/cak-audio/tree/main/cak-audio
>
> **摘要:** We demonstrate that a single 3x3 convolutional kernel can produce emergent audio effects when trained on 200 samples from a personalized corpus. We achieve this through two key techniques: (1) Conditioning Aware Kernels (CAK), where output = input + (learned_pattern x control), with a soft-gate mechanism supporting identity preservation at zero control; and (2) AuGAN (Audit GAN), which reframes adversarial training from "is this real?" to "did you apply the requested value?" Rather than learning to generate or detect forgeries, our networks cooperate to verify control application, discovering unique transformations. The learned kernel exhibits a diagonal structure creating frequency-dependent temporal shifts that are capable of producing musical effects based on input characteristics. Our results show the potential of adversarial training to discover audio transformations from minimal data, enabling new approaches to effect design.
>
---
#### [new 035] Reference-free Adversarial Sex Obfuscation in Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在解决语音中性别特征隐藏的问题，通过参考无的对抗学习框架与正则化技术，在半监督攻击下显著提升了RASO对性特征的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.02295v1](http://arxiv.org/pdf/2508.02295v1)**

> **作者:** Yangyang Qu; Michele Panariello; Massimiliano Todisco; Nicholas Evans
>
> **摘要:** Sex conversion in speech involves privacy risks from data collection and often leaves residual sex-specific cues in outputs, even when target speaker references are unavailable. We introduce RASO for Reference-free Adversarial Sex Obfuscation. Innovations include a sex-conditional adversarial learning framework to disentangle linguistic content from sex-related acoustic markers and explicit regularisation to align fundamental frequency distributions and formant trajectories with sex-neutral characteristics learned from sex-balanced training data. RASO preserves linguistic content and, even when assessed under a semi-informed attack model, it significantly outperforms a competing approach to sex obfuscation.
>
---
## 更新

#### [replaced 001] VAEmo: Efficient Representation Learning for Visual-Audio Emotion with Knowledge Injection
- **分类: cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.02331v2](http://arxiv.org/pdf/2505.02331v2)**

> **作者:** Hao Cheng; Zhiwei Zhao; Yichao He; Zhenzhen Hu; Jia Li; Meng Wang; Richang Hong
>
> **备注:** Source code and pre-trained models will be available at https://github.com/MSA-LMC/VAEmo
>
> **摘要:** Audiovisual emotion recognition (AVER) aims to infer human emotions from nonverbal visual-audio (VA) cues, offering modality-complementary and language-agnostic advantages. However, AVER remains challenging due to the inherent ambiguity of emotional expressions, cross-modal expressive disparities, and the scarcity of reliably annotated data. Recent self-supervised AVER approaches have introduced strong multimodal representations, yet they predominantly rely on modality-specific encoders and coarse content-level alignment, limiting fine-grained emotional semantic modeling. To address these issues, we propose VAEmo, an efficient two-stage framework for emotion-centric joint VA representation learning with external knowledge injection. In Stage~1, a unified and lightweight representation network is pre-trained on large-scale speaker-centric VA corpora via masked reconstruction and contrastive objectives, mitigating the modality gap and learning expressive, complementary representations without emotion labels. In Stage~2, multimodal large language models automatically generate detailed affective descriptions according to our well-designed chain-of-thought prompting for only a small subset of VA samples; these rich textual semantics are then injected by aligning their corresponding embeddings with VA representations through dual-path contrastive learning, further bridging the emotion gap. Extensive experiments on multiple downstream AVER benchmarks show that VAEmo achieves state-of-the-art performance with a compact design, highlighting the benefit of unified cross-modal encoding and emotion-aware semantic guidance for efficient, generalizable VA emotion representations.
>
---
#### [replaced 002] Benchmarking Sub-Genre Classification For Mainstage Dance Music
- **分类: cs.SD; cs.AI; cs.MM; H.5.5; I.2.1**

- **链接: [http://arxiv.org/pdf/2409.06690v3](http://arxiv.org/pdf/2409.06690v3)**

> **作者:** Hongzhi Shu; Xinglin Li; Hongyu Jiang; Minghao Fu; Xinyu Li
>
> **备注:** WASPAA 2025
>
> **摘要:** Music classification, a cornerstone of music information retrieval, supports a wide array of applications. To address the lack of comprehensive datasets and effective methods for sub-genre classification in mainstage dance music, we introduce a novel benchmark featuring a new dataset and baseline. Our dataset expands the scope of sub-genres to reflect the diversity of recent mainstage live sets performed by leading DJs at global music festivals, capturing the vibrant and rapidly evolving electronic dance music (EDM) scene that engages millions of fans worldwide. We employ a continuous soft labeling approach to accommodate tracks blending multiple sub-genres, preserving their inherent complexity. Experiments demonstrate that even state-of-the-art multimodal large language models (MLLMs) struggle with this task, while our specialized baseline models achieve high accuracy. This benchmark supports applications such as music recommendation, DJ set curation, and interactive multimedia systems, with video demos provided. Our code and data are all open-sourced at https://github.com/Gariscat/housex-v2.git.
>
---
#### [replaced 003] Abstract Sound Fusion with Unconditional Inversion Models
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.11811v2](http://arxiv.org/pdf/2506.11811v2)**

> **作者:** Jing Liu; Enqi Lian; Moyao Deng
>
> **摘要:** An abstract sound is defined as a sound that does not disclose identifiable real-world sound events to a listener. Sound fusion aims to synthesize an original sound and a reference sound to generate a novel sound that exhibits auditory features beyond mere additive superposition of the sound constituents. To achieve this fusion, we employ inversion techniques that preserve essential features of the original sample while enabling controllable synthesis. We propose novel SDE and ODE inversion models based on DPMSolver++ samplers that reverse the sampling process by configuring model outputs as constants, eliminating circular dependencies incurred by noise prediction terms. Our inversion approach requires no prompt conditioning while maintaining flexible guidance during sampling.
>
---
#### [replaced 004] Advances in Intelligent Hearing Aids: Deep Learning Approaches to Selective Noise Cancellation
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.07043v2](http://arxiv.org/pdf/2507.07043v2)**

> **作者:** Haris Khan; Shumaila Asif; Hassan Nasir; Kamran Aziz Bhatti; Shahzad Amin Sheikh
>
> **备注:** 9 pages, 4 figures, submitted as a systematic literature review in AI-based hearing assistance. (June 2025)
>
> **摘要:** The integration of artificial intelligence into hearing assistance marks a paradigm shift from traditional amplification-based systems to intelligent, context-aware audio processing. This systematic literature review evaluates advances in AI-driven selective noise cancellation (SNC) for hearing aids, highlighting technological evolution, implementation challenges, and future research directions. We synthesize findings across deep learning architectures, hardware deployment strategies, clinical validation studies, and user-centric design. The review traces progress from early machine learning models to state-of-the-art deep networks, including Convolutional Recurrent Networks for real-time inference and Transformer-based architectures for high-accuracy separation. Key findings include significant gains over traditional methods, with recent models achieving up to 18.3 dB SI-SDR improvement on noisy-reverberant benchmarks, alongside sub-10 ms real-time implementations and promising clinical outcomes. Yet, challenges remain in bridging lab-grade models with real-world deployment - particularly around power constraints, environmental variability, and personalization. Identified research gaps include hardware-software co-design, standardized evaluation protocols, and regulatory considerations for AI-enhanced hearing devices. Future work must prioritize lightweight models, continual learning, contextual-based classification and clinical translation to realize transformative hearing solutions for millions globally.
>
---
#### [replaced 005] Codec-Based Deepfake Source Tracing via Neural Audio Codec Taxonomy
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12994v3](http://arxiv.org/pdf/2505.12994v3)**

> **作者:** Xuanjun Chen; I-Ming Lin; Lin Zhang; Jiawei Du; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Accepted by Interspeech 2025; Update table 3/4
>
> **摘要:** Recent advances in neural audio codec-based speech generation (CoSG) models have produced remarkably realistic audio deepfakes. We refer to deepfake speech generated by CoSG systems as codec-based deepfake, or CodecFake. Although existing anti-spoofing research on CodecFake predominantly focuses on verifying the authenticity of audio samples, almost no attention was given to tracing the CoSG used in generating these deepfakes. In CodecFake generation, processes such as speech-to-unit encoding, discrete unit modeling, and unit-to-speech decoding are fundamentally based on neural audio codecs. Motivated by this, we introduce source tracing for CodecFake via neural audio codec taxonomy, which dissects neural audio codecs to trace CoSG. Our experimental results on the CodecFake+ dataset provide promising initial evidence for the feasibility of CodecFake source tracing while also highlighting several challenges that warrant further investigation.
>
---
#### [replaced 006] Examining Test-Time Adaptation for Personalized Child Speech Recognition
- **分类: cs.LG; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.13095v2](http://arxiv.org/pdf/2409.13095v2)**

> **作者:** Zhonghao Shi; Xuan Shi; Anfeng Xu; Tiantian Feng; Harshvardhan Srivastava; Shrikanth Narayanan; Maja J. Matarić
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) models often experience performance degradation due to data domain shifts introduced at test time, a challenge that is further amplified for child speakers. Test-time adaptation (TTA) methods have shown great potential in bridging this domain gap. However, the use of TTA to adapt ASR models to the individual differences in each child's speech has not yet been systematically studied. In this work, we investigate the effectiveness of two widely used TTA methods-SUTA, SGEM-in adapting off-the-shelf ASR models and their fine-tuned versions for child speech recognition, with the goal of enabling continuous, unsupervised adaptation at test time. Our findings show that TTA significantly improves the performance of both off-the-shelf and fine-tuned ASR models, both on average and across individual child speakers, compared to unadapted baselines. However, while TTA helps adapt to individual variability, it may still be limited with non-linguistic child speech.
>
---
#### [replaced 007] Align-ULCNet: Towards Low-Complexity and Robust Acoustic Echo and Noise Reduction
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2410.13620v2](http://arxiv.org/pdf/2410.13620v2)**

> **作者:** Shrishti Saha Shetu; Naveen Kumar Desiraju; Wolfgang Mack; Emanuël A. P. Habets
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** The successful deployment of deep learning-based acoustic echo and noise reduction (AENR) methods in consumer devices has spurred interest in developing low-complexity solutions, while emphasizing the need for robust performance in real-life applications. In this work, we propose a hybrid approach to enhance the state-of-the-art (SOTA) ULCNet model by integrating time alignment and parallel encoder blocks for the model inputs, resulting in better echo reduction and comparable noise reduction performance to existing SOTA methods. We also propose a channel-wise sampling-based feature reorientation method, ensuring robust performance across many challenging scenarios, while maintaining overall low computational and memory requirements.
>
---
#### [replaced 008] Language-based Audio Moment Retrieval
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.15672v3](http://arxiv.org/pdf/2409.15672v3)**

> **作者:** Hokuto Munakata; Taichi Nishimura; Shota Nakada; Tatsuya Komatsu
>
> **摘要:** In this paper, we propose and design a new task called audio moment retrieval (AMR). Unlike conventional language-based audio retrieval tasks that search for short audio clips from an audio database, AMR aims to predict relevant moments in untrimmed long audio based on a text query. Given the lack of prior work in AMR, we first build a dedicated dataset, Clotho-Moment, consisting of large-scale simulated audio recordings with moment annotations. We then propose a DETR-based model, named Audio Moment DETR (AM-DETR), as a fundamental framework for AMR tasks. This model captures temporal dependencies within audio features, inspired by similar video moment retrieval tasks, thus surpassing conventional clip-level audio retrieval methods. Additionally, we provide manually annotated datasets to properly measure the effectiveness and robustness of our methods on real data. Experimental results show that AM-DETR, trained with Clotho-Moment, outperforms a baseline model that applies a clip-level audio retrieval method with a sliding window on all metrics, particularly improving Recall1@0.7 by 9.00 points. Our datasets and code are publicly available in https://h-munakata.github.io/Language-based-Audio-Moment-Retrieval.
>
---
#### [replaced 009] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.10876v5](http://arxiv.org/pdf/2409.10876v5)**

> **作者:** Tianao Li; Manxiu Cui; Cheng Ma; Emma Alexander
>
> **备注:** Accepted to IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Photoacoustic computed tomography (PACT) is a non-invasive imaging modality, similar to ultrasound, with wide-ranging medical applications. Conventional PACT images are degraded by wavefront distortion caused by the heterogeneous speed of sound (SOS) in tissue. Accounting for these effects can improve image quality and provide medically useful information, but measuring the SOS directly is burdensome and the existing joint reconstruction method is computationally expensive. Traditional supervised learning techniques are currently inaccessible in this data-starved domain. In this work, we introduce an efficient, self-supervised joint reconstruction method that recovers SOS and high-quality images for ring array PACT systems. To solve this semi-blind inverse problem, we parametrize the SOS using either a pixel grid or a neural field (NF) and update it directly by backpropagating the gradients through a differentiable imaging forward model. Our method removes SOS aberrations more accurately and 35x faster than the current SOTA. We demonstrate the success of our method quantitatively in simulation and qualitatively on experimentally-collected and in vivo data. Our code and synthetic numerical phantoms are available on our project page: https://lukeli0425.github.io/Coord-SoS-PACT/.
>
---
#### [replaced 010] Wi-CBR: Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.11616v2](http://arxiv.org/pdf/2506.11616v2)**

> **作者:** Ruobei Zhang; Shengeng Tang; Huan Yan; Xiang Zhang; Jiabao Guo
>
> **摘要:** The challenge in WiFi-based cross-domain Behavior Recognition lies in the significant interference of domain-specific signals on gesture variation. However, previous methods alleviate this interference by mapping the phase from multiple domains into a common feature space. If the Doppler Frequency Shift (DFS) signal is used to dynamically supplement the phase features to achieve better generalization, enabling model to not only explore a wider feature space but also avoid potential degradation of gesture semantic information. Specifically, we propose a novel Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition (Wi-CBR}, which constructs a dual-branch self-attention module that captures temporal features from phase information reflecting dynamic path length variations, while extracting spatial features from DFS correlated with motion velocity. Moreover, we design a Saliency Guidance Module that employs group attention mechanisms to mine critical activity features, and utilizes gating mechanisms to optimize information entropy, facilitating feature fusion and enabling effective interaction between salient and non-salient behavior characteristics. Extensive experiments on two large-scale public datasets (Widar3.0 and XRF55) demonstrate the superior performance of our method in both in-domain and cross-domain scenarios.
>
---
#### [replaced 011] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12332v3](http://arxiv.org/pdf/2505.12332v3)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.
>
---
#### [replaced 012] Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination
- **分类: eess.SP; cs.CV; 68T45; I.5.4**

- **链接: [http://arxiv.org/pdf/2507.22567v2](http://arxiv.org/pdf/2507.22567v2)**

> **作者:** Weicheng Gao
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** This work is completed on a whim after discussions with my junior colleague. The motion direction angle affects the micro-Doppler spectrum width, thus determining the human motion direction can provide important prior information for downstream tasks such as gait recognition. However, Doppler-Time map (DTM)-based methods still have room for improvement in achieving feature augmentation and motion determination simultaneously. In response, a low-cost but accurate radar-based human motion direction determination (HMDD) method is explored in this paper. In detail, the radar-based human gait DTMs are first generated, and then the feature augmentation is achieved using feature linking model. Subsequently, the HMDD is implemented through a lightweight and fast Vision Transformer-Convolutional Neural Network hybrid model structure. The effectiveness of the proposed method is verified through open-source dataset. The open-source code of this work is released at: https://github.com/JoeyBGOfficial/Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination.
>
---
#### [replaced 013] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.00733v2](http://arxiv.org/pdf/2508.00733v2)**

> **作者:** Le Wang; Jun Wang; Feng Deng; Chen Zhang; Di Zhang; Kun Gai
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and songs coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both sung and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
#### [replaced 014] Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07877v2](http://arxiv.org/pdf/2507.07877v2)**

> **作者:** Chen Feng; Yicheng Lin; Shaojie Zhuo; Chenzheng Su; Ramchalam Kinattinkara Ramakrishnan; Zhaocong Yuan; Xiaopeng Zhang
>
> **摘要:** Recent advances in Automatic Speech Recognition (ASR) have demonstrated remarkable accuracy and robustness in diverse audio applications, such as live transcription and voice command processing. However, deploying these models on resource-constrained edge devices (e.g., IoT device, wearables) still presents substantial challenges due to strict limits on memory, compute and power. Quantization, particularly Post-Training Quantization (PTQ), offers an effective way to reduce model size and inference cost without retraining. Despite its importance, the performance implications of various advanced quantization methods and bit-width configurations on ASR models remain unclear. In this work, we present a comprehensive benchmark of eight state-of-the-art (SOTA) PTQ methods applied to two leading edge-ASR model families, Whisper and Moonshine. We systematically evaluate model performances (i.e., accuracy, memory I/O and bit operations) across seven diverse datasets from the open ASR leader-board, analyzing the impact of quantization and various configurations on both weights and activations. Built on an extension of the LLM compression toolkit, our framework integrates edge-ASR models, diverse advanced quantization algorithms, a unified calibration and evaluation data pipeline, with detailed analysis tools. Our results characterize the trade-offs between efficiency and accuracy, demonstrating that even $3$-bit quantization can succeed on high capacity models when using advanced PTQ techniques. These findings provide valuable insights for optimizing ASR models on low-power, always-on edge devices.
>
---
#### [replaced 015] FedMLAC: Mutual Learning Driven Heterogeneous Federated Audio Classification
- **分类: cs.SD; cs.DC; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10207v2](http://arxiv.org/pdf/2506.10207v2)**

> **作者:** Jun Bai; Rajib Rana; Di Wu; Youyang Qu; Xiaohui Tao; Ji Zhang; Carlos Busso; Shivakumara Palaiahnakote
>
> **备注:** updated version for the first submission
>
> **摘要:** Federated Learning (FL) offers a privacy-preserving framework for training audio classification (AC) models across decentralized clients without sharing raw data. However, Federated Audio Classification (FedAC) faces three major challenges: data heterogeneity, model heterogeneity, and data poisoning, which degrade performance in real-world settings. While existing methods often address these issues separately, a unified and robust solution remains underexplored. We propose FedMLAC, a mutual learning-based FL framework that tackles all three challenges simultaneously. Each client maintains a personalized local AC model and a lightweight, globally shared Plug-in model. These models interact via bidirectional knowledge distillation, enabling global knowledge sharing while adapting to local data distributions, thus addressing both data and model heterogeneity. To counter data poisoning, we introduce a Layer-wise Pruning Aggregation (LPA) strategy that filters anomalous Plug-in updates based on parameter deviations during aggregation. Extensive experiments on four diverse audio classification benchmarks, including both speech and non-speech tasks, show that FedMLAC consistently outperforms state-of-the-art baselines in classification accuracy and robustness to noisy data.
>
---
#### [replaced 016] MuteSwap: Visual-informed Silent Video Identity Conversion
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00498v3](http://arxiv.org/pdf/2507.00498v3)**

> **作者:** Yifan Liu; Yu Fang; Zhouhan Lin
>
> **摘要:** Conventional voice conversion modifies voice characteristics from a source speaker to a target speaker, relying on audio input from both sides. However, this process becomes infeasible when clean audio is unavailable, such as in silent videos or noisy environments. In this work, we focus on the task of Silent Face-based Voice Conversion (SFVC), which does voice conversion entirely from visual inputs. i.e., given images of a target speaker and a silent video of a source speaker containing lip motion, SFVC generates speech aligning the identity of the target speaker while preserving the speech content in the source silent video. As this task requires generating intelligible speech and converting identity using only visual cues, it is particularly challenging. To address this, we introduce MuteSwap, a novel framework that employs contrastive learning to align cross-modality identities and minimize mutual information to separate shared visual features. Experimental results show that MuteSwap achieves impressive performance in both speech synthesis and identity conversion, especially under noisy conditions where methods dependent on audio input fail to produce intelligible results, demonstrating both the effectiveness of our training approach and the feasibility of SFVC.
>
---
#### [replaced 017] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.23511v2](http://arxiv.org/pdf/2507.23511v2)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Xunying Liu; Junbo Zhang; Jian Luan
>
> **备注:** 9 main pages, 5 figures, 3 tables, and 14 appendix pages
>
> **摘要:** While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at https://github.com/xiaomi-research/mecat
>
---
#### [replaced 018] Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System
- **分类: cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.23298v2](http://arxiv.org/pdf/2507.23298v2)**

> **作者:** Kazushi Kato; Koji Inoue; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** Accepted by 27th ACM International Conference on Multimodal Interaction (ICMI '25), Long paper
>
> **摘要:** In human dialogue, nonverbal information such as nodding and facial expressions is as crucial as verbal information, and spoken dialogue systems are also expected to express such nonverbal behaviors. We focus on nodding, which is critical in an attentive listening system, and propose a model that predicts both its timing and type in real time. The proposed model builds on the voice activity projection (VAP) model, which predicts voice activity from both listener and speaker audio. We extend it to prediction of various types of nodding in a continuous and real-time manner unlike conventional models. In addition, the proposed model incorporates multi-task learning with verbal backchannel prediction and pretraining on general dialogue data. In the timing and type prediction task, the effectiveness of multi-task learning was significantly demonstrated. We confirmed that reducing the processing rate enables real-time operation without a substantial drop in accuracy, and integrated the model into an avatar attentive listening system. Subjective evaluations showed that it outperformed the conventional method, which always does nodding in sync with verbal backchannel. The code and trained models are available at https://github.com/MaAI-Kyoto/MaAI.
>
---
#### [replaced 019] Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.01690v2](http://arxiv.org/pdf/2504.01690v2)**

> **作者:** Taehan Lee; Hyukjun Lee
>
> **备注:** Accepted at the 28th European Conference on Artificial Intelligence (ECAI 2025). Source code is available at https://github.com/andylee-24/token-pruning-audio-transformer
>
> **摘要:** Vision Transformers (ViTs) have achieved state-of-the-art performance across various computer vision tasks, but their high computational cost remains a challenge. Token pruning has been proposed to reduce this cost by selectively removing less important tokens. While effective in vision tasks by discarding non-object regions, applying this technique to audio tasks presents unique challenges, as distinguishing relevant from irrelevant regions in time-frequency representations is less straightforward. In this study, for the first time, we applied token pruning to ViT-based audio classification models using Mel-spectrograms and analyzed the trade-offs between model performance and computational cost: TopK token pruning can reduce MAC operations of AudioMAE and AST by 30-40%, with less than a 1% drop in accuracy. Our analysis reveals that while high-intensity or high-variation tokens contribute significantly to model accuracy, low-intensity or low variation tokens also remain important when token pruning is applied; pruning solely based on the intensity or variation of signals in a patch leads to a noticeable drop in accuracy. We support our claim by measuring high correlation between attention scores and these statistical features and by showing retained tokens consistently receive distinct attention compared to pruned ones. We also show that AudioMAE retains more low-intensity tokens than AST. This can be explained by AudioMAE's self-supervised reconstruction objective, which encourages attention to all patches, whereas AST's supervised training focuses on label-relevant tokens.
>
---
