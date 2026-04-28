# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] RTCFake: Speech Deepfake Detection in Real-Time Communication
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决实时通信中复杂失真下的检测难题。构建了首个针对RTC的大型数据集RTCFake，并提出PCL策略提升模型泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23742](https://arxiv.org/pdf/2604.23742)**

> **作者:** Jun Xue; Zhuolin Yi; Yihuan Huang; Yanzhen Ren; Yujie Chen; Cunhang Fan; Zicheng Su; Yonghong Zhang; Bo Cai
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** With the rapid advancement of speech generation technologies, the threat posed by speech deepfakes in real-time communication (RTC) scenarios has intensified. However, existing detection studies mainly focus on offline simulations and struggle to cope with the complex distortions introduced during RTC transmission, including unknown speech enhancement processes (e.g., noise suppression) and codec compression. To address this challenge, we present the first large-scale speech deepfake dataset tailored for RTC scenarios, termed \textit{RTCFake}, totaling approximately 600 hours. The dataset is constructed by transmitting speech through multiple mainstream social media and conferencing platforms (e.g., Zoom), enabling precise pairing between offline and online speech. In addition, we propose a phoneme-guided consistency learning (PCL) strategy that enforces models to learn platform-invariant semantic structural representations. In this paper, the RTCFake dataset is divided into training, development, and evaluation sets. The evaluation set further includes both unseen RTC platforms and unseen complex noise conditions, thereby providing a more realistic and challenging evaluation benchmark for speech deepfake detection. Furthermore, the proposed PCL strategy achieves significant improvements in both cross-platform generalization and noise robustness, offering an effective and generalizable modeling paradigm. The \textit{RTCFake} dataset is provided in the {this https URL}.
>
---
#### [new 002] Explainable AI in Speaker Recognition -- Making Latent Representations Understandable
- **分类: eess.AS; cs.AI; eess.SP**

- **简介: 该论文属于语音识别中的可解释AI研究，旨在揭示网络表示中的层次聚类结构，并通过HCCM算法实现语义匹配，提升模型理解性。**

- **链接: [https://arxiv.org/pdf/2604.23354](https://arxiv.org/pdf/2604.23354)**

> **作者:** Yanze Xu; Wenwu Wang; Mark D. Plumbley
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Neural networks can be trained to learn task-relevant representations from data. Understanding how these networks make decisions falls within the Explainable AI (XAI) domain. This paper proposes to study an XAI topic: uncovering unknown organisational patterns in network representations, particularly those representations learned by the speaker recognition network that recognises the speaker identity of utterances. Past studies employed algorithms (e.g. t-distributed Stochastic Neighbour Embedding and K-means) to analyse and visualise how network representations form independent clusters, indicating the presence of flat clustering phenomena within the space defined by these representations. In contrast, this work applies two algorithms -- Single-Linkage Clustering (SLINK) and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) -- to analyse how representations form clusters with hierarchical relationships rather than being independent, thereby demonstrating the existence of hierarchical clustering phenomena within the network representation space. To semantically understand the above hierarchical clustering phenomena, a new algorithm, termed Hierarchical Cluster-Class Matching (HCCM), is designed to perform one-to-one matching between predefined semantic classes and hierarchical representation clusters (i.e. those produced by SLINK or HDBSCAN). Some hierarchical clusters are successfully matched to individual semantic classes (e.g. male, UK), while others to conjunctions of semantic classes (e.g. male and UK, female and Ireland). A new metric, Liebig's score, is proposed to quantify the performance of each matching behaviour, allowing us to diagnose the factor that most strongly limits matching performance.
>
---
#### [new 003] Speech Enhancement Based on Drifting Models
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文提出DriftSE，一种基于漂移模型的语音增强框架，解决单步语音去噪问题，通过分布匹配实现高效增强。**

- **链接: [https://arxiv.org/pdf/2604.24199](https://arxiv.org/pdf/2604.24199)**

> **作者:** Liang Xu; Diego Caviedes-Nozal; Bastiaan Kleijn; Longfei Felix Yan; Rasmus Kongsgaard Olsson
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** We propose Speech Enhancement based on Drifting Models (DriftSE), a novel generative framework that formulates denoising as an equilibrium problem. Rather than relying on iterative sampling, DriftSE natively achieves one-step inference by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution. This evolution is driven by a Drifting Field, a learned correction vector that guides samples toward the high-density regions of the clean distribution, which naturally facilitates training on unpaired data by matching distributions rather than paired samples. We investigate the framework under two formulations: a direct mapping from the noisy observation, and a stochastic conditional generative model from a Gaussian prior. Experiments on the VoiceBank-DEMAND benchmark demonstrate that DriftSE achieves high-fidelity enhancement in a single step, outperforming multi-step diffusion baselines and establishing a new paradigm for speech enhancement.
>
---
#### [new 004] Audio2Tool: Bridging Spoken Language Understanding and Function Calling
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出Audio2Tool，一个用于评估语音语言模型工具调用能力的大规模数据集，解决现有基准不足的问题。任务属于语音理解与工具调用。**

- **链接: [https://arxiv.org/pdf/2604.22821](https://arxiv.org/pdf/2604.22821)**

> **作者:** Ramit Pahwa; Apoorva Beedu; Parivesh Priye; Rutu Gandhi; Saloni Takawale; Aruna Baijal; Zengli Yang
>
> **摘要:** Voice assistants increasingly rely on Speech Language Models (SpeechLMs) to interpret spoken queries and execute complex tasks, yet existing benchmarks lack domain breadth, acoustic diversity, and compositional reasoning complexity to evaluate tool-calling performance. We introduce Audio2Tool, a large-scale dataset comprising approximately 30,000 queries designed to assess tool-calling capabilities of SpeechLMs across three primary domains: Smart Car, Smart Home, and Wearables. Our benchmark features a multi-tier complexity hierarchy, ranging from simple direct commands to complex multi-intent and needle-in-a-haystack extraction to isolate distinct failure modes. To ensure realism, we employ zero-shot voice cloning text-to-speech synthesis and diverse noise profiles to simulate in-the-wild conditions. Evaluations of state-of-the-art SpeechLMs and ASR-LLM pipelines show strong performance on simple commands but significant degradation under compositional and acoustic challenges. We will release the dataset and benchmark upon acceptance.
>
---
#### [new 005] In-Sync: Adaptation of Speech Aware Large Language Models for ASR with Word Level Timestamp Predictions
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，解决ASR中时间戳预测问题。通过改进模型，直接预测单词时间戳，提升对齐精度和整体识别效果。**

- **链接: [https://arxiv.org/pdf/2604.22817](https://arxiv.org/pdf/2604.22817)**

> **作者:** Xulin Fan; Vishal Sunder; Samuel Thomas; Mark Hasegawa-Johnson; Brian Kingsbury; George Saon
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Recent advances in speech-aware language models have coupled strong acoustic encoders with large language models, enabling systems that move beyond transcription to produce richer outputs. Among these, word-level timestamp prediction is critical for applications such as captioning, media search, and multimodal synchronization, yet it is often handled by external alignment tools. In this work, we extend an existing speech-aware language model to predict timestamps directly alongside transcripts. We introduce a set of novel lightweight training strategies that improve alignment robustness while preserving recognition quality. Experiments across multiple datasets show that these strategies not only enhance timestamp accuracy, but also yield gains in overall ASR performance. Together, they demonstrate an efficient and unified approach to speech recognition with precise timestamp prediction.
>
---
#### [new 006] Opening the Design Space: Two Years of Performance with Intelligent Musical Instruments
- **分类: cs.SD; cs.HC**

- **简介: 该论文属于音乐生成与AI交互任务，旨在解决传统AI工具不支持艺术创作的问题。通过构建低成本AI乐器平台，探索智能乐器的设计空间与创新交互方式。**

- **链接: [https://arxiv.org/pdf/2604.23583](https://arxiv.org/pdf/2604.23583)**

> **作者:** Charles Patrick Martin
>
> **备注:** Accepted for publication at the International Conference on New Interfaces for Musical Expression (NIME) 2026
>
> **摘要:** Machine generation of symbolic music and digital audio are hot topics but there have been relatively few digital musical instruments that integrate generative AI. Present musical AI tools are not artist centred and do not support experimentation or integrating into musical instruments or practices. This work introduces an inexpensive generative AI instrument platform based on a single board computer that connects via MIDI to other musical devices. The platform uses artist-collected datasets with models trained on a regular computer. This paper asks what the design space of intelligent musical instruments might look like when accessible and portable AI systems are available for artistic exploration. I contribute five examples of instruments created and tested through a two-year first-person artistic research process. These show that (re)mapping can replace retraining for discovering AI interaction, that fast input interleaving is a new co-creative strategy, that small-data AI models can be a transportable design resource, and that cheap hardware can lower barriers to inclusion. This work could enable artists to explore new interaction and performance schemes with intelligent musical instruments.
>
---
#### [new 007] Spectro-Temporal Modulation Representation Framework for Human-Imitated Speech Detection
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音伪造检测任务，旨在解决人类仿声语音难以检测的问题。通过构建基于听觉感知的时频调制框架，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.23241](https://arxiv.org/pdf/2604.23241)**

> **作者:** Khalid Zaman; Masashi Unoki
>
> **摘要:** Human-imitated speech poses a greater challenge than AI-generated speech for both human listeners and automatic detection systems. Unlike AI-generated speech, which often contains artifacts, over-smoothed spectra, or robotic cues, imitated speech is produced naturally by humans, thereby preserving a higher degree of naturalness that makes imitation-based speech forgery significantly more challenging to detect using conventional acoustic or cepstral features. To overcome this challenge, this study proposes an auditory perception-based Spectro-Temporal Modulation (STM) representation framework for human-imitated speech detection. The STM representations are derived from two cochlear filterbank models: the Gammatone Filterbank (GTFB), which simulates frequency selectivity and can be regarded as a first approximation of cochlear filtering, and the Gammachirp Filterbank (GCFB), which further models both frequency selectivity and level-dependent asymmetry. These STM representations jointly capture temporal and spectral fluctuations in speech signals, corresponding to changes over time in the spectrogram and variations along the frequency axis related to human auditory perception. We also introduce a Segmental-STM representation to analyze short-term modulation patterns across overlapping time windows, enabling high-resolution modeling of temporal speech variations. Experimental results show that STM representations are effective for human-imitated speech detection, achieving accuracy levels close to those of human listeners. In addition, Segmental-STM representations are more effective, surpassing human perceptual performance. The findings demonstrate that perceptually inspired spectro-temporal modeling is promising for detecting imitation-based speech attacks and improving voice authentication robustness.
>
---
#### [new 008] HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频语言模型任务，解决高推理成本问题。通过动态路由注意力头，实现音频令牌压缩，提升效率。**

- **链接: [https://arxiv.org/pdf/2604.23717](https://arxiv.org/pdf/2604.23717)**

> **作者:** Peize He; Yaodi Luo; Xiaoqian Liu; Xuyang Liu; Jiahang Deng; Yaosong Du; Bangyu Li; Xiyan Gui; Yuxuan Chen; Linfeng Zhang
>
> **备注:** Homepage: this https URL
>
> **摘要:** Recent large audio language models (LALMs) demonstrate remarkable capabilities in processing extended multi-modal sequences, yet incur high inference costs. Token compression is an effective method that directly reduces redundant tokens in the sequence. Existing compression methods usually assume that all attention heads in LALMs contribute equally to various audio tasks and calculate token importance by averaging scores across all heads. However, our analysis demonstrates that attention heads exhibit distinct behaviors across diverse audio domains. We further reveal that only a sparse subset of attention heads actively responds to audio, with completely different performance when handling semantic and acoustic tasks. In light of this observation, we propose HeadRouter, a head-importance-aware token pruning method that perceives the varying importance of attention heads in different audio tasks to maximize the retention of crucial tokens. HeadRouter is training-free and can be applied to various LALMs. Extensive experiments on the AudioMarathon and MMAU-Pro benchmarks demonstrate that HeadRouter achieves state-of-the-art compression performance, exceeding the baseline model even when retaining 70% of the audio tokens and achieving 101.8% and 103.0% of the vanilla average on Qwen2.5-Omni-3B and Qwen2.5-Omni-7B, respectively.
>
---
#### [new 009] RAS: a Reliability Oriented Metric for Automatic Speech Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于自动语音识别任务，解决ASR在噪声下误判的问题。提出RAS度量标准与弃权框架，提升转录可靠性同时保持准确。**

- **链接: [https://arxiv.org/pdf/2604.24278](https://arxiv.org/pdf/2604.24278)**

> **作者:** Wenbin Huang; Yuhang Qiu; Bohan Li; Yiwei Guo; Jing Peng; Hankun Wang; Xie Chen; Kai Yu
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Automatic speech recognition systems often produce confident yet incorrect transcriptions under noisy or ambiguous conditions, which can be misleading for both users and downstream applications. Standard evaluation based on Word Error Rate focuses solely on accuracy and fails to capture transcription reliability. We introduce an abstention-aware transcription framework that enables ASR models to explicitly abstain from uncertain segments. To evaluate reliability under abstention, we propose RAS, a reliability-oriented metric that balances transcription informativeness and error aversion, with its trade-off parameter calibrated by human preference. We then train an abstention-aware ASR model through supervised bootstrapping followed by reinforcement learning. Our experiments demonstrate substantial improvements in transcription reliability while maintaining competitive accuracy.
>
---
#### [new 010] Predictive Directional Selective Fixed-Filter Active Noise Control for Moving Sources via a Convolutional Recurrent Neural Network
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于主动降噪任务，旨在解决移动噪声源方向变化时降噪效果差的问题。通过CRNN预测未来噪声，提升动态降噪性能。**

- **链接: [https://arxiv.org/pdf/2604.23144](https://arxiv.org/pdf/2604.23144)**

> **作者:** Boxiang Wang; Zhengding Luo; Dongyuan Shi; Junwei Ji; Xiruo Su; Woon-Seng Gan
>
> **摘要:** Directional Selective Fixed-Filter Active Noise Control (D-SFANC) can effectively attenuate noise from different directions by selecting the suitable pre-trained control filter based on the Direction-of-Arrival (DoA) of the current noise. However, this method is weak at tracking the direction variations of non-stationary noise, such as that from a moving source. Therefore, this work proposes a Predictive Directional SFANC (PD-SFANC) method that uses a Convolutional Recurrent Neural Network (CRNN) to capture the hidden temporal dynamics of the moving noise and predict the control filter to cancel future noise. Accordingly, the proposed method can significantly improve its noise-tracking ability and dynamic noise-reduction performance. Furthermore, numerical simulations confirm the superiority of the proposed method for handling moving sources across various movement scenarios, compared to several representative ANC baselines.
>
---
#### [new 011] All That Glitters Is Not Audio: Rethinking Text Priors and Audio Reliance in Audio-Language Evaluation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频-语言模型评估任务，旨在解决现有基准无法准确衡量音频理解的问题。通过分析文本先验和音频依赖性，发现模型依赖音频程度有限，提出改进评估方法。**

- **链接: [https://arxiv.org/pdf/2604.24401](https://arxiv.org/pdf/2604.24401)**

> **作者:** Leonardo Haw-Yang Foo; Chih-Kai Yang; Chen-An Li; Ke-Han Lu; Hung-yi Lee
>
> **备注:** 6 pages, 3 figures, 5 tables
>
> **摘要:** Large Audio-Language Models show consistent performance gains across speech and audio benchmarks, yet high scores may not reflect true auditory perception. If a model can answer questions without processing the acoustic signal, the benchmark fails as a measure of auditory understanding. We present a diagnostic framework using two axes: text prior, which measures answerability from text and general knowledge alone, and audio reliance, which assesses actual dependency on the acoustic signal. Evaluating eight LALMs across three benchmarks, we find that models retain 60-72% of their full audio scores even without any audio input. Moreover, among items that require audio, only 3.0-4.2% need the complete audio clip; the majority can be resolved using localized fragments. These findings challenge the assumption that benchmark performance equals robust audio understanding, and we conclude with practical guidelines for improving evaluation reliability and benchmark design.
>
---
#### [new 012] An event-based sequence modeling approach to recognizing non-triad chords with oversegmentation minimization
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决自动和弦识别中的过度分割、数据稀缺及不平衡问题。通过段级序列建模和结构化分词提升复杂和弦识别效果。**

- **链接: [https://arxiv.org/pdf/2604.24386](https://arxiv.org/pdf/2604.24386)**

> **作者:** Leekyung Kim; Jonghun Park
>
> **备注:** accepted to ICASSP 2026
>
> **摘要:** Automatic chord recognition (ACR) extracts time-aligned chord labels from music audio recordings. Despite recent advances, ACR still struggles with oversegmentation, data scarcity, and imbalance, especially in recognizing complex chords such as non-triads, which are unpopular in existing datasets. To address these challenges, we reformulate ACR as a segment-level sequence-to-sequence prediction task, where chord sequences are predicted auto-regressively rather than frame by frame. This design mitigates excessive segmentation by detecting chord changes only at segment boundaries. We further introduce two types of token representations and an encoder pre-training method, both specifically designed for time-aligned chord modeling. Experimental results show that our model improves performance in both chord recognition and segmentation, with notable gains for complex and infrequent chord types. These findings demonstrate the effectiveness of segment-level sequence modeling, structured tokenization, and representation learning for advancing chord recognition systems.
>
---
#### [new 013] Come Together: Analyzing Popular Songs Through Statistical Embeddings
- **分类: stat.AP; cs.SD**

- **简介: 该论文属于音乐分析任务，旨在通过统计嵌入方法研究披头士歌曲的结构与风格演变。工作包括构建歌曲嵌入，分析专辑聚类及创作风格变化。**

- **链接: [https://arxiv.org/pdf/2604.22925](https://arxiv.org/pdf/2604.22925)**

> **作者:** Matthew Esmaili Mallory; Mark Glickman; Jason Brown
>
> **摘要:** Statistical modeling of popular music presents a unique challenge due to the complexity of song structures, which cannot be easily analyzed using conventional statistical tools. However, recent advances in data science have shown that converting non-standard data objects into real vector-valued embeddings enables meaningful statistical analysis. In this work, we demonstrate an approach based on logistic principal component analysis to construct embeddings from global song features, allowing for standard multivariate analysis. We apply this method to a corpus of Lennon and McCartney songs from 1962-1966, using embeddings derived from chords, melodic notes, chord and pitch transitions, and melodic contours. Our analysis explores how these song embeddings cluster by Beatles album, how songwriting styles evolved over time, and whether Lennon and McCartney's compositions exhibited convergence or divergence. This embedding-based approach offers a powerful framework for statistically examining musical structure and stylistic development in popular music.
>
---
#### [new 014] Hallo-Live: Real-Time Streaming Joint Audio-Video Avatar Generation with Asynchronous Dual-Stream and Human-Centric Preference Distillation
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于实时音视频虚拟人生成任务，解决现有模型速度慢、加速后质量下降的问题。提出Hallo-Live框架，结合异步双流扩散与人类偏好蒸馏，提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.23632](https://arxiv.org/pdf/2604.23632)**

> **作者:** Chunyu Li; Jiaye Li; Ruiqiao Mei; Haoyuan Xia; Hao Zhu; Jingdong Wang; Siyu Zhu
>
> **摘要:** Real-time text-driven joint audio-video avatar generation requires jointly synthesizing portrait video and speech with high fidelity and precise synchronization, yet existing audio-visual diffusion models remain too slow for interactive use and often degrade noticeably after aggressive acceleration. We present Hallo-Live, a streaming framework for joint audio-visual avatar generation that combines asynchronous dual-stream diffusion with human-centric preference-guided distillation. To reduce articulation lag in causal generation, we introduce Future-Expanding Attention, which allows each video block to access synchronous audio together with a short horizon of future phonetic cues. To mitigate the quality loss of few-step distillation, we further propose Human-Centric Preference-Guided DMD (HP-DMD), which reweights training samples using rewards from visual fidelity, speech naturalness, and audio-visual synchronization. On two NVIDIA H200 GPUs, Hallo-Live runs at 20.38 FPS with 0.94 seconds latency, yielding 16.0x higher throughput and 99.3x lower latency than the teacher model Ovi. Despite this speedup, it retains strong generation quality, reaching comparable VideoAlign overall score and Sync Confidence score while outperforming other accelerated baselines in the overall quality-efficiency trade-off. Qualitative results further show robust generalization across photorealistic, multi-speaker, and stylized scenarios. To the best of our knowledge, Hallo-Live is the first framework to combine streaming dual-stream diffusion with preference-guided distillation for real-time, text-driven audio-visual generation.
>
---
#### [new 015] Talker-T2AV: Joint Talking Audio-Video Generation with Autoregressive Diffusion Modeling
- **分类: cs.CV; cs.CL; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于语音驱动的视频生成任务，旨在解决跨模态一致性问题。提出Talker-T2AV框架，通过共享骨干和专用解码器分离高、低层次信息，提升唇同步与质量。**

- **链接: [https://arxiv.org/pdf/2604.23586](https://arxiv.org/pdf/2604.23586)**

> **作者:** Zhen Ye; Xu Tan; Aoxiong Yin; Hongzhan Lin; Guangyan Zhang; Peiwen Sun; Yiming Li; Chi-Min Chan; Wei Ye; Shikun Zhang; Wei Xue
>
> **摘要:** Joint audio-video generation models have shown that unified generation yields stronger cross-modal coherence than cascaded approaches. However, existing models couple modalities throughout denoising via pervasive attention, treating high-level semantics and low-level details in a fully entangled manner. This is suboptimal for talking head synthesis: while audio and facial motion are semantically correlated, their low-level realizations (acoustic signals and visual textures) follow distinct rendering processes. Enforcing joint modeling across all levels causes unnecessary entanglement and reduces efficiency. We propose Talker-T2AV, an autoregressive diffusion framework where high-level cross-modal modeling occurs in a shared backbone, while low-level refinement uses modality-specific decoders. A shared autoregressive language model jointly reasons over audio and video in a unified patch-level token space. Two lightweight diffusion transformer heads decode the hidden states into frame-level audio and video latents. Experiments on talking portrait benchmarks show Talker-T2AV outperforms dual-branch baselines in lip-sync accuracy, video quality, and audio quality, achieving stronger cross-modal consistency than cascaded pipelines.
>
---
#### [new 016] Robust Audio-Text Retrieval via Cross-Modal Attention and Hybrid Loss
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音频-文本检索任务，旨在解决长且噪声大的音频与文本对齐问题。通过跨模态注意力和混合损失函数提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.23323](https://arxiv.org/pdf/2604.23323)**

> **作者:** Meizhu Liu; Matthew Rowe; Amit Agarwal; Michael Avendi; Yassi Abbasi; Hitesh Laxmichand Patel; Paul Li; Kyu J. Han; Tao Sheng; Sujith Ravi; Dan Roth
>
> **摘要:** Audio-text retrieval enables semantic alignment between audio content and natural language queries, supporting applications in multimedia search, accessibility, and surveillance. However, current state-of-the-art approaches struggle with long, noisy, and weakly labeled audio due to their reliance on contrastive learning and large-batch training. We propose a novel multimodal retrieval framework that refines audio and text embeddings using a cross-modal embedding refinement module combining transformer-based projection, linear mapping, and bidirectional attention. To further improve robustness, we introduce a hybrid loss function blending cosine similarity, $\mathcal{L}_{1}$, and contrastive objectives, enabling stable training even under small-batch constraints. Our approach efficiently handles long-form and noisy audio (SNR 5 to 15) via silence-aware chunking and attention-based pooling. Experiments on benchmark datasets demonstrate improvements over prior methods.
>
---
## 更新

#### [replaced 001] Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出Speech-FT框架，解决语音模型微调后跨任务泛化能力下降的问题。通过两阶段微调保持特征相似性，提升多种任务性能。**

- **链接: [https://arxiv.org/pdf/2502.12672](https://arxiv.org/pdf/2502.12672)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Hao Tang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Model and code available at: this https URL
>
> **摘要:** Fine-tuning speech representation models can enhance performance on specific tasks but often compromises their cross-task generalization ability. This degradation is often caused by excessive changes in the representations, making it difficult to retain information learned during pre-training. Existing approaches, such as regularizing weight changes during fine-tuning, may fail to maintain sufficiently high feature similarity with the pre-trained model, and thus could possibly lose cross-task generalization. To address this issue, we propose Speech-FT, a novel two-stage fine-tuning framework designed to maintain cross-task generalization while benefiting from fine-tuning. Speech-FT first applies fine-tuning specifically designed to reduce representational drift, followed by weight-space interpolation with the pre-trained model to restore cross-task generalization. Extensive experiments on HuBERT, wav2vec 2.0, DeCoAR 2.0, and WavLM Base+ demonstrate that Speech-FT consistently improves performance across a wide range of supervised, unsupervised, and multitask fine-tuning scenarios. Moreover, Speech-FT achieves superior cross-task generalization compared to fine-tuning baselines that explicitly constrain weight changes, such as weight-space regularization and LoRA fine-tuning. Our analysis reveals that Speech-FT maintains higher feature similarity to the pre-trained model compared to alternative strategies, despite allowing larger weight-space updates. Notably, Speech-FT achieves significant improvements on the SUPERB benchmark. For example, when fine-tuning HuBERT on automatic speech recognition, Speech-FT is able to reduce phone error rate from 5.17% to 3.94%, lower word error rate from 6.38% to 5.75%, and increase speaker identification accuracy from 81.86% to 84.11%. Speech-FT provides a simple yet powerful solution for further refining speech representation models after pre-training.
>
---
#### [replaced 002] CodecSep: Prompt-Driven Universal Sound Separation on Neural Audio Codec Latents
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出CodecSep，解决低延迟音频分离问题。它在神经音频编码器潜在空间中实现通用声音分离，提升效率并支持开放式词汇。**

- **链接: [https://arxiv.org/pdf/2509.11717](https://arxiv.org/pdf/2509.11717)**

> **作者:** Adhiraj Banerjee; Vipul Arora
>
> **备注:** main content- 27 pages, total - 53 pages, 12 figure, pre-print, under review
>
> **摘要:** Text-guided sound separation enables flexible audio editing, assistive listening, and open-domain source extraction, but systems such as AudioSep remain too expensive for low-latency edge or codec-mediated deployment. Existing neural audio codec separators are efficient, yet largely restricted to fixed stems or closed taxonomies. We introduce CodecSep, a prompt-driven universal sound separation framework that extracts sources directly in neural audio codec latent space. CodecSep combines a frozen DAC backbone with a lightweight FiLM-conditioned Transformer masker driven by CLAP text embeddings, enabling open-vocabulary separation while preserving codec-native efficiency. Across dnr-v2 and five open-domain benchmarks, CodecSep consistently improves over AudioSep in SI-SDR, remains competitive in ViSQOL, and achieves clear gains in human MOS-LQS. Controlled analyses show that fine-grained prompts outperform coarse labels, and that explicit latent masking is substantially more effective than decoder-style latent generation in codec space. Qualitative diagnostics show that neural audio codec latents retain source-dependent structure, which CodecSep exploits mainly through channel-wise source-conditioned modulation. CodecSep also provides a practical code-stream deployment path. When audio is transmitted as neural audio codec codes, CodecSep maps codes to embeddings, separates directly in codec space, and outputs waveforms or re-quantized codes, avoiding the decode-separate-re-encode loop. In this regime, CodecSep requires only 1.35 GMACs end-to-end: about 54 times less compute than AudioSep in the same pipeline and 25 times lower separator-only compute, with much lower latency and memory. More broadly, CodecSep offers a blueprint for codec-native downstream audio processing.
>
---
#### [replaced 003] pTSE-T: Presentation Target Speaker Extraction using Unaligned Text Cues
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于目标说话人提取任务，旨在利用有限且未对齐的文本线索提取目标语音，解决实际场景中难以获取其他辅助信息的问题。**

- **链接: [https://arxiv.org/pdf/2411.03109](https://arxiv.org/pdf/2411.03109)**

> **作者:** Ziyang Jiang; Jiahe Lei; Xueyan Chen; Yifan Zhang; Zexu Pan; Wei Xue; Xinyuan Qian
>
> **摘要:** Target Speaker Extraction (TSE) aims to extract the clean speech of the target speaker in an audio mixture, eliminating irrelevant background noise and speech. While prior work has explored various auxiliary cues including pre-recorded speech, visual information, and spatial information, the acquisition and selection of such strong cues are infeasible in many practical scenarios. Differently, in this paper, we condition the TSE algorithm on semantic cues extracted from limited and unaligned text contents, such as condensed points from a presentation slide. This method is particularly useful in scenarios like meetings, poster sessions, or lecture presentations, where acquiring other cues in real time may be challenging. To this end, we design two different networks. Specifically, our proposed Text Prompt Extractor Network (TPE) fuses audio features with content-based semantic cues to facilitate time-frequency mask generation to filter out extraneous noise. The experimental results show the efficacy in accurately extracting the target speaker's speech by utilizing semantic cues derived from limited and unaligned text, resulting in SI-SDRi of 12.16 dB, SDRi of 12.66 dB, PESQi of 0.830 and STOIi of 0.150.
>
---
#### [replaced 004] BERT-APC: A Reference-free Framework for Automatic Pitch Correction via Musical Context Inference
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于自动音高校正任务，解决无参考音高校正问题。提出BERT-APC框架，结合音乐上下文推理，提升校正准确性和表现力。**

- **链接: [https://arxiv.org/pdf/2511.20006](https://arxiv.org/pdf/2511.20006)**

> **作者:** Sungjae Kim; Kihyun Na; Jinyoung Choi; Injung Kim
>
> **备注:** 12 pages, 6 figures, 5 tables. Accepted for publication in IEEE/ACM Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Automatic Pitch Correction (APC) enhances vocal recordings by aligning pitch deviations with intended musical notes. However, existing APC systems either rely on reference pitches, which limits practical applicability, or employ simple pitch estimation algorithms that often fail to preserve expressiveness and naturalness. We propose BERT-APC, a reference-free APC framework that corrects pitch errors while maintaining the expressiveness and naturalness of vocal performances. In BERT-APC, a stationary pitch predictor first estimates the stationary pitch of each note from the detuned singing voice, where stationary pitch is the continuous pitch from the stable region of a note and approximates its perceived pitch. A context-aware note pitch predictor then infers the intended pitch sequence using a repurposed music language model that incorporates musical context. Finally, a note-level correction algorithm fixes pitch errors while preserving intentional deviations for emotional expression. We also introduce a learnable data augmentation strategy that improves robustness by simulating realistic detuning patterns. Compared to two recent singing voice transcription models, BERT-APC demonstrated superior target note pitch prediction, outperforming the second-best model, ROSVOT, by 10.49 percentage points on highly detuned samples in raw pitch accuracy. In the MOS test, BERT-APC achieved the highest quality rating of $4.32 \pm 0.15$, significantly higher than Auto-Tune ($3.22 \pm 0.18$) and Melodyne ($3.08 \pm 0.18$), while maintaining a comparable ability to preserve expressive nuances. To the best of our knowledge, this is the first APC model that leverages a music language model to achieve reference-free pitch correction with symbolic musical context. The corrected audio samples are available at this https URL.
>
---
#### [replaced 005] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音翻译任务，比较SpeechLLMs与传统级联系统的性能，旨在验证语音模态集成对翻译质量的影响。**

- **链接: [https://arxiv.org/pdf/2512.16378](https://arxiv.org/pdf/2512.16378)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at this https URL | Accepted at TACL, this version is a pre-MIT Press publication version
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which directly process spoken language and enable speech-to-text translation (ST) and other downstream tasks, bypassing traditional transcription-based pipelines. Whether this integration improves ST quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 6 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable solution overall, but most recent SpeechLLMs can match or even outperform cascades in various settings while SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [replaced 006] Improving Music Source Separation with Diffusion and Consistency Refinement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在提升分离质量同时降低计算成本。通过结合扩散模型和一致性蒸馏，实现高效且高质量的源分离。**

- **链接: [https://arxiv.org/pdf/2412.06965](https://arxiv.org/pdf/2412.06965)**

> **作者:** Tornike Karchkhadze; Mohammad Rasool Izadi; Shuo Zhang; Shlomo Dubnov
>
> **摘要:** In this work, we propose an approach to music source separation that uses a generative diffusion model as a last-stage refinement on top of a deterministic separator, progressively enhancing the separated sources through iterative denoising. While the diffusion refinement yields measurable quality gains, it requires iterative steps at inference, increasing computational cost. To speed up the inference process, we apply consistency distillation, reducing inference to a single step while maintaining quality; with two or more steps, the distilled model even surpasses the diffusion-based approach. Crucially, our method is architecture-agnostic: we demonstrate state-of-the-art results when applied to both a custom U-Net-based separator on Slakh2100 and the state-of-the-art BS-RoFormer model on MUSDB18, showing that the refinement generalizes across backbone architectures. Sound examples are available at: this https URL.
>
---
#### [replaced 007] Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文提出Audio-Omni，解决音频生成与编辑的统一框架问题。整合多模态理解，采用冻结模型与可训练模型结合，构建大规模数据集，实现跨领域高性能音频处理。**

- **链接: [https://arxiv.org/pdf/2604.10708](https://arxiv.org/pdf/2604.10708)**

> **作者:** Zeyue Tian; Binxin Yang; Zhaoyang Liu; Jiexuan Zhang; Ruibin Yuan; Hubery Yin; Qifeng Chen; Chen Li; Jing Lyu; Wei Xue; Yike Guo
>
> **摘要:** Recent progress in multimodal models has spurred rapid advances in audio understanding, generation, and editing. However, these capabilities are typically addressed by specialized models, leaving the development of a truly unified framework that can seamlessly integrate all three tasks underexplored. While some pioneering works have explored unifying audio understanding and generation, they often remain confined to specific domains. To address this, we introduce Audio-Omni, the first end-to-end framework to unify generation and editing across general sound, music, and speech domains, with integrated multi-modal understanding capabilities. Our architecture synergizes a frozen Multimodal Large Language Model for high-level reasoning with a trainable Diffusion Transformer for high-fidelity synthesis. To overcome the critical data scarcity in audio editing, we construct AudioEdit, a new large-scale dataset comprising over one million meticulously curated editing pairs. Extensive experiments demonstrate that Audio-Omni achieves state-of-the-art performance across a suite of benchmarks, outperforming prior unified approaches while achieving performance on par with or superior to specialized expert models. Beyond its core capabilities, Audio-Omni exhibits remarkable inherited capabilities, including knowledge-augmented reasoning generation, in-context generation, and zero-shot cross-lingual control for audio generation, highlighting a promising direction toward universal generative audio intelligence. The code, model, and dataset will be publicly released on this https URL.
>
---
#### [replaced 008] Learning Filters in Feedback Delay Networks from Noisy Room Impulse Responses
- **分类: eess.AS**

- **简介: 该论文属于音频信号处理任务，解决反馈延迟网络中递归衰减滤波器在噪声环境下的优化问题。通过建模噪声，提升滤波器估计的准确性。**

- **链接: [https://arxiv.org/pdf/2512.16318](https://arxiv.org/pdf/2512.16318)**

> **作者:** Gloria Dal Santo; Karolina Prawda; Sebastian J. Schlecht; Vesa Välimäki
>
> **备注:** Submitted to the Journal of Audio Engineering Society
>
> **摘要:** Recursion is a fundamental concept in the design of filters and audio systems. In particular, artificial reverberation systems that use delay networks depend on recursive paths to control both echo density and the decay rate of modal components. The differentiable digital signal processing framework has shown promise in automatically tuning recursive and non-recursive elements using gradient-based optimization with perceptually or physically motivated loss functions, such as energy decay or spectrogram differences. These representations are highly sensitive to model mismatches, which can lead to spurious loss minima. In particular, discrepancies in background noise can result in inaccurate attenuation estimates. This paper addresses the problem of tuning recursive attenuation filters of a feedback delay network when targets are noisy. We analyze the loss profile associated with different optimization objectives and propose a method that explicitly models noise, improving the accuracy of the estimated attenuation filters under low signal-to-noise conditions. We demonstrate the effectiveness of the proposed approach through statistical analysis on both synthetic and real target data. Furthermore, we identify the sensitivity of attenuation filter parameters tuning to perturbations in frequency-independent parameters. These findings provide practical guidelines for more robust and reproducible gradient-based optimization of feedback delay networks.
>
---
#### [replaced 009] Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models
- **分类: eess.AS**

- **简介: 该论文属于语音对话系统任务，旨在解决重叠语音处理问题。提出Full-Duplex-Bench v1.5基准，评估模型在四种重叠场景下的表现。**

- **链接: [https://arxiv.org/pdf/2507.23159](https://arxiv.org/pdf/2507.23159)**

> **作者:** Guan-Ting Lin; Shih-Yun Shan Kuan; Qirui Wang; Jiachen Lian; Tingle Li; Shinji Watanabe; Hung-yi Lee
>
> **备注:** Accepted by ICASSP 2026. Code and Data at this https URL
>
> **摘要:** Full-duplex spoken dialogue systems promise to transform human-machine interaction from a rigid, turn-based protocol into a fluid, natural conversation. However, the central challenge to realizing this vision, managing overlapping speech, remains critically under-evaluated. We introduce Full-Duplex-Bench v1.5, the first fully automated benchmark designed to systematically probe how models behave during speech overlap. The benchmark simulates four representative overlap scenarios: user interruption, user backchannel, talking to others, and background speech. Our framework, compatible with open-source and commercial API-based models, provides a comprehensive suite of metrics analyzing categorical dialogue behaviors, stop and response latency, and prosodic adaptation. Benchmarking five state-of-the-art agents reveals two divergent strategies: a responsive approach prioritizing rapid response to user input, and a floor-holding approach that preserves conversational flow by filtering overlapping events. Our open-source framework enables practitioners to accelerate the development of robust full-duplex systems by providing the tools for reproducible evaluation.
>
---
#### [replaced 010] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于音频语言模型评估任务，旨在解决现有评估体系碎片化问题，提出四维系统分类，梳理研究现状并指明未来方向。**

- **链接: [https://arxiv.org/pdf/2505.15957](https://arxiv.org/pdf/2505.15957)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** EMNLP 2025 (Main). Project Website: this https URL
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [replaced 011] Game-Time: Evaluating Temporal Dynamics in Spoken Language Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决口语模型在时间动态上的不足。提出Game-Time基准，评估模型在时间感知和同步响应方面的能力。**

- **链接: [https://arxiv.org/pdf/2509.26388](https://arxiv.org/pdf/2509.26388)**

> **作者:** Kai-Wei Chang; En-Pei Hu; Chun-Yi Kuan; Wenze Ren; Wei-Chih Chen; Guan-Ting Lin; Yu Tsao; Shao-Hua Sun; Hung-yi Lee; James Glass
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Conversational Spoken Language Models (SLMs) are emerging as a promising paradigm for real-time speech interaction. However, their capacity of temporal dynamics, including the ability to manage timing, tempo and simultaneous speaking, remains a critical and unevaluated challenge for conversational fluency. To address this gap, we introduce the Game-Time Benchmark, a framework to systematically assess these temporal capabilities. Inspired by how humans learn a language through language activities, Game-Time consists of basic instruction-following tasks and advanced tasks with temporal constraints, such as tempo adherence and synchronized responses. Our evaluation of diverse SLM architectures reveals a clear performance disparity: while state-of-the-art models handle basic tasks well, many contemporary systems still struggle with fundamental instruction-following. More critically, nearly all models degrade substantially under temporal constraints, exposing persistent weaknesses in time awareness and full-duplex interaction. The Game-Time Benchmark provides a foundation for guiding future research toward more temporally-aware conversational AI. Demos and datasets are available on our project website this https URL.
>
---
#### [replaced 012] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，解决传统方法依赖配对数据及无法进行细粒度优化的问题。提出TKTO方法，无需配对数据，直接优化token级，提升日语TTS准确率39%。**

- **链接: [https://arxiv.org/pdf/2510.05799](https://arxiv.org/pdf/2510.05799)**

> **作者:** Rikuto Kotoge; Yuichi Sasaki
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens.
>
---
#### [replaced 013] Comparison of sEMG Encoding Accuracy Across Speech Modes Using Articulatory and Phoneme Features
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音编码分析任务，比较SPARC与音素特征在不同说话模式下对sEMG的预测效果，验证SPARC的优越性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.18920](https://arxiv.org/pdf/2604.18920)**

> **作者:** Chenqian Le; Ruisi Li; Beatrice Fumagalli; Yasamin Esmaeili; Xupeng Chen; Amirhossein Khalilian-Gourtani; Tianyu He; Adeen Flinker; Yao Wang
>
> **摘要:** We test whether Speech Articulatory Coding (SPARC) features can linearly predict surface electromyography (sEMG) envelopes across aloud, mimed, and subvocal speech in twenty-four subjects. Using elastic-net multivariate temporal response function (mTRF) with sentence-level cross-validation, SPARC yields higher prediction accuracy than phoneme one-hot representations on nearly all electrodes and in all speech modes. Aloud and mimed speech perform comparably, and subvocal speech remains above chance, indicating detectable articulatory activity. Variance partitioning shows a substantial unique contribution from SPARC and a minimal unique contribution from phoneme features. mTRF weight patterns reveal anatomically interpretable relationships between electrode sites and articulatory movements that remain consistent across modes. This study focuses on representation/encoding analysis (not end-to-end decoding) and supports SPARC as a robust and interpretable intermediate target for sEMG-based silent-speech modeling.
>
---
#### [replaced 014] FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统中的说话人切换检测任务，旨在解决实时全双工通信中的低延迟与鲁棒性问题。提出FastTurn框架，结合声学与语义线索，提升检测准确性与响应速度。**

- **链接: [https://arxiv.org/pdf/2604.01897](https://arxiv.org/pdf/2604.01897)**

> **作者:** Chengyou Wang; Hongfei Xue; Chunjiang He; Jingbin Hu; Shuiyuan Wang; Bo Wu; Yuyu Ji; Jimeng Zheng; Ruofei Chen; Zhou Zhu; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
>
---
#### [replaced 015] MAGIC-TTS: Fine-Grained Controllable Speech Synthesis with Explicit Local Duration and Pause Control
- **分类: cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决现有系统缺乏细粒度时间控制的问题。提出MAGIC-TTS，实现词元级时长和停顿的显式控制，提升合成质量与可编辑性。**

- **链接: [https://arxiv.org/pdf/2604.21164](https://arxiv.org/pdf/2604.21164)**

> **作者:** Jialong Mai; Xiaofen Xing; Xiangmin Xu
>
> **备注:** Release MAGIC-TTS code, pretrained models, and demo: this https URL, this https URL, this https URL
>
> **摘要:** Fine-grained local timing control is still absent from modern text-to-speech systems: existing approaches typically provide only utterance-level duration or global speaking-rate control, while precise token-level timing manipulation remains unavailable. To the best of our knowledge, MAGIC-TTS is the first TTS model with explicit local timing control over token-level content duration and pause. MAGIC-TTS is enabled by explicit token-level duration conditioning, carefully prepared high-confidence duration supervision, and training mechanisms that correct zero-value bias and make the model robust to missing local controls. On our timing-control benchmark, MAGIC-TTS substantially improves token-level duration and pause following over spontaneous synthesis. Even when no timing control is provided, MAGIC-TTS maintains natural high-quality synthesis. We further evaluate practical local editing with a scenario-based benchmark covering navigation guidance, guided reading, and accessibility-oriented code reading. In this setting, MAGIC-TTS realizes a reproducible uniform-timing baseline and then moves the edited regions toward the requested local targets with low mean bias. These results show that explicit fine-grained controllability can be implemented effectively in a high-quality TTS system and can support realistic local timing-editing applications.
>
---
#### [replaced 016] Diagnostic-Driven Layer-Wise Compensation for Post-Training Quantization of Encoder-Decoder ASR Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，解决量化后模型精度下降问题。针对编码器-解码器模型的层间敏感性差异，提出FADE框架，通过自适应补偿系数提升量化效果。**

- **链接: [https://arxiv.org/pdf/2601.02455](https://arxiv.org/pdf/2601.02455)**

> **作者:** Xinyu Wang; Ziyu Zhao; Yajie Luo; Yihong Wu; Liheng Ma; Jingrui Tian; Lei Ding; Xiao-Wen Chang; Peng Lu
>
> **备注:** 9 pages, 4 figures, 3 tables
>
> **摘要:** Deploying Automatic Speech Recognition (ASR) models on memory-constrained edge devices requires aggressive low-bit weight quantization. Layer-wise post-training quantization is practical and effective, but it suffers from cross-layer error accumulation. Existing compensation methods typically use a single global strength for all layers, which is ill-suited to encoder-decoder ASR models whose acoustic encoder and linguistic decoder exhibit markedly different sensitivities to quantization noise. We propose FADE, a diagnostic-driven framework that assigns each layer an adaptive compensation coefficient by combining two complementary signals: an intrinsic vulnerability score from weight geometry and a calibration reliability score from the data-driven solution. The resulting layer-wise coefficient balances local quantization fidelity against cross-layer error correction, enabling tailored compensation without retraining or hyperparameter search. Experiments on Whisper, Moonshine, and Qwen3-ASR across four benchmarks show that FADE consistently improves mean Word Error Rate over strong baselines at both 3- and 4-bit precision while substantially reducing run-to-run variance.
>
---
#### [replaced 017] DreamAudio: Customized Text-to-Audio Generation with Diffusion Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出DreamAudio，解决定制化文本到音频生成任务中的细粒度声学控制问题，通过引入参考概念实现个性化音频生成。**

- **链接: [https://arxiv.org/pdf/2509.06027](https://arxiv.org/pdf/2509.06027)**

> **作者:** Yi Yuan; Xubo Liu; Haohe Liu; Xiyuan Kang; Zhuo Chen; Yuxuan Wang; Mark D. Plumbley; Wenwu Wang
>
> **备注:** Lastest arxiv version. Accepted by IEEE/ACM Transactions on Audio, Speech, and Language Processing. Demos are available at this https URL
>
> **摘要:** With the development of large-scale diffusion-based and language-modeling-based generative models, impressive progress has been achieved in text-to-audio generation. Despite producing high-quality outputs, existing text-to-audio models mainly aim to generate semantically aligned sound and fall short of controlling fine-grained acoustic characteristics of specific sounds. As a result, users who need specific sound content may find it difficult to generate the desired audio clips. In this paper, we present DreamAudio for customized text-to-audio generation (CTTA). Specifically, we introduce a new framework that is designed to enable the model to identify auditory information from user-provided reference concepts for audio generation. Given a few reference audio samples containing personalized audio events, our system can generate new audio samples that include these specific events. In addition, two types of datasets are developed for training and testing the proposed systems. The experiments show that DreamAudio generates audio samples that are highly consistent with the customized audio features and aligned well with the input text prompts. Furthermore, DreamAudio offers comparable performance in general text-to-audio tasks. We also provide a human-involved dataset containing audio events from real-world CTTA cases as the benchmark for customized generation tasks.
>
---
#### [replaced 018] Evaluating Generalization and Robustness in Russian Anti-Spoofing: The RuASD Initiative
- **分类: cs.SD**

- **简介: 该论文属于语音防欺骗任务，旨在评估俄语语音防伪模型的泛化性和鲁棒性。研究构建了RuASD数据集，包含合成和真实语音，模拟多种干扰场景，以测试模型在不同条件下的表现。**

- **链接: [https://arxiv.org/pdf/2604.02374](https://arxiv.org/pdf/2604.02374)**

> **作者:** Ksenia Lysikova; Kirill Borodin; Grach Mkrtchian
>
> **备注:** Submitted to IEEE Access. Under review
>
> **摘要:** RuASD (Russian AntiSpoofing Dataset) is a dedicated, reproducible benchmark for Russian-language speech anti-spoofing designed to evaluate both in-domain discrimination and robustness to deployment-style distribution shifts. It combines a large spoof subset synthesized using 37 modern Russian-capable TTS and voice-cloning systems with a bona fide subset curated from multiple heterogeneous open Russian speech corpora, enabling systematic evaluation across diverse data sources. To emulate typical dissemination and channel effects in a controlled and reproducible manner, RuASD includes configurable simulations of platform and transmission distortions, including room reverberation, additive noise/music, and a range of speech-codec transcodings implemented via a unified processing chain. We benchmark a diverse set of publicly available anti-spoofing countermeasures spanning lightweight supervised architectures, graph-attention models, SSL-based detectors, and large-scale pretrained systems, and report reference results on both clean and simulated conditions to characterize robustness under realistic perturbation pipelines. The dataset is publickly available at \href{this https URL}{\underline{Hugging Face}} and \href{this https URL}{\underline{ModelScope}}.
>
---
#### [replaced 019] When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频干扰对文本推理的影响，属于多模态模型鲁棒性任务。解决无关音频降低推理准确性的问题，通过实验分析干扰因素并测试缓解策略。**

- **链接: [https://arxiv.org/pdf/2510.00626](https://arxiv.org/pdf/2510.00626)**

> **作者:** Chen-An Li; Tzu-Han Lin; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs.
>
---
#### [replaced 020] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner
- **分类: eess.AS**

- **简介: 该论文提出FDB-v2框架，用于评估全双工对话系统的多轮一致性与任务性能，解决其在复杂交互中的稳定性问题。**

- **链接: [https://arxiv.org/pdf/2510.07838](https://arxiv.org/pdf/2510.07838)**

> **作者:** Guan-Ting Lin; Shih-Yun Shan Kuan; Jiatong Shi; Kai-Wei Chang; Siddhant Arora; Shinji Watanabe; Hung-yi Lee
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** While full-duplex speech agents enable natural, low-latency interaction by speaking and listening simultaneously, their consistency and task performance in multi-turn settings remain underexplored. We introduce Full-Duplex-Bench-v2 (FDB-v2), a streaming framework that integrates with an automated examiner that enforces staged goals under two pacing setups (Fast vs. Slow). FDB-v2 covers four task families: daily, correction, entity tracking, and safety. We report turn-taking fluency, multi-turn instruction following, and task-specific competence. The framework is extensible, supporting both commercial APIs and open source models. When we test full-duplex systems with FDB-v2, they often get confused when people talk at the same time, struggle to handle corrections smoothly, and sometimes lose track of who or what is being talked about. Through an open-sourced, standardized streaming protocol and a task set, FDB-v2 makes it easy to extend to new task families, allowing the community to tailor and accelerate evaluation of multi-turn full-duplex systems.
>
---
#### [replaced 021] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于语音识别任务，解决视觉干扰问题，提出VAPO方法通过“看-听”流程提升幻灯片增强的语音识别效果。**

- **链接: [https://arxiv.org/pdf/2510.08618](https://arxiv.org/pdf/2510.08618)**

> **作者:** Rui Hu; Delai Qiu; Yining Wang; Shengping Liu; Jitao Sang
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Omni-modal large language models (OLLMs) offer a promising end-to-end solution for slide-enhanced speech recognition due to their inherent multimodal capabilities. However, we found a fundamental issue faced by OLLMs: \textit{Visual Interference}, where models show a bias towards visible text over auditory signals, causing them to hallucinate slide content that was never spoken. To address this, we propose Visually-Anchored Policy Optimization (VAPO), which aims to reshape models' inference process to follow the human-like ``Look-then-Listen'' inference chain. Specifically, we design a temporally decoupled policy: the model first extracts visual priors in a <think> block to serve as semantic anchors, then generates the transcription in an <answer> block. The policy is optimized via multi-objective reinforcement learning. Furthermore, we introduce SlideASR-Bench, a comprehensive benchmark designed to address the scarcity of entity-rich data, comprising a large-scale synthetic corpus for training and a challenging real-world test set for evaluation. We conduct extensive evaluations demonstrating that VAPO effectively eliminates visual interference and achieves state-of-the-art performance on SlideASR-Bench and public datasets, significantly reducing entity recognition errors in specialized domains.
>
---
