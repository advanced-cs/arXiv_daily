# 音频 cs.SD;  eess.AS

- **最新发布 20 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] On Every Note a Griff: Looking for a Useful Representation of Basso Continuo Performance Style
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于音乐信息检索任务，旨在解决巴洛克通奏低音即兴风格的表示问题。通过提出griff表示法，分析演奏风格差异。**

- **链接: [https://arxiv.org/pdf/2601.20478v1](https://arxiv.org/pdf/2601.20478v1)**

> **作者:** Adam Štefunko; Carlos Eduardo Cancino-Chacón; Jan Hajič
>
> **备注:** 6 pages, 5 figures, accepted to the Music Encoding Conference (MEC) 2026
>
> **摘要:** Basso continuo is a baroque improvisatory accompaniment style which involves improvising multiple parts above a given bass line in a musical score on a harpsichord or organ. Basso continuo is not merely a matter of history; moreover, it is a historically inspired living practice, and The Aligned Continuo Dataset (ACoRD) records the first sample of modern-day basso continuo playing in the symbolic domain. This dataset, containing 175 MIDI recordings of 5 basso continuo scores performed by 7 players, allows us to start observing and analyzing the variety that basso continuo improvisation brings. A recently proposed basso continuo performance-to-score alignment system provides a way of mapping improvised performance notes to score notes. In order to study aligned basso continuo performances, we need an appropriate feature representation. We propose griff, a representation inspired by historical basso continuo treatises. It enables us to encode both pitch content and structure of a basso continuo realization in a transposition-invariant way. Griffs are directly extracted from aligned basso continuo performances by grouping together performance notes aligned to the same score note in a onset-time ordered way, and they provide meaningful tokens that form a feature space in which we can analyze basso continuo performance styles. We statistically describe griffs extracted from the ACoRD dataset recordings, and show in two experiments how griffs can be used for statistical analysis of individuality of different players' basso continuo performance styles. We finally present an argument why it is desirable to preserve the structure of a basso continuo improvisation in order to conduct a refined analysis of personal performance styles of individual basso continuo practitioners, and why griffs can provide a meaningful historically informed feature space worthy of a more robust empirical validation.
>
---
#### [new 002] Decoding Speech Envelopes from Electroencephalogram with a Contrastive Pearson Correlation Coefficient Loss
- **分类: eess.AS**

- **简介: 该论文属于语音注意力解码任务，旨在提升多说话人环境下的听觉注意力识别。通过引入对比皮尔逊相关系数损失，优化模型区分关注与非关注语音包络的能力。**

- **链接: [https://arxiv.org/pdf/2601.20542v1](https://arxiv.org/pdf/2601.20542v1)**

> **作者:** Yayun Liang; Yuanming Zhang; Fei Chen; Jing Lu; Zhibin Lin
>
> **摘要:** Recent advances in reconstructing speech envelopes from Electroencephalogram (EEG) signals have enabled continuous auditory attention decoding (AAD) in multi-speaker environments. Most Deep Neural Network (DNN)-based envelope reconstruction models are trained to maximize the Pearson correlation coefficients (PCC) between the attended envelope and the reconstructed envelope (attended PCC). While the difference between the attended PCC and the unattended PCC plays an essential role in auditory attention decoding, existing methods often focus on maximizing the attended PCC. We therefore propose a contrastive PCC loss which represents the difference between the attended PCC and the unattended PCC. The proposed approach is evaluated on three public EEG AAD datasets using four DNN architectures. Across many settings, the proposed objective improves envelope separability and AAD accuracy, while also revealing dataset- and architecture-dependent failure cases.
>
---
#### [new 003] Self Voice Conversion as an Attack against Neural Audio Watermarking
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频水印安全研究，探讨自语音转换攻击对神经音频水印系统的威胁，旨在揭示其安全性漏洞并提出潜在防御方向。**

- **链接: [https://arxiv.org/pdf/2601.20432v1](https://arxiv.org/pdf/2601.20432v1)**

> **作者:** Yigitcan Özer; Wanying Ge; Zhe Zhang; Xin Wang; Junichi Yamagishi
>
> **备注:** 7 pages; 2 figures; 2 tables; accepted at IEICE, SP/SLP 2026
>
> **摘要:** Audio watermarking embeds auxiliary information into speech while maintaining speaker identity, linguistic content, and perceptual quality. Although recent advances in neural and digital signal processing-based watermarking methods have improved imperceptibility and embedding capacity, robustness is still primarily assessed against conventional distortions such as compression, additive noise, and resampling. However, the rise of deep learning-based attacks introduces novel and significant threats to watermark security. In this work, we investigate self voice conversion as a universal, content-preserving attack against audio watermarking systems. Self voice conversion remaps a speaker's voice to the same identity while altering acoustic characteristics through a voice conversion model. We demonstrate that this attack severely degrades the reliability of state-of-the-art watermarking approaches and highlight its implications for the security of modern audio watermarking techniques.
>
---
#### [new 004] RIR-Mega-Speech: A Reverberant Speech Corpus with Comprehensive Acoustic Metadata and Reproducible Evaluation
- **分类: eess.AS; cs.CL; cs.SD; eess.SP**

- **简介: 该论文提出RIR-Mega-Speech数据集，用于研究混响对语音识别的影响。旨在解决缺乏标准化、可复现的混响语音数据的问题。通过模拟房间脉冲响应生成数据，并提供详细声学元数据和复现工具。**

- **链接: [https://arxiv.org/pdf/2601.19949v1](https://arxiv.org/pdf/2601.19949v1)**

> **作者:** Mandip Goswami
>
> **摘要:** Despite decades of research on reverberant speech, comparing methods remains difficult because most corpora lack per-file acoustic annotations or provide limited documentation for reproduction. We present RIR-Mega-Speech, a corpus of approximately 117.5 hours created by convolving LibriSpeech utterances with roughly 5,000 simulated room impulse responses from the RIR-Mega collection. Every file includes RT60, direct-to-reverberant ratio (DRR), and clarity index ($C_{50}$) computed from the source RIR using clearly defined, reproducible procedures. We also provide scripts to rebuild the dataset and reproduce all evaluation results. Using Whisper small on 1,500 paired utterances, we measure 5.20% WER (95% CI: 4.69--5.78) on clean speech and 7.70% (7.04--8.35) on reverberant versions, corresponding to a paired increase of 2.50 percentage points (2.06--2.98). This represents a 48% relative degradation. WER increases monotonically with RT60 and decreases with DRR, consistent with prior perceptual studies. While the core finding that reverberation harms recognition is well established, we aim to provide the community with a standardized resource where acoustic conditions are transparent and results can be verified independently. The repository includes one-command rebuild instructions for both Windows and Linux environments.
>
---
#### [new 005] VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音语言模型的隐私保护任务，旨在解决多用户环境下模型无法区分说话者导致的隐私泄露问题。工作包括提出VoxPrivacy基准，评估模型的交互隐私能力，并通过微调提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2601.19956v1](https://arxiv.org/pdf/2601.19956v1)**

> **作者:** Yuxiang Wang; Hongyu Liu; Dekun Chen; Xueyao Zhang; Zhizheng Wu
>
> **摘要:** As Speech Language Models (SLMs) transition from personal devices to shared, multi-user environments such as smart homes, a new challenge emerges: the model is expected to distinguish between users to manage information flow appropriately. Without this capability, an SLM could reveal one user's confidential schedule to another, a privacy failure we term interactional privacy. Thus, the ability to generate speaker-aware responses becomes essential for SLM safe deployment. Current SLM benchmarks test dialogue ability but overlook speaker identity. Multi-speaker benchmarks check who said what without assessing whether SLMs adapt their responses. Privacy benchmarks focus on globally sensitive data (e.g., bank passwords) while neglecting contextual privacy-sensitive information (e.g., a user's private appointment). To address this gap, we introduce VoxPrivacy, the first benchmark designed to evaluate interactional privacy in SLMs. VoxPrivacy spans three tiers of increasing difficulty, from following direct secrecy commands to proactively protecting privacy. Our evaluation of nine SLMs on a 32-hour bilingual dataset reveals a widespread vulnerability: most open-source models perform close to random chance (around 50% accuracy) on conditional privacy decisions, while even strong closed-source systems fall short on proactive privacy inference. We further validate these findings on Real-VoxPrivacy, a human-recorded subset, confirming that failures observed on synthetic data persist in real speech. Finally, we demonstrate a viable path forward: by fine-tuning on a new 4,000-hour training set, we improve privacy-preserving abilities while maintaining robustness. To support future work, we release the VoxPrivacy benchmark, the large-scale training set, and the fine-tuned model to foster the development of safer and more context-aware SLMs.
>
---
#### [new 006] T-Mimi: A Transformer-based Mimi Decoder for Real-Time On-Phone TTS
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，解决实时TTS在移动端的延迟问题。通过将Mimi解码器改为纯Transformer结构，显著降低延迟，并发现部分层需保持全精度以保证音质。**

- **链接: [https://arxiv.org/pdf/2601.20094v1](https://arxiv.org/pdf/2601.20094v1)**

> **作者:** Haibin Wu; Bach Viet Do; Naveen Suda; Julian Chan; Madhavan C R; Gene-Ping Yang; Yi-Chiao Wu; Naoyuki Kanda; Yossef Adi; Xin Lei; Yue Liu; Florian Metze; Yuzong Liu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Neural audio codecs provide promising acoustic features for speech synthesis, with representative streaming codecs like Mimi providing high-quality acoustic features for real-time Text-to-Speech (TTS) applications. However, Mimi's decoder, which employs a hybrid transformer and convolution architecture, introduces significant latency bottlenecks on edge devices due to the the compute intensive nature of deconvolution layers which are not friendly for mobile-CPUs, such as the most representative framework XNNPACK. This paper introduces T-Mimi, a novel modification of the Mimi codec decoder that replaces its convolutional components with a purely transformer-based decoder, inspired by the TS3-Codec architecture. This change dramatically reduces on-device TTS latency from 42.1ms to just 4.4ms. Furthermore, we conduct quantization aware training and derive a crucial finding: the final two transformer layers and the concluding linear layers of the decoder, which are close to the waveform, are highly sensitive to quantization and must be preserved at full precision to maintain audio quality.
>
---
#### [new 007] Audio Deepfake Detection in the Age of Advanced Text-to-Speech models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决先进TTS模型生成的合成语音检测问题。通过评估不同TTS模型和检测框架，提出多视角检测方法以提高检测效果。**

- **链接: [https://arxiv.org/pdf/2601.20510v1](https://arxiv.org/pdf/2601.20510v1)**

> **作者:** Robin Singh; Aditya Yogesh Nair; Fabio Palumbo; Florian Barbaro; Anna Dyka; Lohith Rachakonda
>
> **备注:** This work was performed using HPC resources from GENCI-IDRIS (Grant 2025- AD011016076)
>
> **摘要:** Recent advances in Text-to-Speech (TTS) systems have substantially increased the realism of synthetic speech, raising new challenges for audio deepfake detection. This work presents a comparative evaluation of three state-of-the-art TTS models--Dia2, Maya1, and MeloTTS--representing streaming, LLM-based, and non-autoregressive architectures. A corpus of 12,000 synthetic audio samples was generated using the Daily-Dialog dataset and evaluated against four detection frameworks, including semantic, structural, and signal-level approaches. The results reveal significant variability in detector performance across generative mechanisms: models effective against one TTS architecture may fail against others, particularly LLM-based synthesis. In contrast, a multi-view detection approach combining complementary analysis levels demonstrates robust performance across all evaluated models. These findings highlight the limitations of single-paradigm detectors and emphasize the necessity of integrated detection strategies to address the evolving landscape of audio deepfake threats.
>
---
#### [new 008] LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出LTS-VoiceAgent，解决实时语音交互中的延迟与推理效率问题。通过分离聆听、思考与说话阶段，实现并行处理，提升响应速度与准确性。任务属于语音交互系统优化。**

- **链接: [https://arxiv.org/pdf/2601.19952v1](https://arxiv.org/pdf/2601.19952v1)**

> **作者:** Wenhao Zou; Yuwei Miao; Zhanyu Ma; Jun Xu; Jiuchong Gao; Jinghua Hao; Renqing He; Jingwen Xu
>
> **摘要:** Real-time voice agents face a dilemma: end-to-end models often lack deep reasoning, while cascaded pipelines incur high latency by executing ASR, LLM reasoning, and TTS strictly in sequence, unlike human conversation where listeners often start thinking before the speaker finishes. Since cascaded architectures remain the dominant choice for complex tasks, existing cascaded streaming strategies attempt to reduce this latency via mechanical segmentation (e.g., fixed chunks, VAD-based splitting) or speculative generation, but they frequently either break semantic units or waste computation on predictions that must be rolled back. To address these challenges, we propose LTS-VoiceAgent, a Listen-Think-Speak framework that explicitly separates when to think from how to reason incrementally. It features a Dynamic Semantic Trigger to detect meaningful prefixes, and a Dual-Role Stream Orchestrator that coordinates a background Thinker (for state maintenance) and a foreground Speaker (for speculative solving). This parallel design enables "thinking while speaking" without blocking responses. We also introduce a Pause-and-Repair benchmark containing natural disfluencies to stress-test streaming robustness. Experiments across VERA, Spoken-MQA, BigBenchAudio, and our benchmark show that LTS-VoiceAgent achieves a stronger accuracy-latency-efficiency trade-off than serial cascaded baselines and existing streaming strategies.
>
---
#### [new 009] Erasing Your Voice Before It's Heard: Training-free Speaker Unlearning for Zero-shot Text-to-Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决未经授权生成特定人声音的问题。提出TruS框架，在不重新训练模型的情况下，抑制目标说话人特征，保护隐私。**

- **链接: [https://arxiv.org/pdf/2601.20481v1](https://arxiv.org/pdf/2601.20481v1)**

> **作者:** Myungjin Lee; Eunji Shin; Jiyoung Lee
>
> **备注:** ICASSP'2026
>
> **摘要:** Modern zero-shot text-to-speech (TTS) models offer unprecedented expressivity but also pose serious crime risks, as they can synthesize voices of individuals who never consented. In this context, speaker unlearning aims to prevent the generation of specific speaker identities upon request. Existing approaches, reliant on retraining, are costly and limited to speakers seen in the training set. We present TruS, a training-free speaker unlearning framework that shifts the paradigm from data deletion to inference-time control. TruS steers identity-specific hidden activations to suppress target speakers while preserving other attributes (e.g., prosody and emotion). Experimental results show that TruS effectively prevents voice generation on both seen and unseen opt-out speakers, establishing a scalable safeguard for speech synthesis. The demo and code are available on http://mmai.ewha.ac.kr/trus.
>
---
#### [new 010] ASR for Affective Speech: Investigating Impact of Emotion and Speech Generative Strategy
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，研究情感语音和生成策略对ASR性能的影响，通过优化数据生成策略提升情绪语音识别效果。**

- **链接: [https://arxiv.org/pdf/2601.20319v1](https://arxiv.org/pdf/2601.20319v1)**

> **作者:** Ya-Tse Wu; Chi-Chun Lee
>
> **备注:** Accepted for publication at IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) 2025
>
> **摘要:** This work investigates how emotional speech and generative strategies affect ASR performance. We analyze speech synthesized from three emotional TTS models and find that substitution errors dominate, with emotional expressiveness varying across models. Based on these insights, we introduce two generative strategies: one using transcription correctness and another using emotional salience, to construct fine-tuning subsets. Results show consistent WER improvements on real emotional datasets without noticeable degradation on clean LibriSpeech utterances. The combined strategy achieves the strongest gains, particularly for expressive speech. These findings highlight the importance of targeted augmentation for building emotion-aware ASR systems.
>
---
#### [new 011] Mix2Morph: Learning Sound Morphing from Noisy Mixes
- **分类: cs.SD**

- **简介: 该论文提出Mix2Morph，解决无专用数据集下的声音形态转换问题，通过微调噪声混合物实现高质量声音融合。**

- **链接: [https://arxiv.org/pdf/2601.20426v1](https://arxiv.org/pdf/2601.20426v1)**

> **作者:** Annie Chu; Hugo Flores García; Oriol Nieto; Justin Salamon; Bryan Pardo; Prem Seetharaman
>
> **备注:** Accepted into ICASSP 2026
>
> **摘要:** We introduce Mix2Morph, a text-to-audio diffusion model fine-tuned to perform sound morphing without a dedicated dataset of morphs. By finetuning on noisy surrogate mixes at higher diffusion timesteps, Mix2Morph yields stable, perceptually coherent morphs that convincingly integrate qualities of both sources. We specifically target sound infusions, a practically and perceptually motivated subclass of morphing in which one sound acts as the dominant primary source, providing overall temporal and structural behavior, while a secondary sound is infused throughout, enriching its timbral and textural qualities. Objective evaluations and listening tests show that Mix2Morph outperforms prior baselines and produces high-quality sound infusions across diverse categories, representing a step toward more controllable and concept-driven tools for sound design. Sound examples are available at https://anniejchu.github.io/mix2morph .
>
---
#### [new 012] Gen-SER: When the generative model meets speech emotion recognition
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决传统方法分类效果有限的问题。通过生成模型将情感分类转化为分布迁移问题，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2601.20573v1](https://arxiv.org/pdf/2601.20573v1)**

> **作者:** Taihui Wang; Jinzheng Zhao; Rilin Chen; Tong Lei; Wenwu Wang; Dong Yu
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Speech emotion recognition (SER) is crucial in speech understanding and generation. Most approaches are based on either classification models or large language models. Different from previous methods, we propose Gen-SER, a novel approach that reformulates SER as a distribution shift problem via generative models. We propose to project discrete class labels into a continuous space, and obtain the terminal distribution via sinusoidal taxonomy encoding. The target-matching-based generative model is adopted to transform the initial distribution into the terminal distribution efficiently. The classification is achieved by calculating the similarity of the generated terminal distribution and ground truth terminal distribution. The experimental results confirm the efficacy of the proposed method, demonstrating its extensibility to various speech-understanding tasks and suggesting its potential applicability to a broader range of classification tasks.
>
---
#### [new 013] MK-SGC-SC: Multiple Kernel guided Sparse Graph Construction in Spectral Clustering for Unsupervised Speaker Diarization
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决无监督说话人分割问题。通过多核相似性度量构建稀疏图，提升谱聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.19946v1](https://arxiv.org/pdf/2601.19946v1)**

> **作者:** Nikhil Raghav; Avisek Gupta; Swagatam Das; Md Sahidullah
>
> **备注:** 5 pages
>
> **摘要:** Speaker diarization aims to segment audio recordings into regions corresponding to individual speakers. Although unsupervised speaker diarization is inherently challenging, the prospect of identifying speaker regions without pretraining or weak supervision motivates research on clustering techniques. In this work, we share the notable observation that measuring multiple kernel similarities of speaker embeddings to thereafter craft a sparse graph for spectral clustering in a principled manner is sufficient to achieve state-of-the-art performances in a fully unsupervised setting. Specifically, we consider four polynomial kernels and a degree one arccosine kernel to measure similarities in speaker embeddings, using which sparse graphs are constructed in a principled manner to emphasize local similarities. Experiments show the proposed approach excels in unsupervised speaker diarization over a variety of challenging environments in the DIHARD-III, AMI, and VoxConverse corpora. To encourage further research, our implementations are available at https://github.com/nikhilraghav29/MK-SGC-SC.
>
---
#### [new 014] Switchcodec: Adaptive residual-expert sparse quantization for high-fidelity neural audio coding
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频编码任务，旨在解决传统模型在不同音频复杂度下的压缩效率问题。提出SwitchCodec，通过动态选择专家量化器提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2601.20362v1](https://arxiv.org/pdf/2601.20362v1)**

> **作者:** Xiangbo Wang; Wenbin Jiang; Jin Wang; Yubo You; Sheng Fang; Fei Wen
>
> **备注:** 4page,3figure,Accepted by ICASSP 2026,We would like to express our sincere gratitude to Senior Fellow Jing Wang for his continuous support and assistance. He has made an indelible and significant contribution to this work
>
> **摘要:** Recent neural audio compression models often rely on residual vector quantization for high-fidelity coding, but using a fixed number of per-frame codebooks is suboptimal for the wide variability of audio content-especially for signals that are either very simple or highly complex. To address this limitation, we propose SwitchCodec, a neural audio codec based on Residual Experts Vector Quantization (REVQ). REVQ combines a shared quantizer with dynamically routed expert quantizers that are activated according to the input audio, decoupling bitrate from codebook capacity and improving compression efficiency. This design ensures full training and utilization of each quantizer. In addition, a variable-bitrate mechanism adjusts the number of active expert quantizers at inference, enabling multi-bitrate operation without retraining. Experiments demonstrate that SwitchCodec surpasses existing baselines on both objective metrics and subjective listening tests.
>
---
#### [new 015] Pianoroll-Event: A Novel Score Representation for Symbolic Music
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于符号音乐表示任务，旨在解决传统表示方法在编码效率与结构保持间的矛盾。提出Pianoroll-Event，通过事件描述钢琴卷形式，提升编码效率并保留时间与空间特性。**

- **链接: [https://arxiv.org/pdf/2601.19951v1](https://arxiv.org/pdf/2601.19951v1)**

> **作者:** Lekai Qian; Haoyu Gu; Dehan Li; Boyu Cao; Qi Liu
>
> **摘要:** Symbolic music representation is a fundamental challenge in computational musicology. While grid-based representations effectively preserve pitch-time spatial correspondence, their inherent data sparsity leads to low encoding efficiency. Discrete-event representations achieve compact encoding but fail to adequately capture structural invariance and spatial locality. To address these complementary limitations, we propose Pianoroll-Event, a novel encoding scheme that describes pianoroll representations through events, combining structural properties with encoding efficiency while maintaining temporal dependencies and local spatial patterns. Specifically, we design four complementary event types: Frame Events for temporal boundaries, Gap Events for sparse regions, Pattern Events for note patterns, and Musical Structure Events for musical metadata. Pianoroll-Event strikes an effective balance between sequence length and vocabulary size, improving encoding efficiency by 1.36\times to 7.16\times over representative discrete sequence methods. Experiments across multiple autoregressive architectures show models using our representation consistently outperform baselines in both quantitative and human evaluations.
>
---
#### [new 016] Do we really need Self-Attention for Streaming Automatic Speech Recognition?
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，探讨Transformer在流式ASR中的必要性。研究指出其计算成本高、延迟大，提出用可变形卷积替代自注意力机制，证明可提升效率且不影响性能。**

- **链接: [https://arxiv.org/pdf/2601.19960v1](https://arxiv.org/pdf/2601.19960v1)**

> **作者:** Youness Dkhissi; Valentin Vielzeuf; Elys Allesiardo; Anthony Larcher
>
> **摘要:** Transformer-based architectures are the most used architectures in many deep learning fields like Natural Language Processing, Computer Vision or Speech processing. It may encourage the direct use of Transformers in the constrained tasks, without questioning whether it will yield the same benefits as in standard tasks. Given specific constraints, it is essential to evaluate the relevance of transformer models. This work questions the suitability of transformers for specific domains. We argue that the high computational requirements and latency issues associated with these models do not align well with streaming applications. Our study promotes the search for alternative strategies to improve efficiency without sacrificing performance. In light of this observation, our paper critically examines the usefulness of transformer architecture in such constrained environments. As a first attempt, we show that the computational cost for Streaming Automatic Speech Recognition (ASR) can be reduced using deformable convolution instead of Self-Attention. Furthermore, we show that Self-Attention mechanisms can be entirely removed and not replaced, without observing significant degradation in the Word Error Rate.
>
---
#### [new 017] MiLorE-SSL: Scaling Multilingual Capabilities in Self-Supervised Models without Forgetting
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音表示学习任务，解决多语言自监督模型在新增语言时的遗忘问题。提出MiLorE-SSL框架，结合LoRA和软MoE机制，实现高效持续训练。**

- **链接: [https://arxiv.org/pdf/2601.20300v1](https://arxiv.org/pdf/2601.20300v1)**

> **作者:** Jing Xu; Minglin Wu; Xueyuan Chen; Xixin Wu; Helen Meng
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Self-supervised learning (SSL) has greatly advanced speech representation learning, but multilingual SSL models remain constrained to languages encountered during pretraining. Retraining from scratch to incorporate new languages is computationally expensive, while sequential training without migitation strategies often leads to catastrophic forgetting. To address this, we propose MiLorE-SSL, a lightweight framework that combines LoRA modules with a soft mixture-of-experts (MoE) mechanism for efficient continual multilingual training. LoRA provides efficient low-rank adaptation, while soft MoE promotes flexible expert sharing across languages, reducing cross-lingual interference. To further mitigate forgetting, we introduce limited replay data from existing languages, avoiding reliance on large historical corpora. Experiments on ML-SUPERB demonstrate that MiLorE-SSL achieves strong performance in new languages and improves the ability in existing ones with only 2.14% trainable parameters.
>
---
#### [new 018] Mind the Shift: Using Delta SSL Embeddings to Enhance Child ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于儿童自动语音识别（ASR）任务，旨在解决数据少和预训练领域不匹配的问题。通过使用delta SSL嵌入进行特征融合，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20142v1](https://arxiv.org/pdf/2601.20142v1)**

> **作者:** Zilai Wang; Natarajan Balaji Shankar; Kaiyuan Zhang; Zihan Wang; Abeer Alwan
>
> **备注:** ICASSP 2026
>
> **摘要:** Self-supervised learning (SSL) models have achieved impressive results across many speech tasks, yet child automatic speech recognition (ASR) remains challenging due to limited data and pretraining domain mismatch. Fine-tuning SSL models on child speech induces shifts in the representation space. We hypothesize that delta SSL embeddings, defined as the differences between embeddings from a finetuned model and those from its pretrained counterpart, encode task-specific information that complements finetuned features from another SSL model. We evaluate multiple fusion strategies on the MyST childrens corpus using different models. Results show that delta embedding fusion with WavLM yields up to a 10 percent relative WER reduction for HuBERT and a 4.4 percent reduction for W2V2, compared to finetuned embedding fusion. Notably, fusing WavLM with delta W2V2 embeddings achieves a WER of 9.64, setting a new state of the art among SSL models on the MyST corpus. These findings demonstrate the effectiveness of delta embeddings and highlight feature fusion as a promising direction for advancing child ASR.
>
---
#### [new 019] Improving X-Codec-2.0 for Multi-Lingual Speech: 25 Hz Latent Rate and 24 kHz Sampling
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音编码任务，旨在提升多语言语音的压缩效率和音质。通过降低隐层率至25Hz并提高采样率至24kHz，改进X-Codec-2.0模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20185v1](https://arxiv.org/pdf/2601.20185v1)**

> **作者:** Husein Zolkepli
>
> **摘要:** X-Codec-2.0 has shown strong performance in neural audio compression and multilingual speech modeling, operating at a 50 Hz latent rate and a 16 kHz sampling rate using frozen HuBERT features. While effective, this configuration limits temporal efficiency and audio fidelity. In this work, we explore a simple and effective modification by introducing additional pooling and increasing the decoder hop size. This reduces the latent rate from 50 Hz to 25 Hz and simultaneously raises the output sampling rate from 16 kHz to 24 kHz, improving efficiency and perceptual quality without altering the core architecture. Evaluated on the multilingual Common Voice 17 test set, the proposed configuration achieves a 0.29 MOS improvement over the original X-Codec-2.0 baseline based on UTMOSv2, and attains the best reported performance among all codecs operating at 25 Hz. The source code, checkpoints, and generation comparisons are released at \href{https://huggingface.co/Scicom-intl/xcodec2-25TPS-24k}{https://huggingface.co/Scicom-intl/xcodec2-25TPS-24k}.
>
---
#### [new 020] FastWhisper: Adaptive Self-knowledge Distillation for Real-time Automatic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升模型的实时性和泛化能力。通过自知识蒸馏方法改进学生模型，提出FastWhisper，在保持精度的同时显著提升推理速度。**

- **链接: [https://arxiv.org/pdf/2601.19919v1](https://arxiv.org/pdf/2601.19919v1)**

> **作者:** Junseok Lee; Nahoon Kim; Sangyong Lee; Chang-Jae Chun
>
> **摘要:** Knowledge distillation is one of the most effective methods for model compression. Previous studies have focused on the student model effectively training the predictive distribution of the teacher model. However, during training, the student model may inherit the shortcomings of the teacher model, which can lead to a decline in generalization capacity. To mitigate this issue, we propose adaptive self-knowledge distillation (ASKD), which dynamically reduces the dependence of the teacher model to improve the self-training capacity, and performs the self-knowledge distillation method to improve the generalization capacity of the student model. We further distill the Whisper model into a smaller variant, called FastWhisper. In our post-training setting, FastWhisper achieved a word error rate of 1.07% lower than the teacher model Whisper, and its relative inference time was 5 times faster.
>
---
## 更新

#### [replaced 001] Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models
- **分类: eess.AS**

- **简介: 该论文属于语音对话系统任务，解决重叠语音处理问题。提出Full-Duplex-Bench v1.5基准，评估模型在四种重叠场景下的表现。**

- **链接: [https://arxiv.org/pdf/2507.23159v3](https://arxiv.org/pdf/2507.23159v3)**

> **作者:** Guan-Ting Lin; Shih-Yun Shan Kuan; Qirui Wang; Jiachen Lian; Tingle Li; Shinji Watanabe; Hung-yi Lee
>
> **备注:** Accepted by ICASSP 2026. Code and Data at https://github.com/DanielLin94144/Full-Duplex-Bench
>
> **摘要:** Full-duplex spoken dialogue systems promise to transform human-machine interaction from a rigid, turn-based protocol into a fluid, natural conversation. However, the central challenge to realizing this vision, managing overlapping speech, remains critically under-evaluated. We introduce Full-Duplex-Bench v1.5, the first fully automated benchmark designed to systematically probe how models behave during speech overlap. The benchmark simulates four representative overlap scenarios: user interruption, user backchannel, talking to others, and background speech. Our framework, compatible with open-source and commercial API-based models, provides a comprehensive suite of metrics analyzing categorical dialogue behaviors, stop and response latency, and prosodic adaptation. Benchmarking five state-of-the-art agents reveals two divergent strategies: a responsive approach prioritizing rapid response to user input, and a floor-holding approach that preserves conversational flow by filtering overlapping events. Our open-source framework enables practitioners to accelerate the development of robust full-duplex systems by providing the tools for reproducible evaluation
>
---
#### [replaced 002] Blind Source Separation of Radar Signals in Time Domain Using Deep Learning
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 论文属于盲源分离任务，解决同一方向、相似频率雷达信号的分离问题。通过深度学习方法，从单通道接收信号中提取出原始雷达和通信信号。**

- **链接: [https://arxiv.org/pdf/2509.15603v2](https://arxiv.org/pdf/2509.15603v2)**

> **作者:** Sven Hinderer
>
> **摘要:** Identification and further analysis of radar emitters in a contested environment requires detection and separation of incoming signals. If they arrive from the same direction and at similar frequencies, deinterleaving them remains challenging. A solution to overcome this limitation becomes increasingly important with the advancement of emitter capabilities. We propose treating the problem as blind source separation in time domain and apply supervisedly trained neural networks to extract the underlying signals from the received mixture. This allows us to handle highly overlapping and also continuous wave (CW) signals from both radar and communication emitters. We make use of advancements in the field of audio source separation and extend a current state-of-the-art model with the objective of deinterleaving arbitrary radio frequency (RF) signals. Results show, that our approach is capable of separating two unknown waveforms in a given frequency band with a single channel receiver.
>
---
#### [replaced 003] Query-Based Asymmetric Modeling with Decoupled Input-Output Rates for Speech Restoration
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决输入输出速率不匹配导致的冗余计算问题。提出TF-Restormer框架，通过解耦输入输出速率实现高效语音恢复。**

- **链接: [https://arxiv.org/pdf/2509.21003v3](https://arxiv.org/pdf/2509.21003v3)**

> **作者:** Ui-Hyeop Shin; Jaehyun Ko; Woocheol Jeong; Hyung-Min Park
>
> **备注:** Preprint. Under review
>
> **摘要:** Speech restoration in real-world conditions is challenging due to compounded distortions and mismatches between input and desired output rates. Most existing systems assume a fixed and shared input-output rate, relying on external resampling that incurs redundant computation and limits generality. We address this setting by formulating speech restoration under decoupled input-output rates, and propose TF-Restormer, a query-based asymmetric modeling framework. The encoder concentrates analysis on the observed input bandwidth using a time-frequency dual-path architecture, while a lightweight decoder reconstructs missing spectral content via frequency extension queries. This design enables a single model to operate consistently across arbitrary input-output rate pairs without redundant resampling. Experiments across diverse sampling rates, degradations, and operating modes show that TF-Restormer maintains stable restoration behavior and balanced perceptual quality, including in real-time streaming scenarios. Code and demos are available at https://tf-restormer.github.io/demo.
>
---
#### [replaced 004] Adaptive Per-Channel Energy Normalization Front-end for Robust Audio Signal Processing
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频信号处理任务，旨在提升前端模块的鲁棒性。针对传统方法参数固定、适应性差的问题，提出一种基于神经控制器的自适应能量归一化方法，实现动态调整。**

- **链接: [https://arxiv.org/pdf/2510.18206v2](https://arxiv.org/pdf/2510.18206v2)**

> **作者:** Hanyu Meng; Vidhyasaharan Sethu; Eliathamby Ambikairajah; Qiquan Zhang; Haizhou Li
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** In audio signal processing, learnable front-ends have shown strong performance across diverse tasks by optimizing task-specific representation. However, their parameters remain fixed once trained, lacking flexibility during inference and limiting robustness under dynamic complex acoustic environments. In this paper, we introduce a novel adaptive paradigm for audio front-ends that replaces static parameterization with a closed-loop neural controller. Specifically, we simplify the learnable front-end LEAF architecture and integrate a neural controller for adaptive representation via dynamically tuning Per-Channel Energy Normalization. The neural controller leverages both the current and the buffered past subband energies to enable input-dependent adaptation during inference. Experimental results on multiple audio classification tasks demonstrate that the proposed adaptive front-end consistently outperforms prior fixed and learnable front-ends under both clean and complex acoustic conditions. These results highlight neural adaptability as a promising direction for the next generation of audio front-ends.
>
---
#### [replaced 005] WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection
- **分类: eess.AS; cs.CL; eess.SP**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升模型参数效率与泛化能力。通过结合提示调优与小波变换，提出WaveSP-Net架构，有效提取多分辨率特征，增强对合成痕迹的定位。**

- **链接: [https://arxiv.org/pdf/2510.05305v2](https://arxiv.org/pdf/2510.05305v2)**

> **作者:** Xi Xuan; Xuechen Liu; Wenxin Zhang; Yi-Cheng Lin; Xiaojian Lin; Tomi Kinnunen
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Modern front-end design for speech deepfake detection relies on full fine-tuning of large pre-trained models like XLSR. However, this approach is not parameter-efficient and may lead to suboptimal generalization to realistic, in-the-wild data types. To address these limitations, we introduce a new family of parameter-efficient front-ends that fuse prompt-tuning with classical signal processing transforms. These include FourierPT-XLSR, which uses the Fourier Transform, and two variants based on the Wavelet Transform: WSPT-XLSR and Partial-WSPT-XLSR. We further propose WaveSP-Net, a novel architecture combining a Partial-WSPT-XLSR front-end and a bidirectional Mamba-based back-end. This design injects multi-resolution features into the prompt embeddings, which enhances the localization of subtle synthetic artifacts without altering the frozen XLSR parameters. Experimental results demonstrate that WaveSP-Net outperforms several state-of-the-art models on two new and challenging benchmarks, Deepfake-Eval-2024 and SpoofCeleb, with low trainable parameters and notable performance gains. The code and models are available at https://github.com/xxuan-acoustics/WaveSP-Net.
>
---
#### [replaced 006] Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving
- **分类: eess.AS; cs.MM; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决VLA模型无法实时接收用户意图的问题。通过融合音频指令与视觉信息，提出EchoVLA模型，提升驾驶决策的准确性和情感适应性。**

- **链接: [https://arxiv.org/pdf/2601.12142v2](https://arxiv.org/pdf/2601.12142v2)**

> **作者:** Ziang Guo; Feng Yang; Xuefeng Zhang; Jiaqi Guo; Kun Zhao; Yixiao Zhou; Peng Lu; Zufeng Zhang; Sifa Zheng
>
> **备注:** Accepted by IV
>
> **摘要:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.
>
---
#### [replaced 007] AuditoryBench++: Can Language Models Understand Auditory Knowledge without Hearing?
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于多模态认知任务，旨在解决语言模型缺乏 auditory commonsense 的问题。提出 AuditoryBench++ 基准和 AIR-CoT 方法，提升模型在文本中理解听觉知识的能力。**

- **链接: [https://arxiv.org/pdf/2509.17641v2](https://arxiv.org/pdf/2509.17641v2)**

> **作者:** Hyunjong Ok; Suho Yoo; Hyeonjun Kim; Jaeho Lee
>
> **备注:** ICASSP 2026
>
> **摘要:** Even without directly hearing sounds, humans can effortlessly reason about auditory properties, such as pitch, loudness, or sound-source associations, drawing on auditory commonsense. In contrast, language models often lack this capability, limiting their effectiveness in multimodal interactions. As an initial step to address this gap, we present AuditoryBench++, a comprehensive benchmark for evaluating auditory knowledge and reasoning in text-only settings. The benchmark encompasses tasks that range from basic auditory comparisons to contextually grounded reasoning, enabling fine-grained analysis of how models process and integrate auditory concepts. In addition, we introduce AIR-CoT, a novel auditory imagination reasoning method that generates and integrates auditory information during inference through span detection with special tokens and knowledge injection. Extensive experiments with recent LLMs and Multimodal LLMs demonstrate that AIR-CoT generally outperforms both the off-the-shelf models and those augmented with auditory knowledge. The project page is available at https://auditorybenchpp.github.io.
>
---
#### [replaced 008] Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在解决非线性潜在空间难以操作的问题。通过数据增强方法，在不改变模型结构的情况下，使CAE具备线性特性，提升音频处理的直观性与效率。**

- **链接: [https://arxiv.org/pdf/2510.23530v2](https://arxiv.org/pdf/2510.23530v2)**

> **作者:** Bernardo Torres; Manuel Moussallam; Gabriel Meseguer-Brocal
>
> **摘要:** Audio autoencoders learn useful, compressed audio representations, but their non-linear latent spaces prevent intuitive algebraic manipulation such as mixing or scaling. We introduce a simple training methodology to induce linearity in a high-compression Consistency Autoencoder (CAE) by using data augmentation, thereby inducing homogeneity (equivariance to scalar gain) and additivity (the decoder preserves addition) without altering the model's architecture or loss function. When trained with our method, the CAE exhibits linear behavior in both the encoder and decoder while preserving reconstruction fidelity. We test the practical utility of our learned space on music source composition and separation via simple latent arithmetic. This work presents a straightforward technique for constructing structured latent spaces, enabling more intuitive and efficient audio processing.
>
---
#### [replaced 009] Structural and Statistical Audio Texture Knowledge Distillation for Environmental Sound Classification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于环境声音分类任务，旨在解决知识蒸馏中忽略低级音频纹理特征的问题，提出SSATKD框架融合结构与统计纹理特征，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2501.01921v2](https://arxiv.org/pdf/2501.01921v2)**

> **作者:** Jarin Ritu; Amirmohammad Mohammadi; Davelle Carreiro; Alexandra Van Dine; Joshua Peeples
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** While knowledge distillation has shown success in various audio tasks, its application to environmental sound classification often overlooks essential low-level audio texture features needed to capture local patterns in complex acoustic environments. To address this gap, the Structural and Statistical Audio Texture Knowledge Distillation (SSATKD) framework is proposed, which combines high-level contextual information with low-level structural and statistical audio textures extracted from intermediate layers. To evaluate its generalizability to a broad range of applications, SSATKD is tested on four diverse datasets within the environmental sound classification domain, namely two passive sonar datasets: DeepShip and Vessel Type Underwater Acoustic Data (VTUAD) and two general environmental sound datasets: Environmental Sound Classification 50 (ESC-50) and UrbanSound8K. Two teacher adaptation strategies are explored: classifier-head-only adaptation and full fine-tuning. The framework is further evaluated using various convolutional and transformer-based teacher models. Experimental results demonstrate consistent accuracy improvements across all datasets and settings, confirming the effectiveness and robustness of SSATKD in real-world sound classification tasks.
>
---
#### [replaced 010] CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition
- **分类: cs.LG; cs.CL; eess.AS**

- **简介: 该论文针对语音识别中的语言差异问题，提出CTC-DRO方法，通过优化减少不同语言组的性能差距，提升多语言ASR的公平性与整体表现。**

- **链接: [https://arxiv.org/pdf/2502.01777v3](https://arxiv.org/pdf/2502.01777v3)**

> **作者:** Martijn Bartelds; Ananjan Nandi; Moussa Koulako Bala Doumbouya; Dan Jurafsky; Tatsunori Hashimoto; Karen Livescu
>
> **摘要:** Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss not only scales with input length but also varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the diverse ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 47.1% and the average error by up to 32.9%. CTC-DRO can be applied to ASR with minimal computational costs, and, while motivated by multilingual ASR, offers the potential for reducing group disparities in other domains with similar challenges.
>
---
#### [replaced 011] EuleroDec: A Complex-Valued RVQ-VAE for Efficient and Robust Audio Coding
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出一种复值RVQ-VAE音频编码器，解决频域建模中相位信息处理不足的问题，提升编码效率与音质。**

- **链接: [https://arxiv.org/pdf/2601.17517v2](https://arxiv.org/pdf/2601.17517v2)**

> **作者:** Luca Cerovaz; Michele Mancusi; Emanuele Rodolà
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Audio codecs power discrete music generative modelling, music streaming and immersive media by shrinking PCM audio to bandwidth-friendly bit-rates. Recent works have gravitated towards processing in the spectral domain; however, spectrogram-domains typically struggle with phase modeling which is naturally complex-valued. Most frequency-domain neural codecs either disregard phase information or encode it as two separate real-valued channels, limiting spatial fidelity. This entails the need to introduce adversarial discriminators at the expense of convergence speed and training stability to compensate for the inadequate representation power of the audio signal. In this work we introduce an end-to-end complex-valued RVQ-VAE audio codec that preserves magnitude-phase coupling across the entire analysis-quantization-synthesis pipeline and removes adversarial discriminators and diffusion post-filters. Without GANs or diffusion we match or surpass much longer-trained baselines in-domain and reach SOTA out-of-domain performance. Compared to standard baselines that train for hundreds of thousands of steps, our model reducing training budget by an order of magnitude is markedly more compute-efficient while preserving high perceptual quality.
>
---
#### [replaced 012] Diffusion Timbre Transfer Via Mutual Information Guided Inpainting
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐音频风格迁移任务，旨在通过无训练的推理阶段操作实现音色转换，解决音色改变与结构保持的平衡问题。**

- **链接: [https://arxiv.org/pdf/2601.01294v2](https://arxiv.org/pdf/2601.01294v2)**

> **作者:** Ching Ho Lee; Javier Nistal; Stefan Lattner; Marco Pasini; George Fazekas
>
> **备注:** 5 pages, 2 figures, 3 tables
>
> **摘要:** We study timbre transfer as an inference-time editing problem for music audio. Starting from a strong pre-trained latent diffusion model, we introduce a lightweight procedure that requires no additional training: (i) a dimension-wise noise injection that targets latent channels most informative of instrument identity, and (ii) an early-step clamping mechanism that re-imposes the input's melodic and rhythmic structure during reverse diffusion. The method operates directly on audio latents and is compatible with text/audio conditioning (e.g., CLAP). We discuss design choices,analyze trade-offs between timbral change and structural preservation, and show that simple inference-time controls can meaningfully steer pre-trained models for style-transfer use cases.
>
---
#### [replaced 013] Confidence intervals for forced alignment boundaries using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决强制对齐边界不确定性问题。通过神经网络集成方法，生成边界置信区间，提升对齐可靠性。**

- **链接: [https://arxiv.org/pdf/2506.01256v3](https://arxiv.org/pdf/2506.01256v3)**

> **作者:** Matthew C. Kelley
>
> **备注:** revised for publication; 11 pages, 3 figures
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. Having confidence intervals provides an estimate of the uncertainty in the boundary placement, facilitating tasks like finding boundaries that should be reviewed. As a bonus, on the Buckeye and TIMIT corpora, the ensemble boundaries show a slight overall improvement over using just a single model. The confidence intervals can be emitted during the alignment process as JSON files and a main table for programmatic and statistical analysis. For familiarity, they are also output as Praat TextGrids using a point tier to represent the intervals.
>
---
#### [replaced 014] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于Talking Head Generation任务，解决扩散模型推理慢和非自回归范式限制的问题。提出REST框架，通过紧凑视频潜在空间、ID-Context Cache和ASD策略实现实时端到端生成。**

- **链接: [https://arxiv.org/pdf/2512.11229v2](https://arxiv.org/pdf/2512.11229v2)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation (THG). However, slow inference speeds and prevalent non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, a pioneering diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through a spatiotemporal variational autoencoder with a high compression ratio. Additionally, to enable semi-autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles into key-value caching for maintaining identity consistency and temporal coherence during long-term streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) strategy is proposed to mitigate error accumulation and enhance temporal consistency in streaming generation, leveraging a non-streaming teacher with an asynchronous noise schedule to supervise the streaming student. REST bridges the gap between autoregressive and diffusion-based approaches, achieving a breakthrough in efficiency for applications requiring real-time THG. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [replaced 015] Addressing Gradient Misalignment in Data-Augmented Training for Robust Speech Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决数据增强训练中梯度错位问题。通过设计双路径框架，对齐梯度方向，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.20682v2](https://arxiv.org/pdf/2509.20682v2)**

> **作者:** Duc-Tuan Truong; Tianchi Liu; Junjie Li; Ruijie Tao; Kong Aik Lee; Eng Siong Chng
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** In speech deepfake detection (SDD), data augmentation (DA) is commonly used to improve model generalization across varied speech conditions and spoofing attacks. However, during training, the backpropagated gradients from original and augmented inputs may misalign, which can result in conflicting parameter updates. These conflicts could hinder convergence and push the model toward suboptimal solutions, thereby reducing the benefits of DA. To investigate and address this issue, we design a dual-path data-augmented (DPDA) training framework with gradient alignment for SDD. In our framework, each training utterance is processed through two input paths: one using the original speech and the other with its augmented version. This design allows us to compare and align their backpropagated gradient directions to reduce optimization conflicts. Our analysis shows that approximately 25% of training iterations exhibit gradient conflicts between the original inputs and their augmented counterparts when using RawBoost augmentation. By resolving these conflicts with gradient alignment, our method accelerates convergence by reducing the number of training epochs and achieves up to an 18.69% relative reduction in Equal Error Rate on the In-the-Wild dataset compared to the baseline.
>
---
