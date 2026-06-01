# 音频 cs.SD;  eess.AS

- **最新发布 23 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决巴西葡萄牙语区域口音分类中依赖可靠标签的问题。通过使用声学标签和音素对齐工具，提取更有效的方言特征。**

- **链接: [https://arxiv.org/pdf/2605.30457](https://arxiv.org/pdf/2605.30457)**

> **作者:** Pedro H. L. Leite; Pedro Benevenuto Valadares; Luiz W. P. Biscainho
>
> **备注:** This work was submitted to the XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026)
>
> **摘要:** Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels.
>
---
#### [new 002] On the Use of Dereverberation for Acoustic Feedback Cancellation
- **分类: eess.AS**

- **简介: 该论文属于声学反馈抑制任务，旨在解决公共广播系统和助听器中因声反馈导致的增益限制问题。通过证明反馈信号可视为源信号的混响版本，将反馈抑制与去混响结合，简化为单一去混响问题处理。**

- **链接: [https://arxiv.org/pdf/2605.31101](https://arxiv.org/pdf/2605.31101)**

> **作者:** Basil Liekens; Arnout Roebben; Toon van Waterschoot; Marc Moonen
>
> **备注:** Accepted for publication in proceedings of EUSIPCO 2026
>
> **摘要:** In public address systems and hearing aids, the maximally achievable amplification or gain is limited by acoustic feedback. Therefore, in order to be able to apply a higher gain, feedback cancellation methods are required. In addition, it is oftentimes also desirable to dereverberate a recorded signal, that is, remove the late reverberation component of the signal, before playing it back. In this paper, it is shown that under two mild conditions, the acoustic feedback signal can be written as a reverberant version of the source signal. Therefore, it is possible to treat the joint dereverberation and acoustic feedback cancellation problem as a dereverberation-only problem, meaning that dereverberation algorithms can be applied to the joint problem. Simulations corroborate this finding
>
---
#### [new 003] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文属于空间音频生成任务，旨在解决实时性与高质量之间的矛盾及多模态信息融合难题。提出SwanSphere框架，结合扩散Transformer和对比学习，提升空间音频生成效果。**

- **链接: [https://arxiv.org/pdf/2605.30940](https://arxiv.org/pdf/2605.30940)**

> **作者:** Ke Lei; Yu Zhang; Changhao Pan; Xueyi Pu; Wenxiang Guo; Ruiqi Li; Zhou Zhao
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Real-time and accurate spatial audio generation is pivotal for delivering an immersive experience. However, existing spatial audio synthesis technologies are often encumbered by a tradeoff between generation quality and high inference latency, as well as difficulty in capturing precise spatial information from multimodal inputs. To address these challenges, we propose SwanSphere, a unified streaming framework for high-fidelity spatial audio generation from panoramic videos and text prompts. SwanSphere mainly makes the following contributions: 1) We introduce a causal autoregressive diffusion transformer architecture that enables streaming high-quality spatial audio generation. 2) We design a Spatial Video-Audio Contrastive (SVAC) learning strategy to align the video encoder with the acoustic domain, and further employ a multi-objective online direct preference optimization (ODPO) scheme, resulting in strong spatial perception and robust multimodal spatial audio synthesis. 3) To alleviate the current scarcity of spatial audio datasets, we also develop an automated annotation pipeline for generating detailed spatial captions. Experimental results demonstrate that SwanSphere achieves superior performance in both video-to-spatial and text-to-spatial audio generation tasks. Demos can be found at: this https URL.
>
---
#### [new 004] Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation
- **分类: cs.SD; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于符号音乐生成任务，旨在解决属性控制不直观的问题。通过激活调节方法，实现对音高和时长的可解释控制，提升生成音乐的可调控性。**

- **链接: [https://arxiv.org/pdf/2605.31295](https://arxiv.org/pdf/2605.31295)**

> **作者:** Ioannis Prokopiou; Pantelis Vikatos; Maximos Kaliakatsos-Papakostas; Theodoros Giannakopoulos; Themos Stafylakis
>
> **备注:** Accepted at EUSIPCO 2026 (34th European Signal Processing Conference), 5 pages, 2 figures
>
> **摘要:** Transformer-based architectures have significantly advanced the generation of complex symbolic sequences, yet a significant gap remains in achieving fine-grained, interpretable control over discrete signal attributes. This paper investigates the mechanistic interpretability of the Multitrack Music Transformer (MMT) and proposes a framework for deterministic attribute modulation without retraining to bridge this gap via inference-time activation steering. Utilizing the Difference-in-Means (DiffMean) methodology, we isolate latent directions for signal attributes, specifically Pitch and Duration, within the residual stream. We validate the Linear Representation Hypothesis in this domain, achieving high correlation between steering magnitude and attribute shift. To address the inherent feature entanglement in multi-attribute steering, we introduce a Dual Steering framework utilizing Gram-Schmidt Orthogonalization. Experimental results demonstrate that this geometric decoupling reduces conceptual interference and signal degradation compared to naive vector addition, enabling independent deterministic control even against strong autoregressive conditioning.
>
---
#### [new 005] Sound effects in media:A comparative analysis of recorded and synthetic samples in live-action and animation
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频生成任务，旨在解决合成音效与真实音效的可信度问题。通过比较合成与真实音效在不同媒体中的表现，评估其有效性并提出优化方向。**

- **链接: [https://arxiv.org/pdf/2605.31082](https://arxiv.org/pdf/2605.31082)**

> **作者:** Nelly Garcia; Joshua Reiss
>
> **备注:** ArtsIT, Interactivity and Game Creation 2024
>
> **摘要:** Creating sound for storytelling is crucial to establishing the environment in productions such as films, TV series and video games. This process often involves repeating, layering and recording real objects or using sound libraries, which can be time-consuming and repetitive. To address these challenges, procedural audio, also known as digital foley, offers a solution by allowing sound designers to quickly generate samples. Despite its efficiency, questions remain about the believability of synthetic samples compared to real ones. In our study, we compared synthetic samples generated by an online procedural engine and integrated them with both animated and live-action visuals. Our results indicate that procedural audio is highly effective and perceived as believable in drama and sci-fi scenes, particularly for sound models such as lasers, hits, air and rockets, whereas synthetic sounds weren't as believable in cartoon productions when representing everyday actions. Finally, we identified specific models that needed optimisation and highlighted audio features that needed improvement with feedback from audio professionals.
>
---
#### [new 006] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于环境感知的文本转语音任务，旨在解决语音与环境音频融合困难的问题。通过多模态扩散Transformer和领域表示对齐，提升生成语音的自然度与一致性。**

- **链接: [https://arxiv.org/pdf/2605.30965](https://arxiv.org/pdf/2605.30965)**

> **作者:** Jun-Hak Yun; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Accepted to ACL 2026 main conference. Code is available at this https URL
>
> **摘要:** Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests.
>
---
#### [new 007] Mental Damage: Caption Poisoning Attacks on Retrieval-Augmented Text-to-Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究检索增强型文本到音乐生成系统的安全问题，提出通过注入恶意音乐描述来误导生成结果，揭示了此类AI系统存在的完整性风险。**

- **链接: [https://arxiv.org/pdf/2605.30365](https://arxiv.org/pdf/2605.30365)**

> **作者:** Yizhu Wen; Shuhao Zhang; Nan Zhang; Long Cheng; Hanqing Guo
>
> **备注:** This paper was accepted by the S&P 2026 ArtSec Workshop
>
> **摘要:** Retrieval-augmented text-to-music (TTM) systems augment underspecified user prompts using captions retrieved from a music caption dataset. This design introduces an integrity dependency on the music knowledge database. We show that an attacker can poison the database by injecting a small number of crafted music captions, causing the system to retrieve malicious captions that bias prompt augmentation and steer generation away from the user's intended function, without modifying the user prompt, retriever, or generator. To achieve the music caption poisoning attack, we propose a dual-layer caption poisoning strategy that preserves high-level retrieval anchors while injecting low-level acoustic descriptors to steer prompt augmentation and downstream music generation toward an attacker-chosen target intent. In a MusicCaps knowledge database, CLAP retriever, and MusicGen pipeline, poisoned generations move substantially closer to the attacker's target, while remaining comparably aligned with the original user query. These results expose a practical integrity risk for retrieval-augmented creative AI systems. Our demo can be found at: this https URL
>
---
#### [new 008] Improving acoustic drone detection generalization through pretraining and data augmentation
- **分类: eess.AS**

- **简介: 该论文属于声学无人机检测任务，旨在提升模型在不同环境下的泛化能力。通过预训练和数据增强提高检测效果，有效区分无人机声音与背景噪声。**

- **链接: [https://arxiv.org/pdf/2605.31329](https://arxiv.org/pdf/2605.31329)**

> **作者:** Paul M. Reuter; Mattes Ohlenbusch; Christian Rollwage
>
> **备注:** Accepted to Quiet Drones 2026
>
> **摘要:** Detecting unauthorized UAV flights is critical for surveillance, security, and airspace management. Acoustic drone detection, which relies on the distinctive propeller and motor sounds of UAVs, provides a low-cost, passive solution that requires no line of sight. A central challenge is generalization: reliably distinguishing drone signatures from ambient noise across unseen recording setups, environments, and UAV types (out-of-domain). Inspired by advances in large-scale audio pretraining, we develop a compact DNN-based detector and improve its generalization by (1) pretraining the model for broad sound-event classification before fine-tuning on diverse in-house and public drone recordings, and (2) applying on-the-fly augmentations (pitch shifting, noise mixing, microphone transfer function simulation, spectrogram augmentation) to expose the model to varied acoustic conditions. An ablation study quantifies the impact of each augmentation. For evaluation, we set target false-positive rates (FPR) aligned with real-world surveillance needs and report true-positive rates (TPR) on both in-domain data (public IDMT Berne 2022) and out-of-domain data (public AuDroK). Our results show that pretraining is the dominant factor for robust detection, yielding substantial TPR improvements over training from scratch on all benchmarks. The full augmentation chain provides additional gains on acoustically mismatched out-of-domain data, achieving the best mean TPR on the AuDroK subsets and the largest improvements on the most challenging scenarios. We further validate real-world applicability by measuring false positives on public non-drone corpora (IDMT-TRAFFIC and ESC-50), demonstrating equally low FPR on unfamiliar backgrounds. A distance-dependent analysis on IDMT Berne 2022 shows effective detection at distances up to 150 m.
>
---
#### [new 009] A Unified and Reproducible Experimentation Framework for Speech Understanding
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音理解任务，旨在解决模型评估不可比和训练结果难复现的问题。提出SURE框架，统一实验流程，提升评估与复现性。**

- **链接: [https://arxiv.org/pdf/2605.30899](https://arxiv.org/pdf/2605.30899)**

> **作者:** Jing Peng; Junhao Du; Chenghao Wang; Hanqi Li; Yi Yang; Yixuan Wang; Xiaoyu Gu; Guanyu Chen; Yucheng Wang; Jiang Li; Zhangjie Zhao; Haoran Wang; Wenming Tu; Haoyu Li; Duo Ma; Lirong Qian; Yu Xi; Wen Wen; Jiaqi Guo; Hui Zhang; Shuai Fan; Wenbin Jiang; Shuai Wang; Kai Yu
>
> **备注:** This paper is submitted to INTERSPEECH 2026
>
> **摘要:** Speech foundation models and Speech LLMs have advanced speech understanding, yet deployment-oriented model selection is hindered by non-comparable evaluations caused by mismatched post-processing, and by training results that are hard to reproduce across data scales and pipelines. We present SURE, a unified experimentation framework that standardizes prediction formats, normalization, and scoring. SURE evaluates strong systems across paradigms, from conventional pipelines to Speech LLMs, on representative tasks under realistic acoustic and linguistic stressors. Beyond evaluation, SURE introduces an agent-assisted training conversion flow that maps paper and code into versioned, runnable training pipelines under a unified protocol on matched open-data subsets. Overall, SURE improves comparability and reproducibility for deployment-oriented evaluation.
>
---
#### [new 010] AnchorSteer: Self-Discovered Concept Injection for Structure-Preserving Music Editing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐编辑任务，旨在解决语义与结构纠缠问题。提出AnchorSteer框架，通过结构锚定和自发现语义引导实现结构保真编辑。**

- **链接: [https://arxiv.org/pdf/2605.31053](https://arxiv.org/pdf/2605.31053)**

> **作者:** Chih-Heng Chang; Keng-Seng Ho; Chih-Yu Tsai; Kuan-Lin Chen; Yi-Hsuan Yang; Jian-Jiun Ding
>
> **备注:** Accepted by the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)
>
> **摘要:** Controllable music editing is to modify high-level attributes while strictly preserving rhythmic and melodic structures. However, this task is challenged by a semantic-structural entanglement: steering methods often degrade structure to achieve editing performance, while structural adaptors suppress semantic responsiveness. We propose AnchorSteer, a framework that disentangles this tension by coupling structural anchoring with self-discovered semantic steering. The proposed approach probes internal representations to extract interpretable, label-free concept vectors via a self-supervised reconstruction objective, isolating attributes without curated data. During editing, these portable, plug-and-play concept vectors are injected into diffusion hidden manifolds while a structural adaptor enforces consistency. Variants for unconditioned and conditioned injections are provided to balance robustness and semantic strength. Experiments on ZoME-Bench and subjective tests show that the proposed framework outperforms both steering-only and anchoring-only baselines, enabling significant semantic transformations with high-fidelity structural preservation.
>
---
#### [new 011] Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Chatterbox-Flash，解决零样本文本转语音任务中的质量与实时性问题。通过调整预训练模型，实现块扩散解码，提升合成质量并支持流式推理。**

- **链接: [https://arxiv.org/pdf/2605.30748](https://arxiv.org/pdf/2605.30748)**

> **作者:** Deokjin Seo; Gangin Park; Kihyun Nam
>
> **备注:** 8 pages, 4 figures, 9 tables
>
> **摘要:** We present Chatterbox-Flash, a zero-shot text-to-speech model obtained by fine-tuning a pretrained autoregressive TTS decoder into a block-diffusion decoder, enabling parallel token generation within each block while retaining block-by-block streaming. We find that naively transferring mainstream block-diffusion decoding to discrete speech tokens degrades quality, as a long-tail token distribution biases parallel position selection toward a few high-frequency tokens. To mitigate this without architectural modification, we introduce two inference-time techniques: prior-calibrated scoring, which subtracts the block-level marginal token distribution, and an early-decoding schedule, which adaptively terminates iteration based on calibrated confidence. On standard zero-shot TTS benchmarks, Chatterbox-Flash attains high-fidelity synthesis comparable to strong autoregressive and non-autoregressive baselines, while supporting streaming inference with time-to-first-packet on par with streaming AR systems and substantially lower real-time factor. Code and audio samples are available at this https URL.
>
---
#### [new 012] MindVoice: Reconstructing Intelligible Speech from Non-invasive Neural Signals with Pretrained Priors
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音重建任务，旨在从非侵入性神经信号中恢复可理解的语音。针对信号噪声大、信息不全的问题，提出MindVoice框架，通过预训练模型提升重建效果。**

- **链接: [https://arxiv.org/pdf/2605.31173](https://arxiv.org/pdf/2605.31173)**

> **作者:** Guangyin Bao; Taiping Zeng; Jianfeng Feng; Xiangyang Xue
>
> **摘要:** Reconstructing continuous speech from non-invasive neural recordings is a fundamental problem for probing human auditory perception and building safe, scalable speech brain-computer interfaces. Despite recent progress, intelligible reconstruction remains elusive, as non-invasive recordings are inherently noisy, spatially blurred, and only partially preserve information about perceived speech. Existing methods directly map neural activity to entangled speech representations before synthesizing waveforms with neural vocoders, resulting in spectral-similar but unintelligible results. To overcome these limitations, we introduce MindVoice, a neuro-to-speech reconstruction framework that uses pretrained models to compensate for the incomplete semantic and acoustic information in neural recordings. MindVoice disentangles reconstruction into two complementary pathways: one recovers high-level semantic content, while the other estimates fine-grained acoustic attributes. These inferred representations are then fused with powerful speech generation models and in-context voice cloning to synthesize natural and intelligible utterances. Extensive experiments on EEG and MEG demonstrate that MindVoice substantially outperforms existing methods on various metrics. These results show that pretrained priors provide a principled way to bridge the gap between noisy neural recordings and natural speech, highlighting a promising attempt for auditory neuroscience research and non-invasive speech brain-computer interfaces.
>
---
#### [new 013] 3DAE: Binaural Quality Assessment for Audio Novel View Synthesis with Spatial Maps and Benchmark
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频新视角合成任务，旨在解决传统评估方法无法定位失败原因的问题。提出3DAE框架，通过空间图分析音频错误，实现更精准的模型优化。**

- **链接: [https://arxiv.org/pdf/2605.30469](https://arxiv.org/pdf/2605.30469)**

> **作者:** Jialu Xu; Yifan Zhou
>
> **摘要:** 3D audio and novel-view acoustic synthesis models are usually evaluated with global this http URL, global metrics often hide where and why binaural prediction fails. We propose a full-reference diagnostic framework that uses time-frequency audio error maps for magnitude, ILD, IPD, temporal alignment, loudness, and high-frequency failures, forming a 3D Audio Error Map (3DAE Map) for visual inspection. We frame these diagnostics into a model-agnostic benchmark, Spatial Audio Error Bench (3DAE Bench), which takes arbitrary ground-truth and predicted binaural pairs and reports the prediction quality of audio novel-view synthesis models. Experiments on ViGAS outputs over Replay-NVAS and SoundSpaces show different dominant failure modes: temporal misalignment on Replay-NVAS and ILD mismatch on SoundSpaces. Overall, the framework provides interpretable failure-mode summaries and intuitive visual maps for audio Novel-view-synthesis model development optimization.
>
---
#### [new 014] SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决多说话人对话中表达一致性与切换控制的问题。通过构建数据集并提出SwanVoice模型，实现高质量的零样本对话合成。**

- **链接: [https://arxiv.org/pdf/2605.30993](https://arxiv.org/pdf/2605.30993)**

> **作者:** Ruiqi Li; Yu Zhang; Changhao Pan; Ke Lei; Xiang Yin; Cheng Yang
>
> **备注:** Technical Report
>
> **摘要:** Zero-shot text-to-speech (TTS) has improved substantially for single-speaker synthesis, yet expressive long-form multi-speaker dialogue remains difficult. A common workaround is to synthesize each turn with a monologue TTS model and stitch the outputs together. This adds inference cost and often breaks acoustic consistency, conversational coherence, and affective continuity across turns. Recent dialogue TTS systems have begun to address this setting, but they still struggle to keep expressive coherence, controllable speaker switching, and monologue quality at the same time. We present SwanData-Speech and SwanVoice. SwanData-Speech builds monologue and dialogue corpora from in-the-wild audio, using Swan Forced Aligner for pause-aware word-level alignment and RobustMegaTTS3 for pronunciation-hard cases. Built on these data, SwanVoice is a zero-shot TTS model for 1--4 speakers, combining a 25 Hz VAE, raw-text conditioning with pause-aware symbols and pinyin substitution, and a flow-matching DiT with speaker-turn conditioning. Training starts from monologue speech, moves through mixed and real dialogue data, and then uses DiffusionNFT post-training with phone-level and speaker-similarity rewards. On SwanBench-Speech, SwanVoice obtains higher richness and hierarchy scores than all evaluated open-source baselines in both monologue and dialogue settings, while content accuracy remains the main limitation. Audio demos are available at this https URL.
>
---
#### [new 015] FiPA-SR -- FiLM-Conditioned Perceptually Informed Audio Super-Resolution
- **分类: eess.AS**

- **简介: 该论文属于音频带宽扩展任务，旨在从有限带宽信号中重建高频内容。提出FiPA-SR模型，采用GAN和FiLM层实现高效且高质量的音频超分辨率。**

- **链接: [https://arxiv.org/pdf/2605.30594](https://arxiv.org/pdf/2605.30594)**

> **作者:** Wallace Abreu; Luiz W. P. Biscainho
>
> **备注:** Submitted to the XLIV BRAZILIAN SYMPOSIUM ON TELECOMMUNICATIONS AND SIGNAL PROCESSING - SBrT 2026
>
> **摘要:** Audio bandwidth extension aims to reconstruct missing high-frequency content from bandlimited signals. This paper proposes FiPA-SR, a GAN-based perceptual architecture capable of handling different input bandwidths within a single model. Building upon the previous $\textrm{AEROMamba}_\textrm{P}$ framework, the proposed model incorporates FiLM layers to adapt the reconstruction process according to the respective bandwidth. Experiments on the MUSDB dataset show that FiPA-SR outperforms the state-of-the-art AudioSR model across 8, 20, and 32 kHz input sampling rates. Moreover, the proposed architecture uses approximately 3$\times$ less GPU memory and performs inference more than 60$\times$ faster than the diffusion-based baseline.
>
---
#### [new 016] OpenSTBench: Beyond Semantic Evaluation for Speech Translation
- **分类: eess.AS; cs.AI**

- **简介: 该论文提出OpenSTBench，解决语音翻译系统评估不统一的问题。它是一个多维评估框架，支持S2TT和S2ST系统，联合评估翻译质量、语音质量等多方面指标。**

- **链接: [https://arxiv.org/pdf/2605.30792](https://arxiv.org/pdf/2605.30792)**

> **作者:** Yanjie An; Yuxiang Zhao; Yichi Zhang; Qixi Zheng; Yujie Tu; Keqi Deng; Kai Yu; Xie Chen
>
> **备注:** Submitted to EMNLP 2026
>
> **摘要:** Speech translation systems increasingly span speech-to-text translation (S2TT), speech-to-speech translation (S2ST), offline translation, and streaming generation, producing outputs that differ in modality, speech realization, and timing behavior. Existing evaluation practices assess important aspects such as translation quality, speech quality, and temporal quality, but these aspects are often evaluated under separate protocols, making it difficult to compare heterogeneous systems comprehensively. To address this gap, we present OpenSTBench, a unified multidimensional evaluation framework that organizes heterogeneous speech translation outputs into a shared evaluation format. OpenSTBench supports both S2TT and S2ST systems in offline and streaming settings, and jointly evaluates translation quality, speech quality, speaker preservation, emotion and paralinguistic fidelity, temporal consistency, and latency. Through experiments on representative speech translation systems, we show that systems with strong translation quality can still differ substantially in speech quality, as well as in temporal quality. OpenSTBench provides a reproducible protocol for analyzing these cross-dimensional differences and supporting application-oriented comparison of speech translation systems. The code and datasets are available at this https URL.
>
---
#### [new 017] UNISON: A Unified Sound Generation and Editing Framework via Deep LLM Fusion
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出UNISON，一个统一的音频生成与编辑框架，解决多任务音频处理问题。通过深度大语言模型融合和统一架构，实现高效、紧凑的模型设计。**

- **链接: [https://arxiv.org/pdf/2605.31530](https://arxiv.org/pdf/2605.31530)**

> **作者:** Zhaoqing Li; Haoning Xu; Jingran Su; Yaofang Liu; Zhefan Rao; Huimeng Wang; Jiajun Deng; Tianzi Wang; Zengrui Jin; Rui Liu; Haoxuan Che; Xunying Liu
>
> **摘要:** We present UNISON, a latent diffusion framework that unifies speech generation, sound generation, and audio editing within a single model. A single model handles text-to-audio, text-to-speech, zero-shot speaker cloning, mixed speech-and-sound generation, scene-level audio editing, speech-in-scene editing, and timed temporal composition, all of which share a single set of weights. Our architecture features two core designs: (1) Layer-wise deep LLM fusion, which injects hidden states from uniformly sampled layers of a frozen MLLM into corresponding MM-DiT blocks via learned projections, providing depth-matched semantic conditioning that improves instruction following over single-layer baselines; and (2) a unified multi-task architecture where task identity is encoded solely by a channel-wise mask and source audio is provided through VAE-encoded channel concatenation. Training is stabilized by an online GPU-side multi-task data synthesis pipeline with task-homogeneous batching and a two-stage curriculum. With 621M--732M trainable parameters, UNISON achieves results competitive with or exceeding task-specialist models across evaluated domains, while being roughly $4\times$ smaller than comparable unified systems.
>
---
#### [new 018] Scaling Conversational Hungarian ASR: The BEA-Dialogue+ Corpus
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决匈牙利语对话数据不足的问题。通过扩展BEA-Dialogue语料库，增加训练数据量并研究模型在不同分割下的表现。**

- **链接: [https://arxiv.org/pdf/2605.31469](https://arxiv.org/pdf/2605.31469)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Katalin Mády
>
> **摘要:** Conversational automatic speech recognition in Hungarian is constrained by the limited amount of publicly available dialogue-style training data. The BEA-Dialogue corpus addresses this need, but its strictly speaker-disjoint train/dev/eval split reduces the usable material to only 85 hours. In this paper, we introduce BEA-Dialogue+, an expanded version of the corpus that relaxes the split criterion for experimenters and dialogue partners while preserving complete separation of the primary speakers. This results in 200 hours of transcribed natural conversations and enables a controlled study of the trade-off between additional training data and speaker overlap across the splits. We evaluate several Whisper- and FastConformer-based models on both corpus versions, including Serialized Output Training (SOT)-based fine-tuning for dialogue transcription. Our results show that the larger corpus is more challenging for models without fine-tuning, whereas SOT-based adaptation yields consistent improvements in WER, CER, cpWER, and cpCER. Overall, BEA-Dialogue+ provides a substantially larger yet still demanding benchmark for Hungarian dialogue ASR, and a practical resource for training and evaluating dialogue transcription systems.
>
---
#### [new 019] UniAudio-Token: Empowering Semantic Speech Tokenizers with General Audio Perception
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出UniAudio-Token，解决语义语音分词器在音频感知上的不足，通过结构化监督和内容感知机制，提升其通用音频理解能力。**

- **链接: [https://arxiv.org/pdf/2605.31521](https://arxiv.org/pdf/2605.31521)**

> **作者:** Yuhan Song; Linhao Zhang; Aiwei Liu; Chuhan Wu; Sijun Zhang; Wei Jia; Yuan Liu; Houfeng Wang; Xiao Zhou
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Semantic speech tokenizers have become a widely used interface for Audio-LLMs, owing to their compact single-codebook design and strong linguistic alignment. However, their focus on linguistic abstraction induces acoustic blindness, limiting their applicability beyond speech-centric tasks. We propose UniAudio-Token, a framework that empowers semantic tokenizers with general audio perception without compromising speech ability. Instead of altering the semantic paradigm, UniAudio-Token mitigates its information loss through two key innovations: (1) Semantic-Acoustic Primitives (SAP) provide structured supervision by decomposing audio into linguistic content, vocal attributes, and auditory-scene primitives; and (2) Semantic-Acoustic Equilibrium (SAE) introduces a content-aware gating mechanism that adaptively restores fine-grained acoustic details from shallow layers. Extensive evaluations show that UniAudio-Token learns comprehensive universal representations while preserving high-fidelity speech generation. When integrated with downstream LLMs, it outperforms all single-codebook baseline tokenizers on both understanding and generation tasks, effectively serving as a unified audio interface. We publicly release all our code, including training and inference scripts, together with the model checkpoints at this https URL.
>
---
#### [new 020] GaMi: Geometry-Agnostic Material Identification via Cross-Modal Subtractive Disentanglement
- **分类: cs.ET; cs.AI; cs.SD**

- **简介: 该论文属于材料识别任务，解决几何变化和单模态模糊问题。通过融合毫米波与声学传感，利用跨模态减法解耦方法提取材料特征，提升识别准确性。**

- **链接: [https://arxiv.org/pdf/2605.30818](https://arxiv.org/pdf/2605.30818)**

> **作者:** Zhiwei Chen; Yijie Li; Yimo Zhang; Shiyun Shao; Yichao Chen; Dian Ding; Liang Wang; Haiwei Wu; Liwei Guo; Jie Yang; Xiaosong Zhang; Yongzhao Zhang
>
> **备注:** 17 pages, 18 figures
>
> **摘要:** Non-contact material identification enables adaptive interaction for embodied intelligence yet faces challenges from geometry-induced variations (e.g., orientation, shape, distance) and single-modality ambiguities. In this paper, we present GaMi, a multimodal material identification system integrating mmWave and acoustic sensing to robustly operate under unconstrained geometric conditions. By leveraging the insight of shared geometric consistency between co-located bimodal sensors, GaMi employs an intra-sample cross-modal subtractive disentanglement framework. By semantically aligning modalities and subtracting the shared geometric context, it isolates intrinsic material features. Furthermore, GaMi incorporates inter-sample contrastive learning to correct the residual interference caused by cross-modal misalignment. Additionally, a pairing-based adaptation strategy between two modalities enables few-shot generalization across devices. Extensive evaluations on 20 materials show that GaMi achieves 95.2% accuracy, outperforming single-modality baselines across unseen geometric conditions.
>
---
#### [new 021] Escaping the Linearity Trap: Manifold Detours for Black-Box Adversarial Attacks on Singing Audio Deepfake Detection
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文属于歌唱语音深度伪造检测任务，旨在解决SSL-SVDD模型的脆弱性问题。通过构建MARS框架，实现更有效的黑盒攻击，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2605.30366](https://arxiv.org/pdf/2605.30366)**

> **作者:** Yifan Liao; Yule Liu; Zhen Sun; Zongmin Zhang; Yupeng He; Jiaheng Wei; Xinhu Zheng; Xinlei He
>
> **摘要:** Recent Singing Voice Synthesis (SVS) advances enable highly realistic but potentially malicious AI covers, making singing voice deepfake detection (SVDD) crucial. Self-Supervised Learning (SSL)-based detectors achieve state-of-the-art performance by fine-tuning speech SSL backbones to capture singing-specific spoof artifacts. Existing adversarial attacks often fail against SSL-SVDD, creating a false impression of inherent robustness. We reveal this stems from two challenges. First, at the objective level, attacks optimize cross-entropy on local surrogates, crossing surrogate-specific boundaries rather than suppressing shared spoof evidence. Second, at the method level, attacks follow the surrogate's dominant gradient direction. In SSL-SVDD, this aligns with fine-tuned artifact-sensitive directions, limiting transferability to unseen detectors - a geometric failure we term the Linearity Trap. To properly evaluate robustness, we propose MARS (Meta-Adversarial Regression of Semantics), a transfer-based black-box framework tailored to SSL-SVDD. Structurally, MARS shifts to hypothesis-evidence manipulation by constructing a natural semantic anchor from the pre-trained SSL space and an artifact anchor from the fine-tuned space. Algorithmically, MARS escapes the Linearity Trap via bi-level optimization: the inner stage induces tangential exploration, while the outer stage guides the audio toward the natural semantic manifold. Experiments on the CtrSVDD benchmark show MARS improves Attack Success Rate (ASR) in in-distribution transfer (13%), out-of-distribution transfer (10%), and cross-task evaluation (36%), highlighting the urgent need for robust SVDD systems.
>
---
#### [new 022] Audio Pirates: Black-box Audio Watermark Removal via Diffusion Priors
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于音频水印移除任务，解决水印易被移除的问题。提出DiffErase方法，在不破坏音质的前提下，通过扩散模型有效去除水印。**

- **链接: [https://arxiv.org/pdf/2605.30614](https://arxiv.org/pdf/2605.30614)**

> **作者:** Lingfeng Yao; Xincong Zhong; Chenpei Huang; Xuandong Zhao; Hanqing Guo; Aohan Li; Jiang Liu; Tomoaki Ohtsuki; Miao Pan
>
> **摘要:** With the rise of AI-generated audio, watermarking has become widely used for detecting misuse and protecting intellectual property. However, adversaries may try to remove these watermarks, making it critical to evaluate how well watermarking schemes withstand removal attacks. Existing attacks are often impractical: they either noticeably degrade perceptual quality or require access to the watermarking scheme. We propose DiffErase, a black-box watermark removal attack that assumes no knowledge of the target watermarking scheme while maintaining perceptual quality. DiffErase perturbs watermarked audio to an intermediate diffusion noise level and regenerates it using a pretrained denoising model, effectively suppressing watermark signals. Theoretical analysis and extensive experiments demonstrate that inaudible audio watermarks are highly vulnerable: across multiple audio domains, DiffErase consistently removes watermarks while preserving perceptual quality. These findings highlight the need for future audio watermarking designs to consider diffusion-based threats. Code and demos are available at this https URL.
>
---
#### [new 023] DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音翻译任务，解决长文本实时翻译中对齐信号不足的问题。提出DOA方法，利用自注意力生成对齐信号，实现无需训练的高效翻译。**

- **链接: [https://arxiv.org/pdf/2605.31432](https://arxiv.org/pdf/2605.31432)**

> **作者:** Sara Papi; Luisa Bentivogli
>
> **摘要:** Simultaneous speech-to-text translation (SimulST) generates translations while speech is still unfolding, requiring a streaming policy that decides when to read and when to write. State-of-the-art approaches rely on attention-based encoder-decoder models where cross-attention provides explicit alignment signals. In contrast, Speech Large Language Models (SpeechLLMs) are decoder-only architectures relying solely on self-attention. This raises a central question: whether decoder self-attention contains sufficiently stable alignment signals to guide the streaming policy. Moreover, existing approaches typically rely on training-based adaptations or heuristic wait-$k$ policies and have not been validated in long-form settings. To fill these gaps, we propose Decoder-Only Attention (DOA), a training-free policy that enables long-form simultaneous translation with off-the-shelf SpeechLLMs by deriving a proxy alignment from self-attention. Experiments on Phi4-Multimodal and Qwen3-Omni show that DOA provides an effective alignment signal for supporting streaming decisions, enabling low-latency long-form SimulST with quality close to offline decoding without retraining.
>
---
## 更新

#### [replaced 001] G-STAR: End-to-End Global Speaker-Tracking Attributed Recognition
- **分类: eess.AS; cs.AI; cs.HC; cs.MM; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，解决长时多说话人语音中说话人身份一致性问题。提出G-STAR框架，结合说话人跟踪与语音大模型，实现精准的时间戳说话人标注。**

- **链接: [https://arxiv.org/pdf/2603.10468](https://arxiv.org/pdf/2603.10468)**

> **作者:** Jing Peng; Ziyi Chen; Haoyu Li; Yucheng Wang; Duo Ma; Mengtian Li; Yunfan Du; Dezhu Xu; Kai Yu; Shuai Wang
>
> **备注:** submitted to Emnlp 2026
>
> **摘要:** We study timestamped speaker-attributed automatic speech recognition (SA-ASR) for long-form, multi-party speech with overlap. In this setting, chunk-wise inference must preserve meeting-level speaker identity consistency while producing time-stamped, speaker-labeled transcripts. Prior Speech-LLM systems tend to prioritize either local diarization or global labeling, lacking the ability to jointly model fine-grained temporal boundaries and robust cross-chunk identity linking. We propose G-STAR, an end-to-end framework that couples a cache-conditioned speaker-tracking module with a Speech-LLM transcription backbone. The tracker provides structured speaker cues with temporal grounding, and the LLM generates attributed text conditioned on these cues. G-STAR supports component-wise optimization and joint end-to-end training, enabling flexible learning under heterogeneous supervision and domain shift. Under chunk-wise decoding protocols, experiments on both oracle-segmented local evaluation and full-meeting global evaluation show strong speaker-attributed transcription performance.
>
---
#### [replaced 002] Targeted Speaker Poisoning Framework in Zero-Shot Text-to-Speech
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究零样本文本转语音中的说话人污染问题，旨在保护特定说话人的隐私。通过修改模型参数，在保持其他说话人生成质量的同时，阻止特定身份的语音生成。**

- **链接: [https://arxiv.org/pdf/2603.07551](https://arxiv.org/pdf/2603.07551)**

> **作者:** Thanapat Trachu; Thanathai Lertpetchpun; Sai Praneeth Karimireddy; Shrikanth Narayanan
>
> **备注:** Submitted to Interspeech2026
>
> **摘要:** Zero-shot Text-to-Speech (TTS) voice cloning poses severe privacy risks, demanding the removal of specific speaker identities from trained TTS models. Conventional machine unlearning is insufficient in this context, as zero-shot TTS can dynamically reconstruct voices from just reference prompts. We formalize this task as Speech Generation Speaker Poisoning (SGSP), in which we modify trained models to prevent the generation of specific identities while preserving utility for other speakers. We evaluate inference-time filtering and parameter-modification baselines across 1, 15, and 100 forgotten speakers. Performance is assessed through the trade-off between utility (WER) and privacy, quantified using AUC and Forget Speaker Similarity (FSSIM). We achieve strong privacy for up to 15 speakers but reveal scalability limits at 100 speakers due to increased identity overlap. Our study thus introduces a novel problem and evaluation framework toward further advances in generative voice privacy.
>
---
#### [replaced 003] Acoustic Simulation Framework for Multi-channel Replay Speech Detection
- **分类: eess.AS; cs.CR; cs.SD; eess.SP**

- **简介: 该论文属于语音安全任务，旨在解决多通道回放攻击检测问题。提出一种声学仿真框架，生成多通道数据以提升检测模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.14789](https://arxiv.org/pdf/2509.14789)**

> **作者:** Michael Neri; Tuomas Virtanen
>
> **备注:** Submitted to IEEE MMSP 2026
>
> **摘要:** Replay speech attacks pose a significant threat to voice-controlled systems, especially in smart environments where voice assistants are widely deployed. While multi-channel audio offers spatial cues that can enhance replay detection robustness, existing datasets and methods predominantly rely on single-channel recordings. Moreover, previous studies highlighted that generalization of this attack to new environments is challenging, requiring new methods for generating data encompassing various acoustic conditions. Hence, in this work we introduce an acoustic simulation framework designed to simulate multi-channel replay speech configurations using publicly available resources. Using the framework, we train the state-of-the-art multi-channel replay detector M-ALRAD and evaluate its generalisation on the ReMASC real-recording corpus without any real training data. To improve the exploitation of spatial information, we extend M-ALRAD with inter-channel phase difference features computed for adjacent microphone pairs, augmenting the beamformed representation with directional cues. Synthetic datasets will be available upon acceptance of the paper.
>
---
#### [replaced 004] BAT: Better Audio Transformer Guided by Convex Gated Probing
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频自监督学习任务，旨在解决传统微调方法依赖问题，提出CGP探针优化模型训练，推出新SOTA模型BAT。**

- **链接: [https://arxiv.org/pdf/2602.16305](https://arxiv.org/pdf/2602.16305)**

> **作者:** Houtan Ghaffari; Lukas Rauch; Christoph Scholz; Paul Devos
>
> **备注:** Accepted @ ICML26
>
> **摘要:** Probing is widely adopted in computer vision to faithfully evaluate self-supervised learning (SSL) embeddings, as finetuning may misrepresent their inherent quality. In contrast, audio SSL models still rely on finetuning because simple probing fails to unlock their full potential and alters their rankings when competing on AudioSet. Hence, a robust and efficient probing mechanism is required to guide the trajectory of audio SSL towards reliable and reproducible methods. We introduce Convex Gated Probing (CGP), a prototype-based method that significantly closes the gap between finetuning and probing in audio. CGP efficiently utilizes all frozen layers via a gating mechanism and exposes the location of latent task-relevant information. Guided by CGP as a reliable post-hoc evaluation probe, we rework the entire SSL pipeline of current best performing audio models that use legacy implementations of prior SSL methods. By refining data preprocessing, model architecture, and pretraining recipe, we introduce Better Audio Transformer (BAT), and establish new SOTA on audio benchmarks.
>
---
#### [replaced 005] Performance and Complexity Trade-off Optimization of Speech Models During Training
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音机器学习任务，旨在优化模型性能与计算复杂度的平衡。通过引入特征噪声重参数化方法，在训练中联合优化两者，无需后期剪枝。**

- **链接: [https://arxiv.org/pdf/2601.13704](https://arxiv.org/pdf/2601.13704)**

> **作者:** Esteban Gómez; Tom Backström
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In speech machine learning, neural network models are typically designed by choosing an architecture with fixed layer sizes and structure. These models are then trained to maximize performance on metrics aligned with the task's objective. While the overall architecture is usually guided by prior knowledge of the task, the sizes of individual layers are often chosen heuristically. However, this approach does not guarantee an optimal trade-off between performance and computational complexity; consequently, post hoc methods such as weight quantization or model pruning are typically employed to reduce computational cost. This occurs because stochastic gradient descent (SGD) methods can only optimize differentiable functions, while factors influencing computational complexity, such as layer sizes and floating-point operations per second (FLOP/s), are non-differentiable and require modifying the model structure during training. We propose a reparameterization technique based on feature noise injection that enables joint optimization of performance and computational complexity during training using SGD-based methods. Unlike traditional pruning methods, our approach allows the model size to be dynamically optimized for a target performance-complexity trade-off, without relying on heuristic criteria to select which weights or structures to remove. We demonstrate the effectiveness of our method through three case studies, including a synthetic example and two practical real-world applications: voice activity detection and audio anti-spoofing. The code related to our work is publicly available to encourage further research.
>
---
#### [replaced 006] Rethinking Continual Learning for Speech and Audio: A Representation-Centric Taxonomy and Open Problems
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音与音频领域的持续学习任务，旨在解决基础模型在非平稳环境中的表征保持与演化问题。提出新的分类体系，分析现有方法与模型行为的不匹配，并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2605.24863](https://arxiv.org/pdf/2605.24863)**

> **作者:** Yang Xiao; Siyi Wang; Eun-Jung Holden; Ting Dang
>
> **备注:** 4 pages, 1 figure, working in process
>
> **摘要:** Speech and audio systems operate in inherently non-stationary environments, yet continual learning (CL) research in this domain, especially in the foundation model era, remains fragmented that fail to account for the coupled, geometry-sensitive nature of acoustic representations. Modern speech foundation models operate over highly entangled, continuous representations that jointly encode linguistic, speaker, and paralinguistic factors within a shared latent space. CL is therefore fundamentally about preserving and evolving shared representation structure rather than retaining isolated task knowledge. In this work, we revisit CL for speech from a representation-centered perspective, and introduce a new taxonomy that organizes CL according to how underlying representation geometry evolves under non-stationary acoustic conditions. We further identify key mismatches between current CL assumptions and speech foundation model behavior, and finally outline a set of open challenges and future research directions.
>
---
#### [replaced 007] Beyond Hearing: Learning Task-Agnostic ExG Representations from Earphones via Physiology-Informed Tokenization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于生理信号处理任务，旨在解决ExG数据多样性不足和模型任务依赖性问题。通过收集自由生活数据并引入PiMT方法，学习通用的ExG表示，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.20853](https://arxiv.org/pdf/2510.20853)**

> **作者:** Hyungjun Yoon; Seungjoo Lee; Yu Yvonne Wu; Xiaomeng Chen; Taiting Lu; Freddy Yifei Liu; Taeckyung Lee; Hyeongheon Cha; Haochen Zhao; Gaoteng Zhao; Dongyao Chen; Cecilia Mascolo; Sung-Ju Lee; Lili Qiu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Electrophysiological (ExG) signals offer valuable insights into human physiology, yet building foundation models that generalize across everyday tasks remains challenging due to two key limitations: (i)~insufficient data diversity, as most ExG recordings are collected in controlled labs with bulky, expensive devices; and (ii)~task-specific model designs that require tailored processing (i.e., targeted frequency filters) and architectures, which limit generalization across tasks. To address these challenges, we introduce an approach for scalable, task-agnostic ExG monitoring in the wild. We collected 50 hours of unobtrusive free-living ExG data with an earphone-based hardware prototype to narrow the data diversity gap. At the core of our approach is Physiology-informed Multi-band Tokenization (PiMT), which decomposes ExG signals into 12 physiology-informed tokens, followed by a reconstruction task to learn robust representations. This enables adaptive feature recognition across the full frequency spectrum while capturing task-relevant information. Experiments on our new DailySense dataset, the first to enable ExG-based analysis across five human senses, together with four public ExG benchmarks, demonstrate that PiMT consistently outperforms state-of-the-art methods across diverse tasks.
>
---
#### [replaced 008] Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于多标签音频分类任务，旨在解决全局池化导致的嵌入质量失真问题。提出二值原型探测方法，提升探针性能，验证其作为高效评估范式的可行性。**

- **链接: [https://arxiv.org/pdf/2509.24901](https://arxiv.org/pdf/2509.24901)**

> **作者:** Lukas Rauch; René Heinrich; Houtan Ghaffari; Lukas Miklautz; Ilyass Moummad; Bernhard Sick; Christoph Scholz
>
> **备注:** Accepted @ ICLR26
>
> **摘要:** Although probing frozen models has become a standard evaluation paradigm, self-supervised learning in audio defaults to fine-tuning when pursuing state-of-the-art on AudioSet. A key reason is that global pooling creates an information bottleneck causing linear probes to misrepresent the embedding quality: The $\texttt{cls}$-token discards crucial token information about dispersed, localized events in audio. This weakness is rooted in the mismatch between the pretraining objective (globally) and the downstream task (localized). Across a comprehensive benchmark of 13 datasets and 6 spectrogram-based encoders, we investigate the global pooling bottleneck. We introduce binarized prototypical probes: a lightweight and simple pooling method that learns prototypes to perform class-wise information aggregation. Despite its simplicity, our method notably outperforms linear and attentive probing. Our work establishes probing as a competitive and efficient paradigm for evaluating audio SSL models, challenging the reliance on costly fine-tuning.
>
---
