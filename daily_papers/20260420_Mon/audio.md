# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] NVBench: A Benchmark for Speech Synthesis with Non-Verbal Vocalizations
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决NVVs生成与控制的评估问题。提出NVBench基准，涵盖多维度评价体系，评估系统生成NVVs的准确性、位置及显著性。**

- **链接: [https://arxiv.org/pdf/2604.16211](https://arxiv.org/pdf/2604.16211)**

> **作者:** Liumeng Xue; Weizhen Bian; Jiahao Pan; Wenxuan Wang; Yilin Ren; Boyi Kang; Jingbin Hu; Ziyang Ma; Shuai Wang; Xinyuan Qian; Hung-yi Lee; Yike Guo
>
> **摘要:** Non-verbal vocalizations (NVVs) like laugh, sigh, and sob are essential for human-like speech, yet standardized evaluation remains limited in jointly assessing whether systems can generate the intended NVVs, place them correctly, and keep them salient without harming speech. We present Non-verbal Vocalization Benchmark (NVBench), a bilingual (English/Chinese) benchmark that evaluates speech synthesis with NVVs. NVBench pairs a unified 45-type taxonomy with a curated bilingual dataset and introduces a multi-axis protocol that separates general speech naturalness and quality from NVV-specific controllability, placement, and salience. We benchmark 15 TTS systems using objective metrics, listening tests, and an LLM-based multi-rater evaluation. Results reveal that NVVs controllability often decouples from quality, while low-SNR oral cues and long-duration affective NVVs remain persistent bottlenecks. NVBench enables fair cross-system comparison across diverse control interfaces under a unified, standardized framework.
>
---
#### [new 002] ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于AI生成音乐检测任务，旨在解决传统方法在检测精度和参数效率上的不足。作者提出ArtifactNet，通过分析音频编码器留下的物理痕迹来识别AI生成音乐。**

- **链接: [https://arxiv.org/pdf/2604.16254](https://arxiv.org/pdf/2604.16254)**

> **作者:** Heewon Oh
>
> **备注:** 9 pages, 7 figures, 9 tables
>
> **摘要:** We present ArtifactNet, a lightweight framework that detects AI-generated music by reframing the problem as forensic physics -- extracting and analyzing the physical artifacts that neural audio codecs inevitably imprint on generated audio. A bounded-mask UNet (ArtifactUNet, 3.6M parameters) extracts codec residuals from magnitude spectrograms, which are then decomposed via HPSS into 7-channel forensic features for classification by a compact CNN (0.4M parameters; 4.0M total). We introduce ArtifactBench, a multi-generator evaluation benchmark comprising 6,183 tracks (4,383 AI from 22 generators and 1,800 real from 6 diverse sources). Each track is tagged with bench_origin for fair zero-shot evaluation. On the unseen test partition (n=2,263), ArtifactNet achieves F1 = 0.9829 with FPR = 1.49%, compared to CLAM (F1 = 0.7576, FPR = 69.26%) and SpecTTTra (F1 = 0.7713, FPR = 19.43%) evaluated under identical conditions with published checkpoints. Codec-aware training (4-way WAV/MP3/AAC/Opus augmentation) further reduces cross-codec probability drift by 83% (Delta = 0.95 -> 0.16), resolving the primary codec-invariance failure mode. These results establish forensic physics -- direct extraction of codec-level artifacts -- as a more generalizable and parameter-efficient paradigm for AI music detection than representation learning, using 49x fewer parameters than CLAM and 4.8x fewer than SpecTTTra.
>
---
#### [new 003] AST: Adaptive, Seamless, and Training-Free Precise Speech Editing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音编辑任务，解决传统方法依赖训练、影响时间一致性和质量的问题。提出AST框架，通过潜在重组和自适应引导实现无训练的精准语音编辑。**

- **链接: [https://arxiv.org/pdf/2604.16056](https://arxiv.org/pdf/2604.16056)**

> **作者:** Sihan Lv; Yechen Jin; Zhen Li; Jintao Chen; Jinshan Zhang; Ying Li; Jianwei Yin; Meng Xi
>
> **摘要:** Text-based speech editing aims to modify specific segments while preserving speaker identity and acoustic context. Existing methods rely on task-specific training, which incurs high data costs and struggles with temporal fidelity in unedited regions. Meanwhile, adapting Text-to-Speech (TTS) models often faces a trade-off between editing quality and consistency. To address these issues, we propose AST, an Adaptive, Seamless, and Training-free precise speech editing framework. Leveraging a pre-trained autoregressive TTS model, AST introduces Latent Recomposition to selectively stitch preserved source segments with newly synthesized targets. Furthermore, AST extends this latent manipulation to enable precise style editing for specific speech segments. To prevent artifacts at these edit boundaries, the framework incorporates Adaptive Weak Fact Guidance (AWFG). AWFG dynamically modulates a mel-space guidance signal, enforcing structural constraints only where necessary without disrupting the generative manifold. To fill the gap of publicly accessible benchmarks, we introduce LibriSpeech-Edit, a new and larger speech editing dataset. As existing metrics poorly evaluate temporal consistency in unedited regions, we propose Word-level Dynamic Time Warping (WDTW). Extensive experiments demonstrate that AST resolves the controllability-quality trade-off without extra training. Compared to the previous most temporally consistent baseline, AST improves consistency while reducing Word Error Rate by nearly 70%. Moreover, applying AST to a foundation TTS model reduces WDTW by 27%, achieving state-of-the-art speaker preservation and temporal fidelity.
>
---
#### [new 004] NaijaS2ST: A Multi-Accent Benchmark for Speech-to-Speech Translation in Low-Resource Nigerian Languages
- **分类: cs.SD**

- **简介: 该论文属于低资源多语种语音翻译任务，旨在解决数据稀缺问题。提出NaijaS2ST数据集，并对比不同翻译方法的效果。**

- **链接: [https://arxiv.org/pdf/2604.16287](https://arxiv.org/pdf/2604.16287)**

> **作者:** Marie Maltais; Yejin Jeon; Min Ma; Shamsuddeen Hassan Muhammad; Idris Abdulmumin; Maryam Ibrahim Mukhtar; Daud Abolade; Joel Okepefi; Johnson Sewedo; David Ifeoluwa Adelani
>
> **备注:** Preprint
>
> **摘要:** Speech translation for low-resource languages remains fundamentally limited by the scarcity of high-quality, diverse parallel speech data, a challenge that is especially pronounced in African linguistic contexts. To address this, we introduce NaijaS2ST, a parallel speech translation dataset spanning Igbo, Hausa, Yorùbá, and Nigerian Pidgin paired with English. The dataset comprises approximately 50 hours of speech per language and captures substantial variation in speakers and accents, reflecting realistic multilingual and multi-accent conditions. With NaijaS2ST, we conduct a comprehensive benchmark of cascaded, end-to-end (E2E), and AudioLLM-based approaches across bidirectional translation settings. Our results show that audio LLMs with few-shot examples are more effective for speech-to-text translation than cascaded and end-to-end methods trained on fine-tuned data. However, for speech-to-speech translation, the cascaded and audio LLM paradigms yield comparable performance, indicating that there is still considerable room for improvement in developing targeted, task-specific models for this setting. By providing both a high-quality dataset and a systematic benchmark, we hope that NaijaS2ST will serve as a strong foundation for advancing research in low-resource, multilingual speech translation.
>
---
#### [new 005] Hierarchical Codec Diffusion for Video-to-Speech Generation
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于视频到语音生成任务，旨在解决现有方法忽视语音层次结构的问题。提出HiCoDiT模型，利用语音离散编码的层次结构，提升音视频对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.15923](https://arxiv.org/pdf/2604.15923)**

> **作者:** Jiaxin Ye; Gaoxiang Cong; Chenhui Wang; Xin-Cheng Wen; Zhaoyang Li; Boyuan Cao; Hongming Shan
>
> **备注:** CVPR 2026
>
> **摘要:** Video-to-Speech (VTS) generation aims to synthesize speech from a silent video without auditory signals. However, existing VTS methods disregard the hierarchical nature of speech, which spans coarse speaker-aware semantics to fine-grained prosodic details. This oversight hinders direct alignment between visual and speech features at specific hierarchical levels during property matching. In this paper, leveraging the hierarchical structure of Residual Vector Quantization (RVQ)-based codec, we propose HiCoDiT, a novel Hierarchical Codec Diffusion Transformer that exploits the inherent hierarchy of discrete speech tokens to achieve strong audio-visual alignment. Specifically, since lower-level tokens encode coarse speaker-aware semantics and higher-level tokens capture fine-grained prosody, HiCoDiT employs low-level and high-level blocks to generate tokens at different levels. The low-level blocks condition on lip-synchronized motion and facial identity to capture speaker-aware content, while the high-level blocks use facial expression to modulate prosodic dynamics. Finally, to enable more effective coarse-to-fine conditioning, we propose a dual-scale adaptive instance layer normalization that jointly captures global vocal style through channel-wise normalization and local prosody dynamics through temporal-wise normalization. Extensive experiments demonstrate that HiCoDiT outperforms baselines in fidelity and expressiveness, highlighting the potential of discrete modelling for VTS. The code and speech demo are both available at this https URL.
>
---
#### [new 006] VoxMind: An End-to-End Agentic Spoken Dialogue System
- **分类: cs.SD**

- **简介: 该论文提出VoxMind，解决复杂对话任务中模型能力不足的问题。通过引入代理能力与工具使用，提升任务完成率，优化推理与响应机制。**

- **链接: [https://arxiv.org/pdf/2604.15710](https://arxiv.org/pdf/2604.15710)**

> **作者:** Tianle Liang; Yifu Chen; Shengpeng Ji; Yijun Chen; Zhiyang Jia; Jingyu Lu; Fan Zhuo; Xueyi Pu; Yangzhuo Li; Zhou Zhao
>
> **备注:** Accepted to ACL 2026 Main this http URL and data available at this https URL
>
> **摘要:** Recent end-to-end spoken dialogue models enable natural interaction. However, as user demands become increasingly complex, models that rely solely on conversational abilities often struggle to cope. Incorporating agentic capabilities is therefore essential: by enabling tool use, these models can extend their knowledge boundaries and better solve real-world tasks. Yet, existing research has largely concentrated on core perception and generation, with comparatively limited exploration of such tool-augmented extensions. To bridge this gap, we present VoxMind, an integrated framework designed to equip end-to-end spoken dialogue models with comprehensive agentic abilities. Leveraging our curated 470-hour AgentChat dataset, we incorporate a "Think-before-Speak" mechanism, enabling the model to internalize structured reasoning as a critical prerequisite for planning and response generation. Furthermore, to mitigate latency bottlenecks caused by large-scale tool integration, we propose a Multi-Agent Dynamic Tool Management architecture. By asynchronously delegating retrieval tasks to an auxiliary agent aligned with the main model's reasoning trajectory, this system effectively decouples inference latency from toolset size. Experimental results confirm that VoxMind achieves significant improvements in agent performance: compared with strong baselines, the task completion rate increases from 34.88% to 74.57%, outperforming Gemini-2.5-Pro on spoken agent tasks while preserving general conversational quality. The source code and associated data are publicly available at this https URL.
>
---
#### [new 007] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频-语言模型任务，解决统一解码器的时序平滑偏差问题。提出TCD方法，在推理时通过对比不同时间尺度的输出优化结果。**

- **链接: [https://arxiv.org/pdf/2604.15383](https://arxiv.org/pdf/2604.15383)**

> **作者:** Yanda Li; Yuhan Liu; Zirui Song; Yunchao Wei; Martin Takáč; Salem Lahlou
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large audio-language models (LALMs) generalize across speech, sound, and music, but unified decoders can exhibit a \emph{temporal smoothing bias}: transient acoustic cues may be underutilized in favor of temporally smooth context that is better supported by language priors, leading to less specific audio-grounded outputs. We propose \emph{Temporal Contrastive Decoding} (TCD), a training-free decoding method for unified LALMs that mitigates this effect at inference time. TCD constructs a temporally blurred slow-path view by smoothing the input waveform and re-encoding it, then contrasts next-token logits from the original and slow-path views. The contrastive signal is applied as a token-level logit update restricted to a small candidate set. A self-normalized stability score sets the blur window and update scale, and a step-wise gate based on uncertainty and audio reliance activates the update only when needed. Experiments on MMAU and AIR-Bench show consistent improvements on strong unified LALMs. We further conduct ablations and an architectural applicability study to analyze the contributions of key components and how TCD behaves across large audio-language model designs.
>
---
#### [new 008] TinyMU: A Compact Audio-Language Model for Music Understanding
- **分类: cs.SD**

- **简介: 该论文提出TinyMU，一个轻量级音频-语言模型，用于音乐理解。解决大模型计算成本高、部署难的问题，通过优化架构和数据集实现高效性能。**

- **链接: [https://arxiv.org/pdf/2604.15849](https://arxiv.org/pdf/2604.15849)**

> **作者:** Xiquan Li; Aurian Quelennec; Slim Essid
>
> **备注:** ICASSP 2026
>
> **摘要:** Music understanding and reasoning are central challenges in the Music Information Research field, with applications ranging from retrieval and recommendation to music agents and virtual assistants. Recent Large Audio-Language Models (LALMs) have shown remarkable progress in answering music-related questions by following user instructions. However, their massive scale, often billions of parameters, results in expensive training, slow inference, and limited deployability on edge devices. In this work, we present TinyMU, a lightweight (229M) Music-Language Model (MLM) that achieves performance comparable to much larger LALMs while remaining efficient and compact. To train TinyMU, we introduce MusicSkills-3.5M, a carefully curated, music-grounded question-answering dataset with 3.5M samples. Spanning multiple-choice, binary, and open-ended formats, this dataset provides fine-grained supervision across diverse musical concepts. For its architecture, TinyMU leverages MATPAC++, the SOTA self-supervised audio encoder for fine-grained feature extraction. Paired with a lightweight linear projector, it efficiently aligns audio embeddings with the language model. Through extensive evaluation, we show that TinyMU performs strongly in both basic music understanding and complex reasoning. Notably, on the MuChoMusic benchmark, it achieves 82\% of SOTA LALM's performance despite being 35x smaller, highlighting the potential of small MLMs under constrained computational budgets.
>
---
#### [new 009] Qwen3.5-Omni Technical Report
- **分类: cs.CL; eess.AS**

- **简介: 该论文介绍Qwen3.5-Omni，解决多模态理解和交互问题，通过大规模数据训练，提升音频视频处理及语音合成能力。**

- **链接: [https://arxiv.org/pdf/2604.15804](https://arxiv.org/pdf/2604.15804)**

> **作者:** Qwen Team
>
> **摘要:** In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis, often caused by encoding efficiency discrepancies between text and speech tokenizers, we introduce ARIA. ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding.
>
---
#### [new 010] Breakout-picker: Reducing false positives in deep learning-based borehole breakout characterization from acoustic image logs
- **分类: cs.CV; cs.SD; physics.geo-ph**

- **简介: 该论文属于地质图像分析任务，旨在解决深度学习在钻孔崩落识别中误报率高的问题。通过引入负样本和对称性验证，降低误报率，提升识别准确性。**

- **链接: [https://arxiv.org/pdf/2604.16011](https://arxiv.org/pdf/2604.16011)**

> **作者:** Guangyu Wang; Xiaodong Ma; Xinming Wu
>
> **摘要:** Borehole breakouts are stress-induced spalling on the borehole wall, which are identifiable in acoustic image logs as paired zones with near-symmetry azimuths, low acoustic amplitudes, and increased borehole radius. Accurate breakout characterization is crucial for in-situ stress analysis. In recent years, deep learning has been introduced to automate the time-consuming and labor-intensive breakout picking process. However, existing approaches often suffer from misclassification of non-breakout features, leading to high false positive rates. To address this limitation, this study develops a deep learning framework, termed Breakout-picker, with a specific focus on reducing false positives in automatic breakout characterization. Breakout-picker reduces false positives through two strategies. First, the training of Breakout-picker incorporates negative samples of non-breakout features, including natural fractures, keyseats, and logging artifacts. They share similar characteristics with breakouts, such as low acoustic amplitude or locally enlarged borehole radius. These negative training samples enables Breakout-picker to better discriminate true breakouts and similar non-breakout features. Second, candidate breakouts identified by Breakout-picker are further validated by azimuthal symmetry criteria, whereby detections that do not exhibit the near-symmetry characteristics of breakout azimuth are excluded. The performance of Breakout-picker is evaluated using three acoustic image log datasets from different regions. The results demonstrate that Breakout-picker outperforms other automatic methods with higher accuracy and substantially lower false positive rates. By reducing false positives, Breakout-picker enhances the reliability of automatic breakout characterization from acoustic image logs, which in turn benefits in-situ stress analysis based on borehole breakouts.
>
---
## 更新

#### [replaced 001] DASB -- Discrete Audio and Speech Benchmark
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决离散音频表示的优化问题。提出DASB基准，评估不同音频领域中的离散音频标记效果，分析其与连续特征的差距。**

- **链接: [https://arxiv.org/pdf/2406.14294](https://arxiv.org/pdf/2406.14294)**

> **作者:** Pooneh Mousavi; Jarod Duret; Darius Petermann; Artem Ploujnikov; Luca Della Libera; Anastasia Kuznetsova; Cem Subakan; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens have recently gained considerable attention for their potential to bridge audio and language processing, enabling multimodal language models that can both generate and understand audio. However, preserving key information such as phonetic content, speaker identity, and paralinguistic cues remains a major challenge. Identifying the optimal tokenizer and configuration is further complicated by inconsistent evaluation settings across existing studies. To address this, we introduce the Discrete Audio and Speech Benchmark (DASB), a comprehensive framework for benchmarking discrete audio tokens across speech, general audio, and music domains on a range of discriminative and generative tasks. Our results show that discrete representations are less robust than continuous ones and require careful tuning of factors such as model architecture, data size, learning rate, and capacity. Semantic tokens generally outperform acoustic tokens, but a gap remains between discrete tokens and continuous features, highlighting the need for further research. DASB codes, evaluation setup, and leaderboards are publicly available at this https URL.
>
---
#### [replaced 002] MMAudioSep: Taming Video-to-Audio Generative Model Towards Video/Text-Queried Sound Separation
- **分类: cs.SD; cs.CV; cs.LG; eess.AS**

- **简介: 该论文提出MMAudioSep，用于视频/文本查询的音源分离任务。通过预训练视频到音频模型提升效率，解决传统方法训练成本高的问题，并验证其在分离与生成任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2510.09065](https://arxiv.org/pdf/2510.09065)**

> **作者:** Akira Takahashi; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted to ICASSP 2026. 4 pages, 4 figures, 2 tables
>
> **摘要:** We introduce MMAudioSep, a generative model for video/text-queried sound separation that is founded on a pretrained video-to-audio model. By leveraging knowledge about the relationship between video/text and audio learned through a pretrained audio generative model, we can train the model more efficiently, i.e., the model does not need to be trained from scratch. We evaluate the performance of MMAudioSep by comparing it to existing separation models, including models based on both deterministic and generative approaches, and find it is superior to the baseline models. Furthermore, we demonstrate that even after acquiring functionality for sound separation via fine-tuning, the model retains the ability for original video-to-audio generation. This highlights the potential of foundational sound generation models to be adopted for sound-related downstream tasks. Our code is available at this https URL.
>
---
#### [replaced 003] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步检索增强知识获取，提升准确性同时保持交互性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [replaced 004] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
- **分类: eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决真实语音生成带来的安全风险。通过提出XLSR-MamBo框架，结合Mamba与注意力机制，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.02944](https://arxiv.org/pdf/2601.02944)**

> **作者:** Kwok-Ho Ng; Tingting Song; Yongdong Wu; Zhihua Xia
>
> **备注:** 11 pages, 3 figures, Accepted by ACL 2026 Findings
>
> **摘要:** Advanced speech synthesis technologies have enabled highly realistic speech generation, posing security risks that motivate research into audio deepfake detection (ADD). While state space models (SSMs) offer linear complexity, pure causal SSMs architectures often struggle with the content-based retrieval required to capture global frequency-domain artifacts. To address this, we explore the scaling properties of hybrid architectures by proposing XLSR-MamBo, a modular framework integrating an XLSR front-end with synergistic Mamba-Attention backbones. We systematically evaluate four topological designs using advanced SSM variants, Mamba, Mamba2, Hydra, and Gated DeltaNet. Experimental results demonstrate that the MamBo-3-Hydra-N3 configuration achieves competitive performance compared to other state-of-the-art systems on the ASVspoof 2021 LA, DF, and In-the-Wild benchmarks. This performance benefits from Hydra's native bidirectional modeling, which captures holistic temporal dependencies more efficiently than the heuristic dual-branch strategies employed in prior works. Furthermore, evaluations on the DFADD dataset demonstrate robust generalization to unseen diffusion- and flow-matching-based synthesis methods. Crucially, our analysis reveals that scaling backbone depth effectively mitigates the performance variance and instability observed in shallower models. These results demonstrate the hybrid framework's ability to capture artifacts in spoofed speech signals, providing an effective method for ADD.
>
---
#### [replaced 005] Histogram-based Parameter-efficient Tuning for Passive and Active Sonar Classification
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于被动与主动声纳分类任务，解决参数高效迁移学习中特征分布变化的问题，提出基于直方图的调参方法HPT，提升分类性能并接近全微调效果。**

- **链接: [https://arxiv.org/pdf/2504.15214](https://arxiv.org/pdf/2504.15214)**

> **作者:** Amirmohammad Mohammadi; Davelle Carreiro; Alexandra Van Dine; Joshua Peeples
>
> **备注:** 5 pages, 3 figures. This work has been accepted to IEEE IGARSS 2026
>
> **摘要:** Parameter-efficient transfer learning (PETL) methods adapt large artificial neural networks to downstream tasks without fine-tuning the entire model. However, existing additive methods, such as adapters, sometimes struggle to capture distributional shifts in intermediate feature embeddings. We propose a novel histogram-based parameter-efficient tuning (HPT) technique that captures the statistics of the target domain and modulates the embeddings. Experimental results on three downstream passive sonar datasets (ShipsEar, DeepShip, Vessel Type Underwater Acoustic Data (VTUAD)) demonstrate that HPT outperforms conventional adapters. Notably, HPT achieves 91.8% vs. 89.8% accuracy on VTUAD. For active sonar imagery (Watertank, Turntable), HPT is competitive with other PETL methods. Furthermore, HPT yields feature representations closer to those of fully fine-tuned models. Overall, HPT balances parameter savings and provides a distribution-aware alternative to existing adapters and shows a promising direction for transfer learning in resource-constrained environments. The code is publicly available: this https URL.
>
---
#### [replaced 006] Language Models as Semantic Teachers: Post-Training Alignment for Medical Audio Understanding
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于医疗音频理解任务，旨在解决音频模型缺乏临床语义的问题。通过引入语言模型作为语义教师，提升音频模型的诊断能力。**

- **链接: [https://arxiv.org/pdf/2512.04847](https://arxiv.org/pdf/2512.04847)**

> **作者:** Tsai-Ning Wang; Lin-Lin Chen; Neil Zeghidour; Aaqib Saeed
>
> **摘要:** Pre-trained audio models excel at detecting acoustic patterns in auscultation sounds but often fail to grasp their clinical significance, limiting their use and performance in diagnostic tasks. To bridge this gap, we introduce AcuLa (Audio-Clinical Understanding via Language Alignment), a lightweight post-training framework that instills semantic understanding into any audio encoder by aligning it with a medical language model, which acts as a "semantic teacher." To enable alignment at scale, we construct a large-scale dataset by leveraging off-the-shelf large language models to translate the rich, structured metadata accompanying existing audio recordings into coherent clinical reports. Our alignment strategy combines a representation-level contrastive objective with a self-supervised modeling, ensuring that the model learns clinical semantics while preserving fine-grained temporal cues. AcuLa achieves state-of-the-art results across 18 diverse cardio-respiratory tasks from 10 different datasets, improving the mean AUROC on classification benchmarks from 0.68 to 0.79 and, on the most challenging COVID-19 cough detection task, boosting the AUROC from 0.55 to 0.89. Our work demonstrates that this audio-language alignment transforms purely acoustic models into clinically-aware diagnostic tools, establishing a novel paradigm for enhancing physiological understanding in audio-based health monitoring.
>
---
#### [replaced 007] Discrete Token Modeling for Multi-Stem Music Source Separation with Language Models
- **分类: eess.AS**

- **简介: 该论文属于多音轨音乐源分离任务，旨在通过离散token生成实现更高质量的音频分离。工作包括设计联合编码器、解码器和语言模型的框架，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2604.09371](https://arxiv.org/pdf/2604.09371)**

> **作者:** Pengbo Lyu; Xiangyu Zhao; Chengwei Liu; Haoyin Yan; Xiaotao Liang; Hongyu Wang; Shaofei Xue
>
> **备注:** 5 pages, 2 figures, 3 tables. Submitted to INTERSPEECH 2026. Demo page: this https URL
>
> **摘要:** We propose a generative framework for multi-track music source separation (MSS) that reformulates the task as conditional discrete token generation. Unlike conventional approaches that directly estimate continuous signals in the time or frequency domain, our method combines a Conformer-based conditional encoder, a dual-path neural audio codec (HCodec), and a decoder-only language model to autoregressively generate audio tokens for four target tracks. The generated tokens are decoded back to waveforms through the codec decoder. Evaluation on the MUSDB18-HQ benchmark shows that our generative approach achieves perceptual quality approaching state-of-the-art discriminative methods, while attaining the highest NISQA score on the vocals track. Ablation studies confirm the effectiveness of the learnable Conformer encoder and the benefit of sequential cross-track generation.
>
---
#### [replaced 008] BlasBench: An Open Benchmark for Irish Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决爱尔兰语ASR评估中缺乏专用文本归一化的问题。提出BlasBench基准，包含爱尔兰语感知的归一化工具和可复现的评分系统，评估多个模型表现。**

- **链接: [https://arxiv.org/pdf/2604.10736](https://arxiv.org/pdf/2604.10736)**

> **作者:** Jyoutir Raj; John Conway
>
> **备注:** 9 pages, 4 tables, 3 appendices. Code and data: this https URL
>
> **摘要:** Existing multilingual benchmarks include Irish among dozens of languages but apply no Irish-aware text normalisation, leaving reliable and reproducible ASR comparison impossible. We introduce BlasBench, an open evaluation harness that provides a standalone Irish-aware normaliser preserving fadas, lenition, and eclipsis; a reproducible scoring harness and per-utterance predictions released for all evaluated runs. We pilot this by benchmarking 12 systems across four architecture families on Common Voice ga-IE and FLEURS ga-IE. All Whisper variants exceed 100% WER through insertion-driven hallucination. Microsoft Azure reaches 22.2% WER on Common Voice and 57.5% on FLEURS; the best open model, Omnilingual ASR 7B, reaches 30.65% and 39.09% respectively. Models fine-tuned on Common Voice degrade 33-43 points moving to FLEURS, while massively multilingual models degrade only 7-10 - a generalisation gap that single-dataset evaluation misses.
>
---
#### [replaced 009] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出MTR-DuplexBench，用于全面评估全双工语音语言模型的多轮对话能力，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2511.10262](https://arxiv.org/pdf/2511.10262)**

> **作者:** He Zhang; Wenqian Cui; Haoning Xu; Xiaohui Li; Lei Zhu; Haoli Bai; Shaohua Ma; Irwin King
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions, neglecting the complexities of multi-round communication. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. Also, existing benchmarks often focus solely on evaluating conversational features, neglecting other critical aspects. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark designed for a comprehensive multi-round evaluation of FD-SLMs. MTR-DuplexBench not only segments continuous full-duplex dialogues into discrete turns for turn-by-turn assessment but also incorporates various evaluation aspects, including conversational features, dialogue quality, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our benchmark. Code and data are available at: this https URL
>
---
