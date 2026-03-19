# 音频 cs.SD;  eess.AS

- **最新发布 20 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Learnable Pulse Accumulation for On-Device Speech Recognition: How Much Attention Do You Need?
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对边缘设备上的语音识别任务，解决Transformer模型计算复杂度高的问题。提出LPA替代自注意力机制，实现高效推理。**

- **链接: [https://arxiv.org/pdf/2603.16922](https://arxiv.org/pdf/2603.16922)**

> **作者:** Yakov Pyotr Shkolnikov
>
> **摘要:** Self-attention scales quadratically with sequence length, limiting transformer-based speech models on edge devices. We introduce the Learnable Pulse Accumulator (LPA), an O(n) replacement that substitutes key-query dot products with learned gating functions: content-dependent rectangular pulses, periodic windows, and position-dependent basis functions. An MSE diagnostic sweep determines per-layer replacement difficulty and ordering. Replacing 8 of 12 wav2vec2-base layers yields 10.61% word error rate (WER) on LibriSpeech test-clean, +7.24 percentage points (pp) over the 3.37% baseline, with 3.27x speedup at 120s audio on Apple M4 Pro via an optimized MLX inference path. Cross-domain validation on SepFormer speech enhancement shows all 16 intra-chunk attention layers can be replaced without collapse, suggesting the depth wall arises from linguistic computation rather than an LPA limitation. LPA's near-binary gates at inference enable dense GPU computation with no CPU-GPU synchronization, and all operations map to mobile neural accelerators.
>
---
#### [new 002] Robust Nasality Representation Learning for Cleft Palate-Related Velopharyngeal Dysfunction Screening in Real-World Settings
- **分类: eess.AS**

- **简介: 该论文属于语音分析任务，旨在解决语音识别中因环境差异导致的性能下降问题。通过学习聚焦鼻音的语音表示，提升唇裂相关咽部功能障碍筛查的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17383](https://arxiv.org/pdf/2603.17383)**

> **作者:** Weixin Liu; Bowen Qu; Amy Stone; Maria E. Powell; Shama Dufresne; Stephane Braun; Izabela Galdyn; Michael Golinko; Bradley Malin; Zhijun Yin; Matthew E. Pontell
>
> **备注:** 2 figures. Machine learning for speech-based VPD screening under domain shift
>
> **摘要:** Velopharyngeal dysfunction (VPD) is characterized by inadequate velopharyngeal closure during speech and often causes hypernasality and reduced intelligibility. Although speech-based machine learning models can perform well under standardized clinical recording conditions, their performance often drops in real-world settings because of domain shift caused by differences in devices, channels, noise, and room acoustics. To improve robustness, we propose a two-stage framework for VPD screening. First, a nasality-focused speech representation is learned by supervised contrastive pre-training on an auxiliary corpus with phoneme alignments, using oral-context versus nasal-context supervision. Second, the encoder is frozen and used with lightweight classifiers on 0.5-second speech chunks, whose probabilities are aggregated to produce recording-level decisions with a fixed threshold. On an in-domain clinical cohort of 82 subjects, the proposed method achieved perfect recording-level screening performance (macro-F1 = 1.000, accuracy = 1.000). On a separate out-of-domain set of 131 heterogeneous public Internet recordings, large pretrained speech representations degraded substantially, while MFCC was the strongest baseline (macro-F1 = 0.612, accuracy = 0.641). The proposed method achieved the best out-of-domain performance (macro-F1 = 0.679, accuracy = 0.695), improving on the strongest baseline under the same evaluation protocol. These results suggest that learning a nasality-focused representation before clinical classification can reduce sensitivity to recording artifacts and improve robustness for deployable speech-based VPD screening.
>
---
#### [new 003] Over-the-air White-box Attack on the Wav2Vec Speech Recognition Neural Network
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音识别安全任务，旨在解决对抗攻击易被察觉的问题。研究提出方法使空中对抗攻击更隐蔽，同时评估其效果。**

- **链接: [https://arxiv.org/pdf/2603.16972](https://arxiv.org/pdf/2603.16972)**

> **作者:** Protopopov Alexey
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** Automatic speech recognition systems based on neural networks are vulnerable to adversarial attacks that alter transcriptions in a malicious way. Recent works in this field have focused on making attacks work in over-the-air scenarios, however such attacks are typically detectable by human hearing, limiting their potential applications. In the present work we explore different approaches of making over-the-air attacks less detectable, as well as the impact these approaches have on the attacks' effectiveness.
>
---
#### [new 004] SimulU: Training-free Policy for Long-form Simultaneous Speech-to-Speech Translation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于实时多语言通信任务，解决长时连续语音翻译问题。提出SimulU，无需训练，通过历史管理和输出选择策略实现高效翻译。**

- **链接: [https://arxiv.org/pdf/2603.16924](https://arxiv.org/pdf/2603.16924)**

> **作者:** Amirbek Djanibekov; Luisa Bentivogli; Matteo Negri; Sara Papi
>
> **摘要:** Simultaneous speech-to-speech translation (SimulS2S) is essential for real-time multilingual communication, with increasing integration into meeting and streaming platforms. Despite this, SimulS2S remains underexplored in research, where current solutions often rely on resource-intensive training procedures and operate on short-form, pre-segmented utterances, failing to generalize to continuous speech. To bridge this gap, we propose SimulU, the first training-free policy for long-form SimulS2S. SimulU adopts history management and speech output selection strategies that exploit cross-attention in pre-trained end-to-end models to regulate both input history and output generation. Evaluations on MuST-C across 8 languages show that SimulU achieves a better or comparable quality-latency trade-off against strong cascaded models. By eliminating the need for ad-hoc training, SimulU offers a promising path to end-to-end SimulS2S in realistic, long-form scenarios.
>
---
#### [new 005] Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在利用神经音频编解码器的分层量化结构，提升检测效果。通过建模不同量化层级的贡献，提取更有效的声学线索。**

- **链接: [https://arxiv.org/pdf/2603.16914](https://arxiv.org/pdf/2603.16914)**

> **作者:** Jinyang Wu; Zihan Pan; Qiquan Zhang; Sailor Hardik Bhupendra; Soumik Mondal
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Neural audio codecs discretize speech via residual vector quantization (RVQ), forming a coarse-to-fine hierarchy across quantizers. While codec models have been explored for representation learning, their discrete structure remains underutilized in speech deepfake detection. In particular, different quantization levels capture complementary acoustic cues, where early quantizers encode coarse structure and later quantizers refine residual details that reveal synthesis artifacts. Existing systems either rely on continuous encoder features or ignore this quantizer-level hierarchy. We propose a hierarchy-aware representation learning framework that models quantizer-level contributions through learnable global weighting, enabling structured codec representations aligned with forensic cues. Keeping the speech encoder backbone frozen and updating only 4.4% additional parameters, our method achieves relative EER reductions of 46.2% on ASVspoof 2019 and 13.9% on ASVspoof5 over strong baselines.
>
---
#### [new 006] Modeling Overlapped Speech with Shuffles
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于语音处理任务，解决多说话人重叠语音的对齐与转录问题。通过shuffle产品和部分有序有限状态自动机实现单次遍历的对齐与说话人归因。**

- **链接: [https://arxiv.org/pdf/2603.17769](https://arxiv.org/pdf/2603.17769)**

> **作者:** Matthew Wiesner; Samuele Cornell; Alexander Polok; Lucas Ondel Yang; Lukáš Burget; Sanjeev Khudanpur
>
> **摘要:** We propose to model parallel streams of data, such as overlapped speech, using shuffles. Specifically, this paper shows how the shuffle product and partial order finite-state automata (FSAs) can be used for alignment and speaker-attributed transcription of overlapped speech. We train using the total score on these FSAs as a loss function, marginalizing over all possible serializations of overlapping sequences at subword, word, and phrase levels. To reduce graph size, we impose temporal constraints by constructing partial order FSAs. We address speaker attribution by modeling (token, speaker) tuples directly. Viterbi alignment through the shuffle product FSA directly enables one-pass alignment. We evaluate performance on synthetic LibriSpeech overlaps. To our knowledge, this is the first algorithm that enables single-pass alignment of multi-talker recordings. All algorithms are implemented using k2 / Icefall.
>
---
#### [new 007] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FLAIR方法，用于全双工对话系统中的潜意识推理，解决边听边想的实时认知建模问题，通过连续推理提升对话质量。**

- **链接: [https://arxiv.org/pdf/2603.17837](https://arxiv.org/pdf/2603.17837)**

> **作者:** Donghang Wu; Tianyu Zhang; Yuxin Li; Hexin Liu; Chen Chen; Eng Siong Chng; Yoshua Bengio
>
> **摘要:** During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
>
---
#### [new 008] Multi-Source Evidence Fusion for Audio Question Answering
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于音频问答任务，旨在提升系统推理的准确性与可验证性。通过融合多源证据，利用多个模型生成并验证推理链，解决音频内容理解与逻辑推理问题。**

- **链接: [https://arxiv.org/pdf/2603.17822](https://arxiv.org/pdf/2603.17822)**

> **作者:** Aivo Olev; Tanel Alumäe
>
> **摘要:** Large audio language models (LALMs) can answer questions about speech, music, and environmental sounds, yet their internal reasoning is largely opaque and difficult to validate. We describe TalTech's solution to the Agent Track of the Interspeech 2026 Audio Reasoning Challenge, in which systems are evaluated on reasoning process quality, specifically the factual accuracy, logical soundness, and completeness of their reasoning chains. Our multi-source ensemble pipeline uses two LALMs that generate independent observations, while a separate text-only reasoning model cross-checks these against outputs from 25 acoustic tools organized into reliability tiers. By grounding every inference step in explicit, reliability-tagged evidence, the system produces dense, verifiable reasoning chains. Our system ranked first in the challenge, outperforming all competing systems by a wide margin in challenge's reasoning quality metric.
>
---
#### [new 009] Music Source Restoration with Ensemble Separation and Targeted Reconstruction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在从混音中恢复原始音轨。通过两阶段系统，先分离后修复，提升恢复质量。**

- **链接: [https://arxiv.org/pdf/2603.16926](https://arxiv.org/pdf/2603.16926)**

> **作者:** Xinlong Deng; Yu Xia; Jie Jiang
>
> **摘要:** The Inaugural Music Source Restoration (MSR) Challenge targets the recovery of original, unprocessed stems from fully mixed and mastered music. Unlike conventional music source separation, MSR requires reversing complex production processes such as equalization, compression, reverberation, and other real-world degradations. To address MSR, we propose a two-stage system. First, an ensemble of pre-trained separation models produces preliminary source estimates. Then a set of pre-trained BSRNN-based restoration models performs targeted reconstruction to refine these estimates. On the official MSR benchmark, our system surpasses the baselines on all metrics, ranking second among all submissions. The code is available at this https URL
>
---
#### [new 010] Uncertainty Quantification and Risk Control for Multi-Speaker Sound Source Localization
- **分类: eess.AS**

- **简介: 该论文属于声源定位任务，解决多说话人定位中的不确定性量化与风险控制问题。通过引入置信预测框架，提出两种方法，在已知和未知源数情况下实现可靠定位区域估计。**

- **链接: [https://arxiv.org/pdf/2603.17377](https://arxiv.org/pdf/2603.17377)**

> **作者:** Vadim Rozenfeld; Bracha Laufer Goldshtein
>
> **备注:** 13 pages, 4 figures. Code available at: this https URL
>
> **摘要:** Reliable Sound Source Localization (SSL) plays an essential role in many downstream tasks, where informed decision making depends not only on accurate localization but also on the confidence in each estimate. This need for reliability becomes even more pronounced in challenging conditions, such as reverberant environments and multi-source scenarios. However, existing SSL methods typically provide only point estimates, offering limited or no Uncertainty Quantification (UQ). We leverage the Conformal Prediction (CP) framework and its extensions for controlling general risk functions to develop two complementary UQ approaches for SSL. The first assumes that the number of active sources is known and constructs prediction regions that cover the true source locations. The second addresses the more challenging setting where the source count is unknown, first reliably estimating the number of active sources and then forming corresponding prediction regions. We evaluate the proposed methods on extensive simulations and real-world recordings across varying reverberation levels and source configurations. Results demonstrate reliable finite-sample guarantees and consistent performance for both known and unknown source-count scenarios, highlighting the practical utility of the proposed frameworks for uncertainty-aware SSL.
>
---
#### [new 011] The Voice Behind the Words: Quantifying Intersectional Bias in SpeechLLMs
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在检测SpeechLLMs中的口音与性别交叉偏见。通过控制实验和人类评估，发现东欧口音尤其是女性语音存在显著帮助性评分差异。**

- **链接: [https://arxiv.org/pdf/2603.16941](https://arxiv.org/pdf/2603.16941)**

> **作者:** Shree Harsha Bokkahalli Satish; Christoph Minixhofer; Maria Teleki; James Caverlee; Ondřej Klejch; Peter Bell; Gustav Eje Henter; Éva Székely
>
> **备注:** 5 pages, 3 figures, 1 table, Submitted to Interspeech 2026
>
> **摘要:** Speech Large Language Models (SpeechLLMs) process spoken input directly, retaining cues such as accent and perceived gender that were previously removed in cascaded pipelines. This introduces speaker identity dependent variation in responses. We present a large-scale intersectional evaluation of accent and gender bias in three SpeechLLMs using 2,880 controlled interactions across six English accents and two gender presentations, keeping linguistic content constant through voice cloning. Using pointwise LLM-judge ratings, pairwise comparisons, and Best-Worst Scaling with human validation, we detect consistent disparities. Eastern European-accented speech receives lower helpfulness scores, particularly for female-presenting voices. The bias is implicit: responses remain polite but differ in helpfulness. While LLM judges capture the directional trend of these biases, human evaluators exhibit significantly higher sensitivity, uncovering sharper intersectional disparities.
>
---
#### [new 012] Beyond Deep Learning: Speech Segmentation and Phone Classification with Neural Assemblies
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出一种基于神经集合的语音处理框架，用于语音分割和音素分类。解决传统深度学习依赖大量数据的问题，通过生物启发的神经机制实现无需权重训练的语音分析。**

- **链接: [https://arxiv.org/pdf/2603.16923](https://arxiv.org/pdf/2603.16923)**

> **作者:** Trevor Adelson; Vidhyasaharan Sethu; Ting Dang
>
> **备注:** Submitted to Interspeech 2026. 9 Pages
>
> **摘要:** Deep learning dominates speech processing but relies on massive datasets, global backpropagation-guided weight updates, and produces entangled representations. Assembly Calculus (AC), which models sparse neuronal assemblies via Hebbian plasticity and winner-take-all competition, offers a biologically grounded alternative, yet prior work focused on discrete symbolic inputs. We introduce an AC-based speech processing framework that operates directly on continuous speech by combining three key contributions:(i) neural encoding that converts speech into assembly-compatible spike patterns using probabilistic mel binarisation and population-coded MFCCs; (ii) a multi-area architecture organising assemblies across hierarchical timescales and classes; and (iii) cross-area update schemes for downstream tasks. Applied to two core tasks of boundary detection and segment classification, our framework detects phone (F1=0.69) and word (F1=0.61) boundaries without any weight training, and achieves 47.5% and 45.1% accuracy on phone and command recognition. These results show that AC-based dynamical systems are a viable alternative to deep learning for speech processing.
>
---
#### [new 013] Synthetic Data Domain Adaptation for ASR via LLM-based Text and Phonetic Respelling Augmentation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别领域的域适应任务，旨在解决ASR在领域特定数据上性能下降的问题。通过文本和发音重拼接增强方法，提升合成数据的多样性和真实感，从而提高ASR的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.16920](https://arxiv.org/pdf/2603.16920)**

> **作者:** Natsuo Yamashita; Koichi Nagatsuka; Hiroaki Kokubo; Kota Dohi; Tuan Vu Ho
>
> **备注:** accepted by ICASSP 2026
>
> **摘要:** End-to-end automatic speech recognition often degrades on domain-specific data due to scarce in-domain resources. We propose a synthetic-data-based domain adaptation framework with two contributions: (1) a large language model (LLM)-based text augmentation pipeline with a filtering strategy that balances lexical diversity, perplexity, and domain-term coverage, and (2) phonetic respelling augmentation (PRA), a novel method that introduces pronunciation variability through LLM-generated orthographic pseudo-spellings. Unlike conventional acoustic-level methods such as SpecAugment, PRA provides phonetic diversity before speech synthesis, enabling synthetic speech to better approximate real-world variability. Experimental results across four domain-specific datasets demonstrate consistent reductions in word error rate, confirming that combining domain-specific lexical coverage with realistic pronunciation variation significantly improves ASR robustness.
>
---
#### [new 014] Shared Representation Learning for Reference-Guided Targeted Sound Detection
- **分类: eess.AS; cs.AI**

- **简介: 该论文研究目标声音检测任务，旨在从混合音频中定位给定参考音的靶向声音。提出共享表示学习方法，提升检测性能并简化架构。**

- **链接: [https://arxiv.org/pdf/2603.17025](https://arxiv.org/pdf/2603.17025)**

> **作者:** Shubham Gupta; Adarsh Arigala; B. R. Dilleswari; Sri Rama Murty Kodukula
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Human listeners exhibit the remarkable ability to segregate a desired sound from complex acoustic scenes through selective auditory attention, motivating the study of Targeted Sound Detection (TSD). The task requires detecting and localizing a target sound in a mixture when a reference audio of that sound is provided. Prior approaches, rely on generating a sound-discriminative conditional embedding vector for the reference and pairing it with a mixture encoder, jointly optimized with a multi-task learning approach. In this work, we propose a unified encoder architecture that processes both the reference and mixture audio within a shared representation space, promoting stronger alignment while reducing architectural complexity. This design choice not only simplifies the overall framework but also enhances generalization to unseen classes. Following the multi-task training paradigm, our method achieves substantial improvements over prior approaches, surpassing existing methods and establishing a new state-of-the-art benchmark for targeted sound detection, with a segment-level F1 score of 83.15% and an overall accuracy of 95.17% on the URBAN-SED dataset.
>
---
#### [new 015] Amanous: Distribution-Switching for Superhuman Piano Density on Disklavier
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出Amanous系统，解决自动化钢琴创作中不同技法整合问题，通过分布切换统一L-system、随机分布和节拍赋格，提升音符密度与复杂性。**

- **链接: [https://arxiv.org/pdf/2603.16890](https://arxiv.org/pdf/2603.16890)**

> **作者:** Joonhyung Bae
>
> **摘要:** The automated piano enables note densities, polyphony, and register changes far beyond human physical limits, yet the three dominant traditions for composing such textures--Nancarrow's tempo canons, Xenakis's stochastic distributions, and L-system grammars--have developed in isolation. This paper presents Amanous, a hardware-aware composition system for Yamaha Disklavier that unifies these methodologies through distribution-switching: L-system symbols select distinct distributional regimes rather than merely modulating parameters within a fixed family. Four contributions are reported. (1) A four-layer architecture (symbolic, parametric, numeric, physical) produces statistically distinct sections with large effect sizes (d = 3.70-5.34), validated by per-layer degradation and ablation experiments. (2) A hardware abstraction layer formalizes velocity-dependent latency and key reset constraints, keeping superhuman textures within the Disklavier's actuable envelope. (3) A density sweep reveals a computational saturation transition at 24-30 notes/s (bootstrap 95% CI: 23.3-50.0), beyond which single-domain melodic metrics lose discriminative power and cross-domain coupling becomes necessary. (4) A convergence point calculus operationalizes tempo-canon geometry as a control interface, enabling convergence events to trigger distribution switches linking macro-temporal structure to micro-level texture. All results are computational; a psychoacoustic validation protocol is proposed for future work. The pipeline has been deployed on a physical Disklavier, demonstrating algorithmic self-consistency and sub-millisecond software precision. Supplementary materials (Excerpts 1-4): this https URL. Source code: this https URL.
>
---
#### [new 016] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于视觉媒体说话人日志任务，解决开放场景下的多模态说话人识别问题。提出CineSRD框架，融合视觉、音频和语言信息，提升复杂视频中的说话人标注效果。**

- **链接: [https://arxiv.org/pdf/2603.16966](https://arxiv.org/pdf/2603.16966)**

> **作者:** Liangbin Huang; Xiaohua Liao; Chaoqun Cui; Shijing Wang; Zhaolong Huang; Yanlong Du; Wenji Mao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Traditional speaker diarization systems have primarily focused on constrained scenarios such as meetings and interviews, where the number of speakers is limited and acoustic conditions are relatively clean. To explore open-world speaker diarization, we extend this task to the visual media domain, encompassing complex audiovisual programs such as films and TV series. This new setting introduces several challenges, including long-form video understanding, a large number of speakers, cross-modal asynchrony between audio and visual cues, and uncontrolled in-the-wild variability. To address these challenges, we propose Cinematic Speaker Registration & Diarization (CineSRD), a unified multimodal framework that leverages visual, acoustic, and linguistic cues from video, speech, and subtitles for speaker annotation. CineSRD first performs visual anchor clustering to register initial speakers and then integrates an audio language model for speaker turn detection, refining annotations and supplementing unregistered off-screen speakers. Furthermore, we construct and release a dedicated speaker diarization benchmark for visual media that includes Chinese and English programs. Experimental results demonstrate that CineSRD achieves superior performance on the proposed benchmark and competitive results on conventional datasets, validating its robustness and generalizability in open-world visual media settings.
>
---
#### [new 017] Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于L2语音评估任务，旨在解决SpeechLLMs与人类评分者间一致性不足的问题。通过引入基于评分标准的框架和不确定性校准方法，提升模型评估的可靠性与解释性。**

- **链接: [https://arxiv.org/pdf/2603.16889](https://arxiv.org/pdf/2603.16889)**

> **作者:** Aditya Kamlesh Parikh; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** Accepted to LREC 2026. This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants, which is financed by the Dutch Research Council (NWO)
>
> **摘要:** Reliable and interpretable automated assessment of second-language (L2) speech remains a central challenge, as large speech-language models (SpeechLLMs) often struggle to align with the nuanced variability of human raters. To address this, we introduce a rubric-guided reasoning framework that explicitly encodes multi-aspect human assessment criteria: accuracy, fluency, and prosody, while calibrating model uncertainty to capture natural rating variability. We fine-tune the Qwen2-Audio-7B-Instruct model using multi-rater human judgments and develop an uncertainty-calibrated regression approach supported by conformal calibration for interpretable confidence intervals. Our Gaussian uncertainty modeling and conformal calibration approach achieves the strongest alignment with human ratings, outperforming regression and classification baselines. The model reliably assesses fluency and prosody while highlighting the inherent difficulty of assessing accuracy. Together, these results demonstrate that rubric-guided, uncertainty-calibrated reasoning offers a principled path toward trustworthy and explainable SpeechLLM-based speech assessment.
>
---
#### [new 018] Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于多语言语音识别任务，解决语音LLM在数据不平衡下的跨语言干扰与知识迁移问题。提出Zipper-LoRA框架，动态调整参数共享与分离，提升低资源语言性能。**

- **链接: [https://arxiv.org/pdf/2603.17558](https://arxiv.org/pdf/2603.17558)**

> **作者:** Yuxiang Mei; Delai Qiu; Shengping Liu; Jiaen Liang; Yanhua Long
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Speech Large Language Models (Speech-LLMs) have emerged as a powerful approach for automatic speech recognition (ASR) by aligning speech encoders with large language models. However, adapting these systems to multilingual settings with imbalanced data distributions remains challenging. In such scenarios, a stability-plasticity dilemma often arises: fully shared Parameter-Efficient Fine-Tuning (PEFT) can cause negative inter-lingual interference for under-represented languages, while fully language-specific tuning limits the cross-lingual beneficial knowledge transfer needed for low-resource tasks. To address this, we propose Zipper-LoRA, a novel rank-level decoupling framework with three variants (Static, Hard, and Soft) that dynamically synthesizes LoRA updates from shared and language-specific subspaces. By using a lightweight language-conditioned router, Zipper-LoRA dynamically controls the contribution of each subspace at the LoRA rank level, enabling fine-grained sharing where languages are compatible and strict decoupling when conflicts occur. To further stabilize optimization under imbalanced data, we propose a two-stage training strategy with an Initial-B warm start that significantly accelerates convergence. Experiments on a 12-language mixed-resource setting show that Zipper-LoRA consistently outperforms both fully shared and independent baselines, particularly in extremely low-resource scenarios. Moreover, we demonstrate that these gains are robust across both chunked and non-chunked encoder configurations, confirming the framework's reliability for practical, large-scale multilingual ASR. Our code and data will be available at this https URL for reproducibility.
>
---
#### [new 019] Neuron-Level Emotion Control in Speech-Generative Large Audio-Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音生成任务，解决情感控制难题。通过识别情感敏感神经元，实现无需训练的情感调控，提升情感表达准确性。**

- **链接: [https://arxiv.org/pdf/2603.17231](https://arxiv.org/pdf/2603.17231)**

> **作者:** Xiutian Zhao; Ismail Rasim Ulgen; Philipp Koehn; Björn Schuller; Berrak Sisman
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** Large audio-language models (LALMs) can produce expressive speech, yet reliable emotion control remains elusive: conversions often miss the target affect and may degrade linguistic fidelity through refusals, hallucinations, or paraphrase. We present, to our knowledge, the first neuron-level study of emotion control in speech-generative LALMs and demonstrate that compact emotion-sensitive neurons (ESNs) are causally actionable, enabling training-free emotion steering at inference time. ESNs are identified via success-filtered activation aggregation enforcing both emotion realization and content preservation. Across three LALMs (Qwen2.5-Omni-7B, MiniCPM-o 4.5, Kimi-Audio), ESN interventions yield emotion-specific gains that generalize to unseen speakers and are supported by automatic and human evaluation. Controllability depends on selector design, mask sparsity, filtering, and intervention strength. Our results establish a mechanistic framework for training-free emotion control in speech generation.
>
---
#### [new 020] Collecting Prosody in the Wild: A Content-Controlled, Privacy-First Smartphone Protocol and Empirical Evaluation
- **分类: cs.HC; eess.AS**

- **简介: 该论文提出一种隐私优先的智能手机协议，用于收集自然语音数据，解决语调与语义混淆、隐私和参与度问题。通过脚本朗读标准化内容，提取并传输语音特征，评估数据质量和预测任务。**

- **链接: [https://arxiv.org/pdf/2603.17061](https://arxiv.org/pdf/2603.17061)**

> **作者:** Timo K. Koch; Florian Bemmann; Ramona Schoedel; Markus Buehner; Clemens Stachl
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Collecting everyday speech data for prosodic analysis is challenging due to the confounding of prosody and semantics, privacy constraints, and participant compliance. We introduce and empirically evaluate a content-controlled, privacy-first smartphone protocol that uses scripted read-aloud sentences to standardize lexical content (including prompt valence) while capturing natural variation in prosodic delivery. The protocol performs on-device prosodic feature extraction, deletes raw audio immediately, and transmits only derived features for analysis. We deployed the protocol in a large study (N = 560; 9,877 recordings), evaluated compliance and data quality, and conducted diagnostic prediction tasks on the extracted features, predicting speaker sex and concurrently reported momentary affective states (valence, arousal). We discuss implications and directions for advancing and deploying the protocol.
>
---
## 更新

#### [replaced 001] Do You Hear What I Mean? Quantifying the Instruction-Perception Gap in Instruction-Guided Expressive Text-To-Speech Systems
- **分类: eess.AS**

- **简介: 该论文属于文本到语音生成任务，旨在解决指令与感知之间的差距问题。通过分析用户指令与听众感知的对齐情况，提出E-VOC数据集，并评估多个ITTS系统的表现。**

- **链接: [https://arxiv.org/pdf/2509.13989](https://arxiv.org/pdf/2509.13989)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Tzu-Chieh Wei; Kuan-Yu Chen; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Instruction-guided text-to-speech (ITTS) enables users to control speech generation through natural language prompts, offering a more intuitive interface than traditional TTS. However, the alignment between user style instructions and listener perception remains largely unexplored. This work first presents a perceptual analysis of ITTS controllability across two expressive dimensions (adverbs of degree and graded emotion intensity) and collects human ratings on speaker age and word-level emphasis attributes. To comprehensively reveal the instruction-perception gap, we provide a data collection with large-scale human evaluations, named Expressive VOice Control (E-VOC) corpus. Furthermore, we reveal that (1) gpt-4o-mini-tts is the most reliable ITTS model with great alignment between instruction and generated utterances across acoustic dimensions. (2) The 5 analyzed ITTS systems tend to generate Adult voices even when the instructions ask to use child or Elderly voices. (3) Fine-grained control remains a major challenge, indicating that most ITTS systems have substantial room for improvement in interpreting slightly different attribute instructions.
>
---
#### [replaced 002] Integrated Spoofing-Robust Automatic Speaker Verification via a Three-Class Formulation and LLR
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音安全任务，解决 spoofing 问题。提出一种三类框架，实现更可解释的说话人验证，提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.13780](https://arxiv.org/pdf/2603.13780)**

> **作者:** Kai Tan; Lin Zhang; Ruiteng Zhang; Johan Rohdin; Leibny Paola García-Perera; Zexin Cai; Sanjeev Khudanpur; Matthew Wiesner; Nicholas Andrews
>
> **备注:** Submitted to Interspeech 2026; put on arxiv based on requirement from Interspeech: "Interspeech no longer enforces an anonymity period for submissions." and "For authors that prefer to upload their paper online, a note indicating that the paper was submitted for review to Interspeech should be included in the posting."
>
> **摘要:** Spoofing-robust automatic speaker verification (SASV) aims to integrate automatic speaker verification (ASV) and countermeasure (CM). A popular solution is fusion of independent ASV and CM scores. To better modeling SASV, some frameworks integrate ASV and CM within a single network. However, these solutions are typically bi-encoder based, offer limited interpretability, and cannot be readily adapted to new evaluation parameters without retraining. Based on this, we propose a unified end-to-end framework via a three-class formulation that enables log-likelihood ratio (LLR) inference from class logits for a more interpretable decision pipeline. Experiments show comparable performance to existing methods on ASVSpoof5 and better results on SpoofCeleb. The visualization and analysis also prove that the three-class reformulation provides more interpretability.
>
---
#### [replaced 003] Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声音场景情感识别任务，旨在通过图拓扑学习选择关键特征。提出一种结合图学习与信息准则的特征选择框架，以优化情感描述符的选取。**

- **链接: [https://arxiv.org/pdf/2509.16760](https://arxiv.org/pdf/2509.16760)**

> **作者:** Samuel Rey; Luca Martino; Roberto San Millan; Eduardo Morgado
>
> **摘要:** Research on soundscapes has shifted the focus of environmental acoustics from noise levels to the perception of sounds, incorporating contextual factors. Soundscape emotion recognition (SER) models perception using a set of features, with arousal and valence commonly regarded as sufficient descriptors of affect. In this work, we blend \emph{graph learning} techniques with a novel \emph{information criterion} to develop a feature selection framework for SER. Specifically, we estimate a sparse graph representation of feature relations using linear structural equation models (SEM) tailored to the widely used Emo-Soundscapes dataset. The resulting graph captures the relations between input features and the two emotional outputs. To determine the appropriate level of sparsity, we propose a novel \emph{generalized elbow detector}, which provides both a point estimate and an uncertainty interval. We conduct an extensive evaluation of our methods, including visualizations of the inferred relations. While several of our findings align with previous studies, the graph representation also reveals a strong connection between arousal and valence, challenging common SER assumptions.
>
---
#### [replaced 004] Towards Inclusive Communication: A Unified Framework for Generating Spoken Language from Sign, Lip, and Audio
- **分类: cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于多模态语言生成任务，旨在解决聋哑人群通信障碍问题。提出统一框架融合手语、唇读和音频，提升语音文本生成效果。**

- **链接: [https://arxiv.org/pdf/2508.20476](https://arxiv.org/pdf/2508.20476)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Sungjune Park; Junil Won; Yong Man Ro
>
> **摘要:** Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such audio-centric systems inherently exclude individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we propose the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and Audio-Visual Speech Recognition. Furthermore, our analysis reveals a key linguistic insight: explicitly modeling lip movements as a distinct modality significantly improves SLT performance by capturing critical non-manual cues.
>
---
#### [replaced 005] NV-Bench: Benchmark of Nonverbal Vocalization Synthesis for Expressive Text-to-Speech Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到语音生成任务，旨在解决非语言发声评估标准缺失的问题。提出NV-Bench基准，包含多语言数据及双维评估协议，提升合成效果的客观评价。**

- **链接: [https://arxiv.org/pdf/2603.15352](https://arxiv.org/pdf/2603.15352)**

> **作者:** Qinke Ni; Huan Liao; Dekun Chen; Yuxiang Wang; Zhizheng Wu
>
> **备注:** Submit to Interspeech 2026
>
> **摘要:** While recent text-to-speech (TTS) systems increasingly integrate nonverbal vocalizations (NVs), their evaluations lack standardized metrics and reliable ground-truth references. To bridge this gap, we propose NV-Bench, the first benchmark grounded in a functional taxonomy that treats NVs as communicative acts rather than acoustic artifacts. NV-Bench comprises 1,651 multi-lingual, in-the-wild utterances with paired human reference audio, balanced across 14 NV categories. We introduce a dual-dimensional evaluation protocol: (1) Instruction Alignment, utilizing the proposed paralinguistic character error rate (PCER) to assess controllability, (2) Acoustic Fidelity, measuring the distributional gap to real recordings to assess acoustic realism. We evaluate diverse TTS models and develop two baselines. Experimental results demonstrate a strong correlation between our objective metrics and human perception, establishing NV-Bench as a standardized evaluation framework.
>
---
