# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] A Manual Bar-by-Bar Tempo Measurement Protocol for Polyphonic Chamber Music Recordings: Design, Validation, and Application to Beethoven's Piano and Cello Sonatas
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐分析任务，旨在解决历史复调音乐录音中自动节拍检测失败的问题，提出一种手动逐小节计时协议以准确提取节奏数据。**

- **链接: [https://arxiv.org/pdf/2604.15278](https://arxiv.org/pdf/2604.15278)**

> **作者:** Ignasi Sole
>
> **摘要:** Empirical performance analysis depends on the accurate extraction of tempo data from recordings, yet standard computational tools, designed for monophonic audio or modern studio conditions, fail systematically when applied to historical polyphonic chamber music. This paper documents the failure of automated beat-detection software on duo recordings of Beethoven's five piano and cello sonatas (Op.~5 Nos.~1 and~2; Op.~69; Op.~102 Nos.~1 and~2), and presents a formalised manual alternative: a cumulative lap-timer protocol that yields bar-level beats-per-minute data with millisecond resolution. The protocol, developed in cross-disciplinary collaboration with an engineer specialising in VLSI design, rests on a cumulative timestamp architecture that prevents error accumulation, permits internal self-validation, and captures expressive timing phenomena (rubato, fermatas, accelerandi, ritardandi) that automated tools systematically suppress or misread. The mathematical derivation of the BPM formula, the spreadsheet data structure, and the error characterisation are presented in full. Applied to over one hundred movement-level recordings spanning 1930--2012, the protocol generated a dataset subsequently visualised through tempographs, histograms with spline-smoothed probability density functions, ridgeline plots, and combination charts. The paper argues that manual annotation is not a methodological retreat but a principled response to the intrinsic limitations of computational tools when faced with the specific challenges of polyphonic historical recordings. The complete dataset and analysis code are publicly available.
>
---
#### [new 002] UniPASE: A Generative Model for Universal Speech Enhancement with High Fidelity and Low Hallucinations
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在从多种噪声中恢复高质量语音。提出UniPASE模型，通过统一表示增强和声学适配，实现低伪影、高保真的语音重建。**

- **链接: [https://arxiv.org/pdf/2604.14606](https://arxiv.org/pdf/2604.14606)**

> **作者:** Xiaobin Rong; Zheng Wang; Yushi Wang; Jun Gao; Jing Lu
>
> **备注:** Submitted to IEEE TASLP
>
> **摘要:** Universal speech enhancement (USE) aims to restore speech signals from diverse distortions across multiple sampling rates. We propose UniPASE, an extension of the low-hallucination PASE framework tailored for USE. At its core is DeWavLM-Omni, a unified representation-level enhancement module fine-tuned from WavLM via knowledge distillation on a large-scale supervised multi-distortion dataset. This module directly converts degraded waveforms into clean and linguistically faithful phonetic representations, ensuring robust enhancement with minimal linguistic hallucination. Based on these enhanced phonetic representations, an Adapter generates enhanced acoustic representations containing rich acoustic details, which a neural Vocoder uses to reconstruct corresponding high-fidelity 16-kHz waveforms. A PostNet then converts the waveforms to 48~kHz before resampling them to their original rates, enabling seamless handling of inputs and outputs at multiple sampling rates. Experimental results on several evaluation datasets, covering sub-tasks and full tasks, demonstrate that UniPASE achieves superior or competitive performance compared with existing state-of-the-art models. The proposed model also serves as the backbone of our submission to the URGENT 2026 Challenge, which achieved 1st place in the objective evaluation. The source code and audio demos are available at this https URL.
>
---
#### [new 003] Who is Speaking or Who is Depressed? A Controlled Study of Speaker Leakage in Speech-Based Depression Detection
- **分类: eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在解决模型是否依赖说话者身份而非抑郁特征的问题。通过控制说话者重叠，验证模型泛化能力，发现当前模型严重依赖说话者信息。**

- **链接: [https://arxiv.org/pdf/2604.14354](https://arxiv.org/pdf/2604.14354)**

> **作者:** Hsiang-Chen Yeh; Luqi Sun; Aurosweta Mahapatra; Shreeram Suresh Chandra; Emily Mower Provost; Berrak Sisman
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** This study investigates whether speech-based depression detection models learn depression-related acoustic biomarkers or instead rely on speaker identity cues. Using the DAIC-WOZ dataset, we propose a data-splitting strategy that controls speaker overlap between training and test sets while keeping the training size constant, and evaluate three models of varying complexity. Results show that speaker overlap significantly boosts performance, whereas accuracy drops sharply on unseen speakers. Even with a Domain-Adversarial Neural Network, a substantial performance gap remains. These findings indicate that depression-related features extracted by current speech models are highly entangled with speaker identity. Conventional evaluation protocols may therefore overestimate generalization and clinical utility, highlighting the need for strictly speaker-independent evaluation.
>
---
#### [new 004] ClariCodec: Optimising Neural Speech Codes for 200bps Communication using Reinforcement Learning
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决低比特率下语音可懂性下降的问题。通过强化学习优化量化策略，提升200bps下的词错误率表现。**

- **链接: [https://arxiv.org/pdf/2604.14654](https://arxiv.org/pdf/2604.14654)**

> **作者:** Junyi Wang; Chi Zhang; Jing Qian; Haifeng Luo; Hao Wang; Zengrui Jin; Chao Zhang
>
> **摘要:** In bandwidth-constrained communication such as satellite and underwater channels, speech must often be transmitted at ultra-low bitrates where intelligibility is the primary objective. At such extreme compression levels, codecs trained with acoustic reconstruction losses tend to allocate bits to perceptual detail, leading to substantial degradation in word error rate (WER). This paper proposes ClariCodec, a neural speech codec operating at 200 bit per second (bps) that reformulates quantisation as a stochastic policy, enabling reinforcement learning (RL)-based optimisation of intelligibility. Specifically, the encoder is fine-tuned using WER-driven rewards while the acoustic reconstruction pipeline remains frozen. Even without RL, ClariCodec achieves 3.68% WER on the LibriSpeech test-clean set at 200 bps, already competitive with codecs operating at higher bitrates. Further RL fine-tuning reduces WER to 3.20% on test-clean and 8.93% on test-other, corresponding to a 13% relative reduction while preserving perceptual quality.
>
---
#### [new 005] HARNESS: Lightweight Distilled Arabic Speech Foundation Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出HArnESS，一个轻量级阿拉伯语语音基础模型，解决资源受限环境下的语音任务部署问题。通过自蒸馏和PCA压缩，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.14186](https://arxiv.org/pdf/2604.14186)**

> **作者:** Vrunda N. Sukhadia; Shammur Absar Chowdhury
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Large self-supervised speech (SSL) models achieve strong downstream performance, but their size limits deployment in resource-constrained settings. We present HArnESS, an Arabic-centric self-supervised speech model family trained from scratch with iterative self-distillation, together with lightweight student variants that offer strong accuracy-efficiency trade-offs on Automatic Speech Recognition (ASR), Dialect Identification (DID), and Speech Emotion Recognition (SER). Our approach begins with a large bilingual Arabic-English teacher and progressively distills its knowledge into compressed student models while preserving Arabic-relevant acoustic and paralinguistic representations. We further study PCA-based compression of the teacher supervision signal to better match the capacity of shallow and thin students. Compared with HuBERT and XLS-R, HArnESS consistently improves performance on Arabic downstream tasks, while the compressed models remain competitive under substantial structural reduction. These results position HArnESS as a practical and accessible Arabic-centric SSL foundation for real-world speech applications.
>
---
#### [new 006] Disentangled Dual-Branch Graph Learning for Conversational Emotion Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于对话情感识别任务，旨在解决跨模态信息冗余、语义对齐不足及高阶说话人交互建模问题。通过双分支图学习与特征解耦方法提升情感预测效果。**

- **链接: [https://arxiv.org/pdf/2604.14204](https://arxiv.org/pdf/2604.14204)**

> **作者:** Chengling Guo; Yuntao Shou; Tao Meng; Wei Ai; Yun Tan; Keqin Li
>
> **备注:** 16 pages
>
> **摘要:** Multimodal emotion recognition in conversations aims to infer utterance-level emotions by jointly modeling textual, acoustic, and visual cues within context. Despite recent progress, key challenges remain, including redundant cross-modal information, imperfect semantic alignment, and insufficient modeling of high-order speaker interactions. To address these issues, we propose a framework that combines dual-space feature disentanglement with dual-branch graph learning. A shared encoder and modality-specific encoders are used to separate modality-invariant and modality-specific representations. The invariant features are modeled by a Fourier graph neural network to capture global consistency and complementary patterns, with a frequency-domain contrastive objective to enhance discriminability. In parallel, a speaker-aware hypergraph is constructed over modality-specific features to model high-order interactions, along with a speaker-consistency constraint to maintain coherent semantics. Finally, the two branches are fused for utterance-level emotion prediction. Experiments on IEMOCAP and MELD demonstrate that the proposed method achieves superior performance over strong baselines, validating its effectiveness.
>
---
#### [new 007] Listen, Pause, and Reason: Toward Perception-Grounded Hybrid Reasoning for Audio Understanding
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频理解任务，旨在解决音频推理可靠性问题。通过构建PAQA数据集和提出HyPeR框架，实现感知与推理的结合，提升多说话人音频理解效果。**

- **链接: [https://arxiv.org/pdf/2604.14806](https://arxiv.org/pdf/2604.14806)**

> **作者:** Jieyi Wang; Yazhe Niu; Dexuan Xu; Zhongyu Wei
>
> **摘要:** Recent Large Audio Language Models have demonstrated impressive capabilities in audio understanding. However, they often suffer from perceptual errors, while reliable audio reasoning is unattainable without first grounding the model's perception in structured auditory scenes. Inspired by Auditory Scene Analysis, we first introduce a Perception-Aware Question Answering (PAQA) dataset. PAQA implements a hierarchical decoupling strategy that separates speech from environmental sound and distinguishes multiple speakers, providing explicit perceptual reasoning for training. Building on this, we propose HyPeR, a two-stage Hybrid Perception-Reasoning framework. In Stage I, we finetune the model on PAQA to perceive acoustic attributes in complex audio. In Stage II, we leverage GRPO to refine the model's internal deliberation. We also introduce PAUSE tokens to facilitate latent computation during acoustically ambiguous phases and design perceptual consistency reward to align reasoning rationales with raw audio. Experiments across benchmarks demonstrate that HyPeR achieves absolute improvements over the base model, with performance comparable to large-scale models, stressing the effectiveness of hybrid perception-grounded reasoning for robust and multi-speaker audio understanding.
>
---
#### [new 008] From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决ASR错误难以检测的问题。通过分析多模型分歧，构建无需人工参考的不确定性信号，以定位潜在不可靠的文本区域。**

- **链接: [https://arxiv.org/pdf/2604.14152](https://arxiv.org/pdf/2604.14152)**

> **作者:** Abdolamir Karbalaie; Fernando Seoane; Farhad Abtahi
>
> **摘要:** Ambient AI "scribe" systems promise to reduce clinical documentation burden, but automatic speech recognition (ASR) errors can remain unnoticed without careful review, and high-quality human reference transcripts are often unavailable for calibrating uncertainty. We investigate whether cross-model disagreement among heterogeneous ASR systems can act as a reference-free uncertainty signal to prioritize human verification in medical transcription workflows. Using 50 publicly available medical education audio clips (8 h 14 min), we transcribed each clip with eight ASR systems spanning commercial APIs and open-source engines. We aligned multi-model outputs, built consensus pseudo-references, and quantified token-level agreement using a majority-strength metric; we further characterized disagreements by type (content vs. punctuation/formatting) and assessed per-model agreement via leave-one-model-out (jackknife) consensus scoring. Inter-model reliability was low (ICC[2,1] = 0.131), indicating heterogeneous failure modes across systems. Across 76,398 evaluated token positions, 72.1% showed near-unanimous agreement (7-8 models), while 2.5% fell into high-risk bands (0-3 models), with high-risk mass varying from 0.7% to 11.4% across accent groups. Low-agreement regions were enriched for content disagreements, with the content fraction increasing from 53.9% to 73.9% across quintiles of high-risk mass. These results suggest that cross-model disagreement provides a sparse, localizable signal that can surface potentially unreliable transcript spans without human-verified references, enabling targeted review; clinical accuracy of flagged regions remains to be established.
>
---
#### [new 009] VoxSafeBench: Not Just What Is Said, but Who, How, and Where
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出VoxSafeBench，用于评估语音语言模型在安全、公平和隐私方面的社会对齐能力，解决语音上下文中的潜在风险问题。**

- **链接: [https://arxiv.org/pdf/2604.14548](https://arxiv.org/pdf/2604.14548)**

> **作者:** Yuxiang Wang; Hongyu Liu; Yijiang Xu; Qinke Ni; Li Wang; Wan Lin; Kunyu Feng; Dekun Chen; Xu Tan; Lei Wang; Jie Shi; Zhizheng Wu
>
> **摘要:** As speech language models (SLMs) transition from personal devices into shared, multi-user environments, their responses must account for far more than the words alone. Who is speaking, how they sound, and where the conversation takes place can each turn an otherwise benign request into one that is unsafe, unfair, or privacy-violating. Existing benchmarks, however, largely focus on basic audio comprehension, study individual risks in isolation, or conflate content that is inherently harmful with content that only becomes problematic due to its acoustic context. We introduce VoxSafeBench, among the first benchmarks to jointly evaluate social alignment in SLMs across three dimensions: safety, fairness, and privacy. VoxSafeBench adopts a Two-Tier design: Tier1 evaluates content-centric risks using matched text and audio inputs, while Tier2 targets audio-conditioned risks in which the transcript is benign but the appropriate response hinges on the speaker, paralinguistic cues, or the surrounding environment. To validate Tier2, we include intermediate perception probes and confirm that frontier SLMs can successfully detect these acoustic cues yet still fail to act on them appropriately. Across 22 tasks with bilingual coverage, we find that safeguards appearing robust on text often degrade in speech: safety awareness drops for speaker- and scene-conditioned risks, fairness erodes when demographic differences are conveyed vocally, and privacy protections falter when contextual cues arrive acoustically. Together, these results expose a pervasive speech grounding gap: current SLMs frequently recognize the relevant social norm in text but fail to apply it when the decisive cue must be grounded in speech. Code and data are publicly available at: this https URL
>
---
#### [new 010] The Acoustic Camouflage Phenomenon: Re-evaluating Speech Features for Financial Risk Prediction
- **分类: cs.SD; cs.LG; eess.AS; q-fin.CP; q-fin.ST**

- **简介: 该论文属于金融风险预测任务，研究如何利用语音特征识别财务风险。工作是验证声学特征在高训练演讲者中的有效性，发现其反而降低模型性能，提出“声学伪装”现象。**

- **链接: [https://arxiv.org/pdf/2604.14619](https://arxiv.org/pdf/2604.14619)**

> **作者:** Dhruvin Dungrani; Disha Dungrani
>
> **摘要:** In computational paralinguistics, detecting cognitive load and deception from speech signals is a heavily researched domain. Recent efforts have attempted to apply these acoustic frameworks to corporate earnings calls to predict catastrophic stock market volatility. In this study, we empirically investigate the limits of acoustic feature extraction (pitch, jitter, and hesitation) when applied to highly trained speakers in in-the-wild teleconference environments. Utilizing a two-stream late-fusion architecture, we contrast an acoustic-based stream with a baseline Natural Language Processing (NLP) stream. The isolated NLP model achieved a recall of 66.25% for tail-risk downside events. Surprisingly, integrating acoustic features via late fusion significantly degraded performance, reducing recall to 47.08%. We identify this degradation as Acoustic Camouflage, where media-trained vocal regulation introduces contradictory noise that disrupts multimodal meta-learners. We present these findings as a boundary condition for speech processing applications in high-stakes financial forecasting.
>
---
#### [new 011] Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection
- **分类: cs.CR; cs.AI; cs.SD**

- **简介: 该论文属于安全任务，旨在解决音频语言模型被恶意音频注入攻击的问题。研究提出AudioHijack框架，生成不可察觉的对抗音频以操控模型行为。**

- **链接: [https://arxiv.org/pdf/2604.14604](https://arxiv.org/pdf/2604.14604)**

> **作者:** Meng Chen; Kun Wang; Li Lu; Jiaheng Zhang; Tianwei Zhang
>
> **备注:** Accepted by IEEE S&P 2026
>
> **摘要:** Modern Large audio-language models (LALMs) power intelligent voice interactions by tightly integrating audio and text. This integration, however, expands the attack surface beyond text and introduces vulnerabilities in the continuous, high-dimensional audio channel. While prior work studied audio jailbreaks, the security risks of malicious audio injection and downstream behavior manipulation remain underexamined. In this work, we reveal a previously overlooked threat, auditory prompt injection, under realistic constraints of audio data-only access and strong perceptual stealth. To systematically analyze this threat, we propose \textit{AudioHijack}, a general framework that generates context-agnostic and imperceptible adversarial audio to hijack LALMs. \textit{AudioHijack} employs sampling-based gradient estimation for end-to-end optimization across diverse models, bypassing non-differentiable audio tokenization. Through attention supervision and multi-context training, it steers model attention toward adversarial audio and generalizes to unseen user contexts. We also design a convolutional blending method that modulates perturbations into natural reverberation, making them highly imperceptible to users. Extensive experiments on 13 state-of-the-art LALMs show consistent hijacking across 6 misbehavior categories, achieving average success rates of 79\%-96\% on unseen user contexts with high acoustic fidelity. Real-world studies demonstrate that commercial voice agents from Mistral AI and Microsoft Azure can be induced to execute unauthorized actions on behalf of users. These findings expose critical vulnerabilities in LALMs and highlight the urgent need for dedicated defense.
>
---
#### [new 012] Enhancing time-frequency resolution with optimal transport and barycentric fusion of multiple spectrogram
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于信号处理任务，旨在解决时间频率分辨率受限的问题。通过最优传输和谱图融合，生成高分辨率时频表示，提升信号分析效果。**

- **链接: [https://arxiv.org/pdf/2604.15055](https://arxiv.org/pdf/2604.15055)**

> **作者:** David Valdivia; Elsa Cazelles; Cédric Févotte
>
> **备注:** main text: 13 pages, 8 figures. supplementary material: 3 pages, 3 figures
>
> **摘要:** Time-frequency representations, such as the short-time Fourier transform (STFT), are fundamental tools for analyzing non-stationary signals. However, their ability to achieve sharp localization in both time and frequency is inherently limited by the Gabor-Heisenberg uncertainty principle. In this paper, we address this limitation by introducing a method to generate super-resolution spectrograms through the fusion of two or more spectrograms with varying resolutions. Specifically, we compute the super-resolution spectrogram as the barycenter of input spectrograms using optimal transport (OT) divergences. Unlike existing fusion approaches, our method does not require the input spectrograms to share the same time-frequency grid. Instead, the input spectrograms can be computed using any STFT parameters, and the resulting super-resolution spectrogram can be defined on an arbitrary user-specified grid. We explore various OT divergences based on different transportation costs. Notably, we introduce a novel transportation cost that preserves time-frequency geometry while significantly reducing computational complexity compared to standard Wasserstein barycenters. We adopt the unbalanced OT framework and derive a new block majorization-minimization algorithm for efficient barycenter computation. We validate the proposed method on controlled synthetic signals and recorded speech using both quantitative and qualitative evaluations. The results show that our approach combines the best localization properties of the input spectrograms and outperforms an unsupervised state-of-the-art fusion method.
>
---
#### [new 013] Geo2Sound: A Scalable Geo-Aligned Framework for Soundscape Generation from Satellite Imagery
- **分类: cs.MM; cs.SD**

- **简介: 该论文提出Geo2Sound任务，解决卫星图像到真实声景生成的问题。通过融合地理属性建模与声学对齐，生成高保真声景。**

- **链接: [https://arxiv.org/pdf/2604.14707](https://arxiv.org/pdf/2604.14707)**

> **作者:** Kunlin Wu; Yanning Wang; Haofeng Tan; Boyi Chen; Teng Fei; Xianping Ma; Yang Yue; Zan Zhou; Xiaofeng Liu
>
> **备注:** 15 pages, 4 figures, 4 tables. Includes supplementary material and SatSound-Bench dataset details
>
> **摘要:** Recent image-to-audio models have shown impressive performance on object-centric visual scenes. However, their application to satellite imagery remains limited by the complex, wide-area semantic ambiguity of top-down views. While satellite imagery provides a uniquely scalable source for global soundscape generation, matching these views to real acoustic environments with unique spatial structures is inherently difficult. To address this challenge, we introduce Geo2Sound, a novel task and framework for generating geographically realistic soundscapes from satellite imagery. Specifically, Geo2Sound combines structural geospatial attributes modeling, semantic hypothesis expansion, and geo-acoustic alignment in a unified framework. A lightweight classifier summarizes overhead scenes into compact geographic attributes, multiple sound-oriented semantic hypotheses are used to generate diverse acoustically plausible candidates, and a geo-acoustic alignment module projects geographic attributes into the acoustic embedding space and identifies the candidate most consistent with the candidate sets. Moreover, we establish SatSound-Bench, the first benchmark comprising over 20k high-quality paired satellite images, text descriptions, and real-world audio recordings, collected from the field across more than 10 countries and complemented by three public datasets. Experiments show that Geo2Sound achieves a SOTA FAD of 1.765, outperforming the strongest baseline by 50.0%. Human evaluations further confirm substantial gains in both realism (26.5%) and semantic alignment, validating our high-fidelity synthesis on scale. Project page and source code: this https URL
>
---
#### [new 014] ControlFoley: Unified and Controllable Video-to-Audio Generation with Cross-Modal Conflict Handling
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出ControlFoley，解决视频到音频生成中的跨模态冲突与可控性问题。通过联合视觉编码、时频解耦和鲁棒训练，提升文本与音频控制精度。**

- **链接: [https://arxiv.org/pdf/2604.15086](https://arxiv.org/pdf/2604.15086)**

> **作者:** Jianxuan Yang; Xinyue Guo; Zhi Cheng; Kai Wang; Lipan Zhang; Jinjie Hu; Qiang Ji; Yihua Cao; Yihao Meng; Zhaoyue Cui; Mengmei Liu; Meng Meng; Jian Luan
>
> **摘要:** Recent advances in video-to-audio (V2A) generation enable high-quality audio synthesis from visual content, yet achieving robust and fine-grained controllability remains challenging. Existing methods suffer from weak textual controllability under visual-text conflict and imprecise stylistic control due to entangled temporal and timbre information in reference audio. Moreover, the lack of standardized benchmarks limits systematic evaluation. We propose ControlFoley, a unified multimodal V2A framework that enables precise control over video, text, and reference audio. We introduce a joint visual encoding paradigm that integrates CLIP with a spatio-temporal audio-visual encoder to improve alignment and textual controllability. We further propose temporal-timbre decoupling to suppress redundant temporal cues while preserving discriminative timbre features. In addition, we design a modality-robust training scheme with unified multimodal representation alignment (REPA) and random modality dropout. We also present VGGSound-TVC, a benchmark for evaluating textual controllability under varying degrees of visual-text conflict. Extensive experiments demonstrate state-of-the-art performance across multiple V2A tasks, including text-guided, text-controlled, and audio-controlled generation. ControlFoley achieves superior controllability under cross-modal conflict while maintaining strong synchronization and audio quality, and shows competitive or better performance compared to an industrial V2A system. Code, models, datasets, and demos are available at: this https URL.
>
---
#### [new 015] From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音代理评估任务，旨在解决现有基准忽视主动干预的问题。提出ProVoice-Bench框架，包含四项新任务，以评估语音代理的主动性。**

- **链接: [https://arxiv.org/pdf/2604.15037](https://arxiv.org/pdf/2604.15037)**

> **作者:** Ke Xu; Yuhao Wang; Yu Wang
>
> **摘要:** Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents.
>
---
#### [new 016] TurboTalk: Progressive Distillation for One-Step Audio-Driven Talking Avatar Generation
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音频驱动的视频数字人生成任务，旨在解决多步去噪模型计算开销大、部署困难的问题。通过两阶段渐进式蒸馏方法，将多步模型压缩为单步生成器，提升推理速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2604.14580](https://arxiv.org/pdf/2604.14580)**

> **作者:** Xiangyu Liu; Feng Gao; Xiaomei Zhang; Yong Zhang; Xiaoming Wei; Zhen Lei; Xiangyu Zhu
>
> **摘要:** Existing audio-driven video digital human generation models rely on multi-step denoising, resulting in substantial computational overhead that severely limits their deployment in real-world settings. While one-step distillation approaches can significantly accelerate inference, they often suffer from training instability. To address this challenge, we propose TurboTalk, a two-stage progressive distillation framework that effectively compresses a multi-step audio-driven video diffusion model into a single-step generator. We first adopt Distribution Matching Distillation to obtain a strong and stable 4-step student, and then progressively reduce the denoising steps from 4 to 1 through adversarial distillation. To ensure stable training under extreme step reduction, we introduce a progressive timestep sampling strategy and a self-compare adversarial objective that provides an intermediate adversarial reference that stabilizes progressive distillation. Our method achieve single-step generation of video talking avatar, boosting inference speed by 120 times while maintaining high generation quality.
>
---
## 更新

#### [replaced 001] A Multimodal Data Fusion Generative Adversarial Network for Real Time Underwater Sound Speed Field Construction
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于水声参数建模任务，旨在无需现场数据实现高精度声速剖面估计。提出MDF-RAGAN模型融合多源数据，提升声速分布建模精度。**

- **链接: [https://arxiv.org/pdf/2507.11812](https://arxiv.org/pdf/2507.11812)**

> **作者:** Wei Huang; Yuqiang Huang; Jixuan Zhou; Fang Ji; Hao Zhang; Tianhe Xu
>
> **摘要:** Sound speed profiles (SSPs) are essential parameters underwater that affects the propagation mode of underwater signals and has a critical impact on the energy efficiency of underwater acoustic communication and accuracy of underwater acoustic positioning. Traditionally, SSPs can be obtained by matching field processing (MFP), compressive sensing (CS), and deep learning (DL) methods. However, existing methods mainly rely on on-site underwater sonar observation data, which put forward strict requirements on the deployment of sonar observation systems. To achieve high-precision estimation of sound velocity distribution in a given sea area without on-site underwater data measurement, we propose a multi-modal data-fusion generative adversarial network model with residual attention block (MDF-RAGAN) for SSP construction. To improve the model's ability for capturing global spatial feature correlations, we embedded the attention mechanisms, and use residual modules for deeply capturing small disturbances in the deep ocean sound velocity distribution caused by changes of SST. Experimental results on real open dataset show that the proposed model outperforms other state-of-the-art methods, which achieves an accuracy with an error of less than 0.3m/s. Specifically, MDF-RAGAN not only outperforms convolutional neural network (CNN) and spatial interpolation (SITP) by nearly a factor of two, but also achieves about 65.8\% root mean square error (RMSE) reduction compared to mean profile, which fully reflects the enhancement of overall profile matching by multi-source fusion and cross-modal attention.
>
---
#### [replaced 002] MARS: Sound Generation via Multi-Channel Autoregression on Spectrograms
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出MARS，用于声音生成任务，解决如何高效生成高质量音频的问题。通过多通道自回归和谱图建模，提升生成音质与细节。**

- **链接: [https://arxiv.org/pdf/2509.26007](https://arxiv.org/pdf/2509.26007)**

> **作者:** Eleonora Ristori; Luca Bindini; Paolo Frasconi
>
> **备注:** Accepted at IJCNN 2026 (to appear in IEEE/IJCNN proceedings). This arXiv submission corresponds to the camera-ready version
>
> **摘要:** Research on audio generation has progressively developed along both waveform-based and spectrogram-based directions, giving rise to diverse strategies for representing and generating audio. At the same time, advances in image synthesis have shown that autoregression across scales, rather than tokens, improves coherence and detail. Building on these ideas, we introduce MARS (Multi-channel AutoRegression on Spectrograms), which, to the best of our knowledge, is the first adaptation of next-scale autoregressive modeling to the spectrogram domain. MARS treats spectrograms as multi-channel images and employs channel multiplexing (CMX), a reshaping strategy that reduces spatial resolution without information loss. A shared tokenizer provides consistent discrete representations across scales, enabling a transformer-based autoregressor to refine spectrograms from coarse to fine resolutions efficiently. Experiments on a large-scale dataset demonstrate that MARS performs comparably or better than state-of-the-art baselines across multiple evaluation metrics, establishing an efficient and scalable paradigm for high-fidelity sound generation.
>
---
#### [replaced 003] A Lightweight Two-Branch Architecture for Multi-Instrument Transcription via Note-Level Contrastive Clustering
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于多乐器音高转录任务，解决现有模型泛化能力差、计算量大等问题。提出轻量双分支架构，通过音符级对比聚类实现动态分离与联合转录。**

- **链接: [https://arxiv.org/pdf/2509.12712](https://arxiv.org/pdf/2509.12712)**

> **作者:** Ruigang Li; Yongxu Zhu
>
> **备注:** Published in TISMIR, Vol. 9, No. 1, pp. 119-130, 2026
>
> **摘要:** Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments, rigid source-count constraints, and high computational demands that hinder deployment on low-resource devices. We address these limitations with a lightweight model that extends a timbre-agnostic transcription backbone with a dedicated timbre encoder and performs deep clustering at the note level, enabling joint transcription and dynamic separation of arbitrary instruments given a specified number of instrument classes. Practical optimizations including spectral normalization, dilated convolutions, and contrastive clustering further improve efficiency and robustness. Despite its small size and fast inference, the model achieves competitive performance with heavier baselines in terms of transcription accuracy and separation quality, and shows promising generalization ability, making it highly suitable for real-world deployment in practical and resource-constrained settings.
>
---
#### [replaced 004] Tora3: Trajectory-Guided Audio-Video Generation with Physical Coherence
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音视频生成任务，旨在解决运动与声音关系不协调的问题。提出Tora3框架，利用物体轨迹增强物理一致性，提升运动真实性和同步性。**

- **链接: [https://arxiv.org/pdf/2604.09057](https://arxiv.org/pdf/2604.09057)**

> **作者:** Junchao Liao; Zhenghao Zhang; Xiangyu Meng; Litao Li; Ziying Zhang; Siyu Zhu; Long Qin; Weizhi Wang
>
> **备注:** 12 pages, 5 tables, 5 figures
>
> **摘要:** Audio-video (AV) generation has recently made strong progress in perceptual quality and multimodal coherence, yet generating content with plausible motion-sound relations remains challenging. Existing methods often produce object motions that are visually unstable and sounds that are only loosely aligned with salient motion or contact events, largely because they lack an explicit motion-aware structure shared by video and audio generation. We present Tora3, a trajectory-guided AV generation framework that improves physical coherence by using object trajectories as a shared kinematic prior. Rather than treating trajectories as a video-only control signal, Tora3 uses them to jointly guide visual motion and acoustic events. Specifically, we design a trajectory-aligned motion representation for video, a kinematic-audio alignment module driven by trajectory-derived second-order kinematic states, and a hybrid flow matching scheme that preserves trajectory fidelity in trajectory-conditioned regions while maintaining local coherence elsewhere. We further curate PAV, a large-scale AV dataset emphasizing motion-relevant patterns with automatically extracted motion annotations. Extensive experiments show that Tora3 improves motion realism, motion-sound synchronization, and overall AV generation quality over strong open-source baselines.
>
---
#### [replaced 005] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决现有方法缺乏解释性和泛化能力的问题。提出SpeechLLM-as-Judges框架，构建SpeechEval数据集，并训练SQ-LLM模型提升评估性能。**

- **链接: [https://arxiv.org/pdf/2510.14664](https://arxiv.org/pdf/2510.14664)**

> **作者:** Hui Wang; Jinghua Zhao; Yifan Yang; Shujie Liu; Junyang Chen; Yanzhe Zhang; Shiwan Zhao; Jinyu Li; Jiaming Zhou; Haoqin Sun; Yan Lu; Yong Qin
>
> **备注:** ACL 2026
>
> **摘要:** Generative speech technologies are progressing rapidly, but evaluating the perceptual quality of synthetic speech remains a core challenge. Existing methods typically rely on scalar scores or binary decisions, which lack interpretability and generalization across tasks and languages. We present SpeechLLM-as-Judges, a new paradigm for enabling large language models (LLMs) to conduct structured and explanation-based speech quality evaluation. To support this direction, we introduce SpeechEval, a large-scale dataset containing 32,207 multilingual speech clips and 128,754 annotations spanning four tasks: quality assessment, pairwise comparison, improvement suggestion, and deepfake detection. Based on this resource, we develop SQ-LLM, a speech-quality-aware LLM trained with chain-of-thought reasoning and reward optimization to improve capability. Experimental results show that SQ-LLM delivers strong performance across tasks and languages, revealing the potential of this paradigm for advancing speech quality evaluation. The relevant code, models, and data are publicly available at this https URL.
>
---
#### [replaced 006] Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于图像到音乐生成任务，旨在解决现有方法缺乏解释性和高计算成本的问题。提出一种基于VLM的可解释框架，利用ABC记谱法和RAG技术生成高质量音乐，并提供解释。**

- **链接: [https://arxiv.org/pdf/2509.22378](https://arxiv.org/pdf/2509.22378)**

> **作者:** Zijian Zhao; Dian Jin; Zijing Zhou
>
> **摘要:** Recently, Image-to-Music (I2M) generation has garnered significant attention, with potential applications in fields such as gaming, advertising, and multi-modal art creation. However, due to the ambiguous and subjective nature of I2M tasks, most end-to-end methods lack interpretability, leaving users puzzled about the generation results. Even methods based on emotion mapping face controversy, as emotion represents only a singular aspect of art. Additionally, most learning-based methods require substantial computational resources and large datasets for training, hindering accessibility for common users. To address these challenges, we propose the first Vision Language Model (VLM)-based I2M framework that offers high interpretability and low computational cost. Specifically, we utilize ABC notation to bridge the text and music modalities, enabling the VLM to generate music using natural language. We then apply multi-modal Retrieval-Augmented Generation (RAG) and self-refinement techniques to allow the VLM to produce high-quality music without external training. Furthermore, we leverage the generated motivations in text and the attention maps from the VLM to provide explanations for the generated results in both text and image modalities. To validate our method, we conduct both human studies and machine evaluations, where our method outperforms others in terms of music quality and music-image consistency, indicating promising results. Our code is available at this https URL .
>
---
#### [replaced 007] RFM-Editing: Rectified Flow Matching for Text-guided Audio Editing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本引导的音频编辑任务，旨在精准修改音频内容并保持其余部分不变。提出一种基于修正流匹配的高效框架，无需辅助信息即可实现语义对齐与高质量编辑。**

- **链接: [https://arxiv.org/pdf/2509.14003](https://arxiv.org/pdf/2509.14003)**

> **作者:** Liting Gao; Yi Yuan; Yaru Chen; Yuelan Cheng; Zhenbo Li; Juan Wen; Shubin Zhang; Wenwu Wang
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Diffusion models have shown remarkable progress in text-to-audio generation. However, text-guided audio editing remains in its early stages. This task focuses on modifying the target content within an audio signal while preserving the rest, thus demanding precise localization and faithful editing according to the text prompt. Existing training-based and zero-shot methods that rely on full-caption or costly optimization often struggle with complex editing or lack practicality. In this work, we propose a novel end-to-end efficient rectified flow matching-based diffusion framework for audio editing, and construct a dataset featuring overlapping multi-event audio to support training and benchmarking in complex scenarios. Experiments show that our model achieves faithful semantic alignment without requiring auxiliary captions or masks, while maintaining competitive editing quality across metrics.
>
---
#### [replaced 008] Gaussian Process Regression of Steering Vectors With Physics-Aware Deep Composite Kernels for Augmented Listening
- **分类: eess.AS; cs.AI; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于声场重建任务，解决传统方法在非理想环境中无法准确表示声场的问题。通过融合物理模型与高斯过程，提出新型核函数提升超分辨率效果。**

- **链接: [https://arxiv.org/pdf/2509.02571](https://arxiv.org/pdf/2509.02571)**

> **作者:** Diego Di Carlo; Shoichi Koyama; Nugraha Aditya Arie; Fontaine Mathieu; Bando Yoshiaki; Yoshii Kazuyoshi
>
> **摘要:** This paper investigates continuous representations of steering vectors over frequency and microphone/source positions for augmented listening (e.g., spatial filtering and binaural rendering), enabling user-parameterized control of the reproduced sound field. Steering vectors have typically been used for representing the spatial response of a microphone array as a function of the look-up direction. The basic algebraic representation of these quantities assuming an idealized environment cannot deal with the scattering effect of the sound field. One may thus collect a discrete set of real steering vectors measured in dedicated facilities and super-resolve (i.e., upsample) them. Recently, physics-aware deep learning methods have been effectively used for this purpose. Such deterministic super-resolution, however, suffers from the overfitting problem due to the non-uniform uncertainty over the measurement space. To solve this problem, we integrate an expressive representation based on the neural field (NF) into the principled probabilistic framework based on the Gaussian process (GP). Specifically, we propose a physics-aware composite kernel that models the directional incoming waves and the subsequent scattering effect. Our comprehensive comparative experiment showed the effectiveness of the proposed method under data insufficiency conditions. In downstream tasks such as speech enhancement and binaural rendering using the simulated data of the SPEAR challenge, the oracle performances were attained with less than ten times fewer measurements.
>
---
#### [replaced 009] Differentiable Acoustic Radiance Transfer
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出DART，一种可微的声辐射传输方法，用于优化材料属性。属于声场建模任务，解决稀疏测量下能量响应预测问题。**

- **链接: [https://arxiv.org/pdf/2509.15946](https://arxiv.org/pdf/2509.15946)**

> **作者:** Sungho Lee; Matteo Scerbo; Seungu Han; Min Jun Choi; Kyogu Lee; Enzo De Sena
>
> **备注:** Accepted to TASLPRO
>
> **摘要:** Geometric acoustics is an efficient framework for room acoustics modeling, governed by the canonical time-dependent rendering equation. Acoustic radiance transfer (ART) solves the equation by discretization, modeling time- and direction-dependent energy exchange between surface patches with flexible material properties. We introduce DART, an efficient, differentiable implementation of ART that enables gradient-based optimization of material properties. We evaluate DART on a simpler variant of acoustic field learning that aims to predict energy responses for novel source-receiver configurations. Experimental results demonstrate that DART generalizes better under sparse measurement scenarios than existing signal processing and neural network baselines, while maintaining simplicity and full interpretability. We open-source our implementation.
>
---
#### [replaced 010] LLMs and Speech: Integration vs. Combination
- **分类: eess.AS**

- **简介: 该论文研究如何将预训练大语言模型用于语音识别，比较了语音LLM与传统浅层融合方法，旨在提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.15045](https://arxiv.org/pdf/2603.15045)**

> **作者:** Robin Schmitt; Albert Zeyer; Mohammad Zeineldeen; Ralf Schlüter; Hermann Ney
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** In this work, we study how to best utilize pre-trained LLMs for automatic speech recognition. Specifically, we compare the tight integration of an acoustic model (AM) with the LLM ("speech LLM") to the traditional way of combining AM and LLM via shallow fusion. For tight integration, we provide ablations on the effect of different label units, fine-tuning strategies, LLM sizes and pre-training data, attention interfaces, encoder downsampling, text prompts, and length normalization. Additionally, we investigate joint recognition with a CTC model to mitigate hallucinations of speech LLMs and present effective optimizations for this joint recognition. For shallow fusion, we investigate the effect of fine-tuning the LLM on the transcriptions using different label units, and we compare rescoring AM hypotheses to single-pass recognition with label-wise or delayed fusion of AM and LLM scores. We train on Librispeech and Loquacious and evaluate our models on the HuggingFace ASR leaderboard.
>
---
#### [replaced 011] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于自然语言处理任务，研究多轮对话中语音模型风格保持问题，揭示风格遗忘现象并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2512.23578](https://arxiv.org/pdf/2512.23578)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** ACL 2026 Findings
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that while SLMs can recall the style instruction when prompted in later turns, they still fail to express it, but through explicit recall can mitigate style amnesia. In addition, SLMs struggle more when the style instruction is placed in system messages rather than user messages, even though system messages are specifically designed to provide persistent, conversation-level instructions. Our findings highlight a systematic gap in current SLMs' ability to maintain speaking styles, highlighting the need for improved style adherence in future models. Our code and evaluation data are publicly available at this https URL.
>
---
