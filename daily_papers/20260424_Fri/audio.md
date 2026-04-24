# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] PHOTON: Non-Invasive Optical Tracking of Key-Lever Motion in Historical Keyboard Instruments
- **分类: eess.AS**

- **简介: 该论文属于乐器分析任务，旨在非侵入式测量历史键盘乐器的键杠杆运动。通过光学传感系统PHOTON，实现高精度、低延迟的运动捕捉与实时MIDI输出。**

- **链接: [https://arxiv.org/pdf/2604.21682](https://arxiv.org/pdf/2604.21682)**

> **作者:** Noah Jaffe; John Ashley Burgoyne
>
> **备注:** NIME 2026
>
> **摘要:** This paper introduces PHOTON (PHysical Optical Tracking of Notes), a non-invasive optical sensing system for measuring key-lever motion in historical keyboard instruments. PHOTON tracks the vertical displacement of the key lever itself, capturing motion shaped by both performer input and the instrument's mechanically imposed, time-varying load. Reflective optical sensors mounted beneath the distal end of each lever provide continuous displacement, timing, and articulation data without interfering with the action. Unlike existing optical systems designed for modern pianos, PHOTON accommodates the diverse geometries, limited clearances, and non-standard layouts of harpsichords, clavichords, and early fortepianos. Its modular, low-profile architecture enables high-resolution, low-latency sensing across multiple manuals and variable key counts. Beyond performance capture, PHOTON provides real-time MIDI output and supports empirical study of expressive gesture, human-instrument interaction, and the construction of instrument-specific MIDI corpora using real historical mechanisms. The complete system is released as open-source hardware and software, from schematics and PCB layouts developed in KiCad to firmware written in CircuitPython, lowering the barrier to adoption, replication, and extension.
>
---
#### [new 002] Time vs. Layer: Locating Predictive Cues for Dysarthric Speech Descriptors in wav2vec 2.0
- **分类: cs.SD**

- **简介: 该论文属于病理语音分析任务，旨在确定Wav2vec 2.0中哪些表示对特定语音特征预测更有效。通过对比层聚合与时间聚合策略，发现不同语音特征依赖不同的表示方式。**

- **链接: [https://arxiv.org/pdf/2604.21628](https://arxiv.org/pdf/2604.21628)**

> **作者:** Natalie Engert; Dominik Wagner; Korbinian Riedhammer; Tobias Bocklet
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Wav2vec 2.0 (W2V2) has shown strong performance in pathological speech analysis by effectively capturing the characteristics of atypical speech. Despite its success, it remains unclear which components of its learned representations are most informative for specific downstream tasks. In this study, we address this question by investigating the regression of dysarthric speech descriptors using annotations from the Speech Accessibility Project dataset. We focus on five descriptors, each addressing a different aspect of speech or voice production: intelligibility, imprecise consonants, inappropriate silences, harsh voice and monoloudness. Speech representations are derived from a W2V2-based feature extractor, and we systematically compare layer-wise and time-wise aggregation strategies using attentive statistics pooling. Our results show that intelligibility is best captured through layer-wise representations, whereas imprecise consonants, harsh voice and monoloudness benefit from time-wise modeling. For inappropriate silences, no clear advantage could be observed for either approach.
>
---
#### [new 003] MAGIC-TTS: Fine-Grained Controllable Speech Synthesis with Explicit Local Duration and Pause Control
- **分类: cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决现有系统缺乏细粒度时长控制的问题。通过引入显式局部时长和停顿控制，提升合成语音的精确性与可编辑性。**

- **链接: [https://arxiv.org/pdf/2604.21164](https://arxiv.org/pdf/2604.21164)**

> **作者:** Jialong Mai; Xiaofen Xing; Xiangmin Xu
>
> **摘要:** Fine-grained local timing control is still absent from modern text-to-speech systems: existing approaches typically provide only utterance-level duration or global speaking-rate control, while precise token-level timing manipulation remains unavailable. To the best of our knowledge, MAGIC-TTS is the first TTS model with explicit local timing control over token-level content duration and pause. MAGIC-TTS is enabled by explicit token-level duration conditioning, carefully prepared high-confidence duration supervision, and training mechanisms that correct zero-value bias and make the model robust to missing local controls. On our timing-control benchmark, MAGIC-TTS substantially improves token-level duration and pause following over spontaneous synthesis. Even when no timing control is provided, MAGIC-TTS maintains natural high-quality synthesis. We further evaluate practical local editing with a scenario-based benchmark covering navigation guidance, guided reading, and accessibility-oriented code reading. In this setting, MAGIC-TTS realizes a reproducible uniform-timing baseline and then moves the edited regions toward the requested local targets with low mean bias. These results show that explicit fine-grained controllability can be implemented effectively in a high-quality TTS system and can support realistic local timing-editing applications.
>
---
#### [new 004] Full-Duplex Interaction in Spoken Dialogue Systems: A Comprehensive Study from the ICASSP 2026 HumDial Challenge
- **分类: eess.AS**

- **简介: 该论文属于语音对话系统任务，旨在解决传统系统在全双工交互中的不足。通过构建基准数据集和评估框架，提升系统处理实时干扰和动态对话的能力。**

- **链接: [https://arxiv.org/pdf/2604.21406](https://arxiv.org/pdf/2604.21406)**

> **作者:** Chengyou Wang; Hongfei Yue; Guojian Li; Zhixian Zhao; Shuiyuan Wang; Shuai Wang; Xin Xu; Hui Bu; Lei Xie
>
> **备注:** 5 pages, 1 figures
>
> **摘要:** Full-duplex interaction, where speakers and listeners converse simultaneously, is a key element of human communication often missing from traditional spoken dialogue systems. These systems, based on rigid turn-taking paradigms, struggle to respond naturally in dynamic conversations. The Full-Duplex Interaction Track of ICASSP 2026 Human-like Spoken Dialogue Systems Challenge (HumDial Challenge) aims to advance the evaluation of full-duplex systems by offering a framework for handling real-time interruptions, speech overlap, and dynamic turn negotiation. We introduce a comprehensive benchmark for full-duplex spoken dialogue systems, built from the HumDial Challenge. We release a high-quality dual-channel dataset of real human-recorded conversations, capturing interruptions, overlapping speech, and feedback mechanisms. This dataset forms the basis for the HumDial-FDBench benchmark, which assesses a system's ability to handle interruptions while maintaining conversational flow. Additionally, we create a public leaderboard to compare the performance of open-source and proprietary models, promoting transparent, reproducible evaluation. These resources support the development of more responsive, adaptive, and human-like dialogue systems.
>
---
#### [new 005] DiariZen Explained: A Tutorial for the Open Source State-of-the-Art Speaker Diarization Pipeline
- **分类: eess.AS; cs.SD**

- **简介: 该论文介绍DiariZen，一个用于说话人辨识的开源系统，解决多说话人音频中“谁在何时说话”的问题。论文详细解析了其七阶段架构，便于理解和应用。**

- **链接: [https://arxiv.org/pdf/2604.21507](https://arxiv.org/pdf/2604.21507)**

> **作者:** Nikhil Raghav
>
> **备注:** 13 pages, 7 figures, 2 tables. Code available at this https URL
>
> **摘要:** Speaker diarization (SD) is the task of answering "who spoke when" in a multi-speaker audio stream. Classically, an SD system clusters segments of speech belonging to an individual speaker's identity. Recent years have seen substantial progress in SD through end-to-end neural diarization (EEND) approaches. DiariZen, a hybrid SD pipeline built upon a structurally pruned WavLM-Large encoder, a Conformer backend with powerset classification, and VBx clustering, represents the leading open-source state of the art at the time of writing across multiple benchmarks. Despite its strong performance, the DiariZen architecture spans several repositories and frameworks, making it difficult for researchers and practitioners to understand, reproduce, or extend the system as a whole. This tutorial paper provides a self-contained, block-by-block explanation of the complete DiariZen pipeline, decomposing it into seven stages: (1) audio loading and sliding window segmentation, (2) WavLM feature extraction with learned layer weighting, (3) Conformer backend and powerset classification, (4) segmentation aggregation via overlap-add, (5) speaker embedding extraction with overlap exclusion, (6) VBx clustering with PLDA scoring, and (7) reconstruction and RTTM output. For each block, we provide the conceptual motivation, source code references, intermediate tensor shapes, and annotated visualizations of the actual outputs on a 30s excerpt from the AMI Meeting Corpus. The implementation is available at this https URL, which includes standalone executable scripts for each block and a Jupyter notebook that runs the complete pipeline end-to-end.
>
---
#### [new 006] Beyond Rules: Towards Basso Continuo Personal Style Identification
- **分类: cs.SD**

- **简介: 该论文属于音乐风格识别任务，旨在解决如何从巴洛克通奏低音演奏中识别个人风格的问题。通过分析ACoRD数据集，使用griffs和SVM进行分类，验证了个人风格的存在。**

- **链接: [https://arxiv.org/pdf/2604.21822](https://arxiv.org/pdf/2604.21822)**

> **作者:** Adam Štefunko; Jan Hajič jr
>
> **备注:** 8 pages, 4 figures, accepted to the 13th International Conference on Digital Libraries for Musicology (DLfM)
>
> **摘要:** A central part of the contemporary Historically Informed Practice movement is basso continuo, an improvised accompaniment genre with its traditions originating in the baroque era and actively practiced by many keyboard players nowadays. Although computational musicology has studied the theoretical foundations of basso continuo expressed by harmonic and voice-leading rules and constraints, characteristics of basso continuo as an active performing art have been largely overlooked mostly due to a lack of suitable performance data that could be empirically analyzed. This has changed with the introduction of The Aligned Continuo Realization Dataset (ACoRD) and the basso continuo realization-to-score alignment. Basso continuo playing is shaped by stylistic traditions coming from historical treatises, but it also may provide space for showcasing individual performance styles of its practitioners. In this paper, we attempt to explore the question of the presence of personal styles in the basso continuo realizations of players in the ACoRD dataset. We use a historically informed structured representation of basso continuo performance pitch content called griffs and Support Vector Machines to see whether it is possible to classify players based on their performances. The results show that we can identify players from their performances. In addition to the player classification problem, we discuss the elements that make up the individual styles of the players.
>
---
#### [new 007] HHL with a Coherent Fourier Oracle: A Proof-of-Concept Quantum Architecture for Joint Melody-Harmony Generation
- **分类: quant-ph; cs.AI; cs.SD**

- **简介: 该论文将HHL算法应用于音乐生成任务，解决量子计算在音乐创作中的应用问题。通过构建相干傅里叶和声预言机，实现旋律与和声的联合生成。**

- **链接: [https://arxiv.org/pdf/2604.20882](https://arxiv.org/pdf/2604.20882)**

> **作者:** Alexis Kirke
>
> **摘要:** Quantum algorithms with a proven theoretical speedup over classical computation are rare. Among the most prominent is the Harrow-Hassidim-Lloyd (HHL) algorithm for solving sparse linear systems. Here, HHL is applied to encode melodic preference: the system matrix encodes Narmour implication-realisation and Krumhansl-Kessler tonal stability, so its solution vector is a music-cognition-weighted note-pair distribution. The key constraint of HHL is that reading its output classically cancels the quantum speedup; the solution must be consumed coherently. This motivates a coherent Fourier harmonic oracle: a unitary that applies chord-transition weights directly to the HHL amplitude vector, so that a single measurement jointly selects both melody notes and a two-chord progression. A two-note/two-chord (2/2) block is used to contain the exponential growth of the joint state space that would otherwise make classical simulation of larger blocks infeasible. For demonstrations of longer passages, blocks are chained classically - each block's collapsed output conditions the next -- as a temporary workaround until fault-tolerant hardware permits larger monolithic circuits. A four-block chain produces 8 notes over 8 chords with grammatically valid transitions at every block boundary. Independent rule-based harmony validation confirms that 97% of generated chord progressions are rated strong or acceptable. The primary motivation is that HHL carries a proven exponential speedup over classical linear solvers; this work demonstrates that a coherent HHL+oracle pipeline - the prerequisite for that speedup to be realised in a musical setting - is mechanically achievable. Audio realisations of representative outputs are made available for listening online.
>
---
#### [new 008] Materialistic RIR: Material Conditioned Realistic RIR Generation
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属于声学建模任务，旨在解决空间与材料影响纠缠的问题。通过分离空间与材料模块，实现材料可控的房间脉冲响应生成，提升音质真实性和可控制性。**

- **链接: [https://arxiv.org/pdf/2604.21119](https://arxiv.org/pdf/2604.21119)**

> **作者:** Mahnoor Fatima Saad; Sagnik Majumder; Kristen Grauman; Ziad Al-Halah
>
> **备注:** Accepted to CVPR 2026 Findings. Project page: this https URL
>
> **摘要:** Rings like gold, thuds like wood! The sound we hear in a scene is shaped not only by the spatial layout of the environment but also by the materials of the objects and surfaces within it. For instance, a room with wooden walls will produce a different acoustic experience from a room with the same spatial layout but concrete walls. Accurately modeling these effects is essential for applications such as virtual reality, robotics, architectural design, and audio engineering. Yet, existing methods for acoustic modeling often entangle spatial and material influences in correlated representations, which limits user control and reduces the realism of the generated acoustics. In this work, we present a novel approach for material-controlled Room Impulse Response (RIR) generation that explicitly disentangles the effects of spatial and material cues in a scene. Our approach models the RIR using two modules: a spatial module that captures the influence of the spatial layout of the scene, and a material module that modulates this spatial RIR according to a user-specified material configuration. This explicitly disentangled design allows users to easily modify the material configuration of a scene and observe its impact on acoustics without altering the spatial structure or scene content. Our model provides significant improvements over prior approaches on both acoustic-based metrics (up to +16% on RTE) and material-based metrics (up to +70%). Furthermore, through a human perceptual study, we demonstrate the improved realism and material sensitivity of our model compared to the strongest baselines.
>
---
#### [new 009] Dilated CNNs for Periodic Signal Processing: A Low-Complexity Approach
- **分类: cs.LG; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于信号处理任务，解决周期信号去噪与波形估计问题。提出R-DCNN方法，在低计算复杂度下实现高效准确的信号处理。**

- **链接: [https://arxiv.org/pdf/2604.21651](https://arxiv.org/pdf/2604.21651)**

> **作者:** Eli Gildish; Michael Grebshtein; Igor Makienko
>
> **备注:** 16 pages, 8 figures, the use of deep learning in IoT devices
>
> **摘要:** Denoising of periodic signals and accurate waveform estimation are core tasks across many signal processing domains, including speech, music, medical diagnostics, radio, and sonar. Although deep learning methods have recently shown performance improvements over classical approaches, they require substantial computational resources and are usually trained separately for each signal observation. This study proposes a computationally efficient method based on DCNN and Re-sampling, termed R-DCNN, designed for operation under strict power and resource constraints. The approach targets signals with varying fundamental frequencies and requires only a single observation for training. It generalizes to additional signals via a lightweight resampling step that aligns time scales in signals with different frequencies to re-use the same network weights. Despite its low computational complexity, R-DCNN achieves performance comparable to state-of-the-art classical methods, such as autoregressive (AR)-based techniques, as well as conventional DCNNs trained individually for each observation. This combination of efficiency and performance makes the proposed method particularly well suited for deployment in resource-constrained environments without sacrificing denoising or estimation accuracy.
>
---
#### [new 010] Do LLM Decoders Listen Fairly? Benchmarking How Language Model Priors Shape Bias in Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，研究预训练语言模型对不同群体识别公平性的影响。通过实验分析模型在多种条件下的表现，探讨音频编码与模型规模对公平性的作用。**

- **链接: [https://arxiv.org/pdf/2604.21276](https://arxiv.org/pdf/2604.21276)**

> **作者:** Srishti Ginjala; Eric Fosler-Lussier; Christopher W. Myers; Srinivasan Parthasarathy
>
> **摘要:** As pretrained large language models replace task-specific decoders in speech recognition, a critical question arises: do their text-derived priors make recognition fairer or more biased across demographic groups? We evaluate nine models spanning three architectural generations (CTC with no language model, encoder-decoder with an implicit LM, and LLM-based with an explicit pretrained decoder) on about 43,000 utterances across five demographic axes (ethnicity, accent, gender, age, first language) using Common Voice 24 and Meta's Fair-Speech, a controlled-prompt dataset that eliminates vocabulary confounds. On clean audio, three findings challenge assumptions: LLM decoders do not amplify racial bias (Granite-8B has the best ethnicity fairness, max/min WER = 2.28); Whisper exhibits pathological hallucination on Indian-accented speech with a non-monotonic insertion-rate spike to 9.62% at large-v3; and audio compression predicts accent fairness more than LLM scale. We then stress-test these findings under 12 acoustic degradation conditions (noise, reverberation, silence injection, chunk masking) across both datasets, totaling 216 inference runs. Severe degradation paradoxically compresses fairness gaps as all groups converge to high WER, but silence injection amplifies Whisper's accent bias up to 4.64x by triggering demographic-selective hallucination. Under masking, Whisper enters catastrophic repetition loops (86% of 51,797 insertions) while explicit-LLM decoders produce 38x fewer insertions with near-zero repetition; high-compression audio encoding (Q-former) reintroduces repetition pathology even in LLM decoders. These results suggest that audio encoder design, not LLM scaling, is the primary lever for equitable and robust speech recognition.
>
---
#### [new 011] Sema: Semantic Transport for Real-Time Multimodal Agents
- **分类: cs.MM; cs.NI; cs.SD**

- **简介: 该论文提出Sema系统，解决实时多模态代理的传输效率问题。通过语义传输替代传统信号传输，显著降低带宽占用并保持任务准确性。**

- **链接: [https://arxiv.org/pdf/2604.20940](https://arxiv.org/pdf/2604.20940)**

> **作者:** Jiaying Meng; Bojie Li
>
> **摘要:** Real-time multimodal agents transport raw audio and screenshots using networking stacks designed for human receivers, which optimize for perceptual fidelity and smooth playout. Yet agent models act as event-driven processors with no inherent sense of physical time, consuming task-relevant semantics rather than reconstructing signals in real time. This fundamental difference shifts the transport goal from the technical problem of signal fidelity (Shannon-Weaver Level A) to the semantic problem of meaning preservation (Level B). This mismatch imposes significant overhead. In visual pipelines, screenshot upload accounts for over 60% of end-to-end action latency on constrained uplinks, and in voice pipelines, conventional transport carries massive redundancy, sending 43-64x more data than needed to maintain task accuracy. We present Sema, a semantic transport system that combines discrete audio tokenizers with a hybrid screen representation (lossless accessibility-tree or OCR text, plus compact visual tokens) and bursty token delivery that eliminates jitter buffers. In simulations under emulated WAN conditions, Sema reduces uplink bandwidth by 64x for audio and 130-210x for screenshots while preserving task accuracy within 0.7 percentage points of the raw baseline.
>
---
## 更新

#### [replaced 001] From Image to Music Language: A Two-Stage Structure Decoding Approach for Complex Polyphonic OMR
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于光学音乐识别（OMR）任务，解决复杂多声部乐谱的结构解码问题。通过两阶段流程，提出BeadSolver方法提升音符结构准确性。**

- **链接: [https://arxiv.org/pdf/2604.20522](https://arxiv.org/pdf/2604.20522)**

> **作者:** Nan Xu; Shiheng Li; Shengchao Hou
>
> **备注:** 49 pages, 16 figures, 16 tables
>
> **摘要:** We propose a new approach for a practical two-stage Optical Music Recognition (OMR) pipeline, with a particular focus on its second stage. Given symbol and event candidates from the visual pipeline, we decode them into an editable, verifiable, and exportable score structure. We focus on complex polyphonic staff notation, especially piano scores, where voice separation and intra-measure timing are the main bottlenecks. Our approach formulates second-stage decoding as a structure decoding problem and uses topology recognition with probability-guided search (BeadSolver) as its core method. We also describe a data strategy that combines procedural generation with recognition-feedback annotations. The result is a practical decoding component for real OMR systems and a path to accumulate structured score data for future end-to-end, multimodal, and RL-style methods.
>
---
#### [replaced 002] FGAS: Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出FGAS，一种基于固定解码器的音频隐写方法，旨在解决现有方法计算成本高、抗分析性能差的问题。通过对抗扰动嵌入秘密信息，提升隐写音质和安全性。**

- **链接: [https://arxiv.org/pdf/2505.22266](https://arxiv.org/pdf/2505.22266)**

> **作者:** Jialin Yan; Yu Cheng; Zhaoxia Yin; Xinpeng Zhang; Shilin Wang; Tanfeng Sun; Xinghao Jiang
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) has made high-fidelity generated audio widely available across the Internet, driving the advancement of audio steganography. Benefiting from advances in deep learning, current audio steganography schemes are mainly based on encoder-decoder network architectures. While these methods guarantee a certain level of perceptual quality for stego audio, they typically face high computational cost and long implementation time, as well as poor anti-steganalysis performance. To address the aforementioned issues, we pioneer a Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation (FGAS). Adversarial perturbations carrying a secret message are embedded into the cover audio to generate stego audio. The receiver only needs to share the structure and key of the fixed decoder network to accurately extract the secret message from the stego audio. In FGAS, we propose an Audio Adversarial Perturbation Generation (A2PG) strategy with an optional robust extension and design a lightweight fixed decoder. The fixed decoder guarantees reliable extraction of the hidden message, while adversarial perturbations are optimized to keep the stego audio perceptually and statistically close to the cover audio, thereby improving anti-steganalysis performance. The experimental results show that FGAS significantly improves stego audio quality, achieving an average PSNR gain of over 10 dB compared to SOTA methods. Furthermore, FGAS demonstrates strong robustness against common audio processing attacks. Moreover, FGAS exhibits superior anti-steganalysis performance across different relative payloads; under high-capacity embedding, it achieves a classification error rate about 2% higher, indicating stronger anti-steganalysis performance than current SOTA methods.
>
---
#### [replaced 003] Dementia classification from spontaneous speech using wrapper-based feature selection
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于 dementia 分类任务，旨在通过自发言语识别认知缺陷。研究使用特征选择方法分析语音数据，提升分类效率与准确性。**

- **链接: [https://arxiv.org/pdf/2502.03484](https://arxiv.org/pdf/2502.03484)**

> **作者:** Marko Niemelä; Mikaela von Bonsdorff; Sami Äyrämö; Tommi Kärkkäinen
>
> **摘要:** Dementia encompasses a group of syndromes that impair cognitive functions such as memory, reasoning, and the ability to perform daily activities. As populations globally age, over 10 million new dementia diagnoses are reported annually. Currently, clinical diagnosis of dementia remains challenging due to overlapping symptoms, the need to exclude alternative conditions and the requirement for a comprehensive clinical evaluation and cognitive assessment. This underscores the growing need to develop feasible and accurate methods for detecting cognitive deficiencies. Recent advances in machine learning have highlighted spontaneous speech as a promising noninvasive, cost-effective, and scalable biomarker for dementia detection. In this study, spontaneous speech recordings from the ADReSS and Pitt Corpus datasets are analyzed, consisting of picture description tasks performed by cognitively healthy individuals and people with Alzheimer's disease. Unlike prior approaches that focus solely on speech-active segments, acoustic features are extracted from entire recordings using the openSMILE toolkit. This representation reduces the number of feature vectors and improves computational efficiency without compromising classification performance. Classification models with classifier-based wrapper feature selection are employed to estimate feature importance and identify diagnostically relevant acoustic characteristics. Among the evaluated models, the Extreme Minimal Learning Machine achieved competitive classification accuracy with substantially lower computational cost, reflecting an inherent property of the model formulation and learning procedure. Overall, the results demonstrate that the proposed framework is computationally efficient, interpretable, and well suited as a supportive tool for speech-based dementia assessment.
>
---
#### [replaced 004] ATRIE: Adaptive Tuning for Robust Inference and Emotion in Persona-Driven Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决角色语音在不同情绪下保持一致性格特征的问题。提出ATRIE框架，通过双轨架构实现稳定身份识别和丰富情感表达。**

- **链接: [https://arxiv.org/pdf/2604.19055](https://arxiv.org/pdf/2604.19055)**

> **作者:** Aoduo Li; Haoran Lv; Hongjian Xu; Shengmin Li; Sihao Qin; Zimeng Li; Chi Man Pun; Xuhang Chen
>
> **备注:** 10 pages, 6 figures. Accepted to ACM ICMR 2026
>
> **摘要:** High-fidelity character voice synthesis is a cornerstone of immersive multimedia applications, particularly for interacting with anime avatars and digital humans. However, existing systems struggle to maintain consistent persona traits across diverse emotional contexts. To bridge this gap, we present ATRIE, a unified framework utilizing a Persona-Prosody Dual-Track (P2-DT) architecture. Our system disentangles generation into a static Timbre Track (via Scalar Quantization) and a dynamic Prosody Track (via Hierarchical Flow-Matching), distilled from a 14B LLM teacher. This design enables robust identity preservation (Zero-Shot Speaker Verification EER: 0.04) and rich emotional expression. Evaluated on our extended AnimeTTS-Bench (50 characters), ATRIE achieves state-of-the-art performance in both generation and cross-modal retrieval (mAP: 0.75), establishing a new paradigm for persona-driven multimedia content creation.
>
---
#### [replaced 005] Diff-VS: Efficient Audio-Aware Diffusion U-Net for Vocals Separation
- **分类: eess.AS**

- **简介: 该论文属于语音分离任务，旨在解决生成方法在音乐源分离中表现不佳的问题。提出基于EDM框架的高效音频感知扩散U-Net模型，提升分离效果与主观质量。**

- **链接: [https://arxiv.org/pdf/2604.01120](https://arxiv.org/pdf/2604.01120)**

> **作者:** Yun-Ning; Hung; Richard Vogl; Filip Korzeniowski; Igor Pereira
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** While diffusion models are best known for their performance in generative tasks, they have also been successfully applied to many other tasks, including audio source separation. However, current generative approaches to music source separation often underperform on standard objective metrics. In this paper, we address this issue by introducing a novel generative vocal separation model based on the Elucidated Diffusion Model (EDM) framework. Our model processes complex short-time Fourier transform spectrograms and employs an improved U-Net architecture based on music-informed design choices. Our approach matches discriminative baselines on objective metrics and achieves perceptual quality comparable to state-of-the-art systems, as assessed by proxy subjective metrics. We hope these results encourage broader exploration of generative methods for music source separation
>
---
#### [replaced 006] Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，研究如何从原始语音中无监督学习语法基本操作——连接。工作包括提出自发连接现象，并验证其在不同模型中的普遍性。**

- **链接: [https://arxiv.org/pdf/2305.01626](https://arxiv.org/pdf/2305.01626)**

> **作者:** Gašper Beguš; Thomas Lu; Zili Wang
>
> **摘要:** Computational models of syntax are predominantly text-based. Here we propose that the most basic first step in the evolution of syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and elementary suboperations of syntax -- concatenation. We introduce \textit{spontaneous concatenation}: a phenomenon where a ciwGAN/fiwGAN models (based on convolutional neural networks) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the training data. We replicate this finding in several independently trained models with different hyperparameters and training data. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. We also show that the concatenated outputs contain precursors to compositionality. To our knowledge, this is a previously unreported property of CNNs trained in the ciwGAN/fiwGAN setting on raw speech and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution in the brain from raw acoustic inputs. We also propose and formalize a neural mechanism called \textit{disinhibition} that outlines a possible artificial and biological neural pathway towards concatenation and compositionality and suggests our modeling is useful for generating testable predictions for biological and artificial neural processing of spoken language.
>
---
#### [replaced 007] A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，研究预训练数据选择策略。针对SSL模型依赖大量数据的问题，通过实验发现优先选择长语音片段能提升ASR效果，减少训练时间。**

- **链接: [https://arxiv.org/pdf/2601.20896](https://arxiv.org/pdf/2601.20896)**

> **作者:** Ryan Whetten; Titouan Parcollet; Marco Dinarelli; Yannick Estève
>
> **备注:** Accepted for publication in the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)
>
> **摘要:** Self-supervised learning (SSL) has transformed speech processing, yet its reliance on massive pre-training datasets remains a bottleneck. While robustness is often attributed to scale and diversity, the role of the data distribution is less understood. We systematically examine how curated subsets of pre-training data influence Automatic Speech Recognition (ASR) performance. Surprisingly, optimizing for acoustic, speaker, or linguistic diversity yields no clear improvements over random sampling. Instead, we find that prioritizing the longest utterances achieves superior ASR results while using only half the original dataset, reducing pre-training time by 24% on a large corpora. These findings suggest that for pre-training speech SSL models, data length is a more critical factor than either data diversity or overall data quantity for performance and efficiency, offering a new perspective for data selection strategies in SSL speech processing.
>
---
#### [replaced 008] Prosody as Supervision: Bridging the Non-Verbal--Verbal for Multilingual Speech Emotion Recognition
- **分类: eess.AS**

- **简介: 该论文属于多语言语音情感识别任务，解决低资源环境下情感识别效果差的问题。提出NOVA-ARC框架，利用非语言语音作为监督，实现非语言到语言的情感迁移。**

- **链接: [https://arxiv.org/pdf/2604.17647](https://arxiv.org/pdf/2604.17647)**

> **作者:** Girish; Mohd Mujtaba Akhtar; Muskaan Singh
>
> **备注:** Accepted to ACL 2026 (Main)
>
> **摘要:** In this work, we introduce a paralinguistic supervision paradigm for low-resource multilingual speech emotion recognition (LRM-SER) that leverages non-verbal vocalizations to exploit prosody-centric emotion cues. Unlike conventional SER systems that rely heavily on labeled verbal speech and suffer from poor cross-lingual transfer, our approach reformulates LRM-SER as non-verbal-to-verbal transfer, where supervision from a labeled non-verbal source domain is adapted to unlabeled verbal speech across multiple target languages. To this end, we propose NOVA ARC, a geometry-aware framework that models affective structure in the Poincaré ball, discretizes paralinguistic patterns via a hyperbolic vector-quantized prosody codebook, and captures emotion intensity through a hyperbolic emotion lens. For unsupervised adaptation, NOVA-ARC performs optimal transport based prototype alignment between source emotion prototypes and target utterances, inducing soft supervision for unlabeled speech while being stabilized through consistency regularization. Experiments show that NOVA-ARC delivers the strongest performance under both non-verbal-to-verbal adaptation and the complementary verbal-to-verbal transfer setting, consistently outperforming Euclidean counterparts and strong SSL baselines. To the best of our knowledge, this work is the first to move beyond verbal-speech-centric supervision by introducing a non-verbal-to-verbal transfer paradigm for SER.
>
---
#### [replaced 009] Video-Robin: Autoregressive Diffusion Planning for Intent-Grounded Video-to-Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视频到音乐生成任务，旨在解决现有模型在语义控制和音频质量上的不足。提出Video-Robin，结合自回归规划与扩散合成，提升生成音乐的语义对齐和质量。**

- **链接: [https://arxiv.org/pdf/2604.17656](https://arxiv.org/pdf/2604.17656)**

> **作者:** Vaibhavi Lokegaonkar; Aryan Vijay Bhosale; Vishnu Raj; Gouthaman KV; Ramani Duraiswami; Lie Lu; Sreyan Ghosh; Dinesh Manocha
>
> **摘要:** Video-to-music (V2M) is the fundamental task of creating background music for an input video. Recent V2M models achieve audiovisual alignment by typically relying on visual conditioning alone and provide limited semantic and stylistic controllability to the end user. In this paper, we present Video-Robin, a novel text-conditioned video-to-music generation model that enables fast, high-quality, semantically aligned music generation for video content. To balance musical fidelity and semantic understanding, Video-Robin integrates autoregressive planning with diffusion-based synthesis. Specifically, an autoregressive module models global structure by semantically aligning visual and textual inputs to produce high-level music latents. These latents are subsequently refined into coherent, high-fidelity music using local Diffusion Transformers. By factoring semantically driven planning into diffusion-based synthesis, Video-Robin enables fine-grained creator control without sacrificing audio realism. Our proposed model outperforms baselines that solely accept video input and additional feature conditioned baselines on both in-distribution and out-of-distribution benchmarks with a 2.21x speed in inference compared to SOTA. We will open-source everything upon paper acceptance.
>
---
#### [replaced 010] Musical Score Understanding Benchmark: Evaluating Large Language Models' Comprehension of Complete Musical Scores
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出MSU-Bench基准，用于评估大语言模型对完整乐谱的理解能力。任务属于多模态理解，解决模型在音高、节奏等音乐要素上的综合推理问题。工作包括构建基准数据集并测试多种模型表现。**

- **链接: [https://arxiv.org/pdf/2511.20697](https://arxiv.org/pdf/2511.20697)**

> **作者:** Congren Dai; Yue Yang; Krinos Li; Huichi Zhou; Shijie Liang; Bo Zhang; Enyang Liu; Ge Jin; Hongran An; Haosen Zhang; Peiyuan Jing; Kinhei Lee; Z henxuan Zhang; Xiaobing Li; Maosong Sun
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Understanding complete musical scores entails integrated reasoning over pitch, rhythm, harmony, and large-scale structure, yet the ability of Large Language Models and Vision--Language Models to interpret full musical notation remains insufficiently examined. We introduce Musical Score Understanding Benchmark (MSU-Bench), a human-curated benchmark for score-level musical understanding across textual (ABC notation) and visual (PDF) modalities. MSU-Bench contains 1,800 generative question-answer pairs from works by Bach, Beethoven, Chopin, Debussy, and others, organised into four levels of increasing difficulty, ranging from onset information to texture and form. Evaluations of more than fifteen state-of-the-art models, in both zero-shot and fine-tuned settings, reveal pronounced modality gaps, unstable level-wise performance, and challenges in maintaining multilevel correctness. Fine-tuning substantially improves results across modalities while preserving general knowledge, positioning MSU-Bench as a robust foundation for future research in multimodal reasoning. The benchmark and code are available at this https URL.
>
---
