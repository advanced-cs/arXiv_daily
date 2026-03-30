# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] A Human-Inspired Decoupled Architecture for Efficient Audio Representation Learning
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于音频表示学习任务，旨在解决Transformer模型参数过多和计算成本高的问题。提出HEAR架构，通过解耦处理流程实现高效音频建模。**

- **链接: [https://arxiv.org/pdf/2603.26098](https://arxiv.org/pdf/2603.26098)**

> **作者:** Harunori Kawano; Takeshi Sasaki
>
> **摘要:** While self-supervised learning (SSL) has revolutionized audio representation, the excessive parameterization and quadratic computational cost of standard Transformers limit their deployment on resource-constrained devices. To address this bottleneck, we propose HEAR (Human-inspired Efficient Audio Representation), a novel decoupled architecture. Inspired by the human cognitive ability to isolate local acoustic features from global context, HEAR splits the processing pipeline into two dedicated modules: an Acoustic Model for local feature extraction and a Task Model for global semantic integration. Coupled with an Acoustic Tokenizer trained via knowledge distillation, our approach enables robust Masked Audio Modeling (MAM). Extensive experiments demonstrate that HEAR requires only 15M parameters and 9.47 GFLOPs for inference, operating at a fraction of the computational cost of conventional foundation models (which typically require 85M-94M parameters). Despite this high efficiency, HEAR achieves highly competitive performance across diverse audio classification benchmarks. The code and pre-trained models are available at this https URL
>
---
#### [new 002] CA-TCN: A Causal-Anticausal Temporal Convolutional Network for Direct Auditory Attention Decoding
- **分类: cs.SD**

- **简介: 该论文属于 auditory attention decoding 任务，旨在准确识别听觉注意力目标。提出 CA-TCN 模型，通过因果与反因果卷积对齐声音刺激和神经响应，提升解码精度。**

- **链接: [https://arxiv.org/pdf/2603.26394](https://arxiv.org/pdf/2603.26394)**

> **作者:** Iñigo García-Ugarte; Rubén Eguinoa; Ricardo San Martín; Daniel Paternain; Carmen Vidaurre
>
> **摘要:** A promising approach for steering auditory attention in complex listening environments relies on Auditory Attention Decoding (AAD), which aim to identify the attended speech stream in a multiple speaker scenario from neural recordings. Entrainment-based AAD approaches, typically assume access to clean speech sources and electroencephalography (EEG) signals to exploit low-frequency correlations between the neural response and the attended stimulus. In this study, we propose CA-TCN, a Causal-Anticausal Temporal Convolutional Network that directly classifies the attended speaker. The proposed architecture integrates several best practices from convolutional neural networks in sequence processing tasks. Importantly, it explicitly aligns auditory stimuli and neural responses by employing separate causal and anticausal convolutions respectively, with distinct receptive fields operating in opposite temporal directions. Experimental results, obtained through comparisons with three baseline AAD models, demonstrated that CA-TCN consistently improved decoding accuracy across datasets and decision windows, with gains ranging from 0.5% to 3.2% for subject-independent models and from 0.8% to 2.9% for subject-specific models compared with the next best-performing model, AADNet. Moreover, these improvements were statistically significant in four of the six evaluated settings when comparing Minimum Expected Switch Duration distributions. Beyond accuracy, the model demonstrated spatial robustness across different conditions, as the EEG spatial filters exhibited stable patterns across datasets. Overall, this work introduces an accurate and unified AAD model that outperforms existing methods while considering practical benefits for online processing scenarios. These findings contribute to advancing the state of AAD and its applicability in real-world systems.
>
---
#### [new 003] UPV_RIR_DB: A Structured Room Impulse Response Database with Hierarchical Metadata and Acoustic Indicators
- **分类: eess.AS**

- **简介: 该论文介绍UPV_RIR_DB，一个结构化房间冲激响应数据库，用于提供带有空间元数据的声学数据。旨在解决真实声场数据的存储与重用问题，通过标准化组织和元数据实现可重复分析。**

- **链接: [https://arxiv.org/pdf/2603.25947](https://arxiv.org/pdf/2603.25947)**

> **作者:** Jesús García-Gamborino; Laura Fuster; Daniel de la Prida; Luis A. Azpicueta-Ruiz; Gema Piñero
>
> **备注:** RIR Database available at ZENODO
>
> **摘要:** This paper presents UPV_RIR_DB, a structured database of measured room impulse responses (RIRs) designed to provide acoustic data with explicit spatial metadata and traceable acquisition parameters. The dataset currently contains 166 multichannel RIR files measured in three rooms of the Universitat Politècnica de València (UPV). Each multichannel RIR file contains impulse responses for multiple source-receiver pairs, with each pair covering a 25 cm2 area - the typical size of a personal sound zone. Considering the number of sources and receiver channels associated with each microphone modality, the database contains a total of 18,976 single impulse responses. A hierarchical organization is adopted in which directory structure and metadata jointly describe the measurement context. Each room includes a metadata file containing acquisition parameters, hardware description, spatial coordinates of zones and microphones, and acoustic indicators such as reverberation time. A central index links each RIR file with its experimental context, ensuring traceability and enabling reproducible analysis. The resulting database provides a consistent framework for storing, inspecting, and reusing real RIR measurements while preserving compatibility with both MATLAB- and JSON-based workflows. The UPV_RIR_DB dataset is publicly available through the open repository Zenodo.
>
---
#### [new 004] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频预训练任务，旨在解决标签质量差、覆盖有限的问题。通过构建高质量数据和统一标签系统，研究数据质量对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.25767](https://arxiv.org/pdf/2603.25767)**

> **作者:** Xuanru Zhou; Yiwen Shao; Wei-Cheng Tseng; Dong Yu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Current audio pre-training seeks to learn unified representations for broad audio understanding tasks, but it remains fragmented and is fundamentally bottlenecked by its reliance on weak, noisy, and scale-limited labels. Drawing lessons from vision's foundational pre-training blueprint, we argue that the audio field must first establish its own large-scale, strong supervision framework. We introduce a new data-centric pipeline that leverages a high-fidelity captioner to create SOTA-quality captions and the first Unified Tag System (UTS) that bridges speech, music, and environmental sounds. We then conduct a systematic comparative study of different pre-training objectives on these strong source data. Our experiments suggest that data quality and coverage are the primary drivers of performance, while the choice of objective dictates downstream task specialization.
>
---
#### [new 005] LLaDA-TTS: Unifying Speech Synthesis and Zero-Shot Editing via Masked Diffusion Modeling
- **分类: cs.SD**

- **简介: 该论文提出LLaDA-TTS，解决TTS生成效率低和编辑困难的问题，通过掩码扩散模型实现并行生成和零样本语音编辑。**

- **链接: [https://arxiv.org/pdf/2603.26364](https://arxiv.org/pdf/2603.26364)**

> **作者:** Xiaoyu Fan; Huizhi Xie; Wei Zou; Yunzhang Chen
>
> **备注:** 11 pages, 6 figures, 2 tables
>
> **摘要:** Large language model (LLM)-based text-to-speech (TTS) systems achieve remarkable naturalness via autoregressive (AR) decoding, but require N sequential steps to generate N speech tokens. We present LLaDA-TTS, which replaces the AR LLM with a masked diffusion model that completes generation in a fixed number of parallel steps, decoupling inference latency from sequence length. Remarkably, using only 50 hours of fine-tuning data, we successfully transfer a pretrained AR checkpoint to the masked diffusion paradigm via bidirectional attention. At 64 steps, LLaDA-TTS achieves 0.98% CER (zh) and 1.96% WER (en) on Seed-TTS-Eval, matching the original CosyVoice 3 baseline performance while delivering a 2x LLM-stage speedup--a notable acceleration achieved despite the absence of KV cache, an optimization the AR baseline heavily relies on. Beyond acceleration, the bidirectional architecture naturally enables zero-shot speech editing--including word-level insertion, deletion, and substitution--without any additional training. Theoretically, we prove that AR-pretrained weights are near-optimal for bidirectional masked prediction under the locality property of acoustic tokens, explaining this rapid convergence. This general method modifies only the attention mask and objective, applying seamlessly to any LLM-based AR TTS system. Code and audio samples will be available at this https URL.
>
---
#### [new 006] Probabilistic Multilabel Graphical Modelling of Motif Transformations in Symbolic Music
- **分类: cs.SD; stat.ME; stat.ML**

- **简介: 该论文属于音乐结构分析任务，旨在研究音乐主题的变形模式。通过构建概率图模型，分析主题在旋律、节奏等维度上的变化，揭示其结构关系与风格特征。**

- **链接: [https://arxiv.org/pdf/2603.26478](https://arxiv.org/pdf/2603.26478)**

> **作者:** Ron Taieb; Yoel Greenberg; Barak Sober
>
> **备注:** 23 pages (21 pages main text), 2 figures. Submitted to Journal of New Music Research (Special Issue on Computational and Cognitive Musicology)
>
> **摘要:** Motifs often recur in musical works in altered forms, preserving aspects of their identity while undergoing local variation. This paper investigates how such motivic transformations occur within their musical context in symbolic music. To support this analysis, we develop a probabilistic framework for modeling motivic transformations and apply it to Beethoven's piano sonatas by integrating multiple datasets that provide melodic, rhythmic, harmonic, and motivic information within a unified analytical representation. Motif transformations are represented as multilabel variables by comparing each motif instance to a designated reference occurrence within its local context, ensuring consistent labeling across transformation families. We introduce a multilabel Conditional Random Field to model how motif-level musical features influence the occurrence of transformations and how different transformation families tend to co-occur. Our goal is to provide an interpretable, distributional analysis of motivic transformation patterns, enabling the study of their structural relationships and stylistic variation. By linking computational modeling with music-theoretical interpretation, the proposed framework supports quantitative investigation of musical structure and complexity in symbolic corpora and may facilitate the analysis of broader compositional patterns and writing practices.
>
---
#### [new 007] Sommelier: Scalable Open Multi-turn Audio Pre-processing for Full-duplex Speech Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决全双工语音语言模型的数据不足与对话复杂性问题，提出一个可扩展的开源数据处理管道。**

- **链接: [https://arxiv.org/pdf/2603.25750](https://arxiv.org/pdf/2603.25750)**

> **作者:** Kyudan Jung; Jihwan Kim; Soyoon Kim; Jeongoon Kim; Jaegul Choo; Cheonbok Park
>
> **备注:** 34 pages, 7 figures, 11 tables
>
> **摘要:** As the paradigm of AI shifts from text-based LLMs to Speech Language Models (SLMs), there is a growing demand for full-duplex systems capable of real-time, natural human-computer interaction. However, the development of such models is constrained by the scarcity of high-quality, multi-speaker conversational data, as existing large-scale resources are predominantly single-speaker or limited in volume. Addressing the complex dynamics of natural dialogue, such as overlapping and back-channeling remains a challenge, with standard processing pipelines suffering from diarization errors and ASR hallucinations. To bridge this gap, we present a robust and scalable open-source data processing pipeline designed for full-duplex model.
>
---
#### [new 008] Cinematic Audio Source Separation Using Visual Cues
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音频源分离任务，解决电影音频分解问题。提出AV-CASS框架，利用视觉线索提升分离效果，通过合成数据训练模型，实现多模态音频分离。**

- **链接: [https://arxiv.org/pdf/2603.26113](https://arxiv.org/pdf/2603.26113)**

> **作者:** Kang Zhang; Suyeon Lee; Arda Senocak; Joon Son Chung
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Cinematic Audio Source Separation (CASS) aims to decompose mixed film audio into speech, music, and sound effects, enabling applications like dubbing and remastering. Existing CASS approaches are audio-only, overlooking the inherent audio-visual nature of films, where sounds often align with visual cues. We present the first framework for audio-visual CASS (AV-CASS), leveraging visual context to enhance separation quality. Our method formulates CASS as a conditional generative modeling problem using conditional flow matching, enabling multimodal audio source separation. To address the lack of cinematic datasets with isolated sound tracks, we introduce a training data synthesis pipeline that pairs in-the-wild audio and video streams (e.g., facial videos for speech, scene videos for effects) and design a dedicated visual encoder for this dual-stream setup. Trained entirely on synthetic data, our model generalizes effectively to real-world cinematic content and achieves strong performance on synthetic, real-world, and audio-only CASS benchmarks. Code and demo are available at \url{this https URL}.
>
---
#### [new 009] Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决LLM在对话中利用上下文的问题。通过抽象压缩技术，用少量学习的隐变量替代长音频序列，提升效率并保留关键信息。**

- **链接: [https://arxiv.org/pdf/2603.26246](https://arxiv.org/pdf/2603.26246)**

> **作者:** Shashi Kumar; Esaú Villatoro-Tello; Sergio Burdisso; Kadri Hacioglu; Thibault Bañeras-Roux; Hasindri Watawana; Dairazalia Sanchez-Cortes; Srikanth Madikeri; Petr Motlicek; Andreas Stolcke
>
> **备注:** 11 pages
>
> **摘要:** Standard LLM-based speech recognition systems typically process utterances in isolation, limiting their ability to leverage conversational context. In this work, we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio token sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, which replaces the audio portion of prior turns with a fixed number of learned latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compressed model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footprint. We also provide targeted analyses of the compression setup and its trade-offs.
>
---
#### [new 010] A Power-Weighted Noncentral Complex Gaussian Distribution
- **分类: stat.ML; cs.LG; cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出一种新的复数概率模型，解决传统分布无法准确描述信号幅度特性的问题。通过引入幂权非中心复高斯分布，提升信号建模效果。**

- **链接: [https://arxiv.org/pdf/2603.26344](https://arxiv.org/pdf/2603.26344)**

> **作者:** Toru Nakashika
>
> **摘要:** The complex Gaussian distribution has been widely used as a fundamental spectral and noise model in signal processing and communication. However, its Gaussian structure often limits its ability to represent the diverse amplitude characteristics observed in individual source signals. On the other hand, many existing non-Gaussian amplitude distributions derived from hyperspherical models achieve good empirical fit due to their power-law structures, while they do not explicitly account for the complex-plane geometry inherent in complex-valued observations. In this paper, we propose a new probabilistic model for complex-valued random variables, which can be interpreted as a power-weighted noncentral complex Gaussian distribution. Unlike conventional hyperspherical amplitude models, the proposed model is formulated directly on the complex plane and preserves the geometric structure of complex-valued observations while retaining a higher-dimensional interpretation. The model introduces a nonlinear phase diffusion through a single shape parameter, enabling continuous control of the distributional geometry from arc-shaped diffusion along the phase direction to concentration of probability mass toward the origin. We formulate the proposed distribution and analyze the statistical properties of the induced amplitude distribution. The derived amplitude and power distributions provide a unified framework encompassing several widely used distributions in signal modeling, including the Rice, Nakagami, and gamma distributions. Experimental results on speech power spectra demonstrate that the proposed model consistently outperforms conventional distributions in terms of log-likelihood.
>
---
#### [new 011] Relational graph-driven differential denoising and diffusion attention fusion for multimodal conversation emotion recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多模态对话情感识别任务，旨在解决噪声干扰和模态信息不平衡导致的融合偏差问题。提出关系感知的去噪与注意力融合模型，提升情感识别效果。**

- **链接: [https://arxiv.org/pdf/2603.25752](https://arxiv.org/pdf/2603.25752)**

> **作者:** Ying Liu; Yuntao Shou; Wei Ai; Tao Meng; Keqin Li
>
> **备注:** 19 pages
>
> **摘要:** In real-world scenarios, audio and video signals are often subject to environmental noise and limited acquisition conditions, resulting in extracted features containing excessive noise. Furthermore, there is an imbalance in data quality and information carrying capacity between different modalities. These two issues together lead to information distortion and weight bias during the fusion phase, impairing overall recognition performance. Most existing methods neglect the impact of noisy modalities and rely on implicit weighting to model modality importance, thereby failing to explicitly account for the predominant contribution of the textual modality in emotion understanding. To address these issues, we propose a relation-aware denoising and diffusion attention fusion model for MCER. Specifically, we first design a differential Transformer that explicitly computes the differences between two attention maps, thereby enhancing temporally consistent information while suppressing time-irrelevant noise, which leads to effective denoising in both audio and video modalities. Second, we construct modality-specific and cross-modality relation subgraphs to capture speaker-dependent emotional dependencies, enabling fine-grained modeling of intra- and inter-modal relationships. Finally, we introduce a text-guided cross-modal diffusion mechanism that leverages self-attention to model intra-modal dependencies and adaptively diffuses audiovisual information into the textual stream, ensuring more robust and semantically aligned multimodal fusion.
>
---
## 更新

#### [replaced 001] TW-Sound580K: A Regional Audio-Text Dataset with Verification-Guided Curation for Localized Audio-Language Modeling
- **分类: cs.SD**

- **简介: 该论文提出TW-Sound580K数据集，解决大音频语言模型在方言语音上的性能问题，通过验证生成协议构建高质量指令对，并提升模型表现。**

- **链接: [https://arxiv.org/pdf/2603.05094](https://arxiv.org/pdf/2603.05094)**

> **作者:** Hao-Hui Xie; Ho-Lam Chung; Yi-Cheng Lin; Ke-Han Lu; Wenze Ren; Xie Chen; Hung-yi Lee
>
> **备注:** The authors have decided to withdraw this submission as the work is no longer intended for public dissemination at this time
>
> **摘要:** Large Audio-Language Models (LALMs) typically struggle with localized dialectal prosody due to the scarcity of specialized corpora. We present TW-Sound580K, a Taiwanese audio-text instruction dataset developed through a Verify-Generate-Critique (VGC) protocol. This pipeline leverages Dual-ASR validation to filter 522K raw clips, subsequently expanding them into 580,000 high-fidelity instruction pairs using a teacher model. The dataset's utility is demonstrated through Tai-LALM, which fine-tunes a DeSTA 2.5-Audio-initialized backbone and incorporates a dynamic Dual-ASR Arbitration strategy to optimize transcription selection during inference. On the TAU Benchmark, Tai-LALM reaches 49.1% accuracy, marking a 6.5% absolute improvement over the zero-shot baseline (42.6% with ASR text conditioning). This confirms that integrating regional corpora with rigorous curation and dynamic arbitration significantly enhances LALM performance on localized speech.
>
---
#### [replaced 002] Acoustic Imaging for UAV Detection: Dense Beamformed Energy Maps and U-Net SELD
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于声源定位任务，旨在通过U-Net模型实现无人机的360°声源定位。通过生成能量图并进行语义分割，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.00307](https://arxiv.org/pdf/2508.00307)**

> **作者:** Belman Jahir Rodriguez; Sergio F. Chevtchenko; Marcelo Herrera Martinez; Yeshwanth Bethi; Saeed Afshar
>
> **摘要:** We introduce a U-net model for 360° acoustic source localization formulated as a spherical semantic segmentation task. Rather than regressing discrete direction-of-arrival (DoA) angles, our model segments beamformed audio maps (azimuth & elevation) into regions of active sound presence. Using delay-and-sum (DAS) beamforming on a custom 24-microphone array, we generate signals aligned with drone GPS telemetry to create binary supervision masks. A modified U-Net, trained on frequency-domain representations of these maps, learns to identify spatially distributed source regions while addressing class imbalance via the Tversky loss. Because the network operates on beamformed energy maps, the approach is inherently array-independent and can adapt to different microphone configurations and can be transferred to different microphone configurations with minimal adaptation. The segmentation outputs are post-processed by computing centroids over activated regions, enabling robust DoA estimates. Our dataset includes real-world open-field recordings of a DJI Air 3 drone, synchronized with 360° video and flight logs across multiple dates and locations. Experimental results show that U-net generalizes across environments, providing improved angular precision, offering a new paradigm for dense spatial audio understanding beyond traditional Sound Source Localization (SSL). We additionally validate the same beamforming-plus-segmentation formulation on the DCASE 2019 TAU Spatial Sound Events benchmark, showing that the approach generalizes beyond drone acoustics to multiclass Sound Event Localization and Detection (SELD) scenarios.
>
---
#### [replaced 003] Joint Learning Global-Local Speaker Classification to Enhance End-to-End Speaker Diarization and Recognition
- **分类: cs.SD**

- **简介: 该论文属于说话人识别与聚类任务，旨在提升端到端说话人二值化与识别效果。针对模型判别能力不足的问题，提出GLSC-SDR方法，结合全局与局部分类策略，提升细粒度区分能力。**

- **链接: [https://arxiv.org/pdf/2603.25377](https://arxiv.org/pdf/2603.25377)**

> **作者:** Yuhang Dai; Haopeng Lin; Jiale Qian; Ruiqi Yan; Hao Meng; Hanke Xie; Hanlin Wen; Shunshun Yin; Ming Tao; Xie Chen; Lei Xie; Xinsheng Wang
>
> **备注:** 5 pages, 2 figures, 2 tables
>
> **摘要:** Large Audio-Language Models (LALMs) have demonstrated remarkable performance in end-to-end speaker diarization and recognition. However, their speaker discriminability remains limited due to the scarcity of large-scale conversational data and the absence of explicit speaker representation optimization. To address this, we propose GLSC-SDR, a paradigm that jointly trains speaker classification with diarization and recognition. We further introduce a Global-Local Speaker Classification strategy, which uses clustered speakers as global labels and re-encoded intra-cluster speakers as local labels. This hierarchical design enhances fine-grained speaker discrimination while preserving semantic transcription accuracy. Experiments on AliMeeting, AISHELL-4, and AMI-SDM demonstrate that GLSC-SDR achieves competitive or superior performance compared to simulation-based and multi-encoder approaches, without relying on large-scale real conversational data.
>
---
#### [replaced 004] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出文本条件的视频到音频生成任务，旨在从多对象视频中提取用户指定的声音。工作包括模型SELVA设计及自监督视频混合方案。**

- **链接: [https://arxiv.org/pdf/2512.02650](https://arxiv.org/pdf/2512.02650)**

> **作者:** Junwon Lee; Juhan Nam; Jiyoung Lee
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** This work introduces a new task, text-conditioned selective video-to-audio (V2A) generation, which produces only the user-intended sound from a multi-object video. This capability is especially crucial in multimedia production, where audio tracks are handled individually for each sound source for precise editing, mixing, and creative control. We propose SELVA, a novel text-conditioned V2A model that treats the text prompt as an explicit selector to distinctly extract prompt-relevant sound-source visual features from the video encoder. To suppress text-irrelevant activations with efficient video encoder finetuning, the proposed supplementary tokens promote cross-attention to yield robust semantic and temporal grounding. SELVA further employs an autonomous video-mixing scheme in a self-supervised manner to overcome the lack of mono audio track supervision. We evaluate SELVA on VGG-MONOAUDIO, a curated benchmark of clean single-source videos for such a task. Extensive experiments and ablations consistently verify its effectiveness across audio quality, semantic alignment, and temporal synchronization.
>
---
#### [replaced 005] Does Audio Deepfake Detection Generalize?
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决检测模型泛化能力不足的问题。通过系统评估不同方法，发现特征选择对性能有显著影响，并构建新数据集验证真实场景下的检测难度。**

- **链接: [https://arxiv.org/pdf/2203.16263](https://arxiv.org/pdf/2203.16263)**

> **作者:** Nicolas M. Müller; Pavel Czempin; Franziska Dieckmann; Adam Froghyar; Konstantin Böttinger
>
> **备注:** Interspeech 2022
>
> **摘要:** Current text-to-speech algorithms produce realistic fakes of human voices, making deepfake detection a much-needed area of research. While researchers have presented various techniques for detecting audio spoofs, it is often unclear exactly why these architectures are successful: Preprocessing steps, hyperparameter settings, and the degree of fine-tuning are not consistent across related work. Which factors contribute to success, and which are accidental? In this work, we address this problem: We systematize audio spoofing detection by re-implementing and uniformly evaluating architectures from related work. We identify overarching features for successful audio deepfake detection, such as using cqtspec or logspec features instead of melspec features, which improves performance by 37% EER on average, all other factors constant. Additionally, we evaluate generalization capabilities: We collect and publish a new dataset consisting of 37.9 hours of found audio recordings of celebrities and politicians, of which 17.2 hours are deepfakes. We find that related work performs poorly on such real-world data (performance degradation of up to one thousand percent). This may suggest that the community has tailored its solutions too closely to the prevailing ASVSpoof benchmark and that deepfakes are much harder to detect outside the lab than previously thought.
>
---
#### [replaced 006] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音翻译任务，比较SpeechLLMs与传统级联系统的性能，旨在验证语音模态集成对翻译质量的影响。**

- **链接: [https://arxiv.org/pdf/2512.16378](https://arxiv.org/pdf/2512.16378)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at this https URL
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which directly process spoken language and enable speech-to-text translation (ST) and other downstream tasks, bypassing traditional transcription-based pipelines. Whether this integration improves ST quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 6 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable solution overall, but most recent SpeechLLMs can match or even outperform cascades in various settings while SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [replaced 007] Gelina: Unified Speech and Gesture Synthesis via Interleaved Token Prediction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Gelina，解决语音与手势联合生成任务，通过交错令牌预测提升同步性与韵律对齐，实现多说话人和风格克隆。**

- **链接: [https://arxiv.org/pdf/2510.12834](https://arxiv.org/pdf/2510.12834)**

> **作者:** Téo Guichoux; Théodor Lemerle; Shivam Mehta; Jonas Beskow; Gustav Eje Henter; Laure Soulier; Catherine Pelachaud; Nicolas Obin
>
> **备注:** Paper accepted at ICASSP 2026, 5 pages
>
> **摘要:** Human communication is multimodal, with speech and gestures tightly coupled, yet most computational methods for generating speech and gestures synthesize them sequentially, weakening synchrony and prosody alignment. We introduce Gelina, a unified framework that jointly synthesizes speech and co-speech gestures from text using interleaved token sequences in a discrete autoregressive backbone, with modality-specific decoders. Gelina supports multi-speaker and multi-style cloning and enables gesture-only synthesis from speech inputs. Subjective and objective evaluations demonstrate competitive speech quality and improved gesture generation over unimodal baselines.
>
---
#### [replaced 008] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文提出DiFlowDubber，解决视频配音中的语音与唇形同步及语音表现力问题，通过跨模态对齐和离散流匹配实现更精准的自动配音。**

- **链接: [https://arxiv.org/pdf/2603.14267](https://arxiv.org/pdf/2603.14267)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Jeongsoo Choi; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **备注:** Accepted at CVPR 2026 Findings
>
> **摘要:** Video dubbing has broad applications in filmmaking, multimedia creation, and assistive speech technology. Existing approaches either train directly on limited dubbing datasets or adopt a two-stage pipeline that adapts pre-trained text-to-speech (TTS) models, which often struggle to produce expressive prosody, rich acoustic characteristics, and precise synchronization. To address these issues, we propose DiFlowDubber with a novel two-stage training framework that effectively transfers knowledge from a pre-trained TTS model to video-driven dubbing, with a discrete flow matching generative backbone. Specifically, we design a FaPro module that captures global prosody and stylistic cues from facial expressions and leverages this information to guide the modeling of subsequent speech attributes. To ensure precise speech-lip synchronization, we introduce a Synchronizer module that bridges the modality gap among text, video, and speech, thereby improving cross-modal alignment and generating speech that is temporally synchronized with lip movements. Experiments on two primary benchmark datasets demonstrate that DiFlowDubber outperforms previous methods across multiple metrics.
>
---
