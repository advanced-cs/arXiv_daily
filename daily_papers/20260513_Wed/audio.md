# 音频 cs.SD;  eess.AS

- **最新发布 12 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Towards Fine-Grained Multi-Dimensional Speech Understanding: Data Pipeline, Benchmark, and Model
- **分类: eess.AS**

- **简介: 该论文属于语音理解任务，旨在解决现有模型在细粒度、多维语音感知上的不足。通过构建数据集、基准测试和新模型，提升语音系统的感知与共情能力。**

- **链接: [https://arxiv.org/pdf/2605.12036](https://arxiv.org/pdf/2605.12036)**

> **作者:** Guojian Li; Zhixian Zhao; Zhennan Lin; Jingbin Hu; Qirui Zhan; Yuang Cao; Pengyuan Xie; Chuan Xie; Jie Liu; Qiang Zhang; Zhonghua Fu; Lei Xie
>
> **摘要:** While speech Large Language Models (LLMs) excel at conventional tasks like basic speech recognition, they lack fine-grained, multi-dimensional perception. This deficiency is evident in their struggle to disentangle complex features like micro-acoustic cues, acoustic scenes, and paralinguistic signals. This resulting incomplete comprehension of real-world speech fundamentally bottlenecks the development of perceptive and empathetic next-generation speech systems. At its core, this persistent perceptual limitation primarily stems from three interacting factors: scarce high-quality expressive data, absent fine-grained modeling for multi-dimensional attributes, and reliance on restricted coverage, coarse-grained benchmarks. We address these challenges through three pillars: First, our robust data curation pipeline resolves complex acoustic environments and long-audio timestamp alignment challenges to extract a high-quality spontaneous speech corpus from audiovisual sources. Second, we construct FMSU-Bench, a pioneering benchmark covering 14 speech attribute dimensions to rigorously assess the fine-grained, multi-dimensional speech understanding capabilities of current models. Third, empowered by our curated corpus, we introduce FM-Speech. Driven by a decoupled attribute modeling and progressive curriculum fine-tuning framework, it substantially elevates fine-grained, multi-dimensional acoustic perception. Extensive evaluations on FMSU-Bench reveal that current speech LLMs still require significant improvement in multi-dimensional, fine-grained understanding. In contrast, FM-Speech substantially outperforms current open-source models, establishing a robust paradigm for real-world speech understanding.
>
---
#### [new 002] Too Good to Be True: A Study on Modern Automatic Speech Recognition for the Evaluation of Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强评估任务，研究现代自动语音识别（ASR）模型与人类对增强语音识别的关联性，探讨其在评估语音增强效果中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.12107](https://arxiv.org/pdf/2605.12107)**

> **作者:** Danilo de Oliveira; Tal Peer; Timo Gerkmann
>
> **摘要:** Speech enhancement (SE) systems are typically evaluated using a variety of instrumental metrics. The use of automatic speech recognition (ASR) systems to evaluate SE performance is common in literature, usually in terms of word error rate (WER). However, WER scores depend heavily on the choice of ASR system and text normalization pipeline. In this paper, we investigate how modern ASR models correlate with human recognition of enhanced speech. A listening experiment reveals that modern ASR models with large-scale noisy training and embedded language models correlate more with human WER than simpler ones, with a transducer model providing the most reliable transcriptions. Nevertheless, we also show that these models' robustness to noise and use of context can be uninformative to an acoustics-focused evaluation of enhancement performance.
>
---
#### [new 003] Chunkwise Aligners for Streaming Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决流式ASR的训练效率问题。提出Chunkwise Aligner，通过分块对齐提升训练和解码效率，同时保持准确率。**

- **链接: [https://arxiv.org/pdf/2605.11422](https://arxiv.org/pdf/2605.11422)**

> **作者:** Wen Shen Teo; Takafumi Moriya; Masato Mimura
>
> **摘要:** We propose the Chunkwise Aligner, a novel architecture for streaming automatic speech recognition (ASR). While the Transducer is the standard model for streaming ASR, its training is costly due to the need to compute all possible audio-label alignments. The recently introduced Aligner reduces this cost by discarding explicit alignments, but this modification makes it unsuitable for streaming. Our approach overcomes this limitation by dividing the audio into chunks and aligning each label to the leftmost frames of its chunk, whereas transitions between chunks are managed by a learned end-of-chunk probability. Experiments show that the Chunkwise Aligner not only matches the Transducer's accuracy in both offline and streaming scenarios, but also offers superior training and decoding efficiencies.
>
---
#### [new 004] STRUM: A Spectral Transcription and Rhythm Understanding Model for End-to-End Generation of Playable Rhythm-Game Charts
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出STRUM模型，解决音频到节奏游戏谱的端到端生成任务，通过多阶段方法处理不同乐器，提升节奏游戏谱的自动化生成效果。**

- **链接: [https://arxiv.org/pdf/2605.12135](https://arxiv.org/pdf/2605.12135)**

> **作者:** Joshua Opria
>
> **备注:** 9 pages, 4 figures, 3 tables. Code and models: this https URL
>
> **摘要:** We present STRUM (Spectral Transcription and Rhythm Understanding Model), an audio-to-chart pipeline that converts raw recordings into playable Clone Hero / YARG charts for drums, guitar, bass, vocals, and keys without any oracle metadata. STRUM is a multi-stage hybrid: a two-stage CRNN onset detector and a six-model ensemble classifier for drums; neural onset detectors with monophonic pitch tracking for guitar and bass; word-aligned ASR for vocals; and spectral keyboard detection for keys. We evaluate on a 30-song in-envelope benchmark constructed by screening candidate songs on a single audio-quality criterion -- the median 1-second drum-stem RMS after htdemucs_6s source separation. On this benchmark STRUM achieves drums onset F1 = 0.838, bass F1 = 0.694, guitar F1 = 0.651, and vocals F1 = 0.539 at a +/- 100 ms tolerance with per-song global offset search. We report a complete ablation of seven drum-pipeline components with paired per-song Wilcoxon tests, an analysis of ground-truth-to-audio timing distributions in community Clone Hero charts, and a per-class confusion matrix for the drum classifier. Code, model weights, and the full benchmark manifest are released.
>
---
#### [new 005] The SMC Blind Spot: A Failure Mode Analysis of State-of-the-Art Beat Tracking
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究音乐节拍跟踪任务，针对SMC数据集中的模型失败现象，分析出三种错误类型，并提出数据多样化和多假设调速等改进方向。**

- **链接: [https://arxiv.org/pdf/2605.12287](https://arxiv.org/pdf/2605.12287)**

> **作者:** Jaehoon Ahn; Tae Gum Hwang; Moon-Ryul Jung
>
> **备注:** 6 pages, 3 figures. Technical report on beat tracking failure modes; prepared for ISMIR 2026
>
> **摘要:** Over the past two decades, the task of musical beat tracking has transitioned from heuristic onset detection algorithms to highly capable deep neural networks (DNN). Although DNN-based beat tracking models achieve near-perfect performance on mainstream, percussive datasets, the SMC dataset has stubbornly yielded low F-measure scores. By testing how well state-of-the-art models detect beats on individual tracks in the SMC dataset, we identify three distinct failure modes: octave errors, continuity errors, and complete tracking failure where all metrics fall below 0.3. We reveal that state-of-the-art models tend to generate "confident-but-wrong" activations. Furthermore, we show that the standard DBN's default minimum tempo of 55 BPM prevents it from inferring the correct tempo for 21\% of SMC tracks, forcing double-tempo predictions on slow music. By exposing such fundamental oversights, we provide concrete directions for improving beat and downbeat detection, specifically emphasizing training data diversification and multi-hypothesis tempo estimation.
>
---
#### [new 006] Mixture-of-Experts Framework for Field-of-View Enhanced Signal-Dependent Binauralization of Moving Talkers
- **分类: cs.SD; eess.AS; stat.ML**

- **简介: 该论文属于空间音频任务，解决动态说话者声源定位与增强问题。通过混合专家框架实现信号依赖的双耳渲染，支持实时跟踪与方向控制。**

- **链接: [https://arxiv.org/pdf/2509.13548](https://arxiv.org/pdf/2509.13548)**

> **作者:** Manan Mittal; Thomas Deppisch; Joseph Forrer; Chris Le Sueur; Zamir Ben-Hur; David Lou Alon; Daniel D.E. Wong
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** We propose a novel mixture of experts framework for field-of-view enhancement in binaural signal matching. Our approach enables dynamic spatial audio rendering that adapts to continuous talker motion, allowing users to emphasize or suppress sounds from selected directions while preserving natural binaural cues. Unlike traditional methods that rely on explicit direction-of-arrival estimation or operate in the Ambisonics domain, our signal-dependent framework combines multiple binaural filters in an online manner using implicit localization. This allows for real-time tracking and enhancement of moving sound sources, supporting applications such as speech focus, noise reduction, and world-locked audio in augmented and virtual reality. The method is agnostic to array geometry offering a flexible solution for spatial audio capture and personalized playback in next-generation consumer audio devices.
>
---
#### [new 007] Poly-SVC: Polyphony-Aware Singing Voice Conversion with Harmonic Modeling
- **分类: cs.SD**

- **简介: 该论文属于SVC任务，解决伴奏中残留和声难以处理的问题。提出Poly-SVC系统，结合CQT、随机采样和CFM解码器，实现自然的多声部语音转换。**

- **链接: [https://arxiv.org/pdf/2605.12310](https://arxiv.org/pdf/2605.12310)**

> **作者:** Chen Geng; Meng Chen; Ruohua Zhou; Ruolan Liu; Weifeng Zhao
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Singing Voice Conversion (SVC) aims to transform a source singing voice into a target singer while preserving lyrics and melody. Most existing SVC methods depend on F0 extractors to capture the lead melody from clean vocals. However, no existing method can reliably extract clean vocals from accompanied recordings without leaving residual harmonies behind. In this paper, we innovatively propose Poly-SVC, a zero-shot, cross-lingual singing voice conversion system designed to process residual harmonies. Poly-SVC is composed of three key components: a Constant-Q Transform (CQT)-based pitch extractor to preserve both the lead melody and residual harmony, a random sampler to reduce interference information from the CQT and a diffusion decoder based on Conditional Flow Matching (CFM) that fuses pitch, content, and timbre features into natural-sounding polyphonic outputs. Experiments demonstrate that Poly-SVC surpasses the baseline models in naturalness, timbre similarity and harmony reconstruction across both harmony-rich and single-melody recordings.
>
---
#### [new 008] A Semi-Supervised Framework for Speech Confidence Detection using Whisper
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音情感识别任务，旨在解决标注数据不足和主观性问题。提出一种半监督框架，融合语义嵌入与声学特征，提升说话人信心检测效果。**

- **链接: [https://arxiv.org/pdf/2605.12387](https://arxiv.org/pdf/2605.12387)**

> **作者:** Adam Wynn; Jingyun Wang
>
> **备注:** 12 pages, 9 Figures, Submitted to IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Automatic detection of speaker confidence is critical for adaptive computing but remains constrained by limited labelled data and the subjectivity of paralinguistic annotations. This paper proposes a semi-supervised hybrid framework that fuses deep semantic embeddings from the Whisper encoder with an interpretable acoustic feature vector composed of eGeMAPS descriptors and auxiliary probability estimates of vocal stress and disfluency. To mitigate reliance on scarce ground truth data, we introduce an Uncertainty-Aware Pseudo-Labelling strategy where a model generates labels for unlabelled data, retaining only high-quality samples for training. Experimental results demonstrate that the proposed approach achieves a Macro-F1 score of 0.751, outperforming self-supervised baselines, including WavLM, HuBERT, and Wav2Vec 2.0. The hybrid architecture also surpasses the unimodal Whisper baseline, yielding a 3\% improvement in the minority class, confirming that explicit prosodic and auxiliary features provide necessary corrective signals which are otherwise lost in deep semantic representations. Ablation studies further show that a curated set of high confidence pseudo-labels outperforms indiscriminate large scale augmentation, confirming that data quality outweighs quantity for perceived confidence detection.
>
---
#### [new 009] AuDirector: A Self-Reflective Closed-Loop Framework for Immersive Audio Storytelling
- **分类: cs.SD**

- **简介: 该论文提出AuDirector，解决长音频叙事的一致性与表达问题。通过多智能体框架实现自反思闭环，提升语音适配、质量修正与用户交互。属于音频故事生成任务。**

- **链接: [https://arxiv.org/pdf/2605.11866](https://arxiv.org/pdf/2605.11866)**

> **作者:** Yiming Ren; Xuenan Xu; Ziyang Zhang; Wen Wu; Baoxiang Li; Chao Zhang
>
> **摘要:** Despite advances in text and visual generation, creating coherent long-form audio narratives remains challenging. Existing frameworks often exhibit limitations such as mismatched character settings with voice performance, insufficient self-correction mechanisms, and limited human interactivity. To address these challenges, we propose AuDirector, a self-reflective closed-loop multi-agent framework. Specifically, it involves an Identity-Aware Pre-production mechanism that transforms narrative texts into character profiles and utterance-level emotional instructions to retrieve suitable voice candidates and guide expressive speech synthesis, thereby promoting context-aligned voice adaptation. To enhance quality, a Collaborative Synthesis and Correction module introduces a closed-loop self-correction mechanism to systematically audit and regenerate defective audio components. Furthermore, a Human-Guided Interactive Refinement module facilitates user control by interpreting natural language feedback to interactively refine the underlying scripts. Experiments demonstrate that AuDirector achieves superior performance compared to state-of-the-art baselines in structural coherence, emotional expressiveness, and acoustic fidelity. Audio samples can be found at this https URL.
>
---
#### [new 010] Exploring Token-Space Manipulation in Latent Audio Tokenizers
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出LATTE模型，解决音频编码中全局因素难以干预的问题。通过引入可学习的潜在标记，实现音频的可控编辑。属于音频处理任务。**

- **链接: [https://arxiv.org/pdf/2605.11192](https://arxiv.org/pdf/2605.11192)**

> **作者:** Francesco Paissan; Luca Della Libera; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Neural audio codecs provide compact discrete representations for speech generation and manipulation. However, most codecs organize tokens as frame-level sequences, making it difficult to study or intervene on global factors of variation. In this work, we propose the Latent Audio Tokenizer for Token-space Editing (LATTE) that appends a fixed set of learnable latent tokens to the audio feature sequence and retains only these tokens for quantization and decoding. This design produces a compact, non-temporally aligned bottleneck in which each token can aggregate global information across the full utterance. We show that the resulting tokenizer preserves competitive reconstruction quality in low-bitrate speech coding settings while enabling simple token-space interventions. In particular, we find that swapping selected latent token positions between utterances can modify global attributes, such as speaker identity and background noise, and we evaluate these interventions on voice conversion and denoising tasks. Our results suggest that compact latent audio tokenizers can support controllable audio manipulation without supervision in task-specific editing models.
>
---
#### [new 011] AffectCodec: Emotion-Preserving Neural Speech Codec for Expressive Speech Modeling
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决情感信息在量化过程中丢失的问题。通过引入情感引导的编码框架，保留情感线索，同时保证语义和韵律的自然。**

- **链接: [https://arxiv.org/pdf/2605.11098](https://arxiv.org/pdf/2605.11098)**

> **作者:** Jiacheng Shi; Hongfei Du; Xinyuan Song; Y. Alicia Hong; Yanfu Zhang; Ye Gao
>
> **备注:** Accepted to ACL Findings 2026
>
> **摘要:** Neural speech codecs provide discrete representations for speech language models, but emotional cues are often degraded during quantization. Existing codecs mainly optimize acoustic reconstruction, leaving emotion expressiveness insufficiently modeled at the representation level. We propose an emotion-guided neural speech codec that explicitly preserves emotional information while maintaining semantic fidelity and prosodic naturalness. Our framework combines emotion-semantic guided latent modulation, relation-preserving emotional-semantic distillation, and emotion-weighted semantic alignment to retain emotionally salient cues under compression. Extensive evaluations across speech reconstruction, emotion recognition, and downstream text-to-speech generation demonstrate improved emotion consistency and perceptual quality without sacrificing content accuracy.
>
---
#### [new 012] Adaptive Diagonal Loading using Krylov Subspaces for Robust Beamforming
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，解决大麦克风阵列在动态环境中的鲁棒波束成形问题。通过Krylov子空间方法高效估计特征值，提升计算效率并保持良好性能。**

- **链接: [https://arxiv.org/pdf/2605.11286](https://arxiv.org/pdf/2605.11286)**

> **作者:** Manan Mittal; Ryan M. Corey; John R. Buck; Andrew C. Singer
>
> **备注:** 5 pages, 8 figures
>
> **摘要:** Reliable adaptive beamforming is critical for large microphone arrays operating in highly dynamic acoustic environments. In scenarios characterized by fast-moving talkers and interferers, the available sample support for estimating the spatial correlation matrix is often snapshot-deficient. This deficiency degrades the White Noise Gain (WNG), leading to severe target signal cancellation. To ensure stable and robust beamforming, we previously proposed an adaptive diagonal loading method that leverages the Kantorovich inequality to guarantee the WNG remains strictly within specified bounds. However, accurately determining the smallest necessary loading level requires calculating the extreme eigenvalues of the spatial correlation matrix, a computationally expensive $\mathcal{O}(M^3)$ operation for large arrays. In this paper, we introduce a highly efficient $\mathcal{O}(kM^2)$ estimation technique using Lanczos iterations to build a small Krylov subspace. By projecting the correlation matrix onto a tridiagonal matrix of dimension $k \ll M$, we extract Ritz values that rapidly converge to the exact extreme eigenvalues. Our evaluations demonstrate that this Lanczos-accelerated approach achieves performance identical to exact Eigenvalue Decomposition (EVD), ensuring optimal interference suppression and strict WNG adherence at a fraction of the computational cost.
>
---
## 更新

#### [replaced 001] Modality-Inconsistent Continual Learning of Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文研究多模态大语言模型的持续学习问题，针对模态和任务类型不一致导致的灾难性遗忘，提出MoInCL方法，通过生成伪目标和基于指令的知识蒸馏来缓解遗忘。**

- **链接: [https://arxiv.org/pdf/2412.13050](https://arxiv.org/pdf/2412.13050)**

> **作者:** Weiguo Pian; Shijian Deng; Shentong Mo; Mingrui Liu; Yunhui Guo; Yapeng Tian
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR), 2026
>
> **摘要:** In this paper, we introduce Modality-Inconsistent Continual Learning (MICL), a new continual learning scenario for Multimodal Large Language Models (MLLMs) that involves tasks with inconsistent modalities (image, audio, or video) and varying task types (captioning or question-answering). Unlike existing vision-only or modality-incremental settings, MICL combines modality and task type shifts, both of which drive catastrophic forgetting. To address these challenges, we propose MoInCL, which employs a Pseudo Targets Generation Module to mitigate forgetting caused by task type shifts in previously seen modalities. It also incorporates Instruction-based Knowledge Distillation to preserve the model's ability to handle previously learned modalities when new ones are introduced. We benchmark MICL using a total of six tasks and conduct experiments to validate the effectiveness of our MoInCL. The experimental results highlight the superiority of MoInCL, showing significant improvements over representative and state-of-the-art continual learning baselines.
>
---
#### [replaced 002] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步检索增强知识获取，提升准确性同时保持交互性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [replaced 003] Developing a Multi-variate Prediction Model For COVID-19 From Crowd-sourced Respiratory Voice Data
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于疾病诊断任务，旨在通过语音数据检测新冠感染。研究使用深度学习模型分析语音特征，提升诊断准确率。**

- **链接: [https://arxiv.org/pdf/2402.07619](https://arxiv.org/pdf/2402.07619)**

> **作者:** Yuyang Yan; Wafaa Aljbawi; Sami O. Simons; Visara Urovi
>
> **备注:** arXiv admin note: text overlap with arXiv:2209.03727
>
> **摘要:** COVID-19 has affected more than 223 countries worldwide and in the Post-COVID Era, there is a pressing need for non-invasive, low-cost, and highly scalable solutions to detect COVID-19. We develop a deep learning model to identify COVID-19 from voice recording data. The novelty of this work is in the development of deep learning models for COVID-19 identification from only voice recordings. We use the Cambridge COVID-19 Sound database which contains 893 speech samples, crowd-sourced from 4352 participants via a COVID-19 Sounds app. Voice features including Mel-spectrograms and Mel-frequency cepstral coefficients (MFCC) and CNN Encoder features are extracted. Based on the voice data, we develop deep learning classification models to detect COVID-19 cases. These models include Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) and Hidden-Unit BERT (HuBERT). We compare their predictive power to baseline machine learning models. HuBERT achieves the highest accuracy of 86\% and the highest AUC of 0.93. The results achieved with the proposed models suggest promising results in COVID-19 diagnosis from voice recordings when compared to the results obtained from the state-of-the-art.
>
---
#### [replaced 004] Probing Cross-modal Information Hubs in Audio-Visual LLMs
- **分类: cs.AI; eess.AS**

- **简介: 该论文研究音频-视觉大语言模型中的跨模态信息流动，旨在揭示信息如何在不同模态间编码。任务属于多模态模型分析，解决跨模态信息存储机制问题，发现并利用跨模态汇点提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.10815](https://arxiv.org/pdf/2605.10815)**

> **作者:** Jihoo Jung; Chaeyoung Jung; Ji-Hoon Kim; Joon Son Chung
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Audio-visual large language models (AVLLMs) have recently emerged as a powerful architecture capable of jointly reasoning over audio, visual, and textual modalities. In AVLLMs, the bidirectional interaction between audio and video modalities introduces intricate processing dynamics, necessitating a deeper understanding of their internal mechanisms. However, unlike extensively studied text-only or large vision language models, the internal workings of AVLLMs remain largely unexplored. In this paper, we focus on cross-modal information flow between audio and visual modalities in AVLLMs, investigating where information derived from one modality is encoded within the token representations of the other modality. Through an analysis of multiple recent AVLLMs, we uncover two common findings. First, AVLLMs primarily encode integrated audio-visual information in sink tokens. Second, sink tokens do not uniformly hold cross-modal information. Instead, a distinct subset of sink tokens, which we term cross-modal sink tokens, specializes in storing such information. Based on these findings, we further propose a simple training-free hallucination mitigation method by encouraging reliance on integrated cross-modal information within cross-modal sink tokens. Our code is available at this https URL.
>
---
#### [replaced 005] Speech Enhancement Based on Drifting Models
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决单步高保真去噪问题。提出DriftSE框架，通过分布演化实现快速增强，无需迭代采样。**

- **链接: [https://arxiv.org/pdf/2604.24199](https://arxiv.org/pdf/2604.24199)**

> **作者:** Liang Xu; Diego Caviedes-Nozal; Bastiaan Kleijn; Longfei Felix Yan; Rasmus Kongsgaard Olsson
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** We propose Speech Enhancement based on Drifting Models (DriftSE), a novel generative framework that formulates denoising as an equilibrium problem. Rather than relying on iterative sampling, DriftSE natively achieves one-step inference by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution. This evolution is driven by a Drifting Field, a learned correction vector that guides samples toward the high-density regions of the clean distribution, which naturally facilitates training on unpaired data by matching distributions rather than paired samples. We investigate the framework under two formulations: a direct mapping from the noisy observation, and a stochastic conditional generative model from a Gaussian prior. Experiments on the VoiceBank-DEMAND benchmark demonstrate that DriftSE achieves high-fidelity enhancement in a single step, outperforming multi-step diffusion baselines and establishing a new paradigm for speech enhancement.
>
---
#### [replaced 006] Online Single-Channel Audio-Based Sound Speed Estimation for Robust Multi-Channel Audio Control
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频控制任务，解决环境变化导致的声速不准确问题。提出一种在线单麦克风声速估计方法，提升多通道音频控制性能。**

- **链接: [https://arxiv.org/pdf/2602.16416](https://arxiv.org/pdf/2602.16416)**

> **作者:** Andreas Jonas Fuglsig; Mads Græsbøll Christensen; Jesper Rindom Jensen
>
> **备注:** Accepted for publication at EUSIPCO 2026
>
> **摘要:** Robust spatial audio control relies on accurate acoustic propagation models, yet environmental variations, especially changes in the speed of sound, cause systematic mismatches that degrade performance. Existing methods either assume known sound speed, require multiple microphones, or rely on separate calibration, making them impractical for systems with minimal sensing. We propose an online sound speed estimator that operates during general multichannel audio playback and requires only a single observation microphone. The method exploits the structured effect of sound speed on the reproduced signal and estimates it by minimizing the mismatch between the measured audio and a parametric acoustic model. Simulations show accurate tracking of sound speed for diverse input signals and improved spatial control performance when the estimates are used to compensate propagation errors in a sound zone control framework.
>
---
#### [replaced 007] One Prompt, Many Sounds: Modeling Listener Variability in LLM-Based Equalization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频处理任务，旨在解决传统均衡调整静态、手动的问题。通过LLM将文本提示转化为均衡设置，实现更灵活的音效控制。**

- **链接: [https://arxiv.org/pdf/2601.09448](https://arxiv.org/pdf/2601.09448)**

> **作者:** Ioannis Stylianou; Jon Francombe; Pablo Martinez-Nuevo; Sven Ewan Shepstone; Zheng-Hua Tan
>
> **备注:** 13 pages, 15 figures, 2 tables, IEEE JSTSP submission
>
> **摘要:** Conventional audio equalization is a static process that requires manual and cumbersome adjustments to adapt to changing listening contexts (e.g., mood, location, or social setting). In this paper, we introduce a Large Language Model (LLM)-based alternative that maps natural language text prompts to equalization settings. This enables a conversational approach to sound system control. By utilizing data collected from a controlled listening experiment, our models exploit in-context learning and parameter-efficient fine-tuning techniques to reliably align with population-preferred equalization settings. Our evaluation methods, which leverage distributional metrics that capture users' varied preferences, show statistically significant improvements in distributional alignment over random sampling and static preset baselines. These results indicate that LLMs could function as "artificial equalizers," contributing to the development of more accessible, context-aware, and expert-level audio tuning methods.
>
---
#### [replaced 008] SAND: The Challenge on Speech Analysis for Neurodegenerative Disease Assessment
- **分类: eess.AS; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于ALS早期诊断任务，旨在通过语音分析解决神经退行性疾病评估问题。研究构建了标注数据集并发起SAND挑战，推动AI模型的开发与评估。**

- **链接: [https://arxiv.org/pdf/2604.16445](https://arxiv.org/pdf/2604.16445)**

> **作者:** Giovanna Sannino; Ivanoe De Falco; Nadia Brancati; Laura Verde; Maria Frucci; Daniel Riccio; Vincenzo Bevilacqua; Antonio Di Marino; Lucia Aruta; Valentina Virginia Iuzzolino; Gianmaria Senerchia; Myriam Spisto; Raffaele Dubbioso
>
> **摘要:** Recent advances in Artificial Intelligence (AI) and the exploration of noninvasive, objective biomarkers, such as speech signals, have encouraged the development of algorithms to support the early diagnosis of neurodegenerative diseases, including Amyotrophic Lateral Sclerosis (ALS). Voice changes in subjects suffering from ALS typically manifest as progressive dysarthria, which is a prominent neurodegenerative symptom because it affects patients as the disease progresses. Since voice signals are complex data, the development and use of advanced AI techniques are fundamental to extracting distinctive patterns from them. Validating AI algorithms for ALS diagnosis and monitoring using voice signals is challenging, particularly due to the lack of annotated reference datasets. In this work, we present the outcome of a collaboration between a multidisciplinary team of clinicians and Machine Learning experts to create both a clinically annotated validation dataset and the "Speech Analysis for Neurodegenerative Diseases" (SAND) challenge based on it. Specifically, by analyzing voice disorders, the SAND challenge provides an opportunity to develop, test, and evaluate AI models for the automatic early identification and prediction of ALS disease progression.
>
---
#### [replaced 009] Towards Fine-Grained Code-Switch Speech Translation with Semantic Space Alignment
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于代码切换语音翻译任务，旨在解决语义建模复杂和数据稀缺问题。通过引入专家混合模型和多阶段训练，提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.10670](https://arxiv.org/pdf/2511.10670)**

> **作者:** Yan Gao; Yazheng Yang; Zhibin Lan; Yidong Chen; Min Zhang; Daimeng Wei; Derek F. Wong; Jinsong Su
>
> **备注:** Accepted to IJCAI 2026 Main Track
>
> **摘要:** Code-switching (CS) speech translation (ST) aims to translate speech that alternates between multiple languages into a target language text, posing significant challenges due to the complexity of semantic modeling and the scarcity of CS data. Previous studies mainly rely on the models themselves to implicitly learn semantic representations and resort to costly manual annotations. To mitigate these limitations, we propose enhancing Large Language Models (LLMs) with a Mixture-of-Experts (MoE) speech projector composed of language expert groups, where each group specializes in the semantic space of a specific language for fine-grained speech feature modeling. A language-specific loss and an intra-group load balancing loss are jointly introduced to guide efficient token routing across and within expert groups. Furthermore, we introduce a multi-stage training paradigm that utilizes readily available automatic speech recognition (ASR) and monolingual ST data, facilitating speech-text alignment and improving translation performance. To bridge the data gap for smooth domain transfer, a transition loss is employed to improve adaptation to CS scenarios. Extensive experiments on widely used datasets demonstrate the effectiveness and generality of our approach, achieving average improvements of $0.86$ BLEU and $0.93$ COMET over SeamlessM4T, with maximum improvements of $1.49$ BLEU and $1.41$ COMET across different test sets.
>
---
