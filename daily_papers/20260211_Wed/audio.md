# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音转换与匿名化任务，解决实时合成中身份与内容不匹配的问题。提出TVT表示，实现内容同步的时变音色，提升自然度与隐私保护。**

- **链接: [https://arxiv.org/pdf/2602.09389v1](https://arxiv.org/pdf/2602.09389v1)**

> **作者:** Waris Quamer; Mu-Ruei Tseng; Ghady Nasrallah; Ricardo Gutierrez-Osuna
>
> **摘要:** Real-time voice conversion and speaker anonymization require causal, low-latency synthesis without sacrificing intelligibility or naturalness. Current systems have a core representational mismatch: content is time-varying, while speaker identity is injected as a static global embedding. We introduce a streamable speech synthesizer that aligns the temporal granularity of identity and content via a content-synchronous, time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre instance into multiple compact facets; frame-level content attends to this memory, a gate regulates variation, and spherical interpolation preserves identity geometry while enabling smooth local changes. In addition, a factorized vector-quantized bottleneck regularizes content to reduce residual speaker leakage. The resulting system is streamable end-to-end, with <80 ms GPU latency. Experiments show improvements in naturalness, speaker transfer, and anonymization compared to SOTA streaming baselines, establishing TVT as a scalable approach for privacy-preserving and expressive speech synthesis under strict latency budgets.
>
---
#### [new 002] Soft Clustering Anchors for Self-Supervised Speech Representation Learning in Joint Embedding Prediction Architectures
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出GMM-Anchored JEPA，解决自监督语音表示学习中的表示崩溃问题。通过软聚类增强JEPA，提升语音识别等任务性能。**

- **链接: [https://arxiv.org/pdf/2602.09040v1](https://arxiv.org/pdf/2602.09040v1)**

> **作者:** Georgios Ioannides; Adrian Kieback; Judah Goldfeder; Linsey Pang; Aman Chadha; Aaron Elkins; Yann LeCun; Ravid Shwartz-Ziv
>
> **备注:** 15 pages, 5 figures. Code: github.com/gioannides/clustering-anchored-jepa
>
> **摘要:** Joint Embedding Predictive Architectures (JEPA) offer a promising approach to self-supervised speech representation learning, but suffer from representation collapse without explicit grounding. We propose GMM-Anchored JEPA, which fits a Gaussian Mixture Model once on log-mel spectrograms and uses its frozen soft posteriors as auxiliary targets throughout training. A decaying supervision schedule allows GMM regularization to dominate early training before gradually yielding to the JEPA objective. Unlike HuBERT and WavLM, which require iterative re-clustering, our approach clusters input features once with soft rather than hard assignments. On ~50k hours of speech, GMM anchoring improves ASR (28.68% vs. 33.22% WER), emotion recognition (67.76% vs. 65.46%), and slot filling (64.7% vs. 59.1% F1) compared to a WavLM-style baseline with matched compute. Cluster analysis shows GMM-anchored representations achieve up to 98% entropy compared to 31% for WavLM-style, indicating substantially more uniform cluster utilization. Code is made available at https://github.com/gioannides/clustering-anchored-jepa.
>
---
#### [new 003] NarraScore: Bridging Visual Narrative and Musical Dynamics via Hierarchical Affective Control
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于视频配乐生成任务，解决长视频配乐的连贯性和语义对齐问题。提出NarraScore框架，通过情感控制实现视觉叙事与音乐动态的融合。**

- **链接: [https://arxiv.org/pdf/2602.09070v1](https://arxiv.org/pdf/2602.09070v1)**

> **作者:** Yufan Wen; Zhaocheng Liu; YeGuo Hua; Ziyi Guo; Lihua Zhang; Chun Yuan; Jian Wu
>
> **摘要:** Synthesizing coherent soundtracks for long-form videos remains a formidable challenge, currently stalled by three critical impediments: computational scalability, temporal coherence, and, most critically, a pervasive semantic blindness to evolving narrative logic. To bridge these gaps, we propose NarraScore, a hierarchical framework predicated on the core insight that emotion serves as a high-density compression of narrative logic. Uniquely, we repurpose frozen Vision-Language Models (VLMs) as continuous affective sensors, distilling high-dimensional visual streams into dense, narrative-aware Valence-Arousal trajectories. Mechanistically, NarraScore employs a Dual-Branch Injection strategy to reconcile global structure with local dynamism: a \textit{Global Semantic Anchor} ensures stylistic stability, while a surgical \textit{Token-Level Affective Adapter} modulates local tension via direct element-wise residual injection. This minimalist design bypasses the bottlenecks of dense attention and architectural cloning, effectively mitigating the overfitting risks associated with data scarcity. Experiments demonstrate that NarraScore achieves state-of-the-art consistency and narrative alignment with negligible computational overhead, establishing a fully autonomous paradigm for long-video soundtrack generation.
>
---
#### [new 004] BioME: A Resource-Efficient Bioacoustic Foundational Model for IoT Applications
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于生物声学任务，旨在解决资源受限平台上的音频编码问题。提出BioME模型，通过知识蒸馏和多领域预训练，实现高效且泛化能力强的音频表征。**

- **链接: [https://arxiv.org/pdf/2602.09970v1](https://arxiv.org/pdf/2602.09970v1)**

> **作者:** Heitor R. Guimarães; Abhishek Tiwari; Mahsa Abdollahi; Anderson R. Avila; Tiago H. Falk
>
> **摘要:** Passive acoustic monitoring has become a key strategy in biodiversity assessment, conservation, and behavioral ecology, especially as Internet-of-Things (IoT) devices enable continuous in situ audio collection at scale. While recent self-supervised learning (SSL)-based audio encoders, such as BEATs and AVES, have shown strong performance in bioacoustic tasks, their computational cost and limited robustness to unseen environments hinder deployment on resource-constrained platforms. In this work, we introduce BioME, a resource-efficient audio encoder designed for bioacoustic applications. BioME is trained via layer-to-layer distillation from a high-capacity teacher model, enabling strong representational transfer while reducing the parameter count by 75%. To further improve ecological generalization, the model is pretrained on multi-domain data spanning speech, environmental sounds, and animal vocalizations. A key contribution is the integration of modulation-aware acoustic features via FiLM conditioning, injecting a DSP-inspired inductive bias that enhances feature disentanglement in low-capacity regimes. Across multiple bioacoustic tasks, BioME matches or surpasses the performance of larger models, including its teacher, while being suitable for resource-constrained IoT deployments. For reproducibility, code and pretrained checkpoints are publicly available.
>
---
#### [new 005] Gencho: Room Impulse Response Generation from Reverberant Speech and Text via Diffusion Transformers
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Gencho模型，用于生成房间脉冲响应（RIR），解决盲源RIR估计和生成问题，通过扩散Transformer实现从混响语音生成多样且真实的RIR。**

- **链接: [https://arxiv.org/pdf/2602.09233v1](https://arxiv.org/pdf/2602.09233v1)**

> **作者:** Jackie Lin; Jiaqi Su; Nishit Anand; Zeyu Jin; Minje Kim; Paris Smaragdis
>
> **备注:** In Proc. of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026. Audio examples available at https://linjac.github.io/Gencho/
>
> **摘要:** Blind room impulse response (RIR) estimation is a core task for capturing and transferring acoustic properties; yet existing methods often suffer from limited modeling capability and degraded performance under unseen conditions. Moreover, emerging generative audio applications call for more flexible impulse response generation methods. We propose Gencho, a diffusion-transformer-based model that predicts complex spectrogram RIRs from reverberant speech. A structure-aware encoder leverages isolation between early and late reflections to encode the input audio into a robust representation for conditioning, while the diffusion decoder generates diverse and perceptually realistic impulse responses from it. Gencho integrates modularly with standard speech processing pipelines for acoustic matching. Results show richer generated RIRs than non-generative baselines while maintaining strong performance in standard RIR metrics. We further demonstrate its application to text-conditioned RIR generation, highlighting Gencho's versatility for controllable acoustic simulation and generative audio tasks.
>
---
#### [new 006] Evaluation of acoustic Green's function in rectangular rooms with general surface impedance walls
- **分类: eess.AS; cs.CE; cs.SD**

- **简介: 该论文属于声学建模任务，旨在解决矩形房间中一般边界条件下的声学格林函数计算问题。通过引入一阶渐近分析和半解析方法，提高计算精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.09594v1](https://arxiv.org/pdf/2602.09594v1)**

> **作者:** Matteo Calafà; Yuanxin Xia; Jonas Brunskog; Cheol-Ho Jeong
>
> **摘要:** Acoustic room modes and the Green's function mode expansion are well-known for rectangular rooms with perfectly reflecting walls. First-order approximations also exist for nearly rigid boundaries; however, current analytical methods fail to accommodate more general boundary conditions, e.g., when wall absorption is significant. In this work, we present a comprehensive analysis that extends previous studies by including additional first-order asymptotics that account for soft-wall boundaries. In addition, we introduce a semi-analytical, efficient, and reliable method for computing the Green's function in rectangular rooms, which is described and validated through numerical tests. With a sufficiently large truncation order, the resulting error becomes negligible, making the method suitable as a benchmark for numerical simulations. Additional aspects regarding the spectral basis orthogonality and completeness are also addressed, providing a general framework for the validity of the proposed approach.
>
---
#### [new 007] Performance Comparison of CNN and AST Models with Stacked Features for Environmental Sound Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于环境声音分类任务，旨在比较基于特征堆叠的CNN与AST模型的性能。研究通过实验分析不同特征组合和训练策略，探索在数据有限情况下CNN的效率优势。**

- **链接: [https://arxiv.org/pdf/2602.09321v1](https://arxiv.org/pdf/2602.09321v1)**

> **作者:** Parinaz Binandeh Dehaghania; Danilo Penab; A. Pedro Aguiar
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** Environmental sound classification (ESC) has gained significant attention due to its diverse applications in smart city monitoring, fault detection, acoustic surveillance, and manufacturing quality control. To enhance CNN performance, feature stacking techniques have been explored to aggregate complementary acoustic descriptors into richer input representations. In this paper, we investigate CNN-based models employing various stacked feature combinations, including Log-Mel Spectrogram (LM), Spectral Contrast (SPC), Chroma (CH), Tonnetz (TZ), Mel-Frequency Cepstral Coefficients (MFCCs), and Gammatone Cepstral Coefficients (GTCC). Experiments are conducted on the widely used ESC-50 and UrbanSound8K datasets under different training regimes, including pretraining on ESC-50, fine-tuning on UrbanSound8K, and comparison with Audio Spectrogram Transformer (AST) models pretrained on large-scale corpora such as AudioSet. This experimental design enables an analysis of how feature-stacked CNNs compare with transformer-based models under varying levels of training data and pretraining diversity. The results indicate that feature-stacked CNNs offer a more computationally and data-efficient alternative when large-scale pretraining or extensive training data are unavailable, making them particularly well suited for resource-constrained and edge-level sound classification scenarios.
>
---
#### [new 008] Windowed SummaryMixing: An Efficient Fine-Tuning of Self-Supervised Learning Models for Low-resource Speech Recognition
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文针对低资源语音识别任务，提出Windowed SummaryMixing方法，解决自监督学习模型计算复杂度高、缺乏局部上下文的问题，通过引入局部与全局摘要提升性能并降低显存占用。**

- **链接: [https://arxiv.org/pdf/2602.09043v1](https://arxiv.org/pdf/2602.09043v1)**

> **作者:** Aditya Srinivas Menon; Kumud Tripathi; Raj Gohil; Pankaj Wasnik
>
> **备注:** The paper has been accepted at ICASSP 2026, Barcelona, Spain
>
> **摘要:** Self-supervised learning (SSL) has advanced speech processing but suffers from quadratic complexity due to self-attention. To address this, SummaryMixing (SM) has been proposed as a linear-time alternative that summarizes entire utterances using mean pooling but lacks sufficient local context. In this work, we introduce Windowed SummaryMixing (WSM), which enhances SM by integrating local neighborhood summaries alongside the global summary, maintaining efficiency while improving temporal dependencies. Additionally, we introduce a selective fine-tuning approach, replacing self-attention layers in SSL models with WSM blocks and fine-tuning only these blocks in low-resource settings. Our approach improves ASR performance while reducing peak VRAM usage by 40\% in the SSL models. WSM blocks have linear-time complexity with enhanced context awareness. Selectively replacing some attention layers reduces compute, memory, and latency, making it ideal for low-resource speech recognition.
>
---
#### [new 009] DSFlow: Dual Supervision and Step-Aware Architecture for One-Step Flow Matching Speech Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，解决流匹配模型推理步骤多、计算成本高的问题。提出DSFlow框架，通过双监督和步感知结构，提升生成质量并减少参数和计算量。**

- **链接: [https://arxiv.org/pdf/2602.09041v1](https://arxiv.org/pdf/2602.09041v1)**

> **作者:** Bin Lin; Peng Yang; Chao Yan; Xiaochen Liu; Wei Wang; Boyong Wu; Pengfei Tan; Xuerui Yang
>
> **摘要:** Flow-matching models have enabled high-quality text-to-speech synthesis, but their iterative sampling process during inference incurs substantial computational cost. Although distillation is widely used to reduce the number of inference steps, existing methods often suffer from process variance due to endpoint error accumulation. Moreover, directly reusing continuous-time architectures for discrete, fixed-step generation introduces structural parameter inefficiencies. To address these challenges, we introduce DSFlow, a modular distillation framework for few-step and one-step synthesis. DSFlow reformulates generation as a discrete prediction task and explicitly adapts the student model to the target inference regime. It improves training stability through a dual supervision strategy that combines endpoint matching with deterministic mean-velocity alignment, enforcing consistent generation trajectories across inference steps. In addition, DSFlow improves parameter efficiency by replacing continuous-time timestep conditioning with lightweight step-aware tokens, aligning model capacity with the significantly reduced timestep space of the discrete task. Extensive experiments across diverse flow-based text-to-speech architectures demonstrate that DSFlow consistently outperforms standard distillation approaches, achieving strong few-step and one-step synthesis quality while reducing model parameters and inference cost.
>
---
#### [new 010] Beyond the Utterance: An Empirical Study of Very Long Context Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，研究长上下文语音识别问题。传统ASR模型处理短语，本文探索使用长达一小时的音频序列，发现增加上下文可提升性能，并分析了模型结构对长序列处理的影响。**

- **链接: [https://arxiv.org/pdf/2602.09044v1](https://arxiv.org/pdf/2602.09044v1)**

> **作者:** Robert Flynn; Anton Ragni
>
> **备注:** Accepted to IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2026. doi: 10.1109/TASLPRO.2026.3658246
>
> **摘要:** Automatic speech recognition (ASR) models are normally trained to operate over single utterances, with a short duration of less than 30 seconds. This choice has been made in part due to computational constraints, but also reflects a common, but often inaccurate, modelling assumption that treats utterances as independent and identically distributed samples. When long-format audio recordings are available, to work with such systems, these recordings must first be segmented into short utterances and processed independently. In this work, we show that due to recent algorithmic and hardware advances, this is no longer necessary, and current attention-based approaches can be used to train ASR systems that operate on sequences of over an hour in length. Therefore, to gain a better understanding of the relationship between the training/evaluation sequence length and performance, we train ASR models on large-scale data using 10 different sequence lengths from 10 seconds up to 1 hour. The results show a benefit from using up to 21.8 minutes of context, with up to a 14.2% relative improvement from a short context baseline in our primary experiments. Through modifying various architectural components, we find that the method of encoding positional information and the model's width/depth are important factors when working with long sequences. Finally, a series of evaluations using synthetic data are constructed to help analyse the model's use of context. From these results, it is clear that both linguistic and acoustic aspects of the distant context are being used by the model.
>
---
#### [new 011] The SJTU X-LANCE Lab System for MSR Challenge 2025
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐源分离任务，解决音乐信号恢复问题。通过多阶段模型实现分离、去噪和去混响，提升音质。**

- **链接: [https://arxiv.org/pdf/2602.09042v1](https://arxiv.org/pdf/2602.09042v1)**

> **作者:** Jinxuan Zhu; Hao Qiu; Haina Zhu; Jianwei Yu; Kai Yu; Xie Chen
>
> **摘要:** This report describes the system submitted to the music source restoration (MSR) Challenge 2025. Our approach is composed of sequential BS-RoFormers, each dealing with a single task including music source separation (MSS), denoise and dereverb. To support 8 instruments given in the task, we utilize pretrained checkpoints from MSS community and finetune the MSS model with several training schemes, including (1) mixing and cleaning of datasets; (2) random mixture of music pieces for data augmentation; (3) scale-up of audio length. Our system achieved the first rank in all three subjective and three objective evaluation metrics, including an MMSNR score of 4.4623 and an FAD score of 0.1988. We have open-sourced all the code and checkpoints at https://github.com/ModistAndrew/xlance-msr.
>
---
#### [new 012] Stemphonic: All-at-once Flexible Multi-stem Music Generation
- **分类: cs.SD; cs.LG; cs.MM**

- **简介: 该论文属于音乐生成任务，解决多音轨同步生成效率低的问题。提出Stemphonic框架，实现一次推理生成多个同步音轨，提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2602.09891v1](https://arxiv.org/pdf/2602.09891v1)**

> **作者:** Shih-Lun Wu; Ge Zhu; Juan-Pablo Caceres; Cheng-Zhi Anna Huang; Nicholas J. Bryan
>
> **备注:** Accepted for publication at Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Music stem generation, the task of producing musically-synchronized and isolated instrument audio clips, offers the potential of greater user control and better alignment with musician workflows compared to conventional text-to-music models. Existing stem generation approaches, however, either rely on fixed architectures that output a predefined set of stems in parallel, or generate only one stem at a time, resulting in slow inference despite flexibility in stem combination. We propose Stemphonic, a diffusion-/flow-based framework that overcomes this trade-off and generates a variable set of synchronized stems in one inference pass. During training, we treat each stem as a batch element, group synchronized stems in a batch, and apply a shared noise latent to each group. At inference-time, we use a shared initial noise latent and stem-specific text inputs to generate synchronized multi-stem outputs in one pass. We further expand our approach to enable one-pass conditional multi-stem generation and stem-wise activity controls to empower users to iteratively generate and orchestrate the temporal layering of a mix. We benchmark our results on multiple open-source stem evaluation sets and show that Stemphonic produces higher-quality outputs while accelerating the full mix generation process by 25 to 50%. Demos at: https://stemphonic-demo.vercel.app.
>
---
#### [new 013] Evaluating Disentangled Representations for Controllable Music Generation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在评估可控制生成中解耦表示的有效性。研究分析了不同解耦策略，发现现有方法未能实现真正解耦，提出需重新审视可控性设计。**

- **链接: [https://arxiv.org/pdf/2602.10058v1](https://arxiv.org/pdf/2602.10058v1)**

> **作者:** Laura Ibáñez-Martínez; Chukwuemeka Nkama; Andrea Poltronieri; Xavier Serra; Martín Rocamora
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Recent approaches in music generation rely on disentangled representations, often labeled as structure and timbre or local and global, to enable controllable synthesis. Yet the underlying properties of these embeddings remain underexplored. In this work, we evaluate such disentangled representations in a set of music audio models for controllable generation using a probing-based framework that goes beyond standard downstream tasks. The selected models reflect diverse unsupervised disentanglement strategies, including inductive biases, data augmentations, adversarial objectives, and staged training procedures. We further isolate specific strategies to analyze their effect. Our analysis spans four key axes: informativeness, equivariance, invariance, and disentanglement, which are assessed across datasets, tasks, and controlled transformations. Our findings reveal inconsistencies between intended and actual semantics of the embeddings, suggesting that current strategies fall short of producing truly disentangled representations, and prompting a re-examination of how controllability is approached in music generation.
>
---
#### [new 014] Covo-Audio Technical Report
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Covo-Audio，一个7B参数的端到端音频语言模型，解决音频理解和对话生成任务，通过预训练和微调实现高质量语音交互。**

- **链接: [https://arxiv.org/pdf/2602.09823v1](https://arxiv.org/pdf/2602.09823v1)**

> **作者:** Wenfu Wang; Chenxing Li; Liqiang Zhang; Yiyang Zhao; Yuxiang Zou; Hanzhao Li; Mingyu Cui; Hao Zhang; Kun Wei; Le Xu; Zikang Huang; Jiajun Xu; Jiliang Hu; Xiang He; Zeyu Xie; Jiawen Kang; Youjun Chen; Meng Yu; Dong Yu; Rilin Chen; Linlin Di; Shulin Feng; Na Hu; Yang Liu; Bang Wang; Shan Yang
>
> **备注:** Technical Report
>
> **摘要:** In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs.
>
---
#### [new 015] Positive-Unlabelled Active Learning to Curate a Dataset for Orca Resident Interpretation
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于数据集构建任务，旨在解决海洋哺乳动物声学数据的标注问题。通过主动学习方法，构建了大规模的SRKW声学数据集，用于生态研究与保护。**

- **链接: [https://arxiv.org/pdf/2602.09295v1](https://arxiv.org/pdf/2602.09295v1)**

> **作者:** Bret Nestor; Bohan Yao; Jasmine Moore; Jasper Kanes
>
> **摘要:** This work presents the largest curation of Southern Resident Killer Whale (SRKW) acoustic data to date, also containing other marine mammals in their environment. We systematically search all available public archival hydrophone data within the SRKW habitat (over 30 years of audio data). The search consists of a weakly-supervised, positive-unlabelled, active learning strategy to identify all instances of marine mammals. The resulting transformer-based detectors outperform state-of-the-art detectors on the DEEPAL, DCLDE-2026, and two newly introduced expert-annotated datasets in terms of accuracy, energy efficiency, and speed. The detection model has a specificity of 0-28.8% at 95% sensitivity. Our multiclass species classifier obtains a top-1 accuracy of 42.1% (11 train classes, 4 test classes) and our ecotype classifier obtains a top-1 accuracy of 43.0% (4 train classes, 5 test classes) on the DCLDE-2026 dataset. We yield 919 hours of SRKW data, 230 hours of Bigg's orca data, 1374 hours of orca data from unlabelled ecotypes, 1501 hours of humpback data, 88 hours of sea lion data, 246 hours of pacific white-sided dolphin data, and over 784 hours of unspecified marine mammal data. This SRKW dataset is larger than DCLDE-2026, Ocean Networks Canada, and OrcaSound combined. The curated species labels are available under CC-BY 4.0 license, and the corresponding audio data are available under the licenses of the original owners. The comprehensive nature of this dataset makes it suitable for unsupervised machine translation, habitat usage surveys, and conservation endeavours for this critically endangered ecotype.
>
---
#### [new 016] AI-Driven Cardiorespiratory Signal Processing: Separation, Clustering, and Anomaly Detection
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文属于心血管信号处理任务，旨在分离、聚类和检测心肺声音异常。通过AI模型与新型传感器技术结合，提升医疗诊断智能化水平。**

- **链接: [https://arxiv.org/pdf/2602.09210v1](https://arxiv.org/pdf/2602.09210v1)**

> **作者:** Yasaman Torabi
>
> **备注:** PhD thesis
>
> **摘要:** This research applies artificial intelligence (AI) to separate, cluster, and analyze cardiorespiratory sounds. We recorded a new dataset (HLS-CMDS) and developed several AI models, including generative AI methods based on large language models (LLMs) for guided separation, explainable AI (XAI) techniques to interpret latent representations, variational autoencoders (VAEs) for waveform separation, a chemistry-inspired non-negative matrix factorization (NMF) algorithm for clustering, and a quantum convolutional neural network (QCNN) designed to detect abnormal physiological patterns. The performance of these AI models depends on the quality of the recorded signals. Therefore, this thesis also reviews the biosensing technologies used to capture biomedical data. It summarizes developments in microelectromechanical systems (MEMS) acoustic sensors and quantum biosensors, such as quantum dots and nitrogen-vacancy centers. It further outlines the transition from electronic integrated circuits (EICs) to photonic integrated circuits (PICs) and early progress toward integrated quantum photonics (IQP) for chip-based biosensing. Together, these studies show how AI and next-generation sensors can support more intelligent diagnostic systems for future healthcare.
>
---
## 更新

#### [replaced 001] Quantifying Multimodal Imbalance: A GMM-Guided Adaptive Loss for Audio-Visual Learning
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多模态学习任务，旨在解决模态不平衡问题。通过引入模态差距指标和GMM模型，动态调整损失函数，提升模型对不平衡样本的处理能力。**

- **链接: [https://arxiv.org/pdf/2510.21797v3](https://arxiv.org/pdf/2510.21797v3)**

> **作者:** Zhaocheng Liu; Zhiwen Yu; Xiaoqing Liu
>
> **摘要:** Multimodal learning integrates diverse modalities but suffers from modality imbalance, where dominant modalities suppress weaker ones due to inconsistent convergence rates. Existing methods predominantly rely on static modulation or heuristics, overlooking sample-level distributional variations in prediction bias. Specifically, they fail to distinguish outlier samples where the modality gap is exacerbated by low data quality. We propose a framework to quantitatively diagnose and dynamically mitigate this imbalance at the sample level. We introduce the Modality Gap metric to quantify prediction discrepancies. Analysis reveals that this gap follows a bimodal distribution, indicating the coexistence of balanced and imbalanced sample subgroups. We employ a Gaussian Mixture Model (GMM) to explicitly model this distribution, leveraging Bayesian posterior probabilities for soft subgroup separation. Our two-stage framework comprises a Warm-up stage and an Adaptive Training stage. In the latter, a GMM-guided Adaptive Loss dynamically reallocates optimization priorities: it imposes stronger alignment penalties on imbalanced samples to rectify bias, while prioritizing fusion for balanced samples to maximize complementary information. Experiments on CREMA-D, AVE, and KineticSound demonstrate that our method significantly outperforms SOTA baselines. Furthermore, we show that fine-tuning on a GMM-filtered balanced subset serves as an effective data purification strategy, yielding substantial gains by eliminating extreme noisy samples even without the adaptive loss.
>
---
#### [replaced 002] Controllable Dance Generation with Style-Guided Motion Diffusion
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于舞蹈生成任务，旨在解决舞蹈生成中可控性不足和音乐风格匹配不佳的问题。提出SGMD模型，结合音乐特征与风格提示，生成符合音乐风格的舞蹈序列。**

- **链接: [https://arxiv.org/pdf/2406.07871v3](https://arxiv.org/pdf/2406.07871v3)**

> **作者:** Hongsong Wang; Ying Zhu; Xin Geng; Liang Wang
>
> **摘要:** Dance plays an important role as an artistic form and expression in human culture, yet automatically generating dance sequences is a significant yet challenging endeavor. Existing approaches often neglect the critical aspect of controllability in dance generation. Additionally, they inadequately model the nuanced impact of music styles, resulting in dances that lack alignment with the expressive characteristics inherent in the conditioned music. To address this gap, we propose Style-Guided Motion Diffusion (SGMD), which integrates the Transformer-based architecture with a Style Modulation module. By incorporating music features with user-provided style prompts, the SGMD ensures that the generated dances not only match the musical content but also reflect the desired stylistic characteristics. To enable flexible control over the generated dances, we introduce a spatial-temporal masking mechanism. As controllable dance generation has not been fully studied, we construct corresponding experimental setups and benchmarks for tasks such as trajectory-based dance generation, dance in-betweening, and dance inpainting. Extensive experiments demonstrate that our approach can generate realistic and stylistically consistent dances, while also empowering users to create dances tailored to diverse artistic and practical needs. Code is available on Github: https://github.com/mucunzhuzhu/DGSDP
>
---
#### [replaced 003] Bayesian Speech Synthesizers Can Learn from Multiple Teachers
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，解决TTS中确定性预测无法捕捉自然语音不确定性的难题。提出BELLE框架，通过贝叶斯推理建模不确定性，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2510.24372v3](https://arxiv.org/pdf/2510.24372v3)**

> **作者:** Ziyang Zhang; Yifan Gao; Xuenan Xu; Baoxiang Li; Wen Wu; Chao Zhang
>
> **备注:** Code is available at https://github.com/OpenTSLab/BELLE
>
> **摘要:** Text-to-Speech (TTS) is inherently a "one-to-many" mapping characterized by intrinsic uncertainty, yet current paradigms often oversimplify it into a deterministic regression task. While continuous-valued autoregressive (AR) models have recently emerged as a promising alternative to discrete codec-based approaches, they typically rely on a fixed-variance prior, fundamentally constraining generation to a static point estimate that ignores the dynamic variability of natural speech. To bridge this gap, we propose BELLE (Bayesian evidential learning with language modelling), a framework that shifts from deterministic prediction to principled Bayesian inference without increasing model parameters or inference latency. By modeling the acoustic target as a Normal-Inverse-Gamma distribution, BELLE captures data-dependent aleatoric uncertainty. To enable accurate variance estimation on standard single-reference datasets, we introduce a "one-to-many" training strategy that leverages synthetic samples as a statistical support set, allowing the model to learn robust distributional properties rather than merely imitating teacher artifacts. Experiments demonstrate that BELLE, trained on only ~5k hours of data, outperforms leading open-source models trained on 50k hours (achieving a 25.8% relative WER reduction) and naturally supports high-quality streaming generation. Audio samples are available at https://belletts.github.io/Belle/.
>
---
#### [replaced 004] MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification
- **分类: cs.CL; cs.LG; cs.NE; cs.SD**

- **简介: 该论文提出MEGConformer，用于从MEG信号中解码语音和音素信息，解决脑机接口中的信号解析问题。通过改进的Conformer结构和数据增强方法，提升了任务性能。**

- **链接: [https://arxiv.org/pdf/2512.01443v2](https://arxiv.org/pdf/2512.01443v2)**

> **作者:** Xabier de Zuazo; Ibon Saratxaga; Eva Navas
>
> **备注:** 8 pages, 7 figures, 4 tables, v1 presentend in LibriBrain Workshop, NeurIPS 2025; v2 submitted to Odyssey 2026
>
> **摘要:** Decoding speech-related information from non-invasive MEG is a key step toward scalable brain-computer interfaces. We present compact Conformer-based decoders on the LibriBrain 2025 PNPL benchmark for two core tasks: Speech Detection and Phoneme Classification. Our approach adapts a compact Conformer to raw 306-channel MEG signals, with a lightweight convolutional projection layer and task-specific heads. For Speech Detection, a MEG-oriented SpecAugment provided a first exploration of MEG-specific augmentation. For Phoneme Classification, we used inverse-square-root class weighting and a dynamic grouping loader to handle 100-sample averaged examples. In addition, a simple instance-level normalization proved critical to mitigate distribution shifts on the holdout split. Using the official Standard track splits and F1-macro for model selection, our best systems achieved 88.9% (Speech) and 65.8% (Phoneme) on the leaderboard, winning the Phoneme Classification Standard track. For further implementation details, the technical documentation, source code, and checkpoints are available at https://github.com/neural2speech/libribrain-experiments.
>
---
#### [replaced 005] MOVA: Towards Scalable and Synchronized Video-Audio Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出MOVA，解决视频音频同步生成问题，采用MoE架构实现高质量多模态内容生成。**

- **链接: [https://arxiv.org/pdf/2602.08794v2](https://arxiv.org/pdf/2602.08794v2)**

> **作者:** SII-OpenMOSS Team; :; Donghua Yu; Mingshu Chen; Qi Chen; Qi Luo; Qianyi Wu; Qinyuan Cheng; Ruixiao Li; Tianyi Liang; Wenbo Zhang; Wenming Tu; Xiangyu Peng; Yang Gao; Yanru Huo; Ying Zhu; Yinze Luo; Yiyang Zhang; Yuerong Song; Zhe Xu; Zhiyu Zhang; Chenchen Yang; Cheng Chang; Chushu Zhou; Hanfu Chen; Hongnan Ma; Jiaxi Li; Jingqi Tong; Junxi Liu; Ke Chen; Shimin Li; Shiqi Jiang; Songlin Wang; Wei Jiang; Zhaoye Fei; Zhiyuan Ning; Chunguo Li; Chenhui Li; Ziwei He; Zengfeng Huang; Xie Chen; Xipeng Qiu
>
> **备注:** Technical report for MOVA (open-source video-audio generation model). 38 pages, 10 figures, 22 tables. Project page: https://mosi.cn/models/mova Code: https://github.com/OpenMOSS/MOVA Models: https://huggingface.co/collections/OpenMOSS-Team/mova. Qinyuan Cheng and Tianyi Liang are project leader. Xie Chen and Xipeng Qiu are corresponding authors
>
> **摘要:** Audio is indispensable for real-world video, yet generation models have largely overlooked audio components. Current approaches to producing audio-visual content often rely on cascaded pipelines, which increase cost, accumulate errors, and degrade overall quality. While systems such as Veo 3 and Sora 2 emphasize the value of simultaneous generation, joint multimodal modeling introduces unique challenges in architecture, data, and training. Moreover, the closed-source nature of existing systems limits progress in the field. In this work, we introduce MOVA (MOSS Video and Audio), an open-source model capable of generating high-quality, synchronized audio-visual content, including realistic lip-synced speech, environment-aware sound effects, and content-aligned music. MOVA employs a Mixture-of-Experts (MoE) architecture, with a total of 32B parameters, of which 18B are active during inference. It supports IT2VA (Image-Text to Video-Audio) generation task. By releasing the model weights and code, we aim to advance research and foster a vibrant community of creators. The released codebase features comprehensive support for efficient inference, LoRA fine-tuning, and prompt enhancement.
>
---
#### [replaced 006] No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting
- **分类: cs.SD**

- **简介: 该论文属于开放词汇关键词识别任务，旨在解决因前缀偏差导致的误触发问题。通过引入POB数据集和EPS方法，有效提升了模型性能。**

- **链接: [https://arxiv.org/pdf/2602.08930v2](https://arxiv.org/pdf/2602.08930v2)**

> **作者:** Yi Liu; Chuan-Che Jeff Huang; Xiao Quan
>
> **备注:** Published in ICASSP 2026
>
> **摘要:** Open-vocabulary keyword spotting (OV-KWS) enables personalized device control via arbitrary voice commands. Recently, researchers have explored using audio-text joint embeddings, allowing users to enroll phrases with text, and proposed techniques to disambiguate similar utterances. We find that existing OV-KWS solutions often overly bias the beginning phonemes of an enrollment, causing false triggers when negative enrollment-query-pairs share a prefix (``turn the volume up'' vs. ``turn the volume down''). We trace this to two factors: training data bias and position-biased cross-modal scoring. To address these limitations, we introduce the Partial Overlap Benchmark (POB) with two datasets, POB-Spark and POB-LibriPhrase (POB-LP), containing mismatched audio-text pairs with shared prefixes, and propose Equal-weighting Position Scoring (EPS), a lightweight decision layer. Using EPS alone reduces EER on POB-Spark from 64.4\% to 29.3\% and improves POB-LP accuracy from 87.6\% to 96.8\%, while maintaining performance on LibriPhrase and Google Speech Commands (GSC). With POB data added in training, our work achieves the best POB benchmark results while incurring the least amount of degradation on prior metrics among baselines. This degradation is most pronounced in GSC, which contains only one-word commands. We surface mitigating this trade-off as future work.
>
---
#### [replaced 007] TTA: Transcribe, Translate and Alignment for Cross-lingual Speech Representation
- **分类: eess.AS**

- **简介: 该论文提出TTA模型，解决跨语言语音表示问题，通过语音转录、翻译和对齐提升语音-语言模型集成效果。**

- **链接: [https://arxiv.org/pdf/2511.14410v2](https://arxiv.org/pdf/2511.14410v2)**

> **作者:** Wei Liu; Jiahong Li; Yiwen Shao; Dong Yu
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Speech-LLM models have demonstrated great performance in multi-modal and multi-task speech understanding. A typical speech-LLM paradigm is integrating speech modality with a large language model (LLM). While the Whisper encoder was frequently adopted in previous studies for speech input, it shows limitations regarding input format, model scale, and semantic performance. To this end, we propose a lightweight TTA model specialized in speech semantics for more effective LLM integration. With large-scale training of 358k hours of speech data on multilingual speech recognition (ASR), speech translation (ST) and speech-text alignment tasks, TTA is capable of producing robust cross-lingual speech representations. Extensive evaluations across diverse benchmarks, including ASR/ST, speech retrieval, and ASR-LLM performance assessments, demonstrate TTA's superiority over Whisper. Furthermore, we rigorously validate the interplay between cross-lingual capabilities and ASR/ST performance. The model weights and training recipes of TTA will be released as part of an audio understanding toolkit Auden.
>
---
#### [replaced 008] Diffusion-based Signal Refiner for Speech Enhancement and Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强与分离任务，旨在提升语音的感知质量。通过扩散模型学习干净语音先验，优化现有系统的输出，去除处理中的不自然伪影。**

- **链接: [https://arxiv.org/pdf/2305.05857v3](https://arxiv.org/pdf/2305.05857v3)**

> **作者:** Masato Hirano; Ryosuke Sawata; Naoki Murata; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted to IEEE/ACM TASLP. The first two authors contributed equally. Code: https://github.com/sony/diffiner
>
> **摘要:** Although recent speech processing technologies have achieved significant improvements in objective metrics, there still remains a gap in human perceptual quality. This paper proposes Diffiner, a novel solution that utilizes the powerful generative capability of diffusion models' prior distributions to address this fundamental issue. Diffiner leverages the probabilistic generative framework of diffusion models and learns natural prior distributions of clean speech to convert outputs from existing speech processing systems into perceptually natural high-quality audio. In contrast to conventional deterministic approaches, our method simultaneously analyzes both the original degraded speech and the pre-processed speech to accurately identify unnatural artifacts introduced during processing. Then, through the iterative sampling process of the diffusion model, these degraded portions are replaced with perceptually natural and high-quality speech segments. Experimental results indicate that Diffiner can recover a clearer harmonic structure of speech, which is shown to result in improved perceptual quality w.r.t. several metrics as well as in a human listening test. This highlights Diffiner's efficacy as a versatile post-processor for enhancing existing speech processing pipelines.
>
---
#### [replaced 009] Deep Room Impulse Response Completion
- **分类: eess.AS**

- **简介: 该论文提出“RIR补全”任务，解决长混响响应生成效率低的问题。通过深度学习预测后期混响，提升VR和游戏中的音频渲染效果。**

- **链接: [https://arxiv.org/pdf/2402.00859v2](https://arxiv.org/pdf/2402.00859v2)**

> **作者:** Jackie Lin; Georg Götz; Sebastian J. Schlecht
>
> **备注:** This version corresponds to the published article in EURASIP Journal on Audio, Speech, and Music Processing (2025)
>
> **摘要:** Rendering immersive spatial audio in virtual reality (VR) and video games demands a fast and accurate generation of room impulse responses (RIRs) to recreate auditory environments plausibly. However, the conventional methods for simulating or measuring long RIRs are either computationally intensive or challenged by low signal-to-noise ratios. This study is propelled by the insight that direct sound and early reflections encapsulate sufficient information about room geometry and absorption characteristics. Building upon this premise, we propose a novel task termed "RIR completion," aimed at synthesizing the late reverberation given only the early portion (50 ms) of the response. To this end, we introduce DECOR, Deep Exponential Completion Of Room impulse responses, a deep neural network structured as an autoencoder designed to predict multi-exponential decay envelopes of filtered noise sequences. The interpretability of DECOR's output facilitates its integration with diverse rendering techniques. The proposed method is compared against an adapted state-of-the-art network, and comparable performance shows promising results supporting the feasibility of the RIR completion task. The RIR completion can be widely adapted to enhance RIR generation tasks where fast late reverberation approximation is required.
>
---
