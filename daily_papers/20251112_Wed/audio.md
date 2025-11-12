# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Quantizing Whisper-small: How design choices affect ASR performance
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究Whisper-small的后训练量化（PTQ），旨在解决其在边缘设备部署难的问题。对比四类库的量化方案，发现动态int8量化在压缩57%模型体积下仍提升词错误率，实现高效无重训练部署。**

- **链接: []()**

> **作者:** Arthur Söhler; Julian Irigoyen; Andreas Søeborg Kirkedal
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large speech recognition models like Whisper-small achieve high accuracy but are difficult to deploy on edge devices due to their high computational demand. To this end, we present a unified, cross-library evaluation of post-training quantization (PTQ) on Whisper-small that disentangles the impact of quantization scheme, method, granularity, and bit-width. Our study is based on four libraries: PyTorch, Optimum-Quanto, HQQ, and bitsandbytes. Experiments on LibriSpeech test-clean and test-other show that dynamic int8 quantization with Quanto offers the best trade-off, reducing model size by 57% while improving on the baseline's word error rate. Static quantization performed worse, likely due to Whisper's Transformer architecture, while more aggressive formats (e.g., nf4, int3) achieved up to 71% compression at the cost of accuracy in noisy conditions. Overall, our results demonstrate that carefully chosen PTQ methods can substantially reduce model size and inference cost without retraining, enabling efficient deployment of Whisper-small on constrained hardware.
>
---
#### [new 002] Enabling Automatic Self-Talk Detection via Earables
- **分类: cs.SD; cs.AI**

- **简介: 论文提出MutterMeter，首次实现耳戴设备对日常环境中自言自语的自动检测，解决传统语音模型难以处理其碎片化、非规则性的问题，通过多模态层次分类实现高精度识别（F1=0.84）。**

- **链接: []()**

> **作者:** Euihyeok Lee; Seonghyeon Kim; SangHun Im; Heung-Seon Oh; Seungwoo Kang
>
> **摘要:** Self-talk-an internal dialogue that can occur silently or be spoken aloud-plays a crucial role in emotional regulation, cognitive processing, and motivation, yet has remained largely invisible and unmeasurable in everyday life. In this paper, we present MutterMeter, a mobile system that automatically detects vocalized self-talk from audio captured by earable microphones in real-world settings. Detecting self-talk is technically challenging due to its diverse acoustic forms, semantic and grammatical incompleteness, and irregular occurrence patterns, which differ fundamentally from assumptions underlying conventional speech understanding models. To address these challenges, MutterMeter employs a hierarchical classification architecture that progressively integrates acoustic, linguistic, and contextual information through a sequential processing pipeline, adaptively balancing accuracy and computational efficiency. We build and evaluate MutterMeter using a first-of-its-kind dataset comprising 31.1 hours of audio collected from 25 participants. Experimental results demonstrate that MutterMeter achieves robust performance with a macro-averaged F1 score of 0.84, outperforming conventional approaches, including LLM-based and speech emotion recognition models.
>
---
#### [new 003] Pruning as Regularization: Sensitivity-Aware One-Shot Pruning in ASR
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文将剪枝视为ASR中的正则化手段，提出敏感性感知的一次性剪枝方法，识别模型冗余结构（如编码器末层），在不微调下显著降低WER，提升泛化能力，并在高稀疏度下保持性能，重塑剪枝为架构设计工具。**

- **链接: []()**

> **作者:** Julian Irigoyen; Arthur Söhler; Andreas Søeborg Kirkedal
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We challenge the conventional view of neural network pruning as solely a compression technique, demonstrating that one-shot magnitude pruning serves as a powerful implicit regularizer for ASR. Using Whisper-small, we combine gradient- and Fisher-based sensitivity diagnostics with targeted, component-wise pruning. This reveals architectural asymmetries: decoder FFNs are pruning-fragile, whereas decoder self-attention and the last encoder layers contain redundancy that, when removed, improves generalization. Without fine-tuning, pruning 50% of decoder self-attention reduces WER by 2.38% absolute (20.44% relative) on LibriSpeech test-other; pruning the last four encoder layers at 50% instead yields a 1.72% absolute (14.8% relative) improvement. Gains persisted on Common Voice and TED-LIUM datasets. Beyond regularization benefits, our sensitivity-aware approach enables more aggressive one-shot compression. At 40% sparsity, where established global pruning approaches catastrophically fail, our method preserves near-baseline accuracy. This positions pruning as a first-class architectural design tool: knowing where to prune is as important as how much to prune.
>
---
#### [new 004] Uncertainty Calibration of Multi-Label Bird Sound Classifiers
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究生物声学中多标签鸟鸣分类器的不确定性校准问题，系统评估四类模型的校准性能，发现其存在欠置信或过置信现象，并证明简单后校准方法（如Platt缩放）可显著提升校准效果。**

- **链接: []()**

> **作者:** Raphael Schwinger; Ben McEwen; Vincent S. Kather; René Heinrich; Lukas Rauch; Sven Tomforde
>
> **备注:** Under review at ICAART 2026
>
> **摘要:** Passive acoustic monitoring enables large-scale biodiversity assessment, but reliable classification of bioacoustic sounds requires not only high accuracy but also well-calibrated uncertainty estimates to ground decision-making. In bioacoustics, calibration is challenged by overlapping vocalisations, long-tailed species distributions, and distribution shifts between training and deployment data. The calibration of multi-label deep learning classifiers within the domain of bioacoustics has not yet been assessed. We systematically benchmark the calibration of four state-of-the-art multi-label bird sound classifiers on the BirdSet benchmark, evaluating both global, per-dataset and per-class calibration using threshold-free calibration metrics (ECE, MCS) alongside discrimination metrics (cmAP). Model calibration varies significantly across datasets and classes. While Perch v2 and ConvNeXt$_{BS}$ show better global calibration, results vary between datasets. Both models indicate consistent underconfidence, while AudioProtoPNet and BirdMAE are mostly overconfident. Surprisingly, calibration seems to be better for less frequent classes. Using simple post hoc calibration methods we demonstrate a straightforward way to improve calibration. A small labelled calibration set is sufficient to significantly improve calibration with Platt scaling, while global calibration parameters suffer from dataset variability. Our findings highlight the importance of evaluating and improving uncertainty calibration in bioacoustic classifiers.
>
---
#### [new 005] DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究基于LLM生成的更真实声学场景的DOA估计任务，提出轻量级模型LightDOA，利用深度可分离卷积提升多通道音频在复杂环境中的估计精度与效率，解决传统方法泛化性差与计算开销大的问题。**

- **链接: []()**

> **作者:** Haowen Li; Zhengding Luo; Dongyuan Shi; Boxiang Wang; Junwei Ji; Ziyi Yang; Woon-Seng Gan
>
> **摘要:** Direction-of-Arrival (DOA) estimation is critical in spatial audio and acoustic signal processing, with wide-ranging applications in real-world. Most existing DOA models are trained on synthetic data by convolving clean speech with room impulse responses (RIRs), which limits their generalizability due to constrained acoustic diversity. In this paper, we revisit DOA estimation using a recently introduced dataset constructed with the assistance of large language models (LLMs), which provides more realistic and diverse spatial audio scenes. We benchmark several representative neural-based DOA methods on this dataset and propose LightDOA, a lightweight DOA estimation model based on depthwise separable convolutions, specifically designed for mutil-channel input in varying environments. Experimental results show that LightDOA achieves satisfactory accuracy and robustness across various acoustic scenes while maintaining low computational complexity. This study not only highlights the potential of spatial audio synthesized with the assistance of LLMs in advancing robust and efficient DOA estimation research, but also highlights LightDOA as efficient solution for resource-constrained applications.
>
---
#### [new 006] Automatic Music Mixing using a Generative Model of Effect Embeddings
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MEGAMI，一种生成式音乐混音框架，解决传统方法将混音视为确定性回归而忽略多解性的问题。通过条件生成嵌入与排列等变架构，实现对未标注音轨的高质量自动混音，逼近人耳水平。**

- **链接: []()**

> **作者:** Eloi Moliner; Marco A. Martínez-Ramírez; Junghyun Koo; Wei-Hsiang Liao; Kin Wai Cheuk; Joan Serrà; Vesa Välimäki; Yuki Mitsufuji
>
> **备注:** submitted to IEEE ICASSP 2026
>
> **摘要:** Music mixing involves combining individual tracks into a cohesive mixture, a task characterized by subjectivity where multiple valid solutions exist for the same input. Existing automatic mixing systems treat this task as a deterministic regression problem, thus ignoring this multiplicity of solutions. Here we introduce MEGAMI (Multitrack Embedding Generative Auto MIxing), a generative framework that models the conditional distribution of professional mixes given unprocessed tracks. MEGAMI uses a track-agnostic effects processor conditioned on per-track generated embeddings, handles arbitrary unlabeled tracks through a permutation-equivariant architecture, and enables training on both dry and wet recordings via domain adaptation. Our objective evaluation using distributional metrics shows consistent improvements over existing methods, while listening tests indicate performances approaching human-level quality across diverse musical genres.
>
---
#### [new 007] SpeechJudge: Towards Human-Level Judgment for Speech Naturalness
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 论文提出SpeechJudge，面向语音自然度评估，解决人类偏好数据稀缺问题。构建99K语音对数据集、评估基准与生成奖励模型GRM，显著提升模型对人类判断的对齐能力。**

- **链接: []()**

> **作者:** Xueyao Zhang; Chaoren Wang; Huan Liao; Ziniu Li; Yuancheng Wang; Li Wang; Dongya Jia; Yuanzhe Chen; Xiulin Li; Zhuo Chen; Zhizheng Wu
>
> **备注:** Project Page: https://speechjudge.github.io/
>
> **摘要:** Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness--one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences.
>
---
#### [new 008] Speech Emotion Recognition with Phonation Excitation Information and Articulatory Kinematics
- **分类: cs.SD; cs.LG**

- **简介: 该论文面向语音情感识别（SER）任务，旨在利用语音产生的生理信息（发声激励与发音动力学）提升识别性能。作者构建了含EGG/EMA数据的STEM-E2VA数据集，并探索了通过语音逆推生理信号的可行性，验证了生理信息对SER的有效性与实用潜力。**

- **链接: []()**

> **作者:** Ziqian Zhang; Min Huang; Zhongzhe Xiao
>
> **摘要:** Speech emotion recognition (SER) has advanced significantly for the sake of deep-learning methods, while textual information further enhances its performance. However, few studies have focused on the physiological information during speech production, which also encompasses speaker traits, including emotional states. To bridge this gap, we conducted a series of experiments to investigate the potential of the phonation excitation information and articulatory kinematics for SER. Due to the scarcity of training data for this purpose, we introduce a portrayed emotional dataset, STEM-E2VA, which includes audio and physiological data such as electroglottography (EGG) and electromagnetic articulography (EMA). EGG and EMA provide information of phonation excitation and articulatory kinematics, respectively. Additionally, we performed emotion recognition using estimated physiological data derived through inversion methods from speech, instead of collected EGG and EMA, to explore the feasibility of applying such physiological information in real-world SER. Experimental results confirm the effectiveness of incorporating physiological information about speech production for SER and demonstrate its potential for practical use in real-world scenarios.
>
---
#### [new 009] SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech
- **分类: cs.SD**

- **简介: 该论文提出SynTTS-Commands，一个基于TTS合成的多语言语音命令数据集，解决边缘设备KWS训练数据稀缺问题，实现英语99.5%、中文98%的识别准确率，验证合成语音可替代真人录音。**

- **链接: []()**

> **作者:** Lu Gan; Xi Li
>
> **摘要:** The development of high-performance, on-device keyword spotting (KWS) systems for ultra-low-power hardware is critically constrained by the scarcity of specialized, multi-command training datasets. Traditional data collection through human recording is costly, slow, and lacks scalability. This paper introduces SYNTTS-COMMANDS, a novel, multilingual voice command dataset entirely generated using state-of-the-art Text-to-Speech (TTS) synthesis. By leveraging the CosyVoice 2 model and speaker embeddings from public corpora, we created a scalable collection of English and Chinese commands. Extensive benchmarking across a range of efficient acoustic models demonstrates that our synthetic dataset enables exceptional accuracy, achieving up to 99.5\% on English and 98\% on Chinese command recognition. These results robustly validate that synthetic speech can effectively replace human-recorded audio for training KWS classifiers. Our work directly addresses the data bottleneck in TinyML, providing a practical, scalable foundation for building private, low-latency, and energy-efficient voice interfaces on resource-constrained edge devices.
>
---
#### [new 010] Speech Separation for Hearing-Impaired Children in the Classroom
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对听障儿童在教室环境中语音分离困难的问题，提出基于MIMO-TasNet的多通道语音分离模型，利用空间线索并结合教室场景数据训练，实现高效迁移学习，显著提升对儿童语音的分离效果与鲁棒性。**

- **链接: []()**

> **作者:** Feyisayo Olalere; Kiki van der Heijden; H. Christiaan Stronks; Jeroen Briaire; Johan H. M. Frijns; Yagmur Güçlütürk
>
> **备注:** 13 pages
>
> **摘要:** Classroom environments are particularly challenging for children with hearing impairments, where background noise, multiple talkers, and reverberation degrade speech perception. These difficulties are greater for children than adults, yet most deep learning speech separation models for assistive devices are developed using adult voices in simplified, low-reverberation conditions. This overlooks both the higher spectral similarity of children's voices, which weakens separation cues, and the acoustic complexity of real classrooms. We address this gap using MIMO-TasNet, a compact, low-latency, multi-channel architecture suited for real-time deployment in bilateral hearing aids or cochlear implants. We simulated naturalistic classroom scenes with moving child-child and child-adult talker pairs under varying noise and distance conditions. Training strategies tested how well the model adapts to children's speech through spatial cues. Models trained on adult speech, classroom data, and finetuned variants were compared to assess data-efficient adaptation. Results show that adult-trained models perform well in clean scenes, but classroom-specific training greatly improves separation quality. Finetuning with only half the classroom data achieved comparable gains, confirming efficient transfer learning. Training with diffuse babble noise further enhanced robustness, and the model preserved spatial awareness while generalizing to unseen distances. These findings demonstrate that spatially aware architectures combined with targeted adaptation can improve speech accessibility for children in noisy classrooms, supporting future on-device assistive technologies.
>
---
#### [new 011] Unifying Model and Layer Fusion for Speech Foundation Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出一种统一的接口模块，融合多模型多层表征，解决语音基础模型中模型间与层间融合割裂问题，在ASR和副语言分析等任务上超越现有方法，提升性能与可扩展性。**

- **链接: []()**

> **作者:** Yi-Jen Shih; David Harwath
>
> **备注:** Accepted by IEEE ASRU 2025
>
> **摘要:** Speech Foundation Models have gained significant attention recently. Prior works have shown that the fusion of representations from multiple layers of the same model or the fusion of multiple models can improve performance on downstream tasks. We unify these two fusion strategies by proposing an interface module that enables fusion across multiple upstream speech models while integrating information across their layers. We conduct extensive experiments on different self-supervised and supervised models across various speech tasks, including ASR and paralinguistic analysis, and demonstrate that our method outperforms prior fusion approaches. We further analyze its scalability concerning model size and count, highlighting the importance of selecting appropriate upstream models. Our results show that the proposed interface provides an additional performance boost when given a suitable upstream model selection, making it a promising approach for utilizing Speech Foundation Models.
>
---
#### [new 012] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Melodia，一种无训练的音乐编辑方法，通过干预扩散模型中的自注意力图，保留源音乐的时序结构（如旋律、节奏），同时精准修改乐器、风格等属性，无需文本描述，显著提升编辑效果。**

- **链接: []()**

> **作者:** Yi Yang; Haowen Li; Tianxiang Li; Boyu Cao; Xiaohan Zhang; Liqun Chen; Qi Liu
>
> **备注:** AAAI 2026
>
> **摘要:** Text-to-music generation technology is progressing rapidly, creating new opportunities for musical composition and editing. However, existing music editing methods often fail to preserve the source music's temporal structure, including melody and rhythm, when altering particular attributes like instrument, genre, and mood. To address this challenge, this paper conducts an in-depth probing analysis on attention maps within AudioLDM 2, a diffusion-based model commonly used as the backbone for existing music editing methods. We reveal a key finding: cross-attention maps encompass details regarding distinct musical characteristics, and interventions on these maps frequently result in ineffective modifications. In contrast, self-attention maps are essential for preserving the temporal structure of the source music during its conversion into the target music. Building upon this understanding, we present Melodia, a training-free technique that selectively manipulates self-attention maps in particular layers during the denoising process and leverages an attention repository to store source music information, achieving accurate modification of musical characteristics while preserving the original structure without requiring textual descriptions of the source music. Additionally, we propose two novel metrics to better evaluate music editing methods. Both objective and subjective experiments demonstrate that our approach achieves superior results in terms of textual adherence and structural integrity across various datasets. This research enhances comprehension of internal mechanisms within music generation models and provides improved control for music creation.
>
---
#### [new 013] SpikCommander: A High-performance Spiking Transformer with Multi-view Learning for Efficient Speech Command Recognition
- **分类: cs.SD; cs.LG**

- **简介: 论文提出SpikCommander，一种高效脉冲Transformer，用于语音命令识别。通过多视角脉冲时序注意力（MSTASA）和脉冲上下文细化模块，提升时序建模能力，解决传统脉冲网络信息表达不足问题，在多个数据集上以更少参数超越SOTA。**

- **链接: []()**

> **作者:** Jiaqi Wang; Liutao Yu; Xiongri Shen; Sihang Guo; Chenlin Zhou; Leilei Zhao; Yi Zhong; Zhengyu Ma; Zhiguo Zhang
>
> **备注:** Accepted by The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Spiking neural networks (SNNs) offer a promising path toward energy-efficient speech command recognition (SCR) by leveraging their event-driven processing paradigm. However, existing SNN-based SCR methods often struggle to capture rich temporal dependencies and contextual information from speech due to limited temporal modeling and binary spike-based representations. To address these challenges, we first introduce the multi-view spiking temporal-aware self-attention (MSTASA) module, which combines effective spiking temporal-aware attention with a multi-view learning framework to model complementary temporal dependencies in speech commands. Building on MSTASA, we further propose SpikCommander, a fully spike-driven transformer architecture that integrates MSTASA with a spiking contextual refinement channel MLP (SCR-MLP) to jointly enhance temporal context modeling and channel-wise feature integration. We evaluate our method on three benchmark datasets: the Spiking Heidelberg Dataset (SHD), the Spiking Speech Commands (SSC), and the Google Speech Commands V2 (GSC). Extensive experiments demonstrate that SpikCommander consistently outperforms state-of-the-art (SOTA) SNN approaches with fewer parameters under comparable time steps, highlighting its effectiveness and efficiency for robust speech command recognition.
>
---
#### [new 014] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出HQ-SVC，用于低资源场景下的零样本歌唱语音转换，解决传统方法信息丢失与计算昂贵问题。通过联合编码与可微信号处理，提升音质与效率，并支持语音超分辨率。**

- **链接: []()**

> **作者:** Bingsong Bai; Yizhong Geng; Fengping Wang; Cong Wang; Puyuan Guo; Yingming Gao; Ya Li
>
> **备注:** Accepted by AAAI 2026 main technical track
>
> **摘要:** Zero-shot singing voice conversion (SVC) transforms a source singer's timbre to an unseen target speaker's voice while preserving melodic content without fine-tuning. Existing methods model speaker timbre and vocal content separately, losing essential acoustic information that degrades output quality while requiring significant computational resources. To overcome these limitations, we propose HQ-SVC, an efficient framework for high-quality zero-shot SVC. HQ-SVC first extracts jointly content and speaker features using a decoupled codec. It then enhances fidelity through pitch and volume modeling, preserving critical acoustic information typically lost in separate modeling approaches, and progressively refines outputs via differentiable signal processing and diffusion techniques. Evaluations confirm HQ-SVC significantly outperforms state-of-the-art zero-shot SVC methods in conversion quality and efficiency. Beyond voice conversion, HQ-SVC achieves superior voice naturalness compared to specialized audio super-resolution methods while natively supporting voice super-resolution tasks.
>
---
## 更新

#### [replaced 001] TTSOps: A Closed-Loop Corpus Optimization Framework for Training Multi-Speaker TTS Models from Dark Data
- **分类: cs.SD**

- **链接: []()**

> **作者:** Kentaro Seki; Shinnosuke Takamichi; Takaaki Saeki; Hiroshi Saruwatari
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** This paper presents TTSOps, a fully automated closed-loop framework for constructing multi-speaker text-to-speech (TTS) systems from noisy, uncurated web-scale speech data, often referred to as ``dark data,'' such as online videos. Conventional TTS training pipelines require well-curated corpora with high acoustic quality and accurate text-speech alignment, which severely limits scalability, speaker diversity, and real-world applicability. While recent studies have proposed acoustic-quality-based data selection techniques, they often overlook two critical aspects: (1) the inherent robustness of modern TTS models to noise, and (2) the potential contribution of perceptually low-quality yet informative samples. To address these issues, TTSOps introduces a data-centric training pipeline that integrates three core components: (1) automated data collection from dark data sources, (2) utterance-level dynamic selection of data cleansing methods based on training data quality, and (3) evaluation-in-the-loop data selection using automatically predicted mean opinion scores (MOS) to estimate each utterance's impact on model performance. Furthermore, TTSOps jointly optimizes the corpus and the TTS model in a closed-loop framework by dynamically adapting both data selection and data cleansing processes to the characteristics of the target TTS model. Extensive experiments on Japanese YouTube data demonstrate that TTSOps outperforms conventional acoustic-quality-based baselines in both the naturalness and speaker diversity of synthesized speech.
>
---
#### [replaced 002] AcousTools: A 'Full-Stack', Python-Based, Acoustic Holography Library
- **分类: cs.SD; cs.ET**

- **链接: []()**

> **作者:** Joshua Mukherjee; Giorgos Christopoulos; Zhouyang Shen; Sriram Subramanian; Ryuji Hirayama
>
> **备注:** 14 Pages, 7 Figures, 2 Tables, To be submitted to APL Computational Physics
>
> **摘要:** Acoustic Holography is an emerging field where mid-air ultrasound is controlled and manipulated for novel and exciting applications. These range from mid-air haptics, volumetric displays, contactless fabrication, and even chemical and biomedical applications such as drug delivery. To develop these applications, a software framework to predict acoustic behaviour and simulating resulting effects, such as applied forces or scattering patterns is desirable. There have been various software libraries and platforms that attempt to fill this role, but there is yet to be a single piece of software that acts as a 'full-stack' solution. We define this full-stack as the process from abstraction to physicalisation starting with setup, modelling acoustic propagation, transducer phase retrieval, sound field analysis, and control of the acoustic holographic hardware itself. Existing methods fail to fulfil one or more of these categories. To address this, we present AcousTools, a Python-based acoustic holography library, designed to support the full suite of acoustic holographic applications and we show AcousTools's ability to meet each step of the full-stack's requirements. AcousTools has the potential to become the standard code library for acoustic holography, with the uniquely complete suite of features wrapped in a language that is known to be easy to use, AcousTools will increase the ability for researchers to develop novel applications as well as accurately review other's work. The full-stack, aside from software, will also be useful for researchers - providing a way to view and compare methodologies by understanding where they fit into the stack.
>
---
#### [replaced 003] Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model
- **分类: cs.MM; cs.CL; cs.IR; cs.SD; eess.AS**

- **链接: []()**

> **作者:** Ali Vosoughi; Dimitra Emmanouilidou; Hannes Gamper
>
> **备注:** Accepted at EUSIPCO 2025 - 5 pages, 5 figures, 2 tables
>
> **摘要:** Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
>
---
#### [replaced 004] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
- **分类: eess.AS; cs.SD**

- **链接: []()**

> **作者:** Rui-Chen Zheng; Wenrui Liu; Hui-Peng Du; Qinglin Zhang; Chong Deng; Qian Chen; Wen Wang; Yang Ai; Zhen-Hua Ling
>
> **备注:** Accepted to AAAI 2026. Project page: https://zhengrachel.github.io/VARSTok
>
> **摘要:** Existing speech tokenizers typically assign a fixed number of tokens per second, regardless of the varying information density or temporal fluctuations in the speech signal. This uniform token allocation mismatches the intrinsic structure of speech, where information is distributed unevenly over time. To address this, we propose VARSTok, a VAriable-frame-Rate Speech Tokenizer that adapts token allocation based on local feature similarity. VARSTok introduces two key innovations: (1) a temporal-aware density peak clustering algorithm that adaptively segments speech into variable-length units, and (2) a novel implicit duration coding scheme that embeds both content and temporal span into a single token index, eliminating the need for auxiliary duration predictors. Extensive experiments show that VARSTok significantly outperforms strong fixed-rate baselines. Notably, it achieves superior reconstruction naturalness while using up to 23% fewer tokens than a 40 Hz fixed-frame-rate baseline. VARSTok further yields lower word error rates and improved naturalness in zero-shot text-to-speech synthesis. To the best of our knowledge, this is the first work to demonstrate that a fully dynamic, variable-frame-rate acoustic speech tokenizer can be seamlessly integrated into downstream speech language models.
>
---
#### [replaced 005] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: []()**

> **作者:** Jinting Wang; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading via hand coding, offering visual phonemic cues that support precise speech perception for the hearing-impaired. The task of CS Video-to-Speech generation (CSV2S) aims to convert CS videos into intelligible speech signals. Most existing research focuses on CS Recognition (CSR), which transcribes video content into text. Consequently, a common solution for CSV2S is to integrate CSR with a text-to-speech (TTS) system. However, this pipeline relies on text as an intermediate medium, which may lead to error propagation and temporal misalignment between speech and CS video dynamics. In contrast, directly generating audio speech from CS video (direct CSV2S) often suffers from the inherent multimodal complexity and the limited availability of CS data. To address these challenges, we propose UniCUE, the first unified framework for CSV2S that directly generates speech from CS videos without relying on intermediate text. The core innovation of UniCUE lies in integrating an understanding task (CSR) that provides fine-grained CS visual-semantic cues to guide speech generation. Specifically, UniCUE incorporates a pose-aware visual processor, a semantic alignment pool that enables precise visual-semantic mapping, and a VisioPhonetic adapter to bridge the understanding and generation tasks within a unified architecture. To support this framework, we construct UniCUE-HI, a large-scale Mandarin CS dataset containing 11282 videos from 14 cuers, including both hearing-impaired and normal-hearing individuals. Extensive experiments on this dataset demonstrate that UniCUE achieves state-of-the-art performance across multiple evaluation metrics.
>
---
