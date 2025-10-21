# 音频 cs.SD;  eess.AS

- **最新发布 26 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] DDSC: Dynamic Dual-Signal Curriculum for Data-Efficient Acoustic Scene Classification under Domain Shift
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对设备差异导致的域偏移问题，研究低标签条件下的声学场景分类。提出动态双信号课程（DDSC）方法，通过在线调整样本权重，结合域不变性和学习进度信号，提升跨设备分类性能。**

- **链接: [http://arxiv.org/pdf/2510.17345v1](http://arxiv.org/pdf/2510.17345v1)**

> **作者:** Peihong Zhang; Yuxuan Liu; Rui Sang; Zhixin Li; Yiqiang Cai; Yizhou Tan; Shengchen Li
>
> **备注:** Paper has submitted to ICASSP2026
>
> **摘要:** Acoustic scene classification (ASC) suffers from device-induced domain shift, especially when labels are limited. Prior work focuses on curriculum-based training schedules that structure data presentation by ordering or reweighting training examples from easy-to-hard to facilitate learning; however, existing curricula are static, fixing the ordering or the weights before training and ignoring that example difficulty and marginal utility evolve with the learned representation. To overcome this limitation, we propose the Dynamic Dual-Signal Curriculum (DDSC), a training schedule that adapts the curriculum online by combining two signals computed each epoch: a domain-invariance signal and a learning-progress signal. A time-varying scheduler fuses these signals into per-example weights that prioritize domain-invariant examples in early epochs and progressively emphasize device-specific cases. DDSC is lightweight, architecture-agnostic, and introduces no additional inference overhead. Under the official DCASE 2024 Task~1 protocol, DDSC consistently improves cross-device performance across diverse ASC baselines and label budgets, with the largest gains on unseen-device splits.
>
---
#### [new 002] Towards Real-Time Generative Speech Restoration with Flow-Matching
- **分类: eess.AS**

- **简介: 该论文研究基于流匹配的实时生成式语音修复，解决传统方法延迟高、不适用于流式场景的问题。提出因果架构，实现仅20 ms延迟，并优化采样效率，5步即可获得高质量增强语音。**

- **链接: [http://arxiv.org/pdf/2510.16997v1](http://arxiv.org/pdf/2510.16997v1)**

> **作者:** Tsun-An Hsieh; Sebastian Braun
>
> **摘要:** Generative models have shown robust performance on speech enhancement and restoration tasks, but most prior approaches operate offline with high latency, making them unsuitable for streaming applications. In this work, we investigate the feasibility of a low-latency, real-time generative speech restoration system based on flow-matching (FM). Our method tackles diverse real-world tasks, including denoising, dereverberation, and generative restoration. The proposed causal architecture without time-downsampling achieves introduces an total latency of only 20 ms, suitable for real-time communication. In addition, we explore a broad set of architectural variations and sampling strategies to ensure effective training and efficient inference. Notably, our flow-matching model maintains high enhancement quality with only 5 number of function evaluations (NFEs) during sampling, achieving similar performance as when using ~20 NFEs under the same conditions. Experimental results indicate that causal FM-based models favor few-step reverse sampling, and smaller backbones degrade with longer reverse trajectories. We further show a side-by-side comparison of FM to typical adversarial-loss-based training for the same model architecture.
>
---
#### [new 003] Audio dequantization using instantaneous frequency
- **分类: eess.AS**

- **简介: 该论文研究音频去量化任务，旨在减少量化带来的失真。提出基于瞬时频率的相位感知正则化方法（PHADQ），保持时频表示中正弦成分的时间连续性，避免能量损失伪影，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2510.16813v1](http://arxiv.org/pdf/2510.16813v1)**

> **作者:** Vojtěch Kovanda; Pavel Rajmic
>
> **摘要:** We present a dequantization method that employs a phase-aware regularizer, originally successfully applied in an audio inpainting problem. The method maintains the temporal continuity of sinusoidal components in the audio signal time-frequency representation and avoids the energy loss artifacts commonly encountered with l1-based regularization approaches. The proposed method is called the Phase-Aware Audio Dequantizer (PHADQ). The methods are evaluated using the objective metric SDR and PEMO-Q ODG.
>
---
#### [new 004] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification
- **分类: cs.SD**

- **简介: 该论文针对歌唱声音深度伪造的鉴别任务，旨在保护歌手声音权益。提出两阶段方法：先用判别模型过滤低质伪造，再用仅基于真实音频训练的模型识别高质伪造与真实歌手，提升深伪场景下的歌手识别鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.17474v1](http://arxiv.org/pdf/2510.17474v1)**

> **作者:** Davide Salvi; Hendrik Vincent Koops; Elio Quinton
>
> **备注:** Accepted for presentation at the NeurIPS 2025 Workshop on Generative and Protective AI for Content Creation (non-archival)
>
> **摘要:** The proliferation of highly realistic singing voice deepfakes presents a significant challenge to protecting artist likeness and content authenticity. Automatic singer identification in vocal deepfakes is a promising avenue for artists and rights holders to defend against unauthorized use of their voice, but remains an open research problem. Based on the premise that the most harmful deepfakes are those of the highest quality, we introduce a two-stage pipeline to identify a singer's vocal likeness. It first employs a discriminator model to filter out low-quality forgeries that fail to accurately reproduce vocal likeness. A subsequent model, trained exclusively on authentic recordings, identifies the singer in the remaining high-quality deepfakes and authentic audio. Experiments show that this system consistently outperforms existing baselines on both authentic and synthetic content.
>
---
#### [new 005] Interpreting the Dimensions of Speaker Embedding Space
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究说话人嵌入空间的可解释性，旨在揭示嵌入如何反映声学、年龄和性别特征。通过分析10,000名说话人数据，发现9个可解释声学参数能有效预测嵌入，性别影响嵌入结构，但年龄未被良好捕捉。**

- **链接: [http://arxiv.org/pdf/2510.16489v1](http://arxiv.org/pdf/2510.16489v1)**

> **作者:** Mark Huckvale
>
> **摘要:** Speaker embeddings are widely used in speaker verification systems and other applications where it is useful to characterise the voice of a speaker with a fixed-length vector. These embeddings tend to be treated as "black box" encodings, and how they relate to conventional acoustic and phonetic dimensions of voices has not been widely studied. In this paper we investigate how state-of-the-art speaker embedding systems represent the acoustic characteristics of speakers as described by conventional acoustic descriptors, age, and gender. Using a large corpus of 10,000 speakers and three embedding systems we show that a small set of 9 acoustic parameters chosen to be "interpretable" predict embeddings about the same as 7 principal components, corresponding to over 50% of variance in the data. We show that some principal dimensions operate differently for male and female speakers, suggesting there is implicit gender recognition within the embedding systems. However we show that speaker age is not well captured by embeddings, suggesting opportunities exist for improvements in their calculation.
>
---
#### [new 006] AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning
- **分类: eess.AS; cs.AI; cs.MM**

- **简介: 该论文聚焦人机协作中的实时交互问题，提出AsyncVoice Agent系统，通过异步架构解耦语音前端与LLM后端，实现推理与叙述并行，支持用户随时打断和干预，显著降低延迟，提升可解释性与可控性。**

- **链接: [http://arxiv.org/pdf/2510.16156v1](http://arxiv.org/pdf/2510.16156v1)**

> **作者:** Yueqian Lin; Zhengmian Hu; Jayakumar Subramanian; Qinsi Wang; Nikos Vlassis; Hai "Helen" Li; Yiran Chen
>
> **备注:** Accepted to the IEEE ASRU 2025 Demo Track
>
> **摘要:** Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks.
>
---
#### [new 007] SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization
- **分类: eess.AS**

- **简介: 该论文研究神经语音编解码任务，旨在解决现有编解码器在重建质量与语义表征间难以平衡的问题。提出SAC模型，采用语义-声学双流量化架构，分别优化两方面性能，实现高质量语音重建与强语义表示能力。**

- **链接: [http://arxiv.org/pdf/2510.16841v1](http://arxiv.org/pdf/2510.16841v1)**

> **作者:** Wenxi Chen; Xinsheng Wang; Ruiqi Yan; Yushen Chen; Zhikang Niu; Ziyang Ma; Xiquan Li; Yuzhe Liang; Hanlin Wen; Shunshun Yin; Ming Tao; Xie Chen
>
> **摘要:** Speech codecs that convert continuous speech signals into discrete tokens have become essential for speech language models (SLMs). However, existing codecs struggle to balance high-quality reconstruction with semantically rich representations, limiting their effectiveness in both generative and understanding tasks. In this work, we propose SAC, a neural speech codec with semantic-acoustic dual-stream quantization. By disentangling semantic and acoustic modeling into two dedicated streams, SAC enables each to be optimized for its respective role. Comprehensive evaluations show that SAC achieves strong reconstruction performance across diverse bitrates under both clean and noisy conditions, with particularly high scores on UTMOS and WER, demonstrating superior perceptual quality and intelligibility. Moreover, SAC substantially outperforms state-of-the-art codecs in semantic representation, achieving a level comparable to that of self-supervised learning (SSL) continuous embeddings. Finally, our analysis of speech disentanglement highlights the effectiveness of the dual-stream design, offering new potential for controllable speech applications.
>
---
#### [new 008] Transmission of High-Amplitude Sound through Leakages of Ill-fitting Earplugs
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究耳塞泄漏对高声压级声音传输的影响，旨在评估漏声机制及隔音性能。通过实验与仿真分析不同泄漏结构的传声特性，揭示高声压下涡旋导致能量耗散的现象，强调耳塞设计在高噪声环境中的重要性。**

- **链接: [http://arxiv.org/pdf/2510.16355v1](http://arxiv.org/pdf/2510.16355v1)**

> **作者:** Haocheng Yu; Krishan K. Ahuja; Lakshmi N. Sankar; Spencer H. Bryngelson
>
> **摘要:** High sound pressure levels (SPL) pose notable risks in loud environments, particularly due to noise-induced hearing loss. Ill-fitting earplugs often lead to sound leakage, a phenomenon this study seeks to investigate. To validate our methodology, we first obtained computational and experimental acoustic transmission data for stand-alone slit resonators and orifices, for which extensive published data are readily available for comparison. We then examined the frequency-dependent acoustic power absorption coefficient and transmission loss (TL) across various leakage geometries, modeled using different orifice diameters. Experimental approaches spanned a frequency range of 1--5 kHz under SPL conditions of 120--150 dB. Key findings reveal that unsealed silicone rubber earplugs demonstrate an average TL reduction of approximately 18 dB at an overall incident SPL (OISPL) of 120 dB. Direct numerical simulations further highlight SPL-dependent acoustic dissipation mechanisms, showing the conversion of acoustic energy into vorticity in ill-fitting earplug models at an OISPL of 150 dB. These results highlight the role of earplug design for high-sound-pressure-level environments.
>
---
#### [new 009] MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出MuseTok，一种用于符号音乐的离散表示学习方法，旨在统一音乐生成与语义理解任务。基于RQ-VAE与Transformer架构，实现高保真重建并捕捉音乐理论特征，在生成与理解任务中均表现优异。**

- **链接: [http://arxiv.org/pdf/2510.16273v1](http://arxiv.org/pdf/2510.16273v1)**

> **作者:** Jingyue Huang; Zachary Novack; Phillip Long; Yupeng Hou; Ke Chen; Taylor Berg-Kirkpatrick; Julian McAuley
>
> **摘要:** Discrete representation learning has shown promising results across various domains, including generation and understanding in image, speech and language. Inspired by these advances, we propose MuseTok, a tokenization method for symbolic music, and investigate its effectiveness in both music generation and understanding tasks. MuseTok employs the residual vector quantized-variational autoencoder (RQ-VAE) on bar-wise music segments within a Transformer-based encoder-decoder framework, producing music codes that achieve high-fidelity music reconstruction and accurate understanding of music theory. For comprehensive evaluation, we apply MuseTok to music generation and semantic understanding tasks, including melody extraction, chord recognition, and emotion recognition. Models incorporating MuseTok outperform previous representation learning baselines in semantic understanding while maintaining comparable performance in content generation. Furthermore, qualitative analyses on MuseTok codes, using ground-truth categories and synthetic datasets, reveal that MuseTok effectively captures underlying musical concepts from large music collections.
>
---
#### [new 010] DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Supervised Speech Foundational Model
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出DELULU，一种说话人感知的自监督语音基础模型。针对现有模型难以捕捉说话人差异的问题，引入外部说话人监督信号指导聚类伪标签生成，结合掩码与去噪双目标训练，在说话人验证和零样本属性分析任务上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.17662v1](http://arxiv.org/pdf/2510.17662v1)**

> **作者:** Massa Baali; Rita Singh; Bhiksha Raj
>
> **摘要:** Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce DELULU, a speaker-aware self-supervised foundational model that addresses this limitation by integrating external supervision into the pseudo-label generation process. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide the k-means clustering step during pre-training, introducing a strong speaker-discriminative inductive bias that aligns representation learning with speaker identity. The model is trained using a dual objective that combines masked prediction and denoising, further enhancing robustness and generalization. DELULU significantly outperforms prior self-supervised learning (SSL) models across a range of speaker-centric tasks, achieving up to 62% relative improvement in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks such as gender, age, accent, and speaker counting. Our findings demonstrate that DELULU is a strong universal encoder for speaker-aware speech processing, enabling superior performance even without task-specific fine-tuning.
>
---
#### [new 011] Schrödinger Bridge Mamba for One-Step Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出Schrödinger Bridge Mamba（SBM），用于单步语音增强任务，结合Schrödinger Bridge训练与Mamba架构，实现高效生成式去噪去混响。在四个数据集上验证，SBM单步推理即超越多步基线方法，兼具高性能与实时性。**

- **链接: [http://arxiv.org/pdf/2510.16834v1](http://arxiv.org/pdf/2510.16834v1)**

> **作者:** Jing Yang; Sirui Wang; Chao Wu; Fan Fan
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** We propose Schr\"odinger Bridge Mamba (SBM), a new concept of training-inference framework motivated by the inherent compatibility between Schr\"odinger Bridge (SB) training paradigm and selective state-space model Mamba. We exemplify the concept of SBM with an implementation for generative speech enhancement. Experiments on a joint denoising and dereverberation task using four benchmark datasets demonstrate that SBM, with only 1-step inference, outperforms strong baselines with 1-step or iterative inference and achieves the best real-time factor (RTF). Beyond speech enhancement, we discuss the integration of SB paradigm and selective state-space model architecture based on their underlying alignment, which indicates a promising direction for exploring new deep generative models potentially applicable to a broad range of generative tasks. Demo page: https://sbmse.github.io
>
---
#### [new 012] Zero- and One-Shot Data Augmentation for Sentence-Level Dysarthric Speech Recognition in Constrained Scenarios
- **分类: cs.SD**

- **简介: 该论文研究句子级构音障碍语音识别，针对数据稀缺问题，提出一种面向文本匹配的零/一样本数据增强方法，通过生成覆盖目标文本的合成语音，提升对未见说话者的识别性能，适用于康复和日常交流场景。**

- **链接: [http://arxiv.org/pdf/2510.16700v1](http://arxiv.org/pdf/2510.16700v1)**

> **作者:** Shiyao Wang; Shiwan Zhao; Jiaming Zhou; Yong Qin
>
> **备注:** NCMMSC 2025 oral
>
> **摘要:** Dysarthric speech recognition (DSR) research has witnessed remarkable progress in recent years, evolving from the basic understanding of individual words to the intricate comprehension of sentence-level expressions, all driven by the pressing communication needs of individuals with dysarthria. Nevertheless, the scarcity of available data remains a substantial hurdle, posing a significant challenge to the development of effective sentence-level DSR systems. In response to this issue, dysarthric data augmentation (DDA) has emerged as a highly promising approach. Generative models are frequently employed to generate training data for automatic speech recognition tasks. However, their effectiveness hinges on the ability of the synthesized data to accurately represent the target domain. The wide-ranging variability in pronunciation among dysarthric speakers makes it extremely difficult for models trained on data from existing speakers to produce useful augmented data, especially in zero-shot or one-shot learning settings. To address this limitation, we put forward a novel text-coverage strategy specifically designed for text-matching data synthesis. This innovative strategy allows for efficient zero/one-shot DDA, leading to substantial enhancements in the performance of DSR when dealing with unseen dysarthric speakers. Such improvements are of great significance in practical applications, including dysarthria rehabilitation programs and day-to-day common-sentence communication scenarios.
>
---
#### [new 013] SARSteer: Safeguarding Large Audio Language Models via Safe-Ablated Refusal Steering
- **分类: cs.SD; cs.CR**

- **简介: 该论文针对大音频语言模型（LALM）易因音频输入引发有害回应的问题，提出SARSteer框架。通过文本驱动的拒绝引导与安全空间分解消减，实现在不修改音频输入下的安全推理，有效提升拒害能力并减少对正常请求的误拒。**

- **链接: [http://arxiv.org/pdf/2510.17633v1](http://arxiv.org/pdf/2510.17633v1)**

> **作者:** Weilin Lin; Jianze Li; Hui Xiong; Li Liu
>
> **摘要:** Large Audio-Language Models (LALMs) are becoming essential as a powerful multimodal backbone for real-world applications. However, recent studies show that audio inputs can more easily elicit harmful responses than text, exposing new risks toward deployment. While safety alignment has made initial advances in LLMs and Large Vision-Language Models (LVLMs), we find that vanilla adaptation of these approaches to LALMs faces two key limitations: 1) LLM-based steering fails under audio input due to the large distributional gap between activations, and 2) prompt-based defenses induce over-refusals on benign-speech queries. To address these challenges, we propose Safe-Ablated Refusal Steering (SARSteer), the first inference-time defense framework for LALMs. Specifically, SARSteer leverages text-derived refusal steering to enforce rejection without manipulating audio inputs and introduces decomposed safe-space ablation to mitigate over-refusal. Extensive experiments demonstrate that SARSteer significantly improves harmful-query refusal while preserving benign responses, establishing a principled step toward safety alignment in LALMs.
>
---
#### [new 014] AWARE: Audio Watermarking with Adversarial Resistance to Edits
- **分类: cs.SD; cs.LG; cs.MM**

- **简介: 该论文研究音频水印任务，旨在提升水印对各类编辑操作的鲁棒性。提出AWARE框架，通过时频域对抗优化嵌入水印，设计时间顺序无关的检测器与比特读出头，有效应对失真与剪辑，无需依赖模拟攻击，实现了高音质、低误码率的水印检测。**

- **链接: [http://arxiv.org/pdf/2510.17512v1](http://arxiv.org/pdf/2510.17512v1)**

> **作者:** Kosta Pavlović; Lazar Stanarević; Petar Nedić; Slavko Kovačević; Igor Djurović
>
> **摘要:** Prevailing practice in learning-based audio watermarking is to pursue robustness by expanding the set of simulated distortions during training. However, such surrogates are narrow and prone to overfitting. This paper presents AWARE (Audio Watermarking with Adversarial Resistance to Edits), an alternative approach that avoids reliance on attack-simulation stacks and handcrafted differentiable distortions. Embedding is obtained via adversarial optimization in the time-frequency domain under a level-proportional perceptual budget. Detection employs a time-order-agnostic detector with a Bitwise Readout Head (BRH) that aggregates temporal evidence into one score per watermark bit, enabling reliable watermark decoding even under desynchronization and temporal cuts. Empirically, AWARE attains high audio quality and speech intelligibility (PESQ/STOI) and consistently low BER across various audio edits, often surpassing representative state-of-the-art learning-based audio watermarking systems.
>
---
#### [new 015] Adaptive Deterministic Flow Matching for Target Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文研究目标说话人提取（TSE）任务，旨在提升生成式模型的效率与精度。提出自适应判别流匹配（AD-FlowTSE），通过混合比感知的初始化和可变步长，在流匹配框架下沿背景-语音轨迹自适应反向生成，实现少步甚至单步高保真语音提取。**

- **链接: [http://arxiv.org/pdf/2510.16995v1](http://arxiv.org/pdf/2510.16995v1)**

> **作者:** Tsun-An Hsieh; Minje Kim
>
> **摘要:** Generative target speaker extraction (TSE) methods often produce more natural outputs than predictive models. Recent work based on diffusion or flow matching (FM) typically relies on a small, fixed number of reverse steps with a fixed step size. We introduce Adaptive Discriminative Flow Matching TSE (AD-FlowTSE), which extracts the target speech using an adaptive step size. We formulate TSE within the FM paradigm but, unlike prior FM-based speech enhancement and TSE approaches that transport between the mixture (or a normal prior) and the clean-speech distribution, we define the flow between the background and the source, governed by the mixing ratio (MR) of the source and background that creates the mixture. This design enables MR-aware initialization, where the model starts at an adaptive point along the background-source trajectory rather than applying the same reverse schedule across all noise levels. Experiments show that AD-FlowTSE achieves strong TSE with as few as a single step, and that incorporating auxiliary MR estimation further improves target speech accuracy. Together, these results highlight that aligning the transport path with the mixture composition and adapting the step size to noise conditions yields efficient and accurate TSE.
>
---
#### [new 016] Audio-Visual Speech Enhancement for Spatial Audio - Spatial-VisualVoice and the MAVE Database
- **分类: eess.AS**

- **简介: 该论文研究音频-视觉语音增强（AVSE）任务，旨在解决低信噪比下空间音频增强中空间线索丢失的问题。提出Spatial-VisualVoice框架，结合麦克风阵列与视觉信息，并构建MAVE数据库验证方法，有效提升语音质量与空间感知。**

- **链接: [http://arxiv.org/pdf/2510.16437v1](http://arxiv.org/pdf/2510.16437v1)**

> **作者:** Danielle Yaffe; Ferdinand Campe; Prachi Sharma; Dorothea Kolossa; Boaz Rafaely
>
> **摘要:** Audio-visual speech enhancement (AVSE) has been found to be particularly useful at low signal-to-noise (SNR) ratios due to the immunity of the visual features to acoustic noise. However, a significant gap exists in AVSE methods tailored to enhance spatial audio under low-SNR conditions. The latter is of growing interest with augmented reality applications. To address this gap, we present a multi-channel AVSE framework based on VisualVoice that leverages spatial cues from microphone arrays and visual information for enhancing the target speaker in noisy environments. We also introduce MAVe, a novel database containing multi-channel audio-visual signals in controlled, reproducible room conditions across a wide range of SNR levels. Experiments demonstrate that the proposed method consistently achieves significant gains in SI-SDR, STOI, and PESQ, particularly in low SNRs. Binaural signal analysis further confirms the preservation of spatial cues and intelligibility.
>
---
#### [new 017] U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文研究语音合成中的高效编解码，旨在解决低帧率下语音质量下降的问题。提出U-Codec，通过Transformer建模长时依赖，优化残差向量量化，实现5Hz超低帧率下的高保真、快速语音生成，并提升LLM-TTS推理速度3倍。**

- **链接: [http://arxiv.org/pdf/2510.16718v1](http://arxiv.org/pdf/2510.16718v1)**

> **作者:** Xusheng Yang; Long Zhou; Wenfu Wang; Kai Hu; Shulin Feng; Chenxing Li; Meng Yu; Dong Yu; Yuexian Zou
>
> **摘要:** We propose \textbf{U-Codec}, an \textbf{U}ltra low frame-rate neural speech \textbf{Codec} that achieves high-fidelity reconstruction and fast speech generation at an extremely low frame-rate of 5Hz (5 frames per second). Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we introduce a Transformer-based inter-frame long-term dependency module and systematically explore residual vector quantization (RVQ) depth and codebook size to identify optimal configurations. Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. We extend LLM-based TTS from 3-layer RVQ at 50Hz to 32-layer RVQ at 5Hz. Experimental results demonstrate that U-Codec improves LLM-based TTS inference speed by around 3 $\times$ over high-frame-rate codecs while maintaining similarity and naturalness. These results validate the feasibility of using highly compressed 5Hz discrete tokens for fast and high-fidelity speech synthesis.
>
---
#### [new 018] Investigating Safety Vulnerabilities of Large Audio-Language Models Under Speaker Emotional Variations
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究大型音频语言模型在不同说话人情绪下的安全漏洞。针对恶意指令在多情绪表达下的响应不一致问题，构建情感变化数据集并评测主流模型，发现情绪强度与风险呈非单调关系，中等强度最危险，提出需增强模型对情绪变化的鲁棒对齐。**

- **链接: [http://arxiv.org/pdf/2510.16893v1](http://arxiv.org/pdf/2510.16893v1)**

> **作者:** Bo-Han Feng; Chien-Feng Liu; Yu-Hsuan Li Liang; Chih-Kai Yang; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) extend text-based LLMs with auditory understanding, offering new opportunities for multimodal applications. While their perception, reasoning, and task performance have been widely studied, their safety alignment under paralinguistic variation remains underexplored. This work systematically investigates the role of speaker emotion. We construct a dataset of malicious speech instructions expressed across multiple emotions and intensities, and evaluate several state-of-the-art LALMs. Our results reveal substantial safety inconsistencies: different emotions elicit varying levels of unsafe responses, and the effect of intensity is non-monotonic, with medium expressions often posing the greatest risk. These findings highlight an overlooked vulnerability in LALMs and call for alignment strategies explicitly designed to ensure robustness under emotional variation, a prerequisite for trustworthy deployment in real-world settings.
>
---
#### [new 019] TopSeg: A Multi-Scale Topological Framework for Data-Efficient Heart Sound Segmentation
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究心音信号分割任务，旨在解决标注数据稀缺下的模型泛化问题。作者提出TopSeg框架，利用多尺度拓扑特征和轻量TCN实现高效分割，在低数据条件下显著优于传统方法，验证了拓扑表示在跨数据集场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.17346v1](http://arxiv.org/pdf/2510.17346v1)**

> **作者:** Peihong Zhang; Zhixin Li; Yuxuan Liu; Rui Sang; Yiqiang Cai; Yizhou Tan; Shengchen Li
>
> **备注:** Paper has submitted to ICASSP2026
>
> **摘要:** Deep learning approaches for heart-sound (PCG) segmentation built on time--frequency features can be accurate but often rely on large expert-labeled datasets, limiting robustness and deployment. We present TopSeg, a topological representation-centric framework that encodes PCG dynamics with multi-scale topological features and decodes them using a lightweight temporal convolutional network (TCN) with an order- and duration-constrained inference step. To evaluate data efficiency and generalization, we train exclusively on PhysioNet 2016 dataset with subject-level subsampling and perform external validation on CirCor dataset. Under matched-capacity decoders, the topological features consistently outperform spectrogram and envelope inputs, with the largest margins at low data budgets; as a full system, TopSeg surpasses representative end-to-end baselines trained on their native inputs under the same budgets while remaining competitive at full data. Ablations at 10% training confirm that all scales contribute and that combining H_0 and H_1 yields more reliable S1/S2 localization and boundary stability. These results indicate that topology-aware representations provide a strong inductive bias for data-efficient, cross-dataset PCG segmentation, supporting practical use when labeled data are limited.
>
---
#### [new 020] SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文聚焦音频-语言模型中的知识编辑任务，旨在解决现有方法局限于文本和视觉模态的问题。作者提出首个针对听觉属性知识编辑的基准SAKE，评估七种方法在两个大模型上的表现，探讨编辑的可靠性、泛化性等挑战，推动多模态知识更新研究。**

- **链接: [http://arxiv.org/pdf/2510.16917v1](http://arxiv.org/pdf/2510.16917v1)**

> **作者:** Chih-Kai Yang; Yen-Ting Piao; Tzu-Wen Hsu; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** Knowledge editing offers an efficient way to update model knowledge without full retraining, but prior work has concentrated almost exclusively on textual or visual modalities. We introduce SAKE, the first benchmark specifically designed for editing auditory attribute knowledge in Large Audio-Language Models (LALMs). Unlike factual updates, SAKE targets several abstract auditory attributes, capturing knowledge types that go beyond conventional textual and visual domains. We benchmark seven editing methods on two LALMs along four dimensions: reliability, generality, audio/text locality, and portability. Results highlight challenges such as preserving intra-attribute knowledge unrelated to the edit, generalizing edits to multimodal reasoning, and maintaining edits under sequential updates. SAKE provides a principled framework to study how knowledge editing extends to the auditory modalities, opening new directions for maintaining and adapting LALMs in more diverse real-world scenarios.
>
---
#### [new 021] AnyRIR: Robust Non-intrusive Room Impulse Response Estimation in the Wild
- **分类: eess.AS**

- **简介: 该论文研究非侵入式房间脉冲响应（RIR）估计，旨在解决真实噪声环境下传统方法受非平稳声音干扰的问题。提出AnyRIR方法，利用音乐作为激励信号，通过时频域L1范数回归抑制噪声，实现鲁棒RIR估计。**

- **链接: [http://arxiv.org/pdf/2510.17788v1](http://arxiv.org/pdf/2510.17788v1)**

> **作者:** Kyung Yun Lee; Nils Meyer-Kahlen; Karolina Prawda; Vesa Välimäki; Sebastian J. Schlecht
>
> **摘要:** We address the problem of estimating room impulse responses (RIRs) in noisy, uncontrolled environments where non-stationary sounds such as speech or footsteps corrupt conventional deconvolution. We propose AnyRIR, a non-intrusive method that uses music as the excitation signal instead of a dedicated test signal, and formulate RIR estimation as an L1-norm regression in the time-frequency domain. Solved efficiently with Iterative Reweighted Least Squares (IRLS) and Least-Squares Minimal Residual (LSMR) methods, this approach exploits the sparsity of non-stationary noise to suppress its influence. Experiments on simulated and measured data show that AnyRIR outperforms L2-based and frequency-domain deconvolution, under in-the-wild noisy scenarios and codec mismatch, enabling robust RIR estimation for AR/VR and related applications.
>
---
#### [new 022] BREATH: A Bio-Radar Embodied Agent for Tonal and Human-Aware Diffusion Music Generation
- **分类: cs.HC; cs.AI; cs.SD**

- **简介: 该论文提出BREATH系统，属音乐生成任务，旨在通过生理感知实现个性化、文化契合的音乐生成。工作融合雷达生理监测、大模型推理与扩散音频合成，构建从生理信号到五声音阶音乐的生成闭环，支持疗愈与交互应用。**

- **链接: [http://arxiv.org/pdf/2510.15895v1](http://arxiv.org/pdf/2510.15895v1)**

> **作者:** Yunzhe Wang; Xinyu Tang; Zhixun Huang; Xiaolong Yue; Yuxin Zeng
>
> **备注:** Accepted by LLM4Music @ ISMIR 2025
>
> **摘要:** We present a multimodal system for personalized music generation that integrates physiological sensing, LLM-based reasoning, and controllable audio synthesis. A millimeter-wave radar sensor non-invasively captures heart rate and respiration rate. These physiological signals, combined with environmental state, are interpreted by a reasoning agent to infer symbolic musical descriptors, such as tempo, mood intensity, and traditional Chinese pentatonic modes, which are then expressed as structured prompts to guide a diffusion-based audio model in synthesizing expressive melodies. The system emphasizes cultural grounding through tonal embeddings and enables adaptive, embodied music interaction. To evaluate the system, we adopt a research-creation methodology combining case studies, expert feedback, and targeted control experiments. Results show that physiological variations can modulate musical features in meaningful ways, and tonal conditioning enhances alignment with intended modal characteristics. Expert users reported that the system affords intuitive, culturally resonant musical responses and highlighted its potential for therapeutic and interactive applications. This work demonstrates a novel bio-musical feedback loop linking radar-based sensing, prompt reasoning, and generative audio modeling.
>
---
#### [new 023] Event Topology-based Visual Microphone for Amplitude and Frequency Reconstruction
- **分类: physics.app-ph; cs.SD**

- **简介: 该论文研究非接触式振动测量，旨在从事件相机数据中精确恢复振动的幅度和频率。提出基于事件拓扑结构的视觉麦克风方法，结合拓扑数据分析与聚类算法，实现无需外部光照的多声源同步重建，提升了精度与实用性。**

- **链接: [http://arxiv.org/pdf/2510.17092v1](http://arxiv.org/pdf/2510.17092v1)**

> **作者:** Ryogo Niwa; Yoichi Ochiai; Tatsuki Fushimi
>
> **备注:** 6 pages, 5 figures, 2 tables. Submitted for publication
>
> **摘要:** Accurate vibration measurement is vital for analyzing dynamic systems across science and engineering, yet noncontact methods often balance precision against practicality. Event cameras offer high-speed, low-light sensing, but existing approaches fail to recover vibration amplitude and frequency with sufficient accuracy. We present an event topology-based visual microphone that reconstructs vibrations directly from raw event streams without external illumination. By integrating the Mapper algorithm from topological data analysis with hierarchical density-based clustering, our framework captures the intrinsic structure of event data to recover both amplitude and frequency with high fidelity. Experiments demonstrate substantial improvements over prior methods and enable simultaneous recovery of multiple sound sources from a single event stream, advancing the frontier of passive, illumination-free vibration sensing.
>
---
#### [new 024] Hallucination Benchmark for Speech Foundation Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对语音识别中的幻觉问题，提出首个基准框架SHALLOW，从词汇、语音、形态和语义四个维度系统评估幻觉现象，弥补传统错误率指标的不足，实现细粒度诊断与模型改进。**

- **链接: [http://arxiv.org/pdf/2510.16567v1](http://arxiv.org/pdf/2510.16567v1)**

> **作者:** Alkis Koudounas; Moreno La Quatra; Manuel Giollo; Sabato Marco Siniscalchi; Elena Baralis
>
> **备注:** Under Review
>
> **摘要:** Hallucinations in automatic speech recognition (ASR) systems refer to fluent and coherent transcriptions produced by neural ASR models that are completely unrelated to the underlying acoustic input (i.e., the speech signal). While similar to conventional decoding errors in potentially compromising the usability of transcriptions for downstream applications, hallucinations can be more detrimental due to their preservation of syntactically and semantically plausible structure. This apparent coherence can mislead subsequent processing stages and introduce serious risks, particularly in critical domains such as healthcare and law. Conventional evaluation metrics are primarily centered on error-based metrics and fail to distinguish between phonetic inaccuracies and hallucinations. Consequently, there is a critical need for new evaluation frameworks that can effectively identify and assess models with a heightened propensity for generating hallucinated content. To this end, we introduce SHALLOW, the first benchmark framework that systematically categorizes and quantifies hallucination phenomena in ASR along four complementary axes: lexical, phonetic, morphological, and semantic. We define targeted metrics within each category to produce interpretable profiles of model behavior. Through evaluation across various architectures and speech domains, we have found that SHALLOW metrics correlate strongly with word error rate (WER) when recognition quality is high (i.e., low WER). Still, this correlation weakens substantially as WER increases. SHALLOW, therefore, captures fine-grained error patterns that WER fails to distinguish under degraded and challenging conditions. Our framework supports specific diagnosis of model weaknesses and provides feedback for model improvement beyond what aggregate error rates can offer.
>
---
#### [new 025] End-to-end Listen, Look, Speak and Act
- **分类: cs.AI; cs.CL; cs.CV; cs.RO; eess.AS**

- **简介: 该论文提出ELLAS模型，旨在实现人类般的全双工多模态交互。它通过SA-MoE架构统一处理听、看、说、动，支持并发感知与生成，解决了多模态干扰与协作难题，实现了自然的人机交互行为。**

- **链接: [http://arxiv.org/pdf/2510.16756v1](http://arxiv.org/pdf/2510.16756v1)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Chao Zhang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released upon acceptance.
>
---
#### [new 026] Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文探索Whisper模型在二语口语评估中的潜力，旨在挖掘其隐藏表征对语言熟练度评价的有效性。通过提取声学与语言特征并结合轻量分类器，结合图文提示提升性能，验证了无需微调即可捕捉口语流利度与语义信息。**

- **链接: [http://arxiv.org/pdf/2510.16387v1](http://arxiv.org/pdf/2510.16387v1)**

> **作者:** Fu-An Chao; Bi-Cheng Yan; Berlin Chen
>
> **摘要:** In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks.
>
---
## 更新

#### [replaced 001] BINAQUAL: A Full-Reference Objective Localization Similarity Metric for Binaural Audio
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.11915v2](http://arxiv.org/pdf/2505.11915v2)**

> **作者:** Davoud Shariat Panah; Dan Barry; Alessandro Ragano; Jan Skoglund; Andrew Hines
>
> **备注:** Accepted for publication in the Journal of Audio Engineering Society (JAES)
>
> **摘要:** Spatial audio enhances immersion in applications such as virtual reality, augmented reality, gaming, and cinema by creating a three-dimensional auditory experience. Ensuring the spatial fidelity of binaural audio is crucial, given that processes such as compression, encoding, or transmission can alter localization cues. While subjective listening tests like MUSHRA remain the gold standard for evaluating spatial localization quality, they are costly and time-consuming. This paper introduces BINAQUAL, a full-reference objective metric designed to assess localization similarity in binaural audio recordings. BINAQUAL adapts the AMBIQUAL metric, originally developed for localization quality assessment in ambisonics audio format to the binaural domain. We evaluate BINAQUAL across five key research questions, examining its sensitivity to variations in sound source locations, angle interpolations, surround speaker layouts, audio degradations, and content diversity. Results demonstrate that BINAQUAL effectively differentiates between subtle spatial variations and correlates strongly with subjective listening tests, making it a reliable metric for binaural localization quality assessment. The proposed metric provides a robust benchmark for ensuring spatial accuracy in binaural audio processing, paving the way for improved objective evaluations in immersive audio applications.
>
---
#### [replaced 002] SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement
- **分类: eess.AS; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.07634v4](http://arxiv.org/pdf/2506.07634v4)**

> **作者:** Chenyu Yang; Shuai Wang; Hangting Chen; Wei Tan; Jianwei Yu; Haizhou Li
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** Generating music with coherent structure, harmonious instrumental and vocal elements remains a significant challenge in song generation. Existing language models and diffusion-based methods often struggle to balance global coherence with local fidelity, resulting in outputs that lack musicality or suffer from incoherent progression and mismatched lyrics. This paper introduces $\textbf{SongBloom}$, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models. Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process. Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms. Audio samples are available on our demo page: https://cypress-yang.github.io/SongBloom_demo. The code and model weights have been released on https://github.com/Cypress-Yang/SongBloom .
>
---
#### [replaced 003] Speech Foundation Models Generalize to Time Series Tasks from Wearable Sensor Data
- **分类: cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.00221v2](http://arxiv.org/pdf/2509.00221v2)**

> **作者:** Jaya Narain; Zakaria Aldeneh; Shirley Ren
>
> **备注:** Preprint, under review
>
> **摘要:** Both speech and sensor time series data encode information in both the time- and frequency- domains, like spectral powers and waveform shapelets. We show that speech foundation models learn representations that generalize beyond the speech domain and achieve state-of-the-art performance on diverse time-series tasks from wearable sensors. Probes trained on features extracted from HuBERT and wav2vec 2.0 outperform those extracted from self-supervised models trained directly on modality-specific datasets for mood classification, arrhythmia detection, and activity classification tasks. We find that the convolutional feature encoders of speech models are particularly relevant for wearable sensor applications. The proposed approach enhances performance on data-scarce time-series tasks using simple probing methods. This work takes a step toward developing generalized time-series models that unify speech and sensor modalities.
>
---
#### [replaced 004] Nexus: An Omni-Perceptive And -Interactive Model for Language, Audio, And Vision
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.01879v4](http://arxiv.org/pdf/2503.01879v4)**

> **作者:** Che Liu; Yingji Zhang; Dong Zhang; Weijie Zhang; Chenggong Gong; Yu Lu; Shilin Zhou; Ziliang Gan; Ziao Wang; Haipang Wu; Ji Liu; André Freitas; Qifan Wang; Zenglin Xu; Rongjuncheng Zhang; Yong Dai
>
> **备注:** Project: https://github.com/HiThink-Research/NEXUS-O
>
> **摘要:** This work proposes an industry-level omni-modal large language model (LLM) pipeline that integrates auditory, visual, and linguistic modalities to overcome challenges such as limited tri-modal datasets, high computational costs, and complex feature alignments. Our pipeline consists of three main components: First, a modular framework enabling flexible configuration of various encoder-LLM-decoder architectures. Second, a lightweight training strategy that pre-trains audio-language alignment on the state-of-the-art vision-language model Qwen2.5-VL, thus avoiding the costly pre-training of vision-specific modalities. Third, an audio synthesis pipeline that generates high-quality audio-text data from diverse real-world scenarios, supporting applications such as Automatic Speech Recognition and Speech-to-Speech chat. To this end, we introduce an industry-level omni-modal LLM, Nexus. Extensive experiments validate the efficacy of our pipeline, yielding the following key findings:(1) In the visual understanding task, Nexus exhibits superior performance compared with its backbone model - Qwen2.5-VL-7B, validating the efficiency of our training strategy. (2) Within the English Spoken Question-Answering task, the model achieves better accuracy than the same-period competitor (i.e, MiniCPM-o2.6-7B) in the LLaMA Q. benchmark. (3) In our real-world ASR testset, Nexus achieves outstanding performance, indicating its robustness in real scenarios. (4) In the Speech-to-Text Translation task, our model outperforms Qwen2-Audio-Instruct-7B. (5) In the Text-to-Speech task, based on pretrained vocoder (e.g., Fishspeech1.4 or CosyVoice2.0), Nexus is comparable to its backbone vocoder on Seed-TTS benchmark. (6) An in-depth analysis of tri-modal alignment reveals that incorporating the audio modality enhances representational alignment between vision and language.
>
---
#### [replaced 005] Guitar Tone Morphing by Diffusion-based Model
- **分类: eess.AS; 68T45; I.2.7; H.5.5**

- **链接: [http://arxiv.org/pdf/2510.07908v2](http://arxiv.org/pdf/2510.07908v2)**

> **作者:** Kuan-Yu Chen; Kuan-Lin Chen; Yu-Chieh Yu; Jian-Jiun Ding
>
> **备注:** 5 pages, accepted to the APSIPA ASC 2025
>
> **摘要:** In Music Information Retrieval (MIR), modeling and transforming the tone of musical instruments, particularly electric guitars, has gained increasing attention due to the richness of the instrument tone and the flexibility of expression. Tone morphing enables smooth transitions between different guitar sounds, giving musicians greater freedom to explore new textures and personalize their performances. This study explores learning-based approaches for guitar tone morphing, beginning with LoRA fine-tuning to improve the model performance on limited data. Moreover, we introduce a simpler method, named spherical interpolation using Music2Latent. It yields significantly better results than the more complex fine-tuning approach. Experiments show that the proposed architecture generates smoother and more natural tone transitions, making it a practical and efficient tool for music production and real-time audio effects.
>
---
#### [replaced 006] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.23305v3](http://arxiv.org/pdf/2505.23305v3)**

> **作者:** Yunkee Chae; Kyogu Lee
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present MGE-LDM, a unified latent diffusion framework for simultaneous music generation, source imputation, and query-driven source separation. Unlike prior approaches constrained to fixed instrument classes, MGE-LDM learns a joint distribution over full mixtures, submixtures, and individual stems within a single compact latent diffusion model. At inference, MGE-LDM enables (1) complete mixture generation, (2) partial generation (i.e., source imputation), and (3) text-conditioned extraction of arbitrary sources. By formulating both separation and imputation as conditional inpainting tasks in the latent space, our approach supports flexible, class-agnostic manipulation of arbitrary instrument sources. Notably, MGE-LDM can be trained jointly across heterogeneous multi-track datasets (e.g., Slakh2100, MUSDB18, MoisesDB) without relying on predefined instrument categories. Audio samples are available at our project page: https://yoongi43.github.io/MGELDM_Samples/.
>
---
#### [replaced 007] Test-Time Training for Speech Enhancement
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.01847v2](http://arxiv.org/pdf/2508.01847v2)**

> **作者:** Avishkar Behera; Riya Ann Easow; Venkatesh Parvathala; K. Sri Rama Murty
>
> **备注:** Published in the Proceedings of Interspeech 2025
>
> **摘要:** This paper introduces a novel application of Test-Time Training (TTT) for Speech Enhancement, addressing the challenges posed by unpredictable noise conditions and domain shifts. This method combines a main speech enhancement task with a self-supervised auxiliary task in a Y-shaped architecture. The model dynamically adapts to new domains during inference time by optimizing the proposed self-supervised tasks like noise-augmented signal reconstruction or masked spectrogram prediction, bypassing the need for labeled data. We further introduce various TTT strategies offering a trade-off between adaptation and efficiency. Evaluations across synthetic and real-world datasets show consistent improvements across speech quality metrics, outperforming the baseline model. This work highlights the effectiveness of TTT in speech enhancement, providing insights for future research in adaptive and robust speech processing.
>
---
#### [replaced 008] A Self-Attention-Driven Deep Denoiser Model for Real Time Lung Sound Denoising in Noisy Environments
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2404.04365v3](http://arxiv.org/pdf/2404.04365v3)**

> **作者:** Samiul Based Shuvo; Syed Samiul Alam; Taufiq Hasan
>
> **摘要:** Objective: Lung auscultation is a valuable tool in diagnosing and monitoring various respiratory diseases. However, lung sounds (LS) are significantly affected by numerous sources of contamination, especially when recorded in real-world clinical settings. Conventional denoising models prove impractical for LS denoising, primarily owing to spectral overlap complexities arising from diverse noise sources. To address this issue, we propose a specialized deep-learning model (Uformer) for lung sound denoising. Methods: The proposed Uformer model is constituted of three modules: a Convolutional Neural Network (CNN) encoder module, dedicated to extracting latent features; a Transformer encoder module, employed to further enhance the encoding of unique LS features and effectively capture intricate long-range dependencies; and a CNN decoder module, employed to generate the denoised signals. An ablation study was performed in order to find the most optimal architecture. Results: The performance of the proposed Uformer model was evaluated on lung sounds induced with different types of synthetic and real-world noises. Lung sound signals of -12 dB to 15 dB signal-to-noise ratio (SNR) were considered in testing experiments. The proposed model showed an average SNR improvement of 16.51 dB when evaluated with -12 dB LS signals. Our end-to-end model, with an average SNR improvement of 19.31 dB, outperforms the existing model when evaluated with ambient noise and fewer parameters. Conclusion: Based on the qualitative and quantitative findings in this study, it can be stated that Uformer is robust and generalized to be used in assisting the monitoring of respiratory conditions.
>
---
#### [replaced 009] SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.06917v2](http://arxiv.org/pdf/2510.06917v2)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Linjie Li; Chung-Ching Lin; Kevin Lin; Shujie Liu; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **备注:** Work in progress
>
> **摘要:** Current large language models (LLMs) and spoken language models (SLMs) begin thinking and taking actions only after the user has finished their turn. This prevents the model from interacting during the user's turn and can lead to high response latency while it waits to think. Consequently, thinking after receiving the full input is not suitable for speech-to-speech interaction, where real-time, low-latency exchange is important. We address this by noting that humans naturally "think while listening." In this paper, we propose SHANKS, a general inference framework that enables SLMs to generate unspoken chain-of-thought reasoning while listening to the user input. SHANKS streams the input speech in fixed-duration chunks and, as soon as a chunk is received, generates unspoken reasoning based on all previous speech and reasoning, while the user continues speaking. SHANKS uses this unspoken reasoning to decide whether to interrupt the user and to make tool calls to complete the task. We demonstrate that SHANKS enhances real-time user-SLM interaction in two scenarios: (1) when the user is presenting a step-by-step solution to a math problem, SHANKS can listen, reason, and interrupt when the user makes a mistake, achieving 37.1% higher interruption accuracy than a baseline that interrupts without thinking; and (2) in a tool-augmented dialogue, SHANKS can complete 56.9% of the tool calls before the user finishes their turn. Overall, SHANKS moves toward models that keep thinking throughout the conversation, not only after a turn ends. Animated illustrations of Shanks can be found at https://d223302.github.io/SHANKS/
>
---
#### [replaced 010] Post-training for Deepfake Speech Detection
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21090v3](http://arxiv.org/pdf/2506.21090v3)**

> **作者:** Wanying Ge; Xin Wang; Xuechen Liu; Junichi Yamagishi
>
> **备注:** Corrected previous implementation of EER calculation. Slight numerical changes in some of the results
>
> **摘要:** We introduce a post-training approach that adapts self-supervised learning (SSL) models for deepfake speech detection by bridging the gap between general pre-training and domain-specific fine-tuning. We present AntiDeepfake models, a series of post-trained models developed using a large-scale multilingual speech dataset containing over 56,000 hours of genuine speech and 18,000 hours of speech with various artifacts in over one hundred languages. Experimental results show that the post-trained models already exhibit strong robustness and generalization to unseen deepfake speech. When they are further fine-tuned on the Deepfake-Eval-2024 dataset, these models consistently surpass existing state-of-the-art detectors that do not leverage post-training. Model checkpoints and source code are available online.
>
---
#### [replaced 011] Late Fusion and Multi-Level Fission Amplify Cross-Modal Transfer in Text-Speech LMs
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.06211v2](http://arxiv.org/pdf/2503.06211v2)**

> **作者:** Santiago Cuervo; Adel Moumen; Yanis Labrak; Sameer Khurana; Antoine Laurent; Mickael Rouvier; Phil Woodland; Ricard Marxer
>
> **摘要:** Text-Speech Language Models (TSLMs) -- language models trained to jointly process and generate text and speech -- are commonly trained through an early modality fusion/fission approach, in which both modalities are fed and predicted from a shared backbone via linear layers. We hypothesize that this approach limits cross-modal transfer by neglecting feature compositionality -- specifically, the finer-grained nature of speech representations compared to text -- preventing the emergence of a shared feature hierarchy within model layers. In this paper, we argue that this limitation can be addressed through late fusion and fission, with a fission process that accesses both high- and low-level features for speech generation. Our models implementing these principles, SmolTolk, rival or surpass state-of-the-art TSLMs trained with orders of magnitude more compute, and achieve significantly improved cross-modal performance relative to early fusion/fission baselines. Representation analyses further suggest that our method enhances the model's ability to abstract higher-level, more semantic features from speech, and leads to increasingly shared representation spaces across layers.
>
---
#### [replaced 012] CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00885v2](http://arxiv.org/pdf/2506.00885v2)**

> **作者:** Leying Zhang; Yao Qian; Xiaofei Wang; Manthan Thakker; Dongmei Wang; Jianwei Yu; Haibin Wu; Yuxuan Hu; Jinyu Li; Yanmin Qian; Sheng Zhao
>
> **备注:** Neural Information Processing Systems 2025, poster
>
> **摘要:** Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios.
>
---
#### [replaced 013] VGGSounder: Audio-Visual Evaluations for Foundation Models
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08237v3](http://arxiv.org/pdf/2508.08237v3)**

> **作者:** Daniil Zverev; Thaddäus Wiedemer; Ameya Prabhu; Matthias Bethge; Wieland Brendel; A. Sophia Koepke
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The emergence of audio-visual foundation models underscores the importance of reliably assessing their multi-modal understanding. The VGGSound dataset is commonly used as a benchmark for evaluation audio-visual classification. However, our analysis identifies several limitations of VGGSound, including incomplete labelling, partially overlapping classes, and misaligned modalities. These lead to distorted evaluations of auditory and visual capabilities. To address these limitations, we introduce VGGSounder, a comprehensively re-annotated, multi-label test set that extends VGGSound and is specifically designed to evaluate audio-visual foundation models. VGGSounder features detailed modality annotations, enabling precise analyses of modality-specific performance. Furthermore, we reveal model limitations by analysing performance degradation when adding another input modality with our new modality confusion metric.
>
---
