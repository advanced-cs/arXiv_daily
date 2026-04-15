# 音频 cs.SD;  eess.AS

- **最新发布 21 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark
- **分类: eess.AS**

- **简介: 该论文提出VoxEffects数据集和基准，解决语音音频效果识别问题，支持效果检测、分类与强度预测，提升语音处理的系统研究。**

- **链接: [https://arxiv.org/pdf/2604.12389](https://arxiv.org/pdf/2604.12389)**

> **作者:** Zhe Zhang; Yigitcan Özer; Junichi Yamagishi
>
> **摘要:** Speech audio in the wild is often processed by post-production effects, but existing speech datasets rarely provide precise annotations of effects and parameters, limiting systematic study. We introduce VoxEffects, a speech audio effects dataset that pairs produced speech with exact effect-chain supervision at multiple granularities. VoxEffects supports speech-oriented audio effect identification: given a produced waveform, infer which effects are present and how they are applied. Built from minimally edited clean speech, it provides an extensible rendering pipeline for both offline synthesis and on-the-fly rendering for efficient training and evaluation. The audio effect identification benchmark includes effect presence detection, preset classification, and intensity prediction, with a robustness protocol covering capture-side and platform-side degradations. We provide an AudioMAE-based multi-task baseline and analyses of domain shift, robustness, input duration, and gender fairness.
>
---
#### [new 002] On the Distillation Loss Functions of Speech VAE for Unified Reconstruction, Understanding, and Generation
- **分类: cs.SD**

- **简介: 该论文属于语音生成与理解任务，旨在解决VAE模型中潜在表示对齐问题。通过分析不同对齐方法，提出一种自适应加权联合边缘对齐方案，提升重建、理解和生成性能。**

- **链接: [https://arxiv.org/pdf/2604.12383](https://arxiv.org/pdf/2604.12383)**

> **作者:** Changhao Cheng; Wei Wang; Wangyou Zhang; Dongya Jia; Jian Wu; Zhuo Chen; Yanmin Qian
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Continuous speech representations based on Variational Autoencoders (VAEs) have emerged as a promising alternative to traditional spectrogram or discrete token based features for speech generation and reconstruction. Recent research has tried to enrich the structural information in VAE latent representations by aligning with self-supervised learning (SSL) features, aiming for better generation performance. However, it remains unclear whether the widely-used alignment approach based on time-axis distillation is optimal when considering more tasks. To address this problem, this paper systematically explores different alignment approaches and analyzes their impact on the performances over three axes: reconstruction, understanding, and generation. We investigate various design choices in the distillation loss. Extensive experiments show that the joint-marginal alignment approach with adaptive weighting can achieve the best overall performance while allowing for a controllable balance.
>
---
#### [new 003] X-VC: Zero-shot Streaming Voice Conversion in Codec Space
- **分类: eess.AS; cs.AI**

- **简介: 该论文提出X-VC，解决零样本语音转换中的高保真与低延迟问题，通过编码器空间的一步转换实现高效语音转换。**

- **链接: [https://arxiv.org/pdf/2604.12456](https://arxiv.org/pdf/2604.12456)**

> **作者:** Qixi Zheng; Yuxiang Zhao; Tianrui Wang; Wenxi Chen; Kele Xu; Yikang Li; Qinyuan Chen; Xipeng Qiu; Kai Yu; Xie Chen
>
> **摘要:** Zero-shot voice conversion (VC) aims to convert a source utterance into the voice of an unseen target speaker while preserving its linguistic content. Although recent systems have improved conversion quality, building zero-shot VC systems for interactive scenarios remains challenging because high-fidelity speaker transfer and low-latency streaming inference are difficult to achieve simultaneously. In this work, we present X-VC, a zero-shot streaming VC system that performs one-step conversion in the latent space of a pretrained neural codec. X-VC uses a dual-conditioning acoustic converter that jointly models source codec latents and frame-level acoustic conditions derived from target reference speech, while injecting utterance-level target speaker information through adaptive normalization. To reduce the mismatch between training and inference, we train the model with generated paired data and a role-assignment strategy that combines standard, reconstruction, and reversed modes. For streaming inference, we further adopt a chunkwise inference scheme with overlap smoothing that is aligned with the segment-based training paradigm of the codec. Experiments on Seed-TTS-Eval show that X-VC achieves the best streaming WER in both English and Chinese, strong speaker similarity in same-language and cross-lingual settings, and substantially lower offline real-time factor than the compared baselines. These results suggest that codec-space one-step conversion is a practical approach for building high-quality low-latency zero-shot VC systems. Audio samples are available at this https URL. Our code and checkpoints will also be released.
>
---
#### [new 004] Sky-Ear: An Unmanned Aerial Vehicle-Enabled Victim Sound Detection and Localization System
- **分类: eess.AS**

- **简介: 该论文属于搜救任务中的声音检测与定位问题。针对UAV硬件限制，设计了Sky-Ear系统，通过双阶段音频处理和MAE方法实现高效声学感知与精确定位。**

- **链接: [https://arxiv.org/pdf/2604.12455](https://arxiv.org/pdf/2604.12455)**

> **作者:** Yi Hong; Mingyang Wang; Yalin Liu; Yaru Fu; Kevin Hung
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in search-and-rescue (SAR) missions, yet continuous and reliable victim detection and localization remain challenging due to on-board hardware constraints. This paper designs an UAV-Enabled Victim Sound Detection and Localization System (called ``Sky-Ear'' for brevity) to achieve energy-efficient acoustic sensing and sound detection for SAR. Based on a circular-shaped microphone array, two-stage (Sentinel and Responder) audio processing is developed for energy-consuming and highly reliable sound detection. A Masking autoencoder (MAE)-based sound detection method is designed in the Sentinel stage to analyze frequency-time acoustic features. For improved precision, a continuous localization method is designed by optimizing detected directions from multiple observations. Extensive simulation experiments are conducted to validate the system's performance in terms of victim detection accuracy and localization error.
>
---
#### [new 005] CoSyncDiT: Cognitive Synchronous Diffusion Transformer for Movie Dubbing
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文属于电影配音任务，旨在解决语音与唇形同步及自然度不足的问题。提出CoSync-DiT框架，通过认知同步机制提升对齐精度和语音质量。**

- **链接: [https://arxiv.org/pdf/2604.12292](https://arxiv.org/pdf/2604.12292)**

> **作者:** Gaoxiang Cong; Liang Li; Jiaxin Ye; Zhedong Zhang; Hongming Shan; Yuankai Qi; Qingming Huang
>
> **摘要:** Movie dubbing aims to synthesize speech that preserves the vocal identity of a reference audio while synchronizing with the lip movements in a target video. Existing methods fail to achieve precise lip-sync and lack naturalness due to explicit alignment at the duration level. While implicit alignment solutions have emerged, they remain susceptible to interference from the reference audio, triggering timbre and pronunciation degradation in in-the-wild scenarios. In this paper, we propose a novel flow matching-based movie dubbing framework driven by the Cognitive Synchronous Diffusion Transformer (CoSync-DiT), inspired by the cognitive process of professional actors. This architecture progressively guides the noise-to-speech generative trajectory by executing acoustic style adapting, fine-grained visual calibrating, and time-aware context aligning. Furthermore, we design the Joint Semantic and Alignment Regularization (JSAR) mechanism to simultaneously constrain frame-level temporal consistency on the contextual outputs and semantic consistency on the flow hidden states, ensuring robust alignment. Extensive experiments on both standard benchmarks and challenging in-the-wild dubbing benchmarks demonstrate that our method achieves the state-of-the-art performance across multiple metrics.
>
---
#### [new 006] TokenSE: a Mamba-based discrete token speech enhancement framework for cochlear implants
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升耳蜗植入用户在噪声和混响环境中的语音可懂度。提出TokenSE框架，利用Mamba模型进行离散令牌预测，有效改善语音质量。**

- **链接: [https://arxiv.org/pdf/2604.12246](https://arxiv.org/pdf/2604.12246)**

> **作者:** Hsin-Tien Chiang; John H. L. Hansen
>
> **摘要:** Speech enhancement (SE) is critical for improving speech intelligibility and quality in real-world environments, particularly for cochlear implant (CI) users who experience severe degradations in speech understanding under noisy and reverberant conditions. In this study, we propose TokenSE, a discrete token-based SE framework operating in the neural audio codec space, which predicts clean codec token indices from degraded speech using a Mamba-based model. Unlike the earlier Transformer architecture, whose self-attention mechanism has a computational complexity that grows quadratically with sequence length, the input-dependent selection mechanism of Mamba achieves linear complexity, making it a compelling alternative to Transformers, especially for CI and hearing-aid (HA) applications. Objective evaluations show that TokenSE consistently outperforms baseline methods on both in-domain and out-of-domain datasets. Moreover, subjective listening experiments with CI users indicate clear benefit in speech intelligibility under adverse noisy and reverberant environments.
>
---
#### [new 007] SpotSound: Enhancing Large Audio-Language Models with Fine-Grained Temporal Grounding
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频-语言模型任务，旨在解决时间定位问题。针对现有模型在长音频中精确定位事件的不足，提出SpotSound模型及基准测试，提升时间定位准确性。**

- **链接: [https://arxiv.org/pdf/2604.13023](https://arxiv.org/pdf/2604.13023)**

> **作者:** Luoyi Sun; Xiao Zhou; Zeqian Li; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Large Audio-Language Models (ALMs) have recently demonstrated remarkable capabilities in holistic audio understanding, yet they remain unreliable for temporal grounding, i.e., the task of pinpointing exactly when an event occurs within long-form audio. This limitation stems from two factors: training data dominated by clip-level supervision lacking precise timestamps, and benchmarks that fail to simulate real-world scenarios where short events are obscured by dense background sounds. In this paper, we introduce SpotSound, an audio language model designed for grounding audio events. SpotSound incorporates a novel training objective, specifically designed to suppress hallucinated timestamps for events absent from the input. Additionally, we present SpotSound-Bench, a challenging temporal grounding benchmark where target events occupy less than ~10\% of each clip, creating a rigorous `needle-in-a-haystack' evaluation. Experiments demonstrate that SpotSound achieves state-of-the-art results on temporal grounding benchmarks while maintaining robust performance across general downstream audio-language tasks. Code, models and benchmark are released on this https URL
>
---
#### [new 008] Elastic Net Regularization and Gabor Dictionary for Classification of Heart Sound Signals using Deep Learning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于心音信号分类任务，旨在提升分类性能。通过结合弹性网正则化和Gabor字典优化特征表示，并采用深度学习模型进行分类。**

- **链接: [https://arxiv.org/pdf/2604.12483](https://arxiv.org/pdf/2604.12483)**

> **作者:** Mahmoud Fakhry; Ascensión Gallardo-Antolín
>
> **摘要:** In this article, we propose the optimization of the resolution of time-frequency atoms and the regularization of fitting models to obtain better representations of heart sound signals. This is done by evaluating the classification performance of deep learning (DL) networks in discriminating five heart valvular conditions based on a new class of time-frequency feature matrices derived from the fitting models. We inspect several combinations of resolution and regularization, and the optimal one is that provides the highest performance. To this end, a fitting model is obtained based on a heart sound signal and an overcomplete dictionary of Gabor atoms using elastic net regularization of linear models. We consider two different DL architectures, the first mainly consisting of a 1D convolutional neural network (CNN) layer and a long short-term memory (LSTM) layer, while the second is composed of 1D and 2D CNN layers followed by an LSTM layer. The networks are trained with two algorithms, namely stochastic gradient descent with momentum (SGDM) and adaptive moment (ADAM). Extensive experimentation has been conducted using a database containing heart sound signals of five heart valvular conditions. The best classification accuracy of $98.95\%$ is achieved with the second architecture when trained with ADAM and feature matrices derived from optimal models obtained with a Gabor dictionary consisting of atoms with high-time low-frequency resolution and imposing sparsity on the models.
>
---
#### [new 009] Audio-Cogito: Towards Deep Audio Reasoning in Large Audio Language Models
- **分类: eess.AS**

- **简介: 该论文属于音频推理任务，旨在解决大音频语言模型中推理能力不足的问题。通过构建高质量数据集并采用自蒸馏策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12527](https://arxiv.org/pdf/2604.12527)**

> **作者:** Longhao Li; Hongjie Chen; Zehan Li; Qihan Hu; Jian Kang; Jie Li; Lei Xie; Yongxiang Li
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advances in reasoning models have driven significant progress in text and multimodal domains, yet audio reasoning remains relatively limited. Only a few Large Audio Language Models (LALMs) incorporate explicit Chain-of-Thought (CoT) reasoning, and their capabilities are often inconsistent and insufficient for complex tasks. To bridge this gap, we introduce Audio-Cogito, a fully open-source solution for deep audio reasoning. We develop Cogito-pipe for high-quality audio reasoning data curation, producing 545k reasoning samples that will be released after review. Based on this dataset, we adopt a self-distillation strategy for model fine-tuning. Experiments on the MMAR benchmark, the only audio benchmark evaluating the CoT process, show that our model achieves the best performance among open-source models and matches or surpasses certain closed-source models in specific metrics. Our approach also ranks among the top-tier systems in the Interspeech 2026 Audio Reasoning Challenge.
>
---
#### [new 010] Why Your Tokenizer Fails in Information Fusion: A Timing-Aware Pre-Quantization Fusion for Video-Enhanced Audio Tokenization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频语言模型任务，解决音频分词器在多模态融合中重建质量下降的问题。提出时序感知的预量化融合方法，提升音频分词效果。**

- **链接: [https://arxiv.org/pdf/2604.12145](https://arxiv.org/pdf/2604.12145)**

> **作者:** Xiangyu Zhang; Benjamin John Southwell; Siqi Pan; Xinlei Niu; Beena Ahmed; Julien Epps
>
> **摘要:** Audio tokenization has emerged as a critical component in end-to-end audio language models, enabling efficient discrete representation learning for both audio understanding and generation tasks. However, existing audio tokenizers face fundamental limitations in understanding tasks due to single-modality constraints, particularly when audio signals contain ambiguous or incomplete information. While incorporating additional modality information can significantly enhance audio understanding, current multimodal fusion approaches invariably degrade reconstruction quality. This degradation is unacceptable for end-to-end audio systems that require high-fidelity audio generation capabilities. In this work, we investigate the root causes of reconstruction quality degradation in video-enhanced audio tokenization and present three key findings. First, the location of fusion within the tokenizer architecture is crucial for preserving reconstruction quality. Second, we show that contrastive learning, though effective in continuous representation fusion, is unsuitable for discrete tokenizers as it fails to enhance downstream task performance. Third, while feature-dimension fusion approaches achieve moderate success, we discover that fusing along the temporal axis -- guided by the concept of distinctive features -- yields significantly better results. Building on these insights, we introduce the Timing-Aware Pre-Quantization Fusion for Video-Enhanced Audio Tokenization, the first approach to successfully integrate visual information into audio tokenizer architectures while preserving reconstruction fidelity. Our approach not only maintains high-fidelity reconstruction but also achieves superior performance on downstream understanding tasks compared with audio-only tokenizers and established multimodal fusion baselines.
>
---
#### [new 011] StreamMark: A Deep Learning-Based Semi-Fragile Audio Watermarking for Proactive Deepfake Detection
- **分类: eess.AS**

- **简介: 该论文属于音频水印任务，旨在解决深度伪造音频检测问题。提出StreamMark系统，通过深度学习实现半脆弱水印，区分良性与恶意篡改。**

- **链接: [https://arxiv.org/pdf/2604.11917](https://arxiv.org/pdf/2604.11917)**

> **作者:** Zhentao Liu; Milos Cernak
>
> **备注:** ICASSP 2026
>
> **摘要:** The rapid advancement of generative AI has made it increasingly challenging to distinguish between deepfake audio and authentic human speech. To overcome the limitations of passive detection methods, we propose StreamMark, a novel deep learning-based, semi-fragile audio watermarking system. StreamMark is designed to be robust against benign audio conversions that preserve semantic meaning (e.g., compression, noise) while remaining fragile to malicious, semantics-altering manipulations (e.g., voice conversion, speech editing). Our method introduces a complex-domain embedding technique within a unique Encoder-Distortion-Decoder architecture, trained explicitly to differentiate between these two classes of transformations. Comprehensive benchmarks demonstrate that StreamMark achieves high imperceptibility (SNR 24.16 dB, PESQ 4.20), is resilient to real-world distortions like Opus encoding, and exhibits principled fragility against a suite of deepfake attacks, with message recovery accuracy dropping to chance levels (~50%), while remaining robust to benign AI-based style transfers (ACC >98%).
>
---
#### [new 012] Four Decades of Digital Waveguides
- **分类: eess.AS**

- **简介: 论文综述数字波导建模的发展与应用，解决声波模拟效率问题，通过优化方法提升计算性能。**

- **链接: [https://arxiv.org/pdf/2604.12878](https://arxiv.org/pdf/2604.12878)**

> **作者:** Pablo Tablas de Paula; Julius O. Smith III; Vesa Välimäki; Joshua D. Reiss
>
> **摘要:** Digital waveguide physical modeling offers efficient simulation of acoustic wave propagation as compared to general finite-difference schemes commonly used in computational physics. This efficiency has enabled the real-time implementation of physically modeled musical instruments and sound effects, as well as real-time vocal models and artificial reverberation. This paper provides an overview of the historical evolution and applications of digital waveguide modeling and highlights recent advances in the field. Parametric optimization using classical, evolutionary and neural approaches are also discussed and compared. Digital waveguides provide physically accurate simulations with reduced computational cost, and can now be optimized with modern machine learning and differentiable digital signal processing techniques.
>
---
#### [new 013] Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于零样本呼吸音频分类任务，解决标注数据稀缺问题。提出TRIAGE框架，通过分层推理动态分配计算资源，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12647](https://arxiv.org/pdf/2604.12647)**

> **作者:** Tsai-Ning Wang; Herman Teun den Dekker; Lin-Lin Chen; Neil Zeghidour; Aaqib Saeed
>
> **备注:** Accepted at AHLI CHIL 2026
>
> **摘要:** Automated respiratory audio analysis promises scalable, non-invasive disease screening, yet progress is limited by scarce labeled data and costly expert annotation. Zero-shot inference eliminates task-specific supervision, but existing methods apply uniform computation to every input regardless of difficulty. We introduce TRIAGE, a tiered zero-shot framework that adaptively scales test-time compute by routing each audio sample through progressively richer reasoning stages: fast label-cosine scoring in a joint audio-text embedding space (Tier-L), structured matching with clinician-style descriptors (Tier-M), and retrieval-augmented large language model reasoning (Tier-H). A confidence-based router finalizes easy predictions early while allocating additional computation to ambiguous inputs, enabling nearly half of all samples to exit at the cheapest tier. Across nine respiratory classification tasks without task-specific training, TRIAGE achieves a mean AUROC of 0.744, outperforming prior zero-shot methods and matching or exceeding supervised baselines on multiple tasks. Our analysis show that test-time scaling concentrates gains where they matter: uncertain cases see up to 19% relative improvement while confident predictions remain unchanged at minimal cost.
>
---
#### [new 014] An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise Generation and Depth-Wise Codec Decoding
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决实时应用中延迟与音质的平衡问题。提出一种低延迟的端到端架构，通过块生成和深度解码优化，提升效率并减少失真。**

- **链接: [https://arxiv.org/pdf/2604.12438](https://arxiv.org/pdf/2604.12438)**

> **作者:** Tianhui Su; Tien-Ping Tan; Salima Mdhaffar; Yannick Estève; Aghilas Sini
>
> **备注:** 29 pages, 5 figures
>
> **摘要:** Real-time speech synthesis requires balancing inference latency and acoustic fidelity for interactive applications. Conventional continuous text-to-speech pipelines require computationally intensive neural vocoders to reconstruct phase information, creating a significant streaming bottleneck. Furthermore, regression-based acoustic modeling frequently induces spectral over-smoothing artifacts. To address these limitations, this paper proposes a novel end-to-end non-autoregressive architecture optimized for ultra-low latency block-wise generation, directly modeling the highly compressed discrete latent space of the Mimi neural audio codec. Integrating a modified FastSpeech 2 backbone with a progressive depth-wise sequential decoding strategy, the architecture dynamically conditions 32 layers of residual vector quantization codes. This mechanism resolves phonetic alignment degradation and manages the complexity of high-fidelity discrete representations without temporal autoregressive overhead. Experimental evaluations on English and Malay datasets validate its language-independent deployment capability. Compared to conventional continuous regression models, the proposed architecture demonstrates quantitative improvements in fundamental voicing accuracy and mitigates high-frequency spectral degradation. It achieves ultra-low latency inference, translating to a 10.6-fold absolute acceleration over conventional cascaded pipelines. Crucially, the system achieves an average time-to-first-byte latency of 48.99 milliseconds, falling significantly below the human perception threshold for real-time interactive streaming. These results firmly establish the proposed architecture as a highly optimized solution for deploying real-time streaming speech interfaces.
>
---
#### [new 015] Audio Source Separation in Reverberant Environments using $β$-divergence based Nonnegative Factorization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频源分离任务，旨在解决混响环境下的信号分离问题。通过基于β-散度的非负因子分解方法，优化源信号的谱方差和空间协方差估计，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2604.12480](https://arxiv.org/pdf/2604.12480)**

> **作者:** Mahmoud Fakhry; Piergiorgio Svaizer; Maurizio Omologo
>
> **摘要:** In Gaussian model-based multichannel audio source separation, the likelihood of observed mixtures of source signals is parametrized by source spectral variances and by associated spatial covariance matrices. These parameters are estimated by maximizing the likelihood through an Expectation-Maximization algorithm and used to separate the signals by means of multichannel Wiener filtering. We propose to estimate these parameters by applying nonnegative factorization based on prior information on source variances. In the nonnegative factorization, spectral basis matrices can be defined as the prior information. The matrices can be either extracted or indirectly made available through a redundant library that is trained in advance. In a separate step, applying nonnegative tensor factorization, two algorithms are proposed in order to either extract or detect the basis matrices that best represent the power spectra of the source signals in the observed mixtures. The factorization is achieved by minimizing the $\beta$-divergence through multiplicative update rules. The sparsity of factorization can be controlled by tuning the value of $\beta$. Experiments show that sparsity, rather than the value assigned to $\beta$ in the training, is crucial in order to increase the separation performance. The proposed method was evaluated in several mixing conditions. It provides better separation quality with respect to other comparable algorithms.
>
---
#### [new 016] Transformer Based Machine Fault Detection From Audio Input
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于机器故障检测任务，旨在利用音频数据识别机器异常。通过对比Transformer与CNN在频谱图分析中的表现，验证Transformer的有效性。**

- **链接: [https://arxiv.org/pdf/2604.12733](https://arxiv.org/pdf/2604.12733)**

> **作者:** Kiran Voderhobli Holla
>
> **摘要:** In recent years, Sound AI is being increasingly used to predict machine failures. By attaching a microphone to the machine of interest, one can get real time data on machine behavior from the field. Traditionally, Convolutional Neural Net (CNN) architectures have been used to analyze spectrogram images generated from the sounds captured and predict if the machine is functioning as expected. CNN architectures seem to work well empirically even though they have biases like locality and parameter-sharing which may not be completely relevant for spectrogram analysis. With the successful application of transformer-based models in the field of image processing starting with Vision Transformer (ViT) in 2020, there has been significant interest in leveraging these in the field of Sound AI. Since transformer-based architectures have significantly lower inductive biases, they are expected to perform better than CNNs at spectrogram analysis given enough data. This paper demonstrates the effectiveness of transformer-driven architectures in analyzing Sound data and compares the embeddings they generate with CNNs on the specific task of machine fault detection.
>
---
#### [new 017] Contextual Biasing for ASR in Speech LLM with Common Word Cues and Bias Word Position Prediction
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决ASR中罕见词识别问题。通过利用常见词的声学线索和位置预测，提升模型对偏见词的识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.12398](https://arxiv.org/pdf/2604.12398)**

> **作者:** Sashi Novitasari; Takashi Fukuda; Kurata Gakuto; George Saon
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Speech-aware LLMs (SLLMs) have recently achieved state-of-the-art ASR performance; however, they still fail to accurately transcribe bias words that appear rarely or never in the training data. Contextual biasing mechanisms are commonly implemented by introducing a predefined bias word list into the model via a text prompt or additional module. For further improvement, predefined bias words can be paired with their phoneme representations as pronunciation cues. Typically, phoneme sequences are generated through a G2P system that covers the target languages and domains of the bias words. Therefore, when a compatible G2P system is unavailable, phoneme-assisted contextual biasing becomes difficult to perform. Moreover, manually adding accurate phoneme sequences requires advanced phonetic knowledge. In this paper, we explore contextual biasing in SLLM based on acoustic cues associated with a set of common words whose pronunciations are partially similar to those of the target bias words. We assume ASR applications in which end users do not require special knowledge of phonetics or utilize G2P tools for inference. For enhanced robustness, we also introduce bias word positional prediction implemented in a multi-output learning fashion. Our method reduces bias word recognition errors by 16.3% compared to baseline systems, including on out-of-domain data.
>
---
#### [new 018] Room compensation for loudspeaker reproduction using a supporting source
- **分类: eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决房间环境对扬声器音质影响的问题。通过引入辅助声源，同时补偿频谱和空间特性，提升声音还原的准确性。**

- **链接: [https://arxiv.org/pdf/2604.12439](https://arxiv.org/pdf/2604.12439)**

> **作者:** James Brooks-Park; Søren Bech; Jan Østergaard; Steven van de Par
>
> **摘要:** Room compensation aims to improve the accuracy of loudspeaker reproduction in reverberant environments. Traditional methods, however, are limited to improving only spectral (timbral) and temporal accuracy, neglecting the spatial accuracy of loudspeaker reproduction. Proposed is a method that compensates for both spectral and spatial properties of loudspeaker reproduction, by adding energy to the perceived reverberant sound field in a frequency-selective manner using a delayed secondary supporting source. This approach allows for the modification of the direct to reverberant ratio as a function of frequency, altering spatial and spectral reproduction. The proposed method is perceptually evaluated, demonstrating its ability to alter the perception of a primary loudspeaker without the listener perceiving the supporting source. The results show that the proposed method performs comparably to a well-established commercial room compensation algorithm and has several advantages over traditional room compensation methods.
>
---
#### [new 019] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步知识检索，在保持交互性的同时提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [new 020] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音频语言模型任务，解决AudioLLM在细粒度声学感知上的不足。通过提出统一音频框架UAS，提升声学理解能力，同时保持推理性能。**

- **链接: [https://arxiv.org/pdf/2604.12506](https://arxiv.org/pdf/2604.12506)**

> **作者:** Linhao Zhang; Yuhan Song; Aiwei Liu; Chuhan Wu; Sijun Zhang; Wei Jia; Yuan Liu; Houfeng Wang; Xiao Zhou
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Recent Audio Large Language Models (AudioLLMs) exhibit a striking performance inversion: while excelling at complex reasoning tasks, they consistently underperform on fine-grained acoustic perception. We attribute this gap to a fundamental limitation of ASR-centric training, which provides precise linguistic targets but implicitly teaches models to suppress paralinguistic cues and acoustic events as noise. To address this, we propose Unified Audio Schema (UAS), a holistic and structured supervision framework that organizes audio information into three explicit components -- Transcription, Paralinguistics, and Non-linguistic Events -- within a unified JSON format. This design achieves comprehensive acoustic coverage without sacrificing the tight audio-text alignment that enables reasoning. We validate the effectiveness of this supervision strategy by applying it to both discrete and continuous AudioLLM architectures. Extensive experiments on MMSU, MMAR, and MMAU demonstrate that UAS-Audio yields consistent improvements, boosting fine-grained perception by 10.9% on MMSU over the same-size state-of-the-art models while preserving robust reasoning capabilities. Our code and model are publicly available at this https URL.
>
---
#### [new 021] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出StableToken，解决语音词器在噪声下的不稳定性问题。通过多分支结构和投票机制提升词器鲁棒性，增强下游语音大模型性能。**

- **链接: [https://arxiv.org/pdf/2509.22220](https://arxiv.org/pdf/2509.22220)**

> **作者:** Yuhan Song; Linhao Zhang; Chuhan Wu; Aiwei Liu; Wei Jia; Houfeng Wang; Xiao Zhou
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Prevalent semantic speech tokenizers, designed to capture linguistic content, are surprisingly fragile. We find they are not robust to meaning-irrelevant acoustic perturbations; even at high Signal-to-Noise Ratios (SNRs) where speech is perfectly intelligible, their output token sequences can change drastically, increasing the learning burden for downstream LLMs. This instability stems from two flaws: a brittle single-path quantization architecture and a distant training signal indifferent to intermediate token stability. To address this, we introduce StableToken, a tokenizer that achieves stability through a consensus-driven mechanism. Its multi-branch architecture processes audio in parallel, and these representations are merged via a powerful bit-wise voting mechanism to form a single, stable token sequence. StableToken sets a new state-of-the-art in token stability, drastically reducing Unit Edit Distance (UED) under diverse noise conditions. This foundational stability translates directly to downstream benefits, significantly improving the robustness of SpeechLLMs on a variety of tasks. Our code and model are publicly available at this https URL.
>
---
## 更新

#### [replaced 001] MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation
- **分类: cs.SD; cs.AI; cs.MA**

- **简介: 该论文提出MeloTune，用于个性化音乐推荐的任务，解决情绪感知与用户间情绪同步问题。通过CfC网络和PAF实现设备端情绪学习与情绪耦合。**

- **链接: [https://arxiv.org/pdf/2604.10815](https://arxiv.org/pdf/2604.10815)**

> **作者:** Hongwei Xu
>
> **备注:** 31 pages, 1 figures, 3 tables
>
> **摘要:** MeloTune is an iPhone-deployed music agent that instantiates the Mesh Memory Protocol (MMP) and Symbolic-Vector Attention Fusion (SVAF) as a production system for affect-aware music curation with peer-to-peer mood coupling. Each device runs two closed-form continuous-time (CfC) networks: a private listener-level CfC that predicts a short-horizon affective trajectory on Russell's circumplex and drives proactive curation, and a shared mesh-runtime CfC at MMP Layer 6 that integrates Cognitive Memory Blocks (CMBs) from co-listening peers. CfC hidden states never cross the wire; only structured CMBs do. A Personal Arousal Function (PAF) replaces the standard linear mapping from audio intensity to psychological arousal with a per-listener learned adjustment, trained from behavioral signals (skip, completion, favorite, volume) and from drift between user-declared mood and machine inference. The same track receives different arousal predictions for different listeners. The model (94,552 parameters) achieves trajectory MAE 0.414, pattern accuracy 96.6%, and intent accuracy 69.4% on held-out validation. PAF evidence from a live deployment session (46 observations across 11 genres) demonstrates that the learning loop operates end-to-end, with pop reaching full confidence after 22 observations. All inference runs on-device via CoreML. To our knowledge, this is the first production deployment of MMP/SVAF on consumer mobile hardware. The accompanying SDK (sym-swift v0.3.78, SYMCore v0.3.7) enforces strict protocol conformance. Music is the case study; the substrate is the contribution.
>
---
#### [replaced 002] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，探索其如何编码语音特征。任务是分析模型是否具备可解释的音系向量运算能力，通过实验发现模型中存在线性音系向量，并验证其可组合性。**

- **链接: [https://arxiv.org/pdf/2602.18899](https://arxiv.org/pdf/2602.18899)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at this https URL .
>
---
#### [replaced 003] Distributed Multichannel Wiener Filtering for Wireless Acoustic Sensor Networks
- **分类: eess.AS; cs.IT; eess.SP**

- **简介: 该论文属于语音增强任务，解决无线传感器网络中节点协作的信号估计问题。提出非迭代的dMWF算法，实现更高效、更优的语音信号处理。**

- **链接: [https://arxiv.org/pdf/2603.09735](https://arxiv.org/pdf/2603.09735)**

> **作者:** Paul Didier; Toon van Waterschoot; Simon Doclo; Jörg Bitzer; Pourya Behmandpoor; Henri Gode; Marc Moonen
>
> **摘要:** [This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.] In a wireless acoustic sensor network (WASN), devices (i.e., nodes) can collaborate through distributed algorithms to collectively perform audio signal processing tasks. This paper focuses on the distributed estimation of node-specific desired speech signals using network-wide Wiener filtering. The objective is to match the performance of a centralized system that would have access to all microphone signals, while reducing the communication bandwidth usage of the algorithm. Existing solutions, such as the distributed adaptive node-specific signal estimation (DANSE) algorithm, converge towards the multichannel Wiener filter (MWF) which solves a centralized linear minimum mean square error (LMMSE) signal estimation problem. However, they do so iteratively, which can be slow and impractical. Many solutions also assume that all nodes observe the same set of sources of interest, which is often not the case in practice. To overcome these limitations, we propose the distributed multichannel Wiener filter (dMWF) for fully connected WASNs. The dMWF is non-iterative and optimal even when nodes observe different sets of sources. In this algorithm, nodes exchange neighbor-pair-specific, low-dimensional (fused) signals estimating the contribution of sources observed by both nodes in the pair. We formally prove the optimality of dMWF and demonstrate its performance in simulated speech enhancement experiments. The proposed algorithm is shown to outperform DANSE in terms of objective metrics after short operation times, highlighting the benefit of its iterationless design.
>
---
#### [replaced 004] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音对话生成任务，解决传统模型推理慢、稳定性差的问题，提出ZipVoice-Dialog模型，采用非自回归和流匹配方法，提升生成速度与准确性。**

- **链接: [https://arxiv.org/pdf/2507.09318](https://arxiv.org/pdf/2507.09318)**

> **作者:** Han Zhu; Wei Kang; Liyong Guo; Zengwei Yao; Fangjun Kuang; Weiji Zhuang; Zhaoqing Li; Zhifeng Han; Dong Zhang; Xin Zhang; Xingchen Song; Lingxuan Ye; Long Lin; Daniel Povey
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Generating spoken dialogue is inherently more complex than monologue text-to-speech (TTS), as it demands both realistic turn-taking and the maintenance of distinct speaker timbres. While existing autoregressive (AR) models have made progress, they often suffer from high inference latency and stability issues. To overcome these limitations, we propose ZipVoice-Dialog, a non-autoregressive (NAR) zero-shot spoken dialogue generation model based on flow-matching. Observing that applying vanilla flow-matching to dialogue generation leads to poor speech intelligibility and turn-taking precision, we introduce two simple yet effective methods to adapt flow-matching architectures for dialogue generation: (1) a curriculum learning strategy to ensure robust speech-text alignment, and (2) speaker-turn embeddings to govern precise speaker turn-taking. Additionally, we introduce dedicated strategies to support stereo dialogue generation. Recognizing the lack of training datasets in this field, we curate and release OpenDialog, the first large-scale (6.8k hours) open-source spoken dialogue dataset derived from in-the-wild speech data. Moreover, for fair and rigorous evaluations, we established a benchmark to comprehensively evaluate dialogue generation models. Experiments demonstrate the effectiveness of the proposed methods and dataset, showing that ZipVoice-Dialog achieves superior performance in inference speed, intelligibility, speaker turn-taking accuracy, and speaker similarity. Our code, model checkpoints, and the OpenDialog dataset are publicly available at this https URL.
>
---
#### [replaced 005] TellWhisper: Tell Whisper Who Speaks When
- **分类: eess.AS**

- **简介: 该论文属于多说话人语音识别任务，解决快速切换和重叠语音下的说话人与时间联合建模问题。提出TellWhisper框架，结合时间与说话人信息，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.03712](https://arxiv.org/pdf/2601.03712)**

> **作者:** Yifan Hu; Peiji Yang; Zhisheng Wang; Yicheng Zhong; Rui Liu
>
> **备注:** 14 pages, 6 figures, 8 tables, accepted by ACL 2026 (Main)
>
> **摘要:** Multi-speaker automatic speech recognition (MASR) aims to predict ''who spoke when and what'' from multi-speaker speech, a key technology for multi-party dialogue understanding. However, most existing approaches decouple temporal modeling and speaker modeling when addressing ''when'' and ''who'': some inject speaker cues before encoding (e.g., speaker masking), which can cause irreversible information loss; others fuse identity by mixing speaker posteriors after encoding, which may entangle acoustic content with speaker identity. This separation is brittle under rapid turn-taking and overlapping speech, often leading to degraded performance. To address these limitations, we propose TellWhisper, a unified framework that jointly models speaker identity and temporal within the speech encoder. Specifically, we design TS-RoPE, a time-speaker rotary positional encoding: time coordinates are derived from frame indices, while speaker coordinates are derived from speaker activity and pause cues. By applying region-specific rotation angles, the model explicitly captures per-speaker continuity, speaker-turn transitions, and state dynamics, enabling the attention mechanism to simultaneously attend to ''when'' and ''who''. Moreover, to estimate frame-level speaker activity, we develop Hyper-SD, which casts speaker classification in hyperbolic space to enhance inter-class separation and refine speaker-activity estimates. Extensive experiments demonstrate the effectiveness of the proposed approach.
>
---
#### [replaced 006] A General Model for Deepfake Speech Detection: Diverse Bonafide Resources or Diverse AI-Based Generators
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于深度伪造语音检测任务，旨在解决模型泛化能力不足的问题。通过分析真实资源与生成器的影响，提出平衡数据集并验证其对模型性能的关键作用。**

- **链接: [https://arxiv.org/pdf/2603.27557](https://arxiv.org/pdf/2603.27557)**

> **作者:** Lam Pham; Khoi Vu; Dat Tran; David Fischinger; Alexander Schindler; Martin Boyer; Ian McLoughlin
>
> **摘要:** In this paper, we analyze two main factors of Bonafide Resource (BR) or AI-based Generator (AG) which affect the performance and the generality of a Deepfake Speech Detection (DSD) model. To this end, we first propose a deep-learning based model, referred to as the baseline. Then, we conducted experiments on the baseline by which we indicate how Bonafide Resource (BR) and AI-based Generator (AG) factors affect the threshold score used to detect fake or bonafide input audio in the inference process. Given the experimental results, a dataset, which re-uses public Deepfake Speech Detection (DSD) datasets and shows a balance between Bonafide Resource (BR) or AI-based Generator (AG), is proposed. We then train various deep-learning based models on the proposed dataset and conduct cross-dataset evaluation on different benchmark datasets. The cross-dataset evaluation results prove that the balance of Bonafide Resources (BR) and AI-based Generators (AG) is the key factor to train and achieve a general Deepfake Speech Detection (DSD) model.
>
---
#### [replaced 007] Audio-Visual Speech Enhancement: Architectural Design and Deployment Strategies
- **分类: cs.SD; eess.SP**

- **简介: 该论文属于音频-视觉语音增强任务，解决实时交互多媒体服务中的延迟与性能问题。设计并部署了基于云边协同的AVSE系统，分析网络负载与计算资源对系统的影响。**

- **链接: [https://arxiv.org/pdf/2508.08468](https://arxiv.org/pdf/2508.08468)**

> **作者:** Anis Hamadouche; Haifeng Luo; Mathini Sellathurai; Amir Hussain; Tharm Ratnarajah
>
> **摘要:** Real-time audio-visual speech enhancement (AVSE) is a key enabler for immersive and interactive multimedia services, yet its performance is tightly constrained by network latency, uplink capacity, and computational delay. This paper presents the design, deployment, and evaluation of a complete cloud-edge-assisted AVSE system operating over a public 5G edge network. The system integrates CNN-based acoustic enhancement and OpenCV-based facial feature extraction with an LSTM fusion network to preserve temporal coherence, and is deployed on a Vodafone-compatible AWS Wavelength edge cloud. Through extensive stress testing, we analyze end-to-end performance under varying network load and adaptive multimedia profiles. Results show that compute placement at the network edge is critical for meeting real-time coherence constraints, and that uplink capacity is often the dominant bottleneck for interactive AVSE services. Only 5G and wired Ethernet consistently satisfied the required communication delay bound for uncompressed audio-video chunks, while aggressive compression reduced payload sizes by up to 80% with negligible perceptual degradation, enabling robust operation under constrained conditions. We further demonstrate a fundamental trade-off between processing latency and enhancement quality, where reduced model complexity lowers delay but degrades reconstruction performance in low-SNR scenarios. Our findings indicate that public 5G edge environments can sustain real-time, interactive AVSE workloads when network and compute resources are carefully orchestrated, although performance margins remain tighter than in dedicated infrastructures. The architectural insights derived from this study provide practical guidelines for the design of delay-sensitive multimedia and perceptual enhancement services on emerging 5G edge-cloud platforms.
>
---
#### [replaced 008] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和缺乏交互纠错的问题。提出使用大模型进行语义评估，并设计交互式框架提升识别质量与互动能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
#### [replaced 009] animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics
- **分类: cs.SD; cs.AI; eess.AS; q-bio.QM; stat.AP**

- **简介: 该论文提出animal2vec模型和MeerKAT数据集，解决生物声学中罕见事件音频分析难题，通过自监督学习处理稀疏数据，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2406.01253](https://arxiv.org/pdf/2406.01253)**

> **作者:** Julian C. Schäfer-Zimmermann; Vlad Demartsev; Baptiste Averly; Kiran Dhanjal-Adams; Mathieu Duteil; Gabriella Gall; Marius Faiß; Lily Johnson-Ulrich; Dan Stowell; Marta B. Manser; Marie A. Roch; Ariana Strandburg-Peshkin
>
> **备注:** Code available at: this https URL | Dataset available at: this https URL
>
> **摘要:** Bioacoustic research, vital for understanding animal behavior, conservation, and ecology, faces a monumental challenge: analyzing vast datasets where animal vocalizations are rare. While deep learning techniques are becoming standard, adapting them to bioacoustics remains difficult. We address this with animal2vec, an interpretable large transformer model, and a self-supervised training scheme tailored for sparse and unbalanced bioacoustic data. It learns from unlabeled audio and then refines its understanding with labeled data. Furthermore, we introduce and publicly release MeerKAT: Meerkat Kalahari Audio Transcripts, a dataset of meerkat (Suricata suricatta) vocalizations with millisecond-resolution annotations, the largest labeled dataset on non-human terrestrial mammals currently available. Our model outperforms existing methods on MeerKAT and the publicly available NIPS4Bplus birdsong dataset. Moreover, animal2vec performs well even with limited labeled data (few-shot learning). animal2vec and MeerKAT provide a new reference point for bioacoustic research, enabling scientists to analyze large amounts of data even with scarce ground truth information.
>
---
#### [replaced 010] Gradient boundaries through confidence intervals for forced alignment estimates using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音信号处理任务，旨在解决强制对齐中边界估计不准确的问题。通过神经网络集成生成置信区间，得到更精确的边界估计和不确定性表示。**

- **链接: [https://arxiv.org/pdf/2506.01256](https://arxiv.org/pdf/2506.01256)**

> **作者:** Matthew C. Kelley
>
> **备注:** accepted for publication; 12 pages, 4 figures
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only point-estimates of boundaries. The present project introduces a method of producing gradient boundaries by deriving confidence intervals using neural network ensembles. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each classifier. The ensemble is then used to place the point-estimate of a boundary at the median of the boundaries in the ensemble, and the gradient range is placed using a 97.85% confidence interval around the median constructed using order statistics. Gradient boundaries are taken here as a more realistic representation of how segments transition into each other. Moreover, the range indicates the model uncertainty in the boundary placement, facilitating tasks like finding boundaries that should be reviewed. As a bonus, on the Buckeye and TIMIT corpora, the ensemble boundaries show a slight overall improvement over using just a single model. The gradient boundaries can be emitted during alignment as JSON files and a main table for programmatic and statistical analysis. For familiarity, they are also output as Praat TextGrids using a point tier to represent the edges of the boundary regions.
>
---
#### [replaced 011] PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决自动配音中的同步问题。通过引入音素同步方法，提升配音的唇形同步和语义保留效果。**

- **链接: [https://arxiv.org/pdf/2604.09111](https://arxiv.org/pdf/2604.09111)**

> **作者:** Changi Hong; Yoonah Song; Hwayoung Park; Chaewoon Bang; Dayeon Gu; Do Hyun Lee; Hong Kook Kim
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Recently, artificial intelligence-based dubbing technology has advanced, enabling automated dubbing (AD) to convert the source speech of a video into target speech in different languages. However, natural AD still faces synchronization challenges such as duration and lip-synchronization (lip-sync), which are crucial for preserving the viewer experience. Therefore, this paper proposes a synchronization method for AD processes that paraphrases translated text, comprising two steps: isochrony for timing constraints and phonetic synchronization (PS) to preserve lip-sync. First, we achieve isochrony by paraphrasing the translated text with a language model, ensuring the target speech duration matches that of the source speech. Second, we introduce PS, which employs dynamic time warping (DTW) with local costs of vowel distances measured from training data so that the target text composes vowels with pronunciations similar to source vowels. Third, we extend this approach to PSComet, which jointly considers semantic and phonetic similarity to preserve meaning better. The proposed methods are incorporated into text-to-speech systems, PS-TTS and PS-Comet TTS. The performance evaluation using Korean and English lip-reading datasets and a voice-actor dubbing dataset demonstrates that both systems outperform TTS without PS on several objective metrics and outperform voice actors in Korean-to-English and English-to-Korean dubbing. We extend the experiments to French, testing all pairs among these languages to evaluate cross-linguistic applicability. Across all language pairs, PS-Comet performed best, balancing lip-sync accuracy with semantic preservation, confirming that PS-Comet achieves more accurate lip-sync with semantic preservation than PS alone.
>
---
