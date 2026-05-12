# 音频 cs.SD;  eess.AS

- **最新发布 31 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Online Segmented Beamforming via Dynamic Programming
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决动态环境下的波束成形问题。针对传统方法在非平稳环境中的性能下降，提出在线分段波束成形算法，通过动态编程实现实时环境变化跟踪与协方差估计优化。**

- **链接: [https://arxiv.org/pdf/2605.08554](https://arxiv.org/pdf/2605.08554)**

> **作者:** Manan Mittal; Ryan M. Corey; Diego Cuji; John R. Buck; Andrew C. Singer
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** In dynamic acoustic environments characterized by time-varying interferers and moving sources, effective beamforming requires accurately identifying stationary regions over time. Traditional Capon beamformers rely on the instantaneous ensemble covariance matrix, which is inaccessible in practice. Practical implementations overcome this by estimating the sample covariance matrix (SCM) through averaging over a block of temporal samples. However, in non-stationary settings, a naive batch approach fails. Moving interferers smear the SCM, causing the beamformer to place nulls in outdated locations while failing to track newly active interferers, thereby degrading its nulling capabilities. To address this fundamental limitation, an Online Segmented Beamformer is proposed. This algorithm incorporates data-driven temporal segmentation to causally minimize output power while dynamically adapting the SCM estimation windows to local stationarity. By framing the problem through the lens of dynamic programming, the proposed method tracks abrupt environmental changes and resets covariance estimates in real-time. We validate the performance of this framework in a complex, reverberant simulated acoustic environment and in highly reverberant real world experiments, demonstrating its superiority over fixed-window adaptive methods.
>
---
#### [new 002] Reducing Linguistic Hallucination in LM-Based Speech Enhancement via Noise-Invariant Acoustic-Semantic Distillation
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决语言模型在噪声环境下生成不准确语言内容的问题。通过引入噪声不变的声学语义蒸馏框架，提升生成语音的语言一致性。**

- **链接: [https://arxiv.org/pdf/2605.08608](https://arxiv.org/pdf/2605.08608)**

> **作者:** Zheng Wang; Xiaobin Rong; Hang Su; Tianyi Tan; Junnan Wu; Lichun Fan; Zhenbo Luo; Jian Luan; Jing Lu
>
> **摘要:** Language model (LM)-based speech enhancement (SE) can generate natural-sounding speech, but under severe noise it often suffers from unreliable conditioning, leading to perceptually plausible yet linguistically incorrect outputs. To address this issue, we propose L3-SE, a noise-invariant acoustic-semantic distillation framework for reducing linguistic hallucination in LM-based SE. The proposed method learns a noise-invariant conditioning encoder from noisy speech by jointly distilling two complementary clean-speech targets: an acoustic target for reconstruction fidelity and a semantic target for linguistic consistency. The resulting noise-invariant acoustic-semantic representations are used to condition a decoder-only autoregressive language model, which predicts clean acoustic tokens that are decoded into enhanced speech. To support high-quality generation, we further employ a high-fidelity codec built on learnable weighted WavLM layer representations as the discrete acoustic interface. By improving the reliability of conditioning under adverse conditions, the proposed framework substantially reduces hallucination and improves content faithfulness. Experiments show that the proposed method consistently outperforms prior LM-based speech enhancement baselines on linguistic consistency metrics, with especially clear gains under low-SNR and reverberant conditions, while maintaining competitive perceptual quality. Audio samples are available at this https URL. The complete source code will be released after the manuscript is accepted.
>
---
#### [new 003] Single-Microphone Audio Point Source Discriminative Localization From Reverberation Late Tail Estimation
- **分类: eess.AS**

- **简介: 该论文属于音频源定位任务，旨在通过单麦克风估计声源位置。利用混响尾部信息，结合概率框架判断两信号是否来自同一位置。**

- **链接: [https://arxiv.org/pdf/2605.09627](https://arxiv.org/pdf/2605.09627)**

> **作者:** Matthew Maciejewski
>
> **备注:** Published at IEEE ICASSP 2026
>
> **摘要:** Location information can be a valuable signal for audio segmentation tasks, especially as a complement to methods focusing on the content or qualities of the sources. Though audio source localization is typically performed using the observations of the signal captured by multiple microphones in space, information about a source's location is captured by a single microphone through its arrival time and spectral amplitude--given the source's emitted signal is known. Since reverberation originates from the audio sources in a room, it accordingly contains some information about the emitted audio signals. The late-tail part of reverberation is relatively invariant to the local source and microphone geometry, depending primarily on only the room itself, and thus can provide the necessary reference information about audio signals that depends minimally on their location. In this work, we leverage the robust late-tail estimation of Weighted Prediction Error (WPE) dereverberation within a probabilistic framework to estimate the likelihood of two audio signals collected in the same room as having originated from the same location. We demonstrate the effectiveness of our approach on the speaker diarization task in both simulated and real environments.
>
---
#### [new 004] APEX: Audio Prototype EXplanations for Classification Tasks
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频分类的可解释性研究，解决音频领域XAI方法不足的问题。提出APEX框架，通过四种视角解释音频分类结果，无需调整原模型。**

- **链接: [https://arxiv.org/pdf/2605.10153](https://arxiv.org/pdf/2605.10153)**

> **作者:** Piotr Kawa; Kornel Howil; Piotr Borycki; Miłosz Adamczyk; Przemysław Spurek; Piotr Syga
>
> **摘要:** Explainable AI (XAI) has achieved remarkable success in image classification, yet the audio domain lacks equally mature solutions. Current methods apply vision-based attribution techniques to spectrograms, overlooking fundamental differences between visual and acoustic signals. While prototype reasoning is promising, acoustic similarity remains multidimensional. We introduce APEX (Audio Prototype EXplanations), a post-hoc framework for interpreting pre-trained audio classifiers. Crucially, APEX requires no fine-tuning of the original backbone and strictly preserves output invariance. APEX disentangles explanations into four perspectives: Square-based prototypes to localize transient events, Time-based for temporal patterns, Frequency-based highlighting spectral bands, and Time-Frequency-based integrating both. This yields intuitive, example-based explanations that respect acoustic properties, providing greater semantic clarity than standard gradient-based methods.
>
---
#### [new 005] DiffVQE: Hybrid Diffusion Voice Quality Enhancement Under Acoustic Echo and Noise
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决混响和噪声下的语音质量提升问题。提出DiffVQE模型，结合扩散方法实现更优的回声消除与降噪效果。**

- **链接: [https://arxiv.org/pdf/2605.08189](https://arxiv.org/pdf/2605.08189)**

> **作者:** Haljan Lugo Girao; Ernst Seidel; Pejman Mowlaee; Ziyue Zhao; Tim Fingscheidt
>
> **备注:** 6 pages, 4 figures, submitted to Interspeech 2026
>
> **摘要:** Acoustic echo and background noise pose challenges on speech enhancement in hands-free systems and speakerphones. Discriminatively trained end-to-end methods represent a powerful solution for joint acoustic echo control (AEC) and denoising. However, with the advent of generative methods, diffusion-based approaches have seen remarkable performance in speech enhancement tasks. In this work, to the best of our knowledge, we provide the first (still non-causal) diffusion-based AEC model (DiffVQE) that is reproducible in terms of topology, training data, and training framework. So far, without employing diffusion, Microsoft's discriminative DeepVQE model has been shown to excel any of the ICASSP 2023 AEC Challenge entries achieving remarkable performance. Using data from the Interspeech 2025 URGENT Challenge for a diverse, high-quality training dataset, our DiffVQE excels DeepVQE both in echo and noise control performance, as well as in computational complexity and model size.
>
---
#### [new 006] Latent Secret Spin: Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces
- **分类: eess.AS**

- **简介: 该论文属于语音水印任务，旨在实现盲水印嵌入与检测。提出LSS方法，通过潜在空间的正交旋转生成不可感知的水印签名，无需训练，抗干扰性强。**

- **链接: [https://arxiv.org/pdf/2605.08431](https://arxiv.org/pdf/2605.08431)**

> **作者:** Emma Coletta; Massimiliano Todisco; Michele Panariello; Antonio Faonio; Nicholas Evans
>
> **摘要:** We introduce Latent Secret Spin (LSS), a blind speech watermarking method based on geometric operations in codec latent space. Based upon orthogonal rotations to principal components, LSS induces imperceptible but detectable covariance signatures according to a pseudo-random watermarking schedule. The scheme generalises across datasets, preserves perceptual quality and, unlike some learned, neural watermarking schemes, it does not require neural network training, is resistant to common signal manipulations and is flexible to payload size. Analyses show that structured latent-space watermarking is a promising and interpretable alternative to existing approaches.
>
---
#### [new 007] Drum Synthesis from Expressive Drum Grids via Neural Audio Codecs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在将符号化鼓谱转换为真实鼓音频。通过Transformer模型预测神经音频编码器的离散代码，实现高质量鼓音合成。**

- **链接: [https://arxiv.org/pdf/2605.10281](https://arxiv.org/pdf/2605.10281)**

> **作者:** Konstantinos Soiledis; Maximos Kaliakatsos-Papakostas; Dimos Makris; Konstantinos Tsamis
>
> **摘要:** Generating realistic drum audio directly from symbolic representations is a challenging task at the intersection of music perception and machine learning. We propose a system that transforms an expressive drum grid, a time-aligned MIDI representation with microtiming and velocity information, into drum audio by predicting discrete codes of a neural audio codec. Our approach uses a Transformer-based model to map the drum grid input to a sequence of codec tokens, which are then converted to waveform audio via a pre-trained codec decoder. We experiment with multiple state-of-the-art neural codecs, namely EnCodec, DAC, and X-Codec, to assess how the choice of audio representation impacts the quality of the generated drums. The system is trained and evaluated on the Expanded Groove MIDI Dataset, E-GMD, a large collection of human drum performances with paired MIDI and audio. We evaluate the fidelity and musical alignment of the generated audio using objective metrics. Overall, our results establish codec-token prediction as an effective route for drum grid-to-audio generation and provide practical insights into selecting audio tokenizers for percussive synthesis.
>
---
#### [new 008] PoDAR: Power-Disentangled Audio Representation for Generative Modeling
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出PoDAR框架，解决音频生成中潜在空间建模问题。通过分离信号功率与语义内容，提升生成模型效率与质量。属于音频生成任务。**

- **链接: [https://arxiv.org/pdf/2605.10084](https://arxiv.org/pdf/2605.10084)**

> **作者:** Alejandro Luebs; Mithilesh Vaidya; Ishaan Kumar; Sumukh Badam; Stephen W. Bailey; Matthew Bendel; Jose Sotelo; Xingzhe He
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** The performance of audio latent diffusion models is primarily governed by generator expressivity and the modelability of the underlying latent space. While recent research has focused primarily on the former, as well as improving the reconstruction fidelity of audio codecs, we demonstrate that latent modelability can be significantly improved through explicit factor disentanglement. We present PoDAR (Power-Disentangled Audio Representation), a framework that utilizes a randomized power augmentation and latent consistency objective to decouple signal power from invariant semantic content. This factorization makes the latent space easier to model, which both accelerates the convergence of downstream generative models and improves final overall performance. When applied to a Stable Audio 1.0 VAE with an F5-TTS generator, PoDAR achieves about a $2\times$ acceleration in convergence to match baseline performance, while increasing final speaker similarity by 0.055 and UTMOS by 0.22 on the LibriSpeech-PC dataset. Furthermore, isolating power into dedicated channels enables the application of CFG exclusively to power-invariant content, effectively extending the stable guidance regime to higher scales.
>
---
#### [new 009] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐编辑任务，解决多音轨音色迁移问题。提出Polyphonia框架，通过声学引导注意力校准，实现精准音色转换并保留非目标音轨。**

- **链接: [https://arxiv.org/pdf/2605.10203](https://arxiv.org/pdf/2605.10203)**

> **作者:** Haowen Li; Tianxiang Li; Yi Yang; Boyu Cao; Qi Liu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** The advancement of diffusion-based text-to-music generation has opened new avenues for zero-shot music editing. However, existing methods fail to achieve stem-specific timbre transfer, which requires altering specific stems while strictly preserving the background accompaniment. This limitation severely hinders practical application, since real-world production necessitates precise manipulation of components within dense mixtures. Our key finding is that, while vanilla cross-attention captures semantic features of stems, it lacks the spectral resolution to strictly localize targets in dense mixtures, leading to boundary leakage. To resolve this dilemma, we propose Polyphonia, a zero-shot editing framework with Acoustic-Informed Attention Calibration. Rather than relying solely on diffuse semantic attention, Polyphonia leverages a probabilistic acoustic prior to establish coarse boundaries, enabling non-target stems preserved precise semantic synthesis. For evaluation, we propose PolyEvalPrompts, a standardized prompt set with 1,170 timbre transfer tasks in polyphonic music. Specifically, Polyphonia achieves an increase of 15.5% in target alignment compared to baselines, while maintaining competitive music fidelity and non-target integrity.
>
---
#### [new 010] Multi-layer attentive probing improves transfer of audio representations for bioacoustics
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于生物声学任务，解决音频表示迁移问题。通过多层注意力探测提升下游任务性能，验证了传统单层探测的局限性。**

- **链接: [https://arxiv.org/pdf/2605.10494](https://arxiv.org/pdf/2605.10494)**

> **作者:** Marius Miron; David Robinson; Masato Hagiwara; Titouan Parcollet; Jules Cauzinille; Gagan Narula; Milad Alizadeh; Ellen Gilsenan-McMahon; Sara Keen; Emmanuel Chemla; Benjamin Hoffman; Maddie Cusimano; Diane Kim; Felix Effenberger; Jane K. Lawton; Aza Raskin; Olivier Pietquin; Matthieu Geist
>
> **摘要:** Probing heads map the representations learned from audio by a machine learning model to downstream task labels and are a key component in evaluating representation learning. Most bioacoustic benchmarks use a fixed, low-capacity probe, such as a linear layer on the final encoder layer. While this standardization enables model comparisons, it may bias results by overlooking the interaction between encoder features and probe design. In this work, we systematically study different probing strategies across two bioacoustic benchmarks, BEANs and BirdSet. We evaluate last- and multi-layer probing, across linear and attention probes. We show that larger probe heads that leverage time information have superior performance. Our results suggest that current benchmarks may misrepresent encoder quality when relying on a last-layer probing setup. Multi-layer probing improves downstream task performance across all tested models, while attention probing has superior performance to linear probing for transformer models.
>
---
#### [new 011] SF-Flow: Sound field magnitude estimation via flow matching guided by sparse measurements
- **分类: eess.AS**

- **简介: 该论文属于3D声场重建任务，解决从稀疏麦克风测量中恢复声场幅度的问题。提出SF-Flow框架，利用流匹配和U-Net进行高效准确的重建。**

- **链接: [https://arxiv.org/pdf/2605.10398](https://arxiv.org/pdf/2605.10398)**

> **作者:** Ege Erdem; Shoichi Koyama; Tomohiko Nakamura; Orchisama Das; Zoran Cvetković
>
> **摘要:** Reconstructing a 3D sound field from sparse microphone measurements is a fundamental yet ill-posed problem, which we address through Acoustic Transfer Function (ATF) magnitude estimation. ATF magnitude encapsulates key perceptual and acoustic properties of a physical space with applications in room characterization and correction. Although recent generative paradigms such as Flow Matching (FM) have achieved state-of-the-art performance in speech and music generation, their potential in spatial audio remains underexplored. We propose a novel framework for 3D ATF magnitude reconstruction as a guided generation task, with a 3D U-Net conditioned by a permutation-invariant set encoder. This architecture enables reconstruction from an arbitrary number of sparse inputs while leveraging the stable and efficient training properties of FM. Experimental results demonstrate that SF-Flow achieves accurate reconstruction up to \SI{1}{kHz}, trains substantially faster than the autoencoder baseline, and improves significantly with dataset size.
>
---
#### [new 012] A Cold Diffusion Approach for Percussive Dereverberation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频去混响任务，旨在解决鼓声去混响问题。提出冷扩散框架，通过建模混响为确定性退化过程，提升鼓声信号清晰度。**

- **链接: [https://arxiv.org/pdf/2605.10256](https://arxiv.org/pdf/2605.10256)**

> **作者:** Dimos Makris; András Barják; Maximos Kaliakatsos-Papakostas
>
> **备注:** Accepted for the 2026 IEEE World Congress on Computational Intelligence, IJCNN Track, 21-26 June 2026, Maastricht, the Netherlands
>
> **摘要:** Most recent advances in audio dereverberation focus almost exclusively on speech, leaving percussive and drum signals largely unexplored despite their importance in music production. Percussive dereverberation poses distinct challenges due to sharp transients and dense temporal structure. In this work, we propose a cold diffusion framework for dereverberating stereo drum stems (downmixes), modeling reverberation as a deterministic degradation process that progressively transforms anechoic signals into reverberant ones. We investigate two reverse-process parameterizations, Direct (next-state) and a Delta-normalized residual (velocity-style) prediction, and implement the framework using both a UNet and a diffusion Transformer backbone. The models are trained and evaluated on curated datasets comprising both acoustic and electronic drum recordings, with reverberation generated using a combination of synthetic and real room impulse responses. Extensive experiments on in-domain and fully out-of-domain test sets demonstrate that the proposed method consistently outperforms strong score-based and conditional diffusion baselines, evaluated using signal-based and perceptual metrics tailored to percussive audio.
>
---
#### [new 013] Remix the Timbre: Diffusion-Based Style Transfer Across Polyphonic Stems
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐信号处理任务，解决多乐器音色迁移问题。针对现有方法存在分离误差和音色不连贯的问题，提出MixtureTT系统，通过联合扩散过程实现多音轨音色迁移，提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.09259](https://arxiv.org/pdf/2605.09259)**

> **作者:** Leduo Chen; Junchuan Zhao; Shengchen Li
>
> **摘要:** Timbre transfer aims to modify the timbral identity of a musical recording while preserving the original melody and rhythm. While single-instrument timbre transfer has made substantial progress, existing approaches to multi-instrument settings rely on separate-then-transfer pipelines that propagate source separation artifacts and produce incoherent synthesized timbres across stems. This paper proposes MixtureTT, to the best of our knowledge the first system for flexible per-stem timbre transfer directly from a polyphonic mixture. Given a mixture and a separate timbre reference for each target voice, MixtureTT jointly transfers all stems to the specified instruments through a shared diffusion process. Modeling the dependencies across the per-stem content and cross-stem harmonic, the proposed joint stem diffusion transformer eliminates cascaded separation error, reduces inference cost by a factor equal to the number of stems, and yields more coherent multi-stem outputs. Despite operating under a strictly harder input condition, evaluations on the SATB choral dataset show that MixtureTT outperforms single-instrument baselines on both objective and subjective metrics demonstrating the necessity of dedicated multi-instrument timbre transfer over the naive separate-then-transfer pipelines. As a result, this work confirms that the cross-stem modeling is essential for mixture-level timbre transfer as the proposed joint setting consistently exceeds an equivalent single-stem ablation.
>
---
#### [new 014] RADAR Challenge 2026: Robust Audio Deepfake Recognition under Media Transformations
- **分类: eess.AS**

- **简介: 该论文介绍RADAR Challenge 2026，旨在解决多语言和媒体变换下的音频深度伪造检测问题，通过构建数据集和评估协议，测试系统在真实环境下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.09568](https://arxiv.org/pdf/2605.09568)**

> **作者:** Hieu-Thi Luong; Xuechen Liu; Ivan Kukanov; Zheng Xin Chai; Kong Aik Lee
>
> **备注:** Submitted to APSIPA 2026
>
> **摘要:** RADAR Challenge 2026 is an APSIPA Grand Challenge on Robust Audio Deepfake Recognition under Media Transformations, designed to simulate realistic media conditions in real-world audio distribution pipelines, including compression, resampling, noise, and reverberation. It consists of two phases: an English development phase with labeled data for analysis and paper writing, and a multilingual evaluation phase containing more than 100,000 utterances in English, Singapore English, Mandarin Chinese, Taiwanese Mandarin, Japanese, and Vietnamese. Systems are evaluated using equal error rate (EER) for binary real/fake classification. This paper describes the challenge task, the construction of the data set, the evaluation protocol, and the overall results. During the challenge, 33 teams submitted to the development phase and 22 teams submitted to the final evaluation phase. The reported results highlight the remaining challenges of robust audio deepfake detection under multilingual and media-transformed conditions.
>
---
#### [new 015] ShipEcho -- An Interactive Tool for Global Mapping of Underwater Radiated Noise from Vessels
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 论文介绍ShipEcho系统，用于全球船舶辐射噪声映射。解决噪声监测数据稀疏和成本高的问题，通过AIS数据实现近实时噪声地图生成，支持环境评估与管理决策。**

- **链接: [https://arxiv.org/pdf/2605.08194](https://arxiv.org/pdf/2605.08194)**

> **作者:** Mark Shipton; Valentino Denona; Đula Nađ; Roee Diamant
>
> **备注:** 34 pages
>
> **摘要:** Underwater radiated noise from vessels (V-URN) is a recognized environmental stressor that negatively impacts marine ecosystems. Significant resources are invested in the development of V-URN monitoring indicators, regulatory frameworks, and management-oriented assessments. One approach with high potential for impact is V-URN mapping, which can provide actionable spatiotemporal information for environmental assessment and mitigation planning. Producing management-scale maps remains challenging as passive acoustic measurements are spatially sparse and many operational systems depend on specialist workflows and costly access to wide-area vessel activity data. To address these constraints, we introduce ShipEcho, a freely accessible web-based Geographic Information System (GIS) that provides near-real-time V-URN mapping using vessel data acquired through a community-based AIS exchange. Using established vessel SL models and propagation modeling informed by bathymetric data, ShipEcho produces near-real-time and cumulative noise maps across regions worldwide. These include sound pressure levels and sound exposure levels using standard indicators, including the 63~Hz and 125~Hz one-third octave bands and a 20--2000~Hz broadband level. We describe the system architecture, data pipeline, modeling workflow, and key assumptions, and evaluate map accuracy through comparison with acoustic recordings. We then demonstrate how ShipEcho can support management-level assessment, decision-making, and policy initiatives through practical use cases.
>
---
#### [new 016] Omni-DeepSearch: A Benchmark for Audio-Driven Omni-Modal Deep Search
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出Omni-DeepSearch，解决音频驱动的多模态深度搜索任务，通过音频启动跨模态检索与推理，提升多模态模型能力。**

- **链接: [https://arxiv.org/pdf/2605.08762](https://arxiv.org/pdf/2605.08762)**

> **作者:** Tao Yu; yiming ding; Shenghua Chai; Minghui Zhang; Zhongtian Luo; Xinming Wang; Xinlong Chen; Zhaolu Kang; Junhao Gong; Yuxuan Zhou; Haopeng Jin; Zhiqing Cui; Jiabing Yang; YiFan Zhang; Hongzhu Yi; Zheqi He; Xi Yang; Yan Huang; Liang Wang
>
> **备注:** 43 pages
>
> **摘要:** Current omni-modal benchmarks mainly evaluate models under settings where multiple modalities are provided simultaneously, while the ability to start from audio alone and actively search for cross-modal evidence remains underexplored. In this paper, we introduce \textbf{Omni-DeepSearch}, a benchmark for audio-driven omni-modal deep search. Given one or more audio clips and a related question, models must infer useful clues from audio, invoke text, image, and video search tools, and perform multi-hop reasoning to produce a short, objective, and verifiable answer. Omni-DeepSearch contains 640 samples across 15 fine-grained categories, covering four retrieval target modalities and four audio content types. A multi-stage filtering pipeline ensures audio dependence, retrieval necessity, visual modality necessity, and answer uniqueness. Experiments on recent closed-source and open-source omni-modal models show that this task remains highly challenging: the strongest evaluated model, Gemini-3-Pro, achieves only 43.44\% average accuracy. Further analyses illustrate key bottlenecks in audio entity inference, query formulation, tool-use reliability, multi-hop retrieval, and cross-modal verification. These results highlight audio-driven omni-modal deep search as an important and underexplored direction for future multimodal agents.
>
---
#### [new 017] Rethinking Entropy Minimization in Test-Time Adaptation for Autoregressive Models
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文属于测试时自适应任务，旨在解决熵最小化在自回归生成模型中的理论碎片化问题，提出统一的数学框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.08186](https://arxiv.org/pdf/2605.08186)**

> **作者:** Wei-Ping Huang; Chee-En Yu; Guan-Ting Lin; Hung-yi Lee
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Test-Time Adaptation (TTA) via entropy minimization (EM) has proven effective for classification tasks, yet its application to generative autoregressive models remains theoretically fragmented. Existing approaches typically rely on distinct heuristics, such as teacher forcing with pseudo labels or policy-gradient-based reinforcement learning, without a unified mathematical foundation. In this work, we resolve this discrepancy by deriving a rigorous formulation of EM tailored to autoregressive models. We show that the exact objective naturally decomposes into a token-level policy gradient loss and a token-level entropy loss, and we reinterpret prior methods as partial realizations of this unified formulation. Using Whisper ASR as a testbed, we demonstrate that our approach consistently improves performance across more than 20 diverse domains, including acoustic noise, accents, and multilingual settings.
>
---
#### [new 018] Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文针对零样本文本到语音任务，解决MI-DFM中的调度优化与路径误差问题，提出GibbsTTS方法提升语音自然度与说话人相似度。**

- **链接: [https://arxiv.org/pdf/2605.09386](https://arxiv.org/pdf/2605.09386)**

> **作者:** Dong Yang; Yiyi Cai; Haoyu Zhang; Yuki Saito; Hiroshi Saruwatari
>
> **备注:** Under Review
>
> **摘要:** Metric-induced discrete flow matching (MI-DFM) exploits token-latent geometry for discrete generation, but its practical use is limited by two issues: heuristic schedulers requiring hyperparameter search, and finite-step path-tracking error from its first-order continuous-time Markov chain (CTMC) solver. We address both issues. First, we derive a kinetic-optimal scheduler for prescribed scalar-parameterized probability paths, and instantiate it for MI-DFM as a training-free numerical schedule that traverses the path at constant Fisher-Rao speed. Second, we introduce a finite-step moment correction that adjusts the jump probability while preserving the CTMC jump destination distribution. We validate the resulting method, GibbsTTS, on codec-based zero-shot text-to-speech (TTS). Under controlled comparisons with a unified architecture and large-scale dataset, GibbsTTS achieves the best objective naturalness and is preferred in subjective evaluations over masked discrete generative baselines. Additionally, in comparison with the evaluated state-of-the-art TTS systems, GibbsTTS shows strong speaker similarity, achieving the highest similarity on three of four test sets and ranking second on the fourth. Project page: this https URL
>
---
#### [new 019] Bangla-WhisperDiar: Fine-Tuning Whisper and PyAnnote for Bangla Long-Form Speech Recognition and Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对孟加拉语长语音的自动语音识别和说话人二值化任务，通过微调Whisper和PyAnnote模型，提升识别与分 speaker 的准确性。**

- **链接: [https://arxiv.org/pdf/2605.08214](https://arxiv.org/pdf/2605.08214)**

> **作者:** Mohammed Aman Bhuiyan; Md Sazzad Hossain Adib; Samiul Basir Bhuiyan; Amit Chakraborty; Aritra Islam Saswato; Ahmed Faizul Haque Dhrubo; Mohammad Ashrafuzzaman Khan
>
> **备注:** 3 figures and 5 tables
>
> **摘要:** Automatic Speech Recognition (ASR) and speaker diarization in Bangla remain challenging due to long form recordings, diverse acoustic conditions, and significant speaker variability. This work addresses these two core tasks in Bangla spoken language understanding by developing robust systems for long form ASR and speaker diarization. For ASR (Problem 1), we fine tune the tugstugi bengaliai regional asr whisper medium model on a custom-curated dataset of approximately 15,000 chunked and aligned Bangla audio segments, employing full weight training with extensive data augmentation including noise injection, reverb simulation, echo, clipping distortion, and pitch/time perturbation. For speaker diarization (Problem 2), we fine-tune the pyannote/segmentation-3.0 model using PyTorch Lightning on the competition annotated diarization dataset, swapping the fine-tuned segmentation backbone into the pyannote/speaker-diarization-community-1 pipeline while retaining the pretrained speaker embedding and clustering components. Our ASR system achieves a Word Error Rate (WER) of 0.2441, while our diarization system achieves a Diarization Error Rate (DER) of 0.2392, both evaluated on the test set, demonstrating notable improvements over the respective pretrained baselines. We describe our complete pipeline, including data preprocessing, text normalization, audio augmentation, training strategies, inference optimization, and post-processing for both tasks.
>
---
#### [new 020] Towards Trustworthy Audio Deepfake Detection: A Systematic Framework for Diagnosing and Mitigating Gender Bias
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决性别偏差问题。通过诊断偏差来源并提出针对性缓解策略，提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2605.09087](https://arxiv.org/pdf/2605.09087)**

> **作者:** Aishwarya Fursule; Shruti Kshirsagar; Anderson R. Avila
>
> **备注:** Submitted to SMC 2026 conference
>
> **摘要:** Audio deepfake detection systems are increasingly deployed in high-stakes security applications, yet their fairness across demographic groups remains critically underexamined. Prior work measures gender disparity but does not investigate where it comes from or how to fix it systematically. We present the first diagnosis-first framework that identifies bias source before applying targeted mitigation, evaluated on two models, AASIST and Wav2Vec2+ResNet18, on ASVSpoof5. Our diagnosis shows that bias does not stem from imbalanced training data but from acoustic representation differences, gender leakage in learned features, and structural evaluation asymmetry. We test mitigation strategies across in-processing, post-processing and combined families, including novel methods introduced in this work. Adjusting the decision threshold separately per gender reduces unfairness by 54% to 75% at no cost to detection accuracy, and our new epoch-level fairness regularisation method outperforms existing per-batch approaches. Adversarial debiasing succeeds only when gender leakage is localised, and fails when it is diffuse, an outcome correctly predicted by our diagnosis before training. No single method fully closes the fairness gap, confirming that bias sources must be identified before fixes are applied and that fairer benchmark design is equally important
>
---
#### [new 021] Low-Cost Detection of Degraded Voice Clones via Source-Output Acoustic Consistency
- **分类: eess.AS**

- **简介: 该论文属于语音合成质量检测任务，旨在解决 degraded voice clones 的快速识别问题。通过分析源-输出声学一致性特征，如 f0、VTL 和 HNR，实现轻量级检测。**

- **链接: [https://arxiv.org/pdf/2605.08165](https://arxiv.org/pdf/2605.08165)**

> **作者:** Jana Shokr; Minos Papadopoulos; Jeremy Cooperstock; Pavo Orepic
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Recent advances in generative speech have increased the need for automatic detection of obviously failed synthetic outputs. This is particularly important in clinical settings such as AVATAR therapy, in which schizophrenia patients engage with a computer-generated representation of their hallucinated voices and degraded synthesis may disrupt immersion and therapeutic engagement. We investigate whether low-dimensional, interpretable source-output acoustic features can provide a lightweight first-pass detector of degraded voice-cloning outputs. Motivated by source-filter models of speech, we first test median fundamental frequency (f0) as a source-related consistency measure, and compare it with vocal tract length (VTL) as a filter-related measure and Harmonics-to-Noise Ratio (HNR) as a noise-related descriptor. Human-labeled voice-cloning samples generated with two vocoder families, WaveRNN (n=54) and HiFi-GAN (n=40), were evaluated using an asymmetric thresholding procedure in the input-output feature space. For WaveRNN, f0 and HNR both achieved 85.2% accuracy, outperforming VTL (64.8%). For HiFi-GAN, HNR achieved 80.0% accuracy, followed by f0 at 77.5% and VTL at 67.5%. Sample-level overlap and spectrographic inspection showed that f0 and HNR capture partly distinct failure patterns, rather than providing redundant rankings of the same samples. These results show that simple source-output acoustic consistency measures can provide useful first-pass detection of degraded voice clones, and support the use of interpretable threshold-based screening in applications where failed synthetic speech must be rejected quickly.
>
---
#### [new 022] ChladniSonify: A Visual-Acoustic Mapping Method for Chladni Patterns in New Media Art Creation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频可视化任务，旨在解决Chladni图案与声音映射的实时性与准确性问题。通过构建CNN分类模型和频率映射系统，实现高精度、低延迟的音画同步。**

- **链接: [https://arxiv.org/pdf/2605.09846](https://arxiv.org/pdf/2605.09846)**

> **作者:** Yakun Liu; Hai Luan; Dong Liu; Zhiyu Jin
>
> **备注:** 9 pages, 5 figures, IEEE conference format
>
> **摘要:** In new media art creation, the mapping between vision and hearing is often subjective. As a classic carrier of sound visualization, Chladni patterns have great potential in building audio-visual mapping mechanisms. However, existing tools face pain points: high technical barriers for simulation, offline computing failing real-time interaction, and uncontrollable mapping rules in general sonification tools. To address these, this paper proposes ChladniSonify, a real-time visual-acoustic mapping method for Chladni patterns. Based on Kirchhoff-Love plate theory, we build a paired dataset via numerical programming and calibrate it using ANSYS finite element simulation. Focusing on the slender nodal lines of Chladni patterns, we adopt a lightweight CNN with CBAM to achieve high-precision, low-latency pattern classification. Finally, we build an end-to-end system in Python and Max/MSP, mapping recognized patterns to corresponding sine wave frequencies. Results show the system has excellent usability: the classification module achieves 99.33% accuracy on the test set with 7.03 ms inference latency; the mapped frequency matches the theoretical value with zero deviation; the average end-to-end latency is under 50 ms, meeting real-time interactive needs. This work provides a reproducible engineering prototype for Chladni audio-visual art creation.
>
---
#### [new 023] Evaluating the Expressive Appropriateness of Speech in Rich Contexts
- **分类: eess.AS**

- **简介: 该论文属于语音表达评估任务，解决现有方法忽略语境适配性的问题。构建了CEAEval-D数据集，并提出CEAEval-M模型，提升语音表达在丰富语境中的评估效果。**

- **链接: [https://arxiv.org/pdf/2605.09413](https://arxiv.org/pdf/2605.09413)**

> **作者:** Tianrui Wang; Ziyang Ma; Yizhou Peng; Haoyu Wang; Zhikang Niu; Zikang Huang; Yihao Wu; Yi-Wen Chao; Yu Jiang; Yuheng Lu; Guanrou Yang; Xuanchen Li; Hexin Liu; Chunyu Qiang; Cheng Gong; Yifan Yang; Tianchi Liu; Junyu Wang; Nana Hou; Meng Ge; Fuming You; Wei Yang; Zhongqian Sun; Haifeng Hu; Xiaobao Wang; Eng Siong Chng; Xie Chen; Longbiao Wang; Jianwu Dang
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Evaluating expressive speech remains challenging, as existing methods mainly assess emotional intensity and overlook whether a speech sample is expressively appropriate for its contextual setting. This limitation hinders reliable evaluation of speech systems used in narrative-driven and interactive applications, such as audiobooks and conversational agents. We introduce CEAEval, a Context-rich framework for Evaluating Expressive Appropriateness in speech, which assesses whether a speech sample expressively aligns with the underlying communicative intent implied by its discourse-level narrative context. To support this task, we construct CEAEval-D, the first context-rich speech dataset with real human performances in Mandarin conversational speech, providing narrative descriptions together with fifteen dimensions of human annotations covering expressive attributes and expressive appropriateness. We further develop CEAEval-M, a model that integrates knowledge distillation, planner-based multi-model collaboration, adaptive audio attention bias, and reinforcement learning to perform context-rich expressive appropriateness evaluation. Experiments on a human-annotated test set demonstrate that CEAEval-M substantially outperforms existing speech evaluation and analysis systems.
>
---
#### [new 024] Reddit2Deezer: A Scalable Dataset for Real-World Grounded Conversational Music Recommendation
- **分类: cs.IR; cs.SD**

- **简介: 该论文属于对话音乐推荐任务，解决真实对话数据不足与合成数据不自然的矛盾。工作是构建Reddit2Deezer数据集，包含真实对话和音乐实体链接，提升推荐研究的实用性与可复现性。**

- **链接: [https://arxiv.org/pdf/2605.09120](https://arxiv.org/pdf/2605.09120)**

> **作者:** Haven Kim; Julian McAuley
>
> **摘要:** Conversational music recommendation (CMR) research currently faces a tradeoff between authentic dialogue corpora that are limited in scale and synthesized corpora that scale up but whose conversations are artificially constructed rather than naturally observed. In this paper, we introduce Reddit2Deezer, a reality-grounded CMR resource derived from 190k unique {thread, leaf-comment} pairs. We release the resource in two versions: a raw version that preserves authenticity, and a paraphrased version that maximizes long-term reproducibility. Each musical entity is linked to a Deezer identifier, which provides straightforward access to audio previews and rich metadata (e.g., genre tags, popularity, BPM), opening the door to future research on content-grounded conversational recommendation. A human validation confirms the quality of the dialogues, item grounding, and paraphrases. The dataset is available at this https URL.
>
---
#### [new 025] Probing Cross-modal Information Hubs in Audio-Visual LLMs
- **分类: cs.AI; eess.AS**

- **简介: 该论文研究音频-视觉大语言模型中的跨模态信息流动，旨在揭示信息如何在不同模态间编码。任务属于多模态模型分析，解决跨模态信息存储机制问题，发现并利用跨模态汇点令牌提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.10815](https://arxiv.org/pdf/2605.10815)**

> **作者:** Jihoo Jung; Chaeyoung Jung; Ji-Hoon Kim; Joon Son Chung
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Audio-visual large language models (AVLLMs) have recently emerged as a powerful architecture capable of jointly reasoning over audio, visual, and textual modalities. In AVLLMs, the bidirectional interaction between audio and video modalities introduces intricate processing dynamics, necessitating a deeper understanding of their internal mechanisms. However, unlike extensively studied text-only or large vision language models, the internal workings of AVLLMs remain largely unexplored. In this paper, we focus on cross-modal information flow between audio and visual modalities in AVLLMs, investigating where information derived from one modality is encoded within the token representations of the other modality. Through an analysis of multiple recent AVLLMs, we uncover two common findings. First, AVLLMs primarily encode integrated audio-visual information in sink tokens. Second, sink tokens do not uniformly hold cross-modal information. Instead, a distinct subset of sink tokens, which we term cross-modal sink tokens, specializes in storing such information. Based on these findings, we further propose a simple training-free hallucination mitigation method by encouraging reliance on integrated cross-modal information within cross-modal sink tokens. Our code is available at this https URL.
>
---
#### [new 026] Separate First, Fuse Later: Mitigating Cross-Modal Interference in Audio-Visual LLMs Reasoning with Modality-Specific Chain-of-Thought
- **分类: cs.AI; cs.SD**

- **简介: 该论文属于音频-视觉问答任务，旨在解决跨模态干扰问题。通过分离模态推理并后期融合，提升模型准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.09906](https://arxiv.org/pdf/2605.09906)**

> **作者:** Xuanchen Li; Yuheng Lu; Chenrui Cui; Tianrui Wang; Zikang Huang; Yu Jiang; Long Zhou; Longbiao Wang; Jianwu Dang
>
> **摘要:** Audio and vision provide complementary evidence for audio-visual question answering, yet current audio-visual large language models may suffer from cross-modal interference: information from one modality misguides the interpretation of another, thereby inducing hallucinations. We attribute this issue to uncontrolled cross-modal interactions during intermediate reasoning. To mitigate this, we propose Separate First, Fuse Later (SFFL), an audio-visual reasoning framework designed to reduce cross-modal interference. SFFL enforces modality-specific chain-of-thought reasoning, producing separate audio and visual reasoning traces and integrating evidence for answering. We construct modality-preference labels via a data pipeline under different modality input settings. We use these labels as an auxiliary reward in reinforcement learning to encourage a instance-dependent preference for modality cues when answering. We further introduce a modality-specific reasoning mechanism that preserves modality isolation during the separated reasoning stage while enabling full access to cross-modal information at the evidence fusion stage. Experiments demonstrate consistent improvements in both accuracy and robustness, yielding an average relative gain of 5.16\% on general AVQA benchmarks and 11.17\% on a cross-modal hallucination benchmark.
>
---
#### [new 027] Voice Biomarkers for Depression and Anxiety
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于心理健康评估任务，旨在通过语音识别抑郁和焦虑。使用深度学习模型处理原始语音信号，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.09908](https://arxiv.org/pdf/2605.09908)**

> **作者:** Oleksii Abramenko; Noah D. Stein; Colin Vaz
>
> **摘要:** Current approaches to detecting depression and anxiety from speech primarily rely on machine learning techniques that utilize hand-engineered paralinguistic features and related acoustic descriptors derived from time- and frequency-domain representations of speech signals. Applying deep learning methods directly to raw speech signals has the potential to produce biomarker representations with substantially greater predictive power. However, these approaches typically require large volumes of carefully annotated data to learn robust and clinically meaningful representations of the underlying biomarkers. In this paper, we describe our efforts toward developing a deep learning model trained on a large-scale proprietary dataset comprising ~65,000 utterances collected from more than 23,000 subjects representative of relevant United States demographics. We present the techniques employed and analyze their impact on model performance. Our results demonstrate that the proposed models can extract content-agnostic biomarker information, which, when combined with lexical features extracted from audio, yields improved predictive performance in production settings. Our models are evaluated on ~5000 unique subjects and achieve performance of 71% in terms of sensitivity and specificity. To foster further research in mental health assessment from speech, we release the best-performing model described in this paper on HuggingFace.
>
---
#### [new 028] Unison: Harmonizing Motion, Speech, and Sound for Human-Centric Audio-Video Generation
- **分类: cs.CV; cs.GR; cs.MM; cs.SD**

- **简介: 该论文属于音频视频生成任务，旨在解决运动、语音和音效在时间上不一致的问题。通过提出Unison框架，实现多模态的协同与同步。**

- **链接: [https://arxiv.org/pdf/2605.08729](https://arxiv.org/pdf/2605.08729)**

> **作者:** Shihao Cheng; Jiaxu Zhang; Quanyue Song; Shansong Liu; Zhizhi Guo; Xiaolei Zhang; Chi Zhang; Xuelong Li; Zhigang Tu
>
> **摘要:** Motion, speech, and sound effects are fundamental elements of human-centric videos, yet their heterogeneous temporal characteristics make joint generation highly challenging. Existing audio-video generation models often fail to maintain consistent alignment across these modalities, leading to noticeable mismatches between motion, speech, and environmental sounds. We present Unison, a unified framework that explicitly promotes coherence across the motion, speech, and sound modalities. Within the audio stream, Unison employs a semantic-guided harmonization strategy that decouples the generation of speech and sound-effect components. Leveraging bidirectional audio cross-attention and semantic-conditioned gating for semantic-driven adaptive recomposition, this approach effectively mitigates speech dominance and enhances acoustic clarity. For audio-motion synchronization, we propose a bidirectional cross-modal forcing strategy where the cleaner modality guides the noisier one through decoupled denoising schedules, reinforced by a progressive stabilization strategy. Extensive experiments demonstrate that Unison achieves state-of-the-art performance in both audio perceptual quality and cross-modal synchronization, highlighting the importance of explicit multimodal harmonization in human-centric video generation.
>
---
#### [new 029] How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究全双工语音对话中的用户流路由问题，旨在解决LLM在生成回复时如何有效接收和处理用户输入。通过对比两种路由策略，分析其在语义整合与上下文鲁棒性上的权衡。**

- **链接: [https://arxiv.org/pdf/2605.10199](https://arxiv.org/pdf/2605.10199)**

> **作者:** Hui Lu; Xueyuan Chen; Huimeng Wang; Shuhai Peng; Shiyin Kang; Xixin Wu; Zhiyong Wu
>
> **摘要:** Full-duplex spoken dialogue requires a model to keep listening while generating its own spoken response. This is challenging for large language models (LLMs), which are designed to extend a single coherent sequence and do not naturally support user input arriving during generation. We argue that how the user stream is routed into the LLM is therefore a key architectural question for full-duplex modeling. To study this question, we extend a text-only LLM into a unified full-duplex spoken dialogue system and compare two routing strategies under a shared training pipeline: (i) channel fusion, which injects the user stream directly into the LLM input, and (ii) cross-attention routing, which keeps the user stream as external memory accessed through cross-attention adapters. Experiments on spoken question answering and full-duplex interaction benchmarks reveal a clear tradeoff. Channel fusion yields stronger semantic grounding and consistently better question-answering performance. However, under semantically overlapping conditions such as user interruptions, it is more vulnerable to context corruption: if the model fails to stop in time, the overlapping user stream can interfere with ongoing generation and lead to semantically incoherent continuations. Cross-attention routing underperforms on question answering, but better preserves the LLM generation context and is more robust to this failure mode. These results establish user-stream routing as a central design axis in full-duplex spoken dialogue and offer practical guidance on the tradeoff between semantic integration and context robustness. We provide a demo page for qualitative inspection.
>
---
#### [new 030] Uniqueness on a Continuum: Quantifying Tonal Ambiguity Using Information Theory
- **分类: cs.IT; cs.SD; math.HO**

- **简介: 该论文属于音乐理论分析任务，旨在解决 tonal ambiguity 的量化问题。通过信息理论提出连续度量，克服现有方法的局限性，扩展了调性关系的分析工具。**

- **链接: [https://arxiv.org/pdf/2605.08224](https://arxiv.org/pdf/2605.08224)**

> **作者:** Michael Seltenreich
>
> **备注:** 14 pages, 6 figures, 9 tables
>
> **摘要:** We propose a continuous measure of tonal ambiguity that extends the established concept of uniqueness. While uniqueness is widely regarded as necessary for tonality, it cannot (i) discriminate among sets that possess it, (ii) capture hierarchical organization in modes of limited transposition, or (iii) account for temporal unfolding. To address these limitations, we introduce a companion measure, grounded in information theory, that quantifies tonal ambiguity on a continuous scale. The measure applies across pitch-class sets and tuning systems, expanding analytic coverage of tonal relationships and offering a practical tool for theory and analysis.
>
---
#### [new 031] Dolphin-CN-Dialect: Where Chinese Dialects Matter
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升中文方言识别性能。针对数据不平衡问题，提出温度采样策略和改进的分词器，优化模型结构与部署效率。**

- **链接: [https://arxiv.org/pdf/2605.08961](https://arxiv.org/pdf/2605.08961)**

> **作者:** Yangyang Meng; Huihang Zhong; Guodong Lin; Guanbo Wang; Hu Du; Zhiming Shao; Yukai Huang; Ke Li; Wei-Qiang Zhang
>
> **摘要:** We present Dolphin-CN-Dialect, a streaming-capable ASR model with a focus on Chinese and dialect-rich scenarios. Compared to the previous version, Dolphin-CN-Dialect introduces substantial improvements in data processing, tokenization, training stability, and data sampling strategies. To address the challenges of highly imbalanced dialect data, we propose a temperature-based sampling strategy that effectively balances standard Mandarin and low-resource dialects, leading to significant gains in dialect recognition performance. In addition, we redesign the tokenizer to better align with linguistic characteristics, adopting character-level modeling for Chinese and subword modeling for English, while introducing extensible dialect tokens. Experimental results show that Dolphin-CN-Dialect achieves improvement in dialect recognition accuracy and CER reduction compared to Dolphin. Furthermore, Dolphin-CN-Dialect reaches competitive performance with recent SOTA open-source ASR models, while maintaining a significantly smaller model size. Dolphin-CN-Dialect supports both streaming and non-streaming inference, enabling a practical balance between latency and accuracy. It also provides flexible customization through hotword support and efficient deployment optimized for specialized hardware. These improvements make Dolphin-CN-Dialect a strong and practical solution for real-world multi-dialect ASR applications.
>
---
## 更新

#### [replaced 001] Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于音乐生成任务，旨在解决强化学习后训练中的奖励黑客问题。通过对抗训练提升输出多样性与和谐性，增强实时互动中的创作能力。**

- **链接: [https://arxiv.org/pdf/2511.17879](https://arxiv.org/pdf/2511.17879)**

> **作者:** Yusong Wu; Stephen Brade; Aleksandra Teng Ma; Tia-Jane Fowler; Enning Yang; Berker Banar; Aaron Courville; Natasha Jaques; Cheng-Zhi Anna Huang
>
> **备注:** v3: fix the Figure numbering bugs
>
> **摘要:** Most applications of generative AI involve a sequential interaction in which a person inputs a prompt and waits for a response, and where reaction time and adaptivity are not important factors. In contrast, live jamming is a collaborative interaction that requires real-time coordination and adaptation without access to the other player's future moves, while preserving diversity to sustain a creative flow. Reinforcement learning post-training enables effective adaptation through on-policy interaction, yet it often reduces output diversity by exploiting coherence-based rewards. This collapse, known as ``reward hacking'', affects many RL post-training pipelines, but is especially harmful in live jamming, where musical creativity relies on dynamic variation and mutual responsiveness. In this paper, we propose a novel adversarial training method on policy-generated trajectories to mitigate reward hacking in RL post-training for melody-to-chord accompaniment. A co-evolving discriminator separates policy trajectories from the data distribution, while the policy maximizes the discriminator output in addition to coherence rewards to prevent collapse to trivial outputs. We evaluate accompaniment quality and output diversity in simulation with both fixed test melodies and learned melody agents, and we conduct a user study with the model deployed in a real-time interactive system with expert musicians. Quantitative evaluation and user feedback demonstrate improved output diversity, harmonic coherence, adaptation speed and user agency. Our results demonstrate a simple yet effective method to mitigate reward hacking in RL post-training of generative sequence models.
>
---
#### [replaced 002] The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频-语言理解任务，旨在解决空间音频理解问题。提出TWNM框架，通过空间证据增强模型，提升场景分析能力。**

- **链接: [https://arxiv.org/pdf/2601.02954](https://arxiv.org/pdf/2601.02954)**

> **作者:** Yuhuan You; Lai Wei; Xihong Wu; Tianshu Qu
>
> **备注:** 25 pages, 4 figures
>
> **摘要:** Large audio-language models have made rapid progress in recognizing what is present in an audio clip, but spatial audio-language understanding still lacks a clear task interface. A model must also decide where sound events occur, which semantic and spatial attributes belong to the same auditory object, how multiple objects are arranged, and whether a scene-level answer is physically plausible. We formalize this capability as audio scene analysis (ASA), a three-level problem spanning atomic perception, relational integration, and cognitive reasoning. We propose The World is Not Mono (TWNM), a framework that equips audio-language models with explicit spatial evidence. TWNM uses physically grounded First-Order Ambisonics (FOA) simulation for controllable supervision, learns slot-regularized spatial representations from multichannel audio, fuses them with semantic audio features, and trains with a progressive curriculum ending in preference optimization over metadata-derived answers and auxiliary format/evidence rewards. To operationalize ASA, we build a controlled benchmark from scene metadata, covering localization, attribute binding, spatial comparison, scene abduction, and counterfactual reasoning. On this benchmark, TWNM achieves 70.8% overall accuracy, 66.4% on spatial-family tasks, and 79.76% on mixed L3 scene-level multiple-choice QA. We also audit monaural and binaural reference systems as diagnostic references with explicit audit labels, since they differ in spatial input, training interface, and output format. The supported claim is that a clearly defined ASA hierarchy, FOA-conditioned spatial representations, and metadata-grounded training enable controlled, auditable spatial audio-language reasoning, with STARSS23 providing a limited real-recording diagnostic.
>
---
#### [replaced 003] AQUA-Bench: Beyond Finding Answers to Knowing When There Are None in Audio Question Answering
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出AQUA-Bench，用于评估音频问答中的不可回答性问题。任务为音频问答中的不可回答性检测，解决现有基准忽略此类问题的缺陷。工作包括构建基准并评估三种不可回答场景。**

- **链接: [https://arxiv.org/pdf/2601.12248](https://arxiv.org/pdf/2601.12248)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026 (Oral). Project Website: this https URL
>
> **摘要:** Recent advances in audio-aware large language models have shown strong performance on audio question answering. However, existing benchmarks mainly cover answerable questions and overlook the challenge of unanswerable ones, where no reliable answer can be inferred from the audio. Such cases are common in real-world settings, where questions may be misleading, ill-posed, or incompatible with the information. To address this gap, we present AQUA-Bench, a benchmark for Audio Question Unanswerability Assessment. It systematically evaluates three scenarios: Absent Answer Detection (the correct option is missing), Incompatible Answer Set Detection (choices are categorically mismatched with the question), and Incompatible Audio Question Detection (the question is irrelevant or lacks sufficient grounding in the audio). By assessing these cases, AQUA-Bench offers a rigorous measure of model reliability and promotes the development of audio-language systems that are more robust and trustworthy. Our experiments suggest that while models excel on standard answerable tasks, they often face notable challenges with unanswerable ones, pointing to a blind spot in current audio-language understanding.
>
---
#### [replaced 004] FunnelNet: An End-to-End Deep Learning Framework to Monitor Digital Heart Murmur in Real-Time
- **分类: eess.SP; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于心音检测任务，旨在解决传统方法在准确性、成本和实时性上的不足。提出FunnelNet模型，结合滤波与卷积网络，实现高效实时心音识别。**

- **链接: [https://arxiv.org/pdf/2405.09570](https://arxiv.org/pdf/2405.09570)**

> **作者:** Md Jobayer; Md. Mehedi Hasan Shawon; Md Zakir Hossain; Shreya Ghosh; Imre Rudas; Tom Gedeon; Md Rakibul Hasan
>
> **摘要:** Heart murmurs are abnormal sounds caused by turbulent blood flow in the heart. Several diagnostic methods are available to detect heart murmurs and their severity, including cardiac auscultation, echocardiography, and phonocardiography (PCG). However, these methods have limitations, including the need for extensive training among healthcare providers, the cost and accessibility of echocardiography, and noise interference during PCG data processing. This study proposes an end-to-end real-time heart murmur detection approach using traditional and depthwise separable convolutional networks. We applied a Butterworth filter and Continuous Wavelet Transform (CWT) to eliminate noise and extract meaningful features from the PCG data. The proposed network consists of three parts: a Squeeze net that generates a compressed data representation, a Bottleneck layer that minimizes computational complexity using depthwise-separable convolutions, and an Expansion net that up-samples the data to capture fine details. We evaluated our model on the publicly available CirCor pediatric heart sound dataset. Using only $\sim$5.4k parameters, we achieved an accuracy of 85%, a sensitivity of 85%, and a specificity of 92%, successfully outperforming several larger models. Furthermore, we converted our network into a TinyML format and tested it on two resource-constrained devices, achieving an average real-time inference accuracy of 91% on a Raspberry Pi 4B and 80% on an Android smartphone. The proposed lightweight model offers a robust deep learning framework for accurate, real-time heart murmur detection, showing strong promise for accessible medical diagnostics in limited-resource environments. The code is publicly available at this https URL.
>
---
#### [replaced 005] X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出X-Voice，解决多语言零样本语音克隆问题，通过双阶段训练和语言标识注入，实现30种语言的语音克隆。**

- **链接: [https://arxiv.org/pdf/2605.05611](https://arxiv.org/pdf/2605.05611)**

> **作者:** Rixi Xu; Qingyu Liu; Haitao Li; Yushen Chen; Zhikang Niu; Yunting Yang; Jian Zhao; Ke Li; Berrak Sisman; Qinyuan Cheng; Xipeng Qiu; Kai Yu; Xie Chen
>
> **备注:** 16 pages, 4 figures, 9 tables
>
> **摘要:** In this paper, we present X-Voice, a 0.4B multilingual zero-shot voice cloning model that clones arbitrary voices and enables everyone to speak 30 languages. X-Voice is trained on a 420K-hour multilingual corpus using the International Phonetic Alphabet (IPA) as a unified representation. To eliminate the reliance on prompt text without complex preprocessing like forced alignment, we design a two-stage training paradigm. In Stage 1, we establish X-Voice$_{\text{s1}}$ through standard conditional flow-matching training and use it to synthesize 10K hours of speaker-consistent segments as audio prompts. In Stage 2, we fine-tune on these audio pairs with prompt text masked to derive X-Voice$_{\text{s2}}$, which enables zero-shot voice cloning without requiring transcripts of audio prompts. Architecturally, we extend F5-TTS by implementing a dual-level injection of language identifiers and decoupling and scheduling of Classifier-Free Guidance to facilitate multilingual speech synthesis. Subjective and objective evaluation results demonstrate that X-Voice outperforms existing flow-matching based multilingual systems like LEMAS-TTS and achieves zero-shot cross-lingual cloning capabilities comparable to billion-scale models such as Qwen3-TTS. To facilitate research transparency and community advancement, we open-source all related resources.
>
---
#### [replaced 006] AuthGlass: Benchmarking Voice Liveness Detection and Authentication on Smart Glasses via Comprehensive Acoustic Features
- **分类: cs.HC; cs.SD**

- **简介: 该论文属于语音活体检测与认证任务，针对智能眼镜的语音交互安全问题，提出新方法并构建数据集，提升抗欺骗攻击能力。**

- **链接: [https://arxiv.org/pdf/2509.20799](https://arxiv.org/pdf/2509.20799)**

> **作者:** Weiye Xu; Zhang Jiang; Siqi Zheng; Xiyuxing Zhang; Changhao Zhang; Jian Liu; Weiqiang Wang; Yuntao Wang
>
> **备注:** Submitted to IMWUT 2026
>
> **摘要:** With the rapid advancement of smart glasses, voice interaction has been widely adopted due to its naturalness and convenience. However, its practical deployment is often undermined by vulnerability to spoofing attacks, while no public dataset currently exists for voice liveness detection and authentication in smart-glasses scenarios. To address this challenge, we first collect a multi-acoustic-modal dataset comprising 16-channel audio data from 42 subjects, along with corresponding attack samples covering two attack categories. Based on insights derived from this collected data, we propose AuthG-Live, a sound-field-based voice liveness detection method, and AuthG-Net, a multi-acoustic-modal authentication model. We further benchmark seven voice liveness detection methods and four authentication methods across diverse acoustic modalities. The results demonstrate that our proposed approach achieves state-of-the-art performance on four benchmark tasks, and extensive ablation studies validate the generalizability of our methods \red{under real-world constraints}. Finally, we release this dataset, termed AuthGlass, to facilitate future research on voice liveness detection and authentication for smart glasses.
>
---
#### [replaced 007] Advancing Zero-Shot Open-Set Speech Deepfake Source Tracing
- **分类: eess.AS**

- **简介: 该论文属于语音深度伪造溯源任务，解决零样本下攻击源验证问题。提出新框架，结合SSL-AASIST与AAM损失、RegMixup，对比零样本与少样本方法效果。**

- **链接: [https://arxiv.org/pdf/2509.24674](https://arxiv.org/pdf/2509.24674)**

> **作者:** Manasi Chhibber; Jagabandhu Mishra; Tomi H. Kinnunen
>
> **备注:** Accepted to Odyssey 2026
>
> **摘要:** We propose a novel zero-shot source tracing framework inspired by speaker verification. We adapt SSL-AASIST for attack classification, enhancing embeddings with AAM loss and RegMixup, and ensure that training attacks are disjoint from those forming fingerprint-trial pairs. For backend scoring in attack verification, we explore both zero-shot approaches (cosine similarity and Siamese) and few-shot approaches (MLP and Siamese). Experiments on our recently introduced STOPA dataset with an open set setting show that few-shot learning provides advantages in the in-distribution (ID) scenario, while zero-shot approaches perform better in the out-of-distribution (OOD) scenario. In attack source verification with ID trials, few-shot Siamese and MLP achieve equal error rates (EER) of 17.72% and 13.11%, compared to 29.91% for zero-shot cosine scoring. Conversely, in OOD trials, zero-shot cosine scoring reaches 16.43%, outperforming few-shot Siamese at 23.47% and MLP at 21.57%.
>
---
#### [replaced 008] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出MECAT基准，用于细粒度音频理解任务，解决现有基准无法区分模型输出细节的问题。通过多专家协作生成数据，并引入新评估指标DATE提升评估精度。**

- **链接: [https://arxiv.org/pdf/2507.23511](https://arxiv.org/pdf/2507.23511)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Xunying Liu; Junbo Zhang; Jian Luan
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at this https URL
>
---
#### [replaced 009] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决语音对话中的模态和口语化评估问题。提出SDiaReward模型及数据集，提升对话质量评估的准确性。**

- **链接: [https://arxiv.org/pdf/2603.14889](https://arxiv.org/pdf/2603.14889)**

> **作者:** Jingyu Lu; Yuhan Wang; Fan Zhuo; Xize Cheng; Changhao Pan; Xueyi Pu; Yifu Chen; Chenyuhao Wen; Tianle Liang; Zhou Zhao
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL.
>
---
#### [replaced 010] AU-Harness: An Open-Source Toolkit for Holistic Evaluation of Audio LLMs
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出AU-Harness，一个用于音频大语言模型全面评估的开源工具包。针对现有评估工具效率低、不统一的问题，该工作优化了处理流程，支持多轮对话分析，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2509.08031](https://arxiv.org/pdf/2509.08031)**

> **作者:** Hoang Nguyen; Sidharth Surapaneni; Akshay Kalkunte; Jash Mehta; Aman Tiwari; Oluwanifemi Bamgbose; Khyati Mahajan; Jash Shah; Shruthan Radhakrishna; Sathwik Tejaswi Madhusudhan; Vikas Yadav; Sai Rajeswar
>
> **摘要:** Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient and non-standardized toolkits that limit fair comparison and systematic assessment. Existing evaluation frameworks exhibit three critical limitations: (1) slow and inefficient processing pipeline that bottlenecks large-scale studies, (2) inadequate multi-turn dialogue support, leaving fundamental questions about cross-turn context integration and performance dynamics over extended conversations in LALMs unanswered; and (3) the absence of unified and scalable evaluation framework capable of keeping pace with the rapid growth of both LALMs and audio benchmarks. To address these issues, we introduce AU-Harness, an efficient and comprehensive evaluation framework for LALMs. Our system achieves a speedup of up to 151% over existing evaluation toolkits through optimized batch processing and parallel execution, enabling large-scale evaluations previously considered impractical. We provide standardized prompting protocols and flexible configurations for fair model comparison across diverse scenarios. AU-Harness unlocks a range of in-depth analyses difficult to conduct without a unified foundation, including multi-turn dialogue dynamics, enabling the study of true audio reasoning capabilities in existing LALMs. AU-Harness provides both practical evaluation tools and insights into model limitations, advancing systematic LALM development.
>
---
#### [replaced 011] AsymTalker: Identity-Consistent Long-Term Talking Head Generation via Asymmetric Distillation
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于 Talking Head 生成任务，解决长视频生成中的身份一致性问题。提出 AsymTalker 方法，通过时空对齐和知识蒸馏，实现高质量、长时间的身份一致语音驱动面部生成。**

- **链接: [https://arxiv.org/pdf/2605.02948](https://arxiv.org/pdf/2605.02948)**

> **作者:** Yuxin Lu; Jiayang Sun; Guibo Zhu; Min Cao
>
> **摘要:** Diffusion-based talking head generation has achieved remarkable visual quality, yet scaling it to long-term videos remains challenging. The widely adopted chunk-wise paradigm introduces two fundamental failures: (1) temporal-spatial misalignment between static identity references and dynamic audio streams, and (2) cascading identity drift propagated through self-generated continuity references across chunks. To address both issues, we propose AsymTalker, a novel diffusion-based talking head generation method comprising Temporal Reference Encoding (TRE) and Asymmetric Knowledge Distillation (AKD). First, TRE mitigates temporal-spatial misalignment by transforming the static identity image into a temporally coherent latent representation through encoding of a temporally replicated pseudo-video, without introducing additional parameters. Second, AKD resolves the inherent conditioning dilemma in chunk-wise training: using ground-truth references causes train-inference mismatch, while self-generated references entangle supervision with identity drift. Our asymmetric design circumvents this by anchoring the teacher model with ground-truth continuity references to provide drift-free, chunk-level supervision, thereby avoiding the teacher bottleneck. Meanwhile, the student model learns under inference-aligned conditions, conditioned only on self-generated references, and is trained via distribution matching to preserve identity over long horizons. Extensive experiments show AsymTalker achieves state-of-the-art results on HDTF and VFHQ. It guarantees high-fidelity, identity-consistent synthesis over 600-second videos and reaches a real-time inference speed of 66 FPS.
>
---
#### [replaced 012] Adapting a Text-to-Audio Model for Room Impulse Response Generation
- **分类: eess.AS**

- **简介: 该论文属于语音信号处理任务，旨在解决RIR数据稀缺问题。通过适配文本到音频模型，生成合理RIR，提升声学仿真效果。**

- **链接: [https://arxiv.org/pdf/2603.09708](https://arxiv.org/pdf/2603.09708)**

> **作者:** Kirak Kim; Sungyoung Kim
>
> **备注:** 5 pages, 1 figure, submitted to IWAENC 2026
>
> **摘要:** Room Impulse Responses (RIRs) enable realistic acoustic simulation, with applications ranging from multimedia production to speech data augmentation. However, acquiring high-quality real-world RIRs is labor-intensive, and data scarcity remains a challenge for data-driven RIR generation approaches. In this paper, we propose a novel approach to RIR generation by adapting a pre-trained text-to-audio model, demonstrating for the first time that large-scale generative audio priors can be effectively leveraged for the task. To address the lack of text-RIR paired data, we utilize a labeling pipeline leveraging vision-language models to extract acoustic descriptions from existing image-RIR datasets. We introduce an in-context learning strategy to accommodate free-form user prompts during inference. Evaluations including subjective listening test demonstrate that our model generates plausible RIRs. Audio examples are available on our demo website.
>
---
#### [replaced 013] Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决真实通信场景下检测系统鲁棒性不足的问题。通过构建数据集和提出数据增强方法提升检测性能。**

- **链接: [https://arxiv.org/pdf/2504.12423](https://arxiv.org/pdf/2504.12423)**

> **作者:** Haohan Shi; Xiyu Shi; Safak Dogan; Saif Alzubi; Tianjin Huang; Yunxiao Zhang
>
> **备注:** Accepted by EUSIPCO 2025
>
> **摘要:** Existing Audio Deepfake Detection (ADD) systems often struggle to generalise effectively due to the significantly degraded audio quality caused by audio codec compression and channel transmission effects in real-world communication scenarios. To address this challenge, we developed a rigorous benchmark to evaluate the performance of the ADD system under such scenarios. We introduced ADD-C, a new test dataset to evaluate the robustness of ADD systems under diverse communication conditions, including different combinations of audio codecs for compression and packet loss rates. Benchmarking three baseline ADD models on the ADD-C dataset demonstrated a significant decline in robustness under such conditions. A novel Data Augmentation (DA) strategy was proposed to improve the robustness of ADD systems. Experimental results demonstrated that the proposed approach significantly enhances the performance of ADD systems on the proposed ADD-C dataset. Our benchmark can assist future efforts towards building practical and robustly generalisable ADD systems.
>
---
#### [replaced 014] Gender Fairness in Audio Deepfake Detection: Performance and Disparity Analysis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决性别偏差问题。通过分析模型在不同性别的表现差异，揭示公平性不足，并提出使用公平性指标评估模型。**

- **链接: [https://arxiv.org/pdf/2603.09007](https://arxiv.org/pdf/2603.09007)**

> **作者:** Aishwarya Fursule; Shruti Kshirsagar; Anderson R. Avila
>
> **备注:** Paper Accepted to IEEE CAI Conference 2026
>
> **摘要:** Audio deepfake detection aims to detect real human voices from those generated by Artificial Intelligence (AI) and has emerged as a significant problem in the field of voice biometrics systems. With the ever-improving quality of synthetic voice, the probability of such a voice being exploited for illicit practices like identity thest and impersonation increases. Although significant progress has been made in the field of Audio Deepfake Detection in recent times, the issue of gender bias remains underexplored and in its nascent stage In this paper, we have attempted a thorough analysis of gender dependent performance and fairness in audio deepfake detection models. We have used the ASVspoof 5 dataset and train a ResNet-18 classifier and evaluate detection performance across four different audio features, and compared the performance with baseline AASIST model. Beyond conventional metrics such as Equal Error Rate (EER %), we incorporated five established fairness metrics to quantify gender disparities in the model. Our results show that even when the overall EER difference between genders appears low, fairness-aware evaluation reveals disparities in error distribution that are obscured by aggregate performance measures. These findings demonstrate that reliance on standard metrics is unreliable, whereas fairness metrics provide critical insights into demographic-specific failure modes. This work highlights the importance of fairness-aware evaluation for developing a more equitable, robust, and trustworthy audio deepfake detection system.
>
---
#### [replaced 015] PHALAR: Phasors for Learned Musical Audio Representations
- **分类: cs.SD; cs.AI; cs.LG; eess.SP**

- **简介: 该论文提出PHALAR，用于音乐音频表示学习，解决茎音提取任务中时间信息丢失的问题。通过引入谱池化层和复数头，提升准确率并减少参数量。**

- **链接: [https://arxiv.org/pdf/2605.03929](https://arxiv.org/pdf/2605.03929)**

> **作者:** Davide Marincione; Michele Mancusi; Giorgio Strano; Luca Cerovaz; Donato Crisostomi; Roberto Ribuoli; Emanuele Rodolà
>
> **摘要:** Stem retrieval, the task of matching missing stems to a given audio submix, is a key challenge currently limited by models that discard temporal information. We introduce PHALAR, a contrastive framework achieving a relative accuracy increase of up to $\approx 70\%$ over the state-of-the-art while requiring $<50\%$ of the parameters and a 7$\times$ training speedup. By utilizing a Learned Spectral Pooling layer and a complex-valued head, PHALAR enforces pitch-equivariant and phase-equivariant biases. PHALAR establishes new retrieval state-of-the-art across MoisesDB, Slakh, and ChocoChorales, correlating significantly higher with human coherence judgment than semantic baselines. Finally, zero-shot beat tracking and linear chord probing confirm that PHALAR captures robust musical structures beyond the retrieval task.
>
---
#### [replaced 016] EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决真实场景下检测效果下降的问题。通过构建EchoFake数据集，包含多种攻击方式，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.19414](https://arxiv.org/pdf/2510.19414)**

> **作者:** Tong Zhang; Yihuan Huang; Yanzhen Ren
>
> **备注:** ICASSP 2026
>
> **摘要:** The growing prevalence of speech deepfakes has raised serious concerns, particularly in real-world scenarios such as telephone fraud and identity theft. While many anti-spoofing systems have demonstrated promising performance on lab-generated synthetic speech, they often fail when confronted with physical replay attacks-a common and low-cost form of attack used in practical settings. Our experiments show that models trained on existing datasets exhibit severe performance degradation, with average accuracy dropping to 59.6% when evaluated on replayed audio. To bridge this gap, we present EchoFake, a comprehensive dataset comprising more than 120 hours of audio from over 13,000 speakers, featuring both cutting-edge zero-shot text-to-speech (TTS) speech and physical replay recordings collected under varied devices and real-world environmental settings. Additionally, we evaluate three baseline detection models and show that models trained on EchoFake achieve lower average EERs across datasets, indicating better generalization. By introducing more practical challenges relevant to real-world deployment, EchoFake offers a more realistic foundation for advancing spoofing detection methods.
>
---
#### [replaced 017] RIR-Former: Coordinate-Guided Transformer for Continuous Reconstruction of Room Impulse Responses
- **分类: eess.AS; cs.LG**

- **简介: 该论文提出RIR-Former，用于连续重建房间脉冲响应（RIR），解决密集测量困难的问题。通过Transformer结构和位置编码，实现任意阵列位置的插值，提升整体重建效果。**

- **链接: [https://arxiv.org/pdf/2602.01861](https://arxiv.org/pdf/2602.01861)**

> **作者:** Shaoheng Xu; Chunyi Sun; Jihui Zhang; Prasanga N. Samarasinghe; Thushara D. Abhayapala
>
> **备注:** Published in ICASSP 2026. Code: this https URL . Equal contribution: Shaoheng Xu and Chunyi Sun
>
> **摘要:** Room impulse responses (RIRs) are essential for many acoustic signal processing tasks, yet measuring them densely across space is often impractical. In this work, we propose RIR-Former, a grid-free, one-step feed-forward model for RIR reconstruction. By introducing a sinusoidal encoding module into a transformer backbone, our method effectively incorporates microphone position information, enabling interpolation at arbitrary array locations. Furthermore, a segmented multi-branch decoder is designed to separately handle early reflections and late reverberation, improving reconstruction across the entire RIR. Experiments on diverse simulated acoustic environments demonstrate that RIR-Former consistently outperforms state-of-the-art baselines in terms of normalized mean square error (NMSE) and cosine distance (CD), under varying missing rates and array configurations. These results highlight the potential of our approach for practical deployment and motivate future work on scaling from randomly spaced linear arrays to complex array geometries, dynamic acoustic scenes, and real-world environments.
>
---
