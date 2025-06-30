# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Fine-Tuning MIDI-to-Audio Alignment using a Neural Network on Piano Roll and CQT Representations
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于MIDI-to-audio对齐任务，旨在解决人类钢琴演奏音频与松散对齐MIDI文件的同步问题。通过CRNN模型提升对齐精度，并优于传统DTW方法。**

- **链接: [http://arxiv.org/pdf/2506.22237v1](http://arxiv.org/pdf/2506.22237v1)**

> **作者:** Sebastian Murgul; Moritz Reiser; Michael Heizmann; Christoph Seibert
>
> **备注:** 9 pages, 3 figures, 6 tables
>
> **摘要:** In this paper, we present a neural network approach for synchronizing audio recordings of human piano performances with their corresponding loosely aligned MIDI files. The task is addressed using a Convolutional Recurrent Neural Network (CRNN) architecture, which effectively captures spectral and temporal features by processing an unaligned piano roll and a spectrogram as inputs to estimate the aligned piano roll. To train the network, we create a dataset of piano pieces with augmented MIDI files that simulate common human timing errors. The proposed model achieves up to 20% higher alignment accuracy than the industry-standard Dynamic Time Warping (DTW) method across various tolerance windows. Furthermore, integrating DTW with the CRNN yields additional improvements, offering enhanced robustness and consistency. These findings demonstrate the potential of neural networks in advancing state-of-the-art MIDI-to-audio alignment.
>
---
#### [new 002] A Practical Approach to Power Saving in Hearables Using Sub-Nyquist Sampling with Bandwidth Extension
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于低功耗音频处理任务，解决hearables中语音增强的能耗与质量问题。通过子奈奎斯特采样和虚拟判别器实现高效语音增强。**

- **链接: [http://arxiv.org/pdf/2506.22321v1](http://arxiv.org/pdf/2506.22321v1)**

> **作者:** Tarikul Islam Tamiti; Anomadarshi Barua
>
> **摘要:** Hearables are wearable computers that are worn on the ear. Bone conduction microphones (BCMs) are used with air conduction microphones (ACMs) in hearables as a supporting modality for multimodal speech enhancement (SE) in noisy conditions. However, existing works don't consider the following practical aspects for low-power implementations on hearables: (i) They do not explore how lowering the sampling frequencies and bit resolutions in analog-to-digital converters (ADCs) of hearables jointly impact low-power processing and multimodal SE in terms of speech quality and intelligibility. (ii) They don't discuss how GAN-like audio quality can be achieved without using actual GAN discriminators. And (iii) They don't process signals from ACMs/BCMs at sub-Nyquist sampling rate because, in their frameworks, they lack a wideband reconstruction methodology from their narrowband parts. We propose SUBARU (\textbf{Sub}-Nyquist \textbf{A}udio \textbf{R}esolution \textbf{U}psampling), which achieves the following: SUBARU (i) intentionally uses sub-Nyquist sampling and low bit resolution in ADCs, achieving a 3.31x reduction in power consumption; (ii) introduces novel multi-scale and multi-period virtual discriminators, which achieve GAN-like audio quality without using GANs' adversarial training; and (iii) achieves streaming operations on mobile platforms and SE in in-the-wild noisy conditions with an inference time of 1.74ms and a memory footprint of less than 13.77MB.
>
---
#### [new 003] Robust and Efficient Autoregressive Speech Synthesis with Dynamic Chunk-wise Prediction Policy
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，解决长序列生成中的效率与质量问题。提出DCAR框架，通过动态分块预测提升合成效率和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.22023v1](http://arxiv.org/pdf/2506.22023v1)**

> **作者:** Bohan Li; Zhihan Li; Haoran Wang; Hanglei Zhang; Yiwei Guo; Hankun Wang; Xie Chen; Kai Yu
>
> **备注:** 17 pages, 8 figures, 5 tables
>
> **摘要:** Recently, autoregressive (AR) language models have emerged as a dominant approach in speech synthesis, offering expressive generation and scalable training. However, conventional AR speech synthesis models relying on the next-token prediction paradigm often encounter significant challenges when handling long speech sequences. These models often struggle to construct stable frame-to-frame attention, leading to increased latency and degraded synthesis quality, thereby limiting their feasibility for real-time applications. To address these limitations, we introduce a novel dynamic chunk-wise autoregressive synthesis framework, termed DCAR, designed to enhance both efficiency and intelligibility robustness in AR speech generation. DCAR introduces a chunk-to-frame attention mechanism through training with multi-token prediction, enabling dynamic chunk prediction in variable speech contexts using a lightweight module trained on-policy. DCAR dynamically adjusts the token prediction span, significantly reducing the sequence length dependency while obtaining high synthesis quality. Comprehensive empirical evaluations demonstrate that DCAR substantially outperforms traditional next-token prediction models, achieving up to 72.27% intelligibility improvement and 2.61x inference speedup simultaneously on the test set. Furthermore, we conduct comprehensive analysis to support it as a versatile foundation for next-generation speech synthesis systems.
>
---
#### [new 004] Reconstructing Intelligible Speech from the Pressure Sensor Data in HVACs
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于语音重建任务，旨在从HVAC压力传感器数据中恢复可理解语音。针对低分辨率和噪声问题，提出WaLi模型，实现高精度语音重建。**

- **链接: [http://arxiv.org/pdf/2506.22311v1](http://arxiv.org/pdf/2506.22311v1)**

> **作者:** Tarikul Islam Tamiti; Biraj Joshi; Rida Hasan; Anomadarshi Barua
>
> **摘要:** Pressure sensors are an integrated component of modern Heating, Ventilation, and Air Conditioning (HVAC) systems. As these pressure sensors operate within the 0-10 Pa range, support high sampling frequencies of 0.5-2 kHz, and are often placed close to human proximity, they can be used to eavesdrop on confidential conversation, since human speech has a similar audible range of 0-10 Pa and a bandwidth of 4 kHz for intelligible quality. This paper presents WaLi, which reconstructs intelligible speech from the low-resolution and noisy pressure sensor data by providing the following technical contributions: (i) WaLi reconstructs intelligible speech from a minimum of 0.5 kHz sampling frequency of pressure sensors, whereas previous work can only detect hot words/phrases. WaLi uses complex-valued conformer and Complex Global Attention Block (CGAB) to capture inter-phoneme and intra-phoneme dependencies that exist in the low-resolution pressure sensor data. (ii) WaLi handles the transient noise injected from HVAC fans and duct vibrations, by reconstructing both the clean magnitude and phase of the missing frequencies of the low-frequency aliased components. Extensive measurement studies on real-world pressure sensors show an LSD of 1.24 and NISQA-MOS of 1.78 for 0.5 kHz to 8 kHz upsampling. We believe that such levels of accuracy pose a significant threat when viewed from a privacy perspective that has not been addressed before for pressure sensors.
>
---
#### [new 005] Efficient Multilingual ASR Finetuning via LoRA Language Experts
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决多语言干扰问题。通过LoRA语言专家实现高效微调，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2506.21555v1](http://arxiv.org/pdf/2506.21555v1)**

> **作者:** Jiahong Li; Yiwen Shao; Jianheng Zhuo; Chenda Li; Liliang Tang; Dong Yu; Yanmin Qian
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** Recent advancements in deep learning have significantly enhanced multilingual automatic speech recognition (ASR) due to the development of advanced model architectures and available large-scale multilingual datasets. Despite that, multilingual ASR still suffers from the curse of multilinguality in that different languages tend to interfere with each other, making it difficult for the ASR model to identify multiple languages effectively while sharing model capacity across them. This paper proposes an efficient finetuning framework for customized multilingual ASR via prepared LoRA language experts based on Whisper. Through LoRA expert fusion or knowledge distillation, our approach achieves better recognition performance on target languages than standard fine-tuning methods. Experimental results demonstrate that the proposed models yield approximately 10\% and 15\% relative performance gains in language-aware and language-agnostic scenarios, respectively.
>
---
#### [new 006] Adapting Foundation Speech Recognition Models to Impaired Speech: A Semantic Re-chaining Approach for Personalization of German Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决非典型发音（如脑瘫）导致的识别难题。通过语义重构方法提升模型个性化，改善转录质量。**

- **链接: [http://arxiv.org/pdf/2506.21622v1](http://arxiv.org/pdf/2506.21622v1)**

> **作者:** Niclas Pokel; Pehuén Moure; Roman Boehringer; Yingqiang Gao
>
> **摘要:** Speech impairments caused by conditions such as cerebral palsy or genetic disorders pose significant challenges for automatic speech recognition (ASR) systems. Despite recent advances, ASR models like Whisper struggle with non-normative speech due to limited training data and the difficulty of collecting and annotating non-normative speech samples. In this work, we propose a practical and lightweight pipeline to personalize ASR models, formalizing the selection of words and enriching a small, speech-impaired dataset with semantic coherence. Applied to data from a child with a structural speech impairment, our approach shows promising improvements in transcription quality, demonstrating the potential to reduce communication barriers for individuals with atypical speech patterns.
>
---
#### [new 007] ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于 hate speech 检测任务，旨在解决儿童目标仇恨言论识别问题。研究构建了 ChildGuard 数据集，并评估现有模型效果，以促进相关技术发展。**

- **链接: [http://arxiv.org/pdf/2506.21613v1](http://arxiv.org/pdf/2506.21613v1)**

> **作者:** Gautam Siddharth Kashyap; Mohammad Anas Azeez; Rafiq Ali; Zohaib Hasan Siddiqui; Jiechao Gao; Usman Naseem
>
> **摘要:** The increasing prevalence of child-targeted hate speech online underscores the urgent need for specialized datasets to address this critical issue. Existing hate speech datasets lack agespecific annotations, fail to capture nuanced contexts, and overlook the unique emotional impact on children. To bridge this gap, we introduce ChildGuard1, a curated dataset derived from existing corpora and enriched with child-specific annotations. ChildGuard captures diverse contexts of child-targeted hate speech, spanning age groups. We benchmark existing state-of-the-art hate speech detection methods, including Large Language Models (LLMs), and assess their effectiveness in detecting and contextualizing child-targeted hate speech. To foster further research in this area, we publicly release ChildGuard, providing a robust foundation for developing improved methods to detect and mitigate such harm.
>
---
#### [new 008] IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决情感表达与语音时长控制问题。提出IndexTTS2模型，实现情感与音色解耦及零样本情感控制。**

- **链接: [http://arxiv.org/pdf/2506.21619v1](http://arxiv.org/pdf/2506.21619v1)**

> **作者:** Siyi Zhou; Yiquan Zhou; Yi He; Xun Zhou; Jinchao Wang; Wei Deng; Jingchen Shu
>
> **摘要:** Large-scale text-to-speech (TTS) models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This is a key limitation in applications such as video dubbing that require strict audio-visual synchronization. This paper introduces IndexTTS2, which proposes a novel and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens for precise duration control; the other does not require manual input and lets the model freely generate speech while preserving prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model can perfectly reproduce the emotional characteristics of the input prompt. Users may also provide a separate emotion prompt, even from a different speaker, allowing the model to reconstruct the target timbre while conveying the desired emotion. To enhance clarity during strong emotional expressions, we incorporate GPT latent representations to improve speech stability. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This enables effective guidance of speech generation with desired emotional tendencies using natural language input. Experimental results demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in word error rate, speaker similarity, and emotional fidelity.
>
---
#### [new 009] Explainable anomaly detection for sound spectrograms using pooling statistics with quantile differences
- **分类: stat.AP; cs.SD; eess.AS; stat.CO; 62; G.3**

- **简介: 该论文属于异常检测任务，旨在解决声谱图中的异常识别问题，通过统计方法实现可解释的检测算法。**

- **链接: [http://arxiv.org/pdf/2506.21921v1](http://arxiv.org/pdf/2506.21921v1)**

> **作者:** Nicolas Thewes; Philipp Steinhauer; Patrick Trampert; Markus Pauly; Georg Schneider
>
> **摘要:** Anomaly detection is the task of identifying rarely occurring (i.e. anormal or anomalous) samples that differ from almost all other samples in a dataset. As the patterns of anormal samples are usually not known a priori, this task is highly challenging. Consequently, anomaly detection lies between semi- and unsupervised learning. The detection of anomalies in sound data, often called 'ASD' (Anomalous Sound Detection), is a sub-field that deals with the identification of new and yet unknown effects in acoustic recordings. It is of great importance for various applications in Industry 4.0. Here, vibrational or acoustic data are typically obtained from standard sensor signals used for predictive maintenance. Examples cover machine condition monitoring or quality assurance to track the state of components or products. However, the use of intelligent algorithms remains a controversial topic. Management generally aims for cost-reduction and automation, while quality and maintenance experts emphasize the need for human expertise and comprehensible solutions. In this work, we present an anomaly detection approach specifically designed for spectrograms. The approach is based on statistical evaluations and is theoretically motivated. In addition, it features intrinsic explainability, making it particularly suitable for applications in industrial settings. Thus, this algorithm is of relevance for applications in which black-box algorithms are unwanted or unsuitable.
>
---
#### [new 010] WTFormer: A Wavelet Conformer Network for MIMO Speech Enhancement with Spatial Cues Peservation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多通道语音增强任务，旨在解决MIMO处理中空间信息丢失问题。提出WTFormer网络，结合小波变换与Conformer模型，有效保留空间特征。**

- **链接: [http://arxiv.org/pdf/2506.22001v1](http://arxiv.org/pdf/2506.22001v1)**

> **作者:** Lu Han; Junqi Zhao; Renhua Peng
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** Current multi-channel speech enhancement systems mainly adopt single-output architecture, which face significant challenges in preserving spatio-temporal signal integrity during multiple-input multiple-output (MIMO) processing. To address this limitation, we propose a novel neural network, termed WTFormer, for MIMO speech enhancement that leverages the multi-resolution characteristics of wavelet transform and multi-dimensional collaborative attention to effectively capture globally distributed spatial features, while using Conformer for time-frequency modeling. A multi task loss strategy accompanying MUSIC algorithm is further proposed for optimization training to protect spatial information to the greatest extent. Experimental results on the LibriSpeech dataset show that WTFormer can achieve comparable denoising performance to advanced systems while preserving more spatial information with only 0.98M parameters.
>
---
#### [new 011] Language-Aware Prompt Tuning for Parameter-Efficient Seamless Language Expansion in Multilingual ASR
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多语言自动语音识别任务，旨在解决语言干扰和未见语言扩展问题。提出两种提示调优方法，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.21577v1](http://arxiv.org/pdf/2506.21577v1)**

> **作者:** Hongli Yang; Sheng Li; Hao Huang; Ayiduosi Tuohan; Yizhou Peng
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Recent advancements in multilingual automatic speech recognition (ASR) have been driven by large-scale end-to-end models like Whisper. However, challenges such as language interference and expanding to unseen languages (language expansion) without degrading performance persist. This paper addresses these with three contributions: 1) Entire Soft Prompt Tuning (Entire SPT), which applies soft prompts to both the encoder and decoder, enhancing feature extraction and decoding; 2) Language-Aware Prompt Tuning (LAPT), which leverages cross-lingual similarities to encode shared and language-specific features using lightweight prompt matrices; 3) SPT-Whisper, a toolkit that integrates SPT into Whisper and enables efficient continual learning. Experiments across three languages from FLEURS demonstrate that Entire SPT and LAPT outperform Decoder SPT by 5.0% and 16.0% in language expansion tasks, respectively, providing an efficient solution for dynamic, multilingual ASR models with minimal computational overhead.
>
---
#### [new 012] Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决低资源和代码切换场景下的识别问题。通过软提示微调方法提升模型性能，同时保持参数效率。**

- **链接: [http://arxiv.org/pdf/2506.21576v1](http://arxiv.org/pdf/2506.21576v1)**

> **作者:** Hongli Yang; Yizhou Peng; Hao Huang; Sheng Li
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages.
>
---
#### [new 013] UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于材料分割与视图合成任务，解决NeRF缺乏材料属性的问题。通过引入光谱解混，实现无监督材料分割与场景编辑。**

- **链接: [http://arxiv.org/pdf/2506.21884v1](http://arxiv.org/pdf/2506.21884v1)**

> **作者:** Fabian Perez; Sara Rojas; Carlos Hinojosa; Hoover Rueda-Chacón; Bernard Ghanem
>
> **备注:** Paper accepted at ICCV 2025 main conference
>
> **摘要:** Neural Radiance Field (NeRF)-based segmentation methods focus on object semantics and rely solely on RGB data, lacking intrinsic material properties. This limitation restricts accurate material perception, which is crucial for robotics, augmented reality, simulation, and other applications. We introduce UnMix-NeRF, a framework that integrates spectral unmixing into NeRF, enabling joint hyperspectral novel view synthesis and unsupervised material segmentation. Our method models spectral reflectance via diffuse and specular components, where a learned dictionary of global endmembers represents pure material signatures, and per-point abundances capture their distribution. For material segmentation, we use spectral signature predictions along learned endmembers, allowing unsupervised material clustering. Additionally, UnMix-NeRF enables scene editing by modifying learned endmember dictionaries for flexible material-based appearance manipulation. Extensive experiments validate our approach, demonstrating superior spectral reconstruction and material segmentation to existing methods. Project page: https://www.factral.co/UnMix-NeRF.
>
---
#### [new 014] SAGE: Spliced-Audio Generated Data for Enhancing Foundational Models in Low-Resource Arabic-English Code-Switched Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于低资源阿拉伯语-英语代码切换语音识别任务，解决数据稀缺问题。通过生成合成数据和优化模型训练，显著提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.22143v1](http://arxiv.org/pdf/2506.22143v1)**

> **作者:** Muhammad Umar Farooq; Oscar Saz
>
> **备注:** Accepted for IEEE MLSP 2025
>
> **摘要:** This paper investigates the performance of various speech SSL models on dialectal Arabic (DA) and Arabic-English code-switched (CS) speech. To address data scarcity, a modified audio-splicing approach is introduced to generate artificial CS speech data. Fine-tuning an already fine-tuned SSL model with the proposed Spliced-Audio Generated (SAGE) data results in an absolute improvement on Word Error Rate (WER) of 7.8% on Arabic and English CS benchmarks. Additionally, an Experience Replay (ER) inspired approach is proposed to enhance generalisation across DA and CS speech while mitigating catastrophic forgetting. Integrating an out-of-domain 3-gram language model reduces the overall mean WER from 31.7% to 26.6%. Few-shot fine-tuning for code-switching benchmarks further improves WER by 4.9%. A WER of 31.1% on Arabic-English CS benchmarks surpasses large-scale multilingual models, including USM and Whisper-large-v2 (both over ten times larger) by an absolute margin of 5.5% and 8.4%, respectively.
>
---
#### [new 015] Identifying Speaker Information in Feed-Forward Layers of Self-Supervised Speech Transformers
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决自监督语音Transformer中说话人信息编码问题。通过分析前馈层神经元，识别与说话人相关的特征，提升说话人相关任务性能。**

- **链接: [http://arxiv.org/pdf/2506.21712v1](http://arxiv.org/pdf/2506.21712v1)**

> **作者:** Tzu-Quan Lin; Hsi-Chun Cheng; Hung-yi Lee; Hao Tang
>
> **摘要:** In recent years, the impact of self-supervised speech Transformers has extended to speaker-related applications. However, little research has explored how these models encode speaker information. In this work, we address this gap by identifying neurons in the feed-forward layers that are correlated with speaker information. Specifically, we analyze neurons associated with k-means clusters of self-supervised features and i-vectors. Our analysis reveals that these clusters correspond to broad phonetic and gender classes, making them suitable for identifying neurons that represent speakers. By protecting these neurons during pruning, we can significantly preserve performance on speaker-related task, demonstrating their crucial role in encoding speaker information.
>
---
#### [new 016] Single-shot HDR using conventional image sensor shutter functions and optical randomization
- **分类: eess.IV; cs.CV; cs.GR; eess.SP; physics.optics**

- **简介: 该论文属于HDR成像任务，旨在解决单次拍摄获取高动态范围图像的问题。通过光学随机化和传感器GRR模式，实现更优的HDR重建。**

- **链接: [http://arxiv.org/pdf/2506.22426v1](http://arxiv.org/pdf/2506.22426v1)**

> **作者:** Xiang Dai; Kyrollos Yanny; Kristina Monakhova; Nicholas Antipa
>
> **摘要:** High-dynamic-range (HDR) imaging is an essential technique for overcoming the dynamic range limits of image sensors. The classic method relies on multiple exposures, which slows capture time, resulting in motion artifacts when imaging dynamic scenes. Single-shot HDR imaging alleviates this issue by encoding HDR data into a single exposure, then computationally recovering it. Many established methods use strong image priors to recover improperly exposed image detail. These approaches struggle with extended highlight regions. We utilize the global reset release (GRR) shutter mode of an off-the-shelf sensor. GRR shutter mode applies a longer exposure time to rows closer to the bottom of the sensor. We use optics that relay a randomly permuted (shuffled) image onto the sensor, effectively creating spatially randomized exposures across the scene. The exposure diversity allows us to recover HDR data by solving an optimization problem with a simple total variation image prior. In simulation, we demonstrate that our method outperforms other single-shot methods when many sensor pixels are saturated (10% or more), and is competitive at a modest saturation (1%). Finally, we demonstrate a physical lab prototype that uses an off-the-shelf random fiber bundle for the optical shuffling. The fiber bundle is coupled to a low-cost commercial sensor operating in GRR shutter mode. Our prototype achieves a dynamic range of up to 73dB using an 8-bit sensor with 48dB dynamic range.
>
---
## 更新

#### [replaced 001] Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.20995v2](http://arxiv.org/pdf/2506.20995v2)**

> **作者:** Akio Hayakawa; Masato Ishii; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** We propose a novel step-by-step video-to-audio generation method that sequentially produces individual audio tracks, each corresponding to a specific sound event in the video. Our approach mirrors traditional Foley workflows, aiming to capture all sound events induced by a given video comprehensively. Each generation step is formulated as a guided video-to-audio synthesis task, conditioned on a target text prompt and previously generated audio tracks. This design is inspired by the idea of concept negation from prior compositional generation frameworks. To enable this guided generation, we introduce a training framework that leverages pre-trained video-to-audio models and eliminates the need for specialized paired datasets, allowing training on more accessible data. Experimental results demonstrate that our method generates multiple semantically distinct audio tracks for a single input video, leading to higher-quality composite audio synthesis than existing baselines.
>
---
#### [replaced 002] An accurate measurement of parametric array using a spurious sound filter topologically equivalent to a half-wavelength resonator
- **分类: cs.SD; eess.AS; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2504.12398v2](http://arxiv.org/pdf/2504.12398v2)**

> **作者:** Woongji Kim; Beomseok Oh; Junsuk Rho; Wonkyu Moon
>
> **备注:** 12 pages, 11 figures. Accepted for publication in Applied Acoustics
>
> **摘要:** Parametric arrays (PA) offer exceptional directivity and compactness compared to conventional loudspeakers, facilitating various acoustic applications. However, accurate measurement of audio signals generated by PA remains challenging due to spurious ultrasonic sounds arising from microphone nonlinearities. Existing filtering methods, including Helmholtz resonators, phononic crystals, polymer films, and grazing incidence techniques, exhibit practical constraints such as size limitations, fabrication complexity, or insufficient attenuation. To address these issues, we propose and demonstrate a novel acoustic filter based on the design of a half-wavelength resonator. The developed filter exploits the nodal plane in acoustic pressure distribution, effectively minimizing microphone exposure to targeted ultrasonic frequencies. Fabrication via stereolithography (SLA) 3D printing ensures high dimensional accuracy, which is crucial for high-frequency acoustic filters. Finite element method (FEM) simulations guided filter optimization for suppression frequencies at 40 kHz and 60 kHz, achieving high transmission loss (TL) around 60 dB. Experimental validations confirm the filter's superior performance in significantly reducing spurious acoustic signals, as reflected in frequency response, beam pattern, and propagation curve measurements. The proposed filter ensures stable and precise acoustic characterization, independent of measurement distances and incidence angles. This new approach not only improves measurement accuracy but also enhances reliability and reproducibility in parametric array research and development.
>
---
#### [replaced 003] Audio-Plane: Audio Factorization Plane Gaussian Splatting for Real-Time Talking Head Synthesis
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.22605v2](http://arxiv.org/pdf/2503.22605v2)**

> **作者:** Shuai Shen; Wanhua Li; Yunpeng Zhang; Yap-Peng Tan; Jiwen Lu
>
> **备注:** Demo video at \url{https://sstzal.github.io/Audio-Plane/}
>
> **摘要:** Talking head synthesis has emerged as a prominent research topic in computer graphics and multimedia, yet most existing methods often struggle to strike a balance between generation quality and computational efficiency, particularly under real-time constraints. In this paper, we propose a novel framework that integrates Gaussian Splatting with a structured Audio Factorization Plane (Audio-Plane) to enable high-quality, audio-synchronized, and real-time talking head generation. For modeling a dynamic talking head, a 4D volume representation, which consists of three axes in 3D space and one temporal axis aligned with audio progression, is typically required. However, directly storing and processing a dense 4D grid is impractical due to the high memory and computation cost, and lack of scalability for longer durations. We address this challenge by decomposing the 4D volume representation into a set of audio-independent spatial planes and audio-dependent planes, forming a compact and interpretable representation for talking head modeling that we refer to as the Audio-Plane. This factorized design allows for efficient and fine-grained audio-aware spatial encoding, and significantly enhances the model's ability to capture complex lip dynamics driven by speech signals. To further improve region-specific motion modeling, we introduce an audio-guided saliency splatting mechanism based on region-aware modulation, which adaptively emphasizes highly dynamic regions such as the mouth area. This allows the model to focus its learning capacity on where it matters most for accurate speech-driven animation. Extensive experiments on both the self-driven and the cross-driven settings demonstrate that our method achieves state-of-the-art visual quality, precise audio-lip synchronization, and real-time performance, outperforming prior approaches across both 2D- and 3D-based paradigms.
>
---
#### [replaced 004] USM-VC: Mitigating Timbre Leakage with Universal Semantic Mapping Residual Block for Voice Conversion
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.08524v3](http://arxiv.org/pdf/2504.08524v3)**

> **作者:** Na Li; Chuke Wang; Yu Gu; Zhifeng Li
>
> **摘要:** Voice conversion (VC) transforms source speech into a target voice by preserving the content. However, timbre information from the source speaker is inherently embedded in the content representations, causing significant timbre leakage and reducing similarity to the target speaker. To address this, we introduce a Universal Semantic Matching (USM) residual block to a content extractor. The residual block consists of two weighted branches: 1) universal semantic dictionary based Content Feature Re-expression (CFR) module, supplying timbre-free content representation. 2) skip connection to the original content layer, providing complementary fine-grained information. In the CFR module, each dictionary entry in the universal semantic dictionary represents a phoneme class, computed statistically using speech from multiple speakers, creating a stable, speaker-independent semantic set. We introduce a CFR method to obtain timbre-free content representations by expressing each content frame as a weighted linear combination of dictionary entries using corresponding phoneme posteriors as weights. Extensive experiments across various VC frameworks demonstrate that our approach effectively mitigates timbre leakage and significantly improves similarity to the target speaker.
>
---
#### [replaced 005] KNN-MMD: Cross Domain Wireless Sensing via Local Distribution Alignment
- **分类: cs.CV; cs.AI; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.04783v3](http://arxiv.org/pdf/2412.04783v3)**

> **作者:** Zijian Zhao; Zhijie Cai; Tingwei Chen; Xiaoyang Li; Hang Li; Qimei Chen; Guangxu Zhu
>
> **摘要:** Wireless sensing has recently found widespread applications in diverse environments, including homes, offices, and public spaces. By analyzing patterns in channel state information (CSI), it is possible to infer human actions for tasks such as person identification, gesture recognition, and fall detection. However, CSI is highly sensitive to environmental changes, where even minor alterations can significantly distort the CSI patterns. This sensitivity often leads to performance degradation or outright failure when applying wireless sensing models trained in one environment to another. To address this challenge, Domain Alignment (DAL) has been widely adopted for cross-domain classification tasks, as it focuses on aligning the global distributions of the source and target domains in feature space. Despite its popularity, DAL often neglects inter-category relationships, which can lead to misalignment between categories across domains, even when global alignment is achieved. To overcome these limitations, we propose K-Nearest Neighbors Maximum Mean Discrepancy (KNN-MMD), a novel few-shot method for cross-domain wireless sensing. Our approach begins by constructing a help set using KNN from the target domain, enabling local alignment between the source and target domains within each category using MMD. Additionally, we address a key instability issue commonly observed in cross-domain methods, where model performance fluctuates sharply between epochs. Further, most existing methods struggle to determine an optimal stopping point during training due to the absence of labeled data from the target domain. Our method resolves this by excluding the support set from the target domain during training and employing it as a validation set to determine the stopping criterion.The dataset and code are publicly available at https://github.com/RS2002/KNN-MMD .
>
---
#### [replaced 006] LoopGen: Training-Free Loopable Music Generation
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04466v4](http://arxiv.org/pdf/2504.04466v4)**

> **作者:** Davide Marincione; Giorgio Strano; Donato Crisostomi; Roberto Ribuoli; Emanuele Rodolà
>
> **摘要:** Loops--short audio segments designed for seamless repetition--are central to many music genres, particularly those rooted in dance and electronic styles. However, current generative music models struggle to produce truly loopable audio, as generating a short waveform alone does not guarantee a smooth transition from its endpoint back to its start, often resulting in audible discontinuities. We address this gap by modifying a non-autoregressive model (MAGNeT) to generate tokens in a circular pattern, letting the model attend to the beginning of the audio when creating its ending. This inference-only approach results in generations that are aware of future context and loop naturally, without the need for any additional training or data. We evaluate the consistency of loop transitions by computing token perplexity around the seam of the loop, observing a 55% improvement. Blind listening tests further confirm significant perceptual gains over baseline methods, improving mean ratings by 70%. Taken together, these results highlight the effectiveness of inference-only approaches in improving generative models and underscore the advantages of non-autoregressive methods for context-aware music generation.
>
---
#### [replaced 007] State-Space Models in Efficient Whispered and Multi-dialect Speech Recognition
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.16969v2](http://arxiv.org/pdf/2506.16969v2)**

> **作者:** Aref Farhadipour; Homayoon Beigi; Volker Dellwo; Hadi Veisi
>
> **备注:** paper is in 4+1 pages
>
> **摘要:** Whispered speech recognition presents significant challenges for conventional automatic speech recognition systems, particularly when combined with dialect variation. However, utilizing an efficient method to solve this problem using a low-range dataset and processing load is beneficial. This paper proposes a solution using a Mamba-based state-space model and four fine-tuned self-supervised models consisting of Wav2Vec2, WavLM, HuBERT, and Whisper to address the dual challenges of whispered speech and dialect diversity. Based on our knowledge, this represents the best performance reported on the wTIMIT and CHAINS datasets for whispered speech recognition. We trained the models using whispered and normal speech data across Singaporean, US, and Irish dialects. The findings demonstrated that utilizing the proposed Mamba-based model could work as a highly efficient model trained with low amounts of whispered data to simultaneously work on whispered and normal speech recognition. The code for this work is freely available.
>
---
