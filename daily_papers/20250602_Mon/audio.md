# 音频 cs.SD;  eess.SP

- **最新发布 30 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Patient-Aware Feature Alignment for Robust Lung Sound Classification:Cohesion-Separation and Global Alignment Losses
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文针对肺音分类中患者间差异导致模型泛化性差的问题，提出Patient-Aware Feature Alignment框架，通过Patient Cohesion-Separation Loss（聚类同患者特征、分离异患者特征）和Global Patient Alignment Loss（全局对齐患者中心），提升分类鲁棒性，在ICBHI数据集取得优异结果。**

- **链接: [http://arxiv.org/pdf/2505.23834v1](http://arxiv.org/pdf/2505.23834v1)**

> **作者:** Seung Gyu Jeong; Seong Eun Kim
>
> **备注:** Accepted INTERSPEECH 2025
>
> **摘要:** Lung sound classification is vital for early diagnosis of respiratory diseases. However, biomedical signals often exhibit inter-patient variability even among patients with the same symptoms, requiring a learning approach that considers individual differences. We propose a Patient-Aware Feature Alignment (PAFA) framework with two novel losses, Patient Cohesion-Separation Loss (PCSL) and Global Patient Alignment Loss (GPAL). PCSL clusters features of the same patient while separating those from other patients to capture patient variability, whereas GPAL draws each patient's centroid toward a global center, preventing feature space fragmentation. Our method achieves outstanding results on the ICBHI dataset with a score of 64.84\% for four-class and 72.08\% for two-class classification. These findings highlight PAFA's ability to capture individualized patterns and demonstrate performance gains in distinct patient clusters, offering broader applications for patient-centered healthcare.
>
---
#### [new 002] ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出ARECHO，用于语音多指标联合评估，解决不同指标间尺度、假设及依赖关系导致的联合估计难题。其创新包括语音信息标记管道、动态分类链建模指标依赖，及两步置信度解码，实验显示优于基线并提升可解释性。**

- **链接: [http://arxiv.org/pdf/2505.24518v1](http://arxiv.org/pdf/2505.24518v1)**

> **作者:** Jiatong Shi; Yifan Cheng; Bo-Hao Su; Hye-jin Shim; Jinchuan Tian; Samuele Cornell; Yiwen Zhao; Siddhant Arora; Shinji Watanabe
>
> **摘要:** Speech signal analysis poses significant challenges, particularly in tasks such as speech quality evaluation and profiling, where the goal is to predict multiple perceptual and objective metrics. For instance, metrics like PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility), and MOS (Mean Opinion Score) each capture different aspects of speech quality. However, these metrics often have different scales, assumptions, and dependencies, making joint estimation non-trivial. To address these issues, we introduce ARECHO (Autoregressive Evaluation via Chain-based Hypothesis Optimization), a chain-based, versatile evaluation system for speech assessment grounded in autoregressive dependency modeling. ARECHO is distinguished by three key innovations: (1) a comprehensive speech information tokenization pipeline; (2) a dynamic classifier chain that explicitly captures inter-metric dependencies; and (3) a two-step confidence-oriented decoding algorithm that enhances inference reliability. Experiments demonstrate that ARECHO significantly outperforms the baseline framework across diverse evaluation scenarios, including enhanced speech analysis, speech generation evaluation, and noisy speech evaluation. Furthermore, its dynamic dependency modeling improves interpretability by capturing inter-metric relationships.
>
---
#### [new 003] Discl-VC: Disentangled Discrete Tokens and In-Context Learning for Controllable Zero-Shot Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Discl-VC框架，针对零样本语音转换中难以精准控制源/目标说话人韵律风格的问题，通过解耦语音内容与韵律信息，结合自监督表示、上下文学习及非自回归离散韵律标记预测，实现可控且高精度的零样本语音转换。**

- **链接: [http://arxiv.org/pdf/2505.24291v1](http://arxiv.org/pdf/2505.24291v1)**

> **作者:** Kaidi Wang; Wenhao Guan; Ziyue Jiang; Hukai Huang; Peijie Chen; Weijie Wu; Qingyang Hong; Lin Li
>
> **摘要:** Currently, zero-shot voice conversion systems are capable of synthesizing the voice of unseen speakers. However, most existing approaches struggle to accurately replicate the speaking style of the source speaker or mimic the distinctive speaking style of the target speaker, thereby limiting the controllability of voice conversion. In this work, we propose Discl-VC, a novel voice conversion framework that disentangles content and prosody information from self-supervised speech representations and synthesizes the target speaker's voice through in-context learning with a flow matching transformer. To enable precise control over the prosody of generated speech, we introduce a mask generative transformer that predicts discrete prosody tokens in a non-autoregressive manner based on prompts. Experimental results demonstrate the superior performance of Discl-VC in zero-shot voice conversion and its remarkable accuracy in prosody control for synthesized speech.
>
---
#### [new 004] 4,500 Seconds: Small Data Training Approaches for Deep UAV Audio Classification
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究无人机(UAV)音频分类任务，针对数据稀缺问题，使用4500秒9类小规模音频数据，通过参数高效微调和数据增强，比较CNN与Transformer模型。结果显示CNN在准确率和效率上更优，但Transformer潜力显著，未来计划扩大数据集优化模型。**

- **链接: [http://arxiv.org/pdf/2505.23782v1](http://arxiv.org/pdf/2505.23782v1)**

> **作者:** Andrew P. Berg; Qian Zhang; Mia Y. Wang
>
> **备注:** Accepted at the 14th International Conference on Data Science, Technology, and Applications (DATA), 2025
>
> **摘要:** Unmanned aerial vehicle (UAV) usage is expected to surge in the coming decade, raising the need for heightened security measures to prevent airspace violations and security threats. This study investigates deep learning approaches to UAV classification focusing on the key issue of data scarcity. To investigate this we opted to train the models using a total of 4,500 seconds of audio samples, evenly distributed across a 9-class dataset. We leveraged parameter efficient fine-tuning (PEFT) and data augmentations to mitigate the data scarcity. This paper implements and compares the use of convolutional neural networks (CNNs) and attention-based transformers. Our results show that, CNNs outperform transformers by 1-2\% accuracy, while still being more computationally efficient. These early findings, however, point to potential in using transformers models; suggesting that with more data and further optimizations they could outperform CNNs. Future works aims to upscale the dataset to better understand the trade-offs between these approaches.
>
---
#### [new 005] FeatureSense: Protecting Speaker Attributes in Always-On Audio Sensing System
- **分类: cs.SD; cs.HC; eess.AS**

- **简介: 该论文属于隐私保护任务，旨在解决始终在线音频传感系统中保护说话人属性（如年龄、性别）不被推断的问题。现有方法虽遮蔽语音内容，仍存在属性泄露。论文提出隐私评估框架，开发FeatureSense库，设计自适应特征选择算法，在保障效用前提下提升隐私保护达60.6%。**

- **链接: [http://arxiv.org/pdf/2505.24115v1](http://arxiv.org/pdf/2505.24115v1)**

> **作者:** Bhawana Chhaglani; Sarmistha Sarna Gomasta; Yuvraj Agarwal; Jeremy Gummeson; Prashant Shenoy
>
> **摘要:** Audio is a rich sensing modality that is useful for a variety of human activity recognition tasks. However, the ubiquitous nature of smartphones and smart speakers with always-on microphones has led to numerous privacy concerns and a lack of trust in deploying these audio-based sensing systems. This paper addresses this critical challenge of preserving user privacy when using audio for sensing applications while maintaining utility. While prior work focuses primarily on protecting recoverable speech content, we show that sensitive speaker-specific attributes such as age and gender can still be inferred after masking speech and propose a comprehensive privacy evaluation framework to assess this speaker attribute leakage. We design and implement FeatureSense, an open-source library that provides a set of generalizable privacy-aware audio features that can be used for wide range of sensing applications. We present an adaptive task-specific feature selection algorithm that optimizes the privacy-utility-cost trade-off based on the application requirements. Through our extensive evaluation, we demonstrate the high utility of FeatureSense across a diverse set of sensing tasks. Our system outperforms existing privacy techniques by 60.6% in preserving user-specific privacy. This work provides a foundational framework for ensuring trust in audio sensing by enabling effective privacy-aware audio classification systems.
>
---
#### [new 006] Unified AI for Accurate Audio Anomaly Detection
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出了一种用于高精度音频异常检测的统一AI框架，解决噪声环境下的准确检测与实时应用挑战。整合频谱减法、自适应滤波提升音质，结合MFCC、OpenL3等特征提取方法，采用SVM、随机森林、CNN及集成模型，实验证明在TORGO等数据集上分类（如含糊/正常语音）的精度和召回率优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.23781v1](http://arxiv.org/pdf/2505.23781v1)**

> **作者:** Hamideh Khaleghpour; Brett McKinney
>
> **备注:** 6 pages, 14 figures. Based on original research. Submitted to arXiv for public preprint
>
> **摘要:** This paper presents a unified AI framework for high-accuracy audio anomaly detection by integrating advanced noise reduction, feature extraction, and machine learning modeling techniques. The approach combines spectral subtraction and adaptive filtering to enhance audio quality, followed by feature extraction using traditional methods like MFCCs and deep embeddings from pre-trained models such as OpenL3. The modeling pipeline incorporates classical models (SVM, Random Forest), deep learning architectures (CNNs), and ensemble methods to boost robustness and accuracy. Evaluated on benchmark datasets including TORGO and LibriSpeech, the proposed framework demonstrates superior performance in precision, recall, and classification of slurred vs. normal speech. This work addresses challenges in noisy environments and real-time applications and provides a scalable solution for audio-based anomaly detection.
>
---
#### [new 007] DS-Codec: Dual-Stage Training with Mirror-to-NonMirror Architecture Switching for Speech Codec
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DS-Codec，一种基于镜像与非镜像架构双阶段训练的神经语音编解码器，旨在提升语音重建质量。针对现有编解码器鲁棒性不足的问题，通过切换两种架构训练，增强代码本鲁棒性并平衡结构优势，实验验证了其高保真语音重建效果。**

- **链接: [http://arxiv.org/pdf/2505.24314v1](http://arxiv.org/pdf/2505.24314v1)**

> **作者:** Peijie Chen; Wenhao Guan; Kaidi Wang; Weijie Wu; Hukai Huang; Qingyang Hong; Lin Li
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Neural speech codecs are essential for advancing text-to-speech (TTS) systems. With the recent success of large language models in text generation, developing high-quality speech tokenizers has become increasingly important. This paper introduces DS-Codec, a novel neural speech codec featuring a dual-stage training framework with mirror and non-mirror architectures switching, designed to achieve superior speech reconstruction. We conduct extensive experiments and ablation studies to evaluate the effectiveness of our training strategy and compare the performance of the two architectures. Our results show that the mirrored structure significantly enhances the robustness of the learned codebooks, and the training strategy balances the advantages between mirrored and non-mirrored structures, leading to improved high-fidelity speech reconstruction.
>
---
#### [new 008] Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文聚焦多语言语音处理任务（LID和ASR），针对预训练模型在资源有限场景下的微调瓶颈，采用冻结上游、部分微调、低秩适配策略，结合数据增强和LID-CTC正则化，提升ML-SUPERB 2.0性能，实现LID准确率提升14%及ASR CER降低30%。**

- **链接: [http://arxiv.org/pdf/2505.24200v1](http://arxiv.org/pdf/2505.24200v1)**

> **作者:** Qingzheng Wang; Jiancheng Sun; Yifan Peng; Shinji Watanabe
>
> **摘要:** Multilingual speech processing with self-supervised or supervised pre-trained Speech Foundation Models (SFM) has achieved strong performance on tasks like Language Identification (LID) and Automatic Speech Recognition (ASR). However, these models struggle with limited resources during fine-tuning. This paper enhances multilingual LID and ASR on ML-SUPERB 2.0 by exploring multiple strategies for adapting SFMs, including frozen upstream training, partial fine-tuning, and low-rank adaptation. Furthermore, we employ data augmentation to mitigate performance gaps in few-shot settings and introduce LID Connectionist Temporal Classification (CTC) loss for regularization. Our approach achieves a 14% relative improvement in LID accuracy and a 30% relative reduction in ASR CER over the baseline on ML-SUPERB 2.0, securing second place in the Interspeech 2025 ML-SUPERB 2.0 Challenge.
>
---
#### [new 009] Rehearsal with Auxiliary-Informed Sampling for Audio Deepfake Detection
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文针对音频深度伪造检测中现有模型面临新攻击时性能下降的问题，提出RAIS方法。通过辅助标签生成网络指导记忆缓冲区的多样化样本选择，解决传统 rehearsal 技术的样本偏差与知识遗忘问题，提升持续学习效果，实现1.95%的平均EER。**

- **链接: [http://arxiv.org/pdf/2505.24486v1](http://arxiv.org/pdf/2505.24486v1)**

> **作者:** Falih Gozi Febrinanto; Kristen Moore; Chandra Thapa; Jiangang Ma; Vidya Saikrishna; Feng Xia
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** The performance of existing audio deepfake detection frameworks degrades when confronted with new deepfake attacks. Rehearsal-based continual learning (CL), which updates models using a limited set of old data samples, helps preserve prior knowledge while incorporating new information. However, existing rehearsal techniques don't effectively capture the diversity of audio characteristics, introducing bias and increasing the risk of forgetting. To address this challenge, we propose Rehearsal with Auxiliary-Informed Sampling (RAIS), a rehearsal-based CL approach for audio deepfake detection. RAIS employs a label generation network to produce auxiliary labels, guiding diverse sample selection for the memory buffer. Extensive experiments show RAIS outperforms state-of-the-art methods, achieving an average Equal Error Rate (EER) of 1.953 % across five experiences. The code is available at: https://github.com/falihgoz/RAIS.
>
---
#### [new 010] Acoustic Classification of Maritime Vessels using Learnable Filterbanks
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于水下目标声学分类任务，旨在解决不同环境与距离下船舶声纹识别的泛化性问题。提出CATFISH模型，通过可学习Gabor滤波器组动态提取频率特征，结合时序编码器提升分类鲁棒性，在VTUAD数据集上达96.63%准确率，超越此前方法12个百分点。**

- **链接: [http://arxiv.org/pdf/2505.23964v1](http://arxiv.org/pdf/2505.23964v1)**

> **作者:** Jonas Elsborg; Tejs Vegge; Arghya Bhowmik
>
> **备注:** 9 pages, 5 figures, 2 tables
>
> **摘要:** Reliably monitoring and recognizing maritime vessels based on acoustic signatures is complicated by the variability of different recording scenarios. A robust classification framework must be able to generalize across diverse acoustic environments and variable source-sensor distances. To this end, we present a deep learning model with robust performance across different recording scenarios. Using a trainable spectral front-end and temporal feature encoder to learn a Gabor filterbank, the model can dynamically emphasize different frequency components. Trained on the VTUAD hydrophone recordings from the Strait of Georgia, our model, CATFISH, achieves a state-of-the-art 96.63 % percent test accuracy across varying source-sensor distances, surpassing the previous benchmark by over 12 percentage points. We present the model, justify our architectural choices, analyze the learned Gabor filters, and perform ablation studies on sensor data fusion and attention-based pooling.
>
---
#### [new 011] SuPseudo: A Pseudo-supervised Learning Method for Neural Speech Enhancement in Far-field Speech Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于远场语音增强任务，旨在解决真实远场语音数据缺乏标注导致模型泛化性差的问题。提出DSE方法估计真实数据的直接声，结合SuPseudo伪监督学习利用伪标签训练模型，并设计FARNET模型，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.24450v1](http://arxiv.org/pdf/2505.24450v1)**

> **作者:** Longjie Luo; Lin Li; Qingyang Hong
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** Due to the lack of target speech annotations in real-recorded far-field conversational datasets, speech enhancement (SE) models are typically trained on simulated data. However, the trained models often perform poorly in real-world conditions, hindering their application in far-field speech recognition. To address the issue, we (a) propose direct sound estimation (DSE) to estimate the oracle direct sound of real-recorded data for SE; and (b) present a novel pseudo-supervised learning method, SuPseudo, which leverages DSE-estimates as pseudo-labels and enables SE models to directly learn from and adapt to real-recorded data, thereby improving their generalization capability. Furthermore, an SE model called FARNET is designed to fully utilize SuPseudo. Experiments on the MISP2023 corpus demonstrate the effectiveness of SuPseudo, and our system significantly outperforms the previous state-of-the-art. A demo of our method can be found at https://EeLLJ.github.io/SuPseudo/.
>
---
#### [new 012] Learning Normal Patterns in Musical Loops
- **分类: cs.SD; cs.IR; cs.LG; cs.MM; eess.AS**

- **简介: 该论文提出无监督框架，通过深度特征提取与异常检测（如Deep SVDD）解决音乐循环中的正常模式学习问题，克服传统方法依赖手工特征、领域限制及用户交互的缺陷。结合预训练HTS-AT模型与FFM生成音频嵌入，实验表明残差自编码器变体在分离异常方面表现最佳，尤其处理大幅变化时效果显著。**

- **链接: [http://arxiv.org/pdf/2505.23784v1](http://arxiv.org/pdf/2505.23784v1)**

> **作者:** Shayan Dadman; Bernt Arild Bremdal; Børre Bang; Rune Dalmo
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** This paper introduces an unsupervised framework for detecting audio patterns in musical samples (loops) through anomaly detection techniques, addressing challenges in music information retrieval (MIR). Existing methods are often constrained by reliance on handcrafted features, domain-specific limitations, or dependence on iterative user interaction. We address these limitations through an architecture combining deep feature extraction with unsupervised anomaly detection. Our approach leverages a pre-trained Hierarchical Token-semantic Audio Transformer (HTS-AT), paired with a Feature Fusion Mechanism (FFM), to generate representations from variable-length audio loops. These embeddings are processed using one-class Deep Support Vector Data Description (Deep SVDD), which learns normative audio patterns by mapping them to a compact latent hypersphere. Evaluations on curated bass and guitar datasets compare standard and residual autoencoder variants against baselines like Isolation Forest (IF) and and principle component analysis (PCA) methods. Results show our Deep SVDD models, especially the residual autoencoder variant, deliver improved anomaly separation, particularly for larger variations. This research contributes a flexible, fully unsupervised solution for processing diverse audio samples, overcoming previous structural and input limitations while enabling effective pattern identification through distance-based latent space scoring.
>
---
#### [new 013] Masked Self-distilled Transducer-based Keyword Spotting with Semi-autoregressive Decoding
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对基于RNN-T的关键词检测（KWS）中预测网络简单导致过拟合的问题，提出masked自蒸馏（MSD）训练策略缓解过拟合，并设计半自回归（SAR）解码结合AR与NAR优势。实验表明方法有效提升性能。**

- **链接: [http://arxiv.org/pdf/2505.24820v1](http://arxiv.org/pdf/2505.24820v1)**

> **作者:** Yu Xi; Xiaoyu Gu; Haoyu Li; Jun Song; Bo Zheng; Kai Yu
>
> **摘要:** RNN-T-based keyword spotting (KWS) with autoregressive decoding~(AR) has gained attention due to its streaming architecture and superior performance. However, the simplicity of the prediction network in RNN-T poses an overfitting issue, especially under challenging scenarios, resulting in degraded performance. In this paper, we propose a masked self-distillation (MSD) training strategy that avoids RNN-Ts overly relying on prediction networks to alleviate overfitting. Such training enables masked non-autoregressive (NAR) decoding, which fully masks the RNN-T predictor output during KWS decoding. In addition, we propose a semi-autoregressive (SAR) decoding approach to integrate the advantages of AR and NAR decoding. Our experiments across multiple KWS datasets demonstrate that MSD training effectively alleviates overfitting. The SAR decoding method preserves the superior performance of AR decoding while benefits from the overfitting suppression of NAR decoding, achieving excellent results.
>
---
#### [new 014] SwitchCodec: A High-Fidelity Nerual Audio Codec With Sparse Quantization
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于高保真音频压缩任务，解决现有编解码器在极低带宽（<3kbps）下性能显著下降的问题。提出REVQ技术扩展嵌入空间，并设计策略充分利用其容量，同时引入STFT鉴别器优化频谱生成，提升压缩音频质量。**

- **链接: [http://arxiv.org/pdf/2505.24437v1](http://arxiv.org/pdf/2505.24437v1)**

> **作者:** Jin Wang; Wenbin Jiang; Xiangbo Wang
>
> **备注:** 5 pages,4 figures
>
> **摘要:** We present a universal high-fidelity neural audio compression algorithm that can compress speech, music, and general audio below 3 kbps bandwidth. Although current state-of-the-art audio codecs excel in audio compression, their effectiveness significantly declines when embedding space is sharply reduced, which corresponds to higher compression. To address this problem, we propose Residual Experts Vector Quantization (REVQ), which significantly expands the available embedding space and improves the performance while hardly sacrificing the bandwidth. Furthermore, we introduce a strategy to ensure that the vast embedding space can be fully utilized. Additionally, we propose a STFT-based discriminator to guide the generator in producing indistinguishable spectrograms. We demonstrate that the proposed approach outperforms baseline methods through detailed ablations.
>
---
#### [new 015] Pseudo Labels-based Neural Speech Enhancement for the AVSR Task in the MISP-Meeting Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对会议场景下的视听语音识别（AVSR）任务，解决强噪声、混响和重叠语音问题。提出G-SpatialNet语音增强模型，设计TLS框架生成伪标签优化训练，并结合ASR模型微调及多模态信息提升性能，获挑战赛第二名。**

- **链接: [http://arxiv.org/pdf/2505.24446v1](http://arxiv.org/pdf/2505.24446v1)**

> **作者:** Longjie Luo; Shenghui Lu; Lin Li; Qingyang Hong
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** This paper presents our system for the MISP-Meeting Challenge Track 2. The primary difficulty lies in the dataset, which contains strong background noise, reverberation, overlapping speech, and diverse meeting topics. To address these issues, we (a) designed G-SpatialNet, a speech enhancement (SE) model to improve Guided Source Separation (GSS) signals; (b) proposed TLS, a framework comprising time alignment, level alignment, and signal-to-noise ratio filtering, to generate signal-level pseudo labels for real-recorded far-field audio data, thereby facilitating SE models' training; and (c) explored fine-tuning strategies, data augmentation, and multimodal information to enhance the performance of pre-trained Automatic Speech Recognition (ASR) models in meeting scenarios. Finally, our system achieved character error rates (CERs) of 5.44% and 9.52% on the Dev and Eval sets, respectively, with relative improvements of 64.8% and 52.6% over the baseline, securing second place.
>
---
#### [new 016] When Humans Growl and Birds Speak: High-Fidelity Voice Conversion from Human to Animal and Designed Sounds
- **分类: eess.AS; cs.AI; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于人类到非人类语音转换（H2NH-VC）任务，旨在解决现有方法局限于特定动物声音及低采样率的问题。提出预处理流程与改进的CVAE模型，支持高保真转换至多样非语音声音（如狮子吼叫、鸟鸣、合成音），并提升音频质量与自然度。**

- **链接: [http://arxiv.org/pdf/2505.24336v1](http://arxiv.org/pdf/2505.24336v1)**

> **作者:** Minsu Kang; Seolhee Lee; Choonghyeon Lee; Namhyun Cho
>
> **备注:** INTERSPEECH 2025 accepted
>
> **摘要:** Human to non-human voice conversion (H2NH-VC) transforms human speech into animal or designed vocalizations. Unlike prior studies focused on dog-sounds and 16 or 22.05kHz audio transformation, this work addresses a broader range of non-speech sounds, including natural sounds (lion-roars, birdsongs) and designed voice (synthetic growls). To accomodate generation of diverse non-speech sounds and 44.1kHz high-quality audio transformation, we introduce a preprocessing pipeline and an improved CVAE-based H2NH-VC model, both optimized for human and non-human voices. Experimental results showed that the proposed method outperformed baselines in quality, naturalness, and similarity MOS, achieving effective voice conversion across diverse non-human timbres. Demo samples are available at https://nc-ai.github.io/speech/publications/nonhuman-vc/
>
---
#### [new 017] Pretraining Multi-Speaker Identification for Neural Speaker Diarization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于说话人分离任务，解决端到端说话人分离对大规模真实对话数据依赖的问题。提出通过多说话人识别预训练替代传统模拟数据预训练，利用现成的说话人识别数据集训练轻量模型，实验证明无需模拟数据即可实现高精度分离。**

- **链接: [http://arxiv.org/pdf/2505.24545v1](http://arxiv.org/pdf/2505.24545v1)**

> **作者:** Shota Horiguchi; Atsushi Ando; Marc Delcroix; Naohiro Tawara
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** End-to-end speaker diarization enables accurate overlap-aware diarization by jointly estimating multiple speakers' speech activities in parallel. This approach is data-hungry, requiring a large amount of labeled conversational data, which cannot be fully obtained from real datasets alone. To address this issue, large-scale simulated data is often used for pretraining, but it requires enormous storage and I/O capacity, and simulating data that closely resembles real conversations remains challenging. In this paper, we propose pretraining a model to identify multiple speakers from an input fully overlapped mixture as an alternative to pretraining a diarization model. This method eliminates the need to prepare a large-scale simulated dataset while leveraging large-scale speaker recognition datasets for training. Through comprehensive experiments, we demonstrate that the proposed method enables a highly accurate yet lightweight local diarization model without simulated conversational data.
>
---
#### [new 018] MELT: Towards Automated Multimodal Emotion Data Annotation by Leveraging LLM Embedded Knowledge
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出MELT方法，通过GPT-4o利用文本线索自动标注多模态情感数据，解决人工标注成本高、主观性强的问题。工作包括构建Friends电视剧的全AI标注情感数据集，并验证其在语音情感识别任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.24493v1](http://arxiv.org/pdf/2505.24493v1)**

> **作者:** Xin Jing; Jiadong Wang; Iosif Tsangko; Andreas Triantafyllopoulos; Björn W. Schuller
>
> **摘要:** Although speech emotion recognition (SER) has advanced significantly with deep learning, annotation remains a major hurdle. Human annotation is not only costly but also subject to inconsistencies annotators often have different preferences and may lack the necessary contextual knowledge, which can lead to varied and inaccurate labels. Meanwhile, Large Language Models (LLMs) have emerged as a scalable alternative for annotating text data. However, the potential of LLMs to perform emotional speech data annotation without human supervision has yet to be thoroughly investigated. To address these problems, we apply GPT-4o to annotate a multimodal dataset collected from the sitcom Friends, using only textual cues as inputs. By crafting structured text prompts, our methodology capitalizes on the knowledge GPT-4o has accumulated during its training, showcasing that it can generate accurate and contextually relevant annotations without direct access to multimodal inputs. Therefore, we propose MELT, a multimodal emotion dataset fully annotated by GPT-4o. We demonstrate the effectiveness of MELT by fine-tuning four self-supervised learning (SSL) backbones and assessing speech emotion recognition performance across emotion datasets. Additionally, our subjective experiments\' results demonstrate a consistence performance improvement on SER.
>
---
#### [new 019] Dynamic Context-Aware Streaming Pretrained Language Model For Inverse Text Normalization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于逆文本规范化（ITN）任务，旨在解决流式ASR系统中实时ITN的精度、效率与适应性问题，尤其针对低资源场景。提出动态上下文感知的流式预训练语言模型，通过自适应调整分块大小和利用右文信息，在越南语数据集上实现与非流式方法相当的准确率，同时保持低延迟。**

- **链接: [http://arxiv.org/pdf/2505.24229v1](http://arxiv.org/pdf/2505.24229v1)**

> **作者:** Luong Ho; Khanh Le; Vinh Pham; Bao Nguyen; Tan Tran; Duc Chau
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** Inverse Text Normalization (ITN) is crucial for converting spoken Automatic Speech Recognition (ASR) outputs into well-formatted written text, enhancing both readability and usability. Despite its importance, the integration of streaming ITN within streaming ASR remains largely unexplored due to challenges in accuracy, efficiency, and adaptability, particularly in low-resource and limited-context scenarios. In this paper, we introduce a streaming pretrained language model for ITN, leveraging pretrained linguistic representations for improved robustness. To address streaming constraints, we propose Dynamic Context-Aware during training and inference, enabling adaptive chunk size adjustments and the integration of right-context information. Experimental results demonstrate that our method achieves accuracy comparable to non-streaming ITN and surpasses existing streaming ITN models on a Vietnamese dataset, all while maintaining low latency, ensuring seamless integration into ASR systems.
>
---
#### [new 020] Probing the Robustness Properties of Neural Speech Codecs
- **分类: eess.AS; cs.SD**

- **简介: 该论文评估神经语音编解码器在噪声环境中的鲁棒性，解决其实际应用中的可靠性问题。通过系统测试噪声条件下的表现、分析线性特性和频率响应，揭示鲁棒性差异及非线性失真影响，为优化设计提供依据。**

- **链接: [http://arxiv.org/pdf/2505.24248v1](http://arxiv.org/pdf/2505.24248v1)**

> **作者:** Wei-Cheng Tseng; David Harwath
>
> **备注:** Interspeech 2025
>
> **摘要:** Neural speech codecs have revolutionized speech coding, achieving higher compression while preserving audio fidelity. Beyond compression, they have emerged as tokenization strategies, enabling language modeling on speech and driving paradigm shifts across various speech processing tasks. Despite these advancements, their robustness in noisy environments remains underexplored, raising concerns about their generalization to real-world scenarios. In this work, we systematically evaluate neural speech codecs under various noise conditions, revealing non-trivial differences in their robustness. We further examine their linearity properties, uncovering non-linear distortions which partly explain observed variations in robustness. Lastly, we analyze their frequency response to identify factors affecting audio fidelity. Our findings provide critical insights into codec behavior and future codec design, as well as emphasizing the importance of noise robustness for their real-world integration.
>
---
#### [new 021] More-than-Human Storytelling: Designing Longitudinal Narrative Engagements with Generative AI
- **分类: cs.HC; cs.AI; cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于生成式AI叙事系统设计任务，探索人机长期互动中的动态关系。通过开发"Dreamsmithy"应用，让28名参与者连续两周与AI" Makoto"共创故事，研究发现用户在情感依赖与控制矛盾中形成独特纽带，但存在叙事连贯性问题。研究为设计可持续人机叙事系统提供了伦理与交互设计建议。**

- **链接: [http://arxiv.org/pdf/2505.23780v1](http://arxiv.org/pdf/2505.23780v1)**

> **作者:** Émilie Fabre; Katie Seaborn; Shuta Koiwai; Mizuki Watanabe; Paul Riesch
>
> **备注:** CHI EA '25
>
> **摘要:** Longitudinal engagement with generative AI (GenAI) storytelling agents is a timely but less charted domain. We explored multi-generational experiences with "Dreamsmithy," a daily dream-crafting app, where participants (N = 28) co-created stories with AI narrator "Makoto" every day. Reflections and interactions were captured through a two-week diary study. Reflexive thematic analysis revealed themes likes "oscillating ambivalence" and "socio-chronological bonding," highlighting the complex dynamics that emerged between individuals and the AI narrator over time. Findings suggest that while people appreciated the personal notes, opportunities for reflection, and AI creativity, limitations in narrative coherence and control occasionally caused frustration. The results underscore the potential of GenAI for longitudinal storytelling, but also raise critical questions about user agency and ethics. We contribute initial empirical insights and design considerations for developing adaptive, more-than-human storytelling systems.
>
---
#### [new 022] MSDA: Combining Pseudo-labeling and Self-Supervision for Unsupervised Domain Adaptation in ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于无监督领域自适应（UDA）在语音识别（ASR）任务，旨在解决目标领域缺乏标注数据时模型适应性差的问题。提出MSDA框架，通过两阶段结合自监督学习与伪标签半监督方法，提升ASR在低资源语言（如希腊语）和弱监督场景下的鲁棒性，实验显示其效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.24656v1](http://arxiv.org/pdf/2505.24656v1)**

> **作者:** Dimitrios Damianos; Georgios Paraskevopoulos; Alexandros Potamianos
>
> **摘要:** In this work, we investigate the Meta PL unsupervised domain adaptation framework for Automatic Speech Recognition (ASR). We introduce a Multi-Stage Domain Adaptation pipeline (MSDA), a sample-efficient, two-stage adaptation approach that integrates self-supervised learning with semi-supervised techniques. MSDA is designed to enhance the robustness and generalization of ASR models, making them more adaptable to diverse conditions. It is particularly effective for low-resource languages like Greek and in weakly supervised scenarios where labeled data is scarce or noisy. Through extensive experiments, we demonstrate that Meta PL can be applied effectively to ASR tasks, achieving state-of-the-art results, significantly outperforming state-of-the-art methods, and providing more robust solutions for unsupervised domain adaptation in ASR. Our ablations highlight the necessity of utilizing a cascading approach when combining self-supervision with self-training.
>
---
#### [new 023] Voice Conversion Improves Cross-Domain Robustness for Spoken Arabic Dialect Identification
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语方言识别（ADI）任务，针对模型在跨领域语音上的泛化能力差问题，提出基于声纹转换的训练方法，提升跨域鲁棒性。实验在新测试集实现最高34.1%准确率提升，并缓解说话人偏置，开源模型与数据集支持技术研发。**

- **链接: [http://arxiv.org/pdf/2505.24713v1](http://arxiv.org/pdf/2505.24713v1)**

> **作者:** Badr M. Abdullah; Matthew Baas; Bernd Möbius; Dietrich Klakow
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** Arabic dialect identification (ADI) systems are essential for large-scale data collection pipelines that enable the development of inclusive speech technologies for Arabic language varieties. However, the reliability of current ADI systems is limited by poor generalization to out-of-domain speech. In this paper, we present an effective approach based on voice conversion for training ADI models that achieves state-of-the-art performance and significantly improves robustness in cross-domain scenarios. Evaluated on a newly collected real-world test set spanning four different domains, our approach yields consistent improvements of up to +34.1% in accuracy across domains. Furthermore, we present an analysis of our approach and demonstrate that voice conversion helps mitigate the speaker bias in the ADI dataset. We release our robust ADI model and cross-domain evaluation dataset to support the development of inclusive speech technologies for Arabic.
>
---
#### [new 024] Digital twins enable full-reference quality assessment of photoacoustic image reconstructions
- **分类: physics.med-ph; cs.CV; eess.SP**

- **简介: 论文提出数字孪生框架用于光声成像重建算法的全参考质量评估。解决实际中缺乏理想参考图像的问题，通过构建组织和成像系统的数字孪生进行校准。工作包括比较多种算法，并首次实验验证傅里叶方法，显示其效果与迭代法相当但计算成本更低。**

- **链接: [http://arxiv.org/pdf/2505.24514v1](http://arxiv.org/pdf/2505.24514v1)**

> **作者:** Janek Gröhl; Leonid Kunyansky; Jenni Poimala; Thomas R. Else; Francesca Di Cecio; Sarah E. Bohndiek; Ben T. Cox; Andreas Hauptmann
>
> **摘要:** Quantitative comparison of the quality of photoacoustic image reconstruction algorithms remains a major challenge. No-reference image quality measures are often inadequate, but full-reference measures require access to an ideal reference image. While the ground truth is known in simulations, it is unknown in vivo, or in phantom studies, as the reference depends on both the phantom properties and the imaging system. We tackle this problem by using numerical digital twins of tissue-mimicking phantoms and the imaging system to perform a quantitative calibration to reduce the simulation gap. The contributions of this paper are two-fold: First, we use this digital-twin framework to compare multiple state-of-the-art reconstruction algorithms. Second, among these is a Fourier transform-based reconstruction algorithm for circular detection geometries, which we test on experimental data for the first time. Our results demonstrate the usefulness of digital phantom twins by enabling assessment of the accuracy of the numerical forward model and enabling comparison of image reconstruction schemes with full-reference image quality assessment. We show that the Fourier transform-based algorithm yields results comparable to those of iterative time reversal, but at a lower computational cost. All data and code are publicly available on Zenodo: https://doi.org/10.5281/zenodo.15388429.
>
---
#### [new 025] Can Emotion Fool Anti-spoofing?
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音反欺骗任务，针对现有模型对情感化合成语音防御不足的问题，构建EmoSpoof-TTS数据集，揭示传统方法在情感语音上的脆弱性，并提出GEM方法：通过情感识别门控集成多情感专精模型，提升多情感场景下的防欺骗性能，同时发布数据集促进研究。**

- **链接: [http://arxiv.org/pdf/2505.23962v1](http://arxiv.org/pdf/2505.23962v1)**

> **作者:** Aurosweta Mahapatra; Ismail Rasim Ulgen; Abinay Reddy Naini; Carlos Busso; Berrak Sisman
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Traditional anti-spoofing focuses on models and datasets built on synthetic speech with mostly neutral state, neglecting diverse emotional variations. As a result, their robustness against high-quality, emotionally expressive synthetic speech is uncertain. We address this by introducing EmoSpoof-TTS, a corpus of emotional text-to-speech samples. Our analysis shows existing anti-spoofing models struggle with emotional synthetic speech, exposing risks of emotion-targeted attacks. Even trained on emotional data, the models underperform due to limited focus on emotional aspect and show performance disparities across emotions. This highlights the need for emotion-focused anti-spoofing paradigm in both dataset and methodology. We propose GEM, a gated ensemble of emotion-specialized models with a speech emotion recognition gating network. GEM performs effectively across all emotions and neutral state, improving defenses against spoofing attacks. We release the EmoSpoof-TTS Dataset: https://emospoof-tts.github.io/Dataset/
>
---
#### [new 026] SpeechVerifier: Robust Acoustic Fingerprint against Tampering Attacks via Watermarking
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文提出SpeechVerifier，通过多尺度特征提取与对比学习生成抗篡改的音频指纹，并嵌入水印实现自包含语音完整性验证。旨在解决现有方法依赖外部参考或鲁棒性不足的问题，可有效检测恶意篡改且抵御压缩等正常操作。**

- **链接: [http://arxiv.org/pdf/2505.23821v1](http://arxiv.org/pdf/2505.23821v1)**

> **作者:** Lingfeng Yao; Chenpei Huang; Shengyao Wang; Junpei Xue; Hanqing Guo; Jiang Liu; Xun Chen; Miao Pan
>
> **摘要:** With the surge of social media, maliciously tampered public speeches, especially those from influential figures, have seriously affected social stability and public trust. Existing speech tampering detection methods remain insufficient: they either rely on external reference data or fail to be both sensitive to attacks and robust to benign operations, such as compression and resampling. To tackle these challenges, we introduce SpeechVerifer to proactively verify speech integrity using only the published speech itself, i.e., without requiring any external references. Inspired by audio fingerprinting and watermarking, SpeechVerifier can (i) effectively detect tampering attacks, (ii) be robust to benign operations and (iii) verify the integrity only based on published speeches. Briefly, SpeechVerifier utilizes multiscale feature extraction to capture speech features across different temporal resolutions. Then, it employs contrastive learning to generate fingerprints that can detect modifications at varying granularities. These fingerprints are designed to be robust to benign operations, but exhibit significant changes when malicious tampering occurs. To enable speech verification in a self-contained manner, the generated fingerprints are then embedded into the speech signal by segment-wise watermarking. Without external references, SpeechVerifier can retrieve the fingerprint from the published audio and check it with the embedded watermark to verify the integrity of the speech. Extensive experimental results demonstrate that the proposed SpeechVerifier is effective in detecting tampering attacks and robust to benign operations.
>
---
#### [new 027] A Perception-Based L2 Speech Intelligibility Indicator: Leveraging a Rater's Shadowing and Sequence-to-sequence Voice Conversion
- **分类: eess.AS; cs.SD**

- **简介: 论文提出基于感知的L2语音可懂度评估方法，解决传统ASR方法侧重母语相似性而忽略人类实际理解的问题。通过整合母语者影子跟读数据与序列到序列语音转换框架，模拟听觉感知并定位理解困难片段，评估结果更贴近母语者判断，适用于多语言CALL系统。**

- **链接: [http://arxiv.org/pdf/2505.24304v1](http://arxiv.org/pdf/2505.24304v1)**

> **作者:** Haopeng Geng; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Evaluating L2 speech intelligibility is crucial for effective computer-assisted language learning (CALL). Conventional ASR-based methods often focus on native-likeness, which may fail to capture the actual intelligibility perceived by human listeners. In contrast, our work introduces a novel, perception based L2 speech intelligibility indicator that leverages a native rater's shadowing data within a sequence-to-sequence (seq2seq) voice conversion framework. By integrating an alignment mechanism and acoustic feature reconstruction, our approach simulates the auditory perception of native listeners, identifying segments in L2 speech that are likely to cause comprehension difficulties. Both objective and subjective evaluations indicate that our method aligns more closely with native judgments than traditional ASR-based metrics, offering a promising new direction for CALL systems in a global, multilingual contexts.
>
---
#### [new 028] Speech-to-Text Translation with Phoneme-Augmented CoT: Enhancing Cross-Lingual Transfer in Low-Resource Scenarios
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到文本翻译（S2TT）任务，旨在解决低资源及零资源场景下的翻译质量不足问题。提出结合音素表示的Chain-of-Thought（CoT）框架，通过引入音素识别作为中间步骤，利用多语言LLM处理语音及音素，并采用课程学习策略，提升跨语言迁移能力，实现低资源条件下翻译质量提升及零资源翻译支持。**

- **链接: [http://arxiv.org/pdf/2505.24691v1](http://arxiv.org/pdf/2505.24691v1)**

> **作者:** Gerard I. Gállego; Oriol Pareras; Martí Cortada Garcia; Lucas Takanori; Javier Hernando
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** We propose a Speech-to-Text Translation (S2TT) approach that integrates phoneme representations into a Chain-of-Thought (CoT) framework to improve translation in low-resource and zero-resource settings. By introducing phoneme recognition as an intermediate step, we enhance cross-lingual transfer, enabling translation even for languages with no labeled speech data. Our system builds on a multilingual LLM, which we extend to process speech and phonemes. Training follows a curriculum learning strategy that progressively introduces more complex tasks. Experiments on multilingual S2TT benchmarks show that phoneme-augmented CoT improves translation quality in low-resource conditions and enables zero-resource translation, while slightly impacting high-resource performance. Despite this trade-off, our findings demonstrate that phoneme-based CoT is a promising step toward making S2TT more accessible across diverse languages.
>
---
#### [new 029] Identifying Primary Stress Across Related Languages and Dialects with Transformer-based Speech Encoder Models
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究跨语言/方言的主要重音自动识别任务。针对传统方法依赖英语数据和声学特征的问题，提出基于Transformer模型微调的方法。实验使用新克罗地亚数据集及多语言测试集，对比SVM显示Transformer显著更优（克罗地亚/塞尔维亚接近完美，斯洛文尼亚等降10%），并验证仅需几百训练词即可高效建模，同时公开数据与模型。**

- **链接: [http://arxiv.org/pdf/2505.24571v1](http://arxiv.org/pdf/2505.24571v1)**

> **作者:** Nikola Ljubešić; Ivan Porupski; Peter Rupnik
>
> **备注:** Accepted to InterSpeech2025
>
> **摘要:** Automating primary stress identification has been an active research field due to the role of stress in encoding meaning and aiding speech comprehension. Previous studies relied mainly on traditional acoustic features and English datasets. In this paper, we investigate the approach of fine-tuning a pre-trained transformer model with an audio frame classification head. Our experiments use a new Croatian training dataset, with test sets in Croatian, Serbian, the Chakavian dialect, and Slovenian. By comparing an SVM classifier using traditional acoustic features with the fine-tuned speech transformer, we demonstrate the transformer's superiority across the board, achieving near-perfect results for Croatian and Serbian, with a 10-point performance drop for the more distant Chakavian and Slovenian. Finally, we show that only a few hundred multi-syllabic training words suffice for strong performance. We release our datasets and model under permissive licenses.
>
---
#### [new 030] Efficient Estimation of Regularized Tyler's M-Estimator Using Approximate LOOCV
- **分类: stat.ML; cs.CE; cs.CV; cs.LG; eess.SP; I.2.0; I.2.6**

- **简介: 该论文提出一种高效近似LOOCV方法，用于Regularized Tyler's M估计器的正则化参数估计。针对传统LOOCV计算成本高的问题，通过近似降低时间复杂度，实现快速准确的参数选择，实验验证其在高维数据上的有效性。**

- **链接: [http://arxiv.org/pdf/2505.24781v1](http://arxiv.org/pdf/2505.24781v1)**

> **作者:** Karim Abou-Moustafa
>
> **备注:** An extended version of a short article that appeared in 2023 IEEE Workshop on Information Theory, Saint-Malo, France
>
> **摘要:** We consider the problem of estimating a regularization parameter, or a shrinkage coefficient $\alpha \in (0,1)$ for Regularized Tyler's M-estimator (RTME). In particular, we propose to estimate an optimal shrinkage coefficient by setting $\alpha$ as the solution to a suitably chosen objective function; namely the leave-one-out cross-validated (LOOCV) log-likelihood loss. Since LOOCV is computationally prohibitive even for moderate sample size $n$, we propose a computationally efficient approximation for the LOOCV log-likelihood loss that eliminates the need for invoking the RTME procedure $n$ times for each sample left out during the LOOCV procedure. This approximation yields an $O(n)$ reduction in the running time complexity for the LOOCV procedure, which results in a significant speedup for computing the LOOCV estimate. We demonstrate the efficiency and accuracy of the proposed approach on synthetic high-dimensional data sampled from heavy-tailed elliptical distributions, as well as on real high-dimensional datasets for object recognition, face recognition, and handwritten digit's recognition. Our experiments show that the proposed approach is efficient and consistently more accurate than other methods in the literature for shrinkage coefficient estimation.
>
---
## 更新

#### [replaced 001] Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14449v3](http://arxiv.org/pdf/2505.14449v3)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by InterSpeech 2025. 7 pages including 2 pages of appendix
>
> **摘要:** While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 28% with less than a 2% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 4.6% improvement in fairness metrics with a drop of less than 3.6% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential when explicit demographic information is unavailable.
>
---
#### [replaced 002] Versatile Framework for Song Generation with Prompt-based Control
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.19062v3](http://arxiv.org/pdf/2504.19062v3)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Ruiqi Li; Jingyu Lu; Rongjie Huang; Ruiyuan Zhang; Zhiqing Hong; Ziyue Jiang; Zhou Zhao
>
> **摘要:** Song generation focuses on producing controllable high-quality songs based on various prompts. However, existing methods struggle to generate vocals and accompaniments with prompt-based control and proper alignment. Additionally, they fall short in supporting various tasks. To address these challenges, we introduce VersBand, a multi-task song generation framework for synthesizing high-quality, aligned songs with prompt-based control. VersBand comprises these primary models: 1) VocalBand, a decoupled model, leverages the flow-matching method for generating singing styles, pitches, and mel-spectrograms, allowing fast, high-quality vocal generation with style control. 2) AccompBand, a flow-based transformer model, incorporates the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control. This model allows for generating controllable, high-quality accompaniments aligned with vocals. 3) Two generation models, LyricBand for lyrics and MelodyBand for melodies, contribute to the comprehensive multi-task song generation system, allowing for extensive control based on multiple prompts. Experimental results demonstrate that VersBand performs better over baseline models across multiple song generation tasks using objective and subjective metrics. Audio samples are available at https://aaronz345.github.io/VersBandDemo.
>
---
#### [replaced 003] VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21568v2](http://arxiv.org/pdf/2505.21568v2)**

> **作者:** Haiyun Li; Zhiyong Wu; Xiaofeng Xie; Jingran Xie; Yaoxun Xu; Hanyang Peng
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Voice cloning (VC)-resistant watermarking is an emerging technique for tracing and preventing unauthorized cloning. Existing methods effectively trace traditional VC models by training them on watermarked audio but fail in zero-shot VC scenarios, where models synthesize audio from an audio prompt without training. To address this, we propose VoiceMark, the first zero-shot VC-resistant watermarking method that leverages speaker-specific latents as the watermark carrier, allowing the watermark to transfer through the zero-shot VC process into the synthesized audio. Additionally, we introduce VC-simulated augmentations and VAD-based loss to enhance robustness against distortions. Experiments on multiple zero-shot VC models demonstrate that VoiceMark achieves over 95% accuracy in watermark detection after zero-shot VC synthesis, significantly outperforming existing methods, which only reach around 50%. See our code and demos at: https://huggingface.co/spaces/haiyunli/VoiceMark
>
---
#### [replaced 004] SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning
- **分类: eess.SP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19668v2](http://arxiv.org/pdf/2502.19668v2)**

> **作者:** Mingsheng Cai; Jiuming Jiang; Wenhao Huang; Che Liu; Rossella Arcucci
>
> **摘要:** Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) is critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME is pre-trained using structured diagnostic labels derived from ECG report entities through a one-time offline extraction with Large Language Models (LLMs), which help denoise, standardize cardiac concepts, and improve clinical representation learning. By fusing ECG signals with textual cardiac queries instead of fixed labels, SuPreME enables zero-shot classification of unseen conditions without further fine-tuning. We evaluate SuPreME on six downstream datasets covering 106 cardiac conditions, achieving superior zero-shot AUC performance of $77.20\%$, surpassing state-of-the-art eSSLs by $4.98\%$. Results demonstrate SuPreME's effectiveness in leveraging structured, clinically relevant knowledge for high-quality ECG representations.
>
---
#### [replaced 005] SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13128v2](http://arxiv.org/pdf/2502.13128v2)**

> **作者:** Zihan Liu; Shuangrui Ding; Zhixiong Zhang; Xiaoyi Dong; Pan Zhang; Yuhang Zang; Yuhang Cao; Dahua Lin; Jiaqi Wang
>
> **摘要:** Text-to-song generation, the task of creating vocals and accompaniment from textual inputs, poses significant challenges due to domain complexity and data scarcity. Existing approaches often employ multi-stage generation procedures, leading to cumbersome training and inference pipelines, as well as suboptimal overall generation quality due to error accumulation across stages. In this paper, we propose SongGen, a fully open-source, single-stage auto-regressive transformer designed for controllable song generation. The proposed model facilitates fine-grained control over diverse musical attributes, including lyrics and textual descriptions of instrumentation, genre, mood, and timbre, while also offering an optional three-second reference clip for voice cloning. Within a unified auto-regressive framework, SongGen supports two output modes: mixed mode, which generates a mixture of vocals and accompaniment directly, and dual-track mode, which synthesizes them separately for greater flexibility in downstream applications. We explore diverse token pattern strategies for each mode, leading to notable improvements and valuable insights. Furthermore, we design an automated data preprocessing pipeline with effective quality control. To foster community engagement and future research, we will release our model weights, training code, annotated data, and preprocessing pipeline. The code is available at https://github.com/LiuZH-19/SongGen.
>
---
#### [replaced 006] Breaking Resource Barriers in Speech Emotion Recognition via Data Distillation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.15119v2](http://arxiv.org/pdf/2406.15119v2)**

> **作者:** Yi Chang; Zhao Ren; Zhonghao Zhao; Thanh Tam Nguyen; Kun Qian; Tanja Schultz; Björn W. Schuller
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Speech emotion recognition (SER) plays a crucial role in human-computer interaction. The emergence of edge devices in the Internet of Things (IoT) presents challenges in constructing intricate deep learning models due to constraints in memory and computational resources. Moreover, emotional speech data often contains private information, raising concerns about privacy leakage during the deployment of SER models. To address these challenges, we propose a data distillation framework to facilitate efficient development of SER models in IoT applications using a synthesised, smaller, and distilled dataset. Our experiments demonstrate that the distilled dataset can be effectively utilised to train SER models with fixed initialisation, achieving performances comparable to those developed using the original full emotional speech dataset.
>
---
#### [replaced 007] ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation
- **分类: cs.SD; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07217v2](http://arxiv.org/pdf/2503.07217v2)**

> **作者:** Zixuan Wang; Chi-Keung Tang; Yu-Wing Tai
>
> **摘要:** Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
>
---
#### [replaced 008] Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications
- **分类: eess.SP; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18194v2](http://arxiv.org/pdf/2505.18194v2)**

> **作者:** Yubo Peng; Luping Xiang; Bingxin Zhang; Kun Yang
>
> **摘要:** Traditional single-modal sensing systems-based solely on either radio frequency (RF) or visual data-struggle to cope with the demands of complex and dynamic environments. Furthermore, single-device systems are constrained by limited perspectives and insufficient spatial coverage, which impairs their effectiveness in urban or non-line-of-sight scenarios. To overcome these challenges, we propose a novel large language model (LLM)-driven distributed integrated multimodal sensing and semantic communication (LLM-DiSAC) framework. Specifically, our system consists of multiple collaborative sensing devices equipped with RF and camera modules, working together with an aggregation center to enhance sensing accuracy. First, on sensing devices, LLM-DiSAC develops an RF-vision fusion network (RVFN), which employs specialized feature extractors for RF and visual data, followed by a cross-attention module for effective multimodal integration. Second, a LLM-based semantic transmission network (LSTN) is proposed to enhance communication efficiency, where the LLM-based decoder leverages known channel parameters, such as transceiver distance and signal-to-noise ratio (SNR), to mitigate semantic distortion. Third, at the aggregation center, a transformer-based aggregation model (TRAM) with an adaptive aggregation attention mechanism is developed to fuse distributed features and enhance sensing accuracy. To preserve data privacy, a two-stage distributed learning strategy is introduced, allowing local model training at the device level and centralized aggregation model training using intermediate features. Finally, evaluations on a synthetic multi-view RF-visual dataset generated by the Genesis simulation engine show that LLM-DiSAC achieves a good performance.
>
---
#### [replaced 009] Towards Robust Assessment of Pathological Voices via Combined Low-Level Descriptors and Foundation Model Representations
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21356v3](http://arxiv.org/pdf/2505.21356v3)**

> **作者:** Whenty Ariyanti; Kuan-Yu Chen; Sabato Marco Siniscalchi; Hsin-Min Wang; Yu Tsao
>
> **摘要:** Perceptual voice quality assessment is essential for diagnosing and monitoring voice disorders by providing standardized evaluations of vocal function. Traditionally, expert raters use standard scales such as the Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V) and Grade, Roughness, Breathiness, Asthenia, and Strain (GRBAS). However, these metrics are subjective and prone to inter-rater variability, motivating the need for automated, objective assessment methods. This study proposes Voice Quality Assessment Network (VOQANet), a deep learning-based framework with an attention mechanism that leverages a Speech Foundation Model (SFM) to extract high-level acoustic and prosodic information from raw speech. To enhance robustness and interpretability, we also introduce VOQANet+, which integrates low-level speech descriptors such as jitter, shimmer, and harmonics-to-noise ratio (HNR) with SFM embeddings into a hybrid representation. Unlike prior studies focused only on vowel-based phonation (PVQD-A subset) of the Perceptual Voice Quality Dataset (PVQD), we evaluate our models on both vowel-based and sentence-level speech (PVQD-S subset) to improve generalizability. Results show that sentence-based input outperforms vowel-based input, especially at the patient level, underscoring the value of longer utterances for capturing perceptual voice attributes. VOQANet consistently surpasses baseline methods in root mean squared error (RMSE) and Pearson correlation coefficient (PCC) across CAPE-V and GRBAS dimensions, with VOQANet+ achieving even better performance. Additional experiments under noisy conditions show that VOQANet+ maintains high prediction accuracy and robustness, supporting its potential for real-world and telehealth deployment.
>
---
#### [replaced 010] V2SFlow: Video-to-Speech Generation with Speech Decomposition and Rectified Flow
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.19486v2](http://arxiv.org/pdf/2411.19486v2)**

> **作者:** Jeongsoo Choi; Ji-Hoon Kim; Jinyu Li; Joon Son Chung; Shujie Liu
>
> **备注:** ICASSP 2025
>
> **摘要:** In this paper, we introduce V2SFlow, a novel Video-to-Speech (V2S) framework designed to generate natural and intelligible speech directly from silent talking face videos. While recent V2S systems have shown promising results on constrained datasets with limited speakers and vocabularies, their performance often degrades on real-world, unconstrained datasets due to the inherent variability and complexity of speech signals. To address these challenges, we decompose the speech signal into manageable subspaces (content, pitch, and speaker information), each representing distinct speech attributes, and predict them directly from the visual input. To generate coherent and realistic speech from these predicted attributes, we employ a rectified flow matching decoder built on a Transformer architecture, which models efficient probabilistic pathways from random noise to the target speech distribution. Extensive experiments demonstrate that V2SFlow significantly outperforms state-of-the-art methods, even surpassing the naturalness of ground truth utterances. Code and models are available at: https://github.com/kaistmm/V2SFlow
>
---
#### [replaced 011] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v3](http://arxiv.org/pdf/2505.14874v3)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 012] Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; 68T10; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.17076v2](http://arxiv.org/pdf/2505.17076v2)**

> **作者:** Haoyang Zhang; Hexin Liu; Xiangyu Zhang; Qiquan Zhang; Yuchen Hu; Junqi Zhao; Fei Tian; Xuerui Yang; Eng Siong Chng
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
>
---
#### [replaced 013] MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19577v2](http://arxiv.org/pdf/2505.19577v2)**

> **作者:** Yu Xi; Haoyu Li; Xiaoyu Gu; Yidi Jiang; Kai Yu
>
> **备注:** TASLP under review
>
> **摘要:** Keyword spotting (KWS) is essential for voice-driven applications, demanding both accuracy and efficiency. Traditional ASR-based KWS methods, such as greedy and beam search, explore the entire search space without explicitly prioritizing keyword detection, often leading to suboptimal performance. In this paper, we propose an effective keyword-specific KWS framework by introducing a streaming-oriented CTC-Transducer-combined frame-asynchronous system with multi-head frame-asynchronous decoding (MFA-KWS). Specifically, MFA-KWS employs keyword-specific phone-synchronous decoding for CTC and replaces conventional RNN-T with Token-and-Duration Transducer to enhance both performance and efficiency. Furthermore, we explore various score fusion strategies, including single-frame-based and consistency-based methods. Extensive experiments demonstrate the superior performance of MFA-KWS, which achieves state-of-the-art results on both fixed keyword and arbitrary keywords datasets, such as Snips, MobvoiHotwords, and LibriKWS-20, while exhibiting strong robustness in noisy environments. Among fusion strategies, the consistency-based CDC-Last method delivers the best performance. Additionally, MFA-KWS achieves a 47% to 63% speed-up over the frame-synchronous baselines across various datasets. Extensive experimental results confirm that MFA-KWS is an effective and efficient KWS framework, making it well-suited for on-device deployment.
>
---
#### [replaced 014] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14910v3](http://arxiv.org/pdf/2505.14910v3)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Dongyu Yao; Zhiyuan Zhu; Ziyue Jiang; Yuhan Wang; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks. Singing voice samples are available at https://aaronz345.github.io/TCSinger2Demo/.
>
---
#### [replaced 015] EASY: Emotion-aware Speaker Anonymization via Factorized Distillation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.15004v2](http://arxiv.org/pdf/2505.15004v2)**

> **作者:** Jixun Yao; Hexin Liu; Eng Siong Chng; Lei Xie
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Emotion plays a significant role in speech interaction, conveyed through tone, pitch, and rhythm, enabling the expression of feelings and intentions beyond words to create a more personalized experience. However, most existing speaker anonymization systems employ parallel disentanglement methods, which only separate speech into linguistic content and speaker identity, often neglecting the preservation of the original emotional state. In this study, we introduce EASY, an emotion-aware speaker anonymization framework. EASY employs a novel sequential disentanglement process to disentangle speaker identity, linguistic content, and emotional representation, modeling each speech attribute in distinct subspaces through a factorized distillation approach. By independently constraining speaker identity and emotional representation, EASY minimizes information leakage, enhancing privacy protection while preserving original linguistic content and emotional state. Experimental results on the VoicePrivacy Challenge official datasets demonstrate that our proposed approach outperforms all baseline systems, effectively protecting speaker privacy while maintaining linguistic content and emotional state.
>
---
#### [replaced 016] Dialectal Coverage And Generalization in Arabic Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.05872v3](http://arxiv.org/pdf/2411.05872v3)**

> **作者:** Amirbek Djanibekov; Hawau Olamide Toyin; Raghad Alshalan; Abdullah Alitr; Hanan Aldarmaki
>
> **摘要:** Developing robust automatic speech recognition (ASR) systems for Arabic requires effective strategies to manage its diversity. Existing ASR systems mainly cover the modern standard Arabic (MSA) variety and few high-resource dialects, but fall short in coverage and generalization across the multitude of spoken variants. Code-switching with English and French is also common in different regions of the Arab world, which challenges the performance of monolingual Arabic models. In this work, we introduce a suite of ASR models optimized to effectively recognize multiple variants of spoken Arabic, including MSA, various dialects, and code-switching. We provide open-source pre-trained models that cover data from 17 Arabic-speaking countries, and fine-tuned MSA and dialectal ASR models that include at least 11 variants, as well as multi-lingual ASR models covering embedded languages in code-switched utterances. We evaluate ASR performance across these spoken varieties and demonstrate both coverage and performance gains compared to prior models.
>
---
#### [replaced 017] Accelerating Diffusion-based Text-to-Speech Model Training with Dual Modality Alignment
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19595v2](http://arxiv.org/pdf/2505.19595v2)**

> **作者:** Jeongsoo Choi; Zhikang Niu; Ji-Hoon Kim; Chunhui Wang; Joon Son Chung; Xie Chen
>
> **备注:** Interspeech 2025
>
> **摘要:** The goal of this paper is to optimize the training process of diffusion-based text-to-speech models. While recent studies have achieved remarkable advancements, their training demands substantial time and computational costs, largely due to the implicit guidance of diffusion models in learning complex intermediate representations. To address this, we propose A-DMA, an effective strategy for Accelerating training with Dual Modality Alignment. Our method introduces a novel alignment pipeline leveraging both text and speech modalities: text-guided alignment, which incorporates contextual representations, and speech-guided alignment, which refines semantic representations. By aligning hidden states with discriminative features, our training scheme reduces the reliance on diffusion models for learning complex representations. Extensive experiments demonstrate that A-DMA doubles the convergence speed while achieving superior performance over baselines. Code and demo samples are available at: https://github.com/ZhikangNiu/A-DMA
>
---
#### [replaced 018] StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2312.10741v5](http://arxiv.org/pdf/2312.10741v5)**

> **作者:** Yu Zhang; Rongjie Huang; Ruiqi Li; JinZheng He; Yan Xia; Feiyang Chen; Xinyu Duan; Baoxing Huai; Zhou Zhao
>
> **备注:** Accepted by AAAI 2024
>
> **摘要:** Style transfer for out-of-domain (OOD) singing voice synthesis (SVS) focuses on generating high-quality singing voices with unseen styles (such as timbre, emotion, pronunciation, and articulation skills) derived from reference singing voice samples. However, the endeavor to model the intricate nuances of singing voice styles is an arduous task, as singing voices possess a remarkable degree of expressiveness. Moreover, existing SVS methods encounter a decline in the quality of synthesized singing voices in OOD scenarios, as they rest upon the assumption that the target vocal attributes are discernible during the training phase. To overcome these challenges, we propose StyleSinger, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference singing voice samples. StyleSinger incorporates two critical approaches for enhanced effectiveness: 1) the Residual Style Adaptor (RSA) which employs a residual quantization module to capture diverse style characteristics in singing voices, and 2) the Uncertainty Modeling Layer Normalization (UMLN) to perturb the style attributes within the content representation during the training phase and thus improve the model generalization. Our extensive evaluations in zero-shot style transfer undeniably establish that StyleSinger outperforms baseline models in both audio quality and similarity to the reference singing voice samples. Access to singing voice samples can be found at https://aaronz345.github.io/StyleSingerDemo/.
>
---
#### [replaced 019] ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting
- **分类: eess.AS; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.20630v2](http://arxiv.org/pdf/2504.20630v2)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Tao Jin; Zhou Zhao
>
> **摘要:** Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating Drama-MOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos and dataset are available at https://aaronz345.github.io/ISDramaDemo.
>
---
#### [replaced 020] Automatic classification of stop realisation with wav2vec2.0
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.23688v2](http://arxiv.org/pdf/2505.23688v2)**

> **作者:** James Tanner; Morgan Sonderegger; Jane Stuart-Smith; Jeff Mielke; Tyler Kendall
>
> **备注:** Accepted for Interspeech 2025. 5 pages, 3 figures
>
> **摘要:** Modern phonetic research regularly makes use of automatic tools for the annotation of speech data, however few tools exist for the annotation of many variable phonetic phenomena. At the same time, pre-trained self-supervised models, such as wav2vec2.0, have been shown to perform well at speech classification tasks and latently encode fine-grained phonetic information. We demonstrate that wav2vec2.0 models can be trained to automatically classify stop burst presence with high accuracy in both English and Japanese, robust across both finely-curated and unprepared speech corpora. Patterns of variability in stop realisation are replicated with the automatic annotations, and closely follow those of manual annotations. These results demonstrate the potential of pre-trained speech models as tools for the automatic annotation and processing of speech corpus data, enabling researchers to 'scale-up' the scope of phonetic research with relative ease.
>
---
