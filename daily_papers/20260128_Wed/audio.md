# 音频 cs.SD;  eess.AS

- **最新发布 27 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Beyond Lips: Integrating Gesture and Lip Cues for Robust Audio-visual Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文属于音频-视觉说话人提取任务，旨在解决多说话人场景下目标说话人语音分离问题。通过融合唇部和手势信息，提出SeLG模型提升提取效果。**

- **链接: [https://arxiv.org/pdf/2601.19130v1](https://arxiv.org/pdf/2601.19130v1)**

> **作者:** Zexu Pan; Xinyuan Qian; Shengkui Zhao; Kun Zhou; Bin Ma
>
> **备注:** ICASSP 2026
>
> **摘要:** Most audio-visual speaker extraction methods rely on synchronized lip recording to isolate the speech of a target speaker from a multi-talker mixture. However, in natural human communication, co-speech gestures are also temporally aligned with speech, often emphasizing specific words or syllables. These gestures provide complementary visual cues that can be especially valuable when facial or lip regions are occluded or distant. In this work, we move beyond lip-centric approaches and propose SeLG, a model that integrates both lip and upper-body gesture information for robust speaker extraction. SeLG features a cross-attention-based fusion mechanism that enables each visual modality to query and selectively attend to relevant speech features in the mixture. To improve the alignment of gesture representations with speech dynamics, SeLG also employs a contrastive InfoNCE loss that encourages gesture embeddings to align more closely with corresponding lip embeddings, which are more strongly correlated with speech. Experimental results on the YGD dataset, containing TED talks, demonstrate that the proposed contrastive learning strategy significantly improves gesture-based speaker extraction, and that our proposed SeLG model, by effectively fusing lip and gesture cues with an attention mechanism and InfoNCE loss, achieves superior performance compared to baselines, across both complete and partial (i.e., missing-modality) conditions.
>
---
#### [new 002] SICL-AT: Another way to adapt Auditory LLM to low-resource task
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音与音频理解任务，旨在解决低资源场景下模型性能下降的问题。通过提出SICL-AT方法，增强模型的上下文学习能力，提升在低资源任务上的表现。**

- **链接: [https://arxiv.org/pdf/2601.18904v1](https://arxiv.org/pdf/2601.18904v1)**

> **作者:** Haolong Zheng; Siyin Wang; Zengrui Jin; Mark Hasegawa-Johnson
>
> **摘要:** Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource or unfamiliar tasks. In case of labeled in-domain data is scarce or mismatched to the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that \emph{Vanilla ICL}, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose \textbf{Speech In-Context Learning Adaptation Training (SICL-AT)}, a post-training recipe utilizes only high resource speech data intending to strengthen model's in-context learning capability. The enhancement can generalize to audio understanding/reasoning task. Experiments indicate our proposed method consistently outperforms direct fine-tuning in low-resource scenario.
>
---
#### [new 003] Interpretable and Perceptually-Aligned Music Similarity with Pretrained Embeddings
- **分类: cs.SD**

- **简介: 该论文属于音乐相似性任务，旨在提升音乐检索的可解释性与感知对齐。通过预训练嵌入结合源分离和优化方法，实现更可控的音轨检索。**

- **链接: [https://arxiv.org/pdf/2601.19109v1](https://arxiv.org/pdf/2601.19109v1)**

> **作者:** Arhan Vohra; Taketo Akama
>
> **摘要:** Perceptual similarity representations enable music retrieval systems to determine which songs sound most similar to listeners. State-of-the-art approaches based on task-specific training via self-supervised metric learning show promising alignment with human judgment, but are difficult to interpret or generalize due to limited dataset availability. We show that pretrained text-audio embeddings (CLAP and MuQ-MuLan) offer comparable perceptual alignment on similarity tasks without any additional fine-tuning. To surpass this baseline, we introduce a novel method to perceptually align pretrained embeddings with source separation and linear optimization on ABX preference data from listening tests. Our model provides interpretable and controllable instrument-wise weights, allowing music producers to retrieve stem-level loops and samples based on mixed reference songs.
>
---
#### [new 004] LuSeeL: Language-queried Binaural Universal Sound Event Extraction and Localization
- **分类: eess.AS**

- **简介: 该论文提出LuSeeL模型，解决从双耳音频中提取并定位文本描述的声音事件问题。通过结合语言指令和空间信息，提升声音分离与方向估计效果。**

- **链接: [https://arxiv.org/pdf/2601.19153v1](https://arxiv.org/pdf/2601.19153v1)**

> **作者:** Zexu Pan; Shengkui Zhao; Yukun Ma; Haoxu Wang; Yiheng Jiang; Biao Tian; Bin Ma
>
> **备注:** ICASSP 2026
>
> **摘要:** Most universal sound extraction algorithms focus on isolating a target sound event from single-channel audio mixtures. However, the real world is three-dimensional, and binaural audio, which mimics human hearing, can capture richer spatial information, including sound source location. This spatial context is crucial for understanding and modeling complex auditory scenes, as it inherently informs sound detection and extraction. In this work, we propose a language-driven universal sound extraction network that isolates text-described sound events from binaural mixtures by effectively leveraging the spatial cues present in binaural signals. Additionally, we jointly predict the direction of arrival (DoA) of the target sound using spatial features from the extraction network. This dual-task approach exploits complementary location information to improve extraction performance while enabling accurate DoA estimation. Experimental results on the in-the-wild AudioCaps dataset show that our proposed LuSeeL model significantly outperforms single-channel and uni-task baselines.
>
---
#### [new 005] Enhancing Speech Emotion Recognition using Dynamic Spectral Features and Kalman Smoothing
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决噪声干扰导致的分类错误问题。通过引入动态频谱特征和卡尔曼平滑算法，提升分类准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2601.18908v1](https://arxiv.org/pdf/2601.18908v1)**

> **作者:** Marouane El Hizabri; Abdelfattah Bezzaz; Ismail Hayoukane; Youssef Taki
>
> **摘要:** Speech Emotion Recognition systems often use static features like Mel-Frequency Cepstral Coefficients (MFCCs), Zero Crossing Rate (ZCR), and Root Mean Square Energy (RMSE). Because of this, they can misclassify emotions when there is acoustic noise in vocal signals. To address this, we added dynamic features using Dynamic Spectral features (Deltas and Delta-Deltas) along with the Kalman Smoothing algorithm. This approach reduces noise and improves emotion classification. Since emotion changes over time, the Kalman Smoothing filter also helped make the classifier outputs more stable. Tests on the RAVDESS dataset showed that this method achieved a state-of-the-art accuracy of 87\% and reduced misclassification between emotions with similar acoustic features
>
---
#### [new 006] Dual-Strategy-Enhanced ConBiMamba for Neural Speaker Diarization
- **分类: cs.SD**

- **简介: 该论文属于语音说话人日志任务，旨在解决长序列建模和说话人转换点检测问题。提出Dual-Strategy-Enhanced ConBiMamba系统，结合Conformer与Mamba优势，并引入改进损失和特征聚合方法。**

- **链接: [https://arxiv.org/pdf/2601.19472v1](https://arxiv.org/pdf/2601.19472v1)**

> **作者:** Zhen Liao; Gaole Dai; Mengqiao Chen; Wenqing Cheng; Wei Xu
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Conformer and Mamba have achieved strong performance in speech modeling but face limitations in speaker diarization. Mamba is efficient but struggles with local details and nonlinear patterns. Conformer's self-attention incurs high memory overhead for long speech sequences and may cause instability in long-range dependency modeling. These limitations are critical for diarization, which requires both precise modeling of local variations and robust speaker consistency over extended spans. To address these challenges, we first apply ConBiMamba for speaker diarization. We follow the Pyannote pipeline and propose the Dual-Strategy-Enhanced ConBiMamba neural speaker diarization system. ConBiMamba integrates the strengths of Conformer and Mamba, where Conformer's convolutional and feed-forward structures are utilized to improve local feature extraction. By replacing Conformer's self-attention with ExtBiMamba, ConBiMamba efficiently handles long audio sequences while alleviating the high memory cost of self-attention. Furthermore, to address the problem of the higher DER around speaker change points, we introduce the Boundary-Enhanced Transition Loss to enhance the detection of speaker change points. We also propose Layer-wise Feature Aggregation to enhance the utilization of multi-layer representations. The system is evaluated on six diarization datasets and achieves state-of-the-art performance on four of them. The source code of our study is available at https://github.com/lz-hust/DSE-CBM.
>
---
#### [new 007] SLM-SS: Speech Language Model for Generative Speech Separation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音分离任务，旨在提升分离语音的可懂性。通过引入语言模型，将语音分离建模为序列生成，增强语音连贯性与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2601.19533v1](https://arxiv.org/pdf/2601.19533v1)**

> **作者:** Tianhua Li; Chenda Li; Wei Wang; Xin Zhou; Xihui Chen; Jianqing Gao; Yanmin Qian
>
> **摘要:** Speech separation (SS) has advanced significantly with neural network-based methods, showing improved performance on signal-level metrics. However, these methods often struggle to maintain speech intelligibility in the separated signals, which can negatively affect the performance of downstream tasks such as speech recognition. In this work, we propose SLM-SS, a novel approach that applies speech language models to SS, aiming to enhance the intelligibility and coherence of the separated signals. We frame SS as discrete multi-codebook sequence generation, using Encoder-Decoder models to map quantized speech mixtures to target tokens. In addition to the autoregressive modeling strategy, we introduce a non-autoregressive model to improve decoding efficiency for residual tokens. Experimental results on the LibriMix dataset demonstrate that our approach shows significantly better preservation of speech intelligibility, leading to improved linguistic consistency in a variety of downstream tasks compared to existing approaches.
>
---
#### [new 008] Phase-Retrieval-Based Physics-Informed Neural Networks For Acoustic Magnitude Field Reconstruction
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声场重建任务，旨在从稀疏幅度测量中估计声场幅度分布。针对无相位信息的情况，提出基于相位恢复的物理信息神经网络。**

- **链接: [https://arxiv.org/pdf/2601.19297v1](https://arxiv.org/pdf/2601.19297v1)**

> **作者:** Karl Schrader; Shoichi Koyama; Tomohiko Nakamura; Mirco Pezzoli
>
> **备注:** Accepted to International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026
>
> **摘要:** We propose a method for estimating the magnitude distribution of an acoustic field from spatially sparse magnitude measurements. Such a method is useful when phase measurements are unreliable or inaccessible. Physics-informed neural networks (PINNs) have shown promise for sound field estimation by incorporating constraints derived from governing partial differential equations (PDEs) into neural networks. However, they do not extend to settings where phase measurements are unavailable, as the loss function based on the governing PDE relies on phase information. To remedy this, we propose a phase-retrieval-based PINN for magnitude field estimation. By representing the magnitude and phase distributions with separate networks, the PDE loss can be computed based on the reconstructed complex amplitude. We demonstrate the effectiveness of our phase-retrieval-based PINN through experimental evaluation.
>
---
#### [new 009] A Framework for Evaluating Faithfulness in Explainable AI for Machine Anomalous Sound Detection Using Frequency-Band Perturbation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于机器异常声音检测任务，旨在解决XAI解释是否忠实反映模型决策的问题。通过频率带移除方法评估XAI的可靠性，验证不同方法的有效性。**

- **链接: [https://arxiv.org/pdf/2601.19017v1](https://arxiv.org/pdf/2601.19017v1)**

> **作者:** Alexander Buck; Georgina Cosma; Iain Phillips; Paul Conway; Patrick Baker
>
> **备注:** 16 pages, 24 figures
>
> **摘要:** Explainable AI (XAI) is commonly applied to anomalous sound detection (ASD) models to identify which time-frequency regions of an audio signal contribute to an anomaly decision. However, most audio explanations rely on qualitative inspection of saliency maps, leaving open the question of whether these attributions accurately reflect the spectral cues the model uses. In this work, we introduce a new quantitative framework for evaluating XAI faithfulness in machine-sound analysis by directly linking attribution relevance to model behaviour through systematic frequency-band removal. This approach provides an objective measure of whether an XAI method for machine ASD correctly identifies frequency regions that influence an ASD model's predictions. By using four widely adopted methods, namely Integrated Gradients, Occlusion, Grad-CAM and SmoothGrad, we show that XAI techniques differ in reliability, with Occlusion demonstrating the strongest alignment with true model sensitivity and gradient-+based methods often failing to accurately capture spectral dependencies. The proposed framework offers a reproducible way to benchmark audio explanations and enables more trustworthy interpretation of spectrogram-based ASD systems.
>
---
#### [new 010] Permutation-Invariant Physics-Informed Neural Network for Region-to-Region Sound Field Reconstruction
- **分类: eess.AS**

- **简介: 该论文属于声场重建任务，解决真实场景中声源与接收区位置变化导致的ATF插值问题。提出一种不变排列的物理信息神经网络，结合赫姆霍兹方程提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2601.19491v1](https://arxiv.org/pdf/2601.19491v1)**

> **作者:** Xingyu Chen; Sipei Zhao; Fei Ma; Eva Cheng; Ian S. Burnett
>
> **备注:** Accepted to the 31st International Congress on Sound and Vibration (ICSV 2025)
>
> **摘要:** Most existing sound field reconstruction methods target point-to-region reconstruction, interpolating the Acoustic Transfer Functions (ATFs) between a fixed-position sound source and a receiver region. The applicability of these methods is limited because real-world ATFs tend to varying continuously with respect to the positions of sound sources and receiver regions. This paper presents a permutation-invariant physics-informed neural network for region-to-region sound field reconstruction, which aims to interpolate the ATFs across continuously varying sound sources and measurement regions. The proposed method employs a deep set architecture to process the receiver and sound source positions as an unordered set, preserving acoustic reciprocity. Furthermore, it incorporates the Helmholtz equation as a physical constraint to guide network training, ensuring physically consistent predictions.
>
---
#### [new 011] SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于多说话人语音识别任务，解决跨领域泛化能力不足的问题。通过改进的语音日志条件Whisper模型，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.19194v1](https://arxiv.org/pdf/2601.19194v1)**

> **作者:** Alexander Polok; Dominik Klement; Samuele Cornell; Matthew Wiesner; Jan Černocký; Sanjeev Khudanpur; Lukáš Burget
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speaker-attributed automatic speech recognition (ASR) in multi-speaker environments remains a major challenge. While some approaches achieve strong performance when fine-tuned on specific domains, few systems generalize well across out-of-domain datasets. Our prior work, Diarization-Conditioned Whisper (DiCoW), leverages speaker diarization outputs as conditioning information and, with minimal fine-tuning, demonstrated strong multilingual and multi-domain performance. In this paper, we address a key limitation of DiCoW: ambiguity in Silence-Target-Non-target-Overlap (STNO) masks, where two or more fully overlapping speakers may have nearly identical conditioning despite differing transcriptions. We introduce SE-DiCoW (Self-Enrolled Diarization-Conditioned Whisper), which uses diarization output to locate an enrollment segment anywhere in the conversation where the target speaker is most active. This enrollment segment is used as fixed conditioning via cross-attention at each encoder layer. We further refine DiCoW with improved data segmentation, model initialization, and augmentation. Together, these advances yield substantial gains: SE-DiCoW reduces macro-averaged tcpWER by 52.4% relative to the original DiCoW on the EMMA MT-ASR benchmark.
>
---
#### [new 012] Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于说话人验证任务，旨在解决传统方法在建模说话人特征层次结构上的不足。通过引入双曲空间，提出H-Softmax和HAM-Softmax，提升分类性能并保持层次结构建模能力。**

- **链接: [https://arxiv.org/pdf/2601.19709v1](https://arxiv.org/pdf/2601.19709v1)**

> **作者:** Zhihua Fang; Liang He
>
> **备注:** 5 pages, 3 figures, Accepted at ICASSP 2026
>
> **摘要:** Speaker embedding learning based on Euclidean space has achieved significant progress, but it is still insufficient in modeling hierarchical information within speaker features. Hyperbolic space, with its negative curvature geometric properties, can efficiently represent hierarchical information within a finite volume, making it more suitable for the feature distribution of speaker embeddings. In this paper, we propose Hyperbolic Softmax (H-Softmax) and Hyperbolic Additive Margin Softmax (HAM-Softmax) based on hyperbolic space. H-Softmax incorporates hierarchical information into speaker embeddings by projecting embeddings and speaker centers into hyperbolic space and computing hyperbolic distances. HAM-Softmax further enhances inter-class separability by introducing margin constraint on this basis. Experimental results show that H-Softmax and HAM-Softmax achieve average relative EER reductions of 27.84% and 14.23% compared with standard Softmax and AM-Softmax, respectively, demonstrating that the proposed methods effectively improve speaker verification performance and at the same time preserve the capability of hierarchical structure modeling. The code will be released at https://github.com/PunkMale/HAM-Softmax.
>
---
#### [new 013] Residual Tokens Enhance Masked Autoencoders for Speech Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音建模任务，旨在提升语音的表达能力和质量。通过引入残差可训练标记，补充传统属性建模的不足，增强语音的自然性和可控性。**

- **链接: [https://arxiv.org/pdf/2601.19399v1](https://arxiv.org/pdf/2601.19399v1)**

> **作者:** Samir Sadok; Stéphane Lathuilière; Xavier Alameda-Pineda
>
> **备注:** Submitted to ICASSP 2026 (accepted)
>
> **摘要:** Recent speech modeling relies on explicit attributes such as pitch, content, and speaker identity, but these alone cannot capture the full richness of natural speech. We introduce RT-MAE, a novel masked autoencoder framework that augments the supervised attributes-based modeling with unsupervised residual trainable tokens, designed to encode the information not explained by explicit labeled factors (e.g., timbre variations, noise, emotion etc). Experiments show that RT-MAE improves reconstruction quality, preserving content and speaker similarity while enhancing expressivity. We further demonstrate its applicability to speech enhancement, removing noise at inference while maintaining controllability and naturalness.
>
---
#### [new 014] Audio Foundation Models Outperform Symbolic Representations for Piano Performance Evaluation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐性能评估任务，解决传统符号表示无法捕捉音色细节的问题，通过音频基础模型提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.19029v1](https://arxiv.org/pdf/2601.19029v1)**

> **作者:** Jai Dhiman
>
> **备注:** 6 pages, 4 figures, 2 tables. Code available at https://github.com/Jai-Dhiman/crescendai
>
> **摘要:** Automated piano performance evaluation traditionally relies on symbolic (MIDI) representations, which capture note-level information but miss the acoustic nuances that characterize expressive playing. I propose using pre-trained audio foundation models, specifically MuQ and MERT, to predict 19 perceptual dimensions of piano performance quality. Using synthesized audio from PercePiano MIDI files (rendered via Pianoteq), I compare audio and symbolic approaches under controlled conditions where both derive from identical source data. The best model, MuQ layers 9-12 with Pianoteq soundfont augmentation, achieves R^2 = 0.537 (95% CI: [0.465, 0.575]), representing a 55% improvement over the symbolic baseline (R^2 = 0.347). Statistical analysis confirms significance (p < 10^-25) with audio outperforming symbolic on all 19 dimensions. I validate the approach through cross-soundfont generalization (R^2 = 0.534 +/- 0.075), difficulty correlation with an external dataset (rho = 0.623), and multi-performer consistency analysis. Analysis of audio-symbolic fusion reveals high error correlation (r = 0.738), explaining why fusion provides minimal benefit: audio representations alone are sufficient. I release the complete training pipeline, pretrained models, and inference code.
>
---
#### [new 015] Rethinking Discrete Speech Representation Tokens for Accent Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，旨在解决DSRT中Accent信息编码问题。通过提出评估框架，分析不同编码器的DSRT，并设计新型DSRT以提升可控Accent生成效果。**

- **链接: [https://arxiv.org/pdf/2601.19786v1](https://arxiv.org/pdf/2601.19786v1)**

> **作者:** Jinzuomu Zhong; Yi Wang; Korin Richmond; Peter Bell
>
> **摘要:** Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from a variety of speech encoders. Our results reveal that accent information is substantially reduced when ASR supervision is used to fine-tune the encoder, but cannot be effectively disentangled from phonetic and speaker information through naive codebook size reduction. Based on these findings, we propose new content-only and content-accent DSRTs that significantly outperform existing designs in controllable accent generation. Our work highlights the importance of accent-aware evaluation and provides practical guidance for designing DSRTs for accent-controlled speech generation.
>
---
#### [new 016] A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在解决现有基准无法评估模型跨音频任务推理能力的问题。提出Audio Reasoning Tasks（ART）基准以测试模型的音频推理能力。**

- **链接: [https://arxiv.org/pdf/2601.19673v1](https://arxiv.org/pdf/2601.19673v1)**

> **作者:** Iwona Christop; Mateusz Czyżnikiewicz; Paweł Skórzewski; Łukasz Bondaruk; Jakub Kubiak; Marcin Lewandowski; Marek Kubis
>
> **备注:** 31 pages, 2 figures, accepted to EACL 2026
>
> **摘要:** The present benchmarks for testing the audio modality of multimodal large language models concentrate on testing various audio tasks such as speaker diarization or gender identification in isolation. Whether a multimodal model can answer the questions that require reasoning skills to combine audio tasks of different categories, cannot be verified with their use. To address this issue, we propose Audio Reasoning Tasks (ART), a new benchmark for assessing the ability of multimodal models to solve problems that require reasoning over audio signal.
>
---
#### [new 017] Audio Deepfake Detection at the First Greeting: "Hi!"
- **分类: eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决短时、低质量语音中合成语音的检测问题。提出S-MGAA模型，提升短音频的判别能力。**

- **链接: [https://arxiv.org/pdf/2601.19573v1](https://arxiv.org/pdf/2601.19573v1)**

> **作者:** Haohan Shi; Xiyu Shi; Safak Dogan; Tianjin Huang; Yunxiao Zhang
>
> **备注:** Accepted at ICASSP 2026. Copyright 2026 IEEE. The final published version will be available via IEEE Xplore
>
> **摘要:** This paper focuses on audio deepfake detection under real-world communication degradations, with an emphasis on ultra-short inputs (0.5-2.0s), targeting the capability to detect synthetic speech at a conversation opening, e.g., when a scammer says "Hi." We propose Short-MGAA (S-MGAA), a novel lightweight extension of Multi-Granularity Adaptive Time-Frequency Attention, designed to enhance discriminative representation learning for short, degraded inputs subjected to communication processing and perturbations. The S-MGAA integrates two tailored modules: a Pixel-Channel Enhanced Module (PCEM) that amplifies fine-grained time-frequency saliency, and a Frequency Compensation Enhanced Module (FCEM) to supplement limited temporal evidence via multi-scale frequency modeling and adaptive frequency-temporal interaction. Extensive experiments demonstrate that S-MGAA consistently surpasses nine state-of-the-art baselines while achieving strong robustness to degradations and favorable efficiency-accuracy trade-offs, including low RTF, competitive GFLOPs, compact parameters, and reduced training cost, highlighting its strong potential for real-time deployment in communication systems and edge devices.
>
---
#### [new 018] Phonological Tokenizer: Prosody-Aware Phonetic Token via Multi-Objective Fine-Tuning with Differentiable K-Means
- **分类: cs.SD**

- **简介: 该论文提出一种基于多任务微调的语音分词方法，旨在解决语音语言模型中缺乏韵律信息的问题。通过不同iable k-means优化，保留语音的音系信息并去除说话人特征。**

- **链接: [https://arxiv.org/pdf/2601.19781v1](https://arxiv.org/pdf/2601.19781v1)**

> **作者:** Kentaro Onda; Hayato Futami; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** In recent years, there has been growing interest in representing speech with discrete tokens, which serve as pseudo-text for speech language models (speechLMs) and as efficient intermediate representations for downstream tasks. These tokens are typically categorized as acoustic and phonetic tokens: the former holds detailed acoustic information for reconstruction while the latter mainly captures linguistic content. In human speech communication, however, unnecessary acoustic details such as speaker information are abstracted, while both linguistic and prosodic information are utilized for speech comprehension and production. Given this, neither type of token seems an ideal representation for tasks sensitive to prosody, such as speechLMs. In this study, we propose the Phonological Tokenizer, a method that fine-tunes phonetic tokens via differentiable k-means with a multi-task objective of ASR and speech resynthesis. Experimental validation on diverse tasks confirms that our tokens retain phonological (both linguistic and prosodic) information while appropriately discarding speaker identity.
>
---
#### [new 019] A Hybrid Discriminative and Generative System for Universal Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在处理不同噪声和录音条件下的语音。提出混合模型，结合判别与生成方法，提升语音质量。**

- **链接: [https://arxiv.org/pdf/2601.19113v1](https://arxiv.org/pdf/2601.19113v1)**

> **作者:** Yinghao Liu; Chengwei Liu; Xiaotao Liang; Haoyin Yan; Shaofei Xue; Zheng Xue
>
> **备注:** Accepted by ICASSP 2026.This work was submitted to the ICASSP 2026 URGENT Challenge (Track 1)
>
> **摘要:** Universal speech enhancement aims at handling inputs with various speech distortions and recording conditions. In this work, we propose a novel hybrid architecture that synergizes the signal fidelity of discriminative modeling with the reconstruction capabilities of generative modeling. Our system utilizes the discriminative TF-GridNet model with the Sampling-Frequency-Independent strategy to handle variable sampling rates universally. In parallel, an autoregressive model combined with spectral mapping modeling generates detail-rich speech while effectively suppressing generative artifacts. Finally, a fusion network learns adaptive weights of the two outputs under the optimization of signal-level losses and the comprehensive Speech Quality Assessment (SQA) loss. Our proposed system is evaluated in the ICASSP 2026 URGENT Challenge (Track 1) and ranks the third place.
>
---
#### [new 020] SAM Audio Judge: A Unified Multimodal Framework for Perceptual Evaluation of Audio Separation
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于音频分离任务，旨在解决现有评估指标与人类感知不一致的问题。提出SAJ作为无需参考的多模态评估框架，提升自动化评价效果。**

- **链接: [https://arxiv.org/pdf/2601.19702v1](https://arxiv.org/pdf/2601.19702v1)**

> **作者:** Helin Wang; Bowen Shi; Andros Tjandra; John Hoffman; Yi-Chiao Wu; Apoorv Vyas; Najim Dehak; Ann Lee; Wei-Ning Hsu
>
> **摘要:** The performance evaluation remains a complex challenge in audio separation, and existing evaluation metrics are often misaligned with human perception, course-grained, relying on ground truth signals. On the other hand, subjective listening tests remain the gold standard for real-world evaluation, but they are expensive, time-consuming, and difficult to scale. This paper addresses the growing need for automated systems capable of evaluating audio separation without human intervention. The proposed evaluation metric, SAM Audio Judge (SAJ), is a multimodal fine-grained reference-free objective metric, which shows highly alignment with human perceptions. SAJ supports three audio domains (speech, music and general sound events) and three prompt inputs (text, visual and span), covering four different dimensions of evaluation (recall, percision, faithfulness, and overall). SAM Audio Judge also shows potential applications in data filtering, pseudo-labeling large datasets and reranking in audio separation models. We release our code and pre-trained models at: https://github.com/facebookresearch/sam-audio.
>
---
#### [new 021] Advanced Modeling of Interlanguage Speech Intelligibility Benefit with L1-L2 Multi-Task Learning Using Differentiable K-Means for Accent-Robust Discrete Token-Based ASR
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升外语口音语音的识别准确率。通过改进的ISIB模型和多任务学习方法，增强了系统对口音的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.19767v1](https://arxiv.org/pdf/2601.19767v1)**

> **作者:** Kentaro Onda; Satoru Fukayama; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Building ASR systems robust to foreign-accented speech is an important challenge in today's globalized world. A prior study explored the way to enhance the performance of phonetic token-based ASR on accented speech by reproducing the phenomenon known as interlanguage speech intelligibility benefit (ISIB), where foreign-accented speech is more intelligible to listeners sharing the speaker's native language than to native listeners. ISIB was technically implemented by using the speaker's L1 to learn k-means cluster centroids in an SSL feature space to obtain phonetic tokens. In this study, we propose a more advanced modeling of ISIB. By employing differentiable k-means and optimizing the entire module for both L1 and L2 ASR, the proposed method outperformed the baselines, both when using only native speech and when additionally incorporating a limited amount of accented speech. Notably, in the latter scenario, our method achieved approximately a 20% relative improvement in recognition accuracy.
>
---
#### [new 022] Physics-Aware Novel-View Acoustic Synthesis with Vision-Language Priors and 3D Acoustic Environment Modeling
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于新型视角声学合成任务，解决空间音频生成中物理特性建模不足的问题。通过融合视觉语言先验和3D声学环境建模，提升音频的物理一致性与真实感。**

- **链接: [https://arxiv.org/pdf/2601.19712v1](https://arxiv.org/pdf/2601.19712v1)**

> **作者:** Congyi Fan; Jian Guan; Youtian Lin; Dongli Xu; Tong Ye; Qiaoxi Zhu; Pengming Feng; Wenwu Wang
>
> **备注:** ICASSP 2026 Accept, Project page: https://physnvas.github.io/
>
> **摘要:** Spatial audio is essential for immersive experiences, yet novel-view acoustic synthesis (NVAS) remains challenging due to complex physical phenomena such as reflection, diffraction, and material absorption. Existing methods based on single-view or panoramic inputs improve spatial fidelity but fail to capture global geometry and semantic cues such as object layout and material properties. To address this, we propose Phys-NVAS, the first physics-aware NVAS framework that integrates spatial geometry modeling with vision-language semantic priors. A global 3D acoustic environment is reconstructed from multi-view images and depth maps to estimate room size and shape, enhancing spatial awareness of sound propagation. Meanwhile, a vision-language model extracts physics-aware priors of objects, layouts, and materials, capturing absorption and reflection beyond geometry. An acoustic feature fusion adapter unifies these cues into a physics-aware representation for binaural generation. Experiments on RWAVS demonstrate that Phys-NVAS yields binaural audio with improved realism and physical consistency.
>
---
#### [new 023] Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，解决多语言ASR中资源不足与泛化能力差的问题。通过基于语系的连接器共享策略，提升模型效率与跨语言泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.18899v1](https://arxiv.org/pdf/2601.18899v1)**

> **作者:** Yuchen Zhang; Ravi Shekhar; Haralambos Mouratidis
>
> **摘要:** Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
>
---
#### [new 024] Optimizing Conversational Quality in Spoken Dialogue Systems with Reinforcement Learning from AI Feedback
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于对话系统优化任务，解决传统RLHF方法在多维对话质量评估上的不足，提出多奖励RLAIF框架，提升语义、音频和情感一致性。**

- **链接: [https://arxiv.org/pdf/2601.19063v1](https://arxiv.org/pdf/2601.19063v1)**

> **作者:** Siddhant Arora; Jinchuan Tian; Jiatong Shi; Hayato Futami; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **摘要:** Reinforcement learning from human or AI feedback (RLHF/RLAIF) for speech-in/speech-out dialogue systems (SDS) remains underexplored, with prior work largely limited to single semantic rewards applied at the utterance level. Such setups overlook the multi-dimensional and multi-modal nature of conversational quality, which encompasses semantic coherence, audio naturalness, speaker consistency, emotion alignment, and turn-taking behavior. Moreover, they are fundamentally mismatched with duplex spoken dialogue systems that generate responses incrementally, where agents must make decisions based on partial utterances. We address these limitations with the first multi-reward RLAIF framework for SDS, combining semantic, audio-quality, and emotion-consistency rewards. To align utterance-level preferences with incremental, blockwise decoding in duplex models, we apply turn-level preference sampling and aggregate per-block log-probabilities within a single DPO objective. We present the first systematic study of preference learning for improving SDS quality in both multi-turn Chain-of-Thought and blockwise duplex models, and release a multi-reward DPO dataset to support reproducible research. Experiments show that single-reward RLAIF selectively improves its targeted metric, while joint multi-reward training yields consistent gains across semantic quality and audio naturalness. These results highlight the importance of holistic, multi-reward alignment for practical conversational SDS.
>
---
#### [new 025] Uncertainty-Aware 3D Emotional Talking Face Synthesis with Emotion Prior Distillation
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于3D情感人脸合成任务，解决音频-视觉情感对齐差和多视角融合不足的问题。提出UA-3DTalk，通过情感先验蒸馏和不确定性建模提升合成质量。**

- **链接: [https://arxiv.org/pdf/2601.19112v1](https://arxiv.org/pdf/2601.19112v1)**

> **作者:** Nanhan Shen; Zhilei Liu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Emotional Talking Face synthesis is pivotal in multimedia and signal processing, yet existing 3D methods suffer from two critical challenges: poor audio-vision emotion alignment, manifested as difficult audio emotion extraction and inadequate control over emotional micro-expressions; and a one-size-fits-all multi-view fusion strategy that overlooks uncertainty and feature quality differences, undermining rendering quality. We propose UA-3DTalk, Uncertainty-Aware 3D Emotional Talking Face Synthesis with emotion prior distillation, which has three core modules: the Prior Extraction module disentangles audio into content-synchronized features for alignment and person-specific complementary features for individualization; the Emotion Distillation module introduces a multi-modal attention-weighted fusion mechanism and 4D Gaussian encoding with multi-resolution code-books, enabling fine-grained audio emotion extraction and precise control of emotional micro-expressions; the Uncertainty-based Deformation deploys uncertainty blocks to estimate view-specific aleatoric (input noise) and epistemic (model parameters) uncertainty, realizing adaptive multi-view fusion and incorporating a multi-head decoder for Gaussian primitive optimization to mitigate the limitations of uniform-weight fusion. Extensive experiments on regular and emotional datasets show UA-3DTalk outperforms state-of-the-art methods like DEGSTalk and EDTalk by 5.2% in E-FID for emotion alignment, 3.1% in SyncC for lip synchronization, and 0.015 in LPIPS for rendering quality. Project page: https://mrask999.github.io/UA-3DTalk
>
---
#### [new 026] Echoes of the Land: An Interactive Installation Based on Physical Model of Earthquake
- **分类: cs.HC; cs.SD; nlin.AO; physics.pop-ph**

- **简介: 该论文属于艺术与科学交叉研究任务，旨在将地震物理模型转化为互动艺术装置，通过实时声光效果展现地震动态，探索复杂系统与美学的结合。**

- **链接: [https://arxiv.org/pdf/2507.14947v1](https://arxiv.org/pdf/2507.14947v1)**

> **作者:** Ivan C. H. Liu; Chung-En Hao; Jing Xie
>
> **备注:** 7 pages, 8 figures, submitted to Leonardo
>
> **摘要:** Echoes of the Land is an interactive installation that transforms seismic dynamics into a multisensory experience through a scientifically grounded spring-block model. Simulating earthquake recurrence and self-organized criticality, the work generates real-time sound and light via motion capture and concatenative granular synthesis. Each block acts as an agent, producing emergent audiovisual cascades that visualize the physics of rupture and threshold behavior. This work exemplifies the amalgamation of scientific knowledge and artistic practice, opening new avenues for novel forms of musical instrument and narrative medium, while inviting further investigation into the intersection of emergent complexity, aesthetics and interactivity.
>
---
#### [new 027] GMS-CAVP: Improving Audio-Video Correspondence with Multi-Scale Contrastive and Generative Pretraining
- **分类: cs.CV; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于视频-音频对应任务，旨在提升跨模态对齐。解决现有方法对多尺度时空结构建模不足的问题，提出GMS-CAVP框架，结合多尺度对比学习与生成预训练。**

- **链接: [https://arxiv.org/pdf/2601.19606v1](https://arxiv.org/pdf/2601.19606v1)**

> **作者:** Shentong Mo; Zehua Chen; Jun Zhu
>
> **摘要:** Recent advances in video-audio (V-A) understanding and generation have increasingly relied on joint V-A embeddings, which serve as the foundation for tasks such as cross-modal retrieval and generation. While prior methods like CAVP effectively model semantic and temporal correspondences between modalities using contrastive objectives, their performance remains suboptimal. A key limitation is the insufficient modeling of the dense, multi-scale nature of both video and audio signals, correspondences often span fine- to coarse-grained spatial-temporal structures, which are underutilized in existing frameworks. To this end, we propose GMS-CAVP, a novel framework that combines Multi-Scale Video-Audio Alignment and Multi-Scale Spatial-Temporal Diffusion-based pretraining objectives to enhance V-A correspondence modeling. First, GMS-CAVP introduces a multi-scale contrastive learning strategy that captures semantic and temporal relations across varying granularities. Second, we go beyond traditional contrastive learning by incorporating a diffusion-based generative objective, enabling modality translation and synthesis between video and audio. This unified discriminative-generative formulation facilitates deeper cross-modal understanding and paves the way for high-fidelity generation. Extensive experiments on VGGSound, AudioSet, and Panda70M demonstrate that GMS-CAVP outperforms previous methods in generation and retrieval.
>
---
## 更新

#### [replaced 001] Distillation-based Layer Dropping (DLD): Effective End-to-end Framework for Dynamic Speech Networks
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于语音识别任务，旨在解决动态网络中层剪枝导致性能下降的问题。提出DLD框架，结合知识蒸馏与层剪枝，提升动态语音网络性能。**

- **链接: [https://arxiv.org/pdf/2601.16117v2](https://arxiv.org/pdf/2601.16117v2)**

> **作者:** Abdul Hannan; Daniele Falavigna; Shah Nawaz; Mubashir Noman; Markus Schedl; Alessio Brutti
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Edge devices operate in constrained and varying resource settings, requiring dynamic architectures that can adapt to limitations of the available resources. To meet such demands, layer dropping ($\mathcal{LD}$) approach is typically used to transform static models into dynamic ones by skipping parts of the network along with reducing overall computational complexity. However, existing $\mathcal{LD}$ methods greatly impact the dynamic model's performance for low and high dropping cases, deteriorating the performance-computation trade-off. To this end, we propose a distillation-based layer dropping (DLD) framework that effectively combines the capabilities of knowledge distillation and $\mathcal{LD}$ in an end-to-end fashion, thereby achieving state-of-the-art performance for dynamic speech networks. Comprehensive experimentation utilizing well-known speech recognition methods, including conformer and WavLM, on three public benchmarks demonstrates the effectiveness of our framework, reducing the word error rate by $9.32\%$ and $2.25\%$ for high and no dropping cases with $33.3\%$ reduction in training time.
>
---
#### [replaced 002] Dynamically Slimmable Speech Enhancement Network with Metric-Guided Training
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在降低轻量模型的复杂度。提出DSN网络和MGT方法，动态调整计算资源，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2510.11395v2](https://arxiv.org/pdf/2510.11395v2)**

> **作者:** Haixin Zhao; Kaixuan Yang; Nilesh Madhu
>
> **备注:** Preprint version of a paper under review at ICASSP2026
>
> **摘要:** To further reduce the complexity of lightweight speech enhancement models, we introduce a gating-based Dynamically Slimmable Network (DSN). The DSN comprises static and dynamic components. For architecture-independent applicability, we introduce distinct dynamic structures targeting the commonly used components, namely, grouped recurrent neural network units, multi-head attention, convolutional, and fully connected layers. A policy module adaptively governs the use of dynamic parts at a frame-wise resolution according to the input signal quality, controlling computational load. We further propose Metric-Guided Training (MGT) to explicitly guide the policy module in assessing input speech quality. Experimental results demonstrate that the DSN achieves comparable enhancement performance in instrumental metrics to the state-of-the-art lightweight baseline, while using only 73% of its computational load on average. Evaluations of dynamic component usage ratios indicate that the MGT-DSN can appropriately allocate network resources according to the severity of input signal distortion.
>
---
#### [replaced 003] Transfer Learning for Paediatric Sleep Apnoea Detection Using Physiology-Guided Acoustic Models
- **分类: eess.AS**

- **简介: 该论文属于睡眠障碍检测任务，旨在解决儿童阻塞性睡眠呼吸暂停诊断困难的问题。通过迁移学习，将成人声学模型应用于儿童OSA检测，并结合血氧数据提升效果。**

- **链接: [https://arxiv.org/pdf/2509.15008v2](https://arxiv.org/pdf/2509.15008v2)**

> **作者:** Chaoyue Niu; Veronica Rowe; Guy J. Brown; Heather Elphick; Heather Kenyon; Lowri Thomas; Sam Johnson; Ning Ma
>
> **备注:** The paper has been accepted in 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** Paediatric obstructive sleep apnoea (OSA) is clinically significant yet difficult to diagnose, as children poorly tolerate sensor-based polysomnography. Acoustic monitoring provides a non-invasive alternative for home-based OSA screening, but limited paediatric data hinders the development of robust deep learning approaches. This paper proposes a transfer learning framework that adapts acoustic models pretrained on adult sleep data to paediatric OSA detection, incorporating SpO2-based desaturation patterns to enhance model training. Using a large adult sleep dataset (157 nights) and a smaller paediatric dataset (15 nights), we systematically evaluate (i) single- versus multi-task learning, (ii) encoder freezing versus full fine-tuning, and (iii) the impact of delaying SpO2 labels to better align them with the acoustics and capture physiologically meaningful features. Results show that fine-tuning with SpO2 integration consistently improves paediatric OSA detection compared with baseline models without adaptation. These findings demonstrate the feasibility of transfer learning for home-based OSA screening in children and offer its potential clinical value for early diagnosis.
>
---
#### [replaced 004] Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial Debiasing for Out-of-Distribution Robustness
- **分类: eess.AS**

- **简介: 该论文属于多模态呼吸音分类任务，旨在解决因年龄、性别等属性导致的偏差问题，提升模型在分布偏移下的泛化能力。通过对抗去偏和反事实增强方法，学习与元数据无关的表示。**

- **链接: [https://arxiv.org/pdf/2510.22263v2](https://arxiv.org/pdf/2510.22263v2)**

> **作者:** Heejoon Koo; Miika Toikkanen; Yoon Tae Kim; Soo Yong Kim; June-Woo Kim
>
> **备注:** Accepted by ICASSP 2026 (2026 IEEE International Conference on Acoustics, Speech, and Signal Processing)
>
> **摘要:** Multimodal respiratory sound classification offers promise for early pulmonary disease detection by integrating bioacoustic signals with patient metadata. Nevertheless, current approaches remain vulnerable to spurious correlations from attributes such as age, sex, or acquisition device, which hinder their generalization, especially under distribution shifts across clinical sites. To this end, we propose a counterfactual adversarial debiasing framework. First, we employ a causal graph-based counterfactual debiasing methodology to suppress non-causal dependencies from patient metadata. Second, we introduce adversarial debiasing to learn metadata-insensitive representations and reduce metadata-specific biases. Third, we design counterfactual metadata augmentation to mitigate spurious correlations further and strengthen metadata-invariant representations. By doing so, our method consistently outperforms strong baselines in evaluations under both in-distribution and distribution shifts. Code is available at https://github.com/RSC-Toolkit/BTS-CARD.
>
---
#### [replaced 005] Confidence intervals for forced alignment boundaries using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决强制对齐边界不确定性问题。通过神经网络集成方法，生成边界置信区间，提升对齐可靠性与可分析性。**

- **链接: [https://arxiv.org/pdf/2506.01256v2](https://arxiv.org/pdf/2506.01256v2)**

> **作者:** Matthew C. Kelley
>
> **备注:** submitted for publication; 7 pages, 1 figure
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. Having confidence intervals provides an estimate of the uncertainty in the boundary placement, facilitating tasks like finding boundaries that should be reviewed. As a bonus, on the Buckeye and TIMIT corpora, the ensemble boundaries show a slight overall improvement over using just a single model. The confidence intervals can be emitted during the alignment process as JSON files and a main table for programmatic and statistical analysis. For familiarity, they are also output as Praat TextGrids using a point tier to represent the intervals.
>
---
#### [replaced 006] Short-Segment Speaker Verification with Pre-trained Models and Multi-Resolution Encoder
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，针对短段说话人验证问题，提出结合预训练模型与多分辨率编码器的系统，提升小样本下的验证性能。**

- **链接: [https://arxiv.org/pdf/2509.19721v2](https://arxiv.org/pdf/2509.19721v2)**

> **作者:** Jisoo Myoung; Sangwook Han; Kihyuk Kim; Jong Won Shin
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speaker verification (SV) utilizing features obtained from models pre-trained via self-supervised learning has recently demonstrated impressive performances. However, these pre-trained models (PTMs) usually have a temporal resolution of 20 ms, which is lower than typical filterbank features. It may be problematic especially for short-segment SV with an input segment shorter than 2 s, in which we need to extract as much information as possible from the input with a limited length. Although there have been approaches to utilize multi-resolution features from the HuBERT models, the window shifts were 20, 40, and 100 ms when the sampling rate was 16 kHz and thus only lower resolution features were considered. In this study, we propose an SV system which utilizes PTM features along with filterbank features and those from the multi-resolution time domain encoder with window shifts of 1.56, 3.13, 6.25, and 12.5 ms. Experimental results on the VoxCeleb dataset with various input lengths showed consistent improvements over systems with various combinations of input features.
>
---
#### [replaced 007] Why Do Speech Language Models Fail to Generate Semantically Coherent Outputs? A Modality Evolving Perspective
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音语言模型研究，旨在解决SLMs生成语义不连贯输出的问题。通过分析语音特性，揭示了语音序列长度、语音信息复杂性等因素的影响，为改进SLMs提供方向。**

- **链接: [https://arxiv.org/pdf/2412.17048v2](https://arxiv.org/pdf/2412.17048v2)**

> **作者:** Hankun Wang; Haoran Wang; Yiwei Guo; Zhihan Li; Chenpeng Du; Kai Yu
>
> **备注:** 5 pages, 3 figures, 4 tables. Accepted to IEEE ICASSP 2026
>
> **摘要:** Although text-based large language models exhibit human-level writing ability and remarkable intelligence, speech language models (SLMs) still struggle to generate semantically coherent outputs. There are several potential reasons for this performance degradation: (A) speech tokens mainly provide phonetic information rather than semantic information, (B) the length of speech sequences is much longer than that of text sequences, and (C) paralinguistic information, such as prosody, introduces additional complexity and variability. In this paper, we explore the influence of three key factors separately by transiting the modality from text to speech in an evolving manner. Our findings reveal that the impact of the three factors varies. Factor A has a relatively minor impact, factor B influences syntactical and semantic modeling more obviously, and factor C exerts the most significant impact, particularly in the basic lexical modeling. Based on these findings, we provide insights into the unique challenges of training SLMs and highlight pathways to develop more effective end-to-end SLMs.
>
---
#### [replaced 008] Stream-Voice-Anon: Enhancing Utility of Real-Time Speaker Anonymization via Neural Audio Codec and Language Models
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音匿名化任务，旨在保护在线语音应用中的说话人身份。针对实时场景，提出Stream-Voice-Anon方法，结合神经音频编解码器与语言模型，提升语音可懂度和情感保留，同时保障隐私。**

- **链接: [https://arxiv.org/pdf/2601.13948v2](https://arxiv.org/pdf/2601.13948v2)**

> **作者:** Nikita Kuzmin; Songting Liu; Kong Aik Lee; Eng Siong Chng
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Protecting speaker identity is crucial for online voice applications, yet streaming speaker anonymization (SA) remains underexplored. Recent research has demonstrated that neural audio codec (NAC) provides superior speaker feature disentanglement and linguistic fidelity. NAC can also be used with causal language models (LM) to enhance linguistic fidelity and prompt control for streaming tasks. However, existing NAC-based online LM systems are designed for voice conversion (VC) rather than anonymization, lacking the techniques required for privacy protection. Building on these advances, we present Stream-Voice-Anon, which adapts modern causal LM-based NAC architectures specifically for streaming SA by integrating anonymization techniques. Our anonymization approach incorporates pseudo-speaker representation sampling, a speaker embedding mixing and diverse prompt selection strategies for LM conditioning that leverage the disentanglement properties of quantized content codes to prevent speaker information leakage. Additionally, we compare dynamic and fixed delay configurations to explore latency-privacy trade-offs in real-time scenarios. Under the VoicePrivacy 2024 Challenge protocol, Stream-Voice-Anon achieves substantial improvements in intelligibility (up to 46% relative WER reduction) and emotion preservation (up to 28% UAR relative) compared to the previous state-of-the-art streaming method DarkStream while maintaining comparable latency (180ms vs 200ms) and privacy protection against lazy-informed attackers, though showing 15% relative degradation against semi-informed attackers.
>
---
#### [replaced 009] Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于多模态语音识别任务，旨在解决LLM在ASR、VSR和AVSR中独立训练导致资源浪费和效率低的问题。提出Omni-AVSR框架，实现统一高效训练与推理。**

- **链接: [https://arxiv.org/pdf/2511.07253v3](https://arxiv.org/pdf/2511.07253v3)**

> **作者:** Umberto Cappellazzo; Xubo Liu; Pingchuan Ma; Stavros Petridis; Maja Pantic
>
> **备注:** Accepted to IEEE ICASSP 2026 (camera-ready version). Project website (code and model weights): https://umbertocappellazzo.github.io/Omni-AVSR/
>
> **摘要:** Large language models (LLMs) have recently achieved impressive results in speech recognition across multiple modalities, including Auditory Speech Recognition (ASR), Visual Speech Recognition (VSR), and Audio-Visual Speech Recognition (AVSR). Despite this progress, current LLM-based approaches typically address each task independently, training separate models that raise computational and deployment resource use while missing potential cross-task synergies. They also rely on fixed-rate token compression, which restricts flexibility in balancing accuracy with efficiency. These limitations highlight the need for a unified framework that can support ASR, VSR, and AVSR while enabling elastic inference. To this end, we present Omni-AVSR, a unified audio-visual LLM that combines efficient multi-granularity training with parameter-efficient adaptation. Specifically, we adapt the matryoshka representation learning paradigm to efficiently train across multiple audio and visual granularities, reducing its inherent training resource use. Furthermore, we explore three LoRA-based strategies for adapting the backbone LLM, balancing shared and task-specific specialization. Experiments on LRS2 and LRS3 show that Omni-AVSR achieves comparable or superior accuracy to state-of-the-art baselines while training a single model at substantially lower training and deployment resource use. The model also remains robust under acoustic noise, and we analyze its scaling behavior as LLM size increases, providing insights into the trade-off between performance and efficiency.
>
---
#### [replaced 010] Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉语音识别任务，旨在解决LLMs中的注意力陷阱和大量激活问题。通过分析发现中间低语义标记存在高相似性，提出去相关损失优化模型性能。**

- **链接: [https://arxiv.org/pdf/2510.22603v3](https://arxiv.org/pdf/2510.22603v3)**

> **作者:** Anand; Umberto Cappellazzo; Stavros Petridis; Maja Pantic
>
> **备注:** IEEE ICASSP 2026. The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
>
> **摘要:** Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
>
---
#### [replaced 011] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文介绍CAMEO，一个用于情感识别的多语言语音数据集。旨在解决跨语言和情绪状态的语音情感识别问题，通过标准化数据集促进研究与模型评估。**

- **链接: [https://arxiv.org/pdf/2505.11051v3](https://arxiv.org/pdf/2505.11051v3)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Accepted for ICASSP 2026
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [replaced 012] Unsupervised lexicon learning from speech is limited by representations rather than clustering
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于无监督词分割与聚类任务，研究语音中词典学习的限制因素。通过对比不同表示和聚类方法，发现词内表示差异是性能瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.09225v2](https://arxiv.org/pdf/2510.09225v2)**

> **作者:** Danel Slabbert; Simon Malan; Herman Kamper
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Zero-resource word segmentation and clustering systems aim to tokenise speech into word-like units without access to text labels. Despite progress, the induced lexicons are still far from perfect. In an idealised setting with gold word boundaries, we ask whether performance is limited by the representation of word segments, or by the clustering methods that group them into word-like types. We combine a range of self-supervised speech features (continuous/discrete, frame/word-level) with different clustering methods (K-means, hierarchical, graph-based) on English and Mandarin data. The best system uses graph clustering with dynamic time warping on continuous features. Faster alternatives use graph clustering with cosine distance on averaged continuous features or edit distance on discrete unit sequences. Through controlled experiments that isolate either the representations or the clustering method, we demonstrate that representation variability across segments of the same word type -- rather than clustering -- is the primary factor limiting performance.
>
---
#### [replaced 013] Speaking Clearly: A Simplified Whisper-Based Codec for Low-Bitrate Speech Coding
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决语义与声学保真之间的矛盾。通过简化Whisper模型，提出SimWhisper-Codec，在低比特率下实现更好的语义和声学质量。**

- **链接: [https://arxiv.org/pdf/2510.20504v2](https://arxiv.org/pdf/2510.20504v2)**

> **作者:** Xin Zhang; Lin Li; Xiangni Lu; Jianquan Liu; Kong Aik Lee
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Speech codecs serve as bridges between continuous speech signals and large language models, yet face an inherent conflict between acoustic fidelity and semantic preservation. To mitigate this conflict, prevailing methods augment acoustic codecs with complex semantic supervision. We explore the opposite direction: a semantic-first approach that starts from a semantically-capable model and adapts it for high-fidelity acoustic reconstruction. Through empirical analysis, we discover that targeted architectural simplification can unlock the acoustic modeling potential of Whisper, a text-aligned Automatic Speech Recognition (ASR) model. Based on this finding, we propose SimWhisper-Codec, a novel codec that balances the semantic and acoustic preservation by leveraging a frozen, simplified Whisper encoder without requiring external supervision. Experimental results demonstrate that SimWhisper-Codec achieves superior performance in both semantic preservation and acoustic quality compared to semantically-supervised codecs such as Mimi Codec and SpeechTokenizer at similar bitrates, validating the effectiveness of our semantic-first approach. Code is available at https://github.com/ZhangXinWhut/SimWhisper-Codec.
>
---
#### [replaced 014] SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于歌唱质量评估任务，旨在解决主观评估成本高、客观指标不全面的问题。提出SingMOS-Pro数据集，扩展了歌词、旋律等多维度标注，涵盖多种模型生成的歌声片段，为后续研究提供基准。**

- **链接: [https://arxiv.org/pdf/2510.01812v4](https://arxiv.org/pdf/2510.01812v4)**

> **作者:** Yuxun Tang; Lan Liu; Wenhao Feng; Yiwen Zhao; Jionghao Han; Yifeng Yu; Jiatong Shi; Qin Jin
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Singing voice generation progresses rapidly, yet evaluating singing quality remains a critical challenge. Human subjective assessment, typically in the form of listening tests, is costly and time consuming, while existing objective metrics capture only limited perceptual aspects. In this work, we introduce SingMOS-Pro, a dataset for automatic singing quality assessment. Building on our preview version SingMOS, which provides only overall ratings, SingMOS-Pro extends the annotations of the additional data to include lyrics, melody, and overall quality, offering broader coverage and greater diversity. The dataset contains 7,981 singing clips generated by 41 models across 12 datasets, spanning from early systems to recent state-of-the-art approaches. Each clip is rated by at least five experienced annotators to ensure reliability and consistency. Furthermore, we investigate strategies for effectively utilizing MOS data annotated under heterogeneous standards and benchmark several widely used evaluation methods from related tasks on SingMOS-Pro, establishing strong baselines and practical references for future research. The dataset is publicly available at https://huggingface.co/datasets/TangRain/SingMOS-Pro.
>
---
#### [replaced 015] SoundCompass: Navigating Target Sound Extraction With Effective Directional Clue Integration In Complex Acoustic Scenes
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于目标声音提取任务，解决复杂声场中空间信息丢失问题。提出SoundCompass框架，融合频段相关性和方向信息，提升目标声音提取效果。**

- **链接: [https://arxiv.org/pdf/2509.18561v3](https://arxiv.org/pdf/2509.18561v3)**

> **作者:** Dayun Choi; Jung-Woo Choi
>
> **备注:** 5 pages, 4 figures, accepted to ICASSP 2026
>
> **摘要:** Recent advances in target sound extraction (TSE) utilize directional clues derived from direction of arrival (DoA), which represent an inherent spatial property of sound available in any acoustic scene. However, previous DoA-based methods rely on hand-crafted features or discrete encodings, which lose fine-grained spatial information and limit adaptability. We propose SoundCompass, an effective directional clue integration framework centered on a Spectral Pairwise INteraction (SPIN) module that captures cross-channel spatial correlations in the complex spectrogram domain to preserve full spatial information in multichannel signals. The input feature expressed in terms of spatial correlations is fused with a DoA clue represented as spherical harmonics (SH) encoding. The fusion is carried out across overlapping frequency subbands, inheriting the benefits reported in the previous band-split architectures. We also incorporate the iterative refinement strategy, chain-of-inference (CoI), in the TSE framework, which recursively fuses DoA with sound event activation estimated from the previous inference stage. Experiments demonstrate that SoundCompass, combining SPIN, SH embedding, and CoI, robustly extracts target sources across diverse signal classes and spatial configurations.
>
---
#### [replaced 016] Adaptive Multimodal Person Recognition: A Robust Framework for Handling Missing Modalities
- **分类: cs.CV; cs.SD; eess.AS; eess.IV**

- **简介: 该论文属于人物识别任务，解决多模态数据缺失问题。提出一种融合上身运动、人脸和语音的框架，提升识别准确率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14961v3](https://arxiv.org/pdf/2512.14961v3)**

> **作者:** Aref Farhadipour; Teodora Vukovic; Volker Dellwo; Petr Motlicek; Srikanth Madikeri
>
> **备注:** 9 pages and 8 tables
>
> **摘要:** Person identification systems often rely on audio, visual, or behavioral cues, but real-world conditions frequently present with missing or degraded modalities. To address this challenge, we propose a multimodal person identification framework incorporating upper-body motion, face, and voice. Experimental results demonstrate that body motion outperforms traditional modalities such as face and voice in within-session evaluations, while serving as a complementary cue that enhances performance in multi-session scenarios. Our model employs a unified hybrid fusion strategy, fusing both feature-level and score-level information to maximize representational richness and decision accuracy. Specifically, it leverages multi-task learning to process modalities independently, followed by cross-attention and gated fusion mechanisms to exploit both unimodal information and cross-modal interactions. Finally, a confidence-weighted strategy and mistake-correction mechanism dynamically adapt to missing data, ensuring that our single classification head achieves optimal performance even in unimodal and bimodal scenarios. We evaluate our method on CANDOR, a newly introduced interview-based multimodal dataset, which we benchmark in this work for the first time. Our results demonstrate that the proposed trimodal system achieves 99.51% Top-1 accuracy on person identification tasks. In addition, we evaluate our model on the VoxCeleb1 dataset as a widely used evaluation protocol and reach 99.92% accuracy in bimodal mode, outperforming conventional approaches. Moreover, we show that our system maintains high accuracy even when one or two modalities are unavailable, making it a robust solution for real-world person recognition applications. The code and data for this work are publicly available.
>
---
#### [replaced 017] EDM2SE: A Magnitude-Preserving Network Architecture for Diffusion-Based Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，解决噪声环境下语音质量提升问题。提出EDM2SE网络，采用时变预处理和跳跃连接，实现噪声抑制与语音增强。**

- **链接: [https://arxiv.org/pdf/2505.05216v2](https://arxiv.org/pdf/2505.05216v2)**

> **作者:** Julius Richter; Danilo de Oliveira; Timo Gerkmann
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** We study diffusion-based speech enhancement using a Schrodinger bridge formulation and extend the EDM2 framework to this setting. We employ time-dependent preconditioning of network inputs and outputs to stabilize training and explore two skip-connection configurations that allow the network to predict either environmental noise or clean speech. To control activation and weight magnitudes, we adopt a magnitude-preserving architecture and learn the contribution of the noisy input within each network block for improved conditioning. We further analyze the impact of exponential moving average (EMA) parameter smoothing by approximating different EMA profiles post training, finding that, unlike in image generation, short or absent EMA consistently yields better speech enhancement performance. Experiments on VoiceBank-DEMAND and EARS-WHAM demonstrate competitive signal-to-distortion ratios and perceptual scores, with the two skip-connection variants exhibiting complementary strengths. These findings provide new insights into EMA behavior, magnitude preservation, and skip-connection design for diffusion-based speech enhancement.
>
---
