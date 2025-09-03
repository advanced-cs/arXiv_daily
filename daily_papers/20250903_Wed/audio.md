# 音频 cs.SD;  eess.SP

- **最新发布 45 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Adaptive Vehicle Speed Classification via BMCNN with Reinforcement Learning-Enhanced Acoustic Processing
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出基于BMCNN与强化学习的声学车辆速度分类框架，解决城市交通拥堵实时监测问题。通过双分支特征处理和注意力机制提升效率与准确率，适用于异构环境的智能交通系统。**

- **链接: [http://arxiv.org/pdf/2509.00839v1](http://arxiv.org/pdf/2509.00839v1)**

> **作者:** Yuli Zhang; Pengfei Fan; Ruiyuan Jiang; Hankang Gu; Dongyao Jia; Xinheng Wang
>
> **摘要:** Traffic congestion remains a pressing urban challenge, requiring intelligent transportation systems for real-time management. We present a hybrid framework that combines deep learning and reinforcement learning for acoustic vehicle speed classification. A dual-branch BMCNN processes MFCC and wavelet features to capture complementary frequency patterns. An attention-enhanced DQN adaptively selects the minimal number of audio frames and triggers early decisions once confidence thresholds are reached. Evaluations on IDMT-Traffic and our SZUR-Acoustic (Suzhou) datasets show 95.99% and 92.3% accuracy, with up to 1.63x faster average processing via early termination. Compared with A3C, DDDQN, SA2C, PPO, and TD3, the method provides a superior accuracy-efficiency trade-off and is suitable for real-time ITS deployment in heterogeneous urban environments.
>
---
#### [new 002] Evaluating the Effectiveness of Transformer Layers in Wav2Vec 2.0, XLS-R, and Whisper for Speaker Identification Tasks
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 论文评估Wav2Vec 2.0、XLS-R和Whisper在说话人识别任务中的表现，通过微调和层表示分析，比较其特征提取能力，确定最优层数以提升性能。**

- **链接: [http://arxiv.org/pdf/2509.00230v1](http://arxiv.org/pdf/2509.00230v1)**

> **作者:** Linus Stuhlmann; Michael Alexander Saxer
>
> **摘要:** This study evaluates the performance of three advanced speech encoder models, Wav2Vec 2.0, XLS-R, and Whisper, in speaker identification tasks. By fine-tuning these models and analyzing their layer-wise representations using SVCCA, k-means clustering, and t-SNE visualizations, we found that Wav2Vec 2.0 and XLS-R capture speaker-specific features effectively in their early layers, with fine-tuning improving stability and performance. Whisper showed better performance in deeper layers. Additionally, we determined the optimal number of transformer layers for each model when fine-tuned for speaker identification tasks.
>
---
#### [new 003] Algorithms for Collaborative Harmonization
- **分类: cs.SD; eess.AS**

- **简介: 论文研究音乐和声聚合任务，旨在整合多建议生成连贯和声。提出算法分析复杂度，发现Kemeny和多数投票算法效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.00120v1](http://arxiv.org/pdf/2509.00120v1)**

> **作者:** Eyal Briman; Eyal Leizerovich; Nimrod Talmon
>
> **备注:** Presented at the 15th Multidisciplinary Workshop on Advances in Preference Handling M-PREF 2024, Santiago de Compostela, Oct 20, 2024
>
> **摘要:** We consider a specific scenario of text aggregation, in the realm of musical harmonization. Musical harmonization shares similarities with text aggregation, however the language of harmony is more structured than general text. Concretely, given a set of harmonization suggestions for a given musical melody, our interest lies in devising aggregation algorithms that yield an harmonization sequence that satisfies the following two key criteria: (1) an effective representation of the collective suggestions; and (2) an harmonization that is musically coherent. We present different algorithms for the aggregation of harmonies given by a group of agents and analyze their complexities. The results indicate that the Kemeny and plurality-based algorithms are most effective in assessing representation and maintaining musical coherence.
>
---
#### [new 004] FireRedTTS-2: Towards Long Conversational Speech Generation for Podcast and Chatbot
- **分类: cs.SD; eess.AS**

- **简介: 论文提出FireRedTTS-2，解决多说话人长对话生成中的合成不稳定、说话人切换和语调问题，采用流式分词与双Transformer架构，提升播客及聊天机器人语音生成质量。**

- **链接: [http://arxiv.org/pdf/2509.02020v1](http://arxiv.org/pdf/2509.02020v1)**

> **作者:** Kun Xie; Feiyu Shen; Junjie Li; Fenglong Xie; Xu Tang; Yao Hu
>
> **摘要:** Current dialogue generation approaches typically require the complete dialogue text before synthesis and produce a single, inseparable speech containing all voices, making them unsuitable for interactive chat; moreover, they suffer from unstable synthesis, inaccurate speaker transitions, and incoherent prosody. In this work, we present FireRedTTS-2, a long-form streaming TTS system for multi-speaker dialogue generation, delivering stable, natural speech with reliable speaker switching and context-aware prosody. A new 12.5Hz streaming speech tokenizer accelerates training and inference, extends maximum dialogue length, encodes richer semantics to stabilize text-to-token modeling and supports high-fidelity streaming generation for real-time applications. We adopt a text-speech interleaved format, concatenating speaker-labeled text with aligned speech tokens in chronological order, and model it with a dual-transformer: a large decoder-only transformer predicts tokens at the first layer, and a smaller one completes subsequent layers. Experimental results show that FireRedTTS-2 integrates seamlessly with chat frameworks and, with minimal fine-tuning, produces emotionally expressive speech guided by implicit contextual cues. In podcast generation, it surpasses existing systems including MoonCast, Zipvoice-Dialogue, and MOSS-TTSD in objective intelligibility, speaker-turn reliability, and perceived naturalness with context-consistent prosody. Our demos are available at https://fireredteam.github.io/demos/firered_tts_2.
>
---
#### [new 005] The AudioMOS Challenge 2025
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍AudioMOS 2025挑战赛，旨在通过三类任务评估合成音频质量：文本到音乐的总体质量与文本对齐、基于美学维度的多类型音频评估、不同采样率的语音质量分析，推动自动音频生成系统评价技术发展。**

- **链接: [http://arxiv.org/pdf/2509.01336v1](http://arxiv.org/pdf/2509.01336v1)**

> **作者:** Wen-Chin Huang; Hui Wang; Cheng Liu; Yi-Chiao Wu; Andros Tjandra; Wei-Ning Hsu; Erica Cooper; Yong Qin; Tomoki Toda
>
> **备注:** IEEE ASRU 2025
>
> **摘要:** This is the summary paper for the AudioMOS Challenge 2025, the very first challenge for automatic subjective quality prediction for synthetic audio. The challenge consists of three tracks. The first track aims to assess text-to-music samples in terms of overall quality and textual alignment. The second track is based on the four evaluation dimensions of Meta Audiobox Aesthetics, and the test set consists of text-to-speech, text-to-audio, and text-to-music samples. The third track focuses on synthetic speech quality assessment in different sampling rates. The challenge attracted 24 unique teams from both academia and industry, and improvements over the baselines were confirmed. The outcome of this challenge is expected to facilitate development and progress in the field of automatic evaluation for audio generation systems.
>
---
#### [new 006] Towards High-Fidelity and Controllable Bioacoustic Generation via Enhanced Diffusion Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出BirdDiff框架，通过多尺度增强与扩散模型，从噪声数据生成高保真、可控的鸟类鸣叫，解决生物声学数据稀缺问题，提升信噪比与分类准确率。**

- **链接: [http://arxiv.org/pdf/2509.00318v1](http://arxiv.org/pdf/2509.00318v1)**

> **作者:** Tianyu Song; Ton Viet Ta
>
> **摘要:** Generative modeling offers new opportunities for bioacoustics, enabling the synthesis of realistic animal vocalizations that could support biomonitoring efforts and supplement scarce data for endangered species. However, directly generating bird call waveforms from noisy field recordings remains a major challenge. We propose BirdDiff, a generative framework designed to synthesize bird calls from a noisy dataset of 12 wild bird species. The model incorporates a "zeroth layer" stage for multi-scale adaptive bird-call enhancement, followed by a diffusion-based generator conditioned on three modalities: Mel-frequency cepstral coefficients, species labels, and textual descriptions. The enhancement stage improves signal-to-noise ratio (SNR) while minimizing spectral distortion, achieving the highest SNR gain (+10.45 dB) and lowest Itakura-Saito Distance (0.54) compared to three widely used non-training enhancement methods. We evaluate BirdDiff against a baseline generative model, DiffWave. Our method yields substantial improvements in generative quality metrics: Fr\'echet Audio Distance (0.590 to 0.213), Jensen-Shannon Divergence (0.259 to 0.226), and Number of Statistically-Different Bins (7.33 to 5.58). To assess species-specific detail preservation, we use a ResNet50 classifier trained on the original dataset to identify generated samples. Classification accuracy improves from 35.9% (DiffWave) to 70.1% (BirdDiff), with 8 of 12 species exceeding 70% accuracy. These results demonstrate that BirdDiff enables high-fidelity, controllable bird call generation directly from noisy field recordings.
>
---
#### [new 007] A Unified Denoising and Adaptation Framework for Self-Supervised Bengali Dialectal ASR
- **分类: cs.SD; eess.AS**

- **简介: 论文提出统一框架，结合WavLM去噪预训练与多阶段微调，解决孟加拉语方言ASR中的方言多样性与噪声干扰问题，性能优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.00988v1](http://arxiv.org/pdf/2509.00988v1)**

> **作者:** Swadhin Biswas; Imran; Tuhin Sheikh
>
> **摘要:** Automatic Speech Recognition (ASR) for Bengali, the world's fifth most spoken language, remains a significant challenge, critically hindering technological accessibility for its over 270 million speakers. This challenge is compounded by two persistent and intertwined factors: the language's vast dialectal diversity and the prevalence of acoustic noise in real-world environments. While state-of-the-art self-supervised learning (SSL) models have advanced ASR for low-resource languages, they often lack explicit mechanisms to handle environmental noise during pre-training or specialized adaptation strategies for the complex phonetic and lexical variations across Bengali dialects. This paper introduces a novel, unified framework designed to address these dual challenges simultaneously. Our approach is founded on the WavLM model, which is uniquely pre-trained with a masked speech denoising objective, making it inherently robust to acoustic distortions. We propose a specialized multi-stage fine-tuning strategy that first adapts the model to general-domain standard Bengali to establish a strong linguistic foundation and subsequently specializes it for noise-robust dialectal recognition through targeted data augmentation. The framework is rigorously evaluated on a comprehensive benchmark comprising multiple Bengali dialects under a wide range of simulated noisy conditions, from clean audio to low Signal-to-Noise Ratio (SNR) levels. Experimental results demonstrate that the proposed framework significantly outperforms strong baselines, including standard fine-tuned wav2vec 2.0 and the large-scale multilingual Whisper model. This work establishes a new state-of-the-art for this task and provides a scalable, effective blueprint for developing practical ASR systems for other low-resource, high-variation languages globally.
>
---
#### [new 008] Music Genre Classification Using Machine Learning Techniques
- **分类: cs.SD; cs.LG**

- **简介: 该论文比较SVM、集成方法与CNN在音乐流派分类中的表现，发现SVM因特征工程优势在GTZAN数据集上更优，强调传统方法在有限数据下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.01762v1](http://arxiv.org/pdf/2509.01762v1)**

> **作者:** Alokit Mishra; Ryyan Akhtar
>
> **备注:** 10 pages, 20 figures. Submitted in partial fulfillment of the requirements for the Bachelor of Technology (B.Tech) degree in Artificial Intelligence and Data Science
>
> **摘要:** This paper presents a comparative analysis of machine learning methodologies for automatic music genre classification. We evaluate the performance of classical classifiers, including Support Vector Machines (SVM) and ensemble methods, trained on a comprehensive set of hand-crafted audio features, against a Convolutional Neural Network (CNN) operating on Mel spectrograms. The study is conducted on the widely-used GTZAN dataset. Our findings demonstrate a noteworthy result: the SVM, leveraging domain-specific feature engineering, achieves superior classification accuracy compared to the end-to-end CNN model. We attribute this outcome to the data-constrained nature of the benchmark dataset, where the strong inductive bias of engineered features provides a regularization effect that mitigates the risk of overfitting inherent in high-capacity deep learning models. This work underscores the enduring relevance of traditional feature extraction in practical audio processing tasks and provides a critical perspective on the universal applicability of deep learning, especially for moderately sized datasets.
>
---
#### [new 009] EZhouNet:A framework based on graph neural network and anchor interval for the respiratory sound event detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出EZhouNet框架，结合图神经网络与锚区间技术，解决呼吸音事件检测中变长音频处理、时间边界学习困难及位置信息利用不足的问题，提升检测精度与适用性。**

- **链接: [http://arxiv.org/pdf/2509.01153v1](http://arxiv.org/pdf/2509.01153v1)**

> **作者:** Yun Chu; Qiuhao Wang; Enze Zhou; Qian Liu; Gang Zheng
>
> **摘要:** Auscultation is a key method for early diagnosis of respiratory and pulmonary diseases, relying on skilled healthcare professionals. However, the process is often subjective, with variability between experts. As a result, numerous deep learning-based automatic classification methods have emerged, most of which focus on respiratory sound classification. In contrast, research on respiratory sound event detection remains limited. Existing sound event detection methods typically rely on frame-level predictions followed by post-processing to generate event-level outputs, making interval boundaries challenging to learn directly. Furthermore, many approaches can only handle fixed-length audio, lim- iting their applicability to variable-length respiratory sounds. Additionally, the impact of respiratory sound location information on detection performance has not been extensively explored. To address these issues, we propose a graph neural network-based framework with anchor intervals, capable of handling variable-length audio and providing more precise temporal localization for abnormal respi- ratory sound events. Our method improves both the flexibility and applicability of respiratory sound detection. Experiments on the SPRSound 2024 and HF Lung V1 datasets demonstrate the effec- tiveness of the proposed approach, and incorporating respiratory position information enhances the discrimination between abnormal sounds.
>
---
#### [new 010] PicoAudio2: Temporal Controllable Text-to-Audio Generation with Natural Language Description
- **分类: cs.SD; eess.AS; 68Txx; I.2**

- **简介: 该论文提出PicoAudio2，解决文本到音频生成中时间控制有限和音频质量不足的问题。通过融合真实与模拟数据，引入时间戳矩阵，提升细粒度控制与音频质量。**

- **链接: [http://arxiv.org/pdf/2509.00683v1](http://arxiv.org/pdf/2509.00683v1)**

> **作者:** Zihao Zheng; Zeyu Xie; Xuenan Xu; Wen Wu; Chao Zhang; Mengyue Wu
>
> **备注:** Demo page: https://HiRookie9.github.io/PicoAudio2-Page
>
> **摘要:** Controllable text-to-audio generation (TTA) has attracted much attention recently. Although existing works can achieve fine-grained controllability based on timestamp information, sound event categories are limited to a fixed set. Moreover, since only simulated data is used for training, the generated audio quality and generalization performance on real data are limited. To tackle this issue, we propose PicoAudio2, improving temporal-controllable TTA via a new data processing pipeline and model architecture. Specifically, we use a grounding model to annotate event timestamps of real audio-text datasets to curate temporally-strong real data, in addition to simulation data from existing works. The model is trained on the combination of real and simulation data. Moreover, following PicoAudio, we encode timestamp information into a timestamp matrix to provide extra fine-grained time-aligned information to the model, on top of the coarse-grained textual description. Experiments show that PicoAudio2 exhibits superior performance in terms of temporal controllability and audio quality.
>
---
#### [new 011] Spectrogram Patch Codec: A 2D Block-Quantized VQ-VAE and HiFi-GAN for Neural Speech Coding
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出一种低延迟语音编码方法，通过二维块量化梅尔谱图，采用单阶段VQ-VAE与HiFi-GAN结合，简化结构并降低比特率至7.5kbps，实现高保真语音合成，挑战传统复杂RVQ方案。**

- **链接: [http://arxiv.org/pdf/2509.02244v1](http://arxiv.org/pdf/2509.02244v1)**

> **作者:** Luis Felipe Chary; Miguel Arjona Ramirez
>
> **摘要:** We present a neural speech codec that challenges the need for complex residual vector quantization (RVQ) stacks by introducing a simpler, single-stage quantization approach. Our method operates directly on the mel-spectrogram, treating it as a 2D data and quantizing non-overlapping 4x4 patches into a single, shared codebook. This patchwise design simplifies the architecture, enables low-latency streaming, and yields a discrete latent grid. To ensure high-fidelity synthesis, we employ a late-stage adversarial fine-tuning for the VQ-VAE and train a HiFi-GAN vocoder from scratch on the codec's reconstructed spectrograms. Operating at approximately 7.5 kbits/s for 16 kHz speech, our system was evaluated against several state-of-the-art neural codecs using objective metrics such as STOI, PESQ, MCD, and ViSQOL. The results demonstrate that our simplified, non-residual architecture achieves competitive perceptual quality and intelligibility, validating it as an effective and open foundation for future low-latency codec designs.
>
---
#### [new 012] AudioRWKV: Efficient and Stable Bidirectional RWKV for Audio Pattern Recognition
- **分类: cs.SD**

- **简介: 论文提出AudioRWKV，针对音频建模中Transformer的高复杂度和Mamba的不稳定性，设计双向WKV结构，结合2D卷积，实现高效稳定处理，实验显示性能与大模型相当，处理长音频更快。**

- **链接: [http://arxiv.org/pdf/2509.02167v1](http://arxiv.org/pdf/2509.02167v1)**

> **作者:** Jiayu Xiong; Jun Xue; Jianlong Kwan; Jing Wang
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Recently, Transformers (e.g., Audio Spectrogram Transformers, AST) and state-space models (e.g., Audio Mamba, AuM) have achieved remarkable progress in audio modeling. However, the O(L^2) computational complexity of the Transformer architecture hinders efficient long-sequence processing, while the Mamba architecture tends to become unstable when scaling parameters and data. To address these challenges, this paper proposes AudioRWKV (A-RWKV), a highly efficient and stable architecture for audio modeling. Specifically, we inherit the stable and efficient recurrent formulation of RWKV7 and replace its 1D token-shift operation with a 2D depthwise separable convolution to better capture local spectro-temporal patterns. Furthermore, we adapt the original causal WKV kernel into a bidirectional WKV kernel (Bi-WKV), enabling global context modeling over the entire audio sequence while maintaining linear computational complexity. Benefiting from the inherent stability of the RWKV7 foundation, A-RWKV scales seamlessly to larger model sizes. Experimental results demonstrate that, under the same linear-model regime, A-RWKV-S (22M) achieves performance parity with AuM-B (92M) while exhibiting more stable throughput than AST; for long-form audio (~5 minutes 28 seconds), WKV7 achieves up to a 13.3X speedup in processing.
>
---
#### [new 013] A Survey on Evaluation Metrics for Music Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 本文综述音乐生成评估指标，解决现有方法在客观性、文化偏见和标准化方面的不足，提出分类框架及未来研究方向，以构建全面评估体系。**

- **链接: [http://arxiv.org/pdf/2509.00051v1](http://arxiv.org/pdf/2509.00051v1)**

> **作者:** Faria Binte Kader; Santu Karmaker
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Despite significant advancements in music generation systems, the methodologies for evaluating generated music have not progressed as expected due to the complex nature of music, with aspects such as structure, coherence, creativity, and emotional expressiveness. In this paper, we shed light on this research gap, introducing a detailed taxonomy for evaluation metrics for both audio and symbolic music representations. We include a critical review identifying major limitations in current evaluation methodologies which includes poor correlation between objective metrics and human perception, cross-cultural bias, and lack of standardization that hinders cross-model comparisons. Addressing these gaps, we further propose future research directions towards building a comprehensive evaluation framework for music generation evaluation.
>
---
#### [new 014] SaD: A Scenario-Aware Discriminator for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 论文提出场景感知判别器（SaD）用于语音增强，解决现有方法忽略场景信息的问题。通过捕捉场景特征并频域分割，提升生成语音质量评估，实验证明其适应不同生成器架构，增强效果。**

- **链接: [http://arxiv.org/pdf/2509.00405v1](http://arxiv.org/pdf/2509.00405v1)**

> **作者:** Xihao Yuan; Siqi Liu; Yan Chen; Hang Zhou; Chang Liu; Hanting Chen; Jie Hu
>
> **备注:** 5 pages, 2 figures.Accepted by InterSpeech2025
>
> **摘要:** Generative adversarial network-based models have shown remarkable performance in the field of speech enhancement. However, the current optimization strategies for these models predominantly focus on refining the architecture of the generator or enhancing the quality evaluation metrics of the discriminator. This approach often overlooks the rich contextual information inherent in diverse scenarios. In this paper, we propose a scenario-aware discriminator that captures scene-specific features and performs frequency-domain division, thereby enabling a more accurate quality assessment of the enhanced speech generated by the generator. We conducted comprehensive experiments on three representative models using two publicly available datasets. The results demonstrate that our method can effectively adapt to various generator architectures without altering their structure, thereby unlocking further performance gains in speech enhancement across different scenarios.
>
---
#### [new 015] CabinSep: IR-Augmented Mask-Based MVDR for Real-Time In-Car Speech Separation with Distributed Heterogeneous Arrays
- **分类: cs.SD; cs.AI; cs.HC; eess.AS**

- **简介: 该论文提出CabinSep，用于实时车内多说话者语音分离，解决重叠语音导致的ASR错误。通过空间特征提取、MVDR推理及混合数据增强，实现低复杂度与17.5%错误率降低。**

- **链接: [http://arxiv.org/pdf/2509.01399v1](http://arxiv.org/pdf/2509.01399v1)**

> **作者:** Runduo Han; Yanxin Hu; Yihui Fu; Zihan Zhang; Yukai Jv; Li Chen; Lei Xie
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Separating overlapping speech from multiple speakers is crucial for effective human-vehicle interaction. This paper proposes CabinSep, a lightweight neural mask-based minimum variance distortionless response (MVDR) speech separation approach, to reduce speech recognition errors in back-end automatic speech recognition (ASR) models. Our contributions are threefold: First, we utilize channel information to extract spatial features, which improves the estimation of speech and noise masks. Second, we employ MVDR during inference, reducing speech distortion to make it more ASR-friendly. Third, we introduce a data augmentation method combining simulated and real-recorded impulse responses (IRs), improving speaker localization at zone boundaries and further reducing speech recognition errors. With a computational complexity of only 0.4 GMACs, CabinSep achieves a 17.5% relative reduction in speech recognition error rate in a real-recorded dataset compared to the state-of-the-art DualSep model. Demos are available at: https://cabinsep.github.io/cabinsep/.
>
---
#### [new 016] Speech Command Recognition Using LogNNet Reservoir Computing for Embedded Systems
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出低资源语音命令识别方法，针对嵌入式系统资源限制，结合能量基VAD、优化MFCC和LogNNet分类器，在Arduino上实现92.04%准确率与18KB内存占用，适用于物联网设备。**

- **链接: [http://arxiv.org/pdf/2509.00862v1](http://arxiv.org/pdf/2509.00862v1)**

> **作者:** Yuriy Izotov; Andrei Velichko
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** This paper presents a low-resource speech-command recognizer combining energy-based voice activity detection (VAD), an optimized Mel-Frequency Cepstral Coefficients (MFCC) pipeline, and the LogNNet reservoir-computing classifier. Using four commands from the Speech Commands da-taset downsampled to 8 kHz, we evaluate four MFCC aggregation schemes and find that adaptive binning (64-dimensional feature vector) offers the best accuracy-to-compactness trade-off. The LogNNet classifier with architecture 64:33:9:4 reaches 92.04% accuracy under speaker-independent evaluation, while requiring significantly fewer parameters than conventional deep learn-ing models. Hardware implementation on Arduino Nano 33 IoT (ARM Cor-tex-M0+, 48 MHz, 32 KB RAM) validates the practical feasibility, achieving ~90% real-time recognition accuracy while consuming only 18 KB RAM (55% utilization). The complete pipeline (VAD -> MFCC -> LogNNet) thus enables reliable on-device speech-command recognition under strict memory and compute limits, making it suitable for battery-powered IoT nodes, wire-less sensor networks, and hands-free control interfaces.
>
---
#### [new 017] AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出AudioCodecBench基准框架，解决音频编解码器评估标准不统一、维度单一的问题。通过定义语义/声学标记，从重建质量、代码本稳定性、解码器困惑度和下游任务表现四个维度系统评估编解码器性能，建立全面对比基准。**

- **链接: [http://arxiv.org/pdf/2509.02349v1](http://arxiv.org/pdf/2509.02349v1)**

> **作者:** Lu Wang; Hao Chen; Siyu Wu; Zhiyue Wu; Hao Zhou; Chengfeng Zhang; Ting Wang; Haodi Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been widely applied in speech and music. This tendency has led to a focus on audio tokenization for Large Models (LMs). Unlike semantic-only text tokens, audio tokens must both capture global semantic content and preserve fine-grained acoustic details. Moreover, they provide a discrete method for speech and music that can be effectively integrated into MLLMs. However, existing research is unsuitable in the definitions of semantic tokens and acoustic tokens. In addition, the evaluation of different codecs typically concentrates on specific domains or tasks, such as reconstruction or Automatic Speech Recognition (ASR) task, which prevents fair and comprehensive comparisons. To address these problems, this paper provides suitable definitions for semantic and acoustic tokens and introduces a systematic evaluation framework. This framework allows for a comprehensive assessment of codecs' capabilities which evaluate across four dimensions: audio reconstruction metric, codebook index (ID) stability, decoder-only transformer perplexity, and performance on downstream probe tasks. Our results show the correctness of the provided suitable definitions and the correlation among reconstruction metrics, codebook ID stability, downstream probe tasks and perplexity.
>
---
#### [new 018] TinyMusician: On-Device Music Generation with Knowledge Distillation and Mixed Precision Quantization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出TinyMusician，通过知识蒸馏与混合精度量化，解决大模型在边缘设备部署的高资源需求问题，实现轻量化音乐生成，保持93%性能，模型体积减少55%。**

- **链接: [http://arxiv.org/pdf/2509.00914v1](http://arxiv.org/pdf/2509.00914v1)**

> **作者:** Hainan Wang; Mehdi Hosseinzadeh; Reza Rawassizadeh
>
> **备注:** 12 pages for main context, 5 figures
>
> **摘要:** The success of the generative model has gained unprecedented attention in the music generation area. Transformer-based architectures have set new benchmarks for model performance. However, their practical adoption is hindered by some critical challenges: the demand for massive computational resources and inference time, due to their large number of parameters. These obstacles make them infeasible to deploy on edge devices, such as smartphones and wearables, with limited computational resources. In this work, we present TinyMusician, a lightweight music generation model distilled from MusicGen (a State-of-the-art music generation model). TinyMusician integrates two innovations: (i) Stage-mixed Bidirectional and Skewed KL-Divergence and (ii) Adaptive Mixed-Precision Quantization. The experimental results demonstrate that TinyMusician retains 93% of the MusicGen-Small performance with 55% less model size. TinyMusician is the first mobile-deployable music generation model that eliminates cloud dependency while maintaining high audio fidelity and efficient resource usage
>
---
#### [new 019] CoComposer: LLM Multi-agent Collaborative Music Composition
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出CoComposer多智能体系统，解决AI音乐生成在时长、质量与可控性上的局限。通过协作代理与LLM结合，提升音乐质量与生产复杂度，实验表明其在可解释性与编辑性上优于非LLM系统，虽音乐质量略逊于MusicLM。**

- **链接: [http://arxiv.org/pdf/2509.00132v1](http://arxiv.org/pdf/2509.00132v1)**

> **作者:** Peiwen Xing; Aske Plaat; Niki van Stein
>
> **摘要:** Existing AI Music composition tools are limited in generation duration, musical quality, and controllability. We introduce CoComposer, a multi-agent system that consists of five collaborating agents, each with a task based on the traditional music composition workflow. Using the AudioBox-Aesthetics system, we experimentally evaluate CoComposer on four compositional criteria. We test with three LLMs (GPT-4o, DeepSeek-V3-0324, Gemini-2.5-Flash), and find (1) that CoComposer outperforms existing multi-agent LLM-based systems in music quality, and (2) compared to a single-agent system, in production complexity. Compared to non- LLM MusicLM, CoComposer has better interpretability and editability, although MusicLM still produces better music.
>
---
#### [new 020] From Sound to Sight: Towards AI-authored Music Videos
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出AI自动生成音乐视频的方法，解决传统可视化系统表达有限的问题。通过潜在特征分析音频，生成文本描述并用生成模型制作视频，结合用户评估验证叙事性、连贯性和情感匹配。**

- **链接: [http://arxiv.org/pdf/2509.00029v1](http://arxiv.org/pdf/2509.00029v1)**

> **作者:** Leo Vitasovic; Stella Graßhof; Agnes Mercedes Kloft; Ville V. Lehtola; Martin Cunneen; Justyna Starostka; Glenn McGarry; Kun Li; Sami S. Brandt
>
> **备注:** 1st Workshop on Generative AI for Storytelling (AISTORY), 2025
>
> **摘要:** Conventional music visualisation systems rely on handcrafted ad hoc transformations of shapes and colours that offer only limited expressiveness. We propose two novel pipelines for automatically generating music videos from any user-specified, vocal or instrumental song using off-the-shelf deep learning models. Inspired by the manual workflows of music video producers, we experiment on how well latent feature-based techniques can analyse audio to detect musical qualities, such as emotional cues and instrumental patterns, and distil them into textual scene descriptions using a language model. Next, we employ a generative model to produce the corresponding video clips. To assess the generated videos, we identify several critical aspects and design and conduct a preliminary user evaluation that demonstrates storytelling potential, visual coherency and emotional alignment with the music. Our findings underscore the potential of latent feature techniques and deep generative models to expand music visualisation beyond traditional approaches.
>
---
#### [new 021] ArabEmoNet: A Lightweight Hybrid 2D CNN-BiLSTM Model with Attention for Robust Arabic Speech Emotion Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出ArabEmoNet，解决低资源阿拉伯语语音情感识别中的数据不足和传统方法时频模式捕捉不足问题，设计轻量级2D CNN-BiLSTM混合模型，结合注意力机制，参数减少90倍，提升效率。**

- **链接: [http://arxiv.org/pdf/2509.01401v1](http://arxiv.org/pdf/2509.01401v1)**

> **作者:** Ali Abouzeid; Bilal Elbouardi; Mohamed Maged; Shady Shehata
>
> **备注:** Accepted (The Third Arabic Natural Language Processing Conference)
>
> **摘要:** Speech emotion recognition is vital for human-computer interaction, particularly for low-resource languages like Arabic, which face challenges due to limited data and research. We introduce ArabEmoNet, a lightweight architecture designed to overcome these limitations and deliver state-of-the-art performance. Unlike previous systems relying on discrete MFCC features and 1D convolutions, which miss nuanced spectro-temporal patterns, ArabEmoNet uses Mel spectrograms processed through 2D convolutions, preserving critical emotional cues often lost in traditional methods. While recent models favor large-scale architectures with millions of parameters, ArabEmoNet achieves superior results with just 1 million parameters, 90 times smaller than HuBERT base and 74 times smaller than Whisper. This efficiency makes it ideal for resource-constrained environments. ArabEmoNet advances Arabic speech emotion recognition, offering exceptional performance and accessibility for real-world applications.
>
---
#### [new 022] Speech transformer models for extracting information from baby cries
- **分类: cs.SD; cs.LG; stat.AP**

- **简介: 该论文研究迁移学习在婴儿哭声分析中的应用，解决预训练语音模型在非语音数据上的适用性及编码信息问题。通过评估五种模型在八个数据集上的表现，发现其能有效分类哭声并提取声源不稳定性和身份信息，为情感检测等任务提供设计参考。**

- **链接: [http://arxiv.org/pdf/2509.02259v1](http://arxiv.org/pdf/2509.02259v1)**

> **作者:** Guillem Bonafos; Jéremy Rouch; Lény Lego; David Reby; Hugues Patural; Nicolas Mathevon; Rémy Emonet
>
> **备注:** Accepted to WOCCI2025 (interspeech2025 workshop)
>
> **摘要:** Transfer learning using latent representations from pre-trained speech models achieves outstanding performance in tasks where labeled data is scarce. However, their applicability to non-speech data and the specific acoustic properties encoded in these representations remain largely unexplored. In this study, we investigate both aspects. We evaluate five pre-trained speech models on eight baby cries datasets, encompassing 115 hours of audio from 960 babies. For each dataset, we assess the latent representations of each model across all available classification tasks. Our results demonstrate that the latent representations of these models can effectively classify human baby cries and encode key information related to vocal source instability and identity of the crying baby. In addition, a comparison of the architectures and training strategies of these models offers valuable insights for the design of future models tailored to similar tasks, such as emotion detection.
>
---
#### [new 023] High-Density MIMO Localization Using a 32x64 Ultrasonic Transducer-Microphone Array with Real-Time Data Streaming
- **分类: eess.SP; eess.AS**

- **简介: 论文提出基于32x64超声阵列的MIMO定位系统，通过随机相位多频信号提升信道分离与抗多径能力，解决高精度定位问题，经仿真验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.01210v1](http://arxiv.org/pdf/2509.01210v1)**

> **作者:** Rens Baeyens; Dennis Laurijssen; Jan Steckel; Walter Daems
>
> **备注:** Accepted for publication at IEEE IUS 2025
>
> **摘要:** In this work, we present a novel ultrasonic array system designed for high-precision localization using a large-scale MIMO (Multiple-Input Multiple-Output) architecture. The system combines 32 transmitters with 62 microphones, creating an extended virtual aperture that improves channel separability and spatial resolution. Each transmitter is excited by a random-phase multisine within the ultrasonic band, which reduces inter-channel correlation and increases robustness against multipath. The feasibility of the approach is demonstrated through simulations of reflector imaging and analysis of channel separation under realistic transducer bandwidth constraints. Results show that MIMO processing enables improved separation of reflectors compared to single-emitter configurations, although practical limitations such as transducer bandwidth reduce the achievable channel isolation.
>
---
#### [new 024] ESTM: An Enhanced Dual-Branch Spectral-Temporal Mamba for Anomalous Sound Detection
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出ESTM框架，针对工业设备异常声音检测任务，解决现有方法在长程时序和跨频带耦合建模上的不足。通过双路径Mamba架构、SSM和TSG模块融合多特征，提升检测性能，在DCASE 2020数据集上验证有效。**

- **链接: [http://arxiv.org/pdf/2509.02471v1](http://arxiv.org/pdf/2509.02471v1)**

> **作者:** Chengyuan Ma; Peng Jia; Hongyue Guo; Wenming Yang
>
> **备注:** Accepted in IEEE Signal Processing Letters 2025
>
> **摘要:** The core challenge in industrial equipment anoma lous sound detection (ASD) lies in modeling the time-frequency coupling characteristics of acoustic features. Existing modeling methods are limited by local receptive fields, making it difficult to capture long-range temporal patterns and cross-band dynamic coupling effects in machine acoustic features. In this paper, we propose a novel framework, ESTM, which is based on a dual-path Mamba architecture with time-frequency decoupled modeling and utilizes Selective State-Space Models (SSM) for long-range sequence modeling. ESTM extracts rich feature representations from different time segments and frequency bands by fusing enhanced Mel spectrograms and raw audio features, while further improving sensitivity to anomalous patterns through the TriStat-Gating (TSG) module. Our experiments demonstrate that ESTM improves anomalous detection performance on the DCASE 2020 Task 2 dataset, further validating the effectiveness of the proposed method.
>
---
#### [new 025] The Name-Free Gap: Policy-Aware Stylistic Control in Music Generation
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文研究音乐生成中的政策合规风格控制，通过大语言模型生成的描述符替代艺术家名字，评估其效果，发现描述符能部分恢复控制效果，揭示政策限制的局限性，并提出新指标。**

- **链接: [http://arxiv.org/pdf/2509.00654v1](http://arxiv.org/pdf/2509.00654v1)**

> **作者:** Ashwin Nagarajan; Hao-Wen Dong
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Text-to-music models capture broad attributes such as instrumentation or mood, but fine-grained stylistic control remains an open challenge. Existing stylization methods typically require retraining or specialized conditioning, which complicates reproducibility and limits policy compliance when artist names are restricted. We study whether lightweight, human-readable modifiers sampled from a large language model can provide a policy-robust alternative for stylistic control. Using MusicGen-small, we evaluate two artists: Billie Eilish (vocal pop) and Ludovico Einaudi (instrumental piano). For each artist, we use fifteen reference excerpts and evaluate matched seeds under three conditions: baseline prompts, artist-name prompts, and five descriptor sets. All prompts are generated using a large language model. Evaluation uses both VGGish and CLAP embeddings with distributional and per-clip similarity measures, including a new min-distance attribution metric. Results show that artist names are the strongest control signal across both artists, while name-free descriptors recover much of this effect. This highlights that existing safeguards such as the restriction of artist names in music generation prompts may not fully prevent style imitation. Cross-artist transfers reduce alignment, showing that descriptors encode targeted stylistic cues. We also present a descriptor table across ten contemporary artists to illustrate the breadth of the tokens. Together these findings define the name-free gap, the controllability difference between artist-name prompts and policy-compliant descriptors, shown through a reproducible evaluation protocol for prompt-level controllability.
>
---
#### [new 026] TTA-Bench: A Comprehensive Benchmark for Evaluating Text-to-Audio Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出TTA-Bench基准，解决现有文本到音频模型评估方法片面的问题，通过多维度（准确性、鲁棒性、公平性等）和大规模人工标注，全面评估十款模型，推动负责任的TTA系统评价。**

- **链接: [http://arxiv.org/pdf/2509.02398v1](http://arxiv.org/pdf/2509.02398v1)**

> **作者:** Hui Wang; Cheng Liu; Junyang Chen; Haoze Liu; Yuhang Jia; Shiwan Zhao; Jiaming Zhou; Haoqin Sun; Hui Bu; Yong Qin
>
> **摘要:** Text-to-Audio (TTA) generation has made rapid progress, but current evaluation methods remain narrow, focusing mainly on perceptual quality while overlooking robustness, generalization, and ethical concerns. We present TTA-Bench, a comprehensive benchmark for evaluating TTA models across functional performance, reliability, and social responsibility. It covers seven dimensions including accuracy, robustness, fairness, and toxicity, and includes 2,999 diverse prompts generated through automated and manual methods. We introduce a unified evaluation protocol that combines objective metrics with over 118,000 human annotations from both experts and general users. Ten state-of-the-art models are benchmarked under this framework, offering detailed insights into their strengths and limitations. TTA-Bench establishes a new standard for holistic and responsible evaluation of TTA systems. The dataset and evaluation tools are open-sourced at https://nku-hlt.github.io/tta-bench/.
>
---
#### [new 027] AImoclips: A Benchmark for Evaluating Emotion Conveyance in Text-to-Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出AImoclips基准测试，评估文本到音乐生成系统的情感传达效果。针对现有系统情感保真度不足的问题，通过对比开源与商业模型，分析情绪意图传达准确性，揭示模型在情感控制上的局限性。**

- **链接: [http://arxiv.org/pdf/2509.00813v1](http://arxiv.org/pdf/2509.00813v1)**

> **作者:** Gyehun Go; Satbyul Han; Ahyeon Choi; Eunjin Choi; Juhan Nam; Jeong Mi Park
>
> **备注:** to be published in HCMIR25: 3rd Workshop on Human-Centric Music Information Research
>
> **摘要:** Recent advances in text-to-music (TTM) generation have enabled controllable and expressive music creation using natural language prompts. However, the emotional fidelity of TTM systems remains largely underexplored compared to human preference or text alignment. In this study, we introduce AImoclips, a benchmark for evaluating how well TTM systems convey intended emotions to human listeners, covering both open-source and commercial models. We selected 12 emotion intents spanning four quadrants of the valence-arousal space, and used six state-of-the-art TTM systems to generate over 1,000 music clips. A total of 111 participants rated the perceived valence and arousal of each clip on a 9-point Likert scale. Our results show that commercial systems tend to produce music perceived as more pleasant than intended, while open-source systems tend to perform the opposite. Emotions are more accurately conveyed under high-arousal conditions across all models. Additionally, all systems exhibit a bias toward emotional neutrality, highlighting a key limitation in affective controllability. This benchmark offers valuable insights into model-specific emotion rendering characteristics and supports future development of emotionally aligned TTM systems.
>
---
#### [new 028] FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出FLM-Audio模型，通过自然独白与双阶段训练解决全双工对话中文本-音频对齐难题，提升响应速度与对话质量。**

- **链接: [http://arxiv.org/pdf/2509.02521v1](http://arxiv.org/pdf/2509.02521v1)**

> **作者:** Yiqun Yao; Xiang Li; Xin Jiang; Xuezhi Fang; Naitong Yu; Wenjia Ma; Aixin Sun; Yequan Wang
>
> **摘要:** Full-duplex dialog models are designed to listen and speak simultaneously with rapid responses to fast-changing user input. Among existing approaches, native full-duplex models merges different channels (e.g. listen and speak) in a single time step, overcoming the high response latency inherent to time-division multiplexing time-division multiplexing (TDM) alternatives. Yet, a key challenge remains: aligning textual monologues with audio streams that operate at different bitrates. The prevailing solution relies on word-level alignment, but this can degrade the language ability of large pre-trained models. Moreover, it requires highly accurate timestamps for every token, which introduces cascading errors and increases pre-processing costs. In this paper, we propose textual monologues in continuous tokens sequence, namely "natural" monologues, which mimics humanoid cognitive behavior in dialogs. For temporal alignment, we alternate the position of the natural monologue - leading or trailing the audio - across different training stages. This "dual" training paradigm proves highly effective in building FLM-Audio, our 7B spoken dialog model that demonstrates superior responsiveness, duplexity, and chatting experiences, as confirmed by experimental results.
>
---
#### [new 029] From Discord to Harmony: Decomposed Consonance-based Training for Improved Audio Chord Estimation
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文聚焦音频和弦估计任务，旨在解决注释不一致与类别不平衡导致的模型性能瓶颈。提出基于和谐度量的标签平滑方法，通过分解根音、低音和音符激活，提升模型对和弦标注的感知一致性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.01588v1](http://arxiv.org/pdf/2509.01588v1)**

> **作者:** Andrea Poltronieri; Xavier Serra; Martín Rocamora
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Audio Chord Estimation (ACE) holds a pivotal role in music information research, having garnered attention for over two decades due to its relevance for music transcription and analysis. Despite notable advancements, challenges persist in the task, particularly concerning unique characteristics of harmonic content, which have resulted in existing systems' performances reaching a glass ceiling. These challenges include annotator subjectivity, where varying interpretations among annotators lead to inconsistencies, and class imbalance within chord datasets, where certain chord classes are over-represented compared to others, posing difficulties in model training and evaluation. As a first contribution, this paper presents an evaluation of inter-annotator agreement in chord annotations, using metrics that extend beyond traditional binary measures. In addition, we propose a consonance-informed distance metric that reflects the perceptual similarity between harmonic annotations. Our analysis suggests that consonance-based distance metrics more effectively capture musically meaningful agreement between annotations. Expanding on these findings, we introduce a novel ACE conformer-based model that integrates consonance concepts into the model through consonance-based label smoothing. The proposed model also addresses class imbalance by separately estimating root, bass, and all note activations, enabling the reconstruction of chord labels from decomposed outputs.
>
---
#### [new 030] Generalizable Audio Spoofing Detection using Non-Semantic Representations
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出基于非语义音频表示的通用语音欺骗检测方法，解决现有技术泛化能力差的问题。通过TRILL/TRILLsson模型提取非语义特征，在域内和域外数据集上验证，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.00186v1](http://arxiv.org/pdf/2509.00186v1)**

> **作者:** Arnab Das; Yassine El Kheir; Carlos Franzreb; Tim Herzig; Tim Polzehl; Sebastian Möller
>
> **摘要:** Rapid advancements in generative modeling have made synthetic audio generation easy, making speech-based services vulnerable to spoofing attacks. Consequently, there is a dire need for robust countermeasures more than ever. Existing solutions for deepfake detection are often criticized for lacking generalizability and fail drastically when applied to real-world data. This study proposes a novel method for generalizable spoofing detection leveraging non-semantic universal audio representations. Extensive experiments have been performed to find suitable non-semantic features using TRILL and TRILLsson models. The results indicate that the proposed method achieves comparable performance on the in-domain test set while significantly outperforming state-of-the-art approaches on out-of-domain test sets. Notably, it demonstrates superior generalization on public-domain data, surpassing methods based on hand-crafted features, semantic embeddings, and end-to-end architectures.
>
---
#### [new 031] Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出基于深度学习的《古兰经》发音错误检测与纠正系统，解决数据稀缺问题，构建自动化数据集并设计QPS编码及多级CTC模型，实现0.16%低错误率。**

- **链接: [http://arxiv.org/pdf/2509.00094v1](http://arxiv.org/pdf/2509.00094v1)**

> **作者:** Abdullah Abdelfattah; Mahmoud I. Khalil; Hazem Abbas
>
> **摘要:** Assessing spoken language is challenging, and quantifying pronunciation metrics for machine learning models is even harder. However, for the Holy Quran, this task is simplified by the rigorous recitation rules (tajweed) established by Muslim scholars, enabling highly effective assessment. Despite this advantage, the scarcity of high-quality annotated data remains a significant barrier. In this work, we bridge these gaps by introducing: (1) A 98% automated pipeline to produce high-quality Quranic datasets -- encompassing: Collection of recitations from expert reciters, Segmentation at pause points (waqf) using our fine-tuned wav2vec2-BERT model, Transcription of segments, Transcript verification via our novel Tasmeea algorithm; (2) 850+ hours of audio (~300K annotated utterances); (3) A novel ASR-based approach for pronunciation error detection, utilizing our custom Quran Phonetic Script (QPS) to encode Tajweed rules (unlike the IPA standard for Modern Standard Arabic). QPS uses a two-level script: (Phoneme level): Encodes Arabic letters with short/long vowels. (Sifa level): Encodes articulation characteristics of every phoneme. We further include comprehensive modeling with our novel multi-level CTC Model which achieved 0.16% average Phoneme Error Rate (PER) on the testset. We release all code, data, and models as open-source: https://obadx.github.io/prepare-quran-dataset/
>
---
#### [new 032] Deep Learning for Personalized Binaural Audio Reproduction
- **分类: eess.AS; cs.SD**

- **简介: 该论文综述深度学习在个性化双耳音频再现任务中的应用，解决空间定位与沉浸式音频生成问题，提出显式滤波与端到端两类方法，总结数据集与评估指标，并探讨技术挑战与应用方向。**

- **链接: [http://arxiv.org/pdf/2509.00400v1](http://arxiv.org/pdf/2509.00400v1)**

> **作者:** Xikun Lu; Yunda Chen; Zehua Chen; Jie Wang; Mingxing Liu; Hongmei Hu; Chengshi Zheng; Stefan Bleeck; Jinqiu Sang
>
> **摘要:** Personalized binaural audio reproduction is the basis of realistic spatial localization, sound externalization, and immersive listening, directly shaping user experience and listening effort. This survey reviews recent advances in deep learning for this task and organizes them by generation mechanism into two paradigms: explicit personalized filtering and end-to-end rendering. Explicit methods predict personalized head-related transfer functions (HRTFs) from sparse measurements, morphological features, or environmental cues, and then use them in the conventional rendering pipeline. End-to-end methods map source signals directly to binaural signals, aided by other inputs such as visual, textual, or parametric guidance, and they learn personalization within the model. We also summarize the field's main datasets and evaluation metrics to support fair and repeatable comparison. Finally, we conclude with a discussion of key applications enabled by these technologies, current technical limitations, and potential research directions for deep learning-based spatial audio systems.
>
---
#### [new 033] Quantum-Enhanced Analysis and Grading of Vocal Performance
- **分类: eess.AS; cs.SD; H.5.5; I.2.6; I.5.4**

- **简介: 论文提出QuantumMelody，结合量子-经典方法进行客观歌唱评分，利用量子电路处理声乐特征并融合频谱图嵌入，实现74.29%专家一致。**

- **链接: [http://arxiv.org/pdf/2509.00106v1](http://arxiv.org/pdf/2509.00106v1)**

> **作者:** Rohan Agarwal
>
> **备注:** 4 pages, 5 figures. Hybrid quantum - classical feasibility study; simulator - only results
>
> **摘要:** We present QuantumMelody, a hybrid quantum-classical method for objective singing assessment. Grouped vocal features (pitch stability, dynamics, timbre) are encoded into a small simulated quantum circuit; all nine qubits are initialized with a Hadamard on each qubit and then receive Rx, Ry, and Rz rotations, with intra- and cross-group entanglement. The circuit measurement probabilities are fused with spectrogram transformer embeddings to estimate a grade on labels 2-5 and to surface technique-level feedback. On 168 labeled 20 second excerpts, the hybrid reaches 74.29% agreement with expert graders, a +12.86 point gain over a classical-features baseline. Processing is sub-minute per recording on a laptop-class Qiskit simulator; we do not claim hardware speedups. This is a feasibility step toward interpretable, objective singing assessment in applied audio signal processing.
>
---
#### [new 034] NADI 2025: The First Multidialectal Arabic Speech Processing Shared Task
- **分类: cs.CL; cs.SD**

- **简介: 论文介绍NADI 2025共享任务，聚焦阿拉伯语方言语音处理的三个子任务：方言识别、语音识别与重音恢复。评估了44支团队的100份提交，分析挑战并总结方法，为未来研究提供方向。**

- **链接: [http://arxiv.org/pdf/2509.02038v1](http://arxiv.org/pdf/2509.02038v1)**

> **作者:** Bashar Talafha; Hawau Olamide Toyin; Peter Sullivan; AbdelRahim Elmadany; Abdurrahman Juma; Amirbek Djanibekov; Chiyu Zhang; Hamad Alshehhi; Hanan Aldarmaki; Mustafa Jarrar; Nizar Habash; Muhammad Abdul-Mageed
>
> **摘要:** We present the findings of the sixth Nuanced Arabic Dialect Identification (NADI 2025) Shared Task, which focused on Arabic speech dialect processing across three subtasks: spoken dialect identification (Subtask 1), speech recognition (Subtask 2), and diacritic restoration for spoken dialects (Subtask 3). A total of 44 teams registered, and during the testing phase, 100 valid submissions were received from eight unique teams. The distribution was as follows: 34 submissions for Subtask 1 "five teams{\ae}, 47 submissions for Subtask 2 "six teams", and 19 submissions for Subtask 3 "two teams". The best-performing systems achieved 79.8% accuracy on Subtask 1, 35.68/12.20 WER/CER (overall average) on Subtask 2, and 55/13 WER/CER on Subtask 3. These results highlight the ongoing challenges of Arabic dialect speech processing, particularly in dialect identification, recognition, and diacritic restoration. We also summarize the methods adopted by participating teams and briefly outline directions for future editions of NADI.
>
---
#### [new 035] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 论文提出AHAMask方法，通过屏蔽解码器注意力头，无需指令即可可靠指定LALM音频任务，实验表明其性能与指令相当甚至更优，揭示模型存在功能路径。**

- **链接: [http://arxiv.org/pdf/2509.01787v1](http://arxiv.org/pdf/2509.01787v1)**

> **作者:** Yiwei Guo; Bohan Li; Hankun Wang; Zhihan Li; Shuai Wang; Xie Chen; Kai Yu
>
> **备注:** 15 pages, 7 tables, 6 figures
>
> **摘要:** Although current large audio language models (LALMs) extend text large language models (LLMs) with generic acoustic understanding abilities, they usually suffer from instruction sensitivity, where different instructions of the same intention can yield drastically different outcomes. In this work, we propose AHAMask, where we simply mask some of the attention heads in the decoder-only LLM backbone of LALMs, to trigger specific acoustic task functionalities without instructions. These masks are efficiently obtained by training on an LALM, with the number of trainable parameters equal to the attention head count in its LLM backbone. We show by experiments that applying such selective attention head masks achieves comparable or even better performance than using instructions, either on single or composite tasks. Besides achieving reliable acoustic task specification for LALMs, this also reveals that LALMs exhibit certain "functional pathways" in their attention heads.
>
---
#### [new 036] ChipChat: Low-Latency Cascaded Conversational Agent in MLX
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出ChipChat，解决实时语音对话系统中级联架构的延迟问题。通过架构创新与流式优化，集成语音识别、LLM、语音合成等模块，实现无GPU设备的亚秒级响应，保障隐私。**

- **链接: [http://arxiv.org/pdf/2509.00078v1](http://arxiv.org/pdf/2509.00078v1)**

> **作者:** Tatiana Likhomanenko; Luke Carlson; Richard He Bai; Zijin Gu; Han Tran; Zakaria Aldeneh; Yizhe Zhang; Ruixiang Zhang; Huangjie Zheng; Navdeep Jaitly
>
> **备注:** ASRU 2025
>
> **摘要:** The emergence of large language models (LLMs) has transformed spoken dialog systems, yet the optimal architecture for real-time on-device voice agents remains an open question. While end-to-end approaches promise theoretical advantages, cascaded systems (CSs) continue to outperform them in language understanding tasks, despite being constrained by sequential processing latency. In this work, we introduce ChipChat, a novel low-latency CS that overcomes traditional bottlenecks through architectural innovations and streaming optimizations. Our system integrates streaming (a) conversational speech recognition with mixture-of-experts, (b) state-action augmented LLM, (c) text-to-speech synthesis, (d) neural vocoder, and (e) speaker modeling. Implemented using MLX, ChipChat achieves sub-second response latency on a Mac Studio without dedicated GPUs, while preserving user privacy through complete on-device processing. Our work shows that strategically redesigned CSs can overcome their historical latency limitations, offering a promising path forward for practical voice-based AI agents.
>
---
#### [new 037] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文提出SimulMEGA框架，解决同时语音翻译中质量、延迟与语义连贯性的平衡问题。通过MoE路由策略与前缀训练，隐式学习读写决策，无需额外推理开销，提升多语言流式任务的延迟-质量 trade-off。**

- **链接: [http://arxiv.org/pdf/2509.01200v1](http://arxiv.org/pdf/2509.01200v1)**

> **作者:** Chenyang Le; Bing Han; Jinshun Li; Songyong Chen; Yanmin Qian
>
> **摘要:** Simultaneous Speech Translation (SimulST) enables real-time cross-lingual communication by jointly optimizing speech recognition and machine translation under strict latency constraints. Existing systems struggle to balance translation quality, latency, and semantic coherence, particularly in multilingual many-to-many scenarios where divergent read and write policies hinder unified strategy learning. In this paper, we present SimulMEGA (Simultaneous Generation by Mixture-of-Experts Gating), an unsupervised policy learning framework that combines prefix-based training with a Mixture-of-Experts refiner to learn effective read and write decisions in an implicit manner, without adding inference-time overhead. Our design requires only minimal modifications to standard transformer architectures and generalizes across both speech-to-text and text-to-speech streaming tasks. Through comprehensive evaluation on six language pairs, our 500M parameter speech-to-text model outperforms the Seamless baseline, achieving under 7 percent BLEU degradation at 1.5 seconds average lag and under 3 percent at 3 seconds. We further demonstrate the versatility of SimulMEGA by extending it to streaming TTS with a unidirectional backbone, yielding superior latency quality tradeoffs.
>
---
#### [new 038] Noisy Disentanglement with Tri-stage Training for Noise-Robust Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对噪声环境下的语音识别任务，提出NoisyD-CT框架，通过三阶段训练策略和噪声分离模块提升模型鲁棒性，结合一致性损失与重建损失优化特征表示，实验证明在模拟和真实噪声数据上显著降低误识率。**

- **链接: [http://arxiv.org/pdf/2509.01087v1](http://arxiv.org/pdf/2509.01087v1)**

> **作者:** Shuangyuan Chen; Shuang Wei; Dongxing Xu; Yanhua Long
>
> **备注:** 11 pages,4 figures
>
> **摘要:** To enhance the performance of end-to-end (E2E) speech recognition systems in noisy or low signal-to-noise ratio (SNR) conditions, this paper introduces NoisyD-CT, a novel tri-stage training framework built on the Conformer-Transducer architecture. The core of NoisyD-CT is a especially designed compact noisy disentanglement (NoisyD) module (adding only 1.71M parameters), integrated between the Conformer blocks and Transducer Decoder to perform deep noise suppression and improve ASR robustness in challenging acoustic noise environments. To fully exploit the noise suppression capability of the NoisyD-CT, we further propose a clean representation consistency loss to align high-level representations derived from noisy speech with those obtained from corresponding clean speech. Together with a noisy reconstruction loss, this consistency alignment enables the NoisyD module to effectively suppress noise while preserving essential acoustic and linguistic features consistent across both clean and noisy conditions, thereby producing cleaner internal representations that enhance ASR performance. Moreover, our tri-stage training strategy is designed to fully leverage the functionalities of both the noisy disentanglement and speech recognition modules throughout the model training process, ultimately maximizing performance gains under noisy conditions. Our experiments are performed on the LibriSpeech and CHiME-4 datasets, extensive results demonstrate that our proposed NoisyD-CT significantly outperforms the competitive Conformer-Transducer baseline, achieving up to 25.7% and 10.6% relative word error rate reductions on simulated and real-world noisy test sets, respectively, while maintaining or even improving performance on clean speech test sets. The source code, model checkpoint and data simulation scripts will be available at https://github.com/litchimo/NoisyD-CT.
>
---
#### [new 039] Prospects for acoustically monitoring ecosystem tipping points
- **分类: q-bio.PE; cs.SD; math.DS**

- **简介: 该论文提出利用声学监测技术探测生态系统临界点，通过分析声景特征识别早期预警信号，解决传统方法对生态韧性评估的局限，推动高通量技术在生态预警中的应用。**

- **链接: [http://arxiv.org/pdf/2509.02201v1](http://arxiv.org/pdf/2509.02201v1)**

> **作者:** Neel P. Le Penru; Thomas M. Bury; Sarab S. Sethi; Robert M. Ewers; Lorenzo Picinali
>
> **备注:** 44 pages (including Supporting Information), 1 figure. Review article submitted to Global Change Biology
>
> **摘要:** Many ecosystems can undergo important qualitative changes, including sudden transitions to alternative stable states, in response to perturbations or increments in conditions. Such 'tipping points' are often preceded by declines in aspects of ecosystem resilience, namely the capacity to recover from perturbations, that leave various spatial and temporal signatures. These so-called 'early warning signals' have been used to anticipate transitions in diverse real systems, but many of the high-throughput, autonomous monitoring technologies that are transforming ecology have yet to be fully leveraged to this end. Acoustic monitoring in particular is a powerful tool for quantifying biodiversity, tracking ecosystem health, and facilitating conservation. By deploying acoustic recorders in diverse environments, researchers have gained insights from the calls and behaviour of individual species to higher-level soundscape features that describe habitat quality and even predict species occurrence. Here, we draw on theory and practice to advocate for using acoustics to probe ecosystem resilience and identify emerging and established early warning signals of tipping points. With a focus on pragmatic considerations, we emphasise that despite limits to tipping point theory and the current scale and transferability of data, acoustics could be instrumental in understanding resilience and tipping potential across distinct ecosystems and scales.
>
---
#### [new 040] DynaMind: Reconstructing Dynamic Visual Scenes from EEG by Aligning Temporal Dynamics and Multimodal Semantics to Guided Diffusion
- **分类: cs.CV; cs.AI; cs.HC; eess.SP**

- **简介: 论文提出DynaMind框架，通过融合神经动态与语义信息，解决EEG重建动态视觉场景的时空不匹配与语义不足问题，实现高保真视频重建。**

- **链接: [http://arxiv.org/pdf/2509.01177v1](http://arxiv.org/pdf/2509.01177v1)**

> **作者:** Junxiang Liu; Junming Lin; Jiangtong Li; Jie Li
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Reconstruction dynamic visual scenes from electroencephalography (EEG) signals remains a primary challenge in brain decoding, limited by the low spatial resolution of EEG, a temporal mismatch between neural recordings and video dynamics, and the insufficient use of semantic information within brain activity. Therefore, existing methods often inadequately resolve both the dynamic coherence and the complex semantic context of the perceived visual stimuli. To overcome these limitations, we introduce DynaMind, a novel framework that reconstructs video by jointly modeling neural dynamics and semantic features via three core modules: a Regional-aware Semantic Mapper (RSM), a Temporal-aware Dynamic Aligner (TDA), and a Dual-Guidance Video Reconstructor (DGVR). The RSM first utilizes a regional-aware encoder to extract multimodal semantic features from EEG signals across distinct brain regions, aggregating them into a unified diffusion prior. In the mean time, the TDA generates a dynamic latent sequence, or blueprint, to enforce temporal consistency between the feature representations and the original neural recordings. Together, guided by the semantic diffusion prior, the DGVR translates the temporal-aware blueprint into a high-fidelity video reconstruction. On the SEED-DV dataset, DynaMind sets a new state-of-the-art (SOTA), boosting reconstructed video accuracies (video- and frame-based) by 12.5 and 10.3 percentage points, respectively. It also achieves a leap in pixel-level quality, showing exceptional visual fidelity and temporal coherence with a 9.4% SSIM improvement and a 19.7% FVMD reduction. This marks a critical advancement, bridging the gap between neural dynamics and high-fidelity visual semantics.
>
---
#### [new 041] MPO: Multidimensional Preference Optimization for Language Model-based Text-to-Speech
- **分类: eess.AS; cs.SD**

- **简介: 论文提出MPO框架，针对语言模型驱动的TTS系统在多维偏好优化中的挑战，通过引入偏好集和正则化训练，提升可懂度、说话人相似性和语调。**

- **链接: [http://arxiv.org/pdf/2509.00685v1](http://arxiv.org/pdf/2509.00685v1)**

> **作者:** Kangxiang Xia; Xinfa Zhu; Jixun Yao; Lei Xie
>
> **备注:** Accepted by NCMMSC2025
>
> **摘要:** In recent years, text-to-speech (TTS) has seen impressive advancements through large-scale language models, achieving human-level speech quality. Integrating human feedback has proven effective for enhancing robustness in these systems. However, current approaches face challenges in optimizing TTS with preference data across multiple dimensions and often suffer from performance degradation due to overconfidence in rewards. We propose Multidimensional Preference Optimization (MPO) to better align TTS systems with human preferences. MPO introduces a preference set that streamlines the construction of data for multidimensional preference optimization, enabling alignment with multiple dimensions. Additionally, we incorporate regularization during training to address the typical degradation issues in DPO-based approaches. Our experiments demonstrate MPO's effectiveness, showing significant improvements in intelligibility, speaker similarity, and prosody compared to baseline systems.
>
---
#### [new 042] IoT-based Noise Monitoring using Mobile Nodes for Smart Cities
- **分类: cs.ET; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出基于IoT的移动节点实时噪声监测系统，解决传统监测覆盖不足问题，通过移动校准与随机森林回归提升精度，实测验证其在智慧城市中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.00979v1](http://arxiv.org/pdf/2509.00979v1)**

> **作者:** Bhima Sankar Manthina; Shreyash Gujar; Sachin Chaudhari; Kavita Vemuri1; Shivam Chhirolya
>
> **摘要:** Urban noise pollution poses a significant threat to public health, yet existing monitoring infrastructures offer limited spatial coverage and adaptability. This paper presents a scalable, low-cost, IoT-based, real-time environmental noise monitoring solution using mobile nodes (sensor nodes on a moving vehicle). The system utilizes a low-cost sound sensor integrated with GPS-enabled modules to collect geotagged noise data at one-second intervals. The sound nodes are calibrated against a reference sound level meter in a laboratory setting to ensure accuracy using various machine learning (ML) algorithms, such as Simple Linear Regression (SLR), Multiple Linear Regression (MLR), Polynomial Regression (PR), Segmented Regression (SR), Support Vector Regression (SVR), Decision Tree (DT), and Random Forest Regression (RFR). While laboratory calibration demonstrates high accuracy, it is shown that the performance of the nodes degrades during data collection in a moving vehicle. To address this, it is demonstrated that the calibration must be performed on the IoT-based node based on the data collected in a moving environment along with the reference device. Among the employed ML models, RFR achieved the best performance with an R2 of 0.937 and RMSE of 1.09 for mobile calibration. The system was deployed in Hyderabad, India, through three measurement campaigns across 27 days, capturing 436,420 data points. Results highlight temporal and spatial noise variations across weekdays, weekends, and during Diwali. Incorporating vehicular velocity into the calibration significantly improves accuracy. The proposed system demonstrates the potential for widespread deployment of IoT-based noise sensing networks in smart cities, enabling effective noise pollution management and urban planning.
>
---
#### [new 043] Amplifying Emotional Signals: Data-Efficient Deep Learning for Robust Speech Emotion Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对语音情感识别（SER）任务，解决小数据集下的性能瓶颈，通过开发SVM、LSTM、CNN模型并结合迁移学习与数据增强，提升ResNet34模型性能。**

- **链接: [http://arxiv.org/pdf/2509.00077v1](http://arxiv.org/pdf/2509.00077v1)**

> **作者:** Tai Vu
>
> **摘要:** Speech Emotion Recognition (SER) presents a significant yet persistent challenge in human-computer interaction. While deep learning has advanced spoken language processing, achieving high performance on limited datasets remains a critical hurdle. This paper confronts this issue by developing and evaluating a suite of machine learning models, including Support Vector Machines (SVMs), Long Short-Term Memory networks (LSTMs), and Convolutional Neural Networks (CNNs), for automated emotion classification in human speech. We demonstrate that by strategically employing transfer learning and innovative data augmentation techniques, our models can achieve impressive performance despite the constraints of a relatively small dataset. Our most effective model, a ResNet34 architecture, establishes a new performance benchmark on the combined RAVDESS and SAVEE datasets, attaining an accuracy of 66.7% and an F1 score of 0.631. These results underscore the substantial benefits of leveraging pre-trained models and data augmentation to overcome data scarcity, thereby paving the way for more robust and generalizable SER systems.
>
---
#### [new 044] Real-Time Piano Note Frequency Detection Using FPGA and FFT Core
- **分类: cs.AR; cs.SD; eess.AS**

- **简介: 该论文旨在解决传统软件DSP在实时钢琴音符频率检测中的延迟与计算资源消耗问题。通过设计基于FPGA的FFT硬件系统，实现对数字钢琴模拟信号的实时频率分析，提升处理速度与确定性。**

- **链接: [http://arxiv.org/pdf/2509.00589v1](http://arxiv.org/pdf/2509.00589v1)**

> **作者:** Shafayet M. Anik; D. G. Perera
>
> **备注:** 20 pages, 11 Figures
>
> **摘要:** Real-time frequency analysis of musical instruments, such as the piano, is an essential feature in areas like electronic tuners, music visualizers, and live sound monitoring. Traditional methods often rely on software-based digital signal processing (DSP), which may introduce latency and require significant computational power. In contrast, hardware platforms such as FPGAs (Field Programmable Gate Arrays) offer the ability to perform such analyses with greater speed and determinism due to their parallel processing capabilities. The primary objective of this project was to analyze analog audio signals from a digital piano using an FPGA-based real-time Fast Fourier Transform (FFT) system.
>
---
#### [new 045] Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出针对小语言的微型单语ASR模型，挑战多语言模型优势，通过混合高质量数据训练，在27M参数下实现优于多语言及更大模型的性能，推动边缘设备ASR发展。**

- **链接: [http://arxiv.org/pdf/2509.02523v1](http://arxiv.org/pdf/2509.02523v1)**

> **作者:** Evan King; Adam Sabra; Manjunath Kudlur; James Wang; Pete Warden
>
> **摘要:** We present the Flavors of Moonshine, a suite of tiny automatic speech recognition (ASR) models specialized for a range of underrepresented languages. Prevailing wisdom suggests that multilingual ASR models outperform monolingual counterparts by exploiting cross-lingual phonetic similarities. We challenge this assumption, showing that for sufficiently small models (27M parameters), training monolingual systems on a carefully balanced mix of high-quality human-labeled, pseudo-labeled, and synthetic data yields substantially superior performance. On average, our models achieve error rates 48% lower than the comparably sized Whisper Tiny model, outperform the 9x larger Whisper Small model, and in most cases match or outperform the 28x larger Whisper Medium model. These results advance the state of the art for models of this size, enabling accurate on-device ASR for languages that previously had limited support. We release Arabic, Chinese, Japanese, Korean, Ukrainian, and Vietnamese Moonshine models under a permissive open-source license.
>
---
## 更新

#### [replaced 001] Dynamic Fusion Multimodal Network for SpeechWellness Detection
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.18057v2](http://arxiv.org/pdf/2508.18057v2)**

> **作者:** Wenqiang Sun; Han Yin; Jisheng Bai; Jianfeng Chen
>
> **备注:** 6 pages, 5figures
>
> **摘要:** Suicide is one of the leading causes of death among adolescents. Previous suicide risk prediction studies have primarily focused on either textual or acoustic information in isolation, the integration of multimodal signals, such as speech and text, offers a more comprehensive understanding of an individual's mental state. Motivated by this, and in the context of the 1st SpeechWellness detection challenge, we explore a lightweight multi-branch multimodal system based on a dynamic fusion mechanism for speechwellness detection. To address the limitation of prior approaches that rely on time-domain waveforms for acoustic analysis, our system incorporates both time-domain and time-frequency (TF) domain acoustic features, as well as semantic representations. In addition, we introduce a dynamic fusion block to adaptively integrate information from different modalities. Specifically, it applies learnable weights to each modality during the fusion process, enabling the model to adjust the contribution of each modality. To enhance computational efficiency, we design a lightweight structure by simplifying the original baseline model. Experimental results demonstrate that the proposed system exhibits superior performance compared to the challenge baseline, achieving a 78% reduction in model parameters and a 5% improvement in accuracy.
>
---
#### [replaced 002] NeuroAMP: A Novel End-to-end General Purpose Deep Neural Amplifier for Personalized Hearing Aids
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.10822v2](http://arxiv.org/pdf/2502.10822v2)**

> **作者:** Shafique Ahmed; Ryandhimas E. Zezario; Hui-Guan Yuan; Amir Hussain; Hsin-Min Wang; Wei-Ho Chung; Yu Tsao
>
> **备注:** Accepted for publication in IEEE Transactions on Artificial Intelligence
>
> **摘要:** The prevalence of hearing aids is increasing. However, optimizing the amplification processes of hearing aids remains challenging due to the complexity of integrating multiple modular components in traditional methods. To address this challenge, we present NeuroAMP, a novel deep neural network designed for end-to-end, personalized amplification in hearing aids. NeuroAMP leverages both spectral features and the listener's audiogram as inputs, and we investigate four architectures: Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Convolutional Recurrent Neural Network (CRNN), and Transformer. We also introduce Denoising NeuroAMP, an extension that integrates noise reduction along with amplification capabilities for improved performance in real-world scenarios. To enhance generalization, a comprehensive data augmentation strategy was employed during training on diverse speech (TIMIT and TMHINT) and music (Cadenza Challenge MUSIC) datasets. Evaluation using the Hearing Aid Speech Perception Index (HASPI), Hearing Aid Speech Quality Index (HASQI), and Hearing Aid Audio Quality Index (HAAQI) demonstrates that the Transformer architecture within NeuroAMP achieves the best performance, with SRCC scores of 0.9927 (HASQI) and 0.9905 (HASPI) on TIMIT, and 0.9738 (HAAQI) on the Cadenza Challenge MUSIC dataset. Notably, our data augmentation strategy maintains high performance on unseen datasets (e.g., VCTK, MUSDB18-HQ). Furthermore, Denoising NeuroAMP outperforms both the conventional NAL-R+WDRC approach and a two-stage baseline on the VoiceBank+DEMAND dataset, achieving a 10% improvement in both HASPI (0.90) and HASQI (0.59) scores. These results highlight the potential of NeuroAMP and Denoising NeuroAMP to deliver notable improvements in personalized hearing aid amplification.
>
---
#### [replaced 003] Multi-stream Convolutional Neural Network with Frequency Selection for Robust Speaker Verification
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2012.11159v3](http://arxiv.org/pdf/2012.11159v3)**

> **作者:** Wei Yao; Shen Chen; Jiamin Cui; Yaolin Lou
>
> **备注:** 12 pages, 11 figures, 8 tables
>
> **摘要:** Speaker verification aims to verify whether an input speech corresponds to the claimed speaker, and conventionally, this kind of system is deployed based on single-stream scenario, wherein the feature extractor operates in full frequency range. In this paper, we hypothesize that machine can learn enough knowledge to do classification task when listening to partial frequency range instead of full frequency range, which is so called frequency selection technique, and further propose a novel framework of multi-stream Convolutional Neural Network (CNN) with this technique for speaker verification tasks. The proposed framework accommodates diverse temporal embeddings generated from multiple streams to enhance the robustness of acoustic modeling. For the diversity of temporal embeddings, we consider feature augmentation with frequency selection, which is to manually segment the full-band of frequency into several sub-bands, and the feature extractor of each stream can select which sub-bands to use as target frequency domain. Different from conventional single-stream solution wherein each utterance would only be processed for one time, in this framework, there are multiple streams processing it in parallel. The input utterance for each stream is pre-processed by a frequency selector within specified frequency range, and post-processed by mean normalization. The normalized temporal embeddings of each stream will flow into a pooling layer to generate fused embeddings. We conduct extensive experiments on VoxCeleb dataset, and the experimental results demonstrate that multi-stream CNN significantly outperforms single-stream baseline with 20.53 % of relative improvement in minimum Decision Cost Function (minDCF).
>
---
#### [replaced 004] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.14534v4](http://arxiv.org/pdf/2507.14534v4)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [replaced 005] Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.22646v2](http://arxiv.org/pdf/2506.22646v2)**

> **作者:** Weiqing Wang; Taejin Park; Ivan Medennikov; Jinhan Wang; Kunal Dhawan; He Huang; Nithin Rao Koluguri; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** We propose a self-speaker adaptation method for streaming multi-talker automatic speech recognition (ASR) that eliminates the need for explicit speaker queries. Unlike conventional approaches requiring target speaker embeddings or enrollment audio, our technique dynamically adapts individual ASR instances through speaker-wise speech activity prediction. The key innovation involves injecting speaker-specific kernels generated via speaker supervision activations into selected ASR encoder layers. This enables instantaneous speaker adaptation to target speakers while handling fully overlapped speech even in a streaming scenario. Experiments show state-of-the-art performance in both offline and streaming scenarios, demonstrating that our self-adaptive method effectively addresses severe speech overlap through streamlined speaker-focused recognition. The results validate the proposed self-speaker adaptation approach as a robust solution for multi-talker ASR under severe overlapping speech conditions.
>
---
#### [replaced 006] SyncNet: correlating objective for time delay estimation in audio signals
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2203.14639v3](http://arxiv.org/pdf/2203.14639v3)**

> **作者:** Akshay Raina; Vipul Arora
>
> **备注:** Accepted to IEEE-ICASSP 2023
>
> **摘要:** This study addresses the task of performing robust and reliable time-delay estimation in signals in noisy and reverberating environments. In contrast to the popular signal processing based methods, this paper proposes to transform the input signals using a deep neural network into another pair of sequences which show high cross correlation at the actual time delay. This is achieved with the help of a novel correlation function based objective function for training the network. The proposed approach is also intrinsically interpretable as it does not lose temporal information. Experimental evaluations are performed for estimating mutual time delays for different types of audio signals such as pulse, speech and musical beats. SyncNet outperforms other classical approaches, such as GCC-PHAT, and some other learning based approaches.
>
---
#### [replaced 007] A Neural Speech Codec for Noise Robust Speech Coding
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.04132v2](http://arxiv.org/pdf/2309.04132v2)**

> **作者:** Jiayi Huang; Zeyu Yan; Wenbin Jiang; He Wang; Fei Wen
>
> **摘要:** This paper considers the joint compression and enhancement problem for speech signal in the presence of noise. Recently, the SoundStream codec, which relies on end-to-end joint training of an encoder-decoder pair and a residual vector quantizer by a combination of adversarial and reconstruction losses,has shown very promising performance, especially in subjective perception quality. In this work, we provide a theoretical result to show that, to simultaneously achieve low distortion and high perception in the presence of noise, there exist an optimal two-stage optimization procedure for the joint compression and enhancement problem. This procedure firstly optimizes an encoder-decoder pair using only distortion loss and then fixes the encoder to optimize a perceptual decoder using perception loss. Based on this result, we construct a two-stage training framework for joint compression and enhancement of noisy speech signal. Unlike existing training methods which are heuristic, the proposed two-stage training method has a theoretical foundation. Finally, experimental results for various noise and bit-rate conditions are provided. The results demonstrate that a codec trained by the proposed framework can outperform SoundStream and other representative codecs in terms of both objective and subjective evaluation metrics. Code is available at \textit{https://github.com/jscscloris/SEStream}.
>
---
#### [replaced 008] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.10082v4](http://arxiv.org/pdf/2507.10082v4)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [replaced 009] I2TTS: Image-indicated Immersive Text-to-speech Synthesis with Spatial Perception
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.13314v3](http://arxiv.org/pdf/2411.13314v3)**

> **作者:** Jiawei Zhang; Tian-Hao Zhang; Jun Wang; Jiaran Gao; Xinyuan Qian; Xu-Cheng Yin
>
> **备注:** Accepted by APSIPA ASC2025
>
> **摘要:** Controlling the style and characteristics of speech synthesis is crucial for adapting the output to specific contexts and user requirements. Previous Text-to-speech (TTS) works have focused primarily on the technical aspects of producing natural-sounding speech, such as intonation, rhythm, and clarity. However, they overlook the fact that there is a growing emphasis on spatial perception of synthesized speech, which may provide immersive experience in gaming and virtual reality. To solve this issue, in this paper, we present a novel multi-modal TTS approach, namely Image-indicated Immersive Text-to-speech Synthesis (I2TTS). Specifically, we introduce a scene prompt encoder that integrates visual scene prompts directly into the synthesis pipeline to control the speech generation process. Additionally, we propose a reverberation classification and refinement technique that adjusts the synthesized mel-spectrogram to enhance the immersive experience, ensuring that the involved reverberation condition matches the scene accurately. Experimental results demonstrate that our model achieves high-quality scene and spatial matching without compromising speech naturalness, marking a significant advancement in the field of context-aware speech synthesis. Project demo page: https://spatialTTS.github.io/ Index Terms-Speech synthesis, scene prompt, spatial perception
>
---
#### [replaced 010] SyncGuard: Robust Audio Watermarking Capable of Countering Desynchronization Attacks
- **分类: cs.CR; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.17121v2](http://arxiv.org/pdf/2508.17121v2)**

> **作者:** Zhenliang Gan; Xiaoxiao Hu; Sheng Li; Zhenxing Qian; Xinpeng Zhang
>
> **备注:** Accepted at ECAI 2025
>
> **摘要:** Audio watermarking has been widely applied in copyright protection and source tracing. However, due to the inherent characteristics of audio signals, watermark localization and resistance to desynchronization attacks remain significant challenges. In this paper, we propose a learning-based scheme named SyncGuard to address these challenges. Specifically, we design a frame-wise broadcast embedding strategy to embed the watermark in arbitrary-length audio, enhancing time-independence and eliminating the need for localization during watermark extraction. To further enhance robustness, we introduce a meticulously designed distortion layer. Additionally, we employ dilated residual blocks in conjunction with dilated gated blocks to effectively capture multi-resolution time-frequency features. Extensive experimental results show that SyncGuard efficiently handles variable-length audio segments, outperforms state-of-the-art methods in robustness against various attacks, and delivers superior auditory quality.
>
---
