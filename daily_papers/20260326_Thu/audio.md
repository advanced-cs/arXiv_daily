# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] ArrayDPS-Refine: Generative Refinement of Discriminative Multi-Channel Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于多通道语音增强任务，旨在解决传统方法在复杂噪声下产生的非线性失真问题。提出ArrayDPS-Refine，利用纯净语音扩散先验对判别模型输出进行生成式优化。**

- **链接: [https://arxiv.org/pdf/2603.24385](https://arxiv.org/pdf/2603.24385)**

> **作者:** Zhongweiyang Xu; Ashutosh Pandey; Juan Azcarreta; Zhaoheng Ni; Sanjeel Parekh; Buye Xu
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Multi-channel speech enhancement aims to recover clean speech from noisy multi-channel recordings. Most deep learning methods employ discriminative training, which can lead to non-linear distortions from regression-based objectives, especially under challenging environmental noise conditions. Inspired by ArrayDPS for unsupervised multi-channel source separation, we introduce ArrayDPS-Refine, a method designed to enhance the outputs of discriminative models using a clean speech diffusion prior. ArrayDPS-Refine is training-free, generative, and array-agnostic. It first estimates the noise spatial covariance matrix (SCM) from the enhanced speech produced by a discriminative model, then uses this estimated noise SCM for diffusion posterior sampling. This approach allows direct refinement of any discriminative model's output without retraining. Our results show that ArrayDPS-Refine consistently improves the performance of various discriminative models, including state-of-the-art waveform and STFT domain models. Audio demos are provided at this https URL.
>
---
#### [new 002] What and When to Learn: CURriculum Ranking Loss for Large-Scale Speaker Verification
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音识别中的说话人验证任务，旨在解决大规模数据中样本质量不一导致的模型性能下降问题。提出Curry损失函数，通过在线评估样本难度，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2603.24432](https://arxiv.org/pdf/2603.24432)**

> **作者:** Massa Baali; Sarthak Bisht; Rita Singh; Bhiksha Raj
>
> **摘要:** Speaker verification at large scale remains an open challenge as fixed-margin losses treat all samples equally regardless of quality. We hypothesize that mislabeled or degraded samples introduce noisy gradients that disrupt compact speaker manifolds. We propose Curry (CURriculum Ranking), an adaptive loss that estimates sample difficulty online via Sub-center ArcFace: confidence scores from dominant sub-center cosine similarity rank samples into easy, medium, and hard tiers using running batch statistics, without auxiliary annotations. Learnable weights guide the model from stable identity foundations through manifold refinement to boundary sharpening. To our knowledge, this is the largest-scale speaker verification system trained to date. Evaluated on VoxCeleb1-O, and SITW, Curry reduces EER by 86.8\% and 60.0\% over the Sub-center ArcFace baseline, establishing a new paradigm for robust speaker verification on imperfect large-scale data.
>
---
#### [new 003] Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统中的中断检测任务，旨在解决现有方法在响应速度与鲁棒性间的矛盾。提出SID-Bench基准和APT指标，并设计基于大模型的检测方法，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2603.24144](https://arxiv.org/pdf/2603.24144)**

> **作者:** Kangxiang Xia; Bingshen Mu; Xian Shi; Jin Xu; Lei Xie
>
> **备注:** Accepted by ICME 2026
>
> **摘要:** Achieving natural full-duplex interaction in spoken dialogue systems (SDS) remains a challenge due to the difficulty of accurately detecting user interruptions. Current solutions are polarized between "trigger-happy" VAD-based methods that misinterpret backchannels and robust end-to-end models that exhibit unacceptable response delays. Moreover, the absence of real-world benchmarks and holistic metrics hinders progress in the field. This paper presents a comprehensive frame-work to overcome these limitations. We first introduce SID-Bench, the first benchmark for semantic-aware interruption detection built entirely from real-world human dialogues. To provide a rigorous assessment of the responsiveness-robustness trade-off, we propose the Average Penalty Time (APT) metric, which assigns a temporal cost to both false alarms and late responses. Building on this framework, we design an LLM-based detection model optimized through a novel training paradigm to capture subtle semantic cues of intent. Experimental results show that our model significantly outperforms mainstream baselines, achieving a nearly threefold reduction in APT. By successfully resolving the long-standing tension between speed and stability, our work establishes a new state-of-the-art for intelligent interruption handling in SDS. To facilitate future research, SID-Bench and the associated code are available at: this https URL.
>
---
#### [new 004] Autoregressive Guidance of Deep Spatially Selective Filters using Bayesian Tracking for Efficient Extraction of Moving Speakers
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音增强任务，解决动态场景下说话人定位与跟踪问题。通过融合增强信号与轻量跟踪算法，提升跟踪精度，实现高效语音增强。**

- **链接: [https://arxiv.org/pdf/2603.23723](https://arxiv.org/pdf/2603.23723)**

> **作者:** Jakob Kienegger; Timo Gerkmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Deep spatially selective filters achieve high-quality enhancement with real-time capable architectures for stationary speakers of known directions. To retain this level of performance in dynamic scenarios when only the speakers' initial directions are given, accurate, yet computationally lightweight tracking algorithms become necessary. Assuming a frame-wise causal processing style, temporal feedback allows for leveraging the enhanced speech signal to improve tracking performance. In this work, we investigate strategies to incorporate the enhanced signal into lightweight tracking algorithms and autoregressively guide deep spatial filters. Our proposed Bayesian tracking algorithms are compatible with arbitrary deep spatial filters. To increase the realism of simulated trajectories during development and evaluation, we propose and publish a novel dataset based on the social force model. Results validate that the autoregressive incorporation significantly improves the accuracy of our Bayesian trackers, resulting in superior enhancement with none or only negligibly increased computational overhead. Real-world recordings complement these findings and demonstrate the generalizability of our methods to unseen, challenging acoustic conditions.
>
---
#### [new 005] Rethinking Masking Strategies for Masked Prediction-based Audio Self-supervised Learning
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文属于音频表示学习任务，旨在解决自监督学习中的掩码策略问题。提出了一种轻量级的频谱稀疏性掩码方法，提升模型性能并降低计算复杂度。**

- **链接: [https://arxiv.org/pdf/2603.23810](https://arxiv.org/pdf/2603.23810)**

> **作者:** Daisuke Niizumi; Daiki Takeuchi; Masahiro Yasuda; Binh Thien Nguyen; Noboru Harada; Nobutaka Ono
>
> **备注:** 6+1 pages, 2 figures, 3 tables, accepted at IJCNN 2026
>
> **摘要:** Since the introduction of Masked Autoencoders, various improvements to masking techniques have been explored. In this paper, we rethink masking strategies for audio representation learning using masked prediction-based self-supervised learning (SSL) on general audio spectrograms. While recent informed masking techniques have attracted attention, we observe that they incur substantial computational overhead. Motivated by this observation, we propose dispersion-weighted masking (DWM), a lightweight masking strategy that leverages the spectral sparsity inherent in the frequency structure of audio content. Our experiments show that inverse block masking, commonly used in recent SSL frameworks, improves audio event understanding performance while introducing a trade-off in generalization. The proposed DWM alleviates these limitations and computational complexity, leading to consistent performance improvements. This work provides practical guidance on masking strategy design for masked prediction-based audio representation learning.
>
---
#### [new 006] Crab: Multi Layer Contrastive Supervision to Improve Speech Emotion Recognition Under Both Acted and Natural Speech Condition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决真实场景下情感分类困难和数据不平衡问题。提出Crab模型，结合多层对比监督策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.23673](https://arxiv.org/pdf/2603.23673)**

> **作者:** Lucas H. Ueda; João G. T. Lima; Paula D. P. Costa
>
> **备注:** IEEE Transactions on Affective Computing submission
>
> **摘要:** Speech Emotion Recognition (SER) in real-world scenarios remains challenging due to severe class imbalance and the prevalence of spontaneous, natural speech. While recent approaches leverage self-supervised learning (SSL) representations and multimodal fusion of speech and text, most existing methods apply supervision only at the final classification layer, limiting the discriminative power of intermediate representations. In this work, we propose Crab (Contrastive Representation and Multimodal Aligned Bottleneck), a bimodal Cross-Modal Transformer architecture that integrates speech representations from WavLM and textual representations from RoBERTa, together with a novel \textit{Multi Layer Contrastive Supervision} (MLCS) strategy. MLCS injects multi-positive contrastive learning signals at multiple layers of the network, encouraging emotionally discriminative representations throughout the model without introducing additional parameters at inference time. To further address data imbalance, we adopt weighted cross-entropy during training. We evaluate the proposed approach on three benchmark datasets covering different degrees of emotional naturalness: IEMOCAP, MELD, and MSP-Podcast 2.0. Experimental results demonstrate that Crab consistently outperforms strong unimodal and multimodal baselines across all datasets, with particularly large gains under naturalistic and highly imbalanced conditions. These findings highlight the effectiveness of \textit{Multi Layer Contrastive Supervision} as a general and robust strategy for SER. Official implementation can be found in this https URL.
>
---
#### [new 007] ACAVCaps: Enabling large-scale training for fine-grained and diverse audio understanding
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出ACAVCaps数据集，解决音频理解数据不足的问题，用于提升音频语言模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.24038](https://arxiv.org/pdf/2603.24038)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Junbo Zhang; Jian Luan
>
> **备注:** accepted by ICASSP 2026
>
> **摘要:** General audio understanding is a fundamental goal for large audio-language models, with audio captioning serving as a cornerstone task for their development. However, progress in this domain is hindered by existing datasets, which lack the scale and descriptive granularity required to train truly versatile models. To address this gap, we introduce ACAVCaps, a new large-scale, fine-grained, and multi-faceted audio captioning dataset. Derived from the ACAV100M collection, ACAVCaps is constructed using a multi-expert pipeline that analyzes audio from diverse perspectives-including speech, music, and acoustic properties-which are then synthesized into rich, detailed descriptions by a large language model. Experimental results demonstrate that models pre-trained on ACAVCaps exhibit substantially stronger generalization capabilities on various downstream tasks compared to those trained on other leading captioning datasets. The dataset is available at this https URL.
>
---
#### [new 008] Iterate to Differentiate: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决零样本TTS模型评估难题。提出I2D框架，通过迭代合成提升评估可靠性与区分度。**

- **链接: [https://arxiv.org/pdf/2603.24430](https://arxiv.org/pdf/2603.24430)**

> **作者:** Shengfan Shen; Di Wu; Xingchen Song; Dinghao Zhou; Liumeng Xue; Meng Meng; Jian Luan; Shuai Wang
>
> **备注:** submitted to Interspeech 2026, under review
>
> **摘要:** Reliable evaluation of modern zero-shot text-to-speech (TTS) models remains challenging. Subjective tests are costly and hard to reproduce, while objective metrics often saturate, failing to distinguish SOTA systems. To address this, we propose Iterate to Differentiate (I2D), an evaluation framework that recursively synthesizes speech using the model's own outputs as references. Higher-quality models exhibit greater resilience to the distributional shift induced by iterative synthesis, resulting in slower performance degradation. I2D exploits this differential degradation to amplify performance gaps and reveal robustness. By aggregating objective metrics across iterations, I2D improves discriminability and alignment with human judgments, increasing system-level SRCC from 0.118 to 0.464 for UTMOSv2. Experiments on 11 models across Chinese, English, and emotion datasets demonstrate that I2D enables more reliable automated evaluation for zero-shot TTS.
>
---
#### [new 009] How Open is Open TTS? A Practical Evaluation of Open Source TTS Tools for Romanian
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成（TTS）任务，评估开源TTS工具在罗马尼亚语中的适用性，解决低资源语言开发难题，通过多维度实验分析工具链、数据处理和计算效率。**

- **链接: [https://arxiv.org/pdf/2603.24116](https://arxiv.org/pdf/2603.24116)**

> **作者:** Teodora Răgman; Adrian Bogdan Stânea; Horia Cucu; Adriana Stan
>
> **备注:** Published in IEEE Access
>
> **摘要:** Open-source text-to-speech (TTS) frameworks have emerged as highly adaptable platforms for developing speech synthesis systems across a wide range of languages. However, their applicability is not uniform -- particularly when the target language is under-resourced or when computational resources are constrained. In this study, we systematically assess the feasibility of building novel TTS models using four widely adopted open-source architectures: FastPitch, VITS, Grad-TTS, and Matcha-TTS. Our evaluation spans multiple dimensions, including qualitative aspects such as ease of installation, dataset preparation, and hardware requirements, as well as quantitative assessments of synthesis quality for Romanian. We employ both objective metrics and subjective listening tests to evaluate intelligibility, speaker similarity, and naturalness of the generated speech. The results reveal significant challenges in tool chain setup, data preprocessing, and computational efficiency, which can hinder adoption in low-resource contexts. By grounding the analysis in reproducible protocols and accessible evaluation criteria, this work aims to inform best practices and promote more inclusive, language-diverse TTS development. All information needed to reproduce this study (i.e. code and data) are available in our git repository: this https URL
>
---
#### [new 010] Echoes: A semantically-aligned music deepfake detection dataset
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Echoes数据集，用于音乐深度伪造检测任务。旨在解决现有数据集泛化能力差的问题，通过引入多提供者和语义对齐的音频样本，提升检测模型的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.23667](https://arxiv.org/pdf/2603.23667)**

> **作者:** Octavian Pascu; Dan Oneata; Horia Cucu; Nicolas M. Muller
>
> **摘要:** We introduce Echoes, a new dataset for music deepfake detection designed for training and benchmarking detectors under realistic and provider-diverse conditions. Echoes comprises 3,577 tracks (110 hours of audio) spanning multiple genres (pop, rock, electronic), and includes content generated by ten popular AI music generation systems. To prevent shortcut learning and promote robust generalization, the dataset is deliberately constructed to be challenging, enforcing semantic-level alignment between spoofed audio and bona fide references. This alignment is achieved by conditioning generated audio samples directly on bona-fide waveforms or song descriptors. We evaluate Echoes in a cross-dataset setting against three existing AI-generated music datasets using state-of-the-art Wav2Vec2 XLS-R 2B representations. Results show that (i) Echoes is the hardest in-domain dataset; (ii) detectors trained on existing datasets transfer poorly to Echoes; (iii) training on Echoes yields the strongest generalization performance. These findings suggest that provider diversity and semantic alignment help learn more transferable detection cues.
>
---
#### [new 011] Variable-Length Audio Fingerprinting
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文属于音频指纹识别任务，解决固定长度音频处理的局限性。提出VLAFP方法，实现可变长度音频的指纹生成与识别。**

- **链接: [https://arxiv.org/pdf/2603.23947](https://arxiv.org/pdf/2603.23947)**

> **作者:** Hongjie Chen; Hanyu Meng; Huimin Zeng; Ryan A. Rossi; Lie Lu; Josh Kimball
>
> **摘要:** Audio fingerprinting converts audio to much lower-dimensional representations, allowing distorted recordings to still be recognized as their originals through similar fingerprints. Existing deep learning approaches rigidly fingerprint fixed-length audio segments, thereby neglecting temporal dynamics during segmentation. To address limitations due to this rigidity, we propose Variable-Length Audio FingerPrinting (VLAFP), a novel method that supports variable-length fingerprinting. To the best of our knowledge, VLAFP is the first deep audio fingerprinting model capable of processing audio of variable length, for both training and testing. Our experiments show that VLAFP outperforms existing state-of-the-arts in live audio identification and audio retrieval across three real-world datasets.
>
---
#### [new 012] Enhancing Efficiency and Performance in Deepfake Audio Detection through Neuron-level dropin & Neuroplasticity Mechanisms
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于深度伪造音频检测任务，旨在提升检测效率与性能。针对现有方法参数扩展受限的问题，提出dropin和神经可塑性机制，动态调整神经元数量，显著降低错误率并提高计算效率。**

- **链接: [https://arxiv.org/pdf/2603.24343](https://arxiv.org/pdf/2603.24343)**

> **作者:** Yupei Li; Shuaijie Shao; Manuel Milling; Björn Schuller
>
> **备注:** Accepted at IJCNN 2026
>
> **摘要:** Current audio deepfake detection has achieved remarkable performance using diverse deep learning architectures such as ResNet, and has seen further improvements with the introduction of large models (LMs) like Wav2Vec. The success of large language models (LLMs) further demonstrates the benefits of scaling model parameters, but also highlights one bottleneck where performance gains are constrained by parameter counts. Simply stacking additional layers, as done in current LLMs, is computationally expensive and requires full retraining. Furthermore, existing low-rank adaptation methods are primarily applied to attention-based architectures, which limits their scope. Inspired by the neuronal plasticity observed in mammalian brains, we propose novel algorithms, dropin and further plasticity, that dynamically adjust the number of neurons in certain layers to flexibly modulate model parameters. We evaluate these algorithms on multiple architectures, including ResNet, Gated Recurrent Neural Networks, and Wav2Vec. Experimental results using the widely recognised ASVSpoof2019 LA, PA, and FakeorReal dataset demonstrate consistent improvements in computational efficiency with the dropin approach and a maximum of around 39% and 66% relative reduction in Equal Error Rate with the dropin and plasticity approach among these dataset, respectively. The code and supplementary material are available at Github link.
>
---
#### [new 013] Photogrammetry-Reconstructed 3D Head Meshes for Accessible Individual Head-Related Transfer Functions
- **分类: eess.AS**

- **简介: 该论文属于音频渲染任务，旨在解决个体HRTF获取困难的问题。通过摄影测量重建头部模型，生成合成HRTF，并评估其性能。**

- **链接: [https://arxiv.org/pdf/2603.24104](https://arxiv.org/pdf/2603.24104)**

> **作者:** Ludovic Pirard; Lorenzo Picinali; Katarina C. Poole
>
> **备注:** Submitted to Acta Acustica Topical Issue - Spatial and binaural hearing: From neural processes to applications
>
> **摘要:** Individual head-related transfer functions (HRTFs) are essential for accurate spatial audio binaural rendering but remain difficult to obtain due to measurement complexity. This study investigates whether photogrammetry-reconstructed (PR) head and ear meshes, acquired with consumer hardware, can provide a practically useful baseline for individual HRTF synthesis. Using the SONICOM HRTF dataset, 72-image photogrammetry captures per subject were processed with Apple's Object Capture API to generate PR meshes for 150 subjects. Mesh2HRTF was used to compute PR synthetic HRTFs, which were compared against measured HRTFs, high-resolution 3D scan-derived HRTFs, KEMAR, and random HRTFs through numerical evaluation, auditory models, and a behavioural sound localisation experiment (N = 27). PR synthetic HRTFs preserved ITD cues but exhibited increased ILD and spectral errors. Auditory-model predictions and behavioural data showed substantially higher quadrant error rates, reduced elevation accuracy, and greater front-back confusions than measured HRTFs, performing worse than random HRTFs on perceptual metrics. Current photogrammetry pipelines support individual HRTF synthesis but are limited by insufficient pinna morphology details and high-frequency spectral fidelity needed for accurate individual HRTFs containing monaural cues.
>
---
#### [new 014] YingMusic-Singer: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，解决歌词修改时保持旋律一致的问题。提出YingMusic-Singer模型，无需人工对齐即可灵活调整歌词并控制旋律。**

- **链接: [https://arxiv.org/pdf/2603.24589](https://arxiv.org/pdf/2603.24589)**

> **作者:** Chunbo Hao; Junjie Zheng; Guobin Ma; Yuepeng Jiang; Huakang Chen; Wenjie Tian; Gongyu Chen; Zihao Chen; Lei Xie
>
> **摘要:** Regenerating singing voices with altered lyrics while preserving melody consistency remains challenging, as existing methods either offer limited controllability or require laborious manual alignment. We propose YingMusic-Singer, a fully diffusion-based model enabling melody-controllable singing voice synthesis with flexible lyric manipulation. The model takes three inputs: an optional timbre reference, a melody-providing singing clip, and modified lyrics, without manual alignment. Trained with curriculum learning and Group Relative Policy Optimization, YingMusic-Singer achieves stronger melody preservation and lyric adherence than Vevo2, the most comparable baseline supporting melody control without manual alignment. We also introduce LyricEditBench, the first benchmark for melody-preserving lyric modification evaluation. The code, weights, benchmark, and demos are publicly available at this https URL.
>
---
#### [new 015] Bridging Biological Hearing and Neuromorphic Computing: End-to-End Time-Domain Audio Signal Processing with Reservoir Computing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音信号处理任务，旨在提升实时音频处理效率。针对传统MFCC计算复杂的问题，提出基于储备计算的端到端方法，简化特征提取过程。**

- **链接: [https://arxiv.org/pdf/2603.24283](https://arxiv.org/pdf/2603.24283)**

> **作者:** Rinku Sebastian; Simon O'Keefe; Martin Trefzer
>
> **摘要:** Despite the advancements in cutting-edge technologies, audio signal processing continues to pose challenges and lacks the precision of a human speech processing system. To address these challenges, we propose a novel approach to simplify audio signal processing by leveraging time-domain techniques and reservoir computing. Through our research, we have developed a real-time audio signal processing system by simplifying audio signal processing through the utilization of reservoir computers, which are significantly easier to train. Feature extraction is a fundamental step in speech signal processing, with Mel Frequency Cepstral Coefficients (MFCCs) being a dominant choice due to their perceptual relevance to human hearing. However, conventional MFCC extraction relies on computationally intensive time-frequency transformations, limiting efficiency in real-time applications. To address this, we propose a novel approach that leverages reservoir computing to streamline MFCC extraction. By replacing traditional frequency-domain conversions with convolution operations, we eliminate the need for complex transformations while maintaining feature discriminability. We present an end-to-end audio processing framework that integrates this method, demonstrating its potential for efficient and real-time speech analysis. Our results contribute to the advancement of energy-efficient audio processing technologies, enabling seamless deployment in embedded systems and voice-driven applications. This work bridges the gap between biologically inspired feature extraction and modern neuromorphic computing, offering a scalable solution for next-generation speech recognition systems.
>
---
#### [new 016] A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于自然语言处理任务，研究ASR系统在纽卡斯尔英语中的偏见问题。通过分析语音识别错误，揭示社会因素与语音变异对识别准确率的影响。**

- **链接: [https://arxiv.org/pdf/2603.24549](https://arxiv.org/pdf/2603.24549)**

> **作者:** Dana Serditova; Kevin Tang
>
> **备注:** 54 pages, 11 figures
>
> **摘要:** Automatic Speech Recognition (ASR) systems are widely used in everyday communication, education, healthcare, and industry, yet their performance remains uneven across speakers, particularly when dialectal variation diverges from the mainstream accents represented in training data. This study investigates ASR bias through a sociolinguistic analysis of Newcastle English, a regional variety of North-East England that has been shown to challenge current speech recognition technologies. Using spontaneous speech from the Diachronic Electronic Corpus of Tyneside English (DECTE), we evaluate the output of a state-of-the-art commercial ASR system and conduct a fine-grained analysis of more than 3,000 transcription errors. Errors are classified by linguistic domain and examined in relation to social variables including gender, age, and socioeconomic status. In addition, an acoustic case study of selected vowel features demonstrates how gradient phonetic variation contributes directly to misrecognition. The results show that phonological variation accounts for the majority of errors, with recurrent failures linked to dialect-specific features like vowel quality and glottalisation, as well as local vocabulary and non-standard grammatical forms. Error rates also vary across social groups, with higher error frequencies observed for men and for speakers at the extremes of the age spectrum. These findings indicate that ASR errors are not random but socially patterned and can be explained from a sociolinguistic perspective. Thus, the study demonstrates the importance of incorporating sociolinguistic expertise into the evaluation and development of speech technologies and argues that more equitable ASR systems require explicit attention to dialectal variation and community-based speech data.
>
---
## 更新

#### [replaced 001] DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Trained Speech Foundational Model
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出DELULU模型，解决语音任务中缺乏说话人判别特征的问题。通过引入说话人信息提升自监督学习效果，显著提升说话人验证和零样本分析性能。**

- **链接: [https://arxiv.org/pdf/2510.17662](https://arxiv.org/pdf/2510.17662)**

> **作者:** Massa Baali; Rita Singh; Bhiksha Raj
>
> **摘要:** Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce \textsc{DELULU}, a speaker-aware self-trained foundational model that addresses this limitation by incorporating speaker-informed structure into pseudo-label generation. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide k-means clustering during pre-training, introducing a speaker-discriminative inductive bias that aligns representation learning with speaker identity. DELULU significantly outperforms prior SSL models across a range of speaker-centric tasks, achieving up to \textbf{62\% relative improvement} in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks including gender, age, accent, and speaker counting; notably surpassing even its teacher model on zero-shot evaluations. Our findings demonstrate that \textbf{DELULU is a strong universal encoder for speaker-aware speech processing}, enabling superior performance without task-specific fine-tuning.
>
---
#### [replaced 002] Adaptive Federated Fine-Tuning of Self-Supervised Speech Representations
- **分类: eess.AS**

- **简介: 该论文属于语音任务的联邦学习领域，解决联邦环境中的异构性和效率问题。提出自适应联邦微调框架，通过轻量预测头和分层聚合策略提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.21888](https://arxiv.org/pdf/2603.21888)**

> **作者:** Xin Guo; Chunrui Zhao; Hong Jia; Ting Dang; Gongping Huang; Xianrui Zheng; Yan Gao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Integrating Federated Learning (FL) with self-supervised learning (SSL) enables privacy-preserving fine-tuning for speech tasks. However, federated environments exhibit significant heterogeneity: clients differ in computational capacity, causing straggler effects under unified fine-tuning, while diverse downstream tasks require different representation depths, making full-model updates inefficient. To address these challenges, we propose an adaptive federated fine-tuning framework with early exits. Lightweight prediction heads are inserted at intermediate layers of the SSL backbone, allowing clients to terminate computation based on local constraints and task requirements. We further introduce a layer-wise, depth-aware partial aggregation strategy to better utilize representations from different network depths. Experiments show that the framework reduces edge overhead, supports heterogeneous hardware, and maintains competitive performance in resource-constrained federated environments.
>
---
#### [replaced 003] OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出OmniCustom，解决同步音视频定制任务，通过联合生成模型同时保持视频身份和音频音色，实现零样本音视频合成。**

- **链接: [https://arxiv.org/pdf/2602.12304](https://arxiv.org/pdf/2602.12304)**

> **作者:** Maomao Li; Zhen Li; Kaipeng Zhang; Guosheng Yin; Zhifeng Li; Dong Xu
>
> **备注:** code: this https URL
>
> **摘要:** Existing mainstream video customization methods focus on generating identity-consistent videos based on given reference images and textual prompts. Benefiting from the rapid advancement of joint audio-video generation, this paper proposes a more compelling new task: sync audio-video customization, which aims to synchronously customize both video identity and audio timbre. Specifically, given a reference image $I^{r}$ and a reference audio $A^{r}$, this novel task requires generating videos that maintain the identity of the reference image while imitating the timbre of the reference audio, with spoken content freely specifiable through user-provided textual prompts. To this end, we propose OmniCustom, a powerful DiT-based audio-video customization framework that can synthesize a video following reference image identity, audio timbre, and text prompts all at once in a zero-shot manner. Our framework is built on three key contributions. First, identity and audio timbre control are achieved through separate reference identity and audio LoRA modules that operate through self-attention layers within the base audio-video generation model. Second, we introduce a contrastive learning objective alongside the standard flow matching objective. It uses predicted flows conditioned on reference inputs as positive examples and those without reference conditions as negative examples, thereby enhancing the model ability to preserve identity and timbre. Third, we train OmniCustom on our constructed large-scale, high-quality audio-visual human dataset. Extensive experiments demonstrate that OmniCustom outperforms existing methods in generating audio-video content with consistent identity and timbre fidelity. Project page: this https URL.
>
---
#### [replaced 004] An interpretable speech foundation model for depression detection by revealing prediction-relevant acoustic features from long speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在通过长时语音提升检测准确性并增强模型可解释性。工作包括提出一种可解释的语音基础模型，识别关键声学特征，以支持临床应用。**

- **链接: [https://arxiv.org/pdf/2406.03138](https://arxiv.org/pdf/2406.03138)**

> **作者:** Qingkun Deng; Saturnino Luz; Sofia de la Fuente Garcia
>
> **备注:** 5 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2309.13476
>
> **摘要:** Speech-based depression detection tools could aid early screening. Here, we propose an interpretable speech foundation model approach to enhance the clinical applicability of such tools. We introduce a speech-level Audio Spectrogram Transformer (AST) to detect depression using long-duration speech instead of short segments, along with a novel interpretation method that reveals prediction-relevant acoustic features for clinician interpretation. Our experiments show the proposed model outperforms a segment-level AST, highlighting the impact of segment-level labelling noise and the advantage of leveraging longer speech duration for more reliable depression detection. Through interpretation, we observe our model identifies reduced loudness and F0 as relevant depression signals, aligning with documented clinical findings. This interpretability supports a responsible AI approach for speech-based depression detection, rendering such tools more clinically applicable.
>
---
