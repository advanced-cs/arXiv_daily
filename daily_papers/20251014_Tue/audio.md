# 音频 cs.SD;  eess.AS

- **最新发布 33 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Improving Speech Emotion Recognition with Mutual Information Regularized Generative Model
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究语音情感识别（SER）任务，旨在缓解标注数据不足问题。作者提出一种基于互信息正则化的生成模型框架，通过跨模态信息迁移和数据增强提升性能，并在多个基准数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.10078v1](http://arxiv.org/pdf/2510.10078v1)**

> **作者:** Chung-Soo Ahn; Rajib Rana; Sunil Sivadas; Carlos Busso; Jagath C. Rajapakse
>
> **摘要:** Although speech emotion recognition (SER) research has been advanced, thanks to deep learning methods, it still suffers from obtaining inputs from large quality-labelled training data. Data augmentation methods have been attempted to mitigate this issue, generative models have shown success among them recently. We propose a data augmentation framework that is aided by cross-modal information transfer and mutual information regularization. Mutual information based metric can serve as an indicator for the quality. Furthermore, we expand this data augmentation scope to multimodal inputs, thanks to mutual information ensureing dependency between modalities. Our framework was tested on three benchmark datasets: IEMOCAP, MSP-IMPROV and MSP-Podcast. The implementation was designed to generate input features that are fed into last layer for emotion classification. Our framework improved the performance of emotion prediction against existing works. Also, we discovered that our framework is able to generate new inputs without any cross-modal information.
>
---
#### [new 002] MSRBench: A Benchmarking Dataset for Music Source Restoration
- **分类: cs.SD**

- **简介: 该论文针对音乐源修复（MSR）任务，旨在恢复经处理和退化后的原始音频源。现有数据集缺乏真实参考与真实退化混合的配对数据。作者构建了MSRBench，首个包含专业混音与多种真实退化的原始-处理音频对的基准数据集，支持分离与恢复效果评估。**

- **链接: [http://arxiv.org/pdf/2510.10995v1](http://arxiv.org/pdf/2510.10995v1)**

> **作者:** Yongyi Zang; Jiarui Hai; Wanying Ge; Qiuqiang Kong; Zheqi Dai; Helin Wang; Yuki Mitsufuji; Mark D. Plumbley
>
> **摘要:** Music Source Restoration (MSR) extends source separation to realistic settings where signals undergo production effects (equalization, compression, reverb) and real-world degradations, with the goal of recovering the original unprocessed sources. Existing benchmarks cannot measure restoration fidelity: synthetic datasets use unprocessed stems but unrealistic mixtures, while real production datasets provide only already-processed stems without clean references. We present MSRBench, the first benchmark explicitly designed for MSR evaluation. MSRBench contains raw stem-mixture pairs across eight instrument classes, where mixtures are produced by professional mixing engineers. These raw-processed pairs enable direct evaluation of both separation accuracy and restoration fidelity. Beyond controlled studio conditions, the mixtures are augmented with twelve real-world degradations spanning analog artifacts, acoustic environments, and lossy codecs. Baseline experiments with U-Net and BSRNN achieve SI-SNR of -37.8 dB and -23.4 dB respectively, with perceptual quality (FAD CLAP) around 0.7-0.8, demonstrating substantial room for improvement and the need for restoration-specific architectures.
>
---
#### [new 003] Peransformer: Improving Low-informed Expressive Performance Rendering with Score-aware Discriminator
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究低信息输入的音乐表现力渲染（EPR）任务，旨在提升仅用MIDI输入生成自然演奏效果的性能。作者提出Peransformer模型，引入乐谱感知判别器，并构建统一评估指标GEM，显著提升了低信息EPR系统的表达能力和可比性。**

- **链接: [http://arxiv.org/pdf/2510.10175v1](http://arxiv.org/pdf/2510.10175v1)**

> **作者:** Xian He; Wei Zeng; Ye Wang
>
> **备注:** 6 pages, 3 figures, accepted by APSIPA ASC 2025
>
> **摘要:** Highly-informed Expressive Performance Rendering (EPR) systems transform music scores with rich musical annotations into human-like expressive performance MIDI files. While these systems have achieved promising results, the availability of detailed music scores is limited compared to MIDI files and are less flexible to work with using a digital audio workstation (DAW). Recent advancements in low-informed EPR systems offer a more accessible alternative by directly utilizing score-derived MIDI as input, but these systems often exhibit suboptimal performance. Meanwhile, existing works are evaluated with diverse automatic metrics and data formats, hindering direct objective comparisons between EPR systems. In this study, we introduce Peransformer, a transformer-based low-informed EPR system designed to bridge the gap between low-informed and highly-informed EPR systems. Our approach incorporates a score-aware discriminator that leverages the underlying score-derived MIDI files and is trained on a score-to-performance paired, note-to-note aligned MIDI dataset. Experimental results demonstrate that Peransformer achieves state-of-the-art performance among low-informed systems, as validated by subjective evaluations. Furthermore, we extend existing automatic evaluation metrics for EPR systems and introduce generalized EPR metrics (GEM), enabling more direct, accurate, and reliable comparisons across EPR systems.
>
---
#### [new 004] Knowledge-Decoupled Functionally Invariant Path with Synthetic Personal Data for Personalized ASR
- **分类: cs.SD**

- **简介: 该论文研究个性化语音识别（ASR），解决合成个人数据微调中遗忘通用与真实知识的问题。提出知识解耦的FIP框架（KDFIP），分离存储通用与个性化知识，通过门控机制融合，实现知识平衡学习。**

- **链接: [http://arxiv.org/pdf/2510.10401v1](http://arxiv.org/pdf/2510.10401v1)**

> **作者:** Yue Gu; Zhihao Du; Ying Shi; Jiqing Han; Yongjun He
>
> **备注:** Accepted for publication in IEEE Signal Processing Letters, 2025
>
> **摘要:** Fine-tuning generic ASR models with large-scale synthetic personal data can enhance the personalization of ASR models, but it introduces challenges in adapting to synthetic personal data without forgetting real knowledge, and in adapting to personal data without forgetting generic knowledge. Considering that the functionally invariant path (FIP) framework enables model adaptation while preserving prior knowledge, in this letter, we introduce FIP into synthetic-data-augmented personalized ASR models. However, the model still struggles to balance the learning of synthetic, personalized, and generic knowledge when applying FIP to train the model on all three types of data simultaneously. To decouple this learning process and further address the above two challenges, we integrate a gated parameter-isolation strategy into FIP and propose a knowledge-decoupled functionally invariant path (KDFIP) framework, which stores generic and personalized knowledge in separate modules and applies FIP to them sequentially. Specifically, KDFIP adapts the personalized module to synthetic and real personal data and the generic module to generic data. Both modules are updated along personalization-invariant paths, and their outputs are dynamically fused through a gating mechanism. With augmented synthetic data, KDFIP achieves a 29.38% relative character error rate reduction on target speakers and maintains comparable generalization performance to the unadapted ASR baseline.
>
---
#### [new 005] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究通用声音分离，解决传统方法语义干扰严重的问题。提出MARS-Sep框架，采用强化学习优化分离策略，结合多模态奖励和渐进对齐机制，提升分离结果的语义一致性和信号质量。**

- **链接: [http://arxiv.org/pdf/2510.10509v1](http://arxiv.org/pdf/2510.10509v1)**

> **作者:** Zihan Zhang; Xize Cheng; Zhennan Jiang; Dongjie Fu; Jingyuan Chen; Zhou Zhao; Tao Jin
>
> **摘要:** Universal sound separation faces a fundamental misalignment: models optimized for low-level signal metrics often produce semantically contaminated outputs, failing to suppress perceptually salient interference from acoustically similar sources. To bridge this gap, we introduce MARS-Sep, a reinforcement learning framework that reformulates separation as decision making. Instead of simply regressing ground-truth masks, MARS-Sep learns a factorized Beta mask policy that is optimized by a clipped trust-region surrogate with entropy regularization and group-relative advantage normalization. Concretely, we sample masks from a frozen old policy, reconstruct waveforms, and update the current policy using clipped importance ratios-yielding substantially more stable and sample-efficient learning. Multimodal rewards, derived from an audio-text-vision encoder, directly incentivize semantic consistency with query prompts. We further propose a progressive alignment scheme to fine-tune this encoder, boosting its cross-modal discriminability and improving reward faithfulness. Extensive experiments on multiple benchmarks demonstrate consistent gains in Text-, Audio-, and Image-Queried separation, with notable improvements in signal metrics and semantic quality. Our code is available at https://anonymous.4open.science/r/MARS-Sep. Sound separation samples are available at https://mars-sep.github.io/.
>
---
#### [new 006] Automatic Music Sample Identification with Multi-Track Contrastive Learning
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文研究自动音乐采样识别任务，旨在检测音乐中使用的采样片段并溯源。作者提出基于多轨数据的对比学习方法，通过自监督学习提升识别性能，实验证明其方法优于现有技术，且对不同音乐类型鲁棒，强调高质量音轨分离的重要性。**

- **链接: [http://arxiv.org/pdf/2510.11507v1](http://arxiv.org/pdf/2510.11507v1)**

> **作者:** Alain Riou; Joan Serrà; Yuki Mitsufuji
>
> **摘要:** Sampling, the technique of reusing pieces of existing audio tracks to create new music content, is a very common practice in modern music production. In this paper, we tackle the challenging task of automatic sample identification, that is, detecting such sampled content and retrieving the material from which it originates. To do so, we adopt a self-supervised learning approach that leverages a multi-track dataset to create positive pairs of artificial mixes, and design a novel contrastive learning objective. We show that such method significantly outperforms previous state-of-the-art baselines, that is robust to various genres, and that scales well when increasing the number of noise songs in the reference database. In addition, we extensively analyze the contribution of the different components of our training pipeline and highlight, in particular, the need for high-quality separated stems for this task.
>
---
#### [new 007] Perceptual Compensation of Ambisonics Recordings for Reproduction in Room
- **分类: eess.AS; physics.app-ph**

- **简介: 该论文针对Ambisonics在真实房间重放时受混响影响的问题，提出一种感知激励的补偿方法，通过在球谐域对直达声与混响声进行谱和空间补偿，保留关键听觉线索，提升声音场再现的感知准确性。**

- **链接: [http://arxiv.org/pdf/2510.10883v1](http://arxiv.org/pdf/2510.10883v1)**

> **作者:** Ali Fallah; Shun Nakamura; Steven van de Par
>
> **备注:** The manuscript was submitted to the JASA and is under review
>
> **摘要:** Ambisonics is a method for capturing and rendering a sound field accurately, assuming that the acoustics of the playback room does not significantly influence the sound field. However, in practice, the acoustics of the playback room may lead to a noticeable degradation in sound quality. We propose a recording and rendering method based on Ambisonics that utilizes a perceptually-motivated approach to compensate for the reverberation of the playback room. The recorded direct and reverberant sound field components in the spherical harmonics (SHs) domain are spectrally and spatially compensated to preserve the relevant auditory cues including the direction of arrival of the direct sound, the spectral energy of the direct and reverberant sound components, and the Interaural Coherence (IC) across each auditory band. In contrast to the conventional Ambisonics, a flexible number of Ambisonics channels can be used for audio rendering. Listening test results show that the proposed method provides a perceptually accurate rendering of the originally recorded sound field, outperforming both conventional Ambisonics without compensation and even ideal Ambisonics rendering in a simulated anechoic room. Additionally, subjective evaluations of listeners seated at the center of the loudspeaker array demonstrate that the method remains robust to head rotation and minor displacements.
>
---
#### [new 008] Phase Aware Ear-Conditioned Learning for Multi-Channel Binaural Speaker Separation
- **分类: eess.AS**

- **简介: 该论文研究多通道双耳语音分离，旨在保留空间线索的同时提升分离与去混响性能。提出PEASE-8模型，利用八麦克风输入复数STFT和原始STFT直连解码器，端到端优化SI-SDR，实现无需排列不变训练的高效分离。**

- **链接: [http://arxiv.org/pdf/2510.11366v1](http://arxiv.org/pdf/2510.11366v1)**

> **作者:** Ruben Johnson Robert Jeremiah; Peyman Goli; Steven van de Par
>
> **摘要:** Separating competing speech in reverberant environments requires models that preserve spatial cues while maintaining separation efficiency. We present a Phase-aware Ear-conditioned speaker Separation network using eight microphones (PEASE-8) that consumes complex STFTs and directly introduces a raw-STFT input to the early decoder layer, bypassing the entire encoder pathway to improve reconstruction. The model is trained end-to-end with an SI-SDR-based objective against direct-path ear targets, jointly performing separation and dereverberation for two speakers in a fixed azimuth, eliminating the need for permutation invariant training. On spatialized two-speaker mixtures spanning anechoic, reverberant, and noisy conditions, PEASE-8 delivers strong separation and intelligibility. In reverberant environments, it achieves 12.37 dB SI-SDR, 0.87 STOI, and 1.86 PESQ at T60 = 0.6 s, while remaining competitive under anechoic conditions.
>
---
#### [new 009] ProGress: Structured Music Generation via Graph Diffusion and Hierarchical Music Analysis
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文聚焦音乐生成任务，旨在解决现有模型缺乏结构连贯性和可解释性的问题。作者提出ProGress框架，结合Schenkerian分析与图扩散模型，实现可控制、可解释的层次化音乐生成，并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2510.10249v1](http://arxiv.org/pdf/2510.10249v1)**

> **作者:** Stephen Ni-Hahn; Chao Péter Yang; Mingchen Ma; Cynthia Rudin; Simon Mak; Yue Jiang
>
> **摘要:** Artificial Intelligence (AI) for music generation is undergoing rapid developments, with recent symbolic models leveraging sophisticated deep learning and diffusion model algorithms. One drawback with existing models is that they lack structural cohesion, particularly on harmonic-melodic structure. Furthermore, such existing models are largely "black-box" in nature and are not musically interpretable. This paper addresses these limitations via a novel generative music framework that incorporates concepts of Schenkerian analysis (SchA) in concert with a diffusion modeling framework. This framework, which we call ProGress (Prolongation-enhanced DiGress), adapts state-of-the-art deep models for discrete diffusion (in particular, the DiGress model of Vignac et al., 2023) for interpretable and structured music generation. Concretely, our contributions include 1) novel adaptations of the DiGress model for music generation, 2) a novel SchA-inspired phrase fusion methodology, and 3) a framework allowing users to control various aspects of the generation process to create coherent musical compositions. Results from human experiments suggest superior performance to existing state-of-the-art methods.
>
---
#### [new 010] Universal Discrete-Domain Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文研究语音增强任务，旨在解决多种失真共存时的通用性问题。提出UDSE模型，将增强视为离散域分类任务，通过预测预训练语音编解码器的量化token来恢复干净语音，显著提升复杂失真下的鲁棒性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.09974v1](http://arxiv.org/pdf/2510.09974v1)**

> **作者:** Fei Liu; Yang Ai; Ye-Xin Lu; Rui-Chen Zheng; Hui-Peng Du; Zhen-Hua Ling
>
> **摘要:** In real-world scenarios, speech signals are inevitably corrupted by various types of interference, making speech enhancement (SE) a critical task for robust speech processing. However, most existing SE methods only handle a limited range of distortions, such as additive noise, reverberation, or band limitation, while the study of SE under multiple simultaneous distortions remains limited. This gap affects the generalization and practical usability of SE methods in real-world environments.To address this gap, this paper proposes a novel Universal Discrete-domain SE model called UDSE.Unlike regression-based SE models that directly predict clean speech waveform or continuous features, UDSE redefines SE as a discrete-domain classification task, instead predicting the clean discrete tokens quantized by the residual vector quantizer (RVQ) of a pre-trained neural speech codec.Specifically, UDSE first extracts global features from the degraded speech. Guided by these global features, the clean token prediction for each VQ follows the rules of RVQ, where the prediction of each VQ relies on the results of the preceding ones. Finally, the predicted clean tokens from all VQs are decoded to reconstruct the clean speech waveform. During training, the UDSE model employs a teacher-forcing strategy, and is optimized with cross-entropy loss. Experimental results confirm that the proposed UDSE model can effectively enhance speech degraded by various conventional and unconventional distortions, e.g., additive noise, reverberation, band limitation, clipping, phase distortion, and compression distortion, as well as their combinations. These results demonstrate the superior universality and practicality of UDSE compared to advanced regression-based SE methods.
>
---
#### [new 011] FAC-FACodec: Controllable Zero-Shot Foreign Accent Conversion with Factorized Speech Codec
- **分类: cs.SD**

- **简介: 该论文研究外国口音转换任务，旨在解决转换程度不可控的问题。提出FAC-FACodec框架，通过分解语音编码器实现发音调整与超音段特征保留，并引入可调节参数控制转换强度，兼顾说话人身份保持。**

- **链接: [http://arxiv.org/pdf/2510.10785v1](http://arxiv.org/pdf/2510.10785v1)**

> **作者:** Yurii Halychanskyi; Cameron Churchwell; Yutong Wen; Volodymyr Kindratenko
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Previous accent conversion (AC) methods, including foreign accent conversion (FAC), lack explicit control over the degree of modification. Because accent modification can alter the perceived speaker identity, balancing conversion strength and identity preservation is crucial. We present an AC framework that provides an explicit, user-controllable parameter for accent modification. The method targets pronunciation while preserving suprasegmental cues such as intonation and phoneme durations. Results show performance comparable to recent AC systems, stronger preservation of speaker identity, and unique support for controllable accent conversion.
>
---
#### [new 012] Diffusion-Link: Diffusion Probabilistic Model for Bridging the Audio-Text Modality Gap
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文针对音频-文本模态鸿沟问题，提出Diffusion-Link，一种基于扩散模型的轻量模块，将音频嵌入映射到文本分布。用于自动音频描述任务，显著缩小模态差距，在AudioCaps上实现零样本和全监督SOTA性能。**

- **链接: [http://arxiv.org/pdf/2510.11330v1](http://arxiv.org/pdf/2510.11330v1)**

> **作者:** KiHyun Nam; Jongmin Choi; Hyeongkeun Lee; Jungwoo Heo; Joon Son Chung
>
> **备注:** 5 pages. Submitted to IEEE ICASSP 2026
>
> **摘要:** Contrastive audio-language pretraining yields powerful joint representations, yet a persistent audio-text modality gap limits the benefits of coupling multimodal encoders with large language models (LLMs). We present Diffusion-Link, a diffusion-based modality-bridging module that generatively maps audio embeddings into the text-embedding distribution. The module is trained at the output embedding from the frozen multimodal encoder and implemented as a lightweight network with three residual MLP blocks. To assess the effect of Diffusion-Link on multimodal encoder-LLM coupling, we evaluate on Automatic Audio Captioning (AAC); to our knowledge, this is the first application of diffusion-based modality bridging to AAC. We report two results. (1) Modality-gap analysis: on similarity and geometric criteria, Diffusion-Link reduces the modality gap the most among prior diffusion-based methods and shows a collective migration of audio embeddings toward the text distribution. (2) Downstream AAC: attaching Diffusion-Link to the same multimodal LLM baseline achieves state-of-the-art on AudioCaps in both zero-shot and fully supervised captioning without external knowledge, with relative gains up to 52.5% and 7.5%, respectively. These findings show that closing the modality gap is pivotal for effective coupling between multimodal encoders and LLMs, and diffusion-based modality bridging offers a promising direction beyond knowledge-retrieval-centric designs. Code will be released upon acceptance https://github.com/DevKiHyun/Diffusion-Link
>
---
#### [new 013] Dynamically Slimmable Speech Enhancement Network with Metric-Guided Training
- **分类: eess.AS**

- **简介: 该论文研究语音增强任务，旨在降低轻量模型的计算复杂度。提出动态可瘦身网络（DSN）和指标引导训练（MGT），根据输入质量自适应调整计算负载，在保持性能的同时显著减少计算开销。**

- **链接: [http://arxiv.org/pdf/2510.11395v1](http://arxiv.org/pdf/2510.11395v1)**

> **作者:** Haixin Zhao; Kaixuan Yang; Nilesh Madhu
>
> **备注:** Preprint version of a paper under review at ICASSP2026
>
> **摘要:** To further reduce the complexity of lightweight speech enhancement models, we introduce a gating-based Dynamically Slimmable Network (DSN). The DSN comprises static and dynamic components. For architecture-independent applicability, we introduce distinct dynamic structures targeting the commonly used components, namely, grouped recurrent neural network units, multi-head attention, convolutional, and fully connected layers. A policy module adaptively governs the use of dynamic parts at a frame-wise resolution according to the input signal quality, controlling computational load. We further propose Metric-Guided Training (MGT) to explicitly guide the policy module in assessing input speech quality. Experimental results demonstrate that the DSN achieves comparable enhancement performance in instrumental metrics to the state-of-the-art lightweight baseline, while using only 73% of its computational load on average. Evaluations of dynamic component usage ratios indicate that the MGT-DSN can appropriately allocate network resources according to the severity of input signal distortion.
>
---
#### [new 014] ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis
- **分类: cs.SD; cs.AI; cs.HC; cs.LG**

- **简介: 该论文针对波斯语高质量语音数据稀缺问题，提出大规模多说话人语音合成语料库ParsVoice。通过自动化流程处理2000本有声书，构建1804小时高质量语音数据，含470余名说话人，推动波斯语语音技术发展。**

- **链接: [http://arxiv.org/pdf/2510.10774v1](http://arxiv.org/pdf/2510.10774v1)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **摘要:** Persian Language, despite being spoken by over 100 million people worldwide, remains severely underrepresented in high-quality speech corpora, particularly for text-to-speech (TTS) synthesis applications. Existing Persian speech datasets are typically smaller than their English counterparts, which creates a key limitation for developing Persian speech technologies. We address this gap by introducing ParsVoice, the largest Persian speech corpus designed specifically for TTS applications. We created an automated pipeline that transforms raw audiobook content into TTS-ready data, incorporating components such as a BERT-based sentence completion detector, a binary search boundary optimization method for precise audio-text alignment, and multi-dimensional quality assessment frameworks tailored to Persian. The pipeline processes 2,000 audiobooks, yielding 3,526 hours of clean speech, which was further filtered into a 1,804-hour high-quality subset suitable for TTS, featuring more than 470 speakers. ParsVoice is the largest high-quality Persian speech dataset, offering speaker diversity and audio quality comparable to major English corpora. The complete dataset has been made publicly available to accelerate the development of Persian speech technologies and to serve as a template for other low-resource languages. The ParsVoice dataset is publicly available at ParsVoice (https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice).
>
---
#### [new 015] SS-DPPN: A self-supervised dual-path foundation model for the generalizable cardiac audio representation
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出SS-DPPN，一种自监督双路径基础模型，用于可泛化的 cardiac audio 表征。针对标注数据稀缺问题，通过双路径对比学习和原型网络提升数据效率，在心脏音分类等任务中实现SOTA，并成功迁移至肺音分类和心率估计。**

- **链接: [http://arxiv.org/pdf/2510.10719v1](http://arxiv.org/pdf/2510.10719v1)**

> **作者:** Ummy Maria Muna; Md Mehedi Hasan Shawon; Md Jobayer; Sumaiya Akter; Md Rakibul Hasan; Md. Golam Rabiul Alam
>
> **摘要:** The automated analysis of phonocardiograms is vital for the early diagnosis of cardiovascular disease, yet supervised deep learning is often constrained by the scarcity of expert-annotated data. In this paper, we propose the Self-Supervised Dual-Path Prototypical Network (SS-DPPN), a foundation model for cardiac audio representation and classification from unlabeled data. The framework introduces a dual-path contrastive learning based architecture that simultaneously processes 1D waveforms and 2D spectrograms using a novel hybrid loss. For the downstream task, a metric-learning approach using a Prototypical Network was used that enhances sensitivity and produces well-calibrated and trustworthy predictions. SS-DPPN achieves state-of-the-art performance on four cardiac audio benchmarks. The framework demonstrates exceptional data efficiency with a fully supervised model on three-fold reduction in labeled data. Finally, the learned representations generalize successfully across lung sound classification and heart rate estimation. Our experiments and findings validate SS-DPPN as a robust, reliable, and scalable foundation model for physiological signals.
>
---
#### [new 016] MRSAudio: A Large-Scale Multimodal Recorded Spatial Audio Dataset with Refined Annotations
- **分类: cs.SD**

- **简介: 该论文提出MRSAudio，一个大规模多模态空间音频数据集，旨在解决现有数据集中空间音频缺失的问题。涵盖真实场景的同步双耳/环境音频、视频及精细标注，支持空间音频生成与理解五项基础任务，推动VR/AR中听觉感知研究。**

- **链接: [http://arxiv.org/pdf/2510.10396v1](http://arxiv.org/pdf/2510.10396v1)**

> **作者:** Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Xintong Hu; Yu Zhang; Li Tang; Rui Yang; Han Wang; Zongbao Zhang; Yuhan Wang; Yixuan Chen; Hankun Xu; Ke Xu; Pengfei Fan; Zhetao Chen; Yanhao Yu; Qiange Huang; Fei Wu; Zhou Zhao
>
> **备注:** 24 pages
>
> **摘要:** Humans rely on multisensory integration to perceive spatial environments, where auditory cues enable sound source localization in three-dimensional space. Despite the critical role of spatial audio in immersive technologies such as VR/AR, most existing multimodal datasets provide only monaural audio, which limits the development of spatial audio generation and understanding. To address these challenges, we introduce MRSAudio, a large-scale multimodal spatial audio dataset designed to advance research in spatial audio understanding and generation. MRSAudio spans four distinct components: MRSLife, MRSSpeech, MRSMusic, and MRSSing, covering diverse real-world scenarios. The dataset includes synchronized binaural and ambisonic audio, exocentric and egocentric video, motion trajectories, and fine-grained annotations such as transcripts, phoneme boundaries, lyrics, scores, and prompts. To demonstrate the utility and versatility of MRSAudio, we establish five foundational tasks: audio spatialization, and spatial text to speech, spatial singing voice synthesis, spatial music generation and sound event localization and detection. Results show that MRSAudio enables high-quality spatial modeling and supports a broad range of spatial audio research. Demos and dataset access are available at https://mrsaudio.github.io.
>
---
#### [new 017] Perturbation Self-Supervised Representations for Cross-Lingual Emotion TTS: Stage-Wise Modeling of Emotion and Speaker
- **分类: cs.SD**

- **简介: 该论文研究跨语言情感语音合成，旨在解耦情感与音色。提出EMM-TTS框架，通过自监督表示扰动和两阶段建模实现情感迁移与音色保持，引入SCL和SEALN提升控制性与一致性。**

- **链接: [http://arxiv.org/pdf/2510.11124v1](http://arxiv.org/pdf/2510.11124v1)**

> **作者:** Cheng Gong; Chunyu Qiang; Tianrui Wang; Yu Jiang; Yuheng Lu; Ruihao Jing; Xiaoxiao Miao; Xiaolei Zhang; Longbiao Wang; Jianwu Dang
>
> **备注:** Submitted to Expert Systems with Applications,11 pages
>
> **摘要:** Cross-lingual emotional text-to-speech (TTS) aims to produce speech in one language that captures the emotion of a speaker from another language while maintaining the target voice's timbre. This process of cross-lingual emotional speech synthesis presents a complex challenge, necessitating flexible control over emotion, timbre, and language. However, emotion and timbre are highly entangled in speech signals, making fine-grained control challenging. To address this issue, we propose EMM-TTS, a novel two-stage cross-lingual emotional speech synthesis framework based on perturbed self-supervised learning (SSL) representations. In the first stage, the model explicitly and implicitly encodes prosodic cues to capture emotional expressiveness, while the second stage restores the timbre from perturbed SSL representations. We further investigate the effect of different speaker perturbation strategies-formant shifting and speaker anonymization-on the disentanglement of emotion and timbre. To strengthen speaker preservation and expressive control, we introduce Speaker Consistency Loss (SCL) and Speaker-Emotion Adaptive Layer Normalization (SEALN) modules. Additionally, we find that incorporating explicit acoustic features (e.g., F0, energy, and duration) alongside pretrained latent features improves voice cloning performance. Comprehensive multi-metric evaluations, including both subjective and objective measures, demonstrate that EMM-TTS achieves superior naturalness, emotion transferability, and timbre consistency across languages.
>
---
#### [new 018] ILD-VIT: A Unified Vision Transformer Architecture for Detection of Interstitial Lung Disease from Respiratory Sounds
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出ILD-VIT，一种基于视觉Transformer的模型，用于通过呼吸音检测间质性肺病。任务为医学音频分类，解决无创筛查难题。工作包括构建端到端框架，将呼吸音转为梅尔谱图并分块输入VIT，实现高效准确的疾病检测。**

- **链接: [http://arxiv.org/pdf/2510.11458v1](http://arxiv.org/pdf/2510.11458v1)**

> **作者:** Soubhagya Ranjan Hota; Arka Roy; Udit Satija
>
> **摘要:** Interstitial lung disease (ILD) represents a group of restrictive chronic pulmonary diseases that impair oxygen acquisition by causing irreversible changes in the lungs such as fibrosis, scarring of parenchyma, etc. ILD conditions are often diagnosed by various clinical modalities such as spirometry, high-resolution lung imaging techniques, crackling respiratory sounds (RSs), etc. In this letter, we develop a novel vision transformer (VIT)-based deep learning framework namely, ILD-VIT, to detect the ILD condition using the RS recordings. The proposed framework comprises three major stages: pre-processing, mel spectrogram extraction, and classification using the proposed VIT architecture using the mel spectrogram image patches. Experimental results using the publicly available BRACETS and KAUH databases show that our proposed ILD-VIT achieves an accuracy, sensitivity, and specificity of 84.86%, 82.67%, and 86.91%, respectively, for subject-independent blind testing. The successful onboard implantation of the proposed framework on a Raspberry-pi-4 microcontroller indicates its potential as a standalone clinical system for ILD screening in a real clinical scenario.
>
---
#### [new 019] Matchmaker: An Open-source Library for Real-time Piano Score Following and Systematic Evaluation
- **分类: cs.SD**

- **简介: 该论文针对实时乐谱对齐（score following）任务，解决缺乏统一开源框架和系统评估的问题。作者提出Matchmaker——一个兼容现代MIR工具的Python库，支持多种音乐表征与对齐方法的系统比较，并在大规模钢琴数据集上进行综合评估，建立可复现的基准框架。**

- **链接: [http://arxiv.org/pdf/2510.10087v1](http://arxiv.org/pdf/2510.10087v1)**

> **作者:** Jiyun Park; Carlos Cancino-Chacón; Suhit Chiruthapudi; Juhan Nam
>
> **备注:** In Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), 2025
>
> **摘要:** Real-time music alignment, also known as score following, is a fundamental MIR task with a long history and is essential for many interactive applications. Despite its importance, there has not been a unified open framework for comparing models, largely due to the inherent complexity of real-time processing and the language- or system-dependent implementations. In addition, low compatibility with the existing MIR environment has made it difficult to develop benchmarks using large datasets available in recent years. While new studies based on established methods (e.g., dynamic programming, probabilistic models) have emerged, most evaluations compare models only within the same family or on small sets of test data. This paper introduces Matchmaker, an open-source Python library for real-time music alignment that is easy to use and compatible with modern MIR libraries. Using this, we systematically compare methods along two dimensions: music representations and alignment methods. We evaluated our approach on a large test set of solo piano music from the (n)ASAP, Batik, and Vienna4x22 datasets with a comprehensive set of metrics to ensure robust assessment. Our work aims to establish a benchmark framework for score-following research while providing a practical tool that developers can easily integrate into their applications.
>
---
#### [new 020] A Machine Learning Approach for MIDI to Guitar Tablature Conversion
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究MIDI到吉他谱的自动转换任务，旨在解决音符到吉他弦-品位置的合理映射问题。作者提出基于机器学习的方法，结合指法跨度假设与数据增强技术，提升对可演奏性和连续性的建模效果。**

- **链接: [http://arxiv.org/pdf/2510.10619v1](http://arxiv.org/pdf/2510.10619v1)**

> **作者:** Maximos Kaliakatsos-Papakostas; Gregoris Bastas; Dimos Makris; Dorien Herremans; Vassilis Katsouros; Petros Maragos
>
> **备注:** Proceedings of the 19th Sound and Music Computing Conference, June 5-12th, 2022, Saint-\'Etienne (France)
>
> **摘要:** Guitar tablature transcription consists in deducing the string and the fret number on which each note should be played to reproduce the actual musical part. This assignment should lead to playable string-fret combinations throughout the entire track and, in general, preserve parsimonious motion between successive combinations. Throughout the history of guitar playing, specific chord fingerings have been developed across different musical styles that facilitate common idiomatic voicing combinations and motion between them. This paper presents a method for assigning guitar tablature notation to a given MIDI-based musical part (possibly consisting of multiple polyphonic tracks), i.e. no information about guitar-idiomatic expressional characteristics is involved (e.g. bending etc.) The current strategy is based on machine learning and requires a basic assumption about how much fingers can stretch on a fretboard; only standard 6-string guitar tuning is examined. The proposed method also examines the transcription of music pieces that was not meant to be played or could not possibly be played by a guitar (e.g. potentially a symphonic orchestra part), employing a rudimentary method for augmenting musical information and training/testing the system with artificial data. The results present interesting aspects about what the system can achieve when trained on the initial and augmented dataset, showing that the training with augmented data improves the performance even in simple, e.g. monophonic, cases. Results also indicate weaknesses and lead to useful conclusions about possible improvements.
>
---
#### [new 021] Proficiency-Aware Adaptation and Data Augmentation for Robust L2 ASR
- **分类: cs.SD; cs.AI; 68T07 (Primary), 94A12, 68T05 (Secondary); I.5.4; I.2.7**

- **简介: 该论文研究二语（L2）语音识别中的公平性问题，针对不同语言水平学习者识别性能差异大的问题，提出水平感知多任务学习与面向低水平语音的频谱掩蔽增强方法，在减少整体错误的同时缩小水平间性能差距。**

- **链接: [http://arxiv.org/pdf/2510.10738v1](http://arxiv.org/pdf/2510.10738v1)**

> **作者:** Ling Sun; Charlotte Zhu; Shuju Shi
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** General-purpose ASR underperforms for atypical speakers, such as L2 learners, reinforcing bias and limiting use in education and accessibility. Using the CEFR-graded Speak and Improve corpus, we show that naive fine-tuning of Whisper reduces average WER but simultaneously widens disparities and disproportionately harms lower-level learners. To address this, we propose two strategies: (i) proficiency-aware multitask learning, jointly optimizing ASR with proficiency classification, and (ii) targeted augmentation, applying spectrogram masking to low-proficiency speech to counter imbalance. These approaches reduce WER by up to 29.4 percent (relative) and insertion/deletion errors by as much as 58.6 percent (relative). Crucially, despite the severe imbalance of the dataset reflecting real-world distributions, both strategies consistently narrow proficiency gaps, advancing equitable ASR for L2 learners.
>
---
#### [new 022] BridgeCode: A Dual Speech Representation Paradigm for Autoregressive Zero-Shot Text-to-Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文研究零样本文本到语音合成，针对自回归模型存在的速度-质量权衡与监督不匹配问题，提出BridgeTTS框架，通过双语音表示范式BridgeCode，实现稀疏离散预测与连续特征重建，兼顾高效性与高质量合成。**

- **链接: [http://arxiv.org/pdf/2510.11646v1](http://arxiv.org/pdf/2510.11646v1)**

> **作者:** Jingyuan Xing; Mingru Yang; Zhipeng Li; Xiaofen Xing; Xiangmin Xu
>
> **摘要:** Autoregressive (AR) frameworks have recently achieved remarkable progress in zero-shot text-to-speech (TTS) by leveraging discrete speech tokens and large language model techniques. Despite their success, existing AR-based zero-shot TTS systems face two critical limitations: (i) an inherent speed-quality trade-off, as sequential token generation either reduces frame rates at the cost of expressiveness or enriches tokens at the cost of efficiency, and (ii) a text-oriented supervision mismatch, as cross-entropy loss penalizes token errors uniformly without considering the fine-grained acoustic similarity among adjacent tokens. To address these challenges, we propose BridgeTTS, a novel AR-TTS framework built upon the dual speech representation paradigm BridgeCode. BridgeTTS reduces AR iterations by predicting sparse tokens while reconstructing rich continuous features for high-quality synthesis. Joint optimization of token-level and feature-level objectives further enhances naturalness and intelligibility. Experiments demonstrate that BridgeTTS achieves competitive quality and speaker similarity while significantly accelerating synthesis. Speech demos are available at https://test1562.github.io/demo/.
>
---
#### [new 023] Audio-Maestro: Enhancing Large Audio-Language Models with Tool-Augmented Reasoning
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出Audio-Maestro框架，解决大音频语言模型在复杂任务中依赖端到端推理导致的准确性与可解释性不足问题。通过引入工具增强推理，使模型能调用外部工具并融合其时序输出，提升音频理解性能。**

- **链接: [http://arxiv.org/pdf/2510.11454v1](http://arxiv.org/pdf/2510.11454v1)**

> **作者:** Kuan-Yi Lee; Tsung-En Lin; Hung-Yi Lee
>
> **备注:** 9pages
>
> **摘要:** Recent advancements in large multimodal models (LMMs) have shown strong capabilities in audio understanding. However, most systems rely solely on end-to-end reasoning, limiting interpretability and accuracy for tasks that require structured knowledge or specialized signal analysis. In this work, we present Audio-Maestro -- a tool-augmented audio reasoning framework that enables audio-language models to autonomously call external tools and integrate their timestamped outputs into the reasoning process. This design allows the model to analyze, transform, and interpret audio signals through specialized tools rather than relying solely on end-to-end inference. Experiments show that Audio-Maestro consistently improves general audio reasoning performance: Gemini-2.5-flash's average accuracy on MMAU-Test rises from 67.4% to 72.1%, DeSTA-2.5 from 58.3% to 62.8%, and GPT-4o from 60.8% to 63.9%. To our knowledge, Audio-Maestro is the first framework to integrate structured tool output into the large audio language model reasoning process.
>
---
#### [new 024] VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents
- **分类: cs.SD; cs.CL**

- **简介: 该论文针对中文语音对话模型评估不足的问题，构建了基于真实人类语音的评测基准VCB Bench，从指令遵循、知识理解与鲁棒性三方面实现多维度评估，推动音频接地大语言模型的标准化评测。**

- **链接: [http://arxiv.org/pdf/2510.11098v1](http://arxiv.org/pdf/2510.11098v1)**

> **作者:** Jiliang Hu; Wenfu Wang; Zuchao Li; Chenxing Li; Yiyang Zhao; Hanzhao Li; Liqiang Zhang; Meng Yu; Dong Yu
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.
>
---
#### [new 025] Dual Data Scaling for Robust Two-Stage User-Defined Keyword Spotting
- **分类: cs.SD**

- **简介: 该论文研究用户自定义关键词检测任务，旨在提升模型对相似词的区分能力与鲁棒性。提出DS-KWS两阶段框架，结合CTC与QbyT方法，并引入双数据扩展策略，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.10740v1](http://arxiv.org/pdf/2510.10740v1)**

> **作者:** Zhiqi Ai; Han Cheng; Yuxin Wang; Shiyi Mu; Shugong Xu; Yongjin Zhou
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** In this paper, we propose DS-KWS, a two-stage framework for robust user-defined keyword spotting. It combines a CTC-based method with a streaming phoneme search module to locate candidate segments, followed by a QbyT-based method with a phoneme matcher module for verification at both the phoneme and utterance levels. To further improve performance, we introduce a dual data scaling strategy: (1) expanding the ASR corpus from 460 to 1,460 hours to strengthen the acoustic model; and (2) leveraging over 155k anchor classes to train the phoneme matcher, significantly enhancing the distinction of confusable words. Experiments on LibriPhrase show that DS-KWS significantly outperforms existing methods, achieving 6.13\% EER and 97.85\% AUC on the Hard subset. On Hey-Snips, it achieves zero-shot performance comparable to full-shot trained models, reaching 99.13\% recall at one false alarm per hour.
>
---
#### [new 026] LSZone: A Lightweight Spatial Information Modeling Architecture for Real-time In-car Multi-zone Speech Separation
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对车内多区域语音分离实时性差的问题，提出轻量级架构LSZone。通过设计空间信息提取压缩模块和轻量Conv-GRU跨带处理模块，在降低计算量的同时保持性能，实现实时高效语音分离。**

- **链接: [http://arxiv.org/pdf/2510.10687v1](http://arxiv.org/pdf/2510.10687v1)**

> **作者:** Jun Chen; Shichao Hu; Jiuxin Lin; Wenjie Li; Zihan Zhang; Xingchen Li; JinJiang Liu; Longshuai Xiao; Chao Weng; Lei Xie; Zhiyong Wu
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** In-car multi-zone speech separation, which captures voices from different speech zones, plays a crucial role in human-vehicle interaction. Although previous SpatialNet has achieved notable results, its high computational cost still hinders real-time applications in vehicles. To this end, this paper proposes LSZone, a lightweight spatial information modeling architecture for real-time in-car multi-zone speech separation. We design a spatial information extraction-compression (SpaIEC) module that combines Mel spectrogram and Interaural Phase Difference (IPD) to reduce computational burden while maintaining performance. Additionally, to efficiently model spatial information, we introduce an extremely lightweight Conv-GRU crossband-narrowband processing (CNP) module. Experimental results demonstrate that LSZone, with a complexity of 0.56G MACs and a real-time factor (RTF) of 0.37, delivers impressive performance in complex noise and multi-speaker scenarios.
>
---
#### [new 027] Unify Variables in Neural Scaling Laws for General Audio Representations via Embedding Effective Rank
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究通用音频表征的神经缩放律，旨在解决多因素影响下表征质量难以量化的问题。作者引入嵌入有效秩（RankMe）作为统一指标，实证验证其与性能的幂律关系，为音频基础模型的缩放提供理论指导。**

- **链接: [http://arxiv.org/pdf/2510.10948v1](http://arxiv.org/pdf/2510.10948v1)**

> **作者:** Xuyao Deng; Yanjie Sun; Yong Dou; Kele Xu
>
> **摘要:** Scaling laws have profoundly shaped our understanding of model performance in computer vision and natural language processing, yet their application to general audio representation learning remains underexplored. A key challenge lies in the multifactorial nature of general audio representation-representation quality is jointly influenced by variables such as audio length, embedding dimensionality, model depth, model architecture, data volume, etc., many of which are difficult to isolate or express analytically. In this work, we present a systematic study of scaling laws for general audio representations by utilizing embedding effective rank (RankMe) as a unifying metric that encapsulates the impact of diverse variables on representation quality. RankMe enables a label-free, information-theoretic quantification of audio embeddings, allowing us to examine scaling behaviors across a wide hyper-parameter space, including model size, training data volume, computational budget, architectural configurations, etc. Our empirical findings reveal a consistent power-law relationship between RankMe and representation quality, suggesting that embedding effective rank serves as a reliable proxy for assessing and predicting model performance in audio representation learning. This work not only validates the applicability of classical scaling principles to the general audio domain but also offers a theoretically grounded and empirically robust framework for guiding future model scaling strategies in audio foundation models.
>
---
#### [new 028] Bhasha-Rupantarika: Algorithm-Hardware Co-design approach for Multilingual Neural Machine Translation
- **分类: cs.AR; cs.CL; cs.RO; eess.AS**

- **简介: 该论文提出Bhasha-Rupantarika，面向资源受限场景的多语言神经机器翻译系统。通过算法-硬件协同设计，采用超低精度量化（如FP4），在FPGA上实现模型小型化与推理加速，提升吞吐量，支持印度语与国际语言互译。**

- **链接: [http://arxiv.org/pdf/2510.10676v1](http://arxiv.org/pdf/2510.10676v1)**

> **作者:** Mukul Lokhande; Tanushree Dewangan; Mohd Sharik Mansoori; Tejas Chaudhari; Akarsh J.; Damayanti Lokhande; Adam Teman; Santosh Kumar Vishvakarma
>
> **摘要:** This paper introduces Bhasha-Rupantarika, a light and efficient multilingual translation system tailored through algorithm-hardware codesign for resource-limited settings. The method investigates model deployment at sub-octet precision levels (FP8, INT8, INT4, and FP4), with experimental results indicating a 4.1x reduction in model size (FP4) and a 4.2x speedup in inference speed, which correlates with an increased throughput of 66 tokens/s (improvement by 4.8x). This underscores the importance of ultra-low precision quantization for real-time deployment in IoT devices using FPGA accelerators, achieving performance on par with expectations. Our evaluation covers bidirectional translation between Indian and international languages, showcasing its adaptability in low-resource linguistic contexts. The FPGA deployment demonstrated a 1.96x reduction in LUTs and a 1.65x decrease in FFs, resulting in a 2.2x enhancement in throughput compared to OPU and a 4.6x enhancement compared to HPTA. Overall, the evaluation provides a viable solution based on quantisation-aware translation along with hardware efficiency suitable for deployable multilingual AI systems. The entire codes [https://github.com/mukullokhande99/Bhasha-Rupantarika/] and dataset for reproducibility are publicly available, facilitating rapid integration and further development by researchers.
>
---
#### [new 029] Delayed 1T to 2H Phase Transition Upon Electrochemical Delithiation of LiMoS2
- **分类: cond-mat.mtrl-sci; eess.AS**

- **简介: 该论文研究MoS₂脱锂后的相变行为，解决1T相在脱锂后是否稳定的问题。通过单片电化学实验与拉曼光谱，发现脱锂后1T相可长期存在，随后缓慢转为2H相，证实可电化学制备亚稳1T相。**

- **链接: [http://arxiv.org/pdf/2510.10911v1](http://arxiv.org/pdf/2510.10911v1)**

> **作者:** Yerin Hong; Juhwan Lim; Jinhong Min; Nishkarsh Agarwal; Robert Hovden; Ageeth A. Bol; Yiyang Li
>
> **摘要:** Molybdenum disulfide (MoS2) is a widely studied layered material for electronic, optical, and catalytic applications. It can host lithium ions between the van der Waals layers, which triggers a phase transition between the semiconducting 2H phase and metallic 1T phase. While lithium insertion triggers a phase transition to the 1T phase, the phase behavior upon electrochemical lithium removal is not resolved. In this work, we conduct single-flake electrochemical (de)lithiation of MoS2 using microelectrode arrays. Through both electrochemical voltage analysis and correlative Raman spectroscopy, we show that an electrochemically cycled and delithiated MoS2 flake initially remains in the 1T phase. However, over the course of several days, it transitions back into the thermodynamically stable 2H phase. This result resolves the phase transformation pathway upon delithiation and showcases the ability to electrochemically synthesize the metastable 1T-MoS2 phase.
>
---
#### [new 030] Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文研究复数卷积神经网络（CVCNN）在音频信号处理中的应用，旨在保留并利用相位信息。通过理论构建与训练优化，结合MFCC与图神经网络实验，验证了相位作为有效特征的潜力，提升了音频分类性能。**

- **链接: [http://arxiv.org/pdf/2510.09926v1](http://arxiv.org/pdf/2510.09926v1)**

> **作者:** Naman Agrawal
>
> **摘要:** This study explores the design and application of Complex-Valued Convolutional Neural Networks (CVCNNs) in audio signal processing, with a focus on preserving and utilizing phase information often neglected in real-valued networks. We begin by presenting the foundational theoretical concepts of CVCNNs, including complex convolutions, pooling layers, Wirtinger-based differentiation, and various complex-valued activation functions. These are complemented by critical adaptations of training techniques, including complex batch normalization and weight initialization schemes, to ensure stability in training dynamics. Empirical evaluations are conducted across three stages. First, CVCNNs are benchmarked on standard image datasets, where they demonstrate competitive performance with real-valued CNNs, even under synthetic complex perturbations. Although our focus is audio signal processing, we first evaluate CVCNNs on image datasets to establish baseline performance and validate training stability before applying them to audio tasks. In the second experiment, we focus on audio classification using Mel-Frequency Cepstral Coefficients (MFCCs). CVCNNs trained on real-valued MFCCs slightly outperform real CNNs, while preserving phase in input workflows highlights challenges in exploiting phase without architectural modifications. Finally, a third experiment introduces GNNs to model phase information via edge weighting, where the inclusion of phase yields measurable gains in both binary and multi-class genre classification. These results underscore the expressive capacity of complex-valued architectures and confirm phase as a meaningful and exploitable feature in audio processing applications. While current methods show promise, especially with activations like cardioid, future advances in phase-aware design will be essential to leverage the potential of complex representations in neural networks.
>
---
#### [new 031] Efficient Edge Test-Time Adaptation via Latent Feature Coordinate Correction
- **分类: cs.LG; eess.AS; eess.IV**

- **简介: 该论文研究边缘设备上的测试时自适应（TTA）任务，旨在解决资源受限和分布偏移问题。提出TED方法，基于CMA-ES在潜空间主成分方向进行前向优化，无需反向传播，实现高效、低开销的单实例自适应。**

- **链接: [http://arxiv.org/pdf/2510.11068v1](http://arxiv.org/pdf/2510.11068v1)**

> **作者:** Xinyu Luo; Jie Liu; Kecheng Chen; Junyi Yang; Bo Ding; Arindam Basu; Haoliang Li
>
> **备注:** Under review
>
> **摘要:** Edge devices face significant challenges due to limited computational resources and distribution shifts, making efficient and adaptable machine learning essential. Existing test-time adaptation (TTA) methods often rely on gradient-based optimization or batch processing, which are inherently unsuitable for resource-constrained edge scenarios due to their reliance on backpropagation and high computational demands. Gradient-free alternatives address these issues but often suffer from limited learning capacity, lack flexibility, or impose architectural constraints. To overcome these limitations, we propose a novel single-instance TTA method tailored for edge devices (TED), which employs forward-only coordinate optimization in the principal subspace of latent using the covariance matrix adaptation evolution strategy (CMA-ES). By updating a compact low-dimensional vector, TED not only enhances output confidence but also aligns the latent representation closer to the source latent distribution within the latent principal subspace. This is achieved without backpropagation, keeping the model parameters frozen, and enabling efficient, forgetting-free adaptation with minimal memory and computational overhead. Experiments on image classification and keyword spotting tasks across the ImageNet and Google Speech Commands series datasets demonstrate that TED achieves state-of-the-art performance while $\textit{reducing computational complexity by up to 63 times}$, offering a practical and scalable solution for real-world edge applications. Furthermore, we successfully $\textit{deployed TED on the ZYNQ-7020 platform}$, demonstrating its feasibility and effectiveness for resource-constrained edge devices in real-world deployments.
>
---
#### [new 032] MTP-S2UT: Enhancing Speech-to-Speech Translation Quality with Multi-token Prediction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究语音到语音翻译（S2UT），针对单个语音标记语义不足问题，提出多标记预测（MTP-S2UT）损失，在中间层增强语义密度。实验表明该方法有效提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2510.10003v1](http://arxiv.org/pdf/2510.10003v1)**

> **作者:** Jianjin Wang; Runsong Zhao; Xiaoqian Liu; Yuan Ge; Ziqiang Xu; Tong Xiao; Shengxiang Gao; Zhengtao Yu; Jingbo Zhu
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Current direct speech-to-speech translation methods predominantly employ speech tokens as intermediate representations. However, a single speech token is not dense in semantics, so we generally need multiple tokens to express a complete semantic unit. To address this limitation, we introduce multi-token prediction (MTP) loss into speech-to-unit translation (S2UT) models, enabling models to predict multiple subsequent tokens at each position, thereby capturing more complete semantics and enhancing information density per position. Initial MTP implementations apply the loss at the final layer, which improves output representation but initiates information enrichment too late. We hypothesize that advancing the information enrichment process to intermediate layers can achieve earlier and more effective enhancement of hidden representation. Consequently, we propose MTP-S2UT loss, applying MTP loss to hidden representation where CTC loss is computed. Experiments demonstrate that all MTP loss variants consistently improve the quality of S2UT translation, with MTP-S2UT achieving the best performance.
>
---
#### [new 033] Chord Colourizer: A Near Real-Time System for Visualizing Musical Key
- **分类: cs.HC; cs.CY; cs.SD; eess.AS**

- **简介: 该论文提出Chord Colourizer系统，属音乐信息检索任务，旨在实时可视化音频的调性。通过CQT特征与牛顿色轮映射，结合GUI和LED多模态输出，实现音符到色彩的动态呈现，提升音乐教育与表演体验。**

- **链接: [http://arxiv.org/pdf/2510.10173v1](http://arxiv.org/pdf/2510.10173v1)**

> **作者:** Paul Haimes
>
> **备注:** Author copy. This paper is in press for presentation at ADADA 2025. Please cite as: Haimes, P. (in press). Chord Colourizer: A near real-time system for visualizing musical key. In Proceedings of the 23rd International Conference of Asia Digital Art and Design Association (ADADA)
>
> **摘要:** This paper introduces Chord Colourizer, a near real-time system that detects the musical key of an audio signal and visually represents it through a novel graphical user interface (GUI). The system assigns colours to musical notes based on Isaac Newton's original colour wheel, preserving historical links between pitch and hue, and also integrates an Arduino-controlled LED display using 3D-printed star-shaped diffusers to offer a physical ambient media representation. The method employs Constant-Q Transform (CQT) chroma features for chord estimation and visualization, followed by threshold-based filtering and tonal enhancement to isolate the root, third, and fifth. A confidence score is computed for each detection to ensure reliability, and only chords with moderate to very strong certainty are visualized. The graphical interface dynamically updates a colour-coded keyboard layout, while the LED display provides the same colour information via spatial feedback. This multi-modal system enhances user interaction with harmonic content, offering innovative possibilities for education and artistic performance. Limitations include slight latency and the inability to detect extended chords, which future development will aim to address through refined filtering, adaptive thresholds, and support for more complex harmonies such as sevenths and augmented chords. Future work will also explore integration with alternative visualization styles, and the comparison of audio analysis libraries to improve detection speed and precision. Plans also include formal user testing to evaluate perception, usability, and cross-cultural interpretations of colour-pitch mappings.
>
---
## 更新

#### [replaced 001] Modeling nonuniform energy decay through the modal decomposition of acoustic radiance transfer (MoD-ART)
- **分类: cs.SD; cs.SY; eess.AS; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.04534v3](http://arxiv.org/pdf/2412.04534v3)**

> **作者:** Matteo Scerbo; Sebastian J. Schlecht; Randall Ali; Lauri Savioja; Enzo De Sena
>
> **摘要:** Modeling late reverberation in real-time interactive applications is a challenging task when multiple sound sources and listeners are present in the same environment. This is especially problematic when the environment is geometrically complex and/or features uneven energy absorption (e.g. coupled volumes), because in such cases the late reverberation is dependent on the sound sources' and listeners' positions, and therefore must be adapted to their movements in real time. We present a novel approach to the task, named modal decomposition of acoustic radiance transfer (MoD-ART), which can handle highly complex scenarios with efficiency. The approach is based on the geometrical acoustics method of acoustic radiance transfer, from which we extract a set of energy decay modes and their positional relationships with sources and listeners. In this paper, we describe the physical and mathematical significance of MoD-ART, highlighting its advantages and applicability to different scenarios. Through an analysis of the method's computational complexity, we show that it compares very favorably with ray-tracing. We also present simulation results showing that MoD-ART can capture multiple decay slopes and flutter echoes.
>
---
#### [replaced 002] SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2510.02797v2](http://arxiv.org/pdf/2510.02797v2)**

> **作者:** Chunbo Hao; Ruibin Yuan; Jixun Yao; Qixin Deng; Xinyi Bai; Wei Xue; Lei Xie
>
> **摘要:** Music structure analysis (MSA) underpins music understanding and controllable generation, yet progress has been limited by small, inconsistent corpora. We present SongFormer, a scalable framework that learns from heterogeneous supervision. SongFormer (i) fuses short- and long-window self-supervised audio representations to capture both fine-grained and long-range dependencies, and (ii) introduces a learned source embedding to enable training with partial, noisy, and schema-mismatched labels. To support scaling and fair evaluation, we release SongFormDB, the largest MSA corpus to date (over 10k tracks spanning languages and genres), and SongFormBench, a 300-song expert-verified benchmark. On SongFormBench, SongFormer sets a new state of the art in strict boundary detection (HR.5F) and achieves the highest functional label accuracy, while remaining computationally efficient; it surpasses strong baselines and Gemini 2.5 Pro on these metrics and remains competitive under relaxed tolerance (HR3F). Code, datasets, and model are publicly available.
>
---
#### [replaced 003] WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.23238v2](http://arxiv.org/pdf/2509.23238v2)**

> **作者:** Goksenin Yuksel; Pierre Guetschel; Michael Tangermann; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.
>
---
#### [replaced 004] Speech Enhancement and Dereverberation with Diffusion-based Generative Models
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2208.05830v3](http://arxiv.org/pdf/2208.05830v3)**

> **作者:** Julius Richter; Simon Welker; Jean-Marie Lemercier; Bunlong Lay; Timo Gerkmann
>
> **备注:** Proofread version
>
> **摘要:** In this work, we build upon our previous publication and use diffusion-based generative models for speech enhancement. We present a detailed overview of the diffusion process that is based on a stochastic differential equation and delve into an extensive theoretical examination of its implications. Opposed to usual conditional generation tasks, we do not start the reverse process from pure Gaussian noise but from a mixture of noisy speech and Gaussian noise. This matches our forward process which moves from clean speech to noisy speech by including a drift term. We show that this procedure enables using only 30 diffusion steps to generate high-quality clean speech estimates. By adapting the network architecture, we are able to significantly improve the speech enhancement performance, indicating that the network, rather than the formalism, was the main limitation of our original approach. In an extensive cross-dataset evaluation, we show that the improved method can compete with recent discriminative models and achieves better generalization when evaluating on a different corpus than used for training. We complement the results with an instrumental evaluation using real-world noisy recordings and a listening experiment, in which our proposed method is rated best. Examining different sampler configurations for solving the reverse process allows us to balance the performance and computational speed of the proposed method. Moreover, we show that the proposed method is also suitable for dereverberation and thus not limited to additive background noise removal. Code and audio examples are available online, see https://github.com/sp-uhh/sgmse.
>
---
#### [replaced 005] $\texttt{AVROBUSTBENCH}$: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00358v2](http://arxiv.org/pdf/2506.00358v2)**

> **作者:** Sarthak Kumar Maharana; Saksham Singh Kushwaha; Baoming Zhang; Adrian Rodriguez; Songtao Wei; Yapeng Tian; Yunhui Guo
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks
>
> **摘要:** While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur $\textit{simultaneously}$ in both audio and visual modalities, we introduce $\texttt{AVROBUSTBENCH}$, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models. $\texttt{AVROBUSTBENCH}$ comprises four audio-visual benchmark datasets, $\texttt{AUDIOSET-2C}$, $\texttt{VGGSOUND-2C}$, $\texttt{KINETICS-2C}$, and $\texttt{EPICKITCHENS-2C}$, each incorporating 75 bimodal audio-visual corruptions that are $\textit{co-occurring}$ and $\textit{correlated}$. Through extensive evaluations, we observe that state-of-the-art supervised and self-supervised audio-visual models exhibit declining robustness as corruption severity increases. Furthermore, online test-time adaptation (TTA) methods, on $\texttt{VGGSOUND-2C}$ and $\texttt{KINETICS-2C}$, offer minimal improvements in performance under bimodal corruptions. We further propose $\texttt{AV2C}$, a simple TTA approach enabling on-the-fly cross-modal fusion by penalizing high-entropy samples, which achieves improvements on $\texttt{VGGSOUND-2C}$. We hope that $\texttt{AVROBUSTBENCH}$ will steer the development of more effective and robust audio-visual TTA approaches. Our code is available $\href{https://github.com/sarthaxxxxx/AV-C-Robustness-Benchmark}{here}$.
>
---
#### [replaced 006] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.23305v2](http://arxiv.org/pdf/2505.23305v2)**

> **作者:** Yunkee Chae; Kyogu Lee
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present MGE-LDM, a unified latent diffusion framework for simultaneous music generation, source imputation, and query-driven source separation. Unlike prior approaches constrained to fixed instrument classes, MGE-LDM learns a joint distribution over full mixtures, submixtures, and individual stems within a single compact latent diffusion model. At inference, MGE-LDM enables (1) complete mixture generation, (2) partial generation (i.e., source imputation), and (3) text-conditioned extraction of arbitrary sources. By formulating both separation and imputation as conditional inpainting tasks in the latent space, our approach supports flexible, class-agnostic manipulation of arbitrary instrument sources. Notably, MGE-LDM can be trained jointly across heterogeneous multi-track datasets (e.g., Slakh2100, MUSDB18, MoisesDB) without relying on predefined instrument categories. Audio samples are available at our project page: https://yoongi43.github.io/MGELDM_Samples/.
>
---
#### [replaced 007] Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval
- **分类: cs.IR; cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.04273v2](http://arxiv.org/pdf/2508.04273v2)**

> **作者:** Junan Lin; Daizong Liu; Xianke Chen; Xiaoye Qu; Xun Yang; Jixiang Zhu; Sanyuan Zhang; Jianfeng Dong
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Video Moment Retrieval (VMR) aims to retrieve a specific moment semantically related to the given query. To tackle this task, most existing VMR methods solely focus on the visual and textual modalities while neglecting the complementary but important audio modality. Although a few recent works try to tackle the joint audio-vision-text reasoning, they treat all modalities equally and simply embed them without fine-grained interaction for moment retrieval. These designs are counter-practical as: Not all audios are helpful for video moment retrieval, and the audio of some videos may be complete noise or background sound that is meaningless to the moment determination. To this end, we propose a novel Importance-aware Multi-Granularity fusion model (IMG), which learns to dynamically and selectively aggregate the audio-vision-text contexts for VMR. Specifically, after integrating the textual guidance with vision and audio separately, we first design a pseudo-label-supervised audio importance predictor that predicts the importance score of the audio, and accordingly assigns weights to mitigate the interference caused by noisy audio. Then, we design a multi-granularity audio fusion module that adaptively fuses audio and visual modalities at local-, event-, and global-level, fully capturing their complementary contexts. We further propose a cross-modal knowledge distillation strategy to address the challenge of missing audio modality during inference. To evaluate our method, we further construct a new VMR dataset, i.e., Charades-AudioMatter, where audio-related samples are manually selected and re-organized from the original Charades-STA to validate the model's capability in utilizing audio modality. Extensive experiments validate the effectiveness of our method, achieving state-of-the-art with audio-video fusion in VMR methods. Our code is available at https://github.com/HuiGuanLab/IMG.
>
---
#### [replaced 008] Enhancing Noise Robustness for Neural Speech Codecs through Resource-Efficient Progressive Quantization Perturbation Simulation
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2509.19025v2](http://arxiv.org/pdf/2509.19025v2)**

> **作者:** Rui-Chen Zheng; Yang Ai; Hui-Peng Du; Li-Rong Dai
>
> **摘要:** Noise robustness remains a critical challenge for deploying neural speech codecs in real-world acoustic scenarios where background noise is often inevitable. A key observation we make is that even slight input noise perturbations can cause unintended shifts in quantized codewords, thereby degrading the quality of reconstructed speech. Motivated by this finding, we propose a novel and resource-efficient training strategy to enhance the noise robustness of speech codecs by simulating such perturbations directly at the quantization level. Our approach introduces two core mechanisms: (1) a distance-weighted probabilistic top-K sampling strategy that replaces the conventional deterministic nearest-neighbor selection in residual vector quantization (RVQ); and (2) a progressive training scheme that introduces perturbations from the last to the first quantizer in a controlled manner. Crucially, our method is trained exclusively on clean speech, eliminating the need for any paired noisy-clean data. Experiments on two advanced neural speech codecs, Encodec and WavTokenizer, demonstrate that the proposed strategy substantially improves robustness under noisy conditions-for example, boosting UTMOS from 3.475 to 3.586 at 15 dB SNR on Encodec-while also enhancing coding quality for clean speech.
>
---
#### [replaced 009] Discrete-Time Diffusion-Like Models for Speech Synthesis
- **分类: cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.18470v2](http://arxiv.org/pdf/2509.18470v2)**

> **作者:** Xiaozhou Tan; Minghui Zhao; Anton Ragni
>
> **摘要:** Diffusion models have attracted a lot of attention in recent years. These models view speech generation as a continuous-time process. For efficient training, this process is typically restricted to additive Gaussian noising, which is limiting. For inference, the time is typically discretized, leading to the mismatch between continuous training and discrete sampling conditions. Recently proposed discrete-time processes, on the other hand, usually do not have these limitations, may require substantially fewer inference steps, and are fully consistent between training/inference conditions. This paper explores some diffusion-like discrete-time processes and proposes some new variants. These include processes applying additive Gaussian noise, multiplicative Gaussian noise, blurring noise and a mixture of blurring and Gaussian noises. The experimental results suggest that discrete-time processes offer comparable subjective and objective speech quality to their widely popular continuous counterpart, with more efficient and consistent training and inference schemas.
>
---
#### [replaced 010] PicoAudio2: Temporal Controllable Text-to-Audio Generation with Natural Language Description
- **分类: cs.SD; eess.AS; 68Txx; I.2**

- **链接: [http://arxiv.org/pdf/2509.00683v2](http://arxiv.org/pdf/2509.00683v2)**

> **作者:** Zihao Zheng; Zeyu Xie; Xuenan Xu; Wen Wu; Chao Zhang; Mengyue Wu
>
> **备注:** Demo page: https://HiRookie9.github.io/PicoAudio2-Page
>
> **摘要:** While recent work in controllable text-to-audio (TTA) generation has achieved fine-grained control through timestamp conditioning, its scope remains limited by audio quality and input format. These models often suffer from poor audio quality in real datasets due to sole reliance on synthetic data. Moreover, some models are constrained to a closed vocabulary of sound events, preventing them from controlling audio generation for open-ended, free-text queries. This paper introduces PicoAudio2, a framework that advances temporal-controllable TTA by mitigating these data and architectural limitations. Specifically, we use a grounding model to annotate event timestamps of real audio-text datasets to curate temporally-strong real data, in addition to simulation data from existing works. The model is trained on the combination of real and simulation data. Moreover, we propose an enhanced architecture that integrates the fine-grained information from a timestamp matrix with coarse-grained free-text input. Experiments show that PicoAudio2 exhibits superior performance in terms of temporal controllability and audio quality.
>
---
#### [replaced 011] GRAM: Spatial general-purpose audio representation models for real-world applications
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00934v3](http://arxiv.org/pdf/2506.00934v3)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Although audio foundations models have seen great progress on a wide variety of tasks, their application in real-world acoustic environments with reverberation and noise has been less successful. Moreover, as audio foundation models are typically trained on dry, single-channel audio clips, the inherent spatial nature of real-world sound scenes is overlooked and tasks involving sound localization ruled out. To address these limitations, we propose GRAM: a General-purpose Real-world Audio Model utilizing a multi-channel masked auto-encoder approach to efficiently learn spatial audio representations from high-quality simulated real-world scenes. To evaluate the performance of GRAM and other audio foundation models in real-world sound scenes, we release Nat-HEAR: A naturalistic version of the HEAR benchmark suite comprising a simulated real-world version, as well as two new sound localization tasks. We show that the performance of GRAM surpasses all state-of-the-art self-supervised audio foundation models and speech models on both HEAR and Nat-HEAR, while using only a fraction of the training data. GRAM also showcases state-of-the-art localization performance, surpassing even supervised sound localization approaches, and can be flexibly applied either to a two-channel, binaural sound format or a four-channel, Ambisonics format. Validating GRAM's performance on real-world sound recordings demonstrates robust transfer to real-world scenes. Taken together, GRAM presents a significant advancement towards robust, spatial audio foundation models for real-world applications.
>
---
#### [replaced 012] Data Standards in Audiology: A Mixed-Methods Exploration of Community Perspectives and Implementation Considerations
- **分类: cs.SD; eess.AS; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2505.04728v3](http://arxiv.org/pdf/2505.04728v3)**

> **作者:** Charlotte Vercammen; Antje Heinrich; Christophe Lesimple; Alessia Paglialonga; Jan-Willem A. Wasmann; Mareike Buhl
>
> **摘要:** Objective: This study addresses conceptual issues around data standardisation in audiology, and outlines steps toward achieving it. It reports a survey of the computational audiology community on their current understanding, needs, and preferences concerning data standards. Based on survey findings and a panel discussion, recommendations are made concerning moving forward with standardisation in audiology. Design: Mixed-methods: 1) review of existing standardisation efforts; 2) a survey of the computational audiology community; 3) expert panel discussion in a dedicated session at the 2024 Virtual Conference of Computational Audiology. Sample: Survey: 82 members of the global community; Panel discussion: five experts. Results: A prerequisite for any global audiology database are agreed data standards. Although many are familiar with the general idea, few know of existing initiatives, or have actively participated in them. Ninety percent of respondents expressed willingness to follow or contribute to standardisation efforts. The panel discussed relevant initiatives (e.g. OMOP, openEHR, NOAH) and explored both challenges (around harmonisation) and opportunities (alignment with other medical fields and conversion among approaches). Conclusions: Combining conceptual discussion with stakeholder views, the study offers guidance for implementing interoperable data standards in audiology. It highlights community support, key issues to address, and suggests paths for future work.
>
---
#### [replaced 013] Joint Source-Environment Adaptation of Data-Driven Underwater Acoustic Source Ranging Based on Model Uncertainty
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.23258v2](http://arxiv.org/pdf/2503.23258v2)**

> **作者:** Dariush Kari; Hari Vishnu; Andrew C. Singer
>
> **摘要:** Adapting pre-trained deep learning models to new and unknown environments remains a major challenge in underwater acoustic localization. We show that although the performance of pre-trained models suffers from mismatch between the training and test data, they generally exhibit a higher uncertainty in environments where there is more mismatch. Additionally, in the presence of environmental mismatch, spurious peaks can appear in the output of classification-based localization approaches, which inspires us to define and use a method to quantify the "implied uncertainty" based on the number of model output peaks. Leveraging this notion of implied uncertainty, we partition the test samples into sets with more certain and less certain samples, and implement a method to adapt the model to new environments by using the certain samples to improve the labeling for uncertain samples, which helps to adapt the model. Thus, using this efficient method for model uncertainty quantification, we showcase an innovative approach to adapt a pre-trained model to unseen underwater environments at test time. This eliminates the need for labeled data from the target environment or the original training data. This adaptation is enhanced by integrating an independent estimate based on the received signal energy. We validate the approach extensively using real experimental data, as well as synthetic data consisting of model-generated signals with real ocean noise. The results demonstrate significant improvements in model prediction accuracy, underscoring the potential of the method to enhance underwater acoustic localization in diverse, noisy, and unknown environments.
>
---
#### [replaced 014] Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.08078v2](http://arxiv.org/pdf/2510.08078v2)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **摘要:** Video-to-Audio generation has made remarkable strides in automatically synthesizing sound for video. However, existing evaluation metrics, which focus on semantic and temporal alignment, overlook a critical failure mode: models often generate acoustic events, particularly speech and music, that have no corresponding visual source. We term this phenomenon Insertion Hallucination and identify it as a systemic risk driven by dataset biases, such as the prevalence of off-screen sounds, that remains completely undetected by current metrics. To address this challenge, we first develop a systematic evaluation framework that employs a majority-voting ensemble of multiple audio event detectors. We also introduce two novel metrics to quantify the prevalence and severity of this issue: IH@vid (the fraction of videos with hallucinations) and IH@dur (the fraction of hallucinated duration). Building on this, we propose Posterior Feature Correction, a novel training-free inference-time method that mitigates IH. PFC operates in a two-pass process: it first generates an initial audio output to detect hallucinated segments, and then regenerates the audio after masking the corresponding video features at those timestamps. Experiments on several mainstream V2A benchmarks first reveal that state-of-the-art models suffer from severe IH. In contrast, our PFC method reduces both the prevalence and duration of hallucinations by over 50\% on average, without degrading, and in some cases even improving, conventional metrics for audio quality and temporal synchronization. Our work is the first to formally define, systematically measure, and effectively mitigate Insertion Hallucination, paving the way for more reliable and faithful V2A models.
>
---
#### [replaced 015] Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning
- **分类: cs.AI; cs.CV; cs.MM; cs.SD; eess.AS; 68; I.2.10**

- **链接: [http://arxiv.org/pdf/2508.01181v2](http://arxiv.org/pdf/2508.01181v2)**

> **作者:** Zhiyuan Han; Beier Zhu; Yanlong Xu; Peipei Song; Xun Yang
>
> **备注:** ACM Multimedia 2025 Oral Code: https://github.com/ZhiyuanHan-Aaron/MoSEAR Project Page: https://zhiyuanhan-aaron.github.io/MoSEAR-page/
>
> **摘要:** Despite their strong performance in multimodal emotion reasoning, existing Multimodal Large Language Models (MLLMs) often overlook the scenarios involving emotion conflicts, where emotional cues from different modalities are inconsistent. To fill this gap, we first introduce CA-MER, a new benchmark designed to examine MLLMs under realistic emotion conflicts. It consists of three subsets: video-aligned, audio-aligned, and consistent, where only one or all modalities reflect the true emotion. However, evaluations on our CA-MER reveal that current state-of-the-art emotion MLLMs systematically over-rely on audio signal during emotion conflicts, neglecting critical cues from visual modality. To mitigate this bias, we propose MoSEAR, a parameter-efficient framework that promotes balanced modality integration. MoSEAR consists of two modules: (1)MoSE, modality-specific experts with a regularized gating mechanism that reduces modality bias in the fine-tuning heads; and (2)AR, an attention reallocation mechanism that rebalances modality contributions in frozen backbones during inference. Our framework offers two key advantages: it mitigates emotion conflicts and improves performance on consistent samples-without incurring a trade-off between audio and visual modalities. Experiments on multiple benchmarks-including MER2023, EMER, DFEW, and our CA-MER-demonstrate that MoSEAR achieves state-of-the-art performance, particularly under modality conflict conditions.
>
---
#### [replaced 016] Enhancing Speaker Verification with w2v-BERT 2.0 and Knowledge Distillation guided Structured Pruning
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.04213v2](http://arxiv.org/pdf/2510.04213v2)**

> **作者:** Ze Li; Ming Cheng; Ming Li
>
> **摘要:** Large-scale self-supervised Pre-Trained Models (PTMs) have shown significant improvements in the speaker verification (SV) task by providing rich feature representations. In this paper, we utilize w2v-BERT 2.0, a model with approximately 600 million parameters trained on 4.5 million hours of unlabeled data across 143 languages, for the SV task. The MFA structure with Layer Adapter is employed to process the multi-layer feature outputs from the PTM and extract speaker embeddings. Additionally, we incorporate LoRA for efficient fine-tuning. Our model achieves state-of-the-art results with 0.12% and 0.55% EER on the Vox1-O and Vox1-H test sets, respectively. Furthermore, we apply knowledge distillation guided structured pruning, reducing the model size by 80% while achieving only a 0.04% EER degradation. Source code and models are released at https://github.com/ZXHY-82/w2v-BERT-2.0_SV.
>
---
#### [replaced 017] Stimulus Modality Matters: Impact of Perceptual Evaluations from Different Modalities on Speech Emotion Recognition System Performance
- **分类: eess.AS; cs.MM; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.10762v3](http://arxiv.org/pdf/2409.10762v3)**

> **作者:** Huang-Cheng Chou; Haibin Wu; Chi-Chun Lee
>
> **备注:** 5 pages, 2 figures, 4 tables, acceptance for ICASSP 2025
>
> **摘要:** Speech Emotion Recognition (SER) systems rely on speech input and emotional labels annotated by humans. However, various emotion databases collect perceptional evaluations in different ways. For instance, the IEMOCAP dataset uses video clips with sounds for annotators to provide their emotional perceptions. However, the most significant English emotion dataset, the MSP-PODCAST, only provides speech for raters to choose the emotional ratings. Nevertheless, using speech as input is the standard approach to training SER systems. Therefore, the open question is the emotional labels elicited by which scenarios are the most effective for training SER systems. We comprehensively compare the effectiveness of SER systems trained with labels elicited by different modality stimuli and evaluate the SER systems on various testing conditions. Also, we introduce an all-inclusive label that combines all labels elicited by various modalities. We show that using labels elicited by voice-only stimuli for training yields better performance on the test set, whereas labels elicited by voice-only stimuli.
>
---
