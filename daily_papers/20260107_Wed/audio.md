# 音频 cs.SD;  eess.AS

- **最新发布 20 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频-语言模型任务，旨在解决现有模型忽略空间信息的问题。通过构建双耳音频数据集、设计混合特征投影器和渐进式训练方法，提升模型的空间理解能力。**

- **链接: [https://arxiv.org/pdf/2601.02954v1](https://arxiv.org/pdf/2601.02954v1)**

> **作者:** Yuhuan You; Lai Wei; Xihong Wu; Tianshu Qu
>
> **摘要:** Existing large audio-language models perceive the world as "mono" -- a single stream of audio that ignores the critical spatial dimension ("where") required for universal acoustic scene analysis. To bridge this gap, we first introduce a hierarchical framework for Auditory Scene Analysis (ASA). Guided by this framework, we introduce a system that enables models like Qwen2-Audio to understand and reason about the complex acoustic world. Our framework achieves this through three core contributions: First, we build a large-scale, synthesized binaural audio dataset to provide the rich spatial cues. Second, we design a hybrid feature projector, which leverages parallel semantic and spatial encoders to extract decoupled representations. These distinct streams are integrated via a dense fusion mechanism, ensuring the model receives a holistic view of the acoustic scene. Finally, we employ a progressive training curriculum, advancing from supervised fine-tuning (SFT) to reinforcement learning via Group Relative Policy Optimization (GRPO), to explicitly evolve the model's capabilities towards reasoning. On our comprehensive benchmark, the model demonstrates comparatively strong capability for spatial understanding. By enabling this spatial perception, our work provides a clear pathway for leveraging the powerful reasoning abilities of large models towards holistic acoustic scene analysis, advancing from "mono" semantic recognition to spatial intelligence.
>
---
#### [new 002] VocalBridge: Latent Diffusion-Bridge Purification for Defeating Perturbation-Based Voiceprint Defenses
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于语音安全任务，旨在解决扰动防御被净化技术突破的问题。提出VocalBridge框架，通过扩散桥接实现高效净化，提升语音克隆攻击效果。**

- **链接: [https://arxiv.org/pdf/2601.02444v1](https://arxiv.org/pdf/2601.02444v1)**

> **作者:** Maryam Abbasihafshejani; AHM Nazmus Sakib; Murtuza Jadliwala
>
> **摘要:** The rapid advancement of speech synthesis technologies, including text-to-speech (TTS) and voice conversion (VC), has intensified security and privacy concerns related to voice cloning. Recent defenses attempt to prevent unauthorized cloning by embedding protective perturbations into speech to obscure speaker identity while maintaining intelligibility. However, adversaries can apply advanced purification techniques to remove these perturbations, recover authentic acoustic characteristics, and regenerate cloneable voices. Despite the growing realism of such attacks, the robustness of existing defenses under adaptive purification remains insufficiently studied. Most existing purification methods are designed to counter adversarial noise in automatic speech recognition (ASR) systems rather than speaker verification or voice cloning pipelines. As a result, they fail to suppress the fine-grained acoustic cues that define speaker identity and are often ineffective against speaker verification attacks (SVA). To address these limitations, we propose Diffusion-Bridge (VocalBridge), a purification framework that learns a latent mapping from perturbed to clean speech in the EnCodec latent space. Using a time-conditioned 1D U-Net with a cosine noise schedule, the model enables efficient, transcript-free purification while preserving speaker-discriminative structure. We further introduce a Whisper-guided phoneme variant that incorporates lightweight temporal guidance without requiring ground-truth transcripts. Experimental results show that our approach consistently outperforms existing purification methods in recovering cloneable voices from protected speech. Our findings demonstrate the fragility of current perturbation-based defenses and highlight the need for more robust protection mechanisms against evolving voice-cloning and speaker verification threats.
>
---
#### [new 003] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
- **分类: eess.AS**

- **简介: 该论文属于语音与文本表示学习任务，旨在解决细粒度说话风格建模难题。通过构建FCaps数据集和提出CLSP模型，实现多粒度对比预训练，提升语音文本对齐与理解效果。**

- **链接: [https://arxiv.org/pdf/2601.03065v1](https://arxiv.org/pdf/2601.03065v1)**

> **作者:** Yifan Yang; Bing Han; Hui Wang; Wei Wang; Ziyang Ma; Long Zhou; Zengrui Jin; Guanrou Yang; Tianrui Wang; Xu Tan; Xie Chen
>
> **摘要:** Modeling fine-grained speaking styles remains challenging for language-speech representation pre-training, as existing speech-text models are typically trained with coarse captions or task-specific supervision, and scalable fine-grained style annotations are unavailable. We present FCaps, a large-scale dataset with fine-grained free-text style descriptions, encompassing 47k hours of speech and 19M fine-grained captions annotated via a novel end-to-end pipeline that directly grounds detailed captions in audio, thereby avoiding the error propagation caused by LLM-based rewriting in existing cascaded pipelines. Evaluations using LLM-as-a-judge demonstrate that our annotations surpass existing cascaded annotations in terms of correctness, coverage, and naturalness. Building on FCaps, we propose CLSP, a contrastive language-speech pre-trained model that integrates global and fine-grained supervision, enabling unified representations across multiple granularities. Extensive experiments demonstrate that CLSP learns fine-grained and multi-granular speech-text representations that perform reliably across global and fine-grained speech-text retrieval, zero-shot paralinguistic classification, and speech style similarity scoring, with strong alignment to human judgments. All resources will be made publicly available.
>
---
#### [new 004] Segment-Aware Conditioning for Training-Free Intra-Utterance Emotion and Duration Control in Text-to-Speech
- **分类: cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决无需训练的说话人情感与持续时间控制问题。提出段落感知的条件策略，实现细粒度的语句内控制。**

- **链接: [https://arxiv.org/pdf/2601.03170v1](https://arxiv.org/pdf/2601.03170v1)**

> **作者:** Qifan Liang; Yuansen Liu; Ruixin Wei; Nan Lu; Junchuan Zhao; Ye Wang
>
> **备注:** 24 pages, 8 figures, 7 tables, 3 lists
>
> **摘要:** While controllable Text-to-Speech (TTS) has achieved notable progress, most existing methods remain limited to inter-utterance-level control, making fine-grained intra-utterance expression challenging due to their reliance on non-public datasets or complex multi-stage training. In this paper, we propose a training-free controllable framework for pretrained zero-shot TTS to enable intra-utterance emotion and duration expression. Specifically, we propose a segment-aware emotion conditioning strategy that combines causal masking with monotonic stream alignment filtering to isolate emotion conditioning and schedule mask transitions, enabling smooth intra-utterance emotion shifts while preserving global semantic coherence. Based on this, we further propose a segment-aware duration steering strategy to combine local duration embedding steering with global EOS logit modulation, allowing local duration adjustment while ensuring globally consistent termination. To eliminate the need for segment-level manual prompt engineering, we construct a 30,000-sample multi-emotion and duration-annotated text dataset to enable LLM-based automatic prompt construction. Extensive experiments demonstrate that our training-free method not only achieves state-of-the-art intra-utterance consistency in multi-emotion and duration control, but also maintains baseline-level speech quality of the underlying TTS model. Audio samples are available at https://aclanonymous111.github.io/TED-TTS-DemoPage/.
>
---
#### [new 005] UniSRCodec: Unified and Low-Bitrate Single Codebook Codec with Sub-Band Reconstruction
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出UniSRCodec，属于音频编码任务，解决单码本编码器在高采样率、低带宽和高保真上的不足，通过时频压缩和子带重建提升性能。**

- **链接: [https://arxiv.org/pdf/2601.02776v1](https://arxiv.org/pdf/2601.02776v1)**

> **作者:** Zhisheng Zhang; Xiang Li; Yixuan Zhou; Jing Peng; Shengbo Cai; Guoyang Zeng; Zhiyong Wu
>
> **备注:** 6 pages, 2 figures, and 3 tables
>
> **摘要:** Neural Audio Codecs (NACs) can reduce transmission overhead by performing compact compression and reconstruction, which also aim to bridge the gap between continuous and discrete signals. Existing NACs can be divided into two categories: multi-codebook and single-codebook codecs. Multi-codebook codecs face challenges such as structural complexity and difficulty in adapting to downstream tasks, while single-codebook codecs, though structurally simpler, suffer from low-fidelity, ineffective modeling of unified audio, and an inability to support modeling of high-frequency audio. We propose the UniSRCodec, a single-codebook codec capable of supporting high sampling rate, low-bandwidth, high fidelity, and unified. We analyze the inefficiency of waveform-based compression and introduce the time and frequency compression method using the Mel-spectrogram, and cooperate with a Vocoder to recover the phase information of the original audio. Moreover, we propose a sub-band reconstruction technique to achieve high-quality compression across both low and high frequency bands. Subjective and objective experimental results demonstrate that UniSRCodec achieves state-of-the-art (SOTA) performance among cross-domain single-codebook codecs with only a token rate of 40, and its reconstruction quality is comparable to that of certain multi-codebook methods. Our demo page is available at https://wxzyd123.github.io/unisrcodec.
>
---
#### [new 006] Understanding Human Perception of Music Plagiarism Through a Computational Approach
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于音乐相似性分析任务，旨在解决音乐剽窃中人类感知与算法检测的差异问题。通过分析旋律、节奏和和弦进行，提出基于大语言模型的判断框架。**

- **链接: [https://arxiv.org/pdf/2601.02586v1](https://arxiv.org/pdf/2601.02586v1)**

> **作者:** Daeun Hwang; Hyeonbin Hwang
>
> **备注:** 3 pages, D. Hwang and H. Hwang, Understanding Human Perception of Music Plagiarism Through a Computational Approach, in Extended Abstracts for the Late-Breaking Demo Session of the 25th Int. Society for Music Information Retrieval Conf., San Francisco, United States, 2024
>
> **摘要:** There is a wide variety of music similarity detection algorithms, while discussions about music plagiarism in the real world are often based on audience perceptions. Therefore, we aim to conduct a study to examine the key criteria of human perception of music plagiarism, focusing on the three commonly used musical features in similarity analysis: melody, rhythm, and chord progression. After identifying the key features and levels of variation humans use in perceiving musical similarity, we propose a LLM-as-a-judge framework that applies a systematic, step-by-step approach, drawing on modules that extract such high-level attributes.
>
---
#### [new 007] Dynamic Quantization Error Propagation in Encoder-Decoder ASR Quantization
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，解决ASR模型在边缘设备上的量化误差累积问题。提出FADE方法，动态控制误差传播，提升模型稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2601.02455v1](https://arxiv.org/pdf/2601.02455v1)**

> **作者:** Xinyu Wang; Yajie Luo; Yihong Wu; Liheng Ma; Ziyu Zhao; Jingrui Tian; Lei Ding; Yufei Cui; Xiao-Wen Chang
>
> **备注:** 9 pages, 4 figures, 3 tables
>
> **摘要:** Running Automatic Speech Recognition (ASR) models on memory-constrained edge devices requires efficient compression. While layer-wise post-training quantization is effective, it suffers from error accumulation, especially in encoder-decoder architectures. Existing solutions like Quantization Error Propagation (QEP) are suboptimal for ASR due to the model's heterogeneity, processing acoustic features in the encoder while generating text in the decoder. To address this, we propose Fine-grained Alpha for Dynamic Quantization Error Propagation (FADE), which adaptively controls the trade-off between cross-layer error correction and local quantization. Experiments show that FADE significantly improves stability by reducing performance variance across runs, while simultaneously surpassing baselines in mean WER.
>
---
#### [new 008] Omni2Sound: Towards Unified Video-Text-to-Audio Generation
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文提出Omni2Sound，解决视频-文本到音频生成任务中的数据稀缺与多任务竞争问题，通过构建大规模数据集和设计多阶段训练方案实现统一高效生成。**

- **链接: [https://arxiv.org/pdf/2601.02731v1](https://arxiv.org/pdf/2601.02731v1)**

> **作者:** Yusheng Dai; Zehua Chen; Yuxuan Jiang; Baolong Gao; Qiuhong Ke; Jun Zhu; Jianfei Cai
>
> **摘要:** Training a unified model integrating video-to-audio (V2A), text-to-audio (T2A), and joint video-text-to-audio (VT2A) generation offers significant application flexibility, yet faces two unexplored foundational challenges: (1) the scarcity of high-quality audio captions with tight A-V-T alignment, leading to severe semantic conflict between multimodal conditions, and (2) cross-task and intra-task competition, manifesting as an adverse V2A-T2A performance trade-off and modality bias in the VT2A task. First, to address data scarcity, we introduce SoundAtlas, a large-scale dataset (470k pairs) that significantly outperforms existing benchmarks and even human experts in quality. Powered by a novel agentic pipeline, it integrates Vision-to-Language Compression to mitigate visual bias of MLLMs, a Junior-Senior Agent Handoff for a 5 times cost reduction, and rigorous Post-hoc Filtering to ensure fidelity. Consequently, SoundAtlas delivers semantically rich and temporally detailed captions with tight V-A-T alignment. Second, we propose Omni2Sound, a unified VT2A diffusion model supporting flexible input modalities. To resolve the inherent cross-task and intra-task competition, we design a three-stage multi-task progressive training schedule that converts cross-task competition into joint optimization and mitigates modality bias in the VT2A task, maintaining both audio-visual alignment and off-screen audio generation faithfulness. Finally, we construct VGGSound-Omni, a comprehensive benchmark for unified evaluation, including challenging off-screen tracks. With a standard DiT backbone, Omni2Sound achieves unified SOTA performance across all three tasks within a single model, demonstrating strong generalization across benchmarks with heterogeneous input conditions. The project page is at https://swapforward.github.io/Omni2Sound.
>
---
#### [new 009] Quantifying Quanvolutional Neural Networks Robustness for Speech in Healthcare Applications
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究语音识别中的噪声鲁棒性问题，比较量子神经网络与传统CNN在噪声环境下的表现，旨在提升医疗语音应用的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.02432v1](https://arxiv.org/pdf/2601.02432v1)**

> **作者:** Ha Tran; Bipasha Kashyap; Pubudu N. Pathirana
>
> **摘要:** Speech-based machine learning systems are sensitive to noise, complicating reliable deployment in emotion recognition and voice pathology detection. We evaluate the robustness of a hybrid quantum machine learning model, quanvolutional neural networks (QNNs) against classical convolutional neural networks (CNNs) under four acoustic corruptions (Gaussian noise, pitch shift, temporal shift, and speed variation) in a clean-train/corrupted-test regime. Using AVFAD (voice pathology) and TESS (speech emotion), we compare three QNN models (Random, Basic, Strongly) to a simple CNN baseline (CNN-Base), ResNet-18 and VGG-16 using accuracy and corruption metrics (CE, mCE, RCE, RmCE), and analyze architectural factors (circuit complexity or depth, convergence) alongside per-emotion robustness. QNNs generally outperform the CNN-Base under pitch shift, temporal shift, and speed variation (up to 22% lower CE/RCE at severe temporal shift), while the CNN-Base remains more resilient to Gaussian noise. Among quantum circuits, QNN-Basic achieves the best overall robustness on AVFAD, and QNN-Random performs strongest on TESS. Emotion-wise, fear is most robust (80-90% accuracy under severe corruptions), neutral can collapse under strong Gaussian noise (5.5% accuracy), and happy is most vulnerable to pitch, temporal, and speed distortions. QNNs also converge up to six times faster than the CNN-Base. To our knowledge, this is a systematic study of QNN robustness for speech under common non-adversarial acoustic corruptions, indicating that shallow entangling quantum front-ends can improve noise resilience while sensitivity to additive noise remains a challenge.
>
---
#### [new 010] Vulnerabilities of Audio-Based Biometric Authentication Systems Against Deepfake Speech Synthesis
- **分类: cs.SD; cs.CR**

- **简介: 该论文属于语音生物识别安全任务，探讨深度伪造语音对音频生物认证系统的威胁，分析现有系统在小样本攻击和跨方法泛化上的漏洞。**

- **链接: [https://arxiv.org/pdf/2601.02914v1](https://arxiv.org/pdf/2601.02914v1)**

> **作者:** Mengze Hong; Di Jiang; Zeying Xie; Weiwei Zhao; Guan Wang; Chen Jason Zhang
>
> **摘要:** As audio deepfakes transition from research artifacts to widely available commercial tools, robust biometric authentication faces pressing security threats in high-stakes industries. This paper presents a systematic empirical evaluation of state-of-the-art speaker authentication systems based on a large-scale speech synthesis dataset, revealing two major security vulnerabilities: 1) modern voice cloning models trained on very small samples can easily bypass commercial speaker verification systems; and 2) anti-spoofing detectors struggle to generalize across different methods of audio synthesis, leading to a significant gap between in-domain performance and real-world robustness. These findings call for a reconsideration of security measures and stress the need for architectural innovations, adaptive defenses, and the transition towards multi-factor authentication.
>
---
#### [new 011] Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决多类型音频检测与可解释性问题。通过构建频率-时间结构化推理链，提出FT-GRPO方法，提升检测性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.02983v1](https://arxiv.org/pdf/2601.02983v1)**

> **作者:** Yuankun Xie; Xiaoxuan Guo; Jiayi Zhou; Tao Wang; Jian Liu; Ruibo Fu; Xiaopeng Wang; Haonan Cheng; Long Ye
>
> **摘要:** Recent advances in audio large language models (ALLMs) have made high-quality synthetic audio widely accessible, increasing the risk of malicious audio deepfakes across speech, environmental sounds, singing voice, and music. Real-world audio deepfake detection (ADD) therefore requires all-type detectors that generalize across heterogeneous audio and provide interpretable decisions. Given the strong multi-task generalization ability of ALLMs, we first investigate their performance on all-type ADD under both supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). However, SFT using only binary real/fake labels tends to reduce the model to a black-box classifier, sacrificing interpretability. Meanwhile, vanilla RFT under sparse supervision is prone to reward hacking and can produce hallucinated, ungrounded rationales. To address this, we propose an automatic annotation and polishing pipeline that constructs Frequency-Time structured chain-of-thought (CoT) rationales, producing ~340K cold-start demonstrations. Building on CoT data, we propose Frequency Time-Group Relative Policy Optimization (FT-GRPO), a two-stage training paradigm that cold-starts ALLMs with SFT and then applies GRPO under rule-based frequency-time constraints. Experiments demonstrate that FT-GRPO achieves state-of-the-art performance on all-type ADD while producing interpretable, FT-grounded rationales. The data and code are available online.
>
---
#### [new 012] A Music Information Retrieval Approach to Classify Sub-Genres in Role Playing Games
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于音乐信息检索任务，旨在分析RPG游戏音乐的子类型特征，解决如何通过音乐特征区分不同子类型的问题。工作包括提取特征并分析其与游戏类型的关系。**

- **链接: [https://arxiv.org/pdf/2601.02591v1](https://arxiv.org/pdf/2601.02591v1)**

> **作者:** Daeun Hwang; Xuyuan Cai; Edward F. Melcer; Elin Carstensdottir
>
> **备注:** 3 pages, 1 figure. D. Hwang, X. Cai, E. Melcer, and E. Carstensdottir, A Music Information Retrieval Approach to Classify Sub-Genres in Role Playing Games, in Extended Abstracts for the Late-Breaking Demo Session of the 25th Int. Society for Music Information Retrieval Conf., San Francisco, United States, 2024
>
> **摘要:** Video game music (VGM) is often studied under the same lens as film music, which largely focuses on its theoretical functionality with relation to the identified genres of the media. However, till date, we are unaware of any systematic approach that analyzes the quantifiable musical features in VGM across several identified game genres. Therefore, we extracted musical features from VGM in games from three sub-genres of Role-Playing Games (RPG), and then hypothesized how different musical features are correlated to the perceptions and portrayals of each genre. This observed correlation may be used to further suggest such features are relevant to the expected storytelling elements or play mechanics associated with the sub-genre.
>
---
#### [new 013] Multi-channel multi-speaker transformer for speech recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音识别任务，解决远场多说话人识别问题。提出M2Former模型，有效处理多通道音频中的说话人干扰，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.02688v1](https://arxiv.org/pdf/2601.02688v1)**

> **作者:** Guo Yifan; Tian Yao; Suo Hongbin; Wan Yulong
>
> **备注:** Proc. INTERSPEECH 2023, 5 pages
>
> **摘要:** With the development of teleconferencing and in-vehicle voice assistants, far-field multi-speaker speech recognition has become a hot research topic. Recently, a multi-channel transformer (MCT) has been proposed, which demonstrates the ability of the transformer to model far-field acoustic environments. However, MCT cannot encode high-dimensional acoustic features for each speaker from mixed input audio because of the interference between speakers. Based on these, we propose the multi-channel multi-speaker transformer (M2Former) for far-field multi-speaker ASR in this paper. Experiments on the SMS-WSJ benchmark show that the M2Former outperforms the neural beamformer, MCT, dual-path RNN with transform-average-concatenate and multi-channel deep clustering based end-to-end systems by 9.2%, 14.3%, 24.9%, and 52.2% respectively, in terms of relative word error rate reduction.
>
---
#### [new 014] SPO-CLAPScore: Enhancing CLAP-based alignment prediction system with Standardize Preference Optimization, for the first XACLE Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频-文本语义对齐任务，提出SPO-CLAPScore方法，通过标准化偏好优化和听众筛选提升模型与人类判断的相关性。**

- **链接: [https://arxiv.org/pdf/2601.02900v1](https://arxiv.org/pdf/2601.02900v1)**

> **作者:** Taisei Takano; Ryoya Yoshida
>
> **备注:** https://github.com/ttakano398/SPO-CLAPScore
>
> **摘要:** The first XACLE Challenge (x-to-audio alignment challenge) addresses the critical need for automatic evaluation metrics that correlate with human perception of audio-text semantic alignment. In this paper, we describe the "Takano_UTokyo_03" system submitted to XACLE Challenge. Our approach leverages a CLAPScore-based architecture integrated with a novel training method called Standardized Preference Optimization (SPO). SPO standardizes the raw alignment scores provided by each listener, enabling the model to learn relative preferences and mitigate the impact of individual scoring biases. Additionally, we employ listener screening to exclude listeners with inconsistent ratings. Experimental evaluations demonstrate that both SPO and listener screening effectively improve the correlation with human judgment. Our system achieved 6th place in the challenge with a Spearman's rank correlation coefficient (SRCC) of 0.6142, demonstrating competitive performance within a marginal gap from the top-ranked systems. The code is available at https://github.com/ttakano398/SPO-CLAPScore.
>
---
#### [new 015] Vclip: Face-based Speaker Generation by Face-voice Association Learning
- **分类: eess.AS**

- **简介: 该论文属于人脸驱动语音合成任务，旨在解决音视频数据不足导致的合成质量低和领域不匹配问题。提出Vclip方法，通过面部与声音关联学习，提升合成语音与参考人脸的匹配度。**

- **链接: [https://arxiv.org/pdf/2601.02753v1](https://arxiv.org/pdf/2601.02753v1)**

> **作者:** Yao Shi; Yunfei Xu; Hongbin Suo; Yulong Wan; Haifeng Liu
>
> **备注:** work done in 2023
>
> **摘要:** This paper discusses the task of face-based speech synthesis, a kind of personalized speech synthesis where the synthesized voices are constrained to perceptually match with a reference face image. Due to the lack of TTS-quality audio-visual corpora, previous approaches suffer from either low synthesis quality or domain mismatch induced by a knowledge transfer scheme. This paper proposes a new approach called Vclip that utilizes the facial-semantic knowledge of the CLIP encoder on noisy audio-visual data to learn the association between face and voice efficiently, achieving 89.63% cross-modal verification AUC score on Voxceleb testset. The proposed method then uses a retrieval-based strategy, combined with GMM-based speaker generation module for a downstream TTS system, to produce probable target speakers given reference images. Experimental results demonstrate that the proposed Vclip system in conjunction with the retrieval step can bridge the gap between face and voice features for face-based speech synthesis. And using the feedback information distilled from downstream TTS helps to synthesize voices that match closely with reference faces. Demos available at sos1sos2sixteen.github.io/vclip.
>
---
#### [new 016] MoE Adapter for Large Audio Language Models: Sparsity, Disentanglement, and Gradient-Conflict-Free
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多模态语言模型任务，旨在解决音频信息异质性导致的梯度冲突问题。通过引入稀疏MoE-Adapter架构，实现特征解耦与高效学习。**

- **链接: [https://arxiv.org/pdf/2601.02967v1](https://arxiv.org/pdf/2601.02967v1)**

> **作者:** Yishu Lei; Shuwei He; Jing Hu; Dan Zhang; Xianlong Luo; Danxiang Zhu; Shikun Feng; Rui Liu; Jingzhou He; Yu Sun; Hua Wu; Haifeng Wang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Extending the input modality of Large Language Models~(LLMs) to the audio domain is essential for achieving comprehensive multimodal perception. However, it is well-known that acoustic information is intrinsically \textit{heterogeneous}, entangling attributes such as speech, music, and environmental context. Existing research is limited to a dense, parameter-shared adapter to model these diverse patterns, which induces \textit{gradient conflict} during optimization, as parameter updates required for distinct attributes contradict each other. To address this limitation, we introduce the \textit{\textbf{MoE-Adapter}}, a sparse Mixture-of-Experts~(MoE) architecture designed to decouple acoustic information. Specifically, it employs a dynamic gating mechanism that routes audio tokens to specialized experts capturing complementary feature subspaces while retaining shared experts for global context, thereby mitigating gradient conflicts and enabling fine-grained feature learning. Comprehensive experiments show that the MoE-Adapter achieves superior performance on both audio semantic and paralinguistic tasks, consistently outperforming dense linear baselines with comparable computational costs. Furthermore, we will release the related code and models to facilitate future research.
>
---
#### [new 017] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
- **分类: eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决真实语音生成带来的安全风险。通过提出XLSR-MamBo框架，结合Mamba与注意力机制，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.02944v1](https://arxiv.org/pdf/2601.02944v1)**

> **作者:** Kwok-Ho Ng; Tingting Song; Yongdong WU; Zhihua Xia
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Advanced speech synthesis technologies have enabled highly realistic speech generation, posing security risks that motivate research into audio deepfake detection (ADD). While state space models (SSMs) offer linear complexity, pure causal SSMs architectures often struggle with the content-based retrieval required to capture global frequency-domain artifacts. To address this, we explore the scaling properties of hybrid architectures by proposing XLSR-MamBo, a modular framework integrating an XLSR front-end with synergistic Mamba-Attention backbones. We systematically evaluate four topological designs using advanced SSM variants, Mamba, Mamba2, Hydra, and Gated DeltaNet. Experimental results demonstrate that the MamBo-3-Hydra-N3 configuration achieves competitive performance compared to other state-of-the-art systems on the ASVspoof 2021 LA, DF, and In-the-Wild benchmarks. This performance benefits from Hydra's native bidirectional modeling, which captures holistic temporal dependencies more efficiently than the heuristic dual-branch strategies employed in prior works. Furthermore, evaluations on the DFADD dataset demonstrate robust generalization to unseen diffusion- and flow-matching-based synthesis methods. Crucially, our analysis reveals that scaling backbone depth effectively mitigates the performance variance and instability observed in shallower models. These results demonstrate the hybrid framework's ability to capture artifacts in spoofed speech signals, providing an effective method for ADD.
>
---
#### [new 018] The Sonar Moment: Benchmarking Audio-Language Models in Audio Geo-Localization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频地理定位任务，旨在解决音频与地理位置匹配的问题。提出AGL1K基准数据集，评估音频语言模型的地理定位能力，并分析其性能与偏差。**

- **链接: [https://arxiv.org/pdf/2601.03227v1](https://arxiv.org/pdf/2601.03227v1)**

> **作者:** Ruixing Zhang; Zihan Liu; Leilei Sun; Tongyu Zhu; Weifeng Lv
>
> **摘要:** Geo-localization aims to infer the geographic origin of a given signal. In computer vision, geo-localization has served as a demanding benchmark for compositional reasoning and is relevant to public safety. In contrast, progress on audio geo-localization has been constrained by the lack of high-quality audio-location pairs. To address this gap, we introduce AGL1K, the first audio geo-localization benchmark for audio language models (ALMs), spanning 72 countries and territories. To extract reliably localizable samples from a crowd-sourced platform, we propose the Audio Localizability metric that quantifies the informativeness of each recording, yielding 1,444 curated audio clips. Evaluations on 16 ALMs show that ALMs have emerged with audio geo-localization capability. We find that closed-source models substantially outperform open-source models, and that linguistic clues often dominate as a scaffold for prediction. We further analyze ALMs' reasoning traces, regional bias, error causes, and the interpretability of the localizability metric. Overall, AGL1K establishes a benchmark for audio geo-localization and may advance ALMs with better geospatial reasoning capability.
>
---
#### [new 019] Discovering and Causally Validating Emotion-Sensitive Neurons in Large Audio-Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于情感计算任务，旨在揭示大音频语言模型中情感敏感神经元的机制。通过干预实验，验证了情感相关神经元的存在及其对情感识别的影响。**

- **链接: [https://arxiv.org/pdf/2601.03115v1](https://arxiv.org/pdf/2601.03115v1)**

> **作者:** Xiutian Zhao; Björn Schuller; Berrak Sisman
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Emotion is a central dimension of spoken communication, yet, we still lack a mechanistic account of how modern large audio-language models (LALMs) encode it internally. We present the first neuron-level interpretability study of emotion-sensitive neurons (ESNs) in LALMs and provide causal evidence that such units exist in Qwen2.5-Omni, Kimi-Audio, and Audio Flamingo 3. Across these three widely used open-source models, we compare frequency-, entropy-, magnitude-, and contrast-based neuron selectors on multiple emotion recognition benchmarks. Using inference-time interventions, we reveal a consistent emotion-specific signature: ablating neurons selected for a given emotion disproportionately degrades recognition of that emotion while largely preserving other classes, whereas gain-based amplification steers predictions toward the target emotion. These effects arise with modest identification data and scale systematically with intervention strength. We further observe that ESNs exhibit non-uniform layer-wise clustering with partial cross-dataset transfer. Taken together, our results offer a causal, neuron-level account of emotion decisions in LALMs and highlight targeted neuron interventions as an actionable handle for controllable affective behaviors.
>
---
#### [new 020] WearVox: An Egocentric Multichannel Voice Assistant Benchmark for Wearables
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出WearVox基准，用于评估可穿戴设备上的语音助手。解决真实场景下语音识别与理解的挑战，通过多通道音频数据和多种任务测试模型性能。**

- **链接: [https://arxiv.org/pdf/2601.02391v1](https://arxiv.org/pdf/2601.02391v1)**

> **作者:** Zhaojiang Lin; Yong Xu; Kai Sun; Jing Zheng; Yin Huang; Surya Teja Appini; Krish Narang; Renjie Tao; Ishan Kapil Jain; Siddhant Arora; Ruizhi Li; Yiteng Huang; Kaushik Patnaik; Wenfang Xu; Suwon Shon; Yue Liu; Ahmed A Aly; Anuj Kumar; Florian Metze; Xin Luna Dong
>
> **摘要:** Wearable devices such as AI glasses are transforming voice assistants into always-available, hands-free collaborators that integrate seamlessly with daily life, but they also introduce challenges like egocentric audio affected by motion and noise, rapid micro-interactions, and the need to distinguish device-directed speech from background conversations. Existing benchmarks largely overlook these complexities, focusing instead on clean or generic conversational audio. To bridge this gap, we present WearVox, the first benchmark designed to rigorously evaluate voice assistants in realistic wearable scenarios. WearVox comprises 3,842 multi-channel, egocentric audio recordings collected via AI glasses across five diverse tasks including Search-Grounded QA, Closed-Book QA, Side-Talk Rejection, Tool Calling, and Speech Translation, spanning a wide range of indoor and outdoor environments and acoustic conditions. Each recording is accompanied by rich metadata, enabling nuanced analysis of model performance under real-world constraints. We benchmark leading proprietary and open-source speech Large Language Models (SLLMs) and find that most real-time SLLMs achieve accuracies on WearVox ranging from 29% to 59%, with substantial performance degradation on noisy outdoor audio, underscoring the difficulty and realism of the benchmark. Additionally, we conduct a case study with two new SLLMs that perform inference with single-channel and multi-channel audio, demonstrating that multi-channel audio inputs significantly enhance model robustness to environmental noise and improve discrimination between device-directed and background speech. Our results highlight the critical importance of spatial audio cues for context-aware voice assistants and establish WearVox as a comprehensive testbed for advancing wearable voice AI research.
>
---
## 更新

#### [replaced 001] Large Language Model Guided Decoding for Self-Supervised Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决SSL-ASR中使用LLM作为语言模型的挑战，通过融合LLM与声学模型提升解码效果。**

- **链接: [https://arxiv.org/pdf/2508.02228v2](https://arxiv.org/pdf/2508.02228v2)**

> **作者:** Eyal Cohen; Bhiksha Raj; Joseph Keshet
>
> **备注:** 12 pages, 2 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Self-supervised automatic speech recognition (SSL-ASR) is an ASR approach that uses speech encoders pretrained on large amounts of unlabeled audio (e.g., wav2vec2.0 or HuBERT) and then fine-tunes them with limited labeled data to perform transcription. Decoding is usually performed with a CTC decoder, whose hypotheses are scored and refined using an external language model (LM), typically an n-gram or neural LM, which guides beam search to produce the final transcription. Using Large Language Models (LLMs) as external LMs remains a challenge, as their word probabilities are overly confident. The proposed method integrates an LLM with an SSL acoustic model by using the LLM's decoding mechanism to generate a set of candidate next tokens. For each candidate, the SSL model provides an acoustic score by aligning it to the input acoustics of the SSL model. A combined acoustic and LLM score is then calculated based on decomposing the MAP estimator of words given the acoustic signal. The tokens with the highest combined scores are maintained in a beam, which is then used to proceed to the next decoding step. We illustrate the effectiveness of our method through a comprehensive comparison with the current state-of-the-art LLM-based decoding, post-processing, and error-correcting methods across multiple datasets. Our approach proves particularly effective when processing challenging inputs such as complex speech sentences, acronyms, and domain-specific vocabulary.
>
---
#### [replaced 002] Musical Score Understanding Benchmark: Evaluating Large Language Models' Comprehension of Complete Musical Scores
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出MSU-Bench，用于评估大语言模型对完整乐谱的理解能力。任务为多模态音乐理解，解决模型在音高、节奏等层面的推理不足问题。工作包括构建基准数据集并评估多种模型表现。**

- **链接: [https://arxiv.org/pdf/2511.20697v2](https://arxiv.org/pdf/2511.20697v2)**

> **作者:** Congren Dai; Yue Yang; Krinos Li; Huichi Zhou; Shijie Liang; Zhang Bo; Enyang Liu; Ge Jin; Hongran An; Haosen Zhang; Peiyuan Jing; KinHei Lee; Zhenxuan Zhang; Xiaobing Li; Maosong Sun
>
> **摘要:** Understanding complete musical scores entails integrated reasoning over pitch, rhythm, harmony, and large-scale structure, yet the ability of Large Language Models and Vision-Language Models to interpret full musical notation remains insufficiently examined. We introduce the Musical Score Understanding Benchmark (MSU-Bench), the first large-scale, human-curated benchmark for score-level musical understanding across textual (ABC notation) and visual (PDF) modalities. MSU-Bench contains 1,800 generative Question-Answering pairs from works by Bach, Beethoven, Chopin, Debussy, and others, organised into four levels of increasing difficulty, ranging from onset information to texture and form. Evaluations of more than fifteen state-of-the-art models, in both zero-shot and fine-tuned settings, reveal pronounced modality gaps, unstable level-wise performance, and challenges in maintaining multilevel correctness. Fine-tuning substantially improves results across modalities while preserving general knowledge, positioning MSU-Bench as a robust foundation for future research in multimodal reasoning. To facilitate further research, we publicly release MSU-Bench and all associated resources.
>
---
#### [replaced 003] TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出TELEVAL，一个用于评估中文口语交互场景下语音语言模型的动态基准。解决现有评估基准与实际交互脱节的问题，通过内容准确性和互动恰当性两方面进行评价。**

- **链接: [https://arxiv.org/pdf/2507.18061v2](https://arxiv.org/pdf/2507.18061v2)**

> **作者:** Zehan Li; Hongjie Chen; Qing Wang; Yuxin Zhang; Jing Zhou; Xuening Wang; Hang Lv; Mengjie Du; Yaodong Song; Jie Lian; Jian Kang; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** Spoken language models (SLMs) have advanced rapidly in recent years, accompanied by a growing number of evaluation benchmarks. However, most existing benchmarks emphasize task completion and capability scaling, while remaining poorly aligned with how users interact with SLMs in real-world spoken conversations. Effective spoken interaction requires not only accurate understanding of user intent and content, but also the ability to respond with appropriate interactional strategies. In this paper, we present TELEVAL, a dynamic, user-centered benchmark for evaluating SLMs in realistic Chinese spoken interaction scenarios. TELEVAL consolidates evaluation into two core aspects. Reliable Content Fulfillment assesses whether models can comprehend spoken inputs and produce semantically correct responses. Interactional Appropriateness evaluates whether models act as socially capable interlocutors, requiring them not only to generate human-like, colloquial responses, but also to implicitly incorporate paralinguistic cues for natural interaction. Experiments reveal that, despite strong performance on semantic and knowledge-oriented tasks, current SLMs still struggle to produce natural and interactionally appropriate responses, highlighting the need for more interaction-faithful evaluation.
>
---
#### [replaced 004] Exploring How Audio Effects Alter Emotion with Foundation Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频效果对情感的影响，属于情感计算任务。旨在解决音频处理如何影响情绪感知的问题，通过基础模型分析音频效果与情感之间的关系。**

- **链接: [https://arxiv.org/pdf/2509.15151v3](https://arxiv.org/pdf/2509.15151v3)**

> **作者:** Stelios Katsis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **备注:** https://github.com/stelioskt/audioFX
>
> **摘要:** Audio effects (FX) such as reverberation, distortion, modulation, and dynamic range processing play a pivotal role in shaping emotional responses during music listening. While prior studies have examined links between low-level audio features and affective perception, the systematic impact of audio FX on emotion remains underexplored. This work investigates how foundation models - large-scale neural architectures pretrained on multimodal data - can be leveraged to analyze these effects. Such models encode rich associations between musical structure, timbre, and affective meaning, offering a powerful framework for probing the emotional consequences of sound design techniques. By applying various probing methods to embeddings from deep learning models, we examine the complex, nonlinear relationships between audio FX and estimated emotion, uncovering patterns tied to specific effects and evaluating the robustness of foundation audio models. Our findings aim to advance understanding of the perceptual impact of audio production practices, with implications for music cognition, performance, and affective computing.
>
---
#### [replaced 005] CMDAR: A Chinese Multi-scene Dynamic Audio Reasoning Benchmark with Diverse Challenges
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出CMDAR基准，用于评估AI在多场景动态音频推理任务中的能力。针对现有数据集的不足，CMDAR涵盖中文音频和复杂推理任务，测试并分析了多个模型的表现。**

- **链接: [https://arxiv.org/pdf/2509.22461v3](https://arxiv.org/pdf/2509.22461v3)**

> **作者:** Hui Li; Changhao Jiang; Hongyu Wang; Ming Zhang; Jiajun Sun; Zhixiong Yang; Yifei Cao; Shihan Dou; Xiaoran Fan; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** The ability to reason from audio, including speech, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and English audio data and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce CMDAR, a Chinese benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. CMDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on CMDAR and observe that they exhibit limitations in complex reasoning tasks. In CMDAR-main, Qwen2.5-Omni achieves 76.67% accuracy, whereas GPT-4o Audio reaches 68.47%. However, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice with multiple audios and open-ended tasks. And we provide detail analysis corresponding suggestions for the future development of large audio language models.
>
---
#### [replaced 006] MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转写与说话人分离任务，旨在解决传统系统无法端到端处理、上下文受限等问题。提出MOSS Transcribe Diarize模型，实现精准时间戳的说话人转写。**

- **链接: [https://arxiv.org/pdf/2601.01554v2](https://arxiv.org/pdf/2601.01554v2)**

> **作者:** MOSI. AI; Donghua Yu; Zhengyuan Lin; Chen Yang; Yiyang Zhang; Hanfu Chen; Jingqi Chen; Ke Chen; Liwei Fan; Yi Jiang; Jie Zhu; Muchen Li; Wenxuan Wang; Yang Wang; Zhe Xu; Yitian Gong; Yuqian Zhang; Wenbo Zhang; Zhaoye Fei; Qinyuan Cheng; Shimin Li; Xipeng Qiu
>
> **摘要:** Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks.
>
---
