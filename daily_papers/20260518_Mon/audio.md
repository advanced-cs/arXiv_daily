# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Modeling Music as a Time-Frequency Image: A 2D Tokenizer for Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决传统音频分词器在自回归建模中的问题。提出BandTok，一种基于二维梅尔频谱图的分词器，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.15831](https://arxiv.org/pdf/2605.15831)**

> **作者:** Yuqing Cheng; Xingyu Ma; Guochen Yu; Xiaotao Gu
>
> **摘要:** Autoregressive music generation depends strongly on the audio tokenizer. Existing high-fidelity codecs often use residual multi-codebook quantization, which preserves reconstruction quality but complicates language modeling after sequence flattening, as the residual hierarchy imposes strong sequential dependencies and can amplify error accumulation. We propose BandTok, a generation-oriented 2D Mel-spectrogram tokenizer that represents each frame with Mel-frequency band tokens from a single shared codebook. This design yields a physically interpretable time-frequency token grid with a more independent token structure, making it better suited for autoregressive modeling. BandTok improves reconstruction with a multi-scale PatchGAN objective and EMA codebook updates. We further introduce an autoregressive language model with 2D Rotary Position Embedding (2D RoPE) to preserve temporal and frequency-band structure during generation. Experiments show that BandTok improves over residual-codebook tokenizers and achieves strong results in a data-limited setting. The source code and generation demos for this work are publicly available.
>
---
#### [new 002] ARIA: A Diagnostic Framework for Music Training Data Attribution
- **分类: cs.SD**

- **简介: 该论文提出ARIA框架，用于音乐生成训练数据归属分析，解决如何识别影响生成结果的歌曲及其音乐方面的问题。通过分解归属并进行可靠性诊断，提升版权分析的准确性。**

- **链接: [https://arxiv.org/pdf/2605.16181](https://arxiv.org/pdf/2605.16181)**

> **作者:** Changheon Han; Ashkan Panahi; Kıvanç Tatar
>
> **备注:** Working Paper
>
> **摘要:** Training data attribution (TDA) for music generation must answer two questions that copyright analysis requires, namely which training songs influence a generated output and along which musical aspects the influence operates. Existing methods reduce influence to a single scalar, without revealing which musical aspects are dominant in that influence. We propose ARIA, a framework that decomposes attribution along musical aspects (five for symbolic music, three for audio) and pairs the decomposition with reliability diagnostics computed from the segment-level score matrix. It measures within-group similarity among the top-K attributed tracks against random reference groups drawn from the training pool, and diagnoses the score matrix through its singular value decomposition and column statistics. On a symbolic-music model where attribution ground truth is available through counterfactual retraining, the reliability diagnostics rank four attribution methods identically to that ground truth. On an audio music generation model, ARIA reveals attribution behaviors that vary substantially across TDA methods, flags score matrices whose retrieved tracks are nearly identical across queries rather than reflecting per-query attribution, and characterizes embedding-similarity retrieval baselines by the musical aspect each encoder surfaces. Together, ARIA produces per-aspect attribution evidence aligned with the musical aspects considered under the idea-expression distinction in copyright analysis.
>
---
#### [new 003] Improving Automatic Speech Recognition for Speakers Treated for Oral Cancer using Data Augmentation and LLM Error Correction
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升口腔癌患者语音的识别效果。通过数据增强和大语言模型纠错，显著降低了词错误率。**

- **链接: [https://arxiv.org/pdf/2605.15854](https://arxiv.org/pdf/2605.15854)**

> **作者:** Hidde Folkertsma; Thomas Tienkamp; Sebastiaan de Visscher; Max Witjes; Rob van Son; Jiapan Guo; Bence Mark Halpern
>
> **备注:** 7 pages, 3 tables. Accepted by EMBC 2026
>
> **摘要:** In recent years, the performance of automatic speech recognition (ASR) systems has made considerable progress. Unfortunately, for people with speech impairments, such as people treated for oral cancer (OC), ASR performance is still lagging behind. The scarcity and variability of OC speech data makes development of ASR models for this type of speech difficult. In this work, we use data augmentation and large language model (LLM) error correction to mitigate this problem. We apply various augmentation techniques on a corpus of Dutch oral cancer speech to create synthetic data, and evaluate their effect on ASR performance. We finetune Whisper and Massively Multilingual Speech (MMS) models for each augmentation technique and observe, on average, an 8% relative decrease in Word Error Rate (WER) when including data created using text-to-speech (TTS). When employing LLMs for error correction, we see a further 21.4-26.2% relative decrease in WER for finetuned ASR models and a 10.0% relative decrease for non-finetuned models. Overall, we achieve a 40% relative WER decrease for Whisper and a 50% relative WER decrease for MMS, indicating that a combination of data augmentation and LLM correction is a viable strategy for the recognition of OC speech.
>
---
#### [new 004] Beyond Content: A Comprehensive Speech Toxicity Dataset and Detection Framework Incorporating Paralinguistic Cues
- **分类: cs.SD; cs.AI; cs.CR**

- **简介: 该论文属于语音毒性检测任务，旨在解决现有方法忽视韵律特征的问题。构建了包含3万+音频的ToxiAlert-Bench数据集，并提出双头神经网络模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.15984](https://arxiv.org/pdf/2605.15984)**

> **作者:** Zhongjie Ba; Liang Yi; Peng Cheng; Qingcao Li; Qinglong Wang; Li Lu
>
> **摘要:** Toxic speech detection has become a crucial challenge in maintaining safe online communication environments. However, existing approaches to toxic speech detection often neglect the contribution of paralinguistic cues, such as emotion, intonation, and speech rate, which are key to detecting speech toxicity. Moreover, current toxic speech datasets are predominantly text-based, limiting the development of models that can capture paralinguistic this http URL address these challenges, we present ToxiAlert-Bench, a large-scale audio dataset comprising over 30,000 audio clips annotated with seven major toxic categories and twenty fine-grained toxic labels. Uniquely, our dataset annotates toxicity sources -- distinguishing between textual content and paralinguistic origins -- for comprehensive toxic speech this http URL, we propose a dual-head neural network with a multi-stage training strategy tailored for toxic speech detection. This architecture features two task-specific classification headers: one for identifying the source of sensitivity (textual or paralinguistic), and the other for categorizing the specific toxic type. The training process involves independent head training followed by joint fine-tuning to reduce task interference. To mitigate data class imbalance, we incorporate class-balanced sampling and weighted loss this http URL experimental results show that leveraging paralinguistic features significantly improves detection performance. Our method consistently outperforms existing baselines across multiple evaluation metrics, with a 21.1% relative improvement in Macro-F1 score and a 13.0% relative gain in accuracy over the strongest baseline, highlighting its enhanced effectiveness and practical applicability.
>
---
#### [new 005] Mind the Gap: Impact of Synthetic Conversational Data on Multi-Talker ASR and Speaker Diarization
- **分类: eess.AS**

- **简介: 该论文属于多说话人语音识别与说话人二分类任务，旨在研究合成对话数据对模型性能的影响。通过分析不同模拟策略，发现最优方案因任务而异，合成数据可有效提升性能。**

- **链接: [https://arxiv.org/pdf/2605.15442](https://arxiv.org/pdf/2605.15442)**

> **作者:** Alexander Polok; Ivan Medennikov; Jan Černocký; Shinji Watanabe; Lukáš Burget; Samuele Cornell
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Recent breakthroughs in multi-talker ASR (MT-ASR) and speaker diarization (SD) rely on synthetic data to mitigate the scarcity of large-scale conversational recordings, yet the impact of specific simulation choices remains poorly understood. To mind the gap between simulated mixtures and real-world interactions, we present a study of synthetic data generation for leading MT-ASR (DiCoW) and SD (Sortformer) systems. By introducing FastMSS, a highly efficient open-source simulator, we analyze turn-taking dynamics, source domain, acoustic augmentation, and data mixing strategies. Our findings reveal that optimal simulation recipes are highly task-dependent: increasing speech overlap benefits ASR but degrades diarization. Furthermore, broad source diversity consistently outperforms exact domain matching. Ultimately, synthetic-only training approaches real-data baselines, and combining simulated data with real recordings yields substantial gains over real-only training across both tasks.
>
---
#### [new 006] Real-time Speech Restoration using Data Prediction Mean Flows
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决实时语音恢复问题。提出一种低延迟的流匹配模型，减少计算量并提升效率。**

- **链接: [https://arxiv.org/pdf/2605.16251](https://arxiv.org/pdf/2605.16251)**

> **作者:** Sebastian Braun
>
> **摘要:** Generative models are capable to address difficult problems with non-unique solutions like bandwidth extension and gap filling, removing highly non-linear artifacts from codecs, clipping and distortion, as opposed to removing linear additive components like noise and reverb. While large offline processing models have shown impressive results, these tasks have not been solved with real-time capable models with low latency and compute. We propose a few-step flow matching model using Data Prediction Mean Flows in combination with suitable novel low-latency architecture to make flow matching models an attractive choice under theses constraints. Compared to state-of-the-art, our proposed mean flow model uses 120x less compute and introduces no algorithmic latency other than the STFT, while achieving similar audio quality.
>
---
#### [new 007] Sound Sparks Motion: Audio and Text Tuning for Video Editing
- **分类: cs.GR; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频编辑任务，解决生成模型难以精准控制运动的问题。通过音频和文本调优，在不修改模型权重的情况下实现运动编辑。**

- **链接: [https://arxiv.org/pdf/2605.15307](https://arxiv.org/pdf/2605.15307)**

> **作者:** AmirHossein Naghi Razlighi; Aryan Mikaeili; Ali Mahdavi-Amiri; Daniel Cohen-Or; Yiorgos Chrysanthou
>
> **备注:** Project Page: this https URL
>
> **摘要:** Motion-centric video editing remains difficult for large generative video models, which often respond well to appearance changes but struggle to produce specific, localized actions or state transitions in an existing clip. We introduce Sound Sparks Motion, a training-free framework that enables motion editing in an audio-visual video generation model by tuning its internal multimodal conditioning signals at test time. Rather than modifying model weights, our method tunes only two lightweight variables: an audio latent derived from the source video and a residual perturbation in the text-conditioning. We find that this combination can encourage motion edits that the underlying model often struggles to realize under prompt-only control. Since there is no direct way to evaluate temporal alignment between text and motion, we guide the tuning process using a vision-language model that provides feedback indicating whether the intended motion appears in the generated video. This simple supervision yields an effective semantic objective for motion editing, while regularization and perceptual-temporal constraints help preserve content and visual quality. Beyond per-video tuning, we show that the learned latent controls are transferable across videos, suggesting that they capture reusable motion-edit directions rather than overfitting to a single example. Our results highlight multimodal conditioning tuning, particularly through the audio pathway, as a promising direction for motion-aware video editing, and suggest that test-time tuning can serve as a lightweight probing mechanism that helps reveal latent motion controls embedded in the model's multimodal conditioning. Code and data are available via our project page: this https URL
>
---
## 更新

#### [replaced 001] JAM-Flow: Joint Audio-Motion Synthesis with Flow Matching
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出JAM-Flow，解决跨模态生成任务中的语音与面部动作同步问题，通过联合建模实现音频和视觉的统一生成。**

- **链接: [https://arxiv.org/pdf/2506.23552](https://arxiv.org/pdf/2506.23552)**

> **作者:** Mingi Kwon; Joonghyuk Shin; Jaeseok Jung; Jaesik Park; Youngjung Uh
>
> **备注:** project page: this https URL Under review. Preprint published on arXiv
>
> **摘要:** The intrinsic link between facial motion and speech is often overlooked in generative modeling, where talking head synthesis and text-to-speech (TTS) are typically addressed as separate tasks. This paper introduces JAM-Flow, a unified framework to simultaneously synthesize and condition on both facial motion and speech. Our approach leverages flow matching and a novel Multi-Modal Diffusion Transformer (MM-DiT) architecture, integrating specialized Motion-DiT and Audio-DiT modules. These are coupled via selective joint attention layers and incorporate key architectural choices, such as temporally aligned positional embeddings and localized joint attention masking, to enable effective cross-modal interaction while preserving modality-specific strengths. Trained with an inpainting-style objective, JAM-Flow supports a wide array of conditioning inputs-including text, reference audio, and reference motion-facilitating tasks such as synchronized talking head generation from text, audio-driven animation, and much more, within a single, coherent model. JAM-Flow significantly advances multi-modal generative modeling by providing a practical solution for holistic audio-visual synthesis. project page: this https URL
>
---
#### [replaced 002] Two-Dimensional Quantization for Geometry-Aware Audio Coding
- **分类: cs.SD; cs.AI; cs.IT; cs.LG; eess.SP**

- **简介: 该论文属于音频编码任务，旨在解决传统量化方法限制潜在空间几何结构的问题。提出二维量化(Q2D2)，通过网格投影提升压缩效率与代码本利用率。**

- **链接: [https://arxiv.org/pdf/2512.01537](https://arxiv.org/pdf/2512.01537)**

> **作者:** Tal Shuster; Eliya Nachmani
>
> **备注:** accepted to ICML 2026
>
> **摘要:** Recent neural audio codecs have achieved impressive reconstruction quality, typically relying on quantization methods such as Residual Vector Quantization (RVQ), Vector Quantization (VQ) and Finite Scalar Quantization (FSQ). However, these quantization techniques limit the geometric structure of the latent space, make it harder to capture correlations between features leading to inefficiency in representation learning, codebook utilization and token rate. In this paper we introduce Two-Dimensional Quantization (Q2D2), a quantization scheme in which feature pairs are projected onto structured 2D grids, such as hexagonal, rhombic, or rectangular tiling and quantized to the nearest grid values, yielding an implicit codebook defined by the product of grid levels, with codebook sizes comparable to conventional methods. Despite its simple geometric formulation, Q2D2 improves audio compression efficiency, with low token rates and high codebook utilization while maintaining state of the art reconstruction quality. Specifically, Q2D2 achieves competitive to superior performance in various objective and subjective reconstruction metrics, across extensive experiments in speech, audio and music domains compared to state of the art models. Comprehensive ablation studies further confirm the effectiveness of our design choices.
>
---
#### [replaced 003] CIS-BWE: Chaos-Informed Speech Bandwidth Extension
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音带宽扩展任务，旨在恢复受限带宽下丢失的高频成分。提出NDSI-BWE框架，采用多个受非线性系统启发的判别器，提升语音质量与自然度。**

- **链接: [https://arxiv.org/pdf/2507.15970](https://arxiv.org/pdf/2507.15970)**

> **作者:** Tarikul Islam Tamiti; Tonmoy Das; Nursadul Mamun; Anomadarshi Barua
>
> **摘要:** Recovering high-frequency components lost to bandwidth constraints is crucial for applications ranging from telecommunications to high-fidelity audio on limited resources. We introduce NDSI-BWE, a new adversarial Band Width Extension (BWE) framework that leverage four new discriminators inspired by nonlinear dynamical system to capture diverse temporal behaviors: a Multi-Resolution Lyapunov Discriminator (MRLD) for determining sensitivity to initial conditions by capturing deterministic chaos, a Multi-Scale Recurrence Discriminator (MS-RD) for self-similar recurrence dynamics, a Multi-Scale Detrended Fractal Analysis Discriminator (MSDFA) for long range slow variant scale invariant relationship, a Multi-Resolution Poincaré Plot Discriminator (MR-PPD) for capturing hidden latent space relationship, a Multi-Period Discriminator (MPD) for cyclical patterns, a Multi-Resolution Amplitude Discriminator (MRAD) and Multi-Resolution Phase Discriminator (MRPD) for capturing intricate amplitude-phase transition statistics. By using depth-wise convolution at the core of the convolutional block with in each discriminators, NDSI-BWE attains an eight-times parameter reduction. These seven discriminators guide a complex-valued ConformerNeXt based genetor with a dual stream Lattice-Net based architecture for simultaneous refinement of magnitude and phase. The genertor leverage the transformer based conformer's global dependency modeling and ConvNeXt block's local temporal modeling capability. Across six objective evaluation metrics and subjective based texts comprises of five human judges, NDSI-BWE establishes a new SoTA in BWE.
>
---
#### [replaced 004] Global Rotation Equivariant Phase Modeling for Speech Enhancement with Deep Magnitude-Phase Interaction
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决相位建模难题。提出一种具有全局旋转等变特性的双流框架，提升相位建模效果。**

- **链接: [https://arxiv.org/pdf/2602.08556](https://arxiv.org/pdf/2602.08556)**

> **作者:** Chengzhong Wang; Andong Li; Dingding Yao; Junfeng Li
>
> **备注:** Submitted to IEEE TASLP
>
> **摘要:** While deep learning has advanced speech enhancement (SE), effective phase modeling remains challenging, as conventional networks typically operate within a flat Euclidean feature space, which is not easy to model the underlying circular topology of the phase. To address this, we propose a magnitude-phase dual-stream framework that aligns the phase stream with its intrinsic circular geometry by enforcing Global Rotation Equivariance (GRE) characteristic. Specifically, we introduce a Magnitude-Phase Interactive Convolutional Module (MPICM) for modulus-based information exchange and a Hybrid-Attention Dual Feed-Forward Network (HADF) bottleneck for unified feature fusion, both of which are designed to preserve GRE in the phase stream. Comprehensive evaluations are conducted across phase retrieval, denoising, dereverberation, and bandwidth extension tasks to validate the superiority of the proposed method over multiple advanced baselines. Notably, the proposed architecture reduces Phase Distance by over 20\% in the phase retrieval task and improves PESQ by more than 0.1 in zero-shot cross-corpus denoising evaluations. The overall superiority is also established in universal SE tasks involving mixed distortions. Qualitative analysis further reveals that the learned phase features exhibit distinct periodic patterns, which are consistent with the intrinsic circular nature of the phase. The source code is available at this https URL.
>
---
#### [replaced 005] IsoNet: Spatially-aware audio-visual target speech extraction in complex acoustic environments
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音提取任务，解决紧凑设备在复杂声学环境中的目标语音提取问题。提出IsoNet系统，结合音频视觉信息与空间特征，提升提取效果。**

- **链接: [https://arxiv.org/pdf/2605.14736](https://arxiv.org/pdf/2605.14736)**

> **作者:** Dinanath Padhya; Sajen Maharjan; Binita Adhikari; Ishwor Raj Pokharel
>
> **备注:** 8 pages
>
> **摘要:** Target speech extraction remains difficult for compact devices because monaural neural models lack spatial evidence and classical beamformers lose resolving power when the microphone aperture is only a few centimetres. We present IsoNet, a user-selectable audio-visual target speech extraction system for a compact 4-microphone array. IsoNet combines complex multi-channel STFT features, GCC-PHAT spatial cues, face-conditioned visual embeddings, and auxiliary direction-of-arrival supervision inside a U-Net mask estimation network. Three curriculum variants were trained on 25,000 simulated VoxCeleb mixtures with progressively difficult SNR regimes. On a hard test set spanning -1 to 10 dB SNR, IsoNet-CL1 achieves 9.31 dB SI-SDR, a 4.85 dB improvement over the mixture, with PESQ 2.13 and STOI 0.84. Oracle delay-and-sum and MVDR beamformers degrade the same mixtures by 4.82 dB and 6.08 dB SI-SDRi, respectively, showing that the proposed learned multimodal conditioning solves a regime where conventional spatial filtering is ineffective. Ablation studies show consistent gains from visual conditioning, GCC-PHAT features, and extended delay-bin encoding. The results establish a compact-array, face-selectable speech extraction baseline under controlled simulation and identify the remaining barriers to real deployment, especially phase reconstruction, multi-interferer mixtures, and simulation-to-real transfer.
>
---
#### [replaced 006] Leveraging Local and Global Knowledge Integration with Time-Frequency Calibrated Distillation for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升低复杂度模型性能。提出一种融合时频校准的知识蒸馏方法，解决模型知识传递效率问题。**

- **链接: [https://arxiv.org/pdf/2506.13127](https://arxiv.org/pdf/2506.13127)**

> **作者:** Jiaming Cheng; Ruiyu Liang; Ye Ni; Chao Xu; Jing Li; Wei Zhou; Rui Liu; Björn W. Schuller; Xiaoshuai Hao
>
> **备注:** submitted to Neural Networks
>
> **摘要:** In this paper, we propose an intra-set and inter-set recursive fusion framework with time-frequency calibrated knowledge distillation (I$^2$SRF-TFCKD) for SE. Different from previous distillation strategies for SE, the proposed framework fully exploits the time-frequency differential information of speech while facilitating both local information focusing and global knowledge circulation. Firstly, we construct a collaborative distillation paradigm for intra-set and inter-set correlations. Within a correlated set, multi-layer teacher-student features are pairwise matched for calibrated distillation. Subsequently, we generate representative features from each correlated set through recursive fusion to form the fused feature set that enables inter-set knowledge interaction. Secondly, we propose a multi-layer interactive distillation based on dual-stream time-frequency cross-calibration, which calculates the teacher-student similarity calibration weights in the time and frequency domains respectively and performs cross-weighting, thus enabling refined allocation of distillation contributions across different layers according to speech characteristics. The proposed distillation strategy is applied to the dual-path dilated convolutional recurrent network (DPDCRN) that ranked first in the SE track of the L3DAS23 challenge. To evaluate the effectiveness of I$^2$SRF-TFCKD, we conduct experiments on both single-channel and multi-channel SE datasets. Objective evaluations demonstrate that the proposed KD strategy consistently and effectively improves the performance of the low-complexity student model and outperforms other distillation schemes.
>
---
