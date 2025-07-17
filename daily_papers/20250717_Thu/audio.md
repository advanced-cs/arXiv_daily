# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Stereo Sound Event Localization and Detection with Onscreen/offscreen Classification
- **分类: cs.SD; cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于声音事件定位与检测任务，解决立体声数据下的方向和距离估计问题，并引入了屏幕内外分类子任务。**

- **链接: [http://arxiv.org/pdf/2507.12042v1](http://arxiv.org/pdf/2507.12042v1)**

> **作者:** Kazuki Shimada; Archontis Politis; Iran R. Roman; Parthasaarathy Sudarsanam; David Diaz-Guerra; Ruchi Pandey; Kengo Uchida; Yuichiro Koyama; Naoya Takahashi; Takashi Shibuya; Shusuke Takahashi; Tuomas Virtanen; Yuki Mitsufuji
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** This paper presents the objective, dataset, baseline, and metrics of Task 3 of the DCASE2025 Challenge on sound event localization and detection (SELD). In previous editions, the challenge used four-channel audio formats of first-order Ambisonics (FOA) and microphone array. In contrast, this year's challenge investigates SELD with stereo audio data (termed stereo SELD). This change shifts the focus from more specialized 360{\deg} audio and audiovisual scene analysis to more commonplace audio and media scenarios with limited field-of-view (FOV). Due to inherent angular ambiguities in stereo audio data, the task focuses on direction-of-arrival (DOA) estimation in the azimuth plane (left-right axis) along with distance estimation. The challenge remains divided into two tracks: audio-only and audiovisual, with the audiovisual track introducing a new sub-task of onscreen/offscreen event classification necessitated by the limited FOV. This challenge introduces the DCASE2025 Task3 Stereo SELD Dataset, whose stereo audio and perspective video clips are sampled and converted from the STARSS23 recordings. The baseline system is designed to process stereo audio and corresponding video frames as inputs. In addition to the typical SELD event classification and localization, it integrates onscreen/offscreen classification for the audiovisual track. The evaluation metrics have been modified to introduce an onscreen/offscreen accuracy metric, which assesses the models' ability to identify which sound sources are onscreen. In the experimental evaluation, the baseline system performs reasonably well with the stereo audio data.
>
---
#### [new 002] RUMAA: Repeat-Aware Unified Music Audio Analysis for Score-Performance Alignment, Transcription, and Mistake Detection
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于音乐音频分析任务，解决乐谱与演奏对齐、转录及错误检测问题。提出RUMAA框架，统一处理上述任务，提升重复结构下的性能。**

- **链接: [http://arxiv.org/pdf/2507.12175v1](http://arxiv.org/pdf/2507.12175v1)**

> **作者:** Sungkyun Chang; Simon Dixon; Emmanouil Benetos
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** This study introduces RUMAA, a transformer-based framework for music performance analysis that unifies score-to-performance alignment, score-informed transcription, and mistake detection in a near end-to-end manner. Unlike prior methods addressing these tasks separately, RUMAA integrates them using pre-trained score and audio encoders and a novel tri-stream decoder capturing task interdependencies through proxy tasks. It aligns human-readable MusicXML scores with repeat symbols to full-length performance audio, overcoming traditional MIDI-based methods that rely on manually unfolded score-MIDI data with pre-specified repeat structures. RUMAA matches state-of-the-art alignment methods on non-repeated scores and outperforms them on scores with repeats in a public piano music dataset, while also delivering promising transcription and mistake detection results.
>
---
#### [new 003] DoRF: Doppler Radiance Fields for Robust Human Activity Recognition Using Wi-Fi
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于人体活动识别任务，旨在提升Wi-Fi信号在不同环境下的泛化能力。通过构建多普勒辐射场（DoRF），从一维多普勒数据中重建三维运动表示，增强识别鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12132v1](http://arxiv.org/pdf/2507.12132v1)**

> **作者:** Navid Hasanzadeh; Shahrokh Valaee
>
> **摘要:** Wi-Fi Channel State Information (CSI) has gained increasing interest for remote sensing applications. Recent studies show that Doppler velocity projections extracted from CSI can enable human activity recognition (HAR) that is robust to environmental changes and generalizes to new users. However, despite these advances, generalizability still remains insufficient for practical deployment. Inspired by neural radiance fields (NeRF), which learn a volumetric representation of a 3D scene from 2D images, this work proposes a novel approach to reconstruct an informative 3D latent motion representation from one-dimensional Doppler velocity projections extracted from Wi-Fi CSI. The resulting latent representation is then used to construct a uniform Doppler radiance field (DoRF) of the motion, providing a comprehensive view of the performed activity and improving the robustness to environmental variability. The results show that the proposed approach noticeably enhances the generalization accuracy of Wi-Fi-based HAR, highlighting the strong potential of DoRFs for practical sensing applications.
>
---
#### [new 004] Towards Scalable AASIST: Refining Graph Attention for Speech Deepfake Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升AASIST模型在有限数据下的反欺骗能力。通过改进注意力机制和融合策略，显著降低了等错误率。**

- **链接: [http://arxiv.org/pdf/2507.11777v1](http://arxiv.org/pdf/2507.11777v1)**

> **作者:** Ivan Viakhirev; Daniil Sirota; Aleksandr Smirnov; Kirill Borodin
>
> **摘要:** Advances in voice conversion and text-to-speech synthesis have made automatic speaker verification (ASV) systems more susceptible to spoofing attacks. This work explores modest refinements to the AASIST anti-spoofing architecture. It incorporates a frozen Wav2Vec 2.0 encoder to retain self-supervised speech representations in limited-data settings, substitutes the original graph attention block with a standardized multi-head attention module using heterogeneous query projections, and replaces heuristic frame-segment fusion with a trainable, context-aware integration layer. When evaluated on the ASVspoof 5 corpus, the proposed system reaches a 7.6\% equal error rate (EER), improving on a re-implemented AASIST baseline under the same training conditions. Ablation experiments suggest that each architectural change contributes to the overall performance, indicating that targeted adjustments to established models may help strengthen speech deepfake detection in practical scenarios. The code is publicly available at https://github.com/KORALLLL/AASIST_SCALING.
>
---
#### [new 005] Room Impulse Response Generation Conditioned on Acoustic Parameters
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决未知房间布局下生成逼真混响的问题。通过基于声学参数的深度学习模型生成房间脉冲响应，提升感知真实感。**

- **链接: [http://arxiv.org/pdf/2507.12136v1](http://arxiv.org/pdf/2507.12136v1)**

> **作者:** Silvia Arellano; Chunghsin Yeh; Gautam Bhattacharya; Daniel Arteaga
>
> **备注:** 4+1 pages, 2 figures; accepted in IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2025)
>
> **摘要:** The generation of room impulse responses (RIRs) using deep neural networks has attracted growing research interest due to its applications in virtual and augmented reality, audio postproduction, and related fields. Most existing approaches condition generative models on physical descriptions of a room, such as its size, shape, and surface materials. However, this reliance on geometric information limits their usability in scenarios where the room layout is unknown or when perceptual realism (how a space sounds to a listener) is more important than strict physical accuracy. In this study, we propose an alternative strategy: conditioning RIR generation directly on a set of RIR acoustic parameters. These parameters include various measures of reverberation time and direct sound to reverberation ratio, both broadband and bandwise. By specifying how the space should sound instead of how it should look, our method enables more flexible and perceptually driven RIR generation. We explore both autoregressive and non-autoregressive generative models operating in the Descript Audio Codec domain, using either discrete token sequences or continuous embeddings. Specifically, we have selected four models to evaluate: an autoregressive transformer, the MaskGIT model, a flow matching model, and a classifier-based approach. Objective and subjective evaluations are performed to compare these methods with state-of-the-art alternatives. Results show that the proposed models match or outperform state-of-the-art alternatives, with the MaskGIT model achieving the best performance.
>
---
#### [new 006] MambaRate: Speech Quality Assessment Across Different Sampling Rates
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，解决高采样率下MOS预测问题。提出MambaRate模型，利用自监督嵌入和状态空间建模，提升预测准确性。**

- **链接: [http://arxiv.org/pdf/2507.12090v1](http://arxiv.org/pdf/2507.12090v1)**

> **作者:** Panos Kakoulidis; Iakovi Alexiou; Junkwang Oh; Gunu Jho; Inchul Hwang; Pirros Tsiakoulis; Aimilios Chalamandaris
>
> **备注:** Submitted to ASRU 2025 (AudioMOS Challenge 2025 Track 3)
>
> **摘要:** We propose MambaRate, which predicts Mean Opinion Scores (MOS) with limited bias regarding the sampling rate of the waveform under evaluation. It is designed for Track 3 of the AudioMOS Challenge 2025, which focuses on predicting MOS for speech in high sampling frequencies. Our model leverages self-supervised embeddings and selective state space modeling. The target ratings are encoded in a continuous representation via Gaussian radial basis functions (RBF). The results of the challenge were based on the system-level Spearman's Rank Correllation Coefficient (SRCC) metric. An initial MambaRate version (T16 system) outperformed the pre-trained baseline (B03) by ~14% in a few-shot setting without pre-training. T16 ranked fourth out of five in the challenge, differing by ~6% from the winning system. We present additional results on the BVCC dataset as well as ablations with different representations as input, which outperform the initial T16 version.
>
---
#### [new 007] Quantize More, Lose Less: Autoregressive Generation from Residually Quantized Speech Representations
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决单码本表示导致的信息丢失问题。提出QTTS框架，利用多码本模型提升合成质量与表达力。**

- **链接: [http://arxiv.org/pdf/2507.12197v1](http://arxiv.org/pdf/2507.12197v1)**

> **作者:** Yichen Han; Xiaoyang Hao; Keming Chen; Weibo Xiong; Jun He; Ruonan Zhang; Junjie Cao; Yue Liu; Bowen Li; Dongrui Zhang; Hui Xia; Huilei Fu; Kai Jia; Kaixuan Guo; Mingli Jin; Qingyun Meng; Ruidong Ma; Ruiqian Fang; Shaotong Guo; Xuhui Li; Yang Xiang; Ying Zhang; Yulong Liu; Yunfeng Li; Yuyi Zhang; Yuze Zhou; Zhen Wang; Zhaowen Chen
>
> **摘要:** Text-to-speech (TTS) synthesis has seen renewed progress under the discrete modeling paradigm. Existing autoregressive approaches often rely on single-codebook representations, which suffer from significant information loss. Even with post-hoc refinement techniques such as flow matching, these methods fail to recover fine-grained details (e.g., prosodic nuances, speaker-specific timbres), especially in challenging scenarios like singing voice or music synthesis. We propose QTTS, a novel TTS framework built upon our new audio codec, QDAC. The core innovation of QDAC lies in its end-to-end training of an ASR-based auto-regressive network with a GAN, which achieves superior semantic feature disentanglement for scalable, near-lossless compression. QTTS models these discrete codes using two innovative strategies: the Hierarchical Parallel architecture, which uses a dual-AR structure to model inter-codebook dependencies for higher-quality synthesis, and the Delay Multihead approach, which employs parallelized prediction with a fixed delay to accelerate inference speed. Our experiments demonstrate that the proposed framework achieves higher synthesis quality and better preserves expressive content compared to baseline. This suggests that scaling up compression via multi-codebook modeling is a promising direction for high-fidelity, general-purpose speech and audio generation.
>
---
#### [new 008] A Multimodal Data Fusion Generative Adversarial Network for Real Time Underwater Sound Speed Field Construction
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于 underwater sound speed field construction 任务，旨在无需现场数据实现高精度声速分布估计。提出 MDF-RAGAN 模型，融合多模态数据并引入注意力机制，显著提升建模精度。**

- **链接: [http://arxiv.org/pdf/2507.11812v1](http://arxiv.org/pdf/2507.11812v1)**

> **作者:** Wei Huang; Yuqiang Huang; Yanan Wu; Tianhe Xu; Junting Wang; Hao Zhang
>
> **摘要:** Sound speed profiles (SSPs) are essential parameters underwater that affects the propagation mode of underwater signals and has a critical impact on the energy efficiency of underwater acoustic communication and accuracy of underwater acoustic positioning. Traditionally, SSPs can be obtained by matching field processing (MFP), compressive sensing (CS), and deep learning (DL) methods. However, existing methods mainly rely on on-site underwater sonar observation data, which put forward strict requirements on the deployment of sonar observation systems. To achieve high-precision estimation of sound velocity distribution in a given sea area without on-site underwater data measurement, we propose a multi-modal data-fusion generative adversarial network model with residual attention block (MDF-RAGAN) for SSP construction. To improve the model's ability for capturing global spatial feature correlations, we embedded the attention mechanisms, and use residual modules for deeply capturing small disturbances in the deep ocean sound velocity distribution caused by changes of SST. Experimental results on real open dataset show that the proposed model outperforms other state-of-the-art methods, which achieves an accuracy with an error of less than 0.3m/s. Specifically, MDF-RAGAN not only outperforms convolutional neural network (CNN) and spatial interpolation (SITP) by nearly a factor of two, but also achieves about 65.8\% root mean square error (RMSE) reduction compared to mean profile, which fully reflects the enhancement of overall profile matching by multi-source fusion and cross-modal attention.
>
---
#### [new 009] EME-TTS: Unlocking the Emphasis and Emotion Link in Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文属于情感语音合成任务，解决情感与强调互动不足的问题。提出EME-TTS框架，结合强调感知增强模块，提升情感语音的表达力与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.12015v1](http://arxiv.org/pdf/2507.12015v1)**

> **作者:** Haoxun Li; Leyuan Qu; Jiaxi Hu; Taihao Li
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** In recent years, emotional Text-to-Speech (TTS) synthesis and emphasis-controllable speech synthesis have advanced significantly. However, their interaction remains underexplored. We propose Emphasis Meets Emotion TTS (EME-TTS), a novel framework designed to address two key research questions: (1) how to effectively utilize emphasis to enhance the expressiveness of emotional speech, and (2) how to maintain the perceptual clarity and stability of target emphasis across different emotions. EME-TTS employs weakly supervised learning with emphasis pseudo-labels and variance-based emphasis features. Additionally, the proposed Emphasis Perception Enhancement (EPE) block enhances the interaction between emotional signals and emphasis positions. Experimental results show that EME-TTS, when combined with large language models for emphasis position prediction, enables more natural emotional speech synthesis while preserving stable and distinguishable target emphasis across emotions. Synthesized samples are available on-line.
>
---
#### [new 010] Schrödinger Bridge Consistency Trajectory Models for Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，解决扩散模型推理速度慢的问题。通过结合Schrödinger桥与一致性轨迹模型，提升语音质量与推理效率。**

- **链接: [http://arxiv.org/pdf/2507.11925v1](http://arxiv.org/pdf/2507.11925v1)**

> **作者:** Shuichiro Nishigori; Koichi Saito; Naoki Murata; Masato Hirano; Shusuke Takahashi; Yuki Mitsufuji
>
> **摘要:** Speech enhancement (SE) utilizing diffusion models is a promising technology that improves speech quality in noisy speech data. Furthermore, the Schr\"odinger bridge (SB) has recently been used in diffusion-based SE to improve speech quality by resolving a mismatch between the endpoint of the forward process and the starting point of the reverse process. However, the SB still exhibits slow inference owing to the necessity of a large number of function evaluations (NFE) for inference to obtain high-quality results. While Consistency Models (CMs) address this issue by employing consistency training that uses distillation from pretrained models in the field of image generation, it does not improve generation quality when the number of steps increases. As a solution to this problem, Consistency Trajectory Models (CTMs) not only accelerate inference speed but also maintain a favorable trade-off between quality and speed. Furthermore, SoundCTM demonstrates the applicability of CTM techniques to the field of sound generation. In this paper, we present Schr\"odinger bridge Consistency Trajectory Models (SBCTM) by applying the CTM's technique to the Schr\"odinger bridge for SE. Additionally, we introduce a novel auxiliary loss, including a perceptual loss, into the original CTM's training framework. As a result, SBCTM achieves an approximately 16x improvement in the real-time factor (RTF) compared to the conventional Schr\"odinger bridge for SE. Furthermore, the favorable trade-off between quality and speed in SBCTM allows for time-efficient inference by limiting multi-step refinement to cases where 1-step inference is insufficient. Our code, pretrained models, and audio samples are available at https://github.com/sony/sbctm/.
>
---
#### [new 011] Self-Boosted Weight-Constrained FxLMS: A Robustness Distributed Active Noise Control Algorithm Without Internode Communication
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于主动降噪任务，解决分布式系统中通信延迟与计算复杂度问题。提出SB-WCFxLMS算法，在无节点通信下提升降噪性能。**

- **链接: [http://arxiv.org/pdf/2507.12045v1](http://arxiv.org/pdf/2507.12045v1)**

> **作者:** Junwei Ji; Dongyuan Shi; Zhengding Luo; Boxiang Wang; Woon-Seng Gan
>
> **摘要:** Compared to the conventional centralized multichannel active noise control (MCANC) algorithm, which requires substantial computational resources, decentralized approaches exhibit higher computational efficiency but typically result in inferior noise reduction performance. To enhance performance, distributed ANC methods have been introduced, enabling information exchange among ANC nodes; however, the resulting communication latency often compromises system stability. To overcome these limitations, we propose a self-boosted weight-constrained filtered-reference least mean square (SB-WCFxLMS) algorithm for the distributed MCANC system without internode communication. The WCFxLMS algorithm is specifically designed to mitigate divergence issues caused by the internode cross-talk effect. The self-boosted strategy lets each ANC node independently adapt its constraint parameters based on its local noise reduction performance, thus ensuring effective noise cancellation without the need for inter-node communication. With the assistance of this mechanism, this approach significantly reduces both computational complexity and communication overhead. Numerical simulations employing real acoustic paths and compressor noise validate the effectiveness and robustness of the proposed system. The results demonstrate that our proposed method achieves satisfactory noise cancellation performance with minimal resource requirements.
>
---
#### [new 012] Modal Analysis of Multimode Waveguides Based on Large Step Size AdaMax from Far-Field Amplitudes
- **分类: physics.comp-ph; cs.SD; physics.optics**

- **简介: 该论文属于 multimode waveguide 的模态分析任务，旨在解决传统方法精度低、计算成本高的问题。通过引入 AdaMax 优化器，从远场振幅中提取模态分布，提升准确性和效率。**

- **链接: [http://arxiv.org/pdf/2507.12299v1](http://arxiv.org/pdf/2507.12299v1)**

> **作者:** Jingtong Li; Dongting Huang; Minhui Xiong; Mingzhi Li
>
> **摘要:** Optimizing multimode waveguide performance depends on modal analysis; however, current approaches focus predominantly on modal power distribution and, limited by experimental hardware and conditions, exhibit low accuracy, poor adaptability, and high computational cost. In this work, under a power-normalization constraint, we employ the AdaMax optimizer with a large-step-size strategy to perform modal analysis of multimode waveguides from far-field amplitude measurements. Our method retrieves both the modal power distribution and the modal relative-phase distribution, and we elucidate how twin-image ambiguity limits the capability to analyze modal relative-phase distributions. Experimental results demonstrate that the proposed method performs well for both rectangular and circular waveguides, maintaining high accuracy and robustness under noise with signal-to-noise ratios (SNRs) ranging from 20 to 120 dB, and achieving substantial improvements in accuracy and computational cost over comparable methods. This method provides a novel solution for modal analysis with broad application potential.
>
---
#### [new 013] Spontaneous Spatial Cognition Emerges during Egocentric Video Viewing through Non-invasive BCI
- **分类: q-bio.NC; cs.CV; eess.SP**

- **简介: 该论文属于脑机接口与空间认知研究，旨在解码被动观看第一视角视频时的6D姿态。通过EEG实现空间位置与方向的实时解码，揭示大脑在被动状态下的空间表征机制。**

- **链接: [http://arxiv.org/pdf/2507.12417v1](http://arxiv.org/pdf/2507.12417v1)**

> **作者:** Weichen Dai; Yuxuan Huang; Li Zhu; Dongjun Liu; Yu Zhang; Qibin Zhao; Andrzej Cichocki; Fabio Babiloni; Ke Li; Jianyu Qiu; Gangyong Jia; Wanzeng Kong; Qing Wu
>
> **摘要:** Humans possess a remarkable capacity for spatial cognition, allowing for self-localization even in novel or unfamiliar environments. While hippocampal neurons encoding position and orientation are well documented, the large-scale neural dynamics supporting spatial representation, particularly during naturalistic, passive experience, remain poorly understood. Here, we demonstrate for the first time that non-invasive brain-computer interfaces (BCIs) based on electroencephalography (EEG) can decode spontaneous, fine-grained egocentric 6D pose, comprising three-dimensional position and orientation, during passive viewing of egocentric video. Despite EEG's limited spatial resolution and high signal noise, we find that spatially coherent visual input (i.e., continuous and structured motion) reliably evokes decodable spatial representations, aligning with participants' subjective sense of spatial engagement. Decoding performance further improves when visual input is presented at a frame rate of 100 ms per image, suggesting alignment with intrinsic neural temporal dynamics. Using gradient-based backpropagation through a neural decoding model, we identify distinct EEG channels contributing to position -- and orientation specific -- components, revealing a distributed yet complementary neural encoding scheme. These findings indicate that the brain's spatial systems operate spontaneously and continuously, even under passive conditions, challenging traditional distinctions between active and passive spatial cognition. Our results offer a non-invasive window into the automatic construction of egocentric spatial maps and advance our understanding of how the human mind transforms everyday sensory experience into structured internal representations.
>
---
#### [new 014] Exploring Gender Bias in Alzheimer's Disease Detection: Insights from Mandarin and Greek Speech Perception
- **分类: cs.CL; cs.HC; cs.SD**

- **简介: 该论文属于阿尔茨海默病语音检测任务，旨在解决性别偏见问题。研究发现男性语音更易被误判为患病，且声学特征与判断相关。**

- **链接: [http://arxiv.org/pdf/2507.12356v1](http://arxiv.org/pdf/2507.12356v1)**

> **作者:** Liu He; Yuanchao Li; Rui Feng; XinRan Han; Yin-Long Liu; Yuwei Yang; Zude Zhu; Jiahong Yuan
>
> **备注:** 12 pages, 5 figures, conference or other essential info
>
> **摘要:** Gender bias has been widely observed in speech perception tasks, influenced by the fundamental voicing differences between genders. This study reveals a gender bias in the perception of Alzheimer's Disease (AD) speech. In a perception experiment involving 16 Chinese listeners evaluating both Chinese and Greek speech, we identified that male speech was more frequently identified as AD, with this bias being particularly pronounced in Chinese speech. Acoustic analysis showed that shimmer values in male speech were significantly associated with AD perception, while speech portion exhibited a significant negative correlation with AD identification. Although language did not have a significant impact on AD perception, our findings underscore the critical role of gender bias in AD speech perception. This work highlights the necessity of addressing gender bias when developing AD detection models and calls for further research to validate model performance across different linguistic contexts.
>
---
#### [new 015] Soft-Constrained Spatially Selective Active Noise Control for Open-fitting Hearables
- **分类: eess.AS; cs.SY; eess.SP; eess.SY**

- **简介: 该论文属于主动降噪任务，解决开放式耳机在降噪同时保持语音清晰的问题。提出软约束SSANC系统，平衡语音失真与降噪效果。**

- **链接: [http://arxiv.org/pdf/2507.12122v1](http://arxiv.org/pdf/2507.12122v1)**

> **作者:** Tong Xiao; Reinhild Roden; Matthias Blau; Simon Doclo
>
> **备注:** Accepted at IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** Recent advances in spatially selective active noise control (SSANC) using multiple microphones have enabled hearables to suppress undesired noise while preserving desired speech from a specific direction. Aiming to achieve minimal speech distortion, a hard constraint has been used in previous work in the optimization problem to compute the control filter. In this work, we propose a soft-constrained SSANC system that uses a frequency-independent parameter to trade off between speech distortion and noise reduction. We derive both time- and frequency-domain formulations, and show that conventional active noise control and hard-constrained SSANC represent two limiting cases of the proposed design. We evaluate the system through simulations using a pair of open-fitting hearables in an anechoic environment with one speech source and two noise sources. The simulation results validate the theoretical derivations and demonstrate that for a broad range of the trade-off parameter, the signal-to-noise ratio and the speech quality and intelligibility in terms of PESQ and ESTOI can be substantially improved compared to the hard-constrained design.
>
---
## 更新

#### [replaced 001] Epic-Sounds: A Large-scale Dataset of Actions That Sound
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2302.00646v3](http://arxiv.org/pdf/2302.00646v3)**

> **作者:** Jaesung Huh; Jacob Chalk; Evangelos Kazakos; Dima Damen; Andrew Zisserman
>
> **备注:** Accepted at TPAMI
>
> **摘要:** We introduce EPIC-SOUNDS, a large-scale dataset of audio annotations capturing temporal extents and class labels within the audio stream of the egocentric videos. We propose an annotation pipeline where annotators temporally label distinguishable audio segments and describe the action that could have caused this sound. We identify actions that can be discriminated purely from audio, through grouping these free-form descriptions of audio into classes. For actions that involve objects colliding, we collect human annotations of the materials of these objects (e.g. a glass object being placed on a wooden surface), which we verify from video, discarding ambiguities. Overall, EPIC-SOUNDS includes 78.4k categorised segments of audible events and actions, distributed across 44 classes as well as 39.2k non-categorised segments. We train and evaluate state-of-the-art audio recognition and detection models on our dataset, for both audio-only and audio-visual methods. We also conduct analysis on: the temporal overlap between audio events, the temporal and label correlations between audio and visual modalities, the ambiguities in annotating materials from audio-only input, the importance of audio-only labels and the limitations of current models to understand actions that sound.
>
---
#### [replaced 002] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v3](http://arxiv.org/pdf/2505.15670v3)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [replaced 003] Learning Perceptually Relevant Temporal Envelope Morphing
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.01588v2](http://arxiv.org/pdf/2506.01588v2)**

> **作者:** Satvik Dixit; Sungjoon Park; Chris Donahue; Laurie M. Heller
>
> **备注:** Accepted at WASPAA 2025
>
> **摘要:** Temporal envelope morphing, the process of interpolating between the amplitude dynamics of two audio signals, is an emerging problem in generative audio systems that lacks sufficient perceptual grounding. Morphing of temporal envelopes in a perceptually intuitive manner should enable new methods for sound blending in creative media and for probing perceptual organization in psychoacoustics. However, existing audio morphing techniques often fail to produce intermediate temporal envelopes when input sounds have distinct temporal structures; many morphers effectively overlay both temporal structures, leading to perceptually unnatural results. In this paper, we introduce a novel workflow for learning envelope morphing with perceptual guidance: we first derive perceptually grounded morphing principles through human listening studies, then synthesize large-scale datasets encoding these principles, and finally train machine learning models to create perceptually intermediate morphs. Specifically, we present: (1) perceptual principles that guide envelope morphing, derived from our listening studies, (2) a supervised framework to learn these principles, (3) an autoencoder that learns to compress temporal envelope structures into latent representations, and (4) benchmarks for evaluating audio envelope morphs, using both synthetic and naturalistic data, and show that our approach outperforms existing methods in producing temporally intermediate morphs. All code, models, and datasets will be made publicly available upon publication.
>
---
#### [replaced 004] Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04457v3](http://arxiv.org/pdf/2505.04457v3)**

> **作者:** Shigeki Karita; Yuma Koizumi; Heiga Zen; Haruko Ishikawa; Robin Scheibler; Michiel Bacchiani
>
> **备注:** Accepted to IEEE WASPAA2025
>
> **摘要:** Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaveFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators.
>
---
#### [replaced 005] Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.07615v2](http://arxiv.org/pdf/2505.07615v2)**

> **作者:** Riccardo Passoni; Francesca Ronchini; Luca Comanducci; Romain Serizel; Fabio Antonacci
>
> **备注:** Accepted at WASPAA 2025
>
> **摘要:** Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct an analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models.
>
---
#### [replaced 006] Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.13017v4](http://arxiv.org/pdf/2505.13017v4)**

> **作者:** Dang Thoai Phan; Tuan Anh Huynh; Van Tuan Pham; Cao Minh Tran; Van Thuan Mai; Ngoc Quy Tran
>
> **摘要:** The Continuous Wavelet Transform (CWT) is an effective tool for feature extraction in acoustic recognition using Convolutional Neural Networks (CNNs), particularly when applied to non-stationary audio. However, its high computational cost poses a significant challenge, often leading researchers to prefer alternative methods such as the Short-Time Fourier Transform (STFT). To address this issue, this paper proposes a method to reduce the computational complexity of CWT by optimizing the length of the wavelet kernel and the hop size of the output scalogram. Experimental results demonstrate that the proposed approach significantly reduces computational cost while maintaining the robust performance of the trained model in acoustic recognition tasks.
>
---
#### [replaced 007] JIS: A Speech Corpus of Japanese Idol Speakers with Various Speaking Styles
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18296v2](http://arxiv.org/pdf/2506.18296v2)**

> **作者:** Yuto Kondo; Hirokazu Kameoka; Kou Tanaka; Takuhiro Kaneko
>
> **备注:** Accepted on Interspeech 2025
>
> **摘要:** We construct Japanese Idol Speech Corpus (JIS) to advance research in speech generation AI, including text-to-speech synthesis (TTS) and voice conversion (VC). JIS will facilitate more rigorous evaluations of speaker similarity in TTS and VC systems since all speakers in JIS belong to a highly specific category: "young female live idols" in Japan, and each speaker is identified by a stage name, enabling researchers to recruit listeners familiar with these idols for listening experiments. With its unique speaker attributes, JIS will foster compelling research, including generating voices tailored to listener preferences-an area not yet widely studied. JIS will be distributed free of charge to promote research in speech generation AI, with usage restricted to non-commercial, basic research. We describe the construction of JIS, provide an overview of Japanese live idol culture to support effective and ethical use of JIS, and offer a basic analysis to guide application of JIS.
>
---
