# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Non-Intrusive Binaural Speech Intelligibility Prediction Using Mamba for Hearing-Impaired Listeners
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音可懂度预测任务，旨在解决传统模型计算成本高、不适合低延迟设备的问题，提出基于Mamba的模型以提升效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2507.05729v1](http://arxiv.org/pdf/2507.05729v1)**

> **作者:** Katsuhiko Yamamoto; Koichi Miyazaki
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Speech intelligibility prediction (SIP) models have been used as objective metrics to assess intelligibility for hearing-impaired (HI) listeners. In the Clarity Prediction Challenge 2 (CPC2), non-intrusive binaural SIP models based on transformers showed high prediction accuracy. However, the self-attention mechanism theoretically incurs high computational and memory costs, making it a bottleneck for low-latency, power-efficient devices. This may also degrade the temporal processing of binaural SIPs. Therefore, we propose Mamba-based SIP models instead of transformers for the temporal processing blocks. Experimental results show that our proposed SIP model achieves competitive performance compared to the baseline while maintaining a relatively small number of parameters. Our analysis suggests that the SIP model based on bidirectional Mamba effectively captures contextual and spatial speech information from binaural signals.
>
---
#### [new 002] Differentiable Reward Optimization for LLM based TTS system
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，解决传统RLHF方法依赖合成音频的问题，提出DiffRO方法直接基于神经编码器令牌计算奖励，并引入多任务奖励模型提升指令遵循能力。**

- **链接: [http://arxiv.org/pdf/2507.05911v1](http://arxiv.org/pdf/2507.05911v1)**

> **作者:** Changfeng Gao; Zhihao Du; Shiliang Zhang
>
> **摘要:** This paper proposes a novel Differentiable Reward Optimization (DiffRO) method aimed at enhancing the performance of neural codec language models based text-to-speech (TTS) systems. In contrast to conventional reinforcement learning from human feedback (RLHF) approaches applied to TTS, DiffRO directly compute the rewards based on neural codec tokens, rather than relying on synthesized audio. Furthermore, we employ the Gumbel-Softmax technique to render the reward function differentiable, thereby streamlining the RLHF training process. Additionally, we introduce a multi-task reward (MTR) model which can provide feedback from different perspectives and find that it can augment the system's capability to follow instructions effectively.Experimental results indicate that DiffRO significantly improves the pronunciation accuracy of the TTS system, achieving state-of-the-art (SOTA) WER results on the seed-tts-eval benchmark. Moreover, with the integration of the MTR model, we demonstrate the ability to control emotional and quality attributes in a zero-shot manner.
>
---
#### [new 003] Contrastive and Transfer Learning for Effective Audio Fingerprinting through a Real-World Evaluation Protocol
- **分类: cs.SD; cs.AI; cs.IR; cs.LG; eess.AS**

- **简介: 该论文属于音频指纹识别任务，解决真实场景下音频识别准确率下降的问题。通过对比学习和迁移学习提升模型鲁棒性，并设计真实评估协议验证效果。**

- **链接: [http://arxiv.org/pdf/2507.06070v1](http://arxiv.org/pdf/2507.06070v1)**

> **作者:** Christos Nikou; Theodoros Giannakopoulos
>
> **备注:** International Journal of Music Science, Technology and Art, 15 pages, 7 figures
>
> **摘要:** Recent advances in song identification leverage deep neural networks to learn compact audio fingerprints directly from raw waveforms. While these methods perform well under controlled conditions, their accuracy drops significantly in real-world scenarios where the audio is captured via mobile devices in noisy environments. In this paper, we introduce a novel evaluation protocol designed to better reflect such real-world conditions. We generate three recordings of the same audio, each with increasing levels of noise, captured using a mobile device's microphone. Our results reveal a substantial performance drop for two state-of-the-art CNN-based models under this protocol, compared to previously reported benchmarks. Additionally, we highlight the critical role of the augmentation pipeline during training with contrastive loss. By introduction low pass and high pass filters in the augmentation pipeline we significantly increase the performance of both systems in our proposed evaluation. Furthermore, we develop a transformer-based model with a tailored projection module and demonstrate that transferring knowledge from a semantically relevant domain yields a more robust solution. The transformer architecture outperforms CNN-based models across all noise levels, and query durations. In low noise conditions it achieves 47.99% for 1-sec queries, and 97% for 10-sec queries in finding the correct song, surpassing by 14%, and by 18.5% the second-best performing model, respectively, Under heavy noise levels, we achieve a detection rate 56.5% for 15-second query duration. All experiments are conducted on public large-scale dataset of over 100K songs, with queries matched against a database of 56 million vectors.
>
---
#### [new 004] Speech Quality Assessment Model Based on Mixture of Experts: System-Level Performance Enhancement and Utterance-Level Challenge Analysis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在提升模型在不同粒度上的性能。通过引入MoE架构和合成数据，解决句级评估效果有限的问题。**

- **链接: [http://arxiv.org/pdf/2507.06116v1](http://arxiv.org/pdf/2507.06116v1)**

> **作者:** Xintong Hu; Yixuan Chen; Rui Yang; Wenxiang Guo; Changhao Pan
>
> **摘要:** Automatic speech quality assessment plays a crucial role in the development of speech synthesis systems, but existing models exhibit significant performance variations across different granularity levels of prediction tasks. This paper proposes an enhanced MOS prediction system based on self-supervised learning speech models, incorporating a Mixture of Experts (MoE) classification head and utilizing synthetic data from multiple commercial generation models for data augmentation. Our method builds upon existing self-supervised models such as wav2vec2, designing a specialized MoE architecture to address different types of speech quality assessment tasks. We also collected a large-scale synthetic speech dataset encompassing the latest text-to-speech, speech conversion, and speech enhancement systems. However, despite the adoption of the MoE architecture and expanded dataset, the model's performance improvements in sentence-level prediction tasks remain limited. Our work reveals the limitations of current methods in handling sentence-level quality assessment, provides new technical pathways for the field of automatic speech quality assessment, and also delves into the fundamental causes of performance differences across different assessment granularities.
>
---
#### [new 005] Beamforming with Random Projections: Upper and Lower Bounds
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于信号处理任务，解决多阵列语音增强中的噪声与干扰抑制问题。通过随机投影实现降维和波束成形，提升SNR和SINR性能。**

- **链接: [http://arxiv.org/pdf/2507.05662v1](http://arxiv.org/pdf/2507.05662v1)**

> **作者:** Manan Mittal; Ryan M. Corey; Andrew C. Singer
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Beamformers often trade off white noise gain against the ability to suppress interferers. With distributed microphone arrays, this trade-off becomes crucial as different arrays capture vastly different magnitude and phase differences for each source. We propose the use of multiple random projections as a first-stage preprocessing scheme in a data-driven approach to dimensionality reduction and beamforming. We show that a mixture beamformer derived from the use of multiple such random projections can effectively outperform the minimum variance distortionless response (MVDR) beamformer in terms of signal-to-noise ratio (SNR) and signal-to-interferer-and-noise ratio (SINR) gain. Moreover, our method introduces computational complexity as a trade-off in the design of adaptive beamformers, alongside noise gain and interferer suppression. This added degree of freedom allows the algorithm to better exploit the inherent structure of the received signal and achieve better real-time performance while requiring fewer computations. Finally, we derive upper and lower bounds for the output power of the compressed beamformer when compared to the full complexity MVDR beamformer.
>
---
#### [new 006] Stable Acoustic Relay Assignment with High Throughput via Lase Chaos-based Reinforcement Learning
- **分类: cs.SD; cs.LG; eess.AS; math.OC**

- **简介: 该论文属于 underwater acoustic relay assignment 任务，解决稳定与高吞吐量的relay分配问题。采用激光混沌强化学习方法提升适应性和效率。**

- **链接: [http://arxiv.org/pdf/2507.05900v1](http://arxiv.org/pdf/2507.05900v1)**

> **作者:** Zengjing Chen; Lu Wang; Chengzhi Xing
>
> **摘要:** This study addresses the problem of stable acoustic relay assignment in an underwater acoustic network. Unlike the objectives of most existing literature, two distinct objectives, namely classical stable arrangement and ambiguous stable arrangement, are considered. To achieve these stable arrangements, a laser chaos-based multi-processing learning (LC-ML) method is introduced to efficiently obtain high throughput and rapidly attain stability. In order to sufficiently explore the relay's decision-making, this method uses random numbers generated by laser chaos to learn the assignment of relays to multiple source nodes. This study finds that the laser chaos-based random number and multi-processing in the exchange process have a positive effect on higher throughput and strong adaptability with environmental changing over time. Meanwhile, ambiguous cognitions result in the stable configuration with less volatility compared to accurate ones. This provides a practical and useful method and can be the basis for relay selection in complex underwater environments.
>
---
#### [new 007] Adaptive Linearly Constrained Minimum Variance Volumetric Active Noise Control
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于主动噪声控制任务，旨在解决传统方法在空间响应灵活性上的不足。通过引入LCMV ANC框架，实现特定位置的噪声抑制与空间选择性控制。**

- **链接: [http://arxiv.org/pdf/2507.05657v1](http://arxiv.org/pdf/2507.05657v1)**

> **作者:** Manan Mittal; Ryan M. Corey; Andrew C. Singer
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** Traditional volumetric noise control typically relies on multipoint error minimization to suppress sound energy across a region, but offers limited flexibility in shaping spatial responses. This paper introduces a time-domain formulation for linearly constrained minimum variance active noise control (LCMV ANC) for spatial control filter design. We demonstrate how the LCMV ANC optimization framework allows system designers to prioritize noise reduction at specific spatial locations through strategically defined linear constraints, providing a more flexible alternative to uniformly weighted multipoint error minimization. An adaptive algorithm based on filtered-X least mean squares (FxLMS) is derived for online adaptation of filter coefficients. Simulation and experimental results validate the proposed method's noise reduction and constraint adherence, demonstrating effective, spatially selective, and broadband noise control compared to multipoint volumetric noise control.
>
---
#### [new 008] Stereo Reproduction in the Presence of Sample Rate Offsets
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频同步任务，解决无线扬声器因采样率偏移导致的空间音频失真问题。通过空间滤波补偿SRO，提升听觉体验。**

- **链接: [http://arxiv.org/pdf/2507.05402v1](http://arxiv.org/pdf/2507.05402v1)**

> **作者:** Srikanth Korse; Andreas Walther; Emanuel A. P. Habets
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** One of the main challenges in synchronizing wirelessly connected loudspeakers for spatial audio reproduction is clock skew. Clock skew arises from sample rate offsets ( SROs) between the loudspeakers, caused by the use of independent device clocks. While network-based protocols like Precision Time Protocol (PTP) and Network Time Protocol (NTP) are explored, the impact of SROs on spatial audio reproduction and its perceptual consequences remains underexplored. We propose an audio-domain SRO compensation method using spatial filtering to isolate loudspeaker contributions. These filtered signals, along with the original playback signal, are used to estimate the SROs, and their influence is compensated for prior to spatial audio reproduction. We evaluate the effect of the compensation method in a subjective listening test. The results of these tests as well as objective metrics demonstrate that the proposed method mitigates the perceptual degradation introduced by SROs by preserving the spatial cues.
>
---
#### [new 009] How to Evaluate Automatic Speech Recognition: Comparing Different Performance and Bias Measures
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决如何评估ASR系统性能与偏差的问题。通过比较不同度量方法，提出更全面的评估框架。**

- **链接: [http://arxiv.org/pdf/2507.05885v1](http://arxiv.org/pdf/2507.05885v1)**

> **作者:** Tanvina Patel; Wiebke Hutiri; Aaron Yi Ding; Odette Scharenborg
>
> **摘要:** There is increasingly more evidence that automatic speech recognition (ASR) systems are biased against different speakers and speaker groups, e.g., due to gender, age, or accent. Research on bias in ASR has so far primarily focused on detecting and quantifying bias, and developing mitigation approaches. Despite this progress, the open question is how to measure the performance and bias of a system. In this study, we compare different performance and bias measures, from literature and proposed, to evaluate state-of-the-art end-to-end ASR systems for Dutch. Our experiments use several bias mitigation strategies to address bias against different speaker groups. The findings reveal that averaged error rates, a standard in ASR research, alone is not sufficient and should be supplemented by other measures. The paper ends with recommendations for reporting ASR performance and bias to better represent a system's performance for diverse speaker groups, and overall system bias.
>
---
#### [new 010] Self-supervised Deep Learning for Denoising in Ultrasound Microvascular Imaging
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于医学图像处理任务，旨在解决超声微血管成像中的噪声问题。通过自监督学习方法HA2HA提升图像质量，适用于对比剂和无对比剂场景。**

- **链接: [http://arxiv.org/pdf/2507.05451v1](http://arxiv.org/pdf/2507.05451v1)**

> **作者:** Lijie Huang; Jingyi Yin; Jingke Zhang; U-Wai Lok; Ryan M. DeRuiter; Jieyang Jin; Kate M. Knoll; Kendra E. Petersen; James D. Krier; Xiang-yang Zhu; Gina K. Hesley; Kathryn A. Robinson; Andrew J. Bentall; Thomas D. Atwell; Andrew D. Rule; Lilach O. Lerman; Shigao Chen; Chengwu Huang
>
> **备注:** 12 pages, 10 figures. Supplementary materials are available at https://zenodo.org/records/15832003
>
> **摘要:** Ultrasound microvascular imaging (UMI) is often hindered by low signal-to-noise ratio (SNR), especially in contrast-free or deep tissue scenarios, which impairs subsequent vascular quantification and reliable disease diagnosis. To address this challenge, we propose Half-Angle-to-Half-Angle (HA2HA), a self-supervised denoising framework specifically designed for UMI. HA2HA constructs training pairs from complementary angular subsets of beamformed radio-frequency (RF) blood flow data, across which vascular signals remain consistent while noise varies. HA2HA was trained using in-vivo contrast-free pig kidney data and validated across diverse datasets, including contrast-free and contrast-enhanced data from pig kidneys, as well as human liver and kidney. An improvement exceeding 15 dB in both contrast-to-noise ratio (CNR) and SNR was observed, indicating a substantial enhancement in image quality. In addition to power Doppler imaging, denoising directly in the RF domain is also beneficial for other downstream processing such as color Doppler imaging (CDI). CDI results of human liver derived from the HA2HA-denoised signals exhibited improved microvascular flow visualization, with a suppressed noisy background. HA2HA offers a label-free, generalizable, and clinically applicable solution for robust vascular imaging in both contrast-free and contrast-enhanced UMI.
>
---
#### [new 011] ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统ASR模型在上下文理解上的不足。提出ContextASR-Bench基准，评估模型在多领域上下文场景下的表现。**

- **链接: [http://arxiv.org/pdf/2507.05727v1](http://arxiv.org/pdf/2507.05727v1)**

> **作者:** He Wang; Linhan Ma; Dake Guo; Xiong Wang; Lei Xie; Jin Xu; Junyang Lin
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Automatic Speech Recognition (ASR) has been extensively investigated, yet prior evaluative efforts have largely been restricted to contextless paradigms. This constraint stems from the limited proficiency of conventional ASR models in context modeling and their deficiency in memory and reasoning based on world knowledge. Recent breakthroughs in the development of Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of general artificial intelligence capabilities. Consequently, there exists a compelling need for a benchmark that can evaluate both the generality and intelligence of ASR systems. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess contextual speech recognition. This benchmark encompasses up to 40,000 data entries across over 10 domains, enabling a thorough evaluation of model performance in scenarios that omit or incorporate coarse-grained or fine-grained contextual information. Moreover, diverging from conventional ASR evaluations, our benchmark includes an analysis of model efficacy in recognizing named entities mentioned within the auditory input. Our extensive evaluation highlights that LALMs, with strong world knowledge and context learning capabilities, outperform conventional ASR models by a large margin. The dataset and evaluation code have been released at https://github.com/MrSupW/ContextASR-Bench.
>
---
#### [new 012] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决MoE模型中专家协作不足的问题。通过引入共享路由器增强不同层专家的合作与专业化，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.05724v1](http://arxiv.org/pdf/2507.05724v1)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model \emph{Omni-router Transformer}. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [new 013] Robust One-step Speech Enhancement via Consistency Distillation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，解决扩散模型实时性差的问题。通过一致性蒸馏构建单步模型，提升噪声鲁棒性和性能。**

- **链接: [http://arxiv.org/pdf/2507.05688v1](http://arxiv.org/pdf/2507.05688v1)**

> **作者:** Liang Xu; Longfei Felix Yan; W. Bastiaan Kleijn
>
> **备注:** Accepted to IEEE WASPAA 2025. 6 pages, 1 figures
>
> **摘要:** Diffusion models have shown strong performance in speech enhancement, but their real-time applicability has been limited by multi-step iterative sampling. Consistency distillation has recently emerged as a promising alternative by distilling a one-step consistency model from a multi-step diffusion-based teacher model. However, distilled consistency models are inherently biased towards the sampling trajectory of the teacher model, making them less robust to noise and prone to inheriting inaccuracies from the teacher model. To address this limitation, we propose ROSE-CD: Robust One-step Speech Enhancement via Consistency Distillation, a novel approach for distilling a one-step consistency model. Specifically, we introduce a randomized learning trajectory to improve the model's robustness to noise. Furthermore, we jointly optimize the one-step model with two time-domain auxiliary losses, enabling it to recover from teacher-induced errors and surpass the teacher model in overall performance. This is the first pure one-step consistency distillation model for diffusion-based speech enhancement, achieving 54 times faster inference speed and superior performance compared to its 30-step teacher model. Experiments on the VoiceBank-DEMAND dataset demonstrate that the proposed model achieves state-of-the-art performance in terms of speech quality. Moreover, its generalization ability is validated on both an out-of-domain dataset and real-world noisy recordings.
>
---
#### [new 014] Comparative Analysis of Finite Difference and Finite Element Method for Audio Waveform Simulation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于数值方法比较任务，旨在分析FEM与FDM在音频波形模拟中的表现，通过吉他弦和钟声仿真验证两者精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.05396v1](http://arxiv.org/pdf/2507.05396v1)**

> **作者:** Juliette Florin
>
> **备注:** 46 pages, 38 figures, Link to the source code: https://gitlab.eurecom.fr/florin/models
>
> **摘要:** In many industries, including aerospace and defense, waveform analysis is commonly conducted to compute the resonance of physical objects, with the Finite Element Method (FEM) being the standard approach. The Finite Difference Method (FDM) is seldom used, and this preference is often stated without formal justification in the literature. In this work, the accuracy, feasibility, and time of simulation of FEM and FDM are compared by simulating the vibration of a guitar string. Python simulations for both methods are implemented, and their results are compared against analytical solutions and experimental data. Additionally, FDM is applied to analyze the sound of a cycling bell to assess its reliability compared to a real cycling bell. Final results show that both FEM and FDM yield similar error margins and accurately predict the system's behavior. Moreover, the errors from FEM and FDM follow the same periodicity with a phase shift when varying the assumed analytical tension and without a phase shift when changing the time interval. However, FEM converges faster with increasing mesh complexity, whereas FDM demonstrates quicker computational performance and achieves stable solutions even with bigger time intervals. Despite this FDM is limited to simpler configurations and often demands extensive mathematical formulation, which can become cumbersome for intricate shapes. For example, modeling a hemispherical object using FDM results in significant simulation times and big calculations. In conclusion, while FDM may offer faster convergence and computation time in certain cases, FEM remains the preferred method in industrial contexts due to its flexibility, scalability, and ease of implementation for complex geometries.
>
---
#### [new 015] Frequency-Specific Neural Response and Cross-Correlation Analysis of Envelope Following Responses to Native Speech and Music Using Multichannel EEG Signals: A Case Study
- **分类: eess.AS; cs.SD; cs.SY; eess.SP; eess.SY**

- **简介: 该论文属于神经信号分析任务，旨在研究母语语音和音乐的脑电响应特性。通过分析多通道EEG数据，揭示了不同频率下的神经响应及空间相干性，以理解听觉处理机制。**

- **链接: [http://arxiv.org/pdf/2507.05635v1](http://arxiv.org/pdf/2507.05635v1)**

> **作者:** Md. Mahbub Hasan; Md Rakibul Hasan; Md Zakir Hossain; Tom Gedeon
>
> **摘要:** Although native speech and music envelope following responses (EFRs) play a crucial role in auditory processing and cognition, their frequency profile, such as the dominating frequency and spectral coherence, is largely unknown. We have assumed that the auditory pathway - which transmits envelope components of speech and music to the scalp through time-varying neurophysiological processes - is a linear time-varying system, with the envelope and the multi-channel EEG responses as excitation and response, respectively. This paper investigates the transfer function of this system through two analytical techniques - time-averaged spectral responses and cross-spectral density - in the frequency domain at four different positions of the human scalp. Our findings suggest that alpha (8-11 Hz), lower gamma (53-56 Hz), and higher gamma (78-81 Hz) bands are the peak responses of the system. These frequently appearing dominant frequency responses may be the key components of familiar speech perception, maintaining attention, binding acoustic features, and memory processing. The cross-spectral density, which reflects the spatial neural coherence of the human brain, shows that 10-13 Hz, 27-29 Hz, and 62-64 Hz are common for all channel pairs. As neural coherences are frequently observed in these frequencies among native participants, we suggest that these distributed neural processes are also dominant in native speech and music perception.
>
---
## 更新

#### [replaced 001] What's Making That Sound Right Now? Video-centric Audio-Visual Localization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.04667v2](http://arxiv.org/pdf/2507.04667v2)**

> **作者:** Hahyeon Choi; Junhoo Lee; Nojun Kwak
>
> **备注:** Published at ICCV 2025. Project page: https://hahyeon610.github.io/Video-centric_Audio_Visual_Localization/
>
> **摘要:** Audio-Visual Localization (AVL) aims to identify sound-emitting sources within a visual scene. However, existing studies focus on image-level audio-visual associations, failing to capture temporal dynamics. Moreover, they assume simplified scenarios where sound sources are always visible and involve only a single object. To address these limitations, we propose AVATAR, a video-centric AVL benchmark that incorporates high-resolution temporal information. AVATAR introduces four distinct scenarios -- Single-sound, Mixed-sound, Multi-entity, and Off-screen -- enabling a more comprehensive evaluation of AVL models. Additionally, we present TAVLO, a novel video-centric AVL model that explicitly integrates temporal information. Experimental results show that conventional methods struggle to track temporal variations due to their reliance on global audio features and frame-level mappings. In contrast, TAVLO achieves robust and precise audio-visual alignment by leveraging high-resolution temporal modeling. Our work empirically demonstrates the importance of temporal dynamics in AVL and establishes a new standard for video-centric audio-visual localization.
>
---
#### [replaced 002] OpenS2S: Advancing Fully Open-Source End-to-End Empathetic Large Speech Language Model
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05177v2](http://arxiv.org/pdf/2507.05177v2)**

> **作者:** Chen Wang; Tianyu Peng; Wen Yang; Yinan Bai; Guangfu Wang; Jun Lin; Lanpeng Jia; Lingxiang Wu; Jinqiao Wang; Chengqing Zong; Jiajun Zhang
>
> **备注:** Technical Report
>
> **摘要:** Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at https://casia-lm.github.io/OpenS2S
>
---
#### [replaced 003] ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11079v2](http://arxiv.org/pdf/2505.11079v2)**

> **作者:** Hao Gu; Jiangyan Yi; Chenglong Wang; Jianhua Tao; Zheng Lian; Jiayi He; Yong Ren; Yujie Chen; Zhengqi Wen
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Audio deepfake detection (ADD) has grown increasingly important due to the rise of high-fidelity audio generative models and their potential for misuse. Given that audio large language models (ALLMs) have made significant progress in various audio processing tasks, a heuristic question arises: \textit{Can ALLMs be leveraged to solve ADD?}. In this paper, we first conduct a comprehensive zero-shot evaluation of ALLMs on ADD, revealing their ineffectiveness. To this end, we propose ALLM4ADD, an ALLM-driven framework for ADD. Specifically, we reformulate ADD task as an audio question answering problem, prompting the model with the question: ``Is this audio fake or real?''. We then perform supervised fine-tuning to enable the ALLM to assess the authenticity of query audio. Extensive experiments are conducted to demonstrate that our ALLM-based method can achieve superior performance in fake audio detection, particularly in data-scarce scenarios. As a pioneering study, we anticipate that this work will inspire the research community to leverage ALLMs to develop more effective ADD systems. Code is available at https://github.com/ucas-hao/qwen_audio_for_add.git
>
---
#### [replaced 004] MARS: Radio Map Super-resolution and Reconstruction Method under Sparse Channel Measurements
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.04682v3](http://arxiv.org/pdf/2506.04682v3)**

> **作者:** Chuyun Deng; Na Liu; Wei Xie; Lianming Xu; Li Wang
>
> **备注:** The authors withdraw this submission to substantially revise the introduction and experimental sections and incorporate new content. The manuscript has not been submitted or published elsewhere. A revised version may be submitted in the future
>
> **摘要:** Radio maps reflect the spatial distribution of signal strength and are essential for applications like smart cities, IoT, and wireless network planning. However, reconstructing accurate radio maps from sparse measurements remains challenging. Traditional interpolation and inpainting methods lack environmental awareness, while many deep learning approaches depend on detailed scene data, limiting generalization. To address this, we propose MARS, a Multi-scale Aware Radiomap Super-resolution method that combines CNNs and Transformers with multi-scale feature fusion and residual connections. MARS focuses on both global and local feature extraction, enhancing feature representation across different receptive fields and improving reconstruction accuracy. Experiments across different scenes and antenna locations show that MARS outperforms baseline models in both MSE and SSIM, while maintaining low computational cost, demonstrating strong practical potential.
>
---
#### [replaced 005] Evaluating Logit-Based GOP Scores for Mispronunciation Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.12067v2](http://arxiv.org/pdf/2506.12067v2)**

> **作者:** Aditya Kamlesh Parikh; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** Accepted to Interspeech 2025. This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** Pronunciation assessment relies on goodness of pronunciation (GOP) scores, traditionally derived from softmax-based posterior probabilities. However, posterior probabilities may suffer from overconfidence and poor phoneme separation, limiting their effectiveness. This study compares logit-based GOP scores with probability-based GOP scores for mispronunciation detection. We conducted our experiment on two L2 English speech datasets spoken by Dutch and Mandarin speakers, assessing classification performance and correlation with human ratings. Logit-based methods outperform probability-based GOP in classification, but their effectiveness depends on dataset characteristics. The maximum logit GOP shows the strongest alignment with human perception, while a combination of different GOP scores balances probability and logit features. The findings suggest that hybrid GOP methods incorporating uncertainty modeling and phoneme-specific weighting improve pronunciation assessment.
>
---
#### [replaced 006] Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.17380v3](http://arxiv.org/pdf/2502.17380v3)**

> **作者:** Qiuming Zhao; Guangzhi Sun; Chao Zhang
>
> **备注:** 13 pages
>
> **摘要:** Language diversity presents a significant challenge in speech-to-text (S2T) tasks, such as automatic speech recognition and translation. Traditional multi-lingual multi-task training approaches aim to address this by jointly optimising multiple speech recognition and translation tasks across various languages. While models like Whisper, built on these strategies, demonstrate strong performance, they still face issues of high computational cost, language interference, suboptimal training configurations, and limited extensibility. To overcome these challenges, we introduce LoRS-Merging (low-rank and sparse model merging), a novel technique designed to efficiently integrate models trained on different languages or tasks while preserving performance and reducing computational overhead. LoRS-Merging combines low-rank and sparse pruning to retain essential structures while eliminating redundant parameters, mitigating language interference, and enhancing extensibility. Experimental results across 10 languages demonstrate that LoRS-Merging significantly outperforms multi-lingual multi-task training, sequential training, and other merging methods, achieving over 20% improvement in normalised performance. Our findings suggest that model merging, particularly LoRS-Merging, is a scalable and effective complement to traditional multi-lingual training strategies for S2T applications.
>
---
#### [replaced 007] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation
- **分类: cs.CL; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2403.04945v4](http://arxiv.org/pdf/2403.04945v4)**

> **作者:** Zhongwei Wan; Che Liu; Xin Wang; Chaofan Tao; Hui Shen; Jing Xiong; Rossella Arcucci; Huaxiu Yao; Mi Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Electrocardiogram (ECG) is the primary non-invasive diagnostic tool for monitoring cardiac conditions and is crucial in assisting clinicians. Recent studies have concentrated on classifying cardiac conditions using ECG data but have overlooked ECG report generation, which is time-consuming and requires clinical expertise. To automate ECG report generation and ensure its versatility, we propose the Multimodal ECG Instruction Tuning (MEIT) framework, the first attempt to tackle ECG report generation with LLMs and multimodal instructions. To facilitate future research, we establish a benchmark to evaluate MEIT with various LLMs backbones across two large-scale ECG datasets. Our approach uniquely aligns the representations of the ECG signal and the report, and we conduct extensive experiments to benchmark MEIT with nine open-source LLMs using more than 800,000 ECG reports. MEIT's results underscore the superior performance of instruction-tuned LLMs, showcasing their proficiency in quality report generation, zero-shot capabilities, resilience to signal perturbation, and alignment with human expert evaluation. These findings emphasize the efficacy of MEIT and its potential for real-world clinical application.
>
---
#### [replaced 008] Self-supervised learning of speech representations with Dutch archival data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.04554v2](http://arxiv.org/pdf/2507.04554v2)**

> **作者:** Nik Vaessen; Roeland Ordelman; David A. van Leeuwen
>
> **备注:** accepted at interspeech 2025
>
> **摘要:** This paper explores the use of Dutch archival television broadcast data for self-supervised learning of speech foundation models, specifically wav2vec 2.0. We first study data quality assumptions for pre-training, and show how music, noise and speaker overlap affect SSL convergence and downstream fine-tuning performance. Secondly, we explore effectively pre-processing strategies to convert the noisy broadcast dataset into a qualitative dataset for pre-training, by using Whisper and WhisperX. Thirdly, we compare mono-lingual and multi-lingual pre-training with equivalent amounts of data, and show that mono-lingual pre-training is more robust to out-of-domain data. Lastly, we achieve a state-of-the-art LARGE wav2vec 2.0 model for the Dutch language, by a continuation of pre-training a wav2vec 2.0 XLS-R model checkpoint with our 55k hour archival dataset.
>
---
#### [replaced 009] SpeechAccentLLM: A Unified Framework for Foreign Accent Conversion and Text to Speech
- **分类: eess.AS; cs.SD; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.01348v2](http://arxiv.org/pdf/2507.01348v2)**

> **作者:** Zhuangfei Cheng; Guangyan Zhang; Zehai Tu; Yangyang Song; Shuiyang Mao; Xiaoqi Jiao; Jingyu Li; Yiwen Guo; Jiasong Wu
>
> **备注:** 10 pages, includes references, 4 figures, 4 tables
>
> **摘要:** Foreign accent conversion (FAC) in speech processing remains a challenging task. Building on the remarkable success of large language models (LLMs) in Text-to-Speech (TTS) tasks, this study investigates the adaptation of LLM-based techniques for FAC, which we term SpeechAccentLLM. At the core of this framework, we introduce SpeechCodeVAE, the first model to integrate connectionist temporal classification (CTC) directly into codebook discretization for speech content tokenization. This novel architecture generates tokens with a unique "locality" property, as validated by experiments demonstrating optimal trade-offs among content faithfulness, temporal coherence, and structural recoverability. Then, to address data scarcity for the FAC module, we adopted a multitask learning strategy that jointly trains the FAC and TTS modules. Beyond mitigating data limitations, this approach yielded accelerated convergence and superior speech quality compared to standalone FAC training. Moreover, leveraging the salient properties of our discrete speech representations, we introduce SpeechRestorer, a postprocessing architecture designed to refine LLM-generated outputs. This module effectively mitigates stochastic errors prevalent in LLM inference pipelines while enhancing prosodic continuity, as validated by ablation experiments.
>
---
#### [replaced 010] S2ST-Omni: An Efficient Multilingual Speech-to-Speech Translation Framework via Seamless Speech-Text Alignment and Progressive Fine-tuning
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.11160v5](http://arxiv.org/pdf/2506.11160v5)**

> **作者:** Yu Pan; Yuguang Yang; Yanni Hu; Jianhao Ye; Xiang Zhang; Hongbin Zhou; Lei Ma; Jianjun Zhao
>
> **备注:** Working in progress
>
> **摘要:** Despite recent advances in multilingual speech-to-speech translation (S2ST), several critical challenges persist: 1) achieving high-quality translation remains a major hurdle, and 2) most existing methods heavily rely on large-scale parallel speech corpora, which are costly and difficult to obtain. To address these issues, we propose \textit{S2ST-Omni}, an efficient and scalable framework for multilingual S2ST. Specifically, we decompose the S2ST task into speech-to-text translation (S2TT) and text-to-speech synthesis (TTS). For S2TT, we propose an effective speech language model that integrates the pretrained Whisper encoder for robust audio understanding and Qwen 3.0 for advanced text comprehension. A lightweight speech adapter is employed to bridge the modality gap between speech and text representations. To further facilitate the multimodal knowledge learning, a two-stage fine-tuning strategy is introduced. In the TTS stage, we adopt a streaming autoregressive generation approach to produce natural and fluent target speech. Experiments on the CVSS benchmark show that S2ST-Omni consistently outperforms existing state-of-the-art S2ST systems in translation quality, highlighting its effectiveness and superiority.
>
---
