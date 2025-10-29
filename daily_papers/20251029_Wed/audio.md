# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Sound Source Localization for Spatial Mapping of Surgical Actions in Dynamic Scenes
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文属于多模态手术场景理解任务，旨在解决动态手术环境中细粒度上下文建模难题。通过融合相控阵麦克风与RGB-D相机数据，提出4D音视频表示框架，实现手术工具-组织交互的3D声源定位与时空关联，提升手术活动的动态感知能力。**

- **链接: [http://arxiv.org/pdf/2510.24332v1](http://arxiv.org/pdf/2510.24332v1)**

> **作者:** Jonas Hein; Lazaros Vlachopoulos; Maurits Geert Laurent Olthof; Bastian Sigrist; Philipp Fürnstahl; Matthias Seibold
>
> **摘要:** Purpose: Surgical scene understanding is key to advancing computer-aided and intelligent surgical systems. Current approaches predominantly rely on visual data or end-to-end learning, which limits fine-grained contextual modeling. This work aims to enhance surgical scene representations by integrating 3D acoustic information, enabling temporally and spatially aware multimodal understanding of surgical environments. Methods: We propose a novel framework for generating 4D audio-visual representations of surgical scenes by projecting acoustic localization information from a phased microphone array onto dynamic point clouds from an RGB-D camera. A transformer-based acoustic event detection module identifies relevant temporal segments containing tool-tissue interactions which are spatially localized in the audio-visual scene representation. The system was experimentally evaluated in a realistic operating room setup during simulated surgical procedures performed by experts. Results: The proposed method successfully localizes surgical acoustic events in 3D space and associates them with visual scene elements. Experimental evaluation demonstrates accurate spatial sound localization and robust fusion of multimodal data, providing a comprehensive, dynamic representation of surgical activity. Conclusion: This work introduces the first approach for spatial sound localization in dynamic surgical scenes, marking a significant advancement toward multimodal surgical scene representations. By integrating acoustic and visual data, the proposed framework enables richer contextual understanding and provides a foundation for future intelligent and autonomous surgical systems.
>
---
#### [new 002] HergNet: a Fast Neural Surrogate Model for Sound Field Predictions via Superposition of Plane Waves
- **分类: cs.SD; cs.CE; cs.LG; eess.AS**

- **简介: 该论文提出HergNet，一种基于平面波叠加的快速神经代理模型，用于高效预测二维和三维声场。通过自动满足赫姆霍兹方程，确保物理合理性，解决复杂波现象中边界值问题的求解效率与精度难题，显著提升中高频房间声学模拟性能。**

- **链接: [http://arxiv.org/pdf/2510.24279v1](http://arxiv.org/pdf/2510.24279v1)**

> **作者:** Matteo Calafà; Yuanxin Xia; Cheol-Ho Jeong
>
> **摘要:** We present a novel neural network architecture for the efficient prediction of sound fields in two and three dimensions. The network is designed to automatically satisfy the Helmholtz equation, ensuring that the outputs are physically valid. Therefore, the method can effectively learn solutions to boundary-value problems in various wave phenomena, such as acoustics, optics, and electromagnetism. Numerical experiments show that the proposed strategy can potentially outperform state-of-the-art methods in room acoustics simulation, in particular in the range of mid to high frequencies.
>
---
#### [new 003] A Neural Model for Contextual Biasing Score Learning and Filtering
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对语音识别中的上下文偏置任务，提出一种基于注意力机制的评分模型，通过判别性目标提升真实短语得分并抑制干扰项。方法可过滤候选短语，并用于浅层融合，显著提升识别准确率，且兼容任意ASR系统。**

- **链接: [http://arxiv.org/pdf/2510.23849v1](http://arxiv.org/pdf/2510.23849v1)**

> **作者:** Wanting Huang; Weiran Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Contextual biasing improves automatic speech recognition (ASR) by integrating external knowledge, such as user-specific phrases or entities, during decoding. In this work, we use an attention-based biasing decoder to produce scores for candidate phrases based on acoustic information extracted by an ASR encoder, which can be used to filter out unlikely phrases and to calculate bonus for shallow-fusion biasing. We introduce a per-token discriminative objective that encourages higher scores for ground-truth phrases while suppressing distractors. Experiments on the Librispeech biasing benchmark show that our method effectively filters out majority of the candidate phrases, and significantly improves recognition accuracy under different biasing conditions when the scores are used in shallow fusion biasing. Our approach is modular and can be used with any ASR system, and the filtering mechanism can potentially boost performance of other biasing methods.
>
---
#### [new 004] Listening without Looking: Modality Bias in Audio-Visual Captioning
- **分类: eess.AS; cs.CV; eess.IV**

- **简介: 该论文研究音频-视觉字幕生成任务，针对现有模型对音频模态的偏倚问题，通过消融实验和新数据集AudioVisualCaps评估模态互补性与鲁棒性，发现并缓解了模型对音频的过度依赖。**

- **链接: [http://arxiv.org/pdf/2510.24024v1](http://arxiv.org/pdf/2510.24024v1)**

> **作者:** Yuchi Ishikawa; Toranosuke Manabe; Tatsuya Komatsu; Yoshimitsu Aoki
>
> **备注:** under review
>
> **摘要:** Audio-visual captioning aims to generate holistic scene descriptions by jointly modeling sound and vision. While recent methods have improved performance through sophisticated modality fusion, it remains unclear to what extent the two modalities are complementary in current audio-visual captioning models and how robust these models are when one modality is degraded. We address these questions by conducting systematic modality robustness tests on LAVCap, a state-of-the-art audio-visual captioning model, in which we selectively suppress or corrupt the audio or visual streams to quantify sensitivity and complementarity. The analysis reveals a pronounced bias toward the audio stream in LAVCap. To evaluate how balanced audio-visual captioning models are in their use of both modalities, we augment AudioCaps with textual annotations that jointly describe the audio and visual streams, yielding the AudioVisualCaps dataset. In our experiments, we report LAVCap baseline results on AudioVisualCaps. We also evaluate the model under modality robustness tests on AudioVisualCaps and the results indicate that LAVCap trained on AudioVisualCaps exhibits less modality bias than when trained on AudioCaps.
>
---
#### [new 005] Online neural fusion of distortionless differential beamformers for robust speech enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决固定波束成形在动态声学环境中干扰抑制能力弱的问题。针对传统自适应凸组合算法难以跟踪快速变化干扰的缺陷，提出一种在线神经融合框架，通过神经网络实时估计多个无失真差分波束成形器的权重，实现更优的鲁棒性与干扰抑制性能。**

- **链接: [http://arxiv.org/pdf/2510.24497v1](http://arxiv.org/pdf/2510.24497v1)**

> **作者:** Yuanhang Qian; Kunlong Zhao; Jilu Jin; Xueqin Luo; Gongping Huang; Jingdong Chen; Jacob Benesty
>
> **摘要:** Fixed beamforming is widely used in practice since it does not depend on the estimation of noise statistics and provides relatively stable performance. However, a single beamformer cannot adapt to varying acoustic conditions, which limits its interference suppression capability. To address this, adaptive convex combination (ACC) algorithms have been introduced, where the outputs of multiple fixed beamformers are linearly combined to improve robustness. Nevertheless, ACC often fails in highly non-stationary scenarios, such as rapidly moving interference, since its adaptive updates cannot reliably track rapid changes. To overcome this limitation, we propose a frame-online neural fusion framework for multiple distortionless differential beamformers, which estimates the combination weights through a neural network. Compared with conventional ACC, the proposed method adapts more effectively to dynamic acoustic environments, achieving stronger interference suppression while maintaining the distortionless constraint.
>
---
#### [new 006] Bayesian Speech synthesizers Can Learn from Multiple Teachers
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出BELLE，一种基于贝叶斯证据学习的连续值自回归语音合成框架，旨在解决多教师语音合成中多样性建模与不确定性估计难题。通过从多个预训练模型生成的并行数据中学习，实现高质量语音合成，仅用少量真实数据即达先进水平。**

- **链接: [http://arxiv.org/pdf/2510.24372v1](http://arxiv.org/pdf/2510.24372v1)**

> **作者:** Ziyang Zhang; Yifan Gao; Xuenan Xu; Baoxiangli; Wen Wu; Chao Zhang
>
> **摘要:** Codec-based text-to-speech (TTS) models have recently gained traction for their efficiency and strong performance in voice cloning. However, codec-based TTS faces limitations due to the challenges of pretraining robust speech codecs and the quality degradation introduced by quantization errors. Emerging evidence suggests that continuous-valued generative models can alleviate these issues and serve as a promising alternative. Yet, effectively modelling diverse speech patterns and developing reliable sampling strategies for continuous-valued autoregressive (AR) TTS remains underexplored. In this work, we propose BELLE, Bayesian evidential learning with language modelling for TTS, a novel continuous-valued AR framework that directly predicts mel-spectrograms from textual input. BELLE treats each mel-spectrogram frame as a Gaussian distribution sampled from a learned hyper distribution, enabling principled uncertainty estimation, particularly in scenarios with parallel data (i.e., one text-audio prompt paired with multiple speech samples). To obtain such data, diverse speech samples are synthesized using multiple pre-trained TTS models given the same text-audio prompts, which are distilled into BELLE via Bayesian evidential learning. Experimental results indicate that BELLE demonstrates highly competitive performance compared with the current best open-source TTS models, even though BELLE is trained on a large amount of synthetic data and uses only approximately one-tenth of their training data. Audio samples generated by BELLE are available at https://belletts.github.io/Belle/. The code, checkpoints, and synthetic data will be released after the paper is accepted.
>
---
#### [new 007] Optimized Loudspeaker Panning for Adaptive Sound-Field Correction and Non-stationary Listening Areas
- **分类: cs.SD; eess.AS; eess.SP; math.OC**

- **简介: 该论文针对非标准布局下环绕声系统的声音场失真问题，提出基于贝叶斯估计的扬声器归一化与内容混音优化方法。通过动态更新听音位置，自适应调整扬声器响应并优化频域混音系数，在无需声学测量情况下实现标准化声场再现，提升音色、定位与清晰度。**

- **链接: [http://arxiv.org/pdf/2510.23937v1](http://arxiv.org/pdf/2510.23937v1)**

> **作者:** Yuancheng Luo
>
> **摘要:** Surround sound systems commonly distribute loudspeakers along standardized layouts for multichannel audio reproduction. However in less controlled environments, practical layouts vary in loudspeaker quantity, placement, and listening locations / areas. Deviations from standard layouts introduce sound-field errors that degrade acoustic timbre, imaging, and clarity of audio content reproduction. This work introduces both Bayesian loudspeaker normalization and content panning optimization methods for sound-field correction. Conjugate prior distributions over loudspeaker-listener directions update estimated layouts for non-stationary listening locations; digital filters adapt loudspeaker acoustic responses to a common reference target at the estimated listening area without acoustic measurements. Frequency-domain panning coefficients are then optimized via sensitivity / efficiency objectives subject to spatial, electrical, and acoustic domain constraints; normalized and panned loudspeakers form virtual loudspeakers in standardized layouts for accurate multichannel reproduction. Experiments investigate robustness of Bayesian adaptation, and panning optimizations in practical applications.
>
---
#### [new 008] TsetlinKWS: A 65nm 16.58uW, 0.63mm2 State-Driven Convolutional Tsetlin Machine-Based Accelerator For Keyword Spotting
- **分类: cs.SD; cs.AR; eess.AS; B.7; C.3; I.2**

- **简介: 该论文针对语音关键词识别任务，提出TsetlinKWS框架，解决传统卷积Tsetlin机在语音任务中精度低、模型大、能效差的问题。通过创新特征提取、压缩算法与状态驱动硬件架构，实现高精度、低功耗（16.58μW）和小面积（0.63mm²）的高效推理。**

- **链接: [http://arxiv.org/pdf/2510.24282v1](http://arxiv.org/pdf/2510.24282v1)**

> **作者:** Baizhou Lin; Yuetong Fang; Renjing Xu; Rishad Shafik; Jagmohan Chauhan
>
> **备注:** 12 pages, 17 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** The Tsetlin Machine (TM) has recently attracted attention as a low-power alternative to neural networks due to its simple and interpretable inference mechanisms. However, its performance on speech-related tasks remains limited. This paper proposes TsetlinKWS, the first algorithm-hardware co-design framework for the Convolutional Tsetlin Machine (CTM) on the 12-keyword spotting task. Firstly, we introduce a novel Mel-Frequency Spectral Coefficient and Spectral Flux (MFSC-SF) feature extraction scheme together with spectral convolution, enabling the CTM to reach its first-ever competitive accuracy of 87.35% on the 12-keyword spotting task. Secondly, we develop an Optimized Grouped Block-Compressed Sparse Row (OG-BCSR) algorithm that achieves a remarkable 9.84$\times$ reduction in model size, significantly improving the storage efficiency on CTMs. Finally, we propose a state-driven architecture tailored for the CTM, which simultaneously exploits data reuse and sparsity to achieve high energy efficiency. The full system is evaluated in 65 nm process technology, consuming 16.58 $\mu$W at 0.7 V with a compact 0.63 mm$^2$ core area. TsetlinKWS requires only 907k logic operations per inference, representing a 10$\times$ reduction compared to the state-of-the-art KWS accelerators, positioning the CTM as a highly-efficient candidate for ultra-low-power speech applications.
>
---
#### [new 009] STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出STAR-Bench，用于评估音频4D智能——对时间与三维空间中声音动态的细粒度推理能力。针对现有基准依赖文本描述、忽略感知推理的问题，构建包含基础感知与综合时空推理的评测体系，结合物理模拟与人工标注数据，揭示模型在感知与推理上的显著差距，推动更鲁棒的物理世界理解模型发展。**

- **链接: [http://arxiv.org/pdf/2510.24693v1](http://arxiv.org/pdf/2510.24693v1)**

> **作者:** Zihan Liu; Zhikang Niu; Qiuyang Xiao; Zhisheng Zheng; Ruoqi Yuan; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Jianze Liang; Xie Chen; Leilei Sun; Dahua Lin; Jiaqi Wang
>
> **备注:** Homepage: https://internlm.github.io/StarBench/
>
> **摘要:** Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.
>
---
#### [new 010] Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出MGAudio框架，解决开放域视频到音频生成任务中的跨模态一致性与音频真实性问题。通过模型引导的双角色对齐机制，使音频-视觉编码器同时充当条件输入与特征对齐模块，结合流式Transformer模型与自指导目标，显著提升生成质量，在多个基准上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.24103v1](http://arxiv.org/pdf/2510.24103v1)**

> **作者:** Kang Zhang; Trung X. Pham; Suyeon Lee; Axi Niu; Arda Senocak; Joon Son Chung
>
> **备注:** accepted by NeurIPS 2025
>
> **摘要:** We present MGAudio, a novel flow-based framework for open-domain video-to-audio generation, which introduces model-guided dual-role alignment as a central design principle. Unlike prior approaches that rely on classifier-based or classifier-free guidance, MGAudio enables the generative model to guide itself through a dedicated training objective designed for video-conditioned audio generation. The framework integrates three main components: (1) a scalable flow-based Transformer model, (2) a dual-role alignment mechanism where the audio-visual encoder serves both as a conditioning module and as a feature aligner to improve generation quality, and (3) a model-guided objective that enhances cross-modal coherence and audio realism. MGAudio achieves state-of-the-art performance on VGGSound, reducing FAD to 0.40, substantially surpassing the best classifier-free guidance baselines, and consistently outperforms existing methods across FD, IS, and alignment metrics. It also generalizes well to the challenging UnAV-100 benchmark. These results highlight model-guided dual-role alignment as a powerful and scalable paradigm for conditional video-to-audio generation. Code is available at: https://github.com/pantheon5100/mgaudio
>
---
#### [new 011] Audio Signal Processing Using Time Domain Mel-Frequency Wavelet Coefficient
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对语音信号处理中特征提取问题，提出时间域梅尔频域小波系数（TMFWC）方法。旨在结合MFCC与小波变换优势，解决传统MFCC缺乏时序信息、小波变换频率分辨率不足的问题。通过在时域直接构建梅尔尺度特征，降低计算复杂度，提升音频处理效率。**

- **链接: [http://arxiv.org/pdf/2510.24519v1](http://arxiv.org/pdf/2510.24519v1)**

> **作者:** Rinku Sebastian; Simon O'Keefe; Martin Trefzer
>
> **摘要:** Extracting features from the speech is the most critical process in speech signal processing. Mel Frequency Cepstral Coefficients (MFCC) are the most widely used features in the majority of the speaker and speech recognition applications, as the filtering in this feature is similar to the filtering taking place in the human ear. But the main drawback of this feature is that it provides only the frequency information of the signal but does not provide the information about at what time which frequency is present. The wavelet transform, with its flexible time-frequency window, provides time and frequency information of the signal and is an appropriate tool for the analysis of non-stationary signals like speech. On the other hand, because of its uniform frequency scaling, a typical wavelet transform may be less effective in analysing speech signals, have poorer frequency resolution in low frequencies, and be less in line with human auditory perception. Hence, it is necessary to develop a feature that incorporates the merits of both MFCC and wavelet transform. A great deal of studies are trying to combine both these features. The present Wavelet Transform based Mel-scaled feature extraction methods require more computation when a wavelet transform is applied on top of Mel-scale filtering, since it adds extra processing steps. Here we are proposing a method to extract Mel scale features in time domain combining the concept of wavelet transform, thus reducing the computational burden of time-frequency conversion and the complexity of wavelet extraction. Combining our proposed Time domain Mel frequency Wavelet Coefficient(TMFWC) technique with the reservoir computing methodology has significantly improved the efficiency of audio signal processing.
>
---
#### [new 012] emg2speech: synthesizing speech from electromyography using self-supervised speech models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出emg2speech，一种基于自监督语音模型的肌电到语音合成方法。针对无语言能力者无法发声的问题，利用面部肌肉肌电信号（EMG）直接生成语音。通过发现自监督语音特征与肌电功率的强线性关系，实现端到端的EMG到语音转换，无需显式发音模型或声码器训练。**

- **链接: [http://arxiv.org/pdf/2510.23969v1](http://arxiv.org/pdf/2510.23969v1)**

> **作者:** Harshavardhana T. Gowda; Lee M. Miller
>
> **摘要:** We present a neuromuscular speech interface that translates electromyographic (EMG) signals collected from orofacial muscles during speech articulation directly into audio. We show that self-supervised speech (SS) representations exhibit a strong linear relationship with the electrical power of muscle action potentials: SS features can be linearly mapped to EMG power with a correlation of $r = 0.85$. Moreover, EMG power vectors corresponding to different articulatory gestures form structured and separable clusters in feature space. This relationship: $\text{SS features}$ $\xrightarrow{\texttt{linear mapping}}$ $\text{EMG power}$ $\xrightarrow{\texttt{gesture-specific clustering}}$ $\text{articulatory movements}$, highlights that SS models implicitly encode articulatory mechanisms. Leveraging this property, we directly map EMG signals to SS feature space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory models and vocoder training.
>
---
#### [new 013] Forward Convolutive Prediction for Frame Online Monaural Speech Dereverberation Based on Kronecker Product Decomposition
- **分类: eess.AS**

- **简介: 该论文针对语音去混响任务，解决传统前向卷积预测方法中长线性预测滤波器带来的高计算复杂度问题。提出基于克罗内克积分解的新型FCP方法，将长滤波器分解为两个短滤波器的克罗内克积，显著降低计算成本，并设计在线自适应算法迭代更新滤波器，实现高效低耗的实时去混响。**

- **链接: [http://arxiv.org/pdf/2510.24471v1](http://arxiv.org/pdf/2510.24471v1)**

> **作者:** Yujie Zhu; Jilu Jin; Xueqin Luo; Wenxing Yang; Zhong-Qiu Wang; Gongping Huang; Jingdong Chen; Jacob Benesty
>
> **摘要:** Dereverberation has long been a crucial research topic in speech processing, aiming to alleviate the adverse effects of reverberation in voice communication and speech interaction systems. Among existing approaches, forward convolutional prediction (FCP) has recently attracted attention. It typically employs a deep neural network to predict the direct-path signal and subsequently estimates a linear prediction filter to suppress residual reverberation. However, a major drawback of this approach is that the required linear prediction filter is often excessively long, leading to considerable computational complexity. To address this, our work proposes a novel FCP method based on Kronecker product (KP) decomposition, in which the long prediction filter is modeled as the KP of two much shorter filters. This decomposition significantly reduces the computational cost. An adaptive algorithm is then provided to iteratively update these shorter filters online. Experimental results show that, compared to conventional methods, our approach achieves competitive dereverberation performance while substantially reducing computational cost.
>
---
#### [new 014] Your Microphone Array Retains Your Identity: A Robust Voice Liveness Detection System for Smart Speakers
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文针对智能音箱语音欺骗攻击问题，提出基于麦克风阵列的被动活体检测方法。通过引入“阵列指纹”特征，利用阵列布局实现身份识别，提升环境变化和用户移动下的鲁棒性。设计轻量级方案ARRAYID，结合多特征，在大规模数据集上达到99.84%准确率，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24393v1](http://arxiv.org/pdf/2510.24393v1)**

> **作者:** Yan Meng; Jiachun Li; Matthew Pillari; Arjun Deopujari; Liam Brennan; Hafsah Shamsie; Haojin Zhu; Yuan Tian
>
> **备注:** This is a paper accepted by USENIX Security 2022. See: https://www.usenix.org/conference/usenixsecurity22/presentation/meng
>
> **摘要:** Though playing an essential role in smart home systems, smart speakers are vulnerable to voice spoofing attacks. Passive liveness detection, which utilizes only the collected audio rather than the deployed sensors to distinguish between live-human and replayed voices, has drawn increasing attention. However, it faces the challenge of performance degradation under the different environmental factors as well as the strict requirement of the fixed user gestures. In this study, we propose a novel liveness feature, array fingerprint, which utilizes the microphone array inherently adopted by the smart speaker to determine the identity of collected audios. Our theoretical analysis demonstrates that by leveraging the circular layout of microphones, compared with existing schemes, array fingerprint achieves a more robust performance under the environmental change and user's movement. Then, to leverage such a fingerprint, we propose ARRAYID, a lightweight passive detection scheme, and elaborate a series of features working together with array fingerprint. Our evaluation on the dataset containing 32,780 audio samples and 14 spoofing devices shows that ARRAYID achieves an accuracy of 99.84%, which is superior to existing passive liveness detection schemes.
>
---
## 更新

#### [replaced 001] Low-Resource Audio Codec (LRAC): 2025 Challenge Description
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.23312v2](http://arxiv.org/pdf/2510.23312v2)**

> **作者:** Kamil Wojcicki; Yusuf Ziya Isik; Laura Lechler; Mansur Yesilbursa; Ivana Balić; Wolfgang Mack; Rafał Łaganowski; Guoqing Zhang; Yossi Adi; Minje Kim; Shinji Watanabe
>
> **摘要:** While recent neural audio codecs deliver superior speech quality at ultralow bitrates over traditional methods, their practical adoption is hindered by obstacles related to low-resource operation and robustness to acoustic distortions. Edge deployment scenarios demand codecs that operate under stringent compute constraints while maintaining low latency and bitrate. The presence of background noise and reverberation further necessitates designs that are resilient to such degradations. The performance of neural codecs under these constraints and their integration with speech enhancement remain largely unaddressed. To catalyze progress in this area, we introduce the 2025 Low-Resource Audio Codec Challenge, which targets the development of neural and hybrid codecs for resource-constrained applications. Participants are supported with a standardized training dataset, two baseline systems, and a comprehensive evaluation framework. The challenge is expected to yield valuable insights applicable to both codec design and related downstream audio tasks.
>
---
#### [replaced 002] RIR-Mega: a large-scale simulated room impulse response dataset for machine learning and room acoustics modeling
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2510.18917v2](http://arxiv.org/pdf/2510.18917v2)**

> **作者:** Mandip Goswami
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Room impulse responses are a core resource for dereverberation, robust speech recognition, source localization, and room acoustics estimation. We present RIR-Mega, a large collection of simulated RIRs described by a compact, machine friendly metadata schema and distributed with simple tools for validation and reuse. The dataset ships with a Hugging Face Datasets loader, scripts for metadata checks and checksums, and a reference regression baseline that predicts RT60 like targets from waveforms. On a train and validation split of 36,000 and 4,000 examples, a small Random Forest on lightweight time and spectral features reaches a mean absolute error near 0.013 s and a root mean square error near 0.022 s. We host a subset with 1,000 linear array RIRs and 3,000 circular array RIRs on Hugging Face for streaming and quick tests, and preserve the complete 50,000 RIR archive on Zenodo. The dataset and code are public to support reproducible studies.
>
---
#### [replaced 003] BNMusic: Blending Environmental Noises into Personalized Music
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10754v2](http://arxiv.org/pdf/2506.10754v2)**

> **作者:** Chi Zuo; Martin B. Møller; Pablo Martínez-Nuevo; Huayang Huang; Yu Wu; Ye Zhu
>
> **备注:** This paper has been accepted by NeurIPS 2025
>
> **摘要:** While being disturbed by environmental noises, the acoustic masking technique is a conventional way to reduce the annoyance in audio engineering that seeks to cover up the noises with other dominant yet less intrusive sounds. However, misalignment between the dominant sound and the noise-such as mismatched downbeats-often requires an excessive volume increase to achieve effective masking. Motivated by recent advances in cross-modal generation, in this work, we introduce an alternative method to acoustic masking, aiming to reduce the noticeability of environmental noises by blending them into personalized music generated based on user-provided text prompts. Following the paradigm of music generation using mel-spectrogram representations, we propose a Blending Noises into Personalized Music (BNMusic) framework with two key stages. The first stage synthesizes a complete piece of music in a mel-spectrogram representation that encapsulates the musical essence of the noise. In the second stage, we adaptively amplify the generated music segment to further reduce noise perception and enhance the blending effectiveness, while preserving auditory quality. Our experiments with comprehensive evaluations on MusicBench, EPIC-SOUNDS, and ESC-50 demonstrate the effectiveness of our framework, highlighting the ability to blend environmental noise with rhythmically aligned, adaptively amplified, and enjoyable music segments, minimizing the noticeability of the noise, thereby improving overall acoustic experiences. Project page: https://d-fas.github.io/BNMusic_page/.
>
---
#### [replaced 004] RapVerse: Coherent Vocals and Whole-Body Motions Generations from Text
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.20336v2](http://arxiv.org/pdf/2405.20336v2)**

> **作者:** Jiaben Chen; Xin Yan; Yihang Chen; Siyuan Cen; Zixin Wang; Qinwei Ma; Haoyu Zhen; Kaizhi Qian; Lie Lu; Chuang Gan
>
> **备注:** ICCV 2025, Project website: https://jiabenchen.github.io/RapVerse/
>
> **摘要:** In this work, we introduce a challenging task for simultaneously generating 3D holistic body motions and singing vocals directly from textual lyrics inputs, advancing beyond existing works that typically address these two modalities in isolation. To facilitate this, we first collect the RapVerse dataset, a large dataset containing synchronous rapping vocals, lyrics, and high-quality 3D holistic body meshes. With the RapVerse dataset, we investigate the extent to which scaling autoregressive multimodal transformers across language, audio, and motion can enhance the coherent and realistic generation of vocals and whole-body human motions. For modality unification, a vector-quantized variational autoencoder is employed to encode whole-body motion sequences into discrete motion tokens, while a vocal-to-unit model is leveraged to obtain quantized audio tokens preserving content, prosodic information and singer identity. By jointly performing transformer modeling on these three modalities in a unified way, our framework ensures a seamless and realistic blend of vocals and human motions. Extensive experiments demonstrate that our unified generation framework not only produces coherent and realistic singing vocals alongside human motions directly from textual inputs, but also rivals the performance of specialized single-modality generation systems, establishing new benchmarks for joint vocal-motion generation.
>
---
#### [replaced 005] Detecting Neurocognitive Disorders through Analyses of Topic Evolution and Cross-modal Consistency in Visual-Stimulated Narratives
- **分类: eess.AS; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03727v3](http://arxiv.org/pdf/2501.03727v3)**

> **作者:** Jinchao Li; Yuejiao Wang; Junan Li; Jiawen Kang; Bo Zheng; Ka Ho Wong; Brian Mak; Helene H. Fung; Jean Woo; Man-Wai Mak; Timothy Kwok; Vincent Mok; Xianmin Gong; Xixin Wu; Xunying Liu; Patrick C. M. Wong; Helen Meng
>
> **备注:** 16 pages, 5 figures, accepted by "IEEE Journal of Selected Topics in Signal Processing"
>
> **摘要:** Early detection of neurocognitive disorders (NCDs) is crucial for timely intervention and disease management. Given that language impairments manifest early in NCD progression, visual-stimulated narrative (VSN)-based analysis offers a promising avenue for NCD detection. Current VSN-based NCD detection methods primarily focus on linguistic microstructures (e.g., lexical diversity) that are closely tied to bottom-up, stimulus-driven cognitive processes. While these features illuminate basic language abilities, the higher-order linguistic macrostructures (e.g., topic development) that may reflect top-down, concept-driven cognitive abilities remain underexplored. These macrostructural patterns are crucial for NCD detection, yet challenging to quantify due to their abstract and complex nature. To bridge this gap, we propose two novel macrostructural approaches: (1) a Dynamic Topic Model (DTM) to track topic evolution over time, and (2) a Text-Image Temporal Alignment Network (TITAN) to measure cross-modal consistency between narrative and visual stimuli. Experimental results show the effectiveness of the proposed approaches in NCD detection, with TITAN achieving superior performance across three corpora: ADReSS (F1=0.8889), ADReSSo (F1=0.8504), and CU-MARVEL-RABBIT (F1=0.7238). Feature contribution analysis reveals that macrostructural features (e.g., topic variability, topic change rate, and topic consistency) constitute the most significant contributors to the model's decision pathways, outperforming the investigated microstructural features. These findings underscore the value of macrostructural analysis for understanding linguistic-cognitive interactions associated with NCDs.
>
---
#### [replaced 006] Local Density-Based Anomaly Score Normalization for Domain Generalization
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.10951v2](http://arxiv.org/pdf/2509.10951v2)**

> **作者:** Kevin Wilkinghoff; Haici Yang; Janek Ebbers; François G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **摘要:** State-of-the-art anomalous sound detection (ASD) systems in domain-shifted conditions rely on projecting audio signals into an embedding space and using distance-based outlier detection to compute anomaly scores. One of the major difficulties to overcome is the so-called domain mismatch between the anomaly score distributions of a source domain and a target domain that differ acoustically and in terms of the amount of training data provided. A decision threshold that is optimal for one domain may be highly sub-optimal for the other domain and vice versa. This significantly degrades the performance when only using a single decision threshold, as is required when generalizing to multiple data domains that are possibly unseen during training while still using the same trained ASD system as in the source domain. To reduce this mismatch between the domains, we propose a simple local-density-based anomaly score normalization scheme. In experiments conducted on several ASD datasets, we show that the proposed normalization scheme consistently improves performance for various types of embedding-based ASD systems and yields better results than existing anomaly score normalization approaches.
>
---
#### [replaced 007] A Unified Framework for Direction and Diffuseness Estimation Using Tight-Frame Microphone Arrays
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2510.22183v2](http://arxiv.org/pdf/2510.22183v2)**

> **作者:** Akira Omoto
>
> **备注:** 36 pages including 14 files
>
> **摘要:** This work presents a unified framework for estimating both sound-field direction and diffuseness using practical microphone arrays with different spatial configurations. Building on covariance-based diffuseness models, we formulate a velocity-only covariance approach that enables consistent diffuseness evaluation across heterogeneous array geometries without requiring mode whitening or spherical-harmonic decomposition. Three array types -- an A-format array, a rigid-sphere array, and a newly proposed tight-frame array -- are modeled and compared through both simulations and measurement-based experiments. The results show that the tight-frame configuration achieves near-isotropic directional sampling and reproduces diffuseness characteristics comparable to those of higher-order spherical arrays, while maintaining a compact physical structure. We further examine the accuracy of direction-of-arrival estimation based on acoustic intensity within the same framework. These findings connect theoretical diffuseness analysis with implementable array designs and support the development of robust, broadband methods for spatial-sound-field characterization.
>
---
#### [replaced 008] Latent Multi-view Learning for Robust Environmental Sound Representations
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.02500v3](http://arxiv.org/pdf/2510.02500v3)**

> **作者:** Sivan Ding; Julia Wilkins; Magdalena Fuentes; Juan Pablo Bello
>
> **备注:** Accepted to DCASE 2025 Workshop. 4+1 pages, 2 figures, 2 tables
>
> **摘要:** Self-supervised learning (SSL) approaches, such as contrastive and generative methods, have advanced environmental sound representation learning using unlabeled data. However, how these approaches can complement each other within a unified framework remains relatively underexplored. In this work, we propose a multi-view learning framework that integrates contrastive principles into a generative pipeline to capture sound source and device information. Our method encodes compressed audio latents into view-specific and view-common subspaces, guided by two self-supervised objectives: contrastive learning for targeted information flow between subspaces, and reconstruction for overall information preservation. We evaluate our method on an urban sound sensor network dataset for sound source and sensor classification, demonstrating improved downstream performance over traditional SSL techniques. Additionally, we investigate the model's potential to disentangle environmental sound attributes within the structured latent space under varied training configurations.
>
---
#### [replaced 009] Acoustic and Machine Learning Methods for Speech-Based Suicide Risk Assessment: A Systematic Review
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.18195v2](http://arxiv.org/pdf/2505.18195v2)**

> **作者:** Ambre Marie; Marine Garnier; Thomas Bertin; Laura Machart; Guillaume Dardenne; Gwenolé Quellec; Sofian Berrouiguet
>
> **备注:** Preprint version of a manuscript submitted to the Journal of Affective Disorders
>
> **摘要:** Suicide remains a public health challenge, necessitating improved detection methods to facilitate timely intervention and treatment. This systematic review evaluates the role of Artificial Intelligence (AI) and Machine Learning (ML) in assessing suicide risk through acoustic analysis of speech. Following PRISMA guidelines, we analyzed 33 articles selected from PubMed, Cochrane, Scopus, and Web of Science databases. The last search was conducted in February 2025. Risk of bias was assessed using the PROBAST tool. Studies analyzing acoustic features between individuals at risk of suicide (RS) and those not at risk (NRS) were included, while studies lacking acoustic data, a suicide-related focus, or sufficient methodological details were excluded. Sample sizes varied widely and were reported in terms of participants or speech segments, depending on the study. Results were synthesized narratively based on acoustic features and classifier performance. Findings consistently showed significant acoustic feature variations between RS and NRS populations, particularly involving jitter, fundamental frequency (F0), Mel-frequency cepstral coefficients (MFCC), and power spectral density (PSD). Classifier performance varied based on algorithms, modalities, and speech elicitation methods, with multimodal approaches integrating acoustic, linguistic, and metadata features demonstrating superior performance. Among the 29 classifier-based studies, reported AUC values ranged from 0.62 to 0.985 and accuracies from 60% to 99.85%. Most datasets were imbalanced in favor of NRS, and performance metrics were rarely reported separately by group, limiting clear identification of direction of effect.
>
---
#### [replaced 010] Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.08078v3](http://arxiv.org/pdf/2510.08078v3)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **摘要:** Video-to-Audio generation has made remarkable strides in automatically synthesizing sound for video. However, existing evaluation metrics, which focus on semantic and temporal alignment, overlook a critical failure mode: models often generate acoustic events, particularly speech and music, that have no corresponding visual source. We term this phenomenon Insertion Hallucination and identify it as a systemic risk driven by dataset biases, such as the prevalence of off-screen sounds, that remains completely undetected by current metrics. To address this challenge, we first develop a systematic evaluation framework that employs a majority-voting ensemble of multiple audio event detectors. We also introduce two novel metrics to quantify the prevalence and severity of this issue: IH@vid (the fraction of videos with hallucinations) and IH@dur (the fraction of hallucinated duration). Building on this, we propose Posterior Feature Correction, a novel training-free inference-time method that mitigates IH. PFC operates in a two-pass process: it first generates an initial audio output to detect hallucinated segments, and then regenerates the audio after masking the corresponding video features at those timestamps. Experiments on several mainstream V2A benchmarks first reveal that state-of-the-art models suffer from severe IH. In contrast, our PFC method reduces both the prevalence and duration of hallucinations by over 50\% on average, without degrading, and in some cases even improving, conventional metrics for audio quality and temporal synchronization. Our work is the first to formally define, systematically measure, and effectively mitigate Insertion Hallucination, paving the way for more reliable and faithful V2A models.
>
---
#### [replaced 011] SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.23541v2](http://arxiv.org/pdf/2510.23541v2)**

> **作者:** Hanke Xie; Haopeng Lin; Wenxiao Cao; Dake Guo; Wenjie Tian; Jun Wu; Hanlin Wen; Ruixuan Shang; Hongmei Liu; Zhiqi Jiang; Yuepeng Jiang; Wenxi Chen; Ruiqi Yan; Jiale Qian; Yichao Yan; Shunshun Yin; Ming Tao; Xie Chen; Lei Xie; Xinsheng Wang
>
> **摘要:** Recent advances in text-to-speech (TTS) synthesis have significantly improved speech expressiveness and naturalness. However, most existing systems are tailored for single-speaker synthesis and fall short in generating coherent multi-speaker conversational speech. This technical report presents SoulX-Podcast, a system designed for podcast-style multi-turn, multi-speaker dialogic speech generation, while also achieving state-of-the-art performance in conventional TTS tasks. To meet the higher naturalness demands of multi-turn spoken dialogue, SoulX-Podcast integrates a range of paralinguistic controls and supports both Mandarin and English, as well as several Chinese dialects, including Sichuanese, Henanese, and Cantonese, enabling more personalized podcast-style speech generation. Experimental results demonstrate that SoulX-Podcast can continuously produce over 90 minutes of conversation with stable speaker timbre and smooth speaker transitions. Moreover, speakers exhibit contextually adaptive prosody, reflecting natural rhythm and intonation changes as dialogues progress. Across multiple evaluation metrics, SoulX-Podcast achieves state-of-the-art performance in both monologue TTS and multi-turn conversational speech synthesis.
>
---
