# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Real-Time Object Tracking with On-Device Deep Learning for Adaptive Beamforming in Dynamic Acoustic Environments
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文针对动态声学环境中声源定位与定向拾音难题，提出一种基于嵌入式深度学习的实时目标跟踪系统。通过单目深度估计与双目视觉实现3D目标定位，结合微型麦克风阵列实现2D波束成形动态调整，实时同步声学聚焦与目标位置，显著提升信干比，适用于智能会议、智能家居等场景。**

- **链接: [https://arxiv.org/pdf/2511.19396v1](https://arxiv.org/pdf/2511.19396v1)**

> **作者:** Jorge Ortigoso-Narro; Jose A. Belloch; Adrian Amor-Martin; Sandra Roger; Maximo Cobos
>
> **摘要:** Advances in object tracking and acoustic beamforming are driving new capabilities in surveillance, human-computer interaction, and robotics. This work presents an embedded system that integrates deep learning-based tracking with beamforming to achieve precise sound source localization and directional audio capture in dynamic environments. The approach combines single-camera depth estimation and stereo vision to enable accurate 3D localization of moving objects. A planar concentric circular microphone array constructed with MEMS microphones provides a compact, energy-efficient platform supporting 2D beam steering across azimuth and elevation. Real-time tracking outputs continuously adapt the array's focus, synchronizing the acoustic response with the target's position. By uniting learned spatial awareness with dynamic steering, the system maintains robust performance in the presence of multiple or moving sources. Experimental evaluation demonstrates significant gains in signal-to-interference ratio, making the design well-suited for teleconferencing, smart home devices, and assistive technologies.
>
---
#### [new 002] Speech Recognition Model Improves Text-to-Speech Synthesis using Fine-Grained Reward
- **分类: eess.AS; cs.CL**

- **简介: 该论文针对语音合成中评估与优化粒度粗的问题，提出基于自回归语音识别模型（如Whisper）注意力机制的细粒度奖励信号。通过利用ASR模型的跨注意力捕捉语音与文本的词级对齐，实现对TTS模型的精细化优化，提升合成质量与零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.17555v1](https://arxiv.org/pdf/2511.17555v1)**

> **作者:** Guansu Wang; Peijie Sun
>
> **备注:** The paper makes an important contribution to the very challenging problem of training TTS models, with a novel application of reinforcement learning and demonstrating convincing improvements
>
> **摘要:** Recent advances in text-to-speech (TTS) have enabled models to clone arbitrary unseen speakers and synthesize high-quality, natural-sounding speech. However, evaluation methods lag behind: typical mean opinion score (MOS) estimators perform regression over entire utterances, while failures usually occur in a few problematic words. We observe that encoder-decoder ASR models (e.g., Whisper) surface word-level mismatches between speech and text via cross-attention, providing a fine-grained reward signal. Building on this, we introduce Word-level TTS Alignment by ASR-driven Attentive Reward (W3AR). Without explicit reward annotations, W3AR uses attention from a pre-trained ASR model to drive finer-grained alignment and optimization of sequences predicted by a TTS model. Experiments show that W3AR improves the quality of existing TTS systems and strengthens zero-shot robustness on unseen speakers. More broadly, our results suggest a simple recipe for generative modeling: understanding models can act as evaluators, delivering informative, fine-grained feedback for optimization.
>
---
#### [new 003] NSTR: Neural Spectral Transport Representation for Space-Varying Frequency Fields
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出NSTR，一种新型隐式神经表示框架，用于建模空间变化的频率特征。针对现有方法假设全局平稳频谱的问题，NSTR显式引入可学习的频率传输方程，实现局部频谱自适应，提升图像、音频和3D几何重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.18384v1](https://arxiv.org/pdf/2511.18384v1)**

> **作者:** Plein Versace
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a powerful paradigm for representing signals such as images, audio, and 3D scenes. However, existing INR frameworks -- including MLPs with Fourier features, SIREN, and multiresolution hash grids -- implicitly assume a \textit{global and stationary} spectral basis. This assumption is fundamentally misaligned with real-world signals whose frequency characteristics vary significantly across space, exhibiting local high-frequency textures, smooth regions, and frequency drift phenomena. We propose \textbf{Neural Spectral Transport Representation (NSTR)}, the first INR framework that \textbf{explicitly models a spatially varying local frequency field}. NSTR introduces a learnable \emph{frequency transport equation}, a PDE that governs how local spectral compositions evolve across space. Given a learnable local spectrum field $S(x)$ and a frequency transport network $F_θ$ enforcing $\nabla S(x) \approx F_θ(x, S(x))$, NSTR reconstructs signals by spatially modulating a compact set of global sinusoidal bases. This formulation enables strong local adaptivity and offers a new level of interpretability via visualizing frequency flows. Experiments on 2D image regression, audio reconstruction, and implicit 3D geometry show that NSTR achieves significantly better accuracy-parameter trade-offs than SIREN, Fourier-feature MLPs, and Instant-NGP. NSTR requires fewer global frequencies, converges faster, and naturally explains signal structure through spectral transport fields. We believe NSTR opens a new direction in INR research by introducing explicit modeling of space-varying spectrum.
>
---
#### [new 004] Diffusion-based Surrogate Model for Time-varying Underwater Acoustic Channels
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对水下声学信道建模难题，提出基于扩散模型的StableUASim框架。旨在解决传统物理模型依赖环境信息、随机重放方法泛化性差的问题。通过预训练条件隐空间扩散模型，实现少量数据下的快速适应与多样、真实信道样本生成，支持高效分析与压缩，提升通信系统设计与机器学习应用的可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.18078v1](https://arxiv.org/pdf/2511.18078v1)**

> **作者:** Kexin Li; Mandar Chitre
>
> **摘要:** Accurate modeling of time-varying underwater acoustic channels is essential for the design, evaluation, and deployment of reliable underwater communication systems. Conventional physics models require detailed environmental knowledge, while stochastic replay methods are constrained by the limited diversity of measured channels and often fail to generalize to unseen scenarios, reducing their practical applicability. To address these challenges, we propose StableUASim, a pre-trained conditional latent diffusion surrogate model that captures the stochastic dynamics of underwater acoustic communication channels. Leveraging generative modeling, StableUASim produces diverse and statistically realistic channel realizations, while supporting conditional generation from specific measurement samples. Pre-training enables rapid adaptation to new environments using minimal additional data, and the autoencoder latent representation facilitates efficient channel analysis and compression. Experimental results demonstrate that StableUASim accurately reproduces key channel characteristics and communication performance, providing a scalable, data-efficient, and physically consistent surrogate model for both system design and machine learning-driven underwater applications.
>
---
#### [new 005] Three-Class Emotion Classification for Audiovisual Scenes Based on Ensemble Learning Scheme
- **分类: cs.SD; cs.HC**

- **简介: 该论文针对资源受限设备上的情绪识别难题，提出一种基于集成学习的音频单模态三分类方法，旨在实现高效、准确的情绪识别。通过融合支持向量机与神经网络的堆叠架构，结合定制化数据预处理，显著提升分类性能，在真实数据集上达86%准确率，适用于消费级音视频系统。**

- **链接: [https://arxiv.org/pdf/2511.17926v1](https://arxiv.org/pdf/2511.17926v1)**

> **作者:** Xiangrui Xiong; Zhou Zhou; Guocai Nong; Junlin Deng; Ning Wu
>
> **摘要:** Emotion recognition plays a pivotal role in enhancing human-computer interaction, particularly in movie recommendation systems where understanding emotional content is essential. While multimodal approaches combining audio and video have demonstrated effectiveness, their reliance on high-performance graphical computing limits deployment on resource-constrained devices such as personal computers or home audiovisual systems. To address this limitation, this study proposes a novel audio-only ensemble learning framework capable of classifying movie scenes into three emotional categories: Good, Neutral, and Bad. The model integrates ten support vector machines and six neural networks within a stacking ensemble architecture to enhance classification performance. A tailored data preprocessing pipeline, including feature extraction, outlier handling, and feature engineering, is designed to optimize emotional information from audio inputs. Experiments on a simulated dataset achieve 67% accuracy, while a real-world dataset collected from 15 diverse films yields an impressive 86% accuracy. These results underscore the potential of audio-based, lightweight emotion recognition methods for broader consumer-level applications, offering both computational efficiency and robust classification capabilities.
>
---
#### [new 006] Multimodal Real-Time Anomaly Detection and Industrial Applications
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文针对工业场景中的实时异常检测任务，解决多模态数据融合与高精度实时识别难题。提出两代系统：首代基于YOLOv8、ByteTrack与AST；进阶版融合多模型音频、双目标检测器及双向跨模态注意力，实现高效精准的多模态异常检测，适用于工业安全等实际场景。**

- **链接: [https://arxiv.org/pdf/2511.18698v1](https://arxiv.org/pdf/2511.18698v1)**

> **作者:** Aman Verma; Keshav Samdani; Mohd. Samiuddin Shafi
>
> **摘要:** This paper presents the design, implementation, and evolution of a comprehensive multimodal room-monitoring system that integrates synchronized video and audio processing for real-time activity recognition and anomaly detection. We describe two iterations of the system: an initial lightweight implementation using YOLOv8, ByteTrack, and the Audio Spectrogram Transformer (AST), and an advanced version that incorporates multi-model audio ensembles, hybrid object detection, bidirectional cross-modal attention, and multi-method anomaly detection. The evolution demonstrates significant improvements in accuracy, robustness, and industrial applicability. The advanced system combines three audio models (AST, Wav2Vec2, and HuBERT) for comprehensive audio understanding, dual object detectors (YOLO and DETR) for improved accuracy, and sophisticated fusion mechanisms for enhanced cross-modal learning. Experimental evaluation shows the system's effectiveness in general monitoring scenarios as well as specialized industrial safety applications, achieving real-time performance on standard hardware while maintaining high accuracy.
>
---
#### [new 007] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文针对视频到音频生成任务，解决现有方法因目标纠缠导致的多维感知平衡难题。提出PrismAudio框架，通过四类专项思维链（CoT）与对应奖励函数，实现多维度强化学习优化，并引入Fast-GRPO降低计算开销。构建AudioCanvas基准测试集，实验证明其在四项感知维度上均达领先性能。**

- **链接: [https://arxiv.org/pdf/2511.18833v1](https://arxiv.org/pdf/2511.18833v1)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** Preprint
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at https://PrismAudio-Project.github.io.
>
---
#### [new 008] InstructAudio: Unified speech and music generation with natural language instruction
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出InstructAudio，一个统一的语音与音乐生成框架，旨在通过自然语言指令控制音色、情感、风格等声学属性。针对传统TTS和TTM模型在指令控制、跨模态联合建模方面的局限，该工作构建了基于扩散Transformer的统一模型，支持中英文语音、音乐及对话生成，实现了多任务协同与跨模态对齐。**

- **链接: [https://arxiv.org/pdf/2511.18487v1](https://arxiv.org/pdf/2511.18487v1)**

> **作者:** Chunyu Qiang; Kang Yin; Xiaopeng Wang; Yuzhe Liang; Jiahui Zhao; Ruibo Fu; Tianrui Wang; Cheng Gong; Chen Zhang; Longbiao Wang; Jianwu Dang
>
> **摘要:** Text-to-speech (TTS) and text-to-music (TTM) models face significant limitations in instruction-based control. TTS systems usually depend on reference audio for timbre, offer only limited text-level attribute control, and rarely support dialogue generation. TTM systems are constrained by input conditioning requirements that depend on expert knowledge annotations. The high heterogeneity of these input control conditions makes them difficult to joint modeling with speech synthesis. Despite sharing common acoustic modeling characteristics, these two tasks have long been developed independently, leaving open the challenge of achieving unified modeling through natural language instructions. We introduce InstructAudio, a unified framework that enables instruction-based (natural language descriptions) control of acoustic attributes including timbre (gender, age), paralinguistic (emotion, style, accent), and musical (genre, instrument, rhythm, atmosphere). It supports expressive speech, music, and dialogue generation in English and Chinese. The model employs joint and single diffusion transformer layers with a standardized instruction-phoneme input format, trained on 50K hours of speech and 20K hours of music data, enabling multi-task learning and cross-modal alignment. Fig. 1 visualizes performance comparisons with mainstream TTS and TTM models, demonstrating that InstructAudio achieves optimal results on most metrics. To our best knowledge, InstructAudio represents the first instruction-controlled framework unifying speech and music generation. Audio samples are available at: https://qiangchunyu.github.io/InstructAudio/
>
---
#### [new 009] Multidimensional Music Aesthetic Evaluation via Semantically Consistent C-Mixup Augmentation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对生成歌曲美学评价的多维感知挑战，提出一种融合多源多尺度特征提取、层级音频增强与混合损失训练的框架，旨在提升评分准确性与优质歌曲识别能力。**

- **链接: [https://arxiv.org/pdf/2511.18869v1](https://arxiv.org/pdf/2511.18869v1)**

> **作者:** Shuyang Liu; Yuan Jin; Rui Lin; Shizhe Chen; Junyu Dai; Tao Jiang
>
> **摘要:** Evaluating the aesthetic quality of generated songs is challenging due to the multi-dimensional nature of musical perception. We propose a robust music aesthetic evaluation framework that combines (1) multi-source multi-scale feature extraction to obtain complementary segment- and track-level representations, (2) a hierarchical audio augmentation strategy to enrich training data, and (3) a hybrid training objective that integrates regression and ranking losses for accurate scoring and reliable top-song identification. Experiments on the ICASSP 2026 SongEval benchmark demonstrate that our approach consistently outperforms baseline methods across correlation and top-tier metrics.
>
---
#### [new 010] Frequency-Invariant Beamforming in Elevation and Azimuth via Autograd and Concentric Circular Microphone Arrays
- **分类: cs.SD**

- **简介: 该论文研究双轴（俯仰与方位）频率不变波束成形，针对传统线性阵列仅能单轴控制、圆形阵列在低频时俯仰方向性能差的问题，结合自动微分与同心圆麦克风阵列，实现双轴连续优化。通过仿真验证，其主瓣更窄、空间选择性更强，尤其在低频俯仰方向表现优异。**

- **链接: [https://arxiv.org/pdf/2511.19403v1](https://arxiv.org/pdf/2511.19403v1)**

> **作者:** Jorge Ortigoso-Narro; Jose A. Belloch; Maximo Morales-Cespedes; Maximo Cobos
>
> **摘要:** The use of planar and concentric circular microphone arrays in beamforming has gained attention due to their ability to optimize both azimuth and elevation angles, making them ideal for spatial audio tasks like sound source localization and noise suppression. Unlike linear arrays, which restrict steering to a single axis, 2D arrays offer dual-axis optimization, although elevation control remains challenging. This study explores the integration of autograd, an automatic differentiation tool, with concentric circular arrays to impose beamwidth and frequency invariance constraints. This enables continuous optimization over both angles while maintaining performance across a wide frequency range. We evaluate our method through simulations of beamwidth, white noise gain, and directivity across multiple frequencies. A comparative analysis is presented against standard and advanced beamformers, including delay-and-sum, modified delay-and-sum, a Jacobi-Anger expansion-based method, and a Gaussian window-based gradient descent approach. Our method achieves superior spatial selectivity and narrower mainlobes, particularly in the elevation axis at lower frequencies. These results underscore the effectiveness of our approach in enhancing beamforming performance for acoustic sensing and spatial audio applications requiring precise dual-axis control.
>
---
#### [new 011] First Deep Learning Approach to Hammering Acoustics for Stem Stability Assessment in Total Hip Arthroplasty
- **分类: eess.AS**

- **简介: 该论文针对骨科手术中股骨柄初始稳定性评估难题，提出首个基于深度学习的锤击声分类方法。通过TimeMIL模型分析对数梅尔频谱，结合伪标签提升性能，在术中录音上达到91.17%准确率，验证了音频事件分类在髋关节置换术中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.18725v1](https://arxiv.org/pdf/2511.18725v1)**

> **作者:** Dongqi Zhu; Zhuwen Xu; Youyuan Chen; Minghao Jin; Wan Zheng; Yi Zhou; Huiwu Li; Yongyun Chang; Feng Hong; Zanjing Zhai
>
> **摘要:** Audio event classification has recently emerged as a promising approach in medical applications. In total hip arthroplasty (THA), intra-operative hammering acoustics provide critical cues for assessing the initial stability of the femoral stem, yet variability due to femoral morphology, implant size, and surgical technique constrains conventional assessment methods. We propose the first deep learning framework for this task, employing a TimeMIL model trained on Log-Mel Spectrogram features and enhanced with pseudo-labeling. On intra-operative recordings, the method achieved 91.17 % +/- 2.79 % accuracy, demonstrating reliable estimation of stem stability. Comparative experiments further show that reducing the diversity of femoral stem brands improves model performance, although limited dataset size remains a bottleneck. These results establish deep learning-based audio event classification as a feasible approach for intra-operative stability assessment in THA.
>
---
#### [new 012] DHAuDS: A Dynamic and Heterogeneous Audio Benchmark for Test-Time Adaptation
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对音频分类中因声学条件变化导致的域偏移问题，提出DHAuDS基准。通过动态、异构的噪声设置，构建四个标准化音频数据集，包含50个评估场景，用于更真实地测试和比较测试时自适应（TTA）算法的性能，推动鲁棒音频模型研究。**

- **链接: [https://arxiv.org/pdf/2511.18421v1](https://arxiv.org/pdf/2511.18421v1)**

> **作者:** Weichuang Shao; Iman Yi Liao; Tomas Henrique Bode Maul; Tissa Chandesa
>
> **摘要:** Audio classifiers frequently face domain shift, when models trained on one dataset lose accuracy on data recorded in acoustically different conditions. Previous Test-Time Adaptation (TTA) research in speech and sound analysis often evaluates models under fixed or mismatched noise settings, that fail to mimic real-world variability. To overcome these limitations, this paper presents DHAuDS (Dynamic and Heterogeneous Audio Domain Shift), a benchmark designed to assess TTA approaches under more realistic and diverse acoustic shifts. DHAuDS comprises four standardized benchmarks: UrbanSound8K-C, SpeechCommandsV2-C, VocalSound-C, and ReefSet-C, each constructed with dynamic corruption severity levels and heterogeneous noise types to simulate authentic audio degradation scenarios. The framework defines 14 evaluation criteria for each benchmark (8 for UrbanSound8K-C), resulting in 50 unrepeated criteria (124 experiments) that collectively enable fair, reproducible, and cross-domain comparison of TTA algorithms. Through the inclusion of dynamic and mixed-domain noise settings, DHAuDS offers a consistent and publicly reproducible testbed to support ongoing studies in robust and adaptive audio modeling.
>
---
#### [new 013] Explicit Tonal Tension Conditioning via Dual-Level Beam Search for Symbolic Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于符号音乐生成任务，旨在解决现有模型难以显式控制音乐中音调紧张度的问题。提出基于双层束搜索的生成方法，结合音调区间向量分析，在词元和小节层级分别进行概率与张力重排序，实现对紧张度曲线的精确调控，并支持同一张力条件下的多样生成。**

- **链接: [https://arxiv.org/pdf/2511.19342v1](https://arxiv.org/pdf/2511.19342v1)**

> **作者:** Maral Ebrahimzadeh; Gilberto Bernardes; Sebastian Stober
>
> **备注:** 12 pages, 2 Figures, Accepted at the 17th International Symposium on Computer Music Multidisciplinary Research (CMMR) 2025
>
> **摘要:** State-of-the-art symbolic music generation models have recently achieved remarkable output quality, yet explicit control over compositional features, such as tonal tension, remains challenging. We propose a novel approach that integrates a computational tonal tension model, based on tonal interval vector analysis, into a Transformer framework. Our method employs a two-level beam search strategy during inference. At the token level, generated candidates are re-ranked using model probability and diversity metrics to maintain overall quality. At the bar level, a tension-based re-ranking is applied to ensure that the generated music aligns with a desired tension curve. Objective evaluations indicate that our approach effectively modulates tonal tension, and subjective listening tests confirm that the system produces outputs that align with the target tension. These results demonstrate that explicit tension conditioning through a dual-level beam search provides a powerful and intuitive tool to guide AI-generated music. Furthermore, our experiments demonstrate that our method can generate multiple distinct musical interpretations under the same tension condition.
>
---
#### [new 014] Dynamic Multi-Species Bird Soundscape Generation with Acoustic Patterning and 3D Spatialization
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文针对动态多物种鸟鸣声景生成难题，提出一种无需录音或训练数据的算法驱动框架。通过DSP基啁啾生成与3D空间化，模拟多物种鸟类独立运动轨迹与交互，实现可控制、高保真、沉浸式声景生成，适用于计算机音乐与生态声学研究。**

- **链接: [https://arxiv.org/pdf/2511.19275v1](https://arxiv.org/pdf/2511.19275v1)**

> **作者:** Ellie L. Zhang; Duoduo Liao; Callie C. Liao
>
> **备注:** Accepted by IEEE Big Data 2025
>
> **摘要:** Generation of dynamic, scalable multi-species bird soundscapes remains a significant challenge in computer music and algorithmic sound design. Birdsongs involve rapid frequency-modulated chirps, complex amplitude envelopes, distinctive acoustic patterns, overlapping calls, and dynamic inter-bird interactions, all of which require precise temporal and spatial control in 3D environments. Existing approaches, whether Digital Signal Processing (DSP)-based or data-driven, typically focus only on single species modeling, static call structures, or synthesis directly from recordings, and often suffer from noise, limited flexibility, or large data needs. To address these challenges, we present a novel, fully algorithm-driven framework that generates dynamic multi-species bird soundscapes using DSP-based chirp generation and 3D spatialization, without relying on recordings or training data. Our approach simulates multiple independently-moving birds per species along different moving 3D trajectories, supporting controllable chirp sequences, overlapping choruses, and realistic 3D motion in scalable soundscapes while preserving species-specific acoustic patterns. A visualization interface provides bird trajectories, spectrograms, activity timelines, and sound waves for analytical and creative purposes. Both visual and audio evaluations demonstrate the ability of the system to generate dense, immersive, and ecologically inspired soundscapes, highlighting its potential for computer music, interactive virtual environments, and computational bioacoustics research.
>
---
#### [new 015] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文针对多方议事模拟中因语音识别导致的发言者匿名问题，提出一种可复现的管道，将公开Zoom记录转化为带人物画像与行为标签的转录文本，并构建三个地方政府会议数据集。通过微调大模型实现“动作感知”的角色建模，显著提升模拟真实性与说话人一致性。**

- **链接: [https://arxiv.org/pdf/2511.17813v1](https://arxiv.org/pdf/2511.17813v1)**

> **作者:** Scott Merrill; Shashank Srivastava
>
> **备注:** 8 pages (29 pages including appendix), 18 figures. Code and datasets are available at https://github.com/smerrillunc/action-aware-llms. Submitted to ACL 2026
>
> **摘要:** Large language models offer opportunities to simulate multi-party deliberation, but realistic modeling remains limited by a lack of speaker-attributed data. Transcripts produced via automatic speech recognition (ASR) assign anonymous speaker labels (e.g., Speaker_1), preventing models from capturing consistent human behavior. This work introduces a reproducible pipeline to transform public Zoom recordings into speaker-attributed transcripts with metadata like persona profiles and pragmatic action tags (e.g., [propose_motion]). We release three local government deliberation datasets: Appellate Court hearings, School Board meetings, and Municipal Council sessions. Fine-tuning LLMs to model specific participants using this "action-aware" data produces a 67% reduction in perplexity and nearly doubles classifier-based performance metrics for speaker fidelity and realism. Turing-style human evaluations show our simulations are often indistinguishable from real deliberations, providing a practical and scalable method for complex realistic civic simulations.
>
---
#### [new 016] Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction
- **分类: cs.LG; cs.SD**

- **简介: 该论文针对生成式AI在实时人机音乐即兴中因强化学习后训练导致的“奖励黑客”问题，提出一种对抗性训练方法。通过引入协同进化的判别器，提升旋律伴奏的多样性与适应性，有效缓解输出同质化，增强音乐创作的动态变化与用户主导性。**

- **链接: [https://arxiv.org/pdf/2511.17879v1](https://arxiv.org/pdf/2511.17879v1)**

> **作者:** Yusong Wu; Stephen Brade; Teng Ma; Tia-Jane Fowler; Enning Yang; Berker Banar; Aaron Courville; Natasha Jaques; Cheng-Zhi Anna Huang
>
> **摘要:** Most applications of generative AI involve a sequential interaction in which a person inputs a prompt and waits for a response, and where reaction time and adaptivity are not important factors. In contrast, live jamming is a collaborative interaction that requires real-time coordination and adaptation without access to the other player's future moves, while preserving diversity to sustain a creative flow. Reinforcement learning post-training enables effective adaptation through on-policy interaction, yet it often reduces output diversity by exploiting coherence-based rewards. This collapse, known as ``reward hacking'', affects many RL post-training pipelines, but is especially harmful in live jamming, where musical creativity relies on dynamic variation and mutual responsiveness. In this paper, we propose a novel adversarial training method on policy-generated trajectories to mitigate reward hacking in RL post-training for melody-to-chord accompaniment. A co-evolving discriminator separates policy trajectories from the data distribution, while the policy maximizes the discriminator output in addition to coherence rewards to prevent collapse to trivial outputs. We evaluate accompaniment quality and output diversity in simulation with both fixed test melodies and learned melody agents, and we conduct a user study with the model deployed in a real-time interactive system with expert musicians. Quantitative evaluation and user feedback demonstrate improved output diversity, harmonic coherence, adaptation speed and user agency. Our results demonstrate a simple yet effective method to mitigate reward hacking in RL post-training of generative sequence models.
>
---
## 更新

#### [replaced 001] Speech Synthesis From Continuous Features Using Per-Token Latent Diffusion
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2410.16048v2](https://arxiv.org/pdf/2410.16048v2)**

> **作者:** Arnon Turetzky; Avihu Dekel; Nimrod Shabtay; Slava Shechtman; David Haws; Hagai Aronowitz; Ron Hoory; Yossi Adi
>
> **备注:** ASRU 2025
>
> **摘要:** We present SALAD, a zero-shot TTS autoregressive model operating over continuous speech representations. SALAD utilizes a per-token diffusion process to refine and predict continuous representations for the next time step. We compare our approach against a discrete variant of SALAD as well as publicly available zero-shot TTS systems, and conduct a comprehensive analysis of discrete versus continuous modeling techniques. Our results show that SALAD achieves superior intelligibility while matching the speech quality and speaker similarity of ground-truth audio.
>
---
#### [replaced 002] AMAuT: A Flexible and Efficient Multiview Audio Transformer Framework Trained from Scratch
- **分类: cs.SD; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.19368v2](https://arxiv.org/pdf/2510.19368v2)**

> **作者:** Weichuang Shao; Iman Yi Liao; Tomas Henrique Bode Maul; Tissa Chandesa
>
> **备注:** Updating note: 1. CLS+TAL is the distill token from DeiT rather than the alternative class token. Adjust the content to clarify it. 2. Figure 4 presents an error sequence of figures (a) and (b). 3. Remove an unrelated citation about the VS set. 4. A missing citation in section 4.4 (SSAST [19] here is not a correct citation)
>
> **摘要:** Recent foundational models, SSAST, EAT, HuBERT, Qwen-Audio, and Audio Flamingo, achieve top-tier results across standard audio benchmarks but are limited by fixed input rates and durations, hindering their reusability. This paper introduces the Augmentation-driven Multiview Audio Transformer (AMAuT), a training-from-scratch framework that eliminates the dependency on pre-trained weights while supporting arbitrary sample rates and audio lengths. AMAuT integrates four key components: (1) augmentation-driven multiview learning for robustness, (2) a conv1 + conv7 + conv1 one-dimensional CNN bottleneck for stable temporal encoding, (3) dual CLS + TAL tokens for bidirectional context representation, and (4) test-time adaptation/augmentation (TTA^2) to improve inference reliability. Experiments on five public benchmarks, AudioMNIST, SpeechCommands V1 & V2, VocalSound, and CochlScene, show that AMAuT achieves accuracies up to 99.8% while consuming less than 3% of the GPU hours required by comparable pre-trained models. Thus, AMAuT presents a highly efficient and flexible alternative to large pre-trained models, making state-of-the-art audio classification accessible in computationally constrained settings.
>
---
#### [replaced 003] Unrolled Creative Adversarial Network For Generating Novel Musical Pieces
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [https://arxiv.org/pdf/2501.00452v2](https://arxiv.org/pdf/2501.00452v2)**

> **作者:** Pratik Nag
>
> **摘要:** Music generation has emerged as a significant topic in artificial intelligence and machine learning. While recurrent neural networks (RNNs) have been widely employed for sequence generation, generative adversarial networks (GANs) remain relatively underexplored in this domain. This paper presents two systems based on adversarial networks for music generation. The first system learns a set of music pieces without differentiating between styles, while the second system focuses on learning and deviating from specific composers' styles to create innovative music. By extending the Creative Adversarial Networks (CAN) framework to the music domain, this work introduces unrolled CAN to address mode collapse, evaluating both GAN and CAN in terms of creativity and variation.
>
---
#### [replaced 004] CommonVoice-SpeechRE and RPG-MoGe: Advancing Speech Relation Extraction with a New Dataset and Multi-Order Generative Framework
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.08438v2](https://arxiv.org/pdf/2509.08438v2)**

> **作者:** Jinzhong Ning; Paerhati Tulajiang; Yingying Le; Yijia Zhang; Yuanyuan Sun; Hongfei Lin; Haifeng Liu
>
> **摘要:** Speech Relation Extraction (SpeechRE) aims to extract relation triplets directly from speech. However, existing benchmark datasets rely heavily on synthetic data, lacking sufficient quantity and diversity of real human speech. Moreover, existing models also suffer from rigid single-order generation templates and weak semantic alignment, substantially limiting their performance. To address these challenges, we introduce CommonVoice-SpeechRE, a large-scale dataset comprising nearly 20,000 real-human speech samples from diverse speakers, establishing a new benchmark for SpeechRE research. Furthermore, we propose the Relation Prompt-Guided Multi-Order Generative Ensemble (RPG-MoGe), a novel framework that features: (1) a multi-order triplet generation ensemble strategy, leveraging data diversity through diverse element orders during both training and inference, and (2) CNN-based latent relation prediction heads that generate explicit relation prompts to guide cross-modal alignment and accurate triplet generation. Experiments show our approach outperforms state-of-the-art methods, providing both a benchmark dataset and an effective solution for real-world SpeechRE. The source code and dataset are publicly available at https://github.com/NingJinzhong/SpeechRE_RPG_MoGe.
>
---
#### [replaced 005] Difficulty-Controlled Simplification of Piano Scores with Synthetic Data for Inclusive Music Education
- **分类: cs.SD**

- **链接: [https://arxiv.org/pdf/2511.16228v2](https://arxiv.org/pdf/2511.16228v2)**

> **作者:** Pedro Ramoneda; Emilia Parada-Cabaleiro; Dasaem Jeong; Xavier Serra
>
> **摘要:** Despite its potential, AI advances in music education are hindered by proprietary systems that limit the democratization of technology in this domain. In particular, AI-driven music difficulty adjustment is especially promising, as simplifying complex pieces can make music education more inclusive and accessible to learners of all ages and contexts. Nevertheless, recent efforts have relied on proprietary datasets, which prevents the research community from reproducing, comparing, or extending the current state of the art. In addition, while these generative methods offer great potential, most of them use the MIDI format, which, unlike others, such as MusicXML, lacks readability and layout information, thereby limiting their practical use for human performers. This work introduces a transformer-based method for adjusting the difficulty of MusicXML piano scores. Unlike previous methods, which rely on annotated datasets, we propose a synthetic dataset composed of pairs of piano scores ordered by estimated difficulty, with each pair comprising a more challenging and easier arrangement of the same piece. We generate these pairs by creating variations conditioned on the same melody and harmony and leverage pretrained models to assess difficulty and style, ensuring appropriate pairing. The experimental results illustrate the validity of the proposed approach, showing accurate control of playability and target difficulty, as highlighted through qualitative and quantitative evaluations. In contrast to previous work, we openly release all resources (code, dataset, and models), ensuring reproducibility while fostering open-source innovation to help bridge the digital divide.
>
---
#### [replaced 006] Speech Foundation Models Generalize to Time Series Tasks from Wearable Sensor Data
- **分类: cs.LG; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.00221v3](https://arxiv.org/pdf/2509.00221v3)**

> **作者:** Jaya Narain; Zakaria Aldeneh; Shirley Ren
>
> **备注:** Preprint, under review
>
> **摘要:** Both speech and sensor time series data encode information in both the time- and frequency- domains, like spectral powers and waveform shapelets. We show that speech foundation models learn representations that generalize beyond the speech domain and achieve state-of-the-art performance on diverse time-series tasks from wearable sensors. Probes trained on features extracted from HuBERT and wav2vec 2.0 outperform those extracted from self-supervised models trained directly on modality-specific datasets for mood classification, arrhythmia detection, and activity classification tasks. We find that the convolutional feature encoders of speech models are particularly relevant for wearable sensor applications. The proposed approach enhances performance on data-scarce time-series tasks using simple probing methods. This work takes a step toward developing generalized time-series models that unify speech and sensor modalities.
>
---
#### [replaced 007] Principled Coarse-Grained Acceptance for Speculative Decoding in Speech
- **分类: eess.AS; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.13732v2](https://arxiv.org/pdf/2511.13732v2)**

> **作者:** Moran Yanuka; Paul Dixon; Eyal Finkelshtein; Daniel Rotman; Raja Giryes
>
> **摘要:** Speculative decoding accelerates autoregressive speech generation by letting a fast draft model propose tokens that a larger target model verifies. However, for speech LLMs that generate acoustic tokens, exact token matching is overly restrictive: many discrete tokens are acoustically or semantically interchangeable, reducing acceptance rates and limiting speedups. We introduce Principled Coarse-Graining (PCG), which verifies proposals at the level of Acoustic Similarity Groups (ASGs) derived from the target model's embedding space. By splitting each token's probability mass across the overlapping groups that contain it, we define an overlap-aware coarse-grained distribution and perform rejection sampling on the resulting group variable. This yields an exactness guarantee at the group level while allowing the accepted draft token to stand in for any member of the group in practice. On LibriTTS, PCG increases acceptance and throughput relative to standard speculative decoding and prior speech-specific relaxations while maintaining intelligibility and speaker similarity. These results suggest acoustically aware, group-level acceptance as a simple and general way to accelerate speech token generation while maintaining speech quality.
>
---
#### [replaced 008] BemaGANv2: A Tutorial and Comparative Survey of GAN-based Vocoders for Long-Term Audio Generation
- **分类: cs.SD; cs.AI; cs.LG; cs.LO; eess.AS**

- **链接: [https://arxiv.org/pdf/2506.09487v2](https://arxiv.org/pdf/2506.09487v2)**

> **作者:** Taesoo Park; Mungwi Jeong; Mingyu Park; Narae Kim; Junyoung Kim; Mujung Kim; Jisang Yoo; Hoyun Lee; Sanghoon Kim; Soonchul Kwon
>
> **备注:** 11 pages, 7 figures. Survey and tutorial paper. Currently under review at ICT Express as an extended version of our ICAIIC 2025 paper
>
> **摘要:** This paper presents a tutorial-style survey and implementation guide of BemaGANv2, an advanced GANbased vocoder designed for high-fidelity and long-term audio generation. Long-term audio generation is critical for applications in Text-to-Music (TTM) and Text-to-Audio (TTA) systems, where maintaining temporal coherence, prosodic consistency, and harmonic structure over extended durations remains a significant challenge. Built upon the original BemaGAN architecture, BemaGANv2 incorporates major architectural innovations by replacing traditional ResBlocks in the generator with the Anti-aliased Multi-Periodicity composition (AMP) module, which internally applies the Snake activation function to better model periodic structures. In the discriminator framework, we integrate the Multi-Envelope Discriminator (MED), a novel architecture we proposed, to extract rich temporal envelope features crucial for periodicity detection. Coupled with the Multi-Resolution Discriminator (MRD), this combination enables more accurate modeling of long-range dependencies in audio. We systematically evaluate various discriminator configurations, including Multi-Scale Discriminator (MSD) + MED, MSD + MRD, and Multi-Period Discriminator (MPD) + MED + MRD, using objective metrics (Fréchet Audio Distance (FAD), Structural Similarity Index (SSIM), Pearson Correlation Coefficient (PCC), Mel-Cepstral Distortion (MCD)) and subjective evaluations (MOS, SMOS). This paper also provides a comprehensive tutorial on the model architecture, training methodology, and implementation to promote reproducibility. The code and pre-trained models are available at: https://github.com/dinhoitt/BemaGANv2.
>
---
#### [replaced 009] Learning Perceptually Relevant Temporal Envelope Morphing
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [https://arxiv.org/pdf/2506.01588v4](https://arxiv.org/pdf/2506.01588v4)**

> **作者:** Satvik Dixit; Sungjoon Park; Chris Donahue; Laurie M. Heller
>
> **备注:** Accepted at WASPAA 2025
>
> **摘要:** Temporal envelope morphing, the process of interpolating between the amplitude dynamics of two audio signals, is an emerging problem in generative audio systems that lacks sufficient perceptual grounding. Morphing of temporal envelopes in a perceptually intuitive manner should enable new methods for sound blending in creative media and for probing perceptual organization in psychoacoustics. However, existing audio morphing techniques often fail to produce intermediate temporal envelopes when input sounds have distinct temporal structures; many morphers effectively overlay both temporal structures, leading to perceptually unnatural results. In this paper, we introduce a novel workflow for learning envelope morphing with perceptual guidance: we first derive perceptually grounded morphing principles through human listening studies, then synthesize large-scale datasets encoding these principles, and finally train machine learning models to create perceptually intermediate morphs. Specifically, we present: (1) perceptual principles that guide envelope morphing, derived from our listening studies, (2) a supervised framework to learn these principles, (3) an autoencoder that learns to compress temporal envelope structures into latent representations, and (4) benchmarks for evaluating audio envelope morphs, using both synthetic and naturalistic data, and show that our approach outperforms existing methods in producing temporally intermediate morphs. All code, models, and checkpoints are available at https://github.com/TemporalMorphing/EnvelopeMorphing.
>
---
#### [replaced 010] Accelerating Automatic Differentiation of Direct Form Digital Filters
- **分类: eess.SY; eess.AS; eess.SP**

- **链接: [https://arxiv.org/pdf/2511.14390v2](https://arxiv.org/pdf/2511.14390v2)**

> **作者:** Chin-Yun Yu; György Fazekas
>
> **备注:** Accepted at the 1st Workshop on Differentiable Systems and Scientific Machine Learning @ EurIPS 2025
>
> **摘要:** We introduce a general formulation for automatic differentiation through direct form filters, yielding a closed-form backpropagation that includes initial condition gradients. The result is a single expression that can represent both the filter and its gradients computation while supporting parallelism. C++/CUDA implementations in PyTorch achieve at least 1000x speedup over naive Python implementations and consistently run fastest on the GPU. For the low-order filters commonly used in practice, exact time-domain filtering with analytical gradients outperforms the frequency-domain method in terms of speed. The source code is available at https://github.com/yoyolicoris/philtorch.
>
---
#### [replaced 011] SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech
- **分类: cs.SD**

- **链接: [https://arxiv.org/pdf/2511.07821v2](https://arxiv.org/pdf/2511.07821v2)**

> **作者:** Lu Gan; Xi Li
>
> **摘要:** The development of high-performance, on-device keyword spotting (KWS) systems for ultra-low-power hardware is critically constrained by the scarcity of specialized, multi-command training datasets. Traditional data collection through human recording is costly, slow, and lacks scalability. This paper introduces SYNTTS-COMMANDS, a novel, multilingual voice command dataset entirely generated using state-of-the-art Text-to-Speech (TTS) synthesis. By leveraging the CosyVoice 2 model and speaker embeddings from public corpora, we created a scalable collection of English and Chinese commands. Extensive benchmarking across a range of efficient acoustic models demonstrates that our synthetic dataset enables exceptional accuracy, achieving up to 99.5\% on English and 98\% on Chinese command recognition. These results robustly validate that synthetic speech can effectively replace human-recorded audio for training KWS classifiers. Our work directly addresses the data bottleneck in TinyML, providing a practical, scalable foundation for building private, low-latency, and energy-efficient voice interfaces on resource-constrained edge devices. The dataset and source code are publicly available at https://github.com/lugan113/SynTTS-Commands-Official.
>
---
#### [replaced 012] Audio Palette: A Diffusion Transformer with Multi-Signal Conditioning for Controllable Foley Synthesis
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2510.12175v3](https://arxiv.org/pdf/2510.12175v3)**

> **作者:** Junnuo Wang
>
> **备注:** Accepted for publication in the Artificial Intelligence Technology Research (AITR), Vol. 3, No. 2, December 2025
>
> **摘要:** Recent advances in diffusion-based generative models have enabled high-quality text-to-audio synthesis, but fine-grained acoustic control remains a significant challenge in open-source research. We present Audio Palette, a diffusion transformer (DiT) based model that extends the Stable Audio Open architecture to address this "control gap" in controllable audio generation. Unlike prior approaches that rely solely on semantic conditioning, Audio Palette introduces four time-varying control signals: loudness, pitch, spectral centroid, and timbre, for precise and interpretable manipulation of acoustic features. The model is efficiently adapted for the nuanced domain of Foley synthesis using Low-Rank Adaptation (LoRA) on a curated subset of AudioSet, requiring only 0.85 percent of the original parameters to be trained. Experiments demonstrate that Audio Palette achieves fine-grained, interpretable control of sound attributes. Crucially, it accomplishes this novel controllability while maintaining high audio quality and strong semantic alignment to text prompts, with performance on standard metrics such as Frechet Audio Distance (FAD) and LAION-CLAP scores remaining comparable to the original baseline model. We provide a scalable, modular pipeline for audio research, emphasizing sequence-based conditioning, memory efficiency, and a three-scale classifier-free guidance mechanism for nuanced inference-time control. This work establishes a robust foundation for controllable sound design and performative audio synthesis in open-source settings, enabling a more artist-centric workflow.
>
---
#### [replaced 013] FoleyBench: A Benchmark For Video-to-Audio Models
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.13219v2](https://arxiv.org/pdf/2511.13219v2)**

> **作者:** Satvik Dixit; Koichi Saito; Zhi Zhong; Yuki Mitsufuji; Chris Donahue
>
> **摘要:** Video-to-audio generation (V2A) is of increasing importance in domains such as film post-production, AR/VR, and sound design, particularly for the creation of Foley sound effects synchronized with on-screen actions. Foley requires generating audio that is both semantically aligned with visible events and temporally aligned with their timing. Yet, there is a mismatch between evaluation and downstream applications due to the absence of a benchmark tailored to Foley-style scenarios. We find that 74% of videos from past evaluation datasets have poor audio-visual correspondence. Moreover, they are dominated by speech and music, domains that lie outside the use case for Foley. To address this gap, we introduce FoleyBench, the first large-scale benchmark explicitly designed for Foley-style V2A evaluation. FoleyBench contains 5,000 (video, ground-truth audio, text caption) triplets, each featuring visible sound sources with audio causally tied to on-screen events. The dataset is built using an automated, scalable pipeline applied to in-the-wild internet videos from YouTube-based and Vimeo-based sources. Compared to past datasets, we show that videos from FoleyBench have stronger coverage of sound categories from a taxonomy specifically designed for Foley sound. Each clip is further labeled with metadata capturing source complexity, UCS/AudioSet category, and video length, enabling fine-grained analysis of model performance and failure modes. We benchmark several state-of-the-art V2A models, evaluating them on audio quality, audio-video alignment, temporal synchronization, and audio-text consistency. Samples are available at: https://gclef-cmu.org/foleybench
>
---
#### [replaced 014] Warm Chat: Diffuse Emotion-aware Interactive Talking Head Avatar with Tree-Structured Guidance
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [https://arxiv.org/pdf/2508.18337v3](https://arxiv.org/pdf/2508.18337v3)**

> **作者:** Haijie Yang; Zhenyu Zhang; Hao Tang; Jianjun Qian; Jian Yang
>
> **备注:** The submission is withdrawn at the request of the authors due to internal reasons within the research team
>
> **摘要:** Generative models have advanced rapidly, enabling impressive talking head generation that brings AI to life. However, most existing methods focus solely on one-way portrait animation. Even the few that support bidirectional conversational interactions lack precise emotion-adaptive capabilities, significantly limiting their practical applicability. In this paper, we propose Warm Chat, a novel emotion-aware talking head generation framework for dyadic interactions. Leveraging the dialogue generation capability of large language models (LLMs, e.g., GPT-4), our method produces temporally consistent virtual avatars with rich emotional variations that seamlessly transition between speaking and listening states. Specifically, we design a Transformer-based head mask generator that learns temporally consistent motion features in a latent mask space, capable of generating arbitrary-length, temporally consistent mask sequences to constrain head motions. Furthermore, we introduce an interactive talking tree structure to represent dialogue state transitions, where each tree node contains information such as child/parent/sibling nodes and the current character's emotional state. By performing reverse-level traversal, we extract rich historical emotional cues from the current node to guide expression synthesis. Extensive experiments demonstrate the superior performance and effectiveness of our method.
>
---
