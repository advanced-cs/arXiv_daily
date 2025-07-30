# 音频 cs.SD;  eess.SP

- **最新发布 23 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决零样本语音合成中隐私泄露问题。提出教师引导遗忘框架，使模型忘记特定说话人身份，同时保持其他语音合成质量，并设计新指标评估遗忘效果。**

- **链接: [http://arxiv.org/pdf/2507.20140v1](http://arxiv.org/pdf/2507.20140v1)**

> **作者:** Taesoo Kim; Jinju Kim; Dongchan Kim; Jong Hwan Ko; Gyeong-Moon Park
>
> **备注:** Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), Vancouver, Canada. PMLR 267, 2025. Authors Jinju Kim and Taesoo Kim contributed equally
>
> **摘要:** The rapid advancement of Zero-Shot Text-to-Speech (ZS-TTS) technology has enabled high-fidelity voice synthesis from minimal audio cues, raising significant privacy and ethical concerns. Despite the threats to voice privacy, research to selectively remove the knowledge to replicate unwanted individual voices from pre-trained model parameters has not been explored. In this paper, we address the new challenge of speaker identity unlearning for ZS-TTS systems. To meet this goal, we propose the first machine unlearning frameworks for ZS-TTS, especially Teacher-Guided Unlearning (TGU), designed to ensure the model forgets designated speaker identities while retaining its ability to generate accurate speech for other speakers. Our proposed methods incorporate randomness to prevent consistent replication of forget speakers' voices, assuring unlearned identities remain untraceable. Additionally, we propose a new evaluation metric, speaker-Zero Retrain Forgetting (spk-ZRF). This assesses the model's ability to disregard prompts associated with forgotten speakers, effectively neutralizing its knowledge of these voices. The experiments conducted on the state-of-the-art model demonstrate that TGU prevents the model from replicating forget speakers' voices while maintaining high quality for other speakers. The demo is available at https://speechunlearn.github.io/
>
---
#### [new 002] JAM: A Tiny Flow-based Song Generator with Fine-grained Controllability and Aesthetic Alignment
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于歌词到歌曲生成任务，旨在解决现有模型缺乏细粒度控制及生成质量不足的问题。作者提出了JAM模型，基于流匹配实现词级时序控制，并通过美学对齐优化提升生成效果，同时构建了评估数据集JAME。**

- **链接: [http://arxiv.org/pdf/2507.20880v1](http://arxiv.org/pdf/2507.20880v1)**

> **作者:** Renhang Liu; Chia-Yu Hung; Navonil Majumder; Taylor Gautreaux; Amir Ali Bagherzadeh; Chuan Li; Dorien Herremans; Soujanya Poria
>
> **备注:** https://github.com/declare-lab/jamify
>
> **摘要:** Diffusion and flow-matching models have revolutionized automatic text-to-audio generation in recent times. These models are increasingly capable of generating high quality and faithful audio outputs capturing to speech and acoustic events. However, there is still much room for improvement in creative audio generation that primarily involves music and songs. Recent open lyrics-to-song models, such as, DiffRhythm, ACE-Step, and LeVo, have set an acceptable standard in automatic song generation for recreational use. However, these models lack fine-grained word-level controllability often desired by musicians in their workflows. To the best of our knowledge, our flow-matching-based JAM is the first effort toward endowing word-level timing and duration control in song generation, allowing fine-grained vocal control. To enhance the quality of generated songs to better align with human preferences, we implement aesthetic alignment through Direct Preference Optimization, which iteratively refines the model using a synthetic dataset, eliminating the need or manual data annotations. Furthermore, we aim to standardize the evaluation of such lyrics-to-song models through our public evaluation dataset JAME. We show that JAM outperforms the existing models in terms of the music-specific attributes.
>
---
#### [new 003] Sound Safeguarding for Acoustic Measurement Using Any Sounds: Tools and Applications
- **分类: cs.SD; eess.AS; 68-06; J.2**

- **简介: 该论文属于音频信号处理任务，旨在解决传统声学测量对专用声音的依赖问题。作者提出了“声音守护”方法，开发了基于任意声音进行声学测量的工具，包括准备、实时测量和报告生成，并通过实际应用改进方法，最终开源工具以改善声学环境。**

- **链接: [http://arxiv.org/pdf/2507.20485v1](http://arxiv.org/pdf/2507.20485v1)**

> **作者:** Hideki Kawahara; Kohei Yatabe; Ken-Ichi Sakakibara
>
> **备注:** 2 pages, 2 figures, IEEE GCCE 2025 Demo session, Accepted
>
> **摘要:** We demonstrate tools and applications developed based on the method of "sound safeguarding," which enables any sound to be used for acoustic measurements. We developed tools for preparation, interactive and real-time measurement, and report generation. We extended and modified the method during its development based on its application in various practical situations. We have open-sourced these tools and encourage prospective users to use them to improve their acoustic environments.
>
---
#### [new 004] SonicGauss: Position-Aware Physical Sound Synthesis for 3D Gaussian Representations
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于3D视觉与声音合成交叉任务，旨在解决从3D高斯表示生成位置感知的物理碰撞声音问题。作者提出SonicGauss框架，结合扩散模型与PointTransformer，利用3DGS的几何与材质特性合成声音，实现实时、位置相关的听觉反馈。**

- **链接: [http://arxiv.org/pdf/2507.19835v1](http://arxiv.org/pdf/2507.19835v1)**

> **作者:** Chunshi Wang; Hongxing Li; Yawei Luo
>
> **备注:** Accepted by ACMMM'25
>
> **摘要:** While 3D Gaussian representations (3DGS) have proven effective for modeling the geometry and appearance of objects, their potential for capturing other physical attributes-such as sound-remains largely unexplored. In this paper, we present a novel framework dubbed SonicGauss for synthesizing impact sounds from 3DGS representations by leveraging their inherent geometric and material properties. Specifically, we integrate a diffusion-based sound synthesis model with a PointTransformer-based feature extractor to infer material characteristics and spatial-acoustic correlations directly from Gaussian ellipsoids. Our approach supports spatially varying sound responses conditioned on impact locations and generalizes across a wide range of object categories. Experiments on the ObjectFolder dataset and real-world recordings demonstrate that our method produces realistic, position-aware auditory feedback. The results highlight the framework's robustness and generalization ability, offering a promising step toward bridging 3D visual representations and interactive sound synthesis. Project page: https://chunshi.wang/SonicGauss
>
---
#### [new 005] Music Arena: Live Evaluation for Text-to-Music
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文属于文本到音乐生成任务，旨在解决现有文本到音乐系统评估成本高、标准不统一、缺乏公开偏好数据的问题。论文工作设计了Music Arena平台，提供实时用户偏好评估、自然语言反馈收集及滚动数据发布，支持透明、可比较的模型评估。**

- **链接: [http://arxiv.org/pdf/2507.20900v1](http://arxiv.org/pdf/2507.20900v1)**

> **作者:** Yonghyun Kim; Wayne Chi; Anastasios N. Angelopoulos; Wei-Lin Chiang; Koichi Saito; Shinji Watanabe; Yuki Mitsufuji; Chris Donahue
>
> **摘要:** We present Music Arena, an open platform for scalable human preference evaluation of text-to-music (TTM) models. Soliciting human preferences via listening studies is the gold standard for evaluation in TTM, but these studies are expensive to conduct and difficult to compare, as study protocols may differ across systems. Moreover, human preferences might help researchers align their TTM systems or improve automatic evaluation metrics, but an open and renewable source of preferences does not currently exist. We aim to fill these gaps by offering *live* evaluation for TTM. In Music Arena, real-world users input text prompts of their choosing and compare outputs from two TTM systems, and their preferences are used to compile a leaderboard. While Music Arena follows recent evaluation trends in other AI domains, we also design it with key features tailored to music: an LLM-based routing system to navigate the heterogeneous type signatures of TTM systems, and the collection of *detailed* preferences including listening data and natural language feedback. We also propose a rolling data release policy with user privacy guarantees, providing a renewable source of preference data and increasing platform transparency. Through its standardized evaluation protocol, transparent data access policies, and music-specific features, Music Arena not only addresses key challenges in the TTM ecosystem but also demonstrates how live evaluation can be thoughtfully adapted to unique characteristics of specific AI domains. Music Arena is available at: https://music-arena.org
>
---
#### [new 006] Hyperbolic Embeddings for Order-Aware Classification of Audio Effect Chains
- **分类: cs.SD**

- **简介: 该论文属于音频信号处理任务，旨在解决从湿信号中联合估计音频效果器类型及其顺序的问题。现有研究多关注效果器类型和参数估计，忽视了效果器顺序的重要性。为此，论文提出一种基于神经网络的方法，将湿信号嵌入超球面空间以建模音频效果链的非交换性和指数增长特性，从而更准确地识别效果器类型及顺序。实验表明该方法优于传统欧几里得空间方法。**

- **链接: [http://arxiv.org/pdf/2507.20624v1](http://arxiv.org/pdf/2507.20624v1)**

> **作者:** Aogu Wada; Tomohiko Nakamura; Hiroshi Saruwatari
>
> **备注:** 7 pages, 3 figures, accepted for the 28th International Conference on Digital Audio Effects (DAFx25)
>
> **摘要:** Audio effects (AFXs) are essential tools in music production, frequently applied in chains to shape timbre and dynamics. The order of AFXs in a chain plays a crucial role in determining the final sound, particularly when non-linear (e.g., distortion) or time-variant (e.g., chorus) processors are involved. Despite its importance, most AFX-related studies have primarily focused on estimating effect types and their parameters from a wet signal. To address this gap, we formulate AFX chain recognition as the task of jointly estimating AFX types and their order from a wet signal. We propose a neural-network-based method that embeds wet signals into a hyperbolic space and classifies their AFX chains. Hyperbolic space can represent tree-structured data more efficiently than Euclidean space due to its exponential expansion property. Since AFX chains can be represented as trees, with AFXs as nodes and edges encoding effect order, hyperbolic space is well-suited for modeling the exponentially growing and non-commutative nature of ordered AFX combinations, where changes in effect order can result in different final sounds. Experiments using guitar sounds demonstrate that, with an appropriate curvature, the proposed method outperforms its Euclidean counterpart. Further analysis based on AFX type and chain length highlights the effectiveness of the proposed method in capturing AFX order.
>
---
#### [new 007] Improving Deep Learning-based Respiratory Sound Analysis with Frequency Selection and Attention Mechanism
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于呼吸音分类任务，旨在提升深度学习模型对呼吸音的准确分类。为解决现有模型在全局上下文建模和计算效率上的不足，论文提出了轻量级的CNN-TSA网络和频率带选择（FBS）模块，有效抑制噪声并降低计算量。实验表明，该方法在多个数据集上取得优异性能，适合资源受限场景的实时分析。**

- **链接: [http://arxiv.org/pdf/2507.20052v1](http://arxiv.org/pdf/2507.20052v1)**

> **作者:** Nouhaila Fraihi; Ouassim Karrakchou; Mounir Ghogho
>
> **摘要:** Accurate classification of respiratory sounds requires deep learning models that effectively capture fine-grained acoustic features and long-range temporal dependencies. Convolutional Neural Networks (CNNs) are well-suited for extracting local time-frequency patterns but are limited in modeling global context. In contrast, transformer-based models can capture long-range dependencies, albeit with higher computational demands. To address these limitations, we propose a compact CNN-Temporal Self-Attention (CNN-TSA) network that integrates lightweight self-attention into an efficient CNN backbone. Central to our approach is a Frequency Band Selection (FBS) module that suppresses noisy and non-informative frequency regions, substantially improving accuracy and reducing FLOPs by up to 50%. We also introduce age-specific models to enhance robustness across diverse patient groups. Evaluated on the SPRSound-2022/2023 and ICBHI-2017 lung sound datasets, CNN-TSA with FBS sets new benchmarks on SPRSound and achieves state-of-the-art performance on ICBHI, all with a significantly smaller computational footprint. Furthermore, integrating FBS into an existing transformer baseline yields a new record on ICBHI, confirming FBS as an effective drop-in enhancement. These results demonstrate that our framework enables reliable, real-time respiratory sound analysis suitable for deployment in resource-constrained settings.
>
---
#### [new 008] Multipath Interference Suppression in Indirect Time-of-Flight Imaging via a Novel Compressed Sensing Framework
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于深度成像任务，旨在提升间接飞行时间（iToF）系统的深度重建精度和多目标分离能力。论文提出了一种新的压缩感知方法，通过多相移和窄占空比连续波构建传感矩阵，并结合K-Means聚类优化正交匹配追踪（OMP）过程，有效抑制多路径干扰，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2507.19546v1](http://arxiv.org/pdf/2507.19546v1)**

> **作者:** Yansong Du; Yutong Deng; Yuting Zhou; Feiyu Jiao; Bangyao Wang; Zhancong Xu; Zhaoxiang Jiang; Xun Guan
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** We propose a novel compressed sensing method to improve the depth reconstruction accuracy and multi-target separation capability of indirect Time-of-Flight (iToF) systems. Unlike traditional approaches that rely on hardware modifications, complex modulation, or cumbersome data-driven reconstruction, our method operates with a single modulation frequency and constructs the sensing matrix using multiple phase shifts and narrow-duty-cycle continuous waves. During matrix construction, we further account for pixel-wise range variation caused by lens distortion, making the sensing matrix better aligned with actual modulation response characteristics. To enhance sparse recovery, we apply K-Means clustering to the distance response dictionary and constrain atom selection within each cluster during the OMP process, which effectively reduces the search space and improves solution stability. Experimental results demonstrate that the proposed method outperforms traditional approaches in both reconstruction accuracy and robustness, without requiring any additional hardware changes.
>
---
#### [new 009] Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐生成任务，旨在解决现有AI音乐系统参数多、推理慢的问题。作者提出了一种轻量级潜在扩散模型，结合软对齐注意力机制，实现高效的人声条件音乐伴奏生成，具备实时部署能力。**

- **链接: [http://arxiv.org/pdf/2507.19991v1](http://arxiv.org/pdf/2507.19991v1)**

> **作者:** Hei Shing Cheung; Boya Zhang
>
> **备注:** 6 page, 3 figures
>
> **摘要:** We present a lightweight latent diffusion model for vocal-conditioned musical accompaniment generation that addresses critical limitations in existing music AI systems. Our approach introduces a novel soft alignment attention mechanism that adaptively combines local and global temporal dependencies based on diffusion timesteps, enabling efficient capture of multi-scale musical structure. Operating in the compressed latent space of a pre-trained variational autoencoder, the model achieves a 220 times parameter reduction compared to state-of-the-art systems while delivering 52 times faster inference. Experimental evaluation demonstrates competitive performance with only 15M parameters, outperforming OpenAI Jukebox in production quality and content unity while maintaining reasonable musical coherence. The ultra-lightweight architecture enables real-time deployment on consumer hardware, making AI-assisted music creation accessible for interactive applications and resource-constrained environments.
>
---
#### [new 010] Joint Feature and Output Distillation for Low-complexity Acoustic Scene Classification
- **分类: cs.SD; eess.AS; I.2.6**

- **简介: 论文属于DCASE2025任务1的低复杂度声学场景分类，旨在提升轻量模型的分类性能。作者提出双级知识蒸馏方法，结合教师模型的软标签和中间特征，引导学生模型学习语义分布与结构信息，最终在数据集上取得59.30%的准确率。**

- **链接: [http://arxiv.org/pdf/2507.19557v1](http://arxiv.org/pdf/2507.19557v1)**

> **作者:** Haowen Li; Ziyi Yang; Mou Wang; Ee-Leng Tan; Junwei Yeow; Santi Peksi; Woon-Seng Gan
>
> **备注:** 4 pages, submitted to DCASE2025 Challenge Task 1
>
> **摘要:** This report presents a dual-level knowledge distillation framework with multi-teacher guidance for low-complexity acoustic scene classification (ASC) in DCASE2025 Task 1. We propose a distillation strategy that jointly transfers both soft logits and intermediate feature representations. Specifically, we pre-trained PaSST and CP-ResNet models as teacher models. Logits from teachers are averaged to generate soft targets, while one CP-ResNet is selected for feature-level distillation. This enables the compact student model (CP-Mobile) to capture both semantic distribution and structural information from teacher guidance. Experiments on the TAU Urban Acoustic Scenes 2022 Mobile dataset (development set) demonstrate that our submitted systems achieve up to 59.30\% accuracy.
>
---
#### [new 011] Improving Audio Classification by Transitioning from Zero- to Few-Shot
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频分类任务，旨在解决零样本分类中因文本描述不准确导致的分类精度问题。论文通过引入少样本方法，利用音频嵌入按类别分组处理，以替代有噪声的文本嵌入，从而提升分类效果。实验表明，少样本方法通常优于零样本基线。**

- **链接: [http://arxiv.org/pdf/2507.20036v1](http://arxiv.org/pdf/2507.20036v1)**

> **作者:** James Taylor; Wolfgang Mack
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** State-of-the-art audio classification often employs a zero-shot approach, which involves comparing audio embeddings with embeddings from text describing the respective audio class. These embeddings are usually generated by neural networks trained through contrastive learning to align audio and text representations. Identifying the optimal text description for an audio class is challenging, particularly when the class comprises a wide variety of sounds. This paper examines few-shot methods designed to improve classification accuracy beyond the zero-shot approach. Specifically, audio embeddings are grouped by class and processed to replace the inherently noisy text embeddings. Our results demonstrate that few-shot classification typically outperforms the zero-shot baseline.
>
---
#### [new 012] Diffusion-based Symbolic Music Generation with Structured State Space Models
- **分类: cs.SD**

- **简介: 该论文属于符号音乐生成任务，旨在解决现有模型处理长序列时计算效率低的问题。论文提出SMDIM模型，结合扩散模型与结构化状态空间模型，利用MFA模块实现高效全局建模与局部细节保留，提升了生成质量与计算效率。**

- **链接: [http://arxiv.org/pdf/2507.20128v1](http://arxiv.org/pdf/2507.20128v1)**

> **作者:** Shenghua Yuan; Xing Tang; Jiatao Chen; Tianming Xie; Jing Wang; Bing Shi
>
> **备注:** 9 pages,3figures
>
> **摘要:** Recent advancements in diffusion models have significantly improved symbolic music generation. However, most approaches rely on transformer-based architectures with self-attention mechanisms, which are constrained by quadratic computational complexity, limiting scalability for long sequences. To address this, we propose Symbolic Music Diffusion with Mamba (SMDIM), a novel diffusion-based architecture integrating Structured State Space Models (SSMs) for efficient global context modeling and the Mamba-FeedForward-Attention Block (MFA) for precise local detail preservation. The MFA Block combines the linear complexity of Mamba layers, the non-linear refinement of FeedForward layers, and the fine-grained precision of self-attention mechanisms, achieving a balance between scalability and musical expressiveness. SMDIM achieves near-linear complexity, making it highly efficient for long-sequence tasks. Evaluated on diverse datasets, including FolkDB, a collection of traditional Chinese folk music that represents an underexplored domain in symbolic music generation, SMDIM outperforms state-of-the-art models in both generation quality and computational efficiency. Beyond symbolic music, SMDIM's architectural design demonstrates adaptability to a broad range of long-sequence generation tasks, offering a scalable and efficient solution for coherent sequence modeling.
>
---
#### [new 013] Two Views, One Truth: Spectral and Self-Supervised Features Fusion for Robust Speech Deepfake Detection
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有方法对未知伪造算法泛化能力差的问题。作者融合自监督学习特征与手工设计的频谱特征，通过多种策略（如交叉注意力）提升检测鲁棒性，并在多个数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.20417v1](http://arxiv.org/pdf/2507.20417v1)**

> **作者:** Yassine El Kheir; Arnab Das; Enes Erdem Erdogan; Fabian Ritter-Guttierez; Tim Polzehl; Sebastian Möller
>
> **备注:** ACCEPTED WASPAA 2025
>
> **摘要:** Recent advances in synthetic speech have made audio deepfakes increasingly realistic, posing significant security risks. Existing detection methods that rely on a single modality, either raw waveform embeddings or spectral based features, are vulnerable to non spoof disturbances and often overfit to known forgery algorithms, resulting in poor generalization to unseen attacks. To address these shortcomings, we investigate hybrid fusion frameworks that integrate self supervised learning (SSL) based representations with handcrafted spectral descriptors (MFCC , LFCC, CQCC). By aligning and combining complementary information across modalities, these fusion approaches capture subtle artifacts that single feature approaches typically overlook. We explore several fusion strategies, including simple concatenation, cross attention, mutual cross attention, and a learnable gating mechanism, to optimally blend SSL features with fine grained spectral cues. We evaluate our approach on four challenging public benchmarks and report generalization performance. All fusion variants consistently outperform an SSL only baseline, with the cross attention strategy achieving the best generalization with a 38% relative reduction in equal error rate (EER). These results confirm that joint modeling of waveform and spectral views produces robust, domain agnostic representations for audio deepfake detection.
>
---
#### [new 014] Self-Improvement for Audio Large Language Model using Unlabeled Speech
- **分类: cs.SD; eess.AS; I.2.7; H.5.5**

- **简介: 该论文属于语音处理任务，旨在解决音频大模型在特定领域因缺乏标注数据导致的性能下降问题。作者提出了一种基于强化学习的自改进方法SI-SDA，利用未标注语音数据生成伪标签并优化模型，提升了语音识别、问答和翻译等任务的表现。**

- **链接: [http://arxiv.org/pdf/2507.20169v1](http://arxiv.org/pdf/2507.20169v1)**

> **作者:** Shaowen Wang; Xinyuan Chen; Yao Xu
>
> **备注:** To appear in Interspeech 2025. 6 pages, 1 figure
>
> **摘要:** Recent audio LLMs have emerged rapidly, demonstrating strong generalization across various speech tasks. However, given the inherent complexity of speech signals, these models inevitably suffer from performance degradation in specific target domains. To address this, we focus on enhancing audio LLMs in target domains without any labeled data. We propose a self-improvement method called SI-SDA, leveraging the information embedded in large-model decoding to evaluate the quality of generated pseudo labels and then perform domain adaptation based on reinforcement learning optimization. Experimental results show that our method consistently and significantly improves audio LLM performance, outperforming existing baselines in WER and BLEU across multiple public datasets of automatic speech recognition (ASR), spoken question-answering (SQA), and speech-to-text translation (S2TT). Furthermore, our approach exhibits high data efficiency, underscoring its potential for real-world deployment.
>
---
#### [new 015] Learning Neural Vocoder from Range-Null Space Decomposition
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决神经声码器建模不透明和参数性能权衡问题。作者提出了一种基于时频域的神经声码器，结合信号范围-零空间分解理论，设计双路径框架进行频谱编码解码，实现了高性能轻量化语音生成。**

- **链接: [http://arxiv.org/pdf/2507.20731v1](http://arxiv.org/pdf/2507.20731v1)**

> **作者:** Andong Li; Tong Lei; Zhihang Sun; Rilin Chen; Erwei Yin; Xiaodong Li; Chengshi Zheng
>
> **备注:** 10 pages, 7 figures, IJCAI2025
>
> **摘要:** Despite the rapid development of neural vocoders in recent years, they usually suffer from some intrinsic challenges like opaque modeling, and parameter-performance trade-off. In this study, we propose an innovative time-frequency (T-F) domain-based neural vocoder to resolve the above-mentioned challenges. To be specific, we bridge the connection between the classical signal range-null decomposition (RND) theory and vocoder task, and the reconstruction of target spectrogram can be decomposed into the superimposition between the range-space and null-space, where the former is enabled by a linear domain shift from the original mel-scale domain to the target linear-scale domain, and the latter is instantiated via a learnable network for further spectral detail generation. Accordingly, we propose a novel dual-path framework, where the spectrum is hierarchically encoded/decoded, and the cross- and narrow-band modules are elaborately devised for efficient sub-band and sequential modeling. Comprehensive experiments are conducted on the LJSpeech and LibriTTS benchmarks. Quantitative and qualitative results show that while enjoying lightweight network parameters, the proposed approach yields state-of-the-art performance among existing advanced methods. Our code and the pretrained model weights are available at https://github.com/Andong-Li-speech/RNDVoC.
>
---
#### [new 016] Binaural Localization Model for Speech in Noise
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音信号处理任务，旨在解决噪声环境下双耳语音定位问题。作者提出了一个轻量级卷积循环网络模型，模拟人类听觉阈值，用于定位混响噪声中的语音源，并与传统算法对比性能，同时评估其作为双耳语音增强方法中线索保持度的指标。**

- **链接: [http://arxiv.org/pdf/2507.20027v1](http://arxiv.org/pdf/2507.20027v1)**

> **作者:** Vikas Tokala; Eric Grinstein; Rory Brooks; Mike Brookes; Simon Doclo; Jesper Jensen; Patrick A. Naylor
>
> **摘要:** Binaural acoustic source localization is important to human listeners for spatial awareness, communication and safety. In this paper, an end-to-end binaural localization model for speech in noise is presented. A lightweight convolutional recurrent network that localizes sound in the frontal azimuthal plane for noisy reverberant binaural signals is introduced. The model incorporates additive internal ear noise to represent the frequency-dependent hearing threshold of a typical listener. The localization performance of the model is compared with the steered response power algorithm, and the use of the model as a measure of interaural cue preservation for binaural speech enhancement methods is studied. A listening test was performed to compare the performance of the model with human localization of speech in noisy conditions.
>
---
#### [new 017] MIMII-Agent: Leveraging LLMs with Function Calling for Relative Evaluation of Anomalous Sound Detection
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于异常声音检测任务，旨在解决无监督场景下缺乏真实异常数据导致的检测效果难以评估的问题。论文提出MIMII-Agent方法，利用大语言模型生成机器类型特定的异常声音，实现对不同机器类型的检测性能相对评估。**

- **链接: [http://arxiv.org/pdf/2507.20666v1](http://arxiv.org/pdf/2507.20666v1)**

> **作者:** Harsh Purohit; Tomoya Nishida; Kota Dohi; Takashi Endo; Yohei Kawaguchi
>
> **摘要:** This paper proposes a method for generating machine-type-specific anomalies to evaluate the relative performance of unsupervised anomalous sound detection (UASD) systems across different machine types, even in the absence of real anomaly sound data. Conventional keyword-based data augmentation methods often produce unrealistic sounds due to their reliance on manually defined labels, limiting scalability as machine types and anomaly patterns diversify. Advanced audio generative models, such as MIMII-Gen, show promise but typically depend on anomalous training data, making them less effective when diverse anomalous examples are unavailable. To address these limitations, we propose a novel synthesis approach leveraging large language models (LLMs) to interpret textual descriptions of faults and automatically select audio transformation functions, converting normal machine sounds into diverse and plausible anomalous sounds. We validate this approach by evaluating a UASD system trained only on normal sounds from five machine types, using both real and synthetic anomaly data. Experimental results reveal consistent trends in relative detection difficulty across machine types between synthetic and real anomalies. This finding supports our hypothesis and highlights the effectiveness of the proposed LLM-based synthesis approach for relative evaluation of UASD systems.
>
---
#### [new 018] Binaural Speech Enhancement Using Complex Convolutional Recurrent Networks
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音增强任务，旨在提升双耳语音的可懂度与舒适度。针对单目标说话人和各向同性噪声场景，提出一种基于复数卷积循环网络的端到端方法，通过编码器-解码器结构与复数LSTM模块，估计双耳信号的复数比值掩码，并设计保留空间信息的损失函数，有效提升语音质量。**

- **链接: [http://arxiv.org/pdf/2507.20023v1](http://arxiv.org/pdf/2507.20023v1)**

> **作者:** Vikas Tokala; Eric Grinstein; Mike Brookes; Simon Doclo; Jesper Jensen; Patrick A. Naylor
>
> **摘要:** From hearing aids to augmented and virtual reality devices, binaural speech enhancement algorithms have been established as state-of-the-art techniques to improve speech intelligibility and listening comfort. In this paper, we present an end-to-end binaural speech enhancement method using a complex recurrent convolutional network with an encoder-decoder architecture and a complex LSTM recurrent block placed between the encoder and decoder. A loss function that focuses on the preservation of spatial information in addition to speech intelligibility improvement and noise reduction is introduced. The network estimates individual complex ratio masks for the left and right-ear channels of a binaural hearing device in the time-frequency domain. We show that, compared to other baseline algorithms, the proposed method significantly improves the estimated speech intelligibility and reduces the noise while preserving the spatial information of the binaural signals in acoustic situations with a single target speaker and isotropic noise of various types.
>
---
#### [new 019] ChoreoMuse: Robust Music-to-Dance Video Generation with Style Transfer and Beat-Adherent Motion
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音乐驱动舞蹈视频生成任务，旨在解决现有方法在音乐节奏与风格适配性、视频质量及个性化表达上的不足。论文提出ChoreoMuse框架，结合风格迁移与节拍对齐技术，实现高质量、可控风格的舞蹈视频生成。**

- **链接: [http://arxiv.org/pdf/2507.19836v1](http://arxiv.org/pdf/2507.19836v1)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 10 pages, 5 figures, accepted by the 33rd ACM International Conference on Multimedia (ACM MM 2025), demo page: https://choreomuse.github.io
>
> **摘要:** Modern artistic productions increasingly demand automated choreography generation that adapts to diverse musical styles and individual dancer characteristics. Existing approaches often fail to produce high-quality dance videos that harmonize with both musical rhythm and user-defined choreography styles, limiting their applicability in real-world creative contexts. To address this gap, we introduce ChoreoMuse, a diffusion-based framework that uses SMPL format parameters and their variation version as intermediaries between music and video generation, thereby overcoming the usual constraints imposed by video resolution. Critically, ChoreoMuse supports style-controllable, high-fidelity dance video generation across diverse musical genres and individual dancer characteristics, including the flexibility to handle any reference individual at any resolution. Our method employs a novel music encoder MotionTune to capture motion cues from audio, ensuring that the generated choreography closely follows the beat and expressive qualities of the input music. To quantitatively evaluate how well the generated dances match both musical and choreographic styles, we introduce two new metrics that measure alignment with the intended stylistic cues. Extensive experiments confirm that ChoreoMuse achieves state-of-the-art performance across multiple dimensions, including video quality, beat alignment, dance diversity, and style adherence, demonstrating its potential as a robust solution for a wide range of creative applications. Video results can be found on our project page: https://choreomuse.github.io.
>
---
#### [new 020] Binaural Sound Event Localization and Detection based on HRTF Cues for Humanoid Robots
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声音事件定位与检测任务，旨在通过双耳音频联合检测并定位多个声音事件。论文提出了双耳时间频率特征（BTFF）和基于CRNN的BiSELDnet模型，在合成数据集Binaural Set上验证方法有效性，实现了高精度的声音事件定位与检测。**

- **链接: [http://arxiv.org/pdf/2507.20530v1](http://arxiv.org/pdf/2507.20530v1)**

> **作者:** Gyeong-Tae Lee; Hyeonuk Nam; Yong-Hwa Park
>
> **备注:** Submitted to IEEE/ACM TASLP
>
> **摘要:** This paper introduces Binaural Sound Event Localization and Detection (BiSELD), a task that aims to jointly detect and localize multiple sound events using binaural audio, inspired by the spatial hearing mechanism of humans. To support this task, we present a synthetic benchmark dataset, called the Binaural Set, which simulates realistic auditory scenes using measured head-related transfer functions (HRTFs) and diverse sound events. To effectively address the BiSELD task, we propose a new input feature representation called the Binaural Time-Frequency Feature (BTFF), which encodes interaural time difference (ITD), interaural level difference (ILD), and high-frequency spectral cues (SC) from binaural signals. BTFF is composed of eight channels, including left and right mel-spectrograms, velocity-maps, SC-maps, and ITD-/ILD-maps, designed to cover different spatial cues across frequency bands and spatial axes. A CRNN-based model, BiSELDnet, is then developed to learn both spectro-temporal patterns and HRTF-based localization cues from BTFF. Experiments on the Binaural Set show that each BTFF sub-feature enhances task performance: V-map improves detection, ITD-/ILD-maps enable accurate horizontal localization, and SC-map captures vertical spatial cues. The final system achieves a SELD error of 0.110 with 87.1% F-score and 4.4{\deg} localization error, demonstrating the effectiveness of the proposed framework in mimicking human-like auditory perception.
>
---
#### [new 021] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks
- **分类: cs.CL; cs.AI; cs.CV; cs.SD**

- **简介: 该论文提出了MCIF，一个跨语言、多模态的指令跟随评测基准，旨在评估大语言模型在多语言、多模态及长短上下文中的指令理解能力。现有评测集在语言、模态和上下文长度方面存在局限，MCIF填补了这一空白，支持英文、德文、意大利文和中文，包含语音、视觉和文本三种模态，适用于科学讲座场景。**

- **链接: [http://arxiv.org/pdf/2507.19634v1](http://arxiv.org/pdf/2507.19634v1)**

> **作者:** Sara Papi; Maike Züfle; Marco Gaido; Beatrice Savoldi; Danni Liu; Ioannis Douros; Luisa Bentivogli; Jan Niehues
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations -- hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities -- speech, vision, and text -- and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development.
>
---
#### [new 022] ACCESS-AV: Adaptive Communication-Computation Codesign for Sustainable Autonomous Vehicle Localization in Smart Factories
- **分类: eess.SY; cs.AR; cs.NI; cs.RO; cs.SY; eess.SP**

- **简介: 论文提出ACCESS-AV框架，用于智能工厂中自动驾驶配送车辆的节能定位。该任务属于车辆定位优化。为解决能耗高与基础设施成本问题，利用5G同步信号块实现自适应通信计算协同设计，动态平衡能效与精度。实验表明其节能达43.09%，定位精度高，降低成本，适用于可持续智能工厂。**

- **链接: [http://arxiv.org/pdf/2507.20399v1](http://arxiv.org/pdf/2507.20399v1)**

> **作者:** Rajat Bhattacharjya; Arnab Sarkar; Ish Kool; Sabur Baidya; Nikil Dutt
>
> **备注:** 28 pages, 9 figures
>
> **摘要:** Autonomous Delivery Vehicles (ADVs) are increasingly used for transporting goods in 5G network-enabled smart factories, with the compute-intensive localization module presenting a significant opportunity for optimization. We propose ACCESS-AV, an energy-efficient Vehicle-to-Infrastructure (V2I) localization framework that leverages existing 5G infrastructure in smart factory environments. By opportunistically accessing the periodically broadcast 5G Synchronization Signal Blocks (SSBs) for localization, ACCESS-AV obviates the need for dedicated Roadside Units (RSUs) or additional onboard sensors to achieve energy efficiency as well as cost reduction. We implement an Angle-of-Arrival (AoA)-based estimation method using the Multiple Signal Classification (MUSIC) algorithm, optimized for resource-constrained ADV platforms through an adaptive communication-computation strategy that dynamically balances energy consumption with localization accuracy based on environmental conditions such as Signal-to-Noise Ratio (SNR) and vehicle velocity. Experimental results demonstrate that ACCESS-AV achieves an average energy reduction of 43.09% compared to non-adaptive systems employing AoA algorithms such as vanilla MUSIC, ESPRIT, and Root-MUSIC. It maintains sub-30 cm localization accuracy while also delivering substantial reductions in infrastructure and operational costs, establishing its viability for sustainable smart factory environments.
>
---
#### [new 023] Controllable Video-to-Music Generation with Multiple Time-Varying Conditions
- **分类: cs.MM; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于视频到音乐生成任务，旨在解决现有方法控制性差、难以满足用户预期的问题。作者提出了一种基于多条件引导的视频到音乐生成框架，通过两阶段训练策略和多个模块设计，实现对音乐生成过程的精细控制与视听同步，提升了生成效果与用户需求的匹配度。**

- **链接: [http://arxiv.org/pdf/2507.20627v1](http://arxiv.org/pdf/2507.20627v1)**

> **作者:** Junxian Wu; Weitao You; Heda Zuo; Dengming Zhang; Pei Chen; Lingyun Sun
>
> **备注:** Accepted by the 33rd ACM International Conference on Multimedia (ACMMM 2025). The project page is available at https://kita-wjx.github.io/MCV2M/
>
> **摘要:** Music enhances video narratives and emotions, driving demand for automatic video-to-music (V2M) generation. However, existing V2M methods relying solely on visual features or supplementary textual inputs generate music in a black-box manner, often failing to meet user expectations. To address this challenge, we propose a novel multi-condition guided V2M generation framework that incorporates multiple time-varying conditions for enhanced control over music generation. Our method uses a two-stage training strategy that enables learning of V2M fundamentals and audiovisual temporal synchronization while meeting users' needs for multi-condition control. In the first stage, we introduce a fine-grained feature selection module and a progressive temporal alignment attention mechanism to ensure flexible feature alignment. For the second stage, we develop a dynamic conditional fusion module and a control-guided decoder module to integrate multiple conditions and accurately guide the music composition process. Extensive experiments demonstrate that our method outperforms existing V2M pipelines in both subjective and objective evaluations, significantly enhancing control and alignment with user expectations.
>
---
## 更新

#### [replaced 001] BENYO-S2ST-Corpus-1: A Bilingual English-to-Yoruba Direct Speech-to-Speech Translation Corpus
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09342v2](http://arxiv.org/pdf/2507.09342v2)**

> **作者:** Emmanuel Adetiba; Abdultaofeek Abayomi; Raymond J. Kala; Ayodele H. Ifijeh; Oluwatobi E. Dare; Olabode Idowu-Bismark; Gabriel O. Sobola; Joy N. Adetiba; Monsurat Adepeju Lateef; Heather Cole-Lewis
>
> **摘要:** There is a major shortage of Speech-to-Speech Translation (S2ST) datasets for high resource-to-low resource language pairs such as English-to-Yoruba. Thus, in this study, we curated the Bilingual English-to-Yoruba Speech-to-Speech Translation Corpus Version 1 (BENYO-S2ST-Corpus-1). The corpus is based on a hybrid architecture we developed for large-scale direct S2ST corpus creation at reduced cost. To achieve this, we leveraged non speech-to-speech Standard Yoruba (SY) real-time audios and transcripts in the YORULECT Corpus as well as the corresponding Standard English (SE) transcripts. YORULECT Corpus is small scale(1,504) samples, and it does not have paired English audios. Therefore, we generated the SE audios using pre-trained AI models (i.e. Facebook MMS). We also developed an audio augmentation algorithm named AcoustAug based on three latent acoustic features to generate augmented audios from the raw audios of the two languages. BENYO-S2ST-Corpus-1 has 12,032 audio samples per language, which gives a total of 24,064 sample size. The total audio duration for the two languages is 41.20 hours. This size is quite significant. Beyond building S2ST models, BENYO-S2ST-Corpus-1 can be used to build pretrained models or improve existing ones. The created corpus and Coqui framework were used to build a pretrained Yoruba TTS model (named YoruTTS-0.5) as a proof of concept. The YoruTTS-0.5 gave a F0 RMSE value of 63.54 after 1,000 epochs, which indicates moderate fundamental pitch similarity with the reference real-time audio. Ultimately, the corpus architecture in this study can be leveraged by researchers and developers to curate datasets for multilingual high-resource-to-low-resource African languages. This will bridge the huge digital divides in translations among high and low-resource language pairs. BENYO-S2ST-Corpus-1 and YoruTTS-0.5 are publicly available at (https://bit.ly/40bGMwi).
>
---
#### [replaced 002] Scaling Analysis of Interleaved Speech-Text Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.02398v2](http://arxiv.org/pdf/2504.02398v2)**

> **作者:** Gallil Maimon; Michael Hassid; Amit Roth; Yossi Adi
>
> **备注:** Accepted at COLM 2025
>
> **摘要:** Existing Speech Language Model (SLM) scaling analysis paints a bleak picture. It predicts that SLMs require much more compute and data compared to text, leading some to question the feasibility of training high-quality SLMs. However, modern SLMs are often initialised from pre-trained TextLMs using speech-text interleaving to allow knowledge transfer. This raises the question - "Do interleaved SLMs scale more efficiently than textless-SLMs?" In this paper we answer a resounding yes! We conduct scaling analysis of interleaved SLMs by training several dozen and analysing the scaling trends. We see that under this setup SLMs scale more efficiently with compute. Additionally, our results indicate that the scaling dynamics significantly differ from textless-SLMs, suggesting one should allocate notably more of the compute budget to increasing model size over training tokens. We also study the role of synthetic data and TextLM model families in unlocking this potential. Results suggest that our scaled up model achieves comparable semantic speech performance to leading models, while using less compute and data. We open source models, samples, and data - https://pages.cs.huji.ac.il/adiyoss-lab/sims/ .
>
---
#### [replaced 003] ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21613v2](http://arxiv.org/pdf/2506.21613v2)**

> **作者:** Gautam Siddharth Kashyap; Mohammad Anas Azeez; Rafiq Ali; Zohaib Hasan Siddiqui; Jiechao Gao; Usman Naseem
>
> **备注:** Updated Version
>
> **摘要:** Hate speech targeting children on social media is a serious and growing problem, yet current NLP systems struggle to detect it effectively. This gap exists mainly because existing datasets focus on adults, lack age specific labels, miss nuanced linguistic cues, and are often too small for robust modeling. To address this, we introduce ChildGuard, the first large scale English dataset dedicated to hate speech aimed at children. It contains 351,877 annotated examples from X (formerly Twitter), Reddit, and YouTube, labeled by three age groups: younger children (under 11), pre teens (11--12), and teens (13--17). The dataset is split into two subsets for fine grained analysis: a contextual subset (157K) focusing on discourse level features, and a lexical subset (194K) emphasizing word-level sentiment and vocabulary. Benchmarking state of the art hate speech models on ChildGuard reveals notable drops in performance, highlighting the challenges of detecting child directed hate speech.
>
---
#### [replaced 004] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.05167v2](http://arxiv.org/pdf/2412.05167v2)**

> **作者:** Kuofeng Gao; Shu-Tao Xia; Ke Xu; Philip Torr; Jindong Gu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Audio-Language Models (LALMs), such as GPT-4o, have recently unlocked audio dialogue capabilities, enabling direct spoken exchanges with humans. The potential of LALMs broadens their applicability across a wide range of practical scenarios supported by audio dialogues. However, given these advancements, a comprehensive benchmark to evaluate the performance of LALMs in the open-ended audio dialogue understanding remains absent currently. To address this gap, we propose an Audio Dialogue Understanding Benchmark (ADU-Bench), which consists of 4 benchmark datasets. They assess the open-ended audio dialogue ability for LALMs in 3 general scenarios, 12 skills, 9 multilingual languages, and 4 categories of ambiguity handling. Notably, we firstly propose the evaluation of ambiguity handling in audio dialogues that expresses different intentions beyond the same literal meaning of sentences, e.g., "Really!?" with different intonations. In summary, ADU-Bench includes over 20,000 open-ended audio dialogues for the assessment of LALMs. Through extensive experiments on 16 LALMs, our analysis reveals that existing LALMs struggle with mathematical symbols and formulas, understanding human behavior such as roleplay, comprehending multiple languages, and handling audio dialogue ambiguities from different phonetic elements, such as intonations, pause positions, and homophones. The benchmark is available at https://adu-bench.github.io/.
>
---
#### [replaced 005] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14351v2](http://arxiv.org/pdf/2505.14351v2)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** 15 pages
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-\"U-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [replaced 006] SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.18181v2](http://arxiv.org/pdf/2507.18181v2)**

> **作者:** Linye Wei; Shuzhang Zhong; Songqiang Xu; Runsheng Wang; Ru Huang; Meng Li
>
> **备注:** Accepted by Design Automation Conference (DAC) 2025
>
> **摘要:** Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
>
---
#### [replaced 007] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17527v3](http://arxiv.org/pdf/2507.17527v3)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Yujiao Du; Ting Han; Yuxiang Hu; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Jun Zhang; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [replaced 008] RADE: A Neural Codec for Transmitting Speech over HF Radio Channels
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.06671v2](http://arxiv.org/pdf/2505.06671v2)**

> **作者:** David Rowe; Jean-Marc Valin
>
> **备注:** Proc. WASPAA 2025, 5 pages
>
> **摘要:** Speech compression is commonly used to send voice over radio channels in applications such as mobile telephony and two-way push-to-talk (PTT) radio. In classical systems, the speech codec is combined with forward error correction, modulation and radio hardware. In this paper we describe an autoencoder that replaces many of the traditional signal processing elements with a neural network. The encoder takes a vocoder feature set (short term spectrum, pitch, voicing), and produces discrete time, but continuously valued quadrature amplitude modulation (QAM) symbols. We use orthogonal frequency domain multiplexing (OFDM) to send and receive these symbols over high frequency (HF) radio channels. The decoder converts received QAM symbols to vocoder features suitable for synthesis. The autoencoder has been trained to be robust to additive Gaussian noise and multipath channel impairments while simultaneously maintaining a Peak To Average Power Ratio (PAPR) of less than 1 dB. Over simulated and real world HF radio channels we have achieved output speech intelligibility that clearly surpasses existing analog and digital radio systems over a range of SNRs.
>
---
#### [replaced 009] Should Top-Down Clustering Affect Boundaries in Unsupervised Word Discovery?
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19204v2](http://arxiv.org/pdf/2507.19204v2)**

> **作者:** Simon Malan; Benjamin van Niekerk; Herman Kamper
>
> **备注:** Submitted to the IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** We investigate the problem of segmenting unlabeled speech into word-like units and clustering these to create a lexicon. Prior work can be categorized into two frameworks. Bottom-up methods first determine boundaries and then cluster the fixed segmented words into a lexicon. In contrast, top-down methods incorporate information from the clustered words to inform boundary selection. However, it is unclear whether top-down information is necessary to improve segmentation. To explore this, we look at two similar approaches that differ in whether top-down clustering informs boundary selection. Our simple bottom-up strategy predicts word boundaries using the dissimilarity between adjacent self-supervised features, then clusters the resulting segments to construct a lexicon. Our top-down system is an updated version of the ES-KMeans dynamic programming method that iteratively uses K-means to update its boundaries. On the five-language ZeroSpeech benchmarks, both approaches achieve comparable state-of-the-art results, with the bottom-up system being nearly five times faster. Through detailed analyses, we show that the top-down influence of ES-KMeans can be beneficial (depending on factors like the candidate boundaries), but in many cases the simple bottom-up method performs just as well. For both methods, we show that the clustering step is a limiting factor. Therefore, we recommend that future work focus on improved clustering techniques and learning more discriminative word-like representations. Project code repository: https://github.com/s-malan/prom-seg-clus.
>
---
#### [replaced 010] TAIL: Text-Audio Incremental Learning
- **分类: cs.SD; cs.AI; cs.CV; eess.AS; I.2**

- **链接: [http://arxiv.org/pdf/2503.04258v2](http://arxiv.org/pdf/2503.04258v2)**

> **作者:** Yingfei Sun; Xu Gu; Wei Ji; Hanbin Zhao; Yifang Yin; Roger Zimmermann
>
> **备注:** 6 figures, 4 tables
>
> **摘要:** Many studies combine text and audio to capture multi-modal information but they overlook the model's generalization ability on new datasets. Introducing new datasets may affect the feature space of the original dataset, leading to catastrophic forgetting. Meanwhile, large model parameters can significantly impact training performance. To address these limitations, we introduce a novel task called Text-Audio Incremental Learning (TAIL) task for text-audio retrieval, and propose a new method, PTAT, Prompt Tuning for Audio-Text incremental learning. This method utilizes prompt tuning to optimize the model parameters while incorporating an audio-text similarity and feature distillation module to effectively mitigate catastrophic forgetting. We benchmark our method and previous incremental learning methods on AudioCaps, Clotho, BBC Sound Effects and Audioset datasets, and our method outperforms previous methods significantly, particularly demonstrating stronger resistance to forgetting on older datasets. Compared to the full-parameters Finetune (Sequential) method, our model only requires 2.42\% of its parameters, achieving 4.46\% higher performance.
>
---
#### [replaced 011] Computer Audition: From Task-Specific Machine Learning to Foundation Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.15672v2](http://arxiv.org/pdf/2407.15672v2)**

> **作者:** Andreas Triantafyllopoulos; Iosif Tsangko; Alexander Gebhard; Annamaria Mesaros; Tuomas Virtanen; Björn Schuller
>
> **备注:** Accepted for publication to the Proceedings of the IEEE
>
> **摘要:** Foundation models (FMs) are increasingly spearheading recent advances on a variety of tasks that fall under the purview of computer audition -- the use of machines to understand sounds. They feature several advantages over traditional pipelines: among others, the ability to consolidate multiple tasks in a single model, the option to leverage knowledge from other modalities, and the readily-available interaction with human users. Naturally, these promises have created substantial excitement in the audio community, and have led to a wave of early attempts to build new, general-purpose foundation models for audio. In the present contribution, we give an overview of computational audio analysis as it transitions from traditional pipelines towards auditory foundation models. Our work highlights the key operating principles that underpin those models, and showcases how they can accommodate multiple tasks that the audio community previously tackled separately.
>
---
#### [replaced 012] The Impact of LoRA Adapters on LLMs for Clinical Text Classification Under Computational and Data Constraints
- **分类: cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.19299v3](http://arxiv.org/pdf/2407.19299v3)**

> **作者:** Thanh-Dung Le; Ti Ti Nguyen; Vu Nguyen Ha; Symeon Chatzinotas; Philippe Jouvet; Rita Noumeir
>
> **备注:** Accepted for publication in the IEEE Access
>
> **摘要:** Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to domain gap, limited data, and stringent hardware constraints. In this study, we evaluate four adapter techniques-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN) - equivalent to Low-Rank Adaptation (LoRA), for clinical note classification under real-world, resource-constrained conditions. All experiments were conducted on a single NVIDIA Quadro P620 GPU (2 GB VRAM, 512 CUDA cores, 1.386 TFLOPS FP32), limiting batch sizes to <8 sequences and maximum sequence length to 256 tokens. Our clinical corpus comprises only 580 000 tokens, several orders of magnitude smaller than standard LLM pre-training datasets. We fine-tuned three biomedical pre-trained LLMs (CamemBERT-bio, AliBERT, DrBERT) and two lightweight Transformer models trained from scratch. Results show that 1) adapter structures provide no consistent gains when fine-tuning biomedical LLMs under these constraints, and 2) simpler Transformers, with minimal parameter counts and training times under six hours, outperform adapter-augmented LLMs, which required over 1000 GPU-hours. Among adapters, GRN achieved the best metrics (accuracy, precision, recall, F1 = 0.88). These findings demonstrate that, in low-resource clinical settings with limited data and compute, lightweight Transformers trained from scratch offer a more practical and efficient solution than large LLMs, while GRN remains a viable adapter choice when minimal adaptation is needed.
>
---
#### [replaced 013] Compressed Image Generation with Denoising Diffusion Codebook Models
- **分类: eess.IV; cs.AI; cs.CV; cs.IT; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.01189v4](http://arxiv.org/pdf/2502.01189v4)**

> **作者:** Guy Ohayon; Hila Manor; Tomer Michaeli; Michael Elad
>
> **备注:** Published in the International Conference on Machine Learning (ICML) 2025. Code and demo are available at https://ddcm-2025.github.io/
>
> **摘要:** We present a novel generative approach based on Denoising Diffusion Models (DDMs), which produces high-quality image samples along with their losslessly compressed bit-stream representations. This is obtained by replacing the standard Gaussian noise sampling in the reverse diffusion with a selection of noise samples from pre-defined codebooks of fixed iid Gaussian vectors. Surprisingly, we find that our method, termed Denoising Diffusion Codebook Model (DDCM), retains sample quality and diversity of standard DDMs, even for extremely small codebooks. We leverage DDCM and pick the noises from the codebooks that best match a given image, converting our generative model into a highly effective lossy image codec achieving state-of-the-art perceptual image compression results. More generally, by setting other noise selections rules, we extend our compression method to any conditional image generation task (e.g., image restoration), where the generated images are produced jointly with their condensed bit-stream representations. Our work is accompanied by a mathematical interpretation of the proposed compressed conditional generation schemes, establishing a connection with score-based approximations of posterior samplers for the tasks considered.
>
---
#### [replaced 014] Neural Spectral Band Generation for Audio Coding
- **分类: eess.AS; cs.AI; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.06732v2](http://arxiv.org/pdf/2506.06732v2)**

> **作者:** Woongjib Choi; Byeong Hyeon Kim; Hyungseob Lim; Inseon Jang; Hong-Goo Kang
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spectral band replication (SBR) enables bit-efficient coding by generating high-frequency bands from the low-frequency ones. However, it only utilizes coarse spectral features upon a subband-wise signal replication, limiting adaptability to diverse acoustic signals. In this paper, we explore the efficacy of a deep neural network (DNN)-based generative approach for coding the high-frequency bands, which we call neural spectral band generation (n-SBG). Specifically, we propose a DNN-based encoder-decoder structure to extract and quantize the side information related to the high-frequency components and generate the components given both the side information and the decoded core-band signals. The whole coding pipeline is optimized with generative adversarial criteria to enable the generation of perceptually plausible sound. From experiments using AAC as the core codec, we show that the proposed method achieves a better perceptual quality than HE-AAC-v1 with much less side information.
>
---
