# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Enhancing Automatic Speech Recognition Through Integrated Noise Detection Architecture
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决噪声环境下的识别准确率问题。通过在wav2vec2框架中集成噪声检测模块，实现语音转录与噪声识别同步优化，提升复杂声学条件下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.08973v1](https://arxiv.org/pdf/2512.08973v1)**

> **作者:** Karamvir Singh
>
> **备注:** 5 figures
>
> **摘要:** This research presents a novel approach to enhancing automatic speech recognition systems by integrating noise detection capabilities directly into the recognition architecture. Building upon the wav2vec2 framework, the proposed method incorporates a dedicated noise identification module that operates concurrently with speech transcription. Experimental validation using publicly available speech and environmental audio datasets demonstrates substantial improvements in transcription quality and noise discrimination. The enhanced system achieves superior performance in word error rate, character error rate, and noise detection accuracy compared to conventional architectures. Results indicate that joint optimization of transcription and noise classification objectives yields more reliable speech recognition in challenging acoustic conditions.
>
---
#### [new 002] Robust Speech Activity Detection in the Presence of Singing Voice
- **分类: eess.AS**

- **简介: 该论文研究语音活动检测（SAD）任务，旨在解决歌唱干扰下语音误检问题。提出SR-SAD模型，通过可控训练策略、高效结构设计和新评估指标，提升语音与歌唱的区分能力，在多音乐场景中实现高准确率语音检测。**

- **链接: [https://arxiv.org/pdf/2512.09713v1](https://arxiv.org/pdf/2512.09713v1)**

> **作者:** Philipp Grundhuber; Mhd Modar Halimeh; Martin Strauß; Emanuël A. P. Habets
>
> **备注:** This paper has been published in: 2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)
>
> **摘要:** Speech Activity Detection (SAD) systems often misclassify singing as speech, leading to degraded performance in applications such as dialogue enhancement and automatic speech recognition. We introduce Singing-Robust Speech Activity Detection ( SR-SAD ), a neural network designed to robustly detect speech in the presence of singing. Our key contributions are: i) a training strategy using controlled ratios of speech and singing samples to improve discrimination, ii) a computationally efficient model that maintains robust performance while reducing inference runtime, and iii) a new evaluation metric tailored to assess SAD robustness in mixed speech-singing scenarios. Experiments on a challenging dataset spanning multiple musical genres show that SR-SAD maintains high speech detection accuracy (AUC = 0.919) while rejecting singing. By explicitly learning to distinguish between speech and singing, SR-SAD enables more reliable SAD in mixed speech-singing scenarios.
>
---
#### [new 003] ORCA: Open-ended Response Correctness Assessment for Audio Question Answering
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对音频问答中开放性回答评估难的问题，提出ORCA框架，利用Beta分布建模人类评判的不确定性，结合三阶段标注流程提升评估质量，实现高相关性的自动评估，并发布数据与模型。**

- **链接: [https://arxiv.org/pdf/2512.09066v1](https://arxiv.org/pdf/2512.09066v1)**

> **作者:** Šimon Sedláček; Sara Barahona; Bolaji Yusuf; Laura Herrera-Alarcón; Santosh Kesiraju; Cecilia Bolaños; Alicia Lozano-Diez; Sathvik Udupa; Fernando López; Allison Ferner; Ramani Duraiswami; Jan Černocký
>
> **摘要:** Evaluating open-ended responses from large audio language models (LALMs) is challenging because human annotators often genuinely disagree on answer correctness due to multiple valid interpretations, partial correctness, and subjective judgment. Traditional metrics reporting only mean scores fail to capture this uncertainty. We present ORCA (Open-ended Response Correctness Assessment), a framework that models the variability in human judgments using Beta distributions to predict both expected correctness and uncertainty. Our three-stage annotation framework combines human judgment with structured feedback and iterative refinement to simultaneously curate training data and improve benchmark quality. We collected 11,721 annotations across 3,580 question-answer pairs from 15 LALMs on two audio QA benchmarks, achieving inter-annotator agreement of 0.82 (Krippendorff's alpha). ORCA achieves 0.91 Spearman correlation with mean human judgments, matching or outperforming LLM-judge baselines while providing uncertainty estimates and requiring significantly less compute. We release our models, code, and curated dataset.
>
---
#### [new 004] Human perception of audio deepfakes: the role of language and speaking style
- **分类: eess.AS; eess.SP**

- **简介: 该论文研究人类对音频深度伪造的感知，探讨语言和说话风格的影响。通过跨语言听觉实验，分析听众判断语音真假的准确率及依据，揭示其依赖韵律和高层语音特征的感知策略。**

- **链接: [https://arxiv.org/pdf/2512.09221v1](https://arxiv.org/pdf/2512.09221v1)**

> **作者:** Eugenia San Segundo; Aurora López-Jareño; Xin Wang; Junichi Yamagishi
>
> **备注:** Submitted to Speech Communication
>
> **摘要:** Audio deepfakes have reached a level of realism that makes it increasingly difficult to distinguish between human and artificial voices, which poses risks such as identity theft or spread of disinformation. Despite these concerns, research on humans' ability to identify deepfakes is limited, with most studies focusing on English and very few exploring the reasons behind listeners' perceptual decisions. This study addresses this gap through a perceptual experiment in which 54 listeners (28 native Spanish speakers and 26 native Japanese speakers) classified voices as natural or synthetic, and justified their choices. The experiment included 80 stimuli (50% artificial), organized according to three variables: language (Spanish/Japanese), speech style (audiobooks/interviews), and familiarity with the voice (familiar/unfamiliar). The goal was to examine how these variables influence detection and to analyze qualitatively the reasoning behind listeners' perceptual decisions. Results indicate an average accuracy of 59.11%, with higher performance on authentic samples. Judgments of vocal naturalness rely on a combination of linguistic and non-linguistic cues. Comparing Japanese and Spanish listeners, our qualitative analysis further reveals both shared cues and notable cross-linguistic differences in how listeners conceptualize the "humanness" of speech. Overall, participants relied primarily on suprasegmental and higher-level or extralinguistic characteristics - such as intonation, rhythm, fluency, pauses, speed, breathing, and laughter - over segmental features. These findings underscore the complexity of human perceptual strategies in distinguishing natural from artificial speech and align partly with prior research emphasizing the importance of prosody and phenomena typical of spontaneous speech, such as disfluencies.
>
---
#### [new 005] Who Speaks What from Afar: Eavesdropping In-Person Conversations via mmWave Sensing
- **分类: cs.SD**

- **简介: 该论文研究通过毫米波感知远程窃听多人会议中“谁说了什么”。针对现有技术无法区分说话者的问题，提出一种无监督方法，利用物体振动差异识别说话人，并结合深度学习提升语音质量，实验证明其高准确率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.09285v1](https://arxiv.org/pdf/2512.09285v1)**

> **作者:** Shaoying Wang; Hansong Zhou; Yukun Yuan; Xiaonan Zhang
>
> **摘要:** Multi-participant meetings occur across various domains, such as business negotiations and medical consultations, during which sensitive information like trade secrets, business strategies, and patient conditions is often discussed. Previous research has demonstrated that attackers with mmWave radars outside the room can overhear meeting content by detecting minute speech-induced vibrations on objects. However, these eavesdropping attacks cannot differentiate which speech content comes from which person in a multi-participant meeting, leading to potential misunderstandings and poor decision-making. In this paper, we answer the question ``who speaks what''. By leveraging the spatial diversity introduced by ubiquitous objects, we propose an attack system that enables attackers to remotely eavesdrop on in-person conversations without requiring prior knowledge, such as identities, the number of participants, or seating arrangements. Since participants in in-person meetings are typically seated at different locations, their speech induces distinct vibration patterns on nearby objects. To exploit this, we design a noise-robust unsupervised approach for distinguishing participants by detecting speech-induced vibration differences in the frequency domain. Meanwhile, a deep learning-based framework is explored to combine signals from objects for speech quality enhancement. We validate the proof-of-concept attack on speech classification and signal enhancement through extensive experiments. The experimental results show that our attack can achieve the speech classification accuracy of up to $0.99$ with several participants in a meeting room. Meanwhile, our attack demonstrates consistent speech quality enhancement across all real-world scenarios, including different distances between the radar and the objects.
>
---
#### [new 006] DMP-TTS: Disentangled multi-modal Prompting for Controllable Text-to-Speech with Chained Guidance
- **分类: cs.SD**

- **简介: 该论文研究可控文本到语音（TTS）任务，旨在解耦说话人音色与说话风格。提出DMP-TTS框架，通过多模态提示、风格编码器和链式无分类器引导实现独立控制，并引入特征对齐加速训练，提升风格可控性与合成质量。**

- **链接: [https://arxiv.org/pdf/2512.09504v1](https://arxiv.org/pdf/2512.09504v1)**

> **作者:** Kang Yin; Chunyu Qiang; Sirui Zhao; Xiaopeng Wang; Yuzhe Liang; Pengfei Cai; Tong Xu; Chen Zhang; Enhong Chen
>
> **摘要:** Controllable text-to-speech (TTS) systems face significant challenges in achieving independent manipulation of speaker timbre and speaking style, often suffering from entanglement between these attributes. We present DMP-TTS, a latent Diffusion Transformer (DiT) framework with explicit disentanglement and multi-modal prompting. A CLAP-based style encoder (Style-CLAP) aligns cues from reference audio and descriptive text in a shared space and is trained with contrastive learning plus multi-task supervision on style attributes. For fine-grained control during inference, we introduce chained classifier-free guidance (cCFG) trained with hierarchical condition dropout, enabling independent adjustment of content, timbre, and style guidance strengths. Additionally, we employ Representation Alignment (REPA) to distill acoustic-semantic features from a pretrained Whisper model into intermediate DiT representations, stabilizing training and accelerating convergence. Experiments show that DMP-TTS delivers stronger style controllability than open-source baselines while maintaining competitive intelligibility and naturalness. Code and demos will be available at https://y61329697.github.io/DMP-TTS/.
>
---
#### [new 007] LG Uplus System with Multi-Speaker IDs and Discriminator-based Sub-Judges for the WildSpoof Challenge
- **分类: eess.AS**

- **简介: 该论文针对高质文本转语音攻击下的说话人验证任务，提出多说话人ID标签策略与基于判别器的子判决系统，利用ResNet-221和注意力池化融合特征，提升抗未知攻击的检测性能。**

- **链接: [https://arxiv.org/pdf/2512.09000v1](https://arxiv.org/pdf/2512.09000v1)**

> **作者:** Jinyoung Park; Won Jang; Jiwoong Park
>
> **备注:** 3 pages, 2 figures, 2 tables
>
> **摘要:** This paper describes our submission to the WildSpoof Challenge Track 2, which focuses on spoof-aware speaker verification (SASV) in the presence of high-quality text-to-speech (TTS) attacks. We adopt a ResNet-221 back-bone and study two speaker-labeling strategies, namelyDual-Speaker IDs and Multi-Speaker IDs, to explicitly enlarge the margin between bona fide and generated speech in the embedding space. In addition, we propose discriminator-based sub-judge systems that reuse internal features from HiFi-GAN and BigVGAN discriminators, aggregated via multi-query multi-head attentive statistics pooling(MQMHA). Experimental results on the SpoofCeleb corpus show that our system design is effective in improving agnostic detection cost function (a-DCF).
>
---
#### [new 008] TinyDéjàVu: Smaller Memory Footprint & Faster Inference on Sensor Data Streams with Always-On Microcontrollers
- **分类: cs.LG; cs.PF; cs.SD; eess.AS; eess.SP**

- **简介: 该论文针对微控制器上传感器时序数据的持续推理任务，解决内存小、能耗低下的内存占用与计算冗余问题。提出TinyDéjàVu框架，优化层间数据流，显著减少RAM使用和重复计算。**

- **链接: [https://arxiv.org/pdf/2512.09786v1](https://arxiv.org/pdf/2512.09786v1)**

> **作者:** Zhaolan Huang; Emmanuel Baccelli
>
> **摘要:** Always-on sensors are increasingly expected to embark a variety of tiny neural networks and to continuously perform inference on time-series of the data they sense. In order to fit lifetime and energy consumption requirements when operating on battery, such hardware uses microcontrollers (MCUs) with tiny memory budget e.g., 128kB of RAM. In this context, optimizing data flows across neural network layers becomes crucial. In this paper, we introduce TinyDéjàVu, a new framework and novel algorithms we designed to drastically reduce the RAM footprint required by inference using various tiny ML models for sensor data time-series on typical microcontroller hardware. We publish the implementation of TinyDéjàVu as open source, and we perform reproducible benchmarks on hardware. We show that TinyDéjàVu can save more than 60% of RAM usage and eliminate up to 90% of redundant compute on overlapping sliding window inputs.
>
---
#### [new 009] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
- **分类: cs.CV; cs.SD**

- **简介: 该论文研究音频驱动的对话虚拟人生成，旨在解决现有方法难以自然建模听者表情的问题。提出UniLS框架，首次实现仅基于双通道音频的端到端说-听联合生成，通过两阶段训练学习听者内部运动先验并结合音频调节，显著提升听态自然度与多样性。**

- **链接: [https://arxiv.org/pdf/2512.09327v1](https://arxiv.org/pdf/2512.09327v1)**

> **作者:** Xuangeng Chu; Ruicong Liu; Yifei Huang; Yun Liu; Yichen Peng; Bo Zheng
>
> **摘要:** Generating lifelike conversational avatars requires modeling not just isolated speakers, but the dynamic, reciprocal interaction of speaking and listening. However, modeling the listener is exceptionally challenging: direct audio-driven training fails, producing stiff, static listening motions. This failure stems from a fundamental imbalance: the speaker's motion is strongly driven by speech audio, while the listener's motion primarily follows an internal motion prior and is only loosely guided by external speech. This challenge has led most methods to focus on speak-only generation. The only prior attempt at joint generation relies on extra speaker's motion to produce the listener. This design is not end-to-end, thereby hindering the real-time applicability. To address this limitation, we present UniLS, the first end-to-end framework for generating unified speak-listen expressions, driven by only dual-track audio. Our method introduces a novel two-stage training paradigm. Stage 1 first learns the internal motion prior by training an audio-free autoregressive generator, capturing the spontaneous dynamics of natural facial motion. Stage 2 then introduces the dual-track audio, fine-tuning the generator to modulate the learned motion prior based on external speech cues. Extensive evaluations show UniLS achieves state-of-the-art speaking accuracy. More importantly, it delivers up to 44.1\% improvement in listening metrics, generating significantly more diverse and natural listening expressions. This effectively mitigates the stiffness problem and provides a practical, high-fidelity audio-driven solution for interactive digital humans.
>
---
#### [new 010] VABench: A Comprehensive Benchmark for Audio-Video Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文聚焦音频-视频同步生成任务，旨在解决现有基准缺乏对音视频协同生成评估的问题。作者提出VABench，包含三类生成任务、七个内容类别及15个评估维度，全面评测音视频生成质量与同步性。**

- **链接: [https://arxiv.org/pdf/2512.09299v1](https://arxiv.org/pdf/2512.09299v1)**

> **作者:** Daili Hua; Xizhi Wang; Bohan Zeng; Xinyi Huang; Hao Liang; Junbo Niu; Xinlong Chen; Quanqing Xu; Wentao Zhang
>
> **备注:** 24 pages, 25 figures
>
> **摘要:** Recent advances in video generation have been remarkable, enabling models to produce visually compelling videos with synchronized audio. While existing video generation benchmarks provide comprehensive metrics for visual quality, they lack convincing evaluations for audio-video generation, especially for models aiming to generate synchronized audio-video outputs. To address this gap, we introduce VABench, a comprehensive and multi-dimensional benchmark framework designed to systematically evaluate the capabilities of synchronous audio-video generation. VABench encompasses three primary task types: text-to-audio-video (T2AV), image-to-audio-video (I2AV), and stereo audio-video generation. It further establishes two major evaluation modules covering 15 dimensions. These dimensions specifically assess pairwise similarities (text-video, text-audio, video-audio), audio-video synchronization, lip-speech consistency, and carefully curated audio and video question-answering (QA) pairs, among others. Furthermore, VABench covers seven major content categories: animals, human sounds, music, environmental sounds, synchronous physical sounds, complex scenes, and virtual worlds. We provide a systematic analysis and visualization of the evaluation results, aiming to establish a new standard for assessing video generation models with synchronous audio capabilities and to promote the comprehensive advancement of the field.
>
---
## 更新

#### [replaced 001] SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音大语言模型中的检索增强生成任务，旨在解决传统两阶段方法延迟高、错误传播的问题。提出SEAL框架，通过统一语音与文本的嵌入空间，实现端到端语音检索，降低50%延迟并提升准确率。**

- **链接: [https://arxiv.org/pdf/2502.02603v2](https://arxiv.org/pdf/2502.02603v2)**

> **作者:** Chunyu Sun; Bingyu Liu; Zhichao Cui; Junhan Shi; Anbin Qi; Tian-hao Zhang; Dinghao Zhou; Lewei Lu
>
> **摘要:** Embedding-based retrieval models have made significant strides in retrieval-augmented generation (RAG) techniques for text and multimodal large language models (LLMs) applications. However, when it comes to speech larage language models (SLLMs), these methods are limited to a two-stage process, where automatic speech recognition (ASR) is combined with text-based retrieval. This sequential architecture suffers from high latency and error propagation. To address these limitations, we propose a unified embedding framework that eliminates the need for intermediate text representations. Specifically, the framework includes separate speech and text encoders, followed by a shared scaling layer that maps both modalities into a common embedding space. Our model reduces pipeline latency by 50\% while achieving higher retrieval accuracy compared to traditional two-stage methods. We also provide a theoretical analysis of the challenges inherent in end-to-end speech retrieval and introduce architectural principles for effective speech-to-document matching. Extensive experiments demonstrate the robustness of our approach across diverse acoustic conditions and speaker variations, paving the way for a new paradigm in multimodal SLLMs retrieval systems.
>
---
#### [replaced 002] A Low-Complexity Speech Codec Using Parametric Dithering for ASR
- **分类: eess.AS**

- **简介: 该论文研究语音压缩对ASR的影响，提出一种基于参数化抖动（dithering）的低复杂度语音编解码方法。通过优化抖动策略，在极低比特率下显著降低词错误率，兼顾性能与数据率，适应不同熵约束。**

- **链接: [https://arxiv.org/pdf/2512.00511v2](https://arxiv.org/pdf/2512.00511v2)**

> **作者:** Ellison Murray; Morriel Kasher; Predrag Spasojevic
>
> **备注:** 10 pages, 8 figures, Accepted 2026 Data Compression Conference
>
> **摘要:** Dithering is a technique commonly used to improve the perceptual quality of lossy data compression. In this work, we analytically and experimentally justify the use of dithering for ASR input compression. We formalize an understanding of optimal ASR performance under lossy input compression and leverage this to propose a parametric dithering technique for a low-complexity speech compression pipeline. The method performs well at 1-bit resolution, showing a 25\% relative CER improvement, while also demonstrating improvements of 32.4\% and 33.5\% at 2- and 3-bit resolution, respectively, with our second dither choice yielding a reduced data rate. The proposed codec is adaptable to meet performance targets or stay within entropy constraints.
>
---
#### [replaced 003] MambAttention: Mamba with Multi-Head Attention for Generalizable Single-Channel Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究单通道语音增强任务，旨在提升模型跨数据集的泛化能力。针对Mamba等序列模型易过拟合的问题，提出MambAttention架构，融合Mamba与共享的时频多头注意力机制，并构建新训练数据集VB-DemandEx，显著提升了在多个下游数据集上的语音增强性能。**

- **链接: [https://arxiv.org/pdf/2507.00966v3](https://arxiv.org/pdf/2507.00966v3)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing for possible publication
>
> **摘要:** With new sequence models like Mamba and xLSTM, several studies have shown that these models match or outperform the state-of-the-art in single-channel speech enhancement and audio representation learning. However, prior research has demonstrated that sequence models like LSTM and Mamba tend to overfit to the training set. To address this, previous works have shown that adding self-attention to LSTMs substantially improves generalization performance for single-channel speech enhancement. Nevertheless, neither the concept of hybrid Mamba and time-frequency attention models nor their generalization performance have been explored for speech enhancement. In this paper, we propose a novel hybrid architecture, MambAttention, which combines Mamba and shared time- and frequency-multi-head attention modules for generalizable single-channel speech enhancement. To train our model, we introduce VB-DemandEx, a dataset inspired by VoiceBank+Demand but with more challenging noise types and lower signal-to-noise ratios. Trained on VB-DemandEx, MambAttention significantly outperforms existing state-of-the-art discriminative LSTM-, xLSTM-, Mamba-, and Conformer-based systems of similar complexity across all reported metrics on two out-of-domain datasets: DNS 2020 without reverberation and EARS-WHAM_v2. MambAttention also matches or outperforms generative diffusion models in generalization performance while being competitive with language model baselines. Ablation studies highlight the importance of weight sharing between time- and frequency-multi-head attention modules for generalization performance. Finally, we explore integrating the shared time- and frequency-multi-head attention modules with LSTM and xLSTM, which yields a notable performance improvement on the out-of-domain datasets. Yet, MambAttention remains superior for cross-corpus generalization across all reported evaluation metrics.
>
---
#### [replaced 004] MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文研究多声源音频到图像生成任务，旨在解决现有方法忽略自然场景中多声源特性的问题。作者提出MACS方法，首次显式分离多声源音频，并通过语义对齐与上下文显著性优化生成图像，构建了首个多声源音频-图像生成基准。**

- **链接: [https://arxiv.org/pdf/2503.10287v3](https://arxiv.org/pdf/2503.10287v3)**

> **作者:** Hao Zhou; Xiaobao Guo; Yuzhe Zhu; Adams Wai-Kin Kong
>
> **备注:** Accepted at AAAI 2026. Code available at https://github.com/alxzzhou/MACS
>
> **摘要:** Propelled by the breakthrough in deep generative models, audio-to-image generation has emerged as a pivotal cross-modal task that converts complex auditory signals into rich visual representations. However, previous works only focus on single-source audio inputs for image generation, ignoring the multi-source characteristic in natural auditory scenes, thus limiting the performance in generating comprehensive visual content. To bridge this gap, we propose a method called MACS to conduct multi-source audio-to-image generation. To our best knowledge, this is the first work that explicitly separates multi-source audio to capture the rich audio components before image generation. MACS is a two-stage method. In the first stage, multi-source audio inputs are separated by a weakly supervised method, where the audio and text labels are semantically aligned by casting into a common space using the large pre-trained CLAP model. We introduce a ranking loss to consider the contextual significance of the separated audio signals. In the second stage, effective image generation is achieved by mapping the separated audio signals to the generation condition using only a trainable adapter and a MLP layer. We preprocess the LLP dataset as the first full multi-source audio-to-image generation benchmark. The experiments are conducted on multi-source, mixed-source, and single-source audio-to-image generation tasks. The proposed MACS outperforms the current state-of-the-art methods in 17 out of the 21 evaluation indexes on all tasks and delivers superior visual quality.
>
---
#### [replaced 005] CardioLive: Empowering Video Streaming with Online Cardiac Monitoring
- **分类: cs.HC; cs.NI; cs.SD; eess.AS; eess.IV**

- **简介: 该论文提出CardioLive，首个面向视频流的在线心率监测系统。通过音视频联合建模的CardioNet，实现鲁棒的心率估计，并解决帧率变化、音视频不同步等实际问题，支持即插即用的按需服务，在Zoom和YouTube上高效运行。**

- **链接: [https://arxiv.org/pdf/2502.00702v2](https://arxiv.org/pdf/2502.00702v2)**

> **作者:** Sheng Lyu; Ruiming Huang; Sijie Ji; Yasar Abbas Ur Rehman; Lan Ma; Chenshu Wu
>
> **备注:** Preprint
>
> **摘要:** Online Cardiac Monitoring (OCM) emerges as a compelling enhancement for the next-generation video streaming platforms. It enables various applications including remote health, online affective computing, and deepfake detection. Yet the physiological information encapsulated in the video streams has been long neglected. In this paper, we present the design and implementation of CardioLive, the first online cardiac monitoring system in video streaming platforms. We leverage the naturally co-existed video and audio streams and devise CardioNet, the first audio-visual network to learn the cardiac series. It incorporates multiple unique designs to extract temporal and spectral features, ensuring robust performance under realistic video streaming conditions. To enable the Service-On-Demand online cardiac monitoring, we implement CardioLive as a plug-and-play middleware service and develop systematic solutions to practical issues including changing FPS and unsynchronized streams. Extensive experiments have been done to demonstrate the effectiveness of our system. We achieve a Mean Square Error (MAE) of 1.79 BPM error, outperforming the video-only and audio-only solutions by 69.2% and 81.2%, respectively. Our CardioLive service achieves average throughputs of 115.97 and 98.16 FPS when implemented in Zoom and YouTube. We believe our work opens up new applications for video stream systems. We will release the code soon.
>
---
#### [replaced 006] Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2，一个统一且可控的语音与歌声生成框架。针对标注数据稀缺和可控性难题，设计双音频 tokenizer 与分阶段建模，实现文本、韵律、风格与音色的解耦控制，通过联合训练和多目标后训练提升生成质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.16332v2](https://arxiv.org/pdf/2508.16332v2)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** We will release code and model checkpoints at https://github.com/open-mmlab/Amphion
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a unified music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a unified content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during the speech-singing joint training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the Vevo2's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at https://versasinger.github.io/.
>
---
#### [replaced 007] Point Neuron Learning: A New Physics-Informed Neural Network Architecture
- **分类: cs.LG; cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出一种新型物理信息神经网络架构——点神经元学习，用于声场重建任务。通过将波动方程基本解嵌入网络结构，实现无需训练数据、严格满足物理规律的建模，解决了传统方法对数据依赖强、可解释性差和泛化能力弱的问题。**

- **链接: [https://arxiv.org/pdf/2408.16969v2](https://arxiv.org/pdf/2408.16969v2)**

> **作者:** Hanwen Bi; Thushara D. Abhayapala
>
> **备注:** 15 pages, 9 figures. Published in EURASIP Journal on Audio Speech and Music Processing
>
> **摘要:** Machine learning and neural networks have advanced numerous research domains, but challenges such as large training data requirements and inconsistent model performance hinder their application in certain scientific problems. To overcome these challenges, researchers have investigated integrating physics principles into machine learning models, mainly through: (i) physics-guided loss functions, generally termed as physics-informed neural networks, and (ii) physics-guided architectural design. While both approaches have demonstrated success across multiple scientific disciplines, they have limitations including being trapped to a local minimum, poor interpretability, and restricted generalizability. This paper proposes a new physics-informed neural network (PINN) architecture that combines the strengths of both approaches by embedding the fundamental solution of the wave equation into the network architecture, enabling the learned model to strictly satisfy the wave equation. The proposed point neuron learning method can model an arbitrary sound field based on microphone observations without any dataset. Compared to other PINN methods, our approach directly processes complex numbers and offers better interpretability and generalizability. We evaluate the versatility of the proposed architecture by a sound field reconstruction problem in a reverberant environment. Results indicate that the point neuron method outperforms two competing methods and can efficiently handle noisy environments with sparse microphone observations.
>
---
#### [replaced 008] Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究视频到音频生成任务，发现现有模型存在插入幻觉问题，即生成无视觉来源的声音。作者提出新评估框架与指标，并设计无需训练的后处理方法PFC，有效减少幻觉发生率与持续时间，提升生成可靠性。**

- **链接: [https://arxiv.org/pdf/2510.08078v4](https://arxiv.org/pdf/2510.08078v4)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **备注:** The paper has been withdrawn because it will undergo a major revision. The revised version will differ substantially from the current one, making replacement inappropriate
>
> **摘要:** Video-to-Audio generation has made remarkable strides in automatically synthesizing sound for video. However, existing evaluation metrics, which focus on semantic and temporal alignment, overlook a critical failure mode: models often generate acoustic events, particularly speech and music, that have no corresponding visual source. We term this phenomenon Insertion Hallucination and identify it as a systemic risk driven by dataset biases, such as the prevalence of off-screen sounds, that remains completely undetected by current metrics. To address this challenge, we first develop a systematic evaluation framework that employs a majority-voting ensemble of multiple audio event detectors. We also introduce two novel metrics to quantify the prevalence and severity of this issue: IH@vid (the fraction of videos with hallucinations) and IH@dur (the fraction of hallucinated duration). Building on this, we propose Posterior Feature Correction, a novel training-free inference-time method that mitigates IH. PFC operates in a two-pass process: it first generates an initial audio output to detect hallucinated segments, and then regenerates the audio after masking the corresponding video features at those timestamps. Experiments on several mainstream V2A benchmarks first reveal that state-of-the-art models suffer from severe IH. In contrast, our PFC method reduces both the prevalence and duration of hallucinations by over 50\% on average, without degrading, and in some cases even improving, conventional metrics for audio quality and temporal synchronization. Our work is the first to formally define, systematically measure, and effectively mitigate Insertion Hallucination, paving the way for more reliable and faithful V2A models.
>
---
#### [replaced 009] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦ASR（自动语音识别）评估任务，旨在解决现有评测缺乏多语言支持、效率指标缺失及不可复现的问题。作者构建了Open ASR Leaderboard，开源了包含60多个系统的标准化评测基准，统一文本归一化，报告WER和RTFx，支持可复现、透明的多语言ASR性能比较。**

- **链接: [https://arxiv.org/pdf/2510.06961v3](https://arxiv.org/pdf/2510.06961v3)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Adel Moumen; Sanchit Gandhi
>
> **备注:** Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard ; Code: https://github.com/huggingface/open_asr_leaderboard
>
> **摘要:** Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including a dedicated multilingual track. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation.
>
---
