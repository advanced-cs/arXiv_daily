# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] On the Use of Self-Supervised Representation Learning for Speaker Diarization and Separation
- **分类: eess.AS**

- **简介: 该论文属于语音处理任务，聚焦说话人日志（diarization）与分离（separation）。针对现有自监督模型（如wav2vec2.0、WavLM）在这些任务上评估不足的问题，论文系统考察其表征质量，指出基准数据集多样性不足及下游系统覆盖不全等关键缺陷。**

- **链接: [https://arxiv.org/pdf/2512.15224v1](https://arxiv.org/pdf/2512.15224v1)**

> **作者:** Séverin Baroudi; Hervé Bredin; Joseph Razik; Ricard Marxer
>
> **备注:** accepted at ASRU25
>
> **摘要:** Self-supervised speech models such as wav2vec2.0 and WavLM have been shown to significantly improve the performance of many downstream speech tasks, especially in low-resource settings, over the past few years. Despite this, evaluations on tasks such as Speaker Diarization and Speech Separation remain limited. This paper investigates the quality of recent self-supervised speech representations on these two speaker identity-related tasks, highlighting gaps in the current literature that stem from limitations in the existing benchmarks, particularly the lack of diversity in evaluation datasets and variety in downstream systems associated to both diarization and separation.
>
---
#### [new 002] A Conditioned UNet for Music Source Separation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文面向音乐源分离（MSS）任务，解决传统UNet需预设固定乐器词表、泛化性差的问题。提出条件化UNet模型QSCNet，通过音频查询实现任意目标声部提取，在MoisesDB数据集上超越Banquet，SNR提升超1dB且参数减半。**

- **链接: [https://arxiv.org/pdf/2512.15532v1](https://arxiv.org/pdf/2512.15532v1)**

> **作者:** Ken O'Hanlon; Basil Woods; Lin Wang; Mark Sandler
>
> **摘要:** In this paper we propose a conditioned UNet for Music Source Separation (MSS). MSS is generally performed by multi-output neural networks, typically UNets, with each output representing a particular stem from a predefined instrument vocabulary. In contrast, conditioned MSS networks accept an audio query related to a stem of interest alongside the signal from which that stem is to be extracted. Thus, a strict vocabulary is not required and this enables more realistic tasks in MSS. The potential of conditioned approaches for such tasks has been somewhat hidden due to a lack of suitable data, an issue recently addressed with the MoisesDb dataset. A recent method, Banquet, employs this dataset with promising results seen on larger vocabularies. Banquet uses Bandsplit RNN rather than a UNet and the authors state that UNets should not be suitable for conditioned MSS. We counter this argument and propose QSCNet, a novel conditioned UNet for MSS that integrates network conditioning elements in the Sparse Compressed Network for MSS. We find QSCNet to outperform Banquet by over 1dB SNR on a couple of MSS tasks, while using less than half the number of parameters.
>
---
#### [new 003] Audio MultiChallenge: A Multi-Turn Evaluation of Spoken Dialogue Systems on Natural Human Interaction
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出Audio MultiChallenge基准，面向端到端语音对话系统，解决现有评测忽视自然多轮语音交互的问题。工作包括：扩展MultiChallenge框架至音频模态，新增Voice Editing和Audio-Cue挑战，构建452段真实多轮语音对话数据集，并评估模型在推理记忆、指令保持、自一致性及语音编辑鲁棒性上的表现。**

- **链接: [https://arxiv.org/pdf/2512.14865v1](https://arxiv.org/pdf/2512.14865v1)**

> **作者:** Advait Gosai; Tyler Vuong; Utkarsh Tyagi; Steven Li; Wenjia You; Miheer Bavare; Arda Uçar; Zhongwang Fang; Brian Jang; Bing Liu; Yunzhong He
>
> **摘要:** End-to-end (E2E) spoken dialogue systems are increasingly replacing cascaded pipelines for voice-based human-AI interaction, processing raw audio directly without intermediate transcription. Existing benchmarks primarily evaluate these models on synthetic speech and single-turn tasks, leaving realistic multi-turn conversational ability underexplored. We introduce Audio MultiChallenge, an open-source benchmark to evaluate E2E spoken dialogue systems under natural multi-turn interaction patterns. Building on the text-based MultiChallenge framework, which evaluates Inference Memory, Instruction Retention, and Self Coherence, we introduce a new axis Voice Editing that tests robustness to mid-utterance speech repairs and backtracking. We further augment each axis to the audio modality, such as introducing Audio-Cue challenges for Inference Memory that require recalling ambient sounds and paralinguistic signals beyond semantic content. We curate 452 conversations from 47 speakers with 1,712 instance-specific rubrics through a hybrid audio-native agentic and human-in-the-loop pipeline that exposes model failures at scale while preserving natural disfluencies found in unscripted human speech. Our evaluation of proprietary and open-source models reveals that even frontier models struggle on our benchmark, with Gemini 3 Pro Preview (Thinking), our highest-performing model achieving a 54.65% pass rate. Error analysis shows that models fail most often on our new axes and that Self Coherence degrades with longer audio context. These failures reflect difficulty of tracking edits, audio cues, and long-range context in natural spoken dialogue. Audio MultiChallenge provides a reproducible testbed to quantify them and drive improvements in audio-native multi-turn interaction capability.
>
---
#### [new 004] BEAT2AASIST model with layer fusion for ESDD 2026 Challenge
- **分类: cs.SD; cs.LG**

- **简介: 该论文面向环境声音深度伪造检测（ESDD）任务，旨在识别被篡改的环境音频。提出BEAT2AASIST模型：扩展BEATs-AASIST，采用双分支处理频/通道分割特征，并引入top-k层融合（拼接、CNN门控、SE门控）及声码器数据增强，提升泛化性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.15180v1](https://arxiv.org/pdf/2512.15180v1)**

> **作者:** Sanghyeok Chung; Eujin Kim; Donggun Kim; Gaeun Heo; Jeongbin You; Nahyun Lee; Sunmook Choi; Soyul Han; Seungsang Oh; Il-Youp Kwak
>
> **备注:** 3 pages, 1 figure, challenge paper
>
> **摘要:** Recent advances in audio generation have increased the risk of realistic environmental sound manipulation, motivating the ESDD 2026 Challenge as the first large-scale benchmark for Environmental Sound Deepfake Detection (ESDD). We propose BEAT2AASIST which extends BEATs-AASIST by splitting BEATs-derived representations along frequency or channel dimension and processing them with dual AASIST branches. To enrich feature representations, we incorporate top-k transformer layer fusion using concatenation, CNN-gated, and SE-gated strategies. In addition, vocoder-based data augmentation is applied to improve robustness against unseen spoofing methods. Experimental results on the official test sets demonstrate that the proposed approach achieves competitive performance across the challenge tracks.
>
---
#### [new 005] Time-Varying Audio Effect Modeling by End-to-End Adversarial Training
- **分类: cs.SD; cs.LG**

- **简介: 该论文属音频效果建模任务，旨在解决时间变异性硬件效果器（如phaser）在无控制信号下的黑盒建模难题。提出端到端GAN框架，含两阶段训练：先对抗学习调制分布，再监督微调同步内部状态，并设计 chirp-train 指标评估调制精度。**

- **链接: [https://arxiv.org/pdf/2512.15313v1](https://arxiv.org/pdf/2512.15313v1)**

> **作者:** Yann Bourdin; Pierrick Legrand; Fanny Roche
>
> **备注:** Submitted for review to the Journal of the Audio Engineering Society (JAES). Accompanying website: https://ybourdin.github.io/sptvmod
>
> **摘要:** Deep learning has become a standard approach for the modeling of audio effects, yet strictly black-box modeling remains problematic for time-varying systems. Unlike time-invariant effects, training models on devices with internal modulation typically requires the recording or extraction of control signals to ensure the time-alignment required by standard loss functions. This paper introduces a Generative Adversarial Network (GAN) framework to model such effects using only input-output audio recordings, removing the need for modulation signal extraction. We propose a convolutional-recurrent architecture trained via a two-stage strategy: an initial adversarial phase allows the model to learn the distribution of the modulation behavior without strict phase constraints, followed by a supervised fine-tuning phase where a State Prediction Network (SPN) estimates the initial internal states required to synchronize the model with the target. Additionally, a new objective metric based on chirp-train signals is developed to quantify modulation accuracy. Experiments modeling a vintage hardware phaser demonstrate the method's ability to capture time-varying dynamics in a fully black-box context.
>
---
#### [new 006] Synaspot: A Lightweight, Streaming Multi-modal Framework for Keyword Spotting with Audio-Text Synergy
- **分类: cs.SD**

- **简介: 该论文面向开放词汇关键词识别（KWS）任务，解决多模态模型参数大、难部署的问题。提出轻量级流式框架Synaspot：剥离语音注册中的说话人特征，融合音频与文本特征，并设计仅需编码器的数学解码机制，在保持高性能的同时大幅降低参数量。**

- **链接: [https://arxiv.org/pdf/2512.15124v1](https://arxiv.org/pdf/2512.15124v1)**

> **作者:** Kewei Li; Yinan Zhong; Xiaotao Liang; Tianchi Dai; Shaofei Xue
>
> **摘要:** Open-vocabulary keyword spotting (KWS) in continuous speech streams holds significant practical value across a wide range of real-world applications. While increasing attention has been paid to the role of different modalities in KWS, their effectiveness has been acknowledged. However, the increased parameter cost from multimodal integration and the constraints of end-to-end deployment have limited the practical applicability of such models. To address these challenges, we propose a lightweight, streaming multi-modal framework. First, we focus on multimodal enrollment features and reduce speaker-specific (voiceprint) information in the speech enrollment to extract speaker-irrelevant characteristics. Second, we effectively fuse speech and text features. Finally, we introduce a streaming decoding framework that only requires the encoder to extract features, which are then mathematically decoded with our three modal representations. Experiments on LibriPhase and WenetPrase demonstrate the performance of our model. Compared to existing streaming approaches, our method achieves better performance with significantly fewer parameters.
>
---
#### [new 007] Improving Underwater Acoustic Classification Through Learnable Gabor Filter Convolution and Attention Mechanisms
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文面向水下声学目标分类任务，旨在解决噪声复杂、数据稀缺及环境变化导致模型泛化差的问题。提出GSE ResNeXt模型，融合可学习Gabor卷积与Squeeze-and-Excitation注意力机制，提升特征判别性与训练稳定性，在多任务上显著优于主流基准模型。**

- **链接: [https://arxiv.org/pdf/2512.14714v1](https://arxiv.org/pdf/2512.14714v1)**

> **作者:** Lucas Cesar Ferreira Domingos; Russell Brinkworth; Paulo Eduardo Santos; Karl Sammut
>
> **摘要:** Remotely detecting and classifying underwater acoustic targets is critical for environmental monitoring and defence. However, the complex nature of ship-radiated and environmental underwater noise poses significant challenges to accurate signal processing. While recent advancements in machine learning have improved classification accuracy, issues such as limited dataset availability and a lack of standardised experimentation hinder generalisation and robustness. This paper introduces GSE ResNeXt, a deep learning architecture integrating learnable Gabor convolutional layers with a ResNeXt backbone enhanced by squeeze-and-excitation attention mechanisms. The Gabor filters serve as two-dimensional adaptive band-pass filters, extending the feature channel representation. Its combination with channel attention improves training stability and convergence while enhancing the model's ability to extract discriminative features. The model is evaluated on three classification tasks of increasing complexity. In particular, the impact of temporal differences between the training and testing data is explored, revealing that the distance between the vessel and sensor significantly affects performance. Results show that, GSE ResNeXt consistently outperforms baseline models like Xception, ResNet, and MobileNetV2, in terms of classification performance. Regarding stability and convergence, the addition of Gabor convolutions in the initial layers of the model represents a 28% reduction in training time. These results emphasise the importance of signal processing strategies in improving the reliability and generalisation of models under different environmental conditions, especially in data-limited underwater acoustic classification scenarios. Future developments should focus on mitigating the impact of environmental factors on input signals.
>
---
#### [new 008] Adaptive Multimodal Person Recognition: A Robust Framework for Handling Missing Modalities
- **分类: cs.CV; cs.SD; eess.AS; eess.IV**

- **简介: 该论文面向多模态人物识别任务，解决现实场景中模态缺失或退化导致性能下降的问题。提出自适应三模态框架，融合语音、人脸、手势，采用多任务学习、跨模态注意力与置信度加权融合，显著提升缺失模态下的鲁棒性与准确率。**

- **链接: [https://arxiv.org/pdf/2512.14961v1](https://arxiv.org/pdf/2512.14961v1)**

> **作者:** Aref Farhadipour; Teodora Vukovic; Volker Dellwo; Petr Motlicek; Srikanth Madikeri
>
> **备注:** 10 pages and 8 tables
>
> **摘要:** Person recognition systems often rely on audio, visual, or behavioral cues, but real-world conditions frequently result in missing or degraded modalities. To address this challenge, we propose a Trimodal person identification framework that integrates voice, face, and gesture modalities, while remaining robust to modality loss. Our approach leverages multi-task learning to process each modality independently, followed by a cross-attention and gated fusion mechanisms to facilitate interaction across modalities. Moreover, a confidence-weighted fusion strategy dynamically adapts to missing and low-quality data, ensuring optimal classification even in Unimodal or Bimodal scenarios. We evaluate our method on CANDOR, a newly introduced interview-based multimodal dataset, which we benchmark for the first time. Our results demonstrate that the proposed Trimodal system achieves 99.18% Top-1 accuracy on person identification tasks, outperforming conventional Unimodal and late-fusion approaches. In addition, we evaluate our model on the VoxCeleb1 dataset as a benchmark and reach 99.92% accuracy in Bimodal mode. Moreover, we show that our system maintains high accuracy even when one or two modalities are unavailable, making it a robust solution for real-world person recognition applications. The code and data for this work are publicly available.
>
---
#### [new 009] TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文面向音频驱动的长时 talking-video 生成任务，旨在解决现有方法依赖闭源数据、计算成本高、难以复现的问题。作者构建了开源大规模数据集 TalkVerse（230 万片段），并提出轻量级 5B DiT 基线模型，支持分钟级生成、零-shot 拍摄与语音重配音，显著降低推理开销。**

- **链接: [https://arxiv.org/pdf/2512.14938v1](https://arxiv.org/pdf/2512.14938v1)**

> **作者:** Zhenzhi Wang; Jian Wang; Ke Ma; Dahua Lin; Bing Zhou
>
> **备注:** open-sourced single-person full-body talking video generation dataset, training code and checkpoints
>
> **摘要:** We introduce TalkVerse, a large-scale, open corpus for single-person, audio-driven talking video generation designed to enable fair, reproducible comparison across methods. While current state-of-the-art systems rely on closed data or compute-heavy models, TalkVerse offers 2.3 million high-resolution (720p/1080p) audio-video synchronized clips totaling 6.3k hours. These are curated from over 60k hours of video via a transparent pipeline that includes scene-cut detection, aesthetic assessment, strict audio-visual synchronization checks, and comprehensive annotations including 2D skeletons and structured visual/audio-style captions. Leveraging TalkVerse, we present a reproducible 5B DiT baseline built on Wan2.2-5B. By utilizing a video VAE with a high downsampling ratio and a sliding window mechanism with motion-frame context, our model achieves minute-long generation with low drift. It delivers comparable lip-sync and visual quality to the 14B Wan-S2V model but with 10$\times$ lower inference cost. To enhance storytelling in long videos, we integrate an MLLM director to rewrite prompts based on audio and visual cues. Furthermore, our model supports zero-shot video dubbing via controlled latent noise injection. We open-source the dataset, training recipes, and 5B checkpoints to lower barriers for research in audio-driven human video generation. Project Page: https://zhenzhiwang.github.io/talkverse/
>
---
#### [new 010] O-EENC-SD: Efficient Online End-to-End Neural Clustering for Speaker Diarization
- **分类: cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出O-EENC-SD，一种高效在线端到端说话人日志系统。任务是说话人日志（区分谁在何时说话），旨在解决现有在线方法计算开销大、依赖超参数等问题。工作包括：基于EEND-EDA构建在线框架，设计RNN拼接机制与新型质心精炼解码器，并通过消融验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.15229v1](https://arxiv.org/pdf/2512.15229v1)**

> **作者:** Elio Gruttadauria; Mathieu Fontaine; Jonathan Le Roux; Slim Essid
>
> **摘要:** We introduce O-EENC-SD: an end-to-end online speaker diarization system based on EEND-EDA, featuring a novel RNN-based stitching mechanism for online prediction. In particular, we develop a novel centroid refinement decoder whose usefulness is assessed through a rigorous ablation study. Our system provides key advantages over existing methods: a hyperparameter-free solution compared to unsupervised clustering approaches, and a more efficient alternative to current online end-to-end methods, which are computationally costly. We demonstrate that O-EENC-SD is competitive with the state of the art in the two-speaker conversational telephone speech domain, as tested on the CallHome dataset. Our results show that O-EENC-SD provides a great trade-off between DER and complexity, even when working on independent chunks with no overlap, making the system extremely efficient.
>
---
## 更新

#### [replaced 001] Sparse Autoencoders Make Audio Foundation Models more Explainable
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属可解释性研究任务，旨在解决音频基础模型表征难以理解的问题。作者用稀疏自编码器（SAEs）分析预训练模型隐层表示，在歌唱技巧分类案例中验证SAEs既能保留原始信息与标签，又能提升声学属性解耦，从而增强模型可解释性。**

- **链接: [https://arxiv.org/pdf/2509.24793v2](https://arxiv.org/pdf/2509.24793v2)**

> **作者:** Théo Mariotte; Martin Lebourdais; Antonio Almudévar; Marie Tahon; Alfonso Ortega; Nicolas Dugué
>
> **备注:** 5 pages, 5 figures, 1 table, submitted to ICASSP 2026
>
> **摘要:** Audio pretrained models are widely employed to solve various tasks in speech processing, sound event detection, or music information retrieval. However, the representations learned by these models are unclear, and their analysis mainly restricts to linear probing of the hidden representations. In this work, we explore the use of Sparse Autoencoders (SAEs) to analyze the hidden representations of pretrained models, focusing on a case study in singing technique classification. We first demonstrate that SAEs retain both information about the original representations and class labels, enabling their internal structure to provide insights into self-supervised learning systems. Furthermore, we show that SAEs enhance the disentanglement of vocal attributes, establishing them as an effective tool for identifying the underlying factors encoded in the representations.
>
---
#### [replaced 002] Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文面向语音情感识别（SER）任务，旨在解决大模型在SER中易产生幻觉、识别不稳定的问题。提出C²SER模型，融合Whisper与Emotion2Vec-S实现语义与声学感知，并引入显式到隐式思维链的自蒸馏机制，提升稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2502.18186v2](https://arxiv.org/pdf/2502.18186v2)**

> **作者:** Zhixian Zhao; Xinfa Zhu; Xinsheng Wang; Shuiyuan Wang; Xuelong Geng; Wenjie Tian; Lei Xie
>
> **备注:** This work has been submitted to the IEEE TASLP for possible publication
>
> **摘要:** Large-scale audio language models (ALMs), such as Qwen2-Audio, are capable of comprehending diverse audio signal, performing audio analysis and generating textual responses. However, in speech emotion recognition (SER), ALMs often suffer from hallucinations, resulting in misclassifications or irrelevant outputs. To address these challenges, we propose C$^2$SER, a novel ALM designed to enhance the stability and accuracy of SER through Contextual perception and Chain of Thought (CoT). C$^2$SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C$^2$SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C$^2$SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C$^2$SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition. We release the training code, checkpoints, and test sets to facilitate further research.
>
---
#### [replaced 003] Single-channel speech enhancement by using psychoacoustical model inspired fusion framework
- **分类: cs.SD; eess.AS**

- **简介: 该论文属单通道语音增强任务，旨在解决噪声下语音质量与可懂度难以兼顾的问题。提出一种融合声学域（基于听觉心理模型的STSA估计器）和调制域（利用频率选择性）优势的框架，在提升语音质量的同时改善可懂度。**

- **链接: [https://arxiv.org/pdf/2202.05272v2](https://arxiv.org/pdf/2202.05272v2)**

> **作者:** Suman Samui
>
> **备注:** arXiv admin note: text overlap with arXiv:2202.04882
>
> **摘要:** When the parameters of Bayesian Short-time Spectral Amplitude (STSA) estimator for speech enhancement are selected based on the characteristics of the human auditory system, the gain function of the estimator becomes more flexible. Although this type of estimator in acoustic domain is quite effective in reducing the back-ground noise at high frequencies, it produces more speech distortions, which make the high-frequency contents of the speech such as friciatives less perceptible in heavy noise conditions, resulting in intelligibility reduction. On the other hand, the speech enhancement scheme, which exploits the psychoacoustic evidence of frequency selectivity in the modulation domain, is found to be able to increase the intelligibility of noisy speech by a substantial amount, but also suffers from the temporal slurring problem due to its essential design constraint. In order to achieve the joint improvements in both the perceived speech quality and intelligibility, we proposed and investigated a fusion framework by combining the merits of acoustic and modulation domain approaches while avoiding their respective weaknesses. Objective measure evaluation shows that the proposed speech enhancement fusion framework can provide consistent improvements in the perceived speech quality and intelligibility across different SNR levels in various noise conditions, while compared to the other baseline techniques.
>
---
#### [replaced 004] Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出SMIA攻击，属语音安全领域的对抗攻击任务，旨在突破语音认证与反欺骗系统。通过频谱掩蔽和插值技术，在人耳不可听频段扰动AI语音，生成高成功率（最高100%）的隐蔽对抗样本，揭示现有静态防御的严重缺陷。**

- **链接: [https://arxiv.org/pdf/2509.07677v3](https://arxiv.org/pdf/2509.07677v3)**

> **作者:** Kamel Kamel; Hridoy Sankar Dutta; Keshav Sood; Sunil Aryal
>
> **摘要:** Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.
>
---
#### [replaced 005] Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文面向音乐情绪识别（MER）任务，旨在解决高质量标注数据稀缺与跨曲目特征漂移问题。作者构建了专家标注的Memo2496数据集，并提出DAMER框架，含双流注意力融合、渐进置信标注和风格锚定记忆学习三模块，显著提升 arousal 识别精度。**

- **链接: [https://arxiv.org/pdf/2512.13998v2](https://arxiv.org/pdf/2512.13998v2)**

> **作者:** Qilin Li; C. L. Philip Chen; Tong Zhang
>
> **摘要:** Music Emotion Recogniser (MER) research faces challenges due to limited high-quality annotated datasets and difficulties in addressing cross-track feature drift. This work presents two primary contributions to address these issues. Memo2496, a large-scale dataset, offers 2496 instrumental music tracks with continuous valence arousal labels, annotated by 30 certified music specialists. Annotation quality is ensured through calibration with extreme emotion exemplars and a consistency threshold of 0.25, measured by Euclidean distance in the valence arousal space. Furthermore, the Dual-view Adaptive Music Emotion Recogniser (DAMER) is introduced. DAMER integrates three synergistic modules: Dual Stream Attention Fusion (DSAF) facilitates token-level bidirectional interaction between Mel spectrograms and cochleagrams via cross attention mechanisms; Progressive Confidence Labelling (PCL) generates reliable pseudo labels employing curriculum-based temperature scheduling and consistency quantification using Jensen Shannon divergence; and Style Anchored Memory Learning (SAML) maintains a contrastive memory queue to mitigate cross-track feature drift. Extensive experiments on the Memo2496, 1000songs, and PMEmo datasets demonstrate DAMER's state-of-the-art performance, improving arousal dimension accuracy by 3.43%, 2.25%, and 0.17%, respectively. Ablation studies and visualisation analyses validate each module's contribution. Both the dataset and source code are publicly available.
>
---
