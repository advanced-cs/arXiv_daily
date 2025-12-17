# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Adapting Speech Language Model to Singing Voice Synthesis
- **分类: cs.SD**

- **简介: 该论文属歌唱语音合成（SVS）任务，旨在探索大语言模型在非语音任务上的泛化能力。作者将预训练的1.7B参数语音语言模型适配至SVS，仅用135小时合成数据，通过音乐谱条件建模、多流token预测、条件流匹配生成梅尔谱及vocoder合成歌声，性能媲美主流离散token SVS模型。**

- **链接: [https://arxiv.org/pdf/2512.14657v1](https://arxiv.org/pdf/2512.14657v1)**

> **作者:** Yiwen Zhao; Jiatong Shi; Jinchuan Tian; Yuxun Tang; Jiarui Hai; Jionghao Han; Shinji Watanabe
>
> **备注:** Accepted by NeurIPS 2025 workshop AI for Music
>
> **摘要:** Speech Language Models (SLMs) have recently emerged as a unified paradigm for addressing a wide range of speech-related tasks, including text-to-speech (TTS), speech enhancement (SE), and automatic speech recognition (ASR). However, the generalization capability of large-scale pre-trained SLMs remains underexplored. In this work, we adapt a 1.7B parameter TTS pretrained SLM for singing voice synthesis (SVS), using only a 135-hour synthetic singing corpus, ACE-Opencpop. Building upon the ESPNet-SpeechLM, our recipe involves the following procedure: (1) tokenization of music score conditions and singing waveforms, (2) multi-stream language model token prediction, (3) conditional flow matching-based mel-spectrogram generation. (4) a mel-to-wave vocoder. Experimental results demonstrate that our adapted SLM generalizes well to SVS and achieves performance comparable to leading discrete token-based SVS models.
>
---
#### [new 002] GLM-TTS Technical Report
- **分类: cs.SD**

- **简介: 该论文提出GLM-TTS——一种高效、可控、高保真语音合成（TTS）系统。针对生产中质量、可控性与部署效率问题，其采用两阶段架构（文本→音符→波形），引入带基频约束的语音分词器、GRPO多奖励强化学习及LoRA语音定制，仅用10万小时数据达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.14291v1](https://arxiv.org/pdf/2512.14291v1)**

> **作者:** Jiayan Cui; Zhihan Yang; Naihan Li; Jiankun Tian; Xingyu Ma; Yi Zhang; Guangyu Chen; Runxuan Yang; Yuqing Cheng; Yizhi Zhou; Guochen Yu; Xiaotao Gu; Jie Tang
>
> **摘要:** This work proposes GLM-TTS, a production-level TTS system designed for efficiency, controllability, and high-fidelity speech generation. GLM-TTS follows a two-stage architecture, consisting of a text-to-token autoregressive model and a token-to-waveform diffusion model. With only 100k hours of training data, GLM-TTS achieves state-of-the-art performance on multiple open-source benchmarks. To meet production requirements, GLM-TTS improves speech quality through an optimized speech tokenizer with fundamental frequency constraints and a GRPO-based multi-reward reinforcement learning framework that jointly optimizes pronunciation, speaker similarity, and expressive prosody. In parallel, the system enables efficient and controllable deployment via parameter-efficient LoRA-based voice customization and a hybrid phoneme-text input scheme that provides precise pronunciation control. Our code is available at https://github.com/zai-org/GLM-TTS. Real-time speech synthesis demos are provided via Z.ai (audio.z.ai), the Zhipu Qingyan app/web (chatglm.cn).
>
---
#### [new 003] Segmental Attention Decoding With Long Form Acoustic Encodings
- **分类: eess.AS; cs.CL**

- **简介: 该论文属语音识别任务，旨在解决注意力编码器-解码器（AED）模型在长音频输入下因位置编码失效导致的解码失序问题。提出四点改进：显式绝对位置注入、长上下文训练、分段拼接与语义分段对齐，弥合连续与分段音频的性能差距。**

- **链接: [https://arxiv.org/pdf/2512.14652v1](https://arxiv.org/pdf/2512.14652v1)**

> **作者:** Pawel Swietojanski; Xinwei Li; Mingbin Xu; Takaaki Hori; Dogan Can; Xiaodan Zhuang
>
> **备注:** 5 pages, 1 fig
>
> **摘要:** We address the fundamental incompatibility of attention-based encoder-decoder (AED) models with long-form acoustic encodings. AED models trained on segmented utterances learn to encode absolute frame positions by exploiting limited acoustic context beyond segment boundaries, but fail to generalize when decoding long-form segments where these cues vanish. The model loses ability to order acoustic encodings due to permutation invariance of keys and values in cross-attention. We propose four modifications: (1) injecting explicit absolute positional encodings into cross-attention for each decoded segment, (2) long-form training with extended acoustic context to eliminate implicit absolute position encoding, (3) segment concatenation to cover diverse segmentations needed during training, and (4) semantic segmentation to align AED-decoded segments with training segments. We show these modifications close the accuracy gap between continuous and segmented acoustic encodings, enabling auto-regressive use of the attention decoder.
>
---
#### [new 004] MuseCPBench: an Empirical Study of Music Editing Methods through Music Context Preservation
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦音乐编辑任务，旨在解决现有方法忽视音乐上下文保持（MCP）且评估标准不统一的问题。作者构建首个MCP专用基准MuseCPBench，覆盖四类音乐要素，系统评测五种基线方法，揭示普遍存在的MCP缺陷并提供改进建议。**

- **链接: [https://arxiv.org/pdf/2512.14629v1](https://arxiv.org/pdf/2512.14629v1)**

> **作者:** Yash Vishe; Eric Xue; Xunyi Jiang; Zachary Novack; Junda Wu; Julian McAuley; Xin Xu
>
> **摘要:** Music editing plays a vital role in modern music production, with applications in film, broadcasting, and game development. Recent advances in music generation models have enabled diverse editing tasks such as timbre transfer, instrument substitution, and genre transformation. However, many existing works overlook the evaluation of their ability to preserve musical facets that should remain unchanged during editing a property we define as Music Context Preservation (MCP). While some studies do consider MCP, they adopt inconsistent evaluation protocols and metrics, leading to unreliable and unfair comparisons. To address this gap, we introduce the first MCP evaluation benchmark, MuseCPBench, which covers four categories of musical facets and enables comprehensive comparisons across five representative music editing baselines. Through systematic analysis along musical facets, methods, and models, we identify consistent preservation gaps in current music editing methods and provide insightful explanations. We hope our findings offer practical guidance for developing more effective and reliable music editing strategies with strong MCP capability
>
---
#### [new 005] Joint Multimodal Contrastive Learning for Robust Spoken Term Detection and Keyword Spotting
- **分类: cs.SD; cs.LG**

- **简介: 该论文属语音检索任务，旨在解决现有声学词嵌入（AWEs）方法在STD与KWS中依赖单模态监督、音-音与音-文对齐分离优化及需专用模型等问题。提出联合多模态对比学习框架，在统一嵌入空间中同步优化音-文（CLAP式）和音-音（DWD式）对比损失，提升鲁棒性与通用性。**

- **链接: [https://arxiv.org/pdf/2512.14115v1](https://arxiv.org/pdf/2512.14115v1)**

> **作者:** Ramesh Gundluru; Shubham Gupta; Sri Rama Murty K
>
> **摘要:** Acoustic Word Embeddings (AWEs) improve the efficiency of speech retrieval tasks such as Spoken Term Detection (STD) and Keyword Spotting (KWS). However, existing approaches suffer from limitations, including unimodal supervision, disjoint optimization of audio-audio and audio-text alignment, and the need for task-specific models. To address these shortcomings, we propose a joint multimodal contrastive learning framework that unifies both acoustic and cross-modal supervision in a shared embedding space. Our approach simultaneously optimizes: (i) audio-text contrastive learning, inspired by the CLAP loss, to align audio and text representations and (ii) audio-audio contrastive learning, via Deep Word Discrimination (DWD) loss, to enhance intra-class compactness and inter-class separation. The proposed method outperforms existing AWE baselines on word discrimination task while flexibly supporting both STD and KWS. To our knowledge, this is the first comprehensive approach of its kind.
>
---
#### [new 006] Robust Training of Singing Voice Synthesis Using Prior and Posterior Uncertainty
- **分类: cs.SD**

- **简介: 该论文属歌唱语音合成（SVS）任务，旨在缓解数据稀缺导致的长尾问题（如音高分布不均、稀有唱法性能差）。提出基于先验与后验不确定性的鲁棒训练方法：引入样本级可微对抗增强提升先验不确定性，并添加帧级不确定性预测模块以动态聚焦低置信度片段。**

- **链接: [https://arxiv.org/pdf/2512.14653v1](https://arxiv.org/pdf/2512.14653v1)**

> **作者:** Yiwen Zhao; Jiatong Shi; Yuxun Tang; William Chen; Shinji Watanabe
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Singing voice synthesis (SVS) has seen remarkable advancements in recent years. However, compared to speech and general audio data, publicly available singing datasets remain limited. In practice, this data scarcity often leads to performance degradation in long-tail scenarios, such as imbalanced pitch distributions or rare singing styles. To mitigate these challenges, we propose uncertainty-based optimization to improve the training process of end-to-end SVS models. First, we introduce differentiable data augmentation in the adversarial training, which operates in a sample-wise manner to increase the prior uncertainty. Second, we incorporate a frame-level uncertainty prediction module that estimates the posterior uncertainty, enabling the model to allocate more learning capacity to low-confidence segments. Empirical results on the Opencpop and Ofuton-P, across Chinese and Japanese, demonstrate that our approach improves performance in various perspectives.
>
---
#### [new 007] Ensemble-Guided Distillation for Compact and Robust Acoustic Scene Classification on Edge Devices
- **分类: cs.SD**

- **简介: 该论文面向边缘设备的声学场景分类（ASC）任务，解决模型轻量化与鲁棒性难题。提出集成引导蒸馏框架：设计紧凑学生网络（含深度可分离块与全局响应归一化），并构建多样化教师集成，通过双融合头学习加权蒸馏，兼顾精度与部署效率。**

- **链接: [https://arxiv.org/pdf/2512.13905v1](https://arxiv.org/pdf/2512.13905v1)**

> **作者:** Hossein Sharify; Behnam Raoufi; Mahdy Ramezani; Khosrow Hajsadeghi; Saeed Bagheri Shouraki
>
> **摘要:** We present a compact, quantization-ready acoustic scene classification (ASC) framework that couples an efficient student network with a learned teacher ensemble and knowledge distillation. The student backbone uses stacked depthwise-separable "expand-depthwise-project" blocks with global response normalization to stabilize training and improve robustness to device and noise variability, while a global pooling head yields class logits for efficient edge inference. To inject richer inductive bias, we assemble a diverse set of teacher models and learn two complementary fusion heads: z1, which predicts per-teacher mixture weights using a student-style backbone, and z2, a lightweight MLP that performs per-class logit fusion. The student is distilled from the ensemble via temperature-scaled soft targets combined with hard labels, enabling it to approximate the ensemble's decision geometry with a single compact model. Evaluated on the TAU Urban Acoustic Scenes 2022 Mobile benchmark, our approach achieves state-of-the-art (SOTA) results on the TAU dataset under matched edge-deployment constraints, demonstrating strong performance and practicality for mobile ASC.
>
---
#### [new 008] Investigating the impact of stereo processing -- a study for extending the Open Dataset of Audio Quality (ODAQ)
- **分类: eess.AS**

- **简介: 该论文属音频质量评估任务，旨在探究立体声处理（LR/MS）对主观听感的影响。为扩展ODAQ数据集，作者将单声道失真素材结合不同立体声编码方式，开展含直接对比与非对比的听力测试，收集16位专家评分，分析空间特性与呈现方式对音质评价的影响机制。**

- **链接: [https://arxiv.org/pdf/2512.14259v1](https://arxiv.org/pdf/2512.14259v1)**

> **作者:** Sascha Dick; Christoph Thompson; Chih-Wei Wu; Pablo Delgado; Phillip A. Williams; Matteo Torcoli
>
> **备注:** Presented at the Audio Engineering Society (AES) 159th Convention, October 2025, Paper number 365, see https://aes2.org/publications/elibrary-page/?id=23039
>
> **摘要:** In this paper, we present an initial study for extending Open Dataset of Audio Quality (ODAQ) towards the impact of stereo processing. Monaural artifacts from ODAQ were adapted in combinations with left-right (LR) and mid-side (MS) stereo processing, across stimuli including solo instruments, typical wide stereo mixes and and hard-panned mixes. Listening tests in different presentation context -- with and without direct comparison of MS and LR conditions -- were conducted to collect subjective data beyond monaural artifacts while also scrutinizing the listening test methodology. The ODAQ dataset is extended with new material along with subjective scores from 16 expert listeners. The listening test results show substantial influences of the stimuli's spatial characteristics as well as the presentation context. Notably, several significant disparities between LR and MS only occur when presented in direct comparison. The findings suggest that listeners primarily assess timbral impairments when spatial characteristics are consistent and focus on stereo image only when timbral quality is similar. The rating of an additional mono anchor was overall consistent across different stereo characteristics, averaging at 65 on the MUSHRA scale, further corroborating that listeners prioritize timbral over spatial impressions.
>
---
#### [new 009] Scalable Frameworks for Real-World Audio-Visual Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文面向真实场景下的音视频语音识别（AVSR）任务，解决噪声与视觉干扰导致性能下降的问题。提出三层可扩展框架：鲁棒音视频表征学习、自适应多模态架构设计、与大模型模块化集成，以提升系统在现实环境中的鲁棒性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.14083v1](https://arxiv.org/pdf/2512.14083v1)**

> **作者:** Sungnyun Kim
>
> **备注:** PhD Dissertation
>
> **摘要:** The practical deployment of Audio-Visual Speech Recognition (AVSR) systems is fundamentally challenged by significant performance degradation in real-world environments, characterized by unpredictable acoustic noise and visual interference. This dissertation posits that a systematic, hierarchical approach is essential to overcome these challenges, achieving the robust scalability at the representation, architecture, and system levels. At the representation level, we investigate methods for building a unified model that learns audio-visual features inherently robust to diverse real-world corruptions, thereby enabling generalization to new environments without specialized modules. To address architectural scalability, we explore how to efficiently expand model capacity while ensuring the adaptive and reliable use of multimodal inputs, developing a framework that intelligently allocates computational resources based on the input characteristics. Finally, at the system level, we present methods to expand the system's functionality through modular integration with large-scale foundation models, leveraging their powerful cognitive and generative capabilities to maximize final recognition accuracy. By systematically providing solutions at each of these three levels, this dissertation aims to build a next-generation, robust, and scalable AVSR system with high reliability in real-world applications.
>
---
#### [new 010] Sound and Music Biases in Deep Music Transcription Models: A Systematic Analysis
- **分类: cs.SD; cs.LG**

- **简介: 该论文属自动音乐转录（AMT）任务，旨在揭示深度AMT模型在声音与音乐维度上的偏差问题。作者构建MDS数据集，系统评估模型在风格、动态、复调等分布偏移下的泛化能力，发现显著性能下降，并指出动态估计更脆弱、音乐感知指标更有效。**

- **链接: [https://arxiv.org/pdf/2512.14602v1](https://arxiv.org/pdf/2512.14602v1)**

> **作者:** Lukáš Samuel Marták; Patricia Hu; Gerhard Widmer
>
> **备注:** pre-print of the upcoming EURASIP JASM journal article
>
> **摘要:** Automatic Music Transcription (AMT) -- the task of converting music audio into note representations -- has seen rapid progress, driven largely by deep learning systems. Due to the limited availability of richly annotated music datasets, much of the progress in AMT has been concentrated on classical piano music, and even a few very specific datasets. Whether these systems can generalize effectively to other musical contexts remains an open question. Complementing recent studies on distribution shifts in sound (e.g., recording conditions), in this work we investigate the musical dimension -- specifically, variations in genre, dynamics, and polyphony levels. To this end, we introduce the MDS corpus, comprising three distinct subsets -- (1) Genre, (2) Random, and (3) MAEtest -- to emulate different axes of distribution shift. We evaluate the performance of several state-of-the-art AMT systems on the MDS corpus using both traditional information-retrieval and musically-informed performance metrics. Our extensive evaluation isolates and exposes varying degrees of performance degradation under specific distribution shifts. In particular, we measure a note-level F1 performance drop of 20 percentage points due to sound, and 14 due to genre. Generally, we find that dynamics estimation proves more vulnerable to musical variation than onset prediction. Musically informed evaluation metrics, particularly those capturing harmonic structure, help identify potential contributing factors. Furthermore, experiments with randomly generated, non-musical sequences reveal clear limitations in system performance under extreme musical distribution shifts. Altogether, these findings offer new evidence of the persistent impact of the Corpus Bias problem in deep AMT systems.
>
---
#### [new 011] Toward Noise-Aware Audio Deepfake Detection: Survey, SNR-Benchmarks, and Practical Recipes
- **分类: cs.SD; cs.AI**

- **简介: 该论文面向噪声鲁棒的音频深度伪造检测任务，解决现实场景（噪声、混响、信道失真）下检测性能下降问题。工作包括：调研现有模型鲁棒性；构建可控SNR基准（MS-SNSD+ASVspoof）；评估多条件训练效果；验证微调显著降低低SNR下的EER。**

- **链接: [https://arxiv.org/pdf/2512.13744v1](https://arxiv.org/pdf/2512.13744v1)**

> **作者:** Udayon Sen; Alka Luqman; Anupam Chattopadhyay
>
> **备注:** 6 pages
>
> **摘要:** Deepfake audio detection has progressed rapidly with strong pre-trained encoders (e.g., WavLM, Wav2Vec2, MMS). However, performance in realistic capture conditions - background noise (domestic/office/transport), room reverberation, and consumer channels - often lags clean-lab results. We survey and evaluate robustness for state-of-the-art audio deepfake detection models and present a reproducible framework that mixes MS-SNSD noises with ASVspoof 2021 DF utterances to evaluate under controlled signal-to-noise ratios (SNRs). SNR is a measured proxy for noise severity used widely in speech; it lets us sweep from near-clean (35 dB) to very noisy (-5 dB) to quantify graceful degradation. We study multi-condition training and fixed-SNR testing for pretrained encoders (WavLM, Wav2Vec2, MMS), reporting accuracy, ROC-AUC, and EER on binary and four-class (authenticity x corruption) tasks. In our experiments, finetuning reduces EER by 10-15 percentage points at 10-0 dB SNR across backbones.
>
---
#### [new 012] Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文面向音乐情绪识别（MER）任务，旨在解决高质量标注数据稀缺与跨曲目特征漂移问题。提出Memo2496大规模专家标注数据集（2496首器乐曲，连续价态-唤醒标签）和DAMER双视图自适应模型，含DSAF、PCL、SAML三大模块，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.13998v1](https://arxiv.org/pdf/2512.13998v1)**

> **作者:** Qilin Li; C. L. Philip Chen; TongZhang
>
> **摘要:** Music Emotion Recogniser (MER) research faces challenges due to limited high-quality annotated datasets and difficulties in addressing cross-track feature drift. This work presents two primary contributions to address these issues. Memo2496, a large-scale dataset, offers 2496 instrumental music tracks with continuous valence arousal labels, annotated by 30 certified music specialists. Annotation quality is ensured through calibration with extreme emotion exemplars and a consistency threshold of 0.25, measured by Euclidean distance in the valence arousal space. Furthermore, the Dual-view Adaptive Music Emotion Recogniser (DAMER) is introduced. DAMER integrates three synergistic modules: Dual Stream Attention Fusion (DSAF) facilitates token-level bidirectional interaction between Mel spectrograms and cochleagrams via cross attention mechanisms; Progressive Confidence Labelling (PCL) generates reliable pseudo labels employing curriculum-based temperature scheduling and consistency quantification using Jensen Shannon divergence; and Style Anchored Memory Learning (SAML) maintains a contrastive memory queue to mitigate cross-track feature drift. Extensive experiments on the Memo2496, 1000songs, and PMEmo datasets demonstrate DAMER's state-of-the-art performance, improving arousal dimension accuracy by 3.43%, 2.25%, and 0.17%, respectively. Ablation studies and visualisation analyses validate each module's contribution. Both the dataset and source code are publicly available.
>
---
#### [new 013] Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文面向**口语对话摘要任务**，旨在解决**情感感知与语音建模缺乏对齐数据**的问题。作者构建了首个含原始音频、事实摘要、情感摘要及细粒度副语言标签（情感/性别/年龄/语速/音高）的对话数据集Spoken DialogSum（13.46k样本），并验证端到端Audio-LLM优于ASR-LLM级联系统。**

- **链接: [https://arxiv.org/pdf/2512.14687v1](https://arxiv.org/pdf/2512.14687v1)**

> **作者:** Yen-Ju Lu; Kunxiao Gao; Mingrui Liang; Helin Wang; Thomas Thebaud; Laureano Moro-Velazquez; Najim Dehak; Jesus Villalba
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. The dataset is available online at https://fatfat-emosum.github.io/EmoDialog-Sum-Audio-Samples/. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling.
>
---
#### [new 014] Privacy-Enhancing Infant Cry Classification with Federated Transformers and Denoising Regularization
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文面向婴儿哭声分类任务，解决隐私敏感、噪声干扰与域偏移问题。提出融合去噪自编码器、卷积分词器和Transformer的端到端框架，采用带8位适配器与控制变量的联邦学习，并支持设备端去噪、OOD拒识与实时边缘推理。**

- **链接: [https://arxiv.org/pdf/2512.13880v1](https://arxiv.org/pdf/2512.13880v1)**

> **作者:** Geofrey Owino; Bernard Shibwabo
>
> **备注:** This paper was accepted for presentation and presented at the 2025 International Conference on Computer Engineering, Network, and Intelligent Multimedia (CENIM 2025)
>
> **摘要:** Infant cry classification can aid early assessment of infant needs. However, deployment of such solutions is limited by privacy concerns around audio data, sensitivity to background noise, and domain shift across recording environments. We present an end-to-end infant cry analysis pipeline that integrates a denoising autoencoder (DAE), a convolutional tokenizer, and a Transformer encoder trained using communication-efficient federated learning (FL). The system performs on-device denoising, adaptive segmentation, post hoc calibration, and energy-based out-of-distribution (OOD) abstention. Federated training employs a regularized control variate update with 8-bit adapter deltas under secure aggregation. Using the Baby Chillanto and Donate-a-Cry datasets with ESC-50 noise overlays, the model achieves a macro F1 score of 0.938, an AUC of 0.962, and an Expected Calibration Error (ECE) of 0.032, while reducing per-round client upload from approximately 36 to 42 MB to 3.3 MB. Real-time edge inference on an NVIDIA Jetson Nano (4 GB, TensorRT FP16) achieves 96 ms per one-second spectrogram frame. These results demonstrate a practical path toward privacy-preserving, noise-robust, and communication-efficient infant cry classification suitable for federated deployment.
>
---
#### [new 015] Linguists should learn to love speech-based deep learning models
- **分类: cs.CL; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文属学术评论任务，旨在纠正语言学界过度关注文本型大模型的倾向。它指出文本LLM无法覆盖语音相关的语言现象，主张转向语音驱动的深度学习模型，以更好支撑语言学理论解释与实证研究。**

- **链接: [https://arxiv.org/pdf/2512.14506v1](https://arxiv.org/pdf/2512.14506v1)**

> **作者:** Marianne de Heer Kloots; Paul Boersma; Willem Zuidema
>
> **备注:** Commentary on Futrell, R., & Mahowald, K. arXiv:2501.17047 (in press). How Linguistics Learned to Stop Worrying and Love the Language Models. Behavioural and Brain Sciences
>
> **摘要:** Futrell and Mahowald present a useful framework bridging technology-oriented deep learning systems and explanation-oriented linguistic theories. Unfortunately, the target article's focus on generative text-based LLMs fundamentally limits fruitful interactions with linguistics, as many interesting questions on human language fall outside what is captured by written text. We argue that audio-based deep learning models can and should play a crucial role.
>
---
#### [new 016] Multilingual and Continuous Backchannel Prediction: A Cross-lingual Study
- **分类: cs.CL; cs.HC; cs.SD**

- **简介: 该论文研究多语言连续反馈语（backchannel）预测任务，旨在揭示日、英、中三语在反馈时机上的跨语言差异。提出基于Transformer的多语言帧级模型，联合辅助任务训练，分析线索使用、上下文长度影响及零-shot迁移效果，并实现CPU实时推理。**

- **链接: [https://arxiv.org/pdf/2512.14085v1](https://arxiv.org/pdf/2512.14085v1)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Taiga Mori; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at International Workshop on Spoken Dialogue Systems Technology 2026 (IWSDS 2026) and represents the author's version of the work
>
> **摘要:** We present a multilingual, continuous backchannel prediction model for Japanese, English, and Chinese, and use it to investigate cross-linguistic timing behavior. The model is Transformer-based and operates at the frame level, jointly trained with auxiliary tasks on approximately 300 hours of dyadic conversations. Across all three languages, the multilingual model matches or surpasses monolingual baselines, indicating that it learns both language-universal cues and language-specific timing patterns. Zero-shot transfer with two-language training remains limited, underscoring substantive cross-lingual differences. Perturbation analyses reveal distinct cue usage: Japanese relies more on short-term linguistic information, whereas English and Chinese are more sensitive to silence duration and prosodic variation; multilingual training encourages shared yet adaptable representations and reduces overreliance on pitch in Chinese. A context-length study further shows that Japanese is relatively robust to shorter contexts, while Chinese benefits markedly from longer contexts. Finally, we integrate the trained model into a real-time processing software, demonstrating CPU-only inference. Together, these findings provide a unified model and empirical evidence for how backchannel timing differs across languages, informing the design of more natural, culturally-aware spoken dialogue systems.
>
---
## 更新

#### [replaced 001] PhraseVAE and PhraseLDM: Latent Diffusion for Full-Song Multitrack Symbolic Music Generation
- **分类: cs.SD**

- **简介: 该论文属符号音乐生成任务，旨在解决长序列建模难题（如上下文受限、结构弱）。提出PhraseVAE（压缩音符序列为64维短语隐表示）和PhraseLDM（基于隐空间的非自回归全曲生成），支持128小节多轨音乐秒级生成，提升结构连贯性与音乐性。**

- **链接: [https://arxiv.org/pdf/2512.11348v2](https://arxiv.org/pdf/2512.11348v2)**

> **作者:** Longshen Ou; Ye Wang
>
> **摘要:** This technical report presents a new paradigm for full-song symbolic music generation. Existing symbolic models operate on note-attribute tokens and suffer from extremely long sequences, limited context length, and weak support for long-range structure. We address these issues by introducing PhraseVAE and PhraseLDM, the first latent diffusion framework designed for full-song multitrack symbolic music. PhraseVAE compresses an arbitrary variable-length polyphonic note sequence into a single compact 64-dimensional phrase-level latent representation with high reconstruction fidelity, allowing a well-structured latent space and efficient generative modeling. Built on this latent space, PhraseLDM generates an entire multi-track song in a single pass without any autoregressive components. The system eliminates bar-wise sequential modeling, supports up to 128 bars of music (8 minutes at 64 bpm), and produces complete songs with coherent local texture, idiomatic instrument patterns, and clear global structure. With only 45M parameters, our framework generates a full song within seconds while maintaining competitive musical quality and generation diversity. Together, these results show that phrase-level latent diffusion provides an effective and scalable solution to long-sequence modeling in symbolic music generation. We hope this work encourages future symbolic music research to move beyond note-attribute tokens and to consider phrase-level units as a more effective and musically meaningful modeling target.
>
---
#### [replaced 002] Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文面向跨语言语音识别（ASR）任务，旨在消除对发音词典的依赖。提出JSA-SPG方法：将音素建模为离散隐变量，联合训练S2P、P2G与G2P模型，采用联合随机近似（JSA）算法优化，并引入MLS解码与P2G增强。实验验证其在低资源语言上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2507.06249v2](https://arxiv.org/pdf/2507.06249v2)**

> **作者:** Saierdaer Yusuyin; Te Ma; Hao Huang; Zhijian Ou
>
> **备注:** Accepted by IEEE TASLP
>
> **摘要:** Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Furthermore, we propose marginal likelihood scoring (MLS) decoding to align inference with the training objective and P2G augmentation to improve the robustness of P2G mapping. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline.
>
---
