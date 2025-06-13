# 音频 cs.SD;  eess.SP

- **最新发布 19 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] PAL: Probing Audio Encoders via LLMs -- A Study of Information Transfer from Audio Encoders to LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于多模态任务，旨在提升音频编码器到大语言模型的信息传递效率。通过优化架构设计，增强模型对音频表示的探测能力。**

- **链接: [http://arxiv.org/pdf/2506.10423v1](http://arxiv.org/pdf/2506.10423v1)**

> **作者:** Tony Alex; Wish Suharitdamrong; Sara Atito; Armin Mustafa; Philip J. B. Jackson; Imran Razzak; Muhammad Awais
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** The integration of audio perception capabilities into Large Language Models (LLMs) has enabled significant advances in Audio-LLMs. Although application-focused developments, particularly in curating training data for specific capabilities e.g., audio reasoning, have progressed rapidly, the underlying mechanisms that govern efficient transfer of rich semantic representations from audio encoders to LLMs remain under-explored. We conceptualize effective audio-LLM interaction as the LLM's ability to proficiently probe the audio encoder representations to satisfy textual queries. This paper presents a systematic investigation on how architectural design choices can affect that. Beginning with a standard Pengi/LLaVA-style audio-LLM architecture, we propose and evaluate several modifications guided by hypotheses derived from mechanistic interpretability studies and LLM operational principles. Our experiments demonstrate that: (1) delaying audio integration until the LLM's initial layers establish textual context that enhances its ability to probe the audio representations for relevant information; (2) the LLM can proficiently probe audio representations exclusively through LLM layer's attention submodule, without requiring propagation to its Feed-Forward Network (FFN) submodule; (3) an efficiently integrated ensemble of diverse audio encoders provides richer, complementary representations, thereby broadening the LLM's capacity to probe a wider spectrum of audio information. All hypotheses are evaluated using an identical three-stage training curriculum on a dataset of 5.6 million audio-text pairs, ensuring controlled comparisons. Our final architecture, which incorporates all proposed modifications, achieves relative improvements from 10\% to 60\% over the baseline, validating our approach to optimizing cross-modal information transfer in audio-LLMs. Project page: https://ta012.github.io/PAL/
>
---
#### [new 002] Discrete Audio Tokens: More Than a Survey!
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决离散音频标记的系统性评估问题。通过分类、基准测试和分析，探讨不同标记方法的性能与挑战。**

- **链接: [http://arxiv.org/pdf/2506.10274v1](http://arxiv.org/pdf/2506.10274v1)**

> **作者:** Pooneh Mousavi; Gallil Maimon; Adel Moumen; Darius Petermann; Jiatong Shi; Haibin Wu; Haici Yang; Anastasia Kuznetsova; Artem Ploujnikov; Ricard Marxer; Bhuvana Ramabhadran; Benjamin Elizalde; Loren Lugosch; Jinyu Li; Cem Subakan; Phil Woodland; Minje Kim; Hung-yi Lee; Shinji Watanabe; Yossi Adi; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream tasks.They provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: https://poonehmousavi.github.io/dates-website/.
>
---
#### [new 003] Fine-Grained control over Music Generation with Activation Steering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在实现对音乐生成的细粒度控制。通过干预模型激活，解决风格、音色和流派融合问题。**

- **链接: [http://arxiv.org/pdf/2506.10225v1](http://arxiv.org/pdf/2506.10225v1)**

> **作者:** Dipanshu Panda; Jayden Koshy Joe; Harshith M R; Swathi Narashiman; Pranay Mathur; Anish Veerakumar; Aniruddh Krishna; Keerthiharan A
>
> **摘要:** We present a method for fine-grained control over music generation through inference-time interventions on an autoregressive generative music transformer called MusicGen. Our approach enables timbre transfer, style transfer, and genre fusion by steering the residual stream using weights of linear probes trained on it, or by steering the attention layer activations in a similar manner. We observe that modelling this as a regression task provides improved performance, hypothesizing that the mean-squared-error better preserve meaningful directional information in the activation space. Combined with the global conditioning offered by text prompts in MusicGen, our method provides both global and local control over music generation. Audio samples illustrating our method are available at our demo page.
>
---
#### [new 004] Description and Discussion on DCASE 2025 Challenge Task 2: First-shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍DCASE 2025挑战任务2，旨在解决机器状态监测中的首次无监督异常声音检测问题，通过域泛化框架实现快速部署。**

- **链接: [http://arxiv.org/pdf/2506.10097v1](http://arxiv.org/pdf/2506.10097v1)**

> **作者:** Tomoya Nishida; Noboru Harada; Daisuke Niizumi; Davide Albertini; Roberto Sannino; Simone Pradolini; Filippo Augusti; Keisuke Imoto; Kota Dohi; Harsh Purohit; Takashi Endo; Yohei Kawaguchi
>
> **备注:** this article draws heavily from arXiv:2406.07250v1
>
> **摘要:** This paper introduces the task description for the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge Task 2, titled "First-shot unsupervised anomalous sound detection (ASD) for machine condition monitoring." Building on the DCASE 2024 Challenge Task 2, this task is structured as a first-shot problem within a domain generalization framework. The primary objective of the first-shot approach is to facilitate the rapid deployment of ASD systems for new machine types without requiring machine-specific hyperparameter tunings. For DCASE 2025 Challenge Task 2, sounds from previously unseen machine types have been collected and provided as the evaluation dataset. Results and analysis of the challenge submissions will be added following the challenge's submission deadline.
>
---
#### [new 005] Description and Discussion on DCASE 2025 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于DCASE 2025挑战任务4，聚焦空间语义分割声音场景，旨在从多通道信号中分离声事件并提取空间信息。**

- **链接: [http://arxiv.org/pdf/2506.10676v1](http://arxiv.org/pdf/2506.10676v1)**

> **作者:** Masahiro Yasuda; Binh Thien Nguyen; Noboru Harada; Romain Serizel; Mayank Mishra; Marc Delcroix; Shoko Araki; Daiki Takeuchi; Daisuke Niizumi; Yasunori Ohishi; Tomohiro Nakatani; Takao Kawamura; Nobutaka Ono
>
> **摘要:** Spatial Semantic Segmentation of Sound Scenes (S5) aims to enhance technologies for sound event detection and separation from multi-channel input signals that mix multiple sound events with spatial information. This is a fundamental basis of immersive communication. The ultimate goal is to separate sound event signals with 6 Degrees of Freedom (6DoF) information into dry sound object signals and metadata about the object type (sound event class) and representing spatial information, including direction. However, because several existing challenge tasks already provide some of the subset functions, this task for this year focuses on detecting and separating sound events from multi-channel spatial input signals. This paper outlines the S5 task setting of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge Task 4 and the DCASE2025 Task 4 Dataset, newly recorded and curated for this task. We also report experimental results for an S5 system trained and evaluated on this dataset. The full version of this paper will be published after the challenge results are made public.
>
---
#### [new 006] BNMusic: Blending Environmental Noises into Personalized Music
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决环境噪声干扰问题。通过将噪声融入个性化音乐中，提升听觉体验。**

- **链接: [http://arxiv.org/pdf/2506.10754v1](http://arxiv.org/pdf/2506.10754v1)**

> **作者:** Chi Zuo; Martin B. Møller; Pablo Martínez-Nuevo; Huayang Huang; Yu Wu; Ye Zhu
>
> **摘要:** While being disturbed by environmental noises, the acoustic masking technique is a conventional way to reduce the annoyance in audio engineering that seeks to cover up the noises with other dominant yet less intrusive sounds. However, misalignment between the dominant sound and the noise-such as mismatched downbeats-often requires an excessive volume increase to achieve effective masking. Motivated by recent advances in cross-modal generation, in this work, we introduce an alternative method to acoustic masking, aiming to reduce the noticeability of environmental noises by blending them into personalized music generated based on user-provided text prompts. Following the paradigm of music generation using mel-spectrogram representations, we propose a Blending Noises into Personalized Music (BNMusic) framework with two key stages. The first stage synthesizes a complete piece of music in a mel-spectrogram representation that encapsulates the musical essence of the noise. In the second stage, we adaptively amplify the generated music segment to further reduce noise perception and enhance the blending effectiveness, while preserving auditory quality. Our experiments with comprehensive evaluations on MusicBench, EPIC-SOUNDS, and ESC-50 demonstrate the effectiveness of our framework, highlighting the ability to blend environmental noise with rhythmically aligned, adaptively amplified, and enjoyable music segments, minimizing the noticeability of the noise, thereby improving overall acoustic experiences.
>
---
#### [new 007] Ground Reaction Force Estimation via Time-aware Knowledge Distillation
- **分类: eess.SP; cs.CV; cs.HC**

- **简介: 该论文属于人体步态分析任务，旨在解决可穿戴传感器测量地面反作用力（GRF）精度不足的问题。通过时间感知的知识蒸馏框架提升GRF估计性能。**

- **链接: [http://arxiv.org/pdf/2506.10265v1](http://arxiv.org/pdf/2506.10265v1)**

> **作者:** Eun Som Jeon; Sinjini Mitra; Jisoo Lee; Omik M. Save; Ankita Shukla; Hyunglae Lee; Pavan Turaga
>
> **摘要:** Human gait analysis with wearable sensors has been widely used in various applications, such as daily life healthcare, rehabilitation, physical therapy, and clinical diagnostics and monitoring. In particular, ground reaction force (GRF) provides critical information about how the body interacts with the ground during locomotion. Although instrumented treadmills have been widely used as the gold standard for measuring GRF during walking, their lack of portability and high cost make them impractical for many applications. As an alternative, low-cost, portable, wearable insole sensors have been utilized to measure GRF; however, these sensors are susceptible to noise and disturbance and are less accurate than treadmill measurements. To address these challenges, we propose a Time-aware Knowledge Distillation framework for GRF estimation from insole sensor data. This framework leverages similarity and temporal features within a mini-batch during the knowledge distillation process, effectively capturing the complementary relationships between features and the sequential properties of the target and input data. The performance of the lightweight models distilled through this framework was evaluated by comparing GRF estimations from insole sensor data against measurements from an instrumented treadmill. Empirical results demonstrated that Time-aware Knowledge Distillation outperforms current baselines in GRF estimation from wearable sensor data.
>
---
#### [new 008] FedMLAC: Mutual Learning Driven Heterogeneous Federated Audio Classification
- **分类: cs.SD; cs.DC; eess.AS**

- **简介: 该论文属于联邦音频分类任务，解决数据异构、模型异构和数据污染问题。提出FedMLAC框架，通过双模型结构和层剪枝聚合提升性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.10207v1](http://arxiv.org/pdf/2506.10207v1)**

> **作者:** Jun Bai; Rajib Rana; Di Wu; Youyang Qu; Xiaohui Tao; Ji Zhang
>
> **备注:** initial version
>
> **摘要:** Federated Learning (FL) provides a privacy-preserving paradigm for training audio classification (AC) models across distributed clients without sharing raw data. However, Federated Audio Classification (FedAC) faces three critical challenges that substantially hinder performance: data heterogeneity, model heterogeneity, and data poisoning. While prior works have attempted to address these issues, they are typically treated independently, lacking a unified and robust solution suited to real-world federated audio scenarios. To bridge this gap, we propose FedMLAC, a unified mutual learning framework designed to simultaneously tackle these challenges in FedAC. Specifically, FedMLAC introduces a dual-model architecture on each client, comprising a personalized local AC model and a lightweight, globally shared Plug-in model. Through bidirectional knowledge distillation, the Plug-in model enables global knowledge transfer while adapting to client-specific data distributions, thus supporting both generalization and personalization. To further enhance robustness against corrupted audio data, we develop a Layer-wise Pruning Aggregation (LPA) strategy that filters unreliable Plug-in model updates based on parameter deviations during server-side aggregation. Extensive experiments on four diverse audio classification benchmarks, spanning both speech and non-speech tasks, demonstrate that FedMLAC consistently outperforms existing state-of-the-art methods in terms of classification accuracy and robustness to noisy data.
>
---
#### [new 009] Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音翻译任务，解决LLMs从文本到语音的模态适应问题。通过交替使用语音和文本单元进行训练，逐步减少文本比例，提升翻译性能。**

- **链接: [http://arxiv.org/pdf/2506.10299v1](http://arxiv.org/pdf/2506.10299v1)**

> **作者:** Hayato Futami; Emiru Tsunoo; Yosuke Kashiwagi; Yuki Ito; Hassan Shahmohammadi; Siddhant Arora; Shinji Watanabe
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** Speech-to-speech translation (S2ST) has been advanced with large language models (LLMs), which are fine-tuned on discrete speech units. In such approaches, modality adaptation from text to speech has been an issue. LLMs are trained on text-only data, which presents challenges to adapt them to speech modality with limited speech-to-speech data. To address the training difficulty, we propose scheduled interleaved speech--text training in this study. We use interleaved speech--text units instead of speech units during training, where aligned text tokens are interleaved at the word level. We gradually decrease the ratio of text as training progresses, to facilitate progressive modality adaptation from text to speech. We conduct experimental evaluations by fine-tuning LLaMA3.2-1B for S2ST on the CVSS dataset. We show that the proposed method consistently improves the translation performances, especially for languages with limited training data.
>
---
#### [new 010] Unsupervised Deformable Image Registration with Structural Nonparametric Smoothing
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 该论文属于图像配准任务，解决无监督方法在处理稀疏特征和大位移时的挑战。提出SmoothProper模块，提升配准精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.10813v1](http://arxiv.org/pdf/2506.10813v1)**

> **作者:** Hang Zhang; Xiang Chen; Renjiu Hu; Rongguang Wang; Jinwei Zhang; Min Liu; Yaonan Wang; Gaolei Li; Xinxing Cheng; Jinming Duan
>
> **备注:** Accepted for publication at Information Processing in Medical Imaging (IPMI) 2025
>
> **摘要:** Learning-based deformable image registration (DIR) accelerates alignment by amortizing traditional optimization via neural networks. Label supervision further enhances accuracy, enabling efficient and precise nonlinear alignment of unseen scans. However, images with sparse features amid large smooth regions, such as retinal vessels, introduce aperture and large-displacement challenges that unsupervised DIR methods struggle to address. This limitation occurs because neural networks predict deformation fields in a single forward pass, leaving fields unconstrained post-training and shifting the regularization burden entirely to network weights. To address these issues, we introduce SmoothProper, a plug-and-play neural module enforcing smoothness and promoting message passing within the network's forward pass. By integrating a duality-based optimization layer with tailored interaction terms, SmoothProper efficiently propagates flow signals across spatial locations, enforces smoothness, and preserves structural consistency. It is model-agnostic, seamlessly integrates into existing registration frameworks with minimal parameter overhead, and eliminates regularizer hyperparameter tuning. Preliminary results on a retinal vessel dataset exhibiting aperture and large-displacement challenges demonstrate our method reduces registration error to 1.88 pixels on 2912x2912 images, marking the first unsupervised DIR approach to effectively address both challenges. The source code will be available at https://github.com/tinymilky/SmoothProper.
>
---
#### [new 011] DanceChat: Large Language Model-Guided Music-to-Dance Generation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音乐到舞蹈生成任务，旨在解决音乐与舞蹈语义差距大、数据稀缺的问题。通过引入大语言模型生成舞蹈指令，提升生成舞蹈的多样性和风格匹配度。**

- **链接: [http://arxiv.org/pdf/2506.10574v1](http://arxiv.org/pdf/2506.10574v1)**

> **作者:** Qing Wang; Xiaohang Yang; Yilan Dong; Naveen Raj Govindaraj; Gregory Slabaugh; Shanxin Yuan
>
> **备注:** check demos at https://dancechat.github.io/anon/
>
> **摘要:** Music-to-dance generation aims to synthesize human dance motion conditioned on musical input. Despite recent progress, significant challenges remain due to the semantic gap between music and dance motion, as music offers only abstract cues, such as melody, groove, and emotion, without explicitly specifying the physical movements. Moreover, a single piece of music can produce multiple plausible dance interpretations. This one-to-many mapping demands additional guidance, as music alone provides limited information for generating diverse dance movements. The challenge is further amplified by the scarcity of paired music and dance data, which restricts the model\^a\u{A}\'Zs ability to learn diverse dance patterns. In this paper, we introduce DanceChat, a Large Language Model (LLM)-guided music-to-dance generation approach. We use an LLM as a choreographer that provides textual motion instructions, offering explicit, high-level guidance for dance generation. This approach goes beyond implicit learning from music alone, enabling the model to generate dance that is both more diverse and better aligned with musical styles. Our approach consists of three components: (1) an LLM-based pseudo instruction generation module that produces textual dance guidance based on music style and structure, (2) a multi-modal feature extraction and fusion module that integrates music, rhythm, and textual guidance into a shared representation, and (3) a diffusion-based motion synthesis module together with a multi-modal alignment loss, which ensures that the generated dance is aligned with both musical and textual cues. Extensive experiments on AIST++ and human evaluations show that DanceChat outperforms state-of-the-art methods both qualitatively and quantitatively.
>
---
#### [new 012] Leveraging Pre-Trained Models for Multimodal Class-Incremental Learning under Adaptive Fusion
- **分类: cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态类增量学习任务，解决跨模态信息融合与灾难性遗忘问题，提出MIFE、AAVFM和对比损失等方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.09999v1](http://arxiv.org/pdf/2506.09999v1)**

> **作者:** Yukun Chen; Zihuan Qiu; Fanman Meng; Hongliang Li; Linfeng Xu; Qingbo Wu
>
> **摘要:** Unlike traditional Multimodal Class-Incremental Learning (MCIL) methods that focus only on vision and text, this paper explores MCIL across vision, audio and text modalities, addressing challenges in integrating complementary information and mitigating catastrophic forgetting. To tackle these issues, we propose an MCIL method based on multimodal pre-trained models. Firstly, a Multimodal Incremental Feature Extractor (MIFE) based on Mixture-of-Experts (MoE) structure is introduced to achieve effective incremental fine-tuning for AudioCLIP. Secondly, to enhance feature discriminability and generalization, we propose an Adaptive Audio-Visual Fusion Module (AAVFM) that includes a masking threshold mechanism and a dynamic feature fusion mechanism, along with a strategy to enhance text diversity. Thirdly, a novel multimodal class-incremental contrastive training loss is proposed to optimize cross-modal alignment in MCIL. Finally, two MCIL-specific evaluation metrics are introduced for comprehensive assessment. Extensive experiments on three multimodal datasets validate the effectiveness of our method.
>
---
#### [new 013] Multimodal Emotion Coupling via Speech-to-Facial and Bodily Gestures in Dyadic Interaction
- **分类: cs.MM; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于情感计算任务，旨在研究对话中语音与面部、手势的多模态情感耦合，解决情感信号同步与表达问题。通过分析IEMOCAP数据，提取语音特征并映射到面部和手势运动，揭示情感表达的动态规律。**

- **链接: [http://arxiv.org/pdf/2506.10010v1](http://arxiv.org/pdf/2506.10010v1)**

> **作者:** Von Ralph Dane Marquez Herbuela; Yukie Nagai
>
> **摘要:** Human emotional expression emerges through coordinated vocal, facial, and gestural signals. While speech face alignment is well established, the broader dynamics linking emotionally expressive speech to regional facial and hand motion remains critical for gaining a deeper insight into how emotional and behavior cues are communicated in real interactions. Further modulating the coordination is the structure of conversational exchange like sequential turn taking, which creates stable temporal windows for multimodal synchrony, and simultaneous speech, often indicative of high arousal moments, disrupts this alignment and impacts emotional clarity. Understanding these dynamics enhances realtime emotion detection by improving the accuracy of timing and synchrony across modalities in both human interactions and AI systems. This study examines multimodal emotion coupling using region specific motion capture from dyadic interactions in the IEMOCAP corpus. Speech features included low level prosody, MFCCs, and model derived arousal, valence, and categorical emotions (Happy, Sad, Angry, Neutral), aligned with 3D facial and hand marker displacements. Expressive activeness was quantified through framewise displacement magnitudes, and speech to gesture prediction mapped speech features to facial and hand movements. Nonoverlapping speech consistently elicited greater activeness particularly in the lower face and mouth. Sadness showed increased expressivity during nonoverlap, while anger suppressed gestures during overlaps. Predictive mapping revealed highest accuracy for prosody and MFCCs in articulatory regions while arousal and valence had lower and more context sensitive correlations. Notably, hand speech synchrony was enhanced under low arousal and overlapping speech, but not for valence.
>
---
#### [new 014] The 2025 PNPL Competition: Speech Detection and Phoneme Classification in the LibriBrain Dataset
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文介绍2025年PNPL竞赛，聚焦非侵入性脑信号中的语音检测与音素分类任务，旨在推动无创脑机接口发展。**

- **链接: [http://arxiv.org/pdf/2506.10165v1](http://arxiv.org/pdf/2506.10165v1)**

> **作者:** Gilad Landau; Miran Özdogan; Gereon Elvers; Francesco Mantegna; Pratik Somaiya; Dulhan Jayalath; Luisa Kurth; Teyun Kwon; Brendan Shillingford; Greg Farquhar; Minqi Jiang; Karim Jerbi; Hamza Abdelhedi; Yorguin Mantilla Ramos; Caglar Gulcehre; Mark Woolrich; Natalie Voets; Oiwi Parker Jones
>
> **摘要:** The advance of speech decoding from non-invasive brain data holds the potential for profound societal impact. Among its most promising applications is the restoration of communication to paralysed individuals affected by speech deficits such as dysarthria, without the need for high-risk surgical interventions. The ultimate aim of the 2025 PNPL competition is to produce the conditions for an "ImageNet moment" or breakthrough in non-invasive neural decoding, by harnessing the collective power of the machine learning community. To facilitate this vision we present the largest within-subject MEG dataset recorded to date (LibriBrain) together with a user-friendly Python library (pnpl) for easy data access and integration with deep learning frameworks. For the competition we define two foundational tasks (i.e. Speech Detection and Phoneme Classification from brain data), complete with standardised data splits and evaluation metrics, illustrative benchmark models, online tutorial code, a community discussion board, and public leaderboard for submissions. To promote accessibility and participation the competition features a Standard track that emphasises algorithmic innovation, as well as an Extended track that is expected to reward larger-scale computing, accelerating progress toward a non-invasive brain-computer interface for speech.
>
---
#### [new 015] AC/DC: LLM-based Audio Comprehension via Dialogue Continuation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于音频理解任务，旨在解决指令跟随中的标题变异问题。通过对话延续训练，使模型在无多任务调优下实现零样本指令跟随。**

- **链接: [http://arxiv.org/pdf/2506.10312v1](http://arxiv.org/pdf/2506.10312v1)**

> **作者:** Yusuke Fujita; Tomoya Mizumoto; Atsushi Kojima; Lianbo Liu; Yui Sudo
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We propose an instruction-following audio comprehension model that leverages the dialogue continuation ability of large language models (LLMs). Instead of directly generating target captions in training data, the proposed method trains a model to produce responses as if the input caption triggered a dialogue. This dialogue continuation training mitigates the caption variation problem. Learning to continue a dialogue effectively captures the caption's meaning beyond its surface-level words. As a result, our model enables zero-shot instruction-following capability without multitask instruction tuning, even trained solely on audio captioning datasets. Experiments on AudioCaps, WavCaps, and Clotho datasets with AudioBench audio-scene question-answering tests demonstrate our model's ability to follow various unseen instructions.
>
---
#### [new 016] Joint ASR and Speaker Role Tagging with Serialized Output Training
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别与说话人角色标注联合任务，旨在解决传统ASR系统不识别说话人角色的问题。通过SOT方法提升模型生成带角色信息的转录结果。**

- **链接: [http://arxiv.org/pdf/2506.10349v1](http://arxiv.org/pdf/2506.10349v1)**

> **作者:** Anfeng Xu; Tiantian Feng; Shrikanth Narayanan
>
> **备注:** Under review
>
> **摘要:** Automatic Speech Recognition systems have made significant progress with large-scale pre-trained models. However, most current systems focus solely on transcribing the speech without identifying speaker roles, a function that is critical for conversational AI. In this work, we investigate the use of serialized output training (SOT) for joint ASR and speaker role tagging. By augmenting Whisper with role-specific tokens and fine-tuning it with SOT, we enable the model to generate role-aware transcriptions in a single decoding pass. We compare the SOT approach against a self-supervised previous baseline method on two real-world conversational datasets. Our findings show that this approach achieves more than 10% reduction in multi-talker WER, demonstrating its feasibility as a unified model for speaker-role aware speech transcription.
>
---
#### [new 017] Disentangling Dual-Encoder Masked Autoencoder for Respiratory Sound Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于呼吸音分类任务，旨在解决数据稀缺和领域不匹配问题。提出DDE-MAE模型，通过双编码器分离疾病相关与无关特征，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.10698v1](http://arxiv.org/pdf/2506.10698v1)**

> **作者:** Peidong Wei Shiyu Miao Lin Li
>
> **备注:** (Accepted at Interspeech 2025)
>
> **摘要:** Deep neural networks have been applied to audio spectrograms for respiratory sound classification, but it remains challenging to achieve satisfactory performance due to the scarcity of available data. Moreover, domain mismatch may be introduced into the trained models as a result of the respiratory sound samples being collected from various electronic stethoscopes, patient demographics, and recording environments. To tackle this issue, we proposed a modified MaskedAutoencoder(MAE) model, named Disentangling Dual-Encoder MAE (DDE-MAE) for respiratory sound classification. Two independent encoders were designed to capture disease-related and disease-irrelevant information separately, achieving feature disentanglement to reduce the domain mismatch. Our method achieves a competitive performance on the ICBHI dataset.
>
---
#### [new 018] WDMIR: Wavelet-Driven Multimodal Intent Recognition
- **分类: cs.MM; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于多模态意图识别任务，旨在提升对非语言信息的语义理解。通过小波驱动的融合模块和跨模态交互机制，增强情感线索的识别效果。**

- **链接: [http://arxiv.org/pdf/2506.10011v1](http://arxiv.org/pdf/2506.10011v1)**

> **作者:** Weiyin Gong; Kai Zhang; Yanghai Zhang; Qi Liu; Xinjie Sun; Junyu Lu; Linbo Zhu
>
> **备注:** Accepted at IJCAI 2025, 9pages, 6figures
>
> **摘要:** Multimodal intent recognition (MIR) seeks to accurately interpret user intentions by integrating verbal and non-verbal information across video, audio and text modalities. While existing approaches prioritize text analysis, they often overlook the rich semantic content embedded in non-verbal cues. This paper presents a novel Wavelet-Driven Multimodal Intent Recognition(WDMIR) framework that enhances intent understanding through frequency-domain analysis of non-verbal information. To be more specific, we propose: (1) a wavelet-driven fusion module that performs synchronized decomposition and integration of video-audio features in the frequency domain, enabling fine-grained analysis of temporal dynamics; (2) a cross-modal interaction mechanism that facilitates progressive feature enhancement from bimodal to trimodal integration, effectively bridging the semantic gap between verbal and non-verbal information. Extensive experiments on MIntRec demonstrate that our approach achieves state-of-the-art performance, surpassing previous methods by 1.13% on accuracy. Ablation studies further verify that the wavelet-driven fusion module significantly improves the extraction of semantic information from non-verbal sources, with a 0.41% increase in recognition accuracy when analyzing subtle emotional cues.
>
---
#### [new 019] Can Sound Replace Vision in LLaVA With Token Substitution?
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态任务，旨在探索用声音替代视觉在LLaVA中的可行性。通过音频-视觉对齐提升检索性能，但牺牲了文本生成质量。**

- **链接: [http://arxiv.org/pdf/2506.10416v1](http://arxiv.org/pdf/2506.10416v1)**

> **作者:** Ali Vosoughi; Jing Bi; Pinxin Liu; Yunlong Tang; Chenliang Xu
>
> **备注:** 29 pages including references and appendices
>
> **摘要:** While multimodal systems have achieved impressive advances, they typically rely on text-aligned representations rather than directly integrating audio and visual inputs. This reliance can limit the use of acoustic information in tasks requiring nuanced audio understanding. In response, SoundCLIP explores direct audio-visual integration within multimodal large language models (MLLMs) by substituting CLIP's visual tokens with audio representations and selecting sound-relevant patch tokens in models such as LLaVA. We investigate two configurations: (1) projecting audio features into CLIP's visual manifold via a multilayer perceptron trained with InfoNCE on paired audio-video segments, and (2) preserving raw audio embeddings with minimal dimensional adjustments. Experiments with five state-of-the-art audio encoders reveal a fundamental trade-off. While audio-to-video retrieval performance increases dramatically (up to 44 percentage points in Top-1 accuracy) when audio is projected into CLIP's space, text generation quality declines. Encoders pre-trained with text supervision (CLAP, Whisper, ImageBind) maintain stronger generative capabilities than those focused primarily on audiovisual alignment (Wav2CLIP, AudioCLIP), highlighting the value of language exposure for generation tasks. We introduce WhisperCLIP, an architecture that fuses intermediate representations from Whisper, as well as AudioVisual Event Evaluation (AVE-2), a dataset of 580,147 three-second audiovisual clips with fine-grained alignment annotations. Our findings challenge the assumption that stronger cross-modal alignment necessarily benefits all multimodal tasks; instead, a Pareto frontier emerges wherein optimal performance depends on balancing retrieval accuracy with text generation quality. Codes and datasets: https://github.com/ali-vosoughi/SoundCLIP.
>
---
## 更新

#### [replaced 001] Towards a Unified Benchmark for Arabic Pronunciation Assessment: Quranic Recitation as Case Study
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07722v2](http://arxiv.org/pdf/2506.07722v2)**

> **作者:** Yassine El Kheir; Omnia Ibrahim; Amit Meghanani; Nada Almarwani; Hawau Olamide Toyin; Sadeen Alharbi; Modar Alfadly; Lamya Alkanhal; Ibrahim Selim; Shehab Elbatal; Salima Mdhaffar; Thomas Hain; Yasser Hifny; Mostafa Shahin; Ahmed Ali
>
> **备注:** Accepted Interspeech 2025 and ArabicNLP Shared Task 2025
>
> **摘要:** We present a unified benchmark for mispronunciation detection in Modern Standard Arabic (MSA) using Qur'anic recitation as a case study. Our approach lays the groundwork for advancing Arabic pronunciation assessment by providing a comprehensive pipeline that spans data processing, the development of a specialized phoneme set tailored to the nuances of MSA pronunciation, and the creation of the first publicly available test set for this task, which we term as the Qur'anic Mispronunciation Benchmark (QuranMB.v1). Furthermore, we evaluate several baseline models to provide initial performance insights, thereby highlighting both the promise and the challenges inherent in assessing MSA pronunciation. By establishing this standardized framework, we aim to foster further research and development in pronunciation assessment in Arabic language technology and related applications.
>
---
#### [replaced 002] Exploring Performance-Complexity Trade-Offs in Sound Event Detection Models
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11373v2](http://arxiv.org/pdf/2503.11373v2)**

> **作者:** Tobias Morocutti; Florian Schmid; Jonathan Greif; Francesco Foscarin; Gerhard Widmer
>
> **备注:** In Proceedings of the 33rd European Signal Processing Conference (EUSIPCO 2025), Palermo, Italy
>
> **摘要:** We target the problem of developing new low-complexity networks for the sound event detection task. Our goal is to meticulously analyze the performance-complexity trade-off, aiming to be competitive with the large state-of-the-art models, at a fraction of the computational requirements. We find that low-complexity convolutional models previously proposed for audio tagging can be effectively adapted for event detection (which requires frame-wise prediction) by adjusting convolutional strides, removing the global pooling, and, importantly, adding a sequence model before the (now frame-wise) classification heads. Systematic experiments reveal that the best choice for the sequence model type depends on which complexity metric is most important for the given application. We also investigate the impact of enhanced training strategies such as knowledge distillation. In the end, we show that combined with an optimized training strategy, we can reach event detection performance comparable to state-of-the-art transformers while requiring only around 5% of the parameters. We release all our pre-trained models and the code for reproducing this work to support future research in low-complexity sound event detection at https://github.com/theMoro/EfficientSED.
>
---
#### [replaced 003] Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.13017v3](http://arxiv.org/pdf/2505.13017v3)**

> **作者:** Dang Thoai Phan; Tuan Anh Huynh; Van Tuan Pham; Cao Minh Tran; Van Thuan Mai; Ngoc Quy Tran
>
> **摘要:** The Continuous Wavelet Transform (CWT) is an effective tool for feature extraction in acoustic recognition using Convolutional Neural Networks (CNNs), particularly when applied to non-stationary audio. However, its high computational cost poses a significant challenge, often leading researchers to prefer alternative methods such as the Short-Time Fourier Transform (STFT). To address this issue, this paper proposes a method to reduce the computational complexity of CWT by optimizing the length of the wavelet kernel and the hop size of the output scalogram. Experimental results demonstrate that the proposed approach significantly reduces computational cost while maintaining the robust performance of the trained model in acoustic recognition tasks.
>
---
