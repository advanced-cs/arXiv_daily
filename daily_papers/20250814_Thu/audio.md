# 音频 cs.SD;  eess.SP

- **最新发布 9 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Analysis of Domain Shift across ASR Architectures via TTS-Enabled Separation of Target Domain and Acoustic Conditions
- **分类: cs.SD**

- **简介: 论文分析不同ASR架构在领域转移下的性能，通过TTS合成目标域数据分离领域与声学条件，比较模块化与seq2seq架构，揭示特定建模选择对泛化能力的影响。**

- **链接: [http://arxiv.org/pdf/2508.09868v1](http://arxiv.org/pdf/2508.09868v1)**

> **作者:** Tina Raissi; Nick Rossenbach; Ralf Schlüter
>
> **备注:** Accepted for presentation at IEEE ASRU 2025
>
> **摘要:** We analyze automatic speech recognition (ASR) modeling choices under domain mismatch, comparing classic modular and novel sequence-to-sequence (seq2seq) architectures. Across the different ASR architectures, we examine a spectrum of modeling choices, including label units, context length, and topology. To isolate language domain effects from acoustic variation, we synthesize target domain audio using a text-to-speech system trained on LibriSpeech. We incorporate target domain n-gram and neural language models for domain adaptation without retraining the acoustic model. To our knowledge, this is the first controlled comparison of optimized ASR systems across state-of-the-art architectures under domain shift, offering insights into their generalization. The results show that, under domain shift, rather than the decoder architecture choice or the distinction between classic modular and novel seq2seq models, it is specific modeling choices that influence performance.
>
---
#### [new 002] BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model
- **分类: cs.SD**

- **简介: 论文提出BeatFM，通过预训练音乐基础模型及多维度语义聚合模块，解决beat跟踪中数据稀缺导致的泛化差和节奏结构捕捉难题，实现跨风格准确追踪。**

- **链接: [http://arxiv.org/pdf/2508.09790v1](http://arxiv.org/pdf/2508.09790v1)**

> **作者:** Ganghui Ru; Jieying Wang; Jiahao Zhao; Yulun Wu; Yi Yu; Nannan Jiang; Wei Wang; Wei Li
>
> **备注:** This paper has been accepted by ICME2025
>
> **摘要:** Beat tracking is a widely researched topic in music information retrieval. However, current beat tracking methods face challenges due to the scarcity of labeled data, which limits their ability to generalize across diverse musical styles and accurately capture complex rhythmic structures. To overcome these challenges, we propose a novel beat tracking paradigm BeatFM, which introduces a pre-trained music foundation model and leverages its rich semantic knowledge to improve beat tracking performance. Pre-training on diverse music datasets endows music foundation models with a robust understanding of music, thereby effectively addressing these challenges. To further adapt it for beat tracking, we design a plug-and-play multi-dimensional semantic aggregation module, which is composed of three parallel sub-modules, each focusing on semantic aggregation in the temporal, frequency, and channel domains, respectively. Extensive experiments demonstrate that our method achieves state-of-the-art performance in beat and downbeat tracking across multiple benchmark datasets.
>
---
#### [new 003] HingeNet: A Harmonic-Aware Fine-Tuning Approach for Beat Tracking
- **分类: cs.SD; eess.AS**

- **简介: 论文提出HingeNet，通过轻量级、可分离架构与谐波感知机制解决节拍跟踪中标注数据稀缺的问题，实现高效参数微调。**

- **链接: [http://arxiv.org/pdf/2508.09788v1](http://arxiv.org/pdf/2508.09788v1)**

> **作者:** Ganghui Ru; Jieying Wang; Jiahao Zhao; Yulun Wu; Yi Yu; Nannan Jiang; Wei Wang; Wei Li
>
> **备注:** This paper has been accepted by ICME2025
>
> **摘要:** Fine-tuning pre-trained foundation models has made significant progress in music information retrieval. However, applying these models to beat tracking tasks remains unexplored as the limited annotated data renders conventional fine-tuning methods ineffective. To address this challenge, we propose HingeNet, a novel and general parameter-efficient fine-tuning method specifically designed for beat tracking tasks. HingeNet is a lightweight and separable network, visually resembling a hinge, designed to tightly interface with pre-trained foundation models by using their intermediate feature representations as input. This unique architecture grants HingeNet broad generalizability, enabling effective integration with various pre-trained foundation models. Furthermore, considering the significance of harmonics in beat tracking, we introduce harmonic-aware mechanism during the fine-tuning process to better capture and emphasize the harmonic structures in musical signals. Experiments on benchmark datasets demonstrate that HingeNet achieves state-of-the-art performance in beat and downbeat tracking
>
---
#### [new 004] OSUM-EChat: Enhancing End-to-End Empathetic Spoken Chatbot via Understanding-Driven Spoken Dialogue
- **分类: cs.SD**

- **简介: 该论文提出OSUM-EChat系统，旨在解决端到端对话系统在情感理解与资源受限环境下的挑战，通过双重视角机制整合语音与语义信息，提升情感回应能力，并构建EChat-200K数据集与EChat-eval基准，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.09600v1](http://arxiv.org/pdf/2508.09600v1)**

> **作者:** Xuelong Geng; Qijie Shao; Hongfei Xue; Shuiyuan Wang; Hanke Xie; Zhao Guo; Yi Zhao; Guojian Li; Wenjie Tian; Chengyou Wang; Zhixian Zhao; Kangxiang Xia; Ziyu Zhang; Zhennan Lin; Tianlun Zuo; Mingchen Shao; Yuang Cao; Guobin Ma; Longhao Li; Yuhang Dai; Dehui Gao; Dake Guo; Lei Xie
>
> **摘要:** Empathy is crucial in enabling natural interactions within spoken dialogue systems, allowing machines to recognize and respond appropriately to paralinguistic cues such as age, gender, and emotion. Recent advancements in end-to-end speech language models, which unify speech understanding and generation, provide promising solutions. However, several challenges persist, including an over-reliance on large-scale dialogue datasets, insufficient extraction of paralinguistic cues vital for conveying empathy, and the lack of empathy-specific datasets and evaluation frameworks. To address these issues, we introduce OSUM-EChat, an open-source, end-to-end spoken dialogue system designed to enhance empathetic interactions, particularly in resource-limited settings. OSUM-EChat introduces two key innovations: (1) a three-stage understanding-driven spoken dialogue training strategy that extends the capabilities of a large speech understanding model to spoken dialogue tasks, and (2) a linguistic-paralinguistic dual thinking mechanism that integrates paralinguistic understanding through a chain of thought with dialogue generation, enabling the system to produce more empathetic responses. This approach reduces reliance on large-scale dialogue datasets while maintaining high-quality empathetic interactions. Additionally, we introduce the EChat-200K dataset, a rich corpus of empathetic speech-to-speech dialogues, and the EChat-eval benchmark, a comprehensive framework for evaluating the empathetic capabilities of dialogue systems. Experimental results demonstrate that OSUM-EChat outperforms end-to-end spoken dialogue models regarding empathetic responsiveness, validating its effectiveness.
>
---
#### [new 005] MetaGuardian: Enhancing Voice Assistant Security through Advanced Acoustic Metamaterials
- **分类: cs.SD**

- **简介: 论文提出基于声学metamaterials的MetaGuardian系统，解决语音助手对抗攻击问题，通过扩展过滤范围和结构设计实现高效防护，兼顾便携与有效性。**

- **链接: [http://arxiv.org/pdf/2508.09728v1](http://arxiv.org/pdf/2508.09728v1)**

> **作者:** Zhiyuan Ning; Zheng Wang; Zhanyong Tang
>
> **摘要:** We present MetaGuardian, a voice assistant (VA) protection system based on acoustic metamaterials. MetaGuardian can be directly integrated into the enclosures of various smart devices, effectively defending against inaudible, adversarial and laser attacks without relying on additional software support or altering the underlying hardware, ensuring usability. To achieve this, MetaGuardian leverages the mutual impedance effects between metamaterial units to extend the signal filtering range to 16-40 kHz to effectively block wide-band inaudible attacks. Additionally, it adopts a carefully designed coiled space structure to precisely interfere with adversarial attacks while ensuring the normal functioning of VAs. Furthermore, MetaGuardian offers a universal structural design, allowing itself to be flexibly adapted to various smart devices, striking a balance between portability and protection effectiveness. In controled evaluation environments, MetaGuardian achieves a high defense success rate against various attack types, including adversarial, inaudible and laser attacks.
>
---
#### [new 006] A Comparative Analysis on ASR System Combination for Attention, CTC, Factored Hybrid, and Transducer Models
- **分类: cs.SD**

- **简介: 本文比较不同ASR模型组合方法（注意力、CTC、因子混合、传输器），通过两步法优化搜索空间，提升识别性能，评估架构与标签拓扑影响，应用于Librispeech 960h任务。**

- **链接: [http://arxiv.org/pdf/2508.09880v1](http://arxiv.org/pdf/2508.09880v1)**

> **作者:** Noureldin Bayoumi; Robin Schmitt; Tina Raissi; Albert Zeyer; Ralf Schlüter; Hermann Ney
>
> **备注:** Accepted for presentation at IEEE Speech Communication; 16th ITG Conference
>
> **摘要:** Combination approaches for speech recognition (ASR) systems cover structured sentence-level or word-based merging techniques as well as combination of model scores during beam search. In this work, we compare model combination across popular ASR architectures. Our method leverages the complementary strengths of different models in exploring diverse portions of the search space. We rescore a joint hypothesis list of two model candidates. We then identify the best hypothesis through log-linear combination of these sequence-level scores. While model combination during first-pass recognition may yield improved performance, it introduces variability due to differing decoding methods, making direct comparison more challenging. Our two-pass method ensures consistent comparisons across all system combination results presented in this study. We evaluate model pair candidates with varying architectures and label topologies and units. Experimental results are provided for the Librispeech 960h task.
>
---
#### [new 007] ProMode: A Speech Prosody Model Conditioned on Acoustic and Textual Inputs
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 论文提出ProMode模型，结合音频和文本输入建模语音语调，提升F0/能量预测性能，并应用于TTS系统，表现优于基线。**

- **链接: [http://arxiv.org/pdf/2508.09389v1](http://arxiv.org/pdf/2508.09389v1)**

> **作者:** Eray Eren; Qingju Liu; Hyeongwoo Kim; Pablo Garrido; Abeer Alwan
>
> **备注:** Interspeech 2025; demo page at https://promode8272.github.io/promode/index.html
>
> **摘要:** Prosody conveys rich emotional and semantic information of the speech signal as well as individual idiosyncrasies. We propose a stand-alone model that maps text-to-prosodic features such as F0 and energy and can be used in downstream tasks such as TTS. The ProMode encoder takes as input acoustic features and time-aligned textual content, both are partially masked, and obtains a fixed-length latent prosodic embedding. The decoder predicts acoustics in the masked region using both the encoded prosody input and unmasked textual content. Trained on the GigaSpeech dataset, we compare our method with state-of-the-art style encoders. For F0 and energy predictions, we show consistent improvements for our model at different levels of granularity. We also integrate these predicted prosodic features into a TTS system and conduct perceptual tests, which show higher prosody preference compared to the baselines, demonstrating the model's potential in tasks where prosody modeling is important.
>
---
#### [new 008] Leveraging Zipformer Model for Effective Language Identification in Code-Switched Child-Directed Speech
- **分类: cs.CL; cs.SD**

- **简介: 论文提出使用Zipformer处理双语儿童定向语音中的语言识别，通过内层特征提取提升准确率，达到81.89%。**

- **链接: [http://arxiv.org/pdf/2508.09430v1](http://arxiv.org/pdf/2508.09430v1)**

> **作者:** Lavanya Shankar; Leibny Paola Garcia Perera
>
> **摘要:** Code-switching and language identification in child-directed scenarios present significant challenges, particularly in bilingual environments. This paper addresses this challenge by using Zipformer to handle the nuances of speech, which contains two imbalanced languages, Mandarin and English, in an utterance. This work demonstrates that the internal layers of the Zipformer effectively encode the language characteristics, which can be leveraged in language identification. We present the selection methodology of the inner layers to extract the embeddings and make a comparison with different back-ends. Our analysis shows that Zipformer is robust across these backends. Our approach effectively handles imbalanced data, achieving a Balanced Accuracy (BAC) of 81.89%, a 15.47% improvement over the language identification baseline. These findings highlight the potential of the transformer encoder architecture model in real scenarios.
>
---
#### [new 009] $\text{M}^3\text{PDB}$: A Multimodal, Multi-Label, Multilingual Prompt Database for Speech Generation
- **分类: eess.AS; cs.SD**

- **简介: 论文提出M³PDB数据库，解决语音生成中高质量提示缺失问题，通过多模态多标签多语言标注与轻量级策略提升实时场景下的生成效果。**

- **链接: [http://arxiv.org/pdf/2508.09702v1](http://arxiv.org/pdf/2508.09702v1)**

> **作者:** Boyu Zhu; Cheng Gong; Muyang Wu; Ruihao Jing; Fan Liu; Xiaolei Zhang; Chi Zhang; Xuelong Li
>
> **摘要:** Recent advancements in zero-shot speech generation have enabled models to synthesize speech that mimics speaker identity and speaking style from speech prompts. However, these models' effectiveness is significantly limited in real-world scenarios where high-quality speech prompts are absent, incomplete, or out of domain. This issue arises primarily from a significant quality mismatch between the speech data utilized for model training and the input prompt speech during inference. To address this, we introduce $\text{M}^3\text{PDB}$, the first large-scale, multi-modal, multi-label, and multilingual prompt database designed for robust prompt selection in speech generation. Our dataset construction leverages a novel multi-modal, multi-agent annotation framework, enabling precise and hierarchical labeling across diverse modalities. Furthermore, we propose a lightweight yet effective prompt selection strategy tailored for real-time, resource-constrained inference settings. Experimental results demonstrate that our proposed database and selection strategy effectively support various challenging speech generation scenarios. We hope our work can inspire the community to shift focus from improving performance on standard benchmarks to addressing more realistic and diverse application scenarios in speech generation. Code and dataset are available at: https://github.com/hizening/M3PDB.
>
---
## 更新

#### [replaced 001] A2SB: Audio-to-Audio Schrodinger Bridges
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.11311v2](http://arxiv.org/pdf/2501.11311v2)**

> **作者:** Zhifeng Kong; Kevin J Shih; Weili Nie; Arash Vahdat; Sang-gil Lee; Joao Felipe Santos; Ante Jukic; Rafael Valle; Bryan Catanzaro
>
> **摘要:** Real-world audio is often degraded by numerous factors. This work presents an audio restoration model tailored for high-res music at 44.1kHz. Our model, Audio-to-Audio Schr\"odinger Bridges (A2SB), is capable of both bandwidth extension (predicting high-frequency components) and inpainting (re-generating missing segments). Critically, A2SB is end-to-end requiring no vocoder to predict waveform outputs, able to restore hour-long audio inputs, and trained on permissively licensed music data. A2SB is capable of achieving state-of-the-art band-width extension and inpainting quality on several out-of-distribution music test sets.
>
---
#### [replaced 002] Multi-Target Backdoor Attacks Against Speaker Recognition
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08559v2](http://arxiv.org/pdf/2508.08559v2)**

> **作者:** Alexandrine Fortier; Sonal Joshi; Thomas Thebaud; Jesus Villalba Lopez; Najim Dehak; Patrick Cardinal
>
> **备注:** Accepted to IEEE Automatic Speech Recognition and Understanding Workshop 2025
>
> **摘要:** In this work, we propose a multi-target backdoor attack against speaker identification using position-independent clicking sounds as triggers. Unlike previous single-target approaches, our method targets up to 50 speakers simultaneously, achieving success rates of up to 95.04%. To simulate more realistic attack conditions, we vary the signal-to-noise ratio between speech and trigger, demonstrating a trade-off between stealth and effectiveness. We further extend the attack to the speaker verification task by selecting the most similar training speaker - based on cosine similarity - as a proxy target. The attack is most effective when target and enrolled speaker pairs are highly similar, reaching success rates of up to 90% in such cases.
>
---
#### [replaced 003] Acoustic source depth estimation method based on a single hydrophone in Arctic underwater
- **分类: cs.SD; cs.NA; eess.AS; math.NA; physics.ao-ph; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2508.07157v2](http://arxiv.org/pdf/2508.07157v2)**

> **作者:** Jinbao Weng; Yubo Qi; Yanming Yang; Hongtao Wen; Hongtao Zhou; Benqing Chen; Dewei Xu; Ruichao Xue; Caigao Zeng
>
> **摘要:** Based on the normal mode and ray theory, this article discusses the characteristics of surface sound source and reception at the surface layer, and explores depth estimation methods based on normal modes and rays, and proposes a depth estimation method based on the upper limit of modal frequency. Data verification is conducted to discuss the applicability and limitations of different methods. For the surface refracted normal mode waveguide, modes can be separated through warping transformation. Based on the characteristics of normal mode amplitude variation with frequency and number, the sound source depth can be estimated by matching amplitude information. Based on the spatial variation characteristics of eigenfunctions with frequency, a sound source depth estimation method matching the cutoff frequency of normal modes is proposed. For the deep Arctic sea, the sound ray arrival structure at the receiving end is obtained through the analysis of deep inversion sound ray trajectories, and the sound source depth can be estimated by matching the time difference of ray arrivals. Experimental data is used to verify the sound field patterns and the effectiveness of the sound source depth estimation method.
>
---
#### [replaced 004] ReverbFX: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.20533v2](http://arxiv.org/pdf/2505.20533v2)**

> **作者:** Julius Richter; Till Svajda; Timo Gerkmann
>
> **备注:** Accepted at ITG Conference on Speech Communication
>
> **摘要:** We present ReverbFX, a new room impulse response (RIR) dataset designed for singing voice dereverberation research. Unlike existing datasets based on real recorded RIRs, ReverbFX features a diverse collection of RIRs captured from various reverb audio effect plugins commonly used in music production. We conduct comprehensive experiments using the proposed dataset to benchmark the challenge of dereverberation of singing voice recordings affected by artificial reverbs. We train two state-of-the-art generative models using ReverbFX and demonstrate that models trained with plugin-derived RIRs outperform those trained on realistic RIRs in artificial reverb scenarios.
>
---
#### [replaced 005] VGGSounder: Audio-Visual Evaluations for Foundation Models
- **分类: cs.MM; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.08237v2](http://arxiv.org/pdf/2508.08237v2)**

> **作者:** Daniil Zverev; Thaddäus Wiedemer; Ameya Prabhu; Matthias Bethge; Wieland Brendel; A. Sophia Koepke
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The emergence of audio-visual foundation models underscores the importance of reliably assessing their multi-modal understanding. The VGGSound dataset is commonly used as a benchmark for evaluation audio-visual classification. However, our analysis identifies several limitations of VGGSound, including incomplete labelling, partially overlapping classes, and misaligned modalities. These lead to distorted evaluations of auditory and visual capabilities. To address these limitations, we introduce VGGSounder, a comprehensively re-annotated, multi-label test set that extends VGGSound and is specifically designed to evaluate audio-visual foundation models. VGGSounder features detailed modality annotations, enabling precise analyses of modality-specific performance. Furthermore, we reveal model limitations by analysing performance degradation when adding another input modality with our new modality confusion metric.
>
---
#### [replaced 006] MultiFormer: A Multi-Person Pose Estimation System Based on CSI and Attention Mechanism
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.22555v2](http://arxiv.org/pdf/2505.22555v2)**

> **作者:** Yanyi Qu; Haoyang Ma; Wenhui Xiong
>
> **摘要:** Human pose estimation based on Channel State Information (CSI) has emerged as a promising approach for non-intrusive and precise human activity monitoring, yet faces challenges including accurate multi-person pose recognition and effective CSI feature learning. This paper presents MultiFormer, a wireless sensing system that accurately estimates human pose through CSI. The proposed system adopts a Transformer based time-frequency dual-token feature extractor with multi-head self-attention. This feature extractor is able to model inter-subcarrier correlations and temporal dependencies of the CSI. The extracted CSI features and the pose probability heatmaps are then fused by Multi-Stage Feature Fusion Network (MSFN) to enforce the anatomical constraints. Extensive experiments conducted on on the public MM-Fi dataset and our self-collected dataset show that the MultiFormer achieves higher accuracy over state-of-the-art approaches, especially for high-mobility keypoints (wrists, elbows) that are particularly difficult for previous methods to accurately estimate.
>
---
#### [replaced 007] Inversion of Arctic dual-channel sound speed profile based on random airgun signal
- **分类: cs.SD; cs.NA; eess.AS; math.NA; physics.ao-ph; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2508.07152v2](http://arxiv.org/pdf/2508.07152v2)**

> **作者:** Jinbao Weng; Yubo Qi; Yanming Yang; Hongtao Wen; Hongtao Zhou; Benqing Chen; Dewei Xu; Ruichao Xue; Caigao Zeng
>
> **摘要:** For the unique dual-channel sound speed profiles of the Canadian Basin and the Chukchi Plateau in the Arctic, based on the propagation characteristics of refracted normal modes under dual-channel sound speed profiles, an inversion method using refracted normal modes for dual-channel sound speed profiles is proposed. This method proposes a dual-parameter representation method for dual-channel sound speed profiles, tailored to the characteristics of dual-channel sound speed profiles. A dispersion structure extraction method is proposed for the dispersion structure characteristics of refracted normal modes under dual-channel sound speed profiles. Combining the parameter representation method of sound speed profiles and the dispersion structure extraction method, an inversion method for dual-channel sound speed profiles is proposed. For the common horizontal variation of sound speed profiles in long-distance acoustic propagation, a method for inverting horizontally varying dual-channel sound speed profiles is proposed. Finally, this article verifies the effectiveness of the dual-channel sound speed profile inversion method using the Arctic low-frequency long-range acoustic propagation experiment. Compared with previous sound speed profile inversion methods, the method proposed in this article has the advantages of fewer inversion parameters and faster inversion speed. It can be implemented using only a single hydrophone passively receiving random air gun signals, and it also solves the inversion problem of horizontal variation of sound speed profiles. It has significant advantages such as low cost, easy deployment, and fast computation speed.
>
---
#### [replaced 008] Leveraging Audio and Text Modalities in Mental Health: A Study of LLMs Performance
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.10417v2](http://arxiv.org/pdf/2412.10417v2)**

> **作者:** Abdelrahman A. Ali; Aya E. Fouda; Radwa J. Hanafy; Mohammed E. Fouda
>
> **摘要:** Mental health disorders are increasingly prevalent worldwide, creating an urgent need for innovative tools to support early diagnosis and intervention. This study explores the potential of Large Language Models (LLMs) in multimodal mental health diagnostics, specifically for detecting depression and Post Traumatic Stress Disorder through text and audio modalities. Using the E-DAIC dataset, we compare text and audio modalities to investigate whether LLMs can perform equally well or better with audio inputs. We further examine the integration of both modalities to determine if this can enhance diagnostic accuracy, which generally results in improved performance metrics. Our analysis specifically utilizes custom-formulated metrics; Modal Superiority Score and Disagreement Resolvement Score to evaluate how combined modalities influence model performance. The Gemini 1.5 Pro model achieves the highest scores in binary depression classification when using the combined modality, with an F1 score of 0.67 and a Balanced Accuracy (BA) of 77.4%, assessed across the full dataset. These results represent an increase of 3.1% over its performance with the text modality and 2.7% over the audio modality, highlighting the effectiveness of integrating modalities to enhance diagnostic accuracy. Notably, all results are obtained in zero-shot inferring, highlighting the robustness of the models without requiring task-specific fine-tuning. To explore the impact of different configurations on model performance, we conduct binary, severity, and multiclass tasks using both zero-shot and few-shot prompts, examining the effects of prompt variations on performance. The results reveal that models such as Gemini 1.5 Pro in text and audio modalities, and GPT-4o mini in the text modality, often surpass other models in balanced accuracy and F1 scores across multiple tasks.
>
---
#### [replaced 009] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08961v2](http://arxiv.org/pdf/2508.08961v2)**

> **作者:** Yuanyuan Wang; Dongchao Yang; Yiwen Shao; Hangting Chen; Jiankun Zhao; Zhiyong Wu; Helen Meng; Xixin Wu
>
> **摘要:** Extending pre-trained Large Language Models (LLMs)'s speech understanding or generation abilities by introducing various effective speech tokens has attracted great attention in the speech community. However, building a unified speech understanding and generation model still faces the following challenges: (1) Due to the huge modality gap between speech tokens and text tokens, extending text LLMs to unified speech LLMs relies on large-scale paired data for fine-tuning, and (2) Generation and understanding tasks prefer information at different levels, e.g., generation benefits from detailed acoustic features, while understanding favors high-level semantics. This divergence leads to difficult performance optimization in one unified model. To solve these challenges, in this paper, we present two key insights in speech tokenization and speech language modeling. Specifically, we first propose an Understanding-driven Speech Tokenizer (USTokenizer), which extracts high-level semantic information essential for accomplishing understanding tasks using text LLMs. In this way, USToken enjoys better modality commonality with text, which reduces the difficulty of modality alignment in adapting text LLMs to speech LLMs. Secondly, we present DualSpeechLM, a dual-token modeling framework that concurrently models USToken as input and acoustic token as output within a unified, end-to-end framework, seamlessly integrating speech understanding and generation capabilities. Furthermore, we propose a novel semantic supervision loss and a Chain-of-Condition (CoC) strategy to stabilize model training and enhance speech generation performance. Experimental results demonstrate that our proposed approach effectively fosters a complementary relationship between understanding and generation tasks, highlighting the promising strategy of mutually enhancing both tasks in one unified model.
>
---
#### [replaced 010] FlexCTC: GPU-powered CTC Beam Decoding With Advanced Contextual Abilities
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.07315v2](http://arxiv.org/pdf/2508.07315v2)**

> **作者:** Lilit Grigoryan; Vladimir Bataev; Nikolay Karpov; Andrei Andrusenko; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to Automatic Speech Recognition and Understanding Workshop (ASRU) 2025
>
> **摘要:** While beam search improves speech recognition quality over greedy decoding, standard implementations are slow, often sequential, and CPU-bound. To fully leverage modern hardware capabilities, we present a novel open-source FlexCTC toolkit for fully GPU-based beam decoding, designed for Connectionist Temporal Classification (CTC) models. Developed entirely in Python and PyTorch, it offers a fast, user-friendly, and extensible alternative to traditional C++, CUDA, or WFST-based decoders. The toolkit features a high-performance, fully batched GPU implementation with eliminated CPU-GPU synchronization and minimized kernel launch overhead via CUDA Graphs. It also supports advanced contextualization techniques, including GPU-powered N-gram language model fusion and phrase-level boosting. These features enable accurate and efficient decoding, making them suitable for both research and production use.
>
---
#### [replaced 011] Revisiting Your Memory: Reconstruction of Affect-Contextualized Memory via EEG-guided Audiovisual Generation
- **分类: cs.AI; cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.05296v2](http://arxiv.org/pdf/2412.05296v2)**

> **作者:** Joonwoo Kwon; Heehwan Wang; Jinwoo Lee; Sooyoung Kim; Shinjae Yoo; Yuewei Lin; Jiook Cha
>
> **备注:** Accepted at the ACM MM 2025 - The 1st CogMAEC Workshop (Oral)
>
> **摘要:** In this paper, we introduce RevisitAffectiveMemory, a novel task designed to reconstruct autobiographical memories through audio-visual generation guided by affect extracted from electroencephalogram (EEG) signals. To support this pioneering task, we present the EEG-AffectiveMemory dataset, which encompasses textual descriptions, visuals, music, and EEG recordings collected during memory recall from nine participants. Furthermore, we propose RYM (Revisit Your Memory), a three-stage framework for generating synchronized audio-visual contents while maintaining dynamic personal memory affect trajectories. Experimental results demonstrate our method successfully decodes individual affect dynamics trajectories from neural signals during memory recall (F1=0.9). Also, our approach faithfully reconstructs affect-contextualized audio-visual memory across all subjects, both qualitatively and quantitatively, with participants reporting strong affective concordance between their recalled memories and the generated content. Especially, contents generated from subject-reported affect dynamics showed higher correlation with participants' reported affect dynamics trajectories (r=0.265, p<.05) and received stronger user preference (preference=56%) compared to those generated from randomly reordered affect dynamics. Our approaches advance affect decoding research and its practical applications in personalized media creation via neural-based affect comprehension. Codes and the dataset are available at https://github.com/ioahKwon/Revisiting-Your-Memory.
>
---
