# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Online Register for Dual-Mode Self-Supervised Speech Models: Mitigating The Lack of Future Context
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，解决在线模式下因缺乏未来上下文导致的注意力不匹配问题。通过引入可学习的在线寄存器和未来预测损失，提升模型在低延迟场景下的性能。**

- **链接: [https://arxiv.org/pdf/2602.23702](https://arxiv.org/pdf/2602.23702)**

> **作者:** Keita Goto; Takashi Maekaku; Jin Sakuma; Jinchuan Tian; Yusuke Shinohara; Shinji Watanabe
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Dual-mode self-supervised speech models (S3Ms), which jointly pre-trained in the offline and online mode, suffer from attention mismatch in streaming scenarios due to missing future context. To address this challenge, we proposed online registers, learnable tokens appended to each chunk in online mode. These tokens act as virtual placeholders for unseen future frames, enabling the model to compensate for missing context without introducing additional latency. Furthermore, we introduce a future prediction loss that explicitly guides the registers to capture predictive cues, thereby enriching their ability to retain future information. Experiments on LibriSpeech, and out-of-domain benchmarks demonstrate that online registers consistently reduce the performance gap between offline and online modes, achieving a 3.4% relative improvement on LibriSpeech with 160 ms chunks, especially in low-latency settings.
>
---
#### [new 002] Leveraging large multimodal models for audio-video deepfake detection: a pilot study
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频视频深度伪造检测任务，旨在解决现有模型泛化能力弱的问题。工作是构建一个基于大模态模型的检测系统，通过联合分析音视频流提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.23393](https://arxiv.org/pdf/2602.23393)**

> **作者:** Songjun Cao; Yuqi Li; Yunpeng Luo; Jianjun Yin; Long Ma
>
> **备注:** 5pages,ICASSP2026
>
> **摘要:** Audio-visual deepfake detection (AVD) is increasingly important as modern generators can fabricate convincing speech and video. Most current multimodal detectors are small, task-specific models: they work well on curated tests but scale poorly and generalize weakly across domains. We introduce AV-LMMDetect, a supervised fine-tuned (SFT) large multimodal model that casts AVD as a prompted yes/no classification - "Is this video real or fake?". Built on Qwen 2.5 Omni, it jointly analyzes audio and visual streams for deepfake detection and is trained in two stages: lightweight LoRA alignment followed by audio-visual encoder full fine-tuning. On FakeAVCeleb and Mavos-DD, AV-LMMDetect matches or surpasses prior methods and sets a new state of the art on Mavos-DD datasets.
>
---
#### [new 003] DashengTokenizer: One layer is enough for unified audio understanding and generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DashengTokenizer，用于统一的音频理解和生成任务。解决传统方法依赖固定语义特征的问题，通过注入声学信息提升性能。实验显示其在多个任务中表现优异，挑战了VAE架构的必要性。**

- **链接: [https://arxiv.org/pdf/2602.23765](https://arxiv.org/pdf/2602.23765)**

> **作者:** Heinrich Dinkel; Xingwei Sun; Gang Li; Jiahao Mei; Yadong Niu; Jizhong Liu; Xiyang Li; Yifan Liao; Jiahao Zhou; Junbo Zhang; Jian Luan
>
> **摘要:** This paper introduces DashengTokenizer, a continuous audio tokenizer engineered for joint use in both understanding and generation tasks. Unlike conventional approaches, which train acoustic tokenizers and subsequently integrate frozen semantic knowledge, our method inverts this paradigm: we leverage frozen semantic features and inject acoustic information. In linear evaluation across 22 diverse tasks, our method outperforms previous audio codec and audio encoder baselines by a significant margin while maintaining competitive audio reconstruction quality. Notably, we demonstrate that this acoustic injection improves performance for tasks such as speech emotion recognition, music understanding, and acoustic scene classification. We further evaluate the tokenizer's generative performance on text-to-audio (TTA), text-to-music (TTM), and speech enhancement (SE). Our approach surpasses standard variational autoencoder (VAE)-based methods on TTA and TTM tasks, while its effectiveness on SE underscores its capabilities as a general-purpose audio encoder. Finally, our results challenge the prevailing assumption that VAE-based architectures are a prerequisite for audio synthesis. Checkpoints are available at this https URL.
>
---
#### [new 004] An Empirical Analysis of Task-Induced Encoder Bias in Fréchet Audio Distance
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究文本到音频生成的评估问题，指出FAD评分受编码器任务影响，提出分解评估指标并分析不同编码器的优劣。**

- **链接: [https://arxiv.org/pdf/2602.23958](https://arxiv.org/pdf/2602.23958)**

> **作者:** Wonwoo Jeong
>
> **备注:** 6 pages, 4 figures. Submitted to Interspeech 2026. Source code and evaluation pipeline are available at: this https URL
>
> **摘要:** Fréchet Audio Distance (FAD) is the de facto standard for evaluating text-to-audio generation, yet its scores depend on the underlying encoder's embedding space. An encoder's training task dictates which acoustic features are preserved or discarded, causing FAD to inherit systematic task-induced biases. We decompose evaluation into Recall, Precision, and Alignment (split into semantic and structural dimensions), using log-scale normalization for fair cross-encoder comparison. Controlled experiments on six encoders across two datasets reveal a four-axis trade-off: reconstruction-based AudioMAE leads precision sensitivity; ASR-trained Whisper dominates structural detection but is blind to signal degradation; classification-trained VGGish maximizes semantic detection but penalizes legitimate intra-class variation. Since no single encoder is a universal evaluator, future metrics must shift toward evaluation-native encoders intrinsically aligned with human perception.
>
---
#### [new 005] AudioCapBench: Quick Evaluation on Audio Captioning across Sound, Music, and Speech
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频描述任务，旨在评估多模态模型在声音、音乐和语音上的生成能力。构建了AudioCapBench基准，对比分析多个模型的表现。**

- **链接: [https://arxiv.org/pdf/2602.23649](https://arxiv.org/pdf/2602.23649)**

> **作者:** Jielin Qiu; Jianguo Zhang; Zixiang Chen; Liangwei Yang; Ming Zhu; Juntao Tan; Haolin Chen; Wenting Zhao; Rithesh Murthy; Roshan Ram; Akshara Prabhakar; Shelby Heinecke; Caiming; Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** We introduce AudioCapBench, a benchmark for evaluating audio captioning capabilities of large multimodal models. \method covers three distinct audio domains, including environmental sound, music, and speech, with 1,000 curated evaluation samples drawn from established datasets. We evaluate 13 models across two providers (OpenAI, Google Gemini) using both reference-based metrics (METEOR, BLEU, ROUGE-L) and an LLM-as-Judge framework that scores predictions on three orthogonal dimensions: \textit{accuracy} (semantic correctness), \textit{completeness} (coverage of reference content), and \textit{hallucination} (absence of fabricated content). Our results reveal that Gemini models generally outperform OpenAI models on overall captioning quality, with Gemini~3~Pro achieving the highest overall score (6.00/10), while OpenAI models exhibit lower hallucination rates. All models perform best on speech captioning and worst on music captioning. We release the benchmark as well as evaluation code to facilitate reproducible audio understanding research.
>
---
#### [new 006] Hello-Chat: Towards Realistic Social Audio Interactions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出Hello-Chat，解决音频交互中缺乏自然与情感的问题。属于音频语言模型任务，通过真实对话数据和跨模态训练，提升语音的自然度与情感共鸣。**

- **链接: [https://arxiv.org/pdf/2602.23387](https://arxiv.org/pdf/2602.23387)**

> **作者:** Yueran Hou; Peilei Jia; Zihan Sun; Qihang Lu; Wenbing Yang; Yingming Gao; Ya Li; Jun Gao
>
> **摘要:** Recent advancements in Large Audio Language Models (LALMs) have demonstrated exceptional performance in speech recognition and translation. However, existing models often suffer from a disconnect between perception and expression, resulting in a robotic "read-speech" style that lacks the spontaneity and emotional resonance of real human interaction. In this report, we introduce Hello-Chat, an end-to-end audio language model designed for realistic social scenarios. By leveraging a massive dataset of real-life conversations and employing a modality-interleaved training strategy, Hello-Chat achieves a breakthrough in anthropomorphic generation. Experimental results show that our model not only reaches state-of-the-art (SOTA) performance on specific audio understanding tasks but also significantly outperforms existing baselines in prosodic naturalness and emotional alignment, paving the way for the next generation of empathetic AI agents.
>
---
#### [new 007] SHINE: Sequential Hierarchical Integration Network for EEG and MEG
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音检测任务，旨在从MEG信号中重建语音-静音序列。提出SHINE网络，并结合基线方法提升性能。**

- **链接: [https://arxiv.org/pdf/2602.23960](https://arxiv.org/pdf/2602.23960)**

> **作者:** Xiran Xu; Yujie Yan; Xihong Wu; Jing Chen
>
> **备注:** ranked second at LibriBrain Competition 2025 this https URL
>
> **摘要:** How natural speech is represented in the brain constitutes a major challenge for cognitive neuroscience, with cortical envelope-following responses playing a central role in speech decoding. This paper presents our approach to the Speech Detection task in the LibriBrain Competition 2025, utilizing over 50 hours of magnetoencephalography (MEG) signals from a single participant listening to LibriVox audiobooks. We introduce the proposed Sequential Hierarchical Integration Network for EEG and MEG (SHINE) to reconstruct the binary speech-silence sequences from MEG signals. In the Extended Track, we further incorporated auxiliary reconstructions of speech envelopes and Mel spectrograms to enhance training. Ensemble methods combining SHINE with baselines (BrainMagic, AWavNet, ConvConcatNet) achieved F1-macro scores of 0.9155 (Standard Track) and 0.9184 (Extended Track) on the leaderboard test set.
>
---
#### [new 008] SongSong: A Time Phonograph for Chinese SongCi Music from Thousand of Years Away
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出SongSong模型，用于生成古体词曲音乐，解决古代音乐生成困难的问题。工作包括模型设计与数据集构建。**

- **链接: [https://arxiv.org/pdf/2602.24071](https://arxiv.org/pdf/2602.24071)**

> **作者:** Jiajia Li; Jiliang Hu; Ziyi Pan; Chong Chen; Zuchao Li; Ping Wang; Lefei Zhang
>
> **备注:** 9 pages, 6 figures, accepted by AAAI 2025
>
> **摘要:** Recently, there have been significant advancements in music generation. However, existing models primarily focus on creating modern pop songs, making it challenging to produce ancient music with distinct rhythms and styles, such as ancient Chinese SongCi. In this paper, we introduce SongSong, the first music generation model capable of restoring Chinese SongCi to our knowledge. Our model first predicts the melody from the input SongCi, then separately generates the singing voice and accompaniment based on that melody, and finally combines all elements to create the final piece of music. Additionally, to address the lack of ancient music datasets, we create OpenSongSong, a comprehensive dataset of ancient Chinese SongCi music, featuring 29.9 hours of compositions by various renowned SongCi music masters. To assess SongSong's proficiency in performing SongCi, we randomly select 85 SongCi sentences that were not part of the training set for evaluation against SongSong and music generation platforms such as Suno and SkyMusic. The subjective and objective outcomes indicate that our proposed model achieves leading performance in generating high-quality SongCi music.
>
---
#### [new 009] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction
- **分类: cs.AI; cs.SD**

- **简介: 该论文属于对话系统评估任务，旨在检验语音到语音系统是否具备人类对话能力。通过首次图灵测试，发现现有系统尚未达到人类水平，并分析其在情感和语气上的不足。**

- **链接: [https://arxiv.org/pdf/2602.24080](https://arxiv.org/pdf/2602.24080)**

> **作者:** Xiang Li; Jiabao Gao; Sipei Lin; Xuan Zhou; Chi Zhang; Bo Cheng; Jiale Han; Benyou Wang
>
> **备注:** Accepted by ICLR 2026 Conference
>
> **摘要:** The pursuit of human-like conversational agents has long been guided by the Turing test. For modern speech-to-speech (S2S) systems, a critical yet unanswered question is whether they can converse like humans. To tackle this, we conduct the first Turing test for S2S systems, collecting 2,968 human judgments on dialogues between 9 state-of-the-art S2S systems and 28 human participants. Our results deliver a clear finding: no existing evaluated S2S system passes the test, revealing a significant gap in human-likeness. To diagnose this failure, we develop a fine-grained taxonomy of 18 human-likeness dimensions and crowd-annotate our collected dialogues accordingly. Our analysis shows that the bottleneck is not semantic understanding but stems from paralinguistic features, emotional expressivity, and conversational persona. Furthermore, we find that off-the-shelf AI models perform unreliably as Turing test judges. In response, we propose an interpretable model that leverages the fine-grained human-likeness ratings and delivers accurate and transparent human-vs-machine discrimination, offering a powerful tool for automatic human-likeness evaluation. Our work establishes the first human-likeness evaluation for S2S systems and moves beyond binary outcomes to enable detailed diagnostic insights, paving the way for human-like improvements in conversational AI systems.
>
---
#### [new 010] Task-Lens: Cross-Task Utility Based Speech Dataset Profiling for Low-Resource Indian Languages
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Task-Lens，用于评估印度语 speech 数据集在多个任务中的适用性，解决低资源语言数据不足问题，通过分析数据集元数据和任务适配性，识别关键缺失领域。**

- **链接: [https://arxiv.org/pdf/2602.23388](https://arxiv.org/pdf/2602.23388)**

> **作者:** Swati Sharma; Divya V. Sharma; Anubha Gupta
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** The rising demand for inclusive speech technologies amplifies the need for multilingual datasets for Natural Language Processing (NLP) research. However, limited awareness of existing task-specific resources in low-resource languages hinders research. This challenge is especially acute in linguistically diverse countries, such as India. Cross-task profiling of existing Indian speech datasets can alleviate the data scarcity challenge. This involves investigating the utility of datasets across multiple downstream tasks rather than focusing on a single task. Prior surveys typically catalogue datasets for a single task, leaving comprehensive cross-task profiling as an open opportunity. Therefore, we propose Task-Lens, a cross-task survey that assesses the readiness of 50 Indian speech datasets spanning 26 languages for nine downstream speech tasks. First, we analyze which datasets contain metadata and properties suitable for specific tasks. Next, we propose task-aligned enhancements to unlock datasets to their full downstream potential. Finally, we identify tasks and Indian languages that are critically underserved by current resources. Our findings reveal that many Indian speech datasets contain untapped metadata that can support multiple downstream tasks. By uncovering cross-task linkages and gaps, Task-Lens enables researchers to explore the broader applicability of existing datasets and to prioritize dataset creation for underserved tasks and languages.
>
---
#### [new 011] Design of a Hands-Free Short-Range Intercommunication Device Using LoRa for Secure Field Communication
- **分类: eess.SP; eess.AS**

- **简介: 论文设计了一种基于LoRa的微型加密通信设备，用于安全的短距离无线通信。该任务旨在解决传统设备体积大、功耗高、不便于穿戴的问题，通过LoRa技术实现低功耗、远距离、安全的语音通信。**

- **链接: [https://arxiv.org/pdf/2602.23924](https://arxiv.org/pdf/2602.23924)**

> **作者:** Ayush Kumar Agrawal; Soumendu Das; Jayendra Kumar
>
> **摘要:** Short-range reliable and secure communication is a major priority in the tactical, military and disaster response settings where the traditional communication infrastructure is either off-line or prone to interception. Current VHF/UHF radios and software-defined radios are popular but large-sized devices and require lots of power, making them not suitable to be used as lightweight wearable devices with seamless hand-free use. In this paper, the design and theoretical framework of a miniature, LoRa based encrypted intercommunication device that can be used in secure field communication over a range of 1-1.5km and under line-of-sight conditions is provided. The suggested system consists of a voice-activated acquisition block, digital audio compression, an embedded microcontroller processor, and AES-128 encryption followed by a low-power transmission via the LoRa protocol. Through the ability of chirp spread spectrum modulation to utilize the long-range and low-energy properties, the system is guaranteed reliable communications coupled with low power consumption and low electromagnetic footprint. The theoretical analysis of the proposed communication range is justified using a link-budget that justifies the practicability of the communication range in the real propagation conditions. This architecture focuses on infrastructural agnosticism, peer-to-peer security as well as wearable ergonomics. The given scheme shows the possibilities of LoRa technology in the scope of other traditional IoT telemetry, and it can be further extended to include secure tactical voice communication platforms.
>
---
## 更新

#### [replaced 001] Sonic4D: Spatial Audio Generation for Immersive 4D Scene Exploration
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于4D场景生成任务，旨在解决空间音频生成不足的问题。通过三阶段方法，从单目视频中生成与4D场景同步的沉浸式空间音频。**

- **链接: [https://arxiv.org/pdf/2506.15759](https://arxiv.org/pdf/2506.15759)**

> **作者:** Siyi Xie; Hanxin Zhu; Xinyi Chen; Tianyu He; Xin Li; Zhibo Chen
>
> **备注:** 17 pages, 7 figures. Project page: this https URL
>
> **摘要:** Recent advancements in 4D generation have demonstrated its remarkable capability in synthesizing photorealistic renderings of dynamic 3D scenes. However, despite achieving impressive visual performance, almost all existing methods overlook the generation of spatial audio aligned with the corresponding 4D scenes, posing a significant limitation to truly immersive audiovisual experiences. To mitigate this issue, we propose Sonic4D, a novel framework that enables spatial audio generation for immersive exploration of 4D scenes. Specifically, our method is composed of three stages: 1) To capture both the dynamic visual content and raw auditory information from a monocular video, we first employ pre-trained expert models to generate the 4D scene and its corresponding monaural audio. 2) Subsequently, to transform the monaural audio into spatial audio, we localize and track the sound sources within the 4D scene, where their 3D spatial coordinates at different timestamps are estimated via a pixel-level visual grounding strategy. 3) Based on the estimated sound source locations, we further synthesize plausible spatial audio that varies across different viewpoints and timestamps using physics-based simulation. Extensive experiments have demonstrated that our proposed method generates realistic spatial audio consistent with the synthesized 4D scene in a training-free manner, significantly enhancing the immersive experience for users. Generated audio and video examples are available at this https URL.
>
---
#### [replaced 002] CSyMR: Benchmarking Compositional Music Information Retrieval in Symbolic Music Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于符号音乐推理任务，旨在解决自然语言与符号音乐表示不匹配的问题。提出CSyMR-Bench基准和工具增强的检索框架，提升组合式音乐信息检索效果。**

- **链接: [https://arxiv.org/pdf/2601.11556](https://arxiv.org/pdf/2601.11556)**

> **作者:** Boyang Wang; Yash Vishe; Xin Xu; Zachary Novack; Xunyi Jiang; Julian McAuley; Junda Wu
>
> **摘要:** Natural language information needs over symbolic music scores rarely reduce to a single step lookup. Many queries require compositional Music Information Retrieval (MIR) that extracts multiple pieces of evidence from structured notation and aggregates them to answer the question. This setting remains challenging for Large Language Models due to the mismatch between natural language intents and symbolic representations, as well as the difficulty of reliably handling long structured contexts. Existing benchmarks only partially capture these retrieval demands, often emphasizing isolated theoretical knowledge or simplified settings. We introduce CSyMR-Bench, a benchmark for compositional MIR in symbolic music reasoning grounded in authentic user scenarios. It contains 126 multiple choice questions curated from community discussions and professional examinations, where each item requires chaining multiple atomic analyses over a score to derive implicit musical evidence. To support diagnosis, we provide a taxonomy with six query intent categories and six analytical dimension tags. We further propose a tool-augmented retrieval and reasoning framework that integrates a ReAct-style controller with deterministic symbolic analysis operators built with music21. Experiments across prompting baselines and agent variants show that tool-grounded compositional retrieval consistently outperforms Large Language Model-only approaches, yielding 5-7% absolute accuracy gains, with the largest improvements on analysis-heavy categories.
>
---
#### [replaced 003] Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在提升识别准确率。通过引入音频条件扩散大语言模型LLaDA，探索其在ASR中的应用，优化识别效果。**

- **链接: [https://arxiv.org/pdf/2509.16622](https://arxiv.org/pdf/2509.16622)**

> **作者:** Mengqi Wang; Zhan Liu; Zengrui Jin; Guangzhi Sun; Chao Zhang; Philip C. Woodland
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements. Code and model are open-sourced at this https URL.
>
---
#### [replaced 004] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis
- **分类: eess.AS; cs.AI; cs.DB; cs.HC; cs.MA; cs.SD**

- **简介: 该论文属于呼吸音诊断任务，解决深度学习在呼吸音分析中的信息丢失和数据不足问题。提出Resp-Agent系统，结合主动对抗课程代理和多模态生成技术，提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2602.15909](https://arxiv.org/pdf/2602.15909)**

> **作者:** Pengfei Zhang; Tianxin Xie; Minghao Yang; Li Liu
>
> **备注:** 24 pages, 3 figures. Published as a conference paper at ICLR 2026
>
> **摘要:** Deep learning-based respiratory auscultation is currently hindered by two fundamental challenges: (i) inherent information loss, as converting signals into spectrograms discards transient acoustic events and clinical context; (ii) limited data availability, exacerbated by severe class imbalance. To bridge these gaps, we present Resp-Agent, an autonomous multimodal system orchestrated by a novel Active Adversarial Curriculum Agent (Thinker-A$^2$CA). Unlike static pipelines, Thinker-A$^2$CA serves as a central controller that actively identifies diagnostic weaknesses and schedules targeted synthesis in a closed loop. To address the representation gap, we introduce a modality-weaving Diagnoser that weaves clinical text with audio tokens via strategic global attention and sparse audio anchors, capturing both long-range clinical context and millisecond-level transients. To address the data gap, we design a flow matching Generator that adapts a text-only Large Language Model (LLM) via modality injection, decoupling pathological content from acoustic style to synthesize hard-to-diagnose samples. As a foundation for this work, we introduce Resp-229k, a benchmark corpus of 229k recordings paired with LLM-distilled clinical narratives. Extensive experiments demonstrate that Resp-Agent consistently outperforms prior approaches across diverse evaluation settings, improving diagnostic robustness under data scarcity and long-tailed class imbalance. Our code and data are available at this https URL.
>
---
#### [replaced 005] VoiceBridge: General Speech Restoration with One-step Latent Bridge Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出VoiceBridge，解决通用语音修复问题，通过单步潜在桥模型高效恢复多种失真语音，提升重建质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.25275](https://arxiv.org/pdf/2509.25275)**

> **作者:** Chi Zhang; Kaiwen Zheng; Zehua Chen; Jun Zhu
>
> **摘要:** Bridge models have been investigated in speech enhancement but are mostly single-task, with constrained general speech restoration (GSR) capability. In this work, we propose VoiceBridge, a one-step latent bridge model (LBM) for GSR, capable of efficiently reconstructing 48 kHz fullband speech from diverse distortions. To inherit the advantages of data-domain bridge models, we design an energy-preserving variational autoencoder, enhancing the waveform-latent space alignment over varying energy levels. By compressing waveform into continuous latent representations, VoiceBridge models~\textit{various} GSR tasks with a~\textit{single} latent-to-latent generative process backed by a scalable transformer. To alleviate the challenge of reconstructing the high-quality target from distinctively different low-quality priors, we propose a joint neural prior for GSR, uniformly reducing the burden of the LBM in diverse tasks. Building upon these designs, we further investigate bridge training objective by jointly tuning LBM, decoder and discriminator together, transforming the model from a denoiser to generator and enabling \textit{one-step GSR without distillation}. Extensive validation across in-domain (\textit{e.g.}, denoising and super-resolution) and out-of-domain tasks (\textit{e.g.}, refining synthesized speech) and datasets demonstrates the superior performance of VoiceBridge. Demos: this https URL.
>
---
#### [replaced 006] DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos
- **分类: cs.SD**

- **简介: 该论文属于空间音频生成任务，解决复杂场景下360度视频生成高质量FOA的问题。通过视觉与音频联合处理，实现动态声源定位与环境建模，提升音频真实感与沉浸体验。**

- **链接: [https://arxiv.org/pdf/2602.06846](https://arxiv.org/pdf/2602.06846)**

> **作者:** Ziyu Luo; Lin Chen; Qiang Qu; Xiaoming Chen; Yiran Shen
>
> **摘要:** Spatial audio is crucial for creating compelling immersive 360-degree video experiences. However, generating realistic spatial audio, such as first-order ambisonics (FOA), from 360-degree videos in complex acoustic scenes remains challenging. Existing methods often overlook the dynamic nature and acoustic complexity of 360-degree scenes, fail to fully account for dynamic sound sources, and neglect complex environmental effects such as occlusion, reflections, and reverberation, which are influenced by scene geometries and materials. We propose DynFOA, a framework based on dynamic acoustic perception and conditional diffusion, for generating high-fidelity FOA from 360-degree videos. DynFOA first performs visual processing via a video encoder, which detects and localizes multiple dynamic sound sources, estimates their depth and semantics, and reconstructs the scene geometry and materials using a 3D Gaussian Splatting. This reconstruction technique accurately models occlusion, reflections, and reverberation based on the geometries and materials of the reconstructed 3D scene and the listener's viewpoint. The audio encoder then captures the spatial motion and temporal 4D sound source trajectories to fine-tune the diffusion-based FOA generator. The fine-tuned FOA generator adjusts spatial cues in real time, ensuring consistent directional fidelity during listener head rotation and complex environmental changes. Extensive evaluations demonstrate that DynFOA consistently outperforms existing methods across metrics such as spatial accuracy, acoustic fidelity, and distribution matching, while also improving the user experience. Therefore, DynFOA provides a robust and scalable approach to rendering realistic dynamic spatial audio for VR and immersive media applications.
>
---
#### [replaced 007] TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音分离任务，旨在提升模型效率与泛化能力。提出TIGER模型，减少参数和计算量，同时引入EchoSet数据集增强实际场景适应性。**

- **链接: [https://arxiv.org/pdf/2410.01469](https://arxiv.org/pdf/2410.01469)**

> **作者:** Mohan Xu; Kai Li; Guo Chen; Xiaolin Hu
>
> **备注:** Accepted by ICLR 2025, demo page: this https URL
>
> **摘要:** In recent years, much speech separation research has focused primarily on improving model performance. However, for low-latency speech processing systems, high efficiency is equally important. Therefore, we propose a speech separation model with significantly reduced parameters and computational costs: Time-frequency Interleaved Gain Extraction and Reconstruction network (TIGER). TIGER leverages prior knowledge to divide frequency bands and compresses frequency information. We employ a multi-scale selective attention module to extract contextual features while introducing a full-frequency-frame attention module to capture both temporal and frequency contextual information. Additionally, to more realistically evaluate the performance of speech separation models in complex acoustic environments, we introduce a dataset called EchoSet. This dataset includes noise and more realistic reverberation (e.g., considering object occlusions and material properties), with speech from two speakers overlapping at random proportions. Experimental results showed that models trained on EchoSet had better generalization ability than those trained on other datasets compared to the data collected in the physical world, which validated the practical value of the EchoSet. On EchoSet and real-world data, TIGER significantly reduces the number of parameters by 94.3% and the MACs by 95.3% while achieving performance surpassing the state-of-the-art (SOTA) model TF-GridNet.
>
---
#### [replaced 008] Discrete Optimal Transport and Voice Conversion
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音转换任务，旨在通过离散最优传输方法实现高质量语音转换，并揭示其潜在的对抗攻击风险。**

- **链接: [https://arxiv.org/pdf/2505.04382](https://arxiv.org/pdf/2505.04382)**

> **作者:** Anton Selitskiy; Maitreya Kocharekar
>
> **备注:** 4 pages, 7 figure, 1 table
>
> **摘要:** In this work, we address the task of voice conversion (VC) using a vector-based interface. To align audio embeddings across speakers, we employ discrete optimal transport (OT) and approximate the transport map using the barycentric projection. Our evaluation demonstrates that this approach yields high-quality and effective voice conversion. We also perform an ablation study on the number of embeddings used, extending previous work on simple averaging of kNN and OT results. Additionally, we show that applying discrete OT as a post-processing step in audio generation can cause synthetic speech to be misclassified as real, revealing a novel and strong adversarial attack.
>
---
