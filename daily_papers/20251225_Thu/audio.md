# 音频 cs.SD;  eess.AS

- **最新发布 5 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] GenTSE: Enhancing Target Speaker Extraction via a Coarse-to-Fine Generative Language Model
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 论文提出GenTSE，用两阶段生成语言模型提升目标说话人提取效果，分语义与声学建模，结合冻结LM训练与DPO优化，提升语音质量与一致性。**

- **链接: [https://arxiv.org/pdf/2512.20978v1](https://arxiv.org/pdf/2512.20978v1)**

> **作者:** Haoyang Li; Xuyi Zhuang; Azmat Adnan; Ye Ni; Wei Rao; Shreyas Gopal; Eng Siong Chng
>
> **摘要:** Language Model (LM)-based generative modeling has emerged as a promising direction for TSE, offering potential for improved generalization and high-fidelity speech. We present GenTSE, a two-stage decoder-only generative LM approach for TSE: Stage-1 predicts coarse semantic tokens, and Stage-2 generates fine acoustic tokens. Separating semantics and acoustics stabilizes decoding and yields more faithful, content-aligned target speech. Both stages use continuous SSL or codec embeddings, offering richer context than discretized-prompt methods. To reduce exposure bias, we employ a Frozen-LM Conditioning training strategy that conditions the LMs on predicted tokens from earlier checkpoints to reduce the gap between teacher-forcing training and autoregressive inference. We further employ DPO to better align outputs with human perceptual preferences. Experiments on Libri2Mix show that GenTSE surpasses previous LM-based systems in speech quality, intelligibility, and speaker consistency.
>
---
#### [new 002] Towards Practical Automatic Piano Reduction using BERT with Semi-supervised Learning
- **分类: cs.SD; cs.SC**

- **简介: 论文提出基于BERT的半监督学习方法，自动实现钢琴缩编，解决标注数据少问题，通过简化与和声两步生成实用结果。**

- **链接: [https://arxiv.org/pdf/2512.21324v1](https://arxiv.org/pdf/2512.21324v1)**

> **作者:** Wan Ki Wong; Ka Ho To; Chuck-jee Chau; Lucas Wong; Kevin Y. Yip; Irwin King
>
> **摘要:** In this study, we present a novel automatic piano reduction method with semi-supervised machine learning. Piano reduction is an important music transformation process, which helps musicians and composers as a musical sketch for performances and analysis. The automation of such is a highly challenging research problem but could bring huge conveniences as manually doing a piano reduction takes a lot of time and effort. While supervised machine learning is often a useful tool for learning input-output mappings, it is difficult to obtain a large quantity of labelled data. We aim to solve this problem by utilizing semi-supervised learning, so that the abundant available data in classical music can be leveraged to perform the task with little or no labelling effort. In this regard, we formulate a two-step approach of music simplification followed by harmonization. We further propose and implement two possible solutions making use of an existing machine learning framework -- MidiBERT. We show that our solutions can output practical and realistic samples with an accurate reduction that needs only small adjustments in post-processing. Our study forms the groundwork for the use of semi-supervised learning in automatic piano reduction, where future researchers can take reference to produce more state-of-the-art results.
>
---
#### [new 003] USE: A Unified Model for Universal Sound Separation and Extraction
- **分类: eess.AS**

- **简介: 提出USE统一模型，融合声源分离与目标提取，自动推断声源数或解析用户线索，通过联合训练提升双任务性能。**

- **链接: [https://arxiv.org/pdf/2512.21215v1](https://arxiv.org/pdf/2512.21215v1)**

> **作者:** Hongyu Wang; Chenda Li; Xin Zhou; Shuai Wang; Yanmin Qian
>
> **备注:** Accepted as an oral presentation by AAAI 2026
>
> **摘要:** Sound separation (SS) and target sound extraction (TSE) are fundamental techniques for addressing complex acoustic scenarios. While existing SS methods struggle with determining the unknown number of sound sources, TSE approaches require precisely specified clues to achieve optimal performance. This paper proposes a unified framework that synergistically combines SS and TSE to overcome their individual limitations. Our architecture employs two complementary components: 1) An Encoder-Decoder Attractor (EDA) network that automatically infers both the source count and corresponding acoustic clues for SS, and 2) A multi-modal fusion network that precisely interprets diverse user-provided clues (acoustic, semantic, or visual) for TSE. Through joint training with cross-task consistency constraints, we establish a unified latent space that bridges both paradigms. During inference, the system adaptively operates in either fully autonomous SS mode or clue-driven TSE mode. Experiments demonstrate remarkable performance in both tasks, with notable improvements of 1.4 dB SDR improvement in SS compared to baseline and 86\% TSE accuracy.
>
---
#### [new 004] SACodec: Asymmetric Quantization with Semantic Anchoring for Low-Bitrate High-Fidelity Neural Speech Codecs
- **分类: cs.SD**

- **简介: 提出SACodec神经语音编解码器，通过语义锚定与非对称量化，在1.5kbps下兼顾高保真与语义丰富性，突破低码率语音编码瓶颈。**

- **链接: [https://arxiv.org/pdf/2512.20944v1](https://arxiv.org/pdf/2512.20944v1)**

> **作者:** Zhongren Dong; Bin Wang; Jing Han; Haotian Guo; Xiaojun Mo; Yimin Cao; Zixing Zhang
>
> **摘要:** Neural Speech Codecs face a fundamental trade-off at low bitrates: preserving acoustic fidelity often compromises semantic richness. To address this, we introduce SACodec, a novel codec built upon an asymmetric dual-quantizer that employs our proposed Semantic Anchoring mechanism. This design strategically decouples the quantization of Semantic and Acoustic details. The semantic anchoring is achieved via a lightweight projector that aligns acoustic features with a frozen, large-scale mHuBERT codebook, injecting linguistic priors while guaranteeing full codebook utilization. Sequentially, for acoustic details, a residual activation module with SimVQ enables a single-layer quantizer (acoustic path) to faithfully recover fine-grained information. At just 1.5 kbps, SACodec establishes a new state of the art by excelling in both fidelity and semantics: subjective listening tests confirm that its reconstruction quality is perceptually highly comparable to ground-truth audio, while its tokens demonstrate substantially improved semantic richness in downstream tasks.
>
---
#### [new 005] Foundation Model-based Evaluation of Neuropsychiatric Disorders: A Lifespan-Inclusive, Multi-Modal, and Multi-Lingual Study
- **分类: cs.CL; cs.SD**

- **简介: 提出FEND框架，用多模态基础模型评估跨语言、全生命周期神经精神疾病，解决多模态融合与泛化难题，提供统一评测基准。**

- **链接: [https://arxiv.org/pdf/2512.20948v1](https://arxiv.org/pdf/2512.20948v1)**

> **作者:** Zhongren Dong; Haotian Guo; Weixiang Xu; Huan Zhao; Zixing Zhang
>
> **摘要:** Neuropsychiatric disorders, such as Alzheimer's disease (AD), depression, and autism spectrum disorder (ASD), are characterized by linguistic and acoustic abnormalities, offering potential biomarkers for early detection. Despite the promise of multi-modal approaches, challenges like multi-lingual generalization and the absence of a unified evaluation framework persist. To address these gaps, we propose FEND (Foundation model-based Evaluation of Neuropsychiatric Disorders), a comprehensive multi-modal framework integrating speech and text modalities for detecting AD, depression, and ASD across the lifespan. Leveraging 13 multi-lingual datasets spanning English, Chinese, Greek, French, and Dutch, we systematically evaluate multi-modal fusion performance. Our results show that multi-modal fusion excels in AD and depression detection but underperforms in ASD due to dataset heterogeneity. We also identify modality imbalance as a prevalent issue, where multi-modal fusion fails to surpass the best mono-modal models. Cross-corpus experiments reveal robust performance in task- and language-consistent scenarios but noticeable degradation in multi-lingual and task-heterogeneous settings. By providing extensive benchmarks and a detailed analysis of performance-influencing factors, FEND advances the field of automated, lifespan-inclusive, and multi-lingual neuropsychiatric disorder assessment. We encourage researchers to adopt the FEND framework for fair comparisons and reproducible research.
>
---
## 更新

#### [replaced 001] A Data-Centric Approach to Generalizable Speech Deepfake Detection
- **分类: cs.SD; eess.SP**

- **简介: 论文提出数据驱动方法提升语音深度伪造检测泛化能力，通过分析数据规模规律并设计DOSS策略优化多源数据融合，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.18210v2](https://arxiv.org/pdf/2512.18210v2)**

> **作者:** Wen Huang; Yuchen Mao; Yanmin Qian
>
> **摘要:** Achieving robust generalization in speech deepfake detection (SDD) remains a primary challenge, as models often fail to detect unseen forgery methods. While research has focused on model-centric and algorithm-centric solutions, the impact of data composition is often underexplored. This paper proposes a data-centric approach, analyzing the SDD data landscape from two practical perspectives: constructing a single dataset and aggregating multiple datasets. To address the first perspective, we conduct a large-scale empirical study to characterize the data scaling laws for SDD, quantifying the impact of source and generator diversity. To address the second, we propose the Diversity-Optimized Sampling Strategy (DOSS), a principled framework for mixing heterogeneous data with two implementations: DOSS-Select (pruning) and DOSS-Weight (re-weighting). Our experiments show that DOSS-Select outperforms the naive aggregation baseline while using only 3% of the total available data. Furthermore, our final model, trained on a 12k-hour curated data pool using the optimal DOSS-Weight strategy, achieves state-of-the-art performance, outperforming large-scale baselines with greater data and model efficiency on both public benchmarks and a new challenge set of various commercial APIs.
>
---
#### [replaced 002] VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents
- **分类: cs.SD; cs.CL**

- **简介: 提出VCB Bench中文评测基准，用真实人声评估音频大模型在指令遵循、知识理解与鲁棒性三方面表现，弥补现有评测不足。**

- **链接: [https://arxiv.org/pdf/2510.11098v2](https://arxiv.org/pdf/2510.11098v2)**

> **作者:** Jiliang Hu; Wenfu Wang; Zuchao Li; Chenxing Li; Yiyang Zhao; Hanzhao Li; Liqiang Zhang; Meng Yu; Dong Yu
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.
>
---
#### [replaced 003] Speaker Recognition -- Wavelet Packet Based Multiresolution Feature Extraction Approach
- **分类: cs.SD**

- **简介: 提出结合MFCC与小波包的多分辨率特征提取法，提升文本无关说话人识别与验证的噪声鲁棒性，用GMM/HMM分类，在多个语料库上验证有效。**

- **链接: [https://arxiv.org/pdf/2512.18902v2](https://arxiv.org/pdf/2512.18902v2)**

> **作者:** Saurabh Bhardwaj; Smriti Srivastava; Abhishek Bhandari; Krit Gupta; Hitesh Bahl; J. R. P. Gupta
>
> **备注:** This paper was originally written in Summer 2013 and previously made available on Figshare. The present submission is uploaded for archival and citation purposes
>
> **摘要:** This paper proposes a novel Wavelet Packet based feature extraction approach for the task of text independent speaker recognition. The features are extracted by using the combination of Mel Frequency Cepstral Coefficient (MFCC) and Wavelet Packet Transform (WPT).Hybrid Features technique uses the advantage of human ear simulation offered by MFCC combining it with multi-resolution property and noise robustness of WPT. To check the validity of the proposed approach for the text independent speaker identification and verification we have used the Gaussian Mixture Model (GMM) and Hidden Markov Model (HMM) respectively as the classifiers. The proposed paradigm is tested on voxforge speech corpus and CSTR US KED Timit database. The paradigm is also evaluated after adding standard noise signal at different level of SNRs for evaluating the noise robustness. Experimental results show that better results are achieved for the tasks of both speaker identification as well as speaker verification.
>
---
#### [replaced 004] DiTSinger: Scaling Singing Voice Synthesis with Diffusion Transformer and Implicit Alignment
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出DiTSinger，用扩散Transformer和隐式对齐解决歌唱合成数据少、难扩展问题，实现无需音素时长标注的高保真歌声合成。**

- **链接: [https://arxiv.org/pdf/2510.09016v2](https://arxiv.org/pdf/2510.09016v2)**

> **作者:** Zongcai Du; Guilin Deng; Xiaofeng Guo; Xin Gao; Linke Li; Kaichang Cheng; Fubo Han; Siyu Yang; Peng Liu; Pan Zhong; Qiang Fu
>
> **备注:** ICASSP26 under review. Demo page: https://nju-jet.github.io/DiTSinger
>
> **摘要:** Recent progress in diffusion-based Singing Voice Synthesis (SVS) demonstrates strong expressiveness but remains limited by data scarcity and model scalability. We introduce a two-stage pipeline: a compact seed set of human-sung recordings is constructed by pairing fixed melodies with diverse LLM-generated lyrics, and melody-specific models are trained to synthesize over 500 hours of high-quality Chinese singing data. Building on this corpus, we propose DiTSinger, a Diffusion Transformer with RoPE and qk-norm, systematically scaled in depth, width, and resolution for enhanced fidelity. Furthermore, we design an implicit alignment mechanism that obviates phoneme-level duration labels by constraining phoneme-to-acoustic attention within character-level spans, thereby improving robustness under noisy or uncertain alignments. Extensive experiments validate that our approach enables scalable, alignment-free, and high-fidelity SVS.
>
---
#### [replaced 005] ESDD 2026: Environmental Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 提出EnvSDD数据集与ESDD挑战赛，解决环境音深伪检测难题，设双赛道应对真实场景中未知生成器与低资源黑盒情况。**

- **链接: [https://arxiv.org/pdf/2508.04529v2](https://arxiv.org/pdf/2508.04529v2)**

> **作者:** Han Yin; Yang Xiao; Rohan Kumar Das; Jisheng Bai; Ting Dang
>
> **摘要:** Recent advances in audio generation systems have enabled the creation of highly realistic and immersive soundscapes, which are increasingly used in film and virtual reality. However, these audio generators also raise concerns about potential misuse, such as generating deceptive audio content for fake videos and spreading misleading information. Existing datasets for environmental sound deepfake detection (ESDD) are limited in scale and audio types. To address this gap, we have proposed EnvSDD, the first large-scale curated dataset designed for ESDD, consisting of 45.25 hours of real and 316.7 hours of fake sound. Based on EnvSDD, we are launching the Environmental Sound Deepfake Detection Challenge. Specifically, we present two different tracks: ESDD in Unseen Generators and Black-Box Low-Resource ESDD, covering various challenges encountered in real-life scenarios. The challenge will be held in conjunction with the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026).
>
---
#### [replaced 006] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 论文评估SpeechLLMs在语音翻译中的效果，对比级联系统，发现后者更可靠，证明集成LLM对高质量翻译至关重要。**

- **链接: [https://arxiv.org/pdf/2512.16378v2](https://arxiv.org/pdf/2512.16378v2)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at https://github.com/sarapapi/hearing2translate
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [replaced 007] SingingSDS: A Singing-Capable Spoken Dialogue System for Conversational Roleplay Applications
- **分类: cs.SD**

- **简介: 提出SingingSDS系统，首个支持歌唱回复的对话系统，用于角色扮演与娱乐场景，通过ASR-LLM-SVS模块化架构实现可定制、低延迟、多风格歌唱交互。**

- **链接: [https://arxiv.org/pdf/2511.20972v2](https://arxiv.org/pdf/2511.20972v2)**

> **作者:** Jionghao Han; Jiatong Shi; Masao Someki; Yuxun Tang; Lan Liu; Yiwen Zhao; Wenhao Feng; Shinji Watanabe
>
> **摘要:** With recent advances in automatic speech recognition (ASR), large language models (LLMs), and text-to-speech (TTS) technologies, spoken dialogue systems (SDS) have become widely accessible. However, most existing SDS are limited to conventional spoken responses. We present SingingSDS, a cascaded SDS that responds through singing rather than speaking, fostering more affective, memorable, and pleasurable interactions in character-based roleplay and interactive entertainment scenarios. SingingSDS employs a modular ASR-LLM-SVS pipeline and supports a wide range of configurations across character personas, ASR and LLM backends, SVS models, melody sources, and voice profiles, tailored to different needs in terms of latency, quality, and musical style. SingingSDS is available as a plug-and-play web demo, featuring modular, open-source code that supports customization and extension. Demo: https://huggingface.co/spaces/espnet/SingingSDS. Code: https://github.com/SingingSDS/SingingSDS.
>
---
