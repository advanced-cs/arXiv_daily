# 音频 cs.SD;  eess.SP

- **最新发布 38 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Efficient and Microphone-Fault-Tolerant 3D Sound Source Localization
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出高效3D声源定位框架，解决传统方法计算成本高、依赖精准麦克风校准的问题。采用稀疏交叉注意力、预训练及自适应信号相干度量技术，实现少麦克风输入下的精准定位，并具备麦克风故障容错能力，实验验证其多声源定位扩展性，适用于动态或资源受限场景。**

- **链接: [http://arxiv.org/pdf/2505.20961v1](http://arxiv.org/pdf/2505.20961v1)**

> **作者:** Yiyuan Yang; Shitong Xu; Niki Trigoni; Andrew Markham
>
> **备注:** Accepted by Interspeech 2025 Conference
>
> **摘要:** Sound source localization (SSL) is a critical technology for determining the position of sound sources in complex environments. However, existing methods face challenges such as high computational costs and precise calibration requirements, limiting their deployment in dynamic or resource-constrained environments. This paper introduces a novel 3D SSL framework, which uses sparse cross-attention, pretraining, and adaptive signal coherence metrics, to achieve accurate and computationally efficient localization with fewer input microphones. The framework is also fault-tolerant to unreliable or even unknown microphone position inputs, ensuring its applicability in real-world scenarios. Preliminary experiments demonstrate its scalability for multi-source localization without requiring additional hardware. This work advances SSL by balancing the model's performance and efficiency and improving its robustness for real-world scenarios.
>
---
#### [new 002] Towards Robust Automated Perceptual Voice Quality Assessment with Deep Learning
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于自动语音质量评估任务，旨在解决传统主观评估的不一致和噪声敏感问题。提出VOQANet（结合深度学习与注意力机制）及VOQANet+（融合声学特征如抖动、 shimmer），通过整合语音基础模型与领域知识，提升评估鲁棒性与可解释性，适用于临床与远程医疗场景。**

- **链接: [http://arxiv.org/pdf/2505.21356v1](http://arxiv.org/pdf/2505.21356v1)**

> **作者:** Whenty Ariyanti; Kuan-Yu Chen; Sabato Marco Siniscalchi; Hsin-Min Wang; Yu Tsao
>
> **摘要:** Objective: Perceptual voice quality assessment plays a critical role in diagnosing and monitoring voice disorders by providing standardized evaluation of vocal function. Traditionally, this process relies on expert raters utilizing standard scales, such as the Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V) and Grade, Roughness, Breathiness, Asthenia, and Strain (GRBAS). However, these metrics are inherently subjective and susceptible to inter-rater variability, motivating the need for automated and objective assessment methods. Methods: We propose Voice Quality Assessment Network (VOQANet), a deep learning-based framework with an attention mechanism that leverages a Speech Foundation Model (SFM) to capture high-level acoustic and prosodic information from raw speech. To enhance robustness and interpretability, we present VOQANet+, which integrates handcrafted acoustic features such as jitter, shimmer, and harmonics-to-noise ratio (HNR) with SFM embeddings. Results: Sentence-based input yields stronger performance than vowel-based input, especially at the patient level. VOQANet consistently outperforms baseline methods in RMSE and PCC, while VOQANet+ performs even better and maintains robustness under noisy conditions. Conclusion: Combining SFM embeddings with domain-informed acoustic features improves interpretability and resilience. Significance: VOQANet+ shows strong potential for deployment in real-world and telehealth settings, addressing the limitations of subjective perceptual assessments with an interpretable and noise-resilient solution.
>
---
#### [new 003] Can Large Language Models Predict Audio Effects Parameters from Natural Language?
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出LLM2Fx框架，通过大语言模型将自然语言描述直接映射为音频效果（如均衡、混响）参数，解决非专业人士在音乐制作中调整参数的技术门槛问题。方法包括零样本生成及引入三种上下文示例（DSP特征、代码、示例）提升性能，效果优于传统优化方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20770v1](http://arxiv.org/pdf/2505.20770v1)**

> **作者:** Seungheon Doh; Junghyun Koo; Marco A. Martínez-Ramírez; Wei-Hsiang Liao; Juhan Nam; Yuki Mitsufuji
>
> **备注:** Submitted to WASPAA 2025
>
> **摘要:** In music production, manipulating audio effects (Fx) parameters through natural language has the potential to reduce technical barriers for non-experts. We present LLM2Fx, a framework leveraging Large Language Models (LLMs) to predict Fx parameters directly from textual descriptions without requiring task-specific training or fine-tuning. Our approach address the text-to-effect parameter prediction (Text2Fx) task by mapping natural language descriptions to the corresponding Fx parameters for equalization and reverberation. We demonstrate that LLMs can generate Fx parameters in a zero-shot manner that elucidates the relationship between timbre semantics and audio effects in music production. To enhance performance, we introduce three types of in-context examples: audio Digital Signal Processing (DSP) features, DSP function code, and few-shot examples. Our results demonstrate that LLM-based Fx parameter generation outperforms previous optimization approaches, offering competitive performance in translating natural language descriptions to appropriate Fx settings. Furthermore, LLMs can serve as text-driven interfaces for audio production, paving the way for more intuitive and accessible music production tools.
>
---
#### [new 004] Foundation Model Hidden Representations for Heart Rate Estimation from Auscultation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究利用预训练声学基础模型（如CLAP、HuBERT等）从心音听诊数据中估计心率。任务为探索这些模型的隐藏表征对心率估计的适用性，解决其编码效果及领域差异问题。工作包括对六种模型进行逐层分析，对比传统基线方法，发现内部CLAP模型表征在MAE指标上表现最优。**

- **链接: [http://arxiv.org/pdf/2505.20745v1](http://arxiv.org/pdf/2505.20745v1)**

> **作者:** Jingping Nie; Dung T. Tran; Karan Thakkar; Vasudha Kowtha; John Huang; Carlos Avendano; Erdrin Azemi; Vikramjit Mitra
>
> **备注:** 5 pages, Interspeech 2025 conference
>
> **摘要:** Auscultation, particularly heart sound, is a non-invasive technique that provides essential vital sign information. Recently, self-supervised acoustic representation foundation models (FMs) have been proposed to offer insights into acoustics-based vital signs. However, there has been little exploration of the extent to which auscultation is encoded in these pre-trained FM representations. In this work, using a publicly available phonocardiogram (PCG) dataset and a heart rate (HR) estimation model, we conduct a layer-wise investigation of six acoustic representation FMs: HuBERT, wav2vec2, wavLM, Whisper, Contrastive Language-Audio Pretraining (CLAP), and an in-house CLAP model. Additionally, we implement the baseline method from Nie et al., 2024 (which relies on acoustic features) and show that overall, representation vectors from pre-trained foundation models (FMs) offer comparable performance to the baseline. Notably, HR estimation using the representations from the audio encoder of the in-house CLAP model outperforms the results obtained from the baseline, achieving a lower mean absolute error (MAE) across various train/validation/test splits despite the domain mismatch.
>
---
#### [new 005] Training Articulatory Inversion Models for Inter-Speaker Consistency
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于声学-发音逆向建模任务，旨在解决跨说话人发音预测一致性问题。通过比较单/多说话人自监督模型，提出基于最小对集的评估方法及新训练方法提升跨语言（英、俄）发音目标一致性。**

- **链接: [http://arxiv.org/pdf/2505.20529v1](http://arxiv.org/pdf/2505.20529v1)**

> **作者:** Charles McGhee; Mark J. F. Gales; Kate M. Knill
>
> **摘要:** Acoustic-to-Articulatory Inversion (AAI) attempts to model the inverse mapping from speech to articulation. Exact articulatory prediction from speech alone may be impossible, as speakers can choose different forms of articulation seemingly without reference to their vocal tract structure. However, once a speaker has selected an articulatory form, their productions vary minimally. Recent works in AAI have proposed adapting Self-Supervised Learning (SSL) models to single-speaker datasets, claiming that these single-speaker models provide a universal articulatory template. In this paper, we investigate whether SSL-adapted models trained on single and multi-speaker data produce articulatory targets which are consistent across speaker identities for English and Russian. We do this through the use of a novel evaluation method which extracts articulatory targets using minimal pair sets. We also present a training method which can improve inter-speaker consistency using only speech data.
>
---
#### [new 006] BrainStratify: Coarse-to-Fine Disentanglement of Intracranial Neural Dynamics
- **分类: eess.SP; cs.CL; q-bio.NC**

- **简介: 该论文属于脑机接口语音解码任务，针对颅内神经信号稀疏分布及与无关信号纠缠问题，提出分层框架BrainStratify：先通过空间-时间建模识别功能组，再用DPQ解耦目标神经动态。实验显示其显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20480v1](http://arxiv.org/pdf/2505.20480v1)**

> **作者:** Hui Zheng; Hai-Teng Wang; Yi-Tao Jing; Pei-Yang Lin; Han-Qing Zhao; Wei Chen; Peng-Hu Wei; Yong-Zhi Shan; Guo-Guang Zhao; Yun-Zhe Liu
>
> **摘要:** Decoding speech directly from neural activity is a central goal in brain-computer interface (BCI) research. In recent years, exciting advances have been made through the growing use of intracranial field potential recordings, such as stereo-ElectroEncephaloGraphy (sEEG) and ElectroCorticoGraphy (ECoG). These neural signals capture rich population-level activity but present key challenges: (i) task-relevant neural signals are sparsely distributed across sEEG electrodes, and (ii) they are often entangled with task-irrelevant neural signals in both sEEG and ECoG. To address these challenges, we introduce a unified Coarse-to-Fine neural disentanglement framework, BrainStratify, which includes (i) identifying functional groups through spatial-context-guided temporal-spatial modeling, and (ii) disentangling distinct neural dynamics within the target functional group using Decoupled Product Quantization (DPQ). We evaluate BrainStratify on two open-source sEEG datasets and one (epidural) ECoG dataset, spanning tasks like vocal production and speech perception. Extensive experiments show that BrainStratify, as a unified framework for decoding speech from intracranial neural signals, significantly outperforms previous decoding methods. Overall, by combining data-driven stratification with neuroscience-inspired modularity, BrainStratify offers a robust and interpretable solution for speech decoding from intracranial recordings.
>
---
#### [new 007] VibE-SVC: Vibrato Extraction with High-frequency F0 Contour for Singing Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于歌唱语音转换任务，解决颤音动态建模与可控性难题。提出VibE-SVC模型，利用离散小波分解F0轮廓的高频成分，显式提取并操纵颤音，实现风格转换时的精准控制与音色保真。实验验证其有效提升转换灵活性与质量。**

- **链接: [http://arxiv.org/pdf/2505.20794v1](http://arxiv.org/pdf/2505.20794v1)**

> **作者:** Joon-Seung Choi; Dong-Min Byun; Hyung-Seok Oh; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Controlling singing style is crucial for achieving an expressive and natural singing voice. Among the various style factors, vibrato plays a key role in conveying emotions and enhancing musical depth. However, modeling vibrato remains challenging due to its dynamic nature, making it difficult to control in singing voice conversion. To address this, we propose VibESVC, a controllable singing voice conversion model that explicitly extracts and manipulates vibrato using discrete wavelet transform. Unlike previous methods that model vibrato implicitly, our approach decomposes the F0 contour into frequency components, enabling precise transfer. This allows vibrato control for enhanced flexibility. Experimental results show that VibE-SVC effectively transforms singing styles while preserving speaker similarity. Both subjective and objective evaluations confirm high-quality conversion.
>
---
#### [new 008] Music's Multimodal Complexity in AVQA: Why We Need More than General Multimodal LLMs
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐视听问答（Music AVQA）任务，针对通用多模态模型在处理音乐连续音频-视觉内容、复杂时序动态及领域知识时的不足，通过分析现有数据集与方法，提出需专用输入处理、时空架构及音乐建模策略，并指明融合音乐先验知识的未来方向，提供相关研究资源库。**

- **链接: [http://arxiv.org/pdf/2505.20638v1](http://arxiv.org/pdf/2505.20638v1)**

> **作者:** Wenhao You; Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Zhongyu Ouyang; Chiyu Ma; Tingxuan Wu; Noah Wei; Zong Ke; Ming Cheng; Soroush Vosoughi; Jiang Gui
>
> **摘要:** While recent Multimodal Large Language Models exhibit impressive capabilities for general multimodal tasks, specialized domains like music necessitate tailored approaches. Music Audio-Visual Question Answering (Music AVQA) particularly underscores this, presenting unique challenges with its continuous, densely layered audio-visual content, intricate temporal dynamics, and the critical need for domain-specific knowledge. Through a systematic analysis of Music AVQA datasets and methods, this position paper identifies that specialized input processing, architectures incorporating dedicated spatial-temporal designs, and music-specific modeling strategies are critical for success in this domain. Our study provides valuable insights for researchers by highlighting effective design patterns empirically linked to strong performance, proposing concrete future directions for incorporating musical priors, and aiming to establish a robust foundation for advancing multimodal musical understanding. This work is intended to inspire broader attention and further research, supported by a continuously updated anonymous GitHub repository of relevant papers: https://github.com/xid32/Survey4MusicAVQA.
>
---
#### [new 009] ClearSphere: Multi-Earphone Synergy for Enhanced Conversational Clarity
- **分类: cs.SD**

- **简介: 该论文提出ClearSphere系统，属于语音增强任务，解决嘈杂环境中多人对话清晰度问题。通过多耳机关联协作，设计对话驱动网络协议实现设备协同，结合高效语音提取模型，实时分离目标对话。实验显示其分组准确率超90%，提升8.8dB音质，且具备移动端实时性能。**

- **链接: [http://arxiv.org/pdf/2505.21004v1](http://arxiv.org/pdf/2505.21004v1)**

> **作者:** Lixing He
>
> **摘要:** In crowded places such as conferences, background noise, overlapping voices, and lively interactions make it difficult to have clear conversations. This situation often worsens the phenomenon known as "cocktail party deafness." We present ClearSphere, the collaborative system that enhances speech at the conversation level with multi-earphones. Real-time conversation enhancement requires a holistic modeling of all the members in the conversation, and an effective way to extract the speech from the mixture. ClearSphere bridges the acoustic sensor system and state-of-the-art deep learning for target speech extraction by making two key contributions: 1) a conversation-driven network protocol, and 2) a robust target conversation extraction model. Our networking protocol enables mobile, infrastructure-free coordination among earphone devices. Our conversation extraction model can leverage the relay audio in a bandwidth-efficient way. ClearSphere is evaluated in both real-world experiments and simulations. Results show that our conversation network obtains more than 90\% accuracy in group formation, improves the speech quality by up to 8.8 dB over state-of-the-art baselines, and demonstrates real-time performance on a mobile device. In a user study with 20 participants, ClearSphere has a much higher score than baseline with good usability.
>
---
#### [new 010] Unfolding A Few Structures for The Many: Memory-Efficient Compression of Conformer and Speech Foundation Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一种内存高效压缩方法，针对Conformer语音模型及语音基础模型。旨在减少参数量同时保持性能，通过训练紧凑"种子模型"并展开为多深度逻辑模型，联合训练并利用自蒸馏缩小性能差距，实现参数减少35%/30%且性能无损。**

- **链接: [http://arxiv.org/pdf/2505.21237v1](http://arxiv.org/pdf/2505.21237v1)**

> **作者:** Zhaoqing Li; Haoning Xu; Xurong Xie; Zengrui Jin; Tianzi Wang; Xunying Liu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** This paper presents a novel memory-efficient model compression approach for Conformer ASR and speech foundation systems. Our approach features a unique "small-to-large" design. A compact "seed" model containing a few Conformer or Transformer blocks is trained and unfolded many times to emulate the performance of larger uncompressed models with different logical depths. The seed model and many unfolded paths are jointly trained within a single unfolding cycle. The KL-divergence between the largest unfolded and smallest seed models is used in a self-distillation process to minimize their performance disparity. Experimental results show that our foldable model produces ASR performance comparable to individually constructed Conformer and wav2vec2/HuBERT speech foundation models under various depth configurations, while requiring only minimal memory and storage. Conformer and wav2vec2 models with a reduction of 35% and 30% parameters are obtained without loss of performance, respectively.
>
---
#### [new 011] VoxAging: Continuously Tracking Speaker Aging with a Large-Scale Longitudinal Dataset in English and Mandarin
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 论文提出VoxAging数据集，解决说话人验证中年龄影响的难题。该数据集包含293名英、汉双语者多年每周录音（最长17年），用于研究说话人老化对系统性能的影响，分析个体老化差异及年龄、性别等因素的作用。（98字）**

- **链接: [http://arxiv.org/pdf/2505.21445v1](http://arxiv.org/pdf/2505.21445v1)**

> **作者:** Zhiqi Ai; Meixuan Bao; Zhiyong Chen; Zhi Yang; Xinnuo Li; Shugong Xu
>
> **备注:** 5 pages, 4 figures, Accepted by Interspeech 2025
>
> **摘要:** The performance of speaker verification systems is adversely affected by speaker aging. However, due to challenges in data collection, particularly the lack of sustained and large-scale longitudinal data for individuals, research on speaker aging remains difficult. In this paper, we present VoxAging, a large-scale longitudinal dataset collected from 293 speakers (226 English speakers and 67 Mandarin speakers) over several years, with the longest time span reaching 17 years (approximately 900 weeks). For each speaker, the data were recorded at weekly intervals. We studied the phenomenon of speaker aging and its effects on advanced speaker verification systems, analyzed individual speaker aging processes, and explored the impact of factors such as age group and gender on speaker aging research.
>
---
#### [new 012] Model as Loss: A Self-Consistent Training Paradigm
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.SP**

- **简介: 该论文提出Model as Loss训练范式，用于语音增强任务。针对传统损失函数无法有效捕捉关键信号特征的问题，利用模型自身编码器的特征空间构建损失函数，强制输出与清洁语音自洽。实验显示其在感知质量和泛化性上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.21156v1](http://arxiv.org/pdf/2505.21156v1)**

> **作者:** Saisamarth Rajesh Phaye; Milos Cernak; Andrew Harper
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** Conventional methods for speech enhancement rely on handcrafted loss functions (e.g., time or frequency domain losses) or deep feature losses (e.g., using WavLM or wav2vec), which often fail to capture subtle signal properties essential for optimal performance. To address this, we propose Model as Loss, a novel training paradigm that utilizes the encoder from the same model as a loss function to guide the training. The Model as Loss paradigm leverages the encoder's task-specific feature space, optimizing the decoder to produce output consistent with perceptual and task-relevant characteristics of the clean signal. By using the encoder's learned features as a loss function, this framework enforces self-consistency between the clean reference speech and the enhanced model output. Our approach outperforms pre-trained deep feature losses on standard speech enhancement benchmarks, offering better perceptual quality and robust generalization to both in-domain and out-of-domain datasets.
>
---
#### [new 013] Text-Queried Audio Source Separation via Hierarchical Modeling
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于文本查询驱动的音频源分离任务，针对现有方法跨模态对齐效率低与依赖大规模标注数据的问题，提出分层框架HSM-TSS，通过全局-局部语义分阶段分离及结构化声学重建，并设计指令解析模块，实现高效训练与语义一致性的音频提取。**

- **链接: [http://arxiv.org/pdf/2505.21025v1](http://arxiv.org/pdf/2505.21025v1)**

> **作者:** Xinlei Yin; Xiulian Peng; Xue Jiang; Zhiwei Xiong; Yan Lu
>
> **摘要:** Target audio source separation with natural language queries presents a promising paradigm for extracting arbitrary audio events through arbitrary text descriptions. Existing methods mainly face two challenges, the difficulty in jointly modeling acoustic-textual alignment and semantic-aware separation within a blindly-learned single-stage architecture, and the reliance on large-scale accurately-labeled training data to compensate for inefficient cross-modal learning and separation. To address these challenges, we propose a hierarchical decomposition framework, HSM-TSS, that decouples the task into global-local semantic-guided feature separation and structure-preserving acoustic reconstruction. Our approach introduces a dual-stage mechanism for semantic separation, operating on distinct global and local semantic feature spaces. We first perform global-semantic separation through a global semantic feature space aligned with text queries. A Q-Audio architecture is employed to align audio and text modalities, serving as pretrained global-semantic encoders. Conditioned on the predicted global feature, we then perform the second-stage local-semantic separation on AudioMAE features that preserve time-frequency structures, followed by acoustic reconstruction. We also propose an instruction processing pipeline to parse arbitrary text queries into structured operations, extraction or removal, coupled with audio descriptions, enabling flexible sound manipulation. Our method achieves state-of-the-art separation performance with data-efficient training while maintaining superior semantic consistency with queries in complex auditory scenes.
>
---
#### [new 014] Uni-VERSA: Versatile Speech Assessment with a Unified Network
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决现有主观测试成本高及客观指标片面性问题。提出Uni-VERSA统一网络，同时预测自然度、可懂度等多维度指标，构建评估框架并验证其有效性，适用于语音增强、合成等场景，与人类感知高度一致。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20741v1](http://arxiv.org/pdf/2505.20741v1)**

> **作者:** Jiatong Shi; Hye-Jin Shim; Shinji Watanabe
>
> **备注:** Accepted by Interspeech
>
> **摘要:** Subjective listening tests remain the golden standard for speech quality assessment, but are costly, variable, and difficult to scale. In contrast, existing objective metrics, such as PESQ, F0 correlation, and DNSMOS, typically capture only specific aspects of speech quality. To address these limitations, we introduce Uni-VERSA, a unified network that simultaneously predicts various objective metrics, encompassing naturalness, intelligibility, speaker characteristics, prosody, and noise, for a comprehensive evaluation of speech signals. We formalize its framework, evaluation protocol, and applications in speech enhancement, synthesis, and quality control. A benchmark based on the URGENT24 challenge, along with a baseline leveraging self-supervised representations, demonstrates that Uni-VERSA provides a viable alternative to single-aspect evaluation methods. Moreover, it aligns closely with human perception, making it a promising approach for future speech quality assessment.
>
---
#### [new 015] Universal Speech Enhancement with Regression and Generative Mamba
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决跨多种失真类型和语言的通用鲁棒增强问题。提出USEMamba模型，结合回归（处理多数失真）与生成模型（修复包丢失/带宽扩展等缺失内容），通过状态空间架构实现长时建模与采样率无关特征提取，在仅用部分数据训练下获挑战赛第二名。**

- **链接: [http://arxiv.org/pdf/2505.21198v1](http://arxiv.org/pdf/2505.21198v1)**

> **作者:** Rong Chao; Rauf Nasretdinov; Yu-Chiang Frank Wang; Ante Jukić; Szu-Wei Fu; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The Interspeech 2025 URGENT Challenge aimed to advance universal, robust, and generalizable speech enhancement by unifying speech enhancement tasks across a wide variety of conditions, including seven different distortion types and five languages. We present Universal Speech Enhancement Mamba (USEMamba), a state-space speech enhancement model designed to handle long-range sequence modeling, time-frequency structured processing, and sampling frequency-independent feature extraction. Our approach primarily relies on regression-based modeling, which performs well across most distortions. However, for packet loss and bandwidth extension, where missing content must be inferred, a generative variant of the proposed USEMamba proves more effective. Despite being trained on only a subset of the full training data, USEMamba achieved 2nd place in Track 1 during the blind test phase, demonstrating strong generalization across diverse conditions.
>
---
#### [new 016] Hybrid Disagreement-Diversity Active Learning for Bioacoustic Sound Event Detection
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于生物声学声音事件检测（BioSED）任务，旨在解决标注数据稀缺、事件稀疏、物种多样及类别不平衡等挑战。提出结合分歧投票与多样性分析的MFFT主动学习方法，并优化数据集评估，实验证明其以极少标注量（2.3%）高效提升模型性能，尤其在冷启动和稀有物种检测中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.20956v1](http://arxiv.org/pdf/2505.20956v1)**

> **作者:** Shiqi Zhang; Tuomas Virtanen
>
> **备注:** 5 pages, 1 figure, accepted by EUSIPCO 2025
>
> **摘要:** Bioacoustic sound event detection (BioSED) is crucial for biodiversity conservation but faces practical challenges during model development and training: limited amounts of annotated data, sparse events, species diversity, and class imbalance. To address these challenges efficiently with a limited labeling budget, we apply the mismatch-first farthest-traversal (MFFT), an active learning method integrating committee voting disagreement and diversity analysis. We also refine an existing BioSED dataset specifically for evaluating active learning algorithms. Experimental results demonstrate that MFFT achieves a mAP of 68% when cold-starting and 71% when warm-starting (which is close to the fully-supervised mAP of 75%) while using only 2.3% of the annotations. Notably, MFFT excels in cold-start scenarios and with rare species, which are critical for monitoring endangered species, demonstrating its practical value.
>
---
#### [new 017] MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出MelodySim模型，用于音乐抄袭检测任务。针对现有方法难以捕捉旋律相似性的问题，构建了通过音符分割、分解和声等增强Slakh2100生成的旋律相似数据集，并设计基于MERT编码器和三元组网络的分段相似性模型，通过决策矩阵定位抄袭，测试集表现优异。**

- **链接: [http://arxiv.org/pdf/2505.20979v1](http://arxiv.org/pdf/2505.20979v1)**

> **作者:** Tongyu Lu; Charlotta-Marlena Geist; Jan Melechovsky; Abhinaba Roy; Dorien Herremans
>
> **摘要:** We propose MelodySim, a melody-aware music similarity model and dataset for plagiarism detection. First, we introduce a novel method to construct a dataset with focus on melodic similarity. By augmenting Slakh2100; an existing MIDI dataset, we generate variations of each piece while preserving the melody through modifications such as note splitting, arpeggiation, minor track dropout (excluding bass), and re-instrumentation. A user study confirms that positive pairs indeed contain similar melodies, with other musical tracks significantly changed. Second, we develop a segment-wise melodic-similarity detection model that uses a MERT encoder and applies a triplet neural network to capture melodic similarity. The resultant decision matrix highlights where plagiarism might occur. Our model achieves high accuracy on the MelodySim test set.
>
---
#### [new 018] Towards One-bit ASR: Extremely Low-bit Conformer Quantization Using Co-training and Stochastic Precision
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于ASR模型压缩任务，解决极低比特量化（如1/2位）导致性能下降问题。提出多精度协同训练、随机精度及张量级可学习缩放因子方法，在Switchboard和LibriSpeech数据集上实现接近无损的1/2位量化，压缩比达16倍，WER无显著上升。**

- **链接: [http://arxiv.org/pdf/2505.21245v1](http://arxiv.org/pdf/2505.21245v1)**

> **作者:** Zhaoqing Li; Haoning Xu; Zengrui Jin; Lingwei Meng; Tianzi Wang; Huimeng Wang; Youjun Chen; Mingyu Cui; Shujie Hu; Xunying Liu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** Model compression has become an emerging need as the sizes of modern speech systems rapidly increase. In this paper, we study model weight quantization, which directly reduces the memory footprint to accommodate computationally resource-constrained applications. We propose novel approaches to perform extremely low-bit (i.e., 2-bit and 1-bit) quantization of Conformer automatic speech recognition systems using multiple precision model co-training, stochastic precision, and tensor-wise learnable scaling factors to alleviate quantization incurred performance loss. The proposed methods can achieve performance-lossless 2-bit and 1-bit quantization of Conformer ASR systems trained with the 300-hr Switchboard and 960-hr LibriSpeech corpus. Maximum overall performance-lossless compression ratios of 16.2 and 16.6 times are achieved without a statistically significant increase in the word error rate (WER) over the full precision baseline systems, respectively.
>
---
#### [new 019] Spotlight-TTS: Spotlighting the Style via Voiced-Aware Style Extraction and Style Direction Adjustment for Expressive Text-to-Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于表达式文本到语音合成任务，旨在解决现有方法在风格提取和语音质量上的不足。提出Spotlight-TTS，通过聚焦有声区域的风格提取保持风格连续性，并调整风格方向优化模型集成，提升表达力与合成质量。**

- **链接: [http://arxiv.org/pdf/2505.20868v1](http://arxiv.org/pdf/2505.20868v1)**

> **作者:** Nam-Gyu Kim; Deok-Hyeon Cho; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Submitted to Interspeech
>
> **摘要:** Recent advances in expressive text-to-speech (TTS) have introduced diverse methods based on style embedding extracted from reference speech. However, synthesizing high-quality expressive speech remains challenging. We propose Spotlight-TTS, which exclusively emphasizes style via voiced-aware style extraction and style direction adjustment. Voiced-aware style extraction focuses on voiced regions highly related to style while maintaining continuity across different speech regions to improve expressiveness. We adjust the direction of the extracted style for optimal integration into the TTS model, which improves speech quality. Experimental results demonstrate that Spotlight-TTS achieves superior performance compared to baseline models in terms of expressiveness, overall speech quality, and style transfer capability. Our audio samples are publicly available.
>
---
#### [new 020] PromptEVC: Controllable Emotional Voice Conversion with Natural Language Prompts
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出PromptEVC，属于可控情感语音转换任务。解决现有方法依赖固定标签或参考音频、忽视个体情感差异的问题。通过自然语言提示生成精细情感嵌入，结合韵律控制与说话人编码器，实现灵活的情感表达与自然语音合成，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20678v1](http://arxiv.org/pdf/2505.20678v1)**

> **作者:** Tianhua Qi; Shiyan Wang; Cheng Lu; Tengfei Song; Hao Yang; Zhanglin Wu; Wenming Zheng
>
> **备注:** Accepted to INTERSPEECH2025
>
> **摘要:** Controllable emotional voice conversion (EVC) aims to manipulate emotional expressions to increase the diversity of synthesized speech. Existing methods typically rely on predefined labels, reference audios, or prespecified factor values, often overlooking individual differences in emotion perception and expression. In this paper, we introduce PromptEVC that utilizes natural language prompts for precise and flexible emotion control. To bridge text descriptions with emotional speech, we propose emotion descriptor and prompt mapper to generate fine-grained emotion embeddings, trained jointly with reference embeddings. To enhance naturalness, we present a prosody modeling and control pipeline that adjusts the rhythm based on linguistic content and emotional cues. Additionally, a speaker encoder is incorporated to preserve identity. Experimental results demonstrate that PromptEVC outperforms state-of-the-art controllable EVC methods in emotion conversion, intensity control, mixed emotion synthesis, and prosody manipulation. Speech samples are available at https://jeremychee4.github.io/PromptEVC/.
>
---
#### [new 021] PSRB: A Comprehensive Benchmark for Evaluating Persian ASR Systems
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出PSRB基准，用于评估波斯语ASR系统，解决低资源语言评估难题。通过测试10个ASR模型，分析错误类型并提出加权替换误差指标，揭示模型在方言、儿童语音等场景表现不佳，强调需多样化数据与微调优化。**

- **链接: [http://arxiv.org/pdf/2505.21230v1](http://arxiv.org/pdf/2505.21230v1)**

> **作者:** Nima Sedghiyeh; Sara Sadeghi; Reza Khodadadi; Farzin Kashani; Omid Aghdaei; Somayeh Rahimi; Mohammad Sadegh Safari
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Although Automatic Speech Recognition (ASR) systems have become an integral part of modern technology, their evaluation remains challenging, particularly for low-resource languages such as Persian. This paper introduces Persian Speech Recognition Benchmark(PSRB), a comprehensive benchmark designed to address this gap by incorporating diverse linguistic and acoustic conditions. We evaluate ten ASR systems, including state-of-the-art commercial and open-source models, to examine performance variations and inherent biases. Additionally, we conduct an in-depth analysis of Persian ASR transcriptions, identifying key error types and proposing a novel metric that weights substitution errors. This metric enhances evaluation robustness by reducing the impact of minor and partial errors, thereby improving the precision of performance assessment. Our findings indicate that while ASR models generally perform well on standard Persian, they struggle with regional accents, children's speech, and specific linguistic challenges. These results highlight the necessity of fine-tuning and incorporating diverse, representative training datasets to mitigate biases and enhance overall ASR performance. PSRB provides a valuable resource for advancing ASR research in Persian and serves as a framework for developing benchmarks in other low-resource languages. A subset of the PSRB dataset is publicly available at https://huggingface.co/datasets/PartAI/PSRB.
>
---
#### [new 022] Effect of laboratory conditions on the perception of virtual stages for music
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于虚拟声学环境评估任务，旨在解决实验室声学条件对音乐虚拟舞台感知可靠性的影响问题。通过对比无回声室及两种自制听音室（隔音充分/不足）的声学效果，发现隔音不足环境会使虚拟声音感知过强，验证了实验室设计标准的必要性，为后续研究提供方法基础。**

- **链接: [http://arxiv.org/pdf/2505.20552v1](http://arxiv.org/pdf/2505.20552v1)**

> **作者:** Ernesto Accolti
>
> **摘要:** This manuscript presents initial findings critical for supporting augmented acoustics experiments in custom-made hearing booths, addressing a key challenge in ensuring perceptual validity and experimental rigor in these highly sensitive setups. This validation ensures our proposed methodology is sound, guarantees the reliability of future results, and lays the foundational groundwork for subsequent perceptual studies and the development of robust guidelines for laboratory design in virtual acoustics research. A preliminary study on the effect of the acoustical conditions of three different rooms on the perception of virtual stages for music is presented: an anechoic room, a custom-made hearing booth with insufficient sound absorption, and another custom-made hearing booth with achievable sound absorption. The goal of this study is to assess the impact of these different conditions on the perception of virtual stages for music. The results show that the anechoic room and the hearing booth with achievable sound absorption have a difference between the total sound and the virtual sound below the just-noticeable difference, which means that the virtual sound is not perceived louder than it should. In contrast, the hearing booth with insufficient sound absorption has a difference above the just-noticeable difference, which means that the virtual sound is perceived louder than it should. This study provides a preliminary validation of the proposed methodology for assessing the acoustical conditions of custom-made hearing booths in stage acoustics experiments. Future work will include a more comprehensive analysis of the results, including the effect of different sound sources.
>
---
#### [new 023] Study of Lightweight Transformer Architectures for Single-Channel Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究单通道语音增强任务，针对边缘设备计算限制与模型复杂度的矛盾，提出轻量级因果Transformer架构LCT-GAN。通过FTF堆叠结构和对抗训练，减少参数与计算量（较DeepFilterNet2参数仅6%），在主流指标上达SotA性能，优于同类模型。（98字）**

- **链接: [http://arxiv.org/pdf/2505.21057v1](http://arxiv.org/pdf/2505.21057v1)**

> **作者:** Haixin Zhao; Nilesh Madhu
>
> **备注:** Accepted by EUSIPCO 2025
>
> **摘要:** In speech enhancement, achieving state-of-the-art (SotA) performance while adhering to the computational constraints on edge devices remains a formidable challenge. Networks integrating stacked temporal and spectral modelling effectively leverage improved architectures such as transformers; however, they inevitably incur substantial computational complexity and model expansion. Through systematic ablation analysis on transformer-based temporal and spectral modelling, we demonstrate that the architecture employing streamlined Frequency-Time-Frequency (FTF) stacked transformers efficiently learns global dependencies within causal context, while avoiding considerable computational demands. Utilising discriminators in training further improves learning efficacy and enhancement without introducing additional complexity during inference. The proposed lightweight, causal, transformer-based architecture with adversarial training (LCT-GAN) yields SoTA performance on instrumental metrics among contemporary lightweight models, but with far less overhead. Compared to DeepFilterNet2, the LCT-GAN only requires 6% of the parameters, at similar complexity and performance. Against CCFNet+(Lite), LCT-GAN saves 9% in parameters and 10% in multiply-accumulate operations yet yielding improved performance. Further, the LCT-GAN even outperforms more complex, common baseline models on widely used test datasets.
>
---
#### [new 024] REWIND: Speech Time Reversal for Enhancing Speaker Representations in Diffusion-based Voice Conversion
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文属于语音转换（VC）任务，旨在解决说话人与语言表征纠缠问题。提出通过时间反转语音信号保留语调但破坏语言内容，利用反转语音增强说话人表征，作为数据增强策略优化扩散模型。实验表明，该方法显著提升说话人相似度同时保持高质量语音。**

- **链接: [http://arxiv.org/pdf/2505.20756v1](http://arxiv.org/pdf/2505.20756v1)**

> **作者:** Ishan D. Biyani; Nirmesh J. Shah; Ashishkumar P. Gudmalwar; Pankaj Wasnik; Rajiv R. Shah
>
> **备注:** Accepted in INTERSPEECH 2025
>
> **摘要:** Speech time reversal refers to the process of reversing the entire speech signal in time, causing it to play backward. Such signals are completely unintelligible since the fundamental structures of phonemes and syllables are destroyed. However, they still retain tonal patterns that enable perceptual speaker identification despite losing linguistic content. In this paper, we propose leveraging speaker representations learned from time reversed speech as an augmentation strategy to enhance speaker representation. Notably, speaker and language disentanglement in voice conversion (VC) is essential to accurately preserve a speaker's unique vocal traits while minimizing interference from linguistic content. The effectiveness of the proposed approach is evaluated in the context of state-of-the-art diffusion-based VC models. Experimental results indicate that the proposed approach significantly improves speaker similarity-related scores while maintaining high speech quality.
>
---
#### [new 025] Techniques for Quantum-Computing-Aided Algorithmic Composition: Experiments in Rhythm, Timbre, Harmony, and Space
- **分类: quant-ph; cs.ET; cs.SD; eess.AS**

- **简介: 该论文属于量子计算与音乐生成交叉任务，旨在通过量子技术增强算法作曲的控制能力。提出四项技术：量子模拟辅助创作决策、粒子追踪生成噪声音色、基向量旋转调控和声概率、测量误差扰动空间声场，开发相应算法与软件并提供音乐实例验证。**

- **链接: [http://arxiv.org/pdf/2505.20565v1](http://arxiv.org/pdf/2505.20565v1)**

> **作者:** Christopher Dobrian; Omar Costa Hamido
>
> **摘要:** Quantum computing can be employed in computer-aided music composition to control various attributes of the music at different structural levels. This article describes the application of quantum simulation to model compositional decision making, the simulation of quantum particle tracking to produce noise-based timbres, the use of basis state vector rotation to cause changing probabilistic behaviors in granular harmonic textures, and the exploitation of quantum measurement error to cause noisy perturbations of spatial soundpaths. We describe the concepts fundamental to these techniques, we provide algorithms and software enacting them, and we provide examples demonstrating their implementation in computer-generated music.
>
---
#### [new 026] Multimodal Assessment of Speech Impairment in ALS Using Audio-Visual and Machine Learning Approaches
- **分类: eess.AS; cs.SD**

- **简介: 该论文通过结合音频-视频特征与机器学习（如XGBoost回归），开发回归模型预测ALS患者的言语障碍程度，解决临床评估主观性高且成本昂贵的问题。实验显示最佳模型RMSE为0.93，证实多模态方法可提升评估客观性，支持早期检测及家庭监测。**

- **链接: [http://arxiv.org/pdf/2505.21093v1](http://arxiv.org/pdf/2505.21093v1)**

> **作者:** Francesco Pierotti; Andrea Bandini
>
> **备注:** Submitted to Interspeech
>
> **摘要:** The analysis of speech in individuals with amyotrophic lateral sclerosis is a powerful tool to support clinicians in the assessment of bulbar dysfunction. However, current methods used in clinical practice consist of subjective evaluations or expensive instrumentation. This study investigates different approaches combining audio-visual analysis and machine learning to predict the speech impairment evaluation performed by clinicians. Using a small dataset of acoustic and kinematic features extracted from audio and video recordings of speech tasks, we trained and tested some regression models. The best performance was achieved using the extreme boosting machine regressor with multimodal features, which resulted in a root mean squared error of 0.93 on a scale ranging from 5 to 25. Results suggest that integrating audio-video analysis enhances speech impairment assessment, providing an objective tool for early detection and monitoring of bulbar dysfunction, also in home settings.
>
---
#### [new 027] Plug-and-Play Co-Occurring Face Attention for Robust Audio-Visual Speaker Extraction
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于音频视觉说话者提取任务，旨在解决多 person 环境中共现人脸干扰导致的分离精度下降问题。提出插拔式跨说话者注意力模块，处理灵活数量的共现人脸，集成至 AV-DPRNN 和 AV-TFGridNet 模型。实验显示在多个数据集上优于基线方法。（98字）**

- **链接: [http://arxiv.org/pdf/2505.20635v1](http://arxiv.org/pdf/2505.20635v1)**

> **作者:** Zexu Pan; Shengkui Zhao; Tingting Wang; Kun Zhou; Yukun Ma; Chong Zhang; Bin Ma
>
> **备注:** Interspeech 2025
>
> **摘要:** Audio-visual speaker extraction isolates a target speaker's speech from a mixture speech signal conditioned on a visual cue, typically using the target speaker's face recording. However, in real-world scenarios, other co-occurring faces are often present on-screen, providing valuable speaker activity cues in the scene. In this work, we introduce a plug-and-play inter-speaker attention module to process these flexible numbers of co-occurring faces, allowing for more accurate speaker extraction in complex multi-person environments. We integrate our module into two prominent models: the AV-DPRNN and the state-of-the-art AV-TFGridNet. Extensive experiments on diverse datasets, including the highly overlapped VoxCeleb2 and sparsely overlapped MISP, demonstrate that our approach consistently outperforms baselines. Furthermore, cross-dataset evaluations on LRS2 and LRS3 confirm the robustness and generalizability of our method.
>
---
#### [new 028] Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦中文方言及口音的语音识别任务，针对其数据稀缺问题，提出结合自监督预训练（Data2vec2）与LLM的方法。通过30万小时无标签方言数据预训练及4万小时监督数据对齐训练，系统评估不同投影器和LLM对识别效果的影响，实现多数据集SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.21138v1](http://arxiv.org/pdf/2505.21138v1)**

> **作者:** Tianyi Xu; Hongjie Chen; Wang Qing; Lv Hang; Jian Kang; Li Jie; Zhennan Lin; Yongxiang Li; Xie Lei
>
> **摘要:** Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre- training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
>
---
#### [new 029] Phir Hera Fairy: An English Fairytaler is a Strong Faker of Fluent Speech in Low-Resource Indian Languages
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在解决低资源印度语言的语音生成问题。通过对比三种微调策略（从头训练、仅印度数据微调、英印混合微调），发现仅用印度数据微调英文F5模型（获IN-F5）效果最佳，实现多语言流畅合成、声音/风格迁移及零资源语言（如Bhojpuri）的生成，证明英文预训练对低资源TTS达人类水平的关键作用，并提出数据受限场景的计算优化方案。**

- **链接: [http://arxiv.org/pdf/2505.20693v1](http://arxiv.org/pdf/2505.20693v1)**

> **作者:** Praveen Srinivasa Varadhan; Srija Anand; Soma Siddhartha; Mitesh M. Khapra
>
> **摘要:** What happens when an English Fairytaler is fine-tuned on Indian languages? We evaluate how the English F5-TTS model adapts to 11 Indian languages, measuring polyglot fluency, voice-cloning, style-cloning, and code-mixing. We compare: (i) training from scratch, (ii) fine-tuning English F5 on Indian data, and (iii) fine-tuning on both Indian and English data to prevent forgetting. Fine-tuning with only Indian data proves most effective and the resultant IN-F5 is a near-human polyglot; that enables speakers of one language (e.g., Odia) to fluently speak in another (e.g., Hindi). Our results show English pretraining aids low-resource TTS in reaching human parity. To aid progress in other low-resource languages, we study data-constrained setups and arrive at a compute optimal strategy. Finally, we show IN-F5 can synthesize unseen languages like Bhojpuri and Tulu using a human-in-the-loop approach for zero-resource TTS via synthetic data generation.
>
---
#### [new 030] Topological Deep Learning for Speech Data
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升低噪声场景及跨领域适应性。通过拓扑数据分析设计拓扑感知卷积核，理论分解矩阵空间为纤维丛，并提出正交特征层，优化神经网络，显著提升音素识别性能。**

- **链接: [http://arxiv.org/pdf/2505.21173v1](http://arxiv.org/pdf/2505.21173v1)**

> **作者:** Zhiwang Yu
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Topological data analysis (TDA) offers novel mathematical tools for deep learning. Inspired by Carlsson et al., this study designs topology-aware convolutional kernels that significantly improve speech recognition networks. Theoretically, by investigating orthogonal group actions on kernels, we establish a fiber-bundle decomposition of matrix spaces, enabling new filter generation methods. Practically, our proposed Orthogonal Feature (OF) layer achieves superior performance in phoneme recognition, particularly in low-noise scenarios, while demonstrating cross-domain adaptability. This work reveals TDA's potential in neural network optimization, opening new avenues for mathematics-deep learning interdisciplinary studies.
>
---
#### [new 031] ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出ArVoice数据集，用于多说话者阿拉伯语语音合成，解决该领域高质量多说话者数据缺乏的问题。包含专业录制、现有数据集改进及合成语音，共83.5小时（11种声音），并训练TTS及声音转换系统验证其应用，推动语音合成及相关研究。**

- **链接: [http://arxiv.org/pdf/2505.20506v1](http://arxiv.org/pdf/2505.20506v1)**

> **作者:** Hawau Olamide Toyin; Rufael Marew; Humaid Alblooshi; Samar M. Magdy; Hanan Aldarmaki
>
> **备注:** Accepted at INTERSPEECH 2025 The dataset is available at https://huggingface.co/datasets/MBZUAI/ArVoice
>
> **摘要:** We introduce ArVoice, a multi-speaker Modern Standard Arabic (MSA) speech corpus with diacritized transcriptions, intended for multi-speaker speech synthesis, and can be useful for other tasks such as speech-based diacritic restoration, voice conversion, and deepfake detection. ArVoice comprises: (1) a new professionally recorded set from six voice talents with diverse demographics, (2) a modified subset of the Arabic Speech Corpus; and (3) high-quality synthetic speech from two commercial systems. The complete corpus consists of a total of 83.52 hours of speech across 11 voices; around 10 hours consist of human voices from 7 speakers. We train three open-source TTS and two voice conversion systems to illustrate the use cases of the dataset. The corpus is available for research use.
>
---
#### [new 032] Assessment of L2 Oral Proficiency using Speech Large Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究二语（L2）口语自动评估任务，针对传统方法（级联系统信息丢失、端到端模型局限）的不足，利用语音大语言模型（LLMs）探索更优方案。通过对比不同训练策略，验证语音LLMs在评分任务中的优势，其性能超越现有基线，并展现跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.21148v1](http://arxiv.org/pdf/2505.21148v1)**

> **作者:** Rao Ma; Mengjie Qian; Siyuan Tang; Stefano Bannò; Kate M. Knill; Mark J. F. Gales
>
> **备注:** submitted to Interspeech
>
> **摘要:** The growing population of L2 English speakers has increased the demand for developing automatic graders for spoken language assessment (SLA). Historically, statistical models, text encoders, and self-supervised speech models have been utilised for this task. However, cascaded systems suffer from the loss of information, while E2E graders also have limitations. With the recent advancements of multi-modal large language models (LLMs), we aim to explore their potential as L2 oral proficiency graders and overcome these issues. In this work, we compare various training strategies using regression and classification targets. Our results show that speech LLMs outperform all previous competitive baselines, achieving superior performance on two datasets. Furthermore, the trained grader demonstrates strong generalisation capabilities in the cross-part or cross-task evaluation, facilitated by the audio understanding knowledge acquired during LLM pre-training.
>
---
#### [new 033] ReverbFX: A Dataset of Room Impulse Responses Derived from Reverb Effect Plugins for Singing Voice Dereverberation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出ReverbFX数据集，用于歌声去混响任务。针对现有数据集依赖真实房间脉冲响应（RIR）的局限，收集音乐制作常用混响插件生成的多样化RIR。通过实验表明，基于该数据集训练的生成模型在处理人工混响 vocals 时性能优于真实RIR训练的模型。**

- **链接: [http://arxiv.org/pdf/2505.20533v1](http://arxiv.org/pdf/2505.20533v1)**

> **作者:** Julius Richter; Till Svajda; Timo Gerkmann
>
> **备注:** Submitted to ITG Conference on Speech Communication
>
> **摘要:** We present ReverbFX, a new room impulse response (RIR) dataset designed for singing voice dereverberation research. Unlike existing datasets based on real recorded RIRs, ReverbFX features a diverse collection of RIRs captured from various reverb audio effect plugins commonly used in music production. We conduct comprehensive experiments using the proposed dataset to benchmark the challenge of dereverberation of singing voice recordings affected by artificial reverbs. We train two state-of-the-art generative models using ReverbFX and demonstrate that models trained with plugin-derived RIRs outperform those trained on realistic RIRs in artificial reverb scenarios.
>
---
#### [new 034] Towards Emotionally Consistent Text-Based Speech Editing: Introducing EmoCorrector and The ECD-TSE Dataset
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于文本驱动语音编辑（TSE）任务，旨在解决现有方法忽视文本修改导致的语音情感不一致问题。提出EmoCorrector方案，通过检索增强生成匹配情感的语音样本，修正情感偏差，并构建ECD-TSE数据集支持训练与评估，实验验证其提升情感表达一致性。**

- **链接: [http://arxiv.org/pdf/2505.20341v1](http://arxiv.org/pdf/2505.20341v1)**

> **作者:** Rui Liu; Pu Gao; Jiatian Xi; Berrak Sisman; Carlos Busso; Haizhou Li
>
> **备注:** INTERSPEECH2025. Code and audio examples: https://github.com/AI-S2-Lab/EmoCorrector
>
> **摘要:** Text-based speech editing (TSE) modifies speech using only text, eliminating re-recording. However, existing TSE methods, mainly focus on the content accuracy and acoustic consistency of synthetic speech segments, and often overlook the emotional shifts or inconsistency issues introduced by text changes. To address this issue, we propose EmoCorrector, a novel post-correction scheme for TSE. EmoCorrector leverages Retrieval-Augmented Generation (RAG) by extracting the edited text's emotional features, retrieving speech samples with matching emotions, and synthesizing speech that aligns with the desired emotion while preserving the speaker's identity and quality. To support the training and evaluation of emotional consistency modeling in TSE, we pioneer the benchmarking Emotion Correction Dataset for TSE (ECD-TSE). The prominent aspect of ECD-TSE is its inclusion of $<$text, speech$>$ paired data featuring diverse text variations and a range of emotional expressions. Subjective and objective experiments and comprehensive analysis on ECD-TSE confirm that EmoCorrector significantly enhances the expression of intended emotion while addressing emotion inconsistency limitations in current TSE methods. Code and audio examples are available at https://github.com/AI-S2-Lab/EmoCorrector.
>
---
#### [new 035] Scaling and Prompting for Improved End-to-End Spoken Grammatical Error Correction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于端到端口语语法错误纠正（SGEC）与反馈生成任务。针对标注数据不足及模型性能瓶颈，提出伪标记方法扩展训练数据至2500小时，并通过提示优化Whisper模型，提升纠错准确性和反馈效果。实验表明，伪标签对大模型效果有限，但提示策略有效。**

- **链接: [http://arxiv.org/pdf/2505.21137v1](http://arxiv.org/pdf/2505.21137v1)**

> **作者:** Mengjie Qian; Rao Ma; Stefano Bannò; Kate M. Knill; Mark J. F. Gales
>
> **备注:** submitted to Interspeech
>
> **摘要:** Spoken Grammatical Error Correction (SGEC) and Feedback (SGECF) are crucial for second language learners, teachers and test takers. Traditional SGEC systems rely on a cascaded pipeline consisting of an ASR, a module for disfluency detection (DD) and removal and one for GEC. With the rise of end-to-end (E2E) speech foundation models, we investigate their effectiveness in SGEC and feedback generation. This work introduces a pseudo-labelling process to address the challenge of limited labelled data, expanding the training data size from 77 hours to approximately 2500 hours, leading to improved performance. Additionally, we prompt an E2E Whisper-based SGEC model with fluent transcriptions, showing a slight improvement in SGEC performance, with more significant gains in feedback generation. Finally, we assess the impact of increasing model size, revealing that while pseudo-labelled data does not yield performance gain for a larger Whisper model, training with prompts proves beneficial.
>
---
#### [new 036] Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出无文本语音到语音翻译框架Dub-S2ST，用于跨语言配音。针对现有方法忽视源语音时长、语速及身份导致的不匹配问题，其创新采用离散扩散模型结合显式时长控制实现时间对齐，并引入单元级语速适应机制，合成语音保持源特征，实验验证其自然流畅且翻译效果优异。**

- **链接: [http://arxiv.org/pdf/2505.20899v1](http://arxiv.org/pdf/2505.20899v1)**

> **作者:** Jeongsoo Choi; Jaehun Kim; Joon Son Chung
>
> **摘要:** This paper introduces a cross-lingual dubbing system that translates speech from one language to another while preserving key characteristics such as duration, speaker identity, and speaking speed. Despite the strong translation quality of existing speech translation approaches, they often overlook the transfer of speech patterns, leading to mismatches with source speech and limiting their suitability for dubbing applications. To address this, we propose a discrete diffusion-based speech-to-unit translation model with explicit duration control, enabling time-aligned translation. We then synthesize speech based on the predicted units and source identity with a conditional flow matching model. Additionally, we introduce a unit-based speed adaptation mechanism that guides the translation model to produce speech at a rate consistent with the source, without relying on any text. Extensive experiments demonstrate that our framework generates natural and fluent translations that align with the original speech's duration and speaking pace, while achieving competitive translation performance.
>
---
#### [new 037] In-context learning capabilities of Large Language Models to detect suicide risk among adolescents from speech transcripts
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究通过分析青少年语音转录文本检测自杀风险，旨在解决传统评估方法的可扩展性问题。团队利用大型语言模型（LLMs）和DSPy工具进行系统性提示工程，开发基于in-context学习的分类方法，优于传统微调，在竞赛中获第三/四名（准确率0.68，F1 0.7），验证LLMs在心理健康中的应用价值。**

- **链接: [http://arxiv.org/pdf/2505.20491v1](http://arxiv.org/pdf/2505.20491v1)**

> **作者:** Filomene Roquefort; Alexandre Ducorroy; Rachid Riad
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Early suicide risk detection in adolescents is critical yet hindered by scalability challenges of current assessments. This paper presents our approach to the first SpeechWellness Challenge (SW1), which aims to assess suicide risk in Chinese adolescents through speech analysis. Due to speech anonymization constraints, we focused on linguistic features, leveraging Large Language Models (LLMs) for transcript-based classification. Using DSPy for systematic prompt engineering, we developed a robust in-context learning approach that outperformed traditional fine-tuning on both linguistic and acoustic markers. Our systems achieved third and fourth places among 180+ submissions, with 0.68 accuracy (F1=0.7) using only transcripts. Ablation analyses showed that increasing prompt example improved performance (p=0.003), with varying effects across model types and sizes. These findings advance automated suicide risk assessment and demonstrate LLMs' value in mental health applications.
>
---
#### [new 038] Robust fine-tuning of speech recognition models via model merging: application to disordered speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，解决构音障碍语音因数据不足和变异性导致的ASR性能下降问题。通过比较单路径与多运行模型合并方法，基于Whisper模型，发现多运行合并使WER降低12%-16.2%，尤其提升长音频表现，并在低数据和跨架构场景有效。**

- **链接: [http://arxiv.org/pdf/2505.20477v1](http://arxiv.org/pdf/2505.20477v1)**

> **作者:** Alexandre Ducorroy; Rachid Riad
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) has advanced with Speech Foundation Models (SFMs), yet performance degrades on dysarthric speech due to variability and limited data. This study as part of the submission to the Speech Accessibility challenge, explored model merging to improve ASR generalization using Whisper as the base SFM. We compared fine-tuning with single-trajectory merging, combining models from one fine-tuning path, and multi-run merging, merging independently trained models. Our best multi-run merging approach achieved a 12% relative decrease of WER over classic fine-tuning, and a 16.2% relative decrease on long-form audios, a major loss contributor in dysarthric ASR. Merging more and more models led to continuous gains, remained effective in low-data regimes, and generalized across model architectures. These results highlight model merging as an easily replicable adaptation method that consistently improves ASR without additional inference cost or hyperparameter tuning.
>
---
## 更新

#### [replaced 001] X-ARES: A Comprehensive Framework for Assessing Audio Encoder Performance
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16369v2](http://arxiv.org/pdf/2505.16369v2)**

> **作者:** Junbo Zhang; Heinrich Dinkel; Yadong Niu; Chenyu Liu; Si Cheng; Anbei Zhao; Jian Luan
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We introduces X-ARES (eXtensive Audio Representation and Evaluation Suite), a novel open-source benchmark designed to systematically assess audio encoder performance across diverse domains. By encompassing tasks spanning speech, environmental sounds, and music, X-ARES provides two evaluation approaches for evaluating audio representations: linear fine-tuning and unparameterized evaluation. The framework includes 22 distinct tasks that cover essential aspects of audio processing, from speech recognition and emotion detection to sound event classification and music genre identification. Our extensive evaluation of state-of-the-art audio encoders reveals significant performance variations across different tasks and domains, highlighting the complexity of general audio representation learning.
>
---
#### [replaced 002] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2406.11546v2](http://arxiv.org/pdf/2406.11546v2)**

> **作者:** Yifan Yang; Zheshu Song; Jianheng Zhuo; Mingyu Cui; Jinpeng Li; Bo Yang; Yexing Du; Ziyang Ma; Xunying Liu; Ziyuan Wang; Ke Li; Shuai Fan; Kai Yu; Wei-Qiang Zhang; Guoguo Chen; Xie Chen
>
> **备注:** Accepted in ACL 2025 (Main)
>
> **摘要:** The evolution of speech technology has been spurred by the rapid increase in dataset sizes. Traditional speech models generally depend on a large amount of labeled training data, which is scarce for low-resource languages. This paper presents GigaSpeech 2, a large-scale, multi-domain, multilingual speech recognition corpus. It is designed for low-resource languages and does not rely on paired speech and text data. GigaSpeech 2 comprises about 30,000 hours of automatically transcribed speech, including Thai, Indonesian, and Vietnamese, gathered from unlabeled YouTube videos. We also introduce an automated pipeline for data crawling, transcription, and label refinement. Specifically, this pipeline involves Whisper for initial transcription, MMS for forced alignment, and multi-dimensional filtering for data quality assurance. A modified Noisy Student Training is developed to further refine flawed pseudo labels iteratively, thereby enhancing model performance. Experimental results on our manually transcribed evaluation set and two public test sets from Common Voice and FLEURS confirm our corpus's high quality and broad applicability. Notably, ASR models trained on GigaSpeech 2 can reduce the word error rate for Thai, Indonesian, and Vietnamese on our challenging and realistic YouTube test set by 25% to 40% compared to Whisper large-v3, with merely 10% model parameters. Furthermore, our ASR models trained on GigaSpeech 2 yield superior performance compared to commercial services. We hope that our newly introduced corpus and pipeline will open a new avenue for low-resource speech recognition and significantly facilitate research in this area.
>
---
#### [replaced 003] Autoregressive Speech Synthesis without Vector Quantization
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.08551v2](http://arxiv.org/pdf/2407.08551v2)**

> **作者:** Lingwei Meng; Long Zhou; Shujie Liu; Sanyuan Chen; Bing Han; Shujie Hu; Yanqing Liu; Jinyu Li; Sheng Zhao; Xixin Wu; Helen Meng; Furu Wei
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** We present MELLE, a novel continuous-valued token based language modeling approach for text-to-speech synthesis (TTS). MELLE autoregressively generates continuous mel-spectrogram frames directly from text condition, bypassing the need for vector quantization, which is typically designed for audio compression and sacrifices fidelity compared to continuous representations. Specifically, (i) instead of cross-entropy loss, we apply regression loss with a proposed spectrogram flux loss function to model the probability distribution of the continuous-valued tokens; (ii) we have incorporated variational inference into MELLE to facilitate sampling mechanisms, thereby enhancing the output diversity and model robustness. Experiments demonstrate that, compared to the two-stage codec language model VALL-E and its variants, the single-stage MELLE mitigates robustness issues by avoiding the inherent flaws of sampling vector-quantized codes, achieves superior performance across multiple metrics, and, most importantly, offers a more streamlined paradigm. The demos of our work are provided at https://aka.ms/melle.
>
---
#### [replaced 004] Advanced Signal Analysis in Detecting Replay Attacks for Automatic Speaker Verification Systems
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2403.01130v3](http://arxiv.org/pdf/2403.01130v3)**

> **作者:** Lee Shih Kuang
>
> **备注:** https://github.com/shihkuanglee/ADFA
>
> **摘要:** This study proposes novel signal analysis methods for replay speech detection in automatic speaker verification (ASV) systems. The proposed methods -- arbitrary analysis (AA), mel scale analysis (MA), and constant Q analysis (CQA) -- are inspired by the calculation of the Fourier inversion formula. These methods introduce new perspectives in signal analysis for replay speech detection by employing alternative sinusoidal sequence groups. The efficacy of the proposed methods is examined on the ASVspoof 2019 \& 2021 PA databases with experiments, and confirmed by the performance of systems that incorporated the proposed methods; the successful integration of the proposed methods and a speech feature that calculates temporal autocorrelation of speech (TAC) from complex spectra strongly confirms it. Moreover, the proposed CQA and MA methods show their superiority to the conventional methods on efficiency (approximately 2.36 times as fast compared to the conventional constant Q transform (CQT) method) and efficacy, respectively, in analyzing speech signals, making them promising to utilize in music and speech processing works.
>
---
#### [replaced 005] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v2](http://arxiv.org/pdf/2505.14874v2)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 006] Music Foundation Model as Generic Booster for Music Downstream Tasks
- **分类: cs.SD; cs.IR; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.01135v3](http://arxiv.org/pdf/2411.01135v3)**

> **作者:** WeiHsiang Liao; Yuhta Takida; Yukara Ikemiya; Zhi Zhong; Chieh-Hsin Lai; Giorgio Fabbro; Kazuki Shimada; Keisuke Toyama; Kinwai Cheuk; Marco A. Martínez-Ramírez; Shusuke Takahashi; Stefan Uhlich; Taketo Akama; Woosung Choi; Yuichiro Koyama; Yuki Mitsufuji
>
> **备注:** 41 pages with 14 figures
>
> **摘要:** We demonstrate the efficacy of using intermediate representations from a single foundation model to enhance various music downstream tasks. We introduce SoniDo, a music foundation model (MFM) designed to extract hierarchical features from target music samples. By leveraging hierarchical intermediate features, SoniDo constrains the information granularity, leading to improved performance across various downstream tasks including both understanding and generative tasks. We specifically evaluated this approach on representative tasks such as music tagging, music transcription, music source separation, and music mixing. Our results reveal that the features extracted from foundation models provide valuable enhancements in training downstream task models. This highlights the capability of using features extracted from music foundation models as a booster for downstream tasks. Our approach not only benefits existing task-specific models but also supports music downstream tasks constrained by data scarcity. This paves the way for more effective and accessible music processing solutions.
>
---
#### [replaced 007] CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17589v2](http://arxiv.org/pdf/2505.17589v2)**

> **作者:** Zhihao Du; Changfeng Gao; Yuxuan Wang; Fan Yu; Tianyu Zhao; Hao Wang; Xiang Lv; Hui Wang; Chongjia Ni; Xian Shi; Keyu An; Guanrou Yang; Yabin Li; Yanni Chen; Zhifu Gao; Qian Chen; Yue Gu; Mengzhe Chen; Yafeng Chen; Shiliang Zhang; Wen Wang; Jieping Ye
>
> **备注:** Preprint, work in progress
>
> **摘要:** In our prior works, we introduced a scalable streaming speech synthesis model, CosyVoice 2, which integrates a large language model (LLM) and a chunk-aware flow matching (FM) model, and achieves low-latency bi-streaming speech synthesis and human-parity quality. Despite these advancements, CosyVoice 2 exhibits limitations in language coverage, domain diversity, data volume, text formats, and post-training techniques. In this paper, we present CosyVoice 3, an improved model designed for zero-shot multilingual speech synthesis in the wild, surpassing its predecessor in content consistency, speaker similarity, and prosody naturalness. Key features of CosyVoice 3 include: 1) A novel speech tokenizer to improve prosody naturalness, developed via supervised multi-task training, including automatic speech recognition, speech emotion recognition, language identification, audio event detection, and speaker analysis. 2) A new differentiable reward model for post-training applicable not only to CosyVoice 3 but also to other LLM-based speech synthesis models. 3) Dataset Size Scaling: Training data is expanded from ten thousand hours to one million hours, encompassing 9 languages and 18 Chinese dialects across various domains and text formats. 4) Model Size Scaling: Model parameters are increased from 0.5 billion to 1.5 billion, resulting in enhanced performance on our multilingual benchmark due to the larger model capacity. These advancements contribute significantly to the progress of speech synthesis in the wild. We encourage readers to listen to the demo at https://funaudiollm.github.io/cosyvoice3.
>
---
#### [replaced 008] FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.19476v2](http://arxiv.org/pdf/2505.19476v2)**

> **作者:** Ziqian Wang; Zikai Liu; Xinfa Zhu; Yike Zhu; Mingshuai Liu; Jun Chen; Longshuai Xiao; Chao Weng; Lei Xie
>
> **备注:** Accepted to InterSpeech 2025
>
> **摘要:** Generative models have excelled in audio tasks using approaches such as language models, diffusion, and flow matching. However, existing generative approaches for speech enhancement (SE) face notable challenges: language model-based methods suffer from quantization loss, leading to compromised speaker similarity and intelligibility, while diffusion models require complex training and high inference latency. To address these challenges, we propose FlowSE, a flow-matching-based model for SE. Flow matching learns a continuous transformation between noisy and clean speech distributions in a single pass, significantly reducing inference latency while maintaining high-quality reconstruction. Specifically, FlowSE trains on noisy mel spectrograms and optional character sequences, optimizing a conditional flow matching loss with ground-truth mel spectrograms as supervision. It implicitly learns speech's temporal-spectral structure and text-speech alignment. During inference, FlowSE can operate with or without textual information, achieving impressive results in both scenarios, with further improvements when transcripts are available. Extensive experiments demonstrate that FlowSE significantly outperforms state-of-the-art generative methods, establishing a new paradigm for generative-based SE and demonstrating the potential of flow matching to advance the field. Our code, pre-trained checkpoints, and audio samples are available.
>
---
#### [replaced 009] Resampling Filter Design for Multirate Neural Audio Effect Processing
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.18470v2](http://arxiv.org/pdf/2501.18470v2)**

> **作者:** Alistair Carson; Vesa Välimäki; Alec Wright; Stefan Bilbao
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Neural networks have become ubiquitous in audio effects modelling, especially for guitar amplifiers and distortion pedals. One limitation of such models is that the sample rate of the training data is implicitly encoded in the model weights and therefore not readily adjustable at inference. Recent work explored modifications to recurrent neural network architecture to approximate a sample rate independent system, enabling audio processing at a rate that differs from the original training rate. This method works well for integer oversampling and can reduce aliasing caused by nonlinear activation functions. For small fractional changes in sample rate, fractional delay filters can be used to approximate sample rate independence, but in some cases this method fails entirely. Here, we explore the use of real-time signal resampling at the input and output of the neural network as an alternative solution. We investigate several resampling filter designs and show that a two-stage design consisting of a half-band IIR filter cascaded with a Kaiser window FIR filter can give similar or better results to the previously proposed model adjustment method with many fewer filtering operations per sample and less than one millisecond of latency at typical audio rates. Furthermore, we investigate interpolation and decimation filters for the task of integer oversampling and show that cascaded half-band IIR and FIR designs can be used in conjunction with the model adjustment method to reduce aliasing in a range of distortion effect models.
>
---
#### [replaced 010] Sentiment Reasoning for Healthcare
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.21054v4](http://arxiv.org/pdf/2407.21054v4)**

> **作者:** Khai-Nguyen Nguyen; Khai Le-Duc; Bach Phan Tat; Duy Le; Long Vo-Dang; Truong-Son Hy
>
> **备注:** ACL 2025 (Oral)
>
> **摘要:** Transparency in AI healthcare decision-making is crucial. By incorporating rationales to explain reason for each predicted label, users could understand Large Language Models (LLMs)'s reasoning to make better decision. In this work, we introduce a new task - Sentiment Reasoning - for both speech and text modalities, and our proposed multimodal multitask framework and the world's largest multimodal sentiment analysis dataset. Sentiment Reasoning is an auxiliary task in sentiment analysis where the model predicts both the sentiment label and generates the rationale behind it based on the input transcript. Our study conducted on both human transcripts and Automatic Speech Recognition (ASR) transcripts shows that Sentiment Reasoning helps improve model transparency by providing rationale for model prediction with quality semantically comparable to humans while also improving model's classification performance (+2% increase in both accuracy and macro-F1) via rationale-augmented fine-tuning. Also, no significant difference in the semantic quality of generated rationales between human and ASR transcripts. All code, data (five languages - Vietnamese, English, Chinese, German, and French) and models are published online: https://github.com/leduckhai/Sentiment-Reasoning
>
---
#### [replaced 011] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14910v2](http://arxiv.org/pdf/2505.14910v2)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Dongyu Yao; Zhiyuan Zhu; Ziyue Jiang; Yuhan Wang; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks. Singing voice samples are available at https://aaronz345.github.io/TCSinger2Demo/.
>
---
#### [replaced 012] U-SAM: An audio language Model for Unified Speech, Audio, and Music Understanding
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.13880v3](http://arxiv.org/pdf/2505.13880v3)**

> **作者:** Ziqian Wang; Xianjun Xia; Xinfa Zhu; Lei Xie
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The text generation paradigm for audio tasks has opened new possibilities for unified audio understanding. However, existing models face significant challenges in achieving a comprehensive understanding across diverse audio types, such as speech, general audio events, and music. Furthermore, their exclusive reliance on cross-entropy loss for alignment often falls short, as it treats all tokens equally and fails to account for redundant audio features, leading to weaker cross-modal alignment. To deal with the above challenges, this paper introduces U-SAM, an advanced audio language model that integrates specialized encoders for speech, audio, and music with a pre-trained large language model (LLM). U-SAM employs a Mixture of Experts (MoE) projector for task-aware feature fusion, dynamically routing and integrating the domain-specific encoder outputs. Additionally, U-SAM incorporates a Semantic-Aware Contrastive Loss Module, which explicitly identifies redundant audio features under language supervision and rectifies their semantic and spectral representations to enhance cross-modal alignment. Extensive experiments demonstrate that U-SAM consistently outperforms both specialized models and existing audio language models across multiple benchmarks. Moreover, it exhibits emergent capabilities on unseen tasks, showcasing its generalization potential. Code is available (https://github.com/Honee-W/U-SAM/).
>
---
#### [replaced 013] VocalAgent: Large Language Models for Vocal Health Diagnostics with Safety-Aware Evaluation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13577v2](http://arxiv.org/pdf/2505.13577v2)**

> **作者:** Yubin Kim; Taehan Kim; Wonjune Kang; Eugene Park; Joonsik Yoon; Dongjae Lee; Xin Liu; Daniel McDuff; Hyeonhoon Lee; Cynthia Breazeal; Hae Won Park
>
> **摘要:** Vocal health plays a crucial role in peoples' lives, significantly impacting their communicative abilities and interactions. However, despite the global prevalence of voice disorders, many lack access to convenient diagnosis and treatment. This paper introduces VocalAgent, an audio large language model (LLM) to address these challenges through vocal health diagnosis. We leverage Qwen-Audio-Chat fine-tuned on three datasets collected in-situ from hospital patients, and present a multifaceted evaluation framework encompassing a safety assessment to mitigate diagnostic biases, cross-lingual performance analysis, and modality ablation studies. VocalAgent demonstrates superior accuracy on voice disorder classification compared to state-of-the-art baselines. Its LLM-based method offers a scalable solution for broader adoption of health diagnostics, while underscoring the importance of ethical and technical validation.
>
---
#### [replaced 014] The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13971v2](http://arxiv.org/pdf/2505.13971v2)**

> **作者:** Ming Gao; Shilong Wu; Hang Chen; Jun Du; Chin-Hui Lee; Shinji Watanabe; Jingdong Chen; Siniscalchi Sabato Marco; Odette Scharenborg
>
> **备注:** Accepted by Interspeech 2025. Camera-ready version
>
> **摘要:** Meetings are a valuable yet challenging scenario for speech applications due to complex acoustic conditions. This paper summarizes the outcomes of the MISP 2025 Challenge, hosted at Interspeech 2025, which focuses on multi-modal, multi-device meeting transcription by incorporating video modality alongside audio. The tasks include Audio-Visual Speaker Diarization (AVSD), Audio-Visual Speech Recognition (AVSR), and Audio-Visual Diarization and Recognition (AVDR). We present the challenge's objectives, tasks, dataset, baseline systems, and solutions proposed by participants. The best-performing systems achieved significant improvements over the baseline: the top AVSD model achieved a Diarization Error Rate (DER) of 8.09%, improving by 7.43%; the top AVSR system achieved a Character Error Rate (CER) of 9.48%, improving by 10.62%; and the best AVDR system achieved a concatenated minimum-permutation Character Error Rate (cpCER) of 11.56%, improving by 72.49%.
>
---
#### [replaced 015] Rethinking MUSHRA: Addressing Modern Challenges in Text-to-Speech Evaluation
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.12719v3](http://arxiv.org/pdf/2411.12719v3)**

> **作者:** Praveen Srinivasa Varadhan; Amogh Gulati; Ashwin Sankar; Srija Anand; Anirudh Gupta; Anirudh Mukherjee; Shiva Kumar Marepally; Ankur Bhatia; Saloni Jaju; Suvrat Bhooshan; Mitesh M. Khapra
>
> **备注:** Accepted in TMLR
>
> **摘要:** Despite rapid advancements in TTS models, a consistent and robust human evaluation framework is still lacking. For example, MOS tests fail to differentiate between similar models, and CMOS's pairwise comparisons are time-intensive. The MUSHRA test is a promising alternative for evaluating multiple TTS systems simultaneously, but in this work we show that its reliance on matching human reference speech unduly penalises the scores of modern TTS systems that can exceed human speech quality. More specifically, we conduct a comprehensive assessment of the MUSHRA test, focusing on its sensitivity to factors such as rater variability, listener fatigue, and reference bias. Based on our extensive evaluation involving 492 human listeners across Hindi and Tamil we identify two primary shortcomings: (i) reference-matching bias, where raters are unduly influenced by the human reference, and (ii) judgement ambiguity, arising from a lack of clear fine-grained guidelines. To address these issues, we propose two refined variants of the MUSHRA test. The first variant enables fairer ratings for synthesized samples that surpass human reference quality. The second variant reduces ambiguity, as indicated by the relatively lower variance across raters. By combining these approaches, we achieve both more reliable and more fine-grained assessments. We also release MANGO, a massive dataset of 246,000 human ratings, the first-of-its-kind collection for Indian languages, aiding in analyzing human preferences and developing automatic metrics for evaluating TTS systems.
>
---
#### [replaced 016] Multi-Stage Speaker Diarization for Noisy Classrooms
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.10879v2](http://arxiv.org/pdf/2505.10879v2)**

> **作者:** Ali Sartaz Khan; Tolulope Ogunremi; Ahmed Adel Attia; Dorottya Demszky
>
> **摘要:** Speaker diarization, the process of identifying "who spoke when" in audio recordings, is essential for understanding classroom dynamics. However, classroom settings present distinct challenges, including poor recording quality, high levels of background noise, overlapping speech, and the difficulty of accurately capturing children's voices. This study investigates the effectiveness of multi-stage diarization models using Nvidia's NeMo diarization pipeline. We assess the impact of denoising on diarization accuracy and compare various voice activity detection (VAD) models, including self-supervised transformer-based frame-wise VAD models. We also explore a hybrid VAD approach that integrates Automatic Speech Recognition (ASR) word-level timestamps with frame-level VAD predictions. We conduct experiments using two datasets from English speaking classrooms to separate teacher vs. student speech and to separate all speakers. Our results show that denoising significantly improves the Diarization Error Rate (DER) by reducing the rate of missed speech. Additionally, training on both denoised and noisy datasets leads to substantial performance gains in noisy conditions. The hybrid VAD model leads to further improvements in speech detection, achieving a DER as low as 17% in teacher-student experiments and 45% in all-speaker experiments. However, we also identified trade-offs between voice activity detection and speaker confusion. Overall, our study highlights the effectiveness of multi-stage diarization models and integrating ASR-based information for enhancing speaker diarization in noisy classroom environments.
>
---
#### [replaced 017] VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.04962v4](http://arxiv.org/pdf/2501.04962v4)**

> **作者:** Wenqian Cui; Xiaoqi Jiao; Ziqiao Meng; Irwin King
>
> **备注:** The Version of Record of this contribution is accepted to ACL 2025 main conference
>
> **摘要:** With the rising need for speech-based interaction models, end-to-end Spoken Language Models (SLMs) have emerged as a promising solution. While these models require comprehensive world knowledge for meaningful and reliable human interactions, existing question-answering (QA) benchmarks fall short in evaluating SLMs' knowledge understanding due to their inability to support end-to-end speech evaluation and account for varied input audio conditions. To address these limitations, we present VoxEval, a novel SpeechQA benchmark that assesses SLMs' knowledge understanding through pure speech interactions. Our benchmark 1) uniquely maintains speech format for both inputs and outputs, 2) evaluates model robustness across diverse input audio conditions, and 3) pioneers the assessment of complex tasks like mathematical reasoning in spoken format. Systematic evaluation demonstrates that VoxEval presents significant challenges to current SLMs, revealing their sensitivity to varying audio conditions and highlighting the need to enhance reasoning capabilities in future development. We hope this benchmark could guide the advancement of more sophisticated and reliable SLMs. VoxEval dataset is available at: https://github.com/dreamtheater123/VoxEval
>
---
