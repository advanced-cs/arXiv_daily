# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] A Novel Automatic Framework for Speaker Drift Detection in Synthesized Speech
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决合成语音中的说话人漂移问题。通过构建二分类框架，利用余弦相似度和大语言模型检测说话人一致性变化。**

- **链接: [https://arxiv.org/pdf/2604.06327](https://arxiv.org/pdf/2604.06327)**

> **作者:** Jia-Hong Huang; Seulgi Kim; Yi Chieh Liu; Yixian Shen; Hongyi Zhu; Prayag Tiwari; Stevan Rudinac; Evangelos Kanoulas
>
> **备注:** The paper has been accepted by the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026
>
> **摘要:** Recent diffusion-based text-to-speech (TTS) models achieve high naturalness and expressiveness, yet often suffer from speaker drift, a subtle, gradual shift in perceived speaker identity within a single utterance. This underexplored phenomenon undermines the coherence of synthetic speech, especially in long-form or interactive settings. We introduce the first automatic framework for detecting speaker drift by formulating it as a binary classification task over utterance-level speaker consistency. Our method computes cosine similarity across overlapping segments of synthesized speech and prompts large language models (LLMs) with structured representations to assess drift. We provide theoretical guarantees for cosine-based drift detection and demonstrate that speaker embeddings exhibit meaningful geometric clustering on the unit sphere. To support evaluation, we construct a high-quality synthetic benchmark with human-validated speaker drift annotations. Experiments with multiple state-of-the-art LLMs confirm the viability of this embedding-to-reasoning pipeline. Our work establishes speaker drift as a standalone research problem and bridges geometric signal analysis with LLM-based perceptual reasoning in modern TTS.
>
---
#### [new 002] DAT-CFTNet: Speech Enhancement for Cochlear Implant Recipients using Attention-based Dual-Path Recurrent Neural Network
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升耳蜗植入者在噪声环境下的语音可懂性。提出DAT-CFTNet模型，结合注意力机制与双路径RNN，有效区分语音与噪声。**

- **链接: [https://arxiv.org/pdf/2604.06744](https://arxiv.org/pdf/2604.06744)**

> **作者:** Nursadul Mamun; John H.L. Hansen
>
> **备注:** 5 pages
>
> **摘要:** The human auditory system has the ability to selectively focus on key speech elements in an audio stream while giving secondary attention to less relevant areas such as noise or distortion within the background, dynamically adjusting its attention over time. Inspired by the recent success of attention models, this study introduces a dual-path attention module in the bottleneck layer of a concurrent speech enhancement network. Our study proposes an attention-based dual-path RNN (DAT-RNN), which, when combined with the modified complex-valued frequency transformation network (CFTNet), forms the DAT-CFTNet. This attention mechanism allows for precise differentiation between speech and noise in time-frequency (T-F) regions of spectrograms, optimizing both local and global context information processing in the CFTNet. Our experiments suggest that the DAT-CFTNet leads to consistently improved performance over the existing models, including CFTNet and DCCRN, in terms of speech intelligibility and quality. Moreover, the proposed model exhibits superior performance in enhancing speech intelligibility for cochlear implant (CI) recipients, who are known to have severely limited T-F hearing restoration (e.g., >10%) in CI listener studies in noisy settings show the proposed solution is capable of suppressing non-stationary noise, avoiding the musical artifacts often seen in traditional speech enhancement methods. The implementation of the proposed model will be publicly available.
>
---
#### [new 003] AudioKV: KV Cache Eviction in Efficient Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文属于语音处理任务，解决大模型长文本推理中的KV缓存内存问题。提出AudioKV框架，通过语义-声学对齐和频谱平滑技术优化缓存分配，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.06694](https://arxiv.org/pdf/2604.06694)**

> **作者:** Yuxuan Wang; Peize He; Xiyan Gui; Xiaoqian Liu; Junhao He; Xuyang Liu; Zichen Wen; Xuming Hu; Linfeng Zhang
>
> **摘要:** Large Audio-Language Models (LALMs) have set new benchmarks in speech processing, yet their deployment is hindered by the memory footprint of the Key-Value (KV) cache during long-context inference. While general KV cache compression techniques excel in LLMs, they often fail in the audio domain by overlooking the intrinsic temporal continuity of acoustic signals. To bridge this gap, we propose AudioKV, a novel framework that robustly prioritizes audio-critical attention heads through a hardware-friendly semantic-acoustic alignment mechanism. Specifically, we identify these modality-specialized heads by analyzing attention scores in ASR tasks and dynamically allocate KV cache budgets preferentially to them. Furthermore, we introduce Spectral Score Smoothing (SSS), an FFT-based global filtering strategy designed to suppress high-frequency noise and recover smooth global trends from importance scores, ensuring more balanced token selection with unprecedented precision. Extensive evaluations across multiple LALMs, including Qwen and Gemma series, demonstrate that AudioKV significantly outperforms baselines while enhancing computational efficiency. Notably, at a 40% compression ratio, AudioKV maintains near-full accuracy on Qwen3-Omni-30B with only a 0.45% drop, whereas traditional methods suffer from catastrophic performance degradation and repetition. Our code will be released after acceptance.
>
---
#### [new 004] EvoTSE: Evolving Enrollment for Target Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文属于目标说话人提取任务，解决模型在混音中混淆说话人的问题。提出EvoTSE框架，通过动态更新语音档案提升性能。**

- **链接: [https://arxiv.org/pdf/2604.06810](https://arxiv.org/pdf/2604.06810)**

> **作者:** Zikai Liu; Ziqian Wang; Xingchen Li; Yike Zhu; Shuai Wang; Longshuai Xiao; Lei Xie
>
> **摘要:** Target Speaker Extraction (TSE) aims to isolate a specific speaker's voice from a mixture, guided by a pre-recorded enrollment. While TSE bypasses the global permutation ambiguity of blind source separation, it remains vulnerable to speaker confusion, where models mistakenly extract the interfering speaker. Furthermore, conventional TSE relies on static inference pipeline, where performance is limited by the quality of the fixed enrollment. To overcome these limitations, we propose EvoTSE, an evolving TSE framework in which the enrollment is continuously updated through reliability-filtered retrieval over high-confidence historical estimates. This mechanism reduces speaker confusion and relaxes the quality requirements for pre-recorded enrollment without relying on additional annotated data. Experiments across multiple benchmarks demonstrate that EvoTSE achieves consistent improvements, especially when evaluated on out-of-domain (OOD) scenarios. Our code and checkpoints are available.
>
---
#### [new 005] ULTRAS -- Unified Learning of Transformer Representations for Audio and Speech Signals
- **分类: eess.AS**

- **简介: 该论文提出ULTRAS框架，解决音频与语音表示学习的统一问题。通过Transformer模型，在时频域上进行掩码预测，提升跨任务性能。**

- **链接: [https://arxiv.org/pdf/2604.06702](https://arxiv.org/pdf/2604.06702)**

> **作者:** Ameenudeen P E; Charumathi Narayanan; Sriram Ganapathy
>
> **摘要:** Self-supervised learning (SSL) has driven impressive advances in speech processing by adopting time-domain prediction objectives, while audio representation learning frameworks operate on time-frequency spectrograms. Models optimized for one paradigm struggle to transfer to the other, highlighting the need for a joint framework. We propose Unified Learning of Transformer Representations for Audio and Speech (ULTRAS), where the masking and predictive modeling is performed over long patches of the data. The model, based on the transformer architecture, encodes spectral-patches of log-mel spectrogram features. The predictive modeling of masked segments is performed on spectral and temporal targets using a combined loss-function, forcing the representations to encode time and frequency traits. Experiments are performed on a variety of speech and audio tasks, where we illustrate that the ULTRAS framework achieves improved performance over other established baselines.
>
---
#### [new 006] Harf-Speech: A Clinically Aligned Framework for Arabic Phoneme-Level Speech Assessment
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音评估任务，旨在解决阿拉伯语发音评分缺乏有效工具的问题。提出Harf-Speech框架，结合多种模型实现精准的音素级评分。**

- **链接: [https://arxiv.org/pdf/2604.06191](https://arxiv.org/pdf/2604.06191)**

> **作者:** Asif Azad; MD Sadik Hossain Shanto; Mohammad Sadat Hossain; Bdour Alwuqaysi; Sabri Boughorbel; Yahya Bokhari; Abdulrhman Aljouie; Ayah Othman Sindi; Ehsan Hoque
>
> **摘要:** Automated phoneme-level pronunciation assessment is vital for scalable speech therapy and language learning, yet validated tools for Arabic remain scarce. We present Harf-Speech, a modular system scoring Arabic pronunciation at the phoneme level on a clinical scale. It combines an MSA phonetizer, a fine-tuned speech-to-phoneme model, Levenshtein alignment, and a blended scorer using longest common subsequence and edit-distance metrics. We fine-tune three ASR architectures on Arabic phoneme data and benchmark them with zero-shot multimodal models; the best, OmniASR-CTC-1B-v2, achieves 8.92\% phoneme error rate. Three certified speech-language pathologists independently scored 40 utterances for clinical validation. Harf-Speech attains a Pearson correlation of 0.791 and ICC(2,1) of 0.659 with mean expert scores, outperforming existing end-to-end assessment frameworks. These results show Harf-Speech yields clinically aligned, interpretable scores comparable to inter-rater expert agreement.
>
---
#### [new 007] Development of ML model for triboelectric nanogenerator based sign language detection system
- **分类: eess.SP; cs.AI; cs.SD**

- **简介: 该论文属于手势识别任务，旨在解决手语识别中传统方法的局限性。通过设计基于TENG传感器的手套，结合ML和深度学习模型，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.06220](https://arxiv.org/pdf/2604.06220)**

> **作者:** Meshv Patel; Bikash Baro; Sayan Bayan; Mohendra Roy
>
> **备注:** This paper has been accepted at the IEEE GCON 2026 (this https URL) Conference, organized by IIT Guwahati
>
> **摘要:** Sign language recognition (SLR) is vital for bridging communication gaps between deaf and hearing communities. Vision-based approaches suffer from occlusion, computational costs, and physical constraints. This work presents a comparison of machine learning (ML) and deep learning models for a custom triboelectric nanogenerator (TENG)-based sensor glove. Utilizing multivariate time-series data from five flex sensors, the study benchmarks traditional ML algorithms, feedforward neural networks, LSTM-based temporal models, and a multi-sensor MFCC CNN-LSTM architecture across 11 sign classes (digits 1-5, letters A-F). The proposed MFCC CNN-LSTM architecture processes frequency-domain features from each sensor through independent convolutional branches before fusion. It achieves 93.33% accuracy and 95.56% precision, a 23-point improvement over the best ML algorithm (Random Forest: 70.38%). Ablation studies reveal 50-timestep windows offer a tradeoff between temporal context and training data volume, yielding 84.13% accuracy compared to 58.06% with 100-timestep windows. MFCC feature extraction maps temporal variations to execution-speed-invariant spectral representations, and data augmentation methods (time warping, noise injection) are essential for generalization. Results demonstrate that frequency-domain feature representations combined with parallel multi-sensor processing architectures offer enhancement over classical algorithms and time-domain deep learning for wearable sensor-based gesture recognition. This aids assistive technology development.
>
---
## 更新

#### [replaced 001] Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音编辑检测任务，解决现有数据多样性不足和删除类编辑识别困难的问题。提出统一框架，结合音频大模型实现编辑检测与内容定位。**

- **链接: [https://arxiv.org/pdf/2601.21463](https://arxiv.org/pdf/2601.21463)**

> **作者:** Jun Xue; Yi Chai; Yanzhen Ren; Jinshen He; Zhiqiang Tang; Zhuolin Yi; Yihuan Huang; Yuankun Xie; Yujie Chen
>
> **摘要:** Existing speech editing detection (SED) datasets are predominantly constructed using manual splicing or limited editing operations, resulting in restricted diversity and poor coverage of realistic editing scenarios. Meanwhile, current SED methods rely heavily on frame-level supervision to detect observable acoustic anomalies, which fundamentally limits their ability to handle deletion-type edits, where the manipulated content is entirely absent from the signal. To address these challenges, we present a unified framework that bridges speech editing detection and content localization through a generative formulation based on Audio Large Language Models (Audio LLMs). We first introduce AiEdit, a large-scale bilingual dataset (approximately 140 hours) that covers addition, deletion, and modification operations using state-of-the-art end-to-end speech editing systems, providing a more realistic benchmark for modern threats. Building upon this, we reformulate SED as a structured text generation task, enabling joint reasoning over edit type identification, and content localization. To enhance the grounding of generative models in acoustic evidence, we propose a prior-enhanced prompting strategy that injects word-level probabilistic cues derived from a frame-level detector. Furthermore, we introduce an acoustic consistency-aware loss that explicitly enforces the separation between normal and anomalous acoustic representations in the latent space. Experimental results demonstrate that the proposed approach consistently outperforms existing methods across both detection and localization tasks.
>
---
#### [replaced 002] AudioRole: An Audio Dataset for Character Role-Playing in Large Language Models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出AudioRole数据集，解决大语言模型中音频角色扮演的难题。通过构建包含1M+对话的音文同步数据，提升模型的角色扮演能力。**

- **链接: [https://arxiv.org/pdf/2509.23435](https://arxiv.org/pdf/2509.23435)**

> **作者:** Wenyu Li; Xiaoqi Jiao; Yi Chang; Guangyan Zhang; Yiwen Guo
>
> **摘要:** The creation of high-quality multimodal datasets remains fundamental for advancing role-playing capabilities in large language models (LLMs). While existing works predominantly focus on text-based persona simulation, Audio Role-Playing (ARP) presents unique challenges due to the need for synchronized alignment of semantic content and vocal characteristics. To address this gap, we propose AudioRole, a meticulously curated dataset from 13 TV series spanning 1K+ hours with 1M+ character-grounded dialogues, providing synchronized audio-text pairs annotated with speaker identities and contextual metadata. In addition, to demonstrate the effectiveness of the dataset, we introduced ARP-Eval, a dual-aspect evaluation framework that assesses both response quality and role fidelity. Empirical validation showing GLM-4-Voice trained on AudioRole (which we called ARP-Model) achieve an average Acoustic Personalization score of 0.31, significantly outperforming the original GLM-4-voice and the more powerful model MiniCPM-O-2.6, which specifically supports role-playing in one-shot scenarios. The ARP-Model also achieves a Content Personalization score of 0.36, surpassing the untrained original model by about 38% and maintaining the same level as MiniCPM-O-2.6. AudioRole features dialogues from over 115 main characters, 6 trained ARP-Models that role-play different characters, and evaluation protocols. Together, they provide an essential resource for advancing audio-grounded role-playing research.
>
---
#### [replaced 003] PhyAVBench: A Challenging Audio Physics-Sensitivity Benchmark for Physically Grounded Text-to-Audio-Video Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到音视频生成任务，旨在解决现有模型缺乏物理合理性音频的问题。提出PhyAVBench基准，包含可控物理变量的数据集和评估方法，以衡量生成模型的物理感知能力。**

- **链接: [https://arxiv.org/pdf/2512.23994](https://arxiv.org/pdf/2512.23994)**

> **作者:** Tianxin Xie; Wentao Lei; Kai Jiang; Guanjie Huang; Pengfei Zhang; Chunhui Zhang; Fengji Ma; Haoyu He; Han Zhang; Jiangshan He; Jinting Wang; Linghan Fang; Lufei Gao; Orkesh Ablet; Peihua Zhang; Ruolin Hu; Shengyu Li; Weilin Lin; Xiaoyang Feng; Xinyue Yang; Yan Rong; Yanyun Wang; Zihang Shao; Zelin Zhao; Chenxing Li; Shan Yang; Wenfu Wang; Meng Yu; Dong Yu; Li Liu
>
> **备注:** 6 major physical dimensions, 41 fine-grained test points, 337 groups of variable-controlled test samples, 11,605 newly recorded videos
>
> **摘要:** Text-to-audio-video (T2AV) generation is central to applications such as filmmaking and world modeling. However, current models often fail to produce physically plausible sounds. Previous benchmarks primarily focus on audio-video temporal synchronization, while largely overlooking explicit evaluation of audio-physics grounding, thereby limiting the study of physically plausible audio-visual generation. To address this issue, we present PhyAVBench, the first benchmark that systematically evaluates the audio-physics grounding capabilities of T2AV, image-to-audio-video (I2AV), and video-to-audio (V2A) models. PhyAVBench offers PhyAV-Sound-11K, a new dataset of 25.5 hours of 11,605 audible videos collected from 184 participants to ensure diversity and avoid data leakage. It contains 337 paired-prompt groups with controlled physical variations that drive sound differences, each grounded with an average of 17 videos and spanning 6 audio-physics dimensions and 41 fine-grained test points. Each prompt pair is annotated with the physical factors underlying their acoustic differences. Importantly, PhyAVBench leverages paired text prompts to evaluate this capability. We term this evaluation paradigm the Audio-Physics Sensitivity Test (APST) and introduce a novel metric, the Contrastive Physical Response Score (CPRS), which quantifies the acoustic consistency between generated videos and their real-world counterparts. We conduct a comprehensive evaluation of 17 state-of-the-art models. Our results reveal that even leading commercial models struggle with fundamental audio-physical phenomena, exposing a critical gap beyond audio-visual synchronization and pointing to future research directions. We hope PhyAVBench will serve as a foundation for advancing physically grounded audio-visual generation. Prompts, ground-truth, and generated video samples are available at this https URL.
>
---
#### [replaced 004] SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision
- **分类: eess.AS**

- **简介: 该论文提出SongFormer，解决音乐结构分析任务中的数据不足与标签不一致问题，通过融合自监督学习和源嵌入实现高效准确的结构分析。**

- **链接: [https://arxiv.org/pdf/2510.02797](https://arxiv.org/pdf/2510.02797)**

> **作者:** Chunbo Hao; Ruibin Yuan; Jixun Yao; Qixin Deng; Xinyi Bai; Yanbo Wang; Wei Xue; Lei Xie
>
> **摘要:** Music structure analysis (MSA) underpins music understanding and controllable generation, yet progress has been limited by small, inconsistent corpora. We present SongFormer, a scalable framework that learns from heterogeneous supervision. SongFormer (i) fuses short- and long-window self-supervised learning representations to capture both fine-grained and long-range dependencies, and (ii) introduces a learned source embedding to enable training with partial, noisy, and schema-mismatched labels. To support scaling and fair evaluation, we release SongFormDB, the largest MSA corpus to date (over 14k songs spanning languages and genres), and SongFormBench, a 300-song expert-verified benchmark. On SongFormBench, SongFormer sets a new state of the art in strict boundary detection (HR.5F) and achieves the highest functional label accuracy, while remaining computationally efficient; it surpasses strong baselines and Gemini 2.5 Pro on these metrics and remains competitive under relaxed tolerance (HR3F). Code, datasets, and model are open-sourced at this https URL.
>
---
#### [replaced 005] Disentangling peripheral hearing loss from central and cognitive effects on speech intelligibility in older adults
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音可懂度研究任务，旨在区分老年人大脑和认知因素对语音理解的影响。通过模拟听力损失并分析客观指标，探讨非听力因素的作用。**

- **链接: [https://arxiv.org/pdf/2510.25235](https://arxiv.org/pdf/2510.25235)**

> **作者:** Toshio Irino; Ayako Yamamoto; Fuki Miyazaki
>
> **备注:** This manuscript was submitted to Speech Communication on April 8, 2026
>
> **摘要:** Age-related hearing loss (HL) reduces speech intelligibility (SI) in older adults (OAs). However, deficits in central and cognitive processing also substantially impact SI. Understanding these contributions is essential for explaining individual differences and developing effective assistive hearing strategies. This study presents a framework that distinguishes peripheral HL from central and cognitive influences on SI. This framework uses the Wakayama University Hearing Impairment Simulator (WHIS), and the Gammachirp Envelope Similarity Index (GESI), an objective measure of intelligibility. First, speech-in-noise tests were conducted with young, normal-hearing listeners (YNHs) using WHIS to simulate the audiogram of a target OA. The target OA achieved SI scores comparable to or higher than those of YNHs with simulated HL, suggesting contributions beyond peripheral hearing function. Then, GESI was used to predict SI scores for YNHs and OAs across different hearing levels. The prediction accuracy was comparable for both groups. Interestingly, many OAs' subjective SI scores were higher than those predicted using parameters derived from YNHs' experiments. This finding is inconsistent with previous research indicating that speech perception ability declines with age. This issue will be discussed. There was no significant correlation between the average hearing levels and the residual differences between the subjective and predicted SI scores. This suggests that GESI effectively absorbed the effects of peripheral HL. Thus, the proposed framework may facilitate systematic examination and comparison of central and cognitive factors beyond peripheral HL among individual YNHs and OAs with and without HL.
>
---
#### [replaced 006] Modeling and Link Budget Feasibility Analysis of Secure LoRa-Based Peer-to-Peer Communication for Short-Range Tactical Networks
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于通信系统设计任务，旨在解决短距离安全通信问题。设计了一种基于LoRa的微型加密设备，实现1-1.5km可靠通信，采用AES-128加密和低功耗传输。**

- **链接: [https://arxiv.org/pdf/2602.23924](https://arxiv.org/pdf/2602.23924)**

> **作者:** Ayush Kumar Agrawal; Soumendu Das; Saptaparna De; Jayendra Kumar
>
> **摘要:** Short-range reliable and secure communication is a major priority in the tactical, military and disaster response settings where the traditional communication infrastructure is either off-line or prone to interception. Current VHF/UHF radios and software-defined radios are popular but large-sized devices and require lots of power, making them not suitable to be used as lightweight wearable devices with seamless hand-free use. In this paper, the design and theoretical framework of a miniature, LoRa based encrypted intercommunication device that can be used in secure field communication over a range of 1-1.5km and under line-of-sight conditions is provided. The suggested system consists of a voice-activated acquisition block, digital audio compression, an embedded microcontroller processor, and AES-128 encryption followed by a low-power transmission via the LoRa protocol. Through the ability of chirp spread spectrum modulation to utilize the long-range and low-energy properties, the system is guaranteed reliable communications coupled with low power consumption and low electromagnetic footprint. The theoretical analysis of the proposed communication range is justified using a link-budget that justifies the practicability of the communication range in the real propagation conditions. This architecture focuses on infrastructural agnosticism, peer-to-peer security as well as wearable ergonomics. The given scheme shows the possibilities of LoRa technology in the scope of other traditional IoT telemetry, and it can be further extended to include secure tactical voice communication platforms.
>
---
