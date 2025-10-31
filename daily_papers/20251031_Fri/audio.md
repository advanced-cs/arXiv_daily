# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models
- **分类: cs.SD; cs.CR; cs.LG**

- **简介: 该论文针对音频-语言模型（ALM）的安全性问题，提出ALMGuard防御框架。针对ALM特有的越狱攻击，通过识别安全快捷方式并设计通用触发器（SAPs）实现防护，结合梅尔频谱稀疏掩码（M-GSM）确保有效性与实用性。实验表明，可将攻击成功率降至4.6%，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.26096v1](http://arxiv.org/pdf/2510.26096v1)**

> **作者:** Weifei Jin; Yuxin Cao; Junjie Su; Minhui Xue; Jie Hao; Ke Xu; Jin Song Dong; Derui Wang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.
>
---
#### [new 002] UniTok-Audio: A Unified Audio Generation Framework via Generative Modeling on Discrete Codec Tokens
- **分类: cs.SD**

- **简介: 该论文提出UniTok-Audio，一个统一的音频生成框架，旨在解决多任务音频生成中质量差、泛化弱及系统碎片化问题。通过离散编码器令牌建模、任务标识符统一学习模式与双流编解码结构，实现五类时序对齐任务的高性能统一处理。**

- **链接: [http://arxiv.org/pdf/2510.26372v1](http://arxiv.org/pdf/2510.26372v1)**

> **作者:** Chengwei Liu; Haoyin Yan; Shaofei Xue; Xiaotao Liang; Yinghao Liu; Zheng Xue; Gang Song; Boyang Zhou
>
> **备注:** 21 pages, 3 figures
>
> **摘要:** Generative modeling has recently achieved remarkable success across text, image, and audio domains, demonstrating powerful capabilities for unified representation learning. However, audio generation models still face challenges in terms of audio quality and generalization ability across tasks. This fragmentation results in redundant development efforts, inconsistent performance, and limited extensibility. To address these issues, we propose \textbf{UniTok-Audio}, a scalable and extensible framework for unified audio generation tasks. Specifically, 1) UniTok-Audio extracts continuous feature of conditions to generates discrete tokens of target audio in an autoregressive manner; 2) a special task identifier token unifies different learning patterns of multiple tasks in a single framework; 3) a dual-stream audio codec involving acoustic and semantic branch is developed for high-fidelity waveform reconstruction. Experimental results demonstrate that UniTok-Audio achieves competitive performance in comparation with state-of-the-art task-specific or multi-task systems across five time-aligned tasks: speech restoration, target speaker extraction, speech separation, voice conversion, and language-queried audio source separation. To foster future research, we will open-source our codebase. The demo page of our work can be found here: https://alibaba.github.io/unified-audio.
>
---
#### [new 003] Modeling strategies for speech enhancement in the latent space of a neural audio codec
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究神经音频编解码器中连续与离散语音表示在语音增强任务中的效果。对比了基于连续向量和离散标记的自回归与非自回归模型，发现连续表示性能更优，非自回归模型更实用，而编码器微调虽提升增强效果但损害重建质量。**

- **链接: [http://arxiv.org/pdf/2510.26299v1](http://arxiv.org/pdf/2510.26299v1)**

> **作者:** Sofiene Kammoun; Xavier Alameda-Pineda; Simon Leglaive
>
> **摘要:** Neural audio codecs (NACs) provide compact latent speech representations in the form of sequences of continuous vectors or discrete tokens. In this work, we investigate how these two types of speech representations compare when used as training targets for supervised speech enhancement. We consider both autoregressive and non-autoregressive speech enhancement models based on the Conformer architecture, as well as a simple baseline where the NAC encoder is simply fine-tuned for speech enhancement. Our experiments reveal three key findings: predicting continuous latent representations consistently outperforms discrete token prediction; autoregressive models achieve higher quality but at the expense of intelligibility and efficiency, making non-autoregressive models more attractive in practice; and encoder fine-tuning yields the strongest enhancement metrics overall, though at the cost of degraded codec reconstruction. The code and audio samples are available online.
>
---
#### [new 004] SP-MCQA: Evaluating Intelligibility of TTS Beyond the Word Level
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文针对语音合成（TTS）智能评估瓶颈，提出SP-MCQA新任务，通过关键信息多选题评估合成语音的语义理解能力。解决传统词级准确率（WER）无法反映真实听觉理解的问题。构建8.76小时新闻数据集，揭示高WER下关键信息丢失现象，强调需发展更贴近人类认知的评估标准。**

- **链接: [http://arxiv.org/pdf/2510.26190v1](http://arxiv.org/pdf/2510.26190v1)**

> **作者:** Hitomi Jin Ling Tee; Chaoren Wang; Zijie Zhang; Zhizheng Wu
>
> **摘要:** The evaluation of intelligibility for TTS has reached a bottleneck, as existing assessments heavily rely on word-by-word accuracy metrics such as WER, which fail to capture the complexity of real-world speech or reflect human comprehension needs. To address this, we propose Spoken-Passage Multiple-Choice Question Answering, a novel subjective approach evaluating the accuracy of key information in synthesized speech, and release SP-MCQA-Eval, an 8.76-hour news-style benchmark dataset for SP-MCQA evaluation. Our experiments reveal that low WER does not necessarily guarantee high key-information accuracy, exposing a gap between traditional metrics and practical intelligibility. SP-MCQA shows that even state-of-the-art (SOTA) models still lack robust text normalization and phonetic accuracy. This work underscores the urgent need for high-level, more life-like evaluation criteria now that many systems already excel at WER yet may fall short on real-world intelligibility.
>
---
#### [new 005] SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations
- **分类: eess.AS**

- **简介: 该论文提出SPEAR框架，旨在统一学习语音与通用音频的自监督表示。针对现有方法局限于单一领域的问题，SPEAR通过多码本量化生成细粒度离散令牌，设计统一预训练目标，在混合数据上实现跨域表征学习，并在SUPERB和HEAR基准上验证了其卓越性能。**

- **链接: [http://arxiv.org/pdf/2510.25955v1](http://arxiv.org/pdf/2510.25955v1)**

> **作者:** Xiaoyu Yang; Yifan Yang; Zengrui Jin; Ziyun Cui; Wen Wu; Baoxiang Li; Chao Zhang; Phil Woodland
>
> **摘要:** Self-Supervised Learning (SSL) excels at learning generic representations of acoustic signals, yet prevailing methods remain domain-specific, tailored to either speech or general audio, hindering the development of a unified representation model with a comprehensive capability over both domains. To address this, we present SPEAR (SPEech and Audio Representations), the first SSL framework to successfully learn unified speech and audio representations from a mixture of speech and audio data. SPEAR proposes a unified pre-training objective based on masked prediction of fine-grained discrete tokens for both speech and general audio. These tokens are derived from continuous speech and audio representations using a Multi-codebook Vector Quantisation (MVQ) method, retaining rich acoustic detail essential for modelling both speech and complex audio events. SPEAR is applied to pre-train both single-domain and unified speech-and-audio SSL models. Our speech-domain model establishes a new state-of-the-art on the SUPERB benchmark, a speech processing benchmark for SSL models, matching or surpassing the highly competitive WavLM Large on 12 out of 15 tasks with the same pre-training corpora and a similar model size. Crucially, our unified model learns complementary features and demonstrates comprehensive capabilities across two major benchmarks, SUPERB and HEAR, for evaluating audio representations. By further scaling up the model size and pre-training data, we present a unified model with 600M parameters that excels in both domains, establishing it as one of the most powerful and versatile open-source SSL models for auditory understanding. The inference code and pre-trained models will be made publicly available.
>
---
#### [new 006] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders
- **分类: cs.LG; cs.SD**

- **简介: 该论文针对音频生成模型中隐空间语义不透明的问题，提出基于稀疏自编码器（SAE）的可解释特征学习框架。通过在音频潜空间上训练SAE，并建立线性映射至音高、振幅、音色等可听概念，实现对生成过程的可控操控与分析，揭示声学属性的演化机制。**

- **链接: [http://arxiv.org/pdf/2510.23802v1](http://arxiv.org/pdf/2510.23802v1)**

> **作者:** Nathan Paek; Yongyi Zang; Qihui Yang; Randal Leistikow
>
> **备注:** Accepted to NeurIPS 2025 Mechanistic Interpretability Workshop
>
> **摘要:** While sparse autoencoders (SAEs) successfully extract interpretable features from language models, applying them to audio generation faces unique challenges: audio's dense nature requires compression that obscures semantic meaning, and automatic feature characterization remains limited. We propose a framework for interpreting audio generative models by mapping their latent representations to human-interpretable acoustic concepts. We train SAEs on audio autoencoder latents, then learn linear mappings from SAE features to discretized acoustic properties (pitch, amplitude, and timbre). This enables both controllable manipulation and analysis of the AI music generation process, revealing how acoustic properties emerge during synthesis. We validate our approach on continuous (DiffRhythm-VAE) and discrete (EnCodec, WavTokenizer) audio latent spaces, and analyze DiffRhythm, a state-of-the-art text-to-music model, to demonstrate how pitch, timbre, and loudness evolve throughout generation. While our work is only done on audio modality, our framework can be extended to interpretable analysis of visual latent space generation models.
>
---
#### [new 007] POWSM: A Phonetic Open Whisper-Style Speech Foundation Model
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出POWSM，首个统一框架，联合完成语音识别、音素识别、音素转文字等多任务。解决传统方法孤立处理各任务的问题，实现音频、文本与音素间的无缝转换，提升低资源场景下的通用性与效率。**

- **链接: [http://arxiv.org/pdf/2510.24992v1](http://arxiv.org/pdf/2510.24992v1)**

> **作者:** Chin-Jou Li; Kalvin Chang; Shikhar Bharadwaj; Eunjung Yeo; Kwanghee Choi; Jian Zhu; David Mortensen; Shinji Watanabe
>
> **备注:** 14 pages, under review
>
> **摘要:** Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.
>
---
## 更新

#### [replaced 001] ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24518v2](http://arxiv.org/pdf/2505.24518v2)**

> **作者:** Jiatong Shi; Yifan Cheng; Bo-Hao Su; Hye-jin Shim; Jinchuan Tian; Samuele Cornell; Yiwen Zhao; Siddhant Arora; Shinji Watanabe
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Speech signal analysis poses significant challenges, particularly in tasks such as speech quality evaluation and profiling, where the goal is to predict multiple perceptual and objective metrics. For instance, metrics like PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility), and MOS (Mean Opinion Score) each capture different aspects of speech quality. However, these metrics often have different scales, assumptions, and dependencies, making joint estimation non-trivial. To address these issues, we introduce ARECHO (Autoregressive Evaluation via Chain-based Hypothesis Optimization), a chain-based, versatile evaluation system for speech assessment grounded in autoregressive dependency modeling. ARECHO is distinguished by three key innovations: (1) a comprehensive speech information tokenization pipeline; (2) a dynamic classifier chain that explicitly captures inter-metric dependencies; and (3) a two-step confidence-oriented decoding algorithm that enhances inference reliability. Experiments demonstrate that ARECHO significantly outperforms the baseline framework across diverse evaluation scenarios, including enhanced speech analysis, speech generation evaluation, and, noisy speech evaluation. Furthermore, its dynamic dependency modeling improves interpretability by capturing inter-metric relationships. Across tasks, ARECHO offers reference-free evaluation using its dynamic classifier chain to support subset queries (single or multiple metrics) and reduces error propagation via confidence-oriented decoding.
>
---
#### [replaced 002] Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.25054v2](http://arxiv.org/pdf/2510.25054v2)**

> **作者:** Pedro Corrêa; João Lima; Victor Moreno; Lucas Ueda; Paula Dornhofer Paro Costa
>
> **备注:** Submitted to IEEE ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Advancements in spoken language processing have driven the development of spoken language models (SLMs), designed to achieve universal audio understanding by jointly learning text and audio representations for a wide range of tasks. Although promising results have been achieved, there is growing discussion regarding these models' generalization capabilities and the extent to which they truly integrate audio and text modalities in their internal representations. In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results indicate that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.
>
---
#### [replaced 003] DiffRhythm 2: Efficient and High Fidelity Song Generation via Block Flow Matching
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2510.22950v2](http://arxiv.org/pdf/2510.22950v2)**

> **作者:** Yuepeng Jiang; Huakang Chen; Ziqian Ning; Jixun Yao; Zerui Han; Di Wu; Meng Meng; Jian Luan; Zhonghua Fu; Lei Xie
>
> **摘要:** Generating full-length, high-quality songs is challenging, as it requires maintaining long-term coherence both across text and music modalities and within the music modality itself. Existing non-autoregressive (NAR) frameworks, while capable of producing high-quality songs, often struggle with the alignment between lyrics and vocal. Concurrently, catering to diverse musical preferences necessitates reinforcement learning from human feedback (RLHF). However, existing methods often rely on merging multiple models during multi-preference optimization, which results in significant performance degradation. To address these challenges, we introduce DiffRhythm 2, an end-to-end framework designed for high-fidelity, controllable song generation. To tackle the lyric alignment problem, DiffRhythm 2 employs a semi-autoregressive architecture based on block flow matching. This design enables faithful alignment of lyrics to singing vocals without relying on external labels and constraints, all while preserving the high generation quality and efficiency of NAR models. To make this framework computationally tractable for long sequences, we implement a music variational autoencoder (VAE) that achieves a low frame rate of 5 Hz while still enabling high-fidelity audio reconstruction. In addition, to overcome the limitations of multi-preference optimization in RLHF, we propose cross-pair preference optimization. This method effectively mitigates the performance drop typically associated with model merging, allowing for more robust optimization across diverse human preferences. We further enhance musicality and structural coherence by introducing stochastic block representation alignment loss.
>
---
#### [replaced 004] Audio Signal Processing Using Time Domain Mel-Frequency Wavelet Coefficient
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.24519v2](http://arxiv.org/pdf/2510.24519v2)**

> **作者:** Rinku Sebastian; Simon O'Keefe; Martin Trefzer
>
> **摘要:** Extracting features from the speech is the most critical process in speech signal processing. Mel Frequency Cepstral Coefficients (MFCC) are the most widely used features in the majority of the speaker and speech recognition applications, as the filtering in this feature is similar to the filtering taking place in the human ear. But the main drawback of this feature is that it provides only the frequency information of the signal but does not provide the information about at what time which frequency is present. The wavelet transform, with its flexible time-frequency window, provides time and frequency information of the signal and is an appropriate tool for the analysis of non-stationary signals like speech. On the other hand, because of its uniform frequency scaling, a typical wavelet transform may be less effective in analysing speech signals, have poorer frequency resolution in low frequencies, and be less in line with human auditory perception. Hence, it is necessary to develop a feature that incorporates the merits of both MFCC and wavelet transform. A great deal of studies are trying to combine both these features. The present Wavelet Transform based Mel-scaled feature extraction methods require more computation when a wavelet transform is applied on top of Mel-scale filtering, since it adds extra processing steps. Here we are proposing a method to extract Mel scale features in time domain combining the concept of wavelet transform, thus reducing the computational burden of time-frequency conversion and the complexity of wavelet extraction. Combining our proposed Time domain Mel frequency Wavelet Coefficient(TMFWC) technique with the reservoir computing methodology has significantly improved the efficiency of audio signal processing.
>
---
#### [replaced 005] Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.20410v3](http://arxiv.org/pdf/2509.20410v3)**

> **作者:** Weijie Wu; Wenhao Guan; Kaidi Wang; Peijie Chen; Zhuanling Zha; Junbo Li; Jun Fang; Lin Li; Qingyang Hong
>
> **备注:** It requires internal PR approval
>
> **摘要:** Spoken dialogue models have significantly advanced intelligent human-computer interaction, yet they lack a plug-and-play full-duplex prediction module for semantic endpoint detection, hindering seamless audio interactions. In this paper, we introduce Phoenix-VAD, an LLM-based model that enables streaming semantic endpoint detection. Specifically, Phoenix-VAD leverages the semantic comprehension capability of the LLM and a sliding window training strategy to achieve reliable semantic endpoint detection while supporting streaming inference. Experiments on both semantically complete and incomplete speech scenarios indicate that Phoenix-VAD achieves excellent and competitive performance. Furthermore, this design enables the full-duplex prediction module to be optimized independently of the dialogue model, providing more reliable and flexible support for next-generation human-computer interaction.
>
---
#### [replaced 006] Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model
- **分类: cs.MM; cs.CL; cs.IR; cs.SD; eess.AS; 68T, 68T45, 68T10**

- **链接: [http://arxiv.org/pdf/2503.09205v3](http://arxiv.org/pdf/2503.09205v3)**

> **作者:** Ali Vosoughi; Dimitra Emmanouilidou; Hannes Gamper
>
> **备注:** 5 pages, 5 figures, 2 tables. Accepted at EUSIPCO 2025
>
> **摘要:** Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
>
---
