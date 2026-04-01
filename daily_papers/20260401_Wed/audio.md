# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Real-Time Band-Grouped Vocal Denoising Using Sigmoid-Driven Ideal Ratio Masking
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音去噪任务，旨在解决实时应用中延迟高、需长上下文的问题。提出一种基于频带分组的编码器-解码器模型，使用谱损失优化SNR和语音质量。**

- **链接: [https://arxiv.org/pdf/2603.29326](https://arxiv.org/pdf/2603.29326)**

> **作者:** Daniel Williams
>
> **摘要:** Real-time, deep learning-based vocal denoising has seen significant progress over the past few years, demonstrating the capability of artificial intelligence in preserving the naturalness of the voice while increasing the signal-to-noise ratio (SNR). However, many deep learning approaches have high amounts of latency and require long frames of context, making them difficult to configure for live applications. To address these challenges, we propose a sigmoid-driven ideal ratio mask trained with a spectral loss to encourage an increased SNR and maximized perceptual quality of the voice. The proposed model uses a band-grouped encoder-decoder architecture with frequency attention and achieves a total latency of less than 10,ms, with PESQ-WB improvements of 0.21 on stationary noise and 0.12 on nonstationary noise.
>
---
#### [new 002] IQRA 2026: Interspeech Challenge on Automatic Assessment Pronunciation for Modern Standard Arabic (MSA)
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语发音评估任务，旨在解决自动误读检测与诊断问题。通过引入新数据集和多种模型方法，提升了F1分数，推动了相关研究发展。**

- **链接: [https://arxiv.org/pdf/2603.29087](https://arxiv.org/pdf/2603.29087)**

> **作者:** Yassine El Kheir; Amit Meghanani; Mostafa Shahin; Omnia Ibrahim; Shammur Absar Chowdhury; Nada AlMarwani; Youssef Elshahawy; Ahmed Ali
>
> **备注:** 5 pages paper
>
> **摘要:** We present the findings of the second edition of the IQRA Interspeech Challenge, a challenge on automatic Mispronunciation Detection and Diagnosis (MDD) for Modern Standard Arabic (MSA). Building on the previous edition, this iteration introduces \textbf{Iqra\_Extra\_IS26}, a new dataset of authentic human mispronounced speech, complementing the existing training and evaluation resources. Submitted systems employed a diverse range of approaches, spanning CTC-based self-supervised learning models, two-stage fine-tuning strategies, and using large audio-language models. Compared to the first edition, we observe a substantial jump of \textbf{0.28 in F1-score}, attributable both to novel architectures and modeling strategies proposed by participants and to the additional authentic mispronunciation data made available. These results demonstrate the growing maturity of Arabic MDD research and establish a stronger foundation for future work in Arabic pronunciation assessment.
>
---
#### [new 003] LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在解决传统方法依赖中间表示的问题。提出LongCat-AudioDiT模型，直接在波形潜在空间生成语音，提升语音质量与一致性。**

- **链接: [https://arxiv.org/pdf/2603.29339](https://arxiv.org/pdf/2603.29339)**

> **作者:** Detai Xin; Shujie Hu; Chengzuo Yang; Chen Huang; Guoqiao Yu; Guanglu Wan; Xunliang Cai
>
> **备注:** Code and model weights are available at this https URL
>
> **摘要:** We present LongCat-AudioDiT, a novel, non-autoregressive diffusion-based text-to-speech (TTS) model that achieves state-of-the-art (SOTA) performance. Unlike previous methods that rely on intermediate acoustic representations such as mel-spectrograms, the core innovation of LongCat-AudioDiT lies in operating directly within the waveform latent space. This approach effectively mitigates compounding errors and drastically simplifies the TTS pipeline, requiring only a waveform variational autoencoder (Wav-VAE) and a diffusion backbone. Furthermore, we introduce two critical improvements to the inference process: first, we identify and rectify a long-standing training-inference mismatch; second, we replace traditional classifier-free guidance with adaptive projection guidance to elevate generation quality. Experimental results demonstrate that, despite the absence of complex multi-stage training pipelines or high-quality human-annotated datasets, LongCat-AudioDiT achieves SOTA zero-shot voice cloning performance on the Seed benchmark while maintaining competitive intelligibility. Specifically, our largest variant, LongCat-AudioDiT-3.5B, outperforms the previous SOTA model (Seed-TTS), improving the speaker similarity (SIM) scores from 0.809 to 0.818 on Seed-ZH, and from 0.776 to 0.797 on Seed-Hard. Finally, through comprehensive ablation studies and systematic analysis, we validate the effectiveness of our proposed modules. Notably, we investigate the interplay between the Wav-VAE and the TTS backbone, revealing the counterintuitive finding that superior reconstruction fidelity in the Wav-VAE does not necessarily lead to better overall TTS performance. Code and model weights are released to foster further research within the speech community.
>
---
#### [new 004] A Comprehensive Corpus of Biomechanically Constrained Piano Chords: Generation, Analysis, and Implications for Voicing and Psychoacoustics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐与声学交叉研究，旨在分析钢琴和弦的生物力学约束及听觉特性。通过构建大规模数据集，研究和弦结构对音色的影响，揭示音高分布与不和谐感的关系。**

- **链接: [https://arxiv.org/pdf/2603.29710](https://arxiv.org/pdf/2603.29710)**

> **作者:** Mahesh Ramani
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** I present the generation and analysis of the largest known open-source corpus of playable piano chords (approximately 19.3 million entries). This dataset enumerates the two-handed search space subject to biomechanical constraints (two hands, each with 1.5 octave reach) to an unprecedented extent. To demonstrate the corpus's utility, the relationship between voicing shape and psychoacoustic targets was modeled. Harmonicity proved intrinsic to pitch-class identity: voicing statistics added negligible variance ($\Delta R^2 \approx 0.014\%$, $p \approx 0.13$). Conversely, voicing significantly predicted dissonance ($\Delta R^2 \approx 6.75\%$, $p \approx 0.0008$). Crucially, skewness ($\beta \approx +0.145$) was approximately 5.8$\times$ more effective than spread ($\beta \approx -0.025$) at predicting roughness. The analysis challenges the pedagogical emphasis on ``spread'': skewness is a stronger predictor of dissonance than spread. This suggests that clarity in ``open voicings'' is driven less by width than by negative skewness; achieving lower-register clearance by placing wide gaps at the bottom and allowing tighter clustering in the treble. The results demonstrate the corpus's ability to enable future research, especially in areas such as generative modeling, voice-leading topology, and psychoacoustic analysis.
>
---
#### [new 005] SIREN: Spatially-Informed Reconstruction of Binaural Audio with Vision
- **分类: cs.SD**

- **简介: 论文提出SIREN，解决单声道视频转双声道音频的问题。通过视觉信息引导，生成左右声道信号，提升音频空间感。**

- **链接: [https://arxiv.org/pdf/2603.29820](https://arxiv.org/pdf/2603.29820)**

> **作者:** Mingyeong Song; Seoyeon Ko; Junhyug Noh
>
> **备注:** 5 pages, 1 figure, to appear in ICASSP 2026
>
> **摘要:** Binaural audio delivers spatial cues essential for immersion, yet most consumer videos are monaural due to capture constraints. We introduce SIREN, a visually guided mono to binaural framework that explicitly predicts left and right channels. A ViT-based encoder learns dual-head self-attention to produce a shared scene map and end-to-end L/R attention, replacing hand-crafted masks. A soft, annealed spatial prior gently biases early L/R grounding, and a two-stage, confidence-weighted waveform-domain fusion (guided by mono reconstruction and interaural phase consistency) suppresses crosstalk when aggregating multi-crop and overlapping windows. Evaluated on FAIR-Play and MUSIC-Stereo, SIREN yields consistent gains on time-frequency and phase-sensitive metrics with competitive SNR. The design is modular and generic, requires no task-specific annotations, and integrates with standard audio-visual pipelines.
>
---
#### [new 006] Asymmetric Encoder-Decoder Based on Time-Frequency Correlation for Speech Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决复杂环境下的语音分离问题。提出SR-CorrNet框架，通过时间-频率域的分离与重建策略提升分离效果。**

- **链接: [https://arxiv.org/pdf/2603.29097](https://arxiv.org/pdf/2603.29097)**

> **作者:** Ui-Hyeop Shin; Hyung-Min Park
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing (T-ASLP)
>
> **摘要:** Speech separation in realistic acoustic environments remains challenging because overlapping speakers, background noise, and reverberation must be resolved simultaneously. Although recent time-frequency (TF) domain models have shown strong performance, most still rely on late-split architectures, where speaker disentanglement is deferred to the final stage, creating an information bottleneck and weakening discriminability under adverse conditions. To address this issue, we propose SR-CorrNet, an asymmetric encoder-decoder framework that introduces the separation-reconstruction (SepRe) strategy into a TF dual-path backbone. The encoder performs coarse separation from mixture observations, while the weight-shared decoder progressively reconstructs speaker-discriminative features with cross-speaker interaction, enabling stage-wise refinement. To complement this architecture, we formulate speech separation as a structured correlation-to-filter problem: spatio-spectro-temporal correlations computed from the observations are used as input features, and the corresponding deep filters are estimated to recover target signals. We further incorporate an attractor-based dynamic split module to adapt the number of output streams to the actual speaker configuration. Experimental results on WSJ0-2/3/4/5Mix, WHAMR!, and LibriCSS demonstrate consistent improvements across anechoic, noisy-reverberant, and real-recorded conditions in both single- and multi-channel settings, highlighting the effectiveness of TF-domain SepRe with correlation-based filter estimation for speech separation.
>
---
#### [new 007] Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多语言语音识别任务，解决多语言P2G中的语言感知生成和数据不平衡问题。通过改进训练策略提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.29217](https://arxiv.org/pdf/2603.29217)**

> **作者:** Lukuang Dong; Ziwei Li; Saierdaer Yusuyin; Xianyu Zhao; Zhijian Ou
>
> **备注:** Update after INTERSPEECH2026 submission
>
> **摘要:** Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%.
>
---
#### [new 008] Audio Hallucination Attacks: Probing the Reliability of Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型安全研究，旨在检测LALMs的可靠性问题。通过构造攻击数据集，验证模型是否基于真实音频生成回答，并提出防御数据集以提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.29263](https://arxiv.org/pdf/2603.29263)**

> **作者:** Ashish Seth; Sonal Kumar; Ramaneswaran Selvakumar; Nishit Anand; Utkarsh Tyagi; Prem Seetharaman; Ramani Duraiswami; Dinesh Manocha
>
> **摘要:** Large Audio Language Models (LALMs) achieve strong performance on audio-language tasks; however, their reliability in real-world settings remains underexplored. We introduce Audio Hallucination Attacks (AHA), an attack suite called AHA-Eval, comprising 6.5K QA pairs designed to test whether LALMs genuinely ground their responses in the audio input. AHA targets two attack surfaces: (i) query-based attacks, which exploit question structure to induce hallucinations about absent sounds, and (ii) audio-based attacks, which inject synthetic speech describing non-existent events into the audio stream. Evaluating state-of-the-art LALMs, including Audio Flamingo 3 and Gemini 3 Pro, we observe high attack success rates of 95.35% and 79.65%, respectively, revealing a reliability gap that is hidden by standard benchmark performance. To mitigate this, we propose a 120K QA post-alignment dataset, AHA-Guard, which successfully reduces attack success rates by up to 49%.
>
---
#### [new 009] An Information-Theoretic Method for Dynamic System Identification With Output-Only Damping Estimation
- **分类: eess.SP; eess.AS; eess.SY**

- **简介: 该论文属于系统识别任务，旨在解决机械系统中阻尼估计不准确的问题。通过信息理论方法提升振动监测的准确性，实现更可靠的预警。**

- **链接: [https://arxiv.org/pdf/2603.29956](https://arxiv.org/pdf/2603.29956)**

> **作者:** Marios Impraimakis; Feiyu Zhou; Andrew Plummer
>
> **备注:** 18 pages, 16 figures, 4 tables. Published in Journal of Dynamic Systems, Measurement, and Control (ASME), 2026. Licensed under CC BY 4.0
>
> **摘要:** The system identification capabilities of a novel information-theoretic method are examined here. Specifically, this work uses information-theoretic metrics and vibration-based measurements to enhance damping estimation accuracy in mechanical systems. The method refers to a key limitation in system identification, signal processing, monitoring, and alert systems. These systems integrate various components, including sensors, data acquisition devices, and alert mechanisms. They are designed to operate in an environment to calculate key parameters such as peak accelerations and duration of high acceleration values. The current operational modal identification methods, though, suffer from limitations related to obtaining poor damping estimates due to their empirical nature. This has a significant impact on alert warning systems. This occurs when their duration is misestimated; specifically, when using the vibration amplitudes as an indicator of danger alerts for monitoring systems in damage or anomaly detection scenarios. To this end, approaches based on the Shannon entropy and the Kullback-Leibler divergence concept are proposed. The primary objective is to monitor the vibration levels in near real-time and provide immediate alerts when predefined thresholds are exceeded. In considering the proposed approach, both new real-world data from the multi-axis simulation table at the University of Bath, as well as the benchmark International Association for Structural Control-American Society of Civil Engineers (IASC-ASCE) structural health monitoring problem are considered. Importantly, the approach is shown to select the optimal model, which accurately captures the correct alert duration, providing a powerful tool for system identification and monitoring.
>
---
#### [new 010] Covertly improving intelligibility with data-driven adaptations of speech timing
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音处理任务，旨在提升语音可懂性。通过分析语音速率对理解的影响，提出数据驱动的语音调整方法，显著提高不同听众在困难条件下的理解能力。**

- **链接: [https://arxiv.org/pdf/2603.30032](https://arxiv.org/pdf/2603.30032)**

> **作者:** Paige Tuttösí; Angelica Lim; H. Henny Yeung; Yue Wang; Jean-Julien Aucouturier
>
> **摘要:** Human talkers often address listeners with language-comprehension challenges, such as hard-of-hearing or non-native adults, by globally slowing down their speech. However, it remains unclear whether this strategy actually makes speech more intelligible. Here, we take advantage of recent advancements in machine-generated speech allowing more precise control of speech rate in order to systematically examine how targeted speech-rate adjustments may improve comprehension. We first use reverse-correlation experiments to show that the temporal influence of speech rate prior to a target vowel contrast (ex. the tense-lax distinction) in fact manifests in a scissor-like pattern, with opposite effects in early versus late context windows; this pattern is remarkably stable both within individuals and across native L1-English listeners and L2-English listeners with French, Mandarin, and Japanese L1s. Second, we show that this speech rate structure not only facilitates L2 listeners' comprehension of the target vowel contrast, but that native listeners also rely on this pattern in challenging acoustic conditions. Finally, we build a data-driven text-to-speech algorithm that replicates this temporal structure on novel speech sequences. Across a variety of sentences and vowel contrasts, listeners remained unaware that such targeted slowing improved word comprehension. Strikingly, participants instead judged the common strategy of global slowing as clearer, even though it actually increased comprehension errors. Together, these results show that targeted adjustments to speech rate significantly aid intelligibility under challenging conditions, while often going unnoticed. More generally, this paper provides a data-driven methodology to improve the accessibility of machine-generated speech which can be extended to other aspects of speech comprehension and a wide variety of listeners and environments.
>
---
## 更新

#### [replaced 001] Habibi: Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多方言阿拉伯语语音合成任务，旨在解决方言差异大、数据少及缺乏基准的问题。通过构建统一模型和基准数据集，实现跨方言高质量合成。**

- **链接: [https://arxiv.org/pdf/2601.13802](https://arxiv.org/pdf/2601.13802)**

> **作者:** Yushen Chen; Junzhe Liu; Yujie Tu; Zhikang Niu; Yuzhe Liang; Chunyu Qiang; Chen Zhang; Kai Yu; Xie Chen
>
> **摘要:** Arabic spans over 30 spoken varieties, yet no open-source text-to-speech system unifies them. Key barriers include substantial cross-dialect lexical and phonological divergence, scarce synthesis-grade data, and the absence of a standardized multi-dialect evaluation benchmark. We present Habibi, a unified-dialectal Arabic TTS framework that addresses all three. Through a multi-step curation pipeline, we repurpose open-source ASR corpora into TTS training data covering 12+ regional dialects. A linguistically-informed curriculum learning strategy - progressing from Modern Standard Arabic to dialectal data - enables robust zero-shot synthesis without text diacritization. We further release the first standardized multi-dialect Arabic TTS benchmark, comprising over 11,000 utterances across 7 dialect subsets with manually verified transcripts. On this benchmark, our unified model matches or surpasses per-dialect specialized models. Both automatic metrics and human evaluations confirm that Habibi is highly competitive with ElevenLabs' Eleven v3 (alpha) in intelligibility, speaker similarity, and naturalness. Extensive ablations (~8,000 H100 GPU hours, 30+ configurations) validate each design choice. We open-source all checkpoints, training and inference code, and benchmark data - the first such release for multi-dialect Arabic TTS - at this https URL .
>
---
#### [replaced 002] POTSA: A Cross-Lingual Speech Alignment Framework for Speech-to-Text Translation
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音到文本翻译任务，旨在解决多语言翻译中的语义偏差问题。提出POTSA框架，通过跨语言对齐和最优传输技术提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.09232](https://arxiv.org/pdf/2511.09232)**

> **作者:** Xuanchen Li; Chenrui Cui; Tianrui Wang; Meng Ge; Zikang Huang; Yizhou Peng; Jin Li; Yuheng Lu; Yu Jiang; Nyima Tashi; Longbiao Wang; Jianwu Dang
>
> **摘要:** Speech Large Language Models have achieved breakthroughs in multilingual speech-to-text translation. However, existing approaches often overlook semantic commonalities across source languages, leading to biased translation performance. In this work, we propose POTSA (Parallel Optimal Transport for Speech Alignment), a new framework based on cross-lingual parallel speech pairs and Optimal Transport, designed to bridge high- and low-resource translation gaps. First, we introduce a Bias Compensation module to coarsely align initial speech representations. Second, we impose token-level OT constraints on a Q-Former using parallel pairs to establish fine-grained representation consistency. Then, we apply a layer scheduling strategy to focus OT constraints on semantically beneficial layers. Experiments on FLEURS show our method achieves SOTA performance, with +1.29 BLEU over five common languages and +2.93 BLEU on zero-shot languages, using only 10 hours of parallel speech per language.
>
---
#### [replaced 003] Audio Language Model for Deepfake Detection Grounded in Acoustic Chain-of-Thought
- **分类: cs.SD**

- **简介: 该论文属于深度伪造语音检测任务，旨在解决现有系统缺乏可解释性的问题。通过引入结构化声学特征和链式思维推理，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.28021](https://arxiv.org/pdf/2603.28021)**

> **作者:** Runkun Chen; Yixiong Fang; Pengyu Chang; Yuante Li; Massa Baali; Bhiksha Raj
>
> **摘要:** Deepfake speech detection systems are often limited to binary classification tasks and struggle to generate interpretable reasoning or provide context-rich explanations for their decisions. These models primarily extract latent embeddings for authenticity detection but fail to leverage structured acoustic evidence such as prosodic, spectral, and physiological attributes in a meaningful manner. This paper introduces CoLMbo-DF, a Feature-Guided Audio Language Model that addresses these limitations by integrating robust deepfake detection with explicit acoustic chain-of-thought reasoning. By injecting structured textual representations of low-level acoustic features directly into the model prompt, our approach grounds the model's reasoning in interpretable evidence and improves detection accuracy. To support this framework, we introduce a novel dataset of audio pairs paired with chain-of-thought annotations. Experiments show that our method, trained on a lightweight open-source language model, significantly outperforms existing audio language model baselines despite its smaller scale, marking a significant advancement in explainable deepfake speech detection.
>
---
#### [replaced 004] VAANI: Capturing the language landscape for an inclusive digital India
- **分类: eess.AS**

- **简介: 该论文介绍VAANI项目，旨在构建涵盖印度165个地区的多模态数据集，解决语言多样性不足的问题，通过收集语音和图像数据，促进包容性数字技术发展。**

- **链接: [https://arxiv.org/pdf/2603.28714](https://arxiv.org/pdf/2603.28714)**

> **作者:** Sujith Pulikodan; Abhayjeet Singh; Agneedh Basu; Nihar Desai; Pavan Kumar J; Pranav D Bhat; Raghu Dharmaraju; Ritika Gupta; Sathvik Udupa; Saurabh Kumar; Sumit Sharma; Vaibhav Vishwakarma; Visruth Sanka; Dinesh Tewari; Harsh Dhand; Amrita Kamat; Sukhwinder Singh; Shikhar Vashishth; Partha Talukdar; Raj Acharya; Prasanta Kumar Ghosh
>
> **摘要:** Project VAANI is an initiative to create an India-representative multi-modal dataset that comprehensively maps India's linguistic diversity, starting with 165 districts across the country in its first two phases. Speech data is collected through a carefully structured process that uses image-based prompts to encourage spontaneous responses. Images are captured through a separate process that encompasses a broad range of topics, gathered from both within and across districts. The collected data undergoes a rigorous multi-stage quality evaluation, including both automated and manual checks to ensure highest possible standards in audio quality and transcription accuracy. Following this thorough validation, we have open-sourced around 289K images, approximately 31,270 hours of audio recordings, and around 2,067 hours of transcribed speech, encompassing 112 languages from 165 districts from 31 States and Union territories. Notably, significant of these languages are being represented for the first time in a dataset of this scale, making the VAANI project a groundbreaking effort in preserving and promoting linguistic inclusivity. This data can be instrumental in building inclusive speech models for India, and in advancing research and development across speech, image, and multimodal applications.
>
---
#### [replaced 005] EchoMark: Perceptual Acoustic Environment Transfer with Watermark-Embedded Room Impulse Response
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于声学环境迁移任务，旨在解决恶意用户滥用导致的漏洞问题。提出EchoMark框架，通过嵌入水印生成感知相似的RIR，实现高质量环境转换与可靠水印检测。**

- **链接: [https://arxiv.org/pdf/2511.06458](https://arxiv.org/pdf/2511.06458)**

> **作者:** Chenpei Huang; Lingfeng Yao; Kyu In Lee; Lan Emily Zhang; Xun Chen; Miao Pan
>
> **摘要:** Acoustic Environment Matching (AEM) is the task of transferring clean audio into a target acoustic environment, enabling engaging applications such as audio dubbing and auditory immersive virtual reality (VR). Recovering similar room impulse response (RIR) directly from reverberant speech offers more accessible and flexible AEM solution. However, this capability also introduces vulnerabilities of arbitrary ``relocation" if misused by malicious user, such as facilitating advanced voice spoofing attacks or undermining the authenticity of recorded evidence. To address this issue, we propose EchoMark, the first deep learning-based AEM framework that generates perceptually similar RIRs with embedded watermark. Our design tackle the challenges posed by variable RIR characteristics, such as different durations and energy decays, by operating in the latent domain. By jointly optimizing the model with a perceptual loss for RIR reconstruction and a loss for watermark detection, EchoMark achieves both high-quality environment transfer and reliable watermark recovery. Experiments on diverse datasets validate that EchoMark achieves room acoustic parameter matching performance comparable to FiNS, the state-of-the-art RIR estimator. Furthermore, a high Mean Opinion Score (MOS) of 4.22 out of 5, watermark detection accuracy exceeding 99\%, and bit error rates (BER) below 0.3\% collectively demonstrate the effectiveness of EchoMark in preserving perceptual quality while ensuring reliable watermark embedding.
>
---
