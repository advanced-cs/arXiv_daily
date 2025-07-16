# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] FasTUSS: Faster Task-Aware Unified Source Separation
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于音频源分离任务，旨在优化TUSS模型的性能与复杂度平衡。通过设计更高效的模型FasTUSS，显著减少运算量并保持较高分离效果。**

- **链接: [http://arxiv.org/pdf/2507.11435v1](http://arxiv.org/pdf/2507.11435v1)**

> **作者:** Francesco Paissan; Gordon Wichern; Yoshiki Masuyama; Ryo Aihara; François G. Germain; Kohei Saijo; Jonathan Le Roux
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** Time-Frequency (TF) dual-path models are currently among the best performing audio source separation network architectures, achieving state-of-the-art performance in speech enhancement, music source separation, and cinematic audio source separation. While they are characterized by a relatively low parameter count, they still require a considerable number of operations, implying a higher execution time. This problem is exacerbated by the trend towards bigger models trained on large amounts of data to solve more general tasks, such as the recently introduced task-aware unified source separation (TUSS) model. TUSS, which aims to solve audio source separation tasks using a single, conditional model, is built upon TF-Locoformer, a TF dual-path model combining convolution and attention layers. The task definition comes in the form of a sequence of prompts that specify the number and type of sources to be extracted. In this paper, we analyze the design choices of TUSS with the goal of optimizing its performance-complexity trade-off. We derive two more efficient models, FasTUSS-8.3G and FasTUSS-11.7G that reduce the original model's operations by 81\% and 73\% with minor performance drops of 1.2~dB and 0.4~dB averaged over all benchmarks, respectively. Additionally, we investigate the impact of prompt conditioning to derive a causal TUSS model.
>
---
#### [new 002] EditGen: Harnessing Cross-Attention Control for Instruction-Based Auto-Regressive Audio Editing
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频编辑任务，旨在通过交叉注意力控制实现基于指令的自回归音频编辑，解决音频生成中对旋律、动态和节奏的精准控制问题。**

- **链接: [http://arxiv.org/pdf/2507.11096v1](http://arxiv.org/pdf/2507.11096v1)**

> **作者:** Vassilis Sioros; Alexandros Potamianos; Giorgos Paraskevopoulos
>
> **摘要:** In this study, we investigate leveraging cross-attention control for efficient audio editing within auto-regressive models. Inspired by image editing methodologies, we develop a Prompt-to-Prompt-like approach that guides edits through cross and self-attention mechanisms. Integrating a diffusion-based strategy, influenced by Auffusion, we extend the model's functionality to support refinement edits, establishing a baseline for prompt-guided audio editing. Additionally, we introduce an alternative approach by incorporating MUSICGEN, a pre-trained frozen auto-regressive model, and propose three editing mechanisms, based on Replacement, Reweighting, and Refinement of the attention scores. We employ commonly-used music-specific evaluation metrics and a human study, to gauge time-varying controllability, adherence to global text cues, and overall audio realism. The automatic and human evaluations indicate that the proposed combination of prompt-to-prompt guidance with autoregressive generation models significantly outperforms the diffusion-based baseline in terms of melody, dynamics, and tempo of the generated audio. Our code is available at https://github.com/billsioros/EditGen
>
---
#### [new 003] Improving Neural Pitch Estimation with SWIPE Kernels
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在提升神经网络的音高估计性能。通过引入SWIPE核作为音频前端，提高准确性和鲁棒性，同时减少模型参数量。**

- **链接: [http://arxiv.org/pdf/2507.11233v1](http://arxiv.org/pdf/2507.11233v1)**

> **作者:** David Marttila; Joshua D. Reiss
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Neural networks have become the dominant technique for accurate pitch and periodicity estimation. Although a lot of research has gone into improving network architectures and training paradigms, most approaches operate directly on the raw audio waveform or on general-purpose time-frequency representations. We investigate the use of Sawtooth-Inspired Pitch Estimation (SWIPE) kernels as an audio frontend and find that these hand-crafted, task-specific features can make neural pitch estimators more accurate, robust to noise, and more parameter-efficient. We evaluate supervised and self-supervised state-of-the-art architectures on common datasets and show that the SWIPE audio frontend allows for reducing the network size by an order of magnitude without performance degradation. Additionally, we show that the SWIPE algorithm on its own is much more accurate than commonly reported, outperforming state-of-the-art self-supervised neural pitch estimators.
>
---
#### [new 004] Pronunciation Deviation Analysis Through Voice Cloning and Acoustic Comparison
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音识别任务，旨在检测发音错误。通过对比用户原声与修正后的克隆语音，识别发音偏差区域，无需预定义规则或大量训练数据。**

- **链接: [http://arxiv.org/pdf/2507.10985v1](http://arxiv.org/pdf/2507.10985v1)**

> **作者:** Andrew Valdivia; Yueming Zhang; Hailu Xu; Amir Ghasemkhani; Xin Qin
>
> **摘要:** This paper presents a novel approach for detecting mispronunciations by analyzing deviations between a user's original speech and their voice-cloned counterpart with corrected pronunciation. We hypothesize that regions with maximal acoustic deviation between the original and cloned utterances indicate potential mispronunciations. Our method leverages recent advances in voice cloning to generate a synthetic version of the user's voice with proper pronunciation, then performs frame-by-frame comparisons to identify problematic segments. Experimental results demonstrate the effectiveness of this approach in pinpointing specific pronunciation errors without requiring predefined phonetic rules or extensive training data for each target language.
>
---
#### [new 005] Supporting SENĆOTEN Language Documentation Efforts with Automatic Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决SENĆOTEN语言资源匮乏问题，通过ASR技术提升语言文档记录效率。工作包括数据增强与跨语言迁移学习。**

- **链接: [http://arxiv.org/pdf/2507.10827v1](http://arxiv.org/pdf/2507.10827v1)**

> **作者:** Mengzhe Geng; Patrick Littell; Aidan Pine; PENÁĆ; Marc Tessier; Roland Kuhn
>
> **备注:** Accepted by ComputEL-8
>
> **摘要:** The SEN\'{C}OTEN language, spoken on the Saanich peninsula of southern Vancouver Island, is in the midst of vigorous language revitalization efforts to turn the tide of language loss as a result of colonial language policies. To support these on-the-ground efforts, the community is turning to digital technology. Automatic Speech Recognition (ASR) technology holds great promise for accelerating language documentation and the creation of educational resources. However, developing ASR systems for SEN\'{C}OTEN is challenging due to limited data and significant vocabulary variation from its polysynthetic structure and stress-driven metathesis. To address these challenges, we propose an ASR-driven documentation pipeline that leverages augmented speech data from a text-to-speech (TTS) system and cross-lingual transfer learning with Speech Foundation Models (SFMs). An n-gram language model is also incorporated via shallow fusion or n-best restoring to maximize the use of available data. Experiments on the SEN\'{C}OTEN dataset show a word error rate (WER) of 19.34% and a character error rate (CER) of 5.09% on the test set with a 57.02% out-of-vocabulary (OOV) rate. After filtering minor cedilla-related errors, WER improves to 14.32% (26.48% on unseen words) and CER to 3.45%, demonstrating the potential of our ASR-driven pipeline to support SEN\'{C}OTEN language documentation.
>
---
#### [new 006] Grammatical Structure and Grammatical Variations in Non-Metric Iranian Classical Music
- **分类: cs.NE; cs.SD; eess.AS**

- **简介: 该论文属于音乐结构分析任务，旨在解析非节拍伊朗古典音乐的语法结构并生成变奏。通过构建数据集和算法，实现音乐结构解析与变奏生成。**

- **链接: [http://arxiv.org/pdf/2507.10708v1](http://arxiv.org/pdf/2507.10708v1)**

> **作者:** Maziar Kanani; Sean O Leary; James McDermott
>
> **摘要:** In this study we introduce a symbolic dataset composed of non-metric Iranian classical music, and algorithms for structural parsing of this music, and generation of variations. The corpus comprises MIDI files and data sheets of Dastgah Shour from Radif Mirza Abdollah, the foundational repertoire of Iranian classical music. Furthermore, we apply our previously-introduced algorithm for parsing melodic structure (Kanani et al., 2023b)to the dataset. Unlike much Western music, this type of non-metric music does not follow bar-centric organisation. The non-metric organisation can be captured well by our parsing algorithm. We parse each tune (Gusheh) into a grammar to identify motifs and phrases. These grammar representations can be useful for educational and ethnomusicological purposes. We also further develop a previously-introduced method of creating melodic variations (Kanani et al., 2023b). After parsing an existing tune to produce a grammar, by applying mutations to this grammar, we generate a new grammar. Expanding this new version yields a variation of the original tune. Variations are assessed by a domain-expert listener. Additionally, we conduct a statistical analysis of mutation with different representation setups for our parsing and generation algorithms. The overarching conclusion is that the system successfully produces acceptable variations post-mutation. While our case study focuses on Iranian classical music, the methodology can be adapted for Arabic or Turkish classical music.
>
---
#### [new 007] Physics-Informed Transfer Learning for Data-Driven Sound Source Reconstruction in Near-Field Acoustic Holography
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声源重构任务，旨在解决NAH中数据驱动模型迁移的问题。通过物理信息引导的迁移学习，提升不同声源间的重建精度。**

- **链接: [http://arxiv.org/pdf/2507.11070v1](http://arxiv.org/pdf/2507.11070v1)**

> **作者:** Xinmeng Luan; Mirco Pezzoli; Fabio Antonacci; Augusto Sarti
>
> **备注:** to appear in IEEE WASPAA 2025
>
> **摘要:** We propose a transfer learning framework for sound source reconstruction in Near-field Acoustic Holography (NAH), which adapts a well-trained data-driven model from one type of sound source to another using a physics-informed procedure. The framework comprises two stages: (1) supervised pre-training of a complex-valued convolutional neural network (CV-CNN) on a large dataset, and (2) purely physics-informed fine-tuning on a single data sample based on the Kirchhoff-Helmholtz integral. This method follows the principles of transfer learning by enabling generalization across different datasets through physics-informed adaptation. The effectiveness of the approach is validated by transferring a pre-trained model from a rectangular plate dataset to a violin top plate dataset, where it shows improved reconstruction accuracy compared to the pre-trained model and delivers performance comparable to that of Compressive-Equivalent Source Method (C-ESM). Furthermore, for successful modes, the fine-tuned model outperforms both the pre-trained model and C-ESM in accuracy.
>
---
#### [new 008] Parsing Musical Structure to Enable Meaningful Variations
- **分类: cs.AI; cs.NE; cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在通过解析音乐结构并变异语法来生成新曲调，解决如何在保持关联性的同时创造变化的问题。**

- **链接: [http://arxiv.org/pdf/2507.10740v1](http://arxiv.org/pdf/2507.10740v1)**

> **作者:** Maziar Kanani; Sean O Leary; James McDermott
>
> **摘要:** This paper presents a novel rule-based approach for generating music by varying existing tunes. We parse each tune to find the Pathway Assembly (PA) [ 1], that is a structure representing all repetitions in the tune. The Sequitur algorithm [2 ] is used for this. The result is a grammar. We then carry out mutation on the grammar, rather than on a tune directly. There are potentially 19 types of mutations such as adding, removing, swapping or reversing parts of the grammar that can be applied to the grammars. The system employs one of the mutations randomly in this step to automatically manipulate the grammar. Following the mutation, we need to expand the grammar which returns a new tune. The output after 1 or more mutations will be a new tune related to the original tune. Our study examines how tunes change gradually over the course of multiple mutations. Edit distances, structural complexity and length of the tunes are used to show how a tune is changed after multiple mutations. In addition, the size of effect of each mutation type is analyzed. As a final point, we review the musical aspect of the output tunes. It should be noted that the study only focused on generating new pitch sequences. The study is based on an Irish traditional tune dataset and a list of integers has been used to represent each tune's pitch values.
>
---
#### [new 009] Commuting Distance Regularization for Timescale-Dependent Label Inconsistency in EEG Emotion Recognition
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属于EEG情绪识别任务，解决时间尺度标签不一致问题，提出LVL和LGCL正则化方法提升模型性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.10895v1](http://arxiv.org/pdf/2507.10895v1)**

> **作者:** Xiaocong Zeng; Craig Michoski; Yan Pang; Dongyang Kuang
>
> **摘要:** In this work, we address the often-overlooked issue of Timescale Dependent Label Inconsistency (TsDLI) in training neural network models for EEG-based human emotion recognition. To mitigate TsDLI and enhance model generalization and explainability, we propose two novel regularization strategies: Local Variation Loss (LVL) and Local-Global Consistency Loss (LGCL). Both methods incorporate classical mathematical principles--specifically, functions of bounded variation and commute-time distances--within a graph theoretic framework. Complementing our regularizers, we introduce a suite of new evaluation metrics that better capture the alignment between temporally local predictions and their associated global emotion labels. We validate our approach through comprehensive experiments on two widely used EEG emotion datasets, DREAMER and DEAP, across a range of neural architectures including LSTM and transformer-based models. Performance is assessed using five distinct metrics encompassing both quantitative accuracy and qualitative consistency. Results consistently show that our proposed methods outperform state-of-the-art baselines, delivering superior aggregate performance and offering a principled trade-off between interpretability and predictive power under label inconsistency. Notably, LVL achieves the best aggregate rank across all benchmarked backbones and metrics, while LGCL frequently ranks the second, highlighting the effectiveness of our framework.
>
---
#### [new 010] Array-Aware Ambisonics and HRTF Encoding for Binaural Reproduction With Wearable Arrays
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于空间音频渲染任务，解决传统Ambisonics在可穿戴阵列中的空间准确性问题，通过结合HRTF预处理提升音质与空间感。**

- **链接: [http://arxiv.org/pdf/2507.11091v1](http://arxiv.org/pdf/2507.11091v1)**

> **作者:** Yhonatan Gayer; Vladimir Tourbabin; Zamir Ben Hur; David Lou Alon; Boaz Rafaely
>
> **摘要:** This work introduces a novel method for binaural reproduction from arbitrary microphone arrays, based on array-aware optimization of Ambisonics encoding through Head-Related Transfer Function (HRTF) pre-processing. The proposed approach integrates array-specific information into the HRTF processing pipeline, leading to improved spatial accuracy in binaural rendering. Objective evaluations demonstrate superior performance under simulated wearable-array and head rotations compared to conventional Ambisonics encoding method. A listening experiment further confirms that the method achieves significantly higher perceptual ratings in both timbre and spatial quality. Fully compatible with standard Ambisonics, the proposed method offers a practical solution for spatial audio rendering in applications such as virtual reality, augmented reality, and wearable audio capture.
>
---
#### [new 011] Standardized Evaluation of Fetal Phonocardiography Processing Methods
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于胎儿心音处理任务，旨在评估不同心音检测与心率估计方法的性能，通过标准化测试比较其效果。**

- **链接: [http://arxiv.org/pdf/2507.10783v1](http://arxiv.org/pdf/2507.10783v1)**

> **作者:** Kristóf Müller; Janka Hatvani; Márton Áron Goda; Miklós Koller
>
> **备注:** 17 pages, 7 figures, 7 tables
>
> **摘要:** Motivation. Phonocardiography can give access to the fetal heart rate as well as direct heart sound data, and is entirely passive, using no radiation of any kind. Approach. We discuss the currently available methods for fetal heart sound detection and heart rate estimation and compare them using a common benchmarking platform and a pre-selected testing dataset. Compared to previous reviews, we evaluated the discussed methods in a standardized manner for a fair comparison. Our tests included tolerance-based detection accuracy, error rates for label insertions, deletions, and substitutions, and statistical measures for heart rate mean square error. Results. Based on our results, there is no definite best method that can achieve the highest scores in all of the tests, and simpler methods could perform comparably to more complex ones. The best model for first heart sound detection achieved 97.6% F1-score, 97.4% positive predictive value, and 12.2+-8.0 ms mean absolute error. In terms of second heart sound detection the best model had 91.4% F1-score, 91.3% positive predictive value, and 17.3+-12.2 ms mean absolute error. For fetal heart rate a 0.644 mean square error was achieved by the best method. Significance. Our main conclusion is that further standardization is required in fetal heart rate and heart sound detection method evaluation. The tests and algorithm implementations are openly available at: https://github.com/mulkr/standard-fpcg-evaluation.
>
---
## 更新

#### [replaced 001] A Survey on Speech Deepfake Detection
- **分类: cs.SD; cs.CR; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2404.13914v2](http://arxiv.org/pdf/2404.13914v2)**

> **作者:** Menglu Li; Yasaman Ahmadiadli; Xiao-Ping Zhang
>
> **备注:** 38 pages. This paper has been accepted by ACM Computing Surveys
>
> **摘要:** The availability of smart devices leads to an exponential increase in multimedia content. However, advancements in deep learning have also enabled the creation of highly sophisticated Deepfake content, including speech Deepfakes, which pose a serious threat by generating realistic voices and spreading misinformation. To combat this, numerous challenges have been organized to advance speech Deepfake detection techniques. In this survey, we systematically analyze more than 200 papers published up to March 2024. We provide a comprehensive review of each component in the detection pipeline, including model architectures, optimization techniques, generalizability, evaluation metrics, performance comparisons, available datasets, and open source availability. For each aspect, we assess recent progress and discuss ongoing challenges. In addition, we explore emerging topics such as partial Deepfake detection, cross-dataset evaluation, and defences against adversarial attacks, while suggesting promising research directions. This survey not only identifies the current state of the art to establish strong baselines for future experiments but also offers clear guidance for researchers aiming to enhance speech Deepfake detection systems.
>
---
#### [replaced 002] ReverbMiipher: Generative Speech Restoration meets Reverberation Characteristics Controllability
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.05077v2](http://arxiv.org/pdf/2505.05077v2)**

> **作者:** Wataru Nakata; Yuma Koizumi; Shigeki Karita; Robin Scheibler; Haruko Ishikawa; Adriana Guevara-Rukoz; Heiga Zen; Michiel Bacchiani
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Reverberation encodes spatial information regarding the acoustic source environment, yet traditional Speech Restoration (SR) usually completely removes reverberation. We propose ReverbMiipher, an SR model extending parametric resynthesis framework, designed to denoise speech while preserving and enabling control over reverberation. ReverbMiipher incorporates a dedicated ReverbEncoder to extract a reverb feature vector from noisy input. This feature conditions a vocoder to reconstruct the speech signal, removing noise while retaining the original reverberation characteristics. A stochastic zero-vector replacement strategy during training ensures the feature specifically encodes reverberation, disentangling it from other speech attributes. This learned representation facilitates reverberation control via techniques such as interpolation between features, replacement with features from other utterances, or sampling from a latent space. Objective and subjective evaluations confirm ReverbMiipher effectively preserves reverberation, removes other artifacts, and outperforms the conventional two-stage SR and convolving simulated room impulse response approach. We further demonstrate its ability to generate novel reverberation effects through feature manipulation.
>
---
#### [replaced 003] BlueME: Robust Underwater Robot-to-Robot Communication Using Compact Magnetoelectric Antennas
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2411.09241v3](http://arxiv.org/pdf/2411.09241v3)**

> **作者:** Mehron Talebi; Sultan Mahmud; Adam Khalifa; Md Jahidul Islam
>
> **摘要:** We present the design, development, and experimental validation of BlueME, a compact magnetoelectric (ME) antenna array system for underwater robot-to-robot communication. BlueME employs ME antennas operating at their natural mechanical resonance frequency to efficiently transmit and receive very-low-frequency (VLF) electromagnetic signals underwater. We outline the design, simulation, fabrication, and integration of the proposed system on low-power embedded platforms focusing on portable and scalable applications. For performance evaluation, we deployed BlueME on an autonomous surface vehicle (ASV) and a remotely operated vehicle (ROV) in open-water field trials. Our tests demonstrate that BlueME maintains reliable signal transmission at distances beyond 200 meters while consuming only 1 watt of power. Field trials show that the system operates effectively in challenging underwater conditions such as turbidity, obstacles, and multipath interference -- that generally affect acoustics and optics. Our analysis also examines the impact of complete submersion on system performance and identifies key deployment considerations. This work represents the first practical underwater deployment of ME antennas outside the laboratory, and implements the largest VLF ME array system to date. BlueME demonstrates significant potential for marine robotics and automation in multi-robot cooperative systems and remote sensor networks.
>
---
#### [replaced 004] Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09116v2](http://arxiv.org/pdf/2507.09116v2)**

> **作者:** Bingshen Mu; Kun Wei; Pengcheng Guo; Lei Xie
>
> **备注:** IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
>
---
