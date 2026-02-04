# 音频 cs.SD;  eess.AS

- **最新发布 18 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Mići Princ -- A Little Boy Teaching Speech Technologies the Chakavian Dialect
- **分类: eess.AS; cs.CL**

- **简介: 论文发布《小王子》的查卡维亚方言版文本与音频数据集，用于语音技术研究。旨在保护方言内容并提升语音识别模型对地方话的处理能力。**

- **链接: [https://arxiv.org/pdf/2602.03245v1](https://arxiv.org/pdf/2602.03245v1)**

> **作者:** Nikola Ljubešić; Peter Rupnik; Tea Perinčić
>
> **备注:** 2 figures, 14 pages, accepted and presented at JTDH 2024
>
> **摘要:** This paper documents our efforts in releasing the printed and audio book of the translation of the famous novel The Little Prince into the Chakavian dialect, as a computer-readable, AI-ready dataset, with the textual and the audio components of the two releases now aligned on the level of each written and spoken word. Our motivation for working on this release is multiple. The first one is our wish to preserve the highly valuable and specific content beyond the small editions of the printed and the audio book. With the dataset published in the CLARIN.SI repository, this content is from now on at the fingertips of any interested individual. The second motivation is to make the data available for various artificial-intelligence-related usage scenarios, such as the one we follow upon inside this paper already -- adapting the Whisper-large-v3 open automatic speech recognition model, with decent performance on standard Croatian, to Chakavian dialectal speech. We can happily report that with adapting the model, the word error rate on the selected test data has being reduced to a half, while we managed to remove up to two thirds of the error on character level. We envision many more usages of this dataset beyond the set of experiments we have already performed, both on tasks of artificial intelligence research and application, as well as dialectal research. The third motivation for this release is our hope that this, now highly structured dataset, will be transformed into a digital online edition of this work, allowing individuals beyond the research and technology communities to enjoy the beauty of the message of the little boy in the desert, told through the spectacular prism of the Chakavian dialect.
>
---
#### [new 002] GRAM: Spatial general-purpose audio representations for real-world environments
- **分类: cs.SD**

- **简介: 该论文提出GRAM模型，解决真实环境中音频表示学习问题，通过多通道自编码器提升空间音频建模能力。**

- **链接: [https://arxiv.org/pdf/2602.03307v1](https://arxiv.org/pdf/2602.03307v1)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Revise with RealSELD
>
> **摘要:** Audio foundation models learn general-purpose audio representations that facilitate a wide range of downstream tasks. While the performance of these models has greatly increased for conventional single-channel, dry audio clips, their success in real-world acoustic environments with reverberation and noise is limited. Furthermore, most audio foundation models ignore the spatial dimension of real-world acoustic environments, ruling out tasks involving sound localization. To address these limitations, we propose GRAM: a general-purpose real-world audio model that employs a multi-channel masked autoencoder to efficiently learn spatial audio representations. We evaluated GRAM and other audio foundation models in a standardized manner on high-quality simulations of naturalistic, spatial acoustic environments as well as recordings of real-world environments and release these two complementary benchmark task suites: NatHEAR and RealSELD. Our results demonstrate that GRAM outperforms all state-of-the-art self-supervised audio foundation models on NatHEAR and the clean, single-channel version HEAR, while using only a fraction of the training data. GRAM also shows state-of-the-art localization performance in simulated environments and generalizes efficiently to real-world recordings in RealSELD. Taken together, GRAM presents a significant advance toward robust spatial audio foundation models for real-world environments.
>
---
#### [new 003] Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对生物声学分类任务，解决多源证据融合问题。提出FINCH框架，结合音频与时空信息，自适应加权提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.03817v1](https://arxiv.org/pdf/2602.03817v1)**

> **作者:** Oscar Ovanger; Levi Harris; Timothy H. Keitt
>
> **摘要:** Many machine learning systems have access to multiple sources of evidence for the same prediction target, yet these sources often differ in reliability and informativeness across inputs. In bioacoustic classification, species identity may be inferred both from the acoustic signal and from spatiotemporal context such as location and season; while Bayesian inference motivates multiplicative evidence combination, in practice we typically only have access to discriminative predictors rather than calibrated generative models. We introduce \textbf{F}usion under \textbf{IN}dependent \textbf{C}onditional \textbf{H}ypotheses (\textbf{FINCH}), an adaptive log-linear evidence fusion framework that integrates a pre-trained audio classifier with a structured spatiotemporal predictor. FINCH learns a per-sample gating function that estimates the reliability of contextual information from uncertainty and informativeness statistics. The resulting fusion family \emph{contains} the audio-only classifier as a special case and explicitly bounds the influence of contextual evidence, yielding a risk-contained hypothesis class with an interpretable audio-only fallback. Across benchmarks, FINCH consistently outperforms fixed-weight fusion and audio-only baselines, improving robustness and error trade-offs even when contextual information is weak in isolation. We achieve state-of-the-art performance on CBI and competitive or improved performance on several subsets of BirdSet using a lightweight, interpretable, evidence-based approach. Code is available: \texttt{\href{https://anonymous.4open.science/r/birdnoise-85CD/README.md}{anonymous-repository}}
>
---
#### [new 004] CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于情感语音合成任务，解决多情绪表达与文本情感不一致的问题。通过激活控制方法，实现情感的可组合与可控合成。**

- **链接: [https://arxiv.org/pdf/2602.03420v1](https://arxiv.org/pdf/2602.03420v1)**

> **作者:** Siyi Wang; Shihong Tan; Siyi Liu; Hong Jia; Gongping Huang; James Bailey; Ting Dang
>
> **摘要:** Emotional expression in human speech is nuanced and compositional, often involving multiple, sometimes conflicting, affective cues that may diverge from linguistic content. In contrast, most expressive text-to-speech systems enforce a single utterance-level emotion, collapsing affective diversity and suppressing mixed or text-emotion-misaligned expression. While activation steering via latent direction vectors offers a promising solution, it remains unclear whether emotion representations are linearly steerable in TTS, where steering should be applied within hybrid TTS architectures, and how such complex emotion behaviors should be evaluated. This paper presents the first systematic analysis of activation steering for emotional control in hybrid TTS models, introducing a quantitative, controllable steering framework, and multi-rater evaluation protocols that enable composable mixed-emotion synthesis and reliable text-emotion mismatch synthesis. Our results demonstrate, for the first time, that emotional prosody and expressive variability are primarily synthesized by the TTS language module instead of the flow-matching module, and also provide a lightweight steering approach for generating natural, human-like emotional speech.
>
---
#### [new 005] WST-X Series: Wavelet Scattering Transform for Interpretable Speech Deepfake Detection
- **分类: eess.AS; cs.CL; eess.SP**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决传统特征与自监督特征在可解释性和性能上的不足。提出WST-X系列特征提取器，结合小波散射变换，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.02980v1](https://arxiv.org/pdf/2602.02980v1)**

> **作者:** Xi Xuan; Davide Carbone; Ruchi Pandey; Wenxin Zhang; Tomi H. Kinnunen
>
> **备注:** Submitted to IEEE Signal Processing Letters
>
> **摘要:** Designing front-ends for speech deepfake detectors primarily focuses on two categories. Hand-crafted filterbank features are transparent but are limited in capturing high-level semantic details, often resulting in performance gaps compared to self-supervised (SSL) features. SSL features, in turn, lack interpretability and may overlook fine-grained spectral anomalies. We propose the WST-X series, a novel family of feature extractors that combines the best of both worlds via the wavelet scattering transform (WST), integrating wavelets with nonlinearities analogous to deep convolutional networks. We investigate 1D and 2D WSTs to extract acoustic details and higher-order structural anomalies, respectively. Experimental results on the recent and challenging Deepfake-Eval-2024 dataset indicate that WST-X outperforms existing front-ends by a wide margin. Our analysis reveals that a small averaging scale ($J$), combined with high-frequency and directional resolutions ($Q, L$), is critical for capturing subtle artifacts. This underscores the value of translation-invariant and deformation-stable features for robust and interpretable speech deepfake detection.
>
---
#### [new 006] Rethinking Music Captioning with Music Metadata LLMs
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐描述生成任务，解决高质量标注数据稀缺问题。通过基于元数据的captioning方法，提升训练效率与风格灵活性。**

- **链接: [https://arxiv.org/pdf/2602.03023v1](https://arxiv.org/pdf/2602.03023v1)**

> **作者:** Irmak Bukey; Zhepei Wang; Chris Donahue; Nicholas J. Bryan
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Music captioning, or the task of generating a natural language description of music, is useful for both music understanding and controllable music generation. Training captioning models, however, typically requires high-quality music caption data which is scarce compared to metadata (e.g., genre, mood, etc.). As a result, it is common to use large language models (LLMs) to synthesize captions from metadata to generate training data for captioning models, though this process imposes a fixed stylization and entangles factual information with natural language style. As a more direct approach, we propose metadata-based captioning. We train a metadata prediction model to infer detailed music metadata from audio and then convert it into expressive captions via pre-trained LLMs at inference time. Compared to a strong end-to-end baseline trained on LLM-generated captions derived from metadata, our method: (1) achieves comparable performance in less training time over end-to-end captioners, (2) offers flexibility to easily change stylization post-training, enabling output captions to be tailored to specific stylistic and quality requirements, and (3) can be prompted with audio and partial metadata to enable powerful metadata imputation or in-filling--a common task for organizing music data.
>
---
#### [new 007] When Noise Lowers The Loss: Rethinking Likelihood-Based Evaluation in Music Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成模型评估任务，旨在解决现有损失函数无法准确反映生成质量的问题。通过噪声注入实验，发现损失曲线形状可作为质量评估新指标。**

- **链接: [https://arxiv.org/pdf/2602.02738v1](https://arxiv.org/pdf/2602.02738v1)**

> **作者:** Xiaosha Li; Chun Liu; Ziyu Wang
>
> **备注:** Accepted by IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** The rise of music large language models (LLMs) demands robust methods of evaluating output quality, especially in distinguishing high-quality compositions from "garbage music". Curiously, we observe that the standard cross-entropy loss -- a core training metric -- often decrease when models encounter systematically corrupted music, undermining its validity as a standalone quality indicator. To investigate this paradox, we introduce noise injection experiment, where controlled noise signal of varying lengths are injected into musical contexts. We hypothesize that a model's loss reacting positively to these perturbations, specifically a sharp increase ("Peak" area) for short injection, can serve as a proxy for its ability to discern musical integrity. Experiments with MusicGen models in the audio waveform domain confirm that Music LLMs respond more strongly to local, texture-level disruptions than to global semantic corruption. Beyond exposing this bias, our results highlight a new principle: the shape of the loss curve -- rather than its absolute value -- encodes critical information about the quality of the generated content (i.e., model behavior). We envision this profile-based evaluation as a label-free, model-intrinsic framework for assessing musical quality -- opening the door to more principled training objectives and sharper benchmarks.
>
---
#### [new 008] VividVoice: A Unified Framework for Scene-Aware Visually-Driven Speech Synthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出VividVoice，解决场景感知的视觉驱动语音合成任务，旨在提升音频与视觉的一致性。通过构建多模态数据集和设计对齐模块，增强语音真实感与多模态一致性。**

- **链接: [https://arxiv.org/pdf/2602.02591v1](https://arxiv.org/pdf/2602.02591v1)**

> **作者:** Chengyuan Ma; Jiawei Jin; Ruijie Xiong; Chunxiang Jin; Canxiang Yan; Wenming Yang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** We introduce and define a novel task-Scene-Aware Visually-Driven Speech Synthesis, aimed at addressing the limitations of existing speech generation models in creating immersive auditory experiences that align with the real physical world. To tackle the two core challenges of data scarcity and modality decoupling, we propose VividVoice, a unified generative framework. First, we constructed a large-scale, high-quality hybrid multimodal dataset, Vivid-210K, which, through an innovative programmatic pipeline, establishes a strong correlation between visual scenes, speaker identity, and audio for the first time. Second, we designed a core alignment module, D-MSVA, which leverages a decoupled memory bank architecture and a cross-modal hybrid supervision strategy to achieve fine-grained alignment from visual scenes to timbre and environmental acoustic features. Both subjective and objective experimental results provide strong evidence that VividVoice significantly outperforms existing baseline models in terms of audio fidelity, content clarity, and multimodal consistency. Our demo is available at https://chengyuann.github.io/VividVoice/.
>
---
#### [new 009] EarResp-ANS : Audio-Based On-Device Respiration Rate Estimation on Earphones with Adaptive Noise Suppression
- **分类: cs.SD; cs.HC**

- **简介: 该论文属于呼吸率估计任务，解决耳麦上实时、低功耗呼吸率监测问题。通过自适应降噪技术实现精准检测，无需神经网络，适用于可穿戴设备。**

- **链接: [https://arxiv.org/pdf/2602.03549v1](https://arxiv.org/pdf/2602.03549v1)**

> **作者:** Michael Küttner; Valeria Zitz; Supraja Ramesh; Michael Beigl; Tobias Röddiger
>
> **备注:** 31 pages, 11 figures
>
> **摘要:** Respiratory rate (RR) is a key vital sign for clinical assessment and mental well-being, yet it is rarely monitored in everyday life due to the lack of unobtrusive sensing technologies. In-ear audio sensing is promising due to its high social acceptance and the amplification of physiological sounds caused by the occlusion effect; however, existing approaches often fail under real-world noise or rely on computationally expensive models. We present EarResp-ANS, the first system enabling fully on-device, real-time RR estimation on commercial earphones. The system employs LMS-based adaptive noise suppression (ANS) to attenuate ambient noise while preserving respiration-related acoustic components, without requiring neural networks or audio streaming, thereby explicitly addressing the energy and privacy constraints of wearable devices. We evaluate EarResp-ANS in a study with 18 participants under realistic acoustic conditions, including music, cafeteria noise, and white noise up to 80 dB SPL. EarResp-ANS achieves robust performance with a global MAE of 0.84 CPM , reduced to 0.47 CPM via automatic outlier rejection, while operating with less than 2% processor load directly on the earphone.
>
---
#### [new 010] D3PIA: A Discrete Denoising Diffusion Model for Piano Accompaniment Generation From Lead sheet
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文属于钢琴伴奏生成任务，旨在从旋律和和弦信息中生成完整钢琴音乐。提出D3PIA模型，通过局部对齐和邻域注意力提升伴奏的音乐连贯性与和弦准确性。**

- **链接: [https://arxiv.org/pdf/2602.03523v1](https://arxiv.org/pdf/2602.03523v1)**

> **作者:** Eunjin Choi; Hounsu Kim; Hayeon Bang; Taegyun Kwon; Juhan Nam
>
> **备注:** Accepted at 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
>
> **摘要:** Generating piano accompaniments in the symbolic music domain is a challenging task that requires producing a complete piece of piano music from given melody and chord constraints, such as those provided by a lead sheet. In this paper, we propose a discrete diffusion-based piano accompaniment generation model, D3PIA, leveraging local alignment between lead sheet and accompaniment in piano-roll representation. D3PIA incorporates Neighborhood Attention (NA) to both encode the lead sheet and condition it for predicting note states in the piano accompaniment. This design enhances local contextual modeling by efficiently attending to nearby melody and chord conditions. We evaluate our model using the POP909 dataset, a widely used benchmark for piano accompaniment generation. Objective evaluation results demonstrate that D3PIA preserves chord conditions more faithfully compared to continuous diffusion-based and Transformer-based baselines. Furthermore, a subjective listening test indicates that D3PIA generates more musically coherent accompaniments than the comparison models.
>
---
#### [new 011] PACE: Pretrained Audio Continual Learning
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频持续学习任务，解决预训练模型在数据分布变化下的性能下降问题。通过分析挑战并提出PACE方法提升模型适应能力。**

- **链接: [https://arxiv.org/pdf/2602.03355v1](https://arxiv.org/pdf/2602.03355v1)**

> **作者:** Chang Li; Kanglei Zhou; Liyuan Wang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Audio is a fundamental modality for analyzing speech, music, and environmental sounds. Although pretrained audio models have significantly advanced audio understanding, they remain fragile in real-world settings where data distributions shift over time. In this work, we present the first systematic benchmark for audio continual learning (CL) with pretrained models (PTMs), together with a comprehensive analysis of its unique challenges. Unlike in vision, where parameter-efficient fine-tuning (PEFT) has proven effective for CL, directly transferring such strategies to audio leads to poor performance. This stems from a fundamental property of audio backbones: they focus on low-level spectral details rather than structured semantics, causing severe upstream-downstream misalignment. Through extensive empirical study, we identify analytic classifiers with first-session adaptation (FSA) as a promising direction, but also reveal two major limitations: representation saturation in coarse-grained scenarios and representation drift in fine-grained scenarios. To address these challenges, we propose PACE, a novel method that enhances FSA via a regularized analytic classifier and enables multi-session adaptation through adaptive subspace-orthogonal PEFT for improved semantic alignment. In addition, we introduce spectrogram-based boundary-aware perturbations to mitigate representation overlap and improve stability. Experiments on six diverse audio CL benchmarks demonstrate that PACE substantially outperforms state-of-the-art baselines, marking an important step toward robust and scalable audio continual learning with PTMs.
>
---
#### [new 012] A Unified SVD-Modal Solution for Sparse Sound Field Reconstruction with Hybrid Spherical-Linear Microphone Arrays
- **分类: eess.AS**

- **简介: 该论文属于声场重建任务，解决混合球-线阵列的稀疏恢复问题。通过SVD提取正交模式，提升空间选择性，实验表明优于单一球阵列和直接拼接。**

- **链接: [https://arxiv.org/pdf/2602.03398v1](https://arxiv.org/pdf/2602.03398v1)**

> **作者:** Shunxi Xu; Thushara Abhayapala; Craig T. Jin
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** We propose a data-driven sparse recovery framework for hybrid spherical linear microphone arrays using singular value decomposition (SVD) of the transfer operator. The SVD yields orthogonal microphone and field modes, reducing to spherical harmonics (SH) in the SMA-only case, while incorporating LMAs introduces complementary modes beyond SH. Modal analysis reveals consistent divergence from SH across frequency, confirming the improved spatial selectivity. Experiments in reverberant conditions show reduced energy-map mismatch and angular error across frequency, distance, and source count, outperforming SMA-only and direct concatenation. The results demonstrate that SVD-modal processing provides a principled and unified treatment of hybrid arrays for robust sparse sound-field reconstruction.
>
---
#### [new 013] Conditional Flow Matching for Visually-Guided Acoustic Highlighting
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于视觉引导的音频增强任务，解决音视频焦点不一致问题。提出条件流匹配框架，通过生成模型实现更准确的音频重混。**

- **链接: [https://arxiv.org/pdf/2602.03762v1](https://arxiv.org/pdf/2602.03762v1)**

> **作者:** Hugo Malard; Gael Le Lan; Daniel Wong; David Lou Alon; Yi-Chiao Wu; Sanjeel Parekh
>
> **摘要:** Visually-guided acoustic highlighting seeks to rebalance audio in alignment with the accompanying video, creating a coherent audio-visual experience. While visual saliency and enhancement have been widely studied, acoustic highlighting remains underexplored, often leading to misalignment between visual and auditory focus. Existing approaches use discriminative models, which struggle with the inherent ambiguity in audio remixing, where no natural one-to-one mapping exists between poorly-balanced and well-balanced audio mixes. To address this limitation, we reframe this task as a generative problem and introduce a Conditional Flow Matching (CFM) framework. A key challenge in iterative flow-based generation is that early prediction errors -- in selecting the correct source to enhance -- compound over steps and push trajectories off-manifold. To address this, we introduce a rollout loss that penalizes drift at the final step, encouraging self-correcting trajectories and stabilizing long-range flow integration. We further propose a conditioning module that fuses audio and visual cues before vector field regression, enabling explicit cross-modal source selection. Extensive quantitative and qualitative evaluations show that our method consistently surpasses the previous state-of-the-art discriminative approach, establishing that visually-guided audio remixing is best addressed through generative modeling.
>
---
#### [new 014] WAXAL: A Large-Scale Multilingual African Language Speech Corpus
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文介绍WAXAL，一个针对21种非洲语言的大规模语音数据集，旨在解决低资源语言在语音技术中的数字鸿沟问题。任务为语音识别与合成，工作包括数据收集、标注及质量控制。**

- **链接: [https://arxiv.org/pdf/2602.02734v1](https://arxiv.org/pdf/2602.02734v1)**

> **作者:** Abdoulaye Diack; Perry Nelson; Kwaku Agbesi; Angela Nakalembe; MohamedElfatih MohamedKhair; Vusumuzi Dube; Tavonga Siyavora; Subhashini Venugopalan; Jason Hickey; Uche Okonkwo; Abhishek Bapna; Isaac Wiafe; Raynard Dodzi Helegah; Elikem Doe Atsakpo; Charles Nutrokpor; Fiifi Baffoe Payin Winful; Kafui Kwashie Solaga; Jamal-Deen Abdulai; Akon Obu Ekpezu; Audace Niyonkuru; Samuel Rutunda; Boris Ishimwe; Michael Melese; Engineer Bainomugisha; Joyce Nakatumba-Nabende; Andrew Katumba; Claire Babirye; Jonathan Mukiibi; Vincent Kimani; Samuel Kibacia; James Maina; Fridah Emmah; Ahmed Ibrahim Shekarau; Ibrahim Shehu Adamu; Yusuf Abdullahi; Howard Lakougna; Bob MacDonald; Hadar Shemtov; Aisha Walcott-Bryant; Moustapha Cisse; Avinatan Hassidim; Jeff Dean; Yossi Matias
>
> **备注:** Initial dataset release
>
> **摘要:** The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 21 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with over 180 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at https://huggingface.co/datasets/google/WaxalNLP under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages.
>
---
#### [new 015] Synthetic Data Augmentation for Medical Audio Classification: A Preliminary Evaluation
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于医疗音频分类任务，旨在解决数据不足和类别不平衡问题。通过合成数据增强方法提升分类性能，但实验结果表明效果有限。**

- **链接: [https://arxiv.org/pdf/2602.02955v1](https://arxiv.org/pdf/2602.02955v1)**

> **作者:** David McShannon; Anthony Mella; Nicholas Dietrich
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Medical audio classification remains challenging due to low signal-to-noise ratios, subtle discriminative features, and substantial intra-class variability, often compounded by class imbalance and limited training data. Synthetic data augmentation has been proposed as a potential strategy to mitigate these constraints; however, prior studies report inconsistent methodological approaches and mixed empirical results. In this preliminary study, we explore the impact of synthetic augmentation on respiratory sound classification using a baseline deep convolutional neural network trained on a moderately imbalanced dataset (73%:27%). Three generative augmentation strategies (variational autoencoders, generative adversarial networks, and diffusion models) were assessed under controlled experimental conditions. The baseline model without augmentation achieved an F1-score of 0.645. Across individual augmentation strategies, performance gains were not observed, with several configurations demonstrating neutral or degraded classification performance. Only an ensemble of augmented models yielded a modest improvement in F1-score (0.664). These findings suggest that, for medical audio classification, synthetic augmentation may not consistently enhance performance when applied to a standard CNN classifier. Future work should focus on delineating task-specific data characteristics, model-augmentation compatibility, and evaluation frameworks necessary for synthetic augmentation to be effective in medical audio applications.
>
---
#### [new 016] Automated Dysphagia Screening Using Noninvasive Neck Acoustic Sensing
- **分类: cs.LG; cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于医学诊断任务，旨在解决 dysphagia 早期检测问题。通过非侵入性颈部声学传感和机器学习，实现吞咽异常的自动筛查。**

- **链接: [https://arxiv.org/pdf/2602.02725v1](https://arxiv.org/pdf/2602.02725v1)**

> **作者:** Jade Chng; Rong Xing; Yunfei Luo; Kristen Linnemeyer-Risser; Tauhidur Rahman; Andrew Yousef; Philip A Weissbrod
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** Pharyngeal health plays a vital role in essential human functions such as breathing, swallowing, and vocalization. Early detection of swallowing abnormalities, also known as dysphagia, is crucial for timely intervention. However, current diagnostic methods often rely on radiographic imaging or invasive procedures. In this study, we propose an automated framework for detecting dysphagia using portable and noninvasive acoustic sensing coupled with applied machine learning. By capturing subtle acoustic signals from the neck during swallowing tasks, we aim to identify patterns associated with abnormal physiological conditions. Our approach achieves promising test-time abnormality detection performance, with an AUC-ROC of 0.904 under 5 independent train-test splits. This work demonstrates the feasibility of using noninvasive acoustic sensing as a practical and scalable tool for pharyngeal health monitoring.
>
---
#### [new 017] A Multi-decoder Neural Tracking Method for Accurately Predicting Speech Intelligibility
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于语音可懂度预测任务，旨在提升EEG方法的准确性。通过多解码器融合神经追踪特征，建立回归模型预测SRT，实现客观评估。**

- **链接: [https://arxiv.org/pdf/2602.03624v1](https://arxiv.org/pdf/2602.03624v1)**

> **作者:** Rien Sonck; Bernd Accou; Tom Francart; Jonas Vanthornhout
>
> **摘要:** Objective: EEG-based methods can predict speech intelligibility, but their accuracy and robustness lag behind behavioral tests, which typically show test-retest differences under 1 dB. We introduce the multi-decoder method to predict speech reception thresholds (SRTs) from EEG recordings, enabling objective assessment for populations unable to perform behavioral tests; such as those with disorders of consciousness or during hearing aid fitting. Approach: The method aggregates data from hundreds of decoders, each trained on different speech features and EEG preprocessing setups to quantify neural tracking (NT) of speech signals. Using data from 39 participants (ages 18-24), we recorded 29 minutes of EEG per person while they listened to speech at six signal-to-noise ratios and a quiet story. NT values were combined into a high-dimensional feature vector per subject, and a support vector regression model was trained to predict SRTs from these vectors. Main Result: Predictions correlated significantly with behavioral SRTs (r = 0.647, p < 0.001; NRMSE = 0.19), with all differences under 1 dB. SHAP analysis showed theta/delta bands and early lags had slightly greater influence. Using pretrained subject-independent decoders reduced required EEG data collection to 15 minutes (3 minutes of story, 12 minutes across six SNR conditions) without losing accuracy.
>
---
#### [new 018] The Alignment Curse: Cross-Modality Jailbreak Transfer in Omni-Models
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文研究跨模态攻击转移问题，探讨文本到音频的越狱攻击迁移。属于安全红队任务，旨在评估和提升音频模态的安全性。**

- **链接: [https://arxiv.org/pdf/2602.02557v1](https://arxiv.org/pdf/2602.02557v1)**

> **作者:** Yupeng Chen; Junchi Yu; Aoxi Liu; Philip Torr; Adel Bibi
>
> **摘要:** Recent advances in end-to-end trained omni-models have significantly improved multimodal understanding. At the same time, safety red-teaming has expanded beyond text to encompass audio-based jailbreak attacks. However, an important bridge between textual and audio jailbreaks remains underexplored. In this work, we study the cross-modality transfer of jailbreak attacks from text to audio, motivated by the semantic similarity between the two modalities and the maturity of textual jailbreak methods. We first analyze the connection between modality alignment and cross-modality jailbreak transfer, showing that strong alignment can inadvertently propagate textual vulnerabilities to the audio modality, which we term the alignment curse. Guided by this analysis, we conduct an empirical evaluation of textual jailbreaks, text-transferred audio jailbreaks, and existing audio-based jailbreaks on recent omni-models. Our results show that text-transferred audio jailbreaks perform comparably to, and often better than, audio-based jailbreaks, establishing them as simple yet powerful baselines for future audio red-teaming. We further demonstrate strong cross-model transferability and show that text-transferred audio attacks remain effective even under a stricter audio-only access threat model.
>
---
## 更新

#### [replaced 001] SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations
- **分类: eess.AS**

- **简介: 该论文提出SPEAR框架，解决语音与音频表示学习的领域差异问题。通过融合语音和通用音频知识，提升模型在复杂声场中的表现。**

- **链接: [https://arxiv.org/pdf/2510.25955v2](https://arxiv.org/pdf/2510.25955v2)**

> **作者:** Xiaoyu Yang; Yifan Yang; Zengrui Jin; Ziyun Cui; Wen Wu; Baoxiang Li; Chao Zhang; Phil Woodland
>
> **备注:** Preprint. Under review
>
> **摘要:** Self-supervised learning (SSL) has significantly advanced acoustic representation learning. However, most existing models are optimised for either speech or audio event understanding, resulting in a persistent gap between these two domains. We address this gap with SPEAR (SPEech and Audio Representations), a self-supervised framework that distils complementary knowledge from a speech-focused SSL teacher and a general-audio SSL teacher into a single unified model. SPEAR applies multi-codebook vector quantisation to continuous teacher representations to produce fine-grained discrete tokens that capture both semantic and acoustic information. To effectively integrate these heterogeneous representations, SPEAR jointly predicts them given a masked input with an asymmetric pre-training loss. We further improve robustness in complex sound scenes through a novel token mixing mechanism. Extensive experiments demonstrate that SPEAR consistently outperforms existing unified speech and audio models. SPEAR establishes a new state-of-the-art on the SUPERB benchmark, surpassing WavLM Large on 12 of 15 tasks, while achieving competitive performance on the HEAR benchmark. These results position SPEAR as a versatile foundation for general-purpose speech and audio representation learning. The code and pre-trained models will be released.
>
---
#### [replaced 002] Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频大模型与脑科学交叉任务，旨在探究音频LLMs的内部表征是否与人类自然聆听时的脑电活动对齐。通过分析12个模型与EEG信号的相似性，揭示了模型表征的结构特征及与神经动态的关系。**

- **链接: [https://arxiv.org/pdf/2601.16540v2](https://arxiv.org/pdf/2601.16540v2)**

> **作者:** Haoyun Yang; Xin Xiao; Jiang Zhong; Yu Tian; Dong Xiaohua; Yu Mao; Hao Wu; Kaiwen Wei
>
> **摘要:** Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
>
---
#### [replaced 003] DiffRhythm 2: Efficient and High Fidelity Song Generation via Block Flow Matching
- **分类: eess.AS**

- **简介: 该论文属于歌曲生成任务，解决歌词与旋律对齐及多偏好优化问题。提出DiffRhythm 2框架，采用半自回归结构和交叉配对优化，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2510.22950v3](https://arxiv.org/pdf/2510.22950v3)**

> **作者:** Yuepeng Jiang; Huakang Chen; Ziqian Ning; Jixun Yao; Zerui Han; Di Wu; Meng Meng; Jian Luan; Zhonghua Fu; Lei Xie
>
> **摘要:** Generating full-length, high-quality songs is challenging, as it requires maintaining long-term coherence both across text and music modalities and within the music modality itself. Existing non-autoregressive (NAR) frameworks, while capable of producing high-quality songs, often struggle with the alignment between lyrics and vocal. Concurrently, catering to diverse musical preferences necessitates reinforcement learning from human feedback (RLHF). However, existing methods often rely on merging multiple models during multi-preference optimization, which results in significant performance degradation. To address these challenges, we introduce DiffRhythm 2, an end-to-end framework designed for high-fidelity, controllable song generation. To tackle the lyric alignment problem, DiffRhythm 2 employs a semi-autoregressive architecture based on block flow matching. This design enables faithful alignment of lyrics to singing vocals without relying on external labels and constraints, all while preserving the high generation quality and efficiency of NAR models. To make this framework computationally tractable for long sequences, we implement a music variational autoencoder (VAE) that achieves a low frame rate of 5 Hz while still enabling high-fidelity audio reconstruction. In addition, to overcome the limitations of multi-preference optimization in RLHF, we propose cross-pair preference optimization. This method effectively mitigates the performance drop typically associated with model merging, allowing for more robust optimization across diverse human preferences. We further enhance musicality and structural coherence by introducing stochastic block representation alignment loss.
>
---
#### [replaced 004] Evaluating High-Resolution Piano Sustain Pedal Depth Estimation with Musically Informed Metrics
- **分类: cs.IR; cs.SD; eess.AS**

- **简介: 该论文属于钢琴延音踏板深度估计任务，旨在解决传统评估指标无法反映音乐特征的问题。通过引入动作和姿态层面的评估框架，提升模型评价的音乐相关性。**

- **链接: [https://arxiv.org/pdf/2510.03750v2](https://arxiv.org/pdf/2510.03750v2)**

> **作者:** Hanwen Zhang; Kun Fang; Ziyu Wang; Ichiro Fujinaga
>
> **摘要:** Evaluation for continuous piano pedal depth estimation tasks remains incomplete when relying only on conventional frame-level metrics, which overlook musically important features such as direction-change boundaries and pedal curve contours. To provide more interpretable and musically meaningful insights, we propose an evaluation framework that augments standard frame-level metrics with an action-level assessment measuring direction and timing using segments of press/hold/release states and a gesture-level analysis that evaluates contour similarity of each press-release cycle. We apply this framework to compare an audio-only baseline with two variants: one incorporating symbolic information from MIDI, and another trained in a binary-valued setting, all within a unified architecture. Results show that the MIDI-informed model significantly outperforms the others at action and gesture levels, despite modest frame-level gains. These findings demonstrate that our framework captures musically relevant improvements indiscernible by traditional metrics, offering a more practical and effective approach to evaluating pedal depth estimation models.
>
---
#### [replaced 005] AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，旨在提升同时语音翻译的性能。通过利用注意力机制生成对齐信息，提出AlignAtt策略，有效提升翻译质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2305.11408v3](https://arxiv.org/pdf/2305.11408v3)**

> **作者:** Sara Papi; Marco Turchi; Matteo Negri
>
> **摘要:** Attention is the core mechanism of today's most used architectures for natural language processing and has been analyzed from many perspectives, including its effectiveness for machine translation-related tasks. Among these studies, attention resulted to be a useful source of information to get insights about word alignment also when the input text is substituted with audio segments, as in the case of the speech translation (ST) task. In this paper, we propose AlignAtt, a novel policy for simultaneous ST (SimulST) that exploits the attention information to generate source-target alignments that guide the model during inference. Through experiments on the 8 language pairs of MuST-C v1.0, we show that AlignAtt outperforms previous state-of-the-art SimulST policies applied to offline-trained models with gains in terms of BLEU of 2 points and latency reductions ranging from 0.5s to 0.8s across the 8 languages.
>
---
#### [replaced 006] CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音编码任务，解决固定帧率导致的冗余问题。提出CodecSlime方法，通过动态帧率压缩时间冗余，提升编码效率与质量。**

- **链接: [https://arxiv.org/pdf/2506.21074v2](https://arxiv.org/pdf/2506.21074v2)**

> **作者:** Hankun Wang; Yiwei Guo; Chongtian Shao; Bohan Li; Kai Yu
>
> **备注:** 5 pages, accepted to ICASSP 2026
>
> **摘要:** Neural speech codecs have been widely used in audio compression and various downstream tasks. Current mainstream codecs are fixed-frame-rate (FFR), which allocate the same number of tokens to every equal-duration slice. However, speech is inherently non-uniform in temporal information density. As a result, many tokens are wasted on steady-state segments like long vowels and silences. To address this mismatch, we present CodecSlime, a plugin-style method for compressing temporal redundancy through supporting dynamic frame rate (DFR) on neural speech codecs for the first time. Our method is unsupervised and architecture-agnostic, combining two key innovations, ScheDFR and Melt-and-Cool, for adapting inference and training, respectively. When integrated into a typical VQ-GAN codec backbone and operating at 40 Hz DFR ($\approx$ 600 bps), the reconstruction WER of CodecSlime is reduced by up to 32% relative to conventional FFR baselines with the same model architecture and similar bitrates, while other metrics are also competitive. CodecSlime also enables flexible trade-offs between reconstruction quality and bitrate: a single model supports inference at multiple frame rates and consistently outperforms FFR models at the corresponding frame rates. Audio samples are available at https://acadarmeria.github.io/codecslime/.
>
---
#### [replaced 007] Modeling Sarcastic Speech: Semantic and Prosodic Cues in a Speech Synthesis Framework
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在研究讽刺语义与语气的协同作用。通过结合语义模型和语气数据，构建讽刺语音合成框架，验证两者对讽刺识别的增强效果。**

- **链接: [https://arxiv.org/pdf/2510.07096v2](https://arxiv.org/pdf/2510.07096v2)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Sarcasm is a pragmatic phenomenon in which speakers convey meanings that diverge from literal content, relying on an interaction between semantics and prosodic expression. However, how these cues jointly contribute to the recognition of sarcasm remains poorly understood. We propose a computational framework that models sarcasm as the integration of semantic interpretation and prosodic realization. Semantic cues are derived from an LLaMA 3 model fine-tuned to capture discourse-level markers of sarcastic intent, while prosodic cues are extracted through semantically aligned utterances drawn from a database of sarcastic speech, providing prosodic exemplars of sarcastic delivery. Using a speech synthesis testbed, perceptual evaluations demonstrate that both semantic and prosodic cues independently enhance listeners' perception of sarcasm, with the strongest effects emerging when the two are combined. These findings highlight the complementary roles of semantics and prosody in pragmatic interpretation and illustrate how modeling can shed light on the mechanisms underlying sarcastic communication.
>
---
#### [replaced 008] RIR-Former: Coordinate-Guided Transformer for Continuous Reconstruction of Room Impulse Responses
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于声学信号处理任务，旨在解决密集测量房间冲激响应（RIR）不现实的问题。提出RIR-Former模型，通过Transformer结构实现RIR的连续重建。**

- **链接: [https://arxiv.org/pdf/2602.01861v2](https://arxiv.org/pdf/2602.01861v2)**

> **作者:** Shaoheng Xu; Chunyi Sun; Jihui Zhang; Prasanga N. Samarasinghe; Thushara D. Abhayapala
>
> **备注:** Accepted to International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026. Equal contribution: Shaoheng Xu and Chunyi Sun
>
> **摘要:** Room impulse responses (RIRs) are essential for many acoustic signal processing tasks, yet measuring them densely across space is often impractical. In this work, we propose RIR-Former, a grid-free, one-step feed-forward model for RIR reconstruction. By introducing a sinusoidal encoding module into a transformer backbone, our method effectively incorporates microphone position information, enabling interpolation at arbitrary array locations. Furthermore, a segmented multi-branch decoder is designed to separately handle early reflections and late reverberation, improving reconstruction across the entire RIR. Experiments on diverse simulated acoustic environments demonstrate that RIR-Former consistently outperforms state-of-the-art baselines in terms of normalized mean square error (NMSE) and cosine distance (CD), under varying missing rates and array configurations. These results highlight the potential of our approach for practical deployment and motivate future work on scaling from randomly spaced linear arrays to complex array geometries, dynamic acoustic scenes, and real-world environments.
>
---
#### [replaced 009] VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐信息检索任务，旨在解决自动音乐转录中忽略演奏技巧的问题。提出VioPTT模型，联合转录音高和小提琴演奏技巧，并构建了合成数据集MOSA-VPT。**

- **链接: [https://arxiv.org/pdf/2509.23759v3](https://arxiv.org/pdf/2509.23759v3)**

> **作者:** Ting-Kang Wang; Yueh-Po Peng; Li Su; Vincent K. M. Cheung
>
> **摘要:** While automatic music transcription is well-established in music information retrieval, most models are limited to transcribing pitch and timing information from audio, and thus omit crucial expressive and instrument-specific nuances. One example is playing technique on the violin, which affords its distinct palette of timbres for maximal emotional impact. Here, we propose VioPTT (Violin Playing Technique-aware Transcription), a lightweight cascade model that directly transcribes violin playing technique in addition to pitch onset and offset. Furthermore, we release MOSA-VPT, a novel, high-quality synthetic violin playing technique dataset to circumvent the need for manually labeled annotations. Leveraging this dataset, our model demonstrated strong generalization to real-world note-level violin technique recordings in addition to achieving state-of-the-art transcription performance. To our knowledge, VioPTT is the first to jointly combine violin transcription and playing technique prediction within a unified framework.
>
---
#### [replaced 010] Bayesian Speech Synthesizers Can Learn from Multiple Teachers
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，解决TTS中不确定性建模问题。提出BELLE框架，通过贝叶斯推理捕捉语音不确定性，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2510.24372v2](https://arxiv.org/pdf/2510.24372v2)**

> **作者:** Ziyang Zhang; Yifan Gao; Xuenan Xu; Baoxiang Li; Wen Wu; Chao Zhang
>
> **摘要:** Text-to-Speech (TTS) is inherently a "one-to-many" mapping characterized by intrinsic uncertainty, yet current paradigms often oversimplify it into a deterministic regression task. While continuous-valued autoregressive (AR) models have recently emerged as a promising alternative to discrete codec-based approaches, they typically rely on a fixed-variance prior, fundamentally constraining generation to a static point estimate that ignores the dynamic variability of natural speech. To bridge this gap, we propose BELLE (Bayesian evidential learning with language modelling), a framework that shifts from deterministic prediction to principled Bayesian inference without increasing model parameters or inference latency. By modeling the acoustic target as a Normal-Inverse-Gamma distribution, BELLE captures data-dependent aleatoric uncertainty. To enable accurate variance estimation on standard single-reference datasets, we introduce a "one-to-many" training strategy that leverages synthetic samples as a statistical support set, allowing the model to learn robust distributional properties rather than merely imitating teacher artifacts. Experiments demonstrate that BELLE, trained on only ~5k hours of data, outperforms leading open-source models trained on 50k hours (achieving a 25.8% relative WER reduction) and naturally supports high-quality streaming generation. Audio samples are available at https://belletts.github.io/Belle/.
>
---
#### [replaced 011] AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models
- **分类: cs.CR; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究音频越狱攻击，针对端到端大音频语言模型，提出AUDIOJAILBREAK攻击方法，解决在弱攻击者场景下的安全漏洞问题。**

- **链接: [https://arxiv.org/pdf/2505.14103v3](https://arxiv.org/pdf/2505.14103v3)**

> **作者:** Guangke Chen; Fu Song; Zhe Zhao; Xiaojun Jia; Yang Liu; Yanchen Qiao; Weizhe Zhang; Weiping Tu; Yuhong Yang; Bo Du
>
> **备注:** Accepted by IEEE Transactions on Dependable and Secure Computing (TDSC)
>
> **摘要:** Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they exclusively focused on the attack scenario where the adversary can fully manipulate user prompts (named strong adversary) and limited in effectiveness, applicability, and practicability. In this work, we first conduct an extensive evaluation showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to-speech (TTS) techniques. We then propose AUDIOJAILBREAK, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audios do not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into the perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios is concealed by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating reverberation into the perturbation generation. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, and/or over-the-air robustness. Moreover, AUDIOJAILBREAK is also applicable to a more practical and broader attack scenario where the adversary cannot fully manipulate user prompts (named weak adversary). Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AUDIOJAILBREAK, in particular, it can jailbreak openAI's GPT-4o-Audio and bypass Meta's Llama-Guard-3 safeguard, in the weak adversary scenario. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their robustness, especially for the newly proposed weak adversary.
>
---
#### [replaced 012] Joint Estimation of Piano Dynamics and Metrical Structure with a Multi-task Multi-Scale Network
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于钢琴动态与节拍结构联合估计任务，旨在解决从音频中准确提取动态和节拍的问题。通过多任务多尺度网络实现高效预测。**

- **链接: [https://arxiv.org/pdf/2510.18190v2](https://arxiv.org/pdf/2510.18190v2)**

> **作者:** Zhanhong He; Hanyu Meng; David Huang; Roberto Togneri
>
> **备注:** Accepted to ICASSP2026 conference
>
> **摘要:** Estimating piano dynamic from audio recordings is a fundamental challenge in computational music analysis. In this paper, we propose an efficient multi-task network that jointly predicts dynamic levels, change points, beats, and downbeats from a shared latent representation. These four targets form the metrical structure of dynamics in the music score. Inspired by recent vocal dynamic research, we use a multi-scale network as the backbone, which takes Bark-scale specific loudness as the input feature. Compared to log-Mel as input, this reduces model size from 14.7 M to 0.5 M, enabling long sequential input. We use a 60-second audio length in audio segmentation, which doubled the length of beat tracking commonly used. Evaluated on the public MazurkaBL dataset, our model achieves state-of-the-art results across all tasks. This work sets a new benchmark for piano dynamic estimation and delivers a powerful and compact tool, paving the way for large-scale, resource-efficient analysis of musical expression.
>
---
