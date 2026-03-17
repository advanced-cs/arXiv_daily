# 音频 cs.SD;  eess.AS

- **最新发布 46 篇**

- **更新 29 篇**

## 最新发布

#### [new 001] Investigating the Impact of Speech Enhancement on Audio Deepfake Detection in Noisy Environments
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频深度伪造检测任务，旨在研究语音增强对噪声环境下检测性能的影响。通过对比两种增强算法，分析其对语音质量及检测效果的作用。**

- **链接: [https://arxiv.org/pdf/2603.14767](https://arxiv.org/pdf/2603.14767)**

> **作者:** Anacin; Angela; Shruti Kshirsagar; Anderson R. Avila
>
> **摘要:** Logical Access (LA) attacks, also known as audio deepfake attacks, use Text-to-Speech (TTS) or Voice Conversion (VC) methods to generate spoofed speech data. This can represent a serious threat to Automatic Speaker Verification (ASV) systems, as intruders can use such attacks to bypass voice biometric security. In this study, we investigate the correlation between speech quality and the performance of audio spoofing detection systems (i.e., LA task). For that, the performance of two enhancement algorithms is evaluated based on two perceptual speech quality measures, namely Perceptual Evaluation of Speech Quality (PESQ) and Speech-to-Reverberation Modulation Ratio (SRMR), and in respect to their impact on the audio spoofing detection system. We adopted the LA dataset, provided in the ASVspoof 2019 Challenge, and corrupted its test set with different Signal-to-Noise Ratio (SNR) levels, while leaving the training data untouched. Enhancement was applied to attenuate the detrimental effects of noisy speech, and the performances of two models, Speech Enhancement Generative Adversarial Network (SEGAN) and Metric-Optimized Generative Adversarial Network Plus (MetricGAN+), were compared. Although we expect that speech quality will correlate well with speech applications' performance, it can also have as a side effect on downstream tasks if unwanted artifacts are introduced or relevant information is removed from the speech signal. Our results corroborate with this hypothesis, as we found that the enhancement algorithm leading to the highest speech quality scores, MetricGAN+, provided the lowest Equal Error Rate (EER) on the audio spoofing detection task, whereas the enhancement method with the lowest speech quality scores, SEGAN, led to the lowest EER, thus leading to better performance on the LA task.
>
---
#### [new 002] What Counts as Real? Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音防欺骗任务，解决深度伪造检测中因语音转换和修复导致的误判问题。通过多类分类提升系统鲁棒性，保持 spoof 检测能力。**

- **链接: [https://arxiv.org/pdf/2603.14033](https://arxiv.org/pdf/2603.14033)**

> **作者:** Shree Harsha Bokkahalli Satish; Harm Lameris; Joakim Gustafson; Éva Székely
>
> **备注:** 5 pages, 4 figures, 3 tables. Submitted to Interspeech 2026
>
> **摘要:** Audio anti-spoofing systems are typically formulated as binary classifiers distinguishing bona fide from spoofed speech. This assumption fails under layered generative processing, where benign transformations introduce distributional shifts that are misclassified as spoofing. We show that phonation-modifying voice conversion and speech restoration are treated as out-of-distribution despite preserving speaker authenticity. Using a multi-class setup separating bona fide, converted, spoofed, and converted-spoofed speech, we analyse model behaviour through self-supervised learning (SSL) embeddings and acoustic correlates. The benign transformations induce a drift in the SSL space, compressing bona fide and spoofed speech and reducing classifier separability. Reformulating anti-spoofing as a multi-class problem improves robustness to benign shifts while preserving spoof detection, suggesting binary systems model the distribution of raw speech rather than authenticity itself.
>
---
#### [new 003] Evaluating Pretrained General-Purpose Audio Representations for Music Genre Classification
- **分类: eess.AS**

- **简介: 该论文属于音乐流派分类任务，探讨使用预训练音频表示（如BYOL-A）与深度神经网络提升分类性能，解决跨数据集适应问题。**

- **链接: [https://arxiv.org/pdf/2603.13871](https://arxiv.org/pdf/2603.13871)**

> **作者:** Kashish Rai; Mrinmoy Bhattacharjee
>
> **备注:** Accepted and presented at the International Conference on Pattern Recognition and Machine Intelligence (PReMI), 2025
>
> **摘要:** This study investigates the use of self-supervised learning embeddings, particularly BYOL-A, in conjunction with a deep neural network classifier for Music Genre Classification. Our experiments demonstrate that BYOL-A embeddings outperform other pre-trained models, such as PANNs and VGGish, achieving an accuracy of 81.5% on the GTZAN dataset and 64.3% on FMA-Small. The proposed DNN classifier improved performance by 10-16% over linear classifiers. We explore the effects of contrastive and triplet loss and multitask training with optimized loss weights, achieving the highest accuracy. To address cross dataset challenges, we combined GTZAN and FMA-Small into a unified 18-class label space for joint training, resulting in slight performance drops on GTZAN but comparable results on FMA-Small. The scripts developed in this work are publicly available.
>
---
#### [new 004] $τ$-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音代理评估任务，旨在解决全双工语音助手在真实场景下的性能评测问题。工作包括构建τ-voice基准，结合复杂任务、语音交互和真实环境，评估语音与文本性能差异。**

- **链接: [https://arxiv.org/pdf/2603.13686](https://arxiv.org/pdf/2603.13686)**

> **作者:** Soham Ray; Keshav Dhandhania; Victor Barres; Karthik Narasimhan
>
> **摘要:** Full-duplex voice agents--systems that listen and speak simultaneously--are rapidly moving from research to production. However, existing evaluations address conversational dynamics and task completion in isolation. We introduce $\tau$-voice, a benchmark for evaluating voice agents on grounded tasks with real-world complexity: agents must navigate complex multi-turn conversations, adhere to domain policies, and interact with the environment. The framework extends $\tau^2$-bench into a novel voice agent benchmark combining verifiable completion of complex grounded tasks, full-duplex interaction, and realistic audio--enabling direct comparison between voice and text performance. A controllable and realistic voice user simulator provides diverse accents, realistic audio environments, and rich turn-taking dynamics; by decoupling simulation from wall-clock time, the user simulator can use the most capable LLM without real-time constraints. We evaluate task completion (pass@1) and voice interaction quality across 278 tasks: while GPT-5 (reasoning) achieves 85%, voice agents reach only 31--51% under clean conditions and 26--38% under realistic conditions with noise and diverse accents--retaining only 30--45% of text capability; qualitative analysis confirms 79--90% of failures stem from agent behavior, suggesting that observed failures primarily reflect agent behavior under our evaluation setup. $\tau$-voice provides a reproducible testbed for measuring progress toward voice agents that are natural, conversational, and reliable.
>
---
#### [new 005] Deep Filter Estimation from Inter-Frame Correlations for Monaural Speech Dereverberation
- **分类: eess.AS**

- **简介: 该论文属于语音去混响任务，旨在解决远距离麦克风场景下混响与目标信号相关性高导致的泛化能力差问题。提出IF-CorrNet，通过分析帧间相关性估计深度滤波器，提升实际环境下的去混响效果。**

- **链接: [https://arxiv.org/pdf/2603.14986](https://arxiv.org/pdf/2603.14986)**

> **作者:** Ui-Hyeop Shin; Jun Hyung Kim; Jangyeon Kim; Wooseok Kim; Hyung-Min Park
>
> **备注:** Submitted for review to Interspeech
>
> **摘要:** Speech dereverberation in distant-microphone scenarios remains challenging due to the high correlation between reverberation and target signals, often leading to poor generalization in real-world environments. We propose IF-CorrNet, a correlation-to-filter architecture designed for robustness against acoustic variability. Unlike conventional black-box mapping methods that directly estimate complex spectra, IF-CorrNet explicitly exploits inter-frame STFT correlations to estimate multi-frame deep filters for each time-frequency bin. By shifting the learning objective from direct mapping to filter estimation, the network effectively constrains the solution space, which simplifies the training process and mitigates overfitting to synthetic data. Experimental results on the REVERB Challenge dataset demonstrate that IF-CorrNet achieves a substantial gain in the SRMR metric on RealData, confirming its robustness in suppressing reverberation and noise in practical, non-synthetic environments.
>
---
#### [new 006] SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation
- **分类: eess.AS**

- **简介: 该论文提出SoulX-Duplug，用于实时全双工语音对话的状态预测，解决数据获取难、遗忘和扩展性差的问题。通过融合流式语音识别，提升对话管理效率。**

- **链接: [https://arxiv.org/pdf/2603.14877](https://arxiv.org/pdf/2603.14877)**

> **作者:** Ruiqi Yan; Wenxi Chen; Zhanxun Liu; Ziyang Ma; Haopeng Lin; Hanlin Wen; Hanke Xie; Jun Wu; Yuzhe Liang; Yuxiang Zhao; Pengchao Feng; Jiale Qian; Hao Meng; Yuhang Dai; Shunshun Yin; Ming Tao; Lei Xie; Kai Yu; Xinsheng Wang; Xie Chen
>
> **备注:** submitted to Interspeech 2026, under review
>
> **摘要:** Recent advances in spoken dialogue systems have brought increased attention to human-like full-duplex voice interactions. However, our comprehensive review of this field reveals several challenges, including the difficulty in obtaining training data, catastrophic forgetting, and limited scalability. In this work, we propose SoulX-Duplug, a plug-and-play streaming state prediction module for full-duplex spoken dialogue systems. By jointly performing streaming ASR, SoulX-Duplug explicitly leverages textual information to identify user intent, effectively serving as a semantic VAD. To promote fair evaluation, we introduce SoulX-Duplug-Eval, extending widely used benchmarks with improved bilingual coverage. Experimental results show that SoulX-Duplug enables low-latency streaming dialogue state control, and the system built upon it outperforms existing full-duplex models in overall turn management and latency performance. We have open-sourced SoulX-Duplug and SoulX-Duplug-Eval.
>
---
#### [new 007] Two-Stage Adaptation for Non-Normative Speech Recognition: Revisiting Speaker-Independent Initialization for Personalization
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决非典型语音个性化难题。通过两阶段微调框架提升模型适应非典型语音的能力，同时控制域外性能下降。**

- **链接: [https://arxiv.org/pdf/2603.15261](https://arxiv.org/pdf/2603.15261)**

> **作者:** Shan Jiang; Jiawen Qi; Chuanbing Huo; Yingqiang Gao; Qinyu Chen
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Personalizing automatic speech recognition (ASR) systems for non-normative speech, such as dysarthric and aphasic speech, is challenging. While speaker-specific fine-tuning (SS-FT) is widely used, it is typically initialized directly from a generic pre-trained model. Whether speaker-independent adaptation provides a stronger initialization prior under such mismatch remains unclear. In this work, we propose a two-stage adaptation framework consisting of speaker-independent fine-tuning (SI-FT) on multi-speaker non-normative data followed by SS-FT, and evaluate it through a controlled comparison with direct SS-FT under identical per-speaker conditions. Experiments on AphasiaBank and UA-Speech with Whisper-Large-v3 and Qwen3-ASR, alongside evaluation on typical-speech datasets TED-LIUM v3 and FLEURS, show that two-stage adaptation consistently improves personalization while maintaining manageable out-of-domain (OOD) trade-offs.
>
---
#### [new 008] VoXtream2: Full-stream TTS with dynamic speaking rate control
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，解决交互系统中低延迟与动态语速控制问题。提出VoXtream2模型，实现快速合成与灵活调整。**

- **链接: [https://arxiv.org/pdf/2603.13518](https://arxiv.org/pdf/2603.13518)**

> **作者:** Nikita Torgashov; Gustav Eje Henter; Gabriel Skantze
>
> **备注:** 10 pages, 9 figures, Submitted to Interspeech 2026
>
> **摘要:** Full-stream text-to-speech (TTS) for interactive systems must start speaking with minimal delay while remaining controllable as text arrives incrementally. We present VoXtream2, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly. VoXtream2 combines a distribution matching mechanism over duration states with classifier-free guidance across conditioning signals to improve controllability and synthesis quality. Prompt-text masking enables textless audio prompting, removing the need for prompt transcription. Across standard zero-shot benchmarks and a dedicated speaking-rate test set, VoXtream2 achieves competitive objective and subjective results against public baselines despite a smaller model and less training data. In full-stream mode, it runs 4 times faster than real time with 74 ms first-packet latency on a consumer GPU.
>
---
#### [new 009] Evaluating Compositional Structure in Audio Representations
- **分类: cs.SD**

- **简介: 该论文属于音频表示学习任务，旨在评估音频的组合性。通过构建基准测试，解决当前评估协议缺乏组合性分析的问题，提出两个任务并使用合成数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2603.13685](https://arxiv.org/pdf/2603.13685)**

> **作者:** Chuyang Chen; Bea Steers; Brian McFee; Juan Bello
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** We propose a benchmark for evaluating compositionality in audio representations. Audio compositionality refers to representing sound scenes in terms of constituent sources and attributes, and combining them systematically. While central to auditory perception, this property is largely absent from current evaluation protocols. Our framework adapts ideas from vision and language to audio through two tasks: A-COAT, which tests consistency under additive transformations, and A-TRE, which probes reconstructibility from attribute-level primitives. Both tasks are supported by large synthetic datasets with controlled variation in acoustic attributes, providing the first benchmark of compositional structure in audio embeddings.
>
---
#### [new 010] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于目标语音提取任务，解决现实场景中不同重叠比例下的语音分离问题。提出VorTEX模型和PORTE数据集，提升分离效果并避免抑制现象。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** arXiv Preprint
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
#### [new 011] spINAch: A Diachronic Corpus of French Broadcast Speech Controlled for Speakers' Age and Gender
- **分类: eess.AS**

- **简介: 该论文介绍spINAch语料库，用于研究法语广播语音随时间的变化，解决语音演变分析问题，通过收集并分析不同年龄和性别的语音数据进行研究。**

- **链接: [https://arxiv.org/pdf/2603.15516](https://arxiv.org/pdf/2603.15516)**

> **作者:** Simon Devauchelle; David Doukhan; Rémi Uro; Lucas Ondel Yang; Valentin Pelloin; Olympia Imbert-Brégégère; Véronique Lefort; Kévin Picard; Emeline Seignobos; Albert Rilliard
>
> **备注:** 16 pages, 3 figures, to be published in the Fifteenth International Conference on Language Resources and Evaluation (LREC 2026)
>
> **摘要:** We present spINAch, a large diachronic corpus of French speech from radio and television archives, balanced by speakers' gender, age (20-95 years old), and spanning 60 years from 1955 to 2015. The dataset includes over 320 hours of recordings from more than two thousand speakers. The methodology for building the corpus is described, focusing on the quality of collected samples in acoustic terms. The data were automatically transcribed and phonetically aligned to allow studies at a phonemic level. More than 3 million oral vowels have been analyzed to propose their fundamental frequency and formants. The corpus, available to the community for research purposes, is valuable for describing the evolution of Parisian French through the representation of gender and age. The presented analyses also demonstrate that the diachronic nature of the corpus allows the observation of various phonetic phenomena, such as the evolution of voice pitch over time (which does not differ by gender in our data) and the neutralization of the /a/-/$a$/ opposition in Parisian French during this period.
>
---
#### [new 012] Understanding the strengths and weaknesses of SSL models for audio deepfake model attribution
- **分类: eess.AS**

- **简介: 该论文属于音频深度伪造模型溯源任务，旨在识别合成语音的生成模型。研究分析SSL特征如何捕捉模型架构特征，揭示其优势与局限性。**

- **链接: [https://arxiv.org/pdf/2603.13488](https://arxiv.org/pdf/2603.13488)**

> **作者:** Gabriel Pîrlogeanu; Adriana Stan; Horia Cucu
>
> **备注:** Accepted for publication at ICASSP 2026
>
> **摘要:** Audio deepfake model attribution aims to mitigate the misuse of synthetic speech by identifying the source model responsible for generating a given audio sample, enabling accountability and informing vendors. The task is challenging, but self-supervised learning (SSL)-derived acoustic features have demonstrated state-of-the-art attribution capabilities, yet the underlying factors driving their success and the limits of their discriminative power remain unclear. In this paper, we systematically investigate how SSL-derived features capture architectural signatures in audio deepfakes. By controlling multiple dimensions of the audio generation process we reveal how subtle perturbations in model checkpoints, text prompts, vocoders, or speaker identity influence attribution. Our results provide new insights into the robustness, biases, and limitations of SSL-based deepfake attribution, highlighting both its strengths and vulnerabilities in realistic scenarios.
>
---
#### [new 013] Probing neural audio codecs for distinctions among English nuclear tunes
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音处理任务，旨在检验神经音频编解码器是否能区分英语语调类型。通过训练分类器，研究发现编码器可部分捕捉语调模式，但仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2603.14035](https://arxiv.org/pdf/2603.14035)**

> **作者:** Juan Pablo Vigneaux; Jennifer Cole
>
> **备注:** 5 pages; 1 table; 3 figures. Accepted as conference paper at Speech Prosody 2026
>
> **摘要:** State-of-the-art spoken dialogue models (Défossez et al. 2024; Schalkwyk et al. 2025) use neural audio codecs to "tokenize" audio signals into a lower-frequency stream of vectorial latent representations, each quantized using a hierarchy of vector codebooks. A transformer layer allows these representations to reflect some time- and context-dependent patterns. We train probes on labeled audio data from Cole et al. (2023) to test whether the pitch trajectories that characterize English phrase-final (nuclear) intonational tunes are among these patterns. Results: Linear probes trained on the unquantized latents or some of the associated codewords yield above-chance accuracy in distinguishing eight phonologically specified nuclear tunes with monotonal pitch accents (top average test accuracy (TATA): 0.31) and the five clusters of these tunes that are robust in human speech production and perception (TATA: 0.45). Greater accuracy (TATAs: 0.74-0.89) is attained for binary distinctions between classes of rising vs. falling tunes, respectively used for questions and assertions. Information about tunes is spread among all codebooks, which calls into question a distinction between 'semantic' and 'acoustic' codebooks found in the literature. Accuracies improve with nonlinear probes, but discrimination among the five clusters remains far from human performance, suggesting a fundamental limitation of current codecs.
>
---
#### [new 014] Cepstral Smoothing of Binary Masks for Convolutive Blind Separation of Speech Mixtures
- **分类: cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决混响环境下语音信号分离问题。通过结合盲源分离与谱图平滑技术，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2603.14983](https://arxiv.org/pdf/2603.14983)**

> **作者:** Ibrahim Missaoui; Zied Lachiri
>
> **摘要:** In this paper, we propose a novel separation system for extracting two speech signals from two microphone recordings. Our system combines the blind source separation technique with cepstral smoothing of binary time-frequency masks. The last is composed of two steps. First, the two binary masks are estimated from the separated output signals of BSS algorithm. In the second step, a cepstral smoothing is applied of these spectral masks in order to reduce musical noise typically produced by time-frequency masking. Experiments were carried out with both artificially mixed speech signals using simulated room model and two real recordings. The evaluation results are promising and have shown the effectiveness of our system.
>
---
#### [new 015] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决传统方法依赖文本描述导致的语义模糊和声学细节不足问题。通过引入参考音频，实现更精确的音效合成与控制。**

- **链接: [https://arxiv.org/pdf/2603.15597](https://arxiv.org/pdf/2603.15597)**

> **作者:** Pengjun Fang; Yingqing He; Yazhou Xing; Qifeng Chen; Ser-Nam Lim; Harry Yang
>
> **备注:** Accepted at ICLR 2026. 15 pages, 5 figures
>
> **摘要:** Existing video-to-audio (V2A) generation methods predominantly rely on text prompts alongside visual information to synthesize audio. However, two critical bottlenecks persist: semantic granularity gaps in training data, such as conflating acoustically distinct sounds under coarse labels, and textual ambiguity in describing micro-acoustic features. These bottlenecks make it difficult to perform fine-grained sound synthesis using text-controlled modes. To address these limitations, we propose AC-Foley, an audio-conditioned V2A model that directly leverages reference audio to achieve precise and fine-grained control over generated sounds. This approach enables fine-grained sound synthesis, timbre transfer, zero-shot sound generation, and improved audio quality. By directly conditioning on audio signals, our approach bypasses the semantic ambiguities of text descriptions while enabling precise manipulation of acoustic attributes. Empirically, AC-Foley achieves state-of-the-art performance for Foley generation when conditioned on reference audio, while remaining competitive with state-of-the-art video-to-audio methods even without audio conditioning.
>
---
#### [new 016] Spectrogram features for audio and speech analysis
- **分类: eess.AS; cs.AI; cs.LG; eess.SP**

- **简介: 论文探讨了基于频谱图的音频和语音分析方法，研究其与分类器架构的匹配关系，旨在优化特征表示与模型设计的协同效果。**

- **链接: [https://arxiv.org/pdf/2603.14917](https://arxiv.org/pdf/2603.14917)**

> **作者:** Ian McLoughlin; Lam Pham; Yan Song; Xiaoxiao Miao; Huy Phan; Pengfei Cai; Qing Gu; Jiang Nan; Haoyu Song; Donny Soh
>
> **备注:** 30 pages
>
> **摘要:** Spectrogram-based representations have grown to dominate the feature space for deep learning audio analysis systems, and are often adopted for speech analysis also. Initially, the primary motivator for spectrogram-based representations was their ability to present sound as a two dimensional signal in the time-frequency plane, which not only provides an interpretable physical basis for analysing sound, but also unlocks the use of a wide range of machine learning techniques such as convolutional neural networks, that had been developed for image processing. A spectrogram is a matrix characterised by the resolution and span of its two dimensions, as well as by the representation and scaling of each element. Many possibilities for these three characteristics have been explored by researchers across numerous application areas, with different settings showing affinity for various tasks. This paper reviews the use of spectrogram-based representations and surveys the state-of-the-art to question how front-end feature representation choice allies with back-end classifier architecture for different tasks.
>
---
#### [new 017] Sub-Band Spectral Matching with Localized Score Aggregation for Robust Anomalous Sound Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于异常声音检测任务，解决噪声环境下微小偏差检测问题。提出BEAM方法，通过子带匹配和分数聚合降低正常样本得分方差，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.13749](https://arxiv.org/pdf/2603.13749)**

> **作者:** Phurich Saengthong; Takahiro Shinozaki
>
> **备注:** Manuscript under review
>
> **摘要:** Detecting subtle deviations in noisy acoustic environments is central to anomalous sound detection (ASD). A common training-free ASD pipeline temporally pools frame-level representations into a band-preserving feature vector and scores anomalies using a single nearest-neighbor match. However, this global matching can inflate normal-score variance through two effects. First, when normal sounds exhibit band-wise variability, a single global neighbor forces all bands to share the same reference, increasing band-level mismatch. Second, cosine-based matching is energy-coupled, allowing a few high-energy bands to dominate score computation under normal energy fluctuations and further increase variance. We propose BEAM, which stores temporally pooled sub-band vectors in a memory bank, retrieves neighbors per sub-band, and uniformly aggregates scores to reduce normal-score variability and improve discriminability. We further introduce a parameter-free adaptive fusion to better handle diverse temporal dynamics in sub-band responses. Experiments on multiple DCASE Task 2 benchmarks show strong performance without task-specific training, robustness to noise and domain shifts, and complementary gains when combined with encoder fine-tuning.
>
---
#### [new 018] Integrated Spoofing-Robust Automatic Speaker Verification via a Three-Class Formulation and LLR
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音安全任务，旨在提升自动说话人验证的抗欺骗能力。通过三类分类框架实现端到端 spoofing-robust ASV，提高决策可解释性。**

- **链接: [https://arxiv.org/pdf/2603.13780](https://arxiv.org/pdf/2603.13780)**

> **作者:** Kai Tan; Lin Zhang; Ruiteng Zhang; Johan Rohdin; Leibny Paola García-Perera; Zexin Cai; Sanjeev Khudanpur; Matthew Wiesner; Nicholas Andrews
>
> **备注:** Submitted to Interspeech 2026; put on arxiv based on requirement from Interspeech: "Interspeech no longer enforces an anonymity period for submissions." and "For authors that prefer to upload their paper online, a note indicating that the paper was submitted for review to Interspeech should be included in the posting."
>
> **摘要:** Spoofing-robust automatic speaker verification (SASV) aims to integrate automatic speaker verification (ASV) and countermeasure (CM). A popular solution is fusion of independent ASV and CM scores. To better modeling SASV, some frameworks integrate ASV and CM within a single network. However, these solutions are typically bi-encoder based, offer limited interpretability, and cannot be readily adapted to new evaluation parameters without retraining. Based on this, we propose a unified end-to-end framework via a three-class formulation that enables log-likelihood ratio (LLR) inference from class logits for a more interpretable decision pipeline. Experiments show comparable performance to existing methods on ASVSpoof5 and better results on SpoofCeleb. The visualization and analysis also prove that the three-class reformulation provides more interpretability.
>
---
#### [new 019] LLM-Guided Reinforcement Learning for Audio-Visual Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-视觉语音增强任务，旨在解决传统指标与感知质量相关性低的问题。通过引入基于大语言模型的强化学习框架，提升语音增强效果。**

- **链接: [https://arxiv.org/pdf/2603.13952](https://arxiv.org/pdf/2603.13952)**

> **作者:** Chih-Ning Chen; Jen-Cheng Hou; Hsin-Min Wang; Shao-Yi Chien; Yu Tsao; Fan-Gang Zeng
>
> **备注:** 6 pages, 4 figures, submitted to Interspeech 2026
>
> **摘要:** In existing Audio-Visual Speech Enhancement (AVSE) methods, objectives such as Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Mean Squared Error (MSE) are widely used; however, they often correlate poorly with perceptual quality and provide limited interpretability for optimization. This work proposes a reinforcement learning-based AVSE framework with a Large Language Model (LLM)-based interpretable reward model. An audio LLM generates natural language descriptions of enhanced speech, which are converted by a sentiment analysis model into a 1-5 rating score serving as the PPO reward for fine-tuning a pretrained AVSE model. Compared with scalar metrics, LLM-generated feedback is semantically rich and explicitly describes improvements in speech quality. Experiments on the 4th COG-MHEAR AVSE Challenge (AVSEC-4) dataset show that the proposed method outperforms a supervised baseline and a DNSMOS-based RL baseline in PESQ, STOI, neural quality metrics, and subjective listening tests.
>
---
#### [new 020] How Attention Shapes Emotion: A Comparative Study of Attention Mechanisms for Speech Emotion Recognition
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决注意力机制在情感语音建模中的效率与准确性平衡问题。通过对比多种优化注意力机制，评估其性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.15120](https://arxiv.org/pdf/2603.15120)**

> **作者:** Marc Casals-Salvador; Federico Costa; Rodolfo Zevallos; Javier Hernando
>
> **摘要:** Speech Emotion Recognition (SER) plays a key role in advancing human-computer interaction. Attention mechanisms have become the dominant approach for modeling emotional speech due to their ability to capture long-range dependencies and emphasize salient information. However, standard self-attention suffers from quadratic computational and memory complexity, limiting its scalability. In this work, we present a systematic benchmark of optimized attention mechanisms for SER, including RetNet, LightNet, GSA, FoX, and KDA. Experiments on both MSP-Podcast benchmark versions show that while standard self-attention achieves the strongest recognition performance across test sets, efficient attention variants dramatically improve scalability, reducing inference latency and memory usage by up to an order of magnitude. These results highlight a critical trade-off between accuracy and efficiency, providing practical insights for designing scalable SER systems.
>
---
#### [new 021] Controllable Accent Normalization via Discrete Diffusion
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音处理任务，解决accent normalization问题。提出DLM-AN系统，通过离散扩散模型实现可控的口音归一化，可调节口音强度并保持自然节奏。**

- **链接: [https://arxiv.org/pdf/2603.14275](https://arxiv.org/pdf/2603.14275)**

> **作者:** Qibing Bai; Yuhan Du; Tom Ko; Shuai Wang; Yannan Wang; Haizhou Li
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** Existing accent normalization methods do not typically offer control over accent strength, yet many applications-such as language learning and dubbing-require tunable accent retention. We propose DLM-AN, a controllable accent normalization system built on masked discrete diffusion over self-supervised speech tokens. A Common Token Predictor identifies source tokens that likely encode native pronunciation; these tokens are selectively reused to initialize the reverse diffusion process. This provides a simple yet effective mechanism for controlling accent strength: reusing more tokens preserves more of the original accent. DLM-AN further incorporates a flow-matching Duration Ratio Predictor that automatically adjusts the total duration to better match the native rhythm. Experiments on multi-accent English data show that DLM-AN achieves the lowest word error rate among all compared systems while delivering competitive accent reduction and smooth, interpretable accent strength control.
>
---
#### [new 022] Neural Network-Based Time-Frequency-Bin-Wise Linear Combination of Beamformers for Underdetermined Target Source Extraction
- **分类: eess.AS**

- **简介: 该论文属于语音分离任务，解决混音中目标源提取的问题。提出一种基于神经网络的时频 bin 线性组合方法，提升分离性能。**

- **链接: [https://arxiv.org/pdf/2603.15288](https://arxiv.org/pdf/2603.15288)**

> **作者:** Changda Chen; Yichen Yang; Wei Liu; Shoji Makino
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Extracting a target source from underdetermined mixtures is challenging for beamforming approaches. Recently proposed time-frequency-bin-wise switching (TFS) and linear combination (TFLC) strategies mitigate this by combining multiple beamformers in each time-frequency (TF) bin and choosing combination weights that minimize the output power. However, making this decision independently for each TF bin can weaken temporal-spectral coherence, causing discontinuities and consequently degrading extraction performance. In this paper, we propose a novel neural network-based time-frequency-bin-wise linear combination (NN-TFLC) framework that constructs minimum power distortionless response (MPDR) beamformers without explicit noise covariance estimation. The network encodes the mixture and beamformer outputs, and predicts temporally and spectrally coherent linear combination weights via a cross-attention mechanism. On dual-microphone mixtures with multiple interferers, NN-TFLC-MPDR consistently outperforms TFS/TFLC-MPDR and achieves competitive performance with TFS/TFLC built on the minimum variance distortionless response (MVDR) beamformers that require noise priors.
>
---
#### [new 023] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations
- **分类: cs.SD**

- **简介: 该论文属于情感语音合成任务，旨在解决非语言发声（如笑、叹气）在开放场景中数据不足且缺乏监督的问题。提出Affectron框架，通过增强训练策略和结构掩码，生成更自然、多样的非语言发声。**

- **链接: [https://arxiv.org/pdf/2603.14432](https://arxiv.org/pdf/2603.14432)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **摘要:** Nonverbal vocalizations (NVs), such as laughter and sighs, are central to the expression of affective cues in emotional speech synthesis. However, learning diverse and contextually aligned NVs remains challenging in open settings due to limited NV data and the lack of explicit supervision. Motivated by this challenge, we propose Affectron as a framework for affective and contextually aligned NV generation. Built on a small-scale open and decoupled corpus, Affectron introduces an NV-augmented training strategy that expands the distribution of NV types and insertion locations. We further incorporate NV structural masking into a speech backbone pre-trained on purely verbal speech to enable diverse and natural NV synthesis. Experimental results demonstrate that Affectron produces more expressive and diverse NVs than baseline systems while preserving the naturalness of the verbal speech stream.
>
---
#### [new 024] BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出BrainWhisperer，解决脑机接口中从脑信号解码语音的问题。通过结合大规模ASR模型与神经数据，提升解码准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13321](https://arxiv.org/pdf/2603.13321)**

> **作者:** Tommaso Boccato; Michal Olak; Matteo Ferrante
>
> **摘要:** Decoding continuous speech from intracortical recordings is a central challenge for brain-computer interfaces (BCIs), with transformative potential for individuals with conditions that impair their ability to speak. While recent microelectrode array (MEA) decoders achieve impressive accuracy, their performance is fundamentally limited by the small size of existing datasets, they remain brittle to session-to-session variability, and their ability to generalize across participants remains unexplored. We introduce BrainWhisperer, a neural speech decoder that integrates high-resolution MEA recordings with a large pretrained automatic speech recognition (ASR) model. Building on interpretability findings showing that Whisper's encoder learns phoneme-selective representations with localized attention, we train a customized version of Whisper, modified to process neural features, using a hybrid objective that combines CTC loss on phonemes--predicted from the third encoder layer--and cross-entropy loss on word tokens. We introduce domain-informed modifications including windowed self-attention to capture articulatory continuity, hierarchical month/day-specific low-rank projections to address non-stationarity, and subject-specific embedders enabling cross-subject training. Evaluated on a publicly available MEA dataset (Card et al.), BrainWhisperer matches or outperforms prior state-of-the-art decoders. Critically, cross-dataset training improves performance even on individual datasets without fine-tuning, demonstrating unprecedented generalization. The model supports dual decoding paths: a high-accuracy phoneme-based path with external language model rescoring, and a fast direct text generation path enabling sub-100ms inference with minimal hardware requirements.
>
---
#### [new 025] Patient-Level Multimodal Question Answering from Multi-Site Auscultation Recordings
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多模态问答任务，旨在解决听诊信号主观解读的问题。通过将多部位听诊数据与语言模型对齐，实现患者级评估，提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.13362](https://arxiv.org/pdf/2603.13362)**

> **作者:** Fan Wu; Tsai-Ning Wang; Nicolas Zumarraga; Ning Wang; Markus Kreft; Kevin O'Sullivan; Elgar Fleisch; Oliver Aalami; Paul Schmiedmayer; Robert Jakob; Patrick Langer
>
> **摘要:** Auscultation is a vital diagnostic tool, yet its utility is often limited by subjective interpretation. While general-purpose Audio-Language Models (ALMs) excel in general domains, they struggle with the nuances of physiological signals. We propose a framework that aligns multi-site auscultation recordings directly with a frozen Large Language Model (LLM) embedding space via gated cross-attention. By leveraging the LLM's latent world knowledge, our approach moves beyond isolated classification toward holistic, patient-level assessment. On the CaReSound benchmark, our model achieves a state-of-the-art 0.865 F1-macro and 0.952 BERTScore. We demonstrate that lightweight, domain-specific encoders rival large-scale ALMs and that multi-site aggregation provides spatial redundancy that mitigates temporal truncation. This alignment of medical acoustics with text foundations offers a scalable path for bridging signal processing and clinical assessment.
>
---
#### [new 026] Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型推理任务，旨在提升链式思维提示效果。通过无训练的模型调控方法，利用跨模态信息提高推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.14636](https://arxiv.org/pdf/2603.14636)**

> **作者:** Lok-Lam Ieong; Chia-Chien Chen; Chih-Kai Yang; Yu-Han Huang; An-Yu Cheng; Hung-yi Lee
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Chain-of-thought (CoT) prompting has been extended to large audio-language models (LALMs) to elicit reasoning, yet enhancing its effectiveness without training remains challenging. We study inference-time model steering as a training-free approach to improve LALM reasoning. We introduce three strategies using diverse information sources and evaluate them across four LALMs and four benchmarks. Results show general accuracy gains up to 4.4% over CoT prompting. Notably, we identify a cross-modal transfer where steering vectors derived from few text samples effectively guide speech-based reasoning, demonstrating high data efficiency. We also examine hyperparameter sensitivity to understand the robustness of these approaches. Our findings position model steering as a practical direction for strengthening LALM reasoning.
>
---
#### [new 027] LLMs and Speech: Integration vs. Combination
- **分类: eess.AS**

- **简介: 该论文研究如何将预训练大语言模型用于语音识别，比较了语音LLM与传统浅层融合方法。任务是自动语音识别，解决模型整合与优化问题，进行了多种实验和优化。**

- **链接: [https://arxiv.org/pdf/2603.15045](https://arxiv.org/pdf/2603.15045)**

> **作者:** Robin Schmitt; Albert Zeyer; Mohammad Zeineldeen; Ralf Schlüter; Hermann Ney
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** In this work, we study how to best utilize pre-trained LLMs for automatic speech recognition. Specifically, we compare the tight integration of an acoustic model (AM) with the LLM ("speech LLM") to the traditional way of combining AM and LLM via shallow fusion. For tight integration, we provide ablations on the effect of different label units, fine-tuning strategies, LLM sizes and pre-training data, attention interfaces, encoder downsampling, text prompts, and length normalization. Additionally, we investigate joint recognition with a CTC model to mitigate hallucinations of speech LLMs and present effective optimizations for this joint recognition. For shallow fusion, we investigate the effect of fine-tuning the LLM on the transcriptions using different label units, and we compare rescoring AM hypotheses to single-pass recognition with label-wise or delayed fusion of AM and LLM scores. We train on Librispeech and Loquacious and evaluate our models on the HuggingFace ASR leaderboard.
>
---
#### [new 028] Evaluation of Audio Language Models for Fairness, Safety, and Security
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频语言模型的公平性、安全性和安全性评估任务，旨在解决现有评估不系统的问题。通过构建分类体系和统一框架，分析模型在不同输入下的表现差异。**

- **链接: [https://arxiv.org/pdf/2603.13262](https://arxiv.org/pdf/2603.13262)**

> **作者:** Ranya Aloufi; Srishti Gupta; Soumya Shaw; Battista Biggio; Lea Schönherr
>
> **摘要:** Audio large language models (ALLMs) have recently advanced spoken interaction by integrating speech processing with large language models. However, existing evaluations of fairness, safety, and security (FSS) remain fragmented, largely because ALLMs differ fundamentally in how acoustic information is represented and where semantic reasoning occurs. Differences that are rarely made explicit. As a result, evaluations often conflate structurally distinct systems, obscuring the relationship between model design and observed FSS behavior. In this work, we introduce a structural taxonomy (system-level and representational) of ALLMs that categorizes systems along two axes: the form of audio input representation (e.g., discrete vs. continuous) and the locus of semantic reasoning (e.g., cascaded, multimodal, or audio-native). Building on the taxonomy, we propose a unified evaluation framework that assesses semantic invariance under paralinguistic variation, refusal and toxicity behavior under unsafe prompts, and robustness to adversarial audio perturbations. We apply this framework to two representative systems and observe systematic differences in refusal rates, attack success, and toxicity between audio and text inputs. Our findings demonstrate that FSS behavior is tightly coupled to how acoustic information is integrated into semantic reasoning, underscoring the need for structure-aware evaluation of audio language models.
>
---
#### [new 029] Beyond Two-stage Diffusion TTS: Joint Structure and Content Refinement via Jump Diffusion
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决时序结构与谱内容建模的矛盾。提出跳跃扩散框架，联合优化时序与内容，提升语音自然度。**

- **链接: [https://arxiv.org/pdf/2603.14032](https://arxiv.org/pdf/2603.14032)**

> **作者:** Jiabao Ai; Minghui Zhao; Anton Ragni
>
> **备注:** 5 pages, 5 figures. Audio samples available at this https URL
>
> **摘要:** Diffusion and flow matching TTS faces a tension between discrete temporal structure and continuous spectral modeling. Two-stage models diffuse on fixed alignments, often collapsing to mean prosody; single-stage models avoid explicit durations but suffer alignment instability. We propose a jump-diffusion framework where discrete jumps model temporal structure and continuous diffusion refines spectral content within one process. Even in its one-shot degenerate form, our framework achieves 3.37% WER vs. 4.38% for Grad-TTS with improved UTMOSv2 on LJSpeech. The full iterative UDD variant further enables adaptive prosody, autonomously inserting natural pauses in out-of-distribution slow speech rather than stretching uniformly. Audio samples are available at this https URL.
>
---
#### [new 030] Evaluating Semantic Fragility in Text-to-Audio Generation Systems Under Controlled Prompt Perturbations
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到音频生成任务，旨在评估模型在语义相似提示下的鲁棒性。通过控制提示扰动，分析模型的语义脆弱性，提出多维度评估框架。**

- **链接: [https://arxiv.org/pdf/2603.13824](https://arxiv.org/pdf/2603.13824)**

> **作者:** Jiahui Wu
>
> **备注:** 8 pages, 4 figures, Under ICCC'26 review
>
> **摘要:** Recent advances in text-to-audio generation enable models to translate natural-language descriptions into diverse musical output. However, the robustness of these systems under semantically equivalent prompt variations remains largely unexplored. Small linguistic changes may lead to substantial variation in generated audio, raising concerns about reliability in practical use. In this study, we evaluate the semantic fragility of text-to-audio systems under controlled prompt perturbations. We selected MusicGen-small, MusicGen-large, and Stable Audio 2.5 as representative models, and we evaluated them under Minimal Lexical Substitution (MLS), Intensity Shifts (IS), and Structural Rephrasing (SR). The proposed dataset contains 75 prompt groups designed to preserve semantic intent while introducing localized linguistic variation. Generated outputs are compared through complementary spectral, temporal, and semantic similarity measures, enabling robustness analysis across multiple representational levels. Experimental results show that larger models achieve improved semantic consistency, with MusicGen-large reaching cosine similarities of 0.77 under MLS and 0.82 under IS. However, acoustic and temporal analyses reveal persistent divergence across all models, even when embedding similarity remains high. These findings indicate that fragility arises primarily during semantic-to-acoustic realization rather than multi-modal embedding alignment. Our study introduces a controlled framework for evaluating robustness in text-to-audio generation and highlights the need for multi-level stability assessment in generative audio systems.
>
---
#### [new 031] PhonemeDF: A Synthetic Speech Dataset for Audio Deepfake Detection and Naturalness Evaluation
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决合成语音自然度评估难题。提出PhonemeDF数据集，包含真实与合成语音的音素级对比，用于检测与评价。**

- **链接: [https://arxiv.org/pdf/2603.15037](https://arxiv.org/pdf/2603.15037)**

> **作者:** Vamshi Nallaguntla; Aishwarya Fursule; Shruti Kshirsagar; Anderson R. Avila
>
> **备注:** 11 pages, 6 figures, 9 tables. Accepted at the 15th Language Resources and Evaluation Conference (LREC 2026), Palma, Spain
>
> **摘要:** The growing sophistication of speech generated by Artificial Intelligence (AI) has introduced new challenges in audio deepfake detection. Text-to-speech (TTS) and voice conversion (VC) technologies can create highly convincing synthetic speech with naturalness and intelligibility. This poses serious threats to voice biometric security and to systems designed to combat the spread of spoken misinformation, where synthetic voices may be used to disseminate false or malicious content. While interest in AI-generated speech has increased, resources for evaluating naturalness at the phoneme level remain limited. In this work, we address this gap by presenting the Phoneme-Level DeepFake dataset (PhonemeDF), comprising parallel real and synthetic speech segmented at the phoneme level. Real speech samples are derived from a subset of LibriSpeech, while synthetic samples are generated using four TTS and three VC systems. For each system, phoneme-aligned TextGrid files are obtained using the Montreal Forced Aligner (MFA). We compute the Kullback-Leibler divergence (KLD) between real and synthetic phoneme distributions to quantify fidelity and establish a ranking based on similarity to natural speech. Our findings show a clear correlation between the KLD of real and synthetic phoneme distributions and the performance of classifiers trained to distinguish them, suggesting that KLD can serve as an indicator of the most discriminative phonemes for deepfake detection.
>
---
#### [new 032] NV-Bench: Benchmark of Nonverbal Vocalization Synthesis for Expressive Text-to-Speech Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出NV-Bench，用于评估非语言发声合成的文本到语音系统。解决缺乏标准化评估的问题，通过双维度指标衡量控制性和音质真实性。**

- **链接: [https://arxiv.org/pdf/2603.15352](https://arxiv.org/pdf/2603.15352)**

> **作者:** Qinke Ni; Huan Liao; Dekun Chen; Yuxiang Wang; Zhizheng Wu
>
> **摘要:** While recent text-to-speech (TTS) systems increasingly integrate nonverbal vocalizations (NVs), their evaluations lack standardized metrics and reliable ground-truth references. To bridge this gap, we propose NV-Bench, the first benchmark grounded in a functional taxonomy that treats NVs as communicative acts rather than acoustic artifacts. NV-Bench comprises 1,651 multi-lingual, in-the-wild utterances with paired human reference audio, balanced across 14 NV categories. We introduce a dual-dimensional evaluation protocol: (1) Instruction Alignment, utilizing the proposed paralinguistic character error rate (PCER) to assess controllability, (2) Acoustic Fidelity, measuring the distributional gap to real recordings to assess acoustic realism. We evaluate diverse TTS models and develop two baselines. Experimental results demonstrate a strong correlation between our objective metrics and human perception, establishing NV-Bench as a standardized evaluation framework.
>
---
#### [new 033] Music Genre Classification: A Comparative Analysis of Classical Machine Learning and Deep Learning Approaches
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐流派分类任务，旨在解决非西方音乐传统（如尼泊尔音乐）的自动分类问题。研究构建了包含8种尼泊尔音乐类型的8000个音频片段数据集，并比较了经典机器学习与深度学习模型的效果。**

- **链接: [https://arxiv.org/pdf/2603.15440](https://arxiv.org/pdf/2603.15440)**

> **作者:** Sachin Prajuli; Abhishek Karna; OmPrakash Dhakl
>
> **备注:** 8 pages
>
> **摘要:** Automatic music genre classification is a long-standing challenge in Music Information Retrieval (MIR); work on non-Western music traditions remains scarce. Nepali music encompasses culturally rich and acoustically diverse genres--from the call-and-response duets of Lok Dohori to the rhythmic poetry of Deuda and the distinctive melodies of Tamang Selo--that have not been addressed by existing classification systems. In this paper, we construct a novel dataset of approximately 8,000 labeled 30-second audio clips spanning eight Nepali music genres and conduct a systematic comparison of nine classification models across two paradigms. Five classical machine learning classifiers (Logistic Regression, SVM, KNN, Random Forest, and XGBoost) are trained on 51 hand-crafted audio features extracted via Librosa, while four deep learning architectures (CNN, RNN, parallel CNN-RNN, and sequential CNN followed by RNN) operate on Mel spectrograms of dimension 640 x 128. Our experiments reveal that the sequential Convolutional Recurrent Neural Network (CRNN)--in which convolutional layers feed into an LSTM--achieves the highest accuracy of 84%, substantially outperforming both the best classical models (Logistic Regression and XGBoost, both at 71%) and all other deep architectures. We provide per-class precision, recall, F1-score, confusion matrices, and ROC analysis for every model, and offer a culturally grounded interpretation of misclassification patterns that reflects genuine overlaps in Nepal's musical traditions.
>
---
#### [new 034] WhispSynth: Scaling Multilingual Whisper Corpus through Real Data Curation and A Novel Pitch-free Generative Framework
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决 whispered speech 数据稀缺问题。通过构建多语言语料库 WhispSynth 和提出无音高生成框架，提升合成语音质量。**

- **链接: [https://arxiv.org/pdf/2603.14853](https://arxiv.org/pdf/2603.14853)**

> **作者:** Tianyi Tan; Jiaxin Ye; Yuanming Zhang; Xiaohuai Le; Xianjun Xia; Chuanzeng Huang; Jing Lu
>
> **备注:** Under Review
>
> **摘要:** Whisper generation is constrained by the difficulty of data collection. Because whispered speech has low acoustic amplitude, high-fidelity recording is challenging. In this paper, we introduce WhispSynth, a large-scale multilingual corpus constructed via a novel high-fidelity generative framework. Specifically, we propose a pipeline integrating Differentiable Digital Signal Processing (DDSP)-based pitch-free method with Text-to-Speech (TTS) models. This framework refines a comprehensive collection of resources, including our newly constructed WhispNJU dataset, into 118 hours of high-fidelity whispered speech from 479 speakers. Unlike standard synthetic or noisy real data, our data engine faithfully preserves source vocal timbre and linguistic content while ensuring acoustic consistency, providing a robust foundation for text-to-whisper research. Experimental results demonstrate that WhispSynth exhibits significantly higher quality than existing corpora. Moreover, our CosyWhisper, tuned with WhispSynth, achieves speech naturalness on par with ground-truth samples. The official implementation and related resources are available at this https URL.
>
---
#### [new 035] Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决语音对话中的模态和口语化评估问题。提出SDiaReward模型和ESDR-Bench基准，提升对话质量评价的准确性。**

- **链接: [https://arxiv.org/pdf/2603.14889](https://arxiv.org/pdf/2603.14889)**

> **作者:** Jingyu Lu; Yuhan Wang; Fan Zhuo; Xize Cheng; Changhao Pan; Xueyi Pu; Yifu Chen; Chenyuhao Wen; Tianle Liang; Zhou Zhao
>
> **摘要:** The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL.
>
---
#### [new 036] Causal Tracing of Audio-Text Fusion in Large Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于多模态融合研究，旨在揭示大音频语言模型如何整合音频与文本信息。通过因果追踪分析，识别了不同模型的融合策略及关键信息节点。**

- **链接: [https://arxiv.org/pdf/2603.13768](https://arxiv.org/pdf/2603.13768)**

> **作者:** Wei-Chih Chen; Chien-yu Huang; Hung-yi Lee
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Despite the strong performance of large audio language models (LALMs) in various tasks, exactly how and where they integrate acoustic features with textual context remains unclear. We adapt causal tracing to investigate the internal information flow of LALMs during audio comprehension. By conducting layer-wise and token-wise analyses across DeSTA, Qwen, and Voxtral, we evaluate the causal effects of individual hidden states. Layer-wise analysis identifies different fusion strategies, from progressive integration in DeSTA to abrupt late-stage fusion in Qwen. Token-wise analysis shows that the final sequence token acts as an informational bottleneck where the network decisively retrieves relevant information from the audio. We also observe an attention-like query mechanism at intermediate token positions that triggers the model to pull task-relevant audio context. These findings provide a clear characterization of when and where multi-modal integration occurs within LALMs.
>
---
#### [new 037] CodecMOS-Accent: A MOS Benchmark of Resynthesized and TTS Speech from Neural Codecs Across English Accents
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出CodecMOS-Accent数据集，用于评估神经音频编解码器和基于大语言模型的文本转语音系统在不同英语口音中的表现。任务是提升对口音语音合成的人类中心评价。**

- **链接: [https://arxiv.org/pdf/2603.14328](https://arxiv.org/pdf/2603.14328)**

> **作者:** Wen-Chin Huang; Nicholas Sanders; Erica Cooper
>
> **备注:** Preprint
>
> **摘要:** We present the CodecMOS-Accent dataset, a mean opinion score (MOS) benchmark designed to evaluate neural audio codec (NAC) models and the large language model (LLM)-based text-to-speech (TTS) models trained upon them, especially across non-standard speech like accented speech. The dataset comprises 4,000 codec resynthesis and TTS samples from 24 systems, featuring 32 speakers spanning ten accents. A large-scale subjective test was conducted to collect 19,600 annotations from 25 listeners across three dimensions: naturalness, speaker similarity, and accent similarity. This dataset does not only represent an up-to-date study of recent speech synthesis system performance but reveals insights including a tight relationship between speaker and accent similarity, the predictive power of objective metrics, and a perceptual bias when listeners share the same accent with the speaker. This dataset is expected to foster research on more human-centric evaluation for NAC and accented TTS.
>
---
#### [new 038] PARSA-Bench: A Comprehensive Persian Audio-Language Model Benchmark
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出PARSA-Bench，针对波斯语音频语言模型设计基准，解决波斯语独特文化音频理解问题，包含16项任务和8000个样本。**

- **链接: [https://arxiv.org/pdf/2603.14456](https://arxiv.org/pdf/2603.14456)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Mohammad Amini; Parmis Bathayan; Heshaam Faili; Azadeh Shakery
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Persian poses unique audio understanding challenges through its classical poetry, traditional music, and pervasive code-switching - none captured by existing benchmarks. We introduce PARSA-Bench (Persian Audio Reasoning and Speech Assessment Benchmark), the first benchmark for evaluating large audio-language models on Persian language and culture, comprising 16 tasks and over 8,000 samples across speech understanding, paralinguistic analysis, and cultural audio understanding. Ten tasks are newly introduced, including poetry meter and style detection, traditional Persian music understanding, and code-switching detection. Text-only baselines consistently outperform audio counterparts, suggesting models may not leverage audio-specific information beyond what transcription alone provides. Culturally-grounded tasks expose a qualitatively distinct failure mode: all models perform near random chance on vazn detection regardless of scale, suggesting prosodic perception remains beyond the reach of current models. The dataset is publicly available at this https URL
>
---
#### [new 039] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于视频配音任务，旨在解决语音与唇形同步及语音质量不足的问题。提出DiFlowDubber模型，通过跨模态对齐和离散流匹配实现高质量自动配音。**

- **链接: [https://arxiv.org/pdf/2603.14267](https://arxiv.org/pdf/2603.14267)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Jeongsoo Choi; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **备注:** Accepted at CVPR 2026 Findings
>
> **摘要:** Video dubbing has broad applications in filmmaking, multimedia creation, and assistive speech technology. Existing approaches either train directly on limited dubbing datasets or adopt a two-stage pipeline that adapts pre-trained text-to-speech (TTS) models, which often struggle to produce expressive prosody, rich acoustic characteristics, and precise synchronization. To address these issues, we propose DiFlowDubber with a novel two-stage training framework that effectively transfers knowledge from a pre-trained TTS model to video-driven dubbing, with a discrete flow matching generative backbone. Specifically, we design a FaPro module that captures global prosody and stylistic cues from facial expressions and leverages this information to guide the modeling of subsequent speech attributes. To ensure precise speech-lip synchronization, we introduce a Synchronizer module that bridges the modality gap among text, video, and speech, thereby improving cross-modal alignment and generating speech that is temporally synchronized with lip movements. Experiments on two primary benchmark datasets demonstrate that DiFlowDubber outperforms previous methods across multiple metrics.
>
---
#### [new 040] A Hierarchical End-of-Turn Model with Primary Speaker Segmentation for Real-Time Conversational AI
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于实时对话AI任务，解决双人对话中的自然轮换问题。通过结合主说话人分割与分层EOT检测，提升对话状态预测的准确性与实时性。**

- **链接: [https://arxiv.org/pdf/2603.13379](https://arxiv.org/pdf/2603.13379)**

> **作者:** Karim Helwani; Hoang Do; James Luan; Sriram Srinivasan
>
> **备注:** Accepted for presentation at the IEEE Conference on Artificial Intelligence
>
> **摘要:** We present a real-time front-end for voice-based conversational AI to enable natural turn-taking in two-speaker scenarios by combining primary speaker segmentation with hierarchical End-of-Turn (EOT) detection. To operate robustly in multi-speaker environments, the system continuously identifies and tracks the primary user, ensuring that downstream EOT decisions are not confounded by background conversations. The tracked activity segments are fed to a hierarchical, causal EOT model that predicts the immediate conversational state by independently analyzing per-speaker speech features from both the primary speaker and the bot. Simultaneously, the model anticipates near-future states ($t{+}10/20/30$\,ms) through probabilistic predictions that are aware of the conversation partner's speech. Task-specific knowledge distillation compresses wav2vec~2.0 representations (768\,D) into a compact MFCC-based student (32\,D) for efficient deployment. The system achieves 82\% multi-class frame-level F1 and 70.6\% F1 on Backchannel detection, with 69.3\% F1 on a binary Final vs.\ Others task. On an end-to-end turn-detection benchmark, our model reaches 87.7\% recall vs.\ 58.9\% for Smart Turn~v3 while keeping a median detection latency of 36\,ms versus 800--1300\,ms. Despite using only 1.14\,M parameters, the proposed model matches or exceeds transformer-based baselines while substantially reducing latency and memory footprint, making it suitable for edge deployment.
>
---
#### [new 041] Semi-Automatic Flute Robot and Its Acoustic Sensing
- **分类: cs.HC; cs.RO; cs.SD**

- **简介: 该论文属于音乐机器人任务，旨在解决 flute 自动演奏与音区控制问题。设计了一种半自动长笛机器人，实现自动指法和低音区气流偏移辅助，提升演奏准确性与表现力。**

- **链接: [https://arxiv.org/pdf/2603.14180](https://arxiv.org/pdf/2603.14180)**

> **作者:** Hikari Kuriyama; Hiroaki Sonoda; Kouki Tomiyoshi; Gou Koutaki
>
> **备注:** This paper was submitted to a journal and received thorough reviews with high marks from the experts. Despite addressing three rounds of major revisions, it was ultimately rejected due to an unreasonable reviewer. We are uploading it here as a preprint
>
> **摘要:** Flute performance requires mastery of complex fingering combinations and register-dependent embouchure control, particularly jet offset adjustment for low-register production. Existing haptic and semi-automated systems do not address both aspects simultaneously through mechanical actuation. To our knowledge, no prior system fully automates fingering while mechanically assisting low-register tone production without requiring embouchure control. We developed a semi-automatic flute robot with an automatic fingering mechanism: fourteen servo motors actuate all keys via wire-based and rack-and-pinion drives in response to MIDI input, enabling performers to produce complete musical pieces through airflow alone. A jet offset assist mechanism rotates the head joint by a calibrated $22^\circ$ during low-register passages, shifting the jet offset toward a low-register configuration without modifying the instrument or embouchure. Fundamental frequency estimation confirmed correct pitch production across the chromatic range (C4--C7) and during musical performance. All key and lever movements were completed within 77.50~ms, corresponding to tempo capacity exceeding standard requirements. Harmonic analysis ($\Delta\mathrm{SPL} = \mathrm{SPL}_2 - \mathrm{SPL}_3$) showed a consistent increase in $\Delta$SPL for all low-register notes when activated, consistent with the intended jet offset shift. Head joint rotation completed within 40.00~ms. These results demonstrate mechanical feasibility of integrating automated fingering and register-dependent jet offset assistance under controlled conditions.
>
---
#### [new 042] LightBeam: An Accurate and Memory-Efficient CTC Decoder for Speech Neuroprostheses
- **分类: cs.HC; cs.SD**

- **简介: 该论文属于语音神经假体任务，旨在解决CTC解码器内存消耗大的问题。提出LightBeam，采用非WFST结构，仅需10GB内存，提升效率并保持高性能。**

- **链接: [https://arxiv.org/pdf/2603.14002](https://arxiv.org/pdf/2603.14002)**

> **作者:** Ebrahim Feghhi; Junlin Hu; Nima Hadidi; Jonathan C. Kao
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** A promising pathway for restoring communication in patients with dysarthria and anarthria is speech neuroprostheses, which directly decode speech from cortical neural activity. Two benchmarks, Brain-to-Text '24 and '25, released intracranial recordings from patients with dysarthria along with a baseline algorithm trained with Connectionist Temporal Classification (CTC). Despite significant innovation on these benchmarks, all leading published prior work relies on a WFST-based CTC decoder that requires ${\sim}$320 GB of RAM. These memory requirements limit accessibility for both patients and researchers. Here, we propose LightBeam, a non-WFST based CTC decoder that requires only ${\sim}$10 GB of RAM and achieves state-of-the-art performance on both benchmarks. LightBeam achieves this by integrating an LLM into the beam-search process via delayed fusion, obviating the prior need for using a large N-gram LM. LightBeam is implemented in Python and is open-source.
>
---
#### [new 043] ReactMotion: Generating Reactive Listener Motions from Speaker Utterance
- **分类: cs.CV; cs.AI; cs.HC; cs.MM; cs.SD**

- **简介: 该论文提出ReactMotion任务，旨在生成与说话内容相适应的自然听众动作。通过构建数据集和提出统一生成框架，解决非确定性人类反应建模难题。**

- **链接: [https://arxiv.org/pdf/2603.15083](https://arxiv.org/pdf/2603.15083)**

> **作者:** Cheng Luo; Bizhu Wu; Bing Li; Jianfeng Ren; Ruibin Bai; Rong Qu; Linlin Shen; Bernard Ghanem
>
> **备注:** 42 pages, 11 tables, 8 figures
>
> **摘要:** In this paper, we introduce a new task, Reactive Listener Motion Generation from Speaker Utterance, which aims to generate naturalistic listener body motions that appropriately respond to a speaker's utterance. However, modeling such nonverbal listener behaviors remains underexplored and challenging due to the inherently non-deterministic nature of human reactions. To facilitate this task, we present ReactMotionNet, a large-scale dataset that pairs speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness. This dataset design explicitly captures the one-to-many nature of listener behavior and provides supervision beyond a single ground-truth motion. Building on this dataset design, we develop preference-oriented evaluation protocols tailored to evaluate reactive appropriateness, where conventional motion metrics focusing on input-motion alignment ignore. We further propose ReactMotion, a unified generative framework that jointly models text, audio, emotion, and motion, and is trained with preference-based objectives to encourage both appropriate and diverse listener responses. Extensive experiments show that ReactMotion outperforms retrieval baselines and cascaded LLM-based pipelines, generating more natural, diverse, and appropriate listener motions.
>
---
#### [new 044] Sirens' Whisper: Inaudible Near-Ultrasonic Jailbreaks of Speech-Driven LLMs
- **分类: cs.CR; cs.AI; cs.SD**

- **简介: 该论文属于安全领域，针对语音驱动的大型语言模型提出隐蔽攻击方法，解决通过声学通道进行非法指令注入的问题。工作包括构建隐蔽音频通道和设计有效攻击策略。**

- **链接: [https://arxiv.org/pdf/2603.13847](https://arxiv.org/pdf/2603.13847)**

> **作者:** Zijian Ling; Pingyi Hu; Xiuyong Gao; Xiaojing Ma; Man Zhou; Jun Feng; Songfeng Lu; Dongmei Zhang; Bin Benjamin Zhu
>
> **备注:** USENIX Security'26 Camera-ready
>
> **摘要:** Speech-driven large language models (LLMs) are increasingly accessed through speech interfaces, introducing new security risks via open acoustic channels. We present Sirens' Whisper (SWhisper), the first practical framework for covert prompt-based attacks against speech-driven LLMs under realistic black-box conditions using commodity hardware. SWhisper enables robust, inaudible delivery of arbitrary target baseband audio-including long and structured prompts-on commodity devices by encoding it into near-ultrasound waveforms that demodulate faithfully after acoustic transmission and microphone nonlinearity. This is achieved through a simple yet effective approach to modeling nonlinear channel characteristics across devices and environments, combined with lightweight channel-inversion pre-compensation. Building on this high-fidelity covert channel, we design a voice-aware jailbreak generation method that ensures intelligibility, brevity, and transferability under speech-driven interfaces. Experiments across both commercial and open-source speech-driven LLMs demonstrate strong black-box effectiveness. On commercial models, SWhisper achieves up to 0.94 non-refusal (NR) and 0.925 specific-convincing (SC). A controlled user study further shows that the injected jailbreak audio is perceptually indistinguishable from background-only playback for human listeners. Although jailbreaks serve as a case study, the underlying covert acoustic channel enables a broader class of high-fidelity prompt-injection and commandexecution attacks.
>
---
#### [new 045] Multimodal Emotion Regression with Multi-Objective Optimization and VAD-Aware Audio Modeling for the 10th ABAW EMI Track
- **分类: cs.AI; cs.SD**

- **简介: 该论文针对情感模仿强度估计任务，解决六维情绪连续值预测问题。通过多模态特征融合与多目标优化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.13760](https://arxiv.org/pdf/2603.13760)**

> **作者:** Jiawen Huang; Chenxi Huang; Zhuofan Wen; Hailiang Yao; Shun Chen; Longjiang Yang; Cong Yu; Fengyu Zhang; Ran Liu; Bin Liu
>
> **摘要:** We participated in the 10th ABAW Challenge, focusing on the Emotional Mimicry Intensity (EMI) Estimation track on the Hume-Vidmimic2 dataset. This task aims to predict six continuous emotion dimensions: Admiration, Amusement, Determination, Empathic Pain, Excitement, and Joy. Through systematic multimodal exploration of pretrained high-level features, we found that, under our pretrained feature setting, direct feature concatenation outperformed the more complex fusion strategies we tested. This empirical finding motivated us to design a systematic approach built upon three core principles: (i) preserving modality-specific attributes through feature-level concatenation; (ii) improving training stability and metric alignment via multi-objective optimization; and (iii) enriching acoustic representations with a VAD-inspired latent prior. Our final framework integrates concatenation-based multimodal fusion, a shared six-dimensional regression head, multi-objective optimization with MSE, Pearson-correlation, and auxiliary branch supervision, EMA for parameter stabilization, and a VAD-inspired latent prior for the acoustic branch. On the official validation set, the proposed scheme achieved our best mean Pearson Correlation Coefficient of 0.478567.
>
---
#### [new 046] Distributed Acoustic Sensing for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于交通监测任务，旨在解决DAS数据中交通事件识别问题。通过引入注意力机制提升RNN模型的性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.13903](https://arxiv.org/pdf/2603.13903)**

> **作者:** Izhan Fakhruzi; Manuel Titos; Carmen Benítez; Luz García
>
> **摘要:** Effective urban traffic monitoring is essential for improving mobility, enhancing safety, and supporting sustainable cities. Distributed Acoustic Sensing (DAS) enables large-scale traffic observation by transforming existing fiber-optic infrastructure into dense arrays of vibration sensors. However, modeling the high-resolution spatio-temporal structure of DAS data for reliable traffic event recognition remains challenging. This study presents a real-world DAS-based traffic monitoring experiment conducted in Granada, Spain, where vehicles cross a fiber deployed perpendicular to the roadway. Recurrent neural networks (RNNs) are employed to model intra- and inter-event temporal dependencies. Spatial and temporal attention mechanisms are systematically integrated within the RNN architecture to analyze their impact on recognition performance, parameter efficiency, and interpretability. Results show that an appropriate and complementary placement of attention modules improves the balance between accuracy and model complexity. Attention heatmaps provide physically meaningful interpretations of classification decisions by highlighting informative spatial locations and temporal segments. Furthermore, the proposed SA-bi-TA configuration demonstrates spatial transferability, successfully recognizing traffic events at sensing locations different from those used during training, with only moderate performance degradation. These findings support the development of scalable and interpretable DAS-based traffic monitoring systems capable of operating under heterogeneous urban sensing conditions.
>
---
## 更新

#### [replaced 001] Covo-Audio Technical Report
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Covo-Audio，一个7B参数的端到端音频大语言模型，解决音频理解和生成任务，通过预训练和微调实现高质量对话与语音交互。**

- **链接: [https://arxiv.org/pdf/2602.09823](https://arxiv.org/pdf/2602.09823)**

> **作者:** Wenfu Wang; Chenxing Li; Liqiang Zhang; Yiyang Zhao; Yuxiang Zou; Hanzhao Li; Mingyu Cui; Hao Zhang; Kun Wei; Le Xu; Zikang Huang; Jiajun Xu; Jiliang Hu; Xiang He; Zeyu Xie; Jiawen Kang; Youjun Chen; Meng Yu; Dong Yu; Rilin Chen; Linlin Di; Shulin Feng; Na Hu; Yang Liu; Bang Wang; Shan Yang
>
> **备注:** Technical Report
>
> **摘要:** In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs.
>
---
#### [replaced 002] Do You Hear What I Mean? Quantifying the Instruction-Perception Gap in Instruction-Guided Expressive Text-To-Speech Systems
- **分类: eess.AS**

- **简介: 该论文属于文本到语音生成任务，旨在解决指令与感知之间的差异问题。通过分析ITTS系统在情感和语气控制上的表现，提出E-VOC数据集，揭示模型在细粒度控制上的不足。**

- **链接: [https://arxiv.org/pdf/2509.13989](https://arxiv.org/pdf/2509.13989)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Tzu-Chieh Wei; Kuan-Yu Chen; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Instruction-guided text-to-speech (ITTS) enables users to control speech generation through natural language prompts, offering a more intuitive interface than traditional TTS. However, the alignment between user style instructions and listener perception remains largely unexplored. This work first presents a perceptual analysis of ITTS controllability across two expressive dimensions (adverbs of degree and graded emotion intensity) and collects human ratings on speaker age and word-level emphasis attributes. To comprehensively reveal the instruction-perception gap, we provide a data collection with large-scale human evaluations, named Expressive VOice Control (E-VOC) corpus. Furthermore, we reveal that (1) gpt-4o-mini-tts is the most reliable ITTS model with great alignment between instruction and generated utterances across acoustic dimensions. (2) The 5 analyzed ITTS systems tend to generate Adult voices even when the instructions ask to use child or Elderly voices. (3) Fine-grained control remains a major challenge, indicating that most ITTS systems have substantial room for improvement in interpreting slightly different attribute instructions.
>
---
#### [replaced 003] SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型知识编辑任务，旨在解决 auditory attribute knowledge 编辑问题。提出 SAKE 基准，评估多种方法在可靠性、泛化性等方面的表现。**

- **链接: [https://arxiv.org/pdf/2510.16917](https://arxiv.org/pdf/2510.16917)**

> **作者:** Chih-Kai Yang; Yen-Ting Piao; Tzu-Wen Hsu; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Work in progress. Resources: this https URL
>
> **摘要:** Knowledge editing enables targeted updates without retraining, but prior work focuses on textual or visual facts, leaving abstract auditory perceptual knowledge underexplored. We introduce SAKE, the first benchmark for editing perceptual auditory attribute knowledge in large audio-language models (LALMs), which requires modifying acoustic generalization rather than isolated facts. We evaluate eight diverse editing methods on three LALMs across reliability, generality, locality, and portability, under single and sequential edits. Results show that most methods enforce edits reliably but struggle with auditory generalization, intra-attribute locality, and multimodal knowledge propagation, and often exhibit forgetting or degeneration in sequential editing. Additionally, fine-tuning the modality connector emerges as a more robust and balanced baseline compared with directly editing the LLM backbones. SAKE reveals key limitations of current methods and provides a foundation for developing auditory-specific LALM editing techniques.
>
---
#### [replaced 004] SyncSpeech: Efficient and Low-Latency Text-to-Speech based on Temporal Masked Transformer
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决AR模型效率低和NAR模型延迟高的问题。提出SyncSpeech模型，结合两者优势，提升生成效率与降低延迟。**

- **链接: [https://arxiv.org/pdf/2502.11094](https://arxiv.org/pdf/2502.11094)**

> **作者:** Zhengyan Sheng; Zhihao Du; Shiliang Zhang; Zhijie Yan; Liping Chen
>
> **摘要:** Current text-to-speech (TTS) models face a persistent limitation: autoregressive (AR) models suffer from low generation efficiency, while modern non-autoregressive (NAR) models experience high latency due to their unordered temporal nature. To bridge this divide, we introduce SyncSpeech, an efficient and low-latency TTS model based on the proposed Temporal Mask Transformer (TMT) paradigm. TMT synergistically unifies the temporally ordered generation of AR models with the parallel decoding efficiency of NAR models. TMT is realized through a meticulously designed sequence construction rule, a corresponding training objective, and a specialized hybrid attention mask. Furthermore, with the primary aim of enhancing training efficiency, a high-probability masking strategy is introduced, which also leads to a significant improvement in overall model performance. During inference, SyncSpeech achieves high efficiency by decoding all speech tokens corresponding to each newly arrived text token in a single step, and low latency by beginning to generate speech immediately upon receiving the second text token from the streaming input. Evaluations show that SyncSpeech maintains speech quality comparable to the modern AR TTS model, while achieving a 5.8-fold reduction in first-packet latency and an 8.8-fold improvement in real-time factor. Speech samples are available at this https URL}{this https URL.
>
---
#### [replaced 005] Variational Low-Rank Adaptation for Personalized Impaired Speech Recognition
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决非典型语音识别难题。针对数据稀缺和标注困难，提出一种基于变分低秩适应的个性化方法，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2509.20397](https://arxiv.org/pdf/2509.20397)**

> **作者:** Niclas Pokel; Pehuén Moure; Roman Boehringer; Shih-Chii Liu; Yingqiang Gao
>
> **摘要:** Speech impairments resulting from congenital disorders, such as cerebral palsy, down syndrome, or apert syndrome, as well as acquired brain injuries due to stroke, traumatic accidents, or tumors, present major challenges to automatic speech recognition (ASR) systems. Despite recent advancements, state-of-the-art ASR models like Whisper still struggle with non-normative speech due to limited training data availability and high acoustic variability. Moreover, collecting and annotating non-normative speech is burdensome: speaking is effortful for many affected individuals, while laborious annotation often requires caregivers familiar with the speaker. This work introduces a novel ASR personalization method based on Bayesian Low-rank Adaptation for data-efficient fine-tuning. We validate our method on the English UA-Speech dataset and a newly collected German speech dataset, BF-Sprache, from a child with structural speech impairment. The dataset and approach are designed to reflect the challenges of low-resource settings that include individuals with speech impairments. Our method significantly improves ASR accuracy for impaired speech while maintaining data and annotation efficiency, offering a practical path toward inclusive ASR.
>
---
#### [replaced 006] Room Impulse Response Completion Using Signal-Prediction Diffusion Models Conditioned on Simulated Early Reflections
- **分类: eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决RIR生成中缺乏真实感的问题。通过扩散模型结合ISMS模拟的早期反射，生成更真实的RIR。**

- **链接: [https://arxiv.org/pdf/2603.12442](https://arxiv.org/pdf/2603.12442)**

> **作者:** Zeyu Xu; Andreas Brendel; Albert G. Prinn; Emanuël A. P. Habets
>
> **备注:** The following article has been submitted for review to Interspeech 2026
>
> **摘要:** Room impulse responses (RIRs) are fundamental to audio data augmentation, acoustic signal processing, and immersive audio rendering. While geometric simulators such as the image source method (ISM) can efficiently generate early reflections, they lack the realism of measured RIRs due to missing acoustic wave effects. We propose a diffusion-based RIR completion method using signal-prediction conditioned on ISM-simulated direct-path and early reflections. Unlike state-of-the-art methods, our approach imposes no fixed duration constraint on the input early reflections. We further incorporate classifier-free guidance to steer generation toward a target distribution learned from physically realistic RIRs simulated with the Treble SDK. Objective evaluation demonstrates that the proposed method outperforms a state-of-the-art baseline in early RIR completion and energy decay curve reconstruction.
>
---
#### [replaced 007] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决现有模型解释性差、推理能力弱的问题。通过强化学习和韵律增强，提出EmotionThinker模型，提升情感预测准确性和解释质量。**

- **链接: [https://arxiv.org/pdf/2601.15668](https://arxiv.org/pdf/2601.15668)**

> **作者:** Dingdong Wang; Shujie Liu; Tianhua Zhang; Youjun Chen; Jinyu Li; Helen Meng
>
> **备注:** ICLR 2026 (Oral). Project page: this https URL
>
> **摘要:** Emotional information in speech plays a unique role in multimodal perception. However, current Speech Large Language Models (SpeechLLMs), similar to conventional speech emotion recognition (SER) systems, still treat emotion understanding as a simple classification problem. This provides limited interpretability of predictions, while leaving the LLMs' expressive and reasoning capabilities underutilized. In this work, we take the first step to reformulate SER as a deep reasoning problem through reinforcement learning (RL). We propose EmotionThinker, which is designed to generate accurate emotion predictions with interpretable explanations grounded in fine-grained acoustic cues. To achieve this, we first construct EmotionCoT-35K, an emotional reasoning dataset with Chain-of-Thought annotations and detailed captions. Second, we observe that current SpeechLLMs exhibit weak prosody perception, whereas prosodic cues constitute fundamental signals for interpreting emotions. To address this, we develop the prosody-enhanced foundation model EmotionThinker-Base, and demonstrate that prosody enhancement improves emotion understanding. Third, we introduce Group-Relative-Policy-Optimization with Progressive-Trust-aware-Reasoning-Reward (GRPO-PTR) for RL. Different from standard GRPO, which relies only on rule-based outcome rewards, GRPO-PTR progressively introduces reasoning reward, dynamically adjusts it with a trustworthiness weight reflecting the alignment between reasoning and outcome, and evaluates the overall reasoning quality with a reward model based on multi-dimensional criteria. EmotionThinker outperforms previous state-of-the-art evaluation models both in emotion accuracy and explanation quality, advancing SER toward interpretable multimodal reasoning. Project page: this https URL
>
---
#### [replaced 008] AWARE: Audio Watermarking with Adversarial Resistance to Edits
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 该论文提出AWARE，一种抗编辑的音频水印方法，解决传统水印系统对模拟攻击依赖的问题。通过对抗优化和时间无关检测，提升鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2510.17512](https://arxiv.org/pdf/2510.17512)**

> **作者:** Kosta Pavlović; Lazar Stanarević; Petar Nedić; Elena Nešović Slavko Kovačević; Igor Djurović
>
> **摘要:** Prevailing practice in learning-based audio watermarking is to pursue robustness by expanding the set of simulated distortions during training. However, such surrogates are narrow and prone to overfitting. This paper presents AWARE (Audio Watermarking with Adversarial Resistance to Edits), an alternative approach that avoids reliance on attack-simulation stacks and handcrafted differentiable distortions. Embedding is obtained through adversarial optimization in the time-frequency domain under a level-proportional perceptual budget. Detection employs a time-order-agnostic detector with a Bitwise Readout Head (BRH) that aggregates temporal evidence into one score per watermark bit, enabling reliable watermark decoding even under desynchronization and temporal cuts. Empirically, AWARE attains high audio quality and speech intelligibility (PESQ/STOI) and consistently low BER across various audio edits, often surpassing representative state-of-the-art learning-based systems.
>
---
#### [replaced 009] Data-Efficient ASR Personalization for Non-Normative Speech Using an Uncertainty-Based Phoneme Difficulty Score for Guided Sampling
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，解决非标准语音识别准确率低的问题。通过不确定性引导的采样策略提升模型个性化效果。**

- **链接: [https://arxiv.org/pdf/2509.20396](https://arxiv.org/pdf/2509.20396)**

> **作者:** Niclas Pokel; Pehuén Moure; Roman Böhringer; Yingqiang Gao
>
> **摘要:** ASR systems struggle with non-normative speech due to high acoustic variability and data scarcity. We propose a data-efficient method using phoneme-level uncertainty to guide fine-tuning for personalization. Instead of computationally expensive ensembles, we leverage Variational Low-Rank Adaptation (VI LoRA) to estimate epistemic uncertainty in foundation models. These estimates form a composite Phoneme Difficulty Score (PhDScore) that drives a targeted oversampling strategy. Evaluated on English and German datasets, including a longitudinal analysis against two clinical reports taken one year apart, we demonstrate that: (1) VI LoRA-based uncertainty aligns better with expert clinical assessments than standard entropy; (2) PhDScore captures stable, persistent articulatory difficulties; and (3) uncertainty-guided sampling significantly improves ASR accuracy for impaired speech.
>
---
#### [replaced 010] Time-Layer Adaptive Alignment for Speaker Similarity in Flow-Matching Based Zero-Shot TTS
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，解决零样本TTS中说话人相似性不足的问题。提出时间-层自适应对齐方法，提升说话人一致性。**

- **链接: [https://arxiv.org/pdf/2511.09995](https://arxiv.org/pdf/2511.09995)**

> **作者:** Haoyu Li; Mingyang Han; Yu Xi; Dongxiao Wang; Hankun Wang; Haoxiang Shi; Boyu Li; Jun Song; Bo Zheng; Shuai Wang
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Flow-Matching (FM)-based zero-shot text-to-speech (TTS) systems exhibit high-quality speech synthesis and robust generalization capabilities. However, the speaker representation ability of such systems remains underexplored, primarily due to the lack of explicit speaker-specific supervision in the FM framework. To this end, we conduct an empirical analysis of speaker information distribution and reveal its non-uniform allocation across time steps and network layers, underscoring the need for adaptive speaker alignment. Accordingly, we propose Time-Layer Adaptive Speaker Alignment (TLA-SA), a strategy that enhances speaker consistency by jointly leveraging temporal and hierarchical variations. Experimental results show that TLA-SA substantially improves speaker similarity over baseline systems on both research- and industrial-scale datasets and generalizes well across diverse model architectures, including decoder-only language model (LM)-based and free TTS systems. A demo is provided.
>
---
#### [replaced 011] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception
- **分类: cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于多模态细粒度感知任务，旨在解决OLMs在细节描述与幻觉之间的平衡问题。提出Omni-Detective数据生成方法和Omni-Captioner模型，设计Omni-Cloze评估基准。**

- **链接: [https://arxiv.org/pdf/2510.12720](https://arxiv.org/pdf/2510.12720)**

> **作者:** Ziyang Ma; Ruiyang Xu; Zhenghao Xing; Yunfei Chu; Yuxuan Wang; Jinzheng He; Jin Xu; Pheng-Ann Heng; Kai Yu; Junyang Lin; Eng Siong Chng; Xie Chen
>
> **备注:** Accepted by ICLR2026. Open Source at this https URL
>
> **摘要:** Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions.
>
---
#### [replaced 012] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出MMSU基准，用于评估语音语言理解与推理能力。解决现有模型在语音细粒度感知和复杂推理上的不足，涵盖47项任务，推动人机语音交互发展。**

- **链接: [https://arxiv.org/pdf/2506.04779](https://arxiv.org/pdf/2506.04779)**

> **作者:** Dingdong Wang; Junan Li; Jincenzi Wu; Dongchao Yang; Xueyuan Chen; Tianhua Zhang; Helen Meng
>
> **备注:** ICLR 2026. MMSU benchmark is available at this https URL. Project page this https URL
>
> **摘要:** Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at this https URL. Evaluation Code is available at this https URL.
>
---
#### [replaced 013] Speech Recognition on TV Series with Video-guided Post-ASR Correction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决电视节目复杂环境中ASR准确率低的问题。通过引入视频信息，提出VPC框架提升转录精度。**

- **链接: [https://arxiv.org/pdf/2506.07323](https://arxiv.org/pdf/2506.07323)**

> **作者:** Haoyuan Yang; Yue Zhang; Liqiang Jing; John H.L. Hansen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where multiple speakers, overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing approaches fail to explicitly leverage the rich temporal and contextual information available in the video. To address this limitation, we propose a Video-Guided Post-ASR Correction (VPC) framework that uses a Video-Large Multimodal Model (VLMM) to capture video context and refine ASR outputs. Evaluations on a TV-series benchmark show that our method consistently improves transcription accuracy in complex multimedia environments.
>
---
#### [replaced 014] The silence of the weights: a structural pruning strategy for attention-based audio signal architectures with second order metrics
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决注意力机制参数冗余问题。通过引入二阶度量进行通道剪枝，减少参数量并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2509.26207](https://arxiv.org/pdf/2509.26207)**

> **作者:** Andrea Diecidue; Carlo Alberto Barbano; Piero Fraternali; Mathieu Fontaine; Enzo Tartaglione
>
> **摘要:** Transformer-based models have become the state of the art across multiple domains, from natural language processing to machine listening, thanks to the attention mechanisms. However, the attention layers require a large number of parameters and high-end hardware for both training and inference. We propose a novel channel-pruning technique explicitly targeted at the attention mechanism, decoupling the pruning of each head and the four layers in the attention block: query, key, value, and output projection matrices, employing a second-order metric to score the network's parameters. We compare our technique against head-pruning strategies and magnitude-driven scoring metrics, investigating the effects of pruning on Audio Spectrogram Transformer (AST) and Whisper. Our results show that even after pruning 50\% of the parameters in the attention block, performance is largely preserved.
>
---
#### [replaced 015] A Language-Agnostic Hierarchical LoRA-MoE Architecture for CTC-based Multilingual ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决大模型在资源受限设备上的部署问题。提出HLoRA框架，实现高效、语言无关的端到端解码。**

- **链接: [https://arxiv.org/pdf/2601.00557](https://arxiv.org/pdf/2601.00557)**

> **作者:** Yuang Zheng; Dongxu Chen; Yuxiang Mei; Dongxing Xu; Jie Chen; Yanhua Long
>
> **备注:** 5 pages, submitted to IEEE Communications Letters
>
> **摘要:** Large-scale multilingual ASR (mASR) models such as Whisper achieve strong performance but incur high computational and latency costs, limiting their deployment on resource-constrained edge devices. In this study, we propose a lightweight and language-agnostic multilingual ASR system based on a CTC architecture with domain adaptation. Specifically, we introduce a Language-agnostic Hierarchical LoRA-MoE (HLoRA) framework integrated into an mHuBERT-CTC model, enabling end-to-end decoding via LID-posterior-driven LoRA routing. The hierarchical design consists of a multilingual shared LoRA for learning language-invariant acoustic representations and language-specific LoRA experts for modeling language-dependent characteristics. The proposed routing mechanism removes the need for prior language identity information or explicit language labels during inference, achieving true language-agnostic decoding. Experiments on MSR-86K and the MLC-SLM 2025 Challenge datasets demonstrate that HLoRA achieves comparable performance to two-stage inference approaches while reducing RTF by 11.7% and 8.2%, respectively, leading to improved decoding efficiency for low-resource mASR applications.
>
---
#### [replaced 016] VIBEVOICE-ASR Technical Report
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出VibeVoice-ASR，解决长音频中上下文碎片化和多说话人复杂性问题，实现单次处理60分钟音频的端到端语音理解。**

- **链接: [https://arxiv.org/pdf/2601.18184](https://arxiv.org/pdf/2601.18184)**

> **作者:** Zhiliang Peng; Jianwei Yu; Yaoyao Chang; Zilong Wang; Li Dong; Yingbo Hao; Yujie Tu; Chenyu Yang; Wenhui Wang; Songchen Xu; Yutao Sun; Hangbo Bao; Weijiang Xu; Yi Zhu; Zehua Wang; Ting Song; Yan Xia; Zewen Chi; Shaohan Huang; Liang Wang; Chuang Ding; Shuai Wang; Xie Chen; Furu Wei
>
> **摘要:** This report presents VibeVoice-ASR, a general-purpose speech understanding framework built upon VibeVoice, designed to address the persistent challenges of context fragmentation and multi-speaker complexity in long-form audio (e.g., meetings, podcasts) that remain despite recent advancements in short-form speech recognition. Unlike traditional pipelined approaches that rely on audio chunking, VibeVoice-ASRsupports single-pass processing for up to 60 minutes of audio. It unifies Automatic Speech Recognition, Speaker Diarization, and Timestamping into a single end-to-end generation task. In addition, VibeVoice-ASR supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Furthermore, we introduce a prompt-based context injection mechanism that allows users to supply customized conetxt, significantly improving accuracy on domain-specific terminology and polyphonic character disambiguation.
>
---
#### [replaced 017] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于自然语言处理任务，旨在解决多方讨论模拟中缺乏说话者属性数据的问题。通过构建带说话者信息的转录本，提升模型对真实公民对话的模拟效果。**

- **链接: [https://arxiv.org/pdf/2511.17813](https://arxiv.org/pdf/2511.17813)**

> **作者:** Scott Merrill; Shashank Srivastava
>
> **备注:** 8 pages (32 pages including appendix), 18 figures. Code and datasets are available at this https URL. Submitted to ACL 2026
>
> **摘要:** Large language models offer opportunities to simulate multi-party deliberation, but realistic modeling remains limited by a lack of speaker-attributed data. Transcripts produced via automatic speech recognition (ASR) assign anonymous speaker labels (e.g., Speaker_1), preventing models from capturing consistent human behavior. This work introduces a reproducible pipeline to transform public Zoom recordings into speaker-attributed transcripts with metadata like persona profiles and pragmatic action tags (e.g., [propose_motion]). We release three local government deliberation datasets: Appellate Court hearings, School Board meetings, and Municipal Council sessions. Fine-tuning LLMs to model specific participants using this "action-aware" data produces a 67% reduction in perplexity and nearly doubles classifier-based performance metrics for speaker fidelity and realism. Turing-style human evaluations show our simulations are often indistinguishable from real deliberations, providing a practical and scalable method for complex realistic civic simulations.
>
---
#### [replaced 018] LAMB: LLM-based Audio Captioning with Modality Gap Bridging via Cauchy-Schwarz Divergence
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频描述任务，旨在解决音频与文本嵌入空间对齐不足的问题。提出LAMB框架，通过跨模态对齐和双流适配器提升音频描述效果。**

- **链接: [https://arxiv.org/pdf/2601.04658](https://arxiv.org/pdf/2601.04658)**

> **作者:** Hyeongkeun Lee; Jongmin Choi; KiHyun Nam; Joon Son Chung
>
> **备注:** 5 pages, 2 figures; Accepted to ICASSP 2026
>
> **摘要:** Automated Audio Captioning aims to describe the semantic content of input audio. Recent works have employed large language models (LLMs) as a text decoder to leverage their reasoning capabilities. However, prior approaches that project audio features into the LLM embedding space without considering cross-modal alignment fail to fully utilize these capabilities. To address this, we propose LAMB, an LLM-based audio captioning framework that bridges the modality gap between audio embeddings and the LLM text embedding space. LAMB incorporates a Cross-Modal Aligner that minimizes Cauchy-Schwarz divergence while maximizing mutual information, yielding tighter alignment between audio and text at both global and token levels. We further design a Two-Stream Adapter that extracts semantically enriched audio embeddings, thereby delivering richer information to the Cross-Modal Aligner. Finally, leveraging the aligned audio embeddings, a proposed Token Guide directly computes scores within the LLM text embedding space to steer the output logits of generated captions. Experimental results confirm that our framework strengthens the reasoning capabilities of the LLM decoder, achieving state-of-the-art performance on AudioCaps.
>
---
#### [replaced 019] HD-PPT: Hierarchical Decoding of Content- and Prompt-Preference Tokens for Instruction-based TTS
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，解决指令控制不精准的问题。提出HD-PPT框架，通过分层解码和提取偏好令牌，提升指令遵循与自然度。**

- **链接: [https://arxiv.org/pdf/2509.19001](https://arxiv.org/pdf/2509.19001)**

> **作者:** Sihang Nie; Xiaofen Xing; Jingyuan Xing; Baiji Liu; Xiangmin Xu
>
> **备注:** 5 pages, 2 figures, 3 tables; Accepted to ICASSP2026(Oral)
>
> **摘要:** Large Language Model (LLM)-based Text-to-Speech (TTS) models have already reached a high degree of naturalness. However, the precision control of TTS inference is still challenging. Although instruction-based Text-to-Speech (Instruct-TTS) models are proposed, these models still lack fine-grained control due to the modality gap between single-level text instructions and multilevel speech tokens. To address this limitation, we propose HD-PPT, a framework that transforms speech synthesis into a structured, hierarchical task. To enable fine-grained control, we introduce a novel speech codec to extract distinct prompt-preference and content-preference tokens from the complex speech tokens, supervised by automatic speech recognition (ASR) and cross-lingual audio-text pre-training (CLAP) objectives. To bridge the modality gap of these tokens, we propose a hierarchical decoding strategy, where the LLM generates tokens in a structured order: first semantic, then fine-grained style, and finally complete acoustic representation. Extensive experiments demonstrate that this hierarchical paradigm significantly improves instruction adherence and achieves state-of-the-art naturalness, validating our approach for precise and controllable speech synthesis. Audio samples are available at this https URL.
>
---
#### [replaced 020] Latent-Mark: An Audio Watermark Robust to Neural Resynthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频水印任务，解决神经重合成攻击下水印易被破坏的问题。通过将水印嵌入编码器不变的潜在空间，实现鲁棒性与不可感知性的平衡。**

- **链接: [https://arxiv.org/pdf/2603.05310](https://arxiv.org/pdf/2603.05310)**

> **作者:** Yen-Shan Chen; Shih-Yu Lai; Ying-Jung Tsou; Yi-Cheng Lin; Bing-Yu Chen; Yun-Nung Chen; Hung-yi Lee; Shang-Tse Chen
>
> **摘要:** While existing audio watermarking techniques have achieved strong robustness against traditional digital signal processing (DSP) attacks, they remain vulnerable to neural resynthesis. This occurs because modern neural audio codecs act as semantic filters and discard the imperceptible waveform variations used in prior watermarking methods. To address this limitation, we propose Latent-Mark, the first zero-bit audio watermarking framework designed to survive semantic compression. Our key insight is that robustness to the encode-decode process requires embedding the watermark within the codec's invariant latent space. We achieve this by optimizing the audio waveform to induce a detectable directional shift in its encoded latent representation, while constraining perturbations to align with the natural audio manifold to ensure imperceptibility. To prevent overfitting to a single codec's quantization rules, we introduce Cross-Codec Optimization, jointly optimizing the waveform across multiple surrogate codecs to target shared latent invariants. Extensive evaluations demonstrate robust zero-shot transferability to unseen neural codecs, achieving state-of-the-art resilience against traditional DSP attacks while preserving perceptual imperceptibility. Our work inspires future research into universal watermarking frameworks capable of maintaining integrity across increasingly complex and diverse generative distortions.
>
---
#### [replaced 021] Nested Music Transformer: Sequentially Decoding Compound Tokens in Symbolic Music and Audio Generation
- **分类: cs.SD; cs.IR; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决复合标记解码效率与依赖关系捕捉问题。提出Nested Music Transformer架构，通过两级解码提升性能。**

- **链接: [https://arxiv.org/pdf/2408.01180](https://arxiv.org/pdf/2408.01180)**

> **作者:** HaeJun Yoo; Hao-Wen Dong; Jongmin Jung; Dasaem Jeong
>
> **备注:** Accepted at 25th International Society for Music Information Retrieval Conference (ISMIR 2024)
>
> **摘要:** Representing symbolic music with compound tokens, where each token consists of several different sub-tokens representing a distinct musical feature or attribute, offers the advantage of reducing sequence length. While previous research has validated the efficacy of compound tokens in music sequence modeling, predicting all sub-tokens simultaneously can lead to suboptimal results as it may not fully capture the interdependencies between them. We introduce the Nested Music Transformer (NMT), an architecture tailored for decoding compound tokens autoregressively, similar to processing flattened tokens, but with low memory usage. The NMT consists of two transformers: the main decoder that models a sequence of compound tokens and the sub-decoder for modeling sub-tokens of each compound token. The experiment results showed that applying the NMT to compound tokens can enhance the performance in terms of better perplexity in processing various symbolic music datasets and discrete audio tokens from the MAESTRO dataset.
>
---
#### [replaced 022] DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音匿名化攻击任务，旨在检测匿名化语音中的说话人信息。提出双流攻击模型，通过三阶段训练提升攻击效果，有效识别匿名语音中的说话人特征。**

- **链接: [https://arxiv.org/pdf/2603.12840](https://arxiv.org/pdf/2603.12840)**

> **作者:** Ridwan Arefeen; Xiaoxiao Miao; Rong Tong; Aik Beng Ng; Simon See; Timothy Liu
>
> **摘要:** Voice anonymization masks vocal traits while preserving linguistic content, which may still leak speaker-specific patterns. To assess and strengthen privacy evaluation, we propose a dual-stream attacker that fuses spectral and self-supervised learning features via parallel encoders with a three-stage training strategy. Stage I establishes foundational speaker-discriminative representations. Stage II leverages the shared identity-transformation characteristics of voice conversion and anonymization, exposing the model to diverse converted speech to build cross-system robustness. Stage III provides lightweight adaptation to target anonymized data. Results on the VoicePrivacy Attacker Challenge (VPAC) dataset demonstrate that Stage II is the primary driver of generalization, enabling strong attacking performance on unseen anonymization datasets. With Stage III, fine-tuning on only 10\% of the target anonymization dataset surpasses current state-of-the-art attackers in terms of EER.
>
---
#### [replaced 023] SloPal: A 60-Million-Word Slovak Parliamentary Corpus with Aligned Speech and Fine-Tuned ASR Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出SloPal，一个大规模斯洛伐克议会语料库，解决低资源语言ASR问题。构建了对齐语音数据并优化Whisper模型，显著提升识别效果。**

- **链接: [https://arxiv.org/pdf/2509.19270](https://arxiv.org/pdf/2509.19270)**

> **作者:** Erik Božík; Marek Šuppa
>
> **备注:** LREC 2026
>
> **摘要:** Slovak remains a low-resource language for automatic speech recognition (ASR), with fewer than 100 hours of publicly available training data. We present SloPal, a comprehensive Slovak parliamentary corpus comprising 330,000 speaker-segmented transcripts (66 million words, 220 million tokens) spanning 2001--2024, with rich metadata including speaker names, roles, and session information. From this collection, we derive SloPalSpeech, a 2,806-hour aligned speech dataset with segments up to 30 seconds, constructed using a language-agnostic anchor-based alignment pipeline and optimized for Whisper-based ASR training. Fine-tuning Whisper on SloPalSpeech reduces Word Error Rate (WER) by up to 70\%, with the fine-tuned small model (244M parameters) approaching base large-v3 (1.5B parameters) performance at 6$\times$ fewer parameters. We publicly release the SloPal text corpus, SloPalSpeech aligned audio, and four fine-tuned Whisper models at this https URL, providing the most comprehensive open Slovak parliamentary language resource to date.
>
---
#### [replaced 024] SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决开放源代码歌唱语音合成系统的鲁棒性和零样本泛化问题。提出SoulX-Singer系统，支持多语言和灵活控制，提升实际应用性能。**

- **链接: [https://arxiv.org/pdf/2602.07803](https://arxiv.org/pdf/2602.07803)**

> **作者:** Jiale Qian; Hao Meng; Tian Zheng; Pengcheng Zhu; Haopeng Lin; Yuhang Dai; Hanke Xie; Wenxiao Cao; Ruixuan Shang; Jun Wu; Hongmei Liu; Hanlin Wen; Jian Zhao; Zhonglin Jiang; Yong Chen; Shunshun Yin; Ming Tao; Jianguo Wei; Lei Xie; Xinsheng Wang
>
> **备注:** Technical Report
>
> **摘要:** While recent years have witnessed rapid progress in speech synthesis, open-source singing voice synthesis (SVS) systems still face significant barriers to industrial deployment, particularly in terms of robustness and zero-shot generalization. In this report, we introduce SoulX-Singer, a high-quality open-source SVS system designed with practical deployment considerations in mind. SoulX-Singer supports controllable singing generation conditioned on either symbolic musical scores (MIDI) or melodic representations, enabling flexible and expressive control in real-world production workflows. Trained on more than 42,000 hours of vocal data, the system supports Mandarin Chinese, English, and Cantonese and consistently achieves state-of-the-art synthesis quality across languages under diverse musical conditions. Furthermore, to enable reliable evaluation of zero-shot SVS performance in practical scenarios, we construct SoulX-Singer-Eval, a dedicated benchmark with strict training-test disentanglement, facilitating systematic assessment in zero-shot settings.
>
---
#### [replaced 025] Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升ASR在混响环境下的鲁棒性。提出Whisper-RIR-Mega数据集，包含清晰与混响语音配对样本，评估模型性能并分析混响影响。**

- **链接: [https://arxiv.org/pdf/2603.02252](https://arxiv.org/pdf/2603.02252)**

> **作者:** Mandip Goswami
>
> **摘要:** We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convolved with a real room impulse response from the RIR-Mega corpus, with stratified splits by reverberation time (RT60) and direct-to-reverberant ratio (DRR). We evaluate five Whisper models (tiny through large-v3) on 1600 test samples and report word error rate (WER) and character error rate (CER) under clean and reverberant conditions. Reverberation consistently degrades performance across all model sizes; the reverb penalty in WER ranges from 2.31 to 15.50 percentage points depending on the model. Whisper-large-v3 shows the smallest penalty; Whisper-tiny shows the largest. We release the dataset, evaluation code, and baseline results to support reproducible research on robust ASR.
>
---
#### [replaced 026] Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics
- **分类: cs.SD; cs.LG; eess.AS; physics.comp-ph**

- **简介: 该论文属于物理建模任务，旨在解决非线性动力学学习问题。结合标量辅助变量技术和神经微分方程，提出一种稳定可微模型，直接学习系统动力学特性。**

- **链接: [https://arxiv.org/pdf/2601.10453](https://arxiv.org/pdf/2601.10453)**

> **作者:** Victor Zheleznov; Stefan Bilbao; Alec Wright; Simon King
>
> **备注:** Accepted for publication in Journal of the Audio Engineering Society (special issue on New Frontiers in Digital Audio Effects)
>
> **摘要:** Modal methods are a long-standing approach to physical modelling synthesis. Extensions to nonlinear problems are possible, leading to coupled nonlinear systems of ordinary differential equations. Recent work in scalar auxiliary variable techniques has enabled construction of explicit and stable numerical solvers for such systems. On the other hand, neural ordinary differential equations have been successful in modelling nonlinear systems from data. In this work, we examine how scalar auxiliary variable techniques can be combined with neural ordinary differential equations to yield a stable differentiable model capable of learning nonlinear dynamics. The proposed approach leverages the analytical solution for linear vibration of the system's modes so that physical parameters of a system remain easily accessible after the training without the need for a parameter encoder in the model architecture. Compared to our previous work that used multilayer perceptrons to parametrise nonlinear dynamics, we employ gradient networks that allow an interpretation in terms of a closed-form and non-negative potential required by scalar auxiliary variable techniques. As a proof of concept, we generate synthetic data for the nonlinear transverse vibration of a string and show that the model can be trained to reproduce the nonlinear dynamics of the system. Sound examples are presented.
>
---
#### [replaced 027] Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音情感分析任务，旨在解决静态应力标签无法反映时间变化的问题。通过动态标注和序列模型，提升应力检测效果。**

- **链接: [https://arxiv.org/pdf/2510.08586](https://arxiv.org/pdf/2510.08586)**

> **作者:** Vishakha Lall; Yisi Liu
>
> **摘要:** Detecting psychological stress from speech is critical in high-pressure settings. While prior work has leveraged acoustic features for stress detection, most treat stress as a static label. In this work, we model stress as a temporally evolving phenomenon influenced by historical emotional state. We propose a dynamic labelling strategy that derives fine-grained stress annotations from emotional labels and introduce cross-attention-based sequential models, a Unidirectional LSTM and a Transformer Encoder, to capture temporal stress progression. Our approach achieves notable accuracy gains on MuSE (+5%) and StressID (+18%) over existing baselines, and generalises well to a custom real-world dataset. These results highlight the value of modelling stress as a dynamic construct in speech.
>
---
#### [replaced 028] MOS-Bias: From Hidden Gender Bias to Gender-Aware Speech Quality Assessment
- **分类: eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决MOS中的性别偏差问题。通过分析发现男性和女性评分者存在系统性差异，并提出性别感知模型以提升评估公平性。**

- **链接: [https://arxiv.org/pdf/2603.10723](https://arxiv.org/pdf/2603.10723)**

> **作者:** Wenze Ren; Yi-Cheng Lin; Wen-Chin Huang; Erica Cooper; Ryandhimas E. Zezario; Hsin-Min Wang; Hung-yi Lee; Yu Tsao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The Mean Opinion Score (MOS) serves as the standard metric for speech quality assessment, yet biases in human annotations remain underexplored. We conduct the first systematic analysis of gender bias in MOS, revealing that male listeners consistently assign higher scores than female listeners--a gap that is most pronounced in low-quality speech and gradually diminishes as quality improves. This quality-dependent structure proves difficult to eliminate through simple calibration. We further demonstrate that automated MOS models trained on aggregated labels exhibit predictions skewed toward male standards of perception. To address this, we propose a gender-aware model that learns gender-specific scoring patterns through abstracting binary group embeddings, thereby improving overall and gender-specific prediction accuracy. This study establishes that gender bias in MOS constitutes a systematic, learnable pattern demanding attention in equitable speech evaluation.
>
---
#### [replaced 029] Self Voice Conversion as an Attack against Neural Audio Watermarking
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频水印安全研究，探讨自语音转换攻击对神经音频水印的威胁，旨在揭示现有水印技术在深度学习攻击下的脆弱性。**

- **链接: [https://arxiv.org/pdf/2601.20432](https://arxiv.org/pdf/2601.20432)**

> **作者:** Yigitcan Özer; Wanying Ge; Zhe Zhang; Xin Wang; Junichi Yamagishi
>
> **备注:** 7 pages; 2 figures; 2 tables; accepted at IEICE, SP/SLP 2026
>
> **摘要:** Audio watermarking embeds auxiliary information into speech while maintaining speaker identity, linguistic content, and perceptual quality. Although recent advances in neural and digital signal processing-based watermarking methods have improved imperceptibility and embedding capacity, robustness is still primarily assessed against conventional distortions such as compression, additive noise, and resampling. However, the rise of deep learning-based attacks introduces novel and significant threats to watermark security. In this work, we investigate self voice conversion as a universal, content-preserving attack against audio watermarking systems. Self voice conversion remaps a speaker's voice to the same identity while altering acoustic characteristics through a voice conversion model. We demonstrate that this attack severely degrades the reliability of state-of-the-art watermarking approaches and highlight its implications for the security of modern audio watermarking techniques.
>
---
