# 音频 cs.SD;  eess.AS

- **最新发布 23 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] MEBM-Speech: Multi-scale Enhanced BrainMagic for Robust MEG Speech Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音活动检测任务，解决从MEG信号中准确识别语音与静默状态的问题。提出MEBM-Speech模型，融合多尺度时序建模机制，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02255](https://arxiv.org/pdf/2603.02255)**

> **作者:** Li Songyi; Zheng Linze; Liang Jinghua; Zhang Zifeng
>
> **备注:** 5 pages, 1 figure. To appear in the PNPL Competition Workshop at NeurIPS 2025
>
> **摘要:** We propose MEBM-Speech, a multi-scale enhanced neural decoder for speech activity detection from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Speech integrates three complementary temporal modeling mechanisms: a multi-scale convolutional module for short-term pattern extraction, a bidirectional LSTM (BiLSTM) for long-range context modeling, and a depthwise separable convolutional layer for efficient cross-scale feature fusion. A lightweight temporal jittering strategy and average pooling further improve onset robustness and boundary stability. The model performs continuous probabilistic decoding of MEG signals, enabling fine-grained detection of speech versus silence states - an ability crucial for both cognitive neuroscience and clinical applications. Comprehensive evaluations on the LibriBrain Competition 2025 Track1 benchmark demonstrate strong performance, achieving an average F1 macro of 89.3% on the validation set and comparable results on the official test leaderboard. These findings highlight the effectiveness of multi-scale temporal representation learning for robust MEG-based speech decoding.
>
---
#### [new 002] Bias and Fairness in Self-Supervised Acoustic Representations for Cognitive Impairment Detection
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于认知障碍检测任务，研究声学表示中的偏差与公平性问题，分析不同群体的分类性能差异，提出公平性评估的重要性。**

- **链接: [https://arxiv.org/pdf/2603.02937](https://arxiv.org/pdf/2603.02937)**

> **作者:** Kashaf Gulzar; Korbinian Riedhammer; Elmar Nöth; Andreas K. Maier; Paula Andrea Pérez-Toro
>
> **备注:** 12 pages, 4 figures, 6 tables, Journal paper
>
> **摘要:** Speech-based detection of cognitive impairment (CI) offers a promising non-invasive approach for early diagnosis, yet performance disparities across demographic and clinical subgroups remain underexplored, raising concerns around fairness and generalizability. This study presents a systematic bias analysis of acoustic-based CI and depression classification using the DementiaBank Pitt Corpus. We compare traditional acoustic features (MFCCs, eGeMAPS) with contextualized speech embeddings from Wav2Vec 2.0 (W2V2), and evaluate classification performance across gender, age, and depression-status subgroups. For CI detection, higher-layer W2V2 embeddings outperform baseline features (UAR up to 80.6\%), but exhibit performance disparities; specifically, females and younger participants demonstrate lower discriminative power (\(AUC\): 0.769 and 0.746, respectively) and substantial specificity disparities (\(\Delta_{spec}\) up to 18\% and 15\%, respectively), leading to a higher risk of misclassifications than their counterparts. These disparities reflect representational biases, defined as systematic differences in model performance across demographic or clinical subgroups. Depression detection within CI subjects yields lower overall performance, with mild improvements from low and mid-level W2V2 layers. Cross-task generalization between CI and depression classification is limited, indicating that each task depends on distinct representations. These findings emphasize the need for fairness-aware model evaluation and subgroup-specific analysis in clinical speech applications, particularly in light of demographic and clinical heterogeneity in real-world applications.
>
---
#### [new 003] LMU-Based Sequential Learning and Posterior Ensemble Fusion for Cross-Domain Infant Cry Classification
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于跨领域婴儿啼哭分类任务，解决信号短、标注少和领域差异大的问题。提出融合多特征的CNN-LMU框架与后验集成方法，提升模型泛化能力与实时性。**

- **链接: [https://arxiv.org/pdf/2603.02245](https://arxiv.org/pdf/2603.02245)**

> **作者:** Niloofar Jazaeri; Hilmi R. Dajani; Marco Janeczek; Martin Bouchard
>
> **备注:** 7 pages
>
> **摘要:** Decoding infant cry causes remains challenging for healthcare monitoring due to short nonstationary signals, limited annotations, and strong domain shifts across infants and datasets. We propose a compact acoustic framework that fuses MFCC, STFT, and pitch features within a multi-branch CNN encoder and models temporal dynamics using an enhanced Legendre Memory Unit (LMU). Compared to LSTMs, the LMU backbone provides stable sequence modeling with substantially fewer recurrent parameters, supporting efficient deployment. To improve cross-dataset generalization, we introduce calibrated posterior ensemble fusion with entropy-gated weighting to preserve domain-specific expertise while mitigating dataset bias. Experiments on Baby2020 and Baby Crying demonstrate improved macro-F1 under cross-domain evaluation, along with leakageaware splits and real-time feasibility for on-device monitoring.
>
---
#### [new 004] When Scaling Fails: Mitigating Audio Perception Decay of LALMs via Multi-Step Perception-Aware Reasoning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频语言模型任务，解决LALMs在长推理中出现的感知退化问题。通过引入CAFE评估框架和MPAR$^2方法，提升模型的感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.02266](https://arxiv.org/pdf/2603.02266)**

> **作者:** Ruixiang Mao; Xiangnan Ma; Dan Chen; Ziming Zhu; Yuan Ge; Aokai Hao; Haishu Zhao; Yifu Huo; Qing Yang; Kaiyan Chang; Xiaoqian Liu; Chenglong Wang; Qiaozhi He; Tong Xiao; Jingbo Zhu
>
> **备注:** Under Review
>
> **摘要:** Test-Time Scaling has shown notable efficacy in addressing complex problems through scaling inference compute. However, within Large Audio-Language Models (LALMs), an unintuitive phenomenon exists: post-training models for structured reasoning trajectories results in marginal or even negative gains compared to post-training for direct answering. To investigate it, we introduce CAFE, an evaluation framework designed to precisely quantify audio reasoning errors. Evaluation results reveal LALMs struggle with perception during reasoning and encounter a critical bottleneck: reasoning performance suffers from audio perception decay as reasoning length extends. To address it, we propose MPAR$^2$, a paradigm that encourages dynamic perceptual reasoning and decomposes complex questions into perception-rich sub-problems. Leveraging reinforcement learning, MPAR$^2$ improves perception performance on CAFE from 31.74% to 63.51% and effectively mitigates perception decay, concurrently enhancing reasoning capabilities to achieve a significant 74.59% accuracy on the MMAU benchmark. Further analysis demonstrates that MPAR$^2$ reinforces LALMs to attend to audio input and dynamically adapts reasoning budget to match task complexity.
>
---
#### [new 005] Benchmarking Speech Systems for Frontline Health Conversations: The DISPLACE-M Challenge
- **分类: eess.AS**

- **简介: 该论文属于医疗对话理解任务，旨在解决多说话人、噪声环境下的医疗对话分析问题。工作包括构建数据集、提供基线系统，并评估多个任务性能。**

- **链接: [https://arxiv.org/pdf/2603.02813](https://arxiv.org/pdf/2603.02813)**

> **作者:** Dhanya E; Ankita Meena; Manas Nanivadekar; Noumida A; Victor Azad; Ashwini Nagaraj Shenoy; Pratik Roy Chowdhuri; Shobhit Banga; Vanshika Chhabra; Chitralekha Bhat; Shareef babu Kalluri; Srikanth Raj Chetupalli; Deepu Vijayasenan; Sriram Ganapathy
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** The DIarization and Speech Processing for LAnguage understanding in Conversational Environments - Medical (DISPLACE-M) challenge introduces a conversational AI benchmark focused on understanding goal-oriented, real-world medical dialogues collected in the field. The challenge addresses multi-speaker interactions between healthcare workers and seekers characterized by spontaneous, noisy and overlapping speech across Indian languages and dialects. As part of the challenge, medical conversational dataset comprising 25 hours of development data and 10 hours of blind evaluation recordings was released. We provided baseline systems within a unified end-to-end pipeline across 4 tasks - speaker diarization, automatic speech recognition, topic identification and dialogue summarization - to enable consistent benchmarking. System performance is evaluated using established metrics such as diarization error rate (DER), time-constrained minimum-permutation word error rate (tcpWER), and ROUGE-L. During this evaluation (Phase-I), 12 teams, across the globe, actively participated pushing the baseline systems on these metrics. However, even with a 6-8 week dedicated effort from various participants, the task is shown to be substantially challenging, and the existing systems are significantly short of healthcare deployment readiness.
>
---
#### [new 006] Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?
- **分类: eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升模型在未见攻击下的泛化能力。研究采用强化学习方法进行微调，实验表明其优于传统监督微调，有效提升了跨领域性能。**

- **链接: [https://arxiv.org/pdf/2603.02914](https://arxiv.org/pdf/2603.02914)**

> **作者:** Xin Wang; Ge Wanying; Junichi Yamagishi
>
> **备注:** Submitted to Interspeech 2026; put on arxiv based on requirement of paper open-access rule; quote from Interspeech: "Interspeech no longer enforces an anonymity period for submissions. While uploading a version online is permitted, your official submission to Interspeech must not contain any author-identifying information"
>
> **摘要:** Building speech deepfake detection models that are generalizable to unseen attacks remains a challenging problem. Although the field has shifted toward a pre-training and fine-tuning paradigm using speech foundation models, most approaches rely solely on supervised fine-tuning (SFT). Inspired by the field of large language models, wherein reinforcement learning (RL) is used for model fine-tuning, we investigate the impact of RL, specifically Group Relative Policy Optimization (GRPO). The results from experiments using multiple detectors and test sets indicate that pure GRPO-based fine-tuning improves performance on out-of-domain test sets while maintaining performance on target-domain test data. This approach outperforms both SFT-only and hybrid setups. Our ablation studies further suggest that the negative reward in GRPO may be a key factor in this improvement.
>
---
#### [new 007] Sequence-Level Unsupervised Training in Speech Recognition: A Theoretical Study
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，研究无监督语音识别的理论条件与训练目标。通过建立分类误差边界，提出一种单阶段序列交叉熵损失函数。**

- **链接: [https://arxiv.org/pdf/2603.02285](https://arxiv.org/pdf/2603.02285)**

> **作者:** Zijian Yang; Jörg Barkoczi; Ralf Schlüter; Hermann Ney
>
> **备注:** accepted to ICASSP 2026
>
> **摘要:** Unsupervised speech recognition is a task of training a speech recognition model with unpaired data. To determine when and how unsupervised speech recognition can succeed, and how classification error relates to candidate training objectives, we develop a theoretical framework for unsupervised speech recognition grounded in classification error bounds. We introduce two conditions under which unsupervised speech recognition is possible. The necessity of these conditions are also discussed. Under these conditions, we derive a classification error bound for unsupervised speech recognition and validate this bound in simulations. Motivated by this bound, we propose a single-stage sequence-level cross-entropy loss for unsupervised speech recognition.
>
---
#### [new 008] MEBM-Phoneme: Multi-scale Enhanced BrainMagic for End-to-End MEG Phoneme Classification
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音感知分析任务，旨在提升MEG信号中的音素分类准确率。通过多尺度特征融合与动态注意力机制，解决类不平衡和分布偏移问题，增强模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.02254](https://arxiv.org/pdf/2603.02254)**

> **作者:** Liang Jinghua; Zhang Zifeng; Li Songyi; Zheng Linze
>
> **备注:** 5 pages, 1 figure. To appear in the PNPL Competition Workshop at NeurIPS 2025
>
> **摘要:** We propose MEBM-Phoneme, a multi-scale enhanced neural decoder for phoneme classification from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Phoneme integrates a short-term multi-scale convolutional module to augment the native mid-term encoder, with fused representations via depthwise separable convolution for efficient cross-scale integration. A convolutional attention layer dynamically weights temporal dependencies to refine feature aggregation. To address class imbalance and session-specific distributional shifts, we introduce a stacking-based local validation set alongside weighted cross-entropy loss and random temporal augmentation. Comprehensive evaluations on LibriBrain Competition 2025 Track2 demonstrate robust generalization, achieving competitive phoneme decoding accuracy on the validation and official test leaderboard. These results underscore the value of hierarchical temporal modeling and training stabilization for advancing MEG-based speech perception analysis.
>
---
#### [new 009] OnDA: On-device Channel Pruning for Efficient Personalized Keyword Spotting
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于个性化关键词识别任务，解决设备端模型适应性问题。通过结合权重更新与在线结构化通道剪枝，实现模型压缩和效率提升。**

- **链接: [https://arxiv.org/pdf/2603.02247](https://arxiv.org/pdf/2603.02247)**

> **作者:** Matteo Risso; Alessio Burrello; Daniele Jahier Pagliari
>
> **备注:** Submitted for review at Interspeech2026
>
> **摘要:** Always-on keyword spotting (KWS) demands on-device adaptation to cope with user- and environment-specific distribution shifts under tight latency and energy budgets. This paper proposes, for the first time, coupling weight adaptation (i.e., on-device training) with architectural adaptation, in the form of online structured channel pruning, for personalized on-device KWS. Starting from a state-of-the-art self-learning personalized KWS pipeline, we compare data-agnostic and data-aware pruning criteria applied on in-field pseudo-labelled user data. On the HeySnips and HeySnapdragon datasets, we achieve up to 9.63x model-size compression with respect to unpruned baselines at iso-task performance, measured as the accuracy at 0.5 false alarms per hour. When deploying our adaptation pipeline on a Jetson Orin Nano embedded GPU, we achieve up to 1.52x/1.57x and 1.64x/1.77x latency and energy-consumption improvements during online training/inference compared to weights-only adaptation.
>
---
#### [new 010] Decomposing the Influence of Physical Acoustic Modeling on Neural Personal Sound Zone Rendering: An Ablation Study
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于声学建模任务，旨在解决神经个人音区渲染中的模拟与现实差距问题。通过消融实验评估不同物理模型对音区分离的影响。**

- **链接: [https://arxiv.org/pdf/2603.02508](https://arxiv.org/pdf/2603.02508)**

> **作者:** Hao Jiang; Edgar Choueiri
>
> **摘要:** Deep learning-based Personal Sound Zones (PSZs) rely on simulated acoustic transfer functions (ATFs) for training, yet idealized point-source models exhibit large sim-to-real gaps. While physically informed components improve generalization, individual contributions remain unclear. This paper presents a controlled ablation study on a head-pose-conditioned binaural PSZ renderer using the Binaural Spatial Audio Neural Network (BSANN). We progressively enrich simulated ATFs with three components: (i) anechoically measured frequency responses of the particular loudspeakers(FR), (ii) analytic circular-piston directivity (DIR), and (iii) rigid-sphere head-related transfer functions (RS-HRTF). Four configurations are evaluated via in-situ measurements with two dummy heads. Performance metrics include inter-zone isolation (IZI), inter-program interference (IPI), and crosstalk cancellation (XTC) over 100-20000 Hz. Results show FR provides spectral calibration, yielding modest XTC improvements and reduced inter-listener IPI imbalance. DIR delivers the most consistent sound-zone separation gains (10.05 dB average IZI/IPI). RS-HRTF dominates binaural separation, boosting XTC by +2.38/+2.89 dB (average 4.51 to 7.91 dB), primarily above 2 kHz, while introducing mild listener-dependent IZI/IPI shifts. These findings guide prioritization of measurements and models when constructing training ATFs under limited budgets.
>
---
#### [new 011] Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决训练目标选择、失真与感知平衡及数据质量问题。提出时间偏移干净语音作为目标，设计两阶段框架，并分析数据规模与质量的权衡。**

- **链接: [https://arxiv.org/pdf/2603.02641](https://arxiv.org/pdf/2603.02641)**

> **作者:** Szu-Wei Fu; Rong Chao; Xuesong Yang; Sung-Feng Huang; Ryandhimas E. Zezario; Rauf Nasretdinov; Ante Jukić; Yu Tsao; Yu-Chiang Frank Wang
>
> **摘要:** Universal Speech Enhancement (USE) aims to restore speech quality under diverse degradation conditions while preserving signal fidelity. Despite recent progress, key challenges in training target selection, the distortion--perception tradeoff, and data curation remain unresolved. In this work, we systematically address these three overlooked problems. First, we revisit the conventional practice of using early-reflected speech as the dereverberation target and show that it can degrade perceptual quality and downstream ASR performance. We instead demonstrate that time-shifted anechoic clean speech provides a superior learning target. Second, guided by the distortion--perception tradeoff theory, we propose a simple two-stage framework that achieves minimal distortion under a given level of perceptual quality. Third, we analyze the trade-off between training data scale and quality for USE, revealing that training on large uncurated corpora imposes a performance ceiling, as models struggle to remove subtle artifacts. Our method achieves state-of-the-art performance on the URGENT 2025 non-blind test set and exhibits strong language-agnostic generalization, making it effective for improving TTS training data. Code and models will be released upon acceptance.
>
---
#### [new 012] Single Microphone Own Voice Detection based on Simulated Transfer Functions for Hearing Aids
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音检测任务，旨在解决助听器中单麦克风自声检测问题。通过模拟声学传递函数数据增强，训练模型提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2603.02724](https://arxiv.org/pdf/2603.02724)**

> **作者:** Mathuranathan Mayuravaani; W. Bastiaan Kleijn; Andrew Lensen; Charlotte Sørensen
>
> **摘要:** This paper presents a simulation-based approach to own voice detection (OVD) in hearing aids using a single microphone. While OVD can significantly improve user comfort and speech intelligibility, existing solutions often rely on multiple microphones or additional sensors, increasing device complexity and cost. To enable ML-based OVD without requiring costly transfer-function measurements, we propose a data augmentation strategy based on simulated acoustic transfer functions (ATFs) that expose the model to a wide range of spatial propagation conditions. A transformer-based classifier is first trained on analytically generated ATFs and then progressively fine-tuned using numerically simulated ATFs, transitioning from a rigid-sphere model to a detailed head-and-torso representation. This hierarchical adaptation enabled the model to refine its spatial understanding while maintaining generalization. Experimental results show 95.52% accuracy on simulated head-and-torso test data. Under short-duration conditions, the model maintained 90.02% accuracy with one-second utterances. On real hearing aid recordings, the model achieved 80% accuracy without fine-tuning, aided by lightweight test-time feature compensation. This highlights the model's ability to generalize from simulated to real-world conditions, demonstrating practical viability and pointing toward a promising direction for future hearing aid design.
>
---
#### [new 013] DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决低信噪比下传统系统性能下降的问题。提出DBMIF框架，融合空气传导和骨传导语音，提升语音质量与可懂度。**

- **链接: [https://arxiv.org/pdf/2603.02877](https://arxiv.org/pdf/2603.02877)**

> **作者:** Yilei Wu; Changyan Zheng; Xingyu Zhang; Yakun Zhang; Chengshi Zheng; Shuang Yang; Ye Yan; Erwei Yin
>
> **备注:** 10 pages, 7 figures, Applied Intelligence
>
> **摘要:** The performance of conventional speech enhancement systems degrades sharply in extremely low signal-to-noise ratio (SNR) environments where air-conduction (AC) microphones are overwhelmed by ambient noise. Although bone-conduction (BC) sensors offer complementary, noise-tolerant information, existing fusion approaches struggle to maintain consistent performance across a wide range of SNR conditions. To address this limitation, we propose the Deep Balanced Multimodal Iterative Fusion Framework (DBMIF), a three-branch architecture designed to reconstruct high-fidelity speech through rigorous cross-modal interaction. Specifically, grounded in a multi-scale interactive encoder-decoder backbone, the framework orchestrates an iterative attention module and a cross-branch gated module to facilitate adaptive weighting and bidirectional exchange. To complement this dynamic interaction, a balanced-interaction bottleneck is further integrated to learn a compact, stable fused representation. Extensive experiments demonstrate that DBMIF achieves competitive performance compared with recent unimodal and multimodal baselines in both speech quality and intelligibility across diverse noise types. In downstream ASR tasks, the proposed method reduces the character error rate by at least 2.5 percent compared to competing approaches. These results confirm that DBMIF effectively harnesses the robustness of BC speech while preserving the naturalness of AC speech, ensuring reliability in real-world scenarios. The source code is publicly available at this http URL.
>
---
#### [new 014] Quality of Automatic Speech Recognition -- Polish Language case study -- from Wav2Vec to Scribe ElevenLabs
- **分类: eess.AS; cs.SD**

- **简介: 本文研究波兰语自动语音识别任务，比较不同ASR模型在医疗访谈中的表现，结合LLM提升识别质量。**

- **链接: [https://arxiv.org/pdf/2603.02246](https://arxiv.org/pdf/2603.02246)**

> **作者:** Marcin Pietroń; Szymon Piórkowski; Kamil Faber; Dominik Żurek; Michał Karwatowski; Jerzy Duda; Hubert Zieliński; Piotr Lipnicki; Mikołaj Leszczuk
>
> **摘要:** This article concerns comparative studies on the Automatic Speech Recognition (ASR) model incorporated with the Large Language Model (LLM) used for medical interviews. The proposed solution is tested on polish language benchmarks and dataset with medical interviews. The latest ASR technologies are based on convolutional neural networks (CNNs), recurrent neural networks (RNNs) and Transformers. Most of them work as end-to-end solutions. The presented approach in the case of the Whisper model shows a two-stage solution with End-To-End ASR and LLM working together in a pipeline. The ASR output is an input for LLM. The LLM is a component by which the output from ASR is corrected and improved. Comparative studies for automatic recognition of the Polish language between modern End-To-End deep learning architectures and the ASR hybrid model were performed. The medical interview tests were performed with two state-of-the-art ASR models: OpenAI Whisper incorporated with LLM and Scribe ElevenLabs. Additionally, the results were compared with five more end-to-end models (QuartzNet, FastConformer, Wav2Vec 2.0 XLSR and ESPnet Model Zoo) on Mozilla Common Voice and VoxPopuli databases. Tests were conducted for clean audio signal, signal with bandwidth limitation, and degraded. The tested models were evaluated on the basis of Word Error Rate (WER) and Character Error Rate (CER). The results show that the Whisper model performs by far the best among the open-source models. ElevenLabs Scribe model, on the other hand, performs best for Polish on both general benchmark and medical data.
>
---
#### [new 015] When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音 spoof 检测任务，旨在解决跨语言检测鲁棒性问题。通过构建多语言数据集，评估不同模型在66种语言中的检测效果，揭示语言对检测性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.02364](https://arxiv.org/pdf/2603.02364)**

> **作者:** Kirill Borodin; Vasiliy Kudryavtsev; Maxim Maslov; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** This paper has been submitted to Interspeech 2026 for review
>
> **摘要:** We introduce LRLspoof, a large-scale multilingual synthetic-speech corpus for cross-lingual spoof detection, comprising 2,732 hours of audio generated with 24 open-source TTS systems across 66 languages, including 45 low-resource languages under our operational definition. To evaluate robustness without requiring target-domain bonafide speech, we benchmark 11 publicly available countermeasures using threshold transfer: for each model we calibrate an EER operating point on pooled external benchmarks and apply the resulting threshold, reporting spoof rejection rate (SRR). Results show model-dependent cross-lingual disparity, with spoof rejection varying markedly across languages even under controlled conditions, highlighting language as an independent source of domain shift in spoof detection. The dataset is publicly available at \href{this https URL}{\textbf{\underline{\textit{HuggingFace}}}} and \href{this https URL}{\textbf{\underline{\textit{ModelScope}}}}
>
---
#### [new 016] Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出Whisper-RIR-Mega数据集，用于评估ASR在混响环境下的鲁棒性。通过配对干净与混响语音样本，分析模型性能下降情况，旨在提升语音识别在真实声学环境中的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.02252](https://arxiv.org/pdf/2603.02252)**

> **作者:** Mandip Goswami
>
> **摘要:** We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convolved with a real room impulse response from the RIR-Mega corpus, with stratified splits by reverberation time (RT60) and direct-to-reverberant ratio (DRR). We evaluate five Whisper models (tiny through large-v3) on 1600 test samples and report word error rate (WER) and character error rate (CER) under clean and reverberant conditions. Reverberation consistently degrades performance across all model sizes; the reverb penalty in WER ranges from 0.12 to 1.07 percentage points depending on the model. We release the dataset, evaluation code, and baseline results to support reproducible research on robust ASR.
>
---
#### [new 017] Differentiable Time-Varying IIR Filtering for Real-Time Speech Denoising
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音去噪任务，旨在解决非平稳噪声环境下的实时语音增强问题。提出TVF模型，结合DSP与深度学习，实现可解释的动态滤波。**

- **链接: [https://arxiv.org/pdf/2603.02794](https://arxiv.org/pdf/2603.02794)**

> **作者:** Riccardo Rota; Kiril Ratmanski; Jozef Coldenhoff; Milos Cernak
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We present TVF (Time-Varying Filtering), a low-latency speech enhancement model with 1 million parameters. Combining the interpretability of Digital Signal Processing (DSP) with the adaptability of deep learning, TVF bridges the gap between traditional filtering and modern neural speech modeling. The model utilizes a lightweight neural network backbone to predict the coefficients of a differentiable 35-band IIR filter cascade in real time, allowing it to adapt dynamically to non-stationary noise. Unlike ``black-box'' deep learning approaches, TVF offers a completely interpretable processing chain, where spectral modifications are explicit and adjustable. We demonstrate the efficacy of this approach on a speech denoising task using the Valentini-Botinhao dataset and compare the results to a static DDSP approach and a fully deep-learning-based solution, showing that TVF achieves effective adaptation to changing noise conditions.
>
---
#### [new 018] An Investigation Into Various Approaches For Bengali Long-Form Speech Transcription and Bengali Speaker Diarization
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对孟加拉语长语音转录和说话人二分类任务，提出多阶段方法解决“谁在何时说了什么”的问题。通过优化模型和数据处理，提升低资源语言的语音技术性能。**

- **链接: [https://arxiv.org/pdf/2603.03158](https://arxiv.org/pdf/2603.03158)**

> **作者:** Epshita Jahan; Khandoker Md Tanjinul Islam; Pritom Biswas; Tafsir Al Nafin
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Bengali remains a low-resource language in speech technology, especially for complex tasks like long-form transcription and speaker diarization. This paper presents a multistage approach developed for the "DL Sprint 4.0 - Bengali Long-Form Speech Recognition" and "DL Sprint 4.0 - Bengali Speaker Diarization" competitions on Kaggle, addressing the challenge of "who spoke when/what" in hour-long recordings. We implemented Whisper Medium fine-tuned on Bengali data (bengaliAI/tugstugi bengaliai-asr whisper-medium) for transcription and integrated pyannote/speaker-diarization-community-1 with our custom-trained segmentation model to handle diverse and noisy acoustic environments. Using a two-pass method with hyperparameter tuning, we achieved a DER of 0.27 on the private leaderboard and 0.19 on the public leaderboard. For transcription, chunking, background noise cleaning, and algorithmic post-processing yielded a WER of 0.38 on the private leaderboard. These results show that targeted tuning and strategic data utilization can significantly improve AI inclusivity for South Asian languages. All relevant code is available at: this https URL Index Terms: Bengali speech recognition, speaker diarization, Whisper, ASR, low-resource languages, pyannote, voice activity detection
>
---
#### [new 019] Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究自监督学习语音特征中说话人属性的表征，解决如何在特征维度中捕捉语音特性的问题。通过PCA分析，发现不同主成分对应不同语音特征，并验证可通过调整维度控制语音合成结果。**

- **链接: [https://arxiv.org/pdf/2603.03096](https://arxiv.org/pdf/2603.03096)**

> **作者:** Kyle Janse van Rensburg; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 pages, 7 figures, submitted to IEEE Signal Processing Letters
>
> **摘要:** How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. Using WavLM, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. Finally, in synthesis experiments we show that most characteristics can be controlled by changing the corresponding dimensions. This provides a simple method to control characteristics of the output voice in synthesis applications.
>
---
#### [new 020] SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频解释任务，解决音频语言模型中Shapley值解释的可行性问题。通过SGPA方法，实现音频段的稳定对齐，减少计算量并提升解释有效性。**

- **链接: [https://arxiv.org/pdf/2603.02250](https://arxiv.org/pdf/2603.02250)**

> **作者:** Paweł Pozorski; Jakub Muszyński; Maria Ganzha
>
> **备注:** Submitted for admission in Interspeech 2026 conference
>
> **摘要:** Explaining the behavior of end-to-end audio language models via Shapley value attribution is intractable under native tokenization: a typical utterance yields over $150$ encoder frames, inflating the coalition space by roughly $10^{42}$ relative to text; individual audio frames lack standalone meaning; and token boundaries that bisect phonetic transitions introduce masking artifacts. We introduce Spectrogram-Guided Phonetic Alignment (SGPA), a four-stage pipeline that combines Connectionist Temporal Classification forced alignment with spectral boundary refinement to produce acoustically stable, word-aligned audio segments. Controlled diagnostics on LFM2-Audio-1.5B with VoiceBench show that SGPA yields a 43$\times$ reduction in model evaluations. Statistical testing confirms that SGPA significantly alters attribution concentration while preserving the global cumulative profile, establishing it as a feasibility-enabling layer for audio explainability.
>
---
#### [new 021] DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming
- **分类: eess.IV; eess.AS**

- **简介: 该论文提出DLIOS系统，用于抖音直播的实时多模态互动增强。解决直播中弹幕、礼物特效等同步问题，通过LLM实现自动化内容生成与响应。**

- **链接: [https://arxiv.org/pdf/2603.03060](https://arxiv.org/pdf/2603.03060)**

> **作者:** Shuide Wen; Sungil Seok; Beier Ku; Richee Li; Yubin He; Bowen Qu; Yang Yang; Ping Su; Can Jiao
>
> **备注:** 14 pages, 13 figures, 6 tables, 7 algorithms, 16 references, submitted to ACM/IEEE International Conference on Systems and Software Engineering
>
> **摘要:** We present DLIOS, a Large Language Model (LLM)-augmented real-time multi-modal interactive enhancement overlay system for Douyin (TikTok) live streaming. DLIOS employs a three-layer transparent window architecture for independent rendering of danmaku (scrolling text), gift and like particle effects, and VIP entrance animations, built around an event-driven WebView2 capture pipeline and a thread-safe event bus. On top of this foundation we contribute an LLM broadcast automation framework comprising: (1) a per-song four-segment prompt scheduling system (T1 opening/transition, T2 empathy, T3 era story/production notes, T4 closing) that generates emotionally coherent radio-style commentary from lyric metadata; (2) a JSON-serializable RadioPersonaConfig schema supporting hot-swap multi-persona broadcasting; (3) a real-time danmaku quick-reaction engine with keyword routing to static urgent speech or LLM-generated empathetic responses; and (4) the Suwan Li AI singer-songwriter persona case study -- over 100 AI-generated songs produced with Suno. A 36-hour stress test demonstrates: zero danmaku overlap, zero deadlock crashes, gift effect P95 latency <= 180 ms, LLM-to-TTS segment P95 latency <= 2.1 s, and TTS integrated loudness gain of 9.5 LUFS. live streaming; danmaku; large language model; prompt engineering; virtual persona; WebView2; WINMM; TTS; Suno; loudness normalization; real-time scheduling
>
---
#### [new 022] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models
- **分类: cs.LG; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于大模型安全评估任务，旨在解决多模态对齐评估不足的问题。提出MUSE平台，集成多模态攻击与评估方法，提升安全测试效果。**

- **链接: [https://arxiv.org/pdf/2603.02482](https://arxiv.org/pdf/2603.02482)**

> **作者:** Zhongxi Wang; Yueqian Lin; Jingyang Zhang; Hai Helen Li; Yiran Chen
>
> **备注:** Submitted to ACL 2026 System Demonstration Track
>
> **摘要:** Safety evaluation and red-teaming of large language models remain predominantly text-centric, and existing frameworks lack the infrastructure to systematically test whether alignment generalizes to audio, image, and video inputs. We present MUSE (Multimodal Unified Safety Evaluation), an open-source, run-centric platform that integrates automatic cross-modal payload generation, three multi-turn attack algorithms (Crescendo, PAIR, Violent Durian), provider-agnostic model routing, and an LLM judge with a five-level safety taxonomy into a single browser-based system. A dual-metric framework distinguishes hard Attack Success Rate (Compliance only) from soft ASR (including Partial Compliance), capturing partial information leakage that binary metrics miss. To probe whether alignment generalizes across modality boundaries, we introduce Inter-Turn Modality Switching (ITMS), which augments multi-turn attacks with per-turn modality rotation. Experiments across six multimodal LLMs from four providers show that multi-turn strategies can achieve up to 90-100% ASR against models with near-perfect single-turn refusal. ITMS does not uniformly raise final ASR on already-saturated baselines, but accelerates convergence by destabilizing early-turn defenses, and ablation reveals that the direction of modality effects is model-family-specific rather than universal, underscoring the need for provider-aware cross-modal safety testing.
>
---
#### [new 023] RO-N3WS: Enhancing Generalization in Low-Resource ASR with Diverse Romanian Speech Benchmarks
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 论文提出RO-N3WS基准数据集，用于提升低资源场景下的自动语音识别（ASR）泛化能力。针对ASR任务中的低资源和分布外问题，通过多样化罗马尼亚语语音数据进行模型训练与优化。**

- **链接: [https://arxiv.org/pdf/2603.02368](https://arxiv.org/pdf/2603.02368)**

> **作者:** Alexandra Diaconu; Mădălina Vînaga; Bogdan Alexe
>
> **摘要:** We introduce RO-N3WS, a benchmark Romanian speech dataset designed to improve generalization in automatic speech recognition (ASR), particularly in low-resource and out-of-distribution (OOD) conditions. RO-N3WS comprises over 126 hours of transcribed audio collected from broadcast news, literary audiobooks, film dialogue, children's stories, and conversational podcast speech. This diversity enables robust training and fine-tuning across stylistically distinct domains. We evaluate several state-of-the-art ASR systems (Whisper, Wav2Vec 2.0) in both zero-shot and fine-tuned settings, and conduct controlled comparisons using synthetic data generated with expressive TTS models. Our results show that even limited fine-tuning on real speech from RO-N3WS yields substantial WER improvements over zero-shot baselines. We will release all models, scripts, and data splits to support reproducible research in multilingual ASR, domain adaptation, and lightweight deployment.
>
---
## 更新

#### [replaced 001] On Adversarial Attacks In Acoustic Drone Localization
- **分类: cs.SD; cs.RO; eess.AS**

- **简介: 该论文研究对抗攻击对声学无人机定位的影响，属于无人机导航安全任务，旨在分析PGD攻击并提出恢复算法以减轻攻击影响。**

- **链接: [https://arxiv.org/pdf/2502.20325](https://arxiv.org/pdf/2502.20325)**

> **作者:** Tamir Shor; Chaim Baskin; Alex Bronstein
>
> **摘要:** Multi-rotor aerial autonomous vehicles (MAVs, more widely known as "drones") have been generating increased interest in recent years due to their growing applicability in a vast and diverse range of fields (e.g., agriculture, commercial delivery, search and rescue). The sensitivity of visual-based methods to lighting conditions and occlusions had prompted growing study of navigation reliant on other modalities, such as acoustic sensing. A major concern in using drones in scale for tasks in non-controlled environments is the potential threat of adversarial attacks over their navigational systems, exposing users to mission-critical failures, security breaches, and compromised safety outcomes that can endanger operators and bystanders. While previous work shows impressive progress in acoustic-based drone localization, prior research in adversarial attacks over drone navigation only addresses visual sensing-based systems. In this work, we aim to compensate for this gap by supplying a comprehensive analysis of the effect of PGD adversarial attacks over acoustic drone localization. We furthermore develop an algorithm for adversarial perturbation recovery, capable of markedly diminishing the affect of such attacks in our setting.
>
---
#### [replaced 002] AI-Generated Music Detection in Broadcast Monitoring
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于AI音乐检测任务，解决广播环境中AI生成音乐识别问题。针对广播音频特点，构建了AI-OpenBMAT数据集，并评估模型在复杂场景下的性能。**

- **链接: [https://arxiv.org/pdf/2602.06823](https://arxiv.org/pdf/2602.06823)**

> **作者:** David López-Ayala; Asier Cabello; Pablo Zinemanas; Emilio Molina; Martín Rocamora
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** AI music generators have advanced to the point where their outputs are often indistinguishable from human compositions. While detection methods have emerged, they are typically designed and validated in music streaming contexts with clean, full-length tracks. Broadcast audio, however, poses a different challenge: music appears as short excerpts, often masked by dominant speech, conditions under which existing detectors fail. In this work, we introduce AI-OpenBMAT, the first dataset tailored to broadcast-style AI-music detection. It contains 3,294 one-minute audio excerpts (54.9 hours) that follow the duration patterns and loudness relations of real television audio, combining human-made production music with stylistically matched continuations generated with Suno v3.5. We benchmark a CNN baseline and state-of-the-art SpectTTTra models to assess SNR and duration robustness, and evaluate on a full broadcast scenario. Across all settings, models that excel in streaming scenarios suffer substantial degradation, with F1-scores dropping below 60% when music is in the background or has a short duration. These results highlight speech masking and short music length as critical open challenges for AI music detection, and position AI-OpenBMAT as a benchmark for developing detectors capable of meeting industrial broadcast requirements.
>
---
#### [replaced 003] TCG CREST System Description for the DISPLACE-M Challenge
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于说话人二分类任务，旨在提升医疗场景下的语音分离性能。通过比较不同VAD方法和聚类算法，优化了二分类系统，提升了准确率。**

- **链接: [https://arxiv.org/pdf/2603.02030](https://arxiv.org/pdf/2603.02030)**

> **作者:** Nikhil Raghav; Md Sahidullah
>
> **备注:** Report submitted for the DISPLACE-M challenge
>
> **摘要:** This report presents the TCG CREST system description for Track 1 (Speaker Diarization) of the DISPLACE-M challenge, focusing on naturalistic medical conversations in noisy rural-healthcare scenarios. Our study evaluates the impact of various voice activity detection (VAD) methods and advanced clustering algorithms on overall speaker diarization (SD) performance. We compare and analyze two SD frameworks: a modular pipeline utilizing SpeechBrain with ECAPA-TDNN embeddings, and a state-of-the-art (SOTA) hybrid end-to-end neural diarization system, Diarizen, built on top of a pre-trained WavLM. With these frameworks, we explore diverse clustering techniques, including agglomerative hierarchical clustering (AHC), and multiple novel variants of spectral clustering, such as SC-adapt, SC-PNA, and SC-MK. Experimental results demonstrate that the Diarizen system provides an approximate $39\%$ relative improvement in the diarization error rate (DER) on the post-evaluation analysis of Phase~I compared to the SpeechBrain baseline. Our best-performing submitted system employing the Diarizen baseline with AHC employing a median filtering with a larger context window of $29$ achieved a DER of 10.37\% on the development and 9.21\% on the evaluation sets, respectively. Our team ranked sixth out of the 11 participating teams after the Phase~I evaluation.
>
---
#### [replaced 004] UniTAF: A Modular Framework for Joint Text-to-Speech and Audio-to-Face Modeling
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于语音与表情联合建模任务，旨在通过整合TTS和A2F模型，实现音频与面部表情的一致性生成，并探索情感控制机制的扩展。**

- **链接: [https://arxiv.org/pdf/2602.15651](https://arxiv.org/pdf/2602.15651)**

> **作者:** Qiangong Zhou; Nagasaka Tomohiro
>
> **备注:** We have identified inaccuracies in some results that require further verification. To avoid misleading the research community, we are temporarily withdrawing the paper
>
> **摘要:** This work considers merging two independent models, TTS and A2F, into a unified model to enable internal feature transfer, thereby improving the consistency between audio and facial expressions generated from text. We also discuss the extension of the emotion control mechanism from TTS to the joint model. This work does not aim to showcase generation quality; instead, from a system design perspective, it validates the feasibility of reusing intermediate representations from TTS for joint modeling of speech and facial expressions, and provides engineering practice references for subsequent speech expression co-design. The project code has been open source at: this https URL
>
---
#### [replaced 005] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文属于视频到音频生成任务，解决目标混淆和人类偏好对齐问题。提出PrismAudio框架，结合强化学习与分解的思维链模块，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.18833](https://arxiv.org/pdf/2511.18833)**

> **作者:** Huadai Liu; Kaicheng Luo; Wen Wang; Qian Chen; Peiwen Sun; Rongjie Huang; Xiangang Li; Jieping Ye; Wei Xue
>
> **备注:** ICLR 2026
>
> **摘要:** Video-to-Audio (V2A) generation requires balancing four critical perceptual dimensions: semantic consistency, audio-visual temporal synchrony, aesthetic quality, and spatial accuracy; yet existing methods suffer from objective entanglement that conflates competing goals in single loss functions and lack human preference alignment. We introduce PrismAudio, the first framework to integrate Reinforcement Learning into V2A generation with specialized Chain-of-Thought (CoT) planning. Our approach decomposes monolithic reasoning into four specialized CoT modules (Semantic, Temporal, Aesthetic, and Spatial CoT), each paired with targeted reward functions. This CoT-reward correspondence enables multidimensional RL optimization that guides the model to jointly generate better reasoning across all perspectives, solving the objective entanglement problem while preserving interpretability. To make this optimization computationally practical, we propose Fast-GRPO, which employs hybrid ODE-SDE sampling that dramatically reduces the training overhead compared to existing GRPO implementations. We also introduce AudioCanvas, a rigorous benchmark that is more distributionally balanced and covers more realistically diverse and challenging scenarios than existing datasets, with 300 single-event classes and 501 multi-event samples. Experimental results demonstrate that PrismAudio achieves state-of-the-art performance across all four perceptual dimensions on both the in-domain VGGSound test set and out-of-domain AudioCanvas benchmark. The project page is available at this https URL.
>
---
#### [replaced 006] VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures
- **分类: cs.SD**

- **简介: 该论文属于实时语音代理任务，旨在解决RAG的延迟瓶颈。通过双代理架构，分离检索与生成，提升响应速度。**

- **链接: [https://arxiv.org/pdf/2603.02206](https://arxiv.org/pdf/2603.02206)**

> **作者:** Jielin Qiu; Jianguo Zhang; Zixiang Chen; Liangwei Yang; Ming Zhu; Juntao Tan; Haolin Chen; Wenting Zhao; Rithesh Murthy; Roshan Ram; Akshara Prabhakar; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** We present VoiceAgentRAG, an open-source dual-agent memory router that decouples retrieval from response generation. A background Slow Thinker agent continuously monitors the conversation stream, predicts likely follow-up topics using an LLM, and pre-fetches relevant document chunks into a FAISS-backed semantic cache. A foreground Fast Talker agent reads only from this sub-millisecond cache, bypassing the vector database entirely on cache hits.
>
---
#### [replaced 007] Diffusion-based Symbolic Music Generation with Structured State Space Models
- **分类: cs.SD**

- **简介: 该论文属于符号音乐生成任务，旨在解决长序列生成中的计算效率与表达能力问题。提出SMDIM模型，结合SSM和MFA块，实现高效且高质量的音乐生成。**

- **链接: [https://arxiv.org/pdf/2507.20128](https://arxiv.org/pdf/2507.20128)**

> **作者:** Shenghua Yuan; Xing Tang; Jiatao Chen; Tianming Xie; Jing Wang; Bing Shi
>
> **备注:** This is a duplicate submission. The updated and correct version of this paper is available at arXiv:2603.00576, Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation. Please disregard this version
>
> **摘要:** Recent advancements in diffusion models have significantly improved symbolic music generation. However, most approaches rely on transformer-based architectures with self-attention mechanisms, which are constrained by quadratic computational complexity, limiting scalability for long sequences. To address this, we propose Symbolic Music Diffusion with Mamba (SMDIM), a novel diffusion-based architecture integrating Structured State Space Models (SSMs) for efficient global context modeling and the Mamba-FeedForward-Attention Block (MFA) for precise local detail preservation. The MFA Block combines the linear complexity of Mamba layers, the non-linear refinement of FeedForward layers, and the fine-grained precision of self-attention mechanisms, achieving a balance between scalability and musical expressiveness. SMDIM achieves near-linear complexity, making it highly efficient for long-sequence tasks. Evaluated on diverse datasets, including FolkDB, a collection of traditional Chinese folk music that represents an underexplored domain in symbolic music generation, SMDIM outperforms state-of-the-art models in both generation quality and computational efficiency. Beyond symbolic music, SMDIM's architectural design demonstrates adaptability to a broad range of long-sequence generation tasks, offering a scalable and efficient solution for coherent sequence modeling.
>
---
#### [replaced 008] CodecFlow: Efficient Bandwidth Extension via Conditional Flow Matching in Neural Codec Latent Space
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音带宽扩展任务，旨在提升低带宽语音的清晰度和可懂度。提出CodecFlow框架，在神经编解码器潜在空间中高效重建语音，解决高频频段恢复难题。**

- **链接: [https://arxiv.org/pdf/2603.02022](https://arxiv.org/pdf/2603.02022)**

> **作者:** Bowen Zhang; Junchuan Zhao; Ian McLoughlin; Ye Wang; A S Madhukumar
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Speech Bandwidth Extension improves clarity and intelligibility by restoring/inferring appropriate high-frequency content for low-bandwidth speech. Existing methods often rely on spectrogram or waveform modeling, which can incur higher computational cost and have limited high-frequency fidelity. Neural audio codecs offer compact latent representations that better preserve acoustic detail, yet accurately recovering high-resolution latent information remains challenging due to representation mismatch. We present CodecFlow, a neural codec-based BWE framework that performs efficient speech reconstruction in a compact latent space. CodecFlow employs a voicing-aware conditional flow converter on continuous codec embeddings and a structure-constrained residual vector quantizer to improve latent alignment stability. Optimized end-to-end, CodecFlow achieves strong spectral fidelity and enhanced perceptual quality on 8 kHz to 16 kHz and 44.1 kHz speech BWE tasks.
>
---
#### [replaced 009] Using Songs to Improve Kazakh Automatic Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于自动语音识别任务，旨在解决低资源语言Kazakh ASR数据不足的问题。通过使用歌曲数据进行模型微调，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2603.00961](https://arxiv.org/pdf/2603.00961)**

> **作者:** Rustem Yeshpanov
>
> **备注:** 9 pages, 7 tables, to appear in Proceedings of the 2026 Language Resources and Evaluation Conference
>
> **摘要:** Developing automatic speech recognition (ASR) systems for low-resource languages is hindered by the scarcity of transcribed corpora. This proof-of-concept study explores songs as an unconventional yet promising data source for Kazakh ASR. We curate a dataset of 3,013 audio-text pairs (about 4.5 hours) from 195 songs by 36 artists, segmented at the lyric-line level. Using Whisper as the base recogniser, we fine-tune models under seven training scenarios involving Songs, Common Voice Corpus (CVC), and FLEURS, and evaluate them on three benchmarks: CVC, FLEURS, and Kazakh Speech Corpus 2 (KSC2). Results show that song-based fine-tuning improves performance over zero-shot baselines. For instance, Whisper Large-V3 Turbo trained on a mixture of Songs, CVC, and FLEURS achieves 27.6% normalised WER on CVC and 11.8% on FLEURS, while halving the error on KSC2 (39.3% vs. 81.2%) relative to the zero-shot model. Although these gains remain below those of models trained on the 1,100-hour KSC2 corpus, they demonstrate that even modest song-speech mixtures can yield meaningful adaptation improvements in low-resource ASR. The dataset is released on Hugging Face for research purposes under a gated, non-commercial licence.
>
---
