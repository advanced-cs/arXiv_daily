# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] FastWave: Optimized Diffusion Model for Audio Super-Resolution
- **分类: cs.SD; cs.LG**

- **简介: 论文提出FastWave，解决音频超分辨率问题，通过优化扩散模型，在保持高质量的同时降低计算成本，比现有方法更高效。**

- **链接: [https://arxiv.org/pdf/2603.04122](https://arxiv.org/pdf/2603.04122)**

> **作者:** Nikita Kuznetsov; Maksim Kaledin
>
> **摘要:** Audio Super-Resolution is a set of techniques aimed at high-quality estimation of the given signal as if it would be sampled with higher sample rate. Among suggested methods there are diffusion and flow models (which are considered slower), generative adversarial networks (which are considered faster), however both approaches are currently presented by high-parametric networks, requiring high computational costs both for training and inference. We propose a solution to both these problems by re-considering the recent advances in the training of diffusion models and applying them to super-resolution from any to 48 kHz sample rate. Our approach shows better results than NU-Wave 2 and is comparable to state-of-the-art models. Our model called FastWave has around 50 GFLOPs of computational complexity and 1.3 M parameters and can be trained with less resources and significantly faster than the majority of recently proposed diffusion- and flow-based solutions. The code has been made publicly available.
>
---
#### [new 002] FlowW2N: Whispered-to-Normal Speech Conversion via Flow-Matching
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音转换任务，解决从耳语到正常语音的转换问题。通过流匹配方法，利用合成数据训练，无需真实配对数据，提升语音可懂度。**

- **链接: [https://arxiv.org/pdf/2603.04296](https://arxiv.org/pdf/2603.04296)**

> **作者:** Fabian Ritter-Gutierrez; Md Asif Jalal; Pablo Peso Parada; Karthikeyan Saravanan; Yusun Shul; Minseung Kim; Gun-Woo Lee; Han-Gil Moon
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Whispered-to-normal (W2N) speech conversion aims to reconstruct missing phonation from whispered input while preserving content and speaker identity. This task is challenging due to temporal misalignment between whisper and voiced recordings and lack of paired data. We propose FlowW2N, a conditional flow matching approach that trains exclusively on synthetic, time-aligned whisper-normal pairs and conditions on domain-invariant features. We exploit high-level ASR embeddings that exhibits strong invariance between synthetic and real whispered speech, enabling generalization to real whispers despite never observing it during training. We verify this invariance across ASR layers and propose a selection criterion optimizing content informativeness and cross-domain invariance. Our method achieves SOTA intelligibility on the CHAINS and wTIMIT datasets, reducing Word Error Rate by 26-46% relative to prior work while using only 10 steps at inference and requiring no real paired data.
>
---
#### [new 003] The PARLO Dementia Corpus: A German Multi-Center Resource for Alzheimer's Disease
- **分类: eess.AS**

- **简介: 该论文介绍PARLO Dementia Corpus，用于阿尔茨海默病的语音分析研究。任务是开发非侵入性诊断方法，解决语言数据不足的问题。工作包括收集多中心语音数据并进行标注。**

- **链接: [https://arxiv.org/pdf/2603.03471](https://arxiv.org/pdf/2603.03471)**

> **作者:** Franziska Braun; Christopher Witzl; Florian Hönig; Elmar Nöth; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Early and accessible detection of Alzheimer's disease (AD) remains a major challenge, as current diagnostic methods often rely on costly and invasive biomarkers. Speech and language analysis has emerged as a promising non-invasive and scalable approach to detecting cognitive impairment, but research in this area is hindered by the lack of publicly available datasets, especially for languages other than English. This paper introduces the PARLO Dementia Corpus (PDC), a new multi-center, clinically validated German resource for AD collected across nine academic memory clinics in Germany. The dataset comprises speech recordings from individuals with AD-related mild cognitive impairment and mild to moderate dementia, as well as cognitively healthy controls. Speech was elicited using a standardized test battery of eight neuropsychological tasks, including confrontation naming, verbal fluency, word repetition, picture description, story reading, and recall tasks. In addition to audio recordings, the dataset includes manually verified transcriptions and detailed demographic, clinical, and biomarker metadata. Baseline experiments on ASR benchmarking, automated test evaluation, and LLM-based classification illustrate the feasibility of automatic, speech-based cognitive assessment and highlight the diagnostic value of recall-driven speech production. The PDC thus establishes the first publicly available German benchmark for multi-modal and cross-lingual research on neurodegenerative diseases.
>
---
#### [new 004] Multi-Stage Music Source Restoration with BandSplit-RoFormer Separation and HiFi++ GAN
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在从混音中恢复原始乐器音轨。通过分阶段的分离与波形修复方法解决非线性混合问题。**

- **链接: [https://arxiv.org/pdf/2603.04032](https://arxiv.org/pdf/2603.04032)**

> **作者:** Tobias Morocutti; Emmanouil Karystinaios; Jonathan Greif; Gerhard Widmer
>
> **备注:** ICASSP 2026 Music Source Restoration (MSR) Challenge
>
> **摘要:** Music Source Restoration (MSR) targets recovery of original, unprocessed instrument stems from fully mixed and mastered audio, where production effects and distribution artifacts violate common linear-mixture assumptions. This technical report presents the CP-JKU team's system for the MSR ICASSP Challenge 2025. Our approach decomposes MSR into separation and restoration. First, a single BandSplit-RoFormer separator predicts eight stems plus an auxiliary other stem, and is trained with a three-stage curriculum that progresses from 4-stem warm-start fine-tuning (with LoRA) to 8-stem extension via head expansion. Second, we apply a HiFi++ GAN waveform restorer trained as a generalist and then specialized into eight instrument-specific experts.
>
---
#### [new 005] Cyclostationarity Analysis as a Complement to Self-Supervised Representations for Speech Deepfake Detection
- **分类: eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在提升检测效果。通过引入基于谱相关密度的周期性特征，补充自监督学习表示，有效提升了检测性能。**

- **链接: [https://arxiv.org/pdf/2603.03921](https://arxiv.org/pdf/2603.03921)**

> **作者:** Cemal Hanilçi; Md Sahidullah; Tomi Kinnunen
>
> **备注:** submitted to IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Speech deepfake detection (SDD) is essential for maintaining trust in voice-driven technologies and digital media. Although recent SDD systems increasingly rely on self-supervised learning (SSL) representations that capture rich contextual information, complementary signal-driven acoustic features remain important for modeling fine-grained structural properties of speech. Most existing acoustic front ends are based on time-frequency representations, which do not fully exploit higher-order spectral dependencies inherent in speech signals. We introduce a cyclostationarity-inspired acoustic feature extraction framework for SDD based on spectral correlation density (SCD). The proposed features model periodic statistical structures in speech by capturing spectral correlations between frequency components. In particular, we propose temporally structured SCD features that characterize the evolution of spectral and cyclic-frequency components over time. The effectiveness and complementarity of the proposed features are evaluated using multiple countermeasure architectures, including convolutional neural networks, SSL-based embedding systems, and hybrid fusion models. Experiments on ASVspoof 2019 LA, ASVspoof 2021 DF, and ASVspoof 5 demonstrate that SCD-based features provide complementary discriminative information to SSL embeddings and conventional acoustic representations. In particular, fusion of SSL and SCD embeddings reduces the equal error rate on ASVspoof 2019 LA from $8.28\%$ to $0.98\%$, and yields consistent improvements on the challenging ASVspoof 5 dataset. The results highlight cyclostationary signal analysis as a theoretically grounded and effective front end for speech deepfake detection.
>
---
#### [new 006] ACES: Accent Subspaces for Coupling, Explanations, and Stress-Testing in Automatic Speech Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决不同口音导致的性能差异问题。通过提取口音特征子空间，分析模型脆弱性和差异性，揭示口音信息与识别关键线索的深度关联。**

- **链接: [https://arxiv.org/pdf/2603.03359](https://arxiv.org/pdf/2603.03359)**

> **作者:** Swapnil Parekh
>
> **摘要:** ASR systems exhibit persistent performance disparities across accents, yet the internal mechanisms underlying these gaps remain poorly understood. We introduce ACES, a representation-centric audit that extracts accent-discriminative subspaces and uses them to probe model fragility and disparity. Analyzing Wav2Vec2-base with five English accents, we find that accent information concentrates in a low-dimensional early-layer subspace (layer 3, k=8). Projection magnitude correlates with per-utterance WER (r=0.26), and crucially, subspace-constrained perturbations yield stronger coupling between representation shift and degradation (r=0.32) than random-subspace controls (r=0.15). Finally, linear attenuation of this subspace however does not reduce disparity and slightly worsens it. Our findings suggest that accent-relevant features are deeply entangled with recognition-critical cues, positioning accent subspaces as vital diagnostic tools rather than simple "erasure" levers for fairness.
>
---
#### [new 007] ZeSTA: Zero-Shot TTS Augmentation with Domain-Conditioned Training for Data-Efficient Personalized Speech Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，解决低资源个性化语音合成中数据不足的问题。通过引入零样本文本转语音作为数据增强，并采用领域条件训练框架提升说话人相似度。**

- **链接: [https://arxiv.org/pdf/2603.04219](https://arxiv.org/pdf/2603.04219)**

> **作者:** Youngwon Choi; Jinwoo Oh; Hwayeon Kim; Hyeonyu Kim
>
> **备注:** 6 pages, submitted to INTERSPEECH 2026
>
> **摘要:** We investigate the use of zero-shot text-to-speech (ZS-TTS) as a data augmentation source for low-resource personalized speech synthesis. While synthetic augmentation can provide linguistically rich and phonetically diverse speech, naively mixing large amounts of synthetic speech with limited real recordings often leads to speaker similarity degradation during fine-tuning. To address this issue, we propose ZeSTA, a simple domain-conditioned training framework that distinguishes real and synthetic speech via a lightweight domain embedding, combined with real-data oversampling to stabilize adaptation under extremely limited target data, without modifying the base architecture. Experiments on LibriTTS and an in-house dataset with two ZS-TTS sources demonstrate that our approach improves speaker similarity over naive synthetic augmentation while preserving intelligibility and perceptual quality.
>
---
#### [new 008] A Sensitivity Analysis of Multi-Event Audio Grounding in Audio LLMs
- **分类: cs.SD**

- **简介: 该论文属于音频事件定位任务，旨在评估音频大模型在复杂声景中的可靠性。通过大规模实验分析多事件场景下的准确率与误报率，揭示模型不确定性，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2603.03855](https://arxiv.org/pdf/2603.03855)**

> **作者:** Taehan Lee; Jaehan Jung; Hyukjun Lee
>
> **备注:** 6 pages, Submitted to Interspeech 2026
>
> **摘要:** Audio LLMs have shown a strong ability to understand audio samples, yet their reliability in complex acoustic scenes remains under-explored. Unlike prior work limited to small scale or less controlled query construction, we present a large-scale evaluation of event grounding and false alarms as auditory scene complexity increases. Using 71K AudioCapsV2 clips, we extract normalized (source, attribute) events and build two query types: present-event queries for ground-truth detection and absent-event queries to probe hallucinations, using similarity-filtered negative sampling in an audio-aligned text embedding space. We evaluate four SOTA Audio LLMs with 12 prompt variants over 500K yes/no queries per model. Across models, increasing event count consistently lowers true-positive rate and raises false-positive rate, while prompts induce a strong trade-off between the two. Our confidence analysis shows that models become more uncertain on multi-event audio, revealing room for improvement.
>
---
#### [new 009] Low-Resource Guidance for Controllable Latent Audio Diffusion
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于音频生成任务，旨在解决可控生成中计算成本高的问题。通过引入LatCHs，实现低资源的潜在空间控制，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.04366](https://arxiv.org/pdf/2603.04366)**

> **作者:** Zachary Novack; Zack Zukowski; CJ Carr; Julian Parker; Zach Evans; Josiah Taylor; Taylor Berg-Kirkpatrick; Julian McAuley; Jordi Pons
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Generative audio requires fine-grained controllable outputs, yet most existing methods require model retraining on specific controls or inference-time controls (\textit{e.g.}, guidance) that can also be computationally demanding. By examining the bottlenecks of existing guidance-based controls, in particular their high cost-per-step due to decoder backpropagation, we introduce a guidance-based approach through selective TFG and Latent-Control Heads (LatCHs), which enables controlling latent audio diffusion models with low computational overhead. LatCHs operate directly in latent space, avoiding the expensive decoder step, and requiring minimal training resources (7M parameters and $\approx$ 4 hours of training). Experiments with Stable Audio Open demonstrate effective control over intensity, pitch, and beats (and a combination of those) while maintaining generation quality. Our method balances precision and audio fidelity with far lower computational costs than standard end-to-end guidance. Demo examples can be found at this https URL.
>
---
#### [new 010] LabelBuddy: An Open Source Music and Audio Language Annotation Tagging Tool Using AI Assistance
- **分类: cs.SD; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于音乐信息检索任务，旨在解决音频标注主观性与机器理解不匹配的问题。提出LabelBuddy工具，支持AI辅助的动态音频标注。**

- **链接: [https://arxiv.org/pdf/2603.04293](https://arxiv.org/pdf/2603.04293)**

> **作者:** Ioannis Prokopiou; Ioannis Sina; Agisilaos Kounelis; Pantelis Vikatos; Themos Stafylakis
>
> **备注:** Accepted at NLP4MusA 2026 (4th Workshop on NLP for Music and Audio)
>
> **摘要:** The advancement of Machine learning (ML), Large Audio Language Models (LALMs), and autonomous AI agents in Music Information Retrieval (MIR) necessitates a shift from static tagging to rich, human-aligned representation learning. However, the scarcity of open-source infrastructure capable of capturing the subjective nuances of audio annotation remains a critical bottleneck. This paper introduces \textbf{LabelBuddy}, an open-source collaborative auto-tagging audio annotation tool designed to bridge the gap between human intent and machine understanding. Unlike static tools, it decouples the interface from inference via containerized backends, allowing users to plug in custom models for AI-assisted pre-annotation. We describe the system architecture, which supports multi-user consensus, containerized model isolation, and a roadmap for extending agents and LALMs. Code available at this https URL.
>
---
#### [new 011] Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音频-视觉语音识别任务，旨在提升恶劣环境下的识别鲁棒性。通过稀疏模态对齐和视觉单元引导优化，改进了LLM在AVSR中的性能。**

- **链接: [https://arxiv.org/pdf/2603.03811](https://arxiv.org/pdf/2603.03811)**

> **作者:** Fei Su; Cancan Li; Juan Liu; Wei Ju; Hongbin Suo; Ming Li
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) integrates acoustic and visual information to enhance robustness in adverse acoustic conditions. Recent advances in Large Language Models (LLMs) have yielded competitive automatic speech recognition performance and shown effectiveness for AVSR. However, prior approaches project audio and visual features independently or apply shallow fusion, limiting cross-modal alignment and complementary exchange while increasing the LLM's computational load. To address this, we propose AVUR-LLM, an LLM-based Audio-Visual Speech Recognition via Sparse Modality Alignment and Visual Unit-Guided Refinement. Experiments on LRS3 demonstrate state-of-the-art results for AVSR. Under additive-noise conditions at 0 dB SNR, it achieves 37% relative improvement over the baseline system.
>
---
#### [new 012] Automated Measurement of Geniohyoid Muscle Thickness During Speech Using Deep Learning and Ultrasound
- **分类: q-bio.QM; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出SMMA框架，用于自动化测量发音时颏舌肌厚度，解决手动测量耗时、限制大规模研究的问题。通过深度学习与骨架分析实现精准量化，揭示发音差异及性别差异。**

- **链接: [https://arxiv.org/pdf/2603.03350](https://arxiv.org/pdf/2603.03350)**

> **作者:** Alisher Myrgyyassov; Bruce Xiao Wang; Yu Sun; Shuming Huang; Zhen Song; Min Ney Wong; Yongping Zheng
>
> **备注:** 6 pages, including references and acknowledgements. Submitted to Interspeech 2026
>
> **摘要:** Manual measurement of muscle morphology from ultrasound during speech is time-consuming and limits large-scale studies. We present SMMA, a fully automated framework that combines deep-learning segmentation with skeleton-based thickness quantification to analyze geniohyoid (GH) muscle dynamics. Validation demonstrates near-human-level accuracy (Dice = 0.9037, MAE = 0.53 mm, r = 0.901). Application to Cantonese vowel production (N = 11) reveals systematic patterns: /a:/ shows significantly greater GH thickness (7.29 mm) than /i:/ (5.95 mm, p < 0.001, Cohen's d > 1.3), suggesting greater GH activation during production of /a:/ than /i:/, consistent with its role in mandibular depression. Sex differences (5-8% greater in males) reflect anatomical scaling. SMMA achieves expert-validated accuracy while eliminating the need for manual annotation, enabling scalable investigations of speech motor control and objective assessment of speech and swallowing disorders.
>
---
#### [new 013] Escaping the BLEU Trap: A Signal-Grounded Framework with Decoupled Semantic Guidance for EEG-to-Text Decoding
- **分类: cs.CL; cs.AI; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文属于EEG-to-Text解码任务，旨在解决语义偏差、信号忽视和BLEU陷阱问题。提出SemKey框架，通过分离语义目标和强化信号依赖，提升生成质量与真实性。**

- **链接: [https://arxiv.org/pdf/2603.03312](https://arxiv.org/pdf/2603.03312)**

> **作者:** Yuchen Wang; Haonan Wang; Yu Guo; Honglong Yang; Xiaomeng Li
>
> **摘要:** Decoding natural language from non-invasive EEG signals is a promising yet challenging task. However, current state-of-the-art models remain constrained by three fundamental limitations: Semantic Bias (mode collapse into generic templates), Signal Neglect (hallucination based on linguistic priors rather than neural inputs), and the BLEU Trap, where evaluation metrics are artificially inflated by high-frequency stopwords, masking a lack of true semantic fidelity. To address these challenges, we propose SemKey, a novel multi-stage framework that enforces signal-grounded generation through four decoupled semantic objectives: sentiment, topic, length, and surprisal. We redesign the interaction between the neural encoder and the Large Language Model (LLM) by injecting semantic prompts as Queries and EEG embeddings as Key-Value pairs, strictly forcing the model to attend to neural inputs. Furthermore, we move beyond standard translation metrics by adopting N-way Retrieval Accuracy and Fréchet Distance to rigorously assess diversity and alignment. Extensive experiments demonstrate that our approach effectively eliminates hallucinations on noise inputs and achieves SOTA performance on these robust protocols. Code will be released upon acceptance at this https URL.
>
---
## 更新

#### [replaced 001] Benchmarking Speech Systems for Frontline Health Conversations: The DISPLACE-M Challenge
- **分类: eess.AS**

- **简介: 该论文属于医疗对话理解任务，旨在解决多语种、多说话人真实医疗对话的分析问题。工作包括构建数据集和提供基准系统，涵盖语音识别、话题识别等四个任务。**

- **链接: [https://arxiv.org/pdf/2603.02813](https://arxiv.org/pdf/2603.02813)**

> **作者:** Dhanya E; Ankita Meena; Manas Nanivadekar; Noumida A; Victor Azad; Ashwini Nagaraj Shenoy; Pratik Roy Chowdhuri; Shobhit Banga; Vanshika Chhabra; Chitralekha Bhat; Shareef babu Kalluri; Srikanth Raj Chetupalli; Deepu Vijayasenan; Sriram Ganapathy
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** The DIarization and Speech Processing for LAnguage understanding in Conversational Environments - Medical (DISPLACE-M) challenge introduces a conversational AI benchmark focused on understanding goal-oriented, real-world medical dialogues collected in the field. The challenge addresses multi-speaker interactions between healthcare workers and seekers characterized by spontaneous, noisy and overlapping speech across Indian languages and dialects. As part of the challenge, medical conversational dataset comprising 25 hours of development data and 10 hours of blind evaluation recordings was released. We provided baseline systems within a unified end-to-end pipeline across 4 tasks - speaker diarization, automatic speech recognition, topic identification and dialogue summarization - to enable consistent benchmarking. System performance is evaluated using established metrics such as diarization error rate (DER), time-constrained minimum-permutation word error rate (tcpWER), and ROUGE-L. During this evaluation (Phase-I), 12 teams, across the globe, actively participated pushing the baseline systems on these metrics. However, even with a 6-8 week dedicated effort from various participants, the task is shown to be substantially challenging, and the existing systems are significantly short of healthcare deployment readiness.
>
---
#### [replaced 002] CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成模型的评估任务，旨在解决音乐奖励模型评价机制不足的问题。提出CMI-RewardBench基准和相关模型，提升音乐生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.00610](https://arxiv.org/pdf/2603.00610)**

> **作者:** Yinghao Ma; Haiwen Xia; Hewei Gao; Weixiong Chen; Yuxin Ye; Yuchen Yang; Sungkyun Chang; Mingshuo Ding; Yizhi Li; Ruibin Yuan; Simon Dixon; Emmanouil Benetos
>
> **摘要:** While music generation models have evolved to handle complex multimodal inputs mixing text, lyrics, and reference audio, evaluation mechanisms have lagged behind. In this paper, we bridge this critical gap by establishing a comprehensive ecosystem for music reward modeling under Compositional Multimodal Instruction (CMI), where the generated music may be conditioned on text descriptions, lyrics, and audio prompts. We first introduce CMI-Pref-Pseudo, a large-scale preference dataset comprising 110k pseudo-labeled samples, and CMI-Pref, a high-quality, human-annotated corpus tailored for fine-grained alignment tasks. To unify the evaluation landscape, we propose CMI-RewardBench, a unified benchmark that evaluates music reward models on heterogeneous samples across musicality, text-music alignment, and compositional instruction alignment. Leveraging these resources, we develop CMI reward models (CMI-RMs), a parameter-efficient reward model family capable of processing heterogeneous inputs. We evaluate their correlation with human judgments scores on musicality and alignment on CMI-Pref along with previous datasets. Further experiments demonstrate that CMI-RM not only correlates strongly with human judgments, but also enables effective inference-time scaling via top-k filtering. The necessary training data, benchmarks, and reward models are publicly available.
>
---
#### [replaced 003] Knowing When to Quit: Probabilistic Early Exits for Speech Separation
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音分离任务，旨在解决模型在不同计算资源下的适应性问题。通过设计具有早停机制的神经网络和概率框架，实现动态计算优化，提升效率。**

- **链接: [https://arxiv.org/pdf/2507.09768](https://arxiv.org/pdf/2507.09768)**

> **作者:** Kenny Falkær Olsen; Mads Østergaard; Karl Ulbæk; Søren Føns Nielsen; Rasmus Malik Høegh Lindrup; Bjørn Sand Jensen; Morten Mørup
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** In recent years, deep learning-based single-channel speech separation has improved considerably, in large part driven by increasingly compute- and parameter-efficient neural network architectures. Most such architectures are, however, designed with a fixed compute and parameter budget and consequently cannot scale to varying compute demands or resources, which limits their use in embedded and heterogeneous devices such as mobile phones and hearables. To enable such use-cases we design a neural network architecture for speech separation and enhancement capable of early-exit, and we propose an uncertainty-aware probabilistic framework to jointly model the clean speech signal and error variance which we use to derive probabilistic early-exit conditions in terms of desired signal-to-noise ratios. We evaluate our methods on both speech separation and enhancement tasks where we demonstrate that early-exit capabilities can be introduced without compromising reconstruction, and that when trained on variable-length audio our early-exit conditions are well-calibrated and lead to considerable compute savings when used to dynamically scale compute at test time while remaining directly interpretable.
>
---
#### [replaced 004] Better audio representations are more brain-like: linking model-brain alignment with performance in downstream auditory tasks
- **分类: cs.LG; cs.SD**

- **简介: 该论文研究音频模型与大脑活动的对齐问题，旨在探讨模型性能提升是否使表示更接近脑信号。通过分析36个模型和fMRI数据，发现高性能模型更预测听觉皮层活动，且任务表现与脑对齐度正相关。**

- **链接: [https://arxiv.org/pdf/2511.16849](https://arxiv.org/pdf/2511.16849)**

> **作者:** Leonardo Pepino; Pablo Riera; Juan Kamienkowski; Luciana Ferrer
>
> **备注:** In review for journal
>
> **摘要:** Artificial neural networks are increasingly powerful models of brain computation, yet it remains unclear whether improving their performance in downstream tasks also makes their internal representations more similar to brain signals. To address this question in the auditory domain, we quantified the alignment between the internal representations of 36 different audio models and brain activity from two independent fMRI datasets. Using voxel-wise and component-wise regression, and representation similarity analysis, we found that recent self-supervised audio models with strong performance in diverse downstream tasks are better predictors of auditory cortex activity than previously studied models. To assess the quality of the audio representations, we evaluated these models in 6 auditory tasks from the HEAREval benchmark, spanning music, speech, and environmental sounds. This revealed strong positive Pearson correlations (r > 0.8) between a model's overall task performance and its alignment with brain representations. Finally, we analyzed the evolution of the similarity between audio and brain representations during the pretraining of EnCodecMAE, a recent audio representation model. We discovered that brain similarity increases progressively and emerges early during pretraining, despite the model not being explicitly optimized for this objective. This suggests that brain-like representations can be an emergent byproduct of learning to reconstruct missing information from naturalistic audio data.
>
---
#### [replaced 005] MeanFlowSE: one-step generative speech enhancement via conditional mean flow
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出MeanFlowSE，用于实时语音增强任务，解决多步推理效率低的问题，通过单步生成提升计算效率。**

- **链接: [https://arxiv.org/pdf/2509.14858](https://arxiv.org/pdf/2509.14858)**

> **作者:** Duojia Li; Shenghui Lu; Hongchen Pan; Zongyi Zhan; Qingyang Hong; Lin Li
>
> **摘要:** Multistep inference is a bottleneck for real-time generative speech enhancement because flow- and diffusion-based systems learn an instantaneous velocity field and therefore rely on iterative ordinary differential equation (ODE) solvers. We introduce MeanFlowSE, a conditional generative model that learns the average velocity over finite intervals along a trajectory. Using a Jacobian-vector product (JVP) to instantiate the MeanFlow identity, we derive a local training objective that directly supervises finite-interval displacement while remaining consistent with the instantaneous-field constraint on the diagonal. At inference, MeanFlowSE performs single-step generation via a backward-in-time displacement, removing the need for multistep solvers; an optional few-step variant offers additional refinement. On VoiceBank-DEMAND, the single-step model achieves strong intelligibility, fidelity, and perceptual quality with substantially lower computational cost than multistep baselines. The method requires no knowledge distillation or external teachers, providing an efficient, high-fidelity framework for real-time generative speech enhancement. The proposed method is open-sourced at this https URL.
>
---
#### [replaced 006] LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出LadderSym，用于音乐练习错误检测的多模态Transformer模型，解决音频与乐谱对齐及并发音符歧义问题，提升错误检测效果。**

- **链接: [https://arxiv.org/pdf/2510.08580](https://arxiv.org/pdf/2510.08580)**

> **作者:** Benjamin Shiue-Hal Chou; Purvish Jajal; Nick John Eliopoulos; James C. Davis; George K. Thiruvathukal; Kristen Yeon-Ji Yun; Yung-Hsiang Lu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Music learners can greatly benefit from tools that accurately detect errors in their practice. Existing approaches typically compare audio recordings to music scores using heuristics or learnable models. This paper introduces LadderSym, a novel Transformer-based method for music error detection. LadderSym is guided by two key observations about the state-of-the-art approaches: (1) late fusion limits inter-stream alignment and cross-modality comparison capability; and (2) reliance on score audio introduces ambiguity in the frequency spectrum, degrading performance in music with concurrent notes. To address these limitations, LadderSym introduces (1) a two-stream encoder with inter-stream alignment modules to improve audio comparison capabilities and error detection F1 scores, and (2) a multimodal strategy that leverages both audio and symbolic scores by incorporating symbolic representations as decoder prompts, reducing ambiguity and improving F1 scores. We evaluate our method on the MAESTRO-E and CocoChorales-E datasets by measuring the F1 score for each note category. Compared to the previous state of the art, LadderSym more than doubles F1 for missed notes on MAESTRO-E (26.8% -> 56.3%) and improves extra note detection by 14.4 points (72.0% -> 86.4%). Similar gains are observed on CocoChorales-E. Furthermore, we also evaluate our models on real data we curated. This work introduces insights about comparison models that could inform sequence evaluation tasks for reinforcement learning, human skill assessment, and model evaluation. Code: this https URL
>
---
#### [replaced 007] SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations
- **分类: eess.AS**

- **简介: 该论文提出SPEAR框架，解决语音与音频表示学习的领域差异问题。通过融合两种教师模型，提升模型在复杂声景下的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.25955](https://arxiv.org/pdf/2510.25955)**

> **作者:** Xiaoyu Yang; Yifan Yang; Zengrui Jin; Ziyun Cui; Wen Wu; Baoxiang Li; Chao Zhang; Phil Woodland
>
> **备注:** Preprint. Under review
>
> **摘要:** Self-supervised learning (SSL) has significantly advanced acoustic representation learning. However, most existing models are optimised for either speech or audio event understanding, resulting in a persistent gap between these two domains. We address this gap with SPEAR (SPEech and Audio Representations), a self-supervised framework that distils complementary knowledge from a speech-focused SSL teacher and a general-audio SSL teacher into a single unified model. SPEAR applies multi-codebook vector quantisation to continuous teacher representations to produce fine-grained discrete tokens that capture both semantic and acoustic information. To effectively integrate these heterogeneous representations, SPEAR jointly predicts them given a masked input with an asymmetric pre-training loss. We further improve robustness in complex sound scenes through a novel token mixing mechanism. Extensive experiments demonstrate that SPEAR consistently outperforms existing unified speech and audio models. SPEAR establishes a new state-of-the-art on the SUPERB benchmark, surpassing WavLM Large on 12 of 15 tasks, while achieving competitive performance on the HEAR benchmark. These results position SPEAR as a versatile foundation for general-purpose speech and audio representation learning. The code and pre-trained models will be released.
>
---
#### [replaced 008] OASI: Objective-Aware Surrogate Initialization for Multi-Objective Bayesian Optimization in TinyML Keyword Spotting
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于TinyML任务，解决KWS模型在资源受限下的多目标优化问题。提出OASI方法，通过多目标模拟退火生成初始解，提升优化效果和可行性。**

- **链接: [https://arxiv.org/pdf/2512.19739](https://arxiv.org/pdf/2512.19739)**

> **作者:** Soumen Garai; Danilo Pau; Suman Samui
>
> **备注:** Updated version
>
> **摘要:** Voice-triggered interfaces rely on keyword spotting (KWS) models that must operate continuously under strict memory, latency, and energy constraints on microcontroller-class hardware. Designing such models therefore requires not only high recognition accuracy but also predictable deployability within limited Flash and SRAM budgets. Bayesian optimization is known to handle accuracy-efficiency trade-offs effectively in multi-objective optimization; however, it is highly sensitive to initialization, particularly in the low-budget regimes of TinyML model optimization. We propose Objective-Aware Surrogate Initialization (OASI), which seeds surrogate optimization with Pareto-biased solutions generated via multi-objective simulated annealing. Unlike space-filling or heuristic warm-start methods, OASI initializes the surrogate conditioning process with a bias toward feasible accuracy-memory trade-offs, thus avoiding SRAM-violating configurations. OASI improves hypervolume and convergence robustness over Latin hypercube, Sobol, and random initializations under the same budget constraints on a TinyML KWS problem. Hardware-in-the-loop experiments on STM32 microcontrollers verify the existence of deployable and memory-feasible models without incurring extra optimization costs.
>
---
