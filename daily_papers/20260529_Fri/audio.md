# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] Audio Deepfake Detection with Half-Truth Localisation Using Cross-Attentive Feature Fusion
- **分类: cs.SD; cs.CV; cs.LG**

- **简介: 该论文属于音频深度伪造检测任务，旨在识别部分篡改音频并定位篡改位置。提出CAFNet模型，通过特征融合与注意力机制实现分类与定位，效果优于现有方法。**

- **链接: [https://arxiv.org/pdf/2605.29531](https://arxiv.org/pdf/2605.29531)**

> **作者:** S. Sutharya; Remya K. Sasi
>
> **备注:** 13 pages, 5 figures, 11 tables
>
> **摘要:** Audio deepfake detection is well-studied as a binary problem, but partially manipulated speech, where a short synthesised segment is spliced into an otherwise genuine utterance, poses a harder and more realistic threat. Detecting such half-truth audio requires not only distinguishing it from real and fully fake speech, but also localising where the manipulation occurs. We present CAFNet, a 576k-parameter architecture that addresses both tasks jointly: it performs ternary classification (real, fully-fake, or half-truth) and regresses the temporal boundaries of the synthesised region in a single forward pass. CAFNet fuses Mel-Frequency Cepstral Coefficient (MFCC), Linear-Frequency Cepstral Coefficient (LFCC), and Chroma Short-Time Fourier Transform (Chroma-STFT) features through parallel depthwise-separable convolution branches with cross-attention, followed by a Bidirectional Long Short-Term Memory (BiLSTM) regression head for boundary prediction. On the combined Multi-Lingual Audio Deepfake Detection Corpus (MLADDC) T2+T3 test set, CAFNet achieves 92.71% accuracy and macro Area Under the Curve (AUC) of 0.9910, with boundary localisation Mean Absolute Error (MAE) of 0.075s and a median error of 0.052s. On binary detection, it achieves 96.76% accuracy and 3.20% Equal Error Rate (EER), outperforming fine-tuned XLS-R 300M (78.31%) and AST 87M (93.03%) at over 500 times fewer parameters. A cross-dataset study further shows that standard fine-tuning collapses cross-domain representations even under reduced backbone learning rates.
>
---
#### [new 002] MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出MELD模型，解决语音语言建模中编码器与自回归模型分离导致的表示不优问题，通过联合优化提升TTS和STT性能。**

- **链接: [https://arxiv.org/pdf/2605.29859](https://arxiv.org/pdf/2605.29859)**

> **作者:** Sung-Lin Yeh; Wei Zhou; Gil Keren; Duc Le; Zhong Meng; Hao Tang; Jay Mahadeokar; Ozlem Kalinli; Alexandre Mourachko
>
> **摘要:** Recent speech language models rely on encoders that are optimized separately from autoregressive models. Since these encoders are unaware of the downstream objectives, the extracted representations may not be optimal for downstream tasks. To address this limitation, we introduce a discrete latent variable model on mel spectrograms that jointly optimizes the encoder and the speech language model. Joint optimization not only brings improvements over codec-based and other mel-spectrogram-based baselines on zero-shot Text-to-Speech (TTS) and Speech-to-Text (STT) tasks, but also effectively alleviates common issues in autoregressive mel-spectrogram modeling, such as prolonged silence generation and word omissions.
>
---
#### [new 003] The WER Trap: Shattering the Illusion of Unified Tokens in Speech Language Models
- **分类: eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决统一token的误解问题。指出低WER tokens无法保留合成所需细节，提出动态压缩方法并验证其缺陷。**

- **链接: [https://arxiv.org/pdf/2605.29209](https://arxiv.org/pdf/2605.29209)**

> **作者:** Xiangyu Zhang; Yuxin Li; Haoyang Zhang; Shiqi Han; Hexin Liu; Qiquan Zhang; Beena Ahmed; Julien Epps
>
> **摘要:** The pursuit of a "unified" discrete token for both speech understanding and generation has led the Speech Language Model (SLM) community to heavily rely on Word Error Rate (WER) -- the core metric for Whisper-style tokenizers -- as the definitive proxy for representation quality. This fosters the assumption that low-WER tokens inherently preserve the information necessary for intelligible acoustic synthesis. We argue this is fundamentally deceptive. While high-frequency tokens succeed in generation tasks due to implicit information leakage, isolating pure semantic information at ultra-low frame rates strips away the finegrained articulation and micro-dynamics essential for ODE-based generation. Empirically validating this requires extreme compression without sacrificing WER -- a methodological bottleneck, as standard fixed-stride downsampling arbitrarily truncates phonetic boundaries. To overcome this, we develop a dynamic compression tokenizer that intelligently aligns representations with semantic boundaries, achieving ultra-low frame rates with exceptionally low WER. Using these isolated "pure" semantic tokens, we expose the WER trap: when conditioning generative models -- even with oracle duration alignments -- the reconstructed speech suffers from severe articulation blur and is rendered acoustically unintelligible. Our findings demonstrate that semantic categorization rewarded by low WER is inherently orthogonal to the continuous phonetic trajectories required for synthesis, shattering the illusion of the unified token and advocating for explicitly decoupled speech representations.
>
---
#### [new 004] ChildVox: A Speech, Audio, and Large Audio-Language Model Benchmark in Understanding and Characterizing Sound across Childhood
- **分类: cs.SD**

- **简介: 该论文提出ChildVox，一个用于儿童声音理解的基准，涵盖从出生到学龄期的多种音频任务，解决儿童语音识别与分析问题，整合多个数据集并评估多种模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29257](https://arxiv.org/pdf/2605.29257)**

> **作者:** Tiantian Feng; Anfeng Xu; Xuan Shi; Aditya Kommineni; Shakhrul Iman Siam; Megan Micheletti; Zhonghao Shi; Helen Tager-Flusberg; Mi Zhang; Lynn K. Perry; Catherine Lord; Daniel Messinger; Shrikanth Narayanan
>
> **备注:** preprint under review
>
> **摘要:** We present ChildVox, a novel benchmark for characterizing the diverse acoustic signals through which children communicate. Specifically, ChildVox follows the full developmental trajectory from birth through school age, covering physiological sounds, non-linguistic vocalizations, canonical syllables, and spoken language. ChildVox integrates more than 20 sub-tasks across 17 child-centered audio and speech datasets, enabling systematic cross-corpus and cross-domain comparison. We evaluate a representative range of audio and speech foundation models, including self-supervised, ASR-oriented, and large audio-language models, on tasks including physiological sound classification, vocalization and canonical syllables modeling, and speech quality assessment and recognition. Benchmark results show that ChildVox provides a suite of high-performance models in recognizing a wide range of acoustic signals from children, supporting downstream applications such as characterizing children's language levels and tracking speech production with age.
>
---
#### [new 005] Mitigating Stethoscope-Induced Shortcuts in Respiratory Sound Classification under Federated Domain Generalization with Causality-Inspired Interventions
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于呼吸音分类任务，解决多设备部署中的设备差异问题。提出一种基于因果干预的联邦域泛化框架，提升模型在未知设备上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.29862](https://arxiv.org/pdf/2605.29862)**

> **作者:** Heejoon Koo; Yoon Tae Kim; Miika Toikkanen; June-Woo Kim
>
> **备注:** 2 figures, 4 tables, and 5 pages
>
> **摘要:** AI-driven respiratory sound classification (RSC) is promising for automated pulmonary disease detection, yet multi-site deployment is hindered by inter-stethoscope variability. We introduce a federated domain generalization (FedDG) formulation for RSC under stethoscope-induced device shifts, where clients use heterogeneous devices and the model is evaluated on unseen devices. Our empirical analysis shows that stethoscope-induced style and disease-specific content are tightly entangled, making deterministic style removal unreliable. In response, we propose a causality-inspired multimodal FedDG framework that combines: (i) a causality-inspired device style intervention network that performs content-preserving style perturbations, (ii) counterfactual text augmentation that neutralizes metadata shortcuts, and (iii) gradient alignment that facilitates device-invariant representations across clients. Built on a multimodal language-audio pretraining model, it outperforms conventional data augmentation and federated learning baselines in leave-one-device-out validation on ICBHI and SPRSound datasets. Code will be released upon publication.
>
---
#### [new 006] HoliTok:A Coutinuous Holistic Tokenization with Robust Dual Capabilities of Speech Generation and Understanding
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出HoliTok，一种用于统一语音生成与理解的连续整体分词模型。解决现有分词器难以同时满足语言模型学习和高质量波形解码的问题。通过优化分词策略，提升语音合成与识别效果。**

- **链接: [https://arxiv.org/pdf/2605.29948](https://arxiv.org/pdf/2605.29948)**

> **作者:** Bohan Li; Shi Lian; Hankun Wang; Yiwei Guo; Yu Xi; Zhihan Li; Da Zheng; Colin Zhang; Kai Yu
>
> **备注:** 14 pages, 2 figures, 8 tables
>
> **摘要:** Unified speech foundation models require a holistic tokenization space that is both learnable by language models and decodable into high-quality waveforms. Existing speech tokenizers, however, often fail to satisfy these requirements simultaneously, leading to increased architectural complexity and more involved training designs. We propose HoliTok, a continuous Holistic speech Tokenization model designed for unified generation-understanding modeling. HoliTok encodes 48~kHz speech into a compact 25~Hz sequence of 128-dimensional latents. It is trained with a progressive strategy that jointly preserves signal-level fidelity, incorporates semantic information, and maintains strong latent learnability. Based on this tokenization, we build a unified AR+DiT model for speech synthesis and recognition, where the same latent sequence supports both generation-specific and unified generation-understanding tasks. Experiments show that HoliTok achieves competitive reconstruction fidelity, improves generative learnability for high-quality and controllable synthesis, and, among the evaluated representations, is the only one that operates robustly in our unified generation-understanding architecture without additional optimization tricks. These results suggest that HoliTok serves as an effective speech tokenizer and a foundational representation interface for unified spoken language modeling. The code is available at: this https URL.
>
---
#### [new 007] Audio Jailbreaks in Large Audio-Language Models: Taxonomy, Attack-Defense Analysis, and Cost-Aware Evaluation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文研究大音频语言模型的越狱攻击与防御，属于安全评估任务。解决攻击方法与防御机制的比较问题，通过分类与实验分析，评估攻击效果与防御代价。**

- **链接: [https://arxiv.org/pdf/2605.30031](https://arxiv.org/pdf/2605.30031)**

> **作者:** Bo-Han Feng; Yu-Hsuan Li Liang; Chien-Feng Liu; You-Hsuan Chang; Yun-Nung Chen
>
> **备注:** Submitted to ACL ARR 2026 May
>
> **摘要:** Large Audio Language Models (LALMs) expand jailbreak risks from token-level prompting to the full speech perception-to-reasoning pipeline, where unsafe behavior can be induced through semantics, acoustic style, signal artifacts, or internal representations. Existing work studies these risks under heterogeneous threat models and evaluation protocols, making it difficult to compare attack practicality or defense utility. This paper provides a unified taxonomy and a controlled empirical evaluation of LALM jailbreak attacks and defenses. We organize prior work into semantic, acoustic, signal, and embedding-layer attacks; guard-based, training-free, and training-based defenses; and cross-modal, audio-native, and interactive benchmarks. We then evaluate representative attacks and defenses across ten open-source LALMs, measuring not only attack success rate but also benign refusal and latency. Our results show that Acoustic Best-of-N reveals strong worst-case audio-space vulnerabilities, Narrative Framing is an effective low-latency semantic threat, and current defenses trade robustness against benign usability. These findings support cost- and utility-aware evaluation as a necessary complement to success-rate-only LALM safety benchmarks.
>
---
#### [new 008] COMET: Concept Space Dissection of the Modality Gap in Audio-Text Multimodal Contrastive Embeddings
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于多模态学习任务，旨在解决音频与文本嵌入间的模态差距问题。通过概念空间分解，提出COMET框架，揭示模态差距来源并提出训练-free 的谱截断方法，提升零样本音频描述性能。**

- **链接: [https://arxiv.org/pdf/2605.29628](https://arxiv.org/pdf/2605.29628)**

> **作者:** Yonggang Zhu; Liting Gao; Aidong Men; Wenwu Wang
>
> **摘要:** Contrastive Language-Audio Pretraining (CLAP) models are widely used for audio understanding and support modality-agnostic condition swapping in many zero-shot applications. However, their performance is heavily affected by the modality gap between audio and text embeddings. Existing explanations mainly attribute this gap to the cone effect, treating it as a shift between mean embeddings, yet correcting the mean alone yields only limited improvements. Alternative hypotheses, such as information imbalance and dimensionality collapse, have also been proposed, but they remain insufficiently verified and have not been thoroughly studied in the audio domain. Meanwhile, several works attempt to decompose multimodal contrastive embeddings into interpretable concepts, but none explicitly analyze the modality gap from the perspective of concept decomposition. In this work, we introduce COMET (Concept space Organization and Modality gap Explanation with PLS-SVD Transformation), a novel partial least squares singular value decomposition (PLS-SVD) framework for CLAP that unveils a broader perspective of the modality gap. Our framework reveals that only a small, interpretable subset of axes, which captures shared concepts, contributes substantially to similarity computation, and that the mean component represents only partially the modality gap. Building on this insight, we propose a simple spectral truncation method that mitigates the modality gap in a training-free manner. The method enables zero-shot audio captioning with condition swapping to approach fully supervised performance, without requiring large auxiliary memory banks or expensive computation. At the same time, it achieves substantial embedding dimensionality reduction while preserving strong performance on retrieval and audio captioning tasks.
>
---
#### [new 009] Frequency-Modulated and Single-Tone Excitation to Reveal Vibro-Acoustic Nonlinearities in Loosened Bolted Joints
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于结构健康监测任务，旨在检测螺栓松动。通过振动声学方法，结合单频和调频激励，分析非线性特征以区分不同预紧状态。**

- **链接: [https://arxiv.org/pdf/2605.29950](https://arxiv.org/pdf/2605.29950)**

> **作者:** Berkay Kullukcu; Robin Pianowski; Dina Hannebauer
>
> **摘要:** Preload loss in bolted joints results in alterations of the stiffness, damping, and nonlinearity of the structure, but existing monitoring techniques for rail-vehicle systems are often not capable of combining controlled shaker tests and sensing of nonlinear features. This paper proposes a method for detecting bolt loosening using a vibro-acoustic technique, where the structure is subjected to controlled shaker tests to sense the nonlinear features. A triaxial accelerometer was attached to the demonstrator, a microphone was placed in close proximity, and one of the bolts was tested under 0%, 20%, 40%, and 80% preload conditions. Single-tone and frequency-modulated (FM) signals close to the main natural frequency of 130 Hz, which was identified using sine sweep and narrow-band excitation, were applied to the demonstrator. When the structure was subjected to 130 Hz single-tone excitation, the loose state of the bolt exhibited several additional high-frequency spectral peaks. FM excitation between 125 and 135 Hz further distinguished between the states. Harmonic band power ratios, normalized to the carrier, distinguished between the loose state and the 80% preload state, where the difference between the loose and 80% preload states was 17.5 dB for l = 2 and 36.5 dB for l = 6.
>
---
#### [new 010] Decoding Strategies for Diffusion-Based ASR: A Systematic Evaluation of Confidence-Based Thresholding
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升扩散模型的解码效率。通过分析三种解码策略，提出基于置信度的阈值方法，显著提高准确率和速度。**

- **链接: [https://arxiv.org/pdf/2605.29613](https://arxiv.org/pdf/2605.29613)**

> **作者:** Jeong Hun Yeo; Minsu Kim; Hyeongseop Rha; Yong Man Ro
>
> **摘要:** While LLM-based Automatic Speech Recognition (ASR) achieves high accuracy, its speed is limited by sequential autoregressive decoding. Diffusion Language Models (DLMs) offer a parallel alternative, yet their decoding strategies remain under-explored in ASR contexts. This paper analyzes three decoding schemes for DLM-based ASR: fixed-number, static confidence threshold, and dynamic confidence threshold. We propose measuring round-wise accuracy using Negative Log-Likelihood-based uncertainty as a proxy for decoding progress. Our results show that both threshold-based strategies significantly outperform fixed-number schemes in accuracy and speed. We attribute this to a property unique to ASR: most tokens reach high confidence early, allowing reliable ones to be harvested aggressively while leaving only difficult tokens for later rounds. Notably, the static-threshold strategy matches the accuracy of autoregressive decoding while offering superior efficiency.
>
---
#### [new 011] GrowLoop: Self-Evolving Conversation Evaluation Seeded by Human
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出GrowLoop，解决开放问答中人类相似性评估难题。通过自进化机制，持续优化评估标准，提升模型评估准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2605.28882](https://arxiv.org/pdf/2605.28882)**

> **作者:** Yihang Lin; Yunze Gao; Zeyang Lin; Dongbo Li; Kun Peng; Chenglong Song; Yue Liu
>
> **摘要:** With the rapid advancement of large language models, evaluating human-likeness in open-ended conversation has become increasingly important. However, human-likeness is a form of tacit knowledge that humans perceive intuitively, yet the underlying criteria resist explicit formulation. Human judgments vary widely, with strong agreement on some cases and legitimate disagreement on others. Meanwhile, the criteria behind human judgments remain implicit, leaving no clear basis for constructing cases. Further, what counts as human-like is not static, but evolving with model capability and human expectations. Despite progress in evaluation methods such as expert-authored benchmarks, Reward Models, and self-evolving benchmarks, none addresses all three challenges simultaneously. Therefore, we propose GrowLoop, a self-evolving conversation evaluation system that continuously adapts as models advance and scenarios shift. With minimal human seed annotations as the first mover, LLM agents iteratively extract and refine evaluation rubrics through Heuristic Learning. Human-AI agreement is required where annotators converge, while only plausibility is expected where they diverge. Moreover, the Rubric-Case co-evolution mechanism enables continuous evolution, expanded through new seeds when the evaluation target moves. Applied to human-likeness evaluation in open-ended conversation, the generated rubrics not only substantially outperform existing methods in alignment with human judgments, but also uncover issues that annotators overlook. The resulting benchmark effectively discriminates models across capability tiers and reveals where they fall short, while generalizing to new scenarios and adapting as models advance. Our work shifts the benchmarking paradigm from manual updates or difficulty scaling to comprehensive, continuous self-evolution.
>
---
#### [new 012] It`s All About Speed: AI`s Impact on Workflow in Music Production
- **分类: cs.AI; eess.AS**

- **简介: 论文探讨AI工具对音乐制作流程的影响，属于人机交互研究。解决AI与用户在效率、控制和创意自主间的冲突问题，通过分析专业人员使用情况提出设计优化方向。**

- **链接: [https://arxiv.org/pdf/2605.29931](https://arxiv.org/pdf/2605.29931)**

> **作者:** Finn McClellan; Fabio Morreale
>
> **备注:** Audio Engineering Society Conference Paper - Presented at the AES International Conference on Machine Learning and Artificial Intelligence for Audio 2025 - September 8-10, London, UK
>
> **摘要:** In this paper, we present the results of an ethnographic study into the impact of AI and automated tools on music production workflow. Focusing specifically on professional participants who identified as recording engineers, mixers, and producers, we discuss their usage of common AI and automated software, as well as their sentiments on the proliferation of these tools. We discuss tensions that may be created between users and automated tools in key areas such as the need for speed and efficiency, controllability, and maintaining creative agency, and how these tensions may be alleviated through tool design.
>
---
#### [new 013] Benchmarking Single-Factor Physical Video-to-Audio Generation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决模型是否捕捉物理过程的问题。通过构建基准测试，评估模型在物理因素变化下的表现，揭示其依赖文本而非视觉信息的倾向。**

- **链接: [https://arxiv.org/pdf/2605.30339](https://arxiv.org/pdf/2605.30339)**

> **作者:** Tingle Li; Siddharth Gururani; Kevin J. Shih; Gantavya Bhatt; Sang-gil Lee; Zhifeng Kong; Arushi Goel; Gopala Anumanchipalli; Ming-Yu Liu
>
> **备注:** CVPR 2026
>
> **摘要:** Generative video-to-audio (V2A) models produce highly plausible soundtracks, but it remains unclear whether they capture the underlying physical processes. Existing evaluations emphasize perceptual realism and overlook physical correctness under controlled interventions. In this paper, we introduce FlatSounds, a benchmark that audits the physical reasoning of V2A models through: 1) controlled counterfactual pairs in which a single physical factor is varied, and 2) single-video pattern tests that probe internal consistency and directional trends. These settings test whether the generated audio correctly reflects specific physical properties and timings. Our evaluation of state-of-the-art models reveals a consistent trade-off: models rely more on text captions than the visual stream to infer physics and semantics. Captions generally improve physical and semantic accuracy, but paradoxically degrade temporal alignment. Our results highlight the need to move beyond audio quality toward learning physical processes directly from pixels. Finally, we find that our physics-based metrics correlate strongly with human preference tests on our own data. Project webpage: this https URL
>
---
#### [new 014] MusTBENCH: Benchmarking and Advancing Temporal Grounding in Music LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs在时间定位上的不足。提出MusTBENCH基准和MusT优化方法，提升模型的时间对齐能力。**

- **链接: [https://arxiv.org/pdf/2605.29300](https://arxiv.org/pdf/2605.29300)**

> **作者:** Daeyong Kwon; Qiyu Wu; Shinobu Kuriya; Junghyun Koo; Shuyang Cui; Zhi Zhong; Wei-Hsiang Liao; Hiromi Wakaki; Yuki Mitsufuji
>
> **摘要:** Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBENCH, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBENCH show that existing LALMs struggle with precise temporal grounding, while MusT brings significant improvements over strong baselines. These results establish temporal grounding as a key missing capability in current LALMs and position MusTBENCH as a challenging benchmark for future research in temporally grounded music understanding.
>
---
## 更新

#### [replaced 001] Weakly Supervised Detection and Temporal Localization of Whale Calls in Long-Duration Bioacoustic Data
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于生物声学分析任务，解决长时录音中鲸鸣检测与时间定位问题。提出DSMIL-LocNet框架，仅用记录级标签实现分类与定位，提升自动化水平。**

- **链接: [https://arxiv.org/pdf/2502.20838](https://arxiv.org/pdf/2502.20838)**

> **作者:** Ragib Amin Nihal; Benjamin Yen; Runwu Shi; Takeshi Ashizawa; Kazuhiro Nakadai
>
> **备注:** Accepted in European Signal Processing Conference (EUSIPCO) 2026
>
> **摘要:** Passive acoustic monitoring (PAM) systems generate continuous recordings spanning months, yet automated bioacoustic analysis of whale calls requires two separate annotation efforts: binary presence labels for classification and precise temporal boundaries for localization. A binary label for a multi-minute recording can be assigned in seconds, but timestamping every call within it requires hours of expert effort. Providing both is infeasible at operational scale. We present DSMIL-LocNet, a weakly supervised multiple instance learning (MIL) framework that performs both classification and temporal localization using only recording-level presence/absence labels. Our dual-stream architecture integrates spectral and temporal features to process recordings of 2--30 minutes without the temporal compression that degrades existing CNN methods on long inputs. On the AcousticTrends BlueFinLibrary, DSMIL-LocNet achieves F1 scores of 0.88--0.91 on recordings of 300--1800s, where fully supervised CNN baselines degrade to 0.19--0.64. It also provides temporal localization that these baselines cannot produce without frame-level annotation. Code: this https URL
>
---
#### [replaced 002] FNH-TTS: Mixture-of-Experts Duration Modeling for Robust Neural Speech Synthesis
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决非自回归TTS系统中持续时间建模不足的问题。提出MoE-DP和增强的声码器，提升合成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2508.12001](https://arxiv.org/pdf/2508.12001)**

> **作者:** Qingliang Meng; Yuqing Deng; Wei Liang; Limei Yu; Huizhi Liang; Tian Li
>
> **摘要:** Current non-autoregressive (NAR) text-to-speech (TTS) systems still struggle to model diverse and speaker-dependent duration variation. We further observe that richer duration variation can increase the synthesis difficulty of existing HiFi-GAN-based vocoders, leading to spectral artifacts and unstable time-frequency structures. To address these issues, we propose FNH-TTS, a VITS-based end-to-end TTS system with Mixture-of-Experts duration modeling and robust vocoder-side synthesis. Specifically, we introduce a Mixture-of-Experts Duration Predictor (MoE-DP) to capture diverse phoneme duration patterns and speaker-dependent speaking-rate characteristics. To convert richer duration variation into stable waveform generation, we further integrate a VOCOS-style vocoder with Collaborative Multi-Band and Sub-Band Discriminators. Experiments on LJSpeech, VCTK, and LibriTTS show that FNH-TTS achieves improved synthesis quality, duration-category accuracy, vocoder reconstruction quality, and inference efficiency. Further analysis shows that MoE-DP is the main source of improved duration modeling, while stronger vocoder-side components are necessary for robust synthesis under richer duration variation.
>
---
#### [replaced 003] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出JAEGER框架，解决AV-LLMs在3D环境中的空间感知与推理问题。通过融合RGB-D和多通道音频，提升方向估计与空间理解能力。**

- **链接: [https://arxiv.org/pdf/2602.18527](https://arxiv.org/pdf/2602.18527)**

> **作者:** Zhan Liu; Changli Tang; Yuxin Wang; Zhiyuan Zhu; Youjun Chen; Yiwen Shao; Tianzi Wang; Lei Ke; Zengrui Jin; Chao Zhang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Current audio-visual large language models (AV-LLMs) are predominantly restricted to 2D perception, relying on RGB video and monaural audio. This design choice introduces a fundamental dimensionality mismatch that precludes reliable source localization and spatial reasoning in complex 3D environments. We address this limitation by presenting JAEGER, a framework that extends AV-LLMs to 3D space, to enable joint spatial grounding and reasoning through the integration of RGB-D observations and multi-channel first-order ambisonics. A core contribution of our work is the neural intensity vector (Neural IV), a learned spatial audio representation that encodes robust directional cues to enhance direction-of-arrival estimation, even in adverse acoustic scenarios with overlapping sources. To facilitate large-scale training and systematic evaluation, we propose SpatialSceneQA, a benchmark of 61k instruction-tuning samples curated from simulated physical environments. Extensive experiments demonstrate that our approach consistently surpasses 2D-centric baselines across diverse spatial perception and reasoning tasks, underscoring the necessity of explicit 3D modelling for advancing AI in physical environments. Our source code, pre-trained model checkpoints, and datasets are available at this https URL.
>
---
#### [replaced 004] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching
- **分类: cs.SD; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于音频生成任务，解决流模型中表示对齐的层选择问题。提出AG-REPA方法，通过因果贡献选择关键层，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.01006](https://arxiv.org/pdf/2603.01006)**

> **作者:** Pengfei Zhang; Tianxin Xie; Minghao Yang; Li Liu
>
> **备注:** Accepted to ICML 2026. 17 pages, 4 figures, 12 tables
>
> **摘要:** REPresentation Alignment (REPA) improves the training of generative flow models by aligning intermediate hidden states with pretrained teacher features, but its effectiveness in token-conditioned audio Flow Matching critically depends on the choice of supervised layers, which is typically made heuristically based on the depth. In this work, we introduce Attribution-Guided REPresentation Alignment (AG-REPA), a novel causal layer selection strategy for representation alignment in audio Flow Matching. Firstly, we find that layers that best store semantic/acoustic information (high teacher-space similarity) are not necessarily the layers that contribute most to the velocity field that drives generation, and we call it Store-Contribute Dissociation (SCD). To turn this insight into an actionable training guidance, we propose a forward-only gate ablation (FoG-A) that quantifies each layer's causal contribution via the induced change in the predicted velocity field, enabling sparse layer selection and adaptive weighting for alignment. Across unified speech and general-audio training (LibriSpeech + AudioSet) under different token-conditioning topologies, AG-REPA consistently outperforms REPA baselines. Overall, our results show that alignment is most effective when applied to the causally dominant layers that drive the velocity field, rather than to layers that are representationally rich but functionally passive.
>
---
#### [replaced 005] Explainable AI in Speaker Recognition -- Making Latent Representations Understandable
- **分类: eess.AS; cs.AI; eess.SP**

- **简介: 该论文属于语音识别任务，旨在解释神经网络的潜在表示。通过分析层次聚类现象，提出HCCM算法和Liebig分数，以理解聚类与语义类别的关系。**

- **链接: [https://arxiv.org/pdf/2604.23354](https://arxiv.org/pdf/2604.23354)**

> **作者:** Yanze Xu; Wenwu Wang; Mark D. Plumbley
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Neural networks can be trained to learn task-relevant representations from data. Understanding how these networks make decisions falls within the Explainable AI (XAI) domain. This paper proposes to study an XAI topic: uncovering the unknown organisation in the representations, particularly those a speaker recognition network learns from utterances, for recognising speaker identity. Past studies have employed algorithms (e.g. K-means) to analyse how network representations can be naturally organised into independent clusters in different ways, i.e., to analyse flat clustering phenomena within the space defined by these representations, referred to as the network representation space. In contrast, this work applies two algorithms, Single-Linkage Clustering (SLINK) and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN), to analyse how representations form hierarchical clusters in different ways, i.e., to analyse hierarchical clustering phenomena within the network representation space. To further understand these hierarchical clustering phenomena, we propose a new algorithm termed Hierarchical Cluster-Class Matching (HCCM). HCCM provides a semantic interpretation for the hierarchical clusters produced by SLINK and HDBSCAN by matching them to predefined semantic classes. Through this process, some clusters are interpreted as individual semantic classes (e.g. male), whereas others are interpreted as conjunctions of individual semantic classes (e.g. female and Ireland). In addition, we develop a new metric, the Liebig score, to quantify how well a cluster matches a semantic class, which helps identify the factor that most strongly limits each match.
>
---
#### [replaced 006] OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出OmniCustom，解决同步音视频定制任务，通过联合生成模型同时控制视频身份和音频音色，实现高质量音视频生成。**

- **链接: [https://arxiv.org/pdf/2602.12304](https://arxiv.org/pdf/2602.12304)**

> **作者:** Maomao Li; Zhen Li; Kaipeng Zhang; Guosheng Yin; Zhifeng Li; Dong Xu
>
> **备注:** code: this https URL
>
> **摘要:** Existing mainstream video customization methods focus on generating identity-consistent videos based on given reference images and textual prompts. Benefiting from the rapid advancement of joint audio-video generation, this paper proposes a more compelling new task: sync audio-video customization, which aims to synchronously customize both video identity and audio timbre. Specifically, given a reference image $I^{r}$ and a reference audio $A^{r}$, this novel task requires generating videos that maintain the identity of the reference image while imitating the timbre of the reference audio, with spoken content freely specifiable through user-provided textual prompts. To this end, we propose OmniCustom, a powerful DiT-based audio-video customization framework that can synthesize a video following reference image identity, audio timbre, and text prompts all at once in a zero-shot manner. Our framework is built on three key contributions. First, identity and audio timbre control are achieved through separate reference identity and audio LoRA modules that operate through self-attention layers within the base audio-video generation model. Second, we introduce a contrastive learning objective alongside the standard flow matching objective. It uses predicted flows conditioned on reference inputs as positive examples and those without reference conditions as negative examples, thereby enhancing the model ability to preserve identity and timbre. Third, we train OmniCustom on our constructed large-scale, high-quality audio-visual human dataset. Extensive experiments demonstrate that OmniCustom outperforms existing methods in generating audio-video content with consistent identity and timbre fidelity. Project page: this https URL.
>
---
#### [replaced 007] BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决音乐符号化表示中时间不统一的问题。通过将音乐划分为均匀时间步（如节拍）进行编码，提升模型对音乐结构和长期模式的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2604.19532](https://arxiv.org/pdf/2604.19532)**

> **作者:** Lekai Qian; Haoyu Gu; Jingwei Zhao; Ziyu Wang
>
> **摘要:** Tokenizing music to fit the general framework of language models is a compelling challenge, especially considering the diverse symbolic structures in which music can be represented (e.g., sequences, grids, and graphs). To date, most approaches tokenize symbolic music as sequences of musical events, such as onsets, pitches, time shifts, or compound note events. This strategy is intuitive and has proven effective in Transformer-based models, but it treats the regularity of musical time implicitly: individual tokens may span different durations, resulting in non-uniform time progression. In this paper, we instead consider whether an alternative tokenization is possible, where a uniform-length musical step (e.g., a beat) serves as the basic unit. Specifically, we encode all events within a single time step at the same pitch as one token, and group tokens explicitly by time step, which resembles a sparse encoding of a piano-roll representation. We evaluate the proposed tokenization on music continuation and accompaniment generation tasks, comparing it with mainstream event-based methods. Results show improved musical quality and structural coherence, while additional analyses confirm higher efficiency and more effective capture of long-range patterns with the proposed tokenization.
>
---
#### [replaced 008] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多说话人语音识别任务，旨在解决单通道音频中多人重叠语音的识别与归属问题。工作包括梳理E2E架构、分析不同模型结构并评估其性能。**

- **链接: [https://arxiv.org/pdf/2505.10975](https://arxiv.org/pdf/2505.10975)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** Accepted for publication in Computer Speech & Language (CSL)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [replaced 009] EvA: An Evidence-First Audio Understanding Paradigm for LALMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出EvA模型，解决LALMs在复杂声学场景中的证据保留问题。通过双路径架构提升音频理解，优化声学证据提取与融合。**

- **链接: [https://arxiv.org/pdf/2603.27667](https://arxiv.org/pdf/2603.27667)**

> **作者:** Xinyuan Xie; Shunian Chen; Zhiheng Liu; Yuhao Zhang; Zhiqiang Lv; Liyin Liang; Benyou Wang
>
> **摘要:** Large Audio Language Models (LALMs) still struggle in complex acoustic scenes because they often fail to preserve task-relevant acoustic evidence before reasoning begins. We identify this error pattern as the evidence bottleneck: state-of-the-art systems show larger deficits in acoustic evidence extraction than in downstream reasoning, suggesting that upstream perception is often the limiting factor. To address this problem, we propose EvA (Evidence-First Audio), a dual-path architecture that enhances acoustic evidence preservation through hierarchical aggregation and non-compressive, time-aligned fusion. We also build EvA-Perception, a large-scale training set with about 54K event-ordered captions and 500K evidence-grounded QA pairs. Under a unified zero-shot protocol, EvA achieves the best open-source \emph{Perception} results on MMAU, MMAR, and MMSU, with the largest gains on perception-heavy splits. Human evaluation on open-ended captioning further shows improved fine-grained acoustic coverage and caption quality. These results support the evidence-first hypothesis: stronger audio understanding depends on preserving acoustic evidence before reasoning. Project can be found at this https URL.
>
---
#### [replaced 010] An Extensive Analysis of the Singing Voice Conversion Challenge 2025 Evaluation Results
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音转换任务，旨在分析2025年歌唱语音转换挑战的评估结果，解决歌唱风格与身份转换的问题，通过新数据库和多种评估方法对比33个系统的表现。**

- **链接: [https://arxiv.org/pdf/2509.15629](https://arxiv.org/pdf/2509.15629)**

> **作者:** Lester Phillip Violeta; Xueyao Zhang; Jiatong Shi; Yusuke Yasuda; Wen-Chin Huang; Zhizheng Wu; Tomoki Toda
>
> **备注:** Submitted to IEEE TASLP
>
> **摘要:** We present a thorough analysis of the findings of the latest iteration of the Singing Voice Conversion Challenge, a scientific event aiming to compare and understand different voice conversion systems in a controlled environment. Compared to previous iterations which solely focused on converting the singer identity, this year we also focused on converting the singing style of the singer. To create a controlled environment and thorough evaluations, we developed a new challenge database, introduced two tasks, open-sourced baselines, and conducted large-scale crowd-sourced listening tests and objective evaluations. The challenge was run for two months and in total we evaluated 33 different systems. The results of the large-scale crowd-sourced listening test showed that top systems had comparable singer identity scores to ground truth samples. However, modeling the singing style and consequently achieving high naturalness still remains a challenge in this task, primarily due to the difficulty in modeling dynamic information in breathy, glissando, and vibrato singing styles. Further analyses of the challenge also discuss the limitations of both the traditional similarity test and the dynamic preference test in evaluating singing style similarity. Moreover, calculating Spearman's rank correlation coefficient shows that dependent objective metrics such as chroma-alignment and non-match metrics such as speaker embeddings are the most correlated to subjective scores, but are still not at a level where it could be considered as a true replacement for subjective scores.
>
---
#### [replaced 011] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出MedMosaic，一个大规模医学音频问答基准，用于评估语言与音频推理模型。针对医学音频数据收集难、标注成本高的问题，构建多样化数据集，包含46,701个问答对，涵盖多种题型，以测试多跳推理和答案生成能力。**

- **链接: [https://arxiv.org/pdf/2605.00969](https://arxiv.org/pdf/2605.00969)**

> **作者:** Harshit Rajgarhia; Shuubham Ojha; Asif Shaik; Akhil Pothanapalli; Rachuri Lokesh; Abhishek Mukherji; Prasanna Desikan
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Medical audio data is difficult to collect due to privacy regulations and high annotation costs arising from domain expertise. Thus, existing benchmarks tend to underrepresent complex medical audio scenarios. To address this challenge, we present MedMosaic, a medical audio question-answering dataset designed to benchmark language and audio reasoning models under realistic clinical constraints. MedMosaic features a diverse range of medical audio types, including condition-related physiological sounds, carefully constructed synthetic voices to mimic speech with artifacts as well as real short and long length clinical conversations to model varying context lengths. The dataset also features a total of 46,701 question-answer pairs, spanning categories such as multiple-choice, sequential multi-turn, and open-ended question-answers, enabling systematic evaluation of multi-hop reasoning and answer generation capabilities. Benchmarking 13 audio and multimodal reasoning models reveals that reasoning remains challenging for all evaluated systems, with substantial performance variation across question types. In particular, even state-of-the-art model like Gemini-2.5-pro can only achieve 68.1% accuracy approximately. These findings underscore persistent limitations in medical reasoning and highlight the need for more robust, domain-specific multimodal reasoning models. A sample of benchmark data is available here: this https URL
>
---
#### [replaced 012] Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability
- **分类: cs.SD**

- **简介: 该论文属于角色扮演语音合成任务，解决风格一致性不足的问题。提出MCLP作为评估和奖励机制，提升生成语音与角色设定的匹配度。**

- **链接: [https://arxiv.org/pdf/2601.22661](https://arxiv.org/pdf/2601.22661)**

> **作者:** Yong Ren; Jingbei Li; Haiyang Sun; Yujie Chen; Cheng Yi; Yechang Huang; Hao Gu; Ye Bai; Xuerui Yang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Recent advances in Large Audio Language Models (LALMs) have extended Text-to-Speech (TTS) to interactive role-play scenarios, which demand high expressiveness and strict adherence to role-play instructions. However, existing models struggle to maintain stylistic consistency with character profiles and scene descriptions across multi-turn dialogues. A critical bottleneck is the lack of objective metrics for quantifying speaking style. To bridge this gap, we propose Mean Continuation Log-Probability (MCLP) as both an evaluation metric and a reward signal, validated on LALM-based Role-Play TTS (RP-TTS) tasks. MCLP leverages the in-context learning capability of pretrained LALMs to measure the likelihood of ground-truth speech tokens conditioned on a contextual history consisting of the transcript, generated speech, and repeated transcript, serving as a proxy for stylistic continuity. Furthermore, we employ MCLP as a reinforcement learning reward to enhance the style alignment between generated speech and role-play instructions. To support this task, we construct a large-scale RP-TTS dataset with rich scene and character annotations. Experiments demonstrate that MCLP is well aligned with human judgments of stylistic consistency and serves as an effective reward for improving RP-TTS, leading to consistent gains in both objective metrics and subjective evaluations. Our code is publicly available at this https URL.
>
---
#### [replaced 013] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦音频分段任务，解决音频章节划分中的文本依赖、ASR误差及评估方法问题。提出AudioSeg模型，对比不同方法，分析影响因素并建立新评估协议。**

- **链接: [https://arxiv.org/pdf/2602.08979](https://arxiv.org/pdf/2602.08979)**

> **作者:** Fabian Retkowski; Maike Züfle; Thai Binh Nguyen; Jan Niehues; Alexander Waibel
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Audio chaptering, the task of segmenting long-form audio into coherent sections, is increasingly important for navigating podcasts, lectures, and videos. Despite its relevance, research remains limited and text-based, leaving key questions unresolved about leveraging audio information, handling ASR errors, and transcript-free evaluation. We address these gaps through three contributions: (1) a systematic comparison between text-based models with acoustic features, a novel audio-only architecture (AudioSeg) operating on learned audio representations, and multimodal LLMs; (2) empirical analysis of factors affecting performance, including transcript quality, acoustic features, duration, and speaker composition; and (3) formalized evaluation protocols contrasting transcript-dependent text-space protocols with transcript-invariant time-space protocols. Our experiments on YTSeg reveal that AudioSeg substantially outperforms text-based approaches, pauses provide the largest acoustic gains, and MLLMs remain limited by context length and weak instruction following, yet MLLMs are promising on shorter audio.
>
---
#### [replaced 014] EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出EVA-Bench，用于评估语音代理的性能。解决语音代理在真实对话生成和质量测量方面的评估难题，通过模拟对话和引入两个综合指标进行跨架构比较。**

- **链接: [https://arxiv.org/pdf/2605.13841](https://arxiv.org/pdf/2605.13841)**

> **作者:** Tara Bogavelli; Gabrielle Gauthier Melançon; Katrina Stankiewicz; Oluwanifemi Bamgbose; Fanny Riols; Hoang H. Nguyen; Raghav Mehndiratta; Lindsay Devon Brin; Joseph Marinier; Hari Subramani; Anil Madamala; Sridhar Krishna Nemala; Srinivas Sunkara
>
> **备注:** Work in progress
>
> **摘要:** Voice agents, artificial intelligence systems that conduct spoken conversations to complete tasks, are increasingly deployed across enterprise applications. However, no existing benchmark jointly addresses two core evaluation challenges: generating realistic simulated conversations, and measuring quality across the full scope of voice-specific failure modes. We present EVA-Bench, an end-to-end evaluation framework that addresses both. On the simulation side, EVA-Bench orchestrates bot-to-bot audio conversations over dynamic multi-turn dialogues, with automatic simulation validation that detects user simulator error and appropriately regenerates conversations before scoring. On the measurement side, EVA-Bench introduces two composite metrics: EVA-A (Accuracy), capturing task completion, faithfulness, and audio-level speech fidelity; and EVA-X (Experience), capturing conversation progression, spoken conciseness, and turn-taking timing. Both metrics apply to all major agent architectures, enabling direct cross-architecture comparison. EVA-Bench includes 213 scenarios across three enterprise domains, a controlled perturbation suite for accent and noise robustness, and pass@1, pass@k, pass^k measurements that distinguish peak from reliable capability. Across 12 systems spanning all three architectures, we find: (1) no system simultaneously exceeds 0.5 on both EVA-A pass@1 and EVA-X pass@1; (2) peak and reliable performance diverge substantially (median pass@k--pass^k gap of 0.44 on EVA-A); and (3) accent and noise perturbations expose substantial robustness gaps, with effects varying across architectures, systems, and metrics (mean $\Delta$ up to 0.314). We release the full framework, evaluation suite, and benchmark data under an open-source license.
>
---
#### [replaced 015] AV-EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Omni-modal LLMS with Audio-visual Cues
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于情感推理任务，旨在评估多模态大模型的情感理解与回应能力。通过构建基准测试框架，解决情感推理评估不足的问题。**

- **链接: [https://arxiv.org/pdf/2510.07355](https://arxiv.org/pdf/2510.07355)**

> **作者:** Dingkun Zhou; Krish Patel; Ajay Kankipati; Akshaj Gupta; Zeyi Austin Li; Mohul Shukla; Vibhor Narang; Sara Kofman; Zongli Ye; Grace Wang; Xiaoyu Shi; Tingle Li; Guan-Ting Lin; Kan Jen Cheng; Huang-Cheng Chou; Jiachen Lian; Gopala Anumanchipalli
>
> **摘要:** Emotions conveyed through voice and face shape engagement and context in human AI interaction. Despite rapid progress in omni modal large language models, the holistic evaluation of emotional reasoning with audiovisual cues remains limited. To address this gap, we introduce AV EMO Reasoning, a benchmark designed to systematically assess emotional reasoning abilities in large language models. The framework uses a curated audiovisual corpus comprising synthetic single turn and multi turn dialogues and a real world subset, together with emotion perception and interaction reasoning metrics, to evaluate whether models can understand user emotions and produce appropriate responses. By releasing a systematic evaluation benchmark, AV EMO Reasoning offers a reproducible standard for evaluating emotion aware dialogue and advances toward more natural, adaptive human AI interaction.
>
---
#### [replaced 016] SegTune: Structured and Fine-Grained Control for Song Generation
- **分类: cs.SD**

- **简介: 该论文提出SegTune，解决歌曲生成中缺乏细粒度结构控制的问题。通过段落级提示和时序广播实现精准控制，提升音乐结构与歌词对齐的准确性。**

- **链接: [https://arxiv.org/pdf/2510.18416](https://arxiv.org/pdf/2510.18416)**

> **作者:** Pengfei Cai; Joanna Wang; Haorui Zheng; Xu Li; Zihao Ji; Teng Ma; Zhongliang Liu; Chen Zhang; Pengfei Wan
>
> **备注:** This technical report was later revised and published at ACL 2026 (oral). ACL paper link: this https URL , code: this https URL
>
> **摘要:** Recent advancements in song generation have shown promising results in generating songs from lyrics and/or global text prompts. However, most existing systems lack the ability to model the temporally varying attributes of songs, limiting fine-grained control over musical structure and dynamics. In this paper, we propose SegTune, a non-autoregressive framework for structured and controllable song generation. SegTune enables segment-level control by allowing users or large language models to specify local musical descriptions aligned to song this http URL segmental prompts are injected into the model by temporally broadcasting them to corresponding time windows, while global prompts influence the whole song to ensure stylistic coherence. To obtain accurate segment durations and enable precise lyric-to-music alignment, we introduce an LLM-based duration predictor that autoregressively generates sentence-level timestamped lyrics in LRC format. We further construct a large-scale data pipeline for collecting high-quality songs with aligned lyrics and prompts, and propose new evaluation metrics to assess segment-level alignment and vocal attribute consistency. Experimental results show that SegTune achieves superior controllability and musical coherence compared to existing baselines. See this https URL for demos of our work.
>
---
