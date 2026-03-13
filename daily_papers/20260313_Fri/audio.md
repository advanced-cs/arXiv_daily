# 音频 cs.SD;  eess.AS

- **最新发布 25 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Resurfacing Paralinguistic Awareness in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决LALMs忽视语气等副语言信息的问题。通过分析层结构并提出增强微调协议，提升模型对副语言线索的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.11947](https://arxiv.org/pdf/2603.11947)**

> **作者:** Hao Yang; Minghan Wang; Tongtong Wu; Lizhen Qu; Ehsan Shareghi; Gholamreza Haffari
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Large Audio Language Models (LALMs) have expanded the interaction with human to speech modality, which introduces great interactive potential, due to the paralinguistic cues implicitly indicating the user context. However, building on the current content-centred paradigm, LALMs usually neglect such paralinguistic cues and respond solely based on query content. In this work, to resurface the paralinguistic awareness in LALMs, we introduce five diverse layer-wise analyses to jointly identify paralinguistic layers and semantic understanding layers. Based on these insights, we propose a paralinguistic-enhanced fine-tuning (PE-FT) protocol accordingly to equip LALMs with paralinguistic-aware capabilities, including (1) selective-layer fine-tuning, and (2) an auxiliary dual-level classification head. Our experiments demonstrate that PE-FT protocol efficiently and effectively resurfaces the paralinguistic awareness, even surpassing the performance of the all-layer fine-tuning strategy.
>
---
#### [new 002] Acoustic-to-Articulatory Inversion of Clean Speech Using an MRI-Trained Model
- **分类: eess.AS**

- **简介: 该论文属于语音到发音器官反演任务，解决MRI噪声影响的问题。通过使用干净语音替代降噪MRI语音进行建模，验证了干净语音在反演中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.11845](https://arxiv.org/pdf/2603.11845)**

> **作者:** Sofiane Azzouz; Pierre-André Vuissoz; Yves Laprie
>
> **摘要:** Articulatory acoustic inversion reconstructs vocal tract shapes from speech. Real-time magnetic resonance imaging (rt-MRI) allows simultaneous acquisition of both the acoustic speech signal and articulatory information. Besides the complexity of rt-MRI acquisition, the recorded audio is heavily corrupted by scanner noise and requires denoising to be usable. For practical use, it must be possible to invert speech recorded without MRI noise. In this study, we investigate the use of speech recorded in a clean acoustic environment as an alternative to denoised MRI speech. To this end we compare two signals from the same speaker with identical sentences which are aligned using phonetic segmentation. A model trained on denoised MRI speech is evaluated on both denoised MRI and clean speech. We also assess a model trained and tested only on clean speech. Results show that clean speech supports articulatory inversion effectively, achieving an RMSE of 1.56 mm, close to MRI-based performance.
>
---
#### [new 003] Edge-Cloud Collaborative Speech Emotion Captioning via Token-Level Speculative Decoding in Audio-Language Models
- **分类: cs.SD**

- **简介: 该论文属于语音情感描述任务，解决边缘设备计算资源不足和隐私问题。提出一种边云协作框架，通过不确定性引导的推测解码，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.11397](https://arxiv.org/pdf/2603.11397)**

> **作者:** Xiangyuan Xue; Jiajun Lu; Yan Gao; Gongping Huang; Ting Dang; Hong Jia
>
> **摘要:** Speech Emotion Captioning (SEC) leverages large audio-language models to generate rich, context-aware affective descriptions from speech. However, real-world deployment remains challenging due to the substantial computational demands on resource-constrained edge devices and the privacy risks of transmitting biometric audio. While smaller audio-language models enable efficient on-device SEC, their limited capacity often weakens subtle paralinguistic modeling and fine-grained affective grounding. We propose an edge-cloud collaborative framework based on Uncertainty-Guided Speculative Decoding (UGSD). A lightweight edge model drafts captions locally, and only high-uncertainty token blocks are selectively escalated to a stronger cloud verifier for validation. Experiments on the MER2024 benchmark demonstrate substantial BLEU improvements up to 62.7%. UGSD further achieves 1.4x lower latency and 8.5x higher token throughput compared to an edge-only model. These results empirically characterize the quality-efficiency-privacy trade-off in deployable SEC systems.
>
---
#### [new 004] Continued Pretraining for Low-Resource Swahili ASR: Achieving State-of-the-Art Performance with Minimal Labeled Data
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于低资源语言ASR任务，旨在提升斯瓦希里语语音识别性能。通过结合少量标注数据与未标注音频进行持续预训练，显著降低了词错误率。**

- **链接: [https://arxiv.org/pdf/2603.11378](https://arxiv.org/pdf/2603.11378)**

> **作者:** Hillary Mutisya; John Mugane
>
> **摘要:** We investigate continued pretraining (CPT) for adapting wav2vec2-bert-2.0 to Swahili automatic speech recognition (ASR). Our approach combines unlabeled audio with limited labeled data through pseudo-labeled CPT followed by supervised finetuning. With 20,000 labeled samples, we achieve 3.24% WER on Common Voice Swahili-an 82% relative improvement over the baseline. This result surpasses the best previously reported academic system (8.3% WER from XLS-R) by 61% relative improvement. We provide concrete data requirements and a replicable methodology applicable to other low-resource languages.
>
---
#### [new 005] Cough activity detection for automatic tuberculosis screening
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于咳嗽活动检测任务，旨在自动识别音频中的咳嗽段，用于结核病筛查。通过应用预训练模型XLS-R，提升检测精度并优化计算效率。**

- **链接: [https://arxiv.org/pdf/2603.11241](https://arxiv.org/pdf/2603.11241)**

> **作者:** Joshua Jansen van Vüren; Devendra Singh Parihar; Daphne Naidoo; Kimsey Zajac; Willy Ssengooba; Grant Theron; Thomas Niesler
>
> **摘要:** The automatic identification of cough segments in audio through the determination of start and end points is pivotal to building scalable screening tools in health technologies for pulmonary related diseases. We propose the application of two current pre-trained architectures to the task of cough activity detection. A dataset of recordings containing cough from patients symptomatic for tuberculosis (TB) who self-present at community-level care centres in South Africa and Uganda is employed. When automatic start and end points are determined using XLS-R, an average precision of 0.96 and an area under the receiver-operating characteristic of 0.99 are achieved for the test set. We show that best average precision is achieved by utilising only the first three layers of the network, which has the dual benefits of reduced computational and memory requirements, pivotal for smartphone-based applications. This XLS-R configuration is shown to outperform an audio spectrogram transformer (AST) as well as a logistic regression baseline by 9% and 27% absolute in test set average precision respectively. Furthermore, a downstream TB classification model trained using the coughs automatically isolated by XLS-R comfortably outperforms a model trained on the coughs isolated by AST, and is only narrowly outperformed by a classifier trained on the ground truth coughs. We conclude that the application of large pre-trained transformer models is an effective approach to identifying cough end-points and that the integration of such a model into a screening tool is feasible.
>
---
#### [new 006] Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音生物识别任务，解决性别相关的性能差异问题。提出Fair-Gate框架，通过风险外推和特征路由提升公平性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.11360](https://arxiv.org/pdf/2603.11360)**

> **作者:** Yangyang Qu; Todisco Massimiliano; Galdi Chiara; Evans Nicholas
>
> **摘要:** Voice biometric systems can exhibit sex-related performance gaps even when overall verification accuracy is strong. We attribute these gaps to two practical mechanisms: (i) demographic shortcut learning, where speaker classification training exploits spurious correlations between sex and speaker identity, and (ii) feature entanglement, where sex-linked acoustic variation overlaps with identity cues and cannot be removed without degrading speaker discrimination. We propose Fair-Gate, a fairness-aware and interpretable risk-gating framework that addresses both mechanisms in a single pipeline. Fair-Gate applies risk extrapolation to reduce variation in speaker-classification risk across proxy sex groups, and introduces a local complementary gate that routes intermediate features into an identity branch and a sex branch. The gate provides interpretability by producing an explicit routing mask that can be inspected to understand which features are allocated to identity versus sex-related pathways. Experiments on VoxCeleb1 show that Fair-Gate improves the utility--fairness trade-off, yielding more sex-fair ASV performance under challenging evaluation conditions.
>
---
#### [new 007] Affect Decoding in Phonated and Silent Speech Production from Surface EMG
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于情感解码任务，旨在通过表面肌电信号（sEMG）识别情绪，解决情感与发音机制关系不明确的问题。研究采集数据并评估不同特征和模型的解码效果。**

- **链接: [https://arxiv.org/pdf/2603.11715](https://arxiv.org/pdf/2603.11715)**

> **作者:** Simon Pistrosch; Kleanthis Avramidis; Tiantian Feng; Jihwan Lee; Monica Gonzalez-Machorro; Shrikanth Narayanan; Björn W. Schuller
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The expression of affect is integral to spoken communication, yet, its link to underlying articulatory execution remains unclear. Measures of articulatory muscle activity such as EMG could reveal how speech production is modulated by emotion alongside acoustic speech analyses. We investigate affect decoding from facial and neck surface electromyography (sEMG) during phonated and silent speech production. For this purpose, we introduce a dataset comprising 2,780 utterances from 12 participants across 3 tasks, on which we evaluate both intra- and inter-subject decoding using a range of features and model embeddings. Our results reveal that EMG representations reliably discriminate frustration with up to 0.845 AUC, and generalize well across articulation modes. Our ablation study further demonstrates that affective signatures are embedded in facial motor activity and persist in the absence of phonation, highlighting the potential of EMG sensing for affect-aware silent speech interfaces.
>
---
#### [new 008] Reconstruction of the Vocal Tract from Speech via Phonetic Representations Using MRI Data
- **分类: eess.AS**

- **简介: 该论文属于语音到声门结构重建任务，旨在通过语音信号重构声道几何形状。研究比较了不同层次的语音分段精度，验证了结合语音信息对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.11847](https://arxiv.org/pdf/2603.11847)**

> **作者:** Sofiane Azzouz; Pierre-André Vuissoz; Yves Laprie
>
> **摘要:** Articulatory acoustic inversion aims to reconstruct the complete geometry of the vocal tract from the speech signal. In this paper, we present a comparative study of several levels of phonetic segmentation accuracy, together with a comparison to the baseline introduced in our previous work, which is based on Mel-Frequency Cepstral Coefficients (MFCCs). All the approaches considered are based on a denoised speech signal and aim to investigate the impact of incorporating phonetic information through three successive levels: an uncorrected automatic transcription, a temporally aligned phonetic segmentation, and an expert manual correction following alignment. The models are trained to predict articulatory contours extracted from vocal tract MRI images using an automatic contour tracking method. The results show that, among the models relying on phonetic representations, manual correction after alignment yields the best performance, approaching that of the baseline.
>
---
#### [new 009] Resonate: Reinforcing Text-to-Audio Generation via Online Feedback from Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文属于文本到音频生成任务，旨在提升生成质量与语义对齐。通过在线强化学习和大音频语言模型奖励，提出Resonate模型，取得新SOTA。**

- **链接: [https://arxiv.org/pdf/2603.11661](https://arxiv.org/pdf/2603.11661)**

> **作者:** Xiquan Li; Junxi Liu; Wenxi Chen; Haina Zhu; Ziyang Ma; Xie Chen
>
> **摘要:** Reinforcement Learning (RL) has become an effective paradigm for enhancing Large Language Models (LLMs) and visual generative models. However, its application in text-to-audio (TTA) generation remains largely under-explored. Prior work typically employs offline methods like Direct Preference Optimization (DPO) and leverages Contrastive Language-Audio Pretraining (CLAP) models as reward functions. In this study, we investigate the integration of online Group Relative Policy Optimization (GRPO) into TTA generation. We adapt the algorithm for Flow Matching-based audio models and demonstrate that online RL significantly outperforms its offline counterparts. Furthermore, we incorporate rewards derived from Large Audio Language Models (LALMs), which can provide fine-grained scoring signals that are better aligned with human perception. With only 470M parameters, our final model, \textbf{Resonate}, establishes a new SOTA on TTA-Bench in terms of both audio quality and semantic alignment.
>
---
#### [new 010] Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review
- **分类: eess.AS**

- **简介: 该论文属于人机交互领域，解决传统语音交互的局限性，通过分析SSIs技术，提出基于大语言模型的语义对齐方法，推动其在可穿戴设备中的应用。**

- **链接: [https://arxiv.org/pdf/2603.11877](https://arxiv.org/pdf/2603.11877)**

> **作者:** Kele Xu; Yifan Wang; Ming Feng; Qisheng Xu; Wuyang Chen; Yutao Dou; Cheng Yang; Huaimin Wang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Human-computer interaction has traditionally relied on the acoustic channel, a dependency that introduces systemic vulnerabilities to environmental noise, privacy constraints, and physiological speech impairments. Silent Speech Interfaces (SSIs) emerge as a transformative paradigm that bypasses the acoustic stage by decoding linguistic intent directly from the neuro-muscular-articulatory continuum. This review provides a high-level synthesis of the SSI landscape, transitioning from traditional transducer-centric analysis to a holistic intent-to-execution taxonomy. We systematically evaluate sensing modalities across four critical physiological interception points: neural oscillations, neuromuscular activation, articulatory kinematics (ultrasound/magnetometry), and pervasive active probing via acoustic or radio-frequency sensing. Critically, we analyze the current paradigm shift from heuristic signal processing to Latent Semantic Alignment. In this new era, Large Language Models (LLMs) and deep generative architectures serve as high-level linguistic priors to resolve the ``informational sparsity'' and non-stationarity of biosignals. By mapping fragmented physiological gestures into structured semantic latent spaces, modern SSI frameworks have, for the first time, approached the Word Error Rate usability threshold required for real-world deployment. We further examine the transition of SSIs from bulky laboratory instrumentation to ``invisible interfaces'' integrated into commodity-grade wearables, such as earables and smart glasses. Finally, we outline a strategic roadmap addressing the ``user-dependency paradox'' through self-supervised foundation models and define the ethical boundaries of ``neuro-security'' to protect cognitive liberty in an increasingly interfaced world.
>
---
#### [new 011] AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音评估任务，旨在解决缺乏客观评价动漫风格语音标准的问题。通过构建AnimeScore框架，利用偏好排序进行自动评估。**

- **链接: [https://arxiv.org/pdf/2603.11482](https://arxiv.org/pdf/2603.11482)**

> **作者:** Joonyong Park; Jerry Li
>
> **摘要:** Evaluating 'anime-like' voices currently relies on costly subjective judgments, yet no standardized objective metric exists. A key challenge is that anime-likeness, unlike naturalness, lacks a shared absolute scale, making conventional Mean Opinion Score (MOS) protocols unreliable. To address this gap, we propose AnimeScore, a preference-based framework for automatic anime-likeness evaluation via pairwise ranking. We collect 15,000 pairwise judgments from 187 evaluators with free-form descriptions, and acoustic analysis reveals that perceived anime-likeness is driven by controlled resonance shaping, prosodic continuity, and deliberate articulation rather than simple heuristics such as high pitch. We show that handcrafted acoustic features reach a 69.3% AUC ceiling, while SSL-based ranking models achieve up to 90.8% AUC, providing a practical metric that can also serve as a reward signal for preference-based optimization of generative speech models.
>
---
#### [new 012] Causal Prosody Mediation for Text-to-Speech:Counterfactual Training of Duration, Pitch, and Energy in FastSpeech2
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于文本到语音合成任务，旨在提升情感表达的可控性。通过引入因果建模和反事实训练，分离情感与语言内容对韵律的影响，增强语音表现力。**

- **链接: [https://arxiv.org/pdf/2603.11683](https://arxiv.org/pdf/2603.11683)**

> **作者:** Suvendu Sekhar Mohanty
>
> **摘要:** We propose a novel causal prosody mediation framework for expressive text-to-speech (TTS) synthesis. Our approach augments the FastSpeech2 architecture with explicit emotion conditioning and introduces counterfactual training objectives to disentangle emotional prosody from linguistic content. By formulating a structural causal model of how text (content), emotion, and speaker jointly influence prosody (duration, pitch, energy) and ultimately the speech waveform, we derive two complementary loss terms: an Indirect Path Constraint (IPC) to enforce that emotion affects speech only through prosody, and a Counterfactual Prosody Constraint (CPC) to encourage distinct prosody patterns for different emotions. The resulting model is trained on multi-speaker emotional corpora (LibriTTS, EmoV-DB, VCTK) with a combined objective that includes standard spectrogram reconstruction and variance prediction losses alongside our causal losses. In evaluations on expressive speech synthesis, our method achieves significantly improved prosody manipulation and emotion rendering, with higher mean opinion scores (MOS) and emotion accuracy than baseline FastSpeech2 variants. We also observe better intelligibility (low WER) and speaker consistency when transferring emotions across speakers. Extensive ablations confirm that the causal objectives successfully separate prosody attribution, yielding an interpretable model that allows controlled counterfactual prosody editing (e.g. "same utterance, different emotion") without compromising naturalness. We discuss the implications for identifiability in prosody modeling and outline limitations such as the assumption that emotion effects are fully captured by pitch, duration, and energy. Our work demonstrates how integrating causal learning principles into TTS can improve controllability and expressiveness in generated speech.
>
---
#### [new 013] Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于自动语音识别任务，解决LLM与ASR融合在低延迟流式场景中的部署难题。提出Uni-ASR框架，实现非流式与流式识别的统一，提升流式识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.11123](https://arxiv.org/pdf/2603.11123)**

> **作者:** Yinfeng Xia; Jian Tang; Junfeng Hou; Gaopeng Xu; Haitao Yao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Although the deep integration of the Automatic Speech Recognition (ASR) system with Large Language Models (LLMs) has significantly improved accuracy, the deployment of such systems in low-latency streaming scenarios remains challenging. In this paper, we propose Uni-ASR, a unified framework based on LLMs that integrates both non-streaming and streaming speech recognition capabilities. We propose a joint training paradigm that enables the system to seamlessly transition between two recognition modes without any architectural modifications. Furthermore, we introduce a context-aware training paradigm and a co-designed fallback decoding strategy, which can enhance streaming recognition accuracy without introducing additional latency. The experimental results demonstrate that Uni-ASR not only achieves competitive performance within non-streaming mode, but also demonstrates strong effectiveness in streaming scenarios under diverse latency constraints.
>
---
#### [new 014] SEMamba++: A General Speech Restoration Framework Leveraging Global, Local, and Periodic Spectral Patterns
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音恢复任务，旨在解决复杂噪声下语音质量提升问题。提出SEMamba++框架，融合全局、局部和周期性频谱特征，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2603.11669](https://arxiv.org/pdf/2603.11669)**

> **作者:** Yongjoon Lee; Jung-Woo Choi
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** General speech restoration demands techniques that can interpret complex speech structures under various distortions. While State-Space Models like SEMamba have advanced the state-of-the-art in speech denoising, they are not inherently optimized for critical speech characteristics, such as spectral periodicity or multi-resolution frequency analysis. In this work, we introduce an architecture tailored to incorporate speech-specific features as inductive biases. In particular, we propose Frequency GLP, a frequency feature extraction block that effectively and efficiently leverages the properties of frequency bins. Then, we design a multi-resolution parallel time-frequency dual-processing block to capture diverse spectral patterns, and a learnable mapping to further enhance model performance. With all our ideas combined, the proposed SEMamba++ achieves the best performance among multiple baseline models while remaining computationally efficient.
>
---
#### [new 015] Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉语音识别任务，解决模型如何平衡声学与视觉信息的问题。通过Shapley值分析模态贡献，揭示噪声下模态依赖变化及动态平衡机制。**

- **链接: [https://arxiv.org/pdf/2603.12046](https://arxiv.org/pdf/2603.12046)**

> **作者:** Umberto Cappellazzo; Stavros Petridis; Maja Pantic
>
> **备注:** Project website: this https URL
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) leverages both acoustic and visual information for robust recognition under noise. However, how models balance these modalities remains unclear. We present Dr. SHAP-AV, a framework using Shapley values to analyze modality contributions in AVSR. Through experiments on six models across two benchmarks and varying SNR levels, we introduce three analyses: Global SHAP for overall modality balance, Generative SHAP for contribution dynamics during decoding, and Temporal Alignment SHAP for input-output correspondence. Our findings reveal that models shift toward visual reliance under noise yet maintain high audio contributions even under severe degradation. Modality balance evolves during generation, temporal alignment holds under noise, and SNR is the dominant factor driving modality weighting. These findings expose a persistent audio bias, motivating ad-hoc modality-weighting mechanisms and Shapley-based attribution as a standard AVSR diagnostic.
>
---
#### [new 016] Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在加速自回归解码并提升ASR准确率。通过结合CTC编码器生成草案，提出自推测解码方法，有效降低WER并提高实时性。**

- **链接: [https://arxiv.org/pdf/2603.11243](https://arxiv.org/pdf/2603.11243)**

> **作者:** George Saon; Samuel Thomas; Takashi Fukuda; Tohru Nagano; Avihu Dekel; Luis Lastras
>
> **摘要:** We propose self-speculative decoding for speech-aware LLMs by using the CTC encoder as a draft model to accelerate auto-regressive (AR) inference and improve ASR accuracy. Our three-step procedure works as follows: (1) if the frame entropies of the CTC output distributions are below a threshold, the greedy CTC hypothesis is accepted as final; (2) otherwise, the CTC hypothesis is verified in a single LLM forward pass using a relaxed acceptance criterion based on token likelihoods; (3) if verification fails, AR decoding resumes from the accepted CTC prefix. Experiments on nine corpora and five languages show that this approach can simultaneously accelerate decoding and reduce WER. On the HuggingFace Open ASR benchmark with a 1B parameter LLM and 440M parameter CTC encoder, we achieve a record 5.58% WER and improve the inverse real time factor by a factor of 4.4 with only a 12% relative WER increase over AR search. Code and model weights are publicly available under a permissive license.
>
---
#### [new 017] Can LLMs Help Localize Fake Words in Partially Fake Speech?
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音伪造检测任务，旨在定位部分伪造语音中的虚假词语。通过构建语音LLM，利用下一个词预测进行虚假词定位，发现模型依赖特定编辑模式，但需提升对未知编辑风格的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.11205](https://arxiv.org/pdf/2603.11205)**

> **作者:** Lin Zhang; Thomas Thebaud; Zexin Cai; Sanjeev Khudanpur; Daniel Povey; Leibny Paola García-Perera; Matthew Wiesner; Nicholas Andrews
>
> **备注:** Submitted to Interspeech 2026; put on arxiv based on requirement from Interspeech: "Interspeech no longer enforces an anonymity period for submissions." and "For authors that prefer to upload their paper online, a note indicating that the paper was submitted for review to Interspeech should be included in the posting."
>
> **摘要:** Large language models (LLMs), trained on large-scale text, have recently attracted significant attention for their strong performance across many tasks. Motivated by this, we investigate whether a text-trained LLM can help localize fake words in partially fake speech, where only specific words within a speech are edited. We build a speech LLM to perform fake word localization via next token prediction. Experiments and analyses on AV-Deepfake1M and PartialEdit indicates that the model frequently leverages editing-style pattern learned from the training data, particularly word-level polarity substitutions for those two databases we discussed, as cues for localizing fake words. Although such particular patterns provide useful information in an in-domain scenario, how to avoid over-reliance on such particular pattern and improve generalization to unseen editing styles remains an open question.
>
---
#### [new 018] Toward Complex-Valued Neural Networks for Waveform Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决传统神经声码器在处理复数谱时的结构限制。提出ComVo模型，采用复数运算提升波形生成质量，并优化训练效率。**

- **链接: [https://arxiv.org/pdf/2603.11589](https://arxiv.org/pdf/2603.11589)**

> **作者:** Hyung-Seok Oh; Deok-Hyeon Cho; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** ICLR 2026 (accepted)
>
> **摘要:** Neural vocoders have recently advanced waveform generation, yielding natural and expressive audio. Among these approaches, iSTFT-based vocoders have recently gained attention. They predict a complex-valued spectrogram and then synthesize the waveform via iSTFT, thereby avoiding learned upsampling stages that can increase computational cost. However, current approaches use real-valued networks that process the real and imaginary parts independently. This separation limits their ability to capture the inherent structure of complex spectrograms. We present ComVo, a Complex-valued neural Vocoder whose generator and discriminator use native complex arithmetic. This enables an adversarial training framework that provides structured feedback in complex-valued representations. To guide phase transformations in a structured manner, we introduce phase quantization, which discretizes phase values and regularizes the training process. Finally, we propose a block-matrix computation scheme to improve training efficiency by reducing redundant operations. Experiments demonstrate that ComVo achieves higher synthesis quality than comparable real-valued baselines, and that its block-matrix scheme reduces training time by 25%. Audio samples and code are available at this https URL.
>
---
#### [new 019] ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping
- **分类: eess.AS**

- **简介: 该论文提出ReDimNet2，用于说话人验证任务，通过时间池化实现更高效的特征提取，解决计算成本与准确率的平衡问题。**

- **链接: [https://arxiv.org/pdf/2603.11841](https://arxiv.org/pdf/2603.11841)**

> **作者:** Ivan Yakovlev; Anton Okhotnikov
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We present ReDimNet2, an improved neural network architecture for extracting utterance-level speaker representations that builds upon the ReDimNet dimension-reshaping framework. The key modification in ReDimNet2 is the introduction of pooling over the time dimension within the 1D processing pathway. This operation preserves the nature of the 1D feature space, since 1D features remain a reshaped version of 2D features regardless of temporal resolution, while enabling significantly more aggressive scaling of the channel dimension without proportional compute increase. We introduce a family of seven model configurations (B0-B6) ranging from 1.1M to 12.3M parameters and 0.33 to 13 GMACS. Experimental results on VoxCeleb1 benchmarks demonstrate that ReDimNet2 improves the Pareto front of computational cost versus accuracy at every scale point compared to ReDimNet, achieving 0.287% EER on Vox1-O with 12.3M parameters and 13 GMACS.
>
---
#### [new 020] V2A-DPO: Omni-Preference Optimization for Video-to-Audio Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文针对视频到音频生成任务，提出V2A-DPO框架，解决生成音频与人类偏好对齐问题，通过评分系统、数据生成管道和优化策略提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.11089](https://arxiv.org/pdf/2603.11089)**

> **作者:** Nolan Chan; Timmy Gang; Yongqian Wang; Yuzhe Liang; Dingdong Wang
>
> **备注:** Accepted at ICASSP2026
>
> **摘要:** This paper introduces V2A-DPO, a novel Direct Preference Optimization (DPO) framework tailored for flow-based video-to-audio generation (V2A) models, incorporating key adaptations to effectively align generated audio with human preferences. Our approach incorporates three core innovations: (1) AudioScore-a comprehensive human preference-aligned scoring system for assessing semantic consistency, temporal alignment, and perceptual quality of synthesized audio; (2) an automated AudioScore-driven pipeline for generating large-scale preference pair data for DPO optimization; (3) a curriculum learning-empowered DPO optimization strategy specifically tailored for flow-based generative models. Experiments on benchmark VGGSound dataset demonstrate that human-preference aligned Frieren and MMAudio using V2A-DPO outperform their counterparts optimized using Denoising Diffusion Policy Optimization (DDPO) as well as pre-trained baselines. Furthermore, our DPO-optimized MMAudio achieves state-of-the-art performance across multiple metrics, surpassing published V2A models.
>
---
#### [new 021] RAF: Relativistic Adversarial Feedback For Universal Speech Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，解决GAN声码器泛化能力不足的问题。提出RAF训练目标，利用自监督学习提升生成质量，实验表明效果优于传统方法。**

- **链接: [https://arxiv.org/pdf/2603.11678](https://arxiv.org/pdf/2603.11678)**

> **作者:** Yongjoon Lee; Jung-Woo Choi
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We propose Relativistic Adversarial Feedback (RAF), a novel training objective for GAN vocoders that improves in-domain fidelity and generalization to unseen scenarios. Although modern GAN vocoders employ advanced architectures, their training objectives often fail to promote generalizable representations. RAF addresses this problem by leveraging speech self-supervised learning models to assist discriminators in evaluating sample quality, encouraging the generator to learn richer representations. Furthermore, we utilize relativistic pairing for real and fake waveforms to improve the modeling of the training data distribution. Experiments across multiple datasets show consistent gains in both objective and subjective metrics on GAN-based vocoders. Importantly, the RAF-trained BigVGAN-base outperforms the LSGAN-trained BigVGAN in perceptual quality using only 12\% of the parameters. Comparative studies further confirm the effectiveness of RAF as a training framework for GAN vocoders.
>
---
#### [new 022] Huntington Disease Automatic Speech Recognition with Biomarker Supervision
- **分类: cs.LG; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决亨廷顿病患者语音识别问题。通过分析HD语音特征，提出改进模型和生物标志物辅助方法，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.11168](https://arxiv.org/pdf/2603.11168)**

> **作者:** Charles L. Wang; Cady Chen; Ziwei Gong; Julia Hirschberg
>
> **摘要:** Automatic speech recognition (ASR) for pathological speech remains underexplored, especially for Huntington's disease (HD), where irregular timing, unstable phonation, and articulatory distortion challenge current models. We present a systematic HD-ASR study using a high-fidelity clinical speech corpus not previously used for end-to-end ASR training. We compare multiple ASR families under a unified evaluation, analyzing WER as well as substitution, deletion, and insertion patterns. HD speech induces architecture-specific error regimes, with Parakeet-TDT outperforming encoder-decoder and CTC baselines. HD-specific adaptation reduces WER from 6.99% to 4.95% and we also propose a method for using biomarker-based auxiliary supervision and analyze how error behavior is reshaped in severity-dependent ways rather than uniformly improving WER. We open-source all code and models.
>
---
#### [new 023] Stage-Adaptive Reliability Modeling for Continuous Valence-Arousal Estimation
- **分类: cs.MM; cs.AI; cs.SD**

- **简介: 该论文属于连续情绪估计任务，解决多模态信号可靠性不一致问题。提出SAGE框架，通过自适应可靠性建模提升情绪预测稳定性。**

- **链接: [https://arxiv.org/pdf/2603.11468](https://arxiv.org/pdf/2603.11468)**

> **作者:** Yubeen Lee; Sangeun Lee; Junyeop Cha; Eunil Park
>
> **备注:** 8 pages, 3 figures, 2 pages
>
> **摘要:** Continuous valence-arousal estimation in real-world environments is challenging due to inconsistent modality reliability and interaction-dependent variability in audio-visual signals. Existing approaches primarily focus on modeling temporal dynamics, often overlooking the fact that modality reliability can vary substantially across interaction stages. To address this issue, we propose SAGE, a Stage-Adaptive reliability modeling framework that explicitly estimates and calibrates modality-wise confidence during multimodal integration. SAGE introduces a reliability-aware fusion mechanism that dynamically rebalances audio and visual representations according to their stage-dependent informativeness, preventing unreliable signals from dominating the prediction process. By separating reliability estimation from feature representation, the proposed framework enables more stable emotion estimation under cross-modal noise, occlusion, and varying interaction conditions. Extensive experiments on the Aff-Wild2 benchmark demonstrate that SAGE consistently improves concordance correlation coefficient scores compared with existing multimodal fusion approaches, highlighting the effectiveness of reliability-driven modeling for continuous affect prediction.
>
---
#### [new 024] OmniForcing: Unleashing Real-time Joint Audio-Visual Generation
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出OmniForcing，解决实时音视频生成中的延迟问题。通过蒸馏技术将双向模型转为流式自回归生成，提升效率并保持同步与质量。**

- **链接: [https://arxiv.org/pdf/2603.11647](https://arxiv.org/pdf/2603.11647)**

> **作者:** Yaofeng Su; Yuming Li; Zeyue Xue; Jie Huang; Siming Fu; Haoran Li; Ying Li; Zezhong Qian; Haoyang Huang; Nan Duan
>
> **备注:** 14 pages
>
> **摘要:** Recent joint audio-visual diffusion models achieve remarkable generation quality but suffer from high latency due to their bidirectional attention dependencies, hindering real-time applications. We propose OmniForcing, the first framework to distill an offline, dual-stream bidirectional diffusion model into a high-fidelity streaming autoregressive generator. However, naively applying causal distillation to such dual-stream architectures triggers severe training instability, due to the extreme temporal asymmetry between modalities and the resulting token sparsity. We address the inherent information density gap by introducing an Asymmetric Block-Causal Alignment with a zero-truncation Global Prefix that prevents multi-modal synchronization drift. The gradient explosion caused by extreme audio token sparsity during the causal shift is further resolved through an Audio Sink Token mechanism equipped with an Identity RoPE constraint. Finally, a Joint Self-Forcing Distillation paradigm enables the model to dynamically self-correct cumulative cross-modal errors from exposure bias during long rollouts. Empowered by a modality-independent rolling KV-cache inference scheme, OmniForcing achieves state-of-the-art streaming generation at $\sim$25 FPS on a single GPU, maintaining multi-modal synchronization and visual quality on par with the bidirectional teacher.\textbf{Project Page:} \href{this https URL}{this https URL}
>
---
#### [new 025] Multimodal Self-Attention Network with Temporal Alignment for Audio-Visual Emotion Recognition
- **分类: cs.MM; cs.SD; eess.SP**

- **简介: 该论文属于音频-视觉情感识别任务，解决多模态帧率不匹配问题。提出基于Transformer的框架，通过时间对齐和跨时序匹配提升特征融合效果。**

- **链接: [https://arxiv.org/pdf/2603.11095](https://arxiv.org/pdf/2603.11095)**

> **作者:** Inyong Koo; yeeun Seong; Minseok Son; Jaehyuk Jang; Changick Kim
>
> **备注:** 5 pages, 3 figures, accepted to ICASSP 2026
>
> **摘要:** Audio-visual emotion recognition (AVER) methods typically fuse utterance-level features, and even frame-level attention models seldom address the frame-rate mismatch across modalities. In this paper, we propose a Transformer-based framework focusing on the temporal alignment of multimodal features. Our design employs a multimodal self-attention encoder that simultaneously captures intra- and inter-modal dependencies within a shared feature space. To address heterogeneous sampling rates, we incorporate Temporally-aligned Rotary Position Embeddings (TaRoPE), which implicitly synchronize audio and video tokens. Furthermore, we introduce a Cross-Temporal Matching (CTM) loss that enforces consistency among temporally proximate pairs, guiding the encoder toward better alignment. Experiments on CREMA-D and RAVDESS datasets demonstrate consistent improvements over recent baselines, suggesting that explicitly addressing frame-rate mismatch helps preserve temporal cues and enhances cross-modal fusion.
>
---
## 更新

#### [replaced 001] Community-Informed AI Models for Police Accountability
- **分类: cs.CY; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于政府问责任务，旨在通过AI分析警察与公众互动，提升透明度。工作包括整合社区视角，开发多角度AI工具。**

- **链接: [https://arxiv.org/pdf/2402.01703](https://arxiv.org/pdf/2402.01703)**

> **作者:** Benjamin A.T. Grahama; Lauren Brown; Georgios Chochlakis; Morteza Dehghani; Raquel Delerme; Brittany Friedman; Ellie Graeden; Preni Golazizian; Rajat Hebbar; Parsa Hejabi; Aditya Kommineni; Mayagüez Salinas; Michael Sierra-Arévalo; Jackson Trager; Nicholas Weller; Shrikanth Narayanan
>
> **备注:** 33 pages, 4 figures, 2 tables
>
> **摘要:** Face-to-face interactions between police officers and the public affect both individual well-being and democratic legitimacy. Many government-public interactions are captured on video, including interactions between police officers and drivers captured on bodyworn cameras (BWCs). New advances in AI technology enable these interactions to be analyzed at scale, opening promising avenues for improving government transparency and accountability. However, for AI to serve democratic governance effectively, models must be designed to include the preferences and perspectives of the governed. This article proposes a community-informed, approach to developing multi-perspective AI tools for government accountability. We illustrate our approach by describing the research project through which the approach was inductively developed: an effort to build AI tools to analyze BWC footage of traffic stops conducted by the Los Angeles Police Department. We focus on the role of social scientists as members of multidisciplinary teams responsible for integrating the perspectives of diverse stakeholders into the development of AI tools in the domain of police -- and government -- accountability.
>
---
#### [replaced 002] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频大模型可信性评估任务，旨在解决现有框架无法有效评估音频模型安全性和可靠性的问题。工作包括构建AudioTrust框架，涵盖六维评估指标及真实场景数据集，全面测试模型在多种高风险音频场景下的表现。**

- **链接: [https://arxiv.org/pdf/2505.16211](https://arxiv.org/pdf/2505.16211)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Lionel Z. Wang; Shun Zhang; Xingjian Du; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Gelei Deng; Haoyang Li; Yiming Li; Xiaobin Zhuang; Tianlong Chen; Qingsong Wen; Tianwei Zhang; Yang Liu; Haibo Hu; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; Wenyuan Xu; XiaoFeng Wang; Wei Dong; Xinfeng Li
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The rapid development and widespread adoption of Audio Large Language Models (ALLMs) demand rigorous evaluation of their trustworthiness. However, existing evaluation frameworks are primarily designed for text and fail to capture vulnerabilities introduced by the acoustic properties of audio. We find that significant trustworthiness risks in ALLMs arise from non-semantic acoustic cues, such as timbre, accent, and background noise, which can be exploited to manipulate model behavior. To address this gap, we propose AudioTrust, the first large-scale and systematic framework for evaluating ALLM trustworthiness under audio-specific risks. AudioTrust covers six key dimensions: fairness, hallucination, safety, privacy, robustness, and authenticition. It includes 26 sub-tasks and a curated dataset of more than 4,420 audio samples collected from real-world scenarios, including daily conversations, emergency calls, and voice assistant interactions, and is specifically designed to probe trustworthiness across multiple dimensions. Our comprehensive evaluation spans 18 experimental settings and uses human-validated automated pipelines to enable objective and scalable assessment of model outputs. Experimental results on 14 state-of-the-art open-source and closed-source ALLMs reveal important limitations and failure boundaries under diverse high-risk audio scenarios, providing critical insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are publicly available at this https URL.
>
---
#### [replaced 003] Probabilistic Verification of Voice Anti-Spoofing Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音防欺骗任务，解决现有模型缺乏鲁棒性保证的问题。提出PV-VASM框架，验证语音反欺骗模型的鲁棒性，有效应对未知合成技术。**

- **链接: [https://arxiv.org/pdf/2603.10713](https://arxiv.org/pdf/2603.10713)**

> **作者:** Evgeny Kushnir; Alexandr Kozodaev; Dmitrii Korzh; Mikhail Pautov; Oleg Kiriukhin; Oleg Y. Rogov
>
> **备注:** The paper was submitted for review to Interspeech 2026
>
> **摘要:** Recent advances in generative models have amplified the risk of malicious misuse of speech synthesis technologies, enabling adversaries to impersonate target speakers and access sensitive resources. Although speech deepfake detection has progressed rapidly, most existing countermeasures lack formal robustness guarantees or fail to generalize to unseen generation techniques. We propose PV-VASM, a probabilistic framework for verifying the robustness of voice anti-spoofing models (VASMs). PV-VASM estimates the probability of misclassification under text-to-speech (TTS), voice cloning (VC), and parametric signal transformations. The approach is model-agnostic and enables robustness verification against unseen speech synthesis techniques and input perturbations. We derive a theoretical upper bound on the error probability and validate the method across diverse experimental settings, demonstrating its effectiveness as a practical robustness verification tool.
>
---
#### [replaced 004] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，解决语音信息如何编码的问题。通过分析96种语言，发现模型中存在可解释的音系向量，并展示其可计算性。**

- **链接: [https://arxiv.org/pdf/2602.18899](https://arxiv.org/pdf/2602.18899)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Submitted to ACL, code planned to release after acceptance
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at this https URL .
>
---
#### [replaced 005] Text-only adaptation in LLM-based ASR through text denoising
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，解决LLM在新领域适应的问题。通过文本去噪方法，在不破坏跨模态对齐的情况下，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20900](https://arxiv.org/pdf/2601.20900)**

> **作者:** Andrés Carofilis; Sergio Burdisso; Esaú Villatoro-Tello; Shashi Kumar; Kadri Hacioglu; Srikanth Madikeri; Pradeep Rangappa; Manjunath K E; Petr Motlicek; Shankar Venkatesan; Andreas Stolcke
>
> **摘要:** Adapting large language model (LLM)-based automatic speech recognition (ASR) systems to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on the target domain text often disrupts the critical alignment between the speech and text modality learned by the projector, degrading performance. We introduce a novel text-only adaptation method that frames this process as a text denoising task. Our approach trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
>
---
#### [replaced 006] Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有方法泛化能力差和缺乏可解释性的问题。提出HIR-SDD框架，结合大音频语言模型与人类推理，提升检测效果并提供合理解释。**

- **链接: [https://arxiv.org/pdf/2603.10725](https://arxiv.org/pdf/2603.10725)**

> **作者:** Artem Dvirniak; Evgeny Kushnir; Dmitrii Tarasov; Artem Iudin; Oleg Kiriukhin; Mikhail Pautov; Dmitrii Korzh; Oleg Y. Rogov
>
> **摘要:** The modern generative audio models can be used by an adversary in an unlawful manner, specifically, to impersonate other people to gain access to private information. To mitigate this issue, speech deepfake detection (SDD) methods started to evolve. Unfortunately, current SDD methods generally suffer from the lack of generalization to new audio domains and generators. More than that, they lack interpretability, especially human-like reasoning that would naturally explain the attribution of a given audio to the bona fide or spoof class and provide human-perceptible cues. In this paper, we propose HIR-SDD, a novel SDD framework that combines the strengths of Large Audio Language Models (LALMs) with the chain-of-thought reasoning derived from the novel proposed human-annotated dataset. Experimental evaluation demonstrates both the effectiveness of the proposed method and its ability to provide reasonable justifications for predictions.
>
---
#### [replaced 007] Audio-Language Models for Audio-Centric Tasks: A Systematic Survey
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 本文综述音频-语言模型（ALMs），解决多模态音频理解问题。系统梳理ALMs的架构、训练目标及研究现状，为后续研究提供方向与参考。**

- **链接: [https://arxiv.org/pdf/2501.15177](https://arxiv.org/pdf/2501.15177)**

> **作者:** Yi Su; Jisheng Bai; Qisheng Xu; Kele Xu; Yong Dou
>
> **备注:** Under review
>
> **摘要:** Audio-Language Models (ALMs), trained on paired audio-text data, are designed to process, understand, and reason about audio-centric multimodal content. Unlike traditional supervised approaches that use predefined labels, ALMs leverage natural language supervision to better handle complex real-world audio scenes with multiple overlapping events. While demonstrating impressive zero-shot and task generalization capabilities, there is still a notable lack of systematic surveys that comprehensively organize and analyze developments. In this paper, we present the first systematic review of ALMs with three main contributions: (1) comprehensive coverage of ALM works across speech, music, and sound from a general audio perspective; (2) a unified taxonomy of ALM foundations, including model architectures and training objectives; (3) establishment of a research landscape capturing mutual promotion and constraints among different research aspects, aiding in summarizing evaluations, limitations, concerns and promising directions. Our review contributes to helping researchers understand the development of existing technologies and future trends, while also providing valuable references for implementation in practical applications.
>
---
