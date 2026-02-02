# 音频 cs.SD;  eess.AS

- **最新发布 24 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Streaming Speech Recognition with Decoder-Only Large Language Models and Latency Optimization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，解决流式ASR中实时性与准确性的平衡问题。通过引入读写策略网络和MoChA机制，实现动态语音分段与高效推理。**

- **链接: [https://arxiv.org/pdf/2601.22779v1](https://arxiv.org/pdf/2601.22779v1)**

> **作者:** Genshun Wan; Wenhui Zhang; Jing-Xuan Zhang; Shifu Xiong; Jianqing Gao; Zhongfu Ye
>
> **备注:** accepted to ICASSP 2026
>
> **摘要:** Recent advances have demonstrated the potential of decoderonly large language models (LLMs) for automatic speech recognition (ASR). However, enabling streaming recognition within this framework remains a challenge. In this work, we propose a novel streaming ASR approach that integrates a read/write policy network with monotonic chunkwise attention (MoChA) to dynamically segment speech embeddings. These segments are interleaved with label sequences during training, enabling seamless integration with the LLM. During inference, the audio stream is buffered until the MoChA module triggers a read signal, at which point the buffered segment together with the previous token is fed into the LLM for the next token prediction. We also introduce a minimal-latency training objective to guide the policy network toward accurate segmentation boundaries. Furthermore, we adopt a joint training strategy in which a non-streaming LLM-ASR model and our streaming model share parameters. Experiments on the AISHELL-1 and AISHELL-2 Mandarin benchmarks demonstrate that our method consistently outperforms recent streaming ASR baselines, achieving character error rates of 5.1% and 5.5%, respectively. The latency optimization results in a 62.5% reduction in average token generation delay with negligible impact on recognition accuracy
>
---
#### [new 002] A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation
- **分类: cs.SD; cs.HC**

- **简介: 该论文属于声音分离任务，旨在解决复杂声学场景中残留干扰问题。通过构建高质量数据集Hive，提升模型的数据效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.22599v1](https://arxiv.org/pdf/2601.22599v1)**

> **作者:** Kai Li; Jintao Cheng; Chang Zeng; Zijun Yan; Helin Wang; Zixiong Su; Bo Zheng; Xiaolin Hu
>
> **备注:** Technical Report
>
> **摘要:** Query-based universal sound separation is fundamental to intelligent auditory systems, aiming to isolate specific sources from mixtures. Despite recent advances, existing methods continue to suffer from residual interference in complex acoustic scenes. This performance limitation stems largely from a data bottleneck: in-the-wild datasets contain weak labels and severe co-occurrence of events. These flaws induce models to learn spurious correlations between background noise and target categories instead of robust acoustic features. To address this, we propose an automated pipeline that eliminates co-occurrence of events by mining high-purity single-event segments from in-the-wild datasets via a semantically consistent synthesis protocol. Utilizing this pipeline, we constructed Hive, a high-quality synthetic dataset comprising 2.4k hours of raw audio. Experimental results demonstrate that, compared with the state-of-the-art model SAM-Audio which was trained on a huge dataset $\sim$500 times larger than Hive, certain open-source models trained on Hive achieve competitive separation accuracy and perceptual quality. Moreover, these models exhibited remarkable zero-shot generalization on out-of-distribution evaluation benchmarks. These findings highlight that prioritizing purity of supervised signals enables significant data efficiency, offering a new paradigm for training robust auditory foundation models with reduced computational costs. Code and dataset are available at https://shandaai.github.io/Hive.
>
---
#### [new 003] Towards Explicit Acoustic Evidence Perception in Audio LLMs for Speech Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有方法过度依赖语义线索而忽略细微声学特征的问题。通过引入增强听觉感知的音频大模型框架，提升对声学证据的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2601.23066v1](https://arxiv.org/pdf/2601.23066v1)**

> **作者:** Xiaoxuan Guo; Yuankun Xie; Haonan Cheng; Jiayi Zhou; Jian Liu; Hengyan Huang; Long Ye; Qin Zhang
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Speech deepfake detection (SDD) focuses on identifying whether a given speech signal is genuine or has been synthetically generated. Existing audio large language model (LLM)-based methods excel in content understanding; however, their predictions are often biased toward semantically correlated cues, which results in fine-grained acoustic artifacts being overlooked during the decisionmaking process. Consequently, fake speech with natural semantics can bypass detectors despite harboring subtle acoustic anomalies; this suggests that the challenge stems not from the absence of acoustic data, but from its inadequate accessibility when semantic-dominant reasoning prevails. To address this issue, we investigate SDD within the audio LLM paradigm and introduce SDD with Auditory Perception-enhanced Audio Large Language Model (SDD-APALLM), an acoustically enhanced framework designed to explicitly expose fine-grained time-frequency evidence as accessible acoustic cues. By combining raw audio with structured spectrograms, the proposed framework empowers audio LLMs to more effectively capture subtle acoustic inconsistencies without compromising their semantic understanding. Experimental results indicate consistent gains in detection accuracy and robustness, especially in cases where semantic cues are misleading. Further analysis reveals that these improvements stem from a coordinated utilization of semantic and acoustic information, as opposed to simple modality aggregation.
>
---
#### [new 004] Hearing is Believing? Evaluating and Analyzing Audio Language Model Sycophancy with SYAUDIO
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型研究，旨在解决ALMs中的盲从问题。通过构建SYAUDIO基准，评估和分析音频条件下的盲从行为，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2601.23149v1](https://arxiv.org/pdf/2601.23149v1)**

> **作者:** Junchi Yao; Lokranjan Lakshmikanthan; Annie Zhao; Danielle Zhao; Shu Yang; Zikang Ding; Di Wang; Lijie Hu
>
> **摘要:** Audio Language Models (ALMs) have recently shown strong capabilities in unified reasoning over speech, sound, and natural language; yet they inherit behavioral issues observed in Large Language Models, including sycophancy--the tendency to agree with user assertions even when they contradict objective evidence. While sycophancy has been extensively studied in text and vision-language models, its manifestation in audio-conditioned reasoning remains largely unexplored, despite the need for ALMs to rely on auditory cues such as acoustic events, speaker characteristics, and speech rate. To address this gap, we introduce SYAUDIO, the first benchmark dedicated to evaluating sycophancy in ALMs, consisting of 4,319 audio questions spanning Audio Perception, Audio Reasoning, Audio Math, and Audio Ethics. Built upon established audio benchmarks and augmented with TTS-generated arithmetic and moral reasoning tasks, SYAUDIO enables systematic evaluation across multiple domains and sycophancy types with carefully verified data quality. Furthermore, we analyze audio-specific sycophancy under realistic conditions involving noise and rate, and demonstrate that supervised fine-tuning with chain-of-thought data is an effective mitigation strategy for reducing sycophantic behavior in ALMs.
>
---
#### [new 005] Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability
- **分类: cs.SD**

- **简介: 该论文属于角色扮演语音合成任务，解决风格一致性不足的问题。提出MCLP作为评估和奖励机制，提升生成语音与角色指令的匹配度。**

- **链接: [https://arxiv.org/pdf/2601.22661v1](https://arxiv.org/pdf/2601.22661v1)**

> **作者:** Yong Ren; Jingbei Li; Haiyang Sun; Yujie Chen; Cheng Yi; Yechang Huang; Hao Gu; Ye Bai; Xuerui Yang
>
> **摘要:** Recent advances in Large Audio Language Models (LALMs) have extended Text-to-Speech (TTS) to interactive role-play scenarios, which demand high expressiveness and strict adherence to role-play instructions. However, existing models struggle to maintain stylistic consistency with character profiles and scene descriptions across multi-turn dialogues. A critical bottleneck is the lack of objective metrics for quantifying speaking style. To bridge this gap, we propose Mean Continuation Log-Probability (MCLP) as both an evaluation metric and a reward signal, validated on LALM-based Role-Play TTS (RP-TTS) tasks. Critically, we leverage the In-Context Learning capability of pre-trained LALMs to formulate MCLP via a continuation log-probability prediction. This metric quantifies stylistic consistency by measuring the likelihood of the ground-truth speech conditioned on the generated speech. Furthermore, we employ MCLP as a reinforcement learning reward to enhance the style alignment between generated speech and Role-Play instructions. To facilitate evaluation, we construct an RP-TTS dataset with rich scene and character annotations. Experimental results demonstrate that our method significantly outperforms strong LALM baselines on both objective and subjective metrics.
>
---
#### [new 006] Beyond Omnidirectional: Neural Ambisonics Encoding for Arbitrary Microphone Directivity Patterns using Cross-Attention
- **分类: eess.AS**

- **简介: 该论文属于空间音频编码任务，解决任意麦克风阵列配置下的Ambisonics编码问题。通过引入方向响应和交叉注意力机制，提升编码准确性。**

- **链接: [https://arxiv.org/pdf/2601.23196v1](https://arxiv.org/pdf/2601.23196v1)**

> **作者:** Mikko Heikkinen; Archontis Politis; Konstantinos Drossos; Tuomas Virtanen
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** We present a deep neural network approach for encoding microphone array signals into Ambisonics that generalizes to arbitrary microphone array configurations with fixed microphone count but varying locations and frequency-dependent directional characteristics. Unlike previous methods that rely only on array geometry as metadata, our approach uses directional array transfer functions, enabling accurate characterization of real-world arrays. The proposed architecture employs separate encoders for audio and directional responses, combining them through cross-attention mechanisms to generate array-independent spatial audio representations. We evaluate the method on simulated data in two settings: a mobile phone with complex body scattering, and a free-field condition, both with varying numbers of sound sources in reverberant environments. Evaluations demonstrate that our approach outperforms both conventional digital signal processing-based methods and existing deep neural network solutions. Furthermore, using array transfer functions instead of geometry as metadata input improves accuracy on realistic arrays.
>
---
#### [new 007] Rethinking Speech Representation Aggregation in Speech Enhancement: A Phonetic Mutual Information Perspective
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决SSL表示在噪声下语义信息受损的问题。通过引入基于音素互信息的语义聚合层，提升语音识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.22480v1](https://arxiv.org/pdf/2601.22480v1)**

> **作者:** Seungu Han; Sungho Lee; Kyogu Lee
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Recent speech enhancement (SE) models increasingly leverage self-supervised learning (SSL) representations for their rich semantic information. Typically, intermediate features are aggregated into a single representation via a lightweight adaptation module. However, most SSL models are not trained for noise robustness, which can lead to corrupted semantic representations. Moreover, the adaptation module is trained jointly with the SE model, potentially prioritizing acoustic details over semantic information, contradicting the original purpose. To address this issue, we first analyze the behavior of SSL models on noisy speech from an information-theoretic perspective. Specifically, we measure the mutual information (MI) between the corrupted SSL representations and the corresponding phoneme labels, focusing on preservation of linguistic contents. Building upon this analysis, we introduce the linguistic aggregation layer, which is pre-trained to maximize MI with phoneme labels (with optional dynamic aggregation) and then frozen during SE training. Experiments show that this decoupled approach improves Word Error Rate (WER) over jointly optimized baselines, demonstrating the benefit of explicitly aligning the adaptation module with linguistic contents.
>
---
#### [new 008] CALM: Joint Contextual Acoustic-Linguistic Modeling for Personalization of Multi-Speaker ASR
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出CALM框架，用于多说话人ASR的个性化任务，解决语音识别中的说话人干扰问题，通过联合声学与语言建模提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.22792v1](https://arxiv.org/pdf/2601.22792v1)**

> **作者:** Muhammad Shakeel; Yosuke Fukumoto; Chikara Maeda; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** We present CALM, a joint Contextual Acoustic-Linguistic Modeling framework for multi-speaker automatic speech recognition (ASR). In personalized AI scenarios, the joint availability of acoustic and linguistic cues naturally motivates the integration of target-speaker conditioning with contextual biasing in overlapping conversations. CALM implements this integration in an end-to-end framework through speaker embedding-driven target-speaker extraction and dynamic vocabulary-based contextual biasing. We evaluate CALM on simulated English (LibriSpeechMix) and Japanese (Corpus of Spontaneous Japanese mixtures, CSJMix). On two-speaker mixtures, CALM reduces biased word error rate (B-WER) from 12.7 to 4.7 on LibriSpeech2Mix and biased character error rate (B-CER) from 16.6 to 8.4 on CSJMix2 (eval3), demonstrating the effectiveness of joint acoustic-linguistic modeling across languages. We additionally report results on the AMI corpus (IHM-mix condition) to validate performance on standardized speech mixtures.
>
---
#### [new 009] An Effective Energy Mask-based Adversarial Evasion Attacks against Misclassification in Speaker Recognition Systems
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于语音识别领域的对抗攻击任务，旨在提升对说话人识别系统的逃避攻击效果。通过能量掩码扰动方法，减少感知失真并提高攻击有效性。**

- **链接: [https://arxiv.org/pdf/2601.22390v1](https://arxiv.org/pdf/2601.22390v1)**

> **作者:** Chanwoo Park; Chanwoo Kim
>
> **摘要:** Evasion attacks pose significant threats to AI systems, exploiting vulnerabilities in machine learning models to bypass detection mechanisms. The widespread use of voice data, including deepfakes, in promising future industries is currently hindered by insufficient legal frameworks. Adversarial attack methods have emerged as the most effective countermeasure against the indiscriminate use of such data. This research introduces masked energy perturbation (MEP), a novel approach using power spectrum for energy masking of original voice data. MEP applies masking to small energy regions in the frequency domain before generating adversarial perturbations, targeting areas less noticeable to the human auditory model. The study primarily employs advanced speaker recognition models, including ECAPA-TDNN and ResNet34, which have shown remarkable performance in speaker verification tasks. The proposed MEP method demonstrated strong performance in both audio quality and evasion effectiveness. The energy masking approach effectively minimizes the perceptual evaluation of speech quality (PESQ) degradation, indicating that minimal perceptual distortion occurs to the human listener despite the adversarial perturbations. Specifically, in the PESQ evaluation, the relative performance of the MEP method was 26.68% when compared to the fast gradient sign method (FGSM) and iterative FGSM.
>
---
#### [new 010] Layer-Aware Early Fusion of Acoustic and Linguistic Embeddings for Cognitive Status Classification
- **分类: eess.AS**

- **简介: 该论文属于认知状态分类任务，旨在通过融合语音和文本嵌入提升诊断效果。研究提出层感知的早期融合方法，探索不同编码器层对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2601.23004v1](https://arxiv.org/pdf/2601.23004v1)**

> **作者:** Krystof Novotny; Laureano Moro-Velázquez; Jiri Mekyska
>
> **备注:** 5 pages, 3 figures, paper accepted for ICASSP 2026 conference
>
> **摘要:** Speech contains both acoustic and linguistic patterns that reflect cognitive decline, and therefore models describing only one domain cannot fully capture such complexity. This study investigates how early fusion (EF) of speech and its corresponding transcription text embeddings, with attention to encoder layer depth, can improve cognitive status classification. Using a DementiaBank-derived collection of recordings (1,629 speakers; cognitively normal controls$\unicode{x2013}$CN, Mild Cognitive Impairment$\unicode{x2013}$MCI, and Alzheimer's Disease and Related Dementias$\unicode{x2013}$ADRD), we extracted frame-aligned embeddings from different internal layers of wav2vec 2.0 or Whisper combined with DistilBERT or RoBERTa. Unimodal, EF and late fusion (LF) models were trained with a transformer classifier, optimized, and then evaluated across 10 seeds. Performance consistently peaked in mid encoder layers ($\sim$8$\unicode{x2013}$10), with the single best F1 at Whisper + RoBERTa layer 9 and the best log loss at Whisper + DistilBERT layer 10. Acoustic-only models consistently outperformed text-only variants. EF boosts discrimination for genuinely acoustic embeddings, whereas LF improves probability calibration. Layer choice critically shapes clinical multimodal synergy.
>
---
#### [new 011] Brain-Informed Speech Separation for Cochlear Implants
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音分离任务，旨在解决耳蜗植入物中区分目标说话人的问题。通过融合脑电图注意力线索与音频信号，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2601.22260v1](https://arxiv.org/pdf/2601.22260v1)**

> **作者:** Tom Gajecki; Jonas Althoff; Waldo Nogueira
>
> **摘要:** We propose a brain-informed speech separation method for cochlear implants (CIs) that uses electroencephalography (EEG)-derived attention cues to guide enhancement toward the attended speaker. An attention-guided network fuses audio mixtures with EEG features through a lightweight fusion layer, producing attended-source electrodograms for CI stimulation while resolving the label-permutation ambiguity of audio-only separators. Robustness to degraded attention cues is improved with a mixed curriculum that varies cue quality during training, yielding stable gains even when EEG-speech correlation is moderate. In multi-talker conditions, the model achieves higher signal-to-interference ratio improvements than an audio-only electrodogram baseline while remaining slightly smaller (167k vs. 171k parameters). With 2 ms algorithmic latency and comparable cost, the approach highlights the promise of coupling auditory and neural cues for cognitively adaptive CI processing.
>
---
#### [new 012] How Far Can Pretrained LLMs Go in Symbolic Music? Controlled Comparisons of Supervised and Preference-based Adaptation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于符号音乐理解与生成任务，探讨预训练大语言模型在音乐领域的适应效果。研究比较了不同微调策略，分析了领域适应与保留先验信息的权衡。**

- **链接: [https://arxiv.org/pdf/2601.22764v1](https://arxiv.org/pdf/2601.22764v1)**

> **作者:** Deepak Kumar; Emmanouil Karystinaios; Gerhard Widmer; Markus Schedl
>
> **备注:** Accepted at NLP4MusA 2026
>
> **摘要:** Music often shares notable parallels with language, motivating the use of pretrained large language models (LLMs) for symbolic music understanding and generation. Despite growing interest, the practical effectiveness of adapting instruction-tuned LLMs to symbolic music remains insufficiently characterized. We present a controlled comparative study of finetuning strategies for ABC-based generation and understanding, comparing an off-the-shelf instruction-tuned backbone to domain-adapted variants and a music-specialized LLM baseline. Across multiple symbolic music corpora and evaluation signals, we provide some insights into adaptation choices for symbolic music applications. We highlight the domain adaptation vs.~preserving prior information tradeoff as well as the distinct behaviour of metrics used to measure the domain adaptation for symbolic music.
>
---
#### [new 013] EmoShift: Lightweight Activation Steering for Enhanced Emotion-Aware Speech Synthesis
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于情感语音合成任务，旨在提升情感表达的精确性和可控性。针对现有系统依赖固定情感嵌入的问题，提出EmoShift框架，通过EmoSteer层学习情感偏移向量，增强情感表现力并保持自然度。**

- **链接: [https://arxiv.org/pdf/2601.22873v1](https://arxiv.org/pdf/2601.22873v1)**

> **作者:** Li Zhou; Hao Jiang; Junjie Li; Tianrui Wang; Haizhou Li
>
> **备注:** Activation Steering; Emotion-Aware TTS; Speech Synthesis; Accepted by ICASSP 2026
>
> **摘要:** Achieving precise and controllable emotional expression is crucial for producing natural and context-appropriate speech in text-to-speech (TTS) synthesis. However, many emotion-aware TTS systems, including large language model (LLM)-based designs, rely on scaling fixed emotion embeddings or external guidance, limiting their ability to model emotion-specific latent characteristics. To address this gap, we present EmoShift, a lightweight activation-steering framework incorporating a EmoSteer layer, which learns a steering vector for each target emotion in the output embedding space to capture its latent offset and maintain stable, appropriate expression across utterances and categories. With only 10M trainable parameters,less than 1/30 of full fine-tuning, EmoShift outperforms zero-shot and fully fine-tuned baselines in objective and subjective evaluations, enhancing emotional expressiveness while preserving naturalness and speaker similarity. Further analysis confirms the proposed EmoSteer layer's effectiveness and reveals its potential for controllable emotional intensity in speech synthesis.
>
---
#### [new 014] Optimizing Domain-Adaptive Self-Supervised Learning for Clinical Voice-Based Disease Classification
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于临床语音疾病分类任务，旨在解决数据稀缺和领域不匹配问题。通过优化自监督学习方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.22319v1](https://arxiv.org/pdf/2601.22319v1)**

> **作者:** Weixin Liu; Bowen Qu; Matthew Pontell; Maria Powell; Bradley Malin; Zhijun Yin
>
> **备注:** Accepted at IEEE ICASSP 2026
>
> **摘要:** The human voice is a promising non-invasive digital biomarker, yet deep learning for voice-based health analysis is hindered by data scarcity and domain mismatch, where models pre-trained on general audio fail to capture the subtle pathological features characteristic of clinical voice data. To address these challenges, we investigate domain-adaptive self-supervised learning (SSL) with Masked Autoencoders (MAE) and demonstrate that standard configurations are suboptimal for health-related audio. Using the Bridge2AI-Voice dataset, a multi-institutional collection of pathological voices, we systematically examine three performance-critical factors: reconstruction loss (Mean Absolute Error vs. Mean Squared Error), normalization (patch-wise vs. global), and masking (random vs. content-aware). Our optimized design, which combines Mean Absolute Error (MA-Error) loss, patch-wise normalization, and content-aware masking, achieves a Macro F1 of $0.688 \pm 0.009$ (over 10 fine-tuning runs), outperforming a strong out-of-domain SSL baseline pre-trained on large-scale general audio, which has a Macro F1 of $0.663 \pm 0.011$. The results show that MA-Error loss improves robustness and content-aware masking boosts performance by emphasizing information-rich regions. These findings highlight the importance of component-level optimization in data-constrained medical applications that rely on audio data.
>
---
#### [new 015] DIFFA-2: A Practical Diffusion Large Language Model for General Audio Understanding
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频理解任务，旨在解决AR模型训练成本高、推理效率低的问题。提出DIFFA-2，通过扩散模型提升音频理解效果，并优化训练流程。**

- **链接: [https://arxiv.org/pdf/2601.23161v1](https://arxiv.org/pdf/2601.23161v1)**

> **作者:** Jiaming Zhou; Xuxin Cheng; Shiwan Zhao; Yuhang Jia; Cao Liu; Ke Zeng; Xunliang Cai; Yong Qin
>
> **摘要:** Autoregressive (AR) large audio language models (LALMs) such as Qwen-2.5-Omni have achieved strong performance on audio understanding and interaction, but scaling them remains costly in data and computation, and strictly sequential decoding limits inference efficiency. Diffusion large language models (dLLMs) have recently been shown to make effective use of limited training data, and prior work on DIFFA indicates that replacing an AR backbone with a diffusion counterpart can substantially improve audio understanding under matched settings, albeit at a proof-of-concept scale without large-scale instruction tuning, preference alignment, or practical decoding schemes. We introduce DIFFA-2, a practical diffusion-based LALM for general audio understanding. DIFFA-2 upgrades the speech encoder, employs dual semantic and acoustic adapters, and is trained with a four-stage curriculum that combines semantic and acoustic alignment, large-scale supervised fine-tuning, and variance-reduced preference optimization, using only fully open-source corpora. Experiments on MMSU, MMAU, and MMAR show that DIFFA-2 consistently improves over DIFFA and is competitive to strong AR LALMs under practical training budgets, supporting diffusion-based modeling is a viable backbone for large-scale audio understanding. Our code is available at https://github.com/NKU-HLT/DIFFA.git.
>
---
#### [new 016] Sylber 2.0: A Universal Syllable Embedding
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出Sylber 2.0，解决跨语言语音建模中token效率与通用性不足的问题，通过 syllable 级别编码实现高效压缩与高保真重建。**

- **链接: [https://arxiv.org/pdf/2601.22306v1](https://arxiv.org/pdf/2601.22306v1)**

> **作者:** Cheol Jun Cho; Nicholas Lee; Alan W Black; Gopala K. Anumanchipalli
>
> **摘要:** Scaling spoken language modeling requires speech tokens that are both efficient and universal. Recent work has proposed syllables as promising speech tokens at low temporal resolution, but existing models are constrained to English and fail to capture sufficient acoustic detail. To address this gap, we present Sylber 2.0, a self-supervised framework for coding speech at the syllable level that enables efficient temporal compression and high-fidelity reconstruction. Sylber 2.0 achieves a very low token frequency around 5 Hz, while retaining both linguistic and acoustic detail across multiple languages and expressive styles. Experiments show that it performs on par with previous models operating on high-frequency baselines. Furthermore, Sylber 2.0 enables efficient TTS modeling which can generate speech with competitive intelligibility and quality with SOTA models using only 72M parameters. Moreover, the universality of Sylber 2.0 provides more effective features for low resource ASR than previous speech coding frameworks. In sum, we establish an effective syllable-level abstraction for general spoken language.
>
---
#### [new 017] Class-Aware Permutation-Invariant Signal-to-Distortion Ratio for Semantic Segmentation of Sound Scene with Same-Class Sources
- **分类: eess.AS**

- **简介: 该论文针对S5任务，解决同类别声源导致的标签重复问题，提出类感知的排列不变损失函数和改进的评估指标。**

- **链接: [https://arxiv.org/pdf/2601.22504v1](https://arxiv.org/pdf/2601.22504v1)**

> **作者:** Binh Thien Nguyen; Masahiro Yasuda; Daiki Takeuchi; Daisuke Niizumi; Noboru Harada
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** To advance immersive communication, the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge recently introduced Task 4 on Spatial Semantic Segmentation of Sound Scenes (S5). An S5 system takes a multi-channel audio mixture as input and outputs single-channel dry sources along with their corresponding class labels. Although the DCASE 2025 Challenge simplifies the task by constraining class labels in each mixture to be mutually exclusive, real-world mixtures frequently contain multiple sources from the same class. The presence of duplicated labels can significantly degrade the performance of the label-queried source separation (LQSS) model, which is the key component of many existing S5 systems, and can also limit the validity of the official evaluation metric of DCASE 2025 Task 4. To address these issues, we propose a class-aware permutation-invariant loss function that enables the LQSS model to handle queries involving duplicated labels. In addition, we redesign the S5 evaluation metric to eliminate ambiguities caused by these same-class sources. To evaluate the proposed method within the S5 system, we extend the label prediction model to support same-class labels. Experimental results demonstrate the effectiveness of the proposed methods and the robustness of the new metric on mixtures both with and without same-class sources.
>
---
#### [new 018] Attention Isn't All You Need for Emotion Recognition:Domain Features Outperform Transformers on the EAV Dataset
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于情感识别任务，研究在小数据集上复杂注意力机制的效果。通过对比不同模型，发现领域特征优于复杂架构，验证了领域知识的重要性。**

- **链接: [https://arxiv.org/pdf/2601.22161v1](https://arxiv.org/pdf/2601.22161v1)**

> **作者:** Anmol Guragain
>
> **摘要:** We present a systematic study of multimodal emotion recognition using the EAV dataset, investigating whether complex attention mechanisms improve performance on small datasets. We implement three model categories: baseline transformers (M1), novel factorized attention mechanisms (M2), and improved CNN baselines (M3). Our experiments show that sophisticated attention mechanisms consistently underperform on small datasets. M2 models achieved 5 to 13 percentage points below baselines due to overfitting and destruction of pretrained features. In contrast, simple domain-appropriate modifications proved effective: adding delta MFCCs to the audio CNN improved accuracy from 61.9\% to \textbf{65.56\%} (+3.66pp), while frequency-domain features for EEG achieved \textbf{67.62\%} (+7.62pp over the paper baseline). Our vision transformer baseline (M1) reached \textbf{75.30\%}, exceeding the paper's ViViT result (74.5\%) through domain-specific pretraining, and vision delta features achieved \textbf{72.68\%} (+1.28pp over the paper CNN). These findings demonstrate that for small-scale emotion recognition, domain knowledge and proper implementation outperform architectural complexity.
>
---
#### [new 019] MIRRORTALK: Forging Personalized Avatars Via Disentangled Style and Hierarchical Motion Control
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于人脸生成任务，旨在解决个性化说话脸合成中风格与语义混淆的问题。提出MirrorTalk框架，通过分离风格与运动控制，提升唇形同步和个性保留。**

- **链接: [https://arxiv.org/pdf/2601.22501v1](https://arxiv.org/pdf/2601.22501v1)**

> **作者:** Renjie Lu; Xulong Zhang; Xiaoyang Qu; Jianzong Wang; Shangfei Wang
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** Synthesizing personalized talking faces that uphold and highlight a speaker's unique style while maintaining lip-sync accuracy remains a significant challenge. A primary limitation of existing approaches is the intrinsic confounding of speaker-specific talking style and semantic content within facial motions, which prevents the faithful transfer of a speaker's unique persona to arbitrary speech. In this paper, we propose MirrorTalk, a generative framework based on a conditional diffusion model, combined with a Semantically-Disentangled Style Encoder (SDSE) that can distill pure style representations from a brief reference video. To effectively utilize this representation, we further introduce a hierarchical modulation strategy within the diffusion process. This mechanism guides the synthesis by dynamically balancing the contributions of audio and style features across distinct facial regions, ensuring both precise lip-sync accuracy and expressive full-face dynamics. Extensive experiments demonstrate that MirrorTalk achieves significant improvements over state-of-the-art methods in terms of lip-sync accuracy and personalization preservation.
>
---
#### [new 020] Proliferating series by Jean Barraqué: a study and classification in mathematical terms
- **分类: math.HO; cs.SD; eess.AS**

- **简介: 该论文属于音乐理论研究，探讨Jean Barraqué的繁殖序列，解决传统序列主义的局限问题，通过数学方法分析其结构与可能性。**

- **链接: [https://arxiv.org/pdf/2601.22176v1](https://arxiv.org/pdf/2601.22176v1)**

> **作者:** Isabel Tardón; Pablo Martín-Santamaría
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Barraqué's proliferating series give an interesting turn on the concept of classic serialism by creating a new invariant when it comes to constructing the series: rather than the intervals between consecutive notes, what remains unaltered during the construction of the proliferations of the given base series is the permutation of the notes which happens between two consecutive series, that is to say, the transformation of the order of the notes in the series. This presents new possibilities for composers interested in the serial method, given the fact that the variety of intervals obtained by this method is far greater than that of classic serialism. In this manuscript, we will study some unexplored possibilities that the proliferating series offer from a mathematical point of view, which will allow composers to gain much more familiarity with them and potentially result in the creation of pieces that take serialism to the next level.
>
---
#### [new 021] DiffuSpeech: Silent Thought, Spoken Answer via Unified Speech-Text Diffusion
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出DiffuSpeech，解决语音问答中缺乏推理过程的问题，通过统一的扩散模型生成文本推理和语音回答，提升语音质量和理解能力。**

- **链接: [https://arxiv.org/pdf/2601.22889v1](https://arxiv.org/pdf/2601.22889v1)**

> **作者:** Yuxuan Lou; Ziming Wu; Yaochen Wang; Yong Liu; Yingxuan Ren; Fuming Lai; Shaobing Lian; Jie Tang; Yang You
>
> **摘要:** Current speech language models generate responses directly without explicit reasoning, leading to errors that cannot be corrected once audio is produced. We introduce \textbf{``Silent Thought, Spoken Answer''} -- a paradigm where speech LLMs generate internal text reasoning alongside spoken responses, with thinking traces informing speech quality. To realize this, we present \method{}, the first diffusion-based speech-text language model supporting both understanding and generation, unifying discrete text and tokenized speech under a single masked diffusion framework. Unlike autoregressive approaches, \method{} jointly generates reasoning traces and speech tokens through iterative denoising, with modality-specific masking schedules. We also construct \dataset{}, the first speech QA dataset with paired text reasoning traces, containing 26K samples totaling 319 hours. Experiments show \method{} achieves state-of-the-art speech-to-speech QA accuracy, outperforming the best baseline by up to 9 points, while attaining the best TTS quality among generative models (6.2\% WER) and preserving language understanding (66.2\% MMLU). Ablations confirm that both the diffusion architecture and thinking traces contribute to these gains.
>
---
#### [new 022] Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决固定帧率导致序列过长的问题。提出DyCAST，通过动态帧率和字符对齐实现更高效的语音分词。**

- **链接: [https://arxiv.org/pdf/2601.23174v1](https://arxiv.org/pdf/2601.23174v1)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Neural audio codecs are at the core of modern conversational speech technologies, converting continuous speech into sequences of discrete tokens that can be processed by LLMs. However, existing codecs typically operate at fixed frame rates, allocating tokens uniformly in time and producing unnecessarily long sequences. In this work, we introduce DyCAST, a Dynamic Character-Aligned Speech Tokenizer that enables variable-frame-rate tokenization through soft character-level alignment and explicit duration modeling. DyCAST learns to associate tokens with character-level linguistic units during training and supports alignment-free inference with direct control over token durations at decoding time. To improve speech resynthesis quality at low frame rates, we further introduce a retrieval-augmented decoding mechanism that enhances reconstruction fidelity without increasing bitrate. Experiments show that DyCAST achieves competitive speech resynthesis quality and downstream performance while using significantly fewer tokens than fixed-frame-rate codecs.
>
---
#### [new 023] PersonaCite: VoC-Grounded Interviewable Agentic Synthetic AI Personas for Verifiable User and Design Research
- **分类: cs.HC; cs.AI; eess.AS; eess.IV**

- **简介: 该论文提出PersonaCite，解决合成人格在设计研究中证据不足的问题，通过检索增强交互确保回应可验证。属于AI可信性研究任务。**

- **链接: [https://arxiv.org/pdf/2601.22288v1](https://arxiv.org/pdf/2601.22288v1)**

> **作者:** Mario Truss
>
> **摘要:** LLM-based and agent-based synthetic personas are increasingly used in design and product decision-making, yet prior work shows that prompt-based personas often produce persuasive but unverifiable responses that obscure their evidentiary basis. We present PersonaCite, an agentic system that reframes AI personas as evidence-bounded research instruments through retrieval-augmented interaction. Unlike prior approaches that rely on prompt-based roleplaying, PersonaCite retrieves actual voice-of-customer artifacts during each conversation turn, constrains responses to retrieved evidence, explicitly abstains when evidence is missing, and provides response-level source attribution. Through semi-structured interviews and deployment study with 14 industry experts, we identify preliminary findings on perceived benefits, validity concerns, and design tensions, and propose Persona Provenance Cards as a documentation pattern for responsible AI persona use in human-centered design workflows.
>
---
#### [new 024] Compact Hypercube Embeddings for Fast Text-based Wildlife Observation Retrieval
- **分类: cs.IR; cs.CV; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于文本驱动的野生动物观测检索任务，旨在解决大规模数据库中高效检索的问题。通过构建紧凑的二进制超立方体嵌入，实现快速搜索，提升检索效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.22783v1](https://arxiv.org/pdf/2601.22783v1)**

> **作者:** Ilyass Moummad; Marius Miron; David Robinson; Kawtar Zaher; Hervé Goëau; Olivier Pietquin; Pierre Bonnet; Emmanuel Chemla; Matthieu Geist; Alexis Joly
>
> **摘要:** Large-scale biodiversity monitoring platforms increasingly rely on multimodal wildlife observations. While recent foundation models enable rich semantic representations across vision, audio, and language, retrieving relevant observations from massive archives remains challenging due to the computational cost of high-dimensional similarity search. In this work, we introduce compact hypercube embeddings for fast text-based wildlife observation retrieval, a framework that enables efficient text-based search over large-scale wildlife image and audio databases using compact binary representations. Building on the cross-view code alignment hashing framework, we extend lightweight hashing beyond a single-modality setup to align natural language descriptions with visual or acoustic observations in a shared Hamming space. Our approach leverages pretrained wildlife foundation models, including BioCLIP and BioLingual, and adapts them efficiently for hashing using parameter-efficient fine-tuning. We evaluate our method on large-scale benchmarks, including iNaturalist2024 for text-to-image retrieval and iNatSounds2024 for text-to-audio retrieval, as well as multiple soundscape datasets to assess robustness under domain shift. Results show that retrieval using discrete hypercube embeddings achieves competitive, and in several cases superior, performance compared to continuous embeddings, while drastically reducing memory and search cost. Moreover, we observe that the hashing objective consistently improves the underlying encoder representations, leading to stronger retrieval and zero-shot generalization. These results demonstrate that binary, language-based retrieval enables scalable and efficient search over large wildlife archives for biodiversity monitoring systems.
>
---
## 更新

#### [replaced 001] LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音文本对齐任务，解决多语言和长语音中对齐精度低的问题。提出LLM-ForcedAligner，通过槽填充机制实现非自回归对齐，提升准确性和速度。**

- **链接: [https://arxiv.org/pdf/2601.18220v2](https://arxiv.org/pdf/2601.18220v2)**

> **作者:** Bingshen Mu; Xian Shi; Xiong Wang; Hexin Liu; Jin Xu; Lei Xie
>
> **摘要:** Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. Checkpoint and inference code are available at https://github.com/QwenLM/Qwen3-ASR.
>
---
#### [replaced 002] LIWhiz: A Non-Intrusive Lyric Intelligibility Prediction System for the Cadenza Challenge
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出LIWhiz系统，用于歌词可懂性预测任务，解决音频中歌词清晰度评估问题，通过特征提取与模型预测提升性能。**

- **链接: [https://arxiv.org/pdf/2512.17937v2](https://arxiv.org/pdf/2512.17937v2)**

> **作者:** Ram C. M. C. Shekar; Iván López-Espejo
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** We present LIWhiz, a non-intrusive lyric intelligibility prediction system submitted to the ICASSP 2026 Cadenza Challenge. LIWhiz leverages Whisper for robust feature extraction and a trainable back-end for score prediction. Tested on the Cadenza Lyric Intelligibility Prediction (CLIP) evaluation set, LIWhiz achieves a root mean square error (RMSE) of 27.07%, a 22.4% relative RMSE reduction over the STOI-based baseline, yielding a substantial improvement in normalized cross-correlation.
>
---
#### [replaced 003] Diffusion-based Frameworks for Unsupervised Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决无监督下的单通道语音增强问题。通过改进扩散模型，显式建模语音和噪声，提升增强效果。**

- **链接: [https://arxiv.org/pdf/2601.09931v3](https://arxiv.org/pdf/2601.09931v3)**

> **作者:** Jean-Eudes Ayilo; Mostafa Sadeghi; Romain Serizel; Xavier Alameda-Pineda
>
> **摘要:** This paper addresses unsupervised diffusion-based single-channel speech enhancement (SE). Prior work in this direction combines a score-based diffusion model trained on clean speech with a Gaussian noise model whose covariance is structured by non-negative matrix factorization (NMF). This combination is used within an iterative expectation-maximization (EM) scheme, in which a diffusion-based posterior-sampling E-step estimates the clean speech. We first revisit this framework and propose to explicitly model both speech and acoustic noise as latent variables, jointly sampling them in the E-step instead of sampling speech alone as in previous approaches. We then introduce a new unsupervised SE framework that replaces the NMF noise prior with a diffusion-based noise model, learned jointly with the speech prior in a single conditional score model. Within this framework, we derive two variants: one that implicitly accounts for noise and one that explicitly treats noise as a latent variable. Experiments on WSJ0-QUT and VoiceBank-DEMAND show that explicit noise modeling systematically improves SE performance for both NMF-based and diffusion-based noise priors. Under matched conditions, the diffusion-based noise model attains the best overall quality and intelligibility among unsupervised methods, while under mismatched conditions the proposed NMF-based explicit-noise framework is more robust and suffers less degradation than several supervised baselines.
>
---
#### [replaced 004] CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频反欺骗任务，解决组件级音频伪造检测问题。构建了CompSpoof数据集，提出联合学习框架，分别检测音频各组件的伪造情况。**

- **链接: [https://arxiv.org/pdf/2509.15804v2](https://arxiv.org/pdf/2509.15804v2)**

> **作者:** Xueping Zhang; Yechen Wang; Linxi Li; Liwei Jin; Ming Li
>
> **备注:** accepted at ICASSP 2026
>
> **摘要:** Component-level audio Spoofing (Comp-Spoof) targets a new form of audio manipulation where only specific components of a signal, such as speech or environmental sound, are forged or substituted while other components remain genuine. Existing anti-spoofing datasets and methods treat an utterance or a segment as entirely bona fide or entirely spoofed, and thus cannot accurately detect component-level spoofing. To address this, we construct a new dataset, CompSpoof, covering multiple combinations of bona fide and spoofed speech and environmental sound. We further propose a separation-enhanced joint learning framework that separates audio components apart and applies anti-spoofing models to each one. Joint learning is employed, preserving information relevant for detection. Extensive experiments demonstrate that our method outperforms the baseline, highlighting the necessity of separate components and the importance of detecting spoofing for each component separately. Datasets and code are available at: https://github.com/XuepingZhang/CompSpoof.
>
---
#### [replaced 005] Speech Emotion Recognition with ASR Integration
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决真实场景下情感识别的挑战。通过集成自动语音识别技术，提升情感识别的鲁棒性和实用性。**

- **链接: [https://arxiv.org/pdf/2601.17901v2](https://arxiv.org/pdf/2601.17901v2)**

> **作者:** Yuanchao Li
>
> **备注:** PhD Thesis
>
> **摘要:** Speech Emotion Recognition (SER) plays a pivotal role in understanding human communication, enabling emotionally intelligent systems, and serving as a fundamental component in the development of Artificial General Intelligence (AGI). However, deploying SER in real-world, spontaneous, and low-resource scenarios remains a significant challenge due to the complexity of emotional expression and the limitations of current speech and language technologies. This thesis investigates the integration of Automatic Speech Recognition (ASR) into SER, with the goal of enhancing the robustness, scalability, and practical applicability of emotion recognition from spoken language.
>
---
#### [replaced 006] Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究语音增强系统的安全问题，探讨其是否易受对抗攻击。工作包括验证现代模型的脆弱性，并指出扩散模型具有天然鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.21087v3](https://arxiv.org/pdf/2509.21087v3)**

> **作者:** Rostislav Makarov; Lea Schönherr; Timo Gerkmann
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.
>
---
#### [replaced 007] Location-Oriented Sound Event Localization and Detection with Spatial Mapping and Regression Localization
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声事件定位与检测任务，旨在解决多音环境下的定位准确性问题。提出SMRL-SELD方法，通过空间映射和回归损失提升模型性能。**

- **链接: [https://arxiv.org/pdf/2504.08365v3](https://arxiv.org/pdf/2504.08365v3)**

> **作者:** Xueping Zhang; Yaxiong Chen; Ruilin Yao; Yunfei Zi; Shengwu Xiong
>
> **备注:** accepted at ICME 2025
>
> **摘要:** Sound Event Localization and Detection (SELD) combines the Sound Event Detection (SED) with the corresponding Direction Of Arrival (DOA). Recently, adopted event oriented multi-track methods affect the generality in polyphonic environments due to the limitation of the number of tracks. To enhance the generality in polyphonic environments, we propose Spatial Mapping and Regression Localization for SELD (SMRL-SELD). SMRL-SELD segments the 3D spatial space, mapping it to a 2D plane, and a new regression localization loss is proposed to help the results converge toward the location of the corresponding event. SMRL-SELD is location-oriented, allowing the model to learn event features based on orientation. Thus, the method enables the model to process polyphonic sounds regardless of the number of overlapping events. We conducted experiments on STARSS23 and STARSS22 datasets and our proposed SMRL-SELD outperforms the existing SELD methods in overall evaluation and polyphony environments.
>
---
#### [replaced 008] BNMusic: Blending Environmental Noises into Personalized Music
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出BNMusic框架，将环境噪声融入个性化音乐中，解决噪声干扰问题。通过音乐生成与自适应增强，降低噪声可察觉性，提升听觉体验。属于音频处理任务。**

- **链接: [https://arxiv.org/pdf/2506.10754v3](https://arxiv.org/pdf/2506.10754v3)**

> **作者:** Chi Zuo; Martin B. Møller; Pablo Martínez-Nuevo; Huayang Huang; Yu Wu; Ye Zhu
>
> **备注:** This paper has been accepted by NeurIPS 2025
>
> **摘要:** While being disturbed by environmental noises, the acoustic masking technique is a conventional way to reduce the annoyance in audio engineering that seeks to cover up the noises with other dominant yet less intrusive sounds. However, misalignment between the dominant sound and the noise-such as mismatched downbeats-often requires an excessive volume increase to achieve effective masking. Motivated by recent advances in cross-modal generation, in this work, we introduce an alternative method to acoustic masking, aiming to reduce the noticeability of environmental noises by blending them into personalized music generated based on user-provided text prompts. Following the paradigm of music generation using mel-spectrogram representations, we propose a Blending Noises into Personalized Music (BNMusic) framework with two key stages. The first stage synthesizes a complete piece of music in a mel-spectrogram representation that encapsulates the musical essence of the noise. In the second stage, we adaptively amplify the generated music segment to further reduce noise perception and enhance the blending effectiveness, while preserving auditory quality. Our experiments with comprehensive evaluations on MusicBench, EPIC-SOUNDS, and ESC-50 demonstrate the effectiveness of our framework, highlighting the ability to blend environmental noise with rhythmically aligned, adaptively amplified, and enjoyable music segments, minimizing the noticeability of the noise, thereby improving overall acoustic experiences. Project page: https://d-fas.github.io/BNMusic_page/.
>
---
#### [replaced 009] TopSeg: A Multi-Scale Topological Framework for Data-Efficient Heart Sound Segmentation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于心音分割任务，旨在解决数据稀缺下的分割精度问题。提出TopSeg框架，利用多尺度拓扑特征提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.17346v2](https://arxiv.org/pdf/2510.17346v2)**

> **作者:** Peihong Zhang; Zhixin Li; Yuxuan Liu; Rui Sang; Yiqiang Cai; Yizhou Tan; Shengchen Li
>
> **备注:** Accepted at ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
> **摘要:** Deep learning approaches for heart-sound (PCG) segmentation built on time-frequency features can be accurate but often rely on large expert-labeled datasets, limiting robustness and deployment. We present TopSeg, a topological representation-centric framework that encodes PCG dynamics with multi-scale topological features and decodes them using a lightweight temporal convolutional network (TCN) with an order- and duration-constrained inference step. To evaluate data efficiency and generalization, we train exclusively on PhysioNet 2016 dataset with subject-level subsampling and perform external validation on CirCor dataset. Under matched-capacity decoders, the topological features consistently outperform spectrogram and envelope inputs, with the largest margins at low data budgets; as a full system, TopSeg surpasses representative end-to-end baselines trained on their native inputs under the same budgets while remaining competitive at full data. Ablations at 10% training confirm that all scales contribute and that combining H_0 and H_1 yields more reliable S1/S2 localization and boundary stability. These results indicate that topology-aware representations provide a strong inductive bias for data-efficient, cross-dataset PCG segmentation, supporting practical use when labeled data are limited.
>
---
#### [replaced 010] MAPSS: Manifold-based Assessment of Perceptual Source Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决客观评估与主观感知不匹配的问题。提出PS和PM两个指标，分别量化泄漏和自失真，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2509.09212v2](https://arxiv.org/pdf/2509.09212v2)**

> **作者:** Amir Ivry; Samuele Cornell; Shinji Watanabe
>
> **备注:** Preprint. Accepted to ICLR (not camera ready)
>
> **摘要:** Objective assessment of source-separation systems still mismatches subjective human perception, especially when leakage and self-distortion interact. We introduce the Perceptual Separation (PS) and Perceptual Match (PM), the first pair of measures that functionally isolate these two factors. Our intrusive method begins with generating a bank of fundamental distortions for each reference waveform signal in the mixture. Distortions, references, and their respective system outputs from all sources are then independently encoded by a pre-trained self-supervised learning model. These representations are aggregated and projected onto a manifold via diffusion maps, which aligns Euclidean distances on the manifold with dissimilarities of the encoded waveforms. On this manifold, the PM measures the Mahalanobis distance from each output to its attributed cluster that consists of its reference and distortions embeddings, capturing self-distortion. The PS accounts for the Mahalanobis distance of the output to the attributed and to the closest non-attributed clusters, quantifying leakage. Both measures are differentiable and granular, operating at a resolution as low as 50 frames per second. We further derive, for both measures, deterministic error radius and non-asymptotic, high-probability confidence intervals (CIs). Experiments on English, Spanish, and music mixtures show that the PS and PM nearly always achieve the highest linear correlation coefficients with human mean-opinion scores than 14 competitors, reaching as high as 86.36% for speech and 87.21% for music. We observe, at worst, an error radius of 1.39% and a probabilistic 95% CI of 12.21% for these coefficients, which improves reliable and informed evaluation. Using mutual information, the measures complement each other most as their values decrease, suggesting they are jointly more informative as system performance degrades.
>
---
#### [replaced 011] DDSC: Dynamic Dual-Signal Curriculum for Data-Efficient Acoustic Scene Classification under Domain Shift
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于声学场景分类任务，解决设备引起的领域偏移问题。提出动态双信号课程方法（DDSC），通过自适应调整训练样本权重提升跨设备性能。**

- **链接: [https://arxiv.org/pdf/2510.17345v2](https://arxiv.org/pdf/2510.17345v2)**

> **作者:** Peihong Zhang; Yuxuan Liu; Rui Sang; Zhixin Li; Yiqiang Cai; Yizhou Tan; Shengchen Li
>
> **备注:** Accepted at ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
> **摘要:** Acoustic scene classification (ASC) suffers from device-induced domain shift, especially when labels are limited. Prior work focuses on curriculum-based training schedules that structure data presentation by ordering or reweighting training examples from easy-to-hard to facilitate learning; however, existing curricula are static, fixing the ordering or the weights before training and ignoring that example difficulty and marginal utility evolve with the learned representation. To overcome this limitation, we propose the Dynamic Dual-Signal Curriculum (DDSC), a training schedule that adapts the curriculum online by combining two signals computed each epoch: a domain-invariance signal and a learning-progress signal. A time-varying scheduler fuses these signals into per-example weights that prioritize domain-invariant examples in early epochs and progressively emphasize device-specific cases. DDSC is lightweight, architecture-agnostic, and introduces no additional inference overhead. Under the official DCASE 2024 Task~1 protocol, DDSC consistently improves cross-device performance across diverse ASC baselines and label budgets, with the largest gains on unseen-device splits.
>
---
#### [replaced 012] FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决全双工对话模型中语义对齐问题。通过引入连续独白和双阶段训练策略，提升对话质量与效率。**

- **链接: [https://arxiv.org/pdf/2509.02521v3](https://arxiv.org/pdf/2509.02521v3)**

> **作者:** Yiqun Yao; Xiang Li; Xin Jiang; Xuezhi Fang; Naitong Yu; Wenjia Ma; Aixin Sun; Yequan Wang
>
> **摘要:** Full-duplex dialog models aim to listen and speak simultaneously, delivering rapid responses to dynamic user input. Among different solutions to full-duplexity, a native solution merges multiple channels in each time step, achieving the lowest latency. However, prevailing designs break down the textual monologue sentences for word-level alignment with audio streams, which degrades language modeling abilities. To help address this issue, we introduce "contiguous monologues", which are composed by continuous sentences and "waiting" intervals, mimicking human-like cognitive behavior in dialogs. We find a proper training paradigm to be critical for semantically aligning contiguous monologues with audio. To this end, we develop a "dual" training paradigm that alternates the position of the monologues, either leading or trailing the audio, across different training stages. A combination of our contiguous monologue and dual training strategy is applied in developing FLM-Audio, our 7B spoken dialog chatbot with native full-duplexity. As confirmed by experimental results, FLM-Audio achieves superior response qualities and chatting experiences while requiring significantly less training data.
>
---
#### [replaced 013] Text-only adaptation in LLM-based ASR through text denoising
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音识别领域，解决LLM-based ASR在新领域适应的问题。通过文本去噪方法实现文本-only适应，保持跨模态对齐，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.20900v2](https://arxiv.org/pdf/2601.20900v2)**

> **作者:** Sergio Burdisso; Esaú Villatoro-Tello; Andrés Carofilis; Shashi Kumar; Kadri Hacioglu; Srikanth Madikeri; Pradeep Rangappa; Manjunath K E; Petr Motlicek; Shankar Venkatesan; Andreas Stolcke
>
> **备注:** Paper accepted at ICASSP 2026
>
> **摘要:** Adapting automatic speech recognition (ASR) systems based on large language models (LLMs) to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on target-domain text often disrupts the critical alignment between speech and text modalities learned by the projector, degrading performance. We introduce a novel text-only adaptation method that emulates the audio projection task by treating it as a text denoising task. Our approach thus trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
>
---
#### [replaced 014] Impact of Phonetics on Speaker Identity in Adversarial Voice Attack
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **简介: 论文研究对抗语音攻击中发音对说话人身份的影响，属于语音安全任务。解决对抗样本如何影响语音识别和说话人验证的问题，通过分析发音级扰动，揭示其导致识别错误和身份漂移的机制。**

- **链接: [https://arxiv.org/pdf/2509.15437v2](https://arxiv.org/pdf/2509.15437v2)**

> **作者:** Daniyal Kabir Dar; Qiben Yan; Li Xiao; Arun Ross
>
> **备注:** Additional figures for extended visualization: https://daniyalkabir.github.io/icassp-2026-results/
>
> **摘要:** Adversarial perturbations in speech pose a serious threat to automatic speech recognition (ASR) and speaker verification by introducing subtle waveform modifications that remain imperceptible to humans but can significantly alter system outputs. While targeted attacks on end-to-end ASR models have been widely studied, the phonetic basis of these perturbations and their effect on speaker identity remain underexplored. In this work, we analyze adversarial audio at the phonetic level and show that perturbations exploit systematic confusions such as vowel centralization and consonant substitutions. These distortions not only mislead transcription but also degrade phonetic cues critical for speaker verification, leading to identity drift. Using DeepSpeech as our ASR target, we generate targeted adversarial examples and evaluate their impact on speaker embeddings across genuine and impostor samples. Results across 16 phonetically diverse target phrases demonstrate that adversarial audio induces both transcription errors and identity drift, highlighting the need for phonetic-aware defenses to ensure the robustness of ASR and speaker recognition systems.
>
---
#### [replaced 015] SynthCloner: Synthesizer-style Audio Transfer via Factorized Codec with ADSR Envelope Control
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SynthCloner，解决合成器音色迁移问题，通过分解音频为包络、音色和内容三个属性，实现独立控制。**

- **链接: [https://arxiv.org/pdf/2509.24286v2](https://arxiv.org/pdf/2509.24286v2)**

> **作者:** Jeng-Yue Liu; Ting-Chao Hsu; Yen-Tung Yeh; Li Su; Yi-Hsuan Yang
>
> **备注:** ICASSP 2026
>
> **摘要:** Electronic synthesizer sounds are controlled by parameter settings that yield complex timbral characteristics and ADSR envelopes, making synthesizer-style audio transfer particularly challenging. Recent approaches to timbre transfer often rely on spectral objectives or implicit style matching, offering limited control over envelope shaping. Moreover, public synthesizer datasets rarely provide diverse coverage of timbres and ADSR envelopes. To address these gaps, we present SynthCloner, a factorized codec model that disentangles audio into three attributes: ADSR envelope, timbre, and content. This separation enables expressive audio transfer with independent control over these attributes. Additionally, we introduce SynthCAT, a new synthesizer dataset with a task-specific rendering pipeline covering 250 timbres, 120 ADSR envelopes, and 100 MIDI sequences. Experiments show that SynthCloner outperforms baselines on both objective and subjective metrics, while enabling independent attribute control. The code, model checkpoint, and audio examples are available at https://buffett0323.github.io/synthcloner/.
>
---
#### [replaced 016] Qwen3-ASR Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升多语言ASR性能与效率。提出两个ASR模型和一个非自回归对齐模型，解决多语言识别及高效实时处理问题。**

- **链接: [https://arxiv.org/pdf/2601.21337v2](https://arxiv.org/pdf/2601.21337v2)**

> **作者:** Xian Shi; Xiong Wang; Zhifang Guo; Yongqi Wang; Pei Zhang; Xinyu Zhang; Zishan Guo; Hongkun Hao; Yu Xi; Baosong Yang; Jin Xu; Jingren Zhou; Junyang Lin
>
> **备注:** https://github.com/QwenLM/Qwen3-ASR
>
> **摘要:** In this report, we introduce Qwen3-ASR family, which includes two powerful all-in-one speech recognition models and a novel non-autoregressive speech forced alignment model. Qwen3-ASR-1.7B and Qwen3-ASR-0.6B are ASR models that support language identification and ASR for 52 languages and dialects. Both of them leverage large-scale speech training data and the strong audio understanding ability of their foundation model Qwen3-Omni. We conduct comprehensive internal evaluation besides the open-sourced benchmarks as ASR models might differ little on open-sourced benchmark scores but exhibit significant quality differences in real-world scenarios. The experiments reveal that the 1.7B version achieves SOTA performance among open-sourced ASR models and is competitive with the strongest proprietary APIs while the 0.6B version offers the best accuracy-efficiency trade-off. Qwen3-ASR-0.6B can achieve an average TTFT as low as 92ms and transcribe 2000 seconds speech in 1 second at a concurrency of 128. Qwen3-ForcedAligner-0.6B is an LLM based NAR timestamp predictor that is able to align text-speech pairs in 11 languages. Timestamp accuracy experiments show that the proposed model outperforms the three strongest force alignment models and takes more advantages in efficiency and versatility. To further accelerate the community research of ASR and audio understanding, we release these models under the Apache 2.0 license.
>
---
#### [replaced 017] Thinking in cocktail party: Chain-of-Thought and reinforcement learning for target speaker automatic speech recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于目标说话人自动语音识别（TS-ASR）任务，旨在从多人混杂的语音中识别指定说话人的语音。论文提出结合思维链和强化学习的方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.15612v2](https://arxiv.org/pdf/2509.15612v2)**

> **作者:** Yiru Zhang; Hang Su; Lichun Fan; Zhenbo Luo; Jian Luan
>
> **摘要:** Target Speaker Automatic Speech Recognition (TS-ASR) aims to transcribe the speech of a specified target speaker from multi-speaker mixtures in cocktail party scenarios. Recent advancement of Large Audio-Language Models (LALMs) has already brought some new insights to TS-ASR. However, significant room for optimization remains for the TS-ASR task within the LALMs architecture. While Chain of Thoughts (CoT) and Reinforcement Learning (RL) have proven effective in certain speech tasks, TS-ASR, which requires the model to deeply comprehend speech signals, differentiate various speakers, and handle overlapping utterances is particularly well-suited to a reasoning-guided approach. Therefore, we propose a novel framework that incorporates CoT and RL training into TS-ASR for performance improvement. A novel CoT dataset of TS-ASR is constructed, and the TS-ASR model is first trained on regular data and then fine-tuned on CoT data. Finally, the model is further trained with RL using selected data to enhance generalized reasoning capabilities. Experiment results show a significant improvement of TS-ASR performance with CoT and RL training, which demonstrates the effectiveness of the proposed CoT and RL training methods adapted for the TS-ASR task.
>
---
