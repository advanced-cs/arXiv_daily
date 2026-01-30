# 音频 cs.SD;  eess.AS

- **最新发布 25 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] TidyVoice 2026 Challenge Evaluation Plan
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决跨语言说话人验证中的性能下降问题。通过TidyVoice挑战，利用多语言数据集提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.21960v1](https://arxiv.org/pdf/2601.21960v1)**

> **作者:** Aref Farhadipour; Jan Marquenie; Srikanth Madikeri; Teodora Vukovic; Volker Dellwo; Kathy Reid; Francis M. Tyers; Ingo Siegert; Eleanor Chodroff
>
> **备注:** https://tidyvoice2026.github.io/
>
> **摘要:** The performance of speaker verification systems degrades significantly under language mismatch, a critical challenge exacerbated by the field's reliance on English-centric data. To address this, we propose the TidyVoice Challenge for cross-lingual speaker verification. The challenge leverages the TidyVoiceX dataset from the novel TidyVoice benchmark, a large-scale, multilingual corpus derived from Mozilla Common Voice, and specifically curated to isolate the effect of language switching across approximately 40 languages. Participants will be tasked with building systems robust to this mismatch, with performance primarily evaluated using the Equal Error Rate on cross-language trials. By providing standardized data, open-source baselines, and a rigorous evaluation protocol, this challenge aims to drive research towards fairer, more inclusive, and language-independent speaker recognition technologies, directly aligning with the Interspeech 2026 theme, "Speaking Together."
>
---
#### [new 002] PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出PhaseCoder，解决多模态大模型缺乏空间音频理解的问题。通过不依赖麦克风几何结构的编码器，实现鲁棒的空间音频嵌入，使LLM能进行空间推理与定位任务。**

- **链接: [https://arxiv.org/pdf/2601.21124v1](https://arxiv.org/pdf/2601.21124v1)**

> **作者:** Artem Dementyev; Wazeer Zulfikar; Sinan Hersek; Pascal Getreuer; Anurag Kumar; Vivek Kumar
>
> **摘要:** Current multimodal LLMs process audio as a mono stream, ignoring the rich spatial information essential for embodied AI. Existing spatial audio models, conversely, are constrained to fixed microphone geometries, preventing deployment across diverse devices. We present PhaseCoder, a transformer-only spatial audio encoder that is agnostic to microphone geometry. PhaseCoder takes raw multichannel audio and microphone coordinates as inputs to perform localization and produces robust spatial embeddings. We demonstrate that Gemma 3n LLM can be fine-tuned to reason over "Spatial Audio Tokens" produced by PhaseCoder. We show our encoder achieves state-of-the-art results on microphone-invariant localization benchmarks and, for the first time, enables an LLM to perform complex spatial reasoning and targeted transcription tasks from an arbitrary microphone array.
>
---
#### [new 003] Text-only adaptation in LLM-based ASR through text denoising
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于ASR领域，解决LLM在新领域适应时的文本对齐问题。通过文本去噪方法实现文本仅适应，保持跨模态对齐，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.20900v1](https://arxiv.org/pdf/2601.20900v1)**

> **作者:** Sergio Burdisso; Esaú Villatoro-Tello; Andrés Carofilis; Shashi Kumar; Kadri Hacioglu; Srikanth Madikeri; Pradeep Rangappa; Manjunath K E; Petr Motlicek; Shankar Venkatesan; Andreas Stolcke
>
> **备注:** Paper accepted at ICASSP 2026
>
> **摘要:** Adapting automatic speech recognition (ASR) systems based on large language models (LLMs) to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on target-domain text often disrupts the critical alignment between speech and text modalities learned by the projector, degrading performance. We introduce a novel text-only adaptation method that emulates the audio projection task by treating it as a text denoising task. Our approach thus trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
>
---
#### [new 004] SemanticAudio: Audio Generation and Editing in Semantic Space
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SemanticAudio，解决文本到音频生成中的语义对齐问题。通过高阶语义空间生成与编辑音频，提升生成音频与文本描述的一致性。**

- **链接: [https://arxiv.org/pdf/2601.21402v1](https://arxiv.org/pdf/2601.21402v1)**

> **作者:** Zheqi Dai; Guangyan Zhang; Haolin He; Xiquan Li; Jingyu Li; Chunyat Wu; Yiwen Guo; Qiuqiang Kong
>
> **摘要:** In recent years, Text-to-Audio Generation has achieved remarkable progress, offering sound creators powerful tools to transform textual inspirations into vivid audio. However, existing models predominantly operate directly in the acoustic latent space of a Variational Autoencoder (VAE), often leading to suboptimal alignment between generated audio and textual descriptions. In this paper, we introduce SemanticAudio, a novel framework that conducts both audio generation and editing directly in a high-level semantic space. We define this semantic space as a compact representation capturing the global identity and temporal sequence of sound events, distinct from fine-grained acoustic details. SemanticAudio employs a two-stage Flow Matching architecture: the Semantic Planner first generates these compact semantic features to sketch the global semantic layout, and the Acoustic Synthesizer subsequently produces high-fidelity acoustic latents conditioned on this semantic plan. Leveraging this decoupled design, we further introduce a training-free text-guided editing mechanism that enables precise attribute-level modifications on general audio without retraining. Specifically, this is achieved by steering the semantic generation trajectory via the difference of velocity fields derived from source and target text prompts. Extensive experiments demonstrate that SemanticAudio surpasses existing mainstream approaches in semantic alignment. Demo available at: https://semanticaudio1.github.io/
>
---
#### [new 005] Towards Robust Dysarthric Speech Recognition: LLM-Agent Post-ASR Correction Beyond WER
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决失语症语音识别中的语义失真问题。通过引入LLM代理进行后处理，提升识别的语义准确性。**

- **链接: [https://arxiv.org/pdf/2601.21347v1](https://arxiv.org/pdf/2601.21347v1)**

> **作者:** Xiuwen Zheng; Sixun Dong; Bornali Phukon; Mark Hasegawa-Johnson; Chang D. Yoo
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** While Automatic Speech Recognition (ASR) is typically benchmarked by word error rate (WER), real-world applications ultimately hinge on semantic fidelity. This mismatch is particularly problematic for dysarthric speech, where articulatory imprecision and disfluencies can cause severe semantic distortions. To bridge this gap, we introduce a Large Language Model (LLM)-based agent for post-ASR correction: a Judge-Editor over the top-k ASR hypotheses that keeps high-confidence spans, rewrites uncertain segments, and operates in both zero-shot and fine-tuned modes. In parallel, we release SAP-Hypo5, the largest benchmark for dysarthric speech correction, to enable reproducibility and future exploration. Under multi-perspective evaluation, our agent achieves a 14.51% WER reduction alongside substantial semantic gains, including a +7.59 pp improvement in MENLI and +7.66 pp in Slot Micro F1 on challenging samples. Our analysis further reveals that WER is highly sensitive to domain shift, whereas semantic metrics correlate more closely with downstream task performance.
>
---
#### [new 006] Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决LLM-based ASR中提示敏感性问题。通过引入可学习的提示投影模块，提升模型性能并减少波动。**

- **链接: [https://arxiv.org/pdf/2601.20898v1](https://arxiv.org/pdf/2601.20898v1)**

> **作者:** Sergio Burdisso; Esaú Villatoro-Tello; Shashi Kumar; Srikanth Madikeri; Andrés Carofilis; Pradeep Rangappa; Manjunath K E; Kadri Hacioglu; Petr Motlicek; Andreas Stolcke
>
> **备注:** Paper accepted at ICASSP 2026
>
> **摘要:** LLM-based automatic speech recognition (ASR), a well-established approach, connects speech foundation models to large language models (LLMs) through a speech-to-LLM projector, yielding promising results. A common design choice in these architectures is the use of a fixed, manually defined prompt during both training and inference. This setup not only enables applicability across a range of practical scenarios, but also helps maximize model performance. However, the impact of prompt design remains underexplored. This paper presents a comprehensive analysis of commonly used prompts across diverse datasets, showing that prompt choice significantly affects ASR performance and introduces instability, with no single prompt performing best across all cases. Inspired by the speech-to-LLM projector, we propose a prompt projector module, a simple, model-agnostic extension that learns to project prompt embeddings to more effective regions of the LLM input space, without modifying the underlying LLM-based ASR model. Experiments on four datasets show that the addition of a prompt projector consistently improves performance, reduces variability, and outperforms the best manually selected prompts.
>
---
#### [new 007] VoxMorph: Scalable Zero-shot Voice Identity Morphing via Disentangled Embeddings
- **分类: cs.SD; cs.CR; cs.LG; eess.AS**

- **简介: 该论文提出VoxMorph，解决语音生物识别中的零样本身份伪装问题，通过解耦声调与音色特征，生成高质量语音样本。**

- **链接: [https://arxiv.org/pdf/2601.20883v1](https://arxiv.org/pdf/2601.20883v1)**

> **作者:** Bharath Krishnamurthy; Ajita Rattani
>
> **备注:** Accepted to IEEE ICASSP 2026 (51st International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2026). 5 pages, 1 figure, 3 tables. Project page: https://vcbsl.github.io/VoxMorph/
>
> **摘要:** Morphing techniques generate artificial biometric samples that combine features from multiple individuals, allowing each contributor to be verified against a single enrolled template. While extensively studied in face recognition, this vulnerability remains largely unexplored in voice biometrics. Prior work on voice morphing is computationally expensive, non-scalable, and limited to acoustically similar identity pairs, constraining practical deployment. Moreover, existing sound-morphing methods target audio textures, music, or environmental sounds and are not transferable to voice identity manipulation. We propose VoxMorph, a zero-shot framework that produces high-fidelity voice morphs from as little as five seconds of audio per subject without model retraining. Our method disentangles vocal traits into prosody and timbre embeddings, enabling fine-grained interpolation of speaking style and identity. These embeddings are fused via Spherical Linear Interpolation (Slerp) and synthesized using an autoregressive language model coupled with a Conditional Flow Matching network. VoxMorph achieves state-of-the-art performance, delivering a 2.6x gain in audio quality, a 73% reduction in intelligibility errors, and a 67.8% morphing attack success rate on automated speaker verification systems under strict security thresholds. This work establishes a practical and scalable paradigm for voice morphing with significant implications for biometric security. The code and dataset are available on our project page: https://vcbsl.github.io/VoxMorph/
>
---
#### [new 008] Generalizable Prompt Tuning for Audio-Language Models via Semantic Expansion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-语言模型任务，旨在解决提示调优的泛化能力问题。提出SEPT框架，通过语义扩展增强嵌入空间结构，提升模型泛化性能。**

- **链接: [https://arxiv.org/pdf/2601.20867v1](https://arxiv.org/pdf/2601.20867v1)**

> **作者:** Jaehyuk Jang; Wonjun Lee; Kangwook Ko; Changick Kim
>
> **摘要:** Prompt tuning has achieved remarkable progress in vision-language models (VLMs) and is recently being adopted for audio-language models (ALMs). However, its generalization ability in ALMs remains largely underexplored. We observe that conventional prompt tuning for ALMs also suffers from the Base-New Tradeoff, and we identify that this issue stems from the disrupted semantic structure of the embedding space. To address this issue, we propose Semantically Expanded Prompt Tuning (SEPT)-a plug-and-play framework that explicitly regularizes the prompt embedding space by incorporating semantic neighbors generated by large language models. SEPT introduces a novel semantic expansion loss with margin constraints that promote intra-class compactness and inter-class separability, thereby enhancing the semantic structure of the prompt embedding space. For comprehensive evaluation, we establish the first benchmark setup for prompt generalization in ALMs, covering both base-to-new generalization and cross-dataset transferability. Extensive experiments demonstrate that SEPT consistently improves generalization performance across multiple prompt tuning baselines, while maintaining computational cost during inference. Codes are available in https://github.com/jhyukjang/SEPT.
>
---
#### [new 009] DNN-Based Online Source Counting Based on Spatial Generalized Magnitude Squared Coherence
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声源计数任务，解决在线检测声源数量变化的问题。通过空间相干性分析与神经网络，实现对活跃声源数的实时识别。**

- **链接: [https://arxiv.org/pdf/2601.21114v1](https://arxiv.org/pdf/2601.21114v1)**

> **作者:** Henri Gode; Simon Doclo
>
> **备注:** in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026, Barcelona, Spain
>
> **摘要:** The number of active sound sources is a key parameter in many acoustic signal processing tasks, such as source localization, source separation, and multi-microphone speech enhancement. This paper proposes a novel method for online source counting by detecting changes in the number of active sources based on spatial coherence. The proposed method exploits the fact that a single coherent source in spatially white background noise yields high spatial coherence, whereas only noise results in low spatial coherence. By applying a spatial whitening operation, the source counting problem is reformulated as a change detection task, aiming to identify the time frames when the number of active sources changes. The method leverages the generalized magnitude-squared coherence as a measure to quantify spatial coherence, providing features for a compact neural network trained to detect source count changes framewise. Simulation results with binaural hearing aids in reverberant acoustic scenes with up to 4 speakers and background noise demonstrate the effectiveness of the proposed method for online source counting.
>
---
#### [new 010] SW-ASR: A Context-Aware Hybrid ASR Pipeline for Robust Single Word Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于单字语音识别任务，旨在提升噪声和压缩音频下的识别鲁棒性。通过混合ASR前端与验证层，结合上下文和语言模型优化匹配策略，显著提高准确率。**

- **链接: [https://arxiv.org/pdf/2601.20890v1](https://arxiv.org/pdf/2601.20890v1)**

> **作者:** Manali Sharma; Riya Naik; Buvaneshwari G
>
> **摘要:** Single-word Automatic Speech Recognition (ASR) is a challenging task due to the lack of linguistic context and sensitivity to noise, pronunciation variation, and channel artifacts, especially in low-resource, communication-critical domains such as healthcare and emergency response. This paper reviews recent deep learning approaches and proposes a modular framework for robust single-word detection. The system combines denoising and normalization with a hybrid ASR front end (Whisper + Vosk) and a verification layer designed to handle out-of-vocabulary words and degraded audio. The verification layer supports multiple matching strategies, including embedding similarity, edit distance, and LLM-based matching with optional contextual guidance. We evaluate the framework on the Google Speech Commands dataset and a curated real-world dataset collected from telephony and messaging platforms under bandwidth-limited conditions. Results show that while the hybrid ASR front end performs well on clean audio, the verification layer significantly improves accuracy on noisy and compressed channels. Context-guided and LLM-based matching yield the largest gains, demonstrating that lightweight verification and context mechanisms can substantially improve single-word ASR robustness without sacrificing latency required for real-time telephony applications.
>
---
#### [new 011] Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts
- **分类: eess.AS**

- **简介: 该论文属于语音质量评估任务，解决如何提升帧级评分的稳定性与可解释性。通过引入段级一致性约束，优化模型并应用于检测低质量语音和合成伪影。**

- **链接: [https://arxiv.org/pdf/2601.21886v1](https://arxiv.org/pdf/2601.21886v1)**

> **作者:** Michael Kuhlmann; Alexander Werning; Thilo von Neumann; Reinhold Haeb-Umbach
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** A large number of works view the automatic assessment of speech from an utterance- or system-level perspective. While such approaches are good in judging overall quality, they cannot adequately explain why a certain score was assigned to an utterance. frame-level scores can provide better interpretability, but models predicting them are harder to tune and regularize since no strong targets are available during training. In this work, we show that utterance-level speech quality predictors can be regularized with a segment-based consistency constraint which notably reduces frame-level stochasticity. We then demonstrate two applications involving frame-level scores: The partial spoof scenario and the detection of synthesis artefacts in two state-of-the-art text-to-speech systems. For the latter, we perform listening tests and confirm that listeners rate segments to be of poor quality more often in the set defined by low frame-level scores than in a random control set.
>
---
#### [new 012] Unseen but not Unknown: Using Dataset Concealment to Robustly Evaluate Speech Quality Estimation Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量估计任务，旨在解决模型在真实场景中表现不佳的问题。提出Dataset Concealment方法，评估模型泛化能力，并通过Aligner提升多数据集训练效果。**

- **链接: [https://arxiv.org/pdf/2601.21110v1](https://arxiv.org/pdf/2601.21110v1)**

> **作者:** Jaden Pieper; Stephen D. Voran
>
> **备注:** To be appear in Proc. ICASSP 2026
>
> **摘要:** We introduce Dataset Concealment (DSC), a rigorous new procedure for evaluating and interpreting objective speech quality estimation models. DSC quantifies and decomposes the performance gap between research results and real-world application requirements, while offering context and additional insights into model behavior and dataset characteristics. We also show the benefits of addressing the corpus effect by using the dataset Aligner from AlignNet when training models with multiple datasets. We demonstrate DSC and the improvements from the Aligner using nine training datasets and nine unseen datasets with three well-studied models: MOSNet, NISQA, and a Wav2Vec2.0-based model. DSC provides interpretable views of the generalization capabilities and limitations of models, while allowing all available data to be used at training. An additional result is that adding the 1000 parameter dataset Aligner to the 94 million parameter Wav2Vec model during training does significantly improve the resulting model's ability to estimate speech quality for unseen data.
>
---
#### [new 013] Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造定位任务，旨在解决部分音频篡改难以检测的问题。提出Segment-Aware Learning框架，通过关注段内结构提升定位效果。**

- **链接: [https://arxiv.org/pdf/2601.21925v1](https://arxiv.org/pdf/2601.21925v1)**

> **作者:** Yuchen Mao; Wen Huang; Yanmin Qian
>
> **摘要:** Localizing partial deepfake audio, where only segments of speech are manipulated, remains challenging due to the subtle and scattered nature of these modifications. Existing approaches typically rely on frame-level predictions to identify spoofed segments, and some recent methods improve performance by concentrating on the transitions between real and fake audio. However, we observe that these models tend to over-rely on boundary artifacts while neglecting the manipulated content that follows. We argue that effective localization requires understanding the entire segments beyond just detecting transitions. Thus, we propose Segment-Aware Learning (SAL), a framework that encourages models to focus on the internal structure of segments. SAL introduces two core techniques: Segment Positional Labeling, which provides fine-grained frame supervision based on relative position within a segment; and Cross-Segment Mixing, a data augmentation method that generates diverse segment patterns. Experiments across multiple deepfake localization datasets show that SAL consistently achieves strong performance in both in-domain and out-of-domain settings, with notable gains in non-boundary regions and reduced reliance on transition artifacts. The code is available at https://github.com/SentryMao/SAL.
>
---
#### [new 014] DisContSE: Single-Step Diffusion Speech Enhancement Based on Joint Discrete and Continuous Embeddings
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决高计算复杂度和语音识别性能差的问题。提出DisContSE模型，结合离散和连续嵌入，实现高效单步推理和更好语音质量。**

- **链接: [https://arxiv.org/pdf/2601.21940v1](https://arxiv.org/pdf/2601.21940v1)**

> **作者:** Yihui Fu; Tim Fingscheidt
>
> **备注:** Accepted by IEEE ICASSP 2026
>
> **摘要:** Diffusion speech enhancement on discrete audio codec features gain immense attention due to their improved speech component reconstruction capability. However, they usually suffer from high inference computational complexity due to multiple reverse process iterations. Furthermore, they generally achieve promising results on non-intrusive metrics but show poor performance on intrusive metrics, as they may struggle in reconstructing the correct phones. In this paper, we propose DisContSE, an efficient diffusion-based speech enhancement model on joint discrete codec tokens and continuous embeddings. Our contributions are three-fold. First, we formulate both a discrete and a continuous enhancement module operating on discrete audio codec tokens and continuous embeddings, respectively, to achieve improved fidelity and intelligibility simultaneously. Second, a semantic enhancement module is further adopted to achieve optimal phonetic accuracy. Third, we achieve a single-step efficient reverse process in inference with a novel quantization error mask initialization strategy, which, according to our knowledge, is the first successful single-step diffusion speech enhancement based on an audio codec. Trained and evaluated on URGENT 2024 Speech Enhancement Challenge data splits, the proposed DisContSE excels top-reported time- and frequency-domain diffusion baseline methods in PESQ, POLQA, UTMOS, and in a subjective ITU-T P.808 listening test, clearly achieving an overall top rank.
>
---
#### [new 015] Representation-Regularized Convolutional Audio Transformer for Audio Understanding
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于音频理解任务，旨在解决现有方法在音频特征建模和训练效率上的不足。提出CAT框架，结合多分辨率块和表示正则化，提升性能并加快收敛。**

- **链接: [https://arxiv.org/pdf/2601.21612v1](https://arxiv.org/pdf/2601.21612v1)**

> **作者:** Bing Han; Chushu Zhou; Yifan Yang; Wei Wang; Chenda Li; Wangyou Zhang; Yanmin Qian
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Bootstrap-based Self-Supervised Learning (SSL) has achieved remarkable progress in audio understanding. However, existing methods typically operate at a single level of granularity, limiting their ability to model the diverse temporal and spectral structures inherent in complex audio signals. Furthermore, bootstrapping representations from scratch is computationally expensive, often requiring extensive training to converge. In this work, we propose the Convolutional Audio Transformer (CAT), a unified framework designed to address these challenges. First, to capture hierarchical audio features, CAT incorporates a Multi-resolution Block that aggregates information across varying granularities. Second, to enhance training efficiency, we introduce a Representation Regularization objective. Drawing inspiration from generative modeling, this auxiliary task guides the student model by aligning its predictions with high-quality semantic representations from frozen, pre-trained external encoders. Experimental results demonstrate that CAT significantly outperforms baselines on audio understanding benchmarks. Notably, it achieves competitive performance on the AudioSet 20k dataset with 5 times faster convergence than existing methods. Codes and checkpoints will be released soon at https://github.com/realzhouchushu/CAT.
>
---
#### [new 016] A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，研究预训练数据选择策略。旨在提升自监督学习的ASR性能，发现数据长度比多样性更重要，可减少数据量和训练时间。**

- **链接: [https://arxiv.org/pdf/2601.20896v1](https://arxiv.org/pdf/2601.20896v1)**

> **作者:** Ryan Whetten; Titouan Parcollet; Marco Dinarelli; Yannick Estève
>
> **备注:** Accepted for publication in the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)
>
> **摘要:** Self-supervised learning (SSL) has transformed speech processing, yet its reliance on massive pre-training datasets remains a bottleneck. While robustness is often attributed to scale and diversity, the role of the data distribution is less understood. We systematically examine how curated subsets of pre-training data influence Automatic Speech Recognition (ASR) performance. Surprisingly, optimizing for acoustic, speaker, or linguistic diversity yields no clear improvements over random sampling. Instead, we find that prioritizing the longest utterances achieves superior ASR results while using only half the original dataset, reducing pre-training time by 24% on a large corpora. These findings suggest that for pre-training speech SSL models, data length is a more critical factor than either data diversity or overall data quantity for performance and efficiency, offering a new perspective for data selection strategies in SSL speech processing.
>
---
#### [new 017] Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音编辑检测与内容定位任务，旨在解决传统方法难以应对无痕神经语音编辑的问题。通过构建数据集并提出PELM模型，实现更准确的检测与定位。**

- **链接: [https://arxiv.org/pdf/2601.21463v1](https://arxiv.org/pdf/2601.21463v1)**

> **作者:** Jun Xue; Yi Chai; Yanzhen Ren; Jinshen He; Zhiqiang Tang; Zhuolin Yi; Yihuan Huang; Yuankun Xie; Yujie Chen
>
> **摘要:** Speech editing achieves semantic inversion by performing fine-grained segment-level manipulation on original utterances, while preserving global perceptual naturalness. Existing detection studies mainly focus on manually edited speech with explicit splicing artifacts, and therefore struggle to cope with emerging end-to-end neural speech editing techniques that generate seamless acoustic transitions. To address this challenge, we first construct a large-scale bilingual dataset, AiEdit, which leverages large language models to drive precise semantic tampering logic and employs multiple advanced neural speech editing methods for data synthesis, thereby filling the gap of high-quality speech editing datasets. Building upon this foundation, we propose PELM (Prior-Enhanced Audio Large Language Model), the first large-model framework that unifies speech editing detection and content localization by formulating them as an audio question answering task. To mitigate the inherent forgery bias and semantic-priority bias observed in existing audio large models, PELM incorporates word-level probability priors to provide explicit acoustic cues, and further designs a centroid-aggregation-based acoustic consistency perception loss to explicitly enforce the modeling of subtle local distribution anomalies. Extensive experimental results demonstrate that PELM significantly outperforms state-of-the-art methods on both the HumanEdit and AiEdit datasets, achieving equal error rates (EER) of 0.57\% and 9.28\% (localization), respectively.
>
---
#### [new 018] Understanding Frechet Speech Distance for Synthetic Speech Quality Evaluation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音质量评估任务，旨在解决合成语音客观评价难题。通过分析FSD和SMMD在不同嵌入下的表现，验证其与人类感知的相关性。**

- **链接: [https://arxiv.org/pdf/2601.21386v1](https://arxiv.org/pdf/2601.21386v1)**

> **作者:** June-Woo Kim; Dhruv Agarwal; Federica Cerina
>
> **备注:** accepted to ICASSP 2026
>
> **摘要:** Objective evaluation of synthetic speech quality remains a critical challenge. Human listening tests are the gold standard, but costly and impractical at scale. Fréchet Distance has emerged as a promising alternative, yet its reliability depends heavily on the choice of embeddings and experimental settings. In this work, we comprehensively evaluate Fréchet Speech Distance (FSD) and its variant Speech Maximum Mean Discrepancy (SMMD) under varied embeddings and conditions. We further incorporate human listening evaluations alongside TTS intelligibility and synthetic-trained ASR WER to validate the perceptual relevance of these metrics. Our findings show that WavLM Base+ features yield the most stable alignment with human ratings. While FSD and SMMD cannot fully replace subjective evaluation, we show that they can serve as complementary, cost-efficient, and reproducible measures, particularly useful when large-scale or direct listening assessments are infeasible. Code is available at https://github.com/kaen2891/FrechetSpeechDistance.
>
---
#### [new 019] Music Plagiarism Detection: Problem Formulation and a Segment-based Solution
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐抄袭检测任务，旨在明确该任务的定义与问题，提出基于片段转录的解决方案，并提供数据集和代码支持。**

- **链接: [https://arxiv.org/pdf/2601.21260v1](https://arxiv.org/pdf/2601.21260v1)**

> **作者:** Seonghyeon Go; Yumin Kim
>
> **摘要:** Recently, the problem of music plagiarism has emerged as an even more pressing social issue. As music information retrieval research advances, there is a growing effort to address issues related to music plagiarism. However, many studies, including our previous work, have conducted research without clearly defining what the music plagiarism detection task actually involves. This lack of a clear definition has slowed research progress and made it hard to apply results to real-world scenarios. To fix this situation, we defined how Music Plagiarism Detection is different from other MIR tasks and explained what problems need to be solved. We introduce the Similar Music Pair dataset to support this newly defined task. In addition, we propose a method based on segment transcription as one way to solve the task. Our demo and dataset are available at https://github.com/Mippia/ICASSP2026-MPD.
>
---
#### [new 020] Multilingual Dysarthric Speech Assessment Using Universal Phone Recognition and Language-Specific Phonemic Contrast Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言失语语音评估任务，旨在解决跨语言智能识别难题。通过结合通用音素识别与语言特异性音素对比，提出新评估指标，提升不同语言下语音可理解性分析的准确性。**

- **链接: [https://arxiv.org/pdf/2601.21205v1](https://arxiv.org/pdf/2601.21205v1)**

> **作者:** Eunjung Yeo; Julie M. Liss; Visar Berisha; David R. Mortensen
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** The growing prevalence of neurological disorders associated with dysarthria motivates the need for automated intelligibility assessment methods that are applicalbe across languages. However, most existing approaches are either limited to a single language or fail to capture language-specific factors shaping intelligibility. We present a multilingual phoneme-production assessment framework that integrates universal phone recognition with language-specific phoneme interpretation using contrastive phonological feature distances for phone-to-phoneme mapping and sequence alignment. The framework yields three metrics: phoneme error rate (PER), phonological feature error rate (PFER), and a newly proposed alignment-free measure, phoneme coverage (PhonCov). Analysis on English, Spanish, Italian, and Tamil show that PER benefits from the combination of mapping and alignment, PFER from alignment alone, and PhonCov from mapping. Further analyses demonstrate that the proposed framework captures clinically meaningful patterns of intelligibility degradation consistent with established observations of dysarthric speech.
>
---
#### [new 021] Qwen3-ASR Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升多语言ASR与语音对齐效果。提出两个ASR模型和一个非自回归对齐模型，优化准确率与效率。**

- **链接: [https://arxiv.org/pdf/2601.21337v1](https://arxiv.org/pdf/2601.21337v1)**

> **作者:** Xian Shi; Xiong Wang; Zhifang Guo; Yongqi Wang; Pei Zhang; Xinyu Zhang; Zishan Guo; Hongkun Hao; Yu Xi; Baosong Yang; Jin Xu; Jingren Zhou; Junyang Lin
>
> **备注:** https://github.com/QwenLM/Qwen3-ASR
>
> **摘要:** In this report, we introduce Qwen3-ASR family, which includes two powerful all-in-one speech recognition models and a novel non-autoregressive speech forced alignment model. Qwen3-ASR-1.7B and Qwen3-ASR-0.6B are ASR models that support language identification and ASR for 52 languages and dialects. Both of them leverage large-scale speech training data and the strong audio understanding ability of their foundation model Qwen3-Omni. We conduct comprehensive internal evaluation besides the open-sourced benchmarks as ASR models might differ little on open-sourced benchmark scores but exhibit significant quality differences in real-world scenarios. The experiments reveal that the 1.7B version achieves SOTA performance among open-sourced ASR models and is competitive with the strongest proprietary APIs while the 0.6B version offers the best accuracy-efficiency trade-off. Qwen3-ASR-0.6B can achieve an average TTFT as low as 92ms and transcribe 2000 seconds speech in 1 second at a concurrency of 128. Qwen3-ForcedAligner-0.6B is an LLM based NAR timestamp predictor that is able to align text-speech pairs in 11 languages. Timestamp accuracy experiments show that the proposed model outperforms the three strongest force alignment models and takes more advantages in efficiency and versatility. To further accelerate the community research of ASR and audio understanding, we release these models under the Apache 2.0 license.
>
---
#### [new 022] Evaluating Spatialized Auditory Cues for Rapid Attention Capture in XR
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究空间音频在XR中快速引导注意力的可行性，解决如何在短时间内通过听觉线索传递方向信息的问题。通过实验评估空间音频的准确性及短期训练的效果。**

- **链接: [https://arxiv.org/pdf/2601.21264v1](https://arxiv.org/pdf/2601.21264v1)**

> **作者:** Yoonsang Kim; Swapnil Dey; Arie Kaufman
>
> **备注:** 8 pages, 4 figures. This is the author's version of the article that will appear at the IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (IEEE VRW) 2026
>
> **摘要:** In time-critical eXtended reality (XR) scenarios where users must rapidly reorient their attention to hazards, alerts, or instructions while engaged in a primary task, spatial audio can provide an immediate directional cue without occupying visual bandwidth. However, such scenarios can afford only a brief auditory exposure, requiring users to interpret sound direction quickly and without extended listening or head-driven refinement. This paper reports a controlled exploratory study of rapid spatial-audio localization in XR. Using HRTF-rendered broadband stimuli presented from a semi-dense set of directions around the listener, we quantify how accurately users can infer coarse direction from brief audio alone. We further examine the effects of short-term visuo-auditory feedback training as a lightweight calibration mechanism. Our findings show that brief spatial cues can convey coarse directional information, and that even short calibration can improve users' perception of aural signals. While these results highlight the potential of spatial audio for rapid attention guidance, they also show that auditory cues alone may not provide sufficient precision for complex or high-stakes tasks, and that spatial audio may be most effective when complemented by other sensory modalities or visual cues, without relying on head-driven refinement. We leverage this study on spatial audio as a preliminary investigation into a first-stage attention-guidance channel for wearable XR (e.g., VR head-mounted displays and AR smart glasses), and provide design insights on stimulus selection and calibration for time-critical use.
>
---
#### [new 023] MIDI-LLaMA: An Instruction-Following Multimodal LLM for Symbolic Music Understanding
- **分类: cs.MM; cs.SD**

- **简介: 该论文提出MIDI-LLaMA，解决符号音乐理解任务，通过多模态大语言模型提升音乐理解能力。**

- **链接: [https://arxiv.org/pdf/2601.21740v1](https://arxiv.org/pdf/2601.21740v1)**

> **作者:** Meng Yang; Jon McCormack; Maria Teresa Llano; Wanchao Su; Chao Lei
>
> **备注:** Accepted for publication at International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Recent advances in multimodal large language models (MLLM) for audio music have demonstrated strong capabilities in music understanding, yet symbolic music, a fundamental representation of musical structure, remains unexplored. In this work, we introduce MIDI-LLaMA, the first instruction-following MLLM for symbolic music understanding. Our approach aligns the MIDI encoder MusicBERT and Llama-3-8B via a two-stage pipeline comprising feature alignment and instruction tuning. To support training, we design a scalable annotation pipeline that annotates GiantMIDI-Piano with fine-grained metadata, resulting in a MIDI-text dataset. Compared with the baseline trained on converting MIDI into ABC notation under the same instruction-tuning procedure, MIDI-LLaMA substantially outperforms in captioning and semantic alignment in question answering. Human evaluation further confirms the advantages of MIDI-LLaMA in music understanding, emotion recognition, creativity, and overall preference. These findings demonstrate that incorporating symbolic music into large language models enhances their capacity for musical understanding.
>
---
#### [new 024] asr_eval: Algorithms and tools for multi-reference and streaming speech recognition evaluation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别评估任务，解决多参考标注和流式识别评价问题。提出改进的对齐算法，构建新数据集，并开发评估工具。**

- **链接: [https://arxiv.org/pdf/2601.20992v1](https://arxiv.org/pdf/2601.20992v1)**

> **作者:** Oleg Sedukhin; Andrey Kostin
>
> **摘要:** We propose several improvements to the speech recognition evaluation. First, we propose a string alignment algorithm that supports both multi-reference labeling, arbitrary-length insertions and better word alignment. This is especially useful for non-Latin languages, those with rich word formation, to label cluttered or longform speech. Secondly, we collect a novel test set DiverseSpeech-Ru of longform in-the-wild Russian speech with careful multi-reference labeling. We also perform multi-reference relabeling of popular Russian tests set and study fine-tuning dynamics on its corresponding train set. We demonstrate that the model often adopts to dataset-specific labeling, causing an illusion of metric improvement. Based on the improved word alignment, we develop tools to evaluate streaming speech recognition and to align multiple transcriptions to compare them visually. Additionally, we provide uniform wrappers for many offline and streaming speech recognition models. Our code will be made publicly available.
>
---
#### [new 025] Position-invariant Fine-tuning of Speech Enhancement Models with Self-supervised Speech Representations
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，解决SSL模型微调中位置敏感的问题。通过引入零填充和软DTW损失，实现位置不变的微调，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21084v1](https://arxiv.org/pdf/2601.21084v1)**

> **作者:** Amit Meghanani; Thomas Hain
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Integrating front-end speech enhancement (SE) models with self-supervised learning (SSL)-based speech models is effective for downstream tasks in noisy conditions. SE models are commonly fine-tuned using SSL representations with mean squared error (MSE) loss between enhanced and clean speech. However, MSE is prone to exploiting positional embeddings in SSL models, allowing the objective to be minimised through positional correlations instead of content-related information. This work frames the problem as a general limitation of self-supervised representation fine-tuning and investigates it through representation-guided SE. Two strategies are considered: (1) zero-padding, previously explored in SSL pre-training but here examined in the fine-tuning setting, and (2) speed perturbations with a soft-DTW loss. Experiments show that the soft-DTW-based approach achieves faster convergence and improved downstream performance, underscoring the importance of position-invariant fine-tuning in SSL-based speech modelling.
>
---
## 更新

#### [replaced 001] Interpretable Modeling of Articulatory Temporal Dynamics from real-time MRI for Phoneme Recognition
- **分类: eess.IV; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在通过rtMRI数据提升发音识别的准确性和可解释性。研究比较了不同特征表示，发现多特征融合效果最佳，揭示了舌和唇的重要性。**

- **链接: [https://arxiv.org/pdf/2509.15689v2](https://arxiv.org/pdf/2509.15689v2)**

> **作者:** Jay Park; Hong Nguyen; Sean Foley; Jihwan Lee; Yoonjeong Lee; Dani Byrd; Shrikanth Narayanan
>
> **摘要:** Real-time Magnetic Resonance Imaging (rtMRI) visualizes vocal tract action, offering a comprehensive window into speech articulation. However, its signals are high dimensional and noisy, hindering interpretation. We investigate compact representations of spatiotemporal articulatory dynamics for phoneme recognition from midsagittal vocal tract rtMRI videos. We compare three feature types: (1) raw video, (2) optical flow, and (3) six linguistically-relevant regions of interest (ROIs) for articulator movements. We evaluate models trained independently on each representation, as well as multi-feature combinations. Results show that multi-feature models consistently outperform single-feature baselines, with the lowest phoneme error rate (PER) of 0.34 obtained by combining ROI and raw video. Temporal fidelity experiments demonstrate a reliance on fine-grained articulatory dynamics, while ROI ablation studies reveal strong contributions from tongue and lips. Our findings highlight how rtMRI-derived features provide accuracy and interpretability, and establish strategies for leveraging articulatory data in speech processing.
>
---
#### [replaced 002] Position: Towards Responsible Evaluation for Text-to-Speech
- **分类: eess.AS**

- **简介: 该论文属于文本到语音生成领域的评估任务，旨在解决现有评估方法不足的问题，提出负责任的评估框架，涵盖能力反映、标准对比和伦理风险评估。**

- **链接: [https://arxiv.org/pdf/2510.06927v2](https://arxiv.org/pdf/2510.06927v2)**

> **作者:** Yifan Yang; Hui Wang; Bing Han; Shujie Liu; Jinyu Li; Yong Qin; Xie Chen
>
> **摘要:** Recent advances in text-to-speech (TTS) technology have enabled systems to generate speech that is often indistinguishable from human speech, bringing benefits to accessibility, content creation, and human-computer interaction. However, current evaluation practices are increasingly inadequate for capturing the full range of capabilities, limitations, and societal impacts of modern TTS systems. This position paper introduces the concept of Responsible Evaluation and argues that it is essential and urgent for the next phase of TTS development, structured through three progressive levels: (1) ensuring the faithful and accurate reflection of a model's true capabilities and limitations, with more robust, discriminative, and comprehensive objective and subjective scoring methodologies; (2) enabling comparability, standardization, and transferability through standardized benchmarks, transparent reporting, and transferable evaluation metrics; and (3) assessing and mitigating ethical risks associated with forgery, misuse, privacy violations, and security vulnerabilities. Through this concept, we critically examine current evaluation practices, identify systemic shortcomings, and propose actionable recommendations. We hope this concept will not only foster more reliable and trustworthy TTS technology but also guide its development toward ethically sound and societally beneficial applications.
>
---
#### [replaced 003] No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，解决GRPO训练中因缺乏可验证的韵律奖励导致语音单调的问题。通过迭代直接偏好优化，提升韵律自然度，同时保持较低错误率。**

- **链接: [https://arxiv.org/pdf/2509.18531v2](https://arxiv.org/pdf/2509.18531v2)**

> **作者:** Seungyoun Shin; Dongha Ahn; Jiwoo Kim; Sungwook Jeon
>
> **备注:** ICASSP 2026
>
> **摘要:** Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{https://tts.ch.dev}
>
---
#### [replaced 004] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于 Talking Head Generation 任务，旨在解决扩散模型在实时生成中的速度慢和非自回归问题。通过引入ID-Context Cache和ASD策略，提升生成效率与一致性。**

- **链接: [https://arxiv.org/pdf/2512.11229v3](https://arxiv.org/pdf/2512.11229v3)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation (THG). However, slow inference speeds and prevalent non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, a pioneering diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through a spatiotemporal variational autoencoder with a high compression ratio. Additionally, to enable semi-autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles into key-value caching for maintaining identity consistency and temporal coherence during long-term streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) strategy is proposed to mitigate error accumulation and enhance temporal consistency in streaming generation, leveraging a non-streaming teacher with an asynchronous noise schedule to supervise the streaming student. REST bridges the gap between autoregressive and diffusion-based approaches, achieving a breakthrough in efficiency for applications requiring real-time THG. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [replaced 005] LLM2Fx-Tools: Tool Calling For Music Post-Production
- **分类: cs.SD**

- **简介: 该论文提出LLM2Fx-Tools，用于音乐后期制作中的音频效果链生成。解决如何通过大语言模型自动选择和排序音频效果的问题。工作包括构建数据集、设计框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.01559v2](https://arxiv.org/pdf/2512.01559v2)**

> **作者:** Seungheon Doh; Junghyun Koo; Marco A. Martínez-Ramírez; Woosung Choi; Wei-Hsiang Liao; Qiyu Wu; Juhan Nam; Yuki Mitsufuji
>
> **备注:** ICLR 2026
>
> **摘要:** This paper introduces LLM2Fx-Tools, a multimodal tool-calling framework that generates executable sequences of audio effects (Fx-chain) for music post-production. LLM2Fx-Tools uses a large language model (LLM) to understand audio inputs, select audio effects types, determine their order, and estimate parameters, guided by chain-of-thought (CoT) planning. We also present LP-Fx, a new instruction-following dataset with structured CoT annotations and tool calls for audio effects modules. Experiments show that LLM2Fx-Tools can infer an Fx-chain and its parameters from pairs of unprocessed and processed audio, enabled by autoregressive sequence modeling, tool calling, and CoT reasoning. We further validate the system in a style transfer setting, where audio effects information is transferred from a reference source and applied to new content. Finally, LLM-as-a-judge evaluation demonstrates that our approach generates appropriate CoT reasoning and responses for music production queries. To our knowledge, this is the first work to apply LLM-based tool calling to audio effects modules, enabling interpretable and controllable music production.
>
---
#### [replaced 006] MK-SGC-SC: Multiple Kernel Guided Sparse Graph Construction in Spectral Clustering for Unsupervised Speaker Diarization
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音处理中的说话人日志任务，旨在解决无监督说话人聚类问题。通过多核相似性度量构建稀疏图，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.19946v2](https://arxiv.org/pdf/2601.19946v2)**

> **作者:** Nikhil Raghav; Avisek Gupta; Swagatam Das; Md Sahidullah
>
> **备注:** 5 pages
>
> **摘要:** Speaker diarization aims to segment audio recordings into regions corresponding to individual speakers. Although unsupervised speaker diarization is inherently challenging, the prospect of identifying speaker regions without pretraining or weak supervision motivates research on clustering techniques. In this work, we share the notable observation that measuring multiple kernel similarities of speaker embeddings to thereafter craft a sparse graph for spectral clustering in a principled manner is sufficient to achieve state-of-the-art performances in a fully unsupervised setting. Specifically, we consider four polynomial kernels and a degree one arccosine kernel to measure similarities in speaker embeddings, using which sparse graphs are constructed in a principled manner to emphasize local similarities. Experiments show the proposed approach excels in unsupervised speaker diarization over a variety of challenging environments in the DIHARD-III, AMI, and VoxConverse corpora. To encourage further research, our implementations are available at https://github.com/nikhilraghav29/MK-SGC-SC.
>
---
#### [replaced 007] Do We Need EMA for Diffusion-Based Speech Enhancement? Toward a Magnitude-Preserving Network Architecture
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，研究扩散模型中EMA的作用，提出一种保持幅度的网络结构，并通过两种跳跃连接配置提升性能。**

- **链接: [https://arxiv.org/pdf/2505.05216v3](https://arxiv.org/pdf/2505.05216v3)**

> **作者:** Julius Richter; Danilo de Oliveira; Timo Gerkmann
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** We study diffusion-based speech enhancement using a Schrodinger bridge formulation and extend the EDM2 framework to this setting. We employ time-dependent preconditioning of network inputs and outputs to stabilize training and explore two skip-connection configurations that allow the network to predict either environmental noise or clean speech. To control activation and weight magnitudes, we adopt a magnitude-preserving architecture and learn the contribution of the noisy input within each network block for improved conditioning. We further analyze the impact of exponential moving average (EMA) parameter smoothing by approximating different EMA profiles post training, finding that, unlike in image generation, short or absent EMA consistently yields better speech enhancement performance. Experiments on VoiceBank-DEMAND and EARS-WHAM demonstrate competitive signal-to-distortion ratios and perceptual scores, with the two skip-connection variants exhibiting complementary strengths. These findings provide new insights into EMA behavior, magnitude preservation, and skip-connection design for diffusion-based speech enhancement.
>
---
#### [replaced 008] MusicWeaver: Composer-Style Structural Editing and Minute-Scale Coherent Music Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出MusicWeaver，解决音乐生成中结构编辑和长时 coherence 的问题。通过分阶段生成和引入全局局部 Transformer 与动机记忆模块，提升音乐的可控性和连贯性。**

- **链接: [https://arxiv.org/pdf/2509.21714v2](https://arxiv.org/pdf/2509.21714v2)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Recent advances in music generation produce impressive samples, however, practical creation still lacks two key capabilities: composer-style structural editing and minute-scale coherence. We present MusicWeaver, a framework for generating and editing long-range music using a human-interpretable intermediate representation with guaranteed edit locality. MusicWeaver decomposes generation into two stages: it first predicts a structured plan, a multi-level song program encoding musical attributes that composers can directly edit, and then renders audio conditioned on this plan. To ensure minute-scale coherence, we introduce a Global-Local Diffusion Transformer, where a global path captures long-range musical progression via compressed representations and memory, while a local path synthesizes fine-grained acoustic detail. We further propose a Motif Memory Retrieval module that enables consistent motif recurrence with controllable variation. For editing, we propose Projected Diffusion Inpainting, an inpainting method that denoises only user-specified regions and preserves unchanged content, allowing repeated edits without drift. Finally, we introduce Structure Coherence Score and Edit Fidelity Score to evaluate long-range form and edit realization. Experiments demonstrate that MusicWeaver achieves state-of-the-art fidelity, controllability, and long-range coherence.
>
---
#### [replaced 009] Decoding Speech Envelopes from Electroencephalogram with a Contrastive Pearson Correlation Coefficient Loss
- **分类: eess.AS**

- **简介: 该论文属于语音注意力解码任务，旨在提升多说话人环境下的听觉注意力解码效果。通过引入对比PCC损失函数，增强语音包络的区分能力。**

- **链接: [https://arxiv.org/pdf/2601.20542v2](https://arxiv.org/pdf/2601.20542v2)**

> **作者:** Yayun Liang; Yuanming Zhang; Fei Chen; Jing Lu; Zhibin Lin
>
> **摘要:** Recent advances in reconstructing speech envelopes from Electroencephalogram (EEG) signals have enabled continuous auditory attention decoding (AAD) in multi-speaker environments. Most Deep Neural Network (DNN)-based envelope reconstruction models are trained to maximize the Pearson correlation coefficients (PCC) between the attended envelope and the reconstructed envelope (attended PCC). While the difference between the attended PCC and the unattended PCC plays an essential role in auditory attention decoding, existing methods often focus on maximizing the attended PCC. We therefore propose a contrastive PCC loss which represents the difference between the attended PCC and the unattended PCC. The proposed approach is evaluated on three public EEG AAD datasets using four DNN architectures. Across many settings, the proposed objective improves envelope separability and AAD accuracy, while also revealing dataset- and architecture-dependent failure cases.
>
---
#### [replaced 010] Efficient Test-Time Adaptation through Latent Subspace Coefficients Search
- **分类: cs.LG; eess.AS; eess.IV**

- **简介: 该论文属于测试时适应（TTA）任务，解决模型在分布偏移下的适应问题。提出ELaTTA框架，通过优化低维系数向量实现高效单实例适应，降低计算和内存开销。**

- **链接: [https://arxiv.org/pdf/2510.11068v2](https://arxiv.org/pdf/2510.11068v2)**

> **作者:** Xinyu Luo; Jie Liu; Kecheng Chen; Junyi Yang; Bo Ding; Arindam Basu; Haoliang Li
>
> **备注:** Under review
>
> **摘要:** Real-world deployment often exposes models to distribution shifts, making test-time adaptation (TTA) critical for robustness. Yet most TTA methods are unfriendly to edge deployment, as they rely on backpropagation, activation buffering, or test-time mini-batches, leading to high latency and memory overhead. We propose $\textbf{ELaTTA}$ ($\textit{Efficient Latent Test-Time Adaptation}$), a gradient-free framework for single-instance TTA under strict on-device constraints. ELaTTA freezes model weights and adapts each test sample by optimizing a low-dimensional coefficient vector in a source-induced principal latent subspace, pre-computed offline via truncated SVD and stored with negligible overhead. At inference, ELaTTA encourages prediction confidence by optimizing the $k$-D coefficients with CMA-ES, effectively optimizing a Gaussian-smoothed objective and improving stability near decision boundaries. Across six benchmarks and multiple architectures, ELaTTA achieves state-of-the-art accuracy under both strict and continual single-instance protocols, while reducing compute by up to $\textit{63$\times$}$ and peak memory by $\textit{11$\times$}$. We further demonstrate on-device deployment on a ZYNQ-7020 platform. Code will be released upon acceptance.
>
---
#### [replaced 011] MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出MotionBeat，解决音乐与运动对齐的表示学习问题。通过新损失函数和架构设计，提升音乐到舞蹈生成及多种任务性能。**

- **链接: [https://arxiv.org/pdf/2510.13244v2](https://arxiv.org/pdf/2510.13244v2)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 5 pages, 1 figure, accepted by ICASSP 2026. demo page: https://motionbeat2025.github.io/
>
> **摘要:** Music is both an auditory and an embodied phenomenon, closely linked to human motion and naturally expressed through dance. However, most existing audio representations neglect this embodied dimension, limiting their ability to capture rhythmic and structural cues that drive movement. We propose MotionBeat, a framework for motion-aligned music representation learning. MotionBeat is trained with two newly proposed objectives: the Embodied Contrastive Loss (ECL), an enhanced InfoNCE formulation with tempo-aware and beat-jitter negatives to achieve fine-grained rhythmic discrimination, and the Structural Rhythm Alignment Loss (SRAL), which ensures rhythm consistency by aligning music accents with corresponding motion events. Architecturally, MotionBeat introduces bar-equivariant phase rotations to capture cyclic rhythmic patterns and contact-guided attention to emphasize motion events synchronized with musical accents. Experiments show that MotionBeat outperforms state-of-the-art audio encoders in music-to-dance generation and transfers effectively to beat tracking, music tagging, genre and instrument classification, emotion recognition, and audio-visual retrieval. Our project demo page: https://motionbeat2025.github.io/.
>
---
#### [replaced 012] Do Foundational Audio Encoders Understand Music Structure?
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐信息检索任务，研究预训练音频编码器在音乐结构分析中的表现，探讨其学习方法、数据和上下文长度对性能的影响。**

- **链接: [https://arxiv.org/pdf/2512.17209v2](https://arxiv.org/pdf/2512.17209v2)**

> **作者:** Keisuke Toyama; Zhi Zhong; Akira Takahashi; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** In music information retrieval (MIR) research, the use of pretrained foundational audio encoders (FAEs) has recently become a trend. FAEs pretrained on large amounts of music and audio data have been shown to improve performance on MIR tasks such as music tagging and automatic music transcription. However, their use for music structure analysis (MSA) remains underexplored: only a small subset of FAEs has been examined for MSA, and the impact of factors such as learning methods, training data, and model context length on MSA performance remains unclear. In this study, we conduct comprehensive experiments on 11 types of FAEs to investigate how these factors affect MSA performance. Our results demonstrate that FAEs using self-supervised learning with masked language modeling on music data are particularly effective for MSA. These findings pave the way for future research in FAE and MSA.
>
---
#### [replaced 013] Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 该论文属于文本到音频生成任务，旨在解决模型在推理时无意生成训练数据的问题。通过引入抗记忆引导技术，有效减少数据复制，同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2509.14934v2](https://arxiv.org/pdf/2509.14934v2)**

> **作者:** Francisco Messina; Francesca Ronchini; Luca Comanducci; Paolo Bestagini; Fabio Antonacci
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** A persistent challenge in generative audio models is data replication, where the model unintentionally generates parts of its training data during inference. In this work, we address this issue in text-to-audio diffusion models by exploring the use of anti-memorization strategies. We adopt Anti-Memorization Guidance (AMG), a technique that modifies the sampling process of pre-trained diffusion models to discourage memorization. Our study explores three types of guidance within AMG, each designed to reduce replication while preserving generation quality. We use Stable Audio Open as our backbone, leveraging its fully open-source architecture and training dataset. Our comprehensive experimental analysis suggests that AMG significantly mitigates memorization in diffusion-based text-to-audio generation without compromising audio fidelity or semantic alignment.
>
---
#### [replaced 014] Quantitative Measures for Passive Sonar Texture Analysis
- **分类: eess.AS**

- **简介: 该论文属于被动声呐分类任务，旨在解决CNN在统计纹理信号上的性能不足。通过提出量化指标，验证并改进模型对纹理信息的处理。**

- **链接: [https://arxiv.org/pdf/2504.14843v2](https://arxiv.org/pdf/2504.14843v2)**

> **作者:** Jarin Ritu; Alexandra Van Dine; Joshua Peeples
>
> **备注:** 8 pages, 2 figures. This paper has been accepted to the 2026 SPIE Defense + Security Conference: Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications IV
>
> **摘要:** Passive sonar signals contain complex characteristics often arising from environmental noise, vessel machinery, and propagation effects. While convolutional neural networks (CNNs) perform well on passive sonar classification tasks, they can struggle with statistical variations that occur in the data. To investigate this limitation, synthetic underwater acoustic datasets are generated that centered on amplitude and period variations. Two metrics are proposed to quantify and validate these characteristics in the context of statistical and structural texture for passive sonar. These measures are applied to real-world passive sonar datasets to assess texture information in the signals and correlate the performances of the models. Results show that CNNs underperform on statistically textured signals, but incorporating explicit statistical texture modeling yields consistent improvements. These findings highlight the importance of quantifying texture information for passive sonar classification.
>
---
#### [replaced 015] CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出CASTELLA，一个用于音频时刻检索（AMR）的大规模标注数据集，解决现有数据集小且不真实的问题。通过构建更大数据集并建立基线模型，提升AMR性能。**

- **链接: [https://arxiv.org/pdf/2511.15131v2](https://arxiv.org/pdf/2511.15131v2)**

> **作者:** Hokuto Munakata; Takehiro Imamura; Taichi Nishimura; Tatsuya Komatsu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** We introduce CASTELLA, a human-annotated audio benchmark for the task of audio moment retrieval (AMR). Although AMR has various useful potential applications, there is still no established benchmark with real-world data. The initial study of AMR trained the models solely on synthetic datasets. Moreover, the evaluation is based on an annotated dataset of fewer than 100 samples. This resulted in less reliable reported performance. To ensure performance for applications in real-world environments, we present CASTELLA, a large-scale manually annotated AMR dataset. CASTELLA consists of 1009, 213, and 640 audio recordings for training, validation, and test splits, respectively, which is 24 times larger than the previous dataset. We also establish a baseline model for AMR using CASTELLA. Our experiments demonstrate that a model fine-tuned on CASTELLA after pre-training on the synthetic data outperformed a model trained solely on the synthetic data by 10.4 points in Recall1@0.7. CASTELLA is publicly available in https://h-munakata.github.io/CASTELLA-demo/.
>
---
#### [replaced 016] Learning What To Hear: Boosting Sound-Source Association For Robust Audiovisual Instance Segmentation
- **分类: eess.AS; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于音频视觉实例分割任务，旨在解决视觉偏差问题。通过引入声音中心的查询生成和有序计数损失，提升对声音源的定位与跟踪精度。**

- **链接: [https://arxiv.org/pdf/2509.22740v2](https://arxiv.org/pdf/2509.22740v2)**

> **作者:** Jinbae Seo; Hyeongjun Kwon; Kwonyoung Kim; Jiyoung Lee; Kwanghoon Sohn
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Audiovisual instance segmentation (AVIS) requires accurately localizing and tracking sounding objects throughout video sequences. Existing methods suffer from visual bias stemming from two fundamental issues: uniform additive fusion prevents queries from specializing to different sound sources, while visual-only training objectives allow queries to converge to arbitrary salient objects. We propose Audio-Centric Query Generation using cross-attention, enabling each query to selectively attend to distinct sound sources and carry sound-specific priors into visual decoding. Additionally, we introduce Sound-Aware Ordinal Counting (SAOC) loss that explicitly supervises sounding object numbers through ordinal regression with monotonic consistency constraints, preventing visual-only convergence during training. Experiments on AVISeg benchmark demonstrate consistent improvements: +1.64 mAP, +0.6 HOTA, and +2.06 FSLA, validating that query specialization and explicit counting supervision are crucial for accurate audiovisual instance segmentation.
>
---
#### [replaced 017] Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving
- **分类: eess.AS; cs.MM; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决VLA模型无法实时接收用户意图的问题。通过融合音频指令与视觉信息，提出EchoVLA模型，提升驾驶决策的准确性和情感适应性。**

- **链接: [https://arxiv.org/pdf/2601.12142v3](https://arxiv.org/pdf/2601.12142v3)**

> **作者:** Ziang Guo; Feng Yang; Xuefeng Zhang; Jiaqi Guo; Kun Zhao; Yixiao Zhou; Peng Lu; Sifa Zheng; Zufeng Zhang
>
> **备注:** Accepted by IV
>
> **摘要:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.
>
---
#### [replaced 018] SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SPADE框架，用于高效的大语言模型文本转语音（LLM-TTS）任务。针对模型参数大、延迟高的问题，通过结构化剪枝和自适应知识蒸馏，在保持语音质量的同时提升效率。**

- **链接: [https://arxiv.org/pdf/2509.20802v3](https://arxiv.org/pdf/2509.20802v3)**

> **作者:** Tan Dat Nguyen; Jaehun Kim; Ji-Hoon Kim; Shukjae Choi; Youshin Lim; Joon Son Chung
>
> **备注:** ICASSP 2026
>
> **摘要:** The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
>
---
#### [replaced 019] A conversational gesture synthesis system based on emotions and semantics
- **分类: cs.HC; cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于手势生成任务，旨在解决数字人自然生成与文本/语音输入匹配的肢体动作问题。提出DeepGesture框架，结合语义和情感信息生成更逼真手势。**

- **链接: [https://arxiv.org/pdf/2507.03147v3](https://arxiv.org/pdf/2507.03147v3)**

> **作者:** Thanh Hoang-Minh
>
> **摘要:** Along with the explosion of large language models, improvements in speech synthesis, advancements in hardware, and the evolution of computer graphics, the current bottleneck in creating digital humans lies in generating character movements that correspond naturally to text or speech inputs. In this work, we present DeepGesture, a diffusion-based gesture synthesis framework for generating expressive co-speech gestures conditioned on multimodal signals - text, speech, emotion, and seed motion. Built upon the DiffuseStyleGesture model, DeepGesture introduces novel architectural enhancements that improve semantic alignment and emotional expressiveness in generated gestures. Specifically, we integrate fast text transcriptions as semantic conditioning and implement emotion-guided classifier-free diffusion to support controllable gesture generation across affective states. To visualize results, we implement a full rendering pipeline in Unity based on BVH output from the model. Evaluation on the ZeroEGGS dataset shows that DeepGesture produces gestures with improved human-likeness and contextual appropriateness. Our system supports interpolation between emotional states and demonstrates generalization to out-of-distribution speech, including synthetic voices - marking a step forward toward fully multimodal, emotionally aware digital humans.
>
---
#### [replaced 020] End-to-end audio-visual learning for cochlear implant sound coding simulations in noisy environments
- **分类: eess.AS; cs.AI; cs.SD; eess.IV**

- **简介: 该论文属于语音增强任务，旨在解决嘈杂环境中人工耳蜗用户听觉困难的问题。通过融合视听信息，提出AVSE-ECS系统，提升语音可懂度和信号质量。**

- **链接: [https://arxiv.org/pdf/2508.13576v2](https://arxiv.org/pdf/2508.13576v2)**

> **作者:** Meng-Ping Lin; Enoch Hsin-Ho Huang; Shao-Yi Chien; Yu Tsao
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** The cochlear implant (CI) is a successful biomedical device that enables individuals with severe-to-profound hearing loss to perceive sound through electrical stimulation, yet listening in noise remains challenging. Recent deep learning advances offer promising potential for CI sound coding by integrating visual cues. In this study, an audio-visual speech enhancement (AVSE) module is integrated with the ElectrodeNet-CS (ECS) model to form the end-to-end CI system, AVSE-ECS. Simulations show that the AVSE-ECS system with joint training achieves high objective speech intelligibility and improves the signal-to-error ratio (SER) by 7.4666 dB compared to the advanced combination encoder (ACE) strategy. These findings underscore the potential of AVSE-based CI sound coding.
>
---
#### [replaced 021] AudioEval: Automatic Dual-Perspective and Multi-Dimensional Evaluation of Text-to-Audio-Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到音频生成的评估任务，旨在解决自动评估困难的问题。提出AudioEval数据集和Qwen-DisQA模型，实现多维、双视角的自动评估。**

- **链接: [https://arxiv.org/pdf/2510.14570v2](https://arxiv.org/pdf/2510.14570v2)**

> **作者:** Hui Wang; Jinghua Zhao; Junyang Cheng; Cheng Liu; Yuhang Jia; Haoqin Sun; Jiaming Zhou; Yong Qin
>
> **摘要:** Text-to-audio (TTA) generation is advancing rapidly, but evaluation remains challenging because human listening studies are expensive and existing automatic metrics capture only limited aspects of perceptual quality. We introduce AudioEval, a large-scale TTA evaluation dataset with 4,200 generated audio samples (11.7 hours) from 24 systems and 126,000 ratings collected from both experts and non-experts across five dimensions: enjoyment, usefulness, complexity, quality, and text alignment. Using AudioEval, we benchmark diverse automatic evaluators to compare perspective- and dimension-level differences across model families. We also propose Qwen-DisQA as a strong reference baseline: it jointly processes prompts and generated audio to predict multi-dimensional ratings for both annotator groups, modeling rater disagreement via distributional prediction and achieving strong performance. We will release AudioEval to support future research in TTA evaluation.
>
---
