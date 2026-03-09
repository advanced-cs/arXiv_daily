# 音频 cs.SD;  eess.AS

- **最新发布 18 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Koopman Regularized Deep Speech Disentanglement for Speaker Verification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决说话人验证中的内容与说话人特征分离问题。提出DKSD-AE模型，结合Koopman算子和实例归一化，实现高效、无监督的说话人表示学习。**

- **链接: [https://arxiv.org/pdf/2603.05577](https://arxiv.org/pdf/2603.05577)**

> **作者:** Nikos Chazaridis; Mohammad Belal; Rafael Mestre; Timothy J. Norman; Christine Evers
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Human speech contains both linguistic content and speaker dependent characteristics making speaker verification a key technology in identity critical applications. Modern deep learning speaker verification systems aim to learn speaker representations that are invariant to semantic content and nuisance factors such as ambient noise. However, many existing approaches depend on labelled data, textual supervision or large pretrained models as feature extractors, limiting scalability and practical deployment, raising sustainability concerns. We propose Deep Koopman Speech Disentanglement Autoencoder (DKSD-AE), a structured autoencoder that combines a novel multi-step Koopman operator learning module with instance normalization to disentangle speaker and content dynamics. Quantitative experiments across multiple datasets demonstrate that DKSD-AE achieves improved or competitive speaker verification performance compared to state-of-the-art baselines while maintaining high content EER, confirming effective disentanglement. These results are obtained with substantially fewer parameters and without textual supervision. Moreover, performance remains stable under increased evaluation scale, highlighting representation robustness and generalization. Our findings suggest that Koopman-based temporal modelling, when combined with instance normalization, provides an efficient and principled solution for speaker-focused representation learning.
>
---
#### [new 002] StreamVoiceAnon+: Emotion-Preserving Streaming Speaker Anonymization via Frame-Level Acoustic Distillation
- **分类: eess.AS; cs.AI; eess.SP**

- **简介: 该论文属于语音匿名化任务，旨在保留情感信息的同时实现流式语音匿名。通过监督微调和帧级情感蒸馏，提升情感保持率并确保隐私。**

- **链接: [https://arxiv.org/pdf/2603.06079](https://arxiv.org/pdf/2603.06079)**

> **作者:** Nikita Kuzmin; Kong Aik Lee; Eng Siong Chng
>
> **摘要:** We address the challenge of preserving emotional content in streaming speaker anonymization (SA). Neural audio codec language models trained for audio continuation tend to degrade source emotion: content tokens discard emotional information, and the model defaults to dominant acoustic patterns rather than preserving paralinguistic attributes. We propose supervised finetuning with neutral-emotion utterance pairs from the same speaker, combined with frame-level emotion distillation on acoustic token hidden states. All modifications are confined to finetuning, which takes less than 2 hours on 4 GPUs and adds zero inference latency overhead, while maintaining a competitive 180ms streaming latency. On the VoicePrivacy 2024 protocol, our approach achieves a 49.2% UAR (emotion preservation) with 5.77% WER (intelligibility), a +24% relative UAR improvement over the baseline (39.7%->49.2%) and +10% over the emotion-prompt variant (44.6% UAR), while maintaining strong privacy (EER 49.0%). Demo and code are available: this https URL
>
---
#### [new 003] Reconstruct! Don't Encode: Self-Supervised Representation Reconstruction Loss for High-Intelligibility and Low-Latency Streaming Neural Audio Codec
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于音频编码任务，解决语音可懂性与低延迟问题。提出自监督表示重建损失，提升编码器性能，实现高可懂性与实时部署。**

- **链接: [https://arxiv.org/pdf/2603.05887](https://arxiv.org/pdf/2603.05887)**

> **作者:** Junhyeok Lee; Xiluo He; Jihwan Lee; Helin Wang; Shrikanth Narayanan; Thomas Thebaud; Laureano Moro-Velazquez; Jesús Villalba; Najim Dehak
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Neural audio codecs optimized for mel-spectrogram reconstruction often fail to preserve intelligibility. While semantic encoder distillation improves encoded representations, it does not guarantee content preservation in reconstructed speech. In this work, we demonstrate that self-supervised representation reconstruction (SSRR) loss fundamentally improves codec training and performance. First, SSRR significantly accelerates convergence, enabling competitive results using only a single GPU. Second, it enhances intelligibility by reconstructing distilled self-supervised representations from codec outputs. Third, SSRR enables high intelligibility without additional lookahead in streaming Transformer-based codecs, allowing a zero-lookahead architecture for real-time deployment. As a result, our JHCodec achieves state-of-the-art performance while maintaining minimal latency and reduced training cost. We open-source the full implementation, training pipeline, and demo on Github this https URL.
>
---
#### [new 004] Cross-linguistic Prosodic Analysis of Autistic and Non-autistic Child Speech in Finnish, French and Slovak
- **分类: eess.AS**

- **简介: 该论文属于语音分析任务，研究自闭症与非自闭症儿童在芬兰语、法语和斯洛伐克语中的韵律差异，通过声学特征分析揭示跨语言的共性与语言特异性特征。**

- **链接: [https://arxiv.org/pdf/2603.06332](https://arxiv.org/pdf/2603.06332)**

> **作者:** Ida-Lotta Myllylä; Sofoklis Kakouros
>
> **备注:** Accepted to Speech Prosody 2026
>
> **摘要:** Prosodic differences in autism are well-documented, but cross-linguistic evidence remains limited. This study investigates prosody in autism across a multilingual corpus of Finnish, French, and Slovak speakers. 88 acoustic features from over 5,000 inter-pausal units were extracted, and data were reduced via Principal Component Analysis (PCA) and analyzed using Linear Mixed-Effects Models (LMMs). Cross-linguistically, autistic speakers exhibited increased general intensity variability and a clearer, less breathy voice quality (higher Harmonics-to-Noise Ratio and alpha ratio), alongside reduced temporal intensity dynamics and lower central f0. Monolingual analyses revealed language-specific nuances: Slovak results aligned with cross-linguistic f0 patterns but diverged on voice quality, while Finnish results mirrored the broader voice quality findings. These results emphasize including voice quality and intensity dynamics in the study of possible language-independent markers of autism, alongside traditional pitch measures. The findings challenge deficiency-based models, suggesting instead a complex, acoustically distinct prosodic profile across languages.
>
---
#### [new 005] Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于音频深度伪造检测任务，探讨紧凑型SSL模型的有效性。研究对比了HuBERT和WavLM，发现预训练方式比模型规模更重要。**

- **链接: [https://arxiv.org/pdf/2603.06164](https://arxiv.org/pdf/2603.06164)**

> **作者:** Ajinkya Kulkarni; Sandipana Dowerah; Atharva Kulkarni; Tanel Alumäe; Mathew Magimai Doss
>
> **备注:** Submitted to Interspeech 2026, 4 pages, 2 figures
>
> **摘要:** Self-supervised learning (SSL) underpins modern audio deepfake detection, yet most prior work centers on a single large wav2vec2-XLSR backbone, leaving compact under studied. We present RAPTOR, Representation Aware Pairwise-gated Transformer for Out-of-domain Recognition a controlled study of compact SSL backbones from the HuBERT and WavLM within a unified pairwise-gated fusion detector, evaluated across 14 cross-domain benchmarks. We show that multilingual HuBERT pre-training is the primary driver of cross-domain robustness, enabling 100M models to match larger and commercial systems. Beyond EER, we introduce a test-time augmentation protocol with perturbation-based aleatoric uncertainty to expose calibration differences invisible to standard metrics: WavLM variants exhibit overconfident miscalibration under perturbation, whereas iterative mHuBERT remains stable. These findings indicate that SSL pre-training trajectory, not model scale, drives reliable audio deepfake detection.
>
---
#### [new 006] RAMoEA-QA: Hierarchical Specialization for Robust Respiratory Audio Question Answering
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于呼吸音频问答任务，解决多源音频与多样化查询的挑战。提出RAMoEA-QA模型，通过分层专业化提升准确率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.06542](https://arxiv.org/pdf/2603.06542)**

> **作者:** Gaia A. Bertolino; Yuwei Zhang; Tong Xia; Domenico Talia; Cecilia Mascolo
>
> **摘要:** Conversational generative AI is rapidly entering healthcare, where general-purpose models must integrate heterogeneous patient signals and support diverse interaction styles while producing clinically meaningful outputs. In respiratory care, non-invasive audio, such as recordings captured via mobile microphones, enables scalable screening and longitudinal monitoring, but the heterogeneity challenge is particularly acute: recordings vary widely across devices, environments, and acquisition protocols, and questions span multiple intents and question formats. Existing biomedical audio-language QA systems are typically monolithic, without any specialization mechanisms for tackling diverse respiratory corpora and query intents. They are also only validated in limited settings, leaving it unclear how reliably they handle the shifts encountered in real-world settings. To address these limitations, we introduce RAMoEA-QA, a hierarchically routed generative model for respiratory audio question answering that unifies multiple question types and supports both discrete and continuous targets within a single multimodal system. RAMoEA-QA applies two-stage conditional specialization: an Audio Mixture-of-Experts routes each recording to a suitable pre-trained audio encoder, and a Language Mixture-of-Adapters selects a LoRA adapter on a shared frozen LLM to match the query intent and answer format. By specializing both acoustic representations and generation behaviour per example, RAMoEA-QA consistently outperforms strong baselines and routing ablations with minimal parameter overhead, improving in-domain test accuracy to 0.72 (vs. 0.61 and 0.67 for state-of-the-art baselines) and exhibiting the strongest generalization for diagnosis under domain, modality, and task shifts.
>
---
#### [new 007] Classification of Autistic and Non-Autistic Children's Speech: A Cross-Linguistic Study in Finnish, French, and Slovak
- **分类: eess.AS**

- **简介: 该论文属于语音分类任务，旨在识别自闭症与非自闭症儿童的言语特征。研究分析了芬兰语、法语和斯洛伐克语数据，探索跨语言的语音线索差异。**

- **链接: [https://arxiv.org/pdf/2603.06327](https://arxiv.org/pdf/2603.06327)**

> **作者:** Sofoklis Kakouros; Ida-Lotta Myllylä
>
> **备注:** Accepted to Speech Prosody 2026
>
> **摘要:** We present a cross-linguistic study of speech in autistic and non-autistic children speaking Finnish, French, and Slovak. We combine supervised classification with within-language and cross-corpus transfer experiments to evaluate classification performance within and across languages and to probe which acoustic cues are language-specific versus language-general. Using a large set of acoustic-prosodic features, we implement speaker-level classification benchmarks as an analytical tool rather than to seek state-of-the-art performance. Within-language models, evaluated with speaker-level cross-validation, yielded heterogeneous results. The Finnish model performed best (Accuracy 0.84, F1 0.88), followed by Slovak (Accuracy 0.63, F1 0.68) and French (Accuracy 0.68, F1 0.56). We then tested cross-language generalization. A model trained on all pooled corpora reached an overall Accuracy of 0.61 and F1 0.68. Leave-one-corpus-out experiments, which test transfer to an unseen language, showed moderate success when testing on Slovak (F1 0.70) and Finnish (F1 0.78), but poor transfer to French (F1 0.42). Feature-importance analyses across languages highlighted partially shared, but not fully language-invariant, acoustic markers of autism. These findings suggest that some autism-related speech cues generalize across typologically distinct languages, but robust cross-linguistic classifiers will likely require language-aware modeling and more homogeneous recording conditions.
>
---
#### [new 008] Prosodic Boundary-Aware Streaming Generation for LLM-Based TTS with Streaming Text Input
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，解决流式文本输入下的韵律不自然和长文本崩溃问题。通过引入韵律边界感知的微调策略，提升生成质量与连贯性。**

- **链接: [https://arxiv.org/pdf/2603.06444](https://arxiv.org/pdf/2603.06444)**

> **作者:** Changsong Liu; Tianrui Wang; Ye Ni; Yizhou Peng; Eng Siong Chng
>
> **摘要:** Streaming TTS that receives streaming text is essential for interactive systems, yet this scheme faces two major challenges: unnatural prosody due to missing lookahead and long-form collapse due to unbounded context. We propose a prosodic-boundary-aware post-training strategy, adapting a pretrained LLM-based TTS model using weakly time-aligned data. Specifically, the model is adapted to learn early stopping at specified content boundaries when provided with limited future text. During inference, a sliding-window prompt carries forward previous text and speech tokens, ensuring bounded context and seamless concatenation. Evaluations show our method outperforms CosyVoice-Style interleaved baseline in both short and long-form scenarios. In long-text synthesis, especially, it achieves a 66.2% absolute reduction in word error rate (from 71.0% to 4.8%) and increases speaker and emotion similarity by 16.1% and 1.5% relatively, offering a robust solution for streaming TTS with incremental text.
>
---
#### [new 009] Whisper-CD: Accurate Long-Form Speech Recognition using Multi-Negative Contrastive Decoding
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音识别任务，解决长文本识别中的错误累积问题。提出Whisper-CD框架，通过对比解码减少错误，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.06193](https://arxiv.org/pdf/2603.06193)**

> **作者:** Hoseong Ahn; Jeongyun Chae; Yoonji Park; Kyuhong Shim
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Long-form speech recognition with large encoder-decoder models such as Whisper often exhibit hallucinations, repetition loops, and content omissions. These errors can accumulate and be further amplified when the previous segment's transcription is used as decoding context. We propose Whisper-CD, a training-free contrastive decoding framework that contrasts clean-audio logits against negative logits computed from three acoustically motivated perturbations: Gaussian noise injection, silence signal, and audio temporal shift. We aggregate these negatives via the log-sum-exp operator, building a unified multi-negative objective for token-by-token decoding. Across five English long-form benchmarks, Whisper-CD reduces WER by up to 24.3pp on CORAAL and shows 48% faster token generation throughput than beam search. Because Whisper-CD operates purely at inference time, it can be applied as a drop-in replacement to already-deployed Whisper systems without retraining.
>
---
#### [new 010] How Well Do Current Speech Deepfake Detection Methods Generalize to the Real World?
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有方法在真实环境中的泛化能力问题。通过构建多语言数据集ML-ITW并评估多种检测方法，发现其性能显著下降。**

- **链接: [https://arxiv.org/pdf/2603.05852](https://arxiv.org/pdf/2603.05852)**

> **作者:** Daixian Li; Jun Xue; Yanzhen Ren; Zhuolin Yi; Yihuan Huang; Guanxiang Feng; Yi Chai
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advances in speech synthesis and voice conversion have greatly improved the naturalness and authenticity of generated audio. Meanwhile, evolving encoding, compression, and transmission mechanisms on social media platforms further obscure deepfake artifacts. These factors complicate reliable detection in real-world environments, underscoring the need for representative evaluation benchmarks. To this end, we introduce ML-ITW (Multilingual In-The-Wild), a multilingual dataset covering 14 languages, seven major platforms, and 180 public figures, totaling 28.39 hours of audio. We evaluate three detection paradigms: end-to-end neural models, self-supervised feature-based (SSL) methods, and audio large language models (Audio LLMs). Experimental results reveal significant performance degradation across diverse languages and real-world acoustic conditions, highlighting the limited generalization ability of existing detectors in practical scenarios. The ML-ITW dataset is publicly available.
>
---
#### [new 011] Which Data Matter? Embedding-Based Data Selection for Speech Recognition
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决专用模型在广泛数据中训练时的性能问题。通过嵌入表示选择相关数据子集，提升目标领域的识别效果。**

- **链接: [https://arxiv.org/pdf/2603.05819](https://arxiv.org/pdf/2603.05819)**

> **作者:** Zakaria Aldeneh; Skyler Seto; Maureen de Seyssel; Jie Chi; Zijin Gu; Takuya Higuchi; Jee-weon Jung; Shinji Watanabe; David Grangier; Barry-John Theobald; Tatiana Likhomanenko
>
> **摘要:** Modern ASR systems are typically trained on large-scale pseudo-labeled, in-the-wild data spanning multiple domains. While such heterogeneous data benefit generalist models designed for broad deployment, they pose challenges for specialist models targeting specific domains: specialist models lack the capacity to learn from all available data, and one must pay closer attention to addressing the mismatch between training and test conditions. In this work, we study targeted data selection as a strategy to address these challenges, selecting relevant subsets from 100k hours of in-the-wild training data to optimize performance on target domains. We represent speech samples using embeddings that capture complementary characteristic--speaker attributes, phonetic content, and semantic meaning--and analyze how relevance and diversity along these axes when performing data selection affect downstream ASR performance. Our experiments with CTC-based Conformer models show that training on a strategically selected 5% subset can exceed the performance of models trained on the full dataset by up to 36.8% relative WER reduction on target domains.
>
---
#### [new 012] Activation Steering for Accent Adaptation in Speech Foundation Models
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决 accents 引起的识别错误问题。通过分析激活空间，识别并控制accent信息，实现参数无关的accent适应。**

- **链接: [https://arxiv.org/pdf/2603.05813](https://arxiv.org/pdf/2603.05813)**

> **作者:** Jinuo Sun; Yang Xiao; Sung Kyun Chung; Qiuchi Hu; Gongping Huang; Eun-Jung Holden; Ting Dang
>
> **备注:** Submitted to Interspeech. 5 pages
>
> **摘要:** Accent variability remains a major errors in automatic speech recognition, yet most adaptation methods rely on parameter fine-tuning without understanding where accent information is encoded. We treat accent variation as an interpretable subspace in hidden representations and investigate whether it can be identified and controlled directly in activation space. We extract layer-wise encoder activations and estimate mean-shift directions capturing accent-induced representation shifts. By injecting these directions into individual layers and measuring how they align accented and standard embeddings, we derive a layer-wise accent sensitivity profile, revealing that accent information concentrates in a narrow band of middle encoder layers. Leveraging this structure, we further introduce parameter-free accent steering that modifies representations during inference without updating model weights. Experiments across eight accents show consistent word error rate reductions.
>
---
#### [new 013] Activation Steering for Accent-Neutralized Zero-Shot Text-To-Speech
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决零样本TTS中音调与口音混杂的问题。通过提取激活向量，在推理时调整模型输出，实现口音中性化且保留原声特征。**

- **链接: [https://arxiv.org/pdf/2603.05977](https://arxiv.org/pdf/2603.05977)**

> **作者:** Mu Yang; John H. L. Hansen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Zero-shot Text-to-Speech (TTS) models can generate speech that captures both the voice timbre and accent of a reference speaker. However, disentangling these attributes remains challenging, as the output often inherits both the accent and timbre from the reference. In this study, we introduce a novel, post-hoc, and training-free approach to neutralize accent while preserving the speaker's original timbre, utilizing inference-time activation steering. We first extract layer-specific "steering vectors" offline, which are derived from the internal activation differences within the TTS model between accented and native speech. During inference, the steering vectors are applied to guide the model to produce accent-neutralized, timbre-preserving speech. Empirical results demonstrate that the proposed steering vectors effectively mitigate the output accent and exhibit strong generalizability to unseen accented speakers, offering a practical solution for accent-free voice cloning.
>
---
#### [new 014] Doctor or Patient? Synergizing Diarization and ASR for Code-Switched Hinglish Medical Conditions Extraction
- **分类: eess.AS**

- **简介: 该论文属于医疗对话信息提取任务，旨在解决代码切换的Hinglish医学对话中患者病情提取问题。通过结合语音分离和ASR技术，提出EEND-VC方法并优化模型，取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2603.06373](https://arxiv.org/pdf/2603.06373)**

> **作者:** Séverin Baroudi; Yanis Labrak; Shashi Kumar; Joonas Kalda; Sergio Burdisso; Pawel Cyrta; Juan Ignacio Alvarez-Trejos; Petr Motlicek; Hervé Bredin; Ricard Marxer
>
> **备注:** Submitted for review at Interspeech 2026
>
> **摘要:** Extracting patient medical conditions from code-switched clinical spoken dialogues is challenging due to rapid turn-taking and highly overlapped speech. We present a robust system evaluated on the DISPLACE-M dataset of real-world Hinglish medical conversations. We propose an End-to-End Neural Diarization with Vector Clustering approach (EEND-VC) to accurately resolve dense and speaker overlaps in Doctor-Patient Conversations (DoPaCo). For transcription, we adapt a Qwen3 ASR model via domain-specific fine-tuning, Devanagari script normalization, and dialogue-level LLM error correction, achieving an 18.59% tcpWER. We benchmark open and proprietary LLMs on medical condition extraction, comparing our text-based cascade system against a multimodal End-to-End (E2E) audio framework. While proprietary E2E models set the performance ceiling, our open cascaded architecture is highly competitive, as it achieved first place out of 25 participants in the DISPLACE-M challenge. All implementations are publicly released.
>
---
#### [new 015] Continual Adaptation for Pacific Indigenous Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决低资源太平洋原住民语言模型适应问题。研究分析了数据量和语言特征的影响，评估了多种适配策略，指出需改进持续学习方法。**

- **链接: [https://arxiv.org/pdf/2603.06310](https://arxiv.org/pdf/2603.06310)**

> **作者:** Yang Xiao; Aso Mahmudi; Nick Thieberger; Eliathamby Ambikairajah; Eun-Jung Holden; Ting Dang
>
> **备注:** Submitted to Interspeech
>
> **摘要:** Speech foundation models struggle with low-resource Pacific Indigenous languages because of severe data scarcity. Furthermore, full fine-tuning risks catastrophic forgetting. To address this gap, we present an empirical study adapting models to real-world Pacific datasets. We investigate how data volume and linguistic features affect adaptation success. Specifically, we evaluate strategies including Full Fine-Tuning and Low-Rank Adaptation (LoRA). Additionally, we analyze a continual learning framework for sequentially acquiring multiple languages. We demonstrate that adapting to these distant languages causes severe internal representational drift. Consequently, these models face a strict plasticity and stability dilemma. While LoRA adapts well initially, it suffers from catastrophic forgetting during sequential learning. Ultimately, this study highlights the urgent need for robust adaptation strategies tailored to underrepresented languages.
>
---
#### [new 016] ImKWS: Test-Time Adaptation for Keyword Spotting with Class Imbalance
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决KWS中因类别不平衡导致的模型偏差问题。提出ImKWS方法，在无标签测试数据下提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2603.05821](https://arxiv.org/pdf/2603.05821)**

> **作者:** Hanyu Ding; Yang Xiao; Jiaheng Dong; Ting Dang
>
> **备注:** Submitted to Interspeech
>
> **摘要:** Keyword spotting (KWS) identifies words for voice assistants, but environmental noise frequently reduces accuracy. Standard adaptation fixes this issue and strictly requires original or labeled audio. Test time adaptation (TTA) solves this data constraint using only unlabeled test audio. However, current methods fail to handle the severe imbalance between rare keywords and frequent background sounds. Consequently, standard entropy minimization (EM) becomes overconfident and heavily biased toward the frequent background class. To overcome this problem, we propose a TTA method named ImKWS. Our approach splits the entropy process into a reward branch and a penalty branch with separate update strengths. Furthermore, we enforce consistency across multiple audio transformations to ensure stable model updates. Experiments on the Google Speech Commands dataset indicate ImKWS achieves reliable adaptation in realistic imbalanced scenarios. The code is available on GitHub.
>
---
#### [new 017] TempoSyncDiff: Distilled Temporally-Consistent Diffusion for Low-Latency Audio-Driven Talking Head Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于音频驱动的说话头生成任务，旨在解决高延迟、时间不稳定和音视不同步问题。提出TempoSyncDiff框架，通过教师-学生蒸馏提升生成效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.06057](https://arxiv.org/pdf/2603.06057)**

> **作者:** Soumya Mazumdar; Vineet Kumar Rakesh
>
> **摘要:** Diffusion models have recently advanced photorealistic human synthesis, although practical talking-head generation (THG) remains constrained by high inference latency, temporal instability such as flicker and identity drift, and imperfect audio-visual alignment under challenging speech conditions. This paper introduces TempoSyncDiff, a reference-conditioned latent diffusion framework that explores few-step inference for efficient audio-driven talking-head generation. The approach adopts a teacher-student distillation formulation in which a diffusion teacher trained with a standard noise prediction objective guides a lightweight student denoiser capable of operating with significantly fewer inference steps to improve generation stability. The framework incorporates identity anchoring and temporal regularization designed to mitigate identity drift and frame-to-frame flicker during synthesis, while viseme-based audio conditioning provides coarse lip motion control. Experiments on the LRS3 dataset report denoising-stage component-level metrics relative to VAE reconstructions and preliminary latency characterization, including CPU-only and edge computing measurements and feasibility estimates for edge deployment. The results suggest that distilled diffusion models can retain much of the reconstruction behaviour of a stronger teacher while enabling substantially lower latency inference. The study is positioned as an initial step toward practical diffusion-based talking-head generation under constrained computational settings. GitHub: this https URL
>
---
#### [new 018] Omni-C: Compressing Heterogeneous Modalities into a Single Dense Encoder
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出Omni-C，一种统一的多模态编码器，解决多模态系统复杂度高、参数膨胀的问题。通过共享参数和轻量投影头，实现高效多模态学习。**

- **链接: [https://arxiv.org/pdf/2603.05528](https://arxiv.org/pdf/2603.05528)**

> **作者:** Kin Wai Lau; Yasar Abbas Ur Rehman; Lai-Man Po; Pedro Porto Buarque de Gusmão
>
> **摘要:** Recent multimodal systems often rely on separate expert modality encoders which cause linearly scaling complexity and computational overhead with added modalities. While unified Omni-models address this via Mixture-of-Expert (MoE) architectures with specialized experts and routing, they still inflate parameter counts and introduce routing overhead. In this paper, we propose Omni-C (Omni-Compress), a single dense Transformer-based encoder that learns competitive shared representations across heterogeneous modalities--images, audio, and text--through unimodal contrastive pretraining on large-scale unaligned data. By maximizing parameter sharing in the backbone and using lightweight modality-specific projection heads, Omni-C effectively mitigates inter-modality conflicts without requiring MoE, paired supervision, or routing. This design supports efficient deployment on memory-constrained systems via sequential modality processing and low-memory inference, eliminating the need for parallel expert loading or specialized hardware. Experiments show Omni-C achieves performance comparable to expert models in unimodal and cross-model tasks, with modest zero-shot degradation on audio and text that is largely recovered through lightweight linear probing or parameter efficient fine-tuning. The unified architecture substantially reduces inference memory usage compared to multi-encoder baselines, advancing efficient and scalable multimodal learning.
>
---
## 更新

#### [replaced 001] Efficient Emotion and Speaker Adaptation in LLM-Based TTS via Characteristic-Specific Partial Fine-Tuning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，解决LLM-TTS在未见领域中情感和说话人克隆质量下降的问题。通过特征特定的局部微调策略，提升适应效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2501.14273](https://arxiv.org/pdf/2501.14273)**

> **作者:** Tianrui Wang; Meng Ge; Cheng Gong; Chunyu Qiang; Haoyu Wang; Zikang Huang; Yu Jiang; Ye Ni; Yuheng Lu; Xiaobao Wang; Engsiong Chng; Xie Chen; Longbiao Wang; Jianwu Dang
>
> **备注:** 10 pages
>
> **摘要:** While LLM-based TTS models exhibit zero-shot emotion and speaker cloning, their cloning fidelity and pronunciation clarity degrade on unseen domains. Fine-tuning is essential for adaptation, yet uniform approaches overlook specific parameter contributions. Uniform tuning on limited data causes slow training and catastrophic forgetting, leading to degraded pronunciation accuracy. To address this, we propose CSP-FT, a characteristic-specific partial fine-tuning strategy. By dynamically analyzing layer contributions via a weighted sum, we selectively fine-tune only the two layers capturing the most and least emotion and speaker information, maximizing the utility of the former while explicitly strengthening the capacity of the latter. Experiments on a combined corpus of 11 datasets show CSP-FT matches or exceeds the fidelity and intelligibility of full fine-tuning while updating only ~8% of parameters, accelerating training by ~2x, and significantly mitigating catastrophic forgetting.
>
---
#### [replaced 002] Purification Before Fusion: Toward Mask-Free Speech Enhancement for Robust Audio-Visual Speech Recognition
- **分类: eess.AS; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉语音识别任务，旨在解决高噪声下特征融合干扰问题。提出一种无需噪声掩码的端到端框架，通过视频辅助增强音频特征，提升识别鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.12436](https://arxiv.org/pdf/2601.12436)**

> **作者:** Linzhi Wu; Xingyu Zhang; Hao Yuan; Yakun Zhang; Changyan Zheng; Liang Xie; Tiejun Liu; Erwei Yin
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** Audio-visual speech recognition (AVSR) typically improves recognition accuracy in noisy environments by integrating noise-immune visual cues with audio signals. Nevertheless, high-noise audio inputs are prone to introducing adverse interference into the feature fusion process. To mitigate this, recent AVSR methods often adopt mask-based strategies to filter audio noise during feature interaction and fusion, yet such methods risk discarding semantically relevant information alongside noise. In this work, we propose an end-to-end noise-robust AVSR framework coupled with speech enhancement, eliminating the need for explicit noise mask generation. This framework leverages a Conformer-based bottleneck fusion module to implicitly refine noisy audio features with video assistance. By reducing modality redundancy and enhancing inter-modal interactions, our method preserves speech semantic integrity to achieve robust recognition performance. Experimental evaluations on the public LRS3 benchmark suggest that our method outperforms prior advanced mask-based baselines under noisy conditions.
>
---
#### [replaced 003] The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines?
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音大模型与ASR→LLM级联系统的等价性问题，通过测试和分析揭示其行为机制，指出当前语音模型在多数场景下表现类似昂贵的级联系统。**

- **链接: [https://arxiv.org/pdf/2602.17598](https://arxiv.org/pdf/2602.17598)**

> **作者:** Jayadev Billa
>
> **备注:** 10 pages, 6 figures, 7 tables. submitted for review Interspeech 2026
>
> **摘要:** Speech LLMs are widely understood to be better than ASR$\rightarrow$LLM cascades since they have access to the audio directly, and not just the transcript. In this paper, we present an evaluation methodology and a mechanistic interpretation of the observed behavior of speech LLMs. First, we introduce matched-backbone testing which separates out the behavior of the speech LLM from the reasoning capabilities of the underlying LLM. Second, we provide a mechanistic analysis of speech LLMs using logit lens and LEACE and show the literal transcript emerging from the LLM's hidden states and that text representations are causally necessary. We also show that in most deployed use cases, current speech LLMs are expensive cascades, and under noise, they are worse ones, with clean-condition advantages reversing by up to 7.6% at 0dB.
>
---
#### [replaced 004] HVAC-EAR: Eavesdropping Human Speech Using HVAC Systems
- **分类: cs.SD; cs.CR**

- **简介: 该论文提出HVAC-EAR，利用空调系统压力传感器窃听语音。属于语音重建任务，解决低采样率下语音清晰度问题，通过复数模型和频域重构提升可懂度。**

- **链接: [https://arxiv.org/pdf/2510.01082](https://arxiv.org/pdf/2510.01082)**

> **作者:** Tarikul Islam Tamiti; Biraj Joshi; Rida Hasan; Anomadarshi Barua
>
> **摘要:** Pressure sensors are widely integrated into modern Heating, Ventilation and Air Conditioning (HVAC) systems. As they are sensitive to acoustic pressure, they can be a source of eavesdropping. We introduce HVAC-EAR, which reconstructs intelligible speech from low-resolution, noisy pressure data with two key contributions: (i) We achieve intelligible reconstruction from as low as 0.5 kHz sampling rate, surpassing prior work limited to hot word detection, by employing a complex-valued conformer with a Complex Unifed Attention Block to capture phoneme dependencies; (ii) We mitigate transient HVAC noise by reconstructing both magnitude and phase of missing frequencies. For the first time, evaluations on real-world HVAC deployments show significant intelligibility up to 1.2 m distance, raising novel privacy concerns.
>
---
#### [replaced 005] Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出Whisper-RIR-Mega数据集，用于评估ASR在混响环境下的鲁棒性。通过配对干净与混响语音样本，分析不同模型在混响条件下的性能下降情况。**

- **链接: [https://arxiv.org/pdf/2603.02252](https://arxiv.org/pdf/2603.02252)**

> **作者:** Mandip Goswami
>
> **摘要:** We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convolved with a real room impulse response from the RIR-Mega corpus, with stratified splits by reverberation time (RT60) and direct-to-reverberant ratio (DRR). We evaluate five Whisper models (tiny through large-v3) on 1600 test samples and report word error rate (WER) and character error rate (CER) under clean and reverberant conditions. Reverberation consistently degrades performance across all model sizes; the reverb penalty in WER ranges from 0.12 to 1.07 percentage points depending on the model. We release the dataset, evaluation code, and baseline results to support reproducible research on robust ASR.
>
---
#### [replaced 006] PolyBench: A Benchmark for Compositional Reasoning in Polyphonic Audio
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出PolyBench，用于评估音频中的组合推理能力。针对多声部音频中事件共存的推理问题，设计多个评估子集，发现现有模型在该任务上表现不佳。**

- **链接: [https://arxiv.org/pdf/2603.05128](https://arxiv.org/pdf/2603.05128)**

> **作者:** Yuanjian Chen; Yang Xiao; Han Yin; Xubo Liu; Jinjie Huang; Ting Dang
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Large Audio Language Models (LALMs) are increasingly capable of reasoning over audio. However, existing benchmarks provide limited coverage of reasoning in polyphonic audio, where multiple sound events co-occur and induce compositional structure. In this work, we introduce PolyBench, a benchmark designed to evaluate compositional reasoning in polyphonic audio. PolyBench comprises five evaluation subsets covering counting, classification, detection, concurrency, and duration estimation, requiring reasoning over multiple concurrent events and their relations. Evaluation of state-of-the-art LALMs reveals consistent performance degradation in polyphonic audio, indicating a fundamental bottleneck in current LALMs.
>
---
#### [replaced 007] LMU-Based Sequential Learning and Posterior Ensemble Fusion for Cross-Domain Infant Cry Classification
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于跨领域婴儿啼哭分类任务，解决信号短、标注少和领域差异大的问题。提出融合多特征的CNN与LMU模型，并采用后验集成方法提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.02245](https://arxiv.org/pdf/2603.02245)**

> **作者:** Niloofar Jazaeri; Hilmi R. Dajani; Marco Janeczek; Martin Bouchard
>
> **备注:** 7 pages
>
> **摘要:** Decoding infant cry causes remains challenging for healthcare monitoring due to short nonstationary signals, limited annotations, and strong domain shifts across infants and datasets. We propose a compact acoustic framework that fuses MFCC, STFT, and pitch features within a multi-branch CNN encoder and models temporal dynamics using an enhanced Legendre Memory Unit (LMU). Compared to LSTMs, the LMU backbone provides stable sequence modeling with substantially fewer recurrent parameters, supporting efficient deployment. To improve cross-dataset generalization, we introduce calibrated posterior ensemble fusion with entropy-gated weighting to preserve domain-specific expertise while mitigating dataset bias. Experiments on Baby2020 and Baby Crying demonstrate improved macro-F1 under cross-domain evaluation, along with leakageaware splits and real-time feasibility for on-device monitoring.
>
---
#### [replaced 008] The trajectoRIR Database: Room Acoustic Recordings Along a Trajectory of Moving Microphones
- **分类: eess.AS**

- **简介: 本文介绍trajectoRIR数据库，用于声学信号处理任务，解决动态和静态声场数据不足的问题。通过移动麦克风和静止RIR记录，提供多样化的声学数据集。**

- **链接: [https://arxiv.org/pdf/2503.23004](https://arxiv.org/pdf/2503.23004)**

> **作者:** Stefano Damiano; Kathleen MacWilliam; Valerio Lorenzoni; Thomas Dietzen; Toon van Waterschoot
>
> **备注:** 44 pages, 10 figures
>
> **摘要:** Data availability is essential in the development of acoustic signal processing algorithms, especially when it comes to data-driven approaches that demand large and diverse training datasets. For this reason, an increasing number of databases have been published in recent years, including either room impulse responses (RIRs) or audio recordings during motion. In this paper we introduce the trajectoRIR database, an extensive, multi-array collection of both dynamic and stationary acoustic recordings along a controlled trajectory in a room. Specifically, the database contains moving-microphone recordings and stationary RIRs that spatially sample the room acoustics along an L-shaped trajectory. This combination makes trajectoRIR unique and applicable to a wide range of tasks, including sound source localization and tracking, spatially dynamic sound field reconstruction, auralization, and system identification. The recording room has a reverberation time of 0.5 s, and the three different microphone configurations employed include a dummy head, with additional reference microphones located next to the ears, 3 first-order Ambisonics microphones, two circular arrays of 16 and 4 channels, and a 12-channel linear array. The motion of the microphones was achieved using a robotic cart traversing a 4.62 m-long rail at three speeds: [0.2, 0.4, 0.8] m/s. Audio signals were reproduced using two stationary loudspeakers. The collected database features 8648 stationary RIRs, as well as perfect sweeps, speech, music, and stationary noise recorded during motion. Python functions are provided to access the recorded audio and retrieve the associated geometric information.
>
---
#### [replaced 009] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-aware Speech-to-Speech Interaction
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音到语音对话任务，旨在解决S2S模型对副语言特征响应不足的问题。通过引入RL框架ParaS2S，优化响应内容与风格，提升对话自然度。**

- **链接: [https://arxiv.org/pdf/2511.08723](https://arxiv.org/pdf/2511.08723)**

> **作者:** Shu-wen Yang; Ming Tu; Andy T. Liu; Xinghua Qu; Hung-yi Lee; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** To appear in ICLR 2026
>
> **摘要:** Speech-to-Speech (S2S) models have shown promising dialogue capabilities, but their ability to handle paralinguistic cues - such as emotion, tone, and speaker attributes - and to respond appropriately in both content and style remains under-explored. Progress is further hindered by the scarcity of high-quality and expressive demonstrations. To address this, we introduce a new reinforcement learning (RL) framework for paralinguistic-aware S2S, ParaS2S, which evaluates and optimizes both response content and speaking style directly at the waveform level. We first construct ParaS2SBench, a benchmark that evaluates the naturalness of input-output pairs in terms of content and speaking style using expressive and challenging queries. For the automatic judge, we propose a PolyTone training strategy and a multi-stage framework, preventing the style hallucination of end-to-end audio LLM judging. Our judge correlates well with human preferences and is scalable, enabling the model to interact and learn from unlabeled speech via RL. Experiments show that existing S2S models fail to respond appropriately to paralinguistic attributes, performing no better than pipeline-based baselines. Our RL approach (ParaS2SAlign) achieves a 10% relative improvement in the appropriateness of response content and speaking style on ParaS2SBench over supervised fine-tuning (SFT), surpassing all prior models while requiring substantially fewer paired demonstrations than pure SFT. Our findings highlight the need for a scalable and accurate automatic evaluator for speech-to-speech interaction.
>
---
