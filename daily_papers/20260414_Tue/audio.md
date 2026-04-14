# 音频 cs.SD;  eess.AS

- **最新发布 33 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Descriptor-Injected Cross-Modal Learning: A Systematic Exploration of Audio-MIDI Alignment via Spectral and Melodic Features
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于跨模态检索任务，解决音频与MIDI对齐问题。通过引入特征描述符增强编码器，提升模型性能，并分析其有效性。**

- **链接: [https://arxiv.org/pdf/2604.10283](https://arxiv.org/pdf/2604.10283)**

> **作者:** Mariano Fernández Méndez
>
> **备注:** 26 pages, 11 figures, 20 tables. Companion paper to "Harmonic Information Theory: Foundations" (2026). Code: this https URL
>
> **摘要:** Cross-modal retrieval between audio recordings and symbolic music representations (MIDI) remains challenging because continuous waveforms and discrete event sequences encode different aspects of the same performance. We study descriptor injection, the augmentation of modality-specific encoders with hand-crafted domain features, as a bridge across this gap. In a three-phase campaign covering 13 descriptor-mechanism combinations, 6 architectural families, and 3 training schedules, the best configuration reaches a mean S of 84.0 percent across five independent seeds, improving the descriptor-free baseline by 8.8 percentage points. Causal ablation shows that the audio descriptor A4, based on octave-band energy dynamics, drives the gain in the top dual models, while the MIDI descriptor D4 has only a weak inference-time effect despite improving training dynamics. We also introduce reverse cross-attention, where descriptor tokens query encoder features, reducing attention operations relative to the standard formulation while remaining competitive. CKA analysis shows that descriptors substantially increase audio-MIDI transformer layer alignment, indicating representational convergence rather than simple feature concatenation. Perturbation analysis identifies high-frequency octave bands as the dominant discriminative signal. All experiments use MAESTRO v3.0.0 with an evaluation protocol controlling for composer and piece similarity.
>
---
#### [new 002] Ti-Audio: The First Multi-Dialectal End-to-End Speech LLM for Tibetan
- **分类: cs.SD**

- **简介: 该论文属于语音语言模型任务，旨在解决藏语低资源和多方言环境下的语音识别与翻译问题。通过引入动态Q-Former适配器和温度采样策略，构建首个多方言端到端藏语语音LLM——Ti-Audio。**

- **链接: [https://arxiv.org/pdf/2604.11110](https://arxiv.org/pdf/2604.11110)**

> **作者:** Jialing Wang; Yue Zhao; Yuhao Zhang; Jing Yu; Shaosai Li; Zhanchen Dai; Benyou Wang; Haizhou Li
>
> **摘要:** Recent advances in Speech Large Language Models (Speech-LLMs) have made significant progress, greatly enhancing multimodal interaction this http URL, their application in low-resource and dialect-diverse environments still faces challenges. The severe scarcity of Tibetan data, coupled with the phonetic differences among its major dialects (Ü-Tsang, Amdo, and Kham), is a prime example of this challenge. This paper proposes Ti-Audio, the first multi-dialectal end-to-end Speech-LLM for Tibetan. To efficiently align speech and text, we introduce a Dynamic Q-Former Adapter that extracts essential acoustic features from variable-length speech, ensuring stable cross-modal alignment even with limited data. At the data level, we leverage mutual assistance among related dialects to alleviate data scarcity and employ a temperature-based sampling strategy to maximize this synergy. Experimental results demonstrate that Ti-Audio achieves state-of-the-art performance on Tibetan benchmarks for automatic speech recognition and speech translation. Our work validates the effectiveness of cross-dialectal cooperation and provides a scalable paradigm for the development of Speech-LLM in low-resource scenarios.
>
---
#### [new 003] MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation
- **分类: cs.SD; cs.AI; cs.MA**

- **简介: 该论文提出MeloTune，用于个性化音乐推荐的设备端情绪学习与同伴情绪同步系统，解决音乐内容与用户情绪匹配的问题。**

- **链接: [https://arxiv.org/pdf/2604.10815](https://arxiv.org/pdf/2604.10815)**

> **作者:** Hongwei Xu
>
> **备注:** 31 pages, 1 figures, 3 tables
>
> **摘要:** MeloTune is an iPhone-deployed music agent that instantiates the Mesh Memory Protocol (MMP) and Symbolic-Vector Attention Fusion (SVAF) as a production system for affect-aware music curation with peer-to-peer mood coupling. Each device runs two closed-form continuous-time (CfC) networks: a private listener-level CfC that predicts a short-horizon affective trajectory on Russell's circumplex and drives proactive curation, and a shared mesh-runtime CfC at MMP Layer 6 that integrates Cognitive Memory Blocks (CMBs) from co-listening peers. CfC hidden states never cross the wire; only structured CMBs do. A Personal Arousal Function (PAF) replaces the standard linear mapping from audio intensity to psychological arousal with a per-listener learned adjustment, trained from behavioral signals (skip, completion, favorite, volume) and from drift between user-declared mood and machine inference. The same track receives different arousal predictions for different listeners. The model (94,552 parameters) achieves trajectory MAE 0.414, pattern accuracy 96.6%, and intent accuracy 69.4% on held-out validation. PAF evidence from a live deployment session (46 observations across 11 genres) demonstrates that the learning loop operates end-to-end, with pop reaching full confidence after 22 observations. All inference runs on-device via CoreML. To our knowledge, this is the first production deployment of MMP/SVAF on consumer mobile hardware. The accompanying SDK (sym-swift v0.3.78, SYMCore v0.3.7) enforces strict protocol conformance. Music is the case study; the substrate is the contribution.
>
---
#### [new 004] MAGE: Modality-Agnostic Music Generation and Editing
- **分类: cs.SD**

- **简介: 该论文提出MAGE，解决多模态音乐生成与编辑问题，通过统一的连续潜在框架实现跨模态控制，提升生成质量与灵活性。**

- **链接: [https://arxiv.org/pdf/2604.09803](https://arxiv.org/pdf/2604.09803)**

> **作者:** Muhammad Usama Saleem; Tejasvi Ravi; Tianyu Xu; Rajeev Nongpiur; Ishan Chatterjee; Mayur Jagdishbhai Patel; Pu Wang
>
> **摘要:** Multimodal music creation requires models that can both generate audio from high-level cues and edit existing mixtures in a targeted manner. Yet most multimodal music systems are built for a single task and a fixed prompting interface, making their conditioning brittle when guidance is ambiguous, temporally misaligned, or partially missing. Common additive fusion or feature concatenation further weakens cross-modal grounding, often causing prompt drift and spurious musical content during generation and editing. We propose MAGE, a modality-agnostic framework that unifies multimodal music generation and mixture-grounded editing within a single continuous latent formulation. At its core, MAGE uses a Controlled Multimodal FluxFormer, a flow-based Transformer that learns controllable latent trajectories for synthesis and editing under any available subset of conditions. To improve grounding, we introduce Audio-Visual Nexus Alignment to select temporally consistent visual evidence for the audio timeline, and a cross-gated modulation mechanism that applies multiplicative control from aligned visual and textual cues to the audio latents, suppressing unsupported components rather than injecting them. Finally, we train with a dynamic modality-masking curriculum that exposes the model to text-only, visual-only, joint multimodal, and mixture-guided settings, enabling robust inference under missing modalities without training separate models. Experiments on the MUSIC benchmark show that MAGE supports effective multimodal-guided music generation and targeted editing, achieving competitive quality while offering a lightweight and flexible interface tailored to practical music workflows.
>
---
#### [new 005] Whisper-AuT: Domain-Adapted Audio Encoder for Efficient Audio-LLM Training
- **分类: cs.SD**

- **简介: 该论文提出Whisper-AuT，用于改进音频大语言模型的音频编码器。针对Whisper在非语音领域表现弱的问题，通过领域微调提升其性能，以降低下游任务训练成本。**

- **链接: [https://arxiv.org/pdf/2604.10438](https://arxiv.org/pdf/2604.10438)**

> **作者:** Jielin Qiu; Ming Zhu; Wenting Zhao; Zhiwei Liu; Liangwei Yang; Zixiang Chen; Roshan Ram; Akshara Prabhakar; Juntao Tan; Rithesh Murthy; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** Audio-native large language models (audio-LLMs) commonly use Whisper as their audio encoder. However, Whisper was trained exclusively on speech data, producing weak representations for music and environmental sound. This forces downstream audio-LLMs to compensate through extensive training on large-scale non-speech data. We present Whisper-AuT, a domain-adapted audio encoder obtained by fine-tuning Whisper-large-v3 on a curated mixture of speech (80%), environmental sound (10%), and music (10%) totaling approximately 20M samples. The full encoder-decoder is trained end-to-end with a seq2seq captioning objective; the decoder is then discarded and only the encoder is retained. Linear probe evaluations show that Whisper-AuT achieves +23.0% on ESC-50 (environmental sound), +5.0% on GTZAN (music genre), and +0.7% on Speech Commands (keyword spotting) compared to the original Whisperlarge-v3 encoder. Whisper-AuT is designed as a drop-in replacement for Whisper in audio-LLM architectures, with the goal of reducing downstream training cost by providing stronger initial audio representations for non-speech domains.
>
---
#### [new 006] Masked Contrastive Pre-Training Improves Music Audio Key Detection
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐信息检索任务，解决自监督预训练模型在音调敏感任务（如调性检测）中的性能不足问题。通过设计掩码对比预训练方法，提升模型对音调的敏感性，实现最佳效果。**

- **链接: [https://arxiv.org/pdf/2604.10021](https://arxiv.org/pdf/2604.10021)**

> **作者:** Ori Yonay; Tracy Hammond; Tianbao Yang
>
> **备注:** Code and models available at this http URL
>
> **摘要:** Self-supervised music foundation models underperform on key detection, which requires pitch-sensitive representations. In this work, we present the first systematic study showing that the design of self-supervised pretraining directly impacts pitch sensitivity, and demonstrate that masked contrastive embeddings uniquely enable state-of-the-art (SOTA) performance in key detection in the supervised setting. First, we discover that linear evaluation after masking-based contrastive pretraining on Mel spectrograms leads to competitive performance on music key detection out of the box. This leads us to train shallow but wide multi-layer perceptrons (MLPs) on features extracted from our base model, leading to SOTA performance without the need for sophisticated data augmentation policies. We further analyze robustness and show empirically that the learned representations naturally encode common augmentations. Our study establishes self-supervised pretraining as an effective approach for pitch-sensitive MIR tasks and provides insights for designing and probing music foundation models.
>
---
#### [new 007] Teaching the Teachers: Boosting unsupervised domain adaptation in speech recognition by ensemble update
- **分类: eess.AS**

- **简介: 该论文属于语音识别领域的无监督域适应任务，旨在降低跨域数据的词错误率。通过同时更新教师模型和学生模型，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2604.11256](https://arxiv.org/pdf/2604.11256)**

> **作者:** Rehan Ahmad; Muhammad Umar Farooq; Qihang Feng; Thomas Hain
>
> **摘要:** Speech recognition systems often struggle with data domains that have not been included in the training. To address this, unsupervised domain adaptation has been explored with ensemble and multi-stage teacher-student training methods reducing the word error rate. Despite improvements, the error rate remains much higher than that achieved with supervised in-domain training. This work proposes a more efficient strategy by simultaneously updating the ensemble of teacher models along with the single student model eliminating the need for sequential models training. The joint update improves the word error rate of the student model, benefiting the progressively enhanced teacher models. Experiments are conducted with three labelled source datasets, namely AMI, WSJ, LS360, and one unlabeled target domain i.e. SwitchBoard. The results show that the proposed method improves the WER by 4.6% on the Switchboard eval00 test set, thus outperforming multi-stage and iterative training methods.
>
---
#### [new 008] MimicLM: Zero-Shot Voice Imitation through Autoregressive Modeling of Pseudo-Parallel Speech Corpora
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音模仿任务，解决数据稀缺问题。通过使用合成语音作为训练源、真实录音作为目标，提出MimicLM模型，提升语音模仿质量与自然度。**

- **链接: [https://arxiv.org/pdf/2604.11552](https://arxiv.org/pdf/2604.11552)**

> **作者:** Tao Feng; Yuxiang Wang; Yuancheng Wang; Xueyao Zhang; Dekun Chen; Chaoren Wang; Xun Guan; Zhizheng Wu
>
> **摘要:** Voice imitation aims to transform source speech to match a reference speaker's timbre and speaking style while preserving linguistic content. A straightforward approach is to train on triplets of (source, reference, target), where source and target share the same content but target matches the reference's voice characteristics, yet such data is extremely scarce. Existing approaches either employ carefully designed disentanglement architectures to bypass this data scarcity or leverage external systems to synthesize pseudo-parallel training data. However, the former requires intricate model design, and the latter faces a quality ceiling when synthetic speech is used as training targets. To address these limitations, we propose MimicLM, which takes a novel approach by using synthetic speech as training sources while retaining real recordings as targets. This design enables the model to learn directly from real speech distributions, breaking the synthetic quality ceiling. Building on this data construction approach, we incorporate interleaved text-audio modeling to guide the generation of content-accurate speech and apply post-training with preference alignment to mitigate the inherent distributional mismatch when training on synthetic data. Experiments demonstrate that MimicLM achieves superior voice imitation quality with a simple yet effective architecture, significantly outperforming existing methods in naturalness while maintaining competitive similarity scores across speaker identity, accent, and emotion dimensions.
>
---
#### [new 009] Sign-to-Speech Prosody Transfer via Sign Reconstruction-based GAN
- **分类: cs.SD**

- **简介: 该论文提出"手语到语音韵律迁移"任务，解决传统两阶段翻译导致信息丢失的问题。通过构建SignRecGAN框架和S2PFormer模型，直接将手语韵律注入语音合成。**

- **链接: [https://arxiv.org/pdf/2604.10413](https://arxiv.org/pdf/2604.10413)**

> **作者:** Toranosuke Manabe; Yuto Shibata; Shinnosuke Takamichi; Yoshimitsu Aoki
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Deep learning models have improved sign language-to-text translation and made it easier for non-signers to understand signed messages. When the goal is spoken communication, a naive approach is to convert signed messages into text and then synthesize speech via Text-to-Speech (TTS). However, this two-stage pipeline inevitably treat text as a bottleneck representation, causing the loss of rich non-verbal information originally conveyed in the signing. To address this limitation, we propose a novel task, \emph{Sign-to-Speech Prosody Transfer}, which aims to capture the global prosodic nuances expressed in sign language and directly integrate them into synthesized speech. A major challenge is that aligning sign and speech requires expert knowledge, making annotation extremely costly and preventing the construction of large parallel corpora. To overcome this, we introduce \emph{SignRecGAN}, a scalable training framework that leverages unimodal datasets without cross-modal annotations through adversarial learning and reconstruction losses. Furthermore, we propose \emph{S2PFormer}, a new model architecture that preserves the expressive power of existing TTS models while enabling the injection of sign-derived prosody into the synthesized speech. Extensive experiments demonstrate that the proposed method can synthesize speech that faithfully reflects the emotional content of sign language, thereby opening new possibilities for more natural sign language communication. Our code will be available upon acceptance.
>
---
#### [new 010] LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment Generation
- **分类: cs.SD**

- **简介: 该论文属于Vocal-to-Accompaniment生成任务，旨在解决音轨一致性、细节保留与动态编排的难题。提出LaDA-Band框架，结合离散掩码扩散模型与双轨条件生成，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.11052](https://arxiv.org/pdf/2604.11052)**

> **作者:** Qi Wang; Zhexu Shen; Meng Chen; Guoxin Yu; Chaoxu Pang; Weifeng Zhao; Wenjiang Zhou
>
> **备注:** Submitted to ACMMM 2026. Under review
>
> **摘要:** Vocal-to-accompaniment (V2A) generation, which aims to transform a raw vocal recording into a fully arranged accompaniment, inherently requires jointly addressing an accompaniment trilemma: preserving acoustic authenticity, maintaining global coherence with the vocal track, and producing dynamic orchestration across a full song. Existing open-source approaches typically make compromises among these goals. Continuous-latent generation models can capture long musical spans but often struggle to preserve fine-grained acoustic detail. In contrast, discrete autoregressive models retain local fidelity but suffer from unidirectional generation and error accumulation in extended contexts. We present LaDA-Band, an end-to-end framework that introduces Discrete Masked Diffusion to the V2A task. Our approach formulates V2A generation as Discrete Masked Diffusion, i.e., a global, non-autoregressive denoising formulation that combines the representational advantages of discrete audio codec tokens with full-sequence bidirectional context modeling. This design improves long-range structural consistency and temporal synchronization while preserving crisp acoustic details. Built on this formulation, LaDA-Band further introduces a dual-track prefix-conditioning architecture, an auxiliary replaced-token detection objective for weakly anchored accompaniment regions, and a two-stage progressive curriculum to scale Discrete Masked Diffusion to full-song vocal-to-accompaniment generation. Extensive experiments on both academic and real-world benchmarks show that LaDA-Band consistently improves acoustic authenticity, global coherence, and dynamic orchestration over existing baselines, while maintaining strong performance even without auxiliary reference audio. Codes and audio samples are available at this https URL .
>
---
#### [new 011] Speaker Attributed Automatic Speech Recognition Using Speech Aware LLMS
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决多说话人识别问题。通过改进语音感知大模型，引入说话人聚类标签，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.11269](https://arxiv.org/pdf/2604.11269)**

> **作者:** Hagai Aronowitz; Zvi Kons; Avihu Dekel; George Saon; Ron Hoory
>
> **备注:** \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Speaker-Attributed Automatic Speech Recognition (SAA) enhances traditional ASR systems by incorporating relative speaker identity tags directly into the transcript (e.g., [Speaker 1]:, [Speaker 2]:). In this work, we extend the capabilities of Granite-speech, a state-of-the-art speech-aware Large Language Model (LLM) originally trained for transcription and translation. We demonstrate that it can be effectively adapted for SAA with only minimal architectural changes. Our core contribution is the introduction of speaker cluster identification tags (e.g., [Speaker 1 cluster 42]:) which are jointly trained with SAA to significantly improve accuracy. To address limitations in training data, we propose a data augmentation method that uses artificially concatenated multi-speaker conversations. Our approach is evaluated across multiple benchmarks and shows superior performance compared to conventional pipelines that sequentially perform speaker diarization followed by ASR.
>
---
#### [new 012] Multimodal Dataset Normalization and Perceptual Validation for Music-Taste Correspondences
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于跨模态研究任务，旨在解决音乐与风味关联数据收集困难的问题。通过实验验证音频与风味的关联性及计算模型的有效性。**

- **链接: [https://arxiv.org/pdf/2604.10632](https://arxiv.org/pdf/2604.10632)**

> **作者:** Matteo Spanio; Valentina Frezzato; Antonio Rodà
>
> **备注:** Submitted to SMC2026
>
> **摘要:** Collecting large, aligned cross-modal datasets for music-flavor research is difficult because perceptual experiments are costly and small by design. We address this bottleneck through two complementary experiments. The first tests whether audio-flavor correlations, feature-importance rankings, and latent-factor structure transfer from an experimental soundtracks collection (257~tracks with human annotations) to a large FMA-derived corpus ($\sim$49,300 segments with synthetic labels). The second validates computational flavor targets -- derived from food chemistry via a reproducible pipeline -- against human perception in an online listener study (49~participants, 20~tracks). Results from both experiments converge: the quantitative transfer analysis confirms that cross-modal structure is preserved across supervision regimes, and the perceptual evaluation shows significant alignment between computational targets and listener ratings (permutation $p<0.0001$, Mantel $r=0.45$, Procrustes $m^2=0.51$). Together, these findings support the conclusion that sonic seasoning effects are present in synthetic FMA annotations. We release datasets and companion code to support reproducible cross-modal AI research.
>
---
#### [new 013] Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出Audio Flamingo Next，解决音频-语言理解与推理问题，通过增强模型、扩展数据集和引入新推理范式，提升长音频处理能力与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10905](https://arxiv.org/pdf/2604.10905)**

> **作者:** Sreyan Ghosh; Arushi Goel; Kaousheik Jayakumar; Lasha Koroshinadze; Nishit Anand; Zhifeng Kong; Siddharth Gururani; Sang-gil Lee; Jaehyeon Kim; Aya Aljafari; Chao-Han Huck Yang; Sungwon Kim; Ramani Duraiswami; Dinesh Manocha; Mohammad Shoeybi; Bryan Catanzaro; Ming-Yu Liu; Wei Ping
>
> **备注:** Project website: this https URL
>
> **摘要:** We present Audio Flamingo Next (AF-Next), the next-generation and most capable large audio-language model in the Audio Flamingo series, designed to advance understanding and reasoning over speech, environmental sounds and music. Compared to Audio Flamingo 3, AF-Next introduces: (i) a stronger foundational audio-language model that significantly improves accuracy across diverse audio understanding tasks; (ii) scalable strategies for constructing large-scale audio understanding and reasoning data beyond existing academic benchmarks; (iii) support for long and complex audio inputs up to 30 minutes; and (iv) Temporal Audio Chain-of-Thought, a new reasoning paradigm that explicitly grounds intermediate reasoning steps to timestamps in long audio, enabling fine-grained temporal alignment and improved interpretability. To enable these capabilities, we first conduct a systematic analysis of Audio Flamingo 3 to identify key gaps in audio understanding and reasoning. We then curate and scale new large-scale datasets totaling over 1 million hours to address these limitations and expand the existing AudioSkills-XL, LongAudio-XL, AF-Think and AF-Chat datasets. AF-Next is trained using a curriculum-based strategy spanning pre-training, mid-training and post-training stages. Extensive experiments across 20 audio understanding and reasoning benchmarks, including challenging long-audio tasks, show that AF-Next outperforms similarly sized open models by large margins and remains highly competitive with and sometimes surpasses, much larger open-weight and closed models. Beyond benchmark performance, AF-Next exhibits strong real-world utility and transfers well to unseen tasks, highlighting its robustness and generalization ability. In addition to all data, code and methods, we open-source 3 variants of AF-Next, including AF-Next-Instruct, AF-Next-Think and AF-Next-Captioner.
>
---
#### [new 014] Toward using Speech to Sense Student Emotion in Remote Learning Environments
- **分类: eess.AS; cs.HC**

- **简介: 该论文属于情感感知任务，旨在解决远程学习中缺乏情绪线索的问题。通过语音自控任务数据，研究语音情绪特征并实现自动预测。**

- **链接: [https://arxiv.org/pdf/2604.09881](https://arxiv.org/pdf/2604.09881)**

> **作者:** Sargam Vyas; Bogdan Vlasenko; André Mayoraz; Egon Werlen; Per Bergamin; Mathew Magimai.-Doss
>
> **摘要:** With advancements in multimodal communication technologies, remote learning environments such as, distance universities are increasing. Remote learning typically happens asynchronously. As a consequence, unlike face-to-face in-person classroom teaching, this lacks availability of sufficient emotional cues for making learning a pleasant experience. Motivated by advances made in the paralinguistic speech processing community on emotion prediction, in this paper we explore use of speech for sensing students' emotions by building upon speech-based self-control tasks developed to aid effective remote learning. More precisely, we investigate: (a) whether speech acquired through self-control tasks exhibit perceptible variation along valence, arousal, and dominance dimensions? and (b) whether those dimensional emotion variations can be automatically predicted? We address these two research questions by developing a dataset containing spontaneous monologue speech acquired as open responses to self-control tasks and by carrying out subjective listener evaluations and automatic dimensional emotion prediction studies on that dataset. Our investigations indicate that speech-based self-control tasks can be a means to sense student emotion in remote learning environment. This opens potential venues to seamlessly integrate paralinguistic speech processing technologies in the remote learning loop for enhancing learning experiences through instructional design and feedback generation.
>
---
#### [new 015] BMdataset: A Musicologically Curated LilyPond Dataset
- **分类: cs.SD; cs.CL; cs.IR**

- **简介: 该论文提出BMdataset，一个基于LilyPond的巴洛克音乐数据集，用于音乐理解任务。解决传统MIDI数据集的局限性，通过专家标注提升数据质量，并引入LilyBERT模型进行有效学习。**

- **链接: [https://arxiv.org/pdf/2604.10628](https://arxiv.org/pdf/2604.10628)**

> **作者:** Matteo Spanio; Ilay Guler; Antonio Rodà
>
> **备注:** Submitted to SMC2026
>
> **摘要:** Symbolic music research has relied almost exclusively on MIDI-based datasets; text-based engraving formats such as LilyPond remain unexplored for music understanding. We present BMdataset, a musicologically curated dataset of 393 LilyPond scores (2,646 movements) transcribed by experts directly from original Baroque manuscripts, with metadata covering composer, musical form, instrumentation, and sectional attributes. Building on this resource, we introduce LilyBERT (weights can be found at this https URL), a CodeBERT-based encoder adapted to symbolic music through vocabulary extension with 115 LilyPond-specific tokens and masked language model pre-training. Linear probing on the out-of-domain Mutopia corpus shows that, despite its modest size (~90M tokens), fine-tuning on BMdataset alone outperforms continuous pre-training on the full PDMX corpus (~15B tokens) for both composer and style classification, demonstrating that small, expertly curated datasets can be more effective than large, noisy corpora for music understanding. Combining broad pre-training with domain-specific fine-tuning yields the best results overall (84.3% composer accuracy), confirming that the two data regimes are complementary. We release the dataset, tokenizer, and model to establish a baseline for representation learning on LilyPond.
>
---
#### [new 016] HumDial-EIBench: A Human-Recorded Multi-Turn Emotional Intelligence Benchmark for Audio Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出HumDial-EIBench，用于评估音频语言模型的情感智能。针对现有基准的不足，该工作利用真实对话数据，设计多轮情感跟踪和因果推理任务，解决模型在情感理解与跨模态冲突中的问题。**

- **链接: [https://arxiv.org/pdf/2604.11594](https://arxiv.org/pdf/2604.11594)**

> **作者:** Shuiyuan Wang; Zhixian Zhao; Hongfei Yue; Chengyou Wang; Shuai Wang; Hui Bu; Xin Xu; Lei Xie
>
> **摘要:** Evaluating the emotional intelligence (EI) of audio language models (ALMs) is critical. However, existing benchmarks mostly rely on synthesized speech, are limited to single-turn interactions, and depend heavily on open-ended scoring. This paper proposes HumDial-EIBench, a comprehensive benchmark for evaluating ALMs' EI. Using real-recorded human dialogues from the ICASSP 2026 HumDial Challenge, it reformulates emotional tracking and causal reasoning into multiple-choice questions with adversarial distractors, mitigating subjective scoring bias for cognitive tasks. It retains the generation of empathetic responses and introduces an acoustic-semantic conflict task to assess robustness against contradictory multimodal signals. Evaluations of eight ALMs reveal that most models struggle with multi-turn emotional tracking and implicit causal reasoning. Furthermore, all models exhibit decoupled textual and acoustic empathy, alongside a severe text-dominance bias during cross-modal conflicts.
>
---
#### [new 017] Real-Time Voicemail Detection in Telephony Audio Using Temporal Speech Activity Features
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音识别任务，旨在实时区分电话语音中的语音信箱问候与真人回答。通过提取时间特征并使用集成分类器，实现高准确率的检测。**

- **链接: [https://arxiv.org/pdf/2604.09675](https://arxiv.org/pdf/2604.09675)**

> **作者:** Kumar Saurav
>
> **备注:** 16 pages, 5 tables. Preprint
>
> **摘要:** Outbound AI calling systems must distinguish voicemail greetings from live human answers in real time to avoid wasted agent interactions and dropped calls. We present a lightweight approach that extracts 15 temporal features from the speech activity pattern of a pre-trained neural voice activity detector (VAD), then classifies with a shallow tree-based ensemble. Across two evaluation sets totaling 764 telephony recordings, the system achieves a combined 96.1% accuracy (734/764), with 99.3% (139/140) on an expert-labeled test set and 95.4% (595/624) on a held-out production set. In production validation over 77,000 calls, it maintained a 0.3% false positive rate and 1.3% false negative rate. End-to-end inference completes in 46 ms on a commodity dual-core CPU with no GPU, supporting 380+ concurrent WebSocket calls. In our search over 3,780 model, feature, and threshold combinations, feature importance was concentrated in three temporal variables. Adding transcription keywords or beep-based features did not improve the best real-time configuration and increased latency substantially. Our results suggest that temporal speech patterns are a strong signal for distinguishing voicemail greetings from live human answers.
>
---
#### [new 018] ActorMind: Emulating Human Actor Reasoning for Speech Role-Playing
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音角色扮演任务，旨在解决现有工作仅限文本、忽视语音的问题。提出ActorMind框架和ActorMindBench基准，提升模型的语音角色扮演能力。**

- **链接: [https://arxiv.org/pdf/2604.11103](https://arxiv.org/pdf/2604.11103)**

> **作者:** Xi Chen; Wei Xue; Yike Guo
>
> **摘要:** Role-playing has garnered rising attention as it provides a strong foundation for human-machine interaction and facilitates sociological research. However, current work is confined to textual modalities, neglecting speech, which plays a predominant role in daily life, thus limiting genuine role-playing. To bridge this gap, we conceptualize and benchmark speech role-playing through ActorMindBench, and we present a corresponding reasoning framework, called ActorMind. Specifically, (1) Speech Role-Playing enables models to deliver spontaneous responses with personalized verbal traits based on their role, the scene, and spoken dialogue. (2) ActorMindBench is a hierarchical benchmark comprises Utterance-Level content with 7,653 utterances, Scene-Level content with 313 scenes, and Role-Level content with 6 roles. (3) ActorMind is an off-the-shelf, multi-agent, chain-of-though style reasoning framework that emulates how human actors perform in theaters. Concretely, ActorMind first reads its assigned role description via Eye Agent, then comprehends emotional cues within contextual spoken dialogues through Ear Agent. Subsequently, Brain Agent generates a descriptive emotional state, and finally, Mouth Agent delivers the scripts infused with corresponding emotion state. Experimental results demonstrate the effectiveness of ActorMind in enhancing speech role-playing.
>
---
#### [new 019] Learning to Attend to Depression-Related Patterns: An Adaptive Cross-Modal Gating Network for Depression Detection
- **分类: cs.SD**

- **简介: 该论文属于抑郁症检测任务，旨在解决现有方法忽略语音中抑郁相关特征稀疏性的问题。提出ACMG网络，自适应加权多模态帧，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.10181](https://arxiv.org/pdf/2604.10181)**

> **作者:** Hangbin Yu; Yudong Yang; Rongfeng Su; Nan Yan; Lan Wang
>
> **摘要:** Automatic depression detection using speech signals with acoustic and textual modalities is a promising approach for early diagnosis. Depression-related patterns exhibit sparsity in speech: diagnostically relevant features occur in specific segments rather than being uniformly distributed. However, most existing methods treat all frames equally, assuming depression-related information is uniformly distributed and thus overlooking this sparsity. To address this issue, we proposes a depression detection network based on Adaptive Cross-Modal Gating (ACMG) that adaptively reassigns frame-level weights across both modalities, enabling selective attention to depression-related segments. Experimental results show that the depression detection system with ACMG outperforms baselines without it. Visualization analyses further confirm that ACMG automatically attends to clinically meaningful patterns, including low-energy acoustic segments and textual segments containing negative sentiments.
>
---
#### [new 020] Direction-Preserving MIMO Speech Enhancement Using a Neural Covariance Estimator
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决多通道语音增强中保留方向信息的问题。提出一种基于神经网络的协方差估计方法，实现高效的方向保持增强。**

- **链接: [https://arxiv.org/pdf/2604.11179](https://arxiv.org/pdf/2604.11179)**

> **作者:** Thomas Deppisch
>
> **摘要:** Multichannel speech enhancement is widely used as a front-end in microphone array processing systems. While most existing approaches produce a single enhanced signal, direction-preserving multiple-input multiple-output (MIMO) methods instead aim to provide enhanced multichannel signals that retain directional properties, enabling downstream applications such as beamforming, binaural rendering, and direction-of-arrival estimation. In this work, we propose a fully blind, direction-preserving MIMO speech enhancement method based on neural estimation of the spatial noise covariance matrix. A lightweight OnlineSpatialNet estimates a scale-normalized Cholesky factor of the frequency-domain noise covariance, which is combined with a direction-preserving MIMO Wiener filter to enhance speech while preserving the spatial characteristics of both target and residual noise. In contrast to prior approaches relying on oracle information or mask-based covariance estimation for single-output systems, the proposed method directly targets accurate multichannel covariance estimation with low computational complexity. Experimental results show improved speech enhancement, covariance estimation capability, and performance in downstream tasks over a mask-based baseline, approaching oracle performance with significantly fewer parameters and computational cost.
>
---
#### [new 021] Cross-Cultural Bias in Mel-Scale Representations: Evidence and Alternatives from Speech and Music
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频特征提取中的跨文化偏差问题，比较不同频域表示在语音和音乐任务中的表现，提出改进方法以减少性能差异。**

- **链接: [https://arxiv.org/pdf/2604.10503](https://arxiv.org/pdf/2604.10503)**

> **作者:** Shivam Chauhan; Ajay Pundhir
>
> **备注:** 5 pages, 3 figures, 4 tables. Accepted at ICASSP 2026
>
> **摘要:** Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We present a comprehensive evaluation of cross-cultural bias in audio front-ends, comparing mel-scale features with learnable alternatives (LEAF, SincNet) and psychoacoustic variants (ERB, Bark, CQT) across speech recognition (11 languages), music analysis (6 collections), and European acoustic scene classification (10 European cities). Our controlled experiments isolate front-end contributions while holding architecture and training protocols minimal and constant. Results demonstrate that mel-scale features yield 31.2% WER for tonal languages compared to 18.7% for non-tonal languages (12.5% gap), and show 15.7% F1 degradation between Western and non-Western music. Alternative representations significantly reduce these disparities: LEAF reduces the speech gap by 34% through adaptive frequency allocation, CQT achieves 52% reduction in music performance gaps, and ERB-scale filtering cuts disparities by 31% with only 1% computational overhead. We also release FairAudioBench, enabling cross-cultural evaluation, and demonstrate that adaptive frequency decomposition offers practical paths toward equitable audio processing. These findings reveal how foundational signal processing choices propagate bias, providing crucial guidance for developing inclusive audio systems.
>
---
#### [new 022] VidAudio-Bench: Benchmarking V2A and VT2A Generation across Four Audio Categories
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出VidAudio-Bench，用于评估视频到音频（V2A）和视频文本到音频（VT2A）生成任务。针对现有基准不足，构建多任务评测框架，涵盖四类音频，引入新指标并验证人类偏好一致性。**

- **链接: [https://arxiv.org/pdf/2604.10542](https://arxiv.org/pdf/2604.10542)**

> **作者:** Qian Zhang; Yuqin Cao; Yixuan Gao; Xiongkuo Min
>
> **摘要:** Video-to-Audio (V2A) generation is essential for immersive multimedia experiences, yet its evaluation remains underexplored. Existing benchmarks typically assess diverse audio types under a unified protocol, overlooking the fine-grained requirements of distinct audio categories. To address this gap, we propose VidAudio-Bench, a multi-task benchmark for V2A evaluation with four key features: (1) Broad Coverage: It encompasses four representative audio categories - sound effects, music, speech, and singing - under both V2A and Video-Text-to-Audio (VT2A) settings. (2) Extensive Evaluation: It comprises 1,634 video-text pairs and benchmarks 11 state-of-the-art generation models. (3) Comprehensive Metrics: It introduces 13 task-specific, reference-free metrics to systematically assess audio quality, video-audio consistency, and text-audio consistency. (4) Human Alignment: It validates all metrics through subjective studies, demonstrating strong consistency with human preferences. Experimental results reveal that current V2A models perform poorly in speech and singing compared to sound effects. Our VT2A results further highlight a fundamental tension between instruction following and visually grounded generation: stronger visual conditioning improves video-audio alignment, but often at the cost of generating the intended audio category. These findings establish VidAudio-Bench as a comprehensive and scalable framework for diagnosing V2A systems and provide new insights into multimodal audio generation.
>
---
#### [new 023] Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文提出Audio-Omni，解决音频生成与编辑的统一框架问题，整合多模态理解，提升音频处理能力。**

- **链接: [https://arxiv.org/pdf/2604.10708](https://arxiv.org/pdf/2604.10708)**

> **作者:** Zeyue Tian; Binxin Yang; Zhaoyang Liu; Jiexuan Zhang; Ruibin Yuan; Hubery Yin; Qifeng Chen; Chen Li; Jing Lv; Wei Xue; Yike Guo
>
> **摘要:** Recent progress in multimodal models has spurred rapid advances in audio understanding, generation, and editing. However, these capabilities are typically addressed by specialized models, leaving the development of a truly unified framework that can seamlessly integrate all three tasks underexplored. While some pioneering works have explored unifying audio understanding and generation, they often remain confined to specific domains. To address this, we introduce Audio-Omni, the first end-to-end framework to unify generation and editing across general sound, music, and speech domains, with integrated multi-modal understanding capabilities. Our architecture synergizes a frozen Multimodal Large Language Model for high-level reasoning with a trainable Diffusion Transformer for high-fidelity synthesis. To overcome the critical data scarcity in audio editing, we construct AudioEdit, a new large-scale dataset comprising over one million meticulously curated editing pairs. Extensive experiments demonstrate that Audio-Omni achieves state-of-the-art performance across a suite of benchmarks, outperforming prior unified approaches while achieving performance on par with or superior to specialized expert models. Beyond its core capabilities, Audio-Omni exhibits remarkable inherited capabilities, including knowledge-augmented reasoning generation, in-context generation, and zero-shot cross-lingual control for audio generation, highlighting a promising direction toward universal generative audio intelligence. The code, model, and dataset will be publicly released on this https URL.
>
---
#### [new 024] From Speech to Profile: A Protocol-Driven LLM Agent for Psychological Profile Generation
- **分类: cs.SD**

- **简介: 该论文属于心理档案生成任务，解决长对话中记忆丢失和幻觉问题。提出StreamProfile框架，通过分步处理和证据存储，生成可追溯的心理档案。**

- **链接: [https://arxiv.org/pdf/2604.10161](https://arxiv.org/pdf/2604.10161)**

> **作者:** Xingjian Yang; Yudong Yang; Zhixing Guo; Yongjie Zhou; Nan Yan; Lan Wang
>
> **摘要:** The psychological profile that structurally documents the case of a depression patient is essential for psychotherapy. Large language models can be applied to summarize the profiles from counseling speech, however, it may suffer from long-context forgetting and produce unverifiable hallucinations, due to overlong length of speech, multi-party interactions and unstructured chatting. Hereby, we propose a StreamProfile, a streaming framework that processes counseling speech incrementally, extracts evidences grounded from ASR transcriptions by storing it in a Hierarchical Evidence Memory, and then performs a Chain-of-Thought pipeline according to PM+ psychological intervention for clinical reasoning. The final profile is synthesized strictly from those evidences, making every claim traceable. Experiments on real-world teenager counseling speech have shown that the proposed StreamProfile system can accurately generate the profiles and prevent hallucination.
>
---
#### [new 025] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering
- **分类: cs.IR; cs.MM; cs.SD**

- **简介: 该论文属于音乐问答任务，旨在解决多音轨比较问答的问题。构建了Jamendo-MT-QA数据集，包含36,519个比较问答对，用于评估模型在多音轨上的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.09721](https://arxiv.org/pdf/2604.09721)**

> **作者:** Junyoung Koh; Jaeyun Lee; Soo Yong Kim; Gyu Hyeong Choi; Jung In Koh; Jordan Phillips; Yeonjin Lee; Min Song
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Recent work on music question answering (Music-QA) has primarily focused on single-track understanding, where models answer questions about an individual audio clip using its tags, captions, or metadata. However, listeners often describe music in comparative terms, and existing benchmarks do not systematically evaluate reasoning across multiple tracks. Building on the Jamendo-QA dataset, we introduce Jamendo-MT-QA, a dataset and benchmark for multi-track comparative question answering. From Creative Commons-licensed tracks on Jamendo, we construct 36,519 comparative QA items over 12,173 track pairs, with each pair yielding three question types: yes/no, short-answer, and sentence-level questions. We describe an LLM-assisted pipeline for generating and filtering comparative questions, and benchmark representative audio-language models using both automatic metrics and LLM-as-a-Judge evaluation.
>
---
#### [new 026] Knowing What to Stress: A Discourse-Conditioned Text-to-Speech Benchmark
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决TTS系统在语境中正确标注重音的问题。通过构建基准数据集，发现TTS系统在实现语境合适的重音上存在不足。**

- **链接: [https://arxiv.org/pdf/2604.10580](https://arxiv.org/pdf/2604.10580)**

> **作者:** Arnon Turetzky; Avihu Dekel; Hagai Aronowitz; Ron Hoory; Yossi Adi
>
> **备注:** Preprint
>
> **摘要:** Spoken meaning often depends not only on what is said, but also on which word is emphasized. The same sentence can convey correction, contrast, or clarification depending on where emphasis falls. Although modern text-to-speech (TTS) systems generate expressive speech, it remains unclear whether they infer contextually appropriate stress from discourse alone. To address this gap, we present Context-Aware Stress TTS (CAST), a benchmark for evaluating context-conditioned word-level stress in TTS. Items are defined as contrastive context pairs: identical sentences paired with distinct contexts requiring different stressed words. We evaluate state-of-the-art systems and find a consistent gap: text-only language models reliably recover the intended stress from context, yet TTS systems frequently fail to realize it in speech. We release the benchmark, evaluation framework, construction pipeline and a synthetic corpus to support future work on context-aware speech synthesis.
>
---
#### [new 027] Regularized Entropy Information Adaptation with Temporal-Awareness Networks for Simultaneous Speech Translation
- **分类: cs.LG; eess.AS**

- **简介: 该论文属于语音翻译任务，解决实时性与翻译质量的平衡问题。通过改进REINA方法，提出REINA-SAN和REINA-TAN，提升流式效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2604.09916](https://arxiv.org/pdf/2604.09916)**

> **作者:** Joseph Liu; Nameer Hirschkind; Xiao Yu; Mahesh Kumar Nandwana
>
> **备注:** Under review at Interspeech 2026
>
> **摘要:** Simultaneous Speech Translation (SimulST) requires balancing high translation quality with low latency. Recent work introduced REINA, a method that trains a Read/Write policy based on estimating the information gain of reading more audio. However, we find that information-based policies often lack temporal context, leading the policy to bias itself toward reading most of the audio before starting to write. We improve REINA using two distinct strategies: a supervised alignment network (REINA-SAN) and a timestep-augmented network (REINA-TAN). Our results demonstrate that while both methods significantly outperform the baseline and resolve stability issues, REINA-TAN provides a slightly superior Pareto frontier for streaming efficiency, whereas REINA-SAN offers more robustness against 'read loops'. Applied to Whisper, both methods improve the pareto frontier of streaming efficiency as measured by Normalized Streaming Efficiency (NoSE) scores up to 7.1% over existing competitive baselines.
>
---
#### [new 028] BlasBench: An Open Benchmark for Irish Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决爱尔兰语ASR系统缺乏公开基准的问题。作者发布BlasBench，包含爱尔兰语文本规范化的评估框架，并对比了多个系统的表现。**

- **链接: [https://arxiv.org/pdf/2604.10736](https://arxiv.org/pdf/2604.10736)**

> **作者:** Jyoutir Raj; John Conway
>
> **备注:** 8 pages, 4 tables, 3 appendices. Code and data: this https URL
>
> **摘要:** No open Irish-specific benchmark compares end-user ASR systems under a shared Irish-aware evaluation protocol. To solve this, we release BlasBench, an open evaluation harness with Irish-aware text normalisation that preserves fadas, lenition, and eclipsis. We benchmark 12 systems across four architecture families on Common Voice ga-IE and FLEURS ga-IE. All Whisper variants exceed 100% WER. The best open model (omniASR LLM 7B) achieves 30.65% WER on Common Voice and 39.09% on FLEURS. We noticed models fine-tuned on Common Voice lose 33-43 WER points on FLEURS, revealing a generalisation gap that is invisible to single-dataset evaluation.
>
---
#### [new 029] Beyond Monologue: Interactive Talking-Listening Avatar Generation with Conversational Audio Context-Aware Kernels
- **分类: cs.AI; cs.SD**

- **简介: 该论文属于语音驱动视频生成任务，解决全双工交互中语音与视频同步问题。提出多头高斯核模型，提升对话响应自然度与唇形同步性。**

- **链接: [https://arxiv.org/pdf/2604.10367](https://arxiv.org/pdf/2604.10367)**

> **作者:** Yuzhe Weng; Haotian Wang; Xinyi Yu; Xiaoyan Wu; Haoran Xu; Shan He; Jun Du
>
> **摘要:** Audio-driven human video generation has achieved remarkable success in monologue scenarios, largely driven by advancements in powerful video generation foundation models. Moving beyond monologues, authentic human communication is inherently a full-duplex interactive process, requiring virtual agents not only to articulate their own speech but also to react naturally to incoming conversational audio. Most existing methods simply extend conventional audio-driven paradigms to listening scenarios. However, relying on strict frame-to-frame alignment renders the model's response to long-range conversational dynamics rigid, whereas directly introducing global attention catastrophically degrades lip synchronization. Recognizing the unique temporal Scale Discrepancy between talking and listening behaviors, we introduce a multi-head Gaussian kernel to explicitly inject this physical intuition into the model as a progressive temporal inductive bias. Building upon this, we construct a full-duplex interactive virtual agent capable of simultaneously processing dual-stream audio inputs for both talking and listening. Furthermore, we introduce a rigorously cleaned Talking-Listening dataset VoxHear featuring perfectly decoupled speech and background audio tracks. Extensive experiments demonstrate that our approach successfully fuses strong temporal alignment with deep contextual semantics, setting a new state-of-the-art for generating highly natural and responsive full-duplex interactive digital humans. The project page is available at this https URL .
>
---
#### [new 030] ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音交互任务，解决全双工语音语言模型中对话时序优化问题。通过ASPIRin框架分离说话时机与内容，提升交互性并减少重复。**

- **链接: [https://arxiv.org/pdf/2604.10065](https://arxiv.org/pdf/2604.10065)**

> **作者:** Chi-Yuan Hsiao; Ke-Han Lu; Yu-Kuan Fu; Guan-Ting Lin; Hsiao-Tsung Hung; Hung-yi Lee
>
> **摘要:** End-to-end full-duplex Speech Language Models (SLMs) require precise turn-taking for natural interaction. However, optimizing temporal dynamics via standard raw-token reinforcement learning (RL) degrades semantic quality, causing severe generative collapse and repetition. We propose ASPIRin, an interactivity-optimized RL framework that explicitly decouples when to speak from what to say. Using Action Space Projection, ASPIRin maps the text vocabulary into a coarse-grained binary state (active speech vs. inactive silence). By applying Group Relative Policy Optimization (GRPO) with rule-based rewards, it balances user interruption and response latency. Empirical evaluations show ASPIRin optimizes interactivity across turn-taking, backchanneling, and pause handling. Crucially, isolating timing from token selection preserves semantic coherence and reduces the portion of duplicate n-grams by over 50% compared to standard GRPO, effectively eliminating degenerative repetition.
>
---
#### [new 031] Cross-Validated Cross-Channel Self-Attention and Denoising for Automatic Modulation Classification
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于自动调制分类任务，旨在解决噪声环境下深度学习模型性能下降的问题。通过引入交叉通道自注意力和去噪模块，提升模型在低至中等信噪比下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10054](https://arxiv.org/pdf/2604.10054)**

> **作者:** Prakash Suman; Yanzhen Qu
>
> **摘要:** This study addresses a key limitation in deep learning Automatic Modulation Classification (AMC) models, which perform well at high signal-to-noise ratios (SNRs) but degrade under noisy conditions due to conventional feature extraction suppressing both discriminative structure and interference. The goal was to develop a feature-preserving denoising method that mitigates the loss of modulation class separation. A deep learning AMC model was proposed, incorporating a cross-channel self-attention block to capture dependencies between in-phase and quadrature components, along with dual-path deep residual shrinkage denoising blocks to suppress noise. Experiments using the RML2018.01a dataset employed stratified sampling across 24 modulation types and 26 SNR levels. Results showed that denoising depth strongly influences robustness at low and moderate SNRs. Compared to benchmark models PET-CGDNN, MCLDNN, and DAE, the proposed model achieved notable accuracy improvements across -8 dB to +2 dB SNR, with increases of 3%, 2.3%, and 14%, respectively. Cross-validation confirmed the model's robustness, yielding a mean accuracy of 62.6%, macro precision of 65.8%, macro-recall of 62.6%, and macro-F1 score of 62.9%. The architecture advances interference-aware AMC by formalizing baseband modeling as orthogonal subproblems and introducing cross-channel attention as a generalized complex interaction operator, with ablations confirming the critical role of feature-preserving denoising for robustness at low-to-medium SNR.
>
---
#### [new 032] Efficient Training for Cross-lingual Speech Language Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于跨语言语音语言模型任务，旨在解决数据有限和多语言扩展困难的问题。提出CSLM方法，通过离散语音标记实现跨模态与跨语言对齐，提升生成质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2604.11096](https://arxiv.org/pdf/2604.11096)**

> **作者:** Yan Zhou; Qingkai Fang; Yun Hong; Yang Feng
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Currently, large language models (LLMs) predominantly focus on the text modality. To enable more natural human-AI interaction, speech LLMs are emerging, but building effective end-to-end speech LLMs remains challenging due to limited data and the difficulty in expanding to more languages. In this paper, we introduce Cross-lingual Speech Language Model (CSLM), an efficient training method for cross-lingual speech LLMs based on discrete speech tokens. We propose a novel alignment strategy that achieves cross-modal and cross-lingual alignment through continual pre-training. By conducting instruction fine-tuning following a speech-text interleaved chain-of-modality generation process, we enhance modal alignment at a finer granularity, thereby improving generation quality and reducing latency. CSLM aligns different modalities and languages simultaneously without the need for massive speech data, thus exhibiting good language scalability. Evaluations on cross-modal tasks, mono-lingual conversational tasks, and cross-lingual conversational tasks demonstrate CSLM's strong cross-modal alignment capabilities and general task abilities. (Code is available at: this https URL)
>
---
#### [new 033] Speech-preserving active noise control: a deep learning approach in reverberant environments
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决传统ANC系统在复杂环境中噪声抑制时损伤语音的问题。提出一种基于深度学习的ANC系统，有效保留语音同时降低噪声。**

- **链接: [https://arxiv.org/pdf/2604.10979](https://arxiv.org/pdf/2604.10979)**

> **作者:** Shuning Dai
>
> **备注:** 89 pages, 17 figures, master's dissertation
>
> **摘要:** Traditional Active Noise Control (ANC) systems are mostly based on FxLMS algorithms, but such algorithms rely on linear assumptions and are often limited in handling broadband non-stationary noise or nonlinear acoustic paths. Not only that, the traditional method is used to eliminating all signals together, and noise reduction often accidentally damages the voice signal and affects normal communication. To tackle these issues, this study proposes a speech preserving deep learning ANC system, which aims to achieve stable noise reduction while effectively retaining speech in a complex acoustic environment. This study builds an end-to-end control architecture, the core of which adopts a Convolutional Recurrent Network (CRN). The structure uses the long short-term memory (LSTM) network to capture the time-related characteristics of acoustic signals. Combined with complex spectrum mapping (CSM) technology, the nonlinear distortion problem is effectively solved. In order to retain useful voice while removing noise, this study also designs a special voice retention loss function. This design guidance model selectively retains the target voice while suppressing environmental noise by identifying the characteristics of the spectrum structure. In addition, in order to verify whether the system is effective in real scenes, we use the Image Source Method (ISM) to build a high-fidelity acoustic simulation environment, which also simulates the real reverberation effect. Experimental results demonstrate that the proposed Deep ANC system achieves significantly better noise reduction than the traditional FxLMS algorithm, especially for non-stationary noises like crowd babble. Meanwhile, PESQ and STOI based evaluations confirm that the system preserves both the naturalness and intelligibility of the target speech.
>
---
## 更新

#### [replaced 001] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于长文本语音问答任务，旨在解决长音频处理困难的问题。提出CLSR模型，通过对比学习提取相关语音片段，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2511.09282](https://arxiv.org/pdf/2511.09282)**

> **作者:** Jiliang Hu; Zuchao Li; Baoyuan Qi; Liu Guoming; Ping Wang
>
> **备注:** 12 pages, 7 figures, accepted by AAAI 2026
>
> **摘要:** Significant progress has been made in spoken question answering (SQA) in recent years. However, many existing methods, including large audio language models, struggle with processing long audio. Follow the success of retrieval augmented generation, a speech-related retriever shows promising in help preprocessing long-form speech. But the performance of existing speech-related retrievers is lacking. To address this challenge, we propose CLSR, an end-to-end contrastive language-speech retriever that efficiently extracts question-relevant segments from long audio recordings for downstream SQA task. Unlike conventional speech-text contrastive models, CLSR incorporates an intermediate step that converts acoustic features into text-like representations prior to alignment, thereby more effectively bridging the gap between modalities. Experimental results across four cross-modal retrieval datasets demonstrate that CLSR surpasses both end-to-end speech related retrievers and pipeline approaches combining speech recognition with text retrieval, providing a robust foundation for advancing practical long-form SQA applications.
>
---
#### [replaced 002] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和交互纠错缺失的问题。提出基于大模型的语义评估和交互框架，提升识别的语义准确性和交互能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
#### [replaced 003] Hierarchical Decoding for Discrete Speech Synthesis with Multi-Resolution Spoof Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，解决离散语音合成中的伪迹和分布漂移问题。提出MSpoof-TTS框架，通过多分辨率伪冒检测提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.05373](https://arxiv.org/pdf/2603.05373)**

> **作者:** Junchuan Zhao; Minh Duc Vu; Ye Wang
>
> **备注:** 7 pages, 3 figures, 3 tables, 2 algorithms
>
> **摘要:** Neural codec language models enable high-quality discrete speech synthesis, yet their inference remains vulnerable to token-level artifacts and distributional drift that degrade perceptual realism. Rather than relying on preference optimization or retraining, we propose MSpoof-TTS, a training-free inference framework that improves zero-shot synthesis through multi-resolution spoof guidance. We introduce a Multi-Resolution Token-based Spoof Detection framework that evaluates codec sequences at different temporal granularities to detect locally inconsistent or unnatural patterns. We then integrate the spoof detectors into a hierarchical decoding strategy, progressively pruning low-quality candidates and re-ranking hypotheses. This discriminator-guided generation enhances robustness without modifying model parameters. Experiments validate the effectiveness of our framework for robust and high-quality codec-based speech generation. Audio samples are available at this https URL.
>
---
#### [replaced 004] PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决自动配音中的同步问题。通过引入音素同步和语义相似性，提升配音的唇形同步与语义准确性。**

- **链接: [https://arxiv.org/pdf/2604.09111](https://arxiv.org/pdf/2604.09111)**

> **作者:** Changi Hong; Yoonah Song; Hwayoung Park; Chaewoon Bang; Dayeon Gu; Do Hyun Lee; Hong Kook Kim
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Recently, artificial intelligence-based dubbing technology has advanced, enabling automated dubbing (AD) to convert the source speech of a video into target speech in different languages. However, natural AD still faces synchronization challenges such as duration and lip-synchronization (lip-sync), which are crucial for preserving the viewer experience. Therefore, this paper proposes a synchronization method for AD processes that paraphrases translated text, comprising two steps: isochrony for timing constraints and phonetic synchronization (PS) to preserve lip-sync. First, we achieve isochrony by paraphrasing the translated text with a language model, ensuring the target speech duration matches that of the source speech. Second, we introduce PS, which employs dynamic time warping (DTW) with local costs of vowel distances measured from training data so that the target text composes vowels with pronunciations similar to source vowels. Third, we extend this approach to PSComet, which jointly considers semantic and phonetic similarity to preserve meaning better. The proposed methods are incorporated into text-to-speech systems, PS-TTS and PS-Comet TTS. The performance evaluation using Korean and English lip-reading datasets and a voice-actor dubbing dataset demonstrates that both systems outperform TTS without PS on several objective metrics and outperform voice actors in Korean-to-English and English-to-Korean dubbing. We extend the experiments to French, testing all pairs among these languages to evaluate cross-linguistic applicability. Across all language pairs, PS-Comet performed best, balancing lip-sync accuracy with semantic preservation, confirming that PS-Comet achieves more accurate lip-sync with semantic preservation than PS alone.
>
---
#### [replaced 005] ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于语音欺骗检测任务，旨在解决深度伪造和对抗攻击的检测问题。通过分析53支团队的提交结果，评估不同技术在对抗攻击下的表现，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.03944](https://arxiv.org/pdf/2601.03944)**

> **作者:** Xin Wang; Héctor Delgado; Nicholas Evans; Xuechen Liu; Tomi Kinnunen; Hemlata Tak; Kong Aik Lee; Ivan Kukanov; Md Sahidullah; Massimiliano Todisco; Junichi Yamagishi
>
> **备注:** Accepted by IEEE TASLP. Appendix is included. DOI https://doi.org/10.1109/TASLPRO.2026.3682962 (Open Access)
>
> **摘要:** ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake detection solutions. A significant change from previous challenge editions is a new crowdsourced database collected from a substantially greater number of speakers under diverse recording conditions, and a mix of cutting-edge and legacy generative speech technology. With the new database described elsewhere, we provide in this paper an overview of the ASVspoof 5 challenge results for the submissions of 53 participating teams. While many solutions perform well, performance degrades under adversarial attacks and the application of neural encoding/compression schemes. Together with a review of post-challenge results, we also report a study of calibration in addition to other principal challenges and outline a road-map for the future of ASVspoof.
>
---
#### [replaced 006] Woosh: A Sound Effects Foundation Model
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文介绍了一种名为Woosh的声音效果基础模型，用于生成和处理音频。任务是提升声音效果生成性能，解决现有模型的不足。工作包括设计多模块模型并进行性能评估。**

- **链接: [https://arxiv.org/pdf/2604.01929](https://arxiv.org/pdf/2604.01929)**

> **作者:** Gaëtan Hadjeres; Marc Ferras; Khaled Koutini; Benno Weck; Alexandre Bittar; Thomas Hummel; Zineb Lahrici; Hakim Missoum; Joan Serrà; Yuki Mitsufuji
>
> **摘要:** The audio research community depends on open generative models as foundational tools for building novel approaches and establishing baselines. In this report, we present Woosh, Sony AI's publicly released sound effect foundation model, detailing its architecture, training process, and an evaluation against other popular open models. Being optimized for sound effects, we provide (1) a high-quality audio encoder/decoder model and (2) a text-audio alignment model for conditioning, together with (3) text-to-audio and (4) video-to-audio generative models. Distilled text-to-audio and video-to-audio models are also included in the release, allowing for low-resource operation and fast inference. Our evaluation on both public and private data shows competitive or better performance for each module when compared to existing open alternatives like StableAudio-Open and TangoFlux. Inference code and model weights are available at this https URL. Demo samples can be found at this https URL.
>
---
#### [replaced 007] DialogueSidon: Recovering Full-Duplex Dialogue Tracks from In-the-Wild Dialogue Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DialogueSidon，解决从单通道对话中恢复全双工对话的问题。通过结合VAE和扩散模型，实现语音分离与增强，提升对话清晰度和分离质量。**

- **链接: [https://arxiv.org/pdf/2604.09344](https://arxiv.org/pdf/2604.09344)**

> **作者:** Wataru Nakata; Yuki Saito; Kazuki Yamauchi; Emiru Tsunoo; Hiroshi Saruwatari
>
> **备注:** 12 pages, 2 figures, fixed invalid link
>
> **摘要:** Full-duplex dialogue audio, in which each speaker is recorded on a separate track, is an important resource for spoken dialogue research, but is difficult to collect at scale. Most in-the-wild two-speaker dialogue is available only as degraded monaural mixtures, making it unsuitable for systems requiring clean speaker-wise signals. We propose DialogueSidon, a model for joint restoration and separation of degraded monaural two-speaker dialogue audio. DialogueSidon combines a variational autoencoder (VAE) operates on the speech self-supervised learning (SSL) model feature, which compresses SSL model features into a compact latent space, with a diffusion-based latent predictor that recovers speaker-wise latent representations from the degraded mixture. Experiments on English, multilingual, and in-the-wild dialogue datasets show that DialogueSidon substantially improves intelligibility and separation quality over a baseline, while also achieving much faster inference.
>
---
#### [replaced 008] HAFM: Hierarchical Autoregressive Foundation Model for Music Accompaniment Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出HAFM，用于生成与输入人声匹配的伴奏音乐。解决音乐伴奏生成任务，通过创新的编码方案和模型结构，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.09054](https://arxiv.org/pdf/2604.09054)**

> **作者:** Jian Zhu; Jianwei Cui; Shihao Chen; Yubang Zhang; Cheng Luo
>
> **备注:** Music Accompaniment Generation, Music Foundation Model
>
> **摘要:** We present HAFM, a system that generates instrumental music audio to accompany input vocals. Given isolated singing voice, HAFM produces a coherent instrumental accompaniment that can be directly mixed with the input to create complete music. We propose three key innovations over prior work: (1) a dual-rate codec tokenization scheme using HuBERT semantic tokens at 50\,Hz for vocals and EnCodec acoustic tokens at 75\,Hz for instrumentals, enabling time-aligned yet rate-independent modeling; (2) a three-stage hierarchical autoregressive architecture (semantic to coarse acoustic to fine acoustic) with interleaved multi-codebook prediction and classifier-free guidance; and (3) modern Transformer design choices including QK-norm, GEGLU activations, RMSNorm, and T5-style relative position bias for improved training stability and sequence generalization. Experiments on MUSDB18 demonstrate that HAFM achieves a Fréchet Audio Distance (FAD) of 2.08 on isolated vocal inputs, outperforming retrieval baselines and matching prior state-of-the-art systems with fewer parameters. The source code is available at this https URL.
>
---
#### [replaced 009] CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，解决零样本歌唱合成中旋律控制不精准的问题。提出CoMelSinger框架，通过离散编码器实现结构化旋律控制与音色分离。**

- **链接: [https://arxiv.org/pdf/2509.19883](https://arxiv.org/pdf/2509.19883)**

> **作者:** Junchuan Zhao; Wei Zeng; Tianle Lyu; Ye Wang
>
> **备注:** Published in IEEE Transactions on Audio, Speech and Language Processing (TASLP). 13 pages, 5 figures, 5 tables
>
> **摘要:** Singing Voice Synthesis (SVS) aims to generate expressive vocal performances from structured musical inputs such as lyrics and pitch sequences. While recent progress in discrete codec-based speech synthesis has enabled zero-shot generation via in-context learning, directly extending these techniques to SVS remains non-trivial due to the requirement for precise melody control. In particular, prompt-based generation often introduces prosody leakage, where pitch information is inadvertently entangled within the timbre prompt, compromising controllability. We present CoMelSinger, a zero-shot SVS framework that enables structured and disentangled melody control within a discrete codec modeling paradigm. Built on the non-autoregressive MaskGCT architecture, CoMelSinger replaces conventional text inputs with lyric and pitch tokens, preserving in-context generalization while enhancing melody conditioning. To suppress prosody leakage, we propose a coarse-to-fine contrastive learning strategy that explicitly regularizes pitch redundancy between the acoustic prompt and melody input. Furthermore, we incorporate a lightweight encoder-only Singing Voice Transcription (SVT) module to align acoustic tokens with pitch and duration, offering fine-grained frame-level supervision. Experimental results demonstrate that CoMelSinger achieves notable improvements in pitch accuracy, timbre consistency, and zero-shot transferability over competitive baselines. Audio samples are available at this https URL.
>
---
#### [replaced 010] Gradient-based Optimisation of Modulation Effects
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音频效果建模任务，旨在解决模拟调制效果（如混响、相位器等）的计算成本高和延迟问题。通过基于梯度优化的数字信号处理框架，实现低延迟、高质量的声音生成。**

- **链接: [https://arxiv.org/pdf/2601.04867](https://arxiv.org/pdf/2601.04867)**

> **作者:** Alistair Carson; Alec Wright; Stefan Bilbao
>
> **备注:** Accepted for publication in the Journal Audio Engineering Society (JAES) 2026. Original submission Dec. 2025. Revised and accepted March 2026
>
> **摘要:** Modulation effects such as phasers, flangers and chorus effects are heavily used in conjunction with the electric guitar. Machine learning based emulation of analog modulation units has been investigated in recent years, but most methods have either been limited to one class of effect or suffer from a high computational cost or latency compared to canonical digital implementations. Here, we build on previous work and present a framework for modelling flanger, chorus and phaser effects based on differentiable digital signal processing. The model is trained in the time-frequency domain, but at inference operates in the time-domain, requiring zero latency. We investigate the challenges associated with gradient-based optimisation of such effects, and show that low-frequency weighting of loss functions avoids convergence to local minima when learning delay times. We show that when trained against analog effects units, sound output from the model is in some cases perceptually indistinguishable from the reference, but challenges still remain for effects with long delay times and feedback.
>
---
#### [replaced 011] Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出Sat2Sound，解决跨模态声音地图构建问题。通过融合多模态数据，提升声音分布预测的准确性和多样性，实现精准、可解释的声音景观映射。**

- **链接: [https://arxiv.org/pdf/2505.13777](https://arxiv.org/pdf/2505.13777)**

> **作者:** Subash Khanal; Srikumar Sastry; Aayush Dhakal; Adeel Ahmad; Abby Stylianou; Nathan Jacobs
>
> **备注:** Accepted to EarthVision 2026
>
> **摘要:** We present Sat2Sound, a unified multimodal framework for geospatial soundscape understanding, designed to predict and map the distribution of sounds across the Earth's surface. Existing methods for this task rely on paired satellite images and geotagged audio samples, which often fail to capture the full diversity of sound at a location. Sat2Sound overcomes this limitation by augmenting datasets with semantically rich, vision-language model-generated soundscape descriptions, which broaden the range of possible ambient sounds represented at each location. Our framework jointly learns from audio, text descriptions of audio, satellite images, and synthetic image captions through contrastive and codebook-aligned learning, discovering a set of "soundscape concepts" shared across modalities, enabling hyper-localized, explainable soundscape mapping. Sat2Sound achieves state-of-the-art performance in cross-modal retrieval between satellite image and audio on the GeoSound and SoundingEarth benchmarks. Finally, by retrieving detailed soundscape captions that can be rendered through text-to-audio models, Sat2Sound enables location-conditioned soundscape synthesis for immersive and educational applications, even with limited computational resources. Our code and models are available at this https URL.
>
---
#### [replaced 012] Positive-Unlabelled Active Learning to Curate a Dataset for Orca Resident Interpretation
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于数据集构建任务，旨在解决海洋哺乳动物声学数据的标注问题。通过主动学习方法，构建了大规模的SRKW声学数据集，用于物种识别与生态研究。**

- **链接: [https://arxiv.org/pdf/2602.09295](https://arxiv.org/pdf/2602.09295)**

> **作者:** Bret Nestor; Bohan Yao; Jasmine Moore; Jasper Kanes
>
> **摘要:** This work presents the largest curation of Southern Resident Killer Whale (SRKW) acoustic data to date, also containing other marine mammals in their environment. We systematically search all available public archival hydrophone data within the SRKW habitat (over 30 years of audio data). The search consists of a weakly-supervised, positive-unlabelled, active learning strategy to identify all instances of marine mammals. The resulting transformer-based presence or absence classifiers outperform state-of-the-art classifiers on 3 of 4 expert-annotated datasets in terms of accuracy and energy efficiency. The fleet of WHISPER detection models range from 0.58 (0.48-0.67) AUROC with WHISPER-tiny to 0.77 (0.63-0.93) with WHISPER-large-v3. Our multiclass species classifier obtains a top-1 accuracy of 53.2\% (11 train classes, 4 test classes) and our ecotype classifier obtains a top-1 accuracy of 33.6\% (4 train classes, 5 test classes) on the DCLDE-2026 dataset. We yield 919 hours of SRKW data, 230 hours of Bigg's orca data, 1374 hours of orca data from unlabelled ecotypes, 1501 hours of humpback data, 88 hours of sea lion data, 246 hours of pacific white-sided dolphin data, and over 784 hours of unspecified marine mammal data. This SRKW dataset is larger than DCLDE-2026, Ocean Networks Canada, and OrcaSound combined. The curated species labels are available under CC-BY 4.0 license, and the corresponding audio data are available under the licenses of the original owners. The comprehensive nature of this dataset makes it suitable for unsupervised machine translation, habitat usage surveys, and conservation endeavours for this critically endangered ecotype.
>
---
