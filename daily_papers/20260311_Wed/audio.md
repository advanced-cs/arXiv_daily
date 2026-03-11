# 音频 cs.SD;  eess.AS

- **最新发布 25 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决Speech LLMs在情感识别中的零样本不确定性与人类情感主观性问题。提出VoxEmo基准，包含多语言数据集和多种提示策略，提升评估的准确性与现实感。**

- **链接: [https://arxiv.org/pdf/2603.08936](https://arxiv.org/pdf/2603.08936)**

> **作者:** Hezhao Zhang; Huang-Cheng Chou; Shrikanth Narayanan; Thomas Hain
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions.
>
---
#### [new 002] Emotion-Aware Prefix: Towards Explicit Emotion Control in Voice Conversion Models
- **分类: eess.AS**

- **简介: 该论文属于语音转换任务，旨在解决情感控制不准确的问题。提出Emotion-Aware Prefix方法，提升情感转换性能，同时保持语言完整性和语音质量。**

- **链接: [https://arxiv.org/pdf/2603.09120](https://arxiv.org/pdf/2603.09120)**

> **作者:** Haoyuan Yang; Mu Yang; Jiamin Xie; Szu-Jui Chen; John H.L. Hansen
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advances in zero-shot voice conversion have exhibited potential in emotion control, yet the performance is suboptimal or inconsistent due to their limited expressive capacity. We propose Emotion-Aware Prefix for explicit emotion control in a two-stage voice conversion backbone. We significantly improve emotion conversion performance, doubling the baseline Emotion Conversion Accuracy (ECA) from 42.40% to 85.50% while maintaining linguistic integrity and speech quality, without compromising speaker identity. Our ablation study suggests that a joint control of both sequence modulation and acoustic realization is essential to synthesize distinct emotions. Furthermore, comparative analysis verifies the generalizability of proposed method, while it provides insights on the role of acoustic decoupling in maintaining speaker identity.
>
---
#### [new 003] StuPASE: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决生成语音中存在幻觉的问题。通过改进PASE模型，提升语音质量并保持低幻觉，实现高质量语音生成。**

- **链接: [https://arxiv.org/pdf/2603.09234](https://arxiv.org/pdf/2603.09234)**

> **作者:** Xiaobin Rong; Jun Gao; Zheng Wang; Mansur Yesilbursa; Kamil Wojcicki; Jing Lu
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Achieving high perceptual quality without hallucination remains a challenge in generative speech enhancement (SE). A representative approach, PASE, is robust to hallucination but has limited perceptual quality under adverse conditions. We propose StuPASE, built upon PASE to achieve studio-level quality while retaining its low-hallucination property. First, we show that finetuning PASE with dry targets rather than targets containing simulated early reflections substantially improves dereverberation. Second, to address performance limitations under strong additive noise, we replace the GAN-based generative module in PASE with a flow-matching module, enabling studio-quality generation even under highly challenging conditions. Experiments demonstrate that StuPASE consistently produces perceptually high-quality speech while maintaining low hallucination, outperforming state-of-the-art SE methods. Audio demos are available at: this https URL.
>
---
#### [new 004] Universal Speech Content Factorization
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出USCF，用于语音内容解耦，解决开放集语音转换问题。通过线性方法分离说话人音色与语音内容，实现高效训练和零样本转换。**

- **链接: [https://arxiv.org/pdf/2603.08977](https://arxiv.org/pdf/2603.08977)**

> **作者:** Henry Li Xinyuan; Zexin Cai; Lin Zhang; Leibny Paola García-Perera; Berrak Sisman; Sanjeev Khudanpur; Nicholas Andrews; Matthew Wiesner
>
> **摘要:** We propose Universal Speech Content Factorization (USCF), a simple and invertible linear method for extracting a low-rank speech representation in which speaker timbre is suppressed while phonetic content is preserved. USCF extends Speech Content Factorization, a closed-set voice conversion (VC) method, to an open-set setting by learning a universal speech-to-content mapping via least-squares optimization and deriving speaker-specific transformations from only a few seconds of target speech. We show through embedding analysis that USCF effectively removes speaker-dependent variation. As a zero-shot VC system, USCF achieves competitive intelligibility, naturalness, and speaker similarity compared to methods that require substantially more target-speaker data or additional neural training. Finally, we demonstrate that as a training-efficient timbre-disentangled speech feature, USCF features can serve as the acoustic representation for training timbre-prompted text-to-speech models. Speech samples and code are publicly available.
>
---
#### [new 005] TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出TimberAgent，解决音频效果控制中的语义差距问题，通过纹理感知的检索方法实现可编辑的效果配置。**

- **链接: [https://arxiv.org/pdf/2603.09332](https://arxiv.org/pdf/2603.09332)**

> **作者:** Shihao He; Yihan Xia; Fang Liu; Taotao Wang; Shengli Zhang
>
> **摘要:** Digital audio workstations expose rich effect chains, yet a semantic gap remains between perceptual user intent and low-level signal-processing parameters. We study retrieval-grounded audio effect control, where the output is an editable plugin configuration rather than a finalized waveform. Our focus is Texture Resonance Retrieval (TRR), an audio representation built from Gram matrices of projected mid-level Wav2Vec2 activations. This design preserves texture-relevant co-activation structure. We evaluate TRR on a guitar-effects benchmark with 1,063 candidate presets and 204 queries. The evaluation follows Protocol-A, a cross-validation scheme that prevents train-test leakage. We compare TRR against CLAP and internal retrieval baselines (Wav2Vec-RAG, Text-RAG, FeatureNN-RAG), using min-max normalized metrics grounded in physical DSP parameter ranges. Ablation studies validate TRR's core design choices: projection dimensionality, layer selection, and projection type. A near-duplicate sensitivity analysis confirms that results are robust to trivial knowledge-base matches. TRR achieves the lowest normalized parameter error among evaluated methods. A multiple-stimulus listening study with 26 participants provides complementary perceptual evidence. We interpret these results as benchmark evidence that texture-aware retrieval is useful for editable audio effect control, while broader personalization and real-audio robustness claims remain outside the verified evidence presented here.
>
---
#### [new 006] Finetuning a Text-to-Audio Model for Room Impulse Response Generation
- **分类: eess.AS**

- **简介: 该论文属于声学模拟任务，旨在解决真实RIR数据稀缺问题。通过微调文本到音频模型生成RIR，并利用视觉语言模型构建文本描述数据，提升语音数据增强效果。**

- **链接: [https://arxiv.org/pdf/2603.09708](https://arxiv.org/pdf/2603.09708)**

> **作者:** Kirak Kim; Sungyoung Kim
>
> **备注:** 5 pages, 2 figures, submitted to Interspeech 2026
>
> **摘要:** Room Impulse Responses (RIRs) enable realistic acoustic simulation, with applications ranging from multimedia production to speech data augmentation. However, acquiring high-quality real-world RIRs is labor-intensive, and data scarcity remains a challenge for data-driven RIR generation approaches. In this paper, we propose a novel approach to RIR generation by fine-tuning a pre-trained text-to-audio model, demonstrating for the first time that large-scale generative audio priors can be effectively leveraged for the task. To address the lack of text-RIR paired data, we establish a labeling pipeline utilizing vision-language models to extract acoustic descriptions from existing image-RIR datasets. We introduce an in-context learning strategy to accommodate free-form user prompts during inference. Evaluations involving MUSHRA listening tests and downstream ASR performance demonstrate that our model generates plausible RIRs and serves as an effective tool for speech data augmentation.
>
---
#### [new 007] The Costs of Reproducibility in Music Separation Research: a Replication of Band-Split RNN
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐分离任务，旨在解决模型复现困难的问题。通过尝试复现BSRNN模型，分析其设计与训练流程，提出优化方案，并公开代码与模型以促进可复现研究。**

- **链接: [https://arxiv.org/pdf/2603.09187](https://arxiv.org/pdf/2603.09187)**

> **作者:** Paul Magron; Romain Serizel; Constance Douwes
>
> **摘要:** Music source separation is the task of isolating the instrumental tracks from a music song. Despite its spectacular recent progress, the trend towards more complex architectures and training protocols exacerbates reproducibility issues. The band-split recurrent neural networks (BSRNN) model is promising in this regard, since it yields close to state-of-the-art results on public datasets, and requires reasonable resources for training. Unfortunately, it is not straightforward to reproduce since its full code is not available. In this paper, we attempt to replicate BSRNN as closely as possible to the original paper through extensive experiments, which allows us to conduct a critical reflection on this reproducibility issue. Our contributions are three-fold. First, this study yields several insights on the model design and training pipeline, which sheds light on potential future improvements. In particular, since we were unsuccessful in reproducing the original results, we explore additional variants that ultimately yield an optimized BSRNN model, whose performance largely improves that of the original. Second, we discuss reproducibility issues from both methodological and practical perspectives. We notably underline how substantial time and energy costs could have been saved upon availability of the full pipeline. Third, our code and pre-trained models are released publicly to foster reproducible research. We hope that this study will contribute to spread awareness on the importance of reproducible research in the music separation community, and help promoting more transparent and sustainable practices.
>
---
#### [new 008] Speech-Omni-Lite: Portable Speech Interfaces for Vision-Language Models
- **分类: eess.AS**

- **简介: 该论文提出Speech-Omni-Lite，解决视觉-语言模型扩展语音能力的问题，通过轻量模块实现高效语音理解与生成，保持原有性能。**

- **链接: [https://arxiv.org/pdf/2603.09627](https://arxiv.org/pdf/2603.09627)**

> **作者:** Dehua Tao; Xuan Luo; Daxin Tan; Kai Chen; Lanqing Hong; Jing Li; Ruifeng Xu; Xiao Chen
>
> **摘要:** While large-scale omni-models have demonstrated impressive capabilities across various modalities, their strong performance heavily relies on massive multimodal data and incurs substantial computational costs. This work introduces Speech-Omni-Lite, a cost-efficient framework for extending pre-trained Visual-Language (VL) backbones with speech understanding and generation capabilities, while fully preserving the backbones' vision-language performance. Specifically, the VL backbone is equipped with two lightweight, trainable plug-and-play modules, a speech projector and a speech token generator, while keeping the VL backbone fully frozen. To mitigate the scarcity of spoken QA corpora, a low-cost data construction strategy is proposed to generate Question-Text Answer-Text-Speech (QTATS) data from existing ASR speech-text pairs, facilitating effective speech generation training. Experimental results show that, even with only thousands of hours of speech training data, Speech-Omni-Lite achieves excellent spoken QA performance, which is comparable to omni-models trained on millions of hours of speech data. Furthermore, the learned speech modules exhibit strong transferability across VL backbones.
>
---
#### [new 009] SCENEBench: An Audio Understanding Benchmark Grounded in Assistive and Industrial Use Cases
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出SCENEBench基准，用于评估音频理解能力，解决传统ASR之外的音频分析问题，涵盖背景声、噪声定位等四类任务。**

- **链接: [https://arxiv.org/pdf/2603.09853](https://arxiv.org/pdf/2603.09853)**

> **作者:** Laya Iyer; Angelina Wang; Sanmi Koyejo
>
> **备注:** Accepted to EACL 2026 (Main Conference). 10 pages, 10 figures. Camera-ready version
>
> **摘要:** Advances in large language models (LLMs) have enabled significant capabilities in audio processing, resulting in state-of-the-art models now known as Large Audio Language Models (LALMs). However, minimal work has been done to measure audio understanding beyond automatic speech recognition (ASR). This paper closes that gap by proposing a benchmark suite, SCENEBench (Spatial, Cross-lingual, Environmental, Non-speech Evaluation), that targets a broad form of audio comprehension across four real-world categories: background sound understanding, noise localization, cross-linguistic speech understanding, and vocal characterizer recognition. These four categories are selected based on understudied needs from accessibility technology and industrial noise monitoring. In addition to performance, we also measure model latency. The purpose of this benchmark suite is to assess audio beyond just what words are said - rather, how they are said and the non-speech components of the audio. Because our audio samples are synthetically constructed (e.g., by overlaying two natural audio samples), we further validate our benchmark against 20 natural audio items per task, sub-sampled from existing datasets to match our task criteria, to assess ecological validity. We assess five state-of-the-art LALMs and find critical gaps: performance varies across tasks, with some tasks performing below random chance and others achieving high accuracy. These results provide direction for targeted improvements in model capabilities.
>
---
#### [new 010] A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration
- **分类: eess.AS**

- **简介: 该论文针对语音恢复任务，解决扩散模型采样速度慢的问题，提出一种适用于插值SDE的快速求解器，减少神经网络评估次数。**

- **链接: [https://arxiv.org/pdf/2603.09508](https://arxiv.org/pdf/2603.09508)**

> **作者:** Bunlong Lay; Timo Gerkmann
>
> **摘要:** Diffusion Probabilistic Models (DPMs) are a well-established class of diffusion models for unconditional image generation, while SGMSE+ is a well-established conditional diffusion model for speech enhancement. One of the downsides of diffusion models is that solving the reverse process requires many evaluations of a large Neural Network. Although advanced fast sampling solvers have been developed for DPMs, they are not directly applicable to models such as SGMSE+ due to differences in their diffusion processes. Specifically, DPMs transform between the data distribution and a standard Gaussian distribution, whereas SGMSE+ interpolates between the target distribution and a noisy observation. This work first develops a formalism of interpolating Stochastic Differential Equations (iSDEs) that includes SGMSE+, and second proposes a solver for iSDEs. The proposed solver enables fast sampling with as few as 10 Neural Network evaluations across multiple speech restoration tasks.
>
---
#### [new 011] End-to-End Direction-Aware Keyword Spotting with Spatial Priors in Noisy Environments
- **分类: eess.AS**

- **简介: 该论文属于语音识别中的关键词检测任务，旨在提升嘈杂环境下的检测性能。通过多通道端到端框架，结合空间特征和方向先验，增强噪声鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.09505](https://arxiv.org/pdf/2603.09505)**

> **作者:** Rui Wang; Zhifei Zhang; Yu Gao; Xiaofeng Mou; Yi Xu
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** Keyword spotting (KWS) is crucial for many speech-driven applications, but robust KWS in noisy environments remains challenging. Conventional systems often rely on single-channel inputs and a cascaded pipeline separating front-end enhancement from KWS. This precludes joint optimization, inherently limiting performance. We present an end-to-end multi-channel KWS framework that exploits spatial cues to improve noise robustness. A spatial encoder learns inter-channel features, while a spatial embedding injects directional priors; the fused representation is processed by a streaming backbone. Experiments in simulated noisy conditions across multiple signal-to-noise ratios (SNRs) show that spatial modeling and directional priors each yield clear gains over baselines, with their combination achieving the best results. These findings validate end-to-end multi-channel spatial modeling, indicating strong potential for the target-speaker-aware detection in complex acoustic scenarios.
>
---
#### [new 012] A Semi-spontaneous Dutch Speech Dataset for Speech Enhancement and Speech Recognition
- **分类: eess.AS**

- **简介: 该论文提出DRES数据集，用于评估语音增强和语音识别模型在真实噪声环境下的表现。旨在解决实际场景中语音处理的挑战。**

- **链接: [https://arxiv.org/pdf/2603.09725](https://arxiv.org/pdf/2603.09725)**

> **作者:** Dimme de Groot; Yuanyuan Zhang; Jorge Martinez; Odette Scharenborg
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** We present DRES: a 1.5-hour Dutch realistic elicited (semi-spontaneous) speech dataset from 80 speakers recorded in noisy, public indoor environments. DRES was designed as a test set for the evaluation of state-of-the-art (SOTA) automatic speech recognition (ASR) and speech enhancement (SE) models in a real-world scenario: a person speaking in a public indoor space with background talkers and noise. The speech was recorded with a four-channel linear microphone array. In this work we evaluate the speech quality of five well-known single-channel SE algorithms and the recognition performance of eight SOTA off-the-shelf ASR models before and after applying SE on the speech of DRES. We found that five out of the eight ASR models have WERs lower than 22% on DRES, despite the challenging conditions. In contrast to recent work, we did not find a positive effect of modern single-channel SE on ASR performance, emphasizing the importance of evaluating in realistic conditions.
>
---
#### [new 013] Physics-Informed Neural Engine Sound Modeling with Differentiable Pulse-Train Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频合成任务，旨在更准确地模拟发动机声音。通过建模脉冲形状和时序结构，提出PTR模型，提升谐波重建效果并融入物理原理。**

- **链接: [https://arxiv.org/pdf/2603.09391](https://arxiv.org/pdf/2603.09391)**

> **作者:** Robin Doerfler; Lonce Wyse
>
> **备注:** Preprint. 5 pages, 2 figures. Audio examples, code, and model weights available online
>
> **摘要:** Engine sounds originate from sequential exhaust pressure pulses rather than sustained harmonic oscillations. While neural synthesis methods typically aim to approximate the resulting spectral characteristics, we propose directly modeling the underlying pulse shapes and temporal structure. We present the Pulse-Train-Resonator (PTR) model, a differentiable synthesis architecture that generates engine audio as parameterized pulse trains aligned to engine firing patterns and propagates them through recursive Karplus-Strong resonators simulating exhaust acoustics. The architecture integrates physics-informed inductive biases including harmonic decay, thermodynamic pitch modulation, valve-dynamics envelopes, exhaust system resonances and derived engine operating modes such as throttle operation and deceleration fuel cutoff (DCFO). Validated on three diverse engine types totaling 7.5 hours of audio, PTR achieves a 21% improvement in harmonic reconstruction and a 5.7% reduction in total loss over a harmonic-plus-noise baseline model, while providing interpretable parameters corresponding to physical phenomena. Complete code, model weights, and audio examples are openly available.
>
---
#### [new 014] Acoustic and Semantic Modeling of Emotion in Spoken Language
- **分类: eess.AS**

- **简介: 该论文属于情感语音理解与生成任务，旨在提升AI对情感的识别与合成能力。通过结合声学与语义信息，提出预训练方法、对话情感识别架构及无参考语音转换框架，解决情感建模与迁移问题。**

- **链接: [https://arxiv.org/pdf/2603.09212](https://arxiv.org/pdf/2603.09212)**

> **作者:** Soumya Dutta
>
> **备注:** PhD thesis
>
> **摘要:** Emotions play a central role in human communication, shaping trust, engagement, and social interaction. As artificial intelligence systems powered by large language models become increasingly integrated into everyday life, enabling them to reliably understand and generate human emotions remains an important challenge. While emotional expression is inherently multimodal, this thesis focuses on emotions conveyed through spoken language and investigates how acoustic and semantic information can be jointly modeled to advance both emotion understanding and emotion synthesis from speech. The first part of the thesis studies emotion-aware representation learning through pre-training. We propose strategies that incorporate acoustic and semantic supervision to learn representations that better capture affective cues in speech. A speech-driven supervised pre-training framework is also introduced to enable large-scale emotion-aware text modeling without requiring manually annotated text corpora. The second part addresses emotion recognition in conversational settings. Hierarchical architectures combining cross-modal attention and mixture-of-experts fusion are developed to integrate acoustic and semantic information across conversational turns. Finally, the thesis introduces a textless and non-parallel speech-to-speech framework for emotion style transfer that enables controllable emotional transformations while preserving speaker identity and linguistic content. The results demonstrate improved emotion transfer and show that style-transferred speech can be used for data augmentation to improve emotion recognition.
>
---
#### [new 015] Paralinguistic Emotion-Aware Validation Timing Detection in Japanese Empathetic Spoken Dialogue
- **分类: cs.SD**

- **简介: 该论文属于情感对话中的验证时机检测任务，旨在通过非语言语音线索和情绪信息，准确判断何时进行情感验证，以提升人机互动的共情能力。**

- **链接: [https://arxiv.org/pdf/2603.09307](https://arxiv.org/pdf/2603.09307)**

> **作者:** Zi Haur Pang; Yahui Fu; Yuan Gao; Tatsuya Kawahara
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Emotional Validation is a psychotherapy communication technique that involves recognizing, understanding, and explicitly acknowledging another person's feelings and actions, which strengthens alliance and reduces negative affect. To maximize the emotional support provided by validation, it is crucial to deliver it with appropriate timing and frequency. This study investigates validation timing detection from the speech perspective. Leveraging both paralinguistic and emotional information, we propose a paralinguistic- and emotion-aware model for validation timing detection without relying on textual context. Specifically, we first conduct continued self-supervised training and fine-tuning on different HuBERT backbones to obtain (i) a paralinguistics-aware Self-Supervised Learning (SSL) encoder and (ii) a multi-task speech emotion classification encoder. We then fuse these encoders and further fine-tune the combined model on the downstream validation timing detection task. Experimental evaluations on the TUT Emotional Storytelling Corpus (TESC) compare multiple models, fusion mechanisms, and training strategies, and demonstrate that the proposed approach achieves significant improvements over conventional speech baselines. Our results indicate that non-linguistic speech cues, when integrated with affect-related representations, carry sufficient signal to decide when validation should be expressed, offering a speech-first pathway toward more empathetic human-robot interaction.
>
---
#### [new 016] EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐结构分割任务，针对EDM模型效果差的问题，提出EDMFormer模型，结合特定数据集和结构先验，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2603.08759](https://arxiv.org/pdf/2603.08759)**

> **作者:** Sahal Sajeer; Krish Patel; Oscar Chung; Joel Song Bae
>
> **备注:** Published in CUCAI 2026 conference proceedings
>
> **摘要:** Music structure segmentation is a key task in audio analysis, but existing models perform poorly on Electronic Dance Music (EDM). This problem exists because most approaches rely on lyrical or harmonic similarity, which works well for pop music but not for EDM. EDM structure is instead defined by changes in energy, rhythm, and timbre, with different sections such as buildup, drop, and breakdown. We introduce EDMFormer, a transformer model that combines self-supervised audio embeddings using an EDM-specific dataset and taxonomy. We release this dataset as EDM-98: a group of 98 professionally annotated EDM tracks. EDMFormer improves boundary detection and section labelling compared to existing models, particularly for drops and buildups. The results suggest that combining learned representations with genre-specific data and structural priors is effective for EDM and could be applied to other specialized music genres or broader audio domains.
>
---
#### [new 017] MUGEN: Evaluating and Improving Multi-audio Understanding of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于多音频理解任务，旨在评估和提升大音频语言模型的多音频处理能力。针对模型在多音频输入下的性能下降问题，提出MUGEN基准和优化策略。**

- **链接: [https://arxiv.org/pdf/2603.09714](https://arxiv.org/pdf/2603.09714)**

> **作者:** Chih-Kai Yang; Yun-Shao Tsai; Yu-Kai Guo; Ping-Le Tsai; Yen-Ting Piao; Hung-Wei Chen; Ting-Lin Hsiao; Yun-Man Hsu; Ke-Han Lu; Hung-yi Lee
>
> **备注:** 6 pages, 3 figures, 3 tables. Dataset: this https URL
>
> **摘要:** While multi-audio understanding is critical for large audio-language models (LALMs), it remains underexplored. We introduce MUGEN, a comprehensive benchmark evaluating this capability across speech, general audio, and music. Our experiments reveal consistent weaknesses in multi-audio settings, and performance degrades sharply as the number of concurrent audio inputs increases, identifying input scaling as a fundamental bottleneck. We further investigate training-free strategies and observe that Audio-Permutational Self-Consistency, which diversifies the order of audio candidates, helps models form more robust aggregated predictions, yielding up to 6.28% accuracy gains. Combining this permutation strategy with Chain-of-Thought further improves performance to 6.74%. These results expose blind spots in current LALMs and provide a foundation for evaluating complex auditory comprehension.
>
---
#### [new 018] Trade-offs Between Capacity and Robustness in Neural Audio Codecs for Adversarially Robust Speech Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究音频编码器在对抗攻击下的鲁棒性与容量的权衡问题，通过调整RVQ深度探索其对语音识别的影响。**

- **链接: [https://arxiv.org/pdf/2603.09034](https://arxiv.org/pdf/2603.09034)**

> **作者:** Jordan Prescott; Thanathai Lertpetchpun; Shrikanth Narayanan
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Adversarial perturbations exploit vulnerabilities in automatic speech recognition (ASR) systems while preserving human perceived linguistic content. Neural audio codecs impose a discrete bottleneck that can suppress fine-grained signal variations associated with adversarial noise. We examine how the granularity of this bottleneck, controlled by residual vector quantization (RVQ) depth, shapes adversarial robustness. We observe a non-monotonic trade-off under gradient-based attacks: shallow quantization suppresses adversarial perturbations but degrades speech content, while deeper quantization preserves both content and perturbations. Intermediate depths balance these effects and minimize transcription error. We further show that adversarially induced changes in discrete codebook tokens strongly correlate with transcription error. These gains persist under adaptive attacks, where neural codec configurations outperform traditional compression defenses.
>
---
#### [new 019] How Contrastive Decoding Enhances Large Audio Language Models?
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型优化任务，旨在解决Contrastive Decoding（CD）效果不明确的问题。通过评估四种策略，发现音频相关方法更有效，并提出过渡矩阵分析误差变化。**

- **链接: [https://arxiv.org/pdf/2603.09232](https://arxiv.org/pdf/2603.09232)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Yi-Cheng Lin; Hung-yi Lee
>
> **备注:** Submitted to INTERSPEECH 2026. Code and additional analysis results are provided in our repository: this https URL
>
> **摘要:** While Contrastive Decoding (CD) has proven effective at enhancing Large Audio Language Models (LALMs), the underlying mechanisms driving its success and the comparative efficacy of different strategies remain unclear. This study systematically evaluates four distinct CD strategies across diverse LALM architectures. We identify Audio-Aware Decoding and Audio Contrastive Decoding as the most effective methods. However, their impact varies significantly by model. To explain this variability, we introduce a Transition Matrix framework to map error pattern shifts during inference. Our analysis demonstrates that CD reliably rectifies errors in which models falsely claim an absence of audio or resort to uncertainty-driven guessing. Conversely, it fails to correct flawed reasoning or confident misassertions. Ultimately, these findings provide a clear guideline for determining which LALM architectures are most suitable for CD enhancement based on their baseline error profiles.
>
---
#### [new 020] Gender Fairness in Audio Deepfake Detection: Performance and Disparity Analysis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决模型在性别上的公平性问题。通过分析不同性别下的检测性能差异，揭示了传统指标的局限性，并引入公平性指标以提升系统的公正性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.09007](https://arxiv.org/pdf/2603.09007)**

> **作者:** Aishwarya Fursule; Shruti Kshirsagar; Anderson R. Avila
>
> **备注:** 6 pages, 3 Figures
>
> **摘要:** Audio deepfake detection aims to detect real human voices from those generated by Artificial Intelligence (AI) and has emerged as a significant problem in the field of voice biometrics systems. With the ever-improving quality of synthetic voice, the probability of such a voice being exploited for illicit practices like identity thest and impersonation increases. Although significant progress has been made in the field of Audio Deepfake Detection in recent times, the issue of gender bias remains underexplored and in its nascent stage In this paper, we have attempted a thorough analysis of gender dependent performance and fairness in audio deepfake detection models. We have used the ASVspoof 5 dataset and train a ResNet-18 classifier and evaluate detection performance across four different audio features, and compared the performance with baseline AASIST model. Beyond conventional metrics such as Equal Error Rate (EER %), we incorporated five established fairness metrics to quantify gender disparities in the model. Our results show that even when the overall EER difference between genders appears low, fairness-aware evaluation reveals disparities in error distribution that are obscured by aggregate performance measures. These findings demonstrate that reliance on standard metrics is unreliable, whereas fairness metrics provide critical insights into demographic-specific failure modes. This work highlights the importance of fairness-aware evaluation for developing a more equitable, robust, and trustworthy audio deepfake detection system.
>
---
#### [new 021] Distributed Multichannel Wiener Filtering for Wireless Acoustic Sensor Networks
- **分类: eess.AS; cs.IT; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决无线声学传感器网络中节点间协作的信号估计问题。提出非迭代的dMWF算法，实现更优的分布式语音增强效果。**

- **链接: [https://arxiv.org/pdf/2603.09735](https://arxiv.org/pdf/2603.09735)**

> **作者:** Paul Didier; Toon van Waterschoot; Simon Doclo; Jörg Bitzer; Pourya Behmandpoor; Henri Gode; Marc Moonen
>
> **摘要:** In a wireless acoustic sensor network (WASN), devices (i.e., nodes) can collaborate through distributed algorithms to collectively perform audio signal processing tasks. This paper focuses on the distributed estimation of node-specific desired speech signals using network-wide Wiener filtering. The objective is to match the performance of a centralized system that would have access to all microphone signals, while reducing the communication bandwidth usage of the algorithm. Existing solutions, such as the distributed adaptive node-specific signal estimation (DANSE) algorithm, converge towards the multichannel Wiener filter (MWF) which solves a centralized linear minimum mean square error (LMMSE) signal estimation problem. However, they do so iteratively, which can be slow and impractical. Many solutions also assume that all nodes observe the same set of sources of interest, which is often not the case in practice. To overcome these limitations, we propose the distributed multichannel Wiener filter (dMWF) for fully connected WASNs. The dMWF is non-iterative and optimal even when nodes observe different sets of sources. In this algorithm, nodes exchange neighbor-pair-specific, low-dimensional (fused) signals estimating the contribution of sources observed by both nodes in the pair. We formally prove the optimality of dMWF and demonstrate its performance in simulated speech enhancement experiments. The proposed algorithm is shown to outperform DANSE in terms of objective metrics after short operation times, highlighting the benefit of its iterationless design.
>
---
#### [new 022] EmoSURA: Towards Accurate Evaluation of Detailed and Long-Context Emotional Speech Captions
- **分类: cs.SD**

- **简介: 该论文属于情感语音描述评估任务，解决传统评估方法在长文本和语义细节上的不足。提出EmoSURA框架，通过原子单位验证提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.09820](https://arxiv.org/pdf/2603.09820)**

> **作者:** Xin Jing; Andreas Triantafyllopoulos; Jiadong Wang; Shahin Amiriparian; Jun Luo; Björn Schuller
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advancements in speech captioning models have enabled the generation of rich, fine-grained captions for emotional speech. However, the evaluation of such captions remains a critical bottleneck: traditional N-gram metrics fail to capture semantic nuances, while LLM judges often suffer from reasoning inconsistency and context-collapse when processing long-form descriptions. In this work, we propose EmoSURA, a novel evaluation framework that shifts the paradigm from holistic scoring to atomic verification. EmoSURA decomposes complex captions into Atomic Perceptual Units, which are self-contained statements regarding vocal or emotional attributes, and employs an audio-grounded verification mechanism to validate each unit against the raw speech signal. Furthermore, we address the scarcity of standardized evaluation resources by introducing SURABench, a carefully balanced and stratified benchmark. Our experiments show that EmoSURA achieves a positive correlation with human judgments, offering a more reliable assessment for long-form captions compared to traditional metrics, which demonstrated negative correlations due to their sensitivity to caption length.
>
---
#### [new 023] Fish Audio S2 Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文介绍Fish Audio S2，一个开源文本到语音系统，解决多说话人、多轮生成及指令跟随问题。通过多阶段训练和数据管道提升性能，并提供可部署的推理引擎。**

- **链接: [https://arxiv.org/pdf/2603.08823](https://arxiv.org/pdf/2603.08823)**

> **作者:** Shijia Liao; Yuxuan Wang; Songting Liu; Yifan Cheng; Ruoyi Zhang; Tianyu Li; Shidong Li; Yisheng Zheng; Xingwei Liu; Qingzheng Wang; Zhizhuo Zhou; Jiahua Liu; Xin Chen; Dawei Han
>
> **摘要:** We introduce Fish Audio S2, an open-sourced text-to-speech system featuring multi-speaker, multi-turn generation, and, most importantly, instruction-following control via natural-language descriptions. To scale training, we develop a multi-stage training recipe together with a staged data pipeline covering video captioning and speech captioning, voice-quality assessment, and reward modeling. To push the frontier of open-source TTS, we release our model weights, fine-tuning code, and an SGLang-based inference engine. The inference engine is production-ready for streaming, achieving an RTF of 0.195 and a time-to-first-audio below 100 this http URL code and weights are available on GitHub (this https URL) and Hugging Face (this https URL). We highly encourage readers to visit this https URL to try custom voices.
>
---
#### [new 024] Can You Hear, Localize, and Segment Continually? An Exemplar-Free Continual Learning Benchmark for Audio-Visual Segmentation
- **分类: cs.CV; eess.AS**

- **简介: 该论文属于音频-视觉分割任务，解决动态环境中模型持续学习的问题。提出首个无样本的持续学习基准和ATLAS方法，缓解遗忘，提升长期感知能力。**

- **链接: [https://arxiv.org/pdf/2603.08967](https://arxiv.org/pdf/2603.08967)**

> **作者:** Siddeshwar Raghavan; Gautham Vinod; Bruce Coburn; Fengqing Zhu
>
> **摘要:** Audio-Visual Segmentation (AVS) aims to produce pixel-level masks of sound producing objects in videos, by jointly learning from audio and visual signals. However, real-world environments are inherently dynamic, causing audio and visual distributions to evolve over time, which challenge existing AVS systems that assume static training settings. To address this gap, we introduce the first exemplar-free continual learning benchmark for Audio-Visual Segmentation, comprising four learning protocols across single-source and multi-source AVS datasets. We further propose a strong baseline, ATLAS, which uses audio-guided pre-fusion conditioning to modulate visual feature channels via projected audio context before cross-modal attention. Finally, we mitigate catastrophic forgetting by introducing Low-Rank Anchoring (LRA), which stabilizes adapted weights based on loss sensitivity. Extensive experiments demonstrate competitive performance across diverse continual scenarios, establishing a foundation for lifelong audio-visual perception. Code is available at${}^{*}$\footnote{Paper under review} - \hyperlink{this https URL}{this https URL} \keywords{Continual Learning \and Audio-Visual Segmentation \and Multi-Modal Learning}
>
---
#### [new 025] SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出SPAR-K框架，用于加速语音语言模型的推理。针对语音序列长导致的高计算成本问题，通过定期全深度刷新提升效率，保持问答准确率和感知质量。属于语音语言模型优化任务。**

- **链接: [https://arxiv.org/pdf/2603.09215](https://arxiv.org/pdf/2603.09215)**

> **作者:** Hsiao-Ying Huang; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** 6 pages, 1 figures, 2 tables
>
> **摘要:** Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality. SPAR-K introduces a speech alternating-depth schedule: most speech positions exit at a fixed intermediate layer, while periodic full-depth "refresh" steps mitigate distribution shift due to early exit. We evaluate our framework using Step-Audio-2-mini and GLM-4-Voice across four datasets spanning reasoning, factual QA, and dialogue tasks, measuring performance in terms of ASR transcription accuracy and perceptual quality. Experimental results demonstrate that SPAR-K largely preserves question-answering accuracy with a maximum accuracy drop of 0.82\% while reducing average speech decoding depth by up to 11\% on Step-Audio-2-mini and 5\% on GLM-4-Voice, both with negligible changes in MOS and WER and no auxiliary computation overhead. We further demonstrate that confidence-based early exit strategies, widely used in text LLMs, are suboptimal for SLMs, highlighting that the unique statistical nature of speech tokens necessitates a specialized early exit design.
>
---
## 更新

#### [replaced 001] Textless and Non-Parallel Speech-to-Speech Emotion Style Transfer
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感风格迁移任务，解决在无文本和非平行数据下迁移情感特征的问题。提出S2S-ZEST框架，保留说话人和内容的同时转移情感。**

- **链接: [https://arxiv.org/pdf/2505.17655](https://arxiv.org/pdf/2505.17655)**

> **作者:** Soumya Dutta; Avni Jain; Sriram Ganapathy
>
> **备注:** 11 pages, 10 figures, 6 tables
>
> **摘要:** Given a pair of source and reference speech recordings, speech-to-speech (S2S) emotion style transfer involves the generation of an output speech that mimics the emotion characteristics of the reference while preserving the content and speaker attributes of the source. In this paper, we propose a speech-to-speech zero-shot emotion style transfer framework, termed S2S Zero-shot Emotion Style Transfer (S2S-ZEST), that enables the transfer of emotional attributes from the reference to the source while retaining the speaker identity and speech content. The S2S-ZEST framework consists of an analysis-synthesis pipeline in which the analysis module extracts semantic tokens, speaker representations, and emotion embeddings from speech. Using these representations, a pitch contour estimator and a duration predictor are learned. Further, a synthesis module is designed to generate speech based on the input representations and the derived factors. The analysis-synthesis pipeline is trained using an auto-encoding objective to enable efficient resynthesis during inference. For S2S emotion style transfer, the emotion embedding extracted from the reference speech along with the remaining representations from the source speech are used in the synthesis module to generate the style-transferred speech. In our experiments, we evaluate the converted speech on content and speaker preservation (with respect to the source) as well as on the effectiveness of the emotion style transfer (with respect to the reference). The proposed framework demonstrates improved emotion style transfer performance over prior methods in a textless and non-parallel setting. We also illustrate the application of the proposed work for data augmentation in emotion recognition tasks.
>
---
#### [replaced 002] Evaluating pretrained speech embedding systems for dysarthria detection across heterogenous datasets
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在评估预训练语音嵌入系统在不同数据集上检测构音障碍的效果。针对数据量小和不平衡的问题，通过交叉验证和对比分析，验证了模型性能及泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.19946](https://arxiv.org/pdf/2509.19946)**

> **作者:** Lovisa Wihlborg; Jemima Goodall; David Wheatley; Jacob J. Webber; Johnny Tam; Christine Weaver; Suvankar Pal; Siddharthan Chandran; Sohan Seth; Oliver Watts; Cassia Valentini-Botinhao
>
> **备注:** Accepted to ICASSP 2026. This work is supported by NEURii, a collaborative partnership involving the University of Edinburgh, Gates Ventures, Eisai, LifeArc and Health Data Research UK (HDR UK)
>
> **摘要:** We present a comprehensive evaluation of pretrained speech embedding systems for the detection of dysarthric speech using existing accessible data. Dysarthric speech datasets are often small and can suffer from recording biases as well as data imbalance. To address these we selected a range of datasets covering related conditions and adopt the use of several cross-validations runs to estimate the chance level. To certify that results are above chance, we compare the distribution of scores across these runs against the distribution of scores of a carefully crafted null hypothesis. In this manner, we evaluate 17 publicly available speech embedding systems across 6 different datasets, reporting the cross-validation performance on each. We also report cross-dataset results derived when training with one particular dataset and testing with another. We observed that within-dataset results vary considerably depending on the dataset, regardless of the embedding used, raising questions about which datasets should be used for benchmarking. We found that cross-dataset accuracy is, as expected, lower than within-dataset, highlighting challenges in the generalization of the systems. These findings have important implications for the clinical validity of systems trained and tested on the same dataset.
>
---
#### [replaced 003] WhisperVC: Decoupled Cross-Domain Alignment and Speech Generation for Low-Resource Whisper-to-Normal Conversion
- **分类: eess.AS**

- **简介: 该论文提出WhisperVC，解决低资源环境下耳语到正常语音的转换问题，通过三阶段框架实现语义对齐与语音生成分离。**

- **链接: [https://arxiv.org/pdf/2511.01056](https://arxiv.org/pdf/2511.01056)**

> **作者:** Dong Liu; Juan Liu; Wei Ju; Yao Tian; Ming Li
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Whispered speech lacks vocal-fold excitation, making intelligible conversion challenging. We propose WhisperVC, a three-stage framework for low-resource whisper-to-normal (W2N) conversion that decouples cross-domain alignment from speech generation. Stage 1 uses limited paired whisper-normal data with a content encoder and a Conformer-based variational autoencoder (VAE) with soft-DTW alignment to learn domain-invariant semantic representations. Stage 2, trained only on normal speech, employs a Length-Channel Aligner and a two-stage speaker-conditioned mel generator for timbre and prosody modeling. Stage 3 fine-tunes a HiFi-GAN vocoder for waveform synthesis. Experimental results on AISHELL6-Whisper show competitive quality (DNSMOS 3.07, UTMOS 2.83, CER 16.93%) and WavLM speaker similarity (0.95). The framework also supports privacy-preserving communication as well as non-vocal communication and a rehabilitation tool for post-surgical vocal-fold patients. Samples are available online.
>
---
#### [replaced 004] LARA-Gen: Enabling Continuous Emotion Control for Music Generation Models via Latent Affective Representation Alignment
- **分类: cs.SD**

- **简介: 该论文属于音乐生成任务，旨在解决情感控制不足的问题。通过LARA-Gen框架实现连续情感控制，提升音乐生成的情感准确性与质量。**

- **链接: [https://arxiv.org/pdf/2510.05875](https://arxiv.org/pdf/2510.05875)**

> **作者:** Jiahao Mei; Xuenan Xu; Zeyu Xie; Zihao Zheng; Ye Tao; Yue Ding; Mengyue Wu
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent advances in text-to-music models have enabled coherent music generation from text prompts, yet fine-grained emotional control remains unresolved. We introduce LARA-Gen, a framework for continuous emotion control that aligns the internal hidden states with an external music understanding model through Latent Affective Representation Alignment (LARA), enabling effective training. In addition, we design an emotion control module based on a continuous valence-arousal space, disentangling emotional attributes from textual content and bypassing the bottlenecks of text-based prompting. Furthermore, we establish a benchmark with a curated test set and a robust Emotion Predictor, facilitating objective evaluation of emotional controllability in music generation. Extensive experiments demonstrate that LARA-Gen achieves continuous, fine-grained control of emotion and significantly outperforms baselines in both emotion adherence and music quality. Generated samples are available at this https URL.
>
---
#### [replaced 005] VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.SD**

- **简介: 该论文提出VSSFlow，统一解决视频生成声音和视觉文本转语音任务。针对传统方法分离处理的不足，通过联合学习实现高效整合，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2509.24773](https://arxiv.org/pdf/2509.24773)**

> **作者:** Xin Cheng; Yuyue Wang; Xihua Wang; Yihan Wu; Kaisi Guan; Yijing Chen; Peng Zhang; Xiaojiang Liu; Meng Cao; Ruihua Song
>
> **备注:** Paper Under Review
>
> **摘要:** Video-conditioned audio generation, including Video-to-Sound (V2S) and Visual Text-to-Speech (VisualTTS), has traditionally been treated as distinct tasks, leaving the potential for a unified generative framework largely underexplored. In this paper, we bridge this gap with VSSFlow, a unified flow-matching framework that seamlessly solve both problems. To effectively handle multiple input signals within a Diffusion Transformer (DiT) architecture, we propose a disentangled condition aggregation mechanism leveraging distinct intrinsic properties of attention layers: cross-attention for semantic conditions, and self-attention for temporally-intensive conditions. Besides, contrary to the prevailing belief that joint training for the two tasks leads to performance degradation, we demonstrate that VSSFlow maintains superior performance during end-to-end joint learning process. Furthermore, we use a straightforward feature-level data synthesis method, demonstrating that our framework provides a robust foundation that easily adapts to joint sound and speech generation using synthetic data. Extensive experiments on V2S, VisualTTS and joint generation benchmarks show that VSSFlow effectively unifies these tasks and surpasses state-of-the-art domain-specific baselines, underscoring the critical potential of unified generative models. Project page: this https URL
>
---
#### [replaced 006] Latent Speech-Text Transformer
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出LST模型，解决语音与文本模态不平衡问题，通过聚合语音标记为潜在块，提升计算效率并增强跨模态对齐。任务为语音-文本生成与理解。**

- **链接: [https://arxiv.org/pdf/2510.06195](https://arxiv.org/pdf/2510.06195)**

> **作者:** Yen-Ju Lu; Yashesh Gaur; Wei Zhou; Benjamin Muller; Jesus Villalba; Najim Dehak; Luke Zettlemoyer; Gargi Ghosh; Mike Lewis; Srinivasan Iyer; Duc Le
>
> **备注:** Accepted to ICLR 2026 (Oral)
>
> **摘要:** Auto-regressive speech-text models pre-trained on interleaved text tokens and discretized speech tokens demonstrate strong speech understanding and generation, yet remain substantially less compute-efficient than text LLMs, partly due to the much longer sequences of speech tokens relative to text. This modality imbalance disproportionately allocates pre-training and inference compute to speech, potentially hindering effective cross-modal alignment and slowing performance scaling by orders of magnitude. We introduce the Latent Speech-Text Transformer (LST), which aggregates speech tokens into latent speech patches that serve as higher-level autoregressive units. This design aligns the sequence-modeling granularity between speech and text while improving computational efficiency. The resulting patches can align with textual units to facilitate cross-modal knowledge transfer and compactly capture recurring acoustic patterns such as silence. Across story-completion benchmarks under both compute-controlled and data-controlled settings, LST consistently improves speech accuracy while also improving text performance, achieving up to +6.5% absolute gain on speech HellaSwag in compute-controlled training (+5.3% in data-controlled training). Under compute-controlled scaling from 420M to 1.8B parameters in a near compute-optimal regime, gains grow with scale, and improvements persist up to 7B parameters under fixed-token budgets. These benefits extend to downstream tasks: LST stabilizes ASR adaptation and reduces the effective autoregressive sequence length during ASR and TTS inference, lowering computational cost without degrading reconstruction quality. The code is available at this https URL.
>
---
#### [replaced 007] Head, posture, and full-body gestures in unscripted dyadic conversations in noise
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文研究噪声环境下对话中的头部、身体和手势行为，探讨其在交流中的作用。属于人机交互任务，旨在解决噪声对非语言交流影响的问题。通过实验分析手势与语音同步性及变化。**

- **链接: [https://arxiv.org/pdf/2512.03636](https://arxiv.org/pdf/2512.03636)**

> **作者:** Ľuboš Hládek; Bernhard U. Seeber
>
> **备注:** 7 figures, 12 tables, 36 pages. MS heavily revised for clarity, discussion part extended. Annotation data for one participant was revised - some missing labels were added to the annotation
>
> **摘要:** Visual prosody may be critical for communication success in face-to-face conversations in noisy settings. Here, we explore the involvement of hand, head, and whole-body movements, as well as gesturing quality, in dyadic conversations in noisy settings. We hypothesize that increasing background noise would alter the frequency of conversation-related movements to support the roles of the speaker and the listener. Specifically, talkers may increase gesticulation and thus the use of hand, head, trunk, or leg movements more often, while listeners may increase backchanneling or head and trunk movements to improve the signal-to-noise ratio. Additionally, we test whether the synchrony between speech and hand gestures is affected by background noise. Here, pairs of normal hearing participants (n=8) stood in an audiovisual virtual environment while talking freely. The conversational movements were described using a newly developed labeling system with categories that respect their communicative function. The results showed higher gesturing rate during speaking than during listening. Increased levels of background noise led to increased hand-gesture complexity, modulation of head movements, and a change in trunk movements. People spoke 0.7 dB - 1.4 dB louder during hand gesturing in comparison to times with static drop posture but this was unrelated to presence of background noise. The analysis of hand-speech synchrony showed a modest decrease in synchrony for moderate noise level. People adapt their communicative behavior to increased background noise levels by increases in speech production levels and gesturing which may drive additional increase in speech production due to biomechanical coupling; listeners may increase backchanneling to support the exchange and their own signal-to-noise ratio. The synchrony analysis may reflect motivational factors of communication in noisy environments.
>
---
#### [replaced 008] Scalable Neural Vocoder from Range-Null Space Decomposition
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决神经声码器的建模不透明、适应性差和参数性能平衡问题。通过引入时频域分解框架，提升模型性能与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.08574](https://arxiv.org/pdf/2603.08574)**

> **作者:** Andong Li; Tong Lei; Zhihang Sun; Rilin Chen; Xiaodong Li; Dong Yu; Chengshi Zheng
>
> **备注:** 30 pages, 30 figures, 21 tables, Extension journal
>
> **摘要:** Although deep neural networks have facilitated significant progress of neural vocoders in recent years, they usually suffer from intrinsic challenges like opaque modeling, inflexible retraining under different input configurations, and parameter-performance trade-off. These inherent hurdles can heavily impede the development of this field. To resolve these problems, in this paper, we propose a novel neural vocoder in the time-frequency (T-F) domain. Specifically, we bridge the connection between the classical range-null decomposition (RND) theory and the vocoder task, where the reconstruction of the target spectrogram is formulated into the superimposition between range-space and null-space. The former aims to project the representation in the original mel-domain into the target linear-scale domain, and the latter can be instantiated via neural networks to further infill the spectral details. To fully leverage the spectrum prior, an elaborate dual-path framework is devised, where the spectrum is hierarchically encoded and decoded, and the cross- and narrow-band modules are leveraged for effectively modeling along sub-band and time dimensions. To enable inference under various configurations, we propose a simple yet effective strategy, which transforms the multi-condition adaption in the inference stage into the data augmentation in the training stage. Comprehensive experiments are conducted on various benchmarks. Quantitative and qualitative results show that while enjoying lightweight network structure and scalable inference paradigm, the proposed framework achieves state-ofthe-art performance among existing advanced methods. Code is available at this https URL.
>
---
#### [replaced 009] Benchmarking Humans and Machines on Complex Multilingual Speech Understanding Tasks
- **分类: eess.AS**

- **简介: 该论文研究多语言语音理解任务，探讨人类与机器在复杂语音场景下的表现差异。重点解决多语种下语音识别与注意力机制的问题，通过实验对比分析人类与大语言模型的性能。**

- **链接: [https://arxiv.org/pdf/2509.17965](https://arxiv.org/pdf/2509.17965)**

> **作者:** Sai Samrat Kankanala; Ram Chandra; Sriram Ganapathy
>
> **备注:** 5 Pages, 1 Figure, 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing
>
> **摘要:** Auditory attention and selective phase-locking are central to human speech understanding in complex acoustic scenes and cocktail party settings, yet these capabilities in multilingual subjects remain poorly understood. While machine understanding of natural speech has advanced in recent years, questions persist about comprehension of overlapped and mixed-channel speech. We propose a systematic paradigm for studying humans and machines in speech question-answering tasks in multilingual settings with clean and mixed-channel speech. For human listeners, selective attention to a target speaker was significantly better in their native language (L1) than in their second language (L2). For machine listening, speech-based large language models (LLMs) match or exceed human performance in clean, single-speaker conditions but often struggle to selectively attend in two-speaker settings. These results reveal a key divergence: humans rely on attentional cues that are more streamlined in their native language, whereas LLMs default to parallel information extraction which exceed human skills.
>
---
#### [replaced 010] Noise-Conditioned Mixture-of-Experts Framework for Robust Speaker Verification
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于语音识别任务，解决噪声环境下说话人验证的鲁棒性问题。提出噪声条件混合专家框架，通过分解特征空间提升模型在不同噪声条件下的性能。**

- **链接: [https://arxiv.org/pdf/2510.18533](https://arxiv.org/pdf/2510.18533)**

> **作者:** Bin Gu; Haitao Zhao; Jibo Wei
>
> **备注:** Accepted by Signal Processing Letters
>
> **摘要:** Robust speaker verification under noisy conditions remains an open challenge. Conventional deep learning methods learn a robust unified speaker representation space against diverse background noise and achieve significant improvement. In contrast, this paper presents a noise-conditioned mixture-ofexperts framework that decomposes the feature space into specialized noise-aware subspaces for speaker verification. Specifically, we propose a noise-conditioned expert routing mechanism, a universal model based expert specialization strategy, and an SNR-decaying curriculum learning protocol, collectively improving model robustness and generalization under diverse noise conditions. The proposed method can automatically route inputs to expert networks based on noise information derived from the inputs, where each expert targets distinct noise characteristics while preserving speaker identity information. Comprehensive experiments demonstrate consistent superiority over baselines
>
---
#### [replaced 011] Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出音频-视觉世界模型（AVWM），解决多模态环境模拟问题。构建了AVW-4k数据集，设计AV-CDiT模型实现视听联合预测与导航任务优化。**

- **链接: [https://arxiv.org/pdf/2512.00883](https://arxiv.org/pdf/2512.00883)**

> **作者:** Jiahua Wang; Leqi Zheng; Jialong Wu; Yaoxin Mao
>
> **摘要:** World models simulate environmental dynamics to enable agents to plan and reason about future states. While existing approaches have primarily focused on visual observations, real-world perception inherently involves multiple sensory modalities. Audio provides crucial spatial and temporal cues such as sound source localization and acoustic scene properties, yet its integration into world models remains largely unexplored. No prior work has formally defined what constitutes an audio-visual world model or how to jointly capture binaural spatial audio and visual dynamics under precise action control. This work presents the first formal framework for Audio-Visual World Models (AVWM), formulating multimodal environment simulation as a partially observable Markov decision process with synchronized audio-visual observations. To address the lack of suitable training data, we construct AVW-4k, a dataset comprising 30 hours of binaural audio-visual trajectories with action annotations across 76 indoor environments. We propose AV-CDiT, an Audio-Visual Conditional Diffusion Transformer with a novel modality expert architecture that balances visual and auditory learning, optimized through a three-stage training strategy for effective multimodal integration. Extensive experiments demonstrate that AV-CDiT achieves high-fidelity multimodal prediction across visual and auditory modalities. Furthermore, we validate its practical utility in continuous audio-visual navigation tasks, where AVWM significantly enhances the agent's performance.
>
---
#### [replaced 012] TCG CREST System Description for the DISPLACE-M Challenge
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于说话人日志任务，解决医疗场景下的语音分离问题。对比了不同VAD方法和聚类算法，提升了系统性能。**

- **链接: [https://arxiv.org/pdf/2603.02030](https://arxiv.org/pdf/2603.02030)**

> **作者:** Nikhil Raghav; Md Sahidullah
>
> **备注:** Report submitted for the DISPLACE-M challenge
>
> **摘要:** This report presents the TCG CREST system description for Track 1 (Speaker Diarization) of the DISPLACE-M challenge, focusing on naturalistic medical conversations in noisy rural-healthcare scenarios. Our study evaluates the impact of various voice activity detection (VAD) methods and advanced clustering algorithms on overall speaker diarization (SD) performance. We compare and analyze two SD frameworks: a modular pipeline utilizing SpeechBrain with ECAPA-TDNN embeddings, and a state-of-the-art (SOTA) hybrid end-to-end neural diarization system, Diarizen, built on top of a pre-trained WavLM. With these frameworks, we explore diverse clustering techniques, including agglomerative hierarchical clustering (AHC), and multiple novel variants of spectral clustering, such as SC-adapt, SC-PNA, and SC-MK. Experimental results demonstrate that the Diarizen system provides an approximate $39\%$ relative improvement in the diarization error rate (DER) on the post-evaluation analysis of Phase~I compared to the SpeechBrain baseline. Our best-performing submitted system employing the Diarizen baseline with AHC employing a median filtering with a larger context window of $29$ achieved a DER of 10.37\% on the development and 9.21\% on the evaluation sets, respectively. Our team ranked fifth out of the 11 participating teams after the Phase~I evaluation.
>
---
#### [replaced 013] Multiplexing Neural Audio Watermarks
- **分类: eess.AS**

- **简介: 该论文属于音频水印任务，旨在解决单水印方案在对抗攻击下的脆弱性问题。通过引入多水印策略和新框架MaskNet，提升音频水印的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.02278](https://arxiv.org/pdf/2511.02278)**

> **作者:** Zheqi Yuan; Yucheng Huang; Guangzhi Sun; Zengrui Jin; Chao Zhang
>
> **备注:** Submission of Interspeech 2026
>
> **摘要:** Audio watermarking is essential for verifying speech authenticity, yet single-watermark schemes often struggle against sophisticated distortions such as neural reconstruction and adversarial attacks. To address this limitation, we introduce a multiplexing paradigm that combines multiple watermarking techniques to leverage their inherent complementarities. We explore both parallel and sequential multiplexing strategies and propose perceptual-adaptive time-frequency multiplexing (PA-TFM), a robust training-free approach. To further enhance performance, we introduce MaskNet, a novel model-based framework designed to learn effective time-domain multiplexing. Experimental results on the LibriSpeech and Common Voice datasets under 14 diverse attack types, including high-strength white-box and neural reconstruction attacks, demonstrate that both PA-TFM and MaskNet considerably outperform existing single-watermark baselines, establishing a resilient paradigm for real-world audio protection.
>
---
#### [replaced 014] Bottleneck Transformer-Based Approach for Improved Automatic STOI Score Prediction
- **分类: eess.AS; cs.LG; eess.SP**

- **简介: 该论文属于语音质量评估任务，旨在解决传统STOI计算依赖干净参考语音的问题。提出基于瓶颈Transformer的模型，提升STOI预测性能。**

- **链接: [https://arxiv.org/pdf/2602.15484](https://arxiv.org/pdf/2602.15484)**

> **作者:** Amartyaveer; Murali Kadambi; Chandra Mohan Sharma; Anupam Mondal; Prasanta Kumar Ghosh
>
> **备注:** 7 pages, 7 tables, 2 figures, ASRU 2025
>
> **摘要:** In this study, we have presented a novel approach to predict the Short-Time Objective Intelligibility (STOI) metric using a bottleneck transformer architecture. Traditional methods for calculating STOI typically requires clean reference speech, which limits their applicability in the real world. To address this, numerous deep learning-based nonintrusive speech assessment models have garnered significant interest. Many studies have achieved commendable performance, but there is room for further improvement. We propose the use of bottleneck transformer, incorporating convolution blocks for learning frame-level features and a multi-head self-attention (MHSA) layer to aggregate the information. These components enable the transformer to focus on the key aspects of the input data. Our model has shown higher correlation and lower mean squared error for both seen and unseen scenarios compared to the state-of-the-art model using self-supervised learning (SSL) and spectral features as inputs.
>
---
#### [replaced 015] VoiceBridge: General Speech Restoration with One-step Latent Bridge Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出VoiceBridge，解决通用语音修复问题，通过单步潜在桥模型高效恢复多种失真语音，提升重建质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.25275](https://arxiv.org/pdf/2509.25275)**

> **作者:** Chi Zhang; Kaiwen Zheng; Zehua Chen; Jun Zhu
>
> **摘要:** Bridge models have been investigated in speech enhancement but are mostly single-task, with constrained general speech restoration (GSR) capability. In this work, we propose VoiceBridge, a one-step latent bridge model (LBM) for GSR, capable of efficiently reconstructing 48 kHz fullband speech from diverse distortions. To inherit the advantages of data-domain bridge models, we design an energy-preserving variational autoencoder, enhancing the waveform-latent space alignment over varying energy levels. By compressing waveform into continuous latent representations, VoiceBridge models~\textit{various} GSR tasks with a~\textit{single} latent-to-latent generative process backed by a scalable transformer. To alleviate the challenge of reconstructing the high-quality target from distinctively different low-quality priors, we propose a joint neural prior for GSR, uniformly reducing the burden of the LBM in diverse tasks. Building upon these designs, we further investigate bridge training objective by jointly tuning LBM, decoder and discriminator together, transforming the model from a denoiser to generator and enabling \textit{one-step GSR without distillation}. Extensive validation across in-domain (\textit{e.g.}, denoising and super-resolution) and out-of-domain tasks (\textit{e.g.}, refining synthesized speech) and datasets demonstrates the superior performance of VoiceBridge. Demos: this https URL.
>
---
#### [replaced 016] PolyBench: A Benchmark for Compositional Reasoning in Polyphonic Audio
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
#### [replaced 017] Fast-Converging Distributed Signal Estimation in Topology-Unconstrained Wireless Acoustic Sensor Networks
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究无线声学传感器网络中的信号估计任务，解决TI-DANSE算法收敛慢的问题，提出改进的TI-DANSE+算法，提升收敛速度并优化通信效率。**

- **链接: [https://arxiv.org/pdf/2506.02797](https://arxiv.org/pdf/2506.02797)**

> **作者:** Paul Didier; Toon van Waterschoot; Simon Doclo; Jörg Bitzer; Marc Moonen
>
> **摘要:** This paper focuses on distributed signal estimation in topology-unconstrained wireless acoustic sensor networks (WASNs) where sensor nodes only transmit fused versions of their local sensor signals. For this task, the topology-independent (TI) distributed adaptive node-specific signal estimation (DANSE) algorithm (TI-DANSE) has previously been proposed. It converges towards the centralized signal estimation solution in non-fully connected and time-varying network topologies. However, the applicability of TI-DANSE in real-world scenarios is limited due to its slow convergence. The latter results from the fact that, in TI-DANSE, nodes only have access to the in-network sum of all fused signals in the WASN. We address this low convergence speed by introducing an improved TI-DANSE algorithm, referred to as TI-DANSE+, in which updating nodes separately use the partial in-network sums of fused signals coming from each of their neighbors. Nodes can maximize the number of available degrees of freedom in their local optimization problem, leading to faster convergence. This is further exploited by combining TI-DANSE+ with a tree-pruning strategy that maximizes the number of neighbors at the updating node. In fully connected WASNs, TI-DANSE+ converges as fast as the original DANSE algorithm (the latter only defined for fully connected WASNs) while using peer-to-peer data transmission instead of broadcasting and thus saving communication bandwidth. If link failures occur, the convergence of TI-DANSE+ towards the centralized solution is preserved without any change in its formulation. Altogether, the proposed TI-DANSE+ algorithm can be viewed as an all-round alternative to DANSE and TI-DANSE which (i) merges the advantages of both, (ii) reconciliates their differences into a single formulation, and (iii) shows advantages of its own in terms of communication bandwidth usage.
>
---
#### [replaced 018] Rethinking Discrete Speech Representation Tokens for Accent Generation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音生成任务，研究DSRT中口音信息的编码问题。通过新评估框架分析不同语音表示，发现层选择、ASR监督和代码本大小对口音信息的影响。**

- **链接: [https://arxiv.org/pdf/2601.19786](https://arxiv.org/pdf/2601.19786)**

> **作者:** Jinzuomu Zhong; Yi Wang; Korin Richmond; Peter Bell
>
> **摘要:** Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from several widely used speech representations. Our results reveal that: (1) choice of layers has the most significant impact on retaining accent information, (2) accent information is substantially reduced by ASR supervision; (3) naive codebook size reduction cannot effectively disentangle accent from phonetic and speaker information.
>
---
#### [replaced 019] Modeling strategies for speech enhancement in the latent space of a neural audio codec
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究语音增强任务，比较神经音频编解码器的连续和离散表示作为训练目标的效果，提出不同模型并验证其性能。**

- **链接: [https://arxiv.org/pdf/2510.26299](https://arxiv.org/pdf/2510.26299)**

> **作者:** Sofiene Kammoun; Xavier Alameda-Pineda; Simon Leglaive
>
> **摘要:** Neural audio codecs (NACs) provide compact latent speech representations in the form of sequences of continuous vectors or discrete tokens. In this work, we investigate how these two types of speech representations compare when used as training targets for supervised speech enhancement. We consider both autoregressive and non-autoregressive speech enhancement models based on the Conformer architecture, as well as a simple baseline where the NAC encoder is simply fine-tuned for speech enhancement. Our experiments reveal three key findings: predicting continuous latent representations consistently outperforms discrete token prediction; autoregressive models achieve higher quality but at the expense of intelligibility and efficiency, making non-autoregressive models more attractive in practice; and adding encoder fine-tuning yields the strongest enhancement metrics overall, though at the cost of degraded codec reconstruction. The code and audio samples are available online.
>
---
#### [replaced 020] Human-CLAP: Human-perception-based contrastive language-audio pretraining
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频与文本匹配任务，旨在解决CLAPScore与人类主观评价相关性低的问题。通过引入人类感知信息改进CLAP模型，提升其与主观评价的一致性。**

- **链接: [https://arxiv.org/pdf/2506.23553](https://arxiv.org/pdf/2506.23553)**

> **作者:** Taisei Takano; Yuki Okamoto; Yusuke Kanamori; Yuki Saito; Ryotaro Nagase; Hiroshi Saruwatari
>
> **备注:** Submitted to APSIPA ASC 2025
>
> **摘要:** Contrastive language-audio pretraining (CLAP) is widely used for audio generation and recognition tasks. For example, CLAPScore, which utilizes the similarity of CLAP embeddings, has been a major metric for the evaluation of the relevance between audio and text in text-to-audio. However, the relationship between CLAPScore and human subjective evaluation scores is still unclarified. We show that CLAPScore has a low correlation with human subjective evaluation scores. Additionally, we propose a human-perception-based CLAP called Human-CLAP by training a contrastive language-audio model using the subjective evaluation score. In our experiments, the results indicate that our Human-CLAP improved the Spearman's rank correlation coefficient (SRCC) between the CLAPScore and the subjective evaluation scores by more than 0.25 compared with the conventional CLAP.
>
---
