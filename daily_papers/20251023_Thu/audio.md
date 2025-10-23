# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] RIR-Mega: a large-scale simulated room impulse response dataset for machine learning and room acoustics modeling
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出RIR-Mega，一个大规模仿真混响脉冲响应数据集，用于机器学习与声学建模。针对真实RIR获取困难的问题，构建了结构化元数据和工具链，支持快速验证与复用。提供基准模型与在线访问，促进语音处理与声学研究的可复现性。**

- **链接: [http://arxiv.org/pdf/2510.18917v1](http://arxiv.org/pdf/2510.18917v1)**

> **作者:** Mandip Goswami
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Room impulse responses are a core resource for dereverberation, robust speech recognition, source localization, and room acoustics estimation. We present RIR-Mega, a large collection of simulated RIRs described by a compact, machine friendly metadata schema and distributed with simple tools for validation and reuse. The dataset ships with a Hugging Face Datasets loader, scripts for metadata checks and checksums, and a reference regression baseline that predicts RT60 like targets from waveforms. On a train and validation split of 36,000 and 4,000 examples, a small Random Forest on lightweight time and spectral features reaches a mean absolute error near 0.013 s and a root mean square error near 0.022 s. We host a subset with 1,000 linear array RIRs and 3,000 circular array RIRs on Hugging Face for streaming and quick tests, and preserve the complete 50,000 RIR archive on Zenodo. The dataset and code are public to support reproducible studies.
>
---
#### [new 002] Relative Transfer Matrix Estimator using Covariance Subtraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对多源语音分离任务，提出一种基于协方差相减的相对传输矩阵（ReTM）盲估计方法，旨在低信噪比和混响环境下提升分离性能。通过利用多通道信号协方差矩阵，实现对选定独立声源ReTM的有效估计，并在模拟与真实场景中验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2510.19439v1](http://arxiv.org/pdf/2510.19439v1)**

> **作者:** Wageesha N. Manamperi; Thushara D. Abhayapala
>
> **摘要:** The Relative Transfer Matrix (ReTM), recently introduced as a generalization of the relative transfer function for multiple receivers and sources, shows promising performance when applied to speech enhancement and speaker separation in noisy environments. Blindly estimating the ReTM of sound sources by exploiting the covariance matrices of multichannel recordings is highly beneficial for practical applications. In this paper, we use covariance subtraction to present a flexible and practically viable method for estimating the ReTM for a select set of independent sound sources. To show the versatility of the method, we validated it through a speaker separation application under reverberant conditions. Separation performance is evaluated at low signal-to-noise ratio levels in comparison with existing ReTM-based and relative transfer function-based estimators, in both simulated and real-life environments.
>
---
#### [new 003] VBx for End-to-End Neural and Clustering-based Diarization
- **分类: eess.AS**

- **简介: 该论文聚焦于说话人分离任务，针对端到端神经聚类框架中第二阶段聚类不准确的问题，提出通过过滤不可靠嵌入并引入VBx聚类提升鲁棒性，尤其在多说话人、短发言场景下表现优异，无需微调或参数调优即可达到先进性能。**

- **链接: [http://arxiv.org/pdf/2510.19572v1](http://arxiv.org/pdf/2510.19572v1)**

> **作者:** Petr Pálka; Jiangyu Han; Marc Delcroix; Naohiro Tawara; Lukáš Burget
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We present improvements to speaker diarization in the two-stage end-to-end neural diarization with vector clustering (EEND-VC) framework. The first stage employs a Conformer-based EEND model with WavLM features to infer frame-level speaker activity within short windows. The identities and counts of global speakers are then derived in the second stage by clustering speaker embeddings across windows. The focus of this work is to improve the second stage; we filter unreliable embeddings from short segments and reassign them after clustering. We also integrate the VBx clustering to improve robustness when the number of speakers is large and individual speaking durations are limited. Evaluation on a compound benchmark spanning multiple domains is conducted without fine-tuning the EEND model or tuning clustering parameters per dataset. Despite this, the system generalizes well and matches or exceeds recent state-of-the-art performance.
>
---
#### [new 004] StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文提出StutterZero与StutterFormer，首个端到端波形到波形的口吃语音转写与修正模型，直接将口吃语音转为流利语音并生成文本。针对现有方法分离转写与重建导致误差放大问题，通过联合建模提升准确率，在多个基准上显著降低词错误率并提升语义相似度。**

- **链接: [http://arxiv.org/pdf/2510.18938v1](http://arxiv.org/pdf/2510.18938v1)**

> **作者:** Qianheng Xu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems.
>
---
#### [new 005] Auditory Attention Decoding from Ear-EEG Signals: A Dataset with Dynamic Attention Switching and Rigorous Cross-Validation
- **分类: eess.AS; eess.SP**

- **简介: 该论文聚焦于耳部脑电（ear-EEG）信号中的听觉注意力解码任务，旨在解决真实场景下注意力动态切换的建模问题。研究构建了包含三声源空间分布的新型cEEGrid数据集，采用嵌套留一法验证，评估四种模型，发现Wiener滤波与CCA在30秒窗口下表现最优，且能有效追踪注意力切换，验证了动态生态范式与严谨验证的重要性。**

- **链接: [http://arxiv.org/pdf/2510.19174v1](http://arxiv.org/pdf/2510.19174v1)**

> **作者:** Yuanming Zhang; Zeyan Song; Jing Lu; Fei Chen; Zhibin Lin
>
> **摘要:** Recent promising results in auditory attention decoding (AAD) using scalp electroencephalography (EEG) have motivated the exploration of cEEGrid, a flexible and portable ear-EEG system. While prior cEEGrid-based studies have confirmed the feasibility of AAD, they often neglect the dynamic nature of attentional states in real-world contexts. To address this gap, a novel cEEGrid dataset featuring three concurrent speakers distributed across three of five distinct spatial locations is introduced. The novel dataset is designed to probe attentional tracking and switching in realistic scenarios. Nested leave-one-out validation-an approach more rigorous than conventional single-loop leave-one-out validation-is employed to reduce biases stemming from EEG's intricate temporal dynamics. Four rule-based models are evaluated: Wiener filter (WF), canonical component analysis (CCA), common spatial pattern (CSP) and Riemannian Geometry-based classifier (RGC). With a 30-second decision window, WF and CCA models achieve decoding accuracies of 41.5% and 41.4%, respectively, while CSP and RGC models yield 37.8% and 37.6% accuracies using a 10-second window. Notably, both WF and CCA successfully track attentional state switches across all experimental tasks. Additionally, higher decoding accuracies are observed for electrodes positioned at the upper cEEGrid layout and near the listener's right ear. These findings underscore the utility of dynamic, ecologically valid paradigms and rigorous validation in advancing AAD research with cEEGrid.
>
---
#### [new 006] AMAuT: A Flexible and Efficient Multiview Audio Transformer Framework Trained from Scratch
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出AMAuT，一种从零训练的多视图音频变换框架，解决现有预训练模型依赖固定采样率与长度、计算成本高的问题。通过增强驱动的多视图学习、一维CNN瓶颈、双CLS+TAL令牌及测试时适配，实现灵活输入与高效推理，在多个音频分类任务中达高精度且显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.19368v1](http://arxiv.org/pdf/2510.19368v1)**

> **作者:** Weichuang Shao; Iman Yi Liao; Tomas Henrique Bode Maul; Tissa Chandesa
>
> **摘要:** Recent foundational models, SSAST, EAT, HuBERT, Qwen-Audio, and Audio Flamingo, achieve top-tier results across standard audio benchmarks but are limited by fixed input rates and durations, hindering their reusability. This paper introduces the Augmentation-driven Multiview Audio Transformer (AMAuT), a training-from-scratch framework that eliminates the dependency on pre-trained weights while supporting arbitrary sample rates and audio lengths. AMAuT integrates four key components: (1) augmentation-driven multiview learning for robustness, (2) a conv1 + conv7 + conv1 one-dimensional CNN bottleneck for stable temporal encoding, (3) dual CLS + TAL tokens for bidirectional context representation, and (4) test-time adaptation/augmentation (TTA^2) to improve inference reliability. Experiments on five public benchmarks, AudioMNIST, SpeechCommands V1 & V2, VocalSound, and CochlScene, show that AMAuT achieves accuracies up to 99.8% while consuming less than 3% of the GPU hours required by comparable pre-trained models. Thus, AMAuT presents a highly efficient and flexible alternative to large pre-trained models, making state-of-the-art audio classification accessible in computationally constrained settings.
>
---
#### [new 007] EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文针对语音深度伪造检测中真实场景泛化能力不足的问题，提出EchoFake数据集，包含120小时多说话人语音及物理回放录音。旨在提升模型对实际回放攻击的检测性能，解决现有方法在真实环境下游显著失效的问题。**

- **链接: [http://arxiv.org/pdf/2510.19414v1](http://arxiv.org/pdf/2510.19414v1)**

> **作者:** Tong Zhang; Yihuan Huang; Yanzhen Ren
>
> **摘要:** The growing prevalence of speech deepfakes has raised serious concerns, particularly in real-world scenarios such as telephone fraud and identity theft. While many anti-spoofing systems have demonstrated promising performance on lab-generated synthetic speech, they often fail when confronted with physical replay attacks-a common and low-cost form of attack used in practical settings. Our experiments show that models trained on existing datasets exhibit severe performance degradation, with average accuracy dropping to 59.6% when evaluated on replayed audio. To bridge this gap, we present EchoFake, a comprehensive dataset comprising more than 120 hours of audio from over 13,000 speakers, featuring both cutting-edge zero-shot text-to-speech (TTS) speech and physical replay recordings collected under varied devices and real-world environmental settings. Additionally, we evaluate three baseline detection models and show that models trained on EchoFake achieve lower average EERs across datasets, indicating better generalization. By introducing more practical challenges relevant to real-world deployment, EchoFake offers a more realistic foundation for advancing spoofing detection methods.
>
---
#### [new 008] An Efficient Neural Network for Modeling Human Auditory Neurograms for Speech
- **分类: eess.AS**

- **简介: 该论文提出一种高效卷积编码器，用于建模人类听觉神经信号（神经图），解决传统模型计算复杂、随机性强的问题。工作聚焦于确定性速率域神经图的精确映射，实现低延迟、高效率的听觉前端处理，适用于神经科学与音频信号处理。**

- **链接: [http://arxiv.org/pdf/2510.19354v1](http://arxiv.org/pdf/2510.19354v1)**

> **作者:** Eylon Zohar; Israel Nelken; Boaz Rafaely
>
> **摘要:** Classical auditory-periphery models, exemplified by Bruce et al., 2018, provide high-fidelity simulations but are stochastic and computationally demanding, limiting large-scale experimentation and low-latency use. Prior neural encoders approximate aspects of the periphery; however, few are explicitly trained to reproduce the deterministic, rate-domain neurogram , hindering like-for-like evaluation. We present a compact convolutional encoder that approximates the Bruce mean-rate pathway and maps audio to a multi-frequency neurogram. We deliberately omit stochastic spiking effects and focus on a deterministic mapping (identical outputs for identical inputs). Using a computationally efficient design, the encoder achieves close correspondence to the reference while significantly reducing computation, enabling efficient modeling and front-end processing for auditory neuroscience and audio signal processing applications.
>
---
#### [new 009] Time delay embeddings to characterize the timbre of musical instruments using Topological Data Analysis: a study on synthetic and real data
- **分类: cs.SD; math.AT; nlin.AO; physics.data-an; physics.soc-ph**

- **简介: 该论文研究如何用时间延迟嵌入优化拓扑数据分析（TDA）来表征乐器音色。针对传统方法忽略声音细微特征的问题，提出通过选择特定时间延迟（如基频分数周期）增强TDA对谐波结构的捕捉能力，有效区分整数与非整数谐波，适用于合成与真实音频数据。**

- **链接: [http://arxiv.org/pdf/2510.19435v1](http://arxiv.org/pdf/2510.19435v1)**

> **作者:** Gakusei Sato; Hiroya Nakao; Riccardo Muolo
>
> **摘要:** Timbre allows us to distinguish between sounds even when they share the same pitch and loudness, playing an important role in music, instrument recognition, and speech. Traditional approaches, such as frequency analysis or machine learning, often overlook subtle characteristics of sound. Topological Data Analysis (TDA) can capture complex patterns, but its application to timbre has been limited, partly because it is unclear how to represent sound effectively for TDA. In this study, we investigate how different time delay embeddings affect TDA results. Using both synthetic and real audio signals, we identify time delays that enhance the detection of harmonic structures. Our findings show that specific delays, related to fractions of the fundamental period, allow TDA to reveal key harmonic features and distinguish between integer and non-integer harmonics. The method is effective for synthetic and real musical instrument sounds and opens the way for future works, which could extend it to more complex sounds using higher-dimensional embeddings and additional persistence statistics.
>
---
#### [new 010] Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究自动语音识别（ASR）与语音翻译（ST）任务中最小贝叶斯风险（MBR）解码的有效性。针对当前主流使用束搜索而MBR在文本生成中表现更优的现象，作者评估了MBR在英语和日语上的ASR/ST性能，发现其在多数场景下优于束搜索，验证了MBR在高精度离线语音任务中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.19471v1](http://arxiv.org/pdf/2510.19471v1)**

> **作者:** Yuu Jinnai
>
> **摘要:** Recent work has shown that sample-based Minimum Bayes Risk (MBR) decoding outperforms beam search in text-to-text generation tasks, such as machine translation, text summarization, and image captioning. On the other hand, beam search is the current practice for speech-to-text tasks such as automatic speech recognition (ASR) and Speech Translation (ST). Given that MBR decoding is effective in text-to-text generation tasks, it is reasonable to expect it to also be effective for speech-to-text tasks. In this paper, we evaluate MBR decoding for ASR and ST tasks on English and Japanese using Whisper and its derivative models. We observe that the accuracy of MBR decoding outperforms that of beam search in most of the experimental settings we have evaluated. The results show that MBR decoding is a promising method for offline ASR and ST tasks that require high accuracy. The code is available at https://github.com/CyberAgentAILab/mbr-for-asr
>
---
#### [new 011] Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文针对语音模型评估碎片化问题，提出一个三维分类体系，明确“何种模型适用何种评估”。通过梳理现有评测任务，统一评估标准，揭示短板并指导未来基准设计，为语音模型评估提供系统性框架。**

- **链接: [http://arxiv.org/pdf/2510.19509v1](http://arxiv.org/pdf/2510.19509v1)**

> **作者:** Maureen de Seyssel; Eeshan Gunesh Dhekane
>
> **备注:** 57 pages (26 main, 25 appendix, 6 references)
>
> **摘要:** Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the \textbf{evaluation aspect} being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.
>
---
#### [new 012] The MUSE Benchmark: Probing Music Perception and Auditory Relational Reasoning in Audio LLMS
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出MUSE基准，用于评估音频大模型在音乐感知与听觉关系推理方面的能力。针对现有评测忽略深层推理缺陷的问题，设计10项任务，对比4个SOTA模型与200名人类表现，发现模型普遍存在感知短板，且链式思考提示效果不稳定，推动更鲁棒AI系统发展。**

- **链接: [http://arxiv.org/pdf/2510.19055v1](http://arxiv.org/pdf/2510.19055v1)**

> **作者:** Brandon James Carone; Iran R. Roman; Pablo Ripollés
>
> **备注:** 5 pages, 2 figures, 2 tables
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated capabilities in audio understanding, but current evaluations may obscure fundamental weaknesses in relational reasoning. We introduce the Music Understanding and Structural Evaluation (MUSE) Benchmark, an open-source resource with 10 tasks designed to probe fundamental music perception skills. We evaluate four SOTA models (Gemini Pro and Flash, Qwen2.5-Omni, and Audio-Flamingo 3) against a large human baseline (N=200). Our results reveal a wide variance in SOTA capabilities and a persistent gap with human experts. While Gemini Pro succeeds on basic perception, Qwen and Audio Flamingo 3 perform at or near chance, exposing severe perceptual deficits. Furthermore, we find Chain-of-Thought (CoT) prompting provides inconsistent, often detrimental results. Our work provides a critical tool for evaluating invariant musical representations and driving development of more robust AI systems.
>
---
#### [new 013] Steering Autoregressive Music Generation with Recursive Feature Machines
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文针对可控音乐生成任务，解决现有方法需重训练或引入噪音的问题。提出MusicRFM框架，利用递归特征机（RFM）分析预训练模型激活态，提取可解释的音乐概念方向，并实时注入以精细控制音符、和弦等属性，实现高精度引导且不影响文本提示契合度。**

- **链接: [http://arxiv.org/pdf/2510.19127v1](http://arxiv.org/pdf/2510.19127v1)**

> **作者:** Daniel Zhao; Daniel Beaglehole; Taylor Berg-Kirkpatrick; Julian McAuley; Zachary Novack
>
> **摘要:** Controllable music generation remains a significant challenge, with existing methods often requiring model retraining or introducing audible artifacts. We introduce MusicRFM, a framework that adapts Recursive Feature Machines (RFMs) to enable fine-grained, interpretable control over frozen, pre-trained music models by directly steering their internal activations. RFMs analyze a model's internal gradients to produce interpretable "concept directions", or specific axes in the activation space that correspond to musical attributes like notes or chords. We first train lightweight RFM probes to discover these directions within MusicGen's hidden states; then, during inference, we inject them back into the model to guide the generation process in real-time without per-step optimization. We present advanced mechanisms for this control, including dynamic, time-varying schedules and methods for the simultaneous enforcement of multiple musical properties. Our method successfully navigates the trade-off between control and generation quality: we can increase the accuracy of generating a target musical note from 0.23 to 0.82, while text prompt adherence remains within approximately 0.02 of the unsteered baseline, demonstrating effective control with minimal impact on prompt fidelity. We release code to encourage further exploration on RFMs in the music domain.
>
---
## 更新

#### [replaced 001] Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.21087v2](http://arxiv.org/pdf/2509.21087v2)**

> **作者:** Rostislav Makarov; Lea Schönherr; Timo Gerkmann
>
> **备注:** Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.
>
---
#### [replaced 002] Wireless Hearables With Programmable Speech AI Accelerators
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.18698v2](http://arxiv.org/pdf/2503.18698v2)**

> **作者:** Malek Itani; Tuochao Chen; Arun Raghavan; Gavriel Kohlberg; Shyamnath Gollakota
>
> **摘要:** The conventional wisdom has been that designing ultra-compact, battery-constrained wireless hearables with on-device speech AI models is challenging due to the high computational demands of streaming deep learning models. Speech AI models require continuous, real-time audio processing, imposing strict computational and I/O constraints. We present NeuralAids, a fully on-device speech AI system for wireless hearables, enabling real-time speech enhancement and denoising on compact, battery-constrained devices. Our system bridges the gap between state-of-the-art deep learning for speech enhancement and low-power AI hardware by making three key technical contributions: 1) a wireless hearable platform integrating a speech AI accelerator for efficient on-device streaming inference, 2) an optimized dual-path neural network designed for low-latency, high-quality speech enhancement, and 3) a hardware-software co-design that uses mixed-precision quantization and quantization-aware training to achieve real-time performance under strict power constraints. Our system processes 6 ms audio chunks in real-time, achieving an inference time of 5.54 ms while consuming 71.6 mW. In real-world evaluations, including a user study with 28 participants, our system outperforms prior on-device models in speech quality and noise suppression, paving the way for next-generation intelligent wireless hearables that can enhance hearing entirely on-device.
>
---
#### [replaced 003] SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement
- **分类: eess.AS; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.07634v5](http://arxiv.org/pdf/2506.07634v5)**

> **作者:** Chenyu Yang; Shuai Wang; Hangting Chen; Wei Tan; Jianwei Yu; Haizhou Li
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** Generating music with coherent structure, harmonious instrumental and vocal elements remains a significant challenge in song generation. Existing language models and diffusion-based methods often struggle to balance global coherence with local fidelity, resulting in outputs that lack musicality or suffer from incoherent progression and mismatched lyrics. This paper introduces $\textbf{SongBloom}$, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models. Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process. Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms. Audio samples are available on our demo page: https://cypress-yang.github.io/SongBloom_demo. The code and model weights have been released on https://github.com/Cypress-Yang/SongBloom .
>
---
#### [replaced 004] MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06098v2](http://arxiv.org/pdf/2508.06098v2)**

> **作者:** Xiquan Li; Junxi Liu; Yuzhe Liang; Zhikang Niu; Wenxi Chen; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in Text-to-Audio Generation (TTA), providing sound creators with powerful tools to transform inspirations into vivid audio. Yet despite these advances, current TTA systems often suffer from slow inference speed, which greatly hinders the efficiency and smoothness of audio creation. In this paper, we present MeanAudio, a fast and faithful text-to-audio generator capable of rendering realistic sound with only one function evaluation (1-NFE). MeanAudio leverages: (i) the MeanFlow objective with guided velocity target that significantly accelerates inference speed, (ii) an enhanced Flux-style transformer with dual text encoders for better semantic alignment and synthesis quality, and (iii) an efficient instantaneous-to-mean curriculum that speeds up convergence and enables training on consumer-grade GPUs. Through a comprehensive evaluation study, we demonstrate that MeanAudio achieves state-of-the-art performance in single-step audio generation. Specifically, it achieves a real-time factor (RTF) of 0.013 on a single NVIDIA RTX 3090, yielding a 100x speedup over SOTA diffusion-based TTA systems. Moreover, MeanAudio also shows strong performance in multi-step generation, enabling smooth transitions across successive synthesis steps.
>
---
#### [replaced 005] Efficient Interleaved Speech Modeling through Knowledge Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.23670v2](http://arxiv.org/pdf/2506.23670v2)**

> **作者:** Mohammadmahdi Nouriborji; Morteza Rohanian
>
> **摘要:** Current speech language models exceed the size and latency constraints of many deployment environments. We build compact, expressive speech generation models through layer-aligned distillation, matching hidden states, attention maps, and softened logits to compress large multimodal transformers by 3x with minimal loss in performance. We introduce TinyWave, a family of 2B-parameter models for speech-to-speech and interleaved speech-text generation, trained on 50,000 hours of public audio. TinyWave supports (i) speech-only generation using phonetic or expressive tokens and (ii) mixed speech-text continuations. Evaluation on Libri-Light shows TinyWave within 1.4 normalized perplexity points of its teacher. Accuracy on spoken StoryCloze and SALMon reaches 93-97% of the teacher's performance, outperforming size-matched baselines. These models are optimized for deployment on commodity hardware, enabling applications in real-time conversational agents, assistive technologies, and low-resource environments. We release models, training code, and evaluation scripts to support reproducible research on compact, expressive speech generation.
>
---
