# 音频 cs.SD;  eess.AS

- **最新发布 12 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决传统音频编码器的局限性。提出MOSS-Audio-Tokenizer，采用纯Transformer架构实现端到端音频分词，提升重建质量与扩展性。**

- **链接: [https://arxiv.org/pdf/2602.10934v1](https://arxiv.org/pdf/2602.10934v1)**

> **作者:** Yitian Gong; Kuangwei Chen; Zhaoye Fei; Xiaogui Yang; Ke Chen; Yang Wang; Kexin Huang; Mingshu Chen; Ruixiao Li; Qingyuan Cheng; Shimin Li; Xipeng Qiu
>
> **备注:** 27 pages, 8 figures
>
> **摘要:** Discrete audio tokenizers are fundamental to empowering large language models with native audio processing and generation capabilities. Despite recent progress, existing approaches often rely on pretrained encoders, semantic distillation, or heterogeneous CNN-based architectures. These designs introduce fixed inductive biases that limit reconstruction fidelity and hinder effective scaling. In this paper, we argue that discrete audio tokenization should be learned fully end-to-end using a homogeneous and scalable architecture. To this end, we first propose CAT (Causal Audio Tokenizer with Transformer), a purely Transformer-based architecture that jointly optimizes the encoder, quantizer, and decoder from scratch for high-fidelity reconstruction. Building on the CAT architecture, we develop MOSS-Audio-Tokenizer, a large-scale audio tokenizer featuring 1.6 billion parameters, pre-trained on 3 million hours of diverse, general audio data. We show that this simple, fully end-to-end approach built from homogeneous, causal Transformer blocks scales gracefully and supports high-fidelity reconstruction across diverse audio domains. Across speech, sound, and music, MOSS-Audio-Tokenizer consistently outperforms prior codecs over a wide range of bitrates, while exhibiting predictable improvements with increased scale. Notably, leveraging the discrete tokens from our model, we develop the first purely autoregressive TTS model that surpasses prior non-autoregressive and cascaded systems. Furthermore, MOSS-Audio-Tokenizer enables competitive ASR performance without auxiliary encoders. Our findings position the CAT architecture as a unified, scalable interface for the next generation of native audio foundation models.
>
---
#### [new 002] SCRAPL: Scattering Transform with Random Paths for Machine Learning
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出SCRAPL方法，解决波束散射变换在神经网络训练中计算成本高的问题，用于音频处理任务，提升模型收敛与性能。**

- **链接: [https://arxiv.org/pdf/2602.11145v1](https://arxiv.org/pdf/2602.11145v1)**

> **作者:** Christopher Mitcheltree; Vincent Lostanlen; Emmanouil Benetos; Mathieu Lagrange
>
> **备注:** Accepted to ICLR 2026. Code, audio samples, and Python package provided at https://christhetree.github.io/scrapl/
>
> **摘要:** The Euclidean distance between wavelet scattering transform coefficients (known as paths) provides informative gradients for perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing. However, these transforms are computationally expensive when employed as differentiable loss functions for stochastic gradient descent due to their numerous paths, which significantly limits their use in neural network training. Against this problem, we propose "Scattering transform with Random Paths for machine Learning" (SCRAPL): a stochastic optimization scheme for efficient evaluation of multivariable scattering transforms. We implement SCRAPL for the joint time-frequency scattering transform (JTFS) which demodulates spectrotemporal patterns at multiple scales and rates, allowing a fine characterization of intermittent auditory textures. We apply SCRAPL to differentiable digital signal processing (DDSP), specifically, unsupervised sound matching of a granular synthesizer and the Roland TR-808 drum machine. We also propose an initialization heuristic based on importance sampling, which adapts SCRAPL to the perceptual content of the dataset, improving neural network convergence and evaluation performance. We make our code and audio samples available and provide SCRAPL as a Python package.
>
---
#### [new 003] From Diet to Free Lunch: Estimating Auxiliary Signal Properties using Dynamic Pruning Masks in Speech Enhancement Networks
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音增强任务，旨在通过动态剪枝掩码估计信号属性，解决设备端部署多个模型计算负担重的问题。工作包括分析剪枝掩码的潜在信息并实现高效预测。**

- **链接: [https://arxiv.org/pdf/2602.10666v1](https://arxiv.org/pdf/2602.10666v1)**

> **作者:** Riccardo Miccini; Clément Laroche; Tobias Piechowiak; Xenofon Fafoutis; Luca Pezzarossa
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
> **摘要:** Speech Enhancement (SE) in audio devices is often supported by auxiliary modules for Voice Activity Detection (VAD), SNR estimation, or Acoustic Scene Classification to ensure robust context-aware behavior and seamless user experience. Just like SE, these tasks often employ deep learning; however, deploying additional models on-device is computationally impractical, whereas cloud-based inference would introduce additional latency and compromise privacy. Prior work on SE employed Dynamic Channel Pruning (DynCP) to reduce computation by adaptively disabling specific channels based on the current input. In this work, we investigate whether useful signal properties can be estimated from these internal pruning masks, thus removing the need for separate models. We show that simple, interpretable predictors achieve up to 93% accuracy on VAD, 84% on noise classification, and an R2 of 0.86 on F0 estimation. With binary masks, predictions reduce to weighted sums, inducing negligible overhead. Our contribution is twofold: on one hand, we examine the emergent behavior of DynCP models through the lens of downstream prediction tasks, to reveal what they are learning; on the other, we repurpose and re-propose DynCP as a holistic solution for efficient SE and simultaneous estimation of signal properties.
>
---
#### [new 004] Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Calliope，一个开源框架，用于生成精确同步的有声电子书。解决无开源工具的问题，通过TTS直接生成音频时间戳，确保文本与语音同步，同时保留原始排版和隐私。**

- **链接: [https://arxiv.org/pdf/2602.10735v1](https://arxiv.org/pdf/2602.10735v1)**

> **作者:** Hugo L. Hammer; Vajira Thambawita; Pål Halvorsen
>
> **摘要:** A narrated e-book combines synchronized audio with digital text, highlighting the currently spoken word or sentence during playback. This format supports early literacy and assists individuals with reading challenges, while also allowing general readers to seamlessly switch between reading and listening. With the emergence of natural-sounding neural Text-to-Speech (TTS) technology, several commercial services have been developed to leverage these technology for converting standard text e-books into high-quality narrated e-books. However, no open-source solutions currently exist to perform this task. In this paper, we present Calliope, an open-source framework designed to fill this gap. Our method leverages state-of-the-art open-source TTS to convert a text e-book into a narrated e-book in the EPUB 3 Media Overlay format. The method offers several innovative steps: audio timestamps are captured directly during TTS, ensuring exact synchronization between narration and text highlighting; the publisher's original typography, styling, and embedded media are strictly preserved; and the entire pipeline operates offline. This offline capability eliminates recurring API costs, mitigates privacy concerns, and avoids copyright compliance issues associated with cloud-based services. The framework currently supports the state-of-the-art open-source TTS systems XTTS-v2 and Chatterbox. A potential alternative approach involves first generating narration via TTS and subsequently synchronizing it with the text using forced alignment. However, while our method ensures exact synchronization, our experiments show that forced alignment introduces drift between the audio and text highlighting significant enough to degrade the reading experience. Source code and usage instructions are available at https://github.com/hugohammer/TTS-Narrated-Ebook-Creator.git.
>
---
#### [new 005] Self-Supervised Learning for Speaker Recognition: A study and review
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统模型依赖标注数据的问题。通过研究自监督学习方法，探索其在说话人识别中的应用与效果。**

- **链接: [https://arxiv.org/pdf/2602.10829v1](https://arxiv.org/pdf/2602.10829v1)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** accepted for publication in Speech Communication
>
> **摘要:** Deep learning models trained in a supervised setting have revolutionized audio and speech processing. However, their performance inherently depends on the quantity of human-annotated data, making them costly to scale and prone to poor generalization under unseen conditions. To address these challenges, Self-Supervised Learning (SSL) has emerged as a promising paradigm, leveraging vast amounts of unlabeled data to learn relevant representations. The application of SSL for Automatic Speech Recognition (ASR) has been extensively studied, but research on other downstream tasks, notably Speaker Recognition (SR), remains in its early stages. This work describes major SSL instance-invariance frameworks (e.g., SimCLR, MoCo, and DINO), initially developed for computer vision, along with their adaptation to SR. Various SSL methods for SR, proposed in the literature and built upon these frameworks, are also presented. An extensive review of these approaches is then conducted: (1) the effect of the main hyperparameters of SSL frameworks is investigated; (2) the role of SSL components is studied (e.g., data-augmentation, projector, positive sampling); and (3) SSL frameworks are evaluated on SR with in-domain and out-of-domain data, using a consistent experimental setup, and a comprehensive comparison of SSL methods from the literature is provided. Specifically, DINO achieves the best downstream performance and effectively models intra-speaker variability, although it is highly sensitive to hyperparameters and training conditions, while SimCLR and MoCo provide robust alternatives that effectively capture inter-speaker variability and are less prone to collapse. This work aims to highlight recent trends and advancements, identifying current challenges in the field.
>
---
#### [new 006] AudioRouter: Data Efficient Audio Understanding via RL based Dual Reasoning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频理解任务，旨在解决大模型在细粒度听觉感知上的不足。通过强化学习框架AudioRouter，使模型高效使用外部工具，减少数据依赖。**

- **链接: [https://arxiv.org/pdf/2602.10439v1](https://arxiv.org/pdf/2602.10439v1)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **摘要:** Large Audio Language Models (LALMs) have demonstrated strong capabilities in audio understanding and reasoning. However, their performance on fine grained auditory perception remains unreliable, and existing approaches largely rely on data intensive training to internalize perceptual abilities. We propose AudioRouter, a reinforcement learning framework that enables LALMs to improve audio understanding by learning when and how to use external audio tools. Rather than tightly coupling tool usage with audio reasoning, AudioRouter formulates tool use as an explicit decision making problem and optimizes a lightweight routing policy while keeping the underlying reasoning model frozen. Experimental results show that AudioRouter achieves substantial improvements on standard audio understanding benchmarks while requiring up to 600x less training data to learn tool usage compared with conventional training paradigms. These findings suggest that learning effective tool usage offers a data efficient and scalable alternative to internalizing perceptual abilities in LALMs.
>
---
#### [new 007] RE-LLM: Refining Empathetic Speech-LLM Responses by Integrating Emotion Nuance
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于情感计算任务，旨在提升AI在对话中的共情能力。针对现有模型情感理解不足的问题，提出RE-LLM，融合语音与情感嵌入，增强情感表达与共鸣。**

- **链接: [https://arxiv.org/pdf/2602.10716v1](https://arxiv.org/pdf/2602.10716v1)**

> **作者:** Jing-Han Chen; Bo-Hao Su; Ya-Tse Wu; Chi-Chun Lee
>
> **备注:** 5 pages, 1 figure, 2 tables. Accepted at IEEE ASRU 2025
>
> **摘要:** With generative AI advancing, empathy in human-AI interaction is essential. While prior work focuses on emotional reflection, emotional exploration, key to deeper engagement, remains overlooked. Existing LLMs rely on text which captures limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show statistically significant gains in empathy metrics across three datasets. RE-LLM relatively improves the Emotional Reaction score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD. Notably, it raises the Exploration score by 35.42% and 3.91% on IEMOCAP, 139.28% and 9.83% on ESD, and 60.95% and 22.64% on MSP-PODCAST. It also boosts unweighted accuracy by 5.4% on IEMOCAP, 2.3% on ESD, and 6.9% on MSP-PODCAST in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM.
>
---
#### [new 008] Emotion-Coherent Speech Data Augmentation and Self-Supervised Contrastive Style Training for Enhancing Kids's Story Speech Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升儿童故事语音的自然度和风格一致性。通过情感一致的数据增强和自监督对比训练，优化模型生成效果。**

- **链接: [https://arxiv.org/pdf/2602.10164v1](https://arxiv.org/pdf/2602.10164v1)**

> **作者:** Raymond Chung
>
> **备注:** Accepted at IEEE Spoken Language Technology Workshop 2024
>
> **摘要:** Expressive speech synthesis requires vibrant prosody and well-timed pauses. We propose an effective strategy to augment a small dataset to train an expressive end-to-end Text-to-Speech model. We merge audios of emotionally congruent text using a text emotion recognizer, creating augmented expressive speech data. By training with two-sentence audio, our model learns natural breaks between lines. We further apply self-supervised contrastive training to improve the speaking style embedding extraction from speech. During inference, our model produces multi-sentence speech in one step, guided by the text-predicted speaking style. Evaluations showcase the effectiveness of our proposed approach when compared to a baseline model trained with consecutive two-sentence audio. Our synthesized speeches give a closer inter-sentence pause distribution to the ground truth speech. Subjective evaluations reveal our synthesized speech scored higher in naturalness and style suitability than the baseline.
>
---
#### [new 009] AudioRAG: A Challenging Benchmark for Audio Reasoning and Information Retrieval
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出AudioRAG基准，用于评估音频推理与信息检索结合的模型。针对现有基准仅依赖内部知识的问题，该工作构建了包含生成和人工标注数据的基准，旨在推动音频相关模型在真实场景下的研究。**

- **链接: [https://arxiv.org/pdf/2602.10656v1](https://arxiv.org/pdf/2602.10656v1)**

> **作者:** Jingru Lin; Chen Zhang; Tianrui Wang; Haizhou Li
>
> **备注:** Accepted by Audio-AAAI
>
> **摘要:** Due to recent advancements in Large Audio-Language Models (LALMs) that demonstrate remarkable performance across a range of sound-, speech- and music-related tasks, there is a growing interest in proposing benchmarks to assess these models. Existing benchmarks generally focus only on reasoning with internal knowledge, neglecting real-world scenarios that require external information grounding. To bridge this gap, we introduce AudioRAG, a novel benchmark designed to evaluate audio-based reasoning augmented by information retrieval in realistic web environments. This benchmark comprises both LLM-generated and manually curated question-answer pairs. Our evaluations reveal that even the state-of-the-art LALMs struggle to answer these questions. We therefore propose an agentic pipeline that integrates audio reasoning with retrieval-augmented generation, providing a stronger baseline for future research.
>
---
#### [new 010] Simultaneous Speech-to-Speech Translation Without Aligned Data
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，解决无对齐数据下的实时翻译问题。提出Hibiki-Zero模型，无需词级对齐，提升翻译质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.11072v1](https://arxiv.org/pdf/2602.11072v1)**

> **作者:** Tom Labiausse; Romain Fabre; Yannick Estève; Alexandre Défossez; Neil Zeghidour
>
> **备注:** See inference code at: https://github.com/kyutai-labs/hibiki-zero
>
> **摘要:** Simultaneous speech translation requires translating source speech into a target language in real-time while handling non-monotonic word dependencies. Traditional approaches rely on supervised training with word-level aligned data, which is difficult to collect at scale and thus depends on synthetic alignments using language-specific heuristics that are suboptimal. We propose Hibiki-Zero, which eliminates the need for word-level alignments entirely. This fundamentally simplifies the training pipeline and enables seamless scaling to diverse languages with varying grammatical structures, removing the bottleneck of designing language-specific alignment heuristics. We first train on sentence-level aligned data to learn speech translation at high latency, then apply a novel reinforcement learning strategy using GRPO to optimize latency while preserving translation quality. Hibiki-Zero achieves state-of-the-art performance in translation accuracy, latency, voice transfer, and naturalness across five X-to-English tasks. Moreover, we demonstrate that our model can be adapted to support a new input language with less than 1000h of speech. We provide examples, model weights, inference code and we release a benchmark containing 45h of multilingual data for speech translation evaluation.
>
---
#### [new 011] MerkleSpeech: Public-Key Verifiable, Chunk-Localised Speech Provenance via Perceptual Fingerprints and Merkle Commitments
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文提出MerkleSpeech系统，用于验证语音内容的来源和完整性。解决语音篡改检测问题，通过感知指纹和Merkle树实现可公开验证的块级溯源。**

- **链接: [https://arxiv.org/pdf/2602.10166v1](https://arxiv.org/pdf/2602.10166v1)**

> **作者:** Tatsunori Ono
>
> **备注:** 16 pages, 4 figures, 3 tables
>
> **摘要:** Speech provenance goes beyond detecting whether a watermark is present. Real workflows involve splicing, quoting, trimming, and platform-level transforms that may preserve some regions while altering others. Neural watermarking systems have made strides in robustness and localised detection, but most deployments produce outputs with no third-party verifiable cryptographic proof tying a time segment to an issuer-signed original. Provenance standards like C2PA adopt signed manifests and Merkle-based fragment validation, yet their bindings target encoded assets and break under re-encoding or routine processing. We propose MerkleSpeech, a system for public-key verifiable, chunk-localised speech provenance offering two tiers of assurance. The first, a robust watermark attribution layer (WM-only), survives common distribution transforms and answers "was this chunk issued by a known party?". The second, a strict cryptographic integrity layer (MSv1), verifies Merkle inclusion of the chunk's fingerprint under an issuer signature. The system computes perceptual fingerprints over short speech chunks, commits them in a Merkle tree whose root is signed with an issuer key, and embeds a compact in-band watermark payload carrying a random content identifier and chunk metadata sufficient to retrieve Merkle inclusion proofs from a repository. Once the payload is extracted, all subsequent verification steps (signature check, fingerprint recomputation, Merkle inclusion) use only public information. The result is a splice-aware timeline indicating which regions pass each tier and why any given region fails. We describe the protocol, provide pseudocode, and present experiments targeting very low false positive rates under resampling, bandpass filtering, and additive noise, informed by recent audits identifying neural codecs as a major stressor for post-hoc audio watermarks.
>
---
#### [new 012] Frame-Level Internal Tool Use for Temporal Grounding in Audio LMs
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文针对音频语言模型在时间定位任务中的精度和效率问题，提出帧级内部工具使用方法，通过自有的音频表示直接进行时间定位，提升速度和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10230v1](https://arxiv.org/pdf/2602.10230v1)**

> **作者:** Joesph An; Phillip Keung; Jiaqi Wang; Orevaoghene Ahia; Noah A. Smith
>
> **备注:** Under review. See https://github.com/inkitori/taudio/
>
> **摘要:** Large audio language models are increasingly used for complex audio understanding tasks, but they struggle with temporal tasks that require precise temporal grounding, such as word alignment and speaker diarization. The standard approach, where we generate timestamps as sequences of text tokens, is computationally expensive and prone to hallucination, especially when processing audio lengths outside the model's training distribution. In this work, we propose frame-level internal tool use, a method that trains audio LMs to use their own internal audio representations to perform temporal grounding directly. We introduce a lightweight prediction mechanism trained via two objectives: a binary frame classifier and a novel inhomogeneous Poisson process (IHP) loss that models temporal event intensity. Across word localization, speaker diarization, and event localization tasks, our approach outperforms token-based baselines. Most notably, it achieves a >50x inference speedup and demonstrates robust length generalization, maintaining high accuracy on out-of-distribution audio durations where standard token-based models collapse completely.
>
---
## 更新

#### [replaced 001] Towards Efficient Speech-Text Jointly Decoding within One Speech Language Model
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音文本联合解码任务，旨在提升语音语言模型的效率与对齐质量。通过比较不同解码策略，提出一种加速的早停交错方法。**

- **链接: [https://arxiv.org/pdf/2506.04518v3](https://arxiv.org/pdf/2506.04518v3)**

> **作者:** Haibin Wu; Yuxuan Hu; Ruchao Fan; Xiaofei Wang; Kenichi Kumatani; Bo Ren; Jianwei Yu; Heng Lu; Lijuan Wang; Yao Qian; Jinyu Li
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Speech language models (Speech LMs) enable end-to-end speech-text modeling within a single model, offering a promising direction for spoken dialogue systems. The choice of speech-text jointly decoding paradigm plays a critical role in performance, efficiency, and alignment quality. In this work, we systematically compare representative joint speech-text decoding strategies, including the interleaved, and parallel generation paradigms, under a controlled experimental setup using the same base language model, speech tokenizer and training data. Our results show that the interleaved approach achieves the best alignment. However it suffers from slow inference due to long token sequence length. To address this, we propose a novel early-stop interleaved (ESI) pattern that not only significantly accelerates decoding but also yields slightly better performance. Additionally, we curate high-quality question answering (QA) datasets to further improve speech QA performance.
>
---
#### [replaced 002] SLM-S2ST: A multimodal language model for direct speech-to-speech translation
- **分类: eess.AS**

- **简介: 该论文提出SLM-S2ST，用于直接语音到语音翻译的任务。解决生成高效语音输出的问题，通过音频Transformer和流式声码器实现。**

- **链接: [https://arxiv.org/pdf/2506.04392v3](https://arxiv.org/pdf/2506.04392v3)**

> **作者:** Yuxuan Hu; Haibin Wu; Ruchao Fan; Xiaofei Wang; Heng Lu; Yao Qian; Jinyu Li
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Speech-aware language models (LMs) have demonstrated capabilities in understanding spoken language while generating text-based responses. However, enabling them to produce speech output efficiently and effectively remains a challenge. In this paper, we present SLM-S2ST, a multimodal LM for direct speech-to-speech translation (S2ST), built on the open-source Phi4-MM model. SLM-S2ST extends its predecessor by generating translated speech using an audio transformer head that predicts audio tokens with a delay relative to text tokens, followed by a streaming vocoder for waveform synthesis. Our experimental results on the CVSS-C dataset demonstrate SLM-S2ST's superior performance, significantly surpassing existing baseline models trained on the same dataset. Furthermore, when we scale up the training data and the model size, SLM-S2ST reaches on-par performance with the current SOTA model.
>
---
#### [replaced 003] Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于多模态大模型安全任务，旨在解决音频输入下的安全漏洞问题。通过构建SACRED-Bench评估攻击效果，并提出SALMONN-Guard提升防御能力。**

- **链接: [https://arxiv.org/pdf/2511.10222v3](https://arxiv.org/pdf/2511.10222v3)**

> **作者:** Yudong Yang; Xuezhen Zhang; Zhifeng Han; Siyin Wang; Jimin Zhuang; Zengrui Jin; Jing Shao; Guangzhi Sun; Chao Zhang
>
> **摘要:** Recent progress in LLMs has enabled understanding of audio signals, but has also exposed new safety risks arising from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition to enable effective black-box attacks. SACRED-Bench adopts three composition mechanisms: (a) overlap of harmful and benign speech, (b) mixture of benign speech with harmful non-speech audio, and (c) multi-speaker dialogue. These mechanisms focus on evaluating safety in settings where benign and harmful intents co-occur within a single auditory scene. Moreover, questions in SACRED-Bench are designed to implicitly refer to content in the audio, such that no explicit harmful information appears in the text prompt alone. Experiments demonstrate that even Gemini 2.5 Pro, a state-of-the-art proprietary LLM with safety guardrails fully enabled, still exhibits a 66% attack success rate. To bridge this gap, we propose SALMONN-Guard, the first guard model that jointly inspects speech, audio, and text for safety judgments, reducing the attack success rate to 20%. Our results highlight the need for audio-aware defenses to ensure the safety of multimodal LLMs. The dataset and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench.
>
---
#### [replaced 004] AUDETER: A Large-scale Dataset for Deepfake Audio Detection in Open Worlds
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于深度伪造音频检测任务，旨在解决真实世界中检测模型效果下降的问题。提出AUDETER数据集和基于课程学习的检测方法，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.04345v2](https://arxiv.org/pdf/2509.04345v2)**

> **作者:** Qizhou Wang; Hanxun Huang; Guansong Pang; Sarah Erfani; Christopher Leckie
>
> **摘要:** Speech synthesis systems can now produce highly realistic vocalisations that pose significant authenticity challenges. Despite substantial progress in deepfake detection models, their real-world effectiveness is often undermined by evolving distribution shifts between training and test data, driven by the complexity of human speech and the rapid evolution of synthesis systems. Existing datasets suffer from limited real speech diversity, insufficient coverage of recent synthesis systems, and heterogeneous mixtures of deepfake sources, which hinder systematic evaluation and open-world model training. To address these issues, we introduce AUDETER (AUdio DEepfake TEst Range), a large-scale and highly diverse deepfake audio dataset comprising over 4,500 hours of synthetic audio generated by 11 recent TTS models and 10 vocoders, totalling 3 million clips. We further observe that most existing detectors default to binary supervised training, which can induce negative transfer across synthesis sources when the training data contains highly diverse deepfake patterns, impacting overall generalisation. As a complementary contribution, we propose an effective curriculum-learning-based approach to mitigate this effect. Extensive experiments show that existing detection models struggle to generalise to novel deepfakes and human speech in AUDETER, whereas XLR-based detectors trained on AUDETER achieve strong cross-domain performance across multiple benchmarks, achieving an EER of 1.87% on In-the-Wild. AUDETER is available on GitHub.
>
---
#### [replaced 005] Physics-Guided Variational Model for Unsupervised Sound Source Tracking
- **分类: eess.AS**

- **简介: 该论文属于声源定位任务，解决无监督单声源跟踪问题。通过物理引导的变分模型，结合几何约束，从麦克风信号中直接估计声源方向，无需标签数据。**

- **链接: [https://arxiv.org/pdf/2602.08484v2](https://arxiv.org/pdf/2602.08484v2)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Bruno Defraene; Alex Young; Johan David; Ronald M. Aarts
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Sound source tracking is commonly performed using classical array-processing algorithms, while machine-learning approaches typically rely on precise source position labels that are expensive or impractical to obtain. This paper introduces a physics-guided variational model capable of fully unsupervised single-source sound source tracking. The method combines a variational encoder with a physics-based decoder that injects geometric constraints into the latent space through analytically derived pairwise time-delay likelihoods. Without requiring ground-truth labels, the model learns to estimate source directions directly from microphone array signals. Experiments on real-world data demonstrate that the proposed approach outperforms traditional baselines and achieves accuracy and computational complexity comparable to state-of-the-art supervised models. We further show that the method generalizes well to mismatched array geometries and exhibits strong robustness to corrupted microphone position metadata. Finally, we outline a natural extension of the approach to multi-source tracking and present the theoretical modifications required to support it.
>
---
#### [replaced 006] MaskVCT: Masked Voice Codec Transformer for Zero-Shot Voice Conversion With Increased Controllability via Multiple Guidances
- **分类: eess.AS; cs.AI**

- **简介: 该论文提出MaskVCT，属于语音转换任务，解决零样本语音转换中控制不足的问题。通过多因素引导，提升说话人相似度和语言可懂度。**

- **链接: [https://arxiv.org/pdf/2509.17143v2](https://arxiv.org/pdf/2509.17143v2)**

> **作者:** Junhyeok Lee; Helin Wang; Yaohan Guan; Thomas Thebaud; Laureano Moro-Velazquez; Jesús Villalba; Najim Dehak
>
> **备注:** ICASSP 2026 Accepted
>
> **摘要:** We introduce MaskVCT, a zero-shot voice conversion (VC) model that offers multi-factor controllability through multiple classifier-free guidances (CFGs). While previous VC models rely on a fixed conditioning scheme, MaskVCT integrates diverse conditions in a single model. To further enhance robustness and control, the model can leverage continuous or quantized linguistic features to enhance intelligibility and speaker similarity, and can use or omit pitch contour to control prosody. These choices allow users to seamlessly balance speaker identity, linguistic content, and prosodic factors in a zero-shot VC setting. Extensive experiments demonstrate that MaskVCT achieves the best target speaker and accent similarities while obtaining competitive word and character error rates compared to existing baselines. Audio samples are available at https://maskvct.github.io/.
>
---
#### [replaced 007] VoiceBridge: Designing Latent Bridge Models for General Speech Restoration at Scale
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出VoiceBridge，解决大规模通用语音修复任务中的多任务、高保真重建问题，通过潜在桥模型实现高效语音恢复。**

- **链接: [https://arxiv.org/pdf/2509.25275v2](https://arxiv.org/pdf/2509.25275v2)**

> **作者:** Chi Zhang; Zehua Chen; Kaiwen Zheng; Jun Zhu
>
> **摘要:** Bridge models have recently been explored for speech enhancement tasks such as denoising, dereverberation, and super-resolution, while these efforts are typically confined to a single task or small-scale datasets, with constrained general speech restoration (GSR) capability at scale. In this work, we introduce VoiceBridge, a GSR system rooted in latent bridge models (LBMs), capable of reconstructing high-fidelity speech at full-band (\textit{i.e.,} 48~kHz) from various distortions. By compressing speech waveform into continuous latent representations, VoiceBridge models the~\textit{diverse LQ-to-HQ tasks} (namely, low-quality to high-quality) in GSR with~\textit{a single latent-to-latent generative process} backed by a scalable transformer architecture. To better inherit the advantages of bridge models from the data domain to the latent space, we present an energy-preserving variational autoencoder, enhancing the alignment between the waveform and latent space over varying energy levels. Furthermore, to address the difficulty of HQ reconstruction from distinctively different LQ priors, we propose a joint neural prior, uniformly alleviating the reconstruction burden of LBM. At last, considering the key requirement of GSR systems, human perceptual quality, a perceptually aware fine-tuning stage is designed to mitigate the cascading mismatch in generation while improving perceptual alignment. Extensive validation across in-domain and out-of-domain tasks and datasets (\textit{e.g.}, refining recent zero-shot speech and podcast generation results) demonstrates the superior performance of VoiceBridge. Demo samples can be visited at: https://VoiceBridge-demo.github.io/.
>
---
#### [replaced 008] UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型任务，旨在解决音频表示与生成问题。提出ReasoningCodec音频编码器，实现音频的分层理解与高质量重建，并构建统一模型以提升少样本和零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.04683v3](https://arxiv.org/pdf/2602.04683v3)**

> **作者:** Dongchao Yang; Yuanyuan Wang; Dading Chong; Songxiang Liu; Xixin Wu; Helen Meng
>
> **摘要:** We study two foundational problems in audio language models: (1) how to design an audio tokenizer that can serve as an intermediate representation for both understanding and generation; and (2) how to build an audio foundation model that generalizes in few-shot and zero-shot settings, analogous to large language models. To this end, we make the following two contributions. First, we propose ReasoningCodec, a discrete audio codec that factorizes audio into (i) reasoning tokens, which encode text-aligned, high-level analysis and planning representations for audio understanding and hierarchical generation, and (ii) reconstruction tokens, which encode semantic-rich acoustic cues for high-fidelity waveform reconstruction. This design achieves understanding performance comparable to strong continuous representations while improving generation quality and reconstruction fidelity over prior discrete tokenizers. Second, we introduce a unified autoregressive architecture for text and audio, together with multi-stage training and multi-task data construction. Using this framework, we train UniAudio 2.0 on 100B text tokens and 60B audio tokens. Across a wide range of speech, sound, and music tasks, UniAudio 2.0 performs competitively on in-domain evaluations and demonstrates strong few-shot and zero-shot generalization to unseen tasks. Demo, code, and checkpoints will be available at \href{https://dongchaoyang.top/UniAudio2Demo/}{https://dongchaoyang.top/UniAudio2Demo/}.
>
---
#### [replaced 009] Multilingual Dysarthric Speech Assessment Using Universal Phone Recognition and Language-Specific Phonemic Contrast Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言失语语音评估任务，旨在解决跨语言智能评估方法不足的问题。通过结合通用音素识别与语言特异性音素对比建模，提出新的评估指标，提升不同语言下语音可理解性分析的准确性。**

- **链接: [https://arxiv.org/pdf/2601.21205v2](https://arxiv.org/pdf/2601.21205v2)**

> **作者:** Eunjung Yeo; Julie M. Liss; Visar Berisha; David R. Mortensen
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** The growing prevalence of neurological disorders associated with dysarthria motivates the need for automated intelligibility assessment methods that are applicalbe across languages. However, most existing approaches are either limited to a single language or fail to capture language-specific factors shaping intelligibility. We present a multilingual phoneme-production assessment framework that integrates universal phone recognition with language-specific phoneme interpretation using contrastive phonological feature distances for phone-to-phoneme mapping and sequence alignment. The framework yields three metrics: phoneme error rate (PER), phonological feature error rate (PFER), and a newly proposed alignment-free measure, phoneme coverage (PhonCov). Analysis on English, Spanish, Italian, and Tamil show that PER benefits from the combination of mapping and alignment, PFER from alignment alone, and PhonCov from mapping. Further analyses demonstrate that the proposed framework captures clinically meaningful patterns of intelligibility degradation consistent with established observations of dysarthric speech.
>
---
