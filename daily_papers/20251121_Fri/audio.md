# 音频 cs.SD;  eess.AS

- **最新发布 7 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise
- **分类: cs.SD**

- **简介: 该论文提出SceneGuard，一种训练时语音保护方法，通过添加与场景一致的可听背景噪声，抵御语音克隆攻击。针对现有不可感知扰动易被去噪等处理破坏的问题，SceneGuard利用真实声学场景噪声，实现高鲁棒性保护，在保持语音可懂度的同时显著降低说话人相似度，有效提升隐私安全性。**

- **链接: [https://arxiv.org/pdf/2511.16114v1](https://arxiv.org/pdf/2511.16114v1)**

> **作者:** Rui Sang; Yuxuan Liu
>
> **摘要:** Voice cloning technology poses significant privacy threats by enabling unauthorized speech synthesis from limited audio samples. Existing defenses based on imperceptible adversarial perturbations are vulnerable to common audio preprocessing such as denoising and compression. We propose SceneGuard, a training-time voice protection method that applies scene-consistent audible background noise to speech recordings. Unlike imperceptible perturbations, SceneGuard leverages naturally occurring acoustic scenes (e.g., airport, street, park) to create protective noise that is contextually appropriate and robust to countermeasures. We evaluate SceneGuard on text-to-speech training attacks, demonstrating 5.5% speaker similarity degradation with extremely high statistical significance (p < 10^{-15}, Cohen's d = 2.18) while preserving 98.6% speech intelligibility (STOI = 0.986). Robustness evaluation shows that SceneGuard maintains or enhances protection under five common countermeasures including MP3 compression, spectral subtraction, lowpass filtering, and downsampling. Our results suggest that audible, scene-consistent noise provides a more robust alternative to imperceptible perturbations for training-time voice protection. The source code are available at: https://github.com/richael-sang/SceneGuard.
>
---
#### [new 002] Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出Codec2Vec，一种基于离散语音编码器单元的自监督语音表征学习框架。针对传统连续输入模型存储与训练效率低的问题，利用神经语音编码器生成离散单元，实现高效存储与快速训练，同时保障良好性能，在SUPERB基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.16639v1](https://arxiv.org/pdf/2511.16639v1)**

> **作者:** Wei-Cheng Tseng; David Harwath
>
> **备注:** To be presented at ASRU 2025
>
> **摘要:** Recent advancements in neural audio codecs have not only enabled superior audio compression but also enhanced speech synthesis techniques. Researchers are now exploring their potential as universal acoustic feature extractors for a broader range of speech processing tasks. Building on this trend, we introduce Codec2Vec, the first speech representation learning framework that relies exclusively on discrete audio codec units. This approach offers several advantages, including improved data storage and transmission efficiency, faster training, and enhanced data privacy. We explore masked prediction with various training target derivation strategies to thoroughly understand the effectiveness of this framework. Evaluated on the SUPERB benchmark, Codec2Vec achieves competitive performance compared to continuous-input models while reducing storage requirements by up to 16.5x and training time by 2.3x, showcasing its scalability and efficiency.
>
---
#### [new 003] Train Short, Infer Long: Speech-LLM Enables Zero-Shot Streamable Joint ASR and Diarization on Long Audio
- **分类: eess.AS**

- **简介: 该论文提出一种端到端语音大模型Speech-LLM，解决长音频零样本流式联合语音识别与说话人分离问题。模型仅在短音频（<20s）上训练，通过动态更新的说话人提示缓存实现长音频流式推理，并支持预注册说话人信息。引入词级说话人监督提升性能，实现在保持流式特性下超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16046v1](https://arxiv.org/pdf/2511.16046v1)**

> **作者:** Mohan Shi; Xiong Xiao; Ruchao Fan; Shaoshi Ling; Jinyu Li
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** Joint automatic speech recognition (ASR) and speaker diarization aim to answer the question "who spoke what" in multi-speaker scenarios. In this paper, we present an end-to-end speech large language model (Speech-LLM) for Joint strEamable DIarization and aSr (JEDIS-LLM). The model is trained only on short audio under 20s but is capable of streamable inference on long-form audio without additional training. This is achieved by introducing a Speaker Prompt Cache (SPC) with an on-the-fly update mechanism during chunk-wise streaming inference, inspired by the autoregressive nature of LLMs. The SPC also allows the seamless use of pre-enrolled speaker profiles which is common in many scenarios like meeting transcription. To further enhance diarization capability, we incorporate word-level speaker supervision into the speech encoder during training. Experimental results demonstrate that our system outperforms strong baselines, including Sortformer and Meta-Cat in the local setting on audio up to 20s, and DiarizationLM on long-form audio, despite being fully end-to-end and streamable while DiarizationLM follows a cascaded offline pipeline. To the best of our knowledge, this is the first work enabling zero-shot streamable joint ASR and diarization on long audio using a Speech-LLM trained only on short audio, achieving state-of-the-art performance.
>
---
#### [new 004] A Generalized Weighted Overlap-Add (WOLA) Filter Bank for Improved Subband System Identification
- **分类: eess.AS; eess.SP**

- **简介: 该论文针对短时傅里叶变换域子带自适应滤波中的系统辨识问题，提出广义加权重叠相加（WOLA）滤波器组，通过前置子带滤波消除传统方法对滤波器的约束。进一步分析了均方误差性能，并提出低复杂度的每音调WOLA实现，显著提升辨识性能且保持计算效率。**

- **链接: [https://arxiv.org/pdf/2511.15766v1](https://arxiv.org/pdf/2511.15766v1)**

> **作者:** Mohit Sharma; Robbe Van Rompaey; Wouter Lanneer; Marc Moonen
>
> **备注:** For associated MatLab script: https://github.com/mohit-nith/GeneralizedWOLA-SystemIdentification.git
>
> **摘要:** This paper addresses the challenges in short-time Fourier transform (STFT) domain subband adaptive filtering, in particular, subband system identification. Previous studies in this area have primarily focused on setups with subband filtering at a downsampled rate, implemented using the weighted overlap-add (WOLA) filter bank, popular in audio and speech-processing for its reduced complexity. However, this traditional approach imposes constraints on the subband filters when transformed to their full-rate representation. This paper makes three key contributions. First, it introduces a generalized WOLA filter bank that repositions subband filters before the downsampling operation, eliminating the constraints on subband filters inherent in the conventional WOLA filter bank. Second, it investigates the mean square error (MSE) performance of the generalized WOLA filter bank for full-band system identification, establishing analytical ties between the order of subband filters, the full-band system impulse response length, the decimation factor, and the prototype filters. Third, to address the increased computational complexity of the generalized WOLA, the paper proposes a low-complexity implementation termed per-tone weighted overlap-add (PT-WOLA), which maintains computational complexity on par with conventional WOLA. Analytical and empirical evidence demonstrates that the proposed generalized WOLA filter bank significantly enhances the performance of subband system identification.
>
---
#### [new 005] SUNAC: Source-aware Unified Neural Audio Codec
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出SUNAC，一种源感知的统一神经音频编解码器，旨在解决传统神经音频编解码器对多源混合信号进行纠缠编码导致下游任务效率低的问题。通过源类型提示条件编码，实现对特定源的独立编码与选择性解码，提升语音分析、转录等应用的灵活性与效率，同时保持优异的重建与分离性能，计算成本更低。**

- **链接: [https://arxiv.org/pdf/2511.16126v1](https://arxiv.org/pdf/2511.16126v1)**

> **作者:** Ryo Aihara; Yoshiki Masuyama; Francesco Paissan; François G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Neural audio codecs (NACs) provide compact representations that can be leveraged in many downstream applications, in particular large language models. Yet most NACs encode mixtures of multiple sources in an entangled manner, which may impede efficient downstream processing in applications that need access to only a subset of the sources (e.g., analysis of a particular type of sound, transcription of a given speaker, etc). To address this, we propose a source-aware codec that encodes individual sources directly from mixtures, conditioned on source type prompts. This enables user-driven selection of which source(s) to encode, including separately encoding multiple sources of the same type (e.g., multiple speech signals). Experiments show that our model achieves competitive resynthesis and separation quality relative to a cascade of source separation followed by a conventional NAC, with lower computational cost.
>
---
#### [new 006] Difficulty-Controlled Simplification of Piano Scores with Synthetic Data for Inclusive Music Education
- **分类: cs.SD**

- **简介: 该论文针对音乐教育中AI难度调节难题，提出基于Transformer的MusicXML钢琴谱难度可控简化方法。解决现有技术依赖专有数据、格式缺乏可读性及难以复现的问题。通过合成难度配对数据集，利用预训练模型评估难度与风格，实现精准难度控制，并开源全部资源，推动包容性音乐教育发展。**

- **链接: [https://arxiv.org/pdf/2511.16228v1](https://arxiv.org/pdf/2511.16228v1)**

> **作者:** Pedro Ramoneda; Emilia Parada-Cabaleiro; Dasaem Jeong; Xavier Serra
>
> **摘要:** Despite its potential, AI advances in music education are hindered by proprietary systems that limit the democratization of technology in this domain. In particular, AI-driven music difficulty adjustment is especially promising, as simplifying complex pieces can make music education more inclusive and accessible to learners of all ages and contexts. Nevertheless, recent efforts have relied on proprietary datasets, which prevents the research community from reproducing, comparing, or extending the current state of the art. In addition, while these generative methods offer great potential, most of them use the MIDI format, which, unlike others, such as MusicXML, lacks readability and layout information, thereby limiting their practical use for human performers. This work introduces a transformer-based method for adjusting the difficulty of MusicXML piano scores. Unlike previous methods, which rely on annotated datasets, we propose a synthetic dataset composed of pairs of piano scores ordered by estimated difficulty, with each pair comprising a more challenging and easier arrangement of the same piece. We generate these pairs by creating variations conditioned on the same melody and harmony and leverage pretrained models to assess difficulty and style, ensuring appropriate pairing. The experimental results illustrate the validity of the proposed approach, showing accurate control of playability and target difficulty, as highlighted through qualitative and quantitative evaluations. In contrast to previous work, we openly release all resources (code, dataset, and models), ensuring reproducibility while fostering open-source innovation to help bridge the digital divide.
>
---
#### [new 007] Step-Audio-R1 Technical Report
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出Step-Audio-R1，首个成功实现音频领域推理的模型。针对音频语言模型常因缺乏有效推理而表现不佳的问题，研究提出模态锚定推理蒸馏框架（MGRD），使推理链基于真实声学特征，避免幻觉。实验表明其性能超越Gemini 2.5 Pro，接近Gemini 3 Pro，证明推理能力可有效迁移至音频领域。**

- **链接: [https://arxiv.org/pdf/2511.15848v1](https://arxiv.org/pdf/2511.15848v1)**

> **作者:** Fei Tian; Xiangyu Tony Zhang; Yuxin Zhang; Haoyang Zhang; Yuxin Li; Daijiao Liu; Yayue Deng; Donghang Wu; Jun Chen; Liang Zhao; Chengyuan Yao; Hexin Liu; Eng Siong Chng; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **备注:** 15 pages, 5 figures. Technical Report
>
> **摘要:** Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question - can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities.
>
---
## 更新

#### [replaced 001] Pitch Estimation With Mean Averaging Smoothed Product Spectrum And Musical Consonance Evaluation Using MASP
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2510.06625v2](https://arxiv.org/pdf/2510.06625v2)**

> **作者:** Murat Yasar Baskin
>
> **摘要:** This study introduces Mean Averaging Smoothed Product (MASP) Spectrum, which is a modified version of the Harmonic Product Spectrum, designed to enhance pitch estimation for many algorithm-wise deceptive frequency spectra that still lead clear pitches, for both harmonic and inharmonic cases. By introducing a global mean based smoothing for spectrum, the MASP algorithm diminishes the unwanted sensitivity of HPS for spectra with missing partials. The method exhibited robust pitch estimations consistent with perceptual expectations. Motivated upon the strong correlation between consonance and periodicity, the same algorithm is extended and, with the proposition of a harmonicity measure (H), used to evaluate musical consonance for two and three tones; yielding consonance hierarchies that align with perception and practice of music theory. These findings suggest that perception of pitch and consonance may share a similar underlying mechanism that depend on spectrum.
>
---
#### [replaced 002] Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks
- **分类: cs.SD; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2509.22060v2](https://arxiv.org/pdf/2509.22060v2)**

> **作者:** Aravindhan G; Yuvaraj Govindarajulu; Parin Shah
>
> **备注:** Remove due to conflict in authors
>
> **摘要:** Recent studies have demonstrated the vulnerability of Automatic Speech Recognition systems to adversarial examples, which can deceive these systems into misinterpreting input speech commands. While previous research has primarily focused on white-box attacks with constrained optimizations, and transferability based black-box attacks against commercial Automatic Speech Recognition devices, this paper explores cost efficient white-box attack and non transferability black-box adversarial attacks on Automatic Speech Recognition systems, drawing insights from approaches such as Fast Gradient Sign Method and Zeroth-Order Optimization. Further, the novelty of the paper includes how poisoning attack can degrade the performances of state-of-the-art models leading to misinterpretation of audio signals. Through experimentation and analysis, we illustrate how hybrid models can generate subtle yet impactful adversarial examples with very little perturbation having Signal Noise Ratio of 35dB that can be generated within a minute. These vulnerabilities of state-of-the-art open source model have practical security implications, and emphasize the need for adversarial security.
>
---
#### [replaced 003] Segmenting Collision Sound Sources in Egocentric Videos
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.13863v2](https://arxiv.org/pdf/2511.13863v2)**

> **作者:** Kranti Kumar Parida; Omar Emara; Hazel Doughty; Dima Damen
>
> **备注:** Webpage: https://krantiparida.github.io/projects/cs3.html
>
> **摘要:** Humans excel at multisensory perception and can often recognise object properties from the sound of their interactions. Inspired by this, we propose the novel task of Collision Sound Source Segmentation (CS3), where we aim to segment the objects responsible for a collision sound in visual input (i.e. video frames from the collision clip), conditioned on the audio. This task presents unique challenges. Unlike isolated sound events, a collision sound arises from interactions between two objects, and the acoustic signature of the collision depends on both. We focus on egocentric video, where sounds are often clear, but the visual scene is cluttered, objects are small, and interactions are brief. To address these challenges, we propose a weakly-supervised method for audio-conditioned segmentation, utilising foundation models (CLIP and SAM2). We also incorporate egocentric cues, i.e. objects in hands, to find acting objects that can potentially be collision sound sources. Our approach outperforms competitive baselines by $3\times$ and $4.7\times$ in mIoU on two benchmarks we introduce for the CS3 task: EPIC-CS3 and Ego4D-CS3.
>
---
#### [replaced 004] MMVA: Multimodal Matching Based on Valence and Arousal across Images, Music, and Musical Captions
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **链接: [https://arxiv.org/pdf/2501.01094v2](https://arxiv.org/pdf/2501.01094v2)**

> **作者:** Suhwan Choi; Kyu Won Kim; Myungjoo Kang
>
> **备注:** Paper accepted in Artificial Intelligence for Music workshop at AAAI 2025
>
> **摘要:** We introduce Multimodal Matching based on Valence and Arousal (MMVA), a tri-modal encoder framework designed to capture emotional content across images, music, and musical captions. To support this framework, we expand the Image-Music-Emotion-Matching-Net (IMEMNet) dataset, creating IMEMNet-C which includes 24,756 images and 25,944 music clips with corresponding musical captions. We employ multimodal matching scores based on the continuous valence (emotional positivity) and arousal (emotional intensity) values. This continuous matching score allows for random sampling of image-music pairs during training by computing similarity scores from the valence-arousal values across different modalities. Consequently, the proposed approach achieves state-of-the-art performance in valence-arousal prediction tasks. Furthermore, the framework demonstrates its efficacy in various zeroshot tasks, highlighting the potential of valence and arousal predictions in downstream applications.
>
---
#### [replaced 005] UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2510.04593v2](https://arxiv.org/pdf/2510.04593v2)**

> **作者:** Wenhao Guan; Zhikang Niu; Ziyue Jiang; Kaidi Wang; Peijie Chen; Qingyang Hong; Lin Li; Xie Chen
>
> **摘要:** Large language models (LLMs) have demonstrated promising performance in both automatic speech recognition (ASR) and text-to-speech (TTS) systems, gradually becoming the mainstream approach. However, most current approaches address these tasks separately rather than through a unified framework. This work aims to integrate these two tasks into one unified model. Although discrete speech tokenization enables joint modeling, its inherent information loss limits performance in both recognition and generation. In this work, we present UniVoice, a unified LLM framework through continuous representations that seamlessly integrates speech recognition and synthesis within a single model. Our approach combines the strengths of autoregressive modeling for speech recognition with flow matching for high-quality generation. To mitigate the inherent divergence between autoregressive and flow-matching models, we further design a dual attention mechanism, which switches between a causal mask for recognition and a bidirectional attention mask for synthesis. Furthermore, the proposed text-prefix-conditioned speech infilling method enables high-fidelity zero-shot voice cloning. Experimental results demonstrate that our method can achieve or exceed current single-task modeling methods in both ASR and zero-shot TTS tasks. This work explores new possibilities for end-to-end speech understanding and generation. Code is available at https://github.com/gwh22/UniVoice.
>
---
#### [replaced 006] FxSearcher: gradient-free text-driven audio transformation
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.14138v2](https://arxiv.org/pdf/2511.14138v2)**

> **作者:** Hojoon Ki; Jongsuk Kim; Minchan Kwon; Junmo Kim
>
> **摘要:** Achieving diverse and high-quality audio transformations from text prompts remains challenging, as existing methods are fundamentally constrained by their reliance on a limited set of differentiable audio effects. This paper proposes FxSearcher, a novel gradient-free framework that discovers the optimal configuration of audio effects (FX) to transform a source signal according to a text prompt. Our method employs Bayesian Optimization and CLAP-based score function to perform this search efficiently. Furthermore, a guiding prompt is introduced to prevent undesirable artifacts and enhance human preference. To objectively evaluate our method, we propose an AI-based evaluation framework. The results demonstrate that the highest scores achieved by our method on these metrics align closely with human preferences. Demos are available at https://hojoonki.github.io/FxSearcher/
>
---
#### [replaced 007] Recent Advances in Discrete Speech Tokens: A Review
- **分类: eess.AS; cs.AI; cs.MM; cs.SD; eess.SP**

- **链接: [https://arxiv.org/pdf/2502.06490v3](https://arxiv.org/pdf/2502.06490v3)**

> **作者:** Yiwei Guo; Zhihan Li; Hankun Wang; Bohan Li; Chongtian Shao; Hanglei Zhang; Chenpeng Du; Xie Chen; Shujie Liu; Kai Yu
>
> **备注:** 26 pages, 8 figures, 3 tables. This version is a major revision of the previous one, including reorganization of the section structure, more experimental results, and extensive revisions to both text and figures
>
> **摘要:** The rapid advancement of speech generation technologies in the era of large language models (LLMs) has established discrete speech tokens as a foundational paradigm for speech representation. These tokens, characterized by their discrete, compact, and concise nature, are not only advantageous for efficient transmission and storage, but also inherently compatible with the language modeling framework, enabling seamless integration of speech into text-dominated LLM architectures. Current research categorizes discrete speech tokens into two principal classes: acoustic tokens and semantic tokens, each of which has evolved into a rich research domain characterized by unique design philosophies and methodological approaches. This survey systematically synthesizes the existing taxonomy and recent innovations in discrete speech tokenization, conducts a critical examination of the strengths and limitations of each paradigm, and presents systematic experimental comparisons across token types. Furthermore, we identify persistent challenges in the field and propose potential research directions, aiming to offer actionable insights to inspire future advancements in the development and application of discrete speech tokens.
>
---
#### [replaced 008] AcousTools: A 'Full-Stack', Python-Based, Acoustic Holography Library
- **分类: cs.SD; cs.ET**

- **链接: [https://arxiv.org/pdf/2511.07336v5](https://arxiv.org/pdf/2511.07336v5)**

> **作者:** Joshua Mukherjee; Giorgos Christopoulos; Zhouyang Shen; Sriram Subramanian; Ryuji Hirayama
>
> **备注:** 14 Pages, 7 Figures, 1 Table, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Acoustic Holography is an emerging field where mid-air ultrasound is controlled and manipulated for novel and exciting applications. These range from mid-air haptics, volumetric displays, contactless fabrication, and even chemical and biomedical applications such as drug delivery. To develop these applications, a software framework to predict acoustic behaviour and simulating resulting effects, such as applied forces or scattering patterns is desirable. There have been various software libraries and platforms that attempt to fill this role, but there is yet to be a single piece of software that acts as a 'full-stack' solution. We define this full-stack as the process from abstraction to physicalisation starting with setup, modelling acoustic propagation, transducer phase retrieval, sound field analysis, and control of the acoustic holographic hardware itself. Existing methods fail to fulfil one or more of these categories. To address this, we present AcousTools, a Python-based acoustic holography library, designed to support the full suite of acoustic holographic applications and we show AcousTools's ability to meet each step of the full-stack's requirements. AcousTools has the potential to become the standard code library for acoustic holography, with the uniquely complete suite of features wrapped in a language that is known to be easy to use, AcousTools will increase the ability for researchers to develop novel applications as well as accurately review other's work. The full-stack, aside from software, will also be useful for researchers - providing a way to view and compare methodologies by understanding where they fit into the stack.
>
---
