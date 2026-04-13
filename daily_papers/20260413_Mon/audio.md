# 音频 cs.SD;  eess.AS

- **最新发布 19 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] Enhancing Conversational TTS with Cascaded Prompting and ICL-Based Online Reinforcement Learning
- **分类: eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决细粒度语音风格控制难题。通过级联提示和基于ICL的在线强化学习，提升语音自然度与表现力。**

- **链接: [https://arxiv.org/pdf/2604.08709](https://arxiv.org/pdf/2604.08709)**

> **作者:** Zhicheng Ouyang; Seong-Gyun Leem; Bach Viet Do; Haibin Wu; Ariya Rastrow; Yuzong Liu; Florian Metze
>
> **摘要:** Conversational AI has made significant progress, yet generating expressive and controllable text-to-speech (TTS) remains challenging. Specifically, controlling fine-grained voice styles and emotions is notoriously difficult and typically requires massive amounts of heavily annotated training data. To overcome this data bottleneck, we present a scalable, data-efficient cascaded framework that pairs textual style tokens with human-curated, high-quality audio prompts. This approach enables single-shot adaptation to fine-grained speaking styles and character voices. In the context of TTS, this audio prompting acts as In-Context Learning (ICL), guiding the model's prosody and timbre without requiring massive parameter updates or large-scale retraining. To further enhance generation quality and mitigate hallucinations, we introduce a novel ICL-based online reinforcement learning (RL) strategy. This strategy directly optimizes the autoregressive prosody model using subjective aesthetic rewards while being constrained by Connectionist Temporal Classification (CTC) alignment to preserve intelligibility. Comprehensive human perception evaluations demonstrate significant improvements in both the naturalness and expressivity of the synthesized speech, establishing the efficacy of our ICL-based online RL approach.
>
---
#### [new 002] GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频大语言模型的 jailbreak 攻击任务，旨在解决攻击效果与实用性能之间的平衡问题。通过频率选择性扰动，提出 GRM 框架提升攻击成功率同时保持模型实用性。**

- **链接: [https://arxiv.org/pdf/2604.09222](https://arxiv.org/pdf/2604.09222)**

> **作者:** Yunqiang Wang; Hengyuan Na; Di Wu; Miao Hu; Guocong Quan
>
> **备注:** Under Review
>
> **摘要:** Audio large language models (ALLMs) enable rich speech-text interaction, but they also introduce jailbreak vulnerabilities in the audio modality. Existing audio jailbreak methods mainly optimize jailbreak success while overlooking utility preservation, as reflected in transcription quality and question answering performance. In practice, stronger attacks often come at the cost of degraded utility. To study this trade-off, we revisit existing attacks by varying their perturbation coverage in the frequency domain, from partial-band to full-band, and find that broader frequency coverage does not necessarily improve jailbreak performance, while utility consistently deteriorates. This suggests that concentrating perturbation on a subset of bands can yield a better attack-utility trade-off than indiscriminate full-band coverage. Based on this insight, we propose GRM, a utility-aware frequency-selective jailbreak framework. It ranks Mel bands by their attack contribution relative to utility sensitivity, perturbs only a selected subset of bands, and learns a reusable universal perturbation under a semantic-preservation objective. Experiments on four representative ALLMs show that GRM achieves an average Jailbreak Success Rate (JSR) of 88.46% while providing a better attack-utility trade-off than representative baselines. These results highlight the potential of frequency-selective perturbation for better balancing attack effectiveness and utility preservation in audio jailbreak. Content Warning: This paper includes harmful query examples and unsafe model responses.
>
---
#### [new 003] DialogueSidon: Recovering Full-Duplex Dialogue Tracks from In-the-Wild Dialogue Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DialogueSidon，解决从单通道对话音频中恢复全双工对话的问题。通过结合VAE和扩散模型，实现语音分离与增强，提升可懂度和分离质量。**

- **链接: [https://arxiv.org/pdf/2604.09344](https://arxiv.org/pdf/2604.09344)**

> **作者:** Wataru Nakata; Yuki Saito; Kazuki Yamauchi; Emiru Tsunoo; Hiroshi Saruwatari
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Full-duplex dialogue audio, in which each speaker is recorded on a separate track, is an important resource for spoken dialogue research, but is difficult to collect at scale. Most in-the-wild two-speaker dialogue is available only as degraded monaural mixtures, making it unsuitable for systems requiring clean speaker-wise signals. We propose DialogueSidon, a model for joint restoration and separation of degraded monaural two-speaker dialogue audio. DialogueSidon combines a variational autoencoder (VAE) operates on the speech self-supervised learning (SSL) model feature, which compresses SSL model features into a compact latent space, with a diffusion-based latent predictor that recovers speaker-wise latent representations from the degraded mixture. Experiments on English, multilingual, and in-the-wild dialogue datasets show that DialogueSidon substantially improves intelligibility and separation quality over a baseline, while also achieving much faster inference.
>
---
#### [new 004] AudioGS: Spectrogram-Based Audio Gaussian Splatting for Sound Field Reconstruction
- **分类: cs.SD**

- **简介: 论文提出AudioGS，用于声场重建任务，解决从稀疏观测中合成高质量双耳音频的问题。通过谱图构建音频高斯，提升空间线索捕捉能力。**

- **链接: [https://arxiv.org/pdf/2604.08967](https://arxiv.org/pdf/2604.08967)**

> **作者:** Chunhao Bi; Houqiang Zhong; Zhixin Xu; Li Song; Zhengxue Cheng
>
> **摘要:** Spatial audio is fundamental to immersive virtual experiences, yet synthesizing high-fidelity binaural audio from sparse observations remains a significant challenge. Existing methods typically rely on implicit neural representations conditioned on visual priors, which often struggle to capture fine-grained acoustic structures. Inspired by 3D Gaussian Splatting (3DGS), we introduce AudioGS, a novel visual-free framework that explicitly encodes the sound field as a set of Audio Gaussians based on spectrograms. AudioGS associates each time-frequency bin with an Audio Gaussian equipped with dual Spherical Harmonic (SH) coefficients and a decay coefficient. For a target pose, we render binaural audio by evaluating the SH field to capture directionality, incorporating geometry-guided distance attenuation and phase correction, and reconstructing the waveform. Experiments on the Replay-NVAS dataset demonstrate that AudioGS successfully captures complex spatial cues and outperforms state-of-the-art visual-dependent baselines. Specifically, AudioGS reduces the magnitude reconstruction error (MAG) by over 14% and reduces the perceptual quality metric (DPAM) by approximately 25% compared to the best performing visual-guided method.
>
---
#### [new 005] AccompGen: Hierarchical Autoregressive Vocal Accompaniment Generation with Dual-Rate Codec Tokenization
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出AccompGen，用于生成与输入人声匹配的伴奏音乐。解决人声伴奏生成任务，通过双速率编码器和分层自回归架构实现高质量伴奏生成。**

- **链接: [https://arxiv.org/pdf/2604.09054](https://arxiv.org/pdf/2604.09054)**

> **作者:** Jian Zhu; Jianwei Cui; Shihao Chen; Yubang Zhang; Cheng Luo
>
> **摘要:** We present AccompGen, a system that generates instrumental music audio to accompany input vocals. Given isolated singing voice, AccompGen produces a coherent instrumental accompaniment that can be directly mixed with the input to create complete music. We propose three key innovations over prior work: (1) a dual-rate codec tokenization scheme using HuBERT semantic tokens at 50,Hz for vocals and EnCodec acoustic tokens at 75,Hz for instrumentals, enabling time-aligned yet rate-independent modeling; (2) a three-stage hierarchical autoregressive architecture (semantic to coarse acoustic to fine acoustic) with interleaved multi-codebook prediction and classifier-free guidance; and (3) modern Transformer design choices including QK-norm, GEGLU activations, RMSNorm, and T5-style relative position bias for improved training stability and sequence generalization.
>
---
#### [new 006] Phonemes vs. Projectors: An Investigation of Speech-Language Interfaces for LLM-based ASR
- **分类: eess.AS**

- **简介: 该论文研究语音识别中语音-语言接口的设计，比较了音素与投影器方法在ASR中的效果，提出BPE-phoneme接口以提升性能。**

- **链接: [https://arxiv.org/pdf/2604.09332](https://arxiv.org/pdf/2604.09332)**

> **作者:** Ziwei Li; Lukuang Dong; Saierdaer Yusuyin; Xianyu Zhao; Zhijian Ou
>
> **备注:** Update after INTERSPEECH2026 submission
>
> **摘要:** Integrating pretrained speech encoders with large language models (LLMs) is promising for ASR, but performance and data efficiency depend on the speech-language interface. A common choice is a learned projector that maps encoder features into the LLM embedding space, whereas an alternative is to expose discrete phoneme sequences to the LLM. Using the same encoder and LLM backbones, we compare phoneme-based and vanilla projector-based interfaces in high-resource English and low-resource Tatar. We also propose a BPE-phoneme interface that groups frequent local phoneme patterns while preserving explicit word-boundary cues for phoneme-to-grapheme generation. On LibriSpeech, the phoneme-based interface is competitive with the vanilla projector, and the BPE-phoneme interface yields further gains. On Tatar, the phoneme-based interface substantially outperforms the vanilla projector. We further find that phoneme supervision yields a phoneme-informed hybrid interface that is stronger than the vanilla projector.
>
---
#### [new 007] Script Collapse in Multilingual ASR: Defining and Measuring Script Fidelity Rate
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，解决模型输出错误书写系统的问题。提出Script Fidelity Rate（SFR）度量指标，分析多语言ASR模型的脚本一致性。**

- **链接: [https://arxiv.org/pdf/2604.08786](https://arxiv.org/pdf/2604.08786)**

> **作者:** Hanif Rahman
>
> **摘要:** Word error rate (WER) is the dominant metric for automatic speech recognition, yet it cannot detect a systematic failure mode: models that produce fluent output in the wrong writing system. We define Script Fidelity Rate (SFR), the fraction of hypothesis characters in the target script block, computable without reference transcriptions, and report the first systematic measurement of script collapse across six languages spanning four writing systems (Pashto, Urdu, Hindi, Bengali, Malayalam, Somali) and nine ASR models on FLEURS test sets. Across 53 evaluated model-language pairs, 18 (34%; 95% Wilson CI: 23-47%) exhibit script collapse (SFR < 10%); MMS-1B and SeamlessM4T-v2 maintain SFR above 99% on every language evaluated, confirming that SFR correctly identifies high fidelity where it is present. We identify three distinct collapse patterns: Latin phonetic substitution (smaller Whisper on Indic languages), Arabic substitution for Somali's Latin-script orthography, and Devanagari substitution where larger Whisper models treat all Indic audio as Hindi, a failure present even in Whisper large-v3.
>
---
#### [new 008] Data Selection Effects on Self-Supervised Learning of Audio Representations for French Audiovisual Broadcasts
- **分类: eess.AS**

- **简介: 该论文研究自监督学习在法语音视频广播中的数据选择影响，旨在提升音频表示效果。通过构建多样化数据集，评估不同数据对下游任务的影响，并探讨数据去重的重要性。**

- **链接: [https://arxiv.org/pdf/2604.09472](https://arxiv.org/pdf/2604.09472)**

> **作者:** Valentin Pelloin; Lina Bekkali; Reda Dehak; David Doukhan
>
> **备注:** To be published in the Fifteenth International Conference on Language Resources and Evaluation (LREC 2026)
>
> **摘要:** Audio and speech self-supervised encoder models are now widely used for a lot of different tasks. Many of these models are often trained on clean segmented speech content such as LibriSpeech. In this paper, we look into how the pretraining datasets of such SSL (Self-Supervised Learning) models impact their downstream results. We build a large pretraining corpus of highly diverse TV and Radio broadcast audio content, which we describe with automatic tools. We use these annotations to build smaller subsets, which we use to train audio SSL models. Then, we evaluate the models on multiple downstream tasks such as automatic speech recognition, voice activity and music detection, or speaker recognition. The results show the potential of pretraining SSL models on diverse audio content without restricting it to speech. We also perform a membership inference attack to evaluate the encoder ability to memorize their training datasets, which highlight the importance of data deduplication. This unified training could bridge speech and music machine learning communities.
>
---
#### [new 009] AudioGuard: Toward Comprehensive Audio Safety Protection Across Diverse Threat Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频安全任务，旨在解决音频系统中的多种潜在威胁。通过构建AudioSafetyBench和提出AudioGuard，提升音频内容的安全防护能力。**

- **链接: [https://arxiv.org/pdf/2604.08867](https://arxiv.org/pdf/2604.08867)**

> **作者:** Mintong Kang; Chen Fang; Bo Li
>
> **摘要:** Audio has rapidly become a primary interface for foundation models, powering real-time voice assistants. Ensuring safety in audio systems is inherently more complex than just "unsafe text spoken aloud": real-world risks can hinge on audio-native harmful sound events, speaker attributes (e.g., child voice), impersonation/voice-cloning misuse, and voice-content compositional harms, such as child voice plus sexual content. The nature of audio makes it challenging to develop comprehensive benchmarks or guardrails against this unique risk landscape. To close this gap, we conduct large-scale red teaming on audio systems, systematically uncover vulnerabilities in audio, and develop a comprehensive, policy-grounded audio risk taxonomy and AudioSafetyBench, the first policy-based audio safety benchmark across diverse threat models. AudioSafetyBench supports diverse languages, suspicious voices (e.g., celebrity/impersonation and child voice), risky voice-content combinations, and non-speech sound events. To defend against these threats, we propose AudioGuard, a unified guardrail consisting of 1) SoundGuard for waveform-level audio-native detection and 2) ContentGuard for policy-grounded semantic protection. Extensive experiments on AudioSafetyBench and four complementary benchmarks show that AudioGuard consistently improves guardrail accuracy over strong audio-LLM-based baselines with substantially lower latency.
>
---
#### [new 010] Discrete Token Modeling for Multi-Stem Music Source Separation with Language Models
- **分类: eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在将多音轨分离问题转化为条件离散标记生成。通过结合编码器、音频编解码器和语言模型，实现高质量的音频生成与分离。**

- **链接: [https://arxiv.org/pdf/2604.09371](https://arxiv.org/pdf/2604.09371)**

> **作者:** Pengbo Lyu; Xiangyu Zhao; Chengwei Liu; Haoyin Yan; Xiaotao Liang; Hongyu Wang; Shaofei Xue
>
> **备注:** 5 pages, 2 figures, 3 tables. Submitted to INTERSPEECH 2026
>
> **摘要:** We propose a generative framework for multi-track music source separation (MSS) that reformulates the task as conditional discrete token generation. Unlike conventional approaches that directly estimate continuous signals in the time or frequency domain, our method combines a Conformer-based conditional encoder, a dual-path neural audio codec (HCodec), and a decoder-only language model to autoregressively generate audio tokens for four target tracks. The generated tokens are decoded back to waveforms through the codec decoder. Evaluation on the MUSDB18-HQ benchmark shows that our generative approach achieves perceptual quality approaching state-of-the-art discriminative methods, while attaining the highest NISQA score on the vocals track. Ablation studies confirm the effectiveness of the learnable Conformer encoder and the benefit of sequential cross-track generation.
>
---
#### [new 011] PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing
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
#### [new 012] Few-Shot Contrastive Adaptation for Audio Abuse Detection in Low-Resource Indic Languages
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频滥用检测任务，旨在解决低资源印地语环境下的检测问题。通过对比音频-文本预训练模型CLAP，在少量样本下进行适应性学习，验证其跨语言有效性。**

- **链接: [https://arxiv.org/pdf/2604.09094](https://arxiv.org/pdf/2604.09094)**

> **作者:** Aditya Narayan Sankaran; Reza Farahbakhsh; Noel Crespi
>
> **备注:** 14 pages, preprint under review
>
> **摘要:** Abusive speech detection is becoming increasingly important as social media shifts towards voice-based interaction, particularly in multilingual and low-resource settings. Most current systems rely on automatic speech recognition (ASR) followed by text-based hate speech classification, but this pipeline is vulnerable to transcription errors and discards prosodic information carried in speech. We investigate whether Contrastive Language-Audio Pre-training (CLAP) can support abusive speech detection directly from audio. Using the ADIMA dataset, we evaluate CLAP-based representations under few-shot supervised contrastive adaptation in cross-lingual and leave-one-language-out settings, with zero-shot prompting included as an auxiliary analysis. Our results show that CLAP yields strong cross-lingual audio representations across ten Indic languages, and that lightweight projection-only adaptation achieves competitive performance with respect to fully supervised systems trained on complete training data. However, the benefits of few-shot adaptation are language-dependent and not monotonic with shot size. These findings suggest that contrastive audio-text models provide a promising basis for cross-lingual audio abuse detection in low-resource settings, while also indicating that transfer remains incomplete and language-specific in important ways.
>
---
#### [new 013] Noise-Aware In-Context Learning for Hallucination Mitigation in ALLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频语言模型任务，旨在解决 hallucination 问题。通过构建噪声先验库和引入上下文学习，降低模型生成中的错误联想，提升可靠性。**

- **链接: [https://arxiv.org/pdf/2604.09021](https://arxiv.org/pdf/2604.09021)**

> **作者:** Qixuan Huang; Khalid Zaman; Masashi Unoki
>
> **摘要:** Auditory large language models (ALLMs) have demonstrated strong general capabilities in audio understanding and reasoning tasks. However, their reliability is still undermined by hallucination issues. Existing hallucination evaluation methods are formulated as binary classification tasks, which are insufficient to characterize the more complex hallucination patterns that arise in generative tasks. Moreover, current hallucination mitigation strategies rely on fine-tuning, resulting in high computational costs. To address the above limitations, we propose a plug-and-play Noise-Aware In-Context Learning (NAICL) method. Specifically, we construct a noise prior library, retrieve noise examples relevant to the input audio, and incorporate them as contextual priors, thereby guiding the model to reduce speculative associations when acoustic evidence is insufficient and to adopt a more conservative generation strategy. In addition, we establish a hallucination benchmark for audio caption tasks including the construction of the Clotho-1K multi-event benchmark dataset, the definition of four types of auditory hallucinations, and the introduction of metrics such as hallucination type distribution to support fine-grained analysis. Experimental results show that all evaluated ALLMs exhibit same hallucination behaviors. Moreover, the proposed NAICL method reduces the overall hallucination rate from 26.53% to 16.98%.
>
---
#### [new 014] DDSP-QbE++: Improving Speech Quality for Speech Anonymisation for Atypical Speech
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音匿名化任务，针对DDSP-QbE中因相位累积导致的伪影问题，提出改进方法，提升语音质量。**

- **链接: [https://arxiv.org/pdf/2604.09246](https://arxiv.org/pdf/2604.09246)**

> **作者:** Suhita Ghosh; Yamini Sinha; Sebastian Stober
>
> **备注:** accepted in CHI workshop (Speech AI For All) 2026
>
> **摘要:** Differentiable Digital Signal Processing (DDSP) pipelines for voice conversion rely on subtractive synthesis, where a periodic excitation signal is shaped by a learned spectral envelope to reconstruct the target voice. In DDSP-QbE, the excitation is generated via phase accumulation, producing a sawtooth-like waveform whose abrupt discontinuities introduce aliasing artefacts that manifest perceptually as buzziness and spectral distortion, particularly at higher fundamental frequencies. We propose two targeted improvements to the excitation stage of the DDSP-QbE subtractive synthesizer. First, we incorporate explicit voicing detection to gate the harmonic excitation, suppressing the periodic component in unvoiced regions and replacing it with filtered noise, thereby avoiding aliased harmonic content where it is most perceptually disruptive. Second, we apply Polynomial Band-Limited Step (PolyBLEP) correction to the phase-accumulated oscillator, substituting the hard waveform discontinuity at each phase wrap with a smooth polynomial residual that cancels alias-generating components without oversampling or spectral truncation. Together, these modifications yield a cleaner harmonic roll-off, reduced high-frequency artefacts, and improved perceptual naturalness, as measured by MOS. The proposed approach is lightweight, differentiable, and integrates seamlessly into the existing DDSP-QbE training pipeline with no additional learnable parameters.
>
---
#### [new 015] LatentFlowSR: High-Fidelity Audio Super-Resolution via Noise-Robust Latent Flow Matching
- **分类: cs.SD**

- **简介: 该论文属于音频超分辨率任务，旨在恢复低分辨率音频的高频细节。提出LatentFlowSR方法，通过潜在空间的条件流匹配实现高质量音频重建。**

- **链接: [https://arxiv.org/pdf/2604.09188](https://arxiv.org/pdf/2604.09188)**

> **作者:** Fei Liu; Yang Ai; Hui-Peng Du; Yu-Fei Shi; Zhen-Hua Ling
>
> **摘要:** Audio super-resolution aims to recover missing high-frequency details from bandwidth-limited low-resolution audio, thereby improving the naturalness and perceptual quality of the reconstructed signal. However, most existing methods directly operate in the waveform or time-frequency domain, which not only involves high-dimensional generation spaces but is also largely limited to speech tasks, leaving substantial room for improvement on more complex audio types such as sound effects and music. To mitigate these limitations, we introduce LatentFlowSR, a new audio super-resolution approach that leverages conditional flow matching (CFM) within a latent representation space. Specifically, we first train a noise-robust autoencoder, which encodes low-resolution audio into a continuous latent space. Conditioned on the low-resolution latent representation, a CFM mechanism progressively generates the corresponding high-resolution latent representation from a Gaussian prior with a one-step ordinary differential equation (ODE) solver. The resulting high-resolution latent representation is then decoded by the pretrained autoencoder to reconstruct the high-resolution audio. Experimental results demonstrate that LatentFlowSR consistently outperforms baseline methods across various audio types and super-resolution settings. These results indicate that the proposed method possesses strong high-frequency reconstruction capability and robust generalization performance, providing compelling evidence for the effectiveness of latent-space modeling in audio super-resolution. All relevant code will be made publicly available upon completion of the paper review process.
>
---
#### [new 016] Tora3: Trajectory-Guided Audio-Video Generation with Physical Coherence
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音视频生成任务，旨在解决运动与声音关系不协调的问题。提出Tora3框架，利用物体轨迹提升物理一致性，增强运动与声音同步。**

- **链接: [https://arxiv.org/pdf/2604.09057](https://arxiv.org/pdf/2604.09057)**

> **作者:** Junchao Liao; Zhenghao Zhang; Xiangyu Meng; Litao Li; Ziying Zhang; Siyu Zhu; Long Qin; Weizhi Wang
>
> **摘要:** Audio-video (AV) generation has recently made strong progress in perceptual quality and multimodal coherence, yet generating content with plausible motion-sound relations remains challenging. Existing methods often produce object motions that are visually unstable and sounds that are only loosely aligned with salient motion or contact events, largely because they lack an explicit motion-aware structure shared by video and audio generation. We present Tora3, a trajectory-guided AV generation framework that improves physical coherence by using object trajectories as a shared kinematic prior. Rather than treating trajectories as a video-only control signal, Tora3 uses them to jointly guide visual motion and acoustic events. Specifically, we design a trajectory-aligned motion representation for video, a kinematic-audio alignment module driven by trajectory-derived second-order kinematic states, and a hybrid flow matching scheme that preserves trajectory fidelity in trajectory-conditioned regions while maintaining local coherence elsewhere. We further curate PAV, a large-scale AV dataset emphasizing motion-relevant patterns with automatically extracted motion annotations. Extensive experiments show that Tora3 improves motion realism, motion-sound synchronization, and overall AV generation quality over strong open-source baselines.
>
---
#### [new 017] Accessible Fine-grained Data Representation via Spatial Audio
- **分类: cs.HC; cs.SD**

- **简介: 该论文属于数据可视化任务，旨在解决盲人和低视力用户难以获取细粒度数据信息的问题。通过空间音频技术，将数据值表示为声音方向，提升数据细节的可访问性。**

- **链接: [https://arxiv.org/pdf/2604.08979](https://arxiv.org/pdf/2604.08979)**

> **作者:** Can Liu; Wenjie Jiang; Shaolun Ruan; Kotaro Hara; Yong Wang
>
> **备注:** Accepted by IEEE Computer Graphics and Applications (IEEE CG&A)
>
> **摘要:** Pitch-based sonification of quantitative data increases the accessibility of data visualizations that are otherwise inaccessible for blind and low-vision (BLV) individuals. We argue that, although pitch representations can reveal the coarse-grained information of data, such as data trend and value comparison, they cannot effectively convey the fine-grained details like the sign and exact value of individual data points. Informed by existing sound perception research, we propose a spatial audio-based approach by representing data values as the sound direction in the azimuth plane to achieve accessible fine-grained data representation. We conducted a user study with 26 participants (including 10 BLV participants) on four data perception tasks. The results show our approach significantly outperforms pitch representation on fine-grained data perception tasks like recognizing data signs and exact values, and performs similarly on data trend identification, despite its inferior accuracy on data value comparison.
>
---
#### [new 018] Neural networks for Text-to-Speech evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音质量评估任务，旨在解决人工评估成本高、效率低的问题。通过构建神经网络模型（如NeuralSBS和MOSNet）实现自动化评估，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.08562](https://arxiv.org/pdf/2604.08562)**

> **作者:** Ilya Trofimenko; David Kocharyan; Aleksandr Zaitsev; Pavel Repnikov; Mark Levin; Nikita Shevtsov
>
> **摘要:** Ensuring that Text-to-Speech (TTS) systems deliver human-perceived quality at scale is a central challenge for modern speech technologies. Human subjective evaluation protocols such as Mean Opinion Score (MOS) and Side-by-Side (SBS) comparisons remain the de facto gold standards, yet they are expensive, slow, and sensitive to pervasive assessor biases. This study addresses these barriers by formulating, and implementing a suite of novel neural models designed to approximate expert judgments in both relative (SBS) and absolute (MOS) settings. For relative assessment, we propose NeuralSBS, a HuBERT-backed model achieving 73.7% accuracy (on SOMOS dataset). For absolute assessment, we introduce enhancements to MOSNet using custom sequence-length batching, as well as WhisperBert, a multimodal stacking ensemble that combines Whisper audio features and BERT textual embeddings via weak learners. Our best MOS models achieve a Root Mean Square Error (RMSE) of ~0.40, significantly outperforming the human inter-rater RMSE baseline of 0.62. Furthermore, our ablation studies reveal that naively fusing text via cross-attention can degrade performance, highlighting the effectiveness of ensemble-based stacking over direct latent fusion. We additionally report negative results with SpeechLM-based architectures and zero-shot LLM evaluators (Qwen2-Audio, Gemini 2.5 flash preview), reinforcing the necessity of dedicated metric learning frameworks.
>
---
#### [new 019] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和交互纠错缺失的问题。提出基于LLM的语义评估和交互框架，提升识别语义准确性和交互能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
## 更新

#### [replaced 001] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于音乐音频-视觉问答任务，旨在解决音乐领域中多模态理解的特殊挑战。通过分析数据集和方法，提出针对性的模型设计与策略。**

- **链接: [https://arxiv.org/pdf/2505.20638](https://arxiv.org/pdf/2505.20638)**

> **作者:** Wenhao You; Xingjian Diao; Wenjun Huang; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Tingxuan Wu; Ming Cheng; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to Annual Meeting of the Association for Computational Linguistics (ACL 2026). The first two authors contributed equally
>
> **摘要:** While recent Multimodal Large Language Models exhibit impressive capabilities for general multimodal tasks, specialized domains like music necessitate tailored approaches. Music Audio-Visual Question Answering (Music AVQA) particularly underscores this, presenting unique challenges with its continuous, densely layered audio-visual content, intricate temporal dynamics, and the critical need for domain-specific knowledge. Through a systematic analysis of Music AVQA datasets and methods, this paper identifies that specialized input processing, architectures incorporating dedicated spatial-temporal designs, and music-specific modeling strategies are critical for success in this domain. Our study provides valuable insights for researchers by highlighting effective design patterns empirically linked to strong performance, proposing concrete future directions for incorporating musical priors, and aiming to establish a robust foundation for advancing multimodal musical understanding. We aim to encourage further research in this area and provide a GitHub repository of relevant works: this https URL.
>
---
#### [replaced 002] DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos
- **分类: cs.SD**

- **简介: 该论文属于空间音频生成任务，旨在解决360度视频缺乏空间音频的问题。通过整合动态场景重建与条件扩散模型，生成高质量第一阶Ambisonics音频。**

- **链接: [https://arxiv.org/pdf/2604.02781](https://arxiv.org/pdf/2604.02781)**

> **作者:** Ziyu Luo; Lin Chen; Qiang Qu; Xiaoming Chen; Yiran Shen
>
> **备注:** Accidental duplicate submission. This paper was intended to be a replacement (v2) for arXiv:2602.06846
>
> **摘要:** Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
>
---
#### [replaced 003] Is ASMR Engineerable? A Signal Processing and User Experience Study
- **分类: eess.AS**

- **简介: 该论文属于信号处理与用户体验研究，旨在探讨ASMR是否可被工程化。通过设计声学模式并进行用户实验，分析其对ASMR效果的影响，验证了信号级工程的可行性。**

- **链接: [https://arxiv.org/pdf/2504.00621](https://arxiv.org/pdf/2504.00621)**

> **作者:** Zexin Fang; Bin Han; Henrik H. Sveen; C. Clark Cao; Hans D. Schotten
>
> **备注:** Submitted to IEEE Transactions on Human-Machine Systems
>
> **摘要:** Autonomous Sensory Meridian Response (ASMR) has been remarkably popular in the recent decade, yet whether its effects can be deliberately engineered remains an open question. While ASMR effects validated through behavioral studies and neuro-physiological measurements such as electroencephalography (EEG) and related bio-signals, the acoustic mechanisms that trigger it remain poorly understood. We investigate whether ASMR responses can be systematically induced through controlled acoustic design, hypothesizing that cyclic patterns where predictability drives relaxation and variation sustains intrigue are key engineerable parameters. Specifically, we design cyclic sound patterns with varying predictability and randomness, and evaluate their effects via a structured user study. Signal processing-based feature extraction and regression analysis are used to establish an interpretable mapping between acoustic structure and perceived ASMR effects. Results show that relaxing effects accumulate progressively, are independent of spatial orientation, and remain stable across time. Crucially, smoothly spread, energy-dense cyclic patterns most effectively trigger ASMR, suggesting that signal-level engineering of ASMR experiences is achievable
>
---
