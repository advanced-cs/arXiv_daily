# 音频 cs.SD;  eess.SP

- **最新发布 24 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; stat.ML**

- **简介: 该论文提出FasterVoiceGrad，一种更快的一步扩散语音转换模型，通过对抗性扩散转换蒸馏同时蒸馏扩散模型和内容编码器，解决传统方法计算复杂、速度慢的问题，在保持性能的同时显著提升速度。**

- **链接: [http://arxiv.org/pdf/2508.17868v1](http://arxiv.org/pdf/2508.17868v1)**

> **作者:** Takuhiro Kaneko; Hirokazu Kameoka; Kou Tanaka; Yuto Kondo
>
> **备注:** Accepted to Interspeech 2025. Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/fastervoicegrad/
>
> **摘要:** A diffusion-based voice conversion (VC) model (e.g., VoiceGrad) can achieve high speech quality and speaker similarity; however, its conversion process is slow owing to iterative sampling. FastVoiceGrad overcomes this limitation by distilling VoiceGrad into a one-step diffusion model. However, it still requires a computationally intensive content encoder to disentangle the speaker's identity and content, which slows conversion. Therefore, we propose FasterVoiceGrad, a novel one-step diffusion-based VC model obtained by simultaneously distilling a diffusion model and content encoder using adversarial diffusion conversion distillation (ADCD), where distillation is performed in the conversion process while leveraging adversarial and score distillation training. Experimental evaluations of one-shot VC demonstrated that FasterVoiceGrad achieves competitive VC performance compared to FastVoiceGrad, with 6.6-6.9 and 1.8 times faster speed on a GPU and CPU, respectively.
>
---
#### [new 002] WildSpoof Challenge Evaluation Plan
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出WildSpoof挑战评估计划，旨在推动真实场景下语音合成与防欺骗语音验证的研究。通过两个独立但关联的赛道，促进使用真实数据集和跨领域协作，提升系统在实际应用中的鲁棒性与真实性。**

- **链接: [http://arxiv.org/pdf/2508.16858v1](http://arxiv.org/pdf/2508.16858v1)**

> **作者:** Yihan Wu; Jee-weon Jung; Hye-jin Shim; Xin Cheng; Xin Wang
>
> **备注:** ICASSP 2026 challenge
>
> **摘要:** The WildSpoof Challenge aims to advance the use of in-the-wild data in two intertwined speech processing tasks. It consists of two parallel tracks: (1) Text-to-Speech (TTS) synthesis for generating spoofed speech, and (2) Spoofing-robust Automatic Speaker Verification (SASV) for detecting spoofed speech. While the organizers coordinate both tracks and define the data protocols, participants treat them as separate and independent tasks. The primary objectives of the challenge are: (i) to promote the use of in-the-wild data for both TTS and SASV, moving beyond conventional clean and controlled datasets and considering real-world scenarios; and (ii) to encourage interdisciplinary collaboration between the spoofing generation (TTS) and spoofing detection (SASV) communities, thereby fostering the development of more integrated, robust, and realistic systems.
>
---
#### [new 003] TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出TaDiCodec，一种文本感知的扩散语音分词器，用于语音语言建模。解决现有分词器依赖复杂结构、预训练模型和两阶段训练的问题。通过单阶段端到端优化实现低帧率（6.25 Hz）与高重建质量，兼容零样本文语转换。**

- **链接: [http://arxiv.org/pdf/2508.16790v1](http://arxiv.org/pdf/2508.16790v1)**

> **作者:** Yuancheng Wang; Dekun Chen; Xueyao Zhang; Junan Zhang; Jiaqi Li; Zhizheng Wu
>
> **摘要:** Speech tokenizers serve as foundational components for speech language models, yet current designs exhibit several limitations, including: 1) dependence on multi-layer residual vector quantization structures or high frame rates, 2) reliance on auxiliary pre-trained models for semantic distillation, and 3) requirements for complex two-stage training processes. In this work, we introduce the Text-aware Diffusion Transformer Speech Codec (TaDiCodec), a novel approach designed to overcome these challenges. TaDiCodec employs end-to-end optimization for quantization and reconstruction through a diffusion autoencoder, while integrating text guidance into the diffusion decoder to enhance reconstruction quality and achieve optimal compression. TaDiCodec achieves an extremely low frame rate of 6.25 Hz and a corresponding bitrate of 0.0875 kbps with a single-layer codebook for 24 kHz speech, while maintaining superior performance on critical speech generation evaluation metrics such as Word Error Rate (WER), speaker similarity (SIM), and speech quality (UTMOS). Notably, TaDiCodec employs a single-stage, end-to-end training paradigm, and obviating the need for auxiliary pre-trained models. We also validate the compatibility of TaDiCodec in language model based zero-shot text-to-speech with both autoregressive modeling and masked generative modeling, demonstrating its effectiveness and efficiency for speech language modeling, as well as a significantly small reconstruction-generation gap. We will open source our code and model checkpoints. Audio samples are are available at https:/tadicodec.github.io/. We release code and model checkpoints at https:/github.com/HeCheng0625/Diffusion-Speech-Tokenizer.
>
---
#### [new 004] ClearMask: Noise-Free and Naturalness-Preserving Protection Against Voice Deepfake Attacks
- **分类: cs.SD; cs.CR**

- **简介: 该论文针对语音深度伪造攻击提出ClearMask防御机制，通过频域滤波和风格迁移在不引入噪声的前提下保护语音自然度，有效抵御黑盒攻击并适应实时场景。**

- **链接: [http://arxiv.org/pdf/2508.17660v1](http://arxiv.org/pdf/2508.17660v1)**

> **作者:** Yuanda Wang; Bocheng Chen; Hanqing Guo; Guangjing Wang; Weikang Ding; Qiben Yan
>
> **备注:** 14 Pages, Accepted by AsiaCCS 2025
>
> **摘要:** Voice deepfake attacks, which artificially impersonate human speech for malicious purposes, have emerged as a severe threat. Existing defenses typically inject noise into human speech to compromise voice encoders in speech synthesis models. However, these methods degrade audio quality and require prior knowledge of the attack approaches, limiting their effectiveness in diverse scenarios. Moreover, real-time audios, such as speech in virtual meetings and voice messages, are still exposed to voice deepfake threats. To overcome these limitations, we propose ClearMask, a noise-free defense mechanism against voice deepfake attacks. Unlike traditional approaches, ClearMask modifies the audio mel-spectrogram by selectively filtering certain frequencies, inducing a transferable voice feature loss without injecting noise. We then apply audio style transfer to further deceive voice decoders while preserving perceived sound quality. Finally, optimized reverberation is introduced to disrupt the output of voice generation models without affecting the naturalness of the speech. Additionally, we develop LiveMask to protect streaming speech in real-time through a universal frequency filter and reverberation generator. Our experimental results show that ClearMask and LiveMask effectively prevent voice deepfake attacks from deceiving speaker verification models and human listeners, even for unseen voice synthesis models and black-box API services. Furthermore, ClearMask demonstrates resilience against adaptive attackers who attempt to recover the original audio signal from the protected speech samples.
>
---
#### [new 005] Multi-Metric Preference Alignment for Generative Speech Restoration
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文聚焦语音恢复任务，解决生成模型训练目标与人类感知偏好不一致的问题。提出多指标偏好对齐策略，构建80K偏好数据集，通过DPO优化提升模型质量，并验证其在多种生成模型中的有效性及作为伪标签生成器的潜力。**

- **链接: [http://arxiv.org/pdf/2508.17229v1](http://arxiv.org/pdf/2508.17229v1)**

> **作者:** Junan Zhang; Xueyao Zhang; Jing Yang; Yuancheng Wang; Fan Fan; Zhizheng Wu
>
> **备注:** 16 pages, 10 figures. demopage: https://gensr-pref.github.io
>
> **摘要:** Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:https://gensr-pref.github.io
>
---
#### [new 006] RephraseTTS: Dynamic Length Text based Speech Insertion with Speaker Style Transfer
- **分类: cs.SD; cs.CL**

- **简介: 论文提出RephraseTTS方法，解决文本条件下的语音插入任务，即根据文本修改语音内容并保持原说话人特征。采用基于Transformer的非自回归模型，动态确定插入长度，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.17031v1](http://arxiv.org/pdf/2508.17031v1)**

> **作者:** Neeraj Matiyali; Siddharth Srivastava; Gaurav Sharma
>
> **摘要:** We propose a method for the task of text-conditioned speech insertion, i.e. inserting a speech sample in an input speech sample, conditioned on the corresponding complete text transcript. An example use case of the task would be to update the speech audio when corrections are done on the corresponding text transcript. The proposed method follows a transformer-based non-autoregressive approach that allows speech insertions of variable lengths, which are dynamically determined during inference, based on the text transcript and tempo of the available partial input. It is capable of maintaining the speaker's voice characteristics, prosody and other spectral properties of the available speech input. Results from our experiments and user study on LibriTTS show that our method outperforms baselines based on an existing adaptive text to speech method. We also provide numerous qualitative results to appreciate the quality of the output from the proposed method.
>
---
#### [new 007] Dynamic Fusion Multimodal Network for SpeechWellness Detection
- **分类: cs.SD; cs.AI**

- **简介: 论文提出动态融合多模态网络用于SpeechWellness检测，解决青少年自杀风险预测中单一模态信息不足的问题。通过融合语音时域、时频域特征与语义信息，并引入可学习权重的动态融合机制，提升模型准确性与效率。**

- **链接: [http://arxiv.org/pdf/2508.18057v1](http://arxiv.org/pdf/2508.18057v1)**

> **作者:** Wenqiang Sun; Han Yin; Jisheng Bai; Jianfeng Chen
>
> **备注:** 6 pages, 5figures
>
> **摘要:** Suicide is one of the leading causes of death among adolescents. Previous suicide risk prediction studies have primarily focused on either textual or acoustic information in isolation, the integration of multimodal signals, such as speech and text, offers a more comprehensive understanding of an individual's mental state. Motivated by this, and in the context of the 1st SpeechWellness detection challenge, we explore a lightweight multi-branch multimodal system based on a dynamic fusion mechanism for speechwellness detection. To address the limitation of prior approaches that rely on time-domain waveforms for acoustic analysis, our system incorporates both time-domain and time-frequency (TF) domain acoustic features, as well as semantic representations. In addition, we introduce a dynamic fusion block to adaptively integrate information from different modalities. Specifically, it applies learnable weights to each modality during the fusion process, enabling the model to adjust the contribution of each modality. To enhance computational efficiency, we design a lightweight structure by simplifying the original baseline model. Experimental results demonstrate that the proposed system exhibits superior performance compared to the challenge baseline, achieving a 78% reduction in model parameters and a 5% improvement in accuracy.
>
---
#### [new 008] Enhancing Speech Emotion Recognition with Multi-Task Learning and Dynamic Feature Fusion
- **分类: cs.SD**

- **简介: 论文针对语音情感识别任务，提出多任务学习框架与动态特征融合方法，解决情感分类中类别不平衡和语义混淆问题，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2508.17878v1](http://arxiv.org/pdf/2508.17878v1)**

> **作者:** Honghong Wang; Jing Deng; Fanqin Meng; Rong Zheng
>
> **备注:** accepted by interspeech2025
>
> **摘要:** This study investigates fine-tuning self-supervised learn ing (SSL) models using multi-task learning (MTL) to enhance speech emotion recognition (SER). The framework simultane ously handles four related tasks: emotion recognition, gender recognition, speaker verification, and automatic speech recog nition. An innovative co-attention module is introduced to dy namically capture the interactions between features from the primary emotion classification task and auxiliary tasks, en abling context-aware fusion. Moreover, We introduce the Sam ple Weighted Focal Contrastive (SWFC) loss function to ad dress class imbalance and semantic confusion by adjusting sam ple weights for difficult and minority samples. The method is validated on the Categorical Emotion Recognition task of the Speech Emotion Recognition in Naturalistic Conditions Chal lenge, showing significant performance improvements.
>
---
#### [new 009] Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework
- **分类: cs.SD; cs.AI**

- **简介: 论文提出多模态语音增强框架，融合骨传导麦克风（BMS）与空气传导麦克风（AMS）信号，解决BMS高频信息缺失问题。通过专用网络分别增强BMS和去噪AMS，并动态融合以适应噪声环境，提升语音质量。**

- **链接: [http://arxiv.org/pdf/2508.17336v1](http://arxiv.org/pdf/2508.17336v1)**

> **作者:** Yunsik Kim; Yoonyoung Chung
>
> **摘要:** Body\-conduction microphone signals (BMS) bypass airborne sound, providing strong noise resistance. However, a complementary modality is required to compensate for the inherent loss of high\-frequency information. In this study, we propose a novel multi\-modal framework that combines BMS and acoustic microphone signals (AMS) to achieve both noise suppression and high\-frequency reconstruction. Unlike conventional multi\-modal approaches that simply merge features, our method employs two specialized networks\: a mapping-based model to enhance BMS and a masking-based model to denoise AMS. These networks are integrated through a dynamic fusion mechanism that adapts to local noise conditions, ensuring the optimal use of each modality's strengths. We performed evaluations on the TAPS dataset, augmented with DNS\-2023 noise clips, using objective speech quality metrics. The results clearly demonstrate that our approach outperforms single\-modal solutions in a wide range of noisy environments.
>
---
#### [new 010] Vocoder-Projected Feature Discriminator
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; stat.ML**

- **简介: 论文提出Vocoder-Projected Feature Discriminator（VPFD），用于语音转换（VC）任务中，解决传统时域判别器训练耗时耗内存的问题。通过使用预训练声码器提取特征进行对抗训练，显著降低计算开销并保持性能。**

- **链接: [http://arxiv.org/pdf/2508.17874v1](http://arxiv.org/pdf/2508.17874v1)**

> **作者:** Takuhiro Kaneko; Hirokazu Kameoka; Kou Tanaka; Yuto Kondo
>
> **备注:** Accepted to Interspeech 2024. Project page: https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/vpfd/
>
> **摘要:** In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively.
>
---
#### [new 011] Multi-scale Scanning Network for Machine Anomalous Sound Detection
- **分类: cs.SD**

- **简介: 该论文针对机器异常声音检测任务，解决不同机器在时间与频率域中多尺度模式差异未被充分探索的问题。提出多尺度扫描网络（MSN），通过可变尺寸核盒扫描频谱图并共享权重，高效捕捉多尺度特征，在DCASE数据集上达到最先进性能。**

- **链接: [http://arxiv.org/pdf/2508.17194v1](http://arxiv.org/pdf/2508.17194v1)**

> **作者:** Yucong Zhang; Juan Liu; Ming Li
>
> **备注:** Accepted by ICONIP 2025
>
> **摘要:** Machine sounds exhibit consistent and repetitive patterns in both the frequency and time domains, which vary significantly across scales for different machine types. For instance, rotating machines often show periodic features in short time intervals, while reciprocating machines exhibit broader patterns spanning the time domain. While prior studies have leveraged these patterns to improve Anomalous Sound Detection (ASD), the variation of patterns across scales remains insufficiently explored. To address this gap, we introduce a Multi-scale Scanning Network (MSN) designed to capture patterns at multiple scales. MSN employs kernel boxes of varying sizes to scan audio spectrograms and integrates a lightweight convolutional network with shared weights for efficient and scalable feature representation. Experimental evaluations on the DCASE 2020 and DCASE 2023 Task 2 datasets demonstrate that MSN achieves state-of-the-art performance, highlighting its effectiveness in advancing ASD systems.
>
---
#### [new 012] Improving French Synthetic Speech Quality via SSML Prosody Control
- **分类: cs.CL; cs.SD; 68T50; I.2.7; H.5.5**

- **简介: 该论文属于文本到语音合成任务，旨在提升法语合成语音的自然度。通过引入SSML标记控制韵律参数，提出端到端pipeline，显著改善语音质量与听者偏好。**

- **链接: [http://arxiv.org/pdf/2508.17494v1](http://arxiv.org/pdf/2508.17494v1)**

> **作者:** Nassima Ould Ouali; Awais Hussain Sani; Ruben Bueno; Jonah Dauvet; Tim Luka Horstmann; Eric Moulines
>
> **备注:** 13 pages, 9 figures, 6 tables. Accepted for presentation at ICNLSP 2025 (Odense, Denmark). Code and demo: https://github.com/hi-paris/Prosody-Control-French-TTS. ACM Class: I.2.7; H.5.5
>
> **摘要:** Despite recent advances, synthetic voices often lack expressiveness due to limited prosody control in commercial text-to-speech (TTS) systems. We introduce the first end-to-end pipeline that inserts Speech Synthesis Markup Language (SSML) tags into French text to control pitch, speaking rate, volume, and pause duration. We employ a cascaded architecture with two QLoRA-fine-tuned Qwen 2.5-7B models: one predicts phrase-break positions and the other performs regression on prosodic targets, generating commercial TTS-compatible SSML markup. Evaluated on a 14-hour French podcast corpus, our method achieves 99.2% F1 for break placement and reduces mean absolute error on pitch, rate, and volume by 25-40% compared with prompting-only large language models (LLMs) and a BiLSTM baseline. In perceptual evaluation involving 18 participants across over 9 hours of synthesized audio, SSML-enhanced speech generated by our pipeline significantly improves naturalness, with the mean opinion score increasing from 3.20 to 3.87 (p < 0.005). Additionally, 15 of 18 listeners preferred our enhanced synthesis. These results demonstrate substantial progress in bridging the expressiveness gap between synthetic and natural French speech. Our code is publicly available at https://github.com/hi-paris/Prosody-Control-French-TTS.
>
---
#### [new 013] DanceEditor: Towards Iterative Editable Music-driven Dance Generation with Open-Vocabulary Descriptions
- **分类: cs.GR; cs.CV; cs.MM; cs.SD**

- **简介: 该论文针对音乐驱动舞蹈生成任务，解决现有方法无法支持用户迭代编辑的问题。提出DanceEditor框架与DanceRemix数据集，通过预测-编辑范式融合音乐与文本条件，实现音乐同步且语义可控的舞蹈编辑。**

- **链接: [http://arxiv.org/pdf/2508.17342v1](http://arxiv.org/pdf/2508.17342v1)**

> **作者:** Hengyuan Zhang; Zhe Li; Xingqun Qi; Mengze Li; Muyi Sun; Man Zhang; Sirui Han
>
> **摘要:** Generating coherent and diverse human dances from music signals has gained tremendous progress in animating virtual avatars. While existing methods support direct dance synthesis, they fail to recognize that enabling users to edit dance movements is far more practical in real-world choreography scenarios. Moreover, the lack of high-quality dance datasets incorporating iterative editing also limits addressing this challenge. To achieve this goal, we first construct DanceRemix, a large-scale multi-turn editable dance dataset comprising the prompt featuring over 25.3M dance frames and 84.5K pairs. In addition, we propose a novel framework for iterative and editable dance generation coherently aligned with given music signals, namely DanceEditor. Considering the dance motion should be both musical rhythmic and enable iterative editing by user descriptions, our framework is built upon a prediction-then-editing paradigm unifying multi-modal conditions. At the initial prediction stage, our framework improves the authority of generated results by directly modeling dance movements from tailored, aligned music. Moreover, at the subsequent iterative editing stages, we incorporate text descriptions as conditioning information to draw the editable results through a specifically designed Cross-modality Editing Module (CEM). Specifically, CEM adaptively integrates the initial prediction with music and text prompts as temporal motion cues to guide the synthesized sequences. Thereby, the results display music harmonics while preserving fine-grained semantic alignment with text descriptions. Extensive experiments demonstrate that our method outperforms the state-of-the-art models on our newly collected DanceRemix dataset. Code is available at https://lzvsdy.github.io/DanceEditor/.
>
---
#### [new 014] Unseen Speaker and Language Adaptation for Lightweight Text-To-Speech with Adapters
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 论文研究轻量级TTS中的跨语言与未见说话人适应问题，提出基于适配器的方法，在不遗忘原模型信息的前提下，实现目标语言中未见说话人的语音合成，并引入新指标评估语音自然度。**

- **链接: [http://arxiv.org/pdf/2508.18006v1](http://arxiv.org/pdf/2508.18006v1)**

> **作者:** Alessio Falai; Ziyao Zhang; Akos Gangoly
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** In this paper we investigate cross-lingual Text-To-Speech (TTS) synthesis through the lens of adapters, in the context of lightweight TTS systems. In particular, we compare the tasks of unseen speaker and language adaptation with the goal of synthesising a target voice in a target language, in which the target voice has no recordings therein. Results from objective evaluations demonstrate the effectiveness of adapters in learning language-specific and speaker-specific information, allowing pre-trained models to learn unseen speaker identities or languages, while avoiding catastrophic forgetting of the original model's speaker or language information. Additionally, to measure how native the generated voices are in terms of accent, we propose and validate an objective metric inspired by mispronunciation detection techniques in second-language (L2) learners. The paper also provides insights into the impact of adapter placement, configuration and the number of speakers used.
>
---
#### [new 015] Speech Discrete Tokens or Continuous Features? A Comparative Analysis for Spoken Language Understanding in SpeechLLMs
- **分类: cs.CL; cs.SD**

- **简介: 论文研究SpeechLLMs中离散token与连续特征在语音理解任务中的性能差异。通过公平对比发现，连续特征整体表现更优，且两类方法学习模式不同，为语音理解提供新见解。**

- **链接: [http://arxiv.org/pdf/2508.17863v1](http://arxiv.org/pdf/2508.17863v1)**

> **作者:** Dingdong Wang; Junan Li; Mingyu Cui; Dongchao Yang; Xueyuan Chen; Helen Meng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** With the rise of Speech Large Language Models (SpeechLLMs), two dominant approaches have emerged for speech processing: discrete tokens and continuous features. Each approach has demonstrated strong capabilities in audio-related processing tasks. However, the performance gap between these two paradigms has not been thoroughly explored. To address this gap, we present a fair comparison of self-supervised learning (SSL)-based discrete and continuous features under the same experimental settings. We evaluate their performance across six spoken language understanding-related tasks using both small and large-scale LLMs (Qwen1.5-0.5B and Llama3.1-8B). We further conduct in-depth analyses, including efficient comparison, SSL layer analysis, LLM layer analysis, and robustness comparison. Our findings reveal that continuous features generally outperform discrete tokens in various tasks. Each speech processing method exhibits distinct characteristics and patterns in how it learns and processes speech information. We hope our results will provide valuable insights to advance spoken language understanding in SpeechLLMs.
>
---
#### [new 016] Localization using Angle-of-Arrival Triangulation
- **分类: eess.AS; cs.HC; cs.NI; cs.SD; eess.SP; C.3; C.2.1; C.2.4; I.5.4; H.5.2; J.7**

- **简介: 该论文属于室内定位任务，旨在解决无需硬件改造、用户配合或先验知识的语音定位问题。提出GCC+方法，通过多设备声源到达角度估计与三角定位，实现高精度二维定位，实验显示中位误差为1.25米。**

- **链接: [http://arxiv.org/pdf/2508.16908v1](http://arxiv.org/pdf/2508.16908v1)**

> **作者:** Amod K. Agrawal
>
> **备注:** 6 pages, 5 figures, 1 table. Accepted at the ACM International Workshop on Environmental Sensing Systems for Smart Cities (EnvSys 2025). To appear in the MobiSys 2025 Proceedings
>
> **摘要:** Indoor localization is a long-standing challenge in mobile computing, with significant implications for enabling location-aware and intelligent applications within smart environments such as homes, offices, and retail spaces. As AI assistants such as Amazon Alexa and Google Nest become increasingly pervasive, microphone-equipped devices are emerging as key components of everyday life and home automation. This paper introduces a passive, infrastructure-light system for localizing human speakers using speech signals captured by two or more spatially distributed smart devices. The proposed approach, GCC+, extends the Generalized Cross-Correlation with Phase Transform (GCC-PHAT) method to estimate the Angle-of-Arrival (AoA) of audio signals at each device and applies robust triangulation techniques to infer the speaker's two-dimensional position. To further improve temporal resolution and localization accuracy, feature-space expansion and subsample interpolation techniques are employed for precise Time Difference of Arrival (TDoA) estimation. The system operates without requiring hardware modifications, prior calibration, explicit user cooperation, or knowledge of the speaker's signal content, thereby offering a highly practical solution for real-world deployment. Experimental evaluation in a real-world home environment yields a median AoA estimation error of 2.2 degrees and a median localization error of 1.25 m, demonstrating the feasibility and effectiveness of audio-based localization for enabling context-aware, privacy-preserving ambient intelligence.
>
---
#### [new 017] VGGSounder: Audio-Visual Evaluations for Foundation Models
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文针对音频-视觉基础模型的评估问题，指出VGGSound数据集存在标注不全、类别重叠和模态错位等问题。为此提出VGGSounder，一个重新标注的多标签测试集，支持细粒度模态性能分析，并引入模态混淆度量揭示模型在添加新模态时的性能下降。**

- **链接: [http://arxiv.org/pdf/2508.08237v2](http://arxiv.org/pdf/2508.08237v2)**

> **作者:** Daniil Zverev; Thaddäus Wiedemer; Ameya Prabhu; Matthias Bethge; Wieland Brendel; A. Sophia Koepke
>
> **备注:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The emergence of audio-visual foundation models underscores the importance of reliably assessing their multi-modal understanding. The VGGSound dataset is commonly used as a benchmark for evaluation audio-visual classification. However, our analysis identifies several limitations of VGGSound, including incomplete labelling, partially overlapping classes, and misaligned modalities. These lead to distorted evaluations of auditory and visual capabilities. To address these limitations, we introduce VGGSounder, a comprehensively re-annotated, multi-label test set that extends VGGSound and is specifically designed to evaluate audio-visual foundation models. VGGSounder features detailed modality annotations, enabling precise analyses of modality-specific performance. Furthermore, we reveal model limitations by analysing performance degradation when adding another input modality with our new modality confusion metric.
>
---
#### [new 018] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation
- **分类: cs.GR; cs.CV; cs.MM; cs.SD**

- **简介: 论文提出MDD数据集，用于文本和音乐驱动的双人舞蹈动作生成任务，解决多模态舞蹈生成中缺乏高质量标注数据的问题。工作包括构建620分钟高质量动作捕捉数据及10K自然语言描述，并定义两个新任务：Text-to-Duet与Text-to-Dance Accompaniment。**

- **链接: [http://arxiv.org/pdf/2508.16911v1](http://arxiv.org/pdf/2508.16911v1)**

> **作者:** Prerit Gupta; Jason Alexander Fotso-Puepi; Zhengyuan Li; Jay Mehta; Aniket Bera
>
> **备注:** Accepted at ICCV 2025. Project page: https://gprerit96.github.io/mdd-page
>
> **摘要:** We introduce Multimodal DuetDance (MDD), a diverse multimodal benchmark dataset designed for text-controlled and music-conditioned 3D duet dance motion generation. Our dataset comprises 620 minutes of high-quality motion capture data performed by professional dancers, synchronized with music, and detailed with over 10K fine-grained natural language descriptions. The annotations capture a rich movement vocabulary, detailing spatial relationships, body movements, and rhythm, making MDD the first dataset to seamlessly integrate human motions, music, and text for duet dance generation. We introduce two novel tasks supported by our dataset: (1) Text-to-Duet, where given music and a textual prompt, both the leader and follower dance motion are generated (2) Text-to-Dance Accompaniment, where given music, textual prompt, and the leader's motion, the follower's motion is generated in a cohesive, text-aligned manner. We include baseline evaluations on both tasks to support future research.
>
---
#### [new 019] Geolocation-Aware Robust Spoken Language Identification
- **分类: cs.CL; cs.SD**

- **简介: 论文提出地理信息感知的语音语言识别方法，通过引入地理预测辅助任务和条件信号，增强模型对同一语言内方言和口音变化的鲁棒性，提升跨域泛化能力，在多个数据集上取得新最优结果。**

- **链接: [http://arxiv.org/pdf/2508.17148v1](http://arxiv.org/pdf/2508.17148v1)**

> **作者:** Qingzheng Wang; Hye-jin Shim; Jiancheng Sun; Shinji Watanabe
>
> **备注:** Accepted to IEEE ASRU 2025. \c{opyright} 2025 IEEE. Personal use permitted. Permission from IEEE required for all other uses including reprinting/republishing, advertising, resale, redistribution, reuse, or creating collective works
>
> **摘要:** While Self-supervised Learning (SSL) has significantly improved Spoken Language Identification (LID), existing models often struggle to consistently classify dialects and accents of the same language as a unified class. To address this challenge, we propose geolocation-aware LID, a novel approach that incorporates language-level geolocation information into the SSL-based LID model. Specifically, we introduce geolocation prediction as an auxiliary task and inject the predicted vectors into intermediate representations as conditioning signals. This explicit conditioning encourages the model to learn more unified representations for dialectal and accented variations. Experiments across six multilingual datasets demonstrate that our approach improves robustness to intra-language variations and unseen domains, achieving new state-of-the-art accuracy on FLEURS (97.7%) and 9.7% relative improvement on ML-SUPERB 2.0 dialect set.
>
---
#### [new 020] SyncGuard: Robust Audio Watermarking Capable of Countering Desynchronization Attacks
- **分类: cs.CR; cs.MM; cs.SD**

- **简介: 该论文属于音频水印任务，旨在解决音频水印在时间同步被破坏时的定位难和鲁棒性差问题。提出SyncGuard方案，通过帧级广播嵌入策略和畸变层增强鲁棒性，并利用多尺度时频特征提取提升性能。**

- **链接: [http://arxiv.org/pdf/2508.17121v1](http://arxiv.org/pdf/2508.17121v1)**

> **作者:** Zhenliang Gan; Xiaoxiao Hu; Sheng Li; Zhenxing Qian; Xinpeng Zhang
>
> **摘要:** Audio watermarking has been widely applied in copyright protection and source tracing. However, due to the inherent characteristics of audio signals, watermark localization and resistance to desynchronization attacks remain significant challenges. In this paper, we propose a learning-based scheme named SyncGuard to address these challenges. Specifically, we design a frame-wise broadcast embedding strategy to embed the watermark in arbitrary-length audio, enhancing time-independence and eliminating the need for localization during watermark extraction. To further enhance robustness, we introduce a meticulously designed distortion layer. Additionally, we employ dilated residual blocks in conjunction with dilated gated blocks to effectively capture multi-resolution time-frequency features. Extensive experimental results show that SyncGuard efficiently handles variable-length audio segments, outperforms state-of-the-art methods in robustness against various attacks, and delivers superior auditory quality.
>
---
#### [new 021] Relative Navigation and Dynamic Target Tracking for Autonomous Underwater Proximity Operations
- **分类: cs.RO; cs.SY; eess.SP; eess.SY; I.2.9; I.2.8; F.2.2**

- **简介: 论文解决水下近距离自主操作中目标6-DoF运动估计难题，提出基于李群切空间的广义恒定扭转变量先验，通过二元因子和闭式雅可比矩阵实现跨表示的一致轨迹估计，提升USBL仅测量下的跟踪精度。**

- **链接: [http://arxiv.org/pdf/2508.16901v1](http://arxiv.org/pdf/2508.16901v1)**

> **作者:** David Baxter; Aldo Terán Espinoza; Antonio Terán Espinoza; Amy Loutfi; John Folkesson; Peter Sigray; Stephanie Lowry; Jakob Kuttenkeuler
>
> **备注:** 10 pages, 7 figures. Equal contribution by David Baxter and Aldo Ter\'an Espinoza. Supported by SAAB, SMaRC, and WASP. Supported by SAAB and the Swedish Maritime Robotics Centre (SMaRC), and by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation
>
> **摘要:** Estimating a target's 6-DoF motion in underwater proximity operations is difficult because the chaser lacks target-side proprioception and the available relative observations are sparse, noisy, and often partial (e.g., Ultra-Short Baseline (USBL) positions). Without a motion prior, factor-graph maximum a posteriori estimation is underconstrained: consecutive target states are weakly linked and orientation can drift. We propose a generalized constant-twist motion prior defined on the tangent space of Lie groups that enforces temporally consistent trajectories across all degrees of freedom; in SE(3) it couples translation and rotation in the body frame. We present a ternary factor and derive its closed-form Jacobians based on standard Lie group operations, enabling drop-in use for trajectories on arbitrary Lie groups. We evaluate two deployment modes: (A) an SE(3)-only representation that regularizes orientation even when only position is measured, and (B) a mode with boundary factors that switches the target representation between SE(3) and 3D position while applying the same generalized constant-twist prior across representation changes. Validation on a real-world dynamic docking scenario dataset shows consistent ego-target trajectory estimation through USBL-only and optical relative measurement segments with an improved relative tracking accuracy compared to the noisy measurements to the target. Because the construction relies on standard Lie group primitives, it is portable across state manifolds and sensing modalities.
>
---
#### [new 022] HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文提出HunyuanVideo-Foley，解决视频生成中缺乏同步音频的问题。通过多模态数据管道、表示对齐策略和融合双流注意力机制，实现高保真、时序对齐的音频生成，提升沉浸感。**

- **链接: [http://arxiv.org/pdf/2508.16930v1](http://arxiv.org/pdf/2508.16930v1)**

> **作者:** Sizhe Shan; Qiulin Li; Yutao Cui; Miles Yang; Yuehai Wang; Qun Yang; Jin Zhou; Zhao Zhong
>
> **摘要:** Recent advances in video generation produce visually realistic content, yet the absence of synchronized audio severely compromises immersion. To address key challenges in video-to-audio generation, including multimodal data scarcity, modality imbalance and limited audio quality in existing methods, we propose HunyuanVideo-Foley, an end-to-end text-video-to-audio framework that synthesizes high-fidelity audio precisely aligned with visual dynamics and semantic context. Our approach incorporates three core innovations: (1) a scalable data pipeline curating 100k-hour multimodal datasets through automated annotation; (2) a representation alignment strategy using self-supervised audio features to guide latent diffusion training, efficiently improving audio quality and generation stability; (3) a novel multimodal diffusion transformer resolving modal competition, containing dual-stream audio-video fusion through joint attention, and textual semantic injection via cross-attention. Comprehensive evaluations demonstrate that HunyuanVideo-Foley achieves new state-of-the-art performance across audio fidelity, visual-semantic alignment, temporal alignment and distribution matching. The demo page is available at: https://szczesnys.github.io/hunyuanvideo-foley/.
>
---
#### [new 023] ERF-BA-TFD+: A Multimodal Model for Audio-Visual Deepfake Detection
- **分类: cs.AI; cs.SD**

- **简介: 该论文提出ERF-BA-TFD+模型，用于音频-视觉深度伪造检测任务。针对多模态深度伪造难以识别的问题，模型融合音频与视频特征，增强长距离依赖建模能力，在DDL-AV数据集上实现高精度与快速检测，获竞赛第一名。**

- **链接: [http://arxiv.org/pdf/2508.17282v1](http://arxiv.org/pdf/2508.17282v1)**

> **作者:** Xin Zhang; Jiaming Chu; Jian Zhao; Yuchu Jiang; Xu Yang; Lei Jin; Chi Zhang; Xuelong Li
>
> **摘要:** Deepfake detection is a critical task in identifying manipulated multimedia content. In real-world scenarios, deepfake content can manifest across multiple modalities, including audio and video. To address this challenge, we present ERF-BA-TFD+, a novel multimodal deepfake detection model that combines enhanced receptive field (ERF) and audio-visual fusion. Our model processes both audio and video features simultaneously, leveraging their complementary information to improve detection accuracy and robustness. The key innovation of ERF-BA-TFD+ lies in its ability to model long-range dependencies within the audio-visual input, allowing it to better capture subtle discrepancies between real and fake content. In our experiments, we evaluate ERF-BA-TFD+ on the DDL-AV dataset, which consists of both segmented and full-length video clips. Unlike previous benchmarks, which focused primarily on isolated segments, the DDL-AV dataset allows us to assess the model's performance in a more comprehensive and realistic setting. Our method achieves state-of-the-art results on this dataset, outperforming existing techniques in terms of both accuracy and processing speed. The ERF-BA-TFD+ model demonstrated its effectiveness in the "Workshop on Deepfake Detection, Localization, and Interpretability," Track 2: Audio-Visual Detection and Localization (DDL-AV), and won first place in this competition.
>
---
#### [new 024] Objective and Subjective Evaluation of Diffusion-Based Speech Enhancement for Dysarthric Speech
- **分类: eess.AS; cs.SD**

- **简介: 论文研究扩散模型在构音障碍语音增强中的应用，旨在提升ASR系统对这类语音的识别性能。通过对比三种增强算法，评估其对语音可懂度和质量的影响，并微调Whisper-Turbo模型以验证效果。**

- **链接: [http://arxiv.org/pdf/2508.17980v1](http://arxiv.org/pdf/2508.17980v1)**

> **作者:** Dimme de Groot; Tanvina Patel; Devendra Kayande; Odette Scharenborg; Zhengjun Yue
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Dysarthric speech poses significant challenges for automatic speech recognition (ASR) systems due to its high variability and reduced intelligibility. In this work we explore the use of diffusion models for dysarthric speech enhancement, which is based on the hypothesis that using diffusion-based speech enhancement moves the distribution of dysarthric speech closer to that of typical speech, which could potentially improve dysarthric speech recognition performance. We assess the effect of two diffusion-based and one signal-processing-based speech enhancement algorithms on intelligibility and speech quality of two English dysarthric speech corpora. We applied speech enhancement to both typical and dysarthric speech and evaluate the ASR performance using Whisper-Turbo, and the subjective and objective speech quality of the original and enhanced dysarthric speech. We also fine-tuned Whisper-Turbo on the enhanced speech to assess its impact on recognition performance.
>
---
## 更新

#### [replaced 001] CAARMA: Class Augmentation with Adversarial Mixup Regularization
- **分类: cs.SD; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16718v2](http://arxiv.org/pdf/2503.16718v2)**

> **作者:** Massa Baali; Xiang Li; Hao Chen; Syed Abdul Hannan; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. The code is available at: https://github.com/massabaali7/CAARMA/
>
---
#### [replaced 002] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.13237v3](http://arxiv.org/pdf/2505.13237v3)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025 (Oral). Update acknowledgement in this version. Project page: https://github.com/ckyang1124/SAKURA
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [replaced 003] Towards Controllable Speech Synthesis in the Era of Large Language Models: A Systematic Survey
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.06602v3](http://arxiv.org/pdf/2412.06602v3)**

> **作者:** Tianxin Xie; Yan Rong; Pengfei Zhang; Wenwu Wang; Li Liu
>
> **备注:** The first comprehensive survey on controllable TTS. Accepted to the EMNLP 2025 main conference
>
> **摘要:** Text-to-speech (TTS) has advanced from generating natural-sounding speech to enabling fine-grained control over attributes like emotion, timbre, and style. Driven by rising industrial demand and breakthroughs in deep learning, e.g., diffusion and large language models (LLMs), controllable TTS has become a rapidly growing research area. This survey provides the first comprehensive review of controllable TTS methods, from traditional control techniques to emerging approaches using natural language prompts. We categorize model architectures, control strategies, and feature representations, while also summarizing challenges, datasets, and evaluations in controllable TTS. This survey aims to guide researchers and practitioners by offering a clear taxonomy and highlighting future directions in this fast-evolving field. One can visit https://github.com/imxtx/awesome-controllabe-speech-synthesis for a comprehensive paper list and updates.
>
---
#### [replaced 004] ASCMamba: Multimodal Time-Frequency Mamba for Acoustic Scene Classification
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2508.15632v2](http://arxiv.org/pdf/2508.15632v2)**

> **作者:** Bochao Sun; Dong Wang; ZhanLong Yang; Jun Yang; Han Yin
>
> **摘要:** Acoustic Scene Classification (ASC) is a fundamental problem in computational audition, which seeks to classify environments based on the distinctive acoustic features. In the ASC task of the APSIPA ASC 2025 Grand Challenge, the organizers introduce a multimodal ASC task. Unlike traditional ASC systems that rely solely on audio inputs, this challenge provides additional textual information as inputs, including the location where the audio is recorded and the time of recording. In this paper, we present our proposed system for the ASC task in the APSIPA ASC 2025 Grand Challenge. Specifically, we propose a multimodal network, ASCMamba, which integrates audio and textual information for fine-grained acoustic scene understanding and effective multimodal ASC. The proposed ASCMamba employs a DenseEncoder to extract hierarchical spectral features from spectrograms, followed by a dual-path Mamba blocks that capture long-range temporal and frequency dependencies using Mamba-based state space models. In addition, we present a two-step pseudo-labeling mechanism to generate more reliable pseudo-labels. Results show that the proposed system outperforms all the participating teams and achieves a 6.2% improvement over the baseline. Code, model and pre-trained checkpoints are available at https://github.com/S-Orion/ASCMamba.git.
>
---
#### [replaced 005] Automatic Speech Recognition of African American English: Lexical and Contextual Effects
- **分类: cs.CL; cs.SD; eess.AS; I.5; G.3**

- **链接: [http://arxiv.org/pdf/2506.06888v2](http://arxiv.org/pdf/2506.06888v2)**

> **作者:** Hamid Mojarad; Kevin Tang
>
> **备注:** submitted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
>
---
#### [replaced 006] LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.20583v2](http://arxiv.org/pdf/2502.20583v2)**

> **作者:** Keisuke Kamahori; Jungo Kasai; Noriyuki Kojima; Baris Kasikci
>
> **备注:** EMNLP2025 Main
>
> **摘要:** Modern automatic speech recognition (ASR) models, such as OpenAI's Whisper, rely on deep encoder-decoder architectures, and their encoders are a critical bottleneck for efficient deployment due to high computational intensity. We introduce LiteASR, a low-rank compression scheme for ASR encoders that significantly reduces inference costs while maintaining transcription accuracy. Our approach leverages the strong low-rank properties observed in intermediate activations: by applying principal component analysis (PCA) with a small calibration dataset, we approximate linear transformations with a chain of low-rank matrix multiplications, and further optimize self-attention to work in reduced dimensionality. Evaluation results show that our method can compress Whisper large-v3's encoder size by over 50%, matching Whisper medium's size with better transcription accuracy, thereby establishing a new Pareto frontier of accuracy and efficiency. The code of LiteASR is available at https://github.com/efeslab/LiteASR.
>
---
#### [replaced 007] Whilter: A Whisper-based Data Filter for "In-the-Wild" Speech Corpora Using Utterance-level Multi-Task Classification
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.21642v2](http://arxiv.org/pdf/2507.21642v2)**

> **作者:** William Ravenscroft; George Close; Kit Bower-Morris; Jamie Stacey; Dmitry Sityaev; Kris Y. Hong
>
> **备注:** Accepted for Interspeech 2025. Updated Zenodo link for AITW v1.1
>
> **摘要:** Large-scale in-the-wild speech datasets have become more prevalent in recent years due to increased interest in models that can learn useful features from unlabelled data for tasks such as speech recognition or synthesis. These datasets often contain undesirable features, such as multiple speakers, non-target languages, and music, which may impact model learning. The Whilter model is proposed as a multitask solution to identify these undesirable samples. Whilter uses a Whisper encoder with an attention-based classifier to solve five diverse classification problems at once. In addition, an annotated dataset is published for a subset of two popular in-the-wild corpora. Whilter achieves F1 scores above 85% and equal error rates of 6.5% to 7.8% for three of five subtasks, outperforming a state-of-the-art BEATs classifier on speech-specific classes, with a notable decrease in processing time compared to a combination of single-task alternatives.
>
---
#### [replaced 008] LABNet: A Lightweight Attentive Beamforming Network for Ad-hoc Multichannel Microphone Invariant Real-Time Speech Enhancement
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16190v2](http://arxiv.org/pdf/2507.16190v2)**

> **作者:** Haoyin Yan; Jie Zhang; Chengqian Jiang; Shuang Zhang
>
> **摘要:** Multichannel speech enhancement (SE) aims to restore clean speech from noisy measurements by leveraging spatiotemporal signal features. In ad-hoc array conditions, microphone invariance (MI) requires systems to handle different microphone numbers and array geometries. From a practical perspective, multichannel recordings inevitably increase the computational burden for edge-device applications, highlighting the necessity of lightweight and efficient deployments. In this work, we propose a lightweight attentive beamforming network (LABNet) to integrate MI in a low-complexity real-time SE system. We design a three-stage framework for efficient intra-channel modeling and inter-channel interaction. A cross-channel attention module is developed to aggregate features from each channel selectively. Experimental results demonstrate our LABNet achieves impressive performance with ultra-light resource overhead while maintaining the MI, indicating great potential for ad-hoc array processing.
>
---
#### [replaced 009] Playing with Voices: Tabletop Role-Playing Game Recordings as a Diarization Challenge
- **分类: cs.CL; cs.SD; I.5**

- **链接: [http://arxiv.org/pdf/2502.12714v2](http://arxiv.org/pdf/2502.12714v2)**

> **作者:** Lian Remme; Kevin Tang
>
> **备注:** 15 pages, 14 figures, published in NAACL Findings 2025
>
> **摘要:** This paper provides a proof of concept that audio of tabletop role-playing games (TTRPG) could serve as a challenge for diarization systems. TTRPGs are carried out mostly by conversation. Participants often alter their voices to indicate that they are talking as a fictional character. Audio processing systems are susceptible to voice conversion with or without technological assistance. TTRPG present a conversational phenomenon in which voice conversion is an inherent characteristic for an immersive gaming experience. This could make it more challenging for diarizers to pick the real speaker and determine that impersonating is just that. We present the creation of a small TTRPG audio dataset and compare it against the AMI and the ICSI corpus. The performance of two diarizers, pyannote.audio and wespeaker, were evaluated. We observed that TTRPGs' properties result in a higher confusion rate for both diarizers. Additionally, wespeaker strongly underestimates the number of speakers in the TTRPG audio files. We propose TTRPG audio as a promising challenge for diarization systems.
>
---
#### [replaced 010] Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.09226v2](http://arxiv.org/pdf/2507.09226v2)**

> **作者:** Shota Horiguchi; Naohiro Tawara; Takanori Ashihara; Atsushi Ando; Marc Delcroix
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Neural speaker diarization is widely used for overlap-aware speaker diarization, but it requires large multi-speaker datasets for training. To meet this data requirement, large datasets are often constructed by combining multiple corpora, including those originally designed for multi-speaker automatic speech recognition (ASR). However, ASR datasets often feature loosely defined segment boundaries that do not align with the stricter conventions of diarization benchmarks. In this work, we show that such boundary looseness significantly impacts the diarization error rate, reducing evaluation reliability. We also reveal that models trained on data with varying boundary precision tend to learn dataset-specific looseness, leading to poor generalization across out-of-domain datasets. Training with standardized tight boundaries via forced alignment improves not only diarization performance, especially in streaming scenarios, but also ASR performance when combined with simple post-processing.
>
---
#### [replaced 011] Missing Melodies: AI Music Generation and its "Nearly" Complete Omission of the Global South
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.04100v3](http://arxiv.org/pdf/2412.04100v3)**

> **作者:** Atharva Mehta; Shivam Chauhan; Monojit Choudhury
>
> **备注:** Submitted to CACM, 12 pages, 2 figures
>
> **摘要:** Recent advances in generative AI have sparked renewed interest and expanded possibilities for music generation. However, the performance and versatility of these systems across musical genres are heavily influenced by the availability of training data. We conducted an extensive analysis of over one million hours of audio datasets used in AI music generation research and manually reviewed more than 200 papers from eleven prominent AI and music conferences and organizations (AAAI, ACM, EUSIPCO, EURASIP, ICASSP, ICML, IJCAI, ISMIR, NeurIPS, NIME, SMC) to identify a critical gap in the fair representation and inclusion of the musical genres of the Global South in AI research. Our findings reveal a stark imbalance: approximately 86% of the total dataset hours and over 93% of researchers focus primarily on music from the Global North. However, around 40% of these datasets include some form of non-Western music, genres from the Global South account for only 14.6% of the data. Furthermore, approximately 51% of the papers surveyed concentrate on symbolic music generation, a method that often fails to capture the cultural nuances inherent in music from regions such as South Asia, the Middle East, and Africa. As AI increasingly shapes the creation and dissemination of music, the significant underrepresentation of music genres in datasets and research presents a serious threat to global musical diversity. We also propose some important steps to mitigate these risks and foster a more inclusive future for AI-driven music generation.
>
---
#### [replaced 012] Soundscape Captioning using Sound Affective Quality Network and Large Language Model
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2406.05914v3](http://arxiv.org/pdf/2406.05914v3)**

> **作者:** Yuanbo Hou; Qiaoqiao Ren; Andrew Mitchell; Wenwu Wang; Jian Kang; Tony Belpaeme; Dick Botteldooren
>
> **备注:** IEEE Transactions on Multimedia, Code: https://github.com/Yuanbo2020/SoundSCaper
>
> **摘要:** We live in a rich and varied acoustic world, which is experienced by individuals or communities as a soundscape. Computational auditory scene analysis, disentangling acoustic scenes by detecting and classifying events, focuses on objective attributes of sounds, such as their category and temporal characteristics, ignoring their effects on people, such as the emotions they evoke within a context. To fill this gap, we propose the affective soundscape captioning (ASSC) task, which enables automated soundscape analysis, thus avoiding labour-intensive subjective ratings and surveys in conventional methods. With soundscape captioning, context-aware descriptions are generated for soundscape by capturing the acoustic scenes (ASs), audio events (AEs) information, and the corresponding human affective qualities (AQs). To this end, we propose an automatic soundscape captioner (SoundSCaper) system composed of an acoustic model, i.e. SoundAQnet, and a large language model (LLM). SoundAQnet simultaneously models multi-scale information about ASs, AEs, and perceived AQs, while the LLM describes the soundscape with captions by parsing the information captured with SoundAQnet. SoundSCaper is assessed by two juries of 32 people. In expert evaluation, the average score of SoundSCaper-generated captions is slightly lower than that of two soundscape experts on the evaluation set D1 and the external mixed dataset D2, but not statistically significant. In layperson evaluation, SoundSCaper outperforms soundscape experts in several metrics. In addition to human evaluation, compared to other automated audio captioning systems with and without LLM, SoundSCaper performs better on the ASSC task in several NLP-based metrics. Overall, SoundSCaper performs well in human subjective evaluation and various objective captioning metrics, and the generated captions are comparable to those annotated by soundscape experts.
>
---
#### [replaced 013] Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia: A Proof-of-Concept Study
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.17054v2](http://arxiv.org/pdf/2409.17054v2)**

> **作者:** Nur Ahmad Khatim; Azmul Asmar Irfan; Mansur M. Arief
>
> **摘要:** One of the critical issues contributing to inefficiency in Puskesmas (Indonesian community health centers) is the time-consuming nature of documenting doctor-patient interactions. Doctors must conduct thorough consultations and manually transcribe detailed notes into ePuskesmas electronic health records (EHR), which creates substantial administrative burden to already overcapacitated physicians. This paper presents a proof-of-concept framework using large language models (LLMs) to automate real-time transcription and summarization of doctor-patient conversations in Bahasa Indonesia. Our system combines Whisper model for transcription with GPT-3.5 for medical summarization, implemented as a browser extension that automatically populates ePuskesmas forms. Through controlled roleplay experiments with medical validation, we demonstrate the technical feasibility of processing detailed 300+ seconds trimmed consultations in under 30 seconds while maintaining clinical accuracy. This work establishes the foundation for AI-assisted clinical documentation in resource-constrained healthcare environments. However, concerns have also been raised regarding privacy compliance and large-scale clinical evaluation addressing language and cultural biases for LLMs.
>
---
#### [replaced 014] CoLMbo: Speaker Language Model for Descriptive Profiling
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.09375v2](http://arxiv.org/pdf/2506.09375v2)**

> **作者:** Massa Baali; Shuo Han; Syed Abdul Hannan; Purusottam Samal; Karanveer Singh; Soham Deshmukh; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner. This paper introduces CoLMbo, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to adapt dynamically to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach not only enhances traditional speaker profiling but also excels in zero-shot scenarios across diverse datasets, marking a significant advancement in the field of speaker recognition.
>
---
#### [replaced 015] AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.05140v2](http://arxiv.org/pdf/2506.05140v2)**

> **作者:** Chih-Kai Yang; Neo Ho; Yi-Jyun Lee; Hung-yi Lee
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements.
>
---
#### [replaced 016] Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.15442v2](http://arxiv.org/pdf/2508.15442v2)**

> **作者:** Chenlin Liu; Minghui Fang; Patrick Zhang; Wei Zhou; Jie Gao; Jiqing Han
>
> **摘要:** Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness.
>
---
#### [replaced 017] Versatile Framework for Song Generation with Prompt-based Control
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.19062v5](http://arxiv.org/pdf/2504.19062v5)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Ruiqi Li; Jingyu Lu; Rongjie Huang; Ruiyuan Zhang; Zhiqing Hong; Ziyue Jiang; Zhou Zhao
>
> **备注:** Accepted by Findings of EMNLP 2025
>
> **摘要:** Song generation focuses on producing controllable high-quality songs based on various prompts. However, existing methods struggle to generate vocals and accompaniments with prompt-based control and proper alignment. Additionally, they fall short in supporting various tasks. To address these challenges, we introduce VersBand, a multi-task song generation framework for synthesizing high-quality, aligned songs with prompt-based control. VersBand comprises these primary models: 1) VocalBand, a decoupled model, leverages the flow-matching method for generating singing styles, pitches, and mel-spectrograms, allowing fast, high-quality vocal generation with style control. 2) AccompBand, a flow-based transformer model, incorporates the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control. This model allows for generating controllable, high-quality accompaniments aligned with vocals. 3) Two generation models, LyricBand for lyrics and MelodyBand for melodies, contribute to the comprehensive multi-task song generation system, allowing for extensive control based on multiple prompts. Experimental results show that VersBand outperforms baseline models across multiple song generation tasks using objective and subjective metrics. Demos and codes are available at https://aaronz345.github.io/VersBandDemo and https://github.com/AaronZ345/VersBand.
>
---
#### [replaced 018] FlowDubber: Movie Dubbing with LLM-based Semantic-aware Learning and Flow Matching based Voice Enhancing
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.01263v2](http://arxiv.org/pdf/2505.01263v2)**

> **作者:** Gaoxiang Cong; Liang Li; Jiadong Pan; Zhedong Zhang; Amin Beheshti; Anton van den Hengel; Yuankai Qi; Qingming Huang
>
> **摘要:** Movie Dubbing aims to convert scripts into speeches that align with the given movie clip in both temporal and emotional aspects while preserving the vocal timbre of a given brief reference audio. Existing methods focus primarily on reducing the word error rate while ignoring the importance of lip-sync and acoustic quality. To address these issues, we propose a large language model (LLM) based flow matching architecture for dubbing, named FlowDubber, which achieves high-quality audio-visual sync and pronunciation by incorporating a large speech language model and dual contrastive aligning while achieving better acoustic quality via the proposed voice-enhanced flow matching than previous works. First, we introduce Qwen2.5 as the backbone of LLM to learn the in-context sequence from movie scripts and reference audio. Then, the proposed semantic-aware learning focuses on capturing LLM semantic knowledge at the phoneme level. Next, dual contrastive aligning (DCA) boosts mutual alignment with lip movement, reducing ambiguities where similar phonemes might be confused. Finally, the proposed Flow-based Voice Enhancing (FVE) improves acoustic quality in two aspects, which introduces an LLM-based acoustics flow matching guidance to strengthen clarity and uses affine style prior to enhance identity when recovering noise into mel-spectrograms via gradient vector field prediction. Extensive experiments demonstrate that our method outperforms several state-of-the-art methods on two primary benchmarks.
>
---
