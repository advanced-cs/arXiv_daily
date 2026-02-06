# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] HyperPotter: Spell the Charm of High-Order Interactions in Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有方法忽略高阶交互的问题。提出HyperPotter框架，通过超图建模高阶交互，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.05670v1](https://arxiv.org/pdf/2602.05670v1)**

> **作者:** Qing Wen; Haohao Li; Zhongjie Ba; Peng Cheng; Miao He; Li Lu; Kui Ren
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Advances in AIGC technologies have enabled the synthesis of highly realistic audio deepfakes capable of deceiving human auditory perception. Although numerous audio deepfake detection (ADD) methods have been developed, most rely on local temporal/spectral features or pairwise relations, overlooking high-order interactions (HOIs). HOIs capture discriminative patterns that emerge from multiple feature components beyond their individual contributions. We propose HyperPotter, a hypergraph-based framework that explicitly models these synergistic HOIs through clustering-based hyperedges with class-aware prototype initialization. Extensive experiments demonstrate that HyperPotter surpasses its baseline by an average relative gain of 22.15% across 11 datasets and outperforms state-of-the-art methods by 13.96% on 4 challenging cross-domain datasets, demonstrating superior generalization to diverse attacks and speakers.
>
---
#### [new 002] Enabling Automatic Disordered Speech Recognition: An Impaired Speech Dataset in the Akan Language
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言中受损语音数据不足的问题。研究构建了一个包含50小时阿坎语受损语音的数据集，涵盖四种语音障碍类型，以支持相关技术研究。**

- **链接: [https://arxiv.org/pdf/2602.05406v1](https://arxiv.org/pdf/2602.05406v1)**

> **作者:** Isaac Wiafe; Akon Obu Ekpezu; Sumaya Ahmed Salihs; Elikem Doe Atsakpo; Fiifi Baffoe Payin Winful; Jamal-Deen Abdulai
>
> **摘要:** The lack of impaired speech data hinders advancements in the development of inclusive speech technologies, particularly in low-resource languages such as Akan. To address this gap, this study presents a curated corpus of speech samples from native Akan speakers with speech impairment. The dataset comprises of 50.01 hours of audio recordings cutting across four classes of impaired speech namely stammering, cerebral palsy, cleft palate, and stroke induced speech disorder. Recordings were done in controlled supervised environments were participants described pre-selected images in their own words. The resulting dataset is a collection of audio recordings, transcriptions, and associated metadata on speaker demographics, class of impairment, recording environment and device. The dataset is intended to support research in low-resource automatic disordered speech recognition systems and assistive speech technology.
>
---
#### [new 003] Speech-XL: Towards Long-Form Speech Understanding in Large Speech Language Models
- **分类: cs.SD**

- **简介: 该论文属于语音理解任务，旨在解决长音频处理中上下文受限和内存消耗大的问题。通过引入SST模块实现语音信息高效压缩，提升长序列建模能力。**

- **链接: [https://arxiv.org/pdf/2602.05373v1](https://arxiv.org/pdf/2602.05373v1)**

> **作者:** Haoqin Sun; Chenyang Lyu; Shiwan Zhao; Xuanfan Ni; Xiangyu Kong; Longyue Wang; Weihua Luo; Yong Qin
>
> **摘要:** Despite the growing success of Large Speech Language Models (LSLMs) in processing short-term acoustic signals, their extension to long-form audio understanding is severely bottlenecked. This limitation stems from the limited context length and the exorbitant memory footprints required for long-form inference. In this work, we propose Speech-XL, a new model that capitalizes on the intrinsic key-value (KV) sparsification capacity of Large Language Models (LLMs) to achieve high-ratio speech input compression. Specifically, we introduce a novel special token, the Speech Summarization Token (SST), for each speech interval to encapsulate the intra-interval speech information into its associated KV pairs. The SST module is trained via instruction fine-tuning, employing a curriculum learning strategy where the SST learns to compress information in a progressive manner--advancing from low-ratio (simple) to high-ratio (challenging) compression. Despite utilizing significantly less training data than other baselines, our model achieves highly competitive performance on major benchmarks, including LongSpeech and AUDIOMARATHON. By addressing the long-standing bottlenecks in long-form audio modeling, our approach offers a novel perspective on the condensation of extensive acoustic sequences.
>
---
#### [new 004] ARCHI-TTS: A flow-matching-based Text-to-Speech Model with Self-supervised Semantic Aligner and Accelerated Inference
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决文本-语音对齐困难和计算成本高的问题。提出ARCHI-TTS模型，引入语义对齐器和高效推理策略，提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2602.05207v1](https://arxiv.org/pdf/2602.05207v1)**

> **作者:** Chunyat Wu; Jiajun Deng; Zhengxi Liu; Zheqi Dai; Haolin He; Qiuqiang Kong
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Although diffusion-based, non-autoregressive text-to-speech (TTS) systems have demonstrated impressive zero-shot synthesis capabilities, their efficacy is still hindered by two key challenges: the difficulty of text-speech alignment modeling and the high computational overhead of the iterative denoising process. To address these limitations, we propose ARCHI-TTS that features a dedicated semantic aligner to ensure robust temporal and semantic consistency between text and audio. To overcome high computational inference costs, ARCHI-TTS employs an efficient inference strategy that reuses encoder features across denoising steps, drastically accelerating synthesis without performance degradation. An auxiliary CTC loss applied to the condition encoder further enhances the semantic understanding. Experimental results demonstrate that ARCHI-TTS achieves a WER of 1.98% on LibriSpeech-PC test-clean, and 1.47%/1.42% on SeedTTS test-en/test-zh with a high inference efficiency, consistently outperforming recent state-of-the-art TTS systems.
>
---
#### [new 005] AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频模型解释任务，旨在提升音频处理模型的可解释性。通过训练稀疏自编码器（SAEs），分析并提取音频特征，增强模型理解与应用效果。**

- **链接: [https://arxiv.org/pdf/2602.05027v1](https://arxiv.org/pdf/2602.05027v1)**

> **作者:** Georgii Aparin; Tasnima Sadekova; Alexey Rukhovich; Assel Yermekova; Laida Kushnareva; Vadim Popov; Kristian Kuznetsov; Irina Piontkovskaya
>
> **摘要:** Sparse Autoencoders (SAEs) are powerful tools for interpreting neural representations, yet their use in audio remains underexplored. We train SAEs across all encoder layers of Whisper and HuBERT, provide an extensive evaluation of their stability, interpretability, and show their practical utility. Over 50% of the features remain consistent across random seeds, and reconstruction quality is preserved. SAE features capture general acoustic and semantic information as well as specific events, including environmental noises and paralinguistic sounds (e.g. laughter, whispering) and disentangle them effectively, requiring removal of only 19-27% of features to erase a concept. Feature steering reduces Whisper's false speech detections by 70% with negligible WER increase, demonstrating real-world applicability. Finally, we find SAE features correlated with human EEG activity during speech perception, indicating alignment with human neural processing. The code and checkpoints are available at https://github.com/audiosae/audiosae_demo.
>
---
#### [new 006] Zero-Shot TTS With Enhanced Audio Prompts: Bsc Submission For The 2026 Wildspoof Challenge TTS Track
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决真实场景下语音生成的自然性和鲁棒性问题。通过改进模型和音频增强技术，提升零样本合成效果。**

- **链接: [https://arxiv.org/pdf/2602.05770v1](https://arxiv.org/pdf/2602.05770v1)**

> **作者:** Jose Giraldo; Alex Peiró-Lilja; Rodolfo Zevallos; Cristina España-Bonet
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** We evaluate two non-autoregressive architectures, StyleTTS2 and F5-TTS, to address the spontaneous nature of in-the-wild speech. Our models utilize flexible duration modeling to improve prosodic naturalness. To handle acoustic noise, we implement a multi-stage enhancement pipeline using the Sidon model, which significantly outperforms standard Demucs in signal quality. Experimental results show that finetuning enhanced audios yields superior robustness, achieving up to 4.21 UTMOS and 3.47 DNSMOS. Furthermore, we analyze the impact of reference prompt quality and length on zero-shot synthesis performance, demonstrating the effectiveness of our approach for realistic speech generation.
>
---
#### [new 007] Exterior sound field estimation based on physics-constrained kernel
- **分类: eess.AS**

- **简介: 该论文属于声场重建任务，解决外场声场插值问题。提出基于物理约束核的高斯过程方法，无需特定阵列配置，可自动衰减高阶谐波，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2602.05236v1](https://arxiv.org/pdf/2602.05236v1)**

> **作者:** Juliano G. C. Ribeiro; Ryo Matsuda; Jorge Trevino
>
> **备注:** This paper has been accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Exterior sound field interpolation is a challenging problem that often requires specific array configurations and prior knowledge on the source conditions. We propose an interpolation method based on Gaussian processes using a point source reproducing kernel with a trainable inner product formulation made to fit exterior sound fields. While this estimation does not have a closed formula, it allows for the definition of a flexible estimator that is not restricted by microphone distribution and attenuates higher harmonic orders automatically with parameters directly optimized from the recordings, meaning an arbitrary distribution of microphones can be used. The proposed kernel estimator is compared in simulated experiments to the conventional method using spherical wave functions and an established physics-informed machine learning model, achieving lower interpolation error by approximately 2 dB on average within the analyzed frequencies of 100 Hz and 2.5 kHz and reconstructing the ground truth sound field more consistently within the target region.
>
---
#### [new 008] Wave-Trainer-Fit: Neural Vocoder with Trainable Prior and Fixed-Point Iteration towards High-Quality Speech Generation from SSL features
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出WaveTrainerFit，一种基于SSL特征的高质量语音生成神经声码器，解决语音合成中的波形建模问题，通过可训练先验和固定点迭代提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.05443v1](https://arxiv.org/pdf/2602.05443v1)**

> **作者:** Hien Ohnaka; Yuma Shirahata; Masaya Kawamura
>
> **备注:** Accepted by IEEE ICASSP 2026. 5 pages, 3 figures, and 2 tables
>
> **摘要:** We propose WaveTrainerFit, a neural vocoder that performs high-quality waveform generation from data-driven features such as SSL features. WaveTrainerFit builds upon the WaveFit vocoder, which integrates diffusion model and generative adversarial network. Furthermore, the proposed method incorporates the following key improvements: 1. By introducing trainable priors, the inference process starts from noise close to the target speech instead of Gaussian noise. 2. Reference-aware gain adjustment is performed by imposing constraints on the trainable prior to matching the speech energy. These improvements are expected to reduce the complexity of waveform modeling from data-driven features, enabling high-quality waveform generation with fewer inference steps. Through experiments, we showed that WaveTrainerFit can generate highly natural waveforms with improved speaker similarity from data-driven features, while requiring fewer iterations than WaveFit. Moreover, we showed that the proposed method works robustly with respect to the depth at which SSL features are extracted. Code and pre-trained models are available from https://github.com/line/WaveTrainerFit.
>
---
#### [new 009] Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出Bagpiper模型，解决音频理解与生成任务，通过丰富描述实现音频与概念的双向映射，提升音频处理的通用性与质量。**

- **链接: [https://arxiv.org/pdf/2602.05220v1](https://arxiv.org/pdf/2602.05220v1)**

> **作者:** Jinchuan Tian; Haoran Wang; Bo-Hao Su; Chien-yu Huang; Qingzheng Wang; Jiatong Shi; William Chen; Xun Gong; Siddhant Arora; Chin-Jou Li; Masao Someki; Takashi Maekaku; Yusuke Shinohara; Jin Sakuma; Chao-Han Huck Yang; Shinji Watanabe
>
> **摘要:** Current audio foundation models typically rely on rigid, task-specific supervision, addressing isolated factors of audio rather than the whole. In contrast, human intelligence processes audio holistically, seamlessly bridging physical signals with abstract cognitive concepts to execute complex tasks. Grounded in this philosophy, we introduce Bagpiper, an 8B audio foundation model that interprets physical audio via rich captions, i.e., comprehensive natural language descriptions that encapsulate the critical cognitive concepts inherent in the signal (e.g., transcription, audio events). By pre-training on a massive corpus of 600B tokens, the model establishes a robust bidirectional mapping between raw audio and this high-level conceptual space. During fine-tuning, Bagpiper adopts a caption-then-process workflow, simulating an intermediate cognitive reasoning step to solve diverse tasks without task-specific priors. Experimentally, Bagpiper outperforms Qwen-2.5-Omni on MMAU and AIRBench for audio understanding and surpasses CosyVoice3 and TangoFlux in generation quality, capable of synthesizing arbitrary compositions of speech, music, and sound effects. To the best of our knowledge, Bagpiper is among the first works that achieve unified understanding generation for general audio. Model, data, and code are available at Bagpiper Home Page.
>
---
#### [new 010] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于多模态学习任务，解决完整与不完整模态输入下的性能下降问题。提出CyIN框架，通过信息瓶颈和跨模态翻译提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04920v1](https://arxiv.org/pdf/2602.04920v1)**

> **作者:** Ronghao Lin; Qiaolin He; Sijie Mai; Ying Zeng; Aolin Xiong; Li Huang; Yap-Peng Tan; Haifeng Hu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Multimodal machine learning, mimicking the human brain's ability to integrate various modalities has seen rapid growth. Most previous multimodal models are trained on perfectly paired multimodal input to reach optimal performance. In real-world deployments, however, the presence of modality is highly variable and unpredictable, causing the pre-trained models in suffering significant performance drops and fail to remain robust with dynamic missing modalities circumstances. In this paper, we present a novel Cyclic INformative Learning framework (CyIN) to bridge the gap between complete and incomplete multimodal learning. Specifically, we firstly build an informative latent space by adopting token- and label-level Information Bottleneck (IB) cyclically among various modalities. Capturing task-related features with variational approximation, the informative bottleneck latents are purified for more efficient cross-modal interaction and multimodal fusion. Moreover, to supplement the missing information caused by incomplete multimodal input, we propose cross-modal cyclic translation by reconstruct the missing modalities with the remained ones through forward and reverse propagation process. With the help of the extracted and reconstructed informative latents, CyIN succeeds in jointly optimizing complete and incomplete multimodal learning in one unified model. Extensive experiments on 4 multimodal datasets demonstrate the superior performance of our method in both complete and diverse incomplete scenarios.
>
---
#### [new 011] Knowing When to Answer: Adaptive Confidence Refinement for Reliable Audio-Visual Question Answering
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于音频-视觉问答任务，旨在提升模型在不确定时选择不回答的能力。提出ACR方法，通过修正置信度提高可靠性。**

- **链接: [https://arxiv.org/pdf/2602.04924v1](https://arxiv.org/pdf/2602.04924v1)**

> **作者:** Dinh Phu Tran; Jihoon Jeong; Saad Wazir; Seongah Kim; Thao Do; Cem Subakan; Daeyoung Kim
>
> **备注:** Technical Report
>
> **摘要:** We present a formal problem formulation for \textit{Reliable} Audio-Visual Question Answering ($\mathcal{R}$-AVQA), where we prefer abstention over answering incorrectly. While recent AVQA models have high accuracy, their ability to identify when they are likely wrong and their consequent abstention from answering remain underexplored areas of research. To fill this gap, we explore several approaches and then propose Adaptive Confidence Refinement (ACR), a lightweight method to further enhance the performance of $\mathcal{R}$-AVQA. Our key insight is that the Maximum Softmax Probability (MSP) is Bayes-optimal only under strong calibration, a condition usually not met in deep neural networks, particularly in multimodal models. Instead of replacing MSP, our ACR maintains it as a primary confidence signal and applies input-adaptive residual corrections when MSP is deemed unreliable. ACR introduces two learned heads: i) a Residual Risk Head that predicts low-magnitude correctness residuals that MSP does not capture, and ii) a Confidence Gating Head to determine MSP trustworthiness. Our experiments and theoretical analysis show that ACR consistently outperforms existing methods on in- and out-of-disrtibution, and data bias settings across three different AVQA architectures, establishing a solid foundation for $\mathcal{R}$-AVQA task. The code and checkpoints will be available upon acceptance \href{https://github.com/PhuTran1005/R-AVQA}{at here}
>
---
#### [new 012] Phase-Only Positioning in Distributed MIMO Under Phase Impairments: AP Selection Using Deep Learning
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于定位任务，解决相位同步误差下的高精度定位问题。通过提出超椭圆交叉方法和深度学习的AP选择框架，提升定位精度并降低计算复杂度。**

- **链接: [https://arxiv.org/pdf/2602.05034v1](https://arxiv.org/pdf/2602.05034v1)**

> **作者:** Fatih Ayten; Musa Furkan Keskin; Akshay Jain; Mehmet C. Ilter; Ossi Kaltiokallio; Jukka Talvitie; Elena Simona Lohan; Mikko Valkama
>
> **摘要:** Carrier phase positioning (CPP) can enable cm-level accuracy in next-generation wireless systems, while recent literature shows that accuracy remains high using phase-only measurements in distributed MIMO (D-MIMO). However, the impact of phase synchronization errors on such systems remains insufficiently explored. To address this gap, we first show that the proposed hyperbola intersection method achieves highly accurate positioning even in the presence of phase synchronization errors, when trained on appropriate data reflecting such impairments. We then introduce a deep learning (DL)-based D-MIMO antenna point (AP) selection framework that ensures high-precision localization under phase synchronization errors. Simulation results show that the proposed framework improves positioning accuracy compared to prior-art methods, while reducing inference complexity by approximately 19.7%.
>
---
#### [new 013] Instantaneous Spectra Analysis of Pulse Series - Application to Lung Sounds with Abnormalities
- **分类: physics.soc-ph; cs.SD**

- **简介: 该论文属于信号处理任务，旨在解决传统傅里叶分析时间-频率分辨率受限的问题。通过引入LXC替代PBC，实现脉冲序列的瞬时谱分析，并应用于异常肺音研究。**

- **链接: [https://arxiv.org/pdf/2602.03680v1](https://arxiv.org/pdf/2602.03680v1)**

> **作者:** Fumihiko Ishiyama
>
> **备注:** 10 pages, 6 figures.To appear Proc. IEEE CSPA 2026
>
> **摘要:** The origin of the "theoretical limit of time-frequency resolution of Fourier analysis" is from its numerical implementation, especially from an assumption of "Periodic Boundary Condition (PBC)," which was introduced a century ago. We previously proposed to replace this condition with "Linear eXtrapolation Condition (LXC)," which does not require periodicity. This feature makes instantaneous spectra analysis of pulse series available, which replaces the short time Fourier transform (STFT). We applied the instantaneous spectra analysis to two lung sounds with abnormalities (crackles and wheezing) and to a normal lung sound, as a demonstration. Among them, crackles contains a random pulse series. The spectrum of each pulse is available, and the spectrogram of pulse series is available with assembling each spectrum. As a result, the time-frequency structure of given pulse series is visualized.
>
---
#### [new 014] A$^2$-LLM: An End-to-end Conversational Audio Avatar Large Language Model
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于对话式数字人任务，旨在解决传统系统因模块化架构导致的延迟高、情感表达不足问题。提出A$^2$-LLM模型，实现语言、音频和面部动作的端到端联合推理。**

- **链接: [https://arxiv.org/pdf/2602.04913v1](https://arxiv.org/pdf/2602.04913v1)**

> **作者:** Xiaolin Hu; Hang Yuan; Xinzhu Sang; Binbin Yan; Zhou Yu; Cong Huang; Kai Chen
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Developing expressive and responsive conversational digital humans is a cornerstone of next-generation human-computer interaction. While large language models (LLMs) have significantly enhanced dialogue capabilities, most current systems still rely on cascaded architectures that connect independent modules. These pipelines are often plagued by accumulated errors, high latency, and poor real-time performance. Lacking access to the underlying conversational context, these pipelines inherently prioritize rigid lip-sync over emotional depth. To address these challenges, we propose A$^2$-LLM, an end-to-end conversational audio avatar large language model that jointly reasons about language, audio prosody, and 3D facial motion within a unified framework. To facilitate training, we introduce FLAME-QA, a high-quality multimodal dataset designed to align semantic intent with expressive facial dynamics within a QA format. By leveraging deep semantic understanding, A$^2$-LLM generates emotionally rich facial movements beyond simple lip-synchronization. Experimental results demonstrate that our system achieves superior emotional expressiveness while maintaining real-time efficiency (500 ms latency, 0.7 RTF).
>
---
## 更新

#### [replaced 001] UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型任务，解决音频表示与生成问题。提出ReasoningCodec音频编码器，提升理解与生成质量，并构建UniAudio 2.0模型，实现跨任务的少样本和零样本泛化。**

- **链接: [https://arxiv.org/pdf/2602.04683v2](https://arxiv.org/pdf/2602.04683v2)**

> **作者:** Dongchao Yang; Yuanyuan Wang; Dading Chong; Songxiang Liu; Xixin Wu; Helen Meng
>
> **摘要:** We study two foundational problems in audio language models: (1) how to design an audio tokenizer that can serve as an intermediate representation for both understanding and generation; and (2) how to build an audio foundation model that generalizes in few-shot and zero-shot settings, analogous to large language models. To this end, we make the following two contributions. First, we propose ReasoningCodec, a discrete audio codec that factorizes audio into (i) reasoning tokens, which encode text-aligned, high-level analysis and planning representations for audio understanding and hierarchical generation, and (ii) reconstruction tokens, which encode semantic-rich acoustic cues for high-fidelity waveform reconstruction. This design achieves understanding performance comparable to strong continuous representations while improving generation quality and reconstruction fidelity over prior discrete tokenizers. Second, we introduce a unified autoregressive architecture for text and audio, together with multi-stage training and multi-task data construction. Using this framework, we train UniAudio 2.0 on 100B text tokens and 60B audio tokens. Across a wide range of speech, sound, and music tasks, UniAudio 2.0 performs competitively on in-domain evaluations and demonstrates strong few-shot and zero-shot generalization to unseen tasks. Demo, code, and checkpoints will be available at \href{https://dongchaoyang.top/UniAudio2Demo/}{https://dongchaoyang.top/UniAudio2Demo/}.
>
---
#### [replaced 002] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出TASTE方法，解决语音与文本对齐问题，用于提升语音语言模型性能。**

- **链接: [https://arxiv.org/pdf/2504.07053v3](https://arxiv.org/pdf/2504.07053v3)**

> **作者:** Liang-Hsuan Tseng; Yi-Chang Chen; Kuan-Yi Lee; Da-Shan Shiu; Hung-yi Lee
>
> **备注:** ICLR 2026
>
> **摘要:** Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human-LLM interaction. Joint speech-text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Experimental results show that TASTE-based SLMs perform comparable to previous work on SALMON and StoryCloze; while significantly outperform other pre-trained SLMs on speech continuation across subjective and objective evaluations. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and model are available at https://mtkresearch.github.io/TASTE-SpokenLM.github.io.
>
---
#### [replaced 003] BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于符号音乐和弦识别任务，解决数据稀缺与人类分析策略缺失的问题。提出BAHCI模型，通过边界检测和迭代排序提升识别效果。**

- **链接: [https://arxiv.org/pdf/2510.06528v2](https://arxiv.org/pdf/2510.06528v2)**

> **作者:** Mingyang Yao; Ke Chen; Shlomo Dubnov; Taylor Berg-Kirkpatrick
>
> **备注:** Accepted by IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2026
>
> **摘要:** Automatic chord recognition (ACR) via deep learning models has gradually achieved promising recognition accuracy, yet two key challenges remain. First, prior work has primarily focused on audio-domain ACR, while symbolic music (e.g., score) ACR has received limited attention due to data scarcity. Second, existing methods still overlook strategies that are aligned with human music analytical practices. To address these challenges, we make two contributions: (1) we introduce POP909-CL, an enhanced version of POP909 dataset with tempo-aligned content and human-corrected labels of chords, beats, keys, and time signatures; and (2) We propose BACHI, a symbolic chord recognition model that decomposes the task into different decision steps, namely boundary detection and iterative ranking of chord root, quality, and bass (inversion). This mechanism mirrors the human ear-training practices. Experiments demonstrate that BACHI achieves state-of-the-art chord recognition performance on both classical and pop music benchmarks, with ablation studies validating the effectiveness of each module.
>
---
#### [replaced 004] Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection
- **分类: eess.AS; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频亮点检测任务，旨在解决音频模态利用不足的问题。提出双路径音频编码器，融合语义与动态特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.03891v2](https://arxiv.org/pdf/2602.03891v2)**

> **作者:** Seohyun Joo; Yoori Oh
>
> **备注:** 5 pages, 2 figures, to appear in ICASSP 2026
>
> **摘要:** Audio-visual video highlight detection aims to automatically identify the most salient moments in videos by leveraging both visual and auditory cues. However, existing models often underutilize the audio modality, focusing on high-level semantic features while failing to fully leverage the rich, dynamic characteristics of sound. To address this limitation, we propose a novel framework, Dual-Pathway Audio Encoders for Video Highlight Detection (DAViHD). The dual-pathway audio encoder is composed of a semantic pathway for content understanding and a dynamic pathway that captures spectro-temporal dynamics. The semantic pathway extracts high-level information by identifying the content within the audio, such as speech, music, or specific sound events. The dynamic pathway employs a frequency-adaptive mechanism as time evolves to jointly model these dynamics, enabling it to identify transient acoustic events via salient spectral bands and rapid energy changes. We integrate the novel audio encoder into a full audio-visual framework and achieve new state-of-the-art performance on the large-scale MrHiSum benchmark. Our results demonstrate that a sophisticated, dual-faceted audio representation is key to advancing the field of highlight detection.
>
---
#### [replaced 005] Leveraging Whisper Embeddings for Audio-based Lyrics Matching
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频歌词匹配任务，旨在解决现有方法 reproducibility 差和基线不一致的问题。提出 WEALY 管道，利用 Whisper 嵌入进行歌词匹配，并验证其性能与先进方法相当。**

- **链接: [https://arxiv.org/pdf/2510.08176v2](https://arxiv.org/pdf/2510.08176v2)**

> **作者:** Eleonora Mancini; Joan Serrà; Paolo Torroni; Yuki Mitsufuji
>
> **备注:** Accepted at ICASSP 2026 (IEEE International Conference on Acoustics, Speech and Signal Processing)
>
> **摘要:** Audio-based lyrics matching can be an appealing alternative to other content-based retrieval approaches, but existing methods often suffer from limited reproducibility and inconsistent baselines. In this work, we introduce WEALY, a fully reproducible pipeline that leverages Whisper decoder embeddings for lyrics matching tasks. WEALY establishes robust and transparent baselines, while also exploring multimodal extensions that integrate textual and acoustic features. Through extensive experiments on standard datasets, we demonstrate that WEALY achieves a performance comparable to state-of-the-art methods that lack reproducibility. In addition, we provide ablation studies and analyses on language robustness, loss functions, and embedding strategies. This work contributes a reliable benchmark for future research, and underscores the potential of speech technologies for music information retrieval tasks.
>
---
#### [replaced 006] UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于音频超分辨率任务，解决传统方法依赖声码器的问题。通过无声码器的流匹配模型直接重建波形，提升音频质量。**

- **链接: [https://arxiv.org/pdf/2510.00771v2](https://arxiv.org/pdf/2510.00771v2)**

> **作者:** Woongjib Choi; Sangmin Lee; Hyungseob Lim; Hong-Goo Kang
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** In this paper, we present a vocoder-free framework for audio super-resolution that employs a flow matching generative model to capture the conditional distribution of complex-valued spectral coefficients. Unlike conventional two-stage diffusion-based approaches that predict a mel-spectrogram and then rely on a pre-trained neural vocoder to synthesize waveforms, our method directly reconstructs waveforms via the inverse Short-Time Fourier Transform (iSTFT), thereby eliminating the dependence on a separate vocoder. This design not only simplifies end-to-end optimization but also overcomes a critical bottleneck of two-stage pipelines, where the final audio quality is fundamentally constrained by vocoder performance. Experiments show that our model consistently produces high-fidelity 48 kHz audio across diverse upsampling factors, achieving state-of-the-art performance on both speech and general audio datasets.
>
---
#### [replaced 007] Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音大模型任务，解决领域术语识别问题。提出LOGIC框架，在解码层实现高效上下文偏置，提升实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.15397v2](https://arxiv.org/pdf/2601.15397v2)**

> **作者:** Peidong Wang
>
> **备注:** This paper is withdrawn temporarily to ensure full compliance with internal institutional publication approval processes
>
> **摘要:** The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
>
---
#### [replaced 008] ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation
- **分类: cs.SD**

- **简介: 该论文介绍ACE-Step 1.5，一个高效开源音乐生成模型，解决在消费级硬件上实现高质量音乐生成的问题。通过混合架构与内在强化学习，提升生成效率与风格控制能力。**

- **链接: [https://arxiv.org/pdf/2602.00744v2](https://arxiv.org/pdf/2602.00744v2)**

> **作者:** Junmin Gong; Yulin Song; Wenxiao Zhao; Sen Wang; Shengyuan Xu; Jing Guo
>
> **摘要:** We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast -- under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style. At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints -- scaling from short loops to 10-minute compositions -- while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities -- such as cover generation, repainting, and vocal-to-BGM conversion -- while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. The code, the model weights and the demo are available at: https://ace-step.github.io/ace-step-v1.5.github.io/
>
---
#### [replaced 009] Reasoning Beyond Majority Vote: An Explainable SpeechLM Framework for Speech Emotion Recognition
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，解决传统方法依赖多数投票标签导致解释性差的问题。提出可解释的SpeechLM框架，通过生成理由提升模型透明度，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2509.24187v2](https://arxiv.org/pdf/2509.24187v2)**

> **作者:** Bo-Hao Su; Hui-Ying Shih; Jinchuan Tian; Jiatong Shi; Chi-Chun Lee; Carlos Busso; Shinji Watanabe
>
> **摘要:** Speech Emotion Recognition (SER) is typically trained and evaluated on majority-voted labels, which simplifies benchmarking but masks subjectivity and provides little transparency into why predictions are made. This neglects valid minority annotations and limits interpretability. We propose an explainable Speech Language Model (SpeechLM) framework that frames SER as a generative reasoning task. Given an utterance, the model first produces a transcript, then outputs both an emotion label and a concise natural-language rationale grounded in lexical and acoustic cues. Rationales are generated by a reasoning-capable teacher LLM and used as intermediate supervision, combined with majority labels during fine-tuning. Unlike prior work primarily focused on boosting classification accuracy, we aim to enhance explainability while preserving competitive performance. To this end, we complement majority-label metrics with annotator-aware scoring that credits matches with any annotator label. On MSP-Podcast v1.12, our model maintains improvements over zero-shot SpeechLM baselines, and produces rationales that human evaluators find plausible and well grounded. This demonstrates that incorporating rationale supervision offers a practical path toward interpretable SER without sacrificing predictive quality.
>
---
#### [replaced 010] Audio Inpainting in Time-Frequency Domain with Phase-Aware Prior
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频修复任务，解决时间-频率域音频补全问题。通过引入相位感知先验和优化算法，提升重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2601.18535v2](https://arxiv.org/pdf/2601.18535v2)**

> **作者:** Peter Balušík; Pavel Rajmic
>
> **备注:** submitted to IEEE for review
>
> **摘要:** We address the problem of time-frequency audio inpainting, where the goal is to fill missing spectrogram portions with reliable information. Despite recent advances, existing approaches still face limitations in both reconstruction quality and computational efficiency. To bridge this gap, we propose a method that utilizes a phase-aware signal prior which exploits estimates of the instantaneous frequency. An optimization problem is formulated and solved using the generalized Chambolle-Pock algorithm. The proposed method is evaluated against other time-frequency inpainting methods, specifically a deep-prior audio inpainting neural network and the autoregression-based approach known as Janssen-TF. Our proposed approach surpassed these methods by a large margin in the objective evaluation as well as in the conducted subjective listening test, improving the state of the art. In addition, the reconstructions are obtained with a substantially reduced computational cost compared to alternative methods.
>
---
#### [replaced 011] ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 该论文属于语音防伪任务，旨在解决环境噪声中语音和声音的深度伪造检测问题。提出CompSpoofV2数据集和联合学习框架，提升组件级伪造检测能力。**

- **链接: [https://arxiv.org/pdf/2601.07303v5](https://arxiv.org/pdf/2601.07303v5)**

> **作者:** Xueping Zhang; Han Yin; Yang Xiao; Lin Zhang; Ting Dang; Rohan Kumar Das; Ming Li
>
> **摘要:** Audio recorded in real-world environments often contains a mixture of foreground speech and background environmental sounds. With rapid advances in text-to-speech, voice conversion, and other generation models, either component can now be modified independently. Such component-level manipulations are harder to detect, as the remaining unaltered component can mislead the systems designed for whole deepfake audio, and they often sound more natural to human listeners. To address this gap, we have proposed CompSpoofV2 dataset and a separation-enhanced joint learning framework. CompSpoofV2 is a large-scale curated dataset designed for component-level audio anti-spoofing, which contains over 250k audio samples, with a total duration of approximately 283 hours. Based on the CompSpoofV2 and the separation-enhanced joint learning framework, we launch the Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2), focusing on component-level spoofing, where both speech and environmental sounds may be manipulated or synthesized, creating a more challenging and realistic detection scenario. The challenge will be held in conjunction with the IEEE International Conference on Multimedia and Expo 2026 (ICME 2026).
>
---
#### [replaced 012] Video Soundtrack Generation by Aligning Emotions and Temporal Boundaries
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于视频配乐生成任务，旨在解决视频与音乐在情感和时间上对齐的问题。工作包括设计两阶段框架，引入边界偏移和情感映射机制，提升生成音乐的准确性与协调性。**

- **链接: [https://arxiv.org/pdf/2502.10154v3](https://arxiv.org/pdf/2502.10154v3)**

> **作者:** Serkan Sulun; Paula Viana; Matthew E. P. Davies
>
> **备注:** IEEE Transactions on Multimedia, 2026, in print
>
> **摘要:** Providing soundtracks for videos remains a costly and time-consuming challenge for multimedia content creators. We introduce EMSYNC, an automatic video-based symbolic music generator that creates music aligned with a video's emotional content and temporal boundaries. It follows a two-stage framework, where a pretrained video emotion classifier extracts emotional features, and a conditional music generator produces MIDI sequences guided by both emotional and temporal cues. We introduce boundary offsets, a novel temporal conditioning mechanism that enables the model to anticipate upcoming video scene cuts and align generated musical chords with them. We also propose a mapping scheme that bridges the discrete categorical outputs of the video emotion classifier with the continuous valence-arousal inputs required by the emotion-conditioned MIDI generator, enabling seamless integration of emotion information across different representations. Our method outperforms state-of-the-art models in objective and subjective evaluations across different video datasets, demonstrating its effectiveness in generating music aligned to video both emotionally and temporally. Our demo and output samples are available at https://serkansulun.com/emsync.
>
---
#### [replaced 013] Stream-Voice-Anon: Enhancing Utility of Real-Time Speaker Anonymization via Neural Audio Codec and Language Models
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音匿名化任务，旨在保护在线语音应用中的说话人身份。针对现有方法隐私不足的问题，提出Stream-Voice-Anon，结合神经音频编解码器与语言模型，提升语音可懂度和情感保留，同时保障隐私。**

- **链接: [https://arxiv.org/pdf/2601.13948v3](https://arxiv.org/pdf/2601.13948v3)**

> **作者:** Nikita Kuzmin; Songting Liu; Kong Aik Lee; Eng Siong Chng
>
> **备注:** Accepted by ICASSP2026. Demo/code: https://paniquex.github.io/Stream-Voice-Anon/
>
> **摘要:** Protecting speaker identity is crucial for online voice applications, yet streaming speaker anonymization (SA) remains underexplored. Recent research has demonstrated that neural audio codec (NAC) provides superior speaker feature disentanglement and linguistic fidelity. NAC can also be used with causal language models (LM) to enhance linguistic fidelity and prompt control for streaming tasks. However, existing NAC-based online LM systems are designed for voice conversion (VC) rather than anonymization, lacking the techniques required for privacy protection. Building on these advances, we present Stream-Voice-Anon, which adapts modern causal LM-based NAC architectures specifically for streaming SA by integrating anonymization techniques. Our anonymization approach incorporates pseudo-speaker representation sampling, a speaker embedding mixing and diverse prompt selection strategies for LM conditioning that leverage the disentanglement properties of quantized content codes to prevent speaker information leakage. Additionally, we compare dynamic and fixed delay configurations to explore latency-privacy trade-offs in real-time scenarios. Under the VoicePrivacy 2024 Challenge protocol, Stream-Voice-Anon achieves substantial improvements in intelligibility (up to 46% relative WER reduction) and emotion preservation (up to 28% UAR relative) compared to the previous state-of-the-art streaming method DarkStream while maintaining comparable latency (180ms vs 200ms) and privacy protection against lazy-informed attackers, though showing 15% relative degradation against semi-informed attackers.
>
---
#### [replaced 014] Segmentation-free Goodness of Pronunciation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别中的发音评估任务，旨在解决传统方法依赖语音分段的问题。提出两种无需分段的发音质量评估方法，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2507.16838v3](https://arxiv.org/pdf/2507.16838v3)**

> **作者:** Xinwei Cao; Zijian Fan; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** The article has been accepted for publication by IEEE TASLPRO
>
> **摘要:** Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer-aided language learning (CALL) systems. Most systems implementing phoneme-level MDD through goodness of pronunciation (GOP), however, rely on pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general segmentation-free method that takes all possible segmentations of the canonical transcription into account (GOP-SF). We give a theoretical account of our definition of GOP-SF, an implementation that solves potential numerical issues as well as a proper normalization which allows the use of acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-SF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.
>
---
