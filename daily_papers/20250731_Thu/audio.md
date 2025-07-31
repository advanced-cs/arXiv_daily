# 音频 cs.SD;  eess.SP

- **最新发布 7 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Quantum-Inspired Audio Unlearning: Towards Privacy-Preserving Voice Biometrics
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音生物特征隐私保护任务，旨在解决已训练模型中特定语音数据的高效擦除问题。针对现有方法在音频数据上的不足，作者提出了QPAudioEraser框架，结合量子启发机制，实现对目标语音特征的有效擦除，同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2507.22208v1](http://arxiv.org/pdf/2507.22208v1)**

> **作者:** Shreyansh Pathak; Sonu Shreshtha; Richa Singh; Mayank Vatsa
>
> **备注:** 9 pages, 2 figures, 5 tables, Accepted at IJCB 2025 (Osaka, Japan)
>
> **摘要:** The widespread adoption of voice-enabled authentication and audio biometric systems have significantly increased privacy vulnerabilities associated with sensitive speech data. Compliance with privacy regulations such as GDPR's right to be forgotten and India's DPDP Act necessitates targeted and efficient erasure of individual-specific voice signatures from already-trained biometric models. Existing unlearning methods designed for visual data inadequately handle the sequential, temporal, and high-dimensional nature of audio signals, leading to ineffective or incomplete speaker and accent erasure. To address this, we introduce QPAudioEraser, a quantum-inspired audio unlearning framework. Our our-phase approach involves: (1) weight initialization using destructive interference to nullify target features, (2) superposition-based label transformations that obscure class identity, (3) an uncertainty-maximizing quantum loss function, and (4) entanglement-inspired mixing of correlated weights to retain model knowledge. Comprehensive evaluations with ResNet18, ViT, and CNN architectures across AudioMNIST, Speech Commands, LibriSpeech, and Speech Accent Archive datasets validate QPAudioEraser's superior performance. The framework achieves complete erasure of target data (0% Forget Accuracy) while incurring minimal impact on model utility, with a performance degradation on retained data as low as 0.05%. QPAudioEraser consistently surpasses conventional baselines across single-class, multi-class, sequential, and accent-level erasure scenarios, establishing the proposed approach as a robust privacy-preserving solution.
>
---
#### [new 002] Adaptive Duration Model for Text Speech Alignment
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在解决语音与文本对齐不稳定、泛化能力差的问题。作者提出了一种自适应的音素级时长预测模型，提升对齐精度和跨领域适应能力，实验显示其对齐准确率提升11.3%，并增强了零样本TTS模型的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.22612v1](http://arxiv.org/pdf/2507.22612v1)**

> **作者:** Junjie Cao
>
> **备注:** 4 pages, 3 figures, 2 tables
>
> **摘要:** Speech-to-text alignment is a critical component of neural text to-speech (TTS) models. Autoregressive TTS models typically use an attention mechanism to learn these alignments on-line. However, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, leading to missing or repeating words. Most non-autoregressive end to-end TTS models rely on durations extracted from external sources, using additional duration models for alignment. In this paper, we propose a novel duration prediction framework that can give compromising phoneme-level duration distribution with given text. In our experiments, the proposed duration model has more precise prediction and condition adaptation ability compared to previous baseline models. Numerically, it has roughly a 11.3 percents immprovement on alignment accuracy, and makes the performance of zero-shot TTS models more robust to the mismatch between prompt audio and input audio.
>
---
#### [new 003] Next Tokens Denoising for Speech Synthesis
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出Dragon-FM模型，用于语音合成任务。为解决自回归模型生成慢和扩散模型缓存难的问题，结合两者优势，采用分块自回归与块内流匹配方法，实现高效高质量语音生成。**

- **链接: [http://arxiv.org/pdf/2507.22746v1](http://arxiv.org/pdf/2507.22746v1)**

> **作者:** Yanqing Liu; Ruiqing Xue; Chong Zhang; Yufei Liu; Gang Wang; Bohan Li; Yao Qian; Lei He; Shujie Liu; Sheng Zhao
>
> **摘要:** While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact 12.5 tokens per second rate. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Consequently, the proposed model can utilize KV-cache across chunks and incorporate future context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also makes the proposed model particularly effective for generating extended content. Experiment for demos of our work} on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts.
>
---
#### [new 004] Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination
- **分类: eess.SP; cs.CV; 68T45; I.5.4**

- **简介: 该论文属于雷达人体运动方向识别任务，旨在解决低成本高精度运动方向判定问题。通过生成雷达多普勒时域图，结合特征增强模型，并采用轻量混合神经网络结构进行方向识别，验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.22567v1](http://arxiv.org/pdf/2507.22567v1)**

> **作者:** Weicheng Gao
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** This work is completed on a whim after discussions with my junior colleague. The motion direction angle affects the micro-Doppler spectrum width, thus determining the human motion direction can provide important prior information for downstream tasks such as gait recognition. However, Doppler-Time map (DTM)-based methods still have room for improvement in achieving feature augmentation and motion determination simultaneously. In response, a low-cost but accurate radar-based human motion direction determination (HMDD) method is explored in this paper. In detail, the radar-based human gait DTMs are first generated, and then the feature augmentation is achieved using feature linking model. Subsequently, the HMDD is implemented through a lightweight and fast Vision Transformer-Convolutional Neural Network hybrid model structure. The effectiveness of the proposed method is verified through open-source dataset. The open-source code of this work is released at: https://github.com/JoeyBGOfficial/Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination.
>
---
#### [new 005] A Two-Step Learning Framework for Enhancing Sound Event Localization and Detection
- **分类: cs.SD; eess.AS**

- **简介: 本文属于声音事件定位与检测（SELD）任务，旨在解决现有模型在事件检测与方向估计间的优化冲突与信息限制。论文提出一种两步学习框架：首先通过轨迹重排序保持时间一致性，再分别训练事件检测与方向估计网络，最后融合特征提升性能。实验验证了方法在2023 DCASE挑战赛数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22322v1](http://arxiv.org/pdf/2507.22322v1)**

> **作者:** Hogeon Yu
>
> **备注:** 5pages, 2figures
>
> **摘要:** Sound Event Localization and Detection (SELD) is crucial in spatial audio processing, enabling systems to detect sound events and estimate their 3D directions. Existing SELD methods use single- or dual-branch architectures: single-branch models share SED and DoA representations, causing optimization conflicts, while dual-branch models separate tasks but limit information exchange. To address this, we propose a two-step learning framework. First, we introduce a tracwise reordering format to maintain temporal consistency, preventing event reassignments across tracks. Next, we train SED and DoA networks to prevent interference and ensure task-specific feature learning. Finally, we effectively fuse DoA and SED features to enhance SELD performance with better spatial and event representation. Experiments on the 2023 DCASE challenge Task 3 dataset validate our framework, showing its ability to overcome single- and dual-branch limitations and improve event classification and localization.
>
---
#### [new 006] A k-space approach to modeling multi-channel parametric array loudspeaker systems
- **分类: eess.AS; cs.SD**

- **简介: 论文研究多通道参数阵扬声器（MCPAL）系统的建模方法，旨在高效准确预测其声场。该任务属于声学仿真与建模。现有方法因非线性复杂性和多通道处理效率低，论文提出基于k空间的方法，结合角谱法和三维快速傅里叶变换，实现高效计算且无需近似，显著提升速度与精度。**

- **链接: [http://arxiv.org/pdf/2507.22628v1](http://arxiv.org/pdf/2507.22628v1)**

> **作者:** Tao Zhuang; Longbiao He; Feng Niu; Jia-Xin Zhong; Jing Lu
>
> **摘要:** Multi-channel parametric array loudspeaker (MCPAL) systems offer enhanced flexibility and promise for generating highly directional audio beams in real-world applications. However, efficient and accurate prediction of their generated sound fields remains a major challenge due to the complex nonlinear behavior and multi-channel signal processing involved. To overcome this obstacle, we propose a k-space approach for modeling arbitrary MCPAL systems arranged on a baffled planar surface. In our method, the linear ultrasound field is first solved using the angular spectrum approach, and the quasilinear audio sound field is subsequently computed efficiently in k-space. By leveraging three-dimensional fast Fourier transforms, our approach not only achieves high computational and memory efficiency but also maintains accuracy without relying on the paraxial approximation. For typical configurations studied, the proposed method demonstrates a speed-up of more than four orders of magnitude compared to the direct integration method. Our proposed approach paved the way for simulating and designing advanced MCPAL systems.
>
---
#### [new 007] Prediction of acoustic field in 1-D uniform duct with varying mean flow and temperature using neural networks
- **分类: cs.LG; cs.SD; eess.AS; 34A06; G.1.6; I.6.4; J.2**

- **简介: 该论文属于物理建模与机器学习结合的任务，旨在解决一维管道中声场预测问题。考虑了介质流动和温度变化的影响，通过神经网络求解声学控制方程，并与传统方法对比验证。研究了温度梯度对声场的影响，展示了迁移学习和自动微分在声学中的应用。**

- **链接: [http://arxiv.org/pdf/2507.22370v1](http://arxiv.org/pdf/2507.22370v1)**

> **作者:** D. Veerababu; Prasanta K. Ghosh
>
> **备注:** 22 pages
>
> **摘要:** Neural networks constrained by the physical laws emerged as an alternate numerical tool. In this paper, the governing equation that represents the propagation of sound inside a one-dimensional duct carrying a heterogeneous medium is derived. The problem is converted into an unconstrained optimization problem and solved using neural networks. Both the acoustic state variables: acoustic pressure and particle velocity are predicted and validated with the traditional Runge-Kutta solver. The effect of the temperature gradient on the acoustic field is studied. Utilization of machine learning techniques such as transfer learning and automatic differentiation for acoustic applications is demonstrated.
>
---
## 更新

#### [replaced 001] ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling
- **分类: cs.CL; eess.SP; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2412.14373v3](http://arxiv.org/pdf/2412.14373v3)**

> **作者:** William Han; Chaojing Duan; Michael A. Rosenberg; Emerson Liu; Ding Zhao
>
> **备注:** 38 pages, 9 figures; Accepted to MLHC 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional versatility across domains, including applications to electrocardiograms (ECGs). A growing body of work focuses on generating text from multi-channeled ECG signals and corresponding textual prompts. Existing approaches often involve a two-stage process: pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective, followed by finetuning an LLM for natural language generation (NLG) using encoder-derived features. However, these methods face two key limitations: inefficiency due to multi-stage training and challenges in interpreting encoder-generated features. To overcome these issues, we propose ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. ECG-Byte compresses and encodes ECG signals into tokens, enabling direct end-to-end LLM training by combining ECG and text tokens. This approach enhances interpretability, as ECG tokens can be directly mapped back to the original signals. Leveraging ECG-Byte, we achieve competitive NLG performance while training 3 times faster and using just 48\% of the data required by traditional two-stage methods.
>
---
#### [replaced 002] BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.00059v2](http://arxiv.org/pdf/2505.00059v2)**

> **作者:** Paige Tuttösí; Mantaj Dhillon; Luna Sang; Shane Eastwood; Poorvi Bhatia; Quang Minh Dinh; Avni Kapoor; Yewon Jin; Angelica Lim
>
> **备注:** Accepted to Computer Speech and Language, Special issue: Multi-Speaker, Multi-Microphone, and Multi-Modal Distant Speech Recognition. Project Webpage and Data access : https://huggingface.co/datasets/Rosie-Lab/BERSt
>
> **摘要:** Some speech recognition tasks, such as automatic speech recognition (ASR), are approaching or have reached human performance in many reported metrics. Yet, they continue to struggle in complex, real-world, situations, such as with distanced speech. Previous challenges have released datasets to address the issue of distanced ASR, however, the focus remains primarily on distance, specifically relying on multi-microphone array systems. Here we present the B(asic) E(motion) R(andom phrase) S(hou)t(s) (BERSt) dataset. The dataset contains almost 4 hours of English speech from 98 actors with varying regional and non-native accents. The data was collected on smartphones in the actors homes and therefore includes at least 98 different acoustic environments. The data also includes 7 different emotion prompts and both shouted and spoken utterances. The smartphones were places in 19 different positions, including obstructions and being in a different room than the actor. This data is publicly available for use and can be used to evaluate a variety of speech recognition tasks, including: ASR, shout detection, and speech emotion recognition (SER). We provide initial benchmarks for ASR and SER tasks, and find that ASR degrades both with an increase in distance and shout level and shows varied performance depending on the intended emotion. Our results show that the BERSt dataset is challenging for both ASR and SER tasks and continued work is needed to improve the robustness of such systems for more accurate real-world use.
>
---
#### [replaced 003] Controllable joint noise reduction and hearing loss compensation using a differentiable auditory model
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.09372v2](http://arxiv.org/pdf/2507.09372v2)**

> **作者:** Philippe Gonzalez; Torsten Dau; Tobias May
>
> **备注:** Accepted to Clarity 2025 Workshop
>
> **摘要:** Deep learning-based hearing loss compensation (HLC) seeks to enhance speech intelligibility and quality for hearing impaired listeners using neural networks. One major challenge of HLC is the lack of a ground-truth target. Recent works have used neural networks to emulate non-differentiable auditory peripheral models in closed-loop frameworks, but this approach lacks flexibility. Alternatively, differentiable auditory models allow direct optimization, yet previous studies focused on individual listener profiles, or joint noise reduction (NR) and HLC without balancing each task. This work formulates NR and HLC as a multi-task learning problem, training a system to simultaneously predict denoised and compensated signals from noisy speech and audiograms using a differentiable auditory model. Results show the system achieves similar objective metric performance to systems trained for each task separately, while being able to adjust the balance between NR and HLC during inference.
>
---
#### [replaced 004] I Know You're Listening: Adaptive Voice for HRI
- **分类: cs.RO; cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15107v2](http://arxiv.org/pdf/2506.15107v2)**

> **作者:** Paige Tuttösí
>
> **备注:** PhD Thesis Simon Fraser University https://summit.sfu.ca/item/39353 Read the Room: IROS 2023, Mmm whatcha say?: INTERSPEECH 2024, Emojivoice: RO-MAN 2025, You sound a little tense: SSW 2025. Thesis presentation here: https://www.youtube.com/watch?v=9BcEwqYOMYI
>
> **摘要:** While the use of social robots for language teaching has been explored, there remains limited work on a task-specific synthesized voices for language teaching robots. Given that language is a verbal task, this gap may have severe consequences for the effectiveness of robots for language teaching tasks. We address this lack of L2 teaching robot voices through three contributions: 1. We address the need for a lightweight and expressive robot voice. Using a fine-tuned version of Matcha-TTS, we use emoji prompting to create an expressive voice that shows a range of expressivity over time. The voice can run in real time with limited compute resources. Through case studies, we found this voice more expressive, socially appropriate, and suitable for long periods of expressive speech, such as storytelling. 2. We explore how to adapt a robot's voice to physical and social ambient environments to deploy our voices in various locations. We found that increasing pitch and pitch rate in noisy and high-energy environments makes the robot's voice appear more appropriate and makes it seem more aware of its current environment. 3. We create an English TTS system with improved clarity for L2 listeners using known linguistic properties of vowels that are difficult for these listeners. We used a data-driven, perception-based approach to understand how L2 speakers use duration cues to interpret challenging words with minimal tense (long) and lax (short) vowels in English. We found that the duration of vowels strongly influences the perception for L2 listeners and created an "L2 clarity mode" for Matcha-TTS that applies a lengthening to tense vowels while leaving lax vowels unchanged. Our clarity mode was found to be more respectful, intelligible, and encouraging than base Matcha-TTS while reducing transcription errors in these challenging tense/lax minimal pairs.
>
---
#### [replaced 005] Exploring Textual Semantics Diversity for Image Transmission in Semantic Communication Systems using Visual Language Model
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.19386v2](http://arxiv.org/pdf/2503.19386v2)**

> **作者:** Peishan Huang; Dong Li
>
> **摘要:** In recent years, the rapid development of machine learning has brought reforms and challenges to traditional communication systems. Semantic communication has appeared as an effective strategy to effectively extract relevant semantic signals semantic segmentation labels and image features for image transmission. However, the insufficient number of extracted semantic features of images will potentially result in a low reconstruction accuracy, which hinders the practical applications and still remains challenging for solving. In order to fill this gap, this letter proposes a multi-text transmission semantic communication (Multi-SC) system, which uses the visual language model (VLM) to assist in the transmission of image semantic signals. Unlike previous image transmission semantic communication systems, the proposed system divides the image into multiple blocks and extracts multiple text information from the image using a modified large language and visual assistant (LLaVA), and combines semantic segmentation tags with semantic text for image recovery. Simulation results show that the proposed text semantics diversity scheme can significantly improve the reconstruction accuracy compared with related works.
>
---
#### [replaced 006] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02038v2](http://arxiv.org/pdf/2411.02038v2)**

> **作者:** Yongxin Zhu; Bocheng Li; Yifei Xin; Zhihua Xia; Linli Xu
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** Vector Quantization (VQ) is essential for discretizing continuous representations in unsupervised learning but suffers from representation collapse, causing low codebook utilization and limiting scalability. Existing solutions often rely on complex optimizations or reduce latent dimensionality, which compromises model capacity and fails to fully solve the problem. We identify the root cause as disjoint codebook optimization, where only a few code vectors are updated via gradient descent. To fix this, we propose \textbf{Sim}ple\textbf{VQ}, which reparameterizes code vectors through a learnable linear transformation layer over a latent basis, optimizing the \textit{entire linear space} rather than nearest \textit{individual code vectors}. Although the multiplication of two linear matrices is equivalent to applying a single linear layer, this simple approach effectively prevents collapse. Extensive experiments on image and audio tasks demonstrate that SimVQ improves codebook usage, is easy to implement, and generalizes well across modalities and architectures.
>
---
#### [replaced 007] Mmm whatcha say? Uncovering distal and proximal context effects in first and second-language word perception using psychophysical reverse correlation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.05515v2](http://arxiv.org/pdf/2406.05515v2)**

> **作者:** Paige Tuttösí; H. Henny Yeung; Yue Wang; Fenqi Wang; Guillaume Denis; Jean-Julien Aucouturier; Angelica Lim
>
> **备注:** Accepted to INTERSPEECH 2024 Project Webpage : https://rosielab.github.io/vocal_ambiguity/ Code: https://github.com/neuro-team-femto/vocal_ambiguity Data : https://zenodo.org/records/12761242
>
> **摘要:** Acoustic context effects, where surrounding changes in pitch, rate or timbre influence the perception of a sound, are well documented in speech perception, but how they interact with language background remains unclear. Using a reverse-correlation approach, we systematically varied the pitch and speech rate in phrases around different pairs of vowels for second language (L2) speakers of English (/i/-/I/) and French (/u/-/y/), thus reconstructing, in a data-driven manner, the prosodic profiles that bias their perception. Testing English and French speakers (n=25), we showed that vowel perception is in fact influenced by conflicting effects from the surrounding pitch and speech rate: a congruent proximal effect 0.2s pre-target and a distal contrastive effect up to 1s before; and found that L1 and L2 speakers exhibited strikingly similar prosodic profiles in perception. We provide a novel method to investigate acoustic context effects across stimuli, timescales, and acoustic domain.
>
---
#### [replaced 008] BENYO-S2ST-Corpus-1: A Bilingual English-to-Yoruba Direct Speech-to-Speech Translation Corpus
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09342v3](http://arxiv.org/pdf/2507.09342v3)**

> **作者:** Emmanuel Adetiba; Abdultaofeek Abayomi; Raymond J. Kala; Ayodele H. Ifijeh; Oluwatobi E. Dare; Olabode Idowu-Bismark; Gabriel O. Sobola; Joy N. Adetiba; Monsurat Adepeju Lateef; Heather Cole-Lewis
>
> **摘要:** There is a major shortage of Speech-to-Speech Translation (S2ST) datasets for high resource-to-low resource language pairs such as English-to-Yoruba. Thus, in this study, we curated the Bilingual English-to-Yoruba Speech-to-Speech Translation Corpus Version 1 (BENYO-S2ST-Corpus-1). The corpus is based on a hybrid architecture we developed for large-scale direct S2ST corpus creation at reduced cost. To achieve this, we leveraged non speech-to-speech Standard Yoruba (SY) real-time audios and transcripts in the YORULECT Corpus as well as the corresponding Standard English (SE) transcripts. YORULECT Corpus is small scale(1,504) samples, and it does not have paired English audios. Therefore, we generated the SE audios using pre-trained AI models (i.e. Facebook MMS). We also developed an audio augmentation algorithm named AcoustAug based on three latent acoustic features to generate augmented audios from the raw audios of the two languages. BENYO-S2ST-Corpus-1 has 12,032 audio samples per language, which gives a total of 24,064 sample size. The total audio duration for the two languages is 41.20 hours. This size is quite significant. Beyond building S2ST models, BENYO-S2ST-Corpus-1 can be used to build pretrained models or improve existing ones. The created corpus and Coqui framework were used to build a pretrained Yoruba TTS model (named YoruTTS-1.5) as a proof of concept. The YoruTTS-1.5 gave a F0 RMSE value of 63.54 after 1,000 epochs, which indicates moderate fundamental pitch similarity with the reference real-time audio. Ultimately, the corpus architecture in this study can be leveraged by researchers and developers to curate datasets for multilingual high-resource-to-low-resource African languages. This will bridge the huge digital divides in translations among high and low-resource language pairs. BENYO-S2ST-Corpus-1 and YoruTTS-1.5 are publicly available at (https://bit.ly/40bGMwi).
>
---
#### [replaced 009] CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.20040v2](http://arxiv.org/pdf/2502.20040v2)**

> **作者:** Nian Shao; Rui Zhou; Pengyu Wang; Xian Li; Ying Fang; Yujie Yang; Xiaofei Li
>
> **备注:** Submission to IEEE/ACM Trans. on TASLP
>
> **摘要:** In this work, we propose CleanMel, a single-channel Mel-spectrogram denoising and dereverberation network for improving both speech quality and automatic speech recognition (ASR) performance. The proposed network takes as input the noisy and reverberant microphone recording and predicts the corresponding clean Mel-spectrogram. The enhanced Mel-spectrogram can be either transformed to the speech waveform with a neural vocoder or directly used for ASR. The proposed network is composed of interleaved cross-band and narrow-band processing in the Mel-frequency domain, for learning the full-band spectral pattern and the narrow-band properties of signals, respectively. Compared to linear-frequency domain or time-domain speech enhancement, the key advantage of Mel-spectrogram enhancement is that Mel-frequency presents speech in a more compact way and thus is easier to learn, which will benefit both speech quality and ASR. Experimental results on five English and one Chinese datasets demonstrate a significant improvement in both speech quality and ASR performance achieved by the proposed model.Code and audio examples of our model are available online.
>
---
#### [replaced 010] Text-Driven Voice Conversion via Latent State-Space Modeling
- **分类: cs.SD; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.20999v2](http://arxiv.org/pdf/2503.20999v2)**

> **作者:** Wen Li; Sofia Martinez; Priyanka Shah
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship and affiliation
>
> **摘要:** Text-driven voice conversion allows customization of speaker characteristics and prosodic elements using textual descriptions. However, most existing methods rely heavily on direct text-to-speech training, limiting their flexibility in controlling nuanced style elements or timbral features. In this paper, we propose a novel \textbf{Latent State-Space} approach for text-driven voice conversion (\textbf{LSS-VC}). Our method treats each utterance as an evolving dynamical system in a continuous latent space. Drawing inspiration from mamba, which introduced a state-space model for efficient text-driven \emph{image} style transfer, we adapt a loosely related methodology for \emph{voice} style transformation. Specifically, we learn a voice latent manifold where style and content can be manipulated independently by textual style prompts. We propose an adaptive cross-modal fusion mechanism to inject style information into the voice latent representation, enabling interpretable and fine-grained control over speaker identity, speaking rate, and emphasis. Extensive experiments show that our approach significantly outperforms recent baselines in both subjective and objective quality metrics, while offering smoother transitions between styles, reduced artifacts, and more precise text-based style control.
>
---
#### [replaced 011] Uncovering the role of semantic and acoustic cues in normal and dichotic listening
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2411.11308v2](http://arxiv.org/pdf/2411.11308v2)**

> **作者:** Sai Samrat Kankanala; Akshara Soman; Sriram Ganapathy
>
> **备注:** 10 Pages, 6 Figures
>
> **摘要:** Speech comprehension is an involuntary task for the healthy human brain, yet the understanding of the mechanisms underlying this brain functionality remains obscure. In this paper, we aim to quantify the role of acoustic and semantic information streams in complex listening conditions. We propose a paradigm to understand the encoding of the speech cues in electroencephalogram (EEG) data, by designing a match-mismatch (MM) classification task. The MM task involves identifying whether the stimulus (speech) and response (EEG) correspond to each other. We build a multimodal deep-learning based sequence model STEM, which is input with acoustic stimulus (speech envelope), semantic stimulus (textual representations of speech), and the neural response (EEG data). We perform extensive experiments on two separate conditions, i) natural passive listening and, ii) a dichotic listening requiring auditory attention. Using the MM task as the analysis framework, we observe that - a) speech perception is fragmented based on word boundaries, b) acoustic and semantic cues offer similar levels of MM task performance in natural listening conditions, and c) semantic cues offer significantly improved MM classification over acoustic cues in dichotic listening task. The comparison of the STEM with previously proposed MM models shows significant performance improvements for the proposed approach. The analysis and understanding from this study allows the quantification of the roles played by acoustic and semantic cues in diverse listening tasks and in providing further evidences of right-ear advantage in dichotic listening.
>
---
