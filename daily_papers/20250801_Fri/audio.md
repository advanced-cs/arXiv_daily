# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] Identifying Hearing Difficulty Moments in Conversational Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在识别日常对话中听力困难的时刻。为实现这一目标，作者比较了多种机器学习方法，结果显示音频语言模型在检测听力困难时刻上优于传统方法，包括基于关键词的简单启发式方法和基于Wav2Vec的微调方法。**

- **链接: [http://arxiv.org/pdf/2507.23590v1](http://arxiv.org/pdf/2507.23590v1)**

> **作者:** Jack Collins; Adrian Buzea; Chris Collier; Alejandro Ballesta Rosen; Julian Maclaren; Richard F. Lyon; Simon Carlile
>
> **摘要:** Individuals regularly experience Hearing Difficulty Moments in everyday conversation. Identifying these moments of hearing difficulty has particular significance in the field of hearing assistive technology where timely interventions are key for realtime hearing assistance. In this paper, we propose and compare machine learning solutions for continuously detecting utterances that identify these specific moments in conversational audio. We show that audio language models, through their multimodal reasoning capabilities, excel at this task, significantly outperforming a simple ASR hotword heuristic and a more conventional fine-tuning approach with Wav2Vec, an audio-only input architecture that is state-of-the-art for automatic speech recognition (ASR).
>
---
#### [new 002] Balancing Information Preservation and Disentanglement in Self-Supervised Music Representation Learning
- **分类: cs.SD**

- **简介: 该论文属于自监督音乐表示学习任务，旨在解决在无需标注数据的情况下，如何平衡信息保留与语义解耦的问题。作者提出了一种结合对比学习和重建目标的多视图框架，以实现音乐属性的解耦表示，同时保持信息完整性，并通过实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22995v1](http://arxiv.org/pdf/2507.22995v1)**

> **作者:** Julia Wilkins; Sivan Ding; Magdalena Fuentes; Juan Pablo Bello
>
> **备注:** In proceedings of WASPAA 2025. 4 pages, 4 figures, 1 table
>
> **摘要:** Recent advances in self-supervised learning (SSL) methods offer a range of strategies for capturing useful representations from music audio without the need for labeled data. While some techniques focus on preserving comprehensive details through reconstruction, others favor semantic structure via contrastive objectives. Few works examine the interaction between these paradigms in a unified SSL framework. In this work, we propose a multi-view SSL framework for disentangling music audio representations that combines contrastive and reconstructive objectives. The architecture is designed to promote both information fidelity and structured semantics of factors in disentangled subspaces. We perform an extensive evaluation on the design choices of contrastive strategies using music audio representations in a controlled setting. We find that while reconstruction and contrastive strategies exhibit consistent trade-offs, when combined effectively, they complement each other; this enables the disentanglement of music attributes without compromising information integrity.
>
---
#### [new 003] "I made this (sort of)": Negotiating authorship, confronting fraudulence, and exploring new musical spaces with prompt-based AI music generation
- **分类: cs.SD; cs.AI; eess.AS; I.2; J.5**

- **简介: 论文探讨了基于提示的AI音乐生成中的创作权、身份认同与音乐空间拓展问题。作者通过创作两部AI生成音乐专辑，反思自身在AI创作中的角色与地位，并利用大语言模型进行自我访谈，探索AI时代下创作主体性和音乐身份的转变。**

- **链接: [http://arxiv.org/pdf/2507.23365v1](http://arxiv.org/pdf/2507.23365v1)**

> **作者:** Bob L. T. Sturm
>
> **摘要:** I reflect on my experience creating two music albums centered on state-of-the-art prompt-based AI music generation platforms. The first album explicitly poses the question: What happens when I collide my junk mail with these platforms? The second album is a direct response to the first, and toys with the inability of state-of-the-art prompt-based AI music generation platforms to generate music that is not ``practiced'', ``polished'', and ``produced''. I seed a large language model (LLM) with information about these albums and have it interview me, which results in the exploration of several deeper questions: To what extent am I the author? Where am I in the resulting music? How is my musical identity changing as I am faced with machines that are in some ways far more talented than I? What new musical spaces does my work open, for me or anyone/thing else? I conclude by reflecting on my reflections, as well as LLM-mediated self-reflection as method.
>
---
#### [new 004] Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于人机交互任务，旨在解决对话系统中缺乏自然非语言行为的问题。作者提出了一种实时预测虚拟角色点头类型和时机的模型，基于语音活动预测并结合多任务学习与预训练，实现了更自然的倾听系统，主观评价优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.23298v1](http://arxiv.org/pdf/2507.23298v1)**

> **作者:** Kazushi Kato; Koji Inoue; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** Accepted by 27th ACM International Conference on Multimodal Interaction (ICMI '25), Long paper
>
> **摘要:** In human dialogue, nonverbal information such as nodding and facial expressions is as crucial as verbal information, and spoken dialogue systems are also expected to express such nonverbal behaviors. We focus on nodding, which is critical in an attentive listening system, and propose a model that predicts both its timing and type in real time. The proposed model builds on the voice activity projection (VAP) model, which predicts voice activity from both listener and speaker audio. We extend it to prediction of various types of nodding in a continuous and real-time manner unlike conventional models. In addition, the proposed model incorporates multi-task learning with verbal backchannel prediction and pretraining on general dialogue data. In the timing and type prediction task, the effectiveness of multi-task learning was significantly demonstrated. We confirmed that reducing the processing rate enables real-time operation without a substantial drop in accuracy, and integrated the model into an avatar attentive listening system. Subjective evaluations showed that it outperformed the conventional method, which always does nodding in sync with verbal backchannel. The code and trained models are available at https://github.com/MaAI-Kyoto/MaAI.
>
---
#### [new 005] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于音频理解任务，旨在解决现有模型在细粒度音频理解上的不足。作者提出了MECAT基准和DATE评估指标，通过专家模型与大语言模型结合生成细粒度标注，提升模型对细节的捕捉能力，评估显示现有模型仍有改进空间。**

- **链接: [http://arxiv.org/pdf/2507.23511v1](http://arxiv.org/pdf/2507.23511v1)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Xunying Liu; Junbo Zhang; Jian Luan
>
> **备注:** 9 main pages, 5 figures, 3 tables, and 14 appendix pages
>
> **摘要:** While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at https://github.com/xiaomi-research/mecat
>
---
#### [new 006] Investigating the Invertibility of Multimodal Latent Spaces: Limitations of Optimization-Based Methods
- **分类: cs.LG; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于多模态AI模型任务，旨在探索多模态潜在空间的可逆性问题。作者提出基于优化的方法，尝试从输出反推输入，发现在文本-图像和文本-音频模型中，尽管优化能实现文本对齐，但反向映射在感知质量和语义解释上表现混乱，表明当前多模态潜在空间不支持稳健的可逆映射。**

- **链接: [http://arxiv.org/pdf/2507.23010v1](http://arxiv.org/pdf/2507.23010v1)**

> **作者:** Siwoo Park
>
> **摘要:** This paper investigates the inverse capabilities and broader utility of multimodal latent spaces within task-specific AI (Artificial Intelligence) models. While these models excel at their designed forward tasks (e.g., text-to-image generation, audio-to-text transcription), their potential for inverse mappings remains largely unexplored. We propose an optimization-based framework to infer input characteristics from desired outputs, applying it bidirectionally across Text-Image (BLIP, Flux.1-dev) and Text-Audio (Whisper-Large-V3, Chatterbox-TTS) modalities. Our central hypothesis posits that while optimization can guide models towards inverse tasks, their multimodal latent spaces will not consistently support semantically meaningful and perceptually coherent inverse mappings. Experimental results consistently validate this hypothesis. We demonstrate that while optimization can force models to produce outputs that align textually with targets (e.g., a text-to-image model generating an image that an image captioning model describes correctly, or an ASR model transcribing optimized audio accurately), the perceptual quality of these inversions is chaotic and incoherent. Furthermore, when attempting to infer the original semantic input from generative models, the reconstructed latent space embeddings frequently lack semantic interpretability, aligning with nonsensical vocabulary tokens. These findings highlight a critical limitation. multimodal latent spaces, primarily optimized for specific forward tasks, do not inherently possess the structure required for robust and interpretable inverse mappings. Our work underscores the need for further research into developing truly semantically rich and invertible multimodal latent spaces.
>
---
#### [new 007] CUHK-EE Systems for the vTAD Challenge at NCMMSC 2025
- **分类: eess.AS; cs.SD**

- **简介: 该论文参与的是语音音色属性检测（vTAD）任务，旨在通过模型比较语音对的音色属性强度，解决音色感知主观性和数据不平衡带来的挑战。研究团队采用WavLM-Large提取语音特征，并构建两种Diff-Net变体（FFN和SE-ResFFN）进行属性比较。实验表明不同模型在泛化能力和已见设置下表现各异，揭示了模型复杂度与泛化能力之间的权衡。**

- **链接: [http://arxiv.org/pdf/2507.23266v1](http://arxiv.org/pdf/2507.23266v1)**

> **作者:** Aemon Yat Fei Chiu; Jingyu Li; Yusheng Tian; Guangyan Zhang; Tan Lee
>
> **备注:** Under review
>
> **摘要:** This paper presents the Voice Timbre Attribute Detection (vTAD) systems developed by the Digital Signal Processing & Speech Technology Laboratory (DSP&STL) of the Department of Electronic Engineering (EE) at The Chinese University of Hong Kong (CUHK) for the 20th National Conference on Human-Computer Speech Communication (NCMMSC 2025) vTAD Challenge. The proposed systems leverage WavLM-Large embeddings with attentive statistical pooling to extract robust speaker representations, followed by two variants of Diff-Net, i.e., Feed-Forward Neural Network (FFN) and Squeeze-and-Excitation-enhanced Residual FFN (SE-ResFFN), to compare timbre attribute intensities between utterance pairs. Experimental results demonstrate that the WavLM-Large+FFN system generalises better to unseen speakers, achieving 77.96% accuracy and 21.79% EER, while the WavLM-Large+SE-ResFFN model excels in the 'Seen' setting with 94.42% accuracy and 5.49% EER. These findings highlight a trade-off between model complexity and generalisation, and underscore the importance of architectural choices in fine-grained speaker modelling. Our analysis also reveals the impact of speaker identity, annotation subjectivity, and data imbalance on system performance, pointing to future directions for improving robustness and fairness in timbre attribute detection.
>
---
#### [new 008] Exploring Dynamic Parameters for Vietnamese Gender-Independent ASR
- **分类: eess.AS; cs.CL; cs.SD; eess.SP**

- **简介: 该论文属于语音识别任务，旨在提升越南语自动语音识别（ASR）的性别无关性能。通过引入基于子带质心频率的动态参数，结合传统MFCC特征，有效减少频谱变化并增强声学建模，从而降低词错误率，并提高对不同性别的适应性。**

- **链接: [http://arxiv.org/pdf/2507.22964v1](http://arxiv.org/pdf/2507.22964v1)**

> **作者:** Sotheara Leang; Éric Castelli; Dominique Vaufreydaz; Sethserey Sam
>
> **摘要:** The dynamic characteristics of speech signal provides temporal information and play an important role in enhancing Automatic Speech Recognition (ASR). In this work, we characterized the acoustic transitions in a ratio plane of Spectral Subband Centroid Frequencies (SSCFs) using polar parameters to capture the dynamic characteristics of the speech and minimize spectral variation. These dynamic parameters were combined with Mel-Frequency Cepstral Coefficients (MFCCs) in Vietnamese ASR to capture more detailed spectral information. The SSCF0 was used as a pseudo-feature for the fundamental frequency (F0) to describe the tonal information robustly. The findings showed that the proposed parameters significantly reduce word error rates and exhibit greater gender independence than the baseline MFCCs.
>
---
#### [new 009] Feature Importance across Domains for Improving Non-Intrusive Speech Intelligibility Prediction in Hearing Aids
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音可懂度预测任务，旨在提升助听器中非侵入式语音可懂度评估的性能。论文提出FiDo方法，通过跨域特征重要性加权，优化模型对关键特征的关注。结合MBI-Net+模型，实验表明该方法显著降低了RMSE，优于现有最优系统。**

- **链接: [http://arxiv.org/pdf/2507.23223v1](http://arxiv.org/pdf/2507.23223v1)**

> **作者:** Ryandhimas E. Zezario; Sabato M. Siniscalchi; Fei Chen; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Given the critical role of non-intrusive speech intelligibility assessment in hearing aids (HA), this paper enhances its performance by introducing Feature Importance across Domains (FiDo). We estimate feature importance on spectral and time-domain acoustic features as well as latent representations of Whisper. Importance weights are calculated per frame, and based on these weights, features are projected into new spaces, allowing the model to focus on important areas early. Next, feature concatenation is performed to combine the features before the assessment module processes them. Experimental results show that when FiDo is incorporated into the improved multi-branched speech intelligibility model MBI-Net+, RMSE can be reduced by 7.62% (from 26.10 to 24.11). MBI-Net+ with FiDo also achieves a relative RMSE reduction of 3.98% compared to the best system in the 2023 Clarity Prediction Challenge. These results validate FiDo's effectiveness in enhancing neural speech assessment in HA.
>
---
#### [new 010] Impact of a Lower Limb Exosuit Anchor Points on Energetics and Biomechanics
- **分类: physics.med-ph; cs.RO; eess.SP**

- **简介: 该论文研究下肢外骨骼锚点位置对能量消耗和生物力学的影响，属于外骨骼设计优化任务。通过六种实验配置，分析不同锚点对髋、膝、踝关节运动及肌肉激活的影响。结果显示锚点位置显著影响效果，最优位置因人而异，需个性化设计。**

- **链接: [http://arxiv.org/pdf/2507.23579v1](http://arxiv.org/pdf/2507.23579v1)**

> **作者:** Chiara Lambranzi; Giulia Oberti; Christian Di Natali; Darwin G. Caldwell; Manuela Galli; Elena De Momi; Jesùs Ortiz
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** Anchor point placement is a crucial yet often overlooked aspect of exosuit design since it determines how forces interact with the human body. This work analyzes the impact of different anchor point positions on gait kinematics, muscular activation and energetic consumption. A total of six experiments were conducted with 11 subjects wearing the XoSoft exosuit, which assists hip flexion in five configurations. Subjects were instrumented with an IMU-based motion tracking system, EMG sensors, and a mask to measure metabolic consumption. The results show that positioning the knee anchor point on the posterior side while keeping the hip anchor on the anterior part can reduce muscle activation in the hip flexors by up to 10.21\% and metabolic expenditure by up to 18.45\%. Even if the only assisted joint was the hip, all the configurations introduced changes also in the knee and ankle kinematics. Overall, no single configuration was optimal across all subjects, suggesting that a personalized approach is necessary to transmit the assistance forces optimally. These findings emphasize that anchor point position does indeed have a significant impact on exoskeleton effectiveness and efficiency. However, these optimal positions are subject-specific to the exosuit design, and there is a strong need for future work to tailor musculoskeletal models to individual characteristics and validate these results in clinical populations.
>
---
#### [new 011] Moravec's Paradox: Towards an Auditory Turing Test
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出了一项基于莫拉维克悖论的听觉图灵测试任务，旨在评估人工智能在复杂听觉场景中的表现。研究覆盖7类听觉挑战，揭示当前AI模型在重叠语音、噪声等场景中表现极差，准确率仅6.9%，远低于人类水平。论文聚焦于听觉任务中人机差距，指出AI缺乏选择性注意、噪声鲁棒性和情境适应机制，强调需将听觉物理理解和上下文感知融入多模态AI系统。**

- **链接: [http://arxiv.org/pdf/2507.23091v1](http://arxiv.org/pdf/2507.23091v1)**

> **作者:** David Noever; Forrest McKee
>
> **摘要:** This research work demonstrates that current AI systems fail catastrophically on auditory tasks that humans perform effortlessly. Drawing inspiration from Moravec's paradox (i.e., tasks simple for humans often prove difficult for machines, and vice versa), we introduce an auditory Turing test comprising 917 challenges across seven categories: overlapping speech, speech in noise, temporal distortion, spatial audio, coffee-shop noise, phone distortion, and perceptual illusions. Our evaluation of state-of-the-art audio models including GPT-4's audio capabilities and OpenAI's Whisper reveals a striking failure rate exceeding 93%, with even the best-performing model achieving only 6.9% accuracy on tasks that humans solved at 7.5 times higher success (52%). These results expose focusing failures in how AI systems process complex auditory scenes, particularly in selective attention, noise robustness, and contextual adaptation. Our benchmark not only quantifies the human-machine auditory gap but also provides insights into why these failures occur, suggesting that current architectures lack fundamental mechanisms for human-like auditory scene analysis. The traditional design of audio CAPTCHAs highlights common filters that humans evolved but machines fail to select in multimodal language models. This work establishes a diagnostic framework for measuring progress toward human-level machine listening and highlights the need for novel approaches integrating selective attention, physics-based audio understanding, and context-aware perception into multimodal AI systems.
>
---
## 更新

#### [replaced 001] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14874v4](http://arxiv.org/pdf/2505.14874v4)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [replaced 002] DMF2Mel: A Dynamic Multiscale Fusion Network for EEG-Driven Mel Spectrogram Reconstruction
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07526v2](http://arxiv.org/pdf/2507.07526v2)**

> **作者:** Cunhang Fan; Sheng Zhang; Jingjing Zhang; Enrui Liu; Xinhui Li; Minggang Zhao; Zhao Lv
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Decoding speech from brain signals is a challenging research problem. Although existing technologies have made progress in reconstructing the mel spectrograms of auditory stimuli at the word or letter level, there remain core challenges in the precise reconstruction of minute-level continuous imagined speech: traditional models struggle to balance the efficiency of temporal dependency modeling and information retention in long-sequence decoding. To address this issue, this paper proposes the Dynamic Multiscale Fusion Network (DMF2Mel), which consists of four core components: the Dynamic Contrastive Feature Aggregation Module (DC-FAM), the Hierarchical Attention-Guided Multi-Scale Network (HAMS-Net), the SplineMap attention mechanism, and the bidirectional state space module (convMamba). Specifically, the DC-FAM separates speech-related "foreground features" from noisy "background features" through local convolution and global attention mechanisms, effectively suppressing interference and enhancing the representation of transient signals. HAMS-Net, based on the U-Net framework,achieves cross-scale fusion of high-level semantics and low-level details. The SplineMap attention mechanism integrates the Adaptive Gated Kolmogorov-Arnold Network (AGKAN) to combine global context modeling with spline-based local fitting. The convMamba captures long-range temporal dependencies with linear complexity and enhances nonlinear dynamic modeling capabilities. Results on the SparrKULee dataset show that DMF2Mel achieves a Pearson correlation coefficient of 0.074 in mel spectrogram reconstruction for known subjects (a 48% improvement over the baseline) and 0.048 for unknown subjects (a 35% improvement over the baseline).Code is available at: https://github.com/fchest/DMF2Mel.
>
---
#### [replaced 003] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.14534v3](http://arxiv.org/pdf/2507.14534v3)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
