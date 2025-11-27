# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] Musical Score Understanding Benchmark: Evaluating Large Language Models' Comprehension of Complete Musical Scores
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出首个大规模音乐总谱理解基准MSU-Bench，旨在评估大模型对乐谱的多层级理解能力。针对现有模型在符号结构（音高、节奏、和声、形式）理解上的不足，构建涵盖四种认知层次的1800组问答数据，验证了模型在文本与视觉模态下的表现差异及微调效果，为AI与音乐学交叉研究提供基准。**

- **链接: [https://arxiv.org/pdf/2511.20697v1](https://arxiv.org/pdf/2511.20697v1)**

> **作者:** Congren Dai; Yue Yang; Krinos Li; Huichi Zhou; Shijie Liang; Zhang Bo; Enyang Liu; Ge Jin; Hongran An; Haosen Zhang; Peiyuan Jing; KinHei Lee; Zhenxuan Zhang; Xiaobing Li; Maosong Sun
>
> **摘要:** Understanding complete musical scores requires reasoning over symbolic structures such as pitch, rhythm, harmony, and form. Despite the rapid progress of Large Language Models (LLMs) and Vision-Language Models (VLMs) in natural language and multimodal tasks, their ability to comprehend musical notation remains underexplored. We introduce Musical Score Understanding Benchmark (MSU-Bench), the first large-scale, human-curated benchmark for evaluating score-level musical understanding across both textual (ABC notation) and visual (PDF) modalities. MSU-Bench comprises 1,800 generative question-answer (QA) pairs drawn from works spanning Bach, Beethoven, Chopin, Debussy, and others, organised into four progressive levels of comprehension: Onset Information, Notation & Note, Chord & Harmony, and Texture & Form. Through extensive zero-shot and fine-tuned evaluations of over 15+ state-of-the-art (SOTA) models, we reveal sharp modality gaps, fragile level-wise success rates, and the difficulty of sustaining multilevel correctness. Fine-tuning markedly improves performance in both modalities while preserving general knowledge, establishing MSU-Bench as a rigorous foundation for future research at the intersection of Artificial Intelligence (AI), musicological, and multimodal reasoning.
>
---
#### [new 002] SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对深度伪造音频检测中泛化能力差的问题，提出SONAR框架，通过频域引导的对比学习，分离并增强高频残差特征。有效缓解谱偏倚，提升对真实与伪造音频的区分能力，实现更快速收敛与更强泛化性。**

- **链接: [https://arxiv.org/pdf/2511.21325v1](https://arxiv.org/pdf/2511.21325v1)**

> **作者:** Ido Nitzan HIdekel; Gal lifshitz; Khen Cohen; Dan Raviv
>
> **摘要:** Deepfake (DF) audio detectors still struggle to generalize to out of distribution inputs. A central reason is spectral bias, the tendency of neural networks to learn low-frequency structure before high-frequency (HF) details, which both causes DF generators to leave HF artifacts and leaves those same artifacts under-exploited by common detectors. To address this gap, we propose Spectral-cONtrastive Audio Residuals (SONAR), a frequency-guided framework that explicitly disentangles an audio signal into complementary representations. An XLSR encoder captures the dominant low-frequency content, while the same cloned path, preceded by learnable SRM, value-constrained high-pass filters, distills faint HF residuals. Frequency cross-attention reunites the two views for long- and short-range frequency dependencies, and a frequency-aware Jensen-Shannon contrastive loss pulls real content-noise pairs together while pushing fake embeddings apart, accelerating optimization and sharpening decision boundaries. Evaluated on the ASVspoof 2021 and in-the-wild benchmarks, SONAR attains state-of-the-art performance and converges four times faster than strong baselines. By elevating faint high-frequency residuals to first-class learning signals, SONAR unveils a fully data-driven, frequency-guided contrastive framework that splits the latent space into two disjoint manifolds: natural-HF for genuine audio and distorted-HF for synthetic audio, thereby sharpening decision boundaries. Because the scheme operates purely at the representation level, it is architecture-agnostic and, in future work, can be seamlessly integrated into any model or modality where subtle high-frequency cues are decisive.
>
---
#### [new 003] Towards Audio Token Compression in Large Audio Language Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文研究大音频语言模型（LALM）中的音频令牌压缩问题，旨在解决其因注意力复杂度和高令牌率导致的长音频处理难、边缘部署难的问题。通过无监督分段与平均池化减少音频令牌数，并用低秩适配器微调以缓解性能下降。实验表明，压缩后模型在语音识别与语音翻译任务中性能接近帧级模型，令牌数可减少至三分之一。**

- **链接: [https://arxiv.org/pdf/2511.20973v1](https://arxiv.org/pdf/2511.20973v1)**

> **作者:** Saurabhchand Bhati; Samuel Thomas; Hilde Kuehne; Rogerio Feris; James Glass
>
> **摘要:** Large Audio Language Models (LALMs) demonstrate impressive performance across diverse tasks, ranging from speech recognition to general audio understanding. However, their scalability is limited by the quadratic complexity of attention and the high token rates of audio signals. These challenges make it difficult to extend LALMs to long-form audio and to deploy them on resource-constrained platforms such as edge devices. In this paper, we explore techniques such as unsupervised segmentation, uniform average pooling, etc., to reduce the number of audio tokens generated by the LALM's audio encoder but before they are consumed by the LLM decoder. To mitigate potential performance degradation introduced by the compressed representations, we employ low-rank adapters to finetune the model. We evaluate our proposed models on two tasks, automatic speech recognition and speech-to-speech translation tasks, that are dependent on effectively uncovering the underlying lexical content of the input signal and study the effect of downsampling on these tasks. Experimental results show that compressed LALMs can achieve performance closer to frame-level LALMs while reducing the input audio token count upto three times before the LLM backbone.
>
---
#### [new 004] Acoustic neural networks: Identifying design principles and exploring physical feasibility
- **分类: cs.SD; cond-mat.dis-nn; cs.NE; eess.AS; physics.app-ph**

- **简介: 该论文研究声学神经网络的物理可实现设计，旨在解决电子器件在特定环境下效率低的问题。通过数字孪生框架，构建符合声学物理约束的神经网络模型，实现语音分类任务，提出SincHSRNN模型，验证了其高精度与被动元件兼容性，确立了声学神经网络的设计原则。**

- **链接: [https://arxiv.org/pdf/2511.21313v1](https://arxiv.org/pdf/2511.21313v1)**

> **作者:** Ivan Kalthoff; Marcel Rey; Raphael Wittkowski
>
> **备注:** 13 pages, 4 figures, 8 tables
>
> **摘要:** Wave-guide-based physical systems provide a promising route toward energy-efficient analog computing beyond traditional electronics. Within this landscape, acoustic neural networks represent a promising approach for achieving low-power computation in environments where electronics are inefficient or limited, yet their systematic design has remained largely unexplored. Here we introduce a framework for designing and simulating acoustic neural networks, which perform computation through the propagation of sound waves. Using a digital-twin approach, we train conventional neural network architectures under physically motivated constraints including non-negative signals and weights, the absence of bias terms, and nonlinearities compatible with intensity-based, non-negative acoustic signals. Our work provides a general framework for acoustic neural networks that connects learnable network components directly to physically measurable acoustic properties, enabling the systematic design of realizable acoustic computing systems. We demonstrate that constrained recurrent and hierarchical architectures can perform accurate speech classification, and we propose the SincHSRNN, a hybrid model that combines learnable acoustic bandpass filters with hierarchical temporal processing. The SincHSRNN achieves up to 95% accuracy on the AudioMNIST dataset while remaining compatible with passive acoustic components. Beyond computational performance, the learned parameters correspond to measurable material and geometric properties such as attenuation and transmission. Our results establish general design principles for physically realizable acoustic neural networks and outline a pathway toward low-power, wave-based neural computing.
>
---
#### [new 005] CartoonSing: Unifying Human and Nonhuman Timbres in Singing Generation
- **分类: cs.SD**

- **简介: 该论文提出非人类歌唱生成（NHSG）任务，解决现有歌唱合成系统仅限于人声、难以生成非人声歌唱的问题。提出CartoonSing统一框架，通过两阶段设计实现人类与非人类音色的统一生成，利用标注人声数据训练编码器，并结合音色感知声码器，成功生成多样非人声歌唱，拓展了创意应用边界。**

- **链接: [https://arxiv.org/pdf/2511.21045v1](https://arxiv.org/pdf/2511.21045v1)**

> **作者:** Jionghao Han; Jiatong Shi; Zhuoyan Tao; Yuxun Tang; Yiwen Zhao; Gus Xia; Shinji Watanabe
>
> **摘要:** Singing voice synthesis (SVS) and singing voice conversion (SVC) have achieved remarkable progress in generating natural-sounding human singing. However, existing systems are restricted to human timbres and have limited ability to synthesize voices outside the human range, which are increasingly demanded in creative applications such as video games, movies, and virtual characters. We introduce Non-Human Singing Generation (NHSG), covering non-human singing voice synthesis (NHSVS) and non-human singing voice conversion (NHSVC), as a novel machine learning task for generating musically coherent singing with non-human timbral characteristics. NHSG is particularly challenging due to the scarcity of non-human singing data, the lack of symbolic alignment, and the wide timbral gap between human and non-human voices. To address these challenges, we propose CartoonSing, a unified framework that integrates singing voice synthesis and conversion while bridging human and non-human singing generation. CartoonSing employs a two-stage pipeline: a score representation encoder trained with annotated human singing and a timbre-aware vocoder that reconstructs waveforms for both human and non-human audio. Experiments demonstrate that CartoonSing successfully generates non-human singing voices, generalizes to novel timbres, and extends conventional SVS and SVC toward creative, non-human singing generation.
>
---
#### [new 006] Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale
- **分类: cs.SD; cs.CV**

- **简介: 该论文研究单码本文本到语音大模型的稳定性问题。针对其存在的语调不稳、说话人漂移和自然度下降问题，提出多奖励分组相对策略优化框架，融合长度惩罚、熵正则与基于大模型推理的韵律对齐奖励，显著提升语音质量与稳定性，并验证了方法在不同规模下的有效性。**

- **链接: [https://arxiv.org/pdf/2511.21270v1](https://arxiv.org/pdf/2511.21270v1)**

> **作者:** Yicheng Zhong; Peiji Yang; Zhisheng Wang
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.
>
---
#### [new 007] HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对AI生成音频的水印安全问题，提出高效水印移除方法HarmonicAttack。任务为评估音频水印的鲁棒性。通过仅需生成水印的能力，训练通用水印移除模型，利用双路径卷积自编码器与GAN训练，在时频域分离水印，实现近实时、跨样本高效移除，验证了现有水印方案的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.21577v1](https://arxiv.org/pdf/2511.21577v1)**

> **作者:** Kexin Li; Xiao Hu; Ilya Grishchenko; David Lie
>
> **摘要:** The availability of high-quality, AI-generated audio raises security challenges such as misinformation campaigns and voice-cloning fraud. A key defense against the misuse of AI-generated audio is by watermarking it, so that it can be easily distinguished from genuine audio. As those seeking to misuse AI-generated audio may thus seek to remove audio watermarks, studying effective watermark removal techniques is critical to being able to objectively evaluate the robustness of audio watermarks against removal. Previous watermark removal schemes either assume impractical knowledge of the watermarks they are designed to remove or are computationally expensive, potentially generating a false sense of confidence in current watermark schemes. We introduce HarmonicAttack, an efficient audio watermark removal method that only requires the basic ability to generate the watermarks from the targeted scheme and nothing else. With this, we are able to train a general watermark removal model that is able to remove the watermarks generated by the targeted scheme from any watermarked audio sample. HarmonicAttack employs a dual-path convolutional autoencoder that operates in both temporal and frequency domains, along with GAN-style training, to separate the watermark from the original audio. When evaluated against state-of-the-art watermark schemes AudioSeal, WavMark, and Silentcipher, HarmonicAttack demonstrates greater watermark removal ability than previous watermark removal methods with near real-time performance. Moreover, while HarmonicAttack requires training, we find that it is able to transfer to out-of-distribution samples with minimal degradation in performance.
>
---
#### [new 008] RosettaSpeech: Zero-Shot Speech-to-Speech Translation from Monolingual Data
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文提出RosettaSpeech，解决语音翻译中平行语料稀缺问题。通过使用单语语音-文本数据与机器翻译监督，实现零样本语音到语音翻译，无需平行语音对。模型以文本为桥梁训练，推理时端到端直接转换，支持多语言到英语的翻译，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.20974v1](https://arxiv.org/pdf/2511.20974v1)**

> **作者:** Zhisheng Zheng; Xiaohang Sun; Tuan Dinh; Abhishek Yanamandra; Abhinav Jain; Zhu Liu; Sunil Hadap; Vimal Bhat; Manoj Aggarwal; Gerard Medioni; David Harwath
>
> **备注:** Work in progress
>
> **摘要:** The scarcity of parallel speech corpora critically hampers speech-to-speech translation (S2ST), often forcing reliance on complex, multi-stage pipelines. This paper introduces RosettaSpeech, a novel and simplified framework for zero-shot S2ST that is trained on monolingual speech-text data augmented by machine translation supervision. While our method leverages the linguistic knowledge inherent in text-based NMT models, it strictly eliminates the need for parallel speech-to-speech pairs. Our model uniquely uses text as an intermediate bridge during training but functions as a direct, end-to-end speech-to-speech model at inference. This streamlined approach achieves state-of-the-art results on standard benchmarks. For instance, on the CVSS-C test set, RosettaSpeech outperforms leading systems, achieving an ASR-BLEU score of 25.17 for German-to-English and 29.86 for Spanish-to-English-relative gains of over 27% and 14%, respectively. Furthermore, we demonstrate that a single model can deliver strong many-to-one translation performance (FR/ES/DE -> EN). We also provide a foundational analysis of how training data scaling impacts model performance. By prioritizing reliance on abundant parallel text rather than difficult-to-acquire parallel speech, RosettaSpeech offers a scalable path to creating high-quality, speaker-preserving S2ST for a much broader array of languages.
>
---
#### [new 009] The Spheres Dataset: Multitrack Orchestral Recordings for Music Source Separation and Information Retrieval
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出The Spheres数据集，用于古典音乐中的音源分离与音乐信息检索任务。针对复杂管弦乐场景下音源分离困难的问题，构建了含23麦克风的多轨录音数据，涵盖经典作品与独奏片段，并提供声学参数与基准评估，推动模型在分离、定位与沉浸式还原方面的研究。**

- **链接: [https://arxiv.org/pdf/2511.21247v1](https://arxiv.org/pdf/2511.21247v1)**

> **作者:** Jaime Garcia-Martinez; David Diaz-Guerra; John Anderson; Ricardo Falcon-Perez; Pablo Cabañas-Molero; Tuomas Virtanen; Julio J. Carabias-Orti; Pedro Vera-Candeas
>
> **摘要:** This paper introduces The Spheres dataset, multitrack orchestral recordings designed to advance machine learning research in music source separation and related MIR tasks within the classical music domain. The dataset is composed of over one hour recordings of musical pieces performed by the Colibrì Ensemble at The Spheres recording studio, capturing two canonical works - Tchaikovsky's Romeo and Juliet and Mozart's Symphony No. 40 - along with chromatic scales and solo excerpts for each instrument. The recording setup employed 23 microphones, including close spot, main, and ambient microphones, enabling the creation of realistic stereo mixes with controlled bleeding and providing isolated stems for supervised training of source separation models. In addition, room impulse responses were estimated for each instrument position, offering valuable acoustic characterization of the recording space. We present the dataset structure, acoustic analysis, and baseline evaluations using X-UMX based models for orchestral family separation and microphone debleeding. Results highlight both the potential and the challenges of source separation in complex orchestral scenarios, underscoring the dataset's value for benchmarking and for exploring new approaches to separation, localization, dereverberation, and immersive rendering of classical music.
>
---
#### [new 010] Seeing Beyond Sound: Visualization and Abstraction in Audio Data Representation
- **分类: cs.SD; cs.HC; eess.AS**

- **简介: 该论文探讨音频数据可视化与抽象表示，针对传统工具因历史假设导致现代工作流不匹配的问题，提出通过增加维度和交互性提升分析与创作效果。研究以Jellyfish Dynamite软件为例，探索增强型可视化工具在复杂音频研究中的应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.20658v1](https://arxiv.org/pdf/2511.20658v1)**

> **作者:** Ashlae Blum'e
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** In audio signal processing, the interpretation of complex information using visual representation enhances pattern recognition through its alignment with human perceptual systems. Software tools that carry hidden assumptions inherited from their historical contexts risk misalignment with modern workflows as design origins become obscured. We argue that creating tools that align with emergent needs improves analytical and creative outputs due to an increased affinity for using them. This paper explores the potentials associated with adding dimensionality and interactivity into visualization tools to facilitate complex workflows in audio information research using the Jellyfish Dynamite software.
>
---
#### [new 011] SingingSDS: A Singing-Capable Spoken Dialogue System for Conversational Roleplay Applications
- **分类: cs.SD**

- **简介: 该论文提出SingingSDS，一种能以歌唱方式回应的对话系统，解决传统语音对话系统情感表达单一的问题。通过ASR-LLM-SVS级联架构，支持角色扮演与互动娱乐中的多样化音乐风格与个性化声音配置，提升交互的情感吸引力与记忆性。**

- **链接: [https://arxiv.org/pdf/2511.20972v1](https://arxiv.org/pdf/2511.20972v1)**

> **作者:** Jionghao Han; Jiatong Shi; Masao Someki; Yuxun Tang; Lan Liu; Yiwen Zhao; Wenhao Feng; Shinji Watanabe
>
> **摘要:** With recent advances in automatic speech recognition (ASR), large language models (LLMs), and text-to-speech (TTS) technologies, spoken dialogue systems (SDS) have become widely accessible. However, most existing SDS are limited to conventional spoken responses. We present SingingSDS, a cascaded SDS that responds through singing rather than speaking, fostering more affective, memorable, and pleasurable interactions in character-based roleplay and interactive entertainment scenarios. SingingSDS employs a modular ASR-LLM-SVS pipeline and supports a wide range of configurations across character personas, ASR and LLM backends, SVS models, melody sources, and voice profiles, tailored to different needs in terms of latency, quality, and musical style. SingingSDS is available as a plug-and-play web demo, featuring modular, open-source code that supports customization and extension. Demo: https://huggingface.co/spaces/espnet/SingingSDS. Code: https://github.com/SingingSDS/SingingSDS.
>
---
#### [new 012] Evaluation of an ITD-to-ILD Transformation as a Method to Restore the Spatial Benefit in Speech Intelligibility in Hearing Impaired Listeners
- **分类: eess.AS**

- **简介: 该论文研究听力受损者在复杂环境中语音识别困难的问题，旨在通过将低频双耳时间差（ITD）转换为双耳强度差（ILD），恢复其双耳优势。实验表明，此转换能提升听力障碍者的语音理解能力，尤其对侧方声源有显著改善，具有应用于助听器和人工耳蜗的潜力。**

- **链接: [https://arxiv.org/pdf/2511.21222v1](https://arxiv.org/pdf/2511.21222v1)**

> **作者:** Timm-Jonas Bäumer; Johannes W. de Vries; Stephan Töpken; Richard C. Hendriks; Peyman Goli; Steven van de Par
>
> **备注:** 12 pages, 11 figues. Submitted to the special issue for the International Symposium on Hearing 2025 in Trends in Hearing
>
> **摘要:** To improve speech intelligibility in complex everyday situations, the human auditory system partially relies on Interaural Time Differences (ITDs) and Interaural Level Differences (ILDs). However, hearing impaired (HI) listeners often exhibit limited sensitivity to ITDs, resulting in decreased speech intelligibility performance. This study aimed to investigate whether transforming low-frequency ITDs into ILDs could reintroduce a binaural benefit for HI listeners. We conducted two experiments with HI listeners. The first experiment used binaurally phase-shifted sinusoids at different frequencies to evaluate the HI listeners ITD sensitivity threshold. All subjects had an increased ITD threshold at higher frequencies, with different ITD sensitivities between the subjects in the lower frequencies. In the second experiment, Speech Reception Thresholds (SRTs) were measured in different binaural configurations by manipulating Head-Related Transfer Functions (HRTFs). The results showed that, despite the decreased ITD sensitivity, removing ITDs decreased SRTs by approximately 1 dB compared to the unprocessed baseline, where ITDs and ILDs are available. Furthermore, substituting low-frequency ITDs with ILDs yielded an improvement for a lateral target speaker. Adding the low-frequency ILDs while preserving the ITDs caused a significant improvement for speakers in all directions. These findings suggest that the proposed transformation method could be effective in restoring binaural benefits in HI listeners. The results of this study suggest the use of such transformation techniques to be implemented in hearing aids and cochlear implants, directly benefiting HI listeners.
>
---
#### [new 013] Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension
- **分类: cs.SD**

- **简介: 该论文针对音频带宽扩展任务，提出一种基于谐波-打击成分解的神经音频编解码器。通过将音频信号分解为谐波与打击成分，并训练变压器语言模型预测高频频段，实现高质量重建。创新性地联合优化编解码结构与生成建模，提升音质表现。**

- **链接: [https://arxiv.org/pdf/2511.21580v1](https://arxiv.org/pdf/2511.21580v1)**

> **作者:** Benoît Giniès; Xiaoyu Bie; Olivier Fercoq; Gaël Richard
>
> **摘要:** Bandwidth extension, the task of reconstructing the high-frequency components of an audio signal from its low-pass counterpart, is a long-standing problem in audio processing. While traditional approaches have evolved alongside the broader trends in signal processing, recent advances in neural architectures have significantly improved performance across a wide range of audio tasks, In this work, we extend these advances by framing bandwidth extension as an audio token prediction problem. Specifically, we train a transformer-based language model on the discrete representations produced by a disentangled neural audio codec, where the disentanglement is guided by a Harmonic-Percussive decomposition of the input signals, highlighting spectral structures particularly relevant for bandwidth extension. Our approach introduces a novel codec design that explicitly accounts for the downstream token prediction task, enabling a more effective coupling between codec structure and transformer modeling. This joint design yields high-quality reconstructions of the original signal, as measured by both objective metrics and subjective evaluations. These results highlight the importance of aligning codec disentanglement and representation learning with the generative modeling stage, and demonstrate the potential of global, representation-aware design for advancing bandwidth extension.
>
---
#### [new 014] Generating Separated Singing Vocals Using a Diffusion Model Conditioned on Music Mixtures
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音乐中人声分离任务，旨在从混合音频中提取纯净人声。提出一种基于扩散模型的生成方法，以混合音频为条件生成独唱人声，提升分离质量与灵活性。通过可调节采样参数实现质量与效率的平衡，并验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.21342v1](https://arxiv.org/pdf/2511.21342v1)**

> **作者:** Genís Plaja-Roglans; Yun-Ning Hung; Xavier Serra; Igor Pereira
>
> **备注:** Accepted for publication at WASPAA 2025
>
> **摘要:** Separating the individual elements in a musical mixture is an essential process for music analysis and practice. While this is generally addressed using neural networks optimized to mask or transform the time-frequency representation of a mixture to extract the target sources, the flexibility and generalization capabilities of generative diffusion models are giving rise to a novel class of solutions for this complicated task. In this work, we explore singing voice separation from real music recordings using a diffusion model which is trained to generate the solo vocals conditioned on the corresponding mixture. Our approach improves upon prior generative systems and achieves competitive objective scores against non-generative baselines when trained with supplementary data. The iterative nature of diffusion sampling enables the user to control the quality-efficiency trade-off, and also refine the output when needed. We present an ablation study of the sampling algorithm, highlighting the effects of the user-configurable parameters.
>
---
#### [new 015] AV-Edit: Multimodal Generative Sound Effect Editing via Audio-Visual Semantic Joint Control
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出AV-Edit，解决音效编辑中依赖低层信号或粗略文本导致灵活性差的问题。通过联合视觉、音频与文本语义，利用对比音频-视觉掩码自编码器预训练，并引入相关性特征门控的多模态扩散变换器，实现精准音效编辑。构建专用视频音效数据集进行评估，显著提升音效编辑质量与一致性。**

- **链接: [https://arxiv.org/pdf/2511.21146v1](https://arxiv.org/pdf/2511.21146v1)**

> **作者:** Xinyue Guo; Xiaoran Yang; Lipan Zhang; Jianxuan Yang; Zhao Wang; Jian Luan
>
> **摘要:** Sound effect editing-modifying audio by adding, removing, or replacing elements-remains constrained by existing approaches that rely solely on low-level signal processing or coarse text prompts, often resulting in limited flexibility and suboptimal audio quality. To address this, we propose AV-Edit, a generative sound effect editing framework that enables fine-grained editing of existing audio tracks in videos by jointly leveraging visual, audio, and text semantics. Specifically, the proposed method employs a specially designed contrastive audio-visual masking autoencoder (CAV-MAE-Edit) for multimodal pre-training, learning aligned cross-modal representations. These representations are then used to train an editorial Multimodal Diffusion Transformer (MM-DiT) capable of removing visually irrelevant sounds and generating missing audio elements consistent with video content through a correlation-based feature gating training strategy. Furthermore, we construct a dedicated video-based sound editing dataset as an evaluation benchmark. Experiments demonstrate that the proposed AV-Edit generates high-quality audio with precise modifications based on visual content, achieving state-of-the-art performance in the field of sound effect editing and exhibiting strong competitiveness in the domain of audio generation.
>
---
#### [new 016] ASR Error Correction in Low-Resource Burmese with Alignment-Enhanced Transformers using Phonetic Features
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究低资源缅甸语语音识别纠错任务，针对缅甸语缺乏标注数据的难题，提出融合音标（IPA）与对齐信息的增强型Transformer模型。通过特征集成显著降低词错误率（WER）和提升字符级评分，验证了特征设计在提升低资源语音识别性能中的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.21088v1](https://arxiv.org/pdf/2511.21088v1)**

> **作者:** Ye Bhone Lin; Thura Aung; Ye Kyaw Thu; Thazin Myint Oo
>
> **备注:** 7 pages, 2 figures, 7 tables, Accepted to iSAI-NLP 2025
>
> **摘要:** This paper investigates sequence-to-sequence Transformer models for automatic speech recognition (ASR) error correction in low-resource Burmese, focusing on different feature integration strategies including IPA and alignment information. To our knowledge, this is the first study addressing ASR error correction specifically for Burmese. We evaluate five ASR backbones and show that our ASR Error Correction (AEC) approaches consistently improve word- and character-level accuracy over baseline outputs. The proposed AEC model, combining IPA and alignment features, reduced the average WER of ASR models from 51.56 to 39.82 before augmentation (and 51.56 to 43.59 after augmentation) and improving chrF++ scores from 0.5864 to 0.627, demonstrating consistent gains over the baseline ASR outputs without AEC. Our results highlight the robustness of AEC and the importance of feature design for improving ASR outputs in low-resource settings.
>
---
## 更新

#### [replaced 001] Step-Audio-R1 Technical Report
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出Step-Audio-R1，首个成功实现音频领域推理的模型。针对音频模型常因推理而性能下降的问题，通过模态锚定推理蒸馏框架，使推理链基于真实声学特征，避免幻觉。实验表明其性能超越Gemini 2.5 Pro，接近Gemini 3 Pro，证明合理推理可显著提升音频理解能力。**

- **链接: [https://arxiv.org/pdf/2511.15848v2](https://arxiv.org/pdf/2511.15848v2)**

> **作者:** Fei Tian; Xiangyu Tony Zhang; Yuxin Zhang; Haoyang Zhang; Yuxin Li; Daijiao Liu; Yayue Deng; Donghang Wu; Jun Chen; Liang Zhao; Chengyuan Yao; Hexin Liu; Eng Siong Chng; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **备注:** 22 pages, 5 figures. Technical Report
>
> **摘要:** Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question - can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities.
>
---
#### [replaced 002] Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction
- **分类: cs.LG; cs.SD**

- **简介: 该论文针对生成式AI在实时人机音乐即兴中因强化学习后训练导致的“奖励黑客”问题，提出一种对抗性后训练方法。通过引入协同进化的判别器，提升旋律伴奏的多样性与适应性，有效缓解输出单调问题，在模拟与真实交互实验中均验证了其在保持和谐性的同时增强创意流动性的优势。**

- **链接: [https://arxiv.org/pdf/2511.17879v2](https://arxiv.org/pdf/2511.17879v2)**

> **作者:** Yusong Wu; Stephen Brade; Teng Ma; Tia-Jane Fowler; Enning Yang; Berker Banar; Aaron Courville; Natasha Jaques; Cheng-Zhi Anna Huang
>
> **摘要:** Most applications of generative AI involve a sequential interaction in which a person inputs a prompt and waits for a response, and where reaction time and adaptivity are not important factors. In contrast, live jamming is a collaborative interaction that requires real-time coordination and adaptation without access to the other player's future moves, while preserving diversity to sustain a creative flow. Reinforcement learning post-training enables effective adaptation through on-policy interaction, yet it often reduces output diversity by exploiting coherence-based rewards. This collapse, known as ``reward hacking'', affects many RL post-training pipelines, but is especially harmful in live jamming, where musical creativity relies on dynamic variation and mutual responsiveness. In this paper, we propose a novel adversarial training method on policy-generated trajectories to mitigate reward hacking in RL post-training for melody-to-chord accompaniment. A co-evolving discriminator separates policy trajectories from the data distribution, while the policy maximizes the discriminator output in addition to coherence rewards to prevent collapse to trivial outputs. We evaluate accompaniment quality and output diversity in simulation with both fixed test melodies and learned melody agents, and we conduct a user study with the model deployed in a real-time interactive system with expert musicians. Quantitative evaluation and user feedback demonstrate improved output diversity, harmonic coherence, adaptation speed and user agency. Our results demonstrate a simple yet effective method to mitigate reward hacking in RL post-training of generative sequence models.
>
---
#### [replaced 003] Improved Visually Prompted Keyword Localisation in Real Low-Resource Settings
- **分类: cs.CL; cs.CV; eess.AS**

- **简介: 该论文研究视觉提示关键词定位任务，旨在无转录条件下定位语音中出现的图像所指示的词。针对低资源语言缺乏标注的问题，提出无需转录的少样本配对挖掘方法。在英语上性能略有下降，在真实低资源语言尤鲁巴语上表现仍可接受，但因配对不准确导致性能下降更明显。**

- **链接: [https://arxiv.org/pdf/2409.06013v2](https://arxiv.org/pdf/2409.06013v2)**

> **作者:** Leanne Nortje; Dan Oneata; Gabriel Pirlogeanu; Herman Kamper
>
> **备注:** Accepted at SpeD 2025
>
> **摘要:** Given an image query, visually prompted keyword localisation (VPKL) aims to find occurrences of the depicted word in a speech collection. This can be useful when transcriptions are not available for a low-resource language (e.g. if it is unwritten). Previous work showed that VPKL can be performed with a visually grounded speech model trained on paired images and unlabelled speech. But all experiments were done on English. Moreover, transcriptions were used to get positive and negative pairs for the contrastive loss. This paper introduces a few-shot learning scheme to mine pairs automatically without transcriptions. On English, this results in only a small drop in performance. We also - for the first time - consider VPKL on a real low-resource language, Yoruba. While scores are reasonable, here we see a bigger drop in performance compared to using ground truth pairs because the mining is less accurate in Yoruba.
>
---
#### [replaced 004] Spike Encoding for Environmental Sound: A Comparative Benchmark
- **分类: cs.SD; cs.ET; eess.AS**

- **简介: 该论文针对环境声音的脉冲编码问题，比较了三种编码方法（TAE、SF、MW）在多数据集上的表现。研究发现TAE在重建质量与能效上均最优，且在下游分类任务中性能最佳，为神经形态环境声处理提供了有效编码方案。**

- **链接: [https://arxiv.org/pdf/2503.11206v4](https://arxiv.org/pdf/2503.11206v4)**

> **作者:** Andres Larroza; Javier Naranjo-Alcazar; Vicent Ortiz; Maximo Cobos; Pedro Zuccarello
>
> **备注:** Under review ICASSP 2026
>
> **摘要:** Spiking Neural Networks (SNNs) offer energy efficient processing suitable for edge applications, but conventional sensor data must first be converted into spike trains for neuromorphic processing. Environmental sound, including urban soundscapes, poses challenges due to variable frequencies, background noise, and overlapping acoustic events, while most spike based audio encoding research has focused on speech. This paper analyzes three spike encoding methods, Threshold Adaptive Encoding (TAE), Step Forward (SF), and Moving Window (MW) across three datasets: ESC10, UrbanSound8K, and TAU Urban Acoustic Scenes. Our multiband analysis shows that TAE consistently outperforms SF and MW in reconstruction quality, both per frequency band and per class across datasets. Moreover, TAE yields the lowest spike firing rates, indicating superior energy efficiency. For downstream environmental sound classification with a standard SNN, TAE also achieves the best performance among the compared encoders. Overall, this work provides foundational insights and a comparative benchmark to guide the selection of spike encoders for neuromorphic environmental sound processing.
>
---
