# 音频 cs.SD;  eess.AS

- **最新发布 35 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] C2GA: A Class-Controllable Generative Augmentation Framework for Respiratory Sound Classification
- **分类: cs.SD**

- **简介: 该论文属于呼吸音分类任务，解决数据量小、噪声大和类别不平衡问题。提出C2GA框架，通过可控生成增强数据，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2606.02212](https://arxiv.org/pdf/2606.02212)**

> **作者:** Ziqi Ma; Mengyu Han; Anteng Cai; Zhanchong Liu; Bowen Feng; Hang Yu; Sheng Hu
>
> **备注:** 18 pages, 5 figures, submitted to Computer Methods and Programs in Biomedicine
>
> **摘要:** Background: Respiratory sound classification plays a critical role in the clinical identification of pulmonary pathologies. However, its performance is often hindered by the limited size, severe noise, and class imbalance of real-world auscultation datasets. Although conventional audio augmentation techniques are easy to implement, they may inadvertently distort subtle pathological characteristics. Meanwhile, existing Variational Autoencoder (VAE)- or Generative Adversarial Network (GAN)-based generative approaches often suffer from limited sample fidelity and insufficient controllability over class semantics, particularly under conditions of scarce supervision. Methods: To overcome these limitations, we propose C2GA, a class-controllable generative augmentation framework. C2GA first constructs a semantically rich discrete latent space using a conditional Vector-Quantized Variational Autoencoder (VQ-VAE), in which local acoustic tokens are explicitly decoupled from global class prototypes. Subsequently, a Transformer-based autoregressive prior is trained to generate label-consistent token sequences. These generated tokens are then fused with the corresponding class prototypes and decoded into high-fidelity Mel-spectrograms for data augmentation. Conclusion: These results indicate that C2GA provides an effective and semantically reliable augmentation strategy for respiratory sound analysis. By enabling controllable and high-quality data generation, the proposed framework offers a promising solution for improving the robustness and generalization of respiratory sound classification in realistic clinical scenarios.
>
---
#### [new 002] MOSS-Audio Technical Report
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出MOSS-Audio，一个统一的音频-语言模型，解决音频理解任务，通过深度特征注入和时间标记提升音频定位与推理能力。**

- **链接: [https://arxiv.org/pdf/2606.01802](https://arxiv.org/pdf/2606.01802)**

> **作者:** Chen Yang; Chufan Yu; Hanfu Chen; Jie Zhu; Jingqi Chen; Ke Chen; Wenxuan Wang; Yang Wang; Yaozhou Jiang; Yi Jiang; Zhengyuan Lin; Ziqi Chen; Zhaoye Fei; Chenghao Liu; Jun Zhan; Kang Yu; Kexin Huang; Mingshu Chen; Qinyuan Cheng; Ruixiao Li; Shimin Li; Songlin Wang; Yang Gao; Yiyang Zhang; Xipeng Qiu
>
> **摘要:** MOSS-Audio is a unified audio-language model for speech, environmental sound, and music understanding, supporting audio captioning, time-aware question answering, timestamped transcription, and audio-grounded reasoning. MOSS-Audio couples a dedicated audio encoder with a modality adapter and a large language model: the encoder produces 12.5 Hz temporal representations, the adapter projects them into the decoder space, and the decoder generates autoregressive text outputs. Two design choices are central to the system: \textbf{DeepStack cross-layer feature injection}, which exposes the decoder to acoustic information from multiple encoder depths, and \textbf{time markers}, which provide explicit temporal cues by inserting timestamp markers into the audio-token stream. At the data level, we design an event-preserving audio annotation pipeline that segments raw audio at coherent event boundaries, applies branch-specific annotation to speech, music, and general audio, and merges the results into unified captions for pretraining. The intermediate branch-specific captions are further retained to support the construction of task-oriented SFT data. The model is pretrained on large-scale audio-language data, with time-aware objectives incorporated to support temporal grounding, and then undergoes multi-stage post-training to enhance instruction following and audio-grounded reasoning. We release 4B and 8B variants in both Instruct and Thinking configurations. MOSS-Audio achieves strong performance across general audio understanding, speech captioning, ASR, and timestamped ASR, positioning it as a promising understanding foundation for future voice agents.
>
---
#### [new 003] Parameter-efficient Dual-encoder Architecture with Differentiable Choquet Integral Fusion for Underwater Acoustic Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于水下声学分类任务，解决复杂声学环境下的特征融合问题。提出双编码器架构与可微Choquet积分融合方法，提升分类精度并减少参数量。**

- **链接: [https://arxiv.org/pdf/2606.02341](https://arxiv.org/pdf/2606.02341)**

> **作者:** Amirmohammad Mohammadi; Joshua Peeples; Alexandra Van Dine
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Underwater acoustic classification has a wide array of oceanic applications, but faces challenges due to an increasingly complex acoustic environment. Waveform and spectrogram representations have been primarily used as acoustic data features for classification tasks in this domain. Spectrograms model harmonic dependencies, but these reduced representations can filter out acoustic features relevant for discrimination. While phase information from the waveform allows full characterization of the signal, the original waveform can be noisy and complex, rendering this representation difficult for models to process directly. This paper proposes a dual-encoder neural architecture to simultaneously process acoustic waveforms and spectrograms, leveraging pre-trained backbones and parameter-efficient fine-tuning modules, enabling a domain adaptation. To combine these adapted branches, a novel differentiable fuzzy aggregation mechanism based on the Choquet integral is introduced to balance the temporal and spectral representations. This fusion strategy not only yields higher classification accuracy but also provides interpretability. Specifically, by analyzing the learned fuzzy measures, insights are revealed about class-specific shifts in the network's representation reliance. By dynamically shifting attention to the representation least corrupted by potential asymmetric channel distortions, the proposed gating mechanism mitigates the non-stationary challenges of the underwater environment. Evaluations on the DeepShip and ShipsEar datasets demonstrate that the proposed architecture achieves classification improvements over independent single-encoder baselines, while simultaneously restricting the trainable parameter space. This mitigates the risk of overfitting on limited acoustic datasets while alleviating the computational costs associated with fully fine-tuning foundation models.
>
---
#### [new 004] Advancing Electrolaryngeal Speech Enhancement Through Speech-Text Representation Learning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在提升电声喉语音的自然度和可懂度。通过融合语音与文本表示，改进序列到序列语音转换模型，解决EL语音质量差的问题。**

- **链接: [https://arxiv.org/pdf/2606.01905](https://arxiv.org/pdf/2606.01905)**

> **作者:** Ding Ma; Jinyi Mi; Fengji Li; Lester Phillip Violeta; Jiajun He; Wenchin Huang; Kazuhiro Kobayashi; Tomoki Toda
>
> **备注:** 15 pages, 7 figures. Accepted to IEEE TBME
>
> **摘要:** Objective: laryngectomees depend on an electromechanical device to generate electrolaryngeal (EL) speech. Compared with normal speech, EL speech suffers from severe distortion, limited phonetic variation, unnatural prosody, and temporal shifts, degrading naturalness and intelligibility. Although sequence-to-sequence (seq2seq) voice conversion (VC) based EL-speech-to-normal-speech conversion (EL2SP) is promising, substantial mismatches between EL and normal speech inevitably cause cumulative mapping errors that limit performance. To address this, we describe a novel representation learning framework integrating speech and text representations to improve mapping and reconstruction quality within a seq2seq VC model. Methods: our methodology comprises two main stages: 1) representation integration and learning, and 2) reconstruction training. A network capable of incorporating auxiliary text information is first constructed with pretrained modules to learn speech--text-based integrated representations. Then, an autoencoder-style reconstruction strategy finalizes EL2SP model to inherit these representations without increasing model complexity. We introduce three fusion strategies including middle-, input-, and hybrid-level fusion strategies that progressively enhance learning. Moreover, besides standard seq2seq VC objectives, an additional reconstruction loss on the integrated representation is introduced to refine representation transfer. Results: experiments under different EL2SP datasets consistently demonstrate that our methods, combined with data augmentations, outperform baselines relying solely on speech representations. Furthermore, progressive improvements with system design depth validate the effectiveness of our methods. Significance: the proposed methods provide an extensible and practical methodology for EL speech enhancement and assistive communication technologies.
>
---
#### [new 005] SoulX-Transcriber: A Robust End-to-End Framework for Multi-Speaker Speech Transcription
- **分类: eess.AS**

- **简介: 该论文属于多说话人语音转录任务，解决语音相似、快速切换和重叠等问题。提出SoulX-Transcriber框架，结合语音识别与说话人辨识，提升转录准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.02400](https://arxiv.org/pdf/2606.02400)**

> **作者:** Yuhang Dai; Haopeng Lin; Zhennan Lin; Jiale Qian; Jun Wu; Hanke Xie; Hao Meng; Hanlin Wen; Chuang Ding; Shunshun Yin; Ming Tao; Lei Xie; Xinsheng Wang
>
> **备注:** 10 pages, 4 figures, 3tables
>
> **摘要:** Recent advances in Automatic Speech Recognition (ASR) and Large Language Models (LLMs) have significantly improved speech understanding capabilities. However, multi-speaker speech transcription remains challenging task, constrained by highly similar speaker voices, rapid turn-taking transitions, overlapping utterances and inaccurate speaker boundary segmentation. These challenges become particularly pronounced in real-world conversational audio, where speaker dynamics and acoustic conditions are highly variable. This technical report presents SoulX-Transcriber, a unified multi-speaker transcription system that jointly models speaker diarization (SD) and ASR within an LLM-based framework. SoulX-Transcriber adopts a two-stage training strategy to improve both speaker discrimination and transcription robustness. In the first stage, speaker-aware multi-task continuous pre-training enhances speaker representation learning and boundary perception. In the second stage, supervised fine-tuning further optimizes the model for accurate end-to-end speaker-attributed transcription under complex multi-speaker conditions. SoulX-Transcriber delivers strong performance and robustness across multiple public benchmarks, including AliMeeting, AISHELL-4, and AMI, while maintaining high adaptability to multi-domain scenarios.
>
---
#### [new 006] JenBridge: Adaptive Long-Form Video Soundtracking across Scene Transitions
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文属于视频配乐任务，解决长视频在场景切换时音乐连贯性问题。提出JenBridge框架，结合Transformer和LLM实现自然过渡与叙事一致性。**

- **链接: [https://arxiv.org/pdf/2606.01703](https://arxiv.org/pdf/2606.01703)**

> **作者:** Jiashuo Yu; Yao Yao; Boyu Chen; Alex Wang
>
> **摘要:** We address the challenge of generating high-fidelity, long-form soundtracks that remain coherent across scene transitions. Existing AI music systems are mainly designed for short, isolated clips and lack mechanisms to ensure narrative continuity. We present JenBridge, a modular and interpretable framework for adaptive long-form video soundtracking that ensures both high-fidelity audio generation and transition naturalness. The core architecture is a Transformer-based generative model trained with a flow-matching objective, following a two-stage paradigm: pretraining on large-scale text-audio corpora to establish robust musical priors, then adapting to the video domain with dual text-visual conditioning for precise cross-modal alignment. Crucially, to achieve long-form coherence across diverse scene changes, JenBridge incorporates a novel adaptive transition mechanism. This system features a versatile toolkit of transition styles, including a generative transition method, and uniquely employs a Large Language Model (LLM) Agent that acts as a director to select the most appropriate transition for each narrative shift intelligently. To rigorously assess this task, we propose the LVS Benchmark, a new benchmark that includes a curated dataset and novel evaluation metrics focusing on holistic and transition-aware assessment. Extensive experiments on the proposed benchmark demonstrate that JenBridge significantly outperforms existing methods in both objective and subjective metrics, particularly in terms of transition naturalness and overall narrative coherence. JenBridge represents a significant step towards fully automated, professional-quality video soundtracking.
>
---
#### [new 007] SiamCTC: Learning Speech Representations through Monotonic Temporal Alignment
- **分类: eess.AS**

- **简介: 论文提出SiamCTC，结合Siamese网络与CTC，解决语音表示学习中的帧级对齐问题，提升对语速变化的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.02220](https://arxiv.org/pdf/2606.02220)**

> **作者:** SooHwan Eom; Mark Hasegawa-Johnson; ad Chang D. Yoo
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Self-supervised speech representation learning has made significant progress through Siamese networks, which leverage different views of the same input. However, existing methods often require frame-wise alignment between these views, overlooking the broader linguistic context invariance across different speaking styles. We introduce SiamCTC, a framework that integrates Siamese networks with Connectionist Temporal Classification (CTC) to learn speech representations without strict frame-level correspondence. By employing CTC loss to establish flexible, monotonic alignments between differing temporal realizations of the same content, SiamCTC accommodates speed perturbations and other temporal augmentations. This design relaxes frame-wise constraints while preserving temporal coherence and enhancing robustness to speaking-rate variations in downstream tasks. Our experiments demonstrate that SiamCTC leads to more adaptable speech representations, particularly at diverse speaking rates.
>
---
#### [new 008] Quality Audio Prototyping: a prototype system for unified sound retrieval and procedural generation
- **分类: cs.SD; cs.HC; cs.LG; eess.AS**

- **简介: 该论文提出QuAP系统，解决声音设计中检索与生成分离的问题，通过统一接口实现内容检索与程序生成，提升创作效率。**

- **链接: [https://arxiv.org/pdf/2606.00629](https://arxiv.org/pdf/2606.00629)**

> **作者:** Nelly Garcia; Aditya Bhattacharjee; Gabryel Mason-Williams; Israel Mason-Williams; Emmanouil Benetos; Joshua Reiss
>
> **备注:** DaFx 2026
>
> **摘要:** Sound design workflows frequently oscillate between time-consuming library searches and the complexity of procedural synthesis, with practitioners typically relying on disconnected tools to address each challenge separately. This paper introduces Quality Audio Prototyping (QuAP), a working prototype that unifies content-based audio retrieval and procedural sound generation within a single interface, reducing the procedural distance between a narrative concept and its sonic realisation. QuAP integrates a similarity-based retrieval engine with real-time procedural audio models, complemented by a rule-based assistant that provides perceptually informed parameter guidance, offering definitions and recommendations derived from empirical optimisation rather than requiring prior synthesis knowledge. Preliminary evaluation confirms the viability of this approach: subjective assessment demonstrated statistically significant quality improvements in five of six embedded synthesis models, and an encoder ablation study established the preferred retrieval architecture on a sound effect dataset. A user evaluation with 16 practitioners confirmed the tool's workflow utility, with all participants agreeing that the parameter assistant preserved creative agency while lowering the barrier to procedural interaction.
>
---
#### [new 009] Description and Discussion on DCASE 2026 Challenge Task 2: Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring
- **分类: eess.AS; cs.SD**

- **简介: 该论文介绍DCASE 2026挑战任务2，旨在解决机器状态监测中的噪声感知无监督异常声音检测问题，通过双麦克风数据提升噪声环境下的检测性能。**

- **链接: [https://arxiv.org/pdf/2606.01578](https://arxiv.org/pdf/2606.01578)**

> **作者:** Tomoya Nishida; Noboru Harada; Daiki Takeuchi; Daisuke Niizumi; Keisuke Imoto; Kota Dohi; Harsh Purohit; Takashi Endo; Yohei Kawaguchi
>
> **备注:** this article draws heavily from arXiv:2506.10097
>
> **摘要:** This paper presents an overview of DCASE 2026 Challenge Task 2, titled "Noise-aware unsupervised anomalous sound detection (UASD) for machine condition monitoring." The task aims to advance noise-robust anomalous sound detection for machine condition monitoring under the unsupervised setting, where only normal machine sounds are available for training. Reliable detection under noisy conditions is crucial for practical deployment, but previous DCASE Task 2 settings provided limited information about environmental noise, potentially limiting UASD performance in highly noisy situations. To address this limitation, DCASE 2026 allows participants to exploit two-channel audio samples simultaneously captured at locations near and far from the target machine. Since the distant microphone is expected to contain relatively stronger environmental noise and weaker direct machine sounds, it may help distinguish environmental noise components from the target machine sounds. After the challenge submission deadline, challenge results and an analysis of the submitted systems will be added.
>
---
#### [new 010] MelT: GEMM-Native NDFT for Efficient Single-Stage Audio Frontends on Modern Accelerators
- **分类: cs.SD**

- **简介: 该论文提出MelT，解决音频前端处理效率问题，通过GEMM-native NDFT替代传统STFT+Mel，提升计算效率并降低能耗。**

- **链接: [https://arxiv.org/pdf/2606.01009](https://arxiv.org/pdf/2606.01009)**

> **作者:** Augusto Camargo; Marcelo Finger
>
> **摘要:** Modern audio processing networks are commonly deployed on accelerators whose peak throughput is obtained through dense linear algebra, whereas conventional acoustic frontends -- a Short-Time Fourier Transform (STFT) followed by sparse Mel aggregation -- remain structurally heterogeneous. This mismatch can introduce memory-bandwidth, dispatch, and intermediate-allocation overheads on contemporary accelerator backends. This work introduces MelT, a single-stage frontend framework in which Mel-spaced Non-Uniform Discrete Fourier Transform (NDFT) bases are precomputed and applied to time-domain acoustic frames through dense General Matrix Multiplication (GEMM) operations. The contribution is not the NDFT operator itself; rather, it is the formulation of Mel-spaced NDFT projection as a GEMM-native audio frontend and its evaluation as a hardware-efficient alternative to conventional STFT+Mel pipelines. Evaluated across platforms ranging from Apple A18 Pro edge hardware to NVIDIA H100 datacenter acceleration, MelT attains up to a $3.75\times$ speedup in inference latency and a $3.52\times$ reduction in energy consumption while maintaining downstream classification accuracy.
>
---
#### [new 011] A Lightweight Slot-Attention Framework for Multi-Instrument Multi-Pitch Estimation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多乐器音高估计任务，旨在区分混合音频中的音高及对应乐器。提出轻量级槽注意力框架，通过匈牙利匹配提升乐器分解效果。**

- **链接: [https://arxiv.org/pdf/2606.01460](https://arxiv.org/pdf/2606.01460)**

> **作者:** Michael Taenzer
>
> **备注:** Preprint submitted to the IEEE 28th International Workshop on Multimedia Signal Processing (MMSP). This work has been submitted to the IEEE for possible publication. 6 pages, 2 figures
>
> **摘要:** Multi-pitch estimation (MPE) typically predicts which pitches are active in a mixture, but not which instrument or source produced them. This paper investigates a lightweight slot-attention framework for multi-instrument MPE (MI-MPE), where a mixture CQT is mapped to an unordered set of source-like pitch maps. The model uses permutation-invariant Hungarian matching to avoid fixed output semantics and treats the number of slots as an upper bound on the number of active sources. We further study two modular extensions: a self-supervised timbre encoder that provides training-time targets for slot-level timbre embeddings, and a polyphony branch that regularizes the pitch density of mixture- and slot-level predictions. Experiments show that Hungarian matching substantially improves instrument family decomposition on URMP. Stem-level prediction remains more challenging: timbre and polyphony supervision improve selected configurations, but do not consistently resolve source assignment. The results suggest that slot-based architectures are a promising direction for source-aware MPE, while highlighting the need to couple auxiliary musical cues to slot identity more carefully.
>
---
#### [new 012] SpeechEditBench: A Bilingual Multi-Attribute Benchmark for Instruction-Guided Speech Editing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音编辑任务，旨在解决指令引导下多属性语音修改的评估问题。提出SpeechEditBench基准，涵盖多种编辑任务，并设计评估协议以衡量编辑效果与未目标属性的保持。**

- **链接: [https://arxiv.org/pdf/2606.01804](https://arxiv.org/pdf/2606.01804)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Linqi Song
>
> **摘要:** Instruction-guided speech editing requires a model to modify specified speech attributes while preserving unrelated characteristics. Despite rapid progress in Speech Large Language Models (Speech LLMs), systematic evaluation of this capability remains challenging, as existing benchmarks are fragmented across isolated editing tasks. To bridge this gap, we introduce \textbf{SpeechEditBench}, a bilingual multi-attribute benchmark for instruction-guided speech editing. SpeechEditBench encompasses seven atomic editing tasks, as well as compositional editing tasks that integrate multiple operations within a single instruction. We propose an anchor-based evaluation protocol that separately assesses the edit success of target attributes and the preservation of untargeted attributes, leading to three metrics: target success, preservation success, and joint success. Using this benchmark, we evaluate mainstream Speech LLMs and specialized speech editing systems. The results reveal three key findings: (1) no single model performs well across all editing dimensions; (2) closed-source Speech LLMs generally outperform open-source models; (3) compositional editing remains highly challenging, with even the most advanced models struggling to achieve high joint success. SpeechEditBench provides a rigorous diagnostic framework to identify bottlenecks in Speech LLMs, thereby facilitating the development of next-generation Speech LLMs with more robust and precise instruction-guided editing capabilities. Data and code will be released upon acceptance.
>
---
#### [new 013] Breaking the Pair: Evaluating Dyadic Interaction via Speaker Switching
- **分类: eess.AS**

- **简介: 该论文属于对话建模任务，旨在解决如何区分真实对话与随机替换对话的问题。提出DDM和说话人切换测试，验证对话交互结构的有效性。**

- **链接: [https://arxiv.org/pdf/2606.02185](https://arxiv.org/pdf/2606.02185)**

> **作者:** Nishchay Nilabh; Neeraj Kumar Sharma
>
> **摘要:** Speakers in dialogue continuously adapt their communicative behavior across acoustic, lexical, and semantic dimensions, a phenomenon known as conversational entrainment. Modeling this process requires representations that capture the global structure of interaction, yet prior approaches fail to disentangle dyad-specific patterns from speaker-specific traits, limiting their ability to capture true conversational adaptation. We address this with the Dyadic Distance Matrix (DDM), which encodes all pairwise similarities between the turns of two speakers over an entire conversation, capturing long-range cross-speaker dependencies. This raises a key question: does the DDM represent genuine interaction, or merely reflect individual speaker characteristics? We propose the speaker-switch test, a principled control in which one speaker's turns are replaced with those from an unrelated speaker drawn from a different conversation. This preserves turn-level statistics while disrupting the original dyadic coadaptation. The ability to distinguish real from switched DDMs thus directly evaluates whether the representation encodes interaction-specific structure. Across four embedding types and classifiers including ResNet-50 on the CANDOR corpus, real DDMs are consistently distinguishable from their switched counterparts. Comparisons with LibriSpeech show higher discriminability in read speech, highlighting the role of prosodic variability in naturalistic conversations. GradCAM analysis further reveals distinct structural signatures driving classification. These results establish the speaker-switch test as a robust diagnostic for validating representations of dyadic conversational interaction.
>
---
#### [new 014] DUET: Unified Dual-Space Emotion Control for Diffusion and Flow-Matching Driven Text-to-Speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决情感控制不足的问题。通过构建DUET框架，实现对预训练模型的情感精细调控。**

- **链接: [https://arxiv.org/pdf/2606.00066](https://arxiv.org/pdf/2606.00066)**

> **作者:** Xu Zhang; Longbing Cao; Zhangkai Wu
>
> **摘要:** Diffusion and flow-matching based text-to-speech (TTS) models excel in naturalness but often lack explicit emotion control, as emotional signals remain entangled with speaker identity. We discover that emotion embedding emerges as a linearly decodable direction of frozen hidden states, nearly orthogonal to the direction embedding speaker identity. This inspires a plug-and-play framework DUET for emotion control over pretrained diffusion and flow-matching based TTS models. During generation, DUET unifies dual-space control to achieve fine-grained emotion intervention in a single per-step update: hidden space steering shifts generation along the target emotion direction, while mel-space guidance refines spectral details through gradients backpropagated from a differentiable vocoder. We validate DUET on five architecturally diverse pretrained TTS backbones across three datasets, where it outperforms 10 supervised state-of-the-art emotional TTS baselines across paradigms and achieves the highest human-rated emotion appropriateness. To further showcase its qualitative behavior, we deploy DUET on an Ameca humanoid robot, where it produces richly expressive emotional speech on the humanoid, demonstrating the strong potential for plug-and-play affective interaction for embodied agents.
>
---
#### [new 015] Sympatheia: Emotionally Adaptive Voice Assistant with Continuous Affect Conditioning
- **分类: cs.SD; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文提出Sympatheia，一个情感自适应语音助手，解决如何根据用户情绪生成恰当回应的问题。通过合成数据集和连续情绪控制信号，提升对话的情感适应性。**

- **链接: [https://arxiv.org/pdf/2606.00851](https://arxiv.org/pdf/2606.00851)**

> **作者:** Sukru Samet Dindar; Riki Shimizu; Xilin Jiang; Nima Mesgarani
>
> **摘要:** Empathetic spoken dialogue systems must infer a user's emotional state to respond appropriately, yet everyday speech often carries weak, neutral, or ambiguous affective cues. To address this, we introduce Sympatheia, a speech-to-speech dialogue framework conditioned on affect inferred from the user's speech and, when available, explicit affect specifications provided as a continuous valence--arousal (VA) control signal by a multimodal sensing module or user interface. To train our model, we construct Sympatheia-18k, an emotion-conditioned synthetic spoken dialogue corpus with 12 emotion anchors. This dataset includes an emotional split for learning affective speech behavior, and a neutral split that pairs emotionally neutral queries with multiple emotion-conditioned responses to isolate explicit emotion control in emotionally ambiguous cases. Empirical results show that Sympatheia outperforms speech conversational baselines in generating responses whose semantic content and spoken delivery are both emotionally appropriate. We further show that the same VA interface can integrate emotion estimates from diverse sensing modules, including facial expression, biosignals, and textual affect descriptions, improving response alignment when speech alone provides limited emotional evidence. These results suggest that continuous affect conditioning is an effective practical step for building emotionally adaptive voice assistants.
>
---
#### [new 016] Exploiting Noise Inseparability for Weakly-Supervised Discriminative Speech Denoising Using Noisy Targets
- **分类: eess.AS**

- **简介: 该论文属于语音去噪任务，解决真实噪声环境下缺乏清洁语音标注的问题。通过估计人工噪声并利用其消除残留噪声，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.02327](https://arxiv.org/pdf/2606.02327)**

> **作者:** Matthew Maciejewski; Samuele Cornell
>
> **备注:** Submitted to IWAENC 2026
>
> **摘要:** Speech denoising is an often necessary step not only for human listening, but also for downstream processing by systems lacking robustness to noisy, real-world acoustic conditions. Unfortunately, denoising is a problem where conventional in-domain supervised training is not trivial, as the training targets cannot be annotated by humans: producing a clean version of a naturally-noisy speech recording is itself the task to solve. Supervised training is typically performed through the artificial addition of noise to clean speech recordings, which can only be sourced from controlled domains, a significant limitation due to the poor out-of-domain generalization of neural networks. An alternative is noisy target training (NyTT), which simply replaces the clean speech with in-domain noisy recordings, with the hope that learning to remove the artificial noise will extend to the natural. Though having shown promising results, NyTT's training objective is not minimized by clean speech estimates. We show that by estimating the artificial noise in addition to the naturally-noisy speech, the undesirable optimum can actually be exploited: the residual noise in the speech estimate can be canceled by the noise estimate via simple subtraction. Crucially, the optimum is fully compatible with conventional artificial mixtures, enabling joint training using both types of data with consistent optimization targets, opening the door to improved domain adaptability. The effectiveness of our approach is demonstrated through WHAM! and CHiME-3-based benchmarks.
>
---
#### [new 017] Context-aware child-directed speech detection from long-form recordings
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于儿童导向语音检测任务，旨在区分长录音中的儿童与成人导向语音。通过多语言数据和上下文建模提升检测效果，验证了模型在实际流程中的有效性。**

- **链接: [https://arxiv.org/pdf/2606.01134](https://arxiv.org/pdf/2606.01134)**

> **作者:** Théo Charlot; Tarek Kunze; Kaveri K. Sheth; Alejandrina Cristia; Marvin Lavechin
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Automatically distinguishing child-directed speech from adult-directed speech in long-form recordings is key to scalable analyses of children's language environments. Existing approaches process utterances in isolation and have been evaluated primarily on English. We address these gaps along three dimensions. First, we fine-tune and evaluate six-self supervised models on a multilingual dataset of 182 children, showing that in-domain pre-training on child-centered recordings substantially outperforms models trained on adult speech. Second, we demonstrate that incorporating surrounding context substantially improves classification, with an absolute gain of 13.8% in average F1-score. Third, we evaluate our model in a realistic end-to-end pipeline, from adult speech detection to addressee classification, showing that performance drops under automatic segmentation but still consistently outperforms a rule-based baseline.
>
---
#### [new 018] Beyond the Mouth: Upper-Face Affective Cues in Audiovisual Sentence Recognition under Acoustic Uncertainty
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频视觉句子识别任务，旨在解决在噪声环境下如何利用上脸情感线索提升识别效果。通过对比不同视觉特征组合的模型表现，验证了上脸信息对系统鲁棒性和置信度估计的贡献。**

- **链接: [https://arxiv.org/pdf/2606.00670](https://arxiv.org/pdf/2606.00670)**

> **作者:** Zhou Yang; Yueyi Yang
>
> **摘要:** Face-to-face speech comprehension is inherently multimodal, integrating acoustic signals with visible articulation, facial expression, head motion, and other socially relevant cues. While audiovisual speech systems typically focus on the mouth region as the primary visual source of linguistic information, affective facial expressions are often treated separately as emotion-recognition targets. This paper investigates whether upper-face affective information contributes to audiovisual sentence recognition beyond audio and mouth-region cues, particularly under acoustic degradation. Using the CREMA-D audiovisual emotional speech corpus, we train feature-based sentence classifiers under four cue conditions: audio only (A), audio plus mouth/lower-face features (A+M), audio plus upper-face features (A+U), and audio plus both mouth and upper-face features (A+M+U). Models are evaluated on clean audio and pink-noise conditions at +10 dB, +5 dB, and 0 dB SNR using actor-independent splits. Results show that mouth/lower-face features provide substantial robustness benefits under degraded audio. At 0 dB SNR, A+M improves accuracy over A by 0.0794, with an actor-bootstrap 95% confidence interval of [0.0296, 0.1298]. Upper-face affective cues exhibit a more nuanced effect. Although the direct accuracy gain of A+M+U over A+M is small, full-face models consistently improve calibration across SNR levels and outperform shuffled upper-face controls under noisy conditions. These findings suggest that affective facial information may support multimodal robustness and confidence estimation under acoustic uncertainty without directly encoding lexical content. More broadly, the study highlights the potential role of socially expressive facial cues in human-centered audiovisual interaction systems.
>
---
#### [new 019] Localizing broadband noise sources using the Loève spectrum and a 2.5D approach
- **分类: eess.AS; cs.SD**

- **简介: 论文属于声源定位任务，解决运动宽带噪声源的定位问题。通过改进2.5D模型和利用Loève谱，提出一种无需信号预处理的新方法。**

- **链接: [https://arxiv.org/pdf/2606.02127](https://arxiv.org/pdf/2606.02127)**

> **作者:** Christian H. Kasess; Wolfgang Kreuzer; Holger Waubke
>
> **备注:** 31 pages, 13 figures
>
> **摘要:** The localization of moving sound sources using a microphone array is typically based on modifying the signal to compensate for the Doppler effect. In the time domain this compensation is done on a sample-by-sample basis. In the frequency domain short time segments need to be used in which the Doppler effect is assumed to be approximately constant and a discrete Fourier transform is done on each segment. In contrast, the authors developed an inverse 2.5D localization method for uniformly moving single-frequency sources that works in the spectral domain and allows for the use of longer windows. This was achieved by modifying the 2.5D forward model to directly compute the effect of the motion in the static observer position. The method does neither require to modify the measured signal nor does it require quasi-stationary of the measurements within the window used. Unfortunately, this approach is not directly suitable for broad-band stochastic sources, and in the present work we will investigate how the statistical properties of a uniformly moving stochastic source change when observed at a static observer. Using a 2.5D setting, the relation between the power spectral density of the moving source and the Loève spectrum, which is a generalization of the cross-spectral density at the static receivers, was derived. Based on simulated data with speeds up to 100 m\,s$^{-1}$, the work presented here provides a proof of concept for a method based on multi-taper estimates for the Loève spectrum to localize moving broad-band stochastic sources . Currently, the method requires a stationary source signal and that the spectral density is flat within a certain range around the frequency of interest. Also, correlations between sources are currently not considered.
>
---
#### [new 020] Local Diagnostics of Continuous Normalizing Flow for Out-of-Distribution Detection
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于异常检测任务，解决高维数据中分布外样本的检测问题。通过连续归一化流构建子流框架，提出几何诊断信号以提升误读检测效果。**

- **链接: [https://arxiv.org/pdf/2606.00684](https://arxiv.org/pdf/2606.00684)**

> **作者:** Xinwei Cao; Mengxuan Lu; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** We address the problem of out-of-distribution (OOD) detection for target observations embedded in a subspace of the high dimensional data space. Using continuous normalizing flows (CNFs), we propose a Lagrangian sub-flow (LSF) framework designed to isolate and estimate the density for the relevant components in the representation and using the remaining components as context. Through experimentation with models for speech synthesis, we show that CNFs, similarly to other deep generative models (DGMs), are susceptible to the "likelihood paradox", where high likelihood is erroneously assigned to OOD samples. This is attributed to the inductive bias of DGMs that prioritize low-level structural details over high-level semantic coherence. To mitigate this phenomenon, we propose a number of geometric diagnostic signals based on the velocity field over the sub-flow trajectory. Based on these signals, we design metrics for the challenging task of zero-shot phoneme-level mispronunciation detection. Finally, we demonstrate the superiority of these metrics compared to likelihood-based methods on a real-world mispronunciation detection benchmark.
>
---
#### [new 021] Privacy-preserving Prosody Representation Learning
- **分类: eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决语音中身份信息泄露的隐私问题。通过自监督学习方法，提取与说话人无关的韵律表示，提升隐私保护同时保持语音任务性能。**

- **链接: [https://arxiv.org/pdf/2606.00407](https://arxiv.org/pdf/2606.00407)**

> **作者:** Kevin Everson; Mari Ostendorf
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Speech representations that capture prosodic information can be useful for both understanding and generation. However, speaker characteristics are reflected in acoustic-prosodic features (e.g., pitch). To address privacy concerns from the leakage of identity information, we propose a new self-supervised approach to learning prosody representations that incorporates speaker disentanglement strategies. We evaluate our encoder on three tasks to probe representation capabilities, including pitch reconstruction and detection of different prosodic events. Our encoder outperforms raw prosody and HuBERT-base baselines, achieving strong speaker disentanglement without adverse impact on prosody-related downstream tasks.
>
---
#### [new 022] RRP-Voice: A Longitudinal Dataset and Benchmark for Recurrent Respiratory Papillomatosis Detection
- **分类: eess.AS**

- **简介: 该论文属于罕见喉部疾病检测任务，旨在解决RRP数据稀缺与长期监测难题。构建了首个RRP纵向语音数据集，并建立多模型基准进行评估。**

- **链接: [https://arxiv.org/pdf/2606.01639](https://arxiv.org/pdf/2606.01639)**

> **作者:** Wenze Ren; Ke-Han Lu; Kai-Wei Chang; Tiantian Feng; Ching Fang; Zhi-Chi Liao; Dao Thi Hai Yen; Syu-Siang Wang; Yu Tsao; Chi-Te Wang; Shih-Hau Fang
>
> **备注:** Submitted to APSIPA ASC 2026 Special Tracks
>
> **摘要:** Deep learning has advanced pathological voice detection rapidly, yet rare laryngeal diseases remain underexplored due to data scarcity. Recurrent Respiratory Papillomatosis (RRP) exemplifies this gap: an HPV-induced disease of the larynx in which patients oscillate between recurrence and post-surgical remission over the years. RRP demands continuous voice monitoring that existing cross-sectional corpora cannot support. We introduce the first longitudinal voice dataset for RRP, comprising recordings from 26 patients with up to ten years of follow-up. Each session pairs sustained vowels with sentence-level utterances, which are annotated by otolaryngologists and confirmed synchronously with laryngoscopy. Building on this resource, we establish a systematic benchmark spanning handcrafted features, end-to-end deep networks, self-supervised pretrained models, and recent audio large language models, all evaluated under session-level cross-validation with patient-level audit. Per-subject longitudinal analyses further confirm that the cross-sectional discriminative signal reflects laryngoscopic disease state rather than stable speaker attributes. This work lays a foundation for rare longitudinal pathological voice tasks in low-resource clinical settings.
>
---
#### [new 023] Kinship Verification Using Voice
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，解决生物亲属关系验证问题。通过分析语音数据，提出新评估协议，探索说话人验证与亲属验证的关系，并测试多种模型性能。**

- **链接: [https://arxiv.org/pdf/2606.01704](https://arxiv.org/pdf/2606.01704)**

> **作者:** Jagabandhu Mishra; Tomi H. Kinnunen
>
> **备注:** Submited to IEEE TASLP
>
> **摘要:** Kinship verification (KV) from voice, the task of determining whether two speakers are biologically related, has received only little attention. Our work establishes a foundational basis for this emerging frontier, contributing to both performance evaluation and detection methodologies. First, leveraging the speech recordings of the large-scale audio-visual dataset, KAN-AV, we propose a revised evaluation protocol that controls for various confounders and adopts a family-disjoint train--test split to address open-set KV. Second, we analyze the close connection between speaker verification and KV, showing that genealogical similarity of speaker pairs plays opposite roles in the two tasks. Third, we tackle KV using three neural speaker embedding extractors (ECAPA-TDNN, WavLM-ECAPA, and ReDimNet) combined with various back-ends. In zero-shot KV including same-speaker target trials, ReDimNet achieves the lowest equal error rate (EER) of $20.8\%$; however, performance degrades to $39.7\%$ under strict kin trials, where same-speaker target trials are excluded. Our best trainable back-end, which applies asymmetric processing of the embedding pair to mitigate age-difference effects, obtains an EER of $32.0\%$ ($18.6\%$ with speaker target trials included). These results highlight the difficulty of KV while showing that speaker embeddings encode familial cues, offering a promising foundation for voice-based kinship analysis.
>
---
#### [new 024] Domain-Agnostic Incremental Learning for Sound Classification. A DCASE 2026 Challenge task
- **分类: eess.AS**

- **简介: 该论文属于DCASE 2026挑战任务，旨在解决跨域声音分类的增量学习问题。通过在不同声学域中逐步学习相同类别声音，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2606.02173](https://arxiv.org/pdf/2606.02173)**

> **作者:** Riccardo Casciotti; Manjunath Mulimani; Manu Harju; Jesper Rindom Jensen; Annamaria Mesaros
>
> **备注:** White paper. To be completed after the challenge deadline and submitted for the DCASE 2026 Workshop
>
> **摘要:** This paper presents the Domain-Agnostic Incremental Learning for Audio Classification Task of the DCASE 2026 Challenge. Incremental learning refers to sequentially learning new tasks with the same system while maintaining its knowledge and performance on the previously learned task. Domain-incremental learning for sound classification refers to learning the same sound classes but in different acoustic domains, and was formalized as a data challenge for the first time in DCASE 2026. Participants will train a system to learn ten sound classes in three different domains, with learning at each incremental task not having access to previous task data. Submitted systems will be ranked by the overall average accuracy calculated over the three domains. The provided baseline system obtains a modest performance of 44.9\% accuracy over the three domains, mostly due to erroneous inference of the domain for the test sample.
>
---
#### [new 025] HAIM: Human-AI Music Datasets for AI Music Production Tracking Benchmark
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出HAIM数据集，用于AI音乐生成跟踪评估，解决现有检测方法仅限于“AI或人类”的二元分类问题，旨在实现更细致的AI介入阶段识别。**

- **链接: [https://arxiv.org/pdf/2606.01686](https://arxiv.org/pdf/2606.01686)**

> **作者:** Seonghyeon Go; Yumin Kim
>
> **摘要:** As generative platforms such as Suno and Udio reach human-grade audio quality, the scope of AI's utility has expanded across the entire music production workflow. Beyond simple track generation, these advancements have catalyzed the adoption of AI-driven methodologies in diverse forms. These include vocal synthesis, arrangement, and professional mastering. However, current detection research remains largely confined to a binary `AI-or-human' paradigm. It fails to reflect the realities of contemporary music production workflows. In real-world production, AI tools are increasingly used to refine or master human-produced tracks, and human engineers likewise post-process AI-generated material to ensure professional quality. Moreover, users often employ adversarial tactics to bypass AI detectors, such as applying human mastering to AI-generated tracks. This creates a grey area that a simple binary classification fails to capture. In this paper, we define and investigate ``AI Music Tracking'': the challenge of identifying specific AI integration across the multifaceted spectrum of music production. To this end, we introduce HAIM, a dataset with diverse labels for stages of music production. It is designed to isolate stages of AI intervention, including hybrid production and agent-level tracking. Our evaluation of state-of-the-art detectors reveals systemic flaws. By releasing HAIM, we propose a new benchmark that shifts the field beyond binary classification toward a granular, structured evaluation of AI music.
>
---
#### [new 026] Echo: A Joint-Embedding Predictive Architecture for Speaker Diarization and Speech Recognition in a Shared Latent Space
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Echo系统，整合说话人辨识与语音识别任务，使用共享潜空间实现联合嵌入。旨在解决多任务协同问题，通过单一编码器完成多项功能。**

- **链接: [https://arxiv.org/pdf/2606.01909](https://arxiv.org/pdf/2606.01909)**

> **作者:** Louis Mouchon
>
> **备注:** 18 pages, 17 tables, 1 figure. Proof-of-concept, independent research
>
> **摘要:** We present Echo, a proof-of-concept audio system built around a single 25 M-parameter ViT encoder. The encoder is pretrained with a JEPA objective and then specialised by stages to carry speaker identity, phonetic content, and dynamic source routing in the same 512-dimensional latent space, with no per-task fine-tuning at deployment. Light heads handle diarization (ArcFace + VBx) and dynamic source separation (null-target K-set prediction). On synthetic VoxCeleb2 mixtures with unknown K, the canonical stack reaches 15.00% blind DER, 97.80% PIT separation accuracy with +9.52 dB latent SI-SDR, and a +53.50-point speaker/content factorisation gap on a held-out k-NN probe. The point of Echo is not a new SOTA on any single task but the joint coexistence of three tasks on one encoder at this footprint. We document the design stage by stage, report the dead-ends, and identify the structural wall on end-to-end ASR through the VQ bottleneck that still bounds the PoC.
>
---
#### [new 027] UniVocal: Unified Speech-Singing Code-Switching Synthesis
- **分类: cs.SD**

- **简介: 该论文提出UniVocal，解决语音-歌曲混合合成任务（SCS），通过文本上下文隐式推断声乐模式，提升语音与歌曲自然切换效果。**

- **链接: [https://arxiv.org/pdf/2606.01677](https://arxiv.org/pdf/2606.01677)**

> **作者:** Yufei Shi; Qian Chen; Wen Wang; Xiangang Li; Zhen-Hua Ling; Yang Ai
>
> **备注:** accepted by ACL 2026
>
> **摘要:** We propose UniVocal, a unified framework that implicitly infers vocal modes from text context to pioneer Speech-Singing Code-Switching (SCS) Synthesis - a task where transitions are autonomously driven by textual semantics, akin to seamless human language blending. Unlike single-mode generation or systems relying on switching-control tags, our proposed UniVocal implicitly infers vocal modes solely from text context. To achieve this, we employ a data-efficient two-stage curriculum learning strategy that progressively trains a competitive TTS system to acquire the desired SCS capability. Addressing data scarcity, we introduce a scalable pipeline to synthesize diverse code-switching data that is both semantically and acoustically natural, alongside a new multi-scenario benchmark, SCSBench. To address limitations of semantic tokenizers in capturing acoustic details, we also introduce refined cent token and Chain-of-Thought (CoT) generation for planning prosody before content generation, effectively enhancing empathetic speech generation and singing melody. Experimental results demonstrate that UniVocal achieves state-of-the-art performance on SCSBench while maintaining competitive performance on regular speech and singing tasks. Audio samples are available at this https URL. The code and dataset are released at this https URL.
>
---
#### [new 028] A 1000-hour EEG-EMG-audio dataset of Japanese speech production
- **分类: q-bio.NC; cs.HC; cs.SD; eess.AS; eess.SP**

- **简介: 该论文发布了一个包含1000小时日语语音的多模态数据集，用于研究语音产生。任务涉及 EEG、EMG 和音频同步记录，解决多模态信号处理与语音解码问题。**

- **链接: [https://arxiv.org/pdf/2606.01264](https://arxiv.org/pdf/2606.01264)**

> **作者:** Motoshige Sato; Ilya Horiguchi; Masakazu Inoue; Kenichi Tomeoka; Eri Hatakeyama; Yuya Kita; Atsushi Yamamoto; Ippei Fujisawa; Shuntaro Sasai
>
> **摘要:** We present a multimodal dataset of 1020 hours of simultaneously recorded scalp electroencephalography (EEG), facial electromyography (EMG), and speech audio from three healthy native Japanese speakers during open-vocabulary overt speech. Recordings were acquired with three EEG systems-an ultra-high-density system (this http URL) and two cap-type systems (this http URL and eegosports), spanning 62-128 channels-across many sessions over several months. Each session provides time-synchronized EEG, facial EMG, and audio, together with speech-event annotations and transcriptions. Although collected with speech decoding as a primary motivation, the dataset also supports work on multimodal signal processing, artifact modeling, longitudinal and cross-device adaptation, and EEG representation learning. Technical validation included power spectral density and event-related potential analyses across participants, devices, and tasks, which showed the expected 1/f spectral profile, task-related alpha-band attenuation, and time-locked evoked responses. The dataset is released in Brain Imaging Data Structure (BIDS) format via OpenNeuro under a CC0 waiver to support both speech-related and broader EEG research.
>
---
#### [new 029] Diffusion-Based Heart Sound Generation: Evaluation with Physiological Signal Metrics, Classifiers, and Expert Listening
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于心音生成任务，旨在解决PCG数据不足与多样性差的问题。通过扩散模型生成心音，并评估其生理合理性、分类一致性及专家听辨效果。**

- **链接: [https://arxiv.org/pdf/2606.02448](https://arxiv.org/pdf/2606.02448)**

> **作者:** Xinqi Bao; Jia Bi; Xin Chen; Ernest Nlandu Kamavuako; Saikat Chatterjee
>
> **摘要:** Publicly available phonocardiogram (PCG) datasets remain limited in size and pathological diversity, constraining both auscultation training and the generalisation of automated heart-sound classifiers. A class-conditional diffusion model for PCG generation is developed in the log-mel domain and synthetic fidelity is assessed using complementary (i) physiology-inspired plausibility metrics, (ii) downstream label-consistency evaluation, and (iii) expert listening. Experiments use the Phy-sioNet/Computing in Cardiology Challenge 2016 dataset (3240 recordings) with recording-level splits. After preprocessing and quality control, 16,749 non-overlapping 4 s clips are mapped to a normalised 1 x 128 x 128 log-mel representation to train a conditional 2D U-Net denoiser with classifier-free guidance. Signal-level plausibility is quantified on reconstructed waveforms using three lightweight metrics: an envelope-autocorrelation rhythm score, an amplitude-based explosion score, and the dominant cycle lag. Synthetic clips preserve similar dominant cycle durations but exhibit reduced envelope periodicity and increased transient burstiness relative to real clips. For downstream evaluation, a ResNet-50 classifier achieves 92.24% accuracy on the held-out real test set and 82.8% accuracy on class-balanced synthetic batches, indicating that generated signals retain discriminative structure relevant to normal/abnormal classification. In a pilot expert listening study (60 clips, two clinicians), most synthetic clips are judged as heart-sound-like, while abnormality sensitivity is low for both real and synthetic 4 s excerpts. Overall, the results provide a practical baseline for diffusion-based PCG generation while highlighting remaining challenges in retaining abnormal acoustic cues and reducing reconstruction-induced artefacts.
>
---
#### [new 030] Spiking and Event-driven Neuromorphic Mamba Models for Efficient Speech Recognition
- **分类: cs.NE; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决深度神经网络在边缘设备上的高能耗和低效率问题。通过引入脉冲和事件驱动的神经形态模型，提升激活稀疏性，提高计算效率。**

- **链接: [https://arxiv.org/pdf/2606.01135](https://arxiv.org/pdf/2606.01135)**

> **作者:** Tauseef Ahmed; Tao Sun; Jeronimo Castrillon; Kanishkan Vadivel; Guangzhi Tang
>
> **备注:** Accepted at IJCNN2026
>
> **摘要:** Deep learning has greatly advanced automatic speech recognition (ASR), enabling widespread deployment on edge devices such as smartphones and smart home systems. However, the computational and energy demands of deep neural networks pose significant challenges for such resource-constrained deployments, introducing latency and limiting real-time interaction. Neuromorphic computing offers a promising solution by introducing activation sparsity through spiking neural networks (SNNs) and event-driven neural networks, converting dense operations into sparse computations. However, a study that evaluates the hardware benefits of different neuromorphic strategies remains lacking for ASR. This paper explores spiking and event-driven neuromorphic neural networks to improve activation sparsity in the state-of-the-art SpeechMamba model for ASR. We introduce an event-driven SpeechMamba with FATReLU activation, achieving over 60% activation sparsity with less than 1% accuracy degradation on LibriSpeech. We also propose a spiking SpeechMamba that attains over 70% sparsity while using 30% fewer parameters than comparable SNNs. Finally, we develop a cycle-accurate event-driven simulator enabling flexible algorithm-hardware co-exploration, which helps us identify computational bottlenecks and yields over 10% additional efficiency improvements.
>
---
#### [new 031] SALSA: Speech Aware LLM Adaptation via Learned Steering Activation Vectors
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出SALSA方法，解决语音感知大语言模型在域外设置下泛化能力差的问题。通过学习层间引导向量，提升语音识别性能。**

- **链接: [https://arxiv.org/pdf/2606.00460](https://arxiv.org/pdf/2606.00460)**

> **作者:** Yekaterina Yegorova; Argyrios Gerogiannis; Haolong Zheng; Julia Hockenmaier; Chang D. Yoo; Mark A. Hasegawa-Johnson
>
> **摘要:** Speech-aware large language models often generalize poorly to out-of-domain settings. We propose SALSA (Speech-Aware LLM Adaptation via Learned Steering Activations), a lightweight adaptation method that learns layer-wise steering vectors. Unlike commonly used steering approaches that rely on contrastive activation differences, SALSA directly optimizes steering vectors using a supervised objective. Across children's speech, multilingual speech, and Mandarin-English code-switching benchmarks, SALSA substantially improves performance over zero-shot inference and speech in-context learning baselines, achieving up to 46.8% relative improvements over zero-shot. Analysis further demonstrates that steering the encoder, particularly the later layers, is more effective than steering the LLM backbone. These findings suggest that steering improves downstream ASR performance by adapting higher-level acoustic and phonetic representations to better align with the pretrained language model representation space, rather than by modifying the decoder itself.
>
---
#### [new 032] Logit Distillation on Manifolds: Mapping by Learning
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于知识蒸馏任务，旨在降低模型部署成本。通过引入投影映射和LoRA，减少学生模型参数，提升性能。**

- **链接: [https://arxiv.org/pdf/2606.00771](https://arxiv.org/pdf/2606.00771)**

> **作者:** Yiru Yang; Junling Wang; Nishant Kumar Singh; Luohong Wu; Haoran Yan
>
> **摘要:** A simple way to improve the performance of almost any machine learning model is not to train a single but several models with diverse algorithms which will make slightly distinct kinds of predictions and errors on the same data, and thus improve the average predictions and robustness. However, making predictions using a whole ensemble of models is cumbersome and computationally too expensive to allow deployment to a large number of users, especially if the models are large neural nets. In response to this, we introduce a layer and point wise projection mapping, which maps student and teacher representations into an aligned high-dimensional embedding space during training process. The proposed approach combined with LoRA injection reduces the student model trainable parameters to less than 1% of the teacher model, while significantly improving word error rate (WER) compared to other distillation methods, as demonstrated in ablation studies. Unlike a mixture of experts, our method can be trained rapidly and in parallel.
>
---
#### [new 033] MURMUR: An Efficient Inference System for Long-Form ASR
- **分类: cs.LG; cs.AI; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决长文本ASR中准确率与延迟的矛盾。提出MURMUR系统，通过分块处理和注意力优化，在保持精度的同时显著降低延迟。**

- **链接: [https://arxiv.org/pdf/2606.01483](https://arxiv.org/pdf/2606.01483)**

> **作者:** Wei-Tzu Lee; Keisuke Kamahori; Baris Kasikci
>
> **摘要:** Long-form automatic speech recognition (ASR) requires both high accuracy and low latency, but existing systems force a trade-off between the two. Chunk-based pipelines process audio in parallel windows for low latency, but lose cross-chunk context and need brittle heuristics to align speakers and timestamps at boundaries. Long-context ASR models resolve everything in a single pass for better accuracy, but are an order of magnitude slower. We propose Murmur, an inference system that overcomes this trade-off by operating at two levels. At the inter-chunk level, we revisit the chunk-based pipeline for modern long-context ASR, treating chunk size as a tunable hyperparameter, and show that intermediate chunk sizes strike a good balance of accuracy and latency. At the intra-chunk level, we exploit attention sparsity through a sliding window KV cache eviction policy applied to both output and speech tokens. On AMI-IHM, Murmur matches single-pass accuracy while reducing latency by 4.2x, with further gains from token eviction at less than 1% relative tcpWER degradation. The code of Murmur is available at this https URL.
>
---
#### [new 034] PolySpeech-100: A Large-Scale Benchmark for Speech Understanding Across 100+ Languages and Dialects
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出PolySpeech-100，解决多语言与方言语音理解问题，通过混合构建方法覆盖110种语言变体，评估模型在不同场景下的表现。**

- **链接: [https://arxiv.org/pdf/2606.01016](https://arxiv.org/pdf/2606.01016)**

> **作者:** Sicheng Yang; Shulan Ruan; Shiwei Wu; Yu Liu; Lu Fan; Zhi Li; You He
>
> **备注:** 19 pages, 13 figures, KDD 2026
>
> **摘要:** While End-to-End (E2E) Speech-Large Language Models (Speech-LLMs) are rapidly evolving, their evaluation methodologies remain limited to the era of simple transcription. Existing benchmarks suffer from three critical limitations: a pronounced bias towards high-resource languages, a focus on low-level recognition (ASR) rather than semantic reasoning, and a neglect of regional dialects. To bridge this gap, we introduce PolySpeech-100, a massive-scale benchmark designed to assess `native-level' speech comprehension across 110 linguistic variants. We employ a novel hybrid construction pipeline that augments gold-standard human recordings with instruction-driven synthetic speech, allowing us to cover 19 distinct Chinese dialects and over 80 low-resource languages. Extensive evaluation of 22 state-of-the-art models (including Gemini-3, GPT-Audio, and Qwen2.5-Omni) yields pivotal insights. First, we demonstrate that open-source E2E models outperform Cascade (ASR+LLM) systems on heavy dialects, proving that direct audio processing preserves critical paralinguistic cues and prosodic features (e.g., intonation, stress) that are often lost in standard transcription. Second, we reveal a significant performance gap: while commercial models maintain robustness, open-source models suffer catastrophic degradation on low-resource languages. Finally, counter-intuitively, we observe that under standard zero-shot settings, Chain-of-Thought prompting frequently degrades speech understanding performance for most evaluated models, revealing a potential modality alignment gap in current architectures. PolySpeech-100 establishes a rigorous standard for the next generation of inclusive, omni-capable Speech-LLMs. The data, demo, and code are publicly available at this https URL.
>
---
#### [new 035] DAStatFormer: A Hybrid Multibranch Transformer with Statistical Feature Integration for DAS-Based Pattern Recognitions
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于DAS事件分类任务，旨在解决高维数据与复杂时空模式带来的分类难题。提出DAStatFormer，融合统计特征与Transformer，提升准确率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2606.00081](https://arxiv.org/pdf/2606.00081)**

> **作者:** Michel Dione; Jerry Lonlac; Hélène Louis; Anthony Fleury; Stephane Lecoeuche
>
> **摘要:** Distributed Acoustic Sensing (DAS) enables large-scale monitoring through optical fibers, but its high dimensionality and complex spatio-temporal patterns make event classification demanding. Existing deep learning approaches-CNNs, recurrent models, and Transformer variants-either fail to capture long-range dependencies or require processing raw DAS matrices at prohibitive cost. We propose DAStatFormer, a hybrid multibranch Transformer that combines compact multidomain statistical features with Gated Transformer Networks. Instead of raw signals, we extract 24 ANOVA-selected attributes per channel from the temporal, waveform, and spectral domains, reducing data size by orders of magnitude while preserving discriminative information. Each domain is processed via dedicated step-wise and channel-wise attention branches, fused by an adaptive gating mechanism. Experiments on the open $\Phi$-OTDR benchmark and a real-scenario DAS dataset show that DAS-tatFormer achieves up to 99.4% accuracy and near-perfect real-world performance, while using significantly fewer parameters and lower inference cost than models such as DASFormer and DeepViT. These results demonstrate its suitability for scalable, real-time DAS-based monitoring. We release our code at this https URL
>
---
## 更新

#### [replaced 001] DECKER: Domain-invariant Embedding for Cross-Keyboard Extraction and Recognition
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于跨键盘按键识别任务，解决声学侧信道攻击中的设备和用户泛化问题。提出DECKER框架，提升跨设备、跨用户的按键识别准确率。**

- **链接: [https://arxiv.org/pdf/2605.03384](https://arxiv.org/pdf/2605.03384)**

> **作者:** Bikrant Bikram Pratap Maurya; Nitin Choudhury; Daksh Agarwal; Arun Balaji Buduru
>
> **备注:** Accepted to AsiaCCS'26
>
> **摘要:** Acoustic side-channel attacks (ASCA) on keyboards pose a significant security risk, as keystrokes can be inferred from typing acoustics, revealing sensitive information. Prior ASCA studies are limited by small-scale datasets with restricted diversity in users, keyboards, and environments, constraining analysis across devices, microphones, and noise conditions. We introduce HEAR, a dataset designed to study ASCA along three axes: keyboard generalization, noise adaptation, and user bias. HEAR contains recordings from 53 participants using 37 laptop keyboards, collected in three realistic settings: (1) external microphone capture, (2) device microphone capture without network noise, and (3) VoIP-based streaming capture. This enables controlled evaluation across users, keyboards, and environments. On HEAR, we establish an ASCA benchmark spanning conventional features and pre-trained representations from raw audio and spectrograms in unimodal and multimodal settings. We propose DECKER, a domain-invariant keystroke inference framework with four stages: (1) Keyboard Signature Normalization to reduce device coloration, (2) domain-adversarial disentanglement to suppress keyboard identity, (3) supervised cross-keyboard contrastive alignment to enforce key consistency, and (4) Acoustic Style Randomization to synthesize unseen keyboard responses. We further explore sentence-level inference using an LLM-based post-processing layer to refine keystroke sequences via linguistic context. Results on HEAR show DECKER improves keystroke identification over strong baselines, particularly in cross-keyboard and cross-user settings, with further gains from language-model rectification. These findings highlight that ASCA remains effective across diverse users, devices, and noisy environments, underscoring its practical security risk.
>
---
#### [replaced 002] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频-文本检索任务，旨在解决传统基准与真实搜索行为不匹配的问题。提出OEA模型，引入用户意图查询和硬负样本评估指标，提升检索鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.18360](https://arxiv.org/pdf/2604.18360)**

> **作者:** HaeJun Yoo; Yongseop Shin; Insung Lee; Myoung-Wan Koo; Du-Seong Chang
>
> **备注:** Accepted at ACL 2026 Main Conference. Camera-ready version
>
> **摘要:** Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
>
---
#### [replaced 003] Embedding-Space Diffusion for Zero-Shot Environmental Sound Classification
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于零样本环境声音分类任务，旨在解决模型在未见类别上的泛化问题。通过引入生成模型，特别是扩散模型，提升分类性能，并建立首个相关基准。**

- **链接: [https://arxiv.org/pdf/2412.03771](https://arxiv.org/pdf/2412.03771)**

> **作者:** Ysobel Sims; Alexandre Mendes; Stephan Chalup
>
> **摘要:** Zero-shot learning enables models to generalise to unseen classes by leveraging semantic information, bridging the gap between training and testing sets with non-overlapping classes. While much research has focused on zero-shot learning in computer vision, the application of these methods to environmental audio remains underexplored, with poor performance in existing studies. Generative methods, which have demonstrated success in computer vision, are notably absent from zero-shot environmental sound classification studies. To address this gap, this work investigates generative methods for zero-shot learning in environmental audio. Two successful generative models from computer vision are adapted: a cross-aligned and distribution-aligned variational autoencoder (CADA-VAE) and a leveraging invariant side generative adversarial network (LisGAN). Additionally, we introduced a novel diffusion model conditioned on class auxiliary data. Synthetic embeddings generated by the diffusion model are combined with seen class embeddings to train a classifier. Experiments are conducted on five environmental audio datasets, ESC-50, ARCA23K-FSD, FSC22, UrbanSound8k and TAU Urban Acoustics 2019, and one music classification dataset, GTZAN. Results show that the diffusion model outperforms all baseline methods on average across six audio datasets. This work establishes the diffusion model as a promising approach for zero-shot learning and introduces the first benchmark of generative methods for zero-shot environmental sound classification, providing a foundation for future research.
>
---
#### [replaced 004] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于歌词翻译任务，旨在解决跨语言歌词的语义传递与音乐节奏保持问题。构建了多模态数据集MAVL，并提出SylAVL-CoT模型，提升翻译的可唱性和准确性。**

- **链接: [https://arxiv.org/pdf/2505.18614](https://arxiv.org/pdf/2505.18614)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** Accepted to EMNLP 2025, Project Page: this https URL, our codes and datasets are available at this https URL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 005] FastSLM: Hierarchical Temporal Abstraction for Efficient Long-Form Speech Adaptation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决长音频输入导致的模型效率问题。提出FastSLM，通过HTA结构实现高效时间抽象，大幅压缩token数量并保持上下文信息。**

- **链接: [https://arxiv.org/pdf/2601.06199](https://arxiv.org/pdf/2601.06199)**

> **作者:** Junseok Lee; Sangyong Lee; Chang-Jae Chun
>
> **备注:** Title updated
>
> **摘要:** Scaling Multimodal Large Language Models (MLLMs) to long-form speech is bottlenecked by the explosive growth of input tokens. Unlike images or videos, audio lacks overlapping information, making extreme 1-token compression highly susceptible to the loss of fine-grained acoustic cues. To overcome this, we propose FastSLM, a token-efficient architecture featuring the Hierarchical Temporal Abstractor (HTA). HTA progressively distills non-overlapping acoustic features across multiple temporal scales, achieving an extreme compression rate of 1.67 tokens per second a 97% reduction without losing critical context. Experimental results show that FastSLM achieves competitive performance with state-of-the-art models on long-form benchmarks despite operating with significantly fewer FLOPs and parameters. The source code and model checkpoints are available at this https URL.
>
---
#### [replaced 006] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出DSA-Tokenizer，解决语音离散化中的语义与声学分离问题，通过优化策略和层级流匹配解码器，实现高效高保真语音生成与可控克隆。**

- **链接: [https://arxiv.org/pdf/2601.09239](https://arxiv.org/pdf/2601.09239)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Linqi Song
>
> **备注:** Submit to ACL ARR 2026 May
>
> **摘要:** Speech tokenizers are a key building block of fully discrete Speech LLMs. Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably,or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement,we propose DSA-Tokenizer,which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization this http URL,semantic tokens are supervised by ASR to capture linguistic content,while acoustic tokens focus on mel-spectrograms restoration to encode this http URL further introduce a hierarchical Flow Matching decoder and a joint reconstruction-context inpainting training strategy,allowing the model to support both high-fidelity reconstruction and cross-utterance voice this http URL speed up inference,we distill the DiT decoder to reduce sampling steps of inference to 4 and improve synthesis quality with GAN this http URL demonstrate that DSA-Tokenizer provides strong semantic-acoustic disentanglement,reliable controllable voice cloning,and efficient high-fidelity generation with low WER/CER.Moreover, our results suggest that disentangled tokenization provides a more effective interface for downstream large-model speech this http URL samples are avaialble at this https URL.
>
---
#### [replaced 007] Acoustic and perceptual differences between standard and accented speech and their voice clones
- **分类: cs.SD; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于语音处理任务，研究标准与带口音汉语及其语音克隆的声学和感知差异。旨在解决语音克隆中口音保留问题，通过计算与感知实验发现口音影响克隆相似度和可懂度。**

- **链接: [https://arxiv.org/pdf/2604.01562](https://arxiv.org/pdf/2604.01562)**

> **作者:** Tianle Yang; Chengzhe Sun; Phil Rose; Siwei Lyu
>
> **摘要:** Voice cloning is often evaluated in terms of overall quality, but less is known about accent preservation and its perceptual consequences. We compare standard and heavily accented Mandarin speech and their voice clones using a combined computational and perceptual design. Embedding-based analyses showed larger original-clone distances for accented speakers in several speaker-discriminative embedding spaces, but this difference disappeared after normalizing against each speaker's within-original baseline variability. In the perception study, clones are rated as more similar to their originals for standard than for accented speakers, and intelligibility increases from original to clone, with a larger gain for accented speech. These results show that accent variation can shape perceived identity match and intelligibility in voice cloning even when it is not reflected in baseline-normalized speaker-embedding distance, and they motivate treating accent preservation as an explicit component of speaker identity preservation, rather than assuming that it is fully captured by off-the-shelf speaker-discriminative embeddings.
>
---
#### [replaced 008] SARA: Stress Test Reasoning in Audio Deepfake Detection
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决ALM推理可靠性问题。提出SARA框架，评估推理的感知、一致性与矛盾，发现声学攻击降低一致性，而语言攻击保持一致性。**

- **链接: [https://arxiv.org/pdf/2601.03615](https://arxiv.org/pdf/2601.03615)**

> **作者:** Binh Nguyen; Charles Fleming; Thai Le
>
> **备注:** Preprint for ACL 2026 submission
>
> **摘要:** Audio Language Models (ALMs) offer a promising shift towards explainable audio deepfake detections (ADD), moving beyond \textit{black-box} classifiers by providing transparency to their predictions via reasoning traces. However, such reasoning may not support the model predictions, reflecting poor coherence, or, worse, may rationalize incorrect predictions with plausible but misleading explanation. Moreover, the behavior of ALM reasoning under adversarial attacks remains under-explored, raising questions about the practical reliability of such explanation capabilities. To address this gap, this study introduces \textbf{SARA} (\textbf{S}hift \textbf{A}nalysis of \textbf{R}easoning in \textbf{A}udio), a diagnostic framework that evaluates ALM reasoning across three dimensions: acoustic perception, reasoning-verdict coherence and dissonance. We test five open-source ALMs against both acoustic and linguistic adversarial attacks. We show that acoustic attacks significantly degrade reasoning-verdict coherence (average decrease of 14.20\%), frequently inducing internal logical conflicts. Conversely, linguistic attacks achieve higher attack success rates while maintaining reasoning coherence. We further demonstrate that the textual coherence of generated reasoning traces also serves as a latent indicator of adversarial inputs, enabling effective detection of perturbed audio (0.78 in F1) \textit{without accessing the raw acoustic signal}. These findings suggest that reasoning traces provide diagnostic utility that persists even when final classification outputs are compromised.
>
---
#### [replaced 009] HRTFformer: A Spatially-Aware Transformer for Individual HRTF Upsampling in Immersive Audio Rendering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于HRTF上采样任务，解决个体HRTF测量成本高的问题。提出基于Transformer的模型，提升空间一致性与精度。**

- **链接: [https://arxiv.org/pdf/2510.01891](https://arxiv.org/pdf/2510.01891)**

> **作者:** Xuyi Hu; Jian Li; Shaojie Zhang; Stefan Goetz; Lorenzo Picinali; Ozgur B. Akan; Aidan O. T. Hogg
>
> **备注:** Accepted to IEEE Transactions on Multimedia 2026
>
> **摘要:** Individual Head-Related Transfer Functions (HRTFs) are starting to be introduced in many commercial immersive audio applications and are crucial for realistic spatial audio rendering. However, one of the main hesitations regarding their introduction is that creating individual HRTFs is impractical at scale due to the complexities of the HRTF measurement process. To mitigate this drawback, HRTF spatial upsampling has been proposed with the aim of reducing the measurements required. While prior work has seen success with different machine learning (ML) approaches, these models often struggle with long-range preservation of local spatial variation patterns across neighbouring source directions and generalization at high upsampling factors. In this paper, we propose a novel transformer-based architecture for HRTF upsampling, leveraging the attention mechanism to better capture spatial correlations across the HRTF sphere. Working in the spherical harmonic (SH) domain, our model learns to reconstruct high-resolution HRTFs from sparse input measurements with significantly improved accuracy. To enhance spatial coherence, we introduce a neighbour dissimilarity loss that promotes magnitude smoothness, yielding more realistic upsampling. We evaluate our method using both perceptual localization models and objective spectral distortion metrics. Experiments show that our model outperforms existing methods across several evaluation metrics in generating realistic, high-fidelity HRTFs.
>
---
#### [replaced 010] VocSim: A Training-free Benchmark for Zero-shot Content Identity in Single-source Audio
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出VocSim，一个无需训练的基准，用于评估单源音频中的零样本内容身份识别。任务是解决跨域音频表示的对齐问题，通过冻结模型特征和无标签PCA进行评估。**

- **链接: [https://arxiv.org/pdf/2512.10120](https://arxiv.org/pdf/2512.10120)**

> **作者:** Maris Basha; Anja Zai; Sabine Stoll; Richard Hahnloser
>
> **备注:** Accepted at ICML 2026. Code: this https URL
>
> **摘要:** General-purpose audio representations aim to map acoustically variable instances of the same event to nearby points, resolving content identity in a zero-shot setting. Unlike supervised classification benchmarks that measure adaptability via parameter updates, we introduce VocSim, a training-free benchmark probing the intrinsic geometric alignment of frozen embeddings, with no parameters updated and no labels used (a label-free PCA whitening is fit per subset to correct anisotropy). VocSim aggregates 125k single-source clips from 19 corpora spanning human speech, animal vocalizations, and environmental sounds, isolating content representation from source separation (polyphonic mixtures are out of scope). We evaluate embeddings with Precision@k for local purity and the Global Separation Rate (GSR) for point-wise class separation, calibrated by lift over an empirical permutation baseline. A simple pipeline of frozen Whisper features, time-frequency pooling, and label-free PCA yields strong zero-shot performance with stable GSR rankings across domains (Kendall's tau = 0.60). However, on blind low-resource speech (Shipibo-Conibo, Chintang), local retrieval collapses while remaining above chance, exposing a cross-lingual speech generalization gap. As external validation, our top embeddings predict avian perceptual similarity, improve bioacoustic classification, and achieve state-of-the-art on the HEAR benchmark. We release data, code, and a public leaderboard.
>
---
#### [replaced 011] Escaping the BLEU Trap: A Signal-Grounded Framework with Decoupled Semantic Guidance for EEG-to-Text Decoding
- **分类: cs.CL; cs.AI; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文属于EEG-to-Text解码任务，旨在解决语义偏差、信号忽视和BLEU陷阱问题。提出SemKey框架，通过分离语义目标和主动检索解码，提升生成质量与信号一致性。**

- **链接: [https://arxiv.org/pdf/2603.03312](https://arxiv.org/pdf/2603.03312)**

> **作者:** Yuchen Wang; Haonan Wang; Yu Guo; Honglong Yang; Xiaomeng Li
>
> **摘要:** Decoding natural language from non-invasive EEG signals is a promising yet challenging task. However, current state-of-the-art models remain constrained by three fundamental issues: Semantic Bias, where outputs collapse into generic linguistic templates; Signal Neglect, where models rely heavily on LLM priors to hallucinate fluent text even in the absence of meaningful signals; and the "BLEU Trap", where high-frequency stopwords inflate n-gram metrics, masking a lack of true semantic fidelity. To resolve these challenges, we move beyond conventional end-to-end pipelines and propose SemKey, a novel multi-stage framework that enforces signal-grounded generation through four decoupled semantic objectives: sentiment, topic, length, and surprisal. We extract these semantic anchors from EEG embeddings directly, then unify them with an Active Retrieval Decoding mechanism, compelling the LLM to ground its token generation in the neural signals rather than defaulting to linguistic priors. Furthermore, we break the BLEU Trap by establishing a comprehensive evaluation protocol using rigorous retrieval and distribution-based metrics such as Fréchet Distance. Extensive experiments demonstrate that SemKey effectively mitigates hallucinations on noise inputs and achieves SOTA performance on these robust protocols. Code will be released upon acceptance at this https URL.
>
---
#### [replaced 012] DiffAU: Diffusion-Based Ambisonics Upscaling
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频处理任务，旨在解决Ambisonics低分辨率问题。通过提出DiffAU方法，将一阶Ambisonics提升至三阶，以提高空间音频的 realism。**

- **链接: [https://arxiv.org/pdf/2510.00180](https://arxiv.org/pdf/2510.00180)**

> **作者:** Amit Milstein; Nir Shlezinger; Boaz Rafaely
>
> **摘要:** Spatial audio enhances immersion by reproducing 3D sound fields, with Ambisonics offering a scalable format for this purpose. While first-order Ambisonics (FOA) notably facilitates hardware-efficient acquisition and storage of sound fields as compared to high-order Ambisonics (HOA), its low spatial resolution limits realism, highlighting the need for Ambisonics upscaling (AU) as an approach for increasing the order of Ambisonics signals. In this work we propose DiffAU, a cascaded AU method that leverages recent developments in diffusion models combined with novel adaptation to spatial audio to generate 3rd order Ambisonics from FOA. By learning data distributions, DiffAU provides a principled approach that rapidly and reliably reproduces HOA in various settings. Experiments in anechoic conditions with multiple speakers, show strong objective and perceptual performance.
>
---
#### [replaced 013] ASKD-Whisper: Adaptive Self-knowledge Distillation for Efficient and Low-Latency Automatic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决知识蒸馏中学生模型过度依赖教师模型导致的泛化能力下降问题。提出ASGD方法，提升模型压缩效果与推理效率。**

- **链接: [https://arxiv.org/pdf/2601.19919](https://arxiv.org/pdf/2601.19919)**

> **作者:** Junseok Lee; Nahun Kim; Sangyong Lee; Chang-Jae Chun
>
> **备注:** Title and content have been updated
>
> **摘要:** Knowledge distillation (KD) is one of the most effective paradigms for compressing large-scale foundation models into deployable architectures. In the context of Automatic Speech Recognition (ASR), previous studies have predominantly focused on forcing the student model to strictly mimic the predictive distribution of a massive teacher model. However, this static dependency often presents an inherent trade-off: while the student rapidly acquires basic linguistic representations, it simultaneously inherits the teacher's domain-specific blind spots and over-confident hallucinations, leading to a severe decline in out-of-distribution generalization capacity. To effectively mitigate this issue, we propose Adaptive Self-Knowledge Distillation (ASKD), a dynamic curriculum framework. ASKD systematically decays the dependency on the teacher's distribution as training progresses-thereby unlocking the student's independent reasoning capacity-and subsequently employs a self-knowledge distillation phase to act as a structural regularizer. By applying ASKD, we distill the massive Whisper architecture into a compact variant, ASKD-Whisper. In our comprehensive evaluations across diverse acoustic domains, ASKD-Whisper not only achieves a 5x speedup in inference latency but also outperforms its teacher model by yielding a 1.07% lower word error rate (WER). These results demonstrate that ASKD effectively prevents teacher-induced overfitting and establishes a new state-of-the-art for generalizable model compression.
>
---
#### [replaced 014] Step-Audio-R1.5 Technical Report
- **分类: eess.AS**

- **简介: 该论文属于语音推理任务，旨在解决RLVR导致的音频模型缺乏真实对话感的问题，提出Step-Audio-R1.5采用RLHF提升交互体验。**

- **链接: [https://arxiv.org/pdf/2604.25719](https://arxiv.org/pdf/2604.25719)**

> **作者:** Yuxin Zhang; Xiangyu Tony Zhang; Daijiao Liu; Fei Tian; Yayue Deng; Jun Chen; Qingjian Lin; Haoyang Zhang; Yuxin Li; Jinglan Gong; Yechang Huang; Liang Zhao; Chengyuan Yao; Hexin Liu; Eng Siong Chng; Xuerui Yang; Gang Yu; Xiangyu Zhang; Daxin Jiang
>
> **摘要:** Recent advancements in large audio language models have extended Chain-of-Thought (CoT) reasoning into the auditory domain, enabling models to tackle increasingly complex acoustic and spoken tasks. To elicit and sustain these extended reasoning chains, the prevailing paradigm -- driven by the success of text-based reasoning models -- overwhelmingly relies on Reinforcement Learning with Verified Rewards (RLVR). However, as models are strictly optimized to distill rich, continuous auditory contexts into isolated, verifiable text labels, a fundamental question arises: are we fostering true audio intelligence, or merely reducing a continuous sensory medium into a discrete puzzle? We identify this as the "verifiable reward trap." While RLVR yields remarkable scores on standardized objective benchmarks, it systematically degrades the real-world conversational feel of audio models. By prioritizing isolated correctness over acoustic nuance, RLVR reduces dynamic interactions to mechanical "answering machines," severely compromising prosodic naturalness, emotional continuity, and user immersion, particularly in long-turn dialogues. To bridge the gap between mechanical objective verification and genuine sensory empathy, we introduce Step-Audio-R1.5, marking a paradigm shift toward Reinforcement Learning from Human Feedback (RLHF) in audio reasoning. Comprehensive evaluations demonstrate that Step-Audio-R1.5 not only maintains robust analytical reasoning but profoundly transforms the interactive experience, redefining the boundaries of deeply immersive long-turn spoken dialogue.
>
---
#### [replaced 015] HoliTok:A Coutinuous Holistic Tokenization with Robust Dual Capabilities of Speech Generation and Understanding
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出HoliTok，一种用于统一语音生成与理解的连续整体分词模型。解决现有分词器难以同时满足语言模型学习和高质量波形解码的问题。通过优化分词策略，提升语音合成与识别性能。**

- **链接: [https://arxiv.org/pdf/2605.29948](https://arxiv.org/pdf/2605.29948)**

> **作者:** Bohan Li; Shi Lian; Hankun Wang; Yiwei Guo; Yu Xi; Zhihan Li; Da Zheng; Colin Zhang; Kai Yu
>
> **备注:** 14 pages, 2 figures, 8 tables
>
> **摘要:** Unified speech foundation models require a holistic tokenization space that is both learnable by language models and decodable into high-quality waveforms. Existing speech tokenizers, however, often fail to satisfy these requirements simultaneously, leading to increased architectural complexity and more involved training designs. We propose HoliTok, a continuous Holistic speech Tokenization model designed for unified generation-understanding modeling. HoliTok encodes 48~kHz speech into a compact 25~Hz sequence of 128-dimensional latents. It is trained with a progressive strategy that jointly preserves signal-level fidelity, incorporates semantic information, and maintains strong latent learnability. Based on this tokenization, we build a unified AR+DiT model for speech synthesis and recognition, where the same latent sequence supports both generation-specific and unified generation-understanding tasks. Experiments show that HoliTok achieves competitive reconstruction fidelity, improves generative learnability for high-quality and controllable synthesis, and, among the evaluated representations, is the only one that operates robustly in our unified generation-understanding architecture without additional optimization tricks. These results suggest that HoliTok serves as an effective speech tokenizer and a foundational representation interface for unified spoken language modeling. The code is available at: this https URL.
>
---
#### [replaced 016] Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Chatterbox-Flash，用于零样本文本转语音任务，解决块扩散解码质量下降问题，通过引入先验校准评分和早停策略，实现高效流式合成。**

- **链接: [https://arxiv.org/pdf/2605.30748](https://arxiv.org/pdf/2605.30748)**

> **作者:** Deokjin Seo; Gangin Park; Kihyun Nam
>
> **备注:** 8 pages, 4 figures, 9 tables
>
> **摘要:** We present Chatterbox-Flash, a zero-shot text-to-speech model obtained by fine-tuning a pretrained autoregressive TTS decoder into a block-diffusion decoder, enabling parallel token generation within each block while retaining block-by-block streaming. We find that naively transferring mainstream block-diffusion decoding to discrete speech tokens degrades quality, as a long-tail token distribution biases parallel position selection toward a few high-frequency tokens. To mitigate this without architectural modification, we introduce two inference-time techniques: prior-calibrated scoring, which subtracts the block-level marginal token distribution, and an early-decoding schedule, which adaptively terminates iteration based on calibrated confidence. On standard zero-shot TTS benchmarks, Chatterbox-Flash attains high-fidelity synthesis comparable to strong autoregressive and non-autoregressive baselines, while supporting streaming inference with time-to-first-packet on par with streaming AR systems and substantially lower real-time factor. Code and audio samples are available at this https URL.
>
---
#### [replaced 017] Systematic Evaluation of Time-Frequency Features for Binaural Sound Source Localization
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于双耳声源定位任务，研究如何通过时间-频率特征设计提升模型性能，解决不同条件下定位准确率问题。工作包括评估不同特征组合的效果，发现合理特征选择比增加模型复杂度更有效。**

- **链接: [https://arxiv.org/pdf/2511.13487](https://arxiv.org/pdf/2511.13487)**

> **作者:** Davoud Shariat Panah; Alessandro Ragano; Dan Barry; Jan Skoglund; Andrew Hines
>
> **备注:** Accepted at EUSIPCO 2026
>
> **摘要:** This study presents a systematic evaluation of time-frequency feature design for binaural sound source localization (SSL), focusing on how feature selection influences model performance across diverse conditions. We investigate the performance of a convolutional neural network (CNN) model using various combinations of amplitude-based features (magnitude spectrogram, interaural level difference - ILD) and phase-based features (phase spectrogram, interaural phase difference - IPD). Evaluations on in-domain and out-of-domain data with mismatched head-related transfer functions (HRTFs) reveal that carefully chosen feature combinations often outperform increases in model complexity. While two-feature sets such as ILD + IPD are sufficient for in-domain SSL, generalization to diverse content requires richer inputs combining channel spectrograms with both ILD and IPD. Using the optimal feature sets, our low-complexity CNN model achieves competitive performance. Our findings underscore the importance of feature design in binaural SSL and provide practical guidance for both domain-specific and general-purpose localization.
>
---
#### [replaced 018] Do Joint Audio-Video Generation Models Understand Physics?
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于多模态生成任务，旨在评估联合音视频生成模型是否理解物理常识。通过构建基准测试，分析模型在物理一致性上的表现，识别其不足与挑战。**

- **链接: [https://arxiv.org/pdf/2605.07061](https://arxiv.org/pdf/2605.07061)**

> **作者:** Zijun Cui; Xiulong Liu; Hao Fang; Mingwei Xu; Jiageng Liu; Zexin Xu; Weiguo Pian; Shijian Deng; Feiyu Du; Chenming Ge; Yapeng Tian
>
> **备注:** Preprint. Project Page: this https URL. Full abstract appears in the PDF
>
> **摘要:** Joint audio-video generation models are rapidly approaching professional production quality, raising a central question: do they understand audio-visual physics, or merely generate plausible sounds and frames that violate real-world consistency? We introduce AV-Phys Bench, a benchmark for evaluating physical commonsense in joint audio-video generation. AV-Phys Bench tests models across three scene categories: Steady State, Event Transition, and Environment Transition. It covers physics-grounded subcategories drawn from real-world scenes, plus Anti-AV-Physics prompts that deliberately request physically inconsistent audio-video behavior. Each generation is evaluated along five dimensions: visual semantic adherence, audio semantic adherence, visual physical commonsense, audio physical commonsense, and cross-modal physical commonsense. Across three proprietary and four open-source models, we find that Seedance 2.0 performs best overall, but all models remain far from robust physical understanding. Performance drops sharply on event-driven and environment-driven transitions, and even strong proprietary systems collapse on Anti-AV-Physics prompts. We further introduce AV-Phys Agent, a ReAct-style evaluator that combines a multimodal language model with deterministic acoustic measurement tools, producing rankings that closely align with human ratings. Our results identify cross-modal physical consistency and transition-driven scene dynamics as key open challenges for joint audio-video generation.
>
---
#### [replaced 019] Description and Discussion on DCASE 2026 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes
- **分类: eess.AS**

- **简介: 该论文介绍DCASE 2026 Task 4任务，旨在解决复杂声场中声音事件的联合检测与分离问题，提升沉浸式通信基础。工作包括任务设置、评估指标更新及实验分析。**

- **链接: [https://arxiv.org/pdf/2604.00776](https://arxiv.org/pdf/2604.00776)**

> **作者:** Binh Thien Nguyen; Masahiro Yasuda; Noboru Harada; Romain Serizel; Mayank Mishra; Marc Delcroix; Carlos Hernandez-Olivan; Shoko Araki; Daiki Takeuchi; Tomohiro Nakatani; Nobutaka Ono
>
> **摘要:** This paper presents an overview of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2026 Challenge Task 4, Spatial Semantic Segmentation of Sound Scenes (S5). The S5 task focuses on the joint detection and separation of sound events in complex spatial audio mixtures, contributing to the foundation of immersive communication. First introduced in DCASE 2025, the S5 task continues in DCASE 2026 Task 4 with key changes to better reflect real-world conditions, including allowing mixtures to contain multiple sources of the same class and to contain no target sources. In this paper, we describe task setting, along with the corresponding updates to the evaluation metrics and dataset. The experimental results of the submitted systems are also reported and analyzed. The official access point for data and code is this https URL.
>
---
#### [replaced 020] The Alignment Curse: Modality Alignment Supercharges Audio Attacks via Text Transfer
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于安全评估任务，研究文本与音频模态对齐对攻击转移的影响。解决音频安全风险低估问题，通过实验验证文本攻击可有效转移到音频，揭示模态对齐带来的安全隐患。**

- **链接: [https://arxiv.org/pdf/2602.02557](https://arxiv.org/pdf/2602.02557)**

> **作者:** Yupeng Chen; Junchi Yu; Aoxi Liu; Baoyuan Wu; Philip Torr; Adel Bibi
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Recent advances in end-to-end trained omni-models have substantially improved audio capabilities by strengthening text-audio modality alignment. However, whether such alignment inadvertently facilitates the transfer of safety vulnerabilities across modalities remains underexplored. This question is critical as text-based jailbreak attacks are considerably more mature than audio-based ones; if they transfer systematically, current audio safety evaluations may underestimate risks originating from the text modality. In this paper, we introduce the Alignment Curse, a formally characterized and empirically validated principle showing that stronger modality alignment enables more effective transfer of attacks from text to audio, revealing a fundamental tension between capability and safety. Motivated by this principle, we conduct a comprehensive black-box evaluation of three attack categories on recent omni-models (e.g., Qwen2.5-Omni, Qwen3-Omni): text attacks, text-transferred audio attacks, and audio attacks. We find that text-transferred audio attacks perform comparably to, and often better than, audio-based attacks, exhibiting a clear advantage under audio-only access. This suggests that text-based vulnerabilities play a pivotal role in shaping audio safety risks. Finally, we empirically analyze the relationship between modality alignment and transfer effectiveness across attack methods and models, observing consistent support for the Alignment Curse: tighter modality alignment leads to more effective cross-modality attack transfer.
>
---
#### [replaced 021] BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决音乐符号表示中时间不统一的问题。通过将音乐划分为固定时长的节拍作为基本单元，实现更高效的建模与生成。**

- **链接: [https://arxiv.org/pdf/2604.19532](https://arxiv.org/pdf/2604.19532)**

> **作者:** Lekai Qian; Haoyu Gu; Jingwei Zhao; Ziyu Wang
>
> **摘要:** Tokenizing music to fit the general framework of language models is a compelling challenge, especially considering the diverse symbolic structures in which music can be represented (e.g., sequences, grids, and graphs). To date, most approaches tokenize symbolic music as sequences of musical events, such as onsets, pitches, time shifts, or compound note events. This strategy is intuitive and has proven effective in Transformer-based models, but it treats the regularity of musical time implicitly: individual tokens may span different durations, resulting in non-uniform time progression. In this paper, we instead consider whether an alternative tokenization is possible, where a uniform-length musical step (e.g., a beat) serves as the basic unit. Specifically, we encode all events within a single time step at the same pitch as one token, and group tokens explicitly by time step, which resembles a sparse encoding of a piano-roll representation. We evaluate the proposed tokenization on music continuation and accompaniment generation tasks, comparing it with mainstream event-based methods. Results show improved musical quality and structural coherence, while additional analyses confirm higher efficiency and more effective capture of long-range patterns with the proposed tokenization.
>
---
