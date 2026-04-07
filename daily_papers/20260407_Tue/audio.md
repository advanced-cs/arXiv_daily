# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] AffectSpeech: A Large-Scale Emotional Speech Dataset with Fine-Grained Textual Descriptions for Speech Emotion Captioning and Synthesis
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出AffectSpeech数据集，解决情感语音描述与生成任务中缺乏高质量标注的问题，通过多维度标注和人机协作流程提升数据质量与多样性。**

- **链接: [https://arxiv.org/pdf/2604.04160](https://arxiv.org/pdf/2604.04160)**

> **作者:** Tianhua Qi; Wenming Zheng; Björn W. Schuller; Zhaojie Luo; Haizhou Li
>
> **备注:** Submitted to IEEE Transactions
>
> **摘要:** Emotion is essential in spoken communication, yet most existing frameworks in speech emotion modeling rely on predefined categories or low-dimensional continuous attributes, which offer limited expressive capacity. Recent advances in speech emotion captioning and synthesis have shown that textual descriptions provide a more flexible and interpretable alternative for representing affective characteristics in speech. However, progress in this direction is hindered by the lack of an emotional speech dataset aligned with reliable and fine-grained natural language annotations. To tackle this, we introduce AffectSpeech, a large-scale corpus of human-recorded speech enriched with structured descriptions for fine-grained emotion analysis and generation. Each utterance is characterized across six complementary dimensions, including sentiment polarity, open-vocabulary emotion captions, intensity level, prosodic attributes, prominent segments, and semantic content, enabling multi-granular modeling of vocal expression. To balance annotation quality and scalability, we adopt a human-LLM collaborative annotation pipeline that integrates algorithmic pre-labeling, multi-LLM description generation, and human-in-the-loop verification. Furthermore, these annotations are reformulated into diverse descriptive styles to enhance linguistic diversity and reduce stylistic bias in downstream modeling. Experimental results on speech emotion captioning and synthesis demonstrate that models trained on AffectSpeech consistently achieve superior performance across multiple evaluation settings.
>
---
#### [new 002] Rewriting TTS Inference Economics: Lightning V2 on Tenstorrent Achieves 4x Lower Cost Than NVIDIA L40S
- **分类: eess.AS; cs.DC; cs.SD**

- **简介: 该论文属于语音合成任务，解决TTS模型在低精度计算中的质量下降问题。通过硬件软件协同优化，实现高效低精度推理，降低成本。**

- **链接: [https://arxiv.org/pdf/2604.03279](https://arxiv.org/pdf/2604.03279)**

> **作者:** Ranjith M. S.; Akshat Mandloi; Sudarshan Kamath
>
> **摘要:** Text-to-Speech (TTS) models are significantly more numerically fragile than Large Language Models (LLMs) due to their continuous waveform generation and perceptual sensitivity to small numerical perturbations. While aggressive precision reduction techniques such as BlockFloat8 (BFP8) and low-fidelity (LoFi) compute have been widely adopted in language models, applying similar strategies to TTS systems often results in audible artifacts, phase instability, and spectral distortion. In this work, we present Lightning V2, a production-grade TTS model co-optimized for Tenstorrent hardware. Through precision-aware architectural design and hardware-software co-optimization, we achieve over 95% LoFi computational fidelity and more than 80% BlockFloat8 deployment without measurable degradation in audio quality. Leveraging Tenstorrent's Network-on-Chip (NoC), distributed SRAM, and deterministic execution model, we reduce memory movement and redundant weight fetches, enabling efficient low-precision inference. Compared to an NVIDIA L40S baseline, Lightning V2 achieves approximately 4x lower on-prem accelerator cost at equivalent throughput, while maintaining production audio fidelity. Our results demonstrate that precision co-design, combined with hardware-aware optimization, can fundamentally reshape the economics of real-time speech inference.
>
---
#### [new 003] Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FDB-v3基准，用于评估全双工语音代理在真实语境下的表现，解决多步骤工具使用中的语音模型性能问题。**

- **链接: [https://arxiv.org/pdf/2604.04847](https://arxiv.org/pdf/2604.04847)**

> **作者:** Guan-Ting Lin; Chen Chen; Zhehuai Chen; Hung-yi Lee
>
> **备注:** Work in progress. Demo at this https URL
>
> **摘要:** We introduce Full-Duplex-Bench-v3 (FDB-v3), a benchmark for evaluating spoken language models under naturalistic speech conditions and multi-step tool use. Unlike prior work, our dataset consists entirely of real human audio annotated for five disfluency categories, paired with scenarios requiring chained API calls across four task domains. We evaluate six model configurations -- GPT-Realtime, Gemini Live 2.5, Gemini Live 3.1, Grok, Ultravox v0.7, and a traditional Cascaded pipeline (Whisper$\rightarrow$GPT-4o$\rightarrow$TTS) -- across accuracy, latency, and turn-taking dimensions. GPT-Realtime leads on Pass@1 (0.600) and interruption avoidance (13.5\%); Gemini Live 3.1 achieves the fastest latency (4.25~s) but the lowest turn-take rate (78.0\%); and the Cascaded baseline, despite a perfect turn-take rate, incurs the highest latency (10.12~s). Across all systems, self-correction handling and multi-step reasoning under hard scenarios remain the most consistent failure modes.
>
---
#### [new 004] FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统中的端到端检测任务，旨在解决实时全双工通信中的低延迟和鲁棒性问题。提出FastTurn框架，结合声学与语义线索，提升决策准确性与响应速度。**

- **链接: [https://arxiv.org/pdf/2604.01897](https://arxiv.org/pdf/2604.01897)**

> **作者:** Chengyou Wang; Hongfei Xue; Chunjiang He; Jingbin Hu; Shuiyuan Wang; Bo Wu; Yuyu Ji; Jimeng Zheng; Ruofei Chen; Zhou Zhu; Lei Xie
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
>
---
#### [new 005] MALEFA: Multi-grAnularity Learning and Effective False Alarm Suppression for Zero-shot Keyword Spotting
- **分类: eess.AS**

- **简介: 该论文属于零样本关键词检测任务，旨在解决无领域数据下的准确识别与误报问题。提出MALEFA框架，通过多粒度学习降低误报率，提升效率。**

- **链接: [https://arxiv.org/pdf/2604.03689](https://arxiv.org/pdf/2604.03689)**

> **作者:** Lo-Ya Li; Tien-Hong Lo; Jeih-Weih Hung; Shih-Chieh Huang; Berlin Chen
>
> **备注:** Accepted by ICASSP 2026. 5 pages, 4 figures
>
> **摘要:** User-defined keyword spotting (KWS) without resorting to domain-specific pre-labeled training data is of fundamental importance in building adaptable and personalized voice interfaces. However, such systems are still faced with arduous challenges, including constrained computational resources and limited annotated training data. Existing methods also struggle to distinguish acoustically similar keywords, often leading to a pesky false alarm rate (FAR) in real-world deployments. To mitigate these limitations, we put forward MALEFA, a novel lightweight zero-shot KWS framework that jointly learns utterance- and phoneme-level alignments via cross-attention and a multi-granularity contrastive learning objective. Evaluations on four public benchmark datasets show that MALEFA achieves a high accuracy of 90%, significantly reducing FAR to 0.007% on the AMI dataset. Beyond its strong performance, MALEFA demonstrates high computational efficiency and can readily support real-time deployment on resource-constrained devices.
>
---
#### [new 006] Composer Vector: Style-steering Symbolic Music Generation in a Latent Space
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于符号音乐生成任务，旨在解决composer风格控制困难的问题。提出Composer Vector方法，在不重新训练模型的情况下，通过潜空间调节实现风格引导与融合。**

- **链接: [https://arxiv.org/pdf/2604.03333](https://arxiv.org/pdf/2604.03333)**

> **作者:** Xunyi Jiang; Mingyang Yao; Jingyue Huang; Julian McAuley
>
> **摘要:** Symbolic music generation has made significant progress, yet achieving fine-grained and flexible control over composer style remains challenging. Existing training-based methods for composer style conditioning depend on large labeled datasets. Besides, these methods typically support only single-composer generation at a time, limiting their applicability to more creative or blended scenarios. In this work, we propose Composer Vector, an inference-time steering method that operates directly in the model's latent space to control composer style without retraining. Through experiments on multiple symbolic music generation models, we show that Composer Vector effectively guides generations toward target composer styles, enabling smooth and interpretable control through a continuous steering coefficient. It also enables seamless fusion of multiple styles within a unified latent space framework. Overall, our work demonstrates that simple latent space steering provides a practical and general mechanism for controllable symbolic music generation, enabling more flexible and interactive creative workflows. Code and Demo are available here: this https URL and this https URL
>
---
#### [new 007] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文提出OmniSonic，解决视频与文本驱动的全场景音频生成问题，整合屏幕内外声音，支持语音生成。**

- **链接: [https://arxiv.org/pdf/2604.04348](https://arxiv.org/pdf/2604.04348)**

> **作者:** Weiguo Pian; Saksham Singh Kushwaha; Zhimin Chen; Shijian Deng; Kai Wang; Yunhui Guo; Yapeng Tian
>
> **备注:** CVPR 2026
>
> **摘要:** In this paper, we propose Universal Holistic Audio Generation (UniHAGen), a task for synthesizing comprehensive auditory scenes that include both on-screen and off-screen sounds across diverse domains (e.g., ambient events, musical instruments, and human speech). Prior video-conditioned audio generation models typically focus on producing on-screen environmental sounds that correspond to visible sounding events, neglecting off-screen auditory events. While recent holistic joint text-video-to-audio generation models aim to produce auditory scenes with both on- and off-screen sound but they are limited to non-speech sounds, lacking the ability to generate or integrate human speech. To overcome these limitations, we introduce OmniSonic, a flow-matching-based diffusion framework jointly conditioned on video and text. It features a TriAttn-DiT architecture that performs three cross-attention operations to process on-screen environmental sound, off-screen environmental sound, and speech conditions simultaneously, with a Mixture-of-Experts (MoE) gating mechanism that adaptively balances their contributions during generation. Furthermore, we construct UniHAGen-Bench, a new benchmark with over one thousand samples covering three representative on/off-screen speech-environment scenarios. Extensive experiments show that OmniSonic consistently outperforms state-of-the-art approaches on both objective metrics and human evaluations, establishing a strong baseline for universal and holistic audio generation. Project page: this https URL
>
---
#### [new 008] Measuring Robustness of Speech Recognition from MEG Signals Under Distribution Shift
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音识别任务，旨在提升从MEG信号中鲁棒解码语音的能力。通过比较不同模型和预处理方法，研究发现数据预处理对模型性能影响更大，尤其实例归一化对泛化能力至关重要。**

- **链接: [https://arxiv.org/pdf/2604.04129](https://arxiv.org/pdf/2604.04129)**

> **作者:** Sheng-You Chien; Bo-Yi Mao; Yi-Ning Chang; Po-Chih Kuo
>
> **备注:** 17 pages, 6 figures, LibriBrain Competition @NeurIPS2025
>
> **摘要:** This study investigates robust speech-related decoding from non-invasive MEG signals using the LibriBrain phoneme-classification benchmark from the 2025 PNPL competition. We compare residual convolutional neural networks (CNNs), an STFT-based CNN, and a CNN--Transformer hybrid, while also examining the effects of group averaging, label balancing, repeated grouping, normalization strategies, and data augmentation. Across our in-house implementations, preprocessing and data-configuration choices matter more than additional architectural complexity, among which instance normalization emerges as the most influential modification for generalization. The strongest of our own models, a CNN with group averaging, label balancing, repeated grouping, and instance normalization, achieves 60.95% F1-macro on the test split, compared with 39.53% for the plain CNN baseline. However, most of our models, without instance normalization, show substantial validation-to-test degradation, indicating that distribution shift induced by different normalization statistics is a major obstacle to generalization in our experiments. By contrast, MEGConformer maintains 64.09% F1-macro on both validation and test, and saliency-map analysis is qualitatively consistent with this contrast: weaker models exhibit more concentrated or repetitive phoneme-sensitive patterns across splits, whereas MEGConformer appears more distributed. Overall, the results suggest that improving the reliability of non-invasive phoneme decoding will likely require better handling of normalization-related distribution shift while also addressing the challenge of single-trial decoding.
>
---
#### [new 009] Joint Fullband-Subband Modeling for High-Resolution SingFake Detection
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决高分辨率歌唱语音伪造检测问题。通过结合全频段与子频段建模，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.04841](https://arxiv.org/pdf/2604.04841)**

> **作者:** Xuanjun Chen; Chia-Yu Hu; Sung-Feng Huang; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Rapid advances in singing voice synthesis have increased unauthorized imitation risks, creating an urgent need for better Singing Voice Deepfake (SingFake) Detection, also known as SVDD. Unlike speech, singing contains complex pitch, wide dynamic range, and timbral variations. Conventional 16 kHz-sampled detectors prove inadequate, as they discard vital high-frequency information. This study presents the first systematic analysis of high-resolution (44.1 kHz sampling rate) audio for SVDD. We propose a joint fullband-subband modeling framework: the fullband captures global context, while subband-specific experts isolate fine-grained synthesis artifacts unevenly distributed across the spectrum. Experiments on the WildSVDD dataset demonstrate that high-frequency subbands provide essential complementary cues. Our framework significantly outperforms 16 kHz-sampled models, proving that high-resolution audio and strategic subband integration are critical for robust in-the-wild detection.
>
---
#### [new 010] Hierarchical Semantic Correlation-Aware Masked Autoencoder for Unsupervised Audio-Visual Representation Learning
- **分类: cs.MM; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于跨模态表示学习任务，旨在解决弱配对、无标签数据中的对齐问题。提出HSC-MAE框架，通过多层级语义一致性约束提升音频-视觉表示质量。**

- **链接: [https://arxiv.org/pdf/2604.04229](https://arxiv.org/pdf/2604.04229)**

> **作者:** Donghuo Zeng; Hao Niu; Masato Taya
>
> **备注:** 6 pages, 2 tables, 4 figures. Accepted by IEEE ICME 2026
>
> **摘要:** Learning aligned multimodal embeddings from weakly paired, label-free corpora is challenging: pipelines often provide only pre-extracted features, clips contain multiple events, and spurious co-occurrences. We propose HSC-MAE (Hierarchical Semantic Correlation-Aware Masked Autoencoder), a dual-path teacher-student framework that enforces semantic consistency across three complementary levels of representation - from coarse to fine: (i) global-level canonical-geometry correlation via DCCA, which aligns audio and visual embeddings within a shared modality-invariant subspace; (ii) local-level neighborhood-semantics correlation via teacher-mined soft top-k affinities, which preserves multi-positive relational structure among semantically similar instances; and (iii) sample-level conditional-sufficiency correlation via masked autoencoding, which ensures individual embeddings retain discriminative semantic content under partial observation. Concretely, a student MAE path is trained with masked feature reconstruction and affinity-weighted soft top-k InfoNCE; an EMA teacher operating on unmasked inputs via the CCA path supplies stable canonical geometry and soft positives. Learnable multi-task weights reconcile competing objectives, and an optional distillation loss transfers teacher geometry into the student. Experiments on AVE and VEGAS demonstrate substantial mAP improvements over strong unsupervised baselines, validating that HSC-MAE yields robust and well-structured audio-visual representations.
>
---
#### [new 011] DHFP-PE: Dual-Precision Hybrid Floating Point Processing Element for AI Acceleration
- **分类: cs.AR; cs.RO; eess.AS; eess.IV**

- **简介: 该论文属于AI加速任务，旨在解决低功耗高吞吐量浮点运算问题，提出一种双精度混合浮点处理单元DHFP-PE。**

- **链接: [https://arxiv.org/pdf/2604.04507](https://arxiv.org/pdf/2604.04507)**

> **作者:** Shubham Kumar; Vijay Pratap Sharma; Vaibhav Neema; Santosh Kumar Vishvakarma
>
> **备注:** Accepted in ANRF-sponsored 2nd International Conference on Next Generation Electronics (NEleX-2026)
>
> **摘要:** The rapid adoption of low-precision arithmetic in artificial intelligence and edge computing has created a strong demand for energy-efficient and flexible floating-point multiply-accumulate (MAC) units. This paper presents a fully pipelined dual-precision floating-point MAC processing engine supporting FP8 formats (E4M3, E5M2) and FP4 formats (E2M1, E1M2), specifically optimized for low-power and high-throughput AI workloads. The proposed architecture employs a novel bit-partitioning technique that enables a single 4-bit unit multiplier to operate either as a standard 4x4 multiplier for FP8 or as two parallel 2x2 multipliers for 2-bit operands, achieving 100 percent hardware utilization without duplicating logic. Implemented in 28 nm technology, the proposed processing engine achieves an operating frequency of 1.94 GHz with an area of 0.00396 mm^2 and power consumption of 2.13 mW, resulting in up to 60.4 percent area reduction and 86.6 percent power savings compared to state-of-the-art designs.
>
---
#### [new 012] Neurological Plausibility of AI-Generated Music for Commercial Environments: An In-Silico Cortical Investigation Using Wubble and TRIBE v2
- **分类: q-bio.NC; cs.SD**

- **简介: 该论文属于神经科学与人工智能交叉任务，旨在评估AI生成音乐在商业环境中的神经可塑性。通过模拟脑区反应，研究AI音乐对大脑特定区域的影响，为商业音乐生成提供神经科学依据。**

- **链接: [https://arxiv.org/pdf/2604.04025](https://arxiv.org/pdf/2604.04025)**

> **作者:** Shaad Sufi
>
> **备注:** IEEE-style preprint; 4 figures; 4 tables
>
> **摘要:** Background music shapes attention, affect, and approach behavior in commercial environments, yet the neural plausibility of AI-generated music for such settings remains poorly characterized. We present an in-silico pilot study that combines Wubble, a generative music system, with TRIBE v2, a publicly released whole-brain encoding model, to estimate cortical response profiles for prompt-conditioned retail music. Five fully instrumental tracks were generated to span low-to-high arousal, sparse-to-dense arrangement, and neutral-to-positive valence prompts, then analyzed with audio-only TRIBE v2 inference on loudness-normalized waveforms. Analysis focused on fsaverage5 cortical predictions summarized over auditory, superior temporal, temporo-parietal, and inferior frontal HCP parcels. The fast bright major-pop condition produced the largest whole-cortex mean activation (0.0402), the strongest prefrontal ROI composite response (0.0704), and the highest parcel means in IFJa (0.1102), IFJp (0.0995), A5 (0.0188), and area 45 (0.0015). Pairwise spatial correlations ranged from 0.787 to 0.974, indicating that prompt variation modulated predicted cortical states rather than yielding a single undifferentiated response profile. Predicted cortical surface maps further revealed visually distinct spatial organization between low-arousal and high-arousal conditions. These results support a cautious claim of cortical neurological plausibility: prompt-conditioned AI music can systematically shift predicted auditory-temporal-prefrontal patterns relevant to salience and valuation. Although the study does not establish subcortical reward engagement or consumer behavior, it provides a reproducible framework for neural pre-screening and pre-optimization of commercial music generation against biologically informed cortical proxies.
>
---
#### [new 013] A Systematic Study of Cross-Modal Typographic Attacks on Audio-Visual Reasoning
- **分类: cs.CV; cs.SD**

- **简介: 该论文研究跨模态排版攻击对音视频推理的影响，属于多模态安全任务，旨在揭示多模态大模型的脆弱性。工作包括分析多模态干扰交互，验证协同攻击的有效性。**

- **链接: [https://arxiv.org/pdf/2604.03995](https://arxiv.org/pdf/2604.03995)**

> **作者:** Tianle Chen; Deepti Ghadiyaram
>
> **摘要:** As audio-visual multi-modal large language models (MLLMs) are increasingly deployed in safety-critical applications, understanding their vulnerabilities is crucial. To this end, we introduce Multi-Modal Typography, a systematic study examining how typographic attacks across multiple modalities adversely influence MLLMs. While prior work focuses narrowly on unimodal attacks, we expose the cross-modal fragility of MLLMs. We analyze the interactions between audio, visual, and text perturbations and reveal that coordinated multi-modal attack creates a significantly more potent threat than single-modality attacks (attack success rate = $83.43\%$ vs $34.93\%$).Our findings across multiple frontier MLLMs, tasks, and common-sense reasoning and content moderation benchmarks establishes multi-modal typography as a critical and underexplored attack strategy in multi-modal reasoning. Code and data will be publicly available.
>
---
#### [new 014] FlueBricks: A Construction Kit of Flute-like Instruments for Acoustic Reasoning
- **分类: cs.HC; cs.SD**

- **简介: 论文介绍FlueBricks，一个用于声学推理的管乐器构建套件，通过实验帮助用户理解声学原理。属于声学教育任务，解决如何通过动手实践提升声学理解的问题。**

- **链接: [https://arxiv.org/pdf/2604.03636](https://arxiv.org/pdf/2604.03636)**

> **作者:** Bo-Yu Chen; Chiao-Wei Huang; Lung-Pan Cheng
>
> **备注:** Accepted to CHI 2026
>
> **摘要:** We present FlueBricks, a construction kit for acoustic reasoning via building and customizing flute-like instruments. By assembling generator, resonator, and connector modules that embody various aeroacoustic properties, users gain deeper understanding of how blowhole, tube length, and tone-hole placement alter onset, pitch, and timbre through hands-on experimentation. This forms a designer-player loop of configuring and playing to form, test, and refine acoustic behaviors-acoustic reasoning-shifting acoustic instruments from static artifacts to dynamic systems. To understand how users engage with this system, we conducted an exploratory study with 12 participants ranging from novices to professional musicians. During their explorations, we observed participants fluently switching between designer and player roles, scaffolding designs from familiar instruments, forming and refining their acoustic understanding of length, tone holes, and generator geometry, reinterpreting modules beyond their intended functions, and using their creations for performative acts such as pedagogical showing and musical expression. These collectively demonstrated FlueBricks's potential as a pedagogical tool for embodied acoustic reasoning.
>
---
#### [new 015] CoLoRSMamba: Conditional LoRA-Steered Mamba for Supervised Multimodal Violence Detection
- **分类: cs.CV; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出CoLoRSMamba，用于监督式多模态暴力检测任务，解决音频与视频信息不匹配的问题，通过CLoS引导的LoRA机制融合视频和音频特征，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.03329](https://arxiv.org/pdf/2604.03329)**

> **作者:** Damith Chamalke Senadeera; Dimitrios Kollias; Gregory Slabaugh
>
> **摘要:** Violence detection benefits from audio, but real-world soundscapes can be noisy or weakly related to the visible scene. We present CoLoRSMamba, a directional Video to Audio multimodal architecture that couples VideoMamba and AudioMamba through CLS-guided conditional LoRA. At each layer, the VideoMamba CLS token produces a channel-wise modulation vector and a stabilization gate that adapt the AudioMamba projections responsible for the selective state-space parameters (Delta, B, C), including the step-size pathway, yielding scene-aware audio dynamics without token-level cross-attention. Training combines binary classification with a symmetric AV-InfoNCE objective that aligns clip-level audio and video embeddings. To support fair multimodal evaluation, we curate audio-filtered clip level subsets of the NTU-CCTV and DVD datasets from temporal annotations, retaining only clips with available audio. On these subsets, CoLoRSMamba outperforms representative audio-only, video-only, and multimodal baselines, achieving 88.63% accuracy / 86.24% F1-V on NTU-CCTV and 75.77% accuracy / 72.94% F1-V on DVD. It further offers a favorable accuracy-efficiency tradeoff, surpassing several larger models with fewer parameters and FLOPs.
>
---
## 更新

#### [replaced 001] Noise-Robust Contrastive Learning with an MFCC-Conformer For Coronary Artery Disease Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于心血管疾病检测任务，旨在提升心音信号中冠心病识别的噪声鲁棒性。通过多通道噪声段剔除和MFCC-Conformer模型，提高真实环境下的检测准确率。**

- **链接: [https://arxiv.org/pdf/2601.18295](https://arxiv.org/pdf/2601.18295)**

> **作者:** Milan Marocchi; Matthew Fynn; Yue Rong
>
> **备注:** This paper has been accepted for presentation at ICASSP 2026. \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. 5 pages, 1 figure
>
> **摘要:** Cardiovascular diseases (CVD) are the leading cause of death worldwide, with coronary artery disease (CAD) comprising the largest subcategory of CVDs. Recently, there has been increased focus on detecting CAD using phonocardiogram (PCG) signals, with high success in clinical environments with low noise and optimal sensor placement. Multichannel techniques have been found to be more robust to noise; however, achieving robust performance on real-world data remains a challenge. This work utilises a novel multichannel energy-based noisy-segment rejection algorithm, using heart and noise-reference microphones, to discard audio segments with large amounts of nonstationary noise before training a deep learning classifier. This conformer-based classifier takes mel-frequency cepstral coefficients (MFCCs) from multiple channels, further helping improve the model's noise robustness. The proposed method achieved 78.4% accuracy and 78.2% balanced accuracy on 297 subjects, representing improvements of 4.1% and 4.3%, respectively, compared to training without noisy-segment rejection.
>
---
#### [replaced 002] PhiNet: Speaker Verification with Phonetic Interpretability
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出PhiNet，用于说话人验证任务，解决传统系统缺乏透明性的问题。通过引入语音学解释，提升决策的可解释性，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.01590](https://arxiv.org/pdf/2604.01590)**

> **作者:** Yi Ma; Shuai Wang; Tianchi Liu; Haizhou Li
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing. Codes: this https URL
>
> **摘要:** Despite remarkable progress, automatic speaker verification (ASV) systems typically lack the transparency required for high-accountability applications. Motivated by how human experts perform forensic speaker comparison (FSC), we propose a speaker verification network with phonetic interpretability, PhiNet, designed to enhance both local and global interpretability by leveraging phonetic evidence in decision-making. For users, PhiNet provides detailed phonetic-level comparisons that enable manual inspection of speaker-specific features and facilitate a more critical evaluation of verification outcomes. For developers, it offers explicit reasoning behind verification decisions, simplifying error tracing and informing hyperparameter selection. In our experiments, we demonstrate PhiNet's interpretability with practical examples, including its application in analyzing the impact of different hyperparameters. We conduct both qualitative and quantitative evaluations of the proposed interpretability methods and assess speaker verification performance across multiple benchmark datasets, including VoxCeleb, SITW, and LibriSpeech. Results show that PhiNet achieves performance comparable to traditional black-box ASV models while offering meaningful, interpretable explanations for its decisions, bridging the gap between ASV and forensic analysis.
>
---
#### [replaced 003] Compact Hypercube Embeddings for Fast Text-based Wildlife Observation Retrieval
- **分类: cs.IR; cs.CV; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于文本驱动的野生动物观测检索任务，旨在解决大规模数据中高效检索的问题。通过构建紧凑的二进制超立方体嵌入，实现快速搜索。**

- **链接: [https://arxiv.org/pdf/2601.22783](https://arxiv.org/pdf/2601.22783)**

> **作者:** Ilyass Moummad; Marius Miron; David Robinson; Kawtar Zaher; Hervé Goëau; Olivier Pietquin; Pierre Bonnet; Emmanuel Chemla; Matthieu Geist; Alexis Joly
>
> **摘要:** Large-scale biodiversity monitoring platforms increasingly rely on multimodal wildlife observations. While recent foundation models enable rich semantic representations across vision, audio, and language, retrieving relevant observations from massive archives remains challenging due to the computational cost of high-dimensional similarity search. In this work, we introduce compact hypercube embeddings for fast text-based wildlife observation retrieval, a framework that enables efficient text-based search over large-scale wildlife image and audio databases using compact binary representations. Building on the cross-view code alignment hashing framework, we extend lightweight hashing beyond a single-modality setup to align natural language descriptions with visual or acoustic observations in a shared Hamming space. Our approach leverages pretrained wildlife foundation models, including BioCLIP and BioLingual, and adapts them efficiently for hashing using parameter-efficient fine-tuning. We evaluate our method on large-scale benchmarks, including iNaturalist2024 for text-to-image retrieval and iNatSounds2024 for text-to-audio retrieval, as well as multiple soundscape datasets to assess robustness under domain shift. Results show that retrieval using discrete hypercube embeddings achieves competitive, and in several cases superior, performance compared to continuous embeddings, while drastically reducing memory and search cost. Moreover, we observe that the hashing objective consistently improves the underlying encoder representations, leading to stronger retrieval and zero-shot generalization. These results demonstrate that binary, language-based retrieval enables scalable and efficient search over large wildlife archives for biodiversity monitoring systems.
>
---
#### [replaced 004] Validating Computational Markers of Depressive Behavior: Cross-Linguistic Speech-Based Depression Detection with Neurophysiological Validation
- **分类: eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在验证跨语言的语音特征及神经生理基础。通过扩展CDMA框架，结合情绪语音与EEG数据，提升检测性能并建立神经生理验证。**

- **链接: [https://arxiv.org/pdf/2604.01533](https://arxiv.org/pdf/2604.01533)**

> **作者:** Fuxiang Tao; Dongwei Li; Shuning Tang; Xuri Ge; Wei Ma; Anna Esposito; Alessandro Vinciarelli
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Speech-based depression detection has shown promise as an objective diagnostic tool, yet the cross-linguistic robustness of acoustic markers and their neurobiological underpinnings remain underexplored. This study extends Cross-Data Multilevel Attention (CDMA) framework, initially validated on Italian, to investigate these dimensions using a Chinese Mandarin dataset with Electroencephalography (EEG) recordings. We systematically fuse read speech with spontaneous speech across different emotional valences (positive, neutral, negative) to investigate whether emotional arousal is a more critical factor than valence polarity in enhancing detection performance in speech. Additionally, we establish the first neurophysiological validation for a speech-based depression model by correlating its predictions with neural oscillatory patterns during emotional face processing. Our results demonstrate strong cross-linguistic generalizability of the CDMA framework, achieving state-of-the-art performance (F1-score up to 89.6%) on the Chinese dataset, which is comparable to the previous Italian validation. Critically, emotionally valenced speech (both positive and negative) significantly outperformed neutral speech. This comparable performance between positive and negative tasks supports the emotional arousal hypothesis. Most importantly, EEG analysis revealed significant correlations between the model's speech-derived depression estimates and neural oscillatory patterns (theta and alpha bands), demonstrating alignment with established neural markers of emotional dysregulation in depression. This alignment, combined with the model's cross-linguistic robustness, not only supports that the CDMA framework's approach is a universally applicable and neurobiologically validated strategy but also establishes a novel paradigm for the neurophysiological validation of computational mental health models.
>
---
#### [replaced 005] IQRA 2026: Interspeech Challenge on Automatic Pronunciation Assessment for Modern Standard Arabic (MSA)
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语发音评估任务，旨在解决自动误发音检测与诊断问题。通过引入新数据集和多种模型方法，提升了系统性能。**

- **链接: [https://arxiv.org/pdf/2603.29087](https://arxiv.org/pdf/2603.29087)**

> **作者:** Yassine El Kheir; Amit Meghanani; Mostafa Shahin; Omnia Ibrahim; Shammur Absar Chowdhury; Nada AlMarwani; Youssef Elshahawy; Ahmed Ali
>
> **备注:** 5 pages paper
>
> **摘要:** We present the findings of the second edition of the IQRA Interspeech Challenge, a challenge on automatic Mispronunciation Detection and Diagnosis (MDD) for Modern Standard Arabic (MSA). Building on the previous edition, this iteration introduces \textbf{Iqra\_Extra\_IS26}, a new dataset of authentic human mispronounced speech, complementing the existing training and evaluation resources. Submitted systems employed a diverse range of approaches, spanning CTC-based self-supervised learning models, two-stage fine-tuning strategies, and using large audio-language models. Compared to the first edition, we observe a substantial jump of \textbf{0.28 in F1-score}, attributable both to novel architectures and modeling strategies proposed by participants and to the additional authentic mispronunciation data made available. These results demonstrate the growing maturity of Arabic MDD research and establish a stronger foundation for future work in Arabic pronunciation assessment.
>
---
#### [replaced 006] Audio-to-Image Bird Species Retrieval without Audio-Image Pairs via Text Distillation
- **分类: cs.SD; cs.IR; cs.LG**

- **简介: 该论文属于音频到图像的物种检索任务，解决缺乏配对数据的问题。通过文本中介实现音频与图像表征对齐，提升检索性能。**

- **链接: [https://arxiv.org/pdf/2602.00681](https://arxiv.org/pdf/2602.00681)**

> **作者:** Ilyass Moummad; Marius Miron; Lukas Rauch; David Robinson; Alexis Joly; Olivier Pietquin; Emmanuel Chemla; Matthieu Geist
>
> **摘要:** Audio-to-image retrieval offers an interpretable alternative to audio-only classification for bioacoustic species recognition, but learning aligned audio-image representations is challenging due to the scarcity of paired audio-image data. We propose a simple and data-efficient approach that enables audio-to-image retrieval without any audio-image supervision. Our proposed method uses text as a semantic intermediary: we distill the text embedding space of a pretrained image-text model (BioCLIP-2), which encodes rich visual and taxonomic structure, into a pretrained audio-text model (BioLingual) by fine-tuning its audio encoder with a contrastive objective. This distillation transfers visually grounded semantics into the audio representation, inducing emergent alignment between audio and image embeddings without using images during training. We evaluate the resulting model on multiple bioacoustic benchmarks. The distilled audio encoder preserves audio discriminative power while substantially improving audio-text alignment on focal recordings and soundscape datasets. Most importantly, on the SSW60 benchmark, the proposed approach achieves strong audio-to-image retrieval performance exceeding baselines based on zero-shot model combinations or learned mappings between text embeddings, despite not training on paired audio-image data. These results demonstrate that indirect semantic transfer through text is sufficient to induce meaningful audio-image alignment, providing a practical solution for visually grounded species recognition in data-scarce bioacoustic settings.
>
---
#### [replaced 007] WhisperRT -- Turning Whisper into a Causal Streaming Model
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决传统模型不适用于实时流式 transcription 的问题。通过改造 Whisper 模型，使其具备低延迟的流式处理能力。**

- **链接: [https://arxiv.org/pdf/2508.12301](https://arxiv.org/pdf/2508.12301)**

> **作者:** Tomer Krichli; Bhiksha Raj; Joseph Keshet
>
> **备注:** 14 pages, 7 Figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Automatic Speech Recognition (ASR) has seen remarkable progress, with models like OpenAI Whisper and NVIDIA Canary achieving state-of-the-art (SOTA) performance in offline transcription. However, these models are not designed for streaming (online or real-time) transcription, due to limitations in their architecture and training methodology. We propose a method to turn the transformer encoder-decoder model into a low-latency streaming model. The encoder is made causal to process audio incrementally, while the decoder conditions on partial encoder states to generate tokens aligned with the available temporal context. This requires explicit synchronization between encoded input frames and token emissions. Since tokens are produced only after sufficient acoustic evidence is observed, an inherent latency arises, necessitating fine-tuning of the encoder-decoder alignment mechanism. We propose an updated inference mechanism that utilizes the fine-tuned causal encoder and decoder to yield greedy and beam-search decoding, and is shown to be locally optimal. Experiments on low-latency chunk sizes (less than 300 msec) show that our fine-tuned model outperforms existing non-fine-tuned streaming approaches in most cases, while using a lower complexity. We release our training and inference code, along with the fine-tuned models, to support further research and development in streaming ASR.
>
---
#### [replaced 008] When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音 spoof 检测任务，旨在解决跨语言检测鲁棒性问题。通过构建多语言数据集，评估不同模型在66种语言中的检测效果。**

- **链接: [https://arxiv.org/pdf/2603.02364](https://arxiv.org/pdf/2603.02364)**

> **作者:** Kirill Borodin; Vasiliy Kudryavtsev; Maxim Maslov; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** This paper has been submitted to Interspeech 2026 for review
>
> **摘要:** We introduce LRLspoof, a large-scale multilingual synthetic-speech corpus for cross-lingual spoof detection, comprising 2,732 hours of audio generated with 24 open-source TTS systems across 66 languages, including 45 low-resource languages under our operational definition. To evaluate robustness without requiring target-domain bonafide speech, we benchmark 11 publicly available countermeasures using threshold transfer: for each model we calibrate an EER operating point on pooled external benchmarks and apply the resulting threshold, reporting spoof rejection rate (SRR). Results show model-dependent cross-lingual disparity, with spoof rejection varying markedly across languages even under controlled conditions, highlighting language as an independent source of domain shift in spoof detection. The dataset is publicly available at \href{this https URL}{\textbf{\underline{\textit{HuggingFace}}}} and \href{this https URL}{\textbf{\underline{\textit{ModelScope}}}}
>
---
#### [replaced 009] SenSE: Semantic-Aware High-Fidelity Universal Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决生成模型输出语义不一致的问题。提出SenSE框架，结合语言模型和流匹配方法，提升语音的语义保真度和适应性。**

- **链接: [https://arxiv.org/pdf/2509.24708](https://arxiv.org/pdf/2509.24708)**

> **作者:** Xingchen Li; Hanke Xie; Ziqian Wang; Zihan Zhang; Longshuai Xiao; Shuai Wang; Lei Xie
>
> **备注:** Accepted by ICME 2026
>
> **摘要:** Generative Universal Speech Enhancement (USE) methods aim to leverage generative models to improve speech quality under various types of distortions. However, existing generative speech enhancement methods often suffer from semantic inconsistency in the generated outputs. Therefore, we propose SenSE, a novel two-stage generative universal speech enhancement framework, by modeling semantic priors with a language model, the flow matching-based speech enhancement process is guided to generate semantically faithful speech, thereby effectively improving context fidelity. In addition, we introduce a dual-path masked conditioning training strategy that enables flow matching-based enhancement to flexibly integrate multi-source conditioning signals from degraded speech, semantic tokens, and reference speech, thereby improving model flexibility and adaptability. Experimental results demonstrate that SenSE achieves state-of-the-art performance among generative speech enhancement models and exhibits a high performance ceiling, particularly under challenging distortion conditions. Codes and demos are available at this https URL.
>
---
#### [replaced 010] VABench: A Comprehensive Benchmark for Audio-Video Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出VABench，一个用于评估音视频生成的基准框架，解决现有评测缺乏音频-视频同步性的问题，涵盖多种任务和评估维度。**

- **链接: [https://arxiv.org/pdf/2512.09299](https://arxiv.org/pdf/2512.09299)**

> **作者:** Daili Hua; Xizhi Wang; Bohan Zeng; Xinyi Huang; Hao Liang; Junbo Niu; Xinlong Chen; Quanqing Xu; Wentao Zhang
>
> **备注:** 24 pages, 25 figures
>
> **摘要:** Recent advances in video generation have been remarkable, enabling models to produce visually compelling videos with synchronized audio. While existing video generation benchmarks provide comprehensive metrics for visual quality, they lack convincing evaluations for audio-video generation, especially for models aiming to generate synchronized audio-video outputs. To address this gap, we introduce VABench, a comprehensive and multi-dimensional benchmark framework designed to systematically evaluate the capabilities of synchronous audio-video generation. VABench encompasses three primary task types: text-to-audio-video (T2AV), image-to-audio-video (I2AV), and stereo audio-video generation. It further establishes two major evaluation modules covering 15 dimensions. These dimensions specifically assess pairwise similarities (text-video, text-audio, video-audio), audio-video synchronization, lip-speech consistency, and carefully curated audio and video question-answering (QA) pairs, among others. Furthermore, VABench covers seven major content categories: animals, human sounds, music, environmental sounds, synchronous physical sounds, complex scenes, and virtual worlds. We provide a systematic analysis and visualization of the evaluation results, aiming to establish a new standard for assessing video generation models with synchronous audio capabilities and to promote the comprehensive advancement of the field.
>
---
