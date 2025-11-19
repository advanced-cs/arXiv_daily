# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Count The Notes: Histogram-Based Supervision for Automatic Music Transcription
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于自动音乐转录（AMT）任务，旨在解决强对齐标注数据稀缺的问题。提出CountEM框架，利用音符事件直方图作为监督信号，通过EM算法迭代优化，无需局部对齐即可实现高效准确的转录。**

- **链接: [https://arxiv.org/pdf/2511.14250v1](https://arxiv.org/pdf/2511.14250v1)**

> **作者:** Jonathan Yaffe; Ben Maman; Meinard Müller; Amit H. Bermano
>
> **备注:** ISMIR 2025
>
> **摘要:** Automatic Music Transcription (AMT) converts audio recordings into symbolic musical representations. Training deep neural networks (DNNs) for AMT typically requires strongly aligned training pairs with precise frame-level annotations. Since creating such datasets is costly and impractical for many musical contexts, weakly aligned approaches using segment-level annotations have gained traction. However, existing methods often rely on Dynamic Time Warping (DTW) or soft alignment loss functions, both of which still require local semantic correspondences, making them error-prone and computationally expensive. In this article, we introduce CountEM, a novel AMT framework that eliminates the need for explicit local alignment by leveraging note event histograms as supervision, enabling lighter computations and greater flexibility. Using an Expectation-Maximization (EM) approach, CountEM iteratively refines predictions based solely on note occurrence counts, significantly reducing annotation efforts while maintaining high transcription accuracy. Experiments on piano, guitar, and multi-instrument datasets demonstrate that CountEM matches or surpasses existing weakly supervised methods, improving AMT's robustness, scalability, and efficiency. Our project page is available at https://yoni-yaffe.github.io/count-the-notes.
>
---
#### [new 002] Audio Question Answering with GRPO-Based Fine-Tuning and Calibrated Segment-Level Predictions
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对音频问答任务，提出结合BEATs特征提取与GRPO微调的Qwen模型，通过校准段级预测生成事件级答案，提升准确率至62.6%。**

- **链接: [https://arxiv.org/pdf/2511.14307v1](https://arxiv.org/pdf/2511.14307v1)**

> **作者:** Marcel Gibier; Nolwenn Celton; Raphaël Duroselle; Pierre Serrano; Olivier Boeffard; Jean-François Bonastre
>
> **备注:** Submission to Track 5 of the DCASE 2025 Challenge
>
> **摘要:** In this report, we describe our submission to Track 5 of the DCASE 2025 Challenge for the task of Audio Question Answering(AQA). Our system leverages the SSL backbone BEATs to extract frame-level audio features, which are then processed by a classification head to generate segment-level predictions of acoustic events, following the Audioset ontology. These segment-level predictions are subsequently calibrated before producing event-level predictions. Finally, these predictions are incorporated into a structured prompt, along with the question and candidate answers. This prompt is then fed to a fine-tuned version of Qwen2.5-7B-Instruct, trained using the GRPO algorithm with a simple reward function. Our method achieves an accuracy of 62.6 % on the development set, demonstrating the effectiveness of combining acoustic event reasoning with instruction-tuned large language models for AQA.
>
---
#### [new 003] Segmentwise Pruning in Audio-Language Models
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究音频语言模型中的分段剪枝方法，旨在降低长音频输入带来的计算成本。通过引入考虑时间维度的轻量级剪枝策略，在保留四分之一token的情况下，仅带来最小性能损失，有效提升了模型效率。**

- **链接: [https://arxiv.org/pdf/2511.14293v1](https://arxiv.org/pdf/2511.14293v1)**

> **作者:** Marcel Gibier; Raphaël Duroselle; Pierre Serrano; Olivier Boeffard; Jean-François Bonastre
>
> **备注:** Submitted to ICASSP 2026 (under review)
>
> **摘要:** Recent audio-language models have shown impressive performance across a wide range of audio tasks and are increasingly capable of handling long audio inputs. However, the computing costs in these models heavily depend on sequence length, which can become very large given the nature of audio data. In the vision-language domain, token pruning methods have proven effective in reducing token counts while preserving strong performance on standard benchmarks. In this work, we investigate the relevance and effectiveness of such token selection strategies in the context of audio-language models. We also improve them by proposing a lightweight strategy that takes the time dimension into account. While retaining only a quarter of the initial tokens, our approach results in a relative maximum decrease of 2% in CIDEr on Clotho v2 and a relative maximum decrease of 4% in accuracy on MMAU.
>
---
#### [new 004] A Controllable Perceptual Feature Generative Model for Melody Harmonization via Conditional Variational Autoencoder
- **分类: cs.SD**

- **简介: 该论文提出CPFG-Net模型，用于旋律和声化任务，解决音乐生成中缺乏新颖性和表现力的问题。通过条件变分自编码器建模感知特征与和弦的映射关系，实现可控的和弦生成，提升音乐创造力与表达力。**

- **链接: [https://arxiv.org/pdf/2511.14600v1](https://arxiv.org/pdf/2511.14600v1)**

> **作者:** Dengyun Huang; Yonghua Zhu
>
> **备注:** 13 pages, 8 figures, 2 url links
>
> **摘要:** While Large Language Models (LLMs) make symbolic music generation increasingly accessible, producing music with distinctive composition and rich expressiveness remains a significant challenge. Many studies have introduced emotion models to guide the generative process. However, these approaches still fall short of delivering novelty and creativity. In the field of Music Information Retrieval (MIR), auditory perception is recognized as a key dimension of musical experience, offering insights into both compositional intent and emotional patterns. To this end, we propose a neural network named CPFG-Net, along with a transformation algorithm that maps perceptual feature values to chord representations, enabling melody harmonization. The system can controllably predict sequences of perceptual features and tonal structures from given melodies, and subsequently generate harmonically coherent chord progressions. Our network is trained on our newly constructed perceptual feature dataset BCPT-220K, derived from classical music. Experimental results show state-of-the-art perceptual feature prediction capability of our model as well as demonstrate our musical expressiveness and creativity in chord inference. This work offers a novel perspective on melody harmonization and contributes to broader music generation tasks. Our symbolic-based model can be easily extended to audio-based models.
>
---
#### [new 005] Emotion Recognition in Multi-Speaker Conversations through Speaker Identification, Knowledge Distillation, and Hierarchical Fusion
- **分类: cs.SD; eess.AS**

- **简介: 论文聚焦多说话人对话中的情感识别任务，解决说话人混淆和类别不平衡问题。提出结合说话人识别、知识蒸馏与分层融合的框架，提升模型对少数情绪类别的识别性能。**

- **链接: [https://arxiv.org/pdf/2511.13731v1](https://arxiv.org/pdf/2511.13731v1)**

> **作者:** Xiao Li; Kotaro Funakoshi; Manabu Okumura
>
> **摘要:** Emotion recognition in multi-speaker conversations faces significant challenges due to speaker ambiguity and severe class imbalance. We propose a novel framework that addresses these issues through three key innovations: (1) a speaker identification module that leverages audio-visual synchronization to accurately identify the active speaker, (2) a knowledge distillation strategy that transfers superior textual emotion understanding to audio and visual modalities, and (3) hierarchical attention fusion with composite loss functions to handle class imbalance. Comprehensive evaluations on MELD and IEMOCAP datasets demonstrate superior performance, achieving 67.75% and 72.44% weighted F1 scores respectively, with particularly notable improvements on minority emotion classes.
>
---
#### [new 006] IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文属于语音增强任务，旨在解决资源受限设备上模型轻量化与高性能难以兼顾的问题。提出IMSE网络，通过引入Amplitude-Aware Linear Attention和Inception Depthwise Convolution，显著减少参数量（-16.8%）并保持优异性能（PESQ=3.373）。**

- **链接: [https://arxiv.org/pdf/2511.14515v1](https://arxiv.org/pdf/2511.14515v1)**

> **作者:** Xinxin Tang; Bin Qin; Yufang Li
>
> **摘要:** Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex "approximate-compensate" mechanism to mitigate the limitations of Taylor-expansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the "amplitude-ignoring" problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8\% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultra-lightweight speech enhancement.
>
---
#### [new 007] Preference-Based Learning in Audio Applications: A Systematic Analysis
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文研究音频生成任务中的偏好学习，旨在解决音频领域偏好学习应用不足的问题。通过系统性文献综述，分析了500篇论文，发现仅6%涉及音频偏好学习，并揭示其从情感识别向生成任务转变的趋势及多维评估、指标不一致等关键问题。**

- **链接: [https://arxiv.org/pdf/2511.13936v1](https://arxiv.org/pdf/2511.13936v1)**

> **作者:** Aaron Broukhim; Yiran Shen; Prithviraj Ammanabrolu; Nadir Weibel
>
> **摘要:** Despite the parallel challenges that audio and text domains face in evaluating generative model outputs, preference learning remains remarkably underexplored in audio applications. Through a PRISMA-guided systematic review of approximately 500 papers, we find that only 30 (6%) apply preference learning to audio tasks. Our analysis reveals a field in transition: pre-2021 works focused on emotion recognition using traditional ranking methods (rankSVM), while post-2021 studies have pivoted toward generation tasks employing modern RLHF frameworks. We identify three critical patterns: (1) the emergence of multi-dimensional evaluation strategies combining synthetic, automated, and human preferences; (2) inconsistent alignment between traditional metrics (WER, PESQ) and human judgments across different contexts; and (3) convergence on multi-stage training pipelines that combine reward signals. Our findings suggest that while preference learning shows promise for audio, particularly in capturing subjective qualities like naturalness and musicality, the field requires standardized benchmarks, higher-quality datasets, and systematic investigation of how temporal factors unique to audio impact preference learning frameworks.
>
---
#### [new 008] TTA: Transcribe, Translate and Alignment for Cross-lingual Speech Representation
- **分类: eess.AS**

- **简介: 论文提出TTA模型，解决Speech-LLM中Whisper encoder在跨语言语义表示上的不足。通过358k小时多任务训练，TTA在ASR、ST和语音检索等任务上优于Whisper，提升跨语言能力与语音理解性能。**

- **链接: [https://arxiv.org/pdf/2511.14410v1](https://arxiv.org/pdf/2511.14410v1)**

> **作者:** Wei Liu; Jiahong Li; Yiwen Shao; Dong Yu
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** Speech-LLM models have demonstrated great performance in multi-modal and multi-task speech understanding. A typical speech-LLM paradigm is integrating speech modality with a large language model (LLM). While the Whisper encoder was frequently adopted in previous studies for speech input, it shows limitations regarding input format, model scale, and semantic performance. To this end, we propose a lightweight TTA model specialized in speech semantics for more effective LLM integration. With large-scale training of 358k hours of speech data on multilingual speech recognition (ASR), speech translation (ST) and speech-text alignment tasks, TTA is capable of producing robust cross-lingual speech representations. Extensive evaluations across diverse benchmarks, including ASR/ST, speech retrieval, and ASR-LLM performance assessments, demonstrate TTA's superiority over Whisper. Furthermore, we rigorously validate the interplay between cross-lingual capabilities and ASR/ST performance. The model weights and training recipes of TTA will be released as part of an audio understanding toolkit Auden.
>
---
#### [new 009] FxSearcher: gradient-free text-driven audio transformation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出FxSearcher，一种无梯度的文本驱动音频变换框架，旨在解决现有方法受限于可微音频效果的问题。通过贝叶斯优化与CLAP评分函数搜索最优音效配置，并引入引导提示提升质量与人类偏好匹配度。**

- **链接: [https://arxiv.org/pdf/2511.14138v1](https://arxiv.org/pdf/2511.14138v1)**

> **作者:** Hojoon Ki; Jongsuk Kim; Minchan Kwon; Junmo Kim
>
> **摘要:** Achieving diverse and high-quality audio transformations from text prompts remains challenging, as existing methods are fundamentally constrained by their reliance on a limited set of differentiable audio effects. This paper proposes \textbf{FxSearcher}, a novel gradient-free framework that discovers the optimal configuration of audio effects (FX) to transform a source signal according to a text prompt. Our method employs Bayesian Optimization and CLAP-based score function to perform this search efficiently. Furthermore, a guiding prompt is introduced to prevent undesirable artifacts and enhance human preference. To objectively evaluate our method, we propose an AI-based evaluation framework. The results demonstrate that the highest scores achieved by our method on these metrics align closely with human preferences. Demos are available at https://hojoonki.github.io/FxSearcher/
>
---
#### [new 010] Principled Coarse-Grained Acceptance for Speculative Decoding in Speech
- **分类: eess.AS; cs.LG**

- **简介: 论文针对语音大模型生成效率低的问题，提出基于声学相似性的粗粒度接受机制PCG，通过分组验证提高接受率，加速推理同时保持语音质量。**

- **链接: [https://arxiv.org/pdf/2511.13732v1](https://arxiv.org/pdf/2511.13732v1)**

> **作者:** Moran Yanuka; Paul Dixon; Eyal Finkelshtein; Daniel Rotman; Raja Giryes
>
> **摘要:** Speculative decoding accelerates autoregressive speech generation by letting a fast draft model propose tokens that a larger target model verifies. However, for speech LLMs that generate acoustic tokens, exact token matching is overly restrictive: many discrete tokens are acoustically or semantically interchangeable, reducing acceptance rates and limiting speedups. We introduce Principled Coarse-Graining (PCG), which verifies proposals at the level of Acoustic Similarity Groups (ASGs) derived from the target model's embedding space. By splitting each token's probability mass across the overlapping groups that contain it, we define an overlap-aware coarse-grained distribution and perform rejection sampling on the resulting group variable. This yields an exactness guarantee at the group level while allowing the accepted draft token to stand in for any member of the group in practice. On LibriTTS, PCG increases acceptance and throughput relative to standard speculative decoding and prior speech-specific relaxations while maintaining intelligibility and speaker similarity. These results suggest acoustically aware, group-level acceptance as a simple and general way to accelerate speech token generation while maintaining speech quality.
>
---
#### [new 011] Accelerating Automatic Differentiation of Direct Form Digital Filters
- **分类: eess.SY; eess.AS; eess.SP**

- **简介: 论文提出一种加速自动微分的方法，用于直接形式数字滤波器，解决传统方法计算效率低的问题。通过推导闭式反向传播表达式，实现滤波与梯度计算一体化，支持并行化，在GPU上速度提升超1000倍。**

- **链接: [https://arxiv.org/pdf/2511.14390v1](https://arxiv.org/pdf/2511.14390v1)**

> **作者:** Chin-Yun Yu; György Fazekas
>
> **备注:** Accepted at the 1st Workshop on Differentiable Systems and Scientific Machine Learning @ EurIPS 2025
>
> **摘要:** We introduce a general formulation for automatic differentiation through direct form filters, yielding a closed-form backpropagation that includes initial condition gradients. The result is a single expression that can represent both the filter and its gradients computation while supporting parallelism. C++/CUDA implementations in PyTorch achieve at least 1000x speedup over naive Python implementations and consistently run fastest on the GPU. For the low-order filters commonly used in practice, exact time-domain filtering with analytical gradients outperforms the frequency-domain method in terms of speed. The source code is available at https://github.com/yoyolicoris/philtorch.
>
---
#### [new 012] Subject-Independent Imagined Speech Detection via Cross-Subject Generalization and Calibration
- **分类: q-bio.NC; cs.AI; cs.SD**

- **简介: 论文研究脑机接口中跨被试想象言语识别问题，提出循环跨被试训练与少量校准结合的方法，提升模型泛化能力与个性化适应，实现高效、可扩展的解码系统。**

- **链接: [https://arxiv.org/pdf/2511.13739v1](https://arxiv.org/pdf/2511.13739v1)**

> **作者:** Byung-Kwan Ko; Soowon Kim; Seo-Hyun Lee
>
> **备注:** 4 pages, 2 figures, Name of Conference: International Conference on Brain-Computer Interface
>
> **摘要:** Achieving robust generalization across individuals remains a major challenge in electroencephalogram based imagined speech decoding due to substantial variability in neural activity patterns. This study examined how training dynamics and lightweight subject specific adaptation influence cross subject performance in a neural decoding framework. A cyclic inter subject training approach, involving shorter per subject training segments and frequent alternation among subjects, led to modest yet consistent improvements in decoding performance across unseen target data. Furthermore, under the subject calibrated leave one subject out scheme, incorporating only 10 % of the target subjects data for calibration achieved an accuracy of 0.781 and an AUC of 0.801, demonstrating the effectiveness of few shot adaptation. These findings suggest that integrating cyclic training with minimal calibration provides a simple and effective strategy for developing scalable, user adaptive brain computer interface systems that balance generalization and personalization.
>
---
#### [new 013] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation
- **分类: cs.AI; cs.SD**

- **简介: 该论文针对语音识别中的幻觉问题，提出两阶段方法：通过自适应层注意力增强编码鲁棒性，再用多目标知识蒸馏使模型在噪声下更准确。旨在提升Whisper模型在真实噪声环境下的可靠性。**

- **链接: [https://arxiv.org/pdf/2511.14219v1](https://arxiv.org/pdf/2511.14219v1)**

> **作者:** Kumud Tripathi; Aditya Srinivas Menon; Aman Gaurav; Raj Prakash Gohil; Pankaj Wasnik
>
> **备注:** Accepted at AAAI 2026 - Main Technical Track
>
> **摘要:** The Whisper model, an open-source automatic speech recognition system, is widely adopted for its strong performance across multilingual and zero-shot settings. However, it frequently suffers from hallucination errors, especially under noisy acoustic conditions. Previous works to reduce hallucinations in Whisper-style ASR systems have primarily focused on audio preprocessing or post-processing of transcriptions to filter out erroneous content. However, modifications to the Whisper model itself remain largely unexplored to mitigate hallucinations directly. To address this challenge, we present a two-stage architecture that first enhances encoder robustness through Adaptive Layer Attention (ALA) and further suppresses hallucinations using a multi-objective knowledge distillation (KD) framework. In the first stage, ALA groups encoder layers into semantically coherent blocks via inter-layer correlation analysis. A learnable multi-head attention module then fuses these block representations, enabling the model to jointly exploit low- and high-level features for more robust encoding. In the second stage, our KD framework trains the student model on noisy audio to align its semantic and attention distributions with a teacher model processing clean inputs. Our experiments on noisy speech benchmarks show notable reductions in hallucinations and word error rates, while preserving performance on clean speech. Together, ALA and KD offer a principled strategy to improve Whisper's reliability under real-world noisy conditions.
>
---
#### [new 014] Segmenting Collision Sound Sources in Egocentric Videos
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 论文提出碰撞声音源分割（CS3）任务，旨在根据音频在第一人称视频中定位引发碰撞声的物体。针对视觉 clutter 和短时交互挑战，作者提出弱监督方法，结合CLIP、SAM2及手部物体线索，显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.13863v1](https://arxiv.org/pdf/2511.13863v1)**

> **作者:** Kranti Kumar Parida; Omar Emara; Hazel Doughty; Dima Damen
>
> **备注:** Under Review. Webpage: https://krantiparida.github.io/projects/cs3.html
>
> **摘要:** Humans excel at multisensory perception and can often recognise object properties from the sound of their interactions. Inspired by this, we propose the novel task of Collision Sound Source Segmentation (CS3), where we aim to segment the objects responsible for a collision sound in visual input (i.e. video frames from the collision clip), conditioned on the audio. This task presents unique challenges. Unlike isolated sound events, a collision sound arises from interactions between two objects, and the acoustic signature of the collision depends on both. We focus on egocentric video, where sounds are often clear, but the visual scene is cluttered, objects are small, and interactions are brief. To address these challenges, we propose a weakly-supervised method for audio-conditioned segmentation, utilising foundation models (CLIP and SAM2). We also incorporate egocentric cues, i.e. objects in hands, to find acting objects that can potentially be collision sound sources. Our approach outperforms competitive baselines by $3\times$ and $4.7\times$ in mIoU on two benchmarks we introduce for the CS3 task: EPIC-CS3 and Ego4D-CS3.
>
---
## 更新

#### [replaced 001] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.08252v3](https://arxiv.org/pdf/2511.08252v3)**

> **作者:** Yi Yang; Haowen Li; Tianxiang Li; Boyu Cao; Xiaohan Zhang; Liqun Chen; Qi Liu
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Text-to-music generation technology is progressing rapidly, creating new opportunities for musical composition and editing. However, existing music editing methods often fail to preserve the source music's temporal structure, including melody and rhythm, when altering particular attributes like instrument, genre, and mood. To address this challenge, this paper conducts an in-depth probing analysis on attention maps within AudioLDM 2, a diffusion-based model commonly used as the backbone for existing music editing methods. We reveal a key finding: cross-attention maps encompass details regarding distinct musical characteristics, and interventions on these maps frequently result in ineffective modifications. In contrast, self-attention maps are essential for preserving the temporal structure of the source music during its conversion into the target music. Building upon this understanding, we present Melodia, a training-free technique that selectively manipulates self-attention maps in particular layers during the denoising process and leverages an attention repository to store source music information, achieving accurate modification of musical characteristics while preserving the original structure without requiring textual descriptions of the source music. Additionally, we propose two novel metrics to better evaluate music editing methods. Both objective and subjective experiments demonstrate that our approach achieves superior results in terms of textual adherence and structural integrity across various datasets. This research enhances comprehension of internal mechanisms within music generation models and provides improved control for music creation.
>
---
#### [replaced 002] Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems
- **分类: cs.SD; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.07677v2](https://arxiv.org/pdf/2509.07677v2)**

> **作者:** Kamel Kamel; Hridoy Sankar Dutta; Keshav Sood; Sunil Aryal
>
> **备注:** I found a problem in methodology which reflects on the results. So, I want to withdraw it and I am gonna submit another one
>
> **摘要:** Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.
>
---
#### [replaced 003] Systematic Evaluation of Time-Frequency Features for Binaural Sound Source Localization
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.13487v2](https://arxiv.org/pdf/2511.13487v2)**

> **作者:** Davoud Shariat Panah; Alessandro Ragano; Dan Barry; Jan Skoglund; Andrew Hines
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** This study presents a systematic evaluation of time-frequency feature design for binaural sound source localization (SSL), focusing on how feature selection influences model performance across diverse conditions. We investigate the performance of a convolutional neural network (CNN) model using various combinations of amplitude-based features (magnitude spectrogram, interaural level difference - ILD) and phase-based features (phase spectrogram, interaural phase difference - IPD). Evaluations on in-domain and out-of-domain data with mismatched head-related transfer functions (HRTFs) reveal that carefully chosen feature combinations often outperform increases in model complexity. While two-feature sets such as ILD + IPD are sufficient for in-domain SSL, generalization to diverse content requires richer inputs combining channel spectrograms with both ILD and IPD. Using the optimal feature sets, our low-complexity CNN model achieves competitive performance. Our findings underscore the importance of feature design in binaural SSL and provide practical guidance for both domain-specific and general-purpose localization.
>
---
#### [replaced 004] TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech
- **分类: cs.HC; cs.MM; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.05817v2](https://arxiv.org/pdf/2511.05817v2)**

> **作者:** Weiyan Shi; Sunaya Upadhyay; Geraldine Quek; Kenny Tsu Wei Choo
>
> **备注:** Accepted at AAAI 2026 Workshop on Creative AI for Live Interactive Performances (CLIP). To be published in Springer CCIS series
>
> **摘要:** Sketching is a widely used medium for generating and exploring early-stage design concepts. While generative AI (GenAI) chatbots are increasingly used for idea generation, designers often struggle to craft effective prompts and find it difficult to express evolving visual concepts through text alone. In the formative study (N=6), we examined how designers use GenAI during ideation, revealing that text-based prompting disrupts creative flow. To address these issues, we developed TalkSketch, an embedded multimodal AI sketching system that integrates freehand drawing with real-time speech input. TalkSketch aims to support a more fluid ideation process through capturing verbal descriptions during sketching and generating context-aware AI responses. Our work highlights the potential of GenAI tools to engage the design process itself rather than focusing on output.
>
---
#### [replaced 005] Neural Directional Filtering Using a Compact Microphone Array
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2511.07185v3](https://arxiv.org/pdf/2511.07185v3)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Mhd Modar Halimeh; Oliver Thiergart; Emanuël A. P. Habets
>
> **摘要:** Beamforming with desired directivity patterns using compact microphone arrays is essential in many audio applications. Directivity patterns achievable using traditional beamformers depend on the number of microphones and the array aperture. Generally, their effectiveness degrades for compact arrays. To overcome these limitations, we propose a neural directional filtering (NDF) approach that leverages deep neural networks to enable sound capture with a predefined directivity pattern. The NDF computes a single-channel complex mask from the microphone array signals, which is then applied to a reference microphone to produce an output that approximates a virtual directional microphone with the desired directivity pattern. We introduce training strategies and propose data-dependent metrics to evaluate the directivity pattern and directivity factor. We show that the proposed method: i) achieves a frequency-invariant directivity pattern even above the spatial aliasing frequency, ii) can approximate diverse and higher-order patterns, iii) can steer the pattern in different directions, and iv) generalizes to unseen conditions. Lastly, experimental comparisons demonstrate superior performance over conventional beamforming and parametric approaches.
>
---
#### [replaced 006] Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [https://arxiv.org/pdf/2508.02175v3](https://arxiv.org/pdf/2508.02175v3)**

> **作者:** Liang Lin; Miao Yu; Kaiwen Luo; Yibo Zhang; Lilan Peng; Dexian Wang; Xuehai Tang; Yuanhe Zhang; Xikang Yang; Zhenhong Zhou; Kun Wang; Yang Liu
>
> **摘要:** As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.
>
---
#### [replaced 007] Regularized Schrödinger Bridge: Alleviating Distortion and Exposure Bias in Solving Inverse Problems
- **分类: cs.LG; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.11686v2](https://arxiv.org/pdf/2511.11686v2)**

> **作者:** Qing Yao; Lijian Gao; Qirong Mao; Dong Ming
>
> **摘要:** Diffusion models serve as a powerful generative framework for solving inverse problems. However, they still face two key challenges: 1) the distortion-perception tradeoff, where improving perceptual quality often degrades reconstruction fidelity, and 2) the exposure bias problem, where the training-inference input mismatch leads to prediction error accumulation and reduced reconstruction quality. In this work, we propose the Regularized Schrödinger Bridge (RSB), an adaptation of Schrödinger Bridge tailored for inverse problems that addresses the above limitations. RSB employs a novel regularized training strategy that perturbs both the input states and targets, effectively mitigating exposure bias by exposing the model to simulated prediction errors and also alleviating distortion by well-designed interpolation via the posterior mean. Extensive experiments on two typical inverse problems for speech enhancement demonstrate that RSB outperforms state-of-the-art methods, significantly improving distortion metrics and effectively reducing exposure bias.
>
---
#### [replaced 008] MusRec: Zero-Shot Text-to-Music Editing via Rectified Flow and Diffusion Transformers
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.04376v3](https://arxiv.org/pdf/2511.04376v3)**

> **作者:** Ali Boudaghi; Hadi Zare
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Music editing has emerged as an important and practical area of artificial intelligence, with applications ranging from video game and film music production to personalizing existing tracks according to user preferences. However, existing models face significant limitations, such as being restricted to editing synthesized music generated by their own models, requiring highly precise prompts, or necessitating task-specific retraining, thus lacking true zero-shot capability. leveraging recent advances in rectified flow and diffusion transformers, we introduce MusRec, a zero-shot text-to-music editing model capable of performing diverse editing tasks on real-world music efficiently and effectively. Experimental results demonstrate that our approach outperforms existing methods in preserving musical content, structural consistency, and editing fidelity, establishing a strong foundation for controllable music editing in real-world scenarios.
>
---
#### [replaced 009] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification
- **分类: cs.SD**

- **链接: [https://arxiv.org/pdf/2510.17474v2](https://arxiv.org/pdf/2510.17474v2)**

> **作者:** Davide Salvi; Hendrik Vincent Koops; Elio Quinton
>
> **备注:** Accepted for presentation at the NeurIPS 2025 Workshop on Generative and Protective AI for Content Creation (non-archival)
>
> **摘要:** The proliferation of highly realistic singing voice deepfakes presents a significant challenge to protecting artist likeness and content authenticity. Automatic singer identification in vocal deepfakes is a promising avenue for artists and rights holders to defend against unauthorized use of their voice, but remains an open research problem. Based on the premise that the most harmful deepfakes are those of the highest quality, we introduce a two-stage pipeline to identify a singer's vocal likeness. It first employs a discriminator model to filter out low-quality forgeries that fail to accurately reproduce vocal likeness. A subsequent model, trained exclusively on authentic recordings, identifies the singer in the remaining high-quality deepfakes and authentic audio. Experiments show that this system consistently outperforms existing baselines on both authentic and synthetic content.
>
---
