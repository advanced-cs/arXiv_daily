# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] A Survey of Advancing Audio Super-Resolution and Bandwidth Extension from Discriminative to Generative Models
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频超分辨率任务，旨在从低分辨率信号中重建高质量音频。文章回顾了从判别模型到生成模型的演变，分析了不同方法的优缺点，并探讨了未来方向与挑战。**

- **链接: [https://arxiv.org/pdf/2605.16681](https://arxiv.org/pdf/2605.16681)**

> **作者:** Ningyuan Yang; Yize Li; Diego A. Cuji; Ryan M. Corey; Pu Zhao; Xue Lin; Andrew C. Singer
>
> **备注:** Under review
>
> **摘要:** Audio super-resolution (SR), also referred to as bandwidth extension (BWE), aims to reconstruct high-fidelity signals from low-resolution (LR) or band-limited (BL) observations, an inherently ill-posed task due to the ambiguity of missing high-frequency (HF) content. This survey provides a comprehensive overview of the field, with a particular focus on the paradigm shift from discriminative mapping to modern generative modeling. We first review early discriminative deep neural network (DNN) models, which formulate BWE/SR as a deterministic mapping problem and are prone to regression-to-the-mean effects and spectral over-smoothing. We then systematically review generative approaches, including autoregressive (AR) models, variational autoencoders (VAEs), generative adversarial networks (GANs), diffusion and score-based models, flow-based methods, and Schrödinger bridges. Across these approaches, we examine key design aspects, including representation domain, architecture, conditioning mechanisms, and trade-offs among reconstruction fidelity, perceptual quality, robustness, and computational efficiency. Furthermore, we discuss emerging directions involving large language models (LLMs) and multimodal foundation models, and highlight open challenges in perceptual evaluation, phase modeling, and real-world generalization. By providing a structured taxonomy and unified perspective, this survey establishes a comprehensive foundation and offers a practical roadmap for advancing BWE/SR from deterministic point estimation toward distribution-aware generative modeling.
>
---
#### [new 002] Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，解决超低比特率下语音可懂性下降的问题。通过强化学习优化量化策略，提升300bps下的词错误率表现。**

- **链接: [https://arxiv.org/pdf/2605.19541](https://arxiv.org/pdf/2605.19541)**

> **作者:** Junyi Wang; Chi Zhang; Jing Qian; Haifeng Luo; Hao Wang; Zengrui Jin; Chao Zhang
>
> **摘要:** In bandwidth-constrained communication such as satellite and underwater channels, speech must often be transmitted at ultra-low bitrates where intelligibility is the primary objective. At such extreme compression levels, codecs trained with acoustic reconstruction losses tend to allocate bits to perceptual detail, leading to substantial degradation in word error rate (WER). This paper proposes ClariCodec, a neural speech codec operating at 300 bit per second (bps) that reformulates quantisation as a stochastic policy, enabling reinforcement learning (RL)-based optimisation of intelligibility. Specifically, the encoder is fine-tuned using WER-driven rewards while the acoustic reconstruction pipeline remains frozen. Even without RL, ClariCodec achieves 4.64% WER on the LibriSpeech test-clean set at 300 bps, already competitive with codecs operating at higher bitrates. Further RL fine-tuning reduces WER to 3.55% on test-clean and 10.4% on test-other, corresponding to a 23% relative reduction while preserving perceptual quality.
>
---
#### [new 003] Precise and Simple Audio-to-Score Alignment
- **分类: cs.SD**

- **简介: 该论文属于音频与乐谱对齐任务，解决音频与符号化乐谱直接匹配的问题。提出一种新算法，结合音频特征与符号级信息，实现精确且灵活的对齐。**

- **链接: [https://arxiv.org/pdf/2605.20014](https://arxiv.org/pdf/2605.20014)**

> **作者:** Silvan Peter; Patricia Hu; Gerhard Widmer
>
> **备注:** published at the Music Encoding Conference (MEC) 2026
>
> **摘要:** Audio-to-score alignment is a long-standing challenge in music information retrieval and arguably the most widely applicable alignment task for music research. Alignment algorithms match two versions of a piece of music, and for this to work these versions need to be in comparable formats. Audio-to-audio alignment matches audio features; when matching audio files to scores, they must either synthesize the score or derive audio-like features by means of piano rolls or similar feature sequences. Symbolic alignment, by contrast, matches symbolically encoded notes; in an audio-to-score scenario these would be obtained by a transcription of the audio file. In this article, we present an algorithm that bridges audio-like and symbol-level features directly. Sequential audio features encoding onset and spectral activation are matched to score positions by a bespoke dynamic programming-based matching algorithm derived from symbolic alignment methods. The resulting method is both precise - surpassing widely used audio-to-audio approaches based on synthesized scores -, and remains flexible in its digital signal processing components, i.e., the method is adaptable to diverse timbral characteristics without requiring a separate transcription model. Furthermore it inherits some of the symbolic alignment runtime advantages with an algorithmic complexity that is at worst linear in the length of the (typically short) symbolic score and (typically long) audio feature sequence. In the following sections, we provide a detailed algorithm description and evaluate its alignment quality on a large-scale dataset of solo piano recordings.
>
---
#### [new 004] Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决真实环境下的声学鲁棒性问题。通过构建大规模真实声学数据集并优化模型，提升复杂场景下的识别性能。**

- **链接: [https://arxiv.org/pdf/2605.19833](https://arxiv.org/pdf/2605.19833)**

> **作者:** Zhifei Xie; Kaiyu Pang; Haobin Zhang; Deheng Ye; Xiaobin Hu; Shuicheng Yan; Chunyan Miao
>
> **备注:** Project page: this https URL. Code, models, and dataset will be released. A robust ASR framework targeting in-the-wild and compositional acoustic scenarios where conventional ASR systems fail
>
> **摘要:** Despite rapid advances in automatic speech recognition (ASR) and large audio-language models, robust recognition in real-world environments remains limited by an "acoustic robustness bottleneck": models often lose acoustic grounding and produce omissions or hallucinations under severe, compositional distortions. We propose Mega-ASR, a unified ASR-in-the-wild framework that combines scalable compound-data construction with progressive acoustic-to-semantic optimization. We introduce Voices-in-the-Wild-2M, covering 7 classic acoustic phenomena and 54 physically plausible compound scenarios, and train Mega-ASR with Acoustic-to-Semantic Progressive Supervised Fine-Tuning and Dual-Granularity WER-Gated Policy Optimization. Extensive experiments demonstrate that Mega-ASR achieves significant advantages over prior state-of-the-art systems on adverse-condition ASR benchmarks (45.69% vs. 54.01% on VOiCES R4-B-F, and 21.49% vs. 29.34% on NOIZEUS Sta-0). On complex compositional acoustic scenarios, Mega-ASR further delivers over 30% relative WER reduction against strong open- and closed-source baselines, establishing a scalable paradigm for robust ASR in-the-wild.
>
---
#### [new 005] Heterogeneity-Aware Dataset Scheduling for Efficient Audio Large Language Model Training
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频大语言模型训练任务，解决数据集异质性导致的梯度冲突和收敛慢问题。提出GST方法，通过分组顺序训练提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.19101](https://arxiv.org/pdf/2605.19101)**

> **作者:** Yanru Wu; Jianning Wang; Chongxin Gan; Yang Li
>
> **摘要:** Training general-purpose Audio Large Language Models (ALLMs) across diverse datasets is essential for holistic audio understanding, yet it faces significant challenges due to dataset heterogeneity, which often leads to conflicting gradients and slow convergence. Despite its impact, how to explicitly manage this heterogeneity during training remains underexplored, with current practices relying primarily on uniform mixture. In this work, we analyze multi-dataset AudioQA training from a convergence perspective and propose Grouped Sequential Training (GST). GST strategically organizes datasets into affinity-aware groups and introduces them via a progressive scheduling protocol, effectively balancing the stability of parallel training with the efficiency of sequential optimization. To ensure scalability, we develop gradient-based affinity metrics that capture inter-dataset relationships without the prohibitive cost of empirical transferability estimation. Extensive evaluations on 14 AudioQA datasets spanning speech, music, and environmental sounds demonstrate that GST achieves 30--40\% faster convergence than standard parallel training while maintaining or even surpassing the performance of mix-all training. Our results provide both theoretical insights and a practical, model-agnostic framework for efficient large-scale ALLM optimization.
>
---
#### [new 006] A conceptual framework for learning to listen by reward: Curiosity-driven search for novel sources
- **分类: cs.SD**

- **简介: 该论文属于强化学习任务，旨在解决音频领域中通过奖励驱动探索学习聆听的问题。提出一种基于好奇心的新型概念框架，通过持续搜索新声音源实现听觉学习。**

- **链接: [https://arxiv.org/pdf/2605.19984](https://arxiv.org/pdf/2605.19984)**

> **作者:** Andreas Triantafyllopoulos; Jakub Šťastný; Alexios Terpinas; Tianyi Liu; Yuanqi Wang; Björn W. Schuller
>
> **摘要:** Reinforcement learning is a powerful learning paradigm that has spearheaded progress in numerous domains. Its core promise lies in learning through high-level goals without the need for granular labels. However, it still remains elusive in the realm of audio, where it has received substantially less attention than in computer vision or other domains. The key question remains: how can agents learn to listen purely via reward-driven exploration? In this contribution, we present an overview of previous attempts and a new conceptual framework for learning to listen by reward. Our approach depends on the continuous search for novel sound sources. We formulate our framework, discuss open technical challenges, and present a first proof-of-concept implementation that showcases the feasibility of our approach.
>
---
#### [new 007] Fast Multichannel NMF with Block-Diagonal Spatial Covariance Matrices for Efficient Blind Source Separation Using Distributed Microphone Arrays
- **分类: eess.AS**

- **简介: 该论文属于盲源分离任务，解决分布式麦克风阵列中计算效率与性能平衡问题。提出分布式FastMNMF方法，通过块对角空间协方差结构降低计算量，同时提升分离效果。**

- **链接: [https://arxiv.org/pdf/2605.19388](https://arxiv.org/pdf/2605.19388)**

> **作者:** Hirotaka Nishikori; Nobutaka Ito; Kouei Yamaoka; Norihiro Takamune; Hiroshi Saruwatari
>
> **摘要:** Distributed microphone arrays composed of multiple subarrays enable blind source separation over a wide spatial area. Directly applying fast multichannel nonnegative matrix factorization (FastMNMF) to all subarrays can exploit observations from all subarrays, but it requires repeated inversions of large matrices spanning all microphones, causing the computational cost to increase rapidly as the number of microphones grows. In contrast, applying FastMNMF to one subarray reduces the matrix size but cannot exploit observations from other subarrays. We propose distributed FastMNMF, which imposes a block-diagonal structure on the source spatial covariance matrices, so that matrix inversions are performed within subarrays. The NMF-based source spectrogram model is shared across subarrays, allowing the method to aggregate source activity information while discarding inter-subarray covariance. In synchronized, noiseless simulations with fixed room and array/source geometry, the method required less computation time than conventional FastMNMF using all subarrays, achieved a higher average source-to-distortion ratio than conventional FastMNMF using one subarray, and was applicable in the tested five-source condition, where each four-microphone subarray was locally underdetermined.
>
---
#### [new 008] Cross-Talk Speech Reduction, by Separation, for Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决远场混音中交叉干扰的问题。通过提出CTRnet和PuLSS方法，利用真实数据提升分离效果。**

- **链接: [https://arxiv.org/pdf/2605.19695](https://arxiv.org/pdf/2605.19695)**

> **作者:** Zhong-Qiu Wang; Samuele Cornell
>
> **备注:** in submission
>
> **摘要:** In conversational speech separation and recognition tasks, close-talk microphones are typically attached to each speaker during training data collection to capture near-field, close-talk mixture signals, in addition to using far-field microphones to record far-field mixture signals. Each such close-talk mixture exhibits a reasonably high energy level for the wearer and could intuitively serve as weak supervision for training far-field speech separation models directly on real-recorded far-field signals. However, they are not sufficiently clean for this purpose, as they often contain strong cross-talk speech from other speakers in addition to background noise. To address this, we propose cross-talk reduction (CTR), a task aiming to isolate the wearer's speech from each close-talk mixture, and a novel method called CTRnet, which can be trained directly on real-recorded pairs of close-talk and far-field mixtures to accomplish CTR. Building on CTRnet, we further propose pseudo-label based far-field speech separation (PuLSS), which uses CTRnet's estimated clean speech as pseudo-labels to train models for separating far-field mixtures. A key advantage of the proposed framework is that both CTRnet and PuLSS can be trained on real-recorded data from the target domain, addressing the generalization gap commonly observed when models are trained exclusively on simulated data. On the CHiME-6 dataset, our framework achieves state-of-the-art ASR performance under both oracle and estimated speaker diarization, surpassing all CHiME-{7,8} challenge submissions. To our knowledge, it is the first neural speech separation method that substantially outperforms guided source separation on real conversational "speech-in-the-wild" data.
>
---
#### [new 009] CounterFlow: A Two-Phase Inference-Time Sampling for Counterfactual Video Foley Generation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视频音频生成任务，解决视频与文本不一致时的反事实音频生成问题。提出ConterFlow方法，在推理阶段分两步优化音频，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.18916](https://arxiv.org/pdf/2605.18916)**

> **作者:** Gyubin Lee; Junwon Lee; Juhan Nam
>
> **备注:** accepted to CVPR 2026 Workshop on Sight and Sound
>
> **摘要:** We investigate Counterfactual Video Foley Generation, which aims to adopt a sound-source identity that contradicts the visual evidence while remaining temporally synchronized to a silent video. Existing Video&Text-to-Audio (VT2A) models struggle with this, often remaining anchored to the visually implied sound source when video and text contents disagree. We present ConterFlow, an inference-time dual-phase sampling scheme for pretrained flow-matching VT2A models. Phase 1 builds a video-derived temporal structure while suppressing the visually implied source; Phase 2 drops video conditioning to focus entirely on shaping audio timbre toward the target prompt. ConterFlow substantially improves counterfactual Video Foley generation compared to naive negative prompting and state-of-the-art baselines. To evaluate replacement quality, we propose a metric leveraging a text-audio co-embedding space to measure both target-prompt evidence and residual visually implied source leakage. Video demonstrations and code are available at this https URL
>
---
#### [new 010] Executable Boundary Contracts for Sound Event Traces
- **分类: cs.LO; cs.SD**

- **简介: 该论文属于声学事件分析任务，旨在通过可执行边界合约对声事件轨迹进行精确测量，解决传统方法在边界行为描述上的不足。**

- **链接: [https://arxiv.org/pdf/2605.19632](https://arxiv.org/pdf/2605.19632)**

> **作者:** Faruk Alpay; Hamdi Alakkad
>
> **备注:** 39 pages. Finite frame core code, tables, manifests, and Lean checks are ancillary material
>
> **摘要:** Sound event reports often compress timed boundary behavior into frame, segment, or event scores. This paper defines executable boundary contracts for finite sound event traces. The frame fragment is a bounded Boolean fragment embeddable in STL after grid projection. The event layer adds declared interval matching, duration clauses, fragmentation clauses, and obligation restricted vector scoring. The aim is measurement, not a new general temporal logic and not a challenge leaderboard. The artifact evaluates controlled Mini LibriSpeech seeded scenes, MAESTRO Real soundscapes, frozen pretrained timing probes, and an official DCASE 2024 Task 4 baseline track. Across these tracks, standard scores and contract coordinates disagree in interpretable ways. The strongest real corpus finding is that union activity can hide typed boundary failure, while external DCASE outputs provide a class indexed challenge level reference. Code, generated tables, manifests, and Lean checks for the finite frame core are supplied as ancillary material.
>
---
#### [new 011] DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis
- **分类: cs.CR; cs.SD**

- **简介: 该论文属于语音流隐写分析任务，解决多领域数据分布不一致导致的检测性能下降问题。提出DASM优化器，提升模型泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19955](https://arxiv.org/pdf/2605.19955)**

> **作者:** Pengcheng Zhou; Pianran Guo; Shuhua Chen; Mengqin Zhao; Zhongliang Yang; Linna Zhou
>
> **摘要:** The growing use of information hiding in network streaming media for covert communication poses a significant security threat, necessitating the development of robust detection technologies. However, existing steganalysis methods for network voice streams mostly rely on data distributions in specific scenarios, making it difficult to adapt to the practical detection needs of non-homologous data distributions. Through Hessian analysis, we find that the loss landscapes of mainstream models are dominated by numerous saddle points and sharp local minima, rendering them highly sensitive to data distribution shifts and fundamentally limiting generalization. Therefore, we propose a new optimizer, Domain-Aware Sharpness Minimization (DASM). The core mechanisms of DASM consist of two aspects: first, it integrates domain-supervised contrastive learning with sharpness-aware optimization, explicitly preserving inter-domain feature separation while seeking flat minima; second, we design an adaptive domain gap modulation strategy that dynamically calibrates the optimization loss weights by sensing the real-time feature separability of different domains. Extensive experimental results demonstrate that our method outperforms the state-of-the-art methods by a large margin and achieves excellent generalization and robustness.
>
---
## 更新

#### [replaced 001] Exploring Speech Foundation Models for Speaker Diarization Across Lifespan
- **分类: eess.AS**

- **简介: 该论文属于语音任务，研究跨年龄的说话人辨识问题。针对模型在不同年龄段数据上的泛化能力不足，提出多年龄联合训练和针对性适应方法，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.05201](https://arxiv.org/pdf/2604.05201)**

> **作者:** Anfeng Xu; Tiantian Feng; Shrikanth Narayanan
>
> **备注:** Under review
>
> **摘要:** Speech foundation models have shown strong transferability across a wide range of speech applications. However, their robustness to age-related domain shift in speaker diarization remains underexplored. In this work, we present a cross-lifespan evaluation within a unified end-to-end neural diarization framework (EEND-VC), covering speech samples from conversations involving children, adults, and older adults. We compare models under zero-shot cross-age inference, joint multi-age training, and domain-specific adaptation. Results show substantial performance degradation when models trained on adult-specific speech are applied to child and older-adult conversational data. Moreover, joint multi-age training across different age groups improves robustness without reducing diarization performance in canonical adult conversations, while targeted age group adaptation yields further gains in diarization performance, particularly when using the Whisper encoder.
>
---
#### [replaced 002] Deep Neural Network for Musical Instrument Recognition using MFCCs
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐乐器识别任务，旨在通过音频数据准确分类不同乐器。工作包括使用MFCC特征和ANN模型对20类乐器进行训练，取得了先进成果。**

- **链接: [https://arxiv.org/pdf/2105.00933](https://arxiv.org/pdf/2105.00933)**

> **作者:** Saranga Kingkor Mahanta; Abdullah Faiz Ur Rahman Khilji; Partha Pakray
>
> **摘要:** The task of efficient automatic music classification is of vital importance and forms the basis for various advanced applications of AI in the musical domain. Musical instrument recognition is the task of instrument identification by virtue of its audio. This audio, also termed as the sound vibrations are leveraged by the model to match with the instrument classes. In this paper, we use an artificial neural network (ANN) model that was trained to perform classification on twenty different classes of musical instruments. Here we use use only the mel-frequency cepstral coefficients (MFCCs) of the audio data. Our proposed model trains on the full London philharmonic orchestra dataset which contains twenty classes of instruments belonging to the four families viz. woodwinds, brass, percussion, and strings. Based on experimental results our model achieves state-of-the-art accuracy on the same.
>
---
#### [replaced 003] Contextual Biasing for Streaming ASR via CTC-based Word Spotting
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决实时流式ASR中罕见词识别问题。提出一种基于CTC的关键词检测方法，实现低延迟、稳定输出的上下文偏置。**

- **链接: [https://arxiv.org/pdf/2605.18222](https://arxiv.org/pdf/2605.18222)**

> **作者:** Kai-Chen Tsai; Tien-Hong Lo; Yun-Ting Sun; Berlin Chen
>
> **摘要:** Contextual biasing is essential to improving the recognition of rare and domain-specific words in an automatic speech recognition (ASR) system. While numerous methods have been proposed in recent years, most of them focus on offline settings and do not explicitly address the challenges of streaming ASR. For example, CTC-based word spotting (CTC-WS) have demonstrated strong performance by directly detecting keywords from CTC log-probabilities, but they are limited to offline processing and require access to the full utterance. In This work, we present a streaming extension of CTC-WS for real-time contextual biasing. Our method maintains active keyword paths across audio chunks using a stateful token passing algorithm, enabling the detection of keywords that span multiple chunks. To ensure low latency and stable output, we introduce an incremental commitment mechanism that only emits segments guaranteed not to be affected by future audio, while deferring uncertain regions. This method naturally integrates with streaming ASR pipelines and does not require modifications to the underlying acoustic model or additional training, making it practical for real-world deployment. Experimental results show that our method reduces overall WER and effectively improves keyword F-score, demonstrating its effectiveness for real-time ASR applications.
>
---
#### [replaced 004] HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出HarmonicAttack，用于去除音频水印。属于水印移除任务，解决无目标算法访问时的水印移除问题，通过训练模型实现高效且跨域的水印去除。**

- **链接: [https://arxiv.org/pdf/2511.21577](https://arxiv.org/pdf/2511.21577)**

> **作者:** Kexin Li; Xiao Hu; Ilya Grishchenko; David Lie
>
> **备注:** Under Review
>
> **摘要:** The availability of high-quality, AI-generated audio raises security challenges such as misinformation campaigns and voice-cloning fraud. A key defense against the misuse of AI-generated audio is by watermarking it, so that it can be easily distinguished from genuine audio. Those seeking to misuse AI-generated audio may attempt to remove audio watermarks, so studying effective watermark removal techniques is critical to objectively evaluate the robustness of audio watermarks. Previous watermark removal schemes typically assume access to the target watermark detector during the removal process. This assumption is often impractical, which may lead to a false sense of confidence in current watermark schemes. We introduce HarmonicAttack, a novel audio watermark removal method that requires no access to the target watermark algorithm. It only needs a number of original and watermarked samples to train a general model capable of removing watermarks from audio samples. We also find that training samples do not need to share the same distribution as target samples, as our attack generalizes to out-of-distribution samples with minimal degradation. Compared with existing watermark removal attacks, HarmonicAttack is more effective at removing watermarks from state-of-the-art schemes, including AudioSeal, WavMark, SilentCipher, and AudioMarkNet, while maintaining high perceptual quality. Although HarmonicAttack is trained on the LibriSpeech dataset against AudioSeal, it generalizes across unseen datasets and watermarking schemes. For instance, on VCTK, HarmonicAttack achieves a 92% ASR against AudioMarkNet, substantially outperforming the best baseline at 38%. On FMA, HarmonicAttack reaches 100% ASR against all watermarks, whereas the best baseline achieves only 2% against AudioSeal and 44% against WavMark.
>
---
#### [replaced 005] Non-Intrusive Automatic Speech Recognition Refinement: A Survey
- **分类: eess.AS**

- **简介: 该论文属于语音识别优化任务，解决ASR系统在语音多样性及环境干扰下的准确性问题。综述了非侵入式改进方法，分类并分析其优缺点。**

- **链接: [https://arxiv.org/pdf/2508.07285](https://arxiv.org/pdf/2508.07285)**

> **作者:** Mohammad Reza Peyghan; Saman Soleimani Roudi; Saeedreza Zouashkiani; Sajjad Amini; Fatemeh Rajabi; Shahrokh Ghaemmaghami
>
> **摘要:** Automatic Speech Recognition (ASR) is an integral component of modern technology, powering applications such as voice-activated assistants, transcription services, and accessibility tools. Yet ASR systems continue to struggle with the inherent variability of human speech, such as accents, dialects, and speaking styles, as well as environmental interference, including background noise. Moreover, domain-specific conversations often employ specialized terminology, which can exacerbate transcription errors. These shortcomings not only degrade raw ASR accuracy but also propagate mistakes through subsequent natural language processing pipelines. Because redesigning an ASR model is costly and time-consuming, non-intrusive refinement techniques that leave the model's architecture intact have become increasingly popular. In this survey, we review current non-intrusive refinement approaches and group them into five classes: fusion, re-scoring, correction, distillation, and training adjustment. For each class, we outline the main methods, advantages, drawbacks, and ideal application scenarios. Beyond method classification, this work surveys adaptation techniques aimed at refining ASR in domain-specific contexts, reviews commonly used evaluation datasets along with their construction processes, and proposes a standardized set of metrics to facilitate fair comparisons. Finally, we identify open research gaps and suggest promising directions for future work. By providing this structured overview, we aim to equip researchers and practitioners with a clear foundation for developing more robust, accurate ASR refinement pipelines.
>
---
#### [replaced 006] Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于非侵入式物体分类任务，旨在通过声学散射实现头发类型和湿度的无接触识别。研究采用AI深度学习方法进行声波信号分类。**

- **链接: [https://arxiv.org/pdf/2506.14148](https://arxiv.org/pdf/2506.14148)**

> **作者:** Long-Vu Hoang; Tuan Nguyen; Tran Huy Dat
>
> **备注:** This paper has been retracted by the authors. Due to miscommunication, the authorship is incomplete and missing early contributions
>
> **摘要:** This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries.
>
---
#### [replaced 007] TADA! Tuning Audio Diffusion Models through Activation Steering
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频生成任务，旨在解决音乐属性控制难题。通过激活调节方法，提升对特定音乐元素的精细控制能力。**

- **链接: [https://arxiv.org/pdf/2602.11910](https://arxiv.org/pdf/2602.11910)**

> **作者:** Łukasz Staniszewski; Katarzyna Zaleska; Mateusz Modrzejewski; Kamil Deja
>
> **备注:** Preprint
>
> **摘要:** Audio diffusion models can synthesize high-fidelity music from text, yet achieving fine-grained control over specific musical attributes remains challenging, as their internal mechanisms for representing high-level concepts are poorly understood. In this work, we use activation patching to demonstrate that recent audio diffusion architectures exhibit a semantic bottleneck, where a small, shared subset of consecutive attention layers controls distinct musical concepts, such as the presence of specific instruments, vocals, or genres. Building on this, we systematically evaluate a broad spectrum of steering paradigms, comparing activation steering against prompt-level, score-space, and weight-space interventions, analyzing the interaction between the steering mechanism and the intervention site. Our new benchmark, supported by an extensive user study, demonstrates that localized activation steering establishes a new state-of-the-art in audio concept modulation.
>
---
