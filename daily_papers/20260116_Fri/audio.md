# 音频 cs.SD;  eess.AS

- **最新发布 9 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Diffusion-based Frameworks for Unsupervised Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决无监督单通道语音增强问题。通过改进扩散模型，明确建模语音和噪声，提升增强效果。**

- **链接: [https://arxiv.org/pdf/2601.09931v1](https://arxiv.org/pdf/2601.09931v1)**

> **作者:** Jean-Eudes Ayilo; Mostafa Sadeghi; Romain Serizel; Xavier Alameda-Pineda
>
> **摘要:** This paper addresses $\textit{unsupervised}$ diffusion-based single-channel speech enhancement (SE). Prior work in this direction combines a score-based diffusion model trained on clean speech with a Gaussian noise model whose covariance is structured by non-negative matrix factorization (NMF). This combination is used within an iterative expectation-maximization (EM) scheme, in which a diffusion-based posterior-sampling E-step estimates the clean speech. We first revisit this framework and propose to explicitly model both speech and acoustic noise as latent variables, jointly sampling them in the E-step instead of sampling speech alone as in previous approaches. We then introduce a new unsupervised SE framework that replaces the NMF noise prior with a diffusion-based noise model, learned jointly with the speech prior in a single conditional score model. Within this framework, we derive two variants: one that implicitly accounts for noise and one that explicitly treats noise as a latent variable. Experiments on WSJ0-QUT and VoiceBank-DEMAND show that explicit noise modeling systematically improves SE performance for both NMF-based and diffusion-based noise priors. Under matched conditions, the diffusion-based noise model attains the best overall quality and intelligibility among unsupervised methods, while under mismatched conditions the proposed NMF-based explicit-noise framework is more robust and suffers less degradation than several supervised baselines. Our code will be publicly available on this $\href{https://github.com/jeaneudesAyilo/enudiffuse}{URL}$.
>
---
#### [new 002] Self-supervised restoration of singing voice degraded by pitch shifting using shallow diffusion
- **分类: cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决pitch shifting导致的音质退化问题。通过自监督恢复方法，提升音调转换后的语音质量。**

- **链接: [https://arxiv.org/pdf/2601.10345v1](https://arxiv.org/pdf/2601.10345v1)**

> **作者:** Yunyi Liu; Taketo Akama
>
> **摘要:** Pitch shifting has been an essential feature in singing voice production. However, conventional signal processing approaches exhibit well known trade offs such as formant shifts and robotic coloration that becomes more severe at larger transposition jumps. This paper targets high quality pitch shifting for singing by reframing it as a restoration problem: given an audio track that has been pitch shifted (and thus contaminated by artifacts), we recover a natural sounding performance while preserving its melody and timing. Specifically, we use a lightweight, mel space diffusion model driven by frame level acoustic features such as f0, volume, and content features. We construct training pairs in a self supervised manner by applying pitch shifts and reversing them to simulate realistic artifacts while retaining ground truth. On a curated singing set, the proposed approach substantially reduces pitch shift artifacts compared to representative classical baselines, as measured by both statistical metrics and pairwise acoustic measures. The results suggest that restoration based pitch shifting could be a viable approach towards artifact resistant transposition in vocal production workflows.
>
---
#### [new 003] Nearest Kronecker Product Decomposition Based Subband Adaptive Filter: Algorithms and Applications
- **分类: eess.AS; cs.IT**

- **简介: 该论文属于自适应滤波任务，旨在解决输入信号相关性高导致收敛性能下降的问题，提出多种改进的NKP-NSAF算法以提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.10078v1](https://arxiv.org/pdf/2601.10078v1)**

> **作者:** Jianhong Ye; Haiquan Zhao
>
> **备注:** 16 Pages, 19 figures,published to IEEE TASLP
>
> **摘要:** Recently, the nearest Kronecker product (NKP) decomposition-based normalized least mean square (NLMS-NKP) algorithm has demonstrated superior convergence performance compared to the conventional NLMS algorithm. However, its convergence rate exhibits significant degradation when processing highly correlated input signals. To address this problem, we propose a type-I NKP-based normalized subband adaptive filter (NSAF) algorithm, namely NSAF-NKP-I. Nevertheless, this algorithm incurs substantially higher computational overhead than the NLMS-NKP algorithm. Remarkably, our enhanced type-II NKP-based NSAF (NSAF-NKP-II) algorithm achieves equivalent convergence performance while substantially reducing computational complexity. Furthermore, to enhance robustness against impulsive noise interference, we develop two robust variants: the maximum correntropy criterion-based robust NSAF-NKP (RNSAF-NKP-MCC) and logarithmic criterion-based robust NSAF-NKP (RNSAF-NKP-LC) algorithms. Additionally, detailed analyses of computational complexity, step-size range, and theoretical steady-state performance are provided for theproposed algorithms. To enhance the practicability of the NSAF-NKP-II algorithm in complex nonlinear environments, we further devise two nonlinear implementations: the trigonometric functional link network-based NKP-NSAF (TFLN-NSAF-NKP) and Volterra series expansion-based NKP-NSAF (Volterra-NKP-NSAF) algorithms. In active noise control (ANC) systems, we further propose the filtered-x NSAF-NKP-II (NKP-FxNSAF) algorithm. Simulation experiments in echo cancellation, sparse system identification, nonlinear processing, and ANC scenarios are conducted to validate the superiority of the proposed algorithms over existing state-of-the-art counterparts.
>
---
#### [new 004] Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics
- **分类: cs.SD; cs.LG; eess.AS; physics.comp-ph**

- **简介: 该论文属于物理建模与机器学习任务，旨在解决非线性动力学学习问题。结合模态方法与神经微分方程，构建稳定可微模型，实现非线性振动系统的自动学习与模拟。**

- **链接: [https://arxiv.org/pdf/2601.10453v1](https://arxiv.org/pdf/2601.10453v1)**

> **作者:** Victor Zheleznov; Stefan Bilbao; Alec Wright; Simon King
>
> **备注:** Submitted to the Journal of Audio Engineering Society (December 2025)
>
> **摘要:** Modal methods are a long-standing approach to physical modelling synthesis. Extensions to nonlinear problems are possible, including the case of a high-amplitude vibration of a string. A modal decomposition leads to a densely coupled nonlinear system of ordinary differential equations. Recent work in scalar auxiliary variable techniques has enabled construction of explicit and stable numerical solvers for such classes of nonlinear systems. On the other hand, machine learning approaches (in particular neural ordinary differential equations) have been successful in modelling nonlinear systems automatically from data. In this work, we examine how scalar auxiliary variable techniques can be combined with neural ordinary differential equations to yield a stable differentiable model capable of learning nonlinear dynamics. The proposed approach leverages the analytical solution for linear vibration of system's modes so that physical parameters of a system remain easily accessible after the training without the need for a parameter encoder in the model architecture. As a proof of concept, we generate synthetic data for the nonlinear transverse vibration of a string and show that the model can be trained to reproduce the nonlinear dynamics of the system. Sound examples are presented.
>
---
#### [new 005] Multi-Level Embedding Conformer Framework for Bengali Automatic Speech Recognition
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言Bengali的ASR问题。通过多粒度语言信息融合的Conformer框架，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.09710v1](https://arxiv.org/pdf/2601.09710v1)**

> **作者:** Md. Nazmus Sakib; Golam Mahmud; Md. Maruf Bangabashi; Umme Ara Mahinur Istia; Md. Jahidul Islam; Partha Sarker; Afra Yeamini Prity
>
> **摘要:** Bengali, spoken by over 300 million people, is a morphologically rich and lowresource language, posing challenges for automatic speech recognition (ASR). This research presents an end-to-end framework for Bengali ASR, building on a Conformer-CTC backbone with a multi-level embedding fusion mechanism that incorporates phoneme, syllable, and wordpiece representations. By enriching acoustic features with these linguistic embeddings, the model captures fine-grained phonetic cues and higher-level contextual patterns. The architecture employs early and late Conformer stages, with preprocessing steps including silence trimming, resampling, Log-Mel spectrogram extraction, and SpecAugment augmentation. The experimental results demonstrate the strong potential of the model, achieving a word error rate (WER) of 10.01% and a character error rate (CER) of 5.03%. These results demonstrate the effectiveness of combining multi-granular linguistic information with acoustic modeling, providing a scalable approach for low-resource ASR development.
>
---
#### [new 006] VoiceSculptor: Your Voice, Designed By You
- **分类: eess.AS**

- **简介: 该论文提出VoiceSculptor，解决TTS中缺乏细粒度语音控制的问题。整合指令设计与高保真克隆，实现语音属性的精准控制与迭代优化。**

- **链接: [https://arxiv.org/pdf/2601.10629v1](https://arxiv.org/pdf/2601.10629v1)**

> **作者:** Jingbin Hu; Huakang Chen; Linhan Ma; Dake Guo; Qirui Zhan; Wenhao Li; Haoyu Zhang; Kangxiang Xia; Ziyu Zhang; Wenjie Tian; Chengyou Wang; Jinrui Liang; Shuhan Guo; Zihang Yang; Bengu Wu; Binbin Zhang; Pengcheng Zhu; Pengyuan Xie; Chuan Xie; Qiang Zhang; Jie Liu; Lei Xie
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Despite rapid progress in text-to-speech (TTS), open-source systems still lack truly instruction-following, fine-grained control over core speech attributes (e.g., pitch, speaking rate, age, emotion, and style). We present VoiceSculptor, an open-source unified system that bridges this gap by integrating instruction-based voice design and high-fidelity voice cloning in a single framework. It generates controllable speaker timbre directly from natural-language descriptions, supports iterative refinement via Retrieval-Augmented Generation (RAG), and provides attribute-level edits across multiple dimensions. The designed voice is then rendered into a prompt waveform and fed into a cloning model to enable high-fidelity timbre transfer for downstream speech synthesis. VoiceSculptor achieves open-source state-of-the-art (SOTA) on InstructTTSEval-Zh, and is fully open-sourced, including code and pretrained models, to advance reproducible instruction-controlled TTS research.
>
---
#### [new 007] RSA-Bench: Benchmarking Audio Large Models in Real-World Acoustic Scenarios
- **分类: cs.SD**

- **简介: 该论文属于音频模型鲁棒性评估任务，旨在解决真实声学场景下模型性能下降的问题。通过构建多场景噪声数据集，分析模型在不同干扰下的表现。**

- **链接: [https://arxiv.org/pdf/2601.10384v1](https://arxiv.org/pdf/2601.10384v1)**

> **作者:** Yibo Zhang; Liang Lin; Kaiwen Luo; Shilinlu Yan; Jin Wang; Yaoqi Guo; Yitian Chen; Yalan Qin; Zhenhong Zhou; Kun Wang; Li Sun
>
> **摘要:** While Audio Large Models (ALMs) have achieved remarkable proficiency, their robustness remains brittle in real-world deployment. Existing evaluations largely rely on synthetic Gaussian noise or simplistic single-source interference, failing to capture the intricate, multi-layered acoustic dynamics -- or ``Acoustic Ecology'' -- that characterize authentic physical environments. To bridge this ecological gap, we introduce \textbf{RSA-Bench}, a comprehensive robustness benchmark designed to stress-test ALLMs through high-fidelity auditory scene simulations. Unlike traditional methods, we construct evaluation samples by naturally superimposing diverse environmental soundscapes -- spanning \textit{Pasture}, \textit{Extreme Weather}, \textit{Classroom}, and \textit{Outdoors} -- onto clean speech signals across a spectrum of interference intensities. By evaluating models on six core tasks ranging from fundamental perception to complex reasoning, our study unveils three macro-level insights: \textbf{(I) The Perception-Cognition Gap:} Models maintain relative resilience in low-level recognition but suffer a \textbf{functional collapse} in high-order reasoning tasks under stress; \textbf{(II) Scenario Sensitivity:} ``Vocal-like'' interference (e.g., background laughter) proves significantly more destructive than mechanical noise, challenging the model's auditory attention mechanisms; and \textbf{(III) The Denoising Paradox:} Standard speech enhancement often exacerbates performance degradation, as ALLMs prove highly sensitive to the semantic distortions introduced by denoising artifacts.
>
---
#### [new 008] HeartMuLa: A Family of Open Sourced Music Foundation Models
- **分类: cs.SD**

- **简介: 该论文提出HeartMuLa音乐基础模型家族，解决音乐理解与生成任务，涵盖音频文本对齐、歌词识别、音乐编码及歌曲生成，支持多模态应用。**

- **链接: [https://arxiv.org/pdf/2601.10547v1](https://arxiv.org/pdf/2601.10547v1)**

> **作者:** Dongchao Yang; Yuxin Xie; Yuguo Yin; Zheyu Wang; Xiaoyu Yi; Gongxi Zhu; Xiaolong Weng; Zihan Xiong; Yingzhe Ma; Dading Cong; Jingliang Liu; Zihang Huang; Jinghan Ru; Rongjie Huang; Haoran Wan; Peixu Wang; Kuoxi Yu; Helin Wang; Liming Liang; Xianwei Zhuang; Yuanyuan Wang; Haohan Guo; Junjie Cao; Zeqian Ju; Songxiang Liu; Yuewen Cao; Heming Weng; Yuexian Zou
>
> **摘要:** We present a family of open-source Music Foundation Models designed to advance large-scale music understanding and generation across diverse tasks and modalities. Our framework consists of four major components: (1) HeartCLAP, an audio-text alignment model; (2) HeartTranscriptor, a robust lyric recognition model optimized for real-world music scenarios; and (3) HeartCodec, a low-frame-rate (12.5 Hz) yet high-fidelity music codec tokenizer that captures long-range musical structure while preserving fine-grained acoustic details and enabling efficient autoregressive modeling; (4) HeartMuLa, an LLM-based song generation model capable of synthesizing high-fidelity music under rich, user-controllable conditions (e.g., textual style descriptions, lyrics, and reference audio). In addition, it provides two specialized modes: (i) fine-grained musical attribute control, which allows users to specify the style of different song sections (e.g., intro, verse, chorus) using natural language prompts; and (ii) short, engaging music generation, which is suitable as background music for short videos. Lastly, HeartMuLa improves significantly when scaled to 7B parameters. For the first time, we show that a Suno-level, commercial-grade system can be reproduced using academic-scale data and GPU resources. We expect these foundation models to serve as strong baselines for future research and to facilitate practical applications in multimodal content production.
>
---
#### [new 009] MoST: Mixing Speech and Text with Modality-Aware Mixture of Experts
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出MoST模型，解决多模态语音与文本处理问题。通过MAMoE架构实现模态感知的专家混合，提升语音和文本任务性能。**

- **链接: [https://arxiv.org/pdf/2601.10272v1](https://arxiv.org/pdf/2601.10272v1)**

> **作者:** Yuxuan Lou; Kai Yang; Yang You
>
> **摘要:** We present MoST (Mixture of Speech and Text), a novel multimodal large language model that seamlessly integrates speech and text processing through our proposed Modality-Aware Mixture of Experts (MAMoE) architecture. While current multimodal models typically process diverse modality representations with identical parameters, disregarding their inherent representational differences, we introduce specialized routing pathways that direct tokens to modality-appropriate experts based on input type. MAMoE simultaneously enhances modality-specific learning and cross-modal understanding through two complementary components: modality-specific expert groups that capture domain-specific patterns and shared experts that facilitate information transfer between modalities. Building on this architecture, we develop an efficient transformation pipeline that adapts the pretrained MoE language model through strategic post-training on ASR and TTS datasets, followed by fine-tuning with a carefully curated speech-text instruction dataset. A key feature of this pipeline is that it relies exclusively on fully accessible, open-source datasets to achieve strong performance and data efficiency. Comprehensive evaluations across ASR, TTS, audio language modeling, and spoken question answering benchmarks show that MoST consistently outperforms existing models of comparable parameter counts. Our ablation studies confirm that the modality-specific routing mechanism and shared experts design significantly contribute to performance gains across all tested domains. To our knowledge, MoST represents the first fully open-source speech-text LLM built on a Mixture of Experts architecture. \footnote{We release MoST model, training code, inference code, and training data at https://github.com/NUS-HPC-AI-Lab/MoST
>
---
## 更新

#### [replaced 001] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD**

- **简介: 该论文属于语音建模任务，旨在解决语义与声学信息分离不彻底的问题。提出DSA-Tokenizer，通过优化约束分离语音为语义和声学token，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2601.09239v2](https://arxiv.org/pdf/2601.09239v2)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Jianping Wang; Linqi Song
>
> **备注:** Submit to ACL ARR 2026 Jaunary
>
> **摘要:** Speech tokenizers serve as the cornerstone of discrete Speech Large Language Models (Speech LLMs). Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose DSA-Tokenizer, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. To eliminate rigid length constraints between the two sequences, we introduce a hierarchical Flow-Matching decoder that further improve speech generation quality. Furthermore, We employ a joint reconstruction-recombination training strategy to enforce this separation. DSA-Tokenizer enables high fidelity reconstruction and flexible recombination through robust disentanglement, facilitating controllable generation in speech LLMs. Our analysis highlights disentangled tokenization as a pivotal paradigm for future speech modeling. Audio samples are avaialble at https://anonymous.4open.science/w/DSA_Tokenizer_demo/. The code and model will be made publicly available after the paper has been accepted.
>
---
#### [replaced 002] Audio Generation Through Score-Based Generative Modeling: Design Principles and Implementation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频生成任务，旨在解决扩散模型在音频应用中的设计与实现问题。通过分析设计原则和条件机制，提出一个统一框架并提供开源代码。**

- **链接: [https://arxiv.org/pdf/2506.08457v2](https://arxiv.org/pdf/2506.08457v2)**

> **作者:** Ge Zhu; Yutong Wen; Zhiyao Duan
>
> **备注:** Accepted by Foundations and Trends in Signal Processing
>
> **摘要:** Diffusion models have emerged as powerful deep generative techniques, producing high-quality and diverse samples in applications in various domains including audio. While existing reviews provide overviews, there remains limited in-depth discussion of these specific design choices. The audio diffusion model literature also lacks principled guidance for the implementation of these design choices and their comparisons for different applications. This survey provides a comprehensive review of diffusion model design with an emphasis on design principles for quality improvement and conditioning for audio applications. We adopt the score modeling perspective as a unifying framework that accommodates various interpretations, including recent approaches like flow matching. We systematically examine the training and sampling procedures of diffusion models, and audio applications through different conditioning mechanisms. To provide an integrated, unified codebase and to promote reproducible research and rapid prototyping, we introduce an open-source codebase (https://github.com/gzhu06/AudioDiffuser) that implements our reviewed framework for various audio applications. We demonstrate its capabilities through three case studies: audio generation, speech enhancement, and text-to-speech synthesis, with benchmark evaluations on standard datasets.
>
---
#### [replaced 003] Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文聚焦多语言对话语音识别任务，旨在解决LLM与端到端模型性能差距问题。通过融合微调的Whisper和mHuBERT编码器，提升语音表示，优化模型表现。**

- **链接: [https://arxiv.org/pdf/2601.01461v2](https://arxiv.org/pdf/2601.01461v2)**

> **作者:** Yuxiang Mei; Dongxing Xu; Jiaen Liang; Yanhua Long
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
>
---
#### [replaced 004] ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 该论文属于语音与声音深度伪造检测任务，旨在解决真实环境中组件级伪造检测难题。通过构建数据集和联合学习框架，提出ESDD2挑战赛以提升检测能力。**

- **链接: [https://arxiv.org/pdf/2601.07303v2](https://arxiv.org/pdf/2601.07303v2)**

> **作者:** Xueping Zhang; Han Yin; Yang Xiao; Lin Zhang; Ting Dang; Rohan Kumar Das; Ming Li
>
> **摘要:** Audio recorded in real-world environments often contains a mixture of foreground speech and background environmental sounds. With rapid advances in text-to-speech, voice conversion, and other generation models, either component can now be modified independently. Such component-level manipulations are harder to detect, as the remaining unaltered component can mislead the systems designed for whole deepfake audio, and they often sound more natural to human listeners. To address this gap, we have proposed CompSpoofV2 dataset and a separation-enhanced joint learning framework. CompSpoofV2 is a large-scale curated dataset designed for component-level audio anti-spoofing, which contains over 250k audio samples, with a total duration of approximately 283 hours. Based on the CompSpoofV2 and the separation-enhanced joint learning framework, we launch the Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2), focusing on component-level spoofing, where both speech and environmental sounds may be manipulated or synthesized, creating a more challenging and realistic detection scenario. The challenge will be held in conjunction with the IEEE International Conference on Multimedia and Expo 2026 (ICME 2026).
>
---
#### [replaced 005] Keep the beat going: Automatic drum transcription with momentum
- **分类: math.NA; cs.SD; eess.AS**

- **简介: 该论文属于音乐信号处理任务，旨在解决鼓点自动转录问题。通过优化部分固定非负矩阵分解，提出使用带有动量的投影梯度下降方法，提升检测精度与收敛性。**

- **链接: [https://arxiv.org/pdf/2507.12596v2](https://arxiv.org/pdf/2507.12596v2)**

> **作者:** Alisha L. Foster; Robert J. Webber
>
> **摘要:** How can we process a piece of recorded music to detect and visualize the onset of each instrument? A simple, interpretable approach is based on partially fixed nonnegative matrix factorization (NMF). Yet despite the method's simplicity, partially fixed NMF is challenging to apply because the associated optimization problem is high-dimensional and non-convex. This paper explores two optimization approaches that preserve the nonnegative structure, including a multiplicative update rule and projected gradient descent with momentum. These techniques are derived from the previous literature, but they have not been fully developed for partially fixed NMF before now. Results indicate that projected gradient descent with momentum leads to the higher accuracy among the two methods, and it satisfies stronger local convergence guarantees.
>
---
