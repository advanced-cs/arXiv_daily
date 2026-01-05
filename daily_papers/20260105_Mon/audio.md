# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 0 篇**

## 最新发布

#### [new 001] Latent Flow Matching for Expressive Singing Voice Synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决音高和微韵律表达不足的问题。通过引入条件流匹配方法，优化潜在空间分布，提升合成歌声的细腻表现力。**

- **链接: [https://arxiv.org/pdf/2601.00217v1](https://arxiv.org/pdf/2601.00217v1)**

> **作者:** Minhyeok Yun; Yong-Hoon Choi
>
> **摘要:** Conditional variational autoencoder (cVAE)-based singing voice synthesis provides efficient inference and strong audio quality by learning a score-conditioned prior and a recording-conditioned posterior latent space. However, because synthesis relies on prior samples while training uses posterior latents inferred from real recordings, imperfect distribution matching can cause a prior-posterior mismatch that degrades fine-grained expressiveness such as vibrato and micro-prosody. We propose FM-Singer, which introduces conditional flow matching (CFM) in latent space to learn a continuous vector field transporting prior latents toward posterior latents along an optimal-transport-inspired path. At inference time, the learned latent flow refines a prior sample by solving an ordinary differential equation (ODE) before waveform generation, improving expressiveness while preserving the efficiency of parallel decoding. Experiments on Korean and Chinese singing datasets demonstrate consistent improvements over strong baselines, including lower mel-cepstral distortion and fundamental-frequency error and higher perceptual scores on the Korean dataset. Code, pretrained checkpoints, and audio demos are available at https://github.com/alsgur9368/FM-Singer
>
---
#### [new 002] IKFST: IOO and KOO Algorithms for Accelerated and Precise WFST-based End-to-End Automatic Speech Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升WFST解码效率。通过分析CTC输出结构，提出两种算法，在保持精度前提下显著降低解码延迟。**

- **链接: [https://arxiv.org/pdf/2601.00160v1](https://arxiv.org/pdf/2601.00160v1)**

> **作者:** Zhuoran Zhuang; Ye Chen; Chao Luo; Tian-Hao Zhang; Xuewei Zhang; Jian Ma; Jiatong Shi; Wei Zhang
>
> **摘要:** End-to-end automatic speech recognition has become the dominant paradigm in both academia and industry. To enhance recognition performance, the Weighted Finite-State Transducer (WFST) is widely adopted to integrate acoustic and language models through static graph composition, providing robust decoding and effective error correction. However, WFST decoding relies on a frame-by-frame autoregressive search over CTC posterior probabilities, which severely limits inference efficiency. Motivated by establishing a more principled compatibility between WFST decoding and CTC modeling, we systematically study the two fundamental components of CTC outputs, namely blank and non-blank frames, and identify a key insight: blank frames primarily encode positional information, while non-blank frames carry semantic content. Building on this observation, we introduce Keep-Only-One and Insert-Only-One, two decoding algorithms that explicitly exploit the structural roles of blank and non-blank frames to achieve significantly faster WFST-based inference without compromising recognition accuracy. Experiments on large-scale in-house, AISHELL-1, and LibriSpeech datasets demonstrate state-of-the-art recognition accuracy with substantially reduced decoding latency, enabling truly efficient and high-performance WFST decoding in modern speech recognition systems.
>
---
#### [new 003] Investigating the Viability of Employing Multi-modal Large Language Models in the Context of Audio Deepfake Detection
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频深度伪造检测任务，旨在探索多模态大语言模型在该领域的可行性，通过结合音频与文本提示进行实验，验证其潜在效果。**

- **链接: [https://arxiv.org/pdf/2601.00777v1](https://arxiv.org/pdf/2601.00777v1)**

> **作者:** Akanksha Chuchra; Shukesh Reddy; Sudeepta Mishra; Abhijit Das; Abhinav Dhall
>
> **备注:** Accepted at IJCB 2025
>
> **摘要:** While Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) have shown strong generalisation in detecting image and video deepfakes, their use for audio deepfake detection remains largely unexplored. In this work, we aim to explore the potential of MLLMs for audio deepfake detection. Combining audio inputs with a range of text prompts as queries to find out the viability of MLLMs to learn robust representations across modalities for audio deepfake detection. Therefore, we attempt to explore text-aware and context-rich, question-answer based prompts with binary decisions. We hypothesise that such a feature-guided reasoning will help in facilitating deeper multimodal understanding and enable robust feature learning for audio deepfake detection. We evaluate the performance of two MLLMs, Qwen2-Audio-7B-Instruct and SALMONN, in two evaluation modes: (a) zero-shot and (b) fine-tuned. Our experiments demonstrate that combining audio with a multi-prompt approach could be a viable way forward for audio deepfake detection. Our experiments show that the models perform poorly without task-specific training and struggle to generalise to out-of-domain data. However, they achieve good performance on in-domain data with minimal supervision, indicating promising potential for audio deepfake detection.
>
---
#### [new 004] Timed text extraction from Taiwanese Kua-á-hì TV series
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于语音处理任务，旨在解决台湾歌仔戏视频数据质量低、手动处理繁琐的问题。通过OCR和SMAD技术，自动提取语音片段及歌词，提升数据准备效率。**

- **链接: [https://arxiv.org/pdf/2601.00299v1](https://arxiv.org/pdf/2601.00299v1)**

> **作者:** Tzu-Hung Huang; Yun-En Tsai; Yun-Ning Hung; Chih-Wei Wu; I-Chieh Wei; Li Su
>
> **备注:** Accepted to ISMIR 2025 Late-Breaking Demo (LBD)
>
> **摘要:** Taiwanese opera (Kua-á-hì), a major form of local theatrical tradition, underwent extensive television adaptation notably by pioneers like Iûnn Lē-hua. These videos, while potentially valuable for in-depth studies of Taiwanese opera, often have low quality and require substantial manual effort during data preparation. To streamline this process, we developed an interactive system for real-time OCR correction and a two-step approach integrating OCR-driven segmentation with Speech and Music Activity Detection (SMAD) to efficiently identify vocal segments from archival episodes with high precision. The resulting dataset, consisting of vocal segments and corresponding lyrics, can potentially supports various MIR tasks such as lyrics identification and tune retrieval. Code is available at https://github.com/z-huang/ocr-subtitle-editor .
>
---
#### [new 005] Learning Speech Representations with Variational Predictive Coding
- **分类: eess.AS; cs.CL**

- **简介: 该论文研究语音表示学习，旨在解决HuBERT目标缺乏理论基础的问题。通过变分预测编码原理改进其参数化和优化，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2601.00100v1](https://arxiv.org/pdf/2601.00100v1)**

> **作者:** Sung-Lin Yeh; Peter Bell; Hao Tang
>
> **备注:** Accepted to Transactions of the Association for Computational Linguistics (TACL); Pre MIT Press version
>
> **摘要:** Despite being the best known objective for learning speech representations, the HuBERT objective has not been further developed and improved. We argue that it is the lack of an underlying principle that stalls the development, and, in this paper, we show that predictive coding under a variational view is the principle behind the HuBERT objective. Due to its generality, our formulation provides opportunities to improve parameterization and optimization, and we show two simple modifications that bring immediate improvements to the HuBERT objective. In addition, the predictive coding formulation has tight connections to various other objectives, such as APC, CPC, wav2vec, and BEST-RQ. Empirically, the improvement in pre-training brings significant improvements to four downstream tasks: phone classification, f0 tracking, speaker recognition, and automatic speech recognition, highlighting the importance of the predictive coding interpretation.
>
---
#### [new 006] Neural Brain Fields: A NeRF-Inspired Approach for Generating Nonexistent EEG Electrodes
- **分类: eess.SP; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 该论文属于EEG信号处理任务，旨在解决EEG数据不完整、噪声大等问题。通过类比NeRF思想，构建神经脑场模型，生成不存在的电极数据并提升信号重建效果。**

- **链接: [https://arxiv.org/pdf/2601.00012v1](https://arxiv.org/pdf/2601.00012v1)**

> **作者:** Shahar Ain Kedem; Itamar Zimerman; Eliya Nachmani
>
> **摘要:** Electroencephalography (EEG) data present unique modeling challenges because recordings vary in length, exhibit very low signal to noise ratios, differ significantly across participants, drift over time within sessions, and are rarely available in large and clean datasets. Consequently, developing deep learning methods that can effectively process EEG signals remains an open and important research problem. To tackle this problem, this work presents a new method inspired by Neural Radiance Fields (NeRF). In computer vision, NeRF techniques train a neural network to memorize the appearance of a 3D scene and then uses its learned parameters to render and edit the scene from any viewpoint. We draw an analogy between the discrete images captured from different viewpoints used to learn a continuous 3D scene in NeRF, and EEG electrodes positioned at different locations on the scalp, which are used to infer the underlying representation of continuous neural activity. Building on this connection, we show that a neural network can be trained on a single EEG sample in a NeRF style manner to produce a fixed size and informative weight vector that encodes the entire signal. Moreover, via this representation we can render the EEG signal at previously unseen time steps and spatial electrode positions. We demonstrate that this approach enables continuous visualization of brain activity at any desired resolution, including ultra high resolution, and reconstruction of raw EEG signals. Finally, our empirical analysis shows that this method can effectively simulate nonexistent electrodes data in EEG recordings, allowing the reconstructed signal to be fed into standard EEG processing networks to improve performance.
>
---
#### [new 007] A Language-Agnostic Hierarchical LoRA-MoE Architecture for CTC-based Multilingual ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决大模型计算成本高、难以部署在边缘设备的问题。提出HLoRA框架，实现高效、语言无关的单次解码。**

- **链接: [https://arxiv.org/pdf/2601.00557v1](https://arxiv.org/pdf/2601.00557v1)**

> **作者:** Yuang Zheng; Yuxiang Mei; Dongxing Xu; Jie Chen; Yanhua Long
>
> **备注:** 5 pages, submitted to IEEE Signal Processing Letters
>
> **摘要:** Large-scale multilingual ASR (mASR) models such as Whisper achieve strong performance but incur high computational and latency costs, limiting their deployment on resource-constrained edge devices. In this study, we propose a lightweight and language-agnostic multilingual ASR system based on a CTC architecture with domain adaptation. Specifically, we introduce a Language-agnostic Hierarchical LoRA-MoE (HLoRA) framework integrated into an mHuBERT-CTC model, enabling end-to-end decoding via LID-posterior-driven LoRA routing. The hierarchical design consists of a multilingual shared LoRA for learning language-invariant acoustic representations and language-specific LoRA experts for modeling language-dependent characteristics. The proposed routing mechanism removes the need for prior language identity information or explicit language labels during inference, achieving true language-agnostic decoding. Experiments on MSR-86K and the MLC-SLM 2025 Challenge datasets demonstrate that HLoRA achieves competitive performance with state-of-the-art two-stage inference methods using only single-pass decoding, significantly improving decoding efficiency for low-resource mASR applications.
>
---
#### [new 008] MR-DAW: Towards Collaborative Digital Audio Workstations in Mixed Reality
- **分类: cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音乐技术领域，旨在解决传统DAW操作不便和远程协作困难的问题。通过构建MR-DAW系统，实现无束缚的多人实时协作音乐创作。**

- **链接: [https://arxiv.org/pdf/2601.00326v1](https://arxiv.org/pdf/2601.00326v1)**

> **作者:** Torin Hopkins; Shih-Yu Ma; Suibi Che-Chuan Weng; Ming-Yuan Pai; Ellen Yi-Luen Do; Luca Turchet
>
> **摘要:** Digital Audio Workstations (DAWs) are central to modern music production but often encumber the musician's workflow, tethering them to a desk and hindering natural interaction with their instrument. Furthermore, effective remote collaboration remains a significant challenge, with existing solutions hampered by network latency and asynchronous file sharing. This paper investigates the potential of Mixed Reality (MR) to overcome these barriers, creating an intuitive environment for real-time, remote musical collaboration. We employ qualitative and speculative design techniques to better understand: 1) how players currently use DAWs, and 2) to imagine a speculative future of collaborative MR-DAWs. To facilitate this discussion, we developed and evaluated the usability of a design probe, MR-DAW. An MR system enabling multiple, geographically dispersed users to control a single, shared DAW instance while moving freely in their local spaces. Our networked system enables each remote musician to use a physical foot pedal for collaborative looping, merging a familiar, hands-free interaction with a shared virtual session. Based on interviews and system evaluations with 20 musicians, we analyze current practices, report on the user experience with our MR system, and speculate on the future of musical collaboration in MR. Our results highlight the affordances of MR for unencumbered musical interaction and provide a speculative outlook on the future of remote collaborative DAWs in the Musical Metaverse.
>
---
## 更新

