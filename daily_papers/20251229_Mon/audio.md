# 音频 cs.SD;  eess.AS

- **最新发布 4 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Contextual Biasing for LLM-Based ASR with Hotword Retrieval and Reinforcement Learning
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，解决大词典下实体和关键词的上下文偏差问题。通过结合热词检索与强化学习，提升关键词识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.21828v1](https://arxiv.org/pdf/2512.21828v1)**

> **作者:** YuXiang Kong; JunFeng Hou; Jian Tang; Bingqing Zhu; Jicheng Zhang; Shaofei Xue
>
> **摘要:** Large language model (LLM)-based automatic speech recognition (ASR) has recently achieved strong performance across diverse tasks, yet contextual biasing for named entities and hotwords under large vocabularies remains challenging. In this work, we propose a scalable two-stage framework that integrates hotword retrieval with LLM-ASR adaptation. First, we extend the Global-Local Contrastive Language-Audio pre-trained model (GLCLAP) to retrieve a compact top-k set of hotword candidates from a large vocabulary via robustness-aware data augmentation and fuzzy matching. Second, we inject the retrieved candidates as textual prompts into an LLM-ASR model and fine-tune it with Generative Rejection-Based Policy Optimization (GRPO), using a task-driven reward that jointly optimizes hotword recognition and overall transcription accuracy. Experiments on hotword-focused test sets show substantial keyword error rate (KER) reductions while maintaining sentence accuracy on general ASR benchmarks, demonstrating the effectiveness of the proposed framework for large-vocabulary contextual biasing.
>
---
#### [new 002] Semantic Codebooks as Effective Priors for Neural Speech Compression
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于语音压缩任务，旨在提升压缩效率与识别性能。通过引入语义代码本作为先验，提出SemDAC模型，在低比特率下实现更优的语音重建和识别效果。**

- **链接: [https://arxiv.org/pdf/2512.21653v1](https://arxiv.org/pdf/2512.21653v1)**

> **作者:** Liuyang Bai; Weiyi Lu; Li Guo
>
> **摘要:** Speech codecs are traditionally optimized for waveform fidelity, allocating bits to preserve acoustic detail even when much of it can be inferred from linguistic structure. This leads to inefficient compression and suboptimal performance on downstream recognition tasks. We propose SemDAC, a semantic-aware neural audio codec that leverages semantic codebooks as effective priors for speech compression. In SemDAC, the first quantizer in a residual vector quantization (RVQ) stack is distilled from HuBERT features to produce semantic tokens that capture phonetic content, while subsequent quantizers model residual acoustics. A FiLM-conditioned decoder reconstructs audio conditioned on the semantic tokens, improving efficiency in the use of acoustic codebooks. Despite its simplicity, this design proves highly effective: SemDAC outperforms DAC across perceptual metrics and achieves lower WER when running Whisper on reconstructed speech, all while operating at substantially lower bitrates (e.g., 0.95 kbps vs. 2.5 kbps for DAC). These results demonstrate that semantic codebooks provide an effective inductive bias for neural speech compression, producing compact yet recognition-friendly representations.
>
---
#### [new 003] Zero-Shot to Zero-Lies: Detecting Bengali Deepfake Audio through Transfer Learning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于深度伪造音频检测任务，旨在解决 Bengali 语言中深伪音频的识别问题。通过迁移学习和微调模型，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.21702v1](https://arxiv.org/pdf/2512.21702v1)**

> **作者:** Most. Sharmin Sultana Samu; Md. Rakibul Islam; Md. Zahid Hossain; Md. Kamrozzaman Bhuiyan; Farhad Uz Zaman
>
> **备注:** Accepted for publication in 2025 28th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** The rapid growth of speech synthesis and voice conversion systems has made deepfake audio a major security concern. Bengali deepfake detection remains largely unexplored. In this work, we study automatic detection of Bengali audio deepfakes using the BanglaFake dataset. We evaluate zeroshot inference with several pretrained models. These include Wav2Vec2-XLSR-53, Whisper, PANNsCNN14, WavLM and Audio Spectrogram Transformer. Zero-shot results show limited detection ability. The best model, Wav2Vec2-XLSR-53, achieves 53.80% accuracy, 56.60% AUC and 46.20% EER. We then f ine-tune multiple architectures for Bengali deepfake detection. These include Wav2Vec2-Base, LCNN, LCNN-Attention, ResNet18, ViT-B16 and CNN-BiLSTM. Fine-tuned models show strong performance gains. ResNet18 achieves the highest accuracy of 79.17%, F1 score of 79.12%, AUC of 84.37% and EER of 24.35%. Experimental results confirm that fine-tuning significantly improves performance over zero-shot inference. This study provides the first systematic benchmark of Bengali deepfake audio detection. It highlights the effectiveness of f ine-tuned deep learning models for this low-resource language.
>
---
#### [new 004] Rare Word Recognition and Translation Without Fine-Tuning via Task Vector in Speech Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别与翻译任务，旨在解决罕见词识别与翻译问题。提出无需微调的基于任务向量的方法，提升模型性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.21894v1](https://arxiv.org/pdf/2512.21894v1)**

> **作者:** Ruihao Jing; Cheng Gong; Yu Jiang; Boyu Zhu; Shansong Liu; Chi Zhang; Xiao-Lei Zhang; Xuelong Li
>
> **摘要:** Rare words remain a critical bottleneck for speech-to-text systems. While direct fine-tuning improves recognition of target words, it often incurs high cost, catastrophic forgetting, and limited scalability. To address these challenges, we propose a training-free paradigm based on task vectors for rare word recognition and translation. By defining task vectors as parameter differences and introducing word-level task vector arithmetic, our approach enables flexible composition of rare-word capabilities, greatly enhancing scalability and reusability. Extensive experiments across multiple domains show that the proposed method matches or surpasses fine-tuned models on target words, improves general performance by about 5 BLEU, and mitigates catastrophic forgetting.
>
---
## 更新

#### [replaced 001] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出ControlAudio，解决文本引导的音频生成问题，通过渐进式扩散模型实现精确时间控制和可理解语音生成。**

- **链接: [https://arxiv.org/pdf/2510.08878v2](https://arxiv.org/pdf/2510.08878v2)**

> **作者:** Yuxuan Jiang; Zehua Chen; Zeqian Ju; Yusheng Dai; Weibei Dou; Jun Zhu
>
> **备注:** 18 pages, 8 tables, 5 figures
>
> **摘要:** Text-to-audio (TTA) generation with fine-grained control signals, e.g., precise timing control or intelligible speech content, has been explored in recent works. However, constrained by data scarcity, their generation performance at scale is still compromised. In this study, we recast controllable TTA generation as a multi-task learning problem and introduce a progressive diffusion modeling approach, ControlAudio. Our method adeptly fits distributions conditioned on more fine-grained information, including text, timing, and phoneme features, through a step-by-step strategy. First, we propose a data construction method spanning both annotation and simulation, augmenting condition information in the sequence of text, timing, and phoneme. Second, at the model training stage, we pretrain a diffusion transformer (DiT) on large-scale text-audio pairs, achieving scalable TTA generation, and then incrementally integrate the timing and phoneme features with unified semantic representations, expanding controllability. Finally, at the inference stage, we propose progressively guided generation, which sequentially emphasizes more fine-grained information, aligning inherently with the coarse-to-fine sampling nature of DiT. Extensive experiments show that ControlAudio achieves state-of-the-art performance in terms of temporal accuracy and speech clarity, significantly outperforming existing methods on both objective and subjective evaluations. Demo samples are available at: https://control-audio.github.io/Control-Audio.
>
---
#### [replaced 002] Real-Time Streamable Generative Speech Restoration with Flow Matching
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文提出Stream.FM，一种实时流式生成语音恢复模型，解决实时通信中生成模型计算耗时的问题。通过优化架构和算法，实现低延迟的语音增强等任务。**

- **链接: [https://arxiv.org/pdf/2512.19442v2](https://arxiv.org/pdf/2512.19442v2)**

> **作者:** Simon Welker; Bunlong Lay; Maris Hillemann; Tal Peer; Timo Gerkmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Diffusion-based generative models have greatly impacted the speech processing field in recent years, exhibiting high speech naturalness and spawning a new research direction. Their application in real-time communication is, however, still lagging behind due to their computation-heavy nature involving multiple calls of large DNNs. Here, we present Stream$.$FM, a frame-causal flow-based generative model with an algorithmic latency of 32 milliseconds (ms) and a total latency of 48 ms, paving the way for generative speech processing in real-time communication. We propose a buffered streaming inference scheme and an optimized DNN architecture, show how learned few-step numerical solvers can boost output quality at a fixed compute budget, explore model weight compression to find favorable points along a compute/quality tradeoff, and contribute a model variant with 24 ms total latency for the speech enhancement task. Our work looks beyond theoretical latencies, showing that high-quality streaming generative speech processing can be realized on consumer GPUs available today. Stream$.$FM can solve a variety of speech processing tasks in a streaming fashion: speech enhancement, dereverberation, codec post-filtering, bandwidth extension, STFT phase retrieval, and Mel vocoding. As we verify through comprehensive evaluations and a MUSHRA listening test, Stream$.$FM establishes a state-of-the-art for generative streaming speech restoration, exhibits only a reasonable reduction in quality compared to a non-streaming variant, and outperforms our recent work (Diffusion Buffer) on generative streaming speech enhancement while operating at a lower latency.
>
---
#### [replaced 003] SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出SpidR，一种无监督的语音表示学习模型，用于直接从语音中学习语言单位，解决语音语言建模问题。通过自监督训练提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.20308v2](https://arxiv.org/pdf/2512.20308v2)**

> **作者:** Maxime Poli; Mahi Luthra; Youssef Benchekroun; Yosuke Higuchi; Martin Gleize; Jiayi Shen; Robin Algayres; Yu-An Chung; Mido Assran; Juan Pino; Emmanuel Dupoux
>
> **备注:** Published in Transactions on Machine Learning Research. 30 pages, 16 figures
>
> **摘要:** The parallel advances in language modeling and speech representation learning have raised the prospect of learning language directly from speech without textual intermediates. This requires extracting semantic representations directly from speech. Our contributions are threefold. First, we introduce SpidR, a self-supervised speech representation model that efficiently learns representations with highly accessible phonetic information, which makes it particularly suited for textless spoken language modeling. It is trained on raw waveforms using a masked prediction objective combined with self-distillation and online clustering. The intermediate layers of the student model learn to predict assignments derived from the teacher's intermediate layers. This learning objective stabilizes the online clustering procedure compared to previous approaches, resulting in higher quality codebooks. SpidR outperforms wav2vec 2.0, HuBERT, WavLM, and DinoSR on downstream language modeling benchmarks (sWUGGY, sBLIMP, tSC). Second, we systematically evaluate across models and layers the correlation between speech unit quality (ABX, PNMI) and language modeling performance, validating these metrics as reliable proxies. Finally, SpidR significantly reduces pretraining time compared to HuBERT, requiring only one day of pretraining on 16 GPUs, instead of a week. This speedup is enabled by the pretraining method and an efficient codebase, which allows faster iteration and easier experimentation. We open-source the training code and model checkpoints at https://github.com/facebookresearch/spidr.
>
---
#### [replaced 004] Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于视频到音频生成任务，解决模型生成无视觉依据的音频问题（插入幻觉）。通过构建评估框架和提出新方法HALCON，有效减少幻觉现象。**

- **链接: [https://arxiv.org/pdf/2510.08078v5](https://arxiv.org/pdf/2510.08078v5)**

> **作者:** Liyang Chen; Hongkai Chen; Yujun Cai; Sifan Li; Qingwen Ye; Yiwei Wang
>
> **摘要:** Video-to-Audio generation has made remarkable strides in automatically synthesizing sound for video. However, existing evaluation metrics, which focus on semantic and temporal alignment, overlook a critical failure mode: models often generate acoustic events, particularly speech and music, that have no corresponding visual source. We term this phenomenon Insertion Hallucination and identify it as a systemic risk driven by dataset biases, such as the prevalence of off-screen sounds, that remains completely undetected by current metrics. To address this challenge, we first develop a systematic evaluation framework that employs a majority-voting ensemble of multiple audio event detectors. We also introduce two novel metrics to quantify the prevalence and severity of this issue: IH@vid (the fraction of videos with hallucinations) and IH@dur (the fraction of hallucinated duration). Building on this, we introduce HALCON to mitigate IH. HALCON follows a three-stage procedure: it first generates initial audio to expose hallucinated segments, then identifies and masks the corresponding unreliable video features, and finally regenerates the audio using the corrected conditioning. Experiments on several mainstream V2A benchmarks first reveal that state-of-the-art models suffer from severe IH. In contrast, our HALCON method reduces both the prevalence and duration of hallucinations by over 50\% on average, without degrading, and in some cases even improving, conventional metrics for audio quality and temporal synchronization. Our work is the first to formally define, systematically measure, and effectively mitigate Insertion Hallucination, paving the way for more reliable and faithful V2A models.
>
---
#### [replaced 005] Fine-grained Preference Optimization Improves Zero-shot Text-to-Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音生成任务，旨在解决TTS系统中局部音频质量问题。通过细粒度偏好优化方法，提升系统的鲁棒性和 intelligibility。**

- **链接: [https://arxiv.org/pdf/2502.02950v2](https://arxiv.org/pdf/2502.02950v2)**

> **作者:** Jixun Yao; Yuguang Yang; Yu Pan; Yuan Feng; Ziqian Ning; Jianhao Ye; Hongbin Zhou; Lei Xie
>
> **备注:** Accepted By IEEE TASLP
>
> **摘要:** Integrating human feedback to align text-to-speech (TTS) system outputs with human preferences has proven to be an effective approach for enhancing the robustness of language model-based TTS systems. Current approaches primarily focus on using preference data annotated at the utterance level. However, frequent issues that affect the listening experience often only arise in specific segments of audio samples, while other segments are well-generated. In this study, we propose a fine-grained preference optimization approach (FPO) to enhance the robustness of TTS systems. FPO focuses on addressing localized issues in generated samples rather than uniformly optimizing the entire utterance. Specifically, we first analyze the types of issues in generated samples, categorize them into two groups, and propose a selective training loss strategy to optimize preferences based on fine-grained labels for each issue type. Experimental results show that FPO enhances the robustness of zero-shot TTS systems by effectively addressing local issues, significantly reducing the bad case ratio, and improving intelligibility. Furthermore, FPO exhibits superior data efficiency compared with baseline systems, achieving similar performance with fewer training samples.
>
---
#### [replaced 006] Vector Signal Reconstruction Sparse and Parametric Approach of direction of arrival Using Single Vector Hydrophone
- **分类: cs.SD**

- **简介: 论文提出一种基于单矢量水听器的DOA估计方法，解决多源和噪声环境下传统方法精度不足的问题。通过信号重构与SPA算法优化，提升估计准确性和分辨率。**

- **链接: [https://arxiv.org/pdf/2404.15160v2](https://arxiv.org/pdf/2404.15160v2)**

> **作者:** Jiabin Guo
>
> **备注:** The authors have determined that the simulation results presented are preliminary and insufficient. Further simulation work is required to validate the conclusions. The text also requires major linguistic improvements
>
> **摘要:** This article discusses the application of single vector hydrophones in the field of underwater acoustic signal processing for Direction Of Arrival (DOA) estimation. Addressing the limitations of traditional DOA estimation methods in multi-source environments and under noise interference, this study introduces a Vector Signal Reconstruction Sparse and Parametric Approach (VSRSPA). This method involves reconstructing the signal model of a single vector hydrophone, converting its covariance matrix into a Toeplitz structure suitable for the Sparse and Parametric Approach (SPA) algorithm. The process then optimizes it using the SPA algorithm to achieve more accurate DOA estimation. Through detailed simulation analysis, this research has confirmed the performance of the proposed algorithm in single and dual-target DOA estimation scenarios, especially under various signal-to-noise ratio(SNR) conditions. The simulation results show that, compared to traditional DOA estimation methods, this algorithm has significant advantages in estimation accuracy and resolution, particularly in multi-source signals and low SNR environments. The contribution of this study lies in providing an effective new method for DOA estimation with single vector hydrophones in complex environments, introducing new research directions and solutions in the field of vector hydrophone signal processing.
>
---
