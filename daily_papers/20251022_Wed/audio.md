# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] Adaptive Per-Channel Energy Normalization Front-end for Robust Audio Signal Processing
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文针对音频信号处理前端在复杂环境中鲁棒性不足的问题，提出一种基于神经控制器的自适应前端。通过动态调整通道能量归一化，实现输入依赖的实时优化，在多种音频分类任务中提升了抗噪与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.18206v1](http://arxiv.org/pdf/2510.18206v1)**

> **作者:** Hanyu Meng; Vidhyasaharan Sethu; Eliathamby Ambikairajah; Qiquan Zhang; Haizhou Li
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** In audio signal processing, learnable front-ends have shown strong performance across diverse tasks by optimizing task-specific representation. However, their parameters remain fixed once trained, lacking flexibility during inference and limiting robustness under dynamic complex acoustic environments. In this paper, we introduce a novel adaptive paradigm for audio front-ends that replaces static parameterization with a closed-loop neural controller. Specifically, we simplify the learnable front-end LEAF architecture and integrate a neural controller for adaptive representation via dynamically tuning Per-Channel Energy Normalization. The neural controller leverages both the current and the buffered past subband energies to enable input-dependent adaptation during inference. Experimental results on multiple audio classification tasks demonstrate that the proposed adaptive front-end consistently outperforms prior fixed and learnable front-ends under both clean and complex acoustic conditions. These results highlight neural adaptability as a promising direction for the next generation of audio front-ends.
>
---
#### [new 002] Diffusion Buffer for Online Generative Speech Enhancement
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究在线生成式语音增强任务，旨在解决传统生成模型计算复杂度高、难以实时处理的问题。作者提出“扩散缓冲”方法，通过时序对齐与新型网络结构，实现单帧单次推理，显著降低延迟并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.18744v1](http://arxiv.org/pdf/2510.18744v1)**

> **作者:** Bunlong Lay; Rostislav Makarov; Simon Welker; Maris Hillemann; Timo Gerkmann
>
> **摘要:** Online Speech Enhancement was mainly reserved for predictive models. A key advantage of these models is that for an incoming signal frame from a stream of data, the model is called only once for enhancement. In contrast, generative Speech Enhancement models often require multiple calls, resulting in a computational complexity that is too high for many online speech enhancement applications. This work presents the Diffusion Buffer, a generative diffusion-based Speech Enhancement model which only requires one neural network call per incoming signal frame from a stream of data and performs enhancement in an online fashion on a consumer-grade GPU. The key idea of the Diffusion Buffer is to align physical time with Diffusion time-steps. The approach progressively denoises frames through physical time, where past frames have more noise removed. Consequently, an enhanced frame is output to the listener with a delay defined by the Diffusion Buffer, and the output frame has a corresponding look-ahead. In this work, we extend upon our previous work by carefully designing a 2D convolutional UNet architecture that specifically aligns with the Diffusion Buffer's look-ahead. We observe that the proposed UNet improves performance, particularly when the algorithmic latency is low. Moreover, we show that using a Data Prediction loss instead of Denoising Score Matching loss enables flexible control over the trade-off between algorithmic latency and quality during inference. The extended Diffusion Buffer equipped with a novel NN and loss function drastically reduces the algorithmic latency from 320 - 960 ms to 32 - 176 ms with an even increased performance. While it has been shown before that offline generative diffusion models outperform predictive approaches in unseen noisy speech data, we confirm that the online Diffusion Buffer also outperforms its predictive counterpart on unseen noisy speech data.
>
---
#### [new 003] Hearing Health in Home Healthcare: Leveraging LLMs for Illness Scoring and ALMs for Vocal Biomarker Extraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文探索家庭医疗中基于语音的健康评估，利用大语言模型（LLM）融合SOAP笔记与生命体征生成疾病评分，并设计音频预处理流程，结合音频语言模型（ALM）提取可解释的声学生物标志物，验证其与健康状态的关联，实现对患者健康的自动监测。**

- **链接: [http://arxiv.org/pdf/2510.18169v1](http://arxiv.org/pdf/2510.18169v1)**

> **作者:** Yu-Wen Chen; William Ho; Sasha M. Vergez; Grace Flaherty; Pallavi Gupta; Zhihong Zhang; Maryam Zolnoori; Margaret V. McDonald; Maxim Topaz; Zoran Kostic; Julia Hirschberg
>
> **备注:** The Second Workshop on GenAI for Health at NeurIPS 2025
>
> **摘要:** The growing demand for home healthcare calls for tools that can support care delivery. In this study, we explore automatic health assessment from voice using real-world home care visit data, leveraging the diverse patient information it contains. First, we utilize Large Language Models (LLMs) to integrate Subjective, Objective, Assessment, and Plan (SOAP) notes derived from unstructured audio transcripts and structured vital signs into a holistic illness score that reflects a patient's overall health. This compact representation facilitates cross-visit health status comparisons and downstream analysis. Next, we design a multi-stage preprocessing pipeline to extract short speech segments from target speakers in home care recordings for acoustic analysis. We then employ an Audio Language Model (ALM) to produce plain-language descriptions of vocal biomarkers and examine their association with individuals' health status. Our experimental results benchmark both commercial and open-source LLMs in estimating illness scores, demonstrating their alignment with actual clinical outcomes, and revealing that SOAP notes are substantially more informative than vital signs. Building on the illness scores, we provide the first evidence that ALMs can identify health-related acoustic patterns from home care recordings and present them in a human-readable form. Together, these findings highlight the potential of LLMs and ALMs to harness heterogeneous in-home visit data for better patient monitoring and care.
>
---
#### [new 004] SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SAC，一种语义-声学双流神经语音编解码器，旨在解决现有编解码器在语音重建质量与语义表征能力间的权衡问题。通过分离语义和声学建模，提升语音还原质量及语义表达能力，适用于生成与理解任务。**

- **链接: [http://arxiv.org/pdf/2510.16841v1](http://arxiv.org/pdf/2510.16841v1)**

> **作者:** Wenxi Chen; Xinsheng Wang; Ruiqi Yan; Yushen Chen; Zhikang Niu; Ziyang Ma; Xiquan Li; Yuzhe Liang; Hanlin Wen; Shunshun Yin; Ming Tao; Xie Chen
>
> **摘要:** Speech codecs that convert continuous speech signals into discrete tokens have become essential for speech language models (SLMs). However, existing codecs struggle to balance high-quality reconstruction with semantically rich representations, limiting their effectiveness in both generative and understanding tasks. In this work, we propose SAC, a neural speech codec with semantic-acoustic dual-stream quantization. By disentangling semantic and acoustic modeling into two dedicated streams, SAC enables each to be optimized for its respective role. Comprehensive evaluations show that SAC achieves strong reconstruction performance across diverse bitrates under both clean and noisy conditions, with particularly high scores on UTMOS and WER, demonstrating superior perceptual quality and intelligibility. Moreover, SAC substantially outperforms state-of-the-art codecs in semantic representation, achieving a level comparable to that of self-supervised learning (SSL) continuous embeddings. Finally, our analysis of speech disentanglement highlights the effectiveness of the dual-stream design, offering new potential for controllable speech applications.
>
---
#### [new 005] SegTune: Structured and Fine-Grained Control for Song Generation
- **分类: cs.SD**

- **简介: 该论文研究歌曲生成任务，旨在解决现有方法缺乏对音乐结构与动态的细粒度控制问题。提出SegTune框架，支持段落级可控生成，结合局部与全局文本提示，并引入LLM时长预测器实现精准歌词对齐。**

- **链接: [http://arxiv.org/pdf/2510.18416v1](http://arxiv.org/pdf/2510.18416v1)**

> **作者:** Pengfei Cai; Joanna Wang; Haorui Zheng; Xu Li; Zihao Ji; Teng Ma; Zhongliang Liu; Chen Zhang; Pengfei Wan
>
> **摘要:** Recent advancements in song generation have shown promising results in generating songs from lyrics and/or global text prompts. However, most existing systems lack the ability to model the temporally varying attributes of songs, limiting fine-grained control over musical structure and dynamics. In this paper, we propose SegTune, a non-autoregressive framework for structured and controllable song generation. SegTune enables segment-level control by allowing users or large language models to specify local musical descriptions aligned to song sections.The segmental prompts are injected into the model by temporally broadcasting them to corresponding time windows, while global prompts influence the whole song to ensure stylistic coherence. To obtain accurate segment durations and enable precise lyric-to-music alignment, we introduce an LLM-based duration predictor that autoregressively generates sentence-level timestamped lyrics in LRC format. We further construct a large-scale data pipeline for collecting high-quality songs with aligned lyrics and prompts, and propose new evaluation metrics to assess segment-level alignment and vocal attribute consistency. Experimental results show that SegTune achieves superior controllability and musical coherence compared to existing baselines. See https://cai525.github.io/SegTune_demo for demos of our work.
>
---
#### [new 006] A Stage-Wise Learning Strategy with Fixed Anchors for Robust Speaker Verification
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究鲁棒说话人验证任务，旨在解决噪声环境下表征学习中区分性与抗噪性的平衡问题。提出一种基于固定锚点的分阶段学习方法，先训练基础模型获取锚点，再通过锚点约束微调模型，提升噪声下的身份保持能力。**

- **链接: [http://arxiv.org/pdf/2510.18530v1](http://arxiv.org/pdf/2510.18530v1)**

> **作者:** Bin Gu; Lipeng Dai; Huipeng Du; Haitao Zhao; Jibo Wei
>
> **摘要:** Learning robust speaker representations under noisy conditions presents significant challenges, which requires careful handling of both discriminative and noise-invariant properties. In this work, we proposed an anchor-based stage-wise learning strategy for robust speaker representation learning. Specifically, our approach begins by training a base model to establish discriminative speaker boundaries, and then extract anchor embeddings from this model as stable references. Finally, a copy of the base model is fine-tuned on noisy inputs, regularized by enforcing proximity to their corresponding fixed anchor embeddings to preserve speaker identity under distortion. Experimental results suggest that this strategy offers advantages over conventional joint optimization, particularly in maintaining discrimination while improving noise robustness. The proposed method demonstrates consistent improvements across various noise conditions, potentially due to its ability to handle boundary stabilization and variation suppression separately.
>
---
#### [new 007] ParaStyleTTS: Toward Efficient and Robust Paralinguistic Style Control for Expressive Text-to-Speech Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究文本到语音（TTS）中的说话风格控制任务，旨在解决现有方法依赖音频参考或大模型导致的效率低、鲁棒性差问题。提出ParaStyleTTS，通过轻量级双层风格架构实现高效、可解释的副语言风格控制，支持情感、性别、年龄等细粒度调节。**

- **链接: [http://arxiv.org/pdf/2510.18308v1](http://arxiv.org/pdf/2510.18308v1)**

> **作者:** Haowei Lou; Hye-Young Paik; Wen Hu; Lina Yao
>
> **摘要:** Controlling speaking style in text-to-speech (TTS) systems has become a growing focus in both academia and industry. While many existing approaches rely on reference audio to guide style generation, such methods are often impractical due to privacy concerns and limited accessibility. More recently, large language models (LLMs) have been used to control speaking style through natural language prompts; however, their high computational cost, lack of interpretability, and sensitivity to prompt phrasing limit their applicability in real-time and resource-constrained environments. In this work, we propose ParaStyleTTS, a lightweight and interpretable TTS framework that enables expressive style control from text prompts alone. ParaStyleTTS features a novel two-level style adaptation architecture that separates prosodic and paralinguistic speech style modeling. It allows fine-grained and robust control over factors such as emotion, gender, and age. Unlike LLM-based methods, ParaStyleTTS maintains consistent style realization across varied prompt formulations and is well-suited for real-world applications, including on-device and low-resource deployment. Experimental results show that ParaStyleTTS generates high-quality speech with performance comparable to state-of-the-art LLM-based systems while being 30x faster, using 8x fewer parameters, and requiring 2.5x less CUDA memory. Moreover, ParaStyleTTS exhibits superior robustness and controllability over paralinguistic speaking styles, providing a practical and efficient solution for style-controllable text-to-speech generation. Demo can be found at https://parastyletts.github.io/ParaStyleTTS_Demo/. Code can be found at https://github.com/haoweilou/ParaStyleTTS.
>
---
#### [new 008] MVDR Beamforming for Cyclostationary Processes
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究低信噪比下的噪声抑制任务，针对传统MVDR波束成形器无法利用非平稳周期性噪声频谱相关性的问题，提出基于FRESH滤波的cMVDR方法，通过数据驱动估计共振频率并引入频移滤波，有效提升对乐器、风扇等周期性噪声的抑制能力。**

- **链接: [http://arxiv.org/pdf/2510.18391v1](http://arxiv.org/pdf/2510.18391v1)**

> **作者:** Giovanni Bologni; Martin Bo Møller; Richard Heusdens; Richard C. Hendriks
>
> **备注:** Under review for publication from September 2025
>
> **摘要:** Conventional acoustic beamformers assume that noise is stationary within short time frames. This assumption prevents them from exploiting correlations between frequencies in almost-periodic noise sources such as musical instruments, fans, and engines. These signals exhibit periodically varying statistics and are better modeled as cyclostationary processes. This paper introduces the cyclic MVDR (cMVDR) beamformer, an extension of the conventional MVDR that leverages both spatial and spectral correlations to improve noise reduction, particularly in low-SNR scenarios. The method builds on frequency-shifted (FRESH) filtering, where shifted versions of the input are combined to attenuate or amplify components that are coherent across frequency. To address inharmonicity, where harmonic partials deviate from exact integer multiples of the fundamental frequency, we propose a data-driven strategy that estimates resonant frequencies via periodogram analysis and computes the frequency shifts from their spacing. Analytical and experimental results demonstrate that performance improves with increasing spectral correlation. On real recordings, the cMVDR achieves up to 5 dB gain in scale-invariant signal-to-distortion ratio (SI-SDR) over the MVDR and remains effective even with a single microphone. Code is available at https://github.com/Screeen/cMVDR.
>
---
#### [new 009] ProLAP: Probabilistic Language-Audio Pre-Training
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究语言-音频联合表示学习，旨在解决传统方法忽略多对多语义关系的问题。提出ProLAP框架，采用概率化建模和层次化损失函数，在小数据下有效学习语义层级结构，提升音频-文本检索与语义理解性能。**

- **链接: [http://arxiv.org/pdf/2510.18423v1](http://arxiv.org/pdf/2510.18423v1)**

> **作者:** Toranosuke Manabe; Yuchi Ishikawa; Hokuto Munakata; Tatsuya Komatsu
>
> **备注:** Under review
>
> **摘要:** Language-audio joint representation learning frameworks typically depend on deterministic embeddings, assuming a one-to-one correspondence between audio and text. In real-world settings, however, the language-audio relationship is inherently many-to-many: one audio segment can be described by multiple captions and vice versa. To address this, we propose Probabilistic Language-Audio Pre-training (ProLAP), which models multiplicity as the spread of probability distributions in a joint language-audio embedding space. To train the intra-modal hierarchical relationship effectively, we also introduce two objectives: (i) hierarchical inclusion loss to promote semantic hierarchical understanding of inputs and (ii) mask repulsive loss to improve the efficiency of learning when optimizing the hierarchical inclusion loss. With this training strategy, our model can learn the hierarchical structure inherent in the data even from small datasets, in contrast to prior probabilistic approaches that rely on large-scale datasets. In our experiments, ProLAP outperforms existing deterministic approaches on audio-text retrieval tasks. Moreover, through experiments on the audio traversal task introduced in this paper, we demonstrate that ProLAP captures the plausible semantic hierarchy.
>
---
#### [new 010] Joint Estimation of Piano Dynamics and Metrical Structure with a Multi-task Multi-Scale Network
- **分类: eess.AS; cs.LG; cs.SD; H.5.5; I.2.6; I.5.4**

- **简介: 该论文提出一种多任务多尺度网络，联合估计钢琴演奏的力度、变化点、节拍和强拍，解决音频中音乐表现分析问题。以Bark尺度响度为输入，大幅减小模型规模，支持长序列输入，在MazurkaBL数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2510.18190v1](http://arxiv.org/pdf/2510.18190v1)**

> **作者:** Zhanhong He; Hanyu Meng; David Huang; Roberto Togneri
>
> **备注:** Paper submitted to ICASSP2026
>
> **摘要:** Estimating piano dynamic from audio recordings is a fundamental challenge in computational music analysis. In this paper, we propose an efficient multi-task network that jointly predicts dynamic levels, change points, beats, and downbeats from a shared latent representation. These four targets form the metrical structure of dynamics in the music score. Inspired by recent vocal dynamic research, we use a multi-scale network as the backbone, which takes Bark-scale specific loudness as the input feature. Compared to log-Mel as input, this reduces model size from 14.7 M to 0.5 M, enabling long sequential input. We use a 60-second audio length in audio segmentation, which doubled the length of beat tracking commonly used. Evaluated on the public MazurkaBL dataset, our model achieves state-of-the-art results across all tasks. This work sets a new benchmark for piano dynamic estimation and delivers a powerful and compact tool, paving the way for large-scale, resource-efficient analysis of musical expression.
>
---
#### [new 011] Transformer Redesign for Late Fusion of Audio-Text Features on Ultra-Low-Power Edge Hardware
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究超低功耗边缘设备上的多模态情感识别任务，解决现有模型难以在资源受限设备上实现实时、高效音频-文本融合的问题。作者提出一种面向Edge TPU的硬件感知晚融合架构，结合量化Transformer与冻结关键词嵌入，在极低资源下实现高精度实时推理。**

- **链接: [http://arxiv.org/pdf/2510.18036v1](http://arxiv.org/pdf/2510.18036v1)**

> **作者:** Stavros Mitsis; Ermos Hadjikyriakos; Humaid Ibrahim; Savvas Neofytou; Shashwat Raman; James Myles; Eiman Kanjo
>
> **摘要:** Deploying emotion recognition systems in real-world environments where devices must be small, low-power, and private remains a significant challenge. This is especially relevant for applications such as tension monitoring, conflict de-escalation, and responsive wearables, where cloud-based solutions are impractical. Multimodal emotion recognition has advanced through deep learning, but most systems remain unsuitable for deployment on ultra-constrained edge devices. Prior work typically relies on powerful hardware, lacks real-time performance, or uses unimodal input. This paper addresses that gap by presenting a hardware-aware emotion recognition system that combines acoustic and linguistic features using a late-fusion architecture optimised for Edge TPU. The design integrates a quantised transformer-based acoustic model with frozen keyword embeddings from a DSResNet-SE network, enabling real-time inference within a 1.8MB memory budget and 21-23ms latency. The pipeline ensures spectrogram alignment between training and deployment using MicroFrontend and MLTK. Evaluation on re-recorded, segmented IEMOCAP samples captured through the Coral Dev Board Micro microphone shows a 6.3% macro F1 improvement over unimodal baselines. This work demonstrates that accurate, real-time multimodal emotion inference is achievable on microcontroller-class edge platforms through task-specific fusion and hardware-guided model design.
>
---
#### [new 012] Noise-Conditioned Mixture-of-Experts Framework for Robust Speaker Verification
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文针对噪声环境下说话人验证的鲁棒性问题，提出一种噪声条件化的混合专家框架。通过噪声感知的专家路由、专家专业化策略和信噪比衰减课程学习，实现噪声自适应特征建模，提升复杂噪声下的验证性能。**

- **链接: [http://arxiv.org/pdf/2510.18533v1](http://arxiv.org/pdf/2510.18533v1)**

> **作者:** Bin Gu; Lipeng Dai; Huipeng Du; Haitao Zhao; Jibo Wei
>
> **摘要:** Robust speaker verification under noisy conditions remains an open challenge. Conventional deep learning methods learn a robust unified speaker representation space against diverse background noise and achieve significant improvement. In contrast, this paper presents a noise-conditioned mixture-ofexperts framework that decomposes the feature space into specialized noise-aware subspaces for speaker verification. Specifically, we propose a noise-conditioned expert routing mechanism, a universal model based expert specialization strategy, and an SNR-decaying curriculum learning protocol, collectively improving model robustness and generalization under diverse noise conditions. The proposed method can automatically route inputs to expert networks based on noise information derived from the inputs, where each expert targets distinct noise characteristics while preserving speaker identity information. Comprehensive experiments demonstrate consistent superiority over baselines, confirming that explicit noise-dependent feature modeling significantly enhances robustness without sacrificing verification accuracy.
>
---
#### [new 013] Bayesian Low-Rank Factorization for Robust Model Adaptation
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究语音基础模型的鲁棒自适应，解决直接微调易过拟合和灾难性遗忘的问题。提出贝叶斯低秩适配器，在保持泛化的同时有效适应多语言混杂场景，相较LoRA显著减少遗忘。**

- **链接: [http://arxiv.org/pdf/2510.18723v1](http://arxiv.org/pdf/2510.18723v1)**

> **作者:** Enes Yavuz Ugan; Ngoc-Quan Pham; Alexander Waibel
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large speech foundation models achieve strong performance across many domains, but they often require adaptation to handle local needs such as code-switching, where speakers mix languages within the same utterance. Direct fine-tuning of these models risks overfitting to the target domain and overwriting the broad capabilities of the base model. To address this challenge, we explore Bayesian factorized adapters for speech foundation models, which place priors near zero to achieve sparser adaptation matrices and thereby retain general performance while adapting to specific domains. We apply our approach to the Whisper model and evaluate on different multilingual code-switching scenarios. Our results show only minimal adaptation loss while significantly reducing catastrophic forgetting of the base model. Compared to LoRA, our method achieves a backward gain of 54% with only a 4% drop on the new domain. These findings highlight the effectiveness of Bayesian adaptation for fine-tuning speech foundation models without sacrificing generalization.
>
---
#### [new 014] Adapting Language Balance in Code-Switching Speech
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究代码转换语音中的语言平衡问题，旨在提升大模型对混合语言场景的鲁棒性。通过引入可微代理信号突出切换点，增强模型对切换位置的识别，减少替换错误，改善生成中的上下文偏差。**

- **链接: [http://arxiv.org/pdf/2510.18724v1](http://arxiv.org/pdf/2510.18724v1)**

> **作者:** Enes Yavuz Ugan; Ngoc-Quan Pham; Alexander Waibel
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Despite achieving impressive results on standard benchmarks, large foundational models still struggle against code-switching test cases. When data scarcity cannot be used as the usual justification for poor performance, the reason may lie in the infrequent occurrence of code-switched moments, where the embedding of the second language appears subtly. Instead of expecting the models to learn this infrequency on their own, it might be beneficial to provide the training process with labels. Evaluating model performance on code-switching data requires careful localization of code-switching points where recognition errors are most consequential, so that the analysis emphasizes mistakes occurring at those moments. Building on this observation, we leverage the difference between the embedded and the main language to highlight those code-switching points and thereby emphasize learning at those locations. This simple yet effective differentiable surrogate mitigates context bias during generation -- the central challenge in code-switching -- thereby improving the model's robustness. Our experiments with Arabic and Chinese-English showed that the models are able to predict the switching places more correctly, reflected by the reduced substitution error.
>
---
#### [new 015] MLMA: Towards Multilingual with Mamba Based Architectures
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究多语言语音识别任务，旨在解决高低资源语言性能不平衡问题。作者提出MLMA，基于Mamba架构构建高效、可扩展的多语言ASR模型，通过共享表示和语言感知机制提升跨语言识别效果。**

- **链接: [http://arxiv.org/pdf/2510.18684v1](http://arxiv.org/pdf/2510.18684v1)**

> **作者:** Mohamed Nabih Ali; Daniele Falavigna; Alessio Brutti
>
> **备注:** The paper is under review at ICASSP 2026
>
> **摘要:** Multilingual automatic speech recognition (ASR) remains a challenging task, especially when balancing performance across high- and low-resource languages. Recent advances in sequence modeling suggest that architectures beyond Transformers may offer better scalability and efficiency. In this work, we introduce MLMA (Multilingual Language Modeling with Mamba for ASR), a new approach that leverages the Mamba architecture--an efficient state-space model optimized for long-context sequence processing--for multilingual ASR. Using Mamba, MLMA implicitly incorporates language-aware conditioning and shared representations to support robust recognition across diverse languages. Experiments on standard multilingual benchmarks show that MLMA achieves competitive performance compared to Transformer-based architectures. These results highlight Mamba's potential as a strong backbone for scalable, efficient, and accurate multilingual speech recognition.
>
---
## 更新

#### [replaced 001] Lightweight and Robust Multi-Channel End-to-End Speech Recognition with Spherical Harmonic Transform
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2506.11630v2](http://arxiv.org/pdf/2506.11630v2)**

> **作者:** Xiangzhu Kong; Huang Hao; Zhijian Ou
>
> **备注:** Interspeech 2025
>
> **摘要:** This paper presents SHTNet, a lightweight spherical harmonic transform (SHT) based framework, which is designed to address cross-array generalization challenges in multi-channel automatic speech recognition (ASR) through three key innovations. First, SHT based spatial sound field decomposition converts microphone signals into geometry-invariant spherical harmonic coefficients, isolating signal processing from array geometry. Second, the Spatio-Spectral Attention Fusion Network (SSAFN) combines coordinate-aware spatial modeling, refined self-attention channel combinator, and spectral noise suppression without conventional beamforming. Third, Rand-SHT training enhances robustness through random channel selection and array geometry reconstruction. The system achieves 39.26\% average CER across heterogeneous arrays (e.g., circular, square, and binaural) on datasets including Aishell-4, Alimeeting, and XMOS, with 97.1\% fewer computations than conventional neural beamformers.
>
---
#### [replaced 002] 3D Audio-Visual Segmentation
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02236v2](http://arxiv.org/pdf/2411.02236v2)**

> **作者:** Artem Sokolov; Swapnil Bhosale; Xiatian Zhu
>
> **备注:** Accepted at the NeurIPS 2024 Workshop on Audio Imagination; this version updates the project page link
>
> **摘要:** Recognizing the sounding objects in scenes is a longstanding objective in embodied AI, with diverse applications in robotics and AR/VR/MR. To that end, Audio-Visual Segmentation (AVS), taking as condition an audio signal to identify the masks of the target sounding objects in an input image with synchronous camera and microphone sensors, has been recently advanced. However, this paradigm is still insufficient for real-world operation, as the mapping from 2D images to 3D scenes is missing. To address this fundamental limitation, we introduce a novel research problem, 3D Audio-Visual Segmentation, extending the existing AVS to the 3D output space. This problem poses more challenges due to variations in camera extrinsics, audio scattering, occlusions, and diverse acoustics across sounding object categories. To facilitate this research, we create the very first simulation based benchmark, 3DAVS-S34-O7, providing photorealistic 3D scene environments with grounded spatial audio under single-instance and multi-instance settings, across 34 scenes and 7 object categories. This is made possible by re-purposing the Habitat simulator to generate comprehensive annotations of sounding object locations and corresponding 3D masks. Subsequently, we propose a new approach, EchoSegnet, characterized by integrating the ready-to-use knowledge from pretrained 2D audio-visual foundation models synergistically with 3D visual scene representation through spatial audio-aware mask alignment and refinement. Extensive experiments demonstrate that EchoSegnet can effectively segment sounding objects in 3D space on our new benchmark, representing a significant advancement in the field of embodied AI. Project page: https://x-up-lab.github.io/research/3d-audio-visual-segmentation/
>
---
#### [replaced 003] Post-training for Deepfake Speech Detection
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21090v4](http://arxiv.org/pdf/2506.21090v4)**

> **作者:** Wanying Ge; Xin Wang; Xuechen Liu; Junichi Yamagishi
>
> **备注:** Corrected previous implementation of EER calculation. Slight numerical changes in some of the results
>
> **摘要:** We introduce a post-training approach that adapts self-supervised learning (SSL) models for deepfake speech detection by bridging the gap between general pre-training and domain-specific fine-tuning. We present AntiDeepfake models, a series of post-trained models developed using a large-scale multilingual speech dataset containing over 56,000 hours of genuine speech and 18,000 hours of speech with various artifacts in over one hundred languages. Experimental results show that the post-trained models already exhibit strong robustness and generalization to unseen deepfake speech. When they are further fine-tuned on the Deepfake-Eval-2024 dataset, these models consistently surpass existing state-of-the-art detectors that do not leverage post-training. Model checkpoints and source code are available online.
>
---
