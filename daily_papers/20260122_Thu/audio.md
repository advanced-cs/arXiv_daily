# 音频 cs.SD;  eess.AS

- **最新发布 25 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Bangla Music Genre Classification Using Bidirectional LSTMS
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐流派分类任务，旨在解决Bangla音乐自动分类问题。构建了包含10个流派的数据集，使用LSTM和MFCC进行分类，准确率达78%。**

- **链接: [https://arxiv.org/pdf/2601.15083v1](https://arxiv.org/pdf/2601.15083v1)**

> **作者:** Muntakimur Rahaman; Md Mahmudul Hoque; Md Mehedi Hassain
>
> **摘要:** Bangla music is enrich in its own music cultures. Now a days music genre classification is very significant because of the exponential increase in available music, both in digital and physical formats. It is necessary to index them accordingly to facilitate improved retrieval. Automatically classifying Bangla music by genre is essential for efficiently locating specific pieces within a vast and diverse music library. Prevailing methods for genre classification predominantly employ conventional machine learning or deep learning approaches. This work introduces a novel music dataset comprising ten distinct genres of Bangla music. For the task of audio classification, we utilize a recurrent neural network (RNN) architecture. Specifically, a Long Short-Term Memory (LSTM) network is implemented to train the model and perform the classification. Feature extraction represents a foundational stage in audio data processing. This study utilizes Mel-Frequency Cepstral Coefficients (MFCCs) to transform raw audio waveforms into a compact and representative set of features. The proposed framework facilitates music genre classification by leveraging these extracted features. Experimental results demonstrate a classification accuracy of 78%, indicating the system's strong potential to enhance and streamline the organization of Bangla music genres.
>
---
#### [new 002] Test-Time Adaptation For Speech Enhancement Via Mask Polarization
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决模型在未见环境下的适应问题。提出mask polarization方法，在不增加参数的情况下提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14770v1](https://arxiv.org/pdf/2601.14770v1)**

> **作者:** Tobias Raichle; Erfan Amini; Bin Yang
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Adapting speech enhancement (SE) models to unseen environments is crucial for practical deployments, yet test-time adaptation (TTA) for SE remains largely under-explored due to a lack of understanding of how SE models degrade under domain shifts. We observe that mask-based SE models lose confidence under domain shifts, with predicted masks becoming flattened and losing decisive speech preservation and noise suppression. Based on this insight, we propose mask polarization (MPol), a lightweight TTA method that restores mask bimodality through distribution comparison using the Wasserstein distance. MPol requires no additional parameters beyond the trained model, making it suitable for resource-constrained edge deployments. Experimental results across diverse domain shifts and architectures demonstrate that MPol achieves very consistent gains that are competitive with significantly more complex approaches.
>
---
#### [new 003] Training-Efficient Text-to-Music Generation with State-Space Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于文本到音乐生成任务，旨在提升训练效率与数据利用率。通过引入状态空间模型替代Transformer，实现更高效、开放的模型训练。**

- **链接: [https://arxiv.org/pdf/2601.14786v1](https://arxiv.org/pdf/2601.14786v1)**

> **作者:** Wei-Jaw Lee; Fang-Chih Hsieh; Xuanjun Chen; Fang-Duo Tsai; Yi-Hsuan Yang
>
> **备注:** 9 pages, 3 figures. This is a preprint of a paper submitted to IEEE/ACM TASLP
>
> **摘要:** Recent advances in text-to-music generation (TTM) have yielded high-quality results, but often at the cost of extensive compute and the use of large proprietary internal data. To improve the affordability and openness of TTM training, an open-source generative model backbone that is more training- and data-efficient is needed. In this paper, we constrain the number of trainable parameters in the generative model to match that of the MusicGen-small benchmark (with about 300M parameters), and replace its Transformer backbone with the emerging class of state-space models (SSMs). Specifically, we explore different SSM variants for sequence modeling, and compare a single-stage SSM-based design with a decomposable two-stage SSM/diffusion hybrid design. All proposed models are trained from scratch on a purely public dataset comprising 457 hours of CC-licensed music, ensuring full openness. Our experimental findings are three-fold. First, we show that SSMs exhibit superior training efficiency compared to the Transformer counterpart. Second, despite using only 9% of the FLOPs and 2% of the training data size compared to the MusicGen-small benchmark, our model achieves competitive performance in both objective metrics and subjective listening tests based on MusicCaps captions. Finally, our scaling-down experiment demonstrates that SSMs can maintain competitive performance relative to the Transformer baseline even at the same training budget (measured in iterations), when the model size is reduced to four times smaller. To facilitate the democratization of TTM research, the processed captions, model checkpoints, and source code are available on GitHub via the project page: https://lonian6.github.io/ssmttm/.
>
---
#### [new 004] AQAScore: Evaluating Semantic Alignment in Text-to-Audio Generation via Audio Question Answering
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于文本到音频生成的评估任务，旨在解决现有指标在语义对齐和组合推理上的不足。提出AQAScore框架，通过音频感知大模型进行语义验证，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.14728v1](https://arxiv.org/pdf/2601.14728v1)**

> **作者:** Chun-Yi Kuan; Kai-Wei Chang; Hung-yi Lee
>
> **备注:** Manuscript in progress
>
> **摘要:** Although text-to-audio generation has made remarkable progress in realism and diversity, the development of evaluation metrics has not kept pace. Widely-adopted approaches, typically based on embedding similarity like CLAPScore, effectively measure general relevance but remain limited in fine-grained semantic alignment and compositional reasoning. To address this, we introduce AQAScore, a backbone-agnostic evaluation framework that leverages the reasoning capabilities of audio-aware large language models (ALLMs). AQAScore reformulates assessment as a probabilistic semantic verification task; rather than relying on open-ended text generation, it estimates alignment by computing the exact log-probability of a "Yes" answer to targeted semantic queries. We evaluate AQAScore across multiple benchmarks, including human-rated relevance, pairwise comparison, and compositional reasoning tasks. Experimental results show that AQAScore consistently achieves higher correlation with human judgments than similarity-based metrics and generative prompting baselines, showing its effectiveness in capturing subtle semantic inconsistencies and scaling with the capability of underlying ALLMs.
>
---
#### [new 005] Generative Artificial Intelligence, Musical Heritage and the Construction of Peace Narratives: A Case Study in Mali
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于人工智能与文化研究任务，探讨Gen AI在马里构建和平叙事和复兴音乐遗产中的应用，解决技术与文化平衡及社会凝聚力问题，通过实验分析AI在音乐创作中的作用。**

- **链接: [https://arxiv.org/pdf/2601.14931v1](https://arxiv.org/pdf/2601.14931v1)**

> **作者:** Nouhoum Coulibaly; Ousmane Ly; Michael Leventhal; Ousmane Goro
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** This study explores the capacity of generative artificial intelligence (Gen AI) to contribute to the construction of peace narratives and the revitalization of musical heritage in Mali. The study has been made in a political and social context where inter-community tensions and social fractures motivate a search for new symbolic frameworks for reconciliation. The study empirically explores three questions: (1) how Gen AI can be used as a tool for musical creation rooted in national languages and traditions; (2) to what extent Gen AI systems enable a balanced hybridization between technological innovation and cultural authenticity; and (3) how AI-assisted musical co-creation can strengthen social cohesion and cultural sovereignty. The experimental results suggest that Gen AI, embedded in a culturally conscious participatory framework, can act as a catalyst for symbolic diplomacy, amplifying local voices instead of standardizing them. However, challenges persist regarding the availability of linguistic corpora, algorithmic censorship, and the ethics of generating compositions derived from copyrighted sources.
>
---
#### [new 006] VCNAC: A Variable-Channel Neural Audio Codec for Mono, Stereo, and Surround Sound
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出VCNAC，一种支持单声道、立体声和环绕声的神经音频编解码器。解决多通道音频兼容性问题，通过统一编码解码结构实现高效重建与灵活扩展。**

- **链接: [https://arxiv.org/pdf/2601.14960v1](https://arxiv.org/pdf/2601.14960v1)**

> **作者:** Florian Grötschla; Arunasish Sen; Alessandro Lombardi; Guillermo Cámbara; Andreas Schwarz
>
> **备注:** Submitted to EUSIPCO 2026
>
> **摘要:** We present VCNAC, a variable channel neural audio codec. Our approach features a single encoder and decoder parametrization that enables native inference for different channel setups, from mono speech to cinematic 5.1 channel surround audio. Channel compatibility objectives ensure that multi-channel content maintains perceptual quality when decoded to fewer channels. The shared representation enables training of generative language models on a single set of codebooks while supporting inference-time scalability across modalities and channel configurations. Evaluation using objective spatial audio metrics and subjective listening tests demonstrates that our unified approach maintains high reconstruction quality across mono, stereo, and surround audio configurations.
>
---
#### [new 007] Triage knowledge distillation for speaker verification
- **分类: eess.AS**

- **简介: 该论文属于说话人验证任务，解决资源受限设备上高容量模型部署难题。通过引入Triage KD方法，有效提升知识蒸馏效果，降低错误率。**

- **链接: [https://arxiv.org/pdf/2601.14699v1](https://arxiv.org/pdf/2601.14699v1)**

> **作者:** Ju-ho Kim; Youngmoon Jung; Joon-Young Yang; Jaeyoung Roh; Chang Woo Han; Hoon-Young Cho
>
> **备注:** 5 pages, 2 figures, Accepted at ICASSP 2026
>
> **摘要:** Deploying speaker verification on resource-constrained devices remains challenging due to the computational cost of high-capacity models; knowledge distillation (KD) offers a remedy. Classical KD entangles target confidence with non-target structure in a Kullback-Leibler term, limiting the transfer of relational information. Decoupled KD separates these signals into target and non-target terms, yet treats non-targets uniformly and remains vulnerable to the long tail of low-probability classes in large-class settings. We introduce Triage KD (TRKD), a distillation scheme that operationalizes assess-prioritize-focus. TRKD introduces a cumulative-probability cutoff $τ$ to assess per-example difficulty and partition the teacher posterior into three groups: the target class, a high-probability non-target confusion-set, and a background-set. To prioritize informative signals, TRKD distills the confusion-set conditional distribution and discards the background. Concurrently, it transfers a three-mass (target/confusion/background) that capture sample difficulty and inter-class confusion. Finally, TRKD focuses learning via a curriculum on $τ$: training begins with a larger $τ$ to convey broad non-target context, then $τ$ is progressively decreased to shrink the confusion-set, concentrating supervision on the most confusable classes. In extensive experiments on VoxCeleb1 with both homogeneous and heterogeneous teacher-student pairs, TRKD was consistently superior to recent KD variants and attained the lowest EER across all protocols.
>
---
#### [new 008] Towards noise-robust speech inversion through multi-task learning with speech enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音逆向任务，旨在提升噪声环境下语音逆向的鲁棒性。通过多任务学习融合语音增强与语音逆向，利用自监督学习表示提高性能。**

- **链接: [https://arxiv.org/pdf/2601.14516v1](https://arxiv.org/pdf/2601.14516v1)**

> **作者:** Saba Tabatabaee; Carol Espy-Wilson
>
> **备注:** Accepted for presentation at ICASSP 2026
>
> **摘要:** Recent studies demonstrate the effectiveness of Self Supervised Learning (SSL) speech representations for Speech Inversion (SI). However, applying SI in real-world scenarios remains challenging due to the pervasive presence of background noise. We propose a unified framework that integrates Speech Enhancement (SE) and SI models through shared SSL-based speech representations. In this framework, the SSL model is trained not only to support the SE module in suppressing noise but also to produce representations that are more informative for the SI task, allowing both modules to benefit from joint training. At a Signal-to-Noise Ratio of -5 db, our method for the SI task achieves relative improvements over the baseline of 80.95% under babble noise and 38.98% under non-babble noise, as measured by the average Pearson product-moment correlation across all estimated parameters.
>
---
#### [new 009] Inverse-Hessian Regularization for Continual Learning in ASR
- **分类: eess.AS**

- **简介: 该论文属于持续学习任务，解决ASR中的灾难性遗忘问题。提出Inverse Hessian Regularization方法，在不存储数据的情况下提升模型适应新任务的能力。**

- **链接: [https://arxiv.org/pdf/2601.14751v1](https://arxiv.org/pdf/2601.14751v1)**

> **作者:** Steven Vander Eeckt; Hugo Van hamme
>
> **备注:** Accepted for presentation at ICASSP 2026
>
> **摘要:** Catastrophic forgetting remains a major challenge for continual learning (CL) in automatic speech recognition (ASR), where models must adapt to new domains without losing performance on previously learned conditions. Several CL methods have been proposed for ASR, and, recently, weight averaging - where models are averaged in a merging step after fine-tuning - has proven effective as a simple memory-free strategy. However, it is heuristic in nature and ignores the underlying loss landscapes of the tasks, hindering adaptability. In this work, we propose Inverse Hessian Regularization (IHR), a memory-free approach for CL in ASR that incorporates curvature information into the merging step. After fine-tuning on a new task, the adaptation is adjusted through a Kronecker-factored inverse Hessian approximation of the previous task, ensuring that the model moves primarily in directions less harmful to past performance, while keeping the method lightweight. We evaluate IHR on two CL benchmarks and show that it significantly outperforms state-of-the-art baselines, reducing forgetting while improving adaptability. Ablation studies and analyses further confirm its effectiveness.
>
---
#### [new 010] Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于情感识别任务，旨在解决标注稀缺与情感模糊性问题。通过生成合成标注增强人类标注，提升情感分布可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14620v1](https://arxiv.org/pdf/2601.14620v1)**

> **作者:** Wenda Zhang; Hongyu Jin; Siyi Wang; Zhiqiang Wei; Ting Dang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Speech Emotion Recognition models typically use single categorical labels, overlooking the inherent ambiguity of human emotions. Ambiguous Emotion Recognition addresses this by representing emotions as probability distributions, but progress is limited by unreliable ground-truth distributions inferred from sparse human annotations. This paper explores whether Large Audio-Language Models (ALMs) can mitigate the annotation bottleneck by generating high-quality synthetic annotations. We introduce a framework leveraging ALMs to create Synthetic Perceptual Proxies, augmenting human annotations to improve ground-truth distribution reliability. We validate these proxies through statistical analysis of their alignment with human distributions and evaluate their impact by fine-tuning ALMs with the augmented emotion distributions. Furthermore, to address class imbalance and enable unbiased evaluation, we propose DiME-Aug, a Distribution-aware Multimodal Emotion Augmentation strategy. Experiments on IEMOCAP and MSP-Podcast show that synthetic annotations enhance emotion distribution, especially in low-ambiguity regions where annotation agreement is high. However, benefits diminish for highly ambiguous emotions with greater human disagreement. This work provides the first evidence that ALMs could address annotation scarcity in ambiguous emotion recognition, but highlights the need for more advanced prompting or generation strategies to handle highly ambiguous cases.
>
---
#### [new 011] NLP-Based Review for Toxic Comment Detection Tailored to the Chinese Cyberspace
- **分类: eess.AS**

- **简介: 该论文属于中文网络毒害评论检测任务，旨在解决中文网络语言复杂、文化特异性强导致的传统检测方法失效的问题。论文提出新的分类框架和数据策略，并分析模型演化与挑战。**

- **链接: [https://arxiv.org/pdf/2601.14721v1](https://arxiv.org/pdf/2601.14721v1)**

> **作者:** Ruixing Ren; Junhui Zhao; Xiaoke Sun; Qiuping Li
>
> **备注:** 20 pages, 6 figures. This review focuses on toxic comment detection in Chinese cyberspace
>
> **摘要:** With the in-depth integration of mobile Internet and widespread adoption of social platforms, user-generated content in the Chinese cyberspace has witnessed explosive growth. Among this content, the proliferation of toxic comments poses severe challenges to individual mental health, community atmosphere and social trust. Owing to the strong context dependence, cultural specificity and rapid evolution of Chinese cyber language, toxic expressions are often conveyed through complex forms such as homophones and metaphors, imposing notable limitations on traditional detection methods. To address this issue, this review focuses on the core topic of natural language processing based toxic comment detection in the Chinese cyberspace, systematically collating and critically analyzing the research progress and key challenges in this field. This review first defines the connotation and characteristics of Chinese toxic comments, and analyzes the platform ecology and transmission mechanisms they rely on. It then comprehensively reviews the construction methods and limitations of existing public datasets, and proposes a novel fine-grained and scalable framework for toxic comment definition and classification, along with corresponding data annotation and quality assessment strategies. We systematically summarize the evolutionary path of detection models from traditional methods to deep learning, with special emphasis on the importance of interpretability in model design. Finally, we thoroughly discuss the open challenges faced by current research and provide forward-looking suggestions for future research directions.
>
---
#### [new 012] Single-step Controllable Music Bandwidth Extension With Flow Matching
- **分类: cs.SD**

- **简介: 该论文属于音频修复任务，旨在提升音乐录音的音质。针对现有模型控制性不足的问题，提出DSC控制信号，实现更精细的带宽扩展。**

- **链接: [https://arxiv.org/pdf/2601.14356v1](https://arxiv.org/pdf/2601.14356v1)**

> **作者:** Carlos Hernandez-Olivan; Hendrik Vincent Koops; Hao Hao Tan; Elio Quinton
>
> **备注:** Accepted at the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026
>
> **摘要:** Audio restoration consists in inverting degradations of a digital audio signal to recover what would have been the pristine quality signal before the degradation occurred. This is valuable in contexts such as archives of music recordings, particularly those of precious historical value, for which a clean version may have been lost or simply does not exist. Recent work applied generative models to audio restoration, showing promising improvement over previous methods, and opening the door to the ability to perform restoration operations that were not possible before. However, making these models finely controllable remains a challenge. In this paper, we propose an extension of FLowHigh and introduce the Dynamic Spectral Contour (DSC) as a control signal for bandwidth extension via classifier-free guidance. Our experiments show competitive model performance, and indicate that DSC is a promising feature to support fine-grained conditioning.
>
---
#### [new 013] WavLink: Compact Audio--Text Embeddings with a Global Whisper Token
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出WavLink，一种结合Whisper和可学习全局token的音频-文本嵌入模型，解决音频特征表示效率低的问题。通过优化训练策略，实现更小的嵌入尺寸且性能损失小。**

- **链接: [https://arxiv.org/pdf/2601.15118v1](https://arxiv.org/pdf/2601.15118v1)**

> **作者:** Gokul Karthik Kumar; Ludovick Lepauloux; Hakim Hacid
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Whisper has become the de-facto encoder for extracting general-purpose audio features in large audio-language models, where a 30-second clip is typically represented by 1500 frame features projected into an LLM. In contrast, audio-text embedding models like CLAP-based models have largely relied on alternative audio encoders (e.g., HTS-AT, PaSST), and have not leveraged Whisper effectively. We present WavLink, a compact audio-text embedding model that augments Whisper encoder with a learnable global token, trained jointly with a text encoder. Through a systematic study of design choices, including pretrained text encoders, loss functions, training modes, and data mixtures, we identify configurations that yield state-of-the-art retrieval performance. Our two-stage training recipe across three model sizes, combined with Matryoshka-style supervision, improves scalability, enabling 8x smaller embeddings with minimal performance drop. WavLink also demonstrates competitive performance on AIR-Bench with MCQs and zero-shot classification.
>
---
#### [new 014] Fast-ULCNet: A fast and ultra low complexity network for single-channel speech enhancement
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音增强任务，旨在解决嵌入式设备中低延迟和低复杂度的需求。通过改进ULCNet，提出Fast-ULCNet，降低模型大小和延迟，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2601.14925v1](https://arxiv.org/pdf/2601.14925v1)**

> **作者:** Nicolás Arrieta Larraza; Niels de Koeijer
>
> **备注:** ©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Single-channel speech enhancement algorithms are often used in resource-constrained embedded devices, where low latency and low complexity designs gain more importance. In recent years, researchers have proposed a wide variety of novel solutions to this problem. In particular, a recent deep learning model named ULCNet is among the state-of-the-art approaches in this domain. This paper proposes an adaptation of ULCNet, by replacing its GRU layers with FastGRNNs, to reduce both computational latency and complexity. Furthermore, this paper shows empirical evidence on the performance decay of FastGRNNs in long audio signals during inference due to internal state drifting, and proposes a novel approach based on a trainable complementary filter to mitigate it. The resulting model, Fast-ULCNet, performs on par with the state-of-the-art original ULCNet architecture on a speech enhancement task, while reducing its model size by more than half and decreasing its latency by 34% on average.
>
---
#### [new 015] Multi-Tast Transformer for Explainable Speech Deepfake Detection via Formant Modeling
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在区分真实与虚假语音，并提供可解释性。通过多任务Transformer模型，预测音高轨迹和发音模式，提升检测的可解释性与效率。**

- **链接: [https://arxiv.org/pdf/2601.14850v1](https://arxiv.org/pdf/2601.14850v1)**

> **作者:** Viola Negroni; Luca Cuccovillo; Paolo Bestagini; Patrick Aichroth; Stefano Tubaro
>
> **备注:** Accepted @ IEEE ICASSP 2026
>
> **摘要:** In this work, we introduce a multi-task transformer for speech deepfake detection, capable of predicting formant trajectories and voicing patterns over time, ultimately classifying speech as real or fake, and highlighting whether its decisions rely more on voiced or unvoiced regions. Building on a prior speaker-formant transformer architecture, we streamline the model with an improved input segmentation strategy, redesign the decoding process, and integrate built-in explainability. Compared to the baseline, our model requires fewer parameters, trains faster, and provides better interpretability, without sacrificing prediction performance.
>
---
#### [new 016] Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音合成任务，旨在解决神经声码器的韵律建模不足和相位重建不准确问题。通过引入韵律引导的谐波注意力机制和直接预测复频谱，提升语音自然度与音高准确性。**

- **链接: [https://arxiv.org/pdf/2601.14472v1](https://arxiv.org/pdf/2601.14472v1)**

> **作者:** Mohammed Salah Al-Radhi; Riad Larbi; Mátyás Bartalis; Géza Németh
>
> **备注:** 5 pages, 2 figures, 1 table. Accepted for presentation at ICASSP 2026
>
> **摘要:** Neural vocoders are central to speech synthesis; despite their success, most still suffer from limited prosody modeling and inaccurate phase reconstruction. We propose a vocoder that introduces prosody-guided harmonic attention to enhance voiced segment encoding and directly predicts complex spectral components for waveform synthesis via inverse STFT. Unlike mel-spectrogram-based approaches, our design jointly models magnitude and phase, ensuring phase coherence and improved pitch fidelity. To further align with perceptual quality, we adopt a multi-objective training strategy that integrates adversarial, spectral, and phase-aware losses. Experiments on benchmark datasets demonstrate consistent gains over HiFi-GAN and AutoVocoder: F0 RMSE reduced by 22 percent, voiced/unvoiced error lowered by 18 percent, and MOS scores improved by 0.15. These results show that prosody-guided attention combined with direct complex spectrum modeling yields more natural, pitch-accurate, and robust synthetic speech, setting a strong foundation for expressive neural vocoding.
>
---
#### [new 017] Dissecting Performance Degradation in Audio Source Separation under Sampling Frequency Mismatch
- **分类: cs.SD**

- **简介: 该论文属于音频源分离任务，解决采样频率不匹配导致的性能下降问题。通过改进重采样方法，如噪声核和可训练核，提升模型在不同采样频率下的表现。**

- **链接: [https://arxiv.org/pdf/2601.14684v1](https://arxiv.org/pdf/2601.14684v1)**

> **作者:** Kanami Imamura; Tomohiko Nakamura; Kohei Yatabe; Hiroshi Saruwatari
>
> **备注:** Accepted for ICASSP 2026
>
> **摘要:** Audio processing methods based on deep neural networks are typically trained at a single sampling frequency (SF). To handle untrained SFs, signal resampling is commonly employed, but it can degrade performance, particularly when the input SF is lower than the trained SF. This paper investigates the causes of this degradation through two hypotheses: (i) the lack of high-frequency components introduced by up-sampling, and (ii) the greater importance of their presence than their precise representation. To examine these hypotheses, we compare conventional resampling with three alternatives: post-resampling noise addition, which adds Gaussian noise to the resampled signal; noisy-kernel resampling, which perturbs the kernel with Gaussian noise to enrich high-frequency components; and trainable-kernel resampling, which adapts the interpolation kernel through training. Experiments on music source separation show that noisy-kernel and trainable-kernel resampling alleviate the degradation observed with conventional resampling. We further demonstrate that noisy-kernel resampling is effective across diverse models, highlighting it as a simple yet practical option.
>
---
#### [new 018] Synthetic Singers: A Review of Deep-Learning-based Singing Voice Synthesis Approaches
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音合成任务，旨在系统梳理深度学习驱动的歌唱语音合成方法，解决现有研究缺乏全面综述的问题。论文分类分析系统架构，探讨核心技术，并总结数据与评估工具。**

- **链接: [https://arxiv.org/pdf/2601.13910v1](https://arxiv.org/pdf/2601.13910v1)**

> **作者:** Changhao Pan; Dongyu Yao; Yu Zhang; Wenxiang Guo; Jingyu Lu; Zhiyuan Zhu; Zhou Zhao
>
> **备注:** Accepetd by IJCNLP-AACL 2025(Oral)
>
> **摘要:** Recent advances in singing voice synthesis (SVS) have attracted substantial attention from both academia and industry. With the advent of large language models and novel generative paradigms, producing controllable, high-fidelity singing voices has become an attainable goal. Yet the field still lacks a comprehensive survey that systematically analyzes deep-learning-based singing voice synthesis systems and their enabling technologies. To address the aforementioned issue, this survey first categorizes existing systems by task type and then organizes current architectures into two major paradigms: cascaded and end-to-end approaches. Moreover, we provide an in-depth analysis of core technologies, covering singing modeling and control techniques. Finally, we review relevant datasets, annotation tools, and evaluation benchmarks that support training and assessment. In appendix, we introduce training strategies and further discussion of SVS. This survey provides an up-to-date review of the literature on SVS models, which would be a useful reference for both researchers and engineers. Related materials are available at https://github.com/David-Pigeon/SyntheticSingers.
>
---
#### [new 019] WeDefense: A Toolkit to Defend Against Fake Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于虚假音频检测任务，旨在解决虚假音频识别与定位问题。提出WeDefense工具包，提供统一的检测与分析框架。**

- **链接: [https://arxiv.org/pdf/2601.15240v1](https://arxiv.org/pdf/2601.15240v1)**

> **作者:** Lin Zhang; Johan Rohdin; Xin Wang; Junyi Peng; Tianchi Liu; You Zhang; Hieu-Thi Luong; Shuai Wang; Chengdong Liang; Anna Silnova; Nicholas Evans
>
> **备注:** This is an ongoing work. v1 corresponds to the version completed by June 4, 2025 and previously submitted to ASRU 2025
>
> **摘要:** The advances in generative AI have enabled the creation of synthetic audio which is perceptually indistinguishable from real, genuine audio. Although this stellar progress enables many positive applications, it also raises risks of misuse, such as for impersonation, disinformation and fraud. Despite a growing number of open-source fake audio detection codes released through numerous challenges and initiatives, most are tailored to specific competitions, datasets or models. A standardized and unified toolkit that supports the fair benchmarking and comparison of competing solutions with not just common databases, protocols, metrics, but also a shared codebase, is missing. To address this, we propose WeDefense, the first open-source toolkit to support both fake audio detection and localization. Beyond model training, WeDefense emphasizes critical yet often overlooked components: flexible input and augmentation, calibration, score fusion, standardized evaluation metrics, and analysis tools for deeper understanding and interpretation. The toolkit is publicly available at https://github.com/zlin0/wedefense with interactive demos for fake audio detection and localization.
>
---
#### [new 020] Unlocking Large Audio-Language Models for Interactive Language Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语言学习任务，旨在解决L2发音训练中反馈不直观的问题。通过构建数据集并优化音频语言模型，提升发音纠错与改进建议的效果。**

- **链接: [https://arxiv.org/pdf/2601.14744v1](https://arxiv.org/pdf/2601.14744v1)**

> **作者:** Hongfu Liu; Zhouying Cui; Xiangming Gu; Ye Wang
>
> **备注:** Accepted to the Findings of EACL 2026
>
> **摘要:** Achieving pronunciation proficiency in a second language (L2) remains a challenge, despite the development of Computer-Assisted Pronunciation Training (CAPT) systems. Traditional CAPT systems often provide unintuitive feedback that lacks actionable guidance, limiting its effectiveness. Recent advancements in audio-language models (ALMs) offer the potential to enhance these systems by providing more user-friendly feedback. In this work, we investigate ALMs for chat-based pronunciation training by introducing L2-Arctic-plus, an English dataset with detailed error explanations and actionable suggestions for improvement. We benchmark cascaded ASR+LLMs and existing ALMs on this dataset, specifically in detecting mispronunciation and generating actionable feedback. To improve the performance, we further propose to instruction-tune ALMs on L2-Arctic-plus. Experimental results demonstrate that our instruction-tuned models significantly outperform existing baselines on mispronunciation detection and suggestion generation in terms of both objective and human evaluation, highlighting the value of the proposed dataset.
>
---
#### [new 021] A Cloud-Based Cross-Modal Transformer for Emotion Recognition and Adaptive Human-Computer Interaction
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于情感识别任务，旨在解决单模态系统在真实环境中的鲁棒性不足问题。提出一种基于云的跨模态Transformer框架，融合多模态数据并提升情感识别效果与响应速度。**

- **链接: [https://arxiv.org/pdf/2601.14259v1](https://arxiv.org/pdf/2601.14259v1)**

> **作者:** Ziwen Zhong; Zhitao Shu; Yue Zhao
>
> **摘要:** Emotion recognition is a fundamental component of next-generation human-computer interaction (HCI), enabling machines to perceive, understand, and respond to users' affective states. However, existing systems often rely on single-modality analysis such as facial expressions, speech tone, or textual sentiment, resulting in limited robustness and poor generalization in real-world environments. To address these challenges, this study proposes a Cloud-Based Cross-Modal Transformer (CMT) framework for multimodal emotion recognition and adaptive human-computer interaction. The proposed model integrates visual, auditory, and textual signals using pretrained encoders (Vision Transformer, Wav2Vec2, and BERT) and employs a cross-modal attention mechanism to capture complex interdependencies among heterogeneous features. By leveraging cloud computing infrastructure with distributed training on Kubernetes and TensorFlow Serving, the system enables scalable, low-latency emotion recognition for large-scale user interactions. Experiments conducted on benchmark datasets including IEMOCAP, MELD, and AffectNet demonstrate that the CMT achieves state-of-the-art performance, improving the F1-score by 3.0 percent and reducing cross-entropy loss by 12.9 percent compared to strong multimodal baselines. Additionally, cloud deployment evaluations show an average response latency of 128 ms, representing a 35 percent reduction compared with conventional transformer-based fusion systems. These results confirm that the proposed framework enables efficient, real-time emotion recognition and adaptive feedback in applications such as intelligent customer service, virtual tutoring systems, and affective computing interfaces, marking an important step toward cloud-native affective computing and emotionally intelligent interactive systems.
>
---
#### [new 022] Neural Tracking of Sustained Attention, Attention Switching, and Natural Conversation in Audiovisual Environments using Mobile EEG
- **分类: eess.SP; cs.SD; eess.AS**

- **简介: 该论文属于神经注意力追踪任务，旨在解决动态多感官环境中注意力跟踪的问题。通过移动EEG记录，研究了持续注意、注意力切换和自然对话中的注意力变化。**

- **链接: [https://arxiv.org/pdf/2601.15097v1](https://arxiv.org/pdf/2601.15097v1)**

> **作者:** Johanna Wilroth; Oskar Keding; Martin A. Skoglund; Maria Sandsten; Martin Enqvist; Emina Alickovic
>
> **备注:** Submitted to European Journal of Neuroscience
>
> **摘要:** Everyday communication is dynamic and multisensory, often involving shifting attention, overlapping speech and visual cues. Yet, most neural attention tracking studies are still limited to highly controlled lab settings, using clean, often audio-only stimuli and requiring sustained attention to a single talker. This work addresses that gap by introducing a novel dataset from 24 normal-hearing participants. We used a mobile electroencephalography (EEG) system (44 scalp electrodes and 20 cEEGrid electrodes) in an audiovisual (AV) paradigm with three conditions: sustained attention to a single talker in a two-talker environment, attention switching between two talkers, and unscripted two-talker conversations with a competing single talker. Analysis included temporal response functions (TRFs) modeling, optimal lag analysis, selective attention classification with decision windows ranging from 1.1s to 35s, and comparisons of TRFs for attention to AV conversations versus side audio-only talkers. Key findings show significant differences in the attention-related P2-peak between attended and ignored speech across conditions for scalp EEG. No significant change in performance between switching and sustained attention suggests robustness for attention switches. Optimal lag analysis revealed narrower peak for conversation compared to single-talker AV stimuli, reflecting the additional complexity of multi-talker processing. Classification of selective attention was consistently above chance (55-70% accuracy) for scalp EEG, while cEEGrid data yielded lower correlations, highlighting the need for further methodological improvements. These results demonstrate that mobile EEG can reliably track selective attention in dynamic, multisensory listening scenarios and provide guidance for designing future AV paradigms and real-world attention tracking applications.
>
---
#### [new 023] Call2Instruct: Automated Pipeline for Generating Q&A Datasets from Call Center Recordings for LLM Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决从电话录音生成Q&A数据集的问题。通过自动化流程处理音频和文本，提取语义并生成适用于LLM微调的指令数据。**

- **链接: [https://arxiv.org/pdf/2601.14263v1](https://arxiv.org/pdf/2601.14263v1)**

> **作者:** Alex Echeverria; Sávio Salvarino Teles de Oliveira; Fernando Marques Federson
>
> **备注:** 15 pages, 1 figures, conference
>
> **摘要:** The adaptation of Large-Scale Language Models (LLMs) to specific domains depends on high-quality fine-tuning datasets, particularly in instructional format (e.g., Question-Answer - Q&A). However, generating these datasets, particularly from unstructured sources such as call center audio recordings, poses a significant challenge due to the noisy and disorganized nature of the data. This paper presents a solution to this challenge by offering an end-to-end automated pipeline for generating Q&A instructional datasets from such recordings. The methodology developed comprises sequential steps of audio processing (including diarization, noise removal and automatic transcription), textual processing (cleaning, normalization, and anonymization), semantic extraction of customer demands and attendant responses using vector embeddings, and matching via semantic search to form the final Q&A pairs. As a result, the complete pipeline was successfully implemented, generating a dataset specifically formatted for Instruct Fine Tuning. The practical value and feasibility of the generated dataset were substantiated and functionally demonstrated through the successful fine-tuning of an LLM model (based on Llama 2 7B). The conclusion of the paper states that the proposed approach is viable for converting unstructured conversational data from call centers into valuable resources for training LLMs. This development has the potential to open up avenues for creating more effective AI systems for Q&A tasks in the customer service domain. The developed codes have been made publicly available to promote reproducibility and future research.
>
---
#### [new 024] READ-Net: Clarifying Emotional Ambiguity via Adaptive Feature Recalibration for Audio-Visual Depression Detection
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉抑郁检测任务，旨在解决情感模糊性问题。通过自适应特征重新校准（AFR）方法，提升抑郁相关信号的识别准确性。**

- **链接: [https://arxiv.org/pdf/2601.14651v1](https://arxiv.org/pdf/2601.14651v1)**

> **作者:** Chenglizhao Chen; Boze Li; Mengke Song; Dehao Feng; Xinyu Liu; Shanchen Pang; Jufeng Yang; Hui Yu
>
> **备注:** 12 pages
>
> **摘要:** Depression is a severe global mental health issue that impairs daily functioning and overall quality of life. Although recent audio-visual approaches have improved automatic depression detection, methods that ignore emotional cues often fail to capture subtle depressive signals hidden within emotional expressions. Conversely, those incorporating emotions frequently confuse transient emotional expressions with stable depressive symptoms in feature representations, a phenomenon termed \emph{Emotional Ambiguity}, thereby leading to detection errors. To address this critical issue, we propose READ-Net, the first audio-visual depression detection framework explicitly designed to resolve Emotional Ambiguity through Adaptive Feature Recalibration (AFR). The core insight of AFR is to dynamically adjust the weights of emotional features to enhance depression-related signals. Rather than merely overlooking or naively combining emotional information, READ-Net innovatively identifies and preserves depressive-relevant cues within emotional features, while adaptively filtering out irrelevant emotional noise. This recalibration strategy significantly clarifies feature representations, and effectively mitigates the persistent challenge of emotional interference. Additionally, READ-Net can be easily integrated into existing frameworks for improved performance. Extensive evaluations on three publicly available datasets show that READ-Net outperforms state-of-the-art methods, with average gains of 4.55\% in accuracy and 1.26\% in F1-score, demonstrating its robustness to emotional disturbances and improving audio-visual depression detection.
>
---
#### [new 025] Guided by the Plan: Enhancing Faithful Autoregressive Text-to-Audio Generation with Guided Decoding
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决AR模型难以忠实响应复杂文本提示的问题。通过引入Plan-Critic模型，提升生成质量并保持计算效率。**

- **链接: [https://arxiv.org/pdf/2601.14304v1](https://arxiv.org/pdf/2601.14304v1)**

> **作者:** Juncheng Wang; Zhe Hu; Chao Xu; Siyue Ren; Yuxiang Feng; Yang Liu; Baigui Sun; Shujun Wang
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Autoregressive (AR) models excel at generating temporally coherent audio by producing tokens sequentially, yet they often falter in faithfully following complex textual prompts, especially those describing complex sound events. We uncover a surprising capability in AR audio generators: their early prefix tokens implicitly encode global semantic attributes of the final output, such as event count and sound-object category, revealing a form of implicit planning. Building on this insight, we propose Plan-Critic, a lightweight auxiliary model trained with a Generalized Advantage Estimation (GAE)-inspired objective to predict final instruction-following quality from partial generations. At inference time, Plan-Critic enables guided exploration: it evaluates candidate prefixes early, prunes low-fidelity trajectories, and reallocates computation to high-potential planning seeds. Our Plan-Critic-guided sampling achieves up to a 10-point improvement in CLAP score over the AR baseline-establishing a new state of the art in AR text-to-audio generation-while maintaining computational parity with standard best-of-N decoding. This work bridges the gap between causal generation and global semantic alignment, demonstrating that even strictly autoregressive models can plan ahead.
>
---
## 更新

#### [replaced 001] Exploring Fine-Tuning of Large Audio Language Models for Spoken Language Understanding under Limited Speech Data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究在有限语音数据下对大音频语言模型进行微调，解决语音理解任务中的数据不足问题，通过不同微调方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.15389v2](https://arxiv.org/pdf/2509.15389v2)**

> **作者:** Youngwon Choi; Jaeyoon Jung; Hyeonyu Kim; Huu-Kim Nguyen; Hwayeon Kim
>
> **备注:** 4 pages (excluding references), 2 figures, ICASSP 2026 (Accepted)
>
> **摘要:** Large Audio Language Models (LALMs) have emerged as powerful tools for speech-related tasks but remain underexplored for fine-tuning, especially with limited speech data. To bridge this gap, we systematically examine how different fine-tuning schemes including text-only, direct mixing, and curriculum learning affect spoken language understanding (SLU), focusing on scenarios where text-label pairs are abundant while paired speech-label data are limited. Results show that LALMs already achieve competitive performance with text-only fine-tuning, highlighting their strong generalization ability. Adding even small amounts of speech data (2-5%) yields substantial further gains, with curriculum learning particularly effective under scarce data. In cross-lingual SLU, combining source-language speech data with target-language text and minimal target-language speech data enables effective adaptation. Overall, this study provides practical insights into the LALM fine-tuning under realistic data constraints.
>
---
#### [replaced 002] Performance and Complexity Trade-off Optimization of Speech Models During Training
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音机器学习任务，解决模型性能与计算复杂度的平衡问题。通过引入特征噪声重参数化方法，在训练中联合优化两者，无需后期剪枝。**

- **链接: [https://arxiv.org/pdf/2601.13704v2](https://arxiv.org/pdf/2601.13704v2)**

> **作者:** Esteban Gómez; Tom Bäckström
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In speech machine learning, neural network models are typically designed by choosing an architecture with fixed layer sizes and structure. These models are then trained to maximize performance on metrics aligned with the task's objective. While the overall architecture is usually guided by prior knowledge of the task, the sizes of individual layers are often chosen heuristically. However, this approach does not guarantee an optimal trade-off between performance and computational complexity; consequently, post hoc methods such as weight quantization or model pruning are typically employed to reduce computational cost. This occurs because stochastic gradient descent (SGD) methods can only optimize differentiable functions, while factors influencing computational complexity, such as layer sizes and floating-point operations per second (FLOP/s), are non-differentiable and require modifying the model structure during training. We propose a reparameterization technique based on feature noise injection that enables joint optimization of performance and computational complexity during training using SGD-based methods. Unlike traditional pruning methods, our approach allows the model size to be dynamically optimized for a target performance-complexity trade-off, without relying on heuristic criteria to select which weights or structures to remove. We demonstrate the effectiveness of our method through three case studies, including a synthetic example and two practical real-world applications: voice activity detection and audio anti-spoofing. The code related to our work is publicly available to encourage further research.
>
---
#### [replaced 003] Sound2Hap: Learning Audio-to-Vibrotactile Haptic Generation from Human Ratings
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于音频到触觉生成任务，旨在解决现有方法对多样化声音泛化能力差的问题。通过用户研究和深度学习模型Sound2Hap，提升音频驱动触觉的感知一致性与效果。**

- **链接: [https://arxiv.org/pdf/2601.12245v2](https://arxiv.org/pdf/2601.12245v2)**

> **作者:** Yinan Li; Hasti Seifi
>
> **摘要:** Environmental sounds like footsteps, keyboard typing, or dog barking carry rich information and emotional context, making them valuable for designing haptics in user applications. Existing audio-to-vibration methods, however, rely on signal-processing rules tuned for music or games and often fail to generalize across diverse sounds. To address this, we first investigated user perception of four existing audio-to-haptic algorithms, then created a data-driven model for environmental sounds. In Study 1, 34 participants rated vibrations generated by the four algorithms for 1,000 sounds, revealing no consistent algorithm preferences. Using this dataset, we trained Sound2Hap, a CNN-based autoencoder, to generate perceptually meaningful vibrations from diverse sounds with low latency. In Study 2, 15 participants rated its output higher than signal-processing baselines on both audio-vibration match and Haptic Experience Index (HXI), finding it more harmonious with diverse sounds. This work demonstrates a perceptually validated approach to audio-haptic translation, broadening the reach of sound-driven haptics.
>
---
#### [replaced 004] Exploring Resolution-Wise Shared Attention in Hybrid Mamba-U-Nets for Improved Cross-Corpus Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升跨语料库的泛化性能。提出RWSA-MambaUNet模型，结合Mamba和注意力机制，有效减少参数和计算量，同时提高性能。**

- **链接: [https://arxiv.org/pdf/2510.01958v2](https://arxiv.org/pdf/2510.01958v2)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** Recent advances in speech enhancement have shown that models combining Mamba and attention mechanisms yield superior cross-corpus generalization performance. At the same time, integrating Mamba in a U-Net structure has yielded state-of-the-art enhancement performance, while reducing both model size and computational complexity. Inspired by these insights, we propose RWSA-MambaUNet, a novel and efficient hybrid model combining Mamba and multi-head attention in a U-Net structure for improved cross-corpus performance. Resolution-wise shared attention (RWSA) refers to layerwise attention-sharing across corresponding time- and frequency resolutions. Our best-performing RWSA-MambaUNet model achieves state-of-the-art generalization performance on two out-of-domain test sets. Notably, our smallest model surpasses all baselines on the out-of-domain DNS 2020 test set in terms of PESQ, SSNR, and ESTOI, and on the out-of-domain EARS-WHAM_v2 test set in terms of SSNR, ESTOI, and SI-SDR, while using less than half the model parameters and a fraction of the FLOPs.
>
---
#### [replaced 005] MambAttention: Mamba with Multi-Head Attention for Generalizable Single-Channel Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于单通道语音增强任务，旨在解决序列模型泛化能力不足的问题。提出MambAttention架构，结合Mamba与多头注意力机制，提升模型在不同数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2507.00966v4](https://arxiv.org/pdf/2507.00966v4)**

> **作者:** Nikolai Lund Kühne; Jesper Jensen; Jan Østergaard; Zheng-Hua Tan
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech, and Language Processing
>
> **摘要:** With new sequence models like Mamba and xLSTM, several studies have shown that these models match or outperform the state-of-the-art in single-channel speech enhancement and audio representation learning. However, prior research has demonstrated that sequence models like LSTM and Mamba tend to overfit to the training set. To address this, previous works have shown that adding self-attention to LSTMs substantially improves generalization performance for single-channel speech enhancement. Nevertheless, neither the concept of hybrid Mamba and time-frequency attention models nor their generalization performance have been explored for speech enhancement. In this paper, we propose a novel hybrid architecture, MambAttention, which combines Mamba and shared time- and frequency-multi-head attention modules for generalizable single-channel speech enhancement. To train our model, we introduce VB-DemandEx, a dataset inspired by VoiceBank+Demand but with more challenging noise types and lower signal-to-noise ratios. Trained on VB-DemandEx, MambAttention significantly outperforms existing state-of-the-art discriminative LSTM-, xLSTM-, Mamba-, and Conformer-based systems of similar complexity across all reported metrics on two out-of-domain datasets: DNS 2020 without reverberation and EARS-WHAM_v2. MambAttention also matches or outperforms generative diffusion models in generalization performance while being competitive with language model baselines. Ablation studies highlight the importance of weight sharing between time- and frequency-multi-head attention modules for generalization performance. Finally, we explore integrating the shared time- and frequency-multi-head attention modules with LSTM and xLSTM, which yields a notable performance improvement on the out-of-domain datasets. Yet, MambAttention remains superior for cross-corpus generalization across all reported evaluation metrics.
>
---
#### [replaced 006] Mitigating Data Imbalance in Automated Speaking Assessment
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语言评估任务，解决ASA中的数据不平衡问题。通过引入BLV损失函数，提升模型对少数类的识别能力，增强评估公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2509.03010v2](https://arxiv.org/pdf/2509.03010v2)**

> **作者:** Fong-Chun Tsai; Kuan-Tang Huang; Bi-Cheng Yan; Tien-Hong Lo; Berlin Chen
>
> **备注:** Accepted by APSIPA 2025; revised figure, references added
>
> **摘要:** Automated Speaking Assessment (ASA) plays a crucial role in evaluating second-language (L2) learners proficiency. However, ASA models often suffer from class imbalance, leading to biased predictions. To address this, we introduce a novel objective for training ASA models, dubbed the Balancing Logit Variation (BLV) loss, which perturbs model predictions to improve feature representation for minority classes without modifying the dataset. Evaluations on the ICNALE benchmark dataset show that integrating the BLV loss into a celebrated text-based (BERT) model significantly enhances classification accuracy and fairness, making automated speech evaluation more robust for diverse learners.
>
---
#### [replaced 007] Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于抑郁症检测任务，旨在解决现有方法难以捕捉语音中稀疏且异构的抑郁特征的问题。通过构建分层表示编码器，融合声学与语义信息，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2510.08593v2](https://arxiv.org/pdf/2510.08593v2)**

> **作者:** Yuxin Li; Eng Siong Chng; Cuntai Guan
>
> **摘要:** Speech-based depression detection (SDD) has emerged as a non-invasive and scalable alternative to conventional clinical assessments. However, existing methods still struggle to capture robust depression-related speech characteristics, which are sparse and heterogeneous. Although pretrained self-supervised learning (SSL) models provide rich representations, most recent SDD studies extract features from a single layer of the pretrained SSL model for the downstream classifier. This practice overlooks the complementary roles of low-level acoustic features and high-level semantic information inherently encoded in different SSL model layers. To explicitly model interactions between acoustic and semantic representations within an utterance, we propose a hierarchical adaptive representation encoder with prior knowledge that disengages and re-aligns acoustic and semantic information through asymmetric cross-attention, enabling fine-grained acoustic patterns to be interpreted in semantic context. In addition, a Connectionist Temporal Classification (CTC) objective is applied as auxiliary supervision to handle the irregular temporal distribution of depressive characteristics without requiring frame-level annotations. Experiments on DAIC-WOZ and MODMA demonstrate that HAREN-CTC consistently outperforms existing methods under both performance upper-bound evaluation and generalization evaluation settings, achieving Macro F1 scores of 0.81 and 0.82 respectively in upper-bound evaluation, and maintaining superior performance with statistically significant improvements in precision and AUC under rigorous cross-validation. These findings suggest that modeling hierarchical acoustic-semantic interactions better reflects how depressive characteristics manifest in natural speech, enabling scalable and objective depression assessment.
>
---
#### [replaced 008] Extending Audio Context for Long-Form Understanding in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型任务，解决长音频理解中上下文受限的问题。提出Partial YaRN和VLAT方法，扩展音频上下文长度并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.15231v2](https://arxiv.org/pdf/2510.15231v2)**

> **作者:** Yuatyong Chaichana; Pittawat Taveekitworachai; Warit Sirichotedumrong; Potsawee Manakul; Kunat Pipatanakul
>
> **备注:** EACL 2026. Code and dataset are available at: https://github.com/yophis/partial-yarn
>
> **摘要:** Large Audio-Language Models (LALMs) are often constrained by short audio context windows, even when their text backbones support long contexts, limiting long-form audio understanding. Prior work has introduced context-extension methods (e.g. YaRN) on unimodal LLMs, yet their application to LALMs remains unexplored. First, building on RoPE-based context extension, we introduce Partial YaRN, a training-free, modality-decoupled extension method that modifies only audio token positions, leaving text positions intact to preserve the base LLM's text capabilities. Second, we propose Virtual Longform Audio Training (VLAT), a training strategy that extends Partial YaRN into a training-time positional augmentation. VLAT simulates diverse audio lengths during training, enabling generalization to inputs far longer than those seen in training. Our experiments on SALMONN and Qwen2-Audio confirm that Partial YaRN outperforms the original models across wide range of settings, and VLAT provides substantial performance improvement on long audio of unseen lengths.
>
---
#### [replaced 009] Rec-RIR: Monaural Blind Room Impulse Response Identification via DNN-based Reverberant Speech Reconstruction in STFT Domain
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音增强任务，解决单通道房间脉冲响应（RIR）识别问题。通过DNN估计CTF滤波器，并转换为RIR，实现高效准确的声学参数估计。**

- **链接: [https://arxiv.org/pdf/2509.15628v2](https://arxiv.org/pdf/2509.15628v2)**

> **作者:** Pengyu Wang; Xiaofei Li
>
> **备注:** 5 pages
>
> **摘要:** This paper presents Rec-RIR for monaural blind room impulse response (RIR) identification. Rec-RIR is developed based on the convolutive transfer function (CTF) approximation, which models reverberation effect within narrow-band filter banks in the short-time Fourier transform domain. Specifically, we propose a deep neural network (DNN) with cross-band and narrow-band blocks to estimate the CTF filter. The DNN is trained through reconstructing the noise-free reverberant speech spectra. This objective enables stable and straightforward supervised training. Subsequently, a pseudo intrusive measurement process is employed to convert the CTF filter estimate into RIR by simulating a common intrusive RIR measurement procedure. Experimental results demonstrate that Rec-RIR achieves state-of-the-art performance in both RIR identification and acoustic parameter estimation. Open-source codes are available online at https://github.com/Audio-WestlakeU/Rec-RIR.
>
---
#### [replaced 010] Principled Coarse-Grained Acceptance for Speculative Decoding in Speech
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于语音生成任务，解决语音大模型中令牌匹配过严导致加速受限的问题。通过引入基于声学相似性的粗粒度验证方法，提升接受率和吞吐量。**

- **链接: [https://arxiv.org/pdf/2511.13732v3](https://arxiv.org/pdf/2511.13732v3)**

> **作者:** Moran Yanuka; Paul Dixon; Eyal Finkelshtein; Daniel Rotman; Raja Giryes
>
> **摘要:** Speculative decoding accelerates autoregressive speech generation by letting a fast draft model propose tokens that a larger target model verifies. However, for speech LLMs that generate acoustic tokens, exact token matching is overly restrictive: many discrete tokens are acoustically or semantically interchangeable, reducing acceptance rates and limiting speedups. We introduce Principled Coarse-Graining (PCG), which verifies proposals at the level of Acoustic Similarity Groups (ASGs) derived from the target model's embedding space. By splitting each token's probability mass across the overlapping groups that contain it, we define an overlap-aware coarse-grained distribution and perform rejection sampling on the resulting group variable. This yields an exactness guarantee at the group level while allowing the accepted draft token to stand in for any member of the group in practice. On LibriTTS, PCG increases acceptance and throughput relative to standard speculative decoding and prior speech-specific relaxations while maintaining intelligibility and speaker similarity. These results suggest acoustically aware, group-level acceptance as a simple and general way to accelerate speech token generation while maintaining speech quality.
>
---
#### [replaced 011] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音领域，解决语音基础模型在真实场景中因声学域变化导致的性能下降问题。提出E-BATS框架，实现高效无反向传播的测试时适应。**

- **链接: [https://arxiv.org/pdf/2506.07078v2](https://arxiv.org/pdf/2506.07078v2)**

> **作者:** Jiaheng Dong; Hong Jia; Soumyajit Chatterjee; Abhirup Ghosh; James Bailey; Ting Dang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4.1%-13.5% accuracy gains over backpropagation-free baselines and 2.0-6.4 times GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments.
>
---
#### [replaced 012] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音问答任务，旨在解决长音频处理难题。提出CLSR模型，通过对比学习高效提取相关语音片段，提升长格式语音问答性能。**

- **链接: [https://arxiv.org/pdf/2511.09282v2](https://arxiv.org/pdf/2511.09282v2)**

> **作者:** Jiliang Hu; Zuchao Li; Baoyuan Qi; Liu Guoming; Ping Wang
>
> **备注:** 12 pages, 7 figures, accepted by AAAI 2026
>
> **摘要:** Significant progress has been made in spoken question answering (SQA) in recent years. However, many existing methods, including large audio language models, struggle with processing long audio. Follow the success of retrieval augmented generation, a speech-related retriever shows promising in help preprocessing long-form speech. But the performance of existing speech-related retrievers is lacking. To address this challenge, we propose CLSR, an end-to-end contrastive language-speech retriever that efficiently extracts question-relevant segments from long audio recordings for downstream SQA task. Unlike conventional speech-text contrastive models, CLSR incorporates an intermediate step that converts acoustic features into text-like representations prior to alignment, thereby more effectively bridging the gap between modalities. Experimental results across four cross-modal retrieval datasets demonstrate that CLSR surpasses both end-to-end speech related retrievers and pipeline approaches combining speech recognition with text retrieval, providing a robust foundation for advancing practical long-form SQA applications.
>
---
#### [replaced 013] A Comparative Evaluation of Deep Learning Models for Speech Enhancement in Real-World Noisy Environments
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决噪声环境下语音质量与可懂度问题。通过对比Wave-U-Net、CMGAN和U-Net模型，评估其在噪声抑制、感知质量和说话人特征保留方面的性能。**

- **链接: [https://arxiv.org/pdf/2506.15000v2](https://arxiv.org/pdf/2506.15000v2)**

> **作者:** Md Jahangir Alam Khondkar; Ajan Ahmed; Stephanie Schuckers; Masudul Haider Imtiaz
>
> **摘要:** Speech enhancement, particularly denoising, is vital in improving the intelligibility and quality of speech signals for real-world applications, especially in noisy environments. While prior research has introduced various deep learning models for this purpose, many struggle to balance noise suppression, perceptual quality, and speaker-specific feature preservation, leaving a critical research gap in their comparative performance evaluation. This study benchmarks three state-of-the-art models Wave-U-Net, CMGAN, and U-Net, on diverse datasets such as SpEAR, VPQAD, and Clarkson datasets. These models were chosen due to their relevance in the literature and code accessibility. The evaluation reveals that U-Net achieves high noise suppression with SNR improvements of +71.96% on SpEAR, +64.83% on VPQAD, and +364.2% on the Clarkson dataset. CMGAN outperforms in perceptual quality, attaining the highest PESQ scores of 4.04 on SpEAR and 1.46 on VPQAD, making it well-suited for applications prioritizing natural and intelligible speech. Wave-U-Net balances these attributes with improvements in speaker-specific feature retention, evidenced by VeriSpeak score gains of +10.84% on SpEAR and +27.38% on VPQAD. This research indicates how advanced methods can optimize trade-offs between noise suppression, perceptual quality, and speaker recognition. The findings may contribute to advancing voice biometrics, forensic audio analysis, telecommunication, and speaker verification in challenging acoustic conditions.
>
---
#### [replaced 014] Unsupervised Variational Acoustic Clustering
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于音频聚类任务，旨在提升音频数据的聚类效果。提出一种基于变分推理的自编码器模型，利用高斯混合先验优化潜在空间，显著提高聚类准确性。**

- **链接: [https://arxiv.org/pdf/2503.18579v3](https://arxiv.org/pdf/2503.18579v3)**

> **作者:** Luan Vinícius Fiorio; Bruno Defraene; Johan David; Frans Widdershoven; Wim van Houtum; Ronald M. Aarts
>
> **备注:** Please refer to arXiv:2510.01940 for an extended version
>
> **摘要:** We propose an unsupervised variational acoustic clustering model for clustering audio data in the time-frequency domain. The model leverages variational inference, extended to an autoencoder framework, with a Gaussian mixture model as a prior for the latent space. Specifically designed for audio applications, we introduce a convolutional-recurrent variational autoencoder optimized for efficient time-frequency processing. Our experimental results considering a spoken digits dataset demonstrate a significant improvement in accuracy and clustering performance compared to traditional methods, showcasing the model's enhanced ability to capture complex audio patterns.
>
---
#### [replaced 015] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
- **分类: eess.AS**

- **简介: 该论文提出FCaps数据集和CLSP模型，解决语音-文本预训练中细粒度风格建模问题，通过多粒度对比学习提升跨模态理解性能。**

- **链接: [https://arxiv.org/pdf/2601.03065v2](https://arxiv.org/pdf/2601.03065v2)**

> **作者:** Yifan Yang; Bing Han; Hui Wang; Wei Wang; Ziyang Ma; Long Zhou; Zengrui Jin; Guanrou Yang; Tianrui Wang; Xu Tan; Xie Chen
>
> **摘要:** Modeling fine-grained speaking styles remains challenging for language-speech representation pre-training, as existing speech-text models are typically trained with coarse captions or task-specific supervision, and scalable fine-grained style annotations are unavailable. We present FCaps, a large-scale dataset with fine-grained free-text style descriptions, encompassing 47k hours of speech and 19M fine-grained captions annotated via a novel end-to-end pipeline that directly grounds detailed captions in audio, thereby avoiding the error propagation caused by LLM-based rewriting in existing cascaded pipelines. Evaluations using LLM-as-a-judge demonstrate that our annotations surpass existing cascaded annotations in terms of correctness, coverage, and naturalness. Building on FCaps, we propose CLSP, a contrastive language-speech pre-trained model that integrates global and fine-grained supervision, enabling unified representations across multiple granularities. Extensive experiments demonstrate that CLSP learns fine-grained and multi-granular speech-text representations that perform reliably across global and fine-grained speech-text retrieval, zero-shot paralinguistic classification, and speech style similarity scoring, with strong alignment to human judgments. Code and dataset are publicly available at https://github.com/yfyeung/CLSP.
>
---
#### [replaced 016] ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 该论文属于语音与声音深度伪造检测任务，旨在解决真实环境中组件级伪造检测难题。通过构建数据集和联合学习框架，提出ESDD2挑战赛以提升检测能力。**

- **链接: [https://arxiv.org/pdf/2601.07303v4](https://arxiv.org/pdf/2601.07303v4)**

> **作者:** Xueping Zhang; Han Yin; Yang Xiao; Lin Zhang; Ting Dang; Rohan Kumar Das; Ming Li
>
> **摘要:** Audio recorded in real-world environments often contains a mixture of foreground speech and background environmental sounds. With rapid advances in text-to-speech, voice conversion, and other generation models, either component can now be modified independently. Such component-level manipulations are harder to detect, as the remaining unaltered component can mislead the systems designed for whole deepfake audio, and they often sound more natural to human listeners. To address this gap, we have proposed CompSpoofV2 dataset and a separation-enhanced joint learning framework. CompSpoofV2 is a large-scale curated dataset designed for component-level audio anti-spoofing, which contains over 250k audio samples, with a total duration of approximately 283 hours. Based on the CompSpoofV2 and the separation-enhanced joint learning framework, we launch the Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2), focusing on component-level spoofing, where both speech and environmental sounds may be manipulated or synthesized, creating a more challenging and realistic detection scenario. The challenge will be held in conjunction with the IEEE International Conference on Multimedia and Expo 2026 (ICME 2026).
>
---
#### [replaced 017] Acoustic Non-Stationarity Objective Assessment with Hard Label Criteria for Supervised Learning Models
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于声学非平稳性评估任务，解决传统方法计算复杂的问题。提出HLC算法生成全局标签，构建NANSA模型，实现高效准确的非平稳性分类。**

- **链接: [https://arxiv.org/pdf/2508.06405v2](https://arxiv.org/pdf/2508.06405v2)**

> **作者:** Guilherme Zucatelli; Ricardo Barioni; Gabriela Dantas
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Objective non-stationarity measures are resource intensive and impose critical limitations for real-time processing solutions. In this paper, a novel Hard Label Criteria (HLC) algorithm is proposed to generate a global non-stationarity label for acoustic signals, enabling supervised learning strategies to be trained as stationarity estimators. The HLC is first evaluated on state-of-the-art general-purpose acoustic models, demonstrating that these models capture stationarity information. Furthermore, the first-of-its-kind HLC-based Network for Acoustic Non-Stationarity Assessment (NANSA) is proposed. NANSA models outperform competing approaches, achieving up to 99% classification accuracy, while solving the computational infeasibility of traditional objective measures.
>
---
#### [replaced 018] Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 论文提出Omni-AVSR，解决多模态语音识别任务中的模型独立训练与资源消耗问题，通过统一框架实现ASR、VSR和AVSR的高效联合训练与推理。**

- **链接: [https://arxiv.org/pdf/2511.07253v2](https://arxiv.org/pdf/2511.07253v2)**

> **作者:** Umberto Cappellazzo; Xubo Liu; Pingchuan Ma; Stavros Petridis; Maja Pantic
>
> **备注:** Accepted to IEEE ICASSP 2026 (camera-ready version). Project website (code and model weights): https://umbertocappellazzo.github.io/Omni-AVSR/
>
> **摘要:** Large language models (LLMs) have recently achieved impressive results in speech recognition across multiple modalities, including Auditory Speech Recognition (ASR), Visual Speech Recognition (VSR), and Audio-Visual Speech Recognition (AVSR). Despite this progress, current LLM-based approaches typically address each task independently, training separate models that raise computational and deployment resource use while missing potential cross-task synergies. They also rely on fixed-rate token compression, which restricts flexibility in balancing accuracy with efficiency. These limitations highlight the need for a unified framework that can support ASR, VSR, and AVSR while enabling elastic inference. To this end, we present Omni-AVSR, a unified audio-visual LLM that combines efficient multi-granularity training with parameter-efficient adaptation. Specifically, we adapt the matryoshka representation learning paradigm to efficiently train across multiple audio and visual granularities, reducing its inherent training resource use. Furthermore, we explore three LoRA-based strategies for adapting the backbone LLM, balancing shared and task-specific specialization. Experiments on LRS2 and LRS3 show that Omni-AVSR achieves comparable or superior accuracy to state-of-the-art baselines while training a single model at substantially lower training and deployment resource use. The model also remains robust under acoustic noise, and we analyze its scaling behavior as LLM size increases, providing insights into the trade-off between performance and efficiency.
>
---
#### [replaced 019] Clustering of Acoustic Environments with Variational Autoencoders for Hearing Devices
- **分类: eess.AS**

- **简介: 该论文属于声学环境聚类任务，旨在解决传统方法在高维数据表示和标签依赖上的问题。通过改进的变分自编码器实现无监督聚类。**

- **链接: [https://arxiv.org/pdf/2510.01940v3](https://arxiv.org/pdf/2510.01940v3)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Wim van Houtum; Ronald M. Aarts
>
> **摘要:** Traditional acoustic environment classification relies on: i) classical signal processing algorithms, which are unable to extract meaningful representations of high-dimensional data; or on ii) supervised learning, limited by the availability of labels. Knowing that human-imposed labels do not always reflect the true structure of acoustic scenes, we explore the potential of (unsupervised) clustering of acoustic environments using variational autoencoders (VAEs). We employ a VAE model for categorical latent clustering with a Gumbel-Softmax reparameterization which can operate with a time-context windowing scheme for lower memory requirements, tailored for real-world hearing device scenarios. Additionally, general adaptations on VAE architectures for audio clustering are also proposed. The approaches are validated through the clustering of spoken digits, a simpler task where labels are meaningful, and urban soundscapes, where the recordings present strong overlap in time and frequency. While all variational methods succeeded when clustering spoken digits, only the proposed model achieved effective clustering performance on urban acoustic scenes, given its categorical nature.
>
---
#### [replaced 020] Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Falcon3-Audio，一种高效音频-语言模型，解决音频与语言融合不足的问题。使用少量公开数据，实现高性能，强调数据效率与简单训练流程。**

- **链接: [https://arxiv.org/pdf/2509.07526v2](https://arxiv.org/pdf/2509.07526v2)**

> **作者:** Gokul Karthik Kumar; Rishabh Saraf; Ludovick Lepauloux; Abdul Muneer; Billel Mokeddem; Hakim Hacid
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data, less than 30K hours (5K unique), Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors are not required for strong performance, even compared to models trained on over 500K hours of data.
>
---
#### [replaced 021] Categorical Unsupervised Variational Acoustic Clustering
- **分类: eess.AS**

- **简介: 该论文属于音频聚类任务，旨在解决城市声景中数据点重叠导致的聚类困难问题。通过引入类别分布和Gumbel-Softmax方法，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2504.07652v3](https://arxiv.org/pdf/2504.07652v3)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Ronald M. Aarts
>
> **备注:** Please refer to arXiv:2510.01940 for an extended version
>
> **摘要:** We propose a categorical approach for unsupervised variational acoustic clustering of audio data in the time-frequency domain. The consideration of a categorical distribution enforces sharper clustering even when data points strongly overlap in time and frequency, which is the case for most datasets of urban acoustic scenes. To this end, we use a Gumbel-Softmax distribution as a soft approximation to the categorical distribution, allowing for training via backpropagation. In this settings, the softmax temperature serves as the main mechanism to tune clustering performance. The results show that the proposed model can obtain impressive clustering performance for all considered datasets, even when data points strongly overlap in time and frequency.
>
---
#### [replaced 022] Adaptive Rotary Steering with Joint Autoregression for Robust Extraction of Closely Moving Speakers in Dynamic Scenarios
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音增强任务，解决动态场景中紧密移动说话人的跟踪与增强问题。通过联合自回归框架，利用时频相关性提升空间分离效果。**

- **链接: [https://arxiv.org/pdf/2601.12345v2](https://arxiv.org/pdf/2601.12345v2)**

> **作者:** Jakob Kienegger; Timo Gerkmann
>
> **备注:** Accepted at IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** Latest advances in deep spatial filtering for Ambisonics demonstrate strong performance in stationary multi-speaker scenarios by rotating the sound field toward a target speaker prior to multi-channel enhancement. For applicability in dynamic acoustic conditions with moving speakers, we propose to automate this rotary steering using an interleaved tracking algorithm conditioned on the target's initial direction. However, for nearby or crossing speakers, robust tracking becomes difficult and spatial cues less effective for enhancement. By incorporating the processed recording as additional guide into both algorithms, our novel joint autoregressive framework leverages temporal-spectral correlations of speech to resolve spatially challenging speaker constellations. Consequently, our proposed method significantly improves tracking and enhancement of closely spaced speakers, consistently outperforming comparable non-autoregressive methods on a synthetic dataset. Real-world recordings complement these findings in complex scenarios with multiple speaker crossings and varying speaker-to-array distances.
>
---
