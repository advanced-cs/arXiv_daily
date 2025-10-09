# 音频 cs.SD;  eess.SP

- **最新发布 9 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Pitch Estimation With Mean Averaging Smoothed Product Spectrum And Musical Consonance Evaluation Using MASP
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决复杂频谱中准确估计音高和评估音乐协和性的问题。作者提出了MASP算法，改进传统HPS方法，通过全局均值平滑增强对缺失分音的鲁棒性，并扩展用于衡量音高和谐度，实现符合听觉感知的音高与协和性分析。**

- **链接: [http://arxiv.org/pdf/2510.06625v1](http://arxiv.org/pdf/2510.06625v1)**

> **作者:** Murat Yasar Baskin
>
> **摘要:** This study introduces Mean Averaging Smoothed Product (MASP) Spectrum, which is a modified version of the Harmonic Product Spectrum, designed to enhance pitch estimation for many algorithm-wise deceptive frequency spectra that still lead clear pitches, for both harmonic and inharmonic cases. By introducing a global mean based smoothing for spectrum, the MASP algorithm diminishes the unwanted sensitivity of HPS for spectra with missing partials. The method exhibited robust pitch estimations consistent with perceptual expectations. Motivated upon the strong correlation between consonance and periodicity, the same algorithm is extended and, with the proposition of a harmonicity measure (H), used to evaluate musical consonance for two and three tones; yielding consonance hierarchies that align with perception and practice of music theory. These findings suggest that perception of pitch and consonance may share a similar underlying mechanism that depend on spectrum.
>
---
#### [new 002] Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于语音安全任务，旨在解决当前虚假语音检测系统在面对多种生成模型时的鲁棒性不足问题。论文对8种最先进检测模型进行了跨领域评估，并提出统一评估指标，揭示了现有系统的安全漏洞，为提升虚假语音检测技术提供了方法和建议。**

- **链接: [http://arxiv.org/pdf/2510.06544v1](http://arxiv.org/pdf/2510.06544v1)**

> **作者:** Xutao Mao; Ke Li; Cameron Baird; Ezra Xuanru Tao; Dan Lin
>
> **摘要:** As advances in synthetic voice generation accelerate, an increasing variety of fake voice generators have emerged, producing audio that is often indistinguishable from real human speech. This evolution poses new and serious threats across sectors where audio recordings serve as critical evidence. Although fake voice detectors are also advancing, the arms race between fake voice generation and detection has become more intense and complex. In this work, we present the first large-scale, cross-domain evaluation of fake voice detectors, benchmarking 8 state-of-the-art models against datasets synthesized by 20 different fake voice generation systems. To the best of our knowledge, this is the most comprehensive cross-domain assessment conducted to date. Our study reveals substantial security vulnerabilities in current fake voice detection systems, underscoring critical gaps in their real-world robustness. To advance the field, we propose a unified and effective metric that consolidates the diverse and often inconsistent evaluation criteria previously used across different studies. This metric enables standardized, straightforward comparisons of the robustness of fake voice detectors. We conclude by offering actionable recommendations for building more resilient fake voice detection technologies, with the broader goal of reinforcing the foundations of AI security and trustworthiness.
>
---
#### [new 003] AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频理解任务，旨在解决长时音频处理中模型性能下降的问题。作者构建了AudioMarathon基准，包含长时音频输入、多领域覆盖及复杂推理任务，评估现有模型性能并探索加速技术，推动更高效、具时序推理能力的音频模型发展。**

- **链接: [http://arxiv.org/pdf/2510.07293v1](http://arxiv.org/pdf/2510.07293v1)**

> **作者:** Peize He; Zichen Wen; Yubo Wang; Yuxuan Wang; Xiaoqian Liu; Jiajie Huang; Zehui Lei; Zhuangcheng Gu; Xiangqi Jin; Jiabing Yang; Kai Li; Zhifei Liu; Weijia Li; Cunxiang Wang; Conghui He; Linfeng Zhang
>
> **备注:** 26 pages, 23 figures, the code is available at \url{https://github.com/DabDans/AudioMarathon}
>
> **摘要:** Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks.
>
---
#### [new 004] XLSR-Kanformer: A KAN-Intergrated model for Synthetic Speech Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文属于语音安全任务，旨在解决合成语音攻击威胁自动说话人验证系统的问题。作者将XLSR-Conformer模型中的MLP替换为基于Kolmogorov-Arnold定理的KAN网络，提升了检测性能，并验证了其在不同自监督模型中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06706v1](http://arxiv.org/pdf/2510.06706v1)**

> **作者:** Phuong Tuan Dat; Tran Huy Dat
>
> **备注:** Accepted to 2025 IEEE International Conference on Advanced Video and Signal-Based Surveillance
>
> **摘要:** Recent advancements in speech synthesis technologies have led to increasingly sophisticated spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer architecture, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron (MLP) in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a powerful universal approximator based on the Kolmogorov-Arnold representation theorem. Our experimental results on ASVspoof2021 demonstrate that the integration of KAN to XLSR-Conformer model can improve the performance by 60.55% relatively in Equal Error Rate (EER) LA and DF sets, further achieving 0.70% EER on the 21LA set. Besides, the proposed replacement is also robust to various SSL architectures. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection.
>
---
#### [new 005] BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于自动和弦识别（ACR）任务，旨在解决现有方法在符号音乐数据上表现不足及忽视人类音乐分析习惯的问题。作者提出了BACHI模型，通过边界感知与迭代解码机制，分步识别和弦根音、性质与低音，并构建了POP909-CL数据集以支持模型训练与评估。**

- **链接: [http://arxiv.org/pdf/2510.06528v1](http://arxiv.org/pdf/2510.06528v1)**

> **作者:** Mingyang Yao; Ke Chen; Shlomo Dubnov; Taylor Berg-Kirkpatrick
>
> **备注:** Under review
>
> **摘要:** Automatic chord recognition (ACR) via deep learning models has gradually achieved promising recognition accuracy, yet two key challenges remain. First, prior work has primarily focused on audio-domain ACR, while symbolic music (e.g., score) ACR has received limited attention due to data scarcity. Second, existing methods still overlook strategies that are aligned with human music analytical practices. To address these challenges, we make two contributions: (1) we introduce POP909-CL, an enhanced version of POP909 dataset with tempo-aligned content and human-corrected labels of chords, beats, keys, and time signatures; and (2) We propose BACHI, a symbolic chord recognition model that decomposes the task into different decision steps, namely boundary detection and iterative ranking of chord root, quality, and bass (inversion). This mechanism mirrors the human ear-training practices. Experiments demonstrate that BACHI achieves state-of-the-art chord recognition performance on both classical and pop music benchmarks, with ablation studies validating the effectiveness of each module.
>
---
#### [new 006] Moises-Light: Resource-efficient Band-split U-Net For Music Source Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音乐源分离任务，旨在解决现有模型参数多、资源消耗大、难以在低资源设备上应用的问题。作者提出轻量级模型Moises-Light，在保持高性能的同时显著减少参数量，实现了与更大模型相当的分离效果。**

- **链接: [http://arxiv.org/pdf/2510.06785v1](http://arxiv.org/pdf/2510.06785v1)**

> **作者:** Yun-Ning; Hung; Igor Pereira; Filip Korzeniowski
>
> **摘要:** In recent years, significant advances have been made in music source separation, with model architectures such as dual-path modeling, band-split modules, or transformer layers achieving comparably good results. However, these models often contain a significant number of parameters, posing challenges to devices with limited computational resources in terms of training and practical application. While some lightweight models have been introduced, they generally perform worse compared to their larger counterparts. In this paper, we take inspiration from these recent advances to improve a lightweight model. We demonstrate that with careful design, a lightweight model can achieve comparable SDRs to models with up to 13 times more parameters. Our proposed model, Moises-Light, achieves competitive results in separating four musical stems on the MUSDB-HQ benchmark dataset. The proposed model also demonstrates competitive scalability when using MoisesDB as additional training data.
>
---
#### [new 007] Comparison of Speech Tasks in Human Expert and Machine Detection of Parkinson's Disease
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分析与帕金森病检测任务，旨在比较人类专家与机器学习系统（基于Whisper）通过语音识别帕金森病的能力。研究分析五种语音任务中人类判断的依据，并评估Whisper在不同患者子群体中的表现。结果显示Whisper在音频单一模态下表现优于或相当于专家，尤其适用于年轻、轻症及女性患者。**

- **链接: [http://arxiv.org/pdf/2510.07299v1](http://arxiv.org/pdf/2510.07299v1)**

> **作者:** Peter Plantinga; Roozbeh Sattari; Karine Marcotte; Carla Di Gironimo; Madeleine Sharp; Liziane Bouvier; Maiya Geddes; Ingrid Verduyckt; Étienne de Villers-Sidani; Mirco Ravanelli; Denise Klein
>
> **备注:** Accepted to SMASH 2025
>
> **摘要:** The speech of people with Parkinson's Disease (PD) has been shown to hold important clues about the presence and progression of the disease. We investigate the factors based on which humans experts make judgments of the presence of disease in speech samples over five different speech tasks: phonations, sentence repetition, reading, recall, and picture description. We make comparisons by conducting listening tests to determine clinicians accuracy at recognizing signs of PD from audio alone, and we conduct experiments with a machine learning system for detection based on Whisper. Across tasks, Whisper performs on par or better than human experts when only audio is available, especially on challenging but important subgroups of the data: younger patients, mild cases, and female patients. Whisper's ability to recognize acoustic cues in difficult cases complements the multimodal and contextual strengths of human experts.
>
---
#### [new 008] Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决合成带有讽刺语气的自然语音问题。通过结合大语言模型的语义理解和检索增强生成的韵律示例，改进讽刺语音的表达效果。**

- **链接: [http://arxiv.org/pdf/2510.07096v1](http://arxiv.org/pdf/2510.07096v1)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Sarcasm is a subtle form of non-literal language that poses significant challenges for speech synthesis due to its reliance on nuanced semantic, contextual, and prosodic cues. While existing speech synthesis research has focused primarily on broad emotional categories, sarcasm remains largely unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which provide expressive reference patterns of sarcastic delivery. Integrated within a VITS backbone, this dual conditioning enables more natural and contextually appropriate sarcastic speech. Experiments demonstrate that our method outperforms baselines in both objective measures and subjective evaluations, yielding improvements in speech naturalness, sarcastic expressivity, and downstream sarcasm detection.
>
---
#### [new 009] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决当前ASR评估缺乏多语言和长文本支持、效率指标不统一的问题。作者构建了可复现的Open ASR Leaderboard，评估60多个系统在11个数据集上的准确率（WER）与效率（RTFx），并开源代码与数据集加载器，推动评估透明化。**

- **链接: [http://arxiv.org/pdf/2510.06961v1](http://arxiv.org/pdf/2510.06961v1)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Nithin Koluguri; Piotr Żelasko; Somshubra Majumdar; Adel Moumen; Sanchit Gandhi
>
> **备注:** Submitted to ICASSP 2026; Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard; Code: https://github.com/huggingface/open_asr_leaderboard
>
> **摘要:** Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including dedicated multilingual and long-form tracks. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation.
>
---
## 更新

#### [replaced 001] Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.15907v2](http://arxiv.org/pdf/2501.15907v2)**

> **作者:** Haorui He; Zengqiang Shang; Chaoren Wang; Xuyuan Li; Yicheng Gu; Hua Hua; Liwei Liu; Chen Yang; Jiaqi Li; Peiyang Shi; Yuancheng Wang; Kai Chen; Pengyuan Zhang; Zhizheng Wu
>
> **备注:** Full version of arXiv:2407.05361, dataset is available at: https://huggingface.co/datasets/amphion/Emilia-Dataset
>
> **摘要:** Recent advancements in speech generation have been driven by large-scale training datasets. However, current models struggle to capture the spontaneity and variability inherent in real-world human speech, as they are primarily trained on audio-book datasets limited to formal, read-aloud speaking styles. To address this limitation, we introduce Emilia-Pipe, an open-source preprocessing pipeline designed to extract high-quality training data from valuable yet under-explored in-the-wild sources that capture spontaneous human speech in real-world contexts. Using Emilia-Pipe, we construct Emilia, which comprises over 101k hours of speech across six languages: English, Chinese, German, French, Japanese, and Korean. Furthermore, we expand Emilia to Emilia-Large, a dataset exceeding 216k hours, making it one of the largest open-source speech generation resources available. Extensive experiments show that Emilia-trained models produce markedly more spontaneous, human-like speech than those trained on traditional audio-book datasets, while matching their intelligibility. These models better capture diverse speaker timbres and the full spectrum of real-world conversational styles. Our work also highlights the importance of scaling dataset size for advancing speech generation performance and validates the effectiveness of Emilia for both multilingual and crosslingual speech generation tasks.
>
---
#### [replaced 002] Baseline Systems For The 2025 Low-Resource Audio Codec Challenge
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.00264v3](http://arxiv.org/pdf/2510.00264v3)**

> **作者:** Yusuf Ziya Isik; Rafał Łaganowski
>
> **备注:** Low-Resource Audio Codec Challenge 2025
>
> **摘要:** The Low-Resource Audio Codec (LRAC) Challenge aims to advance neural audio coding for deployment in resource-constrained environments. The first edition focuses on low-resource neural speech codecs that must operate reliably under everyday noise and reverberation, while satisfying strict constraints on computational complexity, latency, and bitrate. Track 1 targets transparency codecs, which aim to preserve the perceptual transparency of input speech under mild noise and reverberation. Track 2 addresses enhancement codecs, which combine coding and compression with denoising and dereverberation. This paper presents the official baseline systems for both tracks in the 2025 LRAC Challenge. The baselines are convolutional neural codec models with Residual Vector Quantization, trained end-to-end using a combination of adversarial and reconstruction objectives. We detail the data filtering and augmentation strategies, model architectures, optimization procedures, and checkpoint selection criteria.
>
---
#### [replaced 003] Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.17686v2](http://arxiv.org/pdf/2506.17686v2)**

> **作者:** Alican Gok; Oguzhan Buyuksolak; Osman Erman Okman; Murat Saraclar
>
> **备注:** Submitted to IEEE Signal Processing Letters, 5 pages, 3 figures
>
> **摘要:** Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
>
---
#### [replaced 004] A Differentiable Alignment Framework for Sequence-to-Sequence Modeling via Optimal Transport
- **分类: cs.LG; cs.SD; eess.AS; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.01588v2](http://arxiv.org/pdf/2502.01588v2)**

> **作者:** Yacouba Kaloga; Shashi Kumar; Petr Motlicek; Ina Kodrasi
>
> **摘要:** Accurate sequence-to-sequence (seq2seq) alignment is critical for applications like medical speech analysis and language learning tools relying on automatic speech recognition (ASR). State-of-the-art end-to-end (E2E) ASR systems, such as the Connectionist Temporal Classification (CTC) and transducer-based models, suffer from peaky behavior and alignment inaccuracies. In this paper, we propose a novel differentiable alignment framework based on one-dimensional optimal transport, enabling the model to learn a single alignment and perform ASR in an E2E manner. We introduce a pseudo-metric, called Sequence Optimal Transport Distance (SOTD), over the sequence space and discuss its theoretical properties. Based on the SOTD, we propose Optimal Temporal Transport Classification (OTTC) loss for ASR and contrast its behavior with CTC. Experimental results on the TIMIT, AMI, and LibriSpeech datasets show that our method considerably improves alignment performance compared to CTC and the more recently proposed Consistency-Regularized CTC, though with a trade-off in ASR performance. We believe this work opens new avenues for seq2seq alignment research, providing a solid foundation for further exploration and development within the community.
>
---
#### [replaced 005] Token-based Audio Inpainting via Discrete Diffusion
- **分类: cs.SD; cs.AI; cs.IT; cs.LG; eess.AS; math.IT**

- **链接: [http://arxiv.org/pdf/2507.08333v3](http://arxiv.org/pdf/2507.08333v3)**

> **作者:** Tali Dror; Iftach Shoham; Moshe Buchris; Oren Gal; Haim Permuter; Gilad Katz; Eliya Nachmani
>
> **摘要:** Audio inpainting seeks to restore missing segments in degraded recordings. Previous diffusion-based methods exhibit impaired performance when the missing region is large. We introduce the first approach that applies discrete diffusion over tokenized music representations from a pre-trained audio tokenizer, enabling stable and semantically coherent restoration of long gaps. Our method further incorporates two training approaches: a derivative-based regularization loss that enforces smooth temporal dynamics, and a span-based absorbing transition that provides structured corruption during diffusion. Experiments on the MusicNet and MAESTRO datasets with gaps up to 750 ms show that our approach consistently outperforms strong baselines across range of gap lengths, for gaps of 150 ms and above. This work advances musical audio restoration and introduces new directions for discrete diffusion model training. Audio examples of our proposed method can be found at https://iftach21.github.io/.
>
---
#### [replaced 006] TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling
- **分类: cs.IR; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.01698v3](http://arxiv.org/pdf/2510.01698v3)**

> **作者:** Seungheon Doh; Keunwoo Choi; Juhan Nam
>
> **备注:** Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
>
> **摘要:** While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
>
---
#### [replaced 007] Self-Supervised Speech Quality Assessment (S3QA): Leveraging Speech Foundation Models for a Scalable Speech Quality Metric
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.01655v2](http://arxiv.org/pdf/2506.01655v2)**

> **作者:** Mattson Ogg; Caitlyn Bishop; Han Yi; Sarah Robinson
>
> **备注:** 1 table, seven figures, thirteen pages
>
> **摘要:** Methods for automatically assessing speech quality in real world environments are critical for developing robust human language technologies and assistive devices. Behavioral ratings provided by human raters (e.g., mean opinion scores; MOS) are considered the gold standard, but they are susceptible to variability between individual raters, cannot easily be generalized across corpora, and are labor-intensive to collect, thus limiting the acoustic challenges they can quantify. Here, we present a new, scalable method for automatically assessing speech quality: the self-supervised speech quality assessment (S3QA) model. First, we manipulated high quality utterances from multiple speech corpora, using a wide range of acoustic challenges intended to emulate common sources of quality degradation in the real-world: frequency filtering, reverberation, background noise, and digital compression. Second, we leveraged an existing, pre-trained speech foundation model, WavLM, to computationally derive a self-supervised training target that quantified speech degradation using the cosine distance between the clean and degraded versions of each utterance in the embedding space. Next, we trained a transformer-based model to predict these cosine distances, given only the degraded versions of the utterances. Finally, the trained model was evaluated on unseen test corpora of synthetic mixtures, NISQA, and VOiCES. We show that the S3QA model trained on this task accurately predicts degradation cosine distances across a wide range challenging acoustic conditions and is aligned with both behavioral ratings (MOS), speech technology performance (automatic speech recognition) and other important features of the held-out data (e.g., microphone distances). This model provides an automated, scalable method for assessing speech quality across a wide range of acoustic challenges.
>
---
#### [replaced 008] AbsoluteNet: A Deep Learning Neural Network to Classify Cerebral Hemodynamic Responses of Auditory Processing
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00039v2](http://arxiv.org/pdf/2506.00039v2)**

> **作者:** Behtom Adeli; John Mclinden; Pankaj Pandey; Ming Shao; Yalda Shahriari
>
> **摘要:** In recent years, deep learning (DL) approaches have demonstrated promising results in decoding hemodynamic responses captured by functional near-infrared spectroscopy (fNIRS), particularly in the context of brain-computer interface (BCI) applications. This work introduces AbsoluteNet, a novel deep learning architecture designed to classify auditory event-related responses recorded using fNIRS. The proposed network is built upon principles of spatio-temporal convolution and customized activation functions. Our model was compared against several models, namely fNIRSNET, MDNN, DeepConvNet, and ShallowConvNet. The results showed that AbsoluteNet outperforms existing models, reaching 87.0% accuracy, 84.8% sensitivity, and 89.2% specificity in binary classification, surpassing fNIRSNET, the second-best model, by 3.8% in accuracy. These findings underscore the effectiveness of our proposed deep learning model in decoding hemodynamic responses related to auditory processing and highlight the importance of spatio-temporal feature aggregation and customized activation functions to better fit fNIRS dynamics.
>
---
#### [replaced 009] PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15556v2](http://arxiv.org/pdf/2506.15556v2)**

> **作者:** Shufan Li; Aditya Grover
>
> **备注:** 16 pages,4 figures
>
> **摘要:** Large Language Models (LLMs) are widely used in real-time voice chat applications, typically in combination with text-to-speech (TTS) systems to generate audio responses. However, their large size often leads to noticeable latency between the end of user input and the start of audio output, resulting in suboptimal user experiences. This latency is particularly evident when LLMs are deployed as single-user voice assistants on consumer-grade hardware with limited computing capacity. We discovered that this latency is primarily dominated by the time it takes for the LLMs to generate the first sentence, which is required as input by the TTS systems that synthesize audio responses on a sentence-by-sentence basis. To address this bottleneck, we propose Predictive Generation (PredGen), a novel framework that mitigates-or even eliminates-this delay through speculative decoding at input time. PredGen generates candidate responses while the user is still speaking, enabling the system to begin TTS processing with minimal delay. Simulated experiments on the Lmsys and MT-Bench datasets show that the proposed method can effectively reduce the latency by around 2x across a wide range of use cases, while incurring only minimal additional computation cost at input time-computation that would otherwise go unused.
>
---
#### [replaced 010] Lossy Neural Compression for Geospatial Analytics: A Review
- **分类: eess.SP; cs.AI; cs.CV; cs.LG; physics.geo-ph**

- **链接: [http://arxiv.org/pdf/2503.01505v2](http://arxiv.org/pdf/2503.01505v2)**

> **作者:** Carlos Gomes; Isabelle Wittmann; Damien Robert; Johannes Jakubik; Tim Reichelt; Michele Martone; Stefano Maurogiovanni; Rikard Vinge; Jonas Hurst; Erik Scheurer; Rocco Sedona; Thomas Brunschwiler; Stefan Kesselheim; Matej Batic; Philip Stier; Jan Dirk Wegner; Gabriele Cavallaro; Edzer Pebesma; Michael Marszalek; Miguel A Belenguer-Plomer; Kennedy Adriko; Paolo Fraccaro; Romeo Kienzler; Rania Briq; Sabrina Benassou; Michele Lazzarini; Conrad M Albrecht
>
> **备注:** self-consistent review paper
>
> **摘要:** Over the past decades, there has been an explosion in the amount of available Earth Observation (EO) data. The unprecedented coverage of the Earth's surface and atmosphere by satellite imagery has resulted in large volumes of data that must be transmitted to ground stations, stored in data centers, and distributed to end users. Modern Earth System Models (ESMs) face similar challenges, operating at high spatial and temporal resolutions, producing petabytes of data per simulated day. Data compression has gained relevance over the past decade, with neural compression (NC) emerging from deep learning and information theory, making EO data and ESM outputs ideal candidates due to their abundance of unlabeled data. In this review, we outline recent developments in NC applied to geospatial data. We introduce the fundamental concepts of NC including seminal works in its traditional applications to image and video compression domains with focus on lossy compression. We discuss the unique characteristics of EO and ESM data, contrasting them with "natural images", and explain the additional challenges and opportunities they present. Moreover, we review current applications of NC across various EO modalities and explore the limited efforts in ESM compression to date. The advent of self-supervised learning (SSL) and foundation models (FM) has advanced methods to efficiently distill representations from vast unlabeled data. We connect these developments to NC for EO, highlighting the similarities between the two fields and elaborate on the potential of transferring compressed feature representations for machine--to--machine communication. Based on insights drawn from this review, we devise future directions relevant to applications in EO and ESM.
>
---
#### [replaced 011] The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.16765v2](http://arxiv.org/pdf/2509.16765v2)**

> **作者:** Fagun Patel; Duc Q. Nguyen; Sang T. Truong; Jody Vaynshtok; Sanmi Koyejo; Nick Haber
>
> **备注:** EMNLP 2025 Oral Presentation
>
> **摘要:** According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 10\% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.
>
---
#### [replaced 012] LaunchpadGPT: Language Model as Music Visualization Designer on Launchpad
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2307.04827v3](http://arxiv.org/pdf/2307.04827v3)**

> **作者:** Siting Xu; Yolo Yunlong Tang; Feng Zheng
>
> **备注:** Accepted to International Computer Music Conference (ICMC) 2023
>
> **摘要:** Launchpad is a musical instrument that allows users to create and perform music by pressing illuminated buttons. To assist and inspire the design of the Launchpad light effect, and provide a more accessible approach for beginners to create music visualization with this instrument, we proposed the LaunchpadGPT model to generate music visualization designs on Launchpad automatically. Based on the language model with excellent generation ability, our proposed LaunchpadGPT takes an audio piece of music as input and outputs the lighting effects of Launchpad-playing in the form of a video (Launchpad-playing video). We collect Launchpad-playing videos and process them to obtain music and corresponding video frame of Launchpad-playing as prompt-completion pairs, to train the language model. The experiment result shows the proposed method can create better music visualization than random generation methods and hold the potential for a broader range of music visualization applications. Our code is available at https://github.com/yunlong10/LaunchpadGPT/.
>
---
#### [replaced 013] DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.15690v3](http://arxiv.org/pdf/2507.15690v3)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** Accepted to VCIP 2025
>
> **摘要:** Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
>
---
