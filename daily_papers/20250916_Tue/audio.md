# 音频 cs.SD;  eess.SP

- **最新发布 24 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Combining Audio and Non-Audio Inputs in Evolved Neural Networks for Ovenbird
- **分类: cs.SD; eess.AS**

- **简介: 论文研究如何通过结合音频（声谱图）与非音频数据（如栖息地、分布等）提升神经网络对 Ovenbird 的分类准确率。任务为物种分类，解决单一输入限制问题，验证多模态输入的有效性。**

- **链接: [http://arxiv.org/pdf/2509.10566v1](http://arxiv.org/pdf/2509.10566v1)**

> **作者:** Sergio Poo Hernandez; Vadim Bulitko; Erin Bayne
>
> **摘要:** In the last several years the use of neural networks as tools to automate species classification from digital data has increased. This has been due in part to the high classification accuracy of image classification through Convolutional Neural Networks (CNN). In the case of audio data CNN based recognizers are used to automate the classification of species in audio recordings by using information from sound visualization (i.e., spectrograms). It is common for these recognizers to use the spectrogram as their sole input. However, researchers have other non-audio data, such as habitat preferences of a species, phenology, and range information, available that could improve species classification. In this paper we present how a single-species recognizer neural network's accuracy can be improved by using non-audio data as inputs in addition to spectrogram information. We also analyze if the improvements are merely a result of having a neural network with a higher number of parameters instead of combining the two inputs. We find that networks that use the two different inputs have a higher classification accuracy than networks of similar size that use only one of the inputs.
>
---
#### [new 002] Neural Audio Codecs for Prompt-Driven Universal Source Separation
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出CodecSep，一种基于神经音频编解码器的文本驱动通用源分离模型，解决现有模型计算量大、仅支持固定类别分离的问题。CodecSep结合DAC压缩与CLAP调制的Transformer掩码器，在保持高质量的同时显著降低计算需求。**

- **链接: [http://arxiv.org/pdf/2509.11717v1](http://arxiv.org/pdf/2509.11717v1)**

> **作者:** Adhiraj Banerjee; Vipul Arora
>
> **备注:** 21 pages, 1 figure, pre-print, under review
>
> **摘要:** Text-guided source separation supports flexible audio editing across media and assistive applications, but existing models like AudioSep are too compute-heavy for edge deployment. Neural audio codec (NAC) models such as CodecFormer and SDCodec are compute-efficient but limited to fixed-class separation. We introduce CodecSep, the first NAC-based model for on-device universal, text-driven separation. CodecSep combines DAC compression with a Transformer masker modulated by CLAP-derived FiLM parameters. Across six open-domain benchmarks under matched training/prompt protocols, \textbf{CodecSep} surpasses \textbf{AudioSep} in separation fidelity (SI-SDR) while remaining competitive in perceptual quality (ViSQOL) and matching or exceeding fixed-stem baselines (TDANet, CodecFormer, SDCodec). In code-stream deployments, it needs just 1.35~GMACs end-to-end -- approximately $54\times$ less compute ($25\times$ architecture-only) than spectrogram-domain separators like AudioSep -- while remaining fully bitstream-compatible.
>
---
#### [new 003] Acoustic Overspecification in Electronic Dance Music Taxonomy
- **分类: cs.SD; cs.IR**

- **简介: 论文提出一种无监督方法，揭示EDM音乐的自然声学结构，发现当前商业分类存在过度细分问题。任务为音乐分类优化，解决分类标签与实际声学特征不符的问题，通过时程图特征和多标准选择实现。**

- **链接: [http://arxiv.org/pdf/2509.11474v1](http://arxiv.org/pdf/2509.11474v1)**

> **作者:** Weilun Xu; Tianhao Dai; Oscar Goudet; Xiaoxuan Wang
>
> **备注:** 5 pages, 3 figures, conference paper
>
> **摘要:** Electronic Dance Music (EDM) classification typically relies on industry-defined taxonomies with numerous subgenres, yet the acoustic basis for these distinctions remains unclear. Current approaches use supervised learning with prescribed genre labels, assuming their validity without systematic evaluation. In this paper, we propose an unsupervised approach to discover the natural acoustic structure of EDM independent of commercial labels. Our method combines novel tempogram-based features capturing EDM's layered rhythmic patterns with multi-criteria feature selection. To validate that our findings reflect genuine acoustic structure rather than methodological artifacts, we compare our results against state-of-the-art pre-trained audio embeddings (MERT and CLAP). Both our feature space and embedding representations converge to 19-23 natural acoustic families compared to the prescribed 35, providing consistent evidence of significant overspecification in current EDM taxonomy by approximately one-third.
>
---
#### [new 004] ENJ: Optimizing Noise with Genetic Algorithms to Jailbreak LSMs
- **分类: cs.SD; cs.AI**

- **简介: 论文提出ENJ方法，利用遗传算法优化噪声以攻击LSM。属于语音安全领域，解决传统对抗攻击效果与隐蔽性难以兼顾的问题。通过迭代进化生成融合恶意指令的音频样本，实现高效隐蔽的模型越狱攻击。**

- **链接: [http://arxiv.org/pdf/2509.11128v1](http://arxiv.org/pdf/2509.11128v1)**

> **作者:** Yibo Zhang; Liang Lin
>
> **摘要:** The widespread application of Large Speech Models (LSMs) has made their security risks increasingly prominent. Traditional speech adversarial attack methods face challenges in balancing effectiveness and stealth. This paper proposes Evolutionary Noise Jailbreak (ENJ), which utilizes a genetic algorithm to transform environmental noise from a passive interference into an actively optimizable attack carrier for jailbreaking LSMs. Through operations such as population initialization, crossover fusion, and probabilistic mutation, this method iteratively evolves a series of audio samples that fuse malicious instructions with background noise. These samples sound like harmless noise to humans but can induce the model to parse and execute harmful commands. Extensive experiments on multiple mainstream speech models show that ENJ's attack effectiveness is significantly superior to existing baseline methods. This research reveals the dual role of noise in speech security and provides new critical insights for model security defense in complex acoustic environments.
>
---
#### [new 005] When marine radar target detection meets pretrained large language models
- **分类: eess.SP; cs.CL; cs.LG**

- **简介: 论文将预训练大语言模型应用于海洋雷达目标检测，通过特征预处理和微调提升检测性能。属于雷达信号处理任务，解决传统深度学习方法中冗余特征和模型规模限制的问题。**

- **链接: [http://arxiv.org/pdf/2509.12110v1](http://arxiv.org/pdf/2509.12110v1)**

> **作者:** Qiying Hu; Linping Zhang; Xueqian Wang; Gang Li; Yu Liu; Xiao-Ping Zhang
>
> **摘要:** Deep learning (DL) methods are widely used to extract high-dimensional patterns from the sequence features of radar echo signals. However, conventional DL algorithms face challenges such as redundant feature segments, and constraints from restricted model sizes. To address these issues, we propose a framework that integrates feature preprocessing with large language models (LLMs). Our preprocessing module tokenizes radar sequence features, applies a patch selection algorithm to filter out uninformative segments, and projects the selected patches into embeddings compatible with the feature space of pre-trained LLMs. Leveraging these refined embeddings, we incorporate a pre-trained LLM, fine-tuning only the normalization layers to reduce training burdens while enhancing performance. Experiments on measured datasets demonstrate that the proposed method significantly outperforms the state-of-the-art baselines on supervised learning tests.
>
---
#### [new 006] RadarLLM: Adapting Pretrained Large Language Models for Marine Radar Target Detection with Preference-aware Loss
- **分类: eess.SP; cs.CL**

- **简介: 论文提出RadarLLM框架，用于改进预训练大语言模型在海洋雷达目标检测中的应用。针对低信杂比场景下模型过拟合问题，设计偏好感知损失函数，选择性优化特征块，提升泛化能力。实验表明其优于现有方法，尤其在数据有限时表现突出。**

- **链接: [http://arxiv.org/pdf/2509.12089v1](http://arxiv.org/pdf/2509.12089v1)**

> **作者:** Qiying Hu
>
> **摘要:** Recent advances in pre-trained large language models (LLMs) have demonstrated their capacities to capture universal knowledge, making them promising general-purpose optimization solvers for wireless signal processing. Motivated by these findings, we take the first step towards fine-tuning pre-trained LLMs for the effective analysis of radar signal features in marine target detection tasks. Nevertheless, directly fine-tuning pre-trained LLMs on marine target detection tasks tends to suffer from pronounced overfitting, particularly in challenging low signal-to-clutter ratio (SCR) scenarios. This overfitting primarily stems from the model's tendency to memorize spurious or noisy feature patterns rather than learning discriminative structures that generalize well to unseen data. To address this challenge, we introduce RadarLLM, a novel fine-tuning framework that utilizes an effective preference-aware loss. Unlike conventional training strategies that uniformly optimize all feature tokens, this loss function selectively optimizes different feature patches based on their online evaluated learning values, thus guiding the model to focus on the most generalizable patterns during optimization. We theoretically demonstrate the effectiveness of the evaluated learning values by transforming the problem as selecting useful feature tokens. Extensive experiments on real-world marine radar datasets show that 1) the proposed loss function is much better than the original one, with particularly significant gains in challenging low SCR scenarios and 2) RadarLLM consistently outperforms state-of-the-art baselines across diverse detection scenarios, with particularly notable gains under limited training data conditions.
>
---
#### [new 007] Improving Out-of-Domain Audio Deepfake Detection via Layer Selection and Fusion of SSL-Based Countermeasures
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对音频深度伪造检测任务，解决模型在未知领域（OOD）下的泛化问题。通过分析SSL编码器各层贡献，选择最优层并融合多模型得分，显著提升检测性能并减少参数量。**

- **链接: [http://arxiv.org/pdf/2509.12003v1](http://arxiv.org/pdf/2509.12003v1)**

> **作者:** Pierre Serrano; Raphaël Duroselle; Florian Angulo; Jean-François Bonastre; Olivier Boeffard
>
> **摘要:** Audio deepfake detection systems based on frozen pre-trained self-supervised learning (SSL) encoders show a high level of performance when combined with layer-weighted pooling methods, such as multi-head factorized attentive pooling (MHFA). However, they still struggle to generalize to out-of-domain (OOD) conditions. We tackle this problem by studying the behavior of six different pre-trained SSLs, on four different test corpora. We perform a layer-by-layer analysis to determine which layers contribute most. Next, we study the pooling head, comparing a strategy based on a single layer with automatic selection via MHFA. We observed that selecting the best layer gave very good results, while reducing system parameters by up to 80%. A wide variation in performance as a function of test corpus and SSL model is also observed, showing that the pre-training strategy of the encoder plays a role. Finally, score-level fusion of several encoders improved generalization to OOD attacks.
>
---
#### [new 008] An Entropy-Guided Curriculum Learning Strategy for Data-Efficient Acoustic Scene Classification under Domain Shift
- **分类: cs.SD; cs.AI**

- **简介: 论文提出一种基于熵引导的课程学习策略，用于解决数据高效声学场景分类中的领域偏移问题。该方法通过量化设备域预测的不确定性，逐步从高熵样本过渡到低熵样本进行训练，提升模型在少量标注数据下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.11168v1](http://arxiv.org/pdf/2509.11168v1)**

> **作者:** Peihong Zhang; Yuxuan Liu; Zhixin Li; Rui Sang; Yiqiang Cai; Yizhou Tan; Shengchen Li
>
> **备注:** Accepted at the Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop 2025
>
> **摘要:** Acoustic Scene Classification (ASC) faces challenges in generalizing across recording devices, particularly when labeled data is limited. The DCASE 2024 Challenge Task 1 highlights this issue by requiring models to learn from small labeled subsets recorded on a few devices. These models need to then generalize to recordings from previously unseen devices under strict complexity constraints. While techniques such as data augmentation and the use of pre-trained models are well-established for improving model generalization, optimizing the training strategy represents a complementary yet less-explored path that introduces no additional architectural complexity or inference overhead. Among various training strategies, curriculum learning offers a promising paradigm by structuring the learning process from easier to harder examples. In this work, we propose an entropy-guided curriculum learning strategy to address the domain shift problem in data-efficient ASC. Specifically, we quantify the uncertainty of device domain predictions for each training sample by computing the Shannon entropy of the device posterior probabilities estimated by an auxiliary domain classifier. Using entropy as a proxy for domain invariance, the curriculum begins with high-entropy samples and gradually incorporates low-entropy, domain-specific ones to facilitate the learning of generalizable representations. Experimental results on multiple DCASE 2024 ASC baselines demonstrate that our strategy effectively mitigates domain shift, particularly under limited labeled data conditions. Our strategy is architecture-agnostic and introduces no additional inference cost, making it easily integrable into existing ASC baselines and offering a practical solution to domain shift.
>
---
#### [new 009] FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出FuseCodec，解决神经编解码器忽略语义与上下文信息的问题。通过融合声学、语义和上下文表示，提升语音分词质量。实验表明其在语音转录等任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.11425v1](http://arxiv.org/pdf/2509.11425v1)**

> **作者:** Md Mubtasim Ahasan; Rafat Hasan Khan; Tasnim Mohiuddin; Aman Chadha; Tariq Iqbal; M Ashraful Amin; Amin Ahsan Ali; Md Mofijul Islam; A K M Mahbubur Rahman
>
> **摘要:** Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at https://github.com/mubtasimahasan/FuseCodec.
>
---
#### [new 010] Scaling to Multimodal and Multichannel Heart Sound Classification: Fine-Tuning Wav2Vec 2.0 with Synthetic and Augmented Biosignals
- **分类: cs.SD; cs.LG; eess.SP**

- **简介: 该论文属于心音分类任务，旨在解决心血管疾病早期检测中数据不足的问题。通过结合信号处理与扩散模型生成增强数据，微调Wav2Vec 2.0模型，在多模态和多通道心音数据上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2509.11606v1](http://arxiv.org/pdf/2509.11606v1)**

> **作者:** Milan Marocchi; Matthew Fynn; Kayapanda Mandana; Yue Rong
>
> **备注:** 35 pages, 37 figures, 19 tables
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading cause of death worldwide, accounting for approximately 17.9 million deaths each year. Early detection is critical, creating a demand for accurate and inexpensive pre-screening methods. Deep learning has recently been applied to classify abnormal heart sounds indicative of CVDs using synchronised phonocardiogram (PCG) and electrocardiogram (ECG) signals, as well as multichannel PCG (mPCG). However, state-of-the-art architectures remain underutilised due to the limited availability of synchronised and multichannel datasets. Augmented datasets and pre-trained models provide a pathway to overcome these limitations, enabling transformer-based architectures to be trained effectively. This work combines traditional signal processing with denoising diffusion models, WaveGrad and DiffWave, to create an augmented dataset to fine-tune a Wav2Vec 2.0-based classifier on multimodal and multichannel heart sound datasets. The approach achieves state-of-the-art performance. On the Computing in Cardiology (CinC) 2016 dataset of single channel PCG, accuracy, unweighted average recall (UAR), sensitivity, specificity and Matthew's correlation coefficient (MCC) reach 92.48\%, 93.05\%, 93.63\%, 92.48\%, 94.93\% and 0.8283, respectively. Using the synchronised PCG and ECG signals of the training-a dataset from CinC, 93.14\%, 92.21\%, 94.35\%, 90.10\%, 95.12\% and 0.8380 are achieved for accuracy, UAR, sensitivity, specificity and MCC, respectively. Using a wearable vest dataset consisting of mPCG data, the model achieves 77.13\% accuracy, 74.25\% UAR, 86.47\% sensitivity, 62.04\% specificity, and 0.5082 MCC. These results demonstrate the effectiveness of transformer-based models for CVD detection when supported by augmented datasets, highlighting their potential to advance multimodal and multichannel heart sound classification.
>
---
#### [new 011] Revisiting Meter Tracking in Carnatic Music using Deep Learning Approaches
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 论文研究使用深度学习方法在卡纳提克音乐中进行节拍跟踪。针对该音乐传统节奏复杂、数据不足的问题，评估了TCN和Beat This!模型，并通过迁移学习提升性能，证明深度学习可有效适应非主流音乐体系。**

- **链接: [http://arxiv.org/pdf/2509.11241v1](http://arxiv.org/pdf/2509.11241v1)**

> **作者:** Satyajeet Prabhu
>
> **摘要:** Beat and downbeat tracking, jointly referred to as Meter Tracking, is a fundamental task in Music Information Retrieval (MIR). Deep learning models have far surpassed traditional signal processing and classical machine learning approaches in this domain, particularly for Western (Eurogenetic) genres, where large annotated datasets are widely available. These systems, however, perform less reliably on underrepresented musical traditions. Carnatic music, a rich tradition from the Indian subcontinent, is renowned for its rhythmic intricacy and unique metrical structures (t\=alas). The most notable prior work on meter tracking in this context employed probabilistic Dynamic Bayesian Networks (DBNs). The performance of state-of-the-art (SOTA) deep learning models on Carnatic music, however, remains largely unexplored. In this study, we evaluate two models for meter tracking in Carnatic music: the Temporal Convolutional Network (TCN), a lightweight architecture that has been successfully adapted for Latin rhythms, and Beat This!, a transformer-based model designed for broad stylistic coverage without the need for post-processing. Replicating the experimental setup of the DBN baseline on the Carnatic Music Rhythm (CMR$_f$) dataset, we systematically assess the performance of these models in a directly comparable setting. We further investigate adaptation strategies, including fine-tuning the models on Carnatic data and the use of musically informed parameters. Results show that while off-the-shelf models do not always outperform the DBN, their performance improves substantially with transfer learning, matching or surpassing the baseline. These findings indicate that SOTA deep learning models can be effectively adapted to underrepresented traditions, paving the way for more inclusive and broadly applicable meter tracking systems.
>
---
#### [new 012] PoolingVQ: A VQVAE Variant for Reducing Audio Redundancy and Boosting Multi-Modal Fusion in Music Emotion Analysis
- **分类: cs.SD**

- **简介: 论文提出PoolingVQ，一种改进的VQVAE模型，用于减少音频冗余并提升音乐情感分析中的多模态融合。通过空间池化压缩音频特征，并设计两阶段共注意力机制融合音频与MIDI信息，实验表明其在EMOPIA和VGMIDI数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2509.11976v1](http://arxiv.org/pdf/2509.11976v1)**

> **作者:** Dinghao Zou; Yicheng Gong; Xiaokang Li; Xin Cao; Sunbowen Lee
>
> **摘要:** Multimodal music emotion analysis leverages audio and MIDI modalities to enhance performance. While mainstream approaches focus on complex feature extraction networks, we posit that shortening the length of audio sequence features to mitigate redundancy, especially in contrast to MIDI's compact representation, may effectively boost task performance. To achieve this, we developed PoolingVQ by combining Vector Quantized Variational Autoencoder (VQVAE) with spatial pooling, which directly compresses audio feature sequences through local aggregation to reduce redundancy, then devised a two-stage co-attention approach to fuse audio and MIDI information. Experimental results on the public datasets EMOPIA and VGMIDI demonstrate that our multimodal framework achieves state-of-the-art overall performance, with PoolingVQ yielding some improvement.
>
---
#### [new 013] Spectral and Rhythm Features for Audio Classification with Deep Convolutional Neural Networks
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 该论文研究使用深度卷积神经网络进行音频分类任务，探讨不同谱和节奏特征（如MFCC、梅尔频谱等）的性能。实验表明，MFCC和梅尔频谱表现最佳，使用ESC-50数据集进行验证。**

- **链接: [http://arxiv.org/pdf/2410.06927v2](http://arxiv.org/pdf/2410.06927v2)**

> **作者:** Friedrich Wolf-Monheim
>
> **摘要:** Convolutional neural networks (CNNs) are widely used in computer vision. They can be used not only for conventional digital image material to recognize patterns, but also for feature extraction from digital imagery representing spectral and rhythm features extracted from time-domain digital audio signals for the acoustic classification of sounds. Different spectral and rhythm feature representations like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCCs), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams are investigated in terms of the audio classification performance using a deep convolutional neural network. It can be clearly shown that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCCs) perform significantly better than the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs. The experiments were carried out with the aid of the ESC-50 dataset with 2,000 labeled environmental audio recordings.
>
---
#### [new 014] STASE: A spatialized text-to-audio synthesis engine for music generation
- **分类: cs.SD**

- **简介: 论文提出STASE系统，解决文本到空间音频生成问题。通过LLM解析空间提示，并结合物理渲染引擎实现可控的空间音频合成，提升用户对空间参数的直接控制能力。**

- **链接: [http://arxiv.org/pdf/2509.11124v1](http://arxiv.org/pdf/2509.11124v1)**

> **作者:** Tutti Chi; Letian Gao; Yixiao Zhang
>
> **备注:** Accepted to LLM4Music @ ISMIR 2025
>
> **摘要:** While many text-to-audio systems produce monophonic or fixed-stereo outputs, generating audio with user-defined spatial properties remains a challenge. Existing deep learning-based spatialization methods often rely on latent-space manipulations, which can limit direct control over psychoacoustic parameters critical to spatial perception. To address this, we introduce STASE, a system that leverages a Large Language Model (LLM) as an agent to interpret spatial cues from text. A key feature of STASE is the decoupling of semantic interpretation from a separate, physics-based spatial rendering engine, which facilitates interpretable and user-controllable spatial reasoning. The LLM processes prompts through two main pathways: (i) Description Prompts, for direct mapping of explicit spatial information (e.g., "place the lead guitar at 45{\deg} azimuth, 10 m distance"), and (ii) Abstract Prompts, where a Retrieval-Augmented Generation (RAG) module retrieves relevant spatial templates to inform the rendering. This paper details the STASE workflow, discusses implementation considerations, and highlights current challenges in evaluating generative spatial audio.
>
---
#### [new 015] Emoanti: audio anti-deepfake with refined emotion-guided representations
- **分类: cs.SD**

- **简介: 该论文提出EmoAnti系统，用于检测音频深度伪造。通过结合情绪引导的表示，提升检测性能。使用微调的Wav2Vec2模型和卷积特征提取器，在多个基准测试中取得SOTA结果。属于音频反深度伪造任务。**

- **链接: [http://arxiv.org/pdf/2509.10781v1](http://arxiv.org/pdf/2509.10781v1)**

> **作者:** Xiaokang Li; Yicheng Gong; Dinghao Zou; Xin Cao; Sunbowen Lee
>
> **摘要:** Audio deepfake is so sophisticated that the lack of effective detection methods is fatal. While most detection systems primarily rely on low-level acoustic features or pretrained speech representations, they frequently neglect high-level emotional cues, which can offer complementary and potentially anti-deepfake information to enhance generalization. In this work, we propose a novel audio anti-deepfake system that utilizes emotional features (EmoAnti) by exploiting a pretrained Wav2Vec2 (W2V2) model fine-tuned on emotion recognition tasks, which derives emotion-guided representations, then designing a dedicated feature extractor based on convolutional layers with residual connections to effectively capture and refine emotional characteristics from the transformer layers outputs. Experimental results show that our proposed architecture achieves state-of-the-art performance on both the ASVspoof2019LA and ASVspoof2021LA benchmarks, and demonstrates strong generalization on the ASVspoof2021DF dataset. Our proposed approach's code is available at Anonymous GitHub1.
>
---
#### [new 016] WeaveMuse: An Open Agentic System for Multimodal Music Understanding and Generation
- **分类: cs.SD; eess.AS**

- **简介: 论文提出WeaveMuse，一个多智能体系统，用于音乐理解、符号作曲和音频合成。系统通过协调多个专业代理和管理代理，解决跨模态任务，支持可扩展部署与模型适配，旨在提升音乐信息检索工具的可访问性与可控性。**

- **链接: [http://arxiv.org/pdf/2509.11183v1](http://arxiv.org/pdf/2509.11183v1)**

> **作者:** Emmanouil Karystinaios
>
> **备注:** Accepted at Large Language Models for Music & Audio Workshop (LLM4MA) 2025
>
> **摘要:** Agentic AI has been standardized in industry as a practical paradigm for coordinating specialized models and tools to solve complex multimodal tasks. In this work, we present WeaveMuse, a multi-agent system for music understanding, symbolic composition, and audio synthesis. Each specialist agent interprets user requests, derives machine-actionable requirements (modalities, formats, constraints), and validates its own outputs, while a manager agent selects and sequences tools, mediates user interaction, and maintains state across turns. The system is extendable and deployable either locally, using quantization and inference strategies to fit diverse hardware budgets, or via the HFApi to preserve free community access to open models. Beyond out-of-the-box use, the system emphasizes controllability and adaptation through constraint schemas, structured decoding, policy-based inference, and parameter-efficient adapters or distilled variants that tailor models to MIR tasks. A central design goal is to facilitate intermodal interaction across text, symbolic notation and visualization, and audio, enabling analysis-synthesis-render loops and addressing cross-format constraints. The framework aims to democratize, implement, and make accessible MIR tools by supporting interchangeable open-source models of various sizes, flexible memory management, and reproducible deployment paths.
>
---
#### [new 017] Early Detection of Branched Broomrape (Phelipanche ramosa) Infestation in Tomato Crops Using Leaf Spectral Analysis and Machine Learning
- **分类: cs.LG; cs.AI; cs.CV; eess.SP; 68T07, 68T45, 68U10; I.5.4; I.4.6; I.2.6**

- **简介: 该论文属于植物病害早期检测任务，旨在利用叶面光谱分析与集成机器学习方法，在番茄作物中早期识别寄生性杂草Phelipanche ramosa。通过光谱数据预处理和模型集成，实现了89%的检测准确率，支持早期干预以减少产量损失。**

- **链接: [http://arxiv.org/pdf/2509.12074v1](http://arxiv.org/pdf/2509.12074v1)**

> **作者:** Mohammadreza Narimani; Alireza Pourreza; Ali Moghimi; Parastoo Farajpoor; Hamid Jafarbiglu; Mohsen B. Mesgaran
>
> **备注:** Author-accepted version. Accepted and presented at AGRICONTROL 2025 (8th IFAC Conference on Sensing, Control and Automation Technologies for Agriculture), UC Davis, USA. To appear in IFAC-PapersOnLine (Elsevier)
>
> **摘要:** Branched broomrape (Phelipanche ramosa) is a chlorophyll-deficient parasitic weed that threatens tomato production by extracting nutrients from the host. We investigate early detection using leaf-level spectral reflectance (400-2500 nm) and ensemble machine learning. In a field experiment in Woodland, California, we tracked 300 tomato plants across growth stages defined by growing degree days (GDD). Leaf reflectance was acquired with a portable spectrometer and preprocessed (band denoising, 1 nm interpolation, Savitzky-Golay smoothing, correlation-based band reduction). Clear class differences were observed near 1500 nm and 2000 nm water absorption features, consistent with reduced leaf water content in infected plants at early stages. An ensemble combining Random Forest, XGBoost, SVM with RBF kernel, and Naive Bayes achieved 89% accuracy at 585 GDD, with recalls of 0.86 (infected) and 0.93 (noninfected). Accuracy declined at later stages (e.g., 69% at 1568 GDD), likely due to senescence and weed interference. Despite the small number of infected plants and environmental confounders, results show that proximal sensing with ensemble learning enables timely detection of broomrape before canopy symptoms are visible, supporting targeted interventions and reduced yield losses.
>
---
#### [new 018] Length-Aware Rotary Position Embedding for Text-Speech Alignment
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于文本到语音合成（TTS）任务，旨在提升文本-语音对齐效果。提出长度感知旋转位置编码（LARoPE），通过相对距离计算优化对齐，提高生成质量与稳定性，优于传统RoPE方法。**

- **链接: [http://arxiv.org/pdf/2509.11084v1](http://arxiv.org/pdf/2509.11084v1)**

> **作者:** Hyeongju Kim; Juheon Lee; Jinhyeok Yang; Jacob Morton
>
> **备注:** 5 pages, 3 figures, preprint
>
> **摘要:** Many recent text-to-speech (TTS) systems are built on transformer architectures and employ cross-attention mechanisms for text-speech alignment. Within these systems, rotary position embedding (RoPE) is commonly used to encode positional information in text and speech representations. In this work, we introduce length-aware RoPE (LARoPE), a simple yet effective extension of RoPE that improves text-speech alignment. Unlike RoPE, which relies on absolute indices, LARoPE computes relative distances between query and key positions using length-normalized indices. Experimental results show that LARoPE consistently outperforms RoPE, offering faster loss convergence, more accurate text-speech alignment, and higher overall TTS quality. Furthermore, LARoPE demonstrates greater resilience to variations in utterance duration and maintains stable performance in extended speech generation up to 30 seconds, whereas RoPE suffers from notable degradation. Notably, our method achieves a state-of-the-art word error rate on a standard zero-shot TTS benchmark.
>
---
#### [new 019] MusicSwarm: Biologically Inspired Intelligence for Music Composition
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 论文提出MusicSwarm，一种受生物启发的音乐创作方法，通过去中心化群体协作生成高质量、多样化音乐。其解决传统集中式系统在创造力和结构多样性上的不足，利用局部交互规则实现全局创意结构，适用于跨领域协作任务。**

- **链接: [http://arxiv.org/pdf/2509.11973v1](http://arxiv.org/pdf/2509.11973v1)**

> **作者:** Markus J. Buehler
>
> **摘要:** We show that coherent, long-form musical composition can emerge from a decentralized swarm of identical, frozen foundation models that coordinate via stigmergic, peer-to-peer signals, without any weight updates. We compare a centralized multi-agent system with a global critic to a fully decentralized swarm in which bar-wise agents sense and deposit harmonic, rhythmic, and structural cues, adapt short-term memory, and reach consensus. Across symbolic, audio, and graph-theoretic analyses, the swarm yields superior quality while delivering greater diversity and structural variety and leads across creativity metrics. The dynamics contract toward a stable configuration of complementary roles, and self-similarity networks reveal a small-world architecture with efficient long-range connectivity and specialized bridging motifs, clarifying how local novelties consolidate into global musical form. By shifting specialization from parameter updates to interaction rules, shared memory, and dynamic consensus, MusicSwarm provides a compute- and data-efficient route to long-horizon creative structure that is immediately transferable beyond music to collaborative writing, design, and scientific discovery.
>
---
#### [new 020] Data-Driven Analysis of Text-Conditioned AI-Generated Music: A Case Study with Suno and Udio
- **分类: cs.IR; cs.AI; cs.LG; cs.SD**

- **简介: 该论文分析用户通过Suno和Udio生成的AI音乐数据，探究其使用方式与主题。通过文本嵌入、降维聚类等方法，揭示歌词主题、语言偏好及提示策略，推动AI音乐文化研究。**

- **链接: [http://arxiv.org/pdf/2509.11824v1](http://arxiv.org/pdf/2509.11824v1)**

> **作者:** Luca Casini; Laura Cros Vila; David Dalmazzo; Anna-Kaisa Kaila; Bob L. T. Sturm
>
> **备注:** Submitted for review to TISMIR Digital Musicology special issue
>
> **摘要:** Online AI platforms for creating music from text prompts (AI music), such as Suno and Udio, are now being used by hundreds of thousands of users. Some AI music is appearing in advertising, and even charting, in multiple countries. How are these platforms being used? What subjects are inspiring their users? This article answers these questions for Suno and Udio using a large collection of songs generated by users of these platforms from May to October 2024. Using a combination of state-of-the-art text embedding models, dimensionality reduction and clustering methods, we analyze the prompts, tags and lyrics, and automatically annotate and display the processed data in interactive plots. Our results reveal prominent themes in lyrics, language preference, prompting strategies, as well as peculiar attempts at steering models through the use of metatags. To promote the musicological study of the developing cultural practice of AI-generated music we share our code and resources.
>
---
#### [new 021] Sound Matching an Analogue Levelling Amplifier Using the Newton-Raphson Method
- **分类: eess.AS; cs.SD; cs.SY; eess.SY**

- **简介: 论文提出用牛顿-拉夫森方法优化参数，以数字压缩器模拟模拟均衡放大器的行为。属于音频建模任务，解决高效逼近模拟设备的问题，通过并行算法实现GPU加速训练，并开源了VST插件。**

- **链接: [http://arxiv.org/pdf/2509.10706v1](http://arxiv.org/pdf/2509.10706v1)**

> **作者:** Chin-Yun Yu; György Fazekas
>
> **备注:** Published at 2025 AES International Conference on Artificial Intelligence and Machine Learning for Audio (https://aes2.org/publications/elibrary-page/?id=22991)
>
> **摘要:** Automatic differentiation through digital signal processing algorithms for virtual analogue modelling has recently gained popularity. These algorithms are typically more computationally efficient than black-box neural networks that rely on dense matrix multiplications. Due to their differentiable nature, they can be integrated with neural networks and jointly trained using gradient descent algorithms, resulting in more efficient systems. Furthermore, signal processing algorithms have significantly fewer parameters than neural networks, allowing the application of the Newton-Raphson method. This method offers faster and more robust convergence than gradient descent at the cost of quadratic storage. This paper presents a method to emulate analogue levelling amplifiers using a feed-forward digital compressor with parameters optimised via the Newton-Raphson method. We demonstrate that a digital compressor can successfully approximate the behaviour of our target unit, the Teletronix LA-2A. Different strategies for computing the Hessian matrix are benchmarked. We leverage parallel algorithms for recursive filters to achieve efficient training on modern GPUs. The resulting model is made into a VST plugin and is open-sourced at https://github.com/aim-qmul/4a2a.
>
---
#### [new 022] Synergetic Empowerment: Wireless Communications Meets Embodied Intelligence
- **分类: cs.NI; cs.RO; cs.SY; eess.SP; eess.SY**

- **简介: 论文探讨无线通信与具身智能的协同增强，分析其共进化过程及PCE循环中的互动机制，提出未来研究方向。属于通信与智能融合领域，解决系统优化与协同能力提升问题。**

- **链接: [http://arxiv.org/pdf/2509.10481v1](http://arxiv.org/pdf/2509.10481v1)**

> **作者:** Hongtao Liang; Yihe Diao; YuHang Wu; Fuhui Zhou; Qihui Wu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Wireless communication is evolving into an agent era, where large-scale agents with inherent embodied intelligence are not just users but active participants. The perfect combination of wireless communication and embodied intelligence can achieve a synergetic empowerment and greatly facilitate the development of agent communication. An overview of this synergetic empowerment is presented, framing it as a co-evolutionary process that transforms wireless communication from a simple utility into the digital nervous system of a collective intelligence, while simultaneously elevating isolated agents into a unified superorganism with emergent capabilities far exceeding individual contributions. Moreover, we elaborate how embodied intelligence and wireless communication mutually benefit each other through the lens of the perception-cognition-execution (PCE) loop, revealing a fundamental duality where each PCE stage both challenges network capacity and creates unprecedented opportunities for system-wide optimization. Furthermore, critical open issues and future research directions are identified.
>
---
#### [new 023] Spectral Bottleneck in Deep Neural Networks: Noise is All You Need
- **分类: eess.AS; cs.CV; cs.LG; cs.SD**

- **简介: 论文研究深度神经网络中的频谱瓶颈问题，提出一种基于目标信号频谱特性的噪声初始化方法（WINNER），以提升高频率信号的重建能力。该工作属于信号重建任务，旨在解决网络难以学习高频成分的问题。**

- **链接: [http://arxiv.org/pdf/2509.09719v1](http://arxiv.org/pdf/2509.09719v1)**

> **作者:** Hemanth Chandravamsi; Dhanush V. Shenoy; Itay Zinn; Shimon Pisnoy; Steven H. Frankel
>
> **摘要:** Deep neural networks are known to exhibit a spectral learning bias, wherein low-frequency components are learned early in training, while high-frequency modes emerge more gradually in later epochs. However, when the target signal lacks low-frequency components and is dominated by broadband high frequencies, training suffers from a 'spectral bottleneck', and the model fails to reconstruct the entire signal, including the frequency components that lie within the network's representational capacity. We examine such a scenario in the context of implicit neural representations (INRs) with sinusoidal representation networks (SIRENs), focusing on the challenge of fitting high-frequency-dominant signals that are susceptible to spectral bottleneck. To effectively fit any target signal irrespective of it's frequency content, we propose a generalized target-aware 'weight perturbation scheme' (WINNER - weight initialization with noise for neural representations) for network initialization. The scheme perturbs uniformly initialized weights with Gaussian noise, where the noise scales are adaptively determined by the spectral centroid of the target signal. We show that the noise scales can provide control over the spectra of network activations and the eigenbasis of the empirical neural tangent kernel. This method not only addresses the spectral bottleneck but also yields faster convergence and with improved representation accuracy, outperforming state-of-the-art approaches in audio fitting and achieving notable gains in image fitting and denoising tasks. Beyond signal reconstruction, our approach opens new directions for adaptive weight initialization strategies in computer vision and scientific machine learning.
>
---
#### [new 024] Local Density-Based Anomaly Score Normalization for Domain Generalization
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对领域迁移下的异常声音检测任务，解决源域与目标域异常分数分布不匹配的问题，提出基于局部密度的异常分数归一化方法，提升模型在不同领域的泛化性能。**

- **链接: [http://arxiv.org/pdf/2509.10951v1](http://arxiv.org/pdf/2509.10951v1)**

> **作者:** Kevin Wilkinghoff; Haici Yang; Janek Ebbers; François G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **摘要:** State-of-the-art anomalous sound detection (ASD) systems in domain-shifted conditions rely on projecting audio signals into an embedding space and using distance-based outlier detection to compute anomaly scores. One of the major difficulties to overcome is the so-called domain mismatch between the anomaly score distributions of a source domain and a target domain that differ acoustically and in terms of the amount of training data provided. A decision threshold that is optimal for one domain may be highly sub-optimal for the other domain and vice versa. This significantly degrades the performance when only using a single decision threshold, as is required when generalizing to multiple data domains that are possibly unseen during training while still using the same trained ASD system as in the source domain. To reduce this mismatch between the domains, we propose a simple local-density-based anomaly score normalization scheme. In experiments conducted on several ASD datasets, we show that the proposed normalization scheme consistently improves performance for various types of embedding-based ASD systems and yields better results than existing anomaly score normalization approaches.
>
---
## 更新

#### [replaced 001] Survey on the Evaluation of Generative Models in Music
- **分类: cs.SD; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05104v3](http://arxiv.org/pdf/2506.05104v3)**

> **作者:** Alexander Lerch; Claire Arthur; Nick Bryan-Kinns; Corey Ford; Qianyi Sun; Ashvala Vinay
>
> **备注:** Accepted paper submitted to ACM CSUR on 12-Sep-2025, original manuscript submitted on 26-Jun-2024
>
> **摘要:** Research on generative systems in music has seen considerable attention and growth in recent years. A variety of attempts have been made to systematically evaluate such systems. We present an interdisciplinary review of the common evaluation targets, methodologies, and metrics for the evaluation of both system output and model use, covering subjective and objective approaches, qualitative and quantitative approaches, as well as empirical and computational methods. We examine the benefits and limitations of these approaches from a musicological, an engineering, and an HCI perspective.
>
---
#### [replaced 002] SonicSieve: Bringing Directional Speech Extraction to Smartphones Using Acoustic Microstructures
- **分类: cs.SD; cs.HC; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.10793v2](http://arxiv.org/pdf/2504.10793v2)**

> **作者:** Kuang Yuan; Yifeng Wang; Xiyuxing Zhang; Chengyi Shen; Swarun Kumar; Justin Chan
>
> **摘要:** Imagine placing your smartphone on a table in a noisy restaurant and clearly capturing the voices of friends seated around you, or recording a lecturer's voice with clarity in a reverberant auditorium. We introduce SonicSieve, the first intelligent directional speech extraction system for smartphones using a bio-inspired acoustic microstructure. Our passive design embeds directional cues onto incoming speech without any additional electronics. It attaches to the in-line mic of low-cost wired earphones which can be attached to smartphones. We present an end-to-end neural network that processes the raw audio mixtures in real-time on mobile devices. Our results show that SonicSieve achieves a signal quality improvement of 5.0 dB when focusing on a 30{\deg} angular region. Additionally, the performance of our system based on only two microphones exceeds that of conventional 5-microphone arrays.
>
---
#### [replaced 003] Intramuscular microelectrode arrays enable highly-accurate neural decoding of hand movements
- **分类: q-bio.NC; cs.HC; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2410.11016v2](http://arxiv.org/pdf/2410.11016v2)**

> **作者:** Agnese Grison; Jaime Ibanez Pereda; Silvia Muceli; Aritra Kundu; Farah Baracat; Giacomo Indiveri; Elisa Donati; Dario Farina
>
> **摘要:** Decoding the activity of the nervous system is a critical challenge in neuroscience and neural interfacing. In this study, we present a neuromuscular recording system that enables large-scale sampling of muscle activity using microelectrode arrays with over 100 channels embedded in forearm muscles. These arrays captured intramuscular high-density signals that were decoded into patterns of activation of spinal motoneurons. In two healthy participants, we recorded high-density intramuscular activity during single- and multi-digit contractions, revealing distinct motoneuron recruitment patterns specific to each task. Based on these patterns, we achieved perfect classification accuracy (100%) for 12 single- and multi-digit tasks and over 96% accuracy for up to 16 tasks, significantly outperforming state-of-the-art EMG classification methods. This intramuscular high-density system and classification method represent an advancement in neural interfacing, with the potential to improve human-computer interaction and the control of assistive technologies, particularly for replacing or restoring impaired motor function.
>
---
#### [replaced 004] Single-shot HDR using conventional image sensor shutter functions and optical randomization
- **分类: eess.IV; cs.CV; cs.GR; eess.SP; physics.optics**

- **链接: [http://arxiv.org/pdf/2506.22426v2](http://arxiv.org/pdf/2506.22426v2)**

> **作者:** Xiang Dai; Kyrollos Yanny; Kristina Monakhova; Nicholas Antipa
>
> **备注:** Published in ACM Transactions on Graphics (TOG), Volume 44, Issue 5, October 2025. DOI: 10.1145/3748718
>
> **摘要:** High-dynamic-range (HDR) imaging is an essential technique for overcoming the dynamic range limits of image sensors. The classic method relies on multiple exposures, which slows capture time, resulting in motion artifacts when imaging dynamic scenes. Single-shot HDR imaging alleviates this issue by encoding HDR data into a single exposure, then computationally recovering it. Many established methods use strong image priors to recover improperly exposed image detail. These approaches struggle with extended highlight regions. We utilize the global reset release (GRR) shutter mode of an off-the-shelf sensor. GRR shutter mode applies a longer exposure time to rows closer to the bottom of the sensor. We use optics that relay a randomly permuted (shuffled) image onto the sensor, effectively creating spatially randomized exposures across the scene. The exposure diversity allows us to recover HDR data by solving an optimization problem with a simple total variation image prior. In simulation, we demonstrate that our method outperforms other single-shot methods when many sensor pixels are saturated (10% or more), and is competitive at a modest saturation (1%). Finally, we demonstrate a physical lab prototype that uses an off-the-shelf random fiber bundle for the optical shuffling. The fiber bundle is coupled to a low-cost commercial sensor operating in GRR shutter mode. Our prototype achieves a dynamic range of up to 73dB using an 8-bit sensor with 48dB dynamic range.
>
---
#### [replaced 005] Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection against Voice Deepfakes
- **分类: cs.SD; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.12932v2](http://arxiv.org/pdf/2507.12932v2)**

> **作者:** Zhou Feng; Jiahao Chen; Chunyi Zhou; Yuwen Pu; Qingming Li; Tianyu Du; Shouling Ji
>
> **备注:** Accepted by ACM MM 2025, Open-sourced
>
> **摘要:** The rapid advancement of voice deepfake technologies has raised serious concerns about user audio privacy, as attackers increasingly exploit publicly available voice data to generate convincing fake audio for malicious purposes such as identity theft, financial fraud, and misinformation campaigns. While existing defense methods offer partial protection, they face critical limitations, including weak adaptability to unseen user data, poor scalability to long audio, rigid reliance on white-box knowledge, and high computational and temporal costs during the encryption process. To address these challenges and defend against personalized voice deepfake threats, we propose Enkidu, a novel user-oriented privacy-preserving framework that leverages universal frequential perturbations generated through black-box knowledge and few-shot training on a small amount of user data. These highly malleable frequency-domain noise patches enable real-time, lightweight protection with strong generalization across variable-length audio and robust resistance to voice deepfake attacks, all while preserving perceptual quality and speech intelligibility. Notably, Enkidu achieves over 50 to 200 times processing memory efficiency (as low as 0.004 gigabytes) and 3 to 7000 times runtime efficiency (real-time coefficient as low as 0.004) compared to six state-of-the-art countermeasures. Extensive experiments across six mainstream text-to-speech models and five cutting-edge automated speaker verification models demonstrate the effectiveness, transferability, and practicality of Enkidu in defending against both vanilla and adaptive voice deepfake attacks. Our code is currently available.
>
---
#### [replaced 006] Progressive Facial Granularity Aggregation with Bilateral Attribute-based Enhancement for Face-to-Speech Synthesis
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.07376v2](http://arxiv.org/pdf/2509.07376v2)**

> **作者:** Yejin Jeon; Youngjae Kim; Jihyun Lee; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** EMNLP Findings
>
> **摘要:** For individuals who have experienced traumatic events such as strokes, speech may no longer be a viable means of communication. While text-to-speech (TTS) can be used as a communication aid since it generates synthetic speech, it fails to preserve the user's own voice. As such, face-to-voice (FTV) synthesis, which derives corresponding voices from facial images, provides a promising alternative. However, existing methods rely on pre-trained visual encoders, and finetune them to align with speech embeddings, which strips fine-grained information from facial inputs such as gender or ethnicity, despite their known correlation with vocal traits. Moreover, these pipelines are multi-stage, which requires separate training of multiple components, thus leading to training inefficiency. To address these limitations, we utilize fine-grained facial attribute modeling by decomposing facial images into non-overlapping segments and progressively integrating them into a multi-granular representation. This representation is further refined through multi-task learning of speaker attributes such as gender and ethnicity at both the visual and acoustic domains. Moreover, to improve alignment robustness, we adopt a multi-view training strategy by pairing various visual perspectives of a speaker in terms of different angles and lighting conditions, with identical speech recordings. Extensive subjective and objective evaluations confirm that our approach substantially enhances face-voice congruence and synthesis stability.
>
---
#### [replaced 007] CoPlay: Audio-agnostic Cognitive Scaling for Acoustic Sensing
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2403.10796v3](http://arxiv.org/pdf/2403.10796v3)**

> **作者:** Yin Li; Bo Liu; Rajalakshmi Nanadakumar
>
> **备注:** ICCCN'25
>
> **摘要:** Acoustic sensing manifests great potential in various applications that encompass health monitoring, gesture interface and imaging by leveraging the speakers and microphones on smart devices. However, in ongoing research and development in acoustic sensing, one problem is often overlooked: the same speaker, when used concurrently for sensing and other traditional applications (like playing music), could cause interference in both making it impractical to use in the real world. The strong ultrasonic sensing signals mixed with music would overload the speaker's mixer. To confront this issue of overloaded signals, current solutions are clipping or down-scaling, both of which affect the music playback quality and also sensing range and accuracy. To address this challenge, we propose CoPlay, a deep learning based optimization algorithm to cognitively adapt the sensing signal. It can 1) maximize the sensing signal magnitude within the available bandwidth left by the concurrent music to optimize sensing range and accuracy and 2) minimize any consequential frequency distortion that can affect music playback. In this work, we design a deep learning model and test it on common types of sensing signals (sine wave or Frequency Modulated Continuous Wave FMCW) as inputs with various agnostic concurrent music and speech. First, we evaluated the model performance to show the quality of the generated signals. Then we conducted field studies of downstream acoustic sensing tasks in the real world. A study with 12 users proved that respiration monitoring and gesture recognition using our adapted signal achieve similar accuracy as no-concurrent-music scenarios, while clipping or down-scaling manifests worse accuracy. A qualitative study also manifests that the music play quality is not degraded, unlike traditional clipping or down-scaling methods.
>
---
#### [replaced 008] Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.07756v2](http://arxiv.org/pdf/2509.07756v2)**

> **作者:** Friedrich Wolf-Monheim
>
> **摘要:** Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs.
>
---
#### [replaced 009] Evaluating Automatic Speech Recognition Systems for Korean Meteorological Experts
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.18444v3](http://arxiv.org/pdf/2410.18444v3)**

> **作者:** ChaeHun Park; Hojun Cho; Jaegul Choo
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** This paper explores integrating Automatic Speech Recognition (ASR) into natural language query systems to improve weather forecasting efficiency for Korean meteorologists. We address challenges in developing ASR systems for the Korean weather domain, specifically specialized vocabulary and Korean linguistic intricacies. To tackle these issues, we constructed an evaluation dataset of spoken queries recorded by native Korean speakers. Using this dataset, we assessed various configurations of a multilingual ASR model family, identifying performance limitations related to domain-specific terminology. We then implemented a simple text-to-speech-based data augmentation method, which improved the recognition of specialized terms while maintaining general-domain performance. Our contributions include creating a domain-specific dataset, comprehensive ASR model evaluations, and an effective augmentation technique. We believe our work provides a foundation for future advancements in ASR for the Korean weather forecasting domain.
>
---
#### [replaced 010] YuE: Scaling Open Foundation Models for Long-Form Music Generation
- **分类: eess.AS; cs.AI; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.08638v2](http://arxiv.org/pdf/2503.08638v2)**

> **作者:** Ruibin Yuan; Hanfeng Lin; Shuyue Guo; Ge Zhang; Jiahao Pan; Yongyi Zang; Haohe Liu; Yiming Liang; Wenye Ma; Xingjian Du; Xinrun Du; Zhen Ye; Tianyu Zheng; Zhengxuan Jiang; Yinghao Ma; Minghao Liu; Zeyue Tian; Ziya Zhou; Liumeng Xue; Xingwei Qu; Yizhi Li; Shangda Wu; Tianhao Shen; Ziyang Ma; Jun Zhan; Chunhui Wang; Yatian Wang; Xiaowei Chi; Xinyue Zhang; Zhenzhu Yang; Xiangzhou Wang; Shansong Liu; Lingrui Mei; Peng Li; Junjie Wang; Jianwei Yu; Guojian Pang; Xu Li; Zihao Wang; Xiaohuan Zhou; Lijun Yu; Emmanouil Benetos; Yong Chen; Chenghua Lin; Xie Chen; Gus Xia; Zhaoxiang Zhang; Chao Zhang; Wenhu Chen; Xinyu Zhou; Xipeng Qiu; Roger Dannenberg; Jiaheng Liu; Jian Yang; Wenhao Huang; Wei Xue; Xu Tan; Yike Guo
>
> **备注:** https://github.com/multimodal-art-projection/YuE
>
> **摘要:** We tackle the task of long-form music generation--particularly the challenging \textbf{lyrics-to-song} problem--by introducing YuE, a family of open foundation models based on the LLaMA2 architecture. Specifically, YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through (1) track-decoupled next-token prediction to overcome dense mixture signals, (2) structural progressive conditioning for long-context lyrical alignment, and (3) a multitask, multiphase pre-training recipe to converge and generalize. In addition, we redesign the in-context learning technique for music generation, enabling versatile style transfer (e.g., converting Japanese city pop into an English rap while preserving the original accompaniment) and bidirectional generation. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility. In addition, fine-tuning YuE enables additional controls and enhanced support for tail languages. Furthermore, beyond generation, we show that YuE's learned representations can perform well on music understanding tasks, where the results of YuE match or exceed state-of-the-art methods on the MARBLE benchmark. Keywords: lyrics2song, song generation, long-form, foundation model, music generation
>
---
