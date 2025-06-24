# 音频 cs.SD;  eess.SP

- **最新发布 27 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于符号音乐生成任务，旨在解决作曲家风格数据稀缺问题。通过两阶段训练，利用大规模预训练提升小样本作曲家风格建模效果。**

- **链接: [http://arxiv.org/pdf/2506.17497v1](http://arxiv.org/pdf/2506.17497v1)**

> **作者:** Mingyang Yao; Ke Chen
>
> **备注:** Proceedings of the 6th Conference on AI Music Creativity, AIMC 2025
>
> **摘要:** Despite progress in controllable symbolic music generation, data scarcity remains a challenge for certain control modalities. Composer-style music generation is a prime example, as only a few pieces per composer are available, limiting the modeling of both styles and fundamental music elements (e.g., melody, chord, rhythm). In this paper, we investigate how general music knowledge learned from a broad corpus can enhance the mastery of specific composer styles, with a focus on piano piece generation. Our approach follows a two-stage training paradigm. First, we pre-train a REMI-based music generation model on a large corpus of pop, folk, and classical music. Then, we fine-tune it on a small, human-verified dataset from four renowned composers, namely Bach, Mozart, Beethoven, and Chopin, using a lightweight adapter module to condition the model on style indicators. To evaluate the effectiveness of our approach, we conduct both objective and subjective evaluations on style accuracy and musicality. Experimental results demonstrate that our method outperforms ablations and baselines, achieving more precise composer-style modeling and better musical aesthetics. Additionally, we provide observations on how the model builds music concepts from the generality pre-training and refines its stylistic understanding through the mastery fine-tuning.
>
---
#### [new 002] CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于跨文化音乐表示学习任务，旨在提升音乐基础模型在不同音乐传统中的表现。通过多阶段持续预训练和任务算术方法，增强了模型的跨文化适应能力。**

- **链接: [http://arxiv.org/pdf/2506.17818v1](http://arxiv.org/pdf/2506.17818v1)**

> **作者:** Angelos-Nikolaos Kanatas; Charilaos Papaioannou; Alexandros Potamianos
>
> **备注:** 10 pages, 4 figures, accepted to the 26th International Society for Music Information Retrieval conference (ISMIR 2025), to be held in Daejeon, South Korea
>
> **摘要:** Recent advances in music foundation models have improved audio representation learning, yet their effectiveness across diverse musical traditions remains limited. We introduce CultureMERT-95M, a multi-culturally adapted foundation model developed to enhance cross-cultural music representation learning and understanding. To achieve this, we propose a two-stage continual pre-training strategy that integrates learning rate re-warming and re-decaying, enabling stable adaptation even with limited computational resources. Training on a 650-hour multi-cultural data mix, comprising Greek, Turkish, and Indian music traditions, results in an average improvement of 4.9% in ROC-AUC and AP across diverse non-Western music auto-tagging tasks, surpassing prior state-of-the-art, with minimal forgetting on Western-centric benchmarks. We further investigate task arithmetic, an alternative approach to multi-cultural adaptation that merges single-culture adapted models in the weight space. Task arithmetic performs on par with our multi-culturally trained model on non-Western auto-tagging tasks and shows no regression on Western datasets. Cross-cultural evaluation reveals that single-culture models transfer with varying effectiveness across musical traditions, whereas the multi-culturally adapted model achieves the best overall performance. To support research on world music representation learning, we publicly release CultureMERT-95M and CultureMERT-TA-95M, fostering the development of more culturally aware music foundation models.
>
---
#### [new 003] GD-Retriever: Controllable Generative Text-Music Retrieval with Diffusion Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本-音乐检索任务，旨在解决传统方法在控制性和灵活性上的不足。通过引入生成扩散模型，实现可调控的检索系统。**

- **链接: [http://arxiv.org/pdf/2506.17886v1](http://arxiv.org/pdf/2506.17886v1)**

> **作者:** Julien Guinot; Elio Quinton; György Fazekas
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Multimodal contrastive models have achieved strong performance in text-audio retrieval and zero-shot settings, but improving joint embedding spaces remains an active research area. Less attention has been given to making these systems controllable and interactive for users. In text-music retrieval, the ambiguity of freeform language creates a many-to-many mapping, often resulting in inflexible or unsatisfying results. We introduce Generative Diffusion Retriever (GDR), a novel framework that leverages diffusion models to generate queries in a retrieval-optimized latent space. This enables controllability through generative tools such as negative prompting and denoising diffusion implicit models (DDIM) inversion, opening a new direction in retrieval control. GDR improves retrieval performance over contrastive teacher models and supports retrieval in audio-only latent spaces using non-jointly trained encoders. Finally, we demonstrate that GDR enables effective post-hoc manipulation of retrieval behavior, enhancing interactive control for text-music retrieval tasks.
>
---
#### [new 004] SLAP: Siamese Language-Audio Pretraining Without Negative Samples for Music Understanding
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SLAP框架，解决音乐理解中的多模态嵌入问题，无需负样本，提升模型性能与训练效率。**

- **链接: [http://arxiv.org/pdf/2506.17815v1](http://arxiv.org/pdf/2506.17815v1)**

> **作者:** Julien Guinot; Alain Riou; Elio Quinton; György Fazekas
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Joint embedding spaces have significantly advanced music understanding and generation by linking text and audio through multimodal contrastive learning. However, these approaches face large memory requirement limitations due to relying on large batch sizes to effectively utilize negative samples. Further, multimodal joint embedding spaces suffer from a modality gap wherein embeddings from different modalities lie in different manifolds of the embedding space. To address these challenges, we propose Siamese Language-Audio Pretraining (SLAP), a novel multimodal pretraining framework that allows learning powerful representations without negative samples. SLAP adapts the Bootstrap Your Own Latent (BYOL) paradigm for multimodal audio-text training, promoting scalability in training multimodal embedding spaces. We illustrate the ability of our model to learn meaningful relationships between music and text -- specifically, we show that SLAP outperforms CLAP on tasks such as text-music retrieval and zero-shot classification. We also observe competitive downstream performance on several MIR tasks, including with larger or supervised models (genre and instrument classification, auto-tagging). Additionally, our approach has attractive properties, such as a quantifiably reduced modality gap and improved robustness to batch size variations on retrieval performance. Finally, its novel formulation unlocks large-scale training on a single GPU through gradient accumulation.
>
---
#### [new 005] MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到音乐生成任务，旨在通过轻量级机制提升音乐生成的可控性。工作包括引入位置嵌入以增强时间相关条件控制，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.18729v1](http://arxiv.org/pdf/2506.18729v1)**

> **作者:** Fang-Duo Tsai; Shih-Lun Wu; Weijaw Lee; Sheng-Ping Yang; Bo-Rui Chen; Hao-Chung Cheng; Yi-Hsuan Yang
>
> **备注:** Accepted by the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** We propose MuseControlLite, a lightweight mechanism designed to fine-tune text-to-music generation models for precise conditioning using various time-varying musical attributes and reference audio signals. The key finding is that positional embeddings, which have been seldom used by text-to-music generation models in the conditioner for text conditions, are critical when the condition of interest is a function of time. Using melody control as an example, our experiments show that simply adding rotary positional embeddings to the decoupled cross-attention layers increases control accuracy from 56.6% to 61.1%, while requiring 6.75 times fewer trainable parameters than state-of-the-art fine-tuning mechanisms, using the same pre-trained diffusion Transformer model of Stable Audio Open. We evaluate various forms of musical attribute control, audio inpainting, and audio outpainting, demonstrating improved controllability over MusicGen-Large and Stable Audio Open ControlNet at a significantly lower fine-tuning cost, with only 85M trainble parameters. Source code, model checkpoints, and demo examples are available at: https: //MuseControlLite.github.io/web/.
>
---
#### [new 006] USAD: Universal Speech and Audio Representation via Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在解决领域特定模型无法统一处理语音与音频的问题。通过知识蒸馏方法，USAD融合多种音频类型，构建统一模型。**

- **链接: [http://arxiv.org/pdf/2506.18843v1](http://arxiv.org/pdf/2506.18843v1)**

> **作者:** Heng-Jui Chang; Saurabhchand Bhati; James Glass; Alexander H. Liu
>
> **备注:** Preprint
>
> **摘要:** Self-supervised learning (SSL) has revolutionized audio representations, yet models often remain domain-specific, focusing on either speech or non-speech tasks. In this work, we present Universal Speech and Audio Distillation (USAD), a unified approach to audio representation learning that integrates diverse audio types - speech, sound, and music - into a single model. USAD employs efficient layer-to-layer distillation from domain-specific SSL models to train a student on a comprehensive audio dataset. USAD offers competitive performance across various benchmarks and datasets, including frame and instance-level speech processing tasks, audio tagging, and sound classification, achieving near state-of-the-art results with a single encoder on SUPERB and HEAR benchmarks.
>
---
#### [new 007] Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决传统损失函数影响语音可懂性的问题。通过设计频率加权损失函数，提升语音细节保留效果。**

- **链接: [http://arxiv.org/pdf/2506.18714v1](http://arxiv.org/pdf/2506.18714v1)**

> **作者:** Nasser-Eddine Monir; Paul Magron; Romain Serizel
>
> **备注:** This is the preprint of the paper submitted to the 26th IEEE International Workshop on Multimedia Signal Processing (MMSP)
>
> **摘要:** Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues.
>
---
#### [new 008] Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决性别和音素差异对算法性能影响的问题。通过分析音素级特征，发现女性语音在特定音素上表现更优。**

- **链接: [http://arxiv.org/pdf/2506.18691v1](http://arxiv.org/pdf/2506.18691v1)**

> **作者:** Nasser-Eddine Monir; Paul Magron; Romain Serizel
>
> **摘要:** Multichannel speech enhancement algorithms are essential for improving the intelligibility of speech signals in noisy environments. These algorithms are usually evaluated at the utterance level, but this approach overlooks the disparities in acoustic characteristics that are observed in different phoneme categories and between male and female speakers. In this paper, we investigate the impact of gender and phonetic content on speech enhancement algorithms. We motivate this approach by outlining phoneme- and gender-specific spectral features. Our experiments reveal that while utterance-level differences between genders are minimal, significant variations emerge at the phoneme level. Results show that the tested algorithms better reduce interference with fewer artifacts on female speech, particularly in plosives, fricatives, and vowels. Additionally, they demonstrate greater performance for female speech in terms of perceptual and speech recognition metrics.
>
---
#### [new 009] Selecting N-lowest scores for training MOS prediction models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决MOS预测模型训练中的主观评分偏差问题。通过引入N_low-MOS提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18326v1](http://arxiv.org/pdf/2506.18326v1)**

> **作者:** Yuto Kondo; Hirokazu Kameoka; Kou Tanaka; Takuhiro Kaneko
>
> **备注:** Accepted on ICASSP 2024
>
> **摘要:** The automatic speech quality assessment (SQA) has been extensively studied to predict the speech quality without time-consuming questionnaires. Recently, neural-based SQA models have been actively developed for speech samples produced by text-to-speech or voice conversion, with a primary focus on training mean opinion score (MOS) prediction models. The quality of each speech sample may not be consistent across the entire duration, and it remains unclear which segments of the speech receive the primary focus from humans when assigning subjective evaluation for MOS calculation. We hypothesize that when humans rate speech, they tend to assign more weight to low-quality speech segments, and the variance in ratings for each sample is mainly due to accidental assignment of higher scores when overlooking the poor quality speech segments. Motivated by the hypothesis, we analyze the VCC2018 and BVCC datasets. Based on the hypothesis, we propose the more reliable representative value N_low-MOS, the mean of the $N$-lowest opinion scores. Our experiments show that LCC and SRCC improve compared to regular MOS when employing N_low-MOS to MOSNet training. This result suggests that N_low-MOS is a more intrinsic representative value of subjective speech quality and makes MOSNet a better comparator of VC models.
>
---
#### [new 010] Human Voice is Unique
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于生物特征识别任务，旨在验证人声是否为唯一标识。通过统计方法分析语音特征，证明人声具有高度独特性，为语音应用提供理论依据。**

- **链接: [http://arxiv.org/pdf/2506.18182v1](http://arxiv.org/pdf/2506.18182v1)**

> **作者:** Rita Singh; Bhiksha Raj
>
> **备注:** 15 pages, 1 figure, 2 tables
>
> **摘要:** Voice is increasingly being used as a biometric entity in many applications. These range from speaker identification and verification systems to human profiling technologies that attempt to estimate myriad aspects of the speaker's persona from their voice. However, for an entity to be a true biometric identifier, it must be unique. This paper establishes a first framework for calculating the uniqueness of human voice objectively. The approach in this paper is based on statistical considerations that take into account a set of measurable characteristics of the voice signal that bear a causal relationship to the vocal production process, but are not inter-dependent or derivable from each other. Depending on how we quantize these variables, we show that the chances of two people having the same voice in a world populated by 10 billion people range from one in a few thousand, to one in a septillion or less. The paper also discusses the implications of these calculations on the choices made in voice processing applications.
>
---
#### [new 011] JIS: A Speech Corpus of Japanese Idol Speakers with Various Speaking Styles
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍了一个名为JIS的日本偶像语音语料库，用于提升语音生成AI的研究，解决TTS和VC中说话人相似性评估问题。**

- **链接: [http://arxiv.org/pdf/2506.18296v1](http://arxiv.org/pdf/2506.18296v1)**

> **作者:** Yuto Kondo; Hirokazu Kameoka; Kou Tanaka; Takuhiro Kaneko
>
> **备注:** Accepted on Interspeech 2025
>
> **摘要:** We construct Japanese Idol Speech Corpus (JIS) to advance research in speech generation AI, including text-to-speech synthesis (TTS) and voice conversion (VC). JIS will facilitate more rigorous evaluations of speaker similarity in TTS and VC systems since all speakers in JIS belong to a highly specific category: "young female live idols" in Japan, and each speaker is identified by a stage name, enabling researchers to recruit listeners familiar with these idols for listening experiments. With its unique speaker attributes, JIS will foster compelling research, including generating voices tailored to listener preferences-an area not yet widely studied. JIS will be distributed free of charge to promote research in speech generation AI, with usage restricted to non-commercial, basic research. We describe the construction of JIS, provide an overview of Japanese live idol culture to support effective and ethical use of JIS, and offer a basic analysis to guide application of JIS.
>
---
#### [new 012] Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决不流畅语音的准确转录问题。通过结合音频和文本输入，利用大语言模型生成带时间戳的详细转录文本。**

- **链接: [http://arxiv.org/pdf/2506.18510v1](http://arxiv.org/pdf/2506.18510v1)**

> **作者:** Duygu Altinok
>
> **备注:** Accepted to INTERSPEECH2025 workshop DISS2025
>
> **摘要:** Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints.
>
---
#### [new 013] Large-Scale Training Data Attribution for Music Generative Models via Unlearning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐生成模型的训练数据归属任务，旨在解决AI生成音乐中原创者权益识别问题。通过无监督学习方法实现数据溯源，提升AI创作的伦理与责任性。**

- **链接: [http://arxiv.org/pdf/2506.18312v1](http://arxiv.org/pdf/2506.18312v1)**

> **作者:** Woosung Choi; Junghyun Koo; Kin Wai Cheuk; Joan Serrà; Marco A. Martínez-Ramírez; Yukara Ikemiya; Naoki Murata; Yuhta Takida; Wei-Hsiang Liao; Yuki Mitsufuji
>
> **备注:** accepted at ICML 2025 Workshop on Machine Learning for Audio
>
> **摘要:** This paper explores the use of unlearning methods for training data attribution (TDA) in music generative models trained on large-scale datasets. TDA aims to identify which specific training data points contributed to the generation of a particular output from a specific model. This is crucial in the context of AI-generated music, where proper recognition and credit for original artists are generally overlooked. By enabling white-box attribution, our work supports a fairer system for acknowledging artistic contributions and addresses pressing concerns related to AI ethics and copyright. We apply unlearning-based attribution to a text-to-music diffusion model trained on a large-scale dataset and investigate its feasibility and behavior in this setting. To validate the method, we perform a grid search over different hyperparameter configurations and quantitatively evaluate the consistency of the unlearning approach. We then compare attribution patterns from unlearning with those from a similarity-based approach. Our findings suggest that unlearning-based approaches can be effectively adapted to music generative models, introducing large-scale TDA to this domain and paving the way for more ethical and accountable AI systems for music creation.
>
---
#### [new 014] AI-Generated Song Detection via Lyrics Transcripts
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于AI生成音乐检测任务，旨在解决真实场景下缺乏完美歌词的问题。通过ASR模型转录歌词并使用检测器进行识别，提升了检测效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.18488v1](http://arxiv.org/pdf/2506.18488v1)**

> **作者:** Markus Frohmann; Elena V. Epure; Gabriel Meseguer-Brocal; Markus Schedl; Romain Hennequin
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [new 015] Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于水下声源定位任务，旨在解决复杂环境下的声源精确定位问题。通过多分支网络结合CNN与Conformer，提升定位准确性。**

- **链接: [http://arxiv.org/pdf/2506.17409v1](http://arxiv.org/pdf/2506.17409v1)**

> **作者:** Quoc Thinh Vo; Joe Woods; Priontu Chowdhury; David K. Han
>
> **备注:** This paper has been accepted for the 33rd European Signal Processing Conference (EUSIPCO) 2025 in Palermo, Italy
>
> **摘要:** Localizing acoustic sound sources in the ocean is a challenging task due to the complex and dynamic nature of the environment. Factors such as high background noise, irregular underwater geometries, and varying acoustic properties make accurate localization difficult. To address these obstacles, we propose a multi-branch network architecture designed to accurately predict the distance between a moving acoustic source and a receiver, tested on real-world underwater signal arrays. The network leverages Convolutional Neural Networks (CNNs) for robust spatial feature extraction and integrates Conformers with self-attention mechanism to effectively capture temporal dependencies. Log-mel spectrogram and generalized cross-correlation with phase transform (GCC-PHAT) features are employed as input representations. To further enhance the model performance, we introduce an Adaptive Gain Control (AGC) layer, that adaptively adjusts the amplitude of input features, ensuring consistent energy levels across varying ranges, signal strengths, and noise conditions. We assess the model's generalization capability by training it in one domain and testing it in a different domain, using only a limited amount of data from the test domain for fine-tuning. Our proposed method outperforms state-of-the-art (SOTA) approaches in similar settings, establishing new benchmarks for underwater sound localization.
>
---
#### [new 016] Rethinking Mean Opinion Scores in Speech Quality Assessment: Aggregation through Quantized Distribution Fitting
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在提升MOS预测性能。针对传统评分限制，提出基于量化分布拟合的新评分方法，提高预测准确性。**

- **链接: [http://arxiv.org/pdf/2506.18307v1](http://arxiv.org/pdf/2506.18307v1)**

> **作者:** Yuto Kondo; Hirokazu Kameoka; Kou Tanaka; Takuhiro Kaneko
>
> **备注:** Accepted on ICASSP 2025
>
> **摘要:** Speech quality assessment (SQA) aims to evaluate the quality of speech samples without relying on time-consuming listener questionnaires. Recent efforts have focused on training neural-based SQA models to predict the mean opinion score (MOS) of speech samples produced by text-to-speech or voice conversion systems. This paper targets the enhancement of MOS prediction models' performance. We propose a novel score aggregation method to address the limitations of conventional annotations for MOS, which typically involve ratings on a scale from 1 to 5. Our method is based on the hypothesis that annotators internally consider continuous scores and then choose the nearest discrete rating. By modeling this process, we approximate the generative distribution of ratings by quantizing the latent continuous distribution. We then use the peak of this latent distribution, estimated through the loss between the quantized distribution and annotated ratings, as a new representative value instead of MOS. Experimental results demonstrate that substituting MOSNet's predicted target with this proposed value improves prediction performance.
>
---
#### [new 017] Algebraic Structures in Microtonal Music
- **分类: cs.SD; eess.AS; math.HO; 20-01 (Primary), 00A08 (secondary)**

- **简介: 该论文属于音乐理论与数学交叉研究，探讨24音微分音体系中的代数结构，分析其音乐和声结构如何用群论解释。**

- **链接: [http://arxiv.org/pdf/2506.17778v1](http://arxiv.org/pdf/2506.17778v1)**

> **作者:** Veronica Flynn; Carmen Rovi
>
> **备注:** 17 pages, 12 figures. The content should be accessible for students in a first course of Abstract Algebra. A musical background is not necessary. Comments welcome!
>
> **摘要:** We will discuss how certain group theory structures are found in music theory. Western music splits the octave into 12 equal tones called half-steps. We can take this division further and split the octave into 24 equal tones by splitting each half-step in two, called a quarter-step. By assigning each of these 24 notes a number, we can discuss musical actions mathematically. In this paper, we analyze 24-tone microtonal music and explore how musical and harmonic structures in this system can be interpreted in terms of group-theoretic structures. This work extends the study by Crans, Fiore, and Satyendra.
>
---
#### [new 018] Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于认知障碍检测任务，旨在无需标注数据的情况下通过语音识别认知障碍。工作是利用AudioLLM模型实现零样本检测，并验证其跨语言和任务的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.17351v1](http://arxiv.org/pdf/2506.17351v1)**

> **作者:** Mostafa Shahin; Beena Ahmed; Julien Epps
>
> **摘要:** Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets.
>
---
#### [new 019] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文属于音乐驱动的群体舞蹈生成任务，解决多舞者碰撞、单人脚滑和长序列 abrupt 位移问题，提出 TCDiff++ 模型优化轨迹控制。**

- **链接: [http://arxiv.org/pdf/2506.18671v1](http://arxiv.org/pdf/2506.18671v1)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to better maintain the relative positioning among dancers. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
#### [new 020] Two Sonification Methods for the MindCube
- **分类: cs.HC; cs.AI; cs.SD; eess.AS; H.5.5**

- **简介: 该论文研究如何将MindCube用作音乐接口，探索其在情绪调节中的应用。任务是情感交互设计，解决如何通过设备控制音乐的问题，提出两种映射方法。**

- **链接: [http://arxiv.org/pdf/2506.18196v1](http://arxiv.org/pdf/2506.18196v1)**

> **作者:** Fangzheng Liu; Lancelot Blanchard; Don D. Haddad; Joseph A. Paradiso
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** In this work, we explore the musical interface potential of the MindCube, an interactive device designed to study emotions. Embedding diverse sensors and input devices, this interface resembles a fidget cube toy commonly used to help users relieve their stress and anxiety. As such, it is a particularly well-suited controller for musical systems that aim to help with emotion regulation. In this regard, we present two different mappings for the MindCube, with and without AI. With our generative AI mapping, we propose a way to infuse meaning within a latent space and techniques to navigate through it with an external controller. We discuss our results and propose directions for future work.
>
---
#### [new 021] Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别中的关键词检测任务，旨在提升少样本条件下的检测准确率。通过自监督学习和知识蒸馏方法优化模型性能。**

- **链接: [http://arxiv.org/pdf/2506.17686v1](http://arxiv.org/pdf/2506.17686v1)**

> **作者:** Alican Gok; Oguzhan Buyuksolak; Osman Erman Okman; Murat Saraclar
>
> **备注:** To be submitted to IEEE Signal Processing Letters, 5 pages, 3 figures
>
> **摘要:** Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
>
---
#### [new 022] Blind Source Separation in Biomedical Signals Using Variational Methods
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于盲源分离任务，旨在解决心肺音混合信号的自动分离问题。通过变分自编码器学习潜在空间并重构原始信号，无需标签数据。**

- **链接: [http://arxiv.org/pdf/2506.18281v1](http://arxiv.org/pdf/2506.18281v1)**

> **作者:** Yasaman Torabi; Shahram Shirani; James P. Reilly
>
> **备注:** Presented at Southern Ontario Numerical Analysis Day (SONAD'25), Contributed Talk 03
>
> **摘要:** This study introduces a novel unsupervised approach for separating overlapping heart and lung sounds using variational autoencoders (VAEs). In clinical settings, these sounds often interfere with each other, making manual separation difficult and error-prone. The proposed model learns to encode mixed signals into a structured latent space and reconstructs the individual components using a probabilistic decoder, all without requiring labeled data or prior knowledge of source characteristics. We apply this method to real recordings obtained from a clinical manikin using a digital stethoscope. Results demonstrate distinct latent clusters corresponding to heart and lung sources, as well as accurate reconstructions that preserve key spectral features of the original signals. The approach offers a robust and interpretable solution for blind source separation and has potential applications in portable diagnostic tools and intelligent stethoscope systems.
>
---
#### [new 023] DuetGen: Music Driven Two-Person Dance Generation via Hierarchical Masked Modeling
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于舞蹈生成任务，旨在解决两人舞蹈与音乐同步的问题。通过分阶段的编码与生成模型，实现高质量的双人舞蹈动作生成。**

- **链接: [http://arxiv.org/pdf/2506.18680v1](http://arxiv.org/pdf/2506.18680v1)**

> **作者:** Anindita Ghosh; Bing Zhou; Rishabh Dabral; Jian Wang; Vladislav Golyanik; Christian Theobalt; Philipp Slusallek; Chuan Guo
>
> **备注:** 11 pages, 7 figures, 2 tables, accepted in ACM Siggraph 2025 conference track
>
> **摘要:** We present DuetGen, a novel framework for generating interactive two-person dances from music. The key challenge of this task lies in the inherent complexities of two-person dance interactions, where the partners need to synchronize both with each other and with the music. Inspired by the recent advances in motion synthesis, we propose a two-stage solution: encoding two-person motions into discrete tokens and then generating these tokens from music. To effectively capture intricate interactions, we represent both dancers' motions as a unified whole to learn the necessary motion tokens, and adopt a coarse-to-fine learning strategy in both the stages. Our first stage utilizes a VQ-VAE that hierarchically separates high-level semantic features at a coarse temporal resolution from low-level details at a finer resolution, producing two discrete token sequences at different abstraction levels. Subsequently, in the second stage, two generative masked transformers learn to map music signals to these dance tokens: the first producing high-level semantic tokens, and the second, conditioned on music and these semantic tokens, producing the low-level tokens. We train both transformers to learn to predict randomly masked tokens within the sequence, enabling them to iteratively generate motion tokens by filling an empty token sequence during inference. Through the hierarchical masked modeling and dedicated interaction representation, DuetGen achieves the generation of synchronized and interactive two-person dances across various genres. Extensive experiments and user studies on a benchmark duet dance dataset demonstrate state-of-the-art performance of DuetGen in motion realism, music-dance alignment, and partner coordination.
>
---
#### [new 024] Face-Voice Association for Audiovisual Active Speaker Detection in Egocentric Recordings
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音频视频说话人检测任务，解决egocentric记录中同步方法效果差的问题，提出基于面部与语音关联的框架，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.18055v1](http://arxiv.org/pdf/2506.18055v1)**

> **作者:** Jason Clarke; Yoshihiko Gotoh; Stefan Goetze
>
> **备注:** Accepted to EUSIPCO 2025. 5 pages, 1 figure. To appear in the Proceedings of the 33rd European Signal Processing Conference (EUSIPCO), September 8-12, 2025, Palermo, Italy
>
> **摘要:** Audiovisual active speaker detection (ASD) is conventionally performed by modelling the temporal synchronisation of acoustic and visual speech cues. In egocentric recordings, however, the efficacy of synchronisation-based methods is compromised by occlusions, motion blur, and adverse acoustic conditions. In this work, a novel framework is proposed that exclusively leverages cross-modal face-voice associations to determine speaker activity. An existing face-voice association model is integrated with a transformer-based encoder that aggregates facial identity information by dynamically weighting each frame based on its visual quality. This system is then coupled with a front-end utterance segmentation method, producing a complete ASD system. This work demonstrates that the proposed system, Self-Lifting for audiovisual active speaker detection(SL-ASD), achieves performance comparable to, and in certain cases exceeding, that of parameter-intensive synchronisation-based approaches with significantly fewer learnable parameters, thereby validating the feasibility of substituting strict audiovisual synchronisation modelling with flexible biometric associations in challenging egocentric scenarios.
>
---
#### [new 025] Splitformer: An improved early-exit architecture for automatic speech recognition on edge devices
- **分类: cs.CL; cs.SD; eess.AS; 68T50 (Primary); I.2.7; I.5.4**

- **简介: 该论文属于自动语音识别任务，旨在解决边缘设备上计算资源受限的问题。通过引入并行层提升早期退出模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.18035v1](http://arxiv.org/pdf/2506.18035v1)**

> **作者:** Maxence Lasbordes; Daniele Falavigna; Alessio Brutti
>
> **备注:** 5 pages, 3 Postscript figures
>
> **摘要:** The ability to dynamically adjust the computational load of neural models during inference in a resource aware manner is crucial for on-device processing scenarios, characterised by limited and time-varying computational resources. Early-exit architectures represent an elegant and effective solution, since they can process the input with a subset of their layers, exiting at intermediate branches (the upmost layers are hence removed from the model). From a different perspective, for automatic speech recognition applications there are memory-efficient neural architectures that apply variable frame rate analysis, through downsampling/upsampling operations in the middle layers, reducing the overall number of operations and improving significantly the performance on well established benchmarks. One example is the Zipformer. However, these architectures lack the modularity necessary to inject early-exit branches. With the aim of improving the performance in early-exit models, we propose introducing parallel layers in the architecture that process downsampled versions of their inputs. % in conjunction with standard processing layers. We show that in this way the speech recognition performance on standard benchmarks significantly improve, at the cost of a small increase in the overall number of model parameters but without affecting the inference time.
>
---
#### [new 026] AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System
- **分类: cs.HC; cs.AI; cs.SD; eess.AS; H.5.5**

- **简介: 该论文属于音乐生成任务，旨在解决传统和声器需用户输入音调的问题。提出AI Harmonizer系统，自动生成四部和声，提升演唱表现。**

- **链接: [http://arxiv.org/pdf/2506.18143v1](http://arxiv.org/pdf/2506.18143v1)**

> **作者:** Lancelot Blanchard; Cameron Holt; Joseph A. Paradiso
>
> **备注:** 4 pages, 3 figures
>
> **摘要:** Vocals harmonizers are powerful tools to help solo vocalists enrich their melodies with harmonically supportive voices. These tools exist in various forms, from commercially available pedals and software to custom-built systems, each employing different methods to generate harmonies. Traditional harmonizers often require users to manually specify a key or tonal center, while others allow pitch selection via an external keyboard-both approaches demanding some degree of musical expertise. The AI Harmonizer introduces a novel approach by autonomously generating musically coherent four-part harmonies without requiring prior harmonic input from the user. By integrating state-of-the-art generative AI techniques for pitch detection and voice modeling with custom-trained symbolic music models, our system arranges any vocal melody into rich choral textures. In this paper, we present our methods, explore potential applications in performance and composition, and discuss future directions for real-time implementations. While our system currently operates offline, we believe it represents a significant step toward AI-assisted vocal performance and expressive musical augmentation. We release our implementation on GitHub.
>
---
#### [new 027] Episode-specific Fine-tuning for Metric-based Few-shot Learners with Optimization-based Training
- **分类: cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于少样本分类任务，旨在解决支持样本利用不足的问题。通过提出特定于回合的微调方法，提升模型适应能力并防止过拟合。**

- **链接: [http://arxiv.org/pdf/2506.17499v1](http://arxiv.org/pdf/2506.17499v1)**

> **作者:** Xuanyu Zhuang; Geoffroy Peeters; Gaël Richard
>
> **摘要:** In few-shot classification tasks (so-called episodes), a small set of labeled support samples is provided during inference to aid the classification of unlabeled query samples. Metric-based models typically operate by computing similarities between query and support embeddings within a learned metric space, followed by nearest-neighbor classification. However, these labeled support samples are often underutilized--they are only used for similarity comparison, despite their potential to fine-tune and adapt the metric space itself to the classes in the current episode. To address this, we propose a series of simple yet effective episode-specific, during-inference fine-tuning methods for metric-based models, including Rotational Division Fine-Tuning (RDFT) and its two variants, Iterative Division Fine-Tuning (IDFT) and Augmented Division Fine-Tuning (ADFT). These methods construct pseudo support-query pairs from the given support set to enable fine-tuning even for non-parametric models. Nevertheless, the severely limited amount of data in each task poses a substantial risk of overfitting when applying such fine-tuning strategies. To mitigate this, we further propose to train the metric-based model within an optimization-based meta-learning framework. With the combined efforts of episode-specific fine-tuning and optimization-based meta-training, metric-based models are equipped with the ability to rapidly adapt to the limited support samples during inference while avoiding overfitting. We validate our approach on three audio datasets from diverse domains, namely ESC-50 (environmental sounds), Speech Commands V2 (spoken keywords), and Medley-solos-DB (musical instrument). Experimental results demonstrate that our approach consistently improves performance for all evaluated metric-based models (especially for attention-based models) and generalizes well across different audio domains.
>
---
## 更新

#### [replaced 001] The Voice Timbre Attribute Detection 2025 Challenge Evaluation Plan
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.09382v2](http://arxiv.org/pdf/2505.09382v2)**

> **作者:** Zhengyan Sheng; Jinghao He; Liping Chen; Kong Aik Lee; Zhen-Hua Ling
>
> **摘要:** Voice timbre refers to the unique quality or character of a person's voice that distinguishes it from others as perceived by human hearing. The Voice Timbre Attribute Detection (VtaD) 2025 challenge focuses on explaining the voice timbre attribute in a comparative manner. In this challenge, the human impression of voice timbre is verbalized with a set of sensory descriptors, including bright, coarse, soft, magnetic, and so on. The timbre is explained from the comparison between two voices in their intensity within a specific descriptor dimension. The VtaD 2025 challenge starts in May and culminates in a special proposal at the NCMMSC2025 conference in October 2025 in Zhenjiang, China.
>
---
#### [replaced 002] Evaluation of the Pronunciation of Tajweed Rules Based on DNN as a Step Towards Interactive Recitation Learning
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.23470v2](http://arxiv.org/pdf/2503.23470v2)**

> **作者:** Dim Shaiakhmetov; Gulnaz Gimaletdinova; Kadyrmamat Momunov; Selcuk Cankurt
>
> **摘要:** Proper recitation of the Quran, adhering to the rules of Tajweed, is crucial for preventing mistakes during recitation and requires significant effort to master. Traditional methods of teaching these rules are limited by the availability of qualified instructors and time constraints. Automatic evaluation of recitation can address these challenges by providing prompt feedback and supporting independent practice. This study focuses on developing a deep learning model to classify three Tajweed rules - separate stretching (Al Mad), tight noon (Ghunnah), and hide (Ikhfaa) - using the publicly available QDAT dataset, which contains over 1,500 audio recordings. The input data consisted of audio recordings from this dataset, transformed into normalized mel-spectrograms. For classification, the EfficientNet-B0 architecture was used, enhanced with a Squeeze-and-Excitation attention mechanism. The developed model achieved accuracy rates of 95.35%, 99.34%, and 97.01% for the respective rules. An analysis of the learning curves confirmed the model's robustness and absence of overfitting. The proposed approach demonstrates high efficiency and paves the way for developing interactive educational systems for Tajweed study.
>
---
#### [replaced 003] SuPseudo: A Pseudo-supervised Learning Method for Neural Speech Enhancement in Far-field Speech Recognition
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24450v2](http://arxiv.org/pdf/2505.24450v2)**

> **作者:** Longjie Luo; Lin Li; Qingyang Hong
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** Due to the lack of target speech annotations in real-recorded far-field conversational datasets, speech enhancement (SE) models are typically trained on simulated data. However, the trained models often perform poorly in real-world conditions, hindering their application in far-field speech recognition. To address the issue, we (a) propose direct sound estimation (DSE) to estimate the oracle direct sound of real-recorded data for SE; and (b) present a novel pseudo-supervised learning method, SuPseudo, which leverages DSE-estimates as pseudo-labels and enables SE models to directly learn from and adapt to real-recorded data, thereby improving their generalization capability. Furthermore, an SE model called FARNET is designed to fully utilize SuPseudo. Experiments on the MISP2023 corpus demonstrate the effectiveness of SuPseudo, and our system significantly outperforms the previous state-of-the-art. A demo of our method can be found at https://EeLLJ.github.io/SuPseudo/.
>
---
#### [replaced 004] Introducing voice timbre attribute detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.09661v2](http://arxiv.org/pdf/2505.09661v2)**

> **作者:** Jinghao He; Zhengyan Sheng; Liping Chen; Kong Aik Lee; Zhen-Hua Ling
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2505.09382
>
> **摘要:** This paper focuses on explaining the timbre conveyed by speech signals and introduces a task termed voice timbre attribute detection (vTAD). In this task, voice timbre is explained with a set of sensory attributes describing its human perception. A pair of speech utterances is processed, and their intensity is compared in a designated timbre descriptor. Moreover, a framework is proposed, which is built upon the speaker embeddings extracted from the speech utterances. The investigation is conducted on the VCTK-RVA dataset. Experimental examinations on the ECAPA-TDNN and FACodec speaker encoders demonstrated that: 1) the ECAPA-TDNN speaker encoder was more capable in the seen scenario, where the testing speakers were included in the training set; 2) the FACodec speaker encoder was superior in the unseen scenario, where the testing speakers were not part of the training, indicating enhanced generalization capability. The VCTK-RVA dataset and open-source code are available on the website https://github.com/vTAD2025-Challenge/vTAD.
>
---
#### [replaced 005] Vocoder-Free Non-Parallel Conversion of Whispered Speech With Masked Cycle-Consistent Generative Adversarial Networks
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2306.06514v2](http://arxiv.org/pdf/2306.06514v2)**

> **作者:** Dominik Wagner; Ilja Baumann; Tobias Bocklet
>
> **备注:** Accepted at TSD 2025
>
> **摘要:** Cycle-consistent generative adversarial networks have been widely used in non-parallel voice conversion (VC). Their ability to learn mappings between source and target features without relying on parallel training data eliminates the need for temporal alignments. However, most methods decouple the conversion of acoustic features from synthesizing the audio signal by using separate models for conversion and waveform synthesis. This work unifies conversion and synthesis into a single model, thereby eliminating the need for a separate vocoder. By leveraging cycle-consistent training and a self-supervised auxiliary training task, our model is able to efficiently generate converted high-quality raw audio waveforms. Subjective listening tests showed that our unified approach achieved improvements of up to 6.7% relative to the baseline in whispered VC. Mean opinion score predictions also yielded stable results in conventional VC (between 0.5% and 2.4% relative improvement).
>
---
#### [replaced 006] Hierarchical Control of Emotion Rendering in Speech Synthesis
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.12498v3](http://arxiv.org/pdf/2412.12498v3)**

> **作者:** Sho Inoue; Kun Zhou; Shuai Wang; Haizhou Li
>
> **备注:** Accepted to IEEE Transactions on Affective Computing
>
> **摘要:** Emotional text-to-speech synthesis (TTS) aims to generate realistic emotional speech from input text. However, quantitatively controlling multi-level emotion rendering remains challenging. In this paper, we propose a flow-matching based emotional TTS framework with a novel approach for emotion intensity modeling to facilitate fine-grained control over emotion rendering at the phoneme, word, and utterance levels. We introduce a hierarchical emotion distribution (ED) extractor that captures a quantifiable ED embedding across different speech segment levels. Additionally, we explore various acoustic features and assess their impact on emotion intensity modeling. During TTS training, the hierarchical ED embedding effectively captures the variance in emotion intensity from the reference audio and correlates it with linguistic and speaker information. The TTS model not only generates emotional speech during inference, but also quantitatively controls the emotion rendering over the speech constituents. Both objective and subjective evaluations demonstrate the effectiveness of our framework in terms of speech quality, emotional expressiveness, and hierarchical emotion control.
>
---
#### [replaced 007] Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.22029v2](http://arxiv.org/pdf/2505.22029v2)**

> **作者:** Jinming Zhang; Xuanru Zhou; Jiachen Lian; Shuhe Li; William Li; Zoe Ezzes; Rian Bogley; Lisa Wauters; Zachary Miller; Jet Vonk; Brittany Morin; Maria Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Speech dysfluency detection is crucial for clinical diagnosis and language assessment, but existing methods are limited by the scarcity of high-quality annotated data. Although recent advances in TTS model have enabled synthetic dysfluency generation, existing synthetic datasets suffer from unnatural prosody and limited contextual diversity. To address these limitations, we propose LLM-Dys -- the most comprehensive dysfluent speech corpus with LLM-enhanced dysfluency simulation. This dataset captures 11 dysfluency categories spanning both word and phoneme levels. Building upon this resource, we improve an end-to-end dysfluency detection framework. Experimental validation demonstrates state-of-the-art performance. All data, models, and code are open-sourced at https://github.com/Berkeley-Speech-Group/LLM-Dys.
>
---
#### [replaced 008] Pseudo Labels-based Neural Speech Enhancement for the AVSR Task in the MISP-Meeting Challenge
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24446v2](http://arxiv.org/pdf/2505.24446v2)**

> **作者:** Longjie Luo; Shenghui Lu; Lin Li; Qingyang Hong
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** This paper presents our system for the MISP-Meeting Challenge Track 2. The primary difficulty lies in the dataset, which contains strong background noise, reverberation, overlapping speech, and diverse meeting topics. To address these issues, we (a) designed G-SpatialNet, a speech enhancement (SE) model to improve Guided Source Separation (GSS) signals; (b) proposed TLS, a framework comprising time alignment, level alignment, and signal-to-noise ratio filtering, to generate signal-level pseudo labels for real-recorded far-field audio data, thereby facilitating SE models' training; and (c) explored fine-tuning strategies, data augmentation, and multimodal information to enhance the performance of pre-trained Automatic Speech Recognition (ASR) models in meeting scenarios. Finally, our system achieved character error rates (CERs) of 5.44% and 9.52% on the Dev and Eval sets, respectively, with relative improvements of 64.8% and 52.6% over the baseline, securing second place.
>
---
#### [replaced 009] Protecting Your Voice: Temporal-aware Robust Watermarking
- **分类: cs.CR; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.14832v2](http://arxiv.org/pdf/2504.14832v2)**

> **作者:** Yue Li; Weizhi Liu; Dongdong Lin; Hui Tian; Hongxia Wang
>
> **摘要:** The rapid advancement of generative models has led to the synthesis of real-fake ambiguous voices. To erase the ambiguity, embedding watermarks into the frequency-domain features of synthesized voices has become a common routine. However, the robustness achieved by choosing the frequency domain often comes at the expense of fine-grained voice features, leading to a loss of fidelity. Maximizing the comprehensive learning of time-domain features to enhance fidelity while maintaining robustness, we pioneer a \textbf{\underline{t}}emporal-aware \textbf{\underline{r}}ob\textbf{\underline{u}}st wat\textbf{\underline{e}}rmarking (\emph{True}) method for protecting the speech and singing voice. For this purpose, the integrated content-driven encoder is designed for watermarked waveform reconstruction, which is structurally lightweight. Additionally, the temporal-aware gated convolutional network is meticulously designed to bit-wise recover the watermark. Comprehensive experiments and comparisons with existing state-of-the-art methods have demonstrated the superior fidelity and vigorous robustness of the proposed \textit{True} achieving an average PESQ score of 4.63.
>
---
#### [replaced 010] Information and motor constraints shape melodic diversity across cultures
- **分类: cs.SD; cs.IT; eess.AS; math.IT; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2408.12635v3](http://arxiv.org/pdf/2408.12635v3)**

> **作者:** John M McBride; Nahie Kim; Yuri Nishikawa; Mekhmed Saadakeev; Marcus T Pearce; Tsvi Tlusty
>
> **摘要:** The number of possible melodies is unfathomably large, yet despite this virtually unlimited potential for melodic variation, melodies from different societies can be surprisingly similar. The motor constraint hypothesis accounts for certain similarities, such as scalar motion and contour shape, but not for other major common features, such as repetition, song length, and scale size. Here we investigate the role of information constraints in shaping these hallmarks of melodies. We measure determinants of information rate in 62 corpora of Folk melodies spanning several continents, finding multiple trade-offs that all act to constrain the information rate across societies. By contrast, 39 corpora of Art music from Europe (including Turkey) show longer, more complex melodies, and increased complexity over time, suggesting different cultural-evolutionary selection pressures in Art and Folk music, possibly due to the use of written versus oral transmission. Our parameter-free model predicts the empirical scale degree distribution using information constraints on scalar motion, melody length, and, most importantly, information rate. These results provide strong evidence that information constraints during cultural transmission of music limit the number of notes in a scale, and suggests that a tendency for intermediate melodic complexity reflects a fundamental constraint on the cultural evolution of melody.
>
---
#### [replaced 011] Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
- **分类: cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13642v2](http://arxiv.org/pdf/2506.13642v2)**

> **作者:** Shaolei Zhang; Shoutao Guo; Qingkai Fang; Yan Zhou; Yang Feng
>
> **备注:** Code: https://github.com/ictnlp/Stream-Omni , Model: https://huggingface.co/ICTNLP/stream-omni-8b
>
> **摘要:** The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.
>
---
#### [replaced 012] AnyEnhance: A Unified Generative Model with Prompt-Guidance and Self-Critic for Voice Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.15417v2](http://arxiv.org/pdf/2501.15417v2)**

> **作者:** Junan Zhang; Jing Yang; Zihao Fang; Yuancheng Wang; Zehua Zhang; Zhuo Wang; Fan Fan; Zhizheng Wu
>
> **备注:** Accepted by IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP) 2025
>
> **摘要:** We introduce AnyEnhance, a unified generative model for voice enhancement that processes both speech and singing voices. Based on a masked generative model, AnyEnhance is capable of handling both speech and singing voices, supporting a wide range of enhancement tasks including denoising, dereverberation, declipping, super-resolution, and target speaker extraction, all simultaneously and without fine-tuning. AnyEnhance introduces a prompt-guidance mechanism for in-context learning, which allows the model to natively accept a reference speaker's timbre. In this way, it could boost enhancement performance when a reference audio is available and enable the target speaker extraction task without altering the underlying architecture. Moreover, we also introduce a self-critic mechanism into the generative process for masked generative models, yielding higher-quality outputs through iterative self-assessment and refinement. Extensive experiments on various enhancement tasks demonstrate AnyEnhance outperforms existing methods in terms of both objective metrics and subjective listening tests. Demo audios are publicly available at https://amphionspace.github.io/anyenhance/.
>
---
#### [replaced 013] S2ST-Omni: An Efficient and Scalable Multilingual Speech-to-Speech Translation Framework via Seamless Speech-Text Alignment and Streaming Speech Generation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.11160v4](http://arxiv.org/pdf/2506.11160v4)**

> **作者:** Yu Pan; Yuguang Yang; Yanni Hu; Jianhao Ye; Xiang Zhang; Hongbin Zhou; Lei Ma; Jianjun Zhao
>
> **备注:** V2 and V3 versions contain experimental errors due to incorrect training data. The results and conclusions are invalid. A corrected version is under preparation and will be uploaded soon. Please do not cite these versions. Working in progress
>
> **摘要:** Multilingual speech-to-speech translation (S2ST) aims to directly convert spoken utterances from multiple source languages into fluent and intelligible speech in a target language. Despite recent progress, several critical challenges persist: 1) achieving high-quality S2ST remains a significant obstacle; 2) most existing S2ST methods rely heavily on large-scale parallel speech corpora, which are difficult and resource-intensive to obtain. To tackle these challenges, we introduce S2ST-Omni, a novel, efficient, and scalable framework tailored for multilingual speech-to-speech translation. Specifically, we decompose S2ST into speech-to-text translation (S2TT) and text-to-speech synthesis (TTS). To enable high-quality S2TT while mitigating reliance on large-scale parallel speech corpora, we leverage powerful pretrained models: Whisper for robust audio understanding and Qwen 3.0 for advanced text comprehension. A lightweight speech adapter is introduced to bridge the modality gap between speech and text representations, facilitating effective utilization of pretrained multimodal knowledge. To ensure both translation accuracy and real-time responsiveness, we adopt a streaming speech generation model in the TTS stage, which generates the target speech in an autoregressive manner. Extensive experiments conducted on the CVSS benchmark demonstrate that S2ST-Omni consistently surpasses several state-of-the-art S2ST baselines in translation quality, highlighting its effectiveness and superiority.
>
---
