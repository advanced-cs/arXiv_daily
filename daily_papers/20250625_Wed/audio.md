# 音频 cs.SD;  eess.SP

- **最新发布 12 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Vo-Ve: An Explainable Voice-Vector for Speaker Identity Evaluation
- **分类: cs.SD**

- **简介: 该论文提出Vo-Ve，一种可解释的语音向量，用于说话人身份评估。解决传统嵌入不可解释的问题，通过概率描述语音属性，提升评估的可解释性。**

- **链接: [http://arxiv.org/pdf/2506.19446v1](http://arxiv.org/pdf/2506.19446v1)**

> **作者:** Jaejun Lee; Kyogu Lee
>
> **备注:** Interspeech 2025
>
> **摘要:** In this paper, we propose Vo-Ve, a novel voice-vector embedding that captures speaker identity. Unlike conventional speaker embeddings, Vo-Ve is explainable, as it contains the probabilities of explicit voice attribute classes. Through extensive analysis, we demonstrate that Vo-Ve not only evaluates speaker similarity competitively with conventional techniques but also provides an interpretable explanation in terms of voice attributes. We strongly believe that Vo-Ve can enhance evaluation schemes across various speech tasks due to its high-level explainability.
>
---
#### [new 002] A Fourier Explanation of AI-music Artifacts
- **分类: cs.SD**

- **简介: 该论文属于AI音乐检测任务，旨在解决AI生成音乐的识别问题。通过分析模型输出的频率特征，提出一种简单有效的检测方法。**

- **链接: [http://arxiv.org/pdf/2506.19108v1](http://arxiv.org/pdf/2506.19108v1)**

> **作者:** Darius Afchar; Gabriel Meseguer-Brocal; Kamil Akesbi; Romain Hennequin
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** The rapid rise of generative AI has transformed music creation, with millions of users engaging in AI-generated music. Despite its popularity, concerns regarding copyright infringement, job displacement, and ethical implications have led to growing scrutiny and legal challenges. In parallel, AI-detection services have emerged, yet these systems remain largely opaque and privately controlled, mirroring the very issues they aim to address. This paper explores the fundamental properties of synthetic content and how it can be detected. Specifically, we analyze deconvolution modules commonly used in generative models and mathematically prove that their outputs exhibit systematic frequency artifacts -- manifesting as small yet distinctive spectral peaks. This phenomenon, related to the well-known checkerboard artifact, is shown to be inherent to a chosen model architecture rather than a consequence of training data or model weights. We validate our theoretical findings through extensive experiments on open-source models, as well as commercial AI-music generators such as Suno and Udio. We use these insights to propose a simple and interpretable detection criterion for AI-generated music. Despite its simplicity, our method achieves detection accuracy on par with deep learning-based approaches, surpassing 99% accuracy on several scenarios.
>
---
#### [new 003] ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍ClearerVoice-Studio，一个专注于语音增强、分离等任务的开源工具包，旨在解决研究与应用间的鸿沟。**

- **链接: [http://arxiv.org/pdf/2506.19398v1](http://arxiv.org/pdf/2506.19398v1)**

> **作者:** Shengkui Zhao; Zexu Pan; Bin Ma
>
> **备注:** accepted by Interspeech 2025, 5 pages, 5 tables
>
> **摘要:** This paper introduces ClearerVoice-Studio, an open-source, AI-powered speech processing toolkit designed to bridge cutting-edge research and practical application. Unlike broad platforms like SpeechBrain and ESPnet, ClearerVoice-Studio focuses on interconnected speech tasks of speech enhancement, separation, super-resolution, and multimodal target speaker extraction. A key advantage is its state-of-the-art pretrained models, including FRCRN with 3 million uses and MossFormer with 2.5 million uses, optimized for real-world scenarios. It also offers model optimization tools, multi-format audio support, the SpeechScore evaluation toolkit, and user-friendly interfaces, catering to researchers, developers, and end-users. Its rapid adoption attracting 3000 GitHub stars and 239 forks highlights its academic and industrial impact. This paper details ClearerVoice-Studio's capabilities, architectures, training strategies, benchmarks, community impact, and future plan. Source code is available at https://github.com/modelscope/ClearerVoice-Studio.
>
---
#### [new 004] SHAMaNS: Sound Localization with Hybrid Alpha-Stable Spatial Measure and Neural Steerer
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于声源定位任务，解决多声源环境下定位精度问题。通过结合α-稳定模型与神经网络，提升方向到达估计的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.18954v1](http://arxiv.org/pdf/2506.18954v1)**

> **作者:** Diego Di Carlo; Mathieu Fontaine; Aditya Arie Nugraha; Yoshiaki Bando; Kazuyoshi Yoshii
>
> **备注:** European Signal Processing Conference (EUSIPCO), Sep 2025, Palermo, Italy
>
> **摘要:** This paper describes a sound source localization (SSL) technique that combines an $\alpha$-stable model for the observed signal with a neural network-based approach for modeling steering vectors. Specifically, a physics-informed neural network, referred to as Neural Steerer, is used to interpolate measured steering vectors (SVs) on a fixed microphone array. This allows for a more robust estimation of the so-called $\alpha$-stable spatial measure, which represents the most plausible direction of arrival (DOA) of a target signal. As an $\alpha$-stable model for the non-Gaussian case ($\alpha$ $\in$ (0, 2)) theoretically defines a unique spatial measure, we choose to leverage it to account for residual reconstruction error of the Neural Steerer in the downstream tasks. The objective scores indicate that our proposed technique outperforms state-of-the-art methods in the case of multiple sound sources.
>
---
#### [new 005] Learning to assess subjective impressions from speech
- **分类: cs.SD**

- **简介: 该论文属于语音主观印象评估任务，旨在通过神经网络模型对语音的主观描述进行评分。研究提出使用CCR数据训练模型，以提高评估效果。**

- **链接: [http://arxiv.org/pdf/2506.19335v1](http://arxiv.org/pdf/2506.19335v1)**

> **作者:** Yuto Kondo; Hirokazu Kameoka; Kou Tanaka; Takuhiro Kaneko; Noboru Harada
>
> **备注:** Accepted on EUSIPCO 2024
>
> **摘要:** We tackle a new task of training neural network models that can assess subjective impressions conveyed through speech and assign scores accordingly, inspired by the work on automatic speech quality assessment (SQA). Speech impressions are often described using phrases like `cute voice.' We define such phrases as subjective voice descriptors (SVDs). Focusing on the difference in usage scenarios between the proposed task and automatic SQA, we design a framework capable of accommodating SVDs personalized to each individual, such as `my favorite voice.' In this work, we compiled a dataset containing speech labels derived from both abosolute category ratings (ACR) and comparison category ratings (CCR). As an evaluation metric for assessment performance, we introduce ppref, the accuracy of the predicted score ordering of two samples on CCR test samples. Alongside the conventional model and learning methods based on ACR data, we also investigated RankNet learning using CCR data. We experimentally find that the ppref is moderate even with very limited training data. We also discover the CCR training is superior to the ACR training. These results support the idea that assessment models based on personalized SVDs, which typically must be trained on limited data, can be effectively learned from CCR data.
>
---
#### [new 006] IndieFake Dataset: A Benchmark Dataset for Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有数据集缺乏南亚语种样本的问题。工作包括构建IndieFake数据集并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.19014v1](http://arxiv.org/pdf/2506.19014v1)**

> **作者:** Abhay Kumar; Kunal Verma; Omkar More
>
> **摘要:** Advancements in audio deepfake technology offers benefits like AI assistants, better accessibility for speech impairments, and enhanced entertainment. However, it also poses significant risks to security, privacy, and trust in digital communications. Detecting and mitigating these threats requires comprehensive datasets. Existing datasets lack diverse ethnic accents, making them inadequate for many real-world scenarios. Consequently, models trained on these datasets struggle to detect audio deepfakes in diverse linguistic and cultural contexts such as in South-Asian countries. Ironically, there is a stark lack of South-Asian speaker samples in the existing datasets despite constituting a quarter of the worlds population. This work introduces the IndieFake Dataset (IFD), featuring 27.17 hours of bonafide and deepfake audio from 50 English speaking Indian speakers. IFD offers balanced data distribution and includes speaker-level characterization, absent in datasets like ASVspoof21 (DF). We evaluated various baselines on IFD against existing ASVspoof21 (DF) and In-The-Wild (ITW) datasets. IFD outperforms ASVspoof21 (DF) and proves to be more challenging compared to benchmark ITW dataset. The dataset will be publicly available upon acceptance.
>
---
#### [new 007] A Robust Method for Pitch Tracking in the Frequency Following Response using Harmonic Amplitude Summation Filterbank
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于语音信号处理任务，旨在解决FFR中F0估计问题。通过引入谐波幅度求和滤波器组，提升F0跟踪精度。**

- **链接: [http://arxiv.org/pdf/2506.19253v1](http://arxiv.org/pdf/2506.19253v1)**

> **作者:** Sajad Sadeghkhani; Maryam Karimi Boroujeni; Hilmi R. Dajani; Saeid R. Seydnejad; Christian Giguère
>
> **摘要:** The Frequency Following Response (FFR) reflects the brain's neural encoding of auditory stimuli including speech. Because the fundamental frequency (F0), a physical correlate of pitch, is one of the essential features of speech, there has been particular interest in characterizing the FFR at F0, especially when F0 varies over time. The standard method for extracting F0 in FFRs has been the Autocorrelation Function (ACF). This paper investigates harmonic-structure-based F0 estimation algorithms, originally developed for speech and music, and resolves their poor performance when applied to FFRs in two steps. Firstly, given that unlike in speech or music, stimulus F0 of FFRs is already known, we introduce a stimulus-aware filterbank that selectively aggregates amplitudes at F0 and its harmonics while suppressing noise at non-harmonic frequencies. This method, called Harmonic Amplitude Summation (HAS), evaluates F0 candidates only within a range centered around the stimulus F0. Secondly, unlike other pitch tracking methods that select the highest peak, our method chooses the most prominent one, as it better reflects the underlying periodicity of FFRs. To the best of our knowledge, this is the first study to propose an F0 estimation algorithm for FFRs that relies on harmonic structure. Analyzing recorded FFRs from 16 normal hearing subjects to 4 natural speech stimuli with a wide F0 variation from 89 Hz to 452 Hz showed that this method outperformed ACF by reducing the average Root-Mean-Square-Error (RMSE) within each response and stimulus F0 contour pair by 8.8% to 47.4%, depending on the stimulus.
>
---
#### [new 008] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本转语音（TTS）评估任务，旨在解决主观与客观评价不一致的问题。提出TTSDS2指标，并提供多语言评估资源与基准。**

- **链接: [http://arxiv.org/pdf/2506.19441v1](http://arxiv.org/pdf/2506.19441v1)**

> **作者:** Christoph Minixhofer; Ondrej Klejch; Peter Bell
>
> **摘要:** Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for continually recreating a multilingual test dataset to avoid data leakage; and a continually updated benchmark for TTS in 14 languages.
>
---
#### [new 009] Benchmarking Music Generation Models and Metrics via Human Preference Studies
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于音乐生成评估任务，旨在解决如何通过人类偏好验证生成模型与评价指标的关联性。工作包括生成6000首歌曲并进行15000次对比测试。**

- **链接: [http://arxiv.org/pdf/2506.19085v1](http://arxiv.org/pdf/2506.19085v1)**

> **作者:** Florian Grötschla; Ahmet Solak; Luca A. Lanzendörfer; Roger Wattenhofer
>
> **备注:** Accepted at ICASSP 2025
>
> **摘要:** Recent advancements have brought generated music closer to human-created compositions, yet evaluating these models remains challenging. While human preference is the gold standard for assessing quality, translating these subjective judgments into objective metrics, particularly for text-audio alignment and music quality, has proven difficult. In this work, we generate 6k songs using 12 state-of-the-art models and conduct a survey of 15k pairwise audio comparisons with 2.5k human participants to evaluate the correlation between human preferences and widely used metrics. To the best of our knowledge, this work is the first to rank current state-of-the-art music generation models and metrics based on human preference. To further the field of subjective metric evaluation, we provide open access to our dataset of generated music and human evaluations.
>
---
#### [new 010] Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于视频到音频生成任务，解决音画同步与语义对齐问题。提出Kling-Foley模型，结合多模态扩散Transformer和音频编码器，提升生成质量与一致性。**

- **链接: [http://arxiv.org/pdf/2506.19774v1](http://arxiv.org/pdf/2506.19774v1)**

> **作者:** Jun Wang; Xijuan Zeng; Chunyu Qiang; Ruilong Chen; Shiyao Wang; Le Wang; Wangjing Zhou; Pengfei Cai; Jiahui Zhao; Nan Li; Zihan Li; Yuzhe Liang; Xiaopeng Wang; Haorui Zheng; Ming Wen; Kang Yin; Yiran Wang; Nan Li; Feng Deng; Liang Dong; Chen Zhang; Di Zhang; Kun Gai
>
> **摘要:** We propose Kling-Foley, a large-scale multimodal Video-to-Audio generation model that synthesizes high-quality audio synchronized with video content. In Kling-Foley, we introduce multimodal diffusion transformers to model the interactions between video, audio, and text modalities, and combine it with a visual semantic representation module and an audio-visual synchronization module to enhance alignment capabilities. Specifically, these modules align video conditions with latent audio elements at the frame level, thereby improving semantic alignment and audio-visual synchronization. Together with text conditions, this integrated approach enables precise generation of video-matching sound effects. In addition, we propose a universal latent audio codec that can achieve high-quality modeling in various scenarios such as sound effects, speech, singing, and music. We employ a stereo rendering method that imbues synthesized audio with a spatial presence. At the same time, in order to make up for the incomplete types and annotations of the open-source benchmark, we also open-source an industrial-level benchmark Kling-Audio-Eval. Our experiments show that Kling-Foley trained with the flow matching objective achieves new audio-visual SOTA performance among public models in terms of distribution matching, semantic alignment, temporal alignment and audio quality.
>
---
#### [new 011] Loss functions incorporating auditory spatial perception in deep learning -- a review
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决空间音频质量评估问题。通过综述结合听觉空间感知的损失函数，提升耳机音频的沉浸感和真实感。**

- **链接: [http://arxiv.org/pdf/2506.19404v1](http://arxiv.org/pdf/2506.19404v1)**

> **作者:** Boaz Rafaely; Stefan Weinzierl; Or Berebi; Fabian Brinkmann
>
> **备注:** Submitted to I3DA 2025
>
> **摘要:** Binaural reproduction aims to deliver immersive spatial audio with high perceptual realism over headphones. Loss functions play a central role in optimizing and evaluating algorithms that generate binaural signals. However, traditional signal-related difference measures often fail to capture the perceptual properties that are essential to spatial audio quality. This review paper surveys recent loss functions that incorporate spatial perception cues relevant to binaural reproduction. It focuses on losses applied to binaural signals, which are often derived from microphone recordings or Ambisonics signals, while excluding those based on room impulse responses. Guided by the Spatial Audio Quality Inventory (SAQI), the review emphasizes perceptual dimensions related to source localization and room response, while excluding general spectral-temporal attributes. The literature survey reveals a strong focus on localization cues, such as interaural time and level differences (ITDs, ILDs), while reverberation and other room acoustic attributes remain less explored in loss function design. Recent works that estimate room acoustic parameters and develop embeddings that capture room characteristics indicate their potential for future integration into neural network training. The paper concludes by highlighting future research directions toward more perceptually grounded loss functions that better capture the listener's spatial experience.
>
---
#### [new 012] Enhanced Hybrid Transducer and Attention Encoder Decoder with Text Data
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决数据稀缺问题。通过融合语音和文本信息的联合模型J-TAED，提升ASR准确率并实现文本域适应。**

- **链接: [http://arxiv.org/pdf/2506.19159v1](http://arxiv.org/pdf/2506.19159v1)**

> **作者:** Yun Tang; Eesung Kim; Vijendra Raj Apsingekar
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** A joint speech and text optimization method is proposed for hybrid transducer and attention-based encoder decoder (TAED) modeling to leverage large amounts of text corpus and enhance ASR accuracy. The joint TAED (J-TAED) is trained with both speech and text input modalities together, while it only takes speech data as input during inference. The trained model can unify the internal representations from different modalities, and be further extended to text-based domain adaptation. It can effectively alleviate data scarcity for mismatch domain tasks since no speech data is required. Our experiments show J-TAED successfully integrates speech and linguistic information into one model, and reduce the WER by 5.8 ~12.8% on the Librispeech dataset. The model is also evaluated on two out-of-domain datasets: one is finance and another is named entity focused. The text-based domain adaptation brings 15.3% and 17.8% WER reduction on those two datasets respectively.
>
---
## 更新

#### [replaced 001] SLEEPING-DISCO 9M: A large-scale pre-training dataset for generative music modeling
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.14293v2](http://arxiv.org/pdf/2506.14293v2)**

> **作者:** Tawsif Ahmed; Andrej Radonjic; Gollam Rabby
>
> **摘要:** We present Sleeping-DISCO 9M, a large-scale pre-training dataset for music and song. To the best of our knowledge, there are no open-source high-quality dataset representing popular and well-known songs for generative music modeling tasks such as text-music, music-captioning, singing-voice synthesis, melody reconstruction and cross-model retrieval. Past contributions focused on isolated and constrained factors whose core perspective was to create synthetic or re-recorded music corpus (e.g. GTSinger, M4Singer) and arbitrarily large-scale audio datasets (e.g. DISCO-10M and LAIONDISCO-12M) had been another focus for the community. Unfortunately, adoption of these datasets has been below substantial in the generative music community as these datasets fail to reflect real-world music and its flavour. Our dataset changes this narrative and provides a dataset that is constructed using actual popular music and world-renowned artists.
>
---
#### [replaced 002] Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.08809v3](http://arxiv.org/pdf/2406.08809v3)**

> **作者:** Jaeyong Kang; Dorien Herremans
>
> **摘要:** Deep learning models for music have advanced drastically in recent years, but how good are machine learning models at capturing emotion, and what challenges are researchers facing? In this paper, we provide a comprehensive overview of the available music-emotion datasets and discuss evaluation standards as well as competitions in the field. We also offer a brief overview of various types of music emotion prediction models that have been built over the years, providing insights into the diverse approaches within the field. Through this examination, we highlight the challenges that persist in accurately capturing emotion in music, including issues related to dataset quality, annotation consistency, and model generalization. Additionally, we explore the impact of different modalities, such as audio, MIDI, and physiological signals, on the effectiveness of emotion prediction models. Through this examination, we identify persistent challenges in music emotion recognition (MER), including issues related to dataset quality, the ambiguity in emotion labels, and the difficulties of cross-dataset generalization. We argue that future advancements in MER require standardized benchmarks, larger and more diverse datasets, and improved model interpretability. Recognizing the dynamic nature of this field, we have complemented our findings with an accompanying GitHub repository. This repository contains a comprehensive list of music emotion datasets and recent predictive models.
>
---
#### [replaced 003] MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18729v2](http://arxiv.org/pdf/2506.18729v2)**

> **作者:** Fang-Duo Tsai; Shih-Lun Wu; Weijaw Lee; Sheng-Ping Yang; Bo-Rui Chen; Hao-Chung Cheng; Yi-Hsuan Yang
>
> **备注:** Accepted by the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** We propose MuseControlLite, a lightweight mechanism designed to fine-tune text-to-music generation models for precise conditioning using various time-varying musical attributes and reference audio signals. The key finding is that positional embeddings, which have been seldom used by text-to-music generation models in the conditioner for text conditions, are critical when the condition of interest is a function of time. Using melody control as an example, our experiments show that simply adding rotary positional embeddings to the decoupled cross-attention layers increases control accuracy from 56.6% to 61.1%, while requiring 6.75 times fewer trainable parameters than state-of-the-art fine-tuning mechanisms, using the same pre-trained diffusion Transformer model of Stable Audio Open. We evaluate various forms of musical attribute control, audio inpainting, and audio outpainting, demonstrating improved controllability over MusicGen-Large and Stable Audio Open ControlNet at a significantly lower fine-tuning cost, with only 85M trainble parameters. Source code, model checkpoints, and demo examples are available at: https://musecontrollite.github.io/web/.
>
---
#### [replaced 004] SSPS: Self-Supervised Positive Sampling for Robust Self-Supervised Speaker Verification
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14561v2](http://arxiv.org/pdf/2505.14561v2)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** accepted at Interspeech 2025
>
> **摘要:** Self-Supervised Learning (SSL) has led to considerable progress in Speaker Verification (SV). The standard framework uses same-utterance positive sampling and data-augmentation to generate anchor-positive pairs of the same speaker. This is a major limitation, as this strategy primarily encodes channel information from the recording condition, shared by the anchor and positive. We propose a new positive sampling technique to address this bottleneck: Self-Supervised Positive Sampling (SSPS). For a given anchor, SSPS aims to find an appropriate positive, i.e., of the same speaker identity but a different recording condition, in the latent space using clustering assignments and a memory queue of positive embeddings. SSPS improves SV performance for both SimCLR and DINO, reaching 2.57% and 2.53% EER, outperforming SOTA SSL methods on VoxCeleb1-O. In particular, SimCLR-SSPS achieves a 58% EER reduction by lowering intra-speaker variance, providing comparable performance to DINO-SSPS.
>
---
#### [replaced 005] GD-Retriever: Controllable Generative Text-Music Retrieval with Diffusion Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.17886v2](http://arxiv.org/pdf/2506.17886v2)**

> **作者:** Julien Guinot; Elio Quinton; György Fazekas
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Multimodal contrastive models have achieved strong performance in text-audio retrieval and zero-shot settings, but improving joint embedding spaces remains an active research area. Less attention has been given to making these systems controllable and interactive for users. In text-music retrieval, the ambiguity of freeform language creates a many-to-many mapping, often resulting in inflexible or unsatisfying results. We introduce Generative Diffusion Retriever (GDR), a novel framework that leverages diffusion models to generate queries in a retrieval-optimized latent space. This enables controllability through generative tools such as negative prompting and denoising diffusion implicit models (DDIM) inversion, opening a new direction in retrieval control. GDR improves retrieval performance over contrastive teacher models and supports retrieval in audio-only latent spaces using non-jointly trained encoders. Finally, we demonstrate that GDR enables effective post-hoc manipulation of retrieval behavior, enhancing interactive control for text-music retrieval tasks.
>
---
