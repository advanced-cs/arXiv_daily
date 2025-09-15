# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] AI-enabled tuberculosis screening in a high-burden setting using cough sound analysis and speech foundation models
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出基于咳嗽声和语音基础模型的AI筛查工具，用于高负担地区结核病（TB）检测。通过分析500名参与者的咳嗽录音及临床数据，训练深度学习模型，显著提升TB识别准确率，具备临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.09746v1](http://arxiv.org/pdf/2509.09746v1)**

> **作者:** Ning Ma; Bahman Mirheidari; Guy J. Brown; Minyoi M. Maimbolwa; Nsala Sanjase; Solomon Chifwamba; Seke Muzazu; Monde Muyoyeta; Mary Kagujje
>
> **备注:** submitted to The Lancet Digital Health
>
> **摘要:** Background Artificial intelligence (AI) can detect disease-related acoustic patterns in cough sounds, offering a scalable approach to tuberculosis (TB) screening in high-burden, low-resource settings. Previous studies have been limited by small datasets, under-representation of symptomatic non-TB patients, reliance on simple models, and recordings collected under idealised conditions. Methods We enrolled 512 participants at two hospitals in Zambia, grouped as bacteriologically confirmed TB (TB+), symptomatic patients with other respiratory diseases (OR), and healthy controls (HC). Usable cough recordings plus demographic and clinical data were obtained from 500 participants. Deep learning classifiers based on speech foundation models were trained on cough recordings. The best-performing model, trained on 3-second segments, was further evaluated with demographic and clinical features. Findings The best audio-only classifier achieved an AUROC of 85.2% for distinguishing TB+ from all others (TB+/Rest) and 80.1% for TB+ versus OR. Adding demographic and clinical features improved performance to 92.1% (TB+/Rest) and 84.2% (TB+/OR). At a threshold of 0.38, the multimodal model reached 90.3% sensitivity and 73.1% specificity for TB+/Rest, and 80.6% and 73.1% for TB+/OR. Interpretation Cough analysis using speech foundation models, especially when combined with demographic and clinical data, showed strong potential as a TB triage tool, meeting WHO target product profile benchmarks. The model was robust to confounding factors including background noise, recording time, and device variability, indicating detection of genuine disease-related acoustic patterns. Further validation across diverse regions and case definitions, including subclinical TB, is required before clinical use.
>
---
#### [new 002] Data-independent Beamforming for End-to-end Multichannel Multi-speaker ASR
- **分类: cs.SD**

- **简介: 论文提出一种数据无关的波束成形方法，用于多通道多说话人语音识别任务。该方法通过球面极坐标处理特定角度区域，提升ASR性能，减少词错误率并提高说话人计数准确率。**

- **链接: [http://arxiv.org/pdf/2509.10234v1](http://arxiv.org/pdf/2509.10234v1)**

> **作者:** Can Cui; Paul Magron; Mostafa Sadeghi; Emmanuel Vincent
>
> **备注:** Published in the IEEE 26th International Workshop on Multimedia Signal Processing (MMSP 2025)
>
> **摘要:** Automatic speech recognition (ASR) in multichannel, multi-speaker scenarios remains challenging due to ambient noise, reverberation and overlapping speakers. In this paper, we propose a beamforming approach that processes specific angular sectors based on their spherical polar coordinates before applying an end-to-end multichannel, multi-speaker ASR system. This method is data-independent and training-free. We demonstrate that using a group of beamformed signals improves ASR performance compared to using the same number of raw microphone signals. Moreover, increasing the number of signals used for beamforming further enhances recognition accuracy, leading to a more efficient use of multichannel signals while reducing the overall input load for the ASR system. We conduct experiments on the AMI meeting corpus, where the proposed method reduces word error rate by up to 11% and improves speaker counting accuracy by up to 27% relative compared to a multichannel ASR baseline system that does not exploit beamforming.
>
---
#### [new 003] DiTReducio: A Training-Free Acceleration for DiT-Based TTS via Progressive Calibration
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出DiTReducio，一种无需训练的DiT-TTS加速框架，通过时序跳过和分支跳过压缩计算，结合模式引导策略，在保持生成质量的同时显著降低计算量，提升实时性。属于语音合成任务，解决DiT模型计算开销大的问题。**

- **链接: [http://arxiv.org/pdf/2509.09748v1](http://arxiv.org/pdf/2509.09748v1)**

> **作者:** Yanru Huo; Ziyue Jiang; Zuoli Tang; Qingyang Hong; Zhou Zhao
>
> **摘要:** While Diffusion Transformers (DiT) have advanced non-autoregressive (NAR) speech synthesis, their high computational demands remain an limitation. Existing DiT-based text-to-speech (TTS) model acceleration approaches mainly focus on reducing sampling steps through distillation techniques, yet they remain constrained by training costs. We introduce DiTReducio, a training-free acceleration framework that compresses computations in DiT-based TTS models via progressive calibration. We propose two compression methods, Temporal Skipping and Branch Skipping, to eliminate redundant computations during inference. Moreover, based on two characteristic attention patterns identified within DiT layers, we devise a pattern-guided strategy to selectively apply the compression methods. Our method allows flexible modulation between generation quality and computational efficiency through adjustable compression thresholds. Experimental evaluations conducted on F5-TTS and MegaTTS 3 demonstrate that DiTReducio achieves a 75.4% reduction in FLOPs and improves the Real-Time Factor (RTF) by 37.1%, while preserving generation quality.
>
---
#### [new 004] CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出CoDiCodec音频自编码器，统一连续与离散压缩表示，解决高保真与高压缩比的矛盾。通过FSQ及新策略，实现11Hz连续嵌入与2.38kbps离散码流，提升重建质量与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.09836v1](http://arxiv.org/pdf/2509.09836v1)**

> **作者:** Marco Pasini; Stefan Lattner; George Fazekas
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Efficiently representing audio signals in a compressed latent space is critical for latent generative modelling. However, existing autoencoders often force a choice between continuous embeddings and discrete tokens. Furthermore, achieving high compression ratios while maintaining audio fidelity remains a challenge. We introduce CoDiCodec, a novel audio autoencoder that overcomes these limitations by both efficiently encoding global features via summary embeddings, and by producing both compressed continuous embeddings at ~ 11 Hz and discrete tokens at a rate of 2.38 kbps from the same trained model, offering unprecedented flexibility for different downstream generative tasks. This is achieved through Finite Scalar Quantization (FSQ) and a novel FSQ-dropout technique, and does not require additional loss terms beyond the single consistency loss used for end-to-end training. CoDiCodec supports both autoregressive decoding and a novel parallel decoding strategy, with the latter achieving superior audio quality and faster decoding. CoDiCodec outperforms existing continuous and discrete autoencoders at similar bitrates in terms of reconstruction audio quality. Our work enables a unified approach to audio compression, bridging the gap between continuous and discrete generative modelling paradigms.
>
---
#### [new 005] Prototypical Contrastive Learning For Improved Few-Shot Audio Classification
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频分类任务，旨在解决小样本学习中的数据不足问题。研究提出结合监督对比损失与原型网络的方法，并引入角度损失和自注意力机制，提升模型性能，在MetaAudio基准上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.10074v1](http://arxiv.org/pdf/2509.10074v1)**

> **作者:** Christos Sgouropoulos; Christos Nikou; Stefanos Vlachos; Vasileios Theiou; Christos Foukanelis; Theodoros Giannakopoulos
>
> **备注:** Accepted and Presented at IEEE International Workshop on Machine Learning for Signal Processing, Aug.\ 31-- Sep.\ 3, 2025, Istanbul, Turkey , 6 pages, 2 figures, 1 table
>
> **摘要:** Few-shot learning has emerged as a powerful paradigm for training models with limited labeled data, addressing challenges in scenarios where large-scale annotation is impractical. While extensive research has been conducted in the image domain, few-shot learning in audio classification remains relatively underexplored. In this work, we investigate the effect of integrating supervised contrastive loss into prototypical few shot training for audio classification. In detail, we demonstrate that angular loss further improves the performance compared to the standard contrastive loss. Our method leverages SpecAugment followed by a self-attention mechanism to encapsulate diverse information of augmented input versions into one unified embedding. We evaluate our approach on MetaAudio, a benchmark including five datasets with predefined splits, standardized preprocessing, and a comprehensive set of few-shot learning models for comparison. The proposed approach achieves state-of-the-art performance in a 5-way, 5-shot setting.
>
---
#### [new 006] Improving Audio Event Recognition with Consistency Regularization
- **分类: cs.SD; cs.AI**

- **简介: 论文提出将一致性正则化（CR）应用于音频事件识别任务，以提升模型性能。通过在AudioSet数据集上进行大量实验，验证了CR在有监督和半监督设置下的有效性，尤其在小规模训练集上表现突出。**

- **链接: [http://arxiv.org/pdf/2509.10391v1](http://arxiv.org/pdf/2509.10391v1)**

> **作者:** Shanmuka Sadhu; Weiran Wang
>
> **备注:** Under Review
>
> **摘要:** Consistency regularization (CR), which enforces agreement between model predictions on augmented views, has found recent benefits in automatic speech recognition [1]. In this paper, we propose the use of consistency regularization for audio event recognition, and demonstrate its effectiveness on AudioSet. With extensive ablation studies for both small ($\sim$20k) and large ($\sim$1.8M) supervised training sets, we show that CR brings consistent improvement over supervised baselines which already heavily utilize data augmentation, and CR using stronger augmentation and multiple augmentations leads to additional gain for the small training set. Furthermore, we extend the use of CR into the semi-supervised setup with 20K labeled samples and 1.8M unlabeled samples, and obtain performance improvement over our best model trained on the small set.
>
---
#### [new 007] Combining Textual and Spectral Features for Robust Classification of Pilot Communications
- **分类: cs.SD; cs.CY; eess.AS**

- **简介: 论文提出一种双通道机器学习框架，结合文本和频谱特征分类飞行员通信，解决非塔台机场航空操作识别难题。使用真实音频数据训练模型，取得超过91%的F1分数，方法具备可扩展性和低成本优势。**

- **链接: [http://arxiv.org/pdf/2509.09752v1](http://arxiv.org/pdf/2509.09752v1)**

> **作者:** Abdullah All Tanvir; Chenyu Huang; Moe Alahmad; Chuyang Yang; Xin Zhong
>
> **摘要:** Accurate estimation of aircraft operations, such as takeoffs and landings, is critical for effective airport management, yet remains challenging, especially at non-towered facilities lacking dedicated surveillance infrastructure. This paper presents a novel dual pipeline machine learning framework that classifies pilot radio communications using both textual and spectral features. Audio data collected from a non-towered U.S. airport was annotated by certified pilots with operational intent labels and preprocessed through automatic speech recognition and Mel-spectrogram extraction. We evaluate a wide range of traditional classifiers and deep learning models, including ensemble methods, LSTM, and CNN across both pipelines. To our knowledge, this is the first system to classify operational aircraft intent using a dual-pipeline ML framework on real-world air traffic audio. Our results demonstrate that spectral features combined with deep architectures consistently yield superior classification performance, with F1-scores exceeding 91%. Data augmentation further improves robustness to real-world audio variability. The proposed approach is scalable, cost-effective, and deployable without additional infrastructure, offering a practical solution for air traffic monitoring at general aviation airports.
>
---
#### [new 008] Testing chatbots on the creation of encoders for audio conditioned image generation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 论文测试聊天机器人设计音频编码器的能力，以替代图像生成模型中的文本编码器。研究发现，尽管聊天机器人能提出有效架构，但生成的音频嵌入未能与原始文本编码器对齐，仍需进一步改进。**

- **链接: [http://arxiv.org/pdf/2509.09717v1](http://arxiv.org/pdf/2509.09717v1)**

> **作者:** Jorge E. León; Miguel Carrasco
>
> **摘要:** On one hand, recent advances in chatbots has led to a rising popularity in using these models for coding tasks. On the other hand, modern generative image models primarily rely on text encoders to translate semantic concepts into visual representations, even when there is clear evidence that audio can be employed as input as well. Given the previous, in this work, we explore whether state-of-the-art conversational agents can design effective audio encoders to replace the CLIP text encoder from Stable Diffusion 1.5, enabling image synthesis directly from sound. We prompted five publicly available chatbots to propose neural architectures to work as these audio encoders, with a set of well-explained shared conditions. Each valid suggested encoder was trained on over two million context related audio-image-text observations, and evaluated on held-out validation and test sets using various metrics, together with a qualitative analysis of their generated images. Although almost all chatbots generated valid model designs, none achieved satisfactory results, indicating that their audio embeddings failed to align reliably with those of the original text encoder. Among the proposals, the Gemini audio encoder showed the best quantitative metrics, while the Grok audio encoder produced more coherent images (particularly, when paired with the text encoder). Our findings reveal a shared architectural bias across chatbots and underscore the remaining coding gap that needs to be bridged in future versions of these models. We also created a public demo so everyone could study and try out these audio encoders. Finally, we propose research questions that should be tackled in the future, and encourage other researchers to perform more focused and highly specialized tasks like this one, so the respective chatbots cannot make use of well-known solutions and their creativity/reasoning is fully tested.
>
---
#### [new 009] VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出VStyle基准，研究语音风格适应任务，解决SLMs根据口语指令调整说话风格的问题。构建中英文双语数据集，引入LALM评估框架，揭示当前模型在可控风格适应上的不足，推动人机自然交互发展。**

- **链接: [http://arxiv.org/pdf/2509.09716v1](http://arxiv.org/pdf/2509.09716v1)**

> **作者:** Jun Zhan; Mingyang Han; Yuxuan Xie; Chen Wang; Dong Zhang; Kexin Huang; Haoxiang Shi; DongXiao Wang; Tengtao Song; Qinyuan Cheng; Shimin Li; Jun Song; Xipeng Qiu; Bo Zheng
>
> **摘要:** Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{https://junzhan2000.github.io/VStyle.github.io/}{project's homepage}.
>
---
#### [new 010] SoilSound: Smartphone-based Soil Moisture Estimation
- **分类: cs.SD; cs.AI; cs.ET; cs.HC; eess.SP**

- **简介: 该论文提出SoilSound，一种基于智能手机的非侵入式土壤湿度检测系统。通过手机扬声器和麦克风进行声学扫描，利用卷积神经网络实现无需校准的土壤湿度估计，解决了传统方法需侵入性设备或专业仪器的问题。**

- **链接: [http://arxiv.org/pdf/2509.09823v1](http://arxiv.org/pdf/2509.09823v1)**

> **作者:** Yixuan Gao; Tanvir Ahmed; Shuang He; Zhongqi Cheng; Rajalakshmi Nandakumar
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Soil moisture monitoring is essential for agriculture and environmental management, yet existing methods require either invasive probes disturbing the soil or specialized equipment, limiting access to the public. We present SoilSound, an ubiquitous accessible smartphone-based acoustic sensing system that can measure soil moisture without disturbing the soil. We leverage the built-in speaker and microphone to perform a vertical scan mechanism to accurately measure moisture without any calibration. Unlike existing work that use transmissive properties, we propose an alternate model for acoustic reflections in soil based on the surface roughness effect to enable moisture sensing without disturbing the soil. The system works by sending acoustic chirps towards the soil and recording the reflections during a vertical scan, which are then processed and fed to a convolutional neural network for on-device soil moisture estimation with negligible computational, memory, or power overhead. We evaluated the system by training with curated soils in boxes in the lab and testing in the outdoor fields and show that SoilSound achieves a mean absolute error (MAE) of 2.39% across 10 different locations. Overall, the evaluation shows that SoilSound can accurately track soil moisture levels ranging from 15.9% to 34.0% across multiple soil types, environments, and users; without requiring any calibration or disturbing the soil, enabling widespread moisture monitoring for home gardeners, urban farmers, citizen scientists, and agricultural communities in resource-limited settings.
>
---
#### [new 011] Spectral Bottleneck in Deep Neural Networks: Noise is All You Need
- **分类: eess.AS; cs.SD**

- **简介: 论文研究深度神经网络中的频谱瓶颈问题，提出一种基于目标信号频谱特性的噪声初始化方法（WINNER），有效提升高频率信号的重建能力，应用于图像和音频任务，改善收敛速度与表示精度。属于信号重建与神经网络初始化任务。**

- **链接: [http://arxiv.org/pdf/2509.09719v1](http://arxiv.org/pdf/2509.09719v1)**

> **作者:** Hemanth Chandravamsi; Dhanush V. Shenoy; Itay Zinn; Shimon Pisnoy; Steven H. Frankel
>
> **摘要:** Deep neural networks are known to exhibit a spectral learning bias, wherein low-frequency components are learned early in training, while high-frequency modes emerge more gradually in later epochs. However, when the target signal lacks low-frequency components and is dominated by broadband high frequencies, training suffers from a 'spectral bottleneck', and the model fails to reconstruct the entire signal, including the frequency components that lie within the network's representational capacity. We examine such a scenario in the context of implicit neural representations (INRs) with sinusoidal representation networks (SIRENs), focusing on the challenge of fitting high-frequency-dominant signals that are susceptible to spectral bottleneck. To effectively fit any target signal irrespective of it's frequency content, we propose a generalized target-aware 'weight perturbation scheme' (WINNER - weight initialization with noise for neural representations) for network initialization. The scheme perturbs uniformly initialized weights with Gaussian noise, where the noise scales are adaptively determined by the spectral centroid of the target signal. We show that the noise scales can provide control over the spectra of network activations and the eigenbasis of the empirical neural tangent kernel. This method not only addresses the spectral bottleneck but also yields faster convergence and with improved representation accuracy, outperforming state-of-the-art approaches in audio fitting and achieving notable gains in image fitting and denoising tasks. Beyond signal reconstruction, our approach opens new directions for adaptive weight initialization strategies in computer vision and scientific machine learning.
>
---
#### [new 012] The MSP-Podcast Corpus
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MSP-Podcast语料库，用于提升真实场景下的语音情感识别。解决现有数据库规模小、情感平衡差的问题，构建了包含400小时多情感标注数据的高质量资源。**

- **链接: [http://arxiv.org/pdf/2509.09791v1](http://arxiv.org/pdf/2509.09791v1)**

> **作者:** Carlos Busso; Reza Lotfian; Kusha Sridhar; Ali N. Salman; Wei-Cheng Lin; Lucas Goncalves; Srinivas Parthasarathy; Abinay Reddy Naini; Seong-Gyun Leem; Luz Martinez-Lucas; Huang-Cheng Chou; Pravin Mote
>
> **备注:** IEEE Transactions on Affective Computing submission
>
> **摘要:** The availability of large, high-quality emotional speech databases is essential for advancing speech emotion recognition (SER) in real-world scenarios. However, many existing databases face limitations in size, emotional balance, and speaker diversity. This study describes the MSP-Podcast corpus, summarizing our ten-year effort. The corpus consists of over 400 hours of diverse audio samples from various audio-sharing websites, all of which have Common Licenses that permit the distribution of the corpus. We annotate the corpus with rich emotional labels, including primary (single dominant emotion) and secondary (multiple emotions perceived in the audio) emotional categories, as well as emotional attributes for valence, arousal, and dominance. At least five raters annotate these emotional labels. The corpus also has speaker identification for most samples, and human transcriptions of the lexical content of the sentences for the entire corpus. The data collection protocol includes a machine learning-driven pipeline for selecting emotionally diverse recordings, ensuring a balanced and varied representation of emotions across speakers and environments. The resulting database provides a comprehensive, high-quality resource, better suited for advancing SER systems in practical, real-world scenarios.
>
---
#### [new 013] Error Analysis in a Modular Meeting Transcription System
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究会议转录系统的误差分析，属于语音识别与说话人分离任务。旨在分析语音分离中的泄漏问题，并探讨不同分割方法对性能的影响。通过改进分割方法，提升了系统性能，达到LibriCSS数据集的最先进水平。**

- **链接: [http://arxiv.org/pdf/2509.10143v1](http://arxiv.org/pdf/2509.10143v1)**

> **作者:** Peter Vieting; Simon Berger; Thilo von Neumann; Christoph Boeddeker; Ralf Schlüter; Reinhold Haeb-Umbach
>
> **备注:** Accepted at ITG Conference on Speech Communication 2025
>
> **摘要:** Meeting transcription is a field of high relevance and remarkable progress in recent years. Still, challenges remain that limit its performance. In this work, we extend a previously proposed framework for analyzing leakage in speech separation with proper sensitivity to temporal locality. We show that there is significant leakage to the cross channel in areas where only the primary speaker is active. At the same time, the results demonstrate that this does not affect the final performance much as these leaked parts are largely ignored by the voice activity detection (VAD). Furthermore, different segmentations are compared showing that advanced diarization approaches are able to reduce the gap to oracle segmentation by a third compared to a simple energy-based VAD. We additionally reveal what factors contribute to the remaining difference. The results represent state-of-the-art performance on LibriCSS among systems that train the recognition module on LibriSpeech data only.
>
---
#### [new 014] TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation
- **分类: cs.IR; cs.AI; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出TalkPlayData 2，一个用于多模态对话音乐推荐的合成数据集。通过多个LLM代理生成对话数据，覆盖多种场景，并支持音频和图像模态。旨在解决生成式音乐推荐模型训练数据不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.09685v1](http://arxiv.org/pdf/2509.09685v1)**

> **作者:** Keunwoo Choi; Seungheon Doh; Juhan Nam
>
> **摘要:** We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In TalkPlayData 2 pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at https://talkpl.ai/talkplaydata2.html.
>
---
#### [new 015] Unified Learnable 2D Convolutional Feature Extraction for ASR
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统特征提取方法依赖性强的问题。提出一种统一的2D卷积前端，减少对经典方法的依赖，实现参数高效、性能匹配现有方法的通用特征提取器。**

- **链接: [http://arxiv.org/pdf/2509.10031v1](http://arxiv.org/pdf/2509.10031v1)**

> **作者:** Peter Vieting; Benedikt Hilmes; Ralf Schlüter; Hermann Ney
>
> **备注:** Accepted at ITG Conference on Speech Communication 2025
>
> **摘要:** Neural front-ends represent a promising approach to feature extraction for automatic speech recognition (ASR) systems as they enable to learn specifically tailored features for different tasks. Yet, many of the existing techniques remain heavily influenced by classical methods. While this inductive bias may ease the system design, our work aims to develop a more generic front-end for feature extraction. Furthermore, we seek to unify the front-end architecture contrasting with existing approaches that apply a composition of several layer topologies originating from different sources. The experiments systematically show how to reduce the influence of existing techniques to achieve a generic front-end. The resulting 2D convolutional front-end is parameter-efficient and suitable for a scenario with limited computational resources unlike large models pre-trained on unlabeled audio. The results demonstrate that this generic unified approach is not only feasible but also matches the performance of existing supervised learnable feature extractors.
>
---
## 更新

#### [replaced 001] IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2509.02622v2](http://arxiv.org/pdf/2509.02622v2)**

> **作者:** Clémentine Berger; Paraskevas Stamatiadis; Roland Badeau; Slim Essid
>
> **摘要:** We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics. To this end, we introduce IS${}^3$, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics.
>
---
#### [replaced 002] LoFi: Vision-Aided Label Generator for Wi-Fi Localization and Tracking
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.05074v4](http://arxiv.org/pdf/2412.05074v4)**

> **作者:** Zijian Zhao; Tingwei Chen; Fanyi Meng; Zhijie Cai; Hang Li; Xiaoyang Li; Guangxu Zhu
>
> **摘要:** Data-driven Wi-Fi localization and tracking have shown great promise due to their lower reliance on specialized hardware compared to model-based methods. However, most existing data collection techniques provide only coarse-grained ground truth or a limited number of labeled points, significantly hindering the advancement of data-driven approaches. While systems like lidar can deliver precise ground truth, their high costs make them inaccessible to many users. To address these challenges, we propose LoFi, a vision-aided label generator for Wi-Fi localization and tracking. LoFi can generate ground truth position coordinates solely from 2D images, offering high precision, low cost, and ease of use. Utilizing our method, we have compiled a Wi-Fi tracking and localization dataset using the ESP32-S3 and a webcam. The code and dataset of this paper are available at https://github.com/RS2002/LoFi.
>
---
#### [replaced 003] DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech
- **分类: cs.SD; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09631v2](http://arxiv.org/pdf/2509.09631v2)**

> **作者:** Ngoc-Son Nguyen; Hieu-Nghia Huynh-Nguyen; Thanh V. T. Tran; Truong-Son Hy; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines.
>
---
#### [replaced 004] Enhancing Speech Large Language Models with Prompt-Aware Mixture of Audio Encoders
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.15178v2](http://arxiv.org/pdf/2502.15178v2)**

> **作者:** Weiqiao Shan; Yuang Li; Yuhao Zhang; Yingfeng Luo; Chen Xu; Xiaofeng Zhao; Long Meng; Yunfei Lu; Min Zhang; Hao Yang; Tong Xiao; Jingbo Zhu
>
> **备注:** 16 pages,4 figures, 16 tables, to be published in EMNLP 2025 main conference
>
> **摘要:** Connecting audio encoders with large language models (LLMs) allows the LLM to perform various audio understanding tasks, such as automatic speech recognition (ASR) and audio captioning (AC). Most research focuses on training an adapter layer to generate a unified audio feature for the LLM. However, different tasks may require distinct features that emphasize either semantic or acoustic aspects, making task-specific audio features more desirable. In this paper, we propose Prompt-aware Mixture (PaM) to enhance the Speech LLM that uses multiple audio encoders. Our approach involves using different experts to extract different features based on the prompt that indicates different tasks. Experiments demonstrate that with PaM, only one Speech LLM surpasses the best performances achieved by all single-encoder Speech LLMs on ASR, Speaker Number Verification, and AC tasks. PaM also outperforms other feature fusion baselines, such as concatenation and averaging. Our code would be available at: https://github.com/shanweiqiao/PaM
>
---
#### [replaced 005] Towards Reliable Audio Deepfake Attribution and Model Recognition: A Multi-Level Autoencoder-Based Framework
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02521v3](http://arxiv.org/pdf/2508.02521v3)**

> **作者:** Andrea Di Pierno; Luca Guarnera; Dario Allegra; Sebastiano Battiato
>
> **摘要:** The proliferation of audio deepfakes poses a growing threat to trust in digital communications. While detection methods have advanced, attributing audio deepfakes to their source models remains an underexplored yet crucial challenge. In this paper we introduce LAVA (Layered Architecture for Voice Attribution), a hierarchical framework for audio deepfake detection and model recognition that leverages attention-enhanced latent representations extracted by a convolutional autoencoder trained solely on fake audio. Two specialized classifiers operate on these features: Audio Deepfake Attribution (ADA), which identifies the generation technology, and Audio Deepfake Model Recognition (ADMR), which recognize the specific generative model instance. To improve robustness under open-set conditions, we incorporate confidence-based rejection thresholds. Experiments on ASVspoof2021, FakeOrReal, and CodecFake show strong performance: the ADA classifier achieves F1-scores over 95% across all datasets, and the ADMR module reaches 96.31% macro F1 across six classes. Additional tests on unseen attacks from ASVpoof2019 LA and error propagation analysis confirm LAVA's robustness and reliability. The framework advances the field by introducing a supervised approach to deepfake attribution and model recognition under open-set conditions, validated on public benchmarks and accompanied by publicly released models and code. Models and code are available at https://www.github.com/adipiz99/lava-framework.
>
---
#### [replaced 006] Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.09550v2](http://arxiv.org/pdf/2509.09550v2)**

> **作者:** Harry Julian; Rachel Beeson; Lohith Konathala; Johanna Ulin; Jiameng Gao
>
> **摘要:** Neural Audio Codecs (NACs) have become increasingly adopted in speech processing tasks due to their excellent rate-distortion performance and compatibility with Large Language Models (LLMs) as discrete feature representations for audio generation. While most existing codecs rely on Residual Vector Quantization (RVQ), Finite Scalar Quantization (FSQ) has recently emerged as a compelling alternative that simplifies training and natively supports single codebooks. We introduce NeuCodec, an FSQ-based NAC, and show that FSQ encodes baked-in redundancy which produces an encoding which is robust when transmitted through noisy channels. First, through an encoder distillation experiment, we show that two different encoders can learn to encode identical audio into vastly different code sequences whilst maintaining comparable reconstruction quality with the same quantizer and decoder. Second, we demonstrate that FSQ has vastly superior bit-level perturbation robustness by comparing the performance of RVQ and FSQ codecs when simulating the transmission of code sequences through a noisy channel.
>
---
#### [replaced 007] Out-Of-Distribution Detection for Audio-visual Generalized Zero-Shot Learning: A General Framework
- **分类: cs.MM; cs.CV; cs.SD; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.01284v2](http://arxiv.org/pdf/2408.01284v2)**

> **作者:** Liuyuan Wen
>
> **备注:** Accepted to BMVC 2024
>
> **摘要:** Generalized Zero-Shot Learning (GZSL) is a challenging task requiring accurate classification of both seen and unseen classes. Within this domain, Audio-visual GZSL emerges as an extremely exciting yet difficult task, given the inclusion of both visual and acoustic features as multi-modal inputs. Existing efforts in this field mostly utilize either embedding-based or generative-based methods. However, generative training is difficult and unstable, while embedding-based methods often encounter domain shift problem. Thus, we find it promising to integrate both methods into a unified framework to leverage their advantages while mitigating their respective disadvantages. Our study introduces a general framework employing out-of-distribution (OOD) detection, aiming to harness the strengths of both approaches. We first employ generative adversarial networks to synthesize unseen features, enabling the training of an OOD detector alongside classifiers for seen and unseen classes. This detector determines whether a test feature belongs to seen or unseen classes, followed by classification utilizing separate classifiers for each feature type. We test our framework on three popular audio-visual datasets and observe a significant improvement comparing to existing state-of-the-art works. Codes can be found in https://github.com/liuyuan-wen/AV-OOD-GZSL.
>
---
