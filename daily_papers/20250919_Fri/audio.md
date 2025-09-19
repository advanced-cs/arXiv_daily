# 音频 cs.SD;  eess.SP

- **最新发布 34 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Estimating Respiratory Effort from Nocturnal Breathing Sounds for Obstructive Sleep Apnoea Screening
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出从夜间呼吸声中估计呼吸努力，用于阻塞性睡眠呼吸暂停筛查。通过融合音频与努力信号，提升检测性能，实现无需额外传感器的可扩展筛查方案。**

- **链接: [http://arxiv.org/pdf/2509.14944v1](http://arxiv.org/pdf/2509.14944v1)**

> **作者:** Xiaolei Xu; Chaoyue Niu; Guy J. Brown; Hector Romero; Ning Ma
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Obstructive sleep apnoea (OSA) is a prevalent condition with significant health consequences, yet many patients remain undiagnosed due to the complexity and cost of over-night polysomnography. Acoustic-based screening provides a scalable alternative, yet performance is limited by environmental noise and the lack of physiological context. Respiratory effort is a key signal used in clinical scoring of OSA events, but current approaches require additional contact sensors that reduce scalability and patient comfort. This paper presents the first study to estimate respiratory effort directly from nocturnal audio, enabling physiological context to be recovered from sound alone. We propose a latent-space fusion framework that integrates the estimated effort embeddings with acoustic features for OSA detection. Using a dataset of 157 nights from 103 participants recorded in home environments, our respiratory effort estimator achieves a concordance correlation coefficient of 0.48, capturing meaningful respiratory dynamics. Fusing effort and audio improves sensitivity and AUC over audio-only baselines, especially at low apnoea-hypopnoea index thresholds. The proposed approach requires only smartphone audio at test time, which enables sensor-free, scalable, and longitudinal OSA monitoring.
>
---
#### [new 002] MeanFlowSE: one-step generative speech enhancement via conditional mean flow
- **分类: cs.SD; cs.AI**

- **简介: 论文提出MeanFlowSE，一种单步生成语音增强模型，解决多步推理导致的实时性问题。通过学习轨迹上有限区间的平均速度，实现无需迭代求解器的单步生成，提升计算效率与语音质量。**

- **链接: [http://arxiv.org/pdf/2509.14858v1](http://arxiv.org/pdf/2509.14858v1)**

> **作者:** Duojia Li; Shenghui Lu; Hongchen Pan; Zongyi Zhan; Qingyang Hong; Lin Li
>
> **摘要:** Multistep inference is a bottleneck for real-time generative speech enhancement because flow- and diffusion-based systems learn an instantaneous velocity field and therefore rely on iterative ordinary differential equation (ODE) solvers. We introduce MeanFlowSE, a conditional generative model that learns the average velocity over finite intervals along a trajectory. Using a Jacobian-vector product (JVP) to instantiate the MeanFlow identity, we derive a local training objective that directly supervises finite-interval displacement while remaining consistent with the instantaneous-field constraint on the diagonal. At inference, MeanFlowSE performs single-step generation via a backward-in-time displacement, removing the need for multistep solvers; an optional few-step variant offers additional refinement. On VoiceBank-DEMAND, the single-step model achieves strong intelligibility, fidelity, and perceptual quality with substantially lower computational cost than multistep baselines. The method requires no knowledge distillation or external teachers, providing an efficient, high-fidelity framework for real-time generative speech enhancement.
>
---
#### [new 003] Back to Ear: Perceptually Driven High Fidelity Music Reconstruction
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出{\epsilon}ar-VAE模型，用于高保真音乐重建任务。针对现有模型在相位精度和立体声空间表现上的不足，引入感知滤波、新相位损失函数及谱监督方法，显著提升音频重建质量与空间特性。**

- **链接: [http://arxiv.org/pdf/2509.14912v1](http://arxiv.org/pdf/2509.14912v1)**

> **作者:** Kangdi Wang; Zhiyue Wu; Dinghao Zhou; Rui Lin; Junyu Dai; Tao Jiang
>
> **备注:** Check the Code here: https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE and Model Weights here: https://huggingface.co/earlab/EAR_VAE
>
> **摘要:** Variational Autoencoders (VAEs) are essential for large-scale audio tasks like diffusion-based generation. However, existing open-source models often neglect auditory perceptual aspects during training, leading to weaknesses in phase accuracy and stereophonic spatial representation. To address these challenges, we propose {\epsilon}ar-VAE, an open-source music signal reconstruction model that rethinks and optimizes the VAE training paradigm. Our contributions are threefold: (i) A K-weighting perceptual filter applied prior to loss calculation to align the objective with auditory perception. (ii) Two novel phase losses: a Correlation Loss for stereo coherence, and a Phase Loss using its derivatives--Instantaneous Frequency and Group Delay--for precision. (iii) A new spectral supervision paradigm where magnitude is supervised by all four Mid/Side/Left/Right components, while phase is supervised only by the LR components. Experiments show {\epsilon}ar-VAE at 44.1kHz substantially outperforms leading open-source models across diverse metrics, showing particular strength in reconstructing high-frequency harmonics and the spatial characteristics.
>
---
#### [new 004] Spatial-CLAP: Learning Spatially-Aware audio--text Embeddings for Multi-Source Conditions
- **分类: cs.SD**

- **简介: 该论文提出Spatial-CLAP，解决多声源条件下音频-文本嵌入的空间信息建模问题。通过引入内容感知空间编码器和空间对比学习策略，提升多声源场景下的嵌入效果，建立新的空间感知音频-文本嵌入范式。**

- **链接: [http://arxiv.org/pdf/2509.14785v1](http://arxiv.org/pdf/2509.14785v1)**

> **作者:** Kentaro Seki; Yuki Okamoto; Kouei Yamaoka; Yuki Saito; Shinnosuke Takamichi; Hiroshi Saruwatari
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Contrastive language--audio pretraining (CLAP) has achieved remarkable success as an audio--text embedding framework, but existing approaches are limited to monaural or single-source conditions and cannot fully capture spatial information. The central challenge in modeling spatial information lies in multi-source conditions, where the correct correspondence between each sound source and its location is required. To tackle this problem, we propose Spatial-CLAP, which introduces a content-aware spatial encoder that enables spatial representations coupled with audio content. We further propose spatial contrastive learning (SCL), a training strategy that explicitly enforces the learning of the correct correspondence and promotes more reliable embeddings under multi-source conditions. Experimental evaluations, including downstream tasks, demonstrate that Spatial-CLAP learns effective embeddings even under multi-source conditions, and confirm the effectiveness of SCL. Moreover, evaluation on unseen three-source mixtures highlights the fundamental distinction between conventional single-source training and our proposed multi-source training paradigm. These findings establish a new paradigm for spatially-aware audio--text embeddings.
>
---
#### [new 005] Pushing the Limits of End-to-End Diarization
- **分类: cs.SD**

- **简介: 该论文属于语音说话人分割任务，旨在解决多说话人场景下的语音分割问题。研究提出基于EEND-TA的统一模型，在多个数据集上取得新基准，显著降低错误率，提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.14737v1](http://arxiv.org/pdf/2509.14737v1)**

> **作者:** Samuel J. Broughton; Lahiru Samarakoon
>
> **备注:** As presented at Interspeech 2025
>
> **摘要:** In this paper, we present state-of-the-art diarization error rates (DERs) on multiple publicly available datasets, including AliMeeting-far, AliMeeting-near, AMI-Mix, AMI-SDM, DIHARD III, and MagicData RAMC. Leveraging EEND-TA, a single unified non-autoregressive model for end-to-end speaker diarization, we achieve new benchmark results, most notably a DER of 14.49% on DIHARD III. Our approach scales pretraining through 8-speaker simulation mixtures, ensuring each generated speaker mixture configuration is sufficiently represented. These experiments highlight that EEND-based architectures possess a greater capacity for learning than previously explored, surpassing many existing diarization solutions while maintaining efficient speeds during inference.
>
---
#### [new 006] Two Web Toolkits for Multimodal Piano Performance Dataset Acquisition and Fingering Annotation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 论文提出两个网络工具包PiaRec和ASDF，用于多模态钢琴表演数据的采集与指法标注，旨在解决大规模多模态数据获取困难的问题，提升数据集构建效率。**

- **链接: [http://arxiv.org/pdf/2509.15222v1](http://arxiv.org/pdf/2509.15222v1)**

> **作者:** Junhyung Park; Yonghyun Kim; Joonhyung Bae; Kirak Kim; Taegyun Kwon; Alexander Lerch; Juhan Nam
>
> **备注:** Accepted to the Late-Breaking Demo Session of the 26th International Society for Music Information Retrieval (ISMIR) Conference, 2025
>
> **摘要:** Piano performance is a multimodal activity that intrinsically combines physical actions with the acoustic rendition. Despite growing research interest in analyzing the multimodal nature of piano performance, the laborious process of acquiring large-scale multimodal data remains a significant bottleneck, hindering further progress in this field. To overcome this barrier, we present an integrated web toolkit comprising two graphical user interfaces (GUIs): (i) PiaRec, which supports the synchronized acquisition of audio, video, MIDI, and performance metadata. (ii) ASDF, which enables the efficient annotation of performer fingering from the visual data. Collectively, this system can streamline the acquisition of multimodal piano performance datasets.
>
---
#### [new 007] Doppler Radiance Field-Guided Antenna Selection for Improved Generalization in Multi-Antenna Wi-Fi-based Human Activity Recognition
- **分类: eess.SP; cs.CV**

- **简介: 论文提出基于多天线Wi-Fi的DoRF引导天线选择方法，解决HAR中CSI噪声和异步时钟影响问题，提升模型泛化能力。属于Wi-Fi感知任务，通过优化天线选择提高手势识别性能。**

- **链接: [http://arxiv.org/pdf/2509.15129v1](http://arxiv.org/pdf/2509.15129v1)**

> **作者:** Navid Hasanzadeh; Shahrokh Valaee
>
> **摘要:** With the IEEE 802.11bf Task Group introducing amendments to the WLAN standard for advanced sensing, interest in using Wi-Fi Channel State Information (CSI) for remote sensing has surged. Recent findings indicate that learning a unified three-dimensional motion representation through Doppler Radiance Fields (DoRFs) derived from CSI significantly improves the generalization capabilities of Wi-Fi-based human activity recognition (HAR). Despite this progress, CSI signals remain affected by asynchronous access point (AP) clocks and additive noise from environmental and hardware sources. Consequently, even with existing preprocessing techniques, both the CSI data and Doppler velocity projections used in DoRFs are still susceptible to noise and outliers, limiting HAR performance. To address this challenge, we propose a novel framework for multi-antenna APs to suppress noise and identify the most informative antennas based on DoRF fitting errors, which capture inconsistencies among Doppler velocity projections. Experimental results on a challenging small-scale hand gesture recognition dataset demonstrate that the proposed DoRF-guided Wi-Fi-based HAR approach significantly improves generalization capability, paving the way for robust real-world sensing deployments.
>
---
#### [new 008] Temporally Heterogeneous Graph Contrastive Learning for Multimodal Acoustic event Classification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多模态声事件分类任务，旨在解决音频与视觉信号对齐困难及跨模态噪声问题。提出THGCL框架，构建时序异构图，引入高斯过程与霍克斯过程建模时序依赖，通过对比学习提升分类性能。**

- **链接: [http://arxiv.org/pdf/2509.14893v1](http://arxiv.org/pdf/2509.14893v1)**

> **作者:** Yuanjian Chen; Yang Xiao; Jinjie Huang
>
> **摘要:** Multimodal acoustic event classification plays a key role in audio-visual systems. Although combining audio and visual signals improves recognition, it is still difficult to align them over time and to reduce the effect of noise across modalities. Existing methods often treat audio and visual streams separately, fusing features later with contrastive or mutual information objectives. Recent advances explore multimodal graph learning, but most fail to distinguish between intra- and inter-modal temporal dependencies. To address this, we propose Temporally Heterogeneous Graph-based Contrastive Learning (THGCL). Our framework constructs a temporal graph for each event, where audio and video segments form nodes and their temporal links form edges. We introduce Gaussian processes for intra-modal smoothness, Hawkes processes for inter-modal decay, and contrastive learning to capture fine-grained relationships. Experiments on AudioSet show that THGCL achieves state-of-the-art performance.
>
---
#### [new 009] Deploying UDM Series in Real-Life Stuttered Speech Applications: A Clinical Evaluation Framework
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出UDM系列框架，用于临床环境中检测口吃和言语不流畅。该研究解决AI模型准确性与临床可解释性之间的矛盾，通过模块化结构和可解释输出，实现高精度（F1:0.89）与临床认可（87%接受率），推动AI辅助言语治疗应用。**

- **链接: [http://arxiv.org/pdf/2509.14304v1](http://arxiv.org/pdf/2509.14304v1)**

> **作者:** Eric Zhang; Li Wei; Sarah Chen; Michael Wang
>
> **摘要:** Stuttered and dysfluent speech detection systems have traditionally suffered from the trade-off between accuracy and clinical interpretability. While end-to-end deep learning models achieve high performance, their black-box nature limits clinical adoption. This paper looks at the Unconstrained Dysfluency Modeling (UDM) series-the current state-of-the-art framework developed by Berkeley that combines modular architecture, explicit phoneme alignment, and interpretable outputs for real-world clinical deployment. Through extensive experiments involving patients and certified speech-language pathologists (SLPs), we demonstrate that UDM achieves state-of-the-art performance (F1: 0.89+-0.04) while providing clinically meaningful interpretability scores (4.2/5.0). Our deployment study shows 87% clinician acceptance rate and 34% reduction in diagnostic time. The results provide strong evidence that UDM represents a practical pathway toward AI-assisted speech therapy in clinical environments.
>
---
#### [new 010] Cross-Lingual F5-TTS: Towards Language-Agnostic Voice Cloning and Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文提出Cross-Lingual F5-TTS，解决流匹配TTS模型依赖音频提示文本的问题，实现无文本的跨语言语音克隆。通过强制对齐获取词边界，并训练语速预测器处理时长建模，实现高质量跨语言语音合成。**

- **链接: [http://arxiv.org/pdf/2509.14579v1](http://arxiv.org/pdf/2509.14579v1)**

> **作者:** Qingyu Liu; Yushen Chen; Zhikang Niu; Chunhui Wang; Yunting Yang; Bowen Zhang; Jian Zhao; Pengcheng Zhu; Kai Yu; Xie Chen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models have shown high-quality speech synthesis. However, most current flow-matching-based TTS models still rely on reference transcripts corresponding to the audio prompt for synthesis. This dependency prevents cross-lingual voice cloning when audio prompt transcripts are unavailable, particularly for unseen languages. The key challenges for flow-matching-based TTS models to remove audio prompt transcripts are identifying word boundaries during training and determining appropriate duration during inference. In this paper, we introduce Cross-Lingual F5-TTS, a framework that enables cross-lingual voice cloning without audio prompt transcripts. Our method preprocesses audio prompts by forced alignment to obtain word boundaries, enabling direct synthesis from audio prompts while excluding transcripts during training. To address the duration modeling challenge, we train speaking rate predictors at different linguistic granularities to derive duration from speaker pace. Experiments show that our approach matches the performance of F5-TTS while enabling cross-lingual voice cloning.
>
---
#### [new 011] How Does Instrumental Music Help SingFake Detection?
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文研究乐器音乐对 SingFake 检测的影响，属于语音伪造检测任务。通过分析模型行为与表示变化，发现乐器主要起数据增强作用，而非提供内在线索，并指出微调影响模型对语音特征的依赖性。**

- **链接: [http://arxiv.org/pdf/2509.14675v1](http://arxiv.org/pdf/2509.14675v1)**

> **作者:** Xuanjun Chen; Chia-Yu Hu; I-Ming Lin; Yi-Cheng Lin; I-Hsiang Chiu; You Zhang; Sung-Feng Huang; Yi-Hsuan Yang; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Work in progress
>
> **摘要:** Although many models exist to detect singing voice deepfakes (SingFake), how these models operate, particularly with instrumental accompaniment, is unclear. We investigate how instrumental music affects SingFake detection from two perspectives. To investigate the behavioral effect, we test different backbones, unpaired instrumental tracks, and frequency subbands. To analyze the representational effect, we probe how fine-tuning alters encoders' speech and music capabilities. Our results show that instrumental accompaniment acts mainly as data augmentation rather than providing intrinsic cues (e.g., rhythm or harmony). Furthermore, fine-tuning increases reliance on shallow speaker features while reducing sensitivity to content, paralinguistic, and semantic information. These insights clarify how models exploit vocal versus instrumental cues and can inform the design of more interpretable and robust SingFake detection systems.
>
---
#### [new 012] Measuring Soft Biometric Leakage in Speaker De-Identification Systems
- **分类: cs.SD**

- **简介: 该论文研究语音去标识化系统的软生物特征泄露问题，提出SBLS评分方法，评估系统对非唯一属性的抵抗能力，揭示现有系统在防止零样本攻击中的不足。**

- **链接: [http://arxiv.org/pdf/2509.14469v1](http://arxiv.org/pdf/2509.14469v1)**

> **作者:** Seungmin Seo; Oleg Aulov; P. Jonathon Phillips
>
> **摘要:** We use the term re-identification to refer to the process of recovering the original speaker's identity from anonymized speech outputs. Speaker de-identification systems aim to reduce the risk of re-identification, but most evaluations focus only on individual-level measures and overlook broader risks from soft biometric leakage. We introduce the Soft Biometric Leakage Score (SBLS), a unified method that quantifies resistance to zero-shot inference attacks on non-unique traits such as channel type, age range, dialect, sex of the speaker, or speaking style. SBLS integrates three elements: direct attribute inference using pre-trained classifiers, linkage detection via mutual information analysis, and subgroup robustness across intersecting attributes. Applying SBLS with publicly available classifiers, we show that all five evaluated de-identification systems exhibit significant vulnerabilities. Our results indicate that adversaries using only pre-trained models - without access to original speech or system details - can still reliably recover soft biometric information from anonymized output, exposing fundamental weaknesses that standard distributional metrics fail to capture.
>
---
#### [new 013] Spatial Audio Motion Understanding and Reasoning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于空间音频理解任务，旨在解决动态场景中移动声源的事件检测与空间属性推理问题。提出空间音频编码器和语义对齐模型，并结合大语言模型进行复杂查询回答，构建了相关数据集验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.14666v1](http://arxiv.org/pdf/2509.14666v1)**

> **作者:** Arvind Krishna Sridhar; Yinyi Guo; Erik Visser
>
> **备注:** 5 pages, 2 figures, 3 tables
>
> **摘要:** Spatial audio reasoning enables machines to interpret auditory scenes by understanding events and their spatial attributes. In this work, we focus on spatial audio understanding with an emphasis on reasoning about moving sources. First, we introduce a spatial audio encoder that processes spatial audio to detect multiple overlapping events and estimate their spatial attributes, Direction of Arrival (DoA) and source distance, at the frame level. To generalize to unseen events, we incorporate an audio grounding model that aligns audio features with semantic audio class text embeddings via a cross-attention mechanism. Second, to answer complex queries about dynamic audio scenes involving moving sources, we condition a large language model (LLM) on structured spatial attributes extracted by our model. Finally, we introduce a spatial audio motion understanding and reasoning benchmark dataset and demonstrate our framework's performance against the baseline model.
>
---
#### [new 014] Towards Building Speech Large Language Models for Multitask Understanding in Low-Resource Languages
- **分类: cs.SD**

- **简介: 该论文旨在构建支持多任务理解的低资源语言语音大模型。针对低资源语言（如泰语）中语音编码器性能差、对齐成本高及数据稀缺的问题，提出XLSR-Thai、U-Align和Thai-SUP方法，并开源相关资源以促进研究。**

- **链接: [http://arxiv.org/pdf/2509.14804v1](http://arxiv.org/pdf/2509.14804v1)**

> **作者:** Mingchen Shao; Bingshen Mu; Chengyou Wang; Hai Li; Ying Yan; Zhonghua Fu; Lei Xie
>
> **摘要:** Speech large language models (SLLMs) built on speech encoders, adapters, and LLMs demonstrate remarkable multitask understanding performance in high-resource languages such as English and Chinese. However, their effectiveness substantially degrades in low-resource languages such as Thai. This limitation arises from three factors: (1) existing commonly used speech encoders, like the Whisper family, underperform in low-resource languages and lack support for broader spoken language understanding tasks; (2) the ASR-based alignment paradigm requires training the entire SLLM, leading to high computational cost; (3) paired speech-text data in low-resource languages is scarce. To overcome these challenges in the low-resource language Thai, we introduce XLSR-Thai, the first self-supervised learning (SSL) speech encoder for Thai. It is obtained by continuously training the standard SSL XLSR model on 36,000 hours of Thai speech data. Furthermore, we propose U-Align, a speech-text alignment method that is more resource-efficient and multitask-effective than typical ASR-based alignment. Finally, we present Thai-SUP, a pipeline for generating Thai spoken language understanding data from high-resource languages, yielding the first Thai spoken language understanding dataset of over 1,000 hours. Multiple experiments demonstrate the effectiveness of our methods in building a Thai multitask-understanding SLLM. We open-source XLSR-Thai and Thai-SUP to facilitate future research.
>
---
#### [new 015] Explicit Context-Driven Neural Acoustic Modeling for High-Fidelity RIR Generation
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出MiNAF模型，用于高保真房间脉冲响应（RIR）生成。通过结合显式几何信息提升神经网络对声学场的建模能力，解决传统方法未有效利用环境几何特征的问题，提升RIR预测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.15210v1](http://arxiv.org/pdf/2509.15210v1)**

> **作者:** Chen Si; Qianyi Wu; Chaitanya Amballa; Romit Roy Choudhury
>
> **摘要:** Realistic sound simulation plays a critical role in many applications. A key element in sound simulation is the room impulse response (RIR), which characterizes how sound propagates from a source to a listener within a given space. Recent studies have applied neural implicit methods to learn RIR using context information collected from the environment, such as scene images. However, these approaches do not effectively leverage explicit geometric information from the environment. To further exploit the potential of neural implicit models with direct geometric features, we present Mesh-infused Neural Acoustic Field (MiNAF), which queries a rough room mesh at given locations and extracts distance distributions as an explicit representation of local context. Our approach demonstrates that incorporating explicit local geometric features can better guide the neural network in generating more accurate RIR predictions. Through comparisons with conventional and state-of-the-art baseline methods, we show that MiNAF performs competitively across various evaluation metrics. Furthermore, we verify the robustness of MiNAF in datasets with limited training samples, demonstrating an advance in high-fidelity sound simulation.
>
---
#### [new 016] Exploring How Audio Effects Alter Emotion with Foundation Models
- **分类: cs.SD; cs.AI**

- **简介: 论文研究音频效果对情绪的影响，利用基础模型分析其与情感的关联。任务是探索音频处理技术如何改变情绪感知，通过嵌入分析揭示非线性关系，提升音乐认知与情感计算的理解。**

- **链接: [http://arxiv.org/pdf/2509.15151v1](http://arxiv.org/pdf/2509.15151v1)**

> **作者:** Stelios Katsis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **摘要:** Audio effects (FX) such as reverberation, distortion, modulation, and dynamic range processing play a pivotal role in shaping emotional responses during music listening. While prior studies have examined links between low-level audio features and affective perception, the systematic impact of audio FX on emotion remains underexplored. This work investigates how foundation models - large-scale neural architectures pretrained on multimodal data - can be leveraged to analyze these effects. Such models encode rich associations between musical structure, timbre, and affective meaning, offering a powerful framework for probing the emotional consequences of sound design techniques. By applying various probing methods to embeddings from deep learning models, we examine the complex, nonlinear relationships between audio FX and estimated emotion, uncovering patterns tied to specific effects and evaluating the robustness of foundation audio models. Our findings aim to advance understanding of the perceptual impact of audio production practices, with implications for music cognition, performance, and affective computing.
>
---
#### [new 017] A long-form single-speaker real-time MRI speech dataset and benchmark
- **分类: cs.SD**

- **简介: 该论文发布了一个包含实时MRI视频和音频的单说话人数据集，用于语音生成与识别任务。解决了真实语音生产过程中发音器官动态与声学信号同步研究的问题，提供了数据预处理和基准测试结果。**

- **链接: [http://arxiv.org/pdf/2509.14479v1](http://arxiv.org/pdf/2509.14479v1)**

> **作者:** Sean Foley; Jihwan Lee; Kevin Huang; Xuan Shi; Yoonjeong Lee; Louis Goldstein; Shrikanth Narayanan
>
> **摘要:** We release the USC Long Single-Speaker (LSS) dataset containing real-time MRI video of the vocal tract dynamics and simultaneous audio obtained during speech production. This unique dataset contains roughly one hour of video and audio data from a single native speaker of American English, making it one of the longer publicly available single-speaker datasets of real-time MRI speech data. Along with the articulatory and acoustic raw data, we release derived representations of the data that are suitable for a range of downstream tasks. This includes video cropped to the vocal tract region, sentence-level splits of the data, restored and denoised audio, and regions-of-interest timeseries. We also benchmark this dataset on articulatory synthesis and phoneme recognition tasks, providing baseline performance for these tasks on this dataset which future research can aim to improve upon.
>
---
#### [new 018] FCPE: A Fast Context-based Pitch Estimation Model
- **分类: cs.SD; cs.CL**

- **简介: 论文提出FCPE模型，用于单声道音频的音高估计任务，旨在解决噪声环境下性能下降的问题。该模型采用Lynx-Net结构和深度可分离卷积，在保证低计算成本的同时提升鲁棒性，实验显示其在MIR-1K数据集上表现优异且效率高。**

- **链接: [http://arxiv.org/pdf/2509.15140v1](http://arxiv.org/pdf/2509.15140v1)**

> **作者:** Yuxin Luo; Ruoyi Zhang; Lu-Chuan Liu; Tianyu Li; Hangyu Liu
>
> **备注:** Under review
>
> **摘要:** Pitch estimation (PE) in monophonic audio is crucial for MIDI transcription and singing voice conversion (SVC), but existing methods suffer significant performance degradation under noise. In this paper, we propose FCPE, a fast context-based pitch estimation model that employs a Lynx-Net architecture with depth-wise separable convolutions to effectively capture mel spectrogram features while maintaining low computational cost and robust noise tolerance. Experiments show that our method achieves 96.79\% Raw Pitch Accuracy (RPA) on the MIR-1K dataset, on par with the state-of-the-art methods. The Real-Time Factor (RTF) is 0.0062 on a single RTX 4090 GPU, which significantly outperforms existing algorithms in efficiency. Code is available at https://github.com/CNChTu/FCPE.
>
---
#### [new 019] Efficient Solutions for Mitigating Initialization Bias in Unsupervised Self-Adaptive Auditory Attention Decoding
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于无监督自适应听觉注意力解码任务，旨在解决初始化偏差问题。提出三种计算高效的替代方法，在保持性能的同时显著降低计算复杂度。**

- **链接: [http://arxiv.org/pdf/2509.14764v1](http://arxiv.org/pdf/2509.14764v1)**

> **作者:** Yuanyuan Yao; Simon Geirnaert; Tinne Tuytelaars; Alexander Bertrand
>
> **摘要:** Decoding the attended speaker in a multi-speaker environment from electroencephalography (EEG) has attracted growing interest in recent years, with neuro-steered hearing devices as a driver application. Current approaches typically rely on ground-truth labels of the attended speaker during training, necessitating calibration sessions for each user and each EEG set-up to achieve optimal performance. While unsupervised self-adaptive auditory attention decoding (AAD) for stimulus reconstruction has been developed to eliminate the need for labeled data, it suffers from an initialization bias that can compromise performance. Although an unbiased variant has been proposed to address this limitation, it introduces substantial computational complexity that scales with data size. This paper presents three computationally efficient alternatives that achieve comparable performance, but with a significantly lower and constant computational cost. The code for the proposed algorithms is available at https://github.com/YYao-42/Unsupervised_AAD.
>
---
#### [new 020] From Hype to Insight: Rethinking Large Language Model Integration in Visual Speech Recognition
- **分类: cs.SD**

- **简介: 该论文研究大语言模型（LLM）在视觉语音识别（VSR）中的应用。任务是评估LLM解码器对VSR性能的影响，分析其提升是否源于语言建模而非视觉理解。通过实验发现，LLM主要提升语境推理，而非视觉特征提取，强调需更强视觉编码器推动进展。**

- **链接: [http://arxiv.org/pdf/2509.14880v1](http://arxiv.org/pdf/2509.14880v1)**

> **作者:** Rishabh Jain; Naomi Harte
>
> **备注:** submitted to ICASSP 2026. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Advances in self-supervised encoders have improved Visual Speech Recognition (VSR). Recent approaches integrating these encoders with LLM decoders improves transcription accuracy; however, it remains unclear whether these gains stem from visual understanding or stronger language modeling. In this work, we systematically evaluate LLM decoders by freezing or selectively updating the visual encoder, scaling decoder size, comparing adaptation strategies and architectures, and varying training data across LRS2, LRS3, and their combination. Evaluation on LRS2, LRS3, and WildVSR shows that scaling and adaptation yield limited improvements, while combining datasets enhances generalization. Semantic analysis reveals that gains arise primarily from lexical rather than semantic processing. Our Llama-2-13B model trained on the combined set achieves 24.7\% WER on LRS3 and 47.0\% on WildVSR, establishing SOTA among models trained without additional supervision. Our findings indicate LLM decoders refine contextual reasoning rather than visual features, emphasizing the need for stronger visual encoders to drive meaningful progress.
>
---
#### [new 021] SpeechMLC: Speech Multi-label Classification
- **分类: eess.AS; eess.SP**

- **简介: 论文提出SpeechMLC框架，用于语音多标签分类任务，解决检测语音样本中多种说话风格的问题。通过集成交叉注意力机制和数据增强技术，提升模型在不平衡数据下的性能，并分析人类感知对分类准确性的影响。**

- **链接: [http://arxiv.org/pdf/2509.14677v1](http://arxiv.org/pdf/2509.14677v1)**

> **作者:** Miseul Kim; Seyun Um; Hyeonjin Cha; Hong-goo Kang
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this paper, we propose a multi-label classification framework to detect multiple speaking styles in a speech sample. Unlike previous studies that have primarily focused on identifying a single target style, our framework effectively captures various speaker characteristics within a unified structure, making it suitable for generalized human-computer interaction applications. The proposed framework integrates cross-attention mechanisms within a transformer decoder to extract salient features associated with each target label from the input speech. To mitigate the data imbalance inherent in multi-label speech datasets, we employ a data augmentation technique based on a speech generation model. We validate our model's effectiveness through multiple objective evaluations on seen and unseen corpora. In addition, we provide an analysis of the influence of human perception on classification accuracy by considering the impact of human labeling agreement on model performance.
>
---
#### [new 022] Music4All A+A: A Multimodal Dataset for Music Information Retrieval Tasks
- **分类: cs.MM; cs.IR; cs.SD**

- **简介: 该论文提出Music4All A+A数据集，用于多模态音乐信息检索任务，解决现有数据集粒度不足的问题。该数据集基于艺术家和专辑构建，包含元数据、图像和文本描述，支持多粒度MIR任务，并展示了其在跨域分类中的应用。**

- **链接: [http://arxiv.org/pdf/2509.14891v1](http://arxiv.org/pdf/2509.14891v1)**

> **作者:** Jonas Geiger; Marta Moscati; Shah Nawaz; Markus Schedl
>
> **备注:** 7 pages, 6 tables, IEEE International Conference on Content-Based Multimedia Indexing (IEEE CBMI)
>
> **摘要:** Music is characterized by aspects related to different modalities, such as the audio signal, the lyrics, or the music video clips. This has motivated the development of multimodal datasets and methods for Music Information Retrieval (MIR) tasks such as genre classification or autotagging. Music can be described at different levels of granularity, for instance defining genres at the level of artists or music albums. However, most datasets for multimodal MIR neglect this aspect and provide data at the level of individual music tracks. We aim to fill this gap by providing Music4All Artist and Album (Music4All A+A), a dataset for multimodal MIR tasks based on music artists and albums. Music4All A+A is built on top of the Music4All-Onion dataset, an existing track-level dataset for MIR tasks. Music4All A+A provides metadata, genre labels, image representations, and textual descriptors for 6,741 artists and 19,511 albums. Furthermore, since Music4All A+A is built on top of Music4All-Onion, it allows access to other multimodal data at the track level, including user--item interaction data. This renders Music4All A+A suitable for a broad range of MIR tasks, including multimodal music recommendation, at several levels of granularity. To showcase the use of Music4All A+A, we carry out experiments on multimodal genre classification of artists and albums, including an analysis in missing-modality scenarios, and a quantitative comparison with genre classification in the movie domain. Our experiments show that images are more informative for classifying the genres of artists and albums, and that several multimodal models for genre classification struggle in generalizing across domains. We provide the code to reproduce our experiments at https://github.com/hcai-mms/Music4All-A-A, the dataset is linked in the repository and provided open-source under a CC BY-NC-SA 4.0 license.
>
---
#### [new 023] Real-Time Streaming Mel Vocoding with Generative Flow Matching
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出MelFlow，一种基于生成流匹配的实时流式Mel声码器，解决传统Mel声码器延迟高、非流式的问题。实现了32 ms算法延迟和48 ms总延迟，并在消费级GPU上实现实时流式处理，性能优于HiFi-GAN等非流式基线。**

- **链接: [http://arxiv.org/pdf/2509.15085v1](http://arxiv.org/pdf/2509.15085v1)**

> **作者:** Simon Welker; Tal Peer; Timo Gerkmann
>
> **备注:** (C) 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** The task of Mel vocoding, i.e., the inversion of a Mel magnitude spectrogram to an audio waveform, is still a key component in many text-to-speech (TTS) systems today. Based on generative flow matching, our prior work on generative STFT phase retrieval (DiffPhase), and the pseudoinverse operator of the Mel filterbank, we develop MelFlow, a streaming-capable generative Mel vocoder for speech sampled at 16 kHz with an algorithmic latency of only 32 ms and a total latency of 48 ms. We show real-time streaming capability at this latency not only in theory, but in practice on a consumer laptop GPU. Furthermore, we show that our model achieves substantially better PESQ and SI-SDR values compared to well-established not streaming-capable baselines for Mel vocoding including HiFi-GAN.
>
---
#### [new 024] DAIEN-TTS: Disentangled Audio Infilling for Environment-Aware Text-to-Speech Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出DAIEN-TTS，一种零样本文本到语音合成框架，解决环境感知语音合成问题。通过分离语音与环境音，实现对音色和背景环境的独立控制，提升语音自然度与环境真实感。**

- **链接: [http://arxiv.org/pdf/2509.14684v1](http://arxiv.org/pdf/2509.14684v1)**

> **作者:** Ye-Xin Lu; Yu Gu; Kun Wei; Hui-Peng Du; Yang Ai; Zhen-Hua Ling
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** This paper presents DAIEN-TTS, a zero-shot text-to-speech (TTS) framework that enables ENvironment-aware synthesis through Disentangled Audio Infilling. By leveraging separate speaker and environment prompts, DAIEN-TTS allows independent control over the timbre and the background environment of the synthesized speech. Built upon F5-TTS, the proposed DAIEN-TTS first incorporates a pretrained speech-environment separation (SES) module to disentangle the environmental speech into mel-spectrograms of clean speech and environment audio. Two random span masks of varying lengths are then applied to both mel-spectrograms, which, together with the text embedding, serve as conditions for infilling the masked environmental mel-spectrogram, enabling the simultaneous continuation of personalized speech and time-varying environmental audio. To further enhance controllability during inference, we adopt dual class-free guidance (DCFG) for the speech and environment components and introduce a signal-to-noise ratio (SNR) adaptation strategy to align the synthesized speech with the environment prompt. Experimental results demonstrate that DAIEN-TTS generates environmental personalized speech with high naturalness, strong speaker similarity, and high environmental fidelity.
>
---
#### [new 025] Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 论文针对文本到音频生成模型中的数据复制问题，提出抗记忆引导（AMG）策略，通过修改采样过程减少模型对训练数据的复现，同时保持生成质量。属于生成模型优化任务。**

- **链接: [http://arxiv.org/pdf/2509.14934v1](http://arxiv.org/pdf/2509.14934v1)**

> **作者:** Francisco Messina; Francesca Ronchini; Luca Comanducci; Paolo Bestagini; Fabio Antonacci
>
> **摘要:** A persistent challenge in generative audio models is data replication, where the model unintentionally generates parts of its training data during inference. In this work, we address this issue in text-to-audio diffusion models by exploring the use of anti-memorization strategies. We adopt Anti-Memorization Guidance (AMG), a technique that modifies the sampling process of pre-trained diffusion models to discourage memorization. Our study explores three types of guidance within AMG, each designed to reduce replication while preserving generation quality. We use Stable Audio Open as our backbone, leveraging its fully open-source architecture and training dataset. Our comprehensive experimental analysis suggests that AMG significantly mitigates memorization in diffusion-based text-to-audio generation without compromising audio fidelity or semantic alignment.
>
---
#### [new 026] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS; I.2.7**

- **简介: 论文提出SpeechWeave，用于生成多语言、领域特定的合成文本与语音数据，以解决TTS训练中数据多样性不足、标准化困难及语音一致性问题，提升数据质量与生成效率。**

- **链接: [http://arxiv.org/pdf/2509.14270v1](http://arxiv.org/pdf/2509.14270v1)**

> **作者:** Karan Dua; Puneet Mittal; Ranjeet Gupta; Hitesh Laxmichand Patel
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** High-quality Text-to-Speech (TTS) model training requires extensive and diverse text and speech data. It is challenging to procure such data from real sources due to issues of domain specificity, licensing, and scalability. Large language models (LLMs) can certainly generate textual data, but they create repetitive text with insufficient variation in the prompt during the generation process. Another important aspect in TTS training data is text normalization. Tools for normalization might occasionally introduce anomalies or overlook valuable patterns, and thus impact data quality. Furthermore, it is also impractical to rely on voice artists for large scale speech recording in commercial TTS systems with standardized voices. To address these challenges, we propose SpeechWeave, a synthetic speech data generation pipeline that is capable of automating the generation of multilingual, domain-specific datasets for training TTS models. Our experiments reveal that our pipeline generates data that is 10-48% more diverse than the baseline across various linguistic and phonetic metrics, along with speaker-standardized speech audio while generating approximately 97% correctly normalized text. Our approach enables scalable, high-quality data generation for TTS training, improving diversity, normalization, and voice consistency in the generated datasets.
>
---
#### [new 027] CLAIP-Emo: Parameter-Efficient Adaptation of Language-supervised models for In-the-Wild Audiovisual Emotion Recognition
- **分类: cs.MM; cs.SD**

- **简介: 论文提出CLAIP-Emo，用于真实场景下的音频视觉情绪识别（AVER）。针对姿态变化、遮挡和背景噪声等问题，采用参数高效的适配方法，基于CLIP/CLAP模型进行微调，实现高精度情绪识别。**

- **链接: [http://arxiv.org/pdf/2509.14527v1](http://arxiv.org/pdf/2509.14527v1)**

> **作者:** Yin Chen; Jia Li; Jinpeng Hu; Zhenzhen Hu; Richang Hong
>
> **备注:** The code and models will be available at https://github.com/MSA-LMC/CLAIP-Emo
>
> **摘要:** Audiovisual emotion recognition (AVER) in the wild is still hindered by pose variation, occlusion, and background noise. Prevailing methods primarily rely on large-scale domain-specific pre-training, which is costly and often mismatched to real-world affective data. To address this, we present CLAIP-Emo, a modular framework that reframes in-the-wild AVER as a parameter-efficient adaptation of language-supervised foundation models (CLIP/CLAP). Specifically, it (i) preserves language-supervised priors by freezing CLIP/CLAP backbones and performing emotion-oriented adaptation via LoRA (updating \ensuremath{\le}4.0\% of the total parameters), (ii) allocates temporal modeling asymmetrically, employing a lightweight Transformer for visual dynamics while applying mean pooling for audio prosody, and (iii) applies a simple fusion head for prediction. On DFEW and MAFW, CLAIP-Emo (ViT-L/14) achieves 80.14\% and 61.18\% weighted average recall with only 8M training parameters, setting a new state of the art. Our findings suggest that parameter-efficient adaptation of language-supervised foundation models provides a scalable alternative to domain-specific pre-training for real-world AVER. The code and models will be available at \href{https://github.com/MSA-LMC/CLAIP-Emo}{https://github.com/MSA-LMC/CLAIP-Emo}.
>
---
#### [new 028] Acoustic Simulation Framework for Multi-channel Replay Speech Detection
- **分类: eess.AS; cs.CR; cs.SD; eess.SP**

- **简介: 该论文提出一个声学仿真框架，用于生成多通道重放语音数据，以提升语音检测系统的鲁棒性。任务是解决单通道数据限制下的重放攻击检测问题，通过模拟真实环境增强检测模型的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.14789v1](http://arxiv.org/pdf/2509.14789v1)**

> **作者:** Michael Neri; Tuomas Virtanen
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Replay speech attacks pose a significant threat to voice-controlled systems, especially in smart environments where voice assistants are widely deployed. While multi-channel audio offers spatial cues that can enhance replay detection robustness, existing datasets and methods predominantly rely on single-channel recordings. In this work, we introduce an acoustic simulation framework designed to simulate multi-channel replay speech configurations using publicly available resources. Our setup models both genuine and spoofed speech across varied environments, including realistic microphone and loudspeaker impulse responses, room acoustics, and noise conditions. The framework employs measured loudspeaker directionalities during the replay attack to improve the realism of the simulation. We define two spoofing settings, which simulate whether a reverberant or an anechoic speech is used in the replay scenario, and evaluate the impact of omnidirectional and diffuse noise on detection performance. Using the state-of-the-art M-ALRAD model for replay speech detection, we demonstrate that synthetic data can support the generalization capabilities of the detector across unseen enclosures.
>
---
#### [new 029] From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文综述了全双工语音模型（FD-SLMs），旨在实现自然的人机对话。论文分类模型架构，统一评估框架，分析挑战，为提升人机同步交流提供方向。**

- **链接: [http://arxiv.org/pdf/2509.14515v1](http://arxiv.org/pdf/2509.14515v1)**

> **作者:** Yuxuan Chen; Haoyuan Yu
>
> **摘要:** True Full-Duplex (TFD) voice communication--enabling simultaneous listening and speaking with natural turn-taking, overlapping speech, and interruptions--represents a critical milestone toward human-like AI interaction. This survey comprehensively reviews Full-Duplex Spoken Language Models (FD-SLMs) in the LLM era. We establish a taxonomy distinguishing Engineered Synchronization (modular architectures) from Learned Synchronization (end-to-end architectures), and unify fragmented evaluation approaches into a framework encompassing Temporal Dynamics, Behavioral Arbitration, Semantic Coherence, and Acoustic Performance. Through comparative analysis of mainstream FD-SLMs, we identify fundamental challenges: synchronous data scarcity, architectural divergence, and evaluation gaps, providing a roadmap for advancing human-AI communication.
>
---
#### [new 030] BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出BabyHuBERT，一种基于多语言儿童长时录音的自监督语音模型，用于解决儿童语音中说话人分割任务。其旨在提升对儿童与成人语音区分的能力，尤其在资源匮乏语言中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.15001v1](http://arxiv.org/pdf/2509.15001v1)**

> **作者:** Théo Charlot; Tarek Kunze; Maxime Poli; Alejandrina Cristia; Emmanuel Dupoux; Marvin Lavechin
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Child-centered long-form recordings are essential for studying early language development, but existing speech models trained on clean adult data perform poorly due to acoustic and linguistic differences. We introduce BabyHuBERT, the first self-supervised speech representation model trained on 13,000 hours of multilingual child-centered long-form recordings spanning over 40 languages. We evaluate BabyHuBERT on speaker segmentation, identifying when target children speak versus female adults, male adults, or other children -- a fundamental preprocessing step for analyzing naturalistic language experiences. BabyHuBERT achieves F1-scores from 52.1% to 74.4% across six diverse datasets, consistently outperforming W2V2-LL4300 (trained on English long-forms) and standard HuBERT (trained on clean adult speech). Notable improvements include 13.2 absolute F1 points over HuBERT on Vanuatu and 15.9 points on Solomon Islands corpora, demonstrating effectiveness on underrepresented languages. By sharing code and models, BabyHuBERT serves as a foundation model for child speech research, enabling fine-tuning on diverse downstream tasks.
>
---
#### [new 031] Aligning Audio Captions with Human Preferences
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音频字幕生成任务，旨在解决现有系统依赖昂贵标注数据且不符合人类偏好问题。提出基于RLHF的框架，利用人类偏好数据训练奖励模型，无需真实标注即可优化字幕生成，提升与人类偏好的对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.14659v1](http://arxiv.org/pdf/2509.14659v1)**

> **作者:** Kartik Hegde; Rehana Mahfuz; Yinyi Guo; Erik Visser
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Current audio captioning systems rely heavily on supervised learning with paired audio-caption datasets, which are expensive to curate and may not reflect human preferences in real-world scenarios. To address this limitation, we propose a preference-aligned audio captioning framework based on Reinforcement Learning from Human Feedback (RLHF). To effectively capture nuanced human preferences, we train a Contrastive Language-Audio Pretraining (CLAP)-based reward model using human-labeled pairwise preference data. This reward model is integrated into a reinforcement learning framework to fine-tune any baseline captioning system without relying on ground-truth caption annotations. Extensive human evaluations across multiple datasets show that our method produces captions preferred over those from baseline models, particularly in cases where the baseline models fail to provide correct and natural captions. Furthermore, our framework achieves performance comparable to supervised approaches with ground-truth data, demonstrating its effectiveness in aligning audio captioning with human preferences and its scalability in real-world scenarios.
>
---
#### [new 032] MMED: A Multimodal Micro-Expression Dataset based on Audio-Visual Fusion
- **分类: cs.MM; cs.SD**

- **简介: 该论文提出MMED数据集和AMF-Net方法，用于微表情识别任务。传统研究依赖视觉数据，而本文通过融合音频与视觉信息，提升微表情分析准确性，解决单一模态信息不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.14592v1](http://arxiv.org/pdf/2509.14592v1)**

> **作者:** Junbo Wang; Yan Zhao; Shuo Li; Shibo Wang; Shigang Wang; Jian Wei
>
> **摘要:** Micro-expressions (MEs) are crucial leakages of concealed emotion, yet their study has been constrained by a reliance on silent, visual-only data. To solve this issue, we introduce two principal contributions. First, MMED, to our knowledge, is the first dataset capturing the spontaneous vocal cues that co-occur with MEs in ecologically valid, high-stakes interactions. Second, the Asymmetric Multimodal Fusion Network (AMF-Net) is a novel method that effectively fuses a global visual summary with a dynamic audio sequence via an asymmetric cross-attention framework. Rigorous Leave-One-Subject-Out Cross-Validation (LOSO-CV) experiments validate our approach, providing conclusive evidence that audio offers critical, disambiguating information for ME analysis. Collectively, the MMED dataset and our AMF-Net method provide valuable resources and a validated analytical approach for micro-expression recognition.
>
---
#### [new 033] Mitigating Intra-Speaker Variability in Diarization with Style-Controllable Speech Augmentation
- **分类: eess.AS; cs.AI; eess.SP**

- **简介: 论文提出一种风格可控的语音增强方法，用于缓解说话人分割中的内在说话人变化问题。通过生成多样化风格的语音并融合嵌入，提升系统鲁棒性，在两个数据集上分别降低49%和35%错误率。属于说话人分割任务。**

- **链接: [http://arxiv.org/pdf/2509.14632v1](http://arxiv.org/pdf/2509.14632v1)**

> **作者:** Miseul Kim; Soo Jin Park; Kyungguen Byun; Hyeon-Kyeong Shin; Sunkuk Moon; Shuhua Zhang; Erik Visser
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speaker diarization systems often struggle with high intrinsic intra-speaker variability, such as shifts in emotion, health, or content. This can cause segments from the same speaker to be misclassified as different individuals, for example, when one raises their voice or speaks faster during conversation. To address this, we propose a style-controllable speech generation model that augments speech across diverse styles while preserving the target speaker's identity. The proposed system starts with diarized segments from a conventional diarizer. For each diarized segment, it generates augmented speech samples enriched with phonetic and stylistic diversity. And then, speaker embeddings from both the original and generated audio are blended to enhance the system's robustness in grouping segments with high intrinsic intra-speaker variability. We validate our approach on a simulated emotional speech dataset and the truncated AMI dataset, demonstrating significant improvements, with error rate reductions of 49% and 35% on each dataset, respectively.
>
---
#### [new 034] Multi-Channel Differential ASR for Robust Wearer Speech Recognition on Smart Glasses
- **分类: eess.AS; cs.SD**

- **简介: 论文提出一种多通道差分ASR方法，用于提升智能眼镜上佩戴者语音识别的鲁棒性。针对侧谈干扰问题，融合波束成形、麦克风选择和轻量级侧谈检测模型，有效降低词错误率，提升识别性能。属于语音识别任务。**

- **链接: [http://arxiv.org/pdf/2509.14430v1](http://arxiv.org/pdf/2509.14430v1)**

> **作者:** Yufeng Yang; Yiteng Huang; Yong Xu; Li Wan; Suwon Shon; Yang Liu; Yifeng Fan; Zhaojun Yang; Olivier Siohan; Yue Liu; Ming Sun; Florian Metze
>
> **摘要:** With the growing adoption of wearable devices such as smart glasses for AI assistants, wearer speech recognition (WSR) is becoming increasingly critical to next-generation human-computer interfaces. However, in real environments, interference from side-talk speech remains a significant challenge to WSR and may cause accumulated errors for downstream tasks such as natural language processing. In this work, we introduce a novel multi-channel differential automatic speech recognition (ASR) method for robust WSR on smart glasses. The proposed system takes differential inputs from different frontends that complement each other to improve the robustness of WSR, including a beamformer, microphone selection, and a lightweight side-talk detection model. Evaluations on both simulated and real datasets demonstrate that the proposed system outperforms the traditional approach, achieving up to an 18.0% relative reduction in word error rate.
>
---
## 更新

#### [replaced 001] GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.18309v2](http://arxiv.org/pdf/2502.18309v2)**

> **作者:** Xinran Liu; Xu Dong; Diptesh Kanojia; Wenwu Wang; Zhenhua Feng
>
> **摘要:** Generating high-quality full-body dance sequences from music is a challenging task as it requires strict adherence to genre-specific choreography. Moreover, the generated sequences must be both physically realistic and precisely synchronized with the beats and rhythm of the music. To overcome these challenges, we propose GCDance, a classifier-free diffusion framework for generating genre-specific dance motions conditioned on both music and textual prompts. Specifically, our approach extracts music features by combining high-level pre-trained music foundation model features with hand-crafted features for multi-granularity feature fusion. To achieve genre controllability, we leverage CLIP to efficiently embed genre-based textual prompt representations at each time step within our dance generation pipeline. Our GCDance framework can generate diverse dance styles from the same piece of music while ensuring coherence with the rhythm and melody of the music. Extensive experimental results obtained on the FineDance dataset demonstrate that GCDance significantly outperforms the existing state-of-the-art approaches, which also achieve competitive results on the AIST++ dataset. Our ablation and inference time analysis demonstrate that GCDance provides an effective solution for high-quality music-driven dance generation.
>
---
#### [replaced 002] Omni-CLST: Error-aware Curriculum Learning with guided Selective chain-of-Thought for audio question answering
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.12275v3](http://arxiv.org/pdf/2509.12275v3)**

> **作者:** Jinghua Zhao; Hang Su; Lichun Fan; Zhenbo Luo; Hui Wang; Haoqin Sun; Yong Qin
>
> **备注:** 5 pages, 1 figure, 2 tables submitted to icassp, under prereview
>
> **摘要:** With the rapid progress of large audio-language models (LALMs), audio question answering (AQA) has emerged as a challenging task requiring both fine-grained audio understanding and complex reasoning. While current methods mainly rely on constructing new datasets via captioning or reasoning traces, existing high-quality AQA data remains underutilized. To address this, we propose Omni-CLST, an error-aware Curriculum Learning framework with guided Selective Chain-of-Thought. The framework efficiently leverages existing high-quality dataset through two key strategies: an error-aware curriculum that organizes samples by difficulty, and a guided thought dropout mechanism that focuses reasoning on challenging cases. Experiments show that Omni-CLST achieves 73.80% on MMAU-mini and a new state of the art of 64.30% on MMAR, demonstrating robust generalization in multimodal audio-language understanding.
>
---
#### [replaced 003] FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.08557v2](http://arxiv.org/pdf/2507.08557v2)**

> **作者:** Yuxuan Jiang; Zehua Chen; Zeqian Ju; Chang Li; Weibei Dou; Jun Zhu
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** Text-to-audio (T2A) generation has achieved promising results with the recent advances in generative models. However, because of the limited quality and quantity of temporally-aligned audio-text pairs, existing T2A methods struggle to handle the complex text prompts that contain precise timing control, e.g., "owl hooted at 2.4s-5.2s". Recent works have explored data augmentation techniques or introduced timing conditions as model inputs to enable timing-conditioned 10-second T2A generation, while their synthesis quality is still limited. In this work, we propose a novel training-free timing-controlled T2A framework, FreeAudio, making the first attempt to enable timing-controlled long-form T2A generation, e.g., "owl hooted at 2.4s-5.2s and crickets chirping at 0s-24s". Specifically, we first employ an LLM to plan non-overlapping time windows and recaption each with a refined natural language description, based on the input text and timing prompts. Then we introduce: 1) Decoupling and Aggregating Attention Control for precise timing control; 2) Contextual Latent Composition for local smoothness and Reference Guidance for global consistency. Extensive experiments show that: 1) FreeAudio achieves state-of-the-art timing-conditioned T2A synthesis quality among training-free methods and is comparable to leading training-based methods; 2) FreeAudio demonstrates comparable long-form generation quality with training-based Stable Audio and paves the way for timing-controlled long-form T2A synthesis. Demo samples are available at: https://freeaudio.github.io/FreeAudio/
>
---
#### [replaced 004] A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.05310v2](http://arxiv.org/pdf/2501.05310v2)**

> **作者:** Aemon Yat Fei Chiu; Kei Ching Fung; Roger Tsz Yeung Li; Jingyu Li; Tan Lee
>
> **备注:** Submitted to the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026). Under review
>
> **摘要:** Speech self-supervised learning (SSL) models are known to learn hierarchical representations, yet how they encode different speaker-specific attributes remains under-explored. This study investigates the layer-wise disentanglement of speaker information across multiple speech SSL model families and their variants. Drawing from phonetic frameworks, we conduct a large-scale probing analysis of attributes categorised into functional groups: Acoustic (Gender), Prosodic (Pitch, Tempo, Energy), and Paralinguistic (Emotion), which we use to deconstruct the model's representation of Speaker Identity. Our findings validate a consistent three-stage hierarchy: initial layers encode fundamental timbre and prosody; middle layers synthesise abstract traits; and final layers suppress speaker identity to abstract linguistic content. An ablation study shows that while specialised speaker embeddings excel at identifying speaker identity, the intermediate layers of speech SSL models better represent dynamic prosody. This work is the first large-scale study covering a wide range of speech SSL model families and variants with fine-grained speaker-specific attributes on how they hierarchically separate the dynamic style of speech from its intrinsic characteristics, offering practical implications for downstream tasks.
>
---
#### [replaced 005] Mixture-of-Experts Framework for Field-of-View Enhanced Signal-Dependent Binauralization of Moving Talkers
- **分类: cs.SD; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.13548v2](http://arxiv.org/pdf/2509.13548v2)**

> **作者:** Manan Mittal; Thomas Deppisch; Joseph Forrer; Chris Le Sueur; Zamir Ben-Hur; David Lou Along; Daniel D. E. Wong
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** We propose a novel mixture of experts framework for field-of-view enhancement in binaural signal matching. Our approach enables dynamic spatial audio rendering that adapts to continuous talker motion, allowing users to emphasize or suppress sounds from selected directions while preserving natural binaural cues. Unlike traditional methods that rely on explicit direction-of-arrival estimation or operate in the Ambisonics domain, our signal-dependent framework combines multiple binaural filters in an online manner using implicit localization. This allows for real-time tracking and enhancement of moving sound sources, supporting applications such as speech focus, noise reduction, and world-locked audio in augmented and virtual reality. The method is agnostic to array geometry offering a flexible solution for spatial audio capture and personalized playback in next-generation consumer audio devices.
>
---
#### [replaced 006] Can Large Audio Language Models Understand Audio Well? Speech, Scene and Events Understanding Benchmark for LALMs
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.13148v2](http://arxiv.org/pdf/2509.13148v2)**

> **作者:** Han Yin; Jung-Woo Choi
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Recently, Large Audio Language Models (LALMs) have progressed rapidly, demonstrating their strong efficacy in universal audio understanding through cross-modal integration. To evaluate LALMs' audio understanding performance, researchers have proposed different benchmarks. However, key aspects for real-world interactions are underexplored in existing benchmarks, i.e., audio signals typically contain both speech and non-speech components, and energy levels of these components can vary significantly across different scenarios. Moreover, most benchmarks do not consider the joint understanding of speech, scene, and events within the same audio clip. In this work, we introduce SSEU-Bench, the first versatile audio understanding benchmark that explicitly accounts for energy differences between speech and non-speech audio, with both independent and joint understanding settings for speech, scene, and events. Furthermore, we demonstrate that some LALMs tend to underperform on certain tasks in a joint understanding setting. To address this issue, we introduce Chain-of-Thought, which effectively improves LALMs' joint audio understanding performance by decomposing complex tasks into simpler reasoning steps.
>
---
#### [replaced 007] Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection
- **分类: cs.SD; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13853v2](http://arxiv.org/pdf/2509.13853v2)**

> **作者:** Shun Huang; Zhihua Fang; Liang He
>
> **备注:** Accepted ICASSP 2025
>
> **摘要:** Unsupervised anomalous sound detection aims to detect unknown anomalous sounds by training a model using only normal audio data. Despite advancements in self-supervised methods, the issue of frequent false alarms when handling samples of the same type from different machines remains unresolved. This paper introduces a novel training technique called one-stage supervised contrastive learning (OS-SCL), which significantly addresses this problem by perturbing features in the embedding space and employing a one-stage noisy supervised contrastive learning approach. On the DCASE 2020 Challenge Task 2, it achieved 94.64\% AUC, 88.42\% pAUC, and 89.24\% mAUC using only Log-Mel features. Additionally, a time-frequency feature named TFgram is proposed, which is extracted from raw audio. This feature effectively captures critical information for anomalous sound detection, ultimately achieving 95.71\% AUC, 90.23\% pAUC, and 91.23\% mAUC. The source code is available at: \underline{www.github.com/huangswt/OS-SCL}.
>
---
#### [replaced 008] GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.13093v2](http://arxiv.org/pdf/2509.13093v2)**

> **作者:** Yujie Guo; Jiaming Zhou; Yuhang Jia; Shiwan Zhao; Yong Qin
>
> **摘要:** End-to-end multi-talker automatic speech recognition (MTASR) faces significant challenges in accurately transcribing overlapping speech, especially under high-overlap conditions. To address these challenges, we proposed Global-Local Aware Dynamic (GLAD) Mixture-of-Experts, which dynamically fuse speaker-aware global information and fine-grained local features to guide expert selection. This mechanism enables speaker-specific routing by leveraging both global context and local acoustic cues. Experiments on LibriSpeechMix show that GLAD outperforms existing MTASR approaches, particularly in challenging multi-talker scenarios. To our best knowledge, this is the first work to apply Mixture-of-Experts (MoE) to end-to-end MTASR with a global-local fusion strategy. Our code and train dataset can be found at https://github.com/NKU-HLT/GLAD.
>
---
#### [replaced 009] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18614v4](http://arxiv.org/pdf/2505.18614v4)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** Accepted to EMNLP 2025, Project Page: https://k1064190.github.io/papers/paper1.html, our codes and datasets are available at https://github.com/k1064190/MAVL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 010] The Mean of Multi-Object Trajectories
- **分类: eess.SP; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.20391v2](http://arxiv.org/pdf/2504.20391v2)**

> **作者:** Tran Thien Dat Nguyen; Ba Tuong Vo; Ba-Ngu Vo; Hoa Van Nguyen; Changbeom Shim
>
> **摘要:** This paper introduces the concept of a mean for trajectories and multi-object trajectories (defined as sets or multi-sets of trajectories) along with algorithms for computing them. Specifically, we use the Fr\'{e}chet mean, and metrics based on the optimal sub-pattern assignment (OSPA) construct, to extend the notion of average from vectors to trajectories and multi-object trajectories. Further, we develop efficient algorithms to compute these means using greedy search and Gibbs sampling. Using distributed multi-object tracking as an application, we demonstrate that the Fr\'{e}chet mean approach to multi-object trajectory consensus significantly outperforms state-of-the-art distributed multi-object tracking methods.
>
---
#### [replaced 011] Adaptive Linearly Constrained Minimum Variance Framework for Volumetric Active Noise Control
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05657v2](http://arxiv.org/pdf/2507.05657v2)**

> **作者:** Manan Mittal; Ryan M. Corey; Andrew C. Singer
>
> **备注:** 5 pages, 6 figures
>
> **摘要:** Traditional volumetric noise control typically relies on multipoint error minimization to suppress sound energy across a region, but offers limited flexibility in shaping spatial responses. This paper introduces a time domain formulation for linearly constrained minimum variance active noise control (LCMV ANC) for spatial control filter design. We demonstrate how the LCMV ANC optimization framework allows system designers to prioritize noise reduction at specific spatial locations through strategically defined linear constraints, providing a more flexible alternative to uniformly weighted multi point error minimization. An adaptive algorithm based of filtered X least mean squares (FxLMS) is derived for online adaptation of filter coefficients. Simulation and experimental results validate the proposed method's noise reduction and constraint adherence, demonstrating effective, spatially selective and broadband noise control compared to multipoint volumetric noise control.
>
---
#### [replaced 012] FunAudio-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.12508v2](http://arxiv.org/pdf/2509.12508v2)**

> **作者:** Keyu An; Yanni Chen; Chong Deng; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Wen Wang; Wupeng Wang; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou
>
> **备注:** Authors are listed in alphabetical order
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings.
>
---
#### [replaced 013] SALM: Spatial Audio Language Model with Structured Embeddings for Understanding and Editing
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16724v2](http://arxiv.org/pdf/2507.16724v2)**

> **作者:** Jinbo Hu; Yin Cao; Ming Wu; Zhenbo Luo; Jun Yang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Spatial audio understanding is essential for accurately perceiving and interpreting acoustic environments. However, existing audio-language models exhibit limitations in processing spatial audio and perceiving spatial acoustic scenes. To address this gap, we propose the Spatial Audio Language Model (SALM), a novel framework that bridges spatial audio and language through multi-modal contrastive learning. SALM integrates a text encoder with a dual-branch audio encoder that decomposes spatial sound into semantic and spatial components via structured audio embeddings. Key features of SALM include seamless alignment between spatial audio and natural language, both separate and joint extraction of spatial and semantic representations, zero-shot direction classification, and flexible support for spatial audio editing. Experimental results demonstrate that SALM effectively captures and aligns cross-modal representations, yielding well-structured audio embeddings. Furthermore, SALM enables advanced editing capabilities, such as modifying directional audio using text-based embeddings.
>
---
