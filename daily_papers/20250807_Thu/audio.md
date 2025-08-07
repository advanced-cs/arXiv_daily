# 音频 cs.SD;  eess.SP

- **最新发布 18 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Are Inherently Interpretable Models More Robust? A Study In Music Emotion Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究了解释性强的音乐情感识别模型是否比黑盒模型更鲁棒，通过对比实验验证了其在对抗样本下的性能优势，解决了深度学习模型对输入扰动敏感的问题。**

- **链接: [http://arxiv.org/pdf/2508.03780v1](http://arxiv.org/pdf/2508.03780v1)**

> **作者:** Katharina Hoedt; Arthur Flexer; Gerhard Widmer
>
> **备注:** 8 pages, published in Proceedings of the 22nd Sound and Music Computing Conference 2025 (SMC-25)
>
> **摘要:** One of the desired key properties of deep learning models is the ability to generalise to unseen samples. When provided with new samples that are (perceptually) similar to one or more training samples, deep learning models are expected to produce correspondingly similar outputs. Models that succeed in predicting similar outputs for similar inputs are often called robust. Deep learning models, on the other hand, have been shown to be highly vulnerable to minor (adversarial) perturbations of the input, which manage to drastically change a model's output and simultaneously expose its reliance on spurious correlations. In this work, we investigate whether inherently interpretable deep models, i.e., deep models that were designed to focus more on meaningful and interpretable features, are more robust to irrelevant perturbations in the data, compared to their black-box counterparts. We test our hypothesis by comparing the robustness of an interpretable and a black-box music emotion recognition (MER) model when challenged with adversarial examples. Furthermore, we include an adversarially trained model, which is optimised to be more robust, in the comparison. Our results indicate that inherently more interpretable models can indeed be more robust than their black-box counterparts, and achieve similar levels of robustness as adversarially trained models, at lower computational cost.
>
---
#### [new 002] MiDashengLM: Efficient Audio Understanding with General Audio Captions
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出MiDashengLM，解决传统LALM依赖封闭数据或专用模型的问题，通过通用音频描述融合多模态信息实现高效理解，创新性地利用公开数据集和开源技术实现4x速度提升与20x吞吐量。**

- **链接: [http://arxiv.org/pdf/2508.03983v1](http://arxiv.org/pdf/2508.03983v1)**

> **作者:** Heinrich Dinkel; Gang Li; Jizhong Liu; Jian Luan; Yadong Niu; Xingwei Sun; Tianzi Wang; Qiyang Xiao; Junbo Zhang; Jiahao Zhou
>
> **摘要:** Current approaches for large audio language models (LALMs) often rely on closed data sources or proprietary models, limiting their generalization and accessibility. This paper introduces MiDashengLM, a novel open audio-language model designed for efficient and comprehensive audio understanding through the use of general audio captions using our novel ACAVCaps training dataset. MiDashengLM exclusively relies on publicly available pretraining and supervised fine-tuning (SFT) datasets, ensuring full transparency and reproducibility. At its core, MiDashengLM integrates Dasheng, an open-source audio encoder, specifically engineered to process diverse auditory information effectively. Unlike previous works primarily focused on Automatic Speech Recognition (ASR) based audio-text alignment, our strategy centers on general audio captions, fusing speech, sound and music information into one textual representation, enabling a holistic textual representation of complex audio scenes. Lastly, MiDashengLM provides an up to 4x speedup in terms of time-to-first-token (TTFT) and up to 20x higher throughput than comparable models. Checkpoints are available online at https://huggingface.co/mispeech/midashenglm-7b and https://github.com/xiaomi-research/dasheng-lm.
>
---
#### [new 003] CoughViT: A Self-Supervised Vision Transformer for Cough Audio Representation Learning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文旨在解决医疗呼吸声分析任务中数据稀缺问题，通过自监督学习与掩码数据建模方法训练咳嗽声音特征表示，提升下游诊断任务的性能。**

- **链接: [http://arxiv.org/pdf/2508.03764v1](http://arxiv.org/pdf/2508.03764v1)**

> **作者:** Justin Luong; Hao Xue; Flora D. Salim
>
> **备注:** Accepted to ISWC
>
> **摘要:** Physicians routinely assess respiratory sounds during the diagnostic process, providing insight into the condition of a patient's airways. In recent years, AI-based diagnostic systems operating on respiratory sounds, have demonstrated success in respiratory disease detection. These systems represent a crucial advancement in early and accessible diagnosis which is essential for timely treatment. However, label and data scarcity remain key challenges, especially for conditions beyond COVID-19, limiting diagnostic performance and reliable evaluation. In this paper, we propose CoughViT, a novel pre-training framework for learning general-purpose cough sound representations, to enhance diagnostic performance in tasks with limited data. To address label scarcity, we employ masked data modelling to train a feature encoder in a self-supervised learning manner. We evaluate our approach against other pre-training strategies on three diagnostically important cough classification tasks. Experimental results show that our representations match or exceed current state-of-the-art supervised audio representations in enhancing performance on downstream tasks.
>
---
#### [new 004] NVSpeech: An Integrated and Scalable Pipeline for Human-Like Speech Modeling with Paralinguistic Vocalizations
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出了一种集成并可扩展的人类-语言建模框架，旨在弥补传统语音识别与合成系统对非语言特征（如笑声、呼吸）的认知缺失。通过构建大规模中文语料库并开发基于paralinguistic特征的ASR和TTS模型，实现了跨模态的语义控制，解决了自然语言处理中对多模态特征融合的挑战。**

- **链接: [http://arxiv.org/pdf/2508.04195v1](http://arxiv.org/pdf/2508.04195v1)**

> **作者:** Huan Liao; Qinke Ni; Yuancheng Wang; Yiheng Lu; Haoyue Zhan; Pengyuan Xie; Qiang Zhang; Zhizheng Wu
>
> **摘要:** Paralinguistic vocalizations-including non-verbal sounds like laughter and breathing, as well as lexicalized interjections such as "uhm" and "oh"-are integral to natural spoken communication. Despite their importance in conveying affect, intent, and interactional cues, such cues remain largely overlooked in conventional automatic speech recognition (ASR) and text-to-speech (TTS) systems. We present NVSpeech, an integrated and scalable pipeline that bridges the recognition and synthesis of paralinguistic vocalizations, encompassing dataset construction, ASR modeling, and controllable TTS. (1) We introduce a manually annotated dataset of 48,430 human-spoken utterances with 18 word-level paralinguistic categories. (2) We develop the paralinguistic-aware ASR model, which treats paralinguistic cues as inline decodable tokens (e.g., "You're so funny [Laughter]"), enabling joint lexical and non-verbal transcription. This model is then used to automatically annotate a large corpus, the first large-scale Chinese dataset of 174,179 utterances (573 hours) with word-level alignment and paralingustic cues. (3) We finetune zero-shot TTS models on both human- and auto-labeled data to enable explicit control over paralinguistic vocalizations, allowing context-aware insertion at arbitrary token positions for human-like speech synthesis. By unifying the recognition and generation of paralinguistic vocalizations, NVSpeech offers the first open, large-scale, word-level annotated pipeline for expressive speech modeling in Mandarin, integrating recognition and synthesis in a scalable and controllable manner. Dataset and audio demos are available at https://nvspeech170k.github.io/.
>
---
#### [new 005] ESDD 2026: Environmental Sound Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD**

- **简介: 该论文提出了EnvSDD（首个大规模环境声音深度伪造数据集）作为ESDD挑战的基准，解决了现有数据规模和类型不足的问题，通过45.25小时真实+316.7小时假声数据构建并发布挑战赛，覆盖多种场景挑战。**

- **链接: [http://arxiv.org/pdf/2508.04529v1](http://arxiv.org/pdf/2508.04529v1)**

> **作者:** Han Yin; Yang Xiao; Rohan Kumar Das; Jisheng Bai; Ting Dang
>
> **摘要:** Recent advances in audio generation systems have enabled the creation of highly realistic and immersive soundscapes, which are increasingly used in film and virtual reality. However, these audio generators also raise concerns about potential misuse, such as generating deceptive audio content for fake videos and spreading misleading information. Existing datasets for environmental sound deepfake detection (ESDD) are limited in scale and audio types. To address this gap, we have proposed EnvSDD, the first large-scale curated dataset designed for ESDD, consisting of 45.25 hours of real and 316.7 hours of fake sound. Based on EnvSDD, we are launching the Environmental Sound Deepfake Detection Challenge. Specifically, we present two different tracks: ESDD in Unseen Generators and Black-Box Low-Resource ESDD, covering various challenges encountered in real-life scenarios. The challenge will be held in conjunction with the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026).
>
---
#### [new 006] Live Music Models
- **分类: cs.SD; cs.HC; cs.LG**

- **简介: 该论文介绍了一种基于实时交互的音乐生成模型，解决了传统AI生成音乐无法实现动态实时控制的问题，开发了Magenta RealTime（文本控制）和Lyria RealTime（API控制）两种模型，通过减少参数并提供人机交互能力，实现了高质量实时音乐创作。**

- **链接: [http://arxiv.org/pdf/2508.04651v1](http://arxiv.org/pdf/2508.04651v1)**

> **作者:** Lyria Team; Antoine Caillon; Brian McWilliams; Cassie Tarakajian; Ian Simon; Ilaria Manco; Jesse Engel; Noah Constant; Pen Li; Timo I. Denk; Alberto Lalama; Andrea Agostinelli; Anna Huang; Ethan Manilow; George Brower; Hakan Erdogan; Heidi Lei; Itai Rolnick; Ivan Grishchenko; Manu Orsini; Matej Kastelic; Mauricio Zuluaga; Mauro Verzetti; Michael Dooley; Ondrej Skopek; Rafael Ferrer; Zalán Borsos; Äaron van den Oord; Douglas Eck; Eli Collins; Jason Baldridge; Tom Hume; Chris Donahue; Kehang Han; Adam Roberts
>
> **摘要:** We introduce a new class of generative models for music called live music models that produce a continuous stream of music in real-time with synchronized user control. We release Magenta RealTime, an open-weights live music model that can be steered using text or audio prompts to control acoustic style. On automatic metrics of music quality, Magenta RealTime outperforms other open-weights music generation models, despite using fewer parameters and offering first-of-its-kind live generation capabilities. We also release Lyria RealTime, an API-based model with extended controls, offering access to our most powerful model with wide prompt coverage. These models demonstrate a new paradigm for AI-assisted music creation that emphasizes human-in-the-loop interaction for live music performance.
>
---
#### [new 007] Efficient Scaling for LLM-based ASR
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究了LLM-ASR的高效计算方法，解决了传统预训练与集成方式导致的高计算成本问题，提出EFIN策略以提升性能并降低计算资源消耗。**

- **链接: [http://arxiv.org/pdf/2508.04096v1](http://arxiv.org/pdf/2508.04096v1)**

> **作者:** Bingshen Mu; Yiwen Shao; Kun Wei; Dong Yu; Lei Xie
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Large language model (LLM)-based automatic speech recognition (ASR) achieves strong performance but often incurs high computational costs. This work investigates how to obtain the best LLM-ASR performance efficiently. Through comprehensive and controlled experiments, we find that pretraining the speech encoder before integrating it with the LLM leads to significantly better scaling efficiency than the standard practice of joint post-training of LLM-ASR. Based on this insight, we propose a new multi-stage LLM-ASR training strategy, EFIN: Encoder First Integration. Among all training strategies evaluated, EFIN consistently delivers better performance (relative to 21.1% CERR) with significantly lower computation budgets (49.9% FLOPs). Furthermore, we derive a scaling law that approximates ASR error rates as a computation function, providing practical guidance for LLM-ASR scaling.
>
---
#### [new 008] Text adaptation for speaker verification with speaker-text factorized embeddings
- **分类: eess.AS; cs.SD**

- **简介: 该论文为文本适配语音识别任务，旨在解决训练/测试数据与实际文本不一致导致的文本依赖语音识别（SV）系统性能下降问题。提出基于因素化嵌入的文本适应框架，将语音分解为语者嵌入和文本嵌入后集成，利用少量无独立语者特征的文本示例进行自适应调整。实验在RSR2015集成功能下验证，显著提升了文本匹配场景下的SV性能。**

- **链接: [http://arxiv.org/pdf/2508.04425v1](http://arxiv.org/pdf/2508.04425v1)**

> **作者:** Yexin Yang; Shuai Wang; Xun Gong; Yanmin Qian; Kai Yu
>
> **备注:** ICASSP 2020
>
> **摘要:** Text mismatch between pre-collected data, either training data or enrollment data, and the actual test data can significantly hurt text-dependent speaker verification (SV) system performance. Although this problem can be solved by carefully collecting data with the target speech content, such data collection could be costly and inflexible. In this paper, we propose a novel text adaptation framework to address the text mismatch issue. Here, a speaker-text factorization network is proposed to factorize the input speech into speaker embeddings and text embeddings and then integrate them into a single representation in the later stage. Given a small amount of speaker-independent adaptation utterances, text embeddings of target speech content can be extracted and used to adapt the text-independent speaker embeddings to text-customized speaker embeddings. Experiments on RSR2015 show that text adaptation can significantly improve the performance of text mismatch conditions.
>
---
#### [new 009] Perch 2.0: The Bittern Lesson for Bioacoustics
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出了一种基于生物声学的预训练模型Perch 2.0，解决了细粒度分类与跨物种迁移学习的问题，通过扩展多物种数据集并结合自蒸馏与源预测训练，取得了BirdSet和BEANS基准结果，同时证明了其在海洋任务中的优势，揭示了细粒度分类的通用性机制。**

- **链接: [http://arxiv.org/pdf/2508.04665v1](http://arxiv.org/pdf/2508.04665v1)**

> **作者:** Bart van Merriënboer; Vincent Dumoulin; Jenny Hamer; Lauren Harrell; Andrea Burns; Tom Denton
>
> **摘要:** Perch is a performant pre-trained model for bioacoustics. It was trained in supervised fashion, providing both off-the-shelf classification scores for thousands of vocalizing species as well as strong embeddings for transfer learning. In this new release, Perch 2.0, we expand from training exclusively on avian species to a large multi-taxa dataset. The model is trained with self-distillation using a prototype-learning classifier as well as a new source-prediction training criterion. Perch 2.0 obtains state-of-the-art performance on the BirdSet and BEANS benchmarks. It also outperforms specialized marine models on marine transfer learning tasks, despite having almost no marine training data. We present hypotheses as to why fine-grained species classification is a particularly robust pre-training task for bioacoustics.
>
---
#### [new 010] Parallel GPT: Harmonizing the Independence and Interdependence of Acoustic and Semantic Information for Zero-Shot Text-to-Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在解决零样本文本到语音合成中声学与语义特征间复杂关联性不足的问题，提出通过AR/NAR模块并行结构优化融合机制，显著提升模型表现。研究基于英语/中文数据集验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.04141v1](http://arxiv.org/pdf/2508.04141v1)**

> **作者:** Jingyuan Xing; Zhipeng Li; Jialong Mai; Xiaofen Xing; Xiangmin Xu
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
>
> **摘要:** Advances in speech representation and large language models have enhanced zero-shot text-to-speech (TTS) performance. However, existing zero-shot TTS models face challenges in capturing the complex correlations between acoustic and semantic features, resulting in a lack of expressiveness and similarity. The primary reason lies in the complex relationship between semantic and acoustic features, which manifests independent and interdependent aspects.This paper introduces a TTS framework that combines both autoregressive (AR) and non-autoregressive (NAR) modules to harmonize the independence and interdependence of acoustic and semantic information. The AR model leverages the proposed Parallel Tokenizer to synthesize the top semantic and acoustic tokens simultaneously. In contrast, considering the interdependence, the Coupled NAR model predicts detailed tokens based on the general AR model's output. Parallel GPT, built on this architecture, is designed to improve zero-shot text-to-speech synthesis through its parallel structure. Experiments on English and Chinese datasets demonstrate that the proposed model significantly outperforms the quality and efficiency of the synthesis of existing zero-shot TTS models. Speech demos are available at https://t1235-ch.github.io/pgpt/.
>
---
#### [new 011] Melodic and Metrical Elements of Expressiveness in Hindustani Vocal Music
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在研究印度古典唱本Khayal音乐中旋律与节拍的表达性特征，探讨艺术家在不同演出中的表现差异，并构建计算模型区分同一作品的多个演奏版本。通过音频处理与标注技术分析两首歌曲的两变奏，验证了对艺术表达模式的量化解析。**

- **链接: [http://arxiv.org/pdf/2508.04430v1](http://arxiv.org/pdf/2508.04430v1)**

> **作者:** Yash Bhake; Ankit Anand; Preeti Rao
>
> **备注:** To appear in the proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), Daejeon Korea, 2025
>
> **摘要:** This paper presents an attempt to study the aesthetics of North Indian Khayal music with reference to the flexibility exercised by artists in performing popular compositions. We study expressive timing and pitch variations of the given lyrical content within and across performances and propose computational representations that can discriminate between different performances of the same song in terms of expression. We present the necessary audio processing and annotation procedures, and discuss our observations and insights from the analysis of a dataset of two songs in two ragas each rendered by ten prominent artists.
>
---
#### [new 012] The State Of TTS: A Case Study with Human Fooling Rates
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文旨在探讨TTS系统在人类欺骗测试中的表现，提出HFR指标并分析其有效性，指出当前模型在自然对话等真实语境下仍存在不足，建议优化评估标准以提高准确性。**

- **链接: [http://arxiv.org/pdf/2508.04179v1](http://arxiv.org/pdf/2508.04179v1)**

> **作者:** Praveen Srinivasa Varadhan; Sherry Thomas; Sai Teja M. S.; Suvrat Bhooshan; Mitesh M. Khapra
>
> **备注:** Accepted at InterSpeech 2025
>
> **摘要:** While subjective evaluations in recent years indicate rapid progress in TTS, can current TTS systems truly pass a human deception test in a Turing-like evaluation? We introduce Human Fooling Rate (HFR), a metric that directly measures how often machine-generated speech is mistaken for human. Our large-scale evaluation of open-source and commercial TTS models reveals critical insights: (i) CMOS-based claims of human parity often fail under deception testing, (ii) TTS progress should be benchmarked on datasets where human speech achieves high HFRs, as evaluating against monotonous or less expressive reference samples sets a low bar, (iii) Commercial models approach human deception in zero-shot settings, while open-source systems still struggle with natural conversational speech; (iv) Fine-tuning on high-quality data improves realism but does not fully bridge the gap. Our findings underscore the need for more realistic, human-centric evaluations alongside existing subjective tests.
>
---
#### [new 013] A Foundation Model for DAS Signal Recognition and Visual Prompt Tuning of the Pre-trained Model for Downstream Tasks
- **分类: cs.CV; eess.SP**

- **简介: 该论文提出了一种基于掩码自编码器的DAS信号识别与视觉提示调整模型（MAEPD），解决数据分布不均导致的跨域泛化问题，通过冻结骨干参数并优化视觉提示层实现高效低参数训练，验证了其在管道泄漏检测等任务中的鲁棒性和效率。**

- **链接: [http://arxiv.org/pdf/2508.04316v1](http://arxiv.org/pdf/2508.04316v1)**

> **作者:** Kun Gui; Hongliang Ren; Shang Shi; Jin Lu; Changqiu Yu; Quanjun Cao; Guomin Gu; Qi Xuan
>
> **摘要:** Distributed Acoustic Sensing (DAS) technology finds growing applications across various domains. However, data distribution disparities due to heterogeneous sensing environments pose challenges for data-driven artificial intelligence (AI) models, limiting cross-domain generalization and facing a shortage of labeled training data. To address these issues, this study proposes a foundational model for DAS signal recognition based on a Masked Autoencoder, named MAEPD. The MAEPD model is pretrained on a dataset of 635,860 samples, encompassing DAS gait spatiotemporal signals, 2D GASF images for perimeter security, 2D time-frequency images for pipeline leakage, and open-dataset signals including whale vocalizations and seismic activities, using a self-supervised mask reconstruction task to capture deep semantic features of DAS signals. Visual Prompt Tuning (VPT) is employed for downstream recognition tasks. This method freezes the pretrained backbone parameters and fine-tunes only a small set of learnable visual prompt vectors inserted into the Transformer encoder layers. Experiments on the NVIDIA GeForce RTX 4080 Super platform validate MAEPD using indoor gait recognition as a downstream task. The VPT-Deep approach achieves a classification accuracy of 96.94% with just 0.322% of parameters fine-tuned, surpassing the traditional Full Fine Tuning (FFT) method by 0.61% and reducing training time by 45%. The model also exhibits robust performance in pipeline leakage detection, confirming the generality, efficiency, and scalability of MAEPD as a foundational model. This approach offers a novel paradigm for addressing the limited generalization of signal recognition models in the DAS domain.
>
---
#### [new 014] A Multi-stage Low-latency Enhancement System for Hearing Aids
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出了一种多阶段低延迟增强系统，旨在解决ICASSP 2023 Clarity Challenge中高延迟与复杂处理的挑战。通过引入多阶段系统、异步窗口配对、融合头动信息和后处理模块，提升了助听设备的频率分辨力、延迟控制和听力感知质量（HASPI）。**

- **链接: [http://arxiv.org/pdf/2508.04283v1](http://arxiv.org/pdf/2508.04283v1)**

> **作者:** Chengwei Ouyang; Kexin Fei; Haoshuai Zhou; Congxi Lu; Linkai Li
>
> **备注:** 2 pages, 1 figure, 1 table. accepted to ICASSP 2023
>
> **摘要:** This paper proposes an end-to-end system for the ICASSP 2023 Clarity Challenge. In this work, we introduce four major novelties: (1) a novel multi-stage system in both the magnitude and complex domains to better utilize phase information; (2) an asymmetric window pair to achieve higher frequency resolution with the 5ms latency constraint; (3) the integration of head rotation information and the mixture signals to achieve better enhancement; (4) a post-processing module that achieves higher hearing aid speech perception index (HASPI) scores with the hearing aid amplification stage provided by the baseline system.
>
---
#### [new 015] Binaural Sound Event Localization and Detection Neural Network based on HRTF Localization Cues for Humanoid Robots
- **分类: eess.AS; cs.SD**

- **简介: 该论文旨在开发基于HRTF定位的双声源网络（BiSELD）以提升人形机器人对声事件类型的实时检测与定位能力。解决了传统二通道输入在高度和前后方向感知上的局限性，通过引入八通道时间-频率特征（BTFF）及HRTF定位信号，实现了跨空间平面的高效方向估计与事件分类。**

- **链接: [http://arxiv.org/pdf/2508.04333v1](http://arxiv.org/pdf/2508.04333v1)**

> **作者:** Gyeong-Tae Lee
>
> **备注:** 200 pages
>
> **摘要:** Humanoid robots require simultaneous sound event type and direction estimation for situational awareness, but conventional two-channel input struggles with elevation estimation and front-back confusion. This paper proposes a binaural sound event localization and detection (BiSELD) neural network to address these challenges. BiSELDnet learns time-frequency patterns and head-related transfer function (HRTF) localization cues from binaural input features. A novel eight-channel binaural time-frequency feature (BTFF) is introduced, comprising left/right mel-spectrograms, V-maps, an interaural time difference (ITD) map (below 1.5 kHz), an interaural level difference (ILD) map (above 5 kHz with front-back asymmetry), and spectral cue (SC) maps (above 5 kHz for elevation). The effectiveness of BTFF was confirmed across omnidirectional, horizontal, and median planes. BiSELDnets, particularly one based on the efficient Trinity module, were implemented to output time series of direction vectors for each sound event class, enabling simultaneous detection and localization. Vector activation map (VAM) visualization was proposed to analyze network learning, confirming BiSELDnet's focus on the N1 notch frequency for elevation estimation. Comparative evaluations under urban background noise conditions demonstrated that the proposed BiSELD model significantly outperforms state-of-the-art (SOTA) SELD models with binaural input.
>
---
#### [new 016] Towards interpretable emotion recognition: Identifying key features with machine learning
- **分类: eess.AS; cs.SD**

- **简介: 该论文聚焦于情绪识别任务，旨在通过机器学习识别可解释特征以提升模型在医学等关键领域的应用。研究解决了传统无监督方法缺乏可解释性的局限性，提出了一种更全面的框架来识别关键特征。**

- **链接: [http://arxiv.org/pdf/2508.04230v1](http://arxiv.org/pdf/2508.04230v1)**

> **作者:** Yacouba Kaloga; Ina Kodrasi
>
> **摘要:** Unsupervised methods, such as wav2vec2 and HuBERT, have achieved state-of-the-art performance in audio tasks, leading to a shift away from research on interpretable features. However, the lack of interpretability in these methods limits their applicability in critical domains like medicine, where understanding feature relevance is crucial. To better understand the features of unsupervised models, it remains critical to identify the interpretable features relevant to a given task. In this work, we focus on emotion recognition and use machine learning algorithms to identify and generalize the most important interpretable features for this task. While previous studies have explored feature relevance in emotion recognition, they are often constrained by narrow contexts and present inconsistent findings. Our approach aims to overcome these limitations, providing a broader and more robust framework for identifying the most important interpretable features.
>
---
#### [new 017] Radar-Based NLoS Pedestrian Localization for Darting-Out Scenarios Near Parked Vehicles with Camera-Assisted Point Cloud Interpretation
- **分类: cs.CV; eess.SP**

- **简介: 该论文提出一种基于毫米波雷达和相机的NLoS行人定位框架，解决动态停车车辆遮挡和实时道路变化带来的盲区识别难题，通过图像分割与2D雷达点云融合实现精准空间建模。**

- **链接: [http://arxiv.org/pdf/2508.04033v1](http://arxiv.org/pdf/2508.04033v1)**

> **作者:** Hee-Yeun Kim; Byeonggyu Park; Byonghyok Choi; Hansang Cho; Byungkwan Kim; Soomok Lee; Mingu Jeon; Seung-Woo Seo; Seong-Woo Kim
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025. 8 pages, 3 figures
>
> **摘要:** The presence of Non-Line-of-Sight (NLoS) blind spots resulting from roadside parking in urban environments poses a significant challenge to road safety, particularly due to the sudden emergence of pedestrians. mmWave technology leverages diffraction and reflection to observe NLoS regions, and recent studies have demonstrated its potential for detecting obscured objects. However, existing approaches predominantly rely on predefined spatial information or assume simple wall reflections, thereby limiting their generalizability and practical applicability. A particular challenge arises in scenarios where pedestrians suddenly appear from between parked vehicles, as these parked vehicles act as temporary spatial obstructions. Furthermore, since parked vehicles are dynamic and may relocate over time, spatial information obtained from satellite maps or other predefined sources may not accurately reflect real-time road conditions, leading to erroneous sensor interpretations. To address this limitation, we propose an NLoS pedestrian localization framework that integrates monocular camera image with 2D radar point cloud (PCD) data. The proposed method initially detects parked vehicles through image segmentation, estimates depth to infer approximate spatial characteristics, and subsequently refines this information using 2D radar PCD to achieve precise spatial inference. Experimental evaluations conducted in real-world urban road environments demonstrate that the proposed approach enhances early pedestrian detection and contributes to improved road safety. Supplementary materials are available at https://hiyeun.github.io/NLoS/.
>
---
#### [new 018] Multilingual Source Tracing of Speech Deepfakes: A First Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文旨在构建跨语言的语音深伪造源追踪基准，解决多语言模型追踪问题，通过比较DSP与SSL方法、分析语言差异影响等工作，为多语言模型泛化性研究提供新方向。**

- **链接: [http://arxiv.org/pdf/2508.04143v1](http://arxiv.org/pdf/2508.04143v1)**

> **作者:** Xi Xuan; Yang Xiao; Rohan Kumar Das; Tomi Kinnunen
>
> **备注:** Accepted at Interspeech SPSC 2025 - 5th Symposium on Security and Privacy in Speech Communication (Oral)
>
> **摘要:** Recent progress in generative AI has made it increasingly easy to create natural-sounding deepfake speech from just a few seconds of audio. While these tools support helpful applications, they also raise serious concerns by making it possible to generate convincing fake speech in many languages. Current research has largely focused on detecting fake speech, but little attention has been given to tracing the source models used to generate it. This paper introduces the first benchmark for multilingual speech deepfake source tracing, covering both mono- and cross-lingual scenarios. We comparatively investigate DSP- and SSL-based modeling; examine how SSL representations fine-tuned on different languages impact cross-lingual generalization performance; and evaluate generalization to unseen languages and speakers. Our findings offer the first comprehensive insights into the challenges of identifying speech generation models when training and inference languages differ. The dataset, protocol and code are available at https://github.com/xuanxixi/Multilingual-Source-Tracing.
>
---
## 更新

#### [replaced 001] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02038v2](http://arxiv.org/pdf/2508.02038v2)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively.
>
---
#### [replaced 002] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17937v2](http://arxiv.org/pdf/2507.17937v2)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Memorization in generative models extends far beyond verbatim text reproduction--it manifests through non-literal patterns, semantic associations, and surprisingly, across modalities in transcript-conditioned generation tasks such as Lyrics-to-Song (L2S) and Text-to-Video (T2V) models. We reveal a new class of cross-modality memorization where models trained on these tasks leak copyrighted content through indirect, phonetic pathways invisible to traditional text-based analysis. In this work, we introduce Adversarial PhoneTic Prompting (APT), an attack that replaces iconic phrases with homophonic alternatives--e.g., "mom's spaghetti" becomes "Bob's confetti"--preserving the acoustic form while largely changing semantic content. We demonstrate that models can be prompted to regurgitate memorized songs using phonetically similar but semantically unrelated lyrics. Despite the semantic drift, black-box models like SUNO and open-source models like YuE generate outputs that are strikingly similar to the original songs--melodically, rhythmically, and vocally--achieving high scores on AudioJudge, CLAP, and CoverID. These effects persist across genres and languages. More surprisingly, we find that phonetic prompts alone can trigger visual memorization in text-to-video models: when given altered lyrics from Lose Yourself, Veo 3 generates scenes that mirror the original music video--complete with a hooded rapper and dim urban settings--despite no explicit visual cues in the prompt. This cross-modality leakage represents an unprecedented threat: models memorize deep, structural patterns that transcend their training modality, making traditional safety measures like copyright filters ineffective. Our findings reveal a fundamental vulnerability in transcript-conditioned generative models and raise urgent concerns around copyright, provenance, and secure deployment of multimodal generation systems.
>
---
#### [replaced 003] UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.21884v2](http://arxiv.org/pdf/2506.21884v2)**

> **作者:** Fabian Perez; Sara Rojas; Carlos Hinojosa; Hoover Rueda-Chacón; Bernard Ghanem
>
> **备注:** Paper accepted at ICCV 2025 main conference
>
> **摘要:** Neural Radiance Field (NeRF)-based segmentation methods focus on object semantics and rely solely on RGB data, lacking intrinsic material properties. This limitation restricts accurate material perception, which is crucial for robotics, augmented reality, simulation, and other applications. We introduce UnMix-NeRF, a framework that integrates spectral unmixing into NeRF, enabling joint hyperspectral novel view synthesis and unsupervised material segmentation. Our method models spectral reflectance via diffuse and specular components, where a learned dictionary of global endmembers represents pure material signatures, and per-point abundances capture their distribution. For material segmentation, we use spectral signature predictions along learned endmembers, allowing unsupervised material clustering. Additionally, UnMix-NeRF enables scene editing by modifying learned endmember dictionaries for flexible material-based appearance manipulation. Extensive experiments validate our approach, demonstrating superior spectral reconstruction and material segmentation to existing methods. Project page: https://www.factral.co/UnMix-NeRF.
>
---
#### [replaced 004] SDBench: A Comprehensive Benchmark Suite for Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16136v2](http://arxiv.org/pdf/2507.16136v2)**

> **作者:** Eduardo Pacheco; Atila Orhon; Berkin Durmus; Blaise Munyampirwa; Andrey Leonov
>
> **摘要:** Even state-of-the-art speaker diarization systems exhibit high variance in error rates across different datasets, representing numerous use cases and domains. Furthermore, comparing across systems requires careful application of best practices such as dataset splits and metric definitions to allow for apples-to-apples comparison. We propose SDBench (Speaker Diarization Benchmark), an open-source benchmark suite that integrates 13 diverse datasets with built-in tooling for consistent and fine-grained analysis of speaker diarization performance for various on-device and server-side systems. SDBench enables reproducible evaluation and easy integration of new systems over time. To demonstrate the efficacy of SDBench, we built SpeakerKit, an inference efficiency-focused system built on top of Pyannote v3. SDBench enabled rapid execution of ablation studies that led to SpeakerKit being 9.6x faster than Pyannote v3 while achieving comparable error rates. We benchmark 6 state-of-the-art systems including Deepgram, AWS Transcribe, and Pyannote AI API, revealing important trade-offs between accuracy and speed.
>
---
#### [replaced 005] Are audio DeepFake detection models polyglots?
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.17924v2](http://arxiv.org/pdf/2412.17924v2)**

> **作者:** Bartłomiej Marek; Piotr Kawa; Piotr Syga
>
> **备注:** Keywords: Audio DeepFakes, DeepFake detection, multilingual audio DeepFakes
>
> **摘要:** Since the majority of audio DeepFake (DF) detection methods are trained on English-centric datasets, their applicability to non-English languages remains largely unexplored. In this work, we present a benchmark for the multilingual audio DF detection challenge by evaluating various adaptation strategies. Our experiments focus on analyzing models trained on English benchmark datasets, as well as intra-linguistic (same-language) and cross-linguistic adaptation approaches. Our results indicate considerable variations in detection efficacy, highlighting the difficulties of multilingual settings. We show that limiting the dataset to English negatively impacts the efficacy, while stressing the importance of the data in the target language.
>
---
#### [replaced 006] CCStereo: Audio-Visual Contextual and Contrastive Learning for Binaural Audio Generation
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.02786v2](http://arxiv.org/pdf/2501.02786v2)**

> **作者:** Yuanhong Chen; Kazuki Shimada; Christian Simon; Yukara Ikemiya; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** Binaural audio generation (BAG) aims to convert monaural audio to stereo audio using visual prompts, requiring a deep understanding of spatial and semantic information. However, current models risk overfitting to room environments and lose fine-grained spatial details. In this paper, we propose a new audio-visual binaural generation model incorporating an audio-visual conditional normalisation layer that dynamically aligns the mean and variance of the target difference audio features using visual context, along with a new contrastive learning method to enhance spatial sensitivity by mining negative samples from shuffled visual features. We also introduce a cost-efficient way to utilise test-time augmentation in video data to enhance performance. Our approach achieves state-of-the-art generation accuracy on the FAIR-Play and MUSIC-Stereo benchmarks.
>
---
#### [replaced 007] ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.05727v2](http://arxiv.org/pdf/2507.05727v2)**

> **作者:** He Wang; Linhan Ma; Dake Guo; Xiong Wang; Lei Xie; Jin Xu; Junyang Lin
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Automatic Speech Recognition (ASR) has been extensively investigated, yet prior benchmarks have largely focused on assessing the acoustic robustness of ASR models, leaving evaluations of their linguistic capabilities relatively underexplored. This largely stems from the limited parameter sizes and training corpora of conventional ASR models, leaving them with insufficient world knowledge, which is crucial for accurately recognizing named entities across diverse domains. For instance, drug and treatment names in medicine or specialized technical terms in engineering. Recent breakthroughs in Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of advanced context modeling and general artificial intelligence capabilities. Leveraging LLMs, we envision a unified system capable of robust speech recognition across diverse real-world domains, yet existing benchmarks are inadequate for evaluating this objective. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess the linguistic competence of ASR systems using corpora that feature numerous named entities across multiple domains. It encompasses up to 40,000 data entries with more than 300,000 named entities across over 10 domains. Beyond the audio and its transcription, each sample provides the domain it belongs to and a list of named entities it contains, which are referred to as the context. Based on this, we introduce three evaluation modes to assess how effectively models can exploit such context to improve ASR accuracy. Extensive evaluation on ContextASR-Bench highlights that LALMs outperform conventional ASR models by a large margin thanks to the strong world knowledge and context modeling of LLMs, yet there remains ample room for further improvement. The dataset and evaluation code have been released.
>
---
#### [replaced 008] AudioMiXR: Spatial Audio Object Manipulation with 6DoF for Sound Design in Augmented Reality
- **分类: cs.HC; cs.SD; eess.AS; H.5.2; H.5.5; H.5.1**

- **链接: [http://arxiv.org/pdf/2502.02929v4](http://arxiv.org/pdf/2502.02929v4)**

> **作者:** Brandon Woodard; Margarita Geleta; Joseph J. LaViola Jr.; Andrea Fanelli; Rhonda Wilson
>
> **备注:** Updated abstract
>
> **摘要:** We present AudioMiXR, an augmented reality (AR) interface intended to assess how users manipulate virtual audio objects situated in their physical space using six degrees of freedom (6DoF) deployed on a head-mounted display (Apple Vision Pro) for 3D sound design. Existing tools for 3D sound design are typically constrained to desktop displays, which may limit spatial awareness of mixing within the execution environment. Utilizing an XR HMD to create soundscapes may provide a real-time test environment for 3D sound design, as modern HMDs can provide precise spatial localization assisted by cross-modal interactions. However, there is no research on design guidelines specific to sound design with 6DoF in XR. To provide a first step toward identifying design-related research directions in this space, we conducted an exploratory study where we recruited 27 participants, consisting of expert and non-expert sound designers. The goal was to assess design lessons that can be used to inform future research venues in 3D sound design. We ran a within-subjects study where users designed both a music and cinematic soundscapes. After thematically analyzing participant data, we constructed two design lessons: (1) Proprioception for AR Sound Design, and (2) Balancing Audio-Visual Modalities in AR GUIs. Additionally, we provide application domains that can benefit most from 6DoF sound design based on our results. To expand on these insights, we conducted a second within-subjects study comparing AudioMiXR to a 2D panner baseline. Results show that AudioMiXR significantly improved usability (SUS), reduced frustration and mental workload (NASA-TLX), and enhanced creativity across all subscales. These findings demonstrate that 6DoF AR interaction yields measurable gains in user experience and creative output, positioning AudioMiXR as a promising foundation for future AR-based sound design tools.
>
---
#### [replaced 009] EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.03543v2](http://arxiv.org/pdf/2508.03543v2)**

> **作者:** Tianxin Xie; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **摘要:** Text-to-speech (TTS) has shown great progress in recent years. However, most existing TTS systems offer only coarse and rigid emotion control, typically via discrete emotion labels or a carefully crafted and detailed emotional text prompt, making fine-grained emotion manipulation either inaccessible or unstable. These models also require extensive, high-quality datasets for training. To address these limitations, we propose EmoSteer-TTS, a novel training-free approach, to achieve fine-grained speech emotion control (conversion, interpolation, erasure) by activation steering. We first empirically observe that modifying a subset of the internal activations within a flow matching-based TTS model can effectively alter the emotional tone of synthesized speech. Building on this insight, we then develop a training-free and efficient algorithm, including activation extraction, emotional token searching, and inference-time steering, which can be seamlessly integrated into a wide range of pretrained models (e.g., F5-TTS, CosyVoice2, and E2-TTS). In addition, to derive effective steering vectors, we construct a curated emotional speech dataset with diverse speakers. Extensive experiments demonstrate that EmoSteer-TTS enables fine-grained, interpretable, and continuous control over speech emotion, outperforming the state-of-the-art (SOTA). To the best of our knowledge, this is the first method that achieves training-free and continuous fine-grained emotion control in TTS.
>
---
#### [replaced 010] Environmental Sound Classification on An Embedded Hardware Platform
- **分类: cs.SD; cs.AI; cs.SY; eess.AS; eess.SY**

- **链接: [http://arxiv.org/pdf/2306.09106v2](http://arxiv.org/pdf/2306.09106v2)**

> **作者:** Gabriel Bibbo; Arshdeep Singh; Mark D. Plumbley
>
> **备注:** Accepted in INTER-NOISE and NOISE-CON Congress and Conference Proceedings, INTER-NOISE24, Nantes, France
>
> **摘要:** Convolutional neural networks (CNNs) have exhibited state-of-the-art performance in various audio classification tasks. However, their real-time deployment remains a challenge on resource constrained devices such as embedded systems. In this paper, we analyze how the performance of large-scale pre-trained audio neural networks designed for audio pattern recognition changes when deployed on a hardware such as a Raspberry Pi. We empirically study the role of CPU temperature, microphone quality and audio signal volume on performance. Our experiments reveal that the continuous CPU usage results in an increased temperature that can trigger an automated slowdown mechanism in the Raspberry Pi, impacting inference latency. The quality of a microphone, specifically with affordable devices such as the Google AIY Voice Kit, and audio signal volume, all affect the system performance. In the course of our investigation, we encounter substantial complications linked to library compatibility and the unique processor architecture requirements of the Raspberry Pi, making the process less straightforward compared to conventional computers (PCs). Our observations, while presenting challenges, pave the way for future researchers to develop more compact machine learning models, design heat-dissipative hardware, and select appropriate microphones when AI models are deployed for real-time applications on edge devices.
>
---
#### [replaced 011] AV-SSAN: Audio-Visual Selective DoA Estimation through Explicit Multi-Band Semantic-Spatial Alignment
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07384v2](http://arxiv.org/pdf/2507.07384v2)**

> **作者:** Yu Chen; Hongxu Zhu; Jiadong Wang; Kainan Chen; Xinyuan Qian
>
> **备注:** 9 pages
>
> **摘要:** Audio-visual sound source localization (AV-SSL) estimates the position of sound sources by fusing auditory and visual cues. Current AV-SSL methodologies typically require spatially-paired audio-visual data and cannot selectively localize specific target sources. To address these limitations, we introduce Cross-Instance Audio-Visual Localization (CI-AVL), a novel task that localizes target sound sources using visual prompts from different instances of the same semantic class. CI-AVL enables selective localization without spatially paired data. To solve this task, we propose AV-SSAN, a semantic-spatial alignment framework centered on a Multi-Band Semantic-Spatial Alignment Network (MB-SSA Net). MB-SSA Net decomposes the audio spectrogram into multiple frequency bands, aligns each band with semantic visual prompts, and refines spatial cues to estimate the direction-of-arrival (DoA). To facilitate this research, we construct VGGSound-SSL, a large-scale dataset comprising 13,981 spatial audio clips across 296 categories, each paired with visual prompts. AV-SSAN achieves a mean absolute error of 16.59 and an accuracy of 71.29%, significantly outperforming existing AV-SSL methods. Code and data will be public.
>
---
#### [replaced 012] READ: Real-time and Efficient Asynchronous Diffusion for Audio-driven Talking Head Generation
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.03457v2](http://arxiv.org/pdf/2508.03457v2)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Jianqing Gao; Qingfeng Liu
>
> **备注:** Project page: https://readportrait.github.io/READ/
>
> **摘要:** The introduction of diffusion models has brought significant advances to the field of audio-driven talking head generation. However, the extremely slow inference speed severely limits the practical implementation of diffusion-based talking head generation models. In this study, we propose READ, the first real-time diffusion-transformer-based talking head generation framework. Our approach first learns a spatiotemporal highly compressed video latent space via a temporal VAE, significantly reducing the token count to accelerate generation. To achieve better audio-visual alignment within this compressed latent space, a pre-trained Speech Autoencoder (SpeechAE) is proposed to generate temporally compressed speech latent codes corresponding to the video latent space. These latent representations are then modeled by a carefully designed Audio-to-Video Diffusion Transformer (A2V-DiT) backbone for efficient talking head synthesis. Furthermore, to ensure temporal consistency and accelerated inference in extended generation, we propose a novel asynchronous noise scheduler (ANS) for both the training and inference process of our framework. The ANS leverages asynchronous add-noise and asynchronous motion-guided generation in the latent space, ensuring consistency in generated video clips. Experimental results demonstrate that READ outperforms state-of-the-art methods by generating competitive talking head videos with significantly reduced runtime, achieving an optimal balance between quality and speed while maintaining robust metric stability in long-time generation.
>
---
#### [replaced 013] Adaptive Audio-Visual Speech Recognition via Matryoshka-Based Multimodal LLMs
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.06362v2](http://arxiv.org/pdf/2503.06362v2)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Stavros Petridis
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) leverages audio and visual modalities to improve robustness in noisy environments. Recent advances in Large Language Models (LLMs) show strong performance in speech recognition, including AVSR. However, the long speech representations lead to high computational costs for LLMs. Prior methods compress inputs before feeding them to LLMs, but high compression often harms accuracy. To address this, we propose Llama-MTSK, the first Matryoshka-based Multimodal LLM for AVSR, which flexibly adapts audio-visual token allocation under varying compute constraints. Inspired by Matryoshka Representation Learning, our model encodes representations at multiple granularities with a single architecture, avoiding the need for separate models. For efficient fine-tuning, we introduce three LoRA-based strategies using global and scale-specific modules. Evaluations on major AVSR datasets show Llama-MTSK matches or outperforms models trained at fixed compression levels.
>
---
#### [replaced 014] Silent Speech Sentence Recognition with Six-Axis Accelerometers using Conformer and CTC Algorithm
- **分类: cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.17829v2](http://arxiv.org/pdf/2502.17829v2)**

> **作者:** Yudong Xie; Zhifeng Han; Qinfan Xiao; Liwei Liang; Lu-Qi Tao; Tian-Ling Ren
>
> **摘要:** Silent speech interfaces (SSI) are being actively developed to assist individuals with communication impairments who have long suffered from daily hardships and a reduced quality of life. However, silent sentences are difficult to segment and recognize due to elision and linking. A novel silent speech sentence recognition method is proposed to convert the facial motion signals collected by six-axis accelerometers into transcribed words and sentences. A Conformer-based neural network with the Connectionist-Temporal-Classification algorithm is used to gain contextual understanding and translate the non-acoustic signals into words sequences, solely requesting the constituent words in the database. Test results show that the proposed method achieves a 97.17% accuracy in sentence recognition, surpassing the existing silent speech recognition methods with a typical accuracy of 85%-95%, and demonstrating the potential of accelerometers as an available SSI modality for high-accuracy silent speech sentence recognition.
>
---
#### [replaced 015] Can Sound Replace Vision in LLaVA With Token Substitution?
- **分类: cs.MM; cs.SD; eess.AS; 68T07**

- **链接: [http://arxiv.org/pdf/2506.10416v2](http://arxiv.org/pdf/2506.10416v2)**

> **作者:** Ali Vosoughi; Jing Bi; Pinxin Liu; Yunlong Tang; Chenliang Xu
>
> **备注:** Project page: https://ali-vosoughi.github.io/SoundCLIP/
>
> **摘要:** What happens when we push audio-visual alignment to its absolute limits? To systematically investigate this question, we needed datasets with granular alignment quality annotations, but existing datasets treat alignment as binary, either synchronized or not. To address this limitation, we developed a comprehensive dataset featuring detailed alignment scores that reveal the hidden spectrum of audio-visual perceptual correspondence. Using these precise scores, we create "superaligned" representations by training exclusively on the most perfectly matched audio-visual pairs, then conduct our systematic investigation into how this extreme alignment transforms perceptual model behavior across retrieval and generation tasks. The encoders under study fall into two main groups consisting of image-centric encoders that were pretrained using visual modalities as intermediary hubs for connecting modalities, and text-centric encoders that were pretrained with direct audio-language alignment. We first measure the baseline performance of these encoders on two key tasks, namely cross-modal retrieval and text description generation in vision-language models. Subsequently, we realign all encoders with the CLIP space using highly coherent audio-visual data and observe the performance changes. Our findings reveal that the initial architectural type of the encoder determines how it responds to the alignment process. Image-centric encoders, which are inherently designed for alignment, demonstrate exceptional performance in cross-modal retrieval, but this intensive alignment causes compression of unique linguistic information and reduces the quality of their text description generation in vision-language models. In contrast, text-centric encoders, which possess stronger linguistic authenticity, are able to maintain a better balance between the two objectives.
>
---
