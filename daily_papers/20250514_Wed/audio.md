# 音频 cs.SD;  eess.SP

- **最新发布 7 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Fast Text-to-Audio Generation with Adversarial Post-Training
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决现有模型推理速度慢的问题。作者提出对抗相对对比后训练（ARC）方法，首次将对抗训练应用于扩散/流模型加速而非蒸馏，结合相对对抗机制和对比鉴别器提升生成质量与速度。优化后的模型在H100上生成12秒音频仅需75毫秒，成为当前最快方案。**

- **链接: [http://arxiv.org/pdf/2505.08175v1](http://arxiv.org/pdf/2505.08175v1)**

> **作者:** Zachary Novack; Zach Evans; Zack Zukowski; Josiah Taylor; CJ Carr; Julian Parker; Adnan Al-Sinan; Gian Marco Iodice; Julian McAuley; Taylor Berg-Kirkpatrick; Jordi Pons
>
> **摘要:** Text-to-audio systems, while increasingly performant, are slow at inference time, thus making their latency unpractical for many creative applications. We present Adversarial Relativistic-Contrastive (ARC) post-training, the first adversarial acceleration algorithm for diffusion/flow models not based on distillation. While past adversarial post-training methods have struggled to compare against their expensive distillation counterparts, ARC post-training is a simple procedure that (1) extends a recent relativistic adversarial formulation to diffusion/flow post-training and (2) combines it with a novel contrastive discriminator objective to encourage better prompt adherence. We pair ARC post-training with a number optimizations to Stable Audio Open and build a model capable of generating $\approx$12s of 44.1kHz stereo audio in $\approx$75ms on an H100, and $\approx$7s on a mobile edge-device, the fastest text-to-audio model to our knowledge.
>
---
#### [new 002] Not that Groove: Zero-Shot Symbolic Music Editing
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究符号音乐编辑任务，解决AI音频生成灵活性不足及标注数据缺乏问题。提出使用零样本提示的大语言模型编辑鼓点节奏，设计交互格式连接模型与音乐，并提供与音乐家判断一致的数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2505.08203v1](http://arxiv.org/pdf/2505.08203v1)**

> **作者:** Li Zhang
>
> **摘要:** Most work in AI music generation focused on audio, which has seen limited use in the music production industry due to its rigidity. To maximize flexibility while assuming only textual instructions from producers, we are among the first to tackle symbolic music editing. We circumvent the known challenge of lack of labeled data by proving that LLMs with zero-shot prompting can effectively edit drum grooves. The recipe of success is a creatively designed format that interfaces LLMs and music, while we facilitate evaluation by providing an evaluation dataset with annotated unit tests that highly aligns with musicians' judgment.
>
---
#### [new 003] A Mamba-based Network for Semi-supervised Singing Melody Extraction Using Confidence Binary Regularization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对歌唱旋律提取任务，解决现有方法效率低（Transformer二次计算）、音符建模不准（忽略音乐表演特性）和数据不足问题。提出基于Mamba的SpectMamba网络，通过线性复杂度结构、音符-F0解码器和置信二元正则化模块，在半监督框架下有效利用未标注数据提升性能。**

- **链接: [http://arxiv.org/pdf/2505.08681v1](http://arxiv.org/pdf/2505.08681v1)**

> **作者:** Xiaoliang He; Kangjie Dong; Jingkai Cao; Shuai Yu; Wei Li; Yi Yu
>
> **摘要:** Singing melody extraction (SME) is a key task in the field of music information retrieval. However, existing methods are facing several limitations: firstly, prior models use transformers to capture the contextual dependencies, which requires quadratic computation resulting in low efficiency in the inference stage. Secondly, prior works typically rely on frequencysupervised methods to estimate the fundamental frequency (f0), which ignores that the musical performance is actually based on notes. Thirdly, transformers typically require large amounts of labeled data to achieve optimal performances, but the SME task lacks of sufficient annotated data. To address these issues, in this paper, we propose a mamba-based network, called SpectMamba, for semi-supervised singing melody extraction using confidence binary regularization. In particular, we begin by introducing vision mamba to achieve computational linear complexity. Then, we propose a novel note-f0 decoder that allows the model to better mimic the musical performance. Further, to alleviate the scarcity of the labeled data, we introduce a confidence binary regularization (CBR) module to leverage the unlabeled data by maximizing the probability of the correct classes. The proposed method is evaluated on several public datasets and the conducted experiments demonstrate the effectiveness of our proposed method.
>
---
#### [new 004] Fréchet Power-Scenario Distance: A Metric for Evaluating Generative AI Models across Multiple Time-Scales in Smart Grids
- **分类: cs.LG; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于生成模型评估任务，旨在解决智能电网中生成AI模型产生的合成数据质量评估难题。针对传统欧氏距离无法衡量群体数据差异的缺陷，提出基于特征空间Fréchet距离的新指标，从分布层面评估生成质量。实验验证了该方法在多时间尺度和模型中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.08082v1](http://arxiv.org/pdf/2505.08082v1)**

> **作者:** Yuting Cai; Shaohuai Liu; Chao Tian; Le Xie
>
> **摘要:** Generative artificial intelligence (AI) models in smart grids have advanced significantly in recent years due to their ability to generate large amounts of synthetic data, which would otherwise be difficult to obtain in the real world due to confidentiality constraints. A key challenge in utilizing such synthetic data is how to assess the data quality produced from such generative models. Traditional Euclidean distance-based metrics only reflect pair-wise relations between two individual samples, and could fail in evaluating quality differences between groups of synthetic datasets. In this work, we propose a novel metric based on the Fr\'{e}chet Distance (FD) estimated between two datasets in a learned feature space. The proposed method evaluates the quality of generation from a distributional perspective. Empirical results demonstrate the superiority of the proposed metric across timescales and models, enhancing the reliability of data-driven decision-making in smart grid operations.
>
---
#### [new 005] M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis
- **分类: cs.GR; cs.AI; cs.CV; cs.SD; eess.AS; I.3.6**

- **简介: 该论文研究音频驱动全身人体动作生成（含面部、肢体和全局运动），属于虚拟形象合成任务。针对现有方法固定粒度建模导致动作不自然的问题，提出M3G框架：通过多粒度VQ-VAE编码不同时长的动作模式，并设计音频特征提取器预测多粒度动作令牌，最终重建自然动作，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08293v1](http://arxiv.org/pdf/2505.08293v1)**

> **作者:** Zhizhuo Yin; Yuk Hang Tsui; Pan Hui
>
> **备注:** 9 Pages, 4 figures, submitted to NIPS 2025
>
> **摘要:** Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures.
>
---
#### [new 006] MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，提出MiniMax-Speech模型解决零样本语音合成与高相似度语音克隆问题。通过可学习说话人编码器提取无转录参考音频音色特征，结合Flow-VAE提升音质，支持32种语言。模型实现了SOTA克隆效果，并扩展支持情感控制、文本描述生成音色等功能。**

- **链接: [http://arxiv.org/pdf/2505.07916v1](http://arxiv.org/pdf/2505.07916v1)**

> **作者:** Bowen Zhang; Congchao Guo; Geng Yang; Hang Yu; Haozhe Zhang; Heidi Lei; Jialong Mai; Junjie Yan; Kaiyue Yang; Mingqi Yang; Peikai Huang; Ruiyang Jin; Sitan Jiang; Weihua Cheng; Yawei Li; Yichen Xiao; Yiying Zhou; Yongmao Zhang; Yuan Lu; Yucen He
>
> **摘要:** We introduce MiniMax-Speech, an autoregressive Transformer-based Text-to-Speech (TTS) model that generates high-quality speech. A key innovation is our learnable speaker encoder, which extracts timbre features from a reference audio without requiring its transcription. This enables MiniMax-Speech to produce highly expressive speech with timbre consistent with the reference in a zero-shot manner, while also supporting one-shot voice cloning with exceptionally high similarity to the reference voice. In addition, the overall quality of the synthesized audio is enhanced through the proposed Flow-VAE. Our model supports 32 languages and demonstrates excellent performance across multiple objective and subjective evaluations metrics. Notably, it achieves state-of-the-art (SOTA) results on objective voice cloning metrics (Word Error Rate and Speaker Similarity) and has secured the top position on the public TTS Arena leaderboard. Another key strength of MiniMax-Speech, granted by the robust and disentangled representations from the speaker encoder, is its extensibility without modifying the base model, enabling various applications such as: arbitrary voice emotion control via LoRA; text to voice (T2V) by synthesizing timbre features directly from text description; and professional voice cloning (PVC) by fine-tuning timbre features with additional data. We encourage readers to visit https://minimax-ai.github.io/tts_tech_report for more examples.
>
---
#### [new 007] Unveiling the Best Practices for Applying Speech Foundation Models to Speech Intelligibility Prediction for Hearing-Impaired People
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究如何优化语音基础模型（SFMs）用于听力受损人群的语音清晰度预测任务。针对现有方法优化不足的问题，通过实验分析编码层选择、预测头结构和模型集成对性能的影响，发现单层编码、时序建模及模型集成能有效提升预测效果，为实际应用提供优化方案。**

- **链接: [http://arxiv.org/pdf/2505.08215v1](http://arxiv.org/pdf/2505.08215v1)**

> **作者:** Haoshuai Zhou; Boxuan Cao; Changgeng Mo; Linkai Li; Shan Xiang Wang
>
> **摘要:** Speech foundation models (SFMs) have demonstrated strong performance across a variety of downstream tasks, including speech intelligibility prediction for hearing-impaired people (SIP-HI). However, optimizing SFMs for SIP-HI has been insufficiently explored. In this paper, we conduct a comprehensive study to identify key design factors affecting SIP-HI performance with 5 SFMs, focusing on encoder layer selection, prediction head architecture, and ensemble configurations. Our findings show that, contrary to traditional use-all-layers methods, selecting a single encoder layer yields better results. Additionally, temporal modeling is crucial for effective prediction heads. We also demonstrate that ensembling multiple SFMs improves performance, with stronger individual models providing greater benefit. Finally, we explore the relationship between key SFM attributes and their impact on SIP-HI performance. Our study offers practical insights into effectively adapting SFMs for speech intelligibility prediction for hearing-impaired populations.
>
---
## 更新

#### [replaced 001] Decadal analysis of sea surface temperature patterns, climatology, and anomalies in temperate coastal waters with Landsat-8 TIRS observations
- **分类: physics.ao-ph; cs.CV; eess.IV; eess.SP; physics.geo-ph**

- **链接: [http://arxiv.org/pdf/2503.05843v2](http://arxiv.org/pdf/2503.05843v2)**

> **作者:** Yiqing Guo; Nagur Cherukuru; Eric Lehmann; Xiubin Qi; Mark Doubelld; S. L. Kesav Unnithan; Ming Feng
>
> **备注:** Submitted to GIScience & Remote Sensing
>
> **摘要:** Sea surface temperature (SST) is a fundamental physical parameter characterising the thermal state of sea surface. Due to the intricate thermal interactions between land, sea, and atmosphere, the spatial gradients of SST in coastal waters often appear at finer spatial scales than those in open ocean waters. The Thermal Infrared Sensor (TIRS) onboard Landsat-8, with its 100-meter spatial resolution, offers a unique opportunity to uncover fine-scale coastal SST patterns that would otherwise be overlooked by coarser-resolution thermal sensors. In this study, we first analysed the spatiotemporal patterns of SST in South Australia's temperate coastal waters from 2014 to 2023 by developing an operational approach for SST retrieval from the Landsat-8 TIRS sensor. A buoy was deployed off the coast of Port Lincoln, South Australia, to validate the quality of SST retrievals. Then the daily baseline climatology of SST with 100 m resolution was constructed, which allowed for the detection and analysis of anomalous SST events. Our results suggest the following: (1) the satellite-derived SST data aligned well with the in-situ measured SST values; (2) the semi-enclosed, shallow regions of Upper Spencer Gulf and Upper St Vincent Gulf showed higher temperatures during summer and cooler temperatures during winter than waters closer to the open ocean, resulting in a higher seasonal variation in SST; (3) the near-shore shallow areas in Spencer Gulf and St Vincent Gulf, and regions surrounding Kangaroo Island, were identified to have a higher probability of SST anomalies compared to the rest of the study area; and (4) anomalous SST events were more likely to happen during the warm months than the cool months. We hope these findings would be helpful in supporting the fishing and aquaculture industries in the coastal waters of South Australia.
>
---
#### [replaced 002] A Classification Benchmark for Artificial Intelligence Detection of Laryngeal Cancer from Patient Voice
- **分类: cs.SD; cs.LG; eess.AS; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2412.16267v2](http://arxiv.org/pdf/2412.16267v2)**

> **作者:** Mary Paterson; James Moor; Luisa Cutillo
>
> **备注:** 16 pages, 6 figures, 10 tables
>
> **摘要:** Cases of laryngeal cancer are predicted to rise significantly in the coming years. Current diagnostic pathways are inefficient, putting undue stress on both patients and the medical system. Artificial intelligence offers a promising solution by enabling non-invasive detection of laryngeal cancer from patient voice, which could help prioritise referrals more effectively. A major barrier in this field is the lack of reproducible methods. Our work addresses this challenge by introducing a benchmark suite comprising 36 models trained and evaluated on open-source datasets. These models classify patients with benign and malignant voice pathologies. All models are accessible in a public repository, providing a foundation for future research. We evaluate three algorithms and three audio feature sets, including both audio-only inputs and multimodal inputs incorporating demographic and symptom data. Our best model achieves a balanced accuracy of 83.7%, sensitivity of 84.0%, specificity of 83.3%, and AUROC of 91.8%.
>
---
#### [replaced 003] SonicRAG : High Fidelity Sound Effects Synthesis Based on Retrival Augmented Generation
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03244v2](http://arxiv.org/pdf/2505.03244v2)**

> **作者:** Yu-Ren Guo; Wen-Kai Tai
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing (NLP) and multimodal learning, with successful applications in text generation and speech synthesis, enabling a deeper understanding and generation of multimodal content. In the field of sound effects (SFX) generation, LLMs have been leveraged to orchestrate multiple models for audio synthesis. However, due to the scarcity of annotated datasets, and the complexity of temproal modeling. current SFX generation techniques still fall short in achieving high-fidelity audio. To address these limitations, this paper introduces a novel framework that integrates LLMs with existing sound effect databases, allowing for the retrieval, recombination, and synthesis of audio based on user requirements. By leveraging this approach, we enhance the diversity and quality of generated sound effects while eliminating the need for additional recording costs, offering a flexible and efficient solution for sound design and application.
>
---
#### [replaced 004] BLAB: Brutally Long Audio Bench
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03054v2](http://arxiv.org/pdf/2505.03054v2)**

> **作者:** Orevaoghene Ahia; Martijn Bartelds; Kabir Ahuja; Hila Gonen; Valentin Hofmann; Siddhant Arora; Shuyue Stella Li; Vishal Puttagunta; Mofetoluwa Adeyemi; Charishma Buchireddy; Ben Walls; Noah Bennett; Shinji Watanabe; Noah A. Smith; Yulia Tsvetkov; Sachin Kumar
>
> **摘要:** Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities.
>
---
#### [replaced 005] ImprovNet -- Generating Controllable Musical Improvisations with Iterative Corruption Refinement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.04522v2](http://arxiv.org/pdf/2502.04522v2)**

> **作者:** Keshav Bhandari; Sungkyun Chang; Tongyu Lu; Fareza R. Enus; Louis B. Bradshaw; Dorien Herremans; Simon Colton
>
> **备注:** 10 pages, 6 figures, IJCNN 2025 conference
>
> **摘要:** Despite deep learning's remarkable advances in style transfer across various domains, generating controllable performance-level musical style transfer for complete symbolically represented musical works remains a challenging area of research. Much of this is owed to limited datasets, especially for genres such as jazz, and the lack of unified models that can handle multiple music generation tasks. This paper presents ImprovNet, a transformer-based architecture that generates expressive and controllable musical improvisations through a self-supervised corruption-refinement training strategy. The improvisational style transfer is aimed at making meaningful modifications to one or more musical elements - melody, harmony or rhythm of the original composition with respect to the target genre. ImprovNet unifies multiple capabilities within a single model: it can perform cross-genre and intra-genre improvisations, harmonize melodies with genre-specific styles, and execute short prompt continuation and infilling tasks. The model's iterative generation framework allows users to control the degree of style transfer and structural similarity to the original composition. Objective and subjective evaluations demonstrate ImprovNet's effectiveness in generating musically coherent improvisations while maintaining structural relationships with the original pieces. The model outperforms Anticipatory Music Transformer in short continuation and infilling tasks and successfully achieves recognizable genre conversion, with 79\% of participants correctly identifying jazz-style improvisations of classical pieces. Our code and demo page can be found at https://github.com/keshavbhandari/improvnet.
>
---
