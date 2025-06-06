# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Survey on the Evaluation of Generative Models in Music
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于音乐生成模型评估任务，旨在系统梳理音乐生成系统的评价目标、方法与指标。论文从音乐学、工程学和人机交互角度出发，综述了主观与客观、定性与定量、实证与计算等多种评估方法，探讨其优劣与挑战。**

- **链接: [http://arxiv.org/pdf/2506.05104v1](http://arxiv.org/pdf/2506.05104v1)**

> **作者:** Alexander Lerch; Claire Arthur; Nick Bryan-Kinns; Corey Ford; Qianyi Sun; Ashvala Vinay
>
> **备注:** Submitted to ACM CSUR, 26-Jun-2024
>
> **摘要:** Research on generative systems in music has seen considerable attention and growth in recent years. A variety of attempts have been made to systematically evaluate such systems. We provide an interdisciplinary review of the common evaluation targets, methodologies, and metrics for the evaluation of both system output and model usability, covering subjective and objective approaches, qualitative and quantitative approaches, as well as empirical and computational methods. We discuss the advantages and challenges of such approaches from a musicological, an engineering, and an HCI perspective.
>
---
#### [new 002] Grapheme-Coherent Phonemic and Prosodic Annotation of Speech by Implicit and Explicit Grapheme Conditioning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音标注任务，旨在解决语音的音素和韵律标注与字素不一致的问题。通过引入隐式和显式的字素条件约束，提升了标注的一致性，并验证了其在口音估计等下游任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.04527v1](http://arxiv.org/pdf/2506.04527v1)**

> **作者:** Hien Ohnaka; Yuma Shirahata; Byeongseon Park; Ryuichi Yamamoto
>
> **备注:** 5 pages, 2 figures, and 4 tables, accepted to INTERSPEECH 2025
>
> **摘要:** We propose a model to obtain phonemic and prosodic labels of speech that are coherent with graphemes. Unlike previous methods that simply fine-tune a pre-trained ASR model with the labels, the proposed model conditions the label generation on corresponding graphemes by two methods: 1) Add implicit grapheme conditioning through prompt encoder using pre-trained BERT features. 2) Explicitly prune the label hypotheses inconsistent with the grapheme during inference. These methods enable obtaining parallel data of speech, the labels, and graphemes, which is applicable to various downstream tasks such as text-to-speech and accent estimation from text. Experiments showed that the proposed method significantly improved the consistency between graphemes and the predicted labels. Further, experiments on accent estimation task confirmed that the created parallel data by the proposed method effectively improve the estimation accuracy.
>
---
#### [new 003] LLM-based phoneme-to-grapheme for phoneme-based speech recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决基于音素的多语言语音识别中解码效果受限的问题。作者提出了LLM-P2G方法，将大语言模型用于音素到字素转换，并通过数据增强和训练策略提升性能，在跨语言识别中取得了更好的效果。**

- **链接: [http://arxiv.org/pdf/2506.04711v1](http://arxiv.org/pdf/2506.04711v1)**

> **作者:** Te Ma; Min Bi; Saierdaer Yusuyin; Hao Huang; Zhijian Ou
>
> **备注:** Interspeech 2025
>
> **摘要:** In automatic speech recognition (ASR), phoneme-based multilingual pre-training and crosslingual fine-tuning is attractive for its high data efficiency and competitive results compared to subword-based models. However, Weighted Finite State Transducer (WFST) based decoding is limited by its complex pipeline and inability to leverage large language models (LLMs). Therefore, we propose LLM-based phoneme-to-grapheme (LLM-P2G) decoding for phoneme-based ASR, consisting of speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G). A challenge is that there seems to have information loss in cascading S2P and P2G. To address this challenge, we propose two training strategies: data augmentation with noisy phonemes (DANP), and randomized top-$K$ marginalized (TKM) training and decoding. Our experimental results show that LLM-P2G outperforms WFST-based systems in crosslingual ASR for Polish and German, by relative WER reductions of 3.6% and 6.9% respectively.
>
---
#### [new 004] Domain Adaptation Method and Modality Gap Impact in Audio-Text Models for Prototypical Sound Classification
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频-文本模型在零样本环境声音分类中的应用任务。它旨在解决背景声源干扰导致性能下降的问题。论文提出了一种新方法，量化并整合背景声源对分类的影响，通过域适应技术提升准确率，并分析了音频与文本嵌入之间的模态差距对分类效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.04376v1](http://arxiv.org/pdf/2506.04376v1)**

> **作者:** Emiliano Acevedo; Martín Rocamora; Magdalena Fuentes
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Audio-text models are widely used in zero-shot environmental sound classification as they alleviate the need for annotated data. However, we show that their performance severely drops in the presence of background sound sources. Our analysis reveals that this degradation is primarily driven by SNR levels of background soundscapes, and independent of background type. To address this, we propose a novel method that quantifies and integrates the contribution of background sources into the classification process, improving performance without requiring model retraining. Our domain adaptation technique enhances accuracy across various backgrounds and SNR conditions. Moreover, we analyze the modality gap between audio and text embeddings, showing that narrowing this gap improves classification performance. The method generalizes effectively across state-of-the-art prototypical approaches, showcasing its scalability and robustness for diverse environments.
>
---
#### [new 005] Benchmarking Time-localized Explanations for Audio Classification Models
- **分类: cs.SD**

- **简介: 该论文属于音频分类模型解释任务，旨在解决缺乏时间局部化解释评估标准的问题。作者提出一种基准测试方法，利用事件时间注释作为参考，优化并比较多种后验解释方法效果，接近实现完美解释，并揭示模型中潜在的虚假关联。**

- **链接: [http://arxiv.org/pdf/2506.04391v1](http://arxiv.org/pdf/2506.04391v1)**

> **作者:** Cecilia Bolaños; Leonardo Pepino; Martin Meza; Luciana Ferrer
>
> **摘要:** Most modern approaches for audio processing are opaque, in the sense that they do not provide an explanation for their decisions. For this reason, various methods have been proposed to explain the outputs generated by these models. Good explanations can result in interesting insights about the data or the model, as well as increase trust in the system. Unfortunately, evaluating the quality of explanations is far from trivial since, for most tasks, there is no clear ground truth explanation to use as reference. In this work, we propose a benchmark for time-localized explanations for audio classification models that uses time annotations of target events as a proxy for ground truth explanations. We use this benchmark to systematically optimize and compare various approaches for model-agnostic post-hoc explanation, obtaining, in some cases, close to perfect explanations. Finally, we illustrate the utility of the explanations for uncovering spurious correlations.
>
---
#### [new 006] Improving AI-generated music with user-guided training
- **分类: cs.SD; cs.HC; cs.LG; eess.AS**

- **简介: 该论文属于AI音乐生成任务，旨在解决模型难以准确响应用户主观需求的问题。作者提出一种结合用户反馈的遗传算法，以用户评分为损失函数微调模型，通过迭代提升生成效果。实验显示用户评分逐步提高，验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2506.04852v1](http://arxiv.org/pdf/2506.04852v1)**

> **作者:** Vishwa Mohan Singh; Sai Anirudh Aryasomayajula; Ahan Chatterjee; Beste Aydemir; Rifat Mehreen Amin
>
> **备注:** Select for presentation in HHAI 2025
>
> **摘要:** AI music generation has advanced rapidly, with models like diffusion and autoregressive algorithms enabling high-fidelity outputs. These tools can alter styles, mix instruments, or isolate them. Since sound can be visualized as spectrograms, image-generation algorithms can be applied to generate novel music. However, these algorithms are typically trained on fixed datasets, which makes it challenging for them to interpret and respond to user input accurately. This is especially problematic because music is highly subjective and requires a level of personalization that image generation does not provide. In this work, we propose a human-computation approach to gradually improve the performance of these algorithms based on user interactions. The human-computation element involves aggregating and selecting user ratings to use as the loss function for fine-tuning the model. We employ a genetic algorithm that incorporates user feedback to enhance the baseline performance of a model initially trained on a fixed dataset. The effectiveness of this approach is measured by the average increase in user ratings with each iteration. In the pilot test, the first iteration showed an average rating increase of 0.2 compared to the baseline. The second iteration further improved upon this, achieving an additional increase of 0.39 over the first iteration.
>
---
#### [new 007] Better Semi-supervised Learning for Multi-domain ASR Through Incremental Retraining and Data Filtering
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多领域语音识别（ASR）任务，旨在解决目标领域标注数据不足的问题。通过结合少量目标领域数据与相关领域数据，并采用增量半监督学习方法，利用多模型共识或命名实体识别过滤伪标签，逐步优化模型性能，在Wow和Fisher语料库上取得了显著提升。**

- **链接: [http://arxiv.org/pdf/2506.04981v1](http://arxiv.org/pdf/2506.04981v1)**

> **作者:** Andres Carofilis; Pradeep Rangappa; Srikanth Madikeri; Shashi Kumar; Sergio Burdisso; Jeena Prakash; Esau Villatoro-Tello; Petr Motlicek; Bidisha Sharma; Kadri Hacioglu; Shankar Venkatesan; Saurabh Vyas; Andreas Stolcke
>
> **备注:** Accepted at Interspeech 2025, Netherlands
>
> **摘要:** Fine-tuning pretrained ASR models for specific domains is challenging when labeled data is scarce. But unlabeled audio and labeled data from related domains are often available. We propose an incremental semi-supervised learning pipeline that first integrates a small in-domain labeled set and an auxiliary dataset from a closely related domain, achieving a relative improvement of 4% over no auxiliary data. Filtering based on multi-model consensus or named entity recognition (NER) is then applied to select and iteratively refine pseudo-labels, showing slower performance saturation compared to random selection. Evaluated on the multi-domain Wow call center and Fisher English corpora, it outperforms single-step fine-tuning. Consensus-based filtering outperforms other methods, providing up to 22.3% relative improvement on Wow and 24.8% on Fisher over single-step fine-tuning with random selection. NER is the second-best filter, providing competitive performance at a lower computational cost.
>
---
#### [new 008] LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出LESS框架，属于语音基础模型的半监督学习任务。旨在解决自动生成的伪标签数据质量低的问题。通过结合大语言模型（LLM）优化伪标签，并引入数据过滤策略提升知识迁移效率，在多语言语音识别与翻译任务中取得了显著性能提升。**

- **链接: [http://arxiv.org/pdf/2506.04586v1](http://arxiv.org/pdf/2506.04586v1)**

> **作者:** Wen Ding; Fan Qian
>
> **摘要:** We introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that leverages Large Language Models (LLMs) to correct pseudo labels generated from in-the-wild data. Within the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and augmented by a data filtering strategy to optimize LLM knowledge transfer efficiency. Experiments on both Mandarin ASR and Spanish-to-English AST tasks show that LESS achieves a notable absolute WER reduction of 3.77% on the Wenet Speech test set, as well as BLEU scores of 34.0 and 64.7 on Callhome and Fisher test sets respectively. These results validate the adaptability of LESS across different languages, tasks, and domains. Ablation studies conducted with various LLMs and prompt configurations provide novel insights into leveraging LLM-derived knowledge for speech processing applications.
>
---
#### [new 009] MARS: Radio Map Super-resolution and Reconstruction Method under Sparse Channel Measurements
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于无线信号地图重建任务，旨在解决稀疏信道测量下准确重构无线电图的难题。传统方法缺乏环境感知或依赖详细场景数据，泛化能力弱。为此，作者提出MARS方法，融合CNN与Transformer结构，通过多尺度特征融合和残差连接提升重建精度，在多种场景下均表现出优越性能。**

- **链接: [http://arxiv.org/pdf/2506.04682v1](http://arxiv.org/pdf/2506.04682v1)**

> **作者:** Chuyun Deng; Na Liu; Wei Xie; Lianming Xu; Li Wang
>
> **摘要:** Radio maps reflect the spatial distribution of signal strength and are essential for applications like smart cities, IoT, and wireless network planning. However, reconstructing accurate radio maps from sparse measurements remains challenging. Traditional interpolation and inpainting methods lack environmental awareness, while many deep learning approaches depend on detailed scene data, limiting generalization. To address this, we propose MARS, a Multi-scale Aware Radiomap Super-resolution method that combines CNNs and Transformers with multi-scale feature fusion and residual connections. MARS focuses on both global and local feature extraction, enhancing feature representation across different receptive fields and improving reconstruction accuracy. Experiments across different scenes and antenna locations show that MARS outperforms baseline models in both MSE and SSIM, while maintaining low computational cost, demonstrating strong practical potential.
>
---
#### [new 010] Can we reconstruct a dysarthric voice with the large speech model Parler TTS?
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音重建任务，旨在解决言语障碍者的个性化语音合成问题。研究者尝试使用大语音模型Parler TTS重建失语症患者患病前的语音，通过构建标注数据集进行模型微调，发现模型能生成语音但难以控制清晰度和保持说话人身份一致性，并提出未来改进方向。**

- **链接: [http://arxiv.org/pdf/2506.04397v1](http://arxiv.org/pdf/2506.04397v1)**

> **作者:** Ariadna Sanchez; Simon King
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech disorders can make communication hard or even impossible for those who develop them. Personalised Text-to-Speech is an attractive option as a communication aid. We attempt voice reconstruction using a large speech model, with which we generate an approximation of a dysarthric speaker's voice prior to the onset of their condition. In particular, we investigate whether a state-of-the-art large speech model, Parler TTS, can generate intelligible speech while maintaining speaker identity. We curate a dataset and annotate it with relevant speaker and intelligibility information, and use this to fine-tune the model. Our results show that the model can indeed learn to generate from the distribution of this challenging data, but struggles to control intelligibility and to maintain consistent speaker identity. We propose future directions to improve controllability of this class of model, for the voice reconstruction task.
>
---
#### [new 011] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出了MMSU，一个大规模多任务的口语理解和推理基准。它属于自然语言处理与语音识别任务，旨在解决当前语音大模型在口语细粒度感知和复杂推理上的不足。论文构建了包含5,000个音频问答对、覆盖47项任务的评测集，并基于语言学理论设计丰富样例，评估14种先进模型，指明改进方向。**

- **链接: [http://arxiv.org/pdf/2506.04779v1](http://arxiv.org/pdf/2506.04779v1)**

> **作者:** Dingdong Wang; Jincenzi Wu; Junan Li; Dongchao Yang; Xueyuan Chen; Tianhua Zhang; Helen Meng
>
> **备注:** MMSU benchmark is available at https://huggingface.co/datasets/ddwang2000/MMSU. Evaluation Code is available at https://github.com/dingdongwang/MMSU_Bench
>
> **摘要:** Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at https://huggingface.co/datasets/ddwang2000/MMSU. Evaluation Code is available at https://github.com/dingdongwang/MMSU_Bench.
>
---
#### [new 012] Through-the-Wall Radar Human Activity Recognition WITHOUT Using Neural Networks
- **分类: cs.CV; eess.SP; 68T45; I.5.4**

- **简介: 该论文属于通过墙体雷达识别人类活动（TWR HAR）任务，旨在不使用神经网络的情况下实现智能识别。论文提出了一种基于信号处理和拓扑分析的方法，包括生成雷达图像、分割微多普勒特征，并利用Mapper算法计算点云拓扑相似性以完成识别，挑战传统依赖神经网络的模式。**

- **链接: [http://arxiv.org/pdf/2506.05169v1](http://arxiv.org/pdf/2506.05169v1)**

> **作者:** Weicheng Gao
>
> **备注:** 15 pages, 8 figures, 8 tables
>
> **摘要:** After a few years of research in the field of through-the-wall radar (TWR) human activity recognition (HAR), I found that we seem to be stuck in the mindset of training on radar image data through neural network models. The earliest related works in this field based on template matching did not require a training process, and I believe they have never died. Because these methods possess a strong physical interpretability and are closer to the basis of theoretical signal processing research. In this paper, I would like to try to return to the original path by attempting to eschew neural networks to achieve the TWR HAR task and challenge to achieve intelligent recognition as neural network models. In detail, the range-time map and Doppler-time map of TWR are first generated. Then, the initial regions of the human target foreground and noise background on the maps are determined using corner detection method, and the micro-Doppler signature is segmented using the multiphase active contour model. The micro-Doppler segmentation feature is discretized into a two-dimensional point cloud. Finally, the topological similarity between the resulting point cloud and the point clouds of the template data is calculated using Mapper algorithm to obtain the recognition results. The effectiveness of the proposed method is demonstrated by numerical simulated and measured experiments. The open-source code of this work is released at: https://github.com/JoeyBGOfficial/Through-the-Wall-Radar-Human-Activity-Recognition-Without-Using-Neural-Networks.
>
---
#### [new 013] The NTNU System at the S&I Challenge 2025 SLA Open Track
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文参与的是口语测评任务，旨在解决现有模型在语音和语义评估上的局限性。作者结合wav2vec 2.0与Phi-4多模态大模型，采用分数融合策略，提升了测评准确性，在S&I挑战赛中取得了第二名的成绩。**

- **链接: [http://arxiv.org/pdf/2506.05121v1](http://arxiv.org/pdf/2506.05121v1)**

> **作者:** Hong-Yun Lin; Tien-Hong Lo; Yu-Hsuan Fang; Jhen-Ke Lin; Chung-Chun Wang; Hao-Chien Lu; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively.
>
---
#### [new 014] AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究大型音频-语言模型（LALMs）对听觉属性的内部感知机制，旨在理解其行为并提升性能。通过词汇投影分析，追踪属性信息在模型层和位置间的演变，发现识别失败时信息随深度减少，早期层解析属性更准确。提出增强方法，揭示LALMs依赖查询输入而非隐藏状态进行属性预测。**

- **链接: [http://arxiv.org/pdf/2506.05140v1](http://arxiv.org/pdf/2506.05140v1)**

> **作者:** Chih-Kai Yang; Neo Ho; Yi-Jyun Lee; Hung-yi Lee
>
> **备注:** 8 pages, 5 figures, 3 tables
>
> **摘要:** Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements.
>
---
#### [new 015] Effects of Speaker Count, Duration, and Accent Diversity on Zero-Shot Accent Robustness in Low-Resource ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升低资源条件下自动语音识别（ASR）系统对未见口音的鲁棒性。研究分析了说话人数、语音时长和口音多样性对系统性能的影响，发现增加说话人数量比增加每人录音时长更有效，且口音多样性影响较小。结论建议优先扩充说话人数量以优化ASR训练。**

- **链接: [http://arxiv.org/pdf/2506.04364v1](http://arxiv.org/pdf/2506.04364v1)**

> **作者:** Zheng-Xin Yong; Vineel Pratap; Michael Auli; Jean Maillard
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** To build an automatic speech recognition (ASR) system that can serve everyone in the world, the ASR needs to be robust to a wide range of accents including unseen accents. We systematically study how three different variables in training data -- the number of speakers, the audio duration per each individual speaker, and the diversity of accents -- affect ASR robustness towards unseen accents in a low-resource training regime. We observe that for a fixed number of ASR training hours, it is more beneficial to increase the number of speakers (which means each speaker contributes less) than the number of hours contributed per speaker. We also observe that more speakers enables ASR performance gains from scaling number of hours. Surprisingly, we observe minimal benefits to prioritizing speakers with different accents when the number of speakers is controlled. Our work suggests that practitioners should prioritize increasing the speaker count in ASR training data composition for new languages.
>
---
## 更新

#### [replaced 001] Self-Tuning Spectral Clustering for Speaker Diarization
- **分类: eess.SP; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.00023v2](http://arxiv.org/pdf/2410.00023v2)**

> **作者:** Nikhil Raghav; Avisek Gupta; Md Sahidullah; Swagatam Das
>
> **备注:** This is the camera-ready version accepted for publication in the ICASSP 2025 proceedings
>
> **摘要:** Spectral clustering has proven effective in grouping speech representations for speaker diarization tasks, although post-processing the affinity matrix remains difficult due to the need for careful tuning before constructing the Laplacian. In this study, we present a novel pruning algorithm to create a sparse affinity matrix called spectral clustering on p-neighborhood retained affinity matrix (SC-pNA). Our method improves on node-specific fixed neighbor selection by allowing a variable number of neighbors, eliminating the need for external tuning data as the pruning parameters are derived directly from the affinity matrix. SC-pNA does so by identifying two clusters in every row of the initial affinity matrix, and retains only the top p % similarity scores from the cluster containing larger similarities. Spectral clustering is performed subsequently, with the number of clusters determined as the maximum eigengap. Experimental results on the challenging DIHARD-III dataset highlight the superiority of SC-pNA, which is also computationally more efficient than existing auto-tuning approaches. Our implementations are available at https://github.com/nikhilraghav29/SC-pNA.
>
---
#### [replaced 002] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18614v2](http://arxiv.org/pdf/2505.18614v2)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** 28 pages, 8 figures, our codes and datasets are available at https://github.com/k1064190/MAVL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 003] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
- **分类: cs.MM; cs.CV; cs.GR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.16331v3](http://arxiv.org/pdf/2411.16331v3)**

> **作者:** Xiaozhong Ji; Xiaobin Hu; Zhihong Xu; Junwei Zhu; Chuming Lin; Qingdong He; Jiangning Zhang; Donghao Luo; Yi Chen; Qin Lin; Qinglin Lu; Chengjie Wang
>
> **备注:** refer to our main-page \url{https://jixiaozhong.github.io/Sonic/}
>
> **摘要:** The study of talking face generation mainly explores the intricacies of synchronizing facial movements and crafting visually appealing, temporally-coherent animations. However, due to the limited exploration of global audio perception, current approaches predominantly employ auxiliary visual and spatial knowledge to stabilize the movements, which often results in the deterioration of the naturalness and temporal inconsistencies.Considering the essence of audio-driven animation, the audio signal serves as the ideal and unique priors to adjust facial expressions and lip movements, without resorting to interference of any visual signals. Based on this motivation, we propose a novel paradigm, dubbed as Sonic, to {s}hift f{o}cus on the exploration of global audio per{c}ept{i}o{n}.To effectively leverage global audio knowledge, we disentangle it into intra- and inter-clip audio perception and collaborate with both aspects to enhance overall perception.For the intra-clip audio perception, 1). \textbf{Context-enhanced audio learning}, in which long-range intra-clip temporal audio knowledge is extracted to provide facial expression and lip motion priors implicitly expressed as the tone and speed of speech. 2). \textbf{Motion-decoupled controller}, in which the motion of the head and expression movement are disentangled and independently controlled by intra-audio clips. Most importantly, for inter-clip audio perception, as a bridge to connect the intra-clips to achieve the global perception, \textbf{Time-aware position shift fusion}, in which the global inter-clip audio information is considered and fused for long-audio inference via through consecutively time-aware shifted windows. Extensive experiments demonstrate that the novel audio-driven paradigm outperform existing SOTA methodologies in terms of video quality, temporally consistency, lip synchronization precision, and motion diversity.
>
---
#### [replaced 004] DGMO: Training-Free Audio Source Separation through Diffusion-Guided Mask Optimization
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02858v2](http://arxiv.org/pdf/2506.02858v2)**

> **作者:** Geonyoung Lee; Geonhee Han; Paul Hongsuck Seo
>
> **备注:** Interspeech 2025
>
> **摘要:** Language-queried Audio Source Separation (LASS) enables open-vocabulary sound separation via natural language queries. While existing methods rely on task-specific training, we explore whether pretrained diffusion models, originally designed for audio generation, can inherently perform separation without further training. In this study, we introduce a training-free framework leveraging generative priors for zero-shot LASS. Analyzing naive adaptations, we identify key limitations arising from modality-specific challenges. To address these issues, we propose Diffusion-Guided Mask Optimization (DGMO), a test-time optimization framework that refines spectrogram masks for precise, input-aligned separation. Our approach effectively repurposes pretrained diffusion models for source separation, achieving competitive performance without task-specific supervision. This work expands the application of diffusion models beyond generation, establishing a new paradigm for zero-shot audio separation. The code is available at: https://wltschmrz.github.io/DGMO/
>
---
#### [replaced 005] Hearing Anywhere in Any Environment
- **分类: cs.CV; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.10746v2](http://arxiv.org/pdf/2504.10746v2)**

> **作者:** Xiulong Liu; Anurag Kumar; Paul Calamia; Sebastia V. Amengual; Calvin Murdock; Ishwarya Ananthabhotla; Philip Robinson; Eli Shlizerman; Vamsi Krishna Ithapu; Ruohan Gao
>
> **备注:** CVPR 2025; Project Page: https://dragonliu1995.github.io/hearinganywhereinanyenvironment/
>
> **摘要:** In mixed reality applications, a realistic acoustic experience in spatial environments is as crucial as the visual experience for achieving true immersion. Despite recent advances in neural approaches for Room Impulse Response (RIR) estimation, most existing methods are limited to the single environment on which they are trained, lacking the ability to generalize to new rooms with different geometries and surface materials. We aim to develop a unified model capable of reconstructing the spatial acoustic experience of any environment with minimum additional measurements. To this end, we present xRIR, a framework for cross-room RIR prediction. The core of our generalizable approach lies in combining a geometric feature extractor, which captures spatial context from panorama depth images, with a RIR encoder that extracts detailed acoustic features from only a few reference RIR samples. To evaluate our method, we introduce ACOUSTICROOMS, a new dataset featuring high-fidelity simulation of over 300,000 RIRs from 260 rooms. Experiments show that our method strongly outperforms a series of baselines. Furthermore, we successfully perform sim-to-real transfer by evaluating our model on four real-world environments, demonstrating the generalizability of our approach and the realism of our dataset.
>
---
#### [replaced 006] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00975v2](http://arxiv.org/pdf/2506.00975v2)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
#### [replaced 007] Can Masked Autoencoders Also Listen to Birds?
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.12880v2](http://arxiv.org/pdf/2504.12880v2)**

> **作者:** Lukas Rauch; René Heinrich; Ilyass Moummad; Alexis Joly; Bernhard Sick; Christoph Scholz
>
> **备注:** under review @TMLR
>
> **摘要:** Masked Autoencoders (MAEs) have shown competitive results in audio classification by learning rich semantic representations through an efficient self-supervised reconstruction task. However, general-purpose models fail to generalize well when applied directly to fine-grained audio domains. Specifically, bird-sound classification requires distinguishing subtle inter-species differences and managing high intra-species acoustic variability, thereby revealing the performance limitations of general-domain Audio-MAE models. This work demonstrates that bridging this domain gap requires more than domain-specific pretraining data; adapting the entire training pipeline is crucial. We systematically revisit and adapt the pretraining recipe, fine-tuning methods, and frozen feature utilization to bird sounds using BirdSet, a large-scale bioacoustic dataset comparable to AudioSet. Our resulting Bird-MAE achieves new state-of-the-art results in BirdSet's multi-label classification benchmark. Additionally, we introduce the parameter-efficient prototypical probing, enhancing the utility of frozen MAE representations and closely approaching fine-tuning performance in low-resource settings. Bird-MAE's prototypical probes outperform linear probing by up to 37%$_\text{p}$ in MAP and narrow the gap to fine-tuning to approximately 3.3%$_\text{p}$ on average across BirdSet downstream tasks. Bird-MAE also demonstrates robust few-shot capabilities with prototypical probing in our newly established few-shot benchmark on BirdSet, highlighting the potential of tailored self-supervised learning pipelines for fine-grained audio domains.
>
---
#### [replaced 008] Inter-Speaker Relative Cues for Text-Guided Target Speech Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.01483v2](http://arxiv.org/pdf/2506.01483v2)**

> **作者:** Wang Dai; Archontis Politis; Tuomas Virtanen
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We propose a novel approach that utilizes inter-speaker relative cues for distinguishing target speakers and extracting their voices from mixtures. Continuous cues (e.g., temporal order, age, pitch level) are grouped by relative differences, while discrete cues (e.g., language, gender, emotion) retain their categories. Relative cues offers greater flexibility than fixed speech attribute classification, facilitating much easier expansion of text-guided target speech extraction datasets. Our experiments show that combining all relative cues yields better performance than random subsets, with gender and temporal order being the most robust across languages and reverberant conditions. Additional cues like pitch level, loudness, distance, speaking duration, language, and pitch range also demonstrate notable benefit in complex scenarios. Fine-tuning pre-trained WavLM Base+ CNN encoders improves overall performance over the baseline of using only a Conv1d encoder.
>
---
#### [replaced 009] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11315v2](http://arxiv.org/pdf/2503.11315v2)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Se Jin Park; Yong Man Ro
>
> **备注:** Accepted at Findings of ACL 2025. The code and models are available https://github.com/JeongHun0716/MMS-LLaMA
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) achieves robust speech recognition in noisy environments by combining auditory and visual information. However, recent Large Language Model (LLM) based AVSR systems incur high computational costs due to the high temporal resolution of audio-visual speech processed by LLMs. In this work, we introduce an efficient multimodal speech LLM framework that minimizes token length while preserving essential linguistic content. Our approach employs an early AV-fusion module for streamlined feature integration, an audio-visual speech Q-Former that dynamically allocates tokens based on input duration, and a refined query allocation strategy with a speech rate predictor to adjust token allocation according to speaking speed of each audio sample. Extensive experiments on the LRS3 dataset show that our method achieves state-of-the-art performance with a WER of 0.72% while using only 3.5 tokens per second. Moreover, our approach not only reduces token usage by 86% compared to the previous multimodal speech LLM framework, but also improves computational efficiency by reducing FLOPs by 35.7%.
>
---
#### [replaced 010] FGAS: Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22266v2](http://arxiv.org/pdf/2505.22266v2)**

> **作者:** Jialin Yan; Yu Cheng; Zhaoxia Yin; Xinpeng Zhang; Shilin Wang; Tanfeng Sun; Xinghao Jiang
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) has made high-fidelity generated audio widely available across the Internet, providing diverse cover signals for covert communication. Driven by advances in deep learning, current audio steganography schemes are mainly based on encoding-decoding network architectures. While these methods greatly improve the security of audio steganography, they typically require complex training and large pre-trained models. To address the aforementioned issues, this paper pioneers a Fixed Decoder Network-Based Audio Steganography with Adversarial Perturbation Generation (FGAS). Adversarial perturbations carrying secret message are embedded into the cover audio to generate stego audio. The receiver only needs to share the structure and weights of the fixed decoder network to accurately extract the secret message from the stego audio, this eliminates the reliance on large pre-trained models. In FGAS, we propose an audio Adversarial Perturbation Generation (APG) strategy and design a lightweight fixed decoder. The fixed decoder guarantees reliable extraction of the hidden message, while the adversarial perturbations are optimized to keep the stego audio perceptually and statistically close to the cover audio, thereby improving resistance to steganalysis. The experimental results show that FGAS significantly improves the quality of stego audio, achieving an average PSNR gain of over 10 dB compared to SOTA methods. Moreover, FGAS exhibits superior anti-steganalysis performance under different relative payloads; under high-capacity embedding, it achieves a classification error rate about 2% higher, indicating stronger anti-steganalysis performance compared to current SOTA methods.
>
---
#### [replaced 011] STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution
- **分类: cs.SD; cs.AI; cs.CR; eess.AS; 68T45, 68T10, 94A08; I.2.7; I.5.4; K.4.1**

- **链接: [http://arxiv.org/pdf/2505.19644v2](http://arxiv.org/pdf/2505.19644v2)**

> **作者:** Anton Firc; Manasi Chibber; Jagabandhu Mishra; Vishwanath Pratap Singh; Tomi Kinnunen; Kamil Malinka
>
> **备注:** Accepted to Interspeech 2025 conference
>
> **摘要:** A key research area in deepfake speech detection is source tracing - determining the origin of synthesised utterances. The approaches may involve identifying the acoustic model (AM), vocoder model (VM), or other generation-specific parameters. However, progress is limited by the lack of a dedicated, systematically curated dataset. To address this, we introduce STOPA, a systematically varied and metadata-rich dataset for deepfake speech source tracing, covering 8 AMs, 6 VMs, and diverse parameter settings across 700k samples from 13 distinct synthesisers. Unlike existing datasets, which often feature limited variation or sparse metadata, STOPA provides a systematically controlled framework covering a broader range of generative factors, such as the choice of the vocoder model, acoustic model, or pretrained weights, ensuring higher attribution reliability. This control improves attribution accuracy, aiding forensic analysis, deepfake detection, and generative model transparency.
>
---
